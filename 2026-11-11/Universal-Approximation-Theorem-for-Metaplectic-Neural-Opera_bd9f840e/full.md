# Universal Approximation Theorem for Metaplectic Neural Operators

Ahmed Abdeljawad RICAM, Austrian Academy of Sciences Altenberger Straße 69, A-4040 Linz, Austria Ahmed.Abdeljawad@oeaw.ac.at

## Abstract

Malek Yacine Rezoug RICAM, Austrian Academy of Sciences Altenberger Straße 69, A-4040 Linz, Austria MalekYacine.Rezoug@oeaw.ac.at

We prove universal approximation for metaplectic neural operators on nonempty bounded Lipschitz domains. The nonlocal branches are whole-space metaplectic multipliers with zero extension and restriction, associated with any fixed free symplectic matrix. Assuming a smooth, globally Lipschitz, nonpolynomial activation, we obtain compact-uniform approximation of continuous operators from $C ^ { s }$ to $C ^ { r }$ and from $\smash { \dot { H } ^ { s } }$ to $H ^ { r }$ in the respective output norms, for arbitrary nonnegative integers $s ,$ r and finite input and output channel dimensions. In the Sobolev setting, compact input sets must additionally be uniformly bounded in $L ^ { \infty }$ . One hidden operator layer with one nonlocal branch suffices. A single scalar symbol atom and finitely many positional features are prescribed by the transform and domain independently of the target operator, compact set and accuracy. The proof combines averaging neural operator theory with an exact chirp-conjugated averaging identity and derivative estimates. The result includes Fourier and fixed nondegenerate fractional Fourier multiplier constructions.

## 1 Introduction

Neural operators approximate nonlinear maps between function spaces. A typical target is the solution operator of a partial differential equation, which maps a coefficient field, a forcing term or initial data to the corresponding solution. The dependence on the input need not be local: changing the data in one region may affect the solution elsewhere. The mathematical analysis of neural operators must account both for the finite-dimensional nonlinear approximation performed within the network and for the mechanism by which information is transferred between different points of the domain. This paper studies the latter mechanism for neural operators whose nonlocal branches are multipliers with respect to a prescribed metaplectic transform. We ask whether fixing this transform preserves universal approximation, including approximation in norms that control derivatives of the output. Approximation in such norms is relevant when the predicted field is used to compute gradients or other derivative-dependent quantities.

A common class of neural operators combines pointwise lifting and projection maps with nonlinear integral layers [29, 21]. For a bounded domain $\mathbf { \mathop { \Omega } } ^ { \star } \boldsymbol { \Omega } \subset \mathbb { R } ^ { d }$ , the architecture takes the form

$$
\begin{array} { r } { \mathcal { N } = \mathscr { Q } \circ \mathcal { L } _ { L } \circ \cdot \cdot \circ \mathcal { L } _ { 1 } \circ \mathcal { R } , \qquad \mathcal { L } _ { \ell } ( v ) ( x ) = \sigma \big ( W _ { \ell } v ( x ) + b _ { \ell } ( x ) + ( K _ { \ell } v ) ( x ) \big ) , } \end{array}
$$

where R and $\mathcal { Q }$ act pointwise, possibly using spatial coordinates, σ acts componentwise, and $\displaystyle \kappa _ { \ell }$ is a linear nonlocal operator, with dimensions compatible with the channels of the layer. The lifting maps each input value, together with any positional information, to the hidden channels, while the projection converts the final hidden state into the output. The matrix $W _ { \ell }$ and the bias $b _ { \ell }$ act locally; the nonlocal branch allows the value at x to depend on input values elsewhere in the domain. Transformbased architectures specify this branch through a chosen representation. For a multiplier branch, the input is transformed, multiplied by a symbol, and then mapped back to the spatial variables. The Fourier neural operator (FNO) [30], for example, multiplies finitely many Fourier coefficients by trainable matrices, giving a periodic convolution kernel.

Metaplectic analysis provides a common setting for Fourier and fractional Fourier transforms. Metaplectic operators implement linear symplectic changes of position and frequency and, up to constant phase factors, include Fourier transforms, fractional Fourier transforms, dilations and multiplication by quadratic phases [15, 16, 12]. Prescribing such a transform fixes the coordinates in which the multiplier acts. On the whole space, a metaplectic multiplier need not be translation invariant, even though a Fourier multiplier is. The free symplectic class considered here includes linear couplings of position and frequency beyond the scalar rotations associated with fractional Fourier transforms. This broader class permits a common approximation question for several choices of transform: which nonlocal operations are sufficient when the transform itself is fixed?

The classical properties of metaplectic operators describe the linear branch, but universality also depends on the admissible symbols, positional inputs and nonlinear maps. We therefore work with prescribed finite families of symbol atoms and positional features, and with finite neural networks for the local maps. The objective is to establish universal approximation within this architecture for every fixed free transform. The construction rests on spatial averaging, which Lanthaler, Li and Stuart [23] identified as a sufficient nonlocal operation for universality. Averaging gathers features of the input over the domain and makes the resulting finite vector available to the pointwise maps. Here the averaging operation must be realized by the prescribed metaplectic branch and used in a way compatible with the componentwise activation.

Let $\Omega \subset \mathbb { R } ^ { d }$ be a nonempty bounded Lipschitz domain, and fix

$$
S = { \binom { A } { C } } \ { \begin{array} { l } { B } \\ { D } \end{array} } ) \in \operatorname { S p } ( 2 d , \mathbb { R } )
$$

that is $f r e e ,$ meaning that B is invertible. Let $\widehat { S }$ be the operator in $( 4 )$ , which agrees with a metaplectic operator associated with S up to a constant phase. Using extension by zero $\mathsf { E } _ { \Omega }$ and restriction $\mathsf { R } _ { \Omega }$ the nonlocal branch is

$$
\mathcal { K } _ { \widehat { S } , P } ^ { \Omega } = \mathsf { R } _ { \Omega } \widehat { S } ^ { - 1 } M _ { P } \widehat { S } \mathsf { E } _ { \Omega } , \qquad ( M _ { P } f ) ( \eta ) = P ( \eta ) f ( \eta ) ,
$$

where $P$ is an essentially bounded matrix-valued symbol and the transforms act componentwise. This defines a bounded operator on $L ^ { 2 }$ , and the constant phase cancels in the conjugation. Throughout, the whole-space Fourier transform $\mathcal { F }$ has the convention in (1). The free metaplectic factorization expresses the branch as a Fourier multiplier conjugated by multiplication with the quadratic chirp

$$
\gamma _ { S } ( x ) = e ^ { \pi i x \cdot Q _ { S } x } , \qquad Q _ { S } = B ^ { - 1 } A .
$$

In particular, symbols in the Schwartz space give convolution kernels multiplied by fixed quadratic phases in the input and output variables (Proposition 2.6). The zero-extension and restriction convention places these branches on the same bounded domain without requiring periodicity of the input.

The admissible symbols and biases belong to finite-parameter families

$$
P ( \eta ) = \sum _ { j = 1 } ^ { M } \Theta _ { j } \rho _ { j } ( \eta ) , \qquad b ( x ) = c + G p ( x ) ,
$$

in which the scalar atoms $\rho _ { j }$ and positional vector $p$ are prescribed, while the matrix and vector coefficients are trainable. For a fixed free $S ,$ , our construction uses

$$
p _ { S } ( x ) = \bigl ( x ,  { \mathrm { R e } } \gamma _ { S } ( x ) ,  { \mathrm { I m } } \gamma _ { S } ( x ) \bigr ) .
$$

Each complex channel is represented by two real channels. A metaplectic neural operator $( \mathrm { M p N O } )$ inserts the realified multiplier branches into the layers above with componentwise real activation. Its lifting and projection are finite neural networks with positional inputs; the lifting has an activated output, whereas the projection has an affine output (Definition 2.15).

Our main result, Theorem 3.1, establishes universality for every fixed free $S$ when the symbol class contains the atom constructed in Proposition 2.17 and the positional features include $p _ { S }$ . Assume that σ is smooth, globally Lipschitz and nonpolynomial, as is the case for tanh. For $s , r \in  { \mathbb { N } } _ { 0 }$ and $d _ { a } , d _ { u } \in \mathbb { N }$ , consider either of the settings

$$
\begin{array} { r } { X = C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { a } } ) , \quad Y = C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { u } } ) , } \\ { X = H ^ { s } ( \Omega ; \mathbb { R } ^ { d _ { a } } ) , \quad Y = H ^ { r } ( \Omega ; \mathbb { R } ^ { d _ { u } } ) . } \end{array}
$$

For every continuous $\mathcal { G } : X  Y$ , every compact $K \subset X$ and every $\varepsilon > 0 .$ , there is a finite MpNO $\mathcal { N }$ such that

$$
\operatorname* { s u p } _ { a \in K } \| { \mathcal { G } } ( a ) - { \mathcal { N } } ( a ) \| _ { Y } < \varepsilon ,
$$

where in the Sobolev setting we additionally assume $\begin{array} { r } { \operatorname* { s u p } _ { a \in K } \| a \| _ { L ^ { \infty } } < \infty } \end{array}$ . For $s > d / 2$ , Sobolev embedding [6] makes this bound automatic for compact subsets of $H ^ { s } ;$ ; at lower input regularity it remains a hypothesis. The output norm controls spatial derivatives through order $r ,$ uniformly in the $C ^ { r }$ setting and in $L ^ { 2 }$ in the $H ^ { \star }$ setting.

In both settings the approximants require only one hidden operator layer with one nonlocal branch. For each fixed pair (S, Ω), the transform, a single scalar symbol atom $\rho _ { \widehat { S } , \Omega }$ and the positional features are prescribed independently of $\mathcal { G } _ { : }$ , K and $\varepsilon .$ The branch symbol is a scalar multiple of this atom times the identity matrix. The channel count, that scalar coefficient, the bias coefficients, and the sizes and weights of the lifting and projection networks may depend on the approximation data. Thus the spatial profile of the nonlocal interaction is fixed while the number of features it communicates may increase. The pointwise networks may have their own depth, and the fixed atom is a function of the continuous transform variable rather than a finite selection of Fourier modes.

The proof uses the finite-dimensional reduction of averaging neural operator theory [23]: the target is approximated by a pointwise map of finitely many spatial averages and the output position. For every fixed (S, Ω), we construct an atom such that, for $q \in \mathbb { N }$ and $v \in L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q } )$

$$
A _ { S } v : = K _ { \hat { S } , \rho _ { \hat { S } , \Omega } } ^ { \Omega } I _ { q } v = \overline { { \gamma _ { S } } } \mathrm { A v g } _ { \Omega } ( \gamma _ { S } v ) , \qquad \mathrm { A v g } _ { \Omega } ( v ) = \frac { 1 } { | \Omega | } \int _ { \Omega } v ( y ) d y .
$$

The identity holds almost everywhere. It makes the required averages available through one metaplectic branch. The lifting, bias and projection are chosen so that this representation can be recovered after the componentwise activation. Since the spatial dependence after averaging lies in the fixed smooth chirp, an $L ^ { 2 }$ lifting error yields control of the derivatives needed for the output norm. Only $L ^ { 2 }$ zero extension is used; no higher regularity of the zero-extended input is required.

The Fourier and fractional Fourier cases illustrate the result. For $J = \left( \begin{array} { c c } { 0 } & { I _ { d } } \\ { - I _ { d } } & { 0 } \end{array} \right)$ , one has $\gamma _ { J } = 1$ and $\mathcal { A } _ { J } = \mathrm { A v g } _ { \Omega }$ . For each fixed fractional Fourier angle θ with sin $\theta \neq 0 ,$ the associated symplectic rotation is free and has chirp $e ^ { \pi i \cot \theta | x | ^ { 2 } }$ [32, 16]. The theorem therefore gives Fourier and fixed-angle fractional Fourier universality under the same bounded-domain convention and in the stated derivative norms. These conclusions concern each fixed transform separately; they do not assert uniform approximation complexity as the transform approaches a degenerate case.

## 1.1 Related work

The approximation of operators by neural networks has a long history. Chen and Chen [10] established an early universality theorem, while DeepONet [31] and PCA-based methods [8] combine finite-dimensional input representations with finite-dimensional reconstruction of the output. Their approximation and complexity properties are studied in [24, 22]. Graph-kernel and multipole graph architectures describe nonlocal interactions directly or through hierarchical constructions [29, 28], and a general framework for neural operators is developed in [21].

For FNOs, Kovachki, Lanthaler and Mishra [20] proved compact-uniform universality in Sobolev output norms and quantitative error bounds for selected PDE solution operators. Their nonperiodic formulation uses a bounded Sobolev extension into a periodic domain followed by restriction. Our construction instead uses whole-space multipliers with zero extension and restriction, together with a single prescribed symbol atom. This distinction concerns the realization of the nonlocal branch; it does not identify the complete nonlinear architectures or give a quantitative comparison between them. The parametric complexity of operator learning is studied by Lanthaler and Stuart [25]. Determining whether a particular metaplectic transform improves approximation rates or reduces network size requires further information about the target operator class.

The closest approximation result is the ANO theorem of Lanthaler, Li and Stuart [23]. On bounded Lipschitz domains, it establishes universality in $C ^ { s }$ spaces and in Sobolev spaces, with a uniform $L ^ { \infty }$ input bound in the Sobolev setting. Their Sobolev result covers the more general $W ^ { s , p }$ scales. The finite-dimensional reduction used here is inherited from their proof and is recorded in Proposition 2.21. Our contribution is its realization within the prescribed metaplectic symbol and bias classes, including the compatibility with activation and the estimates needed to retain the output norms. In the Fourier case, the construction realizes the ANO average itself as a whole-space multiplier.

Other transform-based architectures use different representations of the nonlocal interaction. Multiwavelet operator learning [17] and wavelet neural operators [36] use localized multiscale representations, while spectral neural operators [14] learn maps between finite vectors of Fourier or Chebyshev coefficients. Fractional Fourier multiplier constructions are more directly related to the present work. CoNO [35] uses such branches within a complex-valued architecture and learns the fractional order in its numerical implementation. Its stated universality theorem fixes that order and gives compact-uniform $L ^ { 2 }$ approximation for continuous maps between positive-order Sobolev spaces on a fractional torus [35, Theorem 4.3 in the preprint]. Fractional Fourier series on these domains are studied in [37]. Our fractional Fourier case uses whole-space transforms on a bounded Lipschitz domain through zero extension and restriction, and controls $C ^ { r }$ or $H ^ { r }$ errors under the stated input hypotheses. The common transform therefore connects the linear branches, while the domain conventions and nonlinear architectures remain distinct.

Fourier and harmonic-analysis methods provide several approaches to neural approximation. Barron [7] derives approximation bounds for shallow networks under a weighted integrability condition on the Fourier transform, while Candès [9] constructs stable representations by ridge functions using harmonic-analysis methods. Related approaches use Fourier symbols [4], time-frequency dictionaries [2], and metaplectic dictionaries [1]. These works concern the approximation or representation of functions and linear operators. Here the target is a general continuous nonlinear operator, approximated using a prescribed metaplectic interaction.

Approximation in derivative norms also connects the present result with finite-dimensional neural approximation. Simultaneous approximation of functions and their derivatives is studied in [18, 27]. Quantitative Sobolev-norm estimates are obtained under Fourier-based regularity assumptions and suitable conditions on the activation in [34], and for tanh networks in [13]. Further results address Sobolev time-space approximation [5] and anisotropic Fourier–Lebesgue and Bochner–Sobolev settings [3]. In the present theorem, the choice of output norm allows the approximation to control derivative-dependent quantities whenever they define bounded linear maps on the output space.

## 1.2 Organization of the paper

Section 2 introduces the function spaces and metaplectic neural operators, establishes the exact averaging identity and its regularity estimates, and develops the auxiliary approximation results. Section 3 states and proves the universal approximation theorem.

## 2 Preliminaries and metaplectic neural operators

Throughout this paper, $\mathbb { N } _ { 0 } = \{ 0 , 1 , \ldots \}$ and $ { \mathbb { N } } =  { \mathbb { N } } _ { 0 } \setminus \{ 0 \}$ . Let $d , d _ { a } , d _ { u } \in \mathbb { N }$ be fixed, and $\Omega \subset \mathbb { R } ^ { d }$ is a nonempty bounded Lipschitz domain. Its Lebesgue measure is denoted by |Ω|. The vector $\mathbf { 1 } _ { m } \in \mathbb { R } ^ { m }$ has all entries equal to one, and $M ^ { \mathsf { T } }$ denotes the transpose of a matrix $M .$ We write $A \Subset B$ to mean that A is compact and ${ \overline { { A } } } \subset B$ , where the closure is taken in the ambient space. Standard references for the function spaces used below are [6, 33]; for metaplectic analysis, see [11, 15, 16].

## 2.1 Function spaces and Fourier convention

Let $\mathbb { F } \in \{ \mathbb { R } , \mathbb { C } \}$ and $q \in \mathbb { N }$ . We write $L ^ { 2 } ( \Omega ;  { \mathbb { F } } ^ { q } )$ for the set of measurable maps $v : \Omega \to \mathbb { F } ^ { q }$ with

$$
\| v \| _ { L ^ { 2 } ( \Omega ; \mathbb { F } ^ { q } ) } ^ { 2 } : = \int _ { \Omega } | v ( x ) | _ { \mathbb { F } ^ { q } } ^ { 2 } d x < \infty .
$$

For a matrix $T ,$ we denote its Euclidean operator norm by $\| T \| _ { \mathrm { o p } } : = \operatorname* { s u p } _ { | z | = 1 } | T z |$ . For a multi-index $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { d } ) \in  { \mathbb { N } } _ { 0 } ^ { d }$ , we write $| \alpha | : = \alpha _ { 1 } + \cdot \cdot \cdot + \alpha _ { d }$ and $\partial ^ { \alpha } : = \partial _ { 1 } ^ { \alpha _ { 1 } } \cdot \cdot \cdot \partial _ { d } ^ { \alpha _ { d } }$

Let $m \in \mathbb { N } _ { 0 } ^ { d } .$ , the vector-valued Sobolev space is defined by

$$
\begin{array} { r } { { H } ^ { m } ( \Omega ; \mathbb { F } ^ { q } ) : = \left\{ v \in L ^ { 2 } ( \Omega ; \mathbb { F } ^ { q } ) : \partial ^ { \alpha } v \in L ^ { 2 } ( \Omega ; \mathbb { F } ^ { q } ) \mathrm { ~ f o r ~ e v e r y ~ } \alpha \in \mathbb { N } _ { 0 } ^ { d } \mathrm { ~ w i t h ~ } | \alpha | \le m \right\} } \end{array}
$$

and is equipped with the norm

$$
\lVert v \rVert _ { H ^ { m } ( \Omega ; \mathbb { F } ^ { q } ) } ^ { 2 } : = \sum _ { | \alpha | \le m } \lVert \partial ^ { \alpha } v \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { F } ^ { q } ) } ^ { 2 } .
$$

We also use the vector-valued space $C ^ { m } ( \overline { { \Omega } } ; \mathbb { F } ^ { q } )$ , consisting of all $v \in C ^ { m } ( \Omega ; \mathbb { F } ^ { q } )$ such that, for every ${ \boldsymbol { \alpha } } \in  { \mathbb { N } } _ { 0 } ^ { d }$ with $| \alpha | \le m$ , the derivative $\partial ^ { \alpha } v$ admits a continuous extension to ${ \overline { { \Omega } } } .$ . This space is equipped with the norm

$$
\| v \| _ { C ^ { m } ( \overline { { \Omega } } ; \mathbb { F } ^ { q } ) } : = \operatorname* { m a x } _ { | \alpha | \leq m } \operatorname* { s u p } _ { x \in \overline { { \Omega } } } | \partial ^ { \alpha } v ( x ) | _ { \mathbb { F } ^ { q } } .
$$

For later use, observe that

$$
\| v \| _ { H ^ { m } ( \Omega ; \mathbb { R } ^ { q } ) } ^ { 2 } \leq | \Omega | \binom { d + m } { m } \| v \| _ { C ^ { m } ( \overline { { \Omega } } ; \mathbb { R } ^ { q } ) } ^ { 2 } , \qquad v \in C ^ { m } ( \overline { { \Omega } } ; \mathbb { R } ^ { q } ) .
$$

Indeed, each term in the definition of the $H ^ { m }$ norm is bounded by $| \Omega | \| v \| _ { C ^ { m } } ^ { 2 }$ , and there are $\binom { d + m } { m }$ multi-indices ${ \boldsymbol { \alpha } } \in  { \mathbb { N } } _ { 0 } ^ { d }$ satisfying $| \alpha | \leq m$ . Consequently, $C ^ { m } ( \overline { { \Omega } } ; \mathbb { F } ^ { q } )$ is continuously embedded into $H ^ { m } ( \Omega ; \mathbb { F } ^ { q } )$ . The Schwartz space is defined by

$$
S ( \mathbb { R } ^ { d } ) : = \left\{ f \in C ^ { \infty } ( \mathbb { R } ^ { d } ) : \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \left| x ^ { \alpha } \partial ^ { \beta } f ( x ) \right| < \infty { \mathrm { ~ f o r ~ a l l ~ m u l t i - i n d i c e s ~ } } \alpha , \beta \right\} .
$$

It is equipped with the topology generated by the seminorms

$$
p _ { \alpha , \beta } ( f ) : = \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } | x ^ { \alpha } \partial ^ { \beta } f ( x ) | , \qquad \alpha , \beta \in \mathbb { N } _ { 0 } ^ { d } .
$$

We use the following Fourier-transform convention

$$
( \mathcal { F } f ) ( \boldsymbol { \xi } ) = \widehat { f } ( \boldsymbol { \xi } ) : = \int _ { \mathbb { R } ^ { d } } e ^ { - 2 \pi i \boldsymbol { x } \cdot \boldsymbol { \xi } } f ( \boldsymbol { x } ) d \boldsymbol { x } , \qquad f \in S ( \mathbb { R } ^ { d } ) ,\tag{1}
$$

and

$$
( { \mathcal { F } } ^ { - 1 } g ) ( x ) : = \int _ { \mathbb { R } ^ { d } } e ^ { 2 \pi i x \cdot \xi } g ( \xi ) d \xi , \qquad g \in S ( \mathbb { R } ^ { d } ) .
$$

By Plancherel’s theorem, $\mathcal { F }$ extends uniquely to a unitary operator on $L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } )$ , and it acts componentwise on $L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q } )$ . For a matrix-valued function κ and a compatible vector-valued function $f ,$ convolution means

$$
( \kappa * f ) ( x ) : = \int _ { \mathbb { R } ^ { d } } \kappa ( x - y ) f ( y ) d y
$$

whenever the integral is well defined.

For later use, define the bounded linear map

$$
\operatorname { A v g } _ { \Omega } : L ^ { 1 } ( \Omega ; \mathbb { F } ^ { q } ) \longrightarrow \mathbb { F } ^ { q } , \qquad \operatorname { A v g } _ { \Omega } ( v ) : = \frac { 1 } { | \Omega | } \int _ { \Omega } v ( y ) d y .\tag{2}
$$

The average in (2) is a vector in $\mathbb { F } ^ { q }$

## 2.2 Symplectic matrices and the metaplectic representation

We briefly recall the notation and standard facts from symplectic and metaplectic analysis that will be used below. For background, we refer to [12, Chapter 1], [15, Chapter 4] and [16].

Set

$$
J : = \left( \begin{array} { c c } { { 0 } } & { { I _ { d } } } \\ { { - I _ { d } } } & { { 0 } } \end{array} \right) .
$$

The real symplectic group is

$$
\operatorname { S p } ( 2 d , \mathbb { R } ) : = \left\{ S \in \operatorname { G L } ( 2 d , \mathbb { R } ) : S ^ { \mathsf { T } } J S = J \right\} .
$$

We write

$$
S = { \binom { A } { C } } \ { \begin{array} { l } { B } \\ { D } \end{array} } ) \in \operatorname { S p } ( 2 d , \mathbb { R } ) ,\tag{3}
$$

where $A , B , C , D \in \mathbb { R } ^ { d \times d }$ . Throughout the paper, unless stated otherwise, every $S \in \mathrm { S p } ( 2 d , \mathbb { R } )$ is written in the block form (3), with blocks $A , B , C , D \in \mathbb { R } ^ { d \times d }$ . The identity $S ^ { \mathsf { T } } J S = J$ implies, among other relations,

$$
\begin{array} { r } { A ^ { \mathsf { T } } C = C ^ { \mathsf { T } } A , \qquad B ^ { \mathsf { T } } D = D ^ { \mathsf { T } } B , \qquad A ^ { \mathsf { T } } D - C ^ { \mathsf { T } } B = I _ { d } , \qquad A B ^ { \mathsf { T } } = B A ^ { \mathsf { T } } . } \end{array}
$$

Definition 2.1 (Free symplectic matrix). A symplectic matrix $S$ in the block form (3) is calledfree if B is invertible.

So for any free symplectic matrix we associate the quadratic function

$$
W _ { S } ( x , \xi ) = \frac { 1 } { 2 } x ^ { \mathsf { T } } D B ^ { - 1 } x - x ^ { \mathsf { T } } B ^ { - \mathsf { T } } \xi + \frac { 1 } { 2 } \xi ^ { \mathsf { T } } B ^ { - 1 } A \xi .
$$

To each free symplectic matrix $S$ one can associate Metaplectic operator given by

$$
\widehat { S } f ( x ) : = \widehat { S } _ { W } f ( x ) = \frac { 1 } { \sqrt { | \operatorname* { d e t } B | } } \int _ { \mathbb { R } ^ { d } } e ^ { 2 \pi i W _ { S } ( x , \xi ) } f ( \xi ) d \xi .\tag{4}
$$

The Metaplectic operator is the unitary operator with an inverse given by

$$
\widehat { S } ^ { - 1 } f ( x ) = \frac { 1 } { \sqrt { | \operatorname* { d e t } B | } } \int _ { \mathbb { R } ^ { d } } e ^ { 2 \pi i , W _ { S ^ { - 1 } } ( x , \xi ) } f ( \xi ) , d \xi ,
$$

such that

$$
W _ { S ^ { - 1 } } ( \boldsymbol { x } , \boldsymbol { \xi } ) = - \frac { 1 } { 2 } \boldsymbol { x } \cdot \boldsymbol { A } ^ { \top } \boldsymbol { B } ^ { - \top } \boldsymbol { x } + ( \boldsymbol { B } ^ { - 1 } \boldsymbol { x } ) \cdot \boldsymbol { \xi } - \frac { 1 } { 2 } \boldsymbol { \xi } \cdot \boldsymbol { B } ^ { - \top } \boldsymbol { D } ^ { \top } \boldsymbol { \xi }
$$

and

$$
\begin{array} { r l } { S ^ { - 1 } = \left( \begin{array} { l l } { D ^ { \mathsf { T } } } & { - B ^ { \mathsf { T } } } \\ { - C ^ { \mathsf { T } } } & { A ^ { \mathsf { T } } } \end{array} \right) . } \end{array}
$$

For a free symplectic matrix, define

$$
Q _ { S } : = B ^ { - 1 } A , \qquad T _ { S } : = D B ^ { - 1 } , \qquad \gamma _ { S } ( x ) : = e ^ { \pi i x \cdot Q _ { S } x } .\tag{5}
$$

For $f \in { \mathcal { S } } ( \mathbb { R } ^ { d } )$ , we denote

$$
\begin{array} { r l } & { ( \mathcal { C } _ { T s } f ) ( x ) : = e ^ { \pi i \langle T _ { S } x , x \rangle } f ( x ) , } \\ & { ( \mathcal { U } _ { B } f ) ( x ) : = \cfrac { 1 } { \sqrt { | \operatorname* { d e t } B | } } f ( B ^ { - 1 } x ) , } \\ & { ( \mathcal { C } _ { Q _ { S } } f ) ( x ) : = e ^ { \pi i \langle Q _ { S } x , x \rangle } f ( x ) , } \end{array}
$$

where $\langle \cdot , \cdot \rangle$ denotes the usual inner product, defined by $\langle u , v \rangle = u ^ { \mathsf { T } } v$

Moreover the matrices $Q _ { S }$ and $T _ { S }$ are symmetric. Indeed, the symmetry of $Q _ { S }$ follows from $A B ^ { \mathsf { T } } = B A ^ { \mathsf { T } }$ , while the symmetry of $T _ { S }$ follows from $B ^ { \mathsf { T } } D = D ^ { \mathsf { T } } \bar { B }$

Proposition 2.2 (Free metaplectic factorization). Let $S \in \mathrm { S p } ( 2 d , \mathbb { R } )$ befree. Then

$$
S = V _ { T _ { S } } \Lambda _ { B } J V _ { Q _ { S } } .
$$

where

$$
V _ { T s } : = \left( { \begin{array} { c c } { I _ { d } } & { 0 } \\ { T _ { S } } & { I _ { d } } \end{array} } \right) , \qquad \Lambda _ { B } : = \left( { \begin{array} { c c } { B } & { 0 } \\ { 0 } & { B ^ { - \top } } \end{array} } \right) , \qquad J : = \left( { \begin{array} { c c } { 0 } & { I _ { d } } \\ { - I _ { d } } & { 0 } \end{array} } \right) ,
$$

and $V _ { Q s }$ is defined analogously, with $T _ { S }$ replaced by $Q _ { S }$

Moreover, for every $f \in { \mathcal { S } } ( \mathbb { R } ^ { d } )$

$$
\widehat { S } f ( \boldsymbol { x } ) = \big ( \mathcal { C } _ { T _ { S } } \mathcal { U } _ { B } \mathcal { F } \mathcal { C } _ { Q _ { S } } f \big ) ( \boldsymbol { x } ) ,\tag{6}
$$

Proof. The symplectic identities $B ^ { \mathsf { T } } D = D ^ { \mathsf { T } }$ B and $D ^ { \mathsf { T } } A - B ^ { \mathsf { T } } C = I { \mathrm { ~ g i v e ~ } } C = D B ^ { - 1 } A - B ^ { - \mathsf { T } }$ Hence

$$
V _ { T _ { S } } \Lambda _ { B } J V _ { Q _ { S } } = \left( \begin{array} { c c } { { A } } & { { B } } \\ { { D B ^ { - 1 } A - B ^ { - \top } } } & { { D } } \end{array} \right) = S .
$$

moreover, for $f \in { \mathcal { S } } ( \mathbb { R } ^ { d } )$

$$
\begin{array} { r } { \big ( \mathcal { C } _ { T _ { S } } \mathcal { U } _ { B } \mathcal { F } \mathcal { C } _ { Q _ { S } } f \big ) ( x ) = \frac { e ^ { \pi i x \cdot T _ { S } x } } { \sqrt { | \operatorname* { d e t } B | } } \int _ { \mathbb { R } ^ { d } } e ^ { - 2 \pi i ( B ^ { - 1 } x ) \cdot \xi } e ^ { \pi i \xi \cdot Q _ { S } \xi } f ( \xi ) d \xi } \\ { = \frac { 1 } { \sqrt { | \operatorname* { d e t } B | } } \int _ { \mathbb { R } ^ { d } } e ^ { 2 \pi i W _ { S } ( x , \xi ) } f ( \xi ) d \xi = \widehat { S } f ( x ) . } \end{array}
$$

## 2.3 Bounded-domain metaplectic multipliers

For $q \in \mathbb { N } ,$ define extension by zero and restriction by

$$
\begin{array} { r l } { \mathsf { E } _ { \Omega , q } : L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q } ) \longrightarrow L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q } ) , } & { \qquad ( \mathsf { E } _ { \Omega , q } v ) ( x ) : = \left\{ \overset { v ( x ) , } { 0 } , \quad x \in \Omega , \right. } \\ { \mathsf { R } _ { \Omega , q } : L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q } ) , } & { \qquad \mathsf { R } _ { \Omega , q } f : = f | _ { \Omega } . } \end{array}
$$

These operators are bounded and satisfy the following estimates:

$$
\begin{array} { r } { \| \mathsf { E } _ { \Omega , q } v \| _ { L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q } ) } = \| v \| _ { L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q } ) } , \qquad \| \mathsf { R } _ { \Omega , q } f \| _ { L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q } ) } \leq \| f \| _ { L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q } ) } . } \end{array}\tag{7}
$$

For $q \in \mathbb { N }$ , the componentwise metaplectic operator is $\widehat { S } _ { q } : L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q } ) \longrightarrow L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q } )$ such that

$$
{ \widehat { S } } _ { q } ( f ) : = { \widehat { S } } _ { q } ( f _ { 1 } , \dots , f _ { q } ) : = \big ( { \widehat { S } } f _ { 1 } , \dots , { \widehat { S } } f _ { q } \big ) , \qquad f _ { j } \in L ^ { 2 } ( \mathbb { R } ^ { d } ) .
$$

Note that $\widehat { S } _ { q }$ is unitary. Let $q _ { - } , q _ { + } \in \mathbb { N }$ and let

$$
P : \mathbb { R } ^ { d } \longrightarrow \mathbb { C } ^ { q _ { + } \times q _ { - } }\tag{8}
$$

be a measurable and essentially bounded matrix-valued function. We define the associated multiplication operator

$$
M _ { P } : L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q _ { - } } ) \longrightarrow L ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { C } ^ { q _ { + } } ) , \qquad ( M _ { P } g ) ( \xi ) : = P ( \xi ) g ( \xi ) .\tag{9}
$$

The essential boundedness of $P$ guarantees that $M _ { P }$ is a well-defined bounded linear operator. For a matrix-valued symbol $P$ we set

$$
\| P \| _ { L _ { \mathrm { o p } } ^ { \infty } } : = \underset { \xi \in \mathbb { R } ^ { d } } { \mathrm { e s s } } \| P ( \xi ) \| _ { \mathrm { o p } } .
$$

Lemma 2.3 (Norm of a matrix-valued multiplication operator). The operator in (9) is bounded and

$$
\| M _ { P } \| _ { L ^ { 2 } \to L ^ { 2 } } = \| P \| _ { L _ { \mathrm { o p } } ^ { \infty } } .\tag{10}
$$

Proof. The proof is similar to [19, Theorem 1.1]. Therefore, the details are left to the reader.

Definition 2.4. For S free symplectic matrix, and let P be a matrix-valued symbol as in Equation (8). The associated kernel branch

$$
\begin{array} { r l } { { \mathcal K } _ { \widehat S , P } ^ { \Omega } : L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { - } } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { + } } ) } \end{array}
$$

is defined as following

$$
\begin{array} { r } { \mathcal { K } _ { \widehat { S } , P } ^ { \Omega } : = \mathsf { R } _ { \Omega , q _ { + } } \widehat { S } _ { q _ { + } } ^ { - 1 } M _ { P } \widehat { S } _ { q _ { - } } \mathsf { E } _ { \Omega , q _ { - } } . } \end{array}
$$

Remark 2.5. From $( 7 )$ , and (10) , we deduce that is the bounded linear operator and satisfies the following inequality

$$
\| { \mathcal { K } } _ { \widehat { S } , P } ^ { \Omega } \| _ { L ^ { 2 } \to L ^ { 2 } } \leq \| P \| _ { L _ { \mathrm { o p } } ^ { \infty } } .
$$

The following proposition shows that, for a free symplectic matrix, the branch kernel can be represented as a chirp-conjugated Fourier multiplier.

Proposition 2.6 (Free-transform conjugation formula). Let S be free, and let P be as in (8).

Then

$$
\widehat { S } _ { q _ { + } } ^ { - 1 } M _ { P } \widehat { S } _ { q _ { - } } = { \mathcal { C } _ { - Q _ { S } } } \mathcal { F } ^ { - 1 } M _ { P ( B . ) } \mathcal { F } { \mathcal { C } _ { Q _ { S } } } ,\tag{11}
$$

moreover,

$$
\mathcal { K } _ { \widehat { S } , P } ^ { \Omega } = \mathsf { R } _ { \Omega , q _ { + } } \mathcal { C } _ { - Q _ { S } } \mathcal { F } ^ { - 1 } M _ { P ( B \cdot ) } \mathcal { F } \mathcal { C } _ { Q _ { S } } \mathsf { E } _ { \Omega , q _ { - } } .\tag{12}
$$

Proof. Combining Equation (6) and the identities $\mathcal { C } _ { - T _ { S } } M _ { P } \mathcal { C } _ { T _ { S } } = M _ { P }$ and $\mathcal { U } _ { B } ^ { - 1 } M _ { P } \mathcal { U } _ { B } = M _ { P ( B \cdot ) } ,$ we get Equation (11). Definition 2.4 and Equation (11) give (12). 白

This formula will be used below (see Proposition 2.17) with a scalar-valued symbol for which the Fourier multiplier becomes convolution with a kernel that is constant on $\Delta _ { \Omega }$ from (21).

## 2.4 Realification and finite-parameter classes

The metaplectic transform is complex-valued even when its input is real. We therefore encode each complex channel by two real channels.

Definition 2.7 (Realification). For $q \in \mathbb { N }$ , the map

$$
\mathfrak { r } _ { q } : \mathbb { C } ^ { q } \longrightarrow \mathbb { R } ^ { 2 q } , \qquad \mathfrak { r } _ { q } ( z ) : = \binom { \mathrm { R e } z } { \mathrm { I m } z }
$$

is called the realification map.

Remark 2.8. For $\Theta = U + i V \in \mathbb { C } ^ { q _ { + } \times q }$ <sup>−</sup> , define

$$
[ \Theta ] _ { \mathbb { R } } : = { \binom { U } { V } } \quad { \textstyle - V } \bigg ( \mathbb { R } ^ { 2 q _ { + } \times 2 q _ { - } } .
$$

We also call $\Theta \mapsto [ \Theta ] _ { \mathbb { R } }$ realification.

For every $z \in \mathbb { C } ^ { q . }$

$$
\mathfrak { r } _ { q _ { + } } ( \Theta z ) = [ \Theta ] _ { \mathbb { R } } \mathfrak { r } _ { q _ { - } } ( z ) .
$$

Remark 2.9. Realification function commutes with differentiation and preserves the standard $C ^ { r }$ and $H ^ { r }$ norms. For any bounded complex-linear operator $T : L ^ { 2 } ( \Omega ; { \bar { \mathbb { C } } } ^ { q _ { - } } ) \to L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { + } } )$ , its realification

$$
T _ { \mathbb { R } } : = { \mathfrak { r } } _ { q _ { + } } T { \mathfrak { r } } _ { q _ { - } } ^ { - 1 }
$$

is real-linear and satisfies $\| T _ { \mathbb { R } } \| _ { \mathrm { o p } } = \| T \| _ { \mathrm { o p } }$

We can now define the realified bounded-domain metaplectic multiplier by

$$
\mathbf { K } _ { \widehat { S } , P } ^ { \Omega } : = \mathfrak { r } _ { q _ { + } } \mathcal { K } _ { \widehat { S } , P } ^ { \Omega } \mathfrak { r } _ { q _ { - } } ^ { - 1 } .\tag{13}
$$

Definition 2.10 (Finite-parameter symbol class). Fix $q _ { - } , q _ { + } , M \in \mathbb { N }$ and fixed $\rho _ { 1 } , \dotsc , \rho _ { M } \in$ $L ^ { \infty } ( \mathbb { R } ^ { d } ; \mathbb { C } )$ . For trainable matrices $\Theta _ { 1 } , \hdots , \Theta _ { M } \in \mathbb { C } ^ { q _ { + } \times q _ { - } }$ , define

$$
P _ { \theta } ( \xi ) : = \sum _ { r = 1 } ^ { M } \Theta _ { r } \rho _ { r } ( \xi ) , \qquad \theta = ( \Theta _ { 1 } , \ldots , \Theta _ { M } ) .
$$

The admissible trainable biases are restricted to the following finite-dimensional function space. Definition 2.11 (Finite-parameter bias class). Fix m $\in  { \mathbb { N } } _ { 0 }$ and a fixed positional vector

$$
p = ( p _ { 1 } , \dots , p _ { m } ) : \overline { { \Omega } } \longrightarrow \mathbb { R } ^ { m } , \qquad p _ { j } \in C ^ { \infty } ( \overline { { \Omega } } ) .
$$

For $n \in \mathbb { N } ,$ , a trainable bias function has the form

$$
b ( x ) : = c + G p ( x ) , \qquad c \in \mathbb { R } ^ { n } , \quad G \in \mathbb { R } ^ { n \times m } .
$$

When $m = 0 .$ , or when $G = 0$ , this reduces to a constant bias.

For a fixed free transform, a useful positional vector is

$$
p _ { S } ( x ) : = \bigl ( x _ { 1 } , \ldots , x _ { d } , \cos ( \pi x \cdot Q _ { S } x ) , \sin ( \pi x \cdot Q _ { S } x ) \bigr ) .\tag{14}
$$

This allows real-valued networks to use the spatial coordinates and the real and imaginary parts of the fixed chirp $\gamma _ { S }$ as inputs. The architecture may use any fixed finite family of positional features containing those required in the proof; these features are prescribed and are not trainable.

Remark 2.12. More general formulations are possible in which the symbol and the bias function are allowed to vary in infinite-dimensional function classes. We do not pursue such generalizations here, since the finite-dimensional classes introduced above are sufficient for all constructions in this paper.

## 2.5 Activation and local neural maps

Assumption 2.13 (Activation). The activation function $\sigma : \mathbb { R } $ R is nonpolynomial, globally Lipschitz, and smooth. Its Lipschitz constant is denoted by $L _ { \sigma }$

For $n \in \mathbb { N }$ , denote the component wise map by

$$
\Sigma _ { n } ( z _ { 1 } , \dots , z _ { n } ) : = \big ( \sigma ( z _ { 1 } ) , \dots , \sigma ( z _ { n } ) \big ) .
$$

Lemma 2.14 (A locally invertible activation interval). There exist $t _ { * } ~ \in \ \mathbb { R }$ and open intervals $I _ { \sigma } , J _ { \sigma } \subset \mathbb { R }$ such that $t _ { * } \in I _ { \sigma } , \sigma ( t _ { * } ) \in J _ { \sigma }$ , and

$$
\sigma | _ { I _ { \sigma } } : I _ { \sigma } \longrightarrow J _ { \sigma }
$$

is a $C ^ { \infty }$ diffeomorphism.

Proof. The proof follows from applying the inverse function theorem.

We next fix a finite-dimensional neural-network convention. Let $N \in \mathbb N$ and $n _ { 0 } , . . . , n _ { N } \in \mathbb { N }$ N. An activated-output network is defined by

$$
z _ { 0 } = z , \qquad z _ { j } = \Sigma _ { n _ { j } } ( A _ { j } z _ { j - 1 } + a _ { j } ) , \quad j = 1 , \dots , N , \qquad \Phi ( z ) : = z _ { N } .
$$

An affine-output network is defined by the same recursion for $j = 1 , \ldots , N - 1$ and by

$$
\Phi ( z ) : = A _ { N } z _ { N - 1 } + a _ { N } .
$$

Since the scalar activation $\sigma$ is globally Lipschitz with constant $L _ { \sigma } .$ , an activated-output network is globally Lipschitz with constant at most $\begin{array} { r } { L _ { \sigma } ^ { N } \prod _ { j = 1 } ^ { N } { \| A _ { j } \| _ { \mathrm { o p } } } } \end{array}$ . For an affine-output network, the corresponding bound is $\begin{array} { r } { \| A _ { N } \| _ { \mathrm { o p } } L _ { \sigma } ^ { N - 1 } \prod _ { j = 1 } ^ { N - 1 } \| A _ { j } \| _ { \mathrm { o p } } ^ { - } } \end{array}$

Fix $q _ { \mathrm { i n } } , q _ { \mathrm { o u t } } \in \mathbb { N }$ and $m _ { \mathrm { i n } } , m _ { \mathrm { o u t } } \in \mathbb { N } _ { 0 }$ , and let $p _ { \mathrm { i n } } \in C ^ { \infty } ( \overline { { \Omega } } ; \mathbb { R } ^ { m _ { \mathrm { i n } } } ) , p _ { \mathrm { o u t } } \in C ^ { \infty } ( \overline { { \Omega } } ; \mathbb { R } ^ { m _ { \mathrm { o u t } } } )$ be fixed positional vectors.

$\mathrm { L e t } \ \Phi ^ { \mathrm { l i f t } }$ be an activated-output network from $\mathbb { R } ^ { d _ { a } + m _ { \mathrm { i n } } } \ \mathrm { t o } \ \mathbb { R } ^ { 2 q _ { \mathrm { i n } } }$ , and let $\Phi ^ { \mathrm { o u t } }$ be an affine-output network from R<sup>2qout+mout</sup> $\mathbf { t o } \ \mathbb { R } ^ { d _ { u } }$ . Their pointwise realizations are

$$
\begin{array} { r } { \mathcal { R } : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { a } } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { R } ^ { 2 q _ { \mathrm { i n } } } ) , \qquad \mathcal { R } ( a ) ( x ) : = \Phi ^ { \mathrm { l i f t } } \big ( a ( x ) , p _ { \mathrm { i n } } ( x ) \big ) , } \end{array}\tag{15}
$$

$$
\ Q : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { 2 q _ { \mathrm { o u t } } } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { u } } ) , \qquad Q ( v ) ( x ) : = \Phi ^ { \mathrm { o u t } } \big ( v ( x ) , p _ { \mathrm { o u t } } ( x ) \big ) .\tag{16}
$$

The values $m _ { \mathrm { i n } } = 0 \mathrm { o r } m _ { \mathrm { o u t } } = 0$ are allowed. For example, when $m _ { \mathrm { i n } } = 0$ , the lifting network depends only on a(x) and receives no additional positional information.

Since $p _ { \mathrm { i n } }$ and $p _ { \mathrm { o u t } }$ are bounded on $\overline { { \Omega } }$ and $| \Omega | < \infty$ , the global Lipschitz continuity of $\Phi ^ { \mathrm { l i f t } }$ and Φ<sup>out</sup> implies that the pointwise maps R and Q are well defined between the corresponding $L ^ { 2 }$ spaces. With the positional inputs fixed, the same Lipschitz estimates show that R and $\dot { \mathcal { Q } }$ are globally Lipschitz with respect to their function-valued inputs.

The activated-output convention in (15) will be convenient when composing the maps with the metaplectic layers. The final output map is affine-output, so no activation is applied at the last layer, and its values are not constrained to lie in the range of $\sigma$ . Moreover, an internal hidden layer may always be enlarged by adding unused units and setting all outgoing weights from those units equal to zero. Thus, when needed for rectification, its width may be assumed to be even.

Definition 2.15 (Finite metaplectic neural operator). Let $L \in \mathbb { N }$ and let $q _ { 0 } , \ldots , q _ { L } \in \mathbb { N }$ . Let

$$
\mathcal { R } : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { a } } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { R } ^ { 2 q _ { 0 } } )
$$

be a lifting operator of the form (15), and let

$$
\mathcal { Q } : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { 2 q _ { L } } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { u } } )
$$

be a projection operator of the form (16).

For each $\ell \in \{ 1 , \ldots , L \}$ , choose $J _ { \ell } \in  { \mathbb { N } } _ { 0 } ,$ , a matrix $W _ { \ell } \in \mathbb { R } ^ { 2 q _ { \ell } \times 2 q _ { \ell - 1 } }$ , and a finite-parameter bias function $b _ { \ell } : \overline { { \Omega } } \longrightarrow \mathbb { R } ^ { 2 q _ { \ell } }$ as in Definition 2.11. For every $j \in \{ 1 , \ldots , J _ { \ell } \}$ , choose a Symplectic $S _ { \ell , j }$ and a symbol

$$
P _ { \ell , j } : \mathbb { R } ^ { d } \longrightarrow \mathbb { C } ^ { q _ { \ell } \times q _ { \ell - 1 } }
$$

belonging to a finite-parameter symbol class as in Definition 2.10. Let $\mathcal { K } _ { \widehat { S } _ { \ell , j } , P _ { \ell , j } } ^ { \Omega }$ be the associated bounded-domain metaplectic multiplier from Definition 2.4.

Its realification $\mathbf { K } _ { \widehat { S } _ { \ell , j } , P \ell , j } ^ { \Omega }$ is defined by (13). The corresponding hidden metaplectic layer is

$$
\mathcal { L } _ { \ell } : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { 2 q _ { \ell - 1 } } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { R } ^ { 2 q _ { \ell } } ) ,
$$

defined by

$$
\mathcal { L } _ { \ell } ( v ) ( x ) : = \Sigma _ { 2 q _ { \ell } } \left( W _ { \ell } v ( x ) + b _ { \ell } ( x ) + \sum _ { j = 1 } ^ { J _ { \ell } } \left( \mathbf { K } _ { \widehat { S } _ { \ell , j } , P _ { \ell , j } } ^ { \Omega } v \right) ( x ) \right) .\tag{17}
$$

When $J _ { \ell } = 0$ , the sum in (17) is understood to be zero.

The operator

$$
\mathcal { N } : = \mathcal { Q } \circ \mathcal { L } _ { L } \circ \cdots \circ \mathcal { L } _ { 1 } \circ \mathcal { R }\tag{18}
$$

is called a finite metaplectic neural operator. If $L = 1$ , then $\mathcal { N }$ is said to have one hidden metaplectic layer.

Lemma 2.16 (Compact range encoding with an activated output). Let $K \subset \mathbb { R } ^ { n }$ be compact and let $f : K  \mathbb { R } ^ { m }$ be continuous. There exist $\boldsymbol { a } _ { f } \in \mathbb { R } ^ { m }$ and $\tau _ { f } > 0$ such that, for every $\varepsilon > 0 ,$ , an activated-output network Φ can be chosen with

$$
\operatorname* { s u p } _ { z \in K } \bigl | \Phi ( z ) - a _ { f } - \tau _ { f } f ( z ) \bigr | < \varepsilon .\tag{19}
$$

$H ,$ in addition, $r \in  { \mathbb { N } } _ { 0 } ,$ , the compact set K is contained in an open set $U \subset \mathbb { R } ^ { n }$ , and $f \in C ^ { r } ( U ; \mathbb { R } ^ { m } )$ then $a _ { f }$ and $\tau _ { f }$ can be chosen so that, for every $\varepsilon > 0 ,$ , the network Φ satisfies

$$
\operatorname* { m a x } _ { | \alpha | \leq r } \operatorname* { s u p } _ { z \in K } \big | \partial ^ { \alpha } \Phi ( z ) - \partial ^ { \alpha } \big ( a _ { f } + \tau _ { f } f ( z ) \big ) \big | < \varepsilon .\tag{20}
$$

Proof. Let $t _ { * }$ and the intervals $I _ { \sigma } , J _ { \sigma }$ be as in Lemma 2.14. Choose $\rho > 0$ such that $[ \sigma ( t _ { * } ) -$ $\rho , \sigma ( t _ { * } ) + \rho ] \subset J _ { \sigma }$ . Note that $f ( K )$ is a bounded set . Hence there exists $M _ { f } \geq 0$ such that

$$
| f _ { j } ( z ) | \le M _ { f } , \qquad z \in K , \quad j = 1 , \ldots , m .
$$

Set $a _ { f } : = \sigma ( t _ { * } ) \mathbf { 1 } _ { m }$ and choose $\begin{array} { r } { 0 \ < \ \tau _ { f } \ < \ \frac { \rho } { 1 + M _ { f } } } \end{array}$ . Then $| \tau _ { f } f _ { j } ( z ) | < \rho$ for every $z \in K$ and $j = 1 , \dots , m .$ . Consequently, every component of $\dot { a _ { f } } + \tau _ { f } f ( z )$ belongs to $J _ { \sigma }$

The inverse of $\sigma | _ { I _ { \sigma } }$ can therefore be applied componentwise, and the map

$$
g ( z ) : = ( \sigma | _ { I _ { \sigma } } ) ^ { - 1 } \bigl ( a _ { f } + \tau _ { f } f ( z ) \bigr ) , \qquad z \in K ,
$$

is well defined and continuous. By the finite-dimensional universal approximation theorem [26], there exists an affine-output network Ψ that approximates g uniformly on $K .$ . The vector-valued case is obtained by applying the scalar approximation result to each component of $g$

Append one componentwise activation and define

$$
\Phi : = \Sigma _ { m } \circ \Psi .
$$

This is an activated-output network. By the definition of $g , \Sigma _ { m } ( g ( z ) ) = a _ { f } + \tau _ { f } f ( z )$ . Then

$$
| \Phi ( z ) - a _ { f } - \tau _ { f } f ( z ) | = | \Sigma _ { m } ( \Psi ( z ) ) - \Sigma _ { m } ( g ( z ) ) | \leq L _ { \sigma } | \Psi ( z ) - g ( z ) | .
$$

Thus, by approximating g sufficiently accurately with $\Psi$ , we obtain (19).

For the $C ^ { r }$ statement, choose a compact neighborhood $K _ { 1 } \subset U$ such that $K \subset \operatorname { i n t } ( K _ { 1 } )$ . Since $f$ is continuous, it is bounded on $K _ { 1 }$ . We choose $a _ { f } = \sigma ( t _ { * } ) \mathbf { 1 } _ { m }$ as above and choose $\tau _ { f } > 0$ using a bound for $f$ on $K _ { 1 }$ . Then every component of $\dot { a _ { f } } + \tau _ { f } f ( z )$ lies in $J _ { \sigma }$ for every z in a neighborhood of $K$ . Hence

$$
g ( z ) = ( \sigma | _ { I _ { \sigma } } ) ^ { - 1 } \big ( a _ { f } + \tau _ { f } f ( z ) \big )
$$

is of class $C ^ { r }$ on a neighborhood of K.

By the simultaneous approximation theorem for functions and their derivatives [27, Theorem 2.1], the affine-output network $\bar { \Psi }$ can be chosen so that g and all its derivatives up to order r are approximated uniformly on $K .$ . Since $\sigma$ is smooth, the chain rule implies that

$$
\Sigma _ { m } \circ \Psi \longrightarrow \Sigma _ { m } \circ g = a _ { f } + \tau _ { f } f
$$

in the $C ^ { r }$ norm on K. Choosing the approximation sufficiently accurate gives (20).

## 2.6 Exact metaplectic averaging branch

We now construct the single fixed symbol atom used in the later universality proof. Fix a free symplectic matrix S. Define the compact symmetric difference set

$$
\Delta _ { \Omega } : = \overline { { \Omega } } - \overline { { \Omega } } = \{ x - y : x , y \in \overline { { \Omega } } \} .\tag{21}
$$

Choose a real-valued even function

$$
\chi _ { \Omega } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) , \qquad \chi _ { \Omega } ( z ) = 1 \quad \mathrm { o n ~ a n ~ o p e n ~ n e i g h b o r h o o d ~ o f ~ } \Delta _ { \Omega } .\tag{22}
$$

Set

$$
\kappa _ { \Omega } ( z ) : = \frac { 1 } { | \Omega | } \chi _ { \Omega } ( z ) , \qquad \rho _ { \widehat { S } , \Omega } ( \eta ) : = \widehat { \kappa _ { \Omega } } ( B ^ { - 1 } \eta ) .\tag{23}
$$

Proposition 2.17 (Exact chirp-conjugated averaging). For every $q \in \mathbb { N }$ and every $v \in L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q } )$

$$
\mathcal { K } _ { \widehat { S } , \rho _ { \widehat { S } , \Omega } } ^ { \Omega } I _ { q } v ( x ) = \overline { { \gamma _ { S } ( x ) } } \operatorname { A v g } _ { \Omega } ( \gamma _ { S } v )\tag{24}
$$

for almost every $x \in \Omega ,$ , where $I _ { q }$ denotes the $q \times q$ identity matrix.

More generally, $i f q _ { - } , q _ { + } \in \mathbb { N } , \Theta \in \mathbb { C } ^ { q _ { + } \times q _ { - } }$ , and

$$
P _ { \Theta } ( \eta ) : = \Theta \rho _ { \widehat { S } , \Omega } ( \eta ) ,\tag{25}
$$

then

$$
\begin{array} { r } { K _ { \widehat { S } , P _ { \Theta } } ^ { \Omega } v ( x ) = \overline { { \gamma _ { S } ( x ) } } \Theta \operatorname { A v g } _ { \Omega } ( \gamma _ { S } v ) } \end{array}\tag{26}
$$

for every $v \in L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { - } } )$

Proof. By the definition of $\rho _ { \widehat { S } , \Omega }$ in (23), one has

$$
\begin{array} { r } { \rho _ { \widehat { S } , \Omega } ( B \xi ) = \widehat { \kappa _ { \Omega } } ( B ^ { - 1 } B \xi ) = \widehat { \kappa _ { \Omega } } ( \xi ) . } \end{array}\tag{27}
$$

We first consider the symbol $\rho _ { \widehat { S } , \Omega } I _ { q }$ . By Proposition 2.6,

$$
\begin{array} { r } { \mathcal { K } _ { \widehat { S } , \rho _ { \widehat { S } , \Omega } ^ { 2 } I _ { q } } ^ { \Omega } v = \mathsf { R } _ { \Omega , q } \left( \overline { { \gamma _ { S } } } \mathcal { F } ^ { - 1 } \left[ \rho _ { \widehat { S } , \Omega } ( B \cdot ) I _ { q } \mathcal { F } \big ( \mathsf { E } _ { \Omega , q } ( \gamma _ { S } v ) \big ) \right] \right) . } \end{array}
$$

Using (27), this becomes

$$
\begin{array} { r } { { \cal K } _ { \widehat { S } , \rho _ { \widehat { S } , \Omega } I _ { q } } ^ { \Omega } v = { \cal { R } } _ { \Omega , q } \left( \overline { { \gamma _ { S } } } \mathcal { F } ^ { - 1 } \left[ \widehat { \kappa _ { \Omega } } I _ { q } \mathcal { F } \big ( \mathsf { E } _ { \Omega , q } ( \gamma _ { S } v ) \big ) \right] \right) . } \end{array}
$$

The Fourier transform is applied componentwise to $\mathbb { C } ^ { q } .$ -valued functions. Therefore, for almost every $x \in \Omega$

$$
\begin{array} { l } { { \mathcal { K } _ { \widehat { S } , \rho _ { \widehat { S } , \Omega } } ^ { \Omega } I _ { q } ^ { \ } } \displaystyle v ( x ) = \overline { { \gamma _ { S } ( x ) } } \big [ \kappa _ { \Omega } * \mathsf { E } _ { \Omega , q } ( \gamma _ { S } v ) \big ] ( x ) } \\ { \displaystyle \quad = \overline { { \gamma _ { S } ( x ) } } \int _ { \mathbb { R } ^ { d } } \kappa _ { \Omega } ( x - y ) \mathsf { E } _ { \Omega , q } ( \gamma _ { S } v ) ( y ) d y } \\ { \displaystyle \quad = \overline { { \gamma _ { S } ( x ) } } \int _ { \Omega } \kappa _ { \Omega } ( x - y ) \gamma _ { S } ( y ) v ( y ) d y . } \end{array}\tag{28}
$$

The last equality follows from the definition of the zero extension $\mathsf { E } _ { \Omega , q } .$

For $x , y \in \Omega$ , one has $x - y \in \Omega - \Omega \subseteq \Delta _ { \Omega }$ . By (22), $\chi _ { \Omega } ( x - y ) = 1$ , and hence

$$
\kappa \Omega ( x - y ) = \frac { 1 } { | \Omega | } .
$$

Substituting this identity into (28), we obtain

$$
\mathcal { K } _ { \widehat { S } , \rho _ { \widehat { S } , \Omega } } ^ { \Omega } I _ { q } v ( x ) = \overline { { \gamma _ { S } ( x ) } } \frac { 1 } { | \Omega | } \int _ { \Omega } \gamma _ { S } ( y ) v ( y ) d y = \overline { { \gamma _ { S } ( x ) } } \mathrm { A v g } _ { \Omega } ( \gamma _ { S } v ) .
$$

The integral and the average are understood componentwise and therefore belong to $\mathbb { C } ^ { q }$ . This proves (24).

For the second assertion, since $P _ { \Theta } = \Theta \big ( \rho _ { \widehat { S } , \Omega } I _ { q _ { - } } \big )$ and Θ is constant and $I _ { q _ { - } }$ is the $q _ { - } \times q .$ <sub>−</sub> identity matrix, yields

$$
K _ { \widehat { S } , P _ { \Theta } } ^ { \Omega } v = \Theta K _ { \widehat { S } , \rho _ { \widehat { S } , \Omega } I _ { q _ { - } } } ^ { \Omega } v .
$$

Hence (24), applied with $q = q .$ <sub>−</sub>, yields

$$
\begin{array} { r } { \mathcal { K } _ { \widehat { S } , P _ { \Theta } } ^ { \Omega } v ( x ) = \overline { { \gamma _ { S } ( x ) } } \Theta \operatorname { A v g } _ { \Omega } ( \gamma _ { S } v ) } \end{array}
$$

for almost every $x \in \Omega$ , proving (26).

Proposition 2.18 (Finite rank and positive regularity of the averaging branch). Let $r \in  { \mathbb { N } } _ { 0 } ,$ , let $\Theta \in \bar { \mathbb { C } } ^ { q _ { + } \times q _ { - } }$ <sup>−</sup> , and let $P _ { \Theta }$ be defined by (25). The range of $\cdot \kappa _ { \widehat { S } , P \Theta } ^ { \Omega }$ acting on $L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { - } } )$ is contained in the finite-dimensional space

$$
\mathcal { V } _ { S , q _ { + } } : = \left\{ x \longmapsto \overline { { \gamma _ { S } ( x ) } } c : c \in \mathbb { C } ^ { q _ { + } } \right\} .
$$

Moreover, for every $v \in L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { - } } )$

$$
\| K _ { \widehat { S } , P _ { \Theta } } ^ { \Omega } v \| _ { C ^ { r } ( \Omega ; \mathbb { C } ^ { q _ { + } } ) } \leq C _ { r , S , \Omega } ^ { C } \| \Theta \| _ { \mathrm { o p } } | \Omega | ^ { - 1 / 2 } \| v \| _ { L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { - } } ) } ,\tag{29}
$$

$$
\| K _ { \hat { S } , P _ { \Theta } } ^ { \Omega } v \| _ { H ^ { r } ( \Omega ; \mathbb { C } ^ { q _ { + } } ) } \leq C _ { r , S , \Omega } ^ { H } \| \Theta \| _ { \mathrm { o p } } | \Omega | ^ { - 1 / 2 } \| v \| _ { L ^ { 2 } ( \Omega ; \mathbb { C } ^ { q _ { - } } ) } ,\tag{30}
$$

where

$$
C _ { r , S , \Omega } ^ { C } : = \operatorname* { m a x } _ { | \alpha | \leq r } \| \partial ^ { \alpha } \gamma _ { S } \| _ { L ^ { \infty } ( \Omega ; \mathbb { C } ) } , \qquad C _ { r , S , \Omega } ^ { H } : = \left( \sum _ { | \alpha | \leq r } \| \partial ^ { \alpha } \gamma _ { S } \| _ { L ^ { 2 } ( \Omega ; \mathbb { C } ) } ^ { 2 } \right) ^ { 1 / 2 } .
$$

Proof. The range statement follows directly from (26). Since $| \gamma _ { S } | = 1$ , the fact that by Cauchy– Schwarz,

$$
| \operatorname { A v g } _ { \Omega } ( v ) | \leq | \Omega | ^ { - 1 / 2 } \| v \| _ { L ^ { 2 } ( \Omega ) } ,\tag{31}
$$

we get

$$
| \Theta \operatorname { A v g } _ { \Omega } ( \gamma _ { S } v ) | \leq \| \Theta \| _ { \mathrm { o p } } | \Omega | ^ { - 1 / 2 } \| v \| _ { L ^ { 2 } } .\tag{32}
$$

Differentiate the fixed factor $\overline { { \gamma _ { S } } }$ and use (32). This gives (29) and (30). Note that, realification is an isometry and doubles the finite-dimensional real range dimension at most. □

## 2.7 Auxiliary results for the universality proof

This subsection develops the auxiliary results used in the proof of the metaplectic neural operator universal approximation theorem.

## 2.7.1 Finite-code averaging operators

The replacement for the Fourier coefficient vector is a nonlinear finite code obtained by averaging a local feature map. We first isolate this normal form and its continuity properties.

Definition 2.19 (Finite-code averaging operator). Let $m \in \mathbb { N }$ . Let

$$
R : \mathbb { R } ^ { d _ { a } } \times \mathbb { R } ^ { d } \longrightarrow \mathbb { R } ^ { m } , \qquad Q : \mathbb { R } ^ { m } \times \mathbb { R } ^ { d } \longrightarrow \mathbb { R } ^ { d _ { u } }\tag{33}
$$

be finite-dimensional affine-output neural networks with activation σ. For $a \in L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { a } } )$ , define

$$
\mathbf { c } _ { R } ( \mathfrak { a } ) : = \mathrm { A v g } _ { \Omega } \left( R ( \mathfrak { a } ( \cdot ) , \cdot ) \right) \in \mathbb { R } ^ { m }\tag{34}
$$

and

$$
\begin{array} { r } { \mathcal { T } _ { R , Q } ( \mathfrak { a } ) ( x ) : = Q ( \mathbf { c } _ { R } ( \mathfrak { a } ) , x ) , \qquad x \in \Omega . } \end{array}\tag{35}
$$

Lemma 2.20 (Continuity and compactness of the finite code). Let R and Q be as in Definition 2.19. There are constants $L _ { R } , C _ { R } \ge 0$ such that

$$
| R ( z , x ) - R ( z ^ { \prime } , x ) | \leq L _ { R } | z - z ^ { \prime } | , \qquad | R ( 0 , x ) | \leq C _ { R }\tag{36}
$$

for all $z , z ^ { \prime } \in \mathbb { R } ^ { d _ { a } }$ and $x \in { \overline { { \Omega } } } .$ Consequently, $\mathbf { c } _ { R } : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { a } } ) \to \mathbb { R } ^ { m }$ is well defined and satisfies

$$
| \mathbf { c } _ { R } ( a ) - \mathbf { c } _ { R } ( b ) | \leq L _ { R } | \Omega | ^ { - 1 / 2 } \| a - b \| _ { L ^ { 2 } } ,\tag{37}
$$

$$
\begin{array} { r } { | \mathbf { c } _ { R } ( { a } ) | \leq C _ { R } + L _ { R } | \Omega | ^ { - 1 / 2 } \| { a } \| _ { L ^ { 2 } } . } \end{array}\tag{38}
$$

The following set ${ \mathfrak { C } } _ { R } ( K ) : = \{ \mathbf { c } _ { R } ( a ) : a \in K \} \subset \mathbb { R } ^ { m }$ is compact.

For every $r \in  { \mathbb { N } } _ { 0 }$ , the map

$$
\mathcal { T } _ { R , Q } : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { a } } ) \longrightarrow C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { u } } )
$$

is continuous.

Proof. A finite neural network built from affine maps and the globally Lipschitz activation σ is globally Lipschitz. This gives the first estimate in (36); the second follows from continuity of $x \mapsto R ( 0 , x )$ on the compact set Ω. Hence

$$
| R ( a ( x ) , x ) | \leq C _ { R } + L _ { R } | a ( x ) |
$$

for almost every x, so the average in (34) is well defined. The estimates (37) and (38) follow from Cauchy–Schwarz. Compactness of ${ \mathfrak { C } } _ { R } ( K )$ follows because it is the continuous image of $K$

Since $\sigma \in C ^ { \infty } ( \mathbb { R } )$ , the finite-dimensional network Q is smooth. For each multi-index $\alpha ,$ , the function

$$
( c , x ) \longmapsto \partial _ { x } ^ { \alpha } Q ( c , x )
$$

is continuous. Uniform continuity on compact subsets of $\mathbb { R } ^ { m } \times \overline { { \Omega } }$ shows that $c \mapsto Q ( c , \cdot )$ is continuous from $\mathbb { R } ^ { m }$ into $C ^ { r }$ . The last assertion is the composition of this map with the continuous code map $\mathbf { c } _ { R }$ □

The following proposition records the precise external consequence of the averaging-neural-operator theorem that is used later.

Proposition 2.21. Suppose that Assumption 2.13 holds, and let $s , r \in  { \mathbb { N } } _ { 0 }$ . Let ${ \mathcal { G } } : X \to Y$ be continuous and let $K \bar { C } X$ be compact, in either ofthefollowing settings:

(i) $C ^ { s }$ setting: $X = C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { a } } )$ and $Y = C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { u } } )$

(ii) Sobolev setting: $X = H ^ { s } ( \Omega ; \mathbb { R } ^ { d _ { a } } )$ and $Y = H ^ { r } ( \Omega ; \mathbb { R } ^ { d _ { u } } )$ , with the additional assumption

$$
\operatorname* { s u p } _ { a \in K } \| a \| _ { L ^ { \infty } } < \infty .
$$

Then, for every $\varepsilon > 0 ,$ , there exist $m \in \mathbb { N }$ and finite neural networks $R , Q$ as in (33) such that

$$
\operatorname* { s u p } _ { a \in K } \| \mathcal { G } ( a ) - \mathcal { T } _ { R , Q } ( a ) \| _ { Y } < \varepsilon .
$$

Proof. The corresponding ANO universality results are proved in [23, Theorems 1 and 2]. The proof of Lemma A.9 and Remark A.10 in [23] gives the more precise normal form above. Its encoder constructs a local neural feature map $R ( a ( x ) , x )$ and uses a single average over Ω to obtain a finite vector. Because the average is normalized, a constant hidden bias may be absorbed into the affine output of R; the hidden componentwise activation and all subsequent finite-dimensional processing may then be composed into the local decoder $Q ( c , x )$ . Thus the resulting operator has the form (35) and uses one evaluation of the averaging operation. The two norm estimates are the $C ^ { s }  C ^ { s ^ { \prime } }$ theorem and the $H ^ { s }  H ^ { s ^ { \prime } }$ specialization of the Sobolev theorem in that reference. □

## 2.7.2 Local chirp encoding

Fix for the remainder of this section a free symplectic matrix $S ,$ and the chirp $\gamma _ { S }$ from (5). We use the positional vector $p _ { S }$ from (14).

Lemma 2.22. Let $M > 0 ,$ , let $n \in \mathbb N ,$ , and let $R : \mathbb { R } ^ { d _ { a } } \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { m }$ be a neural network with activation σ. There exist a vector $a _ { R } \in \mathbb { R } ^ { 2 m }$ and a number $\tau _ { R } > 0$ , independent of the accuracy below, such thatfor every $\eta > 0$ there is a activated-output network

$$
\Phi _ { R , \eta } ^ { \mathrm { e n c } } : \mathbb { R } ^ { d _ { a } + d + 2 } \longrightarrow \mathbb { R } ^ { 2 m }
$$

satisfying

$$
\operatorname* { s u p } _ { | z | \leq M } \left| \Phi _ { R , \eta } ^ { \mathrm { e n c } } ( z , p _ { S } ( x ) ) - a _ { R } - \tau _ { R } \mathfrak { r } _ { m } \big ( \overline { { \gamma _ { S } ( x ) } } R ( z , x ) \big ) \right| < \eta .\tag{39}
$$

Its pointwise realization

$$
\mathcal { R } _ { R , \eta } ( a ) ( x ) : = \Phi _ { R , \eta } ^ { \mathrm { e n c } } ( a ( x ) , p _ { S } ( x ) )
$$

is an admissible local lifting map of the form (15). $I f a \in L ^ { \infty }$ and $\| a \| _ { L ^ { \infty } } \leq M$ , then, with

$$
c _ { R } ^ { 0 } : = \mathfrak { r } _ { m } ^ { - 1 } ( a _ { R } ) \in \mathbb { C } ^ { m } ,
$$

one has

$$
\begin{array} { r } { \left\| \mathfrak { r } _ { m } ^ { - 1 } \mathcal { R } _ { R , \eta } ( a ) - c _ { R } ^ { 0 } - \tau _ { R } \overline { { \gamma _ { S } } } R ( a ( \cdot ) , \cdot ) \right\| _ { L ^ { 2 } ( \Omega ; \mathbb { C } ^ { m } ) } < | \Omega | ^ { 1 / 2 } \eta . } \end{array}
$$

Proof. Consider

$$
\mathfrak { B } _ { M , S } : = \{ ( z , p _ { S } ( x ) ) : | z | \leq M , \ x \in \overline { { \Omega } } \} \subset \mathbb { R } ^ { d _ { a } + d + 2 } .
$$

This set is compact, since $\{ ( z , x ) : | z | \leq M , \ x \in { \overline { { \Omega } } } \}$ is compact and $( z , x ) \mapsto ( z , p _ { S } ( x ) )$ is continuous.

$\Re _ { M , S }$ , define

$$
F _ { R , S } ( z , p _ { S } ( x ) ) : = \mathfrak { r } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } R ( z , x ) \bigr ) .
$$

This is well defined because the first d components of $p _ { S } ( x )$ are the coordinates of $x ,$ and it is continuous since R is continuous, $\gamma _ { S }$ is smooth, and ${ \mathfrak { r } } _ { m }$ is linear.

Applying Lemma 2.16 to $\Re _ { M , S }$ and $F _ { R , S }$ gives fixed $a _ { R } \in \mathbb { R } ^ { 2 m }$ and $\tau _ { R } > 0$ , independent of η, and, for every $\eta > 0$ , an activated-output network $\Phi _ { R , \eta } ^ { \mathrm { e n c } }$ satisfying (39).

Define

$$
\begin{array} { r } { \mathcal { R } _ { R , \eta } ( a ) ( x ) : = \Phi _ { R , \eta } ^ { \mathrm { e n c } } \big ( a ( x ) , p _ { S } ( x ) \big ) . } \end{array}
$$

This is a local lifting map of the form (15).

For a as in the statement, evaluate (39) at $z = a ( x )$ . Since ${ \mathfrak { r } } _ { m }$ is an isometry and

$$
c _ { R } ^ { 0 } : = \mathfrak { r } _ { m } ^ { - 1 } ( a _ { R } ) ,
$$

we obtain, for almost every $x \in \Omega$

$$
\left| \mathfrak { r } _ { m } ^ { - 1 } \mathcal { R } _ { R , \eta } ( a ) ( x ) - c _ { R } ^ { 0 } - \tau _ { R } \overline { { \gamma _ { S } ( x ) } } R ( a ( x ) , x ) \right| < \eta .
$$

Integrating this pointwise estimate yields

as claimed.

$$
\begin{array} { r } { \big \| \mathfrak { r } _ { m } ^ { - 1 } \mathcal { R } _ { R , \eta } ( a ) - c _ { R } ^ { 0 } - \tau _ { R } \overline { { \gamma _ { S } } } R ( a ( \cdot ) , \cdot ) \big \| _ { L ^ { 2 } } < | \Omega | ^ { 1 / 2 } \eta , } \end{array}
$$

## 2.7.3 One metaplectic branch produces the finite code

We first record explicitly that the constant vector $c _ { R } ^ { 0 }$ in Lemma 2.22 can be canceled by the declared finite-dimensional bias class.

Lemma 2.23. Let $d _ { 0 } = u _ { 0 } + i v _ { 0 } \in \mathbb { C } ^ { m }$ and write

$$
\varphi _ { S } ( x ) : = \pi x \cdot Q _ { S } x .
$$

Then

$$
\mathfrak { r } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } d _ { 0 } \bigr ) = \begin{array} { l } { \left( u _ { 0 } \atop v _ { 0 } \right) \cos \varphi _ { S } ( x ) + \left( \begin{array} { l } { v _ { 0 } } \\ { - u _ { 0 } } \end{array} \right) \sin \varphi _ { S } ( x ) . } \end{array}\tag{40}
$$

Consequently, the map $x \mapsto \mathfrak { r } _ { m } ( \overline { { \gamma _ { S } ( x ) } } d _ { 0 } )$ belongs to the bias class in Definition 2.11 with positional vector $p _ { S }$

Proof. The identity follows by multiplying $u _ { 0 } + i v _ { 0 } \ b y \ e ^ { - i \varphi _ { S } ( x ) }$ . The last two entries of $p _ { S }$ are cos $\varphi _ { S }$ and sin $\varphi _ { S } , \mathbf { s o }$ the right-hand side of (40) is obtained by a fixed $2 m \times ( d + 2 )$ coefficient matrix whose first d columns are zero. □

Lemma 2.24 (Uniform activation interval for a compact code set). Let $I _ { \sigma }$ be the interval from Lemma 2.14 and $C \subset \mathbb { R } ^ { m }$ be compact. There exist compact intervals

$$
I _ { 0 } \Subset I _ { 1 } \Subset I _ { \sigma } , \qquad t _ { * } \in \mathrm { i n t } ( I _ { 0 } ) ,
$$

and a number $\lambda > 0$ such that

$$
t _ { * } \mathbf { 1 } _ { 2 m } + \lambda \mathfrak { r } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } c \bigr ) \in I _ { 0 } ^ { 2 m }\tag{41}
$$

for every $c \in C$ and every $x \in { \overline { { \Omega } } } .$ If

$$
J _ { j } : = \sigma ( I _ { j } ) , \qquad j \in \{ 0 , 1 \} ,
$$

then $J _ { 0 } \Subset J _ { 1 } \Subset J _ { \sigma }$ and the inverse $\varsigma : = ( \sigma | _ { I _ { \sigma } } ) ^ { - 1 }$ is smooth on an open neighborhood of $J _ { 1 }$

Proof. Choose compact intervals $I _ { 0 } \Subset I _ { 1 } \Subset I _ { \sigma }$ containing $t _ { * }$ in their interiors. Since $C$ is compact, there exists $M _ { C }$ with $| c | \le M _ { C }$ for $c \in C$ . Every coordinate of $\mathfrak { r } _ { m } ( \overline { { \gamma _ { S } ( x ) } } c )$ has absolute value at most $M _ { C }$ . Taking $\lambda > 0$ sufficiently small gives (41). Since $\sigma | _ { I _ { \sigma } }$ is a diffeomorphism, it maps compactly contained intervals to compactly contained intervals, and its inverse is smooth on $J _ { \sigma }$ . □

We will later compare two functions $u , v \in C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { n } )$ before applying the activation. We therefore need to control the difference $\Sigma _ { n } ( u ) - \Sigma _ { n } ( v )$ in the $C ^ { r }$ norm in terms of $u - v$

We record the corresponding estimate in the following lemma.

Lemma 2.25. Let $r , n \in  { \mathbb { N } } _ { 0 }$ with $n \geq 1$ , let $I \subset \mathbb { R }$ be compact, and let $M > 0$ . There exists a constant $C = C ( r , n , I , M , \sigma )$ such that

$$
\Vert \Sigma _ { n } ( u ) - \Sigma _ { n } ( v ) \Vert _ { C ^ { r } } \leq C \Vert u - v \Vert _ { C ^ { r } }
$$

whenever $u , v \in C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { n } )$ satisfy

$$
u ( x ) , v ( x ) \in I ^ { n } , \qquad \| u \| _ { C ^ { r } } + \| v \| _ { C ^ { r } } \leq M .
$$

Proof. Write $u \ = \ ( u _ { 1 } , \ldots , u _ { n } )$ and $v ~ = ~ ( v _ { 1 } , \ldots , v _ { n } )$ . Since $\Sigma _ { n }$ acts componentwise, global Lipschitz continuity of σ gives

$$
\begin{array} { r } { \Vert \Sigma _ { n } ( u ) - \Sigma _ { n } ( v ) \Vert _ { C ^ { 0 } } \leq L _ { \sigma } \Vert u - v \Vert _ { C ^ { 0 } } , } \end{array}
$$

which proves the case $r = 0$

Let $r \geq 1$ and fix a component ℓ. For $1 \leq | { \boldsymbol { \alpha } } | \leq r .$ , the chain rule expresses $\partial ^ { \alpha } ( \sigma \circ u _ { \ell } )$ as a finite linear combination of terms

$$
\sigma ^ { ( k ) } ( u _ { \ell } ) \prod _ { j = 1 } ^ { k } \partial ^ { \beta _ { j } } u _ { \ell } , \qquad | \beta _ { j } | \geq 1 , \qquad \sum _ { j = 1 } ^ { k } \beta _ { j } = \alpha .
$$

Compare each term with its counterpart for $v _ { \ell }$

By changing one factor at a time, using $a b - c d = ( a - c ) b + c ( b - d )$ , their difference becomes a sum of terms containing either $\partial ^ { \beta } ( u _ { \ell } - v _ { \ell } )$ or $\sigma ^ { ( k ) } ( u _ { \ell } ) - \sigma ^ { ( k ) } ( v _ { \ell } )$

Let $\widetilde { I }$ be the smallest closed interval containing I. The mean-value theorem gives

$$
\vert \sigma ^ { ( k ) } ( u _ { \ell } ) - \sigma ^ { ( k ) } ( v _ { \ell } ) \vert \leq \operatorname* { s u p } _ { t \in \widetilde { I } } \vert \sigma ^ { ( k + 1 ) } ( t ) \vert \vert u _ { \ell } - v _ { \ell } \vert .
$$

Moreover,

$$
| \partial ^ { \boldsymbol { \beta } } ( u _ { \ell } - v _ { \ell } ) | \leq \| u _ { \ell } - v _ { \ell } \| _ { C ^ { r } } , \qquad | \boldsymbol { \beta } | \leq r .
$$

All remaining factors are uniformly bounded by the $C ^ { r }$ bound $M$ and the bounds on the derivatives of $\sigma$ on ${ \widetilde { I } } .$ . Summing the terms yields

$$
\| \sigma \circ u _ { \ell } - \sigma \circ v _ { \ell } \| _ { C ^ { r } } \leq C _ { 0 } \| u _ { \ell } - v _ { \ell } \| _ { C ^ { r } } , \qquad C _ { 0 } = C _ { 0 } ( r , I , M , \sigma ) .
$$

Combining the n component estimates gives

$$
\Vert \Sigma _ { n } ( u ) - \Sigma _ { n } ( v ) \Vert _ { C ^ { r } } \leq \sqrt { n } C _ { 0 } \Vert u - v \Vert _ { C ^ { r } } .
$$

Thus the lemma holds with $C = { \sqrt { n } } C _ { 0 }$

## 2.7.4 Local demodulation and decoding

Later, the output will be obtained by applying a finite-dimensional map $F$ pointwise to a function $h ,$ namely,

$$
x \longmapsto F ( h ( x ) , x ) .
$$

We need two facts about this substitution. First, if h is replaced by a nearby function ${ \widetilde { h } } ,$ then the corresponding output should remain close in the $C ^ { r }$ norm. Second, if F is approximated by another map ${ \bar { G } } ,$ , then this approximation should remain valid after substituting $h ( x )$ . The next lemma provides both estimates uniformly for h near a compact family of functions.

Lemma 2.26. Let $r \in  { \mathbb { N } } _ { 0 }$ and $d , n , p \in \mathbb { N } .$ . Let $U \subset \mathbb { R } ^ { n }$ and $V \subset \mathbb { R } ^ { d }$ be open with ${ \overline { { \Omega } } } \subset V .$ . Let $H \subset C ^ { r } ( { \overline { { \Omega } } } ; \mathbb { R } ^ { n } )$ be compact and suppose that

$$
\{ h ( x ) : h \in H , x \in { \overline { { \Omega } } } \} \Subset U .
$$

If $F \in C ^ { r + 1 } ( U \times V ; \mathbb { R } ^ { p } )$ , then there are a bounded open neighborhood $\boldsymbol { \mathcal { U } } _ { H }$ of H in $C ^ { r }$ and a constant $C _ { F } > 0$ such that

$$
\| F ( h ( \cdot ) , \cdot ) - F ( \widetilde { h } ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq C _ { F } \| h - \widetilde { h } \| _ { C ^ { r } }\tag{42}
$$

for all $h , \widetilde { h } \in \mathcal { U } _ { H }$

The neighborhood $\boldsymbol { \mathcal { U } } _ { H }$ can be chosen so that there exists a compact set $K _ { 0 } \Subset U \times V$ containing

$$
\{ ( h ( x ) , x ) : h \in \mathcal { U } _ { H } , x \in \overline { { \Omega } } \} .\tag{43}
$$

For this choice of $\mathcal { U } _ { H }$ and $K _ { 0 } ,$ , there exists a constant $C _ { H } > 0$ such that, for every $G \in C ^ { r } ( U \times $ $V ; \mathbb { R } ^ { p } )$

$$
\operatorname* { s u p } _ { h \in \mathcal { U } _ { H } } \| F ( h ( \cdot ) , \cdot ) - G ( h ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq C _ { H } \| F - G \| _ { C ^ { r } ( K _ { 0 } ) } .\tag{44}
$$

Proof. The set

$$
K _ { H } : = \{ h ( x ) : h \in H ,  x \in \overline { { \Omega } } \}
$$

is compact by continuity of evaluation. Choose $\rho > 0$ such that

$$
K _ { U } : = \{ z : \mathrm { d i s t } ( z , K _ { H } ) \leq \rho \} \Subset U ,
$$

and define

$$
\mathcal { U } _ { H } : = \{ h \in C ^ { r } : \mathrm { d i s t } _ { C ^ { r } } ( h , H ) < \rho / 4 \} .
$$

Then $\boldsymbol { \mathcal { U } } _ { H }$ is a bounded open neighborhood of $H ,$ and its functions take values in $K _ { U }$ . Consequently,

$$
K _ { 0 } : = K _ { U } \times \overline { { \Omega } } \Subset U \times V
$$

satisfies (43), and $\mathrm { s u p } _ { h \in \mathcal { U } _ { H } } \| h \| _ { C ^ { r } } < \infty$

Now want to show the stability under perturbations of $h .$

Suppose first that $\| h - \widetilde { h } \| _ { C ^ { r } } < \rho / 2$ . For $h _ { t } : = ( 1 - t ) h + t \widetilde { h } .$ , we have dist $( h _ { t } ( x ) , K _ { H } ) < 3 \rho / 4$ hence $h _ { t } ( x ) \in K _ { U }$ . The fundamental theorem of calculus gives

$$
F ( h ( x ) , x ) - F ( \widetilde { h } ( x ) , x ) = \int _ { 0 } ^ { 1 } D _ { z } F ( h _ { t } ( x ) , x ) \big ( h ( x ) - \widetilde { h } ( x ) \big ) d t .
$$

Since the $h _ { t }$ are uniformly bounded in $C ^ { r }$ and $F \in C ^ { r + 1 }$ , the chain and product rules bound the $C ^ { r }$ norm of this expression by $C _ { 1 } \| h - \widetilde { h } \| _ { C ^ { r } }$

The chain rule also gives a uniform bound $\| F ( h ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq B$ on $\boldsymbol { \mathcal { U } } _ { H }$ . For the remaining pairs,

$$
\| F ( h ( \cdot ) , \cdot ) - F ( \widetilde h ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq 2 B \leq \frac { 4 B } { \rho } \| h - \widetilde h \| _ { C ^ { r } } .
$$

Thus (42) holds with $C _ { F } : = \operatorname* { m a x } \{ C _ { 1 } , 4 B / \rho \}$

Now we want an approximation of the pointwise map. Apply the chain rule to $( F - G ) ( h ( x ) , x )$ Each term contains a derivative of $F - G$ of order at most r, evaluated in $K _ { 0 }$ , multiplied by derivatives of $h .$ The uniform $C ^ { r }$ bound on $\boldsymbol { \mathcal { U } } _ { H }$ therefore yields

$$
\operatorname* { s u p } _ { h \in \mathcal { U } _ { H } } \| F ( h ( \cdot ) , \cdot ) - G ( h ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq C _ { H } \| F - G \| _ { C ^ { r } ( K _ { 0 } ) } ,
$$

where $C _ { H }$ is independent of G. This proves (44).

We conclude by recording the two fixed linear maps used to recover a real averaged vector from its chirp-modulated representation. Write $\gamma _ { S } ( x ) = e ^ { i \varphi _ { S } ( x ) }$ . Multiplication by $\gamma _ { S } ( x )$ on $\mathbb { C } ^ { m }$ is represented, after realification, by the real rotation

$$
\mathcal { U } _ { S } ( x ) : = \binom { \cos \varphi _ { S } ( x ) I _ { m } } { \sin \varphi _ { S } ( x ) I _ { m } } \quad \sp { - \sin \varphi _ { S } ( x ) I _ { m } } \bigg ) \in \mathbb { R } ^ { 2 m \times 2 m } .
$$

Thus, for every $z \in \mathbb { C } ^ { m }$

$$
\begin{array} { r } { \mathcal { U } _ { S } ( x ) \mathfrak { r } _ { m } ( z ) = \mathfrak { r } _ { m } \bigl ( \gamma _ { S } ( x ) z \bigr ) . } \end{array}
$$

We also define the fixed real-linear map

$$
\Pi _ { \mathrm { r e } } : \mathbb { R } ^ { 2 m } \longrightarrow \mathbb { R } ^ { m } , \qquad \Pi _ { \mathrm { r e } } \binom { u } { v } : = u .
$$

The map $\Pi _ { \mathrm { r e } }$ is applied only after the chirp modulation has been removed. Indeed, if $c \in \mathbb { R } ^ { m }$ , then

$$
\Pi _ { \mathrm { r e } } \mathcal { U } _ { S } ( x ) \mathfrak { r } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } c \bigr ) = \Pi _ { \mathrm { r e } } \mathfrak { r } _ { m } ( c ) = c .
$$

Thus, the rotation removes the factor $\overline { { \gamma _ { S } ( x ) } }$ , after which $\Pi _ { \mathrm { r e } }$ recovers the original real vector. Both maps are fixed and introduce no trainable parameters.

## 3 Universal approximation by metaplectic neural operators

We formulate the main approximation theorem in two function-space settings. We retain the nonempty bounded Lipschitz domain $\Omega \subset \mathbb { R } ^ { d }$ and the channel dimensions $d _ { a } , d _ { u } ^ { \bar { \mathbf { \alpha } } } \in$ N fixed in Section $^ { 2 }$ The integers $s , r \in  { \mathbb { N } } _ { 0 }$ denote the input and output regularity orders. For a continuous operator $\mathcal { G } : X \xrightarrow { } Y$ and a compact set $K \subset X$ , we consider the following settings:

(i) $C ^ { s }$ setting: $X = C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { a } } )$ and $Y = C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { u } } )$

(ii) Sobolev setting: $X = H ^ { s } ( \Omega ; \mathbb { R } ^ { d _ { a } } )$ and $Y = H ^ { r } ( \Omega ; \mathbb { R } ^ { d _ { u } } )$ , with the additional assumption

$$
\operatorname* { s u p } _ { a \in K } \| a \| _ { L ^ { \infty } ( \Omega ) } < \infty .
$$

In setting (i), compact subsets of $X$ are automatically bounded in $L ^ { \infty }$ . In setting (ii), this bound holds automatically when $s > d / 2$ , by the Sobolev embedding on Lipschitz domains [6]. For $s \leq d / 2$ , it remains an additional assumption, which also appears in the Sobolev theorem for averaging neural operators [23].

To handle both output norms together, for each $\varepsilon > 0$ we set

$$
C _ { Y } : = \left\{ \begin{array} { l l } { 1 , } & { Y = C ^ { r } , } \\ { \lvert \Omega \rvert ^ { 1 / 2 } \binom { d + r } { r } ^ { 1 / 2 } , } & { Y = H ^ { r } , } \end{array} \right. \qquad \delta : = \varepsilon / C _ { Y } .\tag{45}
$$

Then, for every $w \in C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { d _ { u } } )$

$$
\| w \| _ { Y } \leq C _ { Y } \| w \| _ { C ^ { r } } .
$$

The theorem holds for every fixed free symplectic matrix $S .$ . Its approximants are built from the transform ${ \widehat { S } } ,$ the atom $\rho _ { \widehat { S } , \Omega }$ in (23) and the positional vector $p _ { S }$ in (14). These are fixed before G, K and the accuracy ε are chosen, whereas the number of channels, the coefficients and the pointwise networks may depend on these data.

Theorem 3.1. Suppose that Assumption 2.13 holds, andfix afree symplectic matrix $S .$

Let ${ \mathcal { G } } : X \to Y$ be continuous and let $K \subset X$ be compact, with X, Y, and K satisfying the assumptions of either setting (i) or (ii) above. Then, for every $\varepsilon > 0 ,$ , there exists a metaplectic neural operator $\mathcal { N }$ such that

$$
\operatorname* { s u p } _ { a \in K } \| { \mathcal { G } } ( a ) - { \mathcal { N } } ( a ) \| _ { Y } < \varepsilon .
$$

Proof. Let $x \in \Omega$ . Fix $\varepsilon > 0 ,$ , and let $C _ { Y }$ and $\delta$ be as in (45). By proposition 2.21, there exist $m \in \mathbb { N }$ and maps R and $Q .$ , with the domains specified in (33), such that the finite averaged-code operator

$$
\mathcal { T } _ { R , Q } ( a ) ( x ) = Q ( \mathbf { c } _ { R } ( a ) , x ) ,
$$

Such that $\mathbf { c } _ { R } ( a ) = \mathrm { A v g } _ { \Omega } R ( a ( \cdot ) , \cdot )$ , defined in (35), satisfies

$$
\operatorname* { s u p } _ { a \in K } \| \mathcal { G } ( a ) - \mathcal { T } _ { R , Q } ( a ) \| _ { Y } < \frac { \varepsilon } { 2 } .\tag{46}
$$

We now construct a metaplectic neural operator that approximates $\mathcal { T } _ { R , Q }$ on $K$ . More precisely, we will show that

$$
\operatorname* { s u p } _ { a \in K } \| \mathcal { T } _ { R , Q } ( a ) - \mathcal { N } ( a ) \| _ { C ^ { r } } < \frac { \delta } { 2 } .\tag{47}
$$

For $a \in K$ , introduce the following function

$$
v _ { a } ( y ) : = \overline { { \gamma _ { S } ( y ) } } R ( a ( y ) , y ) , \qquad y \in \Omega .
$$

The uniform $L ^ { \infty }$ bound on $K$ and the continuity of R ensure that $v _ { a } \in L ^ { 2 } ( \Omega ; \mathbb { C } ^ { m } )$ . By Proposition 2.17 we get

$$
\left( { \cal K } _ { \widehat { S } , \rho _ { \widehat { S } , \Omega } { I _ { m } } } ^ { \Omega } v _ { a } \right) ( x ) = \overline { { { \gamma } _ { S } ( x ) } } { \bf c } _ { R } ( a ) .\tag{48}
$$

To approximate $v _ { a }$ by an admissible local lifting we apply Lemma 2.22. which provides constants $c _ { R } ^ { 0 } \in \mathbb { C } ^ { m }$ and $\tau _ { R } > 0$ and, for every $\eta > 0$ , an activated-output neural network $\Phi _ { R , \eta } ^ { \mathrm { e n c } }$ such that

$$
\begin{array} { r } { \mathcal { R } _ { R , \eta } ( a ) ( x ) : = \Phi _ { R , \eta } ^ { \mathrm { e n c } } \bigl ( a ( x ) , p _ { S } ( x ) \bigr ) \in \mathbb { R } ^ { 2 m } . } \end{array}
$$

Here $p _ { S }$ is the positional vector from (14), and $c _ { R } ^ { 0 }$ and $\tau _ { R }$ are independent of η. Set

$$
e _ { R , \eta } ( a ) : = \mathfrak { r } _ { m } ^ { - 1 } \mathscr { R } _ { R , \eta } ( a ) - c _ { R } ^ { 0 } - \tau _ { R } v _ { a } ,\tag{49}
$$

where ${ \mathfrak { r } } _ { m }$ is realification function. Then, Lemma 2.22 gives the uniform estimate

$$
\displaystyle \operatorname* { s u p } _ { a \in K } \| e _ { R , \eta } ( a ) \| _ { L ^ { 2 } } < | \Omega | ^ { 1 / 2 } \eta .\tag{50}
$$

The parameter η will be chosen later.

By the compactness of $\mathcal { C } _ { R , K }$ and Lemma 2.24, we may fix intervals $I _ { 0 } \Subset I _ { 1 } \Subset I _ { \sigma }$ , a point $t _ { * } \in \mathrm { i n t } ( I _ { 0 } )$ , and a number $\lambda > 0$ such that

$$
t _ { * } \mathbf { 1 } _ { 2 m } + \lambda \mathfrak { t } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } c \bigr ) \in I _ { 0 } ^ { 2 m } \qquad ( c \in \mathcal { C } _ { R , K } , \ x \in \overline { { \Omega } } ) .\tag{51}
$$

Set $\alpha : = \lambda / \tau _ { R }$ and choose the matrix-valued symbol

$$
P _ { 1 } ( \xi ) : = \alpha \rho _ { \widehat { S } , \Omega } ( \xi ) I _ { m } , \qquad \xi \in \mathbb { R } ^ { d } .
$$

Identifying $c _ { R } ^ { 0 }$ with the corresponding constant function on $\Omega$ , linearity and (49) give

$$
\begin{array} { r } { \mathcal { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } \left( \mathfrak { r } _ { m } ^ { - 1 } \mathcal { R } _ { R , \eta } ( a ) \right) = \mathcal { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } c _ { R } ^ { 0 } + \tau _ { R } \mathcal { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } v _ { a } + \mathcal { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } e _ { R , \eta } ( a ) . } \end{array}
$$

Proposition 2.17 and (48) identify the first two terms explicitly:

$$
\begin{array} { r l } & { \bigl ( \mathcal { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } c _ { R } ^ { 0 } \bigr ) ( x ) = \overline { { \gamma _ { S } ( x ) } } d _ { R } ^ { 0 } , } \\ & { \tau _ { R } \bigl ( \mathcal { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } v _ { a } \bigr ) ( x ) = \lambda \overline { { \gamma _ { S } ( x ) } } \mathbf { c } _ { R } ( a ) , } \end{array}
$$

where $d _ { R } ^ { 0 } : = \alpha c _ { R } ^ { 0 } \operatorname { A v g } _ { \Omega } ( \gamma _ { S } )$ . The first term is independent of $^ { a , }$ so it can be canceled by the spatial bias

$$
b _ { 1 } ( x ) : = t _ { * } { \bf 1 } _ { 2 m } - \mathfrak { r } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } d _ { R } ^ { 0 } \bigr ) .
$$

By Lemma 2.23, this bias belongs to the class in Definition 2.11. In particular, there exist $c _ { 1 } \in \mathbb { R } ^ { 2 m }$ and $G _ { 1 } \in \mathbb { R } ^ { 2 m \times ( d + 2 ) }$ such that

$$
b _ { 1 } ( x ) = c _ { 1 } + G _ { 1 } p _ { S } ( x ) .
$$

Taking $W _ { 1 } = 0$ , define the admissible hidden layer

$$
\begin{array} { r } { \mathcal { L } _ { 1 } ( w ) ( x ) : = \Sigma _ { 2 m } \Big ( b _ { 1 } ( x ) + \big ( \mathbf { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } w \big ) ( x ) \Big ) . } \end{array}
$$

The resulting preactivation is given by

$$
u _ { R , \eta } ( a ) ( x ) : = b _ { 1 } ( x ) + \bigl ( \mathbf { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } \mathcal { R } _ { R , \eta } ( a ) \bigr ) ( x ) .\tag{52}
$$

Also let

$$
u _ { R } ( a ) ( x ) : = t _ { * } \mathbf { 1 } _ { 2 m } + \lambda \mathfrak { r } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } \mathfrak { c } _ { R } ( a ) \bigr ) .\tag{53}
$$

Then following estimate holds:

$$
\operatorname* { s u p } _ { a \in K } \| u _ { R , \eta } ( a ) - u _ { R } ( a ) \| _ { C ^ { r } } < C _ { \mathrm { p r e } } \eta ,
$$

where $C _ { \mathrm { p r e } } > 0$ is independent of $a$ and $\eta$ . Moreover for $\eta$ sufficiently small, we get $u _ { R , \eta } ( a )$ takes values in $I _ { 1 } ^ { 2 m }$

Indeed, using the definition of $b _ { 1 } .$ , we obtain

$$
\begin{array} { r l } & { u _ { R , \eta } ( a ) ( x ) = \delta _ { 1 , ( } x ) + \big ( \mathbf { K } _ { S , p } ^ { \mathbb { G } } , \mathcal { R } _ { R , p } ( a ) \big ) ( x ) } \\ & { \qquad = b _ { 1 } ( x ) + \mathbf { r } _ { m } ( \Big ( \mathcal { K } _ { S , p _ { 1 } } ^ { \mathbb { G } } , \Gamma _ { m , p } ( x ) \Big ) ( x ) ) } \\ & { \qquad = t _ { * } \mathbf { 1 } _ { 2 m } - \mathbf { r } _ { m } ( \widetilde { \gamma } _ { S } ( x ) ) d _ { k } ^ { \mathbb { G } } ) } \\ & { \qquad + \mathbf { r } _ { m } ( \widetilde { \gamma } _ { S } ( x ) d _ { \mathbb { H } } ^ { \mathbb { G } } + \lambda \widetilde { \gamma } _ { S } ( x ) \ c _ { R } ( a ) + \big ( \mathcal { K } _ { S , p _ { 1 } } ^ { \mathbb { G } } e _ { R , \eta } ( a ) \big ) ( x ) ) } \\ & { \qquad = t _ { * } \mathbf { 1 } _ { 2 m } + \underbrace { - \mathbf { r } _ { m } ( \overline { { \gamma } } \otimes ( x ) d _ { R } ^ { 0 } ) } _ { \begin{array} { c } { = 0 } \\ { \forall x ( \overline { { \gamma } } \otimes ( x ) ) \in R ( a ) } \end{array} } + \mathbf { r } _ { m } ( \overline { { \gamma } } \otimes ( x ) d _ { R } ^ { 0 } ) } \\ & { \qquad + \lambda \mathbf { t } _ { m } ( \overline { { \gamma } } \otimes ( x ) c _ { R } ( a ) ) + \mathbf { r } _ { m } ( \big ( \mathcal { K } _ { S , p _ { 1 } } ^ { \mathbb { G } } e _ { R , \eta } ( a ) \big ) ( x ) ) } \\ &  \qquad = t _ { * } \mathbf { 1 } _ { 2 m } + \lambda \mathbf { r } _ { m } ( \overline { { \gamma } } \otimes ( x ) ) \mathbf { r } _ { R } ( a ) ) + \mathbf { r } _ { m } ( \big ( \mathcal { K } _  S  \end{array}
$$

Then

$$
u _ { R , \eta } ( a ) - u _ { R } ( a ) = \mathfrak { r } _ { m } \bigl ( \mathcal { K } _ { \widehat { S } , P _ { 1 } } ^ { \Omega } e _ { R , \eta } ( a ) \bigr ) .
$$

Since realification preserves the $C ^ { r }$ norm, (36) and (61) give

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { a \in K } \left\| u _ { R , \eta } ( a ) - u _ { R } ( a ) \right\| _ { C ^ { r } } = \displaystyle \operatorname* { s u p } _ { a \in K } \left\| K _ { \widehat { S } , P _ { 1 } } ^ { \Omega } e _ { R , \eta } ( a ) \right\| _ { C ^ { r } } } \\ & { \quad \quad \quad \leq C _ { r , S , \Omega } ^ { \mathbb C } \alpha | \Omega | ^ { - 1 / 2 } \displaystyle \operatorname* { s u p } _ { a \in K } \left\| e _ { R , \eta } ( a ) \right\| _ { L ^ { 2 } } } \\ & { \quad \quad \quad < C _ { r , S , \Omega } ^ { \mathbb C } \frac { \lambda } { \tau _ { R } } \eta . } \end{array}
$$

Since realification preserves the $C ^ { r }$ norm, (29) and (50) therefore imply

$$
\begin{array} { r l } & { \underset { a \in K } { \operatorname* { s u p } } \Vert u _ { R , \eta } ( a ) - u _ { R } ( a ) \Vert _ { C ^ { r } } \leq C _ { r , S , \Omega } ^ { C } \alpha \vert \Omega \vert ^ { - 1 / 2 } \underset { a \in K } { \operatorname* { s u p } } \Vert e _ { R , \eta } ( a ) \Vert _ { L ^ { 2 } } } \\ & { \qquad < C _ { r , S , \Omega } ^ { C } \frac { \lambda } { \tau _ { R } } \eta . } \end{array}
$$

Hence, with $C _ { \mathrm { p r e } } : = C _ { r , S , \Omega } ^ { C } \lambda / \tau _ { R }$ , we have

$$
\operatorname* { s u p } _ { a \in K } \| u _ { R , \eta } ( a ) - u _ { R } ( a ) \| _ { C ^ { r } } < C _ { \mathrm { p r e } } \eta .\tag{54}
$$

By (51), the preactivations in (53) take values in $I _ { 0 } ^ { 2 m }$ . Let

$$
\delta _ { I } : = \mathrm { d i s t } ( I _ { 0 } , \mathbb { R } \setminus I _ { 1 } ) > 0
$$

and choose $\eta _ { 0 } > 0$ so that $C _ { \mathrm { p r e } } \eta _ { 0 } < \delta _ { I }$ . Since

$$
\underset { a \in K } { \operatorname* { s u p } } \left. u _ { R , \eta } ( a ) - u _ { R } ( a ) \right. _ { C ^ { 0 } } \leq \underset { a \in K } { \operatorname* { s u p } } \left. u _ { R , \eta } ( a ) - u _ { R } ( a ) \right. _ { C ^ { r } } .
$$

Then (54) ensures that the preactivations in (52) take values in $I _ { 1 } ^ { 2 m }$ whenever $0 < \eta \leq \eta _ { 0 }$

Both preactivation families are also uniformly bounded in $C ^ { r }$ . Since, the compactness of $\mathcal { C } _ { R , K }$ and the smoothness of $\gamma _ { S }$ give

$$
M _ { 0 } : = \operatorname* { s u p } _ { a \in K } \| u _ { R } ( a ) \| _ { C ^ { r } } < \infty ,
$$

while (54) yields

$$
\operatorname* { s u p } _ { a \in K } \| u _ { R , \eta } ( a ) \| _ { C ^ { r } } \leq M _ { 0 } + C _ { \mathrm { p r e } } \eta _ { 0 } \qquad ( 0 < \eta \leq \eta _ { 0 } ) .
$$

We may therefore apply Lemma 2.25 on $I _ { 1 }$ with a common $C ^ { r }$ bound. Defining

$$
\begin{array} { r } { \mathcal { H } _ { 1 , \eta } ( a ) : = \mathcal { L } _ { 1 } \big ( \mathcal { R } _ { R , \eta } ( a ) \big ) , \qquad h _ { R } ( a ) : = \Sigma _ { 2 m } \big ( \boldsymbol { u } _ { R } ( a ) \big ) . } \end{array}\tag{55}
$$

we obtain a constant $C _ { \mathrm { a c t } } > 0$ , independent of a and $0 < \eta \leq \eta _ { 0 }$ , for which

$$
\Vert \mathcal { H } _ { 1 , \eta } ( a ) - h _ { R } ( a ) \Vert _ { C ^ { r } } \leq C _ { \mathrm { a c t } } \Vert u _ { R , \eta } ( a ) - u _ { R } ( a ) \Vert _ { C ^ { r } } .
$$

In particular,

$$
\operatorname* { s u p } _ { a \in K } \| \mathcal { H } _ { 1 , \eta } ( a ) - h _ { R } ( a ) \| _ { C ^ { r } } < C _ { \mathrm { h i d } } \eta , \qquad C _ { \mathrm { h i d } } : = C _ { \mathrm { a c t } } C _ { \mathrm { p r e } } .\tag{56}
$$

Writing $J _ { j } : = \sigma ( I _ { j } ) \operatorname { f o r } j = 0$ , 1 and $J _ { \sigma } : = \sigma ( I _ { \sigma } )$ , we also have $h _ { R } ( a ) ( x ) \in J _ { 0 } ^ { 2 m }$ and $\mathcal { H } _ { 1 , \eta } ( a ) ( x ) \in$ $J _ { 1 } ^ { 2 m }$ . We next construct a smooth pointwise map $\mathfrak { q } _ { R , Q }$ that recovers $\mathcal { T } _ { R , Q }$ . Let $\varsigma : = ( \sigma | _ { I _ { \sigma } } ) ^ { - 1 }$ and let $\varsigma _ { 2 m }$ act componentwise:

$$
\begin{array} { r } { \varsigma _ { 2 m } ( h _ { 1 } , \ldots , h _ { 2 m } ) : = \big ( \varsigma ( h _ { 1 } ) , \ldots , \varsigma ( h _ { 2 m } ) \big ) , \qquad h \in J _ { \sigma } ^ { 2 m } . } \end{array}
$$

Since $h _ { R } ( a ) = \Sigma _ { 2 m } ( u _ { R } ( a ) )$ and $u _ { R } ( a ) ( x ) \in I _ { 0 } ^ { 2 m } \subset I _ { \sigma } ^ { 2 m }$ , we have

$$
\begin{array} { r } { \varsigma _ { 2 m } ( h _ { R } ( a ) ( x ) ) = \varsigma _ { 2 m } \big ( \Sigma _ { 2 m } ( \boldsymbol { u } _ { R } ( a ) ( x ) ) \big ) = \boldsymbol { u } _ { R } ( a ) ( x ) . } \end{array}
$$

Substituting the definition of $u _ { R } ( a )$ gives

$$
\frac { \varsigma _ { 2 m } ( h _ { R } ( a ) ( x ) ) - t _ { * } { \bf 1 } _ { 2 m } } { \lambda } = \mathfrak { r } _ { m } \bigl ( \overline { { \gamma _ { S } ( x ) } } \mathfrak { c } _ { R } ( a ) \bigr ) .
$$

Recall that $\mathcal { U } _ { S } ( x ) \mathfrak { r } _ { m } ( z ) = \mathfrak { r } _ { m } ( \gamma _ { S } ( x ) z )$ , since $| \gamma _ { S } ( x ) | = 1$

$$
\mathcal { U } _ { S } ( x ) \frac { \varsigma _ { 2 m } ( h _ { R } ( a ) ( x ) ) - t _ { * } \mathbf { 1 } _ { 2 m } } { \lambda } = \mathfrak { r } _ { m } \big ( \gamma _ { S } ( x ) \overline { { \gamma _ { S } ( x ) } } \mathbf { c } _ { R } ( a ) \big ) = \mathfrak { r } _ { m } ( \mathbf { c } _ { R } ( a ) ) .
$$

Moreover, $\mathbf { c } _ { R } ( a ) \in \mathbb { R } ^ { m }$ , so its complexification has zero imaginary part. Thus, if $\Pi _ { \mathrm { r e } } : \mathbb { R } ^ { 2 m }  \mathbb { R } ^ { m }$ denotes projection onto the first m coordinates,

$$
\mathfrak { r } _ { m } ( \mathbf { c } _ { R } ( a ) ) = \left( { \bf c } _ { R } { \left( a \right) } \right) , \qquad \Pi _ { \mathrm { r e } } \mathfrak { r } _ { m } ( \mathbf { c } _ { R } ( a ) ) = \mathbf { c } _ { R } ( a ) .
$$

Consequently,

$$
\Pi _ { \mathrm { r e } } { \mathcal U } _ { S } ( x ) \frac { \varsigma _ { 2 m } ( h _ { R } ( a ) ( x ) ) - t _ { * } \mathbf 1 _ { 2 m } } { \lambda } = \mathbf c _ { R } ( a ) .\tag{57}
$$

For a bounded open $V \supset { \overline { { \Omega } } } .$ , define the smooth auxiliary decoder

$$
\mathfrak { q } _ { R , Q } ( h , x ) : = Q \left( \Pi _ { \mathrm { r e } } \mathcal { U } _ { S } ( x ) \frac { \varsigma _ { 2 m } ( h ) - t _ { * } \mathbf { 1 } _ { 2 m } } { \lambda } , x \right) , \qquad ( h , x ) \in J _ { \sigma } ^ { 2 m } \times V .
$$

Its smoothness follows from that of $\varsigma _ { 2 m } , \mathcal { U } _ { S }$ , and $Q .$ . Substituting (57), we obtain the exact identity

$$
\begin{array} { r } { \mathfrak { q } _ { R , Q } ( h _ { R } ( a ) ( x ) , x ) = Q ( \mathbf { c } _ { R } ( a ) , x ) = \mathcal { T } _ { R , Q } ( a ) ( x ) . } \end{array}\tag{58}
$$

It remains to control the error resulting from replacing $h _ { R } ( a )$ by $\mathcal { H } _ { 1 , \eta } ( a )$ from (55) and to approximate $\mathfrak { q } _ { R , Q }$ by a neural network. Consider

$$
\begin{array} { r } { h _ { c } ( x ) : = \Sigma _ { 2 m } \Big ( t _ { * } { \mathbf 1 } _ { 2 m } + \lambda \mathfrak { r } _ { m } ( \overline { { \gamma _ { S } ( x ) } } c ) \Big ) , \qquad c \in \mathcal { C } _ { R , K } . } \end{array}
$$

The map $c \mapsto h _ { c }$ is continuous into $C ^ { r }$ . Indeed, for every multi-index $\beta$ with $| \beta | \le r$ , the map

$$
( c , x ) \longmapsto \partial _ { x } ^ { \beta } h _ { c } ( x )
$$

is continuous on the compact set $\mathcal { C } _ { R , K } \times \overline { { \Omega } } .$ , hence uniformly continuous. Therefore, if $c _ { n } \to c$ in $\mathcal { C } _ { R , K }$

$$
\operatorname* { s u p } _ { x \in \overline { { \Omega } } } \left| \partial _ { x } ^ { \beta } h _ { c _ { n } } ( x ) - \partial _ { x } ^ { \beta } h _ { c } ( x ) \right| \longrightarrow 0 \qquad ( | \beta | \le r ) .
$$

Since only finitely many multi-indices occur, this implies $\| h _ { c _ { n } } - h _ { c } \| _ { C ^ { r } } \to 0$ . Consequently,

$$
{ \mathcal { H } } _ { R } : = \left\{ h _ { R } ( a ) : a \in K \right\} = \left\{ h _ { c } : c \in { \mathcal { C } } _ { R , K } \right\}
$$

is compact in $C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { 2 m } )$ . Moreover, $\{ h ( x ) ~ : ~ h ~ \in ~ { \mathcal { H } } _ { R } , ~ x ~ \in ~ { \overline { { \Omega } } } \} ~ \subset ~ J _ { 0 } ^ { 2 m } ~ \in ~ J _ { \sigma } ^ { 2 m } . ~ { \mathrm { A p } } { \mathrm { . } }$ ply Lemma $2 . 2 6  { \mathrm { ~ i o ~ } }  { \mathcal { H } _ { R } }$ and $\mathfrak { q } _ { R , Q }$ . It yields a bounded open neighborhood $\boldsymbol { \mathcal { U } } _ { R }$ of ${ \mathcal { H } } _ { R }$ , a compact set $K _ { R , \mathrm { d e c } } \Subset J _ { \sigma } ^ { 2 m } \times V .$ , and constants $C _ { \mathrm { s t a b } } , C _ { \mathrm { c o m p } } ~ > ~ 0$ . In particular, we have $\{ ( h ( x ) , x ) : h \in \mathcal { U } _ { R } , x \in \overline { { \Omega } } \} \subset K _ { R , \mathrm { d e c } }$ . The lemma also gives

$$
\| \mathfrak { q } _ { R , Q } ( h ( \cdot ) , \cdot ) - \mathfrak { q } _ { R , Q } ( \widetilde { h } ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq C _ { \mathrm { s t a b } } \| h - \widetilde { h } \| _ { C ^ { r } } ,\tag{59}
$$

and

$$
\| \mathfrak { q } _ { R , Q } ( h ( \cdot ) , \cdot ) - F ( h ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq C _ { \mathrm { c o m p } } \| \mathfrak { q } _ { R , Q } - F \| _ { C ^ { r } ( K _ { R , \operatorname* { d e c } } ) } .\tag{60}
$$

for $h , \widetilde { h } \in \mathcal { U } _ { R }$ and smooth $F .$ The second estimate controls all derivatives up to order $r$ after substitution. We now choose $\eta$ uniformly over $K .$ . Since ${ \mathcal { H } } _ { R }$ is compact and $\boldsymbol { \mathcal { U } } _ { R }$ is an open neighborhood of ${ \mathcal { H } } _ { R }$

$$
d _ { R } : = \mathrm { d i s t } _ { C ^ { r } } \left( \mathcal { H } _ { R } , C ^ { r } ( \overline { { \Omega } } ; \mathbb { R } ^ { 2 m } ) \backslash \mathcal { U } _ { R } \right) > 0 .
$$

Choose $0 < \eta \leq \eta _ { 0 }$ sufficiently small that

$$
C _ { \mathrm { h i d } } \eta < \operatorname* { m i n } \left\{ d _ { R } , \frac { \delta } { 4 C _ { \mathrm { s t a b } } } \right\} .
$$

This is possible because all quantities on the right-hand side are positive and independent of $\eta .$ Fix this η and write

$$
\mathcal { H } _ { 1 } : = \mathcal { H } _ { 1 , \eta } = \mathcal { L } _ { 1 } \circ \mathcal { R } _ { R , \eta } .
$$

By (56),

$$
\operatorname* { s u p } _ { a \in K } \| \mathcal { H } _ { 1 } ( a ) - h _ { R } ( a ) \| _ { C ^ { r } } < C _ { \mathrm { h i d } } \eta < d _ { R } .
$$

Since $h _ { R } ( a ) \in \mathcal { H } _ { R }$ , the definition of $d _ { R }$ implies $\mathcal { H } _ { 1 } ( a ) \in \mathcal { U } _ { R }$ for all $a \in K$

By (58), $\mathcal { T } _ { R , Q } ( a ) = \mathfrak { q } _ { R , Q } ( h _ { R } ( a ) ( \cdot ) , \cdot )$ for all $a \in K$ . Both $h _ { R } ( a )$ and $\mathcal { H } _ { 1 } ( a )$ belong to $\boldsymbol { \mathcal { U } } _ { R }$ . Thus, by Equation (59) gives

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { a \in K } \| \mathcal { T } _ { R , Q } ( a ) - \ P _ { R , Q } ( \mathcal { H } _ { 1 } ( a ) ( \cdot ) , \cdot ) \| _ { C ^ { r } } \leq C _ { \mathrm { s t a b } } \displaystyle \operatorname* { s u p } _ { a \in K } \| h _ { R } ( a ) - \mathcal { H } _ { 1 } ( a ) \| _ { C ^ { r } } } \\ { \displaystyle \qquad < C _ { \mathrm { s t a b } } C _ { \mathrm { h i d } } \eta < \displaystyle \frac { \delta } { 4 } . } \end{array}\tag{61}
$$

It remains to approximate the map $\mathfrak { q } _ { R , Q }$ by a neural network. The inclusion $\mathcal { H } _ { 1 } ( a ) \in \mathcal { U } _ { R }$ ensures that

$$
\begin{array} { r } { \big ( \mathscr { H } _ { 1 } ( a ) ( x ) , x \big ) \in K _ { R , \mathrm { d e c } } \qquad ( a \in K , x \in \overline { { \Omega } } ) . } \end{array}
$$

Since $\mathfrak { q } _ { R , Q }$ is smooth on a neighborhood of $K _ { R , { \mathrm { d e c } } } ,$ , provides a affine output

network $\Phi : \mathbb { R } ^ { 2 m + d } \longrightarrow \mathbb { R } ^ { d _ { u } }$ such that

$$
\| \mathfrak { q } _ { R , Q } - \Phi \| _ { C ^ { r } ( K _ { R , \operatorname* { d e c } } ) } < \frac { \delta } { 4 C _ { \mathrm { c o m p } } } .
$$

Define the projection as in Definition 2.15 by

$$
\mathcal { Q } ( h ) ( x ) : = \Phi ( h ( x ) , x ) .
$$

Moreover, the prescribed globally Lipschitz activation gives $\Phi$ at most linear growth. Since $\Omega$ is bounded, this defines a map

$$
\begin{array} { r } { \mathcal { Q } : L ^ { 2 } ( \Omega ; \mathbb { R } ^ { 2 m } ) \longrightarrow L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { u } } ) . } \end{array}
$$

The network $\Phi$ is smooth, so the composition estimate (60) yields

$$
\begin{array} { r } { \displaystyle \operatorname* { s u p } _ { a \in K } \| \mathfrak { q } _ { R , Q } ( \mathcal { H } _ { 1 } ( a ) ( \cdot ) , \cdot ) - Q ( \mathcal { H } _ { 1 } ( a ) ) \| _ { C ^ { r } } \leq \displaystyle \operatorname* { s u p } _ { h \in \mathcal { U } _ { R } } \| \mathfrak { q } _ { R , Q } ( h ( \cdot ) , \cdot ) - \Phi ( h ( \cdot ) , \cdot ) \| _ { C ^ { r } } } \\ { \leq C _ { \mathrm { c o m p } } \| \mathfrak { q } _ { R , Q } - \Phi \| _ { C ^ { r } ( K _ { R , \operatorname* { d e c } } ) } < \frac { \delta } { 4 } . } \end{array}\tag{62}
$$

Combining (61) and (62), we obtain

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { a \in K } \| \mathcal { T } _ { R , Q } ( a ) - \mathcal { Q } ( \mathcal { H } _ { 1 } ( a ) ) \| _ { C ^ { r } } \leq \displaystyle \operatorname* { s u p } _ { a \in K } \| \mathcal { T } _ { R , Q } ( a ) - \mathfrak { q } _ { R , Q } ( \mathcal { H } _ { 1 } ( a ) ( \cdot ) , \cdot ) \| _ { C ^ { r } } } \\ { \displaystyle \qquad + \operatorname* { s u p } _ { a \in K } \| \mathfrak { q } _ { R , Q } ( \mathcal { H } _ { 1 } ( a ) ( \cdot ) , \cdot ) - \mathcal { Q } ( \mathcal { H } _ { 1 } ( a ) ) \| _ { C ^ { r } } } \\ { \displaystyle < \frac { \delta } { 4 } + \frac { \delta } { 4 } = \frac { \delta } { 2 } . } \end{array}
$$

Set $\mathcal { N } : = \mathcal { Q } \circ \mathcal { L } _ { 1 } \circ \mathcal { R } _ { R , \eta }$ . Since $\mathcal { H } _ { 1 } = \mathcal { L } _ { 1 } \circ \mathcal { R } _ { R , \eta } ,$ the preceding estimate becomes

$$
\operatorname* { s u p } _ { a \in K } \| \mathcal { T } _ { R , Q } ( a ) - \mathcal { N } ( a ) \| _ { C ^ { r } } < \frac { \delta } { 2 } ,
$$

which prove (47). Finally, the bound $\| w \| _ { Y } \leq C _ { Y } \| w \| _ { C ^ { r } }$ and the identity $\delta = \varepsilon / C _ { Y }$ imply

$$
\operatorname* { s u p } _ { a \in K } \| \mathcal { T } _ { R , Q } ( a ) - \mathcal { N } ( a ) \| _ { Y } \leq C _ { Y } \operatorname* { s u p } _ { a \in K } \| \mathcal { T } _ { R , Q } ( a ) - \mathcal { N } ( a ) \| _ { C ^ { r } } < \frac { \varepsilon } { 2 } .
$$

Together with (46), the triangle inequality therefore yields

$$
\begin{array} { r l } & { \underset { a \in K } { \operatorname* { s u p } } \Vert \mathcal { G } ( a ) - \mathcal { N } ( a ) \Vert _ { Y } \leq \underset { a \in K } { \operatorname* { s u p } } \Vert \mathcal { G } ( a ) - \mathcal { T } _ { R , Q } ( a ) \Vert _ { Y } + \underset { a \in K } { \operatorname* { s u p } } \Vert \mathcal { T } _ { R , Q } ( a ) - \mathcal { N } ( a ) \Vert _ { Y } } \\ & { \qquad < \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 2 } = \varepsilon . } \end{array}
$$

## Acknowledgments

Both authors acknowledge the support of the Austrian Science Fund (FWF) through project PAT4788625 (Grant-DOI: 10.55776/PAT4788625).

## References

[1] Ahmed Abdeljawad, Marcello Carioni, and Elena Cordero. Approximation Ratesfor Metaplectic Neural Networks. 2026. arXiv: 2608.08872.

[2] Ahmed Abdeljawad and Elena Cordero. “Time-frequency analysis for neural networks”. In: Applied and Computational Harmonic Analysis 84 (2026), p. 101883. DOI: 10.1016/j.acha. 2026.101883.

[3] Ahmed Abdeljawad and Thomas Dittrich. Space-Time Approximation with Shallow Neural Networks in Fourier Lebesgue spaces. 2023. arXiv: 2312.08461.

[4] Ahmed Abdeljawad and Thomas Dittrich. “Approximation Rates in Fréchet Metrics: Barron Spaces, Paley–Wiener Spaces, and Fourier Multipliers”. In: Neural Computation 38.2 (2026), pp. 165–197. DOI: 10.1162/NECO.a.1481.

[5] Ahmed Abdeljawad and Philipp Grohs. “Approximations with deep neural networks in Sobolev time-space”. In: Analysis and Applications 20.3 (2022), pp. 499–541. DOI: 10 . 1142/S0219530522500014.

[6] Robert A. Adams and John J. F. Fournier. Sobolev Spaces. 2nd ed. Vol. 140. Pure and Applied Mathematics. Amsterdam: Elsevier/Academic Press, 2003. ISBN: 978-0-12-044143-3.

[7] Andrew R. Barron. “Universal Approximation Bounds for Superpositions of a Sigmoidal Function”. In: IEEE Transactions on Information Theory 39.3 (1993), pp. 930–945. DOI: 10.1109/18.256500.

[8] Kaushik Bhattacharya et al. “Model Reduction and Neural Networks for Parametric PDEs”. In: SMAI Journal of Computational Mathematics 7 (2021), pp. 121–157. DOI: 10.5802/smaijcm.74.

[9] Emmanuel J. Candès. “Harmonic Analysis of Neural Networks”. In: Applied and Computational Harmonic Analysis 6.2 (1999), pp. 197–218. DOI: 10.1006/acha.1998.0248.

[10] Tianping Chen and Hong Chen. “Universal Approximation to Nonlinear Operators by Neural Networks with Arbitrary Activation Functions and Its Application to Dynamical Systems”. In: IEEE Transactions on Neural Networks 6.4 (1995), pp. 911–917. DOI: 10.1109/72.392253.

[11] Elena Cordero, Fabio Nicola, and Luigi Rodino. “Integral Representations for the Class of Generalized Metaplectic Operators”. In: Journal of Fourier Analysis and Applications 21.4 (2015), pp. 694–714. DOI: 10.1007/s00041-014-9384-8.

[12] Elena Cordero and Luigi Rodino. Time-Frequency Analysis ofOperators. Vol. 75. De Gruyter Studies in Mathematics. De Gruyter, 2020. DOI: 10.1515/9783110532456.

[13] Tim De Ryck, Samuel Lanthaler, and Siddhartha Mishra. “On the Approximation of Functions by tanh Neural Networks”. In: Neural Networks 143 (2021), pp. 732–750. DOI: 10.1016/j. neunet.2021.08.015.

[14] Vladimir Fanaskov and Ivan Oseledets. “Spectral Neural Operators”. In: Doklady Mathematics 108.Suppl. 2 (2023), S226–S232. DOI: 10.1134/S1064562423701107.

[15] Gerald B. Folland. Harmonic Analysis in Phase Space. Vol. 122. Annals of Mathematics Studies. Princeton, NJ: Princeton University Press, 1989. ISBN: 978-0-691-08528-9.

[16] Maurice A. de Gosson. Symplectic Methods in Harmonic Analysis and in Mathematical Physics. Vol. 7. Pseudo-Differential Operators: Theory and Applications. Basel: Birkhäuser, 2011. ISBN: 978-3-7643-9991-7. DOI: 10.1007/978-3-7643-9992-4.

[17] Gaurav Gupta, Xiongye Xiao, and Paul Bogdan. “Multiwavelet-based Operator Learning for Differential Equations”. In: Advances in Neural Information Processing Systems. Vol. 34. 2021. URL: https : / / papers . neurips . cc / paper \_ files / paper / 2021 / hash / c9e5c2b59d98488fe1070e744041ea0e-Abstract.html.

[18] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. “Universal Approximation of an Unknown Mapping and Its Derivatives Using Multilayer Feedforward Networks”. In: Neural Networks 3.5 (1990), pp. 551–560. DOI: 10.1016/0893-6080(90)90005-6.

[19] H. Hudzik, Rajeev Kumar, and Romesh Kumar. “Matrix Multiplication Operators on Banach Function Spaces”. In: Proceedings ofthe Indian Academy ofSciences - Section A 116.1 (Feb. 2006), pp. 71–81. ISSN: 0370-0089. DOI: 10.1007/BF02829740.

[20] Nikola Kovachki, Samuel Lanthaler, and Siddhartha Mishra. “On Universal Approximation and Error Bounds for Fourier Neural Operators”. In: Journal ofMachine Learning Research 22.290 (2021), pp. 1–76. URL: https://www.jmlr.org/papers/v22/21-0806.html.

[21] Nikola Kovachki et al. “Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs”. In: Journal of Machine Learning Research 24.89 (2023), pp. 1–97. URL: https://jmlr.org/papers/v24/21-1524.html.

[22] Samuel Lanthaler. “Operator Learning with PCA-Net: Upper and Lower Complexity Bounds”. In: Journal ofMachine Learning Research 24.318 (2023), pp. 1–67. URL: https://jmlr. org/papers/v24/23-0478.html.

[23] Samuel Lanthaler, Zongyi Li, and Andrew M. Stuart. “Nonlocality and Nonlinearity Implies Universality in Operator Learning”. In: Constructive Approximation 62 (2025), pp. 261–303. DOI: 10.1007/s00365-025-09718-3.

[24] Samuel Lanthaler, Siddhartha Mishra, and George E. Karniadakis. “Error Estimates for Deep-ONets: A Deep Learning Framework in Infinite Dimensions”. In: Transactions ofMathematics and Its Applications 6.1 (2022), tnac001. DOI: 10.1093/imatrm/tnac001.

[25] Samuel Lanthaler and Andrew M. Stuart. “The Parametric Complexity of Operator Learning”. In: IMA Journal ofNumerical Analysis 46.2 (2026), pp. 647–712. DOI: 10.1093/imanum/ draf028.

[26] Moshe Leshno et al. “Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function”. In: Neural Networks 6.6 (1993), pp. 861–867. DOI: 10.1016/S0893-6080(05)80131-5.

[27] Xin Li. “Simultaneous Approximations of Multivariate Functions and Their Derivatives by Neural Networks with One Hidden Layer”. In: Neurocomputing 12.4 (1996), pp. 327–343. DOI: 10.1016/0925-2312(95)00070-4.

[28] Zongyi Li et al. “Multipole Graph Neural Operator for Parametric Partial Differential Equations”. In: Advances in Neural Information Processing Systems. Vol. 33. 2020, pp. 6755–6766. arXiv: 2006.09535.

[29] Zongyi Li et al. Neural Operator: Graph Kernel Network for Partial Differential Equations. 2020. arXiv: 2003.03485.

[30] Zongyi Li et al. “Fourier Neural Operator for Parametric Partial Differential Equations”. In: International Conference on Learning Representations. 2021. URL: https://openreview. net/forum?id=c8P9NQVtmnO.

[31] Lu Lu et al. “Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators”. In: Nature Machine Intelligence 3.3 (2021), pp. 218–229. DOI: 10.1038/s42256-021-00302-5.

[32] Victor Namias. “The Fractional Order Fourier Transform and Its Application to Quantum Mechanics”. In: IMA Journal of Applied Mathematics 25.3 (1980), pp. 241–265. DOI: 10. 1093/imamat/25.3.241.

[33] Thomas Runst and Winfried Sickel. Sobolev Spaces ofFractional Order, Nemytskij Operators, and Nonlinear Partial Differential Equations. Vol. 3. De Gruyter Series in Nonlinear Analysis and Applications. Berlin: Walter de Gruyter, 1996. ISBN: 978-3-11-015113-8. DOI: 10.1515/ 9783110812411.

[34] Jonathan W. Siegel and Jinchao Xu. “Approximation Rates for Neural Networks with General Activation Functions”. In: Neural Networks 128 (2020), pp. 313–321. DOI: 10.1016/j. neunet.2020.05.019.

[35] Karn Tiwari, N. M. Anoop Krishnan, and Prathosh A. P. “CoNO: Complex Neural Operator for Continous Dynamical Physical Systems”. In: APL Machine Learning 3.2, 026101 (2025). DOI: 10.1063/5.0254013.

[36] Tapas Tripura and Souvik Chakraborty. “Wavelet Neural Operator for Solving Parametric Partial Differential Equations in Computational Mechanics Problems”. In: Computer Methods in Applied Mechanics and Engineering 404, 115783 (2023). DOI: 10.1016/j.cma.2022. 115783.

[37] Chen Wang et al. “Fractional Fourier Series on the Torus and Applications”. In: Fractal and Fractional 8.8 (2024), p. 494. DOI: 10.3390/fractalfract8080494.