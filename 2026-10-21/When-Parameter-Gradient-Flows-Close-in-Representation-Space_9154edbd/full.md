# When Parameter Gradient Flows Close in Representation Space

Shengye Tao

Beijing Institute of Mathematical Sciences and Applications (BIMSA)

Beijing, China

ShengyeTao@bimsa.cn RanchoTao@gmail.com

October 20, 2026

## Abstract

The represented gradient flow of a parameterized model depends on a mobility that can retain hidden parameter information. We study when one entire represented mobility map works for every autonomous loss from one fixed initialization. For two-factor matrix representations wider than both output dimensions, with individually full-row-rank factors, we classify all admissible joint ranks: closure occurs precisely at efective scalar-balanced minimal joint rank or a two-value signed spectral family at full joint rank. All intermediate ranks are excluded; positive represented flows belong to known square-root families. For general representations with constant ambient representation and gradient-Lie ranks, ordinary closure is equivalent to a local invariant-level metric in the analytic case and specified smooth regimes. An endpoint section forces the initially unrestricted mobility map to admit a locally smooth representative on the represented image. We also separate testing families: linear losses locally admit closure at every submersion, whereas one fixed translated strongly convex quadratic family fails it on a delayed-bracket example. A fixed output readout gives an ordinary squared-loss witness. These supporting results do not characterize quadratic-only closure in general. We retain the unresolved smooth output-ranktwo case with Lie rank at least four. A frozen deterministic experiment checks a separate engineered-loss construction; it is distinguished from the squared-loss proof, universal theorems and empirical learning performance.

## 1 Introduction

For an open parameter domain $\Omega \subset \mathbb { R } ^ { D }$ and a smooth representation $\phi : \Omega \to \mathbb { R } ^ { d }$ , parameter gradient flow of a represented loss f gives

$$
\dot { \theta } = - D \phi ( \theta ) ^ { T } \operatorname { g r a d } f ( \phi ( \theta ) ) , \qquad \dot { z } = - M ( \theta ) \operatorname { g r a d } f ( z ) , \qquad M = D \phi D \phi ^ { T } .\tag{1}
$$

States with the same z may have diferent M. We ask when the entire mobility, rather than only its action on the gradient of a particular loss, is determined by z along all loss trajectories from one fixed initialization. This is the ordinary intrinsic-dynamic contract of [1], distinct from its stronger piecewise-loss testing convention.

The two results proved below are an initialization classification for wide two-factor matrix representations and a converse from ordinary trajectory tests to invariant-level metric factorization for regular representations in stated analytic and smooth regimes. All positive matrix mobilities reduce to known scalar-balanced flows; no new optimizer, optimization rate or task gain is claimed. The distinction is a necessity theory for eliminating hidden mobility, not a new formula for the already known positive flow. The loss-family quantifier is substantive: Proposition 6.1 shows that all linear tests can pass locally at a submersion, while Theorem 6.3 and Corollary 6.4 give exact nonclosure with familiar output least-squares losses in a fixed model. The latter are supporting exclusion controls, not new learning algorithms. Generic continuous initializations in the nonscalar wide model fail this exact closure, as a direct consequence of the classification. This does not exclude approximate closure or a simpler description under one particular loss.

The work uses the conservation-law and Lie-distribution framework of [2, 1]. In particular the gradient Lie algebra and maximal-stack factor-orbit dimensions in the matrix proof are prior ingredients, explicitly attributed below. The additional inference under study is how single-initialization, autonomous-loss tests constrain Gram fibers and a local invariant-level mobility. Familiar Frobenius, constant-rank, inverse-function, orbit and ODE results are used as standard local theorems, with their hypotheses identified at each application.

Results and reading guide. Theorem 3.1 gives an initialization classification, including the distinction between minimal, intermediate, and full joint rank. Theorem 4.2 supplies the fixed-start converse in its stated regular regimes. Theorem 6.3 separates linear testing from a fixed-Hessian quadratic family; Corollary 6.4 transfers the witness to squared error after one fixed readout. The scalar-output case and positive square-root flows are known. Section $7$ compares these statements with specific predecessor results. Complete proofs are in the appendices.

## 2 Problem Setting and Definitions

Fix $\theta _ { 0 } \in \Omega$ . A solution on an open set means its forward maximal local interval before exiting that set; no global-time existence is assumed. Losses are global $\mathcal { C } ^ { 2 }$ maps $f : \mathbb { R } ^ { d } $ R without a common bound on their derivatives. Neighborhoods can depend on initialization but not on the tested loss.

Definition 2.1 (Ordinary entire-mobility closure). Property (O) holds at $\theta _ { 0 }$ if there exist an open neighborhood $U \ \ni \ \theta _ { 0 }$ and a single map $K : \mathbb { R } ^ { \dot { d } }  \mathbb { R } ^ { d \times d }$ such that every solution of (1) from $\theta _ { 0 }$ , for every global $\mathcal { C } ^ { 2 }$ loss, satisfies $M ( \theta ( t ) ) = K ( \phi ( \theta ( t ) ) )$ on its interval in $U$ . The map $K$ is independent of the loss and is initially unrestricted: continuity, boundedness and diferentiability are not assumptions.

Definition 2.2 (Local invariant-level metric). Property (I) holds if, on a parameter neighborhood, there are finitely many common ${ \mathcal { C } } ^ { 1 }$ conservation laws $h = ( h _ { 1 } , \ldots , h _ { k } )$ and one globally ${ \mathcal { C } } ^ { 1 }$ map $\overline { { K } } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times d }$ such that

$$
M ( \theta ) = \overline { { K } } ( \phi ( \theta ) ) \quad \mathrm { f o r ~ e v e r y ~ } \theta \in V \cap \{ h = h ( \theta _ { 0 } ) \}
$$

for some open $V \ni \theta _ { 0 }$ . Common conservation means Dϕ grad $h _ { j } = 0$ on the domain of $h$

The laws h are local, as in the conservation-law convention of [1]; the represented extension K is global. Equality is required on the actual local level, not merely on its tangent. Conservation and substitution immediately give $( \mathrm { I } ) { \Rightarrow } ( \mathrm { O } )$ . The strong intrinsic dynamic property of [1, Definition I.2] instead tests piecewise $\mathcal { C } ^ { 2 }$ represented losses $f _ { \cdot }$ , requiring a continuous trajectory from $\theta _ { 0 }$ to follow $- \nabla ( f \circ \phi )$ wherever $f$ is diferentiable along its represented path, with one common neighborhood and one common $K$ . This is a spatial piecewise-loss convention; arbitrary time-dependent switching is a separate testing class. Strong closure implies ordinary closure, so an obstruction to strong closure alone does not prove failure of ordinary closure. Finite flow concatenations below are auxiliary geometric constructions; each ordinary witness is realized by an actual smooth autonomous loss from the fixed initial state.

No positive definiteness on all target space is part of these definitions: the mobility may be semidefinite when the representation rank is less than d.

Let $X _ { i } = \operatorname { g r a d } \phi _ { i }$ and let $E _ { \theta }$ be the span of the values at $\theta$ of their generated Lie algebra. We use the bracket $[ X , Y ] = D Y X - D X Y$ . When dim $E = p$ is constant on an ambient neighborhood, selecting a nonzero minor of finitely many bracket fields gives a smooth frame for E. The frame brackets lie in $E ,$ so Frobenius applies. It supplies a local product box with common first integrals $h$ and a connected initial plaque $L = \{ h = h ( \theta _ { 0 } ) \}$ of dimension $p .$ . If $p = D$ , take h constant. If rank $D \phi = q$ is constant as well, then

$$
\mathrm { r a n k } D ( \phi | _ { L } ) = q \le p ,\tag{2}
$$

because im $D \phi ^ { T } \subset T L$ and $\mathrm { i m } ( D \phi D \phi ^ { T } ) = \mathrm { i m } D \phi$ . The analytic versions follow by selecting analytic frames and using analytic Frobenius. These statements concern one local plaque and require neither globally embedded leaves nor a global quotient.

Lemma 2.3 (Compatible loss on an embedded represented arc). Let a smooth parameter curve satisfy $\dot { \theta } = - D \phi ( \theta ) ^ { T } g ( t )$ with smooth $^ { g , }$ and suppose its represented curve $\gamma = \phi \circ \theta$ is a regular embedded compact arc admitting a slightly extended embedded arc. Then this curve is a trajectory of one global smooth autonomous represented loss from its given initial point.

Proof. See Appendix A.1.

Lemma 2.4 (Clock-positive endpoint patch). Suppose a local embedded manifold $L \ni \theta _ { 0 }$ is invariant for all fields $Y _ { g } = - D \phi ^ { T } g , \ g \in \mathbb { R } ^ { d }$ , and their Lie values span $T _ { \theta } L$ on L. Assume $M ( \theta _ { 0 } ) \neq 0$ Every suficiently small ambient neighborhood admits a nonempty L-open set of endpoints from $\theta _ { 0 }$ each realized by one global smooth autonomous represented loss with its whole path in that neighborhood. Consequently (O) implies $M = K \circ \phi$ on this set. Constant Lie rank in the ambient parameter space is not required.

Proof. See Appendix A.2.

Lemma 2.5 (Linear losses force local regularity). Suppose $\phi$ is smooth with constant ambient rank $q > 0$ near $\theta _ { 0 }$ . Under (O), its common arbitrary K agrees on the whole local represented image near $z _ { 0 } = \phi ( \theta _ { 0 } )$ with a smooth function, admitting a smooth extension $K _ { \mathrm { l o c } }$ to a target neighborhood. If $\phi$ is analytic the section and local extension can be analytic. No Lie-rank assumption is needed.

Proof. See Appendix A.3.

Lemma 2.6 (Local descent and global represented extension). Let L be a local plaque on which $\phi$ has constant rank q. If $D ( M | _ { L } )$ annihilates ker $D ( \phi | _ { L } )$ throughout a neighborhood $o f \theta _ { 0 }$ in $L$ , then $M = k \circ \phi$ on a smaller plaque neighborhood for a smooth local represented k. If L is the exact local level of common laws, this gives (I) after shrinking the parameter box.

Proof. See Appendix A.4.

## 3 Complete classification for wide two-factor representations

We identify matrix spaces with Euclidean spaces using the Frobenius inner product. Put

$$
\begin{array} { r } { N = n + m , \qquad J = \mathrm { d i a g } ( I _ { n } , - I _ { m } ) , \qquad F = \left[ { \boldsymbol { U } } \right] , \qquad C = F F ^ { \top } = \left[ \begin{array} { l l } { { \boldsymbol { A } } } & { { \boldsymbol { Z } } } \\ { { \boldsymbol { Z } } ^ { \top } } & { { \boldsymbol { B } } } \end{array} \right] , \qquad S = F ^ { \top } J F . } \end{array}
$$

Here $U \in \mathbb { R } ^ { n \times r } , V \in \mathbb { R } ^ { m \times r }$ , and $\phi ( F ) = Z = U V ^ { \top }$ . Direct diferentiation gives

$$
{ \cal D } \phi ( { \boldsymbol F } ) [ \delta { \boldsymbol U } , \delta { \boldsymbol V } ] = \delta { \boldsymbol U } { \boldsymbol V } ^ { \top } + { \boldsymbol U } \delta { \boldsymbol V } ^ { \top } , \qquad { \boldsymbol D } \phi ( { \boldsymbol F } ) ^ { \top } { \boldsymbol G } =  { \left[ G ^ { \boldsymbol V } \right] } .
$$

Consequently the full represented mobility is the linear operator

$$
M _ { F } [ G ] = A G + G B , \qquad A = U U ^ { \top } , \quad B = V V ^ { \top } .\tag{3}
$$

For $G \in \mathbb { R } ^ { n \times m }$ , write

$$
H _ { G } = \left[ \begin{array} { c c } { { 0 } } & { { G } } \\ { { G ^ { \top } } } & { { 0 } } \end{array} \right] , \qquad X _ { G } ( F ) = - H _ { G } F .
$$

These are the parameter gradient fields of the linear represented losses. An arbitrary represented loss $f \in C ^ { 2 } ( \mathbb { R } ^ { n \times m } , \mathbb { R } )$ gives

$$
\dot { F } = - H _ { \mathrm { g r a d } f ( Z ) } F .\tag{4}
$$

For clarity, property (O) at $F _ { 0 }$ means that one map $K : \mathbb { R } ^ { n \times m }  \mathbb { R } ^ { ( n m ) \times ( n m ) }$ , independent of $f ,$ equals the entire $M _ { F }$ along every such autonomous flow from $F _ { 0 }$ , until first exit from one common parameter neighborhood. No regularity of $K$ is assumed. Property (I) means that a finite family of common local $C ^ { 1 }$ conservation laws $h ,$ together with a globally $C ^ { 1 }$ represented map ${ \overline { { K } } } .$ , gives $M _ { F } = \overline { { K } } ( \phi ( F ) )$ on the exact local level $h ( F ) = h ( F _ { 0 } )$ . The conservation laws need only be defined on a parameter neighborhood. Conservation and the chain rule show that (I) implies (O).

The gradient Lie algebra and the factor-orbit dimensions used below are prior ingredients of the conservation-law framework of Marcotte, Gribonval, and Peyré [2], Propositions H.3 and H.5. We include their elementary linear-algebra derivations to make the present argument complete. Likewise the positive scalar-balanced square-root dynamics are known [1, 3]; the classification below concerns their initialization conditions.

Theorem 3.1 (Wide initialization classification). Assume $n , m \geq 1 , r > \operatorname* { m a x } ( n , m )$ , and rank $U _ { 0 } =$ $n _ { ; }$ , rank $V _ { 0 } = m$ . Set $L _ { 0 } = \operatorname* { m a x } ( n , m ) , s = \operatorname { r a n k } F _ { 0 }$ , and $C _ { 0 } = F _ { 0 } F _ { 0 } ^ { \top }$ . Properties (O) and (I), with the latter interpreted in the original width-r parameter space, are equivalent to the union of the following disjoint cases:

$$
s = L _ { 0 } , \qquad C _ { 0 } J C _ { 0 } = \lambda C _ { 0 } \quad f o r \ s o m e \ \lambda \in \mathbb { R } ;\tag{5}
$$

$$
s = N , \qquad \mathrm { s p e c } ( J C _ { 0 } ) = \{ a w i t h ~ m u l t i p l i c i t y ~ n , ~ - b ~ w i t h ~ m u l t i p l i c i t y ~ m \} , \quad a , b > 0 .\tag{6}
$$

In (5), $n > m$ forces $\lambda > 0 , m > n$ forces $\lambda < 0$ , and $n = m$ permits any λ. There are no positive cases with $L _ { 0 } < s < N$ . The mobility in either positive case is

$$
K _ { \sigma } ( Z ) [ G ] = \sqrt { \sigma ^ { 2 } I _ { n } + Z Z ^ { \top } } G + G \sqrt { \sigma ^ { 2 } I _ { m } + Z ^ { \top } Z } ,\tag{7}
$$

where $\sigma = | \lambda | / 2$ in (5) and $\sigma = ( a + b ) / 2$ in (6). For $\sigma = 0$ , the formula is used on an invertible represented neighborhood and extended smoothly by a cutof.

## 3.1 Factor orbits and the ordinary-closure obstruction

Lemma 3.2 (Boost algebra and factor-orbit dimension). The fields $X _ { G }$ generate the left action of

$$
\mathfrak { o } ( n , m ) = \{ T \in \mathbb { R } ^ { N \times N } : T ^ { \top } J + J T = 0 \} .
$$

Which initializations close?
<table><tr><td>Joint rank s</td><td>Exact condition</td><td>Closure</td></tr><tr><td> $s = L _ { 0 }$ </td><td> $C _ { 0 } J C _ { 0 } = \lambda C _ { 0 }$  for some  $\lambda \in \mathbb { R }$ </td><td>Yes iff</td></tr><tr><td> $L _ { 0 } < s < N$ </td><td>Every admissible initialization</td><td>No</td></tr><tr><td> $s = N$ </td><td> $\mathrm { s p e c } ( J C _ { 0 } ) = \{ a ^ { [ n ] } , - b ^ { [ m ] } \} , a , b > 0$ </td><td>Yes iff</td></tr></table>

L<sub>0</sub> = max(n, m), N = n + m, s  min(r, N); $r > L _ { 0 } .$ rank $U _ { 0 } = n ,$ rank $V _ { 0 } = m .$

Figure 1: Initialization classification in Theorem 3.1. Here $L _ { 0 } = \operatorname* { m a x } ( n , m )$ $N = n + m , s =$ rank $F _ { 0 } , C _ { 0 } = F _ { 0 } F _ { 0 } ^ { \top }$ , and $J = \mathrm { d i a g } ( I _ { n } , - I _ { m } )$ . Only ranks $s \leq r$ are attainable. The assumptions $r > L _ { 0 }$ , rank $U _ { 0 } = n$ , and rank $V _ { 0 } = m$ apply to every row. At minimal joint rank, $n > m$ forces $\lambda > 0$ and $m > n$ forces $\lambda < 0 ;$ ; when $n = m$ , either sign or zero is allowed. This is a theorem summary, not a numerical experiment.

At a factor matrix of rank s, its local orbit has dimension

$$
d _ { F } = { \binom { N } { 2 } } - { \binom { N - s } { 2 } } = { \frac { s ( 2 N - s - 1 ) } { 2 } } .\tag{8}
$$

This formula does not require J restricted to im F to be nondegenerate. Every common $C ^ { 1 }$ conservation law is constant on the connected local orbit.

Proof. See Appendix B.1.

Lemma 3.3 (A necessary Gram-orbit dimension bound). If $F _ { 0 } \neq 0$ satisfies (O), then the congruence orbit of $C _ { 0 } = F _ { 0 } F _ { 0 } ^ { \top }$ under the J-orthogonal group has dimension $d _ { C } \le n m$ . The same bound follows from (I), for arbitrary finite families of common $C ^ { 1 }$ conservation laws.

Proof. See Appendix B.2.

## 3.2 The full-column and degenerate-inertia calculations

Lemma 3.4 (Full-column Gram rank). Suppose $\boldsymbol { F } \in \mathbb { R } ^ { N \times s }$ has full column rank, and let $c _ { 1 } , \ldots , c _ { \ell }$ be the multiplicities of all distinct eigenvalues of the symmetric $S = F ^ { \top } J F$ , including zero when present. Then

$$
d _ { C } = \frac { s ( 2 N - s - 1 ) } { 2 } - \sum _ { j = 1 } ^ { \ell } { \binom { c _ { j } } { 2 } } .\tag{9}
$$

Proof. See Appendix B.3.

Lemma 3.5 (Strict excess at intermediate width). For $\operatorname* { m a x } ( n , m ) < s < N$ and full-column $\boldsymbol { F } \in \mathbb { R } ^ { N \times s }$ , one has $d _ { C } > n m$ , whether or not S is singular.

Proof. See Appendix B.4.

Lemma 3.6 (The minimal full-column width). Suppose $s = L _ { 0 } = \operatorname* { m a x } ( n , m )$ , F is full column rank, and both blocks have their full row ranks. Then $d _ { C } \leq$ nm $i f$ and only if $S = \lambda I _ { s }$ . This is equivalent to $C J C = \lambda C$ , with the sign conditions in Theorem 3.1.

Proof. See Appendix B.5.

## 3.3 Full joint rank and the signed spectral family

Lemma 3.7 (Full-row Gram orbit and its positive family). Suppose $\boldsymbol { F } \in \mathbb { R } ^ { N \times \boldsymbol { r } }$ has full row rank, so $\boldsymbol { C } = \boldsymbol { F } \boldsymbol { F } ^ { \intercal } > 0$ . Let $c _ { i }$ and $d _ { j }$ be the multiplicities of the distinct positive and negative eigenvalue values, respectively, of JC. Then

$$
d _ { C } - n m = { \frac { n ^ { 2 } - \sum _ { i } c _ { i } ^ { 2 } + m ^ { 2 } - \sum _ { j } d _ { j } ^ { 2 } } { 2 } } .\tag{10}
$$

Thus $d _ { C } \leq$ nm precisely when JC has one positive value a and one negative value b, with multiplicities $n , m$ . At such an initialization the original ambient canonical level $F ^ { \top } J F = F _ { 0 } ^ { \top } J F _ { 0 }$ has mobility (7), with $\sigma = ( a + b ) / 2 > 0$

Proof. See Appendix B.6.

## 3.4 Efective hidden width and the original ambient level

Lemma 3.8 (Exact width reduction for fixed-start trajectories). Let $s =$ rank $F _ { 0 }$ , and choose an orthogonal hidden matrix Q with $F _ { 0 } Q = [ \widetilde { F } _ { 0 } , 0 ]$ , where $\widetilde { F } _ { 0 }$ has s independent columns. For every represented $C ^ { 2 }$ eloss, the original flow from $F _ { 0 }$ eand the width-s flow from $\widetilde { F } _ { 0 }$ have identical $Z , A , B , M$ eon corresponding existence intervals. Property (O) holds at one of these initializations $i f$ and only if it holds at the other. If both original blocks have full row rank, so do the reduced blocks.

Proof. See Appendix B.7.

Lemma 3.9 (Minimal-rank suficiency in the original width). Under the assumptions of Theorem 3.1, suppose $s = L _ { 0 }$ and $C _ { 0 } J C _ { 0 } = \lambda C _ { 0 }$ . Then (I) holds in the original width-r space, using the canonical common invariant $h ( F ) = F ^ { \top } J F$

Proof. See Appendix B.8.

Proof of Theorem 3.1. See Appendix B.9.

## 3.5 The matching known positive dynamics

Proposition 3.10 (Exact narrow scalar-balanced realization). For $\sigma > 0$ and every $Z \in \mathbb { R } ^ { n \times m }$ the operator (7) is realized by width min $( n , m )$ factors with scalar hidden imbalance. This includes rank-deficient Z. For $\sigma = 0$ and square invertible Z, a balanced realization exists locally as well. Thus the positive cases in Theorem 3.1 do not provide new represented continuous-time trajectories.

Proof. See Appendix B.10.

Corollary 3.11 (Generic continuously sampled wide factors). Let $r > \operatorname* { m a x } ( n , m )$ . For any jointly absolutely continuous initialization law on the full factor space, properties (O) and (I) have probability zero $i f \left( n , m \right) \neq \left( 1 , 1 \right)$ . For the scalar pair they have probability one. No independence of the factor entries is required.

Proof. See Appendix B.11.

Remark 3.12. The corollary is a direct classification consequence. It concerns exact entire-mobility closure, with an unrestricted universal loss-testing class. It gives no quantitative approximate closure, learning-performance, stochastic, or discrete-time claim. Neither the classification nor its proof covers every individually rank-deficient block initialization.

## 4 From initialized trajectory tests to a local metric

We use the ordinary closure and local metric properties in Definitions 2.1 and 2.2. Throughout this section the parameter metric is the original Euclidean metric. Write

$$
M = D \phi D \phi ^ { \mathsf { T } } , \qquad \mathcal { H } = \mathrm { i m a g e } D \phi ^ { \mathsf { T } } , \qquad \mathcal { E } = \mathrm { L i e } \{ \mathrm { g r a d } \phi _ { i } : 1 \leq i \leq d \} .
$$

Here the last expression denotes the pointwise distribution of the generated Lie algebra. Unless stated otherwise, its rank $p$ and the rank $q$ of $D \phi$ are constant on an ambient open neighborhood of the initialization $\theta _ { 0 }$ . All subsequent shrinkages are inside the common neighborhood in the ordinary closure hypothesis. Frobenius, the constant-rank theorem, smooth dependence of ODE solutions, and tubular coordinates are used in their classical local forms.

Lemma 4.1 (The regular initial plaque). There is a smooth Frobenius product box with finitely many smooth common conservation laws h, whose initial level is one connected p-dimensional plaque L tangent to . $I f \phi$ is real analytic, this box, L, and h can be chosen real analytic. On $L , D ( \phi | _ { L } )$ has constant rank $q .$

Proof. See Appendix C.1.

Theorem 4.2 (Regular analytic and specified smooth regimes). For a real-analytic representation with the two constant ambient ranks, ordinary closure is equivalent to a local invariant-level metric. The same equivalence holds for a smooth representation when $q \geq 3$ , when $q \leq 1$ , or when $q = 2$ and $p \leq 3$ . The conservation laws and the global represented extension in the conclusion can be smooth.

## 4.1 The analytic case and the exact descent

Proof of Theorem 4.2 in the analytic case. See Appendix C.2.

## 4.2 Smooth horizontal paths in represented rank at least three

Lemma 4.3 (An elementary embedded-arc approximation). Let $q \geq 3$ and let $w : [ 0 , T ] \to \mathbb { R } ^ { q }$ $T > 0$ , be smooth, with compact image in an open coordinate set. There are vectors $a _ { j }  0 ~ f o r$ which $w _ { j } ( t ) = w ( t ) + t a _ { j }$ are regular embedded smooth arcs, remain in the same coordinate set, have $w _ { j } ( 0 ) = w ( 0 )$ , and converge to w in $C ^ { 1 }$ . Their other endpoints need only converge.

Proof. See Appendix C.3.

Lemma 4.4 (Closure on a smooth horizontal path). Assume ordinary closure, the constant representation rank, and $q \geq 3$ . On a suficiently small parameter neighborhood, every smooth path from $\theta _ { 0 }$ tangent to  satisfies $M = K _ { \mathrm { l o c } } \circ \phi$ , where $K _ { \mathrm { l o c } }$ is the forced smooth representative of Lemma ${ \it 2 . 5 . }$

Proof. See Appendix C.4.

Lemma 4.5 (Local finite-word accessibility). On the regular plaque L, every point of a suficiently small neighborhood of $\dot { } \theta _ { 0 }$ is the endpoint of a finite horizontal flow word lying in a prescribed slightly larger plaque neighborhood. Such words admit smooth horizontal time reparametrizations with the same endpoints.

Proof. See Appendix C.5.

Proof of Theorem 4.2 for $q \geq 3$ and the elementary low-rank cases. See Appendix C.6.

## 4.3 The rank-two, three-dimensional-leaf case

Lemma 4.6 (A finite horizontal curvature $\mathrm { j e t } )$ . Suppose $q = 2 , p = 3$ . On L, choose coordinates $( u , v , \eta )$ with $( u , v ) = \pi$ represented coordinates and $\theta _ { 0 } = ( 0 , 0 , 0 )$ . Write its horizontal lifts as

$$
H _ { 1 } = \partial _ { u } + A \partial _ { \eta } , \qquad H _ { 2 } = \partial _ { v } + B \partial _ { \eta } , \qquad [ H _ { 1 } , H _ { 2 } ] = c \partial _ { \eta } .
$$

Some finite ordered horizontal derivative of c is nonzero at $\theta _ { 0 }$ . If m is the least such order, the leading word tensor is symmetric and the homogeneous polynomial $P _ { m } ( e ) = H _ { e } ^ { m } c ( \theta _ { 0 } ) , H _ { e } = e _ { 1 } H _ { 1 } +$ $e _ { 2 } H _ { 2 }$ , is not identically zero. For $m = 0$ this means $P _ { 0 } = c ( \theta _ { 0 } ) \neq 0$

Proof. See Appendix C.7.

Lemma 4.7 (Four rectangle families and a common fiber interval). Under the preceding hypotheses there are two independent unit base directions and their negatives, and one fixed small $\varepsilon > 0$ , such that the signed rectangle loops in any of these four directions have horizontal endpoints covering one common open interval about $\theta _ { 0 }$ in the initial one-dimensional fiber.

Proof. See Appendix C.8.

For the next lemma, let $\mathcal { D } = \{ \pm e ^ { ( 1 ) } , \pm e ^ { ( 2 ) } \}$ and $\varepsilon > 0$ be the directions and fixed length supplied by Lemma 4.7. For $e \in \mathcal { D } .$ , let $e ^ { \perp }$ be its counterclockwise rotation through $\pi / 2$ . The rectangle of signed width δ is the loop

$$
0 \longrightarrow \varepsilon e \longrightarrow \varepsilon e + \delta e ^ { \perp } \longrightarrow \delta e ^ { \perp } \longrightarrow 0 ,
$$

with $\delta$ suficiently small.

Lemma 4.8 (The required planar approximation). For any suficiently small nonzero base point $b ,$ one may select $e \in \mathcal { D }$ with $e \cdot b < 0$ . For fixed suficiently small nonzero δ, the rectangle in Lemma $4 . 7 ,$ followed by the straight tail from zero to $b ,$ is a limit of regular embedded smooth base arcs with initial point exactly zero and terminal point b. Their velocities converge in $L ^ { 1 }$ and their total variations remain bounded. For $b = 0$ their terminal points may instead converge to zero. These assertions sufice to pass the corresponding horizontal lift endpoints through the forced smooth $K _ { \mathrm { l o c } }$

Proof. See Appendix C.9.

Proof of Theorem 4.2 for $q = 2 , p = 3$ . See Appendix C.10.

## 4.4 All-order and Euclidean necessary conditions

Proposition 4.9 (All-order flatness at the initial plaque). Under smooth constant ambient ranks and ordinary closure, take $K _ { \mathrm { l o c } } = 0$ when $q = 0$ and otherwise use the smooth representative from Lemma 2.5. Set $R = M - K _ { \mathrm { l o c } } \circ \phi$ . Every finite plaque-coordinate derivative of $R \vert _ { L }$ at $\theta _ { 0 }$ vanishes.

Proof. See Appendix C.11.

Proposition 4.10 (A Euclidean output obstruction). Suppose $q = 2$ . Let $u = a \cdot \phi + c _ { 0 }$ and $v = b \cdot \phi$ be constant output-linear combinations with independent gradients X = grad u, Y = grad v on the chosen neighborhood. Assume $D ^ { 2 } u Y \in { \mathcal { H } }$ there. Ordinary closure holds if and only $i f p = 2$ . In particular this applies to an afine u, or to an independent isotropic quadratic $u = \alpha \lVert \theta \rVert ^ { 2 } / 2 + a _ { 0 } . \theta + c _ { 0 }$ that is such an output combination.

Proof. See Appendix C.12.

Same representation, diferent mobility

$$
\begin{array} { c } { { \phi _ { 2 } ( x , y , z ) = ( x , y + x ^ { 2 } z ) ~ { \cal { M } } = \left( \begin{array} { c c } { { 1 } } & { { 2 x z } } \\ { { 2 x z } } & { { 1 + x ^ { 4 } + ( 2 x z ) ^ { 2 } } } \end{array} \right) } } \\ { { { } } } \\ { { \theta _ { A } = ( \frac { 1 } { 2 } , 0 , 0 ) ~ { \longrightarrow } ~ \phi _ { 2 } ( \theta _ { A } ) = ( \frac { 1 } { 2 } , 0 ) ~ { \cal { M } } _ { A } = \left( \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { 0 } } & { { 1 7 / 1 6 } } \end{array} \right) } } \\ { { { } } } \\ { { { } } } \\ { { \theta _ { B } = ( \frac { 1 } { 2 } , - \frac { 1 } { 2 0 } , \frac { 1 } { 5 } ) ~ \longrightarrow ~ \phi _ { 2 } ( \theta _ { B } ) = ( \frac { 1 } { 2 } , 0 ) ~ { \cal { M } } _ { B } = \left( \begin{array} { c c } { { 1 } } & { { 1 / 5 } } \\ { { 1 / 5 } } & { { 4 4 1 / 4 0 0 } } \end{array} \right) } } \\ { { { } } } \\ { { { } { \mathrm { E x a c t ~ f l b e r ~ i l l u s t r a t i o n : } } ~ [ M _ { B } - M _ { A } ] _ { 1 2 } = 1 / 5 \neq 0 . } } \end{array}
$$

Figure 2: An algebraic illustration for $\phi _ { 2 } ( x , y , z ) = ( x , y + x ^ { 2 } z )$ . Two states have the same representation and diferent mobility. These are exact rational states on one fiber; this panel does not assert that they are endpoints of two losses from the same initialization. The actual fixed-start witnesses are proved in Proposition 5.1 and Theorem 6.3. No integration or learning benchmark is shown.

Remark 4.11. The general smooth case $q = 2 , p \geq 4$ is not settled here. Proposition 4.9 makes a counterexample’s actual mobility discrepancy flat to every order at the initialization, but does not rule out a nonzero smooth flat discrepancy. A planar non-density observation or an arbitrary connection is not by itself a Euclidean representation satisfying the universal ordinary premise. None of these results asserts a new positive optimizer, discrete dynamics, or an implication from a single observed loss.

## 5 Exact autonomous-loss exclusion controls

The following constructions illustrate the logical strength of the ordinary premise. They are algebraic witnesses, not experiments or separate novelty claims. They do not refute the bridge theorem, because they fail its ordinary closure assumption.

Proposition 5.1 (Arbitrarily delayed brackets and a two-loss witness). For each integer $k \geq 2$ let $\phi _ { k } ( x , y , z ) \ : = \ : ( x , y + x ^ { k } z )$ . Its representation rank is two and its gradient Lie rank is three everywhere. At the origin all iterated brackets of lengths two through k vanish, while a bracket of length $k + 1$ supplies the third direction. The first diferential mobility test passes at the origin, yet ordinary closure fails there, as witnessed by two global smooth autonomous losses.

Proof. See Appendix D.1.

Proposition 5.2 (A rank-four afine-output exclusion). For $\phi ( x , y , z , w ) = ( x , y + x z + z w ) , q = 2$ and $p = 4$ everywhere, and ordinary closure fails at every initialization.

Proof. See Appendix D.2.

## 6 What restricted loss families test

The arbitrary-loss quantifier in property (O) is substantive. For a family $\mathcal { F }$ of global output losses, write $( \mathrm { O } ) _ { \mathcal { F } }$ for the same property with only $f \in { \mathcal { F } }$ tested: one parameter neighborhood $U$ and one entire matrix map $K : \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times d }$ , independent of $f _ { i }$ must satisfy

$$
M ( \theta _ { f } ( t ) ) = K ( \phi ( \theta _ { f } ( t ) ) )
$$

on every forward local solution interval before exit from U, from the same fixed initialization. The matrix K is initially arbitrary. In particular the equality concerns the entire mobility, rather than only its product with the tested loss gradient. Property (O) restricts to every smaller family, but its necessity conclusions cannot be transferred in the opposite direction without proof. The following supporting results make this distinction explicit.

Proposition 6.1 (Linear tests at a submersion). Suppose ϕ is smooth and $D \phi ( \theta _ { 0 } )$ has full row rank d. Then $( \mathrm { O } ) _ { \mathcal { F } _ { \mathrm { l i n } } }$ holds for $\mathcal { F } _ { \mathrm { l i n } } = \{ z \mapsto g ^ { T } z : g \in \mathbb { R } ^ { d } \}$ . A common K can be chosen globally smooth after shrinking the parameter neighborhood. This holds for all g, without a common bound on its norm.

## Proof. See Appendix E.1.

Remark 6.2 (Linear tests do not recover invariant-level factorization). For $k \geq 2$ , consider

$$
\phi _ { k } ( x , y , z ) = ( x , y + x ^ { k } z ) , \qquad X = \partial _ { x } , \quad Y = ( k x ^ { k - 1 } z , 1 , x ^ { k } ) .
$$

The representation rank is two everywhere, and $\operatorname { a d } _ { X } ^ { k } Y = k ! \partial _ { z }$ gives gradient Lie rank three. Hence Proposition 6.1 applies. Nevertheless property (I) fails at the origin, directly at the level of its actual conservation-law constraint. A common ${ \mathcal { C } } ^ { 1 }$ law h has $h _ { x } = 0$ and

$$
h _ { y } + x ^ { k } h _ { z } = 0
$$

on a product box. The first identity makes $h _ { y } , h _ { z }$ independent of x. Taking $x = 0$ and then any nonzero x in the box forces $h _ { y } = h _ { z } = 0$ . Thus every common law is locally constant, and any finite common-law initial level contains a whole neighborhood. At fixed output $( a , 0 )$ the parameter states $( a , - a ^ { k } z , z )$ have $M _ { 1 2 } = k a ^ { k - 1 } z $ , which varies for arbitrarily small nonzero a and z. One represented mobility cannot serve that actual level.

Theorem 6.3 (One fixed strongly convex quadratic family). Fix an integer $k \geq 2 ,$ , a number $0 < \gamma < 1$ , and

$$
\begin{array} { r } { H = \left( \begin{array} { l l } { 1 } & { \gamma } \\ { \gamma } & { 1 } \end{array} \right) , \qquad f _ { b } ( w ) = \frac 1 2 w ^ { T } H w - b ^ { T } w , \quad b \in \mathbb { R } ^ { 2 } . } \end{array}
$$

For $\phi _ { k } ( x , y , z ) = ( x , y + x ^ { k } z )$ initialized at the origin, $( \mathrm { O } ) _ { \{ f _ { b } : b \in \mathbb { R } ^ { 2 } \} }$ fails. More precisely, for any fixed $0 ~ < ~ c _ { 1 } ~ < ~ c _ { 2 }$ there are $a _ { * } \ > \ 0$ and analytic parameter functions $b _ { i } ( a )$ near $a \ = \ 0$ , with $b _ { i } ( 0 ) = ( 1 / c _ { i } , 0 )$ , such that the actual flows satisfy, for every $0 < a < a _ { * }$

$$
\phi _ { k } ( \theta _ { i } ( c _ { i } a ) ) = ( a , 0 ) ,
$$

and

$$
\left[ M ( \theta _ { 2 } ( c _ { 2 } a ) ) - M ( \theta _ { 1 } ( c _ { 1 } a ) ) \right] _ { 1 2 } = - \frac { \gamma k ^ { 2 } ( c _ { 2 } - c _ { 1 } ) } { 2 ( k + 1 ) ( k + 2 ) } a ^ { 2 k + 1 } + O ( a ^ { 2 k + 2 } ) \neq 0 .\tag{11}
$$

Both complete parameter paths converge uniformly to the origin as $a \downarrow 0$ . The Hessian is fixed for both losses, with eigenvalues $1 - \gamma$ and $1 + \gamma ;$ the translations and endpoint times difer.

Proof. See Appendix E.2.

There is a second check on the coeficient in (11). For the shooting trajectories constructed in Appendix E.2, fix $c \in \{ c _ { 1 } , c _ { 2 } \}$ and write $s = t / ( c a )$ and $x = a X ( s )$ . Write $F ( x ) = x ^ { k }$ , set $v = y + F ( x ) z$ , and let $r _ { 1 } , r _ { 2 }$ be the negative output-gradient components along any horizontal trajectory. Then

$$
{ \dot { x } } = r _ { 1 } + F ^ { \prime } z r _ { 2 } , \quad { \dot { z } } = F r _ { 2 } , \quad { \dot { v } } = F ^ { \prime } z { \dot { x } } + ( 1 + F ^ { 2 } ) r _ { 2 } .
$$

For a path with increasing x, elimination of time gives

$$
{ \frac { d } { d x } } \{ z { \sqrt { 1 + F ^ { 2 } } } \} = { \frac { F } { \sqrt { 1 + F ^ { 2 } } } } v _ { x } .
$$

Since the shooting endpoints have $v ( 0 ) = v ( a ) = 0$ , integration by parts gives the exact hiddencoordinate identity

$$
z ( a ) = - { \frac { 1 } { \sqrt { 1 + F ( a ) ^ { 2 } } } } \int _ { 0 } ^ { a } { \frac { v ( x ) F ^ { \prime } ( x ) } { ( 1 + F ( x ) ^ { 2 } ) ^ { 3 / 2 } } } d x .
$$

The shooting solution has $\dot { x } = 1 / c + O ( a ) > 0$ uniformly. From the limiting equations and smooth inversion of $s \mapsto X ( s )$

$$
v ( x ) = { \frac { c \gamma } { 2 } } x ( a - x ) + O ( a ^ { 3 } ) , \qquad 0 \leq x \leq a .
$$

The uniform remainder contributes $O ( a ^ { k + 3 } )$ to $z ( a )$ ; denominator corrections contribute $O ( a ^ { 3 k + 2 } )$ which is absorbed for $k \geq 2$ . The leading integral again gives $- c \gamma k a ^ { k + 2 } / [ 2 ( k + 1 ) ( k + 2 ) ]$

Corollary 6.4 (Ordinary squared error with one fixed readout). Let $W = H ^ { 1 / 2 }$ be the symmetric positive-definite square root of the fixed Hessian in Theorem $6 . 3 ,$ and fix the representation $\psi _ { k } =$ $W \phi _ { k }$ . Its ordinary entire-mobility closure fails even when the tested losses are only

$$
\begin{array} { r } { \ell _ { \tau } ( w ) = \frac 1 2 \| w - \tau \| _ { 2 } ^ { 2 } , \qquad \tau \in \mathbb { R } ^ { 2 } . } \end{array}
$$

It has representation rank two and gradient Lie rank three everywhere, and $D M _ { \psi _ { k } } ( 0 ) = 0$ . The compared targets and endpoint times difer; the representation is fixed.

## Proof. See Appendix E.3.

Remark 6.5 (Scope and a positive control). The readout in Corollary 6.4 is a fixed change of represented features, not a change in the Euclidean parameter metric. It does not establish identity-Hessian failure on the original untransformed $\phi _ { k }$ , a collision at equal times for the same fixed Hessian, or a failure for one fixed target. Choosing $\gamma = 1 / 2$ fixes Hessian condition number three. The corresponding projection representation $\psi _ { 0 } = W ( x , y )$ has constant mobility H, and therefore admits closure for every loss. Linear tests also pass for $\psi _ { k }$ by Proposition 6.1. These statements distinguish the loss families; they do not constitute a new optimizer, training gain or general quadratic-family classification.

Proposition 6.6 (Quadratic tests force section regularity). Suppose $\phi$ is smooth and has constant ambient rank $q > 0$ on a neighborhood of $\theta _ { 0 }$ . Set $z _ { 0 } = \phi ( \theta _ { 0 } )$ and test only

$$
\begin{array} { r } { f _ { b } ( z ) = \frac 1 2 \| z - z _ { 0 } \| _ { 2 } ^ { 2 } + b ^ { T } ( z - z _ { 0 } ) , \qquad b \in \mathbb { R } ^ { d } . } \end{array}
$$

If $( \mathrm { O } ) _ { \{ f _ { b } : b \in \mathbb { R } ^ { d } \} }$ holds, then its arbitrary common K agrees on a local represented image with a smooth representative. That representative has a local ambient extension, and after shrinking the image it has a global smooth extension. $I f \phi$ is analytic, the representative on the image and its local extension can be analytic. This is an image-section conclusion, not an invariant-level factorization.

Proof. See Appendix E.4.

Proposition 6.6 identifies K on an actual endpoint section. It says nothing by itself about other parameter points of the same represented fiber. Proposition 6.1 and Remark 6.2 explain why section regularity alone cannot substitute for the full arbitrary-loss necessity arguments. The supporting results here remove a loss-family limitation of the explicit negative control while retaining that distinction. They imply no numerical value of $a _ { * } .$ , empirical convergence order, universal quadraticfamily converse, or learning-performance claim.

## 7 Related Work

Intrinsic dynamics and initialized tests. Marcotte, Peyré, and Gribonval [1] introduce the ordinary intrinsic-dynamic and invariant-metric notions used here, together with a stronger definition for piecewise $\mathcal { C } ^ { 2 }$ autonomous losses. Their Theorem 3.8 gives a positive relaxed-balance branch. Their Theorem 3.9 assumes full-rank factors and hidden width at most max $( n , m )$ , and excludes their strong intrinsic dynamic property outside relaxed balance. Proposition 3.10 already resolves $n = m = 1$ at arbitrary width. Our wide classification addresses ordinary closure with individually full-row factors; its scalar subcase is inherited. Their Theorem 2.14 starts from a differentiable invariant-level metric and Theorem 2.17 from recoverability. The converse here starts from an unrestricted K and fixed-start autonomous tests, including specified cases with Lie rank larger than representation rank.

The conservation-law framework of Marcotte, Gribonval, and Peyré [2], Theorem 3.3 and Propositions H.3/H.5, supplies the Lie distribution, the count of independent common laws, and matrix factor-orbit ingredients. Their Proposition 3.7 concerns recovery from invariants and representation. The additional necessity step here passes from actual autonomous endpoint patches to mobility constraints on Gram fibers. A factor-orbit dimension count alone does not supply that step.

Known positive matrix flows and larger closed states. Dominé et al. [3], Theorem 5.2 and Appendix C.2.1, recover the factors up to hidden orthogonal transformations, and thereby determine the two mobility Gram blocks, under their λ-balanced and full-rank-product assumptions, giving the square-root structure. Bah et al. [4], Lemma 2 and Propositions 10–11, give the balanced product flow and its Riemannian interpretation on fixed-rank matrix manifolds. These are positive reduction results. The initialization exclusion and original ambient invariant-level arguments in Theorem 3.1 answer a diferent necessity question. Proposition 3.10 explicitly matches our positive flows to a known narrow family.

Zhang and Peyré [5], Proposition 2.1, close the flow on the full weight covariance, retaining the hidden Gram blocks. The state here is only $Z = U V ^ { \top }$ . Importantly, their Proposition E.2 also gives predictor-only square-root closure on the hyperbolically isotropic leaf $C J C = \alpha ^ { 2 } J$ after the appropriate width normalization; this is the $a = b$ part of our full-rank positive branch. For the unequal pair, the fixed shift $D = C - ( a - b ) J / 2$ satisfies $D J D = ( ( a + b ) / 2 ) ^ { 2 } J$ and has the same covariance flow because ${ \cal H } _ { G } J + J { \cal H } _ { G } = 0$ . Thus this positive reduction remains a known-family consequence. Their Remark E.3 shows that conserved spectral values generally do not determine predictor velocity. That two-state example does not itself establish ordinary failure from one fixed initialization. We make no novelty claim for covariance closure or the positive square-root formula, and do not infer a global priority claim from this comparison.

Geometric criteria and mirror flows. Li et al. [6], Theorem 4.9, obtain mirror flow from a regular commuting parametrization under their flow-domain assumptions. Theorem 4.10 gives a necessary Lie-bracket condition when a mirror representation works for all initializations and timedependent losses. Mirror flow requires an integrable inverse-Hessian metric, whereas properties (O) and (I) allow general mobility. Our initialized autonomous-loss premise and invariant-level conclusion therefore have diferent quantifiers and targets. Horstmeyer and Atay [7], Proposition 1 and Theorem 4, characterize exact lumpability by all-state projectability and invariance of the vertical distribution under Lie diferentiation. Their criterion applies to a fixed vector field on smooth manifolds with the stated submersion/fiber hypotheses. Applying it backwards here would first require proving nearby-fiber consistency from initialized tests; this is precisely the extra implication we study.

Achour, Kohn, and Rauhut [8], Theorem 4 and Corollary 6 in the April 2026 revision, show that for one-dimensional linear convolutional networks with the specified strides larger than one, fixed norm-diference invariants determine the NTK on the smooth neuromanifold, with finite fibers. Their broader work also treats higher-dimensional convolutions. The finite-fiber recovery geometry difers from the matrix-factorization fibers and $p > q$ regimes considered here. We do not extend the matrix classification to convolutional architectures.

Optimization and implicit bias. Saxe et al. [9] provide exact deep-linear learning solutions for structured settings. Arora et al. [10] interpret depth-induced preconditioning; Gunasekar et al. [11] and Arora et al. [12] study implicit regularization in matrix factorization. These motivate examining represented dynamics, but a convergence or bias result under one loss is not an all-loss mobility classification. Min et al. [13] bound convergence using imbalance and product singular values. Nordio [14] studies training-set dynamics with fixed readout and quadratic loss, retaining a hierarchy of Gram operators at depth. Castin et al. [15] use balancing to change LoRA conditioning. Our results concern continuous Euclidean parameter gradient flow and claim neither their discretetime guarantees nor an improvement in learning performance.

## 8 Discussion and Limitations

Theorem 3.1 determines when the prediction matrix can serve as a complete mobility state for every autonomous loss from one fixed initialization. In the nonscalar wide model, its generic nonclosure corollary rules out that exact description under absolutely continuous initialization. It leaves open approximate closure, loss-specific reductions, and dynamics on larger states such as a full covariance.

The factors must be individually full row rank and $r > \operatorname* { m a x } ( n , m )$ . Individually deficient initializations are not classified. The general bridge requires representation and gradient-Lie ranks constant on an ambient open neighborhood; singular width reduction does not transfer that hypothesis. The smooth case $q = 2 , p \geq 4$ remains unresolved. All-order flatness alone cannot replace equality of a smooth function on a neighborhood. The fixed-Hessian quadratic witness is theoretical: no numerical $a _ { * }$ , quadratic grid, or empirical convergence order is asserted. The historical engineered-loss check in Appendix G is a diferent test. Continuous-time mobility equality does not imply equality of finite-step updates.

Use of AI-assisted tools. The research records document AI-assisted development of mathematical arguments, source comparisons, code, and manuscript text, including automated critiques. AI tools were also used to prepare this release, check algebra and references, and inspect the build. These checks do not constitute verification by human experts. The author is responsible for the mathematical claims, citations, experimental reporting, and the decision to release the work. No AI system is listed as an author.

## References

[1] Sibylle Marcotte, Gabriel Peyré, and Rémi Gribonval. Intrinsic training dynamics of deep neural networks. arXiv:2508.07370v2; ICLR 2026, 2026. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2508.07370v2</sub>.

[2] Sibylle Marcotte, Rémi Gribonval, and Gabriel Peyré. Abide by the Law and Follow the Flow: Conservation Laws for Gradient Flows. arXiv:2307.00144v2; NeurIPS 2023, revised 2024, 2023. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2307.00144v2</sub>.

[3] Clémentine C. J. Dominé, Nicolas Anguita, Alexandra M. Proca, Lukas Braun, Daniel Kunin, Pedro A. M. Mediano, and Andrew M. Saxe. From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks. arXiv:2409.14623v2; ICLR 2025, 2025. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2409.14623v2</sub>.

[4] Bubacarr Bah, Holger Rauhut, Ulrich Terstiege, and Michael Westdickenberg. Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers. arXiv:1910.05505v5, 2020. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>1910.05505v5</sub>.

[5] Stephen Y Zhang and Gabriel Peyré. Global Exponential Convergence of Two-Layer Linear Network Training. arXiv:2610.09356v1, 2026. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2610.09356v1</sub>.

[6] Zhiyuan Li, Tianhao Wang, Jason D. Lee, and Sanjeev Arora. Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent. arXiv:2207.04036v1, 2022. <sub>https:</sub> //<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2207.04036v1</sub>.

[7] Leonhard Horstmeyer and Fatihcan M. Atay. Characterization of Exact Lumpability for Vector Fields on Smooth Manifolds. arXiv:1607.01237v2, 2016. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>1607.01237v2</sub>.

[8] El Mehdi Achour, Kathlén Kohn, and Holger Rauhut. The Riemannian Geometry Associated to Gradient Flows of Linear Convolutional Networks. arXiv:2507.06367v2, 2026. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/ <sub>2507.06367v2</sub>.

[9] Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv:1312.6120v3, 2014. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>1312.</sub> <sub>6120v3</sub>.

[10] Sanjeev Arora, Nadav Cohen, and Elad Hazan. On the optimization of deep networks: Implicit acceleration by overparameterization. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 244–253. PMLR, 10–15 Jul 2018.

[11] Suriya Gunasekar, Blake Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nathan Srebro. Implicit Regularization in Matrix Factorization. arXiv:1705.09280v1, 2017. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/ <sub>1705.09280v1</sub>.

[12] Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit Regularization in Deep Matrix Factorization. arXiv:1905.13655v3, 2019. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>1905.13655v3</sub>.

[13] Hancheng Min, Rene Vidal, and Enrique Mallada. On the convergence of gradient flow on multilayer linear models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 24850–24887. PMLR, 23–29 Jul 2023.

[14] Claudio Nordio. Learning Dynamics Reveal a Hierarchy of Weight-Induced Layerwise Gram Metrics. arXiv:2606.09744v4, 2026. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2606.09744v4</sub>.

[15] Valérie Castin, Kimia Nadjahi, Pierre Ablin, and Gabriel Peyré. Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence. arXiv:2605.31484v1, 2026. <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2605.</sub> <sub>31484v1</sub>.

## A Proofs of the common local constructions

## A.1 Proof of Lemma 2.3

Proof. Choose a smooth orthonormal normal frame $e _ { a } ( t )$ along the extended interval. In a tubular neighborhood write $\begin{array} { r } { \Psi ( t , y ) = \gamma ( t ) + \sum _ { a } y _ { a } e _ { a } ( t ) } \end{array}$ and define

$$
A ^ { \prime } ( t ) = \langle g ( t ) , \dot { \gamma } ( t ) \rangle , \qquad f _ { \mathrm { l o c } } ( \Psi ( t , y ) ) = A ( t ) + \sum _ { a } y _ { a } \langle g ( t ) , e _ { a } ( t ) \rangle .
$$

Tangential compatibility and the prescribed normal derivatives give grad $f _ { \mathrm { l o c } } ( \gamma ( t ) ) = g ( t )$ . For $d = 1$ there are no normal variables. Multiply by a smooth cutof supported inside the extended tube and identically one near the original compact arc, and extend by zero to $\mathbb { R } ^ { d }$ . This is one global smooth loss. Uniqueness for its locally Lipschitz parameter ODE identifies the prescribed path with the actual autonomous trajectory. The slight extension prevents endpoint regularity problems. The scalar compatibility also gives $A ^ { \prime } = - g ^ { T } M g = - \| \dot { \theta } \| ^ { 2 }$ along the curve; we have not prescribed inconsistent values around a projected loop. □

## A.2 Proof of Lemma 2.4

Proof. Choose $g _ { c }$ with $g _ { c } ^ { T } M ( \theta _ { 0 } ) g _ { c } > 0 .$ . On a smaller state box and a bounded convex open ball B about $g _ { c } .$ , continuity gives $g _ { c } ^ { T } M ( \theta ) g \geq c > 0$ for all $g \in B$ . Thus along these controls

$$
\frac { d } { d t } \langle g _ { c } , \phi ( \theta ( t ) ) \rangle = - g _ { c } ^ { T } M ( \theta ( t ) ) g ( t ) \leq - c .
$$

Smooth controlled projected paths are regular and embedded. A compact state box and uniform speed bound give a short horizon ε keeping every finite word of controls in B with total duration less than $\varepsilon$ inside the chosen neighborhood. The bound depends on total time, not the number of switches. Diferences of members of B span all controls, so their Lie values still span $T L$

Let Q be the maximum diferential rank of all positive-duration word endpoint maps from $\theta _ { 0 }$ with total time less than $\varepsilon .$ Integer boundedness ensures a word and a strictly positive duration vector attain $Q$ . On a small duration patch its rank is constantly $Q .$ giving an embedded local image $P .$ Append a short positive segment of any fixed $Y _ { g } , \ g \ \in \ B$ . The augmented endpoint map has rank at most $Q$ by maximality. Its $( Q + 1 )$ )-minors remain zero as the appended duration decreases to zero, while the original Q-minor persists. The appended derivative is $Y _ { g }$ , so $Y _ { g }$ is tangent to $P$ throughout a local patch. Brackets of tangent fields are tangent. The Lie spanning hypothesis forces $Q \geq$ dim $L ,$ hence equality. This gives a word endpoint image with nonempty interior in $L ,$ without claiming that it contains $\theta _ { 0 }$ or arbitrary return fibers.

Insert between successive fixed controls smooth flat-ended convex ramps of one fixed duration $\delta ,$ keeping the plateau durations as parameters. On a compact neighborhood, uniform field and state-Jacobian bounds imply for each ramp map $T _ { \delta }$

$$
\| T _ { \delta } - I \| \le C \delta , \qquad \| D T _ { \delta } - I \| \le e ^ { C \delta } - 1 .
$$

These follow from the flow integral and variational equations; no bound on the control’s time derivatives is needed. Finite composition yields convergence of the endpoint maps in state and duration derivatives. A full-rank minor therefore persists for one suficiently small positive $\delta ,$ with the extra times within the strict horizon margin. The smoothed family itself, not merely its limit, has an L-open endpoint image. The clock inequality still holds and each control is constant near the outer endpoints, so slight extensions remain embedded. Lemma 2.3 realizes each by one global autonomous smooth loss. The common K in (O) then gives the identity on the endpoint patch. No derivative or limit of an unrestricted K has been used. □

## A.3 Proof of Lemma 2.5

Proof. Constant rank gives a single local embedded image S of dimension q. Fix a suficiently short $\tau > 0$ . For the global linear loss $f _ { g } ( z ) = g ^ { T } z$ , let $\theta _ { g } ( \tau )$ be its actual endpoint from $\theta _ { 0 }$ . Near $g = 0$ all paths remain in the ordinary neighborhood, and smooth dependence gives

$$
D _ { g } \theta _ { g } ( \tau ) | _ { g = 0 } = - \tau D \phi ( \theta _ { 0 } ) ^ { T } , \qquad D _ { g } \phi ( \theta _ { g } ( \tau ) ) | _ { g = 0 } = - \tau M ( \theta _ { 0 } ) .
$$

Restricted to the q-dimensional space im $D \phi ( \theta _ { 0 } )$ , the second derivative is an isomorphism onto $T _ { z _ { 0 } } S .$ . The inverse function theorem gives a smooth endpoint section $s : S _ { 0 } \to \Omega$ with $\phi ( s ( z ) ) = z$ on an image neighborhood $S _ { 0 }$ . Property (O) forces $K ( z ) = M ( s ( z ) )$ there. Flatten $S$ in target coordinates and extend this map constantly in the normal coordinates to obtain $K _ { \mathrm { l o c } }$ . Shrink the parameter neighborhood so its whole image lies in $S _ { 0 }$ . Analytic dependence of the linear-loss flow and the analytic inverse function theorem give the analytic version. Of-image values of the original K are irrelevant. □

## A.4 Proof of Lemma 2.6

Proof. In a constant-rank product chart on $L ,$ the representation depends only on $q$ coordinates and the assumed derivative identity makes M independent of the other coordinates. Restrict to a product with connected fibers to factor M through the local embedded image. Extend normally in target coordinates to an open $W \ni z _ { 0 } ,$ , obtaining $K _ { \mathrm { l o c } }$ . Choose a smooth cutof $\chi$ compactly supported in $W$ , equal to one near $z _ { \mathrm { 0 } }$ , and define globally

$$
\overline { { K } } = \chi K _ { \mathrm { l o c } } + ( 1 - \chi ) I _ { d } ,
$$

with the first term extended by zero of its support. Shrink the Frobenius product box so its entire initial level lies in the factorization patch and its represented image lies where $\chi = 1$ . This gives the actual local-level identity and global smooth target extension. The argument is local and does not identify disconnected global fibers. The analytic local descent follows in analytic charts; the final cutof is smooth. □

## B Proofs of the wide matrix classification

## B.1 Proof of Lemma 3.2

Proof. The boosts $H _ { E _ { i j } }$ , together with their commutators, span all matrices

$$
T = \left[ \begin{array} { l l } { { P } } & { { G } } \\ { { G ^ { \top } } } & { { Q } } \end{array} \right] , \qquad P ^ { \top } = - P , \quad Q ^ { \top } = - Q .
$$

Indeed, holding the bottom index fixed in two boosts produces, up to sign, each elementary skew matrix in the top block; holding the top index fixed produces each elementary skew matrix in the bottom block. A block of dimension one requires no skew generators.

For the present J-orthogonal linear action, we use the classical local orbit theorem on a homogeneous orbit restricted to an embedded local orbit chart, with tangent $\{ T F : T \in \mathfrak { o } ( n , m ) \}$ , and the short flows of a Lie-generating family generate its connected local orbit. Only a local orbit chart is needed, not global closedness of the orbit. For completeness its tangent rank is computed directly. The map $T \mapsto J T$ identifies ${ \mathfrak { o } } ( n , m )$ with the skew-symmetric matrices. If $W = \mathrm { i m } F$ , the condition $T F = 0$ says that the corresponding alternating form annihilates W in either argument.

These are exactly the alternating forms on $\mathbb { R } ^ { N } / W$ ; their dimension is $\binom { N - s } { 2 }$ . Subtracting from dim $\begin{array} { r } { \mathfrak { o } ( n , m ) = \binom { N } { 2 } } \end{array}$ proves (8). Its rank is constant along the orbit, since left multiplication and conjugation by a group element are invertible.

A common $C ^ { 1 }$ conservation law has zero derivative along each $X _ { G }$ , so it is constant along every short boost flow, in either time direction. The local orbit statement therefore makes it constant on the connected local orbit. Alternatively one may use finite boost products and their commutator limits, passing invariance through continuity. No second derivative of the conservation law is taken. □

## B.2 Proof of Lemma 3.3

Proof. Write $\mathcal { L }$ for the connected local factor orbit. It is a locally embedded invariant leaf and the generated Lie values span its tangent space by Lemma 3.2. This is a statement on the orbit, even if ambient Lie rank jumps at $F _ { 0 }$ . Moreover

$$
\langle G , M _ { F } [ G ] \rangle = \| G ^ { \top } U \| _ { \mathrm { F } } ^ { 2 } + \| G V \| _ { \mathrm { F } } ^ { 2 } .
$$

For $F _ { 0 } \neq 0$ this positive semidefinite operator is nonzero. Lemma 2.4, applied on this orbit, supplies a nonempty -open patch of endpoints of actual single global smooth represented losses, on which

$$
M _ { F } = K ( \phi ( F ) ) .
$$

Its hypotheses require regularity along this invariant leaf, not constant ambient rank at the singular factor.

Choose a point in that patch where the rank of $D ( \phi | _ { \mathcal { L } } )$ is maximal within the patch. A nonzero maximal minor persists, so the rank is constant on a smaller open patch. The constant-rank theorem supplies actual local fiber curves there. Equality through the same, possibly irregular K makes M constant on these curves. Consequently every orbit tangent with $\delta Z = 0$ has $\delta M = 0$ This diferentiates the smooth M along a fiber, not the irregular function K.

The Gram map $F \mapsto F F ^ { \top }$ intertwines the factor action and the congruence action. Every Gram-orbit tangent $T C + C T ^ { \top }$ lifts to the factor tangent $T F$ . Its diferential rank is constant along the factor orbit by invertibility of both actions; its tangent image is the Gram-orbit tangent. Lift any such tangent

$$
\delta C = \left[ \begin{array} { c c } { { \delta A } } & { { 0 } } \\ { { 0 } } & { { \delta B } } \end{array} \right]
$$

to a factor-orbit tangent. The preceding fiber statement gives $\delta A G + G \delta B = 0$ for every G. Testing matrix units first eliminates all of-diagonal entries and then gives

$$
\delta A = t I _ { n } , \qquad \delta B = - t I _ { m }
$$

for one scalar t. The congruence orbit preserves tr $( J C )$ , because $Q ^ { \top } J Q = J$ implies tr $\cdot ( J Q C Q ^ { \top } ) =$ $\mathrm { t r } ( J C )$ . Therefore $0 = \operatorname { t r } ( J \delta C ) = N t .$ , so $\delta C = 0$ . The Gram tangent projects injectively to the nm-dimensional Z-space at the chosen point. Its dimension is constant on the orbit, proving $d _ { C } \leq n m$ at $C _ { 0 }$

If (I) holds, it implies (O) as noted above. One can also work directly on the local orbit, where every chosen common conservation law is constant, and diferentiate the $C ^ { 1 }$ represented factorization. Thus no constant-rank hypothesis on Dh is needed. □

## B.3 Proof of Lemma 3.4

Proof. The diferential of $F \mapsto S$ is onto the symmetric matrices. For any symmetric E, the choice

$$
\delta { \cal F } = \scriptstyle { \frac { 1 } { 2 } } J { \cal F } ( { \boldsymbol { F } } ^ { \top } { \boldsymbol { F } } ) ^ { - 1 } { \boldsymbol { E } }
$$

gives $\boldsymbol { F } ^ { \intercal } J \boldsymbol { \delta F } = \boldsymbol { E } / 2$ and $\delta S = E$ . Thus its kernel has dimension $N s - s ( s + 1 ) / 2$ , exactly (8).   
Every factor-orbit tangent lies in this kernel, so the two tangent spaces coincide.

The kernel of the diferential of $F \mapsto F F ^ { \top }$ is $\delta F = F \Omega$ with $\Omega ^ { \top } = - \Omega$ . To prove the nontrivial direction, project $\delta C = 0$ onto $W ^ { \perp }$ , where $W =$ im F, and multiply on the right by $F ( F ^ { \top } F ) ^ { - 1 }$ This gives $P _ { W ^ { \perp } } \delta F = 0$ , so $\delta F = F \Omega$ . Then $F ( \Omega + \Omega ^ { \top } ) F ^ { \top } = 0$ and a left inverse of $F$ makes Ω skew.

For these variations,

$$
\delta S = \Omega ^ { \top } S + S \Omega = S \Omega - \Omega S .
$$

The Gram diferential kernel restricted to the orbit is therefore exactly the skew matrices commuting with S. An orthogonal spectral decomposition of this real symmetric matrix makes that kernel the direct sum of skew matrices on its eigenspaces, of dimension $\textstyle \sum _ { j } { \binom { c _ { j } } { 2 } }$ . Subtraction proves (9), with the null eigenspace included. □

## B.4 Proof of Lemma 3.5

Proof. Let $( \alpha , \beta , \kappa )$ be the inertia of S. Thus $\alpha + \beta + \kappa = s$ . By Sylvester’s law of inertia, remove a nondegenerate subspace of signature $( \alpha , \beta )$ from im F. Its J-orthogonal complement has signature $( n - \alpha , m - \beta )$ and contains the κ-dimensional totally isotropic radical. Projection of a totally isotropic subspace into either definite summand is injective. Hence $\kappa \leq \operatorname* { m i n } ( n - \alpha , m - \beta )$ . Put

$$
u = n - \alpha - \kappa \ge 0 , \qquad v = m - \beta - \kappa \ge 0 .
$$

The strict width bounds give $\beta > u , \alpha > v$ , and $N - s = \kappa + u + v > 0$ . In particular $\alpha , \beta > 0$ Equal eigenvalues cannot belong to diferent signs. Grouping each sign, and the zero cluster, into a single cluster maximizes the possible centralizer dimension:

$$
\sum _ { j } { \binom { c _ { j } } { 2 } } \leq { \binom { \alpha } { 2 } } + { \binom { \beta } { 2 } } + { \binom { \kappa } { 2 } } .
$$

Substitution into (9), with $n = \alpha + \kappa + u$ and $m = \beta + \kappa + v$ , gives

$$
\begin{array} { c } { d _ { C } - n m \geq \alpha u + \beta v + \kappa ( \alpha + \beta ) - u v } \\ { = u ( \alpha - v ) + \beta v + \kappa ( \alpha + \beta ) > 0 . } \end{array}
$$

If $u > 0$ , the first term is positive; if $v > 0$ , the second is positive. If both vanish, $\kappa > 0$ and the last term is positive. These cover every case, including a nonzero radical. □

## B.5 Proof of Lemma 3.6

Proof. Since $s = L _ { 0 }$ , direct substitution into (9) gives

$$
d _ { C } - n m = \binom { s } { 2 } - \sum _ { j } \binom { c _ { j } } { 2 } = \sum _ { i < j } c _ { i } c _ { j } .
$$

It is zero precisely when $S$ has one eigenvalue value. As $\boldsymbol { C J C } = \boldsymbol { F S F } ^ { \top }$ , scalar $S$ implies $C J C = \lambda C ;$ a left inverse of $F$ proves the converse.

If $n > m$ , then $s = n$ , the reduced $U$ is square invertible, and a nonzero $x \in$ ker V satisfies $\lambda \| x \| ^ { 2 } = \| U x \| ^ { 2 } > 0$ . The case $m > n$ is symmetric and yields $\lambda < 0$ . If $n = m = s$ , both blocks are square invertible and $Z = U V ^ { \top }$ is invertible, without a required sign of λ. □

## B.6 Proof of Lemma 3.7

Proof. The symmetric matrix $L = C ^ { 1 / 2 } J C ^ { 1 / 2 }$ has inertia $( n , m )$ and is similar to $J C .$ . Choose an orthogonal O such that

$$
O ^ { \top } L O = \mathrm { d i a g } ( a _ { 1 } , \ldots , a _ { n } , - b _ { 1 } , \ldots , - b _ { m } ) , \qquad a _ { i } , b _ { j } > 0 ,
$$

and put $D = \mathrm { d i a g } ( a _ { 1 } , \ldots , a _ { n } , b _ { 1 } , \ldots , b _ { m } )$ . The matrix $R = C ^ { 1 / 2 } O D ^ { - 1 / 2 }$ satisfies

$$
R ^ { \top } J R = J , \qquad R D R ^ { \top } = C .
$$

We may therefore compute the congruence stabilizer at D. For $T = \left[ { \begin{array} { c c } { P } & { G } \\ { G ^ { \top } } & { Q } \end{array} } \right]$ in ${ \mathfrak { o } } ( n , m )$ , the equation $T D + D T ^ { \top } = 0$ requires

$$
[ P , \mathrm { d i a g } ( a _ { i } ) ] = 0 , \quad [ Q , \mathrm { d i a g } ( b _ { j } ) ] = 0 , \quad \mathrm { d i a g } ( a _ { i } ) G + G \mathrm { d i a g } ( b _ { j } ) = 0 .
$$

The last equation forces $G = 0$ , entry by entry. The stabilizer dimension is $\textstyle \sum _ { i } { \binom { c _ { i } } { 2 } } + \sum _ { j } { \binom { d _ { j } } { 2 } }$ . The classical orbit–stabilizer dimension formula for this smooth finite-dimensional action now gives

$$
d _ { C } = { \binom { N } { 2 } } - \sum _ { i } { \binom { c _ { i } } { 2 } } - \sum _ { j } { \binom { d _ { j } } { 2 } } ,
$$

which is (10). Each numerator $\begin{array} { r } { n ^ { 2 } - \sum _ { i } c _ { i } ^ { 2 } \ \mathrm { o r } \ m ^ { 2 } - \sum _ { j } d _ { j } ^ { 2 } } \end{array}$ is nonnegative, and vanishes exactly when that sign has one cluster.

We next establish suficiency on the actual ambient invariant level. The identity

$$
{ \cal H } _ { G } J + J { \cal H } _ { G } = 0
$$

shows by diferentiation of (4) that $S = F ^ { \top } J F$ is common to every represented-loss flow. For a full-row $F _ { 0 } , S _ { 0 }$ has rank N. If $F ^ { \top } J F = S _ { 0 }$ , then $F$ also has full row rank, and

$$
\mathrm { i m } \ : F ^ { \top } = \mathrm { i m } S _ { 0 } = \mathrm { i m } F _ { 0 } ^ { \top } .
$$

For example, $\operatorname { i m } ( F ^ { \top } J F ) = \operatorname { i m } F ^ { \top }$ follows because $F$ is onto and J is invertible. Equality of row spaces yields $F = Q F _ { 0 }$ for an invertible $Q ;$ multiplication by a right inverse of $F _ { 0 }$ and its transpose makes the equality of canonical invariants equivalent to $Q ^ { \top } J Q = J$ . Conversely every such $Q$ preserves that invariant. Thus the canonical level is precisely the J-orthogonal factor orbit, not only its tangent space.

Since $J C _ { 0 }$ is diagonalizable with the two stated values, it obeys $( J C _ { 0 } - a I ) ( J C _ { 0 } + b I ) = 0$ Multiplication by $J$ gives

$$
C _ { 0 } J C _ { 0 } - ( a - b ) C _ { 0 } - a b J = 0 .\tag{12}
$$

This identity is preserved by J-orthogonal congruence, using also $Q J Q ^ { \top } = J$ . Put $c = a - b$ and $\sigma = ( a + b ) / 2$ . On the whole canonical level its diagonal blocks give

$$
A ^ { 2 } - c A = Z Z ^ { \top } + a b I _ { n } , \qquad B ^ { 2 } + c B = Z ^ { \top } Z + a b I _ { m } .
$$

The block Grams are positive definite. Diagonalizing either Gram also diagonalizes the polynomial on the other side. Each scalar quadratic has exactly one positive root, because the product of its roots is negative. Since $a b + c ^ { 2 } / 4 = \sigma ^ { 2 }$ , this yields

$$
A = c I _ { n } / 2 + \sqrt { \sigma ^ { 2 } I _ { n } + Z Z ^ { \top } } , \qquad B = - c I _ { m } / 2 + \sqrt { \sigma ^ { 2 } I _ { m } + Z ^ { \top } Z } .
$$

The scalar shifts cancel in (3). The positive definite square-root map is smooth, and $\sigma > 0$ makes the displayed represented formula smooth for every $Z .$ . The finite entries of $S$ and this globally smooth map therefore prove (I). □

## B.7 Proof of Lemma 3.8

Proof. The right action $F \mapsto F Q$ is a Euclidean isometry preserving $U V ^ { \top }$ and $F F ^ { \top }$ , hence intertwines the gradient flows. Along any actual trajectory, (4) is a linear left equation with continuous coeficient $- H _ { \mathrm { g r a d } f ( Z ( t ) ) }$ . Its fundamental solution $P ( t )$ is invertible at every finite time in the existence interval, so $\dot { F } ( t ) = P ( t ) F _ { 0 }$ and ker $F ( t ) = \ker F _ { 0 }$ . The zero hidden columns remain zero and the other columns satisfy exactly the reduced system. Their Gram quantities and represented variables coincide.

If (O) holds in the original neighborhood, pull that neighborhood back under the continuous embedding $\boldsymbol { \widetilde { F } } \mapsto [ \boldsymbol { \widetilde { F } } , 0 ] \boldsymbol { Q } ^ { \intercal }$ . If it holds on a reduced neighborhood, take its inverse image under e ethe projection onto the first s columns of $F Q$ . This is an open original neighborhood, and the actual original trajectory always has zero remaining columns. The same K works in each direction for every loss. These arguments concern trajectories from the fixed initialization; they make no assertion about arbitrary ambient points on an invariant level. Finally an orthogonal column action and deletion of identically zero columns preserve each block’s row rank. □

## B.8 Proof of Lemma 3.9

Proof. Use the fixed hidden orthogonal coordinates of Lemma 3.8. Lemma 3.6 gives

$$
S _ { 0 } = \mathrm { d i a g } ( \lambda I _ { s } , 0 ) .
$$

Consider an arbitrary ambient $F = [ W , E ]$ on this exact canonical level. It satisfies

$$
\boldsymbol { W } ^ { \top } \boldsymbol { J } \boldsymbol { W } = \lambda \boldsymbol { I } _ { s } , \qquad \boldsymbol { W } ^ { \top } \boldsymbol { J } \boldsymbol { E } = 0 , \qquad \boldsymbol { E } ^ { \top } \boldsymbol { J } \boldsymbol { E } = 0 .\tag{13}
$$

First suppose $\lambda \neq 0$ . The sign restrictions in Lemma 3.6 show that $\lambda > 0$ entails $s = n ,$ and $\lambda < 0$ entails $s = m ;$ this includes the square case $n = m = s$ . If $\lambda > 0$ , the subspace spanned by W is positive definite of dimension $n ,$ exhausting the positive index of J. Its J-orthogonal complement is negative definite. Each column of $E$ lies in that complement and has zero J-square by (13), hence is zero. For $\lambda < 0$ the complementary form is positive definite and the same conclusion follows. Thus every ambient point of this invariant level has $E = 0$ and $C J C = \lambda C$

On a small neighborhood of $F _ { 0 }$ , both block Grams remain positive definite, by the assumed separate full row ranks. The block equations of $C J C = \lambda C$ are

$$
A ^ { 2 } - \lambda A = Z Z ^ { \top } , \qquad B ^ { 2 } + \lambda B = Z ^ { \top } Z .
$$

For every $\mu \geq 0 , x ^ { 2 } - \lambda x - \mu = 0$ has at most one positive root; any existing positive Gram eigenvalue selects it. The corresponding statement with λ applies to $B .$ . Functional calculus therefore yields

$$
\begin{array} { l } { A = \lambda I _ { n } / 2 + \sqrt { \lambda ^ { 2 } I _ { n } / 4 + Z Z ^ { \top } } , } \\ { B = - \lambda I _ { m } / 2 + \sqrt { \lambda ^ { 2 } I _ { m } / 4 + Z ^ { \top } Z } . } \end{array}\tag{14}
$$

Their shifts cancel. Since $\lambda \neq 0 .$ , the resulting $K _ { | \lambda | / 2 }$ is smooth on the entire represented space.

Now suppose $\lambda = 0$ . The sign restrictions force $n = m = s$ , and the full original $S _ { 0 }$ is zero. We do not discard or force the extra hidden columns to vanish. At every point of the actual canonical level,

$$
\begin{array} { r } { C J C = F ( \boldsymbol { F } ^ { \top } \boldsymbol { J } \boldsymbol { F } ) \boldsymbol { F } ^ { \top } = 0 . } \end{array}
$$

Consequently $A ^ { 2 } = Z Z ^ { \top }$ and $B ^ { 2 } = Z ^ { \top } Z$ . Both Grams remain positive definite on a suficiently small ambient neighborhood, so $Z$ is invertible and their positive roots give (14) with $\lambda = 0$ . The map $K _ { 0 }$ is smooth on ${ \mathrm { G L } } ( n , \mathbb { R } )$ . Choose a smooth cutof $\chi$ compactly supported in that open set, equal to one near $Z _ { 0 } .$ , and define globally

$$
\overline { { K } } ( Z ) = \chi ( Z ) K _ { 0 } ( Z ) + ( 1 - \chi ( Z ) ) I _ { \mathbb { R } ^ { n \times n } } ,
$$

where the first term is extended by zero of its compact support. This is globally smooth. Shrink the ambient parameter neighborhood so its represented image lies in the region $\chi = 1$ . In both cases the finite canonical entries are common smooth conservation laws, and the required identity holds on their actual initial level. This proves (I) in the original space. □

## B.9 Proof of Theorem 3.1

Proof of Theorem 3.1. Separate full row ranks imply $L _ { 0 } \le s \le \operatorname* { m i n } ( r , N )$ . Apply Lemma 3.8 to property (O). If $L _ { 0 } < s < N$ , the reduced factor is full column rank and Lemmas 3.3 and 3.5 contradict each other. If $s = L _ { 0 }$ , Lemmas 3.3 and 3.6 force (5) and its signs. If $s = N$ , the full-row calculation (10) forces (6). These exhaust every admissible joint rank.

Conversely Lemma 3.9 proves (I) in (5), and Lemma 3.7 proves it in (6). Both use the actual original canonical level, not a reduced-space tangent argument. Conservation gives (O). The mobility formulas were established in the same lemmas. □

## B.10 Proof of Proposition 3.10

Proof. Put

$$
P = \sqrt { \sigma ^ { 2 } I _ { n } + Z Z ^ { \top } } , \qquad Q = \sqrt { \sigma ^ { 2 } I _ { m } + Z ^ { \top } Z } .
$$

If $n \leq m$ , define

$$
U ^ { \prime } = ( P + \sigma I _ { n } ) ^ { 1 / 2 } , \qquad V ^ { \prime } = Z ^ { \top } ( P + \sigma I _ { n } ) ^ { - 1 / 2 } .
$$

The inverse is lawful since $P + \sigma I _ { n } \ge 2 \sigma I _ { n }$ . The product is $U ^ { \prime } V ^ { \prime \top } = Z$ . A singular-value decom position, including all zero singular values, gives

$$
\begin{array} { r l } & { U ^ { \prime } U ^ { \prime \top } = P + \sigma I _ { n } , V ^ { \prime } V ^ { \prime \top } = Z ^ { \top } ( P + \sigma I _ { n } ) ^ { - 1 } Z = Q - \sigma I _ { m } , } \\ & { { U ^ { \prime } } ^ { \top } U ^ { \prime } = P + \sigma I _ { n } , V ^ { \prime \top } V ^ { \prime } = P - \sigma I _ { n } . } \end{array}
$$

For a singular value $d ,$ the scalar identity used here is

$$
{ \frac { d ^ { 2 } } { \sqrt { \sigma ^ { 2 } + d ^ { 2 } } + \sigma } } = \sqrt { \sigma ^ { 2 } + d ^ { 2 } } - \sigma .
$$

It also handles the extra nullspace, where both sides are zero. The hidden imbalance is 2σ $I _ { n } ,$ and its two scalar Gram shifts cancel in the mobility.

If $n > m ,$ , the transposed construction is

$$
V ^ { \prime } = ( Q + \sigma I _ { m } ) ^ { 1 / 2 } , \qquad U ^ { \prime } = Z ( Q + \sigma I _ { m } ) ^ { - 1 / 2 } .
$$

It gives

$$
U ^ { \prime } U ^ { \prime \top } = P - \sigma I _ { n } , \quad V ^ { \prime } V ^ { \prime \top } = Q + \sigma I _ { m } , \quad U ^ { \prime \top } U ^ { \prime } - V ^ { \prime \top } V ^ { \prime } = - 2 \sigma I _ { m } ,
$$

and the same cancellation.

These scalar hidden invariants remain constant along each narrow factor flow. In the first construction $U ^ { \prime }$ stays invertible, since $U ^ { \prime \top } U ^ { \prime } \geq 2 \sigma I _ { n }$ . Its Gram equation has the unique positive solution $U ^ { \prime } U ^ { \prime \top } = P + \sigma I _ { n }$ ; the other block has the unique nonnegative solution $V ^ { \prime } V ^ { \prime \top } = Q - \sigma I _ { m }$ The latter remains unique even at zero singular values, because $x ^ { 2 } + 2 \sigma x - \mu = 0 , \mu \geq 0 ,$ , has a unique nonnegative root. The transposed construction has the identical argument. Hence the narrow flow retains exactly $K _ { \sigma }$ . For square invertible $Z$ and $\sigma = 0$ , set $U ^ { \prime } = P ^ { 1 / 2 } , V ^ { \prime } = Z ^ { \top } P ^ { - 1 / 2 }$ its hidden imbalance is zero and its Grams are $P , Q$ , giving the balanced formula locally.

For each common represented $C ^ { 2 }$ loss, the wide positive flow and these narrow factors therefore solve the same smooth represented ODE, with the same initial represented matrix. Local uniqueness identifies their represented trajectories on a common existence interval. This is a mobility and continuous-time trajectory identity, not equality of factor states or finite-step gradient descent. The scalar-balanced square-root dynamics are the known positive family; the construction supplies the exact matching comparison. □

## B.11 Proof of Corollary 3.11

Proof. Separate maximal row ranks and maximal joint rank hold almost surely: their failures lie in zero sets of nonzero minor polynomials. If $r < N$ , the almost-sure joint rank is $r ,$ strictly between $L _ { 0 }$ and $N$ , so Theorem 3.1 excludes closure. If $r \geq N$ , the almost-sure joint rank is $N$ . For $( n , m ) \neq ( 1 , 1 )$ , the positive spectral case forces a repeated eigenvalue of $J F F ^ { \top }$ . The discriminant

$$
P ( \boldsymbol { F } ) = \mathrm { d i s c } _ { t } \operatorname* { d e t } ( t I _ { N } - J \boldsymbol { F } \boldsymbol { F } ^ { \intercal } )
$$

is a polynomial in the factor entries and must vanish there. It is not identically zero: at $F =$ $[ \mathrm { d i a g } ( 1 , \ldots , N ) , 0 ]$ its signed diagonal eigenvalues are all distinct. The zero set of a nonzero real polynomial has Lebesgue measure zero, and hence probability zero under every such law. For $n = m = 1$ , the two factor rows are nonzero almost surely. Joint rank one automatically gives $C J C = \lambda C ;$ joint rank two automatically gives one positive and one negative signed spectral value. The theorem then gives closure, in agreement with the known scalar case [1, Proposition 3.10].

## C Proofs of the geometric characterization

## C.1 Proof of Lemma 4.1

Proof. Select finitely many bracket fields $V _ { 1 } , \ldots , V _ { p }$ independent at $\theta _ { 0 }$ . A nonzero frame minor stays nonzero on a smaller neighborhood; constant rank makes these fields a frame for $\mathcal { E }$ there. Their brackets belong pointwise to $\mathcal { E } ,$ and their frame coeficients are smooth, or analytic in the analytic case, by inversion of the fixed frame minor. Leibniz’s rule then makes this subbundle involutive. The corresponding smooth or analytic Frobenius theorem supplies coordinates $( x , h )$ in a product box, with ${ \mathcal { E } } =$ ker Dh. The components of $h$ annihilate every grad $\phi _ { i }$ , so they are common conservation laws. The set $h = h ( \theta _ { 0 } )$ in this box is a single connected plaque. If $p = D$ one may take h constant.

Because $\mathcal { H } \subset \mathcal { E } = T L$ , restriction of Dϕ to T L has image containing

$$
\mathrm { i m a g e } ( D \phi D \phi ^ { \mathsf { T } } ) = \mathrm { i m a g e } D \phi .
$$

The reverse containment is automatic. Thus the restriction has the same rank $q .$ . No full-outputrank assumption is involved. □

## C.2 Proof of Theorem 4.2 in the analytic case

Proof of Theorem $4 . 2$ in the analytic case. The metric property implies ordinary closure by conservation, so consider the converse. If $q = 0 , D \phi = 0$ locally, $M = 0$ , and the zero global mobility and a constant conservation law sufice. Assume $q > 0$ . Use the connected analytic plaque $L$ of Lemma 4.1. The clock endpoint construction in Lemma 2.4 gives a nonempty relatively open set $O \subset L$ on which

$$
M ( \theta ) = K ( \phi ( \theta ) ) ,
$$

with the same initially arbitrary K as in ordinary closure. Every vector in ker $D ( \phi | _ { L } )$ at a point of O is tangent to an actual local fiber curve, because the restriction has constant rank $q .$ . Along such a curve $\phi$ is constant, hence so is M. Therefore

$$
\ker D ( \phi | _ { L } ) \subset \ker D ( M | _ { L } ) \quad \mathrm { o n } \ O .\tag{15}
$$

This argument neither diferentiates K nor assumes its continuity.

Represent M by its $d ^ { 2 }$ entries. The augmented diferential $D _ { L } ( \phi , M )$ has rank $q$ on $O .$ In the connected analytic x-coordinate box of $L ,$ every $( q + 1 )$ )-minor of this diferential is analytic and vanishes on O. The real-analytic identity principle makes every such minor zero throughout that box. If $p = q$ , the upper bound is automatic since there are only $p$ columns. Since $D ( \phi | _ { L } )$ has rank q everywhere, the augmented diferential has rank exactly q everywhere. Thus the kernel inclusion above holds on the entire plaque, including near $\theta _ { 0 }$

For completeness, this diferential condition gives an equality on the actual local level, not just on its tangent. Apply the analytic constant-rank theorem to $\phi | _ { L }$ near $\theta _ { 0 }$ . In local source coordinates $( s , t ) \in \mathbb { R } ^ { q } \times \mathbb { R } ^ { p - q }$ it has the form $\phi ( s , t ) = \iota ( s )$ , where ι parametrizes a locally embedded analytic image S. Shrink to a product box so its t-fibers are connected. The kernel inclusion says $\partial _ { t } M = 0$ hence $M ( s , t ) = m ( s )$ . Taking a local section, for example $t = 0$ , constructs an analytic map k on S with $M = k \circ \phi$ on this whole smaller plaque neighborhood.

Flatten S in analytic ambient target coordinates and extend k constantly in the normal coordinates, obtaining $K _ { \mathrm { l o c } }$ on an open target neighborhood W of $\phi ( \theta _ { 0 } )$ . Choose a smooth $0 \leq \chi \leq 1$ with compact support in W and equal to one on a smaller neighborhood. The map

$$
\overline { { K } } ( z ) = \chi ( z ) K _ { \mathrm { l o c } } ( z ) + ( 1 - \chi ( z ) ) I _ { d }
$$

is globally smooth, interpreting $\chi K _ { \mathrm { l o c } }$ as zero outside W. No analyticity of the global cutof is asserted. Finally shrink the original Frobenius product box so its entire initial h-level lies in the factorization neighborhood and its represented image lies where $\chi = 1$ . This proves the exact local metric property. The h are local conservation laws; they need not extend to a larger originally specified parameter domain. □

## C.3 Proof of Lemma 4.3

Proof. A self-intersection at $s \neq t$ would require

$$
a = - \frac { w ( t ) - w ( s ) } { t - s } .
$$

Cover the of-diagonal square by countably many compact two-dimensional pieces bounded away from the diagonal. The secant map is smooth and Lipschitz on each piece. Its image has $q -$ dimensional Lebesgue measure zero when $q > 2$ , by the elementary covering estimate for a Lipschitz image of a two-dimensional set. Failure of immersion requires $a = - w ^ { \prime } ( t )$ ; this one-dimensional smooth image is also null. Choose arbitrarily small $a _ { j }$ outside their union. Each resulting curve is injective and regular; compactness makes it an embedded compact arc. The $C ^ { 1 }$ convergence is immediate, and compact containment gives the last claim. Nothing in this proof asserts density of embedded planar arcs when $q = 2$ □

## C.4 Proof of Lemma 4.4

Proof. Construct $K _ { \mathrm { l o c } }$ first, and shrink into its domain and the ordinary neighborhood. Constant rank gives one embedded represented image $S$ and local coordinates $\psi : S \to \mathbb { R } ^ { q }$ . Set $\pi = \psi \circ \phi .$ The kernels of $D \pi$ and $D \phi$ agree, so their Euclidean horizontal spaces agree. The smooth right inverse

$$
H ( \boldsymbol { \theta } ) = D \pi ( \boldsymbol { \theta } ) ^ { \mathsf { T } } [ D \pi ( \boldsymbol { \theta } ) D \pi ( \boldsymbol { \theta } ) ^ { \mathsf { T } } ] ^ { - 1 }
$$

maps base velocities to their unique horizontal lifts.

For a smooth horizontal path $\theta ( t )$ , put $w ( t ) = \pi ( \theta ( t ) )$ . It solves $\theta ^ { \prime } = H ( \theta ) w ^ { \prime }$ . Apply Lemma 4.3 and lift $w _ { j }$ from exactly $\theta _ { 0 }$ by $\theta _ { j } ^ { \prime } = H ( \theta _ { j } ) w _ { j } ^ { \prime }$ . Diferentiation gives $\pi ( \theta _ { j } ( t ) ) = w _ { j } ( t )$ . Standard ODE stability on a compact state neighborhood gives uniform convergence of lifts on $[ 0 , T ]$ . One explicit bound, while both lifts remain there, is

$$
\operatorname* { s u p } _ { t \leq T } \| \theta _ { j } ( t ) - \theta ( t ) \| \leq C \| w _ { j } ^ { \prime } - w ^ { \prime } \| _ { L ^ { 1 } } \exp \left( C \int _ { 0 } ^ { T } \| w _ { j } ^ { \prime } ( t ) \| d t \right) .
$$

A slightly larger compact neighborhood and continuation show that all large-j lifts exist on the full interval and remain in the required open set. Their represented arcs $z _ { j } = \phi ( \theta _ { j } ) = \psi ^ { - 1 } ( w _ { j } )$ are embedded and regular.

Here is the exact autonomous realization used for each arc. The constant-rank Moore–Penrose inverse $M ^ { \dagger }$ is smooth. Put $g _ { j } = - M ( \theta _ { j } ) ^ { \dagger } z _ { j } ^ { \prime }$ . Horizontality gives $- D \phi ^ { \mathsf { T } } g _ { j } = \theta _ { j } ^ { \prime }$ and

$$
\langle g _ { j } , z _ { j } ^ { \prime } \rangle = - \| \theta _ { j } ^ { \prime } \| ^ { 2 } < 0 .
$$

Slightly extend the regular embedded arc at its two ends. In tubular coordinates $\begin{array} { r } { z _ { j } ( t ) + \sum _ { b } y _ { b } e _ { b } ( t ) } \end{array}$ ， where the $e _ { b }$ form a smooth normal frame, prescribe

$$
A ^ { \prime } ( t ) = \langle g _ { j } ( t ) , z _ { j } ^ { \prime } ( t ) \rangle , \qquad f _ { \mathrm { l o c } } ( t , y ) = A ( t ) + \sum _ { b } y _ { b } \langle g _ { j } ( t ) , e _ { b } ( t ) \rangle .
$$

Its gradient on the arc is exactly $g _ { j } { \mathrm { : } }$ : the primitive supplies the tangential derivative, and the linear terms supply all normal derivatives. A cutof equal to one around the original compact arc and supported in this slightly extended tube yields one global smooth represented loss. ODE uniqueness identifies its actual parameter flow from $\theta _ { 0 }$ with $\theta _ { j }$ . The construction requires no uniform derivative bound over $j .$

Ordinary closure therefore gives $M ( \theta _ { j } ( t ) ) = K ( \phi ( \theta _ { j } ( t ) ) ) = K _ { \mathrm { l o c } } ( \phi ( \theta _ { j } ( t ) ) )$ . Pass to the limit through the separately forced smooth $K _ { \mathrm { l o c } }$ . This proves the asserted identity on the original horizontal path without assuming continuity of the original $K$ □

## C.5 Proof of Lemma 4.5

Proof. Use positive durations and the finite symmetric field family grad $\phi _ { i } , - \operatorname { g r a d } \phi _ { i } \}$ . A uniform speed bound on a compact chart gives a suficiently short total horizon for all words under consideration. Maximize the diferential rank of their endpoint maps over all finite words and strictly positive duration vectors. The maximum Q is attained, since ranks are integers bounded by $p .$ Around a maximizing vector, a nonzero Q-minor persists and no higher rank is possible. The constant-rank theorem gives an embedded image patch P of dimension Q with a strict remaining horizon margin.

Append any one of the allowed fields for an additional positive duration. Maximality bounds the augmented rank by $Q .$ Taking that duration to zero makes its extra derivative tangent to $P ;$ the original Q-minor still persists. Thus all the generators, and hence their brackets, are tangent to $P .$ . Their Lie rank is $p ,$ so $Q = p$ and $P$ is open in L.

Let the fixed maximizing word end at $\theta _ { * }$ . Its inverse word is a local difeomorphism sending $\theta _ { * }$ to $\theta _ { 0 }$ and a smaller open patch $P$ to a neighborhood of $\theta _ { 0 }$ in L. Every point of that neighborhood is obtained by a nearby forward word followed by this fixed inverse word. Finite-path continuous dependence ensures that all these paths stay in the larger chart after shrinking $P .$ . This uses control words only as horizontal paths; it does not call their concatenation a single autonomous loss.

For each segment, replace its time parameter by a smooth increasing function with the same endpoints and all derivatives zero at both ends. The resulting joined path is smooth and horizontal, including at the joins, and has exactly the same image and endpoint. □

## C.6 Proof of Theorem 4.2 for $q \geq 3$ and the elementary low-rank cases

Proof of Theorem $4 . 2$ for $q \geq 3$ and the elementary low-rank cases. For $q \geq 3$ , Lemmas 4.4 and 4.5 give the mobility identity on an entire neighborhood of $\theta _ { 0 }$ in its initial plaque. Shrink the original Frobenius product box so its actual initial level lies there. A target cutof as in the analytic proof makes $K _ { \mathrm { l o c } }$ globally smooth and proves the metric property.

If $q = 0$ , the stationary argument already applies. If $q = 1$ , the regular horizontal line bundle is involutive, so $p = 1$ . More generally, whenever $p = q , \phi$ is locally injective on the initial plaque by the constant-rank theorem. A local section factors M smoothly through its represented image, and target extension and cutof give the metric property without assuming ordinary closure. These are the classical local recoverability cases. It remains to prove the smooth $q = 2 , p = 3$ case. □

## C.7 Proof of Lemma 4.6

Proof. The coordinate-gradient fields and $H _ { 1 } , H _ { 2 }$ are smooth frame changes of the same horizontal bundle. Their generated Lie modules have the same evaluated distribution, namely $T L$

Let  be the smooth ideal generated by all ordered $H _ { i }$ -derivatives of $^ { c , }$ including c. It is stable under $H _ { 1 } , H _ { 2 }$ by Leibniz’s rule. The module

$$
\mathcal { N } = \operatorname { s p a n } _ { C ^ { \infty } } \{ H _ { 1 } , H _ { 2 } \} + \mathcal { T } \partial _ { \eta }
$$

is closed under brackets. In fact, with $A _ { 1 } = A , A _ { 2 } = B$

$$
[ H _ { i } , a \partial _ { \eta } ] = ( H _ { i } a - a \partial _ { \eta } A _ { i } ) \partial _ { \eta } , \quad [ a \partial _ { \eta } , b \partial _ { \eta } ] = ( a \partial _ { \eta } b - b \partial _ { \eta } a ) \partial _ { \eta } .
$$

Both coeficients belong to $\mathcal { T }$ when $a , b \in { \mathcal { T } } ;$ stability under $\partial _ { \eta }$ is not needed. Horizontal brackets are included by $c \in \mathcal { Z }$ , and smooth coeficients cause only the displayed terms and horizontal multiples.

If all generators of $\mathcal { T }$ vanished at $\theta _ { 0 } , \mathcal { N } ( \theta _ { 0 } )$ would have dimension at most two, contradicting Lie rank three. Thus a finite least order m exists.

For $m \geq 2$ , interchange adjacent horizontal operators in an m-word applied to c. Its change is an outer horizontal word applied to $\pm c \partial _ { \eta } ( H _ { \mathrm { i n n e r } } c )$ . The outer word has length at most $m - 2$ Every Leibniz term contains a first c factor acted on by at most $m - 2$ horizontal derivatives, which vanishes at $\theta _ { 0 }$ by minimality. All adjacent interchanges therefore preserve the evaluation. Orders zero and one are immediate. The nonzero leading tensor is symmetric; polarization then shows that its diagonal polynomial $P _ { m }$ is nonzero. □

## C.8 Proof of Lemma 4.7

Proof. Choose independent unit vectors $e ^ { ( 1 ) } , e ^ { ( 2 ) }$ avoiding the zeros of $P _ { m }$ . Their negatives also avoid these zeros by homogeneity. Denote the four directions by $\mathcal { D }$ and let $e ^ { \perp }$ be the counterclockwise unit rotation of e. Since det $( e , e ^ { \perp } ) = 1 , [ H _ { e } , H _ { e ^ { \perp } } ] = c \partial _ { \eta }$

Let $\Phi _ { X } ^ { t }$ denote the local flow of $X$ . The lift of the base rectangle

$$
0 \longrightarrow \varepsilon e \longrightarrow \varepsilon e + \delta e ^ { \perp } \longrightarrow \delta e ^ { \perp } \longrightarrow 0
$$

has endpoint

$$
h _ { e } ( \delta ) = \Phi _ { H _ { e ^ { \perp } } } ^ { - \delta } \Phi _ { H _ { e } } ^ { - \varepsilon } \Phi _ { H _ { e ^ { \perp } } } ^ { \delta } \Phi _ { H _ { e } } ^ { \varepsilon } ( \theta _ { 0 } ) .
$$

It lies in the base-zero fiber, is smooth in signed $\delta ,$ and $h _ { e } ( 0 ) = \theta _ { 0 }$ . Diferentiating at zero gives the vertical vector

$$
\Delta _ { e } ( \varepsilon ) = ( \Phi _ { H e } ^ { - \varepsilon } ) _ { * } H _ { e ^ { \perp } } ( \Phi _ { H _ { e } } ^ { \varepsilon } ( \theta _ { 0 } ) ) - H _ { e ^ { \perp } } ( \theta _ { 0 } ) .
$$

For the iterated bracket $\mathrm { a d } _ { H _ { e } } ^ { j } H _ { e ^ { \perp } }$ , write its vertical coeficient as $f _ { j }$ . Then $f _ { 1 } = c$ and $f _ { j + 1 } =$ $H _ { e } f _ { j } - f _ { j } \partial _ { \eta } A _ { e }$ , where $A _ { e }$ is the vertical coeficient of $H _ { e }$ . Induction yields

$$
f _ { j } = H _ { e } ^ { j - 1 } c + \sum _ { \ell = 0 } ^ { j - 2 } a _ { j \ell } H _ { e } ^ { \ell } c
$$

for smooth coeficients $a _ { j \ell }$ . Minimality of m kills all lower terms at $\theta _ { 0 }$ . The finite Taylor formula for the smooth flow pullback therefore gives

$$
\Delta _ { e } ( \varepsilon ) = \frac { \varepsilon ^ { m + 1 } } { ( m + 1 ) ! } P _ { m } ( e ) \partial _ { \eta } + o ( \varepsilon ^ { m + 1 } ) .
$$

This uses a finite Taylor remainder, not convergence of a smooth Taylor series. One suficiently small positive ε makes all four derivatives nonzero. The one-dimensional inverse function theorem makes each signed δ family cover an open fiber interval around zero. Their finite intersection contains a common open interval I. Shrink the $\delta$ ranges, $I ,$ and all state boxes so the finitely many loop families stay inside the ordinary and $K _ { \mathrm { l o c } }$ domains. □

## C.9 Proof of Lemma 4.8

Proof. For each $b \neq 0 .$ , at least one $e \in \mathcal { D }$ satisfies $e \cdot b < 0$ , because the two chosen axes are independent. The rectangle lies in $e \cdot w \geq 0$ , for either sign of $\delta ;$ the tail lies in $e \cdot w < 0$ except at zero. For $0 < \rho < 1$ , follow its first three edges and the returning fourth edge only to $\rho \delta e ^ { \perp }$ join that point to $\rho b$ by a segment, then follow the remaining tail to b. The connector has negative e-projection after its first point, so it cannot meet the rectangle. It meets the tail only at $\rho b ,$ since b is not parallel to $e ^ { \perp }$ . The full polygon is consequently injective and still starts exactly at zero.

For an explicit velocity estimate, assign each original rectangle edge and the tail one unit of time. Cut the return at time $4 - \rho ,$ traverse the connector until time $4 + \rho ,$ and then resume the original tail. The connector velocity is $( b - \delta e ^ { \perp } ) / 2$ . All changes occur on an interval of length $2 \rho ,$ giving $L ^ { 1 }$ error at most $C \rho ( \left. \left. b \right. \right. + \left. \delta \right. )$ . Round finitely many corners in mutually disjoint shrinking disks. Elementary smooth rounding of these nondegenerate polygonal corners preserves injectivity, endpoints, and regularity, and adds a vanishing $L ^ { 1 }$ velocity error. The disks may depend on the fixed $b , \delta ;$ no uniform radius over these parameters is required. The total variations are bounded.

If $b = 0$ , truncate the final return at $\rho \delta e ^ { \perp }$ and round its corners. On a fixed last time interval one can reduce the return velocity by the factor $1 - \rho ;$ the same $L ^ { 1 }$ conclusion holds and the terminal base point tends to zero. The case $\delta = 0$ is obtained later by continuity in $\delta ,$ without approximating a degenerate out-and-back path by embedded arcs.

For the lift equation $\theta ^ { \prime } = H _ { 1 } ( \theta ) u ^ { \prime } + H _ { 2 } ( \theta ) v ^ { \prime }$ , bounded state derivatives on a compact chart give

$$
\operatorname* { s u p } _ { t \leq T } \| \theta _ { j } ( t ) - \theta ( t ) \| \leq C \| w _ { j } ^ { \prime } - w ^ { \prime } \| _ { L ^ { 1 } } \exp \left( C \int _ { 0 } ^ { T } \| w _ { j } ^ { \prime } ( t ) \| d t \right) .
$$

The bounded variations and vanishing $L ^ { 1 }$ error thus give uniform convergence of lifts from the same $\theta _ { 0 } .$ , and continuation keeps the fine approximants inside the ordinary neighborhood. Each embedded regular represented lift is realized by one global smooth loss using the full compatible jet construction in Lemma 2.3. Ordinary closure holds at its endpoint. Passing to the endpoint limit through the already smooth $K _ { \mathrm { l o c } }$ proves the required equality at every rectangle and tail endpoint, including closed-loop endpoints. Those limiting closed loops are not asserted to be nonstationary autonomous loss trajectories themselves. Continuity of $h _ { e } ( \delta )$ and of its straight-tail transport covers $\delta = 0$ □

## C.10 Proof of Theorem 4.2 for $q = 2 , p = 3$

Proof of Theorem 4.2 for $q = 2 , p = 3$ . Let $P _ { b }$ be horizontal transport along the straight base segment from zero to $b ,$ starting at an arbitrary initial fiber coordinate $\eta \in I$ . For every $b \neq 0$ select an exterior direction e and a signed δ such that $h _ { e } ( \delta )$ is precisely that fiber point. Lemma 4.8 gives

$$
M ( P _ { b } ( \eta ) ) = K _ { \mathrm { l o c } } ( \phi ( P _ { b } ( \eta ) ) ) .
$$

For $b = 0$ the same follows from the truncated-return limit. The map $\Psi ( b , \eta ) = P _ { b } ( \eta )$ is smooth, has base projection $b ,$ and has an invertible fiber derivative at (0, 0). The inverse function theorem therefore makes it a local difeomorphism from a smaller base disk times fiber interval onto a plaque neighborhood of $\theta _ { 0 }$ . The equality holds at every such point. Discontinuous choice of a witnessing direction e is harmless: Ψ itself is smooth, and the equality is pointwise for every $( b , \eta )$

Finally shrink the original ambient Frobenius product box so its entire initial level lies in this neighborhood. Extend $K _ { \mathrm { l o c } }$ globally by a target cutof equal to one near $\phi ( \theta _ { 0 } )$ , and shrink once more into that equality region. This proves the metric property on the actual local invariant level and completes Theorem 4.2. □

## C.11 Proof of Proposition 4.9

Proof. The stationary case is immediate. Otherwise fix one open clock ball B and a state box inside the domain of $K _ { \mathrm { l o c } }$ . For any fixed finite sequence $g _ { 1 } , \dotsc , g _ { k } \in B$ , let $E ( t )$ be the flowword endpoint at durations $t _ { 1 } , \ldots , t _ { k }$ . All suficiently small positive duration vectors have a strict represented clock. Smooth control ramps and the compatible autonomous jet construction give genuine single-global-loss endpoints tending to $E ( t )$ . The smooth representative permits the limit, so $R ( E ( t ) ) = 0$ throughout the positive open duration orthant near zero.

The finite endpoint map extends smoothly to a full neighborhood of zero durations. Every mixed derivative of $R \circ E$ on the positive orthant is zero and remains zero at its corner by continuity. Diferentiating once in each duration yields

$$
Y _ { g _ { 1 } } \cdot \cdot \cdot Y _ { g _ { k } } R ( \theta _ { 0 } ) = 0 , \qquad Y _ { g } = - D \phi ^ { \mathsf { T } } g ,
$$

with the naming order fixed by pullback convention; every sequence order is available. The zeroorder identity follows at the initial time. Negative durations are used only to define derivatives of a smooth map, not as assumed test trajectories.

For each fixed k the displayed evaluation is multilinear in its constant controls. Vanishing on $B ^ { k }$ extends it to all controls, and therefore to all ordered coordinate-gradient words. Every bracket field is a finite signed combination of such diferential operator words, and products of bracket fields retain this property. Choose a finite local bracket frame $V _ { 1 } , \ldots , V _ { p }$ for $\mathcal { E } .$ All its ordered products annihilate $R$ at $\theta _ { 0 }$ . Each plaque-coordinate vector field is a smooth linear combination of this frame. Repeated Leibniz expansion of any coordinate derivative consequently consists of smooth coeficients times ordered frame derivatives of $R ,$ all zero at $\theta _ { 0 }$ . This proves flatness in every plaque chart. The argument does not identify a smooth flat function with zero on a neighborhood. □

## C.12 Proof of Proposition 4.10

Proof. Since $q = 2 , X , Y$ smoothly frame . Put $\lambda = \langle X , Y \rangle = a ^ { \mathsf { T } } M b$ . Euclidean diferentiation gives the exact identity

$$
[ X , Y ] = \operatorname { g r a d } \lambda - 2 D ^ { 2 } u Y .
$$

The bracket lies in and the last term is horizontal, so the full ambient grad λ belongs to . If ordinary closure holds, Lemma 2.4 gives a nonempty plaque-open patch O with the common fullmatrix factorization. Actual constant-rank fibers of $\phi | _ { L }$ make λ constant along them. Its gradient is therefore orthogonal to $\mathcal { E } \cap$ ker $D \phi$ . Because it also lies in $\mathcal { E } ,$ it belongs to $\mathcal { H } .$ , the orthogonal complement of that vertical subspace inside $\mathcal { E } .$ . The identity above then makes $[ X , Y ]$ horizontal on O.

Smooth frame coeficients imply that the horizontal bundle is involutive on this leaf-open patch. Restriction of ambient tangent fields to L commutes with brackets. All coordinate gradients are smooth combinations of $X , Y ,$ , so their generated distribution on O has rank two. This contradicts constant $p > 2 ;$ an ambient-open involutivity patch is not needed. Conversely, when $p = q = 2$ the local recoverability argument already proves the metric property and hence ordinary closure. Finally $D ^ { 2 } u = 0$ in the afine case and $D ^ { 2 } u = \alpha I$ in the isotropic quadratic case, giving the stated specializations without changing the Euclidean parameter metric. □

## D Proofs of the autonomous-loss controls

## D.1 Proof of Proposition 5.1

Proof. Write $F ( x ) = x ^ { k }$ . The coordinate gradients are

$$
X = \partial _ { x } , \qquad Y = ( F ^ { \prime } ( x ) z , 1 , F ( x ) ) .
$$

They are independent everywhere and $\operatorname { a d } _ { X } ^ { k } Y = k ! \partial _ { z } $ , so they generate a rank-three distribution everywhere. At the origin $Y - \partial _ { y }$ has polynomial coeficients homogeneous of degree k, independent

of $y .$ . Every nonzero bracket of length $\ell \geq 2$ contains at least one such nonconstant factor and at most $\ell - 1$ coeficient diferentiations, hence vanishes at the origin when $\ell \leq k$ . Thus the stated delay is exact. The mobility is

$$
M = \left[ \begin{array} { l l } { { 1 } } & { { F ^ { \prime } z } } \\ { { F ^ { \prime } z } } & { { 1 + F ^ { 2 } + ( F ^ { \prime } z ) ^ { 2 } } } \end{array} \right] .
$$

Its first parameter derivative is zero at the origin for $k \geq 2$ . Consequently rank $D ( \phi _ { k } , M ) ( 0 ) = 2$ so this necessary first-order test alone does not detect the failure below.

Fix $a > 0$ and $\epsilon > 0$ , and set, for all real $u ,$

$$
\begin{array} { l } { { h ( u ) = \epsilon u ( a - u ) , } } \\ { { \zeta ( u ) = \displaystyle \frac { 1 } { \sqrt { 1 + F ( u ) ^ { 2 } } } \int _ { 0 } ^ { u } \frac { F ( s ) h ^ { \prime } ( s ) } { \sqrt { 1 + F ( s ) ^ { 2 } } } d s , } } \\ { { B ( u ) = \displaystyle \frac { F ^ { \prime } ( u ) \zeta ( u ) - h ^ { \prime } ( u ) } { 1 + F ( u ) ^ { 2 } } , } } \\ { { A ( u ) = \displaystyle \int _ { 0 } ^ { u } \left[ - 1 - \frac { \left( h ^ { \prime } ( s ) - F ^ { \prime } ( s ) \zeta ( s ) \right) ^ { 2 } } { 1 + F ( s ) ^ { 2 } } \right] d s . } } \end{array}
$$

All these functions are globally smooth, since the denominators are strictly positive. Take

$$
f _ { 1 } ( u , v ) = - u , \qquad f _ { 2 } ( u , v ) = A ( u ) + B ( u ) ( v - h ( u ) ) .
$$

The first trajectory from the origin is $( x , y , z ) = ( t , 0 , 0 )$ . The second is

$$
x = t , \qquad z = \zeta ( t ) , \qquad y = h ( t ) - F ( t ) \zeta ( t ) , \qquad \phi _ { k } = ( t , h ( t ) ) .
$$

Indeed, diferentiating the defining integral gives $\zeta ^ { \prime } = F ( h ^ { \prime } - F ^ { \prime } \zeta ) / ( 1 + F ^ { 2 } ) = - F B$ . Along the displayed graph grad $f _ { 2 } = \left( A ^ { \prime } - B h ^ { \prime } , B \right)$ , and the factor equations are

$$
\dot { x } = - A ^ { \prime } + B ( h ^ { \prime } - F ^ { \prime } \zeta ) = 1 , \qquad \dot { z } = - F B = \zeta ^ { \prime } , \qquad \dot { y } = - B .
$$

The asserted derivative of y equals $( h ^ { \prime } - F ^ { \prime } \zeta ) / ( 1 + F ^ { 2 } ) = - B$ as well. Thus uniqueness verifies the actual autonomous trajectories, not merely a controlled proxy.

At time a both outputs equal $( a , 0 )$ . Integration by parts gives

$$
\zeta ( a ) { \sqrt { 1 + F ( a ) ^ { 2 } } } = - \int _ { 0 } ^ { a } { \frac { h ( s ) F ^ { \prime } ( s ) } { ( 1 + F ( s ) ^ { 2 } ) ^ { 3 / 2 } } } d s < 0 .
$$

The first trajectory has $M _ { 1 2 } = 0$ and the second has $M _ { 1 2 } = F ^ { \prime } ( a ) \zeta ( a ) < 0$ . They cannot share one entire mobility K at the common output. As $a \downarrow 0$ with ϵ fixed, both parameter paths on $[ 0 , a ]$ converge uniformly to the origin. Hence the witness lies inside every proposed ordinary neighborhood for a small enough a. This proves failure of (O) with the exact required quantifiers. □

## D.2 Proof of Proposition 5.2

Proof. Here $X = \left( 1 , 0 , 0 , 0 \right)$ and $\boldsymbol { Y } = ( z , 1 , x + w , z )$ are independent; $[ X , Y ] = \partial _ { z } $ and $[ \partial _ { z } , Y ] =$ $\partial _ { x } + \partial _ { w }$ . These four vectors span $\mathbb { R } ^ { 4 }$ everywhere. The first output is afine, so Proposition 4.10 excludes ordinary closure. Directly,

$$
M = \left[ \stackrel { 1 } { z } \right. \left. 1 + ( x + w ) ^ { 2 } + 2 z ^ { 2 } \right] , \qquad v = ( 0 , - ( x + w ) , 1 , 0 ) \in \ker D \phi
$$

has $D M _ { 1 2 } [ v ] = 1$ , violating the initial-plaque flatness consequence. This example satisfies the rank hypotheses but not the universal ordinary premise; it is not a counterexample to the equivalence in its proved regimes. □

## E Proofs for restricted loss families

## E.1 Proof of Proposition 6.1

Proof. Let $E ( v )$ be the time-one endpoint from $\theta _ { 0 }$ of $\dot { \theta } = - D \phi ( \theta ) ^ { T } v$ . For v in a suficiently small ball, small-field local ODE theory gives existence on [0, 1] and smooth dependence on v. $\mathrm { A t } \ v = 0$

$$
E ( 0 ) = \theta _ { 0 } , \qquad D E ( 0 ) = - D \phi ( \theta _ { 0 } ) ^ { T } .
$$

Put $z _ { 0 } ~ = ~ \phi ( \theta _ { 0 } )$ and $\Psi ( v ) = \phi ( E ( v ) )$ . Then $D \Psi ( 0 ) = - M _ { 0 }$ is invertible. Shrinking to a ball $B _ { \delta } ( 0 )$ , the inverse function theorem makes Ψ a difeomorphism onto an open target neighborhood V. Define

$$
K _ { \mathrm { l o c } } ( z ) = M \big ( E ( \Psi ^ { - 1 } ( z ) ) \big ) , \qquad z \in V .
$$

Full row rank persists on a parameter neighborhood $U _ { 0 }$ , on which $M \geq m I$ for some $m > 0 .$ . Choose a smaller $U \ni \theta _ { 0 }$ with closure in $U _ { 0 }$ , with $\phi ( U )$ contained in a target neighborhood $V _ { 1 }$ whose closure is compact in V, and with $| \phi ( \theta ) - z _ { 0 } | < r$ on $U$ , where $r / m < \delta$

For $g \neq 0$ let $n = g / | g |$ . As long as its actual flow remains in $U$

$$
\frac { d } { d t } \big [ n ^ { T } ( \phi ( \theta ( t ) ) - z _ { 0 } ) \big ] = - | g | n ^ { T } M ( \theta ( t ) ) n \leq - m | g | .
$$

Integration and the target-radius bound imply $t | g | < r / m < \delta$ . Time rescaling of the constant control vector field gives $\theta ( t ) = E ( t g )$ . Thus $t g$ belongs to the inverse chart and

$$
t g = \Psi ^ { - 1 } ( \phi ( \theta ( t ) ) ) , \qquad M ( \theta ( t ) ) = K _ { \mathrm { l o c } } ( \phi ( \theta ( t ) ) ) .
$$

This argument covers the whole interval before exit, including arbitrarily large $g ;$ it does not assume that $t g$ remains small without proving it. For $g = 0$ the solution is stationary. A smooth target cutof equal to one near $\overline { { V } } _ { 1 }$ and supported in $V$ extends $K _ { \mathrm { l o c } }$ by zero to a global smooth K. No positivity outside the tested image is required. □

## E.2 Proof of Theorem 6.3

Proof. Write $v = y + x ^ { k } z$ . The exact parameter gradient equations are

$$
\begin{array} { l } { \dot { x } = b _ { 1 } - x - \gamma v + k x ^ { k - 1 } z ( b _ { 2 } - \gamma x - v ) , } \\ { \dot { y } = b _ { 2 } - \gamma x - v , \qquad \dot { z } = x ^ { k } ( b _ { 2 } - \gamma x - v ) . } \end{array}
$$

For one fixed $c > 0$ , introduce

$$
s = { \frac { t } { c a } } , \quad x = a X , \quad y = a ^ { 2 } Y , \quad z = a ^ { k + 2 } Z , \qquad b _ { 1 } = B / c , \quad b _ { 2 } = a C .
$$

Put $V = Y + a ^ { 2 k } X ^ { k } Z$ , so that $v = a ^ { 2 } V$ . On $0 \leq s \leq 1$ the exact scaled initial-value problem, starting at zero, is

$$
X ^ { \prime } = B - c a X - c \gamma a ^ { 2 } V + c k a ^ { 2 k + 2 } X ^ { k - 1 } Z ( C - \gamma X - a V ) ,\tag{16}
$$

$$
Y ^ { \prime } = c ( C - \gamma X - a V ) ,\tag{17}
$$

$$
Z ^ { \prime } = c X ^ { k } ( C - \gamma X - a V ) .\tag{18}
$$

These equations are polynomial in states and parameters $a , B , C$ . At $a = 0$ the solutions on [0, 1] are bounded polynomials. Local ODE dependence therefore gives existence and analytic dependence on the whole interval for parameters in a small neighborhood of the reference values. Concretely, choose a compact tube around the limiting trajectory; the integral equation and a Gronwall estimate prevent exit for nearby parameters. No global completeness of polynomial vector fields is asserted.

Let $E _ { c } ( a , B , C ) = ( X ( 1 ) , V ( 1 ) )$ . At $a = 0$

$$
E _ { c } ( 0 , B , C ) = \big ( B , c ( C - \gamma B / 2 ) \big ) .
$$

At $( B , C ) = ( 1 , \gamma / 2 )$ the endpoint is (1, 0), and the shooting Jacobian is

$$
D _ { ( B , C ) } E _ { c } ( 0 , 1 , { \gamma } / { 2 } ) = \left( \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { - c { \gamma } / 2 } } & { { c } } \end{array} \right) , \qquad \operatorname * { d e t } D _ { ( B , C ) } E _ { c } = c > 0 .
$$

The analytic implicit function theorem supplies $B _ { c } ( a ) , C _ { c } ( a )$ with $B _ { c } ( 0 ) = 1 , C _ { c } ( 0 ) = \gamma / 2$ and $E _ { c } ( a , B _ { c } ( a ) , C _ { c } ( a ) ) = ( 1 , 0 )$ exactly. Thus $b _ { c } ( a ) = ( B _ { c } ( a ) / c , a C _ { c } ( a ) )$ gives the exact represented endpoint $( a , 0 )$ . Use a common suficiently small a<sub>∗</sub> for $c _ { 1 } , c _ { 2 }$ . The equality is an actual endpoint identity, not a formal-series substitute for a shooting solution.

At $a = 0$ and these limiting shooting parameters, $X ( s ) = s$ and

$$
Z ( 1 ) = c \gamma \int _ { 0 } ^ { 1 } s ^ { k } ( 1 / 2 - s ) d s = - \frac { c \gamma k } { 2 ( k + 1 ) ( k + 2 ) } .
$$

Smooth dependence in (18) yields

$$
z ( c a ) = - \frac { c \gamma k } { 2 ( k + 1 ) ( k + 2 ) } a ^ { k + 2 } + { \cal O } ( a ^ { k + 3 } ) .
$$

The full mobility is

$$
\begin{array} { r } { M ( x , y , z ) = \left( \underset { k x ^ { k - 1 } z } { 1 } \quad 1 + x ^ { 2 k } + ( k x ^ { k - 1 } z ) ^ { 2 } \right) . } \end{array}
$$

At each exact endpoint $x = a ,$ multiplication by $k a ^ { k - 1 }$ and subtraction give (11). The constants implicit in the remainders depend on the fixed $k , \gamma , c _ { 1 } , c _ { 2 }$ . A nonzero leading coeficient implies inequality for all suficiently small positive a.

The scaled states stay uniformly bounded on $[ 0 , 1 ]$ as $a \to 0$ . Consequently $x = O ( a ) , y = O ( a ^ { 2 } )$ and $z = O ( a ^ { k + 2 } )$ uniformly on the whole actual trajectories. They therefore fit inside every proposed common neighborhood for small enough a. A single K cannot take both unequal mobility values at the same output $( a , 0 )$ . This is precisely the required failure of the restricted ordinary-closure property. Its quantifier permits diferent horizons. □

## E.3 Proof of Corollary 6.4

Proof. For $\tau = W ^ { - 1 } b$ ，

$$
\begin{array} { r } { \ell _ { \tau } ( \psi _ { k } ( \boldsymbol { \theta } ) ) = f _ { b } ( \phi _ { k } ( \boldsymbol { \theta } ) ) + \frac { 1 } { 2 } b ^ { T } H ^ { - 1 } b . } \end{array}
$$

The additive constant changes no Euclidean parameter gradient field. Thus the two actual paths in Theorem 6.3 are also paths for ordinary squared error on this one fixed representation, and their represented endpoints both equal $W ( a , 0 ) ^ { T }$ . Moreover

$$
\begin{array} { l l } { { { \cal D } \psi _ { k } = W { \cal D } \phi _ { k } , \qquad M _ { \psi _ { k } } = W M _ { \phi _ { k } } W ^ { T } . } } \end{array}
$$

Invertibility of W preserves the nonzero full mobility diference; a particular matrix entry need not be preserved. The coordinate gradients are invertible constant linear combinations of those of $\phi _ { k } ,$ so their generated Lie distributions are the same. The ranks follow from Remark 6.2. The formula for $M _ { \phi _ { k } }$ gives $D M _ { \phi _ { k } } ( 0 ) = 0$ for $k \geq 2 ;$ constant congruence gives the same property for $M _ { \psi _ { k } }$ □

## E.4 Proof of Proposition 6.6

Proof. At $b = 0$ the parameter flow from $\theta _ { 0 }$ is stationary. Let ${ { J } _ { 0 } } \mathrm { ~ = ~ } D \phi ( { { \theta } _ { 0 } } )$ and $M _ { 0 } = J _ { 0 } J _ { 0 } ^ { T }$ At one fixed suficiently short time $\tau > 0$ , smooth flow dependence defines the endpoint for all suficiently small $b ,$ with the entire path in the ordinary neighborhood. If $P ( t )$ is the derivative of the parameter state with respect to b at zero, variation of the vector field gives

$$
{ \dot { P } } = - J _ { 0 } ^ { T } J _ { 0 } P - J _ { 0 } ^ { T } , \qquad P ( 0 ) = 0 .
$$

Consequently $Q ( t ) = J _ { 0 } P ( t )$ satisfies $\dot { Q } = - M _ { 0 } Q - M _ { 0 } , Q ( 0 ) = 0 ,$ , so

$$
\left. D _ { b } \phi ( \theta _ { b } ( \tau ) ) \right| _ { b = 0 } = - ( I - e ^ { - \tau M _ { 0 } } ) .\tag{19}
$$

This identity does not require $M _ { 0 }$ to be invertible. On $S = \mathrm { i m } M _ { 0 }$ , its eigenvalues are $- ( 1 - e ^ { - \tau \lambda } )$ for $\lambda > 0$ , hence nonzero. Restrict b to the q-dimensional space S. The constant-rank theorem supplies a local embedded image of $\phi$ with tangent $S$ at $z _ { 0 }$ , after shrinking the parameter neighborhood. In a chart of that image, (19) is an invertible q-dimensional endpoint derivative. The inverse function theorem produces a section $\sigma$ satisfying $\phi ( \sigma ( z ) ) = z { \mathrm { ~ o n } }$ a represented neighborhood of $z _ { 0 }$ , where each $\sigma ( z )$ is an actual endpoint of one of the tested flows. Thus

$$
K ( z ) = M ( \sigma ( z ) )
$$

there, by the common-closure hypothesis. The right side is smooth. A local target chart flattening the embedded image extends it normally; a target cutof then gives a global smooth extension agreeing on a smaller image. In the analytic case the local ODE, image charts and inverse theorem give an analytic section and a local analytic extension; no global analytic cutof is asserted. For $q < d$ the inverse theorem has only been used on the image, not on all of $\mathbb { R } ^ { d }$ □

## F Finite checks and reproducibility

The universal statements above rest on the included proofs. Existing exact algebraic checks are finite implementation evidence and were not rerun to produce this manuscript. The latest boundary validation used rational arithmetic on 39 fixtures: 24 efective scalar-imbalance and 15 nonscalar cases, including four zero-scalar cases. It checked 259 hidden-rotation and tangent identities and reported zero mismatches. These were bounded algebraic checks, not model training. These finite checks neither prove the universal classification nor establish task accuracy, statistical uncertainty, approximate-closure error or a practical speedup.

The local provenance audit preserves the available inputs, proof versions, automated review records, primary source texts, and their hashes. Complete local source custody is distinguished from selected theorem-and-proof reading. The two closest source versions are arXiv:2508.07370v2 (12 March 2026, ICLR 2026 rendition) and arXiv:2307.00144v2 (10 July 2024 revision of the NeurIPS 2023 work). The following numerical record is copied from a frozen, actually executed experiment. No new scientific run was used to assemble this manuscript. The algebraic and autonomous-loss records answer diferent finite questions.

## G A finite autonomous-loss implementation check

We executed one deterministic CPU sentinel for the engineered smooth-loss construction in Proposition 5.1. This tests an implementation of a proved mechanism, rather than a learning algorithm. The fixed-Hessian quadratic and ordinary squared-loss extension in this manuscript has not been executed: its numerical $^ { a _ { * } }$ and any executed quadratic-loss grid remain unknown. The measurements below must not be transferred to that new loss family.

Actual objective and parameter flow. Set $F ( u ) = u ^ { k } , h ( u ) = u ( a - u ) \mathrm { a n d } D ( u ) = 1 + F ( u ) ^ { 2 } .$ The implemented global objective was

$$
f _ { 2 } ( u , v ) = A ( u ) + B ( u ) ( v - h ( u ) ) , \qquad B = \frac { F ^ { \prime } \zeta - h ^ { \prime } } { D } , \qquad A ^ { \prime } = - 1 - \frac { ( h ^ { \prime } - F ^ { \prime } \zeta ) ^ { 2 } } { D } , \quad A ( 0 ) = 0 ,
$$

where

$$
\zeta ( u ) = D ( u ) ^ { - 1 / 2 } \int _ { 0 } ^ { u } F ( s ) h ^ { \prime } ( s ) D ( s ) ^ { - 1 / 2 } d s .
$$

Writing $v = y + F ( x ) z$ , its full output gradient and actual parameter ODE were

$$
\begin{array} { c } { { g _ { u } = A ^ { \prime } + B ^ { \prime } ( v - h ) - B h ^ { \prime } , } } \\ { { \displaystyle ( \dot { x } , \dot { y } , \dot { z } ) = - ( g _ { u } + F ^ { \prime } ( x ) z B , ~ B , ~ F ( x ) B ) . } } \end{array}
$$

$$
g _ { v } = B ,
$$

In particular the of-graph term $B ^ { \prime } ( v - h )$ was included; substituting the analytic graph for this ODE would not qualify the implementation. The analytic comparison path is $( t , h ( t ) - F ( t ) \zeta ( t ) , \zeta ( t ) )$ The comparison loss $f _ { 1 } ( u , v ) = - u$ was also integrated, with analytic path $( t , 0 , 0 )$

Frozen protocol and discriminating controls. The twelve witness cases used $k \in \{ 2 , 3 , 4 \}$ and $a \in \{ 0 . 5 , 0 . 3 5 , 0 . 2 5 , 0 . 1 7 5 \}$ , with $\epsilon = 1$ . Four matched controls used $F \equiv 0$ at the same a values; these have $p = q = 2 $ , mobility $I _ { 2 }$ and zero gap. We used DOP853 with relative tolerance $2 \times 1 0 ^ { - 1 2 }$ , absolute tolerance $1 0 ^ { - 1 4 }$ and 21 stored times per trajectory. Working scalar quadratures used absolute and relative tolerances $1 0 ^ { - 1 3 }$ and at most 100 subdivisions. Independently, a 50- decimal-digit quadrature evaluated the cancellation-avoiding reference

$$
\Delta _ { \mathrm { r e f } } = - \frac { k ^ { 2 } a ^ { 2 k + 1 } } { \sqrt { 1 + a ^ { 2 k } } } \int _ { 0 } ^ { 1 } \frac { t ^ { k } ( 1 - t ) } { ( 1 + a ^ { 2 k } t ^ { 2 k } ) ^ { 3 / 2 } } d t .\tag{20}
$$

This is a high-precision estimate, not a certified quadrature interval.

Five-point central diferences, step $2 \times 1 0 ^ { - 4 }$ , diferentiated the globally defined scalar loss composed with $\phi _ { k }$ in all three parameter coordinates. Both probes were of graph: $( x , y , z ) \quad = \quad$ $( 0 , 0 . 1 , 0 . 0 3 )$ , and $x = 0 . 3 a , z = 0 . 0 2 a , v = h ( x ) + 0 . 0 5 a$ , with $y = v - F ( x ) z$ . Each coordinate used tolerance $5 \times 1 0 ^ { - 8 } + 5 \times 1 0 ^ { - 6 } | g _ { \mathrm { F D } } |$ . The deliberately incorrect gradient omitting $B ^ { \prime } ( v - h )$ had to fail the same tolerance at both probes. Its first-probe discrepancy had to exceed $1 0 ^ { - 3 }$ . This matters because the omitted term vanishes on the analytic graph.

Before execution we fixed state error $\leq 3 \times 1 0 ^ { - 1 0 }$ , represented endpoint error $\leq 1 0 ^ { - 1 0 }$ , nonzerogap relative error $\leq 5 \times 1 0 ^ { - 6 }$ , and a zero-control gap threshold $1 0 ^ { - 1 2 }$ . For $L = k ^ { 2 } a ^ { 2 k + 1 } / ( k +$ $1 ) ( k + 2 ) )$ ), the measured $- \Delta / L$ was compared with $[ ( 1 + a ^ { 2 k } ) ^ { - 2 } , ( 1 + a ^ { 2 k } ) ^ { - 1 / 2 } ]$ using slack $5 \times 1 0 ^ { - 6 }$ Nonfinite outputs could not pass. The first completed failed case or exception would stop the sole attempt; no grid or tolerance was selected after observing results. There are no random seeds or statistical confidence intervals.

Table 1: Measured endpoint mobility gap $\Delta = M _ { 1 2 } ( f _ { 2 } ; a ) - M _ { 1 2 } ( f _ { 1 } ; a )$ and relative discrepancy $| \Delta - \Delta _ { \mathrm { r e f } } | / | \Delta _ { \mathrm { r e f } } |$ . All twelve frozen engineered-loss witnesses are shown.
<table><tr><td> $k$ </td><td>a</td><td>Measured  $\Delta$ </td><td>Relative discrepancy</td></tr><tr><td>2</td><td>0.500</td><td> $- 9 . 9 0 9 1 0 6 1 \times 1 0 ^ { - 3 }$ </td><td> $2 . 9 6 \times 1 0 ^ { - }$  -13</td></tr><tr><td>2</td><td>0.350</td><td> $- 1 . 7 2 9 4 2 3 4 \times 1 0 ^ { - 3 }$ </td><td> $8 . 1 2 \times 1 0 ^ { - }$  -14</td></tr><tr><td>2</td><td>0.250</td><td> $- 3 . 2 4 4 7 9 8 3 \times 1 0 ^ { - 4 }$ </td><td> $1 . 4 9 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>2</td><td>0.175</td><td> $- 5 . 4 6 6 8 1 7 1 \times 1 0 ^ { - 5 }$ </td><td> $2 . 8 2 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>3</td><td>0.500</td><td> $- 3 . 4 7 3 7 2 7 5 \times 1 0 ^ { - 3 }$ </td><td> $6 . 9 8 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>3</td><td>0.350</td><td> $- 2 . 8 9 1 1 6 2 0 \times 1 0 ^ { - 4 }$ </td><td> $2 . 1 7 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>3</td><td>0.250</td><td> $- 2 . 7 4 6 0 6 4 0 \times 1 0 ^ { - 5 }$ </td><td> $4 . 2 1 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>3</td><td>0.175</td><td> $- 2 . 2 6 1 8 7 8 2 \times 1 0 ^ { - 6 }$ </td><td> $5 . 4 0 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>4</td><td>0.500</td><td> $- 1 . 0 3 8 6 3 5 9 \times 1 0 ^ { - 3 }$ </td><td> $2 . 8 9 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>4</td><td>0.350</td><td> $- 4 . 2 0 2 7 9 3 5 \times 1 0 ^ { - 5 }$ </td><td> $1 . 9 9 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>4</td><td>0.250</td><td> $- 2 . 0 3 4 4 8 2 0 \times 1 0 ^ { - 6 }$ </td><td> $4 . 5 4 \times 1 0 ^ { - 1 3 }$ </td></tr><tr><td>4</td><td>0.175</td><td> $- 8 . 2 0 9 9 5 7 0 \times 1 0 ^ { - 8 }$ </td><td> $2 . 2 6 \times 1 0 ^ { - 1 3 }$ </td></tr></table>

Observed results and limits. All twelve witnesses and four matched controls passed the frozen criteria. The maximum state discrepancy over the 21 stored times was $2 . 0 1 5 \times 1 0 ^ { - 1 2 }$ and represented endpoint discrepancy $1 . 8 5 9 \times 1 0 ^ { - 1 3 } ;$ all four $F = 0$ gaps were stored as zero and their identitymobility checks passed. All 32 of-graph probes passed, with maximum coordinate FD discrepancy $1 . 0 4 3 \times 1 0 ^ { - 1 3 }$ . Every omitted-term mutant was rejected; the minimum across probes of its largest coordinate error was 0.0175. The $k = 4 , a = 0 . 1 7 5$ ratio band has width $1 . 3 1 9 5 \times 1 0 ^ { - 6 }$ , smaller than the allowed slack, and is explicitly tolerance dominated. Passing that band does not demonstrate sharp-band or next-order asymptotic resolution. The integrator tolerances are not global error certificates, and working quadrature error estimates are heuristic.

The historical implementation used 3122 $f _ { 2 }$ right-hand-side evaluations and 9057 quadrature calls. No GPU computation or model training was used. Raw trajectories, gradient probes, numerical references, code/input hashes are preserved in the local historical record. The supplied numerical payload can be run independently with the listed dependencies; two original external resource-wrapper files are unavailable, so the historical execution environment and resource measurements are not fully reproducible from this package. The public historical data consist of the exported summary and trajectory CSVs; the original raw probe/reference JSON files and sourcecustody receipts are not distributed in this package. These finite observations support implementation and mechanism agreement; the exact endpoint equality and universal quantifiers rest on the proofs. They establish no learning improvement, algorithmic speedup, numerical validation of the new quadratic theorem, or general asymptotic law.

Separate pre-submission replay. A separate pre-submission replay under Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and mpmath 1.3.0 passed all 16 cases and 32 of-graph probes, rejecting every omitted-term mutant. It used 3116 $f _ { 2 }$ right-hand-side evaluations and 7035 quadrature calls. Source copies, outputs, environment, and comparisons are in <sub>anc</sub>/<sub>audit\_replay</sub>/. These new results are separate from the historical measurements; they do not reproduce the historical environment or resource wrappers, and do not test the fixed-Hessian quadratic theorem.