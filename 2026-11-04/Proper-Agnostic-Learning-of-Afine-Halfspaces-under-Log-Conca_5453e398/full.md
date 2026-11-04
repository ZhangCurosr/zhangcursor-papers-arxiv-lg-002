# Proper Agnostic Learning of Afine Halfspaces under Log-Concave Marginals

Zhijie Zheng Peking University 2501210512@stu.pku.edu.cn ORCID: 0009-0006-7610-0691

## Abstract

We give a proper constant-factor agnostic learner for afine halfspaces under full-dimensional log-concave marginals. For arbitrary binary labels, the algorithm uses independent labeled samples and returns a single afine halfspace with error at most C OPT +ε. Its sample and real-arithmetic complexity are polynomial in dimension, inverse accuracy, and the logarithm of the reciprocal failure probability. The mean and covariance of the marginal need not be known, and the sample complexity can be made linear in this logarithmic confidence factor. The main geometric result compares global disagreement with disagreement on a balanced restriction: for a minority halfspace of mass p, global disagreement is within universal constant factors of p times conditional disagreement, provided the latter is suficiently small. To construct the restriction from noisy labels, we estimate the minority cap’s mean and covariance robustly, approximately minimize the classical zonoid-depth dual objective, and expand a superlevel set to a marginal quantile. The disagreement comparison cancels the noise amplification caused by conditioning and allows the learned halfspace to be used on the entire space. The analysis combines degree-four sum-of-squares moment bounds for log-concave laws with a translation-based reduction from balanced afine learning to homogeneous halfspace learning.

## 1 Introduction

Learning a halfspace with arbitrary adversarial label noise is a basic problem in computational learning theory. Under isotropic log-concave marginals, constant-factor agnostic learning is known for origin-centered halfspaces through localization and nonconvex optimization [1, 4]. Arbitrary afine ofsets create a diferent dificulty: a target can describe a very rare tail, and an error bound relative to the conditional tail law need not control the original risk.

For afine halfspaces under general isotropic log-concave marginals, Diakonikolas, Kane, and Stewart [3] give a proper learner with square-root-type dependence on the optimal error. Under Gaussian marginals, constant-factor guarantees are known for arbitrary afine halfspaces [3, 5]. We obtain a constant-factor guarantee for all full-dimensional log-concave marginals, with complexity jointly polynomial in dimension and inverse accuracy.

Write P for the marginal law of X, and let $\mathcal { H } _ { d }$ be the class consisting of all afine halfspaces and the two constants. For a joint law D of (X, Y ) with $Y \in \{ 0 , 1 \}$ , set

$$
\operatorname { e r r } _ { D } ( h ) = \mathbb { P } _ { D } ( h ( X ) \neq Y ) , \qquad { \mathrm { O P T } } _ { D } = \operatorname* { i n f } _ { h \in \mathcal { H } _ { d } } \operatorname { e r r } _ { D } ( h ) .
$$

A full-dimensional law is log-concave if it has a density whose logarithm is concave on its convex support; zero values outside the support are allowed. Isotropy means EX = 0 and Cov(X) = I.

Theorem 1.1 (Isotropic log-concave marginals). There are a universal constant C and a randomized algorithm with the following guarantee. For every joint law D with isotropic log-concave marginal P on $\mathbb { R } ^ { d }$ , every $\varepsilon , \delta \in ( 0 , 1 / 2 )$ , and arbitrary binary labels, the algorithm uses iid labeled samples alone and returns $\widehat { h } \in \mathcal { H } _ { d }$ such that

$$
\mathrm { e r r } _ { D } ( \widehat { h } ) \leq C \mathrm { O P T } _ { D } + \varepsilon
$$

with probability at least $1 - \delta$ . Its sample count and real-arithmetic operation count are

$$
\operatorname { p o l y } ( d , 1 / \varepsilon , \log ( 1 / \delta ) ) .
$$

The learner uses ordinary independent samples; each conditional sample is obtained by rejection from these samples. It requires neither the marginal density nor the comparator’s parameters, error, or minority mass. The marginal may have a nonsmooth density or a compact convex support.

Corollary 1.2 (Unknown mean and covariance). There are a universal constant $C ,$ a fixed polynomial $P _ { i }$ , and a randomized algorithm such that, for every joint law D with a full-dimensional log-concave marginal on $\mathbb { R } ^ { d }$ and binary labels, and every ε $\cdot , \delta \in ( 0 , 1 / 2 )$ , the algorithm returns

$$
\begin{array} { r } { \widehat { h } ( x ) = \mathbf { 1 } \{ w ^ { T } x \geq b \} , \qquad w \neq 0 , \quad b \in \mathbb { R } , } \end{array}
$$

with err $_ { D } ( \widehat { h } ) \leq C \operatorname { O P T } _ { D } + \varepsilon$ with probability at least $1 - \delta$ . It uses at most

$$
P ( d , \varepsilon ^ { - 1 } ) \log ( 2 / \delta )
$$

independent labeled samples and poly $( d , \varepsilon ^ { - 1 } , \log ( 2 / \delta ) )$ real-arithmetic operations. The mean and covariance are unknown, and these bounds are independent of the covariance condition number.

Section 10 proves the corollary by empirical whitening and confidence amplification. It also replaces constant candidates by finite-threshold halfspaces. Theorem 1.1 is stated in isotropic coordinates to simplify the geometric argument.

Computational model. We count real-arithmetic operations, comparisons, square roots, matrix decompositions, and the operations of the fixed-degree semidefinite procedures used for robust moment estimation. All sample requests and optimization stages have predetermined polynomial budgets. Section 9 implements the sigmoid gradient to suficient accuracy using arithmetic operations, so an exact exponential oracle is unnecessary. The bounds concern this real-arithmetic model; they do not bound the bit lengths of intermediate quantities.

Proof overview and related work. Let H be a minority halfspace of mass $p ,$ and let $Q = P \mid H$ In coordinates centered at the mean of $Q$ and normalized by its covariance, $Q$ is isotropic and the target has a bounded ofset. In these coordinates the convex objective $v \mapsto \mathbb { E } _ { P } ( 1 + v ^ { T } Z ) _ { + }$ has minimum p at the normalized target. This objective is the classical zonoid-depth dual, related to the cap-barycenter representations of $[ 7 , 8 ]$ . Robust estimates of the cap moments yield approximate coordinates in which the objective can be minimized to additive $O ( p )$ . We then expand a central superlevel set to a quantile restriction of mass $\Theta ( p )$ . The restriction balances H and has covariance comparable to $Q .$

A balanced afine target is reduced to homogeneous learning by translating along the segment between its two conditional class means. Same-label means and a grid of centers already appear in Gaussian testable afine learning [6]; our reduction verifies the needed two-dimensional density conditions for general log-concave marginals. The main geometric transfer is that a suficiently accurate afine classifier on the balanced restriction has global disagreement at most $C p$ times its conditional disagreement. The factor $p$ cancels the $1 / p$ increase in label noise. The resulting afine halfspace is used on the whole space.

We use Letwin’s quadratic-form variance theorem [10] to obtain a universal fourth-degree sumof-squares certificate, also recorded in [12]. Kothari–Steurer’s robust moment estimator [9] then supplies covariance estimates of constant relative accuracy and a small constant Mahalanobis error for the mean. Only fourth-moment certifiability and fixed estimation accuracy are required. The homogeneous learning step extends the geometric argument of [4] to the translated marginals arising in this reduction.

## 2 Elementary log-concave geometry

Constants $c , C > 0$ below are universal, unless their fixed dependencies are displayed. They may change from line to line. Linear marginals and positive-probability convex restrictions of log-concave laws are log-concave. We use the following standard fixed-dimensional facts [11, 2]. For $j \in \{ 1 , 2 \}$ an isotropic log-concave density f on $\mathbb { R } ^ { j }$ satisfies

$$
f ( z ) \leq A e ^ { - a \| z \| } , \qquad f ( z ) \geq a \quad ( \| z \| \leq r )\tag{1}
$$

for absolute $a , A , r > 0$ . If instead $c _ { 0 } I \preceq \mathrm { C o v } \preceq C _ { 0 } I$ , the same conclusions hold with adjusted constants and with the lower ball centered at the mean. A bounded mean also yields an exponential upper envelope about zero. One-dimensional log-concave densities of standard deviation σ obey

$$
c \leq \sigma \operatorname* { s u p } f \leq C .\tag{2}
$$

Their centered moments of every fixed order are bounded after variance normalization. We use upper-semicontinuous log-concave versions of densities; boundary identities may equivalently be read as traces from the interior.

Lemma 2.1 (Restrictions of constant probability). Let Z be isotropic log-concave and let A be a convex event with $\mathbb { P } ( A ) \geq a _ { 0 } > 0$ . Then

$$
\| \mathbb { E } [ Z \mid A ] \| \leq C ( a _ { 0 } ) , \qquad c ( a _ { 0 } ) I \preceq { \mathrm { C o v } } ( Z \mid A ) \preceq C ( a _ { 0 } ) I .
$$

Proof. For every unit $v ,$ the conditional second moment is at most $1 / a _ { 0 }$ . This bounds the covariance above and, by taking a directional supremum, bounds the norm of the conditional mean. The conditional density of $v ^ { T } Z$ is pointwise at most the unconditional density divided by $a _ { 0 }$ , hence at most $C / a _ { 0 }$ . Any scalar density bounded by B has variance at least $c / B ^ { 2 }$ : for every center t, the interval of radius $1 / ( 4 B )$ contains at most half the mass. This proves the lower bound. Convexity preserves log-concavity for subsequent applications. □

Lemma 2.2 (Interval mass). If a one-dimensional log-concave law gives probability at least $a _ { 0 }$ to $[ - R , R ]$ , then its variance and squared mean are at most $C ( { a } _ { 0 } , R )$

Proof. The density supremum is at least $a _ { 0 } / ( 2 R )$ . Equation (2) bounds its standard deviation by $2 C R / a _ { 0 }$ . If its mean is outside the interval, Chebyshev gives $a _ { 0 } \leq \sigma ^ { 2 } / ( | \mu | - R ) ^ { 2 }$ , which also bounds the mean. □

For a centered log-concave law, every halfspace through its mean has probability at least $1 / e$ [11]. This also follows by the one-dimensional form of the log-concave centroid inequality. It applies in any dimension by projection.

## 3 Robust cap statistics at a fixed degree

We use robust mean and covariance estimates with fixed relative accuracy. This permits a sum-ofsquares algorithm of fixed degree throughout the proof.

Lemma 3.1 (Fixed-accuracy robust log-concave moments). Fix $r \in ( 0 , 1 / 4 )$ . There is a constant $\gamma _ { r } > 0$ and a polynomial-time algorithm such that, from a suficiently large polynomial-size sample from a full-dimensional log-concave law Q with mean $\mu$ and covariance Σ, after at most $a \gamma _ { r }$ fraction of arbitrary replacements, it returns estimates obeying

$$
\begin{array} { r } { ( 1 - r ) \Sigma \preceq \widehat { \Sigma } \preceq ( 1 + r ) \Sigma , \qquad \left\| \Sigma ^ { - 1 / 2 } ( \widehat { \mu } - \mu ) \right\| \leq r } \end{array}\tag{3}
$$

with probability at least a fixed constant greater than $1 / 2$ . Neither $\mu$ nor $\Sigma$ is supplied. Independent repetitions may be retained as a list to obtain high confidence that at least one output obeys (3).

Proof. Let $Z = \Sigma ^ { - 1 / 2 } ( X - \mu )$ . Letwin’s Theorem 1.2 [10] states, for every symmetric $M$

$$
\mathrm { V a r } ( Z ^ { T } M Z ) \leq 2 \mathbb { E } \left\| \nabla ( Z ^ { T } M Z ) \right\| ^ { 2 } = 8 \left\| M \right\| _ { F } ^ { 2 } .\tag{4}
$$

The diference between the two sides is a positive-semidefinite quadratic form in the entries of M. Substituting $M = v v ^ { T }$ therefore gives a degree-four sum-of-squares certificate for

$$
\mathbb { E } ( v ^ { T } Z ) ^ { 4 } \preceq _ { \mathrm { s o s } } 9 \left\| v \right\| ^ { 4 } .\tag{5}
$$

Indeed the variance subtracts $( \operatorname { t r } M ) ^ { 2 }$ , and both $\left\| v v ^ { T } \right\| _ { F } ^ { 2 }$ and $( \mathrm { t r } ( v v ^ { T } ) ) ^ { 2 }$ equal $\| v \| ^ { 4 }$ . This certificate also appears in [12, Lemma 3.2]. Afine changes give the corresponding relative certificate with $( v ^ { T } \Sigma v ) ^ { 2 }$ on the right.

Theorem 1.2 of Kothari–Steurer [9], applied with $k = 4 .$ , gives Mahalanobis mean error $O ( \gamma ^ { 3 / 4 } )$ and relative Loewner covariance error $O ( \sqrt { \gamma } )$ for a universal certificate constant. Taking γ<sub>r</sub> suficiently small gives (3). The degree, and hence the polynomial exponent, remains fixed.

For completeness, the clean empirical distribution meets the required certificate with fixed slack. On the space of symmetric matrices, let $T = \mathbb { E } [ \mathrm { v e c } ( Z Z ^ { T } ) \mathrm { v e c } ( Z Z ^ { T } ) ^ { T } ]$ and let $i = \mathrm { v e c } ( I )$ Equation (4) gives $8 I + i \bar { i } ^ { T } - T \succeq 0$ . Also

$$
\mathbb { E } \Vert Z \Vert ^ { 8 } \leq C d ^ { 4 } ,
$$

by the scalar eighth-moment bound and $\begin{array} { r } { ( \sum _ { j } Z _ { j } ^ { 2 } ) ^ { 4 } \leq d ^ { 3 } \sum _ { j } Z _ { j } ^ { 8 } } \end{array}$ . Thus the empirical tensor $T _ { n }$ satisfies

$$
\mathbb { E } \left. T _ { n } - T \right. _ { F } ^ { 2 } \leq C d ^ { 4 } / n .
$$

Adding fixed identity slack to the Gram matrix before sampling gives a raw empirical quartic certificate with high constant probability for polynomial n. Empirical first and second moments concentrate with polynomial samples as well. The degree-four sum-of-squares inequality $( s - t ) ^ { 4 } \leq$ $8 ( s ^ { 4 } + t ^ { 4 } )$ transfers the certificate to empirical centering, and an empirical covariance lower bound converts $\| v \| ^ { 4 }$ to a fixed multiple of the squared empirical variance. Hence a universal relative empirical certificate is available. No degree-eight certificate is required: ordinary eighth moments are used only for concentration. Applying the robust theorem to this empirical law, and absorbing clean sampling errors by slightly tighter constants, proves the lemma. □

Remark 3.2. Positive-probability halfspace restrictions remain log-concave. Consequently, the same degree-four certificate and robust estimator apply to every clean conditional law used below.

## 4 Minority-cap normalization and a convex restriction

Let P be a full-dimensional log-concave law, H a halfspace with mass $p \in ( 0 , 1 / 2 ]$ , and $Q = P \mid H$ Write $\mu = \mathbb { E } _ { Q } X , \Sigma = \operatorname { C o v } _ { Q } X$ , and use the exact coordinates

$$
Z = \Sigma ^ { - 1 / 2 } ( X - \mu ) .\tag{6}
$$

The law of $Z$ under $Q$ is isotropic. Positive-probability halfspace restrictions have positive-definite covariance.

Lemma 4.1 (Normalized cap ofset). In coordinates (6), write

$$
H = \{ n ^ { T } Z + a \geq 0 \} , \quad \quad \| n \| = 1 .
$$

There is an absolute $A _ { 0 }$ such that $1 \leq a \leq A _ { 0 }$

Proof. Under $Q ,$ the nonnegative residual $W = u ^ { T } X - b$ of the original defining halfspace is log-concave. Its survival function $S$ is log-concave with $S ( 0 ) = 1$ . Concavity of log S implies $S ( s + t ) \leq S ( s ) S ( t )$ for $s , t \geq 0$ . Integrating gives

$$
( \mathbb { E } W ) ^ { 2 } \geq \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { \infty } S ( s + t ) d s d t = { \textstyle \frac { 1 } { 2 } } \mathbb { E } W ^ { 2 } .
$$

Thus Var $W \le ( \mathbb { E } W ) ^ { 2 }$ , so $a = \mathbb { E } W / \sqrt { \operatorname { v a r } W } \geq 1$

For the upper bound, let q be the density of $n ^ { T } Z$ under $Q ,$ and let f be its original marginal density divided by p. On $[ - a , \infty ) , f = q \colon$ , while

$$
\int _ { - \infty } ^ { - a } f ( x ) d x = ( 1 - p ) / p \geq 1 .
$$

By (1), $q ( 0 ) \geq c$ and $q ( - a ) \leq A e ^ { - c a }$ . For large a, the secant slope

$$
s = { \frac { \log q ( 0 ) - \log q ( - a ) } { a } }
$$

is at least a fixed positive constant. Concavity of log f bounds its left extrapolation by

$$
f ( x ) \leq q ( - a ) e ^ { s ( x + a ) } \qquad ( x \leq - a ) .
$$

The outside integral is at most $q ( - a ) / s < 1$ when a is suficiently large, a contradiction. If a boundary density vanishes, a positive value both to its left and at zero would contradict log-concavity. Taking interior traces therefore covers that case as well. □

Lemma 4.2 (The zonoid-depth dual objective). In exact cap coordinates,

$$
L ( v ) = \mathbb { E } _ { P } ( 1 + v ^ { T } Z ) _ { + }
$$

has a global minimizer $v _ { * } = n / a$ , with $\| v _ { * } \| \leq 1$ and $L ( v _ { * } ) = p$ . More precisely,

$$
L ( v ) - p = \mathbb { E } _ { P } \left[ \left| 1 + v ^ { T } Z \right| \mathbf { 1 } _ { H \triangle \{ 1 + v ^ { T } Z \geq 0 \} } \right] .\tag{7}
$$

If (3) holds with a suficiently small fixed r, then in the computable coordinates $\widehat { Z } = \widehat { \Sigma } ^ { - 1 / 2 } ( X - \widehat { \mu } )$ the true halfspace is $\{ 1 + \widehat { v } _ { * } ^ { T } \widehat { Z } \geq 0 \}$ with $\lvert | \widehat { v } _ { * } \rvert | \leq B$ and

$$
\mathbb { E } _ { P } ( 1 + \widehat { v } _ { * } ^ { T } \widehat { Z } ) _ { + } \leq C _ { 0 } p ,
$$

for universal constants $B , C _ { 0 }$

Proof. The exact hinge vanishes of H, and its expectation under $Q$ is one. Also $\mathbb { E } _ { P } [ Z \mathbf { 1 } _ { H } ] = 0$ , so the hinge’s gradient at $v _ { * }$ vanishes. Subtracting this supporting tangent of the positive-part function gives (7) pointwise after integration.

For the approximate assertion, write $Z = A \widehat { Z } + e .$ , where $A = \Sigma ^ { - 1 / 2 } \widehat { \Sigma } ^ { 1 / 2 }$ and $e = \Sigma ^ { - 1 / 2 } ( \widehat { \mu } - \mu )$ The true score is $n ^ { T } A \widehat { Z } + ( a + n ^ { T } e )$ . Its intercept $c _ { * } = a + n ^ { T } e$ is bounded below by $1 - r ,$ while $\left\| A ^ { T } n \right\|$ is bounded above by a universal constant. Dividing by $c _ { * }$ gives the required coeficient bound. The original expectation of the positive score is $p a$ , so the normalized hinge loss is $p a / c _ { * } \leq C _ { 0 } p$ .

The objective in Lemma 4.2 has a classical interpretation. The optimization

$$
\operatorname* { s u p } \{ \mathbb { E } _ { P } g : 0 \leq g \leq 1 , \mathbb { E } _ { P } [ g Z ] = 0 \}
$$

defines zonoid depth at the cap mean. Its formal Lagrange dual is inf<sub>v</sub> $L ( v )$ . Here $g \ = \ \mathbf { 1 } _ { H }$ and $\textstyle v \ = \ n / a$ explicitly attain matching values $p ,$ establishing equality of the primal and dual optima. Zonoid trimming and halfspace barycentric representation are established in $[ 7 , 8 ]$ ; we use approximate cap statistics to construct the balanced restriction below.

Proposition 4.3 (Balancing by a quantile restriction). Suppose estimates obey (3), and find v with $\lVert \boldsymbol { v } \rVert \leq B$ and

$$
\mathbb { E } _ { P } ( 1 + v ^ { T } \widehat { Z } ) _ { + } \leq C _ { 1 } p .
$$

For a suficiently large fixed $K _ { 0 }$ , if p is suficiently small, one can choose a halfspace K containing

$$
S = \{ v ^ { T } \widehat { Z } \geq - 1 / 2 \}
$$

with $P ( K ) \in [ K _ { 0 } p / 2 , 2 K _ { 0 } p ]$ . For $R = P \mid K$ , in the exact cap coordinates (6),

$$
c / K _ { 0 } \leq R ( H ) \leq 2 / K _ { 0 } ,\tag{8}
$$

$$
\| \mathbb { E } _ { R } Z \| \le C , \qquad c I \preceq \operatorname { C o v } _ { R } Z \preceq C I .\tag{9}
$$

The constants in (9) depend only on the fixed $K _ { 0 }$ . All these conclusions are stable under fixed-factor errors in the target mass K<sub>0</sub>p and a suficiently accurate empirical quantile.

Proof. The hinge is at least $1 / 2$ on $S ,$ so $P ( S ) \leq 2 C _ { 1 } p$ . Under $Q ,$ the approximate coordinate mean has norm $O ( r )$ . Take r small enough that $\left| v ^ { T } \mathbb { E } _ { Q } \widehat { Z } \right| \leq 1 / 4$ for every feasible v. Then S contains the halfspace $\{ v ^ { T } \widehat { Z } \geq v ^ { T } \mathbb { E } _ { Q } \widehat { Z } \}$ , and the centered log-concave halfspace bound gives $Q ( S ) \geq 1 / e$ . For $p$ in the rare regime, $v = 0$ is impossible because its hinge loss is one. Its scalar marginal is atomless, so lowering the threshold expands $S$ to any specified larger probability below one. Choose $K _ { 0 }$ with fixed slack beyond $2 C _ { 1 }$

Put $q _ { K } = Q ( K ) \geq 1 / e$ . Then

$$
R ( H ) = \frac { p q _ { K } } { P ( K ) } ,
$$

which gives (8). The law $Q _ { K } = Q \mid K$ has bounded mean and covariance comparable to I by Lemma 2.1. It is a component of R with weight $p q _ { K } / P ( K ) \ge c ^ { \prime } > 0$ . Covariance decomposition gives the lower bound in (9).

For the upper bound, fix any unit w. The law $Q _ { K }$ puts a fixed positive mass in $\left| w ^ { T } Z \right| \le C$ by its bounded directional second moment. So does R, by the same component weight. The w marginal of R is log-concave, since K is a halfspace. Lemma 2.2 bounds its variance and squared mean by a fixed constant. Taking suprema over unit w proves both remaining assertions.

Finally, all probability inequalities have fixed slack. A quantile with additive error a suficiently small multiple of $p$ still expands S and has the required mass bounds, after increasing $K _ { 0 }$ if necessary. □

## 5 A homogeneous primitive for translated marginals

The homogeneous learner will be applied after a translation. We therefore formulate its assumptions directly in terms of two-dimensional marginal densities.

Lemma 5.1 (Translated homogeneous learning). Let a marginal M on $\mathbb { R } ^ { d } , \ d \geq 2$ , have every two-dimensional marginal density f<sub>E</sub> satisfying

$$
f _ { E } ( z ) \geq c _ { 0 } \quad ( \| z \| \leq r _ { 0 } ) , \qquad f _ { E } ( z ) \leq C _ { 0 } e ^ { - c _ { 0 } \| z \| } ,\tag{10}
$$

where $c _ { 0 } , C _ { 0 } , r _ { 0 }$ are fixed positive constants. Given arbitrarily labeled samples and accuracy $\xi > 0$ , a polynomial-time algorithm returns a polynomial list containing a homogeneous halfspace with error at most $C \eta + \xi$ , with any prescribed high confidence, whenever a homogeneous comparator has error $\eta .$ . The polynomial is joint in $d , 1 / \xi$ , and log inverse failure probability. Constants may also be retained.

Proof. Encode labels by $\tilde { Y } = 2 Y - 1$ . For w $\neq 0$ define the bounded, scale-invariant sigmoid objective

$$
F _ { \sigma } ( w ) = \mathbb { E } \left[ \frac { 1 } { 1 + \exp ( \widetilde { Y } w ^ { T } X / ( \sigma \| w \| ) ) } \right] .\tag{11}
$$

We use the sigmoid surrogate introduced in [4]. Its structural gradient bound extends to (10), as the following calculation shows. In the plane of w and a unit comparator normal $u ,$ choose coordinates with $w = e _ { 2 }$ and $u = \sin \theta e _ { 1 } + \cos \theta e _ { 2 }$ , where $0 < \theta < \pi$ . Write

$$
a ( t ) = \frac { e ^ { - | t | } } { ( 1 + e ^ { - | t | } ) ^ { 2 } } , \qquad Y ^ { * } = \mathrm { s i g n } ( u ^ { T } X ) ,
$$

and let $G _ { \sigma }$ denote the objective with the clean label $Y ^ { * }$ . Then

$$
- e _ { 1 } ^ { T } \nabla G _ { \sigma } ( e _ { 2 } ) = \frac { 1 } { \sigma } \mathbb { E } \bigl [ Y ^ { * } X _ { 1 } a ( X _ { 2 } / \sigma ) \bigr ] .
$$

Suppose sin $\theta \geq A \sigma$ for a suficiently large fixed $A ,$ and $\sigma$ is suficiently small. In the two rectangles

$$
r _ { 0 } / 4 \le | x _ { 1 } | \le r _ { 0 } / 2 , \qquad | x _ { 2 } | \le c \sigma ,
$$

choose the fixed $c > 0$ small enough that $Y ^ { * } x _ { 1 } = | x _ { 1 } |$ and the rectangles lie in the core ball of (10). The density lower bound and $a ( x _ { 2 } / \sigma ) \geq c ^ { \prime }$ give a positive contribution bounded below by a fixed constant. The negative contribution can occur only where $| x _ { 1 } | \leq | x _ { 2 } | |$ cot $\theta |$ . Using $f _ { E } \leq C _ { 0 }$ there bounds its magnitude by

$$
\frac { C } { \sigma \sin ^ { 2 } \theta } \int _ { \mathbb { R } } x _ { 2 } ^ { 2 } a ( x _ { 2 } / \sigma ) d x _ { 2 } \leq \frac { C \sigma ^ { 2 } } { \sin ^ { 2 } \theta } .
$$

Consequently,

$$
- e _ { 1 } ^ { T } \nabla G _ { \sigma } ( e _ { 2 } ) \geq c - C { \frac { \sigma ^ { 2 } } { \sin ^ { 2 } \theta } } .
$$

The corrupted labels change this gradient component by at most $C \sqrt { \eta / \sigma }$ . Indeed, Cauchy–Schwarz bounds their contribution by a constant multiple of

$$
\sqrt { \eta } \left( \mathbb { E } \frac { e ^ { - 2 \left| X _ { 2 } \right| / \sigma } X _ { 1 } ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { 1 / 2 } \le C \sqrt { \eta / \sigma } ,
$$

where the final inequality follows by integrating the exponential planar density envelope. Choose A large and then κ small. Whenever sin $\theta \geq A \sigma$ and $\eta \le \kappa \sigma$ , the population gradient has a fixed positive lower bound. Adjusting A to convert the sine bound to an angle bound, there are fixed constants $A , \kappa , \rho , \sigma _ { 0 } > 0$ such that

$$
\| w \| = 1 , \quad \eta \leq \kappa \sigma , \quad 0 < \sigma \leq \sigma _ { 0 } , \quad \| \nabla F _ { \sigma } ( w ) \| \leq \rho \quad \Longrightarrow \quad \operatorname* { m i n } \{ \angle ( w , u ) , \angle ( - w , u ) \} \leq A \sigma .\tag{12}
$$

Randomized labels are allowed: their conditional corruption probability replaces the indicator in this calculation, and its square is at most itself.

The planar upper envelope also bounds disagreement of two homogeneous halfspaces by C times their angle. Hence (12) gives a useful classifier by retaining both signs. Both estimates hold under the density assumptions in (10).

We next verify that stochastic gradient descent finds an iterate with a suficiently small population gradient. The envelope implies EX $X ^ { T } \preceq C I$ . For a unit $w ,$ let $g$ be the stochastic gradient of the summand in (11). Diferentiating the normalized margin gives

$$
w ^ { T } g = 0 , \qquad \mathbb { E } \left. g \right. ^ { 2 } \leq C d / \sigma ^ { 2 } , \qquad \left. \nabla ^ { 2 } F _ { \sigma } ( v ) \right. _ { \mathrm { o p } } \leq C / \sigma ^ { 2 } \quad ( \left. v \right. \geq 1 ) .\tag{13}
$$

The Hessian estimate follows directionally from $\mathbb { E } X X ^ { T } \preceq C I$ and the first two derivative bounds of the sigmoid; it has no factor d. For the update

$$
v = w - s g , \qquad w ^ { + } = v / \left\| v \right\| ,
$$

tangency ensures that the entire segment w − tsg, $0 \leq t \leq 1$ , has norm at least one. Positive scale invariance gives $F _ { \sigma } ( w ^ { + } ) = F _ { \sigma } ( v )$ . Smooth descent and conditional expectation therefore yield

$$
\begin{array} { r } { {  { \mathbb E } } [ F _ { \sigma } ( w ^ { + } ) \mid w ] \le F _ { \sigma } ( w ) - s \left\| \nabla F _ { \sigma } ( w ) \right\| ^ { 2 } + C s ^ { 2 } d / \sigma ^ { 4 } . } \end{array}
$$

Take $s = c \sigma ^ { 4 } / d$ with a suficiently small fixed $^ { c , }$ and $T = C ^ { \prime } d / \sigma ^ { 4 }$ with a suficiently large fixed $C ^ { \prime }$ Since $0 \leq F _ { \sigma } \leq 1$ , the expected average squared gradient can be made less than $\rho ^ { 2 } / 4$ . Markov’s inequality gives, with constant probability, an iterate satisfying (12). Keep all iterates and their negatives; independent repeats amplify confidence without estimating any population gradient.

Finally enumerate a geometric grid of σ from a suficiently small fixed multiple of $\xi$ up to $\sigma _ { 0 }$ . If η is below a fixed constant, some grid point satisfies $\eta \le \kappa \sigma$ and $\sigma \leq C ( \eta + \xi )$ . If $\eta$ is above that constant, a constant hypothesis already meets a suficiently large constant-factor guarantee. Rescale the internal $\xi$ to obtain the displayed additive term. The grid, trajectory lengths, repetitions, and resulting list are polynomial. □

## 6 Balanced afine learning

Proposition 6.1 (Balanced afine primitive). Fix positive constants $b _ { 0 } , c _ { 0 } , C _ { 0 }$ . Let R be log-concave with

$$
\| \mathbb { E } _ { R } X \| \le C _ { 0 } , \qquad c _ { 0 } I \preceq \operatorname { C o v } _ { R } X \preceq C _ { 0 } I .
$$

If an afine comparator H has $R ( H ) \in [ b _ { 0 } , 1 - b _ { 0 } ]$ and label error $\eta _ { R }$ , a sample-only algorithm returns a polynomial list containing an afine classifier of error at most $C \eta _ { R } + \xi$ , with prescribed high confidence and joint polynomial complexity in $d , 1 / \xi$ , and log inverse failure probability.

Proof. Assume $d \geq 2 ;$ one-dimensional threshold ERM handles $d = 1$ . If $\eta _ { R }$ is larger than a suficiently small fixed constant, constants sufice. Otherwise, the observed label-conditional laws are within $O ( \eta _ { R } / b _ { 0 } )$ total variation of the clean laws $R \mid H$ and $R \mid H ^ { c }$ . Both clean conditional covariances are comparable to $I ,$ and their mean norms are bounded, by Lemma 2.1 after a fixed-condition-number afine change.

Lemma 3.1 estimates the two clean means $\mu _ { + } , \mu _ { - }$ to an arbitrarily small fixed Euclidean error $r _ { 0 }$ . Their signed distances from the target boundary are bounded below by a fixed constant. Indeed their nonnegative scalar residuals have density bounded by a constant $B ,$ , so

$$
\mathbb { E } V \geq \int _ { 0 } ^ { 1 / B } ( 1 - B t ) d t = 1 / ( 2 B ) .
$$

Thus suficiently accurate estimates $a _ { + } , a _ { - }$ remain on opposite sides. Repeat these paired estimation attempts independently and retain each resulting segment; with high confidence at least one pair is simultaneously accurate. Norm and segment-length caps discard inadmissible outputs without afecting that pair.

Fix a successful pair and any two-dimensional projection E. Each clean class density is bounded below on a fixed-radius ball about $E \mu _ { + }$ or $E \mu _ { - }$ . The R marginal dominates each by its fixed class probability. Small mean errors shrink these balls by at most a fixed amount. Log-concavity then gives a uniform lower density bound on equal-radius balls centered at every point of $[ E a _ { - } , E a _ { + } ] ;$ ${ \mathrm { i f ~ } } z = ( 1 - t ) E a _ { - } + t E a _ { + } + u$ , use the same displacement u in the two endpoint balls. The R marginal also has a universal exponential envelope, and every point on the segment has bounded norm. Consequently, for every $a \in [ a _ { - } , a _ { + } ]$ , the translated law of $X - a$ obeys (10).

The segment crosses the true boundary. Its length is bounded by a fixed constant, so a grid of Euclidean mesh $\tau$ has $O ( 1 / \tau )$ points and contains one point within distance $\tau$ of the crossing. The homogeneous halfspace through this point with the comparator’s normal difers from H only in a strip of width at most τ. The scalar density bound gives disagreement at most $C \tau$ . Lemma 5.1 at each grid point, with internal accuracy $c \xi$ and $\tau = c \xi$ , supplies a candidate of error at most ${ \mathit { C } } { \eta } _ { R } + { \xi }$ Retain all candidates. The list remains polynomial under confidence amplification. □

## 7 Local-to-global disagreement for a minority cap

Although K need not contain all of $H ,$ suficiently small disagreement on K determines the global disagreement up to a factor proportional to $P ( H )$ . This allows a halfspace learned on K to be used globally.

Lemma 7.1 (Parameter-to-global cap bound). In exact cap coordinates (6), write $H = \{ n ^ { T } Z + a \geq$ $0 \}$ , where $\| n \| = 1$ and $1 \leq a \leq A _ { 0 }$ . Let $h = \{ m ^ { T } Z + b \geq 0 \} , \ \| m \| = 1$ . There are universal $\beta _ { 0 } , C > 0$ such that

$$
\beta : = \| m - n \| + | b - a | \le \beta _ { 0 } \quad \Longrightarrow \quad P ( H \triangle h ) \le C p \beta .\tag{14}
$$

Proof. Project onto the plane of $m , n ,$ completing it arbitrarily if the normals are parallel. The projected law of $Q$ is isotropic log-concave. Use coordinates $z = ( x , t )$ with $n = e _ { 1 }$ and $H = \{ x \geq - a \}$ Let q be this projected conditional density and $f$ the original projected density divided by p. Because the projection contains $n ,$ conditioning commutes with the projection: $f = q$ on $H$ . In particular,

$$
q ( z ) \leq A e ^ { - c \| z \| } , \qquad q ( 0 ) \geq c .
$$

Disagreement implies

$$
| x + a | \leq | ( m - n ) ^ { T } z + b - a | \leq \beta ( 1 + \| z \| ) \leq \beta ( 1 + a + | t | + | x + a | ) .
$$

Absorbing the last term shows that it lies in the strip

$$
| x + a | \leq C \beta ( 1 + | t | ) .\tag{15}
$$

Inside H the envelope for q bounds f by $C e ^ { - c | t | }$

For a point $z = ( x , t )$ in the strip outside H, put $\lambda = a / ( - x ) \in ( 0 , 1 )$ and $y = \lambda z$ . Then $y$ is on the cap boundary. Concavity of log f on the segment between 0 and z gives

$$
\begin{array} { r } { f ( z ) \leq q ( y ) ^ { 1 / \lambda } q ( 0 ) ^ { - ( 1 - \lambda ) / \lambda } } \\ { \leq \exp \{ C / \lambda - c \| z \| \} . } \end{array}
$$

Here ∥y∥ $/ \lambda = \| z \|$ . On (15),

$$
1 / \lambda = \frac { - x } { a } \leq 1 + C \beta ( 1 + | t | ) .
$$

Taking $\beta _ { 0 }$ suficiently small absorbs the resulting positive multiple of $\beta | t |$ into the negative tail exponent. Thus $f ( z ) \leq C e ^ { - c ^ { \prime } | t | }$ throughout the strip, on either side. Integrating its width proves

$$
\int _ { H \triangle h } f \leq C \beta \int _ { \mathbb { R } } ( 1 + | t | ) e ^ { - c ^ { \prime } | t | } d t \leq C \beta .
$$

Multiplying by p gives (14).

This proof is valid for compact supports. If $f ( z ) > 0$ and $f ( 0 ) > 0$ , log-concavity forces a positive trace at every interior point of their segment, including y. A zero boundary trace therefore cannot hide positive density beyond the boundary. In dimension one the same argument bounds the density on an interval of length $O ( \beta )$ ; no extra projected coordinate is needed. □

Lemma 7.2 (Local disagreement controls afine parameters). Suppose R is log-concave in the exact cap coordinates, satisfies (9), and gives both classes of H a fixed positive probability. Then for fixed $\delta _ { 0 } , C > 0$ , every halfspace $h = \{ m ^ { T } Z + b \geq 0 \} , \ \| m \| = 1$ , satisfies

$$
R ( H \triangle h ) \leq \delta _ { 0 } \quad \Longrightarrow \quad \| m - n \| + | b - a | \leq C R ( H \triangle h ) .\tag{16}
$$

Constants cannot meet this small-disagreement premise.

Proof. Project onto the two normals. The two class-conditional laws under R have bounded means and covariance comparable to I, by Lemma 2.1. Their projected mean segment crosses the target boundary at a point $z _ { \mathrm { 0 } }$ of bounded norm. As in Proposition 6.1, log-concavity gives a fixed radius r and density lower bound $\ell > 0$ throughout the ball $\boldsymbol { B } ( z _ { 0 } , r )$

Translate its center to zero. The target becomes $\{ n ^ { T } z \ge 0 \}$ and the candidate becomes $\{ m ^ { T } z + c _ { h } \ge 0 \}$ , where $c _ { h } \ = \ m ^ { T } z _ { 0 } + b$ . Let $D _ { B }$ be the unnormalized planar area of their disagreement in this ball. Then $D _ { B } \leq R ( H \triangle h ) / \ell$

The central target bisects the ball. The diference between the candidate’s area and half the ball’s area is at most $D _ { B }$ . Parallel chords of a circular ball give a lower bound c min $\{ r ^ { 2 } , r | c _ { h } | \}$ on that area diference. For suficiently small $D _ { B }$ , this yields $| c _ { h } | \le C D _ { B }$ . The candidate and its same-normal central halfplane disagree in area at most $C | c _ { h }$ |. By the triangle inequality, the two central halfplanes then disagree in area at most $C D _ { B }$ . Their disagreement area is exactly a fixed multiple of their oriented angle, giving $\| m - n \| \leq C D _ { B }$ and excluding reversed orientation. Finally, $n ^ { T } z _ { 0 } = - a$ and

$$
\left| b - a \right| \leq \left| m ^ { T } z _ { 0 } + b \right| + \left\| m - n \right\| \left\| z _ { 0 } \right\| \leq C D _ { B } .
$$

This proves (16). In dimension one use the corresponding interval-length argument. A constant disagrees with H on at least the smaller class probability, so it is excluded after reducing $\delta _ { 0 }$

Theorem 7.3 (Minority-mass transfer). Let K and $R = P \mid K$ be as in Proposition $4 . 3 .$ . There are fixed $\delta _ { 0 } , c , C > 0$ such that every afine classifier h with $R ( H \triangle h ) \leq \delta _ { 0 }$ obeys

$$
c p R ( H \triangle h ) \leq P ( H \triangle h ) \leq C p R ( H \triangle h ) .\tag{17}
$$

Proof. Reduce $\delta _ { 0 }$ until Lemma 7.2 puts the parameters within the range of Lemma $7 . 1 ;$ combining the lemmas proves the upper bound. For the lower bound, $P ( H \triangle h ) \geq P ( ( H \triangle h ) \cap K ) = P ( K ) R ( H \triangle h )$ 2 and Proposition 4.3 gives $P ( K ) \ge K _ { 0 } p / 2$ □

The small-error premise is essential. An always-positive classifier has conditional disagreement bounded away from zero but may have global disagreement close to one when $p$ is small.

## 8 The learning algorithm

We describe the algorithm and prove its error guarantee; Section 9 establishes the sample and running-time bounds. The constants are chosen in the following order. Take $B = 2$ and choose the robust-estimation tolerance r small enough for Lemma 4.2 and the central-halfspace argument in Proposition 4.3. Fix the hinge-loss bound $C _ { 1 }$ , then choose $K _ { 0 }$ with fixed slack beyond $4 C _ { 1 }$ This determines the balanced-learning constants and the transfer threshold $\delta _ { 0 }$ . Finally choose the noise-ratio threshold α and the internal accuracy multiplier suficiently small. All these choices are independent of $d , p , \varepsilon , \delta $

Fix a comparator H with error $\eta .$ . Orient its minority class as positive, so $p = P ( H ) \leq 1 / 2 $ the algorithm runs both label orientations and includes both constants. For suficiently small fixed $\alpha > 0 , \mathrm { i f } \ \eta \ge \alpha p .$ the negative constant has error at most

$$
p + \eta \leq ( 1 + 1 / \alpha ) \eta .\tag{18}
$$

If $p \leq c \varepsilon$ , it has error at most $\eta + c \varepsilon$ . Consequently the only dificult regime is

$$
p \ge c \varepsilon , \qquad \eta / p \le \alpha .\tag{19}
$$

Choose internal accuracy $\varepsilon _ { 0 } = c _ { * } \varepsilon$ , where $c _ { * }$ is fixed suficiently small after all risk constants are fixed.

Let $q = \mathbb { P } ( Y = 1 )$ in the chosen orientation. Then $| q - p | \leq \eta$ . Estimate it by $\widehat { q }$ to additive $c \varepsilon _ { 0 }$ In the dificult regime it is a fixed-factor proxy for $p ,$ and in fact can be made arbitrarily close in relative constant terms. Branches with $\widehat { q }$ below a fixed multiple of $\varepsilon _ { 0 }$ need only constants. Choose a fixed $K _ { 0 }$ as in Proposition 4.3, large enough for the fixed-factor use of $q ,$ and a cutof

$$
q _ { * } = ( 1 0 0 K _ { 0 } ) ^ { - 1 } .
$$

If $\widehat { q } \geq q _ { * }$ , run Proposition 6.1 directly under the original isotropic law, retaining all its candidates. In a dificult branch $p$ is then bounded below by a fixed constant, so both comparator classes are balanced. No restriction is needed in that branch.

If $\widehat { q } < q _ { * }$ , estimate the moments of the observed positive law. Its total variation distance from $Q = P \mid H$ is ${ \cal { O } } ( \eta / p )$ . For example, comparing the unnormalized measures $\mathbb { P } ( X \in \cdot , Y = 1 )$ and $\mathbb { P } ( X \in \cdot , H )$ , their variation discrepancy is at most $\eta ,$ and normalizing costs only a constant multiple of $1 / p$ under (19). An iid maximal coupling gives iid clean $Q$ samples with iid mismatch indicators of this probability. A Chernof bound yields the replacement model in Lemma 3.1 when α is small enough. The learner uses only the observed samples; the coupling verifies the contamination assumption of the robust estimator.

Independently repeat the robust estimator and retain all admissible coordinate systems. With high confidence one satisfies (3). In each system optimize the unlabeled hinge to additive $c \widehat { q } ,$ and use an empirical quantile to construct K of mass approximately $K _ { 0 } { \widehat { q } } ,$ containing its central set $S .$ In the successful branch Proposition 4.3 applies with universal constants. The fixed cutof ensures $P ( K ) < 1 / 2$ . Rejection sampling supplies labeled examples under $R = P \mid K$ , where the conditional noise satisfies

$$
\eta _ { R } = \mathbb { P } ( Y \neq H \mid K ) \leq \eta / P ( K ) \leq C \eta / p .\tag{20}
$$

Run Proposition 6.1 in the estimated coordinates ${ \widehat { Z } } .$ , with accuracy $\varepsilon _ { \mathrm { 0 } }$ . On a successful estimation attempt, $Z = A \widehat { Z } + e$ from Lemma 4.2 has singular values bounded above and below by universal constants and $\| e \| \leq r$ . Thus (9) also gives bounded mean and comparable covariance for R in the estimated coordinates, as required by that proposition. Pull each candidate back to the original coordinates. One candidate obeys

$$
R ( H \triangle h ) \leq \operatorname { e r r } _ { D | K } ( h ) + \eta _ { R } \leq C \eta / p + C \varepsilon _ { 0 } .
$$

Choose $\alpha$ and the multiplier in $\varepsilon _ { 0 }$ so that this is below the $\delta _ { 0 }$ in Theorem 7.3. Then

$$
P ( H \triangle h ) \leq C p ( \eta / p + \varepsilon _ { 0 } ) \leq C \eta + C \varepsilon _ { 0 } .\tag{21}
$$

Adding η bounds its original label error. The same afine h is used on all of $\mathbb { R } ^ { d }$ . Retain every candidate from every surviving attempt, and select the lowest empirical original-distribution error on a fresh validation sample. This adds at most $C \varepsilon _ { 0 }$ with high probability.

The candidate list includes the outputs of all retained coordinate systems, mean estimates, translations, and optimization runs. Selection uses a fresh labeled validation sample from the original distribution. For the proof, fix any comparator within an arbitrarily small additive tolerance of $\mathrm { O P T } _ { D }$ before the algorithm’s randomness. Its successful branch exists by the argument above. Taking the comparator tolerance and internal accuracy suficiently small proves Theorem 1.1, once the polynomial implementation below is established.

## 9 Polynomial implementation and confidence

We specify polynomial sample and iteration bounds for every stage, including attempts whose statistical estimates are inaccurate. All raw sample batches are independent of prior outputs; accepted conditional samples depend on the previously chosen restriction. A conditional sample request is implemented by a bounded number of ordinary iid draws and rejection using the explicit halfspace.

## 9.1 Bounds on valid coordinates

Lemma 9.1 (A priori cap moment bounds). If the original P is isotropic log-concave and $P ( H ) = p $ then

$$
\lambda _ { \operatorname* { m i n } } ( \Sigma _ { Q } ) \geq c p ^ { 2 } , \quad \lambda _ { \operatorname* { m a x } } ( \Sigma _ { Q } ) \leq C \log ^ { 2 } ( e / p ) , \quad \| \mu _ { Q } \| \leq C \log ( e / p ) .\tag{22}
$$

Proof. Every scalar marginal of $P$ has density at most a universal constant. Its density conditional on H is at most $C / p ,$ , so the density-to-variance bound from Lemma 2.1 gives the first inequality. For every unit $v ,$ the original directional tail obeys $P ( | v ^ { T } X | > t ) \leq A e ^ { - a t }$ . Thus

$$
\mathbb { E } [ ( v ^ { T } X ) ^ { 2 } \mid H ] \leq \int _ { 0 } ^ { \infty } 2 t \operatorname* { m i n } \{ 1 , A e ^ { - a t } / p \} d t \leq C \log ^ { 2 } ( e / p ) .
$$

Taking directional suprema gives the covariance upper bound and the mean norm bound. □

Whenever $p \geq c \varepsilon _ { 0 }$ , a valid robust output lies in explicit polynomial spectral and norm bounds. Reject all outputs outside slightly looser bounds, including nonpositive-definite matrices. On every surviving branch,

$$
\mathbb { E } _ { P } \left\| \widehat { \boldsymbol { \Sigma } } ^ { - 1 / 2 } ( \boldsymbol { X } - \widehat { \boldsymbol { \mu } } ) \right\| ^ { 2 } \leq M _ { 2 }
$$

for a known polynomial $M _ { 2 } = \mathrm { p o l y } ( d , 1 / \varepsilon _ { 0 } )$ . This follows directly from isotropy and the imposed bounds, even when the output is inaccurate. All eigendecompositions and inverses here are in the declared real-arithmetic model.

## 9.2 Sample optimization of the hinge

Fix a surviving coordinate map and let Z denote its transformed feature. Choose a deterministic radius $R _ { 0 }$ polynomial in $M _ { 2 }$ and $1 / \varepsilon _ { 0 }$ . Truncate only the objective:

$$
L _ { R _ { 0 } } ( v ) = \mathbb { E } [ ( 1 + v ^ { T } Z ) _ { + } \mathbf { 1 } _ { \{ \| Z \| \leq R _ { 0 } \} } ] , \qquad \| v \| \leq B .
$$

The population marginal itself is not truncated. Uniformly over feasible $v ,$

$$
0 \leq L ( v ) - L _ { R _ { 0 } } ( v ) \leq M _ { 2 } / R _ { 0 } ^ { 2 } + B M _ { 2 } / R _ { 0 } .\tag{23}
$$

Take $R _ { 0 }$ large enough that the right side is at most $c \varepsilon _ { 0 }$ , hence at most a suficiently small multiple of $p$ in the dificult branch. The truncated summands are bounded by $1 + B R _ { 0 }$ and are $R _ { 0 } { \mathrm { - L i p s c h i t z } }$ in v. A Euclidean net of the radius-B ball with mesh $\zeta / R _ { 0 } .$ a Hoefding bound, and Lipschitz extension show that uniform empirical error at most $\zeta$ holds with

$$
n = O \left( \frac { ( 1 + B R _ { 0 } ) ^ { 2 } } { \zeta ^ { 2 } } \left[ d \log \frac { C B R _ { 0 } } { \zeta } + \log \frac { 1 } { \delta ^ { \prime } } \right] \right)
$$

samples. Here $\zeta = c \varepsilon _ { 0 }$ or a small multiple of ${ \widehat { q } } ,$ both polynomially bounded below in surviving relevant branches. The net is for the proof only. Projected subgradient descent on the empirical convex objective obtains additive $\zeta$ optimization error in $O ( B ^ { 2 } R _ { 0 } ^ { 2 } / \zeta ^ { 2 } )$ steps. Evaluating a full empirical subgradient per step still costs a polynomial. Combining optimization, uniform convergence, and (23) gives the desired additive $O ( p )$ hinge solution. Abort a zero slope in the rare branch; it cannot occur in a valid branch because its loss would be one.

## 9.3 Quantiles and rejection

Use a fresh sample and an empirical upper-tail quantile of $v ^ { T } Z$ with target mass $k = K _ { 0 } { \widehat { q } } .$ The uniform empirical CDF error can be made at most $e = c \varepsilon _ { 0 }$ by a polynomial sample bound, with confidence $1 - \delta ^ { \prime }$ . Include the order-statistic discretization error in e. Define the final threshold as the minimum of this empirical threshold and $- 1 / 2$ . This choice ensures $K \supseteq S$ . In a valid branch the true mass of $S$ is substantially below $k ,$ so this minimum does not change the expanded empirical quantile. Its true mass is $k \pm O ( e ) = \Theta ( p )$

Before substantial rejection sampling, estimate acceptance on another fresh batch. Abort a branch whose estimate is below a fixed polynomial lower threshold. Put a deterministic polynomial cap on every rejection count; Chernof bounds show that all valid branches with mass $\Theta ( p ) \ge c \varepsilon _ { 0 }$ survive with the allocated confidence. The same safeguards apply to observed positive samples and the two label classes in balanced restrictions. Rejection changes the distribution only by selecting accepted iid points; no synthetic or conditional oracle is used.

## 9.4 Balanced lists and optimization budgets

For each balanced-primitive call, independently repeat paired robust class-mean estimation. Retain every pair inside fixed norm and segment-length caps, and run all its ${ \cal O } ( 1 / \varepsilon _ { 0 } )$ grid points. A successful pair lies inside these caps by the proved conditional moment bounds. Inaccurate mean estimates may add candidates to the list, which are evaluated on the independent validation sample.

Every homogeneous run has a fixed geometric bandwidth grid and a predetermined polynomial trajectory length. Even when the statistical estimates are inaccurate, the sigmoid’s stochastic gradient is finite for every finite sample, the initial vector is unit, and the unnormalized update has norm at least one, since $\left\| w - s g \right\| ^ { 2 } = 1 + s ^ { 2 } \left\| g \right\| ^ { 2 }$ . Thus the retraction is defined, and the predetermined iteration budget bounds the operation count without assuming that branch’s geometry. All retained hypotheses are afine functions of the original features.

There is a predetermined polynomial upper bound N on the number of branches and output candidates. Allocate statistical failure probability $\delta ^ { \prime } = \delta / ( C N )$ to each fresh concentration or rejection guarantee. Robust moments need only constant probability per attempt; ${ \cal O } ( \log ( N / \delta ) )$ independent attempts leave at least one good output with high confidence. In balanced calls repeat the paired attempts, so both mean estimates are simultaneously good in at least one retained pair. Homogeneous trajectories are repeated similarly. The products of these fixed-stage polynomial list sizes remain polynomial. Every attempt terminates within its prescribed budget.

## 9.5 Sigmoid evaluation using arithmetic alone

The sigmoid in Section 5 need not be an exact elementary-function oracle. Let $N _ { \mathrm { m a x } }$ be the deterministic polynomial bound on the number of original draws in the complete capped algorithm. From the isotropic coordinate tails and a union bound, a deterministic

$$
B _ { \mathrm { r a w } } = C \sqrt { d } \log ( C d N _ { \mathrm { m a x } } / \delta )
$$

ensures that every original draw has norm at most $B _ { \mathrm { r a w } }$ except with probability at most $\delta / 1 0$ The implemented algorithm aborts if this bound is violated. All surviving coordinate maps, classmean outputs, and translations already have polynomial caps; hence every feature supplied to a homogeneous run then has norm at most a deterministic polynomial M.

The sigmoid derivative uses $a ( t ) = e ^ { - | t | } / ( 1 + e ^ { - | t | } ) ^ { 2 }$ . To approximate its stochastic gradient to norm error at most a small fixed $\zeta > 0$ , it sufices to approximate a uniformly to error

$$
e _ { \mathrm { n u m } } = c \zeta \sigma / ( M + 1 ) .
$$

Find the smallest nonnegative integer m with $2 ^ { - m } \leq e _ { \mathrm { n u m } } / 2$ by repeated halving. For $| t | > m$ return zero: $a ( t ) \leq e ^ { - | t | } \leq e ^ { - m } \leq 2 ^ { - m }$ . On $[ - m , m ]$ , the Taylor polynomial for $e ^ { - | t | }$ of degree Cm has absolute error at most $c e _ { \mathrm { n u m } }$ for a suficiently large fixed $C ,$ , by the exponential-series remainder bound. Clip its value to [0, 1] and apply $s \mapsto s / ( 1 + s ) ^ { 2 }$ , whose derivative is bounded on that interval. Since $m = O ( \log ( 2 / e _ { \mathrm { n u m } } ) )$ , this requires polynomially many arithmetic operations and comparisons. Norms and normalization use square roots as permitted by the computational model.

Keep the approximate gradient as a scalar multiple of the exact tangent projection $X - ( w ^ { T } X ) w$ Tangency, and thus the scale-invariant retraction argument, remains exact. To justify the analysis without conditioning the data on a norm event, consider a fictitious oracle: it uses this approximation on bounded draws and the exact gradient on oversized draws. Its pointwise vector error is at most $\zeta$ everywhere. It is tangent, and its conditional bias has norm at most $\zeta .$ The descent inequality consequently gains terms at most

$$
s \zeta \| \nabla F _ { \sigma } ( w ) \| + C s ^ { 2 } \zeta ^ { 2 } / \sigma ^ { 2 } .
$$

Young’s inequality absorbs the first into a fixed fraction of the gradient descent term plus $C s \zeta ^ { 2 }$ Choosing $\zeta$ suficiently small relative to the fixed stationarity threshold, and adjusting the fixed step and iteration constants, proves the same list guarantee. The implemented and fictitious algorithms coincide unless an oversized original draw occurs, whose probability is at most $\delta / 1 0$ . Since $\sigma \geq c \varepsilon _ { 0 }$ the arithmetic evaluation cost stays polynomial.

## 9.6 Validation and constants

A fresh original-distribution sample of size

$$
O \big ( \varepsilon _ { 0 } ^ { - 2 } ( \log N + \log ( 1 / \delta ) ) \big )
$$

uniformly estimates the errors of all retained candidates to additive $\varepsilon _ { \mathrm { 0 } }$ . Select an empirical minimizer. Combining (18), the balanced branch, and (21) proves the theorem after the fixed rescaling of $\varepsilon _ { 0 }$

In dimension one, threshold empirical risk minimization in both orientations, including constants, has the stronger additive agnostic guarantee and polynomial complexity. If the convention for “proper halfspace” excludes constants, replace them by 1 $\langle X _ { 1 } \geq T \}$ and $1 \{ X _ { 1 } \geq - T \}$ for $T = C \log ( 1 / \varepsilon _ { 0 } )$ choosing C using the universal isotropic log-concave tail bound. These finite thresholds difer from the constants on probability at most $\varepsilon _ { 0 }$ , which is absorbed by the choice of $\varepsilon _ { 0 }$ . The comparator infimum is unchanged by allowing constants, since their errors are limits of finite thresholds.

## 10 Unknown afine normalization and confidence amplification

We now remove the isotropy assumption and obtain the sample bound in Corollary 1.2. The preprocessing uses only empirical moments of the marginal. Labels are ignored in this step.

Lemma 10.1 (Stability under approximate normalization). Fix constants $M , c _ { - } , c _ { + } > 0$ . The guarantee of Theorem 1.1 continues to hold when its isotropy assumption is replaced by

$$
\| \mathbb { E } X \| \leq M , \qquad c _ { - } I \preceq \operatorname { C o v } ( X ) \preceq c _ { + } I .
$$

The risk constant and the coeficients of the polynomial complexity bound may depend on $M , c _ { - } , c _ { + }$

Proof. All cap geometry in Sections 4 and 7 is expressed in the cap’s own mean and covariance, so these arguments are unchanged. In the branch with a comparator of constant minority mass, the displayed assumptions are precisely those of Proposition 6.1.

For the remaining implementation bounds, every unit projection of X has a density bounded by a fixed constant and a tail bounded by $A \exp ( - a t )$ , with constants depending only on $M , c _ { - } , c _ { + }$ The proof of Lemma 9.1 therefore gives

$$
\lambda _ { \operatorname* { m i n } } ( \Sigma _ { Q } ) \geq c p ^ { 2 } , \quad \lambda _ { \operatorname* { m a x } } ( \Sigma _ { Q } ) \leq C \log ^ { 2 } ( e / p ) , \quad \| \mu _ { Q } \| \leq C \log ( e / p ) .
$$

Also $\mathbb { E } \left\| X \right\| ^ { 2 } \leq C ( d + 1 )$ , which supplies the bound $M _ { 2 }$ used in the hinge optimization. The same directional tails give the sample-norm bound in Section 9.5. These are the only uses of isotropy outside the cap normalization. Adjusting the fixed constants in the spectral, norm, and iteration bounds proves the claim. □

Lemma 10.2 (Empirical whitening at constant confidence). Let X have a full-dimensional logconcave law with unknown mean $\mu$ and covariance Σ. For $n = C d ^ { 2 }$ independent samples, let

$$
{ \widehat { \boldsymbol { \mu } } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \boldsymbol { X } } _ { i } , \qquad { \widehat { \boldsymbol { \Sigma } } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } ( { \boldsymbol { X } } _ { i } - { \widehat { \boldsymbol { \mu } } } ) ( { \boldsymbol { X } } _ { i } - { \widehat { \boldsymbol { \mu } } } ) ^ { T } .
$$

For a suficiently large universal $C _ { i }$ , with probability at least $9 / 1 0$

$$
\begin{array} { r } { \left\| \Sigma ^ { - 1 / 2 } ( \widehat { \mu } - \mu ) \right\| \leq \frac { 1 } { 8 } , \qquad \frac { 1 } { 2 } \Sigma \preceq \widehat { \Sigma } \preceq \frac { 3 } { 2 } \Sigma . } \end{array}\tag{24}
$$

On this event, the transformed feature $W = \widehat { \Sigma } ^ { - 1 / 2 } ( X - \widehat { \mu } )$ satisfies

$$
\| \mathbb { E } W \| \leq 1 , \qquad \frac { 2 } { 3 } I \preceq \operatorname { C o v } ( W ) \preceq 2 I .
$$

Proof. Use the true standardized variables $Z _ { i } = \Sigma ^ { - 1 / 2 } ( X _ { i } { - } \mu )$ for the analysis. Their one-dimensional fourth moments are bounded by a universal constant. Cauchy–Schwarz therefore gives

$$
\mathbb { E } \left\| Z \right\| ^ { 4 } = \sum _ { j , k } \mathbb { E } [ Z _ { j } ^ { 2 } Z _ { k } ^ { 2 } ] \leq C d ^ { 2 } .
$$

Writing $\begin{array} { r } { S = n ^ { - 1 } \sum _ { i } Z _ { i } Z _ { i } ^ { T } } \end{array}$ and $\overline { { Z } } = n ^ { - 1 } \sum _ { i } Z _ { i }$ , independence yields

$$
\mathbb { E } \left\| S - I \right\| _ { F } ^ { 2 } = \frac { 1 } { n } \mathbb { E } \left\| Z Z ^ { T } - I \right\| _ { F } ^ { 2 } \leq \frac { C d ^ { 2 } } { n } , \qquad \mathbb { E } \left\| \overline { { Z } } \right\| ^ { 2 } = \frac { d } { n } .
$$

For a suficiently large constant in $n = C d ^ { 2 }$ , Markov’s inequality and a union bound imply

$$
\begin{array} { r } { \| S - I \| _ { \mathrm { o p } } \leq \frac { 1 } { 8 } , \qquad \left\| \overline { { Z } } \right\| \leq \frac { 1 } { 8 } } \end{array}
$$

with probability at least 9/10. Since

$$
\Sigma ^ { - 1 / 2 } \widehat { \Sigma } \Sigma ^ { - 1 / 2 } = S - \overline { { Z } } \overline { { Z } } ^ { T } ,
$$

these estimates imply (24). Congruence by $\widehat { \Sigma } ^ { - 1 / 2 }$ gives the covariance bounds for $W .$ , and

$$
\begin{array} { r } { \left\| \mathbb { E } W \right\| ^ { 2 } = ( \widehat { \mu } - \mu ) ^ { T } \widehat { \Sigma } ^ { - 1 } ( \widehat { \mu } - \mu ) \leq 2 \left\| \Sigma ^ { - 1 / 2 } ( \widehat { \mu } - \mu ) \right\| ^ { 2 } \leq 1 . } \end{array}
$$

The algorithm computes the empirical quantities directly in the original coordinates. The unknown normalization is used only to establish their accuracy. □

Proof of Corollary 1.2. First construct a learner with a fixed success probability. Estimate $\mu$ and Σ as in Lemma 10.2, and check that $\widehat { \Sigma }$ is positive definite. If it is, apply the learner of Lemma 10.1, with internal accuracy $\varepsilon / 4$ and failure probability $1 / 1 0 \ / \tan \angle B$ to fresh samples of $( W , Y )$ . Conditional on the whitening event, its assumptions hold with the fixed bounds $M = 1 , c _ { - } = 2 / 3$ , and $c _ { + } = 2$ . An invertible afine change preserves the class of afine halfspaces, their classification errors, and $\mathrm { O P T } _ { D }$

If the learner returns a nonconstant halfspace $\mathbf { 1 } \{ v ^ { T } W \geq t \}$ , pull it back as

$$
\boldsymbol { w } = \widehat { \boldsymbol { \Sigma } } ^ { - 1 / 2 } \boldsymbol { v } , \qquad \boldsymbol { b } = \boldsymbol { t } + \boldsymbol { w } ^ { T } \widehat { \boldsymbol { \mu } } .\tag{25}
$$

If it returns a constant, replace that constant in $W$ coordinates by

$$
h _ { 0 } ( W ) = { \bf 1 } \{ W _ { 1 } \geq T \} , \qquad h _ { 1 } ( W ) = { \bf 1 } \{ W _ { 1 } \geq - T \} , \qquad T = C \log ( C / \varepsilon ) ,
$$

respectively. On the whitening event the one-dimensional log-concave tail bounds imply

$$
\mathbb { P } ( h _ { 0 } ( W ) \neq 0 ) + \mathbb { P } ( h _ { 1 } ( W ) \neq 1 ) \le \varepsilon / 4
$$

after choosing C suficiently large. Their pullbacks have nonzero normals and finite thresholds. To choose a threshold using arithmetic alone, find the least integer $j \geq 1$ for which $2 ^ { j } \geq C / \varepsilon$ by repeated doubling, and use $T = C ^ { \prime } j$ with a suficiently large fixed $C ^ { \prime }$ . Each successful run thus returns a nonconstant halfspace with error at most $C \mathrm { O P T } _ { D } + \varepsilon / 2$

If the empirical covariance is singular, or a subsequent stage exceeds its prescribed budget, return the fixed halfspace $\mathbf { 1 } \{ x _ { 1 } \geq 0 \}$ . No absolute spectral bound is imposed on the original $\widehat { \Sigma } \colon$ its eigenvalues may reflect an arbitrarily ill-conditioned marginal. The norm and spectral bounds of Section 9 are applied after whitening, in W coordinates. Consequently every run has a fixed polynomial sample and arithmetic budget, including those for which (24) fails. On the successful whitening event, all probabilistic guarantees of the normalized learner apply. A union bound shows that one full run succeeds with probability at least $4 / 5$

Repeat this entire procedure independently

$$
J = \lceil C \log ( 2 / \delta ) \rceil
$$

times and retain its $J$ outputs. With probability at least $1 - \delta / 2$ , one has error at most $C \mathrm { O P T } _ { D } + \varepsilon / 2$ Use an independent validation sample of size

$$
m = O \Bigl ( \varepsilon ^ { - 2 } \bigl ( \log ( 2 J ) + \log ( 2 / \delta ) \bigr ) \Bigr )
$$

to estimate all J errors to additive $\varepsilon / 4$ , simultaneously with probability at least $1 - \delta / 2$ . Their empirical minimizer has true error at most the smallest candidate error plus $\varepsilon / 2$ . This proves the risk bound.

Each fixed-confidence run uses $P _ { 0 } ( d , \varepsilon ^ { - 1 } )$ samples for a fixed polynomial $P _ { 0 }$ . Since $\log ( 2 J ) =$ ${ \cal O } ( \log ( 2 / \delta ) )$ , the total is at most $P ( d , \varepsilon ^ { - 1 } ) \log ( 2 / \delta )$ for a fixed polynomial P. Computing the empirical moments, transforming samples, and evaluating the J candidates on the common validation sample all take polynomially many real-arithmetic operations. None of these operation counts depends on the eigenvalue ratio of Σ. □

## 11 Discussion

The minority-mass comparison (17) explains why learning on a balanced restriction can yield a global constant-factor agnostic guarantee. Conditioning amplifies the label error by a factor of order $1 / p .$ while the comparison between conditional and global disagreement contributes the compensating factor p. Robust cap normalization and the zonoid-depth objective make this construction accessible from labeled samples. The output remains a single afine halfspace.

Empirical whitening extends the result to unknown full-dimensional log-concave marginals, and independent repetition makes the sample complexity linear in log $( 2 / \delta )$ . The polynomial exponents arising from robust moment estimation and the search over translations are not optimized. Obtaining smaller exponents, extending the geometry beyond log-concave marginals, and establishing finite-bit complexity bounds are natural directions for further work.

## References

[1] Pranjal Awasthi, Maria Florina Balcan, and Philip M. Long. The power of localization for eficiently learning linear separators with noise. Journal of the ACM, 63(6):50:1–50:27, 2017. doi:10.1145/3006384.

[2] Sergey Bobkov and Michel Ledoux. One-Dimensional Empirical Measures, Order Statistics, and Kantorovich Transport Distances. Memoirs of the American Mathematical Society, 261(1259), 2019. doi:10.1090/memo/1259.

[3] Ilias Diakonikolas, Daniel M. Kane, and Alistair Stewart. Learning geometric concepts with nasty noise. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC), pages 1061–1073, 2018. doi:10.1145/3188745.3188754. arXiv:1707.01242.

[4] Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, and Nikos Zarifis. Non-convex SGD learns halfspaces with adversarial label noise. In Advances in Neural Information Processing Systems, 33:18540– 18549, 2020. NeurIPS proceedings. arXiv:2006.06742.

[5] Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, and Nikos Zarifis. Learning general halfspaces with adversarial label noise via online gradient descent. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 5118–5141, 2022. PMLR 162:5118–5141.

[6] Ilias Diakonikolas, Daniel M. Kane, Sihan Liu, and Nikos Zarifis. Testable learning of general halfspaces with adversarial label noise. In Proceedings of the 37th Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 1308–1335, 2024. PMLR 247:1308–1335.

[7] Gleb Koshevoy and Karl Mosler. Zonoid trimming for multivariate distributions. The Annals of Statistics, 25(5):1998–2017, 1997. doi:10.1214/aos/1069362382.

[8] Alexei M. Kulik and Taras D. Tymoshkevych. Lift zonoid and barycentric representation on a Banach space with a cylinder measure. Lithuanian Mathematical Journal, 53(2):181–195, 2013. doi:10.1007/s10986-013-9202-z. arXiv:1211.2927.

[9] Pravesh K. Kothari and David Steurer. Outlier-robust moment-estimation via sum-of-squares. arXiv:1711.11581v2, 2017.

[10] Brayden Letwin. The KLS constant is $O ( \log ^ { 1 / 4 } n )$ . arXiv:2607.24164, 2026.

[11] László Lovász and Santosh Vempala. The geometry of logconcave functions and sampling algorithms. Random Structures & Algorithms, 30(3):307–358, 2007. doi:10.1002/rsa.20135.

[12] Aleksandr Storozhenko. On the SoS certifiability of log-concave distributions. arXiv:2609.30105v2, 2026.