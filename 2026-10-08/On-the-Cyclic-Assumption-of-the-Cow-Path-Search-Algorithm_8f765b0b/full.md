# On the Cyclic Assumption of the Cow-Path Search Algorithm

Yuan Ma

Yiqun Lisa Yin<sup>∗</sup>

## Abstract

In the cow-path problem, a cow must find a goal lying at an unknown distance on one of w paths connected only at the origin, and performance is measured by competitive ratio. An eficient randomized algorithm is designed in [2], in which the cow visits the paths in a fixed cyclic order. The algorithm is proved to be optimal for $w = 2$ in [2], and for all w in [3] with a claim that no algorithm does better than the best cyclic one. This note provides a detailed proof of that claim.

## 1 Introduction

In the w-lane cow-path problem, there are w infinite paths from a common origin and a goal lying at an unknown distance on one of these paths. A cow (or robot) starts at the origin but does not know which path contains the goal or how far the goal is, and it discovers the goal only upon reaching it. Its objective is to find the goal while minimising the total distance travelled, and its performance is measured by competitive ratio.

Baeza-Yates, Culberson and Rawlins [1] determined the optimal deterministic competitive ratio, $1 + 2 w ^ { w } / ( w - 1 ) ^ { w - 1 }$ , which is 9 for w = 2. Kao, Reif and Tate [2] introduced a randomized algorithm with competitive ratio

$$
R _ { w } = \operatorname* { m i n } _ { r > 1 } \Bigl \{ 1 + \frac { 2 } { w } \cdot \frac { 1 + r + \cdots + r ^ { w - 1 } } { \ln r } \Bigr \} ,
$$

which is approximately 4.5911 for $w = 2 .$ , proved its optimality for $w = 2$ , and conjectured its optimality for $w > 2$ . Kao, Ma, Sipser and Yin [3] subsequently proved this conjecture.

A common feature of these analyses is that the algorithm visits the paths in a fixed cyclic order, which means that the algorithm visits all the w paths in its first w visits and then repeatedly searches these paths in that order. Fact 4 in [3] stated explicitly that it sufices to analyze these cyclic algorithms but left a detailed proof to a manuscript by us [4], which was never formally published. This note constructs a complete proof from some surviving hand-written notes.

## 2 Preliminaries

In this section, we formally introduce terminology and notation. Appendix B will list all our terms and notation for ease of reference.

## 2.1 Algorithms

A search algorithm is represented by a sequence

$$
\begin{array} { r } { X = \left( ( P _ { t } , \ell _ { t } ) \right) _ { t \geq 1 } , } \end{array}
$$

fixed in advance and independent of the goal’s location, where $P _ { t }$ and $\ell _ { t }$ are the path and length of step $t ,$ respectively. To avoid any ambiguity, a step in an algorithm sweeps a path from the origin to a finite distance and then returns to the origin.

We write $f _ { t } ( P )$ for path $P { ^ { \circ } \mathrm { s } }$ frontier, defined to be the distance to which it has been swept when step t is about to begin. Note that $f _ { t + 1 } ( P _ { t } ) = \ell _ { t }$

Step t costs $2 \ell _ { t }$ in travelled distance, out and back—with one exception, where the goal located at some distance h from the origin is found during this step. On that step the cow stops at the goa on the outward leg and does not return, so step $t = k$ costs h rather than $2 \ell _ { k }$ , and the algorithm ends. The total distance travelled is $\textstyle 2 \sum _ { i < k } \ell _ { i } + h$ , the sum running over steps $1 , \ldots , k - 1$ and excluding step k itself.

Throughout, $w \geq 2 ;$ for $w = 1$ the cow simply walks down the single path and the ratio is 1. Without loss of generality, we only need to work on algorithms with a finite expected competitive ratio, because such an algorithm was constructed in [2] already. This requires that the cow will not continue on a path indefinitely $( \mathrm { i . e . , \ell } _ { t }$ is finite for all t) and it will search every path without a constant upper bound (i.e., each path’s sweep lengths are unbounded): otherwise the goal may never be found, leading to an infinite expected competitive ratio.

## 2.2 The distribution and competitive ratio

At the highest level, Yao’s principle [6] allows us to lower-bound the worst-case expected ratio of every randomized algorithm by proving a lower bound on the expected ratio of every deterministic algorithm under a fixed input distribution. The goal’s distribution is that specified in [2, p. 70], where the path containing the goal is chosen uniformly from the w paths, and the distance x along that path has density

$$
f _ { \epsilon , w } ( x ) = \left\{ { \begin{array} { l l } { \epsilon x ^ { - ( 1 + \epsilon ) } , } & { x \geq 1 , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.
$$

Throughout, $\epsilon > 0$ is fixed.

Let H denote the per-path measure, with density

$$
d H ( h ) = \frac { \epsilon } { w } h ^ { - ( 1 + \epsilon ) } d h , \qquad h \geq 1 .
$$

Thus each path carries mass $1 / w$ . If the goal is found at step $k ,$ the competitive ratio is

$$
1 + \frac { 2 \sum _ { i < k } \ell _ { i } } { h } ,
$$

where the leading 1 accounts for the final trip to the goal and the remaining term accounts for unsuccessful round trips.

Deleting search steps that do not cover any new ground, we can also assume

$$
\ell _ { t } > f _ { t } ( P _ { t } ) \qquad { \mathrm { f o r ~ e v e r y ~ } } t .\tag{1}
$$

This way, the outward leg retraces the already swept portion $[ 0 , f _ { t } ( P _ { t } ) ]$ , and only $( f _ { t } ( P _ { t } ) , \ell _ { t } ] =$ $\left( f _ { t } ( P _ { t } ) , f _ { t + 1 } ( P _ { t } ) \right]$ is searched for the first time. Under this assumption (otherwise the integrals below

would be ill-defined), summing over the goal locations gives

$$
\mathrm { e x p e c t e d \ c o m p e t i t i v e \ r a t i o \ o f \ } X = \sum _ { k } \int _ { f _ { k } ( P _ { k } ) } ^ { \ell _ { k } } \frac { 2 \sum _ { i < k } \ell _ { i } + h } { h } d H ( h ) .\tag{2}
$$

Each unit of unsuccessful travel adds $1 / h$ to the ratio. For a path searched to frontier x, define

$$
\phi ( x ) \ { \stackrel { \mathrm { d e f } } { = } } \ \int _ { x } ^ { \infty } { \frac { d H ( h ) } { h } } = { \frac { \epsilon } { w ( 1 + \epsilon ) } } x ^ { - ( 1 + \epsilon ) } .\tag{3}
$$

For ease of notation, define

$$
\alpha \stackrel { \mathrm { d e f } } { = } 1 + \epsilon , \qquad \beta \stackrel { \mathrm { d e f } } { = } ( 1 + \alpha ) ^ { 1 / \alpha } , \qquad \kappa \stackrel { \mathrm { d e f } } { = } \frac { \epsilon } { w ( 1 + \epsilon ) } .\tag{4}
$$

Then

$$
\phi ( x ) = \kappa x ^ { - \alpha } .\tag{5}
$$

Also,

$$
\beta ^ { \alpha } = 1 + \alpha = 2 + \epsilon ,\tag{6}
$$

so $\beta > 1$ , and $\beta  2$ as $\epsilon \to 0$

## 2.3 States, remaining weights, and contributions

The state of the search is defined by the collection of all paths’ frontiers plus the location of the cow. By default, the cow is understood to be at the origin, when not specified explicitly, and so a state can be understood to be the collection of all frontiers of the paths. With such an understanding, we use capital letter $F$ to denote a state.

In particular, F assigns to each path P a frontier $F ( P )$ . We write $F _ { t }$ for the state when step t is about to be taken, so that $F _ { t } ( P ) = f _ { t } ( P )$

We call $\phi ( x )$ the remaining weight on the path: the probability mass beyond x normalized by $h ,$ reflecting that the same extra travel is more costly relative to a nearby goal. The function ϕ is positive, strictly decreasing, and smooth; searching farther reduces this remaining weight. The constant κ is an overall positive factor in every comparison below and cancels; only the exponent α matters.

For a given state $F _ { ; }$ , summing the remaining weights over all paths gives us the total remaining weight

$$
\Phi ( F ) \ { \stackrel { \mathrm { d e f } } { = } } \ \sum _ { P } \phi { \big ( } F ( P ) { \big ) } , \ { \mathrm { a n d } }\tag{7}
$$

$$
\Phi _ { t } \ { \stackrel { \mathrm { d e f } } { = } } \ \Phi ( F _ { t } ) .\tag{8}
$$

The total remaining weight depends only on the current frontiers and is unchanged by relabelling paths. Note that under such definitions, $\Phi _ { t }$ and $\Phi _ { t + 1 }$ are total remaining weights before and after step t, respectively.

We next define the contribution for a full round-trip step t that travels the distance $2 \ell _ { t }$ for goal locations that remain un-searched after step t. The total remaining weight after step t is $\Phi _ { t + 1 }$ and step t contributes $2 \ell _ { t } \Phi _ { t + 1 }$ to the expected future cost. Summing over all steps of algorithm X starting from F gives us the following:

$$
E _ { F } ( X ) \ { \stackrel { \mathrm { d e f } } { = } } \ 2 \sum _ { t } \ell _ { t } \Phi _ { t + 1 } .\tag{9}
$$

Notice that the above quantity is unchanged by permuting the labels of the paths, since $\Phi$ is a symmetric sum and all paths carry the same $\phi$ and the same dynamics. In addition, $\ell _ { t }$ afects both the travel distance now and the post-step remaining weight through $\phi ( \ell _ { t } )$ : a longer sweep costs more travel now but leaves less remaining weight.

For the initial state $F _ { 0 } ,$ , in which $F _ { 0 } ( P ) = 1$ for every path $P ,$ interchanging the nonnegative terms in (2) gives the expected competitive ratio

$$
1 + E _ { F _ { 0 } } ( X ) .\tag{10}
$$

Minimising the expected competitive ratio is therefore equivalent to minimising $E _ { F _ { 0 } }$ , which we will work on in the rest of the paper.

Equation (9) is a sum over all steps, so comparisons of algorithms modified from a certain step onwards must account for their continuations, defined to be the remaining steps of an algorithm starting from a certain state. By (9) the contribution of a step depends on the past only through the current state. Identical continuations started at the same state have equal contributions, and the infimum of the contribution of a continuation depends only on that state.

Finally, since we only need to deal with algorithms with finite competitive ratios, we assume in all our arguments that the algorithms we will modify all have finite contributions.

## 3 Proposition 1: prioritize a least-frontier path

Proposition 1 (least-frontier first). If algorithm X sweeps a path whose frontier is strictly larger than the least frontier at some step, then X can be modified into another algorithm $X ^ { * }$ such that starting from that state and that step the contribution of $X ^ { \ast }$ is strictly less than that of X.

Proof. Suppose that in X, starting from state F, step t first sweeps path $B ,$ whose frontier is $b ,$ to a distance $y > b ,$ while another path A has frontier $a < b$

Construct $X ^ { * }$ from X as follows. Make no change before step $t ,$ but let step t sweep A to $y ,$ which is larger than $b > a ,$ , and then follow the remaining steps of X with the labels A and B interchanged, keeping all search distances unchanged.

To make $X ^ { * }$ satisfy condition (1), we skip any prescribed sweep of B whose searched distance is at most $b$ (note the relabelled continuation of X starts with A at $y$ and $B$ at $^ { a , }$ whereas $X ^ { * }$ starts its continuation with $A$ at $y$ and B already at $b > a )$ . The prescribed search distances on $B$ are strictly increasing and unbounded, and so only finitely many sweeps are skipped.

![](images/0337bcff85d48c846515e2f2451257c65a1045744769b5a877b177ed3115a7dc.jpg)  
Figure 1: Comparison of X and $X ^ { * }$ in the proof of Proposition 1.

For $X$ , let $C _ { 1 } , C _ { 2 }$ be the contributions of step t and continuation, respectively. Let $C _ { 1 } ^ { * } , C _ { 2 } ^ { * }$ be the corresponding contributions of $X ^ { * }$ . It sufices to show

$$
C _ { 1 } ^ { * } < C _ { 1 } , \qquad C _ { 2 } ^ { * } \leq C _ { 2 } .
$$

For step $t , X$ leaves $A$ at a and advances $B$ to $y ,$ while $X ^ { * }$ advances A to y and leaves B at b. All other frontiers are unchanged. The post-step remaining weights are therefore

$$
\Phi ( F ) - \phi ( b ) + \phi ( y ) \quad { \mathrm { a n d } } \quad \Phi ( F ) - \phi ( a ) + \phi ( y ) ,
$$

for $X$ and $X ^ { \ast }$ respectively. Both steps have length $y ,$ so

$$
\begin{array} { c } { { C _ { 1 } - C _ { 1 } ^ { * } = 2 y [ \Phi ( F ) - \phi ( b ) + \phi ( y ) ] - 2 y [ \Phi ( F ) - \phi ( a ) + \phi ( y ) ] } } \\ { { { } } } \\ { { = 2 y [ \phi ( a ) - \phi ( b ) ] > 0 , } } \end{array}\tag{11}
$$

because $a < b$ and $\phi$ is strictly decreasing.

For the continuation, interchanging the labels A and B preserves the contribution of $X ,$ , since Φ is symmetric in the path frontiers. Compare this relabelled continuation with that of $X ^ { \ast }$ . Until the first prescribed sweep of B beyond $b ,$ the frontier of B in $X ^ { * }$ is at least as large as that of $X$ while all other frontiers agree at corresponding steps. From that sweep onward, the states coincide. Consequently, every retained step has the same length and no larger post-step remaining weight, and so it contributes no more. The skipped steps contribute nothing. Thus

$$
C _ { 2 } ^ { * } \leq C _ { 2 } .
$$

Combining the comparisons gives

$$
E _ { F } ( X ^ { * } ) = C _ { 1 } ^ { * } + C _ { 2 } ^ { * } < C _ { 1 } + C _ { 2 } = E _ { F } ( X ) .
$$

## 4 Proposition 2: each sweep advances its path by strictly more than a factor $\beta$

In what follows, let t be a step in algorithm X, let $t ^ { + }$ be the next step sweeping $P _ { t } .$ , and define

$$
W _ { t } \stackrel { \mathrm { d e f } } { = } \displaystyle \sum _ { t \leq j < t ^ { + } } \ell _ { j } , \qquad \widehat { W } _ { t } \stackrel { \mathrm { d e f } } { = } \displaystyle \sum _ { t < j < t ^ { + } } \ell _ { j } .\tag{12}
$$

Lemma 1 (stationarity). Unless

$$
- \phi ^ { \prime } ( \ell _ { t } ) W _ { t } = \Phi _ { t + 1 } ,\tag{13}
$$

replacing $\ell _ { t }$ by a suitable $y$ (and leaving all other steps unchanged) yields an algorithm contributing strictly less than $X$

Proof. Define

$$
{ \widehat { \Phi } } _ { t + 1 } \ { \stackrel { \mathrm { d e f } } { = } } \ \Phi _ { t + 1 } - \phi ( \ell _ { t } ) ,
$$

the total remaining weight (post-step-t) associated with all paths other than $P _ { t }$

We vary only the length of step $t ,$ replacing $\ell _ { t }$ by $y ,$ while leaving all other steps unchanged. The feasible range is $f _ { t } ( P _ { t } ) < y < \ell _ { t ^ { + } } , \mathrm { s o } ~ y = \ell _ { t }$ is an interior point and can be varied in either direction.

$$
2 y [ \hat { \Phi } _ { t + 1 } + \phi ( y ) ]
$$

$$
y ,
$$

$$
t ^ { + }
$$

$$
P _ { t }
$$

$$
2 \widehat { W } _ { t } \phi ( y )
$$

$$
t ^ { + }
$$

$$
\ell _ { t ^ { + } }
$$

$$
C ( y ) = K + 2 y [ \widehat { \Phi } _ { t + 1 } + \phi ( y ) ] + 2 \widehat { W } _ { t } \phi ( y ) ,\tag{14}
$$

where K collects all contributions unafected by $y$

Diferentiating (14),

$$
C ^ { \prime } ( y ) = 2 \widehat { \Phi } _ { t + 1 } + 2 \phi ( y ) + 2 ( y + \widehat { W } _ { t } ) \phi ^ { \prime } ( y ) .
$$

At $\boldsymbol y = \boldsymbol { \ell _ { t } }$ , we have $\widehat { \Phi } _ { t + 1 } + \phi ( \ell _ { t } ) = \Phi _ { t + 1 }$ and $\ell _ { t } + \widehat { W } _ { t } = W _ { t }$ , and hence

$$
C ^ { \prime } ( \ell _ { t } ) = 2 \Phi _ { t + 1 } + 2 \phi ^ { \prime } ( \ell _ { t } ) W _ { t } .
$$

If this derivative is positive, a small decrease of $\ell _ { t }$ improves the algorithm; if it is negative, a small increase does. Thus an algorithm admitting no one-step variation improvement must satisfy

$$
\Phi _ { t + 1 } = - \phi ^ { \prime } ( \ell _ { t } ) W _ { t } .
$$

Lemma 1 says that each step must balance its current sweep length against the accumulated sweep lengths until that path is swept again. We call X stationary if equality (13) holds at every step.

Proposition 2 (advance factor). For a stationary algorithm, if

$$
\ell _ { t } \leq \beta f _ { t } ( P _ { t } )
$$

at some step t, then deleting step t yields an algorithm with strictly less contribution.

Proof. Define

$$
p \stackrel { \mathrm { d e f } } { = } f _ { t } ( P _ { t } ) .
$$

Deleting step t saves its contribution $2 \ell _ { t } \Phi _ { t + 1 }$ . However, until step $t ^ { + }$ , path $P _ { t }$ remains at frontier p rather than $\ell _ { t } ,$ adding contribution

$$
2 [ \phi ( p ) - \phi ( \ell _ { t } ) ] \widehat { W } _ { t } .
$$

Note that condition (1) holds since the next sweep of this path reaches $\ell _ { t ^ { + } } > \ell _ { t } > f _ { t } ( P _ { t } )$ and that every path remains swept without bound. Thus it sufices to show that the additional contribution is smaller than the saving.

Since X is stationary, by Lemma 1, the following equality holds:

$$
\Phi _ { t + 1 } = - \phi ^ { \prime } ( \ell _ { t } ) W _ { t } .
$$

Since $- \phi ^ { \prime } ( \ell _ { t } ) = \alpha \phi ( \ell _ { t } ) / \ell _ { t }$ , we obtain

$$
\alpha \phi ( \ell _ { t } ) \widehat { W } _ { t } = \ell _ { t } \Phi _ { t + 1 } - \alpha \ell _ { t } \phi ( \ell _ { t } ) < \ell _ { t } \Phi _ { t + 1 } .\tag{15}
$$

Now let $\rho = \ell _ { t } / p$ . Since $1 < \rho \le \beta$ and $\beta ^ { \alpha } = 1 + \alpha$ , we have

$$
\phi ( p ) - \phi ( \ell _ { t } ) = \phi ( \ell _ { t } ) ( \rho ^ { \alpha } - 1 ) \leq \alpha \phi ( \ell _ { t } ) .\tag{16}
$$

Combining inequalities (15) and (16) gives

$$
\begin{array} { r } { [ \phi ( p ) - \phi ( \ell _ { t } ) ] \widehat { W } _ { t } \leq \alpha \phi ( \ell _ { t } ) \widehat { W } _ { t } < \ell _ { t } \Phi _ { t + 1 } . } \end{array}\tag{17}
$$

The deletion penalty is therefore strictly smaller than the saving, so deleting step t yields an algorithm contributing strictly less. □

## 5 Proposition 3: step lengths are nondecreasing

In this section, we analyze the efect of exchanging the search lengths of two consecutive sweeps, especially when the lengths decrease. By relabelling the paths, we can reduce the comparison to the two exchanged steps.

Let X be an algorithm with a finite contribution. Suppose step k sweeps path A from a to $u ,$ and step $k + 1$ sweeps a distinct path B from b to v. Define $X ^ { \ast }$ as follows:

• Steps $1 , \ldots , k - 1$ are the same as in X.

• Step k sweeps A from a to v, and step $k + 1$ sweeps B from b to u.

• For every step $j \ge k + 2$ , use the same sweep length as X, with the path labels A and B interchanged.

Lemma 2 (exchange of adjacent sweep lengths). For any algorithm X and its corresponding $X ^ { * }$ , if min $( u , v ) > \operatorname* { m a x } ( a , b )$ , then

$$
E _ { F _ { 0 } } ( X ) - E _ { F _ { 0 } } ( X ^ { * } ) = 2 [ ( u - v ) \phi ( b ) + v \phi ( u ) - u \phi ( v ) ] .\tag{18}
$$

Proof. First, the two algorithms agree before step k. Secondly, after step $k + 1$ , paths $A , B$ have frontiers u, v in X and $v , u$ in $X ^ { \ast }$ and all other frontiers agree. Interchanging the labels A, B after step $k + 1$ therefore makes the two states identical. So the continuations of X and $X ^ { * }$ are the same.

![](images/2c24f993bad5e2e0850aabc63205e0786b8141d6e191b9934698c6ce1f472817.jpg)  
Figure 2: Comparison of X and $X ^ { * }$ in Lemma 2. After relabelling, the frontiers of $X ^ { * }$ match those of X, allowing the same continuation of the algorithm.

In sum, only steps k and $k { + 1 }$ contribute to any diference. Moreover, since min $( u , v ) > \operatorname* { m a x } ( a , b )$ both algorithms satisfy condition (1) and so we can use our established formulae to compute the diference of $E _ { F _ { 0 } }$ for the two algorithms. Write $\Phi _ { k + 1 } ^ { * }$ and $\Phi _ { k + 2 } ^ { * }$ for the corresponding remaining weights in $X ^ { * }$ . Since the states after the two sweeps difer only by relabelling,

$$
\Phi _ { k + 2 } ^ { * } = \Phi _ { k + 2 } .
$$

Also,

$$
\Phi _ { k + 1 } = \Phi _ { k + 2 } + \phi ( b ) - \phi ( v ) , \qquad \Phi _ { k + 1 } ^ { * } = \Phi _ { k + 2 } + \phi ( b ) - \phi ( u ) .
$$

Thus

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \big [ E _ { F _ { 0 } } ( X ) - E _ { F _ { 0 } } ( X ^ { * } ) \big ] = u \Phi _ { k + 1 } + v \Phi _ { k + 2 } - v \Phi _ { k + 1 } ^ { * } - u \Phi _ { k + 2 } } \\ & { \qquad = u \big [ \phi ( b ) - \phi ( v ) \big ] - v \big [ \phi ( b ) - \phi ( u ) \big ] } \\ & { \qquad = ( u - v ) \phi ( b ) + v \phi ( u ) - u \phi ( v ) . } \end{array}
$$

Note that (18) involves only $\phi ( b )$ , the frontier of the path swept second, and not $\phi ( a )$

Proposition 3 (monotonicity). Let X be an algorithm started from $F _ { 0 }$ such that no algorithm from $F _ { 0 }$ contributes strictly less than X. Then the step lengths of X are nondecreasing.

Proof. Because no algorithm contributes strictly less than X, no local modification improves it either. Also, X has finite contribution, as otherwise an algorithm sweeping the paths cyclically with a fixed geometric ratio larger than 1 will be an improvement. Hence, by Proposition 1, Lemma 1 and Proposition 2 in contrapositive form,

• every step sweeps a path of least current frontier;

• X is stationary;

• every step satisfies $\ell _ { t } > \beta f _ { t } ( P _ { t } )$

Suppose the step lengths are not nondecreasing in X, and choose an adjacent decrease (i.e., $\ell _ { k + 1 } < \ell _ { k } )$ . Write

$$
u \stackrel { \mathrm { d e f } } { = } \ell _ { k } , \qquad v \stackrel { \mathrm { d e f } } { = } \ell _ { k + 1 } , \qquad \mathrm { s o } v < u .
$$

Steps k and $k + 1$ sweep diferent paths, because two consecutive sweeps of the same path must strictly increase its frontier. Let

$$
A \ { \stackrel { \mathrm { d e f } } { = } } \ P _ { k } , \qquad B \ { \stackrel { \mathrm { d e f } } { = } } \ P _ { k + 1 } , \qquad a \ { \stackrel { \mathrm { d e f } } { = } } \ f _ { k } ( A ) , \qquad b \ { \stackrel { \mathrm { d e f } } { = } } \ f _ { k } ( B ) = f _ { k + 1 } ( B ) .
$$

Since step k sweeps a path of least frontier,

$$
a \leq b .\tag{19}
$$

Also, Proposition 2 applied to step $k + 1$ gives

$$
v > \beta b .\tag{20}
$$

The above two inequalities imply min $( u , v ) > \operatorname* { m a x } ( a , b )$ , and so we can use Lemma 2 now. Substituting $\phi ( x ) = \kappa x ^ { - \alpha }$ into (18) and cancelling $2 \kappa > 0$ , the exchange strictly lowers the contribution if and only if

$$
( u - v ) b ^ { - \alpha } + v u ^ { - \alpha } - u v ^ { - \alpha } > 0 .\tag{21}
$$

To prove (21), we may assume $b = 1 \cdot$ : dividing $u , v , b$ by $b > 0$ multiplies its left-hand side by the positive factor $b ^ { \alpha - 1 }$ , preserving its sign. We then have $u > v > \beta$ . Define

$$
g ( u , v ) \stackrel { \mathrm { d e f } } { = } ( u - v ) + v u ^ { - \alpha } - u v ^ { - \alpha } .
$$

We must show $g ( u , v ) > 0 .$

For fixed v, we have $g ( v , v ) = 0$ . Thus it is enough to show that g increases as u moves above v. Diferentiating with respect to u,

$$
\frac { \partial g } { \partial u } = 1 - v ^ { - \alpha } - \alpha v u ^ { - \alpha - 1 } .
$$

The role of Proposition 2 is now clear: $u > v$ alone is not enough. At the diagonal,

$$
\frac { \partial g } { \partial u } ( v , v ) = 1 - ( 1 + \alpha ) v ^ { - \alpha } ,
$$

which is positive exactly when $v > \beta .$ , since $\beta ^ { \alpha } = 1 + \alpha$

Moreover, for $u > v , u ^ { - \alpha - 1 } < v ^ { - \alpha - 1 }$ , so $\alpha v u ^ { - \alpha - 1 } < \alpha v v ^ { - \alpha - 1 } = \alpha v ^ { - \alpha }$ . Therefore,

$$
\frac { \partial g } { \partial u } = 1 - v ^ { - \alpha } - \alpha v u ^ { - \alpha - 1 } > 1 - ( 1 + \alpha ) v ^ { - \alpha } ,
$$

which is already proved to be positive in the diagonal case. Thus (21) is proved. So Lemma 2 gives an algorithm $X ^ { * }$ that contributes strictly less than X, leading to a contradiction. □

## 6 Main result

The theorem below shows that the lower bound established in [3] for cyclic algorithms is a lower bound for all algorithms. Therefore, the cyclic assumption may be dropped, as claimed in Fact 4 of [3].

Theorem. The least expected competitive ratio over all algorithms started from the state $F _ { 0 }$ can be attained by a cyclic algorithm.

Proof. First, we should not take for granted that an optimal algorithm exists: there could be infinitely many algorithms that approach optimality without ever reaching it. The proof that an optimal algorithm exists for our chosen distribution is rather technical and left to Appendix A. Here, we simply start with an optimal algorithm X and prove that X can be converted into a cyclic algorithm.

By Propositions 1 and 3, every step of X sweeps a path of least frontier and the sweeping lengths satisfy $\ell _ { 1 } \leq \ell _ { 2 } \leq \dots$ . Note that each path’s frontier is the length of its last sweep, or 1 if unswept. We construct an algorithm Y from X by breaking every least-frontier tie in favor of the least recently swept path and interchanging the tied labels from that step on; this preserves the multiset frontiers and the contribution.

The resulting algorithm Y is cyclic. Since every $\ell _ { t } > 1$ , the first w steps sweep distinct paths. Inductively, suppose steps $t - w + 1 , \dotsc , t$ sweep distinct paths. The least frontier is then $\ell _ { t - w + 1 }$ held by $P _ { t - w + 1 }$ , which is also the least recently swept path. Step $t + 1$ therefore sweeps $P _ { t - w + 1 } ,$ continuing the cyclic order. The cyclic algorithm Y attains the least competitive ratio. □

## 7 Discussion

Professor Allan Borodin recently pointed out to us that Schuierer [5] introduced another probability distribution to give an alternative and simpler proof of the lower bound established in [3], but his proof relied on the cyclic assumption. It is natural to ask if one can prove an optimal algorithm must be cyclic under his distribution (a truncated $H _ { D }$ where density is 0 beyond a bound D). However, our arguments use the facts that $\phi$ is positive, strictly decreasing, and smooth, whereas under $H _ { D }$ the function $\phi$ vanishes at and beyond D. Moreover, each path is swept to D in finitely many steps in [5], and algorithm modifications like ours become dificult. Nevertheless, numerics for $w = 3$ suggest the optimal algorithm may still be cyclic.

## Acknowledgements

We thank Professor Allan Borodin for his recent inquiry about the cyclic assumption, which prompted this research note. His inquiry made us realize there is a continued interest in our proof, which was obtained in 1994 but probably not typed up.

With eforts, we were able to find only some hand-written claims and proof sketches that survived two cross-country moves. The detailed proof presented here is constructed from those notes, with the aid of AI, especially Claude Opus 5, in September 2026.

We thank Professor Dan Kleitman again for his invaluable discussions, which helped us come up with the original proof when we conducted this research at MIT in the 1990s.

## References

[1] R. A. Baeza-Yates, J. C. Culberson and G. J. E. Rawlins, Searching in the plane, Information and Computation 106 (1993), 234–252.

[2] M.-Y. Kao, J. H. Reif and S. R. Tate, Searching in an unknown environment: an optimal randomized algorithm for the cow-path problem, Information and Computation 131 (1996), no. 1, 63–79; preliminary version in Proc. 4th ACM–SIAM Symposium on Discrete Algorithms (1993), 441–447.

[3] M.-Y. Kao, Y. Ma, M. Sipser and Y. Yin, Optimal constructions of hybrid algorithms, Journal of Algorithms 29 (1998), 142–164; preliminary version in Proc. 5th ACM–SIAM Symposium on Discrete Algorithms (1994), 372–381.

[4] Y. Ma and Y. Yin, On the cyclic assumption of an on-line search algorithm, manuscript, 1994; earlier version of this note.

[5] S. Schuierer, A lower bound for randomized searching on m rays, in: Computer Science in Perspective (Ottmann Festschrift), Lecture Notes in Computer Science 2598, Springer, 2003, 264–277.

[6] A. C.-C. Yao, Probabilistic computations: toward a unified measure of complexity, Proc. 18th IEEE Symposium on Foundations of Computer Science (1977), 222–227.

## A Existence of an optimal algorithm

We begin by defining the following infimum, as opposed to a minimum, because we do not want to assume the infimum is reachable by an algorithm:

$$
V ( F ) \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { i n f } _ { X { \mathrm { ~ s t a r t e d ~ a t ~ } } F } E _ { F } ( X ) .
$$

The proof in this section is rather technical but independent of any of the propositions proved in earlier sections. In particular, the following general statement has nothing to do with cyclicity.

Statement. For $w \geq 2$ , there exists an algorithm X starting from $F _ { 0 }$ such that $E _ { F _ { 0 } } ( X ) = V ( F _ { 0 } )$

Proof. Write $V _ { 0 } = V ( { \cal F } _ { 0 } )$ , which is finite because sweeping the paths cyclically with geometrically increasing lengths gives finite contribution. Choose algorithms ${ \cal X } ^ { ( 1 ) } , { \cal X } ^ { ( 2 ) } , . .$ . from $F _ { 0 }$ such that

$$
E _ { F _ { 0 } } ( X ^ { ( n ) } ) \leq V _ { 0 } + 1 , \qquad E _ { F _ { 0 } } ( X ^ { ( n ) } ) \longrightarrow V _ { 0 } .
$$

Denote step lengths and swept paths of $X ^ { ( n ) }$ by $\ell _ { t } ^ { ( n ) }$ and $P _ { t } ^ { ( n ) }$ . We will extract a limiting sequence and show that it yields an algorithm attaining $V _ { 0 }$

Part 1: bound each step length so that convergent subsequences can be extracted. We show that, for every fixed t, there is a finite constant $B _ { t }$ , independent of n, such that $\ell _ { t } ^ { ( n ) } \leq B _ { t }$ . The proof proceeds by induction on t. Since every term of (9) is nonnegative,

$$
2 \ell _ { t } ^ { ( n ) } \Phi _ { t + 1 } ^ { ( n ) } \leq V _ { 0 } + 1 \qquad { \mathrm { f o r ~ a l l ~ } } n , t .\tag{22}
$$

Base case: $t = 1$ . The $w - 1$ paths other than $P _ { 1 } ^ { ( n ) }$ still have frontier 1 after the first step. Thus $\Phi _ { 2 } ^ { ( n ) } \geq ( w - 1 ) \kappa$ . Define

$$
B _ { 1 } \ { \stackrel { \mathrm { d e f } } { = } } \ { \frac { V _ { 0 } + 1 } { 2 ( w - 1 ) \kappa } } ,
$$

where the denominator is positive as $w \geq 2$ . Therefore, $\ell _ { 1 } ^ { ( n ) } \leq B _ { 1 }$ by (22). Induction step: Now suppose $\ell _ { j } ^ { ( n ) } \leq B _ { j }$ for all $j < t$ and all n, and define

$$
B _ { t }  { \stackrel { \mathrm { d e f } } { = } } \frac { ( V _ { 0 } + 1 ) B ^ { \alpha } } { 2 ( w - 1 ) \kappa } ,
$$

where

$$
B \stackrel { \mathrm { d e f } } { = } \operatorname* { m a x } \{ 1 , B _ { 1 } , \ldots , B _ { t - 1 } \} .
$$

After step $t ,$ every path other than $P _ { t } ^ { ( n ) }$ has frontier at most B: its frontier is either 1 or a length from an earlier step. Since $\phi$ is decreasing, each of these $w - 1$ paths contributes at least $\phi ( B ) = \kappa B ^ { - \alpha }$ and so $\Phi _ { t + 1 } ^ { ( n ) } \ge ( w - 1 ) \kappa B ^ { - \alpha } > 0$ . Hence, $\ell _ { t } ^ { ( n ) } \leq B _ { t }$ by (22), thereby completing the induction.

Part 2: extract a limiting sequence whose contribution is at most $V _ { 0 } .$

Among the algorithms $X ^ { ( n ) }$ , some path label at step 1 occurs infinitely often. Select a subsequence on which this label is constant, and call it $P _ { 1 }$ . Since the lengths $\ell _ { 1 } ^ { ( n ) }$ lie in $[ 1 , B _ { 1 } ]$ , the Bolzano– Weierstrass theorem gives a further subsequence on which they converge to some $\ell _ { 1 }$

Repeat these two selections for step 2, then step 3, and so on, each time working within the preceding subsequence. For each step $t ,$ define $P _ { t }$ to be the constant path label and $\ell _ { t }$ to be the limit of the step lengths. Further selections preserve the labels and limits established earlier. This defines the following algorithm:

$$
X \ { \stackrel { \mathrm { d e f } } { = } } \ ( ( P _ { t } , \ell _ { t } ) ) _ { t \geq 1 } .
$$

We will verify in Part 3 that every path is swept without bound and every step strictly advances its path’s frontier.

Fix T and consider the subsequence obtained after the selections for step $T .$ Along this subsequence, the first T path labels agree with those of $X$ , and the corresponding lengths converge. The frontiers therefore also converge, giving $\ell _ { t } \geq f _ { t } ( P _ { t } )$ for $t \leq T$ . By continuity of $\phi ,$

$$
\ell _ { t } ^ { ( n ) } \Phi _ { t + 1 } ^ { ( n ) } \xrightarrow { n \to \infty } \ell _ { t } \Phi _ { t + 1 } \qquad ( t \leq T ) .\tag{23}
$$

Thus, taking limits along this subsequence,

$$
{ \begin{array} { r l } { { 2 } \displaystyle \sum _ { t \leq T } \ell _ { t } \Phi _ { t + 1 } = \displaystyle \operatorname* { l i m } _ { n \to \infty } 2 \sum _ { t \leq T } \ell _ { t } ^ { ( n ) } \Phi _ { t + 1 } ^ { ( n ) } } & { { \mathrm { ( s i n c e ~ } } T { \mathrm { ~ i s ~ f i n i t e ) } } } \\ { \leq \displaystyle \operatorname* { l i m } _ { n \to \infty } E _ { F _ { 0 } } ( X ^ { ( n ) } ) } \\ { = V _ { 0 } } \end{array} }
$$

where the inequality holds because the omitted terms are nonnegative. Since this holds for every $T ,$ taking $T \to \infty$ gives

$$
2 \sum _ { t \geq 1 } \ell _ { t } \Phi _ { t + 1 } \leq V _ { 0 } .
$$

Part 3: ensure unbounded search and conclude optimality.

First, every path is swept without bound in X (so that the resulting algorithm always finds the goal). Otherwise, some path would have frontier at most M at every step, giving $\Phi _ { t + 1 } \geq \kappa M ^ { - \alpha }$ Since $\ell _ { t } \geq 1$ ,

$$
2 \sum _ { t \geq 1 } \ell _ { t } \Phi _ { t + 1 } \geq 2 \kappa M ^ { - \alpha } \sum _ { t \geq 1 } 1 = + \infty ,
$$

contrary to Part 2.

Note that no step would have $\ell _ { t } = f _ { t } ( P _ { t } )$ : such a step changes no frontier but has a positive contribution and can be deleted with a reduced total contribution, contradicting the definition of $V _ { 0 }$

By the definition of $V _ { 0 }$ and the contribution bound from Part 2,

$$
V _ { 0 } \leq E _ { F _ { 0 } } ( X ) \leq V _ { 0 } .
$$

Thus $E _ { F _ { 0 } } ( X ) = V _ { 0 }$ , as required.

## B Terminology and notation

<table><tr><td>Notation/term</td><td>Meaning</td></tr><tr><td> $w$ </td><td>Number of paths.</td></tr><tr><td> $h$ </td><td>Distance of the goal from the origin.</td></tr><tr><td> $\epsilon > 0$ </td><td>Parameter of the input distribution.</td></tr><tr><td> $f _ { \epsilon , w }$ </td><td>Input distribution: conditional distance density ex−(1+€), for  $x \geq 1$  , and uniform choice of path.</td></tr><tr><td> $H$ </td><td>Per-path measure, with  $\begin{array} { r } { d H ( h ) = \frac { \epsilon } { w } h ^ { - ( 1 + \epsilon ) } d h } \end{array}$  for  $h \geq 1$ </td></tr><tr><td> $\alpha = 1 + \epsilon$ </td><td>Exponent in the remaining-weight function.</td></tr><tr><td> $\begin{array} { r } { \kappa = \frac { \epsilon } { w ( 1 + \epsilon ) } } \end{array}$ </td><td>Scaling constant in  $\phi ( x ) = \kappa x ^ { - \alpha }$ </td></tr><tr><td> $\beta = ( \dot { 1 } + \dot { \alpha } ) ^ { 1 / \alpha }$ </td><td>Advance-factor threshold in Proposition  $2 ; \beta ^ { \alpha } = 1 + \alpha .$ </td></tr><tr><td> $\mathrm { S t a t i o n a r y }$ </td><td>Algorithm is stationary if  $\Phi _ { t + 1 } = - \phi ^ { \prime } ( \boldsymbol { \ell } _ { t } ) W _ { t } \ \forall t .$ </td></tr><tr><td> $t ^ { + }$ </td><td>Next step after t that sweeps the same path  $P _ { t } .$ </td></tr><tr><td> $P _ { t }$ </td><td>Path swept at step t.</td></tr><tr><td> $\ell _ { t }$ </td><td>Sweep length at step  $t ,$  measured from the origin.</td></tr><tr><td> $f _ { t } ( P )$ </td><td>Frontier of path P before step t.</td></tr><tr><td> $F _ { t }$ </td><td>State before step t: the collection of all path frontiers.</td></tr><tr><td> $F _ { 0 }$ </td><td>Initial state, with every frontier set to 1.</td></tr><tr><td> $\phi ( x )$ </td><td>Remaining weight on a path with frontier  $x .$ </td></tr><tr><td> $\Phi ( F )$ </td><td>Total remaining weight in state  $F .$ </td></tr><tr><td> $\Phi _ { t } = \Phi ( F _ { t } )$ </td><td>Total remaining weight right before step  $t .$ </td></tr><tr><td> $\widehat { \Phi } _ { t + 1 }$ </td><td>Total remaining weight associated with all paths other than  $P _ { t }$  after step t. 8</td></tr><tr><td> $2 \ell _ { t } \Phi _ { t + 1 }$ </td><td>Contribution of step t to the expected competitive ratio.</td></tr><tr><td> $E _ { F } ( X )$ </td><td>Contribution of algorithm  $X ,$  started from state F:  $\begin{array} { r } { E _ { F } ( X ) = 2 \sum _ { t } \ell _ { t } \Phi _ { t + 1 } } \end{array}$ </td></tr><tr><td> $W _ { t }$ </td><td>Total (one-way) sweep length from step t up to, but excluding, step  $t ^ { + }$   $\begin{array} { r } { W _ { t } = \sum _ { t \leq j < t ^ { + } } \ell _ { j } . } \end{array}$ </td></tr><tr><td> $\widehat { W } _ { t }$ </td><td>Total (one-way) sweep length strictly between steps t and  $t ^ { + } { : }$   $\begin{array} { r } { \widehat { W } _ { t } = \sum _ { t < j < t ^ { + } } \ell _ { j } } \end{array}$ </td></tr></table>