# Last-Iterate Convergence of Clipped SGD with Linear Decay

Wei Jiang<sup>1,2</sup> Haomin Bai<sup>2</sup> Zechao Li<sup>1</sup>

<sup>1</sup>School of Computer Science and Engineering, Nanjing University of Science and Technology <sup>2</sup>National Key Laboratory for Novel Software Technology, Nanjing University {jiangw, baihm}@lamda.nju.edu.cn, zechao.li@njust.edu.cn

## Abstract

In this paper, we study the last-iterate convergence of clipped stochastic gradient descent under heavy-tailed noise. We assume that the stochastic gradients are unbiased and that the noise has a bounded central �-th moment, where $1 < p \leq 2$ . For smooth convex objectives, a linearly decreasing learning rate gives a high-probability error bound of order ${ \cal O } ( L R ^ { 2 } \tilde { \log } ( 1 / \delta ) / T + \sigma R ( \bar { \log } ( 1 / \delta ) / \bar { T } ) ^ { ( p - 1 ) / p } )$ We further establish last-iterate guarantees for nonsmooth composite objectives and smooth composite objectives with proximal updates. In addition, when a causal adversary replaces at most � gradients, the same clipped update achieves a bound with log(1/�) replaced by $\log ( 1 / \delta ) + q$ . The corresponding lower bounds for the clean and corrupted oracle models are also provided.

## 1 Introduction

In this paper, we study the problem

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } F ( x ) : = f ( x ) + r ( x ) .\tag{1}
$$

When $r = 0$ , the problem reduces to the smooth unconstrained problem. For such a problem, gradient clipping allows stochastic optimization methods to handle noise whose variance may be infinite. For convex objectives, high-probability guarantees with the statistical rate $T ^ { - ( p - 1 ) / p }$ are available for averaged outputs under a bounded central �-th moment (Nguyen et al., 2023; Liu, 2026). Obtaining this rate for the last iterate of clipped SGD requires a diferent analysis, since a bound on the average objective value does not control the objective value after the final update.

This paper studies last-iterate convergence of clipped SGD algorithm. We first establish a last-iterate guarantee for smooth convex objectives. Then, we investigate proximal updates for nonsmooth and smooth composite objectives, seperately. Finally, we analyze the setting in which a number of samples can be replaced by a causal adversary. Table 1 summarizes these bounds.

## 2 Theoretical guarantee

We first introduce the assumptions used.

Assumption 1. Function $f : \mathbb { R } ^ { d }  \mathbb { R }$ is convex, and $r : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ is proper, lower semicontinuous, and convex. There exists $x ^ { \star } \in$ arg min � and we denote $F ^ { \star } = F ( x ^ { \star } )$ . The initial point satisfies $x _ { 1 } \in$ dom � and $\| x _ { 1 } - x ^ { \star } \| \leq R ,$ ,for a known $R > 0$

Assumption 2. For some $1 < p \leq 2$ and $\sigma \geq 0 ,$ , the vector $g _ { t }$ satisfies, almost surely,

$$
h _ { t } : = \mathbb { E } _ { t } [ g _ { t } ] \in \partial f ( x _ { t } ) , \quad \quad \mathbb { E } _ { t } [ \Vert g _ { t } - h _ { t } \Vert ^ { p } ] \leq \sigma ^ { p } .\tag{2}
$$

When � is diferentiable, $h _ { t } = \nabla f ( x _ { t } )$

Assumption 3. Function � has an �-Lipschitz gradient:

$$
\begin{array} { r } { \| \nabla f ( x ) - \nabla f ( y ) \| \leq L \| x - y \| , \quad L > 0 , \quad x , y \in \mathbb { R } ^ { d } . } \end{array}
$$

Table 1: High-probability guarantees for the final update. Here $a = ( p - 1 ) / p , \ell \asymp \log ( 1 / \delta )$
<table><tr><td>Setting</td><td>Objective error</td><td>Oracle calls</td></tr><tr><td>Smooth, Theorem 1</td><td> $L R ^ { 2 } \ell / T + \sigma R ( \ell / T ) ^ { a }$ </td><td>T</td></tr><tr><td>Nonsmooth composite, Theorem 2</td><td> $G R / \sqrt { T } + G R \ell / T + \sigma R ( \ell / T ) ^ { a }$ </td><td>T</td></tr><tr><td>Smooth composite, Theorem 3</td><td> $L R ^ { 2 } \ell / T + \sigma R ( \ell / T ) ^ { a }$ </td><td> $T + O ( \ell )$ </td></tr><tr><td>q replacements, Theorem 4</td><td> $L R ^ { 2 } ( \ell + q ) / T + \sigma R ( ( \ell + q ) / T ) ^ { a }$ </td><td>T</td></tr></table>

Algorithm 1 Clipped SGD with linear decay   
Require: Initial point $x _ { 1 } .$ iteration number �, clipping level $\lambda > 0 ,$ base learning rate $\eta > 0$   
1: for $t = 1 , \dots , T$ do   
2: Query the stochastic gradient $g _ { t }$ at $x _ { t }$   
3: Set $\eta _ { t } = \eta ( T - t + 1 ) / T$ and ${ \widehat { g } } _ { t } = \operatorname { c l i p } ( g _ { t } , \lambda )$   
4: Update $x _ { t + 1 } = x _ { t } - \eta _ { t } \widehat { g } _ { t }$   
5: end for   
6: return $x _ { T + 1 }$

For $\lambda > 0 ,$ , define

$$
\mathrm { c l i p } ( g , \lambda ) = g \operatorname * { m i n } \{ 1 , \lambda / \| g \| \} , \qquad \mathrm { c l i p } ( 0 , \lambda ) = 0 .
$$

Clipped SGD performs one oracle call per iteration, as described in Algorithm 1. Its output is $x _ { T + 1 }$ . The learning rate decreases from � to $\eta / T$ . We use a fixed clipping level � throughout the run. The two terms in the � below have separate roles: 4�� ensures that the true gradient remains below the clipping level, while $\sigma ( T / \ell ) ^ { 1 / p }$ controls the accumulated clipping bias and noise.

Theorem 1. Under Assumptions $_ { I - 3 }$ with $r = 0 ,$ fix an integer $T \geq 1$ and $\delta \in ( 0 , 1 )$ . Run Algorithm 1 with

$$
\ell = \log \frac { 8 } { \delta } , \qquad \lambda = \operatorname* { m a x } \left\{ 4 L R , \sigma ( T / \ell ) ^ { 1 / p } \right\} , \qquad \eta = \frac { R } { 1 2 8 \lambda \ell } .\tag{3}
$$

With probability at least $1 - \delta ,$

$$
f ( x _ { T + 1 } ) - f ( x ^ { \star } ) \leq \frac { 2 5 6 R \lambda \ell } { T }\tag{4}
$$

$$
\leq 1 0 2 4 { \frac { L R ^ { 2 } \ell } { T } } + 2 5 6 \sigma R \left( { \frac { \ell } { T } } \right) ^ { ( p - 1 ) / p } .\tag{5}
$$

For $p = 2 ,$ , the dominant term in (5) is $O ( \sigma R \sqrt { \log ( 1 / \delta ) / T } )$ . The result needs neither a bounded feasible set nor a uniform bound on $\| \nabla f ( x ) \|$ . Proposition 1 gives the corresponding lower bound.

## 3 Clipped SGD with proximal updates

## 3.1 Nonsmooth composite objectives

We now replace smoothness by the following condition.

Assumption 4 (Lipschitz continuity). For a known $G \geq 0 , | f ( x ) - f ( y ) | \leq G \| x - y \| f o r a l l x , y \in \mathbb { R } ^ { d } .$

Every subgradient of � then has norm at most �. The regularizer � need not be Lipschitz. We assume exact access to

$$
\mathrm { p r o x } _ { \alpha r } ( y ) = \arg \operatorname* { m i n } _ { u \in \mathbb { R } ^ { d } } \left\{ r ( u ) + \frac { \| u - y \| ^ { 2 } } { 2 \alpha } \right\} , \qquad \alpha > 0 .
$$

We use the proximal clipped update

$$
x _ { t + 1 } = { \mathrm { p r o x } } _ { \eta _ { t } r } { \big ( } x _ { t } - \eta _ { t } \operatorname { c l i p } ( g _ { t } , \lambda ) { \big ) } , \qquad \eta _ { t } = \eta { \frac { T - t + 1 } { T } } .\tag{6}
$$

Theorem 2. Under Assumptions 1, 2, and 4, suppose $G + \sigma > 0 .$ . For an integer $T \geq 1$ and $\delta \in ( 0 , 1 )$ , set

$$
\ell = \log \frac { 4 } { \delta } , \quad \lambda = \operatorname* { m a x } \{ 2 G , \sigma ( T / \ell ) ^ { 1 / p } \} , \quad H = G \sqrt { T } + \lambda \ell , \quad \eta = \frac { R } { 1 2 8 H } .\tag{7}
$$

After � updates of (6), with probability at least $1 - \delta ,$

$$
\begin{array} { l } { \displaystyle { F ( x _ { T + 1 } ) - F ^ { \star } \le \frac { 2 5 6 R H } { T } } } \\ { \displaystyle { \le 2 5 6 \frac { G R } { \sqrt { T } } + 5 1 2 \frac { G R \ell } { T } + 2 5 6 \sigma R ( \ell / T ) ^ { ( p - 1 ) / p } . } } \end{array}\tag{8}
$$

(9)

$H T \geq \ell ^ { 2 }$ , the right-hand side is at most 768� $R / \sqrt { T } + 2 5 6 \sigma R ( \ell / T ) ^ { ( p - 1 ) / p }$

## 3.2 Smooth composite objectives

At �<sub>1</sub>, query $m = \lceil 3 2 \log ( 2 / \delta ) \rceil$ pilot gradients $g _ { 1 } ^ { ( 0 ) } , \ldots , g _ { m } ^ { ( 0 ) }$ , and let

$$
c \in \arg \operatorname* { m i n } _ { z \in \mathbb { R } ^ { d } } \sum _ { j = 1 } ^ { m } \bigl \| z - g _ { j } ^ { ( 0 ) } \bigr \| .\tag{10}
$$

We use a measurable choice of this geometric median. The pilot gradients satisfy the conditional oracle assumption at $x _ { 1 }$ , and the main-stage filtration includes all pilot data. The updates are

$$
{ \widehat { g } } _ { t } = c + \mathrm { c l i p } ( g _ { t } - c , \lambda ) , \qquad x _ { t + 1 } = \operatorname { p r o x } _ { \eta _ { t } r } ( x _ { t } - \eta _ { t } { \widehat { g } } _ { t } ) .\tag{11}
$$

Theorem 3. Under Assumptions 1–3,fix $\delta \in ( 0 , 1 )$ and construct � by (10). For an integer $T \geq 1$ , choose

$$
\ell = \log \frac { 1 6 } { \delta } , \quad \lambda = \operatorname* { m a x } \{ 6 L R + 4 8 \sigma , \sigma ( T / \ell ) ^ { 1 / p } \} , \quad \eta _ { t } = \frac { R } { 1 2 8 \lambda \ell } \frac { T - t + 1 } { T } .\tag{12}
$$

After � updates of (11), with probability at least $1 - \delta _ { ; }$

$$
F ( x _ { T + 1 } ) - F ^ { \star } \leq \frac { 2 5 6 R \lambda \ell } { T } , \qquad \operatorname* { m a x } _ { 1 \leq t \leq T + 1 } \| x _ { t } - x ^ { \star } \| \leq 2 R .\tag{13}
$$

When $T \geq \ell ,$

$$
F ( x _ { T + 1 } ) - F ^ { \star } \leq 1 5 3 6 { \frac { L R ^ { 2 } \ell } { T } } + 1 2 5 4 4 \sigma R ( \ell / T ) ^ { ( p - 1 ) / p } .\tag{14}
$$

The total number ofstochastic-gradient calls is $N = T + m$

## 4 Adversarially corrupted gradients

Return to the smooth setting with $r = 0$ . After $x _ { t }$ is selected, the oracle generates a latent clean gradient $g _ { t }$ satisfying Assumption 2. A causal adversary may observe $g _ { t }$ and replace it by any vector $\widetilde { g } _ { t }$ . Let

$$
C _ { t } = \mathbf { 1 } _ { \{ \widetilde { g } _ { t } \neq g _ { t } \} } , \qquad \sum _ { t = 1 } ^ { T } C _ { t } \leq q \quad \mathrm { a l m o s t ~ s u r e l y } ,\tag{15}
$$

where $q \in \{ 0 , . . . , T \}$ is a known upper bound. The adversary can depend on the past and the current clean sample, but not on future clean randomness. The filtration in Assumption 2 contains all past latent samples and adversarial decisions. The algorithm observes only $\widetilde { g } _ { t }$

Theorem 4. Under Assumptions $_ { I - 3 }$ with $r = 0$ and the corruption model (15), fix an integer $T \geq 1$ and $\delta \in ( 0 , 1 )$ . Set

$$
\ell = \log \frac { 8 } { \delta } , \quad \Gamma = \ell + q , \quad \lambda = \operatorname* { m a x } \{ 4 L R , \sigma ( T / \Gamma ) ^ { 1 / p } \} , \quad \eta = \frac { R } { 1 2 8 \lambda \Gamma } .\tag{16}
$$

Run Algorithm 1 using $\widetilde { g } _ { t }$ in place $o f g _ { t }$ . With probability at least $1 - \delta ,$

$$
f ( x _ { T + 1 } ) - f ( x ^ { \star } ) \leq { \frac { 2 5 6 R \lambda \Gamma } { T } }\tag{17}
$$

$$
\leq 1 0 2 4 L R ^ { 2 } \frac { \ell + q } { T } + 2 5 6 \sigma R \left( \frac { \ell + q } { T } \right) ^ { ( p - 1 ) / p } .\tag{18}
$$

On the same event, $\| x _ { t } - x ^ { \star } \| \leq 2 R f o r \ 1 \leq t \leq T + 1$

If $q \leq \varepsilon T$ , the two terms in (18) have orders $L R ^ { 2 } ( \ell / T + \varepsilon )$ and $\sigma R ( \ell / T + \varepsilon ) ^ { ( p - 1 ) / p }$ . The model allows the adversary to select samples after inspecting them.

Corollary 1. In Theorem 4, replace the uniform moment condition by $\mathbb { E } _ { t } \| g _ { t } - \nabla f ( x _ { t } ) \| ^ { p } \leq \nu _ { t } ,$ , where $\nu _ { t } \geq 0 \mathrm { ~ } i s \mathcal { F } _ { t - 1 }$ -measurable and $\textstyle \sum _ { t = 1 } ^ { T } \nu _ { t } \leq S$ almost surely for a known deterministic $S ~ \geq 0$ . With $\lambda = \operatorname* { m a x } \{ 4 L R , ( S / \Gamma ) ^ { 1 / p } \}$ and the same learning-rateformula,

$$
f ( x _ { T + 1 } ) - f ( x ^ { \star } ) \leq 1 0 2 4 L R ^ { 2 } \frac { \Gamma } { T } + 2 5 6 R \frac { S ^ { 1 / p } \Gamma ^ { ( p - 1 ) / p } } { T }
$$

with probability at least $1 - \delta .$

## 5 Statistical lower bounds

We give three two-instance constructions to clarify the statistical terms. These are oracle lower bounds: an algorithm may make adaptive stochastic-gradient queries and randomize its output, but it does not have an exact function-value oracle. The proximal oracle in the nonsmooth construction is the same for both instances. The constructions use rare informative samples, as in standard lower-bound arguments.

Proposition 1 (Clean smooth oracle). Fix � $, R , \sigma > 0 , \ 1 < p \ \leq \ 2 , \ 0 \ < \ \delta \ \leq \ 1 / 8 ,$ and an integer $T \geq b : = \log ( 1 / ( 4 \delta ) )$ . For every algorithm making at most � stochastic-gradient queries, there exists a one-dimensional instance satisfying the assumptions of Theorem 1, with $x _ { 1 } = 0 ,$ , such that its output $\widehat { x }$ satisfies

$$
\mathbb { P } \left\{ f ( \widehat { x } ) - f ( x ^ { \star } ) \geq \operatorname* { m i n } \left\{ \frac { L R ^ { 2 } } { 2 } , \frac { \sigma R } { 4 } ( b / T ) ^ { ( p - 1 ) / p } \right\} \right\} \geq 2 \delta .
$$

Proposition 2 (Clean nonsmooth composite oracle). $F i x G , R , \sigma > 0 , 1 < p \leq 2 , 0 < \delta \leq 1 / 8 ,$ , and an integer $T \geq b : = \log ( 1 / ( 4 \delta ) )$ . For every algorithm making at most � stochastic-subgradient queries, even with exact proximal access to �, there exists a one-dimensional instance satisfying the assumptions of Theorem 2, with $x _ { 1 } = 0 _ { ; }$ , such that

$$
\mathbb { P } \left\{ F ( \widehat { x } ) - F ^ { \star } \geq \operatorname* { m i n } \left\{ G R , \frac { \sigma R } { 2 } ( b / T ) ^ { ( p - 1 ) / p } \right\} \right\} \geq 2 \delta .
$$

Proposition 3 (Corrupted smooth oracle). Fix $L , R , \sigma > 0 , 1 < p \leq 2$ , and integers $1 \leq q \leq T$ . For every algorithm making at most � stochastic-gradient queries, there is a one-dimensional clean instance satisfying the assumptions of Theorem $^ { l , }$ with $x _ { 1 } = 0 ,$ , and a causal adversary satisfying (15) such that

$$
\mathbb { P } \left\{ f ( \widehat { x } ) - f ( x ^ { \star } ) \geq \frac { R } { 2 } \operatorname* { m i n } \left\{ L R , \frac { \sigma } { 2 } \left( \frac { q } { 2 T } \right) ^ { ( p - 1 ) / p } \right\} \right\} \geq \frac { 1 } { 4 } .
$$

## A Useful lemmas

Throughout the appendix, denote $D ( u , x ) = \| u - x \| ^ { 2 } / 2$

## A.1 Smoothness and proximal updates

Lemma 1. $_ { I f f }$ is convex and �-smooth, then,for all $x , y \in \mathbb { R } ^ { d }$

$$
f ( x ) \geq f ( y ) + \langle \nabla f ( y ) , x - y \rangle + { \frac { \| \nabla f ( x ) - \nabla f ( y ) \| ^ { 2 } } { 2 L } } ,\tag{19}
$$

$$
\langle \nabla f ( x ) - \nabla f ( y ) , x - y \rangle \geq { \frac { \| \nabla f ( x ) - \nabla f ( y ) \| ^ { 2 } } { L } } .\tag{20}
$$

Proof. Fix � and let $\psi ( z ) = f ( z ) - \langle \nabla f ( y ) , z \rangle$ . This function is convex and �-smooth, with $\nabla \psi ( y ) = 0$ Therefore

$$
\psi ( z ) \geq \psi ( y ) + \langle \nabla \psi ( y ) , z - y \rangle = \psi ( y ) , \qquad z \in \mathbb { R } ^ { d } .
$$

Applying this inequality and then smoothness at � gives

$$
\begin{array} { l } { \displaystyle \psi ( \boldsymbol { y } ) \leq \psi \left( x - \frac { 1 } { L } \nabla \psi ( \boldsymbol { x } ) \right) } \\ { \displaystyle \qquad \leq \psi ( \boldsymbol { x } ) + \left. \nabla \psi ( \boldsymbol { x } ) , - \frac { 1 } { L } \nabla \psi ( \boldsymbol { x } ) \right. + \frac { L } { 2 } \left\| \frac { 1 } { L } \nabla \psi ( \boldsymbol { x } ) \right\| ^ { 2 } } \\ { \displaystyle \qquad = \psi ( \boldsymbol { x } ) - \frac { 1 } { 2 L } \| \nabla \psi ( \boldsymbol { x } ) \| ^ { 2 } . } \end{array}
$$

Substituting the definition of $\psi$ yields

$$
\begin{array} { c } { \displaystyle f ( y ) - \langle \nabla f ( y ) , y \rangle \leq f ( x ) - \langle \nabla f ( y ) , x \rangle - \displaystyle \frac { 1 } { 2 L } \| \nabla f ( x ) - \nabla f ( y ) \| ^ { 2 } , } \\ { f ( x ) - f ( y ) \geq \langle \nabla f ( y ) , x - y \rangle + \displaystyle \frac { 1 } { 2 L } \| \nabla f ( x ) - \nabla f ( y ) \| ^ { 2 } . } \end{array}
$$

This proves (19). Interchanging � and � and adding the resulting inequalities gives

$$
\begin{array} { l } { \displaystyle 0 \geq \langle \nabla f ( y ) , x - y \rangle + \langle \nabla f ( x ) , y - x \rangle + \cfrac { 1 } { L } \| \nabla f ( x ) - \nabla f ( y ) \| ^ { 2 } } \\ { \displaystyle } \\ { \displaystyle = - \langle \nabla f ( x ) - \nabla f ( y ) , x - y \rangle + \cfrac { 1 } { L } \| \nabla f ( x ) - \nabla f ( y ) \| ^ { 2 } , } \end{array}
$$

which is (20).

□

Lemma 2. Let $F = f + r$ satisfy Assumption 1. For $x \in$ dom �, $h \in \partial f ( x )$ , an arbitrary error �, and $\alpha > 0$ , define $x ^ { + } = \operatorname { p r o x } _ { \alpha r } \left( x - \alpha ( h + e ) \right)$ . If � is �-smooth and $\alpha \leq 1 / ( 2 L )$ , then,for every $u \in$ dom �,

$$
\begin{array} { r } { \alpha ( F ( x ^ { + } ) - F ( u ) ) \le D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle + \alpha ^ { 2 } \| e \| ^ { 2 } . } \end{array}\tag{21}
$$

If � is �-Lipschitz, then for every $\alpha > 0$

$$
\begin{array} { r } { \alpha ( F ( x ^ { + } ) - F ( u ) ) \le D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle + \alpha ^ { 2 } ( \| e \| ^ { 2 } + 4 G ^ { 2 } ) . } \end{array}\tag{22}
$$

Proof. The optimality condition for the proximal update gives

$$
\frac { x - x ^ { + } } { \alpha } - h - e \in \partial r ( x ^ { + } ) .
$$

By the subgradient inequality,

$$
\begin{array} { c } { { r ( u ) \geq r ( x ^ { + } ) + \displaystyle \left. \displaystyle \frac { x - x ^ { + } } { \alpha } - h - e , u - x ^ { + } \right. , } } \\ { { \displaystyle \alpha \big ( r ( x ^ { + } ) - r ( u ) + \langle h + e , x ^ { + } - u \rangle \big ) \leq \langle x - x ^ { + } , x ^ { + } - u \rangle . } } \end{array}
$$

Expanding the square gives

$$
\| x - u \| ^ { 2 } = \| x - x ^ { + } \| ^ { 2 } + \| x ^ { + } - u \| ^ { 2 } + 2 \langle x - x ^ { + } , x ^ { + } - u \rangle .
$$

Consequently,

$$
\alpha \big ( r ( x ^ { + } ) - r ( u ) + \langle h + e , x ^ { + } - u \rangle \big ) \leq D ( u , x ) - D ( u , x ^ { + } ) - \frac { 1 } { 2 } \| x ^ { + } - x \| ^ { 2 } .\tag{23}
$$

For smooth $f ,$ we have $h = \nabla f ( x )$ . Smoothness and convexity, respectively, give

$$
\begin{array} { l } { \displaystyle f ( x ^ { + } ) - f ( x ) \leq \langle h , x ^ { + } - x \rangle + \frac { L } { 2 } \| x ^ { + } - x \| ^ { 2 } , } \\ { \displaystyle f ( x ) - f ( u ) \leq \langle h , x - u \rangle . } \end{array}
$$

Adding these two inequalities, we obtain

$$
\begin{array} { l } { f ( x ^ { + } ) - f ( u ) = f ( x ^ { + } ) - f ( x ) + f ( x ) - f ( u ) } \\ { \displaystyle \leq \langle h , x ^ { + } - x \rangle + \langle h , x - u \rangle + \displaystyle \frac { L } { 2 } \| x ^ { + } - x \| ^ { 2 } } \\ { \displaystyle = \langle h , x ^ { + } - u \rangle + \displaystyle \frac { L } { 2 } \| x ^ { + } - x \| ^ { 2 } . } \end{array}
$$

Combining this with (23) yields

$$
\begin{array} { l } { \displaystyle \alpha ( F ( x ^ { + } ) - F ( u ) ) \leq D ( u , x ) - D ( u , x ^ { + } ) - \frac { 1 - L \alpha } { 2 } \| x ^ { + } - x \| ^ { 2 } + \alpha \langle e , u - x ^ { + } \rangle } \\ { \displaystyle = D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle - \frac { 1 - L \alpha } { 2 } \| x ^ { + } - x \| ^ { 2 } - \alpha \langle e , x ^ { + } - x \rangle } \\ { \displaystyle \leq D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle - \frac { 1 } { 4 } \| x ^ { + } - x \| ^ { 2 } + \alpha \| e \| \| x ^ { + } - x \| } \\ { \displaystyle = D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle + \alpha ^ { 2 } \| e \| ^ { 2 } - \left( \frac { \| x ^ { + } - x \| } { 2 } - \alpha \| e \| \right) ^ { 2 } } \\ { \displaystyle \leq D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle + \alpha ^ { 2 } \| e \| ^ { 2 } . } \end{array}
$$

The second inequality uses $L \alpha \leq 1 / 2$ and Cauchy–Schwarz.

For �-Lipschitz $f ,$ first note that $\| h \| \leq G$ . If $h \neq 0$ , then for any $a > 0$

$$
a \| h \| \leq f \left( x + a { \frac { h } { \| h \| } } \right) - f ( x ) \leq G a .
$$

Hence, we can ensure that

$$
\begin{array} { r l } & { f ( x ^ { + } ) - f ( u ) = f ( x ^ { + } ) - f ( x ) + f ( x ) - f ( u ) } \\ & { \qquad \leq G \| x ^ { + } - x \| + \langle h , x - u \rangle } \\ & { \qquad = \langle h , x ^ { + } - u \rangle + \langle h , x - x ^ { + } \rangle + G \| x ^ { + } - x \| } \\ & { \qquad \leq \langle h , x ^ { + } - u \rangle + 2 G \| x ^ { + } - x \| . } \end{array}
$$

Using (23) once more gives

$$
\begin{array} { l } { \displaystyle \alpha ( F ( x ^ { + } ) - F ( u ) ) \leq D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x ^ { + } \rangle + 2 \alpha G \| x ^ { + } - x \| - \displaystyle \frac { 1 } { 2 } \| x ^ { + } - x \| ^ { 2 } } \\ { \displaystyle \qquad \leq D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle + \alpha ( \| e \| + 2 G ) \| x ^ { + } - x \| - \displaystyle \frac { 1 } { 2 } \| x ^ { + } - x \| ^ { 2 } } \\ { \displaystyle \qquad = D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle + \displaystyle \frac { \alpha ^ { 2 } } { 2 } ( \| e \| + 2 G ) ^ { 2 } } \\ { \displaystyle \qquad - \displaystyle \frac { 1 } { 2 } \big ( \| x ^ { + } - x \| - \alpha ( \| e \| + 2 G ) \big ) ^ { 2 } } \\ { \displaystyle \qquad \leq D ( u , x ) - D ( u , x ^ { + } ) + \alpha \langle e , u - x \rangle + \alpha ^ { 2 } ( \| e \| ^ { 2 } + 4 G ^ { 2 } ) , } \end{array}
$$

where $( \| e \| + 2 G ) ^ { 2 } \leq 2 \| e \| ^ { 2 } + 8 G ^ { 2 }$ proves the last step.

Lemma 3. Under Assumptions 1 and 3, let $x ^ { + } = \operatorname { p r o x } _ { \alpha r } ( x - \alpha ( \nabla f ( x ) + e ) )$ and set $0 < \alpha \leq 1 / L$ . Then

$$
\begin{array} { r } { \| { \boldsymbol x } ^ { + } - { \boldsymbol x } ^ { \star } \| ^ { 2 } \leq \| { \boldsymbol x } - { \boldsymbol x } ^ { \star } \| ^ { 2 } - 2 \alpha \langle e , { \boldsymbol x } - { \boldsymbol x } ^ { \star } \rangle + 2 \alpha ^ { 2 } \| e \| ^ { 2 } . } \end{array}\tag{24}
$$

Proof. For $u = \mathrm { p r o x } _ { \alpha r } ( a )$ and $\nu = \operatorname { p r o x } _ { \alpha r } ( b )$ , the proximal optimality conditions and the subgradient inequalities imply

$$
\begin{array} { c c } { \displaystyle { \frac { a - u } { \alpha } \in \partial r ( u ) , \qquad } } & { \displaystyle { \frac { b - \nu } { \alpha } \in \partial r ( \nu ) , } } \\ { \displaystyle r ( \nu ) - r ( u ) \geq \frac { 1 } { \alpha } \langle a - u , \nu - u \rangle , \qquad } & { \displaystyle r ( u ) - r ( \nu ) \geq \frac { 1 } { \alpha } \langle b - \nu , u - \nu \rangle . } \end{array}
$$

Adding the last two inequalities and rearranging gives

$$
\begin{array} { c } { 0 \leq \langle a - b - ( u - \nu ) , u - \nu \rangle , } \\ { \| u - \nu \| ^ { 2 } \leq \langle a - b , u - \nu \rangle \leq \| a - b \| \| u - \nu \| , } \\ { \| \mathrm { p r o x } _ { \alpha r } ( a ) - \mathrm { p r o x } _ { \alpha r } ( b ) \| \leq \| a - b \| . } \end{array}
$$

The last line also holds if $u = \nu$ . Since $f$ is diferentiable on $\mathbb { R } ^ { d }$ , optimality of $x ^ { \star }$ gives

$$
- \nabla f ( { x } ^ { \star } ) \in \partial r ( { x } ^ { \star } ) , \qquad { x } ^ { \star } = \mathrm { p r o x } _ { \alpha r } ( { x } ^ { \star } - \alpha \nabla f ( { x } ^ { \star } ) ) .
$$

Applying the preceding nonexpansiveness inequality, we obtain

$$
\begin{array} { l } { \| x ^ { \star } - x ^ { \star } \| ^ { 2 } \leq \| x - x ^ { \star } - \alpha ( \nabla f ( x ) - \nabla f ( x ^ { \star } ) + \epsilon ) \| ^ { 2 } } \\ { = \| x - x ^ { \star } \| ^ { 2 } - 2 \alpha \langle \nabla f ( x ) - \nabla f ( x ^ { \star } ) , x - x ^ { \star } \rangle - 2 \alpha \langle e , x - x ^ { \star } \rangle } \\ { \qquad + \alpha ^ { 2 } \| \nabla f ( x ) - \nabla f ( x ^ { \star } ) + \epsilon \| ^ { 2 } } \\ { \leq \| x - x ^ { \star } \| ^ { 2 } - 2 \alpha \langle e , x - x ^ { \star } \rangle - \displaystyle \frac { 2 \alpha } { L } \| \nabla f ( x ) - \nabla f ( x ^ { \star } ) \| ^ { 2 } } \\ { \qquad + 2 \alpha ^ { 2 } \| \nabla f ( x ) - \nabla f ( x ^ { \star } ) \| ^ { 2 } + 2 \alpha ^ { 2 } \| e \| ^ { 2 } } \\ { = \| x - x ^ { \star } \| ^ { 2 } - 2 \alpha \langle e , x - x ^ { \star } \rangle + 2 \alpha ^ { 2 } \| e \| ^ { 2 } + 2 \alpha \left( \alpha - \displaystyle \frac { 1 } { L } \right) \| \nabla f ( x ) - \nabla f ( x ^ { \star } ) \| ^ { 2 } } \\ { \leq \| x - x ^ { \star } \| ^ { 2 } - 2 \alpha \langle e , x - x ^ { \star } \rangle + 2 \alpha ^ { 2 } \| e \| ^ { 2 } . } \end{array}
$$

Here the second inequality uses (20) and $\| a + b \| ^ { 2 } \leq 2 \| a \| ^ { 2 } + 2 \| b \| ^ { 2 }$ , and the last uses $\alpha \leq 1 / L$ □

## A.2 A comparison inequality for the last iterate

For positive learning rates $\eta _ { 1 } , . . . , \eta _ { T }$ , define

$$
\begin{array} { l } { { S _ { t } = \displaystyle \sum _ { j = t } ^ { T } \eta _ { j } \mathrm { , } \quad S _ { T + 1 } = 0 \mathrm { , } \quad \nu _ { t } = \frac { \eta _ { T } } { S _ { t } } \left( 1 \le t \le T \right) \mathrm { , } \quad \nu _ { 0 } = \nu _ { 1 } \mathrm { , } } } \\ { { \qquad z _ { 0 } = x ^ { \star } \mathrm { , } \qquad z _ { t } = \frac { \nu _ { t - 1 } } { \nu _ { t } } z _ { t - 1 } + \left( 1 - \frac { \nu _ { t - 1 } } { \nu _ { t } } \right) x _ { t } \mathrm { , } } } \\ { { P _ { 0 } = \nu _ { 0 } D ( x ^ { \star } , x _ { 1 } ) \mathrm { , } \qquad P _ { t } = \nu _ { t } D ( z _ { t } , x _ { t + 1 } ) \quad ( 1 \le t \le T ) \mathrm { . } } } \end{array}\tag{25}
$$

Lemma 4. Let � satisfy Assumption 1, let $x _ { t } \in$ dom $r f o r \ : 1 \leq t \leq T + 1$ , and write $\Delta _ { t } = F ( x _ { t } ) - F ^ { \star }$ Suppose that, for every � ∈ dom � and $1 \leq t \leq T$

$$
\eta _ { t } ( F ( x _ { t + 1 } ) - F ( u ) ) \leq D ( u , x _ { t } ) - D ( u , x _ { t + 1 } ) + \eta _ { t } \langle e _ { t } , u - x _ { t } \rangle + \rho _ { t } ,\tag{26}
$$

where $\rho _ { t } \geq 0 .$ . Then, for every $1 \leq s \leq T _ { \mathrm { { \scriptsize ~ 2 } ~ } }$

$$
P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } + S _ { s + 1 } \sum _ { j = 2 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } \leq P _ { 0 } + \sum _ { t = 1 } ^ { s } \nu _ { t } \rho _ { t } + \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle ,\tag{27}
$$

where $A _ { t } = \eta _ { t } \nu _ { t - 1 } \big ( z _ { t - 1 } - x _ { t } \big )$ . Every term on the left-hand side is nonnegative.

Proof. By equation (25), we know that

$$
\begin{array} { c } { { z _ { t } - x _ { t } = \displaystyle \frac { \nu _ { t - 1 } } { \nu _ { t } } ( z _ { t - 1 } - x _ { t } ) , } } \\ { { \nu _ { t } D ( z _ { t } , x _ { t } ) = \displaystyle \frac { \nu _ { t - 1 } ^ { 2 } } { \nu _ { t } } D ( z _ { t - 1 } , x _ { t } ) \leq \nu _ { t - 1 } D ( z _ { t - 1 } , x _ { t } ) = P _ { t - 1 } , } } \\ { { \eta _ { t } \nu _ { t } \langle e _ { t } , z _ { t } - x _ { t } \rangle = \eta _ { t } \nu _ { t - 1 } \langle e _ { t } , z _ { t - 1 } - x _ { t } \rangle = \langle e _ { t } , A _ { t } \rangle . } } \end{array}
$$

Putting $u = z _ { t }$ in (26) and multiplying by $\nu _ { t }$ gives

$$
P _ { t } + \eta _ { t } \nu _ { t } ( F ( x _ { t + 1 } ) - F ( z _ { t } ) ) \leq P _ { t - 1 } + \nu _ { t } \rho _ { t } + \langle e _ { t } , A _ { t } \rangle .\tag{28}
$$

To expand $F ( z _ { t } )$ , first telescope the recursion for $z _ { t }$ :

$$
\nu _ { t } z _ { t } - \nu _ { t - 1 } z _ { t - 1 } = ( \nu _ { t } - \nu _ { t - 1 } ) x _ { t } ,
$$

$$
\nu _ { t } z _ { t } - \nu _ { 0 } x ^ { \star } = \sum _ { j = 1 } ^ { t } ( \nu _ { j } - \nu _ { j - 1 } ) x _ { j } ,
$$

$$
\nu _ { 0 } + \sum _ { j = 1 } ^ { t } \bigl ( \nu _ { j } - \nu _ { j - 1 } \bigr ) = \nu _ { t } .
$$

Convexity therefore gives

$$
\nu _ { t } F ( z _ { t } ) \leq \nu _ { 0 } F ^ { \star } + \sum _ { j = 1 } ^ { t } ( \nu _ { j } - \nu _ { j - 1 } ) F ( x _ { j } ) ,
$$

$$
\nu _ { t } ( F ( z _ { t } ) - F ^ { \star } ) \leq \sum _ { j = 1 } ^ { t } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } ,
$$

$$
\nu _ { t } ( F ( x _ { t + 1 } ) - F ( z _ { t } ) ) \geq \nu _ { t } \Delta _ { t + 1 } - \sum _ { j = 1 } ^ { t } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } .
$$

Substitution in (28) and summation over $t = 1 , \dots , s$ yield

$$
P _ { s } + \sum _ { t = 1 } ^ { s } \eta _ { t } \nu _ { t } \Delta _ { t + 1 } - \sum _ { t = 1 } ^ { s } \eta _ { t } \sum _ { j = 1 } ^ { t } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } \leq P _ { 0 } + \sum _ { t = 1 } ^ { s } \nu _ { t } \rho _ { t } + \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle .\tag{29}
$$

The two sums containing $\Delta _ { j }$ have the following exact expansions:

$$
\sum _ { t = 1 } ^ { s } \eta _ { t } \nu _ { t } \Delta _ { t + 1 } = \eta _ { s } \nu _ { s } \Delta _ { s + 1 } + \sum _ { j = 2 } ^ { s } \eta _ { j - 1 } \nu _ { j - 1 } \Delta _ { j } ,
$$

$$
\begin{array} { c } { { \displaystyle \sum _ { t = 1 } ^ { s } \eta _ { t } \sum _ { j = 1 } ^ { t } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } = \sum _ { j = 1 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } \sum _ { t = j } ^ { s } \eta _ { t } } } \\ { { = \displaystyle \sum _ { j = 2 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) ( S _ { j } - S _ { s + 1 } ) \Delta _ { j } . } } \end{array}
$$

The last equality uses $\nu _ { 1 } = \nu _ { 0 }$ . For $2 \leq j \leq s ,$

$$
S _ { j - 1 } - S _ { j } = \eta _ { j - 1 } ,
$$

$$
\nu _ { j } - \nu _ { j - 1 } = \eta _ { T } \left( \frac { 1 } { S _ { j } } - \frac { 1 } { S _ { j - 1 } } \right) = \frac { \eta _ { T } \eta _ { j - 1 } } { S _ { j } S _ { j - 1 } } ,
$$

$$
\begin{array} { r l } & { \eta _ { j - 1 } \nu _ { j - 1 } - { \big ( } \nu _ { j } - \nu _ { j - 1 } { \big ) } { \big ( } S _ { j } - S _ { s + 1 } { \big ) } = S _ { j } { \big ( } \nu _ { j } - \nu _ { j - 1 } { \big ) } - { \big ( } S _ { j } - S _ { s + 1 } { \big ) } { \big ( } \nu _ { j } - \nu _ { j - 1 } { \big ) } } \\ & { \qquad = S _ { s + 1 } { \big ( } \nu _ { j } - \nu _ { j - 1 } { \big ) } . } \end{array}
$$

Consequently,

$$
\sum _ { t = 1 } ^ { s } \eta _ { t } \nu _ { t } \Delta _ { t + 1 } - \sum _ { t = 1 } ^ { s } \eta _ { t } \sum _ { j = 1 } ^ { t } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } = \eta _ { s } \nu _ { s } \Delta _ { s + 1 } + S _ { s + 1 } \sum _ { j = 2 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } .
$$

This proves (27). Its left-hand side is nonnegative because $P _ { s } \ge 0 , \Delta _ { j } \ge 0 , S _ { s + 1 } \ge 0 .$ , and $\nu _ { j } \geq \nu _ { j - 1 }$ □

Lemma 5. For $\eta _ { t } = \eta ( T - t + 1 ) / T _ { \mathrm { ~ } }$ , with $\eta > 0 ,$ , the quantities in (25) satisfy

$$
\nu _ { t } = \frac { 2 } { k ( k + 1 ) } , \quad P _ { 0 } \leq \frac { R ^ { 2 } } { T ( T + 1 ) } \leq \frac { R ^ { 2 } } { T ^ { 2 } } , \quad \eta _ { t } ^ { 2 } \nu _ { t } \leq \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } , \quad \eta _ { t } \sqrt { \nu _ { t - 1 } } \leq \frac { \sqrt { 2 } \eta } { T } ,\tag{30}
$$

where $k = T - t + 1$ . Moreover, $\nu _ { T } = 1$ and $\eta _ { T } = \eta / T$

Proof. For $k = T - t + 1$ , the sum defining $S _ { t }$ becomes

$$
{ \cal { S } } _ { t } = \frac { \eta } { T } \sum _ { j = t } ^ { T } ( T - j + 1 ) = \frac { \eta } { T } \sum _ { i = 1 } ^ { k } i = \frac { \eta k ( k + 1 ) } { 2 T } ,
$$

$$
\nu _ { t } = \frac { \eta / T } { \eta k ( k + 1 ) / ( 2 T ) } = \frac { 2 } { k ( k + 1 ) } ,
$$

$$
P _ { 0 } = \frac { 2 } { T ( T + 1 ) } \frac { \| x ^ { \star } - x _ { 1 } \| ^ { 2 } } { 2 } \leq \frac { R ^ { 2 } } { T ( T + 1 ) } \leq \frac { R ^ { 2 } } { T ^ { 2 } } ,
$$

$$
\eta _ { t } ^ { 2 } \nu _ { t } = \frac { \eta ^ { 2 } k ^ { 2 } } { T ^ { 2 } } \frac { 2 } { k ( k + 1 ) } = \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } \frac { k } { k + 1 } \le \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } .
$$

Since $\nu _ { t - 1 } \leq \nu _ { t }$ , the last bound also gives

$$
\eta _ { t } \sqrt { \nu _ { t - 1 } } \leq \eta _ { t } \sqrt { \nu _ { t } } = \sqrt { \eta _ { t } ^ { 2 } \nu _ { t } } \leq \frac { \sqrt { 2 } \eta } { T } .
$$

Putting $t = T$ in the definitions gives $\nu _ { T } = 1$ and $\eta _ { T } = \eta / T$ . All these calculations include $T = 1$ □

## A.3 Clipping error

Lemma 6. Let $\lambda > 0 , 1 < p \leq 2$ , and let ℎ be measurable with respect to the conditioning sigma-field, with $\| h \| \leq \lambda / 2$ . Suppose that $\begin{array} { r } { g = h + \varepsilon , \mathbb { E } _ { t } \varepsilon = 0 , } \end{array}$ , and $\mathbb { E } _ { t } \Vert \varepsilon \Vert ^ { p } \leq \sigma ^ { p }$ . Define $e = \mathrm { c l i p } ( g , \lambda ) - h , b = \mathbb { E } _ { t } e ,$ and $\xi = e - b$ . Then

$$
\| e \| \leq \frac { 3 } { 2 } \lambda , \qquad \mathbb { E } _ { t } \| e \| ^ { 2 } \leq V : = 2 \sigma ^ { p } \lambda ^ { 2 - p } ,\tag{31}
$$

$$
\| b \| \le B : = 2 \sigma ^ { p } \lambda ^ { 1 - p } ,
$$

$$
\| \xi \| \leq 2 \lambda ,
$$

$$
\mathbb { E } _ { t } \Vert \xi \Vert ^ { 2 } \leq V .\tag{32}
$$

Proof. Clipping is Euclidean projection onto the ball of radius �. Its nonexpansiveness follows from the proximal argument in Lemma 3, with � the indicator of that ball. Since $\mathrm { c l i p } ( h , \lambda ) = h$

$$
\| e \| = \| \mathrm { c l i p } ( g , \lambda ) - \mathrm { c l i p } ( h , \lambda ) \| \leq \| g - h \| = \| \varepsilon \| ,
$$

$$
\| e \| \leq \| \mathrm { c l i p } ( g , \lambda ) \| + \| h \| \leq \lambda + \lambda / 2 = 3 \lambda / 2 ,
$$

$$
\begin{array} { r } { \mathbb { E } _ { t } \Vert e \Vert ^ { 2 } \leq ( 3 \lambda / 2 ) ^ { 2 - p } \mathbb { E } _ { t } \Vert e \Vert ^ { p } \leq ( 3 \lambda / 2 ) ^ { 2 - p } \mathbb { E } _ { t } \Vert \varepsilon \Vert ^ { p } \leq 2 \sigma ^ { p } \lambda ^ { 2 - p } . } \end{array}
$$

The last line uses $0 \leq 2 - p < 1$ ; when $p = 2$ , its first inequality is simply an equality.

For the bias, $\mathbb { E } _ { t } \left( g - h \right) = 0$ implies

$$
b = \mathbb { E } _ { t } [ \mathrm { c l i p } ( g , \lambda ) - g ] , \qquad \| \mathrm { c l i p } ( g , \lambda ) - g \| = ( \| g \| - \lambda ) \mathbf { 1 } _ { \{ \| g \| > \lambda \} } .
$$

On $\{ \| g \| > \lambda \}$ , the triangle inequality gives

$$
\| \varepsilon \| \geq \| g \| - \| h \| > \lambda / 2 , \qquad \| g \| - \lambda \leq \| \varepsilon \| + \| h \| - \lambda \leq \| \varepsilon \| .
$$

Thus

$$
\begin{array} { r l } & { \| b \| \leq \mathbb { E } _ { t } \| \mathrm { c l i p } ( g , \lambda ) - g \| } \\ & { \qquad \leq \mathbb { E } _ { t } \left[ \| \varepsilon \| \mathbf { 1 } _ { \{ \| \varepsilon \| > \lambda / 2 \} } \right] } \\ & { \qquad = \mathbb { E } _ { t } \left[ \| \varepsilon \| ^ { p } \| \varepsilon \| ^ { 1 - p } \mathbf { 1 } _ { \{ \| \varepsilon \| > \lambda / 2 \} } \right] } \\ & { \qquad \leq ( \lambda / 2 ) ^ { 1 - p } \mathbb { E } _ { t } \| \varepsilon \| ^ { p } \leq 2 \sigma ^ { p } \lambda ^ { 1 - p } . } \end{array}
$$

The product in the third line is evaluated only on $\{ \| \varepsilon \| > \lambda / 2 \}$ . Finally,

$$
\xi = \mathrm { c l i p } ( g , \lambda ) - \mathbb { E } _ { t } \mathrm { c l i p } ( g , \lambda ) ,
$$

$$
\begin{array} { r } { \mathbb { E } _ { t } \xi = 0 , } \end{array}
$$

$$
\| \xi \| \leq \| \mathrm { c l i p } ( g , \lambda ) \| + \mathbb { E } _ { t } \| \mathrm { c l i p } ( g , \lambda ) \| \leq 2 \lambda ,
$$

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \| \xi \| ^ { 2 } = \mathbb { E } _ { t } \left[ \| e \| ^ { 2 } - 2 \langle e , b \rangle + \| b \| ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } _ { t } \| e \| ^ { 2 } - 2 \langle \mathbb { E } _ { t } e , b \rangle + \| b \| ^ { 2 } = \mathbb { E } _ { t } \| e \| ^ { 2 } - \| b \| ^ { 2 } \leq V . } \end{array}
$$

Here $^ b$ is measurable before the current sample is observed.

## A.4 Concentration inequalities

The following lemmas use the filtration $\mathcal { F } _ { 0 } , \ldots , \mathcal { F } _ { T }$ and $\mathbb { E } _ { t } \left[ \cdot \right] = \mathbb { E } \left[ \cdot \mid \mathcal { F } _ { t - 1 } \right]$

Lemma 7. Suppose $X _ { t }$ is $\mathcal { F } _ { t }$ -measurable, $0 \leq X _ { t } \leq H$ almost surely, and $\mathbb { E } _ { t } X _ { t } \le m _ { t }$ , where $m _ { t } \geq 0$ is F -measurable and $\begin{array} { r } { \sum _ { t = 1 } ^ { T } m _ { t } \leq A } \end{array}$ almost surelyfor a deterministic �. For $u > 0$

$$
\mathbb { P } \left\{ \sum _ { t = 1 } ^ { T } X _ { t } \leq 2 A + H u \right\} \geq 1 - e ^ { - u } .
$$

Proof. If $H = 0$ , then $X _ { t } = 0$ almost surely. For $H > 0$ , convexity of the exponential gives, for $0 \leq x \leq H$

$$
\exp ( x / H ) \le ( 1 - x / H ) \exp ( 0 ) + ( x / H ) \exp ( 1 ) = 1 + ( { \mathrm e } - 1 ) x / H .
$$

Consequently,

$$
\mathbb { E } _ { t } \exp ( X _ { t } / H ) \le 1 + ( \mathrm { e } - 1 ) \mathbb { E } _ { t } X _ { t } / H \le 1 + ( \mathrm { e } - 1 ) m _ { t } / H \le \exp ( ( \mathrm { e } - 1 ) m _ { t } / H ) .
$$

Set

$$
Z _ { s } = \exp \left( \frac { 1 } { H } \sum _ { t = 1 } ^ { s } X _ { t } - \frac { \mathrm { e } - 1 } { H } \sum _ { t = 1 } ^ { s } m _ { t } \right) , \qquad Z _ { 0 } = 1 .
$$

Since $Z _ { s - 1 }$ and $m _ { s }$ are $\mathscr { F } _ { s - 1 }$ -measurable,

$$
\begin{array} { r l } & { \mathbb { E } [ Z _ { s } \mid \mathcal { F } _ { s - 1 } ] = Z _ { s - 1 } \exp \left( - \displaystyle \frac { \mathrm { e } - 1 } { H } m _ { s } \right) \mathbb { E } [ \exp ( X _ { s } / H ) \mid \mathcal { F } _ { s - 1 } ] \leq Z _ { s - 1 } , } \\ & { \quad \quad \mathbb { E } Z _ { T } \leq \mathbb { E } Z _ { T - 1 } \leq \cdots \leq \mathbb { E } Z _ { 0 } = 1 . } \end{array}
$$

Using $\begin{array} { r } { \sum m _ { t } \le A , \mathrm { e } - 1 < 2 } \end{array}$ , and Markov’s inequality,

$$
\begin{array}{c} \mathbb { P } \left\{ \sum _ { t = 1 } ^ { T } X _ { t } > 2 A + H u \right\} \leq \mathbb { P } \left\{ \sum _ { t = 1 } ^ { T } X _ { t } > ( \mathrm { e } - 1 ) A + H u \right\}  \\ { \leq \mathbb { P } \{ Z _ { T } > e ^ { u } \} \leq e ^ { - u } \mathbb { E } Z _ { T } \leq e ^ { - u } . } \end{array}
$$

□

Lemma 8. Let $\begin{array} { r } { M _ { s } = \sum _ { t = 1 } ^ { s } Y _ { t } , } \end{array}$ , where $Y _ { t }$ is $\mathcal { F } _ { t }$ -measurable, $\mathbb { E } _ { t } Y _ { t } = 0 , | Y _ { t } | \leq a ,$ , and $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } Y _ { t } ^ { 2 } \leq } \end{array}$ � almost surelyfor deterministic $a , w \geq 0$ . For every $u > 0$

$$
\mathbb { P } \left\{ \operatorname* { m a x } _ { 1 \leq s \leq T } M _ { s } \leq \sqrt { 2 w u } + 2 a u \right\} \geq 1 - e ^ { - u } .\tag{33}
$$

Proof. If $a = 0 ,$ , all increments vanish. If $w = 0 .$ , then $\begin{array} { r } { \sum _ { t } \mathbb { E } Y _ { t } ^ { 2 } = \mathbb { E } \sum _ { t } \mathbb { E } _ { t } Y _ { t } ^ { 2 } = 0 } \end{array}$ , so the same conclusion holds. Assume $a , w > 0$ and fix $0 < \theta < 3 / a$ . For $k \geq 2 , k ! \geq 2 \cdot 3 ^ { k - 2 }$ and $| Y _ { t } | ^ { k } \leq a ^ { k - 2 } Y _ { t } ^ { 2 }$ . Hence

$$
\begin{array} { r l r } {  { \mathbb { E } _ { t } e ^ { \theta Y _ { t } } = 1 + \theta \mathbb { E } _ { t } Y _ { t } + \sum _ { k = 2 } ^ { \infty } \frac { \theta ^ { k } } { k ! } \mathbb { E } _ { t } Y _ { t } ^ { k } } } \\ & { } & { \leq 1 + \displaystyle \sum _ { k = 2 } ^ { \infty } \frac { \theta ^ { k } a ^ { k - 2 } } { k ! } \mathbb { E } _ { t } Y _ { t } ^ { 2 } } \\ & { } & { \leq 1 + \displaystyle \frac { \theta ^ { 2 } } { 2 } \mathbb { E } _ { t } Y _ { t } ^ { 2 } \sum _ { k = 2 } ^ { \infty } ( \theta a / 3 ) ^ { k - 2 } } \\ & { } & { = 1 + \displaystyle \frac { \theta ^ { 2 } \mathbb { E } _ { t } Y _ { t } ^ { 2 } } { 2 ( 1 - \theta a / 3 ) } \leq \exp ( \frac { \theta ^ { 2 } \mathbb { E } _ { t } Y _ { t } ^ { 2 } } { 2 ( 1 - \theta a / 3 ) } ) . } \end{array}
$$

Boundedness of $Y _ { t }$ justifies taking conditional expectations of the absolutely convergent series. Define

$$
W _ { s } = \sum _ { t = 1 } ^ { s } \mathbb { E } _ { t } Y _ { t } ^ { 2 } , \qquad Z _ { s } = \exp \left( \theta M _ { s } - \frac { \theta ^ { 2 } W _ { s } } { 2 ( 1 - \theta a / 3 ) } \right) , \qquad Z _ { 0 } = 1 .
$$

The preceding conditional bound gives

$$
\mathbb { E } [ Z _ { s } \mid { \mathcal { F } } _ { s - 1 } ] = Z _ { s - 1 } \exp \left( - { \frac { \theta ^ { 2 } \mathbb { E } [ Y _ { s } ^ { 2 } \mid { \mathcal { F } } _ { s - 1 } ] } { 2 ( 1 - \theta a / 3 ) } } \right) \mathbb { E } [ e ^ { \theta Y _ { s } } \mid { \mathcal { F } } _ { s - 1 } ] \leq Z _ { s - 1 } .
$$

For $c > 1$ , let � be the first $s \in \{ 1 , \ldots , T \}$ with $Z _ { s } \geq c .$ , and let $\tau = T + 1$ if there is no such index. Since $\{ \tau \geq s \} \in \mathcal { F } _ { s - 1 }$

$$
\begin{array} { c l } { Z _ { s \wedge \tau } - Z _ { ( s - 1 ) \wedge \tau } = \mathbf { 1 } _ { \{ \tau \geq s \} } ( Z _ { s } - Z _ { s - 1 } ) , } \\ { \mathbb { E } [ Z _ { s \wedge \tau } - Z _ { ( s - 1 ) \wedge \tau } \mid \mathcal { F } _ { s - 1 } ] \leq 0 , } \\ { c \mathbb { P } \{ \tau \leq T \} \leq \mathbb { E } Z _ { T \wedge \tau } \leq 1 . } \end{array}
$$

Thus $\mathbb { P } \{ \operatorname* { m a x } _ { s \leq T } Z _ { s } \geq c \} \leq c ^ { - 1 }$

For a threshold $r > 0$ , take $\theta = r / ( w + a r / 3 )$ . Then $\theta a < 3$ and

$$
1 - \theta a / 3 = \frac { w } { w + a r / 3 } , \qquad \theta r - \frac { \theta ^ { 2 } w } { 2 ( 1 - \theta a / 3 ) } = \frac { r ^ { 2 } } { 2 ( w + a r / 3 ) } .
$$

Since $W _ { s } \leq w$ , we obtain

$$
\begin{array} { r l r } {  { \mathbb { P } \{ \underset { s \leq T } { \operatorname* { m a x } } M _ { s } \geq r \} \leq \mathbb { P } \{ \underset { s \leq T } { \operatorname* { m a x } } Z _ { s } \geq \exp ( \frac { r ^ { 2 } } { 2 ( w + a r / 3 ) } ) \} } } \\ & { } & { \leq \exp ( - \frac { r ^ { 2 } } { 2 ( w + a r / 3 ) } ) . } \end{array}
$$

For $r = \sqrt { 2 w u } + 2 a u$ , direct expansion gives

$$
\begin{array} { c } { { r ^ { 2 } = 2 w u + 4 a u \sqrt { 2 w u } + 4 a ^ { 2 } u ^ { 2 } , } } \\ { { } } \\ { { 2 u ( w + a r / 3 ) = 2 w u + \displaystyle \frac { 2 } { 3 } a u \sqrt { 2 w u } + \displaystyle \frac { 4 } { 3 } a ^ { 2 } u ^ { 2 } , } } \\ { { } } \\ { { r ^ { 2 } - 2 u ( w + a r / 3 ) = \displaystyle \frac { 1 0 } { 3 } a u \sqrt { 2 w u } + \displaystyle \frac { 8 } { 3 } a ^ { 2 } u ^ { 2 } \geq 0 . } } \end{array}
$$

The last probability is therefore at most $e ^ { - u }$ . This proves the stated maximal form of Freedman’s inequality (Freedman, 1975); the maximum over � requires no additional union bound. □

## B Proof of Theorem 1

Write $K = 1 2 8$ and $\Delta _ { t } = f ( x _ { t } ) - f ( x ^ { \star } )$ . The parameters are

$$
\ell = \log ( 8 / \delta ) , \qquad \lambda = \operatorname* { m a x } \{ 4 L R , \sigma ( T / \ell ) ^ { 1 / p } \} , \qquad \eta = \frac { R } { K \lambda \ell } , \qquad \eta _ { t } = \eta \frac { T - t + 1 } { T } .
$$

## B.1 One-step inequalities and the stopping rule

Define

$$
\begin{array} { r } { e _ { t } = \mathrm { c l i p } ( g _ { t } , \lambda ) - \nabla f ( x _ { t } ) , \qquad b _ { t } = \mathbb { E } _ { t } e _ { t } , \qquad \xi _ { t } = e _ { t } - b _ { t } . } \end{array}\tag{34}
$$

Then $\mathbb { E } _ { t } { \xi } _ { t } = 0$ and $x _ { t + 1 } = x _ { t } - \eta _ { t } \big ( \nabla f ( x _ { t } ) + e _ { t } \big )$ . Moreover,

$$
\eta _ { t } \leq \eta = \frac { R } { K \lambda \ell } \leq \frac { R } { 4 K L R \ell } = \frac { 1 } { 4 K L \ell } \leq \frac { 1 } { 2 L } .
$$

Lemmas 2 and 3, with $r = 0$ , imply

$$
\eta _ { t } ( f ( x _ { t + 1 } ) - f ( u ) ) \leq D ( u , x _ { t } ) - D ( u , x _ { t + 1 } ) + \eta _ { t } \langle e _ { t } , u - x _ { t } \rangle + \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } ,\tag{35}
$$

$$
\begin{array} { r } { \| { x } _ { t + 1 } - { x } ^ { \star } \| ^ { 2 } \leq \| { x } _ { t } - { x } ^ { \star } \| ^ { 2 } - 2 \eta _ { t } \langle e _ { t } , { x } _ { t } - { x } ^ { \star } \rangle + 2 \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } . } \end{array}\tag{36}
$$

Use $S _ { t } , \nu _ { t } , z _ { t } , P _ { t }$ from (25) and $A _ { t } = \eta _ { t } \nu _ { t - 1 } \bigl ( z _ { t - 1 } - x _ { t } \bigr )$ . Applying Lemma 4 to (35) gives, for every $1 \leq s \leq T$

$$
\begin{array} { l } { { \displaystyle P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } \leq P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } + S _ { s + 1 } \sum _ { j = 2 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } } } \\ { { \displaystyle \qquad \leq P _ { 0 } + \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } \| e _ { t } \| ^ { 2 } + \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle } . } \end{array}
$$

In particular, the following slightly looser bound will also be used in the smooth extensions:

$$
P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } \leq P _ { 0 } + 2 \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } \Vert e _ { t } \Vert ^ { 2 } + \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle .\tag{37}
$$

Lemma $5$ gives

$$
P _ { 0 } \leq \frac { R ^ { 2 } } { T ^ { 2 } } , \qquad 2 \eta _ { t } ^ { 2 } \nu _ { t } \leq \frac { 4 \eta ^ { 2 } } { T ^ { 2 } } , \qquad \eta _ { t } \sqrt { \nu _ { t - 1 } } \leq \frac { \sqrt { 2 } \eta } { T } .\tag{38}
$$

To bound the gradient and the coeficient $A _ { t } .$ , let

$$
\tau = \operatorname* { m i n } \left\{ t \in \{ 1 , \ldots , T + 1 \} : \| x _ { t } - x ^ { \star } \| > 2 R \ \mathrm { o r } \ P _ { t - 1 } > \frac { 4 R ^ { 2 } } { T ^ { 2 } } \right\} , \qquad I _ { t } = \mathbf { 1 } _ { \{ t < \tau \} } ,
$$

with min $\varnothing = \infty$ . For $1 \leq t \leq T$

$$
\left\{ t < \tau \right\} = \bigcap _ { j = 1 } ^ { t } \left\{ \left. x _ { j } - x ^ { \star } \right. \leq 2 R , \ P _ { j - 1 } \leq \frac { 4 R ^ { 2 } } { T ^ { 2 } } \right\} \in \mathcal { F } _ { t - 1 } .
$$

Indeed, each $x _ { j }$ and $P _ { j - 1 }$ is known before query $j .$ . Thus $I _ { t }$ is predictable, meaning that it is known before $g _ { t }$ is observed. The initial bounds imply $\tau \neq 1$

On $\{ I _ { t } = 1 \}$ , unconstrained optimality gives $\nabla f ( x ^ { \star } ) = 0$ , so

$$
\| \nabla f ( x _ { t } ) \| = \| \nabla f ( x _ { t } ) - \nabla f ( x ^ { \star } ) \| \leq L \| x _ { t } - x ^ { \star } \| \leq 2 L R \leq \lambda / 2 .
$$

Lemma 6 therefore gives, on this predictable event,

$$
\begin{array} { r l r l r l } & { \| e _ { t } \| \le \frac { 3 } { 2 } \lambda , } & { \qquad } & & { \| b _ { t } \| \le B : = 2 \sigma ^ { p } \lambda ^ { 1 - p } , } & { \qquad } & & { \mathbb { E } _ { t } \| e _ { t } \| ^ { 2 } \le V : = 2 \sigma ^ { p } \lambda ^ { 2 - p } , } \\ & { \| \xi _ { t } \| \le 3 \lambda , } & { \qquad } & & { \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } \le V . } \end{array}\tag{39}
$$

We use 3� in place of the lemma’s bound 2�. Since $\lambda ^ { p } \ge \sigma ^ { p } T / \ell ,$

$$
B T = 2 \lambda \frac { { \sigma } ^ { p } T } { \lambda ^ { p } } \leq 2 \lambda \ell , \quad \quad V T = 2 \lambda ^ { 2 } \frac { { \sigma } ^ { p } T } { \lambda ^ { p } } \leq 2 \lambda ^ { 2 } \ell .\tag{40}
$$

Also, $P _ { t - 1 } = \nu _ { t - 1 } \| z _ { t - 1 } - x _ { t } \| ^ { 2 } / 2$ implies

$$
\begin{array} { l } { { \displaystyle I _ { t } \| A _ { t } \| = I _ { t } \eta _ { t } \nu _ { t - 1 } \| z _ { t - 1 } - x _ { t } \| = I _ { t } \eta _ { t } \sqrt { \nu _ { t - 1 } } \sqrt { 2 P _ { t - 1 } } } } \\ { { \displaystyle \quad \leq \frac { \sqrt { 2 } \eta } { T } \sqrt { \frac { 8 R ^ { 2 } } { T ^ { 2 } } } = \frac { 4 \eta R } { T ^ { 2 } } } . } \end{array}\tag{41}
$$

## B.2 Bounds for the stopped errors

First, by predictability and (39),

$$
\begin{array} { l l } { \displaystyle 0 \le I _ { t } \| e _ { t } \| ^ { 2 } \le \frac 9 4 \lambda ^ { 2 } \le 3 \lambda ^ { 2 } , } \\ { \displaystyle \mathbb { E } _ { t } [ I _ { t } \| e _ { t } \| ^ { 2 } ] = I _ { t } \mathbb { E } _ { t } \| e _ { t } \| ^ { 2 } \le I _ { t } V , } \end{array}
$$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } [ I _ { t } \| e _ { t } \| ^ { 2 } ] \leq V T .
$$

Lemma 7, with $H = 3 \lambda ^ { 2 } , A = V T$ , and $u = \ell ,$ gives, with probability at least $1 - e ^ { - \ell }$

$$
\sum _ { t = 1 } ^ { T } I _ { t } \| e _ { t } \| ^ { 2 } \leq 2 V T + 3 \lambda ^ { 2 } \ell \leq 4 \lambda ^ { 2 } \ell + 3 \lambda ^ { 2 } \ell = 7 \lambda ^ { 2 } \ell .\tag{42}
$$

For the radius recursion, define

$$
M _ { s } ^ { r } = - 2 \sum _ { t = 1 } ^ { s } I _ { t } \eta _ { t } \langle \xi _ { t } , x _ { t } - x ^ { \star } \rangle , \qquad M _ { 0 } ^ { r } = 0 .
$$

Its conditional means, increment bound, and variance bound are

$$
\mathbb { E } _ { t } \left[ M _ { t } ^ { r } - M _ { t - 1 } ^ { r } \right] = - 2 I _ { t } \eta _ { t } \langle \mathbb { E } _ { t } \xi _ { t } , x _ { t } - x ^ { \star } \rangle = 0 ,
$$

$$
\begin{array} { r } { \vert M _ { t } ^ { r } - M _ { t - 1 } ^ { r } \vert \le 2 I _ { t } \eta _ { t } \Vert \xi _ { t } \Vert \vert \vert x _ { t } - x ^ { \star } \vert \vert \le 2 \eta ( 3 \lambda ) ( 2 R ) = 1 2 \eta R \lambda , } \end{array}
$$

$$
\begin{array} { r } { \mathbb { E } _ { t } [ ( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } ) ^ { 2 } ] \leq 4 I _ { t } \eta _ { t } ^ { 2 } \| x _ { t } - x ^ { \star } \| ^ { 2 } \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } \leq 1 6 \eta ^ { 2 } R ^ { 2 } V , } \end{array}
$$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } [ ( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } ) ^ { 2 } ] \leq 1 6 \eta ^ { 2 } R ^ { 2 } V T .
$$

Lemma $^ { 8 , }$ with $a = 1 2 \eta R \lambda , w = 1 6 \eta ^ { 2 } R ^ { 2 } V T _ { \mathrm { ~ } }$ , and $u = \ell ,$ gives, with probability at least $1 - e ^ { - \ell }$

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { s \leq T } M _ { s } ^ { r } \leq \sqrt { 2 ( 1 6 \eta ^ { 2 } R ^ { 2 } V T ) \ell } + 2 ( 1 2 \eta R \lambda ) \ell } \\ { \displaystyle \qquad = \eta R \sqrt { 3 2 V T \ell } + 2 4 \eta R \lambda \ell } \\ { \displaystyle \qquad \leq \eta R \sqrt { 6 4 \lambda ^ { 2 } \ell ^ { 2 } } + 2 4 \eta R \lambda \ell = 3 2 \eta R \lambda \ell = \frac { 3 2 R ^ { 2 } } { K } . } \end{array}\tag{43}
$$

The corresponding bias is bounded on every sample path:

$$
\begin{array} { r l r } {  { 2 \sum _ { t = 1 } ^ { T } I _ { t } \eta _ { t } | \langle b _ { t } , x _ { t } - x ^ { \star } \rangle | \leq 2 \sum _ { t = 1 } ^ { T } I _ { t } \eta _ { t } \| b _ { t } \| \| x _ { t } - x ^ { \star } \| } } \\ & { } & { \leq 4 \eta R B T \leq 8 \eta R \lambda \ell = \frac { 8 R ^ { 2 } } { K } . } \end{array}\tag{44}
$$

For the comparison potential, define

$$
M _ { s } ^ { P } = \sum _ { t = 1 } ^ { s } I _ { t } \langle \xi _ { t } , A _ { t } \rangle , \qquad M _ { 0 } ^ { P } = 0 .
$$

The vector $A _ { t }$ is $\mathscr { F } _ { t - 1 }$ -measurable. By (41),

$$
\begin{array} { r l } & { \mathbb { E } _ { t } [ M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ] = I _ { t } \langle \mathbb { E } _ { t } \xi _ { t } , A _ { t } \rangle = 0 , } \\ & { \qquad \quad | M _ { t } ^ { P } - M _ { t - 1 } ^ { P } | \le I _ { t } \| \xi _ { t } \| \| A _ { t } \| \le ( 3 \lambda ) \displaystyle \frac { 4 \eta R } { T ^ { 2 } } = \displaystyle \frac { 1 2 \eta R \lambda } { T ^ { 2 } } , } \\ & { \qquad \mathbb { E } _ { t } [ ( M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ) ^ { 2 } ] \le I _ { t } \| A _ { t } \| ^ { 2 } \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } \le \displaystyle \frac { 1 6 \eta ^ { 2 } R ^ { 2 } V } { T ^ { 4 } } , } \\ & { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } [ ( M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ) ^ { 2 } ] \le \displaystyle \frac { 1 6 \eta ^ { 2 } R ^ { 2 } V T } { T ^ { 4 } } . } \end{array}
$$

Applying Lemma 8 with these two bounds gives, with probability at least $1 - e ^ { - \ell }$

$$
\begin{array} { r l r } {  { \operatorname* { m a x } _ { s \leq T } M _ { s } ^ { P } \leq \sqrt { 2 \frac { 1 6 \eta ^ { 2 } R ^ { 2 } V T } { T ^ { 4 } } } \ell + 2 \frac { 1 2 \eta R \lambda } { T ^ { 2 } } \ell } } \\ & { } & \\ & { } & { \quad = \frac { \eta R } { T ^ { 2 } } \sqrt { 3 2 V T \ell } + \frac { 2 4 \eta R \lambda \ell } { T ^ { 2 } } } \\ & { } & { \quad \leq \frac { 8 \eta R \lambda \ell } { T ^ { 2 } } + \frac { 2 4 \eta R \lambda \ell } { T ^ { 2 } } = \frac { 3 2 R ^ { 2 } } { K T ^ { 2 } } . } \end{array}\tag{45}
$$

The bias satisfies

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } | \langle b _ { t } , A _ { t } \rangle | \leq \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \| b _ { t } \| \| A _ { t } \| \leq \frac { 4 \eta R B T } { T ^ { 2 } } } \\ { \displaystyle \leq \frac { 8 \eta R \lambda \ell } { T ^ { 2 } } = \frac { 8 R ^ { 2 } } { K T ^ { 2 } } . } \end{array}\tag{46}
$$

Let � be the event on which (42), (43), and (45) all hold. A union bound gives

$$
\begin{array} { r } { \mathbb { P } ( E ^ { c } ) \le 3 e ^ { - \ell } = 3 \delta / 8 \le \delta . } \end{array}
$$

Both martingale bounds hold for every prefix on $E .$

## B.3 Removing the stopping rule and bounding the last iterate

Fix a sample path in $E$ and an index $1 \leq s \leq T$ with $s < \tau$ . Then $I _ { t } = 1$ for all $t \leq s .$ . Summing (36) gives

$$
\begin{array} { r l r } {  { \| { \boldsymbol x } _ { s + 1 } - { \boldsymbol x } ^ { \star } \| ^ { 2 } - \| { \boldsymbol x } _ { 1 } - { \boldsymbol x } ^ { \star } \| ^ { 2 } = \displaystyle \sum _ { t = 1 } ^ { s } \big ( \| { \boldsymbol x } _ { t + 1 } - { \boldsymbol x } ^ { \star } \| ^ { 2 } - \| { \boldsymbol x } _ { t } - { \boldsymbol x } ^ { \star } \| ^ { 2 } \big ) } } \\ & { } & { \leq - 2 \displaystyle \sum _ { t = 1 } ^ { s } \eta _ { t } \langle b _ { t } + \xi _ { t } , { \boldsymbol x } _ { t } - { \boldsymbol x } ^ { \star } \rangle + 2 \displaystyle \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } . } \end{array}
$$

Using (42), (43), and (44), we obtain

$$
\begin{array} { l } { \displaystyle \| x _ { s + 1 } - x ^ { \star } \| ^ { 2 } \leq R ^ { 2 } + M _ { s } ^ { r } + 2 \displaystyle \sum _ { t = 1 } ^ { s } I _ { t } \eta _ { t } | \langle b _ { t } , x _ { t } - x ^ { \star } \rangle | + 2 \eta ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \| e _ { t } \| ^ { 2 } } \\ { \leq R ^ { 2 } + \frac { 3 2 R ^ { 2 } } { K } + \frac { 8 R ^ { 2 } } { K } + 1 4 \eta ^ { 2 } \lambda ^ { 2 } \ell } \\ { = \left( 1 + \frac { 4 0 } { K } + \frac { 1 4 } { K ^ { 2 } \ell } \right) R ^ { 2 } < 2 R ^ { 2 } . } \end{array}\tag{47}
$$

Using (37) and the bounds for the comparison potential,

$$
\begin{array} { l } { \displaystyle P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } \leq P _ { 0 } + 2 \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } \| e _ { t } \| ^ { 2 } + \sum _ { t = 1 } ^ { s } \langle b _ { t } + \xi _ { t } , A _ { t } \rangle } \\ { \displaystyle \qquad \leq \frac { R ^ { 2 } } { T ^ { 2 } } + \frac { 4 \eta ^ { 2 } } { T ^ { 2 } } \sum _ { t = 1 } ^ { T } I _ { t } \| e _ { t } \| ^ { 2 } + M _ { s } ^ { P } + \sum _ { t = 1 } ^ { s } I _ { t } | \langle b _ { t } , A _ { t } \rangle } \\ { \displaystyle \qquad \leq \frac { R ^ { 2 } } { T ^ { 2 } } + \frac { 2 8 \eta ^ { 2 } \lambda ^ { 2 } \ell } { T ^ { 2 } } + \frac { 3 2 R ^ { 2 } } { K T ^ { 2 } } + \frac { 8 R ^ { 2 } } { K T ^ { 2 } } } \\ { \displaystyle \qquad = \left( 1 + \frac { 4 0 } { K } + \frac { 2 8 } { K ^ { 2 } \ell } \right) \frac { R ^ { 2 } } { T ^ { 2 } } < \frac { 2 R ^ { 2 } } { T ^ { 2 } } . } \end{array}\tag{48}
$$

The parameter substitutions and numerical inequalities used above are

$$
\eta ^ { 2 } \lambda ^ { 2 } \ell = \frac { R ^ { 2 } } { K ^ { 2 } \ell } , \qquad 1 + \frac { 4 0 } { K } + \frac { 2 8 } { K ^ { 2 } \ell } \le 1 + \frac { 4 0 } { 1 2 8 } + \frac { 2 8 } { 1 2 8 ^ { 2 } } < 2 ,
$$

because $\ell > 1$

If $\tau \leq T + 1$ , then $\tau \neq 1$ and $s = \tau - 1 \in \{ 1 , . . . , T \}$ . Since $s < \tau$ , the two preceding estimates apply and imply

$$
\| x _ { \tau } - x ^ { \star } \| ^ { 2 } < 2 R ^ { 2 } < 4 R ^ { 2 } , \qquad P _ { \tau - 1 } \leq P _ { \tau - 1 } + \eta _ { \tau - 1 } \nu _ { \tau - 1 } \Delta _ { \tau } < { \frac { 2 R ^ { 2 } } { T ^ { 2 } } } < { \frac { 4 R ^ { 2 } } { T ^ { 2 } } } .
$$

Neither stopping condition holds at �, a contradiction. Thus $\tau = \infty$ on $E$ , and $\| x _ { t } - x ^ { \star } \| \leq 2 R$ for $1 \leq t \leq T + 1$ . Taking $s = T$ in (48) now gives

$$
\begin{array} { r l } & { \displaystyle \frac { \eta } { T } \big ( f ( x _ { T + 1 } ) - f ( x ^ { \star } ) \big ) = \eta _ { T } \nu _ { T } \Delta _ { T + 1 } \leq P _ { T } + \eta _ { T } \nu _ { T } \Delta _ { T + 1 } \leq \frac { 2 R ^ { 2 } } { T ^ { 2 } } , } \\ & { \qquad f ( x _ { T + 1 } ) - f ( x ^ { \star } ) \leq \displaystyle \frac { 2 R ^ { 2 } } { \eta T } = \frac { 2 K R \lambda \ell } { T } = \frac { 2 5 6 R \lambda \ell } { T } } \\ & { \qquad \leq \displaystyle \frac { 2 5 6 R \ell } { T } \left( 4 L R + \sigma ( T / \ell ) ^ { 1 / p } \right) } \\ & { \qquad = 1 0 2 4 \frac { L R ^ { 2 } \ell } { T } + 2 5 6 \sigma R \left( \frac { \ell } { T } \right) ^ { ( p - 1 ) / p } . } \end{array}
$$

The last inequality uses max $\{ a , b \} \leq a + b$ for $a , b \geq 0$

□

## C Proof of Theorem 2

Let $K = 1 2 8$ . The parameters in the theorem are

$$
\ell = \log ( 4 / \delta ) , \qquad \lambda = \operatorname* { m a x } \{ 2 G , \sigma ( T / \ell ) ^ { 1 / p } \} , \qquad H = G \sqrt { T } + \lambda \ell , \qquad \eta = \frac { R } { K H } , \qquad \eta _ { t } = \eta \frac { T - t + 1 } { T } .
$$

Since $G + \sigma > 0$ , we have $\lambda , H > 0$ . Write $\Delta _ { t } = F ( x _ { t } ) - F ^ { \star }$ and $D ( u , x ) = \| u - x \| ^ { 2 } / 2$ . The proximal updates belong to dom $r ,$ and $\Delta _ { t } \geq 0$ because $x ^ { \star }$ minimizes �.

## C.1 The one-step inequality and its telescoping sum

Let $h _ { t } = \mathbb { E } _ { t } g _ { t } \in \partial f ( x _ { t } )$ and define

$$
e _ { t } = \operatorname { c l i p } ( g _ { t } , \lambda ) - h _ { t } = b _ { t } + \xi _ { t } , \qquad b _ { t } = \mathbb { E } _ { t } e _ { t } , \qquad \mathbb { E } _ { t } \xi _ { t } = 0 .\tag{49}
$$

Lipschitz continuity gives $\| h _ { t } \| \leq G \leq \lambda / 2$ . Lemma 6 therefore yields, for every �,

$$
\Vert b _ { t } \Vert \leq B : = 2 \sigma ^ { p } \lambda ^ { 1 - p } , \qquad \quad \mathbb { E } _ { t } \Vert e _ { t } \Vert ^ { 2 } \leq V : = 2 \sigma ^ { p } \lambda ^ { 2 - p } ,\tag{50}
$$

$$
\| e _ { t } \| \leq \frac { 3 } { 2 } \lambda ,
$$

$$
\left\| \xi _ { t } \right\| \leq 3 \lambda .\tag{51}
$$

The last bound is a relaxation of the lemma’s 2� bound. Also,

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \Vert \xi _ { t } \Vert ^ { 2 } = \mathbb { E } _ { t } \Vert e _ { t } - b _ { t } \Vert ^ { 2 } = \mathbb { E } _ { t } \Vert e _ { t } \Vert ^ { 2 } - 2 \langle \mathbb { E } _ { t } e _ { t } , b _ { t } \rangle + \Vert b _ { t } \Vert ^ { 2 } } \\ & { \qquad = \mathbb { E } _ { t } \Vert e _ { t } \Vert ^ { 2 } - \Vert b _ { t } \Vert ^ { 2 } \leq V . } \end{array}
$$

Since $\lambda ^ { p } \ge \sigma ^ { p } T / \ell$

$$
B T = 2 \sigma ^ { p } T \lambda ^ { 1 - p } \le 2 \lambda \ell , \quad \quad V T = 2 \sigma ^ { p } T \lambda ^ { 2 - p } \le 2 \lambda ^ { 2 } \ell .\tag{52}
$$

Since $h _ { t } + e _ { t } = \mathrm { c l i p } ( g _ { t } , \lambda )$ , the update is $x _ { t + 1 } = \operatorname { p r o x } _ { m _ { t } r } ( x _ { t } - \eta _ { t } ( h _ { t } + e _ { t } ) )$ . Apply the nonsmooth inequality of Lemma 2 with $\begin{array} { r } { x = x _ { t } , x ^ { + } = x _ { t + 1 } , h = h _ { t } , e = e _ { t } , \mathrm { a n d } \alpha = \eta _ { t } } \end{array}$ . For every � ∈ dom $r ,$ this gives

$$
\begin{array} { r } { \eta _ { t } \big ( F ( x _ { t + 1 } ) - F ( u ) \big ) \leq D ( u , x _ { t } ) - D ( u , x _ { t + 1 } ) + \eta _ { t } \langle e _ { t } , u - x _ { t } \rangle + \eta _ { t } ^ { 2 } ( \| e _ { t } \| ^ { 2 } + 4 G ^ { 2 } ) . } \end{array}\tag{53}
$$

Use $S _ { t } , \nu _ { t } , z _ { t } , P _ { t }$ from (25) and set $A _ { t } ~ = ~ \eta _ { t } \nu _ { t - 1 } { \left( z _ { t - 1 } - x _ { t } \right) }$ . Applying Lemma 4 to (53) with $\rho _ { t } = \eta _ { t } ^ { 2 } ( \| e _ { t } \| ^ { 2 } + 4 G ^ { 2 } )$ gives, for every $1 \leq s \leq T$

$$
\begin{array} { l } { { P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } + S _ { s + 1 } \displaystyle \sum _ { j = 2 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } \leq P _ { 0 } + \displaystyle \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } ( \| e _ { t } \| ^ { 2 } + 4 G ^ { 2 } ) } } \\ { { \displaystyle \qquad + \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle } . } \end{array}\tag{54}
$$

All terms on the left are nonnegative.

The linear learning-rate schedule gives

$$
S _ { t } = \frac { \eta } { T } \sum _ { j = 1 } ^ { T - t + 1 } j = \frac { \eta ( T - t + 1 ) ( T - t + 2 ) } { 2 T } ,
$$

$$
\nu _ { t } = \frac { 2 } { ( T - t + 1 ) ( T - t + 2 ) } ,
$$

$$
\eta _ { t } ^ { 2 } \nu _ { t } = { \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } } { \frac { T - t + 1 } { T - t + 2 } } \leq { \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } } .
$$

In particular,

$$
\begin{array} { c c c } { { P _ { 0 } \leq \displaystyle \frac { R ^ { 2 } } { T ( T + 1 ) } \leq \displaystyle \frac { R ^ { 2 } } { T ^ { 2 } } , \qquad \eta _ { t } ^ { 2 } \nu _ { t } \leq \displaystyle \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } , } } \\ { { \eta _ { t } \sqrt { \nu _ { t - 1 } } \leq \eta _ { t } \sqrt { \nu _ { t } } \leq \displaystyle \frac { \sqrt { 2 } \eta } { T } . } } \end{array}\tag{55}
$$

These inequalities also hold at $t = 1$ , where $\nu _ { 0 } = \nu _ { 1 }$

## C.2 Bounds for the stopped error sums

Define

$$
\tau = \operatorname* { m i n } \left\{ t \in \{ 1 , \dots , T + 1 \} : P _ { t - 1 } > \frac { 4 R ^ { 2 } } { T ^ { 2 } } \right\} , \quad \quad I _ { t } = \mathbf { 1 } _ { \{ t < \tau \} } , \quad 1 \leq t \leq T ,
$$

with min $\varnothing = \infty$ . The initial bound in (55) gives $\tau \neq 1$ . Since $x _ { t } , z _ { t - 1 } , P _ { t - 1 }$ are $\mathcal { F } _ { t - 1 }$ -measurable and

$$
\left\{ t < \tau \right\} = \bigcap _ { j = 1 } ^ { t } \left\{ P _ { j - 1 } \leq \frac { 4 R ^ { 2 } } { T ^ { 2 } } \right\} \in \mathcal { F } _ { t - 1 } ,
$$

both $I _ { t }$ and $A _ { t }$ are measurable before $g _ { t }$ is sampled. When $I _ { t } = 1$

$$
\begin{array} { l } { \displaystyle \| A _ { t } \| = \eta _ { t } \nu _ { t - 1 } \| z _ { t - 1 } - x _ { t } \| = \eta _ { t } \sqrt { \nu _ { t - 1 } } \sqrt { 2 P _ { t - 1 } } } \\ { \displaystyle \leq \frac { \sqrt { 2 } \eta } { T } \sqrt { \frac { 8 R ^ { 2 } } { T ^ { 2 } } } = \frac { 4 \eta R } { T ^ { 2 } } . } \end{array}\tag{56}
$$

First, (50)–(51) imply $0 \le \| e _ { t } \| ^ { 2 } \le ( 9 / 4 ) \lambda ^ { 2 } \le 3 \lambda ^ { 2 }$ and $\mathbb { E } _ { t } \Vert e _ { t } \Vert ^ { 2 } \leq V$ . Lemma $^ { 7 , }$ with upper bound $3 \lambda ^ { 2 }$ , mean budget $V T .$ , and $u = \ell ,$ gives an event $\varepsilon _ { Q }$ of probability at least $1 - e ^ { - \ell }$ on which

$$
\sum _ { t = 1 } ^ { T } \lVert e _ { t } \rVert ^ { 2 } \leq 2 V T + 3 \lambda ^ { 2 } \ell \leq 4 \lambda ^ { 2 } \ell + 3 \lambda ^ { 2 } \ell = 7 \lambda ^ { 2 } \ell .\tag{57}
$$

No stopping is needed for this sum because the moment bounds hold at every iterate.

For the bias terms, (56), (52), and $H \geq \lambda \ell$ give

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } | \langle b _ { t } , A _ { t } \rangle | \leq \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \| b _ { t } \| \| A _ { t } \| \leq \frac { 4 \eta R B T } { T ^ { 2 } } } \\ & { } & { \quad \quad \leq \displaystyle \frac { 8 \eta R \lambda \ell } { T ^ { 2 } } = \frac { 8 R ^ { 2 } } { K T ^ { 2 } } \frac { \lambda \ell } { H } \leq \frac { 8 R ^ { 2 } } { K T ^ { 2 } } . } \end{array}\tag{58}
$$

For the centered terms, let

$$
M _ { s } = \sum _ { t = 1 } ^ { s } I _ { t } \langle \xi _ { t } , A _ { t } \rangle , \qquad M _ { 0 } = 0 .
$$

Predictability and (49) give

$$
\mathbb { E } _ { t } \left( M _ { t } - M _ { t - 1 } \right) = I _ { t } \langle \mathbb { E } _ { t } \xi _ { t } , A _ { t } \rangle = 0 .
$$

Thus $( M _ { s } ) _ { s = 0 } ^ { T }$ is a martingale. Its increments and their conditional second moments satisfy

$$
\begin{array} { c } { { \displaystyle \left| M _ { t } - M _ { t - 1 } \right| \le I _ { t } \| \xi _ { t } \| \| A _ { t } \| \le 3 \lambda \displaystyle \frac { 4 \eta R } { T ^ { 2 } } = \displaystyle \frac { 1 2 \eta R \lambda } { T ^ { 2 } } , } } \\ { { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left( M _ { t } - M _ { t - 1 } \right) ^ { 2 } \le \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \| A _ { t } \| ^ { 2 } \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } } } \\ { { \displaystyle \le \displaystyle \sum _ { t = 1 } ^ { T } \displaystyle \frac { 1 6 \eta ^ { 2 } R ^ { 2 } V } { T ^ { 4 } } = \displaystyle \frac { 1 6 \eta ^ { 2 } R ^ { 2 } V T } { T ^ { 4 } } . } } \end{array}\tag{59}
$$

(60)

Apply Lemma 8 with the right-hand sides of (59) and (60) as � and �, respectively, and with $u = \ell .$ On an event $\mathcal { E } _ { M }$ of probability at least $1 - e ^ { - \ell }$

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { 1 \leq s \leq T } M _ { s } \leq \sqrt { \frac { 3 2 \eta ^ { 2 } R ^ { 2 } V T \ell } { T ^ { 4 } } } + \frac { 2 4 \eta R \lambda \ell } { T ^ { 2 } } } \\ & { \quad \quad \quad = \frac { \eta R } { T ^ { 2 } } \sqrt { 3 2 V T \ell } + \frac { 2 4 \eta R \lambda \ell } { T ^ { 2 } } } \\ & { \quad \quad \quad \leq \frac { \eta R } { T ^ { 2 } } \sqrt { 6 4 \lambda ^ { 2 } \ell ^ { 2 } } + \frac { 2 4 \eta R \lambda \ell } { T ^ { 2 } } } \\ & { \quad \quad \quad = \frac { 3 2 \eta R \lambda \ell } { T ^ { 2 } } = \frac { 3 2 R ^ { 2 } } { K T ^ { 2 } } \frac { \lambda \ell } { H } \leq \frac { 3 2 R ^ { 2 } } { K T ^ { 2 } } . } \end{array}\tag{61}
$$

The third line uses $V T \leq 2 \lambda ^ { 2 } \ell .$ . This single event controls all prefixes �.

## C.3 Closing the stopping argument

Let $\mathcal { E } = \mathcal { E _ { Q } } \cap \mathcal { E } _ { M }$ . Then

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { E } ) \geq 1 - 2 e ^ { - \ell } = 1 - \delta / 2 \geq 1 - \delta . } \end{array}
$$

Fix a sample path in E and an index $1 \leq s \leq T$ with $s < \tau$ . All $I _ { t }$ with $t \leq s$ equal one, so

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle = \displaystyle \sum _ { t = 1 } ^ { s } I _ { t } \langle b _ { t } , A _ { t } \rangle + M _ { s } } \\ { \displaystyle \qquad \leq \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } | \langle b _ { t } , A _ { t } \rangle | + \displaystyle \operatorname* { m a x } _ { 1 \leq j \leq T } M _ { j } } \\ { \displaystyle \qquad \leq \frac { 8 R ^ { 2 } } { K T ^ { 2 } } + \frac { 3 2 R ^ { 2 } } { K T ^ { 2 } } = \frac { 4 0 R ^ { 2 } } { K T ^ { 2 } } . } \end{array}
$$

The other error terms in (54) satisfy

$$
\begin{array} { l } { { \displaystyle \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } ( \| e _ { t } \| ^ { 2 } + 4 G ^ { 2 } ) \leq \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } \left( \sum _ { t = 1 } ^ { T } \| e _ { t } \| ^ { 2 } + 4 G ^ { 2 } T \right) } } \\ { { \displaystyle \qquad \leq \frac { 2 \eta ^ { 2 } } { T ^ { 2 } } ( 7 \lambda ^ { 2 } \ell + 4 G ^ { 2 } T ) } } \\ { { \displaystyle \qquad = \frac { 1 4 \eta ^ { 2 } \lambda ^ { 2 } \ell + 8 \eta ^ { 2 } G ^ { 2 } T } { T ^ { 2 } } . } } \end{array}\tag{62}
$$

Here the first inequality uses (55) and nonnegativity, and the second uses (57). Substituting $\eta = R / ( K H )$ gives

$$
\eta ^ { 2 } \lambda ^ { 2 } \ell = \frac { R ^ { 2 } } { K ^ { 2 } \ell } \left( \frac { \lambda \ell } { H } \right) ^ { 2 } \leq \frac { R ^ { 2 } } { K ^ { 2 } \ell } , \qquad \eta ^ { 2 } G ^ { 2 } T = \frac { R ^ { 2 } } { K ^ { 2 } } \left( \frac { G \sqrt { T } } { H } \right) ^ { 2 } \leq \frac { R ^ { 2 } } { K ^ { 2 } } .
$$

Drop the nonnegative sum of gaps from (54) and combine the preceding bounds:

$$
\begin{array} { l } { \displaystyle P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } \leq \frac { R ^ { 2 } } { T ^ { 2 } } + \frac { 1 4 \eta ^ { 2 } \lambda ^ { 2 } \ell + 8 \eta ^ { 2 } G ^ { 2 } T } { T ^ { 2 } } + \frac { 4 0 R ^ { 2 } } { K T ^ { 2 } } } \\ { \displaystyle \qquad \leq \left( 1 + \frac { 4 0 } { K } + \frac { 1 4 } { K ^ { 2 } \ell } + \frac { 8 } { K ^ { 2 } } \right) \frac { R ^ { 2 } } { T ^ { 2 } } } \\ { \displaystyle \qquad \leq \left( 1 + \frac { 4 0 } { 1 2 8 } + \frac { 2 2 } { 1 2 8 ^ { 2 } } \right) \frac { R ^ { 2 } } { T ^ { 2 } } < \frac { 2 R ^ { 2 } } { T ^ { 2 } } . } \end{array}\tag{63}
$$

The last line uses $\ell >$ log $4 > 1$ and $K = 1 2 8$

Suppose $\tau \leq T + 1$ . Since $\tau \neq 1$ , the index $s = \tau - 1$ lies in $\{ 1 , \ldots , T \}$ and satisfies $s < \tau$ . Thus (63) applies at this index and gives

$$
P _ { \tau - 1 } = P _ { s } \leq P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } < \frac { 2 R ^ { 2 } } { T ^ { 2 } } < \frac { 4 R ^ { 2 } } { T ^ { 2 } } ,
$$

contrary to the definition of �. Hence $\tau = \infty$ on $\varepsilon _ { \mathrm { { } } }$ , and (63) holds for every $s \leq T$ . At � = �, $\nu _ { T } = 1$ and $\eta _ { T } = \eta / T$ , so

$$
\frac { \eta } { T } \Delta _ { T + 1 } \leq P _ { T } + \frac { \eta } { T } \Delta _ { T + 1 } < \frac { 2 R ^ { 2 } } { T ^ { 2 } } .
$$

It follows that

$$
\begin{array} { l } { { \displaystyle F ( x _ { T + 1 } ) - F ^ { \star } \le \frac { 2 R ^ { 2 } } { \eta T } = \frac { 2 K R H } { T } = \frac { 2 5 6 R H } { T } } } \\ { { \displaystyle \qquad = 2 5 6 \frac { G R } { \sqrt { T } } + 2 5 6 \frac { R \lambda \ell } { T } } } \\ { { \displaystyle \qquad \le 2 5 6 \frac { G R } { \sqrt { T } } + 2 5 6 \frac { R \ell } { T } \left( 2 G + \sigma ( T / \ell ) ^ { 1 / p } \right) } } \\ { { \displaystyle \qquad = 2 5 6 \frac { G R } { \sqrt { T } } + 5 1 2 \frac { G R \ell } { T } + 2 5 6 \sigma R ( \ell / T ) ^ { ( p - 1 ) / p } . } } \end{array}
$$

We used max $\{ 2 G , \sigma ( T / \ell ) ^ { 1 / p } \} \le 2 G + \sigma ( T / \ell ) ^ { 1 / p }$ . If $T \geq \ell ^ { 2 }$ , then

$$
2 5 6 { \frac { G R } { \sqrt { T } } } + 5 1 2 { \frac { G R \ell } { T } } = { \frac { G R } { \sqrt { T } } } \left( 2 5 6 + 5 1 2 { \frac { \ell } { \sqrt { T } } } \right) \leq 7 6 8 { \frac { G R } { \sqrt { T } } } .
$$

Both claimed bounds therefore hold on $\varepsilon$

□

## D Proof of Theorem 3

## D.1 Construction of the anchor

Let $g _ { 1 } ^ { ( 0 ) } , \ldots , g _ { m } ^ { ( 0 ) }$ be the pilot gradients at $x _ { 1 }$ , and let $\mathcal { H } _ { j }$ contain the information available after the first � pilot queries. In particular, the conditional oracle assumption gives

$$
\begin{array} { r } { \mathbb { E } \big [ g _ { j } ^ { ( 0 ) } \mid \mathcal { H } _ { j - 1 } \big ] = \nabla f ( x _ { 1 } ) , \qquad \mathbb { E } \big [ \| g _ { j } ^ { ( 0 ) } - \nabla f ( x _ { 1 } ) \| ^ { p } \mid \mathcal { H } _ { j - 1 } \big ] \leq \sigma ^ { p } . } \end{array}
$$

Lemma 9. Let $m = \lceil 3 2 \log ( 2 / \delta ) \rceil$ , and let � ∈ arg $\mathrm { \ m i n } _ { u \in \mathbb { R } ^ { d } } \sum _ { j = 1 } ^ { m } \| g _ { j } ^ { ( 0 ) } - u \|$ be a measurable choice of geometric median. Then

$$
\mathbb { P } \{ \| c - \nabla f ( x _ { 1 } ) \| \le 2 4 \sigma \} \ge 1 - \delta / 2 .
$$

Proof. Write $u _ { 0 } = \nabla f ( x _ { 1 } )$ . If $\sigma = 0$ , the conditional moment bound implies $g _ { j } ^ { ( 0 ) } = u _ { 0 }$ almost surely for every $j .$ . The geometric median is then $u _ { 0 }$ . Suppose that $\sigma > 0$ , and define

$$
J _ { j } = \mathbf { 1 } _ { \{ \| g _ { j } ^ { ( 0 ) } - u _ { 0 } \| > 8 \sigma \} } .
$$

Conditional Markov’s inequality gives

$$
\mathbb { E } [ J _ { j } \mid \mathcal { H } _ { j - 1 } ] \leq \frac { \mathbb { E } [ \| g _ { j } ^ { ( 0 ) } - u _ { 0 } \| ^ { p } \mid \mathcal { H } _ { j - 1 } ] } { ( 8 \sigma ) ^ { p } } \leq 8 ^ { - p } \leq \frac { 1 } { 8 } .
$$

Since $J _ { j } \in \{ 0 , 1 \}$ , we have $2 ^ { J _ { j } } = 1 + J _ { j }$ . Thus

$$
\mathbb { E } [ 2 ^ { J _ { j } } \mid \mathcal { H } _ { j - 1 } ] = 1 + \mathbb { E } [ J _ { j } \mid \mathcal { H } _ { j - 1 } ] \leq \frac { 9 } { 8 } ,
$$

$$
\begin{array} { r l } & { \mathbb { E } \left[ 2 ^ { \sum _ { j = 1 } ^ { m } J _ { j } } \right] = \mathbb { E } \left[ 2 ^ { \sum _ { j = 1 } ^ { m - 1 } J _ { j } } \mathbb { E } [ 2 ^ { J _ { m } } \mid \mathcal { H } _ { m - 1 } ] \right] } \\ & { \quad \quad \leq \displaystyle \frac { 9 } { 8 } \mathbb { E } \left[ 2 ^ { \sum _ { j = 1 } ^ { m - 1 } J _ { j } } \right] \leq \left( \frac { 9 } { 8 } \right) ^ { m } . } \end{array}
$$

Markov’s inequality now yields

$$
\begin{array} { r l r } {  { \mathbb { P } \{ \sum _ { j = 1 } ^ { m } J _ { j } > \frac { m } { 4 } \} \le 2 ^ { - m / 4 } ( \frac { 9 } { 8 } ) ^ { m } } } \\ & { } & { = \exp [ - m ( \frac { \log 2 } { 4 } - \log \frac { 9 } { 8 } ) ] \le e ^ { - m / 3 2 } \le \frac { \delta } { 2 } , } \end{array}
$$

where $( \log 2 ) / 4 - \log ( 9 / 8 ) > 1 / 3 2$ . The pilot samples may depend on the preceding pilot history.

On the complementary event, let $\mathcal { G } = \{ j : \| g _ { j } ^ { ( 0 ) } - u _ { 0 } \| \leq 8 \sigma \}$ . Then $| G | \ge 3 m / 4$ . The triangle inequality gives

$$
\| g _ { j } ^ { ( 0 ) } - c \| - \| g _ { j } ^ { ( 0 ) } - u _ { 0 } \| \ge \left\{ \begin{array} { l l } { \| c - u _ { 0 } \| - 1 6 \sigma , } & { j \in \mathcal { G } , } \\ { - \| c - u _ { 0 } \| , } & { j \notin \mathcal { G } . } \end{array} \right.
$$

By the minimizing property of $c ,$

$$
\begin{array} { c } { 0 \displaystyle \geq \sum _ { j = 1 } ^ { m } \bigl ( \| g _ { j } ^ { ( 0 ) } - c \| - \| g _ { j } ^ { ( 0 ) } - u _ { 0 } \| \bigr ) } \\ { \geq ( 2 | \mathcal G | - m ) \| c - u _ { 0 } \| - 1 6 \sigma | \mathcal G | , } \\ { \| c - u _ { 0 } \| \leq \displaystyle \frac { 1 6 \sigma | \mathcal G | } { 2 | \mathcal G | - m } \leq 2 4 \sigma . } \end{array}
$$

The denominator is positive, and the last inequality is equivalent to $| G | \geq 3 m / 4$ . This proves the claim. □

## D.2 Proximal recursions and localization

Let $\mathcal { A } = \{ \| c - \nabla f ( x _ { 1 } ) \| \leq 2 4 \sigma \}$ . Fix a pilot history in ${ \mathcal { A } } ,$ outside a set of probability zero, and work under the conditional law given $\mathcal { H } _ { m }$ . The anchor � is now fixed. The main-stage filtration $\{ \mathcal { F } _ { t } \}$ includes $\mathcal { H } _ { m }$ , and $\mathbb { E } _ { t }$ denotes conditional expectation given $\mathscr { F } _ { t - 1 }$ . The oracle assumptions hold with respect to this filtration. All concentration probabilities below refer to this conditional law.

Write $\Delta _ { t } = F ( x _ { t } ) - F ^ { \star }$ and define

$$
\eta = \frac { R } { 1 2 8 \lambda \ell } , \qquad e _ { t } = c + \mathrm { c l i p } ( g _ { t } - c , \lambda ) - \nabla f ( x _ { t } ) , \qquad b _ { t } = \mathbb { E } _ { t } e _ { t } , \qquad \xi _ { t } = e _ { t } - b _ { t } ,
$$

so that $\mathbb { E } _ { t } \xi _ { t } ~ = ~ 0$ . Set $B = 2 \sigma ^ { p } \lambda ^ { 1 - p }$ and $V = 2 \sigma ^ { p } \lambda ^ { 2 - p }$ . The parameter choice gives $\lambda \geq 6 L R$ and $\lambda ^ { p } \ge \sigma ^ { p } T / \ell$ . Hence

$$
\begin{array} { c } { { \eta _ { t } \leq \eta = \displaystyle \frac { R } { 1 2 8 \lambda \ell } \leq \displaystyle \frac { 1 } { 7 6 8 L \ell } \leq \displaystyle \frac { 1 } { 2 L } , } } \\ { { B T = 2 \sigma ^ { p } T \lambda ^ { 1 - p } \leq 2 \lambda ^ { p } \ell \lambda ^ { 1 - p } = 2 \lambda \ell , } } \\ { { V T = 2 \sigma ^ { p } T \lambda ^ { 2 - p } \leq 2 \lambda ^ { p } \ell \lambda ^ { 2 - p } = 2 \lambda ^ { 2 } \ell . } } \end{array}\tag{64}
$$

The update is $x _ { t + 1 } = \operatorname { p r o x } _ { \eta _ { t } r } \left( x _ { t } - \eta _ { t } \left( \nabla f ( x _ { t } ) + e _ { t } \right) \right)$ . Apply Lemma 2 with $\boldsymbol { x } = \boldsymbol { x } _ { t } , h = \nabla f ( \boldsymbol { x } _ { t } ) , \boldsymbol { e } = \boldsymbol { e } _ { t }$ and $\alpha = \eta _ { t } \leq 1 / ( 2 L )$ . For every � ∈ dom �,

$$
\eta _ { t } ( F ( x _ { t + 1 } ) - F ( u ) ) \leq D ( u , x _ { t } ) - D ( u , x _ { t + 1 } ) + \eta _ { t } \langle e _ { t } , u - x _ { t } \rangle + \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } .\tag{65}
$$

With $x = x _ { t } , e = e _ { t }$ , and $\alpha = \eta _ { t } \leq 1 / L$ , Lemma 3 also gives

$$
\begin{array} { r } { \| { x } _ { t + 1 } - { x } ^ { \star } \| ^ { 2 } \leq \| { x } _ { t } - { x } ^ { \star } \| ^ { 2 } - 2 \eta _ { t } \langle e _ { t } , { x } _ { t } - { x } ^ { \star } \rangle + 2 \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } . } \end{array}
$$

The radius lemma applies to composite objectives and does not require $\nabla f ( x ^ { \star } ) = 0$

To control the final objective value, define

$$
S _ { t } = \sum _ { j = t } ^ { T } \eta _ { j } , \quad S _ { T + 1 } = 0 , \quad \nu _ { t } = \frac { \eta _ { T } } { S _ { t } } , \quad \nu _ { 0 } = \nu _ { 1 } , \quad z _ { 0 } = x ^ { \star } ,
$$

$$
z _ { t } = \frac { \nu _ { t - 1 } } { \nu _ { t } } z _ { t - 1 } + \left( 1 - \frac { \nu _ { t - 1 } } { \nu _ { t } } \right) x _ { t } , \quad P _ { t } = \nu _ { t } D ( z _ { t } , x _ { t + 1 } ) , \quad P _ { 0 } = \nu _ { 0 } D ( x ^ { \star } , x _ { 1 } ) ,
$$

$$
A _ { t } = \eta _ { t } \nu _ { t - 1 } \big ( z _ { t - 1 } - x _ { t } \big ) .
$$

The weights $\nu _ { t }$ increase, so $z _ { t } \in$ dom �. Applying Lemma 4 to (65) with $\rho _ { t } = \eta _ { t } ^ { 2 } \lVert e _ { t } \rVert ^ { 2 }$ , we obtain, for every $1 \leq s \leq T$

$$
P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } + S _ { s + 1 } \sum _ { j = 2 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } \leq P _ { 0 } + \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } \| e _ { t } \| ^ { 2 } + \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle .
$$

Since $S _ { s + 1 } \geq 0 , \nu _ { j } - \nu _ { j - 1 } \geq 0$ , and $\Delta _ { j } \geq 0$ , we can drop the sum on the left. Enlarging the nonnegative squared-error term gives

$$
P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } \leq P _ { 0 } + 2 \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } \Vert e _ { t } \Vert ^ { 2 } + \sum _ { t = 1 } ^ { s } \langle e _ { t } , A _ { t } \rangle .\tag{66}
$$

For $k = T - t + 1$ , the linear schedule gives $\begin{array} { r } { S _ { t } = ( \eta / T ) \sum _ { i = 1 } ^ { k } j = \eta k ( k + 1 ) / ( 2 T ) } \end{array}$ . Hence

$$
\begin{array} { r l r } & { \displaystyle \nu _ { t } = \frac { 2 } { k ( k + 1 ) } , \qquad P _ { 0 } \le \frac { R ^ { 2 } } { T ( T + 1 ) } \le \frac { R ^ { 2 } } { T ^ { 2 } } , } & \\ & { \displaystyle 2 \eta _ { t } ^ { 2 } \nu _ { t } = \frac { 4 \eta ^ { 2 } } { T ^ { 2 } } \frac { k } { k + 1 } \le \frac { 4 \eta ^ { 2 } } { T ^ { 2 } } , \qquad \eta _ { t } \sqrt { \nu _ { t - 1 } } \le \frac { \sqrt { 2 } \eta } { T } . } & \end{array}\tag{67}
$$

For the last inequality, when $t \geq 2$ its left-hand side is $\sqrt { 2 } \eta k / ( T \sqrt { ( k + 1 ) ( k + 2 ) } )$ ; when $t = 1$ , it is $\eta { \sqrt { 2 / ( T ( T + 1 ) ) } }$ . Both expressions have the stated bound, including when $T = 1$ . Also, $\nu _ { T } = 1$ and $\eta _ { T } = \eta / T$

Let � be the first $t \in \{ 1 , \ldots , T + 1 \}$ } for which $\| x _ { t } - x ^ { \star } \| > 2 R$ or $P _ { t - 1 } > 4 R ^ { 2 } / T ^ { 2 } ;$ ; set $\tau = \infty$ if neither condition occurs. Initially, $\| x _ { 1 } - x ^ { \star } \| \leq R$ and $P _ { 0 } \leq R ^ { 2 } / T ^ { 2 }$ , so $\tau > 1$ . Since $x _ { t }$ and $P _ { t - 1 }$ are known before query $t , I _ { t } = \mathbf { 1 } _ { \{ t < \tau \} }$ is $\mathscr { F } _ { t - 1 }$ -measurable. On $\{ t < \tau \}$ ,

$$
\begin{array} { r l } & { \| \nabla f ( x _ { t } ) - c \| \leq \| \nabla f ( x _ { t } ) - \nabla f ( x _ { 1 } ) \| + \| \nabla f ( x _ { 1 } ) - c \| } \\ & { \qquad \leq L \| x _ { t } - x _ { 1 } \| + 2 4 \sigma } \\ & { \qquad \leq L \big ( \| x _ { t } - x ^ { \star } \| + \| x _ { 1 } - x ^ { \star } \| \big ) + 2 4 \sigma \leq 3 L R + 2 4 \sigma \leq \lambda / 2 . } \end{array}
$$

Conditionally on $\mathscr { F } _ { t - 1 }$ , the shifted sample $g _ { t } - c$ has mean $\nabla f ( x _ { t } ) - c$ and centered noise $g _ { t } - \nabla f ( x _ { t } )$ We can therefore apply Lemma 6 on the predictable event $\{ t < \tau \}$ to obtain

$$
\begin{array} { r l r } & { } & { I _ { t } \| e _ { t } \| \le \frac { 3 } { 2 } \lambda , \qquad I _ { t } \| b _ { t } \| \le B , \qquad { \mathbb E } _ { t } [ I _ { t } \| e _ { t } \| ^ { 2 } ] \le I _ { t } V , } \\ & { } & { I _ { t } \| \xi _ { t } \| \le 2 \lambda \le 3 \lambda , \qquad { \mathbb E } _ { t } [ I _ { t } \| \xi _ { t } \| ^ { 2 } ] \le I _ { t } V . \qquad } \end{array}
$$

Moreover, $A _ { t }$ is $\mathscr { F } _ { t - 1 }$ -measurable, and

$$
\begin{array} { l } { \displaystyle I _ { t } \| A _ { t } \| = I _ { t } \eta _ { t } \sqrt { \nu _ { t - 1 } } \sqrt { 2 P _ { t - 1 } } } \\ { \displaystyle \leq \frac { \sqrt { 2 } \eta } { T } \sqrt { \frac { 8 R ^ { 2 } } { T ^ { 2 } } } = \frac { 4 \eta R } { T ^ { 2 } } . } \end{array}\tag{68}
$$

All these bounds apply to stopped variables; no bound on the full trajectory has yet been assumed.

## D.3 Concentration and completion of the proof

The variables $I _ { t } \| e _ { t } \| ^ { 2 }$ lie in $[ 0 , 3 \lambda ^ { 2 } ]$ , and their conditional means sum to at most ��. Lemma 7 with exponent ℓ gives, with failure probability at most $e ^ { - \ell }$

$$
Q : = \sum _ { t = 1 } ^ { T } I _ { t } \Vert e _ { t } \Vert ^ { 2 } \leq 2 V T + 3 \lambda ^ { 2 } \ell \leq 4 \lambda ^ { 2 } \ell + 3 \lambda ^ { 2 } \ell = 7 \lambda ^ { 2 } \ell .\tag{69}
$$

Every prefix of this sum is at most $Q$ because its terms are nonnegative.

For the distance recursion, define $\begin{array} { r } { M _ { s } ^ { r } = - 2 \sum _ { t = 1 } ^ { s } I _ { t } \eta _ { t } \langle \xi _ { t } , x _ { t } - x ^ { \star } \rangle } \end{array}$ , with $M _ { 0 } ^ { r } = 0$ . Predictability and $\mathbb { E } _ { t } { \xi } _ { t } = 0$ show that $\mathbb { E } _ { t } \big ( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } \big ) = 0$ . Its increments and conditional second moments satisfy

$$
\begin{array} { c } { \displaystyle | M _ { t } ^ { r } - M _ { t - 1 } ^ { r } | \leq 2 I _ { t } \eta _ { t } \| \xi _ { t } \| \| x _ { t } - x ^ { \star } \| \leq 1 2 \eta R \lambda = \displaystyle \frac { 1 2 R ^ { 2 } } { 1 2 8 \ell } , } \\ { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \big [ ( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } ) ^ { 2 } \big ] \leq 4 \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \eta _ { t } ^ { 2 } \| x _ { t } - x ^ { \star } \| ^ { 2 } \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } } \\ { \leq 1 6 \eta ^ { 2 } R ^ { 2 } V T \leq 3 2 \eta ^ { 2 } R ^ { 2 } \lambda ^ { 2 } \ell = \displaystyle \frac { 3 2 R ^ { 4 } } { 1 2 8 ^ { 2 } \ell } . } \end{array}
$$

Lemma $^ 8$ consequently yields, with failure probability at most $e ^ { - \ell } .$

$$
\begin{array} { c } { { \displaystyle \operatorname* { m a x } _ { 1 \leq s \leq T } M _ { s } ^ { r } \leq \sqrt { 2 \ell } \frac { 3 2 R ^ { 4 } } { 1 2 8 ^ { 2 } \ell } + 2 \ell \frac { 1 2 R ^ { 2 } } { 1 2 8 \ell } } } \\ { { = \displaystyle \frac { 8 R ^ { 2 } } { 1 2 8 } + \frac { 2 4 R ^ { 2 } } { 1 2 8 } = \frac { R ^ { 2 } } { 4 } . } } \end{array}\tag{70}
$$

The bias term in that recursion has the deterministic bound

$$
\begin{array} { c } { \displaystyle 2 \sum _ { t = 1 } ^ { T } I _ { t } \eta _ { t } \vert \langle b _ { t } , x _ { t } - x ^ { \star } \rangle \vert \leq 4 \eta R B T \leq 8 \eta R \lambda \ell } \\ { \displaystyle = \frac { 8 R ^ { 2 } } { 1 2 8 } = \frac { R ^ { 2 } } { 1 6 } . } \end{array}\tag{71}
$$

For (66), define $\begin{array} { r } { M _ { s } ^ { P } = \sum _ { t = 1 } ^ { s } I _ { t } \langle \xi _ { t } , A _ { t } \rangle } \end{array}$ , with $M _ { 0 } ^ { P } = 0$ . Since $I _ { t } A _ { t }$ is predictable, E $_ t ( M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ) = 0$ By (68),

$$
| M _ { t } ^ { P } - M _ { t - 1 } ^ { P } | \leq I _ { t } \Vert \xi _ { t } \Vert \Vert A _ { t } \Vert \leq \frac { 1 2 \eta R \lambda } { T ^ { 2 } } = \frac { 1 2 R ^ { 2 } } { 1 2 8 \ell T ^ { 2 } } ,
$$

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } [ ( M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ) ^ { 2 } ] \le \sum _ { t = 1 } ^ { T } I _ { t } \| A _ { t } \| ^ { 2 } \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } } } \\ & { } & { \le \frac { 1 6 \eta ^ { 2 } R ^ { 2 } V T } { T ^ { 4 } } \le \frac { 3 2 \eta ^ { 2 } R ^ { 2 } \lambda ^ { 2 } \ell } { T ^ { 4 } } = \frac { 3 2 R ^ { 4 } } { 1 2 8 ^ { 2 } \ell T ^ { 4 } } . } \end{array}
$$

A second application of Lemma $^ 8$ gives, with failure probability at most $e ^ { - \ell }$

$$
\begin{array} { r l r } & { \displaystyle \operatorname* { m a x } _ { 1 \leq s \leq T } M _ { s } ^ { P } \leq \sqrt { 2 \ell \frac { 3 2 R ^ { 4 } } { 1 2 8 ^ { 2 } \ell T ^ { 4 } } } + 2 \ell \frac { 1 2 R ^ { 2 } } { 1 2 8 \ell T ^ { 2 } } } & \\ & { } & { = \displaystyle \frac { 8 R ^ { 2 } } { 1 2 8 T ^ { 2 } } + \frac { 2 4 R ^ { 2 } } { 1 2 8 T ^ { 2 } } = \frac { R ^ { 2 } } { 4 T ^ { 2 } } . \quad } \end{array}\tag{72}
$$

The bias contribution to (66) satisfies

$$
\begin{array} { r } { \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } | \langle b _ { t } , A _ { t } \rangle | \leq \frac { 4 \eta R B T } { T ^ { 2 } } \leq \frac { 8 \eta R \lambda \ell } { T ^ { 2 } } } \\ { = \frac { 8 R ^ { 2 } } { 1 2 8 T ^ { 2 } } = \frac { R ^ { 2 } } { 1 6 T ^ { 2 } } . } \end{array}\tag{73}
$$

Let E be the intersection of the events in (69), (70), and (72). For every fixed pilot history under consideration, $\mathbb { P } ( \mathcal { E } ^ { c } \mid \mathcal { H } _ { m } ) \le 3 e ^ { - \ell }$ . On E, fix $1 \leq s \leq T$ with $s < \tau$ . Then $I _ { t } = 1$ for $1 \leq t \leq s$ . Sum the distance recursion and write $e _ { t } = \xi _ { t } + b _ { t }$ to obtain

$$
\begin{array} { r l r } {  { \| \boldsymbol { x } _ { s + 1 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } \leq R ^ { 2 } + M _ { s } ^ { r } - 2 \displaystyle \sum _ { t = 1 } ^ { s } I _ { t } \eta _ { t } \langle b _ { t } , \boldsymbol { x } _ { t } - \boldsymbol { x } ^ { \star } \rangle + 2 \sum _ { t = 1 } ^ { s } I _ { t } \eta _ { t } ^ { 2 } \| \boldsymbol { e } _ { t } \| ^ { 2 } } } \\ & { } & { \leq R ^ { 2 } + \frac { R ^ { 2 } } { 4 } + \frac { R ^ { 2 } } { 1 6 } + 2 \eta ^ { 2 } Q } \\ & { } & { \leq ( \frac { 2 1 } { 1 6 } + \frac { 1 4 } { 1 2 8 ^ { 2 } \ell } ) R ^ { 2 } < 2 R ^ { 2 } . } \end{array}\tag{74}
$$

For the potential, (66) and (67) give

$$
\begin{array} { r l r } {  { P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } \leq P _ { 0 } + M _ { s } ^ { P } + \sum _ { t = 1 } ^ { s } I _ { t } \langle b _ { t } , A _ { t } \rangle + 2 \sum _ { t = 1 } ^ { s } I _ { t } \eta _ { t } ^ { 2 } \nu _ { t } \| e _ { t } \| ^ { 2 } } } \\ & { } & \\ & { } & { \leq \frac { R ^ { 2 } } { T ^ { 2 } } + \frac { R ^ { 2 } } { 4 T ^ { 2 } } + \frac { R ^ { 2 } } { 1 6 T ^ { 2 } } + \frac { 4 \eta ^ { 2 } Q } { T ^ { 2 } } } \\ & { } & { \leq ( \frac { 2 1 } { 1 6 } + \frac { 2 8 } { 1 2 8 ^ { 2 } \ell } ) \frac { R ^ { 2 } } { T ^ { 2 } } < \frac { 2 R ^ { 2 } } { T ^ { 2 } } . \qquad } \end{array}\tag{75}
$$

Both strict inequalities follow from $\ell >$ log $1 6 > 1$ and $2 1 / 1 6 + 2 8 / 1 2 8 ^ { 2 } < 2$

If $\tau \leq T + 1$ , then $s = \tau - 1$ lies in $\{ 1 , \ldots , T \}$ and satisfies $s < \tau$ . The preceding two inequalities would give

$$
\| x _ { \tau } - x ^ { \star } \| ^ { 2 } < 2 R ^ { 2 } < 4 R ^ { 2 } , \qquad P _ { \tau - 1 } \leq P _ { \tau - 1 } + \eta _ { \tau - 1 } \nu _ { \tau - 1 } \Delta _ { \tau } < { \frac { 2 R ^ { 2 } } { T ^ { 2 } } } < { \frac { 4 R ^ { 2 } } { T ^ { 2 } } } ,
$$

where $\Delta _ { \tau } \geq 0$ because $x ^ { \star }$ minimizes �. Neither stopping condition could hold at �, a contradiction. Thus $\tau = \infty$ on E, and the claimed distance bound holds for every $1 \leq t \leq T + 1$

Finally, take $s = T$ in (75). Since $P _ { T } \ge 0 , \nu _ { T } = 1$ , and $\eta _ { T } = \eta / T$

$$
\frac { \eta } { T } \big ( F ( x _ { T + 1 } ) - F ^ { \star } \big ) \leq P _ { T } + \eta _ { T } \nu _ { T } \Delta _ { T + 1 } \leq \frac { 2 R ^ { 2 } } { T ^ { 2 } } , \qquad F ( x _ { T + 1 } ) - F ^ { \star } \leq \frac { 2 R ^ { 2 } } { \eta T } = \frac { 2 5 6 R \lambda \ell } { T } .
$$

To remove the conditioning on the pilot, the failure probability of the stated conclusions is at most

$$
\mathbb { P } ( \mathcal { A } ^ { c } ) + \mathbb { E } \big [ \mathbf { 1 } _ { \mathcal { A } } \mathbb { P } ( \mathcal { E } ^ { c } \mid \mathcal { H } _ { m } ) \big ] \leq \frac { \delta } { 2 } + 3 e ^ { - \ell } = \frac { \delta } { 2 } + \frac { 3 \delta } { 1 6 } = \frac { 1 1 \delta } { 1 6 } < \delta .
$$

For $T \geq \ell ,$ , the bound max $\{ a , b \} \leq a + b$ for nonnegative �, � and the definition of � yield

$$
\begin{array} { r l r } {  { \frac { 2 5 6 R \lambda \ell } { T } \le \frac { 2 5 6 R \ell } { T } ( 6 L R + 4 8 \sigma + \sigma ( T / \ell ) ^ { 1 / p } ) } } \\ & { } & \\ & { } & { \qquad = 1 5 3 6 \frac { L R ^ { 2 } \ell } { T } + 1 2 2 8 8 \frac { \sigma R \ell } { T } + 2 5 6 \sigma R ( \ell / T ) ^ { ( p - 1 ) / p } } \\ & { } & \\ & { } & { \qquad \le 1 5 3 6 \frac { L R ^ { 2 } \ell } { T } + 1 2 5 4 4 \sigma R ( \ell / T ) ^ { ( p - 1 ) / p } . } \end{array}
$$

The last step uses $0 < \ell / T \leq 1$ and $\ell / T \leq ( \ell / T ) ^ { ( p - 1 ) / p }$ . There are � pilot calls and � main-stage calls, so $N = T + m$ . If $N \geq 2 m$ and $T = N - m$ , then $T \geq N / 2$ and $T \geq m \geq \ell .$ . In particular, the preceding bound implies

$$
F ( x _ { T + 1 } ) - F ^ { \star } \le 3 0 7 2 \frac { L R ^ { 2 } \ell } { N } + 1 2 5 4 4 \cdot 2 ^ { ( p - 1 ) / p } \sigma R ( \ell / N ) ^ { ( p - 1 ) / p } ,
$$

which has the stated order in the total number of oracle calls.

□

## E Proof of Theorem 4

## E.1 The clean error and the replacement error

Recall that $\ell = \log ( 8 / \delta )$ and $\Gamma = \ell + q$ . Write $K = 1 2 8$ , and retain the notation

$$
\eta = \frac { R } { K \lambda \Gamma } , \qquad \eta _ { t } = \eta \frac { T - t + 1 } { T } , \qquad B = 2 \sigma ^ { p } \lambda ^ { 1 - p } , \qquad V = 2 \sigma ^ { p } \lambda ^ { 2 - p } .
$$

The choice $\lambda = \operatorname* { m a x } \{ 4 L R , \sigma ( T / \Gamma ) ^ { 1 / p } \}$ gives

$$
\begin{array} { c } { { \sigma ^ { p } T \le \lambda ^ { p } \Gamma , } } \\ { { B T = 2 \sigma ^ { p } T \lambda ^ { 1 - p } \le 2 \lambda ^ { p } \Gamma \lambda ^ { 1 - p } = 2 \lambda \Gamma , } } \\ { { V T = 2 \sigma ^ { p } T \lambda ^ { 2 - p } \le 2 \lambda ^ { p } \Gamma \lambda ^ { 2 - p } = 2 \lambda ^ { 2 } \Gamma , } } \\ { { \eta _ { t } \le \eta = \displaystyle \frac { R } { K \lambda \Gamma } \le \displaystyle \frac { R } { K ( 4 L R ) \Gamma } = \displaystyle \frac { 1 } { 4 K L \Gamma } \le \displaystyle \frac { 1 } { 2 L } . } } \end{array}\tag{76}
$$

The last inequality uses $K = 1 2 8$ and $\Gamma \geq \ell > \log 8$

Use $S _ { t } , \nu _ { t } , z _ { t } , P _ { t }$ from (25), with $f ^ { \star } = f ( x ^ { \star } )$ and $\Delta _ { t } = f ( x _ { t } ) - f ^ { \star }$ . Define

$$
\tau = \operatorname* { m i n } \left\{ t \in \{ 1 , \ldots , T + 1 \} : \| x _ { t } - x ^ { \star } \| > 2 R \mathrm { o r } P _ { t - 1 } > { \frac { 4 R ^ { 2 } } { T ^ { 2 } } } \right\} , \qquad I _ { t } = \mathbf { 1 } _ { \{ t < \tau \} } ,
$$

where min $\varnothing = \infty$ . The filtration contains the algorithm’s randomness and all preceding latent clean samples and adversarial choices. Thus $x _ { t } , P _ { t - 1 } , I _ { t }$ are $\mathscr { F } _ { t - 1 }$ -measurable. Also,

$$
\| x _ { 1 } - x ^ { \star } \| \leq R < 2 R , \qquad P _ { 0 } \leq \frac { R ^ { 2 } } { T ( T + 1 ) } < \frac { 4 R ^ { 2 } } { T ^ { 2 } } , \qquad \mathrm { s o ~ } \tau > 1 .
$$

Put $h _ { t } = \nabla f ( x _ { t } )$ . We separate the error of the latent clean sample from the change made by the adversary:

$$
\begin{array} { l l } { { u _ { t } = \mathrm { c l i p } ( g _ { t } , \lambda ) - h _ { t } , } } & { { d _ { t } = \mathrm { c l i p } ( \widetilde { g } _ { t } , \lambda ) - \mathrm { c l i p } ( g _ { t } , \lambda ) , } } \\ { { b _ { t } = \mathbb { E } _ { t } u _ { t } , } } & { { \xi _ { t } = u _ { t } - b _ { t } , } } \\ { { e _ { t } = \mathrm { c l i p } ( \widetilde { g } _ { t } , \lambda ) - h _ { t } = u _ { t } + d _ { t } = b _ { t } + \xi _ { t } + d _ { t } . } } & { { } } \end{array}\tag{77}
$$

Here $\mathbb { E } _ { t } \xi _ { t } = \mathbb { E } _ { t } u _ { t } - b _ { t } = 0$ . We center the latent clean sample before deciding whether it is replaced. In particular, no unbiasedness assumption is made after conditioning on the event $C _ { t } = 0$

On $\{ I _ { t } = 1 \}$ , smoothness and $\nabla f ( x ^ { \star } ) = 0$ imply

$$
\| h _ { t } \| = \| \nabla f ( x _ { t } ) - \nabla f ( x ^ { \star } ) \| \leq L \| x _ { t } - x ^ { \star } \| \leq 2 L R \leq \lambda / 2 .
$$

Lemma $^ { 6 , }$ applied conditionally on $\mathscr { F } _ { t - 1 }$ , therefore gives

$$
\begin{array} { r l } & { \| u _ { t } \| \leq 3 \lambda / 2 , \qquad \| b _ { t } \| \leq B , \qquad \mathbb { E } _ { t } \| u _ { t } \| ^ { 2 } \leq V , } \\ & { \| \xi _ { t } \| \leq 2 \lambda , \qquad \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } \leq V \qquad \mathrm { o n ~ } \{ I _ { t } = 1 \} . } \end{array}\tag{78}
$$

On the same event, both clipped vectors have norm at most $\lambda .$ , so

$$
\begin{array} { r l } & { \| e _ { t } \| \leq \| \mathrm { c l i p } ( \widetilde { g } _ { t } , \lambda ) \| + \| h _ { t } \| \leq 3 \lambda / 2 , } \\ & { \| d _ { t } \| \leq \big ( \| \mathrm { c l i p } ( \widetilde { g } _ { t } , \lambda ) \| + \| \mathrm { c l i p } ( g _ { t } , \lambda ) \| \big ) C _ { t } \leq 2 \lambda C _ { t } . } \end{array}
$$

The second bound also holds without stopping. If $C _ { t } ~ = ~ 0$ , then $e _ { t } ~ = { \boldsymbol { u } } _ { t }$ ; if $C _ { t } ~ = ~ 1$ and $I _ { t } ~ = ~ 1$ , then $\| e _ { t } \| ^ { 2 } \leq 9 \lambda ^ { 2 } / 4$ . Consequently,

$$
\| d _ { t } \| \leq 2 \lambda C _ { t } , \qquad I _ { t } \| e _ { t } \| ^ { 2 } \leq I _ { t } \| u _ { t } \| ^ { 2 } + 3 \lambda ^ { 2 } I _ { t } C _ { t } .\tag{79}
$$

These inequalities hold on every sample path, although $C _ { t }$ may depend on the current clean sample.

## E.2 Bounds for the three random sums

Apply Lemma 7 to the sequence $I _ { t } \| u _ { t } \| ^ { 2 }$ , with $H = 3 \lambda ^ { 2 } , m _ { t } = I _ { t } V$ , and $u = \ell .$ . Its assumptions follow from

$$
0 \le I _ { t } \Vert u _ { t } \Vert ^ { 2 } \le 3 \lambda ^ { 2 } , \qquad \mathbb { E } _ { t } [ I _ { t } \Vert u _ { t } \Vert ^ { 2 } ] = I _ { t } \mathbb { E } _ { t } \Vert u _ { t } \Vert ^ { 2 } \le I _ { t } V , \qquad \sum _ { t = 1 } ^ { T } I _ { t } V \le V T .
$$

Thus, on an event $E _ { Q }$ with $\mathbb { P } ( E _ { O } ) \ge 1 - e ^ { - \ell }$

$$
\begin{array} { l } { \displaystyle Q _ { e } : = \sum _ { t = 1 } ^ { T } I _ { t } \| e _ { t } \| ^ { 2 } \leq \sum _ { t = 1 } ^ { T } I _ { t } \| u _ { t } \| ^ { 2 } + 3 \lambda ^ { 2 } \sum _ { t = 1 } ^ { T } I _ { t } C _ { t } } \\ { \leq 2 V T + 3 \lambda ^ { 2 } \ell + 3 \lambda ^ { 2 } q } \\ { \leq 4 \lambda ^ { 2 } \Gamma + 3 \lambda ^ { 2 } ( \ell + q ) = 7 \lambda ^ { 2 } \Gamma . } \end{array}\tag{80}
$$

Every summand in $Q _ { e }$ is nonnegative, so this also bounds each prefix. The bias and the replacements have the pathwise bound

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } I _ { t } ( \| b _ { t } \| + \| d _ { t } \| ) \leq B T + 2 \lambda \sum _ { t = 1 } ^ { T } I _ { t } C _ { t } } } \\ & { } & { \leq 2 \lambda \Gamma + 2 \lambda q \leq 4 \lambda \Gamma . } \end{array}\tag{81}
$$

For the distance to $x ^ { \star }$ , define the martingale

$$
M _ { s } ^ { r } = - 2 \sum _ { t = 1 } ^ { s } I _ { t } \eta _ { t } \langle \xi _ { t } , x _ { t } - x ^ { \star } \rangle , \qquad M _ { 0 } ^ { r } = 0 .
$$

Predictability and $\mathbb { E } _ { t } { \xi } _ { t } = 0$ give

$$
\mathbb { E } _ { t } \left( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } \right) = - 2 I _ { t } \eta _ { t } \left. \mathbb { E } _ { t } \xi _ { t } , x _ { t } - x ^ { \star } \right. = 0 .
$$

Using $\| \xi _ { t } \| \le 2 \lambda \le 3 \lambda$ on $\{ I _ { t } = 1 \}$ , we obtain

$$
\begin{array} { c } { { | M _ { t } ^ { r } - M _ { t - 1 } ^ { r } | \le 2 I _ { t } \eta _ { t } \| \xi _ { t } \| \| x _ { t } - x ^ { \star } \| \le 2 \eta ( 3 \lambda ) ( 2 R ) = 1 2 \eta R \lambda , } } \\ { { \mathbb { E } _ { t } ( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } ) ^ { 2 } = 4 I _ { t } \eta _ { t } ^ { 2 } \mathbb { E } _ { t } \left[ \langle \xi _ { t } , x _ { t } - x ^ { \star } \rangle ^ { 2 } \right] } } \\ { { \le 4 I _ { t } \eta _ { t } ^ { 2 } \| x _ { t } - x ^ { \star } \| ^ { 2 } \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } \le 1 6 \eta ^ { 2 } R ^ { 2 } V , } } \\ { { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } ( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } ) ^ { 2 } \le 1 6 \eta ^ { 2 } R ^ { 2 } V T \le 3 2 \eta ^ { 2 } R ^ { 2 } \lambda ^ { 2 } \Gamma . } } \end{array}
$$

Lemma 8 with $u = \ell$ gives an event $E _ { r }$ , of probability at least $1 - e ^ { - \ell }$ , on which

$$
\begin{array} { l } { \displaystyle { \operatorname* { m a x } _ { s \leq T } M _ { s } ^ { r } \leq \sqrt { 3 2 \eta ^ { 2 } R ^ { 2 } V T \ell } + 2 4 \eta R \lambda \ell } } \\ { \displaystyle { \quad \leq \sqrt { 6 4 \eta ^ { 2 } R ^ { 2 } \lambda ^ { 2 } \Gamma \ell } + 2 4 \eta R \lambda \ell } } \\ { \displaystyle { \quad = 8 \eta R \lambda \sqrt { \Gamma \ell } + 2 4 \eta R \lambda \ell } } \\ { \displaystyle { \quad = \frac { R ^ { 2 } } { K } \left( 8 \sqrt { \frac { \ell } { \Gamma } } + 2 4 \frac { \ell } { \Gamma } \right) \leq \frac { 3 2 R ^ { 2 } } { K } . } } \end{array}\tag{82}
$$

The last two steps use $\eta = R / ( K \lambda \Gamma )$ and $\ell \leq \Gamma$

For the comparison potential, let $A _ { t } = \eta _ { t } \nu _ { t - 1 } \bigl ( z _ { t - 1 } - x _ { t } \bigr )$ , as in Lemma 4. This vector is $\mathcal { F } _ { t - 1 } { \mathrm { - m e a s u r a b l e } }$ On $\{ I _ { t } = 1 \}$ , the definition $P _ { t - 1 } = \nu _ { t - 1 } \| z _ { t - 1 } - x _ { t } \| ^ { 2 } / 2$ and Lemma 5 give

$$
\begin{array} { l } { \displaystyle \| A _ { t } \| = \eta _ { t } \nu _ { t - 1 } \| z _ { t - 1 } - x _ { t } \| = \eta _ { t } \sqrt { \nu _ { t - 1 } } \sqrt { 2 P _ { t - 1 } } } \\ { \displaystyle \leq \frac { \sqrt { 2 } \eta } { T } \sqrt { 2 \cdot \frac { 4 R ^ { 2 } } { T ^ { 2 } } } = \frac { 4 \eta R } { T ^ { 2 } } . } \end{array}\tag{83}
$$

Thus $\begin{array} { r } { M _ { s } ^ { P } = \sum _ { t = 1 } ^ { s } I _ { t } \langle \xi _ { t } , A _ { t } \rangle } \end{array}$ , with $M _ { 0 } ^ { P } = 0$ , satisfies

$$
\begin{array} { r l r } {  { \mathbb { E } _ { t } ( M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ) = I _ { t } \langle \mathbb { E } _ { t } \xi _ { t } , A _ { t } \rangle = 0 , } } \\ & { } & { \quad \quad | M _ { t } ^ { P } - M _ { t - 1 } ^ { P } | \le I _ { t } \| \xi _ { t } \| \| A _ { t } \| \le ( 3 \lambda ) \frac { 4 \eta R } { T ^ { 2 } } = \frac { 1 2 \eta R \lambda } { T ^ { 2 } } , } \\ & { } & { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } ( M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ) ^ { 2 } \le \sum _ { t = 1 } ^ { T } I _ { t } \| A _ { t } \| ^ { 2 } \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } } \\ & { } & { \quad \quad \le \frac { 1 6 \eta ^ { 2 } R ^ { 2 } V T } { T ^ { 4 } } \le \frac { 3 2 \eta ^ { 2 } R ^ { 2 } \lambda ^ { 2 } \Gamma } { T ^ { 4 } } . \quad \quad } \end{array}
$$

Applying Lemma $^ 8$ again gives an event $E _ { P }$ , of probability at least $1 - e ^ { - \ell }$ , on which

$$
\begin{array} { r l r } {  { \operatorname* { m a x } M _ { s } ^ { P } \le \sqrt { \frac { 3 2 \eta ^ { 2 } R ^ { 2 } V T \ell } { T ^ { 4 } } } + \frac { 2 4 \eta R \lambda \ell } { T ^ { 2 } } } } \\ & { } & \\ & { } & { \le \frac { 8 \eta R \lambda \sqrt { \Gamma \ell } + 2 4 \eta R \lambda \ell } { T ^ { 2 } } } \\ & { } & { = \frac { R ^ { 2 } } { K T ^ { 2 } } ( 8 \sqrt { \frac { \ell } { \Gamma } } + 2 4 \frac { \ell } { \Gamma } ) \le \frac { 3 2 R ^ { 2 } } { K T ^ { 2 } } . } \end{array}\tag{84}
$$

The remaining inner products require no concentration inequality. By Cauchy–Schwarz and (81),

$$
\begin{array} { r l r } {  { 2 \sum _ { t = 1 } ^ { T } I _ { t } \eta _ { t } | \langle b _ { t } + d _ { t } , x _ { t } - x ^ { \star } \rangle | \le 2 \eta ( 2 R ) \sum _ { t = 1 } ^ { T } I _ { t } ( \| b _ { t } \| + \| d _ { t } \| ) } } \\ & { } & { \le 4 \eta R ( 4 \lambda \Gamma ) = \frac { 1 6 R ^ { 2 } } { K } , } \\ & { } & { \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } | \langle b _ { t } + d _ { t } , A _ { t } \rangle | \le \frac { 4 \eta R } { T ^ { 2 } } \sum _ { t = 1 } ^ { T } I _ { t } ( \| b _ { t } \| + \| d _ { t } \| ) } \\ & { } & { \le \frac { 4 \eta R } { T ^ { 2 } } ( 4 \lambda \Gamma ) = \frac { 1 6 R ^ { 2 } } { K T ^ { 2 } } . } \end{array}\tag{85}
$$

(86)

## E.3 Removing the stopping time and bounding the last iterate

Let $E = E _ { Q } \cap E _ { r } \cap E _ { P }$ . The union bound gives

$$
\mathbb { P } ( E ) \ge 1 - \mathbb { P } ( E _ { Q } ^ { c } ) - \mathbb { P } ( E _ { r } ^ { c } ) - \mathbb { P } ( E _ { P } ^ { c } ) \ge 1 - 3 e ^ { - \ell } = 1 - \frac { 3 \delta } { 8 } \ge 1 - \delta .
$$

Fix a sample path in � and an integer $1 \leq s \leq T$ with $s < \tau$ . Then $I _ { t } = 1$ for $1 \leq t \leq s$ . Lemma 3 gives

$$
\begin{array} { r l } & { \| x _ { t + 1 } - x ^ { \star } \| ^ { 2 } - \| x _ { t } - x ^ { \star } \| ^ { 2 } \leq - 2 \eta _ { t } \langle e _ { t } , x _ { t } - x ^ { \star } \rangle + 2 \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } } \\ & { \qquad = - 2 \eta _ { t } \langle \xi _ { t } , x _ { t } - x ^ { \star } \rangle - 2 \eta _ { t } \langle b _ { t } + d _ { t } , x _ { t } - x ^ { \star } \rangle + 2 \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } . } \end{array}
$$

Sum these inequalities over $t = 1 , \ldots , s ,$ , and use (80), (82), and (85):

$$
\begin{array} { r l r } {  { \| x _ { s + 1 } - x ^ { \star } \| ^ { 2 } \le \| x _ { 1 } - x ^ { \star } \| ^ { 2 } + M _ { s } ^ { r } - 2 \displaystyle \sum _ { t = 1 } ^ { s } \eta _ { t } \langle b _ { t } + d _ { t } , x _ { t } - x ^ { \star } \rangle + 2 \displaystyle \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \| e _ { t } \| ^ { 2 } } } \\ & { } & { \le R ^ { 2 } + \frac { 3 2 R ^ { 2 } } { K } + \frac { 1 6 R ^ { 2 } } { K } + 2 \eta ^ { 2 } Q _ { e } } \\ & { } & { \le R ^ { 2 } + \frac { 4 8 R ^ { 2 } } { K } + 2 \frac { R ^ { 2 } } { K ^ { 2 } \lambda ^ { 2 } \Gamma ^ { 2 } } ( 7 \lambda ^ { 2 } \Gamma ) } \\ & { } & { = ( 1 + \frac { 4 8 } { K } + \frac { 1 4 } { K ^ { 2 } \Gamma } ) R ^ { 2 } < 2 R ^ { 2 } . } \end{array}\tag{87}
$$

For the objective gap, (21) with $r = 0$ implies (26) with $\rho _ { t } = 2 \eta _ { t } ^ { 2 } \lVert e _ { t } \rVert ^ { 2 } ;$ the factor 2 only enlarges its nonnegative error term. Lemma 4 consequently gives

$$
\begin{array} { r l r } {  { P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } + S _ { s + 1 } \sum _ { j = 2 } ^ { s } ( \nu _ { j } - \nu _ { j - 1 } ) \Delta _ { j } } } \\ & { } & \\ & { } & { \leq P _ { 0 } + 2 \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } \| e _ { t } \| ^ { 2 } + \sum _ { t = 1 } ^ { s } \langle \xi _ { t } + b _ { t } + d _ { t } , A _ { t } \rangle . } \end{array}
$$

The third term on the left is nonnegative. Since $2 \eta _ { t } ^ { 2 } \nu _ { t } \leq 4 \eta ^ { 2 } / T ^ { 2 }$ by Lemma 5, we get

$$
\begin{array} { l } { \displaystyle P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } \leq P _ { 0 } + M _ { s } ^ { P } + \sum _ { t = 1 } ^ { s } \langle b _ { t } + d _ { t } , A _ { t } \rangle + 2 \sum _ { t = 1 } ^ { s } \eta _ { t } ^ { 2 } \nu _ { t } \| e _ { t } \| ^ { 2 } } \\ { \displaystyle \qquad \leq \frac { R ^ { 2 } } { T ^ { 2 } } + \frac { 3 2 R ^ { 2 } } { K T ^ { 2 } } + \frac { 1 6 R ^ { 2 } } { K T ^ { 2 } } + \frac { 4 \eta ^ { 2 } Q _ { e } } { T ^ { 2 } } } \\ { \displaystyle \qquad \leq \frac { R ^ { 2 } } { T ^ { 2 } } + \frac { 4 8 R ^ { 2 } } { K T ^ { 2 } } + \frac { 4 } { T ^ { 2 } } \frac { R ^ { 2 } } { K ^ { 2 } \lambda ^ { 2 } \Gamma ^ { 2 } } ( \eta \lambda ^ { 2 } \Gamma ) } \\ { \displaystyle \qquad = \left( 1 + \frac { 4 8 } { K } + \frac { 2 8 } { K ^ { 2 } \Gamma } \right) \frac { R ^ { 2 } } { T ^ { 2 } } < \frac { 2 R ^ { 2 } } { T ^ { 2 } } . } \end{array}\tag{88}
$$

Both strict inequalities follow from

$$
1 + \frac { 4 8 } { K } + \frac { 2 8 } { K ^ { 2 } \Gamma } < 1 + \frac { 4 8 } { 1 2 8 } + \frac { 2 8 } { 1 2 8 ^ { 2 } \log 8 } < 1 . 3 7 6 < 2 .
$$

If $\tau < \infty$ , then $2 \leq \tau \leq T + 1$ . Applying (87)–(88) with $s = \tau - 1$ yields

$$
\| x _ { \tau } - x ^ { \star } \| ^ { 2 } < 2 R ^ { 2 } < 4 R ^ { 2 } , \qquad P _ { \tau - 1 } \leq P _ { \tau - 1 } + \eta _ { \tau - 1 } \nu _ { \tau - 1 } \Delta _ { \tau } < { \frac { 2 R ^ { 2 } } { T ^ { 2 } } } < { \frac { 4 R ^ { 2 } } { T ^ { 2 } } } .
$$

Neither stopping condition can hold, a contradiction. Therefore $\tau = \infty$ on $E .$ , and $\| x _ { t } - x ^ { \star } \| \leq 2 R$ for $1 \leq t \leq T + 1$

Finally, take $s = T$ in (88). Since $P _ { T } \ge 0 , \nu _ { T } = 1$ , and $\eta _ { T } = \eta / T$

$$
\frac { \eta } { T } \bigl ( f ( x _ { T + 1 } ) - f ^ { \star } \bigr ) \leq P _ { T } + \eta _ { T } \nu _ { T } \Delta _ { T + 1 } \leq \frac { 2 R ^ { 2 } } { T ^ { 2 } } ,
$$

$$
\begin{array} { l } { f ( x _ { T + 1 } ) - f ^ { \star } \le \displaystyle \frac { 2 R ^ { 2 } } { \eta T } = \displaystyle \frac { 2 R ^ { 2 } } { T } \displaystyle \frac { K \lambda \Gamma } { R } = \displaystyle \frac { 2 5 6 R \lambda \Gamma } { T } } \\ { \displaystyle \qquad \le \displaystyle \frac { 2 5 6 R \Gamma } { T } \left( 4 L R + \sigma ( T / \Gamma ) ^ { 1 / p } \right) } \\ { \displaystyle \qquad = 1 0 2 4 L R ^ { 2 } \displaystyle \frac { \Gamma } { T } + 2 5 6 \sigma R \left( \displaystyle \frac { \Gamma } { T } \right) ^ { ( p - 1 ) / p } . } \end{array}
$$

Together with $\mathbb { P } ( E ) \geq 1 - \delta$ and $\Gamma = \ell + q$ , this proves Theorem 4.

## E.4 Proof of Corollary 1

Keep the same filtration, stopping time, and error decomposition. Now $\nu _ { t }$ is predictable, $\textstyle \sum _ { t = 1 } ^ { T } \nu _ { t } \leq S$ almost surely, and $\lambda = \operatorname* { m a x } \{ 4 L R , ( S / \Gamma ) ^ { 1 / p } \}$ . In particular, $S \le \lambda ^ { p } \Gamma$ . The conditional version of Lemma $6$ gives, on $\{ I _ { t } = 1 \}$ },

$$
\| b _ { t } \| \leq B _ { t } : = 2 \nu _ { t } \lambda ^ { 1 - p } , \qquad \mathbb { E } _ { t } \| u _ { t } \| ^ { 2 } \leq V _ { t } : = 2 \nu _ { t } \lambda ^ { 2 - p } , \qquad \mathbb { E } _ { t } \| \xi _ { t } \| ^ { 2 } \leq V _ { t } .
$$

Thus

$$
\sum _ { t = 1 } ^ { T } I _ { t } B _ { t } = 2 \lambda ^ { 1 - p } \sum _ { t = 1 } ^ { T } I _ { t } \nu _ { t } \leq 2 S \lambda ^ { 1 - p } \leq 2 \lambda ^ { p } \Gamma \lambda ^ { 1 - p } = 2 \lambda \Gamma ,
$$

$$
\sum _ { t = 1 } ^ { T } I _ { t } V _ { t } = 2 \lambda ^ { 2 - p } \sum _ { t = 1 } ^ { T } I _ { t } \nu _ { t } \leq 2 S \lambda ^ { 2 - p } \leq 2 \lambda ^ { p } \Gamma \lambda ^ { 2 - p } = 2 \lambda ^ { 2 } \Gamma .
$$

All pointwise clipping bounds are unchanged. Apply Lemma 7 to the sequence $I _ { t } \| u _ { t } \| ^ { 2 }$ with $H = 3 \lambda ^ { 2 }$ $m _ { t } = I _ { t } V _ { t } , A = 2 S \lambda ^ { 2 - p }$ , and $u = \ell$ . Except on an event of probability $e ^ { - \ell }$

$$
\begin{array} { r l r } {  { Q _ { e } \le \sum _ { t = 1 } ^ { T } I _ { t } \| u _ { t } \| ^ { 2 } + 3 \lambda ^ { 2 } q } } \\ & { } & { \le 4 S \lambda ^ { 2 - p } + 3 \lambda ^ { 2 } \ell + 3 \lambda ^ { 2 } q } \\ & { } & { \le 4 \lambda ^ { 2 } \Gamma + 3 \lambda ^ { 2 } ( \ell + q ) = 7 \lambda ^ { 2 } \Gamma . } \end{array}
$$

The other inputs to the proof satisfy

$$
\sum _ { t = 1 } ^ { T } I _ { t } ( \| b _ { t } \| + \| d _ { t } \| ) \leq \sum _ { t = 1 } ^ { T } I _ { t } B _ { t } + 2 \lambda q \leq 2 \lambda \Gamma + 2 \lambda q \leq 4 \lambda \Gamma ,
$$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } ( M _ { t } ^ { r } - M _ { t - 1 } ^ { r } ) ^ { 2 } \leq 1 6 \eta ^ { 2 } R ^ { 2 } \sum _ { t = 1 } ^ { T } I _ { t } V _ { t } \leq 3 2 \eta ^ { 2 } R ^ { 2 } \lambda ^ { 2 } \Gamma ,
$$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } ( M _ { t } ^ { P } - M _ { t - 1 } ^ { P } ) ^ { 2 } \leq \frac { 1 6 \eta ^ { 2 } R ^ { 2 } } { T ^ { 4 } } \sum _ { t = 1 } ^ { T } I _ { t } V _ { t } \leq \frac { 3 2 \eta ^ { 2 } R ^ { 2 } \lambda ^ { 2 } \Gamma } { T ^ { 4 } } .
$$

The increment bounds remain $1 2 \eta R \lambda$ and $1 2 \eta R \lambda / T ^ { 2 }$ . Hence the two applications of Lemma 8 give exactly (82) and (84). The first of the three bounds above also gives (85) and (86).

On the intersection of these three events, the calculations in (87)–(88) therefore yield

$$
\| x _ { s + 1 } - x ^ { \star } \| ^ { 2 } < 2 R ^ { 2 } , \qquad P _ { s } + \eta _ { s } \nu _ { s } \Delta _ { s + 1 } < 2 R ^ { 2 } / T ^ { 2 } \quad ( 1 \leq s \leq T , s < \tau ) .
$$

If $\tau < \infty$ , setting $s = \tau - 1$ contradicts both possible stopping conditions as above. Thus $\tau = \infty$ and, with probability at least $1 - 3 e ^ { - \ell } \ge 1 - \delta$

$$
\begin{array} { r l } & { f ( x _ { T + 1 } ) - f ^ { \star } \le \displaystyle \frac { 2 5 6 R \lambda \Gamma } { T } } \\ & { \qquad \le \displaystyle \frac { 2 5 6 R \Gamma } { T } \left( 4 L R + ( S / \Gamma ) ^ { 1 / p } \right) } \\ & { \qquad = 1 0 2 4 \displaystyle \frac { L R ^ { 2 } \Gamma } { T } + 2 5 6 \displaystyle \frac { R S ^ { 1 / p } \Gamma ^ { ( p - 1 ) / p } } { T } . } \end{array}
$$

The almost-sure bound $\textstyle \sum _ { t } \nu _ { t } \leq S$ supplies the deterministic variance bounds required by the concentration lemmas. □

## F Proofs of the statistical lower bounds

Each proof constructs two instances indexed by $s \in \{ - 1 , + 1 \}$ . To compare their outputs, run the algorithm on both instances with the same internal random seed and the same Bernoulli variables. Each run still has the distribution required by its own instance. We will identify an event on which the observed transcripts agree, and then show that a common output must have a large error on at least one instance.

## F.1 Proof of Proposition 1

Let $a = ( p - 1 ) / p$ and $b = \log ( 1 / ( 4 \delta ) )$ . Choose

$$
\rho = \frac { b } { 2 T } , \qquad m = \operatorname* { m i n } \left\{ L R , \frac { \sigma \rho ^ { a } } { 2 ^ { 1 / p } } \right\} = \operatorname* { m i n } \left\{ L R , \frac { \sigma } { 2 } ( b / T ) ^ { a } \right\} .\tag{89}
$$

Indeed, $2 ^ { - 1 / p } \rho ^ { a } = 2 ^ { - 1 / p - a } ( b / T ) ^ { a } = ( b / T ) ^ { a } / 2$ . The assumptions $\delta \leq 1 / 8$ and $T \geq b$ imply

$$
b \geq \log 2 > 0 , \qquad 0 < \rho = \frac { b } { 2 T } \leq \frac 1 2 , \qquad m > 0 .
$$

For $s \in \{ - 1 , + 1 \}$ , define

$$
f _ { s } ( x ) = \frac { m } { 2 R } ( x - s R ) ^ { 2 } , \qquad x \in \mathbb { R } .
$$

These functions satisfy

$$
\begin{array} { c } { { f _ { s } ^ { \prime } ( x ) = \displaystyle { \frac { m } { R } } ( x - s R ) = \displaystyle { \frac { m } { R } } x - s m , \qquad f _ { s } ^ { \prime \prime } ( x ) = \displaystyle { \frac { m } { R } } > 0 , } } \\ { { | f _ { s } ^ { \prime } ( x ) - f _ { s } ^ { \prime } ( y ) | = \displaystyle { \frac { m } { R } } | x - y | \leq L | x - y | , } } \\ { { x _ { s } ^ { \star } = s R , \qquad f _ { s } ( x _ { s } ^ { \star } ) = 0 , \qquad | x _ { 1 } - x _ { s } ^ { \star } | = | 0 - s R | = R . } } \end{array}
$$

Thus both instances are convex and �-smooth with the required initial distance.

At the �-th query draw a fresh $B _ { t } \sim \mathrm { B e r n o u l l i } ( \rho )$ , independently of all preceding samples and the algorithm’s random seed. Return

$$
g _ { s , t } ( x ) = \frac { m } { R } x - \frac { s m } { \rho } B _ { t } .\tag{90}
$$

For an adaptive query $x _ { t }$ , include the random seed in $\mathcal { F } _ { t - 1 } ;$ then $x _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable and $\mathbb { E } _ { t } B _ { t } = \rho$ Therefore

$$
\begin{array} { c } { \displaystyle \mathbb { E } _ { t } g _ { s , t } ( x _ { t } ) = \frac { m } { R } x _ { t } - \frac { s m } { \rho } \mathbb { E } _ { t } B _ { t } = \frac { m } { R } x _ { t } - s m = f _ { s } ^ { \prime } ( x _ { t } ) , } \\ { g _ { s , t } ( x _ { t } ) - f _ { s } ^ { \prime } ( x _ { t } ) = s m \left( 1 - \frac { B _ { t } } { \rho } \right) . } \end{array}
$$

Conditionally on $\mathcal { F } _ { t - 1 }$ , the magnitude of the last expression is � with probability $1 - \rho _ { \mathrm { \ell } }$ , and $m ( 1 - \rho ) / \rho$ with probability $\rho$ . Hence

$$
\begin{array} { r l } & { \mathbb { E } _ { t } | g _ { s , t } ( x _ { t } ) - f _ { s } ^ { \prime } ( x _ { t } ) | ^ { p } = m ^ { p } \left[ ( 1 - \rho ) + \rho \left( \displaystyle \frac { 1 - \rho } { \rho } \right) ^ { p } \right] } \\ & { \qquad = m ^ { p } \left[ ( 1 - \rho ) + \rho ^ { 1 - p } ( 1 - \rho ) ^ { p } \right] } \\ & { \qquad \le m ^ { p } \left[ \rho ^ { 1 - p } + \rho ^ { 1 - p } \right] = 2 m ^ { p } \rho ^ { 1 - p } } \\ & { \qquad \le 2 \left( \displaystyle \frac { \sigma \rho ^ { a } } { 2 ^ { 1 / p } } \right) ^ { p } \rho ^ { 1 - p } = \sigma ^ { p } \rho ^ { a p + 1 - p } = \sigma ^ { p } . } \end{array}\tag{91}
$$

We used $1 - \rho \leq 1 \leq \rho ^ { 1 - p } , ( 1 - \rho ) ^ { p } \leq 1$ , and $a p = p - 1$ . The bound does not depend on the query location.

Consider the event $E _ { 0 } = \left\{ B _ { 1 } = \cdots = B _ { T } = 0 \right\}$ . For $0 \leq u \leq 1 / 2$

$$
{ \frac { d } { d u } } { \big ( } \log ( 1 - u ) + 2 u { \big ) } = - { \frac { 1 } { 1 - u } } + 2 = { \frac { 1 - 2 u } { 1 - u } } \geq 0 .
$$

Since $\log ( 1 - 0 ) + 2 \cdot 0 = 0$ , this proves $\log ( 1 - u ) \geq - 2 u$ . Independence of the Bernoulli variables now gives

$$
\begin{array} { r l } & { \mathbb { P } ( E _ { 0 } ) = \displaystyle \prod _ { t = 1 } ^ { T } \mathbb { P } ( B _ { t } = 0 ) = ( 1 - \rho ) ^ { T } = \exp \bigl ( T \log ( 1 - \rho ) \bigr ) } \\ & { \qquad \geq \exp ( - 2 T \rho ) = \exp ( - b ) = 4 \delta . } \end{array}\tag{92}
$$

On $E _ { 0 } ,$ the answer to every query is $( m / R ) x .$ , for either sign. Fixing the common random seed, the first queries agree. If the transcripts through round $t - 1$ agree, the algorithm selects the same next query; the responses $( m / R ) x _ { t }$ then agree as well. This induction also gives the same stopping decision and output. If the algorithm uses fewer than $T$ queries, generate unused Bernoulli variables up to $T ;$ this leaves its output unchanged and keeps (92) valid.

Let $\widehat { x } _ { s }$ denote the output on instance �, and let $\widehat { x }$ be their common value on $E _ { 0 }$ . At that value,

$$
\begin{array} { l } { { \displaystyle \left( f _ { + } ( \widehat x ) - f _ { + } ( R ) \right) + \left( f _ { - } ( \widehat x ) - f _ { - } ( - R ) \right) } \ ~ } \\ { { \displaystyle = \frac { m } { 2 R } \left( ( \widehat x - R ) ^ { 2 } + ( \widehat x + R ) ^ { 2 } \right) = \frac { m } { R } ( \widehat x ^ { 2 } + R ^ { 2 } ) \geq m R . } } \end{array}
$$

At least one of the two summands must therefore be at least � $R / 2$ . Pointwise under the coupling,

$$
{ \bf 1 } _ { E _ { 0 } } \leq { \bf 1 } _ { \{ f _ { + } ( \widehat { x } _ { + } ) - f _ { + } ( R ) \geq m R / 2 \} } + { \bf 1 } _ { \{ f _ { - } ( \widehat { x } _ { - } ) - f _ { - } ( - R ) \geq m R / 2 \} } .
$$

Taking expectations and then the larger of the two probabilities gives

$$
\sum _ { s \in \{ - 1 , + 1 \} } \mathbb { P } \{ f _ { s } ( \widehat { x } _ { s } ) - f _ { s } ( s R ) \geq m R / 2 \} \geq \mathbb { P } ( E _ { 0 } ) \geq 4 \delta ,
$$

$$
\operatorname* { m a x } _ { s \in \{ - 1 , + 1 \} } \mathbb { P } \{ f _ { s } ( \widehat { x } _ { s } ) - f _ { s } ( s R ) \geq m R / 2 \} \geq \frac { 1 } { 2 } \mathbb { P } ( E _ { 0 } ) \geq 2 \delta .
$$

The maximum selects one fixed instance; the sign need not depend on the realized transcript. Finally,

$$
\frac { m R } { 2 } = \frac { R } { 2 } \operatorname * { m i n } \left\{ L R , \frac { \sigma } { 2 } ( b / T ) ^ { a } \right\} = \operatorname * { m i n } \left\{ \frac { L R ^ { 2 } } { 2 } , \frac { \sigma R } { 4 } ( b / T ) ^ { ( p - 1 ) / p } \right\} .
$$

This proves Proposition 1.

## F.2 Proof of Proposition 2

Use the same $a = ( p - 1 ) / p , b = \log ( 1 / ( 4 \delta ) )$ , and $\rho = b / ( 2 T )$ , but now set

$$
m = \operatorname* { m i n } \left\{ G , \frac { \sigma } { 2 } ( b / T ) ^ { a } \right\} = \operatorname* { m i n } \left\{ G , \frac { \sigma \rho ^ { a } } { 2 ^ { 1 / p } } \right\} , \qquad f _ { s } ( x ) = - s m x , \qquad r ( x ) = \iota _ { [ - R , R ] } ( x ) .
$$

The linear function $f _ { s }$ is convex and satisfies

$$
| f _ { s } ( x ) - f _ { s } ( y ) | = m | x - y | \leq G | x - y | , \qquad \partial f _ { s } ( x ) = \{ - s m \} .
$$

The indicator � of the closed convex interval is proper, lower semicontinuous, and convex. Direct minimization on this interval gives

$$
x _ { s } ^ { \star } = s R , \qquad F _ { s } ^ { \star } = f _ { s } ( s R ) = - m R , \qquad | x _ { 1 } - x _ { s } ^ { \star } | = R .
$$

For every $\alpha > 0 .$ , its exact proximal map is

$$
\begin{array} { l } { \displaystyle \operatorname { p r o x } _ { \alpha r } ( y ) = \arg \underset { x \in \mathbb R } { \operatorname* { m i n } } \left\{ r ( x ) + \frac { | x - y | ^ { 2 } } { 2 \alpha } \right\} } \\ { = \displaystyle \arg \underset { x \in [ - R , R ] } { \operatorname* { m i n } } | x - y | ^ { 2 } = \left\{ \begin{array} { l l } { - R , } & { y < - R , } \\ { y , } & { - R \le y \le R , } \\ { R , } & { y > R . } \end{array} \right. } \end{array}
$$

This oracle is identical for the two signs, for every query $y , \alpha$

For each stochastic query, draw a fresh $B _ { t } \sim \mathrm { B e r n o u l l i } ( \rho )$ and return $g _ { s , t } ( x ) = - s m B _ { t } / \rho$ . For any adaptive query,

$$
\begin{array} { r l } & { \mathbb { E } _ { t } g _ { s , t } ( x _ { t } ) = - \frac { s m } { \rho } \mathbb { E } _ { t } B _ { t } = - s m \in \partial f _ { s } ( x _ { t } ) , } \\ & { \mathbb { E } _ { t } | g _ { s , t } ( x _ { t } ) + s m | ^ { p } = m ^ { p } \left[ ( 1 - \rho ) + \rho ^ { 1 - p } ( 1 - \rho ) ^ { p } \right] } \\ & { \qquad \leq 2 m ^ { p } \rho ^ { 1 - p } } \\ & { \qquad \leq 2 \left( \frac { \sigma \rho ^ { a } } { 2 ^ { 1 / p } } \right) ^ { p } \rho ^ { 1 - p } = \sigma ^ { p } . } \end{array}
$$

The moment calculation uses the same two Bernoulli outcomes as (91).

On $E _ { 0 } = \{ B _ { 1 } = \cdots = B _ { T } = 0 \}$ , every stochastic reply is zero. Fix the same random seed in both runs. Induction over all oracle calls, including any intervening proximal calls, gives identical transcripts: stochastic answers are zero, and proximal answers come from the same map displayed above. The outputs therefore agree on $E _ { 0 }$ . As before, unused Bernoulli variables may be generated after early stopping, and $\mathbb { P } ( E _ { 0 } ) \ge 4 \delta \mathrm { b y } ( 9 2 )$

If the common output b� belongs to $[ - R , R ]$ , then

$$
\begin{array} { c } { F _ { + } ( \widehat { x } ) - F _ { + } ^ { \star } = - m \widehat { x } - ( - m R ) = m ( R - \widehat { x } ) , } \\ { F _ { - } ( \widehat { x } ) - F _ { - } ^ { \star } = m \widehat { x } - ( - m R ) = m ( R + \widehat { x } ) , } \\ { \left( F _ { + } ( \widehat { x } ) - F _ { + } ^ { \star } \right) + \left( F _ { - } ( \widehat { x } ) - F _ { - } ^ { \star } \right) = 2 m R . } \end{array}
$$

At least one gap is at least ��. If ${ \widehat { x } } \notin \left[ - R , R \right]$ , both gaps equal +∞, so the same conclusion holds. Consequently,

$$
\begin{array} { c } { { \displaystyle \mathbf { 1 } _ { E _ { 0 } } \leq \displaystyle \sum _ { s \in \{ - 1 , + 1 \} } \mathbf { 1 } _ { \{ F _ { s } ( \widehat { x } _ { s } ) - F _ { s } ^ { \star } \geq m R \} } , } } \\ { { \displaystyle \sum _ { s \in \{ - 1 , + 1 \} } \mathbb { P } \{ F _ { s } ( \widehat { x } _ { s } ) - F _ { s } ^ { \star } \geq m R \} \geq \mathbb { P } ( E _ { 0 } ) \geq 4 \delta , } } \\ { { \displaystyle \operatorname* { m a x } _ { s \in \{ - 1 , + 1 \} } \mathbb { P } \{ F _ { s } ( \widehat { x } _ { s } ) - F _ { s } ^ { \star } \geq m R \} \geq \displaystyle \frac { 1 } { 2 } \mathbb { P } ( E _ { 0 } ) \geq 2 \delta . } } \end{array}
$$

Substituting

$$
m R = \operatorname* { m i n } \left\{ G R , { \frac { \sigma R } { 2 } } ( b / T ) ^ { ( p - 1 ) / p } \right\}
$$

proves Proposition 2.

□

## F.3 Proof of Proposition 3

Choose

$$
a = \frac { p - 1 } { p } , \qquad \rho = \frac { q } { 2 T } , \qquad m = \operatorname * { m i n } \{ L R , ( \sigma / 2 ) \rho ^ { a } \} .
$$

Since $1 \leq q \leq T$ , we have $0 < \rho \leq 1 / 2$ and $m > 0$ . Use the quadratic functions and latent clean oracle

$$
f _ { s } ( x ) = { \frac { m } { 2 R } } ( x - s R ) ^ { 2 } , \qquad g _ { s , t } ( x ) = { \frac { m } { R } } x - { \frac { s m } { \rho } } B _ { t } , \qquad B _ { t } \sim \mathrm { B e r n o u l l i } ( \rho ) .
$$

The calculations in the first proof show that the functions are convex and �-smooth, that $x _ { s } ^ { \star } = s R$ , and that $| x _ { 1 } - x _ { s } ^ { \star } | = R$ . Fresh Bernoulli samples also give

$$
\begin{array} { c } { \displaystyle \mathbb { E } _ { t } g _ { s , t } ( x _ { t } ) = \frac { m } { R } x _ { t } - s m = f _ { s } ^ { \prime } ( x _ { t } ) , } \\ { \displaystyle \mathbb { E } _ { t } | g _ { s , t } ( x _ { t } ) - f _ { s } ^ { \prime } ( x _ { t } ) | ^ { p } = m ^ { p } \left[ ( 1 - \rho ) + \rho ^ { 1 - p } ( 1 - \rho ) ^ { p } \right] } \\ { \displaystyle \leq 2 m ^ { p } \rho ^ { 1 - p } } \\ { \displaystyle \leq 2 \left( \frac { \sigma } { 2 } \rho ^ { a } \right) ^ { p } \rho ^ { 1 - p } = 2 ^ { 1 - p } \sigma ^ { p } \leq \sigma ^ { p } . } \end{array}
$$

These are conditional statements with respect to the enlarged filtration containing past latent samples and past adversarial choices.

For each instance, the adversary replaces an informative sample $( B _ { t } = 1 )$ if fewer than $q$ replacements have already been made. Equivalently, define recursively

$$
\begin{array} { r } { C _ { t } = B _ { t } \mathbf { 1 } _ { \{ \sum _ { j = 1 } ^ { t - 1 } C _ { j } < q \} } , \qquad \widetilde { g } _ { s , t } ( x _ { t } ) = \left\{ \begin{array} { l l } { ( m / R ) x _ { t } , } & { C _ { t } = 1 , } \\ { g _ { s , t } ( x _ { t } ) , } & { C _ { t } = 0 . } \end{array} \right. } \end{array}
$$

When $C _ { t } = 1$ , the replacement difers from the latent reply by �� $/ \rho \neq 0 .$ , so $C _ { t }$ is exactly the replacement indicator. The rule is causal: from the current query and sample the adversary can distinguish

$$
B _ { t } = 0 \implies g _ { s , t } ( x _ { t } ) = ( m / R ) x _ { t } , \qquad B _ { t } = 1 \implies g _ { s , t } ( x _ { t } ) = ( m / R ) x _ { t } - s m / \rho .
$$

Once the running count reaches $q ,$ , all later $C _ { t }$ are zero. Therefore $\textstyle \sum _ { t = 1 } ^ { T } C _ { t } \leq q$ on every sample path. Let $\begin{array} { r } { N _ { B } = \sum _ { t = 1 } ^ { \bar { T } } B _ { t } } \end{array}$ and $E _ { q } = \{ N _ { B } \leq q \}$ . By linearity of expectation and Markov’s inequality,

$$
\begin{array} { c } { \displaystyle \mathbb { E } N _ { B } = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } B _ { t } = T \rho = q / 2 , } \\ { \displaystyle \mathbb { P } ( N _ { B } > q ) \leq \mathbb { P } ( N _ { B } \geq q ) \leq \frac { \mathbb { E } N _ { B } } { q } = \frac { 1 } { 2 } , } \\ { \displaystyle \mathbb { P } ( E _ { q } ) = 1 - \mathbb { P } ( N _ { B } > q ) \geq \frac { 1 } { 2 } . } \end{array}
$$

On $E _ { q } .$ , whenever $B _ { t } = 1$ there have been at most $q - 1$ earlier informative samples, so the replacement budget has not yet been exhausted. Thus every informative sample is replaced, and

$$
{ \widetilde { g } } _ { s , t } ( x _ { t } ) = { \frac { m } { R } } x _ { t } \qquad { \mathrm { f o r ~ a l l ~ } } t \leq T \ { \mathrm { o n ~ } } E _ { q } .
$$

With the same random seed and Bernoulli sequence in the two runs, these responses give identical adaptive query transcripts and hence identical outputs on $E _ { q }$ . For early-stopping algorithms we again generate the unused Bernoulli variables without making further queries.

For the common output $\widehat { x }$ on $E _ { q }$ ,

$$
\begin{array} { l } { { \displaystyle \left( f _ { + } ( \widehat x ) - f _ { + } ( R ) \right) + \left( f _ { - } ( \widehat x ) - f _ { - } ( - R ) \right) } \ ~ } \\ { { \displaystyle = \frac { m } { 2 R } \left( ( \widehat x - R ) ^ { 2 } + ( \widehat x + R ) ^ { 2 } \right) = \frac { m } { R } ( \widehat x ^ { 2 } + R ^ { 2 } ) \geq m R . } } \end{array}
$$

It follows that

$$
\begin{array} { r l r } {  { \mathbf { 1 } _ { E _ { q } } \le \sum _ { s \in \{ - 1 , + 1 \} } \mathbf { 1 } _ { \{ f _ { s } ( \widehat { x } _ { s } ) - f _ { s } ( s R ) \geq m R / 2 \} } , } } \\ & { } & { \displaystyle \sum _ { s \in \{ - 1 , + 1 \} } \mathbb { P } \{ f _ { s } ( \widehat { x } _ { s } ) - f _ { s } ( s R ) \geq m R / 2 \} \geq \mathbb { P } ( E _ { q } ) \geq \frac { 1 } { 2 } , } \\ & { } & { \displaystyle \operatorname* { m a x } _ { s \in \{ - 1 , + 1 \} } \mathbb { P } \{ f _ { s } ( \widehat { x } _ { s } ) - f _ { s } ( s R ) \geq m R / 2 \} \geq \frac { 1 } { 2 } \mathbb { P } ( E _ { q } ) \geq \frac { 1 } { 4 } . } \end{array}
$$

Each probability is the marginal law of one clean instance together with its own causal adversary. Hence the maximizing sign gives one fixed admissible instance and adversary, with threshold

$$
\frac { m R } { 2 } = \frac { R } { 2 } \operatorname* { m i n } \left\{ L R , \frac { \sigma } { 2 } \left( \frac { q } { 2 T } \right) ^ { ( p - 1 ) / p } \right\} .
$$

This proves Proposition 3.

□

## References

D. A. Freedman. On tail probabilities for martingales. The Annals of Probability, 3 (1):100–118, 1975.

Z. Liu. Clipped gradient methods for nonsmooth convex optimization under heavy-tailed noise: A refined analysis. In International Conference on Learning Representations, 2026.

T. D. Nguyen, T. H. Nguyen, A. Ene, and H. L. Nguyen. Improved convergence in high probability of clipped gradient methods with heavy tailed noise. In Advances in Neural Information Processing Systems, volume 36, 2023.