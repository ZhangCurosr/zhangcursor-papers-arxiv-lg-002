# Two-Layer Random ReLU Networks Can Separate Disjoint Classes

Yichen Liu<sup>∗</sup> Jianbin Yang<sup>†</sup>

Department of Mathematics, Hohai University 8 Focheng West Road, Jiangning District Nanjing 211100, China

## Abstract

Set separation by continuous linear functionals is fundamental in functional analysis and convex optimization and underlies binary classification in machine learning. Support vector machines provide maximum-margin afine separators for linearly separable data, while kernel methods represent nonlinear decision rules through linear separation in a reproducing-kernel Hilbert space. We study when two classes, not necessarily convex or linearly separable in the input space, can be separated with positive margin by finite ReLU networks and by random ReLU features.

At the deterministic level, we prove that positive-margin separation by a finite ReLU network is equivalent to the existence of finite closed convex covers whose opposite-label members have positive pairwise distances, and exactly two hidden layers sufice. For random separation, suppose each class is represented as a finite union of nonempty closed convex cells, possibly unbounded. If every opposite-label cell pair has positive distance and its closed diference cone is pointed, then two independent random ReLU layers yield a common positive-margin afine separator with high probability.

The first layer separates the feature image of each negative cell from the feature image of the positive class by a hyperplane through the origin. It also controls all cross-class distances uniformly. The second layer combines cellwise scores through Chebyshev polynomials and realizes the resulting polynomial using Gaussian Hermite features, followed by a uniform finite-width approximation. With the remaining geometric, bias, and confidence parameters fixed, each suficient width is quadratic in the largest Gaussian width of a spherical diference-cone section, hence at most linear in its span dimension. A linearly inseparable construction based on translated circular cones and singletons shows that the squared Gaussian-width dependence is sharp for both layers.

Keywords: Random neural networks; ReLU; linear separability; unbounded classes; Gaussian width; convex cones; random features.

2020 Mathematics Subject Classification: Primary 68T07; Secondary 60B20, 60D05, 52A20.

## 1 Introduction

Separating two sets by continuous linear functionals is a basic problem in functional analysis and convex optimization. Under appropriate hypotheses, separation theorems turn geometric information such as disjointness or positive distance into linear certificates for duality and optimality [25]. In machine learning, the same principle underlies binary classification. When input features are linearly separable, a support vector machine selects a maximum-margin afine separator [5]. When linear separation fails, a kernel method can map the data into a reproducing-kernel Hilbert space (RKHS), where a linear separator gives a nonlinear decision rule in the original space [28]. This approach is powerful, but its geometry depends on the chosen kernel and can entail substantial computational and storage costs. These limitations motivate a complementary question: when can a fixed random network feature map separate two entire classes, including classes that are unbounded or not linearly separable in the input space?

Random neural networks ofer one approach to this question. Their hidden weights are sampled once, producing nonlinear features on which a linear readout can be fitted; random-feature methods use this principle for scalable classification and regression [24]. ReLU is especially natural because it is positively homogeneous and widely used in neural networks. Recent work has studied whether random ReLU layers can create linear separability without training the hidden weights [7, 8]. We study separation of whole classes rather than only finite samples. The classes themselves need not be convex: finite closed convex covers provide the geometric structure used in our deterministic characterization and random-network analysis.

Our first result gives an exact deterministic criterion. For arbitrary nonempty classes $S ^ { + } , S ^ { - } \subset$ $\mathbb { R } ^ { d }$ , a finite ReLU network with an afine output separates them with positive margin if and only if each class is contained in a finite union of nonempty closed convex sets and every pair of oppositelabel covering sets has positive distance. Moreover, exactly two hidden layers sufice whenever such covers exist.

We next state the random result. For this theorem, write

$$
S ^ { + } = \bigcup _ { i = 1 } ^ { N _ { + } } E _ { i } ^ { + } , \qquad S ^ { - } = \bigcup _ { j = 1 } ^ { N _ { - } } E _ { j } ^ { - } ,\tag{1}
$$

where the cells $E _ { i } ^ { + }$ and $E _ { j } ^ { - }$ are nonempty, closed, and convex; neither class is required to be convex. Assume that every opposite-label gap $\delta _ { i j } = \mathrm { d i s t } ( E _ { i } ^ { + } , E _ { j } ^ { - } )$ is positive. For each $i , j ,$ , define the closed diference cone $K _ { i j } = \mathrm { c l } \mathrm { c o n e } ( E _ { i } ^ { + } - E _ { j } ^ { - } )$ . The random separation theorem assumes that each $K _ { i j }$ is pointed:

$$
K _ { i j } \cap ( - K _ { i j } ) = \{ 0 \} .\tag{2}
$$

Put $\delta _ { 0 } = \operatorname* { m i n } _ { i , j } \delta _ { i j }$ , let R and $\kappa _ { \mathrm { m i n } }$ be the scale and cone-depth parameters defined in Section 2, and set

$$
\omega _ { * } : = \operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { d - 1 } ) .
$$

For any $0 < \eta < 1$ , take the second-layer bias radius to be $\widehat { \lambda } = \delta _ { 0 }$ . There are explicit factors $\theta _ { 1 } , \theta _ { 2 }$ given in Theorems 2.5 and 2.7, such that the first-layer bias radius and both widths may be chosen to satisfy

$$
\begin{array} { r l } & { \lambda \ge C \delta _ { 0 } \frac { 1 + R / \delta _ { 0 } } { \sqrt { \kappa _ { \operatorname* { m i n } } } } \sqrt { \log ( 4 N _ { + } ) } , } \\ & { n \ge C \theta _ { 1 } \left[ 1 + \omega _ { * } ^ { 2 } + \log \frac { 4 8 N _ { + } N _ { - } } { \eta } \right] , } \\ & { \widehat { n } \ge C \theta _ { 2 } ( 1 + \omega _ { * } ^ { 2 } ) . } \end{array}
$$

Here $C > 0$ is an absolute constant. Under these conditions, with probability at least $1 - \eta$ over the two independent random layers, there exist a unit vector $v ,$ an ofset $\tau \in \mathbb { R }$ , and a deterministic margin $m > 0$ such that

$$
\langle v , F ( x ) \rangle + \tau \geq m \quad ( x \in S ^ { + } ) , \qquad \langle v , F ( y ) \rangle + \tau \leq - m \quad ( y \in S ^ { - } ) .
$$

Our construction gives $\tau = 0$ . In both widths, the dependence on $\omega _ { * }$ is quadratic, hence at most linear in the intrinsic span dimension when the other geometric parameters are fixed; $\theta _ { 1 }$ and $\theta _ { 2 }$ have at most singly exponential dependence on the remaining parameters.

Our contributions are threefold.

(1) A deterministic characterization of set separation. Positive-margin separation by a finite ReLU network is equivalent to the existence of finite closed convex covers with positive distances between all opposite-label cover sets; exactly two hidden layers sufice (Theorem 2.1). We also give an explicit unbounded example for which no finite-width one-hidden-layer ReLU network achieves strict separation (Example 2.2).

(2) Random two-layer separation beyond bounded data. For an individual pair of convex covering cells at positive distance, pointedness of the diference cone is equivalent to positive probability of strict Gaussian afine separation with a suitable finite uniform bias range; cone depth quantifies the angular margin (Proposition 2.4). Under positive opposite-cell gaps and pairwise pointedness, the first random layer separates the image of each negative cell from the image of the entire positive class and controls all cross-class distances uniformly. A second independent layer combines these cellwise separators into one positive-margin separator for the full classes. The proof uses normalized endpoints, Chebyshev polynomials, and Gaussian Hermite features. A nonpointed strip example shows that positive gaps alone do not ensure random separation at any fixed finite depth (Example 3.1).

(3) Intrinsic width bounds and sharpness. Each suficient width is quadratic in the largest Gaussian width of a spherical diference-cone section, and this order is necessary for both layers on a linearly inseparable family of translated circular cones and singletons (Example 3.2). Thus the width dependence is governed by intrinsic cone geometry rather than ambient dimension, while both prefactors have at most single-exponential dependence on the remaining parameters. Numerical experiments illustrate the width trends for non-linearly-separable classes with convex components.

Random-feature methods have been studied for kernel approximation and learning [24], and recent work gives random-ReLU separation guarantees for bounded data and quadratic-feature guarantees for unions of subspaces [7, 8, 31]. A detailed comparison of these results with our theorems, along with related geometric-embedding results, appears in Subsection 3.3.

The remainder of the paper is organized as follows. Section 2 states the deterministic characterization, the geometric criterion for random afine separation, and the two-layer random separation theorem. Section 3 gives the pointedness obstruction and matching width lower bounds, and compares these results with related work. Section 4 reports the numerical experiments. The main text closes with concluding remarks; the appendices contain the proofs.

## 2 Two-Layer ReLU Separation of Classes

This section develops separation results for two labeled sets, which need not themselves be convex. We first give the deterministic finite-network characterization, discuss the limitations of one hidden layer, and introduce the problem considered in this paper. We then state the geometric criterion for random afine separation and the high-probability results for two independent random ReLU layers.

Let $\lVert x \rVert = ( \langle x , x \rangle ) ^ { 1 / 2 }$ , and let $P _ { A }$ denote the Euclidean projection onto a closed convex set A. We write cone A, cl A, and conv A for the conic hull, closure, and convex hull of a set A, respectively.

Define the closed unit ball and the unit sphere in $\mathbb { R } ^ { d }$ by

$$
B _ { 2 } ^ { d } = \{ u \in \mathbb { R } ^ { d } : \| u \| \leq 1 \} , \qquad S ^ { d - 1 } = \{ u \in \mathbb { R } ^ { d } : \| u \| = 1 \} .
$$

For each positive integer $N .$ , write $[ N ] = \{ 1 , \dots , N \}$ . For a nonempty bounded set $A ,$ define its Gaussian width by

$$
w ( A ) = \mathbb { E } _ { g } \operatorname* { s u p } _ { z \in A } \langle g , z \rangle , \qquad g \sim N ( 0 , I ) ,
$$

where $g$ is a standard Gaussian vector in the ambient space of A.

For a real random variable X, a Euclidean random vector $Y .$ , and $1 \leq p < \infty$ , define

$$
\| X \| _ { L ^ { p } } = ( \mathbb { E } _ { X } | X | ^ { p } ) ^ { 1 / p } , \qquad \| Y \| _ { L ^ { p } } = ( \mathbb { E } _ { Y } \| Y \| ^ { p } ) ^ { 1 / p } .
$$

The inner norm $\| Y \|$ is Euclidean, and the $L ^ { p }$ norm of a function is taken with respect to the random argument specified by the context. For matrices, $\lVert \cdot \rVert _ { \mathrm { F } }$ denotes the Frobenius norm.

## 2.1 Network model and separation criteria

The criterion below applies to arbitrary nonempty sets. It identifies exactly when a finite ReLU network can separate them with positive margin and shows that two hidden layers always sufice.

Theorem 2.1 (Finite convex-cover characterization). For nonempty sets $S ^ { + } , S ^ { - } \subset \mathbb { R } ^ { d }$ , the following are equivalent.

(i) A finite ReLU network followed by an afine output separates $S ^ { + }$ and $S ^ { - }$ with positive margin.

(ii) There exist finite families of nonempty closed convex sets $\{ C _ { i } ^ { + } \} _ { i = 1 } ^ { M _ { + } }$ and $\{ C _ { j } ^ { - } \} _ { j = 1 } ^ { M _ { - } }$ such that

$$
S ^ { + } \subset \bigcup _ { i } C _ { i } ^ { + } , \qquad S ^ { - } \subset \bigcup _ { j } C _ { j } ^ { - } , \qquad \operatorname { d i s t } ( C _ { i } ^ { + } , C _ { j } ^ { - } ) > 0 \quad f o r \ a l l \ i , j .
$$

(iii) A ReLU network with exactly two hidden layers followed by an afine output separates $S ^ { + }$ and $S ^ { - }$ with positive margin.

The proof is in Appendix C.

For bounded classes, the cover condition in Theorem 2.1 is equivalent to dist $( S ^ { + } , S ^ { - } ) > 0$ , and a finite-width one-hidden-layer ReLU network also sufices [8]. For unbounded classes, one hidden layer may fail, as the example below shows.

Why one hidden layer may fail. Let $s = ( s _ { 1 } , s _ { 2 } , s _ { 3 } ) \in \{ - 1 , 1 \} ^ { 3 }$ and define

$$
E _ { s } = \{ x \in \mathbb { R } ^ { 3 } : s _ { r } x _ { r } \geq 1 , \ r = 1 , 2 , 3 \} , \ \chi ( s ) = s _ { 1 } s _ { 2 } s _ { 3 } .\tag{3}
$$

The two classes in Example 2.2 are

$$
S ^ { + } = \bigcup _ { \chi ( s ) = 1 } E _ { s } , \qquad S ^ { - } = \bigcup _ { \chi ( s ) = - 1 } E _ { s } .\tag{4}
$$

The two classes have distance 2.

Example 2.2 (Shallow obstruction). For the classes $S ^ { + } , S ^ { - }$ in (4), with cells from (3), no finitewidth one-hidden-layer ReLU function

$$
h ( x ) = c + \sum _ { k = 1 } ^ { m } a _ { k } \operatorname { R e L U } ( \langle v _ { k } , x \rangle + b _ { k } )
$$

has opposite signs with a uniform positive margin on $S ^ { + }$ and $S ^ { - }$ . The same is true for strict zero-threshold separation. The example embeds into every $\mathbb { R } ^ { d } , d \geq 3$

The proof is in Appendix C.

Remark 2.3 (Necessity of two hidden layers). The example above shows that one hidden layer can fail even when the deterministic finite-cover criterion holds. By Theorem 2.1, two hidden layers sufice for this example. Thus, two hidden layers are necessary for some class pairs.

In the following, we consider the composition $F = \widehat \Phi \circ$ Φ of two independent random ReLU layers,

$$
\Phi ( x ) = \sqrt { \frac { 2 } { n } } \mathrm { R e L U } ( W x + b ) , \qquad \widehat { \Phi } ( z ) = \sqrt { \frac { 2 } { \hat { n } } } \mathrm { R e L U } ( \widehat { W } z + \hat { b } ) ,\tag{5}
$$

where $W \in \mathbb { R } ^ { n \times d }$ and $\widehat { W } \in \mathbb { R } ^ { \widehat { n } \times n }$ have standard Gaussian entries. The bias vectors $b \in \mathbb { R } ^ { n }$ and $\widehat { b } \in \mathbb { R } ^ { \widehat { n } }$ have entries uniform on $[ - \lambda , \lambda ]$ and $[ - \widehat { \lambda } , \widehat { \lambda } ]$ , respectively. All matrix and vector entries are mutually independent.

This sampling scheme follows [7, Definition 1]. The factor $\sqrt { 2 }$ compensates for the secondmoment reduction of ReLU under symmetric inputs, as in random initialization [11]. Here each layer is normalized by the square root of its output width.

Two-layer random ReLU separation problem. Given the classes in (1) and a failure probability $\eta \in ( 0 , 1 )$ , assume that $\delta _ { 0 } > 0$ and every diference cone in (2) is pointed. We seek conditions on the widths $n , \widehat { n }$ and bias scales $\lambda , { \widehat { \lambda } }$ that guarantee the following: there is a margin $m > 0$ , fixed before sampling, such that with probability at least $1 - \eta$ the sampled network admits a unit vector $\boldsymbol { v } \in \mathbb { R } ^ { \widehat { n } }$ and an ofset $\tau \in \mathbb { R }$ satisfying

$$
\langle v , F ( x ) \rangle + \tau \geq m \quad ( x \in S ^ { + } ) , \qquad \langle v , F ( y ) \rangle + \tau \leq - m \quad ( y \in S ^ { - } ) .
$$

The next result concerns the random directions used in our class-separation theorem. For an individual pair of convex covering cells, positive distance guarantees deterministic afine separation, but a Gaussian draw can miss all strictly separating directions with probability one when the cells are unbounded. The parameter $\kappa ( K )$ measures the robustness of a separating direction. A larger value gives greater tolerance to changes in direction. It also enters the bounds on separation probabilities and suficient widths for random ReLU networks.

Proposition 2.4 (Cone geometry of afine separation). Let $E ^ { + } , E ^ { - } \subset \mathbb { R } ^ { d }$ be nonempty closed convex sets with positive distance. Put $K = \operatorname { c l } \operatorname { c o n e } ( E ^ { + } - E ^ { - } )$ . Define

$$
C _ { K } = \mathrm { c o n v } ( K \cap { \cal S } ^ { d - 1 } ) , \qquad \kappa ( K ) = \mathrm { d i s t } ( 0 , C _ { K } ) .
$$

For $a \neq 0$ , the afine hyperplane $\langle a , x \rangle + c = 0$ linearly separates $E ^ { - }$ and $E ^ { + }$ with margin $t > 0$ if

$$
\langle a , x \rangle + c \leq - t \quad ( x \in E ^ { - } ) , \qquad \langle a , x \rangle + c \geq t \quad ( x \in E ^ { + } ) .
$$

When $\| a \| = 1$ , the parameter t is a geometric margin. The following assertions hold.

(i) K is pointed if and only if there exist $\lambda , t > 0$ such that, with positive probability, the random afine hyperplane $\langle g , x \rangle + b = 0$ , where $g \sim N ( 0 , I _ { d } )$ and $b \sim \operatorname { U n i f } [ - \lambda , \lambda ]$ are independent, separates $E ^ { - }$ and $E ^ { + }$ with margin t.

(ii) If K is pointed, set $u _ { K } = P _ { C _ { K } } ( 0 ) / \| P _ { C _ { K } } ( 0 ) \|$ . Then $\kappa ( K ) > 0$ , and $u _ { K }$ is the unique solution of

$$
\operatorname* { m a x } _ { u \in S ^ { d - 1 } } \operatorname* { i n f } _ { z \in K \cap S ^ { d - 1 } } \langle u , z \rangle = \kappa ( K ) .
$$

In particular,

$$
\langle u _ { K } , z \rangle \geq \kappa ( K ) \| z \| \qquad ( z \in K ) .\tag{6}
$$

The proof is in Appendix C.

![](images/4841ef05cc2492fce977135f2a1feb2aafa2758230197795ce08a892afda52f1.jpg)  
Figure 1: Cone depth (a). The purple sector in (b) shows separating directions; its dashed boundary rays are excluded.

## 2.2 Random two-layer separation of the classes

We now state the random separation theorem. To quantify the geometry of the opposite-label cells, define

$$
x _ { i } ^ { \circ } = P _ { E _ { i } ^ { + } } ( 0 ) , \qquad y _ { j } ^ { \circ } = P _ { E _ { j } ^ { - } } ( 0 ) , \qquad R = \operatorname* { m a x } _ { i , j } \{ \| x _ { i } ^ { \circ } \| , \| y _ { j } ^ { \circ } \| \} .
$$

The parameter R records the scale of the nearest points of these cells to the origin and remains finite. Under (2), write $\kappa _ { i j } = \kappa ( K _ { i j } ) , u _ { i j } = u _ { K _ { i j } }$ , and $\kappa _ { \mathrm { m i n } } = \mathrm { m i n } _ { i , j } \kappa _ { i j } > 0$ by Proposition 2.4(ii).

The first random ReLU layer separates each $\Phi ( E _ { j } ^ { - } )$ from $\Phi ( S ^ { + } )$ , as illustrated in Figure 2. The next theorem gives suficient width and bias conditions and a uniform margin that the second layer uses to separate the two classes as a whole.

Theorem 2.5 (First-layer separation). Given the classes in (1), assume that (2) holds and $\delta _ { 0 } > 0$ Let $0 < \eta < 1$ . There exists a positive constant $C ,$ independent of $S ^ { + } , S ^ { - }$ and η, such that if

$$
\lambda \ge C \delta _ { 0 } \frac { 1 + R / \delta _ { 0 } } { \sqrt { \kappa _ { \operatorname* { m i n } } } } \sqrt { \log ( 4 N _ { + } ) }\tag{7}
$$

and

$$
n \geq C \theta _ { 1 } \left[ 1 + \operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { d - 1 } ) ^ { 2 } + \log \frac { 4 8 N _ { + } N _ { - } } { \eta } \right] ,\tag{8}
$$

then, with probability at least $1 - \eta$ , there exist a common constant $\vartheta > 0$ and unit vectors $a _ { j } \in \mathbb { R } ^ { n }$ $j \in [ N _ { - } ]$ , such that, simultaneously for every $j \in [ N _ { - } ] , x \in S ^ { + }$ , and $y \in E _ { j } ^ { - }$

$$
\langle a _ { j } , \Phi ( y ) - \Phi ( x ) \rangle \geq \vartheta \Vert \Phi ( y ) - \Phi ( x ) \Vert ,\tag{9}
$$

and

$$
\begin{array} { r } { \vartheta \| x - y \| \leq \| \Phi ( x ) - \Phi ( y ) \| \leq \frac { 3 } { 2 } \| x - y \| . } \end{array}\tag{10}
$$

In particular, every cl cone $\left( \Phi ( E _ { j } ^ { - } ) - \Phi ( S ^ { + } ) \right)$ is pointed. Here

$$
\theta _ { 1 } = \frac { \lambda } { \delta _ { 0 } } \exp \left\{ \frac { C ( 1 + R / \delta _ { 0 } ) ^ { 2 } \log ( 4 N _ { + } ) } { \kappa _ { \operatorname* { m i n } } } \right\} .
$$

The constant ϑ can be determined from the parameters chosen before sampling.

The proof is in Appendix A.

![](images/6ba3697c92f9c14e56419a2e5128f87d3d970b928e6934659a5f4f1d7ddb9610.jpg)  
Figure 2: First-layer ReLU transformation and radial separation (Theorem 2.5). (a) The two classes are represented by positive and negative convex cells. (b) The map $\Phi$ produces feature cells for which each $\Phi ( E _ { i } ^ { - } )$ is separated from the entire $\Phi ( S ^ { + } )$ by $H _ { j }$ , with normal $a _ { j }$ pointing towards the negative cell. Dashed blue outlines enclose positive convex hulls. (c) For a fixed $j ,$ the diference cone $D _ { j } = \epsilon$ cl cone $( \Phi ( E _ { i } ^ { - } ) - \Phi ( S ^ { + } ) )$ ) lies in the gray angular envelope: every nonzero $d \in D _ { j }$ satisfies $\angle ( a _ { j } , d ) \leq$ arccos $\vartheta < \dot { \pi } / 2$ . The origin is that of the diference space.

The second layer $\widehat { \Phi }$ takes the first-layer features $\Phi ( x ) , x \in S ^ { + } \cup S ^ { - }$ , as its inputs. The following proposition shows that $\left\| \Phi ( x ) \right\|$ is comparable to $\left( \| x \| ^ { 2 } + \lambda ^ { 2 } \right) ^ { 1 / 2 }$ . These norm estimates are used to derive the parameter bounds for the second layer.

Proposition 2.6 (Bounds on $\| \Phi ( x ) \| )$ . Under the assumptions of Theorem 2.5, with probability at least $1 - \eta$ , the conclusions of that theorem hold together with the following bounds, uniformly for all $x \in S ^ { + } \cup S ^ { . }$

$$
{ \frac { 1 } { 8 } } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } \leq \| \Phi ( x ) \| \leq 3 { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } .\tag{11}
$$

The proof is in Appendix A.

The following theorem establishes our main random separation result. The first layer separates each feature image $\Phi ( E _ { j } ^ { - } )$ from $\Phi ( S ^ { + } )$ . The second layer then makes the full feature sets $F ( S ^ { + } )$ and $F ( S ^ { - } )$ linearly separable with a positive margin, as illustrated in Figure 3.

Theorem 2.7 (Second-layer separation). Assume the setting of Theorem ${ \it 2 . 5 , }$ and suppose that the first-layer conditions $( 7 )  { - } ( 8 )$ hold. Set $\widehat { \lambda } = \delta _ { 0 }$ . There exists a universal constant $C > 0$ , independent of the sets and η, such that the following holds. If

$$
\hat { n } \ge C \theta _ { 2 } \left[ 1 + \operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { d - 1 } ) ^ { 2 } \right] ,\tag{12}
$$

then with probability at least $1 - \eta _ { ; }$ , there exist a margin $t > 0$ and a unit vector $\boldsymbol { v } \in \mathbb { R } ^ { \widehat { n } }$ such that

$$
\langle v , F ( x ) \rangle \geq t \quad ( x \in S ^ { + } ) , \qquad \langle v , F ( y ) \rangle \leq - t \quad ( y \in S ^ { - } ) .\tag{13}
$$

Here

$$
\theta _ { 2 } = \exp \Biggl \{ \frac { C N _ { - } ( 1 + R / \delta _ { 0 } ) ^ { 2 } } { \kappa _ { \mathrm { m i n } } } \log \biggl ( \frac { 4 N _ { + } N _ { - } \lambda } { \delta _ { 0 } } \log \frac { 2 } { \eta } \biggr ) \Biggr \} .
$$

On the same event, the conclusions of Theorem 2.5 and Proposition 2.6 also hold. The margin t depends only on parameters fixed before sampling, whereas v may depend on the sampled network.

The proof of Theorem 2.7 is given in Appendix B.

![](images/57c960e310983b903b2527ef334e6fb33f3d1f930c69ec838a73f803de33212c.jpg)  
Figure 3: Linear separation after two random ReLU layers. The feature sets $F ( S ^ { + } )$ and $F ( S ^ { - } )$ are separated with positive margin by $H _ { \mathrm { o u t } } = \{ z : \langle v , z \rangle = 0 \}$ . The normal v points towards $F ( S ^ { + } )$ .

The theorem applies to disjoint classes, which may be unbounded, and provides suficient widths for their separation with probability at least $1 - \eta$ . Fix the remaining geometric parameters and $\eta ,$ and define the intrinsic dimension

$$
r : = \operatorname* { m a x } _ { i , j } \dim \operatorname { s p a n } ( K _ { i j } ) .
$$

For each $i , j$ , if $g \sim N ( 0 , I _ { d } )$ , then

$$
w ( K _ { i j } \cap S ^ { d - 1 } ) \leq \mathbb { E } _ { g } \big \| P _ { \mathrm { s p a n } ( K _ { i j } ) } g \big \| \leq \sqrt { \mathrm { d i m } \mathrm { s p a n } ( K _ { i j } ) } \leq \sqrt { r } ,
$$

where the second inequality follows from the Cauchy–Schwarz inequality. Hence

$$
\operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { d - 1 } ) ^ { 2 } \leq r ,
$$

so both suficient width bounds grow at most linearly with r when the remaining parameters are fixed.

## 3 Geometric Obstructions and Sharp Width Bounds

This section examines the necessity of pointedness and the dependence of network width on Gaussian complexity. We first construct classes at positive distance with nonpointed diference cones. Every fixed finite-depth, finite-width random ReLU network fails to separate these classes almost surely. We then use translated circular cones and singletons to derive necessary widths for both layers. These lower bounds match the squared Gaussian-width dependence in our suficient bounds. The section concludes with a comparison of our results with related work

## 3.1 Necessity of the pointedness assumption

Positive opposite-cell gaps alone do not guarantee random separation. Example 3.1 exhibits classes with nonpointed diference cones for which every fixed finite-depth, finite-width random ReLU network fails to separate the classes almost surely. Thus, a positive gap condition alone cannot yield a uniform random separation guarantee.

Consider the planar convex cover

$$
\begin{array} { r l } { E _ { 1 } ^ { + } = [ 0 , \infty ) \times [ 1 , 2 ] , } & { { } \quad E _ { 2 } ^ { + } = [ 0 , \infty ) \times [ - 2 , - 1 ] , } \\ { E _ { 1 } ^ { - } = [ 0 , \infty ) \times [ - \frac { 1 } { 2 } , \frac { 1 } { 2 } ] , } & { { } \quad S ^ { + } = E _ { 1 } ^ { + } \cup E _ { 2 } ^ { + } , \quad S ^ { - } = E _ { 1 } ^ { - } . } \end{array}\tag{14}
$$

The three cells are closed and convex with nonempty interiors. Both opposite-cell gaps equal $1 / 2$ whereas

$$
K _ { 1 1 } = \mathbb { R } \times [ 0 , \infty ) , \qquad K _ { 2 1 } = \mathbb { R } \times ( - \infty , 0 ] .
$$

Both diference cones contain the line ${ \mathbb R } e _ { 1 }$ , so neither is pointed and $\kappa ( K _ { 1 1 } ) = \kappa ( K _ { 2 1 } ) = 0$

Example 3.1 (Random separation fails at every finite depth). For the classes in (14), fix any finite depth $q \geq 1$ , widths $n _ { 1 } , \ldots , n _ { q } \geq 1$ , and finite bias ranges $\lambda _ { 1 } , \ldots , \lambda _ { q } > 0$ . Let $F _ { 0 } ( x ) = x$ and

$$
F _ { \ell } ( x ) = \sqrt { \frac { 2 } { n _ { \ell } } } \mathrm { R e L U } ( W _ { \ell } F _ { \ell - 1 } ( x ) + b _ { \ell } ) , \qquad 1 \le \ell \le q ,
$$

where all weight entries are independent $N ( 0 , 1 )$ variables and all bias entries are independent uniform variables on their respective $[ - \lambda _ { \ell } , \lambda _ { \ell } ]$ , independently of the weights. Then, almost surely,

$$
F _ { q } ( S ^ { - } ) \cap \mathrm { c o n v } ( F _ { q } ( S ^ { + } ) ) \neq \emptyset .\tag{15}
$$

Consequently, no afine readout separates the two classes for almost every network realization, even if the readout is chosen after observing the network. In particular, positive-margin separation fails almost surely.

The proof is in Appendix C.

The deterministic two-neuron score $h ( x ) = \mathrm { R e L U } ( x _ { 2 } ) + \mathrm { R e L U } ( - x _ { 2 } ) - 3 / 4$ satisfies $h \geq 1 / 4$ on $S ^ { + }$ and $h \leq - 1 / 4$ on $S ^ { - }$ . Random hidden layers encounter a diferent obstruction: on a suficiently distant common tail, their composition is afine and preserves a cross-class midpoint relation. Figure 4 illustrates this geometry;

![](images/d25fd489d73895ce019a6d1dd0e8077956f2ee4050b3800a1aa19b57f8679048.jpg)  
Figure 4: A planar obstruction to random separation at every finite depth (Example 3.1). (a) Two positive strips surround a negative strip, with opposite-cell gaps $1 / 2$ and a common recession direction. (b) Almost surely, the network is afine on $\{ ( t , s ) : t \geq T , | s | \leq 2 \}$ for some finite T; a nondegenerate afine image is shown. In particular, $F _ { q } ( t , 0 ) = [ F _ { q } ( t , 1 ) + F _ { q } ( t , - 1 ) ] / 2$ , so the negative image meets the convex hull of the positive image.

## 3.2 Necessity of the squared Gaussian width

We now give a family of sets that attains a lower bound of this squared order with fixed geometric parameters. Let $r \geq 2 , 0 < \alpha < \pi / 2$ , and define

$$
\begin{array} { c } { K _ { \alpha , r } = \{ z \in \mathbb { R } ^ { r } : z _ { 1 } \geq \cos \alpha \| z \| \} , } \\ { E _ { 1 } ^ { + } = \frac { 3 } { 2 } e _ { 1 } + K _ { \alpha , r } , \qquad E _ { 1 } ^ { - } = - \frac { 3 } { 2 } e _ { 1 } - K _ { \alpha , r } , } \\ { E _ { 2 } ^ { + } = \{ - \frac { 1 } { 2 } e _ { 1 } \} , \qquad E _ { 2 } ^ { - } = \{ \frac { 1 } { 2 } e _ { 1 } \} , \qquad S ^ { \pm } = E _ { 1 } ^ { \pm } \cup E _ { 2 } ^ { \pm } . } \end{array}\tag{16}
$$

Each class consists of a translated circular cone and a singleton. Both convex hulls contain $\{ t e _ { 1 } : | t | \le 1 / 2 \}$ , so the classes are not linearly separable in the input space. They satisfy

$$
\begin{array} { r l r } { N _ { + } = N _ { - } = 2 , \quad } & { { } \delta _ { 0 } = 1 , \quad } & { R = \frac { 3 } { 2 } , \quad } & { \kappa _ { \operatorname* { m i n } } = \cos \alpha , } \\ { K _ { 1 1 } = K _ { 1 2 } = K _ { 2 1 } = K _ { \alpha , r } , \quad } & { { } K _ { 2 2 } = \{ - t e _ { 1 } : t \geq 0 \} . } \end{array}\tag{17}
$$

The spherical section of $K _ { 2 2 }$ is a singleton and has Gaussian width zero. Hence

$$
\operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { r - 1 } ) ^ { 2 } = w ( K _ { \alpha , r } \cap S ^ { r - 1 } ) ^ { 2 } .
$$

For $G _ { r - 1 } \sim N ( 0 , I _ { r - 1 } )$ , the Gaussian width of the spherical section satisfies

$$
\begin{array} { r } { \sin \alpha \mathbb { E } _ { G _ { r - 1 } } \| G _ { r - 1 } \| \le w ( K _ { \alpha , r } \cap S ^ { r - 1 } ) \le \frac { 3 } { 2 } \sin \alpha \mathbb { E } _ { G _ { r - 1 } } \| G _ { r - 1 } \| . } \end{array}\tag{18}
$$

Fix α and increase r. The other geometric parameters remain unchanged, while the squared Gaussian width is of order r.

Example 3.2 (Necessary width of each layer). For (16), consider a two-layer ReLU network with independent Gaussian weights. The biases may take any finite values. Let $0 < \eta < 1$ . If $F ( S ^ { + } )$ and $F ( S ^ { - } )$ are strictly linearly separable with probability at least $1 - \eta ,$ , then each layer must satisfy

$$
\begin{array} { r } { n , \widehat { n } \geq ( 1 - \eta ) r \sin ^ { 2 } \alpha \geq \frac { 4 } { 9 } ( 1 - \eta ) w ( K _ { \alpha , r } \cap S ^ { r - 1 } ) ^ { 2 } . } \end{array}\tag{19}
$$

If only the first-layer radial separation in Theorem 2.5 is required with probability at least $1 - \eta$ then $n \geq ( 1 - \eta ) r \sin ^ { 2 } \alpha$ is still necessary.

The proof is in Appendix C. The obstruction is a pair of points from diferent classes with identical features. Thus adding any constant to $\langle v , F ( x ) \rangle$ cannot separate the classes. Take $\alpha = \pi / 4$ and $\eta = 1 / 4$ . Fix λ as a suficiently large absolute constant according to (7), and take $\widehat { \lambda } = 1$ . Then

$$
\frac { r - 1 } { \pi } \leq w ( K _ { \pi / 4 , r } \cap S ^ { r - 1 } ) ^ { 2 } \leq \frac { 9 } { 8 } ( r - 1 ) .
$$

Theorems 2.5 and 2.7 give a suficient width of $C [ 1 + w ( K _ { \pi / 4 , r } \cap S ^ { r - 1 } ) ^ { 2 } ]$ for each layer. Equation (19) and $\begin{array} { r } { 1 + \frac { 9 } { 8 } ( r - 1 ) \le \frac { 9 } { 8 } r } \end{array}$ give

$$
n , \widehat { n } \geq \frac { 3 r } { 8 } \geq \frac { 1 } { 3 } [ 1 + w ( K _ { \pi / 4 , r } \cap S ^ { r - 1 } ) ^ { 2 } ] .
$$

This shows that, for independent Gaussian random hidden layers, the squared Gaussian-width order cannot be reduced in uniform width bounds over this family of sets. The lower bound constrains each layer separately; widening one layer cannot remove the restriction on the other.

## 3.3 Comparison of results and related work

This subsection compares our results with related work on random neural networks and compressive classification.

Linear separability of shallow features. Studies of trained networks show that, as depth increases, features can become more linearly separable and features within each class can cluster more tightly [1, 10, 21]. For random networks, earlier results establish separation of bounded classes using one or two ReLU layers [7, 8]. Quadratic random features have also been studied for union-ofsubspaces models [31]. Related work establishes generalization rates for deep ReLU classifiers with logistic loss and optimization and generalization guarantees for minimal-width shallow networks [18, 32]. Approximation results give optimal width-depth rates for Hölder functions and nearly optimal bounds for smooth functions [19, 29]. We study entire-class separation by fixed random ReLU layers, including on unbounded sets, with widths controlled by diference-cone geometry. Our result allows finite unions of closed convex sets, provided that opposite-set gaps are positive and the diference cones are pointed. The first layer separates each $\Phi ( E _ { j } ^ { - } )$ from $\Phi ( S ^ { + } )$ . The second layer then makes $F ( S ^ { + } )$ and $F ( S ^ { - } )$ linearly separable with a positive margin. With the remaining parameters fixed, the suficient width of each layer is quadratic in the Gaussian widths of the spherical diference-cone sections and at most linear in their intrinsic span dimension. Table 1 gives the precise set assumptions, network models, and separation guarantees.

Table 1: High-probability separation by random features. Widths are suficient bounds with remaining geometric and covering parameters and failure probability fixed; bias scales satisfy the cited bounds. In the first two rows, intrinsic dimension is the span dimension of the input classes. Weights are Gaussian; ReLU layers have uniform biases and are independent.
<table><tr><td>Work</td><td>Sets</td><td>Network</td><td>Separation and sufficient width</td></tr><tr><td>[7]</td><td>Dirksen et al. Two bounded sets at positive distance</td><td>Two ReLU layers.</td><td>Positive margin. First width: quadratic in Gaussian widths, at most linear in intrinsic dimension. Second: exponential in squared Gaussian widths, at most</td></tr><tr><td>[8]</td><td>Ghosal et al. Two bounded sets at positive distance</td><td>One ReLU layer.</td><td>exponential in dimension. Positive margin. Width exponential in squared Gaussian width, at most exponential in intrinsic dimension; additional mutual-complexity dependence (Theorem 1.4).</td></tr><tr><td>Xu et al. (UoS) [31]</td><td>Two r-dimensional linear One subspaces intersecting only at 0.</td><td>quadratic layer; zero bias.</td><td>Strict separation off 0; no uniform positive margin. Width  $O ( r ^ { 3 } \log ( r / \eta ) )$  for fixed minimum principal angle.</td></tr><tr><td>This work Theorems 2.5, 2.7</td><td>Finite unions of closed convex sets, possibly unbounded; positive opposite-set gaps and</td><td>Two ReLU layers.</td><td>Positive margin. Each width: quadratic in cone Gaussian widths, at most linear in intrinsic dimension; sharp complexity order (Example 3.2).</td></tr></table>

Compressive classification and cone geometry. Compressive classification uses compressed measurements directly, without first reconstructing the signal [6]. Bandeira et al. study the rare eclipse problem [3]. It asks how Gaussian projections can reduce dimension while keeping two disjoint closed convex sets disjoint with high probability. Using Gordon’s escape through a mesh theorem [9], they bound suficient measurement counts in terms of the squared Gaussian width of the spherical diference-cone section. In convex recovery, exact recovery is equivalent to the measurement kernel intersecting the tangent cone at the target signal only at the origin [4]. Suficient Gaussian measurement counts are likewise controlled by the squared Gaussian width of this cone’s spherical section.

Our width lower bounds also connect random kernels with cone geometry. Here the kernel is the nullspace of a random linear map. In Example 3.2, if this nullspace intersects the interior of the specified cone, we can find two points from opposite classes that have identical network features. No subsequent layer or classifier can distinguish these points. Estimating the probability of this event gives the width lower bounds.

Random-feature methods. Random-feature methods and extreme learning machines fix the hidden parameters after random sampling and learn only the output coeficients [14, 24]. Recent empirical studies show that these methods can achieve test performance comparable to networks trained end to end on selected classification and regression benchmarks [13, 20].

Random features also admit theoretical guarantees in kernel learning. For example, under suitable assumptions, ridge regression with random features can match the learning rate of full kernel ridge regression [26]. For shift-invariant kernels, random features allow linear SVMs to approximate kernel SVMs without forming the full kernel matrix [24]. Function approximation theory provides another justification for fixing the hidden parameters. Under suitable activation and sampling assumptions, a linear combination of random features can approximate a target function to a prescribed accuracy with high probability [12, 22, 27]. The suficient width depends on the target function class, the required accuracy, and the input dimension. In some bounds, the dependence on dimension is exponential or superexponential.

Our results give a geometric guarantee for this approach. With high probability, the two fixed random hidden layers admit an afine output function that separates the entire classes with a positive margin. Only the output coeficients need to be chosen after sampling; the hidden weights and biases remain fixed.

## 4 Numerical Experiments on Gaussian-Width Dependence

We examine how the Gaussian width in Theorems 2.5 and 2.7 relates to the widths needed for certified separation. The experiments use four convex cells arranged in an XOR pattern. The first varies Gaussian width while fixing the intrinsic span dimension; the second varies the intrinsic span dimension while keeping Gaussian width nearly fixed.

## 4.1 Geometry and experimental setting

Work in $\mathbb { R } ^ { d }$ with $d = 1 0 2 4$ . Set $a = 1 / 1 0$ , take $2 \leq r \leq d - 1$ , and let $0 < s \le 1$

$$
Q _ { r , s } = \left\{ a \left( u _ { 1 } e _ { 3 } + s \sum _ { \ell = 2 } ^ { r - 1 } u _ { \ell } e _ { \ell + 2 } \right) : u \in B _ { 2 } ^ { r - 1 } \right\} .
$$

Here $B _ { 2 } ^ { r - 1 }$ is the unit Euclidean ball in $\mathbb { R } ^ { r - 1 }$ . The set $Q _ { r , s }$ is an ellipsoidal disk with semiaxis a along $e _ { 3 }$ and semiaxes as along $e _ { 4 } , \ldots , e _ { r + 1 }$ . Set

$$
\begin{array} { r l } & { E _ { 1 } ^ { + } = e _ { 1 } + \frac { 1 } { 2 } e _ { 2 } + Q _ { r , s } , E _ { 2 } ^ { + } = - e _ { 1 } - \frac { 1 } { 2 } e _ { 2 } + Q _ { r , s } , } \\ & { E _ { 1 } ^ { - } = e _ { 1 } - \frac { 1 } { 2 } e _ { 2 } + Q _ { r , s } , E _ { 2 } ^ { - } = - e _ { 1 } + \frac { 1 } { 2 } e _ { 2 } + Q _ { r , s } , } \end{array} \qquad S ^ { \pm } = E _ { 1 } ^ { \pm } \cup E _ { 2 } ^ { \pm } .
$$

Each component is a compact convex ellipsoidal disk of afine dimension $r - 1$ . For every $q \in Q _ { r , s }$ each class contains two points whose midpoint is q. Hence $Q _ { r , s } \subset \mathrm { c o n v } ( S ^ { + } )$ ∩ conv(S<sup>−</sup>), so no afine hyperplane strictly separates the classes. Both experiments keep

$$
N _ { + } = N _ { - } = 2 , \qquad \delta _ { 0 } = 1 , \qquad R = { \frac { \sqrt { 5 } } { 2 } } , \qquad \kappa _ { \mathrm { m i n } } = { \frac { 1 } { \sqrt { 1 + 4 a ^ { 2 } } } } .\tag{20}
$$

The four cell centers are $c = \pm e _ { 1 } \pm \frac { 1 } { 2 } e _ { 2 }$ , and each is orthogonal to $\operatorname { s p a n } ( Q _ { r , s } )$ . Thus the projection of the origin onto each cell is its center, giving $R = { \sqrt { 5 } } / 2$ . For every opposite-label pair, the center diference is orthogonal to span $\left( Q _ { r , s } \right)$ and has length 1 or 2. Since $0 \in Q _ { r , s }$ , the cell distance equals the center-diference length, and therefore $\delta _ { 0 } = 1$ . Each diference cone spans the r-dimensional space generated by its center diference and span $\left( Q _ { r , s } \right)$ . The pairs $( E _ { 1 } ^ { + } , E _ { 1 } ^ { - } )$ and $( E _ { 2 } ^ { + } , E _ { 2 } ^ { - } )$ attain $\omega _ { * }$ . By rotational invariance of Gaussian width, it sufices to compute this width on the spherical section of cl $\mathrm { c o n e } ( e _ { 1 } + 2 Q _ { r , s } )$

We use the network in (5) with first-layer bias radius $\lambda = 4$ and second-layer bias radius $\widehat { \lambda } = 1 = \delta _ { 0 }$ , as prescribed for the second layer in Theorem 2.7. These bias choices are fixed across all configurations. The first-layer radius is an experimental choice; the theorem’s universal constant is not evaluated numerically, so the experiment does not test its explicit suficient-bias threshold. In each trial, we sample and freeze the random layers, then fit afine readouts on the resulting features. We certify margins over each entire ellipsoidal cell using afine bounds for ReLU and outward-rounded arithmetic; a trial counts as successful only when all required certified margins are strictly positive. For the first-layer criterion, each $\Phi ( E _ { j } ^ { - } )$ must be separated with positive margin from $\Phi ( S ^ { + } )$ ; the separating hyperplane may depend on j. We count a two-layer trial as a joint success only when this first-layer criterion holds and a common positive-margin afine separator exists for $F ( S ^ { + } )$ and $F ( S ^ { - } )$ . The empirical final readout allows an ofset, whereas Theorem 2.7 provides a separator with zero output ofset; the reported final-layer thresholds therefore describe the broader afine-readout criterion and illustrate width trends rather than directly test the theorem’s zero-ofset guarantee.

We use nonuniform candidate-width grids, with finer spacing at smaller widths; the first- and second-layer grids range from 8 to 1024 and from 8 to 4096, respectively. All settings and candidate widths are fixed before running 200 random trials per configuration. We use the same seed list across configurations, so the comparisons are paired. Among tested width pairs with at least 190 joint successes, we select the pair minimizing $n + \widehat { n } _ { \mathrm { { ; } } }$ , breaking ties by smaller n. The selected first-layer width can therefore exceed the standalone first-layer 95% threshold. The bootstrap intervals in Figure 5 are obtained by resampling the paired seeds and reselecting the width pair.

## 4.2 Gaussian width and intrinsic dimension

At fixed $r = 1 2 8$ , we vary s to change $\omega _ { * } ^ { 2 }$ while keeping the parameters in (20) fixed. Table 2 reports the smallest tested first-layer width achieving at least 190 successes out of 200 trials. The ratio of this threshold to $1 + \omega _ { * } ^ { 2 }$ ranges from 6.86 to 8.00. This near-proportional trend is consistent with the complexity dependence in (8); it does not estimate the theorem’s unspecified constant.

(a) Increasing Gaussian width, r= 128  
Table 2: Empirical thresholds at $r = 1 2 8$ . The first-layer column gives the smallest tested width with at least 190/200 successes. The selected pair minimizes $n + \widehat { n }$ subject to the joint-success criterion; $\omega _ { * } ^ { 2 }$ is rounded.
<table><tr><td>S</td><td> $\omega _ { * } ^ { 2 }$ </td><td>First layer alone</td><td> $( n , \widehat { n } )$ </td><td>Joint successes</td></tr><tr><td>0.06219</td><td>0.05</td><td>8</td><td>(8,12)</td><td>190/200</td></tr><tr><td>0.30414</td><td>0.50</td><td>12</td><td>(16,24)</td><td>190/200</td></tr><tr><td>0.43946</td><td>1.00</td><td>16</td><td>(20, 40)</td><td>190/200</td></tr><tr><td>0.54280</td><td>1.50</td><td>20</td><td>(24, 80)</td><td>192/200</td></tr><tr><td>0.63007</td><td>2.00</td><td>24</td><td>(40, 112)</td><td>193/200</td></tr><tr><td>0.70724</td><td>2.50</td><td>24</td><td>(56, 160)</td><td>193/200</td></tr><tr><td>0.77732</td><td>3.00</td><td>28</td><td>(56, 256)</td><td>191/200</td></tr></table>

Figure 5(a) and Table 2 report the selected width pairs. The selected widths generally increase with $\omega _ { * } ^ { 2 }$ , with a more pronounced increase in the second layer. Because r and the other parameters in (20) are fixed, this variation is consistent with the Gaussian-width dependence in (12).

For Figure 5(b), we take $r = 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2$ and adjust s so that $\omega _ { * } ^ { 2 } \approx 0 . 5$ . The standalone first-layer threshold is 12 for every tested r. Each selected pair is (16, 24), with 190–192 joint successes out of 200. Thus, the cone span dimension increases by a factor of 16, while the selected widths remain unchanged on this grid. As r grows, reducing the transverse semiaxes keeps the Gaussian width nearly constant. These results are consistent with the width bounds’ dependence on Gaussian width rather than span dimension alone: for fixed Gaussian width and the other geometric parameters, the suficient bounds do not increase with r.

Taken together, these experiments support Gaussian width as a more informative measure of separation complexity than span dimension alone. The term $1 + \omega _ { * } ^ { 2 }$ in (8) and (12) reflects the efect of set geometry on the observed network widths. In particular, high intrinsic dimension need not require wide networks when Gaussian width and the other parameters remain fixed.

![](images/8f87e7c2e1ee22446f40840202cda1acd0800fc671454a156e7e2b64c08247ea.jpg)

![](images/ee7a34427bd5766ebb04b0480c0bf4a4fcc1a76c75146f30860cb53f8071645e.jpg)  
Figure 5: Empirical width pairs based on whole-set separation certificates, selected by minimizing $n + \widehat { n }$ at 95% joint success. Blue diamonds show n; orange triangles show nb. (a) Varying $\omega _ { * } ^ { 2 }$ at fixed $r = 1 2 8$ . (b) Varying r at nearly fixed $\omega _ { * } ^ { 2 } = 0 . 5$ . Bars show paired bootstrap 95% intervals with pair reselection. Lines connect tested configurations.

## 5 Conclusions

This paper establishes geometric conditions for separation by ReLU networks. Positive-margin separation by a finite ReLU network is equivalent to the existence of finite closed convex covers whose opposite-label sets are pairwise at positive distance. Under this condition, two hidden layers always sufice, although one hidden layer may fail. For random networks, pointedness of the diference cones provides a suficient geometric condition, including for unbounded classes. The first layer separates each $\Phi ( E _ { j } ^ { - } )$ from $\Phi ( S ^ { + } )$ and controls all cross-class distances. The second layer combines these cellwise separations into a single positive-margin separator.

The suficient width of each layer is governed by the squared Gaussian widths of the spherical sections of the diference cones. With the cover sizes, remaining geometric parameters, bias ratios, and failure probability fixed, the width bounds grow at most linearly with the intrinsic span dimension. A linearly inseparable family of translated cones and singletons yields a matching lower bound for each layer. Thus, the quadratic dependence on Gaussian width is optimal uniformly over this family. The experiments use XOR classes that are not linearly separable in the input space. Increasing Gaussian width raises the certified thresholds, although the second-layer thresholds do not exhibit approximately linear growth. At nearly fixed Gaussian width, increasing the intrinsic dimension from 32 to 512 leaves both selected widths unchanged.

The counterexamples also clarify the scope of the assumptions. The shallow obstruction shows that two hidden layers are necessary for some classes. The nonpointed-strip example shows that positive distance alone does not guarantee random separation: every fixed finite-depth, finite-width Gaussian ReLU network fails to separate these strips almost surely, even though a deterministic ReLU network separates them. Thus, pointedness cannot simply be omitted from a general random-separation theorem.

## A Proof of the first-layer results

This appendix proves the first-layer separation result and provides the estimates needed for the second layer. Section A.1 derives uniform linear separation margins from the class gaps and pointedness of the diference cones. It also bounds the Gaussian width of $\{ ( x , \delta _ { 0 } ) / \sqrt { \| x \| ^ { 2 } + \delta _ { 0 } ^ { 2 } : x \in S ^ { + } \cup S ^ { - } \} }$ Section A.2 combines the pairwise separation conditions using a polynomial and represents it as an expectation of random ReLU features. Section A.3 uses truncation and error estimates to obtain a finite-width approximation. Section A.4 constructs hyperplanes separating each $\Phi ( E _ { i } ^ { - } )$ from $\Phi ( S ^ { + } )$ and bounds the angular margin and cross-class distances. Section A.5 bounds $\| \Phi ( x ) \|$ and the Gaussian width of $\{ \Phi ( x ) / \| \Phi ( x ) \| : x \in S ^ { + } \cup S ^ { - } \}$ , and combines the failure probabilities from all steps.

Probabilities in statements about the first layer are taken under the joint distribution of $( W , b )$ Probabilities in statements about the two-layer network are taken under the joint distribution of $( W , b , \widehat { W } , \widehat { b } )$ . For conditional probabilities in the proofs, we specify which variables are fixed.

## A.1 Separation margins and Gaussian width

In this subsection, we derive uniform linear separation margins for $( x , \delta _ { 0 } )$ and bound the Gaussian width of $\{ ( x , \delta _ { 0 } ) / \sqrt { \| x \| ^ { 2 } + \delta _ { 0 } ^ { 2 } : x \in S ^ { + } \cup S ^ { - } \} }$ . The class gaps control separation at bounded scales, while pointedness controls separation along unbounded directions. Together, they yield a margin

proportional to $\sqrt { \| x \| ^ { 2 } + \delta _ { 0 } ^ { 2 } }$ . To express these estimates, for any $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , define

$$
U _ { x } = ( x , \delta _ { 0 } ) \in \mathbb { R } ^ { d + 1 } , \qquad \varepsilon = \frac { \kappa _ { \operatorname* { m i n } } } { 1 6 ( 1 + R / \delta _ { 0 } ) ^ { 2 } } .\tag{21}
$$

When the argument is denoted by y, write $U _ { y } = ( y , \delta _ { 0 } )$ accordingly.

Lemma A.1. There exists a vector with $\| q _ { i j } \| \leq 1$ such that

$$
\langle q _ { i j } , U _ { x } \rangle \geq \varepsilon \| U _ { x } \| \quad ( x \in E _ { i } ^ { + } ) , \qquad \langle q _ { i j } , U _ { y } \rangle \leq - \varepsilon \| U _ { y } \| \quad ( y \in E _ { j } ^ { - } ) .\tag{22}
$$

Proof. Apply (6) from Proposition 2.4(ii) with $K _ { i j }$ . For a pair of sets $E _ { i } ^ { + } , E _ { j } ^ { - }$ , let

$$
p _ { i j } = P _ { \mathrm { c l } ( E _ { i } ^ { + } - E _ { j } ^ { - } ) } ( 0 ) , \qquad v _ { i j } = p _ { i j } / \lVert p _ { i j } \rVert .
$$

The projection onto the closed diference set exists, and $\| p _ { i j } \| = \mathrm { d i s t } ( E _ { i } ^ { + } , E _ { j } ^ { - } ) \geq \delta _ { 0 }$ . The projection inequality gives

$$
\begin{array} { r } { \langle v _ { i j } , x - y \rangle \geq \delta _ { 0 } \qquad ( x \in E _ { i } ^ { + } , ~ y \in E _ { j } ^ { - } ) . } \end{array}
$$

To center the threshold for each direction, for $a = u _ { i j }$ or $a = v _ { i j }$ write

$$
\tau _ { a } = \frac { 1 } { 2 } \left( \operatorname* { i n f } _ { E _ { i } ^ { + } } \langle a , x \rangle + \operatorname* { s u p } _ { E _ { j } ^ { - } } \langle a , y \rangle \right) .
$$

Comparison with the two anchors shows that these infima and suprema are fini ${ \mathrm { , e , } }$ and $| \tau _ { a } | \leq R$ . For $x \in E _ { i } ^ { + }$ , the metric direction gives $\langle v _ { i j } , x \rangle - \tau _ { v _ { i j } } \geq \delta _ { 0 } / 2 ;$ also, $\langle u _ { i j } , x \rangle - \tau _ { { u } _ { i j } } > 0$ , and $\langle u _ { i j } , x \rangle - \tau _ { u _ { i j } } \geq$ $\kappa _ { \operatorname* { m i n } } \| x \| - 3 R$ . Their sum is therefore at least

$$
\operatorname* { m a x } \{ \delta _ { 0 } / 2 , \ \kappa _ { \mathrm { m i n } } \| x \| - 3 R \} .
$$

The argument for $y \in E _ { j } ^ { - }$ is the same. Set

$$
q _ { i j } = \frac { ( u _ { i j } + v _ { i j } , - ( \tau _ { u _ { i j } } + \tau _ { v _ { i j } } ) / \delta _ { 0 } ) } { 2 \sqrt { 1 + ( R / \delta _ { 0 } ) ^ { 2 } } } ,
$$

so that $\| q _ { i j } \| \leq 1$ . The lower bound for ${ \frac { \langle q _ { i j } , U _ { x } \rangle } { \left\| U _ { x } \right\| } }$ attains its minimum when $\delta _ { 0 } / 2 = \kappa _ { \mathrm { m i n } } \| x \| - 3 R$ Hence,

$$
\frac { \kappa _ { \operatorname* { m i n } } } { 2 \sqrt { 1 + ( R / \delta _ { 0 } ) ^ { 2 } } \sqrt { 4 \kappa _ { \operatorname* { m i n } } ^ { 2 } + ( 6 R / \delta _ { 0 } + 1 ) ^ { 2 } } } \geq \frac { \kappa _ { \operatorname* { m i n } } } { 1 2 ( 1 + R / \delta _ { 0 } ) ^ { 2 } } .
$$

The last inequality uses $0 < \kappa _ { \mathrm { m i n } } \le 1$ . Thus (22) holds.

The subsequent uniform approximation requires control of the complexity of $U _ { x } / \Vert U _ { x } \Vert$ . Define the compact set

$$
T _ { \delta _ { 0 } } = \mathrm { c l } \left\{ \frac { U _ { x } } { \lVert U _ { x } \rVert } : x \in S ^ { + } \cup S ^ { - } \right\} .
$$

Its closure includes limiting directions of unbounded input sequences. The following width estimate relates its complexity to the diference cones.

Use the anchor $y _ { 1 } ^ { \circ }$ for $S ^ { + }$ and $x _ { 1 } ^ { \circ }$ for $S ^ { - }$ to decompose each input into an anchor and a vector in a diference cone. For example, if $x \in E _ { i } ^ { + }$ , take $z = ( x - y _ { 1 } ^ { \circ } ) / \| x - y _ { 1 } ^ { \circ } \| \in K _ { i 1 } \cap S ^ { d - 1 }$ . Then

$$
\frac { U _ { x } } { \| U _ { x } \| } = \frac { ( y _ { 1 } ^ { \circ } , \delta _ { 0 } ) } { \| U _ { x } \| } + \frac { \| x - y _ { 1 } ^ { \circ } \| } { \| U _ { x } \| } ( z , 0 ) .
$$

The coeficient satisfies $\| x - y _ { 1 } ^ { \circ } \| / \| U _ { x } \| \leq 1 + R / \delta _ { 0 }$ . For $y \in E _ { j } ^ { - }$ , using $x _ { 1 } ^ { \circ }$ and $- K _ { 1 j }$ gives the same estimate. For any nonempty subset A of the unit sphere, fixed $a \in A$ , and standard Gaussian vector $g$ of the same dimension, basic properties of Gaussian width [30, Proposition 7.5.2 and Exercise 7.20] give

$$
\mathbb { E } _ { g } \operatorname* { s u p } _ { u \in A } | \langle g , u \rangle | \leq \mathbb { E } _ { g } \operatorname* { s u p } _ { u , u ^ { \prime } \in A } \langle g , u - u ^ { \prime } \rangle + \mathbb { E } _ { g } | \langle g , a \rangle | = 2 w ( A ) + \sqrt { 2 / \pi } .
$$

Taking $g \sim N ( 0 , I _ { d + 1 } )$ and applying this comparison to the decomposition above yields

$$
\begin{array} { r l } & { w ( T _ { \delta _ { 0 } } ) \leq \mathbb { E } _ { g } \underset { u \in T _ { \delta _ { 0 } } } { \operatorname* { s u p } } | \langle g , u \rangle | } \\ & { \qquad \leq \sqrt { 2 / \pi } ( 1 + 2 R / \delta _ { 0 } ) + ( 1 + R / \delta _ { 0 } ) \left[ 2 w \left( \mathrm { c l } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) \right) + \sqrt { 2 / \pi } \right] . } \end{array}
$$

Hence

$$
w ( T _ { \delta _ { 0 } } ) \leq C ( 1 + R / \delta _ { 0 } ) \left[ 1 + w \left( { \mathrm { c l } } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) \right) \right] .\tag{23}
$$

The constant term in (23) comes from the last coordinate $\delta _ { 0 }$ of $U _ { x }$ and the two anchors.

## A.2 From set margins to random feature representations

For each fixed $j \in [ N _ { - } ]$ , we combine the pairwise separation margins into a single function that distinguishes $E _ { j } ^ { - }$ from $S ^ { + }$ . We construct this function using a polynomial, and then represent it as an expectation of Gaussian ReLU features. By bounding the polynomial degree and the representation coeficients, we obtain the estimates needed for the finite-width approximation in the next subsection.

Lemma A.2 (A gap polynomial preserving parity). Let $0 < \varepsilon < 1$ and $N \geq 1$ . There exists a polynomial P containing only a linear term and even powers, of degree D satisfying $c \varepsilon ^ { - 1 } \log ( 4 N ) \leq$ $D \le C \varepsilon ^ { - 1 } \log ( 4 N )$ , such that

$$
0 \leq P ( t ) \leq 1 , \quad P ( t ) - P ( - t ) = t \quad ( - 1 \leq t \leq 1 ) , \qquad P ( - \varepsilon ) \leq { \frac { \varepsilon } { 1 6 N } } .\tag{24}
$$

Moreover, P is nondecreasing on $[ - 1 , 1 ]$ , and the sum of the absolute values of its nonzero coeficients is at most $e ^ { C D }$

Proof. Take the smallest odd integer L at least $4 / \varepsilon .$ , and $p = \lceil \log _ { 4 } ( 1 6 N ) \rceil \geq 2$ . Let $\mathsf { T } _ { L }$ denote the Chebyshev polynomial of the first kind. In this proof, define

$$
B ( t ) = \left[ { \frac { \mathsf { T } _ { L } ( t ) } { L t } } \right] ^ { 2 p } , \qquad I = \int _ { 0 } ^ { 1 } B ( s ) d s , \qquad S ( t ) = I ^ { - 1 } \int _ { 0 } ^ { t } B ( s ) d s ,
$$

$$
P ( t ) = \frac { 1 } { 2 } \int _ { - 1 } ^ { t } \left[ 1 + { \cal S } ( s ) \right] d s .
$$

Use the continuous extension at zero. Since L is odd, B is a nonnegative even polynomial, and S is an odd, nondecreasing polynomial with $S ( \pm 1 ) = \pm 1$ . Thus $P ^ { \prime } \in [ 0 , 1 ] \mathrm { o n } [ - 1 , 1 ] , P ( - 1 ) = 0$ $P ( 1 ) = 1$ , and $P ^ { \prime } ( t ) + P ^ { \prime } ( - t ) = 1$ . This proves the first two properties in (24) and the required parity structure. The degree is $D = 2 p ( L - 1 ) + 2$ , and $D$ is comparable to $\varepsilon ^ { - 1 } \log ( 4 N )$ up to absolute constants.

The trigonometric identity for Chebyshev polynomials gives

$$
| \mathsf { T } _ { L } ( t ) | = | \sin ( L \arcsin t ) | \geq L t / 2
$$

for $0 \leq t \leq 1 / ( 2 L )$ ; also, $| \mathsf T _ { L } ( t ) | \leq 1$ for $| t | \leq 1$ . Hence

$$
I \geq 2 ^ { - 2 p - 1 } / L , \qquad B ( t ) \leq ( L t ) ^ { - 2 p } \quad ( 0 < t \leq 1 ) .
$$

Using $\begin{array} { r } { P ( - \varepsilon ) = ( 2 I ) ^ { - 1 } \int _ { \varepsilon } ^ { 1 } ( s - \varepsilon ) B ( s ) } \end{array}$ ds, we obtain

$$
P ( - \varepsilon ) \leq \frac { 2 ^ { 2 p } \varepsilon } { ( 2 p - 1 ) ( 2 p - 2 ) ( L \varepsilon ) ^ { 2 p - 1 } } \leq \varepsilon 4 ^ { - p } \leq \frac { \varepsilon } { 1 6 N } .
$$

Finally, the Chebyshev recurrence bounds the sum of the absolute values of the power-basis coeficients of $\mathsf { T } _ { L }$ by $3 ^ { L }$ . Submultiplicativity of this coeficient sum under polynomial multiplication bounds the coeficient sum of $B$ by $3 ^ { 2 p L }$ . The first integration does not increase the coeficient sum, so that of $S$ is at most

$$
I ^ { - 1 } 3 ^ { 2 p L } \leq L 2 ^ { 2 p + 1 } 3 ^ { 2 p L } .
$$

Next write $\begin{array} { r } { P ( t ) = P ( 0 ) + \int _ { 0 } ^ { t } P ^ { \prime } ( s ) d s } \end{array}$ . The second integration does not increase the coeficient sum of the nonconstant terms. Since $0 \leq P ( 0 ) \leq 1$ , the coeficient sum of $P$ is at most

$$
2 + L 2 ^ { 2 p + 1 } 3 ^ { 2 p L } \leq e ^ { C D } ,
$$

The last step uses $D \asymp p L$ and a suficiently large positive constant $C _ { i }$ , independent of the sets and η.

In Lemma A.2, take $N = 3 2 N _ { + } ^ { 2 }$ and retain $\varepsilon$ from (21). The resulting $P$ satisfies $0 \leq P ^ { \prime } \leq 1$ on [−1, 1] and

$$
P ( - \varepsilon ) \leq \frac { \varepsilon } { 5 1 2 N _ { + } ^ { 2 } } , \qquad c \varepsilon ^ { - 1 } \log ( 4 N _ { + } ) \leq D \leq C \varepsilon ^ { - 1 } \log ( 4 N _ { + } ) .\tag{25}
$$

For $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , use $U _ { x }$ from Lemma A.1 to define

$$
h _ { j } ( U _ { x } ) = \| U _ { x } \| \left[ \sum _ { i } P \Big ( \frac { \langle q _ { i j } , U _ { x } \rangle } { \| U _ { x } \| } \Big ) - \frac { \varepsilon } { 2 } \right] .\tag{26}
$$

Since $\| q _ { i j } \| \leq 1$ , all arguments lie in [−1, 1]. For $x \in S ^ { + }$ , at least one argument is at least $\varepsilon ,$ and $P ( t ) \geq t$ for $t \geq 0 \mathrm { : }$ ; for $y \in E _ { j } ^ { - }$ <sup>−</sup>, all arguments are at most $- \varepsilon$ . Thus

$$
\begin{array} { r } { h _ { j } ( U _ { x } ) \geq \frac { \varepsilon } { 2 } \| U _ { x } \| \quad \quad \quad \quad \quad \quad ( x \in S ^ { + } ) , } \\ { \quad \quad \quad \quad \quad \frac { \varepsilon } { 2 } \| U _ { y } \| \leq h _ { j } ( U _ { y } ) \leq \left( - \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 5 1 2 N _ { + } } \right) \| U _ { y } \| \quad ( y \in E _ { j } ^ { - } ) . } \end{array}\tag{27}
$$

For every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , we have $0 \leq h _ { j } ( U _ { x } ) / \Vert U _ { x } \Vert + \varepsilon / 2 \leq N _ { + }$

We next express $h _ { j }$ as an expectation of random ReLU features, so that finitely many neurons can approximate it. For the probabilists’ Hermite polynomials, write

$$
\mu _ { \ell } = \mathbb { E } _ { G } \operatorname { H e } _ { \ell } ( G ) \operatorname { R e L U } ( G ) , \qquad G \sim N ( 0 , 1 ) .
$$

Two integrations by parts give

$$
\mu _ { 0 } = ( 2 \pi ) ^ { - 1 / 2 } , \quad \mu _ { 1 } = 1 / 2 , \quad \mu _ { 2 s } = ( - 1 ) ^ { s - 1 } ( 2 s - 3 ) ! ! / \sqrt { 2 \pi } \quad ( s \geq 1 ) .
$$

Odd coeficients above degree one vanish. Thus the restriction to a linear term and even powers ensures that every denominator in the following representation is nonzero. If $\begin{array} { r } { P ( t ) = \sum _ { \ell = 0 } ^ { D } p _ { \ell } t ^ { \ell } } \end{array}$ , then for $q \in \mathbb { R } ^ { d + 1 }$ with $\| q \| \leq 1$ , define

$$
\mathcal { H } _ { q , D } ( V ) = \sum _ { \ell \in \{ 0 \} \cup \atop \mu _ { \ell } \neq 0 } \frac { p _ { \ell } \| q \| ^ { \ell } } { \mu _ { \ell } } \mathrm { H e } _ { \ell } ( \langle q / \| q \| , V \rangle ) , \qquad V \sim N ( 0 , I _ { d + 1 } ) .
$$

For $q = 0$ , define $\mathcal { H } _ { 0 , D } ( V ) = p _ { 0 } / \mu _ { 0 }$ . Fix $x \in \mathbb { R } ^ { d }$ and $q .$ First condition on $ V , U _ { x } / | | U _ { x } | | \rangle$ and average the generating function over the orthogonal Gaussian component $P _ { U _ { x } ^ { \perp } } V$ . Then average over $\left. V , U _ { x } / \lVert U _ { x } \rVert \right. \sim N ( 0 , 1 )$ to obtain

$$
\begin{array} { r } { \mathbb E _ { V } \mathcal H _ { q , D } ( V ) \mathrm { R e L U } ( \langle V , U _ { x } \rangle ) = \| U _ { x } \| P ( \langle q , U _ { x } \rangle / \| U _ { x } \| ) . } \end{array}\tag{28}
$$

Approximating this expectation by finite samples requires moment bounds on its coeficients. For $G \sim N ( 0 , 1 )$ , Gaussian hypercontractivity [15, Theorem 5.1] (see also [23]) gives

$$
\| \mathrm { H e } _ { \ell } ( G ) \| _ { L ^ { 4 } } \leq 3 ^ { \ell / 2 } \| \mathrm { H e } _ { \ell } ( G ) \| _ { L ^ { 2 } } = 3 ^ { \ell / 2 } { \sqrt { \ell ! } } .
$$

Applying these estimates term by term and using the power-basis coeficient sum, after adjusting the positive constant $C ,$ , independent of the sets and η, gives

$$
\| \mathcal { H } _ { q , D } \| _ { L ^ { 4 } } \leq C e ^ { C D } 3 ^ { D / 2 } ( D + 1 ) ^ { 3 / 2 } \leq e ^ { C D } .
$$

For $x \in S ^ { + } \cup S ^ { - }$ , since $\langle V , U _ { x } \rangle \sim N ( 0 , \lVert U _ { x } \rVert ^ { 2 } )$ , we have

$$
\Vert U _ { x } \Vert = \sqrt { \Vert x \Vert ^ { 2 } + \delta _ { 0 } ^ { 2 } } = \sqrt { 2 \pi } \mathbb { E } _ { V } \operatorname { R e L U } ( \langle V , U _ { x } \rangle ) .
$$

Thus the expectation representation of $\| U _ { x } \|$ has the constant coeficient $\sqrt { 2 \pi }$ . Let

$$
{ \mathcal { H } } _ { j } ( V ) = \sum _ { i } { \mathcal { H } } _ { q _ { i j } , D } ( V ) - ( \varepsilon / 2 ) { \sqrt { 2 \pi } } .
$$

Since $D \geq c \log ( 4 N _ { + } )$ , the factor $N _ { + }$ from summation can be absorbed into $e ^ { C D }$ . Hence $\| \mathcal { H } _ { j } \| _ { L ^ { 4 } } \leq$ $e ^ { C D }$ and

$$
\begin{array} { r } { \mathbb { E } _ { V } \mathcal { H } _ { j } ( V ) \mathrm { R e L U } ( \langle V , U _ { x } \rangle ) = h _ { j } ( U _ { x } ) . } \end{array}
$$

## A.3 Truncation and finite-sample approximation

This subsection derives finite-width approximations of $h _ { j } ( U _ { x } )$ and $\| U _ { x } \|$ from the representations in (28). Truncation and a change of measure adapt these representations to the network’s uniform biases and give bounded coeficients. We then control the truncation and sampling errors uniformly relative to $\| U _ { x } \|$ over both classes.

Write $V = ( g , h )$ , where $g \sim N ( 0 , I _ { d } )$ and $h \ \sim \ N ( 0 , 1 )$ are independent, and let $\varphi ( t ) =$ $( 2 \pi ) ^ { - 1 / 2 } e ^ { - t ^ { 2 } / 2 }$ . We truncate h and the representation coeficients. To bound each target’s total relative error by $\varepsilon / ( 5 1 2 N _ { + } )$ , take

$$
\begin{array} { l } { { \displaystyle K = \frac { 4 0 9 6 N _ { + } } { \varepsilon } \operatorname* { m a x } \{ 1 , 2 \pi , \operatorname* { m a x } _ { j } \lVert \mathcal { H } _ { j } \rVert _ { L ^ { 4 } } ^ { 2 } \} , } \ ~ } \\ { { \displaystyle c _ { 0 } = \operatorname* { m a x } \left\{ 1 , \sqrt { 2 \log \left[ 1 2 8 \sqrt { 3 } \left( \frac { 5 1 2 N _ { + } } { \varepsilon } \right) ^ { 2 } \operatorname* { m a x } \{ 1 , 2 \pi , \operatorname* { m a x } _ { j } \lVert \mathcal { H } _ { j } \rVert _ { L ^ { 4 } } ^ { 2 } \} \right] } \right\} } . } \end{array}\tag{29}
$$

The term $2 \pi$ ensures $K > { \sqrt { 2 \pi } }$ , so the constant coeficient in the representation of $\| U _ { x } \|$ is not truncated. Its truncation error therefore comes only from restricting $| h | \leq c _ { 0 }$ . For $x \in S ^ { + } \cup S ^ { - }$ and $j \in [ N _ { - } ]$ , Hölder’s inequality and elementary truncation estimates give

$$
\begin{array} { r l } & { \mathbb { E } _ { V } | \mathcal { H } _ { j } ( V ) | \mathrm { R e L U } ( \langle V , U _ { x } \rangle ) \mathbf { 1 } _ { \{ | h | > c _ { 0 } \} } \leq 3 ^ { 1 / 4 } \| \mathcal { H } _ { j } \| _ { L ^ { 4 } } \| U _ { x } \| \sqrt { \mathbb { P } ( | h | > c _ { 0 } ) } , } \\ & { \mathbb { E } _ { V } | \mathcal { H } _ { j } ( V ) | \mathrm { R e L U } ( \langle V , U _ { x } \rangle ) \mathbf { 1 } _ { \{ | \mathcal { H } _ { j } | > K \} } \leq \frac { \| \mathcal { H } _ { j } \| _ { L ^ { 4 } } ^ { 2 } } { \sqrt { 2 } K } \| U _ { x } \| . } \end{array}
$$

By the Gaussian tail bound and (29), these two terms are bounded respectively by

$$
\frac { 1 } { 8 } \frac { \varepsilon } { 5 1 2 N _ { + } } \| U _ { x } \| , \qquad \frac { 1 } { 8 \sqrt { 2 } } \frac { \varepsilon } { 5 1 2 N _ { + } } \| U _ { x } \| .
$$

The same estimate for truncating the last Gaussian coordinate applies to the expectation representation of $\| U _ { x } \|$

Since log $( N _ { + } / \varepsilon ) \le C D$ and ma $\mathrm { x } _ { j } \| \mathcal { H } _ { j } \| _ { L ^ { 4 } } \leq e ^ { C D }$ 2

$$
K \leq e ^ { C D } , \qquad 1 \leq c _ { 0 } ^ { 2 } \leq C D .\tag{30}
$$

These thresholds depend on the geometric parameters and deterministic coeficient moments and are fixed before sampling the network. Since all $\mathcal { H } _ { j }$ satisfy the same moment bound, the threshold sizes are uniform over $j \in [ N _ { - } ]$

Let $\beta \sim \mathrm { U n i f } [ - c _ { 0 } , c _ { 0 } ]$ be independent of $g \sim N ( 0 , I _ { d } )$ , and set

$$
B _ { j } ( g , \beta ) = 2 c _ { 0 } \varphi ( \beta ) \mathcal { H } _ { j } ( g , \beta ) \mathbf { 1 } _ { \{ | \mathcal { H } _ { j } ( g , \beta ) | \leq K \} } , \qquad | B _ { j } | \leq c _ { 0 } K .
$$

The density ratio $2 c _ { 0 } \varphi ( \beta )$ rewrites the uniform-bias integral as a truncated Gaussian integral. For all $x \in S ^ { + } \cup S ^ { - }$ ，

$$
\begin{array} { r } { \mathbb { E } _ { g , \beta } [ B _ { j } ( g , \beta ) \operatorname { R e L U } ( \langle ( g , \beta ) , U _ { x } \rangle ) ] = \mathbb { E } _ { V } \Big [ \mathcal { H } _ { j } ( V ) \mathbf { 1 } _ { \{ | h | \leq c _ { 0 } , ~ | \mathcal { H } _ { j } ( V ) | \leq K \} } \operatorname { R e L U } ( \langle V , U _ { x } \rangle ) \Big ] . } \end{array}
$$

For the expectation representation of $\| U _ { x } \|$ , the same change of measure gives

$$
\begin{array} { r } { \mathbb { E } _ { g , \beta } \Big [ 2 c _ { 0 } \varphi ( \beta ) \sqrt { 2 \pi } \operatorname { R e L U } ( \langle ( g , \beta ) , U _ { x } \rangle ) \Big ] = \sqrt { 2 \pi } \mathbb { E } _ { V } \Big [ \mathbf { 1 } _ { \{ | h | \leq c _ { 0 } \} } \operatorname { R e L U } ( \langle V , U _ { x } \rangle ) \Big ] . } \end{array}
$$

The corresponding coeficient for the uniform-bias representation satisfies $2 c _ { 0 } \varphi ( \beta ) \sqrt { 2 \pi } \le 2 c _ { 0 } \le c _ { 0 } K$

Lemma A.3 (An empirical estimate with bounded row coeficients). Let $v = ( g , \beta )$ and $\begin{array} { r } { v _ { a } = ( g _ { a } , \beta _ { a } ) } \end{array}$ $( a \in [ M ] )$ be independent and identically distributed, where $g \sim N ( 0 , I _ { d } )$ and $\beta \sim \mathrm { U n i f } [ - c _ { 0 } , c _ { 0 } ]$

are independent and $c _ { 0 } \geq 1$ . For a deterministic nonempty compact set $T \subset S ^ { d }$ and $B _ { j }$ satisfying $| B _ { j } | \le B$ for $j \in [ J ]$ , with probability at least $1 - \xi$

$$
\begin{array} { r l r } {  { \operatorname* { m a x } _ { j \in [ J ] } \operatorname* { s u p } _ { u \in T } | \frac { 1 } { M } \sum _ { a = 1 } ^ { M } B _ { j } ( v _ { a } ) \operatorname { R e I U } \bigl ( \langle v _ { a } , u \rangle \bigr ) - \mathbb { E } _ { v } B _ { j } ( v ) \operatorname { R e L U } \bigl ( \langle v , u \rangle \bigr ) | } } \\ & { } & \\ & { } & { \leq \frac { C B } { \sqrt { M } } [ w ( T ) + c _ { 0 } \sqrt { \log \frac { 2 e J } { \xi } } ] . } \end{array}\tag{31}
$$

The coeficients may depend on the entire random row. The probability is over the joint distribution of $\{ v _ { a } \} _ { a = 1 } ^ { M }$

Proof. The case $B = 0$ is immediate. Assume $B > 0$ and $0 < \xi < 1$ . Fix $j \in [ J ]$ and $p \geq 2$ , and write

$$
Z _ { j } = \operatorname* { s u p } _ { u \in T } \left| \frac { 1 } { M } \sum _ { a } B _ { j } ( v _ { a } ) \operatorname { R e L U } ( \langle v _ { a } , u \rangle ) - \mathbb { E } _ { v } B _ { j } ( v ) \operatorname { R e L U } ( \langle v , u \rangle ) \right| .
$$

Let $\epsilon _ { a }$ be independent Rademacher signs, independent of the rows. For fixed rows, $s \mapsto B _ { j } ( v _ { a } )$ ReL $\boldsymbol { { \mathrm { U } } } ( s ) / B$ is 1-Lipschitz and vanishes at zero. Symmetrization [17, Chapter $6 ]$ and contraction over the signs [17, Chapter 4], followed by averaging over the rows, give

$$
\| Z _ { j } \| _ { L ^ { p } } \leq \frac { 4 B } { M } \left\| \operatorname* { s u p } _ { u \in T } \left| \langle \sum _ { a } \epsilon _ { a } v _ { a } , u \rangle \right| \right\| _ { L ^ { p } } .
$$

Write $\boldsymbol { u } = \left( u ^ { \prime } , u _ { 0 } \right)$ and let $P ( u ^ { \prime } , u _ { 0 } ) = u ^ { \prime } $ , so $P T \subset B _ { 2 } ^ { d }$ . Conditional on $\begin{array} { r l } { \{ \beta _ { a } , \epsilon _ { a } \} _ { a = 1 } ^ { M } , } & { { } \sum _ { a } \epsilon _ { a } g _ { a } \overset { \mathrm { d } } { = } } \end{array}$ ${ \sqrt { M } } g ,$ where $g \sim N ( 0 , I _ { d } )$ . For a fixed anchor $u _ { * } \in P T$ , Gaussian symmetry gives

$$
\begin{array} { r l } & { \mathbb { E } _ { g } \underset { v \in P T } { \operatorname* { s u p } } | \langle g , v \rangle | \leq \mathbb { E } _ { g } \underset { v , v ^ { \prime } \in P T } { \operatorname* { s u p } } \langle g , v - v ^ { \prime } \rangle + \mathbb { E } _ { g } | \langle g , u _ { * } \rangle | } \\ & { \qquad \leq 2 w ( P T ) + \sqrt { 2 / \pi } . } \end{array}
$$

Jensen’s inequality with an independent $h \sim N ( 0 , 1 )$ gives

$$
w ( P T ) \leq \mathbb { E } _ { g , h } \operatorname* { s u p } _ { u \in T } ( \langle g , u ^ { \prime } \rangle + h u _ { 0 } ) = w ( T ) .
$$

Since $g \mapsto \operatorname* { s u p } _ { v \in P T } | \langle g , v \rangle |$ is 1-Lipschitz, Gaussian concentration and moment estimates [30, Theorem 5.2.3 and Proposition 2.6.6] yield

$$
\begin{array} { r l r } & { } & { \left\| \underset { u \in T } { \operatorname* { s u p } } \left| \langle \sum _ { a } \epsilon _ { a } ( g _ { a } , 0 ) , u \rangle \right| \right\| _ { L ^ { p } } = \sqrt { M } \left\| \underset { v \in P T } { \operatorname* { s u p } } | \langle g , v \rangle | \right\| _ { L ^ { p } } } \\ & { } & { \leq C \sqrt { M } [ w ( T ) + \sqrt { p } ] . } \end{array}
$$

Conditional on $\{ \beta _ { a } \} _ { a = 1 } ^ { M }$ , the Rademacher moment bound [30, Theorem 2.7.5] gives

$$
\left( \mathbb { E } _ { \epsilon } \left| \sum _ { a } \epsilon _ { a } \beta _ { a } \right| ^ { p } \right) ^ { 1 / p } \leq C \sqrt { p } \left( \sum _ { a } \beta _ { a } ^ { 2 } \right) ^ { 1 / 2 } \leq C c _ { 0 } \sqrt { M p } .
$$

Since $| u _ { 0 } | \le 1$ , taking unconditional $L ^ { p }$ norms bounds the bias contribution by $C c _ { 0 } \sqrt { M p }$ . The triangle inequality and $c _ { 0 } \geq 1$ therefore give

$$
\| Z _ { j } \| _ { L ^ { p } } \leq \frac { C B } { \sqrt { M } } [ w ( T ) + c _ { 0 } \sqrt { p } ] .
$$

Set $p = \operatorname* { m a x } \{ 2 , \log ( 2 e J / \xi ) \}$ , so $p \leq 2 \log ( 2 e J / \xi )$ . Markov’s inequality and a union bound yield

$$
\mathbb { P } \Bigg \{ \operatorname* { m a x } _ { j \in [ J ] } Z _ { j } > \frac { C e B } { \sqrt { M } } [ w ( T ) + c _ { 0 } \sqrt { p } ] \Bigg \} \leq J e ^ { - p } \leq \frac { \xi } { 2 e } \leq \xi .
$$

Absorbing the absolute factors into $C$ proves (31). By continuity, all suprema may be taken over a countable dense subset of $T .$ , so the random variables above are measurable. □

Apply Lemma $\mathrm { A . 3 }$ simultaneously to the $N _ { - } + 1$ targets $h _ { j } ( U _ { x } ) ~ ( j \in [ N _ { - } ] )$ and $\| U _ { x } \|$ , taking $J = N _ { - } + 1 , B = c _ { 0 } K$ , and $\xi = \eta / 2 4$ . The joint empirical error is at most

$$
\frac { C c _ { 0 } K } { \sqrt { M } } \left[ w ( T _ { \delta _ { 0 } } ) + c _ { 0 } \sqrt { \log \frac { 2 e ( N _ { - } + 1 ) } { \xi } } \right] .
$$

Squaring the bracket and absorbing absolute factors inside logarithms into $C$ shows that the error is at most $\varepsilon / ( 1 0 2 4 N _ { + } )$ whenever

$$
M \geq C \left( \frac { N _ { + } } { \varepsilon } \right) ^ { 2 } ( c _ { 0 } K ) ^ { 2 } \left[ w ( T _ { \delta _ { 0 } } ) ^ { 2 } + c _ { 0 } ^ { 2 } \log \frac { 2 4 ( N _ { - } + 1 ) } { \eta } \right] .
$$

By the positive homogeneity of ReLU, this gives an error at each $U _ { x }$ of at most

$$
\varepsilon \| U _ { x } \| / ( 1 0 2 4 N _ { + } ) \qquad ( x \in S ^ { + } \cup S ^ { - } ) .
$$

The Gaussian width estimate for finite unions [30, Theorem 5.2.3 and Proposition 2.7.6] gives

$$
w \left( \mathrm { c l } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) \right) \leq \operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { d - 1 } ) + C \sqrt { \log ( N _ { + } N _ { - } ) } .\tag{32}
$$

Substitute (23) and (32), expand the square, and use $( 1 + R / \delta _ { 0 } ) ^ { 2 } \leq 1 / ( 1 6 \varepsilon )$ . Bounding both logarithmic terms by $\log ( 4 8 N _ { + } N _ { - } / \eta )$ reduces the suficient condition to

$$
M \geq C e ^ { C D } \left[ 1 + \operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { d - 1 } ) ^ { 2 } + \log \frac { 4 8 N _ { + } N _ { - } } { \eta } \right] .\tag{33}
$$

Indeed, (30) gives $K \leq e ^ { C D }$ and $c _ { 0 } ^ { 2 } \leq C D$ , while (25) gives $\log ( N _ { + } / \varepsilon ) \leq C D$ . Thus $( N _ { + } / \varepsilon ) ^ { 2 } ( c _ { 0 } K ) ^ { 2 }$ the additional factor $1 / \varepsilon$ , and the remaining fixed powers can all be absorbed into $e ^ { C D }$ , with absolute comparison constants.

The two truncation errors are at most $1 / 8$ and $1 / ( 8 \sqrt { 2 } )$ of the target error, and the empirical error is at most $1 / 2$ . These fractions sum to less than one, so each target’s total relative error is less than $\varepsilon / ( 5 1 2 N _ { + } )$ .

## A.4 $\langle w _ { j } , \Phi ( x ) \rangle , \langle r , \Phi ( x ) \rangle$ , and radial separation

This subsection uses the finite-sample estimates above to construct $w _ { j }$ and $r .$ . The linear function $\langle w _ { j } , \Phi ( x ) \rangle$ separates $\Phi ( E _ { j } ^ { - } )$ from $\Phi ( S ^ { + } )$ , while $\langle r , \Phi ( x ) \rangle$ is comparable to $\| U _ { x } \|$ . Normalizing $w _ { j }$ then gives an angular separation margin and cross-class distance bounds.

For the first layer in (5), let $W _ { a , \cdot } \in \mathbb { R } ^ { d }$ denote row a of $W$ and $b _ { a }$ denote component a of $b ,$ where $a \in [ n ]$ . Each $W _ { a , \cdot } \sim N ( 0 , I _ { d } )$ and each $b _ { a } \sim \mathrm { U n i f } [ - \lambda , \lambda ]$ , with all row vectors and bias components mutually independent.

Proof of Theorem $\it 2 . 5$

construction and geometric estimates. Let $\lambda _ { 0 } = c _ { 0 } \delta _ { 0 }$ . By (30) and (25), choosing the absolute constant in $( 7 )$ appropriately ensures $\lambda \geq \lambda _ { 0 }$ . Among the row indices $a \in [ n ]$ in the first layer of (5), select those satisfying $| b _ { a } | \leq \lambda _ { 0 }$ and define the actual number of accepted rows by

$$
{ \cal M } = \sum _ { a = 1 } ^ { n } { \bf 1 } _ { \{ | b _ { a } | \leq \lambda _ { 0 } \} } .
$$

Here M is the random count after the network is sampled, rather than the total width $n ;$ the preceding sample-size condition is a lower bound required of the actual M.

Each row is accepted with probability $\lambda _ { 0 } / \lambda .$ , so $M \sim$ Binomial $( n , \lambda _ { 0 } / \lambda )$ . Conditioning on all acceptance indicators $\{ \mathbf { 1 } _ { \{ | b _ { a } | \leq \lambda _ { 0 } \} } \} _ { a \in [ n ] }$ makes M deterministic. Under this conditional distribution, the accepted rows remain independent: $W _ { a , }$ <sub>·</sub> is standard Gaussian, $b _ { a } / \delta _ { 0 }$ is $\mathrm { U n i f } [ - c _ { 0 } , c _ { 0 } ]$ , and the two are independent. This is precisely the distribution required by the empirical estimate above. We apply that estimate to the accepted rows under this conditional distribution.

For the indicators $\{ { \bf 1 } _ { \{ | b _ { a } | \leq \lambda _ { 0 } \} } \} _ { a \in [ n ] } ,$ ,the Chernof lower-tail bound gives

$$
\mathbb { P } \bigg ( M < \frac { n \lambda _ { 0 } } { 2 \lambda } \bigg ) \leq \exp \bigg ( { - \frac { n \lambda _ { 0 } } { 8 \lambda } } \bigg ) .
$$

Increasing the absolute constant in (8) and using (32) ensures both $n \lambda _ { 0 } / \lambda \ge 8 \log ( 2 4 / \eta )$ and that $n \lambda _ { 0 } / ( 2 \lambda )$ is at least the right side of (33). Thus, when the acceptance count reaches this threshold, M meets the sample-size condition for joint empirical approximation; the count fails with probability

$$
\mathbb { P } \bigg \{ M < \frac { n \lambda _ { 0 } } { 2 \lambda } \bigg \} \leq \frac { \eta } { 2 4 } .\tag{34}
$$

Finally, let $\theta = \sqrt { M / n }$ . On the count event above, $c _ { 0 } \geq 1$ gives

$$
\theta \geq \sqrt { \frac { \lambda _ { 0 } } { 2 \lambda } } = \sqrt { \frac { c _ { 0 } \delta _ { 0 } } { 2 \lambda } } \geq \sqrt { \frac { \delta _ { 0 } } { 2 \lambda } } .
$$

At each accepted coordinate a, let $v _ { a } = ( W _ { a , \cdot } , b _ { a } / \delta _ { 0 } )$ and set

$$
w _ { j , a } = - \frac { B _ { j } ( v _ { a } ) } { \sqrt { 2 M } } , \qquad r _ { a } = \frac { 2 c _ { 0 } \varphi ( b _ { a } / \delta _ { 0 } ) \sqrt { 2 \pi } } { \sqrt { 2 M } } ;
$$

Set all unaccepted coordinates to zero. Fix any acceptance-indicator configuration satisfying $M \geq n \lambda _ { 0 } / ( 2 \lambda )$ , and apply the approximation estimate from the preceding subsection under the conditional joint distribution of the accepted rows $\{ ( W _ { a , \cdot } , b _ { a } ) \}$ . The conditional probability that the following inequalities fail to hold simultaneously is at most $\eta / 2 4$

$$
\begin{array} { r l } { | \langle w _ { j } , \Phi ( x ) \rangle + \theta h _ { j } ( U _ { x } ) | \le \displaystyle \frac { \varepsilon \theta } { 5 1 2 N _ { + } } \| U _ { x } \| , } & { { } } \\ { | \langle r , \Phi ( x ) \rangle - \theta \| U _ { x } \| | \le \displaystyle \frac { \varepsilon \theta } { 5 1 2 N _ { + } } \| U _ { x } \| , \quad } & { { } x \in S ^ { + } \cup S ^ { - } , } \end{array}\tag{35}
$$

$$
\| w _ { j } \| , \| r \| \leq e ^ { C D } .\tag{36}
$$

The bounds on $\lVert \boldsymbol { w } _ { j } \rVert$ and $\| r \|$ follow directly from the bounded coeficients on accepted coordinates, and each $r _ { a }$ is nonnegative. Together with (27), since $N _ { + } \geq 1$ and $\varepsilon \le 1 / 1 6$ , this gives

$$
\begin{array} { r } { \langle w _ { j } , \Phi ( x ) \rangle \leq - \frac { \varepsilon } { 4 } \theta \| U _ { x } \| \quad ( x \in S ^ { + } ) , \qquad \langle w _ { j } , \Phi ( y ) \rangle \geq \frac { \varepsilon } { 4 } \theta \| U _ { y } \| \quad ( y \in E _ { j } ^ { - } ) , } \end{array}\tag{37}
$$

$$
\begin{array} { r } { | \langle w _ { j } , \Phi ( x ) \rangle | \leq ( N _ { + } + 1 ) \theta \| U _ { x } \| , \qquad \frac { 1 } { 2 } \theta \| U _ { x } \| \leq \langle r , \Phi ( x ) \rangle \leq \frac { 3 } { 2 } \theta \| U _ { x } \| \quad ( x \in S ^ { + } \cup S ^ { - } ) . } \end{array}\tag{38}
$$

Multiply the conditional failure probability above by the indicator that the count reaches its threshold, then average over the distribution of $\{ \mathbf { 1 } _ { \{ | b _ { a } | \leq \lambda _ { 0 } \} } \} _ { a \in [ n ] }$ . This gives the following failure probability over the joint distribution of $( W , b )$

$$
\mathbb { P } \bigg \{ M \geq \frac { n \lambda _ { 0 } } { 2 \lambda } , \quad \mathrm { t h e ~ i n e q u a l i t i e s ~ i n ~ ( 3 5 ) ~ d o ~ n o t ~ a l l ~ h o l d } \bigg \} \leq \frac { \eta } { 2 4 } .\tag{39}
$$

The norm bound (36) follows from deterministic coeficient bounds and adds no failure event.

Normalization of $w _ { j }$ , angular margin, and cross-class distances. To turn a lower bound on $\langle w _ { j } , \Phi ( y ) - \Phi ( x ) \rangle$ into an angular margin in feature space, first bound $\| \Phi ( y ) - \Phi ( x ) \|$

Apply the Gaussian matrix deviation bound [16] to W on $\begin{array} { r } { T = \mathrm { c l } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) } \end{array}$ , taking $t = \sqrt { \log ( 4 8 / \eta ) }$ . If $\begin{array} { r } { n \geq C [ 1 + w \Big ( \mathrm { c l } \bigcup _ { i , j } \big ( K _ { i j } \cap S ^ { d - 1 } \big ) \Big ) ^ { 2 } + \log ( 4 8 / \eta ) ] } \end{array}$ , the norm deviation divided by $\sqrt { n }$ is at most the fixed number $3 / ( 2 { \sqrt { 2 } } ) - 1 > 0$ . By (32), (8) implies this width requirement. For every $x \in S ^ { + }$ and $y \in S ^ { - }$ , the normalized diference vector belongs to $T .$ . Applying the 1-Lipschitz property of ReLU coordinatewise gives

$$
\begin{array} { r } { \| \Phi ( x ) - \Phi ( y ) \| \le \sqrt { 2 / n } \| W ( x - y ) \| \le \frac { 3 } { 2 } \| x - y \| . } \end{array}
$$

The common bias cancels in the first diference bound; the matrix event depends only on $W$ and holds for every allowed b. Since $2 e ^ { - t ^ { 2 } } = \eta / 2 4$ , the failure probability of the cross-class distance upper bound under the joint distribution of $( W , b )$ satisfies

$$
\mathbb { P } \left\{ \operatorname* { s u p } _ { x \in S ^ { + } } \frac { \| \Phi ( x ) - \Phi ( y ) \| } { \| x - y \| } > \frac { 3 } { 2 } \right\} \leq \frac { \eta } { 2 4 } .\tag{40}
$$

Let $a _ { j } = w _ { j } / \| w _ { j } \|$ . By (37) and $( 3 6 ) , - \langle a _ { j } , \Phi ( x ) \rangle$ and $\langle a _ { j } , \Phi ( y ) \rangle$ are at least $\varepsilon \theta / ( 4 e ^ { C D } )$ times $\| U _ { x } \|$ and $\| U _ { y } \|$ , respectively. Bounding $\langle a _ { j } , \Phi ( y ) \rangle - \langle a _ { j } , \Phi ( x ) \rangle$ and using

$$
\begin{array} { r } { \| U _ { x } \| + \| U _ { y } \| \geq \| x - y \| \geq \frac 2 3 \| \Phi ( x ) - \Phi ( y ) \| , } \end{array}
$$

gives an angular margin of at least $\varepsilon \theta / ( 6 e ^ { C D } )$ . By $\theta \ge \sqrt { \delta _ { 0 } / ( 2 \lambda ) }$ $\log ( 1 / \varepsilon ) \leq C D$ , and (25), we may take

$$
\vartheta = c \sqrt { \frac { \delta _ { 0 } } { \lambda } } \exp \Biggl \{ - \frac { C ( 1 + R / \delta _ { 0 } ) ^ { 2 } \log ( 4 N _ { + } ) } { \kappa _ { \operatorname* { m i n } } } \Biggr \} > 0 .
$$

For this choice, all $j , x \in S ^ { + }$ , and $y \in E _ { j } ^ { - }$ simultaneously satisfy

$$
\begin{array} { r } { \langle a _ { j } , \Phi ( x ) \rangle \leq - \vartheta \sqrt { \| x \| ^ { 2 } + \delta _ { 0 } ^ { 2 } } , \qquad \langle a _ { j } , \Phi ( y ) \rangle \geq \vartheta \sqrt { \| y \| ^ { 2 } + \delta _ { 0 } ^ { 2 } } , } \end{array}
$$

and the bound on $\langle a _ { j } , \Phi ( y ) - \Phi ( x ) \rangle$ above gives (9). For any $y \in S$ <sup>−</sup>, choose j such that $y \in E _ { j } ^ { - }$ Since $\| a _ { j } \| = 1$ , these score bounds also give

$$
\begin{array} { r l } & { \| \Phi ( \boldsymbol { y } ) - \Phi ( \boldsymbol { x } ) \| \ge \langle a _ { j } , \Phi ( \boldsymbol { y } ) - \Phi ( \boldsymbol { x } ) \rangle } \\ & { \qquad \ge \vartheta \big ( \| U _ { \boldsymbol { y } } \| + \| U _ { \boldsymbol { x } } \| \big ) \ge \vartheta \| \boldsymbol { x } - \boldsymbol { y } \| . } \end{array}
$$

Together with the preceding distance upper bound, this proves (10). Finally, fix $j \in [ N _ { - } ]$ . For any finite nonnegative linear combination

$$
z = \sum _ { \ell = 1 } ^ { m } t _ { \ell } \big [ \Phi ( y _ { \ell } ) - \Phi ( x _ { \ell } ) \big ] , \qquad t _ { \ell } \geq 0 , \quad x _ { \ell } \in S ^ { + } , \quad y _ { \ell } \in E _ { j } ^ { - } ,
$$

(9) and the triangle inequality give

$$
\begin{array} { c } { \displaystyle \langle a _ { j } , z \rangle = \displaystyle \sum _ { \ell = 1 } ^ { m } t _ { \ell } \langle a _ { j } , \Phi ( y _ { \ell } ) - \Phi ( x _ { \ell } ) \rangle } \\ { \displaystyle \geq \vartheta \displaystyle \sum _ { \ell = 1 } ^ { m } t _ { \ell } \| \Phi ( y _ { \ell } ) - \Phi ( x _ { \ell } ) \| \geq \vartheta \| z \| . } \end{array}
$$

By continuity of the inner product and norm, passing to limits extends this inequality to all $z \in \mathrm { c l } \mathrm { c o n e } ( \Phi ( E _ { j } ^ { - } ) - \Phi ( S ^ { + } ) )$ . If both z and −z belong to this closed cone, then

$$
0 = \langle a _ { j } , z \rangle + \langle a _ { j } , - z \rangle \geq 2 \vartheta \Vert z \Vert .
$$

Since $\vartheta > 0 .$ , this forces $z = 0$ . The closed cone therefore intersects its negative only at {0}, so it is pointed. The probability bookkeeping, including the norm and feature-complexity events, is completed at the end of the proof of Proposition 2.6.

## A.5 Bounds on $\| \Phi ( x ) \|$

In this subsection, we prove Proposition 2.6, including (11), and

$$
w \left( \mathrm { c l } \left\{ \frac { \Phi ( x ) } { \| \Phi ( x ) \| } : x \in S ^ { + } \cup S ^ { - } \right\} \right) ^ { 2 } \leq C ( 1 + R / \delta _ { 0 } ) ^ { 2 } \left[ 1 + w \left( \mathrm { c l } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) \right) ^ { 2 } + \log \frac { 2 4 } { \eta } \right] .\tag{41}
$$

Proof of Proposition 2.6 and (41). The bound on $\left\| \Phi ( x ) \right\|$ uses the scale λ. Write

$$
T _ { \lambda } = \operatorname { c l } \left\{ { \frac { ( x , \lambda ) } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } } : x \in S ^ { + } \cup S ^ { - } \right\} \subset S ^ { d } .
$$

First bound $w ( T _ { \lambda } )$ in terms of $w ( T _ { \delta _ { 0 } } )$ . The first-layer bias condition ensures $\lambda \geq \delta _ { 0 }$ , hence

$$
w ( T _ { \lambda } ) \leq w ( T _ { \delta _ { 0 } } ) + \sqrt { \frac { 2 } { \pi } } .\tag{42}
$$

To prove this, let $g = ( g ^ { \prime } , g _ { 0 } ) \sim N ( 0 , I _ { d + 1 } )$ . Since

$$
\| U _ { x } \| / { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } \leq 1 , \qquad \lambda / { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } \leq 1 ,
$$

for all $x \in S ^ { + } \cup S$ <sup>−</sup> we have

$$
\frac { \langle g ^ { \prime } , x \rangle + \lambda g _ { 0 } } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } \leq \biggl ( \frac { \langle g ^ { \prime } , x \rangle } { \| U _ { x } \| } \biggr ) _ { + } + ( g _ { 0 } ) _ { + } .
$$

Projecting $T _ { \delta _ { 0 } }$ onto its first d coordinates does not increase its Gaussian width. Choose an anchor in this projection, whose norm is at most 1. Adding the origin increases the width by at most $1 / \sqrt { 2 \pi }$ Thus

$$
\mathbb { E } _ { g ^ { \prime } } \operatorname* { s u p } _ { x \in S ^ { + } \cup S ^ { - } } \left( \frac { \langle g ^ { \prime } , x \rangle } { \| U _ { x } \| } \right) _ { + } \leq w ( T _ { \delta _ { 0 } } ) + \frac { 1 } { \sqrt { 2 \pi } } .
$$

Using $\mathbb { E } _ { g _ { 0 } } ( g _ { 0 } ) _ { + } = 1 / \sqrt { 2 \pi }$ , take the supremum in the preceding inequality and average over $( g ^ { \prime } , g _ { 0 } )$ to obtain (42). Taking closures does not change the suprema of these continuous Gaussian processes.

To prove (11), fix $u = ( u ^ { \prime } , u _ { 0 } ) \in T _ { \lambda }$ and take independent $g \sim N ( 0 , I _ { d } )$ and $\beta \sim \mathrm { U n i f } [ - 1 , 1 ]$ Conditional Jensen inequalities give

$$
\begin{array} { r } { \mathbb { E } _ { g , \beta } \operatorname { R e L U } ( \langle g , u ^ { \prime } \rangle + \beta u _ { 0 } ) \geq \operatorname* { m a x } \{ \| u ^ { \prime } \| / \sqrt { 2 \pi } , | u _ { 0 } | / 4 \} \geq 1 / ( 4 \sqrt { 2 } ) . } \end{array}
$$

Apply Lemma $\mathrm { A . 3 }$ on $T _ { \lambda }$ to the sample rows $\{ ( W _ { a , \cdot } , b _ { a } / \lambda ) \} _ { a \in [ n ] }$ , with probability over the joint distribution of $( W , b ) ;$ take coeficient $1 , c _ { 0 } = 1 , J = 1$ , and $\xi = \eta / 2 4$ . If $n \geq C [ 1 + w ( T _ { \lambda } ) ^ { 2 } + \log ( 2 4 / \eta ) ]$ the empirical-mean error is at most $1 / ( 8 \sqrt { 2 } )$

For $x \in S ^ { + } \cup S$ <sup>−</sup>, take $u = ( x , \lambda ) / \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } }$ , so that

$$
\Phi ( x ) / \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } = \sqrt { 2 / n } \operatorname { R e L U } ( W u ^ { \prime } + ( b / \lambda ) u _ { 0 } ) .
$$

Since the root mean square is at least the arithmetic mean, this proves the lower bound in (11), with failure probability satisfying

$$
\mathbb { P } \Bigg \{ \operatorname* { i n f } _ { \substack { x \in S ^ { + } \cup S ^ { - } } } \frac { \| \Phi ( x ) \| } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } < \frac { 1 } { 8 } \Bigg \} \leq \frac { \eta } { 2 4 } .\tag{43}
$$

Projection does not increase Gaussian width. Thus, under the same width condition, the bound in [16] with $t = \sqrt { \log ( 4 8 / \eta ) }$ gives $\mathrm { s u p } _ { u \in T _ { \lambda } } \| W u ^ { \prime } \| / \sqrt { n } \leq 1 1 / 1 0$ . Together with $\left\| b / \lambda \right\| \le { \sqrt { n } }$ and $| u _ { 0 } | \le 1$ , this yields

$$
\frac { \| \Phi ( x ) \| } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } \le \sqrt { 2 / n } \big ( \| W u ^ { \prime } \| + \| b / \lambda \| | u _ { 0 } | \big ) \le 3 .
$$

The matrix event here depends only on $W ,$ and the bias bound holds for all $b \in [ - \lambda , \lambda ] ^ { n }$ . Thus, under the joint distribution of $( W , b )$ , the failure probability of the upper bound on $\left\| \Phi ( x ) \right\|$ satisfies

$$
\mathbb { P } \bigg \{ \operatorname* { s u p } _ { x \in S ^ { + } \cup S ^ { - } } \frac { \| \Phi ( x ) \| } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } > 3 \bigg \} \leq \frac { \eta } { 2 4 } .\tag{44}
$$

Both bounds on $\left\| \Phi ( x ) \right\|$ retain the scale $\lambda$

Next prove (41). Fix b and, for $u = ( u ^ { \prime } , u _ { 0 } ) \in \mathbb { R } ^ { d + 1 }$ , write

$$
F _ { W } ( u ) = \sqrt { 2 / n } \mathrm { R e L U } \bigl ( W u ^ { \prime } + ( b / \lambda ) u _ { 0 } \bigr ) .
$$

By the positive homogeneity of ReLU, for $x \in S ^ { + } \cup S ^ { - }$ ，

$$
F _ { W } \left( { \frac { \left( x , \lambda \right) } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } } \right) = { \frac { \Phi ( x ) } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } } .
$$

Fix $W , b ,$ take $h \sim N ( 0 , I _ { n } )$ independent of the network, and compare the centered Gaussian processes indexed by $u \in T _ { \lambda }$

$$
X _ { u } = \langle h , F _ { W } ( u ) \rangle , \qquad Y _ { u } = \sqrt { 2 / n } \langle h , W u ^ { \prime } + ( b / \lambda ) u _ { 0 } \rangle .
$$

The 1-Lipschitz property of ReLU gives $\mathbb { E } _ { h } ( X _ { u } - X _ { v } ) ^ { 2 } \le \mathbb { E } _ { h } ( Y _ { u } - Y _ { v } ) ^ { 2 }$ for $u , v \in T _ { \lambda }$ . Since $F _ { W } ( 0 ) = 0$ both centered Gaussian processes extend to $T _ { \lambda } \cup \{ 0 \}$ with $X _ { 0 } = Y _ { 0 } = 0$ . if one index is zero, we likewise have

$$
\begin{array} { r } { \mathbb { E } _ { h } ( X _ { u } - X _ { 0 } ) ^ { 2 } = \mathbb { E } _ { h } X _ { u } ^ { 2 } \le \mathbb { E } _ { h } Y _ { u } ^ { 2 } = \mathbb { E } _ { h } ( Y _ { u } - Y _ { 0 } ) ^ { 2 } . } \end{array}
$$

The increment-variance comparison therefore holds on all of $T _ { \lambda } \cup \{ 0 \}$ . The Sudakov–Fernique comparison thus gives

$$
w \big ( F _ { W } ( T _ { \lambda } \cup \{ 0 \} ) \big ) = \mathbb { E } _ { h } \operatorname* { s u p } _ { u \in T _ { \lambda } \cup \{ 0 \} } X _ { u } \leq \mathbb { E } _ { h } \operatorname* { s u p } _ { u \in T _ { \lambda } \cup \{ 0 \} } Y _ { u } .
$$

Keep b fixed and average the right side over W. First fix the auxiliary vector $h$ and randomize only $W ;$ then $W ^ { \top } h$ is distributed as $\| h \|$ times a standard Gaussian vector, and

$$
\big ( \langle h , W u ^ { \prime } + ( b / \lambda ) u _ { 0 } \rangle \big ) _ { + } \leq \big ( \langle W ^ { \top } h , u ^ { \prime } \rangle \big ) _ { + } + | \langle h , b / \lambda \rangle | \qquad ( u \in T _ { \lambda } ) .
$$

Average first over $W$ and apply the projection and anchor estimates above to $T _ { \lambda }$ , then average over the auxiliary vector h. Using $\mathbb { E } _ { h } \| h \| \leq \sqrt { n }$ and $\mathbb { E } _ { h } | \langle h , b / \lambda \rangle | \leq { \sqrt { 2 / \pi } } { \sqrt { n } }$ gives

$$
\mathbb { E } _ { W } \mathbb { E } _ { h } \operatorname* { s u p } _ { u \in T _ { \lambda } } \left( \langle h , F _ { W } ( u ) \rangle \right) _ { + } \leq C [ 1 + w ( T _ { \lambda } ) ] .
$$

The 1-Lipschitz property of ReLU also gives

$$
\operatorname* { s u p } _ { u \in T _ { \lambda } } \Vert F _ { W } ( u ) - F _ { \widetilde { W } } ( u ) \Vert \leq \sqrt { 2 / n } \Vert W - \widetilde { W } \Vert _ { \mathrm { F } } .
$$

Since $\mathbb { E } _ { h } \| h \| \leq \sqrt { n }$ , we have

$$
\left| \mathbb { E } _ { h } \operatorname* { s u p } _ { u \in T _ { \lambda } } \left( \left. h , F _ { W } ( u ) \right. \right) _ { + } - \mathbb { E } _ { h } \operatorname* { s u p } _ { u \in T _ { \lambda } } \left( \left. h , F _ { \widetilde { W } } ( u ) \right. \right) _ { + } \right| \leq \sqrt { 2 } \| W - \widetilde { W } \| _ { \mathrm { F } } .
$$

For each fixed $b \in [ - \lambda , \lambda ] ^ { n }$ , apply Gaussian concentration in W. Averaging the resulting bound over b gives

$$
\mathbb { P } \Bigg \{ \mathbb { E } _ { h } \operatorname* { s u p } _ { u \in T _ { \lambda } } \big ( \langle h , F _ { W } ( u ) \rangle \big ) _ { + } > C \left[ 1 + w ( T _ { \lambda } ) + \sqrt { \log ( 2 4 / \eta ) } \right] \Bigg \} \leq \frac { \eta } { 2 4 } .\tag{45}
$$

The suprema of these processes can be taken over the compact set $T _ { \lambda }$ , so unbounded inputs do not afect the argument or measurability.

On the norm lower-bound event controlled by (43), continuity implies $\| F _ { W } ( u ) \| \ge 1 / 8$ for every $u \in T _ { \lambda }$ . When $u = ( x , \lambda ) / \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } }$ with $x \in S ^ { + } \cup S ^ { - } , F _ { W } ( u ) / \| F _ { W } ( u ) \| = \Phi ( x ) / \| \Phi ( x ) \|$ . Fix (W, b) satisfying this norm lower bound. Gaussian symmetry gives

$$
\begin{array} { r l } & { w \displaystyle \biggl ( \mathrm { c l } \left\{ \frac { \Phi ( x ) } { \| \Phi ( x ) \| } : x \in S ^ { + } \cup S ^ { - } \right\} \biggr ) \leq 8 \mathbb { E } _ { h } \underset { u \in T _ { \lambda } } { \operatorname* { s u p } } | \langle h , F _ { W } ( u ) \rangle | } \\ & { \qquad \leq 1 6 \mathbb { E } _ { h } \underset { u \in T _ { \lambda } } { \operatorname* { s u p } } \left( \langle h , F _ { W } ( u ) \rangle \right) _ { + } . } \end{array}
$$

The last inequality follows by separating positive and negative parts and using Gaussian symmetry. On the norm lower-bound event and the concentration event controlled by (45), (42) and (23) give (41), with no additional failure event. The required sample size is $C [ 1 + w ( T _ { \lambda } ) ^ { 2 } + \log ( 2 4 / \eta ) ]$ ]. These bounds and (32) show that (8) sufices.

Joint probability. Each of the six failure bounds (34), (39), (40), (43), (44), and (45) is at most $\eta / 2 4$ under the joint law of $( W , b )$ . The approximation bound already averages over $\{ \mathbf { 1 } _ { \{ | b _ { a } | \leq \lambda _ { 0 } \} } \} _ { a \in [ n ] }$ and counts failures only when M meets its threshold, so no further union bound over M is needed. Since (41) adds no failure event, a union bound shows that all conclusions of Theorem 2.5 and Proposition 2.6, together with the auxiliary estimates for the second layer, hold with probability at least $1 - \eta / 4$ . This establishes Theorem 2.5 with probability at least $1 - \eta$ and completes its proof.

## B Proof of separation by the second layer

This appendix proves Theorem 2.7 in three steps. Subsection B.1 constructs an even-degree homogeneous polynomial Q separating $\Phi ( S ^ { + } )$ from $\Phi ( S ^ { - } )$ . It also bounds the degree and coeficients of $Q$ . Subsection B.2 gives a random ReLU expectation with the same separating signs and proves the multiplier estimates. Subsection B.3 approximates this expectation uniformly with finitely many neurons. This yields a common separating hyperplane for $F ( S ^ { + } )$ and $F ( S ^ { - } )$ , together with the required width and margin bounds.

Proof of Theorem 2.7. Fix a first-layer realization $( W , b )$ satisfying the events in Appendix $\mathrm { A } ,$ whose total failure probability is at most $\eta / 4$ , and retain the notation and estimates established there. All probabilities below are conditional on this realization until the final probability estimate. The second layer remains independent and follows its original distribution.

## B.1 From $\langle w _ { j } , z \rangle$ to a common separating polynomial

This subsection constructs an even-degree homogeneous polynomial that separates $\Phi ( S ^ { + } )$ from $\Phi ( S ^ { - } )$ . Throughout, write $z = \Phi ( x )$ with $x \in S ^ { + } \cup S ^ { - }$ . Each $\langle w _ { j } , z \rangle$ is positive on $\Phi ( E _ { j } ^ { - } )$ and negative on $\Phi ( S ^ { + } )$ . Define linear forms $s _ { j }$ that are positive on $\Phi ( S ^ { + } \cup S ^ { - } )$ by

$$
s _ { j } ( z ) = \left( \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 1 2 8 N _ { + } } \right) \langle r , z \rangle - \langle w _ { j } , z \rangle .
$$

Substituting (26) into (35) gives, for all $x \in S ^ { + } \cup S ^ { - }$ and $j \in [ N _ { - } ]$

$$
\begin{array} { r l r } {  { \bigg | \frac { s _ { j } ( \Phi ( x ) ) } { \theta \| U _ { x } \| } - \sum _ { i } P \Big ( \frac { \langle q _ { i j } , U _ { x } \rangle } { \| U _ { x } \| } \Big ) - \frac { \varepsilon } { 1 2 8 N _ { + } } \bigg | } } \\ & { } & { \leq \bigg ( 1 + \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 1 2 8 N _ { + } } \bigg ) \frac { \varepsilon } { 5 1 2 N _ { + } } \leq \frac { 2 \varepsilon } { 5 1 2 N _ { + } } . } \end{array}
$$

Here the errors come from $w _ { j }$ and $r ,$ and $\varepsilon \le 1 / 1 6$ . The sum lies in $[ 0 , N _ { + } ]$ . By (26) and (27), it is at least $\varepsilon$ on $S ^ { + }$ and at most $\varepsilon / ( 5 1 2 N _ { + } )$ on $E _ { j } ^ { - }$ . Combining these bounds with the preceding error estimate gives

$$
\begin{array} { r l r } { \displaystyle \frac { \varepsilon \theta } { 5 1 2 N _ { + } } \| U _ { x } \| \le s _ { j } ( \Phi ( x ) ) \le \left( N _ { + } + \frac { 3 \varepsilon } { 2 5 6 N _ { + } } \right) \theta \| U _ { x } \| } & { ( x \in S ^ { + } \cup S ^ { - } ) , } \\ { \displaystyle s _ { j } ( \Phi ( x ) ) \ge \frac { 3 } { 4 } \varepsilon \theta \| U _ { x } \| } & { ( x \in S ^ { + } ) , } \\ { \displaystyle s _ { j } ( \Phi ( y ) ) \le \frac { \varepsilon \theta } { 6 4 N _ { + } } \| U _ { y } \| } & { ( y \in E _ { j } ^ { - } ) . } \end{array}\tag{46}
$$

By (46), all $s _ { j }$ are positive on $\Phi ( S ^ { + } \cup S ^ { - } )$ . Together with (38), this gives

$$
{ \frac { \varepsilon \langle r , z \rangle } { 2 s _ { j } ( z ) } } \leq 1 \quad ( z \in \Phi ( S ^ { + } ) ) , \qquad { \frac { \varepsilon \langle r , z \rangle } { 2 s _ { j } ( z ) } } \geq 1 6 N _ { + } \quad ( z \in \Phi ( E _ { j } ^ { - } ) ) .
$$

To make the jth powered term on $\Phi ( E _ { j } ^ { - } )$ dominate the sum on $\Phi ( S ^ { + } )$ , take

$$
p = \left\lceil \frac { \log ( 4 N _ { - } ) } { \log ( 1 6 N _ { + } ) } \right\rceil .\tag{47}
$$

Here $p \geq 1$ and $( 1 6 N _ { + } ) ^ { p } \geq 4 N _ { - }$ . For $p \leq \varepsilon ^ { - 1 }$ , clearing denominators gives a homogeneous polynomial of degree $N _ { - } p .$ . Otherwise, we use Chebyshev polynomials with common denominator $\langle r , z \rangle$ . This choice depends only on geometric parameters fixed before sampling.

Case one: $p \leq \varepsilon ^ { - 1 }$ . By these ratio bounds,

$$
\sum _ { j } \left( \frac { \varepsilon \langle r , z \rangle } { 2 s _ { j } ( z ) } \right) ^ { p } - 2 N _ { - } \le - N _ { - } \quad ( z \in \Phi ( S ^ { + } ) ) ,
$$

whereas on each $\Phi ( E _ { j } ^ { - } )$ , the $j \mathrm { t h }$ term is at least $4 N _ { - }$ and the other terms are nonnegative, so the expression is at least 2N<sub>−</sub>. Multiplying by the product of the positive denominators preserves the sign and gives

$$
Q ( z ) = \left( \frac { \varepsilon \langle r , z \rangle } { 2 } \right) ^ { p } \sum _ { j } \prod _ { h \neq j } s _ { h } ( z ) ^ { p } - 2 N _ { - } \prod _ { j } s _ { j } ( z ) ^ { p } .\tag{48}
$$

On $\Phi ( S ^ { + } \cup S ^ { - } )$ , the right-hand side is exactly

$$
\left[ \prod _ { h } s _ { h } ( z ) ^ { p } \right] \left[ \sum _ { j } \left( \frac { \varepsilon \langle r , z \rangle } { 2 s _ { j } ( z ) } \right) ^ { p } - 2 N _ { - } \right] .
$$

Each term consists of $N _ { - } p$ linear factors; when $N _ { - } = 1$ , the empty product is 1. If $N _ { - } p$ is odd, multiply the right-hand side by $\langle r , z \rangle$ and still denote it by $Q .$ . Then $Q$ is homogeneous of even degree $k \geq 2$ , with

$$
N _ { - } p \leq k \leq N _ { - } p + 1 \leq 2 N _ { - } p .
$$

The positive factor $\langle r , z \rangle$ preserves the separating signs of $Q .$ . The second bracket has absolute value at least $N _ { - }$ . Applying (46) to each factor and, in the odd case, using $\langle r , \Phi ( x ) \rangle \geq \theta \| U _ { x } \| / 2$ , we obtain

$$
| Q ( \Phi ( x ) ) | \geq \left\{ \begin{array} { l l } { N _ { - } \left( \frac { \varepsilon \theta \| U _ { x } \| } { 5 1 2 N _ { + } } \right) ^ { k } , } & { k = N _ { - } p , } \\ { N _ { - } \left( \frac { \varepsilon \theta \| U _ { x } \| } { 5 1 2 N _ { + } } \right) ^ { k - 1 } \frac { \theta \| U _ { x } \| } { 2 } , } & { k = N _ { - } p + 1 . } \end{array} \right.
$$

Since $N _ { - } \geq 1$ and $\varepsilon / ( 5 1 2 N _ { + } ) \le 1 / 2$ , both cases give

$$
| Q ( \Phi ( x ) ) | \geq \left( { \frac { c \varepsilon } { N _ { + } } } \right) ^ { k } \theta ^ { k } \| U _ { x } \| ^ { k } \qquad ( x \in S ^ { + } \cup S ^ { - } ) .\tag{49}
$$

We may take $c = 1 / 5 1 2$ . This strict margin also rules out $Q \equiv 0 ;$ since every term is homogeneous of degree $k ,$ the actual degree of $Q$ is exactly the even integer k. We next bound the coeficients used in the random feature representation. By (36), the coeficient vector of $s _ { j }$ satisfies

$$
\left\| \left( \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 1 2 8 N _ { + } } \right) r - w _ { j } \right\| \leq 2 e ^ { C D } .
$$

The norms of $\varepsilon r / 2$ in the numerator and of the possible added factor r are both at most $e ^ { C D }$ Keeping the products of linear factors intact, the sum of the absolute scalar coeficients in (48) is $N _ { - } + 2 N _ { - } = 3 N _ { - }$ . Thus there exists a decomposition

$$
Q ( z ) = \sum _ { a } c _ { a } \prod _ { b = 1 } ^ { k } \langle v _ { a b } , z \rangle , \qquad \sum _ { a } | c _ { a } | \prod _ { b } \| v _ { a b } \| \leq e ^ { C D k } .\tag{50}
$$

where the absorption in the last step is

$$
\sum _ { a } | c _ { a } | \prod _ { b } \| v _ { a b } \| \leq 3 N _ { - } ( 2 e ^ { C D } ) ^ { k } \leq e ^ { C D k } ;
$$

using $k \geq N _ { - }$ and $D \geq 1$ and increasing the positive constant $C ,$ independent of the sets and $\eta ,$ as needed. This decomposition need not be expanded into coordinate monomials. Its coeficient bound matches the Wick isometry estimate below and introduces no ambient dimension factor.

Case two: $p > \varepsilon ^ { - 1 }$ . In this case, divide each $s _ { j } ( z )$ by $\langle r , z \rangle$ instead. By (46) and (38),

$$
0 < \frac { s _ { j } ( z ) } { \langle r , z \rangle } \le 3 N _ { + } , \qquad \frac { s _ { j } ( z ) } { \langle r , z \rangle } \ge \frac { \varepsilon } { 2 } \bigl ( z \in \Phi ( S ^ { + } ) ) , \qquad \frac { s _ { j } ( z ) } { \langle r , z \rangle } \le \frac { \varepsilon } { 3 2 N _ { + } } \bigl ( z \in \Phi ( E _ { j } ^ { - } ) \bigr ) .
$$

Apply an afine transformation to the ratios so that their values on $\Phi ( S ^ { + } )$ lie in the interval where Chebyshev polynomials are bounded:

$$
t _ { j } ( z ) = 1 + \frac { \varepsilon } { 4 N _ { + } } - \frac { s _ { j } ( z ) } { 2 N _ { + } \langle r , z \rangle } .
$$

Specifically, the ratio bounds above give

$$
\begin{array} { r l r } { \displaystyle { 1 \geq t _ { j } ( z ) \geq - \frac { 1 } { 2 } + \frac { \varepsilon } { 4 N _ { + } } } } & { { ( z \in \Phi ( S ^ { + } ) ) , } } \\ { \displaystyle { t _ { j } ( z ) \geq 1 + \frac { \varepsilon } { 4 N _ { + } } - \frac { \varepsilon } { 6 4 N _ { + } ^ { 2 } } \geq 1 + \frac { \varepsilon } { 8 N _ { + } } } } & { { ( z \in \Phi ( E _ { j } ^ { - } ) ) . } } \end{array}
$$

Thus the values on $\Phi ( S ^ { + } )$ lie in $[ - 1 , 1 ]$ . Chebyshev polynomials are bounded on $[ - 1 , 1 ]$ and grow exponentially with the degree to the right of 1; after squaring, the terms corresponding to all $E _ { j } ^ { - }$ are nonnegative. We therefore take

$$
k = 2 \left\lceil \sqrt { \frac { 8 N _ { + } } { \varepsilon } } \log ( 4 \sqrt { N _ { - } } ) \right\rceil\tag{51}
$$

and set

$$
Q ( z ) = ( \langle r , z \rangle ) ^ { k } \left[ \sum _ { j } \mathsf { T } _ { k / 2 } ( t _ { j } ( z ) ) ^ { 2 } - 2 N _ { - } \right] .\tag{52}
$$

Here $\top _ { k / 2 }$ is the Chebyshev polynomial of the first kind, with the notation of Lemma A.2. For $0 < a \le 1$

$$
\operatorname { a r c o s h } ( 1 + a ) = \int _ { 0 } ^ { a } { \frac { d s } { \sqrt { 2 s + s ^ { 2 } } } } \geq \int _ { 0 } ^ { a } { \frac { d s } { \sqrt { 3 s } } } = { \frac { 2 } { \sqrt { 3 } } } { \sqrt { a } } \geq { \sqrt { a } } .
$$

Here $0 < \varepsilon / ( 8 N _ { + } ) \le 1$ . Using $\mathsf T _ { k / 2 } ( t ) = \cosh ( ( k / 2 )$ arcosh t) $( t \geq 1 )$ , we have on $\Phi ( E _ { j } ^ { - } )$

$$
\mathsf { T } _ { k / 2 } ( t _ { j } ( z ) ) \geq \frac { 1 } { 2 } \exp \left( \frac { k } { 2 } \sqrt { \frac { \varepsilon } { 8 N _ { + } } } \right) \geq \frac { 1 } { 2 } \exp \bigl ( \log ( 4 \sqrt { N _ { - } } ) \bigr ) = 2 \sqrt { N _ { - } } .
$$

Hence the jth squared term is at least $4 N _ { - }$ . The other squared terms are nonnegative, and each term is at most 1 on $\Phi ( S ^ { + } )$ . Thus

$$
Q ( z ) \leq - N _ { - } ( \langle r , z \rangle ) ^ { k } \quad ( z \in \Phi ( S ^ { + } ) ) , \qquad Q ( z ) \geq 2 N _ { - } ( \langle r , z \rangle ) ^ { k } \quad ( z \in \Phi ( S ^ { - } ) ) .\tag{53}
$$

Although (52) is expressed through ratios, it extends to a polynomial on all of $\mathbb { R } ^ { n }$ . To see how the denominators cancel, write $\mathsf { T } _ { k / 2 } ( t ) = \sum _ { \ell = 0 } ^ { k / 2 } c _ { \ell } t ^ { \ell }$ . When $\langle r , z \rangle \neq 0$

$$
\begin{array} { l } { { \displaystyle ( \langle r , z \rangle ) ^ { k / 2 } \mathsf T _ { k / 2 } ( t _ { j } ( z ) ) } } \\ { { \displaystyle = \sum _ { \ell = 0 } ^ { k / 2 } c _ { \ell } \left[ \left( 1 + \frac { \varepsilon } { 4 N _ { + } } \right) \langle r , z \rangle - \frac { s _ { j } ( z ) } { 2 N _ { + } } \right] ^ { \ell } ( \langle r , z \rangle ) ^ { k / 2 - \ell } . } } \end{array}
$$

The right-hand side has no denominators and each term is homogeneous of degree $k / 2 ,$ so it is also defined when $\langle r , z \rangle = 0$ . Squaring the right-hand side, summing over $j ,$ and subtracting $2 N _ { - } ( \langle r , z \rangle ) ^ { k }$ gives an expression for $Q$ on all of $\mathbb { R } ^ { n }$ . Every term is homogeneous of degree $k ,$ and (53) rules out the zero polynomial, so the actual degree is the even integer $k .$

The Chebyshev recurrence gives $\begin{array} { r } { \sum _ { \ell = 0 } ^ { k / 2 } | c _ { \ell } | \le 3 ^ { k / 2 } } \end{array}$ . The linear forms used in homogenization are $\langle r , z \rangle$ and

$$
\left( 1 + \frac \varepsilon { 4 N _ { + } } \right) \langle r , z \rangle - \frac { s _ { j } ( z ) } { 2 N _ { + } } ,
$$

and their coeficient vectors all have norm at most $C e ^ { C D }$ . Squaring increases the sum of absolute scalar coeficients to at most $3 ^ { k }$ . Summing over $j$ and including $- 2 N _ { - } ( \langle r , z \rangle ) ^ { k }$ gives

$$
\begin{array} { r } { \sum _ { a } | c _ { a } | \prod _ { b } \| v _ { a b } \| \le ( N _ { - } 3 ^ { k } + 2 N _ { - } ) ( C e ^ { C D } ) ^ { k } } \\ { \le 3 N _ { - } ( 3 C e ^ { C D } ) ^ { k } \le e ^ { C D k } . } \end{array}
$$

The last step uses $k \geq c \log ( 2 N _ { - } )$ from (51) and $D \geq 1$ , with only an increase in the absolute constant. Thus both constructions satisfy (50).

Unified bounds for the degree and relative margin. In case one, $p \leq \varepsilon ^ { - 1 }$ directly controls $k ;$ to bound Dk, we combine the denominator of $p$ with the logarithm in D. By (47),

$$
\begin{array} { r l } & { p \log ( 4 N _ { + } ) \leq \left( 1 + \frac { \log ( 4 N _ { - } ) } { \log ( 1 6 N _ { + } ) } \right) \log ( 4 N _ { + } ) } \\ & { \qquad \leq \log ( 4 N _ { + } ) + \log ( 4 N _ { - } ) \leq C \log ( 4 N _ { + } N _ { - } ) . } \end{array}
$$

Thus $k \le 2 N _ { - } p$ and (25) give

$$
\begin{array} { r } { k \leq 2 N _ { - } \varepsilon ^ { - 1 } , \qquad D k \leq C N _ { - } \varepsilon ^ { - 1 } p \log ( 4 N _ { + } ) \leq C N _ { - } \varepsilon ^ { - 1 } \log ( 4 N _ { + } N _ { - } ) . } \end{array}
$$

The condition $p > \varepsilon ^ { - 1 }$ in case two forces $N _ { - }$ to be relatively large. Since $\varepsilon ^ { - 1 } \geq 1 6$ , we have $p \geq 2$ By the definition of the ceiling, this requires log $( 4 N _ { - } ) / \log ( 1 6 N _ { + } ) > 1$ , hence $N _ { - } > 4 N _ { + }$ . Also,

$$
\varepsilon ^ { - 1 } < p \leq 1 + \frac { \log ( 4 N _ { - } ) } { \log ( 1 6 N _ { + } ) } \leq C \log ( 2 N _ { - } ) .
$$

Together with (51), this gives

$$
\begin{array} { r l r } {  { k \le 2 + C \sqrt { \frac { N _ { + } } { \varepsilon } } \log ( 2 N _ { - } ) } } \\ & { } & { \le C \sqrt { N _ { - } } \log ^ { 3 / 2 } ( 2 N _ { - } ) \le C N _ { - } . } \end{array}
$$

The last comparison uses $\mathrm { s u p } _ { u \geq 1 } \log ^ { 3 } ( 2 u ) / u = 5 4 / e ^ { 3 } < 4$ , with an absolute constant. Again by (25),

$$
D k \leq C N _ { - } \varepsilon ^ { - 1 } \log ( 4 N _ { + } ) \leq C N _ { - } \varepsilon ^ { - 1 } \log ( 4 N _ { + } N _ { - } ) .
$$

Thus both cases give $k \le C N _ { - } \varepsilon ^ { - 1 }$ and the preceding bound for Dk. Substituting $\varepsilon _ { \mathrm { ~ y ~ } }$ ields

$$
\begin{array} { r l } & { k \le \frac { C N _ { - } ( 1 + R / \delta _ { 0 } ) ^ { 2 } } { \kappa _ { \operatorname* { m i n } } } , } \\ & { D k \le \frac { C N _ { - } ( 1 + R / \delta _ { 0 } ) ^ { 2 } } { \kappa _ { \operatorname* { m i n } } } \log ( 4 N _ { + } N _ { - } ) . } \end{array}\tag{54}
$$

The two cases are $p \leq \varepsilon ^ { - 1 }$ and $p > \varepsilon ^ { - 1 }$ , with equality included in case one. When $N _ { - } = 1$ , we have $p = 1$ , so case one applies; multiplying the linear polynomial by $\langle r , z \rangle$ gives $k = 2$ . For $N _ { + } = 1$ , the logarithms reduce to log 16 and log 4, so the bounds remain valid.

To apply random feature approximation, we must also express the margin relative to $\| \Phi ( x ) \|$ instead of $\| U _ { x } \|$ . Since $\lambda \geq \delta _ { 0 }$ ，

$$
\frac { \| U _ { x } \| } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } = \sqrt { \frac { \| x \| ^ { 2 } + \delta _ { 0 } ^ { 2 } } { \| x \| ^ { 2 } + \lambda ^ { 2 } } } \ge \frac { \delta _ { 0 } } { \lambda } .
$$

Combining the upper bound in (11) with $\theta \ge \sqrt { \delta _ { 0 } / ( 2 \lambda ) }$ gives

$$
\frac { \theta \| U _ { x } \| } { \| \Phi ( x ) \| } \geq c \sqrt { \frac { \delta _ { 0 } } { \lambda } } \frac { \| U _ { x } \| } { \sqrt { \| x \| ^ { 2 } + \lambda ^ { 2 } } } \geq c \left( \frac { \delta _ { 0 } } { \lambda } \right) ^ { 3 / 2 } .
$$

Case one already satisfies (49); in case two, (53) and (38) give $| Q ( \Phi ( x ) ) | \geq ( \theta \| U _ { x } \| / 2 ) ^ { k }$ , so (49) holds as well. Thus in both cases,

$$
{ \frac { | Q ( \Phi ( x ) ) | } { \| \Phi ( x ) \| ^ { k } } } \geq \left( { \frac { c \varepsilon } { N _ { + } } } \right) ^ { k } \left( { \frac { \delta _ { 0 } } { \lambda } } \right) ^ { 3 k / 2 } \qquad ( x \in S ^ { + } \cup S ^ { - } ) .
$$

By the degree lower bound in (25),

$$
\log \frac { N _ { + } } { c \varepsilon } \leq C \left[ \log ( 4 N _ { + } ) + \varepsilon ^ { - 1 } \right] \leq C \varepsilon ^ { - 1 } \log ( 4 N _ { + } ) \leq C D .
$$

Here $c > 0$ is the absolute constant above, so these absorptions are independent of the geometric parameters. It follows that $| Q ( \Phi ( x ) ) | / \| \Phi ( x ) \| ^ { k }$ is at least $e ^ { - C D k } ( \delta _ { 0 } / \lambda ) ^ { \mathbf { \hat { 3 } } k / 2 }$ . Together with the opposite signs already obtained on the two classes, we may take

$$
\zeta = e ^ { - C D k } \left( \frac { \delta _ { 0 } } { \lambda } \right) ^ { 3 k / 2 } ,\tag{55}
$$

so that $Q ( z ) / \| z \| ^ { k - 1 }$ is at most − $- \zeta \| z \|$ on $\Phi ( S ^ { + } )$ and at least $\zeta \| z \|$ on $\Phi ( S ^ { - } )$

## B.2 Random ReLU representation of the common polynomial

The Hermite identity gives a random ReLU expectation with the same separating signs. The multiplier estimate controls the uniform approximation error and the coeficient norm in (59). These bounds preserve the signs and yield a margin bound for $\langle v , F ( x ) \rangle$

Let $g \sim N ( 0 , I _ { n } )$ and $G \sim N ( 0 , 1 )$ . For even $k \geq 2$ , let $\mu _ { k } ( t ) = \mathbb { E } _ { G } [ \mathrm { H e } _ { k } ( G ) \mathrm { R e L U } ( G + t ) ] =$ $\varphi ( t ) \mathrm { H e } _ { k - 2 } ( - t )$ . Using (50), set

$$
{ \cal A } ( g ) = \frac { 1 } { \mu _ { k } ( 0 ) } \sum _ { a } c _ { a } : \prod _ { b = 1 } ^ { k } \langle v _ { a b } , g \rangle : \ : .
$$

Here the colons denote the Wick polynomial of order k associated with $\textstyle \prod _ { b = 1 } ^ { k } \langle v _ { a b } , g \rangle$ , namely the orthogonal projection of this product onto the Hermite subspace of order k in Gaussian $L ^ { 2 } ~ [ 1 5 $ Definition 3.1]. Its generating function is

$$
: \prod _ { b = 1 } ^ { k } \langle v _ { a b } , g \rangle : = \left. \partial _ { t _ { 1 } } \cdot \cdot \cdot \partial _ { t _ { k } } \exp \Biggl ( \langle g , \sum _ { b } t _ { b } v _ { a b } \rangle - { \textstyle \frac { 1 } { 2 } } \| \sum _ { b } t _ { b } v _ { a b } \| ^ { 2 } \Biggr ) \right| _ { t = 0 } .
$$

Fix $z \neq 0$ and a scalar bias $b \in \mathbb { R }$ . First condition on $\langle g , z / \| z \| \rangle$ and average over the orthogonal Gaussian component $P _ { z ^ { \perp } } g$ , then integrate over $\langle g , z / \lVert z \rVert \rangle \sim N ( 0 , 1 )$ to obtain

$$
\mathbb { E } _ { g } A ( g ) \mathrm { R e L U } ( \langle g , z \rangle + b ) = \frac { \mu _ { k } ( b / \| z \| ) } { \mu _ { k } ( 0 ) } \frac { Q ( z ) } { \| z \| ^ { k - 1 } } .\tag{56}
$$

Since all linear factors of Q come from $w _ { 1 } , \ldots , w _ { N _ { - } } , r ,$ , A also depends only on their span. Keeping the first-layer realization, $Q , z ,$ , and the scalar bias b fixed, the Wick isometry formula [15, Chapters 3–4] gives

$$
\mathbb { E } _ { g } \Bigg [ \Bigg ( : \prod _ { b = 1 } ^ { k } \langle v _ { a b } , g \rangle : \Bigg ) ^ { 2 } \Bigg ] = \sum _ { \pi } \prod _ { b = 1 } ^ { k } \langle v _ { a b } , v _ { a , \pi ( b ) } \rangle \leq k ! \prod _ { b = 1 } ^ { k } \| v _ { a b } \| ^ { 2 } ,
$$

where $\pi$ runs over all permutations of [k]. The $L ^ { 2 }$ triangle inequality gives

$$
\| \mathcal { A } \| _ { L ^ { 2 } } \leq \frac { \sqrt { k ! } } { | \mu _ { k } ( 0 ) | } \sum _ { a } | c _ { a } | \prod _ { b } \| v _ { a b } \| .
$$

To verify (56), fix the value of $\langle g , z / \| z \| \rangle$ , average the generating function over the orthogonal component $P _ { z ^ { \perp } } g$ , and then take the mixed derivative at $t = 0$ . This gives

$$
\mathbb { E } _ { g } \bigg [ \colon \prod _ { b = 1 } ^ { k } \langle v _ { a b } , g \rangle \colon \bigg | \langle g , z / \| z \| \rangle \bigg ] = \mathrm { H e } _ { k } ( \langle g , z / \| z \| \rangle ) \prod _ { b = 1 } ^ { k } \langle v _ { a b } , z / \| z \| \rangle .
$$

Combining this with ReLU $( \langle g , z \rangle \ : + \ : b ) \ : \ : = \ : \ : \| z \| \ : \mathrm { R e L U } ( \langle g , z / \| z \| \rangle \ : + \ : b / \| z \| )$ and averaging over $\langle g , z / \lVert z \rVert \rangle \sim N ( 0 , 1 )$ yields (56); two integrations by parts on $( - t , \infty )$ give the expression for $\mu _ { k } ( t )$ above.

By (50) and $\sqrt { k ! } / | \mu _ { k } ( 0 ) | \le C ( k + 1 ) ^ { 3 / 2 }$ , we obtain

$$
\| \mathcal { A } \| _ { L ^ { 2 } } \leq e ^ { C D k } .\tag{57}
$$

For the random ReLU expectation in (56) to preserve the classification signs and relative separation margin of $Q$ , we must control the multiplier introduced by the bias, $\mu _ { k } ( b / \| z \| ) / \mu _ { k } ( 0 )$ Write $k - 2 = 2 h$ . The explicit Hermite coeficients give

$$
\left| \frac { \mathrm { H e } _ { 2 h } ( t ) } { \mathrm { H e } _ { 2 h } ( 0 ) } - 1 \right| \leq \cosh ( \sqrt { 2 h } | t | ) - 1 .
$$

Thus $\mu _ { k } ( t ) / \mu _ { k } ( 0 ) \geq 1 / 2$ when $| t | \leq 1 / ( 2 \sqrt { k } )$ . For $k = 2$ , verify this directly using $e ^ { - t ^ { 2 } / 2 }$ . This condition restricts only the size of $| b | / \| z \|$ and applies to unbounded feature sets.

When approximating (56) by finitely many samples, the coeficient $\scriptstyle A ( g )$ is correlated with the ReLU feature in the same network row and may be unbounded. The following multiplier estimate uses its polynomial structure and the dimension of the subspace on which it depends to give uniform bounds on high moments.

Lemma B.1 (Polynomial multiplier estimate). Let $g \sim N ( 0 , I _ { n } )$ and let $A : \mathbb { R } ^ { n }  \mathbb { R }$ be a polynomial of total degree at most $k ,$ with $\| A \| _ { L ^ { 2 } } \leq B$ . Suppose there exists a fixed s-dimensional linear subspace $V \subset \mathbb { R } ^ { n }$ such that $A ( x ) = A ( P _ { V } x )$ for every $x \in \mathbb { R } ^ { n }$ , where $P _ { V }$ is the orthogonal projection onto V. Let $( g , \beta )$ and $\left( g _ { a } , \beta _ { a } \right) / a \in \left[ M \right] )$ be independent and identically distributed, with $\beta$ independent of g and $| \beta | \le 1$ almost surely. For a nonempty compact set $T \subset { \sqrt { 2 } } B _ { 2 } ^ { n + 1 }$ , let

$$
Z = \operatorname* { s u p } _ { t \in T } \left| \frac { 1 } { M } \sum _ { a = 1 } ^ { M } A ( g _ { a } ) \operatorname { R e L U } ( \langle ( g _ { a } , \beta _ { a } ) , t \rangle ) - \mathbb { E } _ { g , \beta } A ( g ) \operatorname { R e L U } ( \langle ( g , \beta ) , t \rangle ) \right| .
$$

Then for $p \geq 2$

$$
\| Z \| _ { L ^ { p } } \leq \frac { C 2 ^ { k / 2 } \sqrt { s + 1 } B } { \sqrt { M } } p ^ { k / 2 } [ w ( T ) + p ] ,\tag{58}
$$

$$
B ( p - 1 ) ^ { k / 2 } \geq \left\| \left( M ^ { - 1 } \sum _ { a } A ( g _ { a } ) ^ { 2 } \right) ^ { 1 / 2 } \right\| _ { L ^ { p } } .\tag{59}
$$

There exists a positive constant C, independent of the sets and η, as well as of k, s.

Proof. Write $A _ { a } = A ( g _ { a } )$ and let $\epsilon _ { a }$ be independent Rademacher signs, independent of the sample rows. Minkowski’s inequality and Gaussian hypercontractivity [15, Theorem 5.1] give

$$
\left\| \left( M ^ { - 1 } \sum _ { a } A _ { a } ^ { 2 } \right) ^ { 1 / 2 } \right\| _ { L ^ { p } } \leq \| A \| _ { L ^ { p } } \leq B ( p - 1 ) ^ { k / 2 } ,
$$

which proves (59).

For fixed sample rows, set $f _ { a } ( u ) = A _ { a } \mathrm { R e L U } ( u / A _ { a } )$ if $A _ { a } \neq 0$ and $f _ { a } = 0$ otherwise. These maps vanish at zero and are 1-Lipschitz. Symmetrization and conditional contraction [17, Chapter 4], applied to $A _ { a } \langle ( g _ { a } , \beta _ { a } ) , t \rangle$ and then averaged over the rows, give

$$
\| Z \| _ { L ^ { p } } \leq \frac { 4 } { M } \left\| \operatorname* { s u p } _ { t \in T } \left. \langle \sum _ { a } \epsilon _ { a } A _ { a } ( g _ { a } , \beta _ { a } ) , t \rangle \right. \right\| _ { L ^ { p } } .
$$

Adding the origin does not change the absolute suprema, so $0 \in T$ is not required.

Decompose $g _ { a } = g _ { a , V } + g _ { a , V ^ { \bot } }$ , where $g _ { a , V } = P _ { V } g _ { a }$ , and write $g _ { V } = P _ { V } g$ . We bound the $( g _ { a , V ^ { \bot } } , 0 )$ and $\left( g _ { a , V } , \beta _ { a } \right)$ components separately. Conditional on $\{ g _ { a , V } , \beta _ { a } , \epsilon _ { a } \} _ { a = 1 } ^ { M }$ , the vector $\textstyle \sum _ { a } \epsilon _ { a } A _ { a } g _ { a , V ^ { \bot } }$ is centered Gaussian with covariance $( \sum _ { a } A _ { a } ^ { 2 } ) I _ { V ^ { \bot } }$ . For $P ( u , h ) = \left( P _ { V ^ { \perp } } u , 0 \right)$ , Gaussian concentration and comparison with an anchor point [30, Proposition 2.6.6, Theorem 5.2.3, and Exercise 7.20] bound the conditional $L ^ { p }$ norm of the absolute supremum by $C ( \sum _ { a } A _ { a } ^ { 2 } ) ^ { 1 / 2 } [ w ( P T ) + \sqrt { p } ]$ . Averaging and using the sum-of-squares estimate above and $w ( P T ) \leq w ( T )$ [30, Proposition $7 . 5 . 2 ( \mathrm { g } ) ]$ yield

$$
\left\| \operatorname* { s u p } _ { t \in T } \left. ( \sum _ { a } \epsilon _ { a } A _ { a } ( g _ { a , V ^ { \perp } } , 0 ) , t ) \right. \right\| _ { L ^ { p } } \leq C \sqrt { M } \| A \| _ { L ^ { p } } [ w ( T ) + \sqrt { p } ] .
$$

The scalar Rademacher moment bound [30, Theorem 2.7.5] and Minkowski’s inequality give, for deterministic vectors $u _ { a }$

$$
\left( \mathbb { E } _ { \epsilon } \| \sum _ { a } \epsilon _ { a } u _ { a } \| ^ { p } \right) ^ { 1 / p } \leq C \sqrt { p } \left( \sum _ { a } \| u _ { a } \| ^ { 2 } \right) ^ { 1 / 2 } .
$$

Condition on all sample rows, apply this with $u _ { a } = A _ { a } ( g _ { a , V } , \beta _ { a } )$ , and average over the rows. Hölder’s inequality and hypercontractivity then give

$$
\begin{array} { r l r } {  { \bigg \| \operatorname* { s u p } _ { t \in T } \bigg | \langle \sum _ { a } \epsilon _ { a } A _ { a } ( g _ { a , V } , \beta _ { a } ) , t \rangle \bigg | \bigg \| _ { L ^ { p } } \leq C \sqrt { p M } \| A ( g ) ( g _ { V } , \beta ) \| _ { L ^ { p } } } } \\ & { } & { \leq C \sqrt { p M } \| A \| _ { L ^ { 2 p } } \| ( g _ { V } , \beta ) \| _ { L ^ { 2 p } } } \\ & { } & { \leq C \sqrt { p M } B ( 2 p - 1 ) ^ { k / 2 } ( \sqrt { s + 1 } + C \sqrt { p } ) . } \end{array}
$$

Here we used $T \subset \sqrt { 2 } B _ { 2 } ^ { n + 1 }$ and $\| ( g _ { V } , \beta ) \| _ { L ^ { 2 p } } \leq \sqrt { s + 1 } + C \sqrt { p } .$ which follows from $| \beta | \le 1$ and Gaussian concentration.

Combining the two component bounds $\mathrm { g i }$ ves

$$
\begin{array} { c } { { \displaystyle { \| Z \| _ { L ^ { p } } \leq \frac { C B } { \sqrt { M } } \{ ( p - 1 ) ^ { k / 2 } [ w ( T ) + \sqrt { p } ] \ } } } \\ { { + \sqrt { p } ( 2 p - 1 ) ^ { k / 2 } ( \sqrt { s + 1 } + C \sqrt { p } ) \} . } } \end{array}
$$

Using $2 p - 1 \leq 2 p , p \geq 2$ , and $w ( T ) \geq 0$ gives (58), with constants independent of $k , s , p$ . All suprema may be taken over a countable dense subset of $T ,$ so the random variables above are measurable. □

## B.3 Finite-width approximation and parameter estimates

This subsection uses a finite-width network to approximate the preceding expectation. The resulting width and margin estimates complete the proof of Theorem 2.7.

Finite-width realization and probability estimates. Fix a first-layer realization $( W , b )$ satisfying the preceding estimates. Retain the corresponding A and degree $k ,$ and take $z \in$ $\Phi ( S ^ { + } \cup S ^ { - } )$ . For the second layer in (5), set $\widehat { \lambda } = \delta _ { 0 }$ . Let $\widehat { W } _ { a , \cdot } \in \mathbb { R } ^ { n }$ denote row a of $\widehat { W }$ and $\widehat { b } _ { a }$ component a of ${ \widehat { b } } ,$ where $a \in [ \widehat { n } ]$ . Then $\widehat { W } _ { a , \cdot } \sim N ( 0 , I _ { n } )$ and $\widehat { b } _ { a } \sim \mathrm { U n i f } [ - \delta _ { 0 } , \delta _ { 0 } ]$ ], with all row vectors and bias components mutually independent.

Let $g \sim N ( 0 , I _ { n } )$ , and let the scalar $\widehat { b }$ follow the uniform bias distribution $\mathrm { U n i f } [ - \delta _ { 0 } , \delta _ { 0 } ]$ conditional on $\lvert \widehat { b } \rvert \leq \lambda / ( 1 6 \sqrt { k } )$ , independently of $g$ . Write $f _ { 0 } ( z ) = Q ( z ) / \| z \| ^ { k - 1 }$ and define

$$
f ( \boldsymbol { z } ) = \mathbb { E } _ { \boldsymbol { g } , \widehat { \boldsymbol { b } } } \left[ A ( \boldsymbol { g } ) \operatorname { R e L U } ( \langle \boldsymbol { g } , \boldsymbol { z } \rangle + \widehat { \boldsymbol { b } } ) \right] = f _ { 0 } ( \boldsymbol { z } ) \mathbb { E } _ { \widehat { \boldsymbol { b } } } \left[ \frac { \mu _ { k } ( \widehat { \boldsymbol { b } } / \| \boldsymbol { z } \| ) } { \mu _ { k } ( 0 ) } \right] ,
$$

where the second equality follows from (56), with z and the first-layer realization fixed. All sums without a specified range in this section run only over accepted coordinates.

Neuron selection and definition of v. For $a \in [ \widehat { n } ]$ , select the neurons satisfying $| \widehat { b } _ { a } | \leq \lambda / ( 1 6 \sqrt { k } )$ and denote the actual number accepted by

$$
M = \sum _ { a = 1 } ^ { \widehat { n } } \mathbf { 1 } _ { \{ | \widehat { b } _ { a } | \leq \lambda / ( 1 6 \sqrt { k } ) \} } .
$$

Here $M$ is a random count, distinct from the total width $\widehat { n } .$ . When $M \geq 1$ and the denominator below is nonzero, define the coordinates of $\boldsymbol { v } \in \mathbb { R } ^ { \widehat { n } }$ by

$$
v _ { a } = \left\{ - \frac { \mathcal { A } ( \widehat { W } _ { a , \cdot } ) } { \left( \sum _ { \ell } \mathcal { A } ( \widehat { W } _ { \ell , \cdot } ) ^ { 2 } \right) ^ { 1 / 2 } } , \quad | \widehat { b } _ { a } | \leq \lambda / ( 1 6 \sqrt { k } ) , \right.
$$

By definition, $\lVert \boldsymbol { v } \rVert = 1$ ; the count and approximation bounds below ensure both conditions hold. When $M \geq 1$ , define

$$
f _ { M } ( z ) = \frac { 1 } { M } \sum _ { a } A ( \widehat { W } _ { a , \cdot } ) \operatorname { R e L U } ( \langle \widehat { W } _ { a , \cdot } , z \rangle + \widehat { b } _ { a } ) .
$$

Each neuron has acceptance probability min $\{ 1 , \lambda / ( 1 6 \delta _ { 0 } \sqrt { k } ) \} \ge 1 / ( 1 6 \sqrt { k } )$ , using $\lambda \geq \delta _ { 0 }$ . Keep the first layer $( W , b )$ fixed and also fix the values of all acceptance indicators $\{ \mathbf { 1 } _ { \{ | \widehat { b } _ { a } | \leq \lambda / ( 1 6 \sqrt { k } ) \} } \} _ { a \in [ { \widehat { n } } ] } .$ Under this conditional distribution, the accepted rows remain independent, with $\widehat { W } _ { a , \cdot } \sim N ( 0 , I _ { n } )$ and $\widehat { b } _ { a }$ uniform on $[ - \delta _ { 0 } , \delta _ { 0 } ] \cap [ - \lambda / ( 1 6 \sqrt { k } ) , \lambda / ( 1 6 \sqrt { k } ) ]$ , and the weights and biases are independent. Thus the multiplier estimate applies to these coordinates.

Expectation margin and uniform approximation. By (11), $\| z \| \geq \lambda / 8 .$ , so

$$
\frac { | \widehat { b } _ { a } | } { \| z \| } \leq \frac { \lambda / ( 1 6 \sqrt { k } ) } { \lambda / 8 } = \frac { 1 } { 2 \sqrt { k } } .
$$

Thus $\mu _ { k } ( \widehat { b } _ { a } / \| z \| ) / \mu _ { k } ( 0 ) \geq 1 / 2$ simultaneously for all z. By the definition of $f$ and (55), $f ( z )$ is at most $- \zeta \| z \| / 2$ on $\Phi ( S ^ { + } )$ and at least $\zeta \| z \| / 2$ on $\Phi ( S ^ { - } )$

Continue to fix the first layer $( W , b )$ and acceptance indicators such that $M \geq 1$ . In Lemma B.1, take $A = \mathcal { A } , \beta = 8 \widehat { b } / \lambda$ , and

$$
t _ { z } = \left( \frac { z } { \| z \| } , \frac { \lambda } { 8 \| z \| } \right) , \qquad T = \mathrm { c l } \{ t _ { z } : z \in \Phi ( S ^ { + } \cup S ^ { - } ) \} \subset \sqrt { 2 } B _ { 2 } ^ { n + 1 } .
$$

The selection above ensures $| \beta | \le 1$ , and the positive homogeneity of ReLU gives

$$
\frac { \mathrm { R e L U } ( \langle g , z \rangle + \widehat { b } ) } { \| z \| } = \mathrm { R e L U } ( \langle ( g , \beta ) , t _ { z } \rangle ) .
$$

Thus $Z$ in the lemma is

$$
Z = \operatorname* { s u p } _ { z \in \Phi ( S ^ { + } \cup S ^ { - } ) } { \frac { | f _ { M } ( z ) - f ( z ) | } { \| z \| } } .
$$

By (41),

$$
w ( T ) ^ { 2 } \leq C ( 1 + R / \delta _ { 0 } ) ^ { 2 } \left[ 1 + w \left( { \mathrm { c l } } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) \right) ^ { 2 } + \log { \frac { 2 4 } { \eta } } \right] .
$$

The last coordinate $\frac { \lambda } { 8 \| z \| }$ increases the Gaussian width by at most $\scriptstyle { \sqrt { 2 / \pi } }$ . Both the degree estimate after (48) and (51) give the degree bound $k \geq c \log ( 2 N _ { - } )$ for the common separating polynomial $Q .$ Together with $s \leq N _ { - } + 1$ , this yields

$$
2 ^ { k } ( s + 1 ) \le 2 ^ { k } ( N _ { - } + 2 ) = \exp \bigl ( k \log 2 + \log ( N _ { - } + 2 ) \bigr ) \le e ^ { C k } .
$$

Take moment order $p = \log ( 2 4 / \eta ) > 2 \colon$ ; in this section, $p$ no longer denotes the power used in (47) to combine the ratios. By (58), to ensure $e \| Z \| _ { L ^ { p } } \leq \zeta / 4$ , it sufices that

$$
M \geq C 2 ^ { k } ( s + 1 ) \| A \| _ { L ^ { 2 } } ^ { 2 } \zeta ^ { - 2 } p ^ { k } [ w ( T ) + p ] ^ { 2 } .
$$

Substitute (57), (55), and the Gaussian width bound above, and use (25) to absorb $( 1 + R / \delta _ { 0 } ) ^ { 2 }$ and $2 ^ { k } ( s + 1 )$ into $e ^ { \boldsymbol { C } D \boldsymbol { k } }$ . This gives

$$
\begin{array} { l } { { \displaystyle M \geq \exp ( C D k ) \left( \frac { \lambda } { \delta _ { 0 } } \right) ^ { 3 k } \left( \log \frac { 2 4 } { \eta } \right) ^ { k } } } \\ { { \displaystyle ~ \times \left[ 1 + w \left( \mathrm { c l } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) \right) ^ { 2 } + \log ^ { 2 } \frac { 2 4 } { \eta } \right] . } } \end{array}\tag{60}
$$

Here $( \lambda / \delta _ { 0 } ) ^ { 3 k }$ comes from $\zeta ^ { - 2 }$ . Let $M _ { 0 }$ be the ceiling of the deterministic right-hand side. If

$$
\widehat { n } \operatorname* { m i n } \{ 1 , \lambda / ( 1 6 \delta _ { 0 } \sqrt { k } ) \} \geq \operatorname* { m a x } \{ 2 M _ { 0 } , 8 p \} ,
$$

we keep the first layer $( W , b )$ fixed and apply the Chernof bound only to the distribution of acceptance indicators induced by the second-layer bias vector ${ \widehat { b } } .$ This gives

$$
\mathbb { P } \bigg \{ M < \frac { \widehat { n } } { 2 } \operatorname* { m i n } \bigg \{ 1 , \frac { \lambda } { 1 6 \delta _ { 0 } \sqrt { k } } \bigg \} \bigg | W , b \bigg \} \leq e ^ { - p } = \frac { \eta } { 2 4 } .\tag{61}
$$

Outside this event,

$$
M \geq \frac { \widehat { n } } { 2 } \operatorname* { m i n } \{ 1 , \lambda / ( 1 6 \delta _ { 0 } \sqrt { k } ) \} \geq M _ { 0 } , \qquad \sqrt { M / \widehat { n } } \geq c k ^ { - 1 / 4 } .
$$

Thus a total width $\widehat { n } \geq C \sqrt { k } ( M _ { 0 } + p )$ sufices. Fix the first layer $( W , b )$ and any acceptance indicator configuration satisfying this count condition. Markov’s inequality under the conditional joint distribution of the accepted rows gives

$$
\mathbb { P } \{ Z > \zeta / 4 \} \le e ^ { - p } = \frac { \eta } { 2 4 } ,\tag{62}
$$

where $\mathbb { P }$ denotes this conditional probability. Hence $f _ { M } ( z )$ preserves the sign of $f ( z )$ and has margin at least $\zeta / 4$ relative to $\| z \|$ . This conditional probability bound is uniform over all $( W , b )$ satisfying the six groups of first-layer events and all qualifying acceptance indicator configurations.

Coeficient bound and separation margin. Still fixing the first layer $( W , b )$ and acceptance indicator configuration above, apply (59), (57), and the same Markov estimate under the conditional joint distribution of the accepted rows. Under the same conditioning, this gives

$$
\mathbb { P } \Bigg \{ \Bigg ( \frac { 1 } { 2 M } \sum _ { a } A ( \widehat { W } _ { a , \cdot } ) ^ { 2 } \Bigg ) ^ { 1 / 2 } > e ^ { C D k } \left( \log \frac { 2 4 } { \eta } \right) ^ { k / 2 } \Bigg \} \leq \frac { \eta } { 2 4 } .\tag{63}
$$

On the joint success event, $M \ge M _ { 0 } \ge 1$ , and the strict separation margin for $f _ { M }$ in the preceding paragraph ensures $\textstyle \sum _ { a } { \mathcal { A } } ( { \widehat { W } } _ { a , \cdot } ) ^ { 2 } \ > \ 0$ , so the vector v defined above is valid. The second-layer normalization gives

$$
\langle v , \widehat { \Phi } ( z ) \rangle = - \sqrt { \frac { M } { \widehat { n } } } \frac { f _ { M } ( z ) } { \left( ( 2 M ) ^ { - 1 } \sum _ { a } \mathcal { A } ( \widehat { W } _ { a , \cdot } ) ^ { 2 } \right) ^ { 1 / 2 } } .
$$

$f _ { M } ( z )$ is at most $- \zeta \| z \| / 4$ on $\Phi ( S ^ { + } )$ and at least $\zeta \| z \| / 4$ on $\Phi ( S ^ { - } )$ , so the minus sign gives the sign orientation in (13). Since $\lVert z \rVert \geq \lambda / 8 \geq \delta _ { 0 } / 8$ , absorbing $k ^ { - 1 / 4 }$ into $e ^ { - C D k }$ and using the denominator bound gives the deterministic margin

$$
c \delta _ { 0 } e ^ { - C D k } \left( \frac { \delta _ { 0 } } { \lambda } \right) ^ { 3 k / 2 } \left( \log \frac { 2 4 } { \eta } \right) ^ { - k / 2 } .\tag{64}
$$

Parameter simplification and joint probability. In (60), the factor $e ^ { C D k }$ uses the bound for Dk in (54), while $( \lambda / \delta _ { 0 } ) ^ { 3 k }$ and $( \log ( 2 4 / \eta ) ) ^ { k }$ use only its bound for k. To simplify the discussion, write

$$
L = \log \left( \frac { 4 N _ { + } N _ { - } \lambda } { \delta _ { 0 } } \log \frac { 2 } { \eta } \right) .
$$

Since $\lambda / \delta _ { 0 } \ge 1$ and $0 < \eta < 1$ , we have $L \geq \log ( 4 \log 2 ) > 1$ and

$$
\log ( 4 N _ { + } N _ { - } ) \leq C L , \qquad \log \frac { 2 \lambda } { \delta _ { 0 } } \leq C L , \qquad \log \log \frac { 2 4 } { \eta } \leq C L .
$$

Substituting the respective bounds in (54) gives

$$
C D k + 3 k \log { \frac { \lambda } { \delta _ { 0 } } } + k \log \log { \frac { 2 4 } { \eta } } \leq \frac { C N _ { - } ( 1 + R / \delta _ { 0 } ) ^ { 2 } } { \kappa _ { \operatorname* { m i n } } } L .
$$

On the other hand, (32) gives

$$
1 + w \left( \mathrm { c l } \bigcup _ { i , j } ( K _ { i j } \cap S ^ { d - 1 } ) \right) ^ { 2 } + \log ^ { 2 } \frac { 2 4 } { \eta } \leq C \left[ 1 + \operatorname* { m a x } _ { i , j } w ( K _ { i j } \cap S ^ { d - 1 } ) ^ { 2 } \right] e ^ { C L } .
$$

Using the bound for k in (54) again, absorb $\sqrt { k } , p = \log ( 2 4 / \eta )$ , and the factor $e ^ { C L }$ above into the preceding exponent estimate. This proves (12). Applying the same estimates to (64), we may take

$$
t = c \delta _ { 0 } \exp \Biggl \{ - \frac { C N _ { - } ( 1 + R / \delta _ { 0 } ) ^ { 2 } } { \kappa _ { \operatorname* { m i n } } } \log \biggl ( \frac { 4 N _ { + } N _ { - } \lambda } { \delta _ { 0 } } \log \frac { 2 } { \eta } \biggr ) \Biggr \} > 0 .
$$

Finally, the first-layer events have total failure probability at most $\eta / 4$ , as shown in Section A.5. Fix a first layer satisfying these events. By (61), the count condition fails with conditional probability at most $\eta / 2 4$ . For every acceptance indicator configuration meeting this condition, (62) and (63) each contribute at most $\eta / 2 4$ . Averaging over these configurations and applying a union bound gives conditional second-layer failure probability at most $3 \eta / 2 4$ . Averaging over $( W , b )$ and adding the first-layer failure probability yields

$$
\frac { \eta } { 4 } + \frac { 3 \eta } { 2 4 } = \frac { 9 \eta } { 2 4 } < \eta .
$$

The positive constant $C$ is independent of the sets and $\eta ,$ and c is an absolute constant. This proves Theorem 2.7.

## C Proofs of necessary width and cone geometry

This appendix proves the width lower bounds and geometric separation results. Subsection C.1 proves the finite convex-cover characterization (Theorem 2.1). Subsection C.2 shows why one hidden layer may fail (Example 2.2). Subsection C.3 characterizes random afine separation through pointedness (Proposition 2.4). Subsection C.4 proves the nonpointed counterexample (Example 3.1). Subsection C.5 computes the geometric parameters and Gaussian width of the lower-bound construction (Example 3.2). Finally, Subsection C.6 proves the width lower bounds in Example 3.2 by constructing points from diferent classes with identical features.

## C.1 Finite convex-cover characterization

Proof of Theorem 2.1. Assume (i), with scalar output $h \geq m$ on $S ^ { + }$ and $h \leq - m$ on $S ^ { - }$ for some $m > 0$ . The finitely many activation patterns of a finite ReLU network [2, Definition 3 and

Theorem 2.1] give closed polyhedral cells $Q _ { \ell }$ covering $\mathbb { R } ^ { d }$ , on each of which h agrees with an afine map $h _ { \ell }$ . With $\alpha = m / 2$ , define

$$
\begin{array} { r l r } { C _ { \ell } ^ { + } = Q _ { \ell } \cap \{ h _ { \ell } \geq \alpha \} , } & { { } } & { C _ { \ell } ^ { - } = Q _ { \ell } \cap \{ h _ { \ell } \leq - \alpha \} , } \end{array}
$$

and discard empty cells. A global Lipschitz constant $L _ { h } \mathrm { \ g i v e s }$ , for $x \in C _ { \ell } ^ { + }$ and $y \in C _ { r } ^ { - }$

$$
m = 2 \alpha \leq h ( x ) - h ( y ) \leq L _ { h } \| x - y \| .
$$

Thus every opposite-label pair has distance at least $m / L _ { h }$ , proving (ii).

Assume (ii), and set $\delta _ { * } : = \mathrm { m i n } _ { i , j } \mathrm { d i s t } ( C _ { i } ^ { + } , C _ { j } ^ { - } ) > 0$ . The open convex sets $C _ { i } ^ { + } + \left( \delta _ { * } / 4 \right)$ int $B _ { 2 } ^ { d }$ and $C _ { j } ^ { - } + \left( \delta _ { * } / 4 \right)$ int $B _ { 2 } ^ { d }$ are disjoint. The strong separation theorem [25, Theorem 11.4] gives an afine function $\ell _ { i j }$ which, after normalization, satisfies

$$
\ell _ { i j } ( x ) \leq - \delta _ { * } / 4 \quad ( x \in C _ { i } ^ { + } ) , \qquad \ell _ { i j } ( y ) \geq \delta _ { * } / 4 \quad ( y \in C _ { j } ^ { - } ) .
$$

The distance need not be attained. The first hidden layer contains $z _ { i j } ( x ) = \mathrm { R e L U } ( \ell _ { i j } ( x ) )$ ; the second contains

$$
q _ { i } ( x ) = \mathrm { R e L U } \left( \frac { \delta _ { * } } { 4 } - \sum _ { j = 1 } ^ { M _ { - } } z _ { i j } ( x ) \right) .
$$

For $x \in S ^ { + }$ , choosing $i _ { 0 }$ with $x \in C _ { i _ { 0 } } ^ { + }$ gives $q _ { i _ { 0 } } ( x ) = \delta _ { * } / 4$ . For $y \in S ^ { - }$ , choose $j _ { 0 }$ with $y \in C _ { j _ { 0 } } ^ { - }$ . Then $z _ { i j _ { 0 } } ( y ) \geq \delta _ { * } / 4$ and $q _ { i } ( y ) = 0$ for every i. Consequently,

$$
h _ { \mathrm { o u t } } ( x ) = \sum _ { i } q _ { i } ( x ) - \frac { \delta _ { * } } { 8 }
$$

has opposite signs with a uniform positive margin, proving (iii). The implication $( \mathrm { i i i } ) { \Rightarrow } ( \mathrm { i } )$ is immediate. □

## C.2 Necessity of two hidden layers

Proof of Example 2.2. Suppose that $h > 0$ on $S ^ { + }$ and $h < 0$ on $S ^ { - }$ <sup>−</sup>. Absorb terms with $v _ { k } = 0$ into c and consider the recession function

$$
h _ { \infty } ( v ) = \operatorname* { l i m } _ { r \to \infty } \frac { h ( r v ) } { r } = \sum _ { k = 1 } ^ { m } a _ { k } \operatorname { R e L U } ( \langle v _ { k } , v \rangle ) .
$$

For $s \in \{ - 1 , 1 \} ^ { 3 }$ , let $O _ { s } = \{ v \in \mathbb { R } ^ { 3 } : s _ { r } v _ { r } > 0 , \ r = 1 , 2 , 3 \}$ . The labels imply $\chi ( s ) h _ { \infty } ( v ) \geq 0$ on $O _ { s }$ Here $\chi ( s ) = s _ { 1 } s _ { 2 } s _ { 3 } \in \{ - 1 , 1 \}$ is the class label of $E _ { s }$ in (3). Using $\mathrm { R e L U } ( t ) = ( t + | t | ) / 2$ , write

$$
h _ { \infty } ( v ) = q \cdot v + \frac { 1 } { 2 } \sum _ { k } a _ { k } \lvert \langle v _ { k } , v \rangle \rvert , \qquad q = \frac { 1 } { 2 } \sum _ { k } a _ { k } v _ { k } .
$$

The inequalities at $v \in O _ { s }$ and $- v \in O _ { - s }$ give $\chi ( s ) ( q \cdot v \pm E ( v ) ) \geq 0$ , where $\begin{array} { r } { E ( v ) = \frac { 1 } { 2 } \sum _ { k } a _ { k } | \langle v _ { k } , v \rangle | } \end{array}$ Hence $\chi ( s ) q \cdot v \geq 0$ . Approaching each coordinate axis through orthants of opposite parity forces $q = 0$ . Thus $h _ { \infty }$ is even, and the opposite labels of $O _ { s }$ and $O _ { - s }$ force $h _ { \infty } = 0$ on every open orthant. Continuity gives $h _ { \infty } \equiv 0$ on $\mathbb { R } ^ { 3 }$

Outside the finitely many ridge hyperplanes $\{ \langle v _ { k } , v \rangle = 0 \}$ , the active set along rv is eventually fixed. Since $h _ { \infty } ( v ) = 0$ cancels the coeficient of $r _ { \mathrm { : } }$

$$
G ( v ) : = \operatorname* { l i m } _ { r \to \infty } h ( r v ) = c + \sum _ { \langle v _ { k } , v \rangle > 0 } a _ { k } b _ { k } .
$$

For all suficiently large $r , h ( r v ) = G ( v )$ . Choose $v ^ { 0 } = ( 0 , v _ { 2 } ^ { 0 } , v _ { 3 } ^ { 0 } )$ with $v _ { 2 } ^ { 0 } v _ { 3 } ^ { 0 } \neq 0$ , avoiding ridge hyperplanes that do not contain $v _ { 1 } = 0$ . For suficiently small $\varepsilon > 0$ , the active sets at $\boldsymbol { v } ^ { 0 } \pm \varepsilon \boldsymbol { e } _ { 1 }$ difer only for normals parallel to $e _ { 1 }$ . Hence the jump

$$
D : = G ( v ^ { 0 } + \varepsilon e _ { 1 } ) - G ( v ^ { 0 } - \varepsilon e _ { 1 } ) \qquad \\  = \sum _ { \stackrel { k \in [ m ] , v _ { k } \in \mathbb { R } e _ { 1 } } { \langle v _ { k } , e _ { 1 } \rangle > 0 } } a _ { k } b _ { k } - \sum _ { \stackrel { k \in [ m ] , v _ { k } \in \mathbb { R } e _ { 1 } } { \langle v _ { k } , e _ { 1 } \rangle < 0 } } a _ { k } b _ { k }
$$

is independent of the signs of $v _ { 2 } ^ { 0 }$ and $v _ { 3 } ^ { 0 } . { \mathrm { ~ H ~ } } v _ { 2 } ^ { 0 } v _ { 3 } ^ { 0 } > 0$ , the labels force $D > 0 ;$ if $v _ { 2 } ^ { 0 } v _ { 3 } ^ { 0 } < 0$ , they force $D < 0$ . This contradiction proves both assertions. Restricting a putative separator in $\mathbb { R } ^ { d }$ to the first three coordinates proves the final statement. □

## C.3 Cone geometry of afine separation

Proof of Proposition $\it 2 . 4$ . For (ii), pointedness implies $0 \notin C _ { K } :$ a convex combination of unit vectors in K equal to zero would give a nonzero vector in $K \cap ( - K )$ . Compactness of $C _ { K }$ gives $q = P _ { C _ { K } } ( 0 ) \neq 0$ . The projection inequality $\langle q , z \rangle \geq \| q \| ^ { 2 }$ for $z \in C _ { K }$ yields (6). Testing any unit u at $q \in C _ { K }$ proves optimality; equality in Cauchy–Schwarz gives uniqueness.

Write $K ^ { \ast } = \{ v \in \mathbb { R } ^ { d } : \langle v , z \rangle \geq 0$ for all $z \in K \}$ for the dual cone. For (i), pointedness and (ii) give $u _ { K } \in$ int $K ^ { * }$ . Choose a compact spherical cap $B \subset$ int $K ^ { * }$ of positive surface measure and put

$$
\mu = \operatorname* { i n f } _ { v \in B } \operatorname* { i n f } _ { z \in K \cap S ^ { d - 1 } } \langle v , z \rangle > 0 .
$$

The Gaussian weight event $g = r v$ with $v \in B$ and $1 \leq r \leq 2$ has positive probability. On this event, writing $\delta = \mathrm { d i s t } ( E ^ { + } , E ^ { - } )$ 2

$$
\langle g , x ^ { + } - x ^ { - } \rangle \geq \mu \| x ^ { + } - x ^ { - } \| \geq \mu \delta
$$

for every cross pair. The finite support values $m ^ { + } ( g ) = \operatorname* { i n f } _ { E ^ { + } } \langle g , x \rangle$ and $M ^ { - } ( g ) = \operatorname* { s u p } _ { E ^ { - } } \langle g , y \rangle$ satisfy $m ^ { + } ( g ) - M ^ { - } ( g ) \geq \mu \delta$ . Fixed anchors $x _ { 0 } ^ { \pm } \in E ^ { \pm }$ and $\| g \| \le 2$ bound both values uniformly. For $t = \mu \delta / 4$ , the admissible biases

$$
[ t - m ^ { + } ( g ) , - t - M ^ { - } ( g ) ]
$$

form an interval of length at least $\mu \delta / 2$ , contained in a fixed $[ - \lambda , \lambda ]$ for suficiently large λ. Thus the conditional bias probability is uniformly positive. Conversely, separation with margin parameter $t > 0$ implies $\langle g , z \rangle \geq 0$ on $K$ , hence $g \in K ^ { * }$ . If K is not pointed, it contains a nonzero line, so $K ^ { * }$ lies in a proper orthogonal hyperplane of Gaussian measure zero, regardless of the bias. □

## C.4 Necessity of pointedness

Proof of Example 3.1. The opposite diference sets $\mathbb { R } \times [ 1 / 2 , 5 / 2 ]$ and $\mathbb { R } \times [ - 5 / 2 , - 1 / 2 ]$ give the stated gaps and cones. Inductively, almost surely, each $F _ { \ell }$ is either constant or has the form

$$
F _ { \ell } ( t , s ) = t u + s v + c , \qquad u \neq 0 ,
$$

on $\{ ( t , s ) : t \geq T _ { \ell } , | s | \leq 2 \}$ for some finite $T _ { \ell }$ . The assertion holds for $F _ { 0 } ( t , s ) = t e _ { 1 } + s e _ { 2 }$ , and constant tails persist. For a nonconstant tail of $F _ { \ell - 1 }$ , condition on the first $\ell - 1$ layers. Under this conditioning,

$$
W _ { \ell } u \sim N ( 0 , \| u \| ^ { 2 } I _ { n _ { \ell } } ) ,
$$

so all coordinates are nonzero almost surely. Set

$$
T _ { \ell } = \operatorname* { m a x } \Biggl \{ T _ { \ell - 1 } , 1 + \operatorname* { m a x } _ { a \in [ n _ { \ell } ] } \frac { 2 | ( W _ { \ell } v ) _ { a } | + | ( W _ { \ell } c + b _ { \ell } ) _ { a } | } { | ( W _ { \ell } u ) _ { a } | } \Biggr \} .
$$

This threshold is finite almost surely. For all $t \geq T _ { \ell }$ and $| s | \le 2$ , each preactivation has the sign of the corresponding coordinate of $W _ { \ell } u$ . The next layer is therefore afine there, with coeficient of t given by

$$
\sqrt { \frac { 2 } { n _ { \ell } } } \mathrm { R e L U } ( W _ { \ell } u ) .
$$

This coeficient vanishes only if all preactivation slopes are negative, in which case the output is identically zero there. Positive normalization factors preserve these properties, and the finite architecture completes the induction on a probability-one event.

Thus, for all suficiently large t,

$$
\begin{array} { r } { F _ { q } ( t , 0 ) = \frac { 1 } { 2 } F _ { q } ( t , 1 ) + \frac { 1 } { 2 } F _ { q } ( t , - 1 ) . } \end{array}
$$

Since $( t , \pm 1 ) \in S ^ { + }$ and $( t , 0 ) \in S ^ { - }$ , this proves (15) and rules out strict afine separation. Finally, $h ( x ) = | x _ { 2 } | - 3 / 4$ has the asserted signs and output gaps. □

## C.5 Cone geometry and Gaussian width

For (16), every $z \in K _ { \alpha , r }$ has $z _ { 1 } \geq 0$ . The anchors are therefore $\pm 3 e _ { 1 } / 2$ for $E _ { 1 } ^ { \pm }$ and $\mp e _ { 1 } / 2$ for $E _ { 2 } ^ { \pm }$ giving $R = 3 / 2$ . The intervals joining each class’s anchors overlap on $\{ t e _ { 1 } : | t | \le 1 / 2 \}$ . Thus their convex hulls intersect, which rules out strict afine separation. The opposite-set diferences are

$$
\begin{array} { c } { { E _ { 1 } ^ { + } - E _ { 1 } ^ { - } = 3 e _ { 1 } + K _ { \alpha , r } , \qquad E _ { 1 } ^ { + } - E _ { 2 } ^ { - } = E _ { 2 } ^ { + } - E _ { 1 } ^ { - } = e _ { 1 } + K _ { \alpha , r } , } } \\ { { E _ { 2 } ^ { + } - E _ { 2 } ^ { - } = \{ - e _ { 1 } \} . } } \end{array}
$$

Their distances are respectively 3, 1, 1, 1, attained by the anchors, so $\delta _ { 0 } = 1$ . For every $a > 0$ , the inclusion $a e _ { 1 } + K _ { \alpha , r } \subset K _ { \alpha , r }$ and the limit $t ^ { - 1 } ( a e _ { 1 } + t z )  z$ for $z \in K _ { \alpha , \mathfrak { r } }$ give cl cone $\begin{array} { r l } { \mathrm { \Omega } } & { { } : ( a e _ { 1 } + K _ { \alpha , r } ) = } \end{array}$ $K _ { \alpha , r } .$ . This proves the diference-cone identities in (17). Every direction in the spherical section of $K _ { \alpha , r }$ has first coordinate at least cos $\alpha _ { \mathrm { { ; } } }$ , while two boundary directions with opposite transverse components have midpoint cos α $e _ { 1 }$ . Thus $u _ { K } = e _ { 1 }$ and $\kappa ( K _ { \alpha , r } ) = \cos { \alpha }$ . The ray $K _ { 2 2 }$ has depth 1; its spherical section has Gaussian width $\mathbb { E } _ { g _ { 1 } } ( - g _ { 1 } ) = 0$ for $g _ { 1 } \sim N ( 0 , 1 )$ . Consequently, $\kappa _ { \operatorname* { m i n } } = \cos \alpha$ and the largest squared Gaussian width is exactly that of $K _ { \alpha , r } \cap S ^ { r - 1 }$

Write $( g _ { 1 } , g _ { \perp } ) \sim N ( 0 , I _ { r } )$ . Parametrizing the spherical section by (cos θ, sin θ v), with $0 \leq \theta \leq \alpha$ and $v \in S ^ { r - 2 }$ , gives

$$
w ( K _ { \alpha , r } \cap S ^ { r - 1 } ) = \mathbb { E } _ { g _ { 1 } , g _ { \perp } } \operatorname* { m a x } _ { 0 \leq \theta \leq \alpha } ( g _ { 1 } \cos \theta + \| g _ { \perp } \| \sin \theta ) .
$$

Taking $\theta = \alpha$ and using $\mathbb { E } _ { g _ { 1 } } g _ { 1 } = 0$ gives the lower bound. For the upper bound, use

$$
g _ { 1 } \cos \theta + \| g _ { \perp } \| \sin \theta \leq g _ { 1 } + ( - g _ { 1 } ) _ { + } ( 1 - \cos \alpha ) + \| g _ { \perp } \| \sin \alpha .
$$

Since $\mathbb { E } _ { g _ { 1 } } ( - g _ { 1 } ) _ { + } = 1 / \sqrt { 2 \pi }$ and $g _ { \perp }$ has the law of $G _ { r - 1 }$

$$
w ( K _ { \alpha , r } \cap S ^ { r - 1 } ) \leq \sin \alpha \mathbb { E } _ { G _ { r - 1 } } \| G _ { r - 1 } \| + \frac { 1 - \cos \alpha } { \sqrt { 2 \pi } } .
$$

By $1 ~ - ~ \cos \alpha ~ \leq$ sin α and $\mathbb { E } _ { G _ { r - 1 } } \| G _ { r - 1 } \| \ \geq \ \mathbb { E } _ { g _ { 1 } } | g _ { 1 } | \ = \ \sqrt { 2 / \pi }$ , the last term is at most $\frac { 1 } { 2 }$ sin α $\mathbb { E } _ { G _ { r - 1 } } \Vert G _ { r - 1 } \Vert$ , proving (18). Cauchy–Schwarz gives the upper bound below; bounding the norm from below by the sum of absolute coordinates divided by $\sqrt { r - 1 }$ gives the lower bound:

$$
\begin{array} { r } { \sqrt { 2 ( r - 1 ) / \pi } \leq \mathbb { E } _ { G _ { r - 1 } } \Vert G _ { r - 1 } \Vert \leq \sqrt { r - 1 } , } \end{array}
$$

which yields the stated width bounds for $\alpha = \pi / 4$

## C.6 Random kernels and collisions between classes

Proof of Example 3.2. Let $q = \operatorname* { m i n } \{ n , \widehat { n } \}$ . The probability bound is trivial for $q \geq r ,$ so assume $q < r$ . We show that intersection of a suitable random kernel with int $K _ { \alpha , r }$ produces a cross-class pair with identical features.

The first layer is narrower: $n \leq \widehat { n }$ . Here $n = q < r$ . If ker $W \cap$ int $K _ { \alpha , r } \neq \emptyset$ , choose v in this intersection. For suficiently large t, $\boldsymbol { v } - 3 e _ { 1 } / ( 2 t )$ remains in the cone, so $x = t v \in E _ { 1 } ^ { + }$ and $- x \in E _ { 1 } ^ { - }$ Since $W x = W ( - x ) = 0$ , their first-layer features and all subsequent outputs coincide. This also proves the obstruction for first-layer separation alone, since the condition for $j = 1$ includes this pair.

The second layer is narrower: $\widehat { n } < n$ . Set $A = { \widehat { W } } W$ . Almost surely, row W has dimension min $\{ n , r \} > \widehat { n }$ . Conditional on such a $W$ , the rows of A are independent nondegenerate Gaussian vectors in this space. Thus A has full row rank, and its row space contains none of the nonzero rows $w _ { a } ^ { \top }$ of $W$ , almost surely. Averaging over $W$ gives

$$
\ker { A } \ \mathcal { Z } \ w _ { a } ^ { \perp } \quad ( a \in [ n ] ) \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

Fix weights satisfying these conclusions and arbitrary finite biases $b , { \widehat { b } } .$ If ker A ∩ int $K _ { \alpha , r } \neq \emptyset$ , this relatively open set cannot lie in finitely many proper hyperplanes. Choose v in it with $\langle w _ { a } , v \rangle \neq 0$ for every $^ { a , }$ and let $s _ { a } = \mathrm { s i g n } \langle w _ { a } , v \rangle$ . Full row rank provides $x _ { 0 }$ satisfying $A x _ { 0 } = - \hat { W } \deg ( s ) b$ Because v is interior and all biases are finite, for suficiently large t, the vector $v + ( x _ { 0 } - 3 e _ { 1 } / 2 ) / t$ lies in $K _ { \alpha , r } ,$ and the point $x = t v + x _ { 0 }$ satisfies

$$
x \in E _ { 1 } ^ { + } , \quad - x \in E _ { 1 } ^ { - } , \quad \mathrm { s i g n } ( W x ) = s , \quad | ( W x ) _ { a } | > | b _ { a } | \quad ( a \in [ n ] ) .
$$

The corresponding preactivations have opposite signs in each coordinate, giving

$$
\mathrm { R e L U } ( W x + b ) - \mathrm { R e L U } ( - W x + b ) = W x + \mathrm { d i a g } ( s ) b .
$$

Since $A v = 0$ and $A x _ { 0 } = - \widehat { W } \mathrm { d i a g } ( s ) b .$

$$
\widehat W [ \Phi ( x ) - \Phi ( - x ) ] = \sqrt { 2 / n } \left[ A x + \widehat W \mathrm { d i a g } ( s ) b \right] = 0 .
$$

The common second-layer bias cancels, so $F ( x ) = F ( - x )$ for all finite biases.

Intersection probability and the width lower bound. Take $L =$ row $W$ in the first case and $L = \operatorname { r o w } A$ in the second. The corresponding matrix has rank q and a right-rotation-invariant distribution, so $L$ is a uniform q-dimensional subspace. For any subspace, maximizing the axial projection over unit vectors gives

$$
L ^ { \perp } \cap \operatorname* { i n t } K _ { \alpha , r } \neq \emptyset \quad \Longleftrightarrow \quad \| P _ { L ^ { \perp } } e _ { 1 } \| > \cos \alpha \quad \Longleftrightarrow \quad \| P _ { L } e _ { 1 } \| ^ { 2 } < \sin ^ { 2 } \alpha .
$$

Successful separation therefore requires $\| P _ { L } e _ { 1 } \| ^ { 2 } \ge \sin ^ { 2 } \alpha$ . Here the law of L is induced by W in the first case and by the joint law of $( W , { \hat { W } } )$ in the second; W is not fixed in the latter. Rotational invariance and tr $P _ { L } = q$ give $\mathbb { E } _ { L } P _ { L } = ( q / r ) I _ { r }$ . Markov’s inequality yields

$$
{ \mathbb P } \{ \mathrm { s u c c e s s f u l ~ s e p a r a t i o n } \} \le { \mathbb P } \{ \| P _ { L } e _ { 1 } \| ^ { 2 } \ge \sin ^ { 2 } \alpha \} \le \frac { q } { r \sin ^ { 2 } \alpha } .
$$

Since $q = \operatorname* { m i n } \{ n , \widehat { n } \}$ and probabilities are at most 1,

$$
\mathbb { P } \{ F ( S ^ { + } ) { \mathrm { ~ a n d ~ } } F ( S ^ { - } ) { \mathrm { ~ a r e ~ s t r i c t l y ~ l i n e a r l y ~ s e p a r a b l e } } \} \leq \operatorname* { m i n } \left\{ 1 , \frac { \operatorname* { m i n } \{ n , \widehat { n } \} } { r \sin ^ { 2 } \alpha } \right\} .
$$

By (18) and $( \mathbb { E } _ { G _ { r - 1 } } \Vert G _ { r - 1 } \Vert ) ^ { 2 } \leq r - 1 < r ,$

$$
\begin{array} { r } { w ( K _ { \alpha , r } \cap S ^ { r - 1 } ) ^ { 2 } \leq \frac { 9 } { 4 } ( r - 1 ) \sin ^ { 2 } \alpha \leq \frac { 9 } { 4 } r \sin ^ { 2 } \alpha , } \end{array}
$$

which gives (19). For first-layer separation alone, take $L = \operatorname { r o w } W$ and $q = n ;$ when $n \geq r ,$ the lower bound holds automatically. □

## Funding

This work was supported in part by the National Natural Science Foundation of China under Grant 12271140, and the Fundamental Research Funds for the Central Universities under Grant B260203007.

## References

[1] G. Alain and Y. Bengio, Understanding intermediate layers using linear classifier probes. In International Conference on Learning Representations, Workshop Track, 2017.

[2] R. Arora, A. Basu, P. Mianjy, and A. Mukherjee, Understanding deep neural networks with rectified linear units. In International Conference on Learning Representations, 2018.

[3] A. S. Bandeira, D. G. Mixon, and B. Recht, Compressive classification and the rare eclipse problem. In Compressed sensing and its applications, pp. 197–220, Appl. Numer. Harmon. Anal., Birkhäuser, Cham, 2017. DOI 10.1007/978-3-319-69802-1\_6

[4] V. Chandrasekaran, B. Recht, P. A. Parrilo, and A. S. Willsky, The convex geometry of linear inverse problems. Found. Comput. Math. 12 (2012), no. 6, 805–849. DOI 10.1007/s10208- 012-9135-7

[5] C. Cortes and V. Vapnik, Support-vector networks. Mach. Learn. 20 (1995), no. 3, 273–297. DOI 10.1007/BF00994018

[6] M. A. Davenport, P. T. Boufounos, M. B. Wakin, and R. G. Baraniuk, Signal processing with compressive measurements. IEEE J. Sel. Top. Signal Process. 4 (2010), no. 2, 445–460. DOI 10.1109/JSTSP.2009.2039178

[7] S. Dirksen, M. Genzel, L. Jacques, and A. Stollenwerk, The separation capacity of random neural networks. J. Mach. Learn. Res. 23 (2022), no. 309, 1–47.

[8] P. Ghosal, S. Mahankali, and Y. Sun, Randomly initialized one-layer neural networks make data linearly separable. 2023, arXiv:2205.11716v2. DOI 10.48550/arXiv.2205.11716

[9] Y. Gordon, On Milman’s inequality and random subspaces which escape through a mesh in R<sup>n</sup>. In Geometric aspects of functional analysis: Israel seminar 1986–87, pp. 84–106, Lecture Notes in Math. 1317, Springer, Berlin, 1988. DOI 10.1007/BFb0081737

[10] H. He and W. J. Su, A law of data separation in deep learning. Proc. Natl. Acad. Sci. USA 120 (2023), no. 36, e2221704120. DOI 10.1073/pnas.2221704120

[11] K. He, X. Zhang, S. Ren, and J. Sun, Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1026–1034, IEEE, 2015. DOI 10.1109/ICCV.2015.123

[12] D. Hsu, C. H. Sanford, R. A. Servedio, and E.-V. Vlatakis-Gkaragkounis, On the approximation power of two-layer networks of random ReLUs. In Proceedings of the 34th Conference on Learning Theory, pp. 2423–2461, Proc. Mach. Learn. Res. 134, 2021.

[13] M. Hu, J. H. Chion, P. N. Suganthan, and R. K. Katuwal, Ensemble deep random vector functional link neural network for regression. IEEE Trans. Syst. Man Cybern. Syst. 53 (2023), no. 5, 2604–2615. DOI 10.1109/TSMC.2022.3213628

[14] G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew, Extreme learning machine: theory and applications. Neurocomputing 70 (2006), no. 1–3, 489–501. DOI 10.1016/j.neucom.2005.12.126

[15] S. Janson, Gaussian Hilbert spaces. Cambridge Tracts in Math. 129, Cambridge University Press, Cambridge, 1997. DOI 10.1017/CBO9780511526169

[16] H. Jeong, X. Li, Y. Plan, and Ö. Yılmaz, Sub-Gaussian matrices on sets: optimal tail dependence and applications. Comm. Pure Appl. Math. 75 (2022), no. 8, 1713–1754. DOI 10.1002/cpa. 22024

[17] M. Ledoux and M. Talagrand, Probability in Banach spaces: isoperimetry and processes. Springer, Berlin, Heidelberg, 1991. DOI 10.1007/978-3-642-20212-4

[18] Y. Lei, P. Wang, Y. Ying, and D.-X. Zhou, Optimization and generalization of gradient descent for shallow ReLU networks with minimal width. J. Mach. Learn. Res. 27 (2026), no. 34, 1–35.

[19] J. Lu, Z. Shen, H. Yang, and S. Zhang, Deep network approximation for smooth functions. SIAM J. Math. Anal. 53 (2021), no. 5, 5465–5506. DOI 10.1137/20M134695X

[20] U. Markowska-Kaczmar and M. Kosturek, Extreme learning machine versus classical feedforward network: comparison from the usability perspective. Neural Comput. Appl. 33 (2021), no. 22, 15121–15144. DOI 10.1007/s00521-021-06402-y

[21] W. Masarczyk, M. Ostaszewski, E. Imani, R. Pascanu, P. Miłoś, and T. Trzciński, The tunnel efect: building data representations in deep neural networks. In Advances in Neural Information Processing Systems 36, pp. 76772–76805, 2023. DOI 10.52202/075280-3355

[22] D. Needell, A. A. Nelson, R. Saab, P. Salanevich, and O. Schavemaker, Random vector functional link networks for function approximation on manifolds. Front. Appl. Math. Stat. 10 (2024), article no. 1284706. DOI 10.3389/fams.2024.1284706

[23] R. O’Donnell, R. A. Servedio, and L.-Y. Tan, Fooling Gaussian PTFs via local hyperconcentration. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing (STOC ’20), pp. 1170–1183, Association for Computing Machinery, New York, 2020. DOI 10.1145/3357713.3384281

[24] A. Rahimi and B. Recht, Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems 20, pp. 1177–1184, 2007.

[25] R. T. Rockafellar, Convex analysis. Princeton Math. Ser. 28, Princeton University Press, Princeton, NJ, 1970. DOI 10.1515/9781400873173

[26] A. Rudi and L. Rosasco, Generalization properties of learning with random features. In Advances in Neural Information Processing Systems 30, pp. 3215–3225, 2017.

[27] P. Salanevich and O. Schavemaker, Eficient uniform approximation using random vector functional link networks. In 2023 International Conference on Sampling Theory and Applications (SampTA), pp. 1–5, IEEE, 2023. DOI 10.1109/SampTA59647.2023.10301394 Corrected version (2025): arXiv:2306.17501v2.

[28] B. Schölkopf and A. J. Smola, Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT Press, Cambridge, MA, 2002. DOI 10.7551/mitpress/4175. 001.0001

[29] Z. Shen, H. Yang, and S. Zhang, Optimal approximation rate of ReLU networks in terms of width and depth. J. Math. Pures Appl. 157 (2022), 101–135. DOI 10.1016/j.matpur.2021.07.009

[30] R. Vershynin, High-dimensional probability: an introduction with applications in data science. Second ed., Cambridge Ser. Stat. Probab. Math. 58, Cambridge University Press, Cambridge, 2026. DOI 10.1017/9781009490672

[31] A. S. Xu, C. Yaras, P. Wang, and Q. Qu, Linearly separable features in shallow nonlinear networks: width scales polynomially with intrinsic data dimension. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, pp. 172–180, Proc. Mach. Learn. Res. 300, 2026.

[32] Z. Zhang, L. Shi, and D.-X. Zhou, Classification with deep neural networks and logistic loss. J. Mach. Learn. Res. 25 (2024), no. 125, 1–117.