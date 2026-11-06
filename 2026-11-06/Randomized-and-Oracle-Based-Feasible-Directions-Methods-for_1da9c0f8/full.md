# Randomized and Oracle-Based Feasible Directions Methods for Nonsmooth Nonconvex Constrained Optimization

Dan Greenstein<sup>1</sup> Nadav Hallak<sup>1∗</sup>

<sup>1</sup> Technion Israel Institute of Technology, Faculty of Data and Decision Sciences, sdngreen@campus.technion.ac.il, ndvhllk@technion.ac.il

## Abstract

Minimizing nonsmooth nonconvex objectives over general convex sets using no derivatives of the objective, together with structural access to the constraint set, poses a fundamental geometric dificulty: at boundary points, the feasible direction cone may fail to be closed, preventing the construction of finite sets of search directions that capture all feasible descent directions. This paper develops a feasible-directions framework for problems structured as the diference between a continuously diferentiable function and a convex function over a closed convex set, and accesses the objective without knowledge of its decomposition, through exact line searches along feasible segments. To address the geometric obstruction, we introduce η-feasible directions, which remain uniformly separated from the boundary of the feasible direction cone, thereby enabling finite-direction search even in the presence of nonlinear constraints. We present two algorithmic realizations of this framework. The first assumes access to an η-direction oracle and extends feasibledirections methods from linear to general convex constraints. The second replaces the oracle by a noise-augmented randomization scheme that constructs spanning directions from the afine-hull inner normal and almost surely avoids nonsmooth boundary points when initialized outside that set. Under curvature bounds and Lipschitz continuity of the smooth component’s gradient, we jointly control the spanning coeficients and feasible step lengths of the oracle construction. This yields finite-time approximate d-stationarity along every unit feasible direction admitting a prescribed positive step length. For both methods, we also establish a rate in the proximal directional minimization (PDM) measure, assuming uniformly bounded direction-set sizes. Randomized exploration ensures that, almost surely, every accumulation point is d-stationary.

## 1 Introduction

## 1.1 Problem Formulation

This paper studies the nonsmooth nonconvex constrained problem

$$
\operatorname* { m i n } _ { x \in \mathcal { C } } h ( x ) : = f ( x ) - g ( x ) ,\tag{P}
$$

where $f : \mathbb { R } ^ { n }  \mathbb { R }$ is continuously diferentiable, $g : \mathbb { R } ^ { n }  \mathbb { R }$ is convex and possibly nondiferentiable, the feasible set ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ is closed and convex, and h is lower bounded over C. We further assume that $| { \mathcal { C } } | > 1$ and $n \geq 2$ to exclude trivial or pathological cases. These conditions imply that the dimension of C is at least one, i.e., dim $\mathcal { C } \geq 1$

Problem (P) captures a broad class of nonlinear constrained models, including the prototypical problems below.

Example 1.1 (Diference-of-convex programs). When f is convex, problem (P) reduces to a diference-of-convex (DC) program. DC formulations arise in a wide range of applications; see, e.g., [2] studying DC programs machine learning tasks.

Example 1.2 (Linear constraints). When the feasible set C is a polytope, problem (P) reduces to the model studied in [5, 6] – see application examples therein.

Example 1.3 (Concave regularization). Problem (P) includes sparsity-promoting regularizers $\lambda \textstyle \sum _ { i } \varphi ( | x _ { i } | )$ with $\varphi$ concave and non-decreasing on $[ 0 , \infty )$ and $\varphi ^ { \prime } ( 0 ^ { + } ) < \infty$ , such as the log-sum penalty [8] and SCAD [11]. Adding variables y with the linear constraints $y _ { i } \geq \pm x _ { i }$ and replacing $\varphi ( | x _ { i } | )$ by $\varphi ( y _ { i } )$ gives an equivalent instance of (P); the condition $\varphi ^ { \prime } ( 0 ^ { + } ) < \infty$ lets − $^ - \varphi$ extend to a finite convex function on R.

In nonsmooth nonconvex optimization, a critical point may still admit a feasible direction of first-order descent. Directional stationarity provides a stronger target: it requires a nonnegative directional derivative along every feasible direction and thus rules out all feasible first-order improvements. Our goal is to obtain guarantees of this kind while maintaining feasibility, without requiring objective derivatives or access to the decomposition into f and g. We study methods that access the objective through exact regularized line searches and exploit the geometry of the constraint set.

The central question is whether finitely many feasible line searches can provide directional optimality guarantees beyond the directions actually searched. For the objective structure considered here, non-descent along the search directions extends to their positive span. The geometry of this span therefore determines the scope of the resulting optimality guarantees.

When C is polyhedral, the feasible-direction cone admits a finite positive spanning set. This property underlies the finite-time guarantees in the proximal directional minimization (PDM) measure established in [6], without requiring objective derivatives. For general closed convex sets, however, the feasible-direction cone may be nonclosed at boundary points and therefore admit no finite positive spanning set. This obstruction arises even for basic constraint sets such as Euclidean balls and ellipsoids: finitely many feasible search directions cannot cover all feasible directions through their positive span.

To retain finite-direction guarantees under general convex constraints, we introduce η-feasible directions, which remain angularly separated from the boundary of the feasibledirection cone. This restriction allows finitely many feasible search directions to cover the η-feasible directions through their positive span. It provides the geometric basis for finitetime PDM guarantees with full directional coverage in the relative interior and a controlled restriction at nonlinear boundary points.

We implement this idea through two feasible-directions methods that difer in how they construct the search directions: one uses a constraint-set oracle, while the other uses a randomized construction based on the local boundary geometry. At each iteration, the method performs exact regularized line searches along the resulting directions and selects the candidate with the greatest objective decrease. The searches are restricted to feasible segments, so all iterates remain in C. Neither objective derivatives nor access to the decomposition into f and g is required (Remark 4.2).

Directional coverage alone does not turn small objective decrease into a quantitative bound on directional derivatives: the spanning coeficients and feasible step lengths must also be controlled. Under curvature bounds and Lipschitz continuity of $\nabla f$ , our oracle construction provides this control, yielding finite-time approximate d-stationarity along every unit feasible direction admitting a prescribed positive step length (Theorem 6.3 and Corollary 6.2).

To obtain full d-stationarity despite the restricted directional coverage at boundary points, we supplement these searches with the randomized exploration step of Greenstein and Hallak [16]. At each iterate x, we sample a direction uniformly from the unit sphere in af $( { \mathcal { C } } - x )$

and search along its feasible segment. This exploration ensures that, almost surely, every accumulation point of either method is d-stationary (Theorem 6.2).

Outline. Following the preliminaries in Section 2, the main development is concentrated in Sections 3–6: Section 3 introduces η-feasible directions; Sections 4 and 5 present the oraclebased and noised methods, respectively; and Section 6 establishes PDM rates, almost-sure d-stationarity of accumulation points, and finite-time guarantees under curvature assumptions. Section 7 supplies the supporting spanning and boundary-avoidance results. The extensive appendices collect auxiliary proofs, detailed convergence arguments, and measure-theoretic foundations needed for completeness.

## 1.2 Contribution

Geometric framework. In Section 3 we develop the notion of η-feasible directions (η-FDs) and establish its properties. At smooth boundary points of nonlinear constraint sets, the η-FD cone is contained in the positive span of finitely many feasible directions, yielding a PDM guarantee over that cone.

Feasible-directions framework. We propose two methods utilizing η-FDs: (i) an oraclebased method that extends the linearly constrained feasible-directions method of Beck and Hallak [6] to general convex constraint sets (Section 4), and (ii) a randomized method that generates feasible spanning directions from the inward unit normal in the afine hull and, for α > 0, avoids nonsmooth boundary points almost surely when initialized outside that set (Section 5). With randomized exploration adapted from [16], almost surely every accumulation point of either method is d-stationary (Theorem 6.2).

Approximate d-stationarity under curvature. Under the stated curvature bounds on a C<sup>2</sup> constraint boundary, we use a basis adapted to the boundary geometry and inward tilting to jointly control the spanning coeficients and feasible step lengths. With Lipschitz continuity of ∇f, this yields a finite-time guarantee of approximate d-stationarity along every unit feasible direction admitting a prescribed positive step length (Theorem 6.3 and Corollary 6.2).

Finite-time PDM guarantees. For both methods, we establish an O(1/K) rate in the proximal directional minimization (PDM) measure of [6], assuming uniformly bounded direction-set sizes (Theorem 6.1). For the oracle method, vanishing PDM certifies d-stationarity at relative-interior points and non-descent along all η-feasible directions at boundary points. For suficiently small noise, the noised method retains the interior guarantee and covers a slightly smaller cone at smooth boundary points.

Geometric analysis. We also establish results on the local geometry of feasible directions for closed convex sets, including characterizations of η-FDs at relative-interior and smooth boundary points (Corollary 3.1). Geometry for directional spanning and trajectory behavior within general convex sets is developed in Section 7.

## 1.3 Background

We review related approaches to constrained optimization, emphasizing their information requirements, treatment of feasibility, and stationarity guarantees.

Methods for linear constraints. The closest works to ours are [5, 6], which study the same objective structure under linear constraints. In that setting, the feasible direction cone is closed, explicitly computable, and finitely generated. These works establish convergence to stationary points and introduce the PDM optimality measure. The same spanning construction is classical in bound- and linearly constrained pattern search and generating-set search, which poll a finite set of generators of the tangent cone [19, 20, 22, 23].

Over a general closed convex set, the feasible direction cone need not have a finite set of generators, and sequences of feasible directions may converge to directions that are no longer feasible; Algorithm 1 instead uses a finite set of feasible directions whose positive span contains the η-feasible directions. Gratton et al. [15] analyze direct search with suficient decrease for smooth bound- and linearly constrained problems; we compare their complexity bound with ours after Corollary 6.2.

Diference-of-convex programming. When f is convex, problem (P) becomes a diferenceof-convex (DC) program; see, e.g., [35]. Most DC algorithms guarantee convergence only to DC-critical points; see, e.g., [1, 28]. The optimality hierarchy for nonsmooth DC programs was developed in [27], where directional stationarity was shown to be a stronger condition. Under additional structure, directional stationarity can be attained by some DC methods; see, e.g., [27].

Our setting difers in several aspects: we allow f to be nonconvex, impose general convex constraints, and access h through exact line searches, with no derivatives and without requiring access to the decomposition into f and g. While DC methods benefit from firstorder information, our focus is on procedures that use no derivatives and maintain feasibility. Projection-free Frank–Wolfe methods likewise produce feasible iterates over general convex sets [13, 18], but they require first-order information and a linear minimization oracle over C.

Derivative-free constrained optimization. Conceptually, our approach can be afiliated with the Derivative-free methods (DFO) branch – see [9, 21] for general surveys. DFOs for constrained optimization include direct-search approaches such as MADS [4], which achieve Clarke-type stationarity under general constraints. These methods typically rely on asymptotically dense sampling of directions. For nonlinear inequality constraints, Lucidi et al. [25] keep all iterates feasible by sampling the objective along search arcs obtained by projecting rays onto the feasible set, with directions that approximately generate the cone of first-order feasible variations. Our methods use finite-direction constructions based on η-feasible directions for a finite-time PDM certificate; almost-sure subsequence d-stationarity is obtained from a randomized exploration component developed by the authors in [16].

Penalty and barrier methods. Many derivative-free approaches handle constraints via penalty formulations, allowing infeasible iterates [12, 24]; the interior-point method of [7] keeps iterates strictly feasible with respect to nonrelaxable inequality constraints through a log-barrier, while penalizing equality constraints. In contrast, our framework maintains feasibility with respect to the whole set C at every iteration, including iterates on its boundary, leading to a diferent algorithmic design and analysis.

## 1.4 Notation

We use the common notation of convex analysis and continuous optimization literature. Depending on the context, we use the index set $[ q ] : = \{ 1 , 2 , \dots , q \}$ for a natural number q, and ${ \Delta _ { q } } : = \{ x \in \mathbb { R } ^ { q } : \textstyle \sum { x _ { i } } = 1 , x \ge 0 \}$ is the q-dimensional simplex. For a nonempty set X , the set of all finite subsets of X is denoted by $\mathcal { P } _ { f i n } \left( \mathcal { X } \right) = \{ C \subseteq \mathcal { X } \mid | C | < \infty \}$ . For a convex body C with a $C ^ { 2 }$ boundary and principal curvatures in $\left[ \kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } } \right]$ we write $r _ { \mathrm { i n } } : = 1 / \kappa _ { \mathrm { m a x } }$ and $r _ { \mathrm { o u t } } : = 1 / \kappa _ { \mathrm { m i n } }$ for the inner and outer rolling-ball radii, and $\rho ( x ) : = \mathrm { d i s t } ( x , \mathrm { r b } \left( \mathcal { C } \right) )$ for the depth of $x \in { \mathcal { C } }$ . For $x \in { \mathcal { C } }$ write $\Pi ( x ) : = { \arg \operatorname* { m i n } } _ { y \in { \mathrm { ~ r b ~ } } ( { \mathcal { C } } ) } \left\| y - x \right\|$ . Note that rb (C) is not convex, so Π(x) need not be a singleton; at the center of a ball it is the entire sphere.

## 2 Preliminaries

In this section we recall classical elements to be used throughout the paper, starting with the notions of positive span and cone of feasible directions.

Definition $\mathbf { 2 . 1 \ \Omega \left( [ 5 ] \right. }$ , Definition 2.1). The positive span of a finite set of vectors $S \ =$ $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { k } \} \subseteq \mathbb { R } ^ { n }$ , where $k \in \mathbb N$ , is denoted by pos (S) and is defined as the convex cone

$$
\mathrm { p o s } \left( S \right) : = \left\{ \sum _ { i = 1 } ^ { k } \lambda _ { i } v _ { i } \mid \lambda _ { i } \geq 0 \ \forall i \in \{ 1 , \ldots , k \} \right\} .
$$

Definition 2.2 (positive spanning set [5, Definition 2.2]). A finite set $S \subseteq \mathbb { R } ^ { n }$ is a positive spanning set (PSS) of a convex cone $C \subseteq \mathbb { R } ^ { n } { \mathrm { ~ i f ~ } } \mathrm { p o s } \left( S \right) = C$ . In this case, S is said to positively span C.

Definition 2.3 (cone of feasible directions). Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a convex set and let ${ \bar { x } } \in { \mathcal { C } }$ . The cone of feasible directions is given by

$$
\begin{array} { r } { D c ( \bar { x } ) : = \{ w \mid \exists \lambda > 0 , \bar { x } + \lambda w \in \mathcal C \} . } \end{array}
$$

The next lemma records the elementary fact that feasible directions form a convex cone. A proof is included in Section A for completeness.

Lemma 2.1. Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a convex set, and let ${ \bar { x } } \in { \mathcal { C } }$ . Then $D _ { \mathcal { C } } \left( \bar { x } \right)$ is a convex cone.

Definition 2.4 (directional (d-)stationarity). A point ${ \bar { x } } \in { \mathcal { C } }$ is d-stationary for (P) if

$$
h ^ { \prime } ( \bar { x } ; d ) \geq 0 \qquad \forall d \in D c ( \bar { x } ) .
$$

Since $f$ is continuously diferentiable and $g$ is convex, the map $d \mapsto h ^ { \prime } ( \bar { x } ; d )$ is continuous, so the same condition is equivalent to $h ^ { \prime } ( \bar { x } ; d ) \geq 0$ for every $d \in \mathrm { c l } ( \mathrm { D } c ( \bar { \bf x } ) )$

We will use the Euclidean distance from a point to a nonempty set, the induced distance between two nonempty sets, and the Hausdorf distance between compact sets.

Definition 2.5 (Hausdorf distance). The Hausdorf distance between two nonempty compact sets $A , B \subseteq \mathbb { R } ^ { n }$ is

$$
d _ { H } ( A , B ) : = \operatorname* { m a x } \left\{ \operatorname* { s u p } _ { a \in A } \operatorname* { i n f } _ { b \in B } \left\| a - b \right\| , \ \operatorname* { s u p } _ { b \in B } \operatorname* { i n f } _ { a \in A } \left\| a - b \right\| \right\} .
$$

Definition 2.6 (Euclidean distance). For $x \in \mathbb { R } ^ { n }$ and a nonempty set $A \subseteq \mathbb { R } ^ { n }$ ，

$$
\operatorname { d i s t } ( x , A ) : = \operatorname* { i n f } _ { a \in A } \left\| x - a \right\| .
$$

For nonempty sets A, $B \subseteq \mathbb { R } ^ { n }$ 2

$$
\operatorname { d i s t } ( A , B ) : = \operatorname* { i n f } _ { a \in A } \operatorname { d i s t } ( a , B ) .
$$

Our approach relies on foundational notions which we introduce and utilize in this section – proofs not given in the text are in Section A.

Definition 2.7 (afine hull). Let $S \subseteq \mathbb { R } ^ { n }$ . Then the afine hull of S is given by

$$
\operatorname { a f f } \left( S \right) : = \left\{ \sum _ { i = 1 } ^ { k } \alpha _ { i } x _ { i } \mid k > 0 , \ x _ { 1 } , \ldots , x _ { k } \in S , \ \alpha _ { 1 } , \ldots , \alpha _ { k } \in \mathbb { R } , \ \sum _ { i = 1 } ^ { k } \alpha _ { i } = 1 \right\} .
$$

The relative interior of a set is defined via the afine hull.

Definition 2.8 (relative interior). Let $S \subseteq \mathbb { R } ^ { n }$ . Then the relative interior of S is given by

$$
\mathrm { r i } \left( \mathrm { S } \right) : = \left\{ x \in S \mid \exists \epsilon > 0 , \ B \left( x , \epsilon \right) \cap \mathrm { a f f } \left( S \right) \subseteq S \right\} .
$$

We next define the operation of shifting a set by a reference point.

Definition 2.9. Let $S \subseteq \mathbb { R } ^ { n }$ be a nonempty set, and let ${ \bar { x } } \in S$ . Then the set $S - { \bar { x } }$ is given by

$$
S - { \bar { x } } = \left\{ y \in \mathbb { R } ^ { n } \mid y + { \bar { x } } \in S \right\} .
$$

Remark 2.1. Note that for every nonempty set $S \subseteq \mathbb { R } ^ { n }$ and every ${ \bar { x } } \in S$ , we have $0 \in S - { \bar { x } }$

The next lemma provides key foundational properties we will use, some are derived from [30, Proposition 2.40].

Lemma 2.2 (foundational properties). Let $S \subseteq \mathbb { R } ^ { n }$ be a nonempty set. Then af $( S ) =$ af (cl (S)), and for $\bar { x } , \bar { y } \in S$ it holds that af $( S - { \bar { x } } ) = \operatorname { a f f } \left( S - { \bar { y } } \right) . I f S$ is in addition convex, then ri $\mathrm { ( S ) } = \mathrm { r i } ( \mathrm { c l } ( \mathrm { S } ) ) , \mathrm { r i } ( \mathrm { S } ) \neq \emptyset$ , and the line segment principle extends to the relative interior, that is, $f o r x \in \mathrm { r i } ( \mathrm { S } ) , y \in \mathrm { c l } ( \mathrm { S } )$ , and any $\lambda \in ( 0 , 1 ]$ , it holds that $\lambda x + ( 1 - \lambda ) y \in \mathrm { r i } \left( \mathrm { S } \right)$

For a linear subspace $L \subseteq \mathbb { R } ^ { n }$ , we write $\mathbb { S } _ { L } : = \{ \omega \in L : \| \omega \| = 1 \}$ for the unit sphere of L. When $L = \operatorname { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$ for some ${ \bar { x } } \in { \mathcal { C } }$ , this notation is independent of the reference point by the preceding lemma.

We continue with the definition of the relative boundary.

Definition 2.10 (Relative Boundary). Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ . The relative boundary of ${ \mathcal { C } } ,$ rb (C), is given by rb $( \mathcal { C } ) = \mathrm { c l } \left( \mathcal { C } \right) \backslash \mathrm { r i } \left( \mathcal { C } \right)$

Definition 2.11 (Smoothness at Boundary Point). Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a closed convex set. Then C is smooth at a boundary point x¯ ∈ rb (C) if

$$
\operatorname { c l } \left( \operatorname { D } _ { \mathcal { C } } ( \bar { \operatorname { x } } ) \right) = \{ d \in \mathbb { R } ^ { n } \mid \langle d , u \rangle \geq 0 \} \cap \operatorname { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)
$$

for some non-zero vector $u \in \mathrm { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$

Lemma 2.3. Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a closed convex set, and let ${ \bar { x } } \in { \mathcal { C } }$ . Then,

$$
D _ { \mathcal { C } } ( \bar { x } ) \subseteq \mathrm { c l } \left( \mathrm { D } _ { \mathcal { C } } ( \bar { \bf x } ) \right) \subseteq \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right) .
$$

Proof. The inclusion $D c ( \bar { x } ) \subseteq \mathrm { c l } \left( \mathrm { D } c ( \bar { { \mathrm { x } } } ) \right)$ is immediate. Let $d \in D _ { \mathcal { C } } ( \bar { x } )$ . By definition, there exists $\epsilon _ { d } > 0$ such that $\bar { x } + \epsilon _ { d } d = : x _ { d } \in \mathcal { C }$ , hence $\begin{array} { r } { d = \frac { x _ { d } - \bar { x } } { \epsilon _ { d } } } \end{array}$ . Since $0 \in \mathcal { C } - \bar { x }$ , the set af $( { \mathcal { C } } - { \bar { x } } )$ is a linear subspace, and therefore closed under scalar multiplication; as $x _ { d } - { \bar { x } } \in { \mathcal { C } } - { \bar { x } } \subseteq { \mathrm { a f f ~ } } ( { \mathcal { C } } - { \bar { x } } )$ we conclude $d \in \mathrm { a f f } \ ( { \mathcal { C } } - { \bar { x } } )$ . Hence $D c ( \bar { x } ) \subseteq \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$

Now, let $\bar { d } \in \mathrm { ~ c l } ( \mathrm { D } c ( \bar { \bf x } ) )$ . Then there exists a sequence $\{ d _ { i } \} _ { i \ge 0 } ~ \subseteq ~ D c ( \bar { x } )$ , such that $d _ { i } \xrightarrow { i  \infty } \bar { d } .$ Since $D c ( \bar { x } ) \subseteq \mathrm { a f f } ( \mathcal { C } - \bar { x } )$ , it follows that $\{ d _ { i } \} _ { i \geq 0 } \subseteq \mathrm { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$ . Since C is a finite dimensional set, its afine hull is closed, and hence $\bar { d } \in \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ . We conclude that cl $( \mathrm { D } c ( \bar { \mathbf { x } } ) ) \subseteq \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ ✷

Lemma 2.4. Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a closed convex set, and let $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ . Then $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$

Proof. The inclusion $D c ( \bar { x } ) \subseteq \mathrm { a f f } ( \mathcal { C } - \bar { x } )$ is Lemma 2.3. For the reverse inclusion, let $\tilde { d } \in$ af $( \mathcal { C } - \bar { x } ) , \tilde { d } \neq 0$ . Since $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ , there exists $\epsilon > 0$ with $B ( \bar { x } , \epsilon ) \cap$ af $( { \mathcal { C } } ) \subseteq { \mathcal { C } }$ . Setting $t = \epsilon / ( 2 \left\| \tilde { d } \right\| )$ gives $\bar { x } + t \tilde { d } \in B ( \bar { x } , \epsilon ) \cap \mathrm { a f f } \left( \mathcal { C } \right) \subseteq \mathcal { C } .$ , so $\tilde { d } \in D c ( \bar { x } )$ ✷

The afine hull, relative interior, and relative boundary, together with the smoothness notion of Definition 2.11 and the basic feasible-direction properties of Lemmas 2.3, 2.4, play a central role in defining η-feasible directions, which are discussed in the following section.

## 3 η-Feasible Directions

To address the geometric dificulties induced by general convex constraints, specifically, the structure of the feasible-direction cone near the boundary, we introduce the concept of η- feasible directions (η-FD). These feasible directions are angularly separated from the relative boundary of the feasible-direction cone; a geometric interpretation is given in Remark 3.1.

Definition 3.1 (η-Feasible Directions $\left( \eta \mathrm { - F D } \right) )$ . Let $\eta \in \ [ 0 , 1 ]$ and $x \in { \mathcal { C } }$ . A direction $0 \neq d \in D c ( x )$ is called an η-FD with respect to C at $x ,$ if it holds that

$$
\frac { \left| \langle d , \bar { d } \rangle \right| } { \left\| d \right\| \cdot \left\| \bar { d } \right\| } \leq \sqrt { 1 - \eta ^ { 2 } } , \qquad \forall \bar { d } \in \mathrm { r b } \left( \mathrm { D } c ( \mathbf { x } ) \right) , \bar { d } \neq 0 .
$$

The set of all η-FDs at $x \in { \mathcal { C } }$ is denoted by $D _ { \mathcal { C } } ^ { \eta } ( x )$

Remark 3.1 (angular interpretation). Let $\theta = \arcsin ( \eta ) \in [ 0 , \pi / 2 ]$ . Then the defining inequality is equivalent to

$$
\operatorname { a r c c o s } \left( \frac { | \langle d , \bar { d } \rangle | } { \| d \| \| \bar { d } \| } \right) \ge \theta \qquad \forall \bar { d } \in \mathrm { r b } \left( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } ) \right) , \bar { d } \ne 0 .
$$

Thus, an η-feasible direction is a feasible direction that remains uniformly separated, by an angle of at least $\theta ,$ from the lines generated by the relative-boundary directions of $D _ { \cal C } ( x )$ Equivalently, larger values of η correspond to directions that lie deeper inside the feasibledirection cone.

Example 3.1 (η-FDs in $\ell _ { 2 } { \mathrm { - b a l l } } )$ . We illustrate the structure of η-FDs when the feasible set is the $\ell _ { 2 } { \mathrm { - n o r m } }$ ball. If $\mathcal { C } = \mathcal { B } _ { 2 } [ 0 , 1 ] = \{ x \in \mathbb { R } ^ { n } \mid \| x \| _ { 2 } \leq 1 \}$ . Then rb $( { \mathcal { C } } ) = \{ x \in \mathbb { R } ^ { n } \mid \| x \| _ { 2 } = 1 \}$ , and

$$
D _ { \mathcal { C } } ^ { \eta } ( x ) = \left\{ \begin{array} { l l } { \left\{ 0 \neq d \ | \ \left. x , \frac { d } { \| d \| } \right. \leq - \eta \right\} , } & { \mathrm { i f ~ } \| x \| _ { 2 } = 1 , } \\ { \mathbb { R } ^ { n } - \left\{ 0 \right\} , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The next lemma gives two useful characterizations of η-feasible directions. Although Definition 3.1 is stated in terms of the relative boundary of $D c ( \bar { x } )$ , the condition simplifies in the two cases needed later.

First, if $D _ { \mathcal { C } } ( \bar { x } )$ coincides with the afine hull of ${ \mathcal { C } } - { \bar { x } }$ , then every nonzero feasible direction is an $\eta { - } \mathrm { F D }$ . Second, if $\mathrm { c l } ( D c ( \bar { x } ) )$ is a half-space in this afine hull, then η-feasibility is equivalent to a single normalized inner-product inequality.

These two cases are instantiated in Corollary 3.1 below, which identifies relative-interior points with part (a) and smooth boundary points with part (b).

Lemma 3.1 (on the characterizations of η-FD). Let ${ \bar { x } } \in { \mathcal { C } } $ , and $\eta \in [ 0 , 1 ]$

1. Entire afine hull: If $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ , then every nonzero $d \in D c ( \bar { x } )$ is an $\eta { - } F D _ { - }$

2. Half-space within the $\mathbf { \Gamma } _ { a f f i n e }$ hull: $I f \operatorname { c l } \left( \mathrm { D } _ { \mathcal { C } } ( \bar { \operatorname { x } } ) \right) = \{ d \in \mathbb { R } ^ { n } \mid \langle d , u \rangle \geq 0 \} \cap \operatorname { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$ where $u \in \mathrm { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$ and $\lVert u \rVert = 1$ , then $d \in D _ { \mathcal { C } } ( \bar { x } )$ such that $\| d \| = 1$ is an $\eta { - } F D$ if and only $i f \left. d , u \right. \geq \eta$

Proof. Suppose that $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ . Since $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ is an afine space, it holds that cl $\left( \operatorname { a f f } \left( { \mathcal { C } } - { \bar { \mathbf { x } } } \right) \right) = \operatorname { r i } \left( \operatorname { a f f } \left( { \mathcal { C } } - { \bar { \mathbf { x } } } \right) \right) = \operatorname { a f f } \left( { \mathcal { C } } - { \bar { \mathbf { x } } } \right)$ . Hence, the first part follows trivially from

$$
\operatorname { r b } \left( \operatorname { D } _ { \mathcal { C } } ( \bar { \operatorname { x } } ) \right) = \operatorname { r b } \left( \operatorname { a f f } \left( { \mathcal { C } } - { \bar { \operatorname { x } } } \right) \right) = \operatorname { c l } \left( \operatorname { a f f } \left( { \mathcal { C } } - { \bar { \operatorname { x } } } \right) \right) - \operatorname { r i } \left( \operatorname { a f f } \left( { \mathcal { C } } - { \bar { \operatorname { x } } } \right) \right) = \varnothing .
$$

We proceed to the second part. Let $W = \{ u , v _ { 1 } , \dotsc , v _ { \dim } \dotsc 1 \}$ be some orthogonal spanning set of af $( { \mathcal { C } } - { \bar { x } } )$ that includes u, such that $\| v _ { 1 } \| = \ldots = \left\| v _ { \dim } \mathcal { C } - 1 \right\| = 1$ . Note that

$$
\mathrm { r b } \left( \mathrm { D } _ { \mathcal { C } } ( \bar { \mathbf { x } } ) \right) = \mathrm { r b } \left( \left\{ \mathrm { d } \in \mathbb { R } ^ { { \mathrm { n } } } \mid \langle \mathbf { u } , \mathbf { d } \rangle \geq 0 \right\} \cap \mathrm { a f f } \left( \mathcal { C } - \bar { \mathbf { x } } \right) \right) = \left\{ d \in \mathbb { R } ^ { n } \mid \langle u , d \rangle = 0 \right\} \cap \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right) ,
$$

assume that $\langle d , u \rangle \geq \eta$ , and let $\tilde { d } \in \mathrm { r b } \left( \mathrm { D } \varsigma ( \bar { \bf x } ) \right)$ . Then there exist coeficients $\tilde { a } _ { 1 } , \dots , \tilde { a } _ { \mathrm { d i m } \mathcal { C } - 1 } \in$ dim C=   
$\mathbb { R } .$ such that $\tilde { d } = 0 \cdot u + \sum _ { i = 1 } ^ { \mathrm { u n n c - 1 } } \tilde { a } _ { i } v _ { i }$ . Meanwhile, since $d \in D c ( \bar { x } ) \subseteq \mathrm { c l } \left( \mathrm { D } _ { \mathcal { C } } ( \bar { { \mathrm { x } } } ) \right)$ , there exist   
coeficients $c , a _ { 1 } , \ldots , a _ { \dim { \mathscr { C } } - 1 } \in \mathbb { R } .$ , such that

$$
d = c \cdot u + \sum _ { i = 1 } ^ { \mathrm { d i m } \mathscr { C } - 1 } a _ { i } v _ { i } .
$$

Since $\eta \leq \langle d , u \rangle = c .$ , and $1 = \| d \| ^ { 2 } = c ^ { 2 } + \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } a _ { i } ^ { 2 }$ , it follows that $\sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } a _ { i } ^ { 2 } \leq 1 - \eta ^ { 2 }$ . Note that $\left| \langle d , \tilde { d } \rangle \right| = \operatorname* { m a x } \left\{ \langle d , \tilde { d } \rangle , \langle d , - \tilde { d } \rangle \right\}$ , and that both

$$
\langle d , \tilde { d } \rangle \leq \operatorname* { s u p } _ { q \in \operatorname { r b } D _ { \mathcal { C } } ( \bar { x } ) } \left\{ \langle d , q \rangle \ | \ \| q \| ^ { 2 } \leq \left\| \hat { d } \right\| ^ { 2 } \right\} \mathrm { a n d } \langle d , - \tilde { d } \rangle \leq \operatorname* { s u p } _ { q \in \operatorname { r b } D _ { \mathcal { C } } ( \bar { x } ) } \left\{ \langle d , q \rangle \ | \ \| q \| ^ { 2 } \leq \left\| \hat { d } \right\| ^ { 2 } \right\}
$$

hold true. Therefore,

$$
\left| \langle d , \tilde { d } \rangle \right| \leq \operatorname* { s u p } _ { q \in \mathrm { r b } D _ { \mathcal { C } } ( \bar { x } ) } \left\{ \langle d , q \rangle \mid \Vert q \Vert ^ { 2 } \leq \left. \tilde { d } \right. ^ { 2 } \right\} .
$$

Rewriting the supremum in term of the orthogonal decomposition, we derive

$$
\operatorname* { s u p } _ { b _ { 1 } , \ldots , b _ { \dim } c _ { - 1 } } \left\{ \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } a _ { i } b _ { i } \bigm | \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } b _ { i } ^ { 2 } \leq \big \| \tilde { d } \big \| ^ { 2 } \right\} .
$$

Using Cauchy-Schwarz, $\sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } - 1 } a _ { i } b _ { i } \leq \| a \| \cdot \| b \| \leq \| a \| \cdot \left\| \tilde { d } \right\|$ , and the upper bound is attained for $b = \left\| \tilde { d } \right\| a / \left\| a \right\|$ . Therefore,

$$
\left| \left. d , \tilde { d } \right. \right| \leq \left( \sqrt { \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } a _ { i } ^ { 2 } } \right) \cdot \left\| \tilde { d } \right\| \leq \sqrt { 1 - \eta ^ { 2 } } \cdot \left\| \tilde { d } \right\| .
$$

Combining with the assumption that $\| d \| = 1$ , we have that

$$
\frac { \big | \langle d , \tilde { d } \rangle \big | } { \| d \| \cdot \big \| \tilde { d } \big \| } \leq \sqrt { 1 - \eta ^ { 2 } } .\tag{3.1}
$$

Thus, $\langle d , u \rangle \geq \eta$ implies that d is $\eta { - } \mathrm { F D }$

In the other direction, assume that (3.1) holds true for every $\tilde { d } \in \mathrm { r b } \left( \mathrm { D } \varsigma ( \bar { \bf x } ) \right)$ . As before, let $d = c \cdot u + \sum _ { i = 1 } ^ { \mathrm { d i m } { \mathcal { C } } - 1 } a _ { i } \cdot v _ { i }$ be the decomposition of d according to the normalized orthogonal basis W. Note that if $c < 0$ , then $\langle d , u \rangle < 0$ , and hence d $\notin$ cl $( \mathrm Ḋ c Ḍ Ḍ ( \bar { \mathrm { x } } ) )$ , in contradiction to the assumption that $d \in D _ { \mathcal { C } } ( \bar { x } )$ , and if $c > 1$ , then $\| d \| > 1$ , in contradiction to the assumption that $\| d \| = 1$

Consequently, $c \in [ 0 , 1 ]$ . If $c = 1$ , then $\langle d , u \rangle = 1 \geq \eta$ , and the claim follows. Otherwise, assume $c < 1$ , and hence $\sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } - 1 } a _ { i } ^ { 2 } > 0$ . Consider the vector $\bar { d } = \sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } - 1 } a _ { i } \cdot v _ { i }$ . Since $\langle \bar { d } , u \rangle = 0$ and $\bar { d } \in \mathrm { c l } \left( \mathrm { D } _ { \mathcal { C } } ( \bar { \mathbf { x } } ) \right)$ , it follows that $\bar { d } \in \mathrm { r b } \left( \mathrm { D } _ { \mathcal { C } } ( \bar { \bf x } ) \right)$ , and therefore, $\tilde { d } = \bar { d }$ satisfies (3.1). Since $1 = \| d \| ^ { 2 } = c ^ { 2 } + \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } a _ { i } ^ { 2 }$ , it follows that

$$
\left\| \bar { d } \right\| = \sqrt { \sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } - 1 } a _ { i } ^ { 2 } } = \sqrt { 1 - c ^ { 2 } } .
$$

Additionally, $\left| \langle d , { \bar { d } } \rangle \right| = \sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } - 1 } a _ { i } ^ { 2 } = 1 - c ^ { 2 }$ . Combining the above then yields that

$$
\frac { \left| \langle d , \tilde { d } \rangle \right| } { \left\| d \right\| \cdot \left\| \tilde { d } \right\| } = \frac { 1 - c ^ { 2 } } { 1 \cdot \sqrt { 1 - c ^ { 2 } } } \leq \sqrt { 1 - \eta ^ { 2 } } \Rightarrow c \geq \eta ,
$$

where the last transition follows from the fact that $c , \eta \in [ 0 , 1 ]$ . Thus, $\langle d , u \rangle \geq \eta$ , which concludes the proof for $x \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$

The following corollary specializes Lemma 3.1 to interior points and smooth boundary points; it follows directly from Definition 2.11, Lemma 2.4, and Lemma 3.1.

Corollary 3.1. Let ${ \bar { x } } \in { \mathcal { C } }$ , and $\eta \in [ 0 , 1 ]$

(a) $I f \bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ , then $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ , and every $0 \neq d \in D \mathcal { c } ( \bar { x } )$ is an $\eta { - } F D$

(b) $\begin{array} { l } { I f { \bar { x } } \in \operatorname { r b } \left( { \mathcal { C } } \right) } \\ { i f { \bar { \mathbf { \Theta } } } } \end{array}$ and C is smooth at x¯, then d $\in D _ { \mathcal { C } } ( \bar { x } )$ with $\| d \| = 1$ is an $\eta { - } F D$ if and only

$$
\langle d , u \rangle \geq \eta ,
$$

where $u \in \mathrm { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$ is a unit vector satisfying

$$
\operatorname { c l } \left( \operatorname { D } _ { \mathcal { C } } ( \bar { \mathbf { x } } ) \right) = \left\{ \tilde { d } \in \mathbb { R } ^ { n } \mid \langle u , \tilde { d } \rangle \geq 0 \right\} \cap \operatorname { a f f } \left( \mathcal { C } - \bar { x } \right) .
$$

Proof. Part $( a )$ . By Lemma 2.4, $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ . Since af $( { \mathcal { C } } - { \bar { x } } )$ is a linear subspace, its relative boundary is empty, so the condition in Definition 3.1 is vacuously satisfied for every nonzero d. Part 1 of Lemma 3.1 then gives that every $0 \neq d \in D \ o c ( \bar { x } )$ is an $\eta { - } \mathrm { F D }$

Part (b). By Definition 2.11, smoothness at x¯ means that

$$
\operatorname { c l } \left( \operatorname { D } _ { \mathcal { C } } ( \bar { \operatorname { x } } ) \right) = \{ d \in \mathbb { R } ^ { n } \mid \langle d , u \rangle \geq 0 \} \cap \operatorname { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)
$$

for some non-zero $u \in \mathrm { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$ . Rescaling $u  u / \| u \|$ yields a unit vector satisfying the half-space assumption of Part 2 of Lemma 3.1, from which the $\eta { - } \mathrm { F D }$ characterization $\langle d , u \rangle \geq \eta$ follows directly. ✷

## 4 The Oracle-Based Regularized Feasible Directions Method

This section introduces the Oracle-Based Regularized Feasible Directions Method (Algorithm 1). At each iteration, an oracle produces a positively spanning set of η-feasible directions, and exact line searches are applied to the associated univariate regularized subproblems.

The methods access h only through the exact regularized line searches in line 7 of Algorithm 1 and line 5 of Algorithm 3 (Remark 4.2). Algorithm 1 accesses C through an η-direction oracle, defined next, and through the step-size map τ.

Definition 4.1 (η-Direction Oracle). Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a nonempty closed convex set. A mapping $\mathcal { A } : \mathcal { C } \times ( 0 , 1 ] \to \mathcal { P } _ { f i n } \left( \mathbb { R } ^ { n } \right)$ is called an η-Direction Oracle $\left( \eta \mathrm { - D O } \right)$ if for every ${ \bar { x } } \in { \mathcal { C } }$ and $\eta \in ( 0 , 1 ]$ , the output $X = \mathcal { A } ( \bar { x } , \eta )$ satisfies:

1. Each $d \in X$ lies in the feasible direction cone: $d \in D c ( \bar { x } )$

2. The positive span of X contains all η-feasible directions at x¯: $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } ) \subseteq \mathrm { p o s } \left( X \right)$

Remark 4.1 (On the case $\eta = 0 )$ . The oracle is defined only for $\eta > 0$ . The reason is that, for general closed convex sets, the full feasible-direction cone $D _ { \cal C } ( \bar { x } )$ need not be closed and therefore need not admit a finite positive spanning set of feasible directions. For example, if $\mathcal { C } = B _ { 2 } [ 0 , 1 ]$ and $\| { \boldsymbol { x } } \| _ { 2 } = 1$ , then

$$
D _ { \mathcal { B } [ 0 , 1 ] } ( x ) = \{ d \in \mathbb { R } ^ { n } \ | \ \langle x , d \rangle < 0 \} ,
$$

which is not closed and cannot be represented as the positive span of finitely many feasible directions. When C is polyhedral, $D _ { \mathcal { C } } ( \bar { x } )$ is closed and finitely generated; see, e.g., [6]. In that case, the oracle may be extended to $\eta = 0$ by returning a positive spanning set of $D _ { \cal C } ( \bar { x } )$ , and we use this extension whenever applicable.

Assumption 1 (Constraint-set access). Let $\mathcal { C } \subseteq \mathbb { R } ^ { n }$ be a nonempty closed convex set. Algorithm 1 is given an η-direction oracle (Definition $4 . 1 )$ and, for every $x \in { \mathcal { C } }$ and $v \in \mathbb { R } ^ { n } \backslash \{ 0 \}$ , the value $\tau ( x ; v ) : = \operatorname* { s u p } \{ t \geq 0 : x + t v \in \mathcal { C } \}$

Remark 4.2 (Line-search access). Both methods access h only through exact solutions of the regularized univariate problems

$$
\operatorname* { m i n } _ { \substack { 0 \le t \le \operatorname* { m i n } \{ \tau ( x ; v ) , r \} } } \varphi _ { v } ( t ) , \qquad \varphi _ { v } ( t ) : = h ( x + t v ) + \frac { \gamma } { 2 } t ^ { 2 } ,
$$

as in [6]. For nonconvex h this amounts to global minimization of a nonconvex univariate function; we adopt it as a modeling assumption and do not pursue inexact line searches here. The oracle uses neither derivatives of h nor its decomposition into $f$ and $g .$

Below we provide classes of sets for which an $\eta { - } \mathrm { D O }$ can be implemented straightforwardly.

1. Polyhedral Constraints. Suppose that ${ \mathcal { C } } = \{ x \in \mathbb { R } ^ { n } \mid A x \leq b \}$ , where $A \in \mathbb { R } ^ { m \times n }$ and $b \in \mathbb { R } ^ { m }$ . Let A denote the ith row of A, and define the active set at ${ \bar { x } } \in { \mathcal { C } }$ by $\mathcal { T } _ { \mathcal { C } } ( \bar { x } ) : = \{ i \ | \ [ A \bar { x } ] _ { i } = b _ { i } \}$ . Then, the feasible directions at x¯ are given by

$$
D _ { \mathcal { C } } ( \bar { x } ) = \left\{ d \in \mathbb { R } ^ { n } \ | \ A _ { i } d \leq 0 \quad \forall i \in \mathcal { I } _ { \mathcal { C } } ( \bar { x } ) \right\} .
$$

Computing a finite set of vectors that positively span this cone corresponds to converting its H-representation to a V-representation – a well-studied task in polyhedral theory. Eficient algorithms for this conversion exist depending on the feasible set; see [14, Section 9].

Algorithm 1: Oracle-Based Regularized Feasible Directions Method   
Input: $x _ { 0 } \in \mathcal { C } , \ \eta \in ( 0 , 1 ] , \gamma > 0 , \ r > 0 , \ \eta \mathrm { - D O } \ A : \mathcal { C } \times ( 0 , 1 ] \to \mathcal { P } _ { f i n } ( \mathbb { R } ^ { n } )$   
1 for $k = 0 \ldots , K$ do   
2 $\tilde { v } _ { \mathrm { e x } } ^ { k } \sim U n i f \left( \mathbb { S } _ { L } \right) , \quad L : = \mathrm { a f f } \left( \mathscr { C } - x _ { k } \right)$   
3 $V ^ { k } = \mathcal { A } ( x _ { k } , \eta )$   
4 $\tilde { V } ^ { k } : = \left\{ \tilde { v } _ { 1 } ^ { k } , \dots , \tilde { v } _ { s _ { k } } ^ { k } \right\} = V ^ { k } \cup \left\{ \tilde { v } _ { \mathrm { e x } } ^ { k } \right\}$   
5 for $i \in \{ 1 , \ldots , s _ { k } \}$ do   
6 $\tau ( x _ { k } ; \tilde { v } _ { i } ^ { k } ) : = \operatorname* { s u p } \{ t \geq 0 : x _ { k } + t \tilde { v } _ { i } ^ { k } \in \mathcal { C } \}$   
7 t<sup>k</sup><sub>i</sub> ∈ argmin h(x<sub>k</sub> + tv˜<sup>k</sup><sub>i</sub> ) + $\frac { \gamma } { 2 } t ^ { 2 }$   
$0 { \le } t { \le } \operatorname* { m i n } \{ \tau ( x _ { k } ; \tilde { v } _ { i } ^ { k } ) , r \}$   
8 end   
9 $i ^ { * } \in$ argmin $h \left( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ^ { k } \right)$   
i∈{1,...,s<sub>k</sub>}   
10 $\boldsymbol { x } _ { k + 1 } = \boldsymbol { x } _ { k } + t _ { i ^ { * } } ^ { k } \tilde { \boldsymbol { v } } _ { i ^ { * } } ^ { k }$   
11 end

2. Smooth Convex Sets. When the boundary of C is smooth at every point, such as when C is the $\ell _ { 2 } { \mathrm { - b a l l } }$ (also see Example 3.1) or an ellipsoid, the non-exploration output of Algorithm 2 with zero noise is an η-DO (Lemma 7.2). This holds for every $\rho _ { 0 } \geq 0$ . Under the curvature assumptions of Lemma 6.4 and with $\rho _ { 0 } = r _ { \mathrm { i n } } / 2$ , this $\eta { - } \mathrm { D O }$ also has the spanning-coeficient bounds of Definition 6.2 (Proposition 6.2).

## 3. Orthogonally-Structured Constraints. Suppose that

$$
\mathcal C = \{ x \in \mathbb R ^ { n } \mid h _ { i } ( x ) \leq 0 , i = 1 , \ldots , k \} ,\tag{4.1}
$$

where each $h _ { i }$ is convex and diferentiable, and $\nabla h _ { i } ( x ) \neq 0$ whenever $h _ { i } ( x ) = 0$ . A suficient condition for the latter is Slater’s condition – that is, the existence of some xˆ for which $h _ { i } ( \hat { x } ) < 0$ for all i. Additionally, assume that the gradients of active constraints are orthogonal. This structure includes various settings with mutually exclusive or disjoint constraints. For $\eta \in ( 0 , 1 ]$ , under these assumptions, the set of η-feasible directions at x¯ is given by

$$
D _ { \mathcal { C } } ^ { \eta } ( \bar { x } ) = \left\{ d \in \mathbb { R } ^ { n } \mid d \neq 0 , \quad \left. \frac { \nabla h _ { i } ( \bar { x } ) } { \| \nabla h _ { i } ( \bar { x } ) \| } , \frac { d } { \| d \| } \right. \leq - \eta \quad \forall i \in \mathcal { I } ( \bar { x } ) \right\} ,
$$

where $\mathcal { T } ( \bar { x } ) : = \{ i \mid h _ { i } ( \bar { x } ) = 0 \}$ is the set of active constraints at x¯.

Define $u _ { i } = - \nabla h _ { i } ( \bar { x } ) / \| \nabla h _ { i } ( \bar { x } ) \|$ and $\begin{array} { r } { v = \sum _ { i \in \mathcal { I } ( \bar { x } ) } u _ { i } } \end{array}$ . Let $\bar { \eta } = \eta / ( 2 \sqrt { n } )$ , and let $\{ w _ { j } \}$ be an orthonormal basis for the orthogonal complement of $\{ u _ { i } \} _ { i \in \mathbb { Z } ( \bar { x } ) }$ . Then the set

$$
V : = \{ v \} \cup \bigcup _ { i \in \mathcal { I } ( \bar { x } ) } \left\{ ( 1 - \bar { \eta } ) u _ { i } + \bar { \eta } v \right\} \cup \bigcup _ { j = 1 } ^ { n - | \mathcal { Z } ( \bar { x } ) | } \left\{ ( 1 - \bar { \eta } ) w _ { j } + \bar { \eta } v , - ( 1 - \bar { \eta } ) w _ { j } + \bar { \eta } v \right\}\tag{4.2}
$$

contains only feasible directions, and its positive span includes $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ ; The proof is nearly identical to that of Lemma 7.2. Thus, the mapping $\mathcal { A } ( \bar { x } , \eta ) = V$ defines a valid $\eta { - } \mathrm { D O }$

Access for the constructions above. Algorithm 1 uses only the η-DO and the map τ of Assumption 1. Polyhedra and orthogonally structured constraints supply an η-DO by the constructions above, and do not use Algorithm 2. For smooth convex sets, the η-DO is the non-exploration output of Algorithm 2 with zero noise, which at $\rho _ { 0 } = 0$ requires, at each query point $x \in { \mathcal { C } } .$ , a relative-interior membership test, an orthonormal basis of af $( { \mathcal { C } } - x )$ , and, if $x \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ , the unique unit vector u of Definition 2.11. We assume that these three items are available from the description of C, as they are for the Euclidean ball and for ellipsoids; smoothness of the boundary does not, by itself, yield them for an arbitrary smooth convex set. A membership oracle can be used to estimate them, but does not compute them exactly. Implementation of the same construction at $\rho _ { 0 } > 0$ is discussed in Remark 5.1. The map τ is obtained by one-dimensional root finding along the ray. For polyhedra, one may instead compute the distance to each linear constraint.

## 5 The Noised Regularized Feasible Directions Method

The Noised Regularized Feasible Directions Method (Algorithm 3) replaces the η-direction oracle with an explicit construction based on local boundary geometry (Algorithm 2). It combines basis directions, inward tilting, and random perturbations. For $\alpha > 0$ , the iterates almost surely avoid nonsmooth boundary points when initialized outside that set (Theorem 7.1).

The threshold $\rho _ { 0 } \geq 0$ determines where boundary geometry is used: $\rho _ { 0 } = 0$ restricts its use to boundary points, whereas $\rho _ { 0 } > 0$ also includes nearby interior points. The required constraint queries are described at the end of Section 4 and in Remark 5.1.

Remark 5.1 (Access at $\rho _ { 0 } > 0 )$ . Choosing $\rho _ { 0 } > 0$ requires access to $\rho ( x )$ and, when $0 < \rho ( x ) \le \rho _ { 0 }$ , a nearest boundary point ${ \hat { x } } \in \Pi ( x )$ and its inward unit normal. These quantities can be computed explicitly for Euclidean balls and full-dimensional polyhedra. For ellipsoids, the nearest-point problem reduces to quadratic minimization on a sphere [26].

For general convex sets, this geometric access is an additional assumption; membership queries alone do not provide it exactly. When it is unavailable, one may choose $\rho _ { 0 } = 0$ , using only the queries listed at the end of Section 4. If rb $( { \mathcal { C } } ) = \emptyset$ , we set $\rho ( x ) : = + \infty$ , so the first branch of Algorithm 2 applies and no boundary query is needed.

## 6 Analysis

We establish PDM convergence rates and almost-sure d-stationarity of accumulation points for Algorithms 1 and 3. We also derive finite-time bounds for approximate d-stationarity and specialize them under curvature assumptions. The analysis relies on the feasibility, spanning, and boundary-avoidance results proved in Section 7.

## 6.1 The PDM measure and a finite-direction rate

The analysis of Algorithm 1 and Algorithm 3 is based on a feasible-directions optimality measure that extends the one introduced by [6]. We refer to this quantity as the proximal directional minimization (PDM) measure. The measure quantifies the penalized decrease available along a finite set of feasible directions, and is therefore suited to methods that explore finitely many directions per iteration.

For any feasible ${ \bar { x } } \in { \mathcal { C } }$ , direction set $V = \{ v _ { i } \} _ { i = 1 } ^ { s }$ , penalty parameter $\gamma \geq 0$ , and interval radius $r \in ( 0 , \infty ]$ , the PDM is defined as

$$
\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) : = \sum _ { i = 1 } ^ { s } \bigl ( \mu ^ { \gamma } ( \bar { x } ; v _ { i } , r ) \bigr ) ^ { 2 } ,\tag{6.1}
$$

where the v-proximal directional minimizer scalars are defined by

$$
\mu ^ { \gamma } ( \bar { x } ; v , r ) \in \operatorname * { a r g m i n } _ { 0 \leq t \leq \operatorname * { m i n } \{ \tau ( \bar { x } ; v ) , r \} } \left\{ h ( \bar { x } + t v ) + \frac { \gamma } { 2 } t ^ { 2 } \right\} ,\tag{6.2}
$$

Algorithm 2: ConstructSearchDirections   
Input: $x \in \mathcal { C } , \eta \in ( 0 , 1 ] , \alpha \in [ 0 , 1 ) , \rho _ { 0 } \geq 0$   
Output: Search directions $\tilde { V }$ (including exploration direction)   
// queries $\rho ( x )$ ; if $\rho \le \rho _ { 0 }$ , also $\Pi ( x )$   
1 ${ \tilde { V } }  \emptyset ;$   
2 $\bar { \eta }  \frac { \eta } { 2 \sqrt { \mathrm { d i m } \mathcal { C } } } ;$   
// Step 1: Exploration direction   
3 $\tilde { v } _ { \mathrm { e x } } \sim U n i f \left( \mathbb { S } _ { L } \right) , \quad L : = \mathrm { a f f } \ ( \mathscr { C } - x ) ;$   
// Step 2: Base directions   
4 $\rho \gets \rho ( x ) ;$   
5 if $\rho > \rho _ { 0 }$ then   
6 $W  \{ w _ { 1 } , \ldots , w _ { \mathrm { d i m } } \mathcal { c } \}$ , an orthonormal basis of af $( { \mathcal { C } } - x )$   
7 $U \gets \{ w _ { 1 } , - w _ { 1 } , \dots , w _ { \dim { \mathcal C } } , - w _ { \dim { \mathcal C } } \} ;$ ;   
8 $T \gets \emptyset ;$   
9 else if $\rho = 0$ and $\mathcal { C }$ is not smooth at x then   
10 $\tilde { V } \gets \{ \tilde { v } _ { \mathrm { e x } } \}$ ;   
11 return $\tilde { V }$   
12 else $/ /$ smooth boundary or shallow interior   
13 ${ \hat { x } } \in \Pi ( x )$ ; $\langle \mathcal { I } / \mathrm { ~ \Pi ~ } \Pi ( x ) = \{ x \}$ when $\rho = 0$   
14 $u $ inward unit normal to C at ${ \hat { x } } , u \in \operatorname { a f f } \left( { \mathcal { C } } - x \right)$   
15 Extend to $W = \left\{ u , w _ { 1 } , \ldots , w _ { \mathrm { d i m } } \ldots \right\}$ , an orthonormal basis of af $( { \mathcal { C } } - x )$   
16 $U \gets \{ u , - u , w _ { 1 } , - w _ { 1 } , \dots , w _ { \mathrm { d i m } } \varepsilon _ { - 1 } , - w _ { \mathrm { d i m } } \varepsilon _ { - 1 } \} \cap \mathrm { r i } \left( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } ) \right)$ ;   
17 $T \gets \{ w _ { 1 } , - w _ { 1 } , \ldots , w _ { \dim { \mathcal { C } } - 1 } , - w _ { \dim { \mathcal { C } } - 1 } \} ;$   
18 end   
// Step 3: Tilt and noise   
19 Enumerate the set $U \cup T = \{ v _ { 1 } , \ldots , v _ { m } \} ;$   
20 Sample $\delta \sim U n i f \left( \Delta _ { m } \right)$ and set $\begin{array} { r } { \zeta \gets \sum _ { i = 1 } ^ { m } \delta _ { i } v _ { i } ; } \end{array}$   
21 for $v \in T$ do   
22 $\tilde { V } \gets \tilde { V } \cup \{ \mathrm { n o r m a l i z e } \big ( ( 1 - \alpha ) \bar { \eta } u + \alpha \bar { \eta } \zeta + ( 1 - \bar { \eta } ) v \big ) \} ;$   
23 end   
24 for $v \in U$ do   
25 $\tilde { V } \gets \tilde { V } \cup \{ \mathrm { n o r m a l i z e } \big ( \alpha \bar { \eta } \zeta + \left( 1 - \alpha \bar { \eta } \right) v \big ) \}$ ;   
26 end   
27 $\tilde { V }  \tilde { V } \cup \{ \tilde { v } _ { \mathrm { e x } } \}$ ;   
28 return $\tilde { V }$

Algorithm 3: The Noised Regularized Feasible Directions Method   
Input: $x _ { 0 } \in \mathcal { C } , \gamma \in ( 0 , \infty ) , r > 0 , \eta \in ( 0 , 1 ] , \alpha \in [ 0 , 1 ) , \rho _ { 0 } \geq 0$   
1 for $k = 0 , \ldots , K$ do   
2 V<sup>˜</sup> <sup>k</sup> ← ConstructSearchDirections(x<sub>k</sub>, η, α, ρ<sub>0</sub>) (Algorithm 2);   
3 for $\tilde { v } \in \tilde { V } ^ { k }$ do   
4 $\tau ( x _ { k } ; \tilde { v } ) \gets \operatorname* { s u p } \{ t \geq 0 : x _ { k } + t \tilde { v } \in \mathcal { C } \} ;$   
5 $\begin{array} { r } { t _ { \tilde { v } } \in \arg \operatorname* { m i n } _ { t \in [ 0 , \operatorname* { m i n } \{ \tau ( x _ { k } ; \tilde { v } ) , r \} ] } h ( x _ { k } + t \tilde { v } ) + \frac { \gamma } { 2 } t ^ { 2 } ; } \end{array}$   
6 end   
7 $\tilde { v } ^ { * } \in \arg \operatorname* { m i n } _ { \tilde { v } \in \tilde { V } ^ { k } } h ( x _ { k } + t _ { \tilde { v } } \tilde { v } )$   
8 $x _ { k + 1 } \gets x _ { k } + t _ { \tilde { v } ^ { * } } \tilde { v } ^ { * } ;$   
9 end

and $\tau ( \bar { x } ; v ) : = \operatorname* { s u p } \{ t \geq 0 : \bar { x } + t v \in \mathcal { C } \}$ is the maximal feasible step length at x¯ in direction v.   
When multiple minimizers exist, we select the smallest one.

The quantity $\mu ^ { \gamma } ( \bar { x } ; v , r )$ represents the step size that achieves the best penalized decrease along direction v, while $\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r )$ aggregates the descent attainable over the direction set V. In particular, $\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) = 0$ if and only if no penalized decrease is available along any direction in V; Lemma 6.1 below relates this to first-order conditions.

The following lemma is adapted from [6, Lemma 4.1].

Lemma 6.1. Let $\bar { x } \in \mathcal { C } , r \in ( 0 , \infty ] , V \subseteq D \mathcal { C } ( \bar { x } )$ be finite, and K be a cone such that the positive span of V is a superset of K. ${ \cal I } f { \mathcal O } ^ { \gamma } ( \bar { x } ; V , r ) = 0$ , then

$$
\langle \nabla f ( { \bar { x } } ) , d \rangle - g ^ { \prime } ( { \bar { x } } ; d ) \geq 0 , \qquad \forall d \in { \mathcal { K } } .
$$

The proof is in Section B.

Lemma 6.2 (properties of $h ^ { \prime } ( \bar { x } ; \cdot ) )$ . Let C be closed and convex, let f be continuously diferentiable, let g be convex, and let ${ \bar { x } } \in { \mathcal { C } }$ . Then:

1. $I f { \mathcal { C } }$ is compact and g is finite near C, then

$$
\Lambda : = \operatorname* { s u p } _ { x \in { \mathcal { C } } } \left( \| \nabla f ( x ) \| + \operatorname* { m a x } _ { s \in \partial g ( x ) } \| s \| \right)
$$

is finite, $d \mapsto h ^ { \prime } ( \bar { x } ; d )$ is Λ-Lipschitz, and $\left| h ^ { \prime } ( { \bar { x } } ; d ) \right| \leq \Lambda \left\| d \right\|$

2. $I f V \subseteq D c ( \bar { x } )$ is finite and $\begin{array} { r } { d = \sum _ { v \in V } \lambda _ { v } v } \end{array}$ with $\lambda _ { v } \geq 0$ , then $\begin{array} { r } { - h ^ { \prime } ( \bar { x } ; d ) \leq \sum _ { v \in V } \lambda _ { v } [ - h ^ { \prime } ( \bar { x } ; v ) ] + } \end{array}$

The proof is in Section B. Part 1 is the Lipschitz refinement of the continuity recorded in Lemma B.4 and supplies the constant Λ used below. Part 2 is the quantitative form of Lemma 6.1: taking all $[ - h ^ { \prime } ( \bar { x } ; v ) ] _ { + } = 0$ recovers it.

The PDM measure is nonlocal: it requires the absence of penalized descent anywhere along the feasible segment [0, min $\{ \tau ( \bar { x } ; v ) , r \} ]$ for every $v \in V$ , and is therefore strictly stronger than the first-order condition $h ^ { \prime } ( \bar { x } ; v ) \geq 0$ . Even when the directional derivative is nonnegative, a step $t > 0$ satisfying $\begin{array} { r } { h ( \bar { x } + t v ) + \frac { \gamma } { 2 } t ^ { 2 } < h ( \bar { x } ) } \end{array}$ may still exist, and the line search then returns a positive step.

Under additional smoothness assumptions, the following proposition provides a quantitative lower bound on the PDM measure in terms of directional derivatives and feasible step lengths; see [6, Theorem 4.1].

Proposition 6.1 (Lower bound on the PDM measure; adapted from [6, Theorem 4.1]). Suppose $g \equiv 0$ and $\dot { \boldsymbol { f } } \in C _ { L } ^ { 1 , 1 }$ for some $L > 0$ . Then for any ${ \bar { x } } \in { \mathcal { C } }$ and finite direction set V with $\lVert \boldsymbol { v } \rVert = 1$ for all $v \in V .$

$$
\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) \geq \frac { 1 } { ( L + \gamma ) ^ { 2 } } \sum _ { v \in V } \Bigl ( \operatorname* { m i n } \{ [ - \langle \nabla f ( \bar { x } ) , v \rangle ] _ { + } , ( L + \gamma ) \operatorname* { m i n } \{ \tau ( \bar { x } ; v ) , r \} \} \Bigr ) ^ { 2 } .\tag{6.3}
$$

The proof is in Section B. The bound shows that the PDM measure controls the magnitude of descent available along directions in V , as quantified by the projected gradient $[ - \langle \nabla f ( { \bar { x } } ) , v \rangle ] _ { + }$ truncated by the available step min $\{ \tau ( \bar { x } ; v ) , r \}$

We illustrate the significance of the PDM measure and Proposition 6.1 through three examples.

Example 6.1 (Unconstrained smooth minimization). Let $\mathcal { C } = \mathbb { R } ^ { n } , g \equiv 0 , f \in C _ { L } ^ { 1 , 1 } , r = \infty$ and take $V = \{ \pm e _ { 1 } , \ldots , \pm e _ { n } \}$ . Since $\tau ( \bar { x } ; v ) = \infty$ for every v, the minimum in (6.3) reduces to $[ - \langle \nabla f ( { \bar { x } } ) , v \rangle ] _ { + }$ for each direction. Summing over V then yields

$$
\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) \geq \frac { 1 } { ( L + \gamma ) ^ { 2 } } \| \nabla f ( \bar { x } ) \| ^ { 2 } ,
$$

so the PDM measure provides a lower bound proportional to the standard gradient-norm optimality measure, and $\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) = 0$ implies $\nabla f ( { \bar { x } } ) = 0$

Example 6.2 (Interior point). Suppose $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ , let $\{ w _ { 1 } , \ldots , w _ { \mathrm { d i m } } c \}$ be an orthonormal basis of af $( { \mathcal { C } } - { \bar { x } } )$ , and set $V = \{ w _ { 1 } , - w _ { 1 } , \ldots , w _ { \mathrm { d i m } } c , - w _ { \mathrm { d i m } } c \}$ . By Lemma 2.4, $D _ { \mathcal { C } } ( \bar { x } ) =$ af $( { \mathcal { C } } - { \bar { x } } )$ , and by construction pos $( V ) =$ af $( { \mathcal { C } } - { \bar { x } } )$ as well. Therefore Lemma 6.1 applies to all feasible directions: $\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) = 0$ implies x¯ is d-stationary, without any smoothness assumption on $f \ \mathrm { o r } \ g .$

Under the additional assumptions $g \equiv 0$ and $f \in C _ { L } ^ { 1 , 1 }$ , Proposition 6.1 yields a complementary interpretation. Since $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ , every $v \in V$ satisfies min $\{ \tau ( \bar { x } ; v ) , r \} > 0$ , so each term in the lower bound (6.3) vanishes only if $\langle \nabla f ( { \bar { x } } ) , v \rangle \geq 0$ . Since V contains both v and −v for every basis vector, this forces $\langle \nabla f ( { \bar { x } } ) , v \rangle = 0$ for all $v \in V$ , and hence the projection of $\nabla f ( { \bar { x } } )$ onto af $( { \mathcal { C } } - { \bar { x } } )$ is zero – a necessary optimality condition for smooth minimization over C at a relative interior point.

Example 6.3 (Boundary point with curved nonlinear constraints). Suppose $\mathcal { C }$ is compact, fix $\eta \in ( 0 , \frac { 1 } { 2 } ]$ , and let ${ \hat { x } } \in \operatorname { r b } \left( { \mathcal { C } } \right)$ be a point at which C is smooth, with inward unit normal u. Let V positively span $D _ { \mathcal { C } } ^ { \eta } ( \hat { x } )$ but not necessarily the full cone $D c ( \hat { x } )$ ; Lemma 7.2 shows that the non-exploration output of Algorithm 2 with $\alpha = 0$ has this property. Let Λ be the constant of Lemma 6.2(1), which is finite since C is compact and $g$ is finite near ${ \mathcal { C } } ;$ then $d \mapsto h ^ { \prime } ( x ; d )$ is Λ-Lipschitz for every $x \in { \mathcal { C } }$ . Let $d \in D c ( \hat { x } )$ be a unit feasible direction that is not an η-FD. Then $0 \leq \langle d , u \rangle < \eta$ by Corollary $3 . 1 ( \mathrm { b } )$ . Set $s : = \eta / ( 1 - \eta )$ . Then $\| d + s u \| ^ { 2 } = 1 + 2 s \langle d , u \rangle + s ^ { 2 } \geq 1 + s ^ { 2 } > 0 .$ , and the tilted direction $w : = ( d + s u ) / \| d + s u \|$ satisfies $\langle w , u \rangle \geq \eta$ , so $w \in D _ { \mathcal { C } } ^ { \eta } ( \hat { x } )$ , while $\| w - d \| \leq 2 s \leq 4 \eta$ . Hence

$$
- h ^ { \prime } ( \hat { x } ; d ) \leq - h ^ { \prime } ( \hat { x } ; w ) + 4 \Lambda \eta ,
$$

so the descent rate available outside the $\eta { - } \mathrm { F D }$ cone exceeds that available inside it by at most $4 \Lambda \eta$ . Consequently the PDM measure, constructed using η-FDs, captures all descent directions up to an ${ \cal { O } } ( \eta )$ error in the directional derivative.

Remark 6.1 (Scope of the PDM certificate). By Lemma 6.1, $\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) = 0$ certifies $h ^ { \prime } ( \bar { x } ; d ) \ge 0$ only on pos (V). Table 1 lists, for each method and type of point, a cone contained in pos (V), where V is the set of non-exploration search directions. At relativeinterior points the certificate is full d-stationarity; at smooth boundary points, Example 6.3 bounds the descent missed outside the $\eta { - } \mathrm { F D }$ cone by ${ \cal { O } } ( \eta )$

Table 1: Cones on which vanishing PDM certifies $h ^ { \prime } ( \bar { x } ; \cdot ) \geq 0$ . For Algorithm 3 with $\alpha > 0$ “small” refers to the pointwise thresholds in Corollary 7.2; the last entry assumes $x _ { 0 }$ is almost surely not a nonsmooth boundary point.
<table><tr><td></td><td>Algorithm 1</td><td>Algorithm  $3 , \alpha = 0$ </td><td>Algorithm 3, small  $\alpha > 0$ </td></tr><tr><td rowspan="2"> $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ </td><td> $D _ { \cal C } ( \bar { x } )$ </td><td> $D _ { \cal C } ( \bar { x } )$ </td><td> $D _ { \cal C } ( \bar { x } )$ </td></tr><tr><td>Definition 4.1, Corollary 3.1(a) Lemma 7.2</td><td></td><td>Corollary 7.2(1)</td></tr><tr><td rowspan="2">smooth boundary</td><td> $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ </td><td> $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ </td><td> $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) , \beta \in ( 1 , 1 / \eta )$ </td></tr><tr><td>Definition 4.1</td><td>Lemma 7.2</td><td> $\mathrm { C o r o l l a r y \ 7 . 2 ( 2 ) }$ </td></tr><tr><td rowspan="2">nonsmooth boundary</td><td> $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ </td><td>none</td><td>not reached a.s.</td></tr><tr><td>Definition 4.1</td><td>Remark 7.1</td><td>Theorem 7.1</td></tr></table>

Having related the PDM measure to finite-direction stationarity, we now establish a rate. Theorem 6.1 shows that the iterates approach finite-direction stationarity at an $O ( 1 / K )$ rate in the PDM measure, counted in outer iterations. Classical direct-search guarantees have the same order in a diferent unit, function evaluations; see the comparison after Corollary 6.2. Write $s _ { k } : = | \tilde { V } ^ { k } |$ for the number of search directions at iteration $k ,$ as in Algorithm 1, and set $s _ { \mathrm { m a x } } ^ { K } : = \mathrm { m a x } _ { 0 \leq i \leq K - 1 } s _ { i }$ . When K is clear from the context we write $s _ { \mathrm { m a x } }$ for $s _ { \mathrm { m a x } } ^ { K }$

Theorem 6.1. Let $\{ x _ { k } \} _ { k \ge 0 }$ be a sequence generated by Algorithm 1 or Algorithm 3. Then for any $K \geq 1$ it holds that

$$
\operatorname* { m i n } _ { 0 \leq k \leq K - 1 } \mathcal { O } ^ { \gamma } ( x _ { k } ; \tilde { V } ^ { k } , r ) \leq \frac { 2 s _ { \operatorname* { m a x } } } { \gamma K } \left[ \left( f ( x _ { 0 } ) - g ( x _ { 0 } ) \right) - \operatorname* { i n f } _ { x \in \mathcal { C } } \left\{ f ( x ) - g ( x ) \right\} \right] .
$$

Proof. Enumerate $\tilde { V } ^ { k } = \{ \tilde { v } _ { 1 } , \dots , \tilde { v } _ { s _ { k } } \}$ . Then by the definition of $t _ { i } ^ { k }$ , for all $i \in \{ 1 , \ldots , s _ { k } \}$

$$
\big ( f ( x _ { k } ) - g ( x _ { k } ) \big ) - \Big ( f ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ) - g ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ) \Big ) \geq \frac { \gamma } { 2 } { t _ { i } ^ { k } } ^ { 2 } = \frac { \gamma } { 2 } \big ( \mu ^ { \gamma } ( x _ { k } ; \tilde { v } _ { i } , r ) \big ) ^ { 2 } .
$$

Summing over $i ,$

$$
\sum _ { i = 1 } ^ { s _ { k } } \Big [ ( f ( x _ { k } ) - g ( x _ { k } ) ) - \Big ( f ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ) - g ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ) \Big ) \Big ] \geq \frac { \gamma } { 2 } \mathcal { O } ^ { \gamma } ( x _ { k } ; \tilde { V } ^ { k } , r ) .
$$

Using the fact that $i _ { k } ^ { * }$ maximizes the diference between the function values

$$
\left( f ( x _ { k } ) - g ( x _ { k } ) \right) - \left( f ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ) - g ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ) \right) ,
$$

the fact that $s _ { m a x } \ge s _ { k }$ , and that $x _ { k + 1 } = x _ { k } + t _ { i _ { k } ^ { * } } ^ { k } \tilde { v } _ { i _ { k } ^ { * } }$ , we deduce that

$$
s _ { m a x } \cdot [ ( f ( x _ { k } ) - g ( x _ { k } ) ) - ( f ( x _ { k + 1 } ) - g ( x _ { k + 1 } ) ) ] \geq \frac { \gamma } { 2 } \mathcal { O } ^ { \gamma } ( x _ { k } ; \tilde { V } ^ { k } , r ) .\tag{6.4}
$$

Summing over $k = 0$ to $k = K - 1$ and dividing by K finally yields that

$$
\frac { 2 s _ { m a x } } { \gamma K } \left[ \left( f ( x _ { 0 } ) - g ( x _ { 0 } ) \right) - \operatorname* { i n f } _ { x \in \mathcal { C } } \left\{ f ( x ) - g ( x ) \right\} \right] \geq \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathcal { O } ^ { \gamma } ( x _ { k } ; \tilde { V } ^ { k } , r )
$$

The rate bound requires neither directional spanning nor smoothness of C. Spanning determines the scope of the non-descent certificate in Lemma 6.1 (see Remark 6.1). When $\tilde { V } ^ { \smash { \sum } }$ contains only the exploration direction, the bound concerns that direction alone; Remark 7.1 gives conditions excluding this case. We next bound the direction-set size $s _ { \mathrm { m a x } }$ appearing in Theorem 6.1.

Remark 6.2 (Bounds on $s _ { \mathrm { m a x } } )$ . Let $m = \dim { \mathcal { C } }$ . In Algorithm $^ { 3 , }$ counting the exploration direction gives

$$
s _ { k } \leq \left\{ \begin{array} { l l } { 2 m + 1 , } & { \rho ( x _ { k } ) > \rho _ { 0 } , } \\ { 1 , } & { x _ { k } \mathrm { ~ i s ~ a ~ n o n s m o o t h ~ b o u n d a r y ~ p o i n t } , } \\ { 2 m , } & { x _ { k } \mathrm { ~ i s ~ a ~ s m o o t h ~ b o u n d a r y ~ p o i n t } , } \\ { 4 m - 1 , } & { 0 < \rho ( x _ { k } ) \leq \rho _ { 0 } . } \end{array} \right.
$$

The last bound counts both the tilted and untilted directions generated from $T \subseteq U$ (Remark 7.2). Consequently, $s _ { \mathrm { m a x } } \leq 2 m + 1$ when $\rho _ { 0 } = 0$ , and $s _ { \operatorname* { m a x } } \leq 4 m - 1$ when $\rho _ { 0 } > 0$

In Algorithm 1, $s _ { k } \leq | V ^ { k } | + 1$ , so the bound depends on the oracle implementation. For the constructions in Section 4, the polyhedral case gives one plus the maximum number of cone generators returned by the oracle. For smooth convex sets, the zero-noise construction of Algorithm 2 gives the same bounds as above. For orthogonally structured constraints, (4.2) gives $s _ { k } \leq 2 n + 2 - | \mathcal { T } ( x _ { k } ) | \leq 2 n + 2$

## 6.2 Subsequence convergence to d-stationarity

The PDM rate in Theorem 6.1 concerns the directions searched at each iteration. At boundary points, their positive span may omit feasible directions, so this bound alone does not establish d-stationarity. Theorem 6.2 shows that randomized exploration ensures, almost surely, that every accumulation point is d-stationary. The proof combines two observations: objective decrease forces the exploration step lengths to vanish, while repeated sampling near any nonstationary accumulation point would produce exploration steps bounded away from zero.

Theorem 6.2 (Subsequence convergence; adapted from [16, Theorem 4.1]). Let $\{ x _ { k } \} _ { k = 0 } ^ { \infty }$ be a sequence generated by Algorithm 1 or Algorithm 3. Then, almost surely, every accumulation point of $\{ x _ { k } \}$ is a d-stationary point of (P).

Theorem 6.2 is, in substance, [16, Theorem 4.1(ii)]. There, an arbitrary method producing feasible iterates is augmented with an exploration step whose step size is sampled at random and accepted upon suficient decrease. Algorithm 1 and Algorithm 3 have the same structure, with the non-exploration line searches in the role of the base method, except that the exploration step size is obtained from the regularized line search of Remark 4.2. This afects only the final step of the argument. In [16], a sampled step size bounded away from zero is accepted infinitely often, contradicting the summability of the squared exploration steps; here, such a step size is admissible in the line search, and the optimality of the line-search minimizer yields the contradiction (Lemma B.1). The remaining ingredients are those of [16], with minor modifications. For completeness, Section B.1 gives the full proof and indicates the source of each ingredient.

Remark 6.3 (Existence of accumulation points; cf. [16, Remark 4.1]). A suficient condition for the existence of an accumulation point of $\{ x _ { k } \}$ is boundedness of the level set $\{ x \in { \mathcal { C } }$ $h ( x ) \leq h ( x _ { 0 } ) \}$ }. Indeed, $t = 0$ is admissible in each regularized line search, so every candidate satisfies

$$
\begin{array} { r } { h ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ^ { k } ) \leq h ( x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ^ { k } ) + \frac { \gamma } { 2 } ( t _ { i } ^ { k } ) ^ { 2 } \leq h ( x _ { k } ) , } \end{array}
$$

and therefore $h ( x _ { k + 1 } ) \leq h ( x _ { k } )$ for all $k ,$ independently of which candidate is selected. Hence $h ( x _ { k } ) \leq h ( x _ { 0 } )$ and $\{ x _ { k } \}$ remains in this set. If the set is bounded, then $\{ x _ { k } \}$ is bounded in $\mathbb { R } ^ { n }$ and admits an accumulation point by the Bolzano–Weierstrass theorem.

## 6.3 Finite-time approximate directional stationarity

Theorem 6.1 bounds the PDM measure on the directions actually searched. We now bound the directional derivative over all feasible directions. $\mathrm { B y }$ Proposition 6.1, when $g \equiv 0$ a small PDM value controls $[ - \langle \nabla f ( { \bar { x } } ) , v \rangle ] _ { + }$ only up to truncation by the feasible step min $\{ \tau ( \bar { x } ; v ) , r \}$ and near rb (C) that step can be arbitrarily small.

The truncation is not an artifact of the bound. Let u be a unit vector and ${ \mathcal { C } } : = \{ x \ :$ $\langle x , u \rangle \geq 0 \}$ . Let $f ( x ) : = \langle u , x \rangle , g \equiv 0$ and $\bar { x } : = \rho u$ with $\rho > 0$ . Then $h ^ { \prime } ( \bar { x } ; d ) = \langle d , u \rangle$ , and the steepest unit feasible direction is $- u .$ , with rate 1. A unit d with $c : = \langle d , u \rangle < 0$ has $\tau ( \bar { x } ; d ) = \rho / | c |$ |. Every feasible step along a descent direction therefore decreases h by at most $| c | \tau ( \bar { x } ; d ) = \rho$

As $\rho \downarrow 0$ the decrease available to any method vanishes while the steepest rate stays 1. Hence per-iteration decrease cannot certify a small directional derivative over all of $D _ { \cal C } ( \bar { x } )$ and we discard the directions along which one cannot travel far.

Definition 6.1 (R-deep residual; $( \varepsilon , R )$ -d-stationarity). Let C be closed and convex, let f be diferentiable, and let $g$ be convex. For ${ \bar { x } } \in { \mathcal { C } }$ and $R \geq 0$ let

$$
\Delta ^ { [ R ] } ( \bar { x } ) : = \mathrm { s u p } \{ [ - h ^ { \prime } ( \bar { x } ; d ) ] _ { + } : d \in D _ { \mathcal { C } } ( \bar { x } ) , \| d \| = 1 , \tau ( \bar { x } ; d ) \geq R \} ,\tag{6.5}
$$

with sup $\varnothing : = 0$ , and write $\Delta : = \Delta ^ { [ 0 ] }$ . We call $\bar { x } \ ( \varepsilon , R )$ -d-stationary if $\Delta ^ { [ R ] } ( \bar { x } ) \leq \varepsilon$

In the example above, $\Delta ^ { [ R ] }$ keeps only descent directions with $\rho / | c | \geq R$ , so $\Delta ^ { [ R ] } ( \bar { x } ) \leq \rho / R$ For fixed R this is of the same order as the available decrease, whereas $\Delta ( \bar { x } ) = 1$ . More generally, when C is compact and g is finite near C, a direction with $\tau ( \bar { x } ; d ) < R$ ofers at most $\Lambda R$ of first-order decrease by Lemma $6 . 2 ( 1 )$ . The threshold R is on the feasible step, not on the depth $\rho ( \bar { x } )$

$\Delta ( \bar { x } ) = 0$ if and only if x¯ is d-stationary. Since $h ^ { \prime } ( \bar { x } ; \cdot )$ is continuous, the supremum over $D _ { \cal C } ( \bar { x } )$ agrees with the one over cl $( \mathrm { D } c ( \bar { \bf x } ) )$ , so $\Delta$ measures d-stationarity in the strong sense of Theorem 6.2. The map $R \mapsto \Delta ^ { [ R ] } ( \bar { x } )$ is nonincreasing and increases to $\Delta ( \bar { x } )$ as $R \downarrow 0$ . If every unit feasible direction at x¯ admits a step of length at least R, then $\Delta ^ { [ R ] } ( \bar { x } ) = \Delta ( \bar { x } )$ . In particular this holds at every point when ${ \mathcal { C } } = \mathbb { R } ^ { n }$ , and Theorem 6.3 is then a rate for $\Delta$

We write $\tau _ { r } ( x ; v ) : = \operatorname* { m i n } \{ \tau ( x ; v ) , r \}$ and $H : = h ( x _ { 0 } ) - \operatorname* { i n f } _ { \mathcal { C } } h$ . The decrease along a single direction is quantified by the following analogue of Proposition 6.1 for the objective of (P).

Lemma 6.3. Let C be closed and convex, let $f \in C _ { L } ^ { 1 , 1 }$ , let g be convex, and let $\gamma \geq 0$ . Let ${ \bar { x } } \in { \mathcal { C } }$ , let $v \in D c ( \bar { x } )$ with $\lVert \boldsymbol { v } \rVert = 1$ , set $\theta : = [ - h ^ { \prime } ( \bar { x } ; v ) ] _ { + }$ . Then

$$
\begin{array} { r l r } & { } & { h ( \bar { x } ) - \underset { 0 \leq t \leq \tau _ { r } ( \bar { x } ; v ) } { \operatorname* { m i n } } \Bigl \{ h ( \bar { x } + t v ) + \frac { \gamma } { 2 } t ^ { 2 } \Bigr \} \ \geq \ \frac { 1 } { 2 } \operatorname* { m i n } \Bigl \{ \frac { \theta ^ { 2 } } { L + \gamma } , \ \theta \tau _ { r } ( \bar { x } ; v ) \Bigr \} . } \end{array}
$$

The proof is in Section B. As in the proof of Theorem $6 . 1$ , the left-hand side is the regularized decrease along $v ,$ and is comparable to $\left( \mu ^ { \gamma } ( \bar { x } ; v , r ) \right) ^ { 2 }$

The first term of the minimum yields an $\varepsilon ^ { - 2 }$ rate, but it is the smaller term only when $\tau _ { r } ( \bar { x } ; v ) \geq \theta / ( L + \gamma )$ . To transfer decrease to every R-deep direction we therefore write each such direction as a nonnegative combination of searched directions with controlled coeficients and controlled feasible-step lengths.

Definition 6.2 (η-direction oracle with bounds on the spanning coeficients). Let C be closed and convex and let $r \in ( 0 , \infty ]$ . A mapping $x \mapsto V ( x ) \subseteq D c ( x )$ with $\lVert \boldsymbol { v } \rVert = 1$ for all $v \in V ( x )$ has spanning-coeficient bounds $( A , B , R )$ if for every $x \in { \mathcal { C } }$ and every $d \in D c ( x )$ with $\| d \| = 1$ and $\tau ( x ; d ) \ge R$ there exist $\lambda _ { v } \geq 0 , v \in V ( x )$ , with

$$
d = \sum _ { v } \lambda _ { v } v , \qquad \sum _ { v } \lambda _ { v } \leq A , \qquad \sum _ { v } { \frac { \lambda _ { v } } { \tau _ { r } ( x ; v ) } } \leq B .\tag{6.6}
$$

The first constant is the usual $\ell _ { 1 }$ positive-spanning constant; the second weights each coeficient by the reciprocal of the guaranteed step length along its direction, and the parameter η enters the complexity only through B.

Theorem 6.3 (finite-time rate). Let C be closed and convex, let $f \in C _ { L } ^ { 1 , 1 }$ , let g be convex, and assume $H < \infty$ . Let $\gamma > 0$ and $r \in ( 0 , \infty ]$ . Let $x \mapsto V ( x )$ have spanning-coeficient bounds $( A , B , R )$ and let $\{ x _ { k } \}$ be generated by any method that, at each iteration, solves the regularized line search along every direction of a finite set containing $V ( x _ { k } )$ and moves to the candidate with the smallest value of h. Then for every $K \geq 1$ 2

$$
\operatorname* { m i n } _ { 0 \leq k \leq K - 1 } \Delta ^ { [ R ] } ( x _ { k } ) \leq A \sqrt { \frac { 2 ( L + \gamma ) H } { K } } + \frac { 2 B H } { K } .\tag{6.7}
$$

Proof. Write $q _ { k } : = h ( x _ { k } ) - h ( x _ { k + 1 } )$ . Since the method sets $\begin{array} { r } { t _ { i } ^ { k } \in \arg \operatorname* { m i n } _ { t \in [ 0 , \tau _ { r } ( x _ { k } ; v _ { i } ) ] } \{ h ( x _ { k } + } \end{array}$ $\begin{array} { r } { t v _ { i } ) + \frac { \gamma } { 2 } t ^ { 2 } \} } \end{array}$ along each searched direction $v _ { i }$ and then picks the candidate with the smallest h-value,

$$
\begin{array} { r } { q _ { k } \ = \ \underset { i } { \operatorname* { m a x } } \big [ h ( x _ { k } ) - h ( x _ { k } + t _ { i } ^ { k } v _ { i } ) \big ] \ \geq \ \underset { i } { \operatorname* { m a x } } \Big [ h ( x _ { k } ) - \underset { t \in [ 0 , \tau _ { r } ( x _ { k } ; v _ { i } ) ] } { \operatorname* { m i n } } \big \lbrace h ( x _ { k } + t v _ { i } ) + \frac { \gamma } { 2 } t ^ { 2 } \big \rbrace \Big ] , } \end{array}
$$

using $\begin{array} { r } { \frac { \gamma } { 2 } ( t _ { i } ^ { k } ) ^ { 2 } \geq 0 } \end{array}$ . Combining with Lemma 6.3, for every $v \in V ( x _ { k } )$ ，

$$
\begin{array} { r } { \operatorname* { m i n } \Bigl \{ \frac { \theta _ { v } ^ { 2 } } { L + \gamma } , \theta _ { v } \tau _ { r } ( x _ { k } ; v ) \Bigr \} \ \le \ 2 q _ { k } , \qquad \theta _ { v } : = [ - h ^ { \prime } ( x _ { k } ; v ) ] _ { + } . } \end{array}
$$

Hence, examining which branch attains the minimum,

$$
\theta _ { v } \leq \operatorname * { m a x } \Bigl \{ \sqrt { 2 ( L + \gamma ) q _ { k } } , \frac { 2 q _ { k } } { \tau _ { r } ( x _ { k } ; v ) } \Bigr \} \leq \sqrt { 2 ( L + \gamma ) q _ { k } } + \frac { 2 q _ { k } } { \tau _ { r } ( x _ { k } ; v ) } .
$$

Now let d be a unit feasible direction at $x _ { k }$ with $\tau ( x _ { k } ; d ) \ge R _ { ☉ }$ and take the decomposition guaranteed by (6.6). By Lemma 6.2(2) and the bound on $\theta _ { v }$

$$
- h ^ { \prime } ( x _ { k } ; d ) \leq \sum _ { v } \lambda _ { v } \theta _ { v } \leq \sqrt { 2 ( L + \gamma ) q _ { k } } \sum _ { v } \lambda _ { v } + 2 q _ { k } \sum _ { v } \frac { \lambda _ { v } } { \tau _ { r } ( x _ { k } ; v ) } \leq A \sqrt { 2 ( L + \gamma ) q _ { k } } + 2 B q _ { k } .
$$

Taking the supremum over such d yields $\Delta ^ { [ R ] } ( x _ { k } ) \le A \sqrt { 2 ( L + \gamma ) q _ { k } } + 2 B q _ { k }$ . Finally $\begin{array} { r } { \sum _ { k = 0 } ^ { K - 1 } q _ { k } = } \end{array}$ $h ( x _ { 0 } ) - h ( x _ { K } ) \leq H$ , so min $\natural q _ { k } \leq H / K ;$ evaluating the right-hand side at $k ^ { \star } = \arg \operatorname* { m i n } _ { k } q _ { k }$ gives (6.7). ✷

Theorem 6.3 controls $\Delta ^ { [ R ] }$ over all unit feasible directions with $\tau ( x ; d ) \ge R$ . Its proof uses spanning-coeficient bounds, is deterministic, and yields a bound independent of $s _ { \mathrm { m a x } } .$ The proof also yields an early-stopping rule. It gives $\Delta ^ { [ R ] } ( x _ { k } ) \le A \sqrt { 2 ( L + \gamma ) q _ { k } } + 2 B q _ { k }$ with $q _ { k } : = h ( x _ { k } ) - h ( x _ { k + 1 } )$ , so when A, B and L are known (e.g., via Proposition 6.2) one may stop at the first k at which this bound is at most ε. This certifies $\Delta ^ { [ R ] } ( x _ { k } ) \leq \varepsilon ,$ and since min $\nu { < } K q _ { k } \le H / K$ , it occurs before any K for which the right-hand side of (6.7) is at most ε.

We next establish the required oracle properties for the zero-noise construction in Algorithm 2 under curvature bounds. We begin with rolling-ball geometry and the resulting estimates for $\tau ( \bar { x } ; d )$ . The geometric proofs are deferred to Section B.

Lemma 6.4 (Rolling balls). Let $\mathcal { C } \subseteq \mathbb { R } ^ { n }$ be a convex body with a $C ^ { 2 }$ boundary whose principal curvatures are uniformly bounded in $\left[ \kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } } \right]$ at every boundary point, where $0 < \kappa _ { \mathrm { m i n } } \le \kappa _ { \mathrm { m a x } } < \infty$ . Write $r _ { \mathrm { i n } } : = 1 / \kappa _ { \mathrm { m a x } }$ and $r _ { \mathrm { o u t } } : = 1 / \kappa _ { \mathrm { m i n } }$ . Then:

1. At every ${ \hat { x } } \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ with inward unit normal u, $\begin{array} { r } { \mathcal { B } ( \hat { x } + r _ { \mathrm { i n } } u , r _ { \mathrm { i n } } ) \subseteq \mathcal { C } \subseteq \mathcal { B } ( \hat { x } + r _ { \mathrm { o u t } } u , r _ { \mathrm { o u t } } ) } \end{array}$

2. $I f x \in { \mathcal { C } }$ and $\rho ( x ) < r _ { \mathrm { i n } } .$ , then Π(x) is a singleton.

Lemma 6.5 (Curvature Feasibility Bounds). Suppose the assumptions of Lemma $6 . 4$ hold. Let xˆ ∈ rb (C) with inward unit normal u, let $0 \leq \rho \leq r _ { \mathrm { i n } }$ , set $\bar { x } : = \hat { x } + \rho u$ , and let $d \in S ^ { n - 1 }$ with $c : = \langle d , u \rangle$ . Then:

$$
\begin{array} { r } { I . ~ \tau ( \bar { x } ; d ) \geq ( r _ { \mathrm { i n } } - \rho ) c + \sqrt { ( r _ { \mathrm { i n } } - \rho ) ^ { 2 } c ^ { 2 } + 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } } . } \end{array}
$$

(1a) $H c \ge 0$ , then $\tau ( \bar { x } ; d ) \geq \operatorname* { m i n } \{ 2 c , 1 \} r _ { \mathrm { i n } }$

(1b) If c ≥ 0, then $\tau ( \bar { x } ; d ) \geq \sqrt { r _ { \mathrm { i n } } \rho } .$

(1c) τ (¯x; u) ≥ 2r<sub>in</sub> − ρ ≥ r<sub>in</sub>.

2. τ (¯x; d) ≤ 2r<sub>out</sub>[c]<sub>+</sub> + p2r<sub>out</sub>ρ for every $c \in [ - 1 , 1 ]$

3. $\tau ( \bar { x } ; d ) \leq \rho / | c | ~ f o r ~ c < 0$ , and $\tau ( \bar { x } ; - u ) = \rho$

Corollary 6.1. Suppose the assumptions of Lemma $6 . 4$ hold, and let $\eta \in ( 0 , 1 )$ and ${ \hat { x } } \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ If a unit direction d is an η-FD at xˆ, then $\tau ( \hat { x } ; d ) \geq 2 \eta / \kappa _ { \operatorname* { m a x } } ;$ if d is not an $\eta { - } F D _ { \mathit { \Pi } . }$ , then $\tau ( \hat { x } ; d ) \leq 2 \eta / \kappa _ { \operatorname* { m i n } }$

Under the assumptions of Lemma 6.4 every boundary point of C is smooth, so the analysis of nonsmooth boundary points in Section 7 is not needed. Corollary 6.1 relates $\Delta ^ { [ R ] }$ to the $\eta { - } \mathrm { F D }$ cone. For every $R > 2 \eta / \kappa _ { \mathrm { m i n } }$ and $\bar { x } \in \mathrm { r b } \left( \mathcal { C } \right)$

$$
\Delta ^ { [ R ] } ( \bar { x } ) \leq \Delta _ { \eta } ( \bar { x } ) : = \operatorname* { s u p } _ { d \in D _ { \mathcal { C } } ^ { \eta } ( \bar { x } ) , \| d \| = 1 } [ - h ^ { \prime } ( \bar { x } ; d ) ] _ { + } ,
$$

since by Corollary 6.1 any d with $\tau ( \bar { x } ; d ) > 2 \eta / \kappa _ { \mathrm { m i n } }$ is an η-FD (Corollary 6.2 takes $R =$ $8 \eta / \kappa _ { \mathrm { m i n } } )$ . Thus $\Delta ^ { [ R ] }$ accounts both for the ${ \cal { O } } ( \eta )$ residual of Example 6.3 and for the step truncation in Proposition 6.1.

When C satisfies the assumptions of Lemma 6.4, Theorem 6.3 applies to Algorithm 3 with $\alpha = 0$ and $\rho _ { 0 }$ of order $r _ { \mathrm { i n } }$ , because the non-exploration output of Algorithm 2 implements such an oracle. We provide the constants at $\rho _ { 0 } = r _ { \mathrm { i n } } / 2$ . In this regime $\rho \leq \rho _ { 0 } < r _ { \mathrm { i n } } .$ , so $\Pi ( x )$ is a singleton by Lemma 6.4.

Proposition 6.2. Let C satisfy the assumptions of Lemma $6 . 4 ,$ , let $\eta \ \in \ ( 0 , \frac 1 2 ]$ , and let $r \in ( 0 , \infty ]$ . Then Algorithm 2 with $\rho _ { 0 } = r _ { \mathrm { i n } } / 2$ and $\alpha = 0$ has spanning-coeficient bounds $( A , B , R )$ with

$$
A = \sqrt { \mathrm { d i m } \mathcal { C } } , \qquad R = \frac { 8 \eta } { \kappa _ { \mathrm { m i n } } } , \qquad B \le \frac { \sqrt { \mathrm { d i m } \mathcal { C } } } { r } + \frac { 2 \mathrm { d i m } \mathcal { C } \cdot \kappa _ { \mathrm { m a x } } } { \eta } ,
$$

and $| V ( x ) | \leq 4$ dim C.

The proof is in Section B.

Remark 6.4 (Role of the adapted basis and the tilt). On the Euclidean unit disk, where $r _ { \mathrm { i n } } = r _ { \mathrm { o u t } } = 1$ , dropping either the adapted basis or the tilt from Algorithm 2 yields no finite constant B in Definition 6.2. Let $\hat { x } = ( - 1 , 0 )$ with inward unit normal $u = ( 1 , 0 )$ let $x = { \hat { x } } + \rho u$ with $\rho$ small, and let $\ d = ( { \textstyle { \frac { 1 } { 2 } } } , { \textstyle { \frac { \sqrt { 3 } } { 2 } } } )$ , so $\begin{array} { r } { \langle d , u \rangle = \frac { 1 } { 2 } } \end{array}$ and Lemma 6.5(1a) gives $\tau ( x ; d ) \geq 1$ . This d is R-deep for every $R \leq 1$ and every ρ.

• Unadapted basis. Let $\begin{array} { r } { w _ { 1 } = \frac { 1 } { \sqrt { 2 } } ( 1 , 1 ) } \end{array}$ and $\begin{array} { r } { w _ { 2 } = \frac { 1 } { \sqrt { 2 } } ( - 1 , 1 ) } \end{array}$ , and take $V = \{ \pm w _ { 1 } , \pm w _ { 2 } \}$ This set positively spans $D _ { \mathcal { C } } ( x )$ . Since $\{ w _ { 1 } , w _ { 2 } \}$ is orthonormal, $d = \langle d , w _ { 1 } \rangle w _ { 1 } + \langle d , w _ { 2 } \rangle w _ { 2 }$ with $\langle d , w _ { 1 } \rangle = ( \textstyle { \frac { 1 } { 2 } } + \frac { \sqrt { 3 } } { 2 } ) / \sqrt { 2 } > 0$ and $\langle d , w _ { 2 } \rangle = \bigl ( - \textstyle { \frac { 1 } { 2 } } + \frac { \sqrt { 3 } } { 2 } \bigr ) / \sqrt { 2 } > 0$ . Since $\begin{array} { r } { \langle w _ { 2 } , u \rangle = - \frac { 1 } { \sqrt { 2 } } . } \end{array}$ Lemma 6.5(3) gives $\tau ( x ; w _ { 2 } ) ~ \le ~ \sqrt { 2 } \rho , ~ \mathrm { s o } ~ \lambda _ { w _ { 2 } } / \tau _ { r } ( x ; w _ { 2 } ) ~ \ge ~ \langle d , w _ { 2 } \rangle / ( \sqrt { 2 } \rho ) ~ = ~ ( \sqrt { 3 } ~ - \rho )$ $1 ) / ( 4 \rho ) \to \infty$ . Adding canceling ± pairs only inflates $\sum \lambda _ { v } / \tau _ { r } ( x ; v )$ , so $B = \infty$

• Adapted basis, no tilt. $V = \{ \pm u , \pm w \}$ with $w = ( 0 , 1 )$ forces $\begin{array} { r } { \lambda _ { w } = \frac { \sqrt { 3 } } { 2 } } \end{array}$ , and $\tau ( x ; w ) \leq$ $\sqrt { 2 \rho } \to 0$ by Lemma 6.5(2). Again $B = \infty$

• Adapted basis with tilt. $\begin{array} { r } { \langle d , u \rangle = \frac { 1 } { 2 } \geq \eta } \end{array}$ , so the inward-tilt case of Proposition 6.2 applies: the decomposition uses u and ${ \bar { \eta } } u + ( 1 - { \bar { \eta } } ) w$ , whose feasibility radius is at least 2η¯ independently of $\rho ,$ so the resulting bound is finite.

In this example, both modifications are therefore needed for a bound on B that is uniform in $\rho .$

Lemma 6.6. Suppose the assumptions of Lemma 6.4 hold. Let f be continuously diferentiable, let g be convex and finite near ${ \mathcal { C } } ,$ and let Λ be as in Lemma $6 . \mathcal { Z } ( 1 )$ . Let $R \in ( 0 , r _ { \mathrm { i n } } ]$ and $x \in { \mathcal { C } }$ with $\rho : = \rho ( x )$

1. $I f \rho \geq r _ { \mathrm { i n } }$ then $\Delta ^ { [ R ] } ( x ) = \Delta ( x )$

2. $I f \rho = 0$ then $\Delta ( x ) \leq \Delta ^ { [ R ] } ( x ) + 2 \Lambda \kappa _ { \mathrm { m a x } } R .$

3. $I f 0 < \rho < r _ { \mathrm { i n } }$ , with u the inward normal at the nearest boundary point, the bound of (2) holds for every unit $d \in D c ( x )$ with $\langle d , u \rangle \geq 0$ . A unit d with $\langle d , u \rangle < 0$ enters $\Delta ^ { [ R ] } ( x )$ $i f \tau ( x ; d ) \geq R$ , and is excluded from it $i f \tau ( x ; d ) < R ,$ in the extreme case $d = - u$ $\tau ( x ; - u ) = \rho .$

The proof is in Section B. A bound on $\Delta ^ { [ R ] }$ therefore controls $\Delta$ when $\rho = 0$ or $\rho \geq r _ { \mathrm { i n } } .$ up to an additive $2 \Lambda \kappa _ { \operatorname* { m a x } } R$ on the boundary. At a point with $0 < \rho < r _ { \mathrm { i n } }$ it controls every unit direction with $\tau ( x ; d ) \ge R$ , but not an outward direction that leaves C before distance $R .$ Whether a given outward direction is excluded depends on both R and $\rho ,$ not solely on $\rho \colon$ a nearly tangential outward d can have $\tau ( x ; d )$ much larger than $\rho$ and possibly at least $R .$

Corollary 6.2 (ε-complexity). Suppose the assumptions of Lemma $6 . 4$ hold, let $f \in C _ { L } ^ { 1 , 1 }$ , let $g$ be convex, and assume $H < \infty$ . Let $\varepsilon > 0$ 2 $R \in ( 0 , r _ { \mathrm { i n } } ]$ , and run Algorithm 1 with the nonexploration output of Algorithm 2 at $\rho _ { 0 } = r _ { \mathrm { i n } } / 2$ and $\eta = R \kappa _ { \mathrm { m i n } } / 8$ (then $\eta \leq \kappa _ { \operatorname* { m i n } } / ( 8 \kappa _ { \operatorname* { m a x } } ) \leq \frac 1 8$ so Proposition 6.2 applies). Then some $x _ { k }$ with $k < K$ is $( \varepsilon , R )$ -d-stationary as soon as

$$
K \ge \frac { 8 \dim \mathcal { C } ( L + \gamma ) H } { \varepsilon ^ { 2 } } + \frac { 4 H } { \varepsilon } \left( \frac { \sqrt { \dim \mathcal { C } } } { r } + \frac { 1 6 \dim \mathcal { C } \cdot \kappa _ { \operatorname* { m a x } } } { R \kappa _ { \operatorname* { m i n } } } \right) .\tag{6.8}
$$

If in addition the constant Λ of Lemma $6 . \mathcal { Z } ( 1 )$ satisfies $\Lambda > 0$ and $0 ~ < ~ \varepsilon \le ~ 4 \Lambda$ , then $R = \varepsilon / ( 4 \Lambda \kappa _ { \mathrm { m a x } } ) \in ( 0 , r _ { \mathrm { i n } } ]$ is admissible and gives $K = O \big ( \mathrm { d i m } \mathcal { C } \cdot H ( L + \gamma + \Lambda \kappa _ { \mathrm { m a x } } ^ { 2 } / \kappa _ { \mathrm { m i n } } ) \varepsilon ^ { - 2 } \big )$ $a s \varepsilon \downarrow 0$ with r held fixed. With this R, if K satisfies (6.8) with ε replaced $b y \varepsilon / 2$ , then some $x _ { k }$ with $k < K$ is $( \varepsilon / 2 , R )$ -d-stationary, and at such an iterate $\Delta ( x _ { k } ) \leq \varepsilon$ whenever $x _ { k } \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ or $\rho ( x _ { k } ) \ge r _ { \mathrm { i n } }$

Proof. Substitute $A = { \sqrt { \dim { \mathcal { C } } } }$ and the bound on B from Proposition 6.2 into (6.7) and require each of the two terms to be at most $\varepsilon / 2$ . This is (6.8). For the bound on $\Delta ,$ apply (6.8) at accuracy $\varepsilon / 2$ with $R = \varepsilon / ( 4 \Lambda \kappa _ { \mathrm { m a x } } )$ ; this yields an iterate with $\Delta ^ { [ R ] } ( x _ { k } ) \le \varepsilon / 2$ . If $\rho ( x _ { k } ) \ge r _ { \operatorname* { i n } } .$ , Lemma $6 . 6 ( 1 )$ gives $\Delta ( x _ { k } ) = \Delta ^ { [ R ] } ( x _ { k } ) \leq \varepsilon / 2$ . If $x _ { k } \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ , Lemma $6 . 6 ( 2 )$ gives $\Delta ( x _ { k } ) \le \varepsilon / 2 + 2 \Lambda \kappa _ { \operatorname* { m a x } } R = \varepsilon$ ✷

Under the parameter choices of Corollary 6.2, the guarantee requires $O ( \mathrm { d i m } { \mathcal { C } } \varepsilon ^ { - 2 } )$ outer iterations and $\bar { O } ( ( \mathrm { d i m } \mathcal { C } ) ^ { 2 } \varepsilon ^ { - 2 } )$ exact line searches, since each iteration uses $O ( \dim { \mathcal { C } } )$ directions. These counts are not directly comparable with the function-evaluation bounds of Gratton et al. [15]: each line search here solves an exact one-dimensional minimization problem (Remark 4.2).

The choices of $\rho _ { 0 }$ and η use the curvature bounds; choosing R from ε also uses Λ. Reducing η decreases $R = 8 \eta / \kappa _ { \mathrm { m i n } }$ , extending the guarantee to directions with shorter feasible steps, but increases the spanning-coeficient bound B. The choice $\eta = \Theta ( \varepsilon )$ yields the stated complexity.

Remark 6.5 (On the parameters $\gamma$ and r). The finite-time bound favors smaller $\gamma$ and larger r. However, $\gamma > 0$ is required by the PDM rate and the almost-sure convergence result (Theorems 6.1 and 6.2).

## 7 Geometry for Directional Spanning and Trajectories

This section is devoted to establishing the geometric properties underlying Section 6. We first address feasibility of the non-exploration directions and almost-sure avoidance of nonsmooth boundary points. We then establish directional spanning guarantees and explicit noise thresholds, leading to the PDM certificate in Corollary 7.2.

Lemma 7.1. Let $\{ \tilde { V } ^ { k } \} _ { k \geq 0 }$ be the sequence of finite directions set generated by Algorithm ${ \mathcal { B } } ,$ and let $\left\{ \tilde { v } _ { \mathrm { e x } } ^ { k } \right\} _ { k > 0 }$ denote the generated exploration directions. Then for any $k \geq 0$ , it holds that ${ \tilde { V } } ^ { k } \setminus \left\{ { \tilde { v } } _ { \mathrm { e x } } ^ { k } \right\} \subseteq D \mathcal { c } ( x _ { k } )$

Proof. If $x _ { k }$ is a nonsmooth boundary point, then by Algorithm 2, $\tilde { V } ^ { k } \setminus \left\{ \tilde { v } _ { \mathrm { e x } } ^ { k } \right\} = \emptyset$ , and the result holds trivially.

Otherwise, the non-exploration output of Algorithm 2 consists of the normalized images of $\alpha \bar { \eta } \zeta _ { k } + ( 1 - \alpha \bar { \eta } ) \it { v }$ for $v \in U$ and of $( 1 - \alpha ) \bar { \eta } u + \alpha \bar { \eta } \zeta _ { k } + ( 1 - \bar { \eta } ) v$ for $v \in T$ . Normalization preserves membership in $D _ { \mathcal { C } } ( x _ { k } )$

By construction $U \subseteq \mathrm { r i } \left( \mathrm { D } _ { \mathcal { C } } ( \mathbf { x } _ { \mathbf { k } } ) \right)$ . In the first branch $T = \emptyset$ and U is a signed orthonormal basis of af $( \mathcal { C } - x _ { k } ) = D \mathcal { c } ( x _ { k } )$ . In the third branch there are two cases. If $\rho ( \boldsymbol { x } _ { k } ) = 0$ , then ${ \hat { x } } = x _ { k }$ and C is smooth there, so cl $\left( \mathrm { D } c \mathrm { ( x _ { k } ) } \right)$ is a halfspace in af $( \mathcal { C } - x _ { k } )$ with inward unit normal $u ;$ the set $T = \left\{ \pm w _ { i } \right\}$ lies in the corresponding tangent hyperplane, hence in cl $\left( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } _ { \mathrm { k } } ) \right)$ , and $u \in \mathrm { r i } \left( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } _ { \mathrm { k } } ) \right)$ ). If $0 < \rho ( x _ { k } ) \le \rho _ { 0 }$ , then $x _ { k } \in \operatorname { r i } \left( { \mathcal { C } } \right)$ and $D c ( x _ { k } ) = \mathrm { a f f } \left( \mathcal { C } - x _ { k } \right)$ by Lemma 2.4, so $U \cup T \subseteq \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } _ { \mathrm { k } } ) \right)$ . Thus $U \cup T \subseteq \operatorname { c l } \left( \operatorname { D } c ( \mathbf { x _ { k } } ) \right)$ . Since $D _ { \mathcal { C } } ( x _ { k } )$ is a convex cone (Lemma 2.1), its closure is convex, and for any $\delta \in \Delta _ { m }$ ,

$$
\zeta _ { k } = \sum _ { i = 1 } ^ { m } \delta _ { i } v _ { i } \in \mathrm { c l } \left( \mathrm D _ { \mathcal { C } } ( \mathrm x _ { \mathrm { k } } ) \right) .
$$

For $v \in U$ , the untilted vector is a convex combination of $v \in \mathrm { \Gamma } \mathrm { { r i } } \left( \mathrm { D } c ( \mathrm { x } _ { \mathrm { k } } ) \right)$ and $\zeta _ { k } ~ \in$ cl $\left( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } _ { \mathrm { k } } ) \right)$ , with weight $1 - \alpha \bar { \eta } > 0$ on v. Lemma 2.2 therefore yields that the combination lies in ri $\left( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } _ { \mathrm { k } } ) \right)$

For $v \in T$ , the tilted vector is a convex combination of $u \in \mathrm { r i } ( \mathrm D _ { \mathcal { C } } ( _ { \mathrm { x } _ { \mathrm { k } } } ) ) , _ { \mathit { \zeta } _ { k } } \in \mathrm { c l } ( \mathrm D _ { \mathcal { C } } ( _ { \mathrm { x } _ { \mathrm { k } } } ) )$ and $v \in \mathrm { c l } \left( \mathrm { D } _ { \mathcal { C } } ( \mathbf { x } _ { \mathbf { k } } ) \right)$ , with weight $( 1 - \alpha ) \bar { \eta } > 0$ on u. Lemma 2.2 therefore yields membership in ri $\left( \mathrm { D } c ( \mathrm { x } _ { \mathrm { k } } ) \right)$ ✷

Nonsmooth boundary points form a small subset of rb $( \mathcal { C } )$ , much as the faces of dimension at most dim $\mathcal { C } - 2$ of a polytope form a small subset of its boundary. The precise statement, obtained via an isometric flattening of $\mathcal { C }$ onto a full-dimensional convex set, is given in Section C (Propositions C.4 and C.5). The next theorem shows that Algorithm 3 with $\alpha > 0$ started away from such points, almost surely never visits one. We give a proof sketch; the complete proof is in Section C.

Theorem 7.1. Let $\mathcal { C } _ { n s } \subseteq \mathcal { C }$ be the set of nonsmooth boundary points of C (a Borel $s e t ) _ { \it \Delta }$ and let $\{ x _ { k } \} _ { k = 0 } ^ { K }$ be a sequence generated by the first $K \geq 0$ iterations of Algorithm 3 with noise parameter $\alpha \in ( 0 , 1 )$ and $\eta \in ( 0 , 1 ]$ , initialized at $x _ { 0 } \in { \mathcal { C } }$ with $\mathbb { P } ( x _ { 0 } \in \mathcal { C } _ { n s } ) = 0$ . Then $\mathbb { P } ( x _ { k } \in \mathcal { C } _ { n s } ) = 0$ for all $k \in \{ 0 , \ldots , K \}$

Proof sketch. The argument proceeds by induction on $k ;$ the base case holds by assumption. The set $\mathcal { C } _ { n s }$ is a negligible subset of rb (C), in the sense made precise in Section C. The randomized construction of $\tilde { V } ^ { k }$ in Algorithm 2 ensures that the direction $\omega _ { i } : = \tilde { v } _ { i } ^ { k }$ has a conditional law absolutely continuous with respect to surface measure σ on $\mathbb { S } _ { L }$ , where $L : = \mathrm { a f f } \left( \mathcal { C } - x _ { k } \right)$ . This uses only that, on every branch that returns spanning directions, the diferences of the elements of $U \cup T$ span af $( \mathcal { C } - x _ { k } )$ when dim $\mathcal { C } \geq 2$ (Lemma C.3); when dim $\mathcal { C } = 1$ there are no nonsmooth boundary points. Because negligible subsets of rb (C) are mapped to σ-null sets under radial projection, $\omega _ { i }$ avoids the radial image of $\mathcal { C } _ { n s }$ almost surely. Landing in $\mathcal { C } _ { n s }$ would force $\omega _ { i }$ into that null set, so each candidate iterate $x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ^ { k }$ misses $\mathcal { C } _ { n s }$ with probability one; a union bound over the $s _ { k }$ directions closes the induction. ✷

Example 7.1 (Positive noise is necessary). The assumption $\alpha \in ( 0 , 1 )$ in Theorem 7.1 cannot be relaxed to $\alpha = 0 .$ . Let $\mathcal { C } = \left\{ z \in \mathbb { R } ^ { 2 } : z \geq 0 , \ \| z \| \leq 1 \right\}$ be the quarter disc, whose nonsmooth boundary points are the corners $0 , ( 1 , 0 )$ and $( 0 , 1 )$ . We run Algorithm 3 with $\alpha = 0 , \rho _ { 0 } = 0$ $\gamma > 0$ and $r \geq 1$ , using the standard basis on the first branch of Algorithm 2. Write $[ x _ { i } ] _ { j }$ for the jth coordinate of the ith iterate.

Construction. Fix $\delta \in ( 0 , 1 / 8 ]$ and $K > \gamma / ( 4 \delta )$ . Let $\psi : \mathbb { R }  \mathbb { R }$ be of class $C ^ { 2 }$ nonincreasing on $[ 0 , \infty )$ , constant on $[ 0 , { \textstyle \frac { 1 } { 4 } } ]$ , with $\psi ^ { \prime } \leq - K$ on $[ \frac { 9 } { 1 6 } , 1 ]$ . Let $\bar { \chi } : \mathbb { R } \to [ 0 , 1 ]$ be of class $C ^ { 2 }$ , nonincreasing on $[ 0 , \infty )$ , equal to 1 on $[ 0 , \delta ^ { 2 } ]$ and to 0 on $[ 4 \delta ^ { 2 } , \infty )$ . Both exist by $C ^ { 2 }$ interpolation on the gaps between the prescribed pieces: $\bar { \chi }$ joins the value 1 on $[ 0 , \delta ^ { 2 } ]$ to the value 0 on $[ 4 \delta ^ { 2 } , \infty )$ , and $\psi$ joins its plateau on $[ 0 , { \textstyle \frac { 1 } { 4 } } ]$ to an afine piece of slope −K on $\lbrack \frac { 9 } { 1 6 } , \infty )$ Fix $c > 2 \gamma$ and $M >$ max $\{ \psi ( 0 ) - \psi ( 1 ) + \gamma / 2 , \ 2 \sqrt 2 c \delta \}$ . Set $g \equiv 0$ and

$$
h ( z ) = f ( z ) = \psi \bigl ( \| z \| ^ { 2 } \bigr ) + \bar { \chi } \bigl ( \| z \| ^ { 2 } \bigr ) \bigl ( - M + c ( z _ { 1 } + z _ { 2 } ) \bigr ) .
$$

Then h is of class $C ^ { 2 }$ on $\mathbb { R } ^ { 2 }$ , so its second derivatives are bounded on C. On $\mathcal { C } \cap B ( 0 , 2 \delta )$ one has $z _ { 1 } + z _ { 2 } \leq \sqrt { 2 } ~ \lVert z \rVert < 2 \sqrt { 2 } \delta , \ \mathrm { s o } - M + c ( z _ { 1 } + z _ { 2 } ) < 0$ and $h ( z ) \geq \psi ( 0 ) - M = h ( 0 )$ , with equality only at $z = 0$ . On $\mathcal { C } \setminus B ( 0 , 2 \delta ) , \bar { \chi } ( \| z \| ^ { 2 } ) = 0$ , hence $h ( z ) = \psi ( \| z \| ^ { 2 } ) \geq \psi ( 1 ) > h ( 0 )$ Thus 0 is the unique minimizer of h on C.

Step 1: from $x _ { 0 }$ to the arc. Fix $x _ { 0 } ~ \in ~ \mathcal { C }$ with $[ x _ { 0 } ] _ { 1 } , [ x _ { 0 } ] _ { 2 } ~ > ~ 2 \delta$ and $\begin{array} { r } { \frac { 3 } { 4 } \ < \ \| x _ { 0 } \| \ < \ 1 } \end{array}$ Along e the second coordinate is unchanged, so the segment stays outside $B ( 0 , 2 \delta )$ . There $\begin{array} { r } { \frac { d } { d t } \left\| x _ { 0 } + t e _ { 1 } \right\| ^ { 2 } = 2 [ x _ { 0 } + t e _ { 1 } ] _ { 1 } > 4 \delta } \end{array}$ and $\begin{array} { r } { \left\| x _ { 0 } + t e _ { 1 } \right\| ^ { 2 } \geq \frac { 9 } { 1 6 } } \end{array}$ . The maximal feasible step along $+ e _ { 1 }$ is strictly less than 1, and $4 \delta K > \gamma ,$ so the derivative of $\begin{array} { r } { t \mapsto h ( x _ { 0 } + t e _ { 1 } ) + \frac { \gamma } { 2 } t ^ { 2 } } \end{array}$ is strictly less than $- 4 \delta K + \gamma t < 0$ up to the arc. The unique minimizer of the regularized line search is therefore the maximal feasible step, which lands on the open arc. Along $- e _ { 1 }$ the value of h does not decrease, so the step is $0 ; \pm e _ { 2 }$ are symmetric. Let $E = \{ \langle \tilde { v } _ { \mathrm { e x } } , x _ { 0 } \rangle \geq 0 \}$ . Then $\mathbb { P } ( E ) = 1 / 2$ , and on E one has $\| x _ { 0 } + t \tilde { v } _ { \mathrm { e x } } \| \ge \| x _ { 0 } \| > \frac { 3 } { 4 } > 2 \delta$ for every $t \geq 0$ , so the exploration segment avoids $B ( 0 , 2 \delta )$ . On $E _ { i }$ , every candidate has value at least $\psi ( 1 )$ , with equality only on the arc, and almost surely not at a corner. Hence, on $E , x _ { 1 }$ lies on the open arc.

Step 2: from the arc to the corner. On the event $E _ { i }$ write $p = x _ { 1 }$ , so $p \geq 0 , \| p \| = 1$ and $[ p ] _ { 1 } + [ p ] _ { 2 } \geq 1$ . At p the third branch of Algorithm 2 with $\alpha = 0$ returns the inward normal $u = - p$ , and $\tau ( p ; u ) = 1 \leq r$ . Let $s : = 1 - t , A : = [ p ] _ { 1 } + [ p ] _ { 2 }$ , and $\begin{array} { r } { \varphi ( t ) = h ( s p ) + \frac { \gamma } { 2 } t ^ { 2 } } \end{array}$ on $[ 0 , 1 ]$ so $\varphi ( 1 ) = \psi ( 0 ) - M + \gamma / 2$ . If $s \geq 2 \delta$ , then $\varphi ( t ) \geq \psi ( 1 ) > \varphi ( 1 )$ . On $[ 0 , 2 \delta ]$ the map $s \mapsto h ( s p )$ is $C ^ { 2 }$ with

$$
\frac { d } { d s } h ( s p ) = 2 s \bar { \chi } ^ { \prime } ( s ^ { 2 } ) ( - M + c s A ) + c A \bar { \chi } ( s ^ { 2 } ) \geq 0 ,
$$

since $\bar { \chi } ^ { \prime } \leq 0$ and $- M + c s A < 0 ;$ thus $s \mapsto h ( s p )$ is nondecreasing on [0, 2δ]. If $s \in [ \delta , 2 \delta )$ then $\begin{array} { r } { \varphi ( t ) \ge h ( \delta p ) + \frac { \gamma } { 2 } ( 1 - 2 \delta ) ^ { 2 } \ge \psi ( 0 ) - M + c \delta + \frac { \gamma } { 2 } ( 1 - 2 \delta ) ^ { 2 } } \end{array}$ , which exceeds $\varphi ( 1 )$ since cδ $\begin{array} { r } { > 2 \gamma \delta \geq \frac { \gamma } { 2 } \big ( 1 - ( 1 - 2 \delta ) ^ { 2 } \big ) } \end{array}$ . If $s < \delta$ , then $\varphi ^ { \prime } ( t ) = - c A + \gamma t \leq \gamma - c < 0$ . Hence $t = 1$ is the unique minimizer of $\varphi _ { : }$ , and the candidate along u is 0. The feasible line-search segments along the images of $T$ do not pass through 0, and the exploration segment passes through 0 with probability zero, so every other candidate almost surely has a strictly larger value of $h$ Therefore, conditionally on $E ,$ one has $x _ { 2 } = 0$ almost surely, and $\mathbb { P } ( x _ { 2 } \in { \mathcal { C } } _ { n s } ) \geq \mathbb { P } ( E ) > 0$

Remark 7.1 (Non-degenerate direction sets). Theorem 6.1 bounds the PDM measure on $\tilde { V } ^ { k }$ This is uninformative at an iterate where Algorithm 2 takes its second branch $( \rho = 0$ and C not smooth at $x _ { k } )$ and returns only the exploration direction.

If $\alpha \in ( 0 , 1 )$ and $x _ { 0 }$ is almost surely not a nonsmooth boundary point of C, then by Theorem 7.1 no iterate is a nonsmooth boundary point, almost surely. Since spanning directions are constructed whenever $x _ { k }$ is not a nonsmooth boundary point, $\tilde { V } ^ { k } \ne \{ \tilde { v } _ { \mathrm { e x } } ^ { k } \}$ almost surely for every k.

If C is smooth at every point of rb (C), the second branch is never taken for any $\alpha \in [ 0 , 1 )$ so $\tilde { V } ^ { k } \neq \{ \tilde { v } _ { \mathrm { e x } } ^ { k } \}$ at every iterate. When neither condition holds, Example 7.1 shows that the iterates can reach a nonsmooth boundary point.

The following lemma shows that, at every point that is not a nonsmooth boundary point, the non-exploration output of Algorithm 2 with $\alpha = 0$ positively spans $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ . If C has no nonsmooth boundary points, that output is therefore an $\eta { - } \mathrm { D O }$ , and Algorithm 3 with $\alpha = 0$ is an instance of Algorithm 1.

Lemma 7.2. Let ${ \bar { x } } \in { \mathcal { C } }$ and $\eta \in ( 0 , 1 ]$ , and assume that x¯ is not a nonsmooth boundary point of C. Denote by $\tilde { V }$ the set of directions returned by Algorithm ${ \mathrm { ~ \it ~ 2 ~ f o r ~ } } { \bar { x } }$ , with this η and $\alpha = 0$ where $v _ { \mathrm { e x } } \in \tilde { V }$ is the exploration direction. Then,

$$
D _ { \mathcal { C } } ^ { \eta } ( \bar { x } ) \subseteq \mathrm { p o s } \left( \tilde { V } \setminus \{ v _ { \mathrm { e x } } \} \right) .
$$

Proof. Let $d \in D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ . By Definition 3.1, $d \neq 0$ . Positive homogeneity of the positive span then lets us assume without loss of generality that $\| d \| = 1$ . Set $\bar { \eta } = \eta / ( 2 \sqrt { \dim { \mathscr { C } } } )$ , as in Algorithm 2. Because x¯ is not a nonsmooth boundary point, the second branch of Algorithm 2 is excluded, so $\tilde { V } \setminus \{ \tilde { v } _ { \mathrm { e x } } \}$ is produced by the T- and U-updates of Algorithm 2.

Case 1: $\rho ( \bar { x } ) > \rho _ { 0 }$ . By Corollary 3.1(a), every $0 \neq d \in D \mathcal { c } ( \bar { x } )$ is an $\eta { - } \mathrm { F D }$ , and $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal C - x _ { 0 } \right)$ . The first branch of Algorithm 2 sets $T = \emptyset$ and $U = \{ \pm w _ { 1 } , \dots , \pm w _ { \mathrm { d i m } } \mathcal { c } \}$ an untilted signed orthonormal basis of af (C − x¯). At $\alpha = 0$ the untilted output is U itself, which positively spans af $( \mathcal { C } - x _ { 0 } ) = D \mathcal { c } ( \bar { x } )$

Case 2: $\bar { x } \in \mathrm { r b } ( \mathcal { C } ) , \mathcal { C }$ smooth at x¯. By Corollary 3.1(b), d is an $\eta { - } \mathrm { F D }$ if and only if $\langle d , u \rangle \geq \eta$ . Decompose d in the orthonormal basis $W = \left\{ u , w _ { 1 } , \ldots , w _ { \dim { \mathcal { C } } - 1 } \right\}$

$$
d = c \cdot u + \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } a _ { i } w _ { i } , \qquad c = \langle d , u \rangle \geq \eta .
$$

Since $\| d \| = 1$ and W is orthonormal, $c ^ { 2 } + \textstyle \sum a _ { i } ^ { 2 } = 1$ , and by $\| \cdot \| _ { 1 } \leq \sqrt { \dim { \mathscr { C } } } \| \cdot \| _ { 2 }$

$$
\sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } | a _ { i } | \leq \sqrt { \dim \mathcal { C } } .
$$

Set $\begin{array} { r } { \tilde { c } = c - \frac { \bar { \eta } } { 1 - \bar { \eta } } \sum | a _ { i } | } \end{array}$ and $\begin{array} { r } { \tilde { a } _ { i } = \frac { a _ { i } } { 1 - \bar { \eta } } } \end{array}$ . Since $\textstyle 1 - { \bar { \eta } } > { \frac { 1 } { 2 } }$ , we have $\begin{array} { r } { \frac { \bar { \eta } } { 1 - \bar { \eta } } < \frac { \eta } { \sqrt { \mathrm { d i m } \mathcal { C } } } } \end{array}$ , and combined with $\Sigma \left| a _ { i } \right| \leq { \sqrt { \dim { \mathcal { C } } } }$ and $c \geq \eta$ , this gives $\tilde { c } > c - \eta \ge 0$ . We now rewrite d by introducing the substitution $a _ { i } = ( 1 - \bar { \eta } ) \tilde { a } _ { i }$ and isolating the u-components:

$$
\begin{array} { l } { { \displaystyle d = c \cdot u + \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } \left| a _ { i } \right| \mathrm { s i g n } ( a _ { i } ) \cdot w _ { i } } } \\ { { \displaystyle \quad = \tilde { c } \cdot u + \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } \left| \tilde { a } _ { i } \right| \Big ( \bar { \eta } u + ( 1 - \bar { \eta } ) \mathrm { s i g n } ( a _ { i } ) \cdot w _ { i } \Big ) } . } \end{array}
$$

At a smooth boundary point the third branch sets $U = \{ u \}$ and $T = \{ \pm w _ { 1 } , \dots , \pm w _ { \mathrm { d i m } } c _ { - 1 } \}$ At $\alpha = 0$ the output (excluding exploration) consists of the untilted vector u together with the normalized tilts of the directions in $T ,$ , and therefore positively spans the same cone as

$$
\{ u , \pm ( 1 - \bar { \eta } ) w _ { 1 } + \bar { \eta } u , . . . , \pm ( 1 - \bar { \eta } ) w _ { \mathrm { d i m } } c _ { - 1 } + \bar { \eta } u \} .
$$

Each term in the decomposition of d is a nonnegative multiple of an element of this set, so $d \in \mathrm { p o s } \left( \tilde { V } \setminus \left\{ v _ { \mathrm { e x } } \right\} \right)$

Case 3: $0 < \rho ( \bar { x } ) \le \rho _ { 0 }$ . The point is interior, so $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } ) = D _ { \mathcal { C } } ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ by Corollary 3.1(a). The third branch of Algorithm 2 produces $U = \left\{ \pm u , \pm w _ { 1 } , \dots , \pm w _ { \mathrm { d i m } } c _ { - 1 } \right\}$ which spans af $( { \mathcal { C } } - { \bar { x } } )$ . At $\alpha = 0$ the untilted output includes $U .$ , and the claim follows.

Since $d \in D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ was arbitrary, the result follows in all cases.

Theorem 7.2 extends Lemma 7.2 to $\alpha > 0$ , the regime in which Theorem 7.1 applies. Example 7.1 shows that nonzero noise is necessary for that almost-sure avoidance. For suficiently small α, the directions positively span $D _ { \mathcal { C } } ( \bar { x } )$ at every interior point, independently of $\rho _ { 0 }$ , and the relaxed cone $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ at smooth boundary points.

Theorem 7.2. Let ${ \bar { x } } \in { \mathcal { C } }$ and $\eta \in ( 0 , 1 ]$ . Denote by $\tilde { V } _ { \bar { \alpha } }$ the set of directions returned by Algorithm 2 for x¯, with this η and $\alpha = \bar { \alpha } ,$ where $v _ { \mathrm { e x } } ^ { \bar { \alpha } } \in \tilde { V } _ { \bar { \alpha } }$ is the exploration direction.

1. $I f \bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ , then there exists $\tilde { \alpha } > 0$ such that for every $\bar { \alpha } \in ( 0 , \tilde { \alpha } )$

$$
D c ( \bar { x } ) \subseteq \mathrm { p o s } \left( \tilde { V } _ { \bar { \alpha } } \setminus \left\{ v _ { \mathrm { e x } } ^ { \bar { \alpha } } \right\} \right) .
$$

2. $I f \bar { x } \in \mathrm { r b } \left( \mathcal { C } \right)$ and C is smooth at ${ \bar { x } } ,$ then for every $\beta \in ( 1 , 1 / \eta )$ there exists $\tilde { \alpha } > 0$ such that for every $\bar { \alpha } \in ( 0 , \tilde { \alpha } )$ ，

$$
D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) \subseteq \mathrm { p o s } \left( \tilde { V } _ { \bar { \alpha } } \backslash \left\{ v _ { \mathrm { e x } } ^ { \bar { \alpha } } \right\} \right) .
$$

To prove Theorem 7.2, we first prove two auxiliary lemmas, Lemma 7.3 and Lemma 7.4, which are used to prove cases 1 and 2 of Theorem 7.2, respectively.

Lemma 7.3 (Interior gap). Let $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ . Denote by $\tilde { V } _ { 0 }$ the set of directions returned by Algorithm ${ \mathcal { Q } } f o r { \bar { x } }$ , with $\alpha = 0$ , where $v _ { \mathrm { e x } } ^ { 0 } \in \tilde { V }$ is the exploration direction. Then there exists a compact set $S \subseteq D c ( \bar { x } )$ and a polytope $Q \subseteq \mathrm { p o s } \left( \tilde { V } _ { 0 } \setminus \{ v _ { \mathrm { e x } } ^ { 0 } \} \right)$ with the following properties:

1. $S \subseteq \mathrm { r i } \left( \mathrm { Q } \right)$ and dist $( S , \mathrm { r b } \left( \mathrm { Q } \right) ) > 0$

2. For every $d \in D c ( \bar { x } )$ there exists $\lambda > 0$ such that $\lambda d \in S$

Proof. We construct the objects S and Q claimed by the lemma, and verify both properties.

Define the non-exploration directions $\bar { V } _ { 0 } : = \tilde { V } _ { 0 } \backslash \{ v _ { \mathrm { e x } } ^ { 0 } \}$ , and let $W = \{ w _ { 1 } , \dots , w _ { \mathrm { d i m } } { c } \}$ be the orthonormal basis of af $( { \mathcal { C } } - { \bar { x } } )$ generated by Algorithm 2. Let $U = \{ w _ { 1 } , - w _ { 1 } , \ldots , w _ { \mathrm { d i m } } c , - w _ { \mathrm { d i m } } c \}$ be the untilted signed basis, so that $U \subseteq { \bar { V } } _ { 0 }$ , with equality on the first branch. Define the polytope

$$
Q : = \mathrm { c o n v } \left\{ \sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } } \theta _ { i } w _ { i } \biggm | \theta _ { i } \in \{ - 1 , 1 \} \right\} ,
$$

and the compact set $S : = D c ( \bar { x } ) \cap B [ 0 , 0 . 5 ]$ . We claim that $Q \subseteq \mathrm { p o s } \left( U \right) \subseteq \mathrm { p o s } \left( \bar { V } _ { 0 } \right)$ and $S \subseteq \mathrm { r i } \left( \mathrm { Q } \right)$ , which together establish the setup for both properties.

For the first claim, each vertex $\Sigma _ { i = 1 } ^ { \dim C } \theta _ { i } w _ { i }$ of Q is a nonnegative combination of elements of U, so $Q \subseteq \mathrm { p o s } \left( U \right) \subseteq \mathrm { p o s } \left( \bar { V } _ { 0 } \right)$ . For the second, define the linear isometry

$$
\Phi ( z ) : = \sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } } z _ { i } w _ { i } ,
$$

from $\mathbb { R } ^ { \mathrm { d i m } \it C }$ onto af $( { \mathcal { C } } - { \bar { x } } )$ . Since $B _ { \infty } [ 0 , 1 ]$ is the convex hull of the $2 ^ { \dim { \mathcal { C } } }$ sign vectors in $\{ - 1 , 1 \} ^ { \dim \mathcal { C } }$ , and Φ is linear (hence preserves convex combinations), we have

$$
\Phi ( B _ { \infty } [ 0 , 1 ] ) = \operatorname { c o n v } \left\{ \Phi ( \theta ) \ \Big | \ \theta \in \{ - 1 , 1 \} ^ { \mathrm { d i m } \mathscr { C } } \right\} = \operatorname { c o n v } \left\{ \left. \sum _ { i = 1 } ^ { \mathrm { d i m } \mathscr { C } } \theta _ { i } w _ { i } \ \right| \theta _ { i } \in \{ - 1 , 1 \} \right\} = Q .
$$

As Φ is an isometry, $\Phi ( B _ { 2 } [ 0 , 1 ] ) = \mathcal { B } [ 0 , 1 ] \cap \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ . The norm inequality ${ \cal B } _ { 2 } [ 0 , 1 ] \subseteq { \cal B } _ { \infty } [ 0 , 1 ]$ then gives

$$
\begin{array} { r } { \mathcal { B } [ 0 , 1 ] \cap \operatorname { a f f } \left( \mathcal { C } - \bar { x } \right) = \Phi ( \mathcal { B } _ { 2 } [ 0 , 1 ] ) \subseteq \Phi ( \mathcal { B } _ { \infty } [ 0 , 1 ] ) = Q . } \end{array}\tag{7.1}
$$

Since $S = B [ 0 , 0 . 5 ] \cap$ af $( { \mathcal { C } } - { \bar { x } } ) \subseteq { \frac { 1 } { 2 } } Q$ , and Q is a full-dimensional polytope in af $( { \mathcal { C } } - { \bar { x } } )$ containing the origin in its interior, we have ${ \frac { 1 } { 2 } } Q \subseteq \operatorname { r i } \left( \operatorname { Q } \right)$ , and hence $S \subseteq \mathrm { r i } \left( \mathrm { Q } \right)$

Since S is compact, rb (Q) is closed, and $S \subseteq \mathrm { r i } \left( \mathrm { Q } \right)$ implies S ∩ rb $( \mathrm { Q } ) = \emptyset$ , it follows that dist(S, rb $\mathrm { ( Q ) ) > 0 [ 3 1 }$ , Section 9.5, Problem 72].

For any d $\in D _ { \mathcal { C } } ( \bar { x } )$ with d $\neq 0 ,$ , set $\lambda = 0 . 5 / \| d \| _ { 2 } > 0$ . Then $\lambda d \in D _ { \mathcal { C } } ( \bar { x } )$ (since $D _ { \mathcal { C } } ( \bar { x } )$ is a cone) and $\| \lambda d \| _ { 2 } = 0 . 5$ , so $\lambda d \in S$ . The case $d = 0$ is immediate. ✷

Remark 7.2 (Interior gap at every interior point). Let $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ . By Corollary $\mathrm { 3 . 1 ( a ) }$ ， $D c ( \bar { x } ) = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right)$ , so ri $( \mathrm { D } \boldsymbol { c } ( \bar { \mathbf { x } } ) ) = D \boldsymbol { c } ( \bar { \mathbf { x } } )$ . On either interior branch, U is a signed orthonormal basis of af $( { \mathcal { C } } - { \bar { x } } )$ : on the first branch $T = \emptyset$ and $U = \{ w _ { 1 } , - w _ { 1 } , \ldots , w _ { \mathrm { d i m } } c , - w _ { \mathrm { d i m } } c \}$ ; on the third branch one of the dim C basis vectors is the inward unit normal $u ,$ and

$$
T = \{ w _ { 1 } , - w _ { 1 } , \hdots , w _ { \dim \mathcal { L } - 1 } , - w _ { \dim \mathcal { L } - 1 } \} \subseteq U = \{ u , - u , w _ { 1 } , - w _ { 1 } , \hdots , w _ { \dim \mathcal { L } - 1 } , - w _ { \dim \mathcal { L } - 1 } \} .
$$

The sets S and $Q$ in Lemma 7.3 depend only on this signed basis, so they serve on both interior branches, and $Q \subseteq \operatorname { p o s } \left( U \right)$ . On the third branch at $\alpha = 0$ the construction also returns the tilted images of $T _ { i }$ hence pos (U) is contained in the positive span of the $\alpha = 0$ non-exploration output; on the first branch the two coincide.

Lemma 7.4 (Boundary gap). Let $\bar { x } \in \mathrm { r b } \left( \mathcal { C } \right)$ with C smooth at x¯, let $\eta \in ( 0 , 1 )$ , and let $\beta \in ( 1 , 1 / \eta )$ . Denote by V<sup>˜</sup><sub>0</sub> the set of directions returned by Algorithm ${ \mathcal { Q } } f o r { \bar { x } } ,$ with $\alpha = 0$ where $v _ { \mathrm { e x } } ^ { 0 } \in \tilde { V } _ { 0 }$ is the exploration direction. Then there exist a compact set $S \subseteq D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ and a polytope $P \subseteq \mathrm { p o s } \left( \tilde { V } _ { 0 } \setminus \{ v _ { \mathrm { e x } } ^ { 0 } \} \right)$ with the following properties:

1. $S \subseteq { \mathrm { r i } } \left( \mathrm { P } \right)$ and dist $( S , \mathrm { r b } \left( \mathrm { P } \right) ) > 0$

2. For every $d \in D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ there exists $\lambda > 0$ such that $\lambda d \in S$

Proof. Denote the non-exploration directions generated by Algorithm 2 by

$$
\bar { V } _ { 0 } : = \tilde { V } _ { 0 } \backslash \left\{ v _ { \mathrm { e x } } ^ { 0 } \right\} .
$$

By Corollary 3.1(b),

$$
{ \cal D } _ { \mathcal { C } } ^ { \eta } ( \bar { x } ) = \left\{ 0 \neq d \in \mathbb { R } ^ { n } \ | \ \left. \frac { d } { \| d \| } , u \right. \geq \eta \right\} \cap \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right) ,\tag{7.2}
$$

for a nonzero $u \in$ af (C − x¯) such that

$$
\operatorname { c l } \left( \operatorname { D } _ { \mathcal { C } } ( \bar { \operatorname { x } } ) \right) = \left\{ d \in \mathbb { R } ^ { n } \mid \langle d , u \rangle \geq 0 \right\} \cap \operatorname { a f f } \left( { \mathcal { C } } - { \bar { x } } \right) .\tag{7.3}
$$

By the same arguments,

$$
{ \cal D } _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) = \left\{ 0 \neq d \in \mathbb { R } ^ { n } \ | \ \left. \frac { d } { \| d \| } , u \right. \geq \beta \eta \right\} \cap \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right) .\tag{7.4}
$$

Since $\beta > 1$ , the containment $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) \subseteq D _ { \mathcal { C } } ^ { \eta } ( \bar { x } )$ is strict, and therefore

$$
D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) \subseteq \mathrm { r i } \left( \mathrm { D } _ { \mathcal { C } } ^ { \eta } ( \bar { \mathrm { x } } ) \right) .
$$

By Lemma 7.2, $D _ { \mathcal { C } } ^ { \eta } ( \bar { x } ) \subseteq \mathrm { p o s } \left( \bar { V } _ { 0 } \right)$ , and since both sets have the same afine hull,

$$
D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) \subseteq \mathrm { r i } \left( \mathrm { p o s } \left( \bar { \mathrm { V } } _ { 0 } \right) \right) .\tag{7.5}
$$

Let $u \in \mathbb { R } ^ { n }$ be the unit vector from (7.3), and consider the closed halfspace

$$
H _ { u , - 1 } : = \left\{ x \in \mathbb { R } ^ { n } | \langle x , u \rangle \leq 1 \right\} .
$$

Set $\bar { \eta } = \eta / ( 2 \sqrt { \mathrm { d i m } \mathcal { C } } )$ , as in Algorithm 2. At a smooth boundary point the third branch sets $U = \{ u \}$ and $T = \{ \pm w _ { 1 } , \ldots , \pm w _ { \mathrm { d i m } } c _ { - 1 } \}$ , where $\{ u , w _ { 1 } , \ldots , w _ { \dim { \mathcal { C } } - 1 } \}$ is an orthonormal basis of af $( { \mathcal { C } } - { \bar { x } } )$ $\mathrm { A t } ~ \alpha = 0$ , each vector in $\bar { V } _ { 0 }$ is obtained by normalizing u or $\bar { \eta } u + ( 1 - \bar { \eta } ) v$ for some $v \in T$ . In particular $\langle v , u \rangle > 0$ for every $v \in \bar { V } _ { 0 }$ . We show that the set pos $\left( \hat { V } _ { 0 } \right) \cap H _ { u , - 1 }$ is a polytope with vertex description

$$
P : = \mathrm { p o s } \left( \bar { V } _ { 0 } \right) \cap H _ { u , - 1 } = \mathrm { c o n v } \left( \left\{ 0 \right\} \cup \left\{ \frac { 1 } { \left. v , u \right. } v \bigg | v \in \bar { V } _ { 0 } \right\} \right) .\tag{7.6}
$$

To see this, we consider ext $( P )$ , the extreme points of $P .$ Let $w \in P .$ . Since $w \in \mathrm { p o s } \left( \bar { V } _ { 0 } \right)$ , there exist $\lambda _ { 1 } , \dots , \lambda _ { s } \in \mathbb { R } _ { + } , v _ { 1 } , \dots , v _ { s } \in \bar { V } _ { 0 }$ such that $\begin{array} { r } { w = \sum _ { i = 1 } ^ { s } \lambda _ { i } v _ { i } } \end{array}$ . If $v _ { i } = u .$ , then $\langle v _ { i } , u \rangle = 1$ . If $v _ { i } =$ normalize $( \bar { \eta } u + ( 1 - \bar { \eta } ) v )$ for some $v \in T$ , then

$$
\langle v _ { i } , u \rangle = \frac { \bar { \eta } } { \sqrt { \bar { \eta } ^ { 2 } + ( 1 - \bar { \eta } ) ^ { 2 } } } \in ( \bar { \eta } , 1 ) .
$$

In both cases $v _ { i } \in H _ { u , - 1 }$ , so $v _ { i } \in P$ for all i. Therefore, if w $\notin \mathbb { R } _ { + } v _ { i }$ for some $i ,$ then $w$ is not an extreme point of $P$ . If $w \in \mathbb { R } _ { + } v _ { i }$ for some i but w $\begin{array} { r } { \notin \left\{ 0 , \frac { 1 } { \left. v _ { i } , u \right. } v _ { i } \right\} } \end{array}$ , then w lies in the convex hull of these two points and is again not extreme. Thus,

$$
\mathrm { e x t } ( P ) \subseteq \{ 0 \} \cup \{ { \frac { 1 } {  v , u  } } v  v \in { \bar { V } } _ { 0 } \} .
$$

To apply the Krein–Milman theorem, we verify that P is nonempty, compact, and convex. Nonemptiness is trivial since $0 \in P$ . Convexity holds as $P$ is an intersection of two convex sets. Closedness follows as $P$ is an intersection of two closed sets, where closedness of pos $\left( \hat V _ { 0 } \right)$ follows from [29, Theorem 19.1]. For boundedness, suppose for contradiction that there exists a sequence $\{ w _ { i } \} \subseteq P$ with $\| w _ { i } \|  \infty$ . Writing $\begin{array} { r } { w _ { i } = \sum _ { j } \lambda _ { j } ^ { i } v _ { j } } \end{array}$ with $\lambda _ { j } ^ { i } \geq 0$ , we get $\textstyle \sum _ { j } \lambda _ { j } ^ { i } \to \infty$ and hence

$$
\langle w _ { i } , u \rangle = \left. \sum _ { j } \lambda _ { j } ^ { i } v _ { j } , u \right. \ge \left( \operatorname* { m i n } _ { j \in [ s ] } \langle v _ { j } , u \rangle \right) \sum _ { j } \lambda _ { j } ^ { i } \to \infty ,
$$

contradicting $w _ { i } \in H _ { u , - 1 }$ . Thus $P$ is nonempty, compact, and convex, so by [31, The Krein-Milman Lemma], ext (P) is nonempty, and by [31, The Krein-Milman Theorem], $P = \mathrm { c o n v } ( \mathrm { e x t } \left( P \right) )$ . It is possible that $\{ 0 \} \cup \{ { \textstyle { \frac { 1 } {  v , u  } } } v | v \in \bar { V } _ { 0 } \} \backslash$ ext (P) is nonempty, but since every point in this set belongs to $P ,$ , adding them does not change the convex hull. Therefore, (7.6) holds.

Define

$$
S : = D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) \cap \left\{ d \in \mathbb { R } ^ { n } | 0 . 2 5 \leq \| d \| \leq 0 . 5 \right\} .
$$

We first show $S \subseteq { \mathrm { r i } } \left( \mathrm { P } \right)$ . Since for any d with $\| d \| \leq 0 . 5$ we have $\langle d , u \rangle \leq 0 . 5 < 1$ , it follows that $\{ d : \Vert d \Vert \leq 0 . 5 \} \subseteq \mathrm { { r i } } \left( \mathrm { { H } } _ { \mathrm { { u , - 1 } } } \right)$ , and therefore by (7.5),

$$
S \subseteq \mathrm { r i } \left( \mathrm { p o s } \left( \bar { \mathrm { V } } _ { 0 } \right) \right) \cap \mathrm { r i } \left( \mathrm { H } _ { \mathrm { u } , - 1 } \right) = \mathrm { r i } \left( \mathrm { P } \right) .
$$

It remains to show dis $\bar { \iota } ( S , \mathrm { r b } \left( \mathrm { P } \right) ) > 0$ . The set S is compact: it is bounded by definition, and closed since $f ( d ) = \langle d / \| d \| , u \rangle$ is continuous away from 0, making $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ closed on $\{ \| d \| \ge 0 . 2 5 \}$ . Since S is compact, rb (P) is closed, and $S \cap$ rb $( \mathrm { P } ) = \emptyset$ , it follows that dist $\left( S , \mathrm { r b } \left( \mathrm { P } \right) \right) > 0 \ \left[ 3 1 \right.$ Section 9.5, Problem 72].

Let $d \in D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ . By Definition 3.1, d $\neq 0$ . Set $\lambda : = 0 . 5 / \| d \| > 0$ . Since $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ is a cone (for any $\mu > 0$ , the direction µd satisfies $\langle \mu d / \| \mu d \| , u \rangle = \langle d / \| d \| , u \rangle \geq \beta \eta$ and µd ∈ af $( { \mathcal { C } } - { \bar { x } } )$ so $\mu d \in D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) )$ , we have $\lambda d \in D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ . Moreover, $\lVert \lambda d \rVert = 0 . 5 \in [ 0 . 2 5 , 0 . 5 ]$ , so $\lambda d \in S$ ✷

The proof of Theorem 7.2 combines the gap lemmas above with two elementary facts: a quantitative Hausdorf bound for convex hulls indexed by a common set, and a criterion under which Hausdorf proximity forces containment. We record them here, as both are applied with constants that do not depend on the shared random direction $\zeta .$

Lemma 7.5 (Hausdorf bound for matched convex hulls). Let $N \in  { \mathbb { N } }$ and let $\{ a _ { j } \} _ { j \in [ N ] } , \{ b _ { j } \} _ { j \in [ N ] } \subseteq$ $\mathbb { R } ^ { n }$ . Then

$$
d _ { H } \Big ( \mathrm { c o n v } \{ a _ { j } \} _ { j \in [ N ] } , \ \mathrm { c o n v } \{ b _ { j } \} _ { j \in [ N ] } \Big ) \leq \operatorname* { m a x } _ { j \in [ N ] } \| a _ { j } - b _ { j } \| .
$$

Proof. Set $\varepsilon : = \mathrm { m a x } _ { j \in [ N ] } \| a _ { j } - b _ { j } \|$ . Let $x \in$ conv $\{ a _ { j } \} _ { j \in [ N ] }$ , so $\begin{array} { r } { x = \sum _ { j \in [ N ] } \lambda _ { j } a _ { j } } \end{array}$ for some $\lambda \in \Delta _ { N }$ . The point $\begin{array} { r } { y : = \sum _ { j \in [ N ] } \lambda _ { j } b _ { j } } \end{array}$ lies in conv $\{ b _ { j } \} _ { j \in [ N ] }$ and satisfies

$$
\| x - y \| = \left\| \sum _ { j \in [ N ] } \lambda _ { j } ( a _ { j } - b _ { j } ) \right\| \leq \sum _ { j \in [ N ] } \lambda _ { j } \| a _ { j } - b _ { j } \| \leq \varepsilon .
$$

Hence sup inf $\{ b _ { j } \} ^ { \| x - y \| \leq \varepsilon }$ . Exchanging the roles of $\{ a _ { j } \}$ and $\{ b _ { j } \}$ yields the $x \in \mathrm { c o n } \mathbf { \bar { v } } \{ a _ { j } \}$ y∈ conv reverse bound, so $d _ { H } \leq \varepsilon$ ✷

Lemma 7.6 (Containment under Hausdorf proximity). Let $K \subseteq \mathbb { R } ^ { n }$ be a nonempty compact convex set, and let $S \subseteq \mathrm { r i } \left( \mathrm { K } \right)$ satisfy $\rho : = \mathrm { d i s t } ( S , \mathrm { r b } \left( \mathrm { K } \right) ) > 0$ . Then for every compact convex set $K ^ { \prime } \subseteq \mathrm { a f f } \left( K \right)$ with $d _ { H } ( K , K ^ { \prime } ) < \rho _ { ; }$ it holds that $S \subseteq K ^ { \prime }$

Proof. Fix $s \in S$ and suppose, for contradiction, that $s \notin K ^ { \prime }$ . Since $K ^ { \prime }$ is a nonempty closed convex subset of af $( K )$ and $s \in \mathrm { a f f } \left( K \right)$ , separation within af $( K )$ yields a unit vector n in the direction subspace of af (K) with

$$
\langle n , s \rangle > h _ { K ^ { \prime } } ( n ) : = \operatorname* { s u p } _ { y \in K ^ { \prime } } \langle n , y \rangle .
$$

Because dist $( s , \mathrm { r b } \left( \mathrm { K } \right) ) \ge \rho ,$ the closed ball of radius $\rho$ around s within af $( K )$ is contained in $K _ { i }$ ; in particular $s + \rho n \in K$ , whence $\langle n , s \rangle + \rho \leq h _ { K } ( n )$ . For unit n one has $| h _ { K } ( n ) - h _ { K ^ { \prime } } ( n ) | \leq$ $d _ { H } ( K , K ^ { \prime } )$ , so

$$
\begin{array} { r } { \langle n , s \rangle + \rho \leq h _ { K } ( n ) \leq h _ { K ^ { \prime } } ( n ) + d _ { H } ( K , K ^ { \prime } ) < h _ { K ^ { \prime } } ( n ) + \rho , } \end{array}
$$

giving $\langle n , s \rangle < h _ { K ^ { \prime } } ( n )$ , a contradiction. Hence $s \in K ^ { \prime }$ , and since $s \in S$ was arbitrary, $S \subseteq K ^ { \prime }$ ✷

Proof of Theorem 7.2. Denote the non-exploration directions generated by Algorithm 2 by

$$
\bar { V } _ { \bar { \alpha } } : = \tilde { V } _ { \bar { \alpha } } \setminus \left\{ v _ { \mathrm { e x } } ^ { \bar { \alpha } } \right\} .
$$

Case 1: $\bar { x } \in \mathrm { \Gamma } \mathrm { r i } \left( \mathcal { C } \right)$ . Let $U = \{ w _ { 1 } , - w _ { 1 } , \ldots , w _ { \dim { \mathcal { C } } } , - w _ { \dim { \mathcal { C } } } \}$ be the untilted signed orthonormal basis of af $( { \mathcal { C } } - { \bar { x } } )$ produced by Algorithm 2 at x¯ (in the third branch, one of the dim C basis vectors is the inward unit normal u). By Lemma 7.3 and Remark 7.2, there exist a compact set $S \subseteq D c ( \bar { x } )$ and a polytope $Q \subseteq \operatorname { p o s } \left( U \right)$ such that $S \subseteq \mathrm { r i } \left( \mathrm { Q } \right)$ , dist $( S , \mathrm { r b } \left( \mathrm { Q } \right) ) > 0$ and every direction in $D _ { \cal C } ( \bar { x } )$ can be scaled into S.

We now construct a family of polytopes that track the noise level α¯. By the proof of Lemma 7.3, the polytope $Q$ is the convex hull of the $2 ^ { \dim { \mathcal { C } } }$ vertices

$$
w _ { \theta } = \sum _ { i = 1 } ^ { \mathrm { d i m } \mathcal { C } } \theta _ { i } w _ { i } , \qquad \theta \in \{ - 1 , 1 \} ^ { \mathrm { d i m } \mathcal { C } } ,
$$

corresponding to all sign combinations over the orthonormal basis.

Let $\{ \tilde { v } _ { 1 } ^ { \bar { \alpha } } , \ldots , \tilde { v } _ { 2 \mathrm { d i m } \mathcal { C } } ^ { \bar { \alpha } } \}$ be the unnormalized untilted combinations of Algorithm 2, enumerated so that $\tilde { v } _ { 2 i - 1 } ^ { \bar { \alpha } }$ and $\tilde { v } _ { 2 i } ^ { \bar { \alpha } }$ are the perturbations of $w _ { i }$ and $- w _ { i }$

$$
\tilde { v } _ { j } ^ { \bar { \alpha } } = \bar { \alpha } \bar { \eta } \zeta + \left( 1 - \bar { \alpha } \bar { \eta } \right) v _ { j } , \qquad j \in [ 2 \dim \mathcal { C } ] ,
$$

where $v _ { j }$ denotes the corresponding element of $U$ and $\zeta$ is the shared random direction. For each $j ,$ , Algorithm 2 includes normalize $( \tilde { v } _ { j } ^ { \bar { \alpha } } )$ in $\bar { V } _ { \bar { \alpha } }$ , so $\tilde { v } _ { j } ^ { \bar { \alpha } } \in \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$ . If $T$ is nonempty the algorithm also returns normalized tilts of $T ;$ those are not used below. Note that $\tilde { v } _ { j } ^ { \bar { \alpha } }  v _ { j }$ as $\bar { \alpha }  0 ^ { + }$ for each $j \in [ 2 \dim \mathcal { C } ]$

For each vertex $w _ { \theta }$ of $Q$ with sign vector $\theta ,$ define the perturbed vertex

$$
w _ { \theta } ^ { \bar { \alpha } } : = \sum _ { i = 1 } ^ { \dim \mathcal { C } } \left\{ \begin{array} { l l } { \tilde { v } _ { 2 i - 1 } ^ { \bar { \alpha } } } & { \mathrm { i f ~ } \theta _ { i } = 1 , } \\ { \tilde { v } _ { 2 i } ^ { \bar { \alpha } } } & { \mathrm { i f ~ } \theta _ { i } = - 1 . } \end{array} \right.
$$

Each summand lies in pos $\left( \hat V _ { \hat \alpha } \right)$ , so $w _ { \theta } ^ { \bar { \alpha } } \in \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$ . Define

$$
Q _ { \bar { \alpha } } : = \mathrm { c o n v } \Big \{ w _ { \theta } ^ { \bar { \alpha } } \Big | \theta \in \{ - 1 , 1 \} ^ { \mathrm { d i m } \mathcal { C } } \Big \} .
$$

Then $Q _ { \bar { \alpha } } \subseteq \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$ , since pos $\left( \hat V _ { \hat \alpha } \right)$ is convex. We bound the vertex displacement uniformly in $\zeta .$ . Let $Z : = { \mathrm { c o n v } } ( U \cup T )$ be the compact set in which the shared direction ζ ranges. The vector ζ is a convex combination of $U \cup T$ , so $\zeta \in Z$ . Every vector in $U \cup T$ has unit norm and $B [ 0 , 1 ]$ is convex, hence $Z \subseteq B [ 0 , 1 ]$ and $\| \zeta \| \le 1$ for every $\zeta \in Z$ . The set $Z$ does not depend on α¯. For each $j \in [ 2$ dim C], writing $v _ { j } \in U$ for the corresponding noiseless direction,

$$
\begin{array} { r } { \tilde { v } _ { j } ^ { \bar { \alpha } } - v _ { j } = \bar { \alpha } \bar { \eta } ( \zeta - v _ { j } ) , \qquad \mathrm { s o } \qquad \left\| \tilde { v } _ { j } ^ { \bar { \alpha } } - v _ { j } \right\| = \bar { \alpha } \bar { \eta } \| \zeta - v _ { j } \| \leq 2 \bar { \alpha } \bar { \eta } , } \end{array}
$$

where the last inequality uses $\| \zeta \| \le 1$ and $\| v _ { j } \| = 1 \ \big ( \mathrm { s o } \ \| \zeta - v _ { j } \| \le 2 \big )$ , and holds for every $\zeta \in Z$ . Each vertex $w _ { \theta } ^ { \bar { \alpha } }$ is a sum of dim C such directions and w<sub>θ</sub> the sum of the matching $v _ { j }$ so by the triangle inequality

$$
\left\| w _ { \theta } ^ { \bar { \alpha } } - w _ { \theta } \right\| \leq 2 \dim { \mathcal { C } } \bar { \alpha } \bar { \eta } \qquad \mathrm { f o r ~ e v e r y ~ } \theta \in \{ - 1 , 1 \} ^ { \mathrm { d i m } { \mathcal { C } } } \mathrm { ~ a n d ~ e v e r y ~ } \zeta \in Z .
$$

As Q and $Q _ { \bar { \alpha } }$ are the convex hulls of the matched vertex families $\{ w _ { \theta } \} _ { \theta }$ and $\{ w _ { \theta } ^ { \bar { \alpha } } \} _ { \theta }$ , Lemma 7.5 gives

$$
d _ { H } ( Q _ { \bar { \alpha } } , Q ) \leq 2 \mathrm { d i m } { \mathcal { C } } \bar { \alpha } \bar { \eta } \qquad \mathrm { f o r ~ e v e r y ~ } \zeta \in Z .\tag{7.7}
$$

Each perturbed vertex $w _ { \theta } ^ { \bar { \alpha } }$ is a sum of vectors $\tilde { v } _ { j } ^ { \bar { \alpha } } = \bar { \alpha } \bar { \eta } \zeta + ( 1 - \bar { \alpha } \bar { \eta } ) v _ { j }$ , which are afine combinations of $\zeta , v _ { j } \in \operatorname { a f f } \left( \mathcal { C } - \bar { x } \right)$ and hence lie in the subspace af $( { \mathcal { C } } - { \bar { x } } ) = \operatorname { a f f } \left( Q \right)$ ; therefore $Q _ { \bar { \alpha } } \subseteq \mathrm { a f f } \left( Q \right)$ , so Lemma 7.6 applies.

Set α˜ := min{ $1 , \ \mathrm { d i s t } ( S , \mathrm { r b } ( \mathrm { Q } ) ) / ( 2 \mathrm { d i m } { \mathcal { C } } \bar { \eta } ) \} > 0 .$ . This depends only on dim ${ \mathcal { C } } , { \bar { \eta } } .$ , and the noiseless gap dist(S, rb (Q)) supplied by Lemma 7.3, and is therefore independent of ζ (the cap by 1 keeps α¯ within the admissible noise range). For every $\bar { \alpha } \in ( 0 , \tilde { \alpha } )$ and every $\zeta \in Z$ (7.7) yields $d _ { H } ( Q _ { \bar { \alpha } } , Q ) < \mathrm { d i s t } ( S , \mathrm { r b } \left( \mathrm { Q } \right) )$ , so $S \subseteq Q _ { \bar { \alpha } }$ by Lemma 7.6. For any $d \in D c ( \bar { x } )$ , there exists $\lambda > 0$ with $\lambda d \in S \subseteq Q _ { \bar { \alpha } } \subseteq \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$ , so $d \in \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$ since pos $\left( \hat V _ { \hat \alpha } \right)$ is a cone. Because α˜ does not depend on ζ, the conclusion holds for every realization of the shared direction.

Case 2: $\bar { x } \in \mathrm { r b } ( \mathcal { C } ) , \mathcal { C }$ smooth at x¯. By Lemma 7.4, there exist a compact set $S \subseteq D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ and a polytope $P \subseteq \mathrm { p o s } \left( \bar { V } _ { 0 } \right)$ such that $S \subseteq { \mathrm { r i } } \left( \mathrm { P } \right)$ , dist $( S , \mathrm { r b } \left( \mathrm { P } \right) ) > 0$ , and every direction in $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ can be scaled into S.

Let $u \in \mathrm { a f f } \left( { \mathcal { C } } - { \bar { x } } \right)$ be the unit vector from (7.3). By the proof of Lemma 7.4 (specifically (7.6)), the polytope P satisfies

$$
P = \operatorname { c o n v } \left( \{ 0 \} \cup \left\{ { \frac { v } { \langle v , u \rangle } } \ \middle | \ v \in { \bar { V } } _ { 0 } \right\} \right) .
$$

For any a with $\langle a , u \rangle > 0$ , write $R _ { u } ( a ) : = a / \langle a , u \rangle$ . This map depends only on the positive ray through a: $R _ { u } ( \lambda a ) ~ = ~ R _ { u } ( a )$ for every $\lambda > 0$ At a smooth boundary point the third branch of Algorithm 2 sets $U = \left\{ u \right\}$ and $T = \{ \pm w _ { 1 } , \ldots , \pm w _ { \mathrm { d i m } } c _ { - 1 } \}$ , where $\{ u , w _ { 1 } , \ldots , w _ { \mathrm { d i m } } \mathcal { C } \mathrm { - } 1 \}$ is an orthonormal basis of af $( { \mathcal { C } } - { \bar { x } } )$ Write $m \ : = \ | U \cup T |$ and $V : = U \cup T = \{ u , w _ { 1 } , - w _ { 1 } , \ldots , w _ { \dim { \mathcal { C } } - 1 } , - w _ { \dim { \mathcal { C } } - 1 } \}$ . For each $\hat { v } _ { i } ~ \in ~ V .$ , write $v _ { i }$ for the corresponding unnormalized noiseless generator: u itself if $\hat { v } _ { i } \in U$ , and $\bar { \eta } u + \left( 1 - \bar { \eta } \right) \hat { v } _ { i }$ if $\hat { v } _ { i } \in T$ . These difer from the unit vectors in $\bar { V } _ { 0 }$ only by positive scaling, so

$$
P = \operatorname { c o n v } ( \{ 0 \} \cup \{ R _ { u } ( v _ { i } ) \mid i \in [ m ] \} ) .
$$

Moreover, $\| v _ { i } \| \leq 1$ and $\langle v _ { i } , u \rangle \geq \bar { \eta }$ for every i (the latter inner product equals 1 when $\hat { v } _ { i } \in U$ and $\bar { \eta }$ when $\hat { v } _ { i } \in T )$

For the noisy construction we work with the unnormalized combinations of Algorithm 2, enumerated so that $\tilde { v } _ { i } ^ { \bar { \alpha } }$ is the perturbation of v :

$$
\bullet \mathrm { F o r } \ \hat { v } _ { i } \in U : \quad \tilde { v } _ { i } ^ { \bar { \alpha } } = \bar { \alpha } \bar { \eta } \zeta + \left( 1 - \bar { \alpha } \bar { \eta } \right) \hat { v } _ { i } .
$$

$$
\bullet \mathrm { F o r } \ \hat { v } _ { i } \in T : \quad \tilde { v } _ { i } ^ { \bar { \alpha } } = ( 1 - \bar { \alpha } ) \bar { \eta } u + \bar { \alpha } \bar { \eta } \zeta + ( 1 - \bar { \eta } ) \hat { v } _ { i } .
$$

Let $Z : = \operatorname { c o n v } ( V )$ be the compact set in which the shared direction $\zeta$ ranges. Since every element of V has unit norm, $Z \subseteq B [ 0 , 1 ]$ ; in particular, $\| \zeta \| \le 1$ uniformly over $\zeta \in Z$ and $Z$ does not depend on α¯. In both branches above the displacement has the form $\tilde { v } _ { i } ^ { \bar { \alpha } } - v _ { i } = \bar { \alpha } \bar { \eta } \left( \zeta - c _ { i } \right)$ with $c _ { i } \in \{ \hat { v } _ { i } , u \}$ satisfying $\left\| \boldsymbol { c } _ { i } \right\| = 1$ , hence

$$
\left\| { \tilde { v } } _ { i } ^ { \bar { \alpha } } - v _ { i } \right\| = { \bar { \alpha } } { \bar { \eta } } \left\| \zeta - c _ { i } \right\| \leq 2 { \bar { \alpha } } { \bar { \eta } } \qquad \mathrm { f o r ~ e v e r y ~ } \zeta \in Z .\tag{7.8}
$$

Combining this estimate with $\langle v _ { i } , u \rangle \geq \bar { \eta }$ and $\lVert u \rVert = 1$ gives

$$
\begin{array} { r } { \langle \tilde { v } _ { i } ^ { \bar { \alpha } } , u \rangle \geq \langle v _ { i } , u \rangle - \left\| \tilde { v } _ { i } ^ { \bar { \alpha } } - v _ { i } \right\| \geq \bar { \eta } - 2 \bar { \alpha } \bar { \eta } = \bar { \eta } ( 1 - 2 \bar { \alpha } ) \geq \frac { \bar { \eta } } { 2 } } \end{array}\tag{7.9}
$$

for every $\textstyle { \bar { \alpha } } \leq { \frac { 1 } { 4 } }$ and every $\zeta \in Z ;$ in particular $\langle \tilde { v } _ { i } ^ { \bar { \alpha } } , u \rangle > 0$ . For such ${ \bar { \alpha } } ,$ define the perturbed polytope

$$
P _ { \bar { \alpha } } : = \mathrm { c o n v } \left( \{ 0 \} \cup \left\{ \frac { \tilde { v } _ { i } ^ { \bar { \alpha } } } { \langle \tilde { v } _ { i } ^ { \bar { \alpha } } , u \rangle } \left| i \in [ m ] \right. \right\} \right) .
$$

For each i, Algorithm 2 includes normalize $( \tilde { v } _ { i } ^ { \bar { \alpha } } )$ in $V _ { \bar { \alpha } }$ . Since $R _ { u } ( \tilde { v } _ { i } ^ { \bar { \alpha } } )$ lies on the same positive ray, convexity gives $P _ { \bar { \alpha } } \subseteq \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$

We now bound the vertex displacement uniformly in $\zeta \in Z$ . For any $a , b$ with $\langle a , u \rangle , \langle b , u \rangle >$ 0,

$$
\frac { a } { \left. a , u \right. } - \frac { b } { \left. b , u \right. } = \frac { a \left. b - a , u \right. + \left( a - b \right) \left. a , u \right. } { \left. a , u \right. \left. b , u \right. } , \qquad \mathrm { s o } \qquad \left\| \frac { a } { \left. a , u \right. } - \frac { b } { \left. b , u \right. } \right\| \leq \frac { 2 \left\| a \right\| } { \left. a , u \right. \left. b , u \right. } \left\| a - b \right\| ,
$$

using $| \langle \cdot , u \rangle | \leq \| \cdot \|$ . Apply this with $a \ = \ \tilde { v } _ { i } ^ { \bar { \alpha } }$ and $b \ = \ v _ { i }$ . Since $\left\| v _ { i } \right\| \le 1$ , for $\bar { \alpha } \leq \frac { 1 } { 4 }$ $\left\| \tilde { v } _ { i } ^ { \bar { \alpha } } \right\| \leq \left\| v _ { i } \right\| + 2 \bar { \alpha } \bar { \eta } \leq 2$ , and (7.9) gives $\begin{array} { r } { \langle \tilde { v } _ { i } ^ { \bar { \alpha } } , u \rangle \geq \frac { \bar { \eta } } { 2 } } \end{array}$ , while $\langle v _ { i } , u \rangle \geq \bar { \eta } ;$ combined with (7.8),

$$
\left\| { \frac { \tilde { v } _ { i } ^ { \bar { \alpha } } } { \langle \tilde { v } _ { i } ^ { \bar { \alpha } } , u \rangle } } - { \frac { v _ { i } } { \langle v _ { i } , u \rangle } } \right\| \leq { \frac { 2 \cdot 2 } { ( { \bar { \eta } } / 2 ) { \bar { \eta } } } } \cdot 2 { \bar { \alpha } } { \bar { \eta } } = { \frac { 1 6 { \bar { \alpha } } } { \bar { \eta } } } \qquad \mathrm { f o r ~ e v e r y ~ } \zeta \in Z .
$$

The vertex 0 is common to P and $P _ { \bar { \alpha } }$ , so all matched vertices are within $1 6 \bar { \alpha } / \bar { \eta }$ of one another. By Lemma 7.5,

$$
d _ { H } ( P _ { \bar { \alpha } } , P ) \leq \frac { 1 6 \bar { \alpha } } { \bar { \eta } } \qquad \mathrm { f o r ~ e v e r y ~ } \zeta \in Z .\tag{7.10}
$$

Each vertex of $P _ { \bar { \alpha } }$ is either 0 or a positive multiple of $\tilde { v } _ { i } ^ { \bar { \alpha } }$ , and $\tilde { v } _ { i } ^ { \bar { \alpha } }$ is an afine combination of $\zeta , \hat { v } _ { i } , u \in \mathrm { a f f } \left( C - \bar { x } \right)$ ; hence $P _ { \bar { \alpha } } \subseteq \mathrm { a f f } \left( \mathcal C - \bar { x } \right) = \mathrm { a f f } \left( P \right)$ , so Lemma 7.6 applies.

Set $\tilde { \alpha } : = \operatorname* { m i n } \biggl \{ \textstyle { \frac { 1 } { 4 } } , \ \frac { \bar { \eta } \ \mathrm { d i s t } ( S , \mathrm { r b } \left( \mathrm { P } \right) ) } { 1 6 } \biggr \} > 0$ . This depends only on η¯ and the noiseless gap dist(S, rb (P)) supplied by Lemma 7.4, and is therefore independent of $\zeta .$ . For every $\bar { \alpha } \in ( 0 , \tilde { \alpha } )$ and every $\zeta \in Z , ( 7 . 1 0 )$ yields $d _ { H } ( P _ { \bar { \alpha } } , P ) < \mathrm { d i s t } ( S , \mathrm { r b } ( \mathrm { P } ) )$ , so $S \subseteq P _ { \bar { \alpha } }$ by Lemma 7.6. For any $d \in D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ , there exists $\lambda > 0$ with $\lambda d \in S \subseteq P _ { \bar { \alpha } } \subseteq \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$ , so $d \in \mathrm { p o s } \left( \bar { V } _ { \bar { \alpha } } \right)$ since pos $\left( \hat V _ { \hat \alpha } \right)$ is a cone. Because $\tilde { \alpha }$ does not depend on $\zeta ,$ the conclusion holds for every realization of the shared direction. ✷

Theorem 7.2 asserts that a positive noise threshold exists. We now compute the two gap distances on which it depends; this gives explicit thresholds in terms of $\eta , \beta$ and dim $\mathcal { C } .$ We first record an elementary fact used in the boundary computation.

Lemma 7.7 (inradius of a polyhedron). Let $\nu _ { 1 , \dots , \nu _ { N } } \in \mathbb { R } ^ { m } \setminus \{ 0 \} , \ b _ { 1 , \dots , b _ { N } } \in \mathbb { R }$ , and $C = \bigcap _ { i = 1 } ^ { N } \{ x \in \mathbb { R } ^ { m } : \langle \nu _ { i } , x \rangle \leq b _ { i } \}$ . If C is closed with nonempty interior, then for every $x \in i n t ( C )$ ，

$$
\mathrm { d i s t } \big ( x , \mathrm { r b } \left( \mathrm { C } \right) \big ) = \operatorname* { m i n } _ { i \in \left[ N \right] } \frac { b _ { i } - \langle \nu _ { i } , x \rangle } { \| \nu _ { i } \| } .
$$

Proof. For $x \in \mathbf { i n t } ( C )$ the distance to the boundary equals the largest inscribed-ball radius, dist(x, rb (C)) = sup $\{ r \ge 0 : B [ x , r ] \subseteq C \}$ . For each i, $B [ x , r ] \subseteq \{ \langle \nu _ { i } , \cdot \rangle \leq b _ { i } \}$ if $\left. \nu _ { i } , x \right. +$ $r \| \nu _ { i } \| \leq b _ { i }$ , i.e. if $r \leq ( b _ { i } - \langle \nu _ { i } , x \rangle ) / \| \nu _ { i } \|$ . Intersecting over i and taking the supremum over r gives the claim. ✷

Proposition 7.1 (interior gap value). In the setting of Lemma 7.3, dist $\begin{array} { r } { ( S , \operatorname { r b } \left( \mathrm { Q } \right) ) = \frac { 1 } { 2 } } \end{array}$

Proof. Let Φ $: \mathbb { R } ^ { \dim \mathcal { C } }  L , \Phi ( z ) = \sum _ { i = 1 } ^ { \dim \mathcal { C } } z _ { i } w _ { i } ;$ since $\{ w _ { i } \}$ is orthonormal, Φ is a linear isometry onto $L : =$ af $( { \mathcal { C } } - { \bar { x } } )$ . The cube $\begin{array} { r } { { \cal B } _ { \infty } [ 0 , 1 ] = \bar { \bf \Phi } [ - 1 , 1 ] ^ { \mathrm { d i m } \mathcal { C } } } \end{array}$ is the convex hull of $\{ - 1 , 1 \} ^ { \dim \mathcal { C } }$ , and Φ preserves convex hulls, so $Q = \Phi ( B _ { \infty } [ 0 , 1 ] )$ ; as Φ is a homeomorphism onto $L , \operatorname { r b } \left( \mathrm { Q } \right) = \Phi ( \{ z : \| z \| _ { \infty } = 1 \} )$ . Since $\bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ , Lemma 2.4 gives $D c ( \bar { x } ) = L$ , hence $\begin{array} { r } { S = \mathcal { B } [ 0 , \frac { 1 } { 2 } ] \cap L = \Phi ( \mathcal { B } _ { 2 } [ 0 , \frac { 1 } { 2 } ] ) } \end{array}$

$\mathrm { F o r ~ } s = \Phi ( z _ { s } ) \in S \mathrm { ~ a n d ~ } ^ { - } v = \Phi ( z _ { v } ) \in \mathrm { r b } \left( \mathrm { Q } \right)$ we have $\begin{array} { r } { \| z _ { s } \| _ { 2 } \le \frac { 1 } { 2 } } \end{array}$ and $\| z _ { v } \| _ { 2 } \geq \| z _ { v } \| _ { \infty } = 1$ , so $\begin{array} { r } { \| s - v \| = \| z _ { s } - z _ { v } \| _ { 2 } \geq 1 - \frac { 1 } { 2 } = \frac { 1 } { 2 } } \end{array}$ ; thus dist(S, rb $\mathrm { ( Q ) } ) \geq { \textstyle { \frac { 1 } { 2 } } }$ . Equality holds at $s = \Phi ( \textstyle { \frac { 1 } { 2 } } e _ { 1 } )$ 2 $v = \Phi ( e _ { 1 } )$ ✷

Proposition 7.2 (boundary gap value). In the setting of Lemma ${ 7 . 4 } ,$ set $\begin{array} { r } { \vartheta : = \frac { 1 - \bar { \eta } } { \bar { \eta } } } \end{array}$

1. If dim $\mathcal { C } \geq 2$ , then dist $\big ( S , \mathrm { r b } \left( \mathrm { P } \right) \big ) = \frac { \vartheta \beta \eta - \sqrt { ( \mathrm { d i m } \mathscr { C } - 1 ) ( 1 - \beta ^ { 2 } \eta ^ { 2 } ) } } { 4 \sqrt { \vartheta ^ { 2 } + \mathrm { d i m } \mathscr { C } - 1 } } .$

2. If dim C = 1, then dist $\begin{array} { r } { ( S , \mathrm { r b } \left( \mathrm { P } \right) ) = \frac { 1 } { 4 } } \end{array}$

3. In all cases, dist $\left( S , \operatorname { r b } \left( \operatorname { P } \right) \right) \geq { \frac { \left( \beta - 1 \right) \eta } { 4 { \sqrt { \dim { \mathcal { C } } } } } }$

Proof. We work throughout in the setting of Lemma 7.4: $\eta \in ( 0 , 1 )$ and $\beta \in ( 1 , 1 / \eta )$ , hence $\beta \eta \in \left( \eta , 1 \right)$ and $1 - \beta ^ { 2 } \eta ^ { 2 } > 0$ . Write $L : = \arg \left( { \mathcal { C } } - { \bar { x } } \right)$ and let $\{ u , w _ { 1 } , \ldots , w _ { \dim { \mathcal { C } } - 1 } \}$ be the orthonormal basis of L from the third branch of Algorithm 2, with u the unit vector of (7.3). The two objects furnished by Lemma 7.4 are the compact set $S = D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } ) \cap \{ d : 0 . 2 5 \leq \| d \| \leq $ 0.5} and the truncated polytope $P = \mathrm { p o s } \left( \bar { V } _ { 0 } \right) \cap H _ { u , - 1 }$ . As in Algorithm $2 , \bar { \eta } = \eta / ( 2 \sqrt { \dim \mathcal { C } } )$ Combined with the standing definition $\vartheta \dot { = } ( \bar { 1 } - \bar { \eta } ) / \bar { \eta } .$ , this gives

$$
\vartheta = \frac { 1 - \bar { \eta } } { \bar { \eta } } = \frac { 2 \sqrt { \dim \mathscr { C } } - \eta } { \eta } , \qquad \vartheta \geq 1 .
$$

Coordinatize L by the linear isometry $\Psi : \mathbb { R } \times \mathbb { R } ^ { \mathrm { d i m } \mathcal { C } - 1 } \to L , \Psi ( c , a ) = c u + \sum _ { j = 1 } ^ { \mathrm { d i m } \mathcal { C } - 1 } a _ { j } w _ { j }$ Distances are preserved, so we compute in $\mathbb { R } ^ { \mathrm { d i m } \it C }$ , writing points as $( c , a )$ with $c = \langle d , u \rangle$ and $\| ( c , a ) \| ^ { 2 } = c ^ { 2 } + \| a \| _ { 2 } ^ { 2 }$

Recall from the $\alpha = 0$ construction of Algorithm 2 at a smooth boundary point (cf. Lemma 7.2) that the non-exploration output is obtained by applying normalize to each vector in $\{ u \} \cup \{ \pm ( 1 - \bar { \eta } ) w _ { j } + \bar { \eta } u : j \in [ \mathrm { d i m } \mathcal { C } - 1 ] \}$ . Since $R _ { u } ( v ) = v / \langle v , u \rangle$ depends only on the positive ray through v, the truncated polytope P of Lemma 7.4 is unchanged if each generator is replaced by a positive multiple, and we compute with this unnormalized set. Scaling each direction by its u-component, for $v = u$ we have $\langle v , u \rangle = 1$ , so $R _ { u } ( v ) = u = ( 1 , 0 )$ . For $v = \pm ( 1 - \bar { \eta } ) w _ { j } + \bar { \eta } u$ orthonormality gives $\langle v , u \rangle = \bar { \eta }$ hence $\begin{array} { r } { R _ { u } ( v ) = u \pm \frac { 1 - \bar { \eta } } { \bar { n } } w _ { j } = ( 1 , \pm \vartheta e _ { j } ) } \end{array}$ . Thus the vertices $R _ { u } ( v )$ are (1, 0) and $( 1 , \pm \vartheta e _ { j } ) , j \in [ \mathrm { d i m } { \mathcal { C } } - 1 ]$ . Writing $K : = \{ ( c , a ) : \| a \| _ { 1 } \leq \vartheta c \}$ for the untruncated cone, we have $K \subseteq \{ c \geq 0 \}$ and therefore

$$
P = K \cap \{ c \leq 1 \} = \{ ( c , a ) : 0 \leq c \leq 1 , \ \| a \| _ { 1 } \leq \vartheta c \} .
$$

The inclusion $\subseteq$ holds as the right-hand set is convex and contains every vertex; for $\supseteq { \mathrm { : } }$ , a point with $c > 0$ and $\| a \| _ { 1 } \leq \vartheta c$ has $a / ( \vartheta c ) \in \mathrm { c o n v } \{ \pm e _ { j } \} , \mathrm { s o } ( 1 , a / c ) \in \mathrm { c o n v } \{ ( 1 , \pm \vartheta e _ { j } ) \}$ and $\mathbf { \xi } ( c , a ) = c \left( 1 , a / c \right) + ( 1 - c ) ( 0 , \mathbf { 0 } )$ , while $c = 0$ forces the apex.

We next write $S$ in these coordinates. For d $\neq 0 ,$ since $c = \langle d , u \rangle$ and $\| d \| ^ { 2 } = c ^ { 2 } + \| a \| _ { 2 } ^ { 2 }$ the defining inequality of $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ reads $c / \sqrt { c ^ { 2 } + \| a \| _ { 2 } ^ { 2 } } \geq \beta \eta . \mathrm { ~ A s ~ } \beta \eta \in ( \eta , 1 ) \subseteq ( 0 , 1 )$ while the denominator is positive, the inequality forces $c > 0$ . Set

$$
\mu : = \frac { \beta \eta } { \sqrt { 1 - \beta ^ { 2 } \eta ^ { 2 } } } .
$$

Both sides being positive, squaring is an equivalence, and $1 - \beta ^ { 2 } \eta ^ { 2 } > 0$ because $\beta < 1 / \eta$ , so

$$
\begin{array} { r l } & { \langle d / \left. d \right. , u \rangle \geq \beta \eta \iff \frac { c } { \sqrt { c ^ { 2 } + \left. a \right. _ { 2 } ^ { 2 } } } \geq \beta \eta } \\ & { \qquad \iff c ^ { 2 } ( 1 - \beta ^ { 2 } \eta ^ { 2 } ) \geq \beta ^ { 2 } \eta ^ { 2 } \left. a \right. _ { 2 } ^ { 2 } } \\ & { \qquad \iff c \geq \mu \left. a \right. _ { 2 } . } \end{array}
$$

Every step reverses on $\{ d \ne 0 \}$ . The definition of $\mu$ also gives

$$
1 + \mu ^ { 2 } = \frac { 1 } { 1 - \beta ^ { 2 } \eta ^ { 2 } } , \qquad \frac { \mu } { \sqrt { 1 + \mu ^ { 2 } } } = \beta \eta , \qquad \frac { 1 } { \sqrt { 1 + \mu ^ { 2 } } } = \sqrt { 1 - \beta ^ { 2 } \eta ^ { 2 } } .
$$

Thus

$$
S = \{ ( c , a ) : c \geq \mu \| a \| _ { 2 } , \frac { 1 } { 4 } \leq \sqrt { c ^ { 2 } + \| a \| _ { 2 } ^ { 2 } } \leq \frac { 1 } { 2 } \} .
$$

The cone $K$ is a polyhedron. Writing $\begin{array} { r } { \| a \| _ { 1 } = \operatorname* { m a x } _ { \epsilon \in \{ - 1 , 1 \} } \dim \mathcal { C } - 1 \left. \epsilon , a \right. } \end{array}$ expresses K as the halfspace intersection

$$
K = \bigcap _ { \epsilon \in \{ - 1 , 1 \} ^ { \dim \mathcal { C } - 1 } } \{ \langle \epsilon , a \rangle - \vartheta c \leq 0 \} ,
$$

with facet normals $\nu _ { \epsilon } = ( - \vartheta , \epsilon )$ and right-hand sides 0. Each normal has $\| \nu _ { \epsilon } \| = \sqrt { \vartheta ^ { 2 } + \| \epsilon \| ^ { 2 } }$ hence $\| \nu _ { \epsilon } \| = \sqrt { \vartheta ^ { 2 } + \mathrm { d i m } \mathcal { C } - 1 }$ , and $( 1 , \mathbf { 0 } ) \in \mathrm { i n t } ( K )$ since $0 < \vartheta \cdot 1$ . In particular K and $P$ are full-dimensional in these coordinates, so rb $( \operatorname { K } ) = \mathbf { b d } ( K )$ and rb $( \mathrm { P } ) = \mathbf { b d } ( P )$ throughout. By Lemma $7 . 4 ( 1 ) , S \subseteq \mathrm { r i } \left( \mathrm { P } \right)$ and af $( P ) = L$ , hence in Ψ-coordinates $S \subseteq \{ 0 < c < 1 , \| a \| _ { 1 } <$ $\vartheta c \rbrace \subseteq \mathbf { i n t } ( K )$ . Applying Lemma 7.7 at every $( c , a ) \in \mathbf { i n t } ( K )$ and minimizing the per-facet residual over $\epsilon ,$

$$
\operatorname { d i s t } \bigl ( ( c , a ) , \mathrm { r b } \left( \mathrm { K } \right) \bigr ) = \operatorname* { m i n } _ { \epsilon } \frac { \vartheta c - \left. \epsilon , a \right. } { \sqrt { \vartheta ^ { 2 } + \dim \mathcal { C } - 1 } } = \frac { \vartheta c - \operatorname* { m a x } _ { \epsilon } \left. \epsilon , a \right. } { \sqrt { \vartheta ^ { 2 } + \dim \mathcal { C } - 1 } } = \frac { \vartheta c - \left\| a \right\| _ { 1 } } { \sqrt { \vartheta ^ { 2 } + \dim \mathcal { C } - 1 } } .\tag{7.11}
$$

We first evaluate inf dist $\left( s , \operatorname { r b } \left( \operatorname { K } \right) \right)$ , which (7.11) reduces to a scaled infimum of $\vartheta c - \| a \| _ { 1 }$ $\bar { s } \bar { \in \cal S }$ over S. Write $r = \| ( c , a ) \| \in [ \textstyle { \frac { 1 } { 4 } } , \frac { 1 } { 2 } ]$ . Since $r ^ { 2 } = c ^ { 2 } + \| a \| _ { 2 } ^ { 2 }$ and $c \geq \mu \left\| a \right\| _ { 2 } \geq 0$ , we have $c / r \in [ 0 , 1 ]$ . Set $\psi : = \operatorname { a r c c o s } ( c / r )$ , so $\psi \in [ 0 , \frac { \pi } { 2 } ]$ and $c = r \cos \psi$ . Then sin $\psi \geq 0$ and

$$
\sin \psi = { \sqrt { 1 - \cos ^ { 2 } \psi } } = { \sqrt { 1 - { \frac { c ^ { 2 } } { r ^ { 2 } } } } } = { \frac { \| a \| _ { 2 } } { r } } ,
$$

hence $\| a \| _ { 2 } = r$ sin $\psi$ . The cone constraint $c \geq \mu \| a \| _ { 2 }$ becomes cot $\psi \geq \mu .$ , i.e. $\psi \leq \psi _ { 0 } : =$ arctan $( 1 / \mu )$ . Bounding $\left\| a \right\| _ { 1 } \leq \sqrt { \dim \mathcal { C } - 1 } \left\| a \right\| _ { 2 }$ by Cauchy–Schwarz,

$$
\begin{array} { r } { \vartheta c - \| a \| _ { 1 } \geq r \big ( \vartheta \cos \psi - \sqrt { \mathrm { d i m } \mathscr { C } - 1 } \sin \psi \big ) = : r \phi ( \psi ) . } \end{array}
$$

Since $\phi ^ { \prime } ( \psi ) = - \vartheta$ sin $\psi - { \sqrt { \dim { \mathcal { C } } - 1 } }$ cos $\psi \leq 0$ on $[ 0 , { \frac { \pi } { 2 } } ] , \phi$ is nonincreasing and is minimized over $[ 0 , \psi _ { 0 } ]$ at $\psi _ { 0 }$ . From tan $\psi _ { 0 } = 1 / \mu$ and $\sin ^ { 2 } \psi _ { 0 } + \cos ^ { 2 } \psi _ { 0 } = 1$ we have

$$
\cos \psi _ { 0 } = \frac { \mu } { \sqrt { 1 + \mu ^ { 2 } } } , \qquad \sin \psi _ { 0 } = \frac { 1 } { \sqrt { 1 + \mu ^ { 2 } } } ,
$$

$$
\phi ( \psi _ { 0 } ) = \frac { \vartheta \mu - \sqrt { \mathrm { d i m } \mathcal { C } - 1 } } { \sqrt { 1 + \mu ^ { 2 } } } .
$$

The numerator is nonnegative: substituting the identity for ϑ from the opening,

$$
\vartheta \mu = \frac { \beta \big ( 2 \sqrt { \dim \mathscr { C } } - \eta \big ) } { \sqrt { 1 - \beta ^ { 2 } \eta ^ { 2 } } } \geq \beta \big ( 2 \sqrt { \dim \mathscr { C } } - \eta \big ) > 2 \sqrt { \dim \mathscr { C } } - 1 \geq \sqrt { \dim \mathscr { C } } \geq \sqrt { \dim \mathscr { C } - 1 } ,
$$

the first inequality as $\sqrt { 1 - \beta ^ { 2 } \eta ^ { 2 } } \le 1$ , the second as $\beta > 1$ and $\eta < 1$ , and the last two as dim $\mathcal { C } \geq 1$ . The same chain yields

$$
\vartheta \beta \eta = \beta \big ( 2 \sqrt { \mathrm { d i m } { \mathcal { C } } } - \eta \big ) > \sqrt { \mathrm { d i m } { \mathcal { C } } } > \sqrt { \mathrm { d i m } { \mathcal { C } } - 1 } ,\tag{7.12}
$$

used for the lower bound in (3) below. Thus $\phi ( \psi _ { 0 } ) \geq 0 \quad$

Because $\begin{array} { r } { r \geq \frac { 1 } { 4 } } \end{array}$ and $\phi ( \psi _ { 0 } ) \ge 0$ , we get $\begin{array} { r } { \vartheta c - \| a \| _ { 1 } \geq \frac { 1 } { 4 } \phi ( \psi _ { 0 } ) } \end{array}$ . For dim $\mathcal { C } \geq 2$ this bound is attained, and that is the only place the assumption dim $\mathcal { C } \geq 2$ is used in the evaluation of inf dist $\left( s , \operatorname { r b } \left( \operatorname { K } \right) \right)$ . Explicitly, take $\textstyle r = { \frac { 1 } { 4 } } , \psi = \psi _ { 0 }$ , and a with equal-magnitude entries: s∈S the resulting point sits on the cone boundary $c = \mu \| a \| _ { 2 } { \mathrm { ~ } } ( \operatorname { a s } \psi = \psi _ { 0 } )$ , saturates $\| a \| _ { 1 } =$ ${ \sqrt { \dim C - 1 } } \| a \| _ { 2 }$ (equal magnitudes attain Cauchy–Schwarz), and lies in S since $\begin{array} { r } { r = \frac { 1 } { 4 } \in [ \frac { 1 } { 4 } , \frac { 1 } { 2 } ] } \end{array}$ and $c = \mu \left\| a \right\| _ { 2 } \geq 0$ places it in the closed cone $D _ { \mathcal { C } } ^ { \beta \eta } ( \bar { x } )$ . Therefore

$$
\operatorname* { i n f } _ { s \in S } \operatorname { d i s t } ( s , \operatorname { r b } \left( \operatorname { K } \right) ) = \frac { \phi ( \psi _ { 0 } ) } { 4 \sqrt { \vartheta ^ { 2 } + \dim \mathcal { C } - 1 } } \leq \frac { 1 } { 4 } .\tag{7.13}
$$

For the upper bound, dropping the nonnegative term gives $\phi ( \psi _ { 0 } ) \leq \vartheta \mu / \sqrt { 1 + \mu ^ { 2 } }$ , and

$$
{ \sqrt { ( 1 + \mu ^ { 2 } ) ( \vartheta ^ { 2 } + \dim { \mathcal { C } } - 1 ) } } \geq \vartheta \mu
$$

(square both sides: $\vartheta ^ { 2 } + \big ( \mathrm { d i m } \mathcal { C } - 1 \big ) ( 1 + \mu ^ { 2 } ) \geq 0 \big )$ , so the ratio is at most $\textstyle { \frac { 1 } { 4 } }$

It remains to identify dist(S, rb (P)) with inf dist $( s , \mathrm { r b } \left( \mathrm { K } \right) )$ when dim $\mathcal { C } \geq 2$ . Since s∈S $P = K \cap \{ c \leq 1 \}$ , the elementary inclusion bd $( \bar { A } \cap B ) \subseteq ( \mathbf { b d } ( A ) \cap B ) \cup ( A \cap \mathbf { b d } ( B ) )$ gives rb $( \mathrm { P } ) \subseteq ( \mathrm { r b } ( \mathrm { K } ) \cap \{ 0 \leq c \leq 1 \} ) \cup \{ c = 1 \}$ . Conversely, a point of rb (K) with first coordinate in [0, 1] lies on rb ${ ( \mathrm { P } ) \colon \mathrm { i f ~ } p \in \mathrm { r b } ( \mathrm { K } ) }$ and $c _ { p } \in [ 0 , 1 ]$ , then $p \in K$ and $c _ { p } \leq 1 , \mathrm { s o } p \in P$ . Every neighbourhood of p contains a point outside $K _ { i }$ , and that point lies outside P as well, since $P \subseteq K$ . Thus $p \in \mathrm { r b } \left( \mathrm { P } \right)$ , and rb $( \mathrm { K } ) \cap \{ 0 \leq c \leq 1 \} \subseteq \mathrm { r b } \left( \mathrm { P } \right)$

Fix $s = ( c _ { s } , a _ { s } ) \in S .$ and let $p \in \mathrm { r b } \left( \mathrm { K } \right)$ be a nearest point to s. We check that the first coordinate of p lies in $[ 0 , 1 ]$ , so that $p \in \mathrm { r b } \left( \mathrm { P } \right)$ . Since $c _ { s } \leq \| s \| \leq \frac { 1 } { 2 }$ and p realizes the inradius distance (7.11), writing $\delta : = \vartheta c _ { s } - \left\| a _ { s } \right\| _ { 1 } > 0 ( \mathrm { a s } S \subseteq \mathbf { i n t } ( K ) )$

$$
\| p - s \| = { \frac { \delta } { \sqrt { \vartheta ^ { 2 } + \dim { \mathcal { C } } - 1 } } } \leq c _ { s } ,
$$

using $\delta \leq \vartheta c _ { s }$ and $\sqrt { \vartheta ^ { 2 } + \mathrm { d i m } \mathcal { C } - 1 } \geq \vartheta$ . Hence the first coordinate of p obeys $| c _ { p } - c _ { s } | \leq$ $\left\| p - s \right\| \leq c _ { s } , { \mathrm { ~ i . e . ~ } } c _ { p } \in [ 0 , 2 c _ { s } ] \subseteq [ 0 , 1 ]$ (as $c _ { s } \leq \frac { 1 } { 2 } )$ , so $p \in \mathrm { r b } \left( \mathrm { K } \right) \cap \{ 0 \leq c \leq 1 \} \subseteq$ rb (P) and dist $( s , \mathrm { r b } \left( \mathrm { P } \right) ) \leq$ dist(s, rb (K)). Conversely, the distance from s to the cap $\{ c = 1 \}$ is $\begin{array} { r } { 1 - c _ { s } \ge \frac { 1 } { 2 } } \end{array}$ , and rb $( \mathrm { K } ) \cap \{ 0 \leq c \leq 1 \} \subseteq \mathrm { r b } \left( \mathrm { K } \right)$ only increases distance, so

$$
\mathrm { d i s t } ( s , \mathrm { r b } \left( \mathrm { P } \right) ) \geq \operatorname* { m i n } \big \{ \mathrm { d i s t } ( s , \mathrm { r b } \left( \mathrm { K } \right) ) , 1 - c _ { s } \big \} \geq \operatorname* { m i n } \big \{ \mathrm { d i s t } ( s , \mathrm { r b } \left( \mathrm { K } \right) ) , \frac { 1 } { 2 } \big \} .
$$

Taking infima and using the pointwise bound dist $( s , \mathrm { r b } \left( \mathrm { P } \right) ) \le \mathrm { d i s t } ( s , \mathrm { r b } \left( \mathrm { K } \right) )$ together with   
inf dist $\begin{array} { r } { ( s , \mathrm { r b } \left( \mathrm { K } \right) ) \le \frac { 1 } { 4 } \ \left( 7 . 1 3 \right) } \end{array}$   
s∈S

$$
\operatorname* { i n f } _ { s \in S } \mathrm { d i s t } ( s , \operatorname { r b } { ( \mathrm { P } ) } ) \leq \operatorname* { i n f } _ { s \in S } \mathrm { d i s t } ( s , \operatorname { r b } { ( \mathrm { K } ) } ) ,
$$

$$
\operatorname* { i n f } _ { s \in S } \mathrm { d i s t } ( s , \operatorname { r b } \left( \operatorname { P } \right) ) \geq \operatorname* { m i n } \{ \operatorname* { i n f } _ { s \in S } \mathrm { d i s t } ( s , \operatorname { r b } \left( \operatorname { K } \right) ) , \frac { 1 } { 2 } \} = \operatorname* { i n f } _ { s \in S } \mathrm { d i s t } ( s , \operatorname { r b } \left( \operatorname { K } \right) ) .
$$

Therefore

$$
\operatorname { d i s t } ( S , \operatorname { r b } \left( \operatorname { P } \right) ) = \operatorname* { i n f } _ { s \in S } \operatorname { d i s t } ( s , \operatorname { r b } \left( \operatorname { K } \right) ) = \frac { 1 } { \sqrt { \vartheta ^ { 2 } + \dim \mathscr { C } - 1 } } \operatorname* { i n f } _ { s \in S } \left( \vartheta c _ { s } - \left\| a _ { s } \right\| _ { 1 } \right) .\tag{7.14}
$$

Plugging the attained value into (7.14),

$$
\operatorname { d i s t } ( S , \operatorname { r b } \left( \operatorname { P } \right) ) = { \frac { 1 } { \sqrt { \vartheta ^ { 2 } + \dim { \mathcal { C } } - 1 } } } \cdot { \frac { 1 } { 4 } } \cdot { \frac { \vartheta \mu - { \sqrt { \dim { \mathcal { C } } - 1 } } } { \sqrt { 1 + \mu ^ { 2 } } } } ,
$$

and using $\begin{array} { r } { \frac { \mu } { \sqrt { 1 + \mu ^ { 2 } } } = \beta \eta , \frac { 1 } { \sqrt { 1 + \mu ^ { 2 } } } = \sqrt { 1 - \beta ^ { 2 } \eta ^ { 2 } } } \end{array}$ to clear $\mu$ from the numerator,

$$
{ \frac { \vartheta \mu - { \sqrt { \dim { \mathcal { C } } - 1 } } } { \sqrt { 1 + \mu ^ { 2 } } } } = \vartheta \beta \eta - { \sqrt { \dim { \mathcal { C } } - 1 } } { \sqrt { 1 - \beta ^ { 2 } \eta ^ { 2 } } } = \vartheta \beta \eta - { \sqrt { ( \dim { \mathcal { C } } - 1 ) ( 1 - \beta ^ { 2 } \eta ^ { 2 } ) } } ,
$$

yields the formula in (1).

The case dim $\mathcal { C } = 1$ is computed directly and does not use the identification (7.14). Here $K = \{ c \geq 0 \} , P = [ 0 , 1 ] \cdot u , { \mathrm { ~ r b ~ } } ( \mathrm { P } ) = \{ 0 , u \}$ , and $\begin{array} { r } { S = \{ c u : \frac { 1 } { 4 } \leq c \leq \frac { 1 } { 2 } \} } \end{array}$ , so dist $\begin{array} { r } { ( S , \mathrm { r b } \left( \mathrm { P } \right) ) = \frac { 1 } { 4 } } \end{array}$ establishing (2). The case split is necessary: with no transverse coordinate $^ { a , }$ the minimizing angle ψ<sub>0</sub> is unavailable, and the expression in (1) evaluates to $\beta \eta / 4$ , which is strictly smaller than ${ \frac { 1 } { 4 } } .$

For the lower bound in (3), first take dim $\mathcal { C } = 1$ . Then $( \beta - 1 ) \eta = \beta \eta - \eta < 1 - \eta < 1$ (using $\beta < 1 / \eta )$ , so $\begin{array} { r } { \frac { 1 } { 4 } \geq \frac { ( \beta - 1 ) \dot { \eta } } { 4 } } \end{array}$

Now take dim $\mathcal { C } \geq 2$ and bound formula (1). The numerator satisfies

$$
\vartheta \beta \eta - \sqrt { ( \mathrm { d i m } \mathcal { C } - 1 ) ( 1 - \beta ^ { 2 } \eta ^ { 2 } ) } \geq \vartheta \beta \eta - \sqrt { \mathrm { d i m } \mathcal { C } - 1 } ,
$$

as $1 - \beta ^ { 2 } \eta ^ { 2 } \le 1 \le$ ; the right-hand side is positive by the estimate $\vartheta \beta \eta > \sqrt { \mathrm { d i m } \mathcal { C } - 1 } \left( 7 . 1 2 \right)$ . For the denominator, $\vartheta \geq 1$ as already noted, hence

$$
{ \sqrt { \vartheta ^ { 2 } + \dim { \mathcal { C } } - 1 } } \leq { \sqrt { \vartheta ^ { 2 } + ( \dim { \mathcal { C } } - 1 ) \vartheta ^ { 2 } } } = \vartheta { \sqrt { \dim { \mathcal { C } } } } .
$$

Combining,

$$
\operatorname { d i s t } ( S , \operatorname { r b } \left( \operatorname { P } \right) ) \geq \frac { \vartheta \beta \eta - \sqrt { \dim { \mathscr { C } } - 1 } } { 4 \vartheta \sqrt { \dim { \mathscr { C } } } } = \frac { 1 } { 4 \sqrt { \dim { \mathscr { C } } } } \Big ( \beta \eta - \frac { \sqrt { \dim { \mathscr { C } } - 1 } } { \vartheta } \Big ) .
$$

Finally

$$
\frac { 1 } { \vartheta } = \frac { \bar { \eta } } { 1 - \bar { \eta } } \leq 2 \bar { \eta } = \frac { \eta } { \sqrt { \mathrm { d i m } \mathcal { C } } } ,
$$

from 1 − $\bar { \eta } \geq \frac { 1 } { 2 }$ , so

$$
{ \frac { \sqrt { \dim { \mathcal { C } } - 1 } } { \vartheta } } \leq \eta { \sqrt { \frac { \dim { \mathcal { C } } - 1 } { \dim { \mathcal { C } } } } } \leq \eta .
$$

Therefore $\begin{array} { r } { \beta \eta - \frac { \sqrt { \mathrm { d i m } } \mathcal { C } - 1 } { \vartheta } \geq ( \beta - 1 ) \eta \mathrm { ~ a n d ~ } \mathrm { d i s t } ( S , \mathrm { r b } \left( \mathrm { P } \right) ) \geq \frac { ( \beta - 1 ) \eta } { 4 \sqrt { \mathrm { d i m } \mathcal { C } } } . } \end{array}$

In both cases the argument gives $0 < \mathrm { d i s t } ( S , \mathrm { r b } \left( \mathrm { P } \right) ) \leq \textstyle { \frac { 1 } { 4 } }$

## Corollary 7.1 (explicit admissible noise levels). Recall $\bar { \eta } = \eta / ( 2 \sqrt { \mathrm { d i m } { \mathcal C } } )$ as in Algorithm 2.

1. $I f \bar { x } \in \mathrm { r i } \left( \mathcal { C } \right)$ , the conclusion of Theorem 7.2 holds whenever

$$
\begin{array} { r } { 0 < \bar { \alpha } < \operatorname* { m i n } \Bigl \{ 1 , \frac { 1 } { 4 \dim \mathcal { C } \bar { \eta } } \Bigr \} = \operatorname* { m i n } \Bigl \{ 1 , \frac { 1 } { 2 \sqrt { \dim \mathcal { C } } \eta } \Bigr \} . } \end{array}
$$

2. $I f \bar { x } \in \mathrm { r b } ( \mathcal { C } )$ and C is smooth at x¯, then for every $\beta \in ( 1 , 1 / \eta )$ the conclusion holds whenever $0 < \bar { \alpha } <$ min $\left\{ \frac { 1 } { 4 } , ~ \bar { \eta }\right.$ dist $( S , \mathrm { r b } \left( \mathrm { P } \right) ) / 1 6 \}$ , and in particular whenever

$$
0 < \bar { \alpha } < \operatorname * { m i n } \Bigl \{ { \textstyle { \frac { 1 } { 4 } } } , \frac { ( \beta - 1 ) \eta ^ { 2 } } { 1 2 8 \mathrm { d i m } { \cal C } } \Bigr \} .
$$

Proof. (1): the Case 1 bound $d _ { H } ( Q _ { \bar { \alpha } } , Q ) \leq 2$ dim $\mathcal { C } \bar { \alpha } \bar { \eta }$ and Proposition 7.1 give $d _ { H } ( Q _ { \bar { \alpha } } , Q ) <$ $\begin{array} { r } { \frac { 1 } { \mathcal { D } } = \mathrm { d i s t } ( S , \mathrm { r b } \left( \mathrm { Q } \right) ) } \end{array}$ for $\begin{array} { r } { \bar { \alpha } < \frac { 1 } { 4 \mathrm { d i m } \mathcal { C } \bar { \eta } } } \end{array}$ , whence $S \subseteq Q _ { \bar { \alpha } }$ by Lemma 7.6; the cap at 1 is $\alpha \in$ [0, 1). (2): the first range is the Case 2 threshold; the second uses Proposition $7 . 2 ( 3 )$ , since η¯ dist $( S , { \mathrm { r b } } \left( { \mathrm { P } } \right) ) / 1 6 \geq { \bar { \eta } } ( \beta - 1 ) \eta / ( 6 4 { \sqrt { \dim { \mathscr { C } } } } ) = ( \beta - 1 ) \eta ^ { 2 } / ( 1 2 8 \dim { \mathscr { C } } )$ ✷

Lemma 6.1 turns a zero value of the PDM measure into the first-order condition $h ^ { \prime } ( x _ { k } ; \cdot ) \geq$ 0 on any cone contained in the positive span of the direction set. Theorem 7.2 shows that this cone may be taken to be $D _ { \mathcal { C } } ( x _ { k } )$ at a relative interior point and $D _ { \mathcal { C } } ^ { \beta \eta } ( x _ { k } )$ at a smooth boundary point. By Corollary 7.1 the thresholds depend only on $\eta ,$ dim C and $\beta ,$ so a single admissible α serves at every interior point and every smooth boundary point.

Corollary 7.2 (PDM certificate for the noised method). Let $\{ x _ { k } \} _ { k \ge 0 }$ be generated by Algorithm 3 with $\eta \in ( 0 , 1 ]$ and $\alpha \in ( 0 , 1 )$ , fix $k \geq 0$ , and suppose $\mathcal { O } ^ { \gamma } ( x _ { k } ; \tilde { V } ^ { k } , r ) = 0$

1. $I f x _ { k } \in \mathrm { r i } ( \mathcal { C } )$ and $\alpha \in ( 0 , \tilde { \alpha } )$ for the threshold α˜ of part 1 of Theorem 7.2 at $x _ { k }$ , then

$$
h ^ { \prime } ( x _ { k } ; d ) \geq 0 \qquad \forall d \in D \mathcal { c } ( x _ { k } ) .
$$

2. If $x _ { k } \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ , C is smooth at $x _ { k } , \beta \in ( 1 , 1 / \eta )$ , and $\alpha \in ( 0 , \tilde { \alpha } )$ for the threshold α˜ of part 2 of Theorem $\gamma . \mathcal { Z } ~ a t ~ x _ { k } ,$ , then

$$
h ^ { \prime } ( x _ { k } ; d ) \geq 0 \qquad \forall d \in D _ { \mathcal { C } } ^ { \beta \eta } ( x _ { k } ) .
$$

Proof. Write $\hat V ^ { k } : = \tilde { V } ^ { k } \setminus \{ \tilde { v } _ { \mathrm { e x } } ^ { k } \}$ . By Lemma 7.1, $\bar { V } ^ { k } \subseteq D _ { \mathcal { C } } ( x _ { k } )$ . The measure (6.1) is a sum of squares, so $\mathcal { O } ^ { \gamma } ( x _ { k } ; \tilde { V } ^ { k } , r ) = 0$ implies $\mathcal { O } ^ { \gamma } ( x _ { k } ; \bar { V } ^ { k } , r ) = 0$ . The set $\tilde { V } ^ { k }$ is the output of Algorithm 2 at $x _ { k }$ with this η and $\alpha ,$ so Theorem 7.2 applies.

If $x _ { k } \in \mathrm { r i } \left( \mathcal { C } \right)$ and α is as in part 1, then $D c ( x _ { k } ) \subseteq \mathrm { p o s } \left( { \bar { V } } ^ { k } \right)$ . Lemma 6.1 with $\mathcal { K } = D _ { \mathcal { C } } ( x _ { k } )$ yields $\langle \nabla f ( x _ { k } ) , d \rangle - g ^ { \prime } ( x _ { k } ; d ) \geq 0$ for every $d \in D c ( x _ { k } )$ , which is $h ^ { \prime } ( x _ { k } ; d ) \ge 0$

If $x _ { k } \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ with C smooth at $x _ { k }$ and α is as in part 2, then $D _ { \mathcal { C } } ^ { \beta \eta } ( x _ { k } ) \subseteq \mathrm { p o s } \left( \bar { V } ^ { k } \right)$ . The same lemma with $\mathcal { K } = D _ { \mathcal { C } } ^ { \beta \eta } ( x _ { k } )$ gives the claim. ✷

## 8 Conclusion

In this paper we developed a feasible-directions framework for (P) over a general closed convex set, based on η-feasible directions: feasible directions that remain uniformly separated from the boundary of the feasible direction cone. We proposed two methods that search along finitely many feasible directions whose positive span contains the η-feasible directions, together with a randomized exploration direction. The first assumes access to an η-direction oracle and extends the linearly constrained method of Beck and Hallak [6]. The second constructs spanning directions by a noise-augmented procedure and avoids nonsmooth boundary points almost surely. For both methods we established an $O ( 1 / K )$ rate in the proximal directional minimization measure of Beck and Hallak [6]. In addition, by an adaptation of the exploration result of Greenstein and Hallak [16] to the regularized line search, every accumulation point of either method is almost surely d-stationary. Under curvature bounds on a smooth constraint boundary, Algorithm 1 driven by the depth-aware construction of Algorithm 2 reaches $( \varepsilon , R ) \cdot$ d-stationarity in $O ( \mathrm { d i m } { \mathcal { C } } \cdot \varepsilon ^ { - 2 } )$ iterations. As a byproduct, we obtained a characterization of η-feasible directions at interior and smooth boundary points, and directional spanning properties under noise.

## References

[1] Hadi Abbaszadehpeivasti, Etienne de Klerk, and Moslem Zamani. On the rate of convergence of the diference-of-convex algorithm (dca). Journal of Optimization Theory and Applications, 202(1):475–496, 2024.

[2] M. Ahn, J. S. Pang, and J. Xin. Diference-of-convex learning: Directional stationarity, optimality, and sparsity. SIAM Journal on Optimization, 27(3):1637–1665, jan 2017. doi: 10.1137/16m1084754.

[3] Giovanni Alberti. On the structure of singular sets of convex functions. Calculus of Variations and Partial Diferential Equations, 2:17–27, 1994.

[4] Charles Audet and John E Dennis Jr. Mesh adaptive direct search algorithms for constrained optimization. SIAM Journal on optimization, 17(1):188–217, 2006.

[5] Amir Beck and Nadav Hallak. On the convergence to stationary points of deterministic and randomized feasible descent directions methods. SIAM Journal on Optimization, 30(1): 56–79, 2020. doi: 10.1137/18M1217760. URL https://doi.org/10.1137/18M1217760.

[6] Amir Beck and Nadav Hallak. The regularized feasible directions method for nonconvex optimization. Operations Research Letters, 50(5):517–523, 2022. ISSN 0167-6377. doi: https://doi.org/10.1016/j.orl.2022.07.005. URL https://www.sciencedirect.com/ science/article/pii/S0167637722000906.

[7] Andrea Brilli, Giampaolo Liuzzi, and Stefano Lucidi. An interior point method for nonlinear constrained derivative-free optimization. Optimization Methods and Software, pages 1–39, 2025.

[8] Emmanuel J Candes, Michael B Wakin, and Stephen P Boyd. Enhancing sparsity by reweighted l1 minimization. Journal of Fourier analysis and applications, 14:877–905, 2008.

[9] Andrew R Conn, Katya Scheinberg, and Luis N Vicente. Introduction to derivative-free optimization. SIAM, 2009.

[10] Lawrence C. Evans and Ronald F. Gariepy. Measure Theory and Fine Properties of Functions, Revised Edition. Chapman and Hall/CRC, 1 edition, 2015. doi: 10.1201/ b18333.

[11] Jianqing Fan and Runze Li. Variable selection via nonconcave penalized likelihood and its oracle properties. Journal of the American statistical Association, 96(456):1348–1360, 2001.

[12] G. Fasano, G. Liuzzi, S. Lucidi, and F. Rinaldi. A linesearch-based derivative-free approach for nonsmooth constrained optimization. SIAM Journal on Optimization, 24(3):959–992, 2014. doi: 10.1137/130940037. URL https://doi.org/10.1137/130940037.

[13] Marguerite Frank and Philip Wolfe. An algorithm for quadratic programming. Naval Research Logistics Quarterly, 3(1-2):95–110, 1956.

[14] Komei Fukuda. Lecture: Polyhedral computation, spring 2016. Lecture notes, Department of Mathematics and Institute of Theoretical Computer Science, ETH Zürich, 2016. https: //people.inf.ethz.ch/fukudak/lect/pclect/notes2016/PolyComp2016.pdf.

[15] Serge Gratton, Clément W Royer, Luís Nunes Vicente, and Zaikun Zhang. Direct search based on probabilistic feasible descent for bound and linearly constrained problems. Computational Optimization and Applications, 72(3):525–559, 2019.

[16] Dan Greenstein and Nadav Hallak. Achieving directional-stationarity from a single random direction step. arXiv preprint arXiv:2605.22045, 2026.

[17] Jean-Baptiste Hiriart-Urruty and Claude Lemaréchal. Convex Analysis and Minimization Algorithms I: Fundamentals, volume 305 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 1993. ISBN 978-3-540-56850-6. doi: 10.1007/978-3-662-02796-7.

[18] Martin Jaggi. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In Proceedings of the 30th International Conference on Machine Learning, pages 427–435. PMLR, 2013.

[19] Tamara G Kolda, Robert Michael Lewis, and Virginia Torczon. Optimization by direct search: New perspectives on some classical and modern methods. SIAM review, 45(3): 385–482, 2003.

[20] Tamara G Kolda, Robert Michael Lewis, and Virginia Torczon. Stationarity results for generating set search for linearly constrained optimization. SIAM Journal on Optimization, 17(4):943–968, 2006.

[21] Jefrey Larson, Matt Menickelly, and Stefan M Wild. Derivative-free optimization methods. Acta Numerica, 28:287–404, 2019.

[22] Robert Michael Lewis and Virginia Torczon. Pattern search algorithms for bound constrained minimization. SIAM Journal on Optimization, 9(4):1082–1099, 1999.

[23] Robert Michael Lewis and Virginia Torczon. Pattern search methods for linearly constrained minimization. SIAM Journal on Optimization, 10(3):917–941, 2000.

[24] Giampaolo Liuzzi, Stefano Lucidi, and Marco Sciandrone. Sequential penalty derivativefree methods for nonlinear constrained optimization. SIAM Journal on Optimization, 20 (5):2614–2635, 2010.

[25] Stefano Lucidi, Marco Sciandrone, and Paul Tseng. Objective-derivative-free methods for constrained optimization. Mathematical Programming, 92(1):37–59, 2002.

[26] José Mario Martínez. Local minimizers of quadratic functions on euclidean balls and spheres. SIAM Journal on Optimization, 4(1):159–176, 1994. doi: 10.1137/0804009. URL https://doi.org/10.1137/0804009.

[27] J. S. Pang, M. Razaviyayn, and A. Alvarado. Computing B-Stationary Points of Nonsmooth DC Programs. Mathematics of Operations Research, 42(1):95–118, jan 2017. ISSN 0364-765X.

[28] Duy Nhat Phan and Hoai An Le Thi. Diference-of-convex algorithm with extrapolation for nonconvex, nonsmooth optimization problems. Mathematics of Operations Research, 49(3):1973–1985, 2024.

[29] R Tyrrell Rockafellar. Convex analysis, volume 28. Princeton university press, 1997.

[30] R. Tyrrell Rockafellar and Roger J.-B. Wets. Variational Analysis, volume 317 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, Heidelberg, corrected 3rd printing edition, 2009. ISBN 978-3-540-62772-2. doi: 10.1007/978-3-642-02431-3.

[31] H. L. Royden and P. M. Fitzpatrick. Real Analysis. Pearson, Upper Saddle River, NJ, 4 edition, 2010. ISBN 978-0-13-143747-0.

[32] Rolf Schneider. Convex Bodies: The Brunn–Minkowski Theory, volume 44 of Encyclopedia of Mathematics and its Applications. Cambridge University Press, Cambridge, 1993. ISBN 0521352207. doi: 10.1017/CBO9780511526282.

[33] Albert N. Shiryaev. Probability-1, volume 95 of Graduate Texts in Mathematics. Springer, 3rd edition, 2016. doi: 10.1007/978-0-387-72206-1.

[34] Albert N. Shiryaev. Probability-2, volume 95 of Graduate Texts in Mathematics. Springer, 3rd edition, 2019. doi: 10.1007/978-0-387-72208-5.

[35] Pham Dinh Tao and Le Thi Hoai An. A dc optimization algorithm for solving the trust-region subproblem. SIAM Journal on Optimization, 8(2):476–505, 1998.

## A Proofs of Section 2

Proof of Lemma 2.1. We note that since $0 \in D c ( \bar { x } ) , D c ( \bar { x } ) \neq \emptyset$ for every nonempty set C and element ${ \bar { x } } \in { \mathcal { C } }$

First, we prove that $D _ { \mathcal { C } } \left( \bar { x } \right)$ is a cone. Let $y \in D c \left( \bar { x } \right)$ . By definition, $\exists \lambda > 0 \mathrm { ~ s . t . ~ } \bar { x } + \lambda y \in \mathcal { C }$ For $\omega = 0 , \omega y = 0 \in D c ( \bar { x } )$ . For $\omega > 0$ , we choose $\lambda ^ { \prime } = \frac { \lambda } { \omega }$ , and derive

$$
\bar { x } + \lambda ^ { \prime } \omega y = \bar { x } + \lambda y \in \mathcal { C } .
$$

Therefore, $\omega y \in D c \left( \bar { x } \right)$ for all $\omega \geq 0$ , and $D _ { \cal C } ( \bar { x } )$ is a cone.

Next, we prove that $D _ { \cal C } ( \bar { x } )$ is convex. Let $y , z \in D c \left( \bar { x } \right)$ . By definition,

$$
\exists \eta , \omega > 0 { \mathrm { ~ s u c h ~ t h a t ~ } } \bar { x } + \eta y , \bar { x } + \omega z \in \mathcal { C } .
$$

Assume without loss of generality that $\eta \leq \omega$ . By the convexity of ${ \mathcal { C } } ,$

$$
\bar { x } + \eta z \in \mathcal { C } .
$$

Let $\lambda \in ( 0 , 1 )$ . By the convexity of ${ \mathcal { C } } ,$

$$
\lambda \left( { \bar { x } } + \eta y \right) + \left( 1 - \lambda \right) \left( { \bar { x } } + \eta z \right) = { \bar { x } } + \eta \left( \lambda y + \left( 1 - \lambda \right) z \right) \in { \mathcal { C } } .
$$

Therefore,

$$
\lambda y + ( 1 - \lambda ) z \in D \mathcal { C } ( \bar { x } ) .
$$

Proof of Lemma 2.2. We first show that for $S \subseteq \mathbb { R } ^ { n }$ , it holds that af $( S ) = \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$

If $S = \emptyset$ , then cl $( \mathrm { S } ) = \emptyset$ , and the afine set of an empty set is af $( \varnothing ) = \varnothing$ . Hence, the claim is trivially true.

Otherwise, assume that $S \neq \emptyset$ . The inclusion af $( S ) \subseteq \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$ is trivial since $S \subseteq \mathrm { c l } \left( \mathrm { S } \right)$ In the opposite direction, let $y \in \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$ . Therefore, there exist $\bar { x } _ { 1 } , \ldots , \bar { x } _ { k } \in \mathrm { c l } \left( \mathrm { S } \right)$ , and scalars $a _ { 1 } , \ldots , a _ { k }$ such that $\sum _ { i = 1 } ^ { k } a _ { i } = 1$ and

$$
y = \sum _ { i = 1 } ^ { k } a _ { i } { \bar { x } } _ { i } .
$$

Since $\bar { x } _ { i } \in \mathrm { c l } \left( \mathrm { S } \right)$ , for each i there exists a sequence $\left\{ x _ { i } ^ { j } \right\} _ { j \geq 0 } \subseteq S$ such that $x _ { i } ^ { j } \xrightarrow { j  \infty } \bar { x } _ { i }$ . Let $y ^ { j } = \sum _ { i = 1 } ^ { k } a _ { i } x _ { i } ^ { j }$ . By its construction, $y ^ { j } \in \mathrm { a f f } \left( S \right)$ for all $j$ . Since af (S) is a closed set, and $y ^ { j } \xrightarrow { j  \infty } y$ , it follows that $y \in \operatorname { a f f } \left( S \right)$

We now move to prove that for $\bar { x } , \bar { y } \in S$ it holds that af $( S - { \bar { x } } ) = \operatorname { a f f } \left( S - { \bar { y } } \right)$ . Let $z \in \mathrm { a f f } \left( S - \bar { x } \right)$ . Our goal is to show that $z \in \mathrm { a f f } \left( S - \bar { y } \right)$ , that is, to construct a set of vectors $u _ { 1 } , \ldots , u _ { l } \in S - { \bar { y } }$ and coeficients $b _ { 1 } , \dots , b _ { l } \in \mathbb { R }$ such that $\sum _ { i = 1 } ^ { l } b _ { i } = 1$ and $z = \sum _ { i = 1 } ^ { l } b _ { i } u _ { i } .$

Since $z \in \mathrm { a f f } \left( S - \bar { x } \right)$ , there exist $v _ { 1 } , \dotsc , v _ { k } \in S - { \bar { x } }$ and $a _ { 1 } , \dots , a _ { k } \in \mathbb { R }$ , such that $\sum _ { i = 1 } ^ { k } a _ { i } = 1$ and

$$
z = \sum _ { i = 1 } ^ { k } a _ { i } v _ { i } .
$$

By definition of $S - { \bar { x } }$ , the fact that $v _ { 1 } , \dotsc , v _ { k } \in S - { \bar { x } }$ implies $v _ { 1 } + { \bar { x } } , \dots , v _ { k } + { \bar { x } } \in S$ . Hence, by definition of $S - { \bar { y } }$ , we have that $0 , { \bar { x } } - { \bar { y } } \in S - { \bar { y } }$ , and $v _ { 1 } + { \bar { x } } - { \bar { y } } , \dots , v _ { k } + { \bar { x } } - { \bar { y } } \in S - { \bar { y } }$ Therefore,

$$
z = \sum _ { i = 1 } ^ { k } a _ { i } v _ { i } = 1 \cdot 0 - 1 \cdot \left( { \bar { x } } - { \bar { y } } \right) + \sum _ { i = 1 } ^ { k } a _ { i } \left( v _ { i } + { \bar { x } } - { \bar { y } } \right) .
$$

Since the coeficients $1 , - 1 , a _ { 1 } , \ldots , a _ { k }$ sum to 1, it follows that $z \in \mathrm { a f f } \left( S - \bar { y } \right)$ , and consequently af $( S - { \bar { x } } ) \subseteq \operatorname { a f f } \left( S - { \bar { y } } \right)$ . The reverse inclusion af $( S - \bar { y } ) \subseteq \mathrm { a f f } \left( S - \bar { x } \right)$ is proved in the same way.

Now assume S is convex. Nonemptiness of ri (S) and the line segment principle are [30, Proposition 2.40]. We show ri $\left( \mathrm { S } \right) = \mathrm { r i } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$ . Let $y \in \operatorname { r i } \left( \operatorname { S } \right)$ . Then there exists $\epsilon > 0$ such that

$$
B ( y , \epsilon ) \cap \mathrm { a f f } \left( S \right) \subseteq S .
$$

Since af $( S ) = \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$ and $S \subseteq \mathrm { c l } \left( \mathrm { S } \right)$ , it follows that

$$
B ( y , \epsilon ) \cap \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right) \subseteq \mathrm { c l } \left( \mathrm { S } \right) .
$$

Since $y \in S \subseteq \operatorname { c l } \left( \mathrm { S } \right)$ , it follows that $y \in \mathrm { r i } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$

Let $x \in { \mathrm { r i } } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$ . By definition, there exists $\epsilon > 0$ such that

$$
\mathcal { B } ( x , \epsilon ) \cap \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right) = \mathcal { B } ( x , \epsilon ) \cap \mathrm { a f f } \left( S \right) \subseteq \mathrm { c l } \left( \mathrm { S } \right) .
$$

Let $y \in \mathrm { r i } \left( \mathrm S \right)$ . If $x = y ,$ then $x \in \mathrm { r i } \left( \mathrm S \right)$ and we are done. Otherwise, $\| x - y \| > 0$ . Let $\delta = \frac { \mathsf { c } } { 2 \left\| x - y \right\| }$ , and define

$$
z = \left( 1 + \delta \right) x - \delta y .
$$

Since $x , y \in \operatorname { c l } \left( \mathrm { S } \right)$ , it clearly follows that $z \in \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right) = \mathrm { a f f } \left( S \right)$ . Additionally, $\| z - x \| =$ $\| \left( 1 + \delta \right) x - \delta y - x \| = \delta \left\| y - x \right\| \leq 0 . 5 \epsilon$ , and therefore, $z \in B ( x , \epsilon )$

Now, $z \in { \mathcal { B } } ( x , \epsilon ) \cap \operatorname { a f f } \left( \operatorname { c l } \left( \operatorname { S } \right) \right)$ , and since $x \in { \mathrm { r i } } \left( \mathrm { c l } \left( \mathrm { S } \right) \right)$ , it follows that $B ( x , \epsilon ) \cap \mathrm { a f f } \left( \mathrm { c l } \left( \mathrm { S } \right) \right) \subseteq$ cl (S). Combining both results, we deduce that $z \in \mathrm { c l } \left( \mathrm { S } \right)$ . Together with the equation

$$
x = \frac { 1 } { 1 + \delta } z + \frac { \delta } { 1 + \delta } y ,
$$

and the fact that $y \in \operatorname { r i } \left( \operatorname { S } \right)$ , this in turn implies $x \in { \mathrm { r i } } \left( \mathrm S \right)$ by the line segment principle.

## B Proofs of Section 6

Proof of Lemma 6.1. Write $h ^ { \prime } ( \bar { x } ; d ) = \langle \nabla f ( \bar { x } ) , d \rangle - g ^ { \prime } ( \bar { x } ; d )$ . The assumption $\mathcal { O } ^ { \gamma } ( \bar { x } ; V , r ) = 0$ means $\mu ^ { \gamma } ( \bar { x } ; v , r ) = 0$ for every $v \in V$ , so

$$
\begin{array} { r } { h ( \bar { x } ) \ \leq \ h ( \bar { x } + t v ) + \frac { \gamma } { 2 } t ^ { 2 } \qquad \mathrm { f o r ~ a l l ~ } t \in \left[ 0 , \operatorname* { m i n } \{ \tau ( \bar { x } ; v ) , r \} \right] . } \end{array}
$$

Since $V \subseteq D c ( \bar { x } )$ , each $v \in V$ satisfies $\tau ( \bar { x } ; v ) > 0$ by Definition 2.3. Combined with $r > 0$ the feasible interval has positive length. For every $t \in ( 0$ , min $\{ \tau ( \bar { x } ; v ) , r \} ]$ ，

$$
{ \frac { h ( { \bar { x } } + t v ) - h ( { \bar { x } } ) } { t } } + { \frac { \gamma } { 2 } } t \geq 0 .
$$

Sending $t \to 0 ^ { + }$ yields $h ^ { \prime } ( \bar { x } ; v ) \geq 0 ;$ the directional derivative exists and is finite because $f$ is continuously diferentiable and $g : \mathbb { R } ^ { n } $ R is convex [29, Theorem $2 3 . 1 ]$

Now let $d \in \kappa$ . Then $\begin{array} { r } { d = \sum _ { v \in V } \lambda _ { v } v } \end{array}$ for some $\lambda _ { v } \geq 0$ . The map $g ^ { \prime } ( \bar { x } ; \cdot )$ is sublinear, hence

$$
\begin{array} { r l } & { h ^ { \prime } ( \bar { x } ; d ) = \displaystyle \sum _ { v } \lambda _ { v } \langle \nabla f ( \bar { x } ) , v \rangle - g ^ { \prime } \Big ( \bar { x } ; \displaystyle \sum _ { v } \lambda _ { v } v \Big ) } \\ & { \qquad \quad \geq \displaystyle \sum _ { v } \lambda _ { v } \big ( \langle \nabla f ( \bar { x } ) , v \rangle - g ^ { \prime } ( \bar { x } ; v ) \big ) = \displaystyle \sum _ { v } \lambda _ { v } h ^ { \prime } ( \bar { x } ; v ) \geq 0 . } \end{array}
$$

Proof of Lemma $6 . 2 .$ Since $\mathcal { C }$ is compact and $\nabla f$ is continuous, $\operatorname* { s u p } _ { x \in { \mathcal { C } } } \| \nabla f ( x ) \| < \infty$ . The set $\partial g$ is locally bounded on the interior of dom $g _ { ; }$ which contains a neighborhood of ${ \mathcal { C } } ,$ hence $\partial g$ is uniformly bounded on C and $\Lambda < \infty$

Write $h ^ { \prime } ( \bar { x } ; d ) = \langle \nabla f ( \bar { x } ) , d \rangle - g ^ { \prime } ( \bar { x } ; d )$ . The map $g ^ { \prime } ( \bar { x } ; \cdot )$ is the support function of $\partial g ( \bar { x } )$ , so

$$
\left| g ^ { \prime } ( \bar { x } ; d ) - g ^ { \prime } ( \bar { x } ; d ^ { \prime } ) \right| \ = \ \left| \sigma _ { \partial g ( \bar { x } ) } ( d ) - \sigma _ { \partial g ( \bar { x } ) } ( d ^ { \prime } ) \right| \ \leq \ \operatorname* { m a x } _ { s \in \partial g ( \bar { x } ) } \| s \| \ \left\| d - d ^ { \prime } \right\| \ .
$$

The linear map $d \mapsto \langle \nabla f ( { \bar { x } } ) , d \rangle$ is Lipschitz with modulus $\| \nabla f ( { \bar { x } } ) \|$ . Adding the two moduli,

$$
\left| h ^ { \prime } ( \bar { x } ; d ) - h ^ { \prime } ( \bar { x } ; d ^ { \prime } ) \right| \leq \Big ( \| \nabla f ( \bar { x } ) \| + \operatorname* { m a x } _ { s \in \partial g ( \bar { x } ) } \| s \| \Big ) \left\| d - d ^ { \prime } \right\| \leq \Lambda \left\| d - d ^ { \prime } \right\| .
$$

Finally $h ^ { \prime } ( \bar { x } ; 0 ) = 0$ , so $\left| h ^ { \prime } ( \bar { x } ; d ) \right| = \left| h ^ { \prime } ( \bar { x } ; d ) - h ^ { \prime } ( \bar { x } ; 0 ) \right| \le \Lambda \left\| d \right\|$ . This concludes part 1.

For part 2, $\langle \nabla f ( { \bar { x } } ) , \cdot \rangle$ is linear and $g ^ { \prime } ( { \bar { x } } ; \cdot )$ is sublinear, so

$$
h ^ { \prime } ( \bar { x } ; d ) = \sum _ { v } \lambda _ { v } \langle \nabla f ( \bar { x } ) , v \rangle - g ^ { \prime } \Big ( \bar { x } ; \sum _ { v } \lambda _ { v } v \Big ) \ \geq \ \sum _ { v } \lambda _ { v } \big ( \langle \nabla f ( \bar { x } ) , v \rangle - g ^ { \prime } ( \bar { x } ; v ) \big ) = \sum _ { v } \lambda _ { v } h ^ { \prime } ( \bar { x } ; v ) .
$$

Negate and use $- h ^ { \prime } \leq [ - h ^ { \prime } ] _ { + }$

Proof of Proposition 6.1. The bound is per-direction, so it is enough to treat a single $v \in V$ and sum the squares. Write $\begin{array} { r } { \phi ( t ) : = f ( \bar { x } + t v ) + \frac { \gamma } { 2 } t ^ { 2 } } \end{array}$ for $t \geq 0$ , so $\mu ^ { \gamma } ( \bar { x } ; v , r )$ is the smallest minimizer of $\phi$ on $\left[ 0 , \tau _ { r } ( \bar { x } ; v ) \right]$ , and $\phi ^ { \prime } ( t ) = \langle \nabla f ( \bar { x } + t v ) , v \rangle + \gamma t$

If $\langle \nabla f ( { \bar { x } } ) , v \rangle \geq 0$ , this v contributes 0 to the right-hand side of (6.3), and $\mu ^ { \gamma } ( \bar { x } ; v , r ) \ge 0$ is immediate.

Now assume $\langle \nabla f ( { \bar { x } } ) , v \rangle < 0$ , and set $t ^ { \star } : = - \langle \nabla f ( \bar { x } ) , v \rangle / ( L + \gamma ) > 0 . \mathrm { ~ F o r ~ } 0 \leq t < t ^ { \star }$

$$
\begin{array} { r l } { \phi ^ { \prime } ( t ) \ \leq \ | \phi ^ { \prime } ( t ) - \phi ^ { \prime } ( 0 ) | + \phi ^ { \prime } ( 0 ) } & { } \\ { \leq \ | \langle \nabla f ( \bar { x } + t v ) , v \rangle - \langle \nabla f ( \bar { x } ) , v \rangle | + \gamma t + \phi ^ { \prime } ( 0 ) } & { } \\ { \leq \ ( L + \gamma ) t + \phi ^ { \prime } ( 0 ) \ < \ 0 , } \end{array}
$$

where the third inequality uses $f \in C _ { L } ^ { 1 , 1 }$ and $\lVert \boldsymbol { v } \rVert = 1$ , so $| \langle \nabla f ( { \bar { x } } + t v ) - \nabla f ( { \bar { x } } ) , v \rangle | \leq L t$ Thus $\phi$ is strictly decreasing on $[ 0 , t ^ { \star } ]$ , so every minimizer of $\phi$ on $\left[ 0 , \tau _ { r } ( \bar { x } ; v ) \right]$ is at least min $\{ t ^ { \star } , \tau _ { r } ( \bar { x } ; v ) \}$ . Therefore

$$
\mu ^ { \gamma } ( \bar { x } ; v , r ) \geq \operatorname* { m i n } \{ t ^ { \star } , \tau _ { r } ( \bar { x } ; v ) \} = \frac { 1 } { L + \gamma } \operatorname* { m i n } \{ - \langle \nabla f ( \bar { x } ) , v \rangle , ( L + \gamma ) \tau _ { r } ( \bar { x } ; v ) \} .
$$

Combining with the nonnegative inner-product case,

$$
\mu ^ { \gamma } ( \bar { x } ; v , r ) \geq \frac { 1 } { L + \gamma } \operatorname* { m i n } \{ [ - \langle \nabla f ( \bar { x } ) , v \rangle ] _ { + } , ( L + \gamma ) \tau _ { r } ( \bar { x } ; v ) \}
$$

for every $v \in V$ . Squaring and summing over V gives (6.3).

Proof of Lemma 6.3. If $\theta = 0$ there is nothing to prove; assume $\theta > 0$ , so $h ^ { \prime } ( \bar { x } ; v ) = - \theta$ . For $t \in [ 0 , \tau _ { r } ( \bar { x } ; v ) ]$ the segment $[ { \bar { x } } , { \bar { x } } + t v ] \subseteq C$ . Since $\bar { \boldsymbol { f } } \in C _ { L } ^ { 1 , 1 }$ and $\lVert \boldsymbol { v } \rVert = 1$

$$
\begin{array} { r } { f ( \bar { x } + t v ) \ \leq \ f ( \bar { x } ) + t \langle \nabla f ( \bar { x } ) , v \rangle + \frac { L } { 2 } t ^ { 2 } . } \end{array}
$$

Convexity of g gives $g ( \bar { x } + t v ) \geq g ( \bar { x } ) + t g ^ { \prime } ( \bar { x } ; v )$ for $t \geq 0$ . Subtracting and adding the regularizer,

$$
\begin{array} { r l } & { h ( \bar { x } + t v ) + \frac { \gamma } { 2 } t ^ { 2 } - h ( \bar { x } ) \ \leq \ t \big ( \langle \nabla f ( \bar { x } ) , v \rangle - g ^ { \prime } ( \bar { x } ; v ) \big ) + \frac { L + \gamma } { 2 } t ^ { 2 } } \\ & { \qquad = \ t h ^ { \prime } ( \bar { x } ; v ) + \frac { L + \gamma } { 2 } t ^ { 2 } \ = \ - \theta t + \frac { L + \gamma } { 2 } t ^ { 2 } \ = : \ \psi ( t ) . } \end{array}
$$

Hence

$$
\begin{array} { r l r } & { } & { h ( \bar { x } ) - \underset { 0 \leq t \leq \tau _ { r } ( \bar { x } ; v ) } { \operatorname* { m i n } } \Bigl \{ h ( \bar { x } + t v ) + \frac { \gamma } { 2 } t ^ { 2 } \Bigr \} \ \geq \ - \ \underset { 0 \leq t \leq \tau _ { r } ( \bar { x } ; v ) } { \operatorname* { m i n } } \psi ( t ) . } \end{array}
$$

The derivative $\psi ^ { \prime } ( t ) = - \theta + ( L + \gamma ) t$ vanishes at $t ^ { \star } = \theta / ( L + \gamma )$ , and

$$
\psi ( t ^ { \star } ) = - \frac { \theta ^ { 2 } } { L + \gamma } + \frac { L + \gamma } { 2 } \cdot \frac { \theta ^ { 2 } } { ( L + \gamma ) ^ { 2 } } = - \frac { \theta ^ { 2 } } { 2 ( L + \gamma ) } .
$$

If $t ^ { \star } \leq \tau _ { r } ( \bar { x } ; v )$ then m $\mathrm { i n } _ { [ 0 , \tau _ { r } ( \bar { x } ; v ) ] } \psi = \psi ( t ^ { \star } )$ , and the claim follows. If $t ^ { \star } > \tau _ { r } ( \bar { x } ; v )$ then $\psi$ is decreasing on $[ 0 , \tau _ { r } ( \dot { \bar { x } } ; v ) ]$ , so the minimum is $\psi ( \tau _ { r } ( \bar { x } ; v ) ) ,$ ). Since $\tau _ { r } ( \bar { x } ; v ) < t ^ { \star }$ we have $\tau _ { r } ( \bar { x } ; v ) ^ { 2 } < \tau _ { r } ( \bar { x } ; v ) t ^ { \star }$ , and therefore

$$
\begin{array} { l } { \psi ( \tau _ { r } ( \bar { x } ; v ) ) = - \theta \tau _ { r } ( \bar { x } ; v ) + \frac { L + \gamma } { 2 } \tau _ { r } ( \bar { x } ; v ) ^ { 2 } } \\ { < - \theta \tau _ { r } ( \bar { x } ; v ) + \frac { L + \gamma } { 2 } \tau _ { r } ( \bar { x } ; v ) t ^ { \star } } \\ { ~ = - \theta \tau _ { r } ( \bar { x } ; v ) + \frac { 1 } { 2 } \theta \tau _ { r } ( \bar { x } ; v ) } \\ { ~ = - \frac { 1 } { 2 } \theta \tau _ { r } ( \bar { x } ; v ) . } \end{array}
$$

Thus the left-hand side is greater than $\frac { 1 } { 2 } \theta \tau _ { r } ( \bar { x } ; v )$

Proof of Lemma 6.4. Write $r _ { \mathrm { i n } } = 1 / \kappa _ { \mathrm { m a x } } , r _ { \mathrm { o u t } } = 1 / \kappa _ { \mathrm { m i n } }$ . Let $\hat { x } \in \mathrm { r b } \left( \mathcal { C } \right)$ with inward unit normal u, and let $\nu = - u$ be the outer unit normal at xˆ, unique since C has a $C ^ { 2 }$ boundary. Recall that the support function of a nonempty $\mathcal { X } \subseteq \mathbb { R } ^ { n }$ is $h _ { \mathcal { X } } ( w ) \ = \ \operatorname* { s u p } _ { x \in \mathcal { X } } \left. x , w \right.$ . Let $\mathcal { B } _ { \mathrm { i n } } : = \mathcal { B } ( \hat { x } + r _ { \mathrm { i n } } u , r _ { \mathrm { i n } } )$ . Since all principal curvatures of $B _ { \mathrm { i n } }$ equal $\kappa _ { \mathrm { m a x } }$ , they upper bound those of C at every pair of boundary points sharing an outer unit normal. By [32, Theorem 2.5.4], $h c - h g _ { \mathrm { i n } } $ is a support function, so by [32, Theorem 1.7.1] there is a unique convex body M with $h _ { M } = h _ { \} } - h _ { B _ { \mathrm { i n } } }$ . By [32, Theorem 1.7.5] and uniqueness, $M + B _ { \mathrm { i n } } = \mathcal { C }$ . We claim $0 \in M ;$ then $B _ { \mathrm { i n } } = \{ 0 \} + B _ { \mathrm { i n } } \subseteq M + B _ { \mathrm { i n } } = \mathcal { C }$ . Indeed there are $\bar { m } \in M$ and $\bar { b } \in B _ { \mathrm { i n } }$ with ${ \bar { m } } + { \bar { b } } = { \hat { x } }$ 2 and

$$
h _ { \mathcal { C } } ( \nu ) = \left. \hat { x } , \nu \right. = \left. \bar { b } , \nu \right. + \left. \bar { m } , \nu \right. \leq h _ { \mathcal { B } _ { \mathrm { i n } } } ( \nu ) + h _ { M } ( \nu ) = h _ { \mathcal { C } } ( \nu ) ,
$$

so equality holds throughout. In particular $\langle \bar { b } , \nu \rangle = h _ { B _ { \mathrm { i n } } } ( \nu )$ , so <sup>¯</sup>b maximizes $\langle b , \nu \rangle$ over $b \in B _ { \mathrm { i n } }$ This maximizer is unique and equal to xˆ: the center of $B _ { \mathrm { i n } }$ is $\hat { x } + r _ { \mathrm { i n } } u = \hat { x } - r _ { \mathrm { i n } } \nu ,$ so the unique farthest point of the ball in the direction ν is $( { \hat { x } } - r _ { \mathrm { i n } } \nu ) + r _ { \mathrm { i n } } \nu = { \hat { x } }$ . Thus $\bar { b } = \hat { x }$ , and therefore $\bar { m } = 0$

Now set $B _ { \mathrm { o u t } } : = B ( \hat { x } + r _ { \mathrm { o u t } } u , r _ { \mathrm { o u t } } )$ . The principal curvatures of $ { { B _ { \mathrm { o u t } } } }$ are constantly $\kappa _ { \mathrm { m i n } } .$ hence they lower bound those of C at every pair of boundary points sharing an outer unit normal. By the same comparison [32, Theorem 2.5.4], $h _ { B _ { \mathrm { o u t } } } - h _ { C }$ is a support function, so by [32, Theorem 1.7.1] there is a unique convex body M<sup>′</sup> with $h _ { M ^ { \prime } } = h _ { B _ { \mathrm { o u t } } } - h _ { C }$ , and as above $M ^ { \prime } + \mathcal { C } = \mathcal { B } _ { \mathrm { o u t } }$ . We claim $0 \in M ^ { \prime } ;$ then $\mathcal { C } = \{ 0 \} + \mathcal { C } \subseteq M ^ { \prime } + \mathcal { C } = \mathcal { B } _ { \mathrm { o u t } }$ . Since $\hat { \boldsymbol { v } } \in B _ { \mathrm { o u t } }$ , there are $\bar { m } ^ { \prime } \in { \cal M } ^ { \prime }$ and ${ \bar { c } } \in { \mathcal { C } }$ with $\bar { m } ^ { \prime } + \bar { c } = \hat { x }$ , and

$$
h _ { \mathcal { B } _ { \mathrm { o u t } } } ( \nu ) = \left. \hat { x } , \nu \right. = \left. \bar { c } , \nu \right. + \left. \bar { m } ^ { \prime } , \nu \right. \leq h _ { \mathcal { C } } ( \nu ) + h _ { M ^ { \prime } } ( \nu ) = h _ { \mathcal { B } _ { \mathrm { o u t } } } ( \nu ) ,
$$

so equality holds throughout. In particular $\langle \bar { c } , \nu \rangle = h _ { \cal { C } } ( \nu )$ . But xˆ itself attains $h _ { C } ( \nu )$ , because ν is an outer unit normal to $\mathcal { C }$ at xˆ. This maximizer is unique: $\kappa _ { \operatorname* { m i n } } > 0$ forces C to be strictly convex, so a supporting hyperplane meets C at exactly one point. Thus $\bar { c } = \hat { x }$ and $\bar { m } ^ { \prime } = 0$ We conclude

$$
B _ { \mathrm { i n } } \subseteq \mathcal { C } \subseteq B _ { \mathrm { o u t } } ,
$$

which is part 1. The outer inclusion implies that C is bounded, hence compact.

For part 2, if $\rho ( x ) = 0$ then $x \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ and $\Pi ( x ) = \{ x \}$ . Assume $0 < \rho ( x ) < r _ { \mathrm { i n } }$ . The set rb (C) is compact, so $\Pi ( x )$ is nonempty. Let $y \in \Pi ( x )$ and let u be the inward unit normal at $y .$ Write $\rho : = \rho ( x )$ , so $x = y + \rho u$ , and set $z : = y + r _ { \mathrm { i n } } u$ . Part 1 applied at y gives $B ( z , r _ { \mathrm { i n } } ) \subseteq \mathcal { C }$ The ball of radius ρ about x is contained in the ball of radius $r _ { \mathrm { i n } }$ about z: for every w with $\| w - x \| \leq \rho ,$

$$
\left\| w - z \right\| ~ \leq ~ \left\| w - x \right\| + \left\| x - z \right\| ~ \leq ~ \rho + ( r _ { \mathrm { i n } } - \rho ) ~ = ~ r _ { \mathrm { i n } } .
$$

Equality holds throughout if and only if $\| w - x \| = \rho$ and $w , x , z$ are collinear with x between z and $w .$ , which forces $w = x - \rho u = y$ . Thus the two balls share exactly one boundary point, namely y. Since $B ( x , \rho ) \subseteq B ( z , r _ { \mathrm { i n } } ) \subseteq \mathcal { C }$ , every other point of $B ( x , \rho )$ lies in the open ball of radius $r _ { \mathrm { i n } }$ about $z ,$ hence in ri $( \mathcal { C } )$ . So $B ( x , \rho )$ meets rb (C) only at $y .$ . Every point of $\Pi ( x )$ lies on rb (C) and on the sphere of radius ρ about x, so $\Pi ( x ) = \{ y \}$ ✷

Proof of Lemma 6.5. Write $r _ { \mathrm { i n } } = 1 / \kappa _ { \mathrm { m a x } }$ and $r _ { \mathrm { o u t } } = 1 / \kappa _ { \mathrm { m i n } }$ , and let $\mathcal { B } _ { \mathrm { i n } } : = \mathcal { B } ( \hat { x } + r _ { \mathrm { i n } } u , r _ { \mathrm { i n } } )$ and $B _ { \mathrm { o u t } } : = B ( \hat { x } + r _ { \mathrm { o u t } } u , r _ { \mathrm { o u t } } )$ . By Lemma $6 . 4 , B _ { \mathrm { i n } } \subseteq \mathcal { C } \subseteq B _ { \mathrm { o u t } }$

For part 1, let $z : = \hat { x } + r _ { \mathrm { i n } } u , \mathrm { s o } \bar { x } - z = - ( r _ { \mathrm { i n } } - \rho ) u$ . Then $\begin{array} { r } { \bar { x } + t d \in \mathcal { B } _ { \mathrm { i n } } \mathrm { ~ i f f ~ } \Vert t d - ( r _ { \mathrm { i n } } - \rho ) u \Vert ^ { 2 } \leq } \end{array}$ $r _ { \mathrm { i n } } ^ { 2 } , \mathrm { i . e . }$

$$
t ^ { 2 } - 2 t ( r _ { \mathrm { i n } } - \rho ) c \leq r _ { \mathrm { i n } } ^ { 2 } - ( r _ { \mathrm { i n } } - \rho ) ^ { 2 } = 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } ,
$$

whose largest admissible t is the largest root of the quadratic $t ^ { 2 } - 2 t ( r _ { \mathrm { i n } } - \rho ) c - ( 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } )$ namely $( r _ { \mathrm { i n } } - \rho ) c + \sqrt { ( r _ { \mathrm { i n } } - \rho ) ^ { 2 } c ^ { 2 } + 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } }$ ; since $B _ { \mathrm { i n } } \subseteq { \mathcal { C } }$ this lower bounds $\tau ( \bar { x } ; d )$

For (1a), first take $\begin{array} { r } { 0 \leq c \leq \frac { 1 } { 2 } } \end{array}$ . The inequality

$$
\displaystyle ( r _ { \mathrm { i n } } - \rho ) c + \sqrt { ( r _ { \mathrm { i n } } - \rho ) ^ { 2 } c ^ { 2 } + 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } } ~ \geq ~ 2 r _ { \mathrm { i n } } c
$$

is equivalent, after isolating the square root and squaring (both sides being nonnegative), to $( r _ { \mathrm { i n } } - \rho ) ^ { 2 } c ^ { 2 } + 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } \geq ( \bar { r } _ { \mathrm { i n } } + \rho ) ^ { \bar { 2 } } c ^ { 2 }$ , i.e. to $2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } \geq 4 r _ { \mathrm { i n } } \rho c ^ { 2 }$ , i.e. to $c ^ { 2 } \leq ( 2 r _ { \mathrm { i n } } - \rho ) / ( 4 r _ { \mathrm { i n } } )$ which holds since $\rho \leq r _ { \mathrm { i n } }$ gives $( 2 r _ { \mathrm { i n } } - \rho ) / ( 4 r _ { \mathrm { i n } } ) \geq \frac { 1 } { 4 }$ . The stated expression is nondecreasing in c, so for $\begin{array} { r } { c > \frac { 1 } { 2 } } \end{array}$ it is at least its value at $\begin{array} { r } { c = \frac { 1 } { 2 } } \end{array}$ , hence $\tau ( \bar { x } ; d ) \ge r _ { \mathrm { i n } }$ . Thus (1a) holds in all cases.

For (1b), drop the nonnegative c-terms and use $2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } \ge r _ { \mathrm { i n } } \rho$ for $\rho \leq r _ { \mathrm { i n } }$ . For (1c), substituting $c = 1 \ { \mathrm { g i v e s } } \ ( r _ { \mathrm { i n } } - \rho ) + { \sqrt { ( r _ { \mathrm { i n } } - \rho ) ^ { 2 } + 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } } } = ( r _ { \mathrm { i n } } - \rho ) + r _ { \mathrm { i n } } = 2 r _ { \mathrm { i n } } - \rho \geq r _ { \mathrm { i n } } .$ since $( r _ { \mathrm { i n } } - \rho ) ^ { 2 } + 2 r _ { \mathrm { i n } } \rho - \rho ^ { 2 } = r _ { \mathrm { i n } } ^ { 2 } ,$

For part 2, the same computation with $\mathcal { C } \subseteq B _ { \mathrm { o u t } }$ gives $\tau ( \bar { x } ; d ) \leq ( r _ { \mathrm { o u t } } - \rho ) c + \sqrt { ( r _ { \mathrm { o u t } } - \rho ) ^ { 2 } c ^ { 2 } + X }$ with $X : = 2 r _ { \mathrm { o u t } } \rho - \rho ^ { 2 } \geq 0$ . Put $a : = ( r _ { \mathrm { o u t } } - \rho ) | c | \geq 0$ and use $\sqrt { a ^ { 2 } + X } \leq a + \sqrt { X }$ . For $c \geq 0$ this gives $\tau ( \bar { x } ; d ) \leq 2 ( r _ { \mathrm { o u t } } - \rho ) c + \sqrt { X } \leq 2 r _ { \mathrm { o u t } } c + \sqrt { 2 r _ { \mathrm { o u t } } \rho } ;$ for $c < 0$ the leading term $\mathrm { i s } - a$ so $\tau ( \bar { x } ; d ) \leq \sqrt { X } \leq \sqrt { 2 r _ { \mathrm { o u t } } \rho } .$ . Combining the two cases, $\tau ( \bar { x } ; d ) \leq 2 r _ { \mathrm { o u t } } [ c ] _ { + } + \sqrt { 2 r _ { \mathrm { o u t } } \rho } .$

For part 3, since $\tau ( \bar { x } ; d ) = \operatorname* { s u p } \{ t \geq 0 : \bar { x } + t d \in \mathcal { C } \}$ , any condition necessary for ${ \bar { x } } + t d \in { \mathcal { C } }$ gives an upper bound on $\tau ( \bar { x } ; d )$ . As u is the inward unit normal at ${ \hat { x } } \in \operatorname { r b } \left( { \mathcal { C } } \right)$ , the hyperplane through xˆ orthogonal to u supports ${ \mathcal { C } } ,$ so ${ \mathcal { C } } \subseteq H : = \{ y : \langle y - { \hat { x } } , u \rangle \geq 0 \}$ . Since $\bar { x } = \hat { x } + \rho u$ with $\lVert u \rVert = 1$ we have $\langle \bar { x } - \hat { x } , u \rangle = \rho ,$ hence

$$
\langle ( \bar { x } + t d ) - \hat { x } , u \rangle = \rho + t \langle d , u \rangle = \rho + t c ,
$$

and ${ \bar { x } } + t d \in { \mathcal { C } } \subseteq H$ forces $\rho + t c \ge 0$ . For $c < 0$ this is equivalent to $t \leq \rho / | c |$ , so $\tau ( \bar { x } ; d ) \leq \rho / | c |$ Taking $d = - u \ ( \mathrm { s o \ } c = - 1 )$ gives $\tau ( \bar { x } ; - u ) \le \rho$ . The reverse inequality follows from convexity: ${ \hat { x } } , { \bar { x } } \in { \mathcal { C } }$ , so the segment $[ \bar { x } , \hat { x } ] = \{ \bar { x } + t ( - u ) : 0 \leq t \leq \rho \}$ lies in C. Hence $\tau ( \bar { x } ; - u ) = \rho$ ✷

Proof of Corollary 6.1. This is Lemma 6.5 at $\rho = 0$ . By Corollary 3.1(b) the $\eta { - } \mathrm { F D }$ condition on a unit d reads $c = \langle d , u \rangle \geq \eta ,$ and part 1 at $\rho = 0$ gives $\tau ( \hat { x } ; d ) \geq r _ { \mathrm { i n } } c + \sqrt { r _ { \mathrm { i n } } ^ { 2 } c ^ { 2 } } = 2 r _ { \mathrm { i n } } c \geq$ $2 \eta / \kappa _ { \mathrm { m a x } }$ . If d is not an $\eta { - } \mathrm { F D }$ then $c < \eta ;$ if $c < 0$ then $\tau ( \hat { x } ; d ) = 0$ by part 3, and if $0 \leq c < \eta$ then part 2 at $\rho = 0 ~ \mathrm { g i v e s } ~ \tau ( \hat { x } ; d ) \leq 2 r _ { \mathrm { o u t } } c \leq 2 \eta / \kappa _ { \mathrm { m i n } }$ ✷

Proof of Proposition 6.2. Set $\bar { \eta } : = \eta / ( 2 \sqrt { \mathrm { d i m } { \mathscr C } } )$ . Fix $x \in { \mathcal { C } }$ and a unit vector $d \in D c ( x )$ with $\tau ( x ; d ) \ge R$ . Let $V ( x )$ denote the non-exploration output of Algorithm 2 at $\alpha = 0$ and $\rho _ { 0 } = r _ { \mathrm { i n } } / 2$ . Under the assumptions of Lemma 6.4 the boundary is $C ^ { 2 }$ , so the second branch of Algorithm 2 is never taken. If $\rho > \rho _ { 0 }$ then $T = \emptyset$ and $| U | = 2$ dim C, and Step 3 with $\alpha = 0$ returns U, hence $| V ( x ) | = 2 \dim { \mathcal { C } }$ . If $\rho ~ \leq ~ \rho _ { 0 }$ then $| T | = 2 ( \dim { \mathcal { C } } - 1 )$ and $| U | \le 2$ dim C; Step 3 with $\alpha = 0$ returns one direction for each element of T and of U, hence $| V ( x ) | = | T | + | U | \leq$ 4 dim $\mathcal { C } - 2$ . In all cases $| V ( x ) | \leq 4 \dim { \mathcal { C } }$

Deep interior $( \rho > \rho _ { 0 } = r _ { \mathrm { i n } } / 2 )$ . Write $\begin{array} { r } { d = \sum _ { i } a _ { i } w _ { i } } \end{array}$ and take $\lambda _ { \mathrm { s g n } ( a _ { i } ) w _ { i } } = | a _ { i } |$ . Then $\Sigma \lambda = \| a \| _ { 1 } \leq { \sqrt { \dim { \mathcal { C } } } }$ by Cauchy–Schwarz, hence $A = { \sqrt { \dim { \mathcal { C } } } }$ , and $\tau ( x ; v ) \geq \rho > r _ { \mathrm { i n } } / 2$ for every unit v, so $\tau _ { r } ( x ; v ) \geq \operatorname* { m i n } \{ r _ { \mathrm { i n } } / 2 , r \}$ and

$$
\sum _ { v } \frac { \lambda _ { v } } { \tau _ { r } ( x ; v ) } \leq \sqrt { \dim \mathcal { C } } \big ( \frac { 1 } { r } + 2 \kappa _ { \operatorname* { m a x } } \big ) .
$$

This bound is independent of $\eta .$

Boundary or shallow interior $( \rho \le \rho _ { 0 } )$ . Decompose $d = c u + \textstyle \sum _ { i = 1 } ^ { \dim \mathcal { C } - 1 } a _ { i } w _ { i } , c = \langle d , u \rangle$ $c ^ { 2 } + \textstyle \sum a _ { i } ^ { 2 } = 1$ , hence $\begin{array} { r } { | c | + \sum _ { i } | a _ { i } | \le \sqrt { \dim { \mathcal C } } } \end{array}$ by Cauchy–Schwarz.

Inward $t i l t { : } ~ c \geq \eta .$ , use the tilted set. Let $\bar { v } _ { i }$ be the normalization of $\bar { \eta } u + ( 1 - \bar { \eta } ) \mathrm { s g n } ( a _ { i } ) w _ { i }$ and write $\varrho : = \| { \bar { \eta } } u + ( 1 - { \bar { \eta } } ) \operatorname { s g n } ( a _ { i } ) w _ { i } \| \in [ { \frac { 1 } { \sqrt { 2 } } } , 1 ]$ , so $\varrho \bar { v } _ { i } = \bar { \eta } u + ( 1 - \bar { \eta } ) \mathrm { s g n } ( a _ { i } ) w _ { i }$ . Set

$$
\tilde { c } : = c - \frac { \bar { \eta } } { 1 - \bar { \eta } } \sum _ { j } | a _ { j } | , \qquad \tilde { a } _ { i } : = \frac { a _ { i } } { 1 - \bar { \eta } } .
$$

Then

$$
\begin{array} { r l r } {  { \sum _ { i } | \widetilde { a } _ { i } | \varrho \bar { v } _ { i } = \sum _ { i } \frac { | a _ { i } | } { 1 - \bar { \eta } } ( \bar { \eta } u + ( 1 - \bar { \eta } ) \mathrm { s g n } ( a _ { i } ) w _ { i } ) } } \\ & { } & { = \frac { \bar { \eta } } { 1 - \bar { \eta } } \bigl ( \sum _ { i } | a _ { i } | \bigr ) u + \sum _ { i } a _ { i } w _ { i } , } \end{array}
$$

and therefore $\begin{array} { r } { d = { \tilde { c } } u + \sum _ { i } | { \tilde { a } } _ { i } | \varrho { \bar { v } } _ { i } } \end{array}$ . Since $\bar { \eta } \leq \frac { 1 } { 2 }$ and $\textstyle \sum _ { j } | a _ { j } | \leq { \sqrt { \dim { \mathcal { C } } } }$

$$
\frac { \bar { \eta } } { 1 - \bar { \eta } } \sum _ { j } | a _ { j } | \ \le \ 2 \bar { \eta } \sqrt { \mathrm { d i m } \mathcal { C } } \ = \ \eta \ \le \ c ,
$$

so $\tilde { c } \geq 0$ . Using $\lambda _ { u } = \tilde { c }$ and $\lambda _ { \bar { v } _ { i } } = | \tilde { a } _ { i } | \varrho$

$$
\sum \lambda = { \tilde { c } } + \sum _ { i } | { \tilde { a } } _ { i } | \varrho \leq { \tilde { c } } + \sum _ { i } | { \tilde { a } } _ { i } | = c - { \frac { \bar { \eta } } { 1 - \bar { \eta } } } \sum | a _ { j } | + { \frac { 1 } { 1 - \bar { \eta } } } \sum | a _ { i } | = c + \sum _ { i } | a _ { i } | \leq { \sqrt { \mathrm { d i m } \mathscr { C } } } ,
$$

since $\varrho \le 1$ and $\begin{array} { r } { \frac { 1 } { 1 - \bar { \eta } } - \frac { \bar { \eta } } { 1 - \bar { \eta } } = 1 } \end{array}$ . Hence $A = { \sqrt { \dim { \mathcal { C } } } } ,$ independently of $\eta .$ . For the Bsum, write $c _ { i } : = \langle \bar { v } _ { i } , \dot { u } \rangle = \vec { \eta } / \varrho .$ Since $\varrho \le 1$ we have $c _ { i } \geq \bar { \eta } \geq 0$ , so Lemma $6 . 5 ( 1 \mathrm { a } )$ gives $\tau ( x ; \bar { v } _ { i } ) \geq \operatorname* { m i n } \{ 2 c _ { i } , 1 \} r _ { \mathrm { i n } }$ . Because $c _ { i } \geq \bar { \eta }$ and $\bar { \eta } \leq \frac 1 2$ , we have min $\{ 2 c _ { i } , 1 \} \ge 2 \bar { \eta } .$ , hence $\tau ( x ; \bar { v } _ { i } ) \geq 2 \bar { \eta } r _ { \mathrm { i n } } = 2 \bar { \eta } / \kappa _ { \mathrm { m a x } } = \eta / ( \sqrt { \dim { \mathscr C } } \kappa _ { \mathrm { m a x } } )$ . Also $\tau ( x ; u ) \ge r _ { \mathrm { i n } }$ by (1c). Hence

$$
\sum \frac { \lambda _ { v } } { \tau _ { r } ( x ; v ) } \leq \frac { \sqrt { \dim { \mathscr { C } } } } { r } + \sqrt { \dim { \mathscr { C } } } . \frac { \sqrt { \dim { \mathscr { C } } } \kappa _ { \operatorname* { m a x } } } { \eta } = \frac { \sqrt { \dim { \mathscr { C } } } } { r } + \frac { \dim { \mathscr { C } } \kappa _ { \operatorname* { m a x } } } { \eta } .
$$

Untilted: $c < \eta _ { ; }$ , use the untilted set U. First $\rho$ is bounded below. If $c \geq 0$ , Lemma 6.5(2) gives $R \leq \tau ( x ; d ) \leq 2 r _ { \mathrm { o u t } } c + \sqrt { 2 r _ { \mathrm { o u t } } \rho } < 2 r _ { \mathrm { o u t } } \eta + \sqrt { 2 r _ { \mathrm { o u t } } \rho } ;$ since $R = 8 r _ { \mathrm { o u t } } \eta$ this forces $\sqrt { 2 r _ { \mathrm { o u t } } \rho } > 6 r _ { \mathrm { o u t } } \eta .$ , i.e.

$$
\rho > 1 8 r _ { \mathrm { o u t } } \eta ^ { 2 } .
$$

If $c < 0$ , the same lemma gives $\tau ( x ; d ) \leq \sqrt { 2 r _ { \mathrm { o u t } } \rho } ,$ so $R \leq \sqrt { 2 r _ { \mathrm { o u t } } \rho }$ and therefore

$$
\rho \ge 3 2 r _ { \mathrm { o u t } } \eta ^ { 2 } .
$$

In either subcase $\rho > 1 8 r _ { \mathrm { o u t } } \eta ^ { 2 }$ , so U is the full set of 2 dim C signed basis vectors and lies in $D _ { \mathcal { C } } ( x )$ . Decompose d with $\lambda _ { \operatorname { s g n } ( c ) u } = | c | , \lambda _ { \operatorname { s g n } ( a _ { i } ) w _ { i } } = | a _ { i } | ; \sum \lambda = | c | + \sum | a _ { i } | \leq { \sqrt { \dim { \mathscr { C } } } }$ , hence $A = { \sqrt { \dim { \mathcal { C } } } }$ . For the B-sum:

$\langle \pm w _ { i } , u \rangle = 0$ , so Lemma $6 . 5 ( 1 \mathrm { b } ) \mathrm { g i v e s } \tau ( x ; \pm w _ { i } ) \geq \sqrt { r _ { \mathrm { i n } } \rho } > \eta \sqrt { 1 8 } r _ { \mathrm { i n } } r _ { \mathrm { o u t } }$ . Hence $\begin{array} { r } { \sum _ { i } | a _ { i } | / \tau ( x ; \pm w _ { i } ) \le \sqrt { \mathrm { d i m } \mathcal { C } } \sqrt { \kappa _ { \mathrm { m a x } } \kappa _ { \mathrm { m i n } } } / ( \sqrt { 1 8 } \eta ) \le \sqrt { \mathrm { d i m } \mathcal { C } } \kappa _ { \mathrm { m a x } } / ( 4 \eta ) } \end{array}$

$\mathrm { I f } ~ c \ge 0 \colon \tau ( x ; u ) \ge r _ { \mathrm { i n } } ~ \mathrm { s o } ~ c / \tau ( x ; u ) \le \eta \kappa _ { \mathrm { m a x } } \le \kappa _ { \mathrm { m a x } } .$

• If $c < 0 \colon \tau ( x ; - u ) = \rho$ and Lemma 6.5(3) gives $| c | \leq \rho / \tau ( x ; d ) \leq \rho / R , \mathrm { ~ s o ~ } | c | / \tau ( x ; - u ) \leq$ $1 / R = \kappa _ { \mathrm { m i n } } / ( 8 \eta )$

The last two subcases are mutually exclusive, so their contributions enter as a maximum. Adding the $1 / r$ truncation $\Sigma \lambda / r \leq \sqrt { \dim { \mathscr { C } } } / r$

$$
\begin{array} { r l r } {  { \sum _ { \frac { \lambda _ { v } } { \tau _ { r } ( x ; v ) } } \le \frac { \sqrt { \dim \mathscr { C } } } { r } + \frac { \sqrt { \dim \mathscr { C } } \kappa _ { \operatorname* { m a x } } } { 4 \eta } + \operatorname* { m a x } \Bigl \{ \kappa _ { \operatorname* { m a x } } , \frac { \kappa _ { \operatorname* { m i n } } } { 8 \eta } \Bigr \} } } \\ & { } & { \le \frac { \sqrt { \dim \mathscr { C } } } { r } + \frac { ( \dim \mathscr { C } + 1 ) \kappa _ { \operatorname* { m a x } } } { \eta } \le \frac { \sqrt { \dim \mathscr { C } } } { r } + \frac { 2 \dim \mathscr { C } \kappa _ { \operatorname* { m a x } } } { \eta } . } \end{array}
$$

The second inequality uses ${ \sqrt { \dim C } } / 4 \leq$ dim C and max $\{ \kappa _ { \mathrm { m a x } } , \kappa _ { \mathrm { m i n } } / ( 8 \eta ) \} \le \kappa _ { \mathrm { m a x } } / \eta ,$ , the latter since $\begin{array} { r } { \eta \le { \frac { 1 } { 2 } } } \end{array}$ and $\kappa _ { \mathrm { m i n } } \leq \kappa _ { \mathrm { m a x } } ;$ the third uses dim $\mathcal { C } + 1 \leq 2$ dim C.

Thus $A = { \sqrt { \dim { \mathcal { C } } } }$ in each case. The first-branch bound is $B \le \sqrt { \dim \mathcal { C } } ( 1 / r + 2 \kappa _ { \operatorname* { m a x } } )$ independent of η; the inward-tilt bound is $B \leq \sqrt { \dim \mathcal { C } } / r + \dim \mathcal { C } \kappa _ { \operatorname* { m a x } } / \eta ;$ the untilted bound is $B \le \sqrt { \dim \mathcal { C } } / r + 2 \dim \mathcal { C } \kappa _ { \mathrm { m a x } } / \eta$ . The last of these is the largest, and is the bound of the proposition. ✷

Proof of Lemma 6.6. (1) Every unit d has $\tau ( x ; d ) \geq \rho \geq r _ { \mathrm { i n } } \geq R$ , so no direction is excluded. (2)–(3), inward case. Let d be a unit feasible direction with $c : = \langle d , u \rangle \geq 0$ , put $s : =$ $R \kappa _ { \operatorname* { m a x } } \in ( 0 , 1 ]$ and $w : = ( d + s u ) / \| d + s u \|$ . Since $\left\| d + s u \right\| ^ { 2 } = 1 +$ 2sc + $s ^ { 2 }$ and $c \geq 0$ , we have $\| d + s u \| \in [ 1 , 1 + s ]$ , hence

$$
\langle w , u \rangle \ = \ \frac { c + s } { \| d + s u \| } \ \geq \ \frac { s } { 1 + s } \ \geq \ \frac { s } { 2 } \ = \ \frac { R \kappa _ { \operatorname* { m a x } } } { 2 } .
$$

By Lemma $\begin{array} { r } { 6 . 5 ( 1 \mathrm { a } ) , \tau ( x ; w ) \ge \operatorname* { m i n } \{ 2 \langle w , u \rangle , 1 \} r _ { \mathrm { i n } } . \mathrm { I f } \langle w , u \rangle \ge \frac 1 2 } \end{array}$ this is at least $r _ { \mathrm { i n } } \geq R ;$ if $\begin{array} { r } { \langle w , u \rangle < \frac { 1 } { 2 } } \end{array}$ it is at least $2 \langle w , u \rangle r _ { \mathrm { i n } } \geq R$ . Thus w is admissible in $\Delta ^ { [ R ] }$ . Also $\| w - d \| \leq$ $| 1 - | | d + \bar { s } u | | | + s \leq 2 s = 2 R \kappa _ { \operatorname* { m a x } }$ , so by Lemma 6.2(1)

$$
- h ^ { \prime } ( x ; d ) \ \leq \ - h ^ { \prime } ( x ; w ) + \Lambda \| d - w \| \ \leq \ \Delta ^ { [ R ] } ( x ) + 2 \Lambda \kappa _ { \operatorname* { m a x } } R .
$$

At a boundary point cl $( \mathrm { D } c ( \mathrm { x } ) ) = \{ d : \langle d , u \rangle \geq 0 \}$ , so this covers every feasible direction and gives (2).

(3), outward case: C lies in the supporting halfspace at the nearest boundary point, so $\rho + t \langle d , u \rangle \geq 0$ is necessary for feasibility, which is Lemma $6 . 5 ( 3 ) . \ \mathrm { I f } \ \tau ( x ; d ) \ge R$ then d is included in $\Delta ^ { [ R ] } ( x )$ ; otherwise $\tau ( x ; d ) < R$ , so d is excluded from $\Delta ^ { [ R ] } ( x )$ ✷

## B.1 Proof of Theorem 6.2

The proof of Theorem 6.2 follows that of [16, Theorem 4.1] and combines four ingredients. First, the suficient decrease of the line searches forces the exploration step sizes $t _ { \mathrm { e x } } ^ { k }$ to vanish. Second, almost surely every direction in the relative interior of the feasible-direction cone at an accumulation point is approximated by exploration directions along some subsequence (Lemma B.3). Third, along such a subsequence the approximating directions admit a uniform lower bound on the feasibility radius (Lemma B.2). Fourth, vanishing steps together with that bound force non-descent in the limit (Lemma B.1). This gives $h ^ { \prime } ( x ^ { * } ; v ) \ge 0$ on ri $( \mathrm { D } _ { \cal C } ( \mathrm { x } ^ { \ast } ) )$ , which Corollary B.1 extends to the whole cone. From this point onward, let

$$
L : = \mathrm { a f f } \left( \mathcal { C } - \bar { x } \right) ,
$$

for any fixed ${ \bar { x } } \in { \mathcal { C } } .$ , so $L$ is the linear subspace parallel to af (C). This definition is independent of the choice of ${ \bar { x } } .$ , because af $( { \mathcal { C } } - { \bar { x } } ) = \mathrm { a f f } \left( { \mathcal { C } } - { \bar { y } } \right)$ for all ${ \bar { x } } , { \bar { y } } \in { \mathcal { C } }$ (Lemma 2.2). In particular, af $( { \mathcal { C } } ) - x ^ { k } = L$ for every iterate $x ^ { k } \in { \mathcal { C } }$ . We write $\mathbb { S } _ { L }$ for the unit sphere of L.

Lemma B.1 (Vanishing line-search steps; cf. the proof of [16, Theorem $4 . 1 ( \mathrm { i i } ) ] )$ ). Let $\{ x _ { j } \} _ { j \in \mathcal { I } }$ be a sequence in C converging to $x ^ { * } \in { \mathcal { C } }$ , and let $\{ w ^ { j } \} _ { j \in \mathcal { I } }$ be a sequence of unit vectors in L converging to some $w ^ { \ast } \in \mathbb { R } ^ { n }$ . Suppose that:

(i) there exists $c > 0$ such that $\tau ( x _ { j } ; w ^ { j } ) \ge c$ for all $j \in \mathcal { I }$ ;

(ii) $t ^ { j } \to 0$ as $j \to \infty$ over J , where for each $j \in \mathcal { I }$

$$
\begin{array} { r } { t ^ { j } \in \underset { t \in [ 0 , \operatorname* { m i n } \{ \tau ( x _ { j } ; w ^ { j } ) , r \} ] } { \mathrm { a r g m i n } } h ( x _ { j } + t w ^ { j } ) + \frac { \gamma } { 2 } t ^ { 2 } , } \end{array}
$$

where $\gamma > 0$ . Then $h ^ { \prime } ( x ^ { * } ; w ^ { * } ) \ge 0$

Proof. Assume for contradiction that $h ^ { \prime } ( x ^ { * } ; w ^ { * } ) = - \epsilon < 0$ . By the definition of the directional derivative, there exists $\tau _ { 0 } > 0$ such that

$$
\begin{array} { r } { h ( x ^ { * } + t w ^ { * } ) - h ( x ^ { * } ) < - \frac { \epsilon } { 2 } t \qquad \forall t \in ( 0 , \tau _ { 0 } ] . } \end{array}\tag{B.1}
$$

By local Lipschitz continuity of h at $x ^ { * }$ , there exist $R _ { h } ~ > ~ 0$ and $L _ { h } > 0$ such that h is L<sub>h</sub>-Lipschitz on $B ( x ^ { * } , R _ { h } )$

Define

$$
t ^ { * } = \operatorname* { m i n } \Bigl \{ c , \ \tau _ { 0 } , \ \frac { \epsilon } { 4 \gamma } , \ r , \ \frac { R _ { h } } { 4 } \Bigr \} .
$$

By Assumption (i), for all $j$ suficiently large, $t ^ { * } \le c \le \tau ( x _ { j } ; w ^ { j } )$ and $t ^ { * } \leq r$ , so $t ^ { * } \in$ [0, min $\{ \tau ( x _ { j } ; w ^ { j } ) , \ r \} \ ]$

Since $x _ { j } \to x ^ { * }$ and $w ^ { j } \to w ^ { * }$ , there exists $J _ { 1 } \in \mathbb { N }$ such that for $j > J _ { 1 } , j \in \mathcal { I }$

$$
\left\| x _ { j } - x ^ { * } \right\| < \operatorname * { m i n } \Bigl \{ \frac { R _ { h } } { 4 } , \frac { \epsilon t ^ { * } } { 1 6 L _ { h } } \Bigr \} , \qquad \left\| w ^ { j } - w ^ { * } \right\| < \frac { \epsilon } { 1 6 L _ { h } } .
$$

For such $j ,$ we have

$$
\| ( x ^ { * } + t ^ { * } w ^ { * } ) - x ^ { * } \| = t ^ { * } \leq \frac { R _ { h } } { 4 } < R _ { h } ,
$$

and, since $\| w ^ { j } \| = 1$ 2

$$
\left\| \left( x _ { j } + t ^ { * } w ^ { j } \right) - x ^ { * } \right\| \leq \left\| x _ { j } - x ^ { * } \right\| + t ^ { * } \left\| w ^ { j } \right\| < { \frac { R _ { h } } { 4 } } + { \frac { R _ { h } } { 4 } } = { \frac { R _ { h } } { 2 } } < R _ { h } .
$$

Hence both points $x _ { j } + t ^ { * } w ^ { j }$ and $x ^ { * } + t ^ { * } w ^ { * }$ lie in $B ( x ^ { * } , R _ { h } )$ , where the $L _ { h ^ { - } } \mathrm { I }$ ipschitz bound is valid, and therefore

$$
\begin{array} { r } { | h ( x _ { j } + t ^ { * } w ^ { j } ) - h ( x ^ { * } + t ^ { * } w ^ { * } ) | \leq L _ { h } \left. ( x _ { j } - x ^ { * } ) + t ^ { * } ( w ^ { j } - w ^ { * } ) \right. , } \end{array}
$$

and, by the triangle inequality together with the bounds above,

$$
| h ( x _ { j } + t ^ { * } w ^ { j } ) - h ( x ^ { * } + t ^ { * } w ^ { * } ) | \leq \frac { \epsilon t ^ { * } } { 8 } , \qquad | h ( x ^ { * } ) - h ( x _ { j } ) | \leq \frac { \epsilon t ^ { * } } { 8 } .
$$

For every $t \in [ 0 , \epsilon / ( 4 \gamma ) ]$ , we have

$$
\frac { \gamma t ^ { 2 } } { 2 } \leq \frac { \epsilon t } { 8 } .
$$

Combining with (B.1) at $t = t ^ { * }$ (valid since $t ^ { * } \leq \tau _ { 0 } )$

$$
\begin{array} { r l } & { h ( x _ { j } + t ^ { * } w ^ { j } ) - h ( x _ { j } ) + \displaystyle \frac { \gamma ( t ^ { * } ) ^ { 2 } } { 2 } = \big [ h ( x _ { j } + t ^ { * } w ^ { j } ) - h ( x ^ { * } + t ^ { * } w ^ { * } ) \big ] + \big [ h ( x ^ { * } + t ^ { * } w ^ { * } ) - h ( x ^ { * } ) \big ] } \\ & { \qquad + \left[ h ( x ^ { * } ) - h ( x _ { j } ) \right] + \displaystyle \frac { \gamma ( t ^ { * } ) ^ { 2 } } { 2 } } \\ & { \qquad \leq \displaystyle \frac { \epsilon t ^ { * } } { 8 } - \frac { \epsilon t ^ { * } } { 2 } + \frac { \epsilon t ^ { * } } { 8 } + \frac { \epsilon t ^ { * } } { 8 } \leq - \frac { \epsilon t ^ { * } } { 8 } . } \end{array}
$$

Since $t ^ { j } \to 0$ by Assumption (ii), there exists $J _ { 2 } \in \mathbb { N }$ such that for $j > J _ { 2 } , j \in \mathcal { I }$

$$
t ^ { j } < \mathrm { m i n } \bigg \{ \frac { R _ { h } } { 4 } , \frac { \epsilon t ^ { * } } { 8 L _ { h } } , \frac { \epsilon } { 4 \gamma } \bigg \} .
$$

For such j (and $j > J _ { 1 } )$ , we have $\| x _ { j } - x ^ { * } \| < R _ { h } / 4$ and $\| w ^ { j } \| = 1$ , so

$$
\left\| \left( x _ { j } + t ^ { j } w ^ { j } \right) - x ^ { * } \right\| \leq \| x _ { j } - x ^ { * } \| + t ^ { j } \left\| w ^ { j } \right\| < { \frac { R _ { h } } { 4 } } + { \frac { R _ { h } } { 4 } } = { \frac { R _ { h } } { 2 } } < R _ { h } .
$$

Thus both $x _ { j } + t ^ { j } w ^ { j }$ and $x _ { j }$ belong to $B ( x ^ { * } , R _ { h } )$ , and the Lipschitz bound gives

$$
h ( x _ { j } + t ^ { j } w ^ { j } ) - h ( x _ { j } ) \geq - L _ { h } t ^ { j } ,
$$

Since $\begin{array} { r } { \frac { \gamma ( t ^ { j } ) ^ { 2 } } { 2 } \geq 0 } \end{array}$ , we obtain

$$
h ( x _ { j } + t ^ { j } w ^ { j } ) - h ( x _ { j } ) + \frac { \gamma ( t ^ { j } ) ^ { 2 } } { 2 } \geq h ( x _ { j } + t ^ { j } w ^ { j } ) - h ( x _ { j } ) \geq - L _ { h } t ^ { j } > - \frac { \epsilon t ^ { \ast } } { 8 } .
$$

Since $t ^ { * } \in \mathsf { \Gamma } [ 0$ , min $\{ \tau ( x _ { j } ; w ^ { j } ) , r \} ]$ , optimality of $t ^ { j }$ requires $\begin{array} { r } { h ( x _ { j } + t ^ { j } w ^ { j } ) + \frac { \gamma } { 2 } ( t ^ { j } ) ^ { 2 } \leq h ( x _ { j } + } \end{array}$ $t ^ { * } w ^ { j } ) + \textstyle \frac { \gamma } { 2 } ( t ^ { * } ) ^ { 2 }$ . But the estimates above give $\begin{array} { r } { h ( x _ { j } + t ^ { j } w ^ { j } ) + \frac { \gamma } { 2 } ( t ^ { j } ) ^ { 2 } > \bar { h } ( x _ { j } ) - \epsilon t ^ { * } / 8 \geq } \end{array}$ $\begin{array} { r } { h ( x _ { j } + t ^ { * } \overline { { w ^ { j } } } ) + \frac { \gamma } { 2 } ( t ^ { * } ) ^ { 2 } } \end{array}$ , a contradiction. ✷

Applying Lemma B.1 requires a uniform lower bound on the feasibility radius $\tau ( x _ { j } ; w ^ { j } )$ along the sequence of directions. The following lemma provides such a bound whenever the limit direction $w ^ { * }$ lies in the relative interior of $D _ { \cal C } ( x ^ { * } )$

Lemma B.2 (Uniform step size bound; cf. [16, Lemma 4.3]). Let $\{ x _ { j } \} _ { j \in \mathcal { I } } \subseteq \mathcal { C }$ with $x _ { j } \to x ^ { * }$ and let $\{ w ^ { j } \} _ { j \in \mathcal { I } } \subseteq L$ with $w ^ { j } \to w ^ { * }$ , where $w ^ { * } \in \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$ . Then there exists $\bar { \epsilon } > 0$ and $J _ { 0 } \in \mathbb { N }$ such that

$$
x _ { j } + \bar { \epsilon } w ^ { j } \in \mathcal { C } \qquad \forall j \in \mathcal { I } , \ j \geq J _ { 0 } .
$$

In particular, $w ^ { j } \in D _ { \mathcal { C } } ( x _ { j } )$ and $\tau ( x _ { j } ; w ^ { j } ) \ge \bar { \epsilon }$ for all $j \in \mathcal { I }$ suficiently large.

Proof. Since $w ^ { * } \in \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$ and af $( D c ( x ^ { * } ) ) = L$ , there exists $r > 0$ such that $B ( w ^ { \ast } , r ) \cap L \subseteq$ $D _ { \cal C } ( x ^ { * } )$ . Choose dim $L { + } 1$ directions $d _ { 0 } , \dots , d _ { \dim L } \in \mathcal { B } ( w ^ { * } , r ) \cap L$ such that conv $\{ d _ { 0 } , \dots , d _ { \mathrm { d i m } L } \}$ is a dim L-dimensional simplex in L with $w ^ { * } \in \mathrm { r i } ( \mathrm { c o n v } \{ \mathrm { d } _ { 0 } , \dots , \mathrm { d } _ { \mathrm { d i m L } } \} )$ . Since each $d _ { i } \in$ $D _ { \cal C } ( x ^ { * } )$ , by definition, there exist $\epsilon _ { i } > 0$ with $x ^ { * } + \epsilon _ { i } d _ { i } \in \mathcal { C }$ . Set $\bar { \epsilon } = \operatorname* { m i n } _ { i } \epsilon _ { i } > 0$ . Since $\bar { \epsilon } \leq \epsilon _ { i }$ and C is convex (with $x ^ { \ast } \in \mathcal { C } )$ , each $p _ { i } : = x ^ { * } + \bar { \epsilon } d _ { i } \in \mathcal { C }$

For any $d \in \operatorname { c o n v } \{ d _ { 0 } , \dots , d _ { \dim L } \}$ , write $d = \textstyle \sum _ { i = 0 } ^ { \dim L } \lambda _ { i } d _ { i }$ with $\begin{array} { r } { \lambda _ { i } \geq 0 , \sum _ { i = 0 } ^ { \dim L } \lambda _ { i } = 1 } \end{array}$ . Then $\begin{array} { r } { x ^ { * } + \bar { \epsilon } d = \sum _ { i = 0 } ^ { \dim L } \lambda _ { i } p _ { i } \in \mathcal { C } } \end{array}$ by convexity.

Since $w ^ { \ast } \in$ ri (conv $\{ \mathrm { d } _ { 0 } , \dots , \mathrm { d } _ { \mathrm { d i m L } } \} )$ , there exists $\rho > 0$ with $B ( w ^ { * } , \rho ) \cap L \subseteq \mathrm { c o n v } \{ d _ { 0 } , \dotsc , d _ { \dim L } \}$ In particular, $x ^ { * } + { \bar { \epsilon } } d \in { \mathcal { C } }$ for every $d \in B ( w ^ { * } , \rho ) \cap L .$ so $B ( x ^ { * } + { \bar { \epsilon } } w ^ { * } , { \bar { \epsilon } } \rho ) \cap \operatorname { a f f } \left( { \mathcal { C } } \right) \subseteq { \mathcal { C } } , { \mathrm { i . e . } }$ ${ x ^ { * } + \bar { \epsilon } w ^ { * } \in \mathrm { r i } \left( \mathcal { C } \right) }$

Since $w ^ { j } \in L = \operatorname { a f f } \left( \mathcal { C } - x ^ { * } \right)$ , we have $x _ { j } + \bar { \epsilon } w ^ { j } \in \arg \left( \mathcal { C } \right)$ . Moreover,

$$
\| ( x _ { j } + \bar { \epsilon } w ^ { j } ) - ( x ^ { * } + \bar { \epsilon } w ^ { * } ) \| \leq \| x _ { j } - x ^ { * } \| + \bar { \epsilon } \| w ^ { j } - w ^ { * } \| < \bar { \epsilon } \rho
$$

for all $j$ suficiently large. Therefore $x _ { j } + \bar { \epsilon } w ^ { j } \in \mathcal { C }$ for all $j \geq J _ { 0 }$ for some $J _ { 0 } \in \mathbb { N } .$ ✷

Remark B.1. The condition $w ^ { * } \in \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$ cannot be relaxed to $w ^ { * } \in D c ( x ^ { * } )$ in general. For directions on the relative boundary of $D c ( x ^ { * } ) , \tau ( x _ { j } ; w ^ { j } )$ may tend to zero. When $x ^ { * } \in \mathrm { r i } \left( \mathcal { C } \right)$ this issue does not arise: $D c ( x ^ { * } ) = \arg { ( \mathcal { C } - x ^ { * } ) }$ is a subspace, so ri $( \mathrm { D } c ( \mathbf { x } ^ { * } ) ) = D c ( x ^ { * } )$ and $w ^ { * } \in \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$ holds for every limit direction.

Fix a countable dense set $\mathcal { Q } \subseteq \mathbb { S } _ { L }$ . The following lemma establishes that, almost surely, every direction from Q that is feasible at an accumulation point can be realized as a limit of exploration directions.

Lemma B.3 (Exploration directions approximate all feasible directions; adapted from [16, Lemma 4.2]). Almost surely, for every accumulation point $x ^ { * }$ of $\{ x _ { k } \}$ and every $v \in$ Q ∩ ri $( \mathrm { D } _ { \cal C } ( \mathrm { x } ^ { \ast } ) )$ , there exists a subsequence $\mathcal { I } \subseteq \mathbb { N }$ such that $x _ { j } \to x ^ { * }$ and $v _ { \mathrm { e x } } ^ { j }  v \ a s \ j  \infty$ $j \in \mathcal { I }$

Proof. Let $\mathcal { F } _ { k - 1 }$ denote the σ-algebra generated by $\{ x ^ { 0 } , v _ { \mathrm { e x } } ^ { 0 } , \ldots , x ^ { k - 1 } , v _ { \mathrm { e x } } ^ { k - 1 } , x ^ { k } \}$ . Then $x ^ { k }$ is $\mathcal { F } _ { k - 1 } .$ measurable and $v _ { \mathrm { e x } } ^ { k }$ is independent of $\mathcal { F } _ { k - 1 }$

Fix $y \in \mathcal { C } , \delta > 0 .$ and $V : = \boldsymbol { B } [ w , \epsilon ] \cap \mathbb { S } _ { L }$ with $w \in \mathbb { S } _ { L } , \epsilon > 0$ . Since the sampling law of $v _ { \mathrm { e x } } ^ { k }$ has full support on $\mathbb { S } _ { L }$ , the probability $p : = P ( v _ { \mathrm { e x } } ^ { k } \in V )$ satisfies $p > 0$ and is independent of k. Define

$$
A _ { k } : = \{ v _ { \mathrm { e x } } ^ { k } \in V \} \cap \{ \left\| x ^ { k } - y \right\| < \delta \} .
$$

Then

$$
P ( A _ { k } \mid { \mathcal { F } } _ { k - 1 } ) = \mathbf { 1 } _ { \left\{ \left\| x ^ { k } - y \right\| < \delta \right\} } \cdot p .
$$

By the conditional Borel–Cantelli–Lévy lemma [34, Chapter VII, §5, Corollary 2],

$$
\Big \{ \sum _ { k = 0 } ^ { \infty } P ( A _ { k } \mid \mathcal { F } _ { k - 1 } ) = \infty \Big \} = \{ A _ { k } \mathrm { ~ i . o . } \} \quad \mathrm { a . s . }\tag{B.2}
$$

Hence

$$
\left\{ \left\| x ^ { k } - y \right\| < \delta { \mathrm { ~ i . o . } } \right\} \Longrightarrow \left\{ v _ { \mathrm { e x } } ^ { k } \in V , \ \left\| x ^ { k } - y \right\| < \delta { \mathrm { ~ i . o . } } \right\} \quad { \mathrm { a . s . } }
$$

Let $\{ y _ { n } \} _ { n \in \mathcal { N } }$ be a countable dense subset of $\mathcal { C }$ , let $\mathcal { Q } = \{ q _ { i } \} _ { i \in \mathcal { N } }$ , and let $\delta _ { m } , \epsilon _ { l }$ range over positive rationals. For each tuple $\boldsymbol { u } = ( n , i , m , l )$ , define the logical implication event

$$
E _ { u } : = \Big \{ \Big \| x ^ { k } - y _ { n } \Big \| < \delta _ { m } ~ \mathrm { i . o . } \Longrightarrow \big ( v _ { \mathrm { e x } } ^ { k } \in \mathcal { B } [ q _ { i } , \epsilon _ { l } ] , ~ \Big \| x ^ { k } - y _ { n } \Big \| < \delta _ { m } \big ) ~ \mathrm { i . o . } \Big \} .
$$

By (B.2), $P ( E _ { u } ) = 1$ for each u. Set

$$
\Omega _ { 0 } : = \bigcap _ { u } E _ { u } .
$$

Since the index set is countable, $P ( \Omega _ { 0 } ) = 1$

Fix $\omega \in \Omega _ { 0 }$ . If $\{ x ^ { k } ( \omega ) \}$ has no accumulation point, the lemma is vacuous on this path. Otherwise, let $x ^ { * }$ be any accumulation point of $\{ x ^ { k } ( \omega ) \}$ and fix $v \in \mathrm { r i } \left( \mathrm { D } \mathcal { C } ( \mathbf { x } ^ { * } ) \right) \cap \mathbb { S } _ { L }$ . Write $x ^ { k } : = x ^ { k } ( \omega )$ and $v _ { \mathrm { e x } } ^ { k } : = v _ { \mathrm { e x } } ^ { k } ( \omega )$

Let $\eta > 0$ . By density, choose $n , m , i , l$ such that

$$
\| y _ { n } - x ^ { * } \| < \eta / 4 , \qquad \delta _ { m } \in ( \eta / 2 , 3 \eta / 4 ) , \qquad \| q _ { i } - v \| < \eta / 4 , \qquad \epsilon _ { l } < \eta / 4 .
$$

Since $x ^ { * }$ is an accumulation point, $\left\| x ^ { k } - x ^ { * } \right\| < \eta / 4$ for infinitely many k. For such $k$

$$
\left\| x ^ { k } - y _ { n } \right\| \leq \left\| x ^ { k } - x ^ { * } \right\| + \left\| x ^ { * } - y _ { n } \right\| < \eta / 2 < \delta _ { m } .
$$

Therefore $\left\| x ^ { k } - y _ { n } \right\| < \delta _ { m }$ occurs infinitely often. Because $\omega \in E _ { u }$ for $\boldsymbol { u } = ( n , i , m , l )$ , there are infinitely many such k with $v _ { \mathrm { e x } } ^ { k } \in B [ q _ { i } , \epsilon _ { l } ]$ . For these indices,

$$
\left\| x ^ { k } - x ^ { * } \right\| \leq \left\| x ^ { k } - y _ { n } \right\| + \| y _ { n } - x ^ { * } \| < \delta _ { m } + \eta / 4 < \eta ,
$$

and

$$
\left\| v _ { \mathrm { e x } } ^ { k } - v \right\| \leq \left\| v _ { \mathrm { e x } } ^ { k } - q _ { i } \right\| + \left\| q _ { i } - v \right\| < \epsilon _ { l } + \eta / 4 < \eta / 2 < \eta .
$$

Let $\eta _ { q } \downarrow 0$ . For each $q ,$ let $K _ { q }$ be the infinite set of indices satisfying $\left\| x ^ { k } - x ^ { * } \right\| < \eta _ { q }$ and $\left\| v _ { \mathrm { e x } } ^ { k } - v \right\| < \eta _ { q }$ . Choose inductively $k _ { q } \in K _ { q }$ with $k _ { q } > k _ { q - 1 }$ . Then $x ^ { k _ { q } } \to x ^ { * }$ and $v _ { \mathrm { e x } } ^ { k _ { q } } \to v . \ \perp$

The following lemma records the analytic properties of the directional derivative that the proof of Theorem 6.2 relies on.

Lemma B.4 (Properties of the directional derivative; cf. [16, Lemmas 1.1 and 4.4]). The function

$$
d \mapsto h ^ { \prime } ( x ^ { * } ; d ) = \langle \nabla f ( x ^ { * } ) , d \rangle - g ^ { \prime } ( x ^ { * } ; d )
$$

is continuous, positively homogeneous of degree 1, and concave on $\mathbb { R } ^ { n }$

Proof. Since $g$ is convex and finite-valued on a neighborhood of $x ^ { * }$ , the directional derivative $g ^ { \prime } ( x ^ { * } ; d )$ exists and is finite for every $d \in \mathbb { R } ^ { n } \ [ 2 9$ , Theorem 23.1 and Theorem 23.4].

Since map $d \mapsto g ^ { \prime } ( x ^ { * } ; d )$ is convex and positively homogeneous of degree 1 [29, Theorem $2 3 . 1 ]$ , and is finite everywhere on R<sup>n</sup>, it is continuous [29, Theorem 10.1]. The map $d \mapsto \langle \nabla f ( x ^ { * } ) , d \rangle$ is linear. Therefore $h ^ { \prime } ( x ^ { * } ; \cdot ) = \langle \nabla f ( x ^ { * } ) , \cdot \rangle - g ^ { \prime } ( x ^ { * } ; \cdot )$ is continuous (linear minus continuous), positively homogeneous (linear minus positively homogeneous), and concave (linear minus convex). ✷

Continuity of $h ^ { \prime } ( x ^ { \ast } ; \cdot )$ allows the non-descent inequality to be extended beyond ri $( \mathrm { D } _ { \cal C } ( \mathrm { x } ^ { \ast } ) )$ to its closure cl $( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } ^ { \ast } ) )$ , which is the cone relevant for d-stationarity.

Corollary B.1 (Extension from relative interior to tangent cone; cf. [16, Lemma 4.4]). If $h ^ { \prime } ( x ^ { * } ; v ) \geq 0$ for all $v \in \mathrm { r i } \left( \mathrm Ḋ c Ḍ ^ { * } \right) \big )$ , then $h ^ { \prime } ( x ^ { * } ; d ) \geq 0$ for all $d \in \mathrm { c l } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$

Proof. Let $d \in \mathrm { c l } \left( \mathrm D c ( \mathrm { x } ^ { * } ) \right)$ . Since cl $\left( \mathrm { r i } \left( \mathrm { D } \boldsymbol { c } ( \mathbf { x } ^ { * } ) \right) \right) = \mathrm { c l } \left( \mathrm { D } \boldsymbol { c } ( \mathbf { x } ^ { * } ) \right)$ (by Lemma 2.2), there exists a sequence $\{ v _ { j } \} \subseteq \mathrm { r i } \left( \mathrm { D } \mathscr { c } ( \mathrm { x } ^ { * } ) \right)$ with $v _ { j } \ \to \ d$ . By continuity of the directional derivative (Lemma B.4):

$$
h ^ { \prime } ( x ^ { * } ; d ) = \operatorname* { l i m } _ { j \to \infty } h ^ { \prime } ( x ^ { * } ; v _ { j } ) \geq 0 .
$$

We can now prove Theorem 6.2.

Proof of Theorem 6.2. By Lemma B.3, there exists an event $\Omega _ { 0 }$ with $P ( \Omega _ { 0 } ) = 1$ such that, for every $\omega \in \Omega _ { 0 }$ , every accumulation point $x ^ { * }$ of $\{ x _ { k } ( \omega ) \}$ , and every $v \in \mathcal { Q } \cap \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$ there exists a subsequence $\mathcal { I } \subseteq \mathcal { N }$ with $x _ { j } ( \omega )  x ^ { * }$ and $v _ { \mathrm { e x } } ^ { j } ( \omega )  v$ . Fix $\omega \in \Omega _ { 0 }$ and work pathwise. Let $x ^ { * }$ be an arbitrary accumulation point of $\{ x _ { k } ( \omega ) \}$ . As in [16, Theorem $4 . 1 ( \mathrm { i } ) ]$ since the algorithm selects the iterate with the largest decrease, comparing against $t = 0$ gives

$$
\begin{array} { r } { h ( x _ { k } ) - h ( x _ { k + 1 } ) \geq h ( x _ { k } ) - h ( x _ { k } + t _ { \mathrm { e x } } ^ { k } v _ { \mathrm { e x } } ^ { k } ) \geq \frac { \gamma } { 2 } ( t _ { \mathrm { e x } } ^ { k } ) ^ { 2 } . } \end{array}
$$

Summing over all $k \geq 0$ yields

$$
\frac { \gamma } { 2 } \sum _ { k = 0 } ^ { \infty } ( t _ { \mathrm { e x } } ^ { k } ) ^ { 2 } \le \sum _ { k = 0 } ^ { \infty } [ h ( x _ { k } ) - h ( x _ { k + 1 } ) ] \le h ( x _ { 0 } ) - \operatorname* { i n f } _ { x \in \mathcal { C } } h ( x ) < \infty ,
$$

and therefore

$$
t _ { \mathrm { e x } } ^ { k } \to 0 \quad \mathrm { a s } \ k \to \infty .\tag{B.3}
$$

If ri $\left( \mathrm { D } \boldsymbol { c } ( \mathbf { x } ^ { * } ) \right) \cap \mathbb { S } _ { L } = \emptyset$ , then ri $\left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right) = \{ 0 \}$ and $h ^ { \prime } ( x ^ { * } ; 0 ) = 0 .$ , so $h ^ { \prime } ( x ^ { * } ; d ) \ge 0$ for all $d \in \mathrm { { r i } } ( \mathrm { { D } } c ( \mathrm { { x } } ^ { * } ) ) ;$ ; the remainder is immediate. Otherwise, let $\{ v _ { i } \} _ { i = 1 } ^ { \infty }$ be an enumeration of $\mathcal { Q } \cap \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$ . Since ri $\left( \mathrm { D } c ( \mathbf { x } ^ { * } ) \right) \cap \mathbb { S } _ { L }$ is relatively open in $\mathbb { S } _ { L }$ and Q is dense in $\mathbb { S } _ { L } ,$ the set $\{ v _ { i } \} _ { i = 1 } ^ { \infty }$ is dense in ri $\left( \mathrm { D } \boldsymbol { c } ( \mathbf { x } ^ { * } ) \right) \cap \mathbb { S } _ { L }$ . For each $v _ { i } .$ , Lemma B.3 provides a subsequence $\mathcal { I } _ { i } \subseteq \mathbb { N }$ with $x _ { j } \to x ^ { * }$ and $v _ { \mathrm { e x } } ^ { j } \to v _ { i }$ . Since $v _ { i } \in \mathrm { r i } \left( \mathrm { D } c ( \mathrm { x } ^ { * } ) \right)$ and $v _ { \mathrm { e x } } ^ { j } \in L$ by construction, Lemma B.2 provides $\bar { \epsilon } _ { i } > 0$ such that $\tau ( x _ { j } ; v _ { \mathrm { e x } } ^ { j } ) \geq \bar { \epsilon } _ { i }$ for all $j \in \mathcal { I } _ { i }$ suficiently large. The step sizes satisfy $t _ { \mathrm { e x } } ^ { j }  0$ by (B.3). Applying Lemma B.1 yields $h ^ { \prime } ( x ^ { * } ; v _ { i } ) \ge 0$ for all $i \geq 1$ . By continuity of $h ^ { \prime } ( x ^ { \ast } ; \cdot )$ (Lemma B.4) and density of {v } in ri $( \mathrm { D } c ( \mathbf { x } ^ { * } ) ) \cap \mathbb { S } _ { L }$ , the inequality extends to all of ri $( \mathrm { D } c ( \mathbf { x } ^ { * } ) ) \cap \mathbb { S } _ { L }$ , and hence to all of ri $( \mathrm { D } c ( \mathrm { x } ^ { \ast } ) )$ by positive homogeneity (Lemma B.4). By Corollary B.1, $h ^ { \prime } ( x ^ { * } ; d ) \ge 0$ for all d ∈ cl $( \mathrm { D } _ { \cal C } ( \mathrm { x } ^ { \ast } ) )$ , and $x ^ { * }$ is d-stationary. Since $x ^ { * }$ was an arbitrary accumulation point on this fixed $\omega \in \Omega _ { 0 }$ , every accumulation point of $\{ x _ { k } ( \omega ) \}$ is d-stationary. Because $P ( \Omega _ { 0 } ) = 1$ , the claim follows. ✷

## C Measure-Theoretic Foundations for Theorem 7.1

This appendix makes precise the claim, used in Section 7, that nonsmooth boundary points are negligible in the relative boundary. It collects the measure-theoretic and probabilistic background used there, and states and proves four auxiliary results together with the full proof of Theorem 7.1:

1. smoothness of C at a boundary point is preserved by an isometric flattening onto a full-dimensional convex set (Proposition C.4);

2. $\mathcal { C } _ { n s }$ is a Borel set and is $\mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 }$ -null (Proposition C.5);

3. radial projection from a point of C maps $\mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 }$ -null subsets of af (C) to σ-null subsets of the unit sphere $\mathbb { S } _ { L }$ (Lemma C.2);

4. every search direction generated by Algorithm 2 has a conditional law on $\mathbb { S } _ { L }$ that is absolutely continuous with respect to $\sigma$ (Lemma C.3).

The full proof of Theorem 7.1 combines the last three of these facts and appears at the end of the appendix. Throughout, “measure” means a Borel measure unless stated otherwise, $L : = \arg \left( { \mathcal { C } } - x \right)$ for any $x \in { \mathcal { C } }$ (independent of the choice of x by Lemma 2.2), and dim $\mathcal { C } \geq 1$

Notation.

$$
\begin{array} { r } { \mathbb { S } _ { L } : = \{ \omega \in L : \| \omega \| = 1 \} \qquad \mathrm { u n i t ~ s p h e r e ~ o f } \ L } \end{array}
$$

$$
\begin{array} { r l } { \Pi _ { x } ( y ) : = ( y - x ) / \left. y - x \right. } & { { } \mathrm { r a d i a l ~ p r o j e c t i o n ~ f r o m ~ } x , \mathrm { f o r ~ } y \in ( x + L ) \setminus \left\{ x \right\} } \end{array}
$$

$$
\mu \mid A \qquad { \mathrm { r e s t r i c t i o n : ~ } } \left( \mu \mid A \right) ( B ) = \mu ( A \cap B )
$$

$\lambda _ { M } \ \left( \mathrm { r e s p . } \ \lambda _ { L } \right)$ Lebesgue measure on an afine (resp. linear) subspace

$$
\sigma : = \mathcal { H } ^ { \dim \mathcal { C } - 1 } \lceil \mathbb { S } _ { L }
$$

$$
\mathbb { S } _ { L }
$$

$$
\mu
$$

$$
{ \mathcal { L } } ( X ) , { \mathcal { L } } ( X \mid { \mathcal { G } } )
$$

$$
X ;
$$

$$
\mathcal { G }
$$

## Lebesgue measure on an afine subspace.

Definition C.1. Let $M = p + L _ { 0 } \subseteq \mathbb { R } ^ { n }$ be an afine subspace of dimension m, with direction subspace $L _ { 0 }$ and base point $p \in M$ . For any linear isometry $\iota : \mathbb { R } ^ { m }  L _ { 0 }$ , the m-dimensional Lebesgue measure on M, denoted $\lambda _ { M }$ , is the pushforward of $\lambda _ { m }$ on $\mathbb { R } ^ { m }$ under $x \mapsto p + \iota ( x )$ It is independent of the choice of p and ι, and satisfies $\lambda _ { M } = \mathcal { H } ^ { m } \harpoonright M$ . We write $\lambda _ { L }$ when $M = L$ is linear.

A measure or random vector has a density on M (equivalently, is absolutely continuous on M) means it is absolutely continuous with respect to $\lambda _ { M }$

Absolute continuity, density, and pushforward.

Definition C.2. For measures $\mu , \nu$ on the same measurable space, $\mu \ll \nu$ if $\nu ( B ) = 0 \Rightarrow$ $\mu ( B ) = 0$ for every measurable B. When $\mu , \nu$ are σ-finite, this is equivalent (Radon–Nikodym; [31]) to the existence of a density $\begin{array} { r } { p = \frac { d \mu } { d \nu } \geq 0 } \end{array}$ with $\textstyle \mu ( B ) = \int _ { B } p d \nu$

Definition C.3. For a measurable map T and a measure $\mu ,$ the pushforward is $T _ { \# } \mu ( B ) : =$ $\mu ( T ^ { - 1 } ( B ) )$ . For a random vector X on $( \Omega , \mathcal { F } , \mathbb { P } )$ , its law is $\mathcal { L } ( X ) : = X _ { \# } \mathbb { P } ; \cdots X$ has a density” means $\mathcal { L } ( X ) \ll \lambda$

Proposition C.1. Let M and $M ^ { \prime }$ be afine subspaces $o f \mathbb { R } ^ { n }$ and let $T : M \to M ^ { \prime }$ be a surjective afine map. $I f \mu \ll \lambda _ { M }$ , then $T _ { \# } \mu \ll \lambda _ { M ^ { \prime } }$ . In particular, a bijective afine map $T \colon M \to M$ sends λ<sub>M</sub>-absolutely continuous measures to λ<sub>M</sub>-absolutely continuous measures.

Proof. Let p = dim M and $q = \dim M ^ { \prime }$ . Via Definition C.1, choose linear isometries to identify $\lambda _ { M }$ with $\lambda _ { p }$ on $\mathbb { R } ^ { p }$ and $\lambda _ { M ^ { \prime } }$ with $\lambda _ { q }$ on $\mathbb { R } ^ { q } ;$ in these coordinates $T$ becomes an afine map $\hat { T } : \mathbb { R } ^ { p } \to \mathrm { \dot { \mathbb { R } } ^ { q } }$ with linear part $A : \mathbb { R } ^ { p } \to \mathbb { R } ^ { q }$ of rank $q$ (surjectivity of $T$ forces this). Let $E : = ( \operatorname { K e r } A ) ^ { \perp } \cong \mathbb { R } ^ { q }$ , so the restriction $A | _ { E } : E \to$ R<sup>q</sup> is invertible. For any $B \subseteq \mathbb { R } ^ { q }$ with $\lambda _ { q } ( B ) = 0$ and translation vector b (so ${ \hat { T } } ( s ) = A s + b )$ ,

$$
\hat { T } ^ { - 1 } ( B ) = \lbrace ( u , v ) \in \mathrm { K e r } A \times E : A | _ { E } ( v ) \in B - b \rbrace = \mathrm { K e r } A \times ( A | _ { E } ) ^ { - 1 } ( B - b ) .
$$

Since $A | _ { E }$ is invertible, $\lambda _ { q } \bigl ( ( A | _ { E } ) ^ { - 1 } ( B - b ) \bigr ) = | \mathrm { d e t } ( A | _ { E } ) | ^ { - 1 } \lambda _ { q } ( B - b ) = 0$ . By Fubini (with $\lambda _ { p } = \lambda _ { \mathrm { K e r } A } \otimes \lambda _ { E } ) , \lambda _ { p } ( \hat { T } ^ { - 1 } ( B ) ) = 0$ , hence $T _ { \# } \mu ( B ) = \mu ( T ^ { - 1 } ( B ) ) = 0$ ✷

## Hausdorf measure and dimension.

Definition C.4. For $A \subseteq \mathbb { R } ^ { n } , s \geq 0$ , and $\delta > 0 .$ , set

$$
\begin{array} { r } { \mathcal { H } _ { \delta } ^ { s } ( A ) : = \operatorname* { i n f } \Bigr \{ \sum _ { i = 1 } ^ { \infty } \alpha ( s ) \big ( \frac { \mathrm { d i a m } U _ { i } } { 2 } \big ) ^ { s } : A \subseteq \bigcup _ { i } U _ { i } , \mathrm { d i a m } U _ { i } \leq \delta \Bigr \} , } \end{array}
$$

and define the s-dimensional Hausdorf measure $\mathcal { H } ^ { s } ( A ) : = \mathrm { l i m } _ { \delta \downarrow 0 } \mathcal { H } _ { \delta } ^ { s } ( A )$ . The constant $\alpha ( s ) = \pi ^ { s / 2 } / \Gamma \big ( \textstyle { \frac { s } { 2 } } + 1 \big )$ normalizes so that $\mathcal { H } ^ { m } = \lambda _ { m }$ on $\mathbb { R } ^ { m }$ for integer m; the normalizing constant is irrelevant for null-set statements. A set A is $\mathcal { H } ^ { s }$ -null if $\mathcal { H } ^ { s } ( A ) = 0$

Definition C.5. The Hausdorf dimension of A is dim $\varkappa ( A ) : = \operatorname* { i n f } \{ s \geq 0 : \mathcal { H } ^ { s } ( A ) = 0 \}$

Proposition C.2 (Lipschitz scaling [10, Theorem 2.8(i)]). If $f : A  \mathbb { R } ^ { n ^ { \prime } }$ is Lipschitz with constant $\operatorname { L i p } ( f )$ , then for every $s \geq 0$

$$
\mathcal { H } ^ { s } ( f ( A ) ) \leq \mathrm { L i p } ( f ) ^ { s } \mathcal { H } ^ { s } ( A ) .
$$

Consequently, Lipschitz maps do not increase Hausdorf dimension and map $\mathcal { H } ^ { s }  – n u l l$ sets to H<sup>s</sup>-null sets.

Surface measure and polar coordinates.

Definition C.6. For a linear subspace L with $m = \dim L ,$ the surface measure on $\mathbb { S } _ { L } =$ $\{ \omega \in L : \| \omega \| = 1 \}$ is $\sigma : = \mathcal { H } ^ { m - 1 } \ u \mathopen { } \mathclose \bgroup \lceil \mathbb { S } _ { L }$ . Under any linear isometry $L \cong \mathbb { R } ^ { m } , \sigma$ is the standard spherical measure on $S ^ { m - 1 }$

Proposition C.3 (Polar-coordinate factorization; adapted from [10, Theorem 3.12]). With $m = \dim L$ and σ as above, for every nonnegative Borel function $^ { g , }$

$$
\int _ { L } g d \lambda _ { L } = \int _ { 0 } ^ { \infty } \int _ { \mathbb { S } _ { L } } g ( r \omega ) r ^ { m - 1 } d \sigma ( \omega ) d r .
$$

Theorem 3.12 in [10] states the formula for $\mathbb { R } ^ { n }$ with the inner integral written as $\begin{array} { r l } { \int _ { \partial B ( r ) } g d \mathcal { H } ^ { m - 1 } } \end{array}$ ; the form above follows by identifying $L \cong \mathbb { R } ^ { m }$ via a linear isometry (which preserves $\mathcal { H } ^ { s }$ by [10, Theorem $2 . 2 ( \mathrm { v } ) ] )$ and by the substitution $\begin{array} { r } {  { \boldsymbol { { x } } } \ = \ r \omega } \end{array}$ , which yields $d \mathcal { H } ^ { m - 1 } | _ { \partial B ( r ) } = r ^ { m - 1 } d \sigma ( \omega )$ via the scaling property [10, Theorem $2 . 2 ( \mathrm { i v } ) ]$ ].

Lemma C.1. Let $R : L \setminus \{ 0 \}  \mathbb { S } _ { L } , R ( z ) = z / \| z \|$ . If a measure ν on L satisfies $\nu \ll \lambda _ { L }$ then $R _ { \# } \nu \ll \sigma$

Proof. Let $B \subseteq \mathbb { S } _ { L }$ with $\sigma ( B ) = 0$ . Then $R ^ { - 1 } ( B ) = \{ r \omega : r > 0 , \omega \in B \}$ . Note that $\{ 0 \} \not \in R ^ { - 1 } ( B )$ For $N \in \mathbb { N }$ set $A _ { N } : = \{ z \in L : 1 / N \leq \| z \| \leq N \}$ ; then $R ^ { - 1 } ( B ) =$ $\cup _ { N \geq 1 } ( R ^ { - 1 } ( B ) \cap A _ { N } )$ . For each $N$ , by Proposition C.3 applied with $g = \mathbf { 1 } _ { \{ r \omega : \omega \in B , 1 / N \le r \le N \} }$

$$
\lambda _ { L } \bigl ( R ^ { - 1 } ( B ) \cap A _ { N } \bigr ) = \int _ { 1 / N } ^ { N } r ^ { m - 1 } d r \cdot \sigma ( B ) = 0 .
$$

Hence $\lambda _ { L } ( R ^ { - 1 } ( B ) ) = 0$ by countable subadditivity, so $\nu ( R ^ { - 1 } ( B ) ) = 0 , { \mathrm { i . e . , ~ } } R _ { \# } \nu ( B ) = 0$ . ✷

Natural filtration of Algorithm 3. The randomness drawn at iteration k inside Algorithm 2 depends on whether $x _ { k }$ is a spanning point. Let $\bar { \delta } _ { k } \in \mathbb { R } ^ { 2 \dim \mathcal { C } }$ denote $\delta _ { k }$ padded with trailing zeros on a branch where $\delta _ { k }$ is drawn, and set $\delta _ { k } : = 0$ on the nonsmooth branch where it is not drawn. Define

$$
\xi _ { k } : = ( \bar { \delta } _ { k } , \tilde { v } _ { \mathrm { e x } } ^ { k } ) \in \mathbb R ^ { 2 \dim \mathcal { C } } \times \mathbb R ^ { n } .
$$

The natural filtration of Algorithm 3 is

$$
\mathcal { F } _ { k } : = \sigma ( x _ { 0 } , \xi _ { 0 } , \ldots , \xi _ { k - 1 } ) , \qquad \mathcal { F } _ { 0 } = \sigma ( x _ { 0 } ) .
$$

Then $x _ { k }$ is $\mathcal { F } _ { k }$ -measurable. Conditionally on $\mathcal { F } _ { k }$ :

$\tilde { v } _ { \mathrm { e x } } ^ { k } \sim U n i f \left( \mathbb { S } _ { L } \right)$ in both cases, independent of $\mathcal { F } _ { k }$

$\mathrm { O n ~ } \left\{ x _ { k } ~ \notin ~ \mathcal { C } _ { n s } \right\} : ~ \delta _ { k } \sim U n i f \left( \Delta _ { m _ { k } } \right)$ , independent of $\tilde { v } _ { \mathrm { e x } } ^ { k }$ , where $m _ { k } : = | V ^ { k } |$ satisfies $m _ { k } = 2$ dim C at interior points and $m _ { k } = 2 \dim { \mathcal { C } } - 1$ at smooth boundary points, and $\bar { \delta } _ { k }$ is this $\delta _ { k }$ padded with trailing zeros.

$$
\bullet \mathrm { O n } \{ x _ { k } \in \mathcal { C } _ { n s } \} \colon \bar { \delta } _ { k } = 0 .
$$

Throughout the proofs below, the deterministic choices in Algorithms 2 and 3 (the nearest point, inward normal, orthonormal completion, enumeration of $T$ and $U _ { ; }$ , line-search minimizer, and final tie-breaking) are made using fixed Borel selection and tie-breaking conventions; thus the resulting implementation is a measurable stochastic process.

Write $N : = 4 \dim { \mathcal { C } } - 1$ for the maximum number of candidates returned by Algorithm 2, including exploration. For each $x \in { \mathcal { C } }$ , enumerate the T-updates, then the U-updates, then the exploration direction, and pad any remaining slots as inactive, so that each candidate occupies a fixed index $i \in [ N ]$ . Let $I _ { i } ( x ) \in \{ 0 , 1 \}$ be the Borel activity indicator of slot i. For an active non-exploration slot let $q _ { i } ( x , z )$ be the vector inside normalize in Algorithm 2 at state x with noise coordinate $z ;$ for the exploration slot put $q _ { i } ( x , z )$ equal to the $\mathbb { S } _ { L } .$ -valued coordinate of z. Give inactive slots any fixed nonzero value in $L$ . Only events intersected with $\{ I _ { i } ( x _ { k } ) = 1 \}$ are used below, so the dummy values have no probabilistic role. The maps $I _ { i }$ and $q _ { i }$ are Borel by the selection conventions above.

## Conditional law and conditional absolute continuity.

Definition C.7. Let $\mathcal { X } \in B ( \mathbb { R } ^ { d } )$ for some $d \geq 1$ , let X be an X-valued random element, and let ${ \mathcal { G } } \subseteq { \mathcal { F } }$ be a sub-σ-algebra. A regular conditional distribution of X given $\mathcal { G }$ is a function $\kappa : \Omega \times B ( \mathcal { X } )  [ 0 , 1 ]$ such that

(a) $\kappa ( \omega , \cdot )$ is a probability measure on $B ( \mathcal { X } )$ for every $\omega \in \Omega ;$

(b) for every $B \in B ( { \mathcal { X } } ) , \kappa ( \cdot , B )$ is G-measurable and is a version of $\mathbb { P } ( X \in B \mid { \mathcal { G } } )$ , i.e. $\kappa ( \cdot , B ) = \mathbb { P } ( X \in B \mid { \mathcal { G } } ) \ \mathbb { P } \mathrm { - a . s . }$

Such a κ exists for every such X [33, Chapter 2, §7, Definition 7, Theorem 5 and Corollary]. We write ${ \mathcal { L } } ( X \mid { \mathcal { G } } ) = \kappa$ . For a measure ν on $B ( \mathcal { X } )$ we say $\mathcal { L } ( X \mid \mathcal { G } ) \ll \nu \mathrm { ~ a . s ~ }$ . if some regular conditional distribution κ of X given G satisfies $\kappa ( \omega , \cdot ) \ll \nu$ for $\mathbb { P } \mathrm { - a . e . ~ } \omega .$

We record the conditional law of the iteration noise in these terms. For $x \in { \mathcal { C } }$ let $\varpi ( x , \cdot )$ be the law of $( \bar { \delta } , v )$ , where $v \sim U n i f \left( \mathbb { S } _ { L } \right)$ and, independently, <sup>¯</sup>δ is the zero-padded version of $U n i f \left( \Delta _ { m ( x ) } \right)$ , with $m ( x ) = 2 \dim { \mathcal { C } }$ on ri (C) and $m ( x ) = 2$ dim $\mathcal { C } - 1$ on rb $( { \mathcal { C } } ) \backslash { \mathcal { C } } _ { n s } ;$ on $\mathcal { C } _ { n s }$ let $\bar { \delta } = 0$ . These three sets are Borel $( \mathcal { C } _ { n s }$ by Proposition C.5), $\mathrm { s o } \ \varpi$ is a probability kernel. The conditional sampling rules above state that $\varpi ( x _ { k } ( \omega ) , \cdot )$ is a regular conditional distribution of $\xi _ { k }$ given $\mathcal { F } _ { k }$ in the sense of Definition C.7, with $\chi = \ln ^ { \cdot } \operatorname { d i m } \mathcal { C } \times \mathbb { R } ^ { n }$

Isometric flattening and smoothness. We first record that the smoothness notion of Definition 2.11 is preserved by the isometric flattening of C onto a full-dimensional convex set in $\mathbb { R } ^ { \mathrm { d i m } \it C }$

Proposition C.4. Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a nonempty closed convex set with dim $\mathcal { C } \geq 1$ . Fix $x _ { 0 } \in { \mathcal { C } }$ let $\bar { R } \in \mathbb { R } ^ { \mathrm { d i m } \mathcal { C } \times n }$ be a matrix whose rows form an orthonormal basis of the linear subspace af $( { \mathcal { C } } - x _ { 0 } )$ , and define

$$
G : \mathrm { a f f } ( \mathcal C ) \to \mathbb { R } ^ { \mathrm { d i m } \mathcal C } , \qquad G ( x ) : = \bar { R } ( x - x _ { 0 } ) , \qquad Q : = G ( \mathcal C ) .
$$

For $y \in \partial Q$ write

$$
\mathcal { N } ( Q , y ) : = \left\{ w \in \mathbb { R } ^ { \mathrm { d i m } \mathcal { C } } \mid \| w \| = 1 , \ \langle w , z - y \rangle \leq 0 \ \forall z \in Q \right\}
$$

for the set of unit outer normals to Q at y. Then:

1. $G$ is a bijective $a f f i n e$ isometry of af (C) onto $\mathbb { R } ^ { \mathrm { d i m } \it C }$ , with inverse $G ^ { - 1 } ( y ) = x _ { 0 } + \bar { R } ^ { T } y$ The set $Q$ is closed, convex, and of full dimension dim C in $\mathbb { R } ^ { \mathrm { d i m } \it C }$ , and $G ( \mathrm { r b } \left( \mathcal { C } \right) ) = \partial Q$

2. For every $x \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ , C is smooth at x in the sense of Definition 2.11 if and only if $| { \mathcal { N } } ( Q , G ( x ) ) | = 1$

Proof. For part 1, since the rows of $\bar { R }$ are orthonormal, $\bar { R } \bar { R } ^ { T } = I _ { \mathrm { d i m } } { \it c }$ and $\bar { R } ^ { T } \bar { R } = { \cal P } _ { L }$ , the orthogonal projection onto $L : = \arg \left( { \mathcal { C } } - x _ { 0 } \right)$ . Hence for $d \in L$

$$
\left\| \bar { R } d \right\| ^ { 2 } = d ^ { T } \bar { R } ^ { T } \bar { R } d = d ^ { T } P _ { L } d = \| d \| ^ { 2 } ,
$$

so $\bar { R }$ restricted to L is a linear isometry into $\mathbb { R } ^ { \mathrm { d i m } \it C }$ ; having rank dim C it is onto, and therefore a bijection $L  \mathbb { R } ^ { \mathrm { d i m } \mathcal { C } }$ whose inverse is $\bar { R } ^ { T }$ . It follows that $G$ is a bijective afine isometry of af $( \mathcal { C } ) = x _ { 0 } + L$ onto $\mathbb { R } ^ { \mathrm { d i m } \it C }$ with the stated inverse. Consequently Q is convex of dimension dim C. Moreover $Q$ is closed, since C is closed in af (C) and G is a homeomorphism onto $\mathbb { R } ^ { \mathrm { d i m } \it C }$ . As dim $Q = \dim { \mathcal { C } }$ we have rb $( \mathrm { Q } ) = \partial Q$ . Since G is a homeomorphism of af (C) onto $\mathbb { R } ^ { \mathrm { d i m } \it C }$ , it maps ri (C) onto ri (Q) and rb (C) onto rb $( \mathrm { Q } ) = \partial Q$

For part 2, fix $x \in { \mathrm { ~ r b ~ } } ( { \mathcal { C } } )$ and set $y : = G ( x ) \in \partial Q$ . By Lemma 2.2, af $( { \mathcal { C } } - x ) =$ af $( { \mathcal { C } } - x _ { 0 } ) = L$ . We first relate the two feasible-direction cones. By Lemma 2.3, $D _ { \mathcal { C } } ( x ) \subseteq L$ so for $w \in \mathbb { R } ^ { n }$ ，

$$
w \in D _ { C } ( x ) \iff w \in L { \mathrm { ~ a n d ~ } } \exists \lambda > 0 : x + \lambda w \in { \mathcal { C } } \iff w \in L { \mathrm { ~ a n d ~ } } \exists \lambda > 0 : y + \lambda { \bar { R } } w \in Q ,
$$

where the last equivalence uses $G ( x + \lambda w ) = y + \lambda \bar { R } w$ . Hence $D _ { Q } ( y ) = \bar { R } \big ( D c ( x ) \big )$ , and since $\bar { R }$ restricts to a linear homeomorphism $L  \mathbb { R } ^ { \mathrm { d i m } \mathcal { C } }$ , the same holds after taking closures:

$$
\begin{array} { r } { \operatorname { c l } \left( \operatorname { D } _ { \mathcal { C } } ( \mathrm { x } ) \right) = \left\{ d \in L \ \vert \ \bar { R } d \in \operatorname { c l } \left( \operatorname { D } _ { \operatorname { Q } } ( \mathrm { y } ) \right) \right\} . } \end{array}\tag{C.1}
$$

Throughout we use that for a closed convex set, the closure of the feasible-directions cone is the tangent cone, which is the polar of the normal cone ([30, Theorem 6.9]; [17, Chapter 3, Corollary 5.2.5]):

$$
\begin{array} { r } { \operatorname { c l } { \left( \operatorname { D } _ { \mathrm { Q } } ( \operatorname { y } ) \right) } = \mathcal { T } _ { Q } ( y ) = N _ { Q } ( y ) ^ { \circ } , \qquad N _ { Q } ( y ) = \mathcal { T } _ { Q } ( y ) ^ { \circ } . } \end{array}\tag{C.2}
$$

(⇐) Suppose $\mathcal { N } ( Q , y ) = \{ w \}$ . Since $N _ { Q } ( y ) = \mathbb { R } _ { + } \cdot \mathcal { N } ( Q , y )$ , the normal cone is the ray R $+ w ,$ and by (C.2),

$$
\operatorname { c l } \left( \operatorname { D } _ { \mathbb { Q } } ( \mathbf { y } ) \right) = \left( \mathbb { R } _ { + } w \right) ^ { \diamond } = \left\{ g \in \mathbb { R } ^ { \operatorname { d i m } ^ { C } } \mid \langle g , w \rangle \leq 0 \right\} = \left\{ g \in \mathbb { R } ^ { \operatorname { d i m } ^ { C } } \mid \langle g , u \rangle \geq 0 \right\} , \qquad u : = - w \neq 0 ,
$$

where u is the inward unit normal. Substituting into (C.1) and using $\langle \bar { R } d , u \rangle = \langle d , \bar { R } ^ { T } u \rangle$

$$
\mathrm { c l } \left( \mathrm { D } _ { \mathcal { C } } ( \mathbf { x } ) \right) = \{ d \in \mathbb { R } ^ { n } \mid \langle d , v \rangle \geq 0 \} \cap \mathrm { a f f } \left( \mathcal { C } - x _ { 0 } \right) , \qquad v : = \bar { R } ^ { T } u .
$$

Here v lies in the column space of $\bar { R } ^ { T }$ , which equals L, and $v \neq 0$ because $\bar { R } \bar { R } ^ { T } = I _ { \mathrm { d i m } } { \it c }$ makes $\bar { R } ^ { T }$ injective. This is precisely the half-space condition of Definition 2.11, so $\mathcal { C }$ is smooth at x. (⇒) Suppose C is smooth at $x ,$ so cl $( \mathrm { D } _ { \mathcal { C } } ( \mathrm { x } ) ) = \{ d \mid \langle d , v \rangle \geq 0 \} \cap L$ for some $0 \ne v \in L$ For $d \in L$ we have $d = \bar { R } ^ { T } \bar { R } d$ , hence $\langle d , v \rangle = \langle \bar { R } d , \bar { R } v \rangle$ . Setting $e : = \bar { R } v$ , which is nonzero since R<sup>¯</sup> is injective on L, (C.1) gives

$$
\operatorname { c l } \left( \operatorname { D } _ { \operatorname { Q } } ( \mathbf { y } ) \right) = \bar { R } ( \operatorname { c l } \left( \operatorname { D } \boldsymbol { c } ( \mathbf { x } ) \right) ) = \left\{ g \in \mathbb { R } ^ { \mathrm { d i m } \mathcal { C } } \mid \langle g , e \rangle \geq 0 \right\} .
$$

Taking polars and using (C.2), $N _ { Q } ( y ) = \operatorname { c l } \left( \operatorname { D _ { Q } } ( \mathrm { y } ) \right) ^ { \circ } = \mathbb { R } _ { + } ( - e )$ , a ray. Its unit vectors form the singleton $\mathcal { N } ( Q , y ) = \{ - e / \| e \| \}$ ✷

Nonsmooth boundary points.

Proposition C.5. Let ${ \mathcal { C } } \subseteq \mathbb { R } ^ { n }$ be a nonempty closed convex set with dim $\mathcal { C } \geq 1$ , and let $\mathcal { C } _ { n s } \subseteq \mathrm { r b } \left( \mathcal { C } \right)$ be its set $o f$ nonsmooth boundary points in the sense of Definition 2.11. Then $\mathcal { C } _ { n s }$ is a Borel subset of $\mathbb { R } ^ { n }$ and

$$
\mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 } ( \mathcal { C } _ { n s } ) = 0 .
$$

Proof. Let $G , Q .$ , and $\mathcal N ( Q , \cdot )$ be as in Proposition C.4, and let

$$
\Sigma _ { Q } : = \{ y \in \partial Q \ | \ | { \cal N } ( Q , y ) | \geq 2 \}
$$

denote the set of singular boundary points of $Q .$ . Every $y \in \partial Q$ admits a supporting hyperplane, so $\mathcal { N } ( Q , y ) \neq \emptyset$ . Combined with the second part of Proposition C.4, a point $x \in { \mathrm { r b } } \left( { \mathcal { C } } \right)$ is nonsmooth if and only if $| { \mathcal { N } } ( Q , G ( x ) ) | \geq 2$ , and therefore $\begin{array} { r } { \mathcal { C } _ { n s } = G ^ { - 1 } ( \Sigma _ { Q } ) } \end{array}$

We first show that $\Sigma _ { Q }$ is Borel. For $j \in \mathbb N$ set

$$
\Sigma _ { j } : = \left\{ y \in \partial Q \vert \exists w _ { 1 } , w _ { 2 } \in \mathcal { N } ( Q , y ) \mathrm { ~ w i t h ~ } \Vert w _ { 1 } - w _ { 2 } \Vert \geq 1 / j \right\} ,
$$

so that $\Sigma _ { Q } = \cup _ { j \geq 1 } \Sigma _ { j }$ . We claim that each $\Sigma _ { j }$ is closed. Let $y _ { i } \in \Sigma _ { j }$ with $y _ { i }  y ;$ then $y \in \partial Q$ , since $\partial Q$ is closed. Choose $w _ { 1 } ^ { i } , w _ { 2 } ^ { i } \in \mathcal { N } ( \bar { Q } , y _ { i } )$ with $\lVert \boldsymbol { w } _ { 1 } ^ { i } - \boldsymbol { w } _ { 2 } ^ { i } \rVert \ge 1 / j$ . By compactness of the unit sphere, pass to a subsequence along which $w _ { 1 } ^ { i }  w _ { 1 }$ and $w _ { 2 } ^ { i }  w _ { 2 } ;$ both limits are unit vectors and $\lVert \boldsymbol { w } _ { 1 } - \boldsymbol { w } _ { 2 } \rVert \ge 1 / j$ . For each fixed $z \in Q$ and each $\ell \in \{ 1 , 2 \}$ we have $\langle w _ { \ell } ^ { i } , z - y _ { i } \rangle \leq 0$ , and letting $i \to \infty$ gives $\langle w _ { \ell } , z - y \rangle \le 0$ . Hence $w _ { 1 } , w _ { 2 } \in \mathcal { N } ( Q , y )$ and $y \in \Sigma _ { j }$ Thus $\Sigma _ { Q }$ is $F _ { \sigma } ,$ and therefore Borel.

We next show that $\mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 } ( \Sigma _ { Q } ) = 0$ . By the first part of Proposition C.4, Q is a closed convex subset of $\mathbb { R } ^ { \mathrm { d i m } \it C }$ of dimension dim C. If dim $\mathcal { C } = 1$ , then $Q$ is a closed interval, a ray, or a line, and each of its at most two boundary points has a unique unit outer normal, so $\Sigma _ { { \cal Q } } = \emptyset$ and the claim is immediate. Assume therefore that dim $\mathcal { C } \geq 2$

We identify $\Sigma _ { Q }$ with the first singular stratum of $Q$ in [3, Definition 1.7]. That definition applies to any closed convex set $C \subseteq \mathbb { R } ^ { h }$ of dimension $h \colon { \mathcal { N } } ( C , x )$ is the set of unit outer normals at $x \in \partial C$ , which agrees with $\mathcal { N } ( Q , y )$ in Proposition C.4, and

$$
\Sigma ^ { \ell } ( C ) : = \{ x \in \partial C \ | \ \dim { \cal { N } } ( C , x ) \geq \ell \}
$$

for integers $0 \leq \ell < h$ . The dimension of $\mathcal { N } ( C , x )$ is that of a subset of the unit sphere: [3, Definition 1.7] records that $\mathcal { N } ( C , x )$ is the intersection of the unit sphere with the normal cone $N _ { C } ( x )$ of C at x, and that dim ${ \mathcal { N } } ( C , x ) =$ dim $N _ { C } ( x ) - 1$

Thus $\Sigma _ { Q } = \Sigma ^ { 1 } ( Q )$ . If dim $\mathcal { N } ( Q , y ) \ge 1$ , then $\left| { \mathcal { N } } ( Q , y ) \right| \geq 2$ , so $y \in \Sigma _ { Q }$ . Conversely, if $\left| { \mathcal { N } } ( Q , y ) \right| \geq 2$ , take distinct $w _ { 1 } , w _ { 2 } \in \mathcal { N } ( Q , y )$ . Convexity of Q makes $N _ { Q } ( y )$ a convex cone, so pos $( \{ w _ { 1 } , w _ { 2 } \} ) \subseteq N _ { Q } ( y )$ . If $w _ { 2 } \ne - w _ { 1 }$ , then $0 \not \in \mathrm { c o n v } \{ w _ { 1 } , w _ { 2 } \}$ and this hull is one-dimensional; its radial projection onto the unit sphere remains one-dimensional, and the cone it generates is two-dimensional. The antipodal case would give $\langle w _ { 1 } , z - y \rangle = 0$ for all $z \in Q$ , placing Q in a hyperplane and contradicting dim $Q = \dim { \mathcal { C } }$ . Hence dim $N _ { Q } ( y ) \ge 2$ , so dim $\mathcal { N } ( Q , y ) \ge 1$ and $y \in \Sigma ^ { 1 } ( Q )$

By [3, Theorem 3], which is stated for closed convex sets of full dimension, applied to $Q \subseteq \mathbb { R } ^ { \dim \mathcal { C } }$ with Alberti’s ambient dimension h equal to the present dim C and singularity index 1 (admissible since $0 \leq 1 <$ dim C), the set $\Sigma ^ { 1 } ( Q )$ is an $( \bar { \mathcal { H } } ^ { \mathrm { d i m } \mathcal { C } - 2 }$ , dim $\mathcal { C } - 2 )$ -rectifiable subset of $\mathbb { R } ^ { \mathrm { d i m } \it C }$ of class $C ^ { 2 }$ . By [3, Definition 1.1], such a set is a countable union of sets of finite $\mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 2 }$ measure and therefore has Hausdorf dimension at most dim $\mathcal { C } - 2$ . Since dim $\mathcal { C } - 1 > \dim \mathcal { C } - 2$ , Definition C.5 gives H<sup>dim</sup> <sup>C−</sup> ${ \mathrm { } } ^ { 1 } ( \Sigma _ { Q } ) = 0$

It remains to transfer both conclusions to C. By the first part of Proposition $\mathrm { C . 4 } , G ^ { - 1 }$ is an isometry, and in particular Lipschitz with constant 1. By Proposition C.2,

$$
\begin{array} { r } { \mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 } ( \mathcal { C } _ { n s } ) = \mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 } \big ( G ^ { - 1 } ( \Sigma _ { Q } ) \big ) \leq 1 ^ { \mathrm { d i m } \mathcal { C } - 1 } \cdot \mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 } \big ( \Sigma _ { Q } \big ) = 0 . } \end{array}
$$

Finally, $\begin{array} { r } { \mathcal { C } _ { n s } = G ^ { - 1 } ( \Sigma _ { Q } ) } \end{array}$ is the preimage of the $F _ { \sigma }$ set $\Sigma _ { Q }$ under the continuous map $G$ : af $( \mathcal { C } )  \mathbb { R } ^ { \dim \mathcal { C } }$ , hence $F _ { \sigma }$ in af (C). Since af (C) is closed in $\mathbb { R } ^ { n } , \mathcal { C } _ { n s }$ is an $F _ { \sigma }$ (in particular Borel) subset of $\mathbb { R } ^ { n }$ , and therefore a countable union of compact sets, since every closed subset of $\mathbb { R } ^ { n }$ is one. ✷

Remark C.1. When dim $\mathcal { C } = 1$ the conclusion reads $\mathcal { H } ^ { 0 } ( \mathcal { C } _ { n s } ) = 0 , \mathrm { i . e . } \ \mathcal { C } _ { n s } = \emptyset$ ; this case is treated separately in the proofs of Proposition C.5 and Lemma C.3.

Lemma C.2 (Radial image of a null set). Let $L \subseteq \mathbb { R } ^ { n }$ be a linear subspace, let $x \in \mathbb { R } ^ { n }$ and write $\mathbb { S } _ { L } : = \{ \omega \in L : \| \omega \| = 1 \}$ and $\Pi _ { x } ( y ) : = ( y - x ) / \left. y - x \right.$ for $y \in ( x + L ) \setminus \{ x \}$ . If $A \subseteq ( x + L ) \setminus \{ x \}$ satisfies H<sup>dim</sup> ${ \overset { \cdot - 1 } { \left( A \right) } } = 0$ 0, then H<sup>dim</sup> $^ { L - 1 } ( \Pi _ { x } ( A ) ) = 0$ in $\mathbb { S } _ { L }$

Proof. Write $r ( z ) : = z / \| z \|$ for $z \in L \setminus \{ 0 \}$ , so that $\Pi _ { x } = r \circ \tau$ with $\tau ( y ) : = y - x$ an isometry. For any $z _ { 1 } , z _ { 2 } \in L \setminus \{ 0 \}$ },

$$
z _ { 1 } \left\| z _ { 2 } \right\| - z _ { 2 } \left\| z _ { 1 } \right\| = \left\| z _ { 1 } \right\| ( z _ { 1 } - z _ { 2 } ) + z _ { 1 } \big ( \| z _ { 2 } \| - \| z _ { 1 } \| \big ) ,
$$

so by the triangle and reverse triangle inequalities $\left\| z _ { 1 } \left\| z _ { 2 } \right\| - z _ { 2 } \left\| z _ { 1 } \right\| \right\| \leq 2 \left\| z _ { 1 } \right\| \left\| z _ { 1 } - z _ { 2 } \right\|$ Dividing by $\left\| z _ { 1 } \right\| \left\| z _ { 2 } \right\|$ gives $\| r ( z _ { 1 } ) - r ( z _ { 2 } ) \| \le 2 \| z _ { 1 } - z _ { 2 } \| / \| z _ { 2 } \|$ . By symmetry (swapping z<sub>1</sub> and $z _ { 2 } )$ we also have $\| r ( z _ { 1 } ) - r ( z _ { 2 } ) \| \le 2 \| z _ { 1 } - z _ { 2 } \| / \| z _ { 1 } \|$ . Taking the minimum of the two bounds,

$$
\lVert r ( z _ { 1 } ) - r ( z _ { 2 } ) \rVert \leq \frac { 2 \left. z _ { 1 } - z _ { 2 } \right. } { \operatorname* { m a x } ( \left. z _ { 1 } \right. , \left. z _ { 2 } \right. ) } .
$$

In particular $r ,$ and hence $\Pi _ { x }$ , is Lipschitz with constant 2m on $\{ \| z \| \geq 1 / m \}$ . For $m \in \mathbb { N }$ set $A _ { m } : = A \cap \left\{ y : \left\| y - x \right\| \geq 1 / m \right\}$ . A Lipschitz map does not increase Hausdorf dimension and scales H<sup>dim</sup> <sup>L−1</sup> by at most the (dim $L - 1 )$ -th power of its constant, so $\mathcal { H } ^ { \mathrm { d i m } L - 1 } ( \Pi _ { x } ( A _ { m } ) ) \leq$ $( 2 m )$ dim $L ^ { - 1 } \mathcal { H } ^ { \mathrm { d i m } \bar { L } - 1 } ( A _ { m } ) = 0$ . Since $\Pi _ { x } ( A ) = \bigcup _ { m \geq 1 } \Pi _ { x } ( A _ { m } )$ is a countable union of null sets, it is null. ✷

Lemma C.3 (Conditional density of the step direction). Fix an iteration number $k ,$ suppose $\alpha \in ( 0 , 1 )$ and $\eta \in ( 0 , 1 ]$ , and condition on $\mathcal { F } _ { k }$ on the event $x _ { k } \notin \mathcal { C } _ { n s }$ . Set $L : = \mathrm { a f f } \left( \mathcal { C } - x _ { k } \right)$ For every $i \in [ s _ { k } ]$ , let $q _ { i }$ be the vector inside normalize that produces $\tilde { v } _ { i } ^ { k }$ , with $q _ { i } : = \tilde { v } _ { \mathrm { e x } } ^ { k } \ f o r$ the exploration slot. Then $q _ { i } \neq 0$ for every realization of the iteration noise, $\tilde { v } _ { i } ^ { k } = q _ { i } / \left| | q _ { i } | \right|$ lies in $\mathbb { S } _ { L }$ , and the conditional law of $\omega _ { i } : = \tilde { v } _ { i } ^ { k }$ given $\mathcal { F } _ { k }$ is absolutely continuous with respect to the surface measure $\boldsymbol { \sigma } = \mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 } \ u { \mid } \mathbb { S } _ { L }$

Proof. The index i ranges over the active slots of the enumeration fixed above, so $[ s _ { k } ] = \{ i \in$ $[ N ] : I _ { i } ( x _ { k } ) = 1 \}$

We first verify that every active generator is nonzero pointwise. Write $d : = \dim { \mathcal { C } }$ and recall that $\bar { \eta } = \eta / ( 2 \sqrt { d } ) \leq 1 / 2$ . Since $\begin{array} { r } { \zeta _ { k } = \sum _ { j = 1 } ^ { m _ { k } } \delta _ { k , j } v _ { k , j } } \end{array}$ , where $\delta _ { k } \in \Delta _ { m _ { k } }$ and every $v _ { k , j } \in U \cup T$ has unit norm,

$$
\| \zeta _ { k } \| \leq \sum _ { j = 1 } ^ { m _ { k } } \delta _ { k , j } \| v _ { k , j } \| = 1 .
$$

For an active U-update, the reverse triangle inequality gives

$$
\Vert q _ { i } \Vert = \Vert \alpha \bar { \eta } \zeta _ { k } + ( 1 - \alpha \bar { \eta } ) v \Vert \geq 1 - 2 \alpha \bar { \eta } = 1 - \frac { \alpha \eta } { \sqrt { d } } \geq 1 - \alpha > 0 .
$$

For an active T-update, u and v are orthogonal unit vectors, and hence

$$
\begin{array} { c } { { \left\| q _ { i } \right\| \ge \left\| ( 1 - \alpha ) \bar { \eta } u + ( 1 - \bar { \eta } ) v \right\| - \alpha \bar { \eta } \left\| \zeta _ { k } \right\| } } \\ { { \ge 1 - ( 1 + \alpha ) \bar { \eta } \ge \displaystyle \frac { 1 - \alpha } { 2 } > 0 . } } \end{array}
$$

For the exploration slot, $\left\| q _ { i } \right\| = \left\| \tilde { v } _ { \mathrm { e x } } ^ { k } \right\| = 1$ . Thus $q _ { i } \neq 0$ for every admissible realization of the iteration noise, and $\tilde { v } _ { i } ^ { k } = q _ { i } / | \lvert q _ { i } | | \in \dot { \mathbb { S } } _ { L }$

For the kernel construction, let $R ( q ) : = q / \left\| q \right\|$ for $q \neq 0$ and assign an arbitrary fixed value in $\mathbb { S } _ { L }$ at $q = 0$ . This extension only specifies $R ( q _ { i } ( x , z ) )$ at of-support noise values and does not afect an active algorithmic direction. For each slot $i \in [ N ]$ define

$$
\kappa _ { i } ( \omega , B ) : = \varpi \big ( x _ { k } ( \omega ) , \{ z : R ( q _ { i } ( x _ { k } ( \omega ) , z ) ) \in B \} \big ) .
$$

The map $z \mapsto R ( q _ { i } ( x _ { k } , z ) )$ is Borel, so $\kappa _ { i }$ is a probability kernel as the image of $\varpi ( \boldsymbol { x } _ { k } , \cdot )$ under that map. It is a regular conditional distribution of $R ( q _ { i } ( x _ { k } , \xi _ { k } ) )$ given $\mathcal { F } _ { k } \colon$ for bounded Borel $f ,$

$$
\mathbb { E } [ f ( R ( q _ { i } ( x _ { k } , \xi _ { k } ) ) ) ~ | ~ \mathcal { F } _ { k } ] = \int f ( R ( q _ { i } ( x _ { k } , z ) ) ) \varpi ( x _ { k } , d z ) \qquad \mathrm { a . s . }
$$

The identity holds first for indicator functions of rectangles in the joint variables $( z , x _ { k } )$ by the definition of the conditional kernel and [33, Chapter 2, §7, property $\mathrm { K } ^ { * } ]$ , then for bounded Borel functions by the monotone-class theorem.

If i is an active exploration slot, then by Algorithm $2 \ \omega _ { i } = \tilde { v } _ { \mathrm { e x } } ^ { k }$ is uniform on $\mathbb { S } _ { L }$ , whose law is $\sigma / \sigma ( \mathbb { S } _ { L } ) \ll \sigma$ , and therefore $\kappa _ { i } ( \omega , \cdot ) \ll \sigma$ on this event.

If i is an active non-exploration slot, freeze $x _ { k } = x$ and write the pre-normalized generator as

$$
q _ { i } = b _ { i } + \alpha \bar { \eta } \zeta _ { k } ,
$$

where $b _ { i }$ is determined by $x _ { k }$ via the selection conventions above, hence $\mathcal { F } _ { k }$ -measurable, and $\alpha \bar { \eta } > 0$ . We first show $\zeta _ { k }$ has a conditional density on $L ,$ i.e. $\zeta _ { k } \ll \lambda _ { L }$ . Write $m _ { k } : = | V ^ { k } |$ for the number of base spanning directions. Recall $\begin{array} { r } { \zeta _ { k } = \sum _ { j = 1 } ^ { m _ { k } } \delta _ { k , j } v _ { k , j } } \end{array}$ with $V ^ { k } = \{ v _ { k , 1 } , \ldots , v _ { k , m _ { k } } \} \subseteq$ $L .$

Since $x _ { k } \notin \mathcal { C } _ { n s }$ , the point $x _ { k }$ is interior or a smooth boundary point. If $x _ { k }$ is a smooth boundary point with dim $\mathcal { C } = 1$ , then ${ V } ^ { k } = \{ { u } \}$ , so $q _ { i } = u$ is deterministic and nonzero, ${ \mathbb S } _ { L } = \{ u , - u \}$ and $\sigma = \mathcal { H } ^ { 0 } \lceil \mathbb { S } _ { L }$ is counting measure; the law of $\omega _ { i } = u$ is then trivially absolutely continuous with respect to $\sigma ,$ hence $\kappa _ { i } ( \omega , \cdot ) \ll \sigma$ on this event, and the claim holds. In every other case the diference vectors $\{ v _ { k , j } - v _ { k , 1 } \} _ { j }$ span L. Since $x _ { k } \notin \mathcal { C } _ { n s }$ , Algorithm 2 takes the first branch $( \rho ( x _ { k } ) > \rho _ { 0 } )$ , or the third branch at an interior point $( 0 < \rho ( x _ { k } ) \le \rho _ { 0 } )$ , or the third branch at a smooth boundary point $( \rho ( x _ { k } ) = 0 )$ . In the first case, $V ^ { k } = \{ \pm w _ { 1 } , \ldots , \pm w _ { \mathrm { d i m } } c \}$ . In the second, $V ^ { k } = U \cup T$ as a set equals $U = \{ \pm u , \pm w _ { 1 } , \dots , \pm w _ { \mathrm { d i m } } c _ { - 1 } \}$ . In the third, with dim ${ \mathcal { C } } \geq 2 , V ^ { k } = \{ u , \pm w _ { 1 } , \ldots , \pm w _ { \dim { \mathcal { C } } - 1 } \}$ . The first two contain an antipodal pair in every basis direction of $L ,$ and the third contains the antipodal pair $\pm w _ { 1 }$ together with u.

Hence the linear map $\begin{array} { r } { T : \delta \mapsto \sum _ { j } \delta _ { j } v _ { k , j } } \end{array}$ , restricted to the afine hull of $\Delta _ { m _ { k } }$ , has diferential of rank dim $\mathcal { C }$ onto $L$ . As $\delta _ { k }$ has a density on $\Delta _ { m _ { k } }$ (Algorithm 2), its pushforward $\zeta _ { k } = T ( \delta _ { k } )$ is absolutely continuous with respect to $\lambda _ { L }$ by Proposition C.1.

Consequently $q _ { i } = b _ { i } + \alpha \bar { \eta } \zeta _ { k }$ , with $b _ { i } \in L$ , is a bijective afine image of $\zeta _ { k }$ within $L ,$ hence $q _ { i } \ll \lambda _ { L }$ by the bijective case of Proposition C.1. By the pointwise nonvanishing established above, $\omega _ { i } = \tilde { v } _ { i } ^ { k } = R ( q _ { i } )$ surely. Since $q _ { i } \ll \lambda _ { L }$ , Lemma C.1 gives $R ( q _ { i } ) \ll \sigma$ . As the frozen state $x \in \mathcal { C } \setminus \mathcal { C } _ { n s }$ was arbitrary, $\kappa _ { i } ( \omega , \cdot ) \ll \sigma$ whenever $I _ { i } ( x _ { k } ( \omega ) ) = 1$ and $x _ { k } ( \omega ) \notin \mathcal { C } _ { n s }$ ✷

Proof of Theorem 7.1. Let $\{ \mathcal { F } _ { k } \} _ { k = 0 } ^ { K }$ be the natural filtration of the algorithm, generated by $x _ { 0 }$ together with the randomness of iterations $0 , \ldots , k - 1 ;$ ; thus $x _ { k }$ is $\mathcal { F } _ { k }$ -measurable. Write $L : = \mathrm { a f f } \left( \mathcal { C } - x _ { k } \right)$ (independent of $x _ { k } )$ . We argue by induction on $k ;$ the base case holds by assumption.

Assume $\mathbb { P } ( x _ { j } \in \mathcal { C } _ { n s } ) = 0$ for all $j \le k$ . We work conditionally on $\mathcal { F } _ { k } ,$ on the event $\{ x _ { k } \notin \mathcal { C } _ { n s } \}$ , which has full probability by the inductive assumption. On this event $x _ { k }$ is interior or a smooth boundary point, so every direction in $\tilde { V } ^ { k }$ is either produced by the $T _ { - }$ and U-updates of Algorithm 2 or is the exploration direction, and Lemma C.3 applies to each.

Fix $i \in [ N ]$ and let $A _ { i } : = \{ I _ { i } ( x _ { k } ) = 1 \}$ , which is $\mathcal { F } _ { k } \mathrm { { - m e a s u r a b l e } }$ . On $A _ { i }$ set $Y _ { i } : = x _ { k } + t _ { i } ^ { k } \tilde { v } _ { i } ^ { k }$ and $\omega _ { i } : = \tilde { v } _ { i } ^ { k }$ ; define $Y _ { i }$ arbitrarily of $A _ { i }$ . We claim

$$
A _ { i } \cap \left\{ Y _ { i } \in { \mathcal { C } } _ { n s } \right\} \ \subseteq \ A _ { i } \cap \left\{ \omega _ { i } \in \Pi _ { x _ { k } } ( { \mathcal { C } } _ { n s } \setminus \left\{ x _ { k } \right\} ) \right\} .\tag{C.3}
$$

Indeed, on $A _ { i } \cap \{ Y _ { i } \in \mathcal { C } _ { n s } \}$ we have $Y _ { i } \in \mathcal { C } _ { n s } \subseteq \mathrm { r b } \left( \mathcal { C } \right)$ ; since $x _ { k } \notin \mathcal { C } _ { n s }$ , necessarily $Y _ { i } \neq x _ { k }$ whence $t _ { i } ^ { k } \left\| \tilde { v } _ { i } ^ { k } \right\| = \left\| Y _ { i } - x _ { k } \right\| > 0$ and $\Pi _ { x _ { k } } ( Y _ { i } ) = ( Y _ { i } - x _ { k } ) / \left. \lvert Y _ { i } - x _ { k } \right. \rvert = \omega _ { i }$ with $Y _ { i } \in \mathcal { C } _ { n s } \setminus \{ x _ { k } \}$ This proves (C.3). The inclusion (C.3) holds for every value of the step $t _ { i } ^ { k }$

For $y \in { \mathcal { C } }$ let $D _ { y } : = \Pi _ { y } ( \mathcal { C } _ { n s } \setminus \{ y \} ) \subseteq \mathbb { S } _ { L }$ and

$$
D : = \{ ( \theta , y ) \in \mathbb { S } _ { L } \times \mathcal { C } : \ \theta \in D _ { y } \} = \{ ( \theta , y ) \in \mathbb { S } _ { L } \times \mathcal { C } : \ y + r \theta \in \mathcal { C } _ { n s } \ \mathrm { f o r ~ s o m e } \ r > 0 \} .
$$

By construction $D _ { x _ { k } } = \Pi _ { x _ { k } } ( \mathcal { C } _ { n s } \backslash \{ x _ { k } \} )$ , so the right-hand event of (C.3) is $A _ { i } \cap \{ ( \omega _ { i } , x _ { k } ) \in D \}$ $( i )$ Every section is σ-null. By Proposition $\mathrm { C . 5 } , \mathcal { H } ^ { \mathrm { d i m } \mathcal { C } - 1 } ( \mathcal { C } _ { n s } ) = 0$ , so Lemma C.2 (applied to the subspace L of dimension dim ${ \mathcal { C } } ,$ the point $y ,$ and $A = \mathcal { C } _ { n s } \setminus \{ y \} \subseteq ( y + L ) \setminus \{ y \} )$ gives $\sigma ( D _ { y } ) = 0$ for every $y \in { \mathcal { C } }$

(ii) D is Borel. By the last paragraph of the proof of Proposition ${ \mathrm { C . 5 } } , { \mathcal { C } } _ { n s } = \bigcup _ { m > 1 } K _ { m }$ with $K _ { m }$ compact. For $m , j \in \mathbb { N }$ the set $\{ ( \theta , y , r ) \in \mathbb { S } _ { L } \times ( \mathcal { C } \cap \mathcal { B } [ 0 , j ] ) \times [ 1 / j , j ] : \ y + r \theta \in K _ { m } \}$ is compact, hence so is its projection $D _ { m , j }$ onto the $( \theta , y ) .$ -coordinates, and $D = \cup _ { m , j } D _ { m , j }$ (given $( \theta , y ) \in D$ with $y + r \theta \in K _ { m }$ , take $j \geq \operatorname* { m a x } \{ \left\| y \right\| , r , 1 / r \} \}$ ). Thus D is a countable union of compact sets, hence Borel; in particular every $D _ { y }$ is Borel.

(iii) Conclusion. Let $\kappa _ { i }$ be the regular conditional distribution constructed in the proof of Lemma C.3. For every Borel $E \subseteq \mathbb { S } _ { L } \times \mathcal { C }$

$$
\begin{array} { r } { \mathbb { P } \big ( ( \omega _ { i } , x _ { k } ) \in E \mid \mathcal { F } _ { k } \big ) ( \omega ) = \kappa _ { i } \big ( \omega , E _ { x _ { k } ( \omega ) } \big ) \qquad \mathbb { P } \mathrm { - a . s . } } \end{array}
$$

Indeed, the identity holds for rectangles by property $\mathrm { K } ^ { * }$ and extends to Borel E by the monotone-class theorem. On $A _ { i } \cap \{ x _ { k } \notin \mathcal { C } _ { n s } \}$ , Lemma C.3 gives $\kappa _ { i } ( \omega , \cdot ) \ll \sigma$ , while (i) gives $\sigma ( D _ { x _ { k } ( \omega ) } ) = 0$ . Since $A _ { i } \in \mathcal { F } _ { k }$ , property $\mathrm { K } ^ { * }$ and the inductive assumption therefore give

$$
\mathbb { P } ( A _ { i } \cap \{ Y _ { i } \in \mathcal { C } _ { n s } \} \mid \mathcal { F } _ { k } ) \ \leq \ \mathbf { 1 } _ { A _ { i } } \kappa _ { i } ( \cdot , D _ { x _ { k } } ) \ = \ 0 \qquad \mathrm { a . s . }
$$

Since $x _ { k + 1 }$ is one of the active candidates, $\left\{ x _ { k + 1 } \in \mathcal { C } _ { n s } \right\} \subseteq \bigcup _ { i = 1 } ^ { N } \left( A _ { i } \cap \left\{ Y _ { i } \in \mathcal { C } _ { n s } \right\} \right)$ , and a union bound yields

$$
\mathbb { P } ( x _ { k + 1 } \in \mathcal { C } _ { n s } \mid \mathcal { F } _ { k } ) \le \sum _ { i = 1 } ^ { N } \mathbb { P } ( A _ { i } \cap \{ Y _ { i } \in \mathcal { C } _ { n s } \} \mid \mathcal { F } _ { k } ) = 0 \qquad \mathrm { a . s . }
$$

Taking expectations, and using that the conditioning event $\{ x _ { k } \notin \mathcal { C } _ { n s } \}$ has full probability by the inductive assumption, gives $\mathbb { P } ( x _ { k + 1 } \in \mathcal { C } _ { n s } ) = 0$ , which completes the induction. ✷