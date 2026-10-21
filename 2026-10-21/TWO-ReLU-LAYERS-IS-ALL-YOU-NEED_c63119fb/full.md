# TWO ReLU LAYERS IS ALL YOU NEED

FLORESTAN BRUNCK, CHRISTOPH HERTRICH, KILIAN RUESS, JACK STADE, AND AMIR YEHUDAYOFF

Abstract. We prove that every ReLU network can be exactly simulated by a ReLU network with two hidden layers. We present two proofs: one that is based on fillings in homological algebra and a direct combinatorial proof.

## 1. Introduction

The ubiquity and success of neural networks as a model of computation motivates studying the capabilities of their architectures. This study raises familiar complexity-theoretic questions, like what is the class of functions they can compute, at what depth, with how many gates, etc.?

This work investigates the role of depth in neural networks. On the one hand, deep learning is one of the major successes of modern computer science. On the other hand, classical approximation theorems dating back to the late 1980s show that, for many activation functions including ReLU, a single hidden layer sufices to approximate any continuous function arbitrarily well on a compact domain [10, 19, 24]. Although various explanations for the power of compositions have been proposed [5, 26], a quantitative understanding of the role of depth remains incomplete [20, 17].

The role of depth has been studied from several perspectives. One line of work investigates depth-width tradeofs in expressivity: can additional layers reduce the resources needed to represent or approximate a function [2, 12, 32]? Another asks whether depth can provide advantages for training [29]. Perhaps the most fundamental question, however, is whether deeper networks can represent functions that shallower networks cannot. Until now, this question has remained open.

We focus on neural networks that use the ReL $\mathrm { U } ( x ) = \operatorname* { m a x } \{ 0 , x \}$ activation function, which are among the most widely used ones in practice [13, 20]. A ReLU network with k hidden layers (and depth k + 1) is traditionally defined as the alternating composition

$$
T ^ { ( k ) } \circ { \mathrm { R e L U } } \circ T ^ { ( k - 1 ) } \circ \cdots \circ { \mathrm { R e L U } } \circ T ^ { ( 1 ) } \circ { \mathrm { R e L U } } \circ T ^ { ( 0 ) }
$$

where $T ^ { ( i ) }$ is an afine map between two Euclidean spaces and ReLU stands for entry-wise application.

ReLU networks compute continuous and piecewise linear (CPWL) functions. These are continuous functions that are afine on each cell of a finite polyhedral partition of the input Euclidean space. Conversely, every CPWL function can be computed by a ReLU network (see e.g. [2]). Their ability to capture nonlinear behavior while retaining a simple local afine structure makes CPWL functions useful beyond neural networks. For example, they are central to finite element methods [16], and their structure enables exact verification of stability conditions in robotic control [11].

For a ring $R \subseteq \mathbb { R }$ , denote by ${ \mathsf { C P W L } } _ { n } ( R )$ the space of CPWL functions whose afine pieces have coeficients in R. Denote by $\mathrm { R e L U } _ { n , k } ( R )$ ) the subset of ${ \mathsf { C P W L } } _ { n } ( R )$ representable by ReLU networks with k hidden layers, and weights and biases in R. The obvious inclusions are

$$
\mathrm { R e L U } _ { n , 0 } ( R ) \subseteq \mathrm { R e L U } _ { n , 1 } ( R ) \subseteq \mathrm { R e L U } _ { n , 2 } ( R ) \subseteq \cdots \subseteq \mathsf { C P W L } _ { n } ( R ) .\tag{1}
$$

Our main results work when R is any field containing $\mathbb { Q } .$ . We often write ${ \mathsf { C P W L } } _ { n }$ and $\mathrm { R e L U } _ { n , k }$ for ${ \mathsf { C P W L } } _ { n } ( \mathbb { Q } )$ and $\operatorname { R e L U } _ { n , k } ( \mathbb { Q } )$ respectively.

The main goal of this work is understanding the inclusions in the chain (1). Wang and Sun [34] identified a key structural result; see also [17, 4] for alternative, more geometric proofs. They showed that

$$
\operatorname { M A X } _ { n } ( x _ { 1 } , \dots , x _ { n } ) : = \operatorname* { m a x } \{ x _ { 1 } , \dots , x _ { n } \}
$$

is complete for $\mathsf { C P W L } _ { n - 1 } ( \mathbb { R } )$ with respect to depth. That is, every function $f \in { \mathsf { C P W L } } _ { n } ( \mathbb { R } )$ can be expressed as a finite sum

$$
f ( x ) = \sum _ { i } \sigma _ { i } \operatorname { M A X } _ { n + 1 } ( A _ { i } ( x ) )\tag{2}
$$

where $A _ { i } : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n + 1 }$ is an afine map and $\sigma _ { i } \in \{ \pm 1 \}$ . The geometric proof by [4] for the universality of $\mathrm { M A X } _ { n + 1 }$ relies on the fact that $\mathrm { M A X } _ { n + 1 }$ is the support function of a simplex, and every polytope can be triangulated into simplices.

The first inclusion in (1) is trivially strict because ReL ${ \mathrm { . U } } _ { n , 0 } ( \mathbb { R } )$ is precisely the set of afine functions. Mukherjee and Basu [27] showed that the second inclusion is strict for $n \geq 3$ because $\mathrm { M A X _ { 3 } \in R e L U _ { 3 , 2 } ( \mathbb { R } ) }$ but $\mathrm { M A X _ { 3 } \notin \ R e L U _ { 3 , 1 } ( \mathbb { R } ) }$ . In the other direction, Arora, Basu, Mianjy, and Mukherjee [2] observed that (2) implies that the length of the chain is at most logarithmic:

$$
\mathsf { C P W L } _ { n } = \mathrm { R e L U } _ { n , k } \qquad \mathrm { f o r } \ k \ge \lceil \log _ { 2 } ( n + 1 ) \rceil .
$$

The reason is that a divide-and-conquer network of depth $\lceil \log _ { 2 } ( n + 1 ) \rceil$ computes $\mathrm { M A X } _ { n + 1 }$ Subsequently, Hertrich, Basu, Di Summa, and Skutella [17] conjectured that the length of the chain is exactly $\lceil \log _ { 2 } ( n + 1 ) \rceil$ . In other words, they conjectured that $\mathrm { M A X } _ { n }$ can not be computed by less than $\lceil \log _ { 2 } ( n ) \rceil$ hidden layers.

A series of works validated this conjecture in a variety of special cases. Haase, Hertrich, and Loho [15] proved the conjecture over the integers:

$$
k < \lceil \log _ { 2 } ( n ) \rceil \qquad \Rightarrow \qquad \mathrm { M A X } _ { n } \not \in \mathrm { R e L U } _ { n , k } ( \mathbb { Z } ) .
$$

Averkov, Hojny, and Merkert [3] extended the logarithmic lower bounds to rings of the form $\mathbb { Z } [ 1 / N ]$ for an integer $N > 0$ . They showed that if a prime p does not divide N then

$$
k < \lceil \log _ { p } ( n ) \rceil \qquad \Rightarrow \qquad \mathrm { M A X } _ { n } \not \in \mathrm { R e L U } _ { n , k } ( \mathbb { Z } [ 1 / N ] ) .
$$

There are more lower bounds for special classes of networks. Valerdi [33] proved the conjecture for input convex neural networks (ICNN), which was defined by Amos, Xu and Kolter [1] and studied in subsequent works (e.g. [7, 8, 9, 6, 18]). Grillo, Hertrich, and Loho [14] proved an Ω(log log n) lower bound for ReLU networks that “respect the singularities of $\mathrm { M A X } _ { n } \}$

The conjecture was, however, recently disproved by Bakaev, Brunck, Hertrich, Stade, and Yehudayof [4]. They proved that

$$
\mathrm { M A X _ { 5 } \in R e L U _ { 5 , 2 } }
$$

![](images/3ef627ae31d1a58147198b7a2f065503c81f8785e9814c60c527955865351e75.jpg)  
Figure 1. The depth hierarchy of ReLU networks collapses to its first 3 levels.

and concluded the asymptotic improvement

$$
\mathrm { M A X } _ { n } \in \operatorname { R e L U } _ { n , k } \qquad { \mathrm { f o r ~ } } k = \lceil \log _ { 3 } ( n - 2 ) \rceil + 1 .
$$

Shortly after, computer-assisted constructions [28] showed that

$$
\mathrm { M A X } _ { 1 2 } \in \mathrm { R e L U } _ { 1 2 , 2 } .
$$

Again, this finite case can be converted into an improvement of the asymptotic bound from [4], yielding

$$
\mathrm { M A X } _ { n } \in \operatorname { R e L U } _ { n , k } \qquad \mathrm { f o r } \ k = \lceil \log _ { 6 } ( n / 2 ) \rceil + 1 .
$$

Our main result is that the entire depth hierarchy collapses to the second level, which is optimal by [27]. See Figure 1 for an illustration.

Theorem 1. For all $n \geq 1$ and any field $R \subseteq \mathbb { R }$ containing Q,

$$
\mathsf { C P W L } _ { n } ( R ) = \mathrm { R e L U } _ { n , 2 } ( R ) .
$$

To prove our main result, it sufices to prove the inclusion

$$
\mathrm { M A X } _ { n } \in \mathrm { R e L U } _ { n , 2 } ( \mathbb { Q } )
$$

We provide two proofs of the inclusion. The first is abstract and uses the language of homological algebra. The second is concrete and is based on exact combinatorial cancellations.

We, in fact, show that there is a ReLU network with two hidden layers for $\mathrm { M A X } _ { n }$ over the rationals where the precision required is poly(n) many bits. The lower bound from [3] says that such a network must divide by the product of all primes up to ≈ $\sqrt { n }$ , which requires precision of $\approx \sqrt { n }$ many bits. The combinatorial proof requires division by n!, which is consistent with this barrier. As written, the homological algebra proof also uses division by n!, but this is not strictly necessary. What the homological proof actually uses is division by $2 ^ { k } - 1$ for $k = 1 , \dots , n$ , which is consistent with the barrier by Fermat’s little theorem.

1.1. Building the Simplex. Because $\mathrm { M A X } _ { n }$ is positively homogeneous, we can assume that all biases in the network expressing it are zero, see [17]. Because $\mathrm { M A X _ { 2 } } ( x _ { 1 } , x _ { 2 } ) =$ $x _ { 1 } + \mathrm { R e L U } ( x _ { 2 } - x _ { 1 } )$ , we can and will work with networks that use $\mathrm { M A X _ { 2 } }$ gates instead of ReLU gates, also known as maxout networks.

![](images/14df2429fb287fc9b494861ed9d8a42b2d7bde5a8988fcedab1e0020799a819b.jpg)  
Figure 2. A paramid given as the pyramid over a two-dimensional parallelepiped B with apex $q .$

There is a duality between computing functions with $\mathrm { M A X _ { 2 } }$ networks and building formal Minkowski diferences of polytopes using the two operations: Minkowski sums and convex hulls of a union (see e.g. [17, 4] and references therein). This duality says that computing $\mathrm { M A X } _ { n + 1 }$ is equivalent to building the n-dimensional simplex $\Delta _ { n }$ . (There are $n + 1$ vertices and the dimension is n.)

Computations with one hidden layer exactly correspond to diferences of zonotopes, i.e. Minkowski sums of segments. Every zonotope can be decomposed into parallelepipeds ([31, Theorem (54)]) so one hidden layer is actually captured by parallelepipeds $\textstyle \sum _ { i } [ a _ { i } , b _ { i } ]$ . A general two-layer computation corresponds to the convex hull of two zonotopes.

Both of the constructions we describe use a particular class of two-layer components we call paramids. A paramid is the convex hull of an apex $\gamma \in \mathbb { R } ^ { n }$ and a parallelepiped $\textstyle \sum _ { i } [ a _ { i } , b _ { i } ]$ For an example see Figure 2. The support function of a paramid is thus of the form

$$
\mathrm { M A X _ { 2 } } \left( \langle \gamma , x \rangle , \sum _ { i } \mathrm { M A X _ { 2 } } ( \langle a _ { i } , x \rangle , \langle b _ { i } , x \rangle ) \right) .
$$

Paramids are the simplest two-layer polytopes and are thus a natural starting point to understand two-layer computations. Our results imply that $\mathrm { R e L U } _ { n , 2 }$ is spanned by support functions of paramids. Actually, there is a more direct way to see this (see Section 5.1). Informally, every convex hull of two zonotopes can be expressed algebraically in terms of paramids.

Another natural idea is to work modulo lower-dimensional pieces. A CPWL function $f \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } }$ is said to be lower-dimensional if there is a non-zero direction v such that $f ( x + t v ) = f ( x )$ for all $x \in \mathbb { R } ^ { n }$ and $t \in \mathbb { R }$ . So if, for each n, we can write $\mathrm { M A X } _ { n + 1 }$ as a

$$
\mathrm { M A X } _ { n + 1 } = \sum _ { i } c _ { i } h _ { P _ { i } } + R
$$

where each $P _ { i }$ is a paramid and the remainder R is a sum of lower-dimensional functions, then we are done by induction. The reason is that every lower dimensional function is reducible to $\mathrm { M A X } _ { n }$ , which we already know how to compute.

Subdivisions give us a powerful tool in these arguments. If we can subdivide $\Delta$ into pieces that can be computed in k hidden layers, then $\Delta$ can be computed by k hidden layers. So subdivisions are “free” in the sense that they do not cost us anything in terms of depth. This follows from the following well-known inclusion-exclusion-like formula also known as valuation (see e.g. Lemma 6.1.1 in [30] or 5.20 in [25]):

Lemma 2 (Valuation property). Suppose that a polytope X can be written as a union of polytopes $X = A _ { 1 } \cup \dots \cup A _ { k }$ . Then,

$$
X + \sum _ { \emptyset \neq S \subseteq \{ 1 , \dots , k \} } \bigcap _ { i \in S } A _ { i } = \sum _ { S \subseteq \{ 1 , \dots , k \} } \bigcap _ { i \in S } A _ { i } ,
$$

where an empty polytope is interpreted as zero.

Our new goal is thus to build the simplex using paramids modulo sums of lower dimensional pieces. We present two such constructions. One construction proceeds by induction and uses the language of chain complexes and homological algebra. The second construction is based on a direct inclusion-exclusion formula.

1.2. Overview of the Homological Algebra Implementation. Let us start with some intuition. Imagine we already found a two-layer representation of $\Delta _ { n }$ . By restricting the representation to a facet, we get a two-layer representation of $\Delta _ { n - 1 }$ . Each facet has its own representation and the $n + 1$ representations are somehow glued together coherently. What extra conditions does the boundary representation need to satisfy in order for it to extend to the n-simplex?

This type of “boundary filling” question naturally suggests using the framework of chain complexes and homology. A chain complex $C _ { \bullet }$ is a sequence of vector spaces $C _ { 0 } , C _ { 1 } , C _ { 2 } , . . .$ equipped with linear boundary maps d : $C _ { k } \to C _ { k - 1 }$ . The boundary map satisfies $d ^ { 2 } = 0$ which implies that the image of $d : C _ { k + 1 } \to C _ { k }$ is a subspace of the kernel of $d : C _ { k } \to C _ { k - 1 }$ This leads to homology:

$$
H _ { k } ( C _ { \bullet } ) = \ker ( d : C _ { k } \to C _ { k - 1 } ) _ { \big / \operatorname { i m g } \big ( d : C _ { k + 1 } \to C _ { k } \big ) } .
$$

An element x satisfying $d ( x ) = 0$ is called a cycle (we use lowercase x to distinguish a formal element x of a chain complex from a polytope, which we would represent with a capital X). An element x that is equal to $d ( y )$ for some $y$ is called a boundary. The condition that $d ^ { 2 } = 0$ says that every boundary is a cycle. The homology groups record which cycles are boundaries.

In Section 2, we define the chain complex $P _ { \bullet }$ of formal Minkowski diferences of polytopes modulo lower dimensional polytopes. This is the main chain complex we care about, because it corresponds to ReLU networks by the duality discussed above. A central property for performing induction is that a top-dimensional class in $P _ { \bullet }$ is determined by its boundary. Namely, for chains of polytopes in $\mathbb { R } ^ { n }$ , the map d : $P _ { n }  P _ { n - 1 }$ is injective. This is a consequence of known results (see Theorem 3). So if we can find some $x \in P _ { n }$ such that $d ( x ) = d ( \Delta _ { n } )$ then we actually obtained $\Delta _ { n }$

We let $\Lambda _ { \bullet }$ be the subcomplex of $P _ { \bullet }$ generated by paramids. This chain complex corresponds to the subclass of two-layer ReLU networks we focus on. The homology of $\Lambda _ { \bullet }$ tells us a lot about the functions that can be represented in terms of paramids. If we can represent the n-simplex by paramids, then the boundary of the $( n + 1 )$ )-simplex is a cycle in $\Lambda _ { n }$ By injectivity, the question of whether we can represent the (n + 1)-simplex by paramids exactly asks whether this cycle is a boundary in $\Lambda _ { \bullet }$ .

The Minkowski sum relations in $\Lambda _ { \bullet }$ are complicated, and it is dificult to perform calculations in this complex directly. So we try to approximate $\Lambda _ { \bullet }$ with a chain complex that is easier to understand. We define a chain complex $C _ { \bullet }$ of parallelepipeds. There is a coning map $\rho : C _ { k } \to P _ { k + 1 }$ that sends a parallelepiped X to the paramid with base X and apex at the origin. The image of $\rho$ lies in the subcomplex $\Lambda _ { \bullet }$ of $P _ { \bullet }$ . A parallelepiped is a sum of intervals so it is zero in $P _ { k }$ for $k \geq 2$ . The parallelepiped facet of a k-dimensional paramid is zero in $P _ { k - 1 }$ for $k \geq 3$ . Because the base does not survive in $\Lambda _ { \bullet }$ , this means that $\rho$ commutes with the boundary maps of $C _ { \bullet }$ and $P _ { \bullet }$ for $k \geq 2$

Let us try to implement the inductive construction. We start with a chain of parallelepipeds $x \in C _ { n - 1 }$ such that $\rho ( x )$ is the n-simplex. We then create copies of $x$ for each facet of the $( n + 1 )$ )-simplex to get a cycle $x ^ { \prime } \in C _ { n - 1 }$ such that $\rho ( { \boldsymbol { x } } ^ { \prime } )$ is the boundary of the $n + 1$ simplex. If we can find some $y \in C _ { n }$ such that $d ( y ) = x ^ { \prime }$ , then $\rho ( y )$ satisfies

$$
d ( \rho ( y ) ) = \rho ( d ( y ) ) = \rho ( x ^ { \prime } ) = d ( \Delta _ { n + 1 } ) .
$$

Injectivity tells us that $\rho ( y )$ is the needed (n + 1)-simplex.

This still does not explain how we find $y \in C _ { n }$ with $d ( y ) = x ^ { \prime }$ . We notice that not all cycles in $C _ { \bullet }$ are boundaries. Every boundary x in $C _ { \bullet }$ satisfies the augmentation condition, which says that x becomes 0 if we work modulo translations, because pairs of opposite faces of a parallelepiped give sign-reversed translates of each other after taking the boundary. But there are cycles in $C _ { \bullet }$ that do not have this property, see for example Figure 3.

![](images/31ff29f880ba8fe6c60d4953fb322c090c5da380be212555feaf72f09028384a.jpg)  
Figure 3. This triangle cannot be a boundary because it does not have the augmentation property.

We show that this is the only obstruction to filling boundaries in $C _ { \bullet } { : }$ : any cycle with the augmentation condition is a boundary. The idea is that non-trivial homology in $C _ { \bullet }$ comes from relations among edge directions (see Section 3.1, or Section 5.2 for a diferent perspective). For example, the triangle illustrated in Figure 3 is a cycle in $C _ { 1 }$ that is not a boundary due to the relation $t _ { 1 } + t _ { 2 } + t _ { 3 } = 0$ , where $t _ { 1 } , t _ { 2 }$ and $t _ { 3 }$ are the edge vectors of the triangle.

We first illustrate our solution with the example of a regular hexagon, as shown in Figure 4. Of the three edge directions $t _ { 1 } , t _ { 2 }$ and $t _ { 3 }$ , we choose the free generating set $t _ { 1 }$ and $t _ { 2 }$ . The two edges with direction $t _ { 3 }$ can be connected by parallelepipeds, as shown in the right part of the figure. After subtracting the boundary of these parallelepipeds, the remaining boundary (bottom part of the figure) has only $t _ { 1 }$ and $t _ { 2 }$ as edge directions. Since $t _ { 1 }$ and $t _ { 2 }$ are independent, this remaining boundary can be filled.

More generally, choose a set $E$ of edge directions that have no relations among them. Let $x \in C _ { k }$ be a cycle with the augmentation property. Write $X = x _ { 0 } + \cdot \cdot \cdot + x _ { k }$ , where $x _ { i }$ is a parallelepiped with i edge directions in $E$ and $k - i$ edge directions not in E. Letting $x _ { j }$ be the first non-zero term, we can find a boundary $y = y _ { j } + \dots + y _ { k }$ where $y _ { i }$ has i directions in $E$ such that $y _ { j } = x _ { j }$ . The $j = 0$ case requires the augmentation condition. By subtracting y from x, we reduce the number of non-zero terms. Inductively, this implies that x is a boundary.

![](images/fc4d624141dafa44045bfb3a4b22743fd13aca724825e2e33cf0fd01f2ca542f.jpg)  
Figure 4. The hexagon has the augmentation property, so it can be filled.

We are still not done. By induction, we constructed x such that $\rho ( x )$ is $\Delta _ { n }$ . To perform the induction step, we take $x ^ { \prime }$ to be copies of x for the facets of $\Delta _ { n + 1 }$ so that $\rho ( x ^ { \prime } ) = d ( \Delta _ { n + 1 } )$ We now want to fill $x ^ { \prime }$ , but in general it is not true that $x ^ { \prime }$ satisfies the augmentation condition.

The last ingredient is the doubling map $S : C _ { \bullet }  C _ { \bullet }$ . It takes a k-dimensional parallelepiped, scales it by a factor of 2, and then subdivides the scaled parallelepiped into $2 ^ { k }$ copies of the original one. This is illustrated in Figure 5. The element $x ^ { \prime \prime } = 2 ^ { \bar { k } } \bar { x ^ { \prime } } - S x ^ { \prime }$ is a cycle and it satisfies the augmentation property, so it can be filled. Geometrically, S is just a dilation by a factor of two:

$$
\rho ( S ( x ^ { \prime } ) ) = 2 \rho ( x ^ { \prime } ) .
$$

So, for $n - 1 \geq 2 .$

$$
\rho \left( \frac { x ^ { \prime \prime } } { 2 ^ { n - 1 } - 2 } \right) = \rho ( x ^ { \prime } ) = d ( \Delta _ { n + 1 } ) .
$$

Because $x ^ { \prime \prime }$ can be filled, there is $y$ such that

$$
d ( y ) = { \frac { x ^ { \prime \prime } } { 2 ^ { n - 1 } - 2 } } .
$$

Finally, by injectivity, the induction is complete

$$
\rho ( y ) = \Delta _ { n + 1 } .
$$

1.3. Overview of the Direct Implementation. The second proof we present is based on directly constructing a family of paramids, whose indicator functions exhibit a combinatorial cancellation when being summed up. Here the goal is to build the regular $( n - 1 )$ )-simplex $\Delta = \mathrm { c o n v } \{ e _ { i } : i \in [ n ] \}$ that sits inside the space $\{ x \in \mathbb { R } ^ { n } : \textstyle \sum _ { i } x _ { i } = 1 \}$ . There are two main ingredients to the construction. The first step is a comparably simple subdivision step, and the second step is where the non-trivial cancellation happens. By the valuation property (Theorem 20), it sufices to establish suitable identities up to lower-dimensional terms.

![](images/d98ebd09f2ee2b9ca89693f2e4aee1bf839ca2ce12ad480293934d80dec38429.jpg)  
Figure 5. The doubling map S scales a parallelogram in $C _ { k }$ by a factor of 2, then subdivides the parallelepiped face into $2 ^ { k }$ pieces, which are each a translated copy of the original parallelepiped face. In $P _ { \bullet }$ , this corresponds to scaling a paramid by 2 and then subdividing into $2 ^ { k }$ pieces. But subdivision does not change a class in $P _ { \bullet }$ , so $\rho \circ S$ is equal to $2 \rho$

![](images/969ced6d19c3e923e78d0442bbf564df119bad46afd5b070768ef1c3f57b06a4.jpg)  
Figure 6. On left, a partition of the simplex $\Delta$ to simplices with $R = \{ 3 \}$ On middle and right, the two pyramids $\mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } )$ for $v = 1$ and $v = 2$

The first step is a geometric identity obtained by subdividing the simplex. Fix a nonempty subset $R \subsetneq [ n ]$ . For $v \not \in R$ , consider the simplex

$$
Q _ { v , R } : = ( n - | R | ) \Delta _ { - v } - \sum _ { i \notin { R \cup \{ v \} } } e _ { i } ,
$$

where $\Delta _ { - v } = \mathrm { c o n v } \{ e _ { i } : i \neq v \}$ is the facet opposite to $e _ { v }$ . The simplex $Q _ { v , R }$ is a dilation and translation of the facet $\Delta _ { - v }$ . Cone it with $e _ { v }$ to get the simplex

$$
\operatorname { p y r } _ { e _ { v } } ( Q _ { v , R } ) : = \operatorname { c o n v } ( \{ e _ { v } \} \cup Q _ { v , R } ) .
$$

See Figure 6 for an illustration. The simplices $\mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } )$ for $v \not \in R$ partition the simplex $\Delta$ up to dilation and a translation, so the valuation property of support functions implies

$$
\mathrm { M A X } _ { n } = h _ { \Delta } \equiv \frac { 1 } { n - | R | } \sum _ { v \notin { R } } h _ { \mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } ) } ,
$$

where ≡ means up to support functions of lower-dimensional pieces.

This gives a family of identities expressing the support function of the original simplex in terms of other simplices. These identities alone do not yield the two-layer representations because the pyramids $\mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } )$ are simplices rather than paramids.

![](images/78e77fcc232914965f217ac329b130bb8b130077a89fc0a15a39492398184f3d.jpg)  
Figure 7. At the top, the parallelepipeds $B _ { v , \pi , M }$ and the simplices $Q _ { v , R }$ together with their respective coeficients. At the bottom, the total sum is the same for the parallelepipeds and for the simplices.

The second weightier step replaces the simplices $Q _ { v , R }$ by parallelepipeds, which we cone with the apex $e _ { v }$ to paramids. For each vertex $v ,$ we construct a family of parallelepipeds $B _ { v , \pi , M }$ indexed by permutations $\pi$ and sets of positions M. Their signed indicator sum can be evaluated by viewing the input-coordinates as steps of a walk from 0 to 1. Inclusion– exclusion cancels all walks except the thermostat walks which rise below level 1 and fall above level 1. Their signed count turns out to be exactly $( - 1 ) ^ { s - 1 } s ! ( n - 1 - s ) !$ , where s is the number of positive coordinates of the (generic) input. Binomial inversion expresses this count as a linear combination of the indicator sums of the auxiliary simplices $Q _ { v , R } \mathrm { : }$

$$
\sum _ { \pi } \sum _ { M } ( - 1 ) ^ { | M | } \mathbf { 1 } _ { B _ { v , \pi , M } } \equiv ( n - 1 ) ! \sum _ { R } \frac { ( - 1 ) ^ { | R | - 1 } | R | } { n - | R | } \mathbf { 1 } _ { Q _ { v , R } } .
$$

This identity replaces the simplices $Q _ { v , R }$ by parallelepipeds $B _ { v , \pi , M }$ . See Figure 7 for illustrations.

But this identity by itself is not useful, because parallelepipeds are not enough. We need to cone the identity. Coning by $e _ { v }$ brings the needed paramids to the left-hand side.

Summing over all vertices v and applying the geometric identities from the first step gives $h _ { \Delta }$ up to a lower-dimensional remainder term $R _ { n - 1 }$ :

$$
\mathrm { M A X } _ { n } = h _ { \Delta } = { \frac { ( - 1 ) ^ { n } } { n ! } } \sum _ { v } \sum _ { \pi } \sum _ { M } ( - 1 ) ^ { | M | } h _ { \mathrm { p y r } _ { e _ { v } } ( B _ { v , \pi , M } ) } + R _ { n - 1 } .
$$

This gives a two-layer representation of $\mathrm { M A X } _ { n } .$ because all paramids have support functions representable in two layers and the lower dimensional correction $R _ { n - 1 }$ is handled by induction.

## Statement of AI Use

The authors verified the correctness and originality of all content, including references.

Homological Algebra Approach. The proof contains significant technical contributions from both AI models and the authors. We used ChatGPT Sol 5.6 and ChatGPT Astra 6.

The major contribution of the human authors is the overall proof strategy. The constructions of the chain complexes $P _ { \bullet }$ and $\Lambda _ { \bullet }$ , the idea to represent elements of $\Lambda _ { \bullet }$ as images of a coning map $\rho : C _ { \bullet }  P _ { \bullet }$ , and the injectivity of the top boundary (Theorem 3) are all authors ideas. AI was used throughout the process to check proofs and look for simplifications.

The largest contributions from AI are the proofs of Theorem 10 and the doubling map used to prove Theorem 11. The proof of Theorem 4 that appears in this paper is also due largely to AI, but another proof was previously known to the authors (see Section 5.5).

Direct Combinatorial Approach. We used Claude Opus 5.5 extensively in developing the mathematical content of the combinatorial proof.

The outline of the proof was conceptualized by the authors. Our idea was to generalize the approach of subdividing into pyramids over zonotopes used in [4] to prove $\mathrm { M A X _ { 5 } \in R e L U _ { 5 , 2 } }$ handling lower-dimensional intersections by induction. We proposed combining this approach with ideas from the computational proof that $\mathrm { M A X _ { 1 2 } \in R e L U _ { 1 2 , 2 } }$ in [28]. Working interactively with the authors, the model implemented and carried out calculations in small dimensions, identified patterns in the resulting solutions and proposed the concrete constructions underlying Theorems 22 and 23.

LLMs, primarily ChatGPT Sol 6.1, were also used to produce some of the TikZ figures.

## Acknowledgments

The authors thank the members of the Neural Polytopes Zulip community. F.B. and A.Y. are supported by the DNRF Chair grant 20231101-28697. J.S. is supported by Starting Grant 1054-00032B from the Independent Research Fund Denmark under the Sapere Aude research career programme, and by the Carlsberg Foundation, grant CF24-1929.

## Part 1. A Homological Algebra Proof

## 2. The Chain Complex of Polytopes

We start by defining a chain complex $P _ { \bullet } ^ { n }$ . Here $P _ { k } ^ { n }$ should represent “k-dimensional polytopes in $\mathbb { Q } ^ { n }$ under Minkowski sum, modulo polytopes of dimension $< k "$

For a finite-dimensional real vector space S, let $V _ { S }$ be the vector space of formal Minkowskilinear combinations of polytopes. Formally, V<sub>S</sub> is the Q-vector space spanned by pairs $( X , Y )$ of polytopes, modulo the relation $( X , X ) = 0$ . Addition in $V _ { S }$ is given by Minkowski sum $( X _ { 1 } , Y _ { 1 } ) + ( X _ { 2 } , Y _ { 2 } ) = ( X _ { 1 } + X _ { 2 } , Y _ { 1 } + Y _ { 2 } )$ . Multiplication by a constant is given by $\lambda ( X , Y ) = ( \lambda X , \lambda Y )$ when λ is positive, and $\lambda ( X , Y ) = ( | \lambda | Y , | \lambda | X )$ when λ is negative.

If $S ^ { \prime }$ is a subspace of S, then there is a natural inclusion $V _ { S ^ { \prime } }  V _ { S }$ . We define $\widetilde { V } _ { S }$ to be the quotient of $V _ { S }$ by the sum over all proper subspaces $S ^ { \prime }$ of S of $V _ { S ^ { \prime } }$

Formally, for $k \geq 1$ , we define $P _ { k } ^ { n }$ to be the direct sum of $\widetilde { V } _ { S }$ over each oriented kdimensional subspace S of $\mathbb { Q } ^ { n }$ , with relations that reversing the orientation of a subspace corresponds to reversing sign in $V _ { S }$

$P _ { 0 } ^ { n }$ will be mostly irrelevant for our calculations, but we define it here for completeness. For $k = 0 , P _ { 0 } ^ { n }$ is defined as a direct sum over $\mathbb { Q } ^ { n }$ -indexed copies of $\mathbb { Q }$ , modulo global translations. Note a translation induces a permutation of the summands. Here we are adding relations $x - t ( x ) = 0$ , where $x \in \displaystyle \oplus _ { \mathbb { Q } ^ { n } } \mathbb { Q }$ and t is the linear map induced by the translation t permuting the factors.

We want to define the boundary d $: P _ { k + 1 } ^ { n } \to P _ { k } ^ { n }$ . We start with $k \geq 1$ . Now $P _ { k + 1 }$ is generated (as a Q-vector space) by classes of form $( X , 0 )$ with X a polytope in a $k + 1 \AA$ dimensional oriented subspace $S \subseteq \mathbb { Q } ^ { n }$ . The boundary of $( X , 0 )$ is defined as the sum over all unit vectors $v \in S$ of the face of X in direction v, translated so that it lies in the subspace U orthogonal to v. The orientation A of U is chosen so that $v \wedge A$ represents the orientation of S. Note that this term is zero in $P _ { k } ^ { n }$ unless v is orthogonal to a k-dimensional face of X, so only finitely many terms are non-zero.

When $k = 1$ $P _ { 1 } ^ { n }$ is generated by classes of form $( X , 0 )$ , with X a line segment in a 1-dimensional space S. Let a and b be the endpoints of X, chosen so that the vector $a - b$ induces the orientation on S. We set $d ( ( X , 0 ) )$ to be the class of $e _ { a } - e _ { b }$ , where $e _ { a }$ is the standard basis vector corresponding the a-component of $\bigoplus _ { \mathbb { Q } ^ { n } } \mathbb { Q }$ , and $e _ { b }$ is the one of the b component.

We need to check that d is well-defined and linear. Linearity is straightforward: if X and $Y$ are polytopes, then the face of $X + Y$ in direction v is the Minkowski sum of the faces of X and Y in direction v.

It is clear that $d ( ( X , X ) ) = 0$ . If X is ≤ k-dimensional and S is a $k + 1 \cdot$ -dimensional subspace containing X, then (X, 0) is 0 in $\widetilde { V } _ { S }$ , so we also need to check that $d ( ( X , 0 ) ) = 0$ If X is < k dimensional, then it has no k-dimensional faces, so $d ( ( X , 0 ) )$ is 0 in $P _ { k } ^ { n }$ . If X is exactly k dimensional, then it has two identical k-dimensional faces in the two directions orthogonal to X. These induce opposite orientations of the same class in $P _ { k } ^ { n }$ , so they cancel, and again $d ( X ) = 0$ . This shows that d is well-defined.

We need to check that $d ^ { 2 } = 0$ . Let X be an n-dimensional polytope. Every $n - 2 \cdot$ dimensional face F of X is a face of two $n - 1$ dimensional faces $Y _ { 1 }$ and $Y _ { 2 }$ of X. When

we take $d ^ { 2 } ( x )$ , a class corresponding to $F$ appears twice, but with opposite orientations.   
So $d ^ { 2 } ( ( X , 0 ) ) = 0$ , and $d ^ { 2 } = 0$ by linearity.

We define $\Lambda _ { \bullet } ^ { n }$ to be the subcomplex of $P _ { \bullet } ^ { n }$ where $\Lambda _ { k } ^ { n }$ is generated by classes of paramids of dimension k. For $k \leq 2$ , a paramid is a simplex, so $\Lambda _ { \bullet } ^ { n } = P _ { \bullet } ^ { n }$ . For $k \geq 3$ , the parallelepiped face of a paramid is 0 in $P _ { k - 1 } ^ { n }$ , and all the other faces of a paramid are paramids of one dimension lower. So $\Lambda _ { \bullet } ^ { n }$ is closed under the boundary.

Paramids can be represented in depth 2. If the $n -$ 1-simplex can be represented in depth 2, then anything of dimension $< n$ can be represented in depth 2, so everything in $\Lambda _ { n } ^ { n }$ can be represented in depth 2. Our goal is to show that the class of the n-simplex is in $\Lambda _ { n } ^ { n }$ for each n. By induction, this will show that the n-simplex is in depth 2 for all $n .$

The key property of $P _ { \bullet } ^ { n }$ is the following lemma, which says that a polytope is uniquely defined by its boundary, even modulo lower-dimensional terms:

Lemma 3. The top diferential d : $P _ { n } ^ { n } \to P _ { n - 1 } ^ { n }$ is injective.

In order to prove Theorem 3, we need to understand the space $\widetilde { V }$ better. Our goal is to prove the following characterization of $\widetilde { V } _ { S }$

Lemma 4. Let X and $Y$ be polytopes in $\mathbb { Q } ^ { n }$ , with $n \geq 2$ . Then $( X , Y ) = 0$ in $\widetilde { V } _ { \mathbb { O } ^ { n } }$ if and only if, for each non-zero direction $v ,$ the following property holds: let $F _ { 1 }$ and $F _ { 2 }$ be the faces of X in directions v and −v respectively, and let $G _ { 1 }$ and $G _ { 2 }$ be the faces of Y in directions v and $- v$ respectively. Then $( F _ { 1 } + G _ { 2 } , F _ { 2 } + G _ { 1 } ) = 0$ in $\widetilde { V } _ { S }$ , where $S$ is the subspace of $\mathbb { Q } ^ { n }$ orthogonal to $v ,$ and we translate $F _ { 1 } , F _ { 2 } , G _ { 1 }$ and $G _ { 2 }$ so that they lie in $S$

The “only $\mathrm { i f } ^ { \dag }$ direction appears in [22]. We reproduce the proof below:

Proof of the “only $i f ^ { \prime \prime }$ direction. The condition $( X , Y ) = 0$ in ${ \widetilde { V } } _ { \mathbb { Q } ^ { n } }$ is equivalent to saying that there exist polytopes $A _ { 1 } , \ldots , A _ { k }$ and $B _ { 1 } , \ldots , B _ { j }$ , each of dimension $< n$ , such that $X + A _ { 1 } + \cdot \cdot \cdot + A _ { k } = Y + B _ { 1 } + \cdot \cdot \cdot + B _ { j }$

First, suppose that such $A _ { i }$ and $B _ { i }$ exist. Fix a non-zero direction v. For each $i ,$ let $A _ { i } ^ { \prime }$ be the face of $A _ { i }$ in direction $v ,$ and let $A _ { i } ^ { \prime \prime }$ be the face of $A _ { i }$ in direction −v, and define $B _ { i } ^ { \prime }$ and $B _ { i } ^ { \prime \prime }$ similarly. Let $F _ { 1 }$ and $F _ { 2 }$ be the faces of X in directions v and −v respectively, and $G _ { 1 }$ and $G _ { 2 }$ similarly for $Y$ . Now $F _ { 1 } + A _ { 1 } ^ { \prime } + \cdot \cdot \cdot + A _ { k } ^ { \prime } = G _ { 1 } + B _ { 1 } ^ { \prime } + \cdot \cdot \cdot + B _ { j } ^ { \prime }$ and $F _ { 2 } + A _ { 1 } ^ { \prime \prime } + \cdot \cdot \cdot + A _ { k } ^ { \prime \prime } = G _ { 2 } + B _ { 1 } ^ { \prime \prime } + \cdot \cdot \cdot + B _ { j } ^ { \prime \prime }$ , so:

$$
F _ { 1 } + G _ { 2 } + A _ { 1 } ^ { \prime } + \cdot \cdot \cdot + A _ { k } ^ { \prime } + B _ { 1 } ^ { \prime \prime } + \cdot \cdot \cdot + B _ { j } ^ { \prime \prime } = F _ { 2 } + G _ { 1 } + A _ { 1 } ^ { \prime \prime } + \cdot \cdot \cdot + A _ { k } ^ { \prime \prime } + B _ { 1 } ^ { \prime } + \cdot \cdot \cdot + B _ { j } ^ { \prime }
$$

For each $A _ { i } .$ , we have two cases: either v is not orthogonal to $A _ { i }$ , in which case $A _ { i } ^ { \prime }$ and $A _ { i } ^ { \prime \prime }$ have dimension $< n - 1$ , or v is orthogonal to $A _ { i }$ , in which case $A _ { i } ^ { \prime } = A _ { i } ^ { \prime \prime }$ . In the latter case, we can cancel $A _ { i } ^ { \prime }$ and $A _ { i } ^ { \prime \prime }$ from both sides of the equation. A similar claim holds for $B _ { i } ^ { \prime }$ and $B _ { i } ^ { \prime \prime }$ . After canceling equivalent terms, we see that $( F _ { 1 } + G _ { 2 } , F _ { 2 } + G _ { 1 } ) = 0$ in $\widetilde { V } _ { S }$ where S is the subspace orthogonal to v. The assumption that $n \geq 2$ means that we can translate $F _ { 1 } + G _ { 2 }$ and $F _ { 2 } + G _ { 1 }$ so that they lie in $S .$ □

The other direction of the lemma could be deduced from the results in [23] (see Section 5.5). Instead, we will use a result of Jessen and Thorup [21]. They studied an algebra of polytopes modulo translations and a subdivision relation. Their results are stated for polytopes with vertices in an arbitrary subfield of R. We specialize to the case of vertices in $\mathbb { Q } .$

For a finite-dimensional rational vector space S, we define $B _ { S }$ to the abelian group freely generated by generators corresponding to full-dimensional polytopes in S. There is a homomorphism $\operatorname { V o l } _ { S } : B _ { S }  \mathbb { Q }$ that sends a polytope to its volume.

We define $\Pi _ { S }$ to be $B _ { S }$ modulo translations and subdivision relations that say that $X _ { 1 } + X _ { 2 } = Z$ if $X _ { 1 } , X _ { 2 }$ and $Z$ are polytopes where $X _ { 1 }$ and $X _ { 2 }$ are interior-disjoint and $X _ { 1 } \cup X _ { 2 } = Z$ . It is straightforward to see that Vol<sub>S</sub> descends to a map $\Pi _ { S }  \mathbb { Q }$

Let $Q$ be a codimension 1 subspace of $S$ and let v be a vector not in Q. Then we define a map $h _ { Q , v } : B _ { S }  B _ { Q }$ as follows: for a polytope X in S, let $F _ { 1 }$ and $F _ { 2 }$ be the faces of X in the directions orthogonal to $Q _ { i }$ , with the normal to $F _ { 1 }$ being in the +v direction and the normal of $F _ { 2 }$ being in the $- v$ direction, and translate $F _ { 1 }$ and $F _ { 2 }$ by multiples of $v$ so that they are in Q. We view $F _ { 1 }$ or $F _ { 2 }$ as being 0 if it is not full-dimensional.

Similar to the proof of the “only if” direction of Theorem 4, it is not too hard to see that, if $x \in B _ { S }$ is 0 in Π<sub>S</sub>, then $h _ { Q , v } ( x ) = 0$ in $\Pi _ { Q }$ . Jessen and Thorup prove the following:

Theorem 5. $( / 2 1 ]$ Theorem 1) An element $x \in B _ { S }$ is 0 in Π<sub>S</sub> if and only if $V o l _ { S } ( x ) = 0$ and, for every codimension 1 subspace $Q \subseteq S$ and v $\notin Q , h _ { Q , v } ( x )$ is 0 in Π<sub>Q</sub>.

We define a map $\xi : B _ { S }  V _ { S }$ that sends a polytope X to the class of $( X , 0 )$ . Clearly, $\xi$ is surjective. When $n \geq 1$ , Theorem 2 says that $\xi$ descends to a map $\Pi _ { S }  \widetilde { V } _ { S }$ . In order to use Theorem 5, we need to understand ξ better.

For $k \in \mathbb { N } _ { + }$ , the dilation $\lambda _ { k } : \Pi _ { S }  \Pi _ { S }$ is the map that sends a polytope X to kX. An element $x \in \Pi _ { S }$ is said to have degree i if $\lambda _ { k } ( x ) = k ^ { i } x$ for each k. We let $\Pi _ { S } ^ { i }$ be the subgroup of elements of degree i. Jessen and Thorup show the following:

Theorem 6. $( \it { { [ } 2 1 ] } ,$ Section 5) The group Π<sub>S</sub> is the direct sum of $\Pi _ { S } ^ { i } , \dots , \Pi _ { S } ^ { \mathrm { d i m } ( S ) }$

An example with $S = \mathbb { Q } ^ { 2 }$ is illustrated in Figure 8. In this case, $\Pi _ { \mathbb { Q } ^ { 2 } } ^ { 2 }$ is spanned by parallelograms, while $\Pi _ { \mathbb { Q } ^ { 2 } } ^ { 1 }$ is spanned by terms that are the diference of a triangle and a parallelogram with the same area.

Our next step is to show that ξ gives an isomorphism $\Pi _ { S } ^ { 1 }  \widetilde { V } _ { S }$ . Note that an element of $\Pi _ { S } ^ { i }$ for $i < \dim ( S )$ has volume 0. Indeed, $\mathrm { V o l } _ { S } ( \lambda _ { k } ( x ) ) = \tilde { k } ^ { \dim ( S ) } \mathrm { V o l } _ { S } ( x )$ . We also note that every $h _ { Q , v }$ map commutes with every $\lambda _ { k } .$ , so $h _ { Q , v }$ preserves the decomposition into $\Pi _ { S } ^ { i } .$

If S has dimension $\geq 2$ and $Q \subseteq S$ , then we can define a unique map $h _ { Q , v } : V _ { S } \to V _ { Q }$ similarly to $h _ { Q , v }$ on $B _ { S }$ . Note that $h _ { Q , v } \circ \xi = \xi \circ h _ { Q , v }$ . We can now determine the kernel of ξ:

Lemma 7. The kernel of $\xi : \Pi _ { S }  \widetilde { V } _ { S }$ is the sum of $\Pi _ { S } ^ { i } \ f o r \ i > 1$

Proof. First, we notice that every degree > 1 component of $\Pi _ { S }$ is in the kernel of ξ. This is because the dilation $\lambda _ { k }$ commutes with multiplication by k in $\tilde { V _ { S } }$ . That is, $\xi ( \lambda _ { k } ( x ) ) = k \xi ( x )$

So it remains to show that this is the entire kernel, i.e. that $\xi$ is injective on $\Pi _ { S } ^ { 1 }$ . This is by induction on dim(S). The base case is dim $( S ) = 1$ . In this case, $\Pi _ { S } = \Pi _ { S } ^ { 1 }$ and it is easy to see that $\xi : \Pi _ { S }  \widetilde { V } _ { S }$ is an isomorphism.

Now let dim $( S ) \ge 2$ . Let $x \in \Pi _ { S } ^ { 1 }$ with $\xi ( x ) = 0$ . Represent x as a sum of polytopes $A _ { 1 } + \cdot \cdot \cdot + A _ { k } - B _ { 1 } - \cdot \cdot \cdot - B _ { j } . \mathrm { ~ S o ~ } \xi ( x ) = ( A _ { 1 } + \cdot \cdot \cdot + A _ { k } , B _ { 1 } + \cdot \cdot \cdot + B _ { k } ) = 0 \mathrm { ~ i n ~ } \widetilde V _ { S } .$

$$
\boxed  \begin{array} { c c } { \begin{array} { r c l } \end{array} } \end{array} \qquad = \boxed { \begin{array} { c c } { \boxed { } } & { \begin{array} { r c l } \end{array} } } \end{array}
$$

![](images/c5666c944dd6b2b35992a3a04e616a65cbc4ce873a877183d85ac7773fb0a8c7.jpg)

![](images/f2bcbe6a39c272c47890039583e77b40c151f184f2f3a4fe2774032a71c10c35.jpg)  
Figure 8. Top: the dilation of a parallelogram by 2 can be subdivided into 4 copies of the original parallelogram. Middle: the dilation of a triangle by 2 can be subdivided into two copies of the original triangle and one parallelogram. So the diference of a triangle and a rectangle of the same area is in $\Pi _ { \mathbb { Q } ^ { 2 } } ^ { 1 }$ (bottom).

The “only $\mathrm { i f } ^ { \dag }$ part of Theorem 4 exactly says that $h _ { Q , v } ( ( A _ { 1 } + \cdot \cdot \cdot + A _ { k } , B _ { 1 } + \cdot \cdot \cdot + B _ { k } ) ) = 0$ for all Q, v. So $\xi ( h _ { Q , v } ( x ) ) = h _ { Q , v } ( \xi ( x ) ) = 0$ for all $Q , v .$

Now $h _ { Q , v } ( x )$ is in $\Pi _ { Q } ^ { 1 }$ for all $Q , v ,$ so by inductive assumption, $\xi ( h _ { Q , v } ( x ) ) = 0$ implies $h _ { Q , v } ( x ) = 0$ . We also know that $\mathrm { V o l } _ { S } ( x ) = 0$ . So by Theorem $5 , x = 0$ in Π<sub>S</sub>. By induction, this proves the claim. □

We know that $\xi$ is surjective, so Theorem $7$ shows that $\xi : \Pi _ { S } ^ { 1 } \to \tilde { V } _ { S }$ is an isomorphism. We can now complete the proof of Theorem 4:

Proof of the $^ { \ast } i f ^ { \ast }$ part of Theorem $\it 4 .$ The conditions of the lemma say exactly that $h _ { Q , v } ( ( X , Y ) ) =$ 0 for every codimension 1 subspace $Q \subseteq \mathbb { Q } ^ { n }$ and $v \not \in Q$ . Since $\xi$ is an isomorphism $\Pi _ { S } ^ { 1 }  \widetilde { V } _ { S }$ , there is some $x \in \Pi _ { S } ^ { 1 }$ such that $\xi ( x ) = ( X , Y )$ . For every $Q , v ,$ we have $\xi ( \tilde { h } _ { Q , v } ( x ) ) = h _ { Q , v } ( \xi ( x ) ) = h _ { Q , v } ( ( \boldsymbol { \tilde { X } } , Y ) ) = 0$ . Since $h _ { Q , v } ( x ) \in \Pi _ { S } ^ { 1 }$ , we have $h _ { Q , v } ( x ) = 0$ by Theorem 7. Since $n \geq 2$ , Vol<sub>Q</sub>n (x) = 0, so $x = 0$ in $\Pi _ { S }$ . So $( X , Y ) = \xi ( 0 ) = 0$ in $\widetilde { V } _ { S }$ □

## 3. The Chain Complex of Parallelepipeds

Let $T = \{ t _ { 1 } , \ldots , t _ { r } \}$ be the set of edge directions of the n simplex (with an arbitrary direction chosen for each). So $r = \binom { n + \bar { 1 } } { 2 }$ . Let $\Gamma = \operatorname { S p a n } _ { \mathbb { Z } } \{ t _ { 1 } , \dots , t _ { r } \}$ be the lattice in $\mathbb { Q } ^ { n }$ generated by the $t _ { i }$

The ring Q[Γ] consists of formal Q-linear combinations of elements of Γ, where multiplication is given by the group law on Γ. We write $g _ { 1 } , \ldots , g _ { r }$ for the generators of $\mathbb { Q } [ \Gamma ]$ corresponding to $t _ { 1 } , \ldots , t _ { r } . \ \mathrm { S o }$ , for example, $g _ { i } g _ { j }$ is the element of $\mathbb { Q } [ \Gamma ]$ corresponding to $t _ { i } + t _ { j }$ . The element $g _ { i } + g _ { j }$ is viewed formally as a superposition of $t _ { i }$ and $t _ { j }$

We define a chain complex $C _ { \bullet } ^ { n }$ , which consists of formal oriented parallelepipeds with edge directions in $T$ and vertices in Γ, with the diferential defined in the standard way.

Let $E = \mathbb { Q } ^ { r }$ with basis $e _ { 1 } , \ldots , e _ { r }$ . We can view $C _ { k } ^ { n }$ as the Q-module $\mathbb { Q } [ \Gamma ] \otimes _ { \mathbb { Q } } \bigwedge _ { \mathbb { Q } } ^ { k } E$ . This is generated by elements of the form $\gamma \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } } \ ( \gamma \in \Gamma )$ ), which we view as corresponding to the parallelepiped:

$$
\gamma + [ 0 , t _ { i _ { 1 } } ] + \cdot \cdot \cdot + [ 0 , t _ { i _ { k } } ]
$$

The boundary map on $C _ { k } ^ { n }$ is then defined as the standard cubical boundary. We have $d ( e _ { i } ) = g _ { i } - 1$ , this extends to higher orders using the graded Leibniz rule:

$$
d ( \gamma \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } } ) = \sum _ { j = 1 } ^ { k } ( - 1 ) ^ { j - 1 } \gamma ( g _ { i _ { j } } - 1 ) \otimes x _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge \widehat { x _ { i _ { j } } } \wedge \cdot \cdot \cdot \wedge x _ { k }
$$

The $\rho : C _ { k } ^ { n } \to P _ { k + 1 } ^ { n }$ is defined as follows: let $x = \gamma \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } }$ be a parallelepiped in $C _ { k } ^ { n }$ . Let X be the convex hull of 0 and the vertices of $x ,$ which is a polytope in $S = \operatorname { s p a n } ( \gamma , t _ { i _ { 1 } } , \dots , t _ { i _ { k } } )$ . If $\gamma , t _ { i } , \ldots , t _ { i _ { k } }$ are linearly independent, then the cone is nondegenerate and $\rho ( x )$ is the class corresponding to $X$ in $S ,$ with the orientation of S given by $t _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { k } } \wedge \gamma$ . Otherwise, the cone is degenerate and $\rho ( x ) = 0$

## Lemma 8. The map $\rho$ is well-defined.

Proof. The only thing to check is that $\rho$ respects the relation:

$$
\gamma \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { j } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { \ell } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } } = - \gamma \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { \ell } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { j } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } }
$$

in $C _ { k } ^ { n }$ . This is straightforward.

We now claim that $\rho$ commutes with the diferentials on $C _ { \bullet } ^ { n }$ and $P _ { \bullet } ^ { n }$ in degrees at least 2: Lemma 9. $H x \in C _ { k } ^ { n }$ for $k \geq 2 ,$ , then $d ( \rho ( x ) ) = \rho ( d ( x ) )$

Proof. It is suficient to check the claim for $x = \gamma \otimes x _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge x _ { i _ { k } }$ a parallelepiped. In this case, we have:

$$
d ( x ) = \sum _ { j = 1 } ^ { k } ( - 1 ) ^ { j - 1 } \gamma ( g _ { i _ { j } } - 1 ) \otimes x _ { i _ { 1 } } \wedge \dots \wedge \widehat { x _ { i _ { j } } } \wedge \dots \wedge x _ { k }
$$

There are two cases to check, depending on whether x is degenerate. First, we consider the case that x is not degenerate. In this case, $\rho ( x )$ is the class in $P _ { k + 1 } ^ { n }$ of the polytope $X$ , which is the convex hull of 0 and $\gamma + [ 0 , t _ { i _ { 1 } } ] + \cdot \cdot \cdot + [ 0 , t _ { i _ { k } } ]$ . The orientation is given by $t _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { k } } \wedge \gamma$

X has $2 k + 1$ facets. The face not containing 0 is $\gamma + [ 0 , t _ { i _ { 1 } } ] + \cdot \cdot \cdot + [ 0 , t _ { i _ { k } } ]$ . This is a Minkowski sum of intervals, which have dimension $1 < k$ . So the contribution of this face to $d ( \rho ( x ) )$ is 0.

For each $j \in \{ 1 , \ldots , k \}$ , we have two faces $F _ { j }$ and $G _ { j }$ of $X , ~ F _ { j }$ the convex hull of 0 and $\textstyle \gamma + \sum _ { \ell \neq j } [ 0 , t _ { i _ { \ell } } ]$ and $G _ { j }$ is the convex hull of 0 and $\begin{array} { r } { \gamma + t _ { i _ { j } } + \sum _ { \ell \neq j } [ 0 , t _ { i _ { \ell } } ] } \end{array}$

Now $t _ { i _ { j } }$ points out from $G _ { j }$ and $- t _ { i _ { j } }$ points out from $F _ { j }$ . So the orientation A of $F _ { j }$ is chosen so that $( - t _ { i _ { j } } ) \wedge A$ is the same as the orientation of $t _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { k } } \wedge \gamma$ of X. So $( - 1 ) ^ { j } t _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { j - 1 } } \wedge t _ { i _ { j + 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { k } } \wedge \gamma$ is a representation of the orientation of $F _ { j }$ . Similarly, the orientation of $G _ { j }$ is represented by $( - 1 ) ^ { j + 1 } t _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { j - 1 } } \wedge t _ { i _ { j + 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { k } } \wedge \gamma .$ which is equal to $( - 1 ) ^ { j + 1 } t _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { j - 1 } } \wedge t _ { i _ { j + 1 } } \wedge \cdot \cdot \cdot \wedge t _ { i _ { k } } \wedge ( \gamma + t _ { i _ { j } } )$ . This shows that contributions of $F _ { j }$ and $G _ { j }$ to $d ( \rho ( x ) )$ are exactly:

$$
\rho \left( ( - 1 ) ^ { j - 1 } \gamma ( g _ { i _ { j } } - 1 ) \otimes x _ { i _ { 1 } } \wedge \ldots \wedge x _ { i _ { j - 1 } } \wedge x _ { i _ { j + 1 } } \wedge \ldots \wedge x _ { k } \right)
$$

So $d ( \rho ( x ) ) = \rho ( d ( x ) )$ when the cone of x is non-degenerate. We now move on to the degenerate case. Similarly to before, let X be the convex hull of 0 and $\textstyle \gamma + \sum _ { j } [ 0 , t _ { i _ { j } } ]$ . Since the cone is degenerate, X is at most k dimensional. If $X { \mathrm { ~ i s ~ } } \leq k - 1$ dimensional, then every face in $d ( x )$ is degenerate, so $\rho ( d ( x ) ) = 0$ . The remaining case is when X is exactly k dimensional. Let S be the k-dimensional subspace containing X, and choose an arbitrary orientation of S.

In this case, define $F _ { j }$ and $G _ { j }$ as oriented polytopes in the same way as before. We also define a parallelepiped face $\begin{array} { r } { Z = \gamma + \sum _ { i } [ 0 , t _ { i _ { i } } ] } \end{array}$ . Since $Z$ is zero in $P _ { k } ^ { n }$ , it is clear that $\rho ( d ( x ) )$ is the sum of classes corresponding to the $\bar { F } _ { j } , G _ { j }$ and $Z .$

Now each of the $F _ { j } , G _ { j }$ or $Z$ is either degenerate, has positive orientation (with respect to the orientation chosen for S), or has negative orientation. We claim that the positiveorientation faces give a subdivision of $X$ , and similarly the negative orientation faces also give a subdivision of X. This is straightforward.

So by Theorem 2, the contributions from the positively-oriented pieces cancel with the contributions from the negatively-oriented pieces, so $\rho ( d ( x ) ) = 0$ , as required. □

The image of $\rho$ in $P _ { \bullet } ^ { n }$ lies in the subcomplex $\Lambda _ { \bullet } ^ { n }$ generated by paramids. Note that parallelepipeds in $C _ { \bullet } ^ { n }$ are restricted to having edge directions in $T ,$ so being in the image of $\rho$ is more restrictive than just being in $\Lambda _ { \bullet } ^ { n }$

3.1. The Homology of $C _ { \bullet } ^ { n }$ . We can actually give a nice characterization of the homology of $C _ { \bullet } ^ { n }$ . Let $\hat { \Gamma } = \mathbb { Z } ^ { r }$ , generated by $\hat { t } _ { 1 } , \dots , \hat { t } _ { r }$ , and let $\pi$ be the projection $\hat { \Gamma }  \Gamma$ given by $\hat { t } _ { i } \to t _ { i }$

Let $\hat { C } _ { \bullet } ^ { n }$ be the chain complex of parallelepipeds (in $\mathbb { R } ^ { r } )$ with vertices in $\hat { \Gamma }$ and edge direction in $\{ \hat { t } _ { 1 } , \ldots , \hat { t } _ { r } \}$ . Now $\hat { C } _ { \bullet } ^ { n }$ is a cubicalization of $\mathbb { R } ^ { r }$ , and $C _ { \bullet } ^ { n }$ is obtained from $\hat { C } _ { \bullet } ^ { n }$ by identifying parallelepipeds that difer by a translation in ker $( \pi )$

Now ker $( \pi )$ is a subgroup of $\mathbb { Z } ^ { r }$ , so by the structure theorem for finitely generated abelian groups, it is freely generated by elements $a _ { 1 } , \ldots , a _ { r - n }$ of $\mathbb { Z } ^ { r }$ . The $t _ { i }$ are edge vectors of the simplex, and it is not too hard to find such a set of generators explicitly.

So $C _ { \bullet } ^ { n }$ is a cubicalization of a topological space which is the quotient of $\mathbb { R } ^ { r }$ by the action of the group generated by $a _ { 1 } , \ldots , a _ { r - n }$ . This space is homeomorphic to a product of $\mathbb { R } ^ { n }$ and $r - n$ circles. So $H _ { k } ( C _ { \bullet } ^ { n } )$ is a direct sum of $\binom { r - n } { k }$ copies of $\mathbb { Q }$

We didn’t need to use anything special about $t _ { 1 } , \ldots , t _ { r }$ here. In general, let $T = \{ t _ { 1 } , \ldots , t _ { r } \}$ be a set of directions generating an abelian group Γ, and let $\mathit { C _ { \bullet } ^ { T } }$ be the chain complex of parallelepipeds with edges directions in $T$ and vertices in $\Gamma$ . Then the argument above shows that $C _ { \bullet } ^ { T }$ is a cubicalization of a topological space which is the quotient of $\mathbb { R } ^ { r }$ by the set of relations among the $t _ { i }$ . In particular, if the $t _ { i }$ freely generate $\Gamma _ { \ O }$ then $C _ { \bullet } ^ { T }$ has the homology of a point. This observation will be useful for filling cycles in $C _ { \bullet } ^ { n }$

This last observation is also a consequence of a more general idea about the Koszul resolution of a regular sequence, see Section 5.2.

3.2. Filling Boundaries in $C _ { \bullet } ^ { n }$ . Let $D _ { \bullet } ^ { n }$ be $C _ { \bullet } ^ { n }$ modulo translations. Formally, we can think of $D _ { k } ^ { n }$ as being equal to $\textstyle \bigwedge _ { \mathbb { Q } } ^ { k } E$ (recall $E = \mathbb { Q } ^ { r }$ , generated by $e _ { 1 } , \ldots , e _ { r } )$ . Let $\varepsilon$ be the map $C _ { \bullet } ^ { n } \to D _ { \bullet } ^ { n }$ that forgets translation in $C _ { \bullet } ^ { n }$ . Using $C _ { k } ^ { n } = \mathbb { Q } [ \Gamma ] \otimes { \textstyle \bigwedge } _ { \mathbb { Q } } ^ { k } E , \varepsilon$ can be defined using the map $\mathbb { Q } [ \Gamma ]  \mathbb { Q }$ that sends each $\gamma \in \Gamma$ to 1.

We note that every boundary x in $C _ { k } ^ { n }$ has $\varepsilon ( x ) = 0$ . Indeed, opposite pairs of faces of a parallelepiped cancel after taking ε. So this gives a necessary condition for a cycle x to be a boundary. We claim that it is suficient:

Lemma 10. A cycle $x \in C _ { k } ^ { n }$ is a boundary if and only $i f \varepsilon ( x ) = 0$

Proof. The “only $\mathrm { i f } ^ { \dag }$ direction is clear. So suppose that x is a cycle with $\varepsilon ( x ) = 0$ . We need to show that x is a boundary.

Without loss of generality, suppose that $t _ { 1 } , \ldots , t _ { n }$ freely generate the rank-n free abelian group Γ. For example, we can take $t _ { 1 } , \ldots , t _ { n }$ to be the set of edges of the simplex containing one distinguished vertex. Write $A = \{ t _ { 1 } , \ldots , t _ { n } \}$ and $B = \{ t _ { n + 1 } , \ldots , t _ { r } \}$ . Let $K _ { \bullet }$ be the chain complex of parallelepipeds with edge directions in A and vertices in $\Gamma$ , and similarly let $T _ { \bullet }$ be the chain complex of parallelepipeds with edge direction in $B$ and vertices in Γ. Write δ for the diferential on $K _ { \bullet }$ and ∂ for the diferential on $T _ { \bullet }$

We claim that $C _ { \bullet } ^ { n } \cong K _ { \bullet } \otimes T _ { \bullet }$ . Formally, this says:

$$
C _ { k } ^ { n } \cong \bigoplus _ { p + q = k } K _ { p } \otimes _ { \mathbb { Q } [ \Gamma ] } T _ { q }
$$

In other words, every parallelepiped in $C _ { k } ^ { n }$ can be written as a formal product of one in $K _ { \bullet }$ and one in $T _ { \bullet }$ . The claim is clear from $K _ { p } = \mathbb { Q } [ \Gamma ] \otimes \bigwedge _ { \mathbb { O } } ^ { p } \langle e _ { 1 } , \dotsc , e _ { n } \rangle$ and $T _ { q } =$ $\mathbb { Q } [ \Gamma ] \otimes \Lambda _ { \mathbb { O } } ^ { p } \langle e _ { n + 1 } , \ldots , e _ { r } \rangle$ . We can think of $K _ { p } \otimes T _ { q }$ as consisting of parallelepipeds with $p$ edge directions in A and q edge directions in B.

We extend δ and $\partial$ to maps $\delta : K _ { p + 1 } \otimes T _ { q }  K _ { p } \otimes T _ { q }$ and $\partial : K _ { p } \otimes T _ { q + 1 } \to K _ { p } \otimes T _ { q }$ . In order for these to be compatible with the diferential on $C _ { \bullet } ^ { n }$ , the map $\partial \big : K _ { p } \otimes \mathring { T _ { q + 1 } } \hat {  } K _ { p } \otimes T _ { q }$ has a $( - 1 ) ^ { p }$ factor compared to $\partial : T _ { q + 1 }  T _ { q }$ . Note that $\delta \partial + \partial \delta = 0$ , and the diferential $d$ on $C _ { \bullet } ^ { n }$ is identified with $\delta + \partial$

Now x is a cycle in $C _ { k } ^ { n }$ . For $i = 0 , \ldots , k$ , let $x _ { i }$ be the component of $x$ in $K _ { i } \otimes T _ { k - i }$ . The strategy will be to inductively add boundaries that zero $x _ { i } .$ , starting with x<sub>0</sub>. Note that $\varepsilon$ identifies sets of parallelepipeds with the same set of edge directions, so $\varepsilon ( x ) = 0$ implies that $\varepsilon ( x _ { i } ) = 0$ for each $x _ { i }$

So $x _ { 0 }$ is a class in $K _ { 0 } \otimes T _ { k }$ with $\varepsilon ( x _ { 0 } ) = 0$ . Note that $K _ { 0 } = \mathbb { Q } [ \Gamma ]$ . The image of $\delta : K _ { 1 }  K _ { 0 }$ is the ideal I in $\mathbb { Q } [ \Gamma ]$ generated by $g _ { 1 } - 1 , \dotsc , g _ { n } - 1$ . We claim that $x _ { 0 } \in I \otimes T _ { k }$ . Indeed, the parallelepipeds in $x _ { 0 }$ can be grouped in pairs that are orientation-reversed translates of each other, and each pair is an element of $I \otimes T _ { k }$ . This says that $x _ { 0 }$ is a δ-boundary. Let $y _ { 1 } \in K _ { 1 } \otimes T _ { k }$ with $\delta ( y _ { 1 } ) = x _ { 0 }$ . We can subtract $d ( y _ { 1 } ) = \delta ( y _ { 1 } ) + \partial ( y _ { 1 } )$ from x to get a new cycle with no component in $K _ { 0 } \otimes T _ { k }$ . Note that $d ( y _ { 1 } )$ is a cycle (since $d ^ { 2 } = 0 )$ , so $x - d ( y _ { 1 } )$ is a cycle.

Now proceed by induction: suppose that $x _ { i } = 0$ for $i = 0 , \ldots , j - 1$ for some $j \geq 1$ . We want to add a boundary to x that zeros $x _ { j }$ . Now $x _ { j } \in K _ { j } \otimes T _ { k - j } ,$ , and $\delta ( x _ { j } ) = d ( x _ { j } )$ since no other non-zero $x _ { i }$ has any boundary in $K _ { j - 1 } \otimes T _ { k - j }$ . Since $x _ { j }$ is a cycle, $\delta ( x _ { j } ) = 0$ By exactness of $K _ { \bullet }$ , there is some $y _ { j + 1 } \in K _ { j + 1 } \otimes T _ { k - j }$ with $\delta ( y _ { j + 1 } ) = x _ { j }$ . Subtracting $d ( y _ { j + 1 } ) = \delta ( y _ { j + 1 } ) + \partial ( y _ { j + 1 } )$ gives a new cycle with no $K _ { j } \otimes T _ { k - j }$ component.

By induction, we can eventually zero all the terms in $x .$ . Since we have only subtracted boundaries from x, this proves that x is a boundary, as required. □

Remark. In Section 5.4, we give a shorter, but less concrete proof of Theorem 10. We are also aware of a more explicit proof. We have decided not to include this since the combinatorial construction is explicit.

## 4. Filling Simplices

In order to show that the n-simplex is in $\Lambda _ { n } ^ { n } .$ , we want to find a cycle x in $C _ { n - 2 } ^ { n }$ with $\varepsilon ( x ) = 0$ such that $\rho ( x ) = d ( \Delta ^ { n } )$ is the boundary of the n-simplex. By Theorem 10, x is then the boundary of some $y \in C _ { n - 1 } ^ { n }$ , and by Theorem 9, $d ( \rho ( y ) ) = \rho ( d ( y ) ) = d ( \Delta ^ { n } )$ . By Theorem $3 , \rho ( y )$ is the class of $\Delta ^ { n }$ in $P _ { n } ^ { n }$ . Since the image of $\rho$ is in $\Lambda _ { \bullet } ^ { n }$ , this will show that $\Delta ^ { n }$ is in $\Lambda _ { \bullet } ^ { n }$

For a subset $S \subseteq \{ 1 , \dots , n + 1 \}$ of size $n _ { : }$ , let $F _ { S }$ be the $n - 1$ -dimensional subspace of $\mathbb { Q } ^ { n }$ spanned by edge directions contained in the face of $\Delta ^ { n }$ that is the convex hull of vertices in S. We say that an element $x$ of $C _ { k } ^ { n }$ satisfies the face direction condition if, for every parallelepiped $p$ with non-zero coeficient in $x ,$ all the vertices of $p$ are contained in $F _ { S }$ for some S.

First, we show how to realize condition $\varepsilon = 0 \cdot$

Lemma 11. Let x be a cycle in $C _ { k } ^ { n }$ with $k \geq 2$ . Then there is a cycle $x ^ { \prime } \in C _ { k } ^ { n }$ with $\varepsilon ( x ^ { \prime } ) = 0 \ a n d \ \rho ( x ^ { \prime } ) = \rho ( x )$ . Furthermore, if x satisfies the face direction condition, then $x ^ { \prime }$ does as well.

Proof. Let $S : C _ { k } ^ { n } \to C _ { k } ^ { n }$ be the map that takes a parallelepiped in $C _ { k } ^ { n }$ , multiplies by 2, then subdivides it into $2 ^ { k }$ translated copies of the original parallelepiped (see Figure 5). Formally,

$$
S ( \gamma \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } } ) = \sum _ { ( u _ { 1 } , \ldots , u _ { k } ) \in \{ 0 , 1 \} ^ { k } } ( \gamma ^ { 2 } \cdot g _ { i _ { 1 } } ^ { u _ { 1 } } \cdot \cdot \cdot \cdot \cdot g _ { i _ { k } } ^ { u _ { k } } ) \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } }
$$

Clearly, $\varepsilon ( S ( x ) ) = 2 ^ { k } \varepsilon ( x )$ for $x \in C _ { k } ^ { n }$ . We claim that $\rho ( S ( x ) ) = 2 \rho ( x )$ . To see this, let $x = \gamma \otimes e _ { i _ { 1 } } \wedge \cdot \cdot \cdot \wedge e _ { i _ { k } }$ be a parallelepiped in $C _ { k } ^ { n }$ . Recall that $\rho ( x )$ is the (class of the) paramid X with apex at the origin and base $\gamma + [ 0 , t _ { i _ { 1 } } ] + \cdot \cdot \cdot + [ 0 , t _ { i _ { k } } ]$ . So $2 X$ is the paramid with apex at the origin and base $2 \gamma + [ 0 , 2 t _ { i _ { 1 } } ] + \cdot \cdot \cdot + [ 0 , 2 t _ { i _ { k } } ]$ , which can be subdivided into $2 ^ { k }$ pieces, corresponding to the components of $S ( x )$ (see Figure 5).

So set $\begin{array} { r } { x ^ { \prime } = \frac { 1 } { \mathcal { I } ^ { k } - \mathcal { D } } ( 2 ^ { k } x - S ( x ) ) } \end{array}$ . Since $k \geq 2$ , the denominator is non-zero. Now $\varepsilon ( x ^ { \prime } ) = 0$ and $\rho ( x ^ { \prime } ) = \rho ( x )$ . It is straightforward to check that $x ^ { \prime }$ is a cycle. □

Now, we prove the inductive step that constructs a boundary of $\Delta ^ { n + 1 }$ given a $\Delta ^ { n }$

Lemma 12. Let $n \geq 2$ . Suppose there is an $x \in C _ { n - 1 } ^ { n }$ such that $\rho ( x ) = [ \Delta ^ { n } ]$ , where $d ( x )$ satisfies the face direction condition. Then there is a cycle $y \in C _ { n - 1 } ^ { n + 1 }$ such that $\rho ( y ) = d ( [ \Delta ^ { n + 1 } ] )$ , where y satisfies the face direction condition.

Proof. The cycle y will be constructed from one copy of x for each injective map $\mu :$ $\{ 1 , \dots , n + 1 \}  \{ 1 , \dots , n + 2 \}$ . We can view $\mu$ as a map that realizes $\Delta ^ { n }$ as a face of $\Delta ^ { n + 1 }$ . So each facet of the n-simplex is represented by $( n + 1 ) !$ symmetrized copies of $x .$

Let Γ be the lattice generated by simplex edge directions in $\mathbb { R } ^ { n }$ , and let $\Gamma ^ { \prime }$ be the lattice generated by simplex edge directions in $\mathbb { R } ^ { n + 1 }$ . A map $\mu : \{ 1 , \dots , n + 1 \} \to \{ 1 , \dots , n + 2 \}$ induces a homomorphism $\Gamma  \Gamma ^ { \prime }$ . This defines a map $\psi _ { \mu } : C _ { \bullet } ^ { n }  C _ { \bullet } ^ { n + 1 }$

Each injective map $\mu : \{ 1 , \dots , n + 1 \} \to \{ 1 , \dots , n + 2 \}$ extends uniquely to a permutation $\mu ^ { \prime } : \{ 1 , \dots , n + 2 \}  \{ 1 , \dots , n + 2 \}$ . We define:

$$
y = \frac { 1 } { ( n + 1 ) ! } \sum _ { \mu } \mathrm { s i g n } ( \mu ^ { \prime } ) \psi _ { \mu } ( x )
$$

where the sum is over injective $\mu : \{ 1 , \dots , n + 1 \} \to \{ 1 , \dots , n + 2 \}$

We claim that $\rho ( y ) \in P _ { n } ^ { n + 1 }$ is the boundary of $\Delta ^ { n + 1 }$ . For each $S \subseteq \{ 1 , \dots , n + 2 \}$ of size $n + 1$ , there are $( n + 1 ) !$ maps $\mu : \{ 1 , \dots , n + 1 \} \to \{ 1 , \dots , n + 2 \}$ with image S. For each of these $\mu , \rho ( \psi _ { \mu } ( x ) )$ is the $\Delta ^ { n }$ face of $\Delta ^ { n + 1 }$ in $F _ { S }$ , with sign determined by $\mathrm { s i g n } ( \mu ^ { \prime } )$ (note that the orientation on $F _ { S }$ induced by $\mu$ is determined by $\mathrm { s i g n } ( \mu ^ { \prime } ) )$ . So $\rho ( y )$ is exactly the boundary of $\Delta ^ { n + 1 }$

Next, we claim that y is a cycle in $C _ { n - 1 } ^ { n + 1 }$ . Clearly, $\psi _ { \mu }$ commutes with the boundary d on $C _ { n - 1 } ^ { n + 1 }$ . For each parallelepiped z with non-zero coeficient in $d ( x )$ , we choose some $S \subseteq \{ 1 , \dots , n + 1 \}$ such that $z$ is in $F _ { S }$ . This is possible by the face direction condition for $d ( x )$

Let z be a parallelepiped in $d ( x )$ , which by assumption is in $F _ { S }$ for some S. Let $\mu :$ $\{ 1 , \dots , n + 1 \} \to \{ 1 , \dots , n + 2 \}$ be injective, and let κ be the unique map that agrees with µ on S but is not equal to $\mu .$ . Note that $\mathrm { s i g n } ( \mu ^ { \prime } ) = - \mathrm { s i g n } ( \kappa ^ { \prime } )$ , since $\mu ^ { \prime }$ and $\kappa ^ { \prime }$ difer by a transposition. Now $\psi _ { \mu } ( z ) = \psi _ { \kappa } ( z )$ , so these terms cancel in $d ( y )$ . This shows that $d ( y ) = 0$

Note that every parallelepiped in the image of $\psi _ { \mu }$ is contained in the subspace $F _ { \mathrm { i m g } ( \mu ) }$ , so y satisfies the face direction condition. □

Theorem 13. For each n, $[ \Delta ^ { n } ] \in \Lambda _ { n } ^ { n }$

![](images/6509a14c15681fcd4c6438e0d7cc67a4ff356aa22dea07f8a49df47ba6754ef1.jpg)

![](images/57890ae2af7f0a6b8bbfd40b2d1b493185b654b7e14007081595f811fecc6640.jpg)  
Figure 9. Top: Using inclusion-exclusion, we can write the union P of the standard 3-simplex $\Delta _ { 3 } \quad -$ with vertices at (0, 0, 0), (1, 0, 0), (0, 1, 0) and (0, 0, 1) – and the paramid Π – with vertices (0, 0, 0), (0, 1, 0), (−1, 1, 0), (−1, 0, 1) and (0, 0, 1) – as the sum of the simplex and the paramid minus their intersection – the triangular face F with vertices (0, 0, 0), (0, 1, 0) and (0, 0, 1). Bottom: $P$ is also the Minkowski sum of the triangular face T with vertices at $( 0 , 0 , 0 ) , ( - 1 , 1 , 0 )$ and $( - 1 , 0 , 1 )$ and the segment E with vertices at (0, 0, 0) and (1, 0, 0).

Proof. We will show by induction on n that there is an $x \in C _ { n - 1 } ^ { n }$ such that $\rho ( x ) = \Delta ^ { n }$ and $d ( x )$ satisfies the face direction condition. The claim will follow since img $( \rho ) \subseteq \Lambda _ { \bullet } ^ { n }$

The base case is $n = 3$ . We notice that the 3 simplex $\Delta ^ { 3 }$ is equivalent in $\widetilde { V } _ { \mathbb { R } ^ { n } }$ to the negation of a paramid X that satisfies the face direction conditions. This is illustrated in Figure 9. The two polytopes give a subdivision of a polytope which is the sum of a triangle and a line segment. Since the triangle and line segment are less than 3 dimensional, this shows that $\Delta ^ { 3 } + X = 0$ in $\widetilde { V } _ { { \mathbb R } ^ { n } }$ . It is straightforward to check the face direction conditions for X. So the class of the base of X (with appropriately chosen orientation) gives an $x \in C _ { 2 } ^ { 3 }$ such that $\rho ( x ) = \Delta ^ { 3 }$

For the inductive step, suppose the claim is true for some $n \geq 3$ . By Theorem 12, there is a cycle $y \in C _ { n - 1 } ^ { n + 1 }$ such that $\rho ( y ) = d ( \Delta ^ { n + 1 } )$ . By Theorem 11, there is some $y ^ { \prime }$ with $\rho ( y ^ { \prime } ) = \rho ( y ) = d ( \Delta ^ { n + 1 } )$ , with $\varepsilon ( y ^ { \prime } ) = 0$ . So by Theorem 10, there is some $z \in C _ { n } ^ { n + 1 }$ with $d ( z ) = y ^ { \prime }$ (so d(z) satisfies the face direction condition).

By Theorem 9, we have $d ( \rho ( z ) ) = \rho ( d ( z ) ) = \rho ( y ^ { \prime } ) = d ( \Delta ^ { n + 1 } )$ . By Theorem 3, this means that $\rho ( z ) = \Delta ^ { n + 1 }$ . By induction, this completes the proof. □

## 5. Additional Ideas

This section contains some results and alternate proofs that did not fit into the main body of this part of the paper. We believe that these ideas can be valuable intuition for understanding our result.

5.1. The Universality of Paramids. Haase, Hertrich, and Loho show that $\mathrm { R e L U } _ { n , 2 }$ is generated by support functions of polytopes that are each the convex hull of two zonotopes (see Theorem 8 in [15]). By Propositions 9 and 10 in [15] and Theorem 2, such a polytope can be written in terms of polytopes of form conv $( A \cup B )$ , where $\operatorname { c o n v } ( A \cup B )$ is a join of A and B (i.e. A and B have disjoint afine hulls and their tangent spaces are transverse). Working modulo lower-dimensional pieces, we can ignore pieces except those with $\dim ( A ) + \dim ( B ) = n - 1$

When A or B is a point, the piece we get is a paramid. The purpose of this section is to show that, modulo lower-dimensional terms, every such piece is equivalent to a paramid. This somewhat justifies the choice to focus on paramids with our proof.

We write $\alpha : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ for the antipodal map. The key result is the following:

Lemma 14. Let A and B be polytopes in $\mathbb { R } ^ { n } \ ( n \geq 1 )$ with disjoint afine hulls. Suppose A can be written $A _ { 1 } + A _ { 2 }$ , where $A _ { 1 }$ and $A _ { 2 }$ have transverse tangent spaces. Then in $\widetilde { V } _ { \mathbb { R } ^ { n } }$ 2 the class of conv(A ∪ B) is equal to $( - 1 ) ^ { \dim ( A _ { 2 } ) }$ times the class of conv $( A _ { 1 } \cup ( B + \alpha ( A _ { 2 } ) ) )$ .

Proof. In this proof, we use the convention that a zero-dimensional point has one −1- dimensional facet, which is the empty set. The Minkowski sum of any polytope with the empty set is empty. The way we use the induction hypothesis, we will sometimes need a version of the claim where $A _ { 1 }$ or B empty. We will check this case first.

If both $A _ { 1 }$ and B are empty, then both conv $( A \cup B )$ and conv $\cdot ( A _ { 1 } \cup ( B + \alpha ( A _ { 2 } ) ) )$ are empty, so the claim is trivial. If $A _ { 1 }$ is empty and B is not, then conv $\neg ( A \cup B ) = B$ and conv $( A _ { 1 } \cup ( B + A _ { 2 } ) ) = B + A _ { 2 }$ . Since $A _ { 2 }$ and B have dimension $< n ,$ , these both represent 0 in $\widetilde { V } _ { { \mathbb R } ^ { n } }$

We also handle the case $\mathrm { d i m } ( A _ { 2 } ) = 0$ separately. In this case, $A _ { 2 }$ is a point, and it is clear that conv $\cdot ( A _ { 1 } \cup ( B + \alpha ( A _ { 2 } ) ) )$ is just $\operatorname { c o n v } ( A \cup B )$ translated by $- A _ { 2 }$

Now suppose that A and B are non-empty and $A _ { 2 }$ has dimension at least 1. The proof is by induction on n. When $n = 1$ , the conditions of the lemma are impossible to satisfy, and so the result is true trivially.

We want to use Theorem 4. Since A and B have disjoint afine hulls, the facets of conv(A ∪ B) are of form conv $( F \cup B )$ or conv(A ∪ G), where F is a facet of A or $G$ is a facet of B.

Since $A _ { 1 }$ and $A _ { 2 }$ have transverse tangent spaces, the facets of $A = A _ { 1 } + A _ { 2 }$ are of form $F _ { 1 } + A _ { 2 }$ or $A _ { 1 } + F _ { 2 }$ , where $F _ { 1 }$ is a facet of $A _ { 1 }$ or $F _ { 2 }$ is a facet of $A _ { 2 }$ . So the facets of conv(A ∪ B) are:

• for each facet $G$ of B, conv $( A \cup G )$

• for each facet $F _ { 1 }$ of $A _ { 1 }$ , conv $( ( F _ { 1 } + A _ { 2 } ) \cup B )$ , and

• for each facet $F _ { 2 }$ or $A _ { 2 } .$ , conv $( ( A _ { 1 } + F _ { 2 } ) \cup B )$

Since A and B have transverse tangent spaces, B and $A _ { 2 }$ do as well. So by a similar argument to the above, the facets of conv $( A _ { 1 } \cup ( B + \alpha ( A _ { 2 } ) ) )$ are:

• for each facet $F _ { 1 }$ of $A _ { 1 }$ , conv $( F _ { 1 } \cup ( B + \alpha ( A _ { 2 } ) ) )$

• for each facet G of $B , \operatorname { c o n v } ( A _ { 1 } \cup ( G + \alpha ( A _ { 2 } ) ) )$

• for each facet $F _ { 2 }$ of $A _ { 2 } , \mathrm { c o n v } ( A _ { 1 } \cup ( B + \alpha ( F _ { 2 } ) ) )$

By the inductive hypothesis, the class (in $\widetilde { V } _ { S }$ , where S contains $A _ { 1 } , G$ and $A _ { 2 } )$ of conv $( A _ { 1 } \cup$ $\left( G + \alpha ( A _ { 2 } ) \right) )$ is equal to $( - 1 ) ^ { \dim ( A _ { 2 } ) }$ times the class conv $( A \cup G )$ . Similarly, the class of conv $( F _ { 1 } \cup ( B + \alpha ( A _ { 2 } ) ) )$ ) is equivalent to $( - 1 ) ^ { \dim ( A _ { 2 } ) }$ times the class of conv $( ( F _ { 1 } + A _ { 2 } ) \cup B )$ The dimension of $F _ { 2 }$ is one less than the dimension of $A _ { 2 }$ , so the class of conv $\cdot ( A _ { 1 } \cup ( B +$ $\alpha ( F _ { 2 } ) ) ) \ \mathrm { i s } \ - ( - 1 ) ^ { \dim ( A _ { 2 } ) }$ times the class conv $( ( A _ { 1 } + F _ { 2 } ) \cup B )$

We notice that the outward facing normal vector of conv $( A _ { 1 } \cup ( G + \alpha ( A _ { 2 } ) ) )$ is the same as the one for $\operatorname { c o n v } ( A \cup G )$ , and similarly for conv $\boldsymbol { \cdot } ( F _ { 1 } \cup ( B + \alpha ( A _ { 2 } ) ) )$ and conv $( ( F _ { 1 } + A _ { 2 } ) \cup B )$ while conv $\cdot ( A _ { 1 } \cup ( B + \alpha ( F _ { 2 } ) ) )$ ) and conv $( ( A _ { 1 } + F _ { 2 } ) \cup B )$ have opposite normal vectors. By Theorem 4, this completes the proof. □

In particular, if A and B are zonotopes, then the class of conv $( A \cup B )$ is, up to lower dimensional terms, equal to $( - 1 ) ^ { \dim ( A ) }$ times the class of conv $( \{ 0 \} , B + \alpha ( A ) )$ . The latter polytope is a paramid.

So, up to lower dimensional terms, support functions of paramids generate $\mathrm { R e L U } _ { n , 2 }$ . Of course, this follows from Theorem 13 as well.

Remark. Even if polytopes A and B in $\mathbb { R } ^ { n }$ are equivalent modulo terms of dimension $< n$ it is not true in general that pyramids on A and B in $\mathbb { R } ^ { n + 1 }$ are equivalent modulo terms of dimension $< n + 1$ . So Theorem 14 is not so useful for inductive arguments.

5.2. The Koszul Complex. The chain complex $C _ { \bullet } ^ { n }$ that we define is also known as the Koszul complex of the sequence $t _ { 1 } - 1 , \ldots , t _ { r } - 1$ , viewing $t _ { 1 } - 1 , \ldots , t _ { r } - 1$ as elements of $\mathbb { Q } [ \Gamma ]$

Formally, given a ring U and a set of elements $x _ { 1 } , \ldots , x _ { r }$ , the associated Koszul complex $K _ { \bullet }$ is defined by $K _ { p } = { \textstyle \bigwedge } _ { U } ^ { p } U ^ { r }$ (so $K _ { 0 } = U )$ . Viewing $U ^ { r }$ as being generated by $e _ { 1 } , \ldots , e _ { r }$ the diferential is defined by $d ( e _ { i } ) = x _ { i }$ and graded Leibniz. Note that, if $U$ is an Q-module $( \mathrm { e . g . ~ } \mathbb { Q } [ \Gamma ] )$ , then $K _ { p }$ is naturally isomorphic to $U \otimes _ { \mathbb { Q } } \bigwedge _ { \mathbb { Q } } ^ { p } \mathbb { Q } ^ { r }$

A sequence $F = ( f _ { 1 } , \ldots , f _ { r } )$ is called a regular sequence if the ideal $\left( f _ { 1 } , \ldots , f _ { r } \right)$ is not equal to U and, for each $i , f _ { i }$ is not a zero-divisor in $R / ( f _ { 1 } , \dots , f _ { i - 1 } )$ . If $f _ { i } = t _ { i } - 1$ as in Section $^ { 3 , }$ , then regularity says that the translations $t _ { i }$ freely generate an abelian subgroup of $\mathbb { Q } ^ { n }$

The homology of a Koszul complex of a regular sequence is well understood:

Theorem 15. (See $\begin{array} { l l } { { e . g . } } & { { [ 3 5 ] , } } \end{array}$ Corollary 4.5.5) Let U be a ring, let $f _ { 1 } , \ldots , f _ { r }$ be a regular sequence, and let $K _ { \bullet }$ be the Koszul complex. Then $H _ { 0 } ( K _ { \bullet } ) = U / ( f _ { 1 } , \dots , f _ { r } )$ , and all higher homology is zero.

5.3. The Action of S on $H _ { \bullet } ( C _ { \bullet } ^ { n } )$ . Let S be the doubling map defined in the proof of Theorem 11 (see also Figure 5). Instead of using the augmentation condition for cycle filing in $C _ { \bullet } ^ { n }$ , we could instead try to compute the action of $S$ on $H _ { \bullet } ( C _ { \bullet } )$ . Note that $S$ computes with the boundary on $C _ { \bullet } ^ { n }$ , so it induces a map $S _ { * } : H _ { \bullet } ( C _ { \bullet } ^ { n } )  H _ { \bullet } ( C _ { \bullet } ^ { n } )$

Recall from Section 3.1 that $C _ { \bullet } ^ { n }$ is the quotient of the unit-cubical complex $\hat { C } _ { \bullet } ^ { n }$ of $\mathbb { R } ^ { r }$ by the subgroup of $\mathbb { Z } ^ { r }$ representing relations among the simplex edge directions $t _ { 1 } , \ldots , t _ { r }$ . Now S lifts to a map $\hat { C } _ { \bullet } ^ { n } \to \hat { C } _ { \bullet } ^ { n }$ in a straightforward way.

Note that $\hat { C } _ { \bullet } ^ { n }$ is a cubical complex for $\mathbb { R } ^ { r }$ and $C _ { \bullet } ^ { n }$ is a cubical complex for the product $\mathbb { R } ^ { n } \times ( S ^ { 1 } ) ^ { r - n }$ . The map $\pi : \mathbb { R } ^ { r }  \mathbb { R } ^ { n } \times ( S ^ { 1 } ) ^ { r - n }$ is the universal covering map of $\mathbb { R } ^ { n } \times ( S ^ { 1 } ) ^ { r - n }$ Now S gives a map $\mathbb { R } ^ { n } \times ( S ^ { 1 } ) ^ { r - n } \to \mathbb { R } ^ { n } \times ( S ^ { 1 } ) ^ { r - n }$ , which lifts to a map $S : \mathbb { R } ^ { r }  \mathbb { R } ^ { r }$ 2 which is given by multiplication by 2.

Now $\pi _ { 1 } ( \mathbb { R } ^ { n } \times ( S ^ { 1 } ) ^ { r - n } )$ is naturally associated with the group of deck transformations of $\mathbb { R } ^ { r }$ which is the kernel of $\pi : \hat { \Gamma } \to \Gamma$ . So we see that $S _ { * } : \pi _ { 1 } ( \mathbb { R } ^ { n } \times ( S ^ { 1 } ) ^ { r - n } ) \to \pi _ { 1 } ( R ^ { n } \times ( S ^ { 1 } ) ^ { r - n } )$ is multiplication by 2.

So $S _ { * } : H _ { 1 } ( C _ { \bullet } ^ { n } )  H _ { 1 } ( C _ { \bullet } ^ { n } )$ is multiplication by 2. By functoriality of the product structure on the (co)homology of $( S ^ { 1 } ) ^ { r - n }$ , we see that $S _ { * } : H _ { k } ( C _ { \bullet } )  H _ { k } ( C _ { \bullet } )$ is multiplication by $2 ^ { k }$ This gives us a diferent way of seeing that, for a cycle $x \in C _ { k } ^ { n } , 2 ^ { k } x - S ( x )$ is a boundary.

5.4. An Alternative Proof of Theorem 10. We can make $D _ { \bullet } ^ { n }$ into a chain complex with boundary map 0. The map $\varepsilon : C _ { \bullet } ^ { n } \to D _ { \bullet } ^ { n }$ commutes with the boundary maps, so descends to a map ε<sub>∗</sub> on homology. Since the boundary on $D _ { \bullet } ^ { n }$ is 0, the homology of $D _ { \bullet } ^ { n }$ is just $D _ { \bullet } ^ { n }$

Recall that Theorem 10 says that $\varepsilon : C _ { \bullet } ^ { n } \to D _ { \bullet } ^ { n }$ sends a cycle x to 0 if and only x is a boundary. This is equivalent to saying that $\varepsilon _ { * } : H _ { \bullet } ( C _ { \bullet } ^ { n } )  H _ { \bullet } ( D _ { \bullet } ^ { n } )$ is injective.

Now $D _ { \bullet } ^ { n }$ is a cubicalization of the quotient of $\mathbb { R } ^ { r }$ by the action of $\mathbb { Z } ^ { r }$ . Recall that $C _ { \bullet } ^ { n }$ is the cubicalization of the quotient of $\mathbb { R } ^ { r }$ by the group of relations among the $t _ { i } ,$ call this group G. Now ε corresponds to the natural covering map $\mathbb { R } ^ { r } / G \to \mathbb { R } ^ { r } / \mathbb { Z } ^ { n }$

Now $H _ { k } ( C _ { \bullet } ^ { n } )$ is identified with $\textstyle \bigwedge _ { \mathbb { Q } } ^ { k } ( \mathbb { Q } \otimes G )$ and $H _ { k } ( D _ { \bullet } ^ { n } )$ is identified with ${ \textstyle \bigwedge } _ { \mathbb { Q } } ^ { k } ( \mathbb { Q } \otimes \mathbb { Z } ^ { r } )$ , and $\varepsilon _ { * }$ is the map $\land _ { \mathbb { Q } } ^ { k } ( \mathbb { Q } \otimes G )  \land _ { \mathbb { Q } } ^ { k } ( \mathbb { Q } \otimes \mathbb { Z } ^ { r } )$ induced by the inclusion $G \to \mathbb { Z } ^ { r }$ . This map is injective, giving an alternate proof of Theorem 10.

5.5. An Alternate Proof of Theorem 4. A result of Koutschan, Ponomarchuk, and Schicho [23] gives an exact condition for when a continuous piecewise linear function $f$ in n inputs can be written as a linear combination of “lower-dimensional” functions. Here we show how to derive Theorem 4 from their result. The results of [23] are stated over R, so in this section we work with R coeficients instead of $\mathbb { Q } .$

The results of [23] are stated over the reals. We expect that the results also hold with rational coeficients, but in this section we will allow arbitrary real coeficients.

Let $S$ and Q be finite dimensional real vector spaces, and let $g : S  Q$ be a piecewise constant function. We allow $g$ to be defined only “almost everywhere” (precisely, we say that for “almost all” $x \in S , g ( x )$ is defined and constant on a neighborhood of $x )$ . Let $H \subseteq S$ be an oriented codimension 1 subspace. For $x \in H$ and $\epsilon > 0$ , let $B _ { \epsilon } ^ { + } ( x )$ and $B _ { \epsilon } ^ { - } ( x )$ be the two halves of the ball of radius ϵ around $x ,$ split by $H$ , with $B _ { \epsilon } ^ { + } ( x )$ being the half on the positive side of $H$ . For almost all $x \in H , g$ is constant on $B _ { \epsilon } ^ { + } ( x )$ and $B _ { \epsilon } ^ { - } ( x )$ . We define $\Delta _ { g } ( H ) : H  Q$ to be the piecewise constant, almost everywhere defined function that is the diference of the values of $g$ on $B _ { \epsilon } ^ { + }$ and $B _ { \epsilon } ^ { - }$ <sup>−</sup>, when this exists.

An oriented flag $\mathcal { H } = ( H _ { 1 } , \ldots , H _ { k } )$ of length k in $S$ is a sequence of oriented codimension 1 afine subspaces $H _ { 1 } \subseteq S$ and $H _ { i + 1 } \subseteq H _ { i }$ . For $g : S  Q$ piecewise constant and almost everywhere defined, we define $\Delta _ { g } ( \mathcal { H } )$ inductively as follows: we define $g _ { 1 } = \Delta _ { g } ( H _ { 1 } )$ and $g _ { i + 1 } = \Delta _ { g _ { i } } ( H _ { i + 1 } )$ , and set $\Delta _ { g } ( \mathcal { H } ) = g _ { k }$ . We say that H is critical for g if $g _ { k }$ is non-zero.

Let $\mathcal { H } = ( H _ { 1 } , \ldots , H _ { k } )$ be a flag, and let $v _ { 1 } , \ldots , v _ { k }$ be vectors such that $v _ { i }$ is the normal vector to $H _ { i }$ in $H _ { i - 1 }$ , viewing $H _ { 0 }$ as the base space S. Then, where $\Delta _ { g } ( \mathcal { H } )$ is defined, it is equal to:

$$
\Delta _ { g } ( \mathcal { H } ) ( x ) = \operatorname* { l i m } _ { \epsilon _ { k }  0 ^ { + } } \cdot \cdot \cdot \operatorname* { l i m } _ { \epsilon _ { 1 }  0 ^ { + } } \sum _ { ( s _ { 1 } , \ldots , s _ { k } ) \in \{ - 1 , 1 \} ^ { k } } g ( x + s _ { 1 } \epsilon _ { 1 } v _ { 1 } + \cdot \cdot \cdot + s _ { k } \epsilon _ { k } v _ { k } ) \cdot \prod _ { i = 1 } ^ { k } s _ { k }
$$

If $f : \mathbb { R } ^ { n }  \mathbb { R }$ is a continuous piecewise linear function, then the gradient $\nabla f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ is a piecewise constant, almost everywhere defined function. The function $f$ is said to be a max<sub>k</sub> if it can be written $f = \operatorname* { m a x } ( \ell _ { 1 } , \dots , \ell _ { k } )$ , where each $\ell _ { i }$ is an afine function. The following is proved in [23]:

Theorem 16. $( / 2 3 ] _ { ; }$ Theorem 1) Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be a continuous piecewise linear function. Then we can express $f$ as a linear combination of $\operatorname* { m a x } _ { k + 1 }$ functions if and only if every critical flag for $\nabla f$ has $l e n g t h \le k$

A piecewise linear function is said to be homogeneous if every piece extends to a linear function (i.e. an afine function that is 0 at 0). Given a polytope X in $\mathbb { R } ^ { n }$ , its support function $h _ { X } : \mathbb { R } ^ { n } $ R is defined by $h _ { X } ( v ) = \operatorname* { m a x } _ { x \in X } ( v \cdot x )$ . This is a homogeneous piecewise linear function. There is a linear map from $V _ { S }$ to the space of homogeneous piecewise linear functions on S that sends $( X , Y )$ to $( h _ { X } , h _ { Y } )$ . This is an isomorphism.

A homogeneous max $\tau _ { k + 1 }$ (that is, a max of k linear functions) is the support function of a polytope with $k + 1$ vertices, which is at most k dimensional. So Theorem 16 is close to giving a characterization of $\widetilde { V } _ { S }$ . We need one more lemma:

Lemma 17. $I f f : \mathbb { R } ^ { n }  \mathbb { R }$ is a homogeneous piecewise linear function that can be written as a sum max<sub>k</sub> functions, then f can be written as a sum of homogeneous max<sub>k</sub> functions.

Proof. We write:

$$
f + \sum _ { i = 1 } ^ { k } a _ { i } - \sum _ { i = 1 } ^ { j } b _ { i } = 0
$$

Where each $a _ { i }$ and $b _ { i }$ is a max<sub>k</sub> term. Now obtain $a _ { i } ^ { \prime }$ from $a _ { i }$ by adding a constant so that $a _ { i } ^ { \prime } ( 0 ) = 0$ . Since $f ( 0 ) = 0$ , all the constant terms added must cancel, so:

$$
h _ { X } - h _ { Y } + \sum _ { i = 1 } ^ { j } a _ { i } ^ { \prime } - \sum _ { i = 1 } ^ { m } b _ { i } ^ { \prime } = 0
$$

Now each $a _ { i } ^ { \prime }$ or $b _ { i } ^ { \prime }$ is a max of n terms, plus a constant. Write $a _ { 1 } ^ { \prime } = c _ { 0 } + \operatorname* { m a x } ( \ell _ { 1 } + c _ { 1 } , \ldots , \ell _ { n } +$ $c _ { n } ) = c _ { 0 } -$ +max $\cdot ( \ell _ { 1 } + c _ { 0 } + c _ { 1 } , \ldots , \ell _ { n } + c _ { 0 } + c _ { n } )$ , where each $\ell _ { i }$ is linear and each $c _ { i }$ is a constant. We know that $a _ { 1 } ( 0 ) = 0$ , so all the $c _ { 0 } + c _ { i }$ are $\leq 0$ . Obtain $a _ { 1 } ^ { \prime \prime }$ from $a _ { 1 }$ by removing all the $\ell _ { i } + c _ { 0 } + c _ { i }$ terms that have $c _ { 0 } + c _ { i } < 0$ . Note that $a _ { 1 } ^ { \prime \prime } = a _ { 1 } ^ { \prime }$ in a neighborhood of 0. We similarly define $a _ { i } ^ { \prime \prime }$ and $b _ { i } ^ { \prime \prime }$ for each $a _ { i }$ , and notice that:

$$
f + \sum _ { i = 1 } ^ { j } a _ { i } ^ { \prime \prime } - \sum _ { i = 1 } ^ { m } b _ { i } ^ { \prime \prime } = 0
$$

in a neighborhood of 0. However, both sides are homogeneous, and a homogeneous piecewise linear function is determined by its behavior near 0, so both sides are equal. □

We now have a characterization of $\widetilde { V } _ { S }$ . Let $( X , Y )$ represent a class in $\widetilde { V } _ { S }$ and let $g =$ $\nabla ( h _ { X } - h _ { Y } )$ . Then $( X , Y ) = 0$ in $\widetilde { V } _ { S } \ \mathrm { i f } .$ , for every flag H of length $n , \Delta _ { g } ( \mathcal { H } ) = 0$ . The converse is also true, but this won’t be relevant for our approach.

We say that a flag $\mathcal { H } ( H _ { 1 } , \ldots , H _ { k } )$ is homogeneous if every $H _ { i }$ is a vector space (i.e. it contains 0). The following shows that, if we want to apply Theorem 16 to a homogeneous function, then we only need to consider homogeneous flags.

Lemma 18. If f is a homogeneous piecewise linear function, $g = \nabla f$ and $\mathcal { H }$ is a $f l a g$ then $\Delta _ { g } ( \mathcal { H } ) = 0$ unless H is homogeneous.

Proof. Straightforward.

Lemma 19. Let X be a polytope in $\mathbb { R } ^ { n }$ . Let $\mathcal { H } = \{ H _ { 1 } , \ldots , H _ { n } \}$ be a homogeneous flag (so $H _ { n } \ = \ 0 )$ . Let $v _ { 1 } , \ldots , v _ { n }$ be vectors such that $v _ { i }$ is a normal vector to $H _ { i }$ in $H _ { i - 1 }$ , viewing $H _ { 0 } = \mathbb { R } ^ { n }$ . Let $F _ { 1 }$ and $F _ { 2 }$ be the faces of X in directions $v _ { n }$ and $- v _ { n }$ respectively. Let S be the orthogonal complement $o f v _ { n }$ in $\mathbb { R } ^ { n }$ $F o r i = 1 , \dots , n - 1$ , let $H _ { i } ^ { \prime }$ be the orthogonal complement of $v _ { n }$ in $H _ { i } ,$ so $\mathcal { H } ^ { \prime } = \{ H _ { 1 } ^ { \prime } , \ldots , H _ { n - 1 } ^ { \prime } \}$ is a flag on S. Let $f = \nabla h _ { X }$ and let $f ^ { \prime } = \nabla h _ { F _ { 1 } } - \nabla h _ { F _ { 2 } }$ , where we restrict $f ^ { \prime }$ to a function $S \to \mathbb { R } ^ { n }$ . Then $\Delta _ { f } ( \mathcal { H } ) ( 0 ) = \Delta _ { f ^ { \prime } } ( \mathcal { H } ^ { \prime } ) ( 0 )$

Proof. We know that:

$$
\Delta _ { f } ( \mathcal { H } ) ( 0 ) = \operatorname* { l i m } _ { \epsilon _ { n } \to 0 ^ { + } } \cdot \cdot \cdot \operatorname* { l i m } _ { \epsilon _ { 1 } \to 0 } \sum _ { ( s _ { 1 } , \ldots , s _ { n } ) \in \{ - 1 , 1 \} ^ { n } } g ( s _ { 1 } \epsilon _ { 1 } v _ { 1 } + \cdot \cdot \cdot + s _ { n } \epsilon _ { n } v _ { n } ) \cdot \prod _ { i = 1 } ^ { n } s _ { i }
$$

and:

$$
\Delta _ { f ^ { \prime } } ( \mathcal { H } ^ { \prime } ) ( 0 ) = \operatorname* { l i m } _ { \epsilon _ { n - 1 }  0 ^ { + } } \cdot \cdot \cdot \operatorname* { l i m } _ { \epsilon _ { 1 }  0 ^ { + } } \sum _ { ( s _ { 1 } , \ldots , s _ { n - 1 } ) \in \{ - 1 , 1 \} ^ { n - 1 } } f ^ { \prime } ( s _ { 1 } \epsilon _ { 1 } v _ { 1 } + \cdot \cdot \cdot + s _ { n - 1 } \epsilon _ { n - 1 } v _ { n - 1 } ) \cdot \prod _ { i = 1 } ^ { n - 1 } s _ { i }
$$

Note that $h _ { X } , h _ { F _ { 1 } }$ and $h _ { F _ { 2 } }$ are homogeneous, so $f ( \lambda x ) = f ( x )$ for $\lambda > 0$ , and similarly for $f ^ { \prime }$ . We observe that, for almost all $x \in S$

$$
\nabla h _ { F _ { 1 } } ( x ) = \operatorname* { l i m } _ { \epsilon  0 ^ { + } } f ( v _ { n } + \epsilon x )
$$

To see this, we notice that, for v a direction, $\nabla f ( \boldsymbol { v } )$ is equal to the vertex u of $X$ that maximizes $\boldsymbol { u } \cdot \boldsymbol { v } \ ( \nabla f ( \boldsymbol { x } )$ is defined when this is unique). Now $F _ { 1 }$ is the set of points u in X maximizing $u \cdot v _ { n }$ . So when x is suficiently small, $f ( v _ { n } + x )$ is the vertex u of $F _ { 1 }$ maximizing $u \cdot x ,$ which is equal to $h _ { F _ { 1 } } ( x )$

Similarly, for almost all $x \in S$ , we have:

$$
\nabla h _ { F _ { 2 } } ( x ) = \operatorname* { l i m } _ { \epsilon  0 ^ { + } } f ( - v _ { n } + \epsilon x )
$$

So for almost all $x \in S$ , we have:

$$
f ^ { \prime } ( x ) = \operatorname* { l i m } _ { \epsilon  0 ^ { + } } ( f ( v _ { n } + \epsilon x ) - f ( - v _ { n } + \epsilon _ { x } ) )
$$

It follows that $\Delta _ { f } ( \mathcal { H } ) ( 0 ) = \Delta _ { f ^ { \prime } } ( \mathcal { H } ^ { \prime } ) ( 0 )$

Proof of the $^ { \ast } i f ^ { \ast }$ direction of Theorem 4. Let X and Y be polytopes in $\mathbb { R } ^ { n }$ . Suppose that, for every direction $v , \left( F _ { 1 } + G _ { 2 } , F _ { 2 } + G _ { 1 } \right) = 0$ in $\widetilde { V } _ { S }$ , where $S$ is the orthogonal complement of $v , F _ { 1 }$ and $F _ { 2 }$ are the faces of X in directions v and −v respectively, and $G _ { 1 }$ and $G _ { 2 }$ are faces of Y in directions v and −v respectively.

We want to show that there are $A _ { i }$ and $B _ { i }$ such that $X + A _ { 1 } + \cdot \cdot \cdot + A _ { k } = Y + B _ { 1 } + \cdot \cdot \cdot + B _ { i }$ Let $f = \nabla ( h _ { X } - h _ { Y } )$ . By Theorem 16 and Theorems 17 and 18, it is suficient to show that, for any homogeneous flag ${ \mathcal { H } } = \{ H _ { 1 } , \ldots , H _ { n } \}$ , we have $\Delta _ { f } ( \mathscr { H } ) ( 0 ) = 0$ . This is by induction on n.

Let ${ \mathcal { H } } = \{ H _ { 1 } , \ldots , H _ { n } \}$ be a flag and let v be a vector in $H _ { n }$ representing the orientation of $H _ { n - 1 }$ . Let $F _ { 1 } , F _ { 2 } , G _ { 1 }$ and $G _ { 2 }$ be faces of X and Y in directions $\pm v .$ , as in the statement of the lemma, but not translated. Let $f ^ { \prime } = \nabla ( h _ { F _ { 1 } } + h _ { G _ { 2 } } - h _ { F _ { 2 } } - h _ { G _ { 1 } } )$ . By Theorem 19, we have $\Delta _ { f ^ { \prime } } ( \mathcal { H } ^ { \prime } ) ( 0 ) = \Delta _ { f } ( \mathcal { H } ) ( 0 )$ , where $\mathcal { H ^ { \prime } }$ is the flag obtained as in Theorem 19.

Let $F _ { 1 } ^ { \prime } , F _ { 2 } ^ { \prime } , G _ { 1 } ^ { \prime }$ and $G _ { 2 } ^ { \prime }$ be $F _ { 1 }$ etc. translated to lie in the subspace S orthogonal to v. Let $f ^ { \prime \prime } = \nabla ( h _ { F _ { 1 } ^ { \prime } } + h _ { G _ { 2 } ^ { \prime } } - h _ { F _ { 2 } ^ { \prime } } - h _ { G _ { 1 } ^ { \prime } } )$ . Translating a polytope changes its support function by adding a linear function, so $f ^ { \prime \prime }$ difers from f<sup>′</sup> by a constant. Since $\mathcal { H ^ { \prime } }$ has length $\geq n - 1 \geq 1$ , we have $\Delta _ { f ^ { \prime \prime } } ( \mathcal { H } ^ { \prime } ) ( 0 ) = \Delta _ { f ^ { \prime } } ( \mathcal { H } ^ { \prime } ) ( 0 )$

We know that $( F _ { 1 } ^ { \prime } + G _ { 2 } ^ { \prime } , F _ { 2 } ^ { \prime } + G _ { 1 } ^ { \prime } ) = 0$ in $\widetilde { V } _ { S }$ by assumption. In the base case $n = 2$ , this means that $F _ { 1 } ^ { \prime } + \bar { G _ { 2 } ^ { \prime } } , F _ { 2 } ^ { \prime } \bar { + } \bar { G _ { 1 } ^ { \prime } }$ are translates of each other, so $f ^ { \prime \prime }$ is constant and $\Delta _ { f ^ { \prime \prime } } ( \mathcal { H } ^ { \prime } ) = 0$ When $n > 2$ , the “only $\mathrm { i f } ^ { \dag }$ part of Theorem 4 shows that $( F _ { 1 } ^ { \prime } + G _ { 2 } ^ { \prime } , F _ { 2 } ^ { \prime } + G _ { 1 } ^ { \prime } )$ satisfies the conditions of the lemma. So we know by inductive assumption that $\Delta _ { f ^ { \prime \prime } } ( \mathcal { H } ^ { \prime } ) ( 0 ) = 0$ . In either case, we conclude that $\Delta _ { f } ( \mathcal { H } ) ( 0 )$ , so $h _ { X } - h _ { Y }$ can be written in terms of homogeneous $\operatorname* { m a x } _ { n }$ terms, which means that $( X , Y ) = 0$ in $\widetilde { V } _ { \mathbb { R } ^ { n } }$ □

## Part 2. A Direct Combinatorial Proof

Here we explicitly construct a signed sum of paramids whose support functions equal $\mathrm { M A X } _ { n }$ modulo lower-dimensional terms.

## 6. Preliminaries

Throughout this part, let $[ n ] : = \{ 1 , \dots , n \}$ and let

$$
\Delta : = \operatorname { c o n v } \{ e _ { 1 } , \dots , e _ { n } \} \subset H : = \left\{ x \in \mathbb { R } ^ { n } : \sum _ { i } x _ { i } = 1 \right\}
$$

be the standard $( n - 1 )$ )-simplex living in the afine subspace H such that $h _ { \Delta } = \mathrm { M A X } _ { n }$ For every index $v \in [ n ]$ , let

$$
V _ { - v } : = [ n ] \setminus \{ v \} , \qquad H _ { - v } : = \{ p \in H : p _ { v } = 0 \} , \qquad \Delta _ { - v } : = \mathrm { c o n v } \{ e _ { i } : i \in V _ { - v } \} .
$$

Furthermore, let $\scriptstyle { \mathcal { L } } _ { n - 2 }$ be the span of support functions of polytopes in H of dimension at most $n - 2$ . Write $f \equiv g { \mathrm { ~ i f ~ } } f - g \in { \mathcal { L } } _ { n - 2 }$ . We write “almost everywhere” to mean “outside a finite union of lower dimensional spaces”. For a polytope P and a point q, the pyramid with apex q and base P is denoted as

$$
\operatorname { p y r } _ { q } ( P ) : = \operatorname { c o n v } ( \{ q \} \cup P ) .
$$

6.1. Standard Valuations. Theorem 2 gives an inclusion–exclusion identity for support functions of a polytope covered by polytopes. The following lemma uses the same valuation property to handle arbitrary linear relations between indicator functions that hold almost everywhere, yielding an identity of support functions modulo lower-dimensional terms.

Lemma 20 (Valuation). Let $A \subseteq H$ be a d-dimensional afine subspace, let $P _ { 1 } , \ldots , P _ { k } \subseteq A$ be polytopes, and let $c _ { 1 } , \ldots , c _ { k } \in \mathbb { R } . \ I f \sum _ { i = 1 } ^ { k } c _ { i } \mathbf { 1 } _ { P _ { i } } = 0$ almost everywhere on A, then

$$
\sum _ { i = 1 } ^ { k } c _ { i } h _ { { P } _ { i } }
$$

is a linear combination of support functions of polytopes in A of dimension at most $d - 1$ Moreover, for every $q \in H \setminus A$

$$
\sum _ { i = 1 } ^ { k } c _ { i } h _ { \mathrm { p y r } _ { q } ( P _ { i } ) }
$$

is a linear combination of support functions of polytopes in af $\operatorname { \dot { } } ( A \cup \{ q \} )$ of dimension at most d.

Proof. The first assertion follows from full additivity of support functions after a suitable subdivision (see Theorem 2 and $\mathrm { { e . g . \ [ 3 0 ] ) } }$ . For the second, fix $q \in H \setminus A$ and set $B =$ a $\operatorname { f f } ( A \cup \{ q \} )$ . Every section of the pyramids parallel to A, apart from the apex, is a common homothetic image of the original polytopes. Their signed indicators therefore cancel almost everywhere on B. Apply the first assertion in the $( d + 1 )$ -dimensional afine space B. □

## 7. Replacing Simplices by Paramids

7.1. The Parallelepipeds. Here we describe the parallelepipeds that form the bases of the paramids in our construction. For $v \in [ n ]$ , an ordering $\pi = \left( \pi _ { 1 } , \ldots , \pi _ { n - 1 } \right)$ of $V _ { - v } ,$ and a set $M \subseteq \{ 2 , \ldots , n - 1 \}$ of marked indices, define the polytope

$$
B _ { v , \pi , M } : = { \biggl \{ } p \in H _ { - v } : \quad 0 \leq u _ { i } \leq 1 \quad \forall i \in M ,  \\  - 1 \leq p _ { \pi _ { i } } \leq 0 \quad \forall i \in \{ 2 , \ldots , n - 1 \} \setminus M { \biggr \} } ,\tag{3}
$$

where $\begin{array} { r } { u _ { i } = \sum _ { j < i } p _ { \pi _ { j } } } \end{array}$

Claim 21. The set $B _ { v , \pi , M }$ is a parallelepiped.

Proof. Since $p _ { v } = 0$ and $\textstyle \sum _ { i } p _ { i } = 1$ , the coordinates $p _ { \pi _ { 2 } } , \ldots , p _ { \pi _ { n - 1 } }$ determine p uniquely. The change of coordinates

$$
t _ { i } = \left\{ { \begin{array} { l l } { u _ { i } = 1 - \sum _ { j = i } ^ { n - 1 } p _ { \pi _ { j } } } & { i \in M , } \\ { - p _ { \pi _ { i } } } & { i \notin M } \end{array} } \right.
$$

is triangular with diagonal entries −1, and hence invertible. In these coordinates, $B _ { v , \pi , M }$ is the cube $[ 0 , 1 ] ^ { n - 2 }$ □

7.2. The Replacement Lemma. This is the main step that allows us to replace simplices by paramids. For $v \in [ n ]$ and every nonempty $R \subseteq V _ { - v }$ , define the dilated simplex

$$
Q _ { v , R } = ( n - | R | ) \Delta _ { - v } - \sum _ { i \notin { R \cup \{ v \} } } e _ { i } \subset H _ { - v } ,
$$

a dilated and translated copy of the facet $\Delta _ { - v }$ . Recall Figure 7.

Lemma 22. For each $v \in [ n ]$ , the following identity holds:

$$
\sum _ { \pi } \sum _ { M \subseteq \{ 2 , \ldots , n - 1 \} } ( - 1 ) ^ { | M | } h _ { \mathrm { p y r } _ { e _ { v } } ( B _ { v , \pi , M } ) } \equiv ( n - 1 ) ! \sum _ { \emptyset \neq R \subseteq V _ { - v } } \frac { ( - 1 ) ^ { | R | - 1 } | R | } { n - | R | } h _ { \mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } ) } ,
$$

where π ranges over all orderings of $V _ { - v }$

7.3. The Thermostat Cancellation. Fix $v \in [ n ]$ . We prove that the signed sum of indicator functions over the parallelepipeds $B _ { v , \pi , M }$ defined in Eq. (3) has a remarkably simple expression, when summing over all permutations $\pi$ and sets of markings M, that only depends on the number of positive entries of the argument.

Lemma 23. For every $v \in [ n ]$ and almost every p in $H _ { - v }$

$$
\sum _ { \pi } \sum _ { M \subseteq \{ 2 , \ldots , n - 1 \} } ( - 1 ) ^ { | M | } \mathbf { 1 } _ { B _ { v , \pi , M } } ( p ) = { \left\{ \begin{array} { l l } { ( - 1 ) ^ { s - 1 } s ! ( n - 1 - s ) ! } & { p _ { i } > - 1 \ f o r \ a l l \ i \in V _ { - v } , } \\ { 0 } & { o t h e r w i s e , } \end{array} \right. }
$$

where $s = | \{ i \in V _ { - v } : p _ { i } > 0 \} |$ and π ranges over all orderings of $V _ { - v }$

Proof. We may assume that no coordinate is 0 or −1, and no proper partial sum of the coordinates is 0 or 1, as this excludes only a finite number of lower dimensional hyperplanes. Fix an ordering π. For every M, membership in $B _ { v , \pi , M }$ requires $0 < u _ { i } < 1$ for $i \in M$ and $- 1 < p _ { \pi _ { i } } < 0$ for $i \notin M$ . Summing over all M,

$$
\sum _ { M } ( - 1 ) ^ { | M | } \mathbf 1 _ { B _ { v , \pi , M } } ( p ) = \prod _ { i = 2 } ^ { n - 1 } \left( \mathbf 1 _ { \{ - 1 < p _ { \pi _ { i } } < 0 \} } - \mathbf 1 _ { \{ 0 < u _ { i } < 1 \} } \right) .
$$

Interpret $p _ { \pi _ { 1 } } , \ldots , p _ { \pi _ { n - 1 } }$ as steps of a walk whose levels are $u _ { 1 } = 0 , u _ { 2 } , \ldots , u _ { n - 1 } , u _ { n } = 1$ . As we describe next, for the product above to be nonzero, the walk must satisfy the thermostat properties: it is positive, rises below 1 and drops above 1.

$$
( p _ { 1 } = 0 . 3 , p _ { 2 } = 0 . 9 , p _ { 4 } = 0 . 3 , p _ { 6 } = 0 . 6 ) ~ ( p _ { 3 } = - 0 . 4 , p _ { 5 } = - 0 . 7 )
$$

![](images/49a7792b36c94c2fcb2ba3588f71d96bdf87cc15b49e9b3708c8091ef302c6fc.jpg)  
Figure 10. A thermostat walk obtained by merging the ordered positive and negative coordinate lists shown above it. Take the next positive step below level 1 and the next negative step above level 1.

The walk cannot reach a negative level because once it does, all subsequent permitted steps are negative, so it cannot finish at level 1. It follows that inside (0, 1) the walk must rise (otherwise, it becomes negative). Above 1 it must fall by less than 1. The first step must be positive as well. In particular, all coordinates exceed −1.

Now, assume that every coordinate exceeds −1. Order the s positive coordinates and the $n - 1 - s$ negative coordinates separately. There is a unique way to merge these two lists into a walk of the required type; see Figure 10. Take a positive step first. Take the next positive step whenever the current level is below 1, and the next negative step whenever it is above 1. The required step is always available. Indeed, the remaining steps sum to $1 - u _ { i }$ , which is positive below 1 and negative above 1. Negative steps have size less than 1, so the walk stays positive. The genericity assumption prevents reaching level 1 before the final step.

We obtained a bijection between nonzero products and pairs of orderings of the positive and negative coordinates. There are $s ! ( n - 1 - s ) !$ ! pairs of orderings. Every nonzero product has sign $( - 1 ) ^ { s - 1 }$ , one (−1) factor for each positive step after the first. □

## 7.4. Proof of Theorem 22.

Proof. Fix a generic $p \in H _ { - v }$ . First we prove

$$
\sum _ { \pi } \sum _ { M } ( - 1 ) ^ { | M | } \mathbf { 1 } _ { B _ { v , \pi , M } } = ( n - 1 ) ! \sum _ { \emptyset \neq { \cal R } \subseteq V _ { - v } } \frac { ( - 1 ) ^ { | { \cal R } | - 1 } | { \cal R } | } { n - | { \cal R } | } \mathbf { 1 } _ { Q _ { v , { \cal R } } } .\tag{4}
$$

If some coordinate is less than −1, both sides of (4) vanish, by Theorem 23 and the definition of $Q _ { v , R }$ . Otherwise, let s be the number of positive coordinates. The point p belongs to $Q _ { v , R }$ exactly when R is contained in the set of its positive coordinates. Thus, for each r, exactly $\binom { s } { r }$ sets R of size r contribute. To evaluate the right-hand side, the hockey-stick identity gives, for $1 \leq r \leq n - 1$ 2

$$
\begin{array} { l } { \displaystyle \sum _ { j = 1 } ^ { r } \binom { r } { j } j ! ( n - 1 - j ) ! = r ! ( n - 1 - r ) ! \displaystyle \sum _ { j = 1 } ^ { r } \binom { n - 1 - j } { r - j } } \\ { = r ! ( n - 1 - r ) ! \binom { n - 1 } { r - 1 } = \displaystyle \frac { ( n - 1 ) ! r } { n - r } . } \end{array}
$$

By binomial inversion, the right-hand side is therefore

$$
( n - 1 ) ! \sum _ { r = 1 } ^ { s } ( - 1 ) ^ { r - 1 } { \binom { s } { r } } { \frac { r } { n - r } } = ( - 1 ) ^ { s - 1 } s ! ( n - 1 - s ) ! .
$$

Theorem 23 says that the left-hand side is the same. To conclude the proof, apply the coning part of Theorem 20 to Eq. (4) with $q = e _ { v }$ □

Remark. Before coning, the right-hand side in Theorem 22 is not a single simplex but a signed combination of dilates of the facet $\Delta _ { - v } .$ . Its simplex content cancels, as it must because a simplex is not a signed sum of parallelepipeds. Indeed, the total coeficient of $h _ { \Delta _ { - v } }$ before coning is $\begin{array} { r } { ( n - 1 ) ! \sum _ { r = 1 } ^ { n - 1 } ( - 1 ) ^ { r - 1 } r \binom { n - 1 } { r } = 0 } \end{array}$ for $n \geq 3 .$ . The identity acquires content only after coning with the apex $e _ { v }$ . The pyramids $\mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } )$ are no longer dilates of one another.

## 8. Subdividing the Simplex

This section uses the valuation property to relate the cones of the simplices $Q _ { v , R }$ to the full simplex $\Delta .$ Recall Figure 6.

Lemma 24. For every nonempty proper subset $\varnothing \neq R \subsetneq [ n ]$

$$
\sum _ { v \notin { R } } h _ { \mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } ) } \equiv ( n - | R | ) h _ { \Delta } .\tag{5}
$$

Proof. Let $C : = [ n ] \setminus R$ and consider the barycenter

$$
q _ { C } : = { \frac { 1 } { | C | } } \sum _ { i \in C } e _ { i } .
$$

For every $v \in C$ , let $T _ { v } = \mathrm { p y r } _ { q _ { C } } ( \Delta _ { - v } )$ . We claim that the simplices $T _ { v }$ for $v \in C$ form a subdivision of $\Delta$ up to lower dimensional pieces; see Figure 11. Indeed, consider a generic

![](images/5fca0f536dcb36b34ae1647ae76883db03169479b5c5d58c4ec40650facd01f8.jpg)  
Figure 11. Two subdivisions of the tetrahedron $\Delta$ into the simplices $T _ { v } = \mathrm { p y r } _ { q _ { C } } ( \Delta _ { - v } )$ for $v \in C .$

$\begin{array} { r } { x = \sum _ { i = 1 } ^ { n } \lambda _ { i } e _ { i } \in \Delta } \end{array}$ , where $\lambda _ { i } \geq 0$ and $\textstyle \sum _ { i } \lambda _ { i } = 1$ . The key observation is

$$
x \in T _ { v } \quad \iff \quad \lambda _ { v } = \operatorname* { m i n } _ { i \in C } \lambda _ { i } .
$$

Indeed, if $x = t q _ { C } + ( 1 - t ) y$ with $y \in \Delta _ { - v } ,$ , then $\lambda _ { v } = t / | C |$ and $\lambda _ { i } \geq t / | C |$ for every $i \in C$ Conversely, if $\lambda _ { v }$ is minimal, take $t = | C | \lambda _ { v }$ and $y = ( x - t q _ { C } ) / ( 1 - t ) \in \Delta _ { - v }$ when $t < 1$ and if $t = 1$ , then $x = q _ { C } \in T _ { v }$ . Thus, by Theorem 20,

$$
\sum _ { v \in C } h _ { T _ { v } } \equiv h _ { \Delta } .
$$

By the definitions,

$$
\begin{array} { r l } { \mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } ) = \mathrm { c o n v } \left( e _ { v } , | C | \Delta _ { - v } - \displaystyle \sum _ { i \in C \backslash \{ v \} } e _ { i } \right) } & { } \\ { = | C | T _ { v } - \displaystyle \sum _ { i \in C \backslash \{ v \} } e _ { i } , } & { } \end{array}
$$

because $\begin{array} { r } { e _ { v } = | C | q _ { C } - \sum _ { i \in C \backslash \{ v \} } e _ { i } } \end{array}$ . Translation changes a support function by a linear function, which is $\equiv 0$ . Consequently,

$$
\begin{array} { r } { h _ { \mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } ) } \equiv | C | h _ { T _ { v } } . } \end{array}
$$

Summing over $v \in C$ and applying the above gives

$$
\sum _ { v \notin { R } } h _ { \mathrm { p y r } _ { e _ { v } } ( Q _ { v , R } ) } \equiv | C | h _ { \Delta } = ( n - | R | ) h _ { \Delta } .
$$

□

## 9. Putting it Together

The following theorem expresses the simplex as a signed sum of paramids modulo lower dimensional pieces.

Theorem 25.

$$
h _ { \Delta } \equiv { \frac { ( - 1 ) ^ { n } } { n ! } } \sum _ { v = 1 } ^ { n } \sum _ { \pi } \sum _ { M \subseteq \{ 2 , \ldots , n - 1 \} } ( - 1 ) ^ { | M | } h _ { \mathrm { p y r } _ { e _ { v } } ( B _ { v , \pi , M } ) } .
$$

Proof. Summing the identity from Theorem 22 over all $v \in [ n ]$ gives

$$
\sum _ { v } \sum _ { \pi } \sum _ { M } ( - 1 ) ^ { | M | } h _ { \mathrm { p y r } _ { e v } ( B _ { v , \pi , M } ) } \equiv ( n - 1 ) ! \sum _ { \emptyset \neq R \subseteq [ n ] } \frac { ( - 1 ) ^ { | R | - 1 } | R | } { n - | R | } \sum _ { v \notin R } h _ { \mathrm { p y r } _ { e v } ( Q _ { v , R } ) } .
$$

By Theorem 24, this equals $\left( \mathrm { w i t h } \equiv \right)$

$$
( n - 1 ) ! \sum _ { \emptyset \neq R \subseteq [ n ] } ( - 1 ) ^ { | R | - 1 } | R | h _ { \Delta } = ( n - 1 ) ! \left( \sum _ { r = 1 } ^ { n - 1 } ( - 1 ) ^ { r - 1 } r { \binom { n } { r } } \right) h _ { \Delta } = ( - 1 ) ^ { n } n ! h _ { \Delta } .
$$

## References

[1] Brandon Amos, Lei Xu, and J. Zico Kolter. “Input Convex Neural Networks”. In: Proceedings of the 34th International Conference on Machine Learning. Vol. 70. Proceedings of Machine Learning Research. PMLR, 2017, pp. 146–155. url: https: //proceedings.mlr.press/v70/amos17b.html.

[2] Raman Arora, Amitabh Basu, Poorya Mianjy, and Anirbit Mukherjee. “Understanding Deep Neural Networks with Rectified Linear Units”. In: International Conference on Learning Representations. 2018. arXiv: 1611.01491. url: https: //openreview.net/forum?id=B1J\_rgWRW.

[3] Gennadiy Averkov, Christopher Hojny, and Maximilian Merkert. “On the Expressiveness of Rational ReLU Neural Networks with Bounded Depth”. In: International Conference on Learning Representations. 2025. url: https://openreview.net/ forum?id=uREg3OHjLL.

[4] Egor Bakaev, Florestan Brunck, Christoph Hertrich, Jack Stade, and Amir Yehudayof. “Better Neural Network Expressivity: Subdividing the Simplex”. In: Proceedings of the 58th Annual ACM Symposium on Theory of Computing. Preprint first posted in 2025. Salt Lake City, UT, USA: ACM, 2026, pp. 500–507. doi: 10.1145/3798129.3800768. arXiv: 2505.14338. url: https://arxiv.org/abs/2505.14338.

[5] Yoshua Bengio. “Learning Deep Architectures for AI”. In: Foundations and Trends in Machine Learning 2.1 (2009), pp. 1–127. doi: 10.1561/2200000006.

[6] Charlotte Bunne, Stefan G. Stark, Gabriele Gut, Jacobo Sarabia del Castillo, Mitch Levesque, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar Rätsch. “Learning Single-Cell Perturbation Responses Using Neural Optimal Transport”. In: Nature Methods 20.11 (2023), pp. 1759–1768. doi: 10.1038/s41592-023- 01969-x. url: https://www.nature.com/articles/s41592-023-01969-x.

[7] Felix Bünning, Adrian Schalbetter, Ahmed Aboudonia, Mathias Hudoba de Badyn, Philipp Heer, and John Lygeros. “Input Convex Neural Networks for Building MPC”. In: Proceedings of the 3rd Conference on Learning for Dynamics and Control. Vol. 144. Proceedings of Machine Learning Research. PMLR, 2021, pp. 251–262. url: https://proceedings.mlr.press/v144/bunning21a.html.

[8] Yize Chen, Yuanyuan Shi, and Baosen Zhang. “Optimal Control Via Neural Networks: A Convex Approach”. In: International Conference on Learning Representations. Preprint first posted in 2018; conference publication in 2019. 2019. arXiv: 1805.11835. url: https://openreview.net/forum?id=H1MW72AcK7.

[9] Yize Chen, Yuanyuan Shi, and Baosen Zhang. “Data-Driven Optimal Voltage Regulation Using Input Convex Neural Networks”. In: Electric Power Systems Research 189 (2020), p. 106741. doi: 10.1016/j.epsr.2020.106741. url: https: //www.sciencedirect.com/science/article/pii/S0378779620305447.

[10] George Cybenko. “Approximation by Superpositions of a Sigmoidal Function”. In: Mathematics of Control, Signals and Systems 2.4 (1989), pp. 303–314. doi: 10.1007/ BF02551274. url: https://link.springer.com/article/10.1007/BF02551274.

[11] Hongkai Dai, Benoit Landry, Lujie Yang, Marco Pavone, and Russ Tedrake. “Lyapunov-Stable Neural-Network Control”. In: Proceedings of Robotics: Science and Systems. 2021. doi: 10.15607/RSS.2021.XVII.063. url: https://roboticsproceedings. org/rss17/p063.html.

[12] Ronen Eldan and Ohad Shamir. “The Power of Depth for Feedforward Neural Networks”. In: 29th Annual Conference on Learning Theory. Vol. 49. Proceedings of Machine Learning Research. PMLR, 2016, pp. 907–940. url: https://proceedings. mlr.press/v49/eldan16.html.

[13] Xavier Glorot, Antoine Bordes, and Yoshua Bengio. “Deep Sparse Rectifier Neural Networks”. In: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. Vol. 15. Proceedings of Machine Learning Research. PMLR, 2011, pp. 315–323. url: https://proceedings.mlr.press/v15/glorot11a.html.

[14] Moritz Grillo, Christoph Hertrich, and Georg Loho. “Depth-Bounds for Neural Networks via the Braid Arrangement”. In: Advances in Neural Information Processing Systems. Vol. 38. Curran Associates, Inc., 2025. arXiv: 2502.09324. url: https: //openreview.net/forum?id=XO9fhSZkBh.

[15] Christian Alexander Haase, Christoph Hertrich, and Georg Loho. “Lower Bounds on the Depth of Integral ReLU Neural Networks via Lattice Polytopes”. In: International Conference on Learning Representations. 2023. arXiv: 2302.12553. url: https: //openreview.net/forum?id=2mvALOAWaxY.

[16] Juncai He, Lin Li, Jinchao Xu, and Chunyue Zheng. “ReLU deep neural networks and linear finite elements”. In: Journal of Computational Mathematics 38.3 (2020), pp. 502–527.

[17] Christoph Hertrich, Amitabh Basu, Marco Di Summa, and Martin Skutella. “Towards Lower Bounds on the Depth of ReLU Neural Networks”. In: SIAM Journal on Discrete Mathematics 37.2 (2023). doi: 10.1137/22M1489332.

[18] Christoph Hertrich and Georg Loho. Neural Networks and (Virtual) Extended Formulations. First posted in 2024; revised in 2025. 2024. doi: 10.48550/arXiv.2411.03006. arXiv: 2411.03006. url: https://arxiv.org/abs/2411.03006.

[19] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. “Multilayer Feedforward Networks Are Universal Approximators”. In: Neural Networks 2.5 (1989), pp. 359–366. doi: 10.1016/0893-6080(89)90020-8. url: https://www.sciencedirect.com/ science/article/pii/0893608089900208.

[20] Joey Huchette, Gonzalo Muñoz, Thiago Serra, and Calvin Tsay. “When Deep Learning Meets Polyhedral Theory: A Survey”. In: INFORMS Journal on Computing (2026). Published online 12 March 2026; preprint first posted in 2023. doi: 10.1287/ijoc. 2024.0902. arXiv: 2305.00241. url: https://pubsonline.informs.org/doi/10. 1287/ijoc.2024.0902.

[21] Børge Jessen and Anders Thorup. “THE ALGEBRA OF POLYTOPES IN AFFINE SPACES”. In: Mathematica Scandinavica 43.2 (1978), pp. 211–240. issn: 00255521, 19031807. url: http://www.jstor.org/stable/24491087 (visited on 10/11/2026).

[22] Christoph Koutschan, Bernhard Moser, Anton Ponomarchuk, and Josef Schicho. “Representing piecewise linear functions by functions with small arity”. In: Applicable Algebra in Engineering, Communication and Computing 36.4 (July 1, 2025), pp. 595– 610. issn: 1432-0622. doi: 10.1007/s00200-023-00627-1. url: https://doi.org/ 10.1007/s00200-023-00627-1.

[23] Christoph Koutschan, Anton Ponomarchuk, and Josef Schicho. “Representing Piecewise-Linear Functions by Functions With Minimal Arity”. In: IEEE Transactions on Information Theory 72.9 (2026), pp. 6825–6831. doi: 10.1109/TIT.2026.3705562.

[24] Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus, and Shimon Schocken. “Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function”. In: Neural Networks 6.6 (1993), pp. 861–867. doi: 10.1016/S0893- 6080(05)80131-5.

[25] Peter McMullen and Rolf Schneider. “Valuations on convex bodies”. In: Convexity and its applications. Springer, 1983, pp. 170–247.

[26] Hrushikesh Mhaskar, Qianli Liao, and Tomaso Poggio. “When and Why Are Deep Networks Better Than Shallow Ones?” In: Proceedings of the AAAI Conference on Artificial Intelligence 31.1 (2017). doi: 10.1609/aaai.v31i1.10913. url: https: //ojs.aaai.org/index.php/AAAI/article/view/10913.

[27] Anirbit Mukherjee and Amitabh Basu. Lower Bounds over Boolean Inputs for Deep Neural Networks with ReLU Gates. Tech. rep. TR17-190. Rehovot, Israel: Electronic Colloquium on Computational Complexity, 2017. doi: 10.48550/arXiv.1711.03073. arXiv: 1711.03073. url: https://arxiv.org/abs/1711.03073.

[28] Kilian Rueß, Gennadiy Averkov, Florestan Brunck, Moritz Grillo, Christoph Hertrich, Georg Loho, Jack Stade, Moritz Stargalla, Matthew Sun, and Martin Winter. “Shal lower ReLU network representations via exact linear algebra”. In: arXiv preprint arXiv:2607.21651 (2026). Version 2, 1 September 2026. doi: 10.48550/arXiv.2607. 21651. arXiv: 2607.21651. url: https://arxiv.org/abs/2607.21651v2.

[29] Itay Safran and Jason D. Lee. “Optimization-Based Separations for Neural Networks”. In: Proceedings of Thirty Fifth Conference on Learning Theory. Vol. 178. Proceedings of Machine Learning Research. PMLR, 2022, pp. 3–64. url: https://proceedings. mlr.press/v178/safran22a.html.

[30] Rolf Schneider. “Valuations on convex bodies”. In: Convex Bodies: The Brunn– Minkowski Theory. Encyclopedia of Mathematics and its Applications. Cambridge University Press, 2013, pp. 329–368.

[31] G. C. Shephard. “Combinatorial Properties of Associated Zonotopes”. In: Canadian Journal of Mathematics 26.2 (1974), pp. 302–321. doi: 10.4153/CJM-1974-032-5.

[32] Matus Telgarsky. “Benefits of Depth in Neural Networks”. In: 29th Annual Conference on Learning Theory. Vol. 49. Proceedings of Machine Learning Research. PMLR, 2016, pp. 1517–1539. url: https://proceedings.mlr.press/v49/telgarsky16.html.

[33] Juan L. Valerdi. On Minimal Depth in Neural Networks. arXiv preprint arXiv:2402.15315. First posted in 2024; version 6 revised 22 September 2026. 2026. doi: 10.48550/arXiv. 2402.15315. arXiv: 2402.15315. url: https://arxiv.org/abs/2402.15315v6.

[34] Shuning Wang and Xusheng Sun. “Generalization of Hinging Hyperplanes”. In: IEEE Transactions on Information Theory 51.12 (2005), pp. 4425–4431. doi: 10.1109/TIT. 2005.859246.

[35] Charles A. Weibel. An Introduction to Homological Algebra. Cambridge Studies in Advanced Mathematics. Cambridge University Press, 1994.

Department of Computer Science<sub>,</sub> University of Copenhagen<sub>,</sub> Denmark

Email address: flbr@di.ku.dk

University of Technology Nuremberg<sub>,</sub> Germany

Email address: christoph.hertrich@utn.de

University of Technology Nuremberg<sub>,</sub> Germany

Email address: kilian.ruess@utn.de

Department of Computer Science<sub>,</sub> University of Copenhagen<sub>,</sub> Denmark

Email address: jast@di.ku.dk

Department of Computer Science<sub>,</sub> University of Copenhagen<sub>,</sub> Denmark and Department of Mathematics<sub>,</sub> Technion–IIT

Email address: amye@di.ku.dk