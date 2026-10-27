# Query Complexity of Learning Axis-Aligned Partitions<sup>∗</sup>

Hadley Black<sup>†</sup> Arya Mazumdar<sup>‡</sup> Barna Saha<sup>§</sup> Geelon So<sup>¶</sup>

October 27, 2026

## Abstract

This work establishes the query complexity of learning high-dimensional boxes in the exact query learning setting. Let $X \subset \mathbb { R } ^ { d }$ be a set of n points, labeled by a collection of k boxes: the label of each point in X is determined by the unique box containing it. A learner aims to exactly recover the labels for every point in X given access to a label oracle. We consider three increasingly-structured settings. In the most general setting, there are no conditions on the boxes, and they may overlap. We show that CAL (Cohn-Atlas-Ladner 1994) a classical active learning algorithm, is minimax-optimal with query complexity Θ(kd log(n/kd)). When the boxes are disjoint, we examine two regimes. In the large-k regime, CAL remains minimax-optimal. For the constant-k regime, an optimized variant of the Halving algorithm (Hegedüs 1995) can beat CAL by a $\log ( d )$ factor. Finally, in the most-structured settings, the boxes correspond to the leaves of a constant-sized decision tree. We prove a tight query complexity of $\Theta ( d + \log n )$ that is additive in d and log n, generalizing a recent result (Black-Larsen-Mazumdar-Saha-So 2026) that establishes the query complexity of learning decision stumps.

## Contents

1 Introduction 3   
1.1 Main Results . . 3   
1.2 Technical Overview 5   
1.3 Open Questions 7   
1.4 Related Work 7   
2 General Box Partitions 7   
2.1 Learning General Box Partitions . 8   
2.2 Matching Lower Bound 9   
3 Disjoint Box Partitions 11   
3.1 Matching Lower Bound in the Large-k Regime 11   
3.2 Beating the General Box Lower Bound in the Small-k Regime . 14   
4 Decision Trees 16   
4.1 Decision Tree Learner . . 16   
4.1.1 Correctness Proof . . 18   
4.1.2 Query Complexity Proof . 18   
4.2 Lower Bound 19   
A Query Complexity with Round Constraints 21   
B Separation between CAL and Halving++ 23

## 1 Introduction

This work establishes the query complexity of learning high-dimensional boxes under the setting of pool-based active learning with label queries [Ang87, Das04, Set09]. Let $X \subset \mathbb { R } ^ { d }$ be any finite set of n points and $Y : = [ k ]$ be a set of k labels. The goal of a learner is to recover an unknown labeling $f : X \to Y$ when provided access to a label oracle: upon being given a query $x \in X$ , the oracle supplies the label $f ( x )$ . The basic problem is to understand how many queries the learner needs before being able to deduce the label of every point in X.

We study the family of axis-aligned partitions, or more plainly, box partitions. A labeling f is a box partition if it is induced by k boxes: for each label $y \in Y .$ , there is a box $\mathcal { \bar { B } } _ { y } \subset \mathbb { R } ^ { d }$ such that the points in the yth class are given by $f ^ { - 1 } \overset { \vartriangle } { \boldsymbol { ( y ) } } = X \cap B _ { y }$ . There are $a _ { y , i } \leq b _ { y , i } \in \mathbb { R } \cup \{ \pm \infty \}$ so that

$$
\forall x \in X , \quad f ( x ) = y \quad \Longleftrightarrow \quad x \in B _ { y } , \qquad \mathrm { w h e r e } B _ { y } = \prod _ { i = 1 } ^ { d } [ a _ { y , i } , b _ { y , i } ] .
$$

A simple example of box partitions is the family of axis-aligned halfspaces (decision stumps). These are box partitions that split X in two, based on which side of an axis-aligned halfspace points fall; the splitting halfspace can also be seen as the face along which two adjacent, unbounded boxes meet. Another example is the family of box partitions induced by the leaves of decision trees, which generalize decision stumps via recursion. Decision trees assign two instances the same label if and only if they are contained in the same leaf of the tree. Beyond decision stumps and decision trees, this work considers a hierarchy of box partitions (see Figure 1). With increasing structure, they are:

• General box partitions: These constitute the broadest class of axis-aligned partitions. The only requirement is that the induced labeling is well-defined, meaning that each point in X is contained in precisely one of the underlying boxes. The boxes themselves may overlap in $\mathbf { \hat { \mathbb { R } } } ^ { d }$

• Disjoint box partitions: The boxes must be disjoint, $B _ { y } \cap B _ { y ^ { \prime } } = \emptyset$ for distinct y, $y ^ { \prime } \in Y$

• Decision trees: The boxes are disjoint and admit a decision-tree structure. This means that there is a binary tree where each leaf is identified with a label in Y . For each internal node in the binary tree, there is a dimension $i \in [ d ]$ and a threshold $\tau \in$ R that splits its left and right descendants: a descendant $y \in Y$ is to the left if and only if $f ^ { - 1 } ( y ) \subset \{ x \in \mathbb { R } ^ { d } : x _ { i } < \tau \}$

## 1.1 Main Results

We study the query complexity of learning axis-aligned partitions through a series of classic active learning algorithms. Besides giving specialized upper bounds, the main contribution of this work establishes lower bounds across these settings. This illuminates the way in which diferent sources of geometric structure afect the hardness of learning, and how diferent algorithms can leverage it.

The first result is Theorem 1.1 for learning general class of box partitions. It establishes a tight query complexity of $\Theta ( k d \log ( n / k d ) )$ , where $n : = | X |$ is the number of points, $k : = | Y |$ is the number of labels, and d is the number of dimensions. Formally:

Theorem 1.1 (General Boxes). Let $n \geq k \geq 2 .$ . Consider the problem of learning a general k-box partition over a worst-case set of n points in d-dimensions.

• The expected query complexity of this problem is $\Theta \left( k d \log ( 1 + { \frac { n } { k d } } ) \right)$

• For any $1 \leq r \leq \log _ { 2 } ( n / k d )$ , the r-round query complexity is $\begin{array} { r l r } { \Theta \left( r k d ( 1 + \frac { n } { k d } ) ^ { 1 / r } \right) } \end{array}$

In particular, there is an algorithm achieving the optimal query complexity with a $1 + \log _ { 2 } ( 1 + n / d k )$ round budget.   
Our lower bounds hold even against randomized algorithms.

![](images/1d6e3d98dc9406c84abf0109e4cb4cae7503b4405af910264f8eca6dbf514d45.jpg)  
(a)

![](images/625d82e2524797c2007d6376f55ffce010d6cc04bda002c546c5ba2f539f8192.jpg)  
(b)

![](images/b6af6c9bcabb65915e6ca89137564f8ebf3db9cc4c48d37b368242d3509932d9.jpg)  
(c)  
Figure 1: Axis-aligned partitions induced by (a) general boxes, (b) disjoint boxes, and (c) the leaves of a decision tree. These classes form a strictly descending chain: (a) is a box partition with two labels that cannot be induced by any disjoint box partition, and (b) is a disjoint box partition with four labels that cannot be induced by any decision tree.

A canonical active learning algorithm achieves minimax-optimality: it is enough to run CAL, which is named after the authors who introduced it in [CAL94]. This algorithm sequentially chooses a random point whose label the learner cannot determine yet, and asks the oracle for the ground-truth label. Furthermore, batched-CAL is minimax-optimal even in settings where the learner is limited to r rounds of interactions with the oracle. We prove this theorem in Section 2.

The upper bounds achieved by CAL and its round-aware variant certainly apply to any class of box partitions, including disjoint box partitions and decision trees. The same is not necessarily true for the lower bound. If there is additional structure, can it be leveraged to speed up learning?

The next result Theorem 1.2 focuses on disjoint box partitions in two main regimes. In the large-k regime with $k \geq 2 ^ { d }$ boxes, CAL remains minimax-optimal: the query complexity is lower bounded by $\Omega ( \bar { k } d \log ( n / k d ) )$ , which matches the upper bound for general box partitions. On the other hand, in the small-k regime where the number of boxes $k = O ( 1 )$ is constant, the upper bound for disjoint boxes can be a log(d)-factor smaller than the lower bound for general boxes:

Theorem 1.2 (Disjoint Boxes). Let $d , k \geq 2 .$ . There is a randomized algorithm that learns any disjoint k-box partition over a worst-case set of n points in d-dimensions using in expectation

$$
O \left( \operatorname* { m i n } \left\{ n , k d \log \left( 1 + { \frac { n } { k d } } \right) , { \frac { k ^ { 3 } d \log n } { \log ( k d ) } } \right\} \right) q u e r i e s .
$$

When $k \geq 2 ^ { \Omega ( d ) }$ , the upper bound is matched with a lower bound of $\Omega \left( k d \log { \frac { n } { k d } } \right)$ queries.

We prove this theorem in Section 3. In the small-k regime, the algorithm that achieves the faster rate is another classic active learning algorithm called the Halving algorithm, or rather, a slightly more optimized version of it found in [Mos82] and $[ \mathrm { H e g } 9 5 ]$ . Let’s call it Halving++. While CAL is essentially content to query a random point with an ambiguous label, Halving++ seeks particularly informative points to query. One combinatorial quantity that controls the query complexity of Halving++ is the extended teaching dimension [Heg95]. We prove a query complexity upper bound by way of bounding the extended teaching dimension of disjoint boxes. The proofs are in Section 3. Compared to general boxes, the faster rate for disjoint boxes cannot only be attributed to the presence of additional structure; the learner also needs to take advantage of it. Section B gives an instance of a disjoint-box problem where CAL provably fails to exploit the structure that Halving++ successfully leverages.

The final result is Theorem 1.3, which considers box partitions induced by the leaves of a decision tree. Importantly, these decision trees are distinct from the more common binary-valued decision trees; the induced labeling here is a multi-class function, where the number of classes k is equal to the number of non-empty leaves.<sup>1</sup> For constant-sized trees, the query complexity has additive upper and lower bounds of $\Theta _ { k } { \left( d + \log n \right) }$ Thus, there is a further separation between this setting and the general box setting, which has a multiplicative lower bound of $\Omega _ { k } ( d \log { n } )$ . We show:

<table><tr><td>Partition class</td><td>Regime Bound</td><td></td><td>Result</td></tr><tr><td>Boxes</td><td>all</td><td> $\Theta \left( k d \log { \frac { n } { k d } } \right)$ </td><td>Theorem 1.1</td></tr><tr><td rowspan="2">Disjoint boxes</td><td> $k \geq 2 ^ { d }$ </td><td> $\Theta \left( k d \log { \frac { n } { k d } } \right)$ </td><td rowspan="2">Theorem 1.2</td></tr><tr><td> $k = O ( 1 )$ </td><td> $\begin{array} { r } { O \left( { \frac { d \log n } { \log d } } \right) } \end{array}$ </td></tr><tr><td>Decision-tree leaves</td><td> $k = O ( 1 )$ </td><td> $\Theta \left( d + \log n \right)$ </td><td>Theorem 1.3</td></tr><tr><td>Decision stumps</td><td> $k = 2$ </td><td> $\Theta \left( d + \log n \right)$ </td><td> $\left[ \mathrm { B L M } ^ { + } 2 6 \right]$ </td></tr></table>

Table 1: Summary of the query complexity of learning box partitions.

Theorem 1.3 (Decision Trees). Let k, $d \geq 2$ and $n \geq d ( k - 1 )$ . There is a deterministic algorithm that learns any box-partition induced by the leaves of a k-leaf decision tree using at most

$$
2 ^ { O ( k ) } \left( d + \log n \right) { q u e r i e s } .
$$

If the decision tree is a decision list, then the query complexity is upper bounded by $O ( k ^ { 2 } ( d + \log n ) )$ , and it is lower bounded by $\Omega ( k ( d + \log ( n / k ) ) )$ ) when $n \geq d .$

To achieve this additive upper bound on query complexity, we make use of a ‘parallel binary search’ algorithm, which was introduced by [BLM<sup>+</sup>26]. The parallel binary search algorithm is a minimax-optimal learner for decision stumps. Theorem 1.3 can be viewed as a generalization of that result, establishing the query complexity not just for decision stumps, but for constant-sized decision trees. The lower bound we provide here holds even for decision lists, which are decision trees whose internal nodes form a path graph. The proofs of these results are provided in Section 4.

A high-level summary of our main results is given in Table 1. The next section also provides some high-level sketch of the technical ideas in this work.

## 1.2 Technical Overview

General box partitions. Theorem 1.1 provides a query complexity upper bound for CAL and a roundconstrained variant. It also gives a matching lower bound construction. The upper bounds rely on a simple observation that applies to any box partition. Suppose that a learner has seen a set of points $S = \{ x _ { 1 } , \dots , x _ { m } \}$ with the same label y. Then, every point in X contained in the bounding box of S must also be labeled $y ,$ and CAL will not query these points. The upper bounds follow from determining the rate that these labeled bounding boxes grow before they fully cover X.

The lower bounds throughout this work are built on the fact that learning a monotone function $f : [ n ] $ {0, 1} from label queries requires Θ(log n) queries. The constructions find ways to embed many independent copies of these one-dimensional threshold learning problems into the various box learning settings. For general box partitions, we embed $\Theta ( k d )$ threshold problems, each consisting of $\Theta ( n / k d )$ points. Thus, we obtain an $\Omega ( \dot { k } d \log ( n / k d ) )$ ) lower bound.

These lower bounds are rather non-trivial to find. Consider even $k = 2$ boxes. It is clear how one threshold learning problem over n points can be embedded into a 2-box learning problem: if we arrange the points in a line, then each threshold can be induced by a pair of adjacent boxes. But this construction only gives an $\Omega ( \log n )$ lower bound, not the stronger $\Omega ( d \log ( \bar { n } / d ) )$ ). The problem is that only a single face of each of the boxes contributes to hardness. Intuitively, naive construction under-utilizes the set of 2d faces available from each box. The question is how to place the points so that all $\Theta ( d )$ faces of the two boxes can interact with each other to admit $\Theta ( d )$ independent threshold learning problems. The construction we give achieves this using overlapping boxes. It turns out that to extract this hardness for general $k ,$ it is necessary that boxes are able to overlap.

Disjoint box partitions. Theorem 1.2 consists of two parts. It shows a lower bound for disjoint boxes in the large-k regime that continues to match the general-box upper bound. It also shows that in the small-k regime with disjoint boxes, it is possible to beat the general-box lower bound.

The lower bound in the large-k regime is a construction that embeds $\Theta ( k d )$ independent threshold learning problems into a disjoint box learning problem. When $k = 2 ^ { d } ,$ , we will embed a threshold learning problem onto every edge of a Boolean hypercube (note there are $d 2 ^ { d - 1 } = \Theta ( k d )$ edges). To do this, place points along each edge of the hypercube, and extend a box out of each vertex onto its incident edges; see Figure 3. The problem is that the threshold learning problems induced this way are not automatically independent, as the boxes may intersect near the center of the Boolean hypercube. However, we carefully design point configurations that manage to preserve independence, thus obtaining an $\Omega ( k d \log ( n / k d ) )$ lower bound.

The second part shows that Halving++ beats the general-box lower bound by a log(d)-factor in the $k = \Theta ( 1 )$ constant regime. This follows from an upper bound on the query complexity of Halving++ [Mos82, Heg95]. To specialize the bound, we bound the extended teaching dimension of disjoint-box partitions, as well as the number of such partitions of X. Beating the general-box lower bound in certain regimes is possible because disjoint boxes have additional structure: every pair of box can be separated by an axis-aligned halfspace. The labeling induced by disjoint boxes can be specified either through the set of 2kd thresholds associated to the faces of the boxes, or through a set of $\Theta ( k ^ { \hat { 2 } } )$ halfspaces that separate every pair of label class. In a sense, CAL learns only by recovering the 2kd thresholds. Halving++ can be strictly more label-eficient, and it does so intuitively by leveraging this alternative specification, which is independent of d. For constant order k, we show that this is ultimately the source of the $\log ( d )$ -factor savings.

Decision trees. In Theorem 1.3 we consider the class of disjoint box partitions that are induced by k-leaf decision trees. While the upper bounds from Theorem 1.1 and Theorem 1.2 extend to this setting, we show that the additional structure of decision trees enables improved query bounds, at least when k is relatively small. $\left[ \mathrm { B L M } ^ { + } 2 6 \right]$ gave a $O ( d$ + log n) query learner for the class of 2-leaf decision trees (or decision stumps). Intuitively, if we compare 2-box partitions, the gap between the $\Theta ( d \log ( n / d ) )$ sample complexity for general boxes and $\Theta ( d + \log n )$ for decision stumps comes from the fact that a 2-leaf tree is specified by a single axis-aligned hyperplane. Our lower bound for general boxes shows that one essentially needs to learn all the $\Omega ( d )$ faces of the two boxes.

More generally, for $k \geq 2 .$ , when a labeling $f \colon X \to Y$ is induced by a decision tree, we are guaranteed the existence of a single axis-aligned hyperplane that separates the labels into two non-empty sets (see Definition 4.1), $A \subset Y$ and $Y \setminus A$ , and this same structure repeats on each side. If this subset A is known in advance, then one can apply the decision stump learner to recover $g _ { A } ( x ) = \mathbf { 1 } ( f ( x ) \in A )$ . Extending this observation, if the label-splits are provided at each level of the tree, this would yield a $O ( k ( d + \log n ) )$ ) query algorithm. This shows that naively guessing every such tree yields a $k ^ { O ( k ) } ( d + \log n )$ query algorithm (see Remark 4.4). In Theorem 1.3, we reduce the k-dependence to $2 ^ { \mathbf { \check { O } } ( k ) }$ with a more careful recursive strategy that avoids brute-force enumeration. At a very high level, we obtain this improvement by certifying correct label splits at the leaves and propagating this information through recursion. This allows us to make permanent progress without guessing the correct tree all at once.

## 1.3 Open Questions

The aim of our work is to obtain a comprehensive understanding of the query complexity of learning natural axis-aligned partitions over worst-case point-sets. The following gaps remain, which we leave as interesting directions for follow up work, and which we believe will require new insights.

• Improved boundsfor disjoint boxes in the intermediate-k regime. When $k \geq 2 ^ { \Omega ( d ) }$ , we obtain a $\Omega ( k d \log ( n / k d ) )$ lower bound, matching our upper bound for general k-box partitions. However, when $k < 2 ^ { o ( d ) }$ , our bounds are not tight, and our upper bounds in Theorem 1.2 demonstrate a separation from general boxes when k is suficiently small. Therefore, as k becomes small, there is a moment where box disjointness supplies more geometric structure that can be exploited by a learner. Obtaining sharper bounds in this regime would require finding new ways to use this geometry, and we believe this is a worthwhile future direction.

• Improved bounds for decision trees. Decision trees are an extremely natural specialized class of box partitions to consider. While we obtain bounds that are tight for any constant number of classes, it is not clear what the true dependence on k is for this problem. We believe this is a very interesting open question to resolve.

## 1.4 Related Work

There is an extensive line of research in exact query learning and active learning; as an inexhaustive list, see [Ang87, Das04, Set09, HY15], and references therein. This work belongs to a subarea concerned with learning geometric concept classes, in which the label function is induced by fundamental objects like balls, boxes, halfspaces, convex sets, and decision trees [BDBK97, BGGM98, BGM<sup>+</sup>98, GSSS13, AKBD16, BCLP24, BL $\mathrm { . M ^ { + } } 2 6 , \mathrm { S \bar { M } L G ^ { + } } 2 6 ]$

This work establishes the query complexity of active learning axis-aligned partitions under various structural assumptions, and our main contributions are our lower bound constructions. Our results complement existing work. A prior result for learning general k-box partitions is given by [BCLP24]. They provide a generic algorithm for multi-class active learning, which uses O(kd log k log n) queries when specialized to boxes. In a more specific two-class setting, [BLM<sup>+</sup>26] establishes that the query complexity for axis-aligned halfspaces is $\Theta ( d \bar { + } \log n )$

This work shows that CAL [CAL94], a standard active learning algorithm, is minimax-optimal for general boxes. We also show that a variant of the Halving algorithm [Heg95] can exploit additional structure, and it strictly beats CAL in certain regimes for learning disjoint boxes. The most structured concept class we consider generalizes decision stumps; they are induced by the leaves of decision trees [MDRF20, HKLM21]. For these tree-based partitions, we also give a minimax-optimal learner for constant number of classes.

The lower bounds we prove arise from embedding many independent copies of a threshold learning problem into various box-learning problems. In the round-constrained setting, lower bounds also arise from corresponding bounds for threshold learning provided by [BPR20].

Notation. For any $a \leq b \in \mathbb { R } \cup \{ \pm \infty \}$ , we let $[ a , b ]$ denote the closed interval from a to b. Given a finite set of points $S \subset \mathbb { R } ^ { d }$ , denote its bounding box by

$$
{ \mathsf { b b o x } } ( S ) = \prod _ { i = 1 } ^ { d } \left[ \operatorname* { m i n } _ { x \in S } x _ { i } , \operatorname* { m a x } _ { x \in S } x _ { i } \right] .
$$

## 2 General Box Partitions

In this section, we prove Theorem 1.1, which considers learning a general box partition: there is a set of k potentially-overlapping boxes $B _ { 1 } , \ldots , B _ { k } \subset \mathbb { R } ^ { d }$ that induce a partition of X

$$
X = ( X \cap B _ { 1 } ) \sqcup \cdots \sqcup ( X \cap B _ { k } ) .
$$

We divide up the results. The upper bound without round constraints is given by Theorem 2.1. The upper bound with round constraints is in Theorem 2.2. The lower bound is given in Theorem 2.4.

## 2.1 Learning General Box Partitions

A standard active learning algorithm CAL [CAL94] achieves the query complexity upper bound. It is an iterative algorithm that maintains a disagreement region. This region is the subset of points of X whose labels cannot be deduced yet: it contains a point x if there are at least two valid labelings that disagree on how to label $x ,$ but that are nevertheless consistent with the observed data. Every round, the learner randomly selects a point from this region and asks the oracle for its label. In the case of learning boxes, if S is a set of points that have been labeled $y ,$ then the learner may deduce that every point in the bounding box bbox(S) is also labeled $y .$ The disagreement region shrinks over the course of the algorithm, and the learner stops when this region vanishes.

Theorem 2.1. There is a randomized algorithm that learns an arbitrary k-box partition in d dimensions over n points using at most 4kd $\textstyle \log ( 1 + { \frac { n } { k d } } )$ label queries in expectation.

Proof. The algorithm is defined as follows.

1. Initialize a partial function h : $X \to [ k ] \cup \{ * \}$ as $h ( x ) = *$ for all $x \in X$ . For each $y \in [ k ]$ , let $X _ { h , y } = \{ x \in$ $X \colon h ( x ) = { \bar { y } } \}$ throughout the execution. We will maintain the invariant

$$
X _ { h , y } \subseteq X _ { y } .
$$

2. Choose a permutation $\pi \colon [ n ] \to X$ uniformly at random.

3. For each $i \leq n ,$ let $x = \pi ( i )$ and do the following:

(a) If $h ( x ) \neq * ,$ then continue.

(b) If $h ( x ) = * ,$ then query f(x). Set $h ( x ) \ = \ f ( x )$ , and also set $h ( z ) ~ = ~ f ( x )$ for all points $z \in$ bb $\mathsf { \Pi } \circ \mathsf { x } \bigl ( X _ { h , f ( x ) } \bigr )$ .

The correctness of the algorithm is clear by inspection. We now analyze the expected number of queries over the randomness of π. Whether or not a point with label $y$ is queried depends only on the ordering that π induces on $X _ { y }$ . Let $\pi _ { y } \colon [ n _ { y } ]  X _ { y }$ denoted this restriction of π where $\bar { n _ { y } } : = | X _ { y } |$ . Observe that for each $t \leq n _ { y } , \pi _ { y } ( t )$ is queried if and only if it is not contained in the bounding box of $\pi _ { y } ( 1 ) , \ldots , \pi _ { y } ( t - 1 )$ . Equivalently, $\pi _ { y } ( t )$ is queried if and only if it is a maximal $p o i n t ^ { 2 }$ among $\pi _ { y } ( 1 ) , \ldots , \pi _ { y } ( t )$ . Since d-dimensional boxes have at most 2d maximal points, and $\pi _ { y } ( t )$ is uniform random among $\pi _ { y } ( 1 ) , \ldots , \pi _ { y } ( t )$ , the probability $\pi _ { y } ( t )$ is queried is at most $2 d / t$ . Let $\begin{array} { r } { Q ( \pi ) = \sum _ { u } Q _ { y } ( \pi ) } \end{array}$ denote the total number of queries made by the algorithm on permutation $\pi ,$ where $Q _ { y } ( \pi )$ is the number of queries in $X _ { y }$ . By linearity of expectation,

$$
\begin{array} { r l r } {  { \mathbb { E } _ { \pi _ { y } } [ Q _ { y } ( \pi ) ] = \sum _ { t = 1 } ^ { n _ { y } } \operatorname* { P r } _ { y } [ \pi _ { y } ( t ) \mathrm { i s ~ q u e r i e d } ] } } \\ & { } & { \leq \displaystyle \sum _ { t = 1 } ^ { n _ { y } } \operatorname* { m i n } ( 1 , \frac { 2 d } { t } ) \leq \sum _ { t = 1 } ^ { n _ { y } } \frac { 4 d } { 2 d + t } \leq 4 d \int _ { 0 } ^ { n _ { y } } \frac { d x } { 2 d + x } = 4 d \ln ( 1 + \frac { n _ { y } } { 2 d } ) . } \end{array}
$$

Applying linearity of expectation over all k labels yields

$$
\mathbb { E } _ { \boldsymbol { \pi } } \left[ Q ( \boldsymbol { \pi } ) \right] \leq 4 d \sum _ { y = 1 } ^ { k } \ln \left( 1 + \frac { n _ { y } } { 2 d } \right) \leq 4 k d \ln \left( 1 + \frac { n } { 2 k d } \right)
$$

by Jensen’s inequality.

![](images/fb75e1fcd52af9459ea139fc625105b4c556e44afcb780b20ed5f3cfb28ace4b.jpg)  
Figure 2: A visualization of the construction in Theorem 2.4. This shows a projection of the 2-box construction onto the $( 2 j - 1 , 2 j )$ )th coordinates. It embeds a threshold problem, where the threshold is marked by a white diamond. Many other points project to the black origin.

Next we consider an r-round variant of CAL. The proof is in Section A.

Theorem 2.2 (r-round algorithm). For any $1 \leq r \leq \log ( n / 2 d k )$ , there is an r-round randomized algorithm which learns an arbitrary k-box partition in d-dimensions over n points using

$$
2 r k d \cdot \left( { \frac { n } { 2 d k } } \right) ^ { 1 / r }
$$

queries in expectation. In particular, when $r = \log ( n / 2 d k )$ , this recovers the optimal query complexity $O ( k d \log ( n / k d ) )$ , while improving on the round complexity in Theorem 2.1 by a Θ(kd) factor.

## 2.2 Matching Lower Bound

The simplest, nontrivial setting for learning box partitions is when $k = 2$ and $d = 1$ . Here, the problem is simply to learn a threshold between two classes. In this case, we can of course run binary search, which optimally solves the problem using at most $\log _ { 2 }$ n queries. The following result provides a lower bound for the threshold learning problem, with and without round constraints.

Theorem 2.3 (Threshold Learning Lower Bound, Theorem 3 of [BPR20]). Consider exactly learning a monotone function $f \colon [ n ]  \{ 0 , 1 \}$ with query access. Suppose that a randomized algorithm A solves this problem on every input f with success probability at least $2 / 3 .$ . Then, there exists f such that the expected number of queries made is at least Ω(log n). Moreover, if A uses $1 \leq r \leq \log _ { 2 }$ n rounds ofadaptivity, then there exists f such that the expected number of queries made is at least $\Omega ( r n ^ { 1 / r } )$

To obtain query lower bounds for general $k , d \geq 2 .$ , we will show that a set of $\Theta ( k d )$ independent threshold learning problems with m points each can be reduced to learning a general k-box partition over a set of points of size $n = \Theta ( k d m )$ . We obtain a lower bound $\Omega ( k d \log ( n / k d ) )$ ) for the round-unconstrained setting, and a lower bound of $\Omega ( r k d ( n / k d ) ^ { 1 / r } )$ for the r-round setting:

Theorem 2.4 (Lower bound for general boxes). Let $k , d \geq 2$ be even. Let $\begin{array} { r } { 2 \leq m \leq \frac { 4 n } { k d } } \end{array}$ be a natural number. Suppose a randomized algorithm learns any k-box partition over n points in $\mathbb { R } ^ { d }$ with success probability $2 / 3$ . Then, there exists a problem instance such that the expected number of queries made is at least $\Omega ( k d \log m )$ . Moreover, if the algorithm uses $1 \leq r \leq \log _ { 2 }$ m rounds of adaptivity, then there exists a problem instance such that the expected number of queries made is at least $\bar { \Omega } ( r k d m ^ { 1 / r } )$ .

Proof. Let $C = { \frac { k d } { A } }$ . Let $f _ { i , j } : [ m ] \to \{ 0 , 1 \}$ be a collection of C independent monotone functions, where $( i , j )$ ranges over $[ k / 2 ] \times [ d / 2 ]$ . We construct a reduction from learning these thresholds to learning a k-box partition over Cm points as follows.

Let’s start with the $k = 2$ case. We drop the index i as it now ranges over the singleton {1}. Define the map $\phi : [ d / 2 ] \times [ m ]  \mathbb { R } ^ { d }$ b

$$
\phi ( j , t ) \equiv \phi _ { j } ( t ) = \Big ( 0 , \underbrace { \imath \dots , 0 } _ { ( 2 j - 1 , 2 j ) \mathrm { t h } } , \underbrace { m - t , t } _ { \mathrm { c o r d i n a t e } } , , 0 , \dots , 0 \Big ) .
$$

We can think of $\phi _ { j } \mathrm { ' s }$ as a collection of $d / 2$ afine embeddings of $[ m ]$ into $\mathbb { R } ^ { d }$ . Define the set

$$
X = \bigcup _ { j \in [ d / 2 ] } \phi _ { j } ( [ m ] ) .
$$

By construction, the map $\phi : [ d / 2 ] \times [ m ]  X$ is bijective. Define the labeling $f : X \to \{ 0 , 1 \}$ by

$$
x = \phi ( j , t ) \quad \implies \quad f ( x ) = f _ { j } ( t ) .
$$

This is visualized in Figure 2. We claim the following:

Claim 2.5. The labeling f is $a 2 \cdot$ box partition. Moreover, thefollowing hold:

(a) The set of C threshold functions $f _ { j } { ' } s$ can be computed from $f .$

(b) Every query to a box-label oracle can be simulated by a query to one threshold-label oracle.

In particular, this claim shows that any box-learning algorithm that learns f with success probability p can be converted into an algorithm that learns the original set of thresholds with the same success probability p using the same number of queries. By Theorem 2.3, we immediately have that the number of queries that the box-learning algorithm makes to a label oracle is at least

$$
C \cdot \Omega \left( \operatorname* { m a x } \left\{ r m ^ { 1 / r } , \log _ { 2 } m \right\} \right) .
$$

We now prove the claim. The subclaims $\mathbf { \eta } ( \mathtt { a } )$ and (b) are direct, and they are conceptually converses of each other. We can compute $f _ { j }$ from $f$ since $f _ { j } ( t )$ is equal to $f ( \phi _ { j } ( t ) )$ . On the other hand, a query $x = \phi ( j , t )$ to the box-label oracle can be simulated by querying $f _ { j } ( t )$

Proof that f is a 2-box partition. Let $0 < \tau _ { j } < m + 1$ be a threshold for $f _ { j } ,$ , so that

$$
f _ { j } ( t ) = \left\{ \begin{array} { l l } { 0 } & { t < \tau _ { j } } \\ { 1 } & { t > \tau _ { j } . } \end{array} \right.
$$

Define the boxes $B _ { 0 }$ and $B _ { 1 }$ as follows. Let $\Pi _ { j } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ be the projection onto the $( 2 j - 1 )$ th and $( 2 j )$ th components for $j \in [ d / 2 ]$ . Then:

$$
\Pi _ { j } ( B _ { 0 } ) = [ 0 , m ] \times [ 0 , \tau _ { j } ] \qquad \mathrm { a n d } \qquad \Pi _ { j } ( B _ { 1 } ) = [ 0 , m - \tau _ { j } ] \times [ 0 , m ] .
$$

These two boxes are defined so that $y = f _ { j } ( t )$ if and only if $\phi ( j , t ) \in B _ { y }$ . Indeed for all $t \in [ m ] .$

$$
f _ { j } ( t ) = 0 \quad \Longleftrightarrow \qquad t < \tau _ { j } \quad \Longleftrightarrow \qquad m - t > m - \tau _ { j } .
$$

Recall that the $( 2 j - 1 , 2 j )$ -coordinates of $\phi ( j , t )$ is $( m - t , t )$ . The second condition is equivalent to $\phi ( j , t ) \in B _ { 0 }$ while the third is equivalent to $\phi ( j , t ) \notin B _ { 1 }$ . Thus, f is a 2-box partition.

Generalization to $k \geq 2 .$ . We now make $k / 2$ translated copies of the 2-box construction with $\begin{array} { r } { m \leq \frac { 4 n } { k d } } \end{array}$ , so that there are a total of k-classes. Define the point set as

$$
X = \bigcup _ { i \in [ k / 2 ] } \bigcup _ { j \in [ d / 2 ] } \phi _ { j } ( [ m ] ) + ( m i ) \mathbf { 1 } ,
$$

where in the ith copy, we translate each point by the vector $( m i , \ldots , m i )$ . As before, each point in X corresponds to a unique set of indices $( i , \bar { j } , t ) \in [ \bar { k } / 2 ] \times [ d / 2 ] \times [ m ]$ . Letting $\phi : ( i , j , t ) \mapsto$ x denote this bijection, we define $\bar { f }$ similarly, where $f ( x ) = 2 i + f _ { i , j } ( t )$ when $x = \phi ( i , j , t )$ . In this way, we obtain an embedding of C independent threshold problems and the stated lower bound. □

## 3 Disjoint Box Partitions

In this section we prove our results for disjoint k-box partitions stated in Theorem 1.2. We start by proving our lower bound in Section 3.1 for the large-k regime, then obtain a separation between disjoint and general boxes in Section 3.2 with an improved algorithm for the small-k regime.

## 3.1 Matching Lower Bound in the Large-k Regime

We prove the following lower bound for learning disjoint k-box partitions. In particular, we obtain a $\Omega ( k d \log ( n / k d ) )$ query lower bound for fully adaptive, randomized algorithms when $k \geq 2 ^ { d }$ , showing that when k is suficiently large, disjointness does not make the learning problem easier.

Theorem 3.1 (Lower bound for disjoint boxes). Let $k \geq 2$ be a power of two and $d \geq 1$ . Define $L = \operatorname* { m i n } ( d , \log _ { 2 } k )$ and let $\begin{array} { r } { 2 \leq m \leq \frac { 2 n } { k L } } \end{array}$ be a natural number. Suppose a randomized algorithm learns any disjoint k-box partition over n points in $\mathbb { R } ^ { d }$ with success probability $2 / 3 .$ . Then, there exists a problem instance such that the expected number of queries made is at least $\Omega ( k L \log m )$ . Moreover, if the algorithm uses $1 \leq r \leq \log _ { 2 }$ m rounds of adaptivity, then there exists a problem instance such that the expected number of queries made is at least $\bar { \Omega } ( r k L m ^ { 1 / r } )$

Proof. We first prove the theorem when $k = 2 ^ { d } .$ , and thus $L = d $ . The more general case will follow easily from this setting. For the sake of simplicity, assume that kd divides 2n. We will construct a set $X \subseteq \mathbb { R } ^ { d }$ of n points and a family of $k = 2 ^ { d }$ boxes that induce $k d / 2$ independent threshold learning problems, each on a set of $m = 2 n / d k$ points. The lower bound will then follow by applying Theorem 2.3 across each independent threshold problem.

Our construction is built on the d-dimensional hypercube, $\{ 0 , 1 \} ^ { d }$ . Each vertex $u \in \{ 0 , 1 \} ^ { d }$ corresponds to a label induced by a box $B _ { u }$ such that $u \in B _ { u } \subseteq [ 0 , \dot { 1 } ] ^ { d }$ . These boxes are disjoint.

Notation. Given $u \in \{ 0 , 1 \} ^ { d }$ and $p \in [ d ] ,$ , we let $u _ { - p } \in \{ 0 , 1 \} ^ { d - 1 }$ denote the context vector obtained by removing the p-th bit of u. For a context vector $z \in \{ 0 , \bar { 1 } \} ^ { d - 1 }$ and coordinate $p \in [ d ]$ , we use the pair $( p , z )$ to identify the hypercube edge with context z in direction p.

The construction. Our objective is to induce an independent threshold learning problem along each edge $( p , z )$ . For each dimension $p \in [ d ]$ and context $z \in \{ 0 , 1 \} ^ { \star } d \mathrm { - } 1$ , we define a disjoint family of intervals as follows:

$$
\mathcal { Z } ( p ) = \Big \{ I ( p , z ) \subset [ 0 , 1 ] \colon z \in \{ 0 , 1 \} ^ { d - 1 } \Big \} \mathrm { ~ s u c h ~ t h a t ~ } \bigcup _ { z \in \{ 0 , 1 \} ^ { d - 1 } } I ( p , z ) = [ 0 , 1 ] .\tag{1}
$$

These intervals are of uniform length, $2 ^ { - ( d - 1 ) }$ . Note that what is currently left undefined is the ordering of the intervals according to their context vectors, z. This turns out to be a critical aspect of the construction, which we will discuss and define later.

![](images/177e3c73731d45b6c2be6de807a4c842156b58ee60788f2656a0cca58f68ab16.jpg)  
Figure 3: A visualization for the construction in Theorem 3.1. The left and right subfigures give two point configurations on the edges of the two-dimensional Boolean hypercube. The left configuration fails to embed a set of independent threshold problems, since thresholds induced by overlapping boxes, like the one shown, are immediately ruled out. The point configuration on the right embeds an independent set of threshold problems per edge using a coherent family of total orderings over context vectors. In particular, the horizontal edges are ordered top-to-bottom and the vertical edges are ordered left-to-right.

On each edge $( p , z ) \in [ d ] \times \{ 0 , 1 \} ^ { d - 1 }$ , we induce a threshold learning problem on the segment

$$
L ( p , z ) = { \Big \{ } x \in [ 0 , 1 ] ^ { d } \colon x _ { - p } = z \operatorname { a n d } x _ { p } \in I ( p , z ) { \Big \} } .
$$

To do this, we let $X ( p , z ) \subset L ( p , z )$ be a set of $\begin{array} { r } { \frac { n } { d 2 ^ { d - 1 } } = \frac { 2 n } { k d } = m } \end{array}$ uniformly spaced interior points along this segment. Concretely, if $I ( p , z ) = [ a , a + h ]$ , then define

$$
\delta = \frac { h } { 2 m + 2 } \mathrm { ~ \ a n d ~ \ } X ( p , z ) = \Big \{ x \in [ 0 , 1 ] ^ { d } \colon x _ { - p } = z , x _ { p } \in \{ a + 2 i \delta \colon 1 \leq i \leq m \} \Big \} .
$$

We define the full point-set as

$$
X = \bigcup _ { p \in [ d ] , z \in \{ 0 , 1 \} ^ { d - 1 } } X ( p , z ) .
$$

A threshold $\tau ( p , z ) \in I ( p , z )$ is then chosen somewhere in this interval. Concretely,

$$
\tau ( p , z ) \in \Big \{ a + ( 2 i + 1 ) \delta \colon 0 \leq i \leq m \Big \}
$$

is one of m + 1 possible values. Then, given a threshold profile

$$
T = \left\{ \tau ( p , z ) \colon p \in [ d ] , z \in \{ 0 , 1 \} ^ { d - 1 } \right\}
$$

the box for label $u \in \{ 0 , 1 \} ^ { d }$ is defined as

$$
B _ { u } = \left( \prod _ { p \in [ d ] : u _ { p } = 0 } [ 0 , \tau ( p , u _ { - p } ) - \delta / 2 ] \right) \times \left( \prod _ { p \in [ d ] : u _ { p } = 1 } [ \tau ( p , u _ { - p } ) + \delta / 2 , 1 ] \right) .
$$

The additive $\pm \delta / 2$ term is simply to ensure that the following closed intervals are disjoint:

$$
[ 0 , \tau ( p , u _ { - p } ) - \delta / 2 ] \quad \mathrm { a n d } \quad [ \tau ( p , u _ { - p } ) + \delta / 2 , 1 ] .
$$

Note that the union of these boxes indeed cover $X ,$ regardless of the threshold profile T. The crux of the construction is to show that all $\begin{array} { r } { ( m + 1 ) ^ { k d / 2 } = ( \frac { 2 n } { k d } + 1 ) ^ { k d / 2 } } \end{array}$ threshold profiles are realizable while maintaining box disjointness, i.e. $B _ { u } \cap B _ { v } = \varnothing$ for all $u \neq v \in \{ 0 , 1 \} ^ { d }$

Completing the proof assuming realizability of all threshold profiles. We first note that if the above is possible, then this completes the proof. Learning each threshold on $X ( p , z )$ amounts to learning a monotone function on a domain of size $m = \textstyle { \frac { 2 n } { k d } }$ . The needed lower bound for learning each threshold is therefore given by Theorem 2.3 due to [BPR20].

Achieving box disjointness for all threshold profiles. Note that $B _ { u } \cap B _ { v } = \varnothing$ exactly when in some dimension, the corresponding intervals are disjoint. When u, v are adjacent, this holds as

$$
[ 0 , \tau ( p , u _ { - p } ) - \delta / 2 ] \cap [ \tau ( p , u _ { - p } ) + \delta / 2 , 1 ] = \emptyset .
$$

However, it is not clear a priori how to make this hold for all $u , v$ where $d ( u , v ) \geq 2 .$ . Suppose $u _ { p } = 0 , v _ { p } = 1$ and $u _ { q } \neq v _ { q }$ for some $q \neq p \in [ d ]$ . A suficient property to guarantee disjointness is $\tau ( p , u _ { - p } ) < \tau ( p , v _ { - p } )$ More generally, for every u, v where $d ( u , v ) \geq 2 .$ , we need the following to hold: there exists a coordinate $p$ such that $u _ { p } = 0 , v _ { p } = 1$ and $\tau ( p , u _ { - p } ) < \tau ( p , v _ { - p } )$ . Recall that the threshold $\tau ( p , u _ { - p } ) \in I ( p , u _ { - p } )$ is chosen from the interval partition corresponding to the edge $( p , u _ { - p } )$ (recall $\mathrm { e q . } \left( 1 \right) )$ . The ordering on these intervals has not yet been defined. We now define their ordering in such a way that the desired property is satisfied for all threshold profiles. For each $p \in [ d ] .$ , we will define a total order $\prec _ { p }$ over $z \in \{ 0 , 1 \} ^ { d - 1 }$ so that $( \prec _ { 1 } , \prec _ { 2 } , \ldots , \prec _ { d } )$ together have the following property.

Definition 3.2 (Coherent family of orderings). Call total orders $( \prec _ { 1 } , \prec _ { 2 } , \ldots , \prec _ { d } )$ over context vectors $\{ 0 , 1 \} ^ { d - 1 }$ coherent if thefollowing holds. Let $x \neq y \in \{ 0 , 1 \} ^ { d }$ be any two non-adjacent vertices, i.e. $d ( x , y ) \geq 2$ . Then, there is a coordinate p on which they difer such that the context of the $\scriptstyle p - b i t - 0$ vertex precedes the context of the $\scriptstyle p - b i t - 1$ vertex in $\prec _ { p } . \ I e . \ i f x _ { p } = 0 , y _ { p } = 1$ , we have $x _ { - p } \prec _ { p } y _ { - p }$ and $i f x _ { p } = 1 , y _ { p } = 0$ , we have $y _ { - p } \prec _ { p } x _ { - p } .$

The existence of a coherent family of total orders $( \prec _ { 1 } , \prec _ { 2 } , \ldots , \prec _ { d } )$ (according to Definition 3.2) completes the proof by the above argument. We now establish the existence of such a family. (See Figure 3 for an illustration of the final construction in 2-dimensions.)

Coherent orderings of context vectors. The heart of the construction is the following coherent family of orderings yielding the necessary structure.

Definition 3.3. For each $p \in [ d ] ,$ , let $\prec _ { p }$ be a total order on the contexts $\{ 0 , 1 \} ^ { [ d ] \backslash \{ p \} }$ defined as follows. For each $u , v \in \{ 0 , 1 \} ^ { [ d ] \backslash \{ p \} }$ , look at the first bit q where they difer. Suppose $u _ { q } = 0 , v _ { q } = 1$ $H q < p ,$ then $v \prec _ { p } u . ~ I f q > p ,$ then $u \prec _ { p } v .$ . In other words, $\prec _ { p }$ is a lexicographic order where all bits before p use the reversed bit order and all bits after p use the standard bit order.

Claim 3.4 (Coherence property). The family of orderings $( \prec _ { 1 } , \prec _ { 2 } , \ldots , \prec _ { d } )$ defined in Definition 3.3 satisfies the coherence property of Definition 3.2.

Proof. Let $a < b$ be the first two bits where x and y difer. First, suppose that $x _ { a b } = 0 0 , y _ { a b } = 1 1$ (the opposite case is symmetric). Observe that b is the first bit on which the context vectors $x _ { - a }$ and $y _ { - a }$ disagree. Since $a < b ,$ , we then have $x _ { - a } \prec _ { a } y _ { - a } $ by definition of the ordering $\prec _ { a }$

Now, suppose $x _ { a b } = 0 1 , y _ { a b } = 1 0$ (again, the opposite case is symmetric). Observe that a is the first bit on which the context vectors $x _ { - b }$ and $y _ { - b }$ disagree. Since $a < b ,$ , we then have $y _ { - b } \prec _ { b } x _ { - b }$ by definition of the ordering ≺<sub>b</sub>. □

Generalizing beyond the $k = 2 ^ { d }$ case. We have completed the proof for the case when $k = 2 ^ { d }$ . We now extend beyond this setting. For simplicity, we will assume $k = 2 ^ { \ell }$ is some power of 2.

$\ell < d .$ In this case, we can simply apply the previous construction in ℓ-dimensions and pad the instance with $d - \ell$ dummy coordinates. We have ${ \bar { \ell } } = \log _ { 2 } k = \operatorname* { m i n } ( d , \log _ { 2 } k )$ , establishing the lower bound claimed in Theorem 3.1.

$\ell > d .$ In this case, we create $2 ^ { \ell - d }$ copies of the previous construction in d dimensions, translated along the first coordinate. The n points are divided evenly among these copies. It can be easily verified that this again results in the lower bound claimed in Theorem 3.1.

This completes the proof of Theorem 3.1.

## 3.2 Beating the General Box Lower Bound in the Small-k Regime

In Section 2, we saw that learning a labeling induced by k boxes can require at least $\Omega _ { k } ( d \log ( n / d ) )$ queries when k is constant order. These boxes were allowed to overlap, although no data points may lie in the overlapping region. It turns out that if the boxes are guaranteed to be induced by disjoint boxes, then in the small-k regime, it is possible to beat the lower bound. The next result, Theorem 3.5, shows that when k is constant order, there is an upper bound of $O _ { k } ( d \log ( n ) / \log ( d ) )$ ) queries for learning the restricted class of disjoint boxes, which shaves of a factor of log(d).

The algorithm that achieves this faster rate is a variant of a classic learning algorithm, which we refer to as Halving++, described in [Heg95]. Compared to CAL, Halving++ selects query points more carefully by searching for more informative points to query. It turns out that the more methodical selection is needed; we show in Section B that there are disjoint-box problems where CAL requires $\Omega ( k d \log ( n / k d ) )$ queries to learn, strictly separating CAL and Halving++.

Theorem 3.5 (Disjoint Boxes). Let $d , k \geq 2$ . There is a deterministic algorithm that learns the labeling of a disjoint k-box partition over a worst-case set of n points in $\mathbb { R } ^ { d }$ using

$$
O \left( \operatorname* { m i n } \left\{ n , { \frac { k ^ { 3 } d \log n } { \log ( k d ) } } \right\} \right) q u e r i e s .
$$

Proof. The proof of this uses a classic learning theory result bounding the query complexity through the extended teaching dimension of a binary concept class $[ \mathrm { \dot { M } o s 8 2 } , \mathrm { H e g 9 5 } ]$

Definition 3.6 (Extended teaching dimension). Let Z be afinite set and let $\mathcal { C } \subset \{ 0 , 1 \} ^ { Z }$ . For any $h \in \{ 0 , 1 \} ^ { Z }$ a subset $S \subset Z$ is a specifying set for h with respect to the concept class C if at most one element ${ \dot { f } } \in { \mathcal { C } }$ satisfies $f ( z ) = h ( z ) f o r a l l z \in S$ . The extended teaching dimension ofC is

$$
\mathrm { E T D } ( \mathcal { C } ) = \operatorname* { m a x } _ { h \in \{ 0 , 1 \} ^ { Z } } \operatorname* { m i n } \big \{ | S | : S i s a s p e c i f y i n g s e t f o r h w i t h r e s p e c t t o \mathcal { C } \big \} .
$$

Theorem 3.7 (Query complexity bound, [Mos82, Heg95]). Let $\mathcal { C } \subset \{ 0 , 1 \} ^ { Z }$ be a concept class. $I f \mathrm { E T D } ( { \mathcal { C } } ) \geq 2$ and $\operatorname* { m a x } \{ 3 , \mathrm { E T D } ( \mathscr { C } ) \} \leq E$ , there is a deterministic algorithm that identifies any concept in $\dot { \boldsymbol { { \mathcal { C } } } }$ using at most

$$
{ \frac { 2 E \log _ { 2 } | { \mathcal { C } } | } { \log _ { 2 } E } } a d a p t i v e c o o r d i n a t e q u e r i e s .
$$

First, we convert the problem of learning a k-box partition into one of learning a binary concept class. Let $Z = X \times [ k ]$ . Each labeling $f : X \to [ k ]$ is associated to an element of $\tilde { f } \in \{ 0 , 1 \} ^ { \breve { Z } }$

$$
{ \tilde { f } } ( x , y ) = 1 \quad \Longleftrightarrow \quad f ( x ) = y .
$$

Let ${ \mathcal C } _ { X } ^ { k \mathrm { - D I S J } } \subset \{ 0 , 1 \} ^ { Z }$ denote the concept class

$$
\mathcal { C } _ { X } ^ { k \mathrm { - D I S J } } = \{ \tilde { f } : f \mathrm { ~ i s ~ a ~ d i s j o i n t ~ } k \mathrm { - b o x ~ p a r t i t i o n ~ o n ~ } X \} .\tag{2}
$$

To recover the ground truth labeling $f : X \to [ k ] .$ , it is enough to recover its associated concept ${ \tilde { f } } .$ Every coordinate query to $z = ( x , y )$ can be simulated by a single label query to $x ,$ so it follows that by running the algorithm from Theorem $3 . 7$ as a black-box, we can learn any f by making at most

$$
\operatorname* { m i n } \left\{ n , { \frac { 2 E \log _ { 2 } | { \mathcal { C } } _ { X } ^ { k \mathrm { - D I S J } } | } { \log _ { 2 } E } } \right\} { \mathrm { ~ a d a p t i v e ~ l a b e l ~ q u e r i e s } } ,
$$

where $E \geq 3$ is any upper bound on the extended teaching dimension. The result follows immediately from Lemmas 3.8 and 3.9 below, which bound $\mathrm { E T D } ( \mathcal { C } _ { X } ^ { k - \mathrm { D I S J } } )$ and $| { \mathcal C } _ { X } ^ { k - \mathrm { D I S J } } |$ □

Lemma 3.8. Let $\mathcal { C } _ { X } ^ { k - \mathrm { { D I S J } } }$ be the class of disjoint k-box partitions on $X \subset \mathbb { R } ^ { d }$ defined in (2). Then:

$$
\mathrm { E T D } ( \mathcal { C } _ { X } ^ { \mathrm { k - D I S J } } ) \leq 2 k d .
$$

Proof. Fix any $h \in \{ 0 , 1 \} ^ { Z }$ . While h is not necessary associated with a disjoint k-box labeling, but we can still interpret it as assigning each point $x \in X$ to a possibly empty set of labels, where the assignment is $x \mapsto \{ y \in { \bar { [ k ] } } : h ( x , y ) = { \bar { 1 } } \}$ . There are three cases:

• If there is an $x \in X$ assigned zero labels or strictly more than one label, then the following set $S = \{ ( x , y ) : y \in [ k ] \}$ is a specifying set for h with respect to C. No $\tilde { f } \in \mathcal { C } _ { X } ^ { k - \mathrm { D I S J } }$ such that $\tilde { f } ( x , y ) = h ( x , y )$ for all $x \in S$ and $y \in [ k ]$

• Suppose that each $x \in X$ is assigned precisely one label. For each $y \in [ k ]$ , let $B _ { y }$ be the bounding box of points assigned the label y by h. In each dimension of this bounding box, we can let $x _ { y , i , L }$ and $x _ { y , i , R }$ be the left- and rightmost points out of all points labeled $y .$ . Define $S$ to be the union of these bounding points, so that S is a set of size at most 2kd. There is precisely one (not necessarily disjoint) box partition that induces the same labeling on S.

By definition, the extending teaching dimension of $\mathcal { C } _ { X } ^ { k - \mathrm { { D I S J } } }$ is at most 2kd.

Lemma 3.9. There are at most $( 2 k d n ) ^ { \binom { k } { 2 } }$ disjoint k-box partitions on any n-point set $X \subset \mathbb { R } ^ { d }$

Proof. Define $\Theta = X \times \left[ d \right] \times \{ - 1 , + 1 \}$ . We will construct a family of test functions $t _ { \theta } : X  \{ 0 , 1 \}$ parametrized by $\dot { \theta } \in \Theta$ such that the following holds: for any disjoint k-box labeling $f : X \to [ k ]$ and pair of distinct labels $y _ { 0 } , y _ { 1 } \in [ k ]$ , there is a test function $t _ { \theta }$ such that

$$
f ( x ) = y _ { 0 } \implies t _ { \theta } ( x ) = 0 \qquad \mathrm { a n d } \qquad f ( x ) = y _ { 1 } \implies t _ { \theta } ( x ) = 1 .
$$

Thus, $f$ is completely determined by a set of $\binom { k } { 2 }$ parameters $\{ \theta ( f , \ell _ { 0 } , \ell _ { 1 } ) : 1 \le \ell _ { 0 } < \ell _ { 1 } \le k \}$ , and

$$
| { \mathcal { C } } _ { X } ^ { k - \mathrm { D I S J } } | \leq | \Theta | ^ { \binom { k } { 2 } } .
$$

The claim holds because $| \Theta | = 2 d n$ . We just need to construct the family of test functions.

For any labeling $f : X \to [ k ]$ and pair of distinct labels $y _ { 0 } , y _ { 1 } \in [ k ]$ , let $B _ { 0 }$ and $B _ { 1 }$ respectively be the bounding boxes of points in class y<sub>0</sub> and $y _ { 1 }$ . These two boxes are disjoint if and only if there exists a separating dimension $i \in [ d ]$ such that the projection of these boxes onto the ith dimension is two disjoint intervals. Let $x \in X \cap B _ { 1 }$ minimize the distance to $B _ { 0 }$ after projecting to the ith dimension

$$
\operatorname* { m i n } _ { x _ { 0 } \in B _ { 0 } } | x _ { 0 } ( i ) - x ( i ) | \leq \operatorname* { m i n } _ { x _ { j } \in X \cap B _ { j } } | x _ { 0 } ( i ) - x _ { 1 } ( i ) | .
$$

Let $s = \mathrm { s i g n } \bigl ( x ( i ) - x _ { 0 } ( i ) \bigr )$ for any $x _ { 0 } \in B _ { 0 }$ . This is well-defined because $x ( i )$ is either strictly greater or strictly less than the ith coordinate of every $x _ { 0 } \in B _ { 0 }$ . A simple test to distinguish points in class $\ell _ { 0 }$ and $\ell _ { 1 }$ is the following test parametrized by $\theta = \bar { ( } i , x , s )$

$$
t _ { \theta } ( x ^ { \prime } ) = \mathbf { 1 } \{ s \cdot x ^ { \prime } ( i ) \geq s \cdot x ( i ) \} .
$$

By construction, $t _ { \theta } ( x _ { 0 } ) = 0$ for all $x _ { 0 } \in B _ { 0 }$ and $t _ { \theta } ( x _ { 1 } ) = 1$ for all $x _ { 1 } \in B _ { 1 }$

Remark 3.10 (Applying the ETD argument to general boxes). The same argument can be applied to the earlier, general setting of learning k potentially-overlapping boxes. The proof of Lemma 3.8, which gave a bound of 2kd on the extended teaching dimension, also applies to general boxes; nowhere was disjointness used. The bound on the number of

k-box partitions can be upper bounded by $n ^ { 2 k d }$ , each box can be identified with two endpoints in each dimension. Then, Theorem 3.7 yields

$$
{ \frac { 8 k ^ { 2 } d ^ { 2 } \log n } { \log ( 2 k d ) } } q u e r i e s .
$$

This upper bound is worse than Theorem 1.1, although it is likely that the analysis is loose. But, for disjoint boxes, the savings achieved in Theorem 3.5 intuitively relies the fact that we can distinguish those boxes by a set $o f \Theta ( k ^ { 2 } )$ separating hyperplanes, rather than their $\Theta ( k d )$ endpoints.

## 4 Decision Trees

In this section we prove our results for decision trees stated in Theorem 1.3.

Definition 4.1 (Decision trees). Let Y be a label set of size $| Y | = k \geq 1 . A$ function $f \colon  { \mathbb { R } ^ { d } } \to Y$ is a k-leaf decision tree if either $( a ) \ k = 1 , o r \ ( b )$ there exists $i \in [ d ] , \tau \in \mathbb { R }$ , and a strict subset $\emptyset \neq S \subsetneq Y$ such that thefollowing hold:

• f restricted on $\{ x \in \mathbb { R } ^ { d } \colon x _ { i } \leq \tau \}$ is a |S|-leaf decision tree with label set S and

• f restricted on $\{ x \in \mathbb { R } ^ { d } \colon x _ { i } > \tau \}$ is a $\left( k - | S | \right)$ )-leaf decision tree with label set $Y \setminus S .$

A k-leaf decision list is defined analogously, with the added restriction that $| S | = 1$ is a singleton.

A decision stump is a 2-leaf decision tree. Our results in this section use the following learner for decision stumps as a black-box.

Theorem 4.2 (Corollary 2 item (1) of $\left[ \mathrm { B L M } ^ { + } 2 6 \right] )$ . Let $X \subset \mathbb { R } ^ { d }$ be an arbitrary set of n points. There exists a deterministic algorithm, that exactly learns an arbitrary decision stump $f \colon X \to \{ 0 , 1 \}$ using $O ( d + \log n )$ queries. If the input f is not realizable by a decision stump, the algorithm still halts within the claimed query budget and returns some labeling $g \colon X  \{ 0 , 1 \}$

We use stumpLearn $( f , X )$ to denote the algorithm of Theorem 4.2. To see that the claim holds regarding inputs not realizable by decision stumps, one can simply truncate the learner at its promised worst-case query depth. If the learner is not done by this point, complete the labeling arbitrarily and return the result. Of course, correctness is only guaranteed in the realizable case.

## 4.1 Decision Tree Learner

We prove the following theorem, obtaining a $2 ^ { O ( k ) } ( d + \log n )$ upper bound.

Theorem 4.3. Let Y be an arbitrary label-set ofsize $| Y | = k \geq 1$ . There is a deterministic algorithm, which for any set $X \subset \mathbb { R } ^ { d }$ of n points, learns an unknown labeling $f \colon X \to Y$ induced by a k-leaf decision tree using

$$
2 ^ { 2 k + O ( \log ^ { 2 } k ) } ( d + \log n ) = 2 ^ { O ( k ) } ( d + \log n )
$$

queries. If f is induced by a k-leaf decision list, then the query complexity is $O ( k ^ { 2 } ( d + \log n ) )$

Before proving Theorem 4.3, we remark that it is fairly direct to obtain a $k ^ { O ( k ) } ( d + \log n )$ learner by enumerating all decision trees over the label set and applying stumpLearn at every internal node.

Remark 4.4 (A simple tree-enumeration algorithm). We give a sketch ofa $k ^ { O ( k ) } ( d + \log n )$ query algorithm. Let T be a binary tree with k leaves where every non-leaf has exactly two children. Note that there are exactly $C _ { k - 1 }$ (the (k −1)-th Catalan number) choices for T. Then, let ϕ be a bijective map from the label set Y to the leaves of T (there are k! such maps). For a node u in T, let $\overset { \cdot } { Y _ { u } } \subseteq Y$ denote the set of labels associated by ϕ to some leaf-descendant of u. Now, for each of the $k - 1$ non-leaf nodes u, run stumpLearn (Theorem 4.2) with Boolean oracle $g _ { u } ( x ) = \mathbf { 1 } ( f ( x ) \in Y _ { \mathsf { r i g h t } _ { T } ( u ) } )$ Combining the outputs of these calls to the stump learner yields a candidate labeling $f _ { T , \phi } \colon X \to Y$ . By Definition 4.1, there exists at least one choice of $( T , \phi )$ such that $f _ { T , \phi } = f .$ . It is then straightforward to determine which of these $C _ { k - 1 } k !$ candidate labelings is the correct one using at most one query per candidate. The total query complexity is

$$
O ( C _ { k - 1 } k ! k ( d + \log n ) ) = k ^ { O ( k ) } ( d + \log n ) .
$$

Our improvement over the brute-force enumeration described in Remark 4.4 comes from the following two observations:

• There is a simple procedure (Claim 4.5) to certify correctness at the leaves, i.e. when the label-set is a singleton. This allows us to make permanent progress (by labeling one class) without needing to guess the entire correct binary tree over the labels all at once.

• Given label-set $Y ,$ there exists $\varnothing \subsetneq A ^ { * } \subsetneq Y$ where the root-level partition on the labels is $( A ^ { * } , Y \setminus A ^ { * } )$ such that $| A ^ { * } | \le \lfloor | Y | / 2 \rfloor$ . As a result, the recurrence governing the query complexity has the form $Q ( k ) \lesssim 2 ^ { k } Q ( k / 2 )$

Decision tree learner. We now describe the algorithm, which uses the following subroutine in addition to the stump learner of Theorem 4.2.

Claim 4.5 (Monochromatic Certification). Let $f \colon X \to Y$ be a labeling induced by a box partition, $Z \subseteq X$ , and $y \in Y$ . There is a deterministic procedure monoC $\cdot \mathsf { e r t } ( f , Z , y )$ using at most 2d queries that decides whether or not $f | _ { Z } \equiv y .$

Proof. The procedure simply queries all points in the set

E<sub>Z</sub> = arg min zi, arg max zi : i ∈ [d]   
z∈Z z∈Z   
of extremal points in Z with respect to each coordinate. Clearly $| E _ { Z } | \le 2 d$ . We then accept if and only if   
$f | _ { E _ { Z } } \equiv y .$ . Clearly, if $f | _ { E _ { z } } \notin y ,$ , then $f | _ { Z } \not \equiv y .$ Conversely, since $Z \subseteq { \mathsf { b b o x } } ( E _ { Z } ) , { \mathrm { i f ~ } } f | _ { E _ { z } } \equiv y ,$ , then $f | _ { Z } \equiv y$ □   
Given $f \colon X \to Y .$ , our algorithm is defined recursively as follows.   
DTLearn $( f , X , Y ) ;$   
1. Base cases. If $X = \varnothing ,$ return the empty labeling. If $Y = \{ y \}$ , run mono $\mathsf { \Phi } _ { \mathsf { c e r t } } ( f , X , y )$ (Claim 4.5). If   
it accepts, return the constant labeling $f | _ { X } \equiv y ,$ otherwise return FAIL.   
2. If $| Y | = k \geq 2 ,$ , then let $R  X$ and repeat the following procedure defined in item (3) k times, or   
until $R = \varnothing .$ (See line (3e).)   
3. For every $A \subset Y$ where $1 \leq | A | \leq \lfloor k / 2 \rfloor$ , do the following:   
(a) Define $g _ { A } \colon R \to \{ 0 , 1 \}$ where $g _ { A } ( x ) = \mathbf { 1 } ( f ( x ) \in A )$   
(b) Run stumpLearn $( g _ { A } , R )$ and let $\widetilde { g } _ { A }$ denote the returned Boolean labeling.   
(c) Let $X _ { A } = \{ x \in R \colon \widetilde { g } _ { A } ( x ) = 1 \}$   
(d) Run DTLearn $( f | _ { X _ { A } } , X _ { A } , A )$ . If it returns FAIL, then continue. If it returns a labeling $h _ { A } \colon X _ { A } \to$   
A, then assign these labels to $X _ { A }$ and set $R  R \backslash X _ { A }$   
(e) If $R = \varnothing ,$ , return the accumulated labeling on the original input set.   
4. If after k repetitions of item (3), we have $R \neq \emptyset ,$ , then return FAIL.

Before proving correctness, we give some intuition for the above learner. Since f is a decision tree, for some choice of A in line (3), the Boolean labeling $g _ { A }$ in line (3a) is a decision stump. By correctness of the decision stump learner, this implies that $f | _ { X _ { A } }$ is a $| A |$ -leaf decision tree and thus by induction the points in $X _ { A }$ are correctly labeled in line (3d). This removes at least one of the k classes from $R ,$ which is the reason that at most k repetitions of (3) are needed. Importantly, even when $g _ { A }$ is not a decision stump, any labeling returned by the recursive call is correct, preventing us from ever mislabeling points.

## 4.1.1 Correctness Proof

Lemma 4.6 (Soundness). Suppose $f \colon X \to Y$ satisfies $f ^ { - 1 } ( y ) = X \cap B _ { y }$ where $\{ B _ { y } \colon y \in Y \}$ are disjoint axis-parallel boxes. Let $Y ^ { \prime }$ be a candidate label set for f. If DTLearn $( f , X , Y ^ { \prime } )$ returns a labeling h, then $f = h$

Proof. Every label assignment returned by DTLearn ultimately comes from a successful monochromatic certification at a singleton base case. Claim 4.5 guarantees that every such assignment is correct, regardless of whether the candidate label set $Y ^ { \prime }$ contains all true labels. A complete labeling is returned only after every point has received a label. □

Lemma 4.7 (Correctness). Suppose $f \colon X \ \to \ Y$ is induced by a ℓ-leaf decision tree where $\ell \ \leq \ | Y |$ . Then DTLearn $( f , X , Y )$ always returns the correct labeling f.

Proof. The proof is by induction on $k = | Y |$ . The case $X = \emptyset$ is clear and the case $k = 1$ follows from Claim 4.5. Now, let $k \geq 2 .$ . Consider the set R defined in line (2) and consider some iteration of step (3) in which $R \neq \emptyset$ We argue that one iteration of step (3) correctly labels all points in at least one of the k labels. (Therefore, k repetitions are enough, as claimed in line (2).) Note that by Lemma $4 . 6 ,$ a recursive call will never incorrectly label points.

First, suppose $f | _ { R } \equiv y$ is constant, for some $y \in Y$ . Then, in step (3), the algorithm will try $A = \{ y \}$ , in which $g _ { \{ y \} } \equiv 1$ (in line $\left( 3 \mathsf { a } \right) ,$ ). This will result stumpLearn $( g _ { \{ y \} } , R )$ returning the all-1 labeling of R and sets $X _ { A } = R$ in line (3c). Line (3d) then invokes the base case, accepts, and labels all of R.

Now, suppose R contains at least 2 labels from Y. Note that $f | _ { R } \colon R \to Y$ is itself induced by a k<sup>′</sup>-leaf decision tree where $k ^ { \prime } \leq k$ (this holds for any subset of X). Therefore, there exists some $A ^ { * } \subset Y$ where $1 \leq | A ^ { * } | \leq \lfloor k / 2 \rfloor$ such that either $f ( x ) \in A ^ { * } \operatorname { i f f } x _ { i } \leq \tau$ (for some $i , \tau )$ or $f ( x ) \in A ^ { * }$ if $x _ { i } > \tau$ . In some iteration of line (3), the algorithm tries this A<sup>∗</sup>. The guarantee of stumpLearn $( g _ { A ^ { * } } , R )$ (Theorem $4 . 2 )$ implies that $\widetilde g _ { A ^ { * } } = g _ { A ^ { * } }$ ∗ and thus $X _ { A ^ { * } } = \{ x \in R \colon f ( x ) \in A ^ { * } \bar { \} }$ . By induction, the recursive call to DTLearn $( f _ { X \mid _ { A ^ { * } } } , X _ { A ^ { * } } , A ^ { * } )$ successfully returns a labeling on $X _ { A ^ { * } }$ , which by Lemma 4.6 is guaranteed to be correct. Therefore, at the end of step (3d) all points with labels in $A ^ { * }$ are removed from R.

We have shown that every repetition of line (3) successfully learns at least one of the k labels. Combining with Lemma 4.6, this shows that after at most k repetitions, the algorithm returns the correct labeling, f.

## 4.1.2 Query Complexity Proof

We derive the query complexity of DTLearn as follows. Fix d and n. Let $q _ { s } = O ( d + \log n )$ denote the query complexity of the stump learner in d-dimensions over n points and let $q _ { c } = 2 d$ denote the query complexity of the monochromatic certification. For $k \geq 1$ , let $Q ( k )$ denote the query complexity of DTLearn on n points in d dimensions with k labels. Then, by inspection, Q satisfies the following recurrence relation:

$$
Q ( 1 ) \leq q _ { c } { \mathrm { ~ a n d ~ } } Q ( k ) \leq k 2 ^ { k } ( q _ { s } + Q ( \lfloor k / 2 \rfloor ) ) { \mathrm { ~ w h e n ~ } } k \geq 2 .\tag{3}
$$

We now bound the recurrence, which establishes the claimed query complexity. We can expand the recurrence for $k \geq 2$ as follows:

$$
\begin{array} { r l r } {  { Q ( k ) \le k 2 ^ { k } ( q _ { s } + Q ( \lfloor k / 2 \rfloor ) ) } } \\ & { } & { \le k 2 ^ { k } q _ { s } + k 2 ^ { k } ( \frac { k } { 2 } \cdot 2 ^ { k / 2 } ( q _ { s } + Q ( \lfloor \lfloor k / 2 \rfloor / 2 \rfloor ) ) ) } \end{array}
$$

$$
\leq ( q _ { s } + q _ { c } ) \sum _ { t = 0 } ^ { \lfloor \log _ { 2 } k \rfloor } \left( \prod _ { j = 0 } ^ { t } \frac { k } { 2 ^ { j } } 2 ^ { k / 2 ^ { j } } \right)\tag{4}
$$

Using upper bounds $t \leq \lfloor \log _ { 2 } k \rfloor$ and $k / 2 ^ { j } \leq k$ shows that the RHS above is at most

$$
\left( q _ { s } + q _ { c } \right) \cdot ( \log _ { 2 } k + 1 ) \cdot \prod _ { j = 0 } ^ { \lfloor \log _ { 2 } k \rfloor } k 2 ^ { k / 2 ^ { j } } \leq ( q _ { s } + q _ { c } ) ( \log _ { 2 } k + 1 ) \cdot k ^ { \log _ { 2 } k + 1 } \cdot 2 ^ { 2 k } \leq 2 ^ { 2 k + O ( \log ^ { 2 } k ) } \cdot ( q _ { s } + q _ { c } )\tag{5}
$$

and this completes the proof.

Quadratic bound for decision lists. We now describe how to modify DTLearn to get an improved query complexity for decision lists (recall Definition 4.1). The simple change is to modify step (3) to try only singleton sets for A. The algorithm is correct by the same arguments in Lemma 4.6 and Lemma $4 . { \dot { 7 } }$ . The algorithm correctly learns the labeling simply after k repetitions of step (3) and only ever recurses on a singleton set of labels. Therefore, by inspection the query complexity is at most $k ^ { 2 } ( q _ { s } + q _ { c } )$

## 4.2 Lower Bound

We now prove an elementary lower bound establishing optimality of our algorithm for constant k.

Theorem 4.8 (Decision List Lower Bound). Let $d , k \geq 2 .$ . There exists set $X \subset \mathbb { R } ^ { d } o f n \geq d ( k - 1 )$ points such that any randomized adaptive algorithm that learns a labeling $f \colon X \to [ k ]$ induced by a k-leaf decision list with success probability at least $2 / 3$ must use $\Omega ( k ( d + \log ( n / k ) ) )$ queries.

Proof. It sufices to prove an $\Omega ( k d )$ lower bound and a $\Omega ( k \log ( n / k ) )$ lower bound separately.

The $\Omega ( k d )$ bound. We first describe the construction for $n = d ( k - 1 )$ points. Let $U = \{ e _ { 1 } , \ldots , e _ { d } \}$ denote the standard unit basis vectors. For a vector v, let $U + v \ = \ \{ u + v \colon u \in U \}$ . For a scalar $c \in \mathbb { R }$ , let $\overline { { c } } = ( c , c , \ldots , c ) \in \mathbb { R } ^ { d }$ . Our point-set X is defined

$$
X : = \bigcup _ { y = 1 } ^ { k - 1 } ( U + { \overline { { y } } } ) .
$$

Then, we consider the family of decision lists formed as follows. For each $y \in [ k - 1 ] ,$ choose a dimension $i _ { y } \in [ d ]$ . Given $\alpha = ( i _ { 1 } , \ldots , i _ { k - 1 } ) \in [ d ] ^ { k - 1 }$ , we then define $f _ { \alpha } \colon  { \mathbb { R } ^ { d } } \to [ k ]$ as

$$
f _ { \alpha } ( x ) = \operatorname* { m i n } \left( \left\{ y \in [ k - 1 ] \colon x _ { i _ { y } } \leq y + 0 . 5 \right\} \cup \{ k \} \right) .
$$

To reach any $n \geq d ( k - 1 )$ points, we can pad the point-set with $n - d ( k - 1 )$ arbitrary points z such that $z _ { i } > k$ for all $i \in [ d ]$

Observe that such a labeling is a valid decision list (Definition 4.1). The key property of this construction is the following: for each $y \in [ k - 1 ] , f ( e _ { i _ { y } } + \overline { { y } } ) = y + 1$ and $f ( e _ { i } + { \overline { { y } } } ) = y$ for all $\mathsf { \bar { \Phi } } _ { i } \in \mathsf { \bar { [ } } d ] \setminus \{ i _ { y } \}$ . Therefore, learning the Boolean labeling in $U + { \overline { { y } } }$ requires solving an unstructured search problem on d elements, which requires $\Omega ( d )$ queries for any randomized algorithm. Summing over $U + { \overline { { y } } }$ for each $y \in [ k - 1 ]$ shows that $\Omega ( \bar { k } d )$ queries are needed.

The $\Omega ( k \log ( n / k ) )$ bound. Assume n is a multiple of $k - 1$ . It sufices to prove a lower bound for learning monotone functions $f \colon [ n ]  [ k ]$ . First, when $k = 2 ,$ , a lower bound of $\Omega ( \log n )$ holds by a standard information-theoretic argument, since one needs to find a threshold in one of n possible locations using queries that return a single bit of information.

Now, consider the following family of monotone functions. First, partition [n] into $k - 1$ intervals $I _ { 1 } , \ldots , I _ { k - 1 }$ of length $n / ( k - 1 )$ . In interval $I _ { y } ,$ we allow the function to take values in $\{ y , y + 1 \}$ . Learning such a function requires the algorithm to learn the correct Booleanfunction restricted on each $I _ { y }$ and crucially a query to $x \in I _ { y }$ gives no information about the function on any other interval. Thus, leveraging the $k = 2$ lower bound, we obtain a lower bound of

$$
( k - 1 ) \cdot \Omega \left( \log \left( { \frac { n } { k - 1 } } \right) \right) = \Omega ( k \log ( n / k ) )
$$

and this completes the proof.

## AI Disclosure

ChatGPT 5.6 was used during this work to explore proof ideas. All final proofs and exposition have been written and verified by the authors.

## References

[AKBD16] Hassan Ashtiani, Shrinu Kushagra, and Shai Ben-David. Clustering with same-cluster queries. Advances in Neural Information Processing Systems, 29, 2016. 7

[Ang87] Dana Angluin. Queries and concept learning. Mach. Learn., 2(4):319–342, 1987. 3, 7

[BCLP24] Marco Bressan, Nicolò Cesa-Bianchi, Silvio Lattanzi, and Andrea Paudice. Margin-based active learning of classifiers. J. Mach. Learn. Res., 25:127:1–127:45, 2024. 7

[BDBK97] Shai Ben-David, Nader H Bshouty, and Eyal Kushilevitz. A composition theorem for learning algorithms with applications to geometric concept classes. In Proceedings of the Twenty-Ninth Annual ACM Symposium on Theory ofComputing, pages 324–333, 1997. 7

[BGGM98] Nader H Bshouty, Paul W Goldberg, Sally A Goldman, and H David Mathias. Exact learning of discretized geometric concepts. SIAM journal on Computing, 28(2):674–699, 1998. 7

[BGM<sup>+</sup>98] Nader H Bshouty, Sally A Goldman, H David Mathias, Subhash Suri, and Hisao Tamaki. Noisetolerant distribution-free learning of general geometric concepts. Journal of the ACM (JACM), 45(5):863–890, 1998. 7

[BLM<sup>+</sup>26] Hadley Black, Kasper Green Larsen, Arya Mazumdar, Barna Saha, and Geelon So. Actively learning halfspaces without synthetic data. In Steve Hanneke and Tor Lattimore, editors, The Thirty Ninth Annual Conference on Learning Theory, June 29 - July 3, 2026, San Diego, California, USA, volume 336 of Proceedings ofMachine Learning Research, pages 708–728. PMLR, 2026. 5, 6, 7, 16

[BPR20] Simina Brânzei, Dimitris Paparas, and Nicholas J. Recker. Searching, sorting, and cake cutting in rounds. CoRR, abs/2012.00738, 2020. 7, 9, 13

[CAL94] David A. Cohn, Les E. Atlas, and Richard E. Ladner. Improving generalization with active learning. Mach. Learn., 15(2):201–221, 1994. 4, 7, 8

[Das04] Sanjoy Dasgupta. Analysis of a greedy active learning strategy. In Advances in Neural Information Processing Systems 17, pages 337–344, 2004. 3, 7

[GSSS13] Alon Gonen, Sivan Sabato, and Shai Shalev-Shwartz. Eficient active learning of halfspaces: an aggressive approach. In International Conference on Machine Learning, pages 480–488. PMLR, 2013. 7

[Heg95] Tibor Hegedüs. Generalized teaching dimensions and the query complexity of learning. In Wolfgang Maass, editor, Proceedings ofthe Eighth Annual Conference on Computational Learning Theory, COLT 1995, Santa Cruz, California, USA, July 5-8, 1995, pages 108–117. ACM, 1995. 4, 6, 7, 14

[HKLM21] Max Hopkins, Daniel Kane, Shachar Lovett, and Michal Moshkovitz. Bounded memory active learning through enriched queries. In Conference on Learning Theory, pages 2358–2387. PMLR, 2021. 7

[HY15] Steve Hanneke and Liu Yang. Minimax analysis of active learning. The Journal of Machine Learning Research, 16(1):3487–3602, 2015. 7

[MDRF20] Michal Moshkovitz, Sanjoy Dasgupta, Cyrus Rashtchian, and Nave Frost. Explainable k-means and k-medians clustering. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings ofMachine Learning Research, pages 7055–7065. PMLR, 2020. 7

[Mos82] Mikhail Yuryevich Moshkov. On conditional tests. In Doklady Akademii Nauk, volume 265, pages 550–552. Russian Academy of Sciences, 1982. 4, 6, 14

[Set09] Burr Settles. Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009. $3 , { \dot { 7 } }$

[SMLG<sup>+</sup>26] Arshia Soltani Moakhar, Tanapoom Laoaron, Faraz Ghahremani, Kiarash Banihashem, and MohammadTaghi Hajiaghayi. Active learning for decision trees with provable guarantees. In International Conference on Learning Representations, volume 2026, pages 8333–8375, 2026. 7

## A Query Complexity with Round Constraints

This section proves Theorem 2.2. Before we do so, we first prove the following key lemma about labelings that are induced by k boxes. Consider any set of points X with initially unknown labels. This lemma shows that if we label a random subset of points $Z \subseteq { \bar { X } }$ of size $O ( k d m )$ , then in expectation, we will be able to deduce the labels for all but a $\begin{array} { r } { O ( \frac { 1 } { m } ) } \end{array}$ )-fraction of points in $X .$

Lemma A.1 (Expected coverage of a random set). Let $X \subset \mathbb { R } ^ { d }$ be a set $o f \left| X \right| = n$ points and let $f \colon X \to [ k ]$ be a labeling induced by k boxes. Given afixed integer $0 \leq m \leq n ,$ let $Z \subseteq X$ be a random subset of X of size exactly m. Let $Q$ be a random variable denoting the number ofpoints whose label cannot be inferredfrom $f ( Z )$ . Then,

$$
\mathbb { E } _ { Z } [ Q ] \le 2 d \sum _ { j = 1 } ^ { k } \operatorname* { m i n } \left( | f ^ { - 1 } ( j ) | , \frac { n - m } { m + 1 } \right) \le 2 d k \cdot \frac { n - m } { m + 1 } .
$$

Proof. For each label $y \in [ k ]$ , define $X _ { y }$ and $Z _ { y }$ to be the points with label y in X and $Z ,$ respectively. Let $n _ { y } \doteq | X _ { y } |$ be the number of points in $X _ { y } .$ . Let $\begin{array} { r } { Q = \sum _ { y } \bar { Q _ { y } } } \end{array}$ where $Q _ { y }$ is the number of points in $X _ { y }$ whose label cannot be inferred from the labels $f ( Z )$ , where

the label $f ( x ) = y$ can be inferred from $f ( Z ) \iff x \in { \mathsf { b b o x } } { \big ( } Z _ { y } { \big ) } = \prod _ { i = 1 } ^ { d } [ a _ { y , i } , b _ { y , i } ] .$

For each dimension $i \in [ d ] ,$ , we also count the number of points in $X _ { y }$ that land to the left and right of bbox $\left( Z _ { y } \right)$ in the ith dimension:

$$
L _ { y , i } = \# \big \{ x \in X _ { y } : x _ { i } < a _ { y , i } \big \} \qquad \mathrm { a n d } \qquad R _ { y , i } = \# \big \{ x \in X _ { y } : x _ { i } > b _ { y , i } \big \} .
$$

When $Z _ { y } = \emptyset .$ , we use the convention $L _ { y , i } = R _ { y , i } = n _ { y }$ . We can upper bound $Q _ { y }$ by the sum

$$
Q _ { y } \leq \sum _ { i = 1 } ^ { d } L _ { y , i } + R _ { y , i } .
$$

We now bound $\mathbb { E } _ { Z } [ L _ { y , i } ]$ . The bound on $\mathbb { E } _ { Z } [ R _ { y , i } ]$ is the same and has an analogous argument.

For each dimension $i \in [ d ]$ , let $\mathsf { r a n k } _ { y , i } ( x )$ denote the ordering of the points $x \in X _ { y }$ with respect to the ith coordinate, breaking ties arbitrarily. For each integer $\ell \geq 0$ , define the ℓ leftmost set of points under this ordering to be $X _ { y , i } ^ { \le \ell } = \{ x \in X _ { y } : \mathsf { r a n k } _ { y , i } ( x ) \leq \ell \}$ . Then:

$$
L _ { y , i } \geq \ell \qquad \implies \qquad Z \cap X _ { y , i } ^ { \leq \ell } = \emptyset .
$$

To see this, suppose that $L _ { y , i } \geq \ell .$ . Every point $z \in Z _ { y }$ must strictly dominate a set of $L _ { y , i }$ points with respect to $\mathsf { r a n k } _ { y , i }$ , which means that ran $| \mathsf { k } _ { y , i } ( z ) > \ell .$ . As Z is chosen uniformly at random,

$$
\operatorname* { P r } _ { Z } \left[ L _ { y , i } \geq \ell \right] \leq \operatorname* { P r } _ { Z } \left[ Z \cap X _ { y , i } ^ { \leq \ell } = \varnothing \right] \leq \frac { { \binom { n - \ell } { m } } } { { \binom { n } { m } } } , \qquad \mathrm { f o r ~ a l l } ~ 1 \leq \ell \leq n _ { y } .
$$

The last inequality holds because $Z$ must miss a set of size ℓ. Using a layer-cake representation and applying the hockey-stick identity, we obtain

$$
\mathbb { E } _ { Z } \left[ L _ { y , i } \right] \le \sum _ { \ell = 1 } ^ { n - m } \operatorname* { P r } _ { Z } \left[ L _ { y , i } \ge \ell \right] \le \sum _ { \ell = 1 } ^ { n - m } \frac { { \binom { n - \ell } { m } } } { { \binom { n } { m } } } = \sum _ { t = m } ^ { n - 1 } \frac { { \binom { t } { m } } } { { \binom { n } { m } } } = \frac { { \binom { n } { m + 1 } } } { { \binom { n } { m } } } = \frac { n - m } { m + 1 } .
$$

Putting everything together observing that $Q _ { y } \leq n _ { y }$ always holds, we have

$$
\mathbb { E } _ { Z } \left[ Q \right] = \sum _ { y \in \left[ k \right] } \mathbb { E } _ { Z } \left[ Q _ { y } \right] \leq 2 d \sum _ { y \in \left[ k \right] } \operatorname* { m i n } \left\{ n _ { y } , \frac { n - m } { m + 1 } \right\} ,
$$

which completes the proof.

Proof of Theorem 2.2. The algorithm, defined as follows, can be seen as a batched-variant of CAL. Each round, instead of picking a single point from the disagreement region, randomly selects a subset of points. To formally define the algorithm, pick a permutation π : $[ n ] \to X$ , then batch the points into r sets. In the t-th round where $1 \leq t \leq r ,$ query all points in the t-th batch whose labels cannot already be inferred from previous queries. We will fix a sequence of r batch sizes so that the expected query complexity over π is minimized. Towards this, given an arbitrary schedule

$$
0 = s _ { 0 } < s _ { 1 } < \cdot \cdot \cdot < s _ { r } = n ,
$$

the t-th batch will be $( \pi ( i ) \colon s _ { t - 1 } < i \leq s _ { t } )$ . We first obtain an expression for the expected query complexity of a general schedule using Lemma A.1. For $1 \leq t \leq r .$ , let $Q _ { t }$ be the random variable denoting the number of queries made in the t-th round (over the randomness of π). For $t \geq 2$ , conditioned on $Y \subseteq { \check { X } }$ being the first $| Y | = s _ { t }$ points in the ordering induced by $\pi ,$ , the first $s _ { t - 1 }$ such points form a uniformly random subset of Y of size $s _ { t - 1 }$ . Therefore, to bound $\mathbb { E } _ { \pi } [ Q _ { t } ]$ we can simply apply Lemma A.1 with $s _ { t } , s _ { t - 1 }$ playing the roles of $n ,$ m respectively. We obtain

$$
\mathbb { E } _ { \pi } [ Q ] = s _ { 1 } + \sum _ { t = 2 } ^ { r } \mathbb { E } _ { \pi } [ Q _ { t } ] \leq s _ { 1 } + 2 d k \sum _ { t = 2 } ^ { r } \frac { s _ { t } - s _ { t - 1 } } { s _ { t - 1 } + 1 } \leq s _ { 1 } + 2 d k \sum _ { t = 2 } ^ { r } \frac { s _ { t } } { s _ { t - 1 } + 1 }\tag{6}
$$

To choose a schedule that optimizes the above expected query complexity, for each $1 \leq t \leq r ,$ set

$$
s _ { t } = \left\lfloor 2 d k \cdot \left( { \frac { n } { 2 d k } } \right) ^ { t / r } \right\rfloor .\tag{7}
$$

Substituting this into Equation (6) yields

$$
\mathbb { E } _ { \boldsymbol \pi } [ Q ] \le 2 d \boldsymbol { k } \cdot \Big ( \frac { n } { 2 d \boldsymbol { k } } \Big ) ^ { 1 / r } + 2 d \boldsymbol { k } \sum _ { t = 2 } ^ { r } \Big ( \frac { n } { 2 d \boldsymbol { k } } \Big ) ^ { 1 / r } = 2 d \boldsymbol { k } r \Big ( \frac { n } { 2 d \boldsymbol { k } } \Big ) ^ { 1 / r }
$$

as claimed.

Remark A.2. The optimal $O ( k d \log ( n / d k ) )$ query complexity can be obtained by setting the round complexity to only $r = \log _ { 2 } ( n / 2 d k )$ in Theorem 2.2. From Equation (7) in the proof, we see that the t-th round considers points in a batch of size

$$
s _ { t } - s _ { t - 1 } \approx 2 d k \cdot 2 ^ { t - 1 }
$$

and the number of queries made in that batch is expected to be

$$
\stackrel { < } { \approx } 2 d k \cdot \frac { s _ { t } } { s _ { t - 1 } } \approx 2 k d .
$$

## B Separation between CAL and Halving++

This section gives a construction of a disjoint-box problem where CAL uses strictly more queries in expectation than Halving++. We restate the query complexities from Sections 2 and 3 here. Treating the number of boxes $k = O ( 1 )$ as a constant, they are

$$
\mathrm { C A L : } \ O _ { k } \left( d \log ( n / d ) \right) \mathsf { q u e r i e s } \qquad \quad \mathrm { H a l v i n g } + + : O _ { k } \left( d \log ( n ) / \log ( d ) \right) \mathsf { q u e r i e s } .
$$

While it seems that Halving++ attains a log(d)-factor saving, these bounds cannot be directly compared because the bound for CAL applies to general boxes, while the one for Halving++ applies to disjoint boxes; we cannot deduce that Halving++ is more label-eficient than $\mathrm { { C A L } }$

Nevertheless, the next result strictly separates these two algorithms. We construct a disjoint-box problem where the upper bound we proved for CAL is tight. In particular, if we let $n = d ^ { 2 }$ , then CAL requires $\Omega ( d \log d )$ queries. In contrast, for the same problem, the above query complexity of the more careful Halving++ algorithm resolves to $O ( d )$ queries.

Proposition B.1 (Query lower bound for CAL). Let $k \geq 2 .$ . There is an n-point set X in $\mathbb { R } ^ { d }$ so that the expected number of queries that CAL requires to learn any k-disjoint box partition over $\dot { X }$ is

$$
\Omega ( k d \log ( n / k d ) ) q u e r i e s .
$$

Proof. We first give a proof when $k = 2$ and $d = 1$ . Let $X = [ n ]$ . Unbeknownst to the learner, let the labeling $f ( x ) \equiv y$ be a constant map that assigns every point to the same class. This is a valid two-disjoint box partition, where one of the boxes is empty.

Let’s see how CAL works; it is also described in the proof of Theorem 1.1. Say CAL makes a sequence of queries $( q _ { t } ) _ { t }$ . At the end of round $t ,$ it can infer the labels of points only in the set

$$
X \cap { \mathrm { b b o } } \times ( q _ { 1 } , \ldots , q _ { t } ) .
$$

This is because any point x $\in X \backslash$ bbox $( q _ { 1 } , \dots , q _ { t } )$ can be split of from the queried points via a threshold; there is a valid two-disjoint box partition that is consistent with the observed labels that assigns x a diferent label. And $\mathbf { S O } ,$ in the next round, CAL chooses $q _ { t + 1 }$ uniformly at random from the points not covered by the previous bounding box, and $\mathrm { { C A L } }$ terminates only when the bounding box covers every point in X. It follows that the query complexity $Q _ { \mathrm { C A L } }$ of CAL for this problem instance is simply the amount of time it takes for the bounding box over the queries to cover X.

We can compute the expectation $\mathbb { E } [ Q _ { \mathrm { C A L } } ]$ as follows. Let $\pi : [ n ] \to X$ be a random ordering of the points in $X ,$ , where $\mathrm { C } \hat { \mathrm { A L } }$ iterates through $\pi ( s )$ and queries it if it cannot yet infer its label. Notice that CAL will query a point $i \in X$ if it has not previously a point to the left:

$$
\forall j < i , \quad \pi ^ { - 1 } ( i ) < \pi ^ { - 1 } ( j ) \qquad \Longrightarrow \qquad \mathrm { C A L } \operatorname { q u e r i e s } i .
$$

Since the ordering is random, we have

$$
\mathbb { E } [ Q _ { \mathrm { C A L } } ] \geq \sum _ { i = 1 } ^ { n } { \mathrm { P r } } ( \mathrm { C A L ~ q u e r i e s ~ } i ) \geq \sum _ { i = 1 } ^ { n } { \frac { 1 } { i } } = H ( n ) ,
$$

where $H ( n ) \geq \log ( n )$ is the nth harmonic number.

In higher dimensions, we can improve the bound by making a copy for each dimension. Define

$$
X = \bigcup _ { i = 1 } ^ { d } \big \{ \alpha e _ { i } : \alpha \in [ n / d ] \big \} .
$$

When $k = 2 ,$ let $f ( x ) \equiv y$ be constant. As in the 1-dimensional case, CAL can only infer the labels of points contained in the bounding box of its previous queries: there is always an axis-aligned halfspace splitting bbox $( q _ { 1 } , \ldots , q _ { t } )$ from any point from $X$ not contained in it. Notice that if CAL queries a point $\alpha e _ { i }$ lying along the ith dimension, the bounding box only grows in the ith dimension, and so we may treat each dimension independently. This gives a lower bound

$$
\mathbb { E } [ Q _ { \mathrm { C A L } } ] \geq d \log ( n / d ) .
$$

Finally, with general $k \geq 2$ , we can make $\left( k - 1 \right)$ translated copies of this construction, where each copy is assigned a separate label. The same argument yields the claimed lower bound

$$
\mathbb { E } [ Q _ { \mathrm { C A L } } ] \geq ( k - 1 ) d \log ( n / k d ) .
$$

□