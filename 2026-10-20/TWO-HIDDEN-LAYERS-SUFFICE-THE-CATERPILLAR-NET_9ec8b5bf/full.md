# TWO HIDDEN LAYERS SUFFICE: THE CATERPILLAR NET

Jakob Paul Zimmermann   
FU Berlin, TU Berlin   
jakob.paul.zimmermann@campus.tu-berlin.de   
https://jakob-paul-zimmermann.com

## ABSTRACT

Every continuous piecewise linear (CPWL) function is a finite signed sum of maxima of affine functions, making maximum representation a central problem in the study of the expressiveness of ReLU networks. A now-disproved conjecture predicted logarithmic depth for maxima. Inspired by the proof by Brunck et al. (2026) that two hidden layers suffice for general CPWL functions, we give the Caterpillar $N e t ,$ a nonrecursive construction for the maximum of N inputs with two hidden layers of widths $O ( N ^ { 2 } )$ and $2 ^ { O ( N \log N ) }$ . The construction has a direct geometric interpretation.<sup>1</sup>

## 1 INTRODUCTION

In this paper we investigate the representation capabilities of ReLU networks: What functions are representable by such networks of different depth and widths? As ReLU neural networks represent continuous piecewise-linear (CPWL) functions, a sharp question is: given any CPWL function, what is the ReLU network of smallest depth or smallest width representing this function? The maximum is a universal building block for such CPWL functions. Write

$$
\operatorname* { m a x } _ { N } ( x ) : = \operatorname* { m a x } _ { 0 \leq i < N } x _ { i } .\tag{1}
$$

Theorem (Wang and Sun, 2005). Let $d \geq 1$ and let $f : \mathbb { R } ^ { d } $ R be continuous and affine on each member of a finite polyhedral cover. There are $R \in \mathbb N ,$ affine maps and signs

$$
A _ { t } : \mathbb { R } ^ { d } \longrightarrow \mathbb { R } ^ { d + 1 } , \qquad \varepsilon _ { t } \in \{ - 1 , 1 \} \qquad ( 1 \leq t \leq R )\tag{2}
$$

such that

$$
f ( \boldsymbol { x } ) = \sum _ { t = 1 } ^ { R } \varepsilon _ { t } \operatorname* { m a x } _ { d + 1 } ( A _ { t } ( \boldsymbol { x } ) ) \qquad f o r e \nu e r y \boldsymbol { x } \in \mathbb { R } ^ { d } .\tag{3}
$$

Thus a two-hidden-layer representation of $\operatorname* { m a x } _ { d + 1 }$ gives one for every CPWL function on $\mathbb { R } ^ { d } ;$ precompose with the affine maps and add the networks in parallel.

Historical conjecture (Hertrich et al., 2023, Conjecture 1.5, now disproved). For every integer $k \geq 1$ , the function x 7→ max $\{ 0 , x _ { 1 } , \ldots , x _ { 2 ^ { k } } \}$ cannot be represented by a ReLU network with k hidden layers, regardless of their widths.

The case $k = 2$ asserts impossibility with two hidden layers. Bakaev et al. (2026) disproved it by constructing a two-hidden-layer representation of the maximum of five inputs.

This reduction made the geometry of the maximum a central approach to the representation problem well before the work of Brunck et al. (2026). Polyhedral approaches to depth include the Newtonpolytope framework of Hertrich et al. (2023), the simplex subdivisions of Bakaev et al. (2026), and the depth lower bounds for braid-fan-compatible networks of Grillo et al. (2025). For width, Safran (2026) used extremal graph theory to prove lower bounds for exact maximum representation, while

Lu (2026) studied first-layer width thresholds for approximation and exact representation. Brunck et al. (2026) then constructed two-hidden-layer ReLU representations for arbitrary CPWL functions in every dimension.

Theorem 1 (Brunck et al., 2026). Every continuous piecewise affine function $f : \mathbb { R } ^ { d }  \mathbb { R }$ with finitely many affine pieces can be represented exactly by a ReLU network with two hidden layers.

Our contribution is to specialize their general construction to max $\cdot N$ and simplify it into the Caterpillar Net. Taking $2 ^ { \Theta ( N ^ { 2 } ) }$ as the second-layer width baseline via applying the Brunck et al. (2026) recursive construction, we pursue two aims:

1. Give a nonrecursive construction for max<sub>N</sub> with a direct geometric interpretation.

2. Reduce its second hidden-layer width.

Write $\sigma ( t ) : = \operatorname* { m a x } ( 0 , t )$ applied coordinatewise to vectors. A network of widths $( h _ { 1 } , h _ { 2 } )$ has parameters

$$
\begin{array} { r l r } & { { W _ { 1 } } \in { { \mathbb R } ^ { { h _ { 1 } } \times { N } } } , } & { { b _ { 1 } } \in { { \mathbb R } ^ { { h _ { 1 } } } } , } \\ & { { W _ { 2 } } \in { { \mathbb R } ^ { { h _ { 2 } } \times { h _ { 1 } } } } , } & { { b _ { 2 } } \in { { \mathbb R } ^ { { h _ { 2 } } } } , } \\ & { { w _ { 3 } } \in { { \mathbb R } ^ { { h _ { 2 } } } } , } & { { b _ { 3 } } \in { { \mathbb R } } , } \end{array}\tag{4}
$$

and evaluates as the function

$$
x \longmapsto w _ { 3 } ^ { \top } \sigma \big ( W _ { 2 } \sigma ( W _ { 1 } x + b _ { 1 } ) + b _ { 2 } \big ) + b _ { 3 } .\tag{5}
$$

The output is affine, there are no skip connections, and equality is required on all of $\mathbb { R } ^ { N }$

Theorem 2 (Caterpillar Net, Lean\_1, Lean\_2, proof). For every $N \geq 3 ,$ define

$$
C _ { N } = \sum _ { m = 2 } ^ { N - 1 } \binom { N } { m } \frac { m ! } { 2 } ( N - m ) m ^ { N - m - 1 } .\tag{6}
$$

There is a two-hidden-layer ReLU network computing max<sub>N</sub> with widths

$$
h _ { 1 } = { \binom { N } { 2 } } + N + 1 , \qquad h _ { 2 } = 3 ^ { N - 2 } C _ { N } - N ( N - 1 ) + 2 .\tag{7}
$$

In particular, these widths satisfy, as $N  \infty$

$$
h _ { 1 } = O ( N ^ { 2 } ) , \qquad h _ { 2 } = 2 ^ { O ( N \log N ) } .\tag{8}
$$

Remark (small arities). The linked theorem also covers $N = 1$ and $N = 2$ by separate networks of widths (2, 2) and (4, 2). At $N = 3$ , formula (7) gives widths (7, 5).

## 2 THE CATERPILLAR NET

Two one-line identities drive the construction. For reals s, t and every linear function ℓ on $\mathbb { R } ^ { N }$

$$
\operatorname* { m a x } ( s , t ) = t + \sigma ( s - t ) ,\tag{9}
$$

$$
\sigma ( \ell ( x ) ) = \sigma ( - \ell ( x ) ) + \ell ( x ) .\tag{10}
$$

The first suggests computing the maximum along a path with attached leaves. Alternating sums of support functions of caterpillar polytopes give the correct support-function class; support functions of their proper faces, together with linear functions, correct the remaining error to pointwise equality. The second reverses an inner hinge at the cost of a linear term; it merges equal units in Section 4.

Fix $N \geq 3$ and write $x = ( x _ { 0 } , \ldots , x _ { N - 1 } ) \in \mathbb { R } ^ { N }$

Definition 3 (Labelled caterpillar, Lean\_3). A caterpillar description is a permutation–set pair

$$
\pi \in \mathrm { S y m } \{ 0 , \ldots , N - 1 \} , \qquad \varnothing \neq S \subseteq \{ 1 , \ldots , r \} , \qquad r : = N - 2 .\tag{11}
$$

Position $j$ carries the label $\pi ( j + 1 )$ , for $0 \leq j \leq r ,$ and the label $\pi ( 0 )$ is the apex. The spine positions are $\{ 0 \} \cup S$ . We also call S the cut set: a cut at $j \in S$ separates positions $j - 1$ and $j$ in the ordered list. The last spine position, and the parent of a position $j \geq 1$ , the nearest spine position before it, are

$$
q : = \operatorname* { m a x } S , \qquad p ( j ) : = \operatorname* { m a x } \bigl ( \{ 0 \} \cup \{ i \in S : i < j \} \bigr ) \qquad ( 1 \le j \le r ) .\tag{12}
$$

Joining each position $j \geq 1$ to its parent gives a directed tree on the labels $\pi ( 1 ) , \ldots , \pi ( N - 1 )$ , the labelled caterpillar: its spine is the path through the spine positions, and every other label is a leaf attached to the preceding spine vertex (Figure 1). The apex is separate from this tree.

Definition 4 (Caterpillar function and face functions, $\mathtt { L e a n \_ 4 } , \mathtt { L e a n \_ 5 } )$ . For labels i, j, for the tree edges, and for the head, put

$$
\ell _ { i j } ( x ) : = x _ { i } - x _ { j } ,\tag{13}
$$

$$
L _ { \pi , S } : = \big \{ \ell _ { \pi ( p ( j ) + 1 ) , \pi ( j + 1 ) } : 1 \le j \le r \big \} ,\tag{14}
$$

$$
h _ { \pi , S } : = \ell _ { \pi ( q + 1 ) , \pi ( 0 ) } .\tag{15}
$$

The edgefunctions $L _ { \pi , S }$ subtract each child’s coordinate from its parent’s, one for each of the r tree edges, and the head function $h _ { \pi , S }$ compares the last spine vertex with the apex. For a linear function $h ,$ a finite set L of linear functions, and a state $\tau ( \ell ) \in \{ 0 , \ell , \sigma \circ \ell \}$ chosen for every $\ell \in L$ , define

$$
U _ { h , L } ^ { \tau } ( \boldsymbol { x } ) : = \sigma \Big ( h ( \boldsymbol { x } ) + \sum _ { \ell \in L } \tau ( \ell ) ( \boldsymbol { x } ) \Big ) .\tag{16}
$$

Thus each $\ell \in L$ is omitted, kept linearly, or kept as a hinge. Write $U _ { h , L } : = U _ { h , L } ^ { \tau }$ for the all-hinge state $\tau ( \ell ) = \sigma \circ \ell .$ . We call $( h , L ) { \mathrm { ~ a ~ } }$ function description of $U _ { h , L } :$ it records the linear head and the inner hinges. The caterpillar description $( \pi , S )$ specifies the labelled tree from which such functions are generated. The caterpillarfunction of $( \pi , S )$ and itsfacefunctions are

$$
U _ { \pi , S } : = U _ { h _ { \pi , S } , L _ { \pi , S } } , \qquad U _ { \pi , S } ^ { \tau } : = U _ { h _ { \pi , S } , L _ { \pi , S } } ^ { \tau } .\tag{17}
$$

![](images/57e9148cedf22382640155040047ea54ba5b00c11353ce53829370862ebea0af.jpg)  
Figure 1: A labelled caterpillar with $N = 1 0$ and $S = \{ 3 , 6 \}$ . The spine $\pi ( 1 )  \pi ( 4 )  \pi ( 7 )$ has two leaves at each vertex; all arrows point from parent to child. The apex π(0) is separate from the tree. The last spine position is $q = 6$ , so the head function is $h _ { \pi , S } ( x ) = x _ { \pi ( 7 ) } - x _ { \pi ( 0 ) }$

The term face function comes from the following geometry. Identify linear functions with their coefficient vectors. For a polytope $P .$ , its supportfunction, and the polytope whose support function is $U _ { h , L }$ , are

$$
h _ { P } ( x ) : = \operatorname* { m a x } _ { p \in P } \langle p , x \rangle ,\tag{18}
$$

$$
Q _ { h , L } : = \mathrm { c o n v } \Big ( \{ 0 \} \cup \Big ( h + \sum _ { \ell \in L } [ 0 , \ell ] \Big ) \Big ) , \qquad U _ { h , L } = h _ { { Q } _ { h , L } } .\tag{19}
$$

Indeed, the segment $[ 0 , \ell ]$ has support function $\sigma \circ \ell ,$ , Minkowski sums add support functions, and adjoining 0 applies the outer ReLU. The head and the $N - 2$ edge functions are jointly linearly independent: their label edges form a tree on all N labels, since the head connects the apex to the last spine vertex. Thus the base is an $( N - 2 )$ -dimensional parallelotope whose affine hull does not contain 0, and $Q _ { h , L }$ is an $( N - 1 )$ -dimensional pyramid. A face of $Q _ { h , L }$ containing the apex 0 and another point is the convex hull of 0 and a face of the base. Since the base is a translated sum of segments, the face exposed by a linear functional is obtained by maximizing on each segment separately, which selects the endpoint 0, the endpoint $\ell ,$ or the whole segment. These have support functions $0 , \ell ,$ and $\sigma \circ \ell ,$ and the outer ReLU adjoins the apex. Hence the face functions (16) include the support function of every face of $Q _ { h , L }$ through the apex.

Definition 5 (Dictionary, Lean\_6). The dictionary D is the finite set of functions $\mathbb { R } ^ { N }  \mathbb { R }$ consisting of every face function $U _ { \pi , S } ^ { \tau }$ , over all caterpillar descriptions $( \pi , S )$ and all states $\tau ,$ together with the auxiliaryfunctions

$$
\begin{array} { r l r } & { x \mapsto \sigma ( x _ { i } ) , \quad x \mapsto \sigma ( - x _ { i } ) } & { ( 0 \leq i < N ) , } \\ & { x \mapsto \sigma ( x _ { i } - x _ { j } ) } & { ( 0 \leq i < j < N ) . } \end{array}\tag{20}
$$

Heads and edge functions are coordinate differences, so face functions and pair hinges are translation invariant: with $\mathbf { 1 } : = ( 1 , \ldots , 1 ) \in \mathbb { R } ^ { N }$

$$
U _ { \pi , S } ^ { \tau } ( x + t \mathbf { 1 } ) = U _ { \pi , S } ^ { \tau } ( x ) \qquad ( t \in \mathbb { R } ) .\tag{21}
$$

Only the coordinate features $\sigma ( \pm x _ { i } )$ change under $x \mapsto x + t \mathbf { 1 }$ . Theorem 7 below proves max $_ N \in$ span $\mathcal { D } ;$ the following rule then fixes one real coefficient for each dictionary function.

To turn coefficients of distinct functions into literal network data, retain the finite tags used to generate the dictionary. Let $\mathcal { A } = \mathcal { A } _ { \mathrm { f a c e } } \sqcup \mathcal { A } _ { \mathrm { a u x } } ,$ where $\mathcal { A } _ { \mathrm { f a c e } }$ contains the distinct head–edge-set pairs obtained by collecting each linear state into the head as in (88), and $\mathcal { A } _ { \mathrm { a u x } }$ contains the indexed auxiliary features in (20). If $v ( a )$ is the function represented by a tag, then

$$
v : { \mathcal { A } } \longrightarrow { \mathcal { D } } , \qquad { \mathcal { D } } = v ( { \mathcal { A } } ) .\tag{22}
$$

Different tags may represent the same function.

An exact coefficient rule (Lean\_7) Let

$$
B : = \{ S \subseteq { \mathcal { D } } : S { \mathrm { ~ i s ~ a ~ b a s i s ~ o f ~ } } \operatorname { s p a n } { \mathcal { D } } \} .\tag{23}
$$

This set is nonempty because the finite spanning set D contains a basis. For $S \in B$ , form

$$
G _ { S } : = \left( \langle g , h \rangle \right) _ { g , h \in S } ,\tag{24}
$$

$$
b _ { S } : = \left( \langle g , \operatorname* { m a x } _ { N } \rangle \right) _ { g \in S } .\tag{25}
$$

using the inner product

$$
\langle f , g \rangle : = \int _ { [ - 1 , 1 ] ^ { N } } f ( x ) g ( x ) d x .\tag{26}
$$

Every dictionary function is continuous and positively homogeneous. If a linear combination of the basis functions has zero squared integral, continuity makes it zero throughout the cube, and positive rescaling then makes it zero on all of $\mathbf { \overline { { \mathbb { R } } } } ^ { N }$ . Linear independence forces every coefficient to vanish, so $G _ { S }$ is positive definite and invertible. Write $G _ { S } ^ { - 1 }$ for its inverse matrix. For each $g \in \mathcal { D }$ , average over the bases containing g:

$$
c _ { g } : = \frac { 1 } { | \boldsymbol { \mathcal { B } } | } \sum _ { \boldsymbol { S } \in \boldsymbol { B } \atop g \in \boldsymbol { S } } ( \boldsymbol { G } _ { \boldsymbol { S } } ^ { - 1 } \boldsymbol { b } _ { \boldsymbol { S } } ) _ { g } .\tag{27}
$$

Lemma 6 (Canonical expansion, Lean\_8). Assuming Theorem $^ { 7 , }$ the coefficients (27) satisfy

$$
\operatorname* { m a x } _ { N } = \sum _ { g \in \mathcal { D } } c _ { g } g .\tag{28}
$$

Proof of Lemma 6. Fix $S \in B$ . By Theorem 7, there is $d \in \mathbb { R } ^ { S }$ such that

$$
\operatorname* { m a x } _ { N } = \sum _ { g \in S } d _ { g } g .\tag{29}
$$

Taking the inner product with each $h \in S$ gives

$$
( b _ { S } ) _ { h } = \sum _ { g \in S } \langle h , g \rangle d _ { g } = ( G _ { S } d ) _ { h } ,\tag{30}
$$

$$
d = G _ { S } ^ { - 1 } b _ { S } .
$$

Interchanging the finite sums in (27) therefore yields

$$
\sum _ { g \in \mathcal { D } } c _ { g } g = \frac { 1 } { | \mathcal { B } | } \sum _ { S \in \mathcal { B } } \sum _ { g \in S } ( G _ { S } ^ { - 1 } b _ { S } ) _ { g } g = \frac { 1 } { | \mathcal { B } | } \sum _ { S \in \mathcal { B } } \operatorname* { m a x } _ { N } = \operatorname* { m a x } _ { N } .\tag{31}
$$

This finishes the proof of Lemma 6.

For each tag, split the coefficient of its represented function equally over its nonempty finite fiber:

$$
{ \widetilde { c } } _ { a } : = { \frac { c _ { v ( a ) } } { | \{ b \in { \mathcal { A } } : v ( b ) = v ( a ) \} | } } .\tag{32}
$$

While computationally intractable to extract explicit weights for large $N _ { \ast }$ , averaging over all bases avoids arbitrary symmetry-breaking and provides a canonical coefficient map ideal for formal verification. Regrouping by the fibers of v gives the literal tagged expansion

$$
\operatorname* { m a x } _ { N } = \sum _ { a \in \mathcal { A } } \widetilde { c } _ { a } v ( a ) .\tag{33}
$$

Thus duplicate face and auxiliary tags neither duplicate nor lose any coefficient.

## 3 WHY THE DICTIONARY SPANS THE MAXIMUM

The maximum max<sub>N</sub> is the support function of conv $\{ e _ { 0 } , \ldots , e _ { N - 1 } \}$ . To work with a full dimensional polytope, let us do the following substitution: for

$$
\Delta : = \mathrm { c o n v } \{ 0 , e _ { 1 } , \ldots , e _ { N - 1 } \} \subseteq H _ { 0 } : = \{ x \in \mathbb { R } ^ { N } : x _ { 0 } = 0 \}\tag{34}
$$

we observe for an arbitrary input $\boldsymbol { x } \in \mathbb { R } ^ { N }$ that

$$
\mathrm { m a x } _ { N } ( x ) = x _ { 0 } + h _ { \Delta } ( x - x _ { 0 } { \bf 1 } ) .\tag{35}
$$

To prove that max $_ { N } \in$ span D it suffices to show that $h _ { \Delta }$ lies in the span of the restrictions of D to $H _ { 0 }$ . Indeed, $x _ { 0 } = \sigma ( x _ { 0 } ) - \sigma ( - x _ { 0 } )$ lies in the span of the auxiliary functions (20), and by (21) the substitution $x \mapsto x - x _ { 0 } \mathbf { 1 }$ 1 fixes every dictionary function except the coordinate features $\sigma ( \pm x _ { i } )$ which become $\sigma ( \pm ( x _ { i } - x _ { 0 } ) )$ ; by (10) these are linear combinations of auxiliary functions. Hence $x \mapsto g ( x - x _ { 0 } { \bf 1 } )$ belongs to span D for every $g \in \mathcal { D }$ , and (35) gives the claimed implication.

All polytopes in the geometric argument below lie in the $( N - 1 )$ -dimensional space $H _ { 0 }$ . We restrict heads and edge functions to $H _ { 0 }$ and identify them with their coefficient vectors there. Let $\mathcal { L } _ { k }$ denote the real span of support functions of polytopes in $H _ { 0 }$ of affine dimension at most k. In particular, $\mathcal { L } _ { 0 }$ is the space of linear functions, since the support function of a point p is $x \mapsto \langle p , x \rangle$

For a direction

$$
u \in H _ { 0 } ,\tag{36}
$$

the exposed face of a polytope $P \subseteq H _ { 0 }$ with normal u is the set of points of $P$ maximizing the linear function $p \mapsto \langle p , u \rangle$ :

$$
P ^ { u } : = \{ p \in P : \langle p , u \rangle = h _ { P } ( u ) \} .\tag{37}
$$

For $u \neq 0 ,$ , it is where the supporting hyperplane with outward normal u meets $P .$ The opposite face $P ^ { - u }$ consists of the minimizers; for u = $u = 0 ,$ both faces are $P$ itself. Define the function

$$
D _ { u } h _ { P } ( x ) : = h _ { P ^ { u } } ( x ) - h _ { P ^ { - u } } ( x ) \qquad ( x \in H _ { 0 } ) .\tag{38}
$$

Here u selects the two faces, while x is the input to their support functions. In particular, $D _ { 0 } = 0 .$ Geometrically, $D _ { u }$ acts as a derivative that replaces the support function of a polytope by the difference of the support functions of its opposite exposed faces. $\mathrm { I f } \ \langle u , \cdot \rangle$ is nonconstant on $P ,$ these are proper boundary faces of smaller dimension; if it is constant, the two support functions cancel. This removes the full-dimensional contribution and explains the dimension drop in (40). For a simplex and a facet normal $u , D _ { u }$ is a facet-isolating operator: it returns that facet’s support function minus the linear support function of the opposite vertex. This is the mechanism used in Step 7.2, rather than a local gradient calculation.

For a finite linear combination g of support functions, define its one-sided directional derivative at u by

$$
\begin{array} { l } { { \partial _ { u } g ( x ) : = \displaystyle \operatorname* { l i m } _ { t \downarrow 0 } \frac { g ( u + t x ) - g ( u ) } { t } , } } \\ { { \partial _ { u } h _ { P } = h _ { P ^ { u } } , \qquad D _ { u } g = \partial _ { u } g - \partial _ { - u } g . } } \end{array}\tag{39}
$$

Thus the map $h _ { P } \mapsto h _ { P ^ { 2 } }$ extends linearly to linear combinations of support functions, independently of their decomposition. If dim $P \leq k ,$ either $\langle u , \cdot \rangle$ is constant on $P _ { \mathrm { : } }$ , so $D _ { u } h _ { P } = 0$ , or both selected faces have dimension at most $k - 1$ . Consequently,

$$
D _ { u } \mathcal { L } _ { k } \subseteq \mathcal { L } _ { k - 1 } \qquad ( k \geq 1 ) .\tag{40}
$$

The signed caterpillar sum. We seek a combination that agrees with $h _ { \Delta }$ up to a function in $\mathcal { L } _ { N - 2 }$ Adding $x _ { \pi ( 0 ) }$ makes the caterpillar function the following maximum

$$
x _ { \pi ( 0 ) } + U _ { \pi , S } ( x ) = \operatorname* { m a x } \left\{ x _ { \pi ( 0 ) } , x _ { \pi ( q + 1 ) } + \sum _ { \ell \in L _ { \pi , S } } \sigma ( \ell ( x ) ) \right\} .\tag{41}
$$

Averaging over permutations gives the required symmetry

$$
F _ { N } ( x ) : = \frac { 1 } { N ! } \sum _ { \pi } \sum _ { \substack { \partial \neq S \subseteq \{ 1 , \dots , r \} } } ( - 1 ) ^ { N + | S | } \big ( x _ { \pi ( 0 ) } + U _ { \pi , S } ( x ) \big ) ,\tag{42}
$$

where the signs $( - 1 ) ^ { N + | S | }$ are chosen to cancel the extra contributions from support functions of exposed caterpillar faces, modulo $\mathcal { L } _ { N - 3 }$ . Establishing the required cancellations is a main step of the proof below.

Theorem 7 (Dictionary sufficiency, L $\tt { a n \_ 9 }$ , proof). For every $N \geq 3 ,$ , the dictionary of Definition 5 satisfies

$$
\operatorname* { m a x } _ { N } \in \operatorname { s p a n } { \mathcal { D } } .\tag{43}
$$

Proof of Theorem 7. Recall from this section’s introduction that it suffices to show that $h _ { \Delta }$ lies in the span of the restrictions of $\mathcal { D }$ to $H _ { 0 }$

Define the error by

$$
\begin{array} { r l } & { f : H _ { 0 } \longrightarrow \mathbb { R } , } \\ & { \quad f ( x ) : = F _ { N } ( x ) - h _ { \Delta } ( x ) } \\ & { \qquad = F _ { N } ( 0 , x _ { 1 } , \ldots , x _ { N - 1 } ) - \operatorname* { m a x } \{ 0 , x _ { 1 } , \ldots , x _ { N - 1 } \} . } \end{array}\tag{44}
$$

Step 7.1 reduces dictionary membership to $f \in \mathcal L _ { N - 2 }$ . Step 7.2 shows that it suffices to prove $D _ { u } f \in \mathcal { L } _ { N - 3 }$ for every direction u. Step 7.3 proves these bounds by induction on $N :$ dimension bounds and cancellation handle all directions except those with one coordinate different from all the others, which share a common value. In that case, deleting the exceptional coordinate invokes the induction hypothesis.

Step 7.1 (Reduce to a lower-dimensional error). In this step we reduce the problem to proving

$$
f = F _ { N } - h _ { \Delta } \in \mathcal { L } _ { N - 2 } .\tag{45}
$$

Assume this relation temporarily, and write $Q _ { \pi , S } : = Q _ { h _ { \pi , S } , L _ { \pi , S } } ,$ , so that $U _ { \pi , S } = h _ { Q _ { \pi , S } }$ by (19). Since the apex terms in (42) are linear, the assumed relation gives

$$
\frac { 1 } { N ! } \sum _ { \pi \atop \alpha \neq S \subseteq \{ 1 , \dots , N - 2 \} } ( - 1 ) ^ { N + | S | } h _ { Q _ { \pi , S } } - h _ { \Delta } \in \mathcal { L } _ { N - 2 } .\tag{46}
$$

We will correct this error by differentiating its support-function terms via $\partial _ { u }$ to lower the residual dimension until the residual is linear. The resulting support functions of proper faces of the $Q _ { \pi , S }$ already lie in the dictionary span: the support function of the convex hull of the apex with a base face is a face function from Definition $\bar { 4 , }$ given by (16), while the support function of a base face is a linear function plus inner hinges. Coordinate features represent the linear function, and (10) puts either orientation of each hinge in the span of (20). The apex alone has zero support function, and $h _ { Q _ { \pi , S } } = U _ { \pi , S }$ is itself a dictionary function. Differentiating the simplex support-function term also produces support functions of proper simplex faces. Their membership in the dictionary span is supplied by induction on the number of vertices.

Claim (induction on $| T | )$ . For every nonempty $T \subseteq \{ 0 , e _ { 1 } , \ldots , e _ { N - 1 } \}$ , the support function of $\Delta _ { T } : = \mathrm { c o n v } T$ lies in the dictionary span.

Start $( | T | = 1 )$ . The support function of a singleton is linear and therefore lies in the dictionary span.

Step $( | T | \geq 2 )$ . Fix $T$ and assume that $h _ { \Delta _ { A } } \in$ span D for every nonempty vertex set A with $| A | < | T |$ , in particular for every nonempty proper subset of $T ,$ . Consider a current vertex set $A \subseteq \{ 0 , e _ { 1 } , \ldots , e _ { N - 1 } \}$ with $| { \cal A } | \geq 2$ . To remove $v \in A$ , choose a direction whose value at v is strictly larger than its common value at all other vertices. The positive exposed face is then the singleton $ { \{ \boldsymbol { v } \} }$ , and the negative exposed face is the simplex on the remaining vertices. Writing $\Delta _ { A } : = \mathrm { c o n v } \tilde { A }$ , we have

$$
\begin{array} { r } { u _ { v } : = \left\{ \begin{array} { l l } { - \sum _ { i = 1 } ^ { N - 1 } e _ { i } } & { v = 0 , } \\ { v } & { v \neq 0 , } \end{array} \right. } \\ { ( \Delta _ { A } ) ^ { u _ { v } } = \{ v \} , \qquad ( \Delta _ { A } ) ^ { - u _ { v } } = \Delta _ { A \backslash \{ v \} } , } \\ { D _ { u _ { v } } h _ { \Delta _ { A } } ( x ) = \langle v , x \rangle - h _ { \Delta _ { A \backslash \{ v \} } } ( x ) . } \end{array}\tag{47}
$$

Choose an ordering $v _ { 1 } , \ldots , v _ { N - | T | }$ of the vertices outside T. Denote the error associated with the remaining vertex set T by

$$
\varepsilon _ { T } : = D _ { u _ { v _ { N - \mid T \mid } } } \cdot \cdot \cdot D _ { u _ { v _ { 1 } } } \left( \frac { 1 } { N ! } \sum _ { \substack { \pi \neq S \subseteq \{ 1 , \ldots , N - 2 \} } } ( - 1 ) ^ { N + \mid S \mid } h _ { Q _ { \pi , S } } - h _ { \Delta } \right) .\tag{48}
$$

The operators act from right to left, removing $v _ { 1 }$ , then $v _ { 2 }$ , and so on. In (47), the support function of the remaining simplex carries a minus sign, while the removed singleton contributes the linear function $x \mapsto \langle v , x \rangle$ . Every subsequent $D _ { u }$ annihilates linear functions. Thus the $N - | T |$ | deletions give

$$
D _ { u _ { v _ { N - } | T | } } \cdot \cdot \cdot D _ { u _ { v _ { 1 } } } h _ { \Delta } = ( - 1 ) ^ { N - | T | } h _ { \Delta _ { T } } + \ell _ { T } ,\tag{49}
$$

where $\ell _ { T }$ is linear. The original error contains $- h _ { \Delta }$ , so the coefficient of $h _ { \Delta _ { T } }$ in $\varepsilon _ { T } \operatorname { i s } - ( - 1 ) ^ { N - | T | } =$ $( - 1 ) ^ { N - | T | + 1 }$ . All other terms are linear functions or support functions of faces of the $Q _ { \pi , S } ,$ , since each $D _ { u }$ replaces a support function by a difference of support functions of exposed faces. The coefficient of $h _ { \Delta _ { T } }$ in $\varepsilon _ { T } \mathrm { i } \mathbf { s } \pm 1$ , and all other terms already lie in the dictionary span. Thus proving $\varepsilon _ { T } ~ \in$ span D completes the induction step. To prove this membership, we use the additional information $\varepsilon _ { T } \in \mathcal { L } _ { | T | - 2 }$ , obtained by repeated application of (40). This bound starts the inner induction below: each reduction lowers the dimension bound by one while subtracting a function in the dictionary span, until the remaining error lies in $\mathcal { L } _ { 0 }$ and is therefore linear. For the full vertex set, the operator composition is empty and means the identity, so $\varepsilon _ { T }$ is exactly the error in (46).

For this fixed T, define the remaining errors as functions $\varepsilon _ { T } ^ { ( j ) } : H _ { 0 } \to \mathbb { R }$ recursively by

$$
\begin{array} { r l } & { \varepsilon _ { T } ^ { ( 0 ) } ( x ) : = \varepsilon _ { T } ( x ) , } \\ & { \varepsilon _ { T } ^ { ( j + 1 ) } ( x ) : = \varepsilon _ { T } ^ { ( j ) } ( x ) - \displaystyle \sum _ { u \in E _ { T , j } } \bigl ( \partial _ { u } \varepsilon _ { T } ^ { ( j ) } \bigr ) ( x ) . } \end{array}\tag{50}
$$

Here $x \in H _ { 0 }$ and $0 \leq j < | T | - 2$ . At stage $j ,$ the finite set of directions $E _ { T , j } \subseteq H _ { 0 }$ is chosen from the current error by the construction below. Thus $\varepsilon _ { T } ^ { ( j ) }$ is the error left after j derivative sums have been subtracted from $\varepsilon _ { T } ;$ the superscript counts stages and is not a power.

Claim (induction on the reduction stage $j ) .$ . The sets $E _ { T , j }$ can be chosen successively so that, for $0 \leq j \leq | T | - 2$

$$
\begin{array} { c } { \varepsilon _ { T } ^ { \left( j \right) } \in \mathcal { L } _ { \left| T \right| - 2 - j } , } \\ { \varepsilon _ { T } - \varepsilon _ { T } ^ { \left( j \right) } \in \mathrm { s p a n } \mathcal { D } . } \end{array}\tag{51}
$$

Each remaining error is a linear combination of $h _ { \Delta _ { T } } ,$ , support functions of proper faces of $\Delta _ { T }$ support functions of faces of the $Q _ { \pi , S }$ , and linear functions. The second condition says that the part already removed is represented by the dictionary.

Start $( j = 0 )$ . By definition, $\varepsilon _ { T } ^ { ( 0 ) } = \varepsilon _ { T }$ . Its dimension bound and support-function expression were established above, and the removed part is zero.

Step $( j < | T | - 2 )$ . Assume the directions for the first $j$ stages have been chosen and the claim holds at stage $j .$ . Put $k : = | T | - 2 - j \geq 1$ . We require each new direction u to be nonorthogonal to the simplex direction space dir $\Delta _ { T } : = \operatorname { s p a n } \{ v - w : v , w \in T \}$ . Its values on $T$ are then not all equal, so the maximizing vertices form a nonempty proper subset:

$$
\begin{array} { r l } & { \quad A _ { u } : = \big \{ v \in T : \langle u , v \rangle = h _ { \Delta _ { T } } ( u ) \big \} , \qquad \emptyset \neq A _ { u } \subsetneq T , } \\ & { \quad \partial _ { u } h _ { \Delta _ { T } } = h _ { \Delta _ { A _ { u } } } \in \mathrm { s p a n } \mathscr { D } . } \end{array}\tag{52}
$$

The last membership uses the outer induction hypothesis, since $\vert A _ { u } \vert < \vert T \vert$ . An orthogonal direction would leave $h _ { \Delta _ { T } }$ unchanged and would not allow this use of the hypothesis.

Choose an auxiliary expression for $\varepsilon _ { T } ^ { ( j ) }$ by support functions of polytopes of dimension at most $k _ { : }$ and collect their distinct k-dimensional direction spaces. For each such space $V ,$ , choose $u _ { V } \in V ^ { \perp }$ not orthogonal to dir $\Delta _ { T }$ or to any other collected space $V ^ { \prime }$ . This is possible because dim $\Delta _ { T } =$ $| T | - 1 > k \colon$ each of dir $\Delta _ { T }$ and $\mathbf { \hat { \Sigma } } _ { V ^ { \prime } }$ contains a vector outside $V ,$ whose nonzero projection onto $\dot { V } ^ { \perp }$ is not orthogonal to that vector. Adding sufficiently small multiples of these projections avoids each orthogonality condition in turn while preserving those already avoided. Choose $E _ { T , j } : = \{ u _ { V } :$ $V$ is a collected direction space}, with $E _ { T , j } = \mathcal { O }$ if there are none. In the sum of derivatives over $E _ { T , j }$ , each term coming from a k-dimensional polytope is preserved exactly once. Every other derivative of that term is the support function of a proper face, while terms of lower dimension remain in $\mathcal { L } _ { k - 1 }$ . Subtracting this derivative sum therefore cancels all k-dimensional terms. By the recursive definition (50), the resulting error satisfies $\varepsilon _ { T } ^ { ( j + 1 ) } \in \mathcal { L } _ { k - 1 } = \mathcal { L } _ { | T | - 3 - j }$

Now compute these same derivatives using the support-function expression in the claim. The derivative of $h _ { \Delta _ { T } }$ lies in the dictionary span by (52). Differentiating the support function of an already proper simplex face still gives a support function of a proper face of $\Delta _ { T }$ , so the same induction hypothesis applies. Differentiating the support function of a face of $Q _ { \pi , S }$ gives a support function of a face of the same polytope, covered by the opening dictionary description, and linear functions remain linear. Thus the derivative sum lies in the dictionary span. $\mathrm { B y } \left( 5 0 \right) , \varepsilon _ { T } - \varepsilon _ { T } ^ { \left( j + 1 \right) }$ is the sum of the previously removed part and this derivative sum, so it also lies in the dictionary span. Subtracting the derivative sum preserves the stated types of support-function terms. This proves the claim at stage $j + 1$

At stage $j = | T | - 2$ , the remaining error lies in $\mathcal { L } _ { 0 }$ and is linear. Together with (51), this gives $\varepsilon _ { T } \in$ span D. Its vertex-deletion expression has coefficient $( - 1 ) ^ { N - | T | + 1 } \in \{ - 1 , 1 \}$ on $h _ { \Delta \prime }$ and all other terms in the dictionary span, so $h _ { \Delta _ { T } } \in$ span D. This completes the induction step on $| T |$ Taking T to be the full vertex set proves $h _ { \Delta } \in$ span $\mathcal { D }$ under the temporary assumption (45). This finishes the proof of Step 7.1.

Step 7.2 (Why the facet bounds imply the class identity). For $N \geq 4$ and $f = F _ { N } - h _ { \Delta }$ , we show that $f \in { \mathcal { L } } _ { N - 2 }$ follows from

$$
D _ { u } f \in \mathcal { L } _ { N - 3 } \qquad \mathrm { f o r e v e r y } \ u \in H _ { 0 } .\tag{53}
$$

The idea is to express $f$ modulo $\mathcal { L } _ { N - 2 }$ using support functions of full-dimensional simplices. The facet bounds then force their weighted contributions to cancel modulo $\mathcal { L } _ { N - 2 }$

The base of each $Q _ { \pi , S }$ is an affine cube. Order its edge vectors by a bijection $\omega : \{ 1 , \dots , N - 2 \} $ $L _ { \pi , S }$ , add them one at a time, and adjoin the apex:

$$
K _ { \pi , S , \omega } : = { \mathrm { c o n v } } \left( \{ 0 \} \cup \left\{ h _ { \pi , S } + \sum _ { a = 1 } ^ { k } \omega ( a ) : 0 \leq k \leq N - 2 \right\} \right) .\tag{54}
$$

These $\left( N - 2 \right) !$ simplices form the Kuhn subdivision: in cube coordinates, each occupies one ordering region $1 \geq t _ { \omega ( 1 ) } \geq \cdot \cdot \cdot \geq t _ { \omega ( N - 2 ) } \geq 0$ . For a hyperplane cut $P = P _ { + } \cup P _ { - }$ , the support-function identity

$$
h _ { P } + h _ { P _ { + } \cap P _ { - } } = h _ { P _ { + } } + h _ { P _ { - } }\tag{55}
$$

shows that subdivision changes the sum of support functions only by support functions of the cutting sections. Hence

$$
h _ { Q _ { \pi , S } } - \sum _ { \omega } h _ { K _ { \pi , S , \omega } } \in \mathcal { L } _ { N - 2 } .\tag{56}
$$

Substituting (56) into (42) expresses $f$ as a finite linear combination of support functions of $( N - 1 )$ dimensional simplices in $H _ { 0 } ,$ , plus a remainder in $\mathcal { L } _ { N - 2 }$ . The simplex $\Delta$ contributes its support function with coefficient −1, and the linear apex terms are included in the remainder.

For a simplex $P ,$ , an outward facet normal $\nu _ { F }$ exposes the facet $F$ on one side and its opposite vertex $v _ { F }$ on the other. Thus

$$
D _ { \nu _ { F } } h _ { P } ( x ) = h _ { F } ( x ) - \langle v _ { F } , x \rangle .\tag{57}
$$

Directions exposing no facet on either side give support functions of faces of dimension at most $N - 3$ . Consequently, (53) says that each weighted, signed sum of parallel-facet support functions belongs to $\mathcal { L } _ { N - 3 }$

Our assumption controls the weighted facet contributions, whereas our conclusion concerns $f .$ To recover the original sum modulo $\mathcal { L } _ { N - 2 }$ , we use the same linear operator for every simplex, so cancellations between facet contributions survive reconstruction. Define

$$
( T g ) ( x ) : = \left\{ \begin{array} { l l } { { g ( x ) } } & { { x _ { 1 } \geq 0 , } } \\ { { g ( x - x _ { 1 } e _ { 1 } ) } } & { { x _ { 1 } < 0 . } } \end{array} \right.\tag{58}
$$

The sign in this definition concerns the input x. To interpret it geometrically, first suppose that every point $p \in P$ satisfies $p _ { 1 } \geq 0$ , and write $\mathrm { p r } ( p ) : = p - p _ { 1 } e _ { 1 }$ for projection onto $p _ { 1 } = 0$ . Since $\langle p , x \rangle - \langle \mathrm { p r } ( p ) , x \rangle = p _ { 1 } x _ { 1 }$ , each point dominates its projection when $x _ { 1 } \geq 0$ , and its projection dominates when $x _ { 1 } < 0$ . Consequently,

$$
\mathcal { T } h _ { P } ( x ) = \operatorname* { m a x } \{ h _ { P } ( x ) , h _ { \mathrm { p r } P } ( x ) \} = h _ { \mathrm { c o n v } ( P \cup \mathrm { p r } P ) } ( x ) .\tag{59}
$$

This convex hull fills the segments joining points of $P$ to their projections, so its dimension is at most dim $P + 1$ . For an arbitrary $P ,$ choose $c \geq 0$ such that $P + c e _ { 1 }$ lies in $p _ { 1 } \geq 0$ . Translation adds cx to the support function, and ${ \mathcal T } ( x _ { 1 } ) = \sigma ( x _ { 1 } )$ , giving

$$
\begin{array} { r } { \mathcal { T } h _ { P } ( x ) = h _ { \mathrm { c o n v } ( ( P + c e _ { 1 } ) \cup \mathrm { p r } ( P + c e _ { 1 } ) ) } ( x ) - c \sigma ( x _ { 1 } ) . } \end{array}\tag{60}
$$

The correction $\sigma ( x _ { 1 } )$ is the support function of the segment $[ 0 , e _ { 1 } ]$ . Thus linearity gives

$$
{ \mathcal { T L } } _ { k } \subseteq { \mathcal { L } } _ { k + 1 } .\tag{61}
$$

Claim (induction on $k )$ . Every $( N - 1 )$ -dimensional simplex $P \subseteq H _ { 0 }$ with exactly k facets not parallel to $e _ { 1 }$ satisfies

$$
h _ { P } - { \mathcal { T } } \left( \sum _ { F \mathrm { f a c e t } 0 \mathrm { f } P } \mathrm { s g n } \langle \nu _ { F } , e _ { 1 } \rangle h _ { F } \right) \in { \mathcal { L } } _ { N - 2 } .\tag{62}
$$

Here $\nu _ { F }$ is an outward normal of $F .$ . Its sign against $e _ { 1 }$ records whether the facet bounds lines parallel to $e _ { 1 }$ from above or below; parallel facets contribute zero. Boundedness ensures both nonzero signs occur, so $k \geq 2$

Start $\left( k = 2 \right)$ There is one upper facet $F ^ { + }$ and one lower facet $F ^ { - }$ . Every line parallel to $e _ { 1 }$ that meets P has its two intersection endpoints on these facets. Thus $h _ { P } ( x ) = h _ { F ^ { + } } ( x )$ when $x _ { 1 } \geq 0$ and $h _ { P } ( x ) = h _ { F ^ { - } } ( x )$ when $x _ { 1 } < 0$ . When $x _ { 1 } \geq 0$ , T evaluates at $x ,$ so subtracting $\mathcal { T } ( h _ { F ^ { + } } - h _ { F ^ { - } } )$ from $h _ { P }$ leaves $h _ { F ^ { - } }$ . When $x _ { 1 } < 0 , \tau$ evaluates both facet support functions at $x - x _ { 1 } e _ { 1 }$ , where they agree because the facets have the same projection. Their difference is therefore zero, again leaving $h _ { F ^ { - } }$ . The reconstruction error is therefore $h _ { F ^ { - } } \in \mathcal { L } _ { N - 2 } ,$ , proving the claim for $k = 2 .$

Step $\left( k \ > \ 2 \right)$ . Assume the claim for $k - 1$ , and let P have k facets not parallel to $e _ { 1 }$ . Two of them have the same nonzero sign. Their intersection is an $( N - 3 )$ -dimensional face $C .$ . Let $W : = \operatorname { s p a n } \{ p - q : p , q \in C \}$ , and project orthogonally within $H _ { 0 }$ onto $W ^ { \perp }$ . This is a twodimensional plane, since dim $\dot { H _ { 0 } } = \dot { N _ { \mathrm { ~ - ~ } } } \bar { 1 }$ and dim $\mathbf { \bar { \boldsymbol { W } } } = \mathbf { \bar { \boldsymbol { N } } } - 3$ . All points of C have the same projection $c ,$ because their differences lie in $W$ . There are just two vertices of $P$ outside $C ,$ , so the projected simplex is a triangle with vertices c and the projections of those two vertices. The chosen facets project to its two sides meeting at $c .$

The chosen facet normals are orthogonal to W, so their signs against $e _ { 1 }$ equal their signs against its projection. This projected direction is nonzero, since the chosen facets are not parallel to $e _ { 1 }$ . If both sides are upper, moving from c opposite to this direction enters the triangle; if both are lower, moving in this direction enters it. Continuing meets the interior of the opposite edge. The inverse image of the line through c in this direction is a hyperplane in $H _ { 0 }$ containing $C$ and parallel to $e _ { 1 }$ . It cuts the edge joining the two vertices outside $C$ at an interior point and divides $P$ into two $( N - 1 )$ -dimensional simplices. Each keeps one of the two chosen facets and replaces the other by the new facet parallel to $e _ { 1 }$ . Its other facets retain their normal directions, so each new simplex has exactly $k - 1$ facets not parallel to $e _ { 1 }$

Write $E ( Q )$ for the reconstruction error on the left of (62). Let $P _ { + }$ and $P _ { - }$ be the two new simplices, let $R : = \dot { P } _ { + } \cap P _ { - }$ be their common cutting facet, and write $F _ { 1 } , F _ { 2 }$ for the chosen facets and $R _ { F } : = R \cap F$ for every other facet of $P _ { - }$ . With $s _ { F } : = \mathrm { s g n } \langle \nu _ { F } , e _ { 1 } \rangle$ ⟩, applying (55) to $P$ and to every split facet gives the exact residual identity

$$
E ( P ) = E ( P _ { + } ) + E ( P _ { - } ) - h _ { R } + \mathcal { T } \left( \sum _ { \stackrel { F \mathrm { f a c e t ~ o f } P } { F \notin \{ F _ { 1 } , F _ { 2 } \} } } s _ { F } h _ { R _ { F } } \right) .\tag{63}
$$

Indeed, the inherited facets retain their signs, while the two copies of $R$ are parallel to $e _ { 1 }$ and therefore have sign zero. By the induction hypothesis, $E ( P _ { + } ) , E ( P _ { - } ) \in { \mathcal { L } } _ { N - 2 } ;$ moreover, dim $R = N - 2$ and dim $R _ { F } \le N - 3$ . Equation (61) therefore puts every term on the right of (63) in $\mathcal { L } _ { N - 2 }$ . This proves the claim for k and completes the induction.

Conclusion for f. Apply (62) to every simplex support-function term in the expansion of $f ,$ keeping its original coefficient. $\mathbf { A }$ facet $F$ parallel to $e _ { 1 }$ has $\langle \nu _ { F } , e _ { 1 } \rangle = 0$ , so its term in the reconstruction formula is zero and can be omitted. For each remaining occurrence of a facet, choose $\nu _ { F }$ outward from its simplex and put

$$
u _ { F } : = \frac { \nu _ { F } } { \langle \nu _ { F } , e _ { 1 } \rangle } , \qquad \langle u _ { F } , e _ { 1 } \rangle = 1 .\tag{64}
$$

Thus the coordinate with index 1 is normalized to 1; the coordinate with index 0 remains zero because $u _ { F } \in H _ { 0 }$ . Oppositely oriented parallel normals give the same $u _ { F }$ . Only the normal is rescaled: the support function and its coefficient are unchanged.

Let U be the finite set of vectors $u _ { F }$ obtained this way. For each $u \in \mathcal { U }$ , define $G _ { u }$ by adding all facet support-function terms with $u _ { F } = u _ { \mathrm { : } }$ retaining their signs and weights. An occurrence of $F$ as a facet of $K _ { \pi , S , \omega }$ contributes $( - 1 ) ^ { N + | S | } \operatorname { s g n } \langle \nu _ { F } , e _ { 1 } \rangle h _ { F } / N !$ , while an occurrence as a facet of $\Delta$ contributes $- \mathrm { s g n } \langle \nu _ { F } , e _ { 1 } \rangle$ h<sub>F</sub>. These are precisely the terms with the same normalized normal, including all repetitions.

More explicitly, define $\rho$ to be the sum of the linear apex terms and the weighted Kuhn subdivision errors in (56). Then

$$
f = \frac { 1 } { N ! } \sum _ { \substack { \pi  \infty \sum _ { \mathcal { C } \in S \subseteq S = \{ 1 , \dots , N - 2 \} } } } ( - 1 ) ^ { N + | S | } \sum _ { \omega } h _ { K _ { \pi , S , \omega } } - h _ { \Delta } + \rho , \qquad \rho \in \mathcal { L } _ { N - 2 } .\tag{65}
$$

For a simplex $P ,$ collect its facets with normalized normal u as

$$
B _ { u } ( P ) : = \sum _ { \stackrel { F \operatorname { f a c e t } \operatorname { o f } P } { \langle \nu _ { F } , e _ { 1 } \rangle \neq 0 , \ : u _ { F } = u } } \mathrm { s g n } \langle \nu _ { F } , e _ { 1 } \rangle h _ { F } .\tag{66}
$$

An empty sum is zero. If $\langle \nu _ { F } , e _ { 1 } \rangle > 0$ , the direction $u _ { F }$ exposes $F ;$ if it is negative, $- u _ { F }$ exposes $F .$ The opposite face is a single vertex, so (57) gives the signed facet term up to a linear function. If neither u nor −u exposes a facet of $P ,$ , both exposed faces have dimension at most $N - 3$ . Thus

$D _ { u } h _ { P } - B _ { u } ( P ) \in \mathcal { L } _ { N - 3 }$ . Applying this to the full weighted expansion, with the same permutations and nonempty cut sets as in (42), gives

$$
\begin{array} { r l r } & { } & { D _ { u } f - G _ { u } = \displaystyle \sum _ { \pi , S \neq \infty } \frac { ( - 1 ) ^ { N + | S | } } { N ! } \sum _ { \omega } \big ( D _ { u } h _ { K _ { \pi , S , \omega } } - B _ { u } ( K _ { \pi , S , \omega } ) \big ) } \\ & { } & { \qquad - \left( D _ { u } h _ { \Delta } - B _ { u } ( \Delta ) \right) + D _ { u } \rho \in \mathcal { L } _ { N - 3 } \qquad ( u \in \mathcal { U } ) . } \end{array}\tag{67}
$$

Together with $( 5 3 ) .$ , relation (67) puts every $G _ { u }$ in $\mathcal { L } _ { N - 3 }$ . By linearity, the sum of the reconstructed facet terms is ${ \mathcal { T } } ( \sum _ { u \in { \mathcal { U } } } G _ { u } )$ , which lies in $\mathcal { L } _ { N - 2 }$ by (61). Its difference from $f$ consists of the reconstruction and subdivision remainders, also in $\mathcal { L } _ { N - 2 }$ . Hence $f \in { \mathcal { L } } _ { N - 2 }$ . This finishes the proof of Step 7.2.

## Step 7.3 (Prove the class identity by induction).

Claim (induction on N). The class identity (45) holds for every $N \geq 3$

Start $( N = 3 )$ . Here $r = 1$ , so the nonempty cut set $S$ must be {1}. The tree has the single edge $\pi ( 1 )  \pi ( 2 )$ ), with $q = 1$ and $p ( 1 ) = 0$ . Equation (41) gives

$$
\begin{array} { r l r } & { } & { x _ { \pi ( 0 ) } + U _ { \pi , \{ 1 \} } ( x ) = \operatorname* { m a x } \bigl \{ x _ { \pi ( 0 ) } , x _ { \pi ( 2 ) } + \sigma ( x _ { \pi ( 1 ) } - x _ { \pi ( 2 ) } ) \bigr \} } \\ & { } & { ~ = \operatorname* { m a x } \bigl \{ x _ { \pi ( 0 ) } , x _ { \pi ( 1 ) } , x _ { \pi ( 2 ) } \bigr \} = \operatorname* { m a x } _ { 3 } ( x ) . } \end{array}\tag{68}
$$

Each term has coefficient $( - 1 ) ^ { 3 + 1 } / 3 ! = 1 / 6$ . For each apex label there are two orders of the remaining labels, and both give max . Grouping the six positive terms by their apex gives

$$
F _ { 3 } ( x ) = { \frac { 1 } { 6 } } \left( \underbrace { 2 \operatorname* { m a x } _ { 3 } ( x ) } _ { \pi ( 0 ) = 0 } + \underbrace { 2 \operatorname* { m a x } _ { 3 } ( x ) } _ { \pi ( 0 ) = 1 } + \underbrace { 2 \operatorname* { m a x } _ { 3 } ( x ) } _ { \pi ( 0 ) = 2 } \right) = \operatorname* { m a x } _ { 3 } ( x ) .\tag{69}
$$

On $H _ { 0 } , x _ { 0 } = 0$ and $\mathrm { m a x } _ { 3 } ( x ) = \mathrm { m a x } \{ 0 , x _ { 1 } , x _ { 2 } \} = h _ { \Delta } ( x )$ . Hence $f = 0 \in \mathcal { L } _ { 1 }$ , proving (45) for $N = 3$

Step $( N \geq 4 )$ . Assume the class identity for $N - 1$ . By Step 7.2, it suffices to prove $D _ { u } f \in \mathcal { L } _ { N - 3 }$ for every $u \in H _ { 0 }$ . By (38), applying $D _ { u }$ replaces each support function $h _ { P }$ by $h _ { P ^ { u } } - h _ { P ^ { - u } } ; $ we select the maximizing and minimizing faces in direction u and subtract their support functions. The linear apex terms disappear. For $\Delta$ , these faces contain exactly the vertices with largest and smallest coordinate values $u _ { i }$ , including $u _ { 0 } = 0$ . For a caterpillar base, the segment $[ 0 , \ell _ { i j } ]$ remains free when $u _ { i } = u _ { j } ;$ its maximizing endpoint is $\ell _ { i j }$ when $u _ { i } > u _ { j }$ and 0 when $u _ { i } < u _ { j }$ . The minimizing face uses the opposite choices.

We discard support-function contributions already in $\mathcal { L } _ { N - 3 }$ and cancel the remaining ones. Induction is needed only when one of $u _ { 0 } , \ldots , u _ { N - 1 }$ differs from the common value of all the others: the exposed simplex facet and the surviving caterpillar sum then give the same problem with $N - 1$ inputs.

Discarding contributions in $\mathcal { L } _ { N - 3 } .$ . Fix $u \neq 0$ , since $D _ { 0 } f = 0 . \mathrm { ~ A ~ }$ face of $Q _ { \pi , S }$ contained in its base has a support function that is a linear function plus support functions of segments, hence belongs to $\mathcal { L } _ { 1 } \subseteq \mathcal { L } _ { N - 3 } ;$ the apex alone contributes zero. If at least two edges join labels with unequal $u _ { i } ,$ at most $N - 4$ segments remain free, so even adjoining the apex gives dimension at most $\bar { N } - 3$ Thus we need only faces containing the apex and an exposed base face, with at most one edge joining unequal values.

Removing the unique unequal edge splits the tree into two components on which the $u _ { i }$ are constant. Treat the selected exposed sign as positive, replacing u $ { \mathbf { b } } \mathbf { y } - u$ for the negative face and renaming the levels. The condition that the apex and exposed base tie gives

$$
\begin{array} { r l r } & { u _ { \pi ( 0 ) } \in \{ c , d \} } & { \mathrm { f o r ~ a ~ s p i n e ~ e d g e ~ j o i n i n g ~ l e v e l s ~ } c , d , } \\ & { u _ { \pi ( 0 ) } = c + \sigma ( c - d ) \in \{ c , 2 c - d \} } & { \mathrm { f o r ~ a ~ l e a f ~ o f ~ l e v e l ~ } d \mathrm { ~ a t ~ l e v e l ~ } c . } \end{array}\tag{70}
$$

Indeed, the exposed base value is max $\{ c , d \}$ in the spine case and $c + \sigma ( c - d )$ in the leaf case. With no unequal edge all tree values agree. Consequently, only two distinct coordinate values or the three values $c + t , c - t , c , \ldots , c$ with $t \neq 0$ can contribute. This exhausts both exposed signs; all other directions are already handled by the dimension bound.

Cancellation for three distinct values. For $c + t , c - t , c , \ldots , c$ , let $R$ be the reduced linear map induced by exchanging the two exceptional simplex vertices and subtracting the new value at vertex 0. It is a rank-one reflection: for a linear functional b with $\langle b , u \rangle = 1$ , and a linear function ℓ, permutation symmetry gives

$$
{ \cal R } x = x - 2 \langle b , x \rangle u , { \cal R } u = - u , { \cal F } _ { N } ( { \cal R } x ) = { \cal F } _ { N } ( x ) + \ell ( x ) .\tag{71}
$$

Positive homogeneity then compares the one-sided derivatives explicitly:

$$
D _ { u } ( F _ { N } | _ { H _ { 0 } } ) ( x ) = 2 F _ { N } ( - u ) \langle b , x \rangle - \ell ( x ) .\tag{72}
$$

Thus

$$
D _ { u } ( F _ { N } | _ { H _ { 0 } } ) \in \mathcal { L } _ { 0 } .\tag{73}
$$

The two exposed simplex faces are vertices, so their support functions are linear as well.

Cancellation for two values, each repeated. After the preceding dimension bound, every term still requiring cancellation has a unique split $u _ { \pi ( 1 ) } = \cdot \cdot \cdot = u _ { \pi ( k ) } \neq u _ { \pi ( k + 1 ) } = \cdot \cdot \cdot = u _ { \pi ( r + 1 ) }$ with $k \in S$ . If the apex label has the latter value, toggle membership of $k - 1$ in S. Otherwise, exchange the equal-valued labels $\pi ( k + 1 ) , \pi ( k + 2 )$ and toggle membership of $k + 1$ . Repetition of both values gives $k \geq 2$ in the first case and $k + 2 \leq r + 1$ in the second, so the operations are defined; the unchanged cut at k keeps the cut set nonempty. The transformed term remains in the same case, and repeating the operation returns the original term. If Φ denotes this involution on the active terms and $w _ { \pi , S } : = ( - 1 ) ^ { N + | S | } / N !$ , then

$$
w _ { \Phi ( \pi , S ) } = - w _ { \pi , S } , \qquad D _ { u } h _ { Q _ { \pi , S } } - D _ { u } h _ { Q _ { \Phi ( \pi , S ) } } \in { \mathcal L } _ { 1 } .\tag{74}
$$

Inactive fixed points of the global transformation are already in $\mathcal { L } _ { N - 3 }$ by the dimension bound. Summing the active pairs therefore puts $D _ { u } F _ { N }$ in $\mathcal { L } _ { N - 3 }$ . Each exposed simplex face has at most $N - 2$ vertices and hence dimension at most $N - 3$ , so $D _ { u } f \in \mathcal { L } _ { N - 3 }$ without using induction.

One exceptional coordinate: the induction hypothesis. We are still working in $H _ { 0 } ,$ so $u _ { 0 } = x _ { 0 } =$ 0. It remains to consider two values, one occurring only once. Choose a permutation τ such that $\tau ( 0 )$ is the exceptional label. Relabeling must be followed by subtracting the new zeroth coordinate to stay in $H _ { 0 } { \mathrm { : } }$

$$
B _ { \tau } : H _ { 0 } \longrightarrow H _ { 0 } , \qquad ( B _ { \tau } x ) _ { i } : = x _ { \tau ( i ) } - x _ { \tau ( 0 ) } \qquad ( 0 \le i < N ) .\tag{75}
$$

Its zeroth coordinate is zero, and its inverse is $B _ { \tau ^ { - 1 } }$ . It sends u to $( 0 , t , \ldots , t )$ with $t \neq 0$ . This change preserves every $\mathcal { L } _ { k }$ and changes $f$ only by a linear term, annihilated by D; we transform both the input and the direction. Rescaling the direction, and reversing the sign of $D _ { u }$ if necessary, reduces to $u = ( 0 , 1 , \ldots , 1 )$ within $H _ { 0 }$

This direction exposes the $\left( N - 2 \right)$ -dimensional facet $\Delta ^ { \prime } : = \operatorname { c o n v } \{ e _ { 1 } , \ldots , e _ { N - 1 } \}$ and the opposite vertex 0. Hence

$$
D _ { u } h _ { \Delta } ( x ) = h _ { \Delta ^ { \prime } } ( x ) = \operatorname* { m a x } \{ x _ { 1 } , \ldots , x _ { N - 1 } \} \qquad ( x \in H _ { 0 } ) .\tag{76}
$$

Here is the exact use of the induction hypothesis. Every caterpillar maximum increases by a common added constant. Averaging over the $( { \bar { N } } - 1 ) !$ ! lower label orders cancels the factorial in the total signed coefficient, giving

$$
\begin{array} { r l } {  { W _ { N - 1 } : = \sum _ { \substack { \emptyset \neq T \subseteq \{ 1 , \dots , N - 3 \} } } ( - 1 ) ^ { N - 1 + | T | } , } } \\ & { F _ { N - 1 } ( x _ { 1 } , \dots , x _ { N - 1 } ) - \operatorname* { m a x } _ { 1 \leq i \leq N - 1 } x _ { i } } \\ & { = F _ { N - 1 } ( 0 , x _ { 2 } - x _ { 1 } , \dots , x _ { N - 1 } - x _ { 1 } ) } \\ & { - \operatorname* { m a x } \{ 0 , x _ { 2 } - x _ { 1 } , \dots , x _ { N - 1 } - x _ { 1 } \} + ( W _ { N - 1 } - 1 ) x _ { 1 } . } \end{array}\tag{77}
$$

The remaining $N - 1$ inputs need not start with zero. Subtracting $x _ { 1 }$ puts them in the lower-arity slice $\{ z \in \mathbb { R } ^ { N - 1 } : z _ { 0 } = 0 \}$ , exactly the domain of the induction hypothesis. Apply the $( N - 1 ) \mathrm { \bar { - i n p u t } }$ class identity to $( 0 , x _ { 2 } - x _ { 1 } , \ldots , x _ { N - 1 } - x _ { 1 } )$ on the right. Linear precomposition preserves $\mathcal { L } _ { N - 3 }$ and the last term is linear. Since $( N - 1 ) - \dot { 2 } = N - 3$ , this gives

$$
( x \mapsto F _ { N - 1 } ( x _ { 1 } , \dots , x _ { N - 1 } ) - \operatorname* { m a x } \{ x _ { 1 } , \dots , x _ { N - 1 } \} ) \in { \mathcal { L } } _ { N - 3 } .\tag{78}
$$

Thus we must match the exposed caterpillar support functions with the lower-arity sum. We prove

$$
D _ { ( 0 , 1 , \ldots , 1 ) } ( F _ { N } | _ { H _ { 0 } } ) - \left( x \mapsto F _ { N - 1 } ( x _ { 1 } , \ldots , x _ { N - 1 } ) \right) \in { \mathcal { L } } _ { N - 3 } .\tag{79}
$$

To identify the contributing terms, put $p : = \pi ^ { - 1 } ( 0 ) \in \{ 0 , \dots , N - 1 \}$ . Deleting this position gives an order π¯ that maps positions $\{ 0 , \ldots , N - 2 \}$ bijectively to labels $\{ \bar { 1 } , \ldots , N - 1 \}$ . Modulo $\mathcal { L } _ { N - 3 }$ it is enough to retain precisely the cases

$$
( p = 1 { \mathrm { ~ a n d ~ } } 1 \in S ) , \qquad { \mathrm { o r } } \qquad p = j + 2 { \mathrm { ~ a n d ~ } } ( j + 1 \notin S { \mathrm { ~ o r ~ } } j = N - 3 ) , \quad 0 \leq j \leq N - 3 .\tag{80}
$$

The omitted cases have a base-only face, an apex alone, or an apex-base face with too few free segments, and therefore lie individually in $\mathcal { L } _ { N - 3 }$ . For $0 \leq i \leq N - 3$ , let $\iota _ { i }$ insert a gap at the prospective pendant position $i + 1$ . Given a lower cut set $T \subseteq \{ 1 , \dots , N - 3 \}$ , the three active insertion rules are

$$
\begin{array} { r l } { \iota _ { i } ( t ) : = \left\{ \begin{array} { l l } { t , } & { t < i + 1 , } \\ { t + 1 , } & { t \geq i + 1 , } \end{array} \right. } \\ { \mathrm { f i r s t ~ e n d : ~ } } & { ( p , S ) = \left( 1 , \left\{ 1 \right\} \cup \left\{ t + 1 : t \in T \right\} \right) , } \\ { \mathrm { p e n d a n t ~ } i : } & { ( p , S ) = \left( i + 2 , \left\{ \iota _ { i } ( t ) : t \in T \right\} \right) \qquad ( 0 \leq i \leq N - 3 ) , } \\ { \mathrm { l a s t ~ e n d : ~ } } & { ( p , S ) = \left( N - 1 , T \cup \left\{ N - 2 \right\} \right) . } \end{array}\tag{81}
$$

In every row the upper permutation is uniquely obtained by inserting label 0 at position $p$ in $\bar { \pi } .$ Conversely, with t restricted to $\{ 1 , \ldots , N - 3 \}$ , the deletion rules are $T = \{ t : t + 1 \in S \}$ for the first end, $\dot { T } = \{ t : \iota _ { i } ( t ) \in S \}$ for pendant i, and $T = S \setminus \{ N - 2 \}$ for the last end. These operations are inverse. Criterion $( 8 0 )$ assigns every active index uniquely to the first row, to the pendant row when $j + 1 \not \in S$ , or to the last row when $j = N - 3$ and $N - 2 \in S$ . Thus a nonempty $T$ has $N - 2$ pendant and two endpoint preimages, whereas $T = \emptyset$ has only the two endpoints because upper cut sets are nonempty.

Extend the lower caterpillar notation to $T = \emptyset$ by setting its last spine position and every parent equal to 0, so its head begins at the root $\bar { \pi } ( 1 )$ . For every lower label order $\bar { \pi }$ put

$$
\begin{array} { r l r } & { } & { C _ { \bar { \pi } , T } ( x ) : = x _ { \bar { \pi } ( 0 ) } + U _ { \bar { \pi } , T } ( x ) , } \\ & { } & { q _ { T } : = \operatorname* { m a x } ( \{ 0 \} \cup T ) , \qquad p _ { T } ( j ) : = \operatorname* { m a x } ( \{ 0 \} \cup \{ t \in T : t < j \} ) . } \end{array}\tag{82}
$$

For each upper preimage $( \pi , S )$ of $( \bar { \pi } , T )$ , the insertion identities and the literal factorial coefficients are

$$
\begin{array} { r l } { \mathrm { p e n d a n t : } } & { \displaystyle \frac { ( - 1 ) ^ { N + | S | } } { N ! } = - \frac { 1 } { N } \frac { ( - 1 ) ^ { N - 1 + | T | } } { ( N - 1 ) ! } , \quad D _ { u } \big ( x _ { \pi ( 0 ) } + U _ { \pi , S } \big ) + C _ { \bar { \pi } , T } \in \mathcal { L } _ { 1 } , } \\ { \mathrm { e n d p o i n t : } } & { \displaystyle \frac { ( - 1 ) ^ { N + | S | } } { N ! } = \frac { 1 } { N } \frac { ( - 1 ) ^ { N - 1 + | T | } } { ( N - 1 ) ! } , \qquad D _ { u } \big ( x _ { \pi ( 0 ) } + U _ { \pi , S } \big ) - C _ { \bar { \pi } , T } \in \mathcal { L } _ { 1 } . } \end{array}\tag{83}
$$

Thus every one of the $N - 2 + 2 = N$ preimages contributes $1 / N$ of the lower summand modulo $\mathcal { L } _ { N - 3 }$ , and together they recover its coefficient in $F _ { N - 1 }$

The extra terms are the two endpoint fibres whose deletion leaves $T = \emptyset$ , hence a rooted star. $\mathrm { B y } \left( 8 3 \right)$ each contributes $1 / N$ of the weighted lower summand modulo $\mathcal { L } _ { 1 }$ . For a fixed lower order π¯, the nonroot positions are $J = \{ 0 , 2 , \stackrel { - } { \dots } , N - 2 \}$ , including the original apex position 0. For $a \in J$ , let $\bar { \pi } ^ { ( a ) }$ exchange positions 0 and $^ { a , }$ keeping the root at position 1; the exchange at $a = 0$ is the identity. Put

$$
\begin{array} { r c l } { { q : = | J | = N - 2 , \quad } } & { { d _ { a } : = x _ { \bar { \pi } ( 1 ) } - x _ { \bar { \pi } ( a ) } \quad ( a \in J ) , } } \\ { { } } & { { } } & { { } } \\ { { T _ { A } : = \sigma \left( \displaystyle \sum _ { a \in A } d _ { a } + \displaystyle \sum _ { a \in J \setminus A } \sigma ( d _ { a } ) \right) \quad ( A \subseteq J ) , } } \\ { { } } & { { } } & { { } } \\ { { C _ { \bar { \pi } ^ { ( a ) } , \otimes } = x _ { \bar { \pi } ( a ) } + T _ { \{ a \} } \quad ( a \in J ) . } } \end{array}\tag{84}
$$

The Boolean alternating identity gives

$$
\sum _ { A \subseteq J } ( - 1 ) ^ { | A | } T _ { A } = 0 \quad \Longrightarrow \quad \sum _ { a \in J } T _ { \{ a \} } = T _ { \emptyset } + \sum _ { A \subseteq J } ( - 1 ) ^ { | A | } T _ { A } .\tag{85}
$$

The empty term is a sum of support functions of segments. For $| { \cal A } | \ge 2$ , the polytope with support function $T _ { A }$ has one head direction and $q \mathrm { ~ - ~ } | A |$ segment directions, hence dimension at most $1 + q - | A | \le q - 1 = N - 3$ . Thus the singleton sum lies in $\mathcal { L } _ { N - 3 } ;$ the sum of the linear apex coordinates lies in $\mathcal { L } _ { 0 }$ . Each fixed-position exchange is a bijection on lower label orders, so

$$
q \sum _ { \bar { \pi } } C _ { \bar { \pi } , \mathcal { O } } = \sum _ { \bar { \pi } } \sum _ { a \in J } C _ { \bar { \pi } ^ { \left( a \right) } , \mathcal { O } } \in \mathcal { L } _ { q - 1 } .\tag{86}
$$

Since $q \ > \ 0 ,$ , the full empty-cut sum belongs to $\mathcal { L } _ { q - 1 } ~ = ~ \mathcal { L } _ { N - 3 }$ . The empty-cut coefficient $( - 1 ) ^ { N - 1 } / ( N - 1 ) !$ and the factor $2 / N$ from the two endpoint fibres are scalars, so the extra star sum lies there as well. This proves $( 7 9 )$ .

Combining (76), (78), and (79) now gives

$$
\begin{array} { r l } & { D _ { u } f ( x ) = \left[ D _ { u } ( F _ { N } | _ { H _ { 0 } } ) ( x ) - F _ { N - 1 } ( x _ { 1 } , \dots , x _ { N - 1 } ) \right] } \\ & { \qquad + \left[ F _ { N - 1 } ( x _ { 1 } , \dots , x _ { N - 1 } ) - \operatorname* { m a x } \{ x _ { 1 } , \dots , x _ { N - 1 } \} \right] . } \end{array}\tag{87}
$$

The first bracket belongs to $\mathcal { L } _ { N - 3 }$ by coordinate deletion, the second by the induction hypothesis. This proves the bound for every direction. Step 7.2 gives $f \in \mathcal L _ { N - 2 } $ , completing the induction, and Step 7.1 gives the dictionary representation. This finishes the proof of Theorem 7. □

## 4 NETWORK SIZE

We prove Theorem 2 directly. First we bound the number of normalized caterpillar face descriptions.   
Then we construct one shared first layer and account for both widths.

Proof of Theorem 2. Fix $N \geq 3$ . Theorem 7 and Lemma 6, followed by (33), give the exact tagged expansion of max .

Step 2.1 (Count the caterpillar face functions). We first rewrite each face function using only its adjusted head and the hinges that remain. This lets us recognize when different face descriptions require the same second-layer neuron. An edge kept linearly is added to the head, an omitted edge disappears, and an edge kept as a hinge stays in the set M. Thus

$$
U _ { h , L } ^ { \tau } = U _ { h ^ { \prime } , M } , \qquad h ^ { \prime } : = h + \sum _ { \tau ( \ell ) = \ell } \ell , \qquad M : = \{ \ell \in L : \tau ( \ell ) = \sigma \circ \ell \} .\tag{88}
$$

The resulting function description is $( h ^ { \prime } , M )$ in the sense of Definition 4: $h ^ { \prime }$ is the adjusted head, and M contains precisely the retained hinges. When two resulting pairs have the same adjusted head and the same hinge set, they can use one second-layer neuron, with their output coefficients added. Distinct pairs may still represent the same function, so counting these pairs gives an upper bound. The shared first layer is constructed in the next step. We next orient the remaining hinges consistently. By (10), applied to each $\ell _ { i j } \in M$ with $i > j$ , the same face function $U _ { h ^ { \prime } , M }$ has the normalized description $\bar { ( } \bar { h } ^ { + } , M ^ { + } )$ , where

$$
\begin{array} { l } { { \displaystyle M ^ { + } : = \{ \ell _ { \operatorname* { m i n } ( i , j ) , \operatorname* { m a x } ( i , j ) } : \ell _ { i j } \in M \} } , } \\ { { \displaystyle h ^ { + } : = h ^ { \prime } + \sum _ { \ell _ { i j } \in M , \ i > j } \ell _ { i j } . } } \end{array}\tag{89}
$$

Every remaining hinge now has orientation $x _ { i } - x _ { j }$ with $i < j$ , and the additional linear terms have been absorbed into $\bar { h ^ { + } }$ . No edge function of a caterpillar is the negative of another one, so M and $M ^ { + }$ have the same size.

For a caterpillar description $( \pi , S )$ , the spine endpoints are labelled $\pi ( 1 )$ and $\pi ( q + 1 )$ , and every label other than $\pi ( 1 )$ and the apex is the child of exactly one edge. Hence the normalized description of $U _ { \pi , S }$ is

$$
\begin{array} { r l } & { { h } _ { \pi , S } ^ { + } ( x ) = \displaystyle \sum _ { \ell _ { i j } \in L _ { \pi , S } } x _ { \operatorname* { m a x } ( i , j ) } + x _ { \pi ( 1 ) } + x _ { \pi ( q + 1 ) } - \sum _ { i = 0 } ^ { N - 1 } x _ { i } , } \\ & { \quad \quad L _ { \pi , S } ^ { + } = \{ \ell _ { \operatorname* { m i n } ( i , j ) , \operatorname* { m a x } ( i , j ) } : \ell _ { i j } \in L _ { \pi , S } \} . } \end{array}\tag{90}
$$

It depends only on the undirected tree, the apex, and the unordered pair of spine endpoints, not on the order of the leaves at a spine vertex or on the orientation of the spine. For a spine with m vertices, choose its vertex set in $\binom { \dot { N } } { m }$ ways, its order up to reversal in m! $! / 2$ ways, its apex in $N - m$ ways, and the attachment of each of its $N - m - 1$ leaves in m ways. Reversal has no fixed point since $m \geq 2$ Every caterpillar has such data, so by (6)

$$
\left| \{ ( h _ { \pi , S } ^ { + } , L _ { \pi , S } ^ { + } ) : ( \pi , S ) \mathrm { a c a t e r p i l l a r d e s c r i p t i o n } \} \right| \le C _ { N } .\tag{91}
$$

Reversing an edge function swaps its omitted and linear states, so normalizing before or after choosing the states gives the same descriptions. Every face tag therefore has a normalized description obtained from $( h _ { \pi , S } ^ { + } , L _ { \pi , S } ^ { + } )$ by choosing one of three states for each of its $N - 2$ edge functions. Let $\kappa _ { 0 }$ be the set of these normalized descriptions and let $\nu : \mathcal { A } _ { \mathrm { f a c e } }  \mathcal { K } _ { 0 }$ be normalization. For $k = ( h , M ) \in \mathcal { K } _ { 0 }$ write $U _ { k } : = U _ { h , M }$ . With (91), $| { \cal K } _ { 0 } | \le 3 ^ { N - 2 } C _ { N }$ . Only equal normalized descriptions are merged; distinct members of $\kappa _ { 0 }$ are not identified merely because they represent the same function.

Among them are the pair-hinge descriptions $( \ell _ { i j } , \mathcal { O } )$ with $i \neq j \colon$ for the all-cut path $S = \{ 1 , \ldots , r \}$ with apex $\pi ( 0 ) = j$ and last spine vertex $\pi ( r + 1 ) = i .$ , omitting every edge function gives

$$
U _ { \pi , S } ^ { 0 } = \sigma \circ \ell _ { i j } \qquad ( 0 \leq i \neq j < N ) .\tag{92}
$$

The linear heads $\ell _ { i j }$ are distinct for distinct ordered pairs. Their pair-hinge functions are distinct as well: equality of two such ReLUs makes their positive half-spaces equal, so their heads are positive scalar multiples, and equality on that half-space makes the scalar one. Let $\mathcal { P }$ be the subset of $\kappa _ { 0 }$ consisting of every normalized description with empty hinge set and head equal to a coordinate difference; these are exactly the keys eligible for absorption into the first-layer remainder. The descriptions above show that $| \mathcal { P } | \ge \operatorname { \bar { N } } ( N - 1 )$ ). Aggregate the tagged coefficients only along equal normalized face descriptions:

$$
d _ { k } : = \sum _ { \stackrel { a \in \mathcal { A } _ { \mathrm { f a c e } } } { \nu ( a ) = k } } \widetilde { c } _ { a } .\tag{93}
$$

Discard zero fiber sums, then remove the absorbable keys:

$$
\begin{array} { r } { \mathcal { K } _ { \mathrm { a c t } } : = \{ k \in \mathcal { K } _ { 0 } : d _ { k } \neq 0 \} . } \end{array}\tag{94}
$$

$$
K : = \mathcal { K } _ { \mathrm { a c t } } \setminus \mathcal { P } .\tag{95}
$$

By (10), every unit indexed by $\mathcal { P }$ lies in the span of the auxiliary functions and needs no second-layer unit. Since $\dot { K } \subseteq \mathcal { K } _ { 0 } \setminus \mathcal { P }$

$$
| K | \le 3 ^ { N - 2 } C _ { N } - N ( N - 1 ) .\tag{96}
$$

This finishes the proof of Step 2.1.

Step 2.2 (Compile the network and establish its widths). The terms not assigned a retained neuron form the explicitly known remainder

$$
R : = \sum _ { a \in \mathcal { A } _ { \mathrm { a u x } } } \widetilde { c } _ { a } v ( a ) + \sum _ { k \in { \mathcal K } _ { \mathrm { a c t } } \cap { \mathcal P } } d _ { k } U _ { k } .\tag{97}
$$

By (33) and (93)–(95), max $\begin{array} { r } { \mathbf { \sigma } _ { N } = \sum _ { k \in K } d _ { k } U _ { k } + R } \end{array}$ . Expand the absorbed pair hinges using (10) to write $R$ as a linear combination of the auxiliary functions (20). The first layer outputs the vector

$$
\Phi ( x ) : = \big ( \sigma ( x _ { 0 } ) , \sigma ( - x _ { 0 } ) , ( \sigma ( x _ { i } - x _ { 0 } ) ) _ { i > 0 } , ( \sigma ( x _ { i } - x _ { j } ) ) _ { i < j } \big ) .\tag{98}
$$

Thus $W _ { 1 }$ has rows $e _ { 0 } ^ { \top }$ and $- e _ { 0 } ^ { \top }$ , then $e _ { i } ^ { \top } - e _ { 0 } ^ { \top }$ for $i > 0 ,$ , then $e _ { i } ^ { \top } - e _ { j } ^ { \top }$ for $i < j$ , using a fixed ordering within each block. Its row count is

$$
{ \binom { N } { 2 } } + ( N - 1 ) + 2 = { \binom { N } { 2 } } + N + 1 = h _ { 1 } .\tag{99}
$$

By (10), these features recover all coordinates,

$$
\begin{array} { l l } { x _ { 0 } = \sigma ( x _ { 0 } ) - \sigma ( - x _ { 0 } ) , } \\ { x _ { i } = x _ { 0 } + \sigma ( x _ { i } - x _ { 0 } ) - \sigma ( x _ { 0 } - x _ { i } ) \quad } & { ( i > 0 ) . } \end{array}\tag{100}
$$

For each $k = ( h , M ) \in K$ , substitute (100) into $\begin{array} { r } { h ( x ) + \sum _ { \ell \in M } \sigma ( \ell ( x ) ) } \end{array}$ and collect the coefficients of $\Phi ( x )$ into a row vector $\boldsymbol { w _ { k } ^ { \intercal } }$ . Every hinge in M already has the orientation $i < j$ , so its coefficient is added directly to the corresponding feature.

To specify the remaining row, collect the auxiliary coefficients of R as

$$
R ( x ) = \sum _ { i } p _ { i } \sigma ( x _ { i } ) + \sum _ { i } n _ { i } \sigma ( - x _ { i } ) + \sum _ { i < j } s _ { i j } \sigma ( x _ { i } - x _ { j } ) .\tag{101}
$$

Each retained unit is translation invariant by (21), so

$$
R ( x + t 1 ) = R ( x ) + t , \qquad \sum _ { i } p _ { i } = 1 .\tag{102}
$$

The coefficient sum follows by setting $x = 0$ and $t = 1$ . Now choose t large enough that every $x _ { i } + t$ is nonnegative. Substituting the auxiliary expansion into (102) cancels t and gives

$$
R ( x ) = \sum _ { i } p _ { i } x _ { i } + \sum _ { i < j } s _ { i j } \sigma ( x _ { i } - x _ { j } ) .\tag{103}
$$

Substitute (100) into this expression and collect the coefficients of $\Phi ( x )$ into $w _ { R }$ . Thus both $w _ { k }$ and $w _ { R }$ are specified by finite sums of the previously defined coefficients.

For any fixed ordering of $K$ , the Caterpillar Net has second-layer rows and output weights

$$
W _ { 2 } = \left( \begin{array} { c } { { ( w _ { k } ^ { \top } ) _ { k \in K } } } \\ { { w _ { R } ^ { \top } } } \\ { { - w _ { R } ^ { \top } } } \end{array} \right) , \qquad w _ { 3 } = \left( \begin{array} { c } { { ( d _ { k } ) _ { k \in K } } } \\ { { 1 } } \\ { { - 1 } } \end{array} \right) , \qquad b _ { 1 } = b _ { 2 } = b _ { 3 } = 0 .\tag{104}
$$

The first $| K |$ neurons output $U _ { k } ( x )$ , and the last two output $\sigma ( R ( x ) )$ and $\sigma ( - R ( x ) )$ . Thus

$$
w _ { 3 } ^ { \top } \sigma ( W _ { 2 } \Phi ( x ) ) = \sum _ { k \in { \cal K } } d _ { k } U _ { k } ( x ) + \sigma ( R ( x ) ) - \sigma ( - R ( x ) ) = \operatorname* { m a x } _ { N } ( x ) .\tag{105}
$$

By (96), the second-layer width is at most

$$
| K | + 2 \le 3 ^ { N - 2 } C _ { N } - N ( N - 1 ) + 2 = h _ { 2 } .\tag{106}
$$

Zero padding realizes the stated widths exactly.

For the growth rates, count the ordered choices of m spine labels:

$$
{ \binom { N } { m } } { \frac { m ! } { 2 } } = { \frac { N ! } { 2 ( N - m ) ! } } \leq N ^ { m } \qquad ( 2 \leq m \leq N - 1 ) .\tag{107}
$$

Using $N - m \leq N$ and $m ^ { N - m - 1 } \leq N ^ { N - m - 1 }$ in each summand of (6) gives

$$
\begin{array} { l } { { \displaystyle C _ { N } \leq \sum _ { m = 2 } ^ { N - 1 } N ^ { m } N N ^ { N - m - 1 } = ( N - 2 ) N ^ { N } } } \\ { { \displaystyle \leq N ^ { N + 1 } \leq N ^ { 4 N } . } } \end{array}\tag{108}
$$

Thus no Stirling estimate is needed. Since $3 ^ { N - 2 } \leq N ^ { N }$ for $N \geq 3 ,$ , the widths satisfy

$$
h _ { 2 } \leq 3 ^ { N - 2 } ( N - 2 ) N ^ { N } + 2 \leq 2 N ^ { 5 N } ,\tag{109}
$$

$$
h _ { 1 } \leq 4 N ^ { 2 } .\tag{110}
$$

Taking logarithms gives (8). This finishes the proof of Theorem 2.

Formalization (Lean\_10) Linked definitions, lemmas, and theorems are audited against exact types; only propext, Classical.choice, and Quot.sound are allowed. The repository contains the maximum construction and its proof. The general CPWL results in the introduction are cited background results; the equation links in the Wang–Sun theorem lead to its published source. Claude and Codex assisted with formalization and manuscript preparation.

## 5 CONCLUSION

The first hidden-layer width is asymptotically sharp. Building on the extremal-graph approach of Safran (2026), our separate Lean development extends the bound $h _ { 1 } \geq { \binom { N } { 2 } } - \lfloor 4 N / 3 \rfloor$ of Lu (2026, Theorem 2.13), for $\bar { N } \geq 4 ,$ , to networks with arbitrary affine skip connections. When every first-layer weight vector is proportional to some $e _ { i } - e _ { j }$ , we sharpen it to $h _ { 1 } \geq { \binom { N } { 2 } } - N$ . We obtain no growing lower bound for $h _ { 2 } ;$ finding smaller exact maximum representations and establishing such lower bounds remain open.

## REFERENCES

Egor Bakaev, Florestan Brunck, Christoph Hertrich, Jack Stade, and Amir Yehudayoff. Better neural network expressivity: Subdividing the simplex. In Proceedings ofthe 58th Annual ACM Symposium on Theory ofComputing, pages 500–507, 2026. doi:10.1145/3798129.3800768.

Florestan Brunck, Jack Stade, and Amir Yehudayoff. Computing ma $\mathrm { x } _ { n }$ in Depth 2—Proof Overview. Unpublished draft supplied to the author, October 2026.

Moritz Grillo, Christoph Hertrich, and Georg Loho. Depth-bounds for neural networks via the braid arrangement. Advances in Neural Information Processing Systems, 38:127800–127818, 2025. doi:10.52202/085713-4257.

Christoph Hertrich, Amitabh Basu, Marco Di Summa, and Martin Skutella. Towards lower bounds on the depth of ReLU neural networks. SIAM Journal on Discrete Mathematics, 37(2):997–1029, 2023. doi:10.1137/22M1489332.

Shunqi Lu. Sharp closure thresholds for two-hidden-layer ReLU networks. Preprint, version 1, September 2026. doi:10.5281/zenodo.22728502. Author’s PDF.

Itay Safran. A depth hierarchy for computing the maximum in ReLU networks via extremal graph theory. In Proceedings of the 39th Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 5721–5742, 2026. PMLR 336.

Shuning Wang and Xusheng Sun. Generalization of hinging hyperplanes. IEEE Transactions on Information Theory, 51(12):4425–4431, 2005. doi:10.1109/TIT.2005.859246.