# THE MINIMAL WIDTH OF NEURAL NETWORKS AND IMMERSIONS OF REAL PROJECTIVE SPACE

BAOZHI CHU AND BEN WEINKOVE

Abstract. We investigate the minimal width of deep neural networks that can approximate all continuous functions from $\mathbb { R } ^ { n } \mathrm { ~ t o ~ } \mathbb { R } ^ { m }$ in the topology of uniform convergence on compact subsets. We show that whenever the real projective space $\mathbb { R } \mathbb { P } ^ { n - 1 }$ admits a smooth immersion into $\mathbb { R } ^ { n }$ , the minimal width is at least n + min $\{ n , m \}$ , assuming that the activation function is continuous and monotone. As a consequence, the minimal width of ReLU networks is n+m whenever n is equal to 2, 3, 4, 7 or 8 and $1 \leq m \leq n ,$ , extending results of Hanin-Sellke and Hwang. This suggests an intriguing connection between the approximation theory of neural networks and diferential topology.

## 1. Introduction

A fundamental problem concerning expressivity of neural networks is to understand what architectures can approximate arbitrary continuous functions.

We define a neural network $F : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ to be a map of the form

$$
\begin{array} { r } { F = A ^ { ( L ) } \circ \sigma \circ A ^ { ( L - 1 ) } \circ \cdots \circ \sigma \circ A ^ { ( 1 ) } , } \end{array}
$$

where $A ^ { ( \ell ) } : \mathbb { R } ^ { d _ { \ell - 1 } }  \mathbb { R } ^ { d _ { \ell } }$ are afine maps, L is a positive integer called the depth and

$$
n = d _ { 0 } , d _ { 1 } , \dots , d _ { L - 1 } , d _ { L } = m
$$

are the dimensions of F. The map $\sigma : \mathbb { R } $ R is a continuous activation function, which is extended to arbitrary dimensions $\sigma : \mathbb { R } ^ { k }  \mathbb { R } ^ { k }$ by $\sigma ( x _ { 1 } , \ldots , x _ { k } ) = ( \sigma ( x _ { 1 } ) , \ldots , \sigma ( x _ { k } ) )$ The architecture of $F$ consists of the dimensions $d _ { 0 } , \ldots , d _ { L }$ and the choice of activation $\sigma .$ . The width of $F$ is

$$
\operatorname { w i d t h } ( F ) : = \operatorname* { m a x } _ { 0 \leq \ell \leq L } d _ { \ell } .
$$

A now classical result is the universal approximation theorem for shallow networks. This states that networks of depth 2 (shallow networks) and arbitrarily large width can approximate any continuous function $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ uniformly on compact subsets, as long as the activation function is not a polynomial [2, 3, 5, 12].

On the other hand, narrow deep networks, which have bounded width but arbitrarily large depth, have powerful approximation properties (see [17, 18], for example). A natural question that arises is the minimal width problem. Given an activation $\sigma ,$ we ask for the least integer $w$ such that any continuous function $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ can be approximated uniformly on compact subsets by networks of arbitrary depth and width at most $w .$ . Surprisingly, for most values of n and $m ,$ and standard activations, this minimal width, which we denote by $w _ { \mathrm { m i n } } ( n , m )$ , remains unknown. Here and throughout the paper, we set $w _ { \mathrm { m i n } } ( n , m ) = \infty \mathrm { i f }$ the minimal width does not exist.

For the ReLU activation,

$$
\sigma ( x ) = \operatorname* { m a x } \{ x , 0 \} ,
$$

Hanin–Sellke [4] showed that

$$
w _ { \mathrm { m i n } } ( n , m ) \leq n + m ,\tag{1.1}
$$

and equality holds when the output dimension m is 1:

$$
w _ { \mathrm { m i n } } ( n , 1 ) = n + 1 .
$$

When the output dimension is strictly larger than 1, the space of maps $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ is much richer from a geometric-topological point of view. When $m = 2$ , Park–Yun–Lee– Shin [14] showed that equality holds in (1.1) for $n = 1$ , namely

$$
w _ { \mathrm { m i n } } ( 1 , 2 ) = 3 .
$$

Combining results of Kim–Min–Park [11, Theorem 3] and Hanin-Sellke [4, Theorem 1], this can be generalized to

$$
w _ { \operatorname* { m i n } } ( n , m ) \geq \operatorname* { m a x } \{ n , m \} + 1 , \quad \mathrm { i f } \ m \leq 2 n .
$$

Hwang [8] proved that

$$
w _ { \mathrm { m i n } } ( 2 , 2 ) = 4 .\tag{1.2}
$$

The question of minimal width has also been extensively studied under diferent hypotheses on the activation function and the approximation space (for example, using $L ^ { p }$ instead of $C ^ { 0 }$ norms). We refer the reader to [1, 9, 10, 11, 13, 14, 15, 16] and the references therein.

The crux of Hwang’s argument for (1.2) is a two-to-one map from the unit circle $S ^ { 1 }$ to itself, which “wraps around twice”. More precisely, one defines a function $f : A \to \mathbb { R } ^ { 2 }$ for an annulus A in $\mathbb { R } ^ { 2 }$ containing the unit circle $S ^ { 1 }$ , with the property that f maps $S ^ { 1 }$ onto $S ^ { 1 }$ with winding number 2. Any neural network approximation to f of width at most 3 would give rise to an injective map from a simple closed curve in $\mathbb { R } ^ { 2 }$ to the cylinder $S ^ { 1 } \times \mathbb { R } \subset \mathbb { R } ^ { 3 }$ which, after projection to $S ^ { 1 }$ , would wrap around twice. No such injective map exists and hence the width of the network must be at least 4.

Our main result extends Hwang’s geometric-topological argument to higher dimensions. The map $x \mapsto - x$ sending a point on the sphere $S ^ { n - 1 }$ to its antipodal point defines a two-to-one quotient map from $S ^ { n - 1 }$ to real projective space $\mathbb { R } \mathbb { P } ^ { n - 1 }$ , the set of lines through the origin in $\mathbb { R } ^ { n }$ . Assuming there is a smooth immersion from RP<sup>n−1</sup> to $\mathbb { R } ^ { n }$ , the composition gives a smooth immersion from $S ^ { n - 1 }$ onto its image in $\mathbb { R } ^ { n }$ This map, extended to an annulus in $\mathbb { R } ^ { n }$ , plays the role of the function $f$ in Hwang’s argument.

We prove the following minimal width lower bound for continuous and monotone activations, assuming the existence of smooth immersions of $\mathbb { R } \mathbb { P } ^ { n - 1 }$ into $\mathbb { R } ^ { n }$ . We say that $\sigma : \mathbb { R } $ R is monotone if it is either non-increasing on R or non-decreasing on R (for example, constant functions are monotone).

Theorem 1.1. Assume the activation function σ is continuous and monotone. Let $n \geq 2$ and $m \geq 1$ . Assume that there is a smooth immersion $i : \mathbb { R P } ^ { n - 1 } \to \mathbb { R } ^ { n }$ Then there exist a compactly supported smooth map $f _ { * } : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ , a compact annulus $A \subset \mathbb { R } ^ { n }$ , and $\varepsilon _ { 0 } > 0$ such that the following holds. For every network $F : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ of width at most $n + \operatorname* { m i n } \{ n , m \} - 1$ , we have

$$
\operatorname* { s u p } _ { A } | f _ { * } - F | \geq \varepsilon _ { 0 } .
$$

In particular,

$$
w _ { \mathrm { m i n } } ( n , m ) \geq n + \operatorname* { m i n } \{ n , m \} .
$$

Combining this bound with (1.1) and a well-known classification of the dimensions n for which $\mathbb { R } \mathbb { P } ^ { n - 1 }$ admits a smooth immersion into $\mathbb { R } ^ { n }$ , we obtain the following.

Corollary 1.1. For the ReLU activation,

$$
w _ { \mathrm { m i n } } ( n , m ) = n + m \quad i f n \in \{ 2 , 3 , 4 , 7 , 8 \} \ a n d \ 1 \leq m \leq n .
$$

The upper bound (1.1) extends easily from the ReLU activation to the Leaky ReLU activations defined by $\sigma ( x ) = \operatorname* { m a x } \{ x , a x \}$ , for a fixed $a \in ( 0 , 1 )$ . Therefore, the conclusion of Corollary 1.1 also holds for Leaky ReLU activations.

We do not expect that the formula $w _ { \mathrm { m i n } } ( n , m ) = n + m$ holds for all values of $n ,$ m with $1 \leq m \leq n$ , although we cannot yet provide a counterexample. On the other hand, it is known that if $m > 2 n \geq 4$ then $w _ { \mathrm { m i n } } ( n , m ) = n + m$ fails [8, Theorem 4.8]. We believe that this question is connected to diferential topology in a non-trivial way, and deserves further study.

The paper is organized as follows. In $\ S 2$ , we record a needed homotopy theorem. The construction of the map $f$ and its key property Theorem 3.1 are established in $\ S 3$ , while a more explicit and shorter construction for the $n = 2$ case is provided in $\ S 4$ . Theorem 1.1 is proved in $\ S 5$ , and Corollary 1.1 is proved in $\ S 6$

AI Disclosure. The authors used ChatGPT to help discover the main results. All arguments were independently checked by the authors and the exposition is entirely their own.

## 2. A persistence theorem for zeros under odd homotopies

We will make use of the following result on the persistence of zeros under odd homotopies.

Theorem 2.1. Let M be a smooth compact n-manifold with boundary. Let $\tau : M \to M$ be a smooth free involution $( i . e$ . a smooth map with $\tau ^ { 2 } = \mathrm { I d }$ and with no points q with $\tau ( q ) = q )$ . Let $F : M \times \lbrack 0 , 1 \rbrack  \rbrack \mathbb { R } ^ { n }$ be a smooth map and define $F _ { t } : M \to \mathbb { R } ^ { n }$ by $F _ { t } ( x ) : = F ( x , t )$ . Assume the following:

(1) $F _ { 0 }$ has exactly two zeros on $M$ , at p and $\tau ( p )$ , and the derivative of $F _ { 0 }$ at $p$ is invertible.

(2) $F _ { t } ( \tau ( x ) ) = - F _ { t } ( x )$ for all $t \in [ 0 , 1 ]$ . (Namely, $F _ { t }$ is an odd homotopy).

(3) $F _ { t } | _ { \partial M }$ is nowhere zero, for all $t \in [ 0 , 1 ]$

Then $F _ { 1 }$ has a zero.

To prove it, we will require a standard intersection mod 2 result (see [7], Theorem 2.1 and the discussion on pages 132-133).

Theorem 2.2. Let X be a smooth 2n-manifold with boundary and $N \subset X$ be a compact n-dimensional submanifold such that $\partial N = N \cap \partial X$ and N meets ∂X transversely. Let $h _ { 0 } , h _ { 1 } : N \to X$ be smooth maps transverse to N, joined by a smooth homotopy $h _ { t } : N \to X f o r t \in [ 0 , 1 ]$ such that $h _ { t } ( \partial N ) \subset \partial X \setminus \partial N$ . Then $h _ { 0 } ^ { - 1 } ( N )$ and $h _ { 1 } ^ { - 1 } ( N )$ have the same number of points mod 2.

Theorem 2.1 then follows easily.

Proof of Theorem 2.1. Define

$$
X = ( M \times \mathbb { R } ^ { n } ) / \sim , \quad { \mathrm { f o r ~ } } ( x , y ) \sim ( \tau ( x ) , - y ) ,
$$

and

$$
N = \{ [ x , 0 ] \in X \mid x \in M \} .
$$

Then X is a smooth 2n-manifold with boundary and N is a compact submanifold of dimension n with boundary $\partial N = N \cap \partial X$ . Moreover, N meets ∂X transversely. For $t \in [ 0 , 1 ]$ , define $h _ { t } : N \to X$ by

$$
h _ { t } ( [ x , 0 ] ) = [ x , F _ { t } ( x ) ] ,
$$

which is well-defined by (2). The condition (3) implies that $h _ { t } ( \partial N ) \subset \partial X \setminus \partial N$ . By (1), the set $h _ { 0 } ^ { - 1 } ( N )$ is the singleton $\{ [ p , 0 ] \}$ and the intersection is transverse because $D F _ { 0 } ( p )$ is invertible. If $F _ { 1 }$ did not have a zero then $h _ { 1 } ^ { - 1 } ( N )$ would be empty, and $h _ { 1 }$ would be transverse to N. But then Theorem 2.2 would imply that, mod 2,

$$
1 = \# h _ { 0 } ^ { - 1 } ( N ) = \# h _ { 1 } ^ { - 1 } ( N ) = 0 ,
$$

a contradiction.

## 3. Construction of the map

For a small $\delta > 0$ , define the compact annulus $A = A ( \delta )$ in $\mathbb { R } ^ { n }$ by

$$
\begin{array} { r l } & { A : = \{ y \in \mathbb { R } ^ { n } \mid d ( y , S ^ { n - 1 } ) \leq \delta \} } \\ & { \quad = \{ ( 1 + t ) x \in \mathbb { R } ^ { n } \mid x \in S ^ { n - 1 } , - \delta \leq t \leq \delta \} , } \end{array}
$$

where d denotes distance in the Euclidean norm. We prove the following key result. (Since the proof is somewhat technical, the reader may wish to first read a simpler proof in the case $n = 2$ which is presented in Section 4.)

Theorem 3.1. Assume there is a smooth immersion $i : \mathbb { R P } ^ { n - 1 } \to \mathbb { R } ^ { n }$ . Then there exist $\delta , \varepsilon > 0$ and a smooth map $f : A  \mathbb { R } ^ { n }$ such that the following holds. If $g : A  \mathbb { R } ^ { n }$ and $h : A \to \mathbb { R } ^ { n - 1 }$ are smooth maps, and g satisfies

$$
\operatorname* { s u p } _ { A } | f - g | \leq \varepsilon ,
$$

then $G : A \to \mathbb { R } ^ { 2 n - 1 }$ given by $G : = ( g , h )$ is not injective.

Proof. Let

$$
\pi : S ^ { n - 1 } \to \mathbb { R P } ^ { n - 1 } = S ^ { n - 1 } / \sim ,
$$

be the quotient map $\pi ( x ) = [ x ]$ , where $x \sim y$ for $x , y \in S ^ { n - 1 }$ if x and y are equal or are antipodal points $( { \mathrm { n a m e l y } } , x = - y )$ . Define

$$
b = i \circ \pi : S ^ { n - 1 } \to \mathbb { R } ^ { n } ,
$$

which is again a smooth immersion, since π is a local difeomorphism. Observe that $b ( - x ) = b ( x )$

Next we will define a smooth map $\nu : S ^ { n - 1 } \to S ^ { n - 1 }$ which takes a point $x \in S ^ { n - 1 }$ and returns $\nu ( x )$ , a unit normal vector to $b _ { * } ( T _ { x } S ^ { n - 1 } )$ at the point $b ( x )$

We first set up some notation. We define the orientation of a basis of vectors $v _ { 1 } , \ldots , v _ { n }$ of $\mathbb { R } ^ { n }$ to be

$$
\left[ v _ { 1 } , \ldots , v _ { n } \right] = { \left\{ \begin{array} { l l } { 1 , \qquad \operatorname* { d e t } ( v _ { 1 } , \ldots , v _ { n } ) > 0 } \\ { - 1 , \qquad \operatorname* { d e t } ( v _ { 1 } , \ldots , v _ { n } ) < 0 , } \end{array} \right. }
$$

where $( v _ { 1 } , \ldots , v _ { n } )$ is the $n \times n$ matrix whose column vectors are $v _ { i }$ . Note that replacing one of the $v _ { i } \ \mathrm { b y } \ - v _ { i }$ changes the sign of the orientation. If $e _ { 1 } , \ldots , e _ { n }$ are the standard basis vectors of $\mathbb { R } ^ { n }$ then $[ e _ { 1 } , \ldots , e _ { n } ] = 1$

Let $v _ { 1 } , \ldots , v _ { n - 1 }$ be basis vectors of $T _ { x } S ^ { n - 1 }$ such that $[ v _ { 1 } , \dotsc , v _ { n - 1 } , x ] = 1$ . Note that x is normal to $T _ { x } S ^ { n - 1 }$ so $v _ { 1 } , \ldots , v _ { n - 1 } ,$ x gives a basis of $\mathbb { R } ^ { n }$ . Define $\nu ( x )$ to be the unique unit normal vector to $b _ { * } ( T _ { x } S ^ { n - 1 } )$ in $\mathbb { R } ^ { n }$ so that

$$
[ D b | _ { x } ( v _ { 1 } ) , \ldots , D b | _ { x } ( v _ { n - 1 } ) , \nu ( x ) ] = 1 .
$$

Note that since b is an immersion, $b _ { * } ( T _ { x } S ^ { n - 1 } )$ is a hyperplane in $\mathbb { R } ^ { n }$ spanned by the vectors $D b | _ { x } ( v _ { 1 } ) , \dots , D b | _ { x } ( v _ { n - 1 } )$ . There are exactly two unit normal vectors to $b _ { * } ( T _ { x } S ^ { n - 1 } ) ;$ the orientation determines which one we pick. The definition of ν is independent of the particular choice of $v _ { 1 } , \ldots , v _ { n - 1 } ;$ it only requires that $[ v _ { 1 } , \dotsc , v _ { n - 1 } , x ] = 1$

We need an elementary lemma.

Lemma 3.1. For all $x \in S ^ { n - 1 }$ we have $\nu ( - x ) = ( - 1 ) ^ { n } \nu ( x )$

Proof of Lemma 3.1. Fix $x \in S ^ { n - 1 }$ and let $v _ { 1 } , \ldots , v _ { n - 1 }$ be a basis of $T _ { x } S ^ { n - 1 }$ with $[ v _ { 1 } , \dotsc , v _ { n - 1 } , x ] = 1$ . By definition of $\nu ( x )$ , we have

$$
[ D b | _ { x } ( v _ { 1 } ) , \ldots , D b | _ { x } ( v _ { n - 1 } ) , \nu ( x ) ] = 1 .
$$

On the other hand, $v _ { 1 } , \dotsc , v _ { n - 2 } , - v _ { n - 1 }$ is a basis of $T _ { - x } S ^ { n - 1 }$ with

$$
[ v _ { 1 } , \ldots , v _ { n - 2 } , - v _ { n - 1 } , - x ] = 1 .
$$

$$
\begin{array} { c l } { { \mathrm { S i n c e ~ } b ( - x ) = b ( x ) \mathrm { ~ w e ~ h a v e ~ } D b | _ { - x } = - D b | x . \mathrm { ~ H e n c e } } } \\ { { } } & { { [ D b | _ { - x } ( v _ { 1 } ) , \ldots , D b | _ { - x } ( v _ { n - 2 } ) , D b | _ { - x } ( - v _ { n - 1 } ) , ( - 1 ) ^ { n } \nu ( x ) ] } } \\ { { } } & { { = ( - 1 ) ^ { n - 2 + n } [ D b | _ { x } ( v _ { 1 } ) , \ldots , D b | _ { x } ( v _ { n - 1 } ) , \nu ( x ) ] } } \\ { { } } & { { = 1 , } } \end{array}
$$

and $\nu ( - x ) = ( - 1 ) ^ { n } \nu ( x )$ as required.

Let $\delta > 0$ be a small constant to be determined later. We now define $f : A  \mathbb { R } ^ { n }$ by

$$
\begin{array} { r } { f ( ( 1 + t ) x ) = b ( x ) + t \nu ( x ) , \quad x \in S ^ { n - 1 } , \ - \delta \leq t \leq \delta . } \end{array}\tag{3.1}
$$

We claim that the derivative $D f | _ { x }$ of $f$ at every $x \in S ^ { n - 1 }$ is invertible. Indeed, without loss of generality, assume that $\boldsymbol { x } \ = \ e _ { n } .$ , the North pole of $S ^ { n - 1 }$ and use coordinates $x _ { 1 } , \ldots , x _ { n - 1 }$ in a neighborhood $U \subset S ^ { n - 1 }$ of x. Then $e _ { 1 } = \partial _ { x _ { 1 } } , \ldots , e _ { n - 1 } = \partial _ { x _ { n - 1 } }$ span $T _ { x } S ^ { n - 1 }$ . Compute

$$
\begin{array} { l } { { D f | _ { x } ( e _ { i } ) = D b | _ { x } ( e _ { i } ) , \quad i = 1 , \ldots , n - 1 , } } \\ { { \qquad D f | _ { x } ( e _ { n } ) = \nu ( e _ { n } ) . } } \end{array}\tag{3.2}
$$

Since b is an immersion, the vectors $D b | _ { x } ( e _ { 1 } ) , \dots , D b | _ { x } ( e _ { n - 1 } )$ span the hyperplane $b _ { * } T _ { x } S ^ { n - 1 } \subset \mathbb { R } ^ { n }$ and by definition $\nu ( e _ { n } )$ is orthogonal to the hyperplane. Hence $D f | _ { x }$ is invertible, completing the claim. Assuming that $\delta > 0$ is suficiently small, it follows that $f$ is a local difeomorphism.

Assume first that n is even. Then

$$
f ( - y ) = f ( y ) , \quad { \mathrm { f o r ~ a l l ~ } } y \in A ,\tag{3.3}
$$

since $b ( - x ) = b ( x )$ and $\nu ( - x ) = \nu ( x )$

Define a compact 2n − 1-manifold M by $M = S ^ { n - 1 } \times \overline { { B } } _ { \delta }$ where $B _ { \delta }$ is the open ball of radius $\delta$ centered at the origin in $\mathbb { R } ^ { n }$ . The manifold M has a boundary $S ^ { n - 1 } \times \partial B _ { \delta }$ Define a smooth free involution $\tau : M \to M$ given by

$$
\tau ( x , w ) = ( - x , w ) .
$$

Observe that if $( x , w ) \in M$ then $x + w$ and $- x + w$ lie in the set A.

Define a map $F _ { 0 } : M \to \mathbb { R } ^ { 2 n - 1 }$ by

$$
F _ { 0 } ( x , w ) = ( x _ { 1 } , \ldots , x _ { n - 1 } , f ( x + w ) - f ( - x + w ) ) ,
$$

where $x _ { 1 } , \ldots , x _ { n - 1 }$ are the first $n - 1$ components of $x = ( x _ { 1 } , \ldots , x _ { n } )$

We first show that $F _ { 0 }$ satisfies assumption (1) of Theorem 2.1. First note that for fixed x and small w, using (3.3) and Taylor’s Theorem,

$$
f ( x + w ) - f ( - x + w ) = f ( x + w ) - f ( x - w ) = 2 D f | _ { x } ( w ) + O ( | w | ^ { 2 } ) .\tag{3.4}
$$

If $F _ { 0 } ( x , w ) = 0$ then $x _ { 1 } = \cdot \cdot \cdot = x _ { n - 1 } = 0$ which implies that $x = e _ { n } { \mathrm { o r } } - e _ { n }$ . The points $( e _ { n } , 0 )$ and $( - e _ { n } , 0 )$ are zeros of $F _ { 0 }$ in M. Moreover, the derivative of $F _ { 0 }$ at $( \pm e _ { n } , 0 )$ 2 using $x _ { 1 } , \ldots , x _ { n - 1 }$ as local coordinates on $S ^ { n - 1 }$ near $\pm e _ { n } ,$ is

$$
( D F _ { 0 } ) _ { ( \pm e _ { n } , 0 ) } = \left( \begin{array} { c c } { { I _ { n - 1 } } } & { { 0 } } \\ { { 0 } } & { { 2 D f _ { \pm e _ { n } } } } \end{array} \right) ,
$$

which is invertible since $D f _ { \pm e _ { n } }$ is invertible. This implies that there exists an open neighborhood U of $( e _ { n } , 0 )$ such that $F _ { 0 }$ is non-zero on $U \setminus \{ ( e _ { n } , 0 ) \}$ , and similarly for $( - e _ { n } , 0 )$ . Hence, shrinking $\delta > 0$ if necessary, $F _ { 0 }$ has exactly two zeros $( e _ { n } , 0 )$ and $\tau ( e _ { n } , 0 )$ and the derivative of $F _ { 0 }$ at these zeros is invertible.

Next, define $F _ { 1 } : M \to \mathbb { R } ^ { 2 n - 1 }$ by

$$
F _ { 1 } ( x , w ) = ( h ( x + w ) - h ( - x + w ) , g ( x + w ) - g ( - x + w ) )
$$

and $F : M \times [ 0 , 1 ] \to \mathbb { R } ^ { 2 n - 1 } { \mathrm { ~ b y ~ } } F ( x , t ) = F _ { t } ( x ) : = ( 1 - t ) F _ { 0 } ( x ) + t F _ { 1 } ( x )$ . We have

$$
F _ { t } ( \tau ( x , w ) ) = F _ { t } ( - x , w ) = - F _ { t } ( x , w ) ,
$$

so (2) is satisfied.

Finally, for (3), write $\pi _ { 2 } : \mathbb { R } ^ { n - 1 } \times \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ for the projection $\pi _ { 2 } ( x , y ) = y$ onto the second factor. There exists a constant $c > 0$ such that

$$
| \pi _ { 2 } \circ F _ { 0 } ( x , w ) | = | f ( x + w ) - f ( - x + w ) | \geq c ,
$$

for $( x , w ) \in \partial M$ , since we already showed that $f ( x + w ) = f ( - x + w )$ only when $w = 0 ,$ and $F _ { 0 }$ is continuous on the compact boundary ∂M. Since $f$ and g are uniformly close on $A$ we may choose $\varepsilon > 0$ suficiently small, depending on $\delta ,$ such that for any $( x , w ) \in M$

$$
| \pi _ { 2 } \circ F _ { 1 } ( x , w ) - \pi _ { 2 } \circ F _ { 0 } ( x , w ) | = | f ( x + w ) - f ( - x + w ) - g ( x + w ) + g ( - x + w ) | < c / 2 .
$$

Since

$$
\begin{array} { r l } & { { \pi _ { 2 } } \circ F _ { t } ( x , w ) = ( 1 - t ) { \pi _ { 2 } } \circ F _ { 0 } ( x , w ) + t { \pi _ { 2 } } \circ F _ { 1 } ( x , w ) } \\ & { \qquad = { \pi _ { 2 } } \circ F _ { 0 } ( x , w ) + t ( { \pi _ { 2 } } \circ F _ { 1 } ( x , w ) - { \pi _ { 2 } } \circ F _ { 0 } ( x , w ) ) , } \end{array}
$$

we have for $( x , w ) \in \partial M$

$$
| F _ { t } ( x , w ) | \geq | \pi _ { 2 } \circ F _ { t } ( x , w ) | \geq c / 2 ,
$$

and this proves (3).

By Theorem 2.1, $F _ { 1 }$ has a zero, namely a point $( x , w )$ in M such that

$$
g ( x + w ) = g ( - x + w ) , \quad h ( x + w ) = h ( - x + w ) .
$$

Hence $G ( x + w ) = G ( - x + w )$ with $x + w$ and $- x + w$ being distinct points in A. This completes the proof in the case of n even.

Now assume n is odd. Rather than repeat the entire proof, we merely indicate the necessary changes. Crucially, we define for $x \in S ^ { n - 1 }$ and $w \in \mathbb { R } ^ { n }$ the map $R _ { x } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ by

$$
\begin{array} { r } { R _ { x } ( w ) = w - 2 ( x \cdot w ) x . } \end{array}
$$

This satisfies $R _ { x } w = w$ if w is perpendicular to $x ,$ and $R _ { x } x = - x$ , so that the component of w which is in the x direction gets reflected. Observe that $R _ { - x } = R _ { x }$ and $R _ { x } ^ { \bar { 2 } } = \mathrm { I d }$ We define the involution $\tau : M \to M$ by

$$
\tau ( x , w ) = ( - x , R _ { x } w ) ,
$$

noting that $\tau ^ { 2 } ( x , w ) = ( x , R _ { - x } R _ { x } w ) = ( x , w )$ as required.

${ \mathrm { I f } } \left( x , w \right) \in M$ then x+w and $- x + R _ { x } w$ lie in the set A. Define a map $F _ { 0 } : M \to \mathbb { R } ^ { 2 n - 1 }$ by

$$
F _ { 0 } ( x , w ) = ( x _ { 1 } , \ldots , x _ { n - 1 } , f ( x + w ) - f ( - x + R _ { x } w ) ) ,
$$

where $x _ { 1 } , \ldots , x _ { n - 1 }$ are the first $n - 1$ components of $x = ( x _ { 1 } , \ldots , x _ { n } )$

Taking the Taylor expansion in w at a fixed $x \in S ^ { n - 1 }$ gives

$$
f ( x + w ) - f ( - x + R _ { x } w ) = D f | _ { x } w - ( D f | _ { - x } \circ R _ { x } ) ( w ) + O ( | w | ^ { 2 } ) .\tag{3.5}
$$

We claim that for all $x \in S ^ { n - 1 }$ ，

$$
\begin{array} { r } { D f | _ { - x } \circ R _ { x } = - D f | _ { x } . } \end{array}\tag{3.6}
$$

Indeed, assume without loss of generality that $\boldsymbol { x } = \boldsymbol { e } _ { n }$ and use coordinates $x _ { 1 } , \ldots , x _ { n - 1 }$ in neighborhoods $U ^ { + } , U ^ { - } \subset S ^ { n - 1 }$ of x and −x respectively. Then $e _ { 1 } = \partial _ { x _ { 1 } } , \dotsc , e _ { n - 1 } =$

$\partial _ { x _ { n - 1 } }$ span $T _ { x } S ^ { n - 1 }$ and $T _ { - x } S ^ { n - 1 }$ . Recalling that $b ( x ) = b ( - x )$ and $\nu ( - x ) = - \nu ( x )$ , we compute from (3.1) and (3.2),

$$
\begin{array} { r l } & { \indent D f | _ { - x } ( e _ { i } ) = D b | _ { - x } ( e _ { i } ) = - D b | _ { x } ( e _ { i } ) = - D f | _ { x } ( e _ { i } ) , \quad i = 1 , \ldots , n - 1 , } \\ & { \indent D f | _ { - x } ( e _ { n } ) = - \nu ( - e _ { n } ) = \nu ( e _ { n } ) = D f | _ { x } ( e _ { n } ) . } \end{array}
$$

On the other hand, for $\begin{array} { r } { x = e _ { n } , } \end{array}$ the map $R _ { x } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ is given by $R _ { x } ( e _ { i } ) = e _ { i }$ for $i = 1 , \ldots , n - 1$ and $R _ { x } ( e _ { n } ) = - e _ { n }$ . Then

$$
\begin{array} { r l } & {   D f | _ { - x } \circ R _ { x } ( e _ { i } ) = { \cal D } f | _ { - x } ( e _ { i } ) = - { \cal D } f | _ { x } ( e _ { i } ) , \quad i = 1 , \ldots , n - 1 , } \\ & {   { \cal D } f | _ { - x } \circ R _ { x } ( e _ { n } ) = { \cal D } f | _ { - x } ( - e _ { n } ) = - { \cal D } f | _ { x } ( e _ { n } ) , } \end{array}
$$

completing the proof of (3.6).

From (3.5) we have

$$
f ( x + w ) - f ( - x + R _ { x } w ) = 2 D f | _ { x } w + O ( | w | ^ { 2 } ) .
$$

Then $F _ { 0 }$ satisfies (1) by the same proof as in the case of n even.

We define $F _ { t }$ in the analogous way using $g$ and h. It is straightforward to check (2) and (3) and so we can again apply Theorem 2.1 to obtain two points $x + w$ and $- x + R _ { x } w$ which are distinct because w is small, with $G ( x + w ) = G ( - x + R _ { x } w )$ □

## 4. A simpler proof of Theorem 3.1 in the case of $n = 2$

When $n = 2$ there is a more elementary and explicit proof of Theorem 3.1, the case already treated by Hwang [8]. For the reader’s convenience, we write down the selfcontained proof which is similar in spirit to [8]. Rather than define our function f using an immersion of $\mathbb { R } \mathbb { P } ^ { 1 } = S ^ { 1 }$ into $\mathbb { R } ^ { 2 }$ (of which there are many), we simply write down an explicit function $f .$

Let $f : \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 }$ be the map $f ( x _ { 1 } , x _ { 2 } ) = ( x _ { 1 } ^ { 2 } - x _ { 2 } ^ { 2 } , 2 x _ { 1 } x _ { 2 } )$ , namely the map $z \mapsto z ^ { 2 }$ in complex coordinates. On the unit circle $\{ ( \cos \theta , \sin \theta ) \} _ { \theta \in [ 0 , 2 \pi ] }$ , the map f sends the point with angle θ to the point with angle 2θ, and antipodal points x and −x have the same image under $f .$

Define the compact 3-manifold $M = S ^ { 1 } \times \overline { { B } } _ { \delta }$ , where $B _ { \delta }$ is the open ball of radius δ centered at the origin in $\mathbb { R } ^ { 2 }$ . The boundary of M is $S ^ { 1 } \times \partial \overline { { B } } _ { \delta }$ . Identify $S ^ { 1 }$ with $\mathbb { R } / 2 \pi \mathbb { Z }$ via

$$
\mathbb { R } / 2 \pi \mathbb { Z } \ni [ \theta ] \mapsto x _ { \theta } : = ( \cos \theta , \sin \theta ) \in S ^ { 1 } .
$$

There is a smooth free involution of M given by $\tau ( [ \theta ] , w ) = ( [ \theta + \pi ] , w )$ . Observe that if $( [ \theta ] , w ) \in M$ then $x _ { \theta } +$ w and − $^ { - x _ { \theta } + }$ w are in the annulus A. Define a map $F _ { 0 } : M \to \mathbb { R } ^ { 3 }$ by

$$
F _ { 0 } ( [ \theta ] , w ) = ( f ( x _ { \theta } + w ) - f ( - x _ { \theta } + w ) , \cos \theta ) .
$$

More explicitly, writing $w = ( w _ { 1 } , w _ { 2 } )$ , we obtain

$$
F _ { 0 } ( [ \theta ] , w ) = \big ( 4 w _ { 1 } \cos \theta - 4 w _ { 2 } \sin \theta , ~ 4 w _ { 1 } \sin \theta + 4 w _ { 2 } \cos \theta , ~ \cos \theta \big ) .
$$

Observe that $F _ { 0 }$ has exactly two zeros, at $( [ \pi / 2 ] , 0 )$ and $( [ 3 \pi / 2 ] , 0 )$ . Let’s compute the derivative of $F _ { 0 }$ at $( [ \pi / 2 ] , 0 )$ . With columns ordered as $( \theta , w _ { 1 } , w _ { 2 } )$ , its derivative is

$$
D F _ { 0 } | _ { ( [ \theta ] , w ) } = \left( \begin{array} { c c c } { { - 4 w _ { 1 } \sin \theta - 4 w _ { 2 } \cos \theta } } & { { 4 \cos \theta } } & { { - 4 \sin \theta } } \\ { { 4 w _ { 1 } \cos \theta - 4 w _ { 2 } \sin \theta } } & { { 4 \sin \theta } } & { { 4 \cos \theta } } \\ { { - \sin \theta } } & { { 0 } } & { { 0 } } \end{array} \right) .
$$

In particular,

$$
D F _ { 0 } | _ { ( [ \pi / 2 ] , 0 ) } = \left( \begin{array} { c c c } { { 0 } } & { { 0 } } & { { - 4 } } \\ { { 0 } } & { { 4 } } & { { 0 } } \\ { { - 1 } } & { { 0 } } & { { 0 } } \end{array} \right) ,
$$

which is invertible. This shows that $F _ { 0 }$ satisfies assumption (1) of Theorem 2.1.

Next, define $F _ { 1 } : M \to \mathbb { R } ^ { 3 }$ by

$$
F _ { 1 } ( [ \theta ] , w ) = ( g ( x _ { \theta } + w ) - g ( - x _ { \theta } + w ) , h ( x _ { \theta } + w ) - h ( - x _ { \theta } + w ) ) ,
$$

and $F : M \times [ 0 , 1 ]  \mathbb { R } ^ { 3 } { \mathrm { ~ b y ~ } } F ( x , t ) = F _ { t } ( x ) : = ( 1 - t ) F _ { 0 } ( x ) + t F _ { 1 } ( x )$ . Observe that

$$
F _ { t } ( \tau ( [ \theta ] , w ) ) = F _ { t } ( [ \theta + \pi ] , w ) = - F _ { t } ( [ \theta ] , w ) .
$$

Hence (2) is satisfied.

Finally we check (3). Let $( [ \theta ] , w )$ be in the boundary ∂M. We have $| w | = \delta$ . Then

$$
\begin{array} { r } { | F _ { 0 } ( [ \theta ] , w ) | ^ { 2 } \geq ( 4 w _ { 1 } \cos \theta - 4 w _ { 2 } \sin \theta ) ^ { 2 } + ( 4 w _ { 1 } \sin \theta + 4 w _ { 2 } \cos \theta ) ^ { 2 } = 1 6 \delta ^ { 2 } . } \end{array}
$$

Since $g$ is close to $f$ on A we may assume, if $\varepsilon > 0$ is small enough, that $F _ { t }$ does not vanish on the boundary of M for $t \in [ 0 , 1 ]$

Then by Theorem 2.1, $F _ { 1 }$ has a zero, namely a point $( [ \theta ] , w )$ in M such that

$$
g ( x _ { \theta } + w ) = g ( - x _ { \theta } + w ) , \quad h ( x _ { \theta } + w ) = h ( - x _ { \theta } + w ) .
$$

Hence $G ( x _ { \theta } + w ) = G ( - x _ { \theta } + w )$ and $x _ { \theta } + w , - x _ { \theta } -$ +w are distinct points in A, completing the proof of Theorem 3.1 in the case $n = 2$

## 5. Proof of Theorem 1.1

Besides the key ingredient in $\ S 3 .$ , we need the following lemma on the networks, which shows that we can approximate neural networks by injective maps composed with projections (cf. [9, Section 5]).

For integers $N \geq k \geq 1$ , write

$$
\pi _ { N , k } : \mathbb { R } ^ { N } \to \mathbb { R } ^ { k } , \quad \pi _ { N , k } ( x _ { 1 } , \dots , x _ { N } ) : = ( x _ { 1 } , \dots , x _ { k } ) .
$$

Lemma 5.1. Assume the activation function σ is continuous and monotone. Fix $N \geq$ max $\{ n , m \}$ . Suppose that $F : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ is a network of width at most N. For every nonempty compact $K \subset \mathbb { R } ^ { n }$ and every $\varepsilon > 0$ , there is a smooth injective map $G : \mathbb { R } ^ { n } $ $\mathbb { R } ^ { N }$ such that

$$
\operatorname* { s u p } _ { K } | \pi _ { N , m } \circ G - F | < \varepsilon .
$$

Proof. We first approximate $\sigma$ by smooth strictly monotone functions. Choose $\eta \in$ $\{ 1 , - 1 \}$ such that ησ is non-decreasing. Choose a standard nonnegative mollifier $\varphi \in$ $C _ { c } ^ { \infty } ( - 1 , 1 )$ with $\begin{array} { r } { \int _ { \mathbb { R } } \varphi ( r ) d r = 1 } \end{array}$ . For $t \geq 0$ , define

$$
\sigma _ { t } ( s ) : = \int _ { - 1 } ^ { 1 } \sigma ( s - t r ) \varphi ( r ) d r + \eta t s , \quad s \in \mathbb { R } .
$$

Clearly, $\sigma _ { 0 } = \sigma$ . By standard properties of mollification, $\sigma _ { t }$ is smooth for every $t > 0$ and $\sigma _ { t }  \sigma$ locally uniformly as $t \to 0 ^ { + }$ . For $t > 0$ , we verify that $\sigma _ { t }$ is strictly monotone

and thus injective on R. Indeed, for $s _ { 1 } < s _ { 2 }$ , monotonicity of $\sigma$ and $\varphi \geq 0$ give

$$
\begin{array} { c l l } { \eta \sigma _ { t } ( s _ { 2 } ) - \eta \sigma _ { t } ( s _ { 1 } ) = \displaystyle \int _ { - 1 } ^ { 1 } \big ( \eta \sigma ( s _ { 2 } - t r ) - \eta \sigma ( s _ { 1 } - t r ) \big ) \varphi ( r ) d r + t ( s _ { 2 } - s _ { 1 } ) } & & { } \\ { \geq t ( s _ { 2 } - s _ { 1 } ) > 0 . } & & { } \end{array}
$$

We again use $\sigma _ { t }$ to denote its higher dimensional extensions in the same way as for σ. Then $\sigma _ { t }$ as a map from $\mathbb { R } ^ { N }$ to $\mathbb { R } ^ { N }$ is smooth and injective for every $t > 0$

Next we put all the afine maps of the network on the same space $\mathbb { R } ^ { N }$ . Let F be expressed by afine maps $A ^ { ( 1 ) } , \ldots , A ^ { ( L ) }$ and dimensions $d _ { 0 } = n , d _ { L } = m$ , with $d _ { \ell } \leq N$ for $\ell = 0 , \ldots , L$ . For $t \geq 0$ , define $\overline { { A } } _ { t } ^ { ( \ell ) } : \mathbb { R } ^ { N }  \mathbb { R } ^ { N }$ by

$$
\overline { { A } } _ { t } ^ { ( \ell ) } ( x ) : = \big ( A ^ { ( \ell ) } ( x _ { 1 } , \dots , x _ { d _ { \ell - 1 } } ) , 0 , \dots , 0 \big ) + t x , \quad 1 \le \ell \le L ,
$$

for ${ \boldsymbol x } = ( x _ { 1 } , \dots , x _ { N } )$ . Here there are $( N - d _ { \ell } )$ zeros in the above definition; when $d _ { \ell } = N$ there are no added zero coordinates. There is a finite set $E \subset \mathbb { R }$ such that for $t \not \in E$ , the afine maps $\overline { { A } } _ { t } ^ { ( \ell ) }$ are isomorphisms. This follows from the fact that if M is any $N \times N$ matrix and $I _ { N }$ the $N \times N$ identity matrix, det $( M + t I _ { N } )$ is a polynomial in t of degree $N$ and hence vanishes for at most N values of t.

Let $J : \mathbb { R } ^ { n }  \mathbb { R } ^ { N }$ be defined by $J ( x ) : = ( x , 0 , \ldots , 0 )$ , and define

$$
G _ { t } : = \overline { { A } } _ { t } ^ { ( L ) } \circ \sigma _ { t } \circ \overline { { A } } _ { t } ^ { ( L - 1 ) } \circ \cdots \circ \sigma _ { t } \circ \overline { { A } } _ { t } ^ { ( 1 ) } \circ J .
$$

When $L = 1$ , this means $G _ { t } = \overline { { A } } _ { t } ^ { ( 1 ) } \circ J .$ . For every $t > 0$ with $t \not \in E$ , all factors are smooth and injective, including J. Their composition $G _ { t }$ is consequently smooth and injective on $\mathbb { R } ^ { n }$ . Moreover,

$$
G _ { 0 } ( x ) = ( F ( x ) , 0 , \ldots , 0 ) .
$$

Since $( t , x ) \mapsto G _ { t } ( x )$ is continuous on the compact set $[ 0 , 1 ] \times K$ , we obtain, for all suficiently small $t > 0$

$$
\operatorname* { s u p } _ { K } | G _ { t } - G _ { 0 } | < \varepsilon .
$$

Choose such a positive t outside the finite set $E ,$ and let $G : = G _ { t }$ . Then, by $\pi _ { N , m } \circ G _ { 0 } =$ F, we have

$$
\operatorname* { s u p } _ { K } | \pi _ { N , m } \circ G - F | \leq \operatorname* { s u p } _ { K } | G _ { t } - G _ { 0 } | < \varepsilon ,
$$

which proves the lemma.

Now we are ready to present the following:

Proof of Theorem 1.1. Let $\varepsilon , \delta , A , f$ be those given by Theorem 3.1. We may assume $\delta < 1 / 4$ . Note that the same defining formula (3.1) extends f to a smooth map on $\mathbb { R } ^ { n } \setminus \{ 0 \}$ . Choose a standard cutof $\chi \in C _ { c } ^ { \infty } ( \mathbb { R } )$ supported on $( 1 / 2 , 3 / 2 )$ such that $0 \leq \chi \leq 1$ on R and $\chi = 1$ on $[ 3 / 4 , 5 / 4 ]$ . Define

$$
{ \overline { { f } } } ( y ) : = \chi ( | y | ) f ( y ) \quad { \mathrm { f o r ~ } } y \in \mathbb { R } ^ { n } \setminus \{ 0 \} , \quad { \mathrm { a n d } } \quad { \overline { { f } } } ( 0 ) = 0 .
$$

Then $\overline { { f } } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathbb { R } ^ { n } )$ agrees with $f$ on an open neighborhood of A.

Let

$$
\varepsilon _ { 0 } : = \varepsilon / 2 , \quad d : = \operatorname* { m i n } \{ n , m \} , \quad \mathrm { a n d } \quad N : = n + d - 1 .
$$

Write $f = ( f _ { 1 } , \ldots , f _ { n } )$ on A and ${ \overline { { f } } } = ( { \overline { { f } } } _ { 1 } , \dots , { \overline { { f } } } _ { n } )$ on $\mathbb { R } ^ { n }$ . Define

$$
f _ { \ast } : = ( \overline { { f } } _ { 1 } , \dots , \overline { { f } } _ { d } , 0 , \dots , 0 ) \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathbb { R } ^ { m } ) .
$$

Here there are $( m - d )$ zeros in the definition of $f _ { * }$

Fix σ to be any continuous and monotone activation. Suppose the contrary that a network $F : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ of width at most N satisfies

$$
\operatorname* { s u p } _ { A } | f _ { * } - F | < \varepsilon _ { 0 } .
$$

By the definition of the width, we must have $N \geq \operatorname* { m a x } \{ n , m \}$ . We may therefore apply Lemma 5.1, with $K = A$ , to obtain a smooth injective map $G : \mathbb { R } ^ { n } \to \mathbb { R } ^ { N }$ such that

$$
\operatorname* { s u p } _ { A } | \pi _ { N , m } \circ G - F | < \varepsilon _ { 0 } .
$$

Write $G = ( g , h )$ , where $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { d }$ consists of the first d coordinates and $h$ : $\mathbb { R } ^ { n } \to \mathbb { R } ^ { n - 1 }$ consists of the remaining $N - d = n - 1$ coordinates. By $d \leq m$ , the above inequalities yield

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { A } | g - ( f _ { 1 } , \dotsc , f _ { d } ) | \leq \operatorname* { s u p } _ { A } \left| \pi _ { N , m } \circ G - f _ { * } \right| } \\ { \leq \operatorname* { s u p } _ { A } \left| \pi _ { N , m } \circ G - F \right| + \operatorname* { s u p } _ { A } \left| f _ { * } - F \right| } \\ { < 2 \varepsilon _ { 0 } = \varepsilon . } \end{array}
$$

Here we used ${ \overline { { f } } } _ { j } = f _ { j }$ on A.

If $d < n$ , define ${ \widehat { g } } : = ( g _ { 1 } , \ldots , g _ { d } , f _ { d + 1 } , \ldots , f _ { n } ) : A \to \mathbb { R } ^ { n }$ . If $d = n$ , simply let ${ \widehat { g } } : = g | _ { A }$ In either case, $\widehat g$ is smooth on a neighborhood of A and

$$
\operatorname* { s u p } _ { A } | { \widehat { g } } - f | = \operatorname* { s u p } _ { A } | g - ( f _ { 1 } , \ldots , f _ { d } ) | < \varepsilon .
$$

Then Theorem 3.1 shows that the map $( \widehat { g } , h ) : A \to \mathbb { R } ^ { 2 n - 1 }$ is not injective. Hence there are distinct points $y , z \in A$ with ${ \widehat { g } } ( y ) = { \widehat { g } } ( z )$ and $h ( y ) = h ( z )$ . The first d coordinates of the first equation give $g ( y ) = g ( z )$ . Together with the second equation, this implies

$$
G ( y ) = ( g ( y ) , h ( y ) ) = ( g ( z ) , h ( z ) ) = G ( z ) ,
$$

contradicting the injectivity of G. The theorem is therefore proved.

## 6. Proof of Corollary 1.1

The following theorem is well-known in diferential topology.

Theorem 6.1. For $n \geq 2$ , the real projective space $\mathbb { R } \mathbb { P } ^ { n - 1 }$ admits a smooth immersion into $\mathbb { R } ^ { n }$ if and only $i f n \in \{ 2 , 3 , 4 , 7 , 8 \}$

We recall how this classification follows from the classical results recorded in [6]. The case $n = 2$ is immediate, since $\mathbb { R } ^ { \mathbb { P } ^ { 1 } }$ is difeomorphic to the circle.

Suppose first that $n \geq 3$ is odd. By [6, Theorem $7 . 2 ] , \ \mathbb { R P } ^ { n - 1 }$ admits a smooth immersion into $\mathbb { R } ^ { n }$ if and only if $\mathbb { R P } ^ { n }$ is parallelizable (i.e. admits a global smooth frame). According to results of Kervaire, Bott, and Milnor recalled in [6, footnote $^ { 3 , }$ p. 272], $\mathbb { R P } ^ { n }$ is parallelizable precisely when n is 3 or $^ { 7 }$

Now suppose that $n \geq 4$ is even, and let $i : \mathbb { R P } ^ { n - 1 } \to \mathbb { R } ^ { n }$ be a smooth immersion. Write $\pi : S ^ { n - 1 } \to \mathbb { R P } ^ { n - 1 }$ as the antipodal quotient map. As in the proof of [6, Theorem 7.2], $i \circ \pi \mathrm { \ g i v e s }$ an immersion $S ^ { n - 1 } \to \mathbb { R } ^ { n }$ of even normal degree, then Milnor’s result implies that $S ^ { n - 1 }$ is parallelizable. Then the result recalled in [6, footnote 3, p. 272] forces $n = 4$ or $n = 8$ . Conversely, the required immersions for $n = 4 , 8$ are recorded in [6, Theorem 7.1]. Combining these cases gives precisely $n \in \{ 2 , 3 , 4 , 7 , 8 \}$

Proof of Corollary 1.1. The desired conclusion follows directly from Theorem 1.1, Theorem 6.1, and the upper bound (1.1). □

## References

1. Y. Cai, Achieve the minimum width of neural networks for universal approximation, in International Conference on Learning Representations (ICLR), 2023.

2. G. Cybenko, Approximation by superpositions of a sigmoidal function, Mathematics of Control, Signals, and Systems 2 (1989), 303–314.

3. K.-I. Funahashi, On the approximate realization of continuous mappings by neural networks, Neural Networks 2 (1989), 183–192.

4. B. Hanin and M. Sellke, Approximating continuous functions by ReLU nets of minimal width, arXiv:1710.11278v2 (2018).

5. K. Hornik, M. Stinchcombe, and H. White, Multilayer feedforward networks are universal approximators, Neural Networks 2 (1989), 359–366.

6. M. W. Hirsch, Immersions of manifolds, Trans. Amer. Math. Soc. 93 (1959), 242–276.

7. M. W. Hirsch, Diferential Topology, Graduate Texts in Mathematics, vol. 33, Springer-Verlag, New York–Heidelberg, 1976.

8. G. Hwang, Minimum width for deep, narrow MLP: A difeomorphism approach, Advances in Neural Information Processing Systems 38 (2025), 162057–162082.

9. J. Johnson, Deep, skinny neural networks are not universal approximators, in International Conference on Learning Representations (ICLR), 2019.

10. P. Kidger and T. Lyons, Universal approximation with deep narrow networks, in Conference on Learning Theory, PMLR, vol.125, 2020, 2306–2327.

11. N. Kim, C. Min, and S. Park, Minimum width for universal approximation using ReLU networks on compact domain, in International Conference on Learning Representations (ICLR), 2024.

12. M. Leshno, V. Ya. Lin, A. Pinkus, and S. Schocken, Multilayer feedforward networks with a nonpolynomial activation function can approximate any function, Neural Networks 6 (1993), 861–867.

13. Z. Lu, H. Pu, F. Wang, Z. Hu, and L. Wang, The expressive power of neural networks: A view from the width, in Advances in Neural Information Processing Systems 30 (2017), 6231–6239.

14. S. Park, C. Yun, J. Lee, and J. Shin, Minimum width for universal approximation, in Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021.

15. D. Rochau, R. Chan, and H. Gottschalk, New advances in universal approximation with neural networks of minimal width, arXiv:2411.08735v3 (2025).

16. J. Shin, N. Kim, G. Hwang, and S. Park, Minimum width for universal approximation using squashable activation functions, in Proceedings of the 42nd International Conference on Machine Learning (ICML), vol.267, 2025, 55096–55121.

17. M. Telgarsky, Representation benefits of deep feedforward networks, arXiv:1509.08101 (2015).

18. D. Yarotsky, Error bounds for approximations with deep ReLU networks, Neural Networks 94 (2017), 103–114.

Department of Mathematics<sub>,</sub> Northwestern University<sub>,</sub> 2033 Sheridan Road<sub>,</sub> Evanston<sub>,</sub> IL 60208<sub>,</sub> USA