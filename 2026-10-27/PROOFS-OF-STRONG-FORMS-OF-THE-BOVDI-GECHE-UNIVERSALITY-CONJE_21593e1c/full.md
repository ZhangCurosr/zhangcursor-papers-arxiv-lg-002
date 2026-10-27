# PROOFS OF STRONG FORMS OF THE BOVDI–GECHE UNIVERSALITY CONJECTURE AND A COMPLEX ANALOGUE

VICTOR BOVDI AND AMBRUS PAL<sup>´</sup>

Abstract. We prove two universality theorems for neural networks consisting of a single neuron. First we prove a strong form of the Bovdi–Geche universality conjecture on neurons over finite fields, where the activation function is a certain modification of the discrete logarithm, using the Weil bound on multiplicative character sums over finite fields. Then we show a generalisation of Aizenberg’s universality theorem for complex binary neurons with a weight vector independent of the target function, positively answering another question in the literature. The proof of the latter uses an equidistribution theorem for the dilates of certain maps into tori.

## 1. Introduction

Neural networks consisting of a single neuron have been much studied in the literature, including a class of functions which are usually called threshold functions, both over finite fields (for example see [9], [11], [12, Chapter 4], and [13]), and over the complex field (for example see [1], [2], [3], [4], [6] and [7]). The universality of such functions on finite subsets is of fundamental importance, and much efort was exerted on this circle of problems. In this paper we prove two universality theorems for threshold functions. In order to state them, we will need to review some basic definitions.

Definition 1.1. Let K be a field, let Y be a set, and let $\Phi : K  Y$ be a partially defined map, which we will call the activation function, with domain of definition $\mathcal { D } \subset K$ . As usual for every $n \in \mathbb { N }$ let $ \cdot , \cdot  : K ^ { n } \times K ^ { n }  K$ denote the usual scalar product:

$$
\langle { \underline { { x } } } , { \underline { { y } } } \rangle = \sum _ { i = 1 } ^ { n } x _ { i } y _ { i } , { \mathrm { ~ w h e r e ~ } } { \underline { { x } } } = ( x _ { 1 } , \dots , x _ { n } ) , { \underline { { y } } } = ( y _ { 1 } , \dots , y _ { n } ) \in K ^ { n } .
$$

For every $w \in K ^ { n }$ and $T \in K$ the threshold function attached to $( { \underline { { w } } } , T )$ with respect to $\Phi$ is the partially defined function:

$$
\theta : K ^ { n } \to Y
$$

which maps every $\underline { { x } } \in K ^ { n }$ such that $\langle \underline { { x } } , \underline { { w } } \rangle + T \in \mathcal { D }$ to the value $\theta ( \underline { { x } } ) = \Phi ( \langle \underline { { x } } , \underline { { w } } \rangle + T )$

Definition 1.2. A pair $( { \underline { { w } } } , T ) \in K ^ { n } \times K$ is admissible with respect to a subset $X \subseteq K ^ { n }$ if the associated threshold function θ is defined on every element of $X$ . The threshold function θ attached to $( { \underline { { w } } } , T ) \in K ^ { n } \times K$ realises a function $f : X \to Y$ if the pair $( { \underline { { w } } } , T )$ is admissible with respect to X and the restriction of θ to X is $f$ . In this case we will also sometimes say that $( { \underline { { w } } } , T )$ realises $f$ for the sake of simplicity. The pair $( K , \Phi )$ is universal for X if every map $f : X \to Y$ is realised by some threshold function.

Definition 1.3. Fix integers

$$
n \geq 1 , \qquad 2 \leq k _ { 1 } \leq \cdot \cdot \cdot \leq k _ { n } , \qquad q \geq 2 ,\tag{1.3.1}
$$

set $N = k _ { 1 } \cdots k _ { n }$ , and let $L = \operatorname { l c m } ( k _ { 1 } , \dots , k _ { n } , q )$ . A finite field F of cardinality $| \mathbb { F } | = Q$ is called admissible for the parameters in (1.3.1) if

$$
L \mid Q - 1 \quad { \mathrm { a n d } } \quad Q - 1 > \operatorname* { m a x } \{ k _ { 1 } , . . . , k _ { n } , q \} .
$$

By Dirichlet’s theorem on primes in arithmetic progressions there are admissible finite fields of arbitrarily large cardinality.

Definition 1.4. For every divisor d of $Q - 1$ let $C _ { d }$ denote the unique subgroup of $\mathbb { F } ^ { * }$ of order $d .$ Set

$$
X = C _ { k _ { 1 } } \times \cdot \cdot \cdot \times C _ { k _ { n } } \subset ( \mathbb { F } ^ { * } ) ^ { n } \subset \mathbb { F } ^ { n } , \qquad Y = C _ { q } \subset \mathbb { F } ^ { * } .
$$

Fix a primitive element $\varepsilon \in \mathbb { F } ^ { * }$ , set $\begin{array} { r } { h = { \frac { Q - 1 } { q } } } \end{array}$ and $\sigma = \varepsilon ^ { h }$ . The activation function attached to $( \mathbb { F } , \varepsilon )$ is the map

$$
\Phi _ { \varepsilon } : \mathbb { F } ^ { * } \longrightarrow Y
$$

given by the rule:

$$
\Phi _ { \varepsilon } \mathopen { } \mathclose \bgroup \left( \varepsilon ^ { a } \aftergroup \egroup \right) = \sigma ^ { \lfloor a / h \rfloor } , \qquad 0 \leq a \leq Q - 2 .\tag{1.4.1}
$$

For every function $f : X \to Y$ let $R _ { f } ( \mathbb { F } , \varepsilon )$ be the number of pairs $( \underline { w } , T ) \in \mathbb { F } ^ { n } \times \mathbb { F }$ admissible with respect to X such that the threshold function attached to $( { \underline { { w } } } , T )$ with respect to $\Phi _ { \varepsilon }$ realises $f .$

Theorem 1.5. There exists a constant $C _ { N } > 0$ , depending only on $N _ { \cdot }$ such that for every admissible finite field F of cardinality $Q$ , every map $f : X \to Y$ , and every primitive element $\varepsilon \in \mathbb { F } ^ { * }$ we have

$$
\left| R _ { f } ( \mathbb { F } , \varepsilon ) - { \frac { Q ^ { n + 1 } } { q ^ { N } } } \right| \leq C _ { N } Q ^ { n + 1 / 2 } ( 1 + \log Q ) ^ { N } .\tag{1.5.1}
$$

Since in the estimate above the main term has order $Q ^ { n + 1 }$ and the error is bounded by a constant multiple of $Q ^ { n + 1 / 2 } ( 1 + \log Q ) ^ { N }$ , we immediately get the following

Corollary 1.6. Let $\mathbb { F }$ be an admissible finite field of cardinality $Q$ , and let $\varepsilon \in \mathbb { F } ^ { * }$ be a primitive element. If Q is suficiently large (for fixed N and $q )$ then the pair $( \mathbb { F } , \Phi _ { \varepsilon } )$ is universal for X. □

This claim is stronger than the conjecture formulated by Bovdi– Geche on page 125 of [12], which only postulated the existence of an admissible field $\mathbb { F }$ such that $( \mathbb { F } , \Phi _ { \varepsilon } )$ is universal for some choice of a primitive element $\varepsilon \in \mathbb { F } ^ { * }$

The proof of Theorem 1.5 is a rather standard application of the Weil bound for multiplicative character sums; for similar arguments see for example [10], [17] and [16]. As usual it proceeds by expressing the characteristic function of the set of pairs $( { \underline { { w } } } , T )$ whose threshold function realises $f$ in terms of multiplicative characters. The way this proof mainly difers from the arguments in the aforementioned papers is that while they work with the partition of the multiplicative group of the field into the cosets of a subgroup of fixed index, we work with the level sets of the activation function $\Phi _ { \varepsilon }$ which are not cosets. However this diference is taken care of by the elementary Lemma 2.3.

Definition 1.7. Now fix an integer $q \geq 2$ , and set $K = \mathbb { C }$ and $Y =$ $\mathbb { Z } / q \mathbb { Z }$ . As usual let $\{ \cdot \} : \mathbb { R } \to [ 0 , 1 )$ denote the fractional-part function. For every integer $\ell \geq 1$ let $\Phi _ { q , \ell } : \mathbb { C } \mathrm { ~ -- ~ } Y$ denote the partially defined function given by the rule:

$$
\Phi _ { q , \ell } ( z ) = j { \mathrm { ~ m o d ~ } } q { \mathrm { ~ i f ~ a n d ~ o n l y ~ i f ~ } } \{ { \frac { \ell \cdot \arg ( z ) } { 2 \pi } } \} \in \left( { \frac { j } { q } } , { \frac { j + 1 } { q } } \right)
$$

for every $j \in \{ 0 , 1 , \ldots , q - 1 \}$ , with domain of definition $\mathbb { C } ^ { * } \backslash \mathcal { R } _ { q , \ell } ,$ where

$$
\mathcal { R } _ { q , \ell } = \big \{ z \in \mathbb { C } ^ { * } \ | \ \{ \frac { \ell \cdot \arg ( z ) } { 2 \pi } \} = \frac { j } { q } \ \mathrm { f o r ~ s o m e } \ 0 \leq j < q \big \} ,
$$

and arg : $\mathbb { C } \mathrm { ~ - } \mathrm { \to ~ } \mathbb { R }$ is a branch of the argument function on $\mathbb { C } ^ { * } \setminus \mathcal { R } _ { q , \ell } .$

Theorem 1.8. Let $X \subset \mathbb { C } ^ { n }$ be nonempty and finite. Then there exist $a \underline { { w } } \in \mathbb { C } ^ { n }$ and an integer $\ell _ { 0 } \geq 1$ such that for every integer $\ell \geq \ell _ { 0 }$ and every map $f : X \to Y$ there is a $b _ { f , \ell } \in \mathbb { C }$ such that $( \underline { { w } } , b _ { f , \ell } )$ realises f with respect to $\Phi _ { q , \ell }$

Remark 1.9. The construction of complex threshold functions in this paper originates in the works $[ 1 ] , [ 2 ] , [ 3 ]$ and [4]. Since it was known that these functions are not universal in general when $\ell = 1 , \operatorname { I } .$ . N. Aizenberg introduced the more general activation function in Definition 1.7 (see [6, formula (5), p. 350]), and on p. 348 of the same paper he asked whether increasing ℓ makes arbitrary q-valued functions realisable. For $q = 2 ,$ , that is for Boolean output, universality was already known for $\bar { X } = \{ - 1 , 1 \} ^ { n }$ (see [5, pp. 467–473]), but it was open for general q until our theorem. The latter is actually much stronger, since we can choose the weight vector w uniformly which will work for every function f and every suficiently large ℓ.

The proof of Theorem 1.8 uses an equidistribution theorem for the dilates of certain maps into tori. Such results are classical, see for example [18] and its extension to nilmanifolds in [14, Theorem 1.2]. The vector w is chosen so that the conditions of the equidistribution theorem apply. Our strategy is very similar to Yarotsky’s neural-network construction in [19, pp. 11934–11935], although the latter uses Kronecker’s density theorem instead.

Contents 1.10. In Section 2 we prove Theorem 1.5, while in Section 3 we show Theorem 1.8 using the methods we described above.

Acknowledgement 1.11. The authors acknowledge that the initial sketches of the paper were developed via conversations with ChatGPT, but the authors thoroughly reviewed and revised the manuscript, and they take full responsibility for the final version of the paper.

## 2. The Bovdi–Geche conjecture and multiplicative character sums

Notation 2.1. Let $\mathbb { F }$ be a finite field of cardinality $| \mathbb { F } | = Q$ which is admissible for the parameters in (1.3.1) and let $\varepsilon \in \mathbb { F } ^ { * }$ be a primitive element as above. Write $\mathbb { F } ^ { * }$ as the disjoint union of the sets

$$
B _ { j } = \{ \varepsilon ^ { j h } , \varepsilon ^ { j h + 1 } , \cdot \cdot . . , \varepsilon ^ { ( j + 1 ) h - 1 } \} , \qquad 0 \leq j \leq q - 1 .\tag{2.1.1}
$$

Let ${ \widehat { \mathbb { F } } } ^ { * }$ denote the group of complex-valued multiplicative characters of $\mathbb { F } ^ { * }$ . Let $\chi _ { 0 } \in \widehat { \mathbb { F } } ^ { * }$ be the principal character. We extend every character $\chi \in \widehat { \mathbb { F } } ^ { * }$ to F by setting $\chi ( 0 ) = 0$ . For $0 \leq j < q , z \in \mathbb { F }$ and $\chi \in \widehat { \mathbb { F } } ^ { * }$ 2 define

$$
c _ { j } ( \chi ) = \frac { 1 } { Q - 1 } \sum _ { z \in B _ { j } } \overline { { { \chi ( z ) } } } .\tag{2.1.2}
$$

As usual, for every subset $B \subseteq \mathbb { F }$ let $1 _ { B } : \mathbb { F }  \mathbb { C }$ denote the characteristic function of $B$

Lemma 2.2. We have:

$$
1 _ { B _ { j } } ( z ) = \sum _ { \chi \in \widehat { \mathbb { F } } ^ { * } } c _ { j } ( \chi ) \chi ( z ) \qquad ( \forall z \in \mathbb { F } ) .\tag{2.2.1}
$$

Proof. The equality for $z = 0$ follows from our convention for extending characters, while for other values it is immediate from the multiplicative Fourier inversion formula. □

Lemma 2.3. There is an absolute constant C such that

$$
c _ { j } ( \chi _ { 0 } ) = \frac { 1 } { q } , \quad a n d \quad \sum _ { \chi \in \widehat { \mathbb { F } } ^ { * } } | c _ { j } ( \chi ) | \leq C ( 1 + \log Q )
$$

for every choice of $q _ { ; }$ , the field F, the primitive element ε and $j$

Proof. Clearly $\begin{array} { r } { c _ { j } ( \chi _ { 0 } ) = \frac { h } { Q - 1 } = \frac { 1 } { q } } \end{array}$ , so the first claim holds. For simplicity let M denote $Q - 1$ , and let $\rho$ be the unique generator of $\widehat { \mathbb { F } } ^ { * }$ such that $\rho ( \varepsilon ) = e ^ { 2 \pi i / M }$ . For $s \in \{ 1 , 2 , \ldots , M - 1 \}$ using the definition of the set $B _ { j }$ we get

$$
\begin{array} { l l } { \displaystyle | \mathcal { E } _ { j } ( \rho ^ { s } ) | = \left| \frac { 1 } { M } \sum _ { u = 0 } ^ { h - 1 } \overline { { \rho ^ { s } ( z ^ { h + h + \delta } ) } } \right| } \\ { \displaystyle = \left| \frac { e ^ { - 2 \pi i \delta h / M } } { M } \sum _ { \alpha = 0 } ^ { h - 1 } e ^ { - 2 \pi i \alpha s u / \Delta } \right| } \\ { \displaystyle ~ - \frac { 1 } { M } \left| \sum _ { u = 0 } ^ { h - 1 } e ^ { 2 \pi i \alpha s / M } \right| } \\ { \displaystyle ~ = \frac { 1 } { M } \left| \frac { 1 - e ^ { - 2 \pi i \delta h / N / \delta } } { 1 - e ^ { - 2 \pi i \alpha s / M / M } } \right| } \\ { \displaystyle ~ = \frac { 1 } { M } \left| \frac { \sin ( \pi s h / M ) } { \sin ( \pi s h / M ) } \right| \le \frac { 1 } { M \sin ( \pi s h / M ) } , } \end{array}
$$

where we used that the factor $e ^ { - 2 \pi i s j h / M }$ has absolute value one in the third equation, the geometric-sum formula in the fourth equation, the identity $| 1 - e ^ { i t } | = \hat { 2 | } \sin ( t / 2 ) |$ in the fifth equation, and the inequality $| \sin ( t ) | \leq 1$ in the last inequality. Therefore

$$
\begin{array} { l } { \displaystyle \sum _ { \chi \in \widehat { \mathbb { R } } ^ { * } } | c _ { j } ( \chi ) | = \displaystyle \sum _ { s = 0 } ^ { M - 1 } | c _ { j } ( \rho ^ { s } ) | } \\ { \displaystyle \qquad \leq \displaystyle \frac { 1 } { q } + \displaystyle \sum _ { s = 1 } ^ { M - 1 } \displaystyle \frac { 1 } { M \sin ( \pi s / M ) } } \end{array}
$$

$$
\leq \frac { 1 } { q } + 2 \sum _ { s = 1 } ^ { \lfloor M / 2 \rfloor } \frac { 1 } { M \sin ( \pi s / M ) }
$$

$$
\leq \frac { 1 } { q } + 2 \sum _ { s = 1 } ^ { \lfloor M / 2 \rfloor } \frac { 1 } { 2 s } \leq C ( 1 + \log Q ) ,
$$

where we used

$$
\sin \left( { \frac { \pi ( M - s ) } { M } } \right) = \sin \left( { \frac { \pi s } { M } } \right)
$$

in the second inequality, the elementary inequality:

$$
\sin u \ge \frac { 2 u } { \pi } \qquad ( 0 \le u \le \pi / 2 ) ,
$$

which follows from the concavity of sine on this interval, in the third inequality, and the usual estimate for the harmonic series in the last inequality. □

Lemma 2.4. Let Z be a set of N distinct points $o f \mathbb { F } ^ { n }$ . For each $\underline { { x } } \in Z$ let $\chi _ { \underline { { x } } } \in \widehat { \mathbb { F } } ^ { * }$ , and assume that not all $\chi _ { x }$ are principal. Then

$$
\left| \sum _ { \underline { { w } } \in \mathbb { F } ^ { n } } \sum _ { T \in \mathbb { F } } \prod _ { \underline { { x } } \in Z } \chi _ { \underline { { x } } } ( T + \langle \underline { { x } } , \underline { { w } } \rangle ) \right| \leq c _ { N } Q ^ { n + 1 / 2 } ,\tag{2.4.1}
$$

where $c _ { N }$ depends only on $N$ .

Proof. Let ${ \mathcal { A } } \subseteq \mathbb { F } ^ { n }$ denote the set:

$$
\begin{array} { r } { \mathcal { A } = \{ \underline { { w } } \in \mathbb { F } ^ { n } | \exists \underline { { x } } , \underline { { y } } \in Z \ \mathrm { s u c h ~ t h a t } \underline { { x } } \neq \underline { { y } } \ \mathrm { a n d } \ \langle \underline { { x } } , \underline { { w } } \rangle = \langle \underline { { y } } , \underline { { w } } \rangle \} , } \end{array}
$$

and let $B \subseteq \mathbb { F } ^ { n }$ be its complement. Clearly A is the union of the hyperplanes

$$
\{ { \underline { { w } } } \in \mathbb { F } ^ { n } \mid \langle { \underline { { x } } } - { \underline { { y } } } , { \underline { { w } } } \rangle = 0 \} ,
$$

where $\underline { { x } } , y \in Z$ are diferent, so its cardinality is at most $\binom { N } { 2 } Q ^ { n - 1 }$ Therefore

$$
\left| \sum _ { \underline { { w } } \in \mathcal { A } } \sum _ { T \in \mathbb { F } } \prod _ { \underline { { x } } \in Z } \chi _ { \underline { { x } } } ( T + \langle \underline { { x } } , \underline { { w } } \rangle ) \right| \leq \binom { N } { 2 } Q ^ { n } ,\tag{2.4.2}
$$

Let $\rho$ generate the character group $\widehat { \mathbb { F } } ^ { * }$ as above, and for every $\underline { { x } } \in Z$ let $e ( \underline { { x } } ) \in \{ 0 , 1 , \dots , Q - 2 \}$ be the unique number such that $\chi _ { \underline { { x } } } = \rho ^ { e ( \underline { { x } } ) }$

Now let $w \in B$ be arbitrary. By assumption at least one of the $e ( \underline { { x } } )$ is nonzero, so the polynomial

$$
P _ { \underline { { w } } } ( t ) = \prod _ { e \in Z \atop e ( \underline { { x } } ) \neq 0 } ( t + \langle \underline { { x } } , \underline { { w } } \rangle ) ^ { e ( \underline { { x } } ) }
$$

is well-defined and has positive degree. Moreover the elements $\langle { \underline { { w } } } , { \underline { { x } } } \rangle$ are pairwise distinct as x ranges over $Z ,$ so every root of $P _ { \underline { { w } } } ( t )$ has multiplicity at most $Q - 2$ , and hence this polynomial is not a constant multiple of the $( Q - 1 ) { \mathrel { - } } \mathrm { t h }$ power of a polynomial. As the order of $\rho$ is $Q - 1$ , we may apply the standard Weil bound for the multiplicative character $\rho$ evaluated on the polynomial $P _ { \underline { { w } } } ( t )$ (see Theorem 5.41 of [15] on page 225) to get that

$$
\left| \sum _ { T \in \mathbb { F } } \rho ( P _ { \underline { { w } } } ( T ) ) \right| \leq ( N - 1 ) Q ^ { 1 / 2 } .
$$

Note that

$$
\rho ( P _ { \underline { { w } } } ( T ) ) = \prod _ { \underline { { x } } \in Z } \chi _ { \underline { { x } } } ( T + \langle \underline { { x } } , \underline { { w } } \rangle )
$$

for every $T \in \mathbb { F }$ outside the set

$$
E _ { \underline { { w } } } = \{ - \langle \underline { { x } } , \underline { { w } } \rangle \mid \underline { { x } } \in Z \} .
$$

There are at most $N$ elements of $E _ { w } ,$ and at each such value the two expressions difer by a quantity of absolute value at most 1, so we get that

$$
\left| \sum _ { \underline { { w } } \in B } \sum _ { T \in \mathbb { F } } \prod _ { \underline { { x } } \in Z } \chi _ { \underline { { x } } } ( T + \langle \underline { { x } } , \underline { { w } } \rangle ) \right| \le Q ^ { n } \bigl ( ( N - 1 ) Q ^ { 1 / 2 } + N \bigr ) .\tag{2.4.3}
$$

Combining the inequalities (2.4.2) and (2.4.3) together we get the claim. □

Notation 2.5. For every function $f : X \to Y$ and for every $x \in X$ let $j ( f , \underline { { x } } ) \in \{ 0 , 1 , \dots , q - 1 \}$ be the unique number such that

$$
f ( { \underline { { x } } } ) = \sigma ^ { j ( f , { \underline { { x } } } ) } .
$$

Lemma 2.6. We have:

$$
R _ { f } ( \mathbb { F } , \varepsilon ) = \sum _ { w \in \mathbb { F } ^ { n } } \sum _ { T \in \mathbb { F } } \prod _ { x \in X } 1 _ { B _ { j ( f , \underline { { x } } ) } } ( T + \langle \underline { { x } } , \underline { { w } } \rangle ) .\tag{2.6.1}
$$

Proof. For every $\left( \underline { w } , T \right) \in \mathbb { F } ^ { n + 1 }$ and for every $\underline { { x } } \in X$ we have

$$
1 _ { B _ { j ( f , \underline { { x } } ) } } ( T + \langle \underline { { x } } , \underline { { w } } \rangle ) = 1
$$

if and only if

$$
T + \langle \underline { { x } } , \underline { { w } } \rangle \not = 0 \quad \mathrm { a n d } \quad \Phi _ { \varepsilon } ( T + \langle \underline { { x } } , \underline { { w } } \rangle ) = \sigma ^ { j ( f , \underline { { x } } ) } = f ( \underline { { x } } ) .
$$

Therefore

$$
\prod _ { \underline { { x } } \in X } 1 _ { B _ { j ( f , \underline { { x } } ) } } ( T + \langle \underline { { x } } , \underline { { w } } \rangle )
$$

is equal to 1 if and only if $( { \underline { { w } } } , T )$ is admissible with respect to $X$ and realises $f ,$ , and it is zero otherwise. Summing this product over all pairs $( { \underline { { w } } } , T )$ gives (2.6.1). □

Proof of Theorem 1.5. By plugging equation (2.2.1) into the identity in Lemma 2.6 we get

$$
\begin{array} { r l } { \bar { H } _ { f } ( \bar { \mathbb { P } } , z ) } & { = \displaystyle \sum _ { z \in \mathbb { P } ^ { n } } \displaystyle \sum _ { \tau \in \mathbb { P } ^ { n } } \prod _ { \zeta \in \mathcal { X } } \Bigg ( \displaystyle \sum _ { \zeta \in \mathcal { X } / \tau \in \mathcal { X } ( \mathcal { X } , z ) } c _ { \zeta \zeta } ( T \cdot | \zeta , \zeta , w ) \rangle \Bigg ) } \\ & { = \displaystyle \sum _ { z \in \mathbb { P } ^ { n } } \displaystyle \sum _ { \tau \in \mathcal { X } ( \mathcal { X } , z ) \times z \in \mathcal { X } ( \frac { \tau } { z \cdot } ) \times \tau } \Bigg ( \prod _ { \zeta \in \mathcal { X } } c _ { \zeta ( T , \zeta ) \cdot ( \zeta , w ) } \Bigg ) } \\ & { \qquad \quad \cdot \displaystyle \prod _ { \zeta \in \mathcal { X } } \big ( T + \langle \underline { { \zeta } } , w , w \rangle \big ) } \\ & { = \displaystyle \sum _ { \zeta \in \mathcal { X } ( \bar { \mathbb { P } } , z ) \times \tau } \Bigg ( \prod _ { \zeta \in \mathcal { X } } c _ { \zeta ( T , \zeta ) \cdot ( \zeta , z ) } \Bigg ) } \\ & { \qquad \quad \cdot \displaystyle \sum _ { \zeta \in \mathcal { X } ^ { n } } \displaystyle \sum _ { z \in \mathcal { X } ( \mathcal { X } , z ) \times z } \big ( T + \langle \underline { { x } } , w \rangle \big ) , } \end{array}\tag{2.6.2}
$$

where we have expanded the finite product of finite sums in the second equality, and we have exchanged the finite sums over $( { \underline { { w } } } , T )$ and over the tuples $( \chi _ { \underline { { x } } } ) _ { \underline { { x } } \in X } \in ( \widehat { \mathbb { F } } ^ { * } ) ^ { X }$ in the third equality.

Consider first the tuple $( \chi _ { \underline { { x } } } ) _ { \underline { { x } } \in X } \in ( \widehat { \mathbb { F } } ^ { * } ) ^ { X }$ in which every $\chi _ { x }$ is the principal character $\chi _ { 0 }$ . By Lemma 2.3 we have $c _ { j ( f , \underline { { { x } } } ) } ( \chi _ { 0 } ) = 1 / q$ for every $\underline { { x } } ,$ therefore the corresponding contribution in the right hand side of (2.6.2) is independent of $f$ and equals

$$
{ \frac { 1 } { q ^ { N } } } \# \{ ( \underline { { w } } , T ) \mid T + \langle \underline { { x } } , \underline { { w } } \rangle \not = 0 { \mathrm { ~ f o r ~ e v e r y ~ } } \underline { { x } } \in X \} .
$$

Each equation $T + \langle { \underline { { w } } } , { \underline { { x } } } \rangle = 0$ defines a hyperplane of cardinality $Q ^ { n }$ in $\mathbb { F } ^ { n + 1 }$ <sup>1</sup>, so

$$
| \big ( \prod _ { \underline { { x } } \in X } c _ { j ( f , \underline { { x } } ) } ( \chi _ { 0 } ) \big ) \cdot \sum _ { \underline { { w } } \in \mathbb { F } ^ { n } } \sum _ { T \in \mathbb { F } } \prod _ { \underline { { x } } \in X } \chi _ { 0 } ( T + \langle \underline { { x } } , \underline { { w } } \rangle ) - \frac { Q ^ { n + 1 } } { q ^ { N } } | \leq C _ { N } ^ { \prime } Q ^ { n }
$$

for some constant $C _ { N } ^ { \prime }$ depending only on N. All remaining character tuples satisfy the conditions of Lemma 2.4, so the latter combined with Lemma 2.3 bounds the absolute value of their total contribution to the right hand side of (2.6.2) by

$$
c _ { N } Q ^ { n + 1 / 2 } \prod _ { \underline { { x } } \in X } \left( \sum _ { \chi \in \widehat { \mathbb { F } } ^ { * } } | c _ { j ( f , \underline { { x } } ) } ( \chi ) | \right) \leq C _ { N } ^ { \prime \prime } Q ^ { n + 1 / 2 } ( 1 + \log Q ) ^ { N }
$$

for some constant $C _ { N } ^ { \prime \prime }$ depending only on N. Now (1.5.1) is clear. □

## 3. The universality of periodic complex neurons and equidistribution of maps

We will start with the following easy

Lemma 3.1. Let $Z \subset \mathbb { C } ^ { n }$ be finite. Then there is a vector $\underline { w } \in \mathbb { C } ^ { n }$ such that the numbers $\langle { \underline { { w } } } , { \underline { { x } } } \rangle$ are pairwise distinct $a s \ \underline { { x } }$ ranges over $Z$

Proof. For every $\underline { { x } } , y \in Z$ such that $\underline { { x } } \neq y$ the equation $\langle \underline { { x } } , \underline { { w } } \rangle = \langle y , \underline { { w } } \rangle$ defines a proper complex hyperplane in $\mathbf { \bar { \mathbb { C } } } ^ { n }$ . A finite union of proper hyperplanes cannot cover $\mathbb { C } ^ { n }$ , so there is $\textrm { a } \underline { w }$ outside their union with the required property. □

Notation 3.2. Now we are going to use the notation of Definition 1.7 and Theorem 1.8, and fix a vector $\underline { w }$ as in Lemma 3.1 above for $Z = X \subset \mathbb { C } ^ { n }$ . Write $a _ { \underline { { x } } } ~ = ~ \langle \underline { { x } } , \underline { { w } } \rangle$ for every $\underline { { x } } \in X$ and choose an open disk $U \subset \mathbb { C }$ whose closure does not intersect the finite set $\{ - a _ { \underline { { x } } } \mid$ $\underline { { x } } \in X \}$ . On U for each $\underline { { x } } \in X$ choose a real-analytic branch $\theta _ { \underline { { x } } }$ of $\begin{array} { r } { b \mapsto \frac { 1 } { 2 \pi } \arg ( b + a _ { \underline { { x } } } ) } \end{array}$ , and set

$$
\begin{array} { r l r } { \Theta : U \longrightarrow \mathbb { R } ^ { X } , } & { { } } & { \Theta ( b ) = ( \theta _ { \underline { { x } } } ( b ) ) _ { \underline { { x } } \in X } . } \end{array}\tag{3.2.1}
$$

For every $\underline { { h } } = ( h _ { \underline { { x } } } ) _ { \underline { { x } } \in X } \in \mathbb { Z } ^ { X }$ let $\phi _ { \underline { { h } } } : U \to \mathbb { R }$ denote the real-analytic function given by the rule:

$$
b \longmapsto \langle \underline { { h } } , \Theta ( b ) \rangle = \sum _ { \underline { { x } } \in X } h _ { \underline { { x } } } \theta _ { \underline { { x } } } ( b ) .
$$

Lemma 3.3. For every nonzero $\underline { { h } } \in \mathbb { Z } ^ { X }$ the function $\phi _ { \underline { { h } } }$ is not constant on $U$

Proof. First note that the function

$$
R _ { \underline { { h } } } ( b ) = \prod _ { \underline { { x } } \in X } ( b + a _ { \underline { { x } } } ) ^ { h _ { \underline { { x } } } }
$$

is well-defined and nonzero everywhere on $U ,$ , since $- a _ { \underline { { x } } } \notin U$ for every $\underline { { x } } \in X$ . Now assume that $\phi _ { \underline { { h } } }$ is constant. Then $R _ { \underline { { h } } }$ has constant argument on $U$ , so by the open mapping theorem $R _ { \underline { { h } } }$ is constant. Therefore its logarithmic derivative:

$$
\frac { R _ { \underline { { h } } } ^ { \prime } ( b ) } { R _ { \underline { { h } } } ( b ) } = \sum _ { \underline { { x } } \in X } \frac { h _ { \underline { { x } } } } { b + a _ { \underline { { x } } } }\tag{3.3.1}
$$

vanishes on $U .$ . This is a rational function, so it vanishes identically by the unique continuation principle. Since the $a _ { \underline { { x } } }$ are distinct, the residue of this logarithmic derivative at $- a _ { x }$ is $h _ { \underline { { x } } }$ . Hence $h _ { x } = 0$ for every $\underline { { x } } \in X$ which is a contradiction. □

Definition 3.4. Let $\mathbb { T } ^ { X }$ denote the quotient torus $\mathbb { R } ^ { X } / \mathbb { Z } ^ { X }$ , and let mod : $\mathbb { R } ^ { X } \to \mathbb { T } ^ { X }$ denote the quotient map. Let $\mu$ denote the usual planar measure (or area) on subsets of $\mathbb { C } .$ , and let $\psi : U \to \mathbb { R }$ be a smooth, compactly supported and nonnegative function such that $\textstyle \int _ { U } \psi d \mu = 1$ . For every positive integer ℓ let $\mu _ { \ell }$ denote the measure on $\bar { \mathbb { T } } ^ { X }$ which for every continuous $F : \mathbb { T } ^ { X } \to \mathbb { C }$ is given by the formula:

$$
\int _ { \mathbb { T } ^ { X } } F ( \underline { { y } } ) d \mu _ { \ell } ( \underline { { y } } ) = \int _ { U } F \big ( \mathrm { m o d } ( \ell \Theta ( b ) ) \big ) \psi ( b ) d \mu ( b ) .
$$

Lemma 3.5. The measures $\mu _ { \ell }$ converge weakly to the Haar probability measure on $\mathbb { T } ^ { X }$ as $\ell \to \infty$

Proof. For every $\underline { { h } } \in \mathbb { Z } ^ { X }$ let $\widehat { \mu } _ { \ell } ( \underline { { h } } )$ denote the integral:

$$
\int _ { \mathbb { T } ^ { X } } e ^ { 2 \pi \mathbf { i } \langle \underline { { h } } , \underline { { y } } \rangle } d \mu _ { \ell } ( \underline { { y } } ) = \int _ { U } e ^ { 2 \pi \mathbf { i } \ell \langle \underline { { h } } , \Theta ( b ) \rangle } \psi ( b ) d \mu ( b ) .
$$

When $\underline { { h } } = 0$ we get that

$$
\widehat { \mu } _ { \ell } ( \underline { { { 0 } } } ) = \int _ { U } \psi d \mu = 1
$$

by our assumptions on $\psi .$ . Hence it will be suficient to show that $\widehat { \mu } _ { \ell } ( \underline { { h } } ) \to 0$ as $\ell \to \infty$ for every non-zero $\underline h$ by the Fourier criterion for weak convergence of measures on a torus.

By Lemma 3.3 the real-analytic function $\phi _ { \underline { { h } } }$ is not constant on the disk U. Its critical set

$$
C _ { \underline { { h } } } = \{ b \in U \mid \nabla \phi _ { \underline { { h } } } ( b ) = 0 \}
$$

has planar measure zero. Indeed, if both first partial derivatives of $\phi _ { \underline { { h } } }$ vanished identically, then $\phi _ { \underline { { h } } }$ would be constant because $U$ is connected. Hence at least one of these partial derivatives is a nonzero real-analytic function. The zero set of a nonzero real-analytic function has planar measure zero, and $C _ { \underline { { h } } }$ is contained in that zero set.

Let $K = \operatorname { s u p p } ( \psi )$ , which is a compact subset of $U _ { : }$ and let $\varepsilon > 0$ Since $C _ { \underline { { h } } }$ has measure zero, there is an open set $V$ whose closure lies in $U$ and which contains $C _ { \underline { { h } } } \cap K$ such that

$$
\int _ { V } \psi ( b ) d \mu ( b ) < \varepsilon .
$$

Choose a smooth, compactly supported function $\eta : U \to \mathbb { R }$ with $0 \leq \eta \leq 1$ such that η vanishes on a neighborhood of $C _ { \underline { { h } } } \cap K$ and $\eta = 1$ on $K \backslash V$ . Then

$$
\left| \int _ { U } e ^ { 2 \pi i \ell \phi _ { \underline { { h } } } ( b ) } ( 1 - \eta ( b ) ) \psi ( b ) d \mu ( b ) \right| \le \int _ { V } \psi ( b ) d \mu ( b ) < \varepsilon .
$$

Set $a \ = \ \eta \psi$ , and for this calculation write $\phi ~ = ~ \phi _ { h }$ and $\lambda \ = \ 2 \pi \ell .$ Since the compact set $\operatorname { s u p p } ( a )$ is disjoint from $C _ { \underline { { h } } }$ , the function $| \nabla \phi |$ is bounded below by a positive constant on a neighborhood of this set. On that neighborhood define

$$
W = W _ { \frac { h } { \rho } } = { \frac { \nabla \phi } { | \nabla \phi | ^ { 2 } } } .
$$

Here gradients and divergences are taken with respect to the two real coordinates of complex number and we will let · denote the corresponding real scalar product. The vector field $a W$ extends by zero to a smooth, compactly supported vector field on $U$

Choose an open disk D containing supp(a) such that ${ \overline { { D } } } \subset U$ , and let n denote the outward unit normal on $\partial D$ . Green’s theorem in divergence form, applied separately to the real and imaginary parts, gives

$$
\int _ { D } \operatorname { d i v } \left( e ^ { i \lambda \phi } a W \right) d \mu = \int _ { \partial D } e ^ { i \lambda \phi } a W \cdot \mathbf { n } d s = 0 ,
$$

because aW vanishes near ∂D. The product rule and the identity $\boldsymbol { W } \cdot \nabla \phi = 1$ on a neighborhood of $\operatorname { s u p p } ( a )$ imply that

$$
\begin{array} { r l } & { \mathrm { d i v } \big ( e ^ { i \lambda \phi } a W \big ) = \nabla \big ( e ^ { i \lambda \phi } \big ) \cdot ( a W ) + e ^ { i \lambda \phi } \mathrm { d i v } ( a W ) } \\ & { \qquad = i \lambda e ^ { i \lambda \phi } a + e ^ { i \lambda \phi } \mathrm { d i v } ( a W ) . } \end{array}
$$

Integrating this identity and rearranging therefore gives

$$
\int _ { U } e ^ { i \lambda \phi } a d \mu = - { \frac { 1 } { i \lambda } } \int _ { U } e ^ { i \lambda \phi } \mathrm { d i v } ( a W ) d \mu .
$$

Here the integrals over $D$ may be replaced by integrals over U because the integrands are supported in $\operatorname { s u p p } ( a ) \subset D$ . Taking absolute values,

applying the triangle inequality for integrals, and using $| e ^ { i \lambda \phi } | = 1$ , we obtain

$$
\begin{array} { r l } & { \left| \displaystyle \int _ { U } e ^ { 2 \pi i \ell \phi _ { \underline { { h } } } } \eta \psi d \mu \right| = \frac { 1 } { 2 \pi \ell } \left| \displaystyle \int _ { U } e ^ { i \lambda \phi } \mathrm { d i v } ( a W ) d \mu \right| } \\ & { \quad \quad \quad \le \frac { 1 } { 2 \pi \ell } \displaystyle \int _ { U } \left| \mathrm { d i v } ( a W ) \right| d \mu } \\ & { \quad \quad \quad = \frac { C _ { \underline { { h } } , V , \eta , \psi } } { \ell } , } \end{array}
$$

where explicitly

$$
C _ { \underline { { h } } , V , \eta , \psi } = \frac { 1 } { 2 \pi } \int _ { U } \left| \mathrm { d i v } ( \eta \psi W _ { \underline { { h } } } ) \right| d \mu .
$$

This constant is finite because aW is smooth and compactly supported, and it is independent of ℓ once $\underline { { h } } , V , \eta .$ , ψ are fixed. Combining these two estimates we get that

$$
\operatorname* { l i m } _ { \ell \to \infty } | { \widehat { \mu } } _ { \ell } ( { \underline { { h } } } ) | \leq \varepsilon .
$$

Since $\varepsilon > 0$ was arbitrary we get that $\widehat { \mu } _ { \ell } ( \underline { { h } } ) \to 0 .$

Proof of Theorem 1.8. We continue to use the notation introduced in this section so far. By slight abuse of notation for every congruence class a mod q let the same symbol a denote its unique representative in $\{ 0 , 1 , \ldots , q - 1 \}$ . For every function $f : X \to Y$ let $B _ { f } \subset \mathbb { T } ^ { X }$ denote the image of the open box

$$
\prod _ { x \in X } \left( { \frac { f ( { \underline { { x } } } ) } { q } } , { \frac { f ( { \underline { { x } } } ) + 1 } { q } } \right) \subset \mathbb { R } ^ { X }
$$

under the quotient map mod : $\mathbb { R } ^ { X } \to \mathbb { T } ^ { X }$ . The set $B _ { f }$ has Haar measure $q ^ { - | X | }$ and its boundary has Haar measure zero, so by Lemma 3.5 we get that

$$
\mu _ { \ell } ( B _ { f } ) \longrightarrow q ^ { - | X | } .
$$

In particular $\mu _ { \ell } ( B _ { f } ) > 0$ for every suficiently large ℓ. Since there are only finitely many functions from X to Y , we get that there is an $\ell _ { 0 }$ such that

$$
\mu _ { \ell } ( B _ { f } ) > 0 \qquad ( \forall \ell \geq \ell _ { 0 } )
$$

for every $f$ as above.

Now let $\ell \geq \ell _ { 0 }$ and f be arbitrary. By the above there is a $b \in U$ such that mod $( \ell \Theta ( b ) ) \in B _ { f }$ . Since $b \neq - a _ { x }$ we get that $b + a _ { \underline { { x } } }$ is nonzero for every $\underline { { x } } \in X$ . Moreover $b + a _ { \underline { { x } } } \notin \mathcal { R } _ { q , \ell }$ and

$$
\Phi _ { q , \ell } ( b + a _ { \underline { { x } } } ) = f ( \underline { { x } } ) \qquad ( \forall \underline { { x } } \in X ) .
$$

Since $a _ { \underline { { x } } } = \langle \underline { { w } } , \underline { { x } } \rangle$ we get that (w, b) realises f.

## References

[1] N. N. Aizenberg, Yu. L. Ivas’kiv and D. A. Pospelov, Ob odnom obobshchenii porogovykh funktsii [On one generalization of threshold functions], Dokl. Akad. Nauk SSSR 196 (1971), no. 6, 1287–1290 (in Russian).

[2] N. N. Aizenberg, Yu. L. Ivas’kiv, D. A. Pospelov and G. F. Khudyakov, Multivalued threshold functions. I. Boolean complex-threshold functions and their generalization, Cybernetics and Systems Analysis 7 (1971), 626–635; translated from Kibernetika, no. 4 (1971), 44–51.

[3] N. N. Aizenberg, Yu. L. Ivas’kiv, D. A. Pospelov and G. F. Khudyakov, Multivalued threshold functions. II. Synthesis of multivalued threshold elements, Cybernetics and Systems Analysis 9 (1973), 61–77; translated from Kibernetika, no. 1 (1973), 53–66.

[4] N. N. Aizenberg and Yu. L. Ivas’kiv, Mnogoznachnaya porogovaya logika [Multivalued threshold logic], Naukova Dumka, Kiev, 1977 (in Russian).

[5] I. N. Aizenberg, A universal logic element over the complex field, Cybernetics and Systems Analysis 27 (1991), 467–473; translated from Kibernetika, no. 3 (1991), 116–121.

[6] I. Aizenberg, A multi-valued neuron with a periodic activation function, in: Proceedings of the International Joint Conference on Computational Intelligence, Funchal, Madeira, 2009, 347–354.

[7] I. Aizenberg, Learning of the non-threshold functions of multiple-valued logic by a single multi-valued neuron with a periodic activation function, in: Proceedings of the 40th IEEE International Symposium on Multiple-Valued Logic, IEEE, 2010, 33–38.

[8] I. Aizenberg, Periodic activation function and a modified learning algorithm for the multivalued neuron, IEEE Trans. Neural Networks 21 (2010), 1939–1949.

[9] V. Bovdi and V. Laver, Thelma: THreshold ELements, Modeling and Applications, version 1.3, GAP package, 2022.

[10] H. Davenport and P. Erd˝os, The distribution of quadratic and higher residues, Publ. Math. Debrecen 2 (1952), 252–265.

[11] F. E. Geche, Neural elements over finite fields, Information Technologies and Systems 1 (1998), no. 1–2, 100–104.

[12] F. E. Geche, Analysis of discrete functions and logical schemes synthesis in neurobasis, Uzhhorod National University, Uzhhorod, 2010 (in Ukrainian).

[13] F. Geche, O. Mulesa, A. Batyuk and V. Voloshchuk, Properties of logical functions implemented by one generalized neural element over the Galois field, in: Advances in Intelligent Systems and Computing IV, Advances in Intelligent Systems and Computing 1080, Springer, Cham, 2020, 202–213.

[14] B. Kra, N. A. Shah and W. Sun, Equidistribution of dilated curves on nilmanifolds, arXiv:1712.02907v2, 2018.

[15] R. Lidl and H. Niederreiter, Finite fields, second edition, Encyclopedia of Mathematics and its Applications 20, Cambridge University Press, Cambridge, 1997.

[16] B. McDonald, A. Sahay and E. L. Wyman, The VC-dimension of quadratic residues in finite fields, arXiv:2210.03789v2, 2024.

[17] R. Peralta, On the distribution of quadratic residues and nonresidues modulo a prime number, Math. Comp. 58 (1992), 433–440.

[18] B. Randol, The behavior under projection of dilating sets in a covering space, Trans. Amer. Math. Soc. 285 (1984), no. 2, 855–859.

[19] D. Yarotsky, Elementary superexpressive activations, in: Proceedings of the 38th International Conference on Machine Learning, Proc. Mach. Learn. Res. 139 (2021), 11932–11940.

Department of Mathematical Sciences<sub>,</sub> UAEU<sub>,</sub> UAE

Email address: vbovdi@gmail.com

Mathematical Institute<sub>,</sub> Eotv¨ os Lor¨ and University 1117 Budapest<sub>,</sub>´ Pazm<sup>´</sup> any P<sup>´</sup> eter s<sup>´</sup> et<sup>´</sup> any 1/C, Hungary;<sup>´</sup> Department of Mathematical Sciences<sub>,</sub> UAEU<sub>,</sub> UAE Email address: ambrus.pal@uaeu.ac.ae