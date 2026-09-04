# What is Smoothness?

Zachary P. Bradshaw<sup>∗</sup>

QodeX Quantum

## Abstract

Smoothness of a function on the real line is reflected in the decay of its Fourier transform, which suggests that smoothness of a function in $L ^ { 2 } ( G )$ for a group G should mean concentration of the Fourier coeficients at low frequency. Such a reading presupposes an ordering of the irreducible representations of $G ,$ but for non-abelian G, no ordering is canonical. Given a symmetric generating set $S ,$ the Laplacian of the associated Cayley graph is block diagonal over the dual, and we order the irreps by the mean of the eigenvalues in each block. This produces an ordering function ω : $\widehat { G } $ R that depends only on the pair (G, S). This function is bounded between zero and two, vanishing only at the trivial representation and achieving the upper bound exactly when the Cayley graph is bipartite. We then ask how much freedom the construction has. Within the class of operators that are Hermitian, left invariant, annihilate the constant functions, and assign an energy insensitive to complex conjugation, the induced orderings are exactly the real functions on the dual vanishing at the trivial representation and agreeing on conjugate pairs, and the orderings coming from inversion orbits of conjugacy classes form a basis for them. We cut the freedom down further by requiring two additional inputs: nonnegativity of the class weights and a declaration of which group elements count as uniform incremental changes, which pins the operator to the Cayley-Laplacian up to positive scale. We observe that the construction persists for compact groups even though the Cayley graph does not, and we extend the theory to finite sets carrying a transitive group action, where the acting group selects which frequencies exist and the generating set orders them. The answer to the title question is therefore that smoothness is a property of a function together with a choice of group and generating set, not of the function alone.

## 1 Introduction

The canonical notion of smoothness taught in a single variable calculus course is given by a function which is continuously diferentiable to any order. By applying integration by parts repeatedly, it can be shown that a smooth function with derivatives in $L ^ { 1 } ( \mathbb { R } )$ for any order m has a Fourier transform that decays superpolynomially. However, the converse is false; the decay of the Fourier transform does not fully characterize the space of smooth functions. The Fourier transform does, however, preserve the closely related Schwartz space. Indeed, denoting the Schwartz space by S(R), one finds that $f \in S ( \mathbb { R } )$ if and only if ${ \widehat { f } } \in { \mathcal { S } } ( \mathbb { R } )$ making Schwartz functions the natural function space to do Fourier analysis.

This result motivates the perspective of Belis et al. [1] that smoothness on $L ^ { 2 } ( G )$ for an arbitrary group G should be interpreted as a bias in which higher order frequencies rapidly decay [2, 3]. In their work, they construct a generalized set of “frequencies” from the irreducible representations (irreps) of $G$ which can be low-pass filtered to induce smoothness on a data distribution. For abelian groups, such as $\mathbb { Z } _ { 2 } ^ { n }$ , a natural ordering of the irreps is available in the form of the Hamming weight; the irreps are labeled by the elements of $\mathbb { Z } _ { 2 } ^ { n }$ , and the elements which contain more zeros are interpreted as smoother than those with fewer. This ordering makes great sense when one Fourier expands a function $f : \mathbb { Z } _ { 2 } ^ { n } \to \mathbb { C }$ , producing

$$
f ( x ) = \sum _ { k \in \mathbb { Z } _ { 2 } ^ { n } } { \widehat { f } } ( k ) ( - 1 ) ^ { k \cdot x } .\tag{1}
$$

The dependence on x shows up entirely in the sign function $( - 1 ) ^ { k \cdot x }$ , and when a component $k _ { i }$ of $k$ is zero, the corresponding component $x _ { i }$ of $x$ does not afect the value of $f .$ Thus, those k with high Hamming weight have the most potential to change $f ,$ validating the perspective that such k should be interpreted as higher frequency.

This perspective is perfectly natural, but non-abelian groups introduce an added complexity: the ordering of the irreps is no longer obvious, and the problem is exacerbated by the fact that non-abelian groups have irreps of dimension greater than one. In this work, we show that such an ordering does exist, and the mathematics used to obtain it has existed for decades in the works of Diaconis [4], Babai [5], and Terras [6]. Already, an ordering has been constructed for the irreps of $S _ { n }$ in the context of random walks, where the object of interest is not the ordering itself, but the spectral gap question of which nontrivial irrep sits lowest [7, 8]. The same eigenvalues appear in the study of fitness landscapes, where the energy of a function across the Laplacian eigenspaces of a Cayley graph is called its amplitude spectrum and is read as a measure of ruggedness [9, 10, 11], and in graph signal processing on Cayley graphs, where the block structure over the dual is used to select a Fourier basis or a frame rather than to compare one generating set against another [12, 13, 14]. The same spectral data organizes stationary kernels on groups and their homogeneous spaces [15]. Here we use this pre-existing spectral machinery to extract an irrep-level ordering for arbitrary finite groups, followed by an extension to compact groups, using tools from graph signal processing, in which frequencies are characterized by Laplacian spectra [16, 17, 18].

Crucially, our notion of smoothness depends on the choice of generating set for the group of interest. This choice serves as a declaration of which elements of the group count as a small perturbation, and this structure is exhibited by the corresponding Cayley graph, in which group elements that difer by few elements of the generating set are closer together. The quadratic form of the Laplacian of this graph quantifies how much a given function changes when its argument is replaced by the group elements adjacent to it, thereby encoding a natural notion of smoothness. For this reason, we block diagonalize the Laplacian and associate a scalar to each block and therefore each irrep, and this scalar encodes the ordering of the irreps.

Concretely, we define the ordering parameter of an irrep σ to be the mean of the eigenvalues of the Laplacian on the σ-isotypic Peter-Weyl block $A _ { \sigma }$ , which takes the form $\omega ( \sigma ) = 1 - \mathrm { T r } [ M _ { \sigma } ] / d _ { \sigma }$ , where $M _ { \sigma }$ is the average of σ over the generating set. This definition requires no hypothesis on the generating set beyond symmetry, but it is common in the literature on random walks to also assume that the generating set is closed under conjugation. In this case, Schur’s lemma collapses each block to a scalar, and the ordering parameter is determined by the normalized character sum. Defining the ordering parameter as a mean of block eigenvalues retains a single number per irrep; although, it is worth noting that the ordering on $\widehat { G }$ is coarser than the ordering on the spectrum of the Laplacian itself.

A construction of this kind invites the objection that the Laplacian was an arbitrary starting point. To address this concern, we isolate four properties that any operator serving as a measure of smoothness should have, namely that it be Hermitian, that it commute with all left translations, that it annihilate the constant functions, and that the energy it assigns be unchanged under the complex conjugation of the argument. We then show that these properties alone do not narrow the ordering down; every real function on $\widehat { G }$ vanishing at the trivial representation and agreeing on conjugate pairs is realized. However, the ordering $\omega _ { K }$ associated to inversion-closed unions $K = { \bar { C ^ { \cup } } } C ^ { - { \bar { 1 } } }$ of conjugacy classes form a basis so that what the axioms provide is a coordinate system, in which an ordering is a weight vector recording which conjugacy classes count as small steps and how much they count. Two further requirements do cut the freedom down. Nonnegativity of that weight vector produces a positive semidefinite weighted Cayley-Laplacian, and declaring in addition that the elements of a fixed generating set are the incremental changes and that none counts for more than another pins the operator to the Cayley-Laplacian up to positive scale. In this sense, the Cayley-Laplacian is not one choice among many, and it is the axioms about the generating set rather than those about the operator that make it so.

We additionally show how to remove the assumption that the domain of the function is a group. For a function defined on a finite set that carries a transitive group action, the domain is identified with a coset space, and we lift the function to a function on the group. The theory developed for functions on groups then applies to the lifted functions. Here, a second layer of choice arises. The acting group determines which frequencies of the group are accessible since an irrep contributes to the function space precisely when it contains a vector fixed by the stabilizer. Meanwhile, the generating set orders whatever frequencies survive. These two choices are independent, and we exhibit pairs of groups acting on the same set for which the resulting notions of smoothness substantially difer. Finally, we observe that finiteness is used in only one essential place, namely in drawing the Cayley graph, and that the ordering parameter and its basic properties

persist for any compact group.

The remainder of the paper is organized as follows. Section 2 develops the construction for finite groups, establishes the block diagonalization of the Laplacian, defines the ordering parameter, and characterizes the irreps attaining its extreme values. Section 3 contains the characterization of admissible operators and their induced orderings. Section 4 extends the theory to sets carrying a group action and determines which frequencies survive the descent. Section 5 works through several examples, with figures illustrating how a single function changes its apparent smoothness when the generating set is changed. We use a final example to demonstrate how to interpret smoothness as an inductive bias and showcase how a designer might choose the various parameters of the smoothness model in Section 6. Section 7 treats compact groups, and Section 8 gives concluding remarks.

## 2 Smoothness on Finite Groups

Given a finite group G, the space of functions $f : G \to \mathbb { C }$ is denoted $L ^ { 2 } ( G )$ , and it carries an inner product

$$
\langle f , h \rangle = { \frac { 1 } { | G | } } \sum _ { g \in G } { \overline { { f ( g ) } } } h ( g ) .\tag{2}
$$

This space contains the functions that we study here. Our goal is to introduce a notion of smoothness on G which is analogous to that on R. An orthonormal basis for this space is given by the normalized delta functions $\sqrt { | G | } \delta _ { g }$ defined by $\delta _ { g } ( g ^ { \prime } ) = \delta _ { g , g ^ { \prime } }$ , where $\delta _ { g , g ^ { \prime } }$ denotes the Kronecker delta function which takes the value one if $g = g ^ { \prime }$ and zero otherwise. In this basis, a function is expanded as

$$
f = { \frac { 1 } { \sqrt { | G | } } } \sum _ { g \in G } f ( g ) { \sqrt { | G | } } \delta _ { g } .\tag{3}
$$

By the Peter-Weyl theorem [19], another orthonormal basis is given by the normalized matrix entries of the irreps of $G ,$ denoted $\sqrt { d _ { \sigma } } \sigma _ { i j }$ for irrep σ. The collection of all irreps is known as the dual ${ \widehat { G } } .$ When G is abelian, all irreps are one-dimensional and equivalent to the characters of $G ,$ which are generally defined by $\chi _ { \sigma } ( g ) = \operatorname { T r } [ \sigma ( g ) ]$ . When this is the case, the dual has the structure of a group [20], and although that structure is lost for non-abelian groups, the characters do still form a basis for the space of class functions. Note that every representation of a finite group is unitarizable, and so we make the assumption throughout this work that each irrep is unitary without loss of generality.

As noted in the introduction, smoothness on R is linked to the decay of the Fourier transform. In order to establish such a connection on $G ,$ , we define the Fourier transform on $L ^ { 2 } ( G )$ as a map ${ \mathcal { F } } _ { G } : L ^ { 2 } ( G ) \to$ $\textstyle \bigoplus _ { \sigma \in { \widehat { G } } } \operatorname { E n d } ( { \mathcal { H } } _ { \sigma } )$ defined by

$$
{ \widehat { f } } ( \sigma ) : = ( { \mathcal { F } } _ { G } f ) ( \sigma ) : = { \frac { { \sqrt { d _ { \sigma } } } } { | G | } } \sum _ { g \in G } f ( g ) { \overline { { \sigma ( g ) } } } .\tag{4}
$$

An analog of Parseval’s identity holds and reads $\textstyle \| f \| ^ { 2 } = \sum _ { \sigma \in { \widehat { G } } } \| { \widehat { f } } ( \sigma ) \| _ { F } ^ { 2 }$ , where $\| \cdot \| _ { F }$ denotes the Frobenius norm. Concentration of $\widehat { f }$ at low $\omega$ is therefore a statement about how the squared norm of $f$ distributes over ${ \widehat { G } } ,$ and we shall have more to say about this in the next section. When $G$ is abelian, the codomain is equivalent to $L ^ { 2 } ( { \widehat { G } } )$ , and the Fourier transform takes the form

$$
{ \widehat { f } } ( \chi ) = { \frac { 1 } { | G | } } \sum _ { g \in G } f ( g ) { \overline { { \chi ( g ) } } } ,\tag{5}
$$

where $\chi \in { \widehat { G } }$ is a character of G. In particular, when $G = \mathbb { Z } _ { d }$ , the characters are $\chi _ { k } ( x ) = e ^ { 2 \pi i k x / d }$ with $k , x \in  { \mathbb { Z } } _ { d }$ , and the Fourier transform takes the familiar form

$$
{ \widehat { f } } { \left( \chi _ { k } \right) } = { \frac { 1 } { | G | } } \sum _ { x \in G } f ( x ) e ^ { - 2 \pi i k x / d } .\tag{6}
$$

When $G = \mathbb { Z } _ { 2 } ^ { n }$ , the characters are $\chi _ { k } ( x ) = ( - 1 ) ^ { k \cdot x }$ with $k , x \in \mathbb { Z } _ { 2 } ^ { n }$ , and we recover (1).

As suggested by Belis et al. [1], we interpret a smooth function to be a function with Fourier coeficients concentrated at low frequency. However, this assertion presupposes an ordering on the Fourier coeficients, and for non-abelian groups, a natural choice is not particularly obvious. The case of abelian groups is fortunately simpler, and we return to the case $G = \mathbb { Z } _ { 2 } ^ { n }$ as an example. Observe that the character $( - 1 ) ^ { k \cdot x }$ depends only on the x coordinates for which the corresponding k coordinate is nonzero; any change in $x _ { i }$ for which $k _ { i } = 0$ does not afect the outcomes of the Fourier transform. Thus, the characters with more nonzero coordinates are more susceptible to a change in the function domain. This motivates the Hamming weight of k [21] as a natural choice for an ordering on ${ \widehat { G } } ,$ , so that smooth functions are those with Fourier coeficients concentrated in the low Hamming weight regime.

In order to define smoothness in an analogous way for a general finite group, let us now produce a method for equipping their Fourier coeficients with an ordering. To this end, let $G$ be a finite group and choose a generating set S. We assume that S is symmetric, meaning it is closed under inversion: $s ^ { - 1 } \in S$ for all $s \in S$ The Cayley graph associated to this choice of generating set is the graph with a vertex for every group element and edges between vertices whenever the group elements labeling them difer by a right multiplication by an element of S. The Cayley graph carries a natural ordering of the group elements in the form of a generalized Hamming weight given by the minimum number of generators needed to express a given group element. One recovers the Hamming weight by choosing $G = \mathbb { Z } _ { 2 } ^ { n }$ and $S = \{ e _ { i } \} _ { i = 1 } ^ { n }$ , where $e _ { i } = ( 0 , \ldots , 0 , 1 , 0 , \ldots , 0 )$ denotes the standard basis vector with a one in the i-th component and zero otherwise. The Cayley graph treats elements of the generating set S as the smallest possible perturbations of a group element, and so any notion of smoothness with respect to this structure should probe what happens to the value of a function on $G$ as its argument is perturbed by a generator. Thus, the notion of smoothness depends on the choice of generating set. We remark that this generalization appears in the study of symmetry-based quantum codes as a natural notion of distance [22], and it would be interesting to determine whether the Hamming weight circuits of Rethinasamy et al. [23] can be extended to compute it.

With a notion of incremental change placed on the group, we now measure the change of $f$ under a small perturbation in its argument. We define an averaging operator $A : L ^ { 2 } ( G ) \to L ^ { 2 } ( G )$ by

$$
( A f ) ( g ) = \frac { 1 } { | S | } \sum _ { s \in S } f ( g s ) ,\tag{7}
$$

which replaces the value of the function f at the given g with its average over all adjacent vertices in the Cayley graph. Then the Laplacian operator ${ \mathcal { L } } = I - A \left[ 5 , 2 4 \right]$ defines the average change in f over all adjacent vertices in the Cayley graph, equivalently all incremental changes with respect to the generalized Hamming weight. The reader familiar with the combinatorial Laplacian of a graph may wonder why the degree matrix does not appear in our definition. This is because we use the normalized Laplacian and the Cayley graph is |S|-regular, so that the normalized degree matrix becomes the identity operator. The following proposition makes the interpretation of $\mathcal { L }$ as a measure of change in $f$ precise.

Proposition 1. Let $f \in L ^ { 2 } ( G )$ . Then

$$
\langle f , \mathcal { L } f \rangle = \frac { 1 } { 2 | G | | S | } \sum _ { g \in G } \sum _ { s \in S } | f ( g ) - f ( g s ) | ^ { 2 } .\tag{8}
$$

Proof. We first show that A is Hermitian. Indeed,

$$
{ \begin{array} { r l } { \langle f , A h \rangle = { \frac { 1 } { | G | } } \displaystyle \sum _ { g \in G } { \overline { { f ( g ) } } } \displaystyle \sum _ { | S | } \displaystyle \sum _ { s \in S } h ( g s ) } \\ & { = { \frac { 1 } { | G | } } \displaystyle \frac { 1 } { | S | } \displaystyle \sum _ { s \in S } \sum _ { g \in G } { \overline { { f ( g ) } } } h ( g s ) } \\ & { = \displaystyle \frac { 1 } { | G | } \displaystyle \frac { 1 } { | S | } \displaystyle \sum _ { s \in S } \sum _ { g \in G } { \overline { { f ( g s ^ { - 1 } ) } } } h ( g ) } \\ & { = \displaystyle \frac { 1 } { | G | } \displaystyle \sum _ { g \in G } \left( \frac { 1 } { | S | } \displaystyle \sum _ { s \in S } { \overline { { f ( g s ^ { - 1 } ) } } } \right) h ( g ) , } \end{array} }\tag{9}
$$

and since S is symmetric, we may re-index the sum over $S ,$ producing

$$
\langle f , A h \rangle = \frac { 1 } { | G | } \sum _ { g \in G } \left( \frac { 1 } { | S | } \sum _ { s \in S } \overline { { f ( g s ) } } \right) h ( g ) = \langle A f , h \rangle .\tag{10}
$$

Thus, A is Hermitian.

We now prove the proposition. Expanding the right hand side, we have

$$
\begin{array} { r } { \displaystyle \frac { 1 } { 2 | G | | S | } \sum _ { g \in G } \sum _ { s \in S } \displaystyle \lvert f ( g ) - f ( g s ) \rvert ^ { 2 } = \displaystyle \frac { 1 } { 2 | G | | S | } \sum _ { g \in G } \sum _ { s \in S } \left( | f ( g ) | ^ { 2 } + | f ( g s ) | ^ { 2 } - 2 \operatorname { R e } ( \overline { { f ( g ) } } f ( g s ) ) \right) } \\ { = \displaystyle \frac { 1 } { 2 } \left( \frac { 1 } { | G | } \sum _ { g \in G } | f ( g ) | ^ { 2 } + \frac { 1 } { | G | } \sum _ { g \in G } | f ( g ) | ^ { 2 } - 2 \operatorname { R e } \langle f , A f \rangle \right) . } \end{array}\tag{11}
$$

Now using the fact that A is Hermitian, we have

$$
\operatorname { R e } \langle f , A f \rangle = \langle f , A f \rangle ,\tag{12}
$$

from which it follows that

$$
{ \frac { 1 } { 2 | G | | S | } } \sum _ { g \in G } \sum _ { s \in S } | f ( g ) - f ( g s ) | ^ { 2 } = \langle f , ( I - A ) f \rangle .\tag{13}
$$

Since ${ \mathcal { L } } = I - A$ , the proposition holds.

We complete the construction of our ordering by block diagonalizing the Laplacian in the irrep basis. The eigenvectors of the Laplacian determine the direction of change, and the eigenvalues determine the magnitude of that change. The larger eigenvalues therefore correspond to larger amounts of change. However, we are looking for a scalar in each block; that is, one for each irrep. For this reason, we order the irreps by the average of the eigenvalues in each block.

Proposition 2. The Laplacian $\mathcal { L }$ is block diagonal with a block for every irreducible representation $\sigma .$ Moreover, in each block, L acts as $I _ { d _ { \sigma } } \otimes ( I _ { d _ { \sigma } } - M _ { \sigma } )$ , where $\begin{array} { r } { M _ { \sigma } : = \frac { 1 } { | S | } \sum _ { s \in S } \sigma ( s ) } \end{array}$

Proof. Let σ be an irrep of G and observe that

$$
( A \sigma _ { i j } ) ( g ) = \frac { 1 } { | S | } \sum _ { s \in S } \sigma _ { i j } ( g s ) = \frac { 1 } { | S | } \sum _ { s \in S } \sum _ { k = 1 } ^ { d _ { \sigma } } \sigma _ { i k } ( g ) \sigma _ { k j } ( s ) = \sum _ { k = 1 } ^ { d _ { \sigma } } \sigma _ { i k } ( g ) \frac { 1 } { | S | } \sum _ { s \in S } \sigma _ { k j } ( s ) ,\tag{14}
$$

where in the second equality, we have used the fact $\sigma ( g s ) = \sigma ( g ) \sigma ( s )$ that $\sigma$ is a homomorphism. Defining $\begin{array} { r } { M _ { \sigma } : = \frac { 1 } { | S | } \sum _ { s \in S } \sigma ( s ) } \end{array}$ , we have

$$
( A \sigma _ { i j } ) ( g ) = \sum _ { k = 1 } ^ { d _ { \sigma } } \sigma _ { i k } ( g ) ( M _ { \sigma } ) _ { k , j } .\tag{15}
$$

That is, A preserves σ and $i ,$ but mixes along $j .$ In other words, A preserves the σ-isotypic Peter-Weyl block $\mathcal { A } _ { \sigma } : = \operatorname { s p a n } \{ \sigma _ { i j } \}$ and is therefore block diagonal with one block per irrep, and in each block, A acts as ${ { I } _ { d _ { \sigma } } } \otimes M _ { \sigma }$ . Noting that ${ \mathcal { L } } = I - A$ , we therefore have that L acts as $I _ { d _ { \sigma } } \otimes ( I _ { d _ { \sigma } } - M _ { \sigma } )$ in each block.

By Proposition 2, the eigenvalues of $\mathcal { L }$ in the block corresponding to irrep σ are determined by the eigenvalues of $M _ { \sigma }$ . Since the Laplacian acts on the σ-isotypic Peter-Weyl block $A _ { \sigma }$ as $I _ { d _ { \sigma } } \otimes ( I _ { d _ { \sigma } } - M _ { \sigma } )$ , it has up to $d _ { \sigma }$ distinct eigenvalues, each of multiplicity $d _ { \sigma }$ . Their magnitudes dictate the amount of change in $f$ with an incremental change in its argument, and so we order the irreps by the average of the eigenvalues in each block.

Definition 1. The constant $\begin{array} { r } { \omega ( \sigma ) : = 1 - \frac { 1 } { d _ { \sigma } } \operatorname { T r } [ M _ { \sigma } ] } \end{array}$ is called the ordering parameter of $\sigma .$

If $\sigma ^ { \prime } = U \sigma U ^ { \dag }$ is an equivalent unitary irrep, then $M _ { \sigma ^ { \prime } } = U M _ { \sigma } U ^ { \dagger }$ , so $\mathrm { T r } [ M _ { \sigma ^ { \prime } } ] = \mathrm { T r } [ M _ { \sigma } ]$ and $\omega ( { \sigma } ^ { \prime } ) = \omega ( { \sigma } )$ The ordering parameter therefore depends only on the equivalence class and is well defined as a function of ${ \widehat { G } } .$ . Since the eigenvalues of $M _ { \sigma }$ are in $[ - 1 , 1 ]$ (we show $M _ { \sigma }$ is Hermitian in Proposition 6), the ordering parameter is bounded by $0 \leq \omega ( \sigma ) \leq 2$ . Indeed, since σ is unitary,

$$
\| M _ { \sigma } \| \leq \frac { 1 } { | S | } \sum _ { s \in S } \| \sigma ( s ) \| = 1 ,\tag{16}
$$

and it follows that $\omega ( \sigma ) \in [ 0 , 2 ]$ . In fact, the lower bound is only achieved by the trivial representation, and we can characterize when the upper bound is achieved by the structure of the Cayley graph.

Proposition 3. Let σ be an irrep of G and let S denote a generating set. Then $\omega ( \sigma ) = 0$ if and only if σ is the trivial representation, and ma $\mathrm { x } _ { \sigma } \omega ( \sigma ) = 2$ if and only if the Cayley graph associated to $( G , S )$ is bipartite, in which case the unique maximizer is the associated sign character.

Proof. Let $\varepsilon \in \{ \pm 1 \}$ and suppose $\omega ( \sigma ) = 1 - \varepsilon$ . The mean of the eigenvalues of $M _ { \sigma }$ is given by ε, which is an endpoint of the interval $[ - 1 , 1 ]$ containing all eigenvalues. This can only be the case if $M _ { \sigma } = \varepsilon I _ { d _ { \sigma } }$ . Let v be a unit vector. Then by the unitarity of $\sigma ,$ we have

$$
\frac { 1 } { | { \cal S } | } \sum _ { s \in { \cal S } } \| ( \sigma ( s ) - \varepsilon I _ { d _ { \sigma } } ) v \| ^ { 2 } = 2 ( 1 - \varepsilon \langle v , M _ { \sigma } v \rangle ) = 0 .\tag{17}
$$

It follows that $\sigma ( s ) = \varepsilon I _ { d _ { \sigma } }$ for all $s \in S$ . Hence, $\sigma ( s _ { 1 } \cdot \cdot \cdot s _ { k } ) = \varepsilon ^ { k } I _ { d _ { \sigma } }$ , which means that every one-dimensional subspace is invariant, and so irreducibility forces $d _ { \sigma } = 1$

If $\varepsilon = 1$ , then $\sigma$ is trivial since S generates G. If $\varepsilon = - 1$ , then $\sigma : G  \{ \pm 1 \}$ is a character with $\sigma | _ { S } = - 1$ , so $H = \ker ( \sigma )$ has index two and misses S, i.e. the $\mathrm { g r a p h }$ is bipartite. Conversely, each case attains its bound. Indeed, the trivial character gives $M _ { \sigma } = 1$ , and for $( G , S )$ with a bipartite Cayley graph, the sign character of the bipartition gives $M _ { \sigma } = - 1$ . For the uniqueness, if $\sigma _ { 1 } , \sigma _ { 2 }$ are both −1 on $S ,$ then $\sigma _ { 1 } \sigma _ { 2 }$ is 1 on S. But S generates $G ,$ and so $\sigma _ { 1 } \sigma _ { 2 }$ is the trivial representation, establishing that $\sigma _ { 2 } = \sigma _ { 1 } ^ { - 1 } .$ But $\sigma _ { 1 }$ takes values in {±1}, so $\sigma _ { 1 } ^ { - 1 } = \sigma _ { 1 }$ and hence $\sigma _ { 2 } = \sigma _ { 1 }$ □

The ordering parameter is the average of the quadratic form of Proposition 1 over the block. Indeed, let $\{ f _ { a } \} _ { a = 1 } ^ { d _ { \sigma } ^ { 2 } }$ be an orthonormal basis of $A _ { \sigma }$ . Since L acts on $A _ { \sigma }$ as $I _ { d _ { \sigma } } \otimes ( I _ { d _ { \sigma } } - M _ { \sigma } )$ , its trace there is $d _ { \sigma } \operatorname { T r } [ I _ { d _ { \sigma } } - M _ { \sigma } ]$ , and dividing by dim $( A _ { \sigma } ) = d _ { \sigma } ^ { 2 }$ gives

$$
{ \frac { 1 } { d _ { \sigma } ^ { 2 } } } \sum _ { a = 1 } ^ { d _ { \sigma } ^ { 2 } } { \frac { 1 } { 2 | G | | S | } } \sum _ { g \in G } \sum _ { s \in S } | f _ { a } ( g ) - f _ { a } ( g s ) | ^ { 2 } = \omega ( \sigma ) .\tag{18}
$$

Equivalently, $\omega ( \sigma ) = \mathbb { E } \langle f , \mathcal { L } f \rangle$ for $f$ drawn uniformly from the unit sphere of $A _ { \sigma }$ . For an individual unit $f \in A _ { \sigma }$ , the energy $\langle f , \mathcal { L } f \rangle$ lies in $[ \lambda _ { \operatorname* { m i n } } ( I _ { d _ { \sigma } } - M _ { \sigma } ) , \lambda _ { \operatorname* { m a x } } ( I _ { d _ { \sigma } } - M _ { \sigma } ) ]$ , and the two agree with $\omega ( \sigma )$ for every $f$ exactly when the block is scalar, which by Lemma 1, is the case when $S$ is closed under conjugation.

Lemma 1. If S is closed under conjugation, then $I _ { d _ { \sigma } } \otimes M _ { \sigma } = c _ { \sigma } I _ { d _ { \sigma } ^ { 2 } }$ for some constant $c _ { \sigma }$

Proof. Since S is closed under conjugation, we have

$$
\sigma ( h ) M _ { \sigma } \sigma ( h ) ^ { - 1 } = \frac { 1 } { | S | } \sum _ { s \in S } \sigma ( h s h ^ { - 1 } ) = \frac { 1 } { | S | } \sum _ { s \in S } \sigma ( s ) = M _ { \sigma } .\tag{19}
$$

Then by Schur’s lemma, $I _ { d _ { \sigma } } \otimes M _ { \sigma } = c _ { \sigma } I _ { d _ { \sigma } ^ { 2 } }$ for some constant $c _ { \sigma }$

The reader may note that the choice of a mean to obtain a scalar from each block of the Laplacian seems arbitrary when one could just as well use the maximum eigenvalue or even the minimum. However, the normalized trace is the unique linear basis-independent scalar summary of a block normalized by $F ( I ) = 1$ ; every linear functional satisfying $F ( U B U ^ { \dagger } ) = F ( B )$ for all unitaries $U$ is a scalar multiple of the trace. As we will see, the linearity hypothesis is necessary for Corollary 2 and Theorem 1 to hold.

Importantly, computing ω never requires forming L as a $| G | \times | G |$ matrix. Since $\begin{array} { r } { \mathrm { T r } [ M _ { \sigma } ] = \frac { 1 } { | S | } \sum _ { s \in S } \chi _ { \sigma } ( s ) } \end{array}$ the ordering parameter at σ is determined by a sum of |S| character values. We stress that the character sum formula requires no hypothesis on S beyond symmetry. What closure under conjugation adds is that the block is scalar, so that the mean is a lossless summary of the block rather than a lossy one. For $S$ not closed under conjugation, the number $\omega ( \sigma )$ is still available, but it discards the spread of the block.

Given two choices $( G , S )$ and $( H , T )$ of groups and generating sets defining smooth structures, we can construct a new smooth structure on the product group. We do so in two distinct ways which difer in how we construct the generating set for the product group.

Proposition 4. Let $( G , S )$ and $( H , T )$ be pairs of groups and generating sets with ordering parameters ω<sub>G</sub> and $\omega _ { H }$ , respectively. Assume e $\notin S$ and $e \not \in T$ . Then the ordering parameter corresponding to the group $G \times H$ with generating set $( S \times \{ e \} ) \cup ( \{ e \} \times T )$ is given by

$$
\omega _ { G \times H } ( \sigma \otimes \rho ) = \frac { | { \cal S } | } { | { \cal S } | + | { \cal T } | } \omega _ { G } ( \sigma ) + \frac { | { \cal T } | } { | { \cal S } | + | { \cal T } | } \omega _ { H } ( \rho ) .\tag{20}
$$

Proof. The irreps of $G \times H$ are given by the tensor products of the irreps of $G$ and $H$ , which have dimension $d _ { \sigma \otimes \rho } = d _ { \sigma } d _ { \rho }$ . Fixing an irrep $\sigma \otimes \rho .$ , we have

$$
\begin{array} { l } { { \displaystyle \omega _ { G \times H } \big ( \sigma \otimes \rho \big ) = 1 - \frac { 1 } { d _ { \sigma } d _ { \rho } } \frac { 1 } { | S | + | T | } \left( \sum _ { s \in S } \chi _ { \sigma } ( s ) d _ { \rho } + \sum _ { t \in T } d _ { \sigma } \chi _ { \rho } ( t ) \right) } } \\ { { \displaystyle \qquad = 1 - \frac { | S | } { | S | + | T | } ( 1 - \omega _ { G } ( \sigma ) ) - \frac { | T | } { | S | + | T | } ( 1 - \omega _ { H } ( \rho ) ) } } \\ { { \displaystyle \qquad = \frac { | S | } { | S | + | T | } \omega _ { G } ( \sigma ) + \frac { | T | } { | S | + | T | } \omega _ { H } ( \rho ) . } } \end{array}\tag{21}
$$

Proposition 5. Let (G, S) and $( H , T )$ be pairs of groups and generating sets with ordering parameters ω<sub>G</sub> and $\omega _ { H } .$ , respectively. Then the ordering parameter corresponding to the group $G \times H$ with generating set $S \times T$ is given $b y$

$$
1 - \omega _ { G \times H } ( \sigma \otimes \rho ) = ( 1 - \omega _ { G } ( \sigma ) ) ( 1 - \omega _ { H } ( \rho ) ) .\tag{22}
$$

Proof. The irreps of $G \times H$ are given by the tensor products of the irreps of G and H, which have dimension $d _ { \sigma \otimes \rho } = d _ { \sigma } d _ { \rho }$ . Fixing an irrep $\sigma \otimes \rho ,$ , we have

$$
\begin{array} { r l } & { 1 - \omega _ { G \times H } ( \sigma \otimes \rho ) = \frac { 1 } { d _ { \sigma } d _ { \rho } | S | | T | } \displaystyle \sum _ { s \in S } \sum _ { t \in T } \chi _ { \sigma } ( s ) \chi _ { \rho } ( t ) } \\ & { \qquad = \left( \frac { 1 } { d _ { \sigma } | S | } \displaystyle \sum _ { s \in S } \chi _ { \sigma } ( s ) \right) \left( \frac { 1 } { d _ { \rho } | T | } \displaystyle \sum _ { t \in T } \chi _ { \rho } ( t ) \right) } \\ & { \qquad = ( 1 - \omega _ { G } ( \sigma ) ) ( 1 - \omega _ { H } ( \rho ) ) . } \end{array}\tag{23}
$$

The quantity $\omega ( \sigma )$ is not new as a number. When S is a union of conjugacy classes, the spectrum of A is classical, computed by Diaconis and Shahshahani in the study of random walks on finite groups [4, 7], and the block diagonalization of Proposition 2 is a standard result. Such generating sets are called quasi-abelian in the signal processing literature, and Proposition 2 without that hypothesis is the Laplacian form of the eigendecomposition of a weighted Cayley graph [11, 14]. Our use of this quantity difers in that we retain a value at every irrep and read it as an ordering, whereas a mixing time argument typical of the random walk literature extracts a single eigenvalue and discards the remainder. Ordering frequencies by a Laplacian eigenvalue is likewise routine in graph signal processing, where a low pass filter on a graph is by definition the suppression of the large eigenvalues of L [25], and it is the working definition of smoothness in landscape theory, where a correlated landscape is one whose amplitude spectrum sits on the low eigenvalues [26]. The ordering parameter we define is a specialization of this concept to Cayley graphs, for which the corresponding Laplacian is block diagonal over the dual. The block at irrep σ is determined by $M _ { \sigma }$ alone, and the eigenvalue ordering therefore descends to the dual ${ \widehat { G } } ,$ , a set that exists prior to any diagonalization. As a consequence, the ordering parameter is a function $\omega : { \widehat { G } } \to$ R that depends only on the pair (G, S). For a generic graph, the Laplacian spectrum is a sorted list of real numbers with no domain attached, whereas Proposition 2 delivers the spectrum already partitioned by ${ \widehat { G } } ,$ a set determined by $G$ alone. The ordering is therefore a function on a fixed domain, which allows the orderings induced by diferent generating sets to be compared pointwise.

Although the use of the Laplacian is natural due in part to its interpretation as a quantification of the average change in $f$ over all incremental steps in the Cayley graph, we have not yet established that this is the only such choice one can make to construct an ordering on ${ \widehat { G } } .$ . In fact, there is some leeway in this choice, and we now determine exactly how much.

## 3 Characterization of the Ordering

Call an operator $T : L ^ { 2 } ( G ) \to L ^ { 2 } ( G )$ admissible if it is Hermitian, commutes with all left translations, annihilates the constant functions, and satisfies $\langle { \overline { { f } } } , T { \overline { { f } } } \rangle = \langle f , T f \rangle$ for every $f \in L ^ { 2 } ( G )$ . These conditions are natural axioms for an operator replacing the Laplacian in our construction. Hermiticity makes each individual σ-block Hermitian so that its eigenvalues are real and their mean is a real number, left invariance says that no point of G is distinguished, annihilation of the constant functions records that constant functions are considered smooth, and the last condition asks that the energy assigned to a function not depend on an overall choice of complex conjugate, which the quadratic form of Proposition 1 manifestly satisfies. Given an admissible $T ,$ we write $\omega _ { T } ( \sigma )$ for the mean of the eigenvalues of T on the block $A _ { \sigma }$ , extending the definition of $\omega ( \sigma )$ from L to $T .$

Let K denote the collection of sets $C \cup C ^ { - 1 }$ as $C$ ranges over the nontrivial conjugacy classes of $G .$ Each $K \in \mathcal { K }$ is symmetric, so $\mathcal { L } _ { K } \mathrel { \mathop : } = I - A _ { K }$ is the Laplacian of the Cayley graph on G generated by K whenever $K$ generates, and is the Laplacian of a disconnected Cayley graph otherwise. We write $\omega _ { K }$ for the associated ordering parameter. Finally, for $\varphi \in L ^ { 2 } ( G )$ , let

$$
( T _ { \varphi } f ) ( g ) = \sum _ { h \in G } \varphi ( g ^ { - 1 } h ) f ( h )\tag{24}
$$

so that $\begin{array} { r } { \mathcal { L } = T _ { \delta _ { e } - \mu _ { S } } , } \end{array}$ , where $\mu _ { S }$ is the uniform probability measure on S. We begin by identifying the admissible operators concretely, which amounts to the standard observation that left invariance forces convolution [6].

Lemma 2. An operator T on $L ^ { 2 } ( G )$ is admissible if and only if $T = T _ { \varphi }$ for a real valued $\varphi$ satisfying $\varphi ( x ^ { - 1 } ) = \varphi ( x )$ and $\textstyle \sum _ { x \in G } \varphi ( x ) = 0$

Proof. Let T have matrix κ so that $\begin{array} { r } { ( T f ) ( g ) = \sum _ { h \in G } \kappa ( g , h ) f ( h ) } \end{array}$ , and write $( L _ { a } f ) ( g ) = f ( a ^ { - 1 } g )$ . Then

$$
( L _ { a } T f ) ( g ) = \sum _ { h \in G } \kappa ( a ^ { - 1 } g , h ) f ( h )\tag{25}
$$

and

$$
( T L _ { a } f ) ( g ) = \sum _ { h \in G } \kappa ( g , a h ) f ( h ) .\tag{26}
$$

Comparing (25) with (26) shows that T commutes with every left translation if and only if $\kappa ( a ^ { - 1 } g , h ) =$ $\kappa ( g , a h )$ for all $a , g , h \in G$ . Taking $a = g$ gives $\kappa ( g , g h ) = \kappa ( e , h )$ , so that $\kappa ( g , h ) = \varphi ( g ^ { - 1 } h ) $ with $\varphi ( x ) : =$ $\kappa ( e , x )$ , and hence $T = T _ { \varphi }$ . The converse is the same computation read backwards.

A direct computation gives $T _ { \varphi } ^ { \dag } = T _ { \varphi ^ { \ast } }$ , where $\varphi ^ { * } ( x ) = { \overline { { \varphi ( x ^ { - 1 } ) } } }$ , so $T _ { \varphi }$ is Hermitian if and only if $\varphi ( x ) =$ $\overline { { \varphi ( x ^ { - 1 } ) } }$ . Writing $\overline { T }$ for the operator with matrix ${ \overline { { \kappa } } } ,$ we have

$$
{ \overline { { \langle { \overline { { f } } } , T { \overline { { f } } } \rangle } } } = \langle f , { \overline { { T } } } f \rangle ,\tag{27}
$$

and Hermiticity makes both quadratic forms real. Thus, the fourth axiom of admissibility holds if and only if ${ \overline { { T } } } \ = \ T ;$ that is, if and only if $\varphi$ is real valued. Combining the two conditions, $\varphi$ is real and

satisfies $\varphi ( x ^ { - 1 } ) = \varphi ( x )$ . Finally, $\begin{array} { r } { T _ { \varphi } 1 = ( \sum _ { x } \varphi ( x ) ) 1 } \end{array}$ , so $T$ annihilates the constant functions if and only if $\textstyle \sum _ { x } \varphi ( x ) = 0$ □

Each $\mathcal { L } _ { K }$ is of this form since $\delta _ { e } - \mu _ { K }$ is real, is invariant under inversion due to symmetry of $K .$ , and has total mass zero. The same is true of $\mathcal { L }$ itself. Moreover, the block structure of Proposition 2 persists for every admissible operator.

Proposition 6. Let $T = T _ { \varphi }$ be admissible. Then T acts on $A _ { \sigma }$ as $I _ { d _ { \sigma } } \otimes M _ { \sigma } ^ { \varphi }$ , where $\begin{array} { r } { M _ { \sigma } ^ { \varphi } : = \sum _ { x \in G } \varphi ( x ) \sigma ( x ) } \end{array}$ is Hermitian, and

$$
\omega _ { T } ( \sigma ) = { \frac { 1 } { d _ { \sigma } } } \mathrm { T r } [ M _ { \sigma } ^ { \varphi } ] = { \frac { 1 } { d _ { \sigma } } } \sum _ { x \in G } \varphi ( x ) \chi _ { \sigma } ( x ) .\tag{28}
$$

Moreover, we have

$$
\omega _ { T _ { \varphi } } = \omega _ { T _ { \varphi ^ { \sharp } } } ,\tag{29}
$$

where $\begin{array} { r } { \varphi ^ { \natural } ( x ) = { \frac { 1 } { | G | } } \sum _ { a \in G } \varphi ( a x a ^ { - 1 } ) } \end{array}$

Proof. As in the proof of Proposition 2, we have

$$
( T _ { \varphi } \sigma _ { i j } ) ( g ) = \sum _ { x \in G } \varphi ( x ) \sigma _ { i j } ( g x ) = \sum _ { k = 1 } ^ { d _ { \sigma } } \sigma _ { i k } ( g ) ( M _ { \sigma } ^ { \varphi } ) _ { k j } ,\tag{30}
$$

so $T _ { \varphi }$ preserves $A _ { \sigma }$ and acts there as $I _ { d _ { \sigma } } \otimes M _ { \sigma } ^ { \varphi }$ . Hermiticity of the block follows from Lemma 2 and unitarity of σ since

$$
( M _ { \sigma } ^ { \varphi } ) ^ { \dagger } = \sum _ { x \in G } \varphi ( x ) \sigma ( x ) ^ { - 1 } = \sum _ { y \in G } \varphi ( y ^ { - 1 } ) \sigma ( y ) = M _ { \sigma } ^ { \varphi } .\tag{31}
$$

Taking the mean of the $d _ { \sigma }$ eigenvalues of the block gives (28). The final claim follows by averaging the substitution $x \mapsto a x a ^ { - 1 }$ over $a \in G$ in (28), using that $\chi _ { \sigma }$ is a class function. □

Taking $\varphi = \delta _ { e } - \mu _ { S }$ in (28) recovers $\omega ( \sigma ) = 1 - \mathrm { T r } [ M _ { \sigma } ] / d _ { \sigma } ,$ , so the ordering parameter of Proposition 2 is the case $T = { \mathcal { L } }$ . Proposition 6 also further justifies the choice of the mean in the definition of the ordering parameter because the operator $T _ { \varphi ^ { \sharp } }$ is diagonal in the irrep basis when $\varphi = \delta _ { e } - \mu _ { S }$ , as we show in the next corollary. Thus, an ordering induced by a non-diagonal Cayley-Laplacian is not just a lossy description of its spectrum; rather, it takes into account the exact spectrum of the conjugation-symmetrized weighted Cayley-Laplacian.

Corollary 1. Let $\varphi = \delta _ { e } - \mu _ { S }$ so that $T _ { \varphi } = { \mathcal { L } }$ . Then $\mathcal { L } ^ { \natural } : = T _ { \varphi ^ { \natural } }$ acts on $A _ { \sigma }$ as $\omega ( \sigma ) I _ { d _ { \sigma } ^ { 2 } }$

Proof. By Proposition $6 , T _ { \varphi ^ { \sharp } }$ ♮ acts on $A _ { \sigma }$ as ${ { I } _ { d _ { \sigma } } } \otimes M _ { \sigma }$ with

$$
M _ { \sigma } = \sum _ { x \in G } \varphi ^ { \natural } ( x ) \sigma ( x ) .\tag{32}
$$

Then for all $g \in G$ we have

$$
\sigma ( g ) M _ { \sigma } \sigma ( g ^ { - 1 } ) = \sum _ { x \in G } \varphi ^ { \sharp } ( x ) \sigma ( g x g ^ { - 1 } ) = \sum _ { x \in G } \varphi ^ { \sharp } ( g ^ { - 1 } x g ) \sigma ( x ) .\tag{33}
$$

But $\varphi ^ { \tt L }$ is a class function, and so $\sigma ( g ) M _ { \sigma } = M _ { \sigma } \sigma ( g )$ for all $g \in G$ . Thus, by Schur’s lemma, $M _ { \sigma } = c _ { \sigma } I$ for some constant $c _ { \sigma }$ . Since $\omega _ { T _ { \varphi } } = \omega _ { T _ { \varphi } ^ { \sharp } }$ , we have $c _ { \sigma } = \omega ( \sigma )$ □

Combining Proposition 6 and Corollary 1 shows that every ordering parameter comes from a diagonal admissible operator. We now show that no admissible operator produces an ordering outside the real span of those coming from the Laplacians $\mathcal { L } _ { K }$

Theorem 1. The Laplacians $\{ { \mathcal { L } } _ { K } \} _ { K \in K }$ are linearly independent, and $f o r$ every admissible $T ,$ there is an $\alpha \in \mathbb { R } ^ { | \mathcal { K } | }$ with

$$
\omega _ { T } = \sum _ { K \in \mathcal K } \alpha _ { K } \omega _ { K } .\tag{34}
$$

Proof. Let $T = T _ { \varphi }$ be admissible. By Lemma 2, the function $\varphi ^ { \tt L }$ is again real, inversion invariant, and of total mass zero, and it is in addition a class function, hence constant on $\{ e \}$ and on each member of K. Since these sets partition G,

$$
\varphi ^ { \natural } = \lambda _ { e } \delta _ { e } + \sum _ { K \in \mathcal K } \lambda _ { K } \mu _ { K }\tag{35}
$$

with $\lambda _ { e } , \lambda _ { K } \in \mathbb { R }$ , and vanishing total mass forces $\begin{array} { r } { \lambda _ { e } + \sum _ { K \in \mathcal { K } } \lambda _ { K } = 0 } \end{array}$ . Substituting $A _ { K } = I - { \mathcal { L } } _ { K }$ then gives

$$
T _ { \varphi ^ { \sharp } } = \left( \lambda _ { e } + \sum _ { K \in \mathcal { K } } \lambda _ { K } \right) I - \sum _ { K \in \mathcal { K } } \lambda _ { K } \mathcal { L } _ { K } = - \sum _ { K \in \mathcal { K } } \lambda _ { K } \mathcal { L } _ { K } ,\tag{36}
$$

and (34) follows from Proposition 6 with $\alpha _ { K } = - \lambda _ { K }$

Suppose now that $\begin{array} { r } { \sum _ { K } \alpha _ { K } \mathcal { L } _ { K } = 0 } \end{array}$ for some real α. Then $\textstyle \sum _ { K } \alpha _ { K } A _ { K } = ( \sum _ { K } \alpha _ { K } ) I $ , and comparing the associated functions gives $\textstyle \sum _ { K } \alpha _ { K } \mu _ { K } = ( \sum _ { K } \alpha _ { K } ) \delta _ { e }$ . Evaluating at any x $\neq e ,$ which lies in exactly one member K of K, gives $\alpha _ { K } / | \bar { K } | = 0$ . Hence, every $\alpha _ { K }$ vanishes, and the $\mathcal { L } _ { K }$ are linearly independent.

The theorem does not select a unique ordering. Rather, it identifies the $\omega _ { K }$ as a canonical basis for the space of orderings arising from admissible operators. The next result identifies exactly which orderings arise and shows that this space is unconstrained in any way that a candidate ordering would not already satisfy. Reality is forced by any construction taking eigenvalue means of a Hermitian operator, agreement on conjugate pairs by any construction not distinguishing a representation from its dual, and vanishing at the trivial representation is a normalization. Admissibility therefore supplies coordinates rather than a restriction, in which an ordering becomes a weight vector on K recording which conjugacy classes count as small steps and how much they count.

Theorem 2. The map $T \mapsto \omega _ { T }$ carries the admissible operators onto the real valued functions on $\widehat { G }$ which vanish at the trivial representation and agree on complex conjugate pairs. Its kernel is $\{ T _ { \varphi } : \varphi ^ { \natural } = 0 \}$ , and it restricts to a linear bijection on $\mathrm { s p a n } _ { \mathbb { R } } \{ \mathcal { L } _ { K } \}$ , which is exactly the set of admissible operators commuting with all right translations.

Proof. By Lemma 2, the assignment $\varphi \mapsto T _ { \varphi }$ is a bijection, so it sufices to study the map $\varphi \mapsto \omega _ { T _ { \varphi } }$ . By Proposition 6, this map factors as $\varphi  \varphi ^ { \natural } \mapsto \omega _ { T _ { \varphi ^ { \natural } } }$ , and by (28), its second stage sends a class function ψ to the function $\begin{array} { r } { \sigma \mapsto \frac { 1 } { d _ { \sigma } } \sum _ { x \in G } \psi ( x ) \chi _ { \sigma } ( x ) } \end{array}$ on ${ \widehat { G } } ;$ that is, to the coeficients of ψ in the character basis divided by $d _ { \sigma }$ . Since the irreducible characters form a basis for the class functions, that second stage is a linear bijection onto all functions from $\widehat { G }$ to C. Writing $\varphi ^ { \vee } ( x ) = \varphi ( x ^ { - 1 } )$ and using $\chi _ { \overline { { \sigma } } } = \overline { { \chi _ { \sigma } } }$ together with $\chi _ { \sigma } ( x ^ { - 1 } ) = { \overline { { \chi _ { \sigma } ( x ) } } }$ we have

$$
\omega _ { T _ { \varphi } } ( \sigma ) = \overline { { \omega _ { T _ { \varphi } } ( \overline { { \sigma } } ) } }\tag{37}
$$

and

$$
\omega _ { T _ { \varphi } \vee } ( \sigma ) = \omega _ { T _ { \varphi } } ( \overline { { \sigma } } ) ,\tag{38}
$$

and $\begin{array} { r } { \omega _ { T _ { \varphi } } ( \sigma _ { \mathrm { t r i v } } ) = \sum _ { x \in G } \varphi ( x ) } \end{array}$ . Hence, $\varphi ^ { \tt L }$ is real if and only if $\omega _ { T } ( \sigma ) = \overline { { \omega _ { T } ( \overline { { \sigma } } ) } }$ , it is inversion invariant if and only if $\omega _ { T } ( { \overline { { \sigma } } } ) = \omega _ { T } ( \sigma )$ , and it has vanishing total mass if and only if $\omega _ { T } ( \sigma _ { \mathrm { t r i v } } ) = 0$ . The first two conditions together say exactly that $\omega _ { T }$ is real valued and agrees on complex conjugate pairs, and the kernel is as stated.

For the last claim, write $( R _ { b } f ) ( g ) = f ( g b )$ and substitute $h \mapsto h b ^ { - 1 }$ to obtain

$$
( T _ { \varphi } R _ { b } f ) ( g ) = \sum _ { h \in G } \varphi ( g ^ { - 1 } h b ^ { - 1 } ) f ( h )\tag{39}
$$

and

$$
( R _ { b } T _ { \varphi } f ) ( g ) = \sum _ { h \in G } \varphi ( b ^ { - 1 } g ^ { - 1 } h ) f ( h ) ,\tag{40}
$$

so $T _ { \varphi }$ commutes with every right translation if and only i $\mathrm { ~ f ~ } \varphi ( x y ) = \varphi ( y x )$ for all $x , y \in G ;$ ; that is, if and only if $\varphi = \varphi ^ { \natural }$ . On that subspace, the map is therefore injective, and by the proof of Theorem 1, the subspace is $\mathrm { s p a n } _ { \mathbb { R } } \{ \mathcal { L } _ { K } \}$ □

Proposition 7. $I f \alpha _ { K } \geq 0$ for every $K \in \kappa$ , then $T _ { \varphi ^ { \sharp } }$ is positive semidefinite and is the Laplacian of the weighted Cayley graph on G carrying weight $\alpha _ { K } / | K |$ on the edges labeled by $K$

Proof. Applying Proposition 1 termwise to $\begin{array} { r } { T _ { \varphi ^ { \sharp } } = \sum _ { K } \alpha _ { K } \mathcal { L } _ { F } } \end{array}$ gives

$$
\langle f , T _ { \varphi ^ { \sharp } } f \rangle = \sum _ { K \in { \cal K } } \frac { \alpha _ { K } } { 2 | G | | K | } \sum _ { g \in G } \sum _ { s \in K } | f ( g ) - f ( g s ) | ^ { 2 } \geq 0 ,\tag{41}
$$

which is the Dirichlet energy of the weighted Cayley graph described.

Admissibility alone does not imply positivity. Proposition 7 identifies a natural Dirichlet subclass: nonnegative class coeficients produce positive-semidefinite weighted Cayley-Laplacians. Specializing to $T = \mathcal { L }$ makes the weights explicit and exhibits the ordering induced by an arbitrary symmetric generating set as a convex combination of those induced by conjugation closed ones.

Corollary 2. Let $S = S ^ { - 1 }$ be a generating set with e /∈ S. Then

$$
\omega = \sum _ { K \in \mathcal K } \frac { | S \cap K | } { | S | } \omega _ { K } ,\tag{42}
$$

so the weights $\alpha _ { K } = | S \cap K | / | S |$ form a probability vector on $\kappa .$

Proof. Symmetry of S gives $| S \cap C | = | S \cap C ^ { - 1 } |$ | for every conjugacy class $C ,$ so averaging $\mu _ { S }$ over conjugation yields $\mu _ { S } ^ { \natural } = \sum _ { K }$ α<sub>K</sub>µ<sub>K</sub> with $\textstyle \sum _ { K } \alpha _ { K } = 1$ . Hence, $\begin{array} { r } { \varphi ^ { \sharp } = \sum _ { K } \alpha _ { K } ( \delta _ { e } - \mu _ { K } ) } \end{array}$ , and the claim follows from Proposition 6. □

The positivity hypothesis of Proposition 7 has a probabilistic reading. Writing $\begin{array} { r } { T _ { \varphi } f ( g ) = \sum _ { x \ne e } \varphi ( x ) [ f ( g x ) - } \end{array}$ $f ( g ) ]$ ], which is the content of vanishing total mass, the operator $- T _ { \varphi ^ { \sharp } }$ is the generator of a symmetric convolution Markov semigroup precisely when $\varphi ^ { \natural } \leq 0$ away from the identity, and by the computation above, this holds exactly when every $\alpha _ { K }$ is nonnegative.

The number of distinct ω is quantified by the size of $\kappa ,$ which can be determined to a great extent. Indeed, inversion acts on the set of conjugacy classes as an involution, and K is the set of its orbits on the nontrivial classes. ${ \mathrm { S o } } ,$ if G has k conjugacy classes and r of them are real (meaning $C = C ^ { - 1 }$ , so that the character is real on $C )$ [27], the non-real ones pair of and we have

$$
| \mathcal { K } | = r - 1 + \frac { k - r } { 2 } = \frac { k + r } { 2 } - 1 ,\tag{43}
$$

where the −1 is included to remove the class of the identity, which is always real. When $G$ is ambivalent, meaning every element is conjugate to its inverse, $r = k$ and $| \kappa | = k - 1$ , so K is just the nontrivial conjugacy classes. This is the case for all symmetric and dihedral groups, for example. However, it does not hold in general, as exhibited by the group $\mathbb { Z } _ { 3 }$ , which has $k = 3 , r = 1$ , and $| \mathcal { K } | = 1$

There are two more assumptions one can make beyond admissibility which narrow down the choice of ordering completely. We separate them from the above because they deal not with the group but with the generating set for the group. If we interpret the choice of S as a declaration of which group elements count as incremental changes, then the following two assumptions are natural. First, we can assume that an admissible operator $T _ { \varphi }$ is S-local in the sense that the support of $\varphi$ is contained in $\{ e \} \cup S$ . The second is that $T _ { \varphi }$ is S-uniform in the sense that $\varphi$ is constant on S. Locality is the assumption that elements of $S$ count as incremental changes, and uniformity is the assumption that no element of $S$ counts for more than another. With these additional assumptions, the induced Laplacian is narrowed down to the Laplacian of Section 2 up to a rescaling, which does not afect the ordering of irreps provided the constant is strictly positive; a negative constant reverses the ordering and a zero constant destroys it. Requiring positive semidefiniteness as in Proposition $7$ rules out the negative case, and requiring that $T$ be nonzero rules out the remaining one.

Proposition 8. Let $S = S ^ { - 1 }$ generate G with $e \not \in S .$ . An admissible $T$ is S-local and S-uniform $i f$ and only $i f T = c { \mathcal { L } }$ for some constant $c \in \mathbb { R }$

Proof. By Lemma 2, $T = T _ { \varphi }$ for some real-valued $\varphi$ satisfying $\varphi ( x ^ { - 1 } ) = \varphi ( x )$ and $\textstyle \sum _ { x \in G } \varphi ( x ) = 0$ . If T is S-local and S-uniform, then $\varphi = \lambda _ { e } \delta _ { e } + \lambda \delta _ { S }$ , where $\delta _ { S }$ is the indicator function on S and $\lambda _ { e } , \lambda \in \mathbb { R }$ . It follows that

$$
( T _ { \varphi } f ) ( g ) = \sum _ { h \in G } ( \lambda _ { e } \delta _ { e } + \lambda \delta _ { S } ) ( g ^ { - 1 } h ) f ( h ) = \lambda _ { e } f ( g ) + \lambda \sum _ { s \in S } f ( g s ) = ( ( \lambda _ { e } I + \lambda | S | A ) f ) ( g ) .\tag{44}
$$

Since $\varphi$ has vanishing total mass, $\lambda _ { e } = - \lambda | S |$ , and so $T = c { \mathcal { L } }$ for a constant $c \in \mathbb { R }$ . Conversely, if $T = c { \mathcal { L } } .$ ， then $\varphi = c ( \delta _ { e } - \mu _ { S } )$ , which is S-local and S-uniform. □

We stress that Theorem 1 characterizes the orderings available once one fixes the group. If one additionally declares which elements of G count as incremental changes, the operator is pinned down up to a scaling factor. We therefore view our choice of the Laplacian in Section 2 not as an arbitrary choice, but as the only choice satisfying the several natural axioms outlined here.

In the first paragraph of the introduction, we noted that smoothness on $\mathbb { R }$ can be viewed as a constraint on the Fourier transform of the function of interest. With our ordering parameter in hand, we may now produce an analog of this statement for the case of finite groups. Specifically, we show that a bandlimiting hypothesis on the Fourier coeficients implies an upper bound on the forward diference of a function and vice versa.

Theorem 3. Let $f \in L ^ { 2 } ( G )$ . If fb is supported on $\{ \sigma : \omega ( \sigma ) \leq \lambda \}$ , then

$$
\langle f , { \mathcal { L } } ^ { \natural } f \rangle \leq \lambda \| f \| ^ { 2 } .\tag{45}
$$

Conversely, $i f \left| f ( g ) - f ( g x ) \right| \le \Lambda$ for all g and all x in each K intersecting $S ,$ then

$$
\sum _ { \omega ( \sigma ) > \lambda } \| \widehat { f } ( \sigma ) \| _ { F } ^ { 2 } < \frac { \Lambda ^ { 2 } } { 2 \lambda } .\tag{46}
$$

Proof. Let $f _ { \sigma }$ denote the projection of $f$ onto the σ-block. By Corollary 1, $\mathcal { L } ^ { \sharp }$ acts on the $\sigma$ block as multiplication by $\omega ( \sigma )$ . Thus, $\mathcal { L } ^ { \natural } f _ { \sigma } = \omega ( \sigma ) f _ { \sigma }$ . The blocks are mutually orthogonal, so we have

$$
\langle f , \mathcal { L } ^ { \natural } f \rangle = \left. \sum _ { \sigma \in \widehat { G } } f _ { \sigma } , \sum _ { \tau \in \widehat { G } } \omega ( \tau ) f _ { \tau } \right. = \sum _ { \sigma \in \widehat { G } } \omega ( \sigma ) \Vert f _ { \sigma } \Vert ^ { 2 } .\tag{47}
$$

By Parseval’s identity, it follows that

$$
\langle f , \mathcal { L } ^ { \natural } f \rangle = \sum _ { \sigma \in \widehat { G } } \omega ( \sigma ) \| \widehat { f } _ { \sigma } \| _ { F } ^ { 2 } .\tag{48}
$$

If $\widehat { f }$ is supported on $\{ \sigma : \omega ( \sigma ) \leq \lambda \}$ , it therefore follows that

$$
\langle f , \mathcal { L } ^ { \natural } f \rangle = \sum _ { \omega ( \sigma ) \leq \lambda } \omega ( \sigma ) \| \widehat { f } _ { \sigma } \| _ { F } ^ { 2 } \leq \lambda \sum _ { \omega ( \sigma ) \leq \lambda } \| \widehat { f } _ { \sigma } \| _ { F } ^ { 2 } ,\tag{49}
$$

and by Parseval’s identity, the right hand side is $\lambda \| f \| ^ { 2 }$

For the converse, suppose $| f ( x ) - f ( g x ) | \leq \Lambda$ for all g and all x in each K intersecting S. Then by Theorem 1, there are constants α such that $\begin{array} { r } { \mathcal { L } ^ { \natural } = \sum _ { K } \alpha _ { K } \mathcal { L } _ { K } } \end{array}$ and by Proposition 1, we have

$$
\langle f , \mathcal { L } ^ { \natural } f \rangle = \sum _ { K \in \mathcal { K } } \alpha _ { K } \langle f , \mathcal { L } _ { K } f \rangle \leq \frac { \Lambda ^ { 2 } } { 2 } \sum _ { K \in \mathcal { K } } \alpha _ { K } = \frac { \Lambda ^ { 2 } } { 2 } .\tag{50}
$$

Applying this inequality to (48) produces

$$
\frac { \Lambda ^ { 2 } } { 2 } \geq \langle f , \mathcal { L } ^ { \sharp } f \rangle = \sum _ { \sigma \in \widehat { G } } \omega ( \sigma ) \| \widehat { f } ( \sigma ) \| _ { F } ^ { 2 } \geq \sum _ { \omega ( \sigma ) > \lambda } \omega ( \sigma ) \| \widehat { f } ( \sigma ) \| _ { F } ^ { 2 } \geq \lambda \sum _ { \omega ( \sigma ) > \lambda } \| \widehat { f } ( \sigma ) \| _ { F } ^ { 2 } ,\tag{51}
$$

which is what we set out to prove.

This theorem justifies the interpretation of smoothness as having Fourier support concentrated at low frequency. Indeed, an upper bound on the ordering parameter, restricting the Fourier support, produces a bound on the quadratic form associated to the conjugation-symmetrized Laplacian, which measures the amount that a function value changes when its domain is changed incrementally. Conversely, if one upper bounds the change in the function, this produces an upper bound on the weight of the Fourier support in the upper end of the spectrum

## 4 Smoothness on Sets Carrying a Group Action

The theory of the previous sections applies to functions on a group, but it can be extended to functions on sets carrying a group action. This section removes the restriction using standard methods in harmonic analysis and is briefly discussed by Belis et al. [1]. We let a group act on the domain of the function and lift the function to a function on the group, where the theory we developed applies. This adds another layer to the construction; the ordering, and hence the notion of smoothness, is now a consequence of which group one chooses to act with. In Section 5, we will discuss how two diferent groups acting on the same function domain can induce entirely diferent simplicity biases.

Let X be a finite set and suppose a finite group G acts transitively on X, so that for any $x , y \in X$ , there is a $g \in G$ with $g \cdot x = y$ . Fix a basepoint $x _ { 0 } \in X$ and let

$$
H = \{ h \in G : h \cdot x _ { 0 } = x _ { 0 } \}\tag{52}
$$

be its stabilizer. The map $H g \mapsto g ^ { - 1 } \cdot x _ { 0 }$ is then a bijection, giving the identification $X \cong H \setminus G$ and in particular, $| X | = | G | / | H |$ . Transitivity is not much of a restriction in practice, since a group acting with several orbits can be treated one orbit at a time.

Given a function $f : X \to \mathbb { C }$ , define its lift $f ^ { \uparrow } : G \to \mathbb { C }$ by

$$
f ^ { \uparrow } ( g ) = f ( g ^ { - 1 } \cdot x _ { 0 } ) .\tag{53}
$$

For any $h \in H$ , we have $f ^ { \uparrow } ( h g ) = f ( g ^ { - 1 } \cdot ( h ^ { - 1 } \cdot x _ { 0 } ) ) = f ^ { \uparrow } ( g )$ , so the lift is constant on the left cosets $H g .$ Write

$$
L ^ { 2 } ( G ) ^ { H } = \{ F \in L ^ { 2 } ( G ) : F ( h g ) = F ( g ) \forall h \in H \}\tag{54}
$$

for the space of such functions. Lifting is a linear isomorphism $L ^ { 2 } ( X ) \to L ^ { 2 } ( G ) ^ { H }$ , and both spaces have dimension |X|.

For the theory above to descend, the averaging operator must preserve $L ^ { 2 } ( G ) ^ { H }$ so that it can be thought of as a map $A : \dot { L } ^ { 2 } ( G ) ^ { H } \to L ^ { 2 } ( G ) ^ { H }$ . It does, as we now show.

Lemma 3. Let $H \leq G$ be any subgroup. Then A preserves $L ^ { 2 } ( G ) ^ { H }$ , and hence so does ${ \mathcal { L } } = I - A$

Proof. Let $F \in L ^ { 2 } ( G ) ^ { H }$ and $h \in H$ . Then

$$
( A F ) ( h g ) = \frac { 1 } { | S | } \sum _ { s \in S } F ( h g s ) = \frac { 1 } { | S | } \sum _ { s \in S } F ( g s ) = ( A F ) ( g ) ,\tag{55}
$$

where the second equality applies the left H-invariance of F to each argument gs separately.

Concretely, the operator induced on X is the averaging operator of the Schreier graph. Define $A _ { X }$ : $L ^ { 2 } ( X ) \to L ^ { 2 } ( X )$ by

$$
( A _ { X } f ) ( x ) = { \frac { 1 } { | S | } } \sum _ { s \in S } f ( s \cdot x ) .\tag{56}
$$

Substituting $x = g ^ { - 1 } \cdot x _ { 0 }$ into the definition of the lift gives

$$
( A f ^ { \uparrow } ) ( g ) = \frac { 1 } { | S | } \sum _ { s \in S } f ( s ^ { - 1 } g ^ { - 1 } \cdot x _ { 0 } ) = ( A _ { X } f ) ( g ^ { - 1 } \cdot x _ { 0 } ) ,\tag{57}
$$

where the last equality uses symmetry of S. Hence, $( A _ { X } f ) ^ { \uparrow } = A f ^ { \uparrow }$ , and lifting intertwines L with the Laplacian of the Schreier graph of $( X , S )$ . In particular, ω may be computed on X directly, without passing through G.

Not every Fourier coeficient of G is visible on X. To see which are, introduce the averaging operator over the stabilizer,

$$
( \Pi F ) ( g ) = { \frac { 1 } { | H | } } \sum _ { h \in H } F ( h g ) .\tag{58}
$$

Since H is a group, Π is idempotent and Hermitian, and its image is exactly $L ^ { 2 } ( G ) ^ { H }$ . It is therefore the orthogonal projection onto the space of lifted functions. The question of which frequencies survive is the question of how Π interacts with the blocks of Proposition 2, and the computation is the one already performed there.

Proposition 9. The projection Π preserves the block $A _ { \sigma } = \operatorname { s p a n } \{ \sigma _ { i j } \}$ , and

$$
\dim ( { \mathcal { A } } _ { \sigma } \cap L ^ { 2 } ( G ) ^ { H } ) = d _ { \sigma } m _ { \sigma } ,\tag{59}
$$

where $\begin{array} { r } { m _ { \sigma } = \frac { 1 } { | H | } \sum _ { h \in H } \chi _ { \sigma } ( h ) } \end{array}$ . Moreover, $\begin{array} { r } { \sum _ { \sigma } d _ { \sigma } m _ { \sigma } = | X | } \end{array}$

Proof. Let $\begin{array} { r } { P _ { \sigma } ^ { H } = \frac { 1 } { | H | } \sum _ { h \in H } \sigma ( h ) } \end{array}$ . Similar to the proof of Proposition 2, using $\sigma ( h g ) = \sigma ( h ) \sigma ( g )$ , we have

$$
( \Pi \sigma _ { i j } ) ( g ) = \frac { 1 } { | H | } \sum _ { h \in H } \sigma _ { i j } ( h g ) = \sum _ { k = 1 } ^ { d _ { \sigma } } \left( P _ { \sigma } ^ { H } \right) _ { i k } \sigma _ { k j } ( g ) ,\tag{60}
$$

so Π preserves $\sigma$ and the index j and acts on the index i by the matrix $P _ { \sigma } ^ { H }$ . In particular, $A _ { \sigma }$ is preserved. Re-indexing the double sum by $h \mapsto h h ^ { \prime }$ shows $( P _ { \sigma } ^ { H } ) ^ { 2 } = \bar { P } _ { \sigma } ^ { H }$ , so $P _ { \sigma } ^ { H }$ is idempotent and its rank equals its trace,

$$
\mathrm { r a n k } ( P _ { \sigma } ^ { H } ) = \mathrm { T r } \left[ P _ { \sigma } ^ { H } \right] = { \frac { 1 } { | H | } } \sum _ { h \in H } \chi _ { \sigma } ( h ) = m _ { \sigma } .\tag{61}
$$

The i index is confined to an $m _ { \sigma }$ -dimensional image while the j index remains free, giving $d _ { \sigma } m _ { \sigma }$ surviving dimensions in the block. Summing over σ and using dim $( L ^ { 2 } ( G ) ^ { H } ) = | G | / | H | = | X |$ gives the last claim.

Proposition 9 says that σ contributes to $L ^ { 2 } ( X )$ exactly when $m _ { \sigma } > 0 ;$ that is, exactly when $\mathcal { H } _ { \sigma }$ contains a nonzero H-fixed vector, and that it then contributes $d _ { \sigma } m _ { \sigma }$ dimensions rather than the $d _ { \sigma } ^ { 2 }$ it occupies in $L ^ { 2 } ( G )$ The frequencies visible on X are therefore a subset of ${ \widehat { G } } ,$ and the ordering induced on X is the restriction of ω to that subset. Two extreme cases are worth recording. The trivial representation always survives since $m _ { \sigma } = 1$ , so the constant functions are present on X and remain the smoothest, as they must be. At the other end, taking H trivial gives $m _ { \sigma } = d _ { \sigma }$ for every σ, recovering $L ^ { 2 } ( X ) = L ^ { 2 } ( G )$ and the theory of the previous sections unchanged. Between these, enlarging H shrinks X and prunes the spectrum, and the pruning is not uniform across ${ \widehat { G } } ;$ an irrep with no H-fixed vector is invisible on X no matter what its ordering parameter is. In this sense, the pair (G, H) selects which frequencies exist and the pair $( G , S )$ orders them, and the two choices are independent. The multiplicity $m _ { \sigma }$ can exceed one, in which case the surviving copies of σ all carry the same block of L and hence the same value $\omega ( \sigma )$ , so the ordering does not distinguish them. The case in which this degeneracy is absent is that of a Gelfand pair [4, 28], where $m _ { \sigma } \le 1$ for every σ and $L ^ { 2 } ( X )$ is multiplicity free. Each surviving irrep then occurs exactly once, in a d<sub>σ</sub>-dimensional irreducible subspace, and the ordering parameter assigns a single value $\omega ( \sigma )$ to that entire component.

## 5 Examples

We now illustrate the general theory with four examples, two of which showcase the way our notion of smoothness depends on the choice of generating set. Moreover, once X is a bare set, nothing selects the group acting on it, and diferent choices induce genuinely diferent notions of smoothness. We first discuss the latter point using two concrete examples. We then give two examples in which the generating set produces diferent orderings and therefore diferent notions of smoothness.

## 5.1 Actions on the first n integers

Let $X = \{ 0 , \ldots , n - 1 \}$ . One natural choice of action is $G = \mathbb { Z } _ { n }$ acting by rotation; that is, $k \cdot x = x + k$ mod n. The action is simply transitive, so H is trivial regardless of basepoint, and $m _ { \sigma } = 1$ for every σ. No frequency is lost in the descent to X. The identification $X \cong G$ sends x to the group element $x _ { 0 } - x _ { : }$ and $L ^ { 2 } ( X ) \cong L ^ { 2 } ( G )$ as spaces of functions.

Take $S = \{ \pm 1 \}$ , which is symmetric, generates G, and is closed under conjugation since G is abelian. The Cayley graph is the n-cycle. All irreps are one-dimensional, given by the characters $\chi _ { k } ( x ) = e ^ { 2 \pi i k x / n }$ for $k \in \mathbb { Z } _ { n }$ , so ${ \widehat { G } } \cong \mathbb { Z } _ { n }$ , and each block of Proposition 2 is a single number. Explicitly, we have

$$
M _ { \chi _ { k } } = \frac { 1 } { 2 } \left( \chi _ { k } ( 1 ) + \chi _ { k } ( - 1 ) \right) = \cos ( 2 \pi k / n ) ,\tag{62}
$$

and since $d _ { \chi _ { k } } = 1$ , the trace is the value itself, giving

$$
\omega ( \chi _ { k } ) = 1 - \cos ( 2 \pi k / n ) .\tag{63}
$$

The ordering this induces is the familiar one. The trivial character $k = 0$ has $\omega = 0$ as required, and ω increases with |k| measured cyclically, so the frequencies are ordered by how far k sits from 0 in $\mathbb { Z } _ { n }$ Conjugate pairs $\chi _ { k }$ and $\chi _ { n - k } \ = \ \overline { { \chi _ { k } } }$ receive equal values, consistent with Theorem 2. The spectrum is therefore symmetric and ω takes $\lfloor n / 2 \rfloor + 1$ distinct values. The upper bound $\omega = 2$ is attained exactly when n is even by $k = n / 2$ , which is the alternating character $\chi _ { n / 2 } ( x ) = ( - 1 ) ^ { x }$ . This matches the bipartiteness criterion since the n-cycle is bipartite precisely for even n.

Another natural choice of action is $G = \mathbb { S } _ { n }$ acting by permutation; that is, $\pi \cdot x = \pi ( x )$ . The action is transitive but not simply so, and the stabilizer of any basepoint is the copy of $\mathbb { S } _ { n - 1 }$ fixing that point, so $\begin{array} { r } { | X | = \frac { n ! } { ( n - 1 ) ! } = n } \end{array}$ as it must be. The frequencies visible on X are those σ with $m _ { \sigma } > 0$ . Here $\mathbb { C } [ X ]$ carries the natural permutation representation of $\mathbb { S } _ { n }$ on n points, which decomposes as the trivial representation plus the standard representation, each once. Since dim $( A _ { \sigma } \cap L ^ { 2 } ( G ) ^ { H } ) = d _ { \sigma } m _ { \sigma }$ and $L ^ { 2 } ( X ) \cong \mathbb { C } [ X ]$ , exactly two of the irreps of $\mathbb { S } _ { n }$ survive, both with $m _ { \sigma } = 1$ . Every other irrep, including all those of large dimension, is invisible on X no matter what generating set is chosen.

The two surviving carrier spaces are the constant functions of dimension one and the mean-zero functions of dimension $n - 1$ . Since $m _ { \sigma } = 1$ for both, $( \mathbb { S } _ { n } , \mathbb { S } _ { n - 1 } )$ is a Gelfand pair and the decomposition is multiplicity free. The trivial representation has $\omega = 0$ for any choice of S. The standard representation receives a single value $\omega _ { \mathrm { s t d } } > 0$ whose size depends on S but whose position in the ordering does not, since it is the only other frequency present.

The consequence is that the ordering on X carries no information. Every $f \in L ^ { 2 } ( X )$ splits uniquely as its mean plus a mean-zero remainder, and the only smoothness prior available is that f be close to a constant. Contrasted with $\mathbb { Z } _ { n }$ on the same X, where all n frequencies survive, and the ordering distinguishes them, this shows that the meaning of smoothness is dependent on the choice of group action.

## 5.2 Actions on bitstrings

Let $X = \{ 0 , 1 \} ^ { n }$ and for a first action take $G = \mathbb { Z } _ { 2 } ^ { n }$ acting by translation. With $\boldsymbol { x } _ { 0 } = ( 0 , \ldots , 0 )$ , the stabilizer H is trivial, and we recover the 2<sup>n</sup> frequencies ordered by Hamming weight studied by Belis et al. [1]. Indeed, we take $S = \{ e _ { i } \} _ { i = 1 } ^ { n }$ , where $e _ { i } = ( 0 , \ldots , 0 , 1 , 0 \ldots , 0 )$ denotes the standard basis vector with a one in the i-th component and a zero otherwise. The irreps are given by the characters $\chi _ { k } ( x ) = ( - 1 ) ^ { k \cdot x }$ and we have

$$
( A \chi _ { k } ) ( x ) = \frac { 1 } { | S | } \sum _ { s \in S } \chi _ { k } ( x s ) = \chi _ { k } ( x ) \frac { 1 } { | S | } \sum _ { s \in S } \chi _ { k } ( s ) = \chi _ { k } ( x ) \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \chi _ { k } ( e _ { i } ) = \chi _ { k } ( x ) \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( - 1 ) ^ { k _ { i } } .\tag{64}
$$

Letting |k| denote the Hamming weight, this becomes

$$
( A \chi _ { k } ) ( x ) = \chi _ { k } ( x ) \frac { 1 } { n } ( n - 2 | k | ) .\tag{65}
$$

It follows that $\omega ( \chi _ { k } ) = 2 | k | / n$ , so that the ordering parameter reduces to a simple rescaling of the Hamming weight ordering.

![](images/98905571c5a917aa6dbd6585f1826a730faf2f091bdc560d0e9c710d53b8e0c9.jpg)

![](images/d1ebd69de5ddbb1e8690d21a298e2c2dd31157de65f342bb99f4289644fae169.jpg)  
Figure 1: Plot of $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } ( \chi _ { 2 } + \chi _ { 1 4 } ) = \cos \left( \frac { \pi } { 4 } x \right) } \end{array}$ over the Cayley graph of $\mathbb { Z } _ { 1 6 }$ with generating set $\{ \pm 1 \}$ The plot above the actual Cayley graph is shown in the left panel, while the plot in a planar embedding is shown in the right panel.

A second action is given by the hyperoctahedral group $B _ { n } = \mathbb { Z } _ { 2 } ^ { n } \rtimes \mathbb { S } _ { n }$ , which permutes the coordinates in addition to the translation action. This action is also transitive on X with $H = \mathbb { S } _ { n }$ the stabilizer of $x _ { 0 } = ( 0 , \ldots , 0 )$ . We again take $S = \{ ( e _ { i } , \mathrm { i d } ) \} _ { i = 1 } ^ { n }$ , which generates only the normal subgroup $\mathbb { Z } _ { 2 } ^ { n }$ . This is harmless because generation was assumed in Section 2 only to make the Cayley graph connected, and $\mathbb { Z } _ { 2 } ^ { n }$ already acts transitively on $X ,$ so the Schreier graph on $X$ is connected and ω still vanishes only at the trivial representation among the irreps visible on X. Restricting an irrep of $B _ { n }$ to the normal subgroup $\mathbb { Z } _ { 2 } ^ { n }$ decomposes it into characters, and $B _ { n }$ acts on those characters through its quotient $\mathbb { S } _ { n }$ by

$$
( \pi \cdot \chi _ { k } ) ( x ) = \chi _ { k } ( \pi ^ { - 1 } x ) = \chi _ { \pi ( k ) } ( x ) ,\tag{66}
$$

where π permutes the coordinates of k. Since the orbits of $\mathbb { S } _ { n }$ on $\mathbb { Z } _ { 2 } ^ { n }$ are the Hamming levels, the $2 ^ { n }$ characters of $\mathbb { Z } _ { 2 } ^ { n }$ fuse into $n + 1$ classes under the larger group, one for each value of $| k |$ . The surviving irrep at level $j$ has carrier space span $\left\{ \chi _ { k } : | k | = j \right\}$ of dimension $\binom { n } { i }$ . That each such space is irreducible under $B _ { n } .$ , and that no two are equivalent, is the statement that $( B _ { n } , \mathbb { S } _ { n } )$ is a Gelfand pair [28], so every $m _ { \sigma }$ is at most one and the decomposition of $L ^ { 2 } ( X )$ is multiplicity free.

The comparison with the first action is the point. Passing from Z<sup>n</sup> to $B _ { n }$ does not change X and does not change the values taken by the ordering parameter, since $\begin{array} { r } { \omega ( \chi _ { k } ) = \frac { 2 | k | } { n } } \end{array}$ already depended on $k$ only through $| k |$ . What changes is the resolution of the ordering. Under $\mathbb { Z } _ { 2 } ^ { n }$ , there are $2 ^ { n }$ separate frequencies which happen to be assigned equal values in groups, while under $B _ { n }$ , those groups have merged into single frequencies of dimension $\textstyle { \binom { n } { j } }$ . The larger symmetry group removes the ability to distinguish coordinates, and the ordering becomes an ordering of $n + 1$ objects rather than of $2 ^ { n }$ . This phenomenon is no accident, and the reason is elementary.

Lemma 4. $L e t \gamma$ be an automorphism of G with $\gamma ( S ) = S$ , and for an irrep σ write $\sigma ^ { \gamma } = \sigma \circ \gamma$ . Then $\omega ( \sigma ^ { \gamma } ) = \omega ( \sigma )$

Proof. Since γ restricts to a bijection of $S ,$ we have

$$
M _ { \sigma ^ { \gamma } } = \frac { 1 } { | S | } \sum _ { s \in S } \sigma ( \gamma ( s ) ) = \frac { 1 } { | S | } \sum _ { s \in S } \sigma ( s ) = M _ { \sigma } ,\tag{67}
$$

and so the ordering parameters agree.

![](images/7d73ab352061c36538fcfe3e23e330729bf17e7442384ee431544957a04494dc.jpg)

![](images/81d4f6df5905f8e22da6559f9625ec6a942d410dd9a3fbb37411dea716a72312.jpg)  
Figure 2: Plot of $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } ( \chi _ { 2 } + \chi _ { 1 4 } ) = \cos \left( \frac { \pi } { 4 } x \right) } \end{array}$ over the Cayley graph of $\mathbb { Z } _ { 1 6 }$ with generating set $\{ \pm 3 \}$ The plot above the actual Cayley graph is shown in the left panel, while the plot in a planar embedding is shown in the right panel.

## 5.3 Visualizing Smoothness on a Cyclic Group

Consider the group $\mathbb { Z } _ { 1 6 }$ whose irreps are the 16-th roots of unity $\chi _ { k } ( x ) = e ^ { 2 \pi i k x / 1 6 } = e ^ { \pi i k x / 8 }$ , and let us compare two diferent choices of generating set, $S _ { 1 } = \left\{ \pm 1 \right\}$ and $S _ { 3 } = \{ \pm 3 \}$ . For $S _ { 1 }$ , the Cayley graph is given by the 16-cycle, but for visualization purposes, let us chop the edge between points 15 and 0 in half, and note that they are identified with each other. Thus, we may draw the Cayley graph as a straight line with 16 points labeled 0 through 15 with the caveat that when one reaches the end of the last edge, they loop back around to the edge entering 0. This allows us to plot the values of a function on $\mathbb { Z } _ { 1 6 }$ in a plane as one usually would. Consider the function $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } ( \chi _ { 2 } ( x ) + \chi _ { 1 4 } ( x ) ) = \cos \left( \frac { \pi } { 4 } x \right) } \end{array}$ . Its plot along an axis orthogonal to the Cayley graph is pictured in Figure 1 in both the circular Cayley graph format and the linear axis format.

For $S _ { 3 } .$ , the Cayley graph is also given by the 16-cycle, but the group elements labeling each node have been permuted. Again plotting $f ,$ but with respect to this new ordering of the integers induced by the choice of generating set, we see that the traditionally smooth behavior of the cosine function disappears. The corresponding plots are shown in Figure 2.

Comparing Figures 1 and 2, we see that $f$ appears more smooth when choosing the generating set $S _ { 1 }$ Therefore, we expect that $f$ has Fourier coeficients concentrated at lower frequency when choosing $S _ { 1 }$ for the generating set, and we expect $f$ to have Fourier coeficients concentrated at higher frequency when choosing $S _ { 3 }$ . Indeed, for $S _ { 1 }$ , the averaging operator is

$$
( A f ) ( x ) = { \frac { 1 } { 2 } } ( f ( x + 1 ) + f ( x - 1 ) ) ,\tag{68}
$$

and so

$$
( A \chi _ { k } ) ( x ) = \frac { 1 } { 2 } \left( e ^ { 2 \pi i k ( x + 1 ) / 1 6 } + e ^ { 2 \pi i k ( x - 1 ) / 1 6 } \right) = \cos \left( \frac { \pi } { 8 } k \right) \chi _ { k } ( x )\tag{69}
$$

and the ordering is therefore

$$
\omega ( \chi _ { k } ) = 1 - \cos \left( \frac { \pi } { 8 } k \right) .\tag{70}
$$

Explicitly, we have $\omega ( \chi _ { 0 } ) < \omega ( \chi _ { 1 } ) = \omega ( \chi _ { 1 5 } ) < \omega ( \chi _ { 2 } ) = \omega ( \chi _ { 1 4 } ) < \omega ( \chi _ { 3 } ) = \omega ( \chi _ { 1 3 } ) < \omega ( \chi _ { 4 } ) = \omega ( \chi _ { 1 2 } ) < \omega ( \chi _ { 1 3 } ) .$ $\begin{array} { r } { \omega ( \chi _ { 5 } ) = \omega ( \chi _ { 1 1 } ) < \omega ( \chi _ { 6 } ) = \omega ( \chi _ { 1 0 } ) < \omega ( \chi _ { 7 } ) = \omega ( \chi _ { 9 } ) < \omega ( \chi _ { 8 } ) . \mathrm { ~ S i n c e } ~ f = \frac { 1 } { 2 } ( \chi _ { 2 } + \chi _ { 1 4 } ) } \end{array}$ , we see that $f$ is low frequency for $S _ { 1 }$ as expected. For $S _ { 3 }$ , the averaging operator is

$$
( A f ) ( x ) = { \frac { 1 } { 2 } } ( f ( x + 3 ) + f ( x - 3 ) ) ,\tag{71}
$$

and so

$$
( A \chi _ { k } ) ( x ) = \frac { 1 } { 2 } \left( e ^ { 2 \pi i k ( x + 3 ) / 1 6 } + e ^ { 2 \pi i k ( x - 3 ) / 1 6 } \right) = \cos \left( \frac { 3 \pi } { 8 } k \right) \chi _ { k } ( x )\tag{72}
$$

and the ordering is therefore

$$
\omega ( \chi _ { k } ) = 1 - \cos \left( \frac { 3 \pi } { 8 } k \right) .\tag{73}
$$

Explicitly, we have $\omega ( \chi _ { 0 } ) < \omega ( \chi _ { 5 } ) = \omega ( \chi _ { 1 1 } ) < \omega ( \chi _ { 6 } ) = \omega ( \chi _ { 1 0 } ) < \omega ( \chi _ { 1 } ) = \omega ( \chi _ { 1 5 } ) < \omega ( \chi _ { 4 } ) = \omega ( \chi _ { 1 2 } ) < \omega ( \chi _ { 1 3 } ) .$ $\omega ( \chi _ { 7 } ) = \omega ( \chi _ { 9 } ) < \omega ( \chi _ { 2 } ) = \omega ( \chi _ { 1 4 } ) < \omega ( \chi _ { 3 } ) = \omega ( \chi _ { 1 3 } ) < \omega ( \chi _ { 8 } )$ . Since $f = { \textstyle \frac { 1 } { 2 } } ( \chi _ { 2 } + \chi _ { 1 4 } )$ , we see that $f$ is high frequency for $S _ { 3 }$ as expected.

## 5.4 Visualizing Smoothness on a Dihedral Group

Consider the dihedral group $G = D _ { 3 }$ and let r and $f$ denote the rotation by 120 degrees and the flip about an axis through the triangle, respectively. This group has three irreps: the trivial representation defined by $\sigma _ { \mathrm { t r i v } } ( g ) = 1$ , the sign representation defined by $\sigma _ { \mathrm { s g n } } ( r ) = 1$ and $\sigma _ { \mathrm { s g n } } ( f ) = - 1$ , and the standard representation defined by

$$
\sigma _ { \mathrm { s t d } } ( r ) = \frac { 1 } { 2 } \left( \frac { - 1 } { \sqrt { 3 } } \quad \frac { - \sqrt { 3 } } { - 1 } \right)\tag{74}
$$

and

$$
\sigma _ { \mathrm { s t d } } ( f ) = \binom { 1 } { 0 } \begin{array} { c c } { { 0 } } \\ { { - 1 } } \end{array} ) .\tag{75}
$$

We again consider two generating sets, $S _ { 1 } = \{ r , f , r ^ { 2 } \}$ and $S _ { 2 } = \{ f , r f \}$ . For $S _ { 1 }$ , the averaging operator is

$$
( A h ) ( g ) = \frac { 1 } { 3 } ( h ( g r ^ { 2 } ) + h ( g r ) + h ( g f ) ) ,\tag{76}
$$

and so

$$
( A \sigma _ { \mathrm { t r i v } } ) ( g ) = 1 = \sigma _ { \mathrm { t r i v } } ,\tag{77}
$$

from which it follows that $\omega ( \sigma _ { \mathrm { t r i v } } ) = 0$ . Similarly, we have

$$
( A \sigma _ { \mathrm { s g n } } ) ( g ) = \frac { 1 } { 3 } ( 1 + 1 - 1 ) \sigma _ { \mathrm { s g n } } ( g ) = \frac { 1 } { 3 } \sigma _ { \mathrm { s g n } } ( g ) ,\tag{78}
$$

from which it follows that $\begin{array} { r } { \omega ( \sigma _ { \mathrm { s g n } } ) = \frac { 2 } { 3 } } \end{array}$ . Finally, for the standard representation, we have

$$
( A \sigma _ { \mathrm { s t d } } ) ( g ) = \frac { 1 } { 3 } \sigma _ { \mathrm { s t d } } ( g ) \left( \begin{array} { c c } { { 0 } } & { { 0 } } \\ { { 0 } } & { { - 2 } } \end{array} \right) ,\tag{79}
$$

from which it follows that $\begin{array} { r } { \omega ( \sigma _ { \mathrm { s t d } } ) = \frac { 4 } { 3 } } \end{array}$ . So the ordering with respect to $S _ { 1 }$ is $\omega ( \sigma _ { \mathrm { t r i v } } ) < \omega ( \sigma _ { \mathrm { s g n } } ) < \omega ( \sigma _ { \mathrm { s t d } } )$ Considering instead the generating set $S _ { 2 }$ , the averaging operator becomes

$$
( A h ) ( g ) = \frac { 1 } { 2 } ( h ( g f ) + h ( g r f ) ) ,\tag{80}
$$

and so

$$
( A \sigma _ { \mathrm { t r i v } } ) ( g ) = 1 = \sigma _ { \mathrm { t r i v } } ,\tag{81}
$$

from which it follows that $\omega ( \sigma _ { \mathrm { t r i v } } ) = 0$ . Similarly, we have

$$
( A \sigma _ { \mathrm { s g n } } ) ( g ) = - \sigma _ { \mathrm { s g n } } ( g ) ,\tag{82}
$$

from which it follows that $\omega ( \sigma _ { \mathrm { s g n } } ) = 2$ . Finally, for the standard representation, we have $\omega ( \sigma _ { \mathrm { s t d } } ) = 1$ . So the ordering with respect to $S _ { 2 }$ is $\omega ( \sigma _ { \mathrm { t r i v } } ) < \omega ( \sigma _ { \mathrm { s t d } } ) < \omega ( \sigma _ { \mathrm { s g n } } )$

We plot the function $\sigma _ { \mathrm { s g n } }$ over the Cayley graph in Figures 3 and 4 for both generating sets. For $S _ { 1 }$ , the sign representation is the second frequency, whereas for $S _ { 2 } ,$ , it is the third. Thus, the sign function should be interpreted as smoother in the former case. Indeed, looking at the figures, the function value changes dramatically less often in the case of $S _ { 1 }$ than it does in the case of $S _ { 2 }$ . In fact, for $S _ { 2 }$ , every edge in the Cayley graph corresponds to a maximal amount of change.

![](images/aa3b77c89b01e8f0ce490038d7436e6c4f6488abf59af57228cd59d441e903d5.jpg)  
Figure 3: The sign representation of $D _ { 3 }$ on the Cayley graph of $( D _ { 3 } , S _ { 1 } )$ with $S _ { 1 } = \{ r , r ^ { 2 } , f \}$ , shown in the plane and lifted to the value of the function. Dashed edges are those across which the sign changes. Three of the nine edges are dashed, and $\omega ( \sigma _ { \mathrm { s g n } } ) = 2 / 3$

![](images/0fb23f51fd5b021e83844ab93bd7483df6d96756929caae2fb8fe67724097c68.jpg)  
Figure 4: The sign representation of $D _ { 3 }$ on the Cayley graph of $( D _ { 3 } , S _ { 2 } )$ with $S _ { 2 } = \{ f , r f \}$ , shown in the plane and lifted to the value of the function. Dashed edges are those across which the sign changes. Every generator is a reflection, so every edge joins a rotation to a reflection and all six edges are dashed, giving the maximal value $\omega ( \sigma _ { \mathrm { s g n } } ) = 2$

![](images/4f0f749603bba7347e17ba3693972545607159d138051befa46b4b53c18bbf6b.jpg)  
Figure 5: A hairpin, the simplest pseudoknot-free secondary structure. Dashed rungs are base pairs and the unpaired letters at the top form the loop closed by the stem. Each adjacent pair of base pairs is a stack that lowers the free energy, while the enclosed loop raises it by an amount growing with its length.

## 6 Smoothness as an Inductive Bias

Suppose we are not analyzing a function, but attempting to build one. Training an interpretable machine learning model is the prototypical example [29, 30, 31]. We wish to bias the construction toward smooth functions, but this presupposes a choice of group action and generating set and begs the question of how these objects should be chosen, given a problem. What are the features of the problem that make one choose group G over group H or generating set S over generating set T? Let us fix a problem and explore how a designer might settle these questions.

An RNA molecule is a string over the four letters A, C, G, and U. Unlike DNA, it is single stranded, so it folds back on itself. A letter at one position can pair with a complementary letter elsewhere in the same string, A with U, G with C, and G with U. A secondary structure is a set of such pairs with each position occurring in at most one pair. Two common additional assumptions are made: a minimum hairpin length so that paired positions $i \ < \ j$ satisfy $j - i > 3$ , and the absence of pseudoknots so that any two pairs $( i , j )$ and $( k , l )$ with $i < k$ satisfy either $j < k$ or $l < j$ . Each such structure is assigned a free energy by the standard nearest-neighbor thermodynamic model [32], in which stacked pairs lower the energy and the unpaired loops they enclose raise it by an amount growing with the length of the loop (see Figure 5). The molecule occupies low energy structures, and the minimum free energy over all pseudoknot-free secondary structures of a given string, which we write E(string), is computable by dynamic programming [33] in time cubic in the length once internal loops are capped at the conventional size. Dropping the pseudoknot-free restriction changes the problem, and minimization over general pseudoknotted structures is NP-hard for standard energy models [34].

Now suppose $n = 1 2$ short RNA blocks called modules are fixed in advance, and a construct is built by concatenating all twelve in some order. The designer wants the arrangement whose fold is most stable, so the response is $f ( \pi ) = E ( { \mathrm { c o n c a t e n a t i o n } }$ in the order π) and the goal is to minimize it over X. Each evaluation is one dynamic program and is cheap, but $| X | = 1 2 ! = 4 7 9 0 0 1 6 0 0$ , so enumeration is out of reach and the designer budgets 500 evaluations. The task is to fit a surrogate to those 500 values and minimize the surrogate exactly.

The domain and the group are settled by asking what leaves f unchanged. If two modules had identical sequences, exchanging their labels would be an invariance, the domain would be the quotient of X by that exchange, and Proposition 9 would delete the frequencies with no fixed vector. An RNA string has a direction and the reverse of a construct is a diferent molecule with a diferent fold, not the same one relabeled, so the one symmetry a designer might hope for beyond the above is unavailable. With twelve distinct modules and no reversal symmetry, we record the full order, take $X = G = \mathbb { S } _ { 1 2 }$ , and let it act on itself by translation. This is the smallest transitive choice and the stabilizer is trivial. Moreover, $m _ { \lambda } = \chi _ { \lambda } ( e ) = d _ { \lambda }$ for every partition λ ⊢ 12, so nothing is deleted.

When choosing the generating set, our goal is to choose group elements which can be naturally considered incremental steps on our data. Two such generating sets come to mind. The first is the adjacent transpositions

$$
S _ { \mathrm { a d j } } = \{ ( i i ( 1 + 1 ) : 1 \leq i \leq n - 1 \} ,\tag{83}
$$

which interchange two neighbors and leave every other item alone. Indeed, in the context of a ranking, such a transposition performs the minimal change by swapping the ordering of two adjacent items. The Cayley graph corresponding to this choice of generator is the permutohedron, and the generalized Hamming weight of Section 2 is the number of inversions, which we note is Kendall’s tau distance [35]. The second generating set is the insertions

$$
S _ { \mathrm { i n s } } = \{ ( i ~ i + 1 ~ \cdots ~ j ) ^ { \pm 1 } : 1 \leq i < j \leq n \}\tag{84}
$$

which are obtained by removing an item in position $j$ and reinserting it in position $i ,$ sliding the intervening items over by one. The size of this set is $| S _ { \mathrm { i n s } } | ~ = ~ ( n ~ - ~ 1 ) ^ { 2 }$ , with $n - 1$ moves of cycle length two and $2 ( n - m + 1 )$ of cycle length m for each $3 \leq m \leq n$ . Insertion is the elementary edit of a sorted list and the move underlying insertion sort, and it is no less local a description of small change than a neighbor swap; the two notions are simply diferent and should be applied in diferent contexts.

In the context of our RNA secondary structure problem, the generating set is chosen by asking which rearrangements the designer is willing to call small. A stem (the stack of dashed rungs in Figure 5) formed between two modules closes a loop containing everything between its arms, and the penalty for that loop grows with how far apart the two modules sit on the chain. An adjacent transposition changes the number of intervening modules between any pair by at most one, and since the modules are fixed, it changes any separation measured in nucleotides by at most twice the length of the longest module. An insertion moves a module across up to eleven others and can change the separations involving it by a distance comparable to the length of most of the construct. Holding the pairing set fixed, adjacent transpositions therefore perturb loop lengths on a scale set by individual module lengths, whereas insertions can perturb them on the scale of the whole construct. What that argument bounds is the change in the separations, rather than the change in the energy, since the minimizing structure itself moves with the order and the stacking and junction terms depend on the letters adjacent to each pair; a single swap can create or destroy a stem, and an insertion that preserves the relative order of a module and its partners can leave the energy nearly fixed. Here, we take $S = S _ { \mathrm { a d j } }$ as a declaration of which rearrangements count as incremental.

The designer may further believe that the modules near the ends of the construct matter less than those in the middle. After all, the ends have fewer ways to be enclosed. That belief cannot be expressed here. All adjacent transpositions lie in the single conjugacy class K of transpositions, so a weighting $\varphi =$ $\lambda _ { e } \delta _ { e } - \textstyle \sum _ { i } w _ { i } \delta _ { s _ { i } }$ with $w _ { i } \geq 0$ and $\lambda _ { e } = \textstyle \sum _ { i } w _ { i }$ , the latter being the condition that $T _ { \varphi }$ annihilate the constants, has conjugation average $\begin{array} { r } { \varphi ^ { \natural } = \sum _ { i } w _ { i } ( \delta _ { e } - \mu _ { K } ) } \end{array}$ , and by Proposition 6, the induced ordering is a nonnegative multiple of $\omega _ { K }$ no matter what the individual $w _ { i }$ are, and a positive one as soon as some $w _ { i } > 0$ . Position dependent weights change the operator but not the ordering, so the belief has to enter through the model rather than the prior.

With these choices made, the designer chooses a bandlimit to enforce smoothness with, and this is settled by counting. Every generator is a transposition, so $\omega ( \lambda ) = 1 - \chi _ { \lambda } ( \tau ) / d _ { \lambda }$ for a single transposition τ, and Frobenius’ formula [4] gives

$$
\omega ( \lambda ) = 1 - \frac { 2 } { n ( n - 1 ) } \sum _ { ( i , j ) \in \lambda } ( j - i ) ,\tag{85}
$$

the sum of the contents of the cells of λ. For $n = 1 2$ , the smallest values are $\omega ( ( 1 2 ) ) = 0$ , then $\begin{array} { r } { \omega ( ( 1 1 , 1 ) ) = \frac { 2 } { 1 1 } } \end{array}$ then $\begin{array} { r } { \omega ( ( 1 0 , 2 ) ) = \frac { 1 } { 3 } } \end{array}$ . Since $m _ { \lambda } = d _ { \lambda }$ , the parameter count below a cutof is $\textstyle \sum _ { \lambda } d _ { \lambda } ^ { 2 }$ . A cutof at <sup>2</sup><sub>11</sub> gives $1 + 1 1 ^ { 2 } = 1 2 2$ parameters and a cutof at $\frac 1 3$ gives $1 2 2 + 5 4 ^ { 2 } = 3 0 3 8$ . With 500 evaluations, the first is estimable and the second is not, so the cutof is chosen to be $\frac { 2 } { 1 1 }$

We now identify the 122 dimensional space concretely, rather than as an abstract isotypic sum. Consider the permutation matrices $P ( \pi )$ with $( j , i )$ entry $\delta _ { \pi ( i ) , j }$ . This matrix is $1 2 \times 1 2$ , and so we have 144 functions $\pi \mapsto \delta _ { \pi ( i ) , j } .$ The map P is the natural permutation representation on 12 points, which decomposes as $( 1 2 ) \oplus ( \mathrm { { i } 1 , 1 ) }$ , and so every matrix entry $\delta _ { \pi ( i ) , j }$ lies in the span of the two surviving blocks after the cutof. Conversely, the span is all of it, and we can check the dimension by hand. We ask which matrices $A = \left( a _ { i j } \right)$ give

$$
\sum _ { i , j } a _ { i j } \delta _ { \pi ( i ) , j } = \sum _ { i } a _ { i , \pi ( i ) } = 0\tag{86}
$$

for every permutation π. Comparing two permutations that difer by a transposition forces $a _ { i j } = r _ { i } + c _ { j }$ for some constants $r _ { i }$ and $c _ { j } .$ , and it follows that

$$
\sum _ { i } r _ { i } = - \sum _ { i } c _ { \pi ( i ) } = - \sum _ { i } c _ { i } .\tag{87}
$$

The pairs $( r , c )$ give 24 parameters, minus 1 for the shift $( r + t , c - t )$ that leaves A unchanged, and minus 1 for the sum condition (87). That leaves a 22-dimensional kernel, so the span has dimension $1 4 4 - 2 2 = 1 2 2$ as expected.

What survives is span $\{ \pi \mapsto \delta _ { \pi ( i ) , j } \}$ , the first order model of ranked data [4, 13], in which the surrogate is

$$
\widetilde f ( \pi ) = \sum _ { i , j } a _ { i j } \delta _ { \pi ( i ) , j }\tag{88}
$$

for a matrix of module-position energies. The declaration has done two things at once. It has cut the parameter count from 479001600 to 122, which is what makes the fit possible from one budget of folds, and it has left an objective whose minimization over $\mathbb { S } _ { 1 2 }$ is a linear assignment problem and therefore exactly solvable in cubic time rather than by search. The four choices were decided by four diferent features of the problem. What the energy is invariant under fixed the domain and group, the designer’s declaration of which rearrangements count as incremental fixed the generating set, the conjugacy structure of that set determined which further beliefs the prior could carry, and the evaluation budget fixed the bandlimit. Together, these choices produce an inductive bias which simplifies the problem in much the same way a symmetry constraint does in geometric deep learning [36, 37, 38] and perhaps more explicitly so its quantum counterpart known as geometric quantum machine learning [39, 40, 41].

## 7 Smoothness on Compact Groups

The construction of Section 2 used finiteness of G in exactly two places: to normalize the counting measure and to draw the Cayley graph. The first is inessential and the second is not, and separating the two clarifies what the ordering parameter really depends on. In this section, we let G be a compact Hausdorf group with normalized Haar measure $d g .$ , and we write $L ^ { 2 } ( G )$ for the associated Hilbert space with inner product

$$
\langle f , h \rangle = \int _ { G } { \overline { { f ( g ) } } } h ( g ) \ d g .\tag{89}
$$

The Peter–Weyl theorem gives an orthogonal decomposition

$$
L ^ { 2 } ( G ) = \bigoplus _ { \sigma \in { \widehat { G } } } A _ { \sigma } ,\tag{90}
$$

where $\mathcal { A } _ { \sigma } \cong \mathcal { H } _ { \sigma } \otimes \mathcal { H } _ { \sigma } ^ { * }$ and $\mathcal { H } _ { \sigma }$ is the carrier space for the irrep σ. The decomposition is an orthogonal Hilbert direct sum over ${ \widehat { G } } ,$ , with each $d _ { \sigma }$ finite, and the matrix entries $\sqrt { d _ { \sigma } } \sigma _ { i j }$ again form an orthonormal basis. Fix a finite symmetric set $S \subset G$ with $e \not \in S$ and define the averaging operator A and the Laplacian $\mathcal { L }$ as we

did before. Each right translation is unitary on $L ^ { 2 } ( G )$ , so A is a convex combination of unitaries and L is bounded with $\| A \| \leq 1$ . Both of the following basic results carry over without much modification as neither proof uses finiteness.

Proposition 10. Let $f \in L ^ { 2 } ( G )$ . Then

$$
\langle f , \mathcal { L } f \rangle = \frac { 1 } { 2 | S | } \sum _ { s \in S } \int _ { G } | f ( g ) - f ( g s ) | ^ { 2 } \ d g .\tag{91}
$$

Proposition 11. The Laplacian L preserves each $A _ { \sigma }$ and acts there as $I _ { d _ { \sigma } } \otimes ( I _ { d _ { \sigma } } - M _ { \sigma } )$ , where $M _ { \sigma } =$ $\frac { 1 } { | S | } \sum _ { s \in S } \sigma ( s )$ . Consequently, $\omega ( \sigma ) = 1 - \mathrm { T r } [ M _ { \sigma } ] / d _ { \sigma }$ is defined for every $\sigma \in { \widehat { G } }$ and satisfies $\omega ( \sigma ) \in [ 0 , 2 ]$

The proof of Proposition 10 is that of Proposition 1 with the sum over G replaced by the Haar integral, using right invariance of $d g$ in place of the re-indexing of a finite sum. The proof of Proposition 11 is that of Proposition 2 verbatim, since it uses only $\sigma$ as a finite-dimensional unitary homomorphism. The ordering parameter is therefore available for every compact group, and it is again an invariant of the pair (G, S).

What does not carry over is the Cayley graph picture. For continuous $G ,$ the vertex set can be uncountable, and more seriously, the graph is never connected when G is infinite. Indeed, a compact Hausdorf group is homogeneous, so it has no isolated points unless it is discrete, and a countable space without isolated points is not Baire; hence an infinite compact Hausdorf group is uncountable. A finitely generated subgroup is countable and therefore proper. The components are the cosets of $\langle S \rangle$ , and the word metric is finite only within a component. The graph is thus a device internal to $G$ rather than a combinatorial model of $G ,$ and the geometric reading in which nearby group elements difer by few generators is not available. What replaces it is Proposition 10. The quadratic form survives intact as an integral, and it is this functional rather than the graph that the ordering measures.

Now suppose in addition that $G$ is a compact connected Lie group. A diferent limit recovers a geometric picture. Replacing the finite set S by a family of conjugation invariant probability measures concentrated near the identity and imposing the usual difusive scaling and second-moment hypotheses produces a family of averaging operators whose associated Laplacians converge, after suitable rescaling, to the Laplace–Beltrami operator [42, 43] of an Ad-invariant metric on $G ;$ that is, to the Casimir element. For $G$ connected with simple Lie algebra, such a metric is unique up to scale, so the induced frequency ordering is canonical up to an overall positive scale. That limiting ordering is the familiar one. The Casimir eigenvalue is the weight against which the Sobolev scale on $\widehat { G }$ is built [44], and smoothness in the classical sense is equivalent to superpolynomial decay of the Fourier coeficients measured against it [45]. The scaling limit is also the difusion half of Hunt’s classification of the generators of convolution semigroups on a Lie group [46], the jump half of which is the continuous counterpart of the weighted Cayley-Laplacians of Proposition 7. In particular, when G has simple Lie algebra, the dependence on a generating set disappears, up to an overall positive scale, in this conjugation-invariant infinitesimal limit.

## 8 Conclusion

We have argued that smoothness on $L ^ { 2 } ( G )$ is not a property a function possesses on its own. A function is smooth relative to a declaration of which group elements count as a small step, and once that declaration is made, the Cayley-Laplacian construction determines an ordering. The declaration is a symmetric generating set $S ,$ the structure it induces is the Cayley graph of $( G , S )$ , and the roughness of a function is the Dirichlet energy of that graph. Because the graph is a Cayley graph, its Laplacian is block diagonal over the dual, and the mean of the eigenvalues in the block at σ is a single number $\omega ( \sigma )$ attached to the irrep rather than to a choice of basis. Ordering $\widehat { G }$ by ω is what makes the phrase “concentrated at low frequency” meaningful for a non-abelian group.

Section 3 measures how much of this was forced. The four requirements, that the operator be Hermitian, that it distinguish no point of $G ,$ that it assign zero energy to the constant functions, and that the energy it assigns not depend on a choice of complex conjugate, turn out to constrain the ordering only up to reality, agreement on conjugate pairs, and vanishing at the trivial representation. Every function on $\widehat { G }$ with those properties is realized by an admissible operator, and the orderings $\omega _ { K }$ coming from inversion orbits of conjugacy classes are a basis for them. So these axioms should be read as supplying coordinates rather than a restriction; an ordering becomes a weight vector on $\kappa ,$ , and a symmetric generating set is the case in which that vector is the probability distribution induced on the classes the set meets. The restrictions come afterward. Nonnegativity of the weights identifies a natural positive semidefinite subclass, namely the weighted Cayley-Laplacians, among all admissible operators. Adding locality and uniformity with respect to a fixed generating set leaves only the Laplacian of Section 2 up to positive scale.

Two further layers of dependence emerged along the way. When the domain is a set rather than a group, the acting group determines which frequencies exist at all, since an irrep is visible precisely when it contains a vector fixed by the stabilizer, while the generating set orders whatever survives. These choices are independent, and the examples of Section 5 show that they can be made to disagree sharply, with $\mathbb { Z } _ { n }$ and $\mathbb { S } _ { n }$ acting on the same n labels producing an ordering of n frequencies in one case and of two in the other. Within a fixed group, changing the generating set can invert the ordering outright, as the sign representation of $D _ { 3 }$ illustrates by moving from second to last.

Several directions remain. The ordering parameter is a mean over a block, and when S is not closed under conjugation, the block genuinely splits, so the ordering on $\widehat { G }$ is strictly coarser than the ordering on the spectrum of $\mathcal { L } .$ For compact groups, we established that the block diagonalization and the bounds persist, but the characterization theorem does not transfer directly, since its proof uses finiteness both in decomposing an invariant function over the partition G by conjugacy classes and in the evaluation argument establishing linear independence. A compact analogue replacing K by conjugation invariant measures seems plausible and would complete the picture. Hunt’s theorem indicates the form that analogue should take, with a Casimir term in place of the diagonal of $\varphi$ and a conjugation invariant L´evy measure in place of the weights $\alpha _ { K }$ [46, 47].

## References

[1] Vasilis Belis, Joseph Bowles, Rishabh Gupta, Evan Peters, and Maria Schuld. Spectral methods: crucial for machine learning, natural for quantum computers? arXiv preprint arXiv:2603.24654, 2026.

[2] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.

[3] Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, and Zheng Ma. Frequency principle: Fourier analysis sheds light on deep neural networks. arXiv preprint arXiv:1901.06523, 2019.

[4] Persi Diaconis. Group representations in probability and statistics, volume 11. Institute of mathematical statistics Hayward, CA, 1988.

[5] L´aszl´o Babai. Spectra of Cayley graphs. Journal of Combinatorial Theory, Series B, 27(2):180–189, 1979.

[6] Audrey Terras. Fourier analysis on finite groups and applications. Number 43. Cambridge University Press, 1999.

[7] Persi Diaconis and Mehrdad Shahshahani. Generating a random permutation with random transpositions. Zeitschrift f¨ur Wahrscheinlichkeitstheorie und verwandte Gebiete, 57(2):159–179, 1981.

[8] Pietro Caputo, Thomas Liggett, and Thomas Richthammer. Proof of Aldous’ spectral gap conjecture. Journal of the American Mathematical Society, 23(3):831–851, 2010.

[9] Peter F Stadler. Landscapes and their correlation functions. Journal of Mathematical chemistry, 20(1):1– 45, 1996.

[10] Wim Hordijk and Peter F Stadler. Amplitude spectra of fitness landscapes. Advances in Complex Systems, 1(01):39–66, 1998.

[11] Dan Rockmore, Peter Kostelec, Wim Hordijk, and Peter F Stadler. Fast fourier transform for fitness landscapes. Applied and Computational Harmonic Analysis, 12(1):57–76, 2002.

[12] Mahya Ghandehari, Dominique Guillot, and Kristopher Hollingsworth. A non-commutative viewpoint on graph signal processing. In 2019 13th International conference on Sampling Theory and Applications (SampTA), pages 1–4. IEEE, 2019.

[13] Yilin Chen, Jennifer DeJong, Tom Halverson, and David I Shuman. Signal processing on the permutahedron: tight spectral frames for ranked data analysis. Journal of Fourier Analysis and Applications, 27(4):70, 2021.

[14] Kathryn Beck, Mahya Ghandehari, Skyler Hudson, and Jenna Paltenstein. Frames for signal processing on cayley graphs. Journal of Fourier Analysis and Applications, 30(6):66, 2024.

[15] Iskander Azangulov, Andrei Smolensky, Alexander Terenin, and Viacheslav Borovitskiy. Stationary kernels and gaussian processes on lie groups and their homogeneous spaces i: the compact case. Journal of Machine Learning Research, 25(280):1–52, 2024.

[16] Antonio Ortega, Pascal Frossard, Jelena Kovaˇcevi´c, Jos´e MF Moura, and Pierre Vandergheynst. Graph signal processing: Overview, challenges, and applications. Proceedings of the IEEE, 106(5):808–828, 2018.

[17] Geert Leus, Antonio G Marques, Jos´e MF Moura, Antonio Ortega, and David I Shuman. Graph signa processing: History, development, impact, and outlook. IEEE Signal Processing Magazine, 40(4):49–60, 2023.

[18] Xiaowen Dong, Dorina Thanou, Laura Toni, Michael Bronstein, and Pascal Frossard. Graph signa processing for machine learning: A review and new perspectives. IEEE Signal processing magazine, 37(6):117–127, 2020.

[19] William Fulton and Joe Harris. Representation theory: a first course. Springer Science & Business Media, 2013.

[20] Walter Rudin. Fourier analysis on groups, volume 12. Interscience publishers New York, 1962.

[21] Ryan O’Donnell. Analysis of boolean functions, volume 2. Cambridge University Press Cambridge, 2014.

[22] Zachary P Bradshaw, Margarite L LaBorde, and Dillon Montero. Symmetry-based quantum codes beyond the Pauli group. Physical Review A, 113(4):042431, 2026.

[23] Soorya Rethinasamy, Margarite L LaBorde, and Mark M Wilde. Logarithmic-depth quantum circuits for Hamming weight projections. Physical Review A, 110(5):052401, 2024.

[24] Fan RK Chung. Spectral graph theory, volume 92. American Mathematical Soc., 1997.

[25] David I Shuman, Sunil K Narang, Pascal Frossard, Antonio Ortega, and Pierre Vandergheynst. The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. IEEE signal processing magazine, 30(3):83–98, 2013.

[26] Peter F Stadler. Fitness landscapes. In Biological evolution and statistical physics, pages 183–204. Springer, 2002.

[27] I Martin Isaacs. Character theory of finite groups, volume 69. Courier Corporation, 1994.

[28] Tullio Ceccherini-Silberstein, Fabio Scarabotti, and Filippo Tolli. Harmonic analysis on finite groups. Cambridge studies in advanced mathematics, 108, 2008.

[29] Leilani H Gilpin, David Bau, Ben Z Yuan, Ayesha Bajwa, Michael Specter, and Lalana Kagal. Explaining explanations: An overview of interpretability of machine learning. In 2018 IEEE 5th International Conference on data science and advanced analytics (DSAA), pages 80–89. IEEE, 2018.

[30] W James Murdoch, Chandan Singh, Karl Kumbier, Reza Abbasi-Asl, and Bin Yu. Definitions, methods, and applications in interpretable machine learning. Proceedings of the National Academy of Sciences, 116(44):22071–22080, 2019.

[31] Kaitlin Gili and Zachary P Bradshaw. Inherent interpretability provides inherent value in quantum machine learning. arXiv preprint arXiv:2607.13827, 2026.

[32] Douglas H Turner and David H Mathews. Nndb: the nearest neighbor parameter database for predicting stability of nucleic acid secondary structure. Nucleic acids research, 38(suppl 1):D280–D282, 2010.

[33] Michael Zuker and Patrick Stiegler. Optimal computer folding of large rna sequences using thermodynamics and auxiliary information. Nucleic acids research, 9(1):133–148, 1981.

[34] Rune B Lyngsø and Christian NS Pedersen. Pseudoknots in rna secondary structures. In Proceedings of the fourth annual international conference on Computational molecular biology, pages 201–209, 2000.

[35] Maurice G Kendall et al. A new measure of rank correlation. Biometrika, 30(1-2):81–93, 1938.

[36] Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 34(4):18–42, 2017.

[37] Michael M Bronstein, Joan Bruna, Taco Cohen, and Petar Veliˇckovi´c. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

[38] Jan E Gerken, Jimmy Aronsson, Oscar Carlsson, Hampus Linander, Fredrik Ohlsson, Christofer Petersson, and Daniel Persson. Geometric deep learning and equivariant neural networks. Artificial Intelligence Review, 56(12):14605–14662, 2023.

[39] Johannes Jakob Meyer, Marian Mularski, Elies Gil-Fuster, Antonio Anna Mele, Francesco Arzani, Alissa Wilms, and Jens Eisert. Exploiting symmetry in variational quantum machine learning. PRX quantum, 4(1):010328, 2023.

[40] Mart´ın Larocca, Fr´ed´eric Sauvage, Faris M Sbahi, Guillaume Verdon, Patrick J Coles, and Marco Cerezo. Group-invariant quantum machine learning. PRX quantum, 3(3):030341, 2022.

[41] Zachary P Bradshaw, Ethan N Evans, Matthew Cook, and Margarite L LaBorde. Learning equivariant maps with variational quantum circuits. Physical Review Applied, 23(4):044007, 2025.

[42] David Applebaum. Probability on compact Lie groups. Springer, 2014.

[43] Ming Liao. L´evy processes in Lie groups, volume 162. Cambridge university press, 2004.

[44] Michael Ruzhansky and Ville Turunen. Pseudo-diferential operators and symmetries. Background analysis and advanced topics, Pseudo-Diferential Operators. Theory and Applications, 2, 2010.

[45] Mitsuo Sugiura. Fourier series of smooth functions on compact lie groups. Osaka Journal of Mathematics, 8:33–47, 1971.

[46] Gilbert Agnew Hunt. Semi-groups of measures on lie groups. Transactions of the American Mathematical Society, 81(2):264–293, 1956.

[47] David Applebaum. Infinitely divisible central probability measures on compact lie groups—regularity, semigroups and transition kernels. The Annals of Probability, 39(6):2474–2496, 2011.