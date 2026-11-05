# Softmax Attention with Any Temperature in Fixed Dimension: Matching the Best Known Maximum-Inner-Product Bounds

Zixuan Huang<sup>1,2</sup> Guozhang Chen<sup>1,3∗</sup>

<sup>1</sup> State Key Laboratory of Multimedia Information Processing, School of Computer Science <sup>2</sup> School of Physics

<sup>3</sup> National Engineering Research Center of Visual Technology, School of Computer Science Peking University, Beijing 100871, China

2400011442@stu.pku.edu.cn guozhang.chen@pku.edu.cn

## Abstract

Attention is central to modern AI models for language, vision, and scientific modeling. However, direct evaluation compares every query with every key, requiring quadratic time in the context length, which limits its scalability to long contexts. Consequently, characterizing the complexity bounds of attention is of fundamental importance.

For every fixed dimension d, we give an algorithm for m queries, n key–value pairs, and additive error ε, with expected time

$$
\widetilde { \mathrm { O } } ( ( m n ) ^ { k / ( k + 1 ) } + m + n ) , \qquad k = \lfloor d / 2 \rfloor .
$$

For fixed dimension and a fixed number of value channels, soft-O hides factors of the form $\mathrm { p o l y } _ { d } \big ( b , \log ( m + n ) , \log B , \log ( 1 / \varepsilon ) \big )$ , where b bounds input encoding lengths and B bounds coordinate and value magnitudes. In particular, the dependence on B and $1 / \varepsilon$ is polylogarithmic, which allows arbitrary positive temperature. When $m = n$ and $d \geq 2$ , the bound is $\widetilde { \mathrm { O } } ( n ^ { 2 - 2 / ( \lfloor d / 2 \rfloor + 1 ) } )$ : near-linear in dimensions two and three, with exponent $4 / 3$ in dimensions four and five and $3 / 2$ in dimensions six and seven. It improves the previous $\widetilde { \mathrm { O } } ( n ^ { 2 - 1 / d } )$ upper bound and matches the best known batch maximum-inner-product bound in each fixed dimension. Additionally, we reduce integer batch maximum inner product computation to one attention call with the same dimension and input sizes $m , n .$

## Contents

1 Introduction 1   
1.1 Results . 2   
1.2 Technical Overview 4   
1.2.1 Normalization and the Planar Base Case 4   
1.2.2 Temperature Interpolation and Common-Anchor Blocks 5   
1.2.3 Random Prefix Chains and Heavy/Light Recursion 6   
1.3 Related Work . 8   
2 Preliminaries 9   
2.1 Computation and Error Bounds . 11   
2.2 Random Sampling 11   
2.3 Analytic Estimates 11   
3 A Dimension-Preserving Reduction from Maximum Inner Product 13   
3.1 Computing Exact Maxima from Attention 13   
3.2 Conditional Hardness in Superconstant Dimension 14   
4 A Deterministic Algorithm in Two Dimensions 14   
4.1 Shifted-Kernel Products in the Plane 15   
4.2 Centering and Radial Coordinates 15   
4.3 Radial and Boundary Ranges 16   
4.4 Polynomial Approximation and Moment Sweeps 17   
4.5 Kernel Products and Attention 18   
4.6 Online Queries after Preprocessing the Keys 19   
5 Shifted Kernels and Dimension Induction 19   
5.1 Exact Maxima and Normalized Directions 20   
5.2 Shifted-Kernel Products 20   
5.3 Biased Products and Temperature Interpolation 21   
5.4 The Induction on the Dimension 22   
6 Geometric Data Structures in Fixed Dimension 23   
6.1 Classical Geometric Primitives 23   
6.2 Support Queries and Strict Reporting 23   
6.3 Envelope Cells 24   
6.4 Sampling Bounds 25   
Biased Exponential Products and the Main Theorem 25   
7.1 Orientations, Hits, and Parameters 26   
7.2 Anchor Blocks and Direct Blocks 27   
7.3 Random Prefix Chains . 28   
7.4 The Full Random Prefix 32   
7.5 Truncated Prefixes and the Heavy/Light Recursion 33   
7.6 Error Analysis and the Proof of Proposition 5.6 34   
7.7 Proof of Theorem 1.1 35   
8 A Static-Query Comparison with Semigroup Halfplane Aggregation 35   
9 Conclusion and Further Directions 36   
A Numerical Stability and Precision 38   
A.1 Input Size, Temperature, and Reduction Cost . 38   
A.2 Arithmetic for Planar Kernel Products 38   
A.3 Arithmetic for Biased Block Products 41   
A.4 Precision under Composition and Recursion . 41   
A.5 Exact Geometric Computation 42   
A.6 Uniform Perturbations and Caps 43   
A.7 Support Queries and Envelope Construction . 44   
A.8 Interior Shifts for Cell Location 45   
A.9 Integer Encodings of Halfspace Ranges 45   
B Geometric Constructions and Sampling Proofs 45   
B.1 Classical Polytope Facts . . 45   
B.2 Constructing Cuttings with a Fixed Bounding Simplex . 46   
B.3 Strict Halfspace Reporting . 51   
B.4 Canonical Cells and Point Location 51   
B.5 Sampling and Potential Estimates 53   
B.6 Implementing Batch Maximum Inner Product . 54   
AI Methodology 56   
Acknowledgment 56   
References 56

## 1 Introduction

Transformers and related attention-based architectures are central to modern AI, with wide applications in areas such as language processing [VSP+17], visual recognition [DBK+21], and biomolecular structure prediction [AAD+24]. By exponentiating query–key inner products into a sharp distribution concentrated on the most relevant keys, attention also exhibits close connections to classical problems in Theoretical Computer Science (TCS), such as Maximum Inner Product (Max-IP) problem, making it a fundamental subject of study in TCS.

As frontier problems grow in complexity, extending the context length of Transformers has become essential. However, the cost of direct attention evaluation grows quadratically, with $n ^ { 2 }$ pairwise interactions for a context of length n even in fixed dimension. This quadratic bottleneck restricts the deployment of Transformers in a wide range of long-context applications, such as repository-level code generation or full-document analysis.

Practical algorithms reduce the cost of attention through memory-eficient exact evaluation, as in FlashAttention [DFERR22], and random-feature approximation with provable error guarantees, as in Performer [CLD+21]. Nevertheless, these methods did not establish a non-trivial asymptotic upper bound for attention that faithfully preserves the exact softmax semantics. Establishing theoretical upper bounds for attention helps us delineate the fundamental boundaries of where attention can be applied.

Formally, softmax attention maps queries, keys, and values to normalized exponential averages. For query vectors $\pmb q _ { 1 } , \dots , \pmb q _ { m } \in \mathbb { R } ^ { d }$ , key vectors $\pmb { k } _ { 1 } , \ldots , \pmb { k } _ { n } \in \mathbb { R } ^ { d }$ , and scalar values $v _ { 1 } , \ldots , v _ { n }$ , the output associated with query $\pmb q _ { i }$ is

$$
\frac { \sum _ { j = 1 } ^ { n } \mathrm { e } ^ { \boldsymbol { q } _ { i } \cdot \boldsymbol { k } _ { j } } \boldsymbol { v } _ { j } } { \sum _ { j = 1 } ^ { n } \mathrm { e } ^ { \boldsymbol { q } _ { i } \cdot \boldsymbol { k } _ { j } } } .
$$

More generally, each key $k _ { j }$ carries a vector $\pmb { v } _ { j } = ( v _ { j 1 } , \dots , v _ { j r } )$ of r values, one for each value channel, and the output of a query has one such average for each channel. Direct evaluation processes all mn query–key pairs, although the input and output sizes are only linear in $m + n$ when the dimension and number of value channels are fixed.

An inverse temperature $\beta > 0$ rescales each score to $\beta q _ { i } \cdot k _ { j }$ and is absorbed into the query coordinates. The polylogarithmic dependence on coordinate magnitudes therefore extends the computational bounds to every positive rational temperature (Remark 2.3).

Gupta, Huang, Saha, $\mathrm { X u . }$ , and Ye [GHSXY26] gave an algorithm with running time

$$
\widetilde { \mathrm { O } } ( n ^ { 2 - 1 / d } )
$$

for every fixed dimension d. For each query, their algorithm retains keys whose scores lie within a logarithmic window of the row maximum and approximates the exponential on that window by a polynomial. It evaluates the resulting monomial sums with a simplex range-searching data structure [Mat92b], whose query cost yields the exponent $2 - 1 / d .$ Their reduction from maximum inner product also established conditional hardness in dimension $2 ^ { \mathrm { O } ( \log ^ { * } n ) }$ via the lower bound of Chen [Che20]. They further observe that attention whose query or key matrix has rank at most d reduces to dimension d through a rank factorization [GHSXY26, Theorem 1.2], so fixeddimensional algorithms also apply to low-rank attention with a large head dimension. Additionally, they gave a reduction from d-dimensional Max-IP to $( d + 1 )$ -dimensional attention, where Max-IP is the natural geometric core of attention: given query vectors $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and points $\mathbf { \Delta } _ { \mathbf { \mathcal { Y } } \mathcal { I } }$ , the batch problem asks for all maxima $M _ { i } = \operatorname* { m a x } _ { j } { \pmb x _ { i } } \cdot { \pmb y _ { j } }$

However, the above upper bound is not suficiently tight, considering that in fixed dimension $d ,$ the best known algorithms answer m maximum-inner-product queries on n points in $\widetilde { \mathrm { O } } ( ( m n ) ^ { k / ( k + 1 ) } + m +$ n) time, where $k = \lfloor d / 2 \rfloor$ [MS93; Cha96b]; for $d \leq 3$ this is near-linear [PS85; DK90]. Furthermore, because the reduction increases the dimension by one, one cannot obtain non-trivial hardness for 4-dimensional attention.

While Max-IP only seeks the extreme inner product, computing attention requires aggregating the exponentially weighted contributions of all keys. This leaves the question whether the global aggregation in softmax attention makes it strictly harder than Max-IP, or if we can still match the best known complexity of batch Max-IP and give a dimension-preserving reduction in every fixed dimension.

In this work, we study the bit complexity of softmax attention, with prescribed entrywise additive error ε. Our running-time bounds depend polynomially on the input encoding length and on $\log ( 1 / \varepsilon )$ , so inverse-polynomial error contributes only polylogarithmic factors.

## 1.1 Results

In every fixed dimension, our attention algorithm matches the best known dependence on m and n for batch maximum inner product. To state the result, let $N = m + n$ , suppose the input coordinates are rationals of magnitude at most $B \geq 1$ and encoding length at most $b ,$ and let $\varepsilon$ be the target additive error.

Theorem 1.1 (Attention in every fixed dimension; proof in Section 7.7). Fix integers d $\geq 1$ and $r \geq 1$ and let $k = k ( d ) = \lfloor d / 2 \rfloor$ . There is a Las Vegas algorithm for Attention (Definition 2.1) with r value channels whose expected running time is

$$
\widetilde { \mathrm { O } } ( ( m n ) ^ { k / ( k + 1 ) } + m + n ) .
$$

Here soft-O suppresses factors of the form

$$
\mathrm { p o l y } _ { d } \big ( b , \log ( m + n ) , \log B , \log ( 1 / \varepsilon ) \big ) ,
$$

with constants that may also depend on the fixed number r of value channels.

For $d ~ \geq ~ 2 , ~ m ~ = ~ n , ~ b ~ = ~ \mathrm { p o l y l o g } n$ and $B , \varepsilon ^ { - 1 } \ = \ n ^ { \mathrm { O ( 1 ) } }$ , the expected running time is $\widetilde { \mathrm { O } } ( n ^ { 2 - 2 / ( \lfloor d / 2 \rfloor + 1 ) } )$ . This is near-linear for $d \in \{ 2 , 3 \}$ ; Table 1 lists the exponents for $d \leq 9$ and compares them with the previous bounds. For $d = 1$ the bound is ${ \widetilde { \mathrm { O } } } ( m + n ) ;$ an $\widetilde { \mathrm { O } } ( n )$ algorithm for this case was already known [GHSXY26, Lemma A.1]. Since $2 / ( \lfloor d / 2 \rfloor + 1 ) > 1 / d$ for every $d \geq 2$ Theorem 1.1 improves the bound $\widetilde { \mathrm { O } } ( n ^ { 2 - 1 / d } )$ of [GHSXY26] in every dimension $d \geq 2$ , while retaining their polylogarithmic dependence on B and $1 / \varepsilon$ . Moreover, in the plane $( d = 2 )$ , our algorithm is completely deterministic and runs in $\widetilde { \mathrm { O } } ( n )$ time (Theorem 4.1).

Writing $F _ { d } ( m , n ) = ( m n ) ^ { k ( d ) / ( k ( d ) + 1 ) } + m + n$ , the same algorithm supports a variable number r of value channels in expected time $r \tilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ ), where soft-O suppresses factors polynomial in log(r + 2).

Batch maximum inner product also reduces to attention in the same dimension. For integer matrices $\pmb { X } \in \mathbb { Z } ^ { m \times d }$ and $Y \in \mathbb { Z } ^ { n \times d }$ with rows $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and ${ \bf { \nabla } } _ { \bf { { y } } } \mathrm { { _ { j } } }$ , the problem BatchMaxI $\mathrm { P } _ { d }$ asks for every $M _ { i } = \operatorname* { m a x } _ { j } { \pmb x _ { i } } \cdot { \pmb y _ { j } }$

When the values equal the keys, the inner product of the attention output with the original query is a softmax average of the scores. Its distance from the maximum is at most $( \ln n ) / \beta _ { 0 }$ , so integer rounding determines every maximum exactly at a suitable precision; Section 3.1 contains the formal reduction and proof. Whereas the earlier reduction of [GHSXY26] used polylogarithmically many calls in dimension $d + 1$ , Theorem 1.2 reduces batch maximum inner product to a single attention call within the same dimension d.

Table 1: Time bounds and the ofline counting baseline derived in Section 1.2.1 by combining polynomial approximation with ofline halfspace range counting. Here d is fixed, $m = n$ , coordinate encoding lengths are polylogarithmic, magnitudes are polynomial, and $\varepsilon = n ^ { - \mathrm { O } ( 1 ) }$ . The exponent is the same in dimensions 2k and $2 k + 1$ , where $k = \lfloor d / 2 \rfloor$ ; the algorithm is Las Vegas for $d \geq 3$ and deterministic for $d = 2$ . The last column lists the best known deterministic real-RAM bound for exact batch maximum inner product [PS85; DK90; MS93; Cha96b], which reduces to attention by Theorem 3.1.
<table><tr><td rowspan="2">Dimension</td><td colspan="4">Time upper bounds  $\widetilde { \mathrm { O } } ( \cdot )$ </td></tr><tr><td>Prior work [GHSXY26]</td><td>Offline counting baseline</td><td></td><td>This work Batch Max-IP</td></tr><tr><td>2</td><td> $n ^ { 3 / 2 }$ </td><td> $n ^ { 4 / 3 }$ </td><td>n</td><td>n</td></tr><tr><td>3</td><td> $n ^ { 5 / 3 }$ </td><td> $n ^ { 3 / 2 }$ </td><td> $n$ </td><td>n</td></tr><tr><td>4</td><td> $n ^ { 7 / 4 }$ </td><td> $n ^ { 8 / 5 }$ </td><td> $n ^ { 4 / 3 }$ </td><td> $n ^ { 4 / 3 }$ </td></tr><tr><td>5</td><td> $n ^ { 9 / 5 }$ </td><td> $n ^ { 5 / 3 }$ </td><td> $n ^ { 4 / 3 }$ </td><td> $n ^ { 4 / 3 }$ </td></tr><tr><td>6</td><td> $n ^ { 1 1 / 6 }$ </td><td> $n ^ { 1 2 / 7 }$ </td><td> $n ^ { 3 / 2 }$ </td><td> $n ^ { 3 / 2 }$ </td></tr><tr><td>7</td><td> $n ^ { 1 3 / 7 }$ </td><td> $n ^ { 7 / 4 }$ </td><td> $n ^ { 3 / 2 }$ </td><td> $n ^ { 3 / 2 }$ </td></tr><tr><td>8</td><td> $n ^ { 1 5 / 8 }$ </td><td> $n ^ { 1 6 / 9 }$ </td><td> $n ^ { 8 / 5 }$ </td><td> $n ^ { 8 / 5 }$ </td></tr><tr><td>9</td><td> $n ^ { 1 7 / 9 }$ </td><td> $n ^ { 9 / 5 }$ </td><td> $n ^ { 8 / 5 }$ </td><td> $n ^ { 8 / 5 }$ </td></tr><tr><td> $d \geq 2$ </td><td> $n ^ { 2 - 1 / d }$ </td><td> $n ^ { 2 - 2 / ( d + 1 ) }$ </td><td> $n ^ { 2 - 2 / ( k + 1 ) }$ </td><td> $n ^ { 2 - 2 / ( k + 1 ) }$ </td></tr></table>

Theorem 1.2 (Dimension-preserving reduction, informal). The problem BatchMaxI $\mathrm { P } _ { d }$ reduces to one attention call with d value channels, namely

$$
\begin{array} { r } { Q = \beta _ { 0 } { X } , \qquad K = { Y } , \qquad { V } = { Y } , \qquad \beta _ { 0 } = \Theta ( \log ( n + 1 ) ) . } \end{array}
$$

For a suitable absolute constant in $\beta _ { 0 }$ , the attention instance has the same $m , n , d$ and uses inversepolynomial error for polynomially bounded integer inputs; the reduction requires deterministic near-linear additional bit operations for fixed d. Scaling V by dB, where B bounds the integer coordinates, yields a constant-error version.

The reduction also transfers the conditional quadratic lower bound for maximum inner product [SXY26] under the Orthogonal Vectors Hypothesis (OVH) [Wil05] to every constructible dimension $d ( n )$ with $d ( n ) \to \infty$ and $d ( n ) = n ^ { \mathrm { o ( 1 ) } }$ . This hardness consequence was already observed in [SXY26] through the earlier reduction. Section 3.2 gives the precise statement.

Consequences in fixed dimension. Theorems 1.1 and 1.2 show that, in every fixed dimension, the best known exponents for attention and for batch maximum inner product coincide. Hence, for $d \geq 4$ , an attention algorithm with exponent below $2 - 2 / ( \lfloor d / 2 \rfloor + 1 )$ on instances with d value channels, polynomial magnitudes, and inverse-polynomial or constant error would improve the exponent for integer batch maximum inner product with polynomially bounded coordinates, and hence for ofline halfspace emptiness, in the same dimension. Applying a single-channel attention algorithm to each of the d value columns gives this implication as well (Theorem 3.1(c)), since d is fixed. Both problems admit near-linear algorithms for $d \leq 3$ , whereas whether improvements for batch maximum inner product in higher dimensions transfer back to attention remains open (Section 9).

The planar algorithm also yields a static data structure that preprocesses fixed keys, values, numerical bounds, and error tolerance in $\widetilde { \mathrm { O } } ( n )$ time and space and answers each query satisfying these bounds in $\widetilde { \mathrm { O } } ( 1 )$ time (Corollary 4.6). For exact halfplane aggregation over a faithful semigroup with near-linear storage, a query may require $\widetilde { \Omega } ( n ^ { 1 / 3 } )$ operations [AMX12] (Section 8). The attention data structure avoids this barrier by using exponential decay to sum over a geometrically convenient superset of near-maximal keys with a small omitted contribution. Similarly, while relative-error halfplane counting admits near-linear storage through random sampling [AH08], with query time polylogarithmic in the input size and polynomial in the reciprocal of the relative error, the attention bound is polylogarithmic in both the input size and $1 / \varepsilon$

![](images/ec69ce206756796622e55dfed1f9c556f41a626be802b8f44338c0b8da805129.jpg)  
Figure 1: Scores in one attention row. The shaded interval contains the near-maximal keys; each omitted key contributes at most $B \mathrm { e } ^ { - L }$ to the shifted numerator.

## 1.2 Technical Overview

Subtracting the maximum score from each row leaves the attention output unchanged and puts the exponential weights in $( 0 , 1 ]$ , with a weight of 1 in every row. By exponential decay, the total contribution from keys outside a logarithmic window of the maximum is small, so it sufices to aggregate the potentially numerous keys within that window.

In the plane, we describe these keys by radial and boundary ranges over which polynomial moments can be summed in near-linear time. In higher dimensions, we cover the significant pairs by pairwise disjoint Cartesian blocks $I \times J$ and evaluate them through lower-dimensional kernels, in total expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ . For $d \geq 3$ , the reductions are

$$
\mathrm { A t t e n t i o n } _ { d } \to \mathrm { K P } _ { d } \xrightarrow { \mathrm { t e m p e r a t u r e ~ i n t e r p o l a t i o n } } \mathrm { B P } _ { d - 1 } \xrightarrow { \mathrm { a n c h o r ~ b l o c k s } } \mathrm { K P } _ { d - 1 } \xrightarrow { \mathrm { ( \Omega \to ~ \cdots \to K P _ { 2 } ~ . } }
$$

Here $\mathrm { K P } _ { d }$ denotes shifted-kernel products in dimension $d ,$ and $\mathrm { B P } _ { d - 1 }$ denotes biased exponential products with $( d - 1 )$ -dimensional inner products.

## 1.2.1 Normalization and the Planar Base Case

For one value channel, write the shifted numerator and denominator as

$$
F _ { i } = \sum _ { j } \mathrm { e } ^ { q _ { i } \cdot k _ { j } - M _ { i } } v _ { j } , \qquad D _ { i } = \sum _ { j } \mathrm { e } ^ { q _ { i } \cdot k _ { j } - M _ { i } } , \qquad M _ { i } = \operatorname* { m a x } _ { j } q _ { i } \cdot k _ { j } .
$$

Since $D _ { i } \geq 1$ and $| F _ { i } | \le B D _ { i }$ , approximating both sums to a small constant multiple of $\varepsilon / ( B + 1 )$ ensures error at most ε in their ratio (Lemma 2.5).

For a window width $L > 0$ , every key whose score is below $M _ { i } - L$ has shifted weight at most $\mathrm { e } ^ { - L }$ Discarding any subset of such keys changes $F _ { i }$ by at most n $B \mathrm { e } ^ { - L }$ and $D _ { i }$ by at most $n \mathrm { e } ^ { - L }$ . Thus a logarithmic window sufices (Figure 1), as in [GHSXY26]. We may retain any superset of this window, which allows us to use radial ranges in the plane and common-anchor blocks in higher dimensions.

The score threshold defines a halfspace range, over which the approach of [GHSXY26] evaluates monomial sums. $\mathrm { I f } \ b = \mathrm { p o l y l o g } ( m + n )$ and $B , \varepsilon ^ { - 1 } = ( m + n ) ^ { \mathrm { O } ( 1 ) }$ , combining this approach with ofline range counting and the standard space/query tradeof $\lceil \mathrm { A E 9 9 } \rceil$ gives time $\widetilde { \mathrm { O } } ( ( m n ) ^ { d / ( d + 1 ) } + m + n )$ For $m = n$ , we obtain the ofline counting baseline ${ \tilde { \mathrm { O } } } ( n ^ { \dot { 2 } - 2 / ( d + 1 ) } )$ in Table 1, with exponent $8 / 5$ in dimension four compared with $4 / 3$ for batch maximum inner product.

To go beyond the baseline and achieve the maximum-inner-product exponent determined by $\lfloor d / 2 \rfloor$ we use the geometry of halfspace emptiness and convex polytopes to organize near-maximal pairs. Polynomial moments and common-anchor blocks yield the corresponding weighted sums within this bound, including for near-maximal sets containing almost all keys.

In the plane, we center the key polygon ${ \mathcal C } = \mathrm { c o n v } \{ { k _ { 1 } , . . . , k _ { n } } \}$ at its area centroid c. This translation adds the same scalar − $\mathbf { \partial } ^ { - } \mathbf { q } _ { i } \cdot \mathbf { c }$ to every score in a row and preserves the attention output. We continue to write $\boldsymbol { k } _ { j } , \boldsymbol { \mathcal { C } }$ for the centered keys and polygon.

Centering ensures that every score lies between $- 2 h _ { i }$ and the support value $h _ { i } = \operatorname* { m a x } _ { j } { \pmb q _ { i } \cdot \pmb k _ { j } }$ Thus, when $h _ { i } \leq L$ , all shifted scores lie in [−3L, 0] and we retain every key. For $h _ { i } > L$ , write each nonzero key as $\pmb { k } _ { j } = \rho _ { j } \pmb { k } _ { j } ^ { \partial }$ , where $k _ { j } ^ { \partial }$ lies on the boundary of $\mathcal { C }$ and $0 < \rho _ { j } \leq 1$ . Separate thresholds on $\rho _ { j }$ and $q _ { i } \cdot k _ { j } ^ { \partial }$ retain every key within L of the maximum and only keys within 2L of it. The boundary threshold selects a single arc, so the selected range is specified by a radial threshold and at most two intervals in the boundary order.

Expanding the approximating polynomial in query and key coordinates reduces the product to range sums of weighted key monomials. An ofline sweep with binary indexed trees [Fen94] computes these moments, which we combine with the query coeficients to obtain the kernel product. Reversing the updates and queries computes the transpose product, and storing successive versions of the sweep in persistent segment trees [DSST89] yields the online data structure.

## 1.2.2 Temperature Interpolation and Common-Anchor Blocks

For general d we induct on the dimension. For rows $\pmb q _ { 1 } , \dots , \pmb q _ { m } \in \mathbb { Q } ^ { d }$ and columns $\pmb { k } _ { 1 } , \dots , \pmb { k } _ { n } \in \mathbb { Q } ^ { d }$ ， the shifted kernel is the $m \times n$ matrix T with entries

$$
T _ { i j } = \exp ( \pmb q _ { i } \cdot \pmb k _ { j } - M _ { i } ) , \qquad M _ { i } = \operatorname* { m a x } _ { j \in [ n ] } \pmb q _ { i } \cdot \pmb k _ { j } .
$$

Attention is the ratio of a product Tw for a value channel and the product for the all-ones vector. The shifted-kernel problem $\mathrm { K P } _ { d }$ (Definition 5.3) asks for T w with a given vector $\pmb { w } \in \mathbb { Q } ^ { n }$ or $\pmb { T } ^ { \top } z$ with a given vector $z \in \mathbb { Q } ^ { m }$ , to additive error $\eta ;$ the entries of w and z have absolute value at most W. We prove by induction that a Las Vegas algorithm computes either product in expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ . The suppressed numerical parameters include the weight bound W and error tolerance η.

The induction includes both the forward product Tw and the transpose product $\pmb { T } ^ { \top } z .$ , since the anchor’s orientation determines which lower-dimensional product evaluates a block. The planar algorithm computes both products deterministically in $\widetilde { \mathrm { O } } ( m + n )$ time and serves as the base case (Lemma 5.7).

For the first reduction, group nonzero queries by the position and sign of a largest coordinate. After applying the corresponding signed permutation to queries and keys, write $\pmb q _ { i } = \beta _ { i } ( { \pmb u } _ { i } , 1 )$ and $\pmb { k } _ { j } = ( p _ { j } , c _ { j } )$ , where $\beta _ { i } = \| \pmb q _ { i } \| _ { \infty }$ and ${ \pmb u } _ { i } \in [ - 1 , 1 ] ^ { d - 1 }$ . Then

$$
T _ { i j } = \mathrm { e } ^ { - \beta _ { i } \mu _ { i j } } , \qquad \mu _ { i j } = h _ { i } - \boldsymbol { u } _ { i } \cdot \boldsymbol { p } _ { j } - c _ { j } \geq 0 , \qquad h _ { i } = M _ { i } / \beta _ { i } .
$$

Here the row-dependent factor $\beta _ { i }$ still multiplies the inner product. To decouple them, we partition the rows into dyadic intervals of $\beta _ { i }$ and use Lagrange interpolation on each interval to approximate $\mathrm { e } ^ { - \beta \mu }$ by a linear combination of exponentials $\mathrm { e } ^ { - \tau \mu }$ at fixed nodes τ. By Lemma 2.6, the error decreases exponentially with the number of nodes, uniformly over $\mu \geq 0$ , and the sum of the absolute interpolation coeficients grows at most exponentially in that number. Consequently, $\widetilde { \mathrm { O } } ( 1 )$ nodes and precision bits sufice. At a fixed node $\tau _ { \ast }$ we rescale $( p _ { j } , c _ { j } )$ by $\tau$ to write the exponent as

$$
x _ { i j } = { \pmb u } _ { i } \cdot { \pmb p } _ { j } + b _ { i } + c _ { j } \le 0 ,
$$

with $d - 1$ bilinear coordinates and a bias on each side, defining the intermediate product problem $\mathrm { B P } _ { d - 1 }$ for the matrix $\left( \mathrm { e } ^ { x _ { i j } } \right)$ . In Proposition 5.5, we combine its forward outputs using the interpolation coeficients, or exchange rows and columns and absorb those coeficients into the input weights for a transpose product.

Directly factoring the entire biased matrix into a row-shifted kernel and diagonal matrices would give diagonal factors $\mathrm { e } ^ { b _ { i } + M _ { i } ^ { \prime } }$ and $\mathrm { e } ^ { c _ { j } }$ , where $M _ { i } ^ { \prime } = \operatorname* { m a x } _ { j } { \pmb u _ { i } \cdot \pmb p _ { j } }$ . However, these factors can be exponentially large in the input magnitudes. Instead, we obtain polynomially bounded factors by partitioning the significant pairs into blocks with a common anchor, a row or column whose scores across the block lie in a window of width $L = \Theta ( \log ( N W / \eta ) )$ below zero, with constant slack, where $N = m + n$ . If column $j _ { 0 }$ anchors $I \times J$ , subtracting its score cancels the row bias, so that

$$
x _ { i j } - x _ { i j _ { 0 } } = { \pmb u } _ { i } \cdot ( { \pmb p } _ { j } - { \pmb p } _ { j _ { 0 } } ) + c _ { j } - c _ { j _ { 0 } } .
$$

After shifting these inner products by their maxima, the block factors as

$$
\begin{array} { r } { S _ { I J } = ( \mathrm { e } ^ { x _ { i j } } ) _ { i \in I , j \in J } = \mathrm { d i a g } ( d _ { \mathrm { r o w } } ) \pmb { T } ^ { \top } \mathrm { d i a g } ( d _ { \mathrm { c o l } } ) , } \end{array}
$$

where $_ { \pmb { T } }$ is a shifted kernel in dimension $d - 1$ , with rows $p _ { j } - p _ { j _ { 0 } }$ and columns $\mathbf { \Delta } \mathbf { u } _ { i }$ . For a row anchor, the analogous factorization uses $_ { \pmb { T } }$ in place of $\pmb { T } ^ { \top }$ . In both cases, the anchor condition bounds the diagonal factors by $ { \mathrm { e } } ^ { \mathrm { O } ( L ) }$ , giving expected evaluation time $\widetilde { \mathrm { O } } ( F _ { d - 1 } ( | I | , | J | ) )$

We construct disjoint anchor blocks with total expected evaluation time $\bar { \mathrm { O } } ( F _ { d } ( m , n ) )$ by sampling one side and grouping the other side according to its first near-maximal sample.

## 1.2.3 Random Prefix Chains and Heavy/Light Recursion

To construct the blocks, we use one side as sources and the other as targets. When columns serve as sources, the points $( p _ { s } , c _ { s } ) \in \mathbb { Q } ^ { d }$ define afine functions ${ \pmb u } \mapsto { \pmb u } \cdot { \pmb p } _ { s } + c _ { s }$ evaluated at the row targets. A small rational perturbation shared by all samples puts these functions in general position and changes each score by at most a constant slack in the window. We then use the resulting geometry to define the blocks while retaining the original scores to evaluate them.

For a sample of h sources, we triangulate their maximizing regions into $\mathrm { O } _ { d } ( h ^ { k } )$ canonical cells, where $k = \lfloor d / 2 \rfloor$ , so that each cell $\Delta$ has a sampled owner $s _ { \Delta }$ whose function equals the envelope on all of $\Delta$ . After $\widetilde { \mathrm { O } } ( h ^ { k } )$ preprocessing, ray shooting locates each target in $\widetilde { \mathrm { O } } ( 1 )$ time. The cell’s conflict set $C _ { \Delta }$ consists of the sources that exceed its owner somewhere on the cell, a condition we test at its vertices using a constant number of strict halfspace-reporting queries.

Start with logarithmically many sources and expose doubling prefixes $P _ { 0 } , P _ { 1 } , . . .$ . of a random permutation, with batches $B _ { \ell } = P _ { \ell + 1 } \setminus P _ { \ell }$ . For a source set of size $n ,$ the sampling estimates imply

$$
| C _ { \Delta } | = \widetilde { \mathrm { O } } ( n / | P _ { \ell } | ) , \qquad | C _ { \Delta } \cap B _ { \ell } | = \widetilde { \mathrm { O } } ( 1 )
$$

simultaneously for all prefix cells with constant probability, by a union bound over the possible cells fixed by the shared perturbation. We test both inequalities and resample upon failure.

A source is a hit for a target if its perturbed score is at least $- L$ . We assign each target to the first of $P _ { 0 } , B _ { 0 } , B _ { 1 } , . . .$ . containing a hit and use a maximizing source in that set as its anchor. For a first hit in $B _ { \ell }$ , the preceding prefix’s envelope lies below the hit threshold at the target, so every hit exceeds the owner of the target’s prefix cell and belongs to its conflict set $C _ { \Delta }$

![](images/eab192d854cca0ffe23e517b6d746d411e232a76d107485d424fa7efeb2ac4df.jpg)  
Figure 2: A schematic level of the random doubling prefix for $d = 3$ , in the column orientation. (a) A square window of the maximizing regions of the perturbed functions in $P _ { \ell } ,$ with each polygon triangulated for the illustration. Thick lines bound the maximizing regions, and thin lines triangulate them. (b) The shaded cell $\Delta$ , with owner $s _ { \Delta }$ , enlarged. Dashed lines separate the regions where diferent sources of $C _ { \Delta } \cap B _ { \ell }$ have the largest perturbed score. The markers represent targets in $\Delta$ whose first hit occurs after $P _ { \ell } .$ . Filled markers of the same shape share an anchor $s _ { * } \in C _ { \Delta } \cap B _ { \ell }$ and form one anchor block with source set $C _ { \Delta } ;$ hollow markers have their first hit in a later batch.

Targets with the same cell and anchor form a block with source set $C _ { \Delta }$ (Figure 2), and the bound on $C _ { \Delta } \cap B _ { \ell }$ limits each cell to $\widetilde { \mathrm { O } } ( 1 )$ such blocks. Targets first hit in $P _ { 0 }$ form blocks with the full source set, completing a collection with disjoint target sets that covers every significant pair when the chain reaches the full source set.

For n sources and m targets, the cell counts and conflict sizes over the complete doubling chain sum to $\widetilde { \mathrm { O } } ( n ^ { k } + m )$ , which is also the time for constructing the blocks. By concavity, the logarithmic number of anchors per cell and the disjoint target sets imply the same bound for anchor evaluation, so the complete algorithm is near-linear when $k = 1$ or $m \geq n ^ { k }$

When $d \geq 4$ and the two sides are balanced, however, a full prefix is too costly. For $d = 4$ and two sides of size $n ,$ we have $k = 2$ and instead truncate the prefix at $\widetilde { \Theta } ( \kappa )$ sources. Suppressing logarithmic factors, the final prefix has $\mathrm { O } ( \kappa ^ { 2 } )$ cells with $\mathrm { O } ( n / \kappa )$ conflicts each, so reporting costs O(nκ) operations. In a light cell, the $\mathrm { O } ( ( n / \kappa ) ^ { 1 / 2 } )$ unassigned targets become the sources of a full prefix, with cost $\mathrm { O } ( n / \kappa )$ per cell and $\mathrm { O } ( n \kappa )$ time across all light cells. We recurse on the heavy cells, whose total potential is of order at most $n ( n / \kappa ) ^ { 1 / 2 }$ . The two bounds balance at $\kappa \asymp n ^ { 1 / 3 }$ , for a total expected running time of $\widetilde { \mathrm { O } } ( n ^ { 4 / 3 } )$

In general, at a recursive node v with side sizes $m _ { v } \le n _ { v } ,$ a full prefix with the smaller side as sources runs in near-linear time when $m _ { v } ^ { k } \le n _ { v }$ . Otherwise, at a balanced node, we use the larger side as sources and truncate at $\widetilde { \Theta } ( \kappa )$ sources. Cells with $\mathrm { O } ( ( n _ { v } / \kappa ) ^ { 1 / k } )$ unassigned targets are processed by a full prefix in the reverse orientation; heavy cells are handled recursively. The reporting and light-cell cost and the total heavy-child potential have respective bounds

$$
n _ { v } \kappa ^ { k - 1 } \qquad \mathrm { a n d } \qquad m _ { v } ( n _ { v } / \kappa ) ^ { ( k - 1 ) / k } .
$$

Balancing them gives local expected time $\widetilde { \mathrm { O } } ( \varPhi ( m _ { v } , n _ { v } ) )$ , where $\begin{array} { r } { \phi ( m _ { v } , n _ { v } ) = ( m _ { v } n _ { v } ) ^ { k / ( k + 1 ) } } \end{array}$ . Section 7.5 specifies the integer prefix size and light-cell threshold, for which the heavy children’s potentials sum to at most one quarter of their parent’s. The total expected running time is therefore $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ 1

Since the evaluated blocks are disjoint, each output coordinate receives at most N contributions. A block tolerance of order $\eta / N$ and a score window of order $\log ( N W / \eta )$ keep the combined approximation error and omitted tail within η. Appendix A.4 shows that these parameters require polynomially many precision bits at every recursive level.

## 1.3 Related Work

Complexity of attention. Alman and Song [AS23] established a sharp magnitude threshold for logarithmic dimension. Entries of magnitude $\mathbf { o } ( { \sqrt { \log n } } )$ permit $n ^ { 1 + \mathbf { o } ( 1 ) }$ time at inverse-polynomial error, and magnitude of order log n supports a conditional quadratic lower bound at the same error tolerance.<sup>2</sup>

Duman Keles, Wijewardena and Hegde [DWH23] proved quadratic lower bounds under the Orthogonal Vectors Hypothesis (OVH) [Wil05] for exact and approximate attention in every dimension d with $d / \log n  \infty$ , and Alman and Song [AS24] extended the magnitude threshold of [AS23] to gradient computation. Further lower bounds concern subquadratic alternatives to attention [AY25] and multi-head, multi-layer transformers [SXYY26]. For exact integer maximum inner product, the dimension in which OVH implies quadratic hardness was lowered to a polynomial in log log n by Williams [Wil18], to $2 ^ { \mathrm { O } ( \log ^ { \overline { { * } } } n ) }$ by Chen [Che20], and to every constructible superconstant dimension by Saha, Xu and Ye [SXY26].

Kernel sums and approximate attention. The fast multipole method and the fast Gauss transform [GR87; GS91] compute kernel sums in fixed dimension in near-linear time, with error relative to the total weight. Alman, Chu, Schild and Song [ACSS20] determine fine-grained limits of such methods as the dimension grows. The identity $\mathrm { e } ^ { q _ { i } \cdot k _ { j } } = \mathrm { e } ^ { \| q _ { i } \| ^ { 2 } / 2 } \mathrm { e } ^ { \| k _ { j } \| ^ { 2 } / 2 } \mathrm { e } ^ $ −∥q<sub>i</sub> $- k _ { j } \| ^ { 2 } / 2$ expresses attention through Gaussian kernel sums with weights $\mathrm { e } ^ { \| \boldsymbol { k } _ { j } \| ^ { 2 } / 2 }$ , whose ratios can be as large as $\mathrm { e } ^ { d B ^ { 2 } / 2 }$ An error bound relative to the total weight can then require relative precision $\varepsilon _ { \mathrm { r e l } } = \mathrm { e } ^ { - \Omega ( d B ^ { 2 } ) }$ leading to polynomial dependence on B in fixed dimension even with a $\mathrm { p o l y l o g } ( 1 / \varepsilon _ { \mathrm { r e l } } )$ dependence on precision.

KDEformer [ZHDK23] and HyperAttention [HJK+24] achieve subquadratic time under bounds on entry magnitudes or other data-dependent parameters. Using halfspace reporting, Chen, Liang, Sha, Shi, and Song [CLSSS25] locate large attention entries with error guarantees under distributional assumptions. In contrast, our algorithms achieve worst-case guarantees for arbitrary rational inputs, with factors polynomial in log B, log(1/ε), and the coordinate encoding lengths.

Maximum inner product and halfspace range searching. Following Matousek’s introduction of shallow cuttings and partitions for halfspace reporting and emptiness [Mat92a], Matousek and Schwarzkopf [MS93] constructed static ray-shooting structures with $\widetilde { \mathrm { O } } ( \bar { n } ^ { \lfloor d / 2 \rfloor } )$ preprocessing for n halfspaces, logarithmic query time, and a space/query tradeof. Matousek’s linear-optimization queries [Mat93] then led to an algorithm for the extreme points of an n-point set with time exponent $2 - 2 / ( \lfloor d / 2 \rfloor + 1 )$ plus an arbitrarily small positive constant. Chan [Cha96a; Cha96b] simplified these query methods, derived output-sensitive applications, and observed that static data structures replace the small polynomial loss by polylogarithmic factors. Deterministic structures for ray shooting, range reporting, and linear-programming queries were subsequently given by Ramos [Ram99; Ram00]. For m ray-shooting queries in a polytope defined by n halfspaces in dimension $d > 3$ , Chan [Cha96b, Corollary 2.4] stated the batch bound

$$
\widetilde { \mathrm { O } } ( ( m n ) ^ { \lfloor d / 2 \rfloor / ( \lfloor d / 2 \rfloor + 1 ) } + m + n ) ;
$$

Corollary 2.2 there treats $d \leq 3$ . The polylogarithmic factors were later improved for the extreme points of a point set in even dimension [Cha12] and for the decision version [CZ22]; these remain the best known exponents. We refer to the surveys [Mat94; AE99; Aga17].

Lower bounds. Erickson [Eri96; Eri00] proved an $\Omega ( n ^ { 4 / 3 } )$ lower bound for ofline halfspace emptiness in every dimension at least five, for polyhedral partitioning algorithms. Such algorithms cover space by a constant number of constant-complexity polyhedra and recurse on the points and halfspaces that meet each of them. This lower bound matches the best known upper bound for $d = 5$ up to polylogarithmic factors, whereas establishing comparable bounds for general algorithms or in dimension four remains open.

Convex hulls, triangulations, and random sampling. Our geometric decomposition relies on the upper bound theorem [McM70], the randomized incremental construction of convex hulls [CS89], with a deterministic alternative due to Chazelle [Cha93] (see [Sei17]), the pulling triangulation [Cla88] and canonical triangulations [CF90], and the random-sampling analysis of Clarkson and Shor [Cla87; CS89]. Our random prefixes follow randomized incremental constructions, with exact tests on conflict sizes and batch intersections ensuring deterministic participation bounds for every accepted decomposition. To remove degeneracies within the bit-complexity model, we apply an explicit rational moment-curve perturbation shared by all samples. For background on perturbation methods and exact computation, we refer to the symbolic schemes of [EM90; EC95] and the geometric and algebraic methods in [Yap97; BPR06].

Independent and concurrent work. Very recently, during the final preparation of this manuscript, we became aware of independent and concurrent work by Mausberg [Mau26] on the same problem. They improve upon the previously best-known result of Gupta, Huang, Saha, Xu and Ye [GHSXY26], obtaining an almost-linear<sup>3</sup> Las Vegas algorithm for Attention<sub>2</sub> and Attention<sub>3</sub>. In comparison, our work gives a deterministic near-linear algorithm for Attention , a near-linear Las Vegas algorithm for Attention , and a Las Vegas algorithm for higher dimensions with expected running time $\widetilde { \mathrm { O } } ( ( m n ) ^ { k / ( k + 1 ) } + m + n )$ , where $k = \lfloor d / 2 \rfloor$ . The techniques we use are also very diferent from those of Mausberg.

## 2 Preliminaries

For a positive integer n, write $[ n ] = \{ 1 , \dots , n \}$ . We use ln for the natural logarithm and log for the binary logarithm. For a matrix A, let $\left\| A \right\| _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i , j } \left| A _ { i j } \right|$ , and let ⊙ denote entrywise multiplication. Throughout, $\widetilde { \mathrm { ~ O ~ } }$ suppresses factors polynomial in the input encoding lengths, the logarithms of the instance size and numerical magnitudes, and the required precision in bits; Section 2.1 collects these parameters in λ. The notation $\widetilde \Omega$ suppresses polylogarithmic factors in the instance size. A subscript d permits dependence on the fixed dimension and number of value channels.

Definition 2.1. Given $Q \in \mathbb { Q } ^ { m \times d } , \ K \in \mathbb { Q } ^ { n \times d }$ , and $V \in \mathbb { Q } ^ { n \times r }$ with rows $q _ { i } , \ k _ { j }$ , and $v _ { j } =$ $( v _ { j 1 } , \ldots , v _ { j r } )$ , define

$$
\operatorname { A t t } ( Q , K , V ) _ { i t } = { \frac { \sum _ { j = 1 } ^ { n } \mathrm { e } ^ { q _ { i } \cdot k _ { j } } v _ { j t } } { \sum _ { j = 1 } ^ { n } \mathrm { e } ^ { q _ { i } \cdot k _ { j } } } } , \qquad i \in [ m ] , \quad t \in [ r ] .
$$

All input coordinates are rationals of magnitude at most $B \geq 1$ , with signed numerators and positive denominators of bit length at most $b ,$ the coordinate encoding length. Given a target additive error $\varepsilon = 2 ^ { - p }$ for an integer $p \geq 0$ , the problem $\mathrm { A t t e n t i o n } _ { d }$ asks for a rational matrix $\hat { O }$ satisfying

$$
\| \widehat { O } - \mathrm { A t t } ( Q , K , V ) \| _ { \mathrm { m a x } } \leq \varepsilon .
$$

Our geometric algorithms allow any fixed number r of value channels; the one-call reductions use $r = d .$

Every entry of the ideal output $\operatorname { A t t } ( Q , K , V ) \in \mathbb { R } ^ { m \times r }$ is a convex combination of the corresponding value coordinates and has absolute value at most B.

Extend the values by the all-ones channel $v _ { j 0 } = 1$ , and define the shifted kernel by

$$
M _ { i } = \operatorname* { m a x } _ { j \in [ n ] } q _ { i } \cdot k _ { j } , \qquad T _ { i j } = \mathrm { e } ^ { q _ { i } \cdot k _ { j } - M _ { i } } , \qquad T = ( T _ { i j } ) .
$$

For $0 \leq t \leq r$ , write ${ \pmb v } ^ { ( t ) } = ( v _ { j t } ) _ { j \in [ n ] }$ and

$$
D _ { i } = \sum _ { j = 1 } ^ { n } T _ { i j } , \qquad F _ { i t } = ( \mathbf { \mathbf { \mathbf { \mathit { T } } } } \mathbf { \mathbf { \mathbf { \mathit { v } } } } ^ { ( t ) } ) _ { i } , \qquad F _ { i 0 } = D _ { i } .
$$

Then $\mathrm { A t t } ( Q , K , V ) _ { i t } = F _ { i t } / D _ { i }$ for $t \in [ r ]$ . Each row of T contains a unit entry, so $1 \leq D _ { i } \leq n$ and $| F _ { i t } | \le B D _ { i } ;$ we denote their approximations by $\widehat { D } _ { i }$ and $\widehat { F } _ { i t }$ . Thus $r + 1$ kernel products sufice for all value channels, with the $t = 0$ product equal to the denominator shared by the r ratios.

A zero query has maximum zero and an all-ones shifted-kernel row, so its attention output is the average of the values. In a forward kernel product, such a row returns the sum of the input weights; in a transpose product, the weights on all zero rows contribute their sum to every output coordinate. We process these rows together in $\widetilde { \mathrm { O } } ( m + n )$ time per channel before applying the algorithms below. Coordinate normalization and support queries may therefore assume nonzero query vectors.

We compare the complexity of attention with that of the batch maximum-inner-product problem, defined as follows.

Definition 2.2. For integer matrices $\pmb { X } \in \mathbb { Z } ^ { m \times d }$ and $Y \in \mathbb { Z } ^ { n \times d }$ with rows $\mathbf { \mathcal { x } } _ { i }$ and ${ \bf { \nabla } } _ { \bf { { y } } } \mathrm { { _ { j } } }$ and coordinate magnitudes at most an integer $B \geq 1$ , the batch problem BatchMaxI $\mathrm { P } _ { d }$ asks for

$$
M _ { i } = \operatorname* { m a x } _ { j \in [ n ] } { \pmb x } _ { i } \cdot { \pmb y } _ { j } , \qquad i \in [ m ] .
$$

The global problem $\mathrm { M a x I P } _ { d }$ asks for $\operatorname* { m a x } _ { i } M _ { i }$ . Each $M _ { i }$ is an integer in $[ - d B ^ { 2 } , d B ^ { 2 } ]$

## 2.1 Computation and Error Bounds

We measure complexity in bit operations, so each batch or preprocessing time bound also applies to space. Our randomized algorithms are Las Vegas, and their running times are measured in expectation.

Throughout the paper, $\lambda \geq 1$ bounds an instance’s rational input encoding lengths, the logarithms of its size and numerical magnitudes, and the required precision in bits. Geometric predicates are evaluated exactly. Intermediate geometric quantities, scalar exponentials, polynomial coeficients, and weighted sums use $\mathrm { p o l y } _ { d } ( \lambda )$ -bit representations or approximations, as established in $\mathrm { A p p e n d i x ~ A }$ Approximate outputs lie on a common dyadic grid whose mesh is proportional to the target tolerance, uniformly in d and r. For a data structure, the magnitude and encoding-length bounds on future queries are fixed before preprocessing; a query or target satisfying these bounds is admissible.

Subproblems formed by restricting index sets inherit the numerical bounds of their enclosing instance. For variable $^ { r , }$ soft-O also suppresses factors polynomial in $\log ( r + 2 )$ , with constants and degrees depending only on $d .$

We write the running-time bounds using

$$
k ( d ) = \lfloor d / 2 \rfloor , \qquad F _ { d } ( m , n ) = ( m n ) ^ { k ( d ) / ( k ( d ) + 1 ) } + m + n \qquad ( m , n \geq 1 ) .\tag{2.1}
$$

When the dimension is clear, we abbreviate $k ( d )$ to $k ;$ the function $F _ { d }$ is symmetric and nondecreasing in both arguments.

Remark 2.3 (Arbitrary positive temperature). A positive rational inverse temperature $\beta$ is absorbed into the queries by replacing $Q$ with $_ { \beta Q }$ . This adds only the encoding length of $\beta$ to the coordinate encoding lengths and replaces the magnitude bound by B max $\{ 1 , \beta \}$ . The resulting dependence on temperature is polylogarithmic in its magnitude.

An irrational inverse temperature can be replaced by a positive dyadic approximation at the precision derived in Appendix A.1, including for the usual scaling $\beta = d ^ { - 1 / 2 }$

## 2.2 Random Sampling

The algorithms in Sections 6 and 7 sample uniform subsets and permutations, using standard probability estimates for sampling without replacement.

Lemma 2.4. Let Σ be a nonempty finite set, let $\Sigma ^ { \prime \prime } \subseteq \Sigma$ , and let $\Sigma ^ { \prime }$ be a uniformly random subset of Σ of a given size $| \Sigma ^ { \prime } |$ . Then $\mathrm { P r } [ \Sigma ^ { \prime } \cap \Sigma ^ { \prime \prime } = \emptyset ] \leq \mathrm { e } ^ { - | \Sigma ^ { \prime } | | \Sigma ^ { \prime \prime } | / | \Sigma | }$ , and $X _ { \Sigma ^ { \prime \prime } } = | \Sigma ^ { \prime } \cap \Sigma ^ { \prime \prime } |$ satisfies $\operatorname* { P r } [ X _ { \Sigma ^ { \prime \prime } } \geq \theta ] \leq ( \mathrm { e } \mathbb { E } X _ { \Sigma ^ { \prime \prime } } / \theta ) ^ { \theta }$ for every positive integer θ.

Proof. Apply Hoefding’s comparison theorem for sampling without replacement [Hoe63, Theorem 4] to the indicators of membership in $\Sigma ^ { \prime \prime }$ . Writing $s = | \Sigma ^ { \prime } | , p = | \Sigma ^ { \prime \prime } | / | \Sigma |$ , and $\mu = s p = \mathbb { E } X _ { \Sigma ^ { \prime \prime } }$ gives

$$
\mathbb { E } \mathrm { e } ^ { t X _ { \Sigma ^ { \prime \prime } } } \leq ( 1 - p + p \mathrm { e } ^ { t } ) ^ { s } \leq \mathrm { e } ^ { \mu ( \mathrm { e } ^ { t } - 1 ) } \qquad ( t \in \mathbb { R } ) .
$$

The first bound follows as $t \to - \infty$ . For $\theta > \mu > 0$ , the second follows from the exponential Markov inequality with $t = \ln ( \theta / \mu )$ , and is immediate when $\theta \leq \mu { \mathrm { ~ o r ~ } } \mu = 0$ □

## 2.3 Analytic Estimates

Lemma 2.5 (Perturbation and ratio bounds). For x, $\pmb { x } ^ { \prime } \in \mathbb { R } ^ { n }$ with $\| { \pmb x } - { \pmb x } ^ { \prime } \| _ { \infty } \le \delta$ , let $h = \operatorname* { m a x } _ { j } x _ { j }$ and $h ^ { \prime } = \operatorname* { m a x } _ { j } x _ { j } ^ { \prime }$ . Then

$$
\begin{array} { r } { \| \operatorname { s o f t m a x } ( { \pmb x } ) - \operatorname { s o f t m a x } ( \pmb x ^ { \prime } ) \| _ { 1 } \leq 2 \delta , \qquad | h - h ^ { \prime } | \leq \delta , } \end{array}
$$

$$
| ( x _ { j } - h ) - ( x _ { j } ^ { \prime } - h ^ { \prime } ) | \leq 2 \delta , \qquad | \mathrm { e } ^ { x _ { j } - h } - \mathrm { e } ^ { x _ { j } ^ { \prime } - h ^ { \prime } } | \leq 2 \delta .
$$

If $D \geq 1 , | F | \leq W D$ , and $| \widehat { D } - D | , | \widehat { F } - F | \leq \zeta < 1$ , then

$$
\left| \frac { \widehat { F } } { \widehat { D } } - \frac { F } { D } \right| \leq \frac { ( W + 1 ) \zeta } { 1 - \zeta } .
$$

Proof. The softmax Jacobian has operator norm at most 2 from $\ell _ { \infty }$ to $\ell _ { 1 } ;$ the maximum and the exponential on $( - \infty , 0 ]$ are 1-Lipschitz. The ratio bound follows by writing its numerator as $D ( \widehat { F } - F ) - F ( \widehat { D } - D )$ and using $\widehat { D } \geq 1 - \zeta$ □

Temperature interpolation uses a single set of nodes to approximate $\mathrm { e } ^ { - \beta \mu }$ uniformly over $\mu \geq 0$ Placing the nodes in the middle half of each dyadic interval keeps their distance from any point in the interval within three quarters of its lower endpoint, yielding exponential decay of the interpolation error.

Lemma 2.6. Let $n _ { \mathrm { i n t } } \geq 2$ be an integer, fix a lower endpoint $\beta _ { \mathrm { l o } } > 0$ , and let

$$
\tau _ { \ell } = \beta _ { \mathrm { l o } } \Big ( \frac { 5 } { 4 } + \frac { \ell } { 2 \big ( n _ { \mathrm { i n t } } - 1 \big ) } \Big ) \quad ( 0 \le \ell < n _ { \mathrm { i n t } } ) , \qquad \mathscr { L } _ { \ell } ( \beta ) = \prod _ { 0 \le \ell ^ { \prime } < n _ { \mathrm { i n t } } } \frac { \beta - \tau _ { \ell ^ { \prime } } } { \tau _ { \ell } - \tau _ { \ell ^ { \prime } } } .
$$

The $n _ { \mathrm { i n t } }$ nodes define a Lagrange interpolation polynomial of degree at most $n _ { \mathrm { i n t } } - 1$ . For all $\mu \geq 0$ and $\beta \in [ \beta _ { \mathrm { l o } } , 2 \beta _ { \mathrm { l o } } ]$ 2

$$
\left| \mathrm { e } ^ { - \beta \mu } - \sum _ { \ell = 0 } ^ { n _ { \mathrm { i n t } } - 1 } \mathcal { L } _ { \ell } ( \beta ) \mathrm { e } ^ { - \tau _ { \ell } \mu } \right| \leq \left( \frac { 3 } { 4 } \right) ^ { n _ { \mathrm { i n t } } } \qquad a n d \qquad \sum _ { \ell = 0 } ^ { n _ { \mathrm { i n t } } - 1 } | \mathcal { L } _ { \ell } ( \beta ) | \leq 9 ^ { n _ { \mathrm { i n t } } - 1 } .
$$

Proof. For fixed $\mu \geq 0$ , consider the function $\beta \mapsto \mathrm { e } ^ { - \beta \mu }$ , whose $n _ { \mathrm { i n t } } – \mathrm { t h }$ derivative is $( - \mu ) ^ { n _ { \mathrm { i n t } } } \mathrm { e } ^ { - \beta \mu }$ The nodes are distinct and lie in $[ 5 \beta _ { \mathrm { l o } } / 4 , 7 \beta _ { \mathrm { l o } } / 4 ] , \mathrm { s o } \ | \beta - \tau _ { \ell } | \leq 3 \beta _ { \mathrm { l o } } / 4$ for every $\beta \in [ \beta _ { \mathrm { l o } } , 2 \beta _ { \mathrm { l o } } ]$ and every ℓ. By the Lagrange remainder formula [DLMF, Eq. (3.3.5)], for some $\beta ^ { \prime } \in [ \beta _ { \mathrm { l o } } , 2 \beta _ { \mathrm { l o } } ]$

$$
\mathrm { e } ^ { - \beta \mu } - \sum _ { \ell = 0 } ^ { n _ { \mathrm { i n t } } - 1 } \mathcal { L } _ { \ell } ( \beta ) \mathrm { e } ^ { - \tau _ { \ell } \mu } = \frac { ( - \mu ) ^ { n _ { \mathrm { i n t } } } \mathrm { e } ^ { - \beta ^ { \prime } \mu } } { n _ { \mathrm { i n t } } ! } \prod _ { \ell = 0 } ^ { n _ { \mathrm { i n t } } - 1 } ( \beta - \tau _ { \ell } ) .
$$

The function $x \mapsto x ^ { n _ { \mathrm { i n t } } } \mathrm { e } ^ { - \beta _ { \mathrm { l o } } x }$ on [0, ∞) is maximized at $x = n _ { \mathrm { i n t } } / \beta _ { \mathrm { l o } }$ , so

$$
\mu ^ { n _ { \mathrm { i n t } } } \mathrm { e } ^ { - \beta ^ { \prime } \mu } \leq \mu ^ { n _ { \mathrm { i n t } } } \mathrm { e } ^ { - \beta _ { \mathrm { l o } } \mu } \leq \left( \frac { n _ { \mathrm { i n t } } } { \mathrm { e } \beta _ { \mathrm { l o } } } \right) ^ { n _ { \mathrm { i n t } } } .
$$

Together with $n _ { \mathrm { i n t } } ! \geq ( n _ { \mathrm { i n t } } / \mathrm { e } ) ^ { n _ { \mathrm { i n t } } }$ and $| \beta - \tau _ { \ell } | \leq 3 \beta _ { \mathrm { l o } } / 4$ , the remainder has absolute value at most

$$
\Big ( \frac { n _ { \mathrm { i n t } } } { \mathrm { e } \beta _ { \mathrm { l o } } } \Big ) ^ { n _ { \mathrm { i n t } } } \Big ( \frac { 3 \beta _ { \mathrm { l o } } } { 4 } \Big ) ^ { n _ { \mathrm { i n t } } } \Big ( \frac { \mathrm { e } } { n _ { \mathrm { i n t } } } \Big ) ^ { n _ { \mathrm { i n t } } } = \Big ( \frac { 3 } { 4 } \Big ) ^ { n _ { \mathrm { i n t } } } .
$$

For the coeficients, consecutive nodes are $\beta _ { \mathrm { l o } } / ( 2 ( n _ { \mathrm { i n t } } - 1 ) )$ apart, so

$$
\prod _ { \ell ^ { \prime } \neq \ell } \vert \tau _ { \ell } - \tau _ { \ell ^ { \prime } } \vert = \Big ( \frac { \beta _ { \mathrm { l o } } } { 2 ( n _ { \mathrm { i n t } } - 1 ) } \Big ) ^ { n _ { \mathrm { i n t } } - 1 } \ell ! ( n _ { \mathrm { i n t } } - 1 - \ell ) ! .
$$

Each of the $n _ { \mathrm { i n t } } - 1$ factors $| \beta - \tau _ { \ell ^ { \prime } } |$ in the numerator of $\mathcal { L } _ { \ell } ( \beta )$ is at most $3 \beta _ { \mathrm { l o } } / 4$ , hence

$$
| \mathcal { L } _ { \ell } ( \beta ) | \le \frac { \big ( 3 \big ( n _ { \mathrm { i n t } } - 1 \big ) / 2 \big ) ^ { n _ { \mathrm { i n t } } - 1 } } { \ell ! \big ( n _ { \mathrm { i n t } } - 1 - \ell \big ) ! } .
$$

Summing over ℓ and using $\begin{array} { r } { \sum _ { \ell = 0 } ^ { n _ { \mathrm { i n t } } - 1 } \binom { n _ { \mathrm { i n t } } - 1 } { \ell } = 2 ^ { n _ { \mathrm { i n t } } - 1 } } \end{array}$ and $( n _ { \mathrm { i n t } } - 1 ) ! \geq ( ( n _ { \mathrm { i n t } } - 1 ) / \mathrm { e } ) ^ { n _ { \mathrm { i n t } } - 1 }$ , we obtain

$$
\sum _ { \ell = 0 } ^ { n _ { \mathrm { i n t } } - 1 } \left| \mathcal { L } _ { \ell } ( \beta ) \right| \leq \frac { \left( 3 ( n _ { \mathrm { i n t } } - 1 ) / 2 \right) ^ { n _ { \mathrm { i n t } } - 1 } 2 ^ { n _ { \mathrm { i n t } } - 1 } } { ( n _ { \mathrm { i n t } } - 1 ) ! } = \frac { \left( 3 ( n _ { \mathrm { i n t } } - 1 ) \right) ^ { n _ { \mathrm { i n t } } - 1 } } { ( n _ { \mathrm { i n t } } - 1 ) ! } \leq ( 3 \mathrm { e } ) ^ { n _ { \mathrm { i n t } } - 1 } \leq 9 ^ { n _ { \mathrm { i n t } } - 1 } .
$$

## 3 A Dimension-Preserving Reduction from Maximum Inner Product

We now show that $\mathrm { B a t c h M a x I P } _ { d }$ reduces to a single call to $\mathrm { A t t e n t i o n } _ { d }$ in the same dimension. The idea is to set both the keys and the values equal to Y at a logarithmic inverse temperature $\beta _ { 0 } \mathrm { { : } }$ taking the inner product of the resulting attention output with $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { i } }$ yields a softmax average of the scores $x _ { i } \cdot y _ { j }$ , which lies within $( \ln n ) / \beta _ { 0 }$ of the maximum $M _ { i }$ and can be rounded to recover $M _ { i }$ exactly.

## 3.1 Computing Exact Maxima from Attention

Consider an instance of $\mathrm { B a t c h M a x I P } _ { d }$ with matrices $\pmb { X } \in \mathbb { Z } ^ { m \times d }$ and $Y \in \mathbb { Z } ^ { n \times d }$ and coordinate magnitudes at most an integer $B \geq 1$ . To ensure that the softmax average and the attention approximation error each contribute less than $1 / 4$ to the inner product with $\mathbf { \delta } _ { \mathbf { x } _ { i } }$ , we set $C = 8$ $\beta _ { 0 } = C \lceil \log _ { 2 } ( n + 1 ) \rceil$ , and $\varepsilon _ { 0 } = 2 ^ { - \lceil \log _ { 2 } ( C d B ) \rceil }$ , so $\varepsilon _ { 0 } = \Theta ( 1 / ( d B ) )$ and $\varepsilon _ { 0 } \leq 1 / ( C d B )$ . The dimension may grow in this section, and soft-O suppresses only logarithmic factors in $m + n , d , B$

Theorem 3.1 (Dimension-preserving reduction). Set $\pmb { Q } = \beta _ { 0 } \pmb { X }$ and $K = Y$ . Each of the attention reductions below solves $\mathrm { B a t c h M a x I P } _ { d }$ with the same $m , n , d$ and deterministic additional time $\widetilde { \mathrm { O } } ( ( m + n ) d )$

(a) One call with $V = Y , r = d$ value channels, magnitude bound $\beta _ { 0 } B$ , and error tolerance $\varepsilon _ { 0 }$

(b) One call with $V = d B Y , \ r = d$ value channels, magnitude bound max $\{ \beta _ { 0 } B , d B ^ { 2 } \}$ , and constant error tolerance $\varepsilon _ { c } = 1 / 4$

(c) For either value matrix V, d calls with one value channel each, using its columns separately and the same error tolerance.

Proof. Let $M _ { i } = \operatorname* { m a x } _ { j } { \pmb x _ { i } } \cdot { \pmb y _ { j } }$ , let $\mathbf { o } _ { i }$ be the exact output in part (a), and write $\begin{array} { r } { a _ { i j } = \mathrm { e } ^ { \beta _ { 0 } \pmb { x } _ { i } \cdot \pmb { y } _ { j } } / Z _ { i } , } \end{array}$ where $\begin{array} { r } { Z _ { i } = \sum _ { j } \mathrm { e } ^ { \beta _ { 0 } \pmb { x } _ { i } \cdot \pmb { y } _ { j } } } \end{array}$ . The entropy identity ln $\begin{array} { r } { Z _ { i } - \beta _ { 0 } \pmb { x } _ { i } \cdot \pmb { o } _ { i } = - \sum _ { j } a _ { i j } } \end{array}$ ln $a _ { i j } \leq$ ln n, together with $Z _ { i } \geq \mathrm { e } ^ { \beta _ { 0 } M _ { i } }$ , implies

$$
E _ { i } = { \pmb x } _ { i } \cdot { \pmb o } _ { i } = \frac { \sum _ { j } \mathrm { e } ^ { \beta _ { 0 } { \pmb x } _ { i } \cdot { \bf y } _ { j } } \left( { \pmb x } _ { i } \cdot { \pmb y } _ { j } \right) } { \sum _ { j } \mathrm { e } ^ { \beta _ { 0 } { \pmb x } _ { i } \cdot { \bf y } _ { j } } } , \qquad 0 \leq M _ { i } - E _ { i } \leq \frac { \ln n } { \beta _ { 0 } } \leq \frac { 1 } { C } .
$$

For an approximate output $\widehat { \mathbf { o } } _ { i }$ with coordinatewise error at most $\varepsilon _ { 0 }$ , we have $| { \pmb x } _ { i } \cdot \widehat { \pmb \sigma } _ { i } - M _ { i } | \leq$ $d B \varepsilon _ { 0 } + 1 / C \le 2 / C < 1 / 2$ . Rounding to the nearest integer determines $M _ { i }$ exactly.

In part (b), scaling the values scales the exact output to $d B o _ { i } .$ , so dividing the inner product of its approximation with $\mathbf { \mathcal { x } } _ { i }$ by dB estimates $E _ { i }$ with error at most $\varepsilon _ { c } .$ . Since $\varepsilon _ { c } + 1 / C = 3 / 8 < 1 / 2$ the same rounding applies. All value channels share the same softmax probabilities, so the scalar calls compute the vector output in part (c). In all three cases, the inputs and dyadic outputs have logarithmic encoding length in $m + n , d , B$ , and forming the inputs and rounding the outputs require $\mathrm { O } ( ( m + n ) d )$ coordinate operations (Appendix A.1). □

Computing max<sub>i</sub> $M _ { i }$ also solves $\mathrm { M a x I P } _ { d }$ with the same additional cost, which is near-linear for fixed d and polynomially bounded integer inputs. The reductions are deterministic and preserve Las Vegas guarantees.

## 3.2 Conditional Hardness in Superconstant Dimension

We use the randomized version of the Orthogonal Vectors Hypothesis (OVH) [Wil05], which states that for every $\delta > 0$ , there is a constant $c > 0$ such that randomized algorithms with bounded error cannot determine in ${ \mathrm { O } } ( n ^ { 2 - \delta } )$ time whether two sets of n Boolean vectors in $\{ 0 , 1 \} ^ { \lceil c \log n \rceil }$ contain an orthogonal pair. Although the attention lower bounds of [AS23; GHSXY26] were originally stated under the Strong Exponential Time Hypothesis (SETH) [IP01], their reductions factor through orthogonal vectors and hold directly under OVH; hence they are unafected by recent subexponential-time results refuting ETH and SETH (while all algorithmic bounds and reductions elsewhere in this paper are unconditional).

We say that a dimension function $d ( n )$ is constructible if its value can be computed in $n ^ { \mathrm { o ( 1 ) } }$ time. Saha, ${ \mathrm { X u } } ,$ and Ye [SXY26, Theorem 4.7] establish conditional $n ^ { 2 - \mathbf { o } ( 1 ) }$ hardness for integer orthogonal vectors in every constructible superconstant dimension, with $\mathrm { O } ( \log n )$ -bit entries. Their Lemma B.1, based on [Wil18; Che20], reduces integer orthogonal vectors in dimension $d _ { \mathrm { O V } }$ to exact integer maximum inner product in dimension $d _ { \mathrm { O V } } ^ { 2 }$ . Applying their theorem with $d _ { \mathrm { O V } } ( n ) = \lfloor { \sqrt { d ( n ) } } \rfloor$ , which is constructible and tends to infinity, and padding with zeros establishes the same hardness for exact integer maximum inner product in dimension $d ( n )$ . They observe that this lower bound transfers to attention through the reduction of [GHSXY26, Lemma C.3], which uses polylogarithmically many attention calls with $n + 1$ queries and keys in dimension $d + 1$ and already works at constant error. In superconstant dimension, the extra dimension and the extra calls of that reduction are immaterial; even so, applying Theorem 3.1 directly yields this hardness at the same dimension and input sizes.

Corollary 3.2. Assume the randomized OVH, and let $d ( n )$ be a constructible function with $d ( n ) $ ∞ and $d ( n ) = n ^ { \mathrm { o ( 1 ) } }$ . Then attention in dimension $d ( n )$ , with $m = n , r \in \{ 1 , d ( n ) \}$ value channels, and polynomial coordinate magnitudes, requires $n ^ { 2 - \mathrm { o ( 1 ) } }$ time at either inverse-polynomial error or constant error tolerance $\varepsilon _ { c } = 1 / 4$

Proof. Suppose attention on these instances has running time ${ \mathrm { O } } ( n ^ { 2 - \delta } )$ for some constant $0 < \delta < 1$ Theorem 3.1 solves the maximum-inner-product hard instances with one vector-valued call or $d ( n )$ scalar calls, on inputs of polynomial magnitude and with inverse-polynomial error tolerance. Since $d ( n ) = n ^ { \mathrm { o ( 1 ) } }$ , the additional work is $\widetilde { \mathrm { O } } ( n d ( n ) ) = n ^ { 1 + \mathrm { o } ( 1 ) }$ and either implementation runs in total time $n ^ { 2 - \delta + \mathrm { o } ( 1 ) }$ . Parts (b) and (c) imply the same complexity at constant error, so composing either reduction with the deterministic $\mathrm { O V - t o - M a x { - } I P }$ reductions of [Wil05; SXY26] contradicts randomized OVH. □

## 4 A Deterministic Algorithm in Two Dimensions

We begin with the two-dimensional case (d = 2), which both establishes Theorem 4.1 and serves as the base case for our dimension induction in Section 5. Our main geometric observation is that centering the key polygon at its area centroid allows us to enclose all near-maximal keys for each query within a simple shell region specified by a radial threshold and at most two intervals along the polygon boundary, over which polynomial moments can be summed by a one-dimensional sweep. Throughout this section, the number r of value channels is fixed, and $N = m + n$ denotes the total number of query and key vectors.

Theorem 4.1 (Planar attention; proof in Section 4.5). For $d = 2$ and any fixed number r of value channels, approximate attention as defined in Definition 2.1 can be computed deterministically in $\widetilde { \mathrm { O } } ( N )$ time.

## 4.1 Shifted-Kernel Products in the Plane

For query vectors $\pmb q _ { i } \in \mathbb { Q } ^ { 2 } \ ( i \in [ m ] )$ and keys $\pmb { k } _ { j } \in \mathbb { Q } ^ { 2 } \ ( j \in [ n ] )$ , define the shifted kernel of the original inputs by

$$
T _ { i j } = \mathrm { e } ^ { q _ { i } \cdot k _ { j } - \operatorname * { m a x } _ { j ^ { \prime } \in [ n ] } q _ { i } \cdot k _ { j ^ { \prime } } } , \qquad { \cal T } = ( T _ { i j } ) , \qquad { \cal N } = m + n .
$$

We compute either the forward product $\mathbf { \mathit { T w } } ^ { \mathrm { { i n } } }$ or the transpose product $\pmb { T } ^ { \top } z ^ { \mathrm { i n } }$ , with input weights ${ \pmb w } ^ { \mathrm { i n } } \in \mathbb { Q } ^ { n }$ and $z ^ { \mathrm { i n } } \in \mathbb { Q } ^ { m }$

Lemma 4.2. Let $U , W \geq 1$ be integer bounds on the absolute values of the geometric coordinates and input weights, respectively, and let b bound their rational encoding lengths and that of the error tolerance $\eta \in \mathbb { Q } \cap ( 0 , 1 ]$ . Each $o f T w ^ { \mathrm { i n } }$ and $\pmb { T } ^ { \top } z ^ { \mathrm { i n } }$ can be computed deterministically to coordinatewise additive error at most η in $\widetilde { \mathrm { O } } ( m + n )$ time.

To prove Lemma 4.2, we construct a rational approximation P whose rowwise $\ell _ { 1 }$ error implies product error $\eta$ (Section 4.5), and show in Sections 4.2–4.4 how to evaluate its forward and transpose products through moment sweeps.

## 4.2 Centering and Radial Coordinates

Given a target rowwise error $\eta _ { \mathrm { r o w } } \in \mathbb { Q } \cap ( 0 , 1 ]$ , we round the input coordinates to the dyadic grid of Appendix A.2, changing each kernel row by at most $\eta _ { \mathrm { r o w } } / 1 6$ in $\ell _ { 1 }$ norm (Lemma A.1). Collinear rounded keys admit a one-dimensional sweep, described in the same appendix. Otherwise, when their hull has positive area, we compute its vertices in cyclic order and translate all rounded keys by the negative of its area centroid c. The resulting common shift of each row and its maximum leaves the rounded shifted scores unchanged.

In Sections 4.2–4.4, write $\pmb q _ { i }$ for the rounded queries and $k _ { j }$ for the rounded and centered keys, and let

$$
\mathcal C = \operatorname { c o n v } \{ k _ { 1 } , \ldots , k _ { n } \}
$$

be the centered key polygon, with $h _ { i } = \operatorname* { m a x } _ { j } { \pmb q _ { i } \cdot \pmb k _ { j } }$ . The matrix T always denotes the original shifted kernel. The origin is now the area centroid and lies in the interior of ${ \mathcal { C } } ;$ since $\| \pmb { c } \| _ { \infty } \le U + 1$ the working coordinates satisfy $\| \pmb { k } _ { j } \| _ { \infty } \leq 2 ( U + 1 )$ and $\| \pmb { q } _ { i } \| _ { \infty } \le U + 1$ . Computing the hull costs $\widetilde { \mathrm { O } } ( n )$ time, after which each $h _ { i }$ is computed by an $\widetilde { \mathrm { O } } ( 1 )$ -time support query [PS85; DK90], with ties resolved by a fixed vertex order.

Set the score window and polynomial degree to

$$
L = \Theta \Bigl ( \log \frac { N } { \eta _ { \mathrm { r o w } } } \Bigr ) , \qquad g = \Theta ( L ) ,
$$

with the constants specified in Appendix $\mathrm { { A . 2 } } .$ , which also includes all rounding errors in this section.

Lemma 4.3. For the centered two-dimensional polygon $\mathcal { C }$ above and $\pmb q \in \mathbb { R } ^ { 2 } \setminus \lbrace 0 \rbrace$ , set

$$
h = \operatorname* { m a x } _ { { \boldsymbol { x } } \in { \mathcal { C } } } { \boldsymbol { q } } \cdot { \boldsymbol { x } } , \qquad h ^ { - } = \operatorname* { m i n } _ { { \boldsymbol { x } } \in { \mathcal { C } } } { \boldsymbol { q } } \cdot { \boldsymbol { x } } .
$$

Then $h > 0$ and $h ^ { - } \geq - 2 h$

Proof. The origin lies in the interior, so $h > 0$ . To bound the minimum, triangulate the polygon by a fan from a vertex x<sup>−</sup> with $\pmb q \cdot \pmb x ^ { - } = h ^ { \pmb { \operatorname {  } } }$ <sup>−</sup>. Each triangle then has one vertex projecting to h<sup>−</sup> and two projecting to at most $h ,$ so its centroid projects to at most $( h ^ { - } + 2 h ) / 3$ . Their area-weighted average is the polygon’s centroid projection, zero, so $0 \leq ( h ^ { - } + 2 h ) / 3$ □

By Lemma 4.3, every centered key satisfies

$$
- 3 h _ { i } \le { \pmb q } _ { i } \cdot { \pmb k } _ { j } - h _ { i } \le 0 .
$$

For each nonzero key, the ray from the origin through it meets the boundary at a unique point $k _ { j } ^ { \partial }$ so we can write

$$
k _ { j } = \rho _ { j } k _ { j } ^ { \partial } , \qquad 0 < \rho _ { j } \leq 1 .
$$

We call $\rho _ { j }$ the key’s radial ratio and sort the nonzero keys by the cyclic order of their boundary points, equivalently by angle about the origin. Each comparison uses a halfplane test and a cross product, with keys sharing a boundary point stored as separate consecutive entries that retain their own indices and weights.

Binary search among the angularly ordered vertices locates the ray’s sector; intersecting its edge determines $k _ { j } ^ { \partial }$ in $\widetilde { \mathrm { O } } ( 1 )$ time, with a ray through a vertex returning that vertex. Keys at the origin contribute only to the constant moment in Section 4.4.

## 4.3 Radial and Boundary Ranges

For each row, we select a range containing every key within L of the maximum. When $h _ { i } \leq L$ , all shifted scores lie in $[ - 3 L , 0 ]$ by the centroid bound, so we set $\kappa _ { i } = [ n ]$ . For $h _ { i } > L ,$ , on the other hand, we define

$$
{ \cal K } _ { i } = \left\{ j : { \pmb k } _ { j } \neq 0 , \quad { \rho } _ { j } \geq 1 - \frac { L } { h _ { i } } , \quad { \pmb q } _ { i } \cdot { \pmb k } _ { j } ^ { \partial } \geq h _ { i } - L \right\} .
$$

Lemma 4.4. For $h _ { i } > L$

$$
\{ j : q _ { i } \cdot k _ { j } \geq h _ { i } - L \} \subseteq K _ { i } \subseteq \{ j : q _ { i } \cdot k _ { j } \geq h _ { i } - 2 L \} .
$$

Proof. If $\pmb q _ { i } \cdot \pmb k _ { j } \geq h _ { i } - L > 0$ , then $k _ { j } \neq 0$ and $q _ { i } \cdot k _ { j } ^ { \partial } > 0$ . Since $q _ { i } \cdot k _ { j } ^ { \partial } \leq h _ { i }$ and $\rho _ { j } \leq 1$ ，

$$
\rho _ { j } = \frac { \pmb q _ { i } \cdot \pmb k _ { j } } { \pmb q _ { i } \cdot \pmb k _ { j } ^ { \partial } } \geq 1 - \frac { L } { h _ { i } } , \qquad \pmb q _ { i } \cdot \pmb { k } _ { j } ^ { \partial } \geq \pmb q _ { i } \cdot \pmb { k } _ { j } \geq h _ { i } - L .
$$

Conversely, both lower bounds defining $\kappa _ { i }$ are positive, so every $j \in \mathcal { K } _ { i }$ satisfies

$$
q _ { i } \cdot k _ { j } \geq \left( 1 - { \frac { L } { h _ { i } } } \right) \left( h _ { i } - L \right) = h _ { i } - 2 L + { \frac { L ^ { 2 } } { h _ { i } } } \geq h _ { i } - 2 L .
$$

Thus $\kappa _ { i }$ contains every near-maximal key, including a maximizer, and widens the score window by at most a factor of two. Set

$$
s _ { i j } = { \bf q } _ { i } \cdot { \bf k } _ { j } - h _ { i } + 3 L ,
$$

so that the centroid bound for $h _ { i } \leq L$ and Lemma 4.4 for $h _ { i } > L$ imply

$$
0 \leq s _ { i j } \leq 3 L \quad ( j \in { \cal K } _ { i } ) , \qquad q _ { i } \cdot k _ { j } - h _ { i } < - L \quad ( j \notin { \cal K } _ { i } ) .
$$

For a row with $h _ { i } > L$ , the boundary condition selects a closed arc containing the maximizing face. Since each intermediate level line meets the boundary at the two ends of its intersection segment, projection is monotone along either chain from a maximum to a minimum. Binary search locates the arc endpoints, and cutting the cyclic order at a fixed position represents the arc by at most two disjoint intervals. All radial and boundary thresholds are inclusive, so the searches and sweeps below include equality at every endpoint. Together with the radial threshold, these intervals describe the part of the closed shell between $( 1 - L / h _ { i } ) \mathcal { C }$ and C in the arc’s angular sector (Figure 3).

![](images/d4a4c5063660aace6ab950abeb4231b768fc11ec5dd81a4d7cdddca4bf3c09f3.jpg)  
Figure 3: A non-symmetric convex pentagon centered at its area centroid, with $h _ { i } = 3 L > L$ . Blue marks the eligible boundary arc, and the orange dashed polygon is $( 2 / 3 ) \mathcal { C }$ . All points in the shaded region, including its boundary, satisfy both retention conditions.

## 4.4 Polynomial Approximation and Moment Sweeps

For $j \in { \cal K } _ { i } , s _ { i j } \in [ 0 , 3 L ]$ . The degree-g Taylor polynomial $\begin{array} { r } { \exp _ { g } ( s ) = \sum _ { \ell = 0 } ^ { g } s ^ { \ell } / \ell ! } \end{array}$ satisfies

$$
0 \leq \mathrm { e } ^ { - 3 L } ( \mathrm { e } ^ { s } - \exp _ { g } ( s ) ) = \mathrm { O } ( \eta _ { \mathrm { r o w } } / N ) \qquad ( 0 \leq s \leq 3 L )
$$

by the standard remainder bound (Lemma A.2). Let $0 < \omega \leq \mathrm { e } ^ { - 3 L }$ be a rational approximation at the precision specified in Appendix A.2. Define

$$
P _ { i j } = \left\{ \begin{array} { l l } { \omega \exp _ { g } ( s _ { i j } ) , } & { j \in \mathcal { K } _ { i } , } \\ { 0 , } & { j \not \in \mathcal { K } _ { i } . } \end{array} \right.
$$

Both approximations are from below, so $0 \leq P _ { i j } \leq \mathrm { e } ^ { { \pmb q } _ { i } \cdot { \pmb k } _ { j } - h _ { i } } \leq 1$

Lemma 4.5. For the original shifted kernel T, the matrix P satisfies

$$
0 \leq P _ { i j } \leq 1 , \qquad \sum _ { j = 1 } ^ { n } | T _ { i j } - P _ { i j } | \leq \eta _ { \mathrm { r o w } } \qquad ( i \in [ m ] ) .
$$

Proof. For the rounded coordinates, omitted keys have weight below $\mathrm { e } ^ { - L }$ , and the Taylor and scalar approximations contribute $\mathrm { O } ( \eta _ { \mathrm { r o w } } / N )$ per selected key. Summing these errors and adding the coordinate-rounding error proves the rowwise estimate. □

Separated polynomial weights. Setting $\gamma _ { i } = 3 L - h _ { i }$ and expanding first in $q _ { i } \cdot k _ { j }$ and then in the two coordinates gives

$$
\exp _ { g } ( s _ { i j } ) = \sum _ { \ell _ { x } + \ell _ { y } \leq g } C _ { i , \ell _ { x } , \ell _ { y } } k _ { j x } ^ { \ell _ { x } } k _ { j y } ^ { \ell _ { y } } , \qquad C _ { i , \ell _ { x } , \ell _ { y } } = \frac { q _ { i x } ^ { \ell _ { x } } q _ { i y } ^ { \ell _ { y } } } { \ell _ { x } ! \ell _ { y } ! } \exp _ { g - \ell _ { x } - \ell _ { y } } ( \gamma _ { i } ) .
$$

The expansion has $\binom { g + 2 } { 2 }$ terms indexed by nonnegative $\ell _ { x } , \ell _ { y } ,$ each factoring into a query coeficient and a key monomial. For the rounded weights $\mathbf { \Delta } w , z .$ , the common denominators of Appendix A.2 allow exact summation of these terms with polynomial bit lengths.

Keys at the origin belong to $\kappa _ { i }$ exactly when $h _ { i } \leq L$ . Their total weight contributes to the constant moment in forward products, and their transpose products involve only the $( \ell _ { x } , \ell _ { y } ) = ( 0 , 0 )$ term.

Forward multiplication. For a rounded input vector ${ \pmb w } _ { \mathrm { : } }$ , the required moments are

$$
\operatorname { M o m } _ { i } ( \ell _ { x } , \ell _ { y } ) = \sum _ { j \in { \mathcal { K } } _ { i } } k _ { j x } ^ { \ell _ { x } } k _ { j y } ^ { \ell _ { y } } w _ { j } .
$$

We first precompute the moments over all keys for rows with $h _ { i } \leq L$ . To handle rows with $h _ { i } > L$ a Fenwick tree [Fen94] on the array of keys in boundary order supports the sweep below.

(1) Sort the rows with $h _ { i } > L$ by $1 - L / h _ { i }$ and the nonzero keys by $\rho _ { j }$ , both in nonincreasing order.

(2) Before answering a row, insert the vectors of weighted monomials for every key at or above the row’s radial threshold.

(3) Query the row’s one or two boundary intervals to obtain its moments.

For every row, the sum of these moments weighted by $C _ { i , \ell _ { x } , \ell _ { y } }$ and scaled by $\omega \ \mathrm { i s } \ ( P w ) _ { i }$ <sub>i</sub>.

Transpose multiplication. For a rounded input vector z, we reverse updates and queries to evaluate

$$
( \boldsymbol { P } ^ { \top } \boldsymbol { z } ) _ { j } = \omega \sum _ { \ell _ { x } + \ell _ { y } \leq g } k _ { j x } ^ { \ell _ { x } } k _ { j y } ^ { \ell _ { y } } \sum _ { i : j \in { \cal K } _ { i } } C _ { i , \ell _ { x } , \ell _ { y } } z _ { i } .
$$

Use a diference Fenwick tree on the same array in boundary order for interval additions and point queries.

(1) Sort the rows with $h _ { i } > L$ by $1 - L / h _ { i }$ and the nonzero keys by $\rho _ { j }$ , both in nondecreasing order.

(2) Before processing key j, add the vector $\left( C _ { i , \ell _ { x } , \ell _ { y } } z _ { i } \right)$ over the boundary intervals of every row whose threshold is at most $\rho _ { j }$

(3) Query the boundary index of key j to obtain the contribution from rows with $h _ { i } > L$ to each inner sum.

Adding $\begin{array} { r } { \sum _ { i : h _ { i } \leq L } C _ { i , \ell _ { x } , \ell _ { y } } z _ { i } } \end{array}$ completes the inner sum for every key; its weighted sum with the key monomials, scaled by $\omega ,$ is the transpose product.

## 4.5 Kernel Products and Attention

Proof of Lemma 4.2. Set $\eta _ { \mathrm { r o w } } = \eta / ( 8 N ( W + 1 ) )$ . By Lemma 4.5,

$$
\| ( T - P ) \ b { w } ^ { \mathrm { i n } } \| _ { \infty } \leq W \eta _ { \mathrm { r o w } } , \qquad \| ( T - P ) ^ { \top } \ b { z } ^ { \mathrm { i n } } \| _ { \infty } \leq m W \eta _ { \mathrm { r o w } } .
$$

The moment sweeps evaluate the products with P in $\widetilde { \mathrm { O } } ( N )$ time. Including the input and output rounding detailed in Appendix A.2, both errors are at most $\eta .$ □

Proof of Theorem $4 \cdot 1 .$ By Lemma 4.2, the products in Section 2 can be computed to tolerance $\eta = \varepsilon / ( 8 ( B + 1 ) )$ in $\widetilde { \mathrm { O } } ( N )$ time for fixed r. Lemma 2.5 then implies an attention error at most ε. □

## 4.6 Online Queries after Preprocessing the Keys

Because the forward sweep in Section 4.4 inserts keys in nonincreasing order of $\rho _ { j }$ independent of the queries, we can make the sweep structure persistent [DSST89]: storing successive versions of the boundary-order segment tree allows any arriving query to retrieve the exact moments it needs in $\widetilde { \mathrm { O } } ( 1 )$ time. This yields the following static data structure.

Corollary 4.6. Fix a constant number r of value channels, matrices $\ b { K } \in \mathbb { Q } ^ { n \times 2 }$ and $V \in \mathbb { Q } ^ { n \times r }$ with rows $k _ { j }$ and $v _ { j }$ , and an error tolerance $\varepsilon = 2 ^ { - p }$ as in Definition 2.1. Given bounds $B \geq 1$ and b on the magnitudes and rational encoding lengths of all key, value, and future query coordinates, there is a deterministic data structure with $\widetilde { \mathrm { O } } ( n )$ preprocessing time and storage that, for any query $q \in \mathbb { Q } ^ { 2 }$ satisfying these bounds, returns an ε-additive approximation to

$$
\frac { \sum _ { j = 1 } ^ { n } \mathrm { e } ^ { \boldsymbol { q } \cdot \boldsymbol { k } _ { j } } \boldsymbol { v } _ { j } } { \sum _ { j = 1 } ^ { n } \mathrm { e } ^ { \boldsymbol { q } \cdot \boldsymbol { k } _ { j } } }
$$

in $\widetilde { \mathrm { O } } ( 1 )$ time. The suppressed factors depend on b, log n, log $B ,$ and $\log ( 1 / \varepsilon )$ ; the guarantee holds for every query in an arbitrarily long, possibly adaptive stream.

Proof. Set the parameters of Section 4.2 with $U = W = \lceil B \rceil , N = n + 1$ , and $\eta _ { \mathrm { r o w } } = \Theta ( \varepsilon / ( W + 1 ) )$ Round the keys and values as in Appendix $\mathrm { A . 2 } ;$ these choices depend only on the given bounds and tolerance.

For a two-dimensional hull, we center the keys and use their radial ratios and boundary order to build a persistent binary segment tree [DSST89] storing interval moments for every value channel and the all-ones channel. We insert the nonzero keys in nonincreasing order of radial ratio and also store the global moments over all keys.

Upon arrival, a query is rounded to the same geometric grid and its support value h is computed by a support query. We read the global moments when $h \leq L ,$ or select the version at the radial threshold when $h > L$ , and read the moments from its one or two boundary intervals. Combining these moments with the row coeficients and dividing the numerator by the denominator yields the approximate attention output, with the common factor ω canceling from the ratio.

By Lemmas 4.5 and 2.5, the error is $\mathrm { O } ( ( W + 1 ) \eta _ { \mathrm { r o w } } )$ . Appendix A.2 includes the rounding errors and establishes the preprocessing, storage, and query costs. Because the stored moments are exact, the error bound holds for every query satisfying the input bounds, including throughout an adaptive stream. □

## 5 Shifted Kernels and Dimension Induction

Using the planar shifted-kernel algorithm as the base case, we prove the fixed-dimensional bound for $d \geq 3$ by induction on the dimension. Each inductive step consists of two reductions (Figure 4):

$$
\mathrm { K P } _ { d } \longrightarrow \mathrm { B P } _ { d - 1 } \longrightarrow \mathrm { K P } _ { d - 1 } ( d \geq 3 ) .
$$

First, Lagrange temperature interpolation decouples the row scales from the inner products, reducing $\mathrm { K P } _ { d }$ to $\mathrm { B P } _ { d - 1 }$ with an additive scalar bias on each side. Second, partitioning the significant pairs into common-anchor blocks allows us to cancel these biases and reduce $\mathrm { B P } _ { d - 1 }$ to $\mathrm { K P } _ { d - 1 }$ . Throughout Sections 5–7, d is fixed and we use the precision convention of Section 2.1.

## 5.1 Exact Maxima and Normalized Directions

The reduction begins by computing the row maxima exactly and normalizing each nonzero row by a coordinate of largest absolute value. Both these maxima and those used later in the anchor factorization can be computed within Chan’s ofline ray-shooting bound [Cha96b, Corollaries 2.2 and 2.4], with the bit complexity stated below.

Proposition 5.1 (Exact maxima; proof in Appendix B.6). For fixed $d \geq 2$ , let $\pmb q _ { i } \in \mathbb { Q } ^ { d } \ ( i \in [ m ] )$ be queries and $\pmb { k } _ { j } \in \mathbb { Q } ^ { d } \mathrm { ~ } ( j \in [ n ] )$ be points. A Las Vegas algorithm computes every maximum $M _ { i } = \operatorname* { m a x } _ { j \in [ n ] } { \pmb q _ { i } \cdot \pmb k _ { j } }$ and a maximizing index in expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ . For $d \leq 3 .$ , the algorithm is deterministic and runs in $\widetilde { \mathrm { O } } ( m + n )$ time.

![](images/d8c13d5bd502f73e271dff734354416dd2862b976ba8e87183d0308b2830b4b4.jpg)  
Figure 4: One induction step, read clockwise from the upper left. Row shifts express attention through $\operatorname { K P } _ { d } ;$ Lagrange temperature interpolation then gives $\mathrm { B P } _ { d - 1 }$ , whose anchor blocks factor through $\mathrm { K P } _ { d - 1 }$ . The induction ends at the planar kernel algorithm.

Definition 5.2 (Coordinate normalization). For $\ell \in [ d ]$ , let $\pi _ { \ell , + }$ exchange coordinates ℓ and $d ,$ and let $\pi _ { \ell , \ell } $ <sub>−</sub> also change the sign of the last coordinate. For $q \neq 0$ , let ℓ be the least index with $| q _ { \ell } | = \| q \| _ { \infty }$ and let $\pi _ { q }$ be the corresponding permutation, with the sign chosen so that $\pi _ { q } q$ has a positive last coordinate. Write

$$
\begin{array} { r } { \pi _ { q } q = \beta _ { q } ( u _ { q } , 1 ) , \qquad \beta _ { q } = \| \pmb { q } \| _ { \infty } > 0 , \qquad \pmb { u } _ { q } \in [ - 1 , 1 ] ^ { d - 1 } . } \end{array}
$$

Rows sharing $\pi _ { q }$ form a coordinate group; there are at most 2d groups. For a signed coordinate permutation π and $\boldsymbol { y } \in \mathbb { R } ^ { d }$ , define the afine function $f _ { y } ^ { \pi } ( { \pmb u } ) = ( \pi { \pmb y } ) \cdot ( { \pmb u } , 1 )$ , and write $f _ { y } = f _ { y } ^ { \mathrm { i d } }$ . By orthogonality,

$$
\begin{array} { r } { \pmb q \cdot \pmb y = \beta _ { \pmb q } f _ { \pmb y } ^ { \pi _ { q } } ( \pmb u _ { q } ) . } \end{array}
$$

For $\pmb { y } = ( \pmb { p } , c )$ , we have $f _ { y } ( { \pmb u } ) = { \pmb u } \cdot { \pmb p } + c .$

## 5.2 Shifted-Kernel Products

Definition 5.3 (Shifted-kernel products). For $d \geq 2$ , an instance of $\mathrm { K P } _ { d }$ consists of rational rows $\pmb q _ { i } \in \mathbb { Q } ^ { d } \mathrm { ~ } ( i \in [ m ] )$ , rational columns $\pmb { k } _ { j } \in \mathbb { Q } ^ { d } \mathrm { ~ } ( j \in [ n ] )$ , and an error tolerance $\eta \in \mathbb { Q } \cap ( 0 , 1 ]$ . Their shifted kernel is

$$
M _ { i } = \operatorname* { m a x } _ { j \in [ n ] } { \pmb q _ { i } \cdot \pmb k _ { j } } , \qquad T _ { i j } = \mathrm { e } ^ { { \pmb q _ { i } \cdot \pmb k _ { j } } - M _ { i } } .
$$

The task is to compute either the forward product $\mathbf { \nabla } ^ { T w }$ , with $w \in \mathbb { Q } ^ { n }$ , or the transpose product $\boldsymbol { T ^ { \intercal } z }$ , with $z \in \mathbb { Q } ^ { m }$ . Let integers $U , W \geq 1$ bound the magnitudes of coordinates and input weights, respectively, and let b bound the encoding lengths of all input rationals, including η.

A solution is a vector $\widehat { \mathbf { o } }$ approximating the specified product to coordinatewise additive error η.

In dimensions $d \geq 3 .$ , the algorithm shifts each row by its exact maximum, computed by Proposition 5.1 or provided with the instance, and evaluates the products over blocks. In recursive calls from the anchor factorizations of Lemma 7.4, these maxima are already computed, so the kernel calls reuse them. Moreover, because these factorizations involve both forward and transpose products, the induction maintains both operations, which the planar algorithm computes in $\widetilde { \mathrm { O } } ( m + n )$ time for the base case (Lemma 5.7).

## 5.3 Biased Products and Temperature Interpolation

The intermediate problem $\mathrm { B P } _ { d - 1 }$ has $d - 1$ bilinear coordinates and one scalar bias on each side. Its lifted source points lie in $\mathbb { Q } ^ { d }$

Definition 5.4 (Biased exponential products). For $d \geq 3 ,$ , an instance of $\mathrm { B P } _ { d - 1 }$ consists of rational rows $( \boldsymbol { \mathscr { u } } _ { i } , \boldsymbol { b } _ { i } ) \in \mathbb { Q } ^ { d - 1 } \times \mathbb { Q }$ indexed by $I _ { 0 } = [ m ]$ and columns $( \pmb { { p } } _ { j } , c _ { j } ) \in \mathbb { Q } ^ { d - 1 } \times \mathbb { Q }$ indexed by $J _ { 0 } = [ n ]$ with nonpositive scores

$$
x _ { i j } = { \pmb u } _ { i } \cdot { \pmb p } _ { j } + b _ { i } + c _ { j } \leq 0 , \qquad S _ { i j } = { \mathrm e } ^ { x _ { i j } } .
$$

Given weights $\pmb { w } \in \mathbb { Q } ^ { J _ { 0 } }$ and an error tolerance $\eta \in \mathbb { Q } \cap ( 0 , 1 ]$ , the task is to approximate $_ { s w }$ to coordinatewise additive error $\eta .$ Let integers $U , W \geq 1$ bound the magnitudes of coordinates and biases, and of input weights, respectively, and let b bound the encoding lengths of all input rationals. Write $N = m + n$

The score formula is symmetric under exchanging $( { \pmb u } _ { i } , b _ { i } )$ with $( p _ { j } , c _ { j } )$ , which transposes $\pmb { S }$ and preserves its numerical parameters. A forward biased-product call on the exchanged instance therefore computes $S ^ { \top }$ times a vector with the same symmetric running-time bound $F _ { d } ( m , n )$ . By contrast, in $\mathrm { K P } _ { d - 1 }$ , rowwise normalization breaks this symmetry by fixing an orientation, so the anchor factorization of Lemma 7.4 uses the forward product for a row anchor and the transpose product for a column anchor.

Proposition 5.5. Let $d \geq 3$ $I f \mathrm { B P } _ { d - 1 }$ can be solved by a Las Vegas algorithm in expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ , then $\mathrm { K P } _ { d }$ can be solved by a Las Vegas algorithm in the same expected time.

Proof of Proposition 5.5. Consider a $\mathrm { K P } _ { d }$ instance with parameters $( \eta , U , W , b ) , N = m + n$ , and input weights $w \in \mathbb { Q } ^ { n }$ for the forward product or $z = ( z _ { i } ) _ { i \in [ m ] } \in \mathbb { Q } ^ { m }$ for the transpose product. Use the supplied row maxima or compute them by Proposition 5.1 in expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ . After handling zero rows as in Section 2, we apply Definition 5.2 to write each nonzero row as $\pmb { q } _ { i } = \beta _ { i } \pi _ { i } ^ { - 1 } ( \pmb { u } _ { i } , 1 )$ with $\beta _ { i } > 0$ . Setting $h _ { i } = M _ { i } / \beta _ { i }$ , we have

$$
\begin{array} { r } { \pmb { q } _ { i } \cdot \pmb { k } _ { j } = \beta _ { i } f _ { \pmb { k } _ { j } } ^ { \pi _ { i } } ( \pmb { u } _ { i } ) , \qquad \pmb { h } _ { i } = \operatorname* { m a x } _ { j } f _ { \pmb { k } _ { j } } ^ { \pi _ { i } } ( \pmb { u } _ { i } ) , \qquad T _ { i j } = \mathrm { e } ^ { - \beta _ { i } \mu _ { i j } } , \quad \mu _ { i j } = h _ { i } - f _ { \pmb { k } _ { j } } ^ { \pi _ { i } } ( \pmb { u } _ { i } ) \geq 0 . } \end{array}
$$

We partition these nonzero rows by their normalizing permutation and dyadic temperature interval into groups

$$
\mathcal { G } _ { \pi , \beta _ { \mathrm { l o } } } = \{ i : \pi _ { i } = \pi , ~ \beta _ { \mathrm { l o } } \leq \beta _ { i } < 2 \beta _ { \mathrm { l o } } \} ,
$$

where $\beta _ { \mathrm { l o } }$ ranges over powers of two. Since $2 ^ { - b } < \beta _ { i } \leq U$ , the number $n _ { \mathrm { g r p } }$ of nonempty groups is $\widetilde { \mathrm { O } } ( 1 )$

Within each nonempty group $\mathcal { G } = \mathcal { G } _ { \pi , \beta _ { \mathrm { l o } } }$ , we apply Lemma 2.6 with $n _ { \mathrm { i n t } } = \Theta ( \log ( N W / \eta ) )$ nodes $\tau _ { \ell }$ and coeficients $\mathcal { L } _ { i \ell } = \mathcal { L } _ { \ell } ( \beta _ { i } )$ . Because τ is shared by all rows in the group, we can absorb it into the column vectors and row shifts: each node defines a $\mathrm { B P } _ { d - }$ <sub>−1</sub> instance with rows $( { \pmb u } _ { i } , - \tau _ { \ell } h _ { i } )$ and columns $( { \pmb p } _ { j } , { \pmb c } _ { j } ) = \tau _ { \ell } \pi { \pmb k } _ { j }$ . Its matrix $S ^ { ( \mathcal { G } , \ell ) }$ has nonpositive scores

$$
\begin{array} { r } { ( { \pmb u } _ { i } , 1 ) \cdot ( { \tau } _ { \ell } { \pi } { \pmb k } _ { j } ) - { \tau } _ { \ell } { h } _ { i } = - { \tau } _ { \ell } { \mu } _ { i j } \le 0 . } \end{array}
$$

We solve these instances at tolerance

$$
\eta _ { \mathrm { B P } } = \Theta \left( \frac { \eta } { n _ { \mathrm { g r p } } n _ { \mathrm { i n t } } 9 ^ { n _ { \mathrm { i n t } } } } \right)
$$

with a small absolute constant. For forward multiplication, we apply each matrix to w and combine the returned vectors $\widehat { \mathbf { \xi } } _ { \widehat { \mathbf { \xi } } } \widehat { \mathbf { \xi } } _ { \widehat { \mathbf { \xi } } } ( \mathcal { G } , \mathcal { \ell } )$ for $i \in \mathcal { G }$ as

$$
\sum _ { \ell } \mathcal { L } _ { i \ell } \widehat { o } _ { i } ^ { ( \mathcal { G } , \ell ) } .
$$

Conversely, for transpose multiplication, we exchange the two sides of each biased instance and use weights $( \mathcal { L } _ { i \ell } z _ { i } ) _ { i \in \mathcal { G } }$ , then sum the resulting vectors over all groups and nodes. Since $| { \mathcal { L } } _ { i \ell } z _ { i } | \leq 9 ^ { n _ { \mathrm { i n t } } } W$ the logarithm of the weight bound remains ${ \mathrm { O } } ( \lambda )$

By Lemma 2.6, the interpolation error is at most $N W ( 3 / 4 ) ^ { n _ { \mathrm { i n } } }$ <sup>t</sup> for either product, to which the biased calls add at most $9 ^ { n _ { \mathrm { i n t } } }$ η<sub>BP</sub> per forward output coordinate and $n _ { \mathrm { g r p } } n _ { \mathrm { i n t } } \eta _ { \mathrm { B P } }$ per transpose output coordinate. A large enough constant in $n _ { \mathrm { i n t } }$ and a small enough constant in η<sub>BP</sub> give error at most $\eta / 2 ,$ leaving $\eta / 2$ for the arithmetic errors analyzed in Appendix $\mathrm { A . 4 }$

Since there are ${ \mathrm { O } } ( 1 )$ groups and nodes, each defining a biased instance with side sizes at most $m , n ,$ possibly exchanged, the total expected cost is $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ by symmetry and monotonicity of $F _ { d }$ . Normalizing the rows, computing the interpolation coeficients, and combining the outputs require an additional $\widetilde { \mathrm { O } } ( N )$ operations. □

To complete the induction, Section 7 partitions the significant pairs of a biased instance into common-anchor blocks. Lemma 7.4 reduces each block product to a $\mathrm { K P } _ { d - 1 }$ call, and the total expected evaluation time is $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ .

Proposition 5.6 (Proof in Section 7.6). Let $d \geq 3$ and suppose a Las Vegas algorithm solves $\mathrm { K P } _ { d - 1 }$ in expected time $\widetilde { \mathrm { O } } ( F _ { d - 1 } ( m , n ) )$ . Then $\mathrm { B P } _ { d - 1 }$ can be solved by a Las Vegas algorithm in expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ .

## 5.4 The Induction on the Dimension

Lemma 5.7. $\mathrm { K P _ { 2 } }$ can be solved deterministically in $\widetilde { \mathrm { O } } ( m + n )$ time.

Proof. Lemma 4.2 computes either product to error η in $\widetilde { \mathrm { O } } ( m + n )$ time.

Theorem 5.8 (Kernel products in fixed dimension). For every fixed integer $d \geq 2$ , a Las Vegas algorithm solves $\mathrm { K P } _ { d }$ in expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$

Proof. The base case is Lemma 5.7. Assuming the result in dimension $d - 1$ , Proposition 5.6 solves $\mathrm { B P } _ { d - 1 }$ in expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ , and Proposition 5.5 then solves $\mathrm { K P } _ { d }$ within this bound.

Across the $d - 2$ reductions, the numerical parameters grow by a fixed polynomial at each step and thus remain polynomial in the original parameters for fixed d $( \mathrm { A p p e n d i x ~ A . 4 } )$ □

## 6 Geometric Data Structures in Fixed Dimension

The biased-product algorithm uses the support, reporting, point-location, and sampling results in Lemmas 6.2, 6.3, 6.8, and 6.9. The underlying geometric constructions are usually analyzed in the real-RAM model [MS93; CS89; Cha96a; Cha96b]; Appendices A and B provide the rational implementations and bit-complexity bounds needed here.

In this section, $d \ge 2$ is fixed and $k = \lfloor d / 2 \rfloor$ . Input and query bounds are fixed before preprocessing, with parameter λ as in Section 2.1. We use polytope for a bounded intersection $P$ of closed halfspaces $H = \{ { \pmb x } : { \pmb a } _ { H } \cdot { \pmb x } \le b _ { H } \}$ with ${ \pmb a } _ { H } \neq { \bf 0 }$ , and denote its interior and vertex set by int P and vert(P). For $\pmb { x } \in \mathbb { R } ^ { d }$ , write $\bar { \pmb { x } } \in \mathbb { R } ^ { d - 1 }$ for its projection onto the first $d - 1$ coordinates. Ties may be resolved by returning any maximizing index in a support query and any closed cell containing the point in a location query.

Definition 6.1. A finite indexed collection of afine hyperplanes in $\mathbb { R } ^ { d }$ is in general position if the normal vectors of any at most d of its members are linearly independent and no $d + 1$ of its members have a common point. A finite indexed collection of closed halfspaces is in general position if their bounding hyperplanes are.

## 6.1 Classical Geometric Primitives

McMullen’s Upper Bound Theorem implies that there are $\mathrm { O } _ { d } ( h ^ { k } )$ vertices for a d-polytope defined by h halfspaces [McM70; Sei17]. For $d \geq 3$ , these vertices and their incidences can be enumerated by Clarkson and Shor’s algorithm in expected $\widetilde { \mathrm { O } } ( h ^ { k } )$ arithmetic operations [CS89, Theorems 4.7 and 4.8]. For queries, we use the high-storage ray-shooting structure of Matousek and Schwarzkopf [MS93, Lemma $7 ]$ , with deterministic alternatives in dimensions two and three [PS85; DK90].

The rational version, stated in Proposition B.3 in Appendix B.2, has the same soft-O bounds. Its implementation fixes a bounding simplex and constructs each geometric object from a constant number of original constraints.

## 6.2 Support Queries and Strict Reporting

A support query asks for a point maximizing a given linear objective, and polarity reduces it to ray shooting. The nonzero gaps between rational scores allow a general-position perturbation for which each perturbed maximizing index also maximizes the original objective.

Lemma 6.2 (Proof in Appendix A.7). Let $\mathscr { y } = ( \pmb { y } _ { s } ) _ { s \in \Sigma }$ be a nonempty list of points in $\mathbb { Q } ^ { d }$ , with repetitions allowed. A Las Vegas algorithm builds a support-query structure in expected time $\widetilde { \mathrm { O } } ( | \Sigma | ^ { k } )$ ， deterministically for $d \leq 3$ . Given an admissible $\pmb q \in \mathbb { Q } ^ { d }$ , the structure returns $\operatorname* { m a x } _ { s \in \Sigma } \pmb { q } \cdot \pmb { y } _ { s }$ and a maximizing index in time $\widetilde { \mathrm { O } } ( 1 )$

Support queries also test whether a strict halfspace contains any source. Applying the standard binary-tree reduction from emptiness to reporting [Cha96a, Observation 2.3] within groups of size at most $g _ { \mathrm { r e p } }$ yields a tradeof between preprocessing and query time.

Lemma 6.3 (Proof in Appendix B.3). Let $\mathscr { y } = ( \pmb { y } _ { s } ) _ { s \in \Sigma }$ be a nonempty list of points in $\mathbb { Q } ^ { d }$ , with repetitions allowed, and let $g _ { \mathrm { r e p } }$ be an integer with $1 \leq g _ { \mathrm { r e p } } \leq | \Sigma |$ . A Las Vegas algorithm builds a reporting structure in expected time $\widetilde { \mathrm { O } } ( | \Sigma | g _ { \mathrm { r e p } } ^ { k - 1 } )$ , deterministically for $d \leq 3$ . Given an admissible query $( \pmb q , \pmb \theta ) \in \mathbb { Q } ^ { d } \times \mathbb { Q }$ , the structure reports $Z = \{ s \in \Sigma : \pmb { q } \cdot \pmb { y } _ { s } > \theta \}$ in time $\widetilde { \mathrm { O } } ( | \Sigma | / g _ { \mathrm { r e p } } + 1 + | Z | )$ For $g _ { \mathrm { r e p } } = | \Sigma |$ , the preprocessing and query costs are $\widetilde { \mathrm { O } } ( | \Sigma | ^ { k } )$ and $\widetilde { \mathrm { O } } ( 1 + | Z | )$ , respectively.

## 6.3 Envelope Cells

For $d \geq 3$ , we organize near-maximal pairs by sampling sources and partitioning the upper envelope of their afine functions (Definition 5.2) into cells, each with a designated maximizing source. The perturbation and the bounding region containing all target positions are fixed across samples. We index sources by s and targets by i, in either orientation of the matrix.

Definition 6.4. Let $U , N _ { \Sigma } \geq 1$ be integers, let $\emptyset \neq \Sigma _ { 0 } \subseteq [ N _ { \Sigma } ]$ , and let $\pmb { y } _ { s } \in \mathbb { Q } ^ { d } \ ( s \in \Sigma _ { 0 } )$ satisfy $\| { \pmb y } _ { s } \| _ { \infty } \le U$ . The instance parameter λ also bounds $\log _ { 2 } ( N _ { \Sigma } + 2 )$ and $\log _ { 2 } ( U + 2 )$

Apply once the uniform perturbation of Appendix $\mathrm { A . 6 } _ { }$ obtaining source points $\widetilde { \mathbf { y } } _ { s }$ shared by all samples. For every nonempty $\Sigma ^ { \prime } \subseteq \Sigma _ { 0 }$ , define

$$
f _ { s } ( \pmb { u } ) = \pmb { y } _ { s } \cdot ( \pmb { u } , 1 ) , \qquad \widetilde { f } _ { s } ( \pmb { u } ) = \widetilde { \pmb { y } } _ { s } \cdot ( \pmb { u } , 1 ) , \qquad E _ { \Sigma ^ { \prime } } ( \pmb { u } ) = \operatorname* { m a x } _ { s \in \Sigma ^ { \prime } } \widetilde { f } _ { s } ( \pmb { u } ) ,
$$

and let the envelope polytope $\mathcal { P } ( \Sigma ^ { \prime } )$ be the epigraph of $E _ { \Sigma ^ { \prime } }$ clipped by the d fixed rational bounding halfspaces constructed in Appendix $\mathrm { A . 7 }$ . These halfspaces are shared by every sample, and the lower boundary over $[ - U , U ] ^ { d - 1 }$ is the graph of $E _ { \Sigma ^ { \prime } }$ . For $s \in \Sigma ^ { \prime }$ , the facet of source s is the possibly empty subset of $\mathcal { P } ( \Sigma ^ { \prime } )$ where its source constraint $\widetilde { f } _ { s } ( u ) \le z$ is tight.

Lemma 6.5 (Proof in Appendix A.7). The perturbed envelope in Definition $6 . 4$ satisfies:

(a) $| \widetilde { f } _ { s } ( u ) - f _ { s } ( u ) | \leq 1 / 8$ for every $s \in \Sigma _ { 0 }$ and every $\pmb { u } \in \mathbb { R } ^ { d - 1 }$ with $\| \mathbf { \boldsymbol { u } } \| _ { \infty } \leq U$

(b) Let $\pmb { u } \in \mathbb { Q } ^ { d - 1 }$ be an admissible target with $\| \mathbf { \boldsymbol { u } } \| _ { \infty } \leq U$ , and let $\emptyset \neq \Sigma ^ { \prime } \subseteq \Sigma _ { 0 }$ . Every $s \in \Sigma ^ { \prime }$ that maximizes $\widetilde { f } _ { s } ( \boldsymbol { u } )$ over Σ<sup>′</sup> also maximizes $f _ { s } ( { \pmb u } )$ over $\Sigma ^ { \prime }$

(c) For every nonempty $\Sigma ^ { \prime } \subseteq \Sigma _ { 0 }$ , the $| \Sigma ^ { \prime } | + d$ constraints of $\mathcal { P } ( \Sigma ^ { \prime } )$ are in general position, and $\mathcal { P } ( \Sigma ^ { \prime } )$ is bounded and has nonempty interior. Its constraint coeficients and vertices have $\mathrm { O } _ { d } ( \lambda )$ bits.

The perturbed sources, envelope vertices, and their afine values at vertices or targets satisfying the input bounds all have $\mathrm { O } _ { d } ( \lambda )$ bits. Lemmas 6.2 and 6.3 therefore apply with a uniform bit bound.

To decompose the envelope into cells, we apply Clarkson’s recursive vertex construction [Cla88, Section 3] to triangulate the source facets of the sampled envelope and project the simplices onto the target coordinates. Each projected cell has an owner whose function equals the sampled maximum on the entire cell, and its conflict set consists of the sources that exceed this owner somewhere on the cell. These sets contain every source that can improve a target’s sampled maximum.

Definition 6.6. Let $\emptyset \neq \Sigma ^ { \prime } \subseteq \Sigma \subseteq \Sigma _ { 0 }$ . Triangulate the source facets of $\mathcal { P } ( \Sigma ^ { \prime } )$ by the fixed rule of Definition B.7 in Appendix B.4, and project its simplices onto their first $d - 1$ coordinates. These projected simplices are the canonical cells. A cell $\Delta$ inherits its owner $s _ { \Delta } \in \Sigma ^ { \prime }$ from its source facet. The cells and their order depend only on $\Sigma ^ { \prime }$ , with the perturbation and bounding halfspaces fixed. The conflict set of $\Delta$ relative to the ground set Σ is

$$
C _ { \Delta } = \{ s ^ { \prime } \in \Sigma : \ \widetilde { f } _ { s ^ { \prime } } ( \pmb { u } ) > \widetilde { f } _ { s _ { \Delta } } ( \pmb { u } ) \ \mathrm { f o r \ s o m e \ } \pmb { u } \in \Delta \} .
$$

Lemma 6.7 (Proof in Appendix B.4). Let $\emptyset \neq \Sigma ^ { \prime } \subseteq \Sigma \subseteq \Sigma _ { 0 }$ . The canonical cells of $\mathcal { P } ( \Sigma ^ { \prime } )$ satisfy:

(a) Every cell $\Delta$ is a closed $( d - 1 )$ -dimensional simplex with d rational vertices, and its owner $s _ { \Delta }$ satisfies $\widetilde { f } _ { s _ { \Delta } } ( \pmb { u } ) \ = \ E _ { \Sigma ^ { \prime } } ( \pmb { u } )$ for every $\textbf { \textit { u } } \in \textbf { \textit { \Delta } }$ . The union of the cells contains $[ - U , U ] ^ { d - 1 }$

(b) For every cell $\Delta$ with owner $s _ { \Delta }$ , we have $C _ { \Delta } = \bigcup _ { \bar { \boldsymbol { v } } \in \mathrm { v e r t } ( \Delta ) } \{ s \in \Sigma : \widetilde { f } _ { s } ( \bar { \boldsymbol { v } } ) > \widetilde { f } _ { s _ { \Delta } } ( \bar { \boldsymbol { v } } ) \}$ and $C _ { \Delta } \cap$ $\Sigma ^ { \prime } = \emptyset$

(c) $\mathcal { P } ( \Sigma ^ { \prime } )$ has at most $C _ { \mathrm { c e l l } } | \Sigma ^ { \prime } | ^ { k }$ canonical cells, where $C _ { \mathrm { c e l l } } \geq 1$ depends only on d.

(d) Over all nonempty $\Sigma ^ { \prime } \subseteq \Sigma$ , at most $( | \Sigma | + d ) ^ { d ^ { 2 } + 1 }$ distinct owner–cell pairs $( s _ { \Delta } , \Delta )$ occur. For a fixed ground set Σ, each such pair has a fixed conflict set, independent of the sample in which it occurs.

Because the cells are closed, a target on a shared boundary may be placed in any cell containing it. To find such a cell, we use the face-by-face ray-shooting descent of Matousek and Schwarzkopf [MS93, proof of Theorem 10] and Chan [Cha96b, proof of Theorem 4.1], using the fixed triangulation order for the vertices. The descent then identifies a cell in the enumerated list.

Lemma 6.8. Let ∅ ̸= $\Sigma ^ { \prime } \subseteq \Sigma _ { 0 }$ . A Las Vegas algorithm enumerates and locates the canonical cells with expected preprocessing time $\widetilde { \mathrm { O } } ( | \Sigma ^ { \prime } | ^ { k } )$ . It returns

(a) the list of all canonical cells of $\mathcal { P } ( \Sigma ^ { \prime } )$ in the order of Definition $6 . 6 ,$ each with its owner and vertices;

(b) a structure that, given an admissible target $\pmb { u } \in \mathbb { Q } ^ { d - 1 }$ with $\| \mathbf { \boldsymbol { u } } \| _ { \infty } \leq U$ , locates a canonical cell $\Delta$ of $\mathcal { P } ( \Sigma ^ { \prime } )$ containing u, and returns its index in the list and its owner in time $\widetilde { \mathrm { O } } ( 1 )$

The list in part (a) is determined by $\Sigma ^ { \prime }$

Appendix B.4 proves Lemma 6.8, including the exact representation and comparison of cel indices.

## 6.4 Sampling Bounds

We resample whenever a cell has too many conflicts. A uniform sample passes these tests with high probability by Clarkson–Shor sampling [Cla87; CS89], with a union bound over the finite set of canonical configurations fixed by the shared perturbation.

Sampling parameters. For the original instance size $N = m + n \ge 1$ , we fix a bound $G$ on the number of doubling prefix levels and an initial prefix size $s _ { 0 }$ by setting

$$
G = \lceil \log _ { 2 } ( N + 8 ) \rceil , \qquad s _ { 0 } = 2 C _ { \mathrm { s a m p } } G ,\tag{6.1}
$$

where $C _ { \mathrm { s a m p } }$ depends only on d and is chosen in Appendix B.5.

Lemma 6.9 (Proof in Appendix B.5). In the setup of Definition $6 . 4 ,$ , assume $| \Sigma _ { 0 } | \leq N$ and use the parameters (6.1). Let $\emptyset \neq \Sigma \subseteq \Sigma _ { 0 }$ , and let $\Sigma ^ { \prime }$ be a uniformly random subset of Σ of a given size $| \Sigma ^ { \prime } | \geq 1$ . With probability at least $1 - 1 / ( 8 G )$ , every canonical cell $\Delta$ of $\mathcal { P } ( \Sigma ^ { \prime } )$ satisfies $| C _ { \Delta } | \le$ $C _ { \mathrm { s a m p } } | \Sigma | G / | \Sigma ^ { \prime } |$ , where the conflict sets are taken relative to $\Sigma$

## 7 Biased Exponential Products and the Main Theorem

Let $d \geq 3$ be fixed and set $k = k ( d )$ as in (2.1). Assuming a Las Vegas algorithm for $\mathrm { K P } _ { d - 1 }$ with expected running time $\widetilde { \mathrm { O } } ( F _ { d - 1 } ( m , n ) )$ , we solve $\mathrm { B P } _ { d - 1 }$ by covering its significant pairs with disjoint blocks, with total expected construction and evaluation time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ .

For an instance with row set $I _ { 0 } = [ m ]$ , column set $J _ { 0 } = [ n ]$ , and logarithmic parameter λ as in Section 2.1, we evaluate each block $I \times J$ against the restricted input vector ${ \pmb w } _ { J }$ and add its contribution to ${ \widehat { \mathbf { o } } } ,$ initially zero and indexed by $I _ { 0 }$ . The block tolerances ensure that their accumulated errors and the omitted tail remain within η.

Proposition 7.1 (Block decomposition). Under the preceding inductive hypothesis, let $N =$ $m + n$ and set the window $L ,$ anchor window $L _ { \mathrm { a n c } }$ , and block tolerance η as in (7.1) below. There is a Las Vegas algorithm that constructs and evaluates a finite collection $\mathcal { D } _ { \mathrm { b l k } }$ of nonempty blocks $D = I _ { D } \times J _ { D }$ with the following properties.

(a) Coverage and anchors. The blocks are pairwise disjoint subsets of $I _ { 0 } \times J _ { 0 }$ , and every pair outside them has score less than $- L + 1 / 8$ . Each block is evaluated directly or has an anchor: either a column $j _ { 0 } \in J _ { D }$ with $x _ { i j _ { 0 } } \geq - L _ { \mathrm { a n c } }$ for every $i \in I _ { D }$ , or a row $i _ { 0 } \in I _ { D }$ with $x _ { i _ { 0 } j } \geq - L _ { \mathrm { a n c } }$ for every $j \in J _ { D }$

(b) Block evaluation. Each block product is approximated to coordinatewise error at most $\eta _ { \mathrm { b l k } }$ Evaluating an anchor block involves one $\mathrm { K P } _ { d - 1 }$ call and one computation of exact batch maxima, in expected time $\widetilde { \mathrm { O } } ( F _ { d - 1 } ( | I _ { D } | , | J _ { D } | ) )$ ; direct evaluation costs $\widetilde { \mathrm { O } } ( | I _ { D } | | J _ { D } | )$ time.

(c) Total cost and error. The construction and evaluation have expected running time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ Extend each block output by zero to $I _ { 0 }$ and let $\widehat { \mathbf { o } }$ be their sum. Then

$$
\| \widehat { \pmb { o } } - \pmb { S } \pmb { w } \| _ { \infty } \leq n ( \eta _ { \mathrm { b l k } } + W \mathrm { e } ^ { - L + 1 / 8 } ) \leq \eta / 2 .
$$

## 7.1 Orientations, Hits, and Parameters

Because our recursive decomposition samples whichever side of a subproblem is more favorable, and reverses the roles of the two sides on light cells, we distinguish between the fixed row/column indices $I , J$ and the current source/target sets Σ, Θ. The two orientations are as follows:

$$
\left. \begin{array} { c c c } { { \mathrm { c o l u m n \ o r i e n t a t i o n } } } \\ { { \mathrm { r o w \ o r i e n t a t i o n } } } \end{array} \right| \begin{array} { c c c } { { \mathrm { s o u r c e s \ \Sigma } } } & { { \mathrm { t a r g e t s \ \Theta } } } \\ { { J } } & { { I } } \\ { { I } } & { { J } } \end{array}
$$

In the geometric procedures, we index sources by $s \in \Sigma$ and targets by $i \in \Theta$ in either orientation, as in Section 6.3. Write $( p _ { s } , c _ { s } )$ for the source data and $( { \pmb u } _ { i } , b _ { i } )$ for the target data in the current orientation. With $\pmb { y } _ { s } = ( p _ { s } , c _ { s } )$ and $f _ { s } ( { \pmb u } ) = { \pmb u } \cdot { \pmb p } _ { s } + c _ { s } .$ , the quantity $b _ { i } + f _ { s } ( { \pmb u } _ { i } )$ is the score of the corresponding original row–column pair. Reversing the orientation exchanges the two data families. Each block product maps the original column weights to the original row outputs, through $\pmb { T } ^ { \top }$ for a column anchor and T for a row anchor (Lemma 7.4).

For each orientation, fix the perturbation and bounding halfspaces of Definition 6.4 before sampling, with $\Sigma _ { 0 } = J _ { 0 }$ in the column orientation and $\Sigma _ { 0 } = I _ { 0 }$ in the row orientation, with $N _ { \Sigma } = N$ The instance bounds satisfy that definition for both sources and target positions, so the perturbed functions and the rule for forming canonical cells are fixed across all recursive calls. Lemmas 6.5 and 6.8(b) guarantee the perturbation bounds and point-location queries at every node.

Shared parameters. The sampling parameters $G , C _ { \mathrm { s a m p } } .$ , and $s _ { 0 }$ are those in (6.1). For the entire biased-product instance, set

$$
L = \Theta \bigg ( \log \frac { N W } { \eta } \bigg ) , \qquad L _ { \mathrm { a n c } } = L + 1 , \qquad \eta _ { \mathrm { b l k } } = \Theta ( \eta / N ) ,\tag{7.1}
$$

with integer $L \geq 4$ . The constants in the window and tolerance satisfy

$$
N ( \eta _ { \mathrm { b l k } } + W \mathrm { e } ^ { - L + 1 / 8 } ) \le \eta / 2 .
$$

Scores below the window contribute at most the tail term above, and the extra unit in $L _ { \mathrm { a n c } }$ allows for the perturbation. Every node restricts the original index sets with the perturbations, windows, and tolerances fixed. Thus the numerical factor in every local running time is a single polynomial in the original parameter λ (Appendix A.3).

Hits. We select blocks using the perturbed functions $\widetilde { f } _ { s }$ and evaluate them with the original scores $b _ { i } + f _ { s } ( { \pmb u } _ { i } )$ . The perturbation error of at most $1 / 8$ (Lemma 6.5) relates the geometric hit threshold to the original score window.

Lemma 7.2. A source $s \in \Sigma _ { 0 }$ is a hit for target i if $b _ { i } + \widetilde { f } _ { s } ( { \pmb u } _ { i } ) \geq - L$

(a) A hit has original score at least $- L - 1 / 8$

(b) A source that is not a hit has original score below $- L + 1 / 8$

(c) Let $\emptyset \neq \Sigma ^ { \prime } \subseteq \Sigma \subseteq \Sigma _ { 0 }$ , and let $\Delta$ be a canonical cell of $\mathcal { P } ( \Sigma ^ { \prime } )$ containing $\mathbf { \Delta } \mathbf { u } _ { i }$ . If no source in Σ<sup>′</sup> hits i, every hit in Σ belongs to $C _ { \Delta }$

Proof. Parts (a) and (b) follow from the perturbation bound $| \widetilde { f } _ { s } ( { \pmb u } _ { i } ) - f _ { s } ( { \pmb u } _ { i } ) | \leq 1 / 8$ (Lemma $6 . 5 ( \mathrm { a } ) )$ ). For part (c), the owner $s _ { \Delta } \in \Sigma ^ { \prime }$ has score below the hit threshold at ${ \mathbf { } } u _ { i } ,$ , so every hit s satisfies $\widetilde { f } _ { s } ( \pmb { u } _ { i } ) > \widetilde { f } _ { s _ { \Delta } } ( \pmb { u } _ { i } )$ and belongs to $C _ { \Delta }$ by Definition 6.6. □

## 7.2 Anchor Blocks and Direct Blocks

Definition 7.3. A block is a set $I \times J$ of pairs with $\emptyset \neq I \subseteq I _ { 0 }$ and $\emptyset \neq J \subseteq J _ { 0 }$ . Its matrix is ${ \cal S } _ { I J } = ( S _ { i j } ) _ { i \in I , j \in J } $ . A column anchor of the block is a column $j _ { 0 } \in J$ with $x _ { i j _ { 0 } } \geq - L _ { \mathrm { a n c } }$ for all $i \in I ,$ and a row anchor is a row $i _ { 0 } \in I$ with $x _ { i _ { 0 } j } \geq - L _ { \mathrm { a n c } }$ for all $j \in J$ . An anchor block is a block with a specified column or row anchor.

The product of a block $I \times J$ is $S _ { I J } \pmb { w } _ { J } \in \mathbb { R } ^ { I }$ . Subtracting a common anchor score reduces it to a shifted-kernel product in dimension $d - 1$ , using the transpose for a column anchor and the forward product for a row anchor (Figure 5). Small blocks are evaluated directly.

![](images/4f5e7756fc3d622fc00f1595d5c4f76e5a3d6e57bed81df7911f3a40657e920f.jpg)  
(b) Row anchor i<sub>0</sub>

$$
\pmb { T } ^ { \top }
$$

![](images/c6114659e0d0527dbe595acbae338c7828ed27f4090ea3e97ba3b08638377b2d.jpg)

$$
\mathbf { T }
$$

Figure 5: Two possible anchor types for an original block $S _ { I J }$ . The highlighted scores lie in $[ - L _ { \mathrm { a n c } } , 0 ]$ (a) Subtracting a column anchor’s score cancels the row bias; each kernel row then has its maximum over I, so $_ { \pmb { T } }$ has indices $J \times I$ and appears transposed. (b) Subtracting a row anchor’s score cancels the column bias; the maxima are over J and $\mathbf { T }$ keeps indices $I \times J$ . Lemma 7.4 proves both factorizations and estimates their diagonal factors.

Lemma 7.4 (Anchor factorization). An anchor block $I \times J$ can be evaluated to any error tolerance $\eta ^ { \prime } \in \mathbb { Q } \cap ( 0 , 1 ]$ by a Las Vegas algorithm in expected running time

$$
{ \widetilde { \mathrm { O } } } ( F _ { d - 1 } ( | I | , | J | ) ) .
$$

The algorithm computes exact batch maxima once in dimension $d - 1$ and uses one $\mathrm { K P } _ { d } .$ <sub>−1</sub> call.

Proof. Column anchor. Suppose $j _ { 0 } \in J$ anchors the block, so $- L _ { \mathrm { a n c } } \leq x _ { i j _ { 0 } } \leq 0$ for all $i \in I$ Subtracting its score cancels the row bias $b _ { i }$ and yields $x _ { i j } - x _ { i j _ { 0 } } = { \pmb u } _ { i } \cdot { \pmb p } _ { j } ^ { \prime } + c _ { j } ^ { \prime } ,$ where ${ \pmb p } _ { j } ^ { \prime } = { \pmb p } _ { j } - { \pmb p } _ { j _ { 0 } }$ and $c _ { j } ^ { \prime } = c _ { j } - c _ { j _ { 0 } }$ for $j \in J$ . Define

$$
M _ { j } ^ { \prime } = \operatorname* { m a x } _ { i \in I } { p _ { j } ^ { \prime } \cdot u _ { i } } , \qquad T _ { j i } = \mathrm { e } ^ { p _ { j } ^ { \prime } \cdot u _ { i } - M _ { j } ^ { \prime } } \qquad ( j \in J , \ i \in I ) .
$$

Thus $\mathbf { T }$ is the shifted kernel of Definition 5.3 in dimension $d - 1$ with rows p<sup>′</sup><sub>j</sub> $( j \in J )$ , columns ${ { \mathbf { } } { \mathbf { } } } u _ { i } \ ( i \in$ $I )$ , and shifts $M _ { j } ^ { \prime }$ . Define the vectors ${ \pmb d } _ { \mathrm { r o w } } = ( d _ { i } ^ { \mathrm { r o w } } ) _ { i \in { I } }$ and $\pmb { d } _ { \mathrm { c o l } } = ( d _ { j } ^ { \mathrm { c o l } } ) _ { j \in J }$ by

$$
d _ { i } ^ { \mathrm { r o w } } = \mathrm { e } ^ { L _ { \mathrm { a n c } } + x _ { i j _ { 0 } } } , \qquad d _ { j } ^ { \mathrm { c o l } } = \mathrm { e } ^ { c _ { j } ^ { \prime } + M _ { j } ^ { \prime } - L _ { \mathrm { a n c } } } ,
$$

so the identity $x _ { i j } = ( L _ { \mathrm { a n c } } + x _ { i j _ { 0 } } ) + ( \pmb { p } _ { j } ^ { \prime } \cdot \pmb { u } _ { i } - M _ { j } ^ { \prime } ) + ( c _ { j } ^ { \prime } + M _ { j } ^ { \prime } - L _ { \mathrm { a n c } } )$ implies $S _ { i j } = d _ { i } ^ { \mathrm { r o w } } T _ { j i } d _ { j } ^ { \mathrm { c o l } }$ , that is,

$$
\begin{array} { r } { S _ { I J } \pmb { w } _ { J } = \pmb { d } _ { \mathrm { r o w } } \odot \pmb { T } ^ { \top } ( \pmb { d } _ { \mathrm { c o l } } \odot \pmb { w } _ { J } ) . } \end{array}
$$

We have $1 \ \leq \ d _ { i } ^ { \mathrm { r o w } } \ \leq \ \mathrm { e } ^ { L _ { \mathrm { a n c } } }$ because $x _ { i j _ { 0 } } \in [ - L _ { \mathrm { a n c } } , 0 ]$ ; if $i ^ { * } \in I$ is a maximizing index for $M _ { j } ^ { \prime }$ then $c _ { j } ^ { \prime } + M _ { j } ^ { \prime } = x _ { i ^ { * } j } - x _ { i ^ { * } j _ { 0 } } \leq 0 + L _ { \mathrm { a n c } }$ , hence $0 < d _ { i } ^ { \mathrm { c o l } } \leq 1$

Row anchor. Symmetrically, for a row anchor $i _ { 0 } \in I$ , set $\pmb { u } _ { i } ^ { \prime } = \pmb { u } _ { i } - \pmb { u } _ { i _ { 0 } }$ and define the shifted kernel

$$
M _ { i } ^ { \prime } = \operatorname* { m a x } _ { j \in J } { \pmb u } _ { i } ^ { \prime } \cdot { \pmb p } _ { j } , \qquad T _ { i j } = \mathrm { e } ^ { { \pmb u } _ { i } ^ { \prime } \cdot { \pmb p } _ { j } - M _ { i } ^ { \prime } } \qquad ( i \in I , ~ j \in J ) .
$$

Set

$$
d _ { i } ^ { \mathrm { r o w } } = \mathrm { e } ^ { b _ { i } - b _ { i _ { 0 } } + M _ { i } ^ { \prime } } , \qquad d _ { j } ^ { \mathrm { c o l } } = \mathrm { e } ^ { x _ { i _ { 0 } j } } .
$$

Subtracting the anchor score cancels the column bias, so the block product equals

$$
S _ { I J } { \pmb w } _ { J } = d _ { \mathrm { r o w } } \odot { \pmb T } ( d _ { \mathrm { c o l } } \odot { \pmb w } _ { J } ) .
$$

By the same maximum comparison as above, $0 < d _ { i } ^ { \mathrm { r o w } } \leq \mathrm { e } ^ { L _ { \mathrm { a n c } } }$ , and by the anchor condition, $\mathrm { e } ^ { - L _ { \mathrm { a n c } } } \leq$ $d _ { j } ^ { \mathrm { c o l } } \leq 1$

Proposition 5.1 computes the maxima in expected time $\widetilde { \mathrm { O } } ( F _ { d - 1 } ( | I | , | J | ) )$ . We pass them to the inductive kernel algorithm; its call and the diagonal multiplications have the same expected time bound by Appendix $\mathrm { { A . 3 } }$ □

Lemma 7.5. The product of a block $I \times J$ can be approximated to any error tolerance $\eta ^ { \prime } \in$ $\mathbb { Q } \cap ( 0 , 1 ]$ deterministically in running time $\widetilde { \mathrm { O } } ( | I | | J | )$

Proof. Since every score is nonpositive, Lemma A.3 approximates each exponential with $\widetilde { \mathrm { O } } ( 1 )$ bit operations. Multiplying by the weights and summing gives the block product to error $\eta ^ { \prime }$ □

## 7.3 Random Prefix Chains

The top-level procedure Decompose processes thin nodes with full prefixes and balanced nodes with truncated prefixes. In both cases, Chain builds the prefix chain and Assign groups targets by their first hits, and in truncated mode, the final prefix’s envelope also locates the remaining targets and defines the recursive subproblems.

A node is a block with a designated sampling side. Its source and target data follow the orientation convention of Section 7.1. Conflict sets are relative to $\Sigma ,$ and the support and reporting structures are built on the fixed perturbed points $\widetilde { \mathbf { y } } _ { s }$

Procedure Chain. The input is a node with $| \Sigma | > s _ { 0 }$ , a top size $s _ { 0 } < s _ { \mathrm { t o p } } \leq | \Sigma |$ , and a reporting structure $\mathcal { R } _ { \Sigma }$ for $( \widetilde { \pmb { y } } _ { s } ) _ { s \in \Sigma }$ . The procedure runs in full mode if $s _ { \mathrm { t o p } } = | \Sigma |$ and in truncated mode otherwise.

1. Draw a uniformly random permutation of Σ and store the source positions. Let $P _ { \ell }$ be its prefix of size $s _ { \ell } ,$ , where $s _ { \ell + 1 } = \operatorname* { m i n } \{ 2 s _ { \ell } , s _ { \mathrm { t o p } } \}$ , and stop at the first $\ell _ { \mathrm { m a x } }$ with $s _ { \ell _ { \mathrm { m a x } } } = s _ { \mathrm { t o p } } .$ For $\ell < \ell _ { \mathrm { m a x } }$ , put $B _ { \ell } = P _ { \ell + 1 } \setminus P _ { \ell }$

2. For $\ell = 0 , 1 , \ldots , \ell _ { \mathrm { m a x } }$ , where the level $\ell = \ell _ { \mathrm { m a x } }$ is included only in truncated mode, perform the following steps.

(a) Build the envelope structures of $\mathcal { P } ( P _ { \ell } )$ of Lemma 6.8.

(b) For each canonical cell $\Delta$ of $\mathcal { P } ( P _ { \ell } )$ , in the order of Definition $6 . 6 ,$ use $\mathcal { R } _ { \Sigma }$ to report the d sets $\{ s \in \Sigma : \widetilde { \pmb { y } } _ { s } \cdot ( \bar { \pmb { v } } , 1 ) > \widetilde { \pmb { y } } _ { s _ { \Delta } } \cdot ( \bar { \pmb { v } } , 1 ) \}$ for the vertices $\bar { \pmb { v } } \in \mathrm { v e r t } ( \Delta )$ (Lemma $6 . 7 ( \mathrm { b } ) )$ . Let $C _ { \Delta }$ be the union of these sets, with duplicates removed. Then check the conditions

$$
| C _ { \Delta } | \leq \frac { C _ { \mathrm { s a m p } } | \Sigma | G } { s _ { \ell } } ,
$$

$$
\left. C _ { \Delta } \cap B _ { \ell } \right. \leq 1 6 C _ { \mathrm { s a m p } } G \ \mathrm { ~ i f ~ } \ell < \ell _ { \mathrm { m a x } } .
$$

At the first cell that violates (V1) or (V2), discard all structures built in steps 1 and 2 and return to step 1.

3. Return the prefixes, the batches, the envelope structures of the levels processed in step 2, and all their canonical cells with their conflict sets.

An attempt of Procedure Chain is one pass through steps 1 and 2; it is accepted when it reaches step 3. Since the source permutation determines the canonical cells, their order, and their conflict sets (Definition 6.6 and Lemma $6 . 8 ( \mathrm { a } ) )$ , acceptance with exact reporting depends only on this permutation, independently of the randomness used in geometric preprocessing.

Procedure Assign. The input is the output of an accepted attempt of Procedure Chain for a node, together with the target set Θ of the node. The mode is that of the chain.

1. For each set $\Sigma ^ { * } \in \{ P _ { 0 } , B _ { 0 } , B _ { 1 } , \dots , B _ { \ell _ { \operatorname* { m a x } } - 1 } \}$ , build the structure of Lemma 6.2 for the points $( \widetilde { \pmb { y } } _ { s } ) _ { s \in \Sigma ^ { * } }$

2. For each target $i \in \Theta$ , apply the first applicable rule.

(A1) Query the structure of $P _ { 0 }$ with the vector $( \mathbf { \boldsymbol { u } } _ { i } , 1 )$ . It returns $\operatorname* { m a x } _ { s \in P _ { 0 } } \widetilde { f } _ { s } ( { \pmb u } _ { i } )$ and a maximizing source $s _ { * } \in P _ { 0 }$ . If $b _ { i } + \widetilde { f } _ { s _ { * } } ( { \pmb u } _ { i } ) \geq - L$ , assign the target the label $( \operatorname { i n i t } , s _ { * } )$

(A2) Query the structures of $B _ { 0 } , B _ { 1 } , \ldots$ . in this order with the vector $( \mathbf { \boldsymbol { u } } _ { i } , 1 )$ , and let $\ell <$ $\ell _ { \mathrm { m a x } }$ be the least index such that the returned maximizer $s _ { * } \in B _ { \ell }$ satisfies $b _ { i } { + } \widetilde { f } _ { s _ { * } } ( { \pmb u } _ { i } ) \geq$ $- L ,$ if such an index exists. Locate $\mathbf { \delta } \mathbf { u } _ { i }$ in $\mathcal { P } ( P _ { \ell } )$ by Lemma 6.8(b), which identifies a canonical cell $\Delta$ containing ${ \bf { u } } _ { i } .$ , and assign the target the label $( \ell , \Delta , s _ { * } )$

(A3) In full mode, call the target a tail target. In truncated mode, locate $\mathbf { \delta } \mathbf { u } _ { i }$ in $\mathcal { P } ( P _ { \ell _ { \mathrm { m a x } } } )$ which identifies a canonical cell $\Delta$ containing $\mathbf { \delta } \mathbf { u } _ { i }$ , and assign the target the label $( \mathrm { f i n } , \Delta )$

3. Sort the labeled targets into groups with equal labels. The source set of a group is Σ for a

label $( \operatorname { i n i t } , s _ { * } )$ , and $C _ { \Delta }$ for labels $( \ell , \Delta , s _ { * } )$ and $( \mathrm { f i n } , \Delta )$

We represent the cell $\Delta$ in a target label by its index in the list of Lemma $6 . 8 ( \mathrm { a } )$ , where Chain has already stored its conflict set.

Level sums. For $\ell < \ell _ { \mathrm { m a x } }$ we have $s _ { \ell } = s _ { 0 } 2 ^ { \ell } < s _ { \mathrm { t o p } } \leq | \Sigma |$ , and $s _ { \ell _ { \mathrm { m a x } } } = s _ { \mathrm { t o p } }$ . Hence $2 ^ { \ell _ { \mathrm { m a x } } - 1 } <$ $| \Sigma | / s _ { 0 }$ , and since s ≥ 4, the number of levels satisfies $\ell _ { \mathrm { m a x } } + 1 < \log _ { 2 } | \Sigma | \leq \log _ { 2 } N < G$ . Since the sizes $s _ { \ell }$ with $\ell < \ell _ { \mathrm { m a x } }$ double from one level to the next and are less than $s _ { \mathrm { t o p } }$

$$
\sum _ { \ell < \ell _ { \mathrm { m a x } } } s _ { \ell } ^ { k } \le 2 s _ { \mathrm { t o p } } ^ { k } , \qquad \sum _ { \ell \le \ell _ { \mathrm { m a x } } } s _ { \ell } ^ { k } \le 3 s _ { \mathrm { t o p } } ^ { k } ,\tag{7.2}
$$

and the same bounds hold with $k - 1$ in place of k if $k \geq 2$ . For $k = 1$ the sums of $s _ { \ell } ^ { k - 1 }$ over $\ell <$ $\ell _ { \mathrm { m a x } }$ and over $\ell \leq \ell _ { \mathrm { m a x } }$ are $\ell _ { \mathrm { m a x } }$ and $\ell _ { \mathrm { m a x } } + 1$ , both less than G.

Lemma 7.6. After an accepted attempt of Procedure Chain, Procedure Assign assigns at most one label to each target. In truncated mode every target receives a label, and in full mode every target without a label is a tail target. Moreover, the following hold.

(a) For a group with label (init, s<sub>∗</sub>) or $( \ell , \Delta , s _ { * } )$ , the source s<sub>∗</sub> lies in the source set of the group, and it anchors the group’s block. For a label $( \ell , \Delta , s _ { * } )$ , moreover, $s _ { * } \in C _ { \Delta } \cap B _ { \ell }$

(b) Consider a target $i \in \Theta$ and a source $s \in \Sigma$ . If i is a tail target or belongs to a labeled group whose source set excludes s, their original pair has score below $- L + 1 / 8$

(c) The Cartesian products of each target group with its nonempty source set are pairwise disjoint subsets of the pairs of the node, and every other pair of the node has score less than $- L + 1 / 8$

Step 1 has expected running time $\widetilde { \mathrm { O } } ( s _ { \mathrm { t o p } } ^ { k } )$ ; steps 2 and 3 require $\widetilde { \mathrm { O } } ( | \Theta | )$ time.

Proof. The first applicable rule assigns each target at most one label, and (A3) assigns every remaining target in truncated mode. A support query returns a hit precisely when its queried set contains one (Lemma 6.2).

Part (a). For a group labeled (init, s<sub>∗</sub>) or $( \ell , \Delta , s _ { * } )$ , the assigned source hits every target of the group, so $b _ { i } + f _ { s _ { * } } ( { \pmb u } _ { i } ) \geq - L - 1 / 8 \geq - L _ { \mathrm { a n c } }$ by Lemma $7 . 2 ( \mathrm { a } )$ . In an initial group, $s _ { * } \in P _ { 0 } \subseteq \Sigma$ . In a later group, the preceding prefix $P _ { \ell }$ has no hit at $\mathbf { \Delta } \mathbf { u } _ { i }$ , so Lemma 7.2(c) implies $s _ { * } \in C _ { \Delta } \cap B _ { \ell }$ . Thus $s _ { * }$ is an anchor in the group’s source set.

Part (b). A tail target has no hit in $P _ { 0 } \cup B _ { 0 } \cup \ldots \cup B _ { \ell _ { \operatorname* { m a x } } - 1 } = P _ { \ell _ { \operatorname* { m a x } } } = \Sigma ,$ so all its scores are strictly below $- L + 1 / 8$ by Lemma 7.2(b). An initial label (init, s<sub>∗</sub>) includes all sources in $\Sigma .$ For a label $( \ell , \Delta , s _ { * } )$ , the proof of part (a) shows that every hit lies in $C _ { \Delta }$ . For a label $( \mathrm { f i n } , \Delta )$ rule (A3) was applied, so the target has no hit in $P _ { 0 } \cup B _ { 0 } \cup \dots \cup B _ { \ell _ { \mathrm { m a x } } - 1 } = P _ { \ell _ { \mathrm { m a x } } }$ , and $\mathbf { { \boldsymbol { u } } } _ { i } \in \Delta$ again every hit lies in $C _ { \Delta }$ by Lemma $7 . 2 ( \mathrm { c } )$ . In both cases, every source $s \notin C _ { \Delta }$ has original score below $- L + 1 / 8$ by Lemma 7.2(b).

Part (c). Every target has at most one label, so the groups have pairwise disjoint target sets, and these blocks are pairwise disjoint. Every pair outside these blocks either has a tail target or joins a labeled target to a source outside its group’s source set, so its score is less than $- L + 1 / 8$ by part (b).

Cost. Since $| P _ { 0 } | = s _ { 0 }$ and $| B _ { \ell } | \le s _ { \ell }$ , the sum of their k-th powers is $\mathrm { O } ( s _ { \mathrm { t o p } } ^ { k } )$ by (7.2), so Lemma 6.2 implies the preprocessing bound. Processing each target requires at most G support queries and one cell-location query, for a total of $\widetilde { \mathrm { O } } ( | \Theta | )$ time including the final sorting. □

For the running-time analysis, let $g _ { \mathrm { r e p } }$ be the group size of $\mathcal { R } _ { \Sigma }$ in Lemma 6.3 and set $c _ { \mathrm { r e p } } =$ $\lceil | \Sigma | / g _ { \mathrm { r e p } } \rceil + 1$ . A query returning Z then costs $\widetilde { \mathrm { O } } ( c _ { \mathrm { r e p } } + | Z | )$ time.

Lemma 7.7. Each attempt of Chain is accepted with probability greater than $3 / 4$ , independently of earlier attempts. For every fixed permutation, its expected running time, over the remaining random choices, is

$$
\widetilde { \mathrm { O } } \left( | \Sigma | + \sum _ { \ell = 0 } ^ { \ell _ { \mathrm { m a x } } } \left( c _ { \mathrm { r e p } } s _ { \ell } ^ { k } + | \Sigma | s _ { \ell } ^ { k - 1 } \right) \right) .
$$

Summing over all attempts increases the bound by at most a constant factor.

Proof. Acceptance. At each level, $P _ { \ell }$ is a uniform subset of Σ. By Lemma 6.9, the probability of violating (V1) is at most $1 / ( 8 G )$ per level and less than $1 / 8$ over the fewer than $G$ levels.

For (V2), condition on $P _ { \ell }$ and a cell $\Delta$ satisfying (V1). The batch $B _ { \ell }$ is a uniform subset of $\Sigma \setminus P _ { \ell }$ of size at most min $\{ s _ { \ell } , | \Sigma | - s _ { \ell } \}$ . Since $C _ { \Delta } \cap P _ { \ell } = \mathcal { O }$ , the random variable $X = | C _ { \Delta } \cap B _ { \ell } |$ is hypergeometric, with mean at most $2 C _ { \mathrm { s a m p } } G$ . Lemma 2.4 implies

$$
\operatorname* { P r } [ X \geq 1 6 C _ { \mathrm { s a m p } } G ] \leq ( \mathrm { e } / 8 ) ^ { 1 6 C _ { \mathrm { s a m p } } G } .
$$

There are $\mathrm { O } _ { d } ( G N ^ { k } )$ cell–level pairs, and a union bound with the sampling constants in Appendix B.5 gives probability less than $1 / 8$ for any such failure. Each attempt is therefore accepted with probability greater than $3 / 4$ , independently of earlier attempts because acceptance depends only on the newly drawn permutation.

Cost. After drawing the permutation in $\widetilde { \mathrm { O } } ( | \Sigma | )$ time, we construct the envelope at each level ℓ in expected time $\widetilde { \mathrm { O } } ( s _ { \ell } ^ { k } )$ and perform d reports for each of its $\mathrm { O } _ { d } ( s _ { \ell } ^ { k } )$ cells. The cells satisfying (V1) contribute $\widetilde { \mathrm { O } } ( c _ { \mathrm { r e p } } s _ { \ell } ^ { \bar { k } } + | \Sigma | s _ { \ell } ^ { k - 1 } )$ time at that level, including forming unions and counting batch intersections. A cell violating (V1) contributes at most an additional $\widetilde { \mathrm { O } } ( | \Sigma | )$ before the attempt stops. This bound holds for every fixed permutation and conditionally on all earlier attempts, so the constant acceptance probability increases the expected total cost by at most a constant factor.

Geometric cost of a prefix chain. After preprocessing $\mathcal { R } _ { \Sigma }$ , running Chain and Assign, including failed attempts, has expected cost

$$
\widetilde { \mathrm { O } } ( | \Theta | + | \Sigma | s _ { \mathrm { t o p } } ^ { k - 1 } + c _ { \mathrm { r e p } } s _ { \mathrm { t o p } } ^ { k } ) .\tag{7.3}
$$

This follows for both full and truncated chains by combining Lemmas 7.7 and 7.6 with (7.2), using the logarithmic number of levels when $k = 1$ . For evaluating the resulting blocks, each cell has at most logarithmically many anchors and the target groups are disjoint, so concavity relates the sum of their lower-dimensional kernel costs to the node sizes.

Exponents in the block-cost analysis. Set $k ^ { \prime } = k ( d - 1 ) , \alpha _ { d } = k ^ { \prime } / ( k ^ { \prime } + 1 )$ , and $\gamma _ { d } =$ $( k - k ^ { \prime } ) / ( k ^ { \prime } + 1 )$ , where $\alpha _ { d }$ is the exponent in the (d − 1)-dimensional bound $F _ { d - 1 }$ and $\gamma _ { d }$ governs how the total block cost at level ℓ scales with the prefix size $s _ { \ell }$

Lemma 7.8. Run Procedure Assign after an accepted attempt of Procedure Chain on a node with source set Σ and target set Θ. For every group labeled $( \mathrm { i n i t } , s _ { * } ) \ o r \ ( \ell , \Delta , s _ { * } )$ , let D be its block, with sides $\Theta _ { D }$ and $\Sigma _ { D }$ . The target sets are disjoint, so $\begin{array} { r } { \sum _ { D } \left| \Theta _ { D } \right| \leq \left| \Theta \right| . } \end{array}$ ; at most $s _ { 0 }$ blocks have an initial label $( \operatorname { i n i t } , s _ { * } )$ , and each has $\Sigma _ { D } = \Sigma$ . For each level $\ell < \ell _ { \mathrm { m a x } }$ , the collection $\mathcal { D } _ { \ell }$ of blocks labeled $( \ell , \Delta , s _ { * } )$ satisfies

$$
| \mathcal { D } _ { \ell } | = \widetilde { \boldsymbol { \mathrm { O } } } ( s _ { \ell } ^ { k } ) , \qquad \sum _ { D \in \mathcal { D } _ { \ell } } | \Sigma _ { D } | = \widetilde { \boldsymbol { \mathrm { O } } } \bigl ( | \Sigma | s _ { \ell } ^ { k - 1 } \bigr ) ,
$$

$$
\sum _ { D \in \mathcal { D } _ { \ell } } \bigl ( \bigl | \Theta _ { D } \bigr | \bigl | \Sigma _ { D } \bigr | \bigr ) ^ { \alpha _ { d } } = \widetilde { \ O } \Bigl ( \bigl ( \bigl | \Sigma \bigr | \bigr | \Theta \bigr | \bigr ) ^ { \alpha _ { d } } s _ { \ell } ^ { \gamma _ { d } } \Bigr ) .
$$

All these anchor blocks together satisfy

$$
\sum _ { D } F _ { d - 1 } ( | \Theta _ { D } | , | \Sigma _ { D } | ) = \widetilde \mathrm { O } \Big ( | \Theta | + | \Sigma | s _ { \mathrm { t o p } } ^ { k - 1 } + ( | \Theta | | \Sigma | ) ^ { \alpha _ { d } } s _ { \mathrm { t o p } } ^ { \gamma _ { d } } \Big ) .\tag{7.4}
$$

This also bounds the expected time to evaluate the blocks to tolerance $\eta _ { \mathrm { b l k } }$ and accumulate their outputs. For odd $d , \gamma _ { d } = 0$ , so the product term is independent of $s _ { \mathrm { t o p } }$ , while for even d, $\alpha _ { d } = ( k - 1 ) / k$ and $\gamma _ { d } = 1 / k$

Proof. By Lemma 7.6, the target groups are disjoint and at most $s _ { 0 }$ initial groups use the full source set Σ. At level $\ell ,$ each of the $\mathrm { O } _ { d } ( s _ { \ell } ^ { k } )$ cells has at most $\mathrm { O } _ { d } ( G )$ anchors by (V2), and each associated source set has size $\mathrm { O } _ { d } ( | \Sigma | G / s _ { \ell } )$ by (V1). The first two estimates follow by summing over these cells.

For nonempty $\mathcal { D } _ { \ell } .$ , concavity and disjointness imply

$$
\begin{array} { r l } & { \displaystyle \sum _ { D \in { \cal D } _ { \ell } } ( | \Theta _ { D } | | \Sigma _ { D } | ) ^ { \alpha _ { d } } \leq \operatorname* { m a x } _ { D \in { \cal D } _ { \ell } } | \Sigma _ { D } | ^ { \alpha _ { d } } | { \cal D } _ { \ell } | ^ { 1 - \alpha _ { d } } \left( \displaystyle \sum _ { D \in { \cal D } _ { \ell } } | \Theta _ { D } | \right) ^ { \alpha _ { d } } } \\ & { \quad \quad \quad \quad = \widetilde { \cal O } \big ( ( | \Sigma | | \Theta | ) ^ { \alpha _ { d } } s _ { \ell } ^ { k ( 1 - \alpha _ { d } ) - \alpha _ { d } } \big ) = \widetilde { \cal O } \big ( ( | \Sigma | | \Theta | ) ^ { \alpha _ { d } } s _ { \ell } ^ { \gamma d } \big ) . } \end{array}
$$

Summing over the logarithmically many levels proves $( 7 . 4 )$ , with target terms totaling at most $| \Theta |$ source terms $\widetilde { \mathrm { O } } ( | \Sigma | s _ { \mathrm { t o p } } ^ { \breve { k } - 1 } )$ , and product terms $\widetilde { \mathrm { O } } ( ( | \Theta | | \Sigma | ) ^ { \alpha _ { d } } s _ { \mathrm { t o p } } ^ { \gamma _ { d } } )$ . The initial blocks contribute at most $s _ { 0 } F _ { d - 1 } ( | \Theta | , | \Sigma | )$ , which also satisfies (7.4) because $s _ { 0 } = \tilde { \mathrm { O } } ( 1 )$ and $k - 1 , \gamma _ { d } \geq 0$

Lemma 7.4 bounds each block evaluation by its $F _ { d - 1 }$ term. Summing these conditional expectations and accumulating the outputs gives (7.4). □

## 7.4 The Full Random Prefix

FullPrefix specializes Chain and Assign to $s _ { \mathrm { t o p } } = | \Sigma |$ , then evaluates the resulting anchor groups.

Procedure FullPrefix. The input is a node with source set Σ and target set $\Theta$

1. If $| \Sigma | \le s _ { 0 }$ , evaluate the block of the node as a direct block by Lemma 7.5 with $\eta ^ { \prime } = \eta _ { \mathrm { b l k } }$ ， add the result to ${ \widehat { \mathbf { o } } } ,$ and stop.

2. Build the reporting structure $\mathcal { R } _ { \Sigma }$ with group size $g _ { \mathrm { r e p } } = | \Sigma |$

3. Run Procedure Chain with the top size $s _ { \mathrm { t o p } } = | \Sigma |$ , which is full mode, and the structure $\mathcal { R } _ { \Sigma }$ Then run Procedure Assign with the target set Θ.

4. For every group with a label $( \operatorname { i n i t } , s _ { * } )$ or $( \ell , \Delta , s _ { * } )$ , evaluate the group’s block with anchor $s _ { * }$ by Lemma 7.4 with $\eta ^ { \prime } = \eta _ { \mathrm { b l k } }$ , and add the result to ${ \widehat { \mathbf { o } } } .$

Lemma 7.9. Procedure FullPrefix has expected running time $\widetilde { \mathrm { O } } ( | \Sigma | ^ { k } + | \Theta | )$

Proof. $\mathrm { I f ~ } | \Sigma | \le s _ { 0 }$ , direct evaluation costs $\widetilde { \mathrm { O } } ( | \Theta | )$ time. Otherwise, the reporting structure is built in expected time $\widetilde { \mathrm { O } } ( | \Sigma | ^ { k } )$ and returns a report $Z$ in $\widetilde { \mathrm { O } } ( 1 + | Z | )$ time, with $c _ { \mathrm { r e p } } = 2$ because $g _ { \mathrm { r e p } } = | \Sigma |$ Substituting $s _ { \mathrm { t o p } } = | \Sigma |$ in (7.3) gives expected time $\widetilde { \mathrm { O } } ( | \Theta | + | \Sigma | ^ { k } )$ for the chain and assignment. Equation (7.4) bounds anchor evaluation by this expression as well, since

$$
( | \Theta | | \Sigma | ) ^ { \alpha _ { d } } | \Sigma | ^ { \gamma _ { d } } = | \Theta | ^ { \alpha _ { d } } ( | \Sigma | ^ { k } ) ^ { 1 / ( k ^ { \prime } + 1 ) } \leq | \Theta | + | \Sigma | ^ { k } .
$$

## 7.5 Truncated Prefixes and the Heavy/Light Recursion

At a recursive node v with block $I \times J .$ , write $m _ { v } = \operatorname* { m i n } \{ | I | , | J | \}$ and ${ n _ { v } = \operatorname* { m a x } \{ | I | , | J | \} }$ . If $m _ { v } ^ { k } \le n _ { v }$ FullPrefix with the smaller side as sources takes near-linear time. Otherwise $k \geq 2$ , and we use the larger side as sources with a truncated prefix. Define

$$
\varPhi ( m , n ) = ( m n ) ^ { k / ( k + 1 ) } , \qquad F _ { d } ( m , n ) = \varPhi ( m , n ) + m + n .
$$

For balanced sizes $m , n$ with $m ^ { k } > n$ and $n ^ { k } > m$ , we choose the prefix scale $\kappa _ { 0 }$ to balance the reporting and light-cell cost $n \kappa _ { 0 } ^ { k - 1 }$ against the heavy-child potential bound $m ( n / \kappa _ { 0 } ) ^ { ( k - 1 ) / k }$ . Setting both expressions equal to $\varPhi ( m , n )$ gives

$$
\kappa _ { 0 } = ( { m ^ { k } } / { n } ) ^ { 1 / ( k ^ { 2 } - 1 ) } , \qquad \kappa = \lceil { \kappa _ { 0 } } \rceil , \qquad n _ { \kappa } = n / \kappa , \qquad s _ { f } = s _ { 0 } \kappa .
$$

This choice balances $n \kappa _ { 0 } ^ { k - 1 } = m ( n / \kappa _ { 0 } ) ^ { ( k - 1 ) / k } = \varPhi ( m , n )$ . Rounding to κ gives

$$
\begin{array} { r l } { { 2 } \leq \kappa \leq \operatorname* { m i n } \{ 2 \kappa _ { 0 } , n \} , \quad } & { \phi ( m , n ) > \operatorname* { m a x } \{ m , n \} , } \\ { n \kappa ^ { k - 1 } \leq 2 ^ { k - 1 } \phi ( m , n ) , \quad } & { m ( n / \kappa ) ^ { ( k - 1 ) / k } \leq \phi ( m , n ) . } \end{array}\tag{7.5}
$$

If $s _ { f } \geq n$ , then $\kappa _ { 0 } > 1$ and $\kappa < 2 \kappa _ { 0 }$ give $n < 2 s _ { 0 } \kappa _ { 0 }$ . Raising this inequality to the power $k ^ { 2 } - 1$ and substituting $\kappa _ { 0 } ^ { k ^ { 2 } - 1 } = m ^ { k } / n$ yields $n ^ { k ^ { 2 } } < ( 2 s _ { 0 } ) ^ { k ^ { 2 } - 1 } m ^ { k }$ , hence

$$
n ^ { k } \leq ( 4 C _ { \mathrm { s a m p } } G ) ^ { k } m , \qquad n \leq ( 4 C _ { \mathrm { s a m p } } G ) ^ { 2 } \mathrm { w h e n } n \geq m .\tag{7.6}
$$

In the latter case, direct evaluation costs mn $\leq ( 4 C _ { \mathrm { s a m p } } G ) ^ { 2 } \varPhi ( m , n )$ . These estimates also apply to the batch-maxima algorithm in Appendix B.6.

For a final cell $\Delta .$ let $m _ { \Delta }$ and $n _ { \Delta }$ count its unassigned targets and conflicting sources. We process light cells with $m _ { \Delta } \leq { T n _ { \kappa } ^ { 1 / k } }$ by reversing the orientation and applying FullPrefix, and recurse on the heavy cells; setting $T = 4 ^ { k + 1 }$ ensures that the heavy children’s potentials sum to at most $r ^ { - 1 / ( k + 1 ) } = 1 / 4$ of their parent’s potential in the next lemma.

Lemma 7.10 (Heavy cells; proof in Appendix B.5). For balanced sizes m, n and parameters as in (7.5), let $( m _ { \Delta } , n _ { \Delta } ) _ { \Delta \in \mathcal { D } }$ be a finite list of pairs of positive integers with $\begin{array} { r } { \sum _ { \Delta \in \mathcal { D } } m _ { \Delta } \le m } \end{array}$ and $n _ { \Delta } \leq$ $n _ { \kappa } ~ f o r$ all $\Delta .$ . Call $\Delta$ heavy if $m _ { \Delta } > T n _ { \kappa } ^ { 1 / k }$ . Then

$$
\sum _ { \Delta { \mathrm { \scriptsize ~ h e a v y } } } \phi ( m _ { \Delta } , n _ { \Delta } ) \leq { \frac { 1 } { 4 } } \phi ( m , n ) , \qquad \sum _ { \Delta { \mathrm { \scriptsize ~ h e a v y } } } ( m _ { \Delta } + n _ { \Delta } ) \leq ( 1 + { \cal T } ^ { - 1 } ) \phi ( m , n ) ,
$$

and $m _ { \Delta } n _ { \Delta } \leq m n / 2$ for every $\Delta \in { \mathcal { D } }$

Procedure Decompose. The input is the block $I \times J$ at a recursive node v.

1. Let $m _ { v } = \operatorname* { m i n } \{ | I | , | J | \}$ and ${ n _ { v } = \operatorname* { m a x } \{ | I | , | J | \} }$

2. Thin node. If $m _ { v } ^ { k } \le n _ { v } .$ run FullPrefix with the smaller side as sources and stop.

3. Balanced node. Use the larger side as sources Σ and the smaller side as targets Θ. Set $\kappa = \lceil ( m _ { v } ^ { k } / n _ { v } ) ^ { 1 / ( k ^ { 2 } - 1 ) } \rceil , n _ { \kappa } = n _ { v } / \kappa ,$ and $s _ { f } = s _ { 0 } \kappa$ . If $s _ { f } \geq n _ { v } { } ;$ evaluate the block directly with tolerance $\eta _ { \mathrm { b l k } }$ by Lemma 7.5, add its output to ${ \widehat { \mathbf { o } } } ,$ and stop.

4. Build $\mathcal { R } _ { \Sigma }$ with $g _ { \mathrm { r e p } } = \kappa ,$ run Chain with $s _ { \mathrm { t o p } } = s _ { f }$ in truncated mode, and apply Assign to Θ. Evaluate the groups with assigned anchors as in step 4 of FullPrefix.

5. For every nonempty group with label $( \mathrm { f i n } , \Delta )$ , let $\Theta _ { \Delta }$ be its target set, and set $m _ { \Delta } =$

$| \Theta _ { \Delta } |$ and $n _ { \Delta } = | C _ { \Delta } |$

• If $C _ { \Delta } = \mathcal { O }$ , do nothing.

• If $C _ { \Delta } \neq \varnothing$ and $m _ { \Delta } ^ { k } \le { \cal T } ^ { k } n _ { \kappa }$ , call the cell light and reverse the parent orientation. Run FullPrefix with source set $\Sigma _ { \mathrm { n e w } } = \Theta _ { \Delta }$ and target set $\Theta _ { \mathrm { n e w } } = C _ { \Delta }$

• Otherwise, call the cell heavy and recurse on its block with Decompose.

Lemma 7.11. Procedure Decompose is a Las Vegas algorithm. Applied to $I _ { 0 } \times J _ { 0 }$ , it evaluates pairwise disjoint direct or anchor blocks, each to coordinatewise error at most $\eta _ { \mathrm { b l k } }$ . Every omitted pair has score less than $- L + 1 / 8$ , and the procedure has expected running time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$

Proof. Coverage. By Lemma 7.6, each assignment partitions its node’s pairs into anchor blocks, blocks associated with final cells, and discarded pairs with scores below $- L + 1 / 8$ . A full prefix evaluates its anchor blocks, and a truncated prefix also passes each nonempty block from a final cell to FullPrefix or recursion. Direct evaluation includes every pair in a block, and changing orientation leaves its original row–column pairs unchanged. Induction therefore proves that the evaluated blocks are disjoint and cover every pair with score at least $- L + 1 / 8$ . Their individual errors are at most $\eta _ { \mathrm { b l k } }$ by Lemmas 7.4 and 7.5.

Local cost. A thin leaf is processed in expected time $\widetilde { \mathrm { O } } ( m _ { v } + n _ { v } )$ by Lemma 7.9. If step 3 evaluates a balanced node directly, the running time is $\tilde { \mathrm { O } } ( \varPhi ( m _ { v } , n _ { v } ) )$ by (7.6).

At a node reaching step 4, the final prefix has size $s _ { f } = s _ { 0 } \kappa$ . By (V1) and $s _ { 0 } = 2 C _ { \mathrm { s a m p } } G$ , its final groups satisfy

$$
n _ { \Delta } \le C _ { \mathrm { s a m p } } n _ { v } G / s _ { f } = n _ { \kappa } / 2 , \qquad \sum _ { \Delta } m _ { \Delta } \le m _ { v } ,
$$

where the second inequality follows from disjointness of the target groups. By (7.5), we have $\kappa \leq$ $n _ { v } , n _ { v } \kappa ^ { k - 1 } = \mathrm { O } _ { d } ( \varPhi ( m _ { v } , n _ { v } ) )$ , and $m _ { v } + n _ { v } = \mathrm { O } ( \varPhi ( m _ { v } , n _ { v } ) )$ . Building the reporting structure with group size κ thus costs $\bar { \mathrm { O } } ( \varPhi ( m _ { v } , n _ { v } ) )$ in expectation. Its query parameter is $c _ { \mathrm { r e p } } = \left\lceil n _ { \kappa } \right\rceil + 1 = \mathrm { O } ( n _ { \kappa } )$ so (7.3) with $s _ { \mathrm { t o p } } = s _ { 0 } \kappa$ implies the same expected cost for the chain and assignment.

For the anchor blocks, the cost in (7.4) is

$$
\widetilde { \mathrm { O } } \bigl ( m _ { v } + n _ { v } \kappa ^ { k - 1 } + ( m _ { v } n _ { v } ) ^ { \alpha _ { d } } \kappa ^ { \gamma _ { d } } \bigr ) .
$$

For odd $d ,$ the last term is $\mathrm { O } _ { d } ( \varPhi ( m _ { v } , n _ { v } ) )$ because $k ^ { \prime } = k$ and $\gamma _ { d } = 0$ . For even d, use $k ^ { \prime } = k - 1 , \gamma _ { d } =$ $1 / k$ and $\kappa \leq 2 \kappa _ { 0 }$ to obtain

$$
( m _ { v } n _ { v } ) ^ { ( k - 1 ) / k } \kappa ^ { 1 / k } \leq 2 ^ { 1 / k } \varPhi ( m _ { v } , n _ { v } ) \left( \frac { m _ { v } } { n _ { v } ^ { k } } \right) ^ { 1 / ( k ( k ^ { 2 } - 1 ) ) } \leq 2 \varPhi ( m _ { v } , n _ { v } ) .
$$

There are $\widetilde { \mathrm { O } } ( \kappa ^ { k } )$ final cells. A nonempty light group satisfies $m _ { \Delta } ^ { k } \le { \cal T } ^ { k } n _ { \kappa }$ and $n _ { \Delta } \le n _ { \kappa } / 2$ , so its call to FullPrefix, with the parent’s targets as sources, requires expected time $\widetilde { \mathrm { O } } ( n _ { \kappa } )$ , for a total of $\widetilde { \mathrm { O } } ( n _ { v } \kappa ^ { k - 1 } )$ over all light cells. These local bounds include failed sampling attempts and hold conditionally on the input to each node.

Summing the recursion. By Lemma 7.10, the heavy children have total potential at most one quarter of their parent’s and total linear size $\mathrm { O } _ { d } ( \varPhi ( m _ { v } , n _ { v } ) )$ ; each child also satisfies $m _ { \Delta } n _ { \Delta } \leq m _ { v } n _ { v } / 2$ Summing over the recursion as in Appendix B.5 yields total expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ . □

## 7.6 Error Analysis and the Proof of Proposition 5.6

Proof of Propositions 7.1 and 5.6. Consider a $\mathrm { B P } _ { d - 1 }$ instance with the notation of this section. The algorithm fixes the perturbations and bounding halfspaces for both orientations and the

parameters (6.1) and (7.1) as in Section $7 . 1$ , sets $\widehat { \mathbf { o } } = \mathbf { 0 }$ , runs Procedure Decompose on $I _ { 0 } \times J _ { 0 }$ ， and returns the accumulated vector ${ \widehat { \mathbf { o } } } .$

Correctness. By Lemma 7.11, the evaluated blocks are disjoint and each has error at most $\eta _ { \mathrm { b l k } }$ In any row, at most n block errors contribute, and each omitted pair has weight below $\mathrm { e } ^ { - L + 1 / 8 }$ Hence

$$
\| \widehat { o } - S w \| _ { \infty } \leq n \big ( \eta _ { \mathrm { b l k } } + W \mathrm { e } ^ { - L + 1 / 8 } \big ) \leq \eta / 2
$$

by (7.1). The total error, including final rounding, is at most $\eta .$

Cost. The shared perturbations, accumulation of block outputs, and final rounding require $\widetilde { \mathrm { O } } ( N )$ time plus the linear terms in the block bounds (Appendix A.3). Adding the cost of Decompose from Lemma 7.11 gives total expected time $\tilde { \mathrm { O } } ( \dot { F _ { d } } ( m , n ) )$ □

## 7.7 Proof of Theorem 1.1

Proof of Theorem 1.1. For $d \geq 2$ , apply Theorem 5.8 to the products defined in Section 2, using $U = W = \lceil B \rceil$ and tolerance $\eta = \varepsilon / ( 8 ( B + 1 ) )$ . The returned ratios are ε-accurate by Lemma 2.5. The kernel algorithms are Las Vegas and have total expected cost

$$
r \tilde { \mathrm { O } } ( F _ { d } ( m , n ) ) ,
$$

which also includes preparing the weight vectors and computing the final ratios, at cost $r \widetilde { \mathrm { O } } ( m + n )$ For fixed $r ,$ this is $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ .

Adding a zero coordinate embeds a one-dimensional instance in the plane with identical scores. The planar algorithm therefore gives the required bound for $d = 1$ , since $F _ { 2 } ( m , n ) = \mathrm { O } ( m + n ) =$ $\mathrm { O } ( F _ { 1 } ( m , n ) )$ ). □

## 8 A Static-Query Comparison with Semigroup Halfplane Aggregation

Corollary 4.6 preprocesses fixed keys and values in $\widetilde { \mathrm { O } } ( n )$ time and space and answers each query in $\widetilde { \mathrm { O } } ( 1 )$ time. Because truncating the exponentially decaying tail reduces each query to aggregating keys whose scores are close to the row maximum, it is natural to compare this bound with exact halfplane aggregation.

The comparison model. In the semigroup arithmetic model, preprocessing stores linear forms with nonnegative integer coeficients in the point weights, which belong to a commutative semigroup. A query combines stored forms by the semigroup operation, and the scheme must work for every weight assignment and every query in the prescribed set of ranges. Space counts the stored forms, and query cost counts the distinct forms used in an answer, allowing nonnegative integer coeficients (combining ℓ distinct forms requires at least $\ell - 1$ binary semigroup operations).

Auxiliary geometric computation is free, with an adjoined identity element serving as the answer to an empty range. The lower bound applies to faithful semigroups, in which identically equal linear forms involve the same set of variables. Exact rational summation using only additions of weights fits this model.

Lemma 8.1. Let $d \geq 2$ be a fixed integer. For all large enough n, there is a labeled set $P _ { n } \subseteq$ $[ 0 , 1 ] ^ { d } ~ o f n$ points with ${ \mathrm { O } } _ { d } ( \log n ) - b i t$ rational coordinates, and a collection of halfspaces with $\mathrm { O } _ { d } ( \log n ) .$ bit integer coeficients, such that every faithful-semigroup storage scheme using $n _ { \mathrm { s t } } \geq n$ stored forms

has worst-case query cost

$$
\widetilde \Omega _ { d } \left( \frac { n ^ { d / ( d + 1 ) } } { n _ { \mathrm { s t } } ^ { 1 / ( d + 1 ) } } \right) .
$$

These finite-precision halfspaces realize every halfspace range on $P _ { n }$

Proof. Arya, Mount, and Xia [AMX12, Theorem 4.1(i)] prove this lower bound for scattered point sets, and their Lemma 4.6 shows that independent uniform points in $[ 0 , 1 ] ^ { d }$ are scattered with probability $1 - \mathrm { o } ( 1 )$ . We use such a set whose order type survives rounding to a dyadic grid with $\mathrm { O } _ { d } ( \log n )$ -bit coordinates.

Set $\delta _ { \mathrm { s e p } } = n ^ { - ( d + 3 ) }$ . A uniform point lies within distance $\delta _ { \mathrm { s e p } }$ of a fixed proper afine subspace with probability $\mathrm { O } _ { d } ( \delta _ { \mathrm { s e p } } )$ . A union bound over the successive afine heights of all ordered $( d + 1 )$ -tuples shows that, with probability $1 - \mathrm { O } _ { d } ( n ^ { - 2 } )$ , every orientation determinant has magnitude at least $\delta _ { \mathrm { s e p } } ^ { d }$ . Fix a realization P satisfying both this event and the scattered-set condition.

Round each coordinate to the nearest multiple of $2 ^ { - p }$ , with

$$
2 ^ { - p } \leq \frac { \delta _ { \mathrm { s e p } } ^ { d } } { 4 d d ! } , \qquad p = \mathrm { O } _ { d } ( \log n ) .
$$

By multilinearity, each orientation determinant changes by at most 2d d! $2 ^ { - p } \leq \delta _ { \mathrm { s e p } } ^ { d } / 2$ . Thus the rounded set $P ^ { \prime }$ has the same order type as $P ,$ , hence the same labeled halfspace ranges. Since the semigroup storage/query tradeof depends only on these ranges, the lower bound transfers to $P ^ { \prime }$

Finally, $P ^ { \prime }$ lies on a grid with denominator $2 ^ { p } = n ^ { \mathrm O _ { d } ( 1 ) }$ . Lemma A.8 in Appendix A.9 represents each range by an $\mathrm { O } _ { d } ( \log n )$ -bit integer halfspace. □

Corollary 8.2. For polynomial coordinate and value bounds, inverse-polynomial additive error tolerance, polylogarithmic coordinate encoding lengths, and a fixed number of value channels, planar attention admits a deterministic static structure with $\widetilde { \mathrm { O } } ( n )$ preprocessing time and space and $\widetilde { \mathrm { O } } ( 1 )$ query time. In contrast, a faithful-semigroup scheme for exact planar halfspace aggregation with $\widetilde { \mathrm { O } } ( n )$ stored forms has worst-case query cost

$$
{ \widetilde \Omega } ( n ^ { 1 / 3 } ) ,
$$

even for points and halfspaces whose coordinates and coeficients have $\mathrm { O } ( \log n ) { - } b i t$ encodings.

Proof. Apply Corollary 4.6 for attention and Lemma 8.1 with $d = 2$ and $n _ { \mathrm { s t } } = \widetilde { \mathrm { O } } ( n )$ for halfspaces. Both structures are preprocessed before the query arrives. □

## 9 Conclusion and Further Directions

We have shown that softmax attention in fixed dimension d can be computed at arbitrary positive temperature within the best known batch maximum-inner-product bound in m and $n$

$$
\widetilde { \mathrm { O } } \left( ( m n ) ^ { k / ( k + 1 ) } + m + n \right) , \qquad k = \lfloor d / 2 \rfloor ,
$$

and deterministically in near-linear time in the plane. Together with the dimension-preserving reduction in Section 3, this identifies the current best exponents for the two problems.

The planar data structure also separates attention from exact range aggregation. After $\widetilde { \mathrm { O } } ( n )$ preprocessing and storage, each query requires $\widetilde { \mathrm { O } } ( 1 )$ time, whereas faithful-semigroup halfplane aggregation with near-linear storage has a polynomial worst-case query lower bound (Section 8).

The central open question is whether the agreement between the current bounds reflects equality of the optimal exponents. A dimension-preserving reduction from attention to batch maximum inner product with a subpolynomial additional factor in the running time would establish that equality.

Optimal exponents. For fixed $d \geq 2$ , consider instances with $m = n$ . Let $\alpha _ { \mathrm { A t t e n t i o n } } ( d )$ be the infimum of exponents α for which attention with $r = d$ value channels can be computed by a Las Vegas algorithm in expected time $n ^ { \alpha + \mathbf { o } ( 1 ) }$ whenever coordinate and value magnitudes are polynomially bounded, $\varepsilon = n ^ { - \mathrm { O } ( 1 ) }$ , and encoding lengths are polylogarithmic in $n .$ . The algorithm may depend on the fixed exponents in these bounds. Define $\alpha _ { \mathrm { B a t c h M a x I P } } ( d )$ analogously for BatchMaxI $\mathrm { P } _ { d }$ with polynomially bounded integer coordinates. Our reduction and upper bound imply

$$
\alpha _ { \mathrm { B a t c h M a x I P } } ( d ) \leq \alpha _ { \mathrm { A t t e n t i o n } } ( d ) \leq 2 - { \frac { 2 } { \lfloor d / 2 \rfloor + 1 } } ,
$$

with equality to 1 for $d = 2 , 3 .$

Conjecture 9.1 (Dimensional complexity of attention). For every fixed integer $d \geq 4$

$$
\alpha _ { \mathrm { A t t e n t i o n } } ( d ) \leq \alpha _ { \mathrm { B a t c h M a x I P } } ( d ) ,
$$

and hence $\alpha _ { \mathrm { A t t e n t i o n } } ( d ) = \alpha _ { \mathrm { B a t c h M a x I P } } ( d )$

Such a black-box reduction would transfer future improvements for batch maximum inner product directly to attention, but constructing it for $d \geq 4$ remains open. For lower bounds, the known unconditional results of the form $n ^ { 1 + \Omega \bar { ( } 1 ) }$ apply to restricted models, including Erickson’s polyhedral partitioning algorithms [Eri96] (Section 1.3).

Precision dependence and determinism. The factors suppressed by soft-O may be reduced through faster integer arithmetic, improved polynomial bases or interpolation nodes, and techniques that remove logarithmic losses in related geometric problems [Cha12; CZ22]. To derandomize the algorithms, we need deterministic replacements for the random prefixes, the source order in the batch-maxima procedure ThinMax (Appendix B.6), and the geometric preprocessing. Deterministic constructions of cuttings and convex hulls [Mat92a; Cha93] ofer starting points for these replacements.

Ofline aggregation. For ofline halfspace aggregation, all queries are given in advance. A faithful-semigroup lower bound with exponent above $2 - 2 / ( \lfloor d / 2 \rfloor + 1 )$ would establish a batch analogue of the comparison in Section 8. For simplex ranges, Chazelle [Cha97] proved an ofline semigroup lower bound of $\widetilde \Omega ( n ^ { 2 - 2 / ( d + 1 ) } )$ ) for n points and n simplices.

Implications for AI. Our near-linear algorithms in dimensions two and three motivate architectures with many micro-heads, each with small query and key dimensions and few value channels. By distributing representation across these heads, such architectures could use the algorithms to process longer contexts, with exact factorizations of small fixed rank ofering a related route to low-dimensional scores [GHSXY26]. For a fixed key–value cache, the planar structure answers queries in $\widetilde { \mathrm { O } } ( 1 )$ time from stored aggregate moments (Corollary 4.6), suggesting that an extension to growing caches could reduce the repeated reads of keys and values that limit autoregressive decoding bandwidth [Sha19]. Developing these applications calls for studying the modeling capacity of many small heads, reducing the arithmetic cost, and updating the cached geometric information eficiently, and could inspire new designs for both AI models and hardware architectures.

## A Numerical Stability and Precision

This appendix verifies that all intermediate quantities appearing in our geometric primitives, polynomial evaluations, and dimension reductions have bit lengths polynomial in the instance parameter λ of Section 2.1. The table locates the algorithms and their arithmetic analyses.
<table><tr><td>Component</td><td>Algorithm and correctness</td><td>Bit complexity</td></tr><tr><td>Planar kernel products and Section 4 online queries</td><td></td><td>Appendix A.2</td></tr><tr><td>Anchor blocks</td><td>Section 7.2</td><td>Appendix A.3</td></tr><tr><td>Temperature interpolation</td><td>Section 5.3</td><td>Appendix A.4</td></tr><tr><td>Ray shooting</td><td>Appendix B.2</td><td>Appendix A.5</td></tr><tr><td>Support and reporting</td><td>Appendices A.7 and B.3</td><td>Appendix A.6</td></tr><tr><td>Envelope cells and location Appendix B.4</td><td></td><td>Appendices A.7 and A.8</td></tr><tr><td>Batch Max-IP</td><td>Appendix B.6</td><td>Appendices A.6-A.8</td></tr></table>

## A.1 Input Size, Temperature, and Reduction Cost

Reading and rounding a rational coordinate to a prescribed dyadic precision requires time polynomial in its encoding length and the number of precision bits. For an irrational inverse temperature, use a positive dyadic $\widetilde { \beta }$ with $| \widetilde { \beta } - \beta | \le \operatorname* { m i n } \{ \beta / 2 , \varepsilon / ( 4 d B ^ { 3 } ) \}$ . The scores change by at most $\varepsilon / ( 4 B )$ , so the output changes by at most $\varepsilon / 2$ by Lemma 2.5, leaving tolerance $\varepsilon / 2$ for the attention algorithm. This approximation needs $\mathrm { O } _ { d } ( \log ( 1 / \varepsilon ) + \log ( B + 2 ) + | \log \beta | + 1 )$ bits. For $\beta = d ^ { - 1 / 2 }$ , it is computable within the same polynomial bit cost. Thus temperature scaling and input preparation require $\widetilde { \mathrm { O } } ( 1 )$ time per entry.

The reductions in Section 3 consist of scaling input coordinates, computing inner products of length $d ,$ and rounding. The resulting numbers have magnitudes polynomial in $m + n , d , B$ and denominators of logarithmic encoding length. Consequently all these numbers have polynomially bounded encoding lengths in log $( m + n + d + B )$ , and the additional work is $\widetilde { \mathrm { O } } ( ( m + n ) d )$ . In particular, this remains $n ^ { 1 + \mathrm { o } ( 1 ) }$ when $m = n , d = n ^ { \mathrm { o ( 1 ) } }$ , and $B = n ^ { \mathrm { O ( 1 ) } }$

## A.2 Arithmetic for Planar Kernel Products

For each call to Lemma 4.2, use the instance parameter λ of Section 2.1.

Geometric rounding. For the rowwise tolerance $\eta _ { \mathrm { r o w } }$ of Section 4.2, choose dyadic parameters

$$
\delta = \Theta \left( \frac { \eta _ { \mathrm { r o w } } } { N ( U + 1 ) } \right) , \qquad \eta _ { \mathrm { e n t } } = \Theta ( \eta _ { \mathrm { r o w } } / N )
$$

with small absolute constants satisfying the error estimates below. Both parameters have $\mathrm { O } ( \lambda )$ bits. Lemma A.1. Let $\widetilde { q } _ { i } , \widetilde { k } _ { j }$ be obtained by rounding each coordinate of the original $\mathbf { \Delta } q _ { i } , k _ { j }$ to δZ with error at most δ, and set

$$
\widetilde { h } _ { i } = \operatorname* { m a x } _ { j \in [ n ] } \widetilde { \pmb { q } } _ { i } \cdot \widetilde { \pmb { k } } _ { j } .
$$

For every $i \in [ m ]$ , the original kernel T satisfies

$$
\sum _ { j = 1 } ^ { n } \left| T _ { i j } - \mathrm { e } ^ { \widetilde { q } _ { i } \cdot \widetilde { k } _ { j } - \widetilde { h } _ { i } } \right| \leq \frac { \eta _ { \mathrm { r o w } } } { 1 6 } .
$$

Proof. Rounding changes each inner product by $\mathrm { O } ( ( U + 1 ) \delta )$ . Lemma 2.5 bounds the change in each shifted exponential by $\mathrm { O } ( ( U + 1 ) \delta )$ as well, giving row error $\mathrm { O } ( N ( U + 1 ) \delta ) \le \eta _ { \mathrm { r o w } } / 1 6$ for a small enough constant in $\delta$ □

Collinear keys. For collinear rounded keys, translate one key to the origin and write $k _ { j } = t _ { j } { \pmb u }$ with $\| \pmb { u } \| _ { \infty } = 1$ , using $t _ { j }$ from a coordinate of maximum spread or setting every $t _ { j } = 0$ when all keys coincide. For query i, let $\alpha _ { i } = { \pmb q } _ { i } \cdot { \pmb u }$ and $h _ { i } = \operatorname* { m a x } _ { j } \alpha _ { i } t _ { j }$ . In the order of increasing $t _ { j }$ , the selected range $\{ j : \alpha _ { i } t _ { j } \geq h _ { i } - L \}$ is a prefix, sufix, or the whole array. On this range, $s _ { i j } = \alpha _ { i } t _ { j } - h _ { i } + 3 L \in [ 2 L , 3 L ]$ , so expanding the polynomial below in powers of $t _ { j }$ reduces forward products and online queries to prefix sums of $t _ { j } ^ { \ell } w _ { j }$ for $0 \leq \ell \leq g _ { \mathrm { : } }$ , and transpose products to interval additions of weighted query coeficients. Every omitted shifted score is below $- L ,$ , as required for the error estimates of Section 4.5.

Taylor approximation. Write $\begin{array} { r } { \exp _ { a } ( s ) = \sum _ { \ell = 0 } ^ { a } s ^ { \ell } / \ell ! } \end{array}$

Lemma A.2. For $H \geq 0 , s \in [ 0 , H ]$ and an integer $a \geq 0$

$$
\exp _ { a } ( s ) \geq 1 , \qquad 0 \leq \mathrm { e } ^ { s } - \exp _ { a } ( s ) \leq \mathrm { e } ^ { H } \left( { \frac { \mathrm { e } H } { a + 1 } } \right) ^ { a + 1 } .
$$

In particular, with the parameters of Section $4 . 2 ,$

$$
\mathrm { e } ^ { - 3 L } ( \mathrm { e } ^ { s } - \mathrm { e x p } _ { g } ( s ) ) \leq \eta _ { \mathrm { e n t } } \qquad ( 0 \leq s \leq 3 L ) .
$$

Proof. The Taylor terms are nonnegative. The standard remainder formula [DLMF, Eq. (1.4.36)], together with $( a + 1 ) ! \geq ( ( a + 1 ) / \mathrm { e } ) ^ { a + 1 }$ , implies the first bound. For a large enough constant in $g = \Theta ( L )$ , the rescaled remainder is at most $2 ^ { - g } \leq \eta _ { \mathrm { e n t } }$ □

Scalar exponentials.

Lemma A.3. For $\lambda \geq 1$ , let $x \in \mathbb { Q }$ and $\delta _ { x } \in \mathbb { Q } \cap ( 0 , 1 ]$ have encoding lengths $\mathrm { O } ( \lambda )$ , and suppose max $\{ x , 0 \} = \operatorname { O } ( \lambda )$ . In poly(λ) time, one can compute a nonnegative dyadic ω such that $0 \leq \mathrm { e } ^ { x } - \omega \leq \delta _ { x }$ and its numerator and denominator have $\mathrm { O } ( \lambda )$ bits.

Proof. If $x \leq - \lceil \log _ { 2 } ( 8 / \delta _ { x } ) \rceil$ , then $\mathrm { e } ^ { x } \ \leq \ \delta _ { x } / 8$ and we return $\omega = 0$ Otherwise $| x | =  { \operatorname { O } ( \lambda ) }$ so evaluating ${ \mathrm { O } } ( \lambda )$ Taylor terms by Lemma A.2 produces an interval $[ A , A + \rho ]$ containing $\mathrm { e } ^ { | x | }$ with $A \geq 1$ and $\rho \leq \delta _ { x } / 8$ . For $x < 0$ , the reciprocal endpoints enclose $\mathrm { e } ^ { x }$ in an interval of width at most $\rho ,$ since $A \geq 1$ . Rounding the lower endpoint down to a dyadic grid of mesh $\Theta ( \delta _ { x } ) \leq \delta _ { x } / 2$ yields a nonnegative ω satisfying $0 \leq \mathrm { e } ^ { x } - \omega \leq \rho + \delta _ { x } / 2 \leq 5 \delta _ { x } / 8 < \delta _ { x }$ . The Taylor sums and endpoints have polynomial bit length, and the computation requires poly(λ) time. The dyadic output has $\mathrm { O } ( \lambda ) { - } \mathrm { b i t }$ numerator and denominator because log $( 1 / \delta _ { x } ) = \mathrm { O } ( \lambda )$ and $\operatorname* { m a x } \{ x , 0 \} = \mathrm { O } ( \lambda )$ □

For the planar kernel, $\mathrm { e } ^ { - 3 L } \geq 2 ^ { - 5 L }$ , so an absolute tolerance of $\eta _ { \mathrm { e n t } } 2 ^ { - 5 L }$ ensures the relative error for $\omega$ in Section 4.4, with $\omega > 0$ for a small enough constant in $\eta _ { \mathrm { e n t } }$

Rowwise kernel error. Every omitted rounded exponential weight is less than $\mathrm { e } ^ { - L } \leq \eta _ { \mathrm { e n t } }$ . For an entry with $j \in \mathcal { K } _ { i }$ , the polynomial error is at most $\eta _ { \mathrm { e n t } }$ by Lemma $\mathrm { A . 2 }$ , and the approximation to the common factor contributes at most

$$
( \mathrm { e } ^ { - 3 L } - \omega ) \mathrm { e x p } _ { g } ( s _ { i j } ) \leq \eta _ { \mathrm { e n t } } \mathrm { e } ^ { - 3 L } \mathrm { e } ^ { 3 L } = \eta _ { \mathrm { e n t } } .
$$

Thus the row error relative to the rounded exponential weights is at most $3 n \eta _ { \mathrm { e n t } } \leq 3 \eta _ { \mathrm { r o w } } / 1 6$ Centering leaves the rounded shifted scores unchanged, so the triangle inequality and Lemma A.1 imply the error estimate relative to the original T.

Product error. For either product, set

$$
\eta _ { \mathrm { r o w } } = \frac { \eta } { 8 N ( W + 1 ) } .
$$

We round the input weights to a working vector w or z on a common dyadic grid with coordinatewise error at most $\eta / ( 8 N )$ , so the rounded entries have magnitude at most $W + 1$ . The corresponding sweep then evaluates Pw or $P ^ { \top } z$ exactly, and we round each output coordinate to within $\eta / 8$

By Lemma 4.5, the rowwise kernel error is at most $\eta _ { \mathrm { r o w } }$ , whose contribution for the original input vector satisfies

$$
| ( ( T - P ) w ^ { \mathrm { i n } } ) _ { i } | \leq W \eta _ { \mathrm { r o w } } \leq \frac { \eta } { 8 } , \qquad | ( ( T - P ) ^ { \top } z ^ { \mathrm { i n } } ) _ { j } | \leq m W \eta _ { \mathrm { r o w } } \leq \frac { \eta } { 8 } .
$$

The extra factor m in the transpose bound comes from summing one entrywise error from each row. Since $0 \leq P _ { i j } \leq 1$ , the subsequent rounding of the input vector contributes at most $n \eta / ( 8 N )$ for the forward product and $m \eta / ( 8 N )$ for the transpose product. Each is at most $\eta / 8 ,$ so the total error including output rounding is at most $3 \eta / 8 < \eta$

Attention and online queries. For planar attention, set $W = \lceil B \rceil$ and $\eta = \varepsilon / ( 8 ( B + 1 ) )$ in the kernel calls. By Lemma 2.5, the ratio error is at most $( B + 1 ) \eta / ( 1 - \eta ) < \varepsilon / 4$ , so the total remains below ε after output rounding to error $\varepsilon / 1 6$

For the static structure, use $N = n + 1$ and $\eta _ { \mathrm { r o w } } = \varepsilon / ( 1 6 ( W + 1 ) )$ . Round the keys and each arriving query with the geometric mesh $\delta _ { : }$ , and round the values with error at most $\varepsilon / 1 6$ . Let t be the original shifted-kernel row and $\mathbfit { p }$ its row in P. Then $\| \pmb { t } - \pmb { p } \| _ { 1 } \le \eta _ { \mathrm { r o w } }$ and $\begin{array} { r } { \sum _ { j } p _ { j } \ge 1 - \eta _ { \mathrm { r o w } } > 0 } \end{array}$ The ratio error for an original value channel of magnitude at most W is at most

$$
\frac { 2 W \eta _ { \mathrm { r o w } } } { 1 - \eta _ { \mathrm { r o w } } } + \frac { \varepsilon } { 1 6 } .
$$

The second term is the value-rounding error under the nonnegative normalized weights. Rounding the output to error $\varepsilon / 1 6$ keeps the total below ε.

Exact moment sums. The geometric and weight grids have common dyadic denominators $\mathrm { O } ( \lambda )$ bits, and the polynomial degree is $g = \mathrm { O } ( \log ( N W / \eta ) ) = \mathrm { O } ( \lambda )$ . Before centering, scale the rounded keys of Lemma A.1 to integer coordinates. Their polygon’s centroid is a ratio of sums of fixed-degree products of these coordinates, so the shoelace formula expresses all centered keys with polynomial bit length and a common denominator. The collinear case has the same property, since its parameters are diferences of grid coordinates and one shared direction vector.

Because the weighted key monomials and row coeficients involve powers through degree g and Taylor factorials, their numerators and common denominators have polynomial bit length. Summing at most N signed terms over these denominators adds only $\mathrm { O } ( \log N )$ numerator bits, so the sums and the polynomially many terms in each weighted sum have bit lengths polynomial in $\lambda .$ The remaining geometric computations have the same bound because they consist of fixed-degree arithmetic and fixed-dimensional linear systems.

Each sweep requires $\widetilde { \mathrm { O } } ( N )$ bit operations and storage, as required by Lemma 4.2, including product rounding and attention divisions by denominators bounded away from zero. For the static structure, persistence stores $\widetilde { \mathrm { O } } ( n )$ nodes, and a query visits $\widetilde { \mathrm { O } } ( 1 )$ nodes to combine the moments with polynomially many coeficients. The common-denominator bounds hold uniformly over queries satisfying the input bounds, proving Corollary 4.6, and the query workspace can be reused throughout the stream.

## A.3 Arithmetic for Biased Block Products

We now establish the bit-complexity bounds suppressed by $\widetilde { \mathrm { ~ O ~ } }$ in Lemmas 7.4 and 7.5. All blocks belong to one original biased-product instance and share its fixed window L and tolerance $\eta _ { \mathrm { b l k } }$ . All precision bounds refer to the parameter λ of this original instance (Section 2.1).

Anchor products. The transformed coordinates in Lemma 7.4 are input coordinates or their diferences. Hence the exact inner-product maxima and the rational exponents defining the diagonal factors have bit length polynomial in λ. The factors are bounded by $ { \mathrm { e } } ^ { \mathrm { O } ( L ) }$ , so we can approximate them by Lemma A.3 with a number of bits polynomial in λ and the target precision. For a block with $N _ { I J } = \vert I \vert + \vert J \vert$ , approximate the factors and kernel product to a common dyadic error δ. Including final rounding, the total error is $\mathrm { O } ( \mathrm { e } ^ { 2 L _ { \mathrm { a n c } } } N _ { I J } ( W + 1 ) \delta )$ . It sufices to use

$$
\mathrm { O } _ { d } \big ( L + \log N _ { I J } + \log W + \log ( 1 / \eta ^ { \prime } ) \big )
$$

additional bits to ensure block error at most $\eta ^ { \prime } .$ The scaled weights have magnitude at most $\mathrm { e } ^ { \mathrm { O } ( L ) } W$ and polynomial bit length, so the lower-dimensional kernel call has only a polynomial increase in its logarithmic parameters. The remaining scalar computations require $\widetilde { \mathrm { O } } ( | I | + | J | )$ time.

Direct products and accumulation. For a direct block, approximate each exponential and round each weighted term to a single dyadic grid before summing. The common denominator keeps intermediate bit lengths polynomial, and a suitable grid mesh ensures the error bound of Lemma 7.5 in ${ \widetilde { \mathrm { O } } } ( | I | | J | )$ time. In both kinds of blocks, round the block outputs to one grid shared by the whole decomposition, with mesh proportional to $\eta _ { \mathrm { b l k } }$ as in Section 2.1. Each output row receives at most n block contributions, so the common denominator and all intermediate numerators have polynomial bit length in the original λ.

Group indices and accumulation. The labels distinguishing initial, first-hit, and final groups contain only a constant number of source, cell, and level indices. Label comparisons and the computation of sampling parameters involve numbers of polynomial bit length in λ. Their running time, together with accumulation and final rounding, is bounded by the linear terms in the block and geometric estimates.

## A.4 Precision under Composition and Recursion

Here λ is the parameter of Section 2.1 for the current kernel-product or biased-product instance.

Interpolation coeficients and sums. In Proposition 5.5, round each exact coeficient $\mathcal { L } _ { i \ell }$ to a single dyadic mesh $\delta _ { c } = \Theta ( \eta / ( N W n _ { \mathrm { i n t } } ) )$ , with a small absolute constant, and denote the result by $\widehat { \mathcal { L } } _ { i \ell }$ . Then

$$
| \widehat { \mathcal { L } } _ { i \ell } - \mathcal { L } _ { i \ell } | \leq \delta _ { c } , \qquad \sum _ { \ell } | \widehat { \mathcal { L } } _ { i \ell } | \leq 9 ^ { n _ { \mathrm { i n t } } } .
$$

Before interpolation, round the original weights toward zero to a common dyadic grid of mesh $\mathrm { O } ( \eta / N )$ . The rounded weights have magnitude at most $W$ and change the exact kernel product by $\mathrm { O } ( \eta )$ , since kernel entries are at most one. We apply the interpolation scheme to these rounded weights, using the rounded coeficients in both the forward output combination and the transpose input weights. Because each entry of the biased matrix lies in (0, 1], coeficient rounding adds at most $N W n _ { \mathrm { i n t } } \delta _ { c }$ to either product, and the bound on the sum of absolute coeficients ensures the estimates for errors from the biased products in the main proof. With suficiently small grid constants and final rounding error, these additional errors sum to at most $\eta / 2$

One reduction step. The normalized directions and exact maxima have encoding lengths $\mathrm { O } _ { d } ( \lambda )$ since normalization uses a fixed number of rational operations per coordinate and each maximum equals an input inner product. Each interpolation coeficient is a product of polynomially many rationals of polynomial encoding length, and the number of nodes and groups is polynomial in λ. The coeficients therefore have polynomial encoding length, and their absolute sum requires only polynomially many additional precision bits in the biased calls. The transformed coordinates, weights, and error tolerances have the same polynomial bound. With weights and outputs on common dyadic grids, summing contributions adds logarithmically many bits in the number of terms, establishing the numerical cost in Proposition 5.5. These estimates also apply to the diagonal factors of anchor blocks, whose exponents are bounded above by the score window and whose entries below the target tolerance can be rounded to zero (Appendix A.3).

Recursion at a fixed dimension. The shared parameters of Section 7.1 and the exact geometry in Appendix A.5 ensure polynomial operand lengths at every node of one biased-product instance.

Recursion on the dimension. Each lower-dimensional call decreases the dimension and may replace λ by a fixed polynomial in λ. There are at most $d - 2$ reductions before the planar base case, so their composition is a polynomial whose degree depends only on the fixed dimension. The numerical factors accumulated along this constant-length chain have the same property. Hence the hidden factors remain polynomial in the original encoding lengths and logarithmic parameters.

Computing attention. For Theorem 1.1, the tolerance $\eta = \varepsilon / ( 8 ( B + 1 ) )$ adds ${ \mathrm { O } } ( \log ( B + 1 ) )$ precision bits. The kernel outputs have polynomial encoding length, so each division and output rounding requires time polynomial in the original numerical parameters.

## A.5 Exact Geometric Computation

For fixed $d \geq 2$ , we use the notation of Section 6 and the instance parameter λ of Section 2.1.

Lemma A.4. Suppose a geometric algorithm performs T arithmetic or combinatorial operations and stores S objects. If each geometric quantity is obtained from ${ \mathrm O } _ { d } ( 1 )$ input points or constraints by a bounded-depth, bounded-degree algebraic computation, and each combinatorial operation acts on encodings of bit length polynomial in λ, then it has an exact implementation with running time $\widetilde { \mathrm { O } } ( T )$ and space $\widetilde { \mathrm { O } } ( S )$ . The implementation has the same expected bounds and is deterministic or Las Vegas according to the original algorithm.

Proof. The fixed-dimensional linear-algebra computations involve rationals of $\mathrm { O } _ { d } ( \lambda )$ bits. More generally, bounded-degree algebraic numbers obtained at bounded depth have defining coeficients of $\mathrm { O } _ { d } ( \lambda )$ bits, and exact sign tests, root isolation, and comparisons require time polynomial in λ [BPR06, Chapters 8 and 10]. Applying these bounds to each operation proves the claim, also after taking expectations. □

Hull vertices and vertices in the low-dimensional hierarchies are intersections of original constraints [PS85; DK90; CS89], as are the vertices of the first triangulation in Appendix B.2. Its local arrangement vertices are intersections of original hyperplanes and facets of a first-stage simplex, using two fixed levels of linear algebra. The fixed-dimensional linear-programming predicates [Meg84, Sections 2 and 3.2] and auxiliary algebraic decompositions also have a fixed number of elimination levels. Since recursive nodes reuse subsets of the original constraints and the same bounding simplex, the same depth bounds hold at every recursive node and the bit lengths remain $\mathrm { O } _ { d } ( \lambda )$ .

Distinct rational values obtained by a bounded number of these operations difer by at least $2 ^ { - \mathrm { O } _ { d } ( \lambda ) }$ . If each value changes by less than one quarter of this gap, every strict comparison has the same outcome and every perturbed minimizer or maximizer remains valid for the original values. This is a standard use of exact geometric computation [Yap97].

## A.6 Uniform Perturbations and Caps

Lemma A.5. Let $\lambda \geq 1$ be an integer and let $( y _ { h } ) _ { h \in \mathcal { T } }$ be points in $\mathbb { Q } ^ { d }$ of bit length at most $\lambda _ { i }$ indexed by distinct positive integers at most $2 ^ { \lambda }$ . For a large enough positive integer $c _ { \mathrm { p e r t } }$ depending only on $d ,$ set $\delta _ { \mathrm { p e r t } } = 2 ^ { - c _ { \mathrm { p e r t } } \lambda }$ and

$$
\widetilde { \pmb { y } } _ { h } = \pmb { y } _ { h } + \big ( \delta _ { \mathrm { p e r t } } h , ( \delta _ { \mathrm { p e r t } } h ) ^ { 2 } , \dots , ( \delta _ { \mathrm { p e r t } } h ) ^ { d } \big ) .
$$

(a) The perturbed points have $\mathrm { O } _ { d } ( \lambda )$ bits and satisfy $\| \widetilde { \pmb { y } } _ { h } - \pmb { y } _ { h } \| _ { \infty } \le 2 ^ { - ( c _ { \mathrm { p e r t } } - 1 ) \lambda }$

(b) In every signed coordinate permutation π, the hyperplanes $z = f _ { \tilde { y } _ { h } } ^ { \pi } ( { \pmb u } )$ are in general position.

(c) Every intersection of d of these hyperplanes has $ \mathrm { O } _ { d } ( \lambda ) { - } b i t$ coordinates.

The constant $c _ { \mathrm { p e r t } }$ may be increased to satisfy any fixed exponential separation bound of the form $2 ^ { - \mathrm { O } _ { d } ( \lambda ) }$

Proof. For at most d normals, test a square minor containing the constant coordinate and distinct remaining coordinates; for $d + 1$ hyperplanes, test their augmented determinant. As a polynomial in $\delta _ { \mathrm { p e r t } }$ , the determinant has a highest-degree coeficient equal to a generalized Vandermonde determinant $( h ^ { \ell } )$ with distinct positive indices and distinct nonnegative exponents. Descartes’ rule of signs implies that this coeficient is nonzero [DLMF, Section 1.11(ii)]. Clearing denominators gives a nonzero integer polynomial of bounded degree with $\mathrm { O } _ { d } ( \lambda ) { \mathrm { - b i t } }$ coeficients. For suficiently small $\delta _ { \mathrm { p e r t } } = 2 ^ { - c _ { \mathrm { p e r t } } \lambda }$ , its lowest-degree nonzero term dominates the rest. The coeficient bounds are uniform, so one value of $\delta _ { \mathrm { p e r t } }$ works for every index subset and signed coordinate permutation. The perturbation estimate follows from the definition, and the intersection coordinates have $\mathrm { O } _ { d } ( \lambda )$ bits by Lemma A.4. □

We close the region above an envelope with explicit cap halfspaces.

Definition A.6. Let $U _ { Y } , U _ { u } \ge 1$ be integers and set

$$
a _ { \mathrm { c a p } } = d ^ { 2 } ( U _ { Y } + 2 ) , \qquad Z _ { \mathrm { t o p } } = d ( U _ { Y } + 2 ) ( U _ { u } + 1 ) , \qquad Z _ { \mathrm { c a p } } = Z _ { \mathrm { t o p } } + 1 + ( d - 1 ) a _ { \mathrm { c a p } } U _ { u } .
$$

Let $e _ { h }$ be the h-th unit vector of $\mathbb { R } ^ { d - 1 }$ for $h < d ,$ and let $\pmb { e _ { d } } = - ( 1 , \dots , 1 )$ . For a signed coordinate permutation $\pi ,$ set

$$
\pmb { c } _ { h } ^ { \pi } = \pi ^ { - 1 } ( - a _ { \mathrm { c a p } } \pmb { e } _ { h } , Z _ { \mathrm { c a p } } ) \qquad ( h \in [ d ] ) .
$$

For a nonempty finite set Y of points in R<sup>d</sup> with $\| \pmb { y } \| _ { \infty } \le U _ { Y } + 1$ and cap perturbations satisfying $\| \widetilde { \boldsymbol { c } } _ { h } - \boldsymbol { c } _ { h } ^ { \pi } \| _ { \infty } \leq 1 / ( 8 d ( U _ { u } + 1 ) )$ , write $g _ { h } ( \pmb { u } ) = ( \pi \widetilde { \pmb { c } } _ { h } ) \cdot ( \pmb { u } , 1 )$ and define

$$
\begin{array} { r } { P ^ { \pi } ( \mathcal { V } ; \tilde { c } ) = \{ ( \boldsymbol { u } , \boldsymbol { z } ) \in \mathbb { R } ^ { d - 1 } \times \mathbb { R } : f _ { \boldsymbol { y } } ^ { \pi } ( \boldsymbol { u } ) \leq \boldsymbol { z } ( \boldsymbol { y } \in \mathcal { V } ) , \boldsymbol { z } \leq g _ { h } ( \boldsymbol { u } ) ( h \in [ d ] ) \} . } \end{array}
$$

The lower constraints are the source constraints, the upper ones are the cap constraints, and ${ \pmb x } _ { \mathrm { t o p } } = ( { \bf 0 } , Z _ { \mathrm { t o p } } )$

Lemma A.7. The capped polytope $P = P ^ { \pi } ( \mathcal { V } ; \widetilde { c } )$ satisfies:

(a) $( { \pmb u } , Z _ { \mathrm { t o p } } ) \in$ int P whenever $\| \mathbf { \boldsymbol { u } } \| _ { \infty } \leq U _ { u }$ ;

(b) P is bounded;

(c) a vertical downward ray from an interior point $( u , z _ { 0 } )$ exits at $z = \operatorname* { m a x } _ { y \in \mathcal { Y } } f _ { y } ^ { \pi } ( u )$ , and every exit constraint with positive directional inner product is the constraint of a maximizing source.

Proof. On the target box, the source functions lie below $Z _ { \mathrm { t o p } } ,$ while every perturbed cap lies above it with a fixed positive margin. For boundedness, note that

$$
\operatorname* { m a x } _ { h } { e _ { h } } \cdot { \pmb u } \ge \| { \pmb u } \| _ { \infty } / ( d - 1 ) .
$$

For the chosen $a _ { \mathrm { c a p } }$ , the minimum cap lies below every source function whenever $\| \mathbf { \boldsymbol { u } } \| _ { \infty }$ is suficiently large, uniformly over the allowed cap perturbations. Their intersection is therefore bounded in both u and z. A downward ray exits through a source constraint at the largest source value. □

## A.7 Support Queries and Envelope Construction

Proof of Lemma 6.2. For each normalizing permutation, we form the cap halfspaces of Definition A.6 with $U _ { Y } = 2 ^ { \lambda }$ and $U _ { u } = 1$ , indexing the caps by $1 , \ldots , d$ and the sources by distinct subsequent integers. Applying Lemma A.5 with the same source indexing and a uniform $\mathrm { O } _ { d } ( \lambda )$ bit bound puts the combined constraints in general position. The caps together with any one source define the bounding simplex for Proposition B.3, whose structure we build using the interior point ${ \pmb x } _ { \mathrm { t o p } }$ from Lemma $\mathrm { A . 7 ( a ) }$

For a nonzero query ${ \mathbf { } } q ,$ use its permutation $\pi _ { q }$ from Definition 5.2 and shoot vertically downward from $( \boldsymbol { u } _ { q } , Z _ { \mathrm { t o p } } )$ . Lemma A.7 gives a maximizer of the perturbed scores. Distinct original scores have gaps at least $2 ^ { - \mathrm { O } _ { d } ( \lambda ) }$ , and the perturbation changes each score by less than one quarter of this gap uniformly over queries satisfying the input bounds. The returned index therefore maximizes the original scores as well, and we evaluate its original score exactly. Since there are a fixed number of coordinate groups, Proposition B.3 gives preprocessing time $\widetilde { \mathrm { O } } ( | \Sigma | ^ { k } )$ and query time $\widetilde { \mathrm { O } } ( 1 )$ , deterministically for $d \leq 3$ □

Envelope construction and proof of Lemma 6.5. Use the identity permutation and $U _ { Y } = U _ { u } = U$ Index the caps by h ∈ [d] and the sources by $d + s ,$ and apply Lemma A.5 once to all these points with a uniform $\mathrm { O } _ { d } ( \lambda )$ bit bound. Define $g _ { h } ( \pmb { u } ) = \widetilde { \pmb { c } } _ { h } \cdot ( \pmb { u } , 1 )$ and set

$$
\mathcal { P } ( \Sigma ^ { \prime } ) = \{ ( \boldsymbol { u } , \boldsymbol { z } ) : \widetilde { f } _ { s } ( \boldsymbol { u } ) \leq \boldsymbol { z } \mathrm { ~ } ( s \in \Sigma ^ { \prime } ) , \quad \boldsymbol { z } \leq g _ { h } ( \boldsymbol { u } ) \mathrm { ~ } ( h \in [ d ] ) \} .
$$

The bounding halfspaces $z \leq g _ { h } ( { \pmb u } )$ and the height $Z _ { \mathrm { t o p } }$ are shared by every sample. By Lemma A.7, each $( \mathbf { \boldsymbol { u } } , Z _ { \mathrm { t o p } } )$ lies in the interior when $\| \mathbf { \boldsymbol { u } } \| _ { \infty } \leq U$ , so the lower boundary over this box is exactly the graph of $E _ { \Sigma ^ { \prime } }$

The cap coeficients, indices, and source coordinates have $\mathrm { O } _ { d } ( \lambda )$ bits. Set the perturbation so that its efect on every source value in $[ - U , U ] ^ { d - 1 }$ is at most $1 / 8$ and is less than one quarter of the gaps between distinct original values at every rational target satisfying the input bounds. These gaps are at least $2 ^ { - \mathrm { O } _ { d } ( \lambda ) }$ , so one perturbation constant depending only on the dimension sufices and also satisfies the cap perturbation bound, proving parts (a) and (b). For part (c), every subset remains in general position, its envelope polytope is bounded by Lemma $\mathrm { A . 7 , }$ and its vertex bit lengths follow from Lemma A.4. The interior point $( u , Z _ { \mathrm { t o p } } )$ needed for ray shooting depends only on the target and the fixed height, so it has $\mathrm { O } _ { d } ( \lambda )$ bits and is computable in $\widetilde { \mathrm { O } } ( 1 )$ time. □

## A.8 Interior Shifts for Cell Location

For a query satisfying the bounds of Lemma 6.8, the descent in Appendix B.4 follows a ray from the least vertex $v _ { F }$ of a face F through a point ${ \bf { \mathcal { x } } } \ne { \bf { \mathcal { v } } } _ { F }$ in that face. Write ${ \pmb w } _ { \mathrm { r a y } } = { \pmb x } - { \pmb v } _ { F }$ For a shift $\delta _ { \mathrm { s h } } \in ( 0 , 1 )$ specified below, we move the ray origin into the interior by setting $v _ { \mathrm { s h } } =$ $( 1 - \delta _ { \mathrm { s h } } ) { \pmb v } _ { F } + \delta _ { \mathrm { s h } } { \pmb x } _ { \mathrm { t o p } }$ . For every constraint with $\mathbf { a } _ { H } \cdot \mathbf { w } _ { \mathrm { r a y } } > 0$ , its exit parameter changes from $\tau _ { H } = ( b _ { H } - \pmb { a } _ { H } \cdot \pmb { v } _ { F } ) / ( \pmb { a } _ { H } \cdot \pmb { w } _ { \mathrm { r a y } } )$ to

$$
\tau _ { H } ^ { \mathrm { s h } } = \tau _ { H } - \delta _ { \mathrm { s h } } \frac { \pmb { a } _ { H } \cdot \left( \pmb { x } _ { \mathrm { t o p } } - \pmb { v } _ { F } \right) } { \pmb { a } _ { H } \cdot \pmb { w } _ { \mathrm { r a y } } } .
$$

Every descent has at most $d - 1$ steps, each using one original vertex and constraint, so its coordinates and exit parameters have $\mathrm { O } _ { d } ( \lambda )$ bits. Their nonzero gaps and denominators are at least $2 ^ { - \mathrm { O } _ { d } ( \lambda ) }$ . A dyadic $\delta _ { \mathrm { s h } } \in ( 0 , 1 )$ with $\mathrm { O } _ { d } ( \lambda )$ bits then preserves every strict ordering among the exit parameters, so each shifted minimizer also minimizes the original parameters. Since the bounds are uniform over samples and targets, a single shift sufices for all location queries.

## A.9 Integer Encodings of Halfspace Ranges

Lemma A.8. Fix $d \ge 2$ and an integer $D \geq 1$ . Let $P ^ { \prime } = \{ z _ { i } / D \ : \ i \in \ [ n ] \} \ \subseteq \ [ 0 , 1 ] ^ { d }$ with $z _ { i } \in \{ 0 , \ldots , D \} ^ { d }$ . Every halfspace range on $P ^ { \prime }$ is realized by a halfspace with a nonzero integer normal and coeficients of $\mathrm { O } _ { d } ( \log ( D + 2 ) )$ bits.

Proof. Without loss of generality, the selected range is nonempty and proper. On the integer points $z _ { i } ,$ maximize the nonnegative separation margin subject to $\| ( \pmb { a } , c ) \| _ { \infty } \leq 1$ ; its positive maximum occurs at a vertex determined by $d + 2$ tight constraints with integer coeficients of magnitude ${ \mathrm { O } } _ { d } ( D + 1 )$ . Cramer’s rule followed by clearing denominators yields integer ${ \mathbf { } } a , c$ of magnitude $( D \dot { + } 1 ) ^ { \mathrm { O } _ { d } \dot { ( 1 ) } }$ (Lemma $\mathrm { A . 4 } )$ , and rescaling the normal by D realizes the original range with the required encoding length. □

## B Geometric Constructions and Sampling Proofs

This appendix proves the geometric data structure bounds and sampling estimates stated in Section $6 ,$ together with the exact batch-maxima bound of Proposition 5.1; Appendix A details their exact rational implementations. Throughout, we follow the notation of Section 6 with fixed $d \geq 2$ and $k = \lfloor d / 2 \rfloor$

## B.1 Classical Polytope Facts

We use the active sets of faces to label the canonical cells. Relative to defining halfspaces $\mathcal { H } ,$ , the active set of a face F is $S _ { F } = \{ H \in \mathcal { H } : F \subseteq \partial H \}$ }, with the standard properties below [Zie95, Propositions 2.2, 2.3, and 2.16, and Theorem 2.7].

Lemma B.1. Let H be a finite set of closed halfspaces in $\mathbb { R } ^ { d }$ in general position whose intersection P is bounded and has nonempty interior. For $\tau \subseteq \mathcal { H }$ define $\begin{array} { r } { A _ { \mathcal { T } } = \bigcap _ { H \in \mathcal { T } } \partial H } \end{array}$ , with $\mathcal { A } _ { \mathcal { O } } = \mathbb { R } ^ { d }$ 2 and $F _ { \mathcal { T } } = P \cap { \mathcal { A } } _ { \mathcal { T } }$

(a) Every vertex of P lies on exactly d of the hyperplanes ∂H, $H \in { \mathcal { H } }$

(b) If $F \tau \neq \emptyset$ , then F contains a vertex of $P , | T | \leq d ,$ $F _ { T }$ is a face of P of dimension $d -$ $| \tau |$ , af $F _ { \mathcal { T } } = A _ { \mathcal { T } }$ , and $\mathcal { T } = \{ H \in \mathcal { H } : F _ { \mathcal { T } } \subseteq \partial H \}$ . Every nonempty face F of P equals $F _ { T }$ for exactly one set T, its active set $\boldsymbol { S _ { F } }$ . In particular $P \cap$ af $F = F$

(c) The number of chains $F ^ { ( 0 ) } \supset F ^ { ( 1 ) } \supset \cdots \supset F ^ { ( d - 1 ) }$ of faces of P with dim $F ^ { ( i ) } = d - 1 - i$ is at most $d ! \left| \operatorname { v e r t } ( P ) \right|$ . Moreover, | vert ${ \mathrm { \Omega } } ( P ) | = \mathrm { O } _ { d } ( | \mathcal { H } | ^ { k } )$ [McM70; Sei17].

The face identities follow from simplicity and independence of the active normals. Ordering the d facets at each vertex gives the flag count in part (c), and McMullen’s Upper Bound Theorem gives the vertex bound by polarity [McM70; Sei17].

Proposition B.2. Let $d \geq 3$ . Let H be a set of $| \mathscr { H } | \geq d + 1$ closed halfspaces in $\mathbb { R } ^ { d }$ in general position whose intersection P is bounded and has nonempty interior, and let $d + 1$ members of H be given whose intersection is bounded and has nonempty interior. A Las Vegas algorithm computes all vertices of P, each with the d members of H whose bounding hyperplanes contain $i t ,$ in expected $\mathrm { O } ( | \mathcal { H } | ^ { k } + | \mathcal { H } | \log ( | \mathcal { H } | + 2 ) )$ arithmetic operations [CS89, Theorems 4.7 and $4 . 8 ]$

Starting from the given bounding simplex, insert the remaining halfspaces in uniformly random order. Each feature is defined by ${ \mathrm O } _ { d } ( 1 )$ variable constraints, and an intersection with q such constraints has complexity $\mathrm { O } _ { d } \big ( ( q + 1 ) ^ { k } \big )$ . Thus the analysis of [CS89, Theorem 3.9] applies with the fixed constraints present in every sample. For rational inputs, Lemma A.4 implies expected running time $\widetilde { \mathrm { O } } ( | \mathcal { H } | ^ { k } )$

## B.2 Constructing Cuttings with a Fixed Bounding Simplex

Proposition B.3 (Ray shooting). Let H be a finite collection of closed halfspaces in $\mathbb { R } ^ { d }$ with rational coeficients, in general position, and let $\textstyle P = \bigcap _ { H \in { \mathcal { H } } } H$ . Suppose that an interior point $\pmb { x } _ { \mathrm { i n } } \in$ int P and a subset $B \subseteq { \mathcal { H } }$ of $d + 1$ members with bounded intersection are given. Let $\lambda \geq \log _ { 2 } ( | \mathcal { H } | + 2 )$ bound the bit lengths of the constraint coeficients and of $\scriptstyle { \mathbf { x } } _ { \mathrm { i n } }$ . A Las Vegas algorithm builds a data structure in expected time $\widetilde { \mathrm { O } } ( | \mathcal { H } | ^ { k } )$ ; for $d \leq 3$ , these bounds hold deterministically. Given a ray $\{ \pmb { x } + \tau _ { \mathrm { r a y } } \pmb { w } _ { \mathrm { r a y } } : \tau _ { \mathrm { r a y } } \ge 0 \}$ with x ∈ int P, ${ \pmb w } _ { \mathrm { r a y } } \neq { \pmb 0 }$ , and coordinates of bit length at most λ, the structure returns in time $\widetilde { \mathrm { O } } ( 1 )$ a constraint $H \in { \mathcal { H } }$ such that $\mathbf { \pmb { a } } _ { H } \cdot \mathbf { \pmb { w } } _ { \mathrm { r a y } } > 0$ and ∂H contains the ray’s exit point from P.

We give the details of the classical constructions of [CS89; MS93] to establish their bounds in our bit-complexity model (Section 2.1).

The ray-shooting algorithm uses the same bounding simplex for every sample and recursive node. Let H be a finite collection of closed halfspaces in $\mathbb { R } ^ { d }$ in general position with a common interior point $\mathbf { \hat { x } } _ { \mathrm { i n } } .$ , and let $B \subseteq { \mathcal { H } }$ consist of $d + 1$ halfspaces with bounded intersection B. Write $\mathcal { A } = \mathcal { H } \backslash B$ For $R \subseteq A$ , define

$$
Q _ { R } = B \cap \bigcap _ { H \in R } H .
$$

Because the fixed constraints are present in every sample, $Q _ { R }$ is always a full-dimensional polytope containing $\scriptstyle { \mathbf { x } } _ { \mathrm { i n } }$ in its interior. We triangulate by coning the lexicographically least vertex of each positive-dimensional face to the triangulations of the facets that omit it. A resulting full-dimensional simplex has vertices $\pmb { v } _ { 0 } , \ldots , \pmb { v } _ { d }$ along a chain $F _ { 0 } = Q _ { R } \supset F _ { 1 } \supset \cdot \cdot \cdot \supset F _ { d }$ , where dim $F _ { i } = d - i , \pmb { v } _ { i }$ is the least vertex of $F _ { i } ,$ and ${ \pmb v } _ { i } \notin F _ { i + 1 }$ for $0 \leq i < d$ . Denote the triangulation by ${ \mathcal { T } } _ { \mathrm { c u t } } ( R )$ , describing each simplex by the ordered active sets of its vertices.

Lemma B.4. Let $R _ { 0 } \subseteq { \mathcal { A } }$ , let $\sigma \in \mathcal { T } _ { \mathrm { c u t } } ( R _ { 0 } )$ , and let $\pmb { v } _ { 0 } , \ldots , \pmb { v } _ { d }$ be its vertices in pulling-chain order. Define

$$
D _ { \sigma } = \mathcal { A } \cap \bigcup _ { i = 0 } ^ { d } \{ H \in \mathcal { H } : v _ { i } \in \partial H \} , \qquad K _ { \sigma } = \{ H \in \mathcal { A } : \sigma \not \subseteq H \} .
$$

Then $| D _ { \sigma } | \leq d ( d + 1 )$ , and for every $R \subseteq A$

$$
\sigma \in { \mathcal { T } } _ { \mathrm { c u t } } ( R ) \quad \iff \quad D _ { \sigma } \subseteq R a n d R \cap K _ { \sigma } = \emptyset .
$$

Moreover, $| \mathcal { T } _ { \mathrm { c u t } } ( R ) | = \mathrm { O } _ { d } ( ( | R | + 1 ) ^ { k } )$ , where $k = \lfloor d / 2 \rfloor$

Proof. By Lemma B.1, each ${ \bf { v } } _ { i }$ lies on exactly d hyperplanes of $B \cup R _ { 0 }$ . General position of H implies that these are also all the hyperplanes of H through ${ \boldsymbol { v } } _ { i } .$ . Hence $| D _ { \sigma } | \leq d ( d + 1 )$ , and every sample in which σ occurs contains $D _ { \sigma }$ and avoids $K _ { \sigma }$

For the converse, suppose that $D _ { \sigma } \subseteq R$ and $R \cap K _ { \sigma } = \emptyset$ . Then $\sigma \subseteq Q _ { R }$ , and each ${ \bf { v } } _ { i }$ has the same complete active set. Let $s _ { i }$ be the active set of the original face $F _ { i }$ , and set

$$
F _ { i } ^ { \prime } = Q _ { R } \cap \bigcap _ { H \in S _ { i } } \partial H .
$$

The constraints of $s _ { i }$ belong to $B \cup D _ { \sigma }$ . The afinely independent points $\pmb { v } _ { i } , \ldots , \pmb { v } _ { d }$ lie in $F _ { i } ^ { \prime }$ and span af $F _ { i }$ , so dim $F _ { i } ^ { \prime } = d - i$ . Consequently the $F _ { i } ^ { \prime }$ form the same sequence of facet incidences, with ${ \pmb v } _ { i } \notin { \cal F } _ { i + 1 } ^ { \prime }$ for $0 \leq i < d$

We show that ${ \mathbf { } } v _ { i }$ is the least point of $F _ { i } ^ { \prime }$ . A bounded polytope is the convex hull of its vertices, so ${ \bf { v } } _ { i }$ is the least point of the original face $F _ { i }$ . If $\pmb { y } \in F _ { i } ^ { \prime }$ were lexicographically smaller than ${ \bf { v } } _ { i }$ , the point ${ \pmb v } _ { i } + t ( { \pmb y } - { \pmb v } _ { i } )$ would satisfy all constraints active at ${ \mathbf { } } v _ { i }$ for $t \in [ 0 , 1 ]$ . Every other constraint of $B \cup R _ { 0 }$ is strict at ${ \mathbf { } } v _ { i }$ and holds for all small enough $t > 0$ . The same point also satisfies the equalities of $S _ { i }$ . This would be a point of $F _ { i }$ smaller than $v _ { i } ,$ a contradiction. Thus ${ \mathbf { } } v _ { i }$ is the bottom vertex of $F _ { i } ^ { \prime } { . }$ , and the original pulling chain occurs in $Q _ { R }$

Finally, the full pulling chains are a subset of the complete face chains counted in Lemma B.1(c), with $Q _ { R }$ prepended. By that lemma, their number is at most d! $| \operatorname { v e r t } ( Q _ { R } ) | = \mathrm { O } _ { d } ( ( | R | + d + 1 ) ^ { k } ) =$ $\mathrm { O } _ { d } ( ( | R | + 1 ) ^ { k } )$ □

The set $K _ { \sigma }$ contains all halfspaces that prevent $\sigma$ from occurring in a sample. To build a cutting, we also count the hyperplanes crossing its interior,

$$
c _ { \sigma } = | \{ H \in { \mathcal { A } } : \partial H \cap \operatorname { i n t } \sigma \neq \emptyset \} | .
$$

A hyperplane crossing the interior of $\sigma$ has points of $\sigma$ on both sides, so $c _ { \sigma } \leq | K _ { \sigma } |$ . The time for refining simplices with large $c _ { \sigma }$ depends on a moment estimate obtained by comparing a sample with a smaller uniform sample.

Lemma B.5. Let $n = | { \mathcal { A } } | \geq 1$ , and let R be a uniformly random q-element subset of ${ \mathcal { A } } ,$ , where $q$ is an integer with $1 \leq q \leq n$ . For every fixed real $\nu \geq 1$

$$
\mathbb { E } \sum _ { \sigma \in { \mathcal { T } } _ { \mathrm { c u t } } ( R ) } \operatorname* { m a x } \{ 1 , q | K _ { \sigma } | / n \} ^ { \nu } = \mathrm { O } _ { d , \nu } ( q ^ { k } ) .
$$

The estimate also applies to interior crossings, with $c _ { \sigma }$ in place $o f \left| K _ { \sigma } \right|$

Proof. Set $b _ { \mathrm { d e f } } = d ( d + 1 )$ . For $q < 4 b _ { \mathrm { d e f } }$ , the number of simplices and every summand are bounded by constants depending only on $d , \nu ,$ so the claim follows.

For $q \geq 4 b _ { \mathrm { d e f } }$ , let $q _ { - } = \lfloor q / 2 \rfloor$ . For a fixed configuration $\sigma ,$ let $a = | D _ { \sigma } |$ and $b = | K _ { \sigma } |$ , and write $p _ { j } ( \sigma )$ for its probability of occurring in a uniformly random j-element sample. Configurations with $p _ { q } ( \sigma ) = 0$ contribute zero to the desired expectation.

For the remaining configurations, drawing a uniform q-element sample and then a uniform $q _ { - } -$ element subset is equivalent to drawing the smaller sample first and extending it uniformly. Lemma B.4 implies

$$
p _ { q } ( \sigma ) \frac { ( q _ { - } ) _ { a } } { ( q ) _ { a } } = p _ { q _ { - } } ( \sigma ) \frac { \binom { n - q _ { - } - b } { q - q _ { - } } } { \binom { n - q _ { - } } { q - q _ { - } } } ,
$$

where $( j ) _ { a } = j ( j - 1 ) \cdot \cdot \cdot ( j - a + 1 )$ and an impossible selection has probability zero. Since $a \leq$ $b _ { \mathrm { d e f } }$ and $q _ { - } \geq 2 b _ { \mathrm { d e f } }$ , we have $( q ) _ { a } / ( q _ { - } ) _ { a } \leq 4 ^ { b _ { \mathrm { d e f } } }$ . The probability that the extension avoids $K _ { \sigma }$ is at most $\mathrm { e } ^ { - ( q - q _ { - } ) b / ( n - q _ { - } ) } < \mathrm { e } ^ { - q b / ( 2 \bar { n } ) }$ . It follows that

$$
p _ { q } ( \sigma ) \leq 4 ^ { b _ { \mathrm { d e f } } } \mathrm { e } ^ { - q b / ( 2 n ) } p _ { q _ { - } } ( \sigma ) .
$$

For fixed $\nu ,$ the function max $\{ 1 , t \} ^ { \nu } \mathrm { e } ^ { - t / 2 }$ is bounded on $[ 0 , \infty )$ . Multiplying this inequality by max $\{ 1 , q b / n \} ^ { \nu }$ and summing over the finitely many configurations yields

$$
\mathbb { E } \sum _ { \sigma \in { \mathcal T } _ { \mathrm { c u t } } ( R ) } \operatorname* { m a x } \{ 1 , q | K _ { \sigma } | / n \} ^ { \nu } \leq \mathrm { O } _ { d , \nu } ( 1 ) \sum _ { \sigma } p _ { q _ { - } } ( \sigma ) = \mathrm { O } _ { d , \nu } ( q ^ { k } ) ,
$$

by Lemma B.4. The last assertion follows from $c _ { \sigma } \leq | K _ { \sigma } |$ . This is the double-sampling argument underlying the configuration bounds of [CS89; CF90], with B fixed in every sample. □

Lemma B.6. Under the preceding assumptions, let $n = | { \mathcal { A } } | \geq 1$ and let q be an integer with $1 \leq q \leq n$ . A Las Vegas algorithm constructs $\mathrm { O } _ { d } ( q ^ { k } )$ closed full-dimensional simplices whose union contains $Q _ { A }$ , each meeting $Q _ { A }$ , such that the interior of each simplex is crossed by at most $n / q$ hyperplanes of A. It also computes their closed conflict lists

$$
\mathcal A _ { \tau } = \{ H \in \mathcal A : \partial H \cap \tau \neq \emptyset \} , \qquad | \mathcal A _ { \tau } | \leq n / q + d ( d + 1 ) .
$$

The algorithm uses $n q ^ { \mathrm { O } _ { d } ( 1 ) }$ arithmetic and combinatorial operations in expectation. Every vertex is obtained from ${ \mathrm { O } } _ { d } ( 1 )$ original constraints by linear algebra of depth at most two.

Proof. We draw a uniform q-element sample $R \subseteq A$ , enumerate $Q _ { R }$ and its pulling triangulation, and count the interior crossings $c _ { \sigma }$ for each $\sigma \in \mathcal { T } _ { \mathrm { c u t } } ( R )$ in $n q ^ { \mathrm { O } _ { d } ( 1 ) }$ time. Set $\kappa _ { \sigma } = q c _ { \sigma } / n$ and refine each simplex with $\kappa _ { \sigma } > 1$ as follows.

Local refinement. Let $\scriptstyle { \mathcal { C } } _ { \sigma }$ be the set of $c _ { \sigma }$ hyperplanes crossing int $\sigma ,$ which contains every hyperplane that can cross the interior of a subsimplex of $\sigma _ { : }$ , and set $u _ { \sigma } = \lceil \kappa _ { \sigma } \rceil$ . We draw a uniform sample of size

$$
h _ { \sigma } = \operatorname* { m i n } \{ c _ { \sigma } , C _ { d } u _ { \sigma } \lceil \log _ { 2 } ( u _ { \sigma } + 1 ) \rceil \}
$$

from $\scriptstyle { \mathcal { C } } _ { \sigma }$ , with a positive integer $C _ { d }$ depending only on d and satisfying the sampling estimate below. Integer division and bit-length comparisons compute this size, which is $\mathrm { O } _ { d } ( \kappa _ { \sigma } \log ( \kappa _ { \sigma } + 1 ) )$ . We form the sample’s arrangement inside $\sigma ,$ triangulate the closures of its full-dimensional cells by pulling, and verify that every resulting simplex has at most $n / q$ interior crossings. If the test fails, we repeat the local sampling.

The ranges on $\scriptstyle { \mathcal { C } } _ { \sigma }$ consist of the hyperplanes crossing the interior of a full-dimensional simplex. For a simplex with vertices $\pmb { w } _ { 0 } , \ldots , \pmb { w } _ { d }$ , the crossing condition is

$$
\operatorname* { m i n } _ { i } ( \pmb { a } _ { H } \cdot \pmb { w } _ { i } - b _ { H } ) < 0 < \operatorname* { m a x } _ { i } ( \pmb { a } _ { H } \cdot \pmb { w } _ { i } - b _ { H } ) .
$$

On any h hyperplanes, these ranges have $h ^ { \operatorname { O } _ { d } ( 1 ) }$ distinct traces, determined by the signs of $( d +$ $1 ) h$ linear forms in the $d ( d + 1 )$ vertex coordinates. Thus the range space has VC dimension depending only on d.

If $h _ { \sigma } = c _ { \sigma }$ , the full sample satisfies the crossing bound. Otherwise the random ε-net theorem [HW87], with $\varepsilon = 1 / \kappa _ { \sigma }$ , shows that $h _ { \sigma }$ independent draws intersect every range of size greater than $c _ { \sigma } / \kappa _ { \sigma } = n / q$ with probability at least $1 / 2$

Deleting repetitions from the independent draws and filling uniformly from the unused elements gives a uniform $h _ { \sigma }$ -element subset by symmetry. It intersects every range met by the independent draws and is therefore an ε-net with probability at least $1 / 2$ . Since the interiors of the triangulated arrangement cells have no sample crossings, every such net satisfies the crossing bound, so local refinement requires at most two attempts in expectation.

If the local sample has h hyperplanes, the arrangement inside σ has $\mathrm O _ { d } ( ( h + 1 ) ^ { d } )$ vertices. At each vertex, at most d original hyperplanes and at most $d + 1$ facet hyperplanes of $\sigma$ are incident. There are thus ${ \mathrm { O } } _ { d } ( 1 )$ complete face chains through each vertex, including incidences between cells, and $\mathrm { O } _ { d } \big ( ( h + 1 ) ^ { d } \big )$ simplices in the pulling triangulations. This count remains valid in the presence of dependencies among the original hyperplanes and the facet hyperplanes of $\sigma .$ Since $\log ^ { d } ( \kappa + 1 ) = \mathrm { O } _ { d } ( \kappa )$ for $\kappa \geq 1$ , every accepted refinement has $\mathrm { O } _ { d } ( \kappa _ { \sigma } ^ { d + 1 } )$ simplices.

Acceptance and cost. After all refinements, discard every simplex τ with $\tau \cap Q _ { \cal A } = \emptyset$ . This is a fixed-dimensional linear-programming feasibility test on $n + \mathrm O _ { d } ( 1 )$ constraints using ${ \mathrm { O } } _ { d } ( n )$ operations $\mathrm { [ M e g 8 4 ] }$ . Appendix A.5 specifies the predicates of bounded algebraic depth for this feasibility test. The remaining simplices cover $Q _ { A }$ , because the refinements cover $Q _ { R } \supseteq Q _ { A }$ . Accept if at most $C _ { d } ^ { \prime } q ^ { k }$ simplices remain, and otherwise draw a new outer sample R, where the constant $C _ { d } ^ { \prime }$ depends only on d and satisfies the estimate below.

Conditional on $R ,$ , the number of simplices before discarding is at most

$$
\mathrm { O } _ { d } ( 1 ) \sum _ { \sigma \in { \mathcal { T } } _ { \mathrm { c u t } } ( R ) } \operatorname* { m a x } \{ 1 , \kappa _ { \sigma } \} ^ { d + 1 } .
$$

Its expectation is $\mathrm { O } _ { d } ( q ^ { k } )$ by Lemma B.5, and Markov’s inequality gives outer acceptance probability at least $1 / 2$ for a large enough $C _ { d } ^ { \prime }$ . Each attempt, including the local retries, uses $n q ^ { \mathrm { O } _ { d } ( 1 ) }$ operations in expectation, since $\kappa _ { \sigma } \leq q$ and all candidate lists are checked by enumeration. Repetition therefore increases the expected cost by at most a factor of two.

Each remaining simplex has at most $n / q$ interior crossings. A hyperplane that meets a closed simplex while avoiding its interior supports a face containing a vertex. By general position, at most d original hyperplanes pass through any vertex, so at most $d ( d + 1 )$ additional hyperplanes belong to the closed conflict list.

Coordinates. Appendix A.5 constructs every cutting vertex from ${ \mathrm { O } } _ { d } ( 1 )$ original constraints through two levels of linear algebra, with polynomial exact arithmetic cost. □

Proof of Proposition B.3. For d $l \leq 3 .$ the deterministic algorithms described in Appendix A.5 achieve the required real-RAM complexity [Cha96b, Lemma 2.1 and Corollary 2.2]. For $d \geq 4$ , we use the static recursion of Matousek and Schwarzkopf [MS93, Lemma 7], with the cuttings constructed above. The implementation uses the same bounding simplex at every node and closed conflict lists, with bounded coordinate complexity.

Recursive structure. Every node keeps B and stores a variable subset $\mathcal { A } ^ { \prime } \subseteq \mathcal { H } \setminus B$ , representing $\begin{array} { r } { Q _ { A ^ { \prime } } = B \cap \bigcap _ { H \in \mathcal { A } ^ { \prime } } H } \end{array}$ . For $n ^ { \prime } = | \mathcal { A } ^ { \prime } |$ above a constant threshold, set $q = \lfloor ( n ^ { \prime } ) ^ { \alpha } \rfloor$ , where the positive rational α depends only on d and satisfies the preprocessing estimate below, and construct the cutting of Lemma B.6. The child for τ stores its closed conflict list $\mathbf { \mathcal { A } } _ { \tau } ^ { \prime }$ , together with B. As in [MS93, Lemmas 7 and 11], a query descends through a simplex containing the last point of the ray in the union of the cutting simplices. This union is compact and contains $Q _ { A ^ { \prime } }$ . A leaf scans its constraints and returns one minimizing the exit parameter

$$
\frac { b _ { H } - \mathbf { a } _ { H } \cdot \mathbf { \boldsymbol { x } } } { \mathbf { a } _ { H } \cdot \mathbf { \boldsymbol { w } } _ { \mathrm { r a y } } } \qquad ( \mathbf { a } _ { H } \cdot \mathbf { \boldsymbol { w } } _ { \mathrm { r a y } } > 0 ) .
$$

Exact auxiliary queries. To select the last simplex met by the ray, we use the sign-vector method of [MS93, Lemmas 11–12]. For each simplex facet H, define

$$
U _ { H } = b _ { H } - \mathbf { a } _ { H } \cdot \mathbf { x } , \qquad V _ { H } = \mathbf { a } _ { H } \cdot \mathbf { w } _ { \mathrm { r a y } } .
$$

The signs of $U _ { H } , V _ { H }$ and $U _ { H } V _ { H ^ { \prime } } - U _ { H ^ { \prime } } V _ { H }$ determine the ray’s intersection interval with each simplex and the order of all finite endpoints, including the cases $V _ { H } = 0$ and equal exit parameters. They come from $q ^ { \mathrm { O } _ { d } ( 1 ) }$ polynomials of degree at most two in the 2d ray parameters. Construct a signinvariant cylindrical algebraic decomposition, including cells of every dimension, and associate each cell representing a ray that meets the union with a simplex of largest exit parameter; break ties by simplex index [BPR06].

This decomposition requires $q ^ { \mathrm { O } _ { d } ( 1 ) }$ operations because the dimension is fixed. At query time, binary search among the ordered roots at each of the 2d levels locates the ray in $\mathrm { O } _ { d } ( \log ( q + 2 ) )$ exact geometric operations. Thus we obtain an exact implementation of the point-location structure of [CEGS91] used in the cited recursion.

Closed conflicts and ray exits. Let $\scriptstyle { \pmb { x } } ^ { * }$ be the exit point from the current node polytope. The selected simplex τ contains a point $\pmb { x } _ { \mathrm { r a y } } ^ { \prime }$ of the ray at or beyond $\ b { x } ^ { * }$ , and Lemma B.6 ensures the existence of a point $\pmb { x } _ { P } ^ { \prime } \in \tau \cap Q _ { A ^ { \prime } }$

The query origin remains interior to every node polytope, since their constraints are drawn from H. For any $H \in { \mathcal { A } } ^ { \prime }$ tight at $\scriptstyle { \pmb { x } } ^ { * }$ , we have $\mathbf { a } _ { H } \cdot \mathbf { w } _ { \mathrm { r a y } } > 0$ and

$$
\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { }  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { }  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { }  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { }  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf
$$

The segment joining these two points lies in τ and intersects ∂H, so $H \in A _ { \tau } ^ { \prime }$ . Together with the fixed constraints $B ,$ , the child contains every constraint tight at $\mathbf { \boldsymbol { x } } ^ { * }$ . Its polytope contains the parent’s and has the same ray exit, so induction along the query path proves that the leaf returns a valid original constraint.

Complexity. An accepted node has at most $C _ { d } ^ { \prime } q ^ { k }$ children, each with at most $n ^ { \prime } / q + c _ { d }$ variable constraints, where $c _ { d } = d ( d + 1 )$ . For small enough α, its expected preprocessing cost is $\mathrm { O } _ { d } ( ( n ^ { \prime } ) ^ { k } )$ since $k \geq 2$ . Let $S ( t )$ be the worst expected preprocessing time of any node with at most t variable constraints. Independent random choices for the children give

$$
S ( n ^ { \prime } ) \le C _ { d } ^ { \prime } q ^ { k } S ( n ^ { \prime } / q + c _ { d } ) + \mathrm { O } _ { d } ( ( n ^ { \prime } ) ^ { k } ) .
$$

Set the leaf threshold so that $n ^ { \prime } > 2 c _ { d }$ and $q \geq 2$ at every internal node. Then

$$
( n ^ { \prime } / q + c _ { d } ) - 2 c _ { d } \leq ( n ^ { \prime } - 2 c _ { d } ) / q .
$$

For these shifted node sizes, the storage analysis of [MS93, Lemma $7 ]$ implies $S ( n ^ { \prime } ) = \widetilde { \mathrm { O } } ( ( n ^ { \prime } + 1 ) ^ { k } )$ and the single-path query analysis implies $\widetilde { \mathrm { O } } ( 1 )$ operations per query. The resampling procedures terminate with probability one, and the node sizes decrease above the leaf threshold.

Lemma A.4, applied to the representations in Appendix $\mathrm { A . 5 } ,$ converts these operation counts to bit complexity. Preprocessing is deterministic for $d \leq 3$ and Las Vegas in higher dimensions.

## B.3 Strict Halfspace Reporting

Proof of Lemma 6.3. Partition Σ, in its fixed enumeration order, into $\lceil | \Sigma | / g _ { \mathrm { r e p } } \rceil$ groups of size at most $g _ { \mathrm { r e p } }$ . Within each group, build the binary tree of [Cha96a, Observation 2.3] and use Lemma 6.2 to test its nodes. At a node containing $\Sigma ^ { \prime }$ , the query continues precisely when ma ${ \mathfrak { c } } _ { s \in \Sigma ^ { \prime } } { \pmb q } \cdot { \pmb y } _ { s } > \theta$ A leaf stores the original source index, so repeated points are reported separately.

For a group of size $^ { g , }$ the tree requires $\widetilde { \mathrm { O } } ( g ^ { k } )$ expected time to preprocess, for a total of $\widetilde { \mathrm { O } } ( | \Sigma | g _ { \mathrm { r e p } } ^ { k - 1 } )$ over all groups. A query visits the group roots and only $\mathrm { O } ( \log ( g _ { \mathrm { r e p } } + 1 ) )$ additional nodes per reported index, each requiring one support query. Its total time is $\widetilde { \mathrm { O } } ( | \Sigma | / g _ { \mathrm { r e p } } + 1 + | Z | )$ . The algorithm is deterministic for $d \leq 3$ □

## B.4 Canonical Cells and Point Location

Work in the setup of Definition 6.4, with $d \geq 3$

Definition B.7. Let $\emptyset \neq \Sigma ^ { \prime } \subseteq \Sigma _ { 0 }$ and $\mathcal { P } = \mathcal { P } ( \Sigma ^ { \prime } )$ . Order the vertices of $\mathcal { P }$ lexicographically by their coordinates. For a nonempty face F of ${ \mathcal { P } } _ { : }$ let $v _ { F }$ be its least vertex; it exists by Lemma B.1(b). A pulling chain is a sequence of faces $F ^ { ( 0 ) } \supset F ^ { ( 1 ) } \supset \cdots \supset F ^ { ( d - 1 ) }$ such that $F ^ { ( 0 ) }$ is the nonempty facet of a source $s \in \Sigma ^ { \prime }$ and, for $1 \leq i \leq d - 1 , F ^ { ( i ) }$ is a facet of $F ^ { ( i - 1 ) }$ with ${ \pmb v } _ { F ^ { ( i - 1 ) } } \notin F ^ { ( i ) }$ . Its projected simplex is

$$
\Delta = \mathrm { c o n v } \big \{ \bar { v } _ { F ^ { ( 0 ) } } , \bar { v } _ { F ^ { ( 1 ) } } , \dots , \bar { v } _ { F ^ { ( d - 1 ) } } \big \} ,
$$

with owner $s _ { \Delta } = s$ and label

$$
\big ( s _ { \Delta } , S _ { v _ { F } ( 0 ) } , \ldots , S _ { v _ { F } ( d - 1 ) } \big ) ,
$$

where each active set is the sorted tuple of its constraint indices. The canonical cells are listed in the lexicographic order of these labels.

Proof of Lemma 6.7. Write $\mathcal { P } = \mathcal { P } ( \Sigma ^ { \prime } )$ . The recursive vertex triangulation of [Cla88, Section 3], using the least vertex on each face, triangulates every source facet into the simplices of Definition B.7. For a cell $\Delta ,$ projection from the graph $z = \widetilde { f } _ { s _ { \Delta } } ( \pmb { u } )$ to u is an afine bijection. Hence each simplex projects to a closed (d−1)-simplex with rational vertices, and its owner realizes the sampled envelope on the entire cell.

For every $\pmb { u } \in [ - U , U ] ^ { d - 1 }$ , the point $( { \pmb u } , E _ { \Sigma ^ { \prime } } ( { \pmb u } ) )$ lies on the lower boundary of P by Definition 6.4. Hence it belongs to a simplex on a source facet, whose projection is a canonical cell containing the target, proving the coverage in part (a).

The sufix vertices of a pulling chain have the same afine hull as the corresponding face; the identity $\mathcal { P } \cap$ af $F = F$ in Lemma B.1 then determines the chain from its vertex active sets. Thus distinct chains have distinct labels, and the labeled list depends only on the sample.

For part (b), the afine function $\widetilde { f } _ { s } - \widetilde { f } _ { s _ { \Delta } }$ is positive somewhere on a simplex exactly when it is positive at a vertex. The owner is maximal among the sampled sources on its entire facet, so $C _ { \Delta } \cap \Sigma ^ { \prime } = \emptyset$

For part (c), every pulling chain is a complete face chain. By Lemma B.1(c), their number is at most

$$
d ! | \operatorname { v e r t } ( { \mathcal { P } } ( \Sigma ^ { \prime } ) ) | = \mathrm { O } _ { d } ( ( | \Sigma ^ { \prime } | + d ) ^ { k } ) = \mathrm { O } _ { d } ( | \Sigma ^ { \prime } | ^ { k } ) .
$$

Let $C _ { \mathrm { c e l l } }$ be the implied constant in this bound, increased to at least one.

For part (d), each vertex is the unique intersection of the d hyperplanes in its active set. Every such set is a subset of the $| \Sigma | + d$ fixed source and cap constraints. Thus the owner and d vertex active sets have at most

$$
| \Sigma | ( | \Sigma | + d ) ^ { d ^ { 2 } } \leq ( | \Sigma | + d ) ^ { d ^ { 2 } + 1 }
$$

possible encodings. Each encoding determines the vertices and owner, and hence the cell; because the source functions are fixed, its conflict set is fixed as well. □

Proof of Lemma 6.8. Write $\mathcal { P } = \mathcal { P } ( \Sigma ^ { \prime } )$ and let H be its set of $| \Sigma ^ { \prime } | + d$ constraints. By Lemma 6.5(c), these constraints are in general position and their intersection is bounded. The point $\scriptstyle { \pmb { x } } _ { \mathrm { t o p } } =$ $( \mathbf { 0 } , Z _ { \mathrm { t o p } } )$ is interior by the cap inequalities in Appendix $\mathrm { A . 7 } .$ To descend from a face F through a point ${ \pmb x } \in \ b { F } \setminus \{ { \pmb v } _ { F } \}$ , put ${ \pmb w } _ { \mathrm { r a y } } = { \pmb x } - { \pmb v } _ { F }$ and extend the ray to its exit point ${ \pmb x } ^ { \prime } = { \pmb v } _ { F } + \tau _ { * } { \pmb w } _ { \mathrm { r a y } }$ , where

$$
\tau _ { H } = \frac { b _ { H } - \pmb { a } _ { H } \cdot \pmb { v } _ { F } } { \pmb { a } _ { H } \cdot \pmb { w } _ { \mathrm { r a y } } } , \qquad \tau _ { * } = \operatorname* { m i n } _ { \pmb { a } _ { H } \cdot \pmb { w } _ { \mathrm { r a y } } > 0 } \tau _ { H } \geq 1 .
$$

By Lemma $\mathrm { { B . 1 ( b ) } }$ , every minimizing constraint defines a facet $F ^ { \prime } = F \cap \partial H$ omitting $v _ { F }$ , and $\pmb { x } = ( 1 - 1 / \tau _ { * } ) \pmb { v } _ { F } + ( 1 / \tau _ { * } ) \pmb { x } ^ { \prime }$ is the convex-combination relation used in the descent.

Faces. We begin by enumerating the vertices and their active constraints with Proposition B.2. Use the d cap constraints and the source constraint of some $s \in \Sigma ^ { \prime }$ as the initial constraints. Their intersection is $\mathcal { P } ( \{ s \} )$ , which is bounded and has nonempty interior by Lemma 6.5(c). The output consists of all vertices v of $\mathcal { P }$ with their active sets $S _ { v }$

These active sets determine every face. By Lemma B.1(b), a set $\tau \subseteq \mathcal { H }$ is the active set of a nonempty face if and only if $\mathcal { T } \subseteq S _ { v }$ for some vertex v; the vertices of $F \tau$ are exactly those with $\tau \subseteq S _ { v }$ . List the $2 ^ { d } | \mathrm { v e r t } ( \mathcal { P } ) |$ pairs $( \tau , v )$ with $\mathcal { T } \subseteq S _ { v }$ and sort by $\tau$ , represented as a sorted tuple of indices. This groups the vertices of each nonempty face F and identifies its least vertex $v _ { F }$

The facets of F are exactly $F _ { S _ { F } \cup \{ H \} }$ with $H \in { \cal S } _ { v } \backslash { \cal S } _ { F }$ for some vertex $v \in F$ . Each such face has dimension dim $F - 1$ by Lemma B.1(b), and every facet of $F$ contains a vertex whose active set contains its own. Thus, for every pair $( \tau , v )$ and every $H \in S _ { v } \setminus \mathcal T$ , form $( { \mathcal { T } } , { \mathcal { T } } \cup \{ H \} )$ and remove repetitions by sorting. There are at most $d 2 ^ { d } | \mathrm { v e r t } ( \mathcal { P } ) |$ such pairs. We store the faces in a dictionary indexed by their active sets, together with these facet lists. For each $F$ with dim $F \geq 1$ let $\operatorname { n e x t } ( F )$ be the first facet in this order that omits $v _ { F }$

Cells. A depth-first search now enumerates the pulling chains, starting at each nonempty source facet and, at a face $F ,$ continuing into the facets $F ^ { \prime }$ with $\pmb { v } _ { F } \notin F ^ { \prime }$ . Every branch extends to a complete chain, and at most dim $F < d$ facets are skipped at $F ;$ their active sets have the form $S _ { F } \cup \{ H \}$ with $H \in { \cal S } _ { v _ { F } } \backslash { \cal S } _ { F }$ . Each other facet starts a branch, so the search uses $\mathrm { O } ( d ^ { 2 } )$ steps per complete chain.

Assign each chain its label from Definition B.7. The labels are distinct by the proof of Lemma $6 . 7 ;$ sorting them produces the list in part (a) and a dictionary from labels to positions. Each label and its vertices have $\mathrm { O } _ { d } ( \lambda ) { \mathrm { - b i t } }$ descriptions, so comparisons require time $\widetilde { \mathrm { O } } ( 1 )$ . There are at most $d ! \left| \operatorname { v e r t } ( { \mathcal { P } } ) \right|$ chains by Lemma $\mathrm { B . 1 ( c ) }$

Ray shooting. Build the structure of Proposition B.3 for H, using ${ \pmb x } _ { \mathrm { t o p } }$ as the interior point and the d cap constraints together with the source constraint used above as B. Their intersection is the bounded simplex $\mathcal { P } ( \{ s \} )$ ).

Location. Given ${ \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } ^ { \mathbf { } } \mathbf { u } ,$ first shoot a vertical ray from $( \mathbf { { u } } , Z _ { \mathrm { { t o p } } } )$ in direction $( \mathbf { 0 } , - 1 )$ . Its origin is interior by Lemma $\mathrm { A . 7 ( a ) }$ , and part (c) identifies the answer as the source constraint of some $s _ { \mathrm { o w n } } \in \Sigma ^ { \prime }$ with $\widetilde { f } _ { s _ { \mathrm { o w n } } } ( \pmb { u } ) = \widetilde { E } _ { \Sigma ^ { \prime } } ( \pmb { u } )$ . Thus the exit point ${ \pmb x } ^ { ( 0 ) } = ( { \pmb u } , E _ { \Sigma ^ { \prime } } ( { \pmb u } ) )$ lies in the nonempty facet $F ^ { ( 0 ) }$ of $s _ { \mathrm { o w n } }$ . We then descend through faces: for $i \geq 0$ , suppose we have a face $F ^ { ( i ) }$ of dimension $d - 1 - i$ and a point $\pmb { x } ^ { ( i ) } \in F ^ { ( i ) }$

$\textit { \textbf { I f } i } = d - 1$ , then $F ^ { ( i ) }$ is a vertex and $\pmb { x } ^ { ( i ) } = \pmb { v } _ { F ^ { ( i ) } } ; \mathrm { s t o p }$

• If $i < d - 1$ and $\pmb { x } ^ { ( i ) } = \pmb { v } _ { F ^ { ( i ) } }$ , complete the chain by $F ^ { ( i + 1 ) } = \mathrm { n e x t } ( F ^ { ( i ) } ) , F ^ { ( i + 2 ) } = \mathrm { n e x t } ( F ^ { ( i + 1 ) } )$ and so on, and stop.

• Otherwise, set $F = F ^ { ( i ) }$ and $\pmb { x } = \pmb { x } ^ { ( i ) }$ in the formulas for the exit point above, so that ${ \pmb w } _ { \mathrm { r a y } } =$ $\pmb { x } ^ { ( i ) } - \pmb { v } _ { F ^ { ( i ) } }$ , and set ${ \pmb v } _ { \mathrm { s h } } = ( 1 - \delta _ { \mathrm { s h } } ) { \pmb v } _ { F ^ { ( i ) } } + \delta _ { \mathrm { s h } } { \pmb x } _ { \mathrm { t o p } }$ , with $\delta _ { \mathrm { s h } } \in ( 0 , 1 )$ fixed in Appendix $\mathrm { A . 8 }$ The point ${ v _ { \mathrm { s h } } }$ is interior, since ${ \pmb a } _ { H } \cdot { \pmb v } _ { \mathrm { s h } } < b _ { H }$ for every $H \in { \mathcal { H } }$ . Query the structure with the ray $\{ { \pmb v } _ { \mathrm { s h } } + \tau _ { \mathrm { r a y } } { \pmb w } _ { \mathrm { r a y } } : \tau _ { \mathrm { r a y } } \ge 0 \}$ . Let H be the answer, compute $\tau _ { * } = \tau _ { H }$ and $\pmb { x } ^ { ( i + 1 ) } =$ ${ \pmb v } _ { F ^ { ( i ) } } + \tau _ { * } { \pmb w } _ { \mathrm { r a y } }$ , and let $F ^ { ( i + 1 ) } = F _ { S _ { F } ( i ) } { \cup } \{ H \}$ , found in the dictionary.

Finally, find the resulting chain in the dictionary of tuples and return its position in the canonical cell list and its owner $s _ { \mathrm { o w n } }$

Correctness of a step. The shift $\delta _ { \mathrm { s h } }$ from Appendix A.8 ensures that the constraint returned by the interior ray minimizes the original exit parameter $\tau _ { H }$ . The exit point thus lies in a facet omitting the least vertex, with $\pmb { x } ^ { ( i ) } \in \mathrm { c o n v } \{ \pmb { v } _ { F ^ { ( i ) } } , \pmb { x } ^ { ( i + 1 ) } \}$ .

Correctness of the location. Each step selects a facet omitting the least vertex, and next completes the chain with the same property if the descent stops early. The resulting faces form a pulling chain, and iterating the convex-combination relation shows that

$$
\pmb { x } ^ { ( 0 ) } \in \mathrm { c o n v } \{ \pmb { v } _ { F ^ { ( 0 ) } } , \dotsc , \pmb { v } _ { F ^ { ( d - 1 ) } } \} .
$$

Thus ${ \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } ^ { \mathbf { } } \mathbf { u } ,$ the projection onto the first $d - 1$ coordinates, belongs to the enumerated cell of this chain, whose owner is $s _ { \Delta } = s _ { \mathrm { o w n } }$

Cost. The sample determines $\mathrm { O } _ { d } ( | \Sigma ^ { \prime } | ^ { k } )$ vertices, face incidences, and pulling chains, so enumeration, sorting, and preprocessing for ray shooting require expected time $\widetilde { \mathrm { O } } ( | \Sigma ^ { \prime } | ^ { k } )$ , including the exact arithmetic of Appendix $\mathrm { A . 5 }$ . Locating a cell requires at most d ray queries and a constant number of dictionary lookups in $\widetilde { \mathrm { O } } ( 1 )$ total time, with coordinates obtained at bounded depth as in Appendix $\mathrm { A } . 8 .$ □

## B.5 Sampling and Potential Estimates

To fix the sampling parameters (6.1), let $c _ { \mathrm { c o n f } } \geq 1$ be an integer depending only on d such that for every integer $x \geq 1$

$$
( x + d ) ^ { d ^ { 2 } + 1 } [ \log _ { 2 } ( x + 8 ) ] \leq ( x + 8 ) ^ { c _ { \mathrm { c o n f } } } , \qquad C _ { \mathrm { c e l l } } x ^ { k } [ \log _ { 2 } ( x + 8 ) ] \leq ( x + 8 ) ^ { c _ { \mathrm { c o n f } } } .
$$

Such a constant exists because both counts grow polynomially in x times a logarithmic factor. Set

$$
G = \lceil \log _ { 2 } ( N + 8 ) \rceil , \qquad C _ { \mathrm { s a m p } } = 6 4 ( c _ { \mathrm { c o n f } } + 1 ) , \qquad s _ { 0 } = 2 C _ { \mathrm { s a m p } } G , \qquad T = 4 ^ { k + 1 } .
$$

Then $G \ge 4$ exceeds the number of prefix levels used below, and $( N + 8 ) ^ { c _ { \mathrm { c o n f } } - C _ { \mathrm { s a m p } } } < 1 / 8$

Proof of Lemma 6.9. By Lemma $6 . 7 ( \mathrm { d } )$ , at most $( | \Sigma | + d ) ^ { d ^ { 2 } + 1 }$ owner–cell pairs $( s _ { \Delta } , \Delta )$ occur over all nonempty subsets of Σ, and each pair determines its conflict set $C _ { \Delta }$ . These pairs and their conflict sets are determined before sampling by the shared perturbation (Definition 6.4). Consider such a pair with $| C _ { \Delta } | > C _ { \mathrm { s a m p } } | \Sigma | G / | \Sigma ^ { \prime } |$ . If this pair occurs in $\mathcal { P } ( \Sigma ^ { \prime } )$ , then $\Sigma ^ { \prime } \cap C _ { \Delta } = \emptyset$ by Lemma $6 . 7 ( \mathrm { b } )$ The probability of this event is at most $\mathrm { e } ^ { - | \Sigma ^ { \prime } | | C _ { \Delta } | / | \Sigma | }$ by Lemma 2.4, and

$$
\mathrm { e } ^ { - | \Sigma ^ { \prime } | | C _ { \Delta } | / | \Sigma | } < \mathrm { e } ^ { - C _ { \mathrm { s a m p } } G } \leq 2 ^ { - C _ { \mathrm { s a m p } } G } \leq ( N + 8 ) ^ { - C _ { \mathrm { s a m p } } } .
$$

By the definition of $c _ { \mathrm { c o n f } }$ with $x = N$ , and since $| \Sigma | \leq N$ , the number of pairs is at most $\begin{array} { r } { ( N + d ) ^ { d ^ { 2 } + 1 } \leq } \end{array}$ $( N + 8 ) ^ { c _ { \mathrm { c o n f } } } / G . \mathrm { ~ A ~ }$ union bound over the pairs with large conflict sets implies failure probability at most $( N + 8 ) ^ { c _ { \mathrm { c o n f } } - C _ { \mathrm { s a m p } } } / G < 1 / ( 8 G )$ □

Proof of Lemma 7.10. Let $\mathcal { D } _ { \mathrm { ~ h ~ } } \subseteq ~ \mathcal { D }$ be the set of heavy $\Delta .$ . Since $\Sigma _ { \Delta } m _ { \Delta } \ \leq \ m$ and every heavy $\Delta$ has $m _ { \Delta } > T n _ { \kappa } ^ { 1 / k }$ , we have $| \mathcal { D } _ { \mathrm { h } } | \leq m / ( { \varGamma } n _ { \kappa } ^ { 1 / k } ) . ~ \mathrm { B y } \ : n _ { \Delta } \leq n _ { \kappa }$ and by the concavity of the power function with exponent $k / ( k + 1 )$ , under which $\begin{array} { r } { \sum _ { \Delta \in \mathcal { D } _ { \mathrm { h } } } m _ { \Delta } ^ { k / ( k + 1 ) } \le | \mathcal { D } _ { \mathrm { h } } | ^ { 1 / ( k + 1 ) } \big ( \sum _ { \Delta \in \mathcal { D } _ { \mathrm { h } } } m _ { \Delta } \big ) ^ { k / ( k + 1 ) } } \end{array}$

$$
\begin{array} { r l } & { \displaystyle \sum _ { \Delta \mathrm { \scriptsize ~ h e a v y } } ( m _ { \Delta } n _ { \Delta } ) ^ { k / ( k + 1 ) } \le n _ { \kappa } ^ { k / ( k + 1 ) } | \mathcal D _ { \mathrm { h } } | ^ { 1 / ( k + 1 ) } m ^ { k / ( k + 1 ) } } \\ & { \qquad \le n _ { \kappa } ^ { k / ( k + 1 ) } \Big ( \displaystyle \frac { m } { { \cal T } n _ { \kappa } ^ { 1 / k } } \Big ) ^ { 1 / ( k + 1 ) } m ^ { k / ( k + 1 ) } = { \cal T } ^ { - 1 / ( k + 1 ) } m n _ { \kappa } ^ { ( k - 1 ) / k } , } \end{array}
$$

where we used $k / ( k + 1 ) - 1 / ( k ( k + 1 ) ) = ( k - 1 ) / k$ . Here $r ^ { 1 / ( k + 1 ) } = 4$ , and $m n _ { \kappa } ^ { ( k - 1 ) / k } =$ $m ( n / \kappa ) ^ { ( k - 1 ) / k } \leq \dot { \phi } ( m , n )$ by (7.5), proving the first bound. For the second,

$$
\sum _ { \Delta { \mathrm { ~ h e a v y } } } ( m _ { \Delta } + n _ { \Delta } ) \leq m + | \mathcal { D } _ { \mathrm { h } } | n _ { \kappa } \leq m + \frac { m n _ { \kappa } ^ { ( k - 1 ) / k } } { T } \leq \Phi ( m , n ) + \frac { \varPhi ( m , n ) } { T } ,
$$

by (7.5). Finally, $m _ { \Delta } n _ { \Delta } \le m n _ { \kappa } = m n / \kappa \le m n / 2$ , because $\kappa \geq 2$

Summing recursive costs. For a recursion satisfying the child-size bounds of Lemma 7.10, the sum of the potentials at depth ℓ is at most $4 ^ { - \ell } \varPhi ( m , n )$ , where m, n are the root sizes. Consequently,

$$
\sum _ { v } \varPhi ( m _ { v } , n _ { v } ) \leq \frac { 4 } { 3 } \varPhi ( m , n ) , \qquad \sum _ { v } ( m _ { v } + n _ { v } ) \leq m + n +  { \operatorname { O } _ { d } } \biggl ( \sum _ { v \mathrm { ~ i n t e r m a l } } \varPhi ( m _ { v } , n _ { v } ) \biggr ) .
$$

The recursion has finite depth because $m _ { \Delta } n _ { \Delta } \leq m _ { v } n _ { v } / 2$ . If each node has conditional expected cost $\widetilde { \mathrm { O } } ( \varPhi ( m _ { v } , n _ { v } ) + m _ { v } + n _ { v } )$ with uniform numerical factors, summing these bounds and taking expectations establishes total expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ .

## B.6 Implementing Batch Maximum Inner Product

Finally, we prove Proposition 5.1 for $d \ge 4 \ ( k \ge 2 )$ by adapting the envelope-sampling and heavy/light recursion framework of Section 7.5 to exact maximization. Working in the setup of Definition 6.4 with sources ${ \pmb y } _ { s } = ( { \pmb p } _ { s } , c _ { s } ) , s \in \Sigma _ { 0 }$ and a nonempty finite list of rational targets $( \pmb { u } _ { i } ) _ { i \in \Theta _ { 0 } } \mathrm { i n } [ - 1 , 1 ] ^ { d - 1 }$ , we use the sampling parameters (6.1) with $N \geq | \Sigma _ { 0 } | + | \Theta _ { 0 } |$ and bit bound $\lambda \geq \log _ { 2 } ( N + 8 )$ . Both procedures below take nonempty subsets $\Theta \subseteq \Theta _ { 0 }$ and $\Sigma \subseteq \Sigma _ { 0 }$ and return the exact maximum of $f _ { s } ( { \pmb u } _ { i } ) = { \pmb u } _ { i } \cdot { \pmb p } _ { s } + c _ { s }$ over $s \in \Sigma$ for every $i \in \Theta$

As in Section 7.1, calls restrict the original point sets and share one perturbation and precision. Appendix A.5 provides $\mathrm { O } _ { d } ( \lambda )$ -bit representations for their geometric quantities.

Procedure ThinM $\operatorname { A X } ( \Theta , \Sigma )$ . We permute Σ uniformly at random and process doubling prefixes, from one source to all of $\Sigma ,$ maintaining for each target $i \in \Theta$ its current maximum $h _ { i }$ and a maximizing index. By duality, the reporting structure is built on lifted targets and queried by sources. In each round, we apply Lemma 6.3 with group size $g _ { \mathrm { r e p } } = | \Theta |$ to the points $( { \bf { u } } _ { i } , h _ { i } )$ , then query with direction $( p _ { s } , - 1 )$ and threshold $- c _ { s }$ for each source s in the next batch. The values $h _ { i }$ stay fixed during these reports; afterward, we update the maxima and maximizing indices using all reported values.

Lemma B.8. Procedure ThinMax(Θ, Σ) returns ma $\boldsymbol { \mathfrak { r } } _ { s \in \Sigma } f _ { s } ( \boldsymbol { \mathbf { \mathit { u } } } _ { i } )$ and a maximizing index for every $i \in \Theta$ , in expected running time $\widetilde { \mathrm { O } } ( | \Theta | ^ { k } + | \Sigma | )$ .

Proof. The reporting inequality is exactly $f _ { s } ( { \pmb u } _ { i } ) > h _ { i } .$ , so each update computes the maximum over the enlarged prefix. For the cost, let $q$ and $q _ { + } \leq 2 q$ be the sizes of two consecutive prefixes. Conditional on the larger prefix, its preceding prefix is a uniform $q \mathrm { - }$ -element subset. For a fixed target, order these $q _ { + }$ sources by decreasing value, breaking ties by index. Every reported source precedes the first sampled source, so the expected number reported is at most

$$
{ \frac { q _ { + } - q } { q _ { + 1 } } } \leq 1 .
$$

Ties can only reduce this count. There are ${ \mathrm { O } } ( \log ( N + 2 ) )$ rounds, each preprocessing |Θ| points in expected time $\widetilde { \mathrm { O } } ( | \Theta | ^ { k } )$ . Each source is queried at most once, and the expected total number of reports is ${ \mathrm { O } } ( | \Theta | \log ( N + 2 ) )$ ), proving the bound. □

Procedure Maxima(Θ, Σ). Let $m = | \Theta |$ and $n = | \Sigma |$ , both positive. If $m ^ { k } \leq n .$ , run ThinMax and return. If $n ^ { k } \le ( 4 C _ { \mathrm { s a m p } } G ) ^ { k } m$ , use the support-query structure of Lemma 6.2 to answer every target directly and return.

Otherwise, compute $\kappa , n _ { \kappa } , s _ { f }$ from Section 7.5 and build the reporting structure on the perturbed sources with group size $\kappa .$ Here $s _ { f } < n$ by (7.6). Repeatedly draw a uniform sample of size $s _ { f } ,$ , build its envelope cells, and compute each conflict set by the d vertex reports of Lemma $6 . 7 ( \mathrm { b } )$ . Reject the sample at the first cell with $| C _ { \Delta } | > n _ { \kappa } / 2$ , and otherwise accept it. Locate the targets in the accepted cells, obtaining groups $\Theta _ { \Delta }$ . For each nonempty group, return the larger of the owner’s original value and the maximum over $C _ { \Delta }$ . An empty conflict set needs no further call; otherwise use ThinMax when $| \Theta _ { \Delta } | ^ { k } \le { \varGamma } ^ { k } n _ { \kappa }$ and recurse with Maxima when this inequality fails.

Lemma B.9. Procedure Maxim $\operatorname { \ a u } ( \Theta , \Sigma )$ is a Las Vegas algorithm returning ma $\mathrm { x } _ { s \in \Sigma } f _ { s } ( { \pmb u } _ { i } )$ and a maximizing index for every $i \in \Theta$ . Its expected running time is $\widetilde { \mathrm { O } } ( \varPhi ( m , n ) + m + n )$ , where $m = | \Theta |$ and $n = | \Sigma |$

Proof. For a target in cell $\Delta$ , the owner or some source in $C _ { \Delta }$ maximizes the perturbed score over Σ and, by Lemma 6.5(b), also maximizes the original score. Thus

$$
\operatorname* { m a x } _ { s \in \Sigma } f _ { s } ( \pmb { u } _ { i } ) = \operatorname* { m a x } \Big \{ f _ { s _ { \Delta } } ( \pmb { u } _ { i } ) , \ \operatorname* { m a x } _ { s \in C _ { \Delta } } f _ { s } ( \pmb { u } _ { i } ) \Big \} ,
$$

where max $\mathcal { O } = - \infty$ . This proves correctness by induction on mn, since Lemma 7.10 halves this product in every recursive child.

Both thin cases run in expected time $\widetilde { \mathrm { O } } ( m + n )$ . At a balanced node, the reporting structure requires expected preprocessing time $\widetilde { \mathrm { O } } ( n \kappa ^ { k - 1 } )$ Each sampling attempt has $\widetilde { \mathrm { O } } ( \kappa ^ { k } )$ cells, with reporting cost $\widetilde { \mathrm { O } } ( n _ { \kappa } )$ per cell passing the conflict check and at most $\widetilde { \mathrm { O } } ( n )$ for the first failing cell. The expected cost per attempt is thus $\widetilde { \mathrm { O } } ( n \kappa ^ { k - 1 } )$ . Since each attempt is accepted with probability at least $7 / 8$ (Lemma 6.9), the expected total over all attempts has the same order. Each light cell costs $\widetilde { \mathrm { O } } ( n _ { \kappa } )$ in expectation by Lemma B.8; locating targets and returning answers adds $\widetilde { \mathrm { O } } ( m )$ . By (7.5), the total nonrecursive work is $\widetilde { \mathrm { O } } ( \varPhi ( m , n ) + m + n )$

The target groups are disjoint and every conflict set has size at most $n _ { \kappa } / 2$ . Summing over the recursion using Lemma 7.10 and Appendix B.5 yields total expected time $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ . □

Proof of Proposition 5.1. For $d \leq 3$ , the structure of Lemma 6.2 answers all queries deterministically in $\widetilde { \mathrm { O } } ( m + n )$ time. For $d \geq 4 ,$ , partition the nonzero queries into the at most 2d groups of Definition 5.2. In each group π, apply Maxima to sources $\pi \boldsymbol { k } _ { s } ~ ( \boldsymbol { s } \in [ n ] )$ and targets ${ \pmb u } _ { { \pmb q } _ { i } }$ , with $U = 2 ^ { \lambda }$ for an input bit bound λ. Each returned index maximizes the original inner product because $\pmb q _ { i } \cdot \pmb k _ { s } = \beta _ { \pmb q _ { i } } f _ { k _ { s } } ^ { \pi } ( \pmb u _ { \pmb q _ { i } } )$ and $\beta _ { q _ { i } } > 0$ . Summing the running-time bounds in Lemma B.9 over the groups and evaluating the selected inner products exactly requires $\widetilde { \mathrm { O } } ( F _ { d } ( m , n ) )$ time. □

## AI Methodology

We used DeepSeek Harness to assist with the proof of Proposition 5.6. The tool materially afected Section 7. More details can be found in the footnote in Section 1.2.3. The authors verified the correctness and originality of all content, including references.

## Acknowledgment

This work was supported by the National Key R&D Program of China, Project Number 2025YFA1016700. This work was also supported by the National Natural Science Foundation of China (NSFC) under Grant No. 62576011.

## References

[AAD+24] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, Sebastian W. Bodenstein, David A. Evans, Chia-Chun Hung, Michael O’Neill, David Reiman, Kathryn Tunyasuvunakool, Zachary Wu, Akvil˙e Žemgulyt˙e, Eirini Arvaniti, Charles Beattie, Ottavia Bertolli, Alex Bridgland, Alexey Cherepanov, Miles Congreve, Alexander I. Cowen-Rivers, Andrew Cowie, Michael Figurnov, Fabian B. Fuchs, Hannah Gladman, Rishub Jain, Yousuf A. Khan, Caroline M. R. Low, Kuba Perlin, Anna Potapenko, Pascal Savy, Sukhdeep Singh, Adrian Stecula, Ashok Thillaisundaram, Catherine Tong, Sergei Yakneen, Ellen D. Zhong, Michal Zielinski, Augustin Žídek, Victor Bapst, Pushmeet Kohli, Max Jaderberg, Demis Hassabis, and John M. Jumper. “Accurate structure prediction of biomolecular interactions with AlphaFold 3”. In: Nature 630.8016 (June 2024), pp. 493–500. issn: 1476-4687. doi: 10.1038/s41586-024-07487-w.

[Aga17] Pankaj K. Agarwal. “Range Searching”. In: Handbook of Discrete and Computational Geometry. Ed. by Jacob E. Goodman, Joseph O’Rourke, and Csaba D. Tóth. 3rd ed. CRC Press, 2017. Chap. 40. doi: 10.1201/9781315119601.

[AE99] Pankaj K. Agarwal and Jef Erickson. “Geometric Range Searching and Its Relatives”. In: Advances in Discrete and Computational Geometry. Ed. by Bernard Chazelle, Jacob E. Goodman, and Richard Pollack. Vol. 223. Contemporary Mathematics. American Mathematical Society, 1999, pp. 1–56. doi: 10.1090/conm/223/03131.

[ACSS20] Josh Alman, Timothy Chu, Aaron Schild, and Zhao Song. “Algorithms and Hardness for Linear Algebra on Geometric Graphs”. In: 61st IEEE Annual Symposium on Foundations of Computer Science, FOCS 2020, Durham, NC, USA, November 16-19, 2020. Ed. by Sandy Irani. IEEE, 2020, pp. 541–552. doi: 10.1109/FOCS46700.2020.00057. url: https://doi.org/10.1109/FOCS46700.2020.00057.

[AS23] Josh Alman and Zhao Song. “Fast Attention Requires Bounded Entries”. In: Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023. Ed. by Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine. 2023. url:

http://papers.nips.cc/paper\_files/paper/2023/hash/ c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html.

[AS24] Josh Alman and Zhao Song. “The Fine-Grained Complexity of Gradient Computation for Training Large Language Models”. In: Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024. 2024. url: http://papers.nips.cc/paper\_files/paper/2024/hash/ 70fa5df8e3300dc30bf19bee44a56155-Abstract-Conference.html.

[AY25] Josh Alman and Hantao Yu. “Fundamental Limitations on Subquadratic Alternatives to Transformers”. In: The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. url: https://openreview.net/forum?id=T2d0geb6y0.

[AH08] Boris Aronov and Sariel Har-Peled. “On Approximating the Depth and Related Problems”. In: SIAM J. Comput. 38.3 (2008), pp. 899–921. doi: 10.1137/060669474. url: https://doi.org/10.1137/060669474.

[AMX12] Sunil Arya, David M. Mount, and Jian Xia. “Tight Lower Bounds for Halfspace Range Searching”. In: Discret. Comput. Geom. 47.4 (2012), pp. 711–730. doi: 10.1007/S00454-012-9412-X. url: https://doi.org/10.1007/s00454-012-9412-x.

[BPR06] Saugata Basu, Richard Pollack, and Marie-Françoise Roy. Algorithms in real algebraic geometry. Springer, 2006.

[Cha96a] Timothy M. Chan. “Fixed-Dimensional Linear Programming Queries Made Easy”. In: Proceedings of the Twelfth Annual Symposium on Computational Geometry, Philadelphia, PA, USA, May 24-26, 1996. Ed. by Sue Whitesides. ACM, 1996, pp. 284–290. doi: 10.1145/237218.237397. url: https://doi.org/10.1145/237218.237397.

[Cha96b] Timothy M. Chan. “Output-Sensitive Results on Convex Hulls, Extreme Points, and Related Problems”. In: Discret. Comput. Geom. 16.4 (1996), pp. 369–387. doi: 10.1007/BF02712874. url: https://doi.org/10.1007/BF02712874.

[Cha12] Timothy M. Chan. “Optimal Partition Trees”. In: Discret. Comput. Geom. 47.4 (2012), pp. 661–690. doi: 10.1007/S00454-012-9410-Z. url: https://doi.org/10.1007/s00454-012-9410-z.

[CZ22] Timothy M. Chan and Da Wei Zheng. “Hopcroft’s Problem, Log-Star Shaving, 2D Fractional Cascading, and Decision Trees”. In: Proceedings of the 2022 ACM-SIAM Symposium on Discrete Algorithms, SODA 2022, Virtual Conference / Alexandria, VA, USA, January 9 - 12, 2022. Ed. by Joseph (Sefi) Naor and Niv Buchbinder. SIAM, 2022, pp. 190–210. doi: 10.1137/1.9781611977073.10. url: https://doi.org/10.1137/1.9781611977073.10.

[Cha93] Bernard Chazelle. “An Optimal Convex Hull Algorithm in Any Fixed Dimension”. In: Discret. Comput. Geom. 10 (1993), pp. 377–409. doi: 10.1007/BF02573985. url: https://doi.org/10.1007/BF02573985.

[Cha97] Bernard Chazelle. “Lower Bounds for Of-Line Range Searching”. In: Discret. Comput. Geom. 17.1 (1997), pp. 53–65. doi: 10.1007/BF02770864. url: https://doi.org/10.1007/BF02770864.

[CEGS91] Bernard Chazelle, Herbert Edelsbrunner, Leonidas J. Guibas, and Micha Sharir. “A Singly Exponential Stratification Scheme for Real Semi-Algebraic Varieties and its Applications”. In: Theor. Comput. Sci. 84.1 (1991), pp. 77–105. doi: 10.1016/0304-3975(91)90261-Y. url: https://doi.org/10.1016/0304-3975(91)90261-Y.

[CF90] Bernard Chazelle and Joel Friedman. “A deterministic view of random sampling and its use in geometry”. In: Comb. 10.3 (1990), pp. 229–249. doi: 10.1007/BF02122778. url: https://doi.org/10.1007/BF02122778.

[CLSSS25] Bo Chen, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, and Zhao Song. “HSR-Enhanced Sparse Attention Acceleration”. In: Conference on Parsimony and Learning, Stanford University, USA, 24-27 March 2025. Ed. by Beidi Chen, Shijia Liu, Mert Pilanci, Weijie Su, Jeremias Sulam, Yuxiang Wang, and Zhihui Zhu. Vol. 280. Proceedings of Machine Learning Research. PMLR, 2025, pp. 105–133. url: https://proceedings.mlr.press/v280/chen25a.html.

[Che20] Lijie Chen. “On The Hardness of Approximate and Exact (Bichromatic) Maximum Inner Product”. In: Theory Comput. 16 (2020), pp. 1–50. doi: 10.4086/TOC.2020.V016A004. url: https://doi.org/10.4086/toc.2020.v016a004.

[CLD+21] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. “Rethinking Attention with Performers”. In: 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. url: https://openreview.net/forum?id=Ua6zuk0WRH.

[Cla87] Kenneth L. Clarkson. “New Applications of random Sampling in Computational Geometry”. In: Discret. Comput. Geom. 2 (1987), pp. 195–222. doi: 10.1007/BF02187879. url: https://doi.org/10.1007/BF02187879.

[Cla88] Kenneth L. Clarkson. “A Randomized Algorithm for Closest-Point Queries”. In: SIAM J. Comput. 17.4 (1988), pp. 830–847. doi: 10.1137/0217052. url: https://doi.org/10.1137/0217052.

[CS89] Kenneth L. Clarkson and Peter W. Shor. “Applications of Random Sampling in Computational Geometry, II”. In: Discrete & Computational Geometry 4.5 (Oct. 1, 1989), pp. 387–421. issn: 1432-0444. doi: 10.1007/BF02187740. url: https://doi.org/10.1007/BF02187740.

[DFERR22] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. “FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness”. In: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022. Ed. by Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh. 2022. url: http://papers.nips.cc/paper\_files/paper/2022/hash/ 67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html.

[DLMF] NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/, Release 1.2.8 of 2026-09-15. F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl, and M. A. McClain, eds. url: https://dlmf.nist.gov/.

[DK90] David P. Dobkin and David G. Kirkpatrick. “Determining the Separation of Preprocessed Polyhedra - A Unified Approach”. In: Automata, Languages and Programming, 17th International Colloquium, ICALP90, Warwick University, England, UK, July 16-20, 1990, Proceedings. Ed. by Mike Paterson. Vol. 443. Lecture Notes in Computer Science. Springer, 1990, pp. 400–413. doi: 10.1007/BFB0032047. url: https://doi.org/10.1007/BFb0032047.

[DBK+21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”. In: 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. url: https://openreview.net/forum?id=YicbFdNTTy.

[DSST89] James R. Driscoll, Neil Sarnak, Daniel Dominic Sleator, and Robert Endre Tarjan. “Making Data Structures Persistent”. In: J. Comput. Syst. Sci. 38.1 (1989), pp. 86–124. doi: 10.1016/0022-0000(89)90034-2. url: https://doi.org/10.1016/0022-0000(89)90034-2.

[DWH23] Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. “On The Computational Complexity of Self-Attention”. In: International Conference on Algorithmic Learning Theory, February 20-23, 2023, Singapore. Ed. by Shipra Agrawal and Francesco Orabona. Vol. 201. Proceedings of Machine Learning Research. PMLR, 2023, pp. 597–619. url: https://proceedings.mlr.press/v201/duman-keles23a.html.

[EM90] Herbert Edelsbrunner and Ernst P. Mücke. “Simulation of simplicity: a technique to cope with degenerate cases in geometric algorithms”. In: ACM Trans. Graph. 9.1 (1990), pp. 66–104. doi: 10.1145/77635.77639. url: https://doi.org/10.1145/77635.77639.

[EC95] Ioannis Z. Emiris and John F. Canny. “A General Approach to Removing Degeneracies”. In: SIAM J. Comput. 24.3 (1995), pp. 650–664. doi: 10.1137/S0097539792235918. url: https://doi.org/10.1137/S0097539792235918.

[Eri96] J. Erickson. “New lower bounds for halfspace emptiness”. In: Proceedings of 37th Conference on Foundations of Computer Science. 1996, pp. 472–481. doi: 10.1109/SFCS.1996.548506.

[Eri00] Jef Erickson. “Space-Time Tradeofs for Emptiness Queries”. In: SIAM J. Comput. 29.6 (2000), pp. 1968–1996. doi: 10.1137/S0097539798337212. url: https://doi.org/10.1137/S0097539798337212.

[Fen94] Peter M. Fenwick. “A New Data Structure for Cumulative Frequency Tables”. In: Softw. Pract. Exp. 24.3 (1994), pp. 327–336. doi: 10.1002/SPE.4380240306. url: https://doi.org/10.1002/spe.4380240306.

[GR87] L Greengard and V Rokhlin. “A fast algorithm for particle simulations”. In: Journal of Computational Physics 73.2 (1987), pp. 325–348. issn: 0021-9991. doi: 10.1016/0021-9991(87)90140-9. url: https://www.sciencedirect.com/science/article/pii/0021999187901409.

[GS91] Leslie Greengard and John Strain. “The Fast Gauss Transform”. In: SIAM J. Sci. Comput. 12.1 (1991), pp. 79–94. doi: 10.1137/0912004. url: https://doi.org/10.1137/0912004.

[GHSXY26] Shreya Gupta, Boyang Huang, Barna Saha, Yinzhan Xu, and Christopher Ye. “Subquadratic Algorithms and Hardness for Attention with Any Temperature”. In: The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. Ed. by Carl Vondrick, Bharath Hariharan, Colin Rafel, Lerrel Pinto, Diyi Yang, and Aleksandra Faust. proceedings.iclr.cc, 2026. url: https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 8a01099096c85890b1d1aff3c6b4ea56-Abstract-Conference.html.

[HJK+24] Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P. Woodruf, and Amir Zandieh. “HyperAttention: Long-context Attention in Near-Linear Time”. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. url: https://openreview.net/forum?id=Eh0Od2BJIM.

[HW87] David Haussler and Emo Welzl. “ε-Nets and Simplex Range Queries”. In: Discrete & Computational Geometry 2.2 (June 1, 1987), pp. 127–151. issn: 1432-0444. doi: 10.1007/BF02187876. url: https://doi.org/10.1007/BF02187876.

[Hoe63] Wassily Hoefding. “Probability Inequalities for Sums of Bounded Random Variables”. In: Journal of the American Statistical Association 58.301 (1963), pp. 13–30.

[IP01] Russell Impagliazzo and Ramamohan Paturi. “On the Complexity of k-SAT”. In: J. Comput. Syst. Sci. 62.2 (2001), pp. 367–375. doi: 10.1006/JCSS.2000.1727. url: https://doi.org/10.1006/jcss.2000.1727.

[Mat92a] Jirí Matousek. “Reporting Points in Halfspaces”. In: Comput. Geom. 2 (1992), pp. 169–186. doi: 10.1016/0925-7721(92)90006-E. url: https://doi.org/10.1016/0925-7721(92)90006-E.

[Mat93] Jirí Matousek. “Linear Optimization Queries”. In: J. Algorithms 14.3 (1993), pp. 432–448. doi: 10.1006/JAGM.1993.1023. url: https://doi.org/10.1006/jagm.1993.1023.

[Mat94] Jirí Matousek. “Geometric Range Searching”. In: ACM Comput. Surv. 26.4 (1994), pp. 421–461. doi: 10.1145/197405.197408. url: https://doi.org/10.1145/197405.197408.

[Mat92b] Jiří Matoušek. “Eficient Partition Trees”. In: Discrete & Computational Geometry 8.3 (Aug. 1992), pp. 315–334. issn: 1432-0444. doi: 10.1007/BF02293051.

[MS93] Jirí Matousek and Otfried Schwarzkopf. “On Ray Shooting in Convex Polytopes”. In: Discret. Comput. Geom. 10 (1993), pp. 215–232. doi: 10.1007/BF02573975. url: https://doi.org/10.1007/BF02573975.

[Mau26] Samuel Mausberg. Near-Linear Attention in Three Dimensions. Version v2. Oct. 2026. doi: 10.5281/zenodo.23195937. url: https://doi.org/10.5281/zenodo.23195937.

[McM70] P. McMullen. “The maximum numbers of faces of a convex polytope”. In: Mathematika 17.2 (1970), pp. 179–184. doi: 10.1112/S0025579300002850.

[Meg84] Nimrod Megiddo. “Linear Programming in Linear Time When the Dimension Is Fixed”. In: J. ACM 31.1 (1984), pp. 114–127. doi: 10.1145/2422.322418. url: https://doi.org/10.1145/2422.322418.

[PS85] Franco P. Preparata and Michael Ian Shamos. Computational Geometry : An Introduction. Third corrected printing, 1990. Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.

[Ram99] Edgar A. Ramos. “On Range Reporting, Ray Shooting and k-Level Construction”. In: Proceedings of the Fifteenth Annual Symposium on Computational Geometry, Miami Beach, Florida, USA, June 13-16, 1999. Ed. by Victor Milenkovic. ACM, 1999, pp. 390–399. doi: 10.1145/304893.304993. url: https://doi.org/10.1145/304893.304993.

[Ram00] Edgar A. Ramos. “Linear programming queries revisited”. In: Proceedings of the Sixteenth Annual Symposium on Computational Geometry, Clear Water Bay, Hong Kong, China, June 12-14, 2000. Ed. by Siu-Wing Cheng, Otfried Cheong, Pankaj K. Agarwal, and Steven Fortune. ACM, 2000, pp. 176–181. doi: 10.1145/336154.336198. url: https://doi.org/10.1145/336154.336198.

[SXY26] Barna Saha, Yinzhan Xu, and Christopher Ye. “Furthest Pair Requires Quadratic Time in Superconstant Dimension under SETH”. In: CoRR abs/2606.25887 (2026). doi: 10.48550/ARXIV.2606.25887. arXiv: 2606.25887. url: https://doi.org/10.48550/arXiv.2606.25887.

[SXYY26] Barna Saha, Yinzhan Xu, Christopher Ye, and Hantao Yu. “On the Computational Hardness of Transformers”. In: Proceedings of the 58th Annual ACM Symposium on Theory of Computing, STOC 2026, Salt Lake City, UT, USA, June 22-26, 2026. Ed. by Aditya Bhaskara and Artur Czumaj. ACM, 2026, pp. 1073–1084. doi: 10.1145/3798129.3800819. url: https://doi.org/10.1145/3798129.3800819.

[Sei17] Raimund Seidel. “Convex hull computations”. In: Handbook of discrete and computational geometry. Chapman and Hall/CRC, 2017, pp. 687–703.

[Sha19] Noam Shazeer. “Fast Transformer Decoding: One Write-Head is All You Need”. In: CoRR abs/1911.02150 (2019). doi: 10.48550/ARXIV.1911.02150. arXiv: 1911.02150. url: https://doi.org/10.48550/arXiv.1911.02150.

[VSP+17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. “Attention is All you Need”. In: Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA. Ed. by Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett. 2017, pp. 5998–6008. url: https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[Wil05] Ryan Williams. “A new algorithm for optimal 2-constraint satisfaction and its implications”. In: Theor. Comput. Sci. 348.2-3 (2005), pp. 357–365. doi: 10.1016/J.TCS.2005.09.023. url: https://doi.org/10.1016/j.tcs.2005.09.023.

[Wil18] Ryan Williams. “On the Diference Between Closest, Furthest, and Orthogonal Pairs: Nearly-Linear vs Barely-Subquadratic Complexity”. In: Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2018, New Orleans, LA, USA, January 7-10, 2018. Ed. by Artur Czumaj. SIAM, 2018, pp. 1207–1215. doi: 10.1137/1.9781611975031.78. url: https://doi.org/10.1137/1.9781611975031.78.

[Yap97] Chee-Keng Yap. “Towards Exact Geometric Computation”. In: Comput. Geom. 7 (1997), pp. 3–23. doi: 10.1016/0925-7721(95)00040-2. url: https://doi.org/10.1016/0925-7721(95)00040-2.

[ZHDK23] Amir Zandieh, Insu Han, Majid Daliri, and Amin Karbasi. “KDEformer: Accelerating Transformers via Kernel Density Estimation”. In: International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA. Ed. by Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett. Vol. 202. Proceedings of Machine Learning Research. PMLR, 2023, pp. 40605–40623. url: https://proceedings.mlr.press/v202/zandieh23a.html.

[Zie95] Günter M. Ziegler. Lectures on polytopes. Graduate texts in mathematics; 152. New York: Springer, 1995.