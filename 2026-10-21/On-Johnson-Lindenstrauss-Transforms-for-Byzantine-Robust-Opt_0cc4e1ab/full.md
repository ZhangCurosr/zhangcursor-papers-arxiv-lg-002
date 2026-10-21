# On Johnson-Lindenstrauss Transforms for Byzantine-Robust Optimization

Aleksandr Shestakov<sup>1,</sup> <sup>3</sup>, Vladimir Aletov<sup>1,</sup> <sup>2</sup>, Aleksandr Beznosikov<sup>1,</sup> <sup>2</sup>

<sup>1</sup>Basic Research of Artificial Intelligence Laboratory

<sup>2</sup>Federated Learning Problems Laboratory

<sup>3</sup>MBZUAI

The growing scale of data and models in machine learning calls for eficient optimization methods. Distributed optimization addresses this challenge, with compression mechanisms helping to reduce communication costs. A classical tool for dimensionality reduction is the Johnson-Lindenstrauss (JL) transform, which is simple to implement and ofers strong geometric guarantees. In particular, JL transforms approximately preserve inner products and pairwise distances between vectors with high probability. In this paper, we exploit this property in Byzantine-robust optimization, where worker messages are weighted according to their alignment with a trusted reference gradient. We further support our theoretical findings with experimental results.

## Introduction

The growing scale of data and computational demands of machine learning [1] have motivated distributed optimization methods [2] for the global objective<sub>v</sub>

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { n } } \left\{ f ( x ) = \frac { 1 } { G } \sum _ { i \in \mathcal { G } } f _ { i } ( x ) \right\} ,\tag{1}
$$

where0 $f _ { i }$ denotes the local loss at worker i, ${ \mathcal { G } } \subseteq \{ 1 , \ldots , M \}$ is the fixed set of honest workers, and $G = | \mathcal { G } | \ge 1$ The objective is the average loss over the honest workers, whose identities need not be known to the server; when all workers are honest, it reduces to the usual average over all M workers. Distributed methods make large-scale2 training tractable when a single machine is insuficient, but introduce communication bottlenecks, synchronization<sup>:</sup> overhead, and bandwidth limitations that can substantially hinder performance [3].<sup>i</sup>

<sub>To</sub> <sub>mitigate</sub> <sub>the</sub> <sub>communication</sub> <sub>cost,</sub> <sub>a</sub> <sub>broad</sub> <sub>range</sub> <sub>of</sub> <sub>compression</sub> <sub>techniques</sub> <sub>has</sub> <sub>been</sub> <sub>proposed</sub> <sub>[4,</sub> <sub>5],</sub> <sub>such</sub>X as quantization [6], which lowers numerical precision, and sparsification [7], which communicates only the mosta informative coordinates. Workers transmit either compressed gradients [5] or auxiliary sequences that compensate for compression errors across iterations [8, 9, 10]. In both cases, the gradient estimator is obtained by averaging compressed vectors.

However, some distributed setups rely not on the mean itself, but on geometric relations between gradients. A prominent example is Byzantine-robust optimization [11], in which Byzantine workers attempt to derail training by sending arbitrary adversarial messages. One remedy is to assign reliability weights to workers [12], computed from inner products of local gradients with a trusted reference gradient [13, 14], so that workers not aligned with the reference direction are suppressed:

$$
\omega _ { i } ^ { t } = \frac { \displaystyle \operatorname* { m a x } _ { M } \left\{ \left. \nabla f _ { i } ( x ^ { t } ) , \nabla f _ { 0 } ( x ^ { t } ) \right. , 0 \right\} } { \displaystyle \sum _ { j = 1 } ^ { M } \operatorname* { m a x } \left\{ \left. \nabla f _ { j } ( x ^ { t } ) , \nabla f _ { 0 } ( x ^ { t } ) \right. , 0 \right\} } , \quad x ^ { t + 1 } = x ^ { t } - \gamma \sum _ { i = 1 } ^ { M } \omega _ { i } ^ { t } \nabla f _ { i } ( x ^ { t } ) ,\tag{2}
$$

where $f _ { 0 }$ is a loss function defined on a trusted device. In this illustrative update, the iterate is left unchanged if the denominator is zero, and Byzantine workers may replace their gradients by arbitrary messages.

Such methods depend on preserving ℓ -geometry rather than exact coordinate values, which motivates compression operators that approximately preserve Euclidean structure. Johnson–Lindenstrauss (JL) transforms [15] map high dimensional vectors into a lower-dimensional space while approximately preserving pairwise distances and inner products. We focus on stochastic JL transforms, as they are simple to implement in practice. They have been successfully applied in nearest-neighbor search [16, 17], random partitioning [18], linear regression [19, 20], clustering [21, 22, 23, 24, 25, 26], diferential privacy [27, 28], and numerical linear algebra [29, 30, 31].

In distributed optimization and federated learning, however, JL transforms have mostly been treated as unbiased sketching operators [32, 33, 34], leading to guarantees in expectation, or derandomized [35]. These works do not capitalize on the preservation of distances and inner products, and thus do not clearly distinguish JL transforms from standard sparsification or sketching schemes.

Our contributions. In this paper, we

• Propose a JL-based communication mechanism for Byzantine-robust distributed optimization methods tha rely on gradient alignment with a trusted reference gradient;

• Use high-probability control of compressed honest gradients to derive convergence guarantees without convexity, under average data similarity and an explicit accuracy condition on the trusted reference gradient;

• Validate our theoretical findings experimentally and demonstrate the practical efectiveness of the proposed approach.

## 2 Proposed Algorithm

## 2.1 Notation

We equip $\mathbb { R } ^ { n }$ with the standard Euclidean norm, $\left\| x \right\| \ { \overset { \underset { \mathrm { d e f } } { } } { = } } \ \left. x , x \right. ^ { 1 / 2 } , x \in \mathbb R ^ { n }$ . Throughout the paper, $f : \mathbb { R } ^ { n } $ R denotes a diferentiable objective function.

Assumption 1 (Smoothness)

Diferentiable function $f _ { 0 }$ has L-Lipschitz gradient for some $L > 0$ , i.e.

$$
\begin{array} { r } { \| \nabla f _ { 0 } ( x ) - \nabla f _ { 0 } ( y ) \| \leq L \| x - y \| \quad \forall x , y \in \mathbb { R } ^ { n } . } \end{array}\tag{3}
$$

Moreover, $\begin{array} { r } { f _ { 0 } ^ { \mathrm { i n f } } \stackrel { \mathrm { d e f } } { = } \operatorname* { i n f } _ { x \in \mathbb { R } ^ { n } } f _ { 0 } ( x ) > - \infty } \end{array}$

## 2.2 JL Transforms

Definition 1 (JL Transform [15])

Given $\varepsilon , \delta \in ( 0 , 1 )$ ), a stochastic mapping $h : \mathbb { R } ^ { n }  \mathbb { R } ^ { k }$ is called a JL transform if, for all $u , v \in \mathbb { R } ^ { n }$ ,

$$
\begin{array} { r } { \mathbb { P } \left[ ( 1 - \varepsilon ) \| u - v \| ^ { 2 } \leq \| h ( u ) - h ( v ) \| ^ { 2 } \leq ( 1 + \varepsilon ) \| u - v \| ^ { 2 } \right] \geq 1 - \delta . } \end{array}\tag{4}
$$

In words, a JL transform preserves all pairwise distances up to a factor $1 \pm \varepsilon$ with probability at least $1 - \delta$ , while reducing the dimension from n to k. If h is linear, then $h ( 0 ) = 0$ , and taking $v = 0$ shows that norms are preserved as well: $( 1 - \varepsilon ) \| u \| ^ { 2 } \leq \| h ( u ) \| ^ { 2 } \leq ( 1 + \varepsilon ) \| u \| ^ { 2 }$ . Since $\begin{array} { r } { \langle \dot { u } , \dot { v } \rangle = \frac { 1 } { 4 } \left( \lVert u + v \rVert ^ { 2 } - \lVert u - v \rVert ^ { 2 } \right) } \end{array}$ , preserving the norms of $u + v$ and $u - v$ also preserves inner products

```latex
Lemma 1
If h is a linear JL Transform, then $\begin{array} { r } { \mathbb { P } \left[ \mathbf { \Phi } | \left. h ( u ) , h ( v ) \right. - \left. u , v \right. | \leq \varepsilon \| u \| \mathbf { \Phi } \| v \| \right] \geq 1 - 2 \delta } \end{array}$ for all $u , v \in \mathbb { R } ^ { n }$
We consider two standard linear constructions.
Definition 2 (Gaussian JL matrix)
$S \in \mathbb { R } ^ { k \times n }$ with i.i.d. entries $\begin{array} { r } { [ S ] _ { i j } \sim \mathcal { N } ( 0 , \frac { 1 } { k } ) } \end{array}$
Definition 3 (Rademacher JL matrix)
$S \in \mathbb { R } ^ { k \times n }$ with i.i.d. entries $[ S ] _ { i j } = \pm \frac { 1 } { \sqrt { k } }$ , each with probability ${ \frac { 1 } { 2 } } .$
Both are JL transforms with distortion $\varepsilon \in ( 0 , 1 )$ and failure probability $\delta \in ( 0 , 1 )$ whenever $k \geq C \varepsilon ^ { - 2 } \log ( 2 / \delta )$
where $C > 0$ is a universal constant [36, 37]. To return to $\mathbb { R } ^ { n }$ for the optimization update, we use the lifted operator
$\operatorname { J L } _ { S } ( x ) \ { \stackrel { \mathrm { d e f } } { = } } \ S ^ { \top } S x \in \mathbb { R } ^ { n }$
Lemma 2 (Unbiasedness)
For S as in Definitions 2 and 3, $\mathbb { E } { \mathrm { J } } \mathrm { L } _ { S } ( x ) = x$ for all $x \in \mathbb { R } ^ { n }$
Lemma 3 (Second moment)
For all $\begin{array} { r } { x \in \mathbb { R } ^ { n } , \mathbb { E } \| \mathrm { J L } _ { S } ( x ) \| ^ { 2 } = \frac { n + k + 1 } { k } \| x \| ^ { 2 } } \end{array}$ for Gaussian and $\begin{array} { r } { \mathbb { E } \| \mathrm { J } \mathrm { L } _ { S } ( x ) \| ^ { 2 } = \frac { n + k - 1 } { k } \| x \| ^ { 2 } } \end{array}$ for Rademacher S.
Thus $\operatorname { J L } _ { S }$ resembles unbiased sparsification such as Randk, $\begin{array} { r } { \mathcal { C } ( x ) = \frac { n } { k } \sum _ { i \in I } x _ { i } e _ { i } } \end{array}$ with $I \subset [ n ]$ a uniformly random
subset of size $k ,$ which is also unbiased with variance of order $\frac { n } { k } \ \left[ 3 8 \right]$ . The key distinction is that the transmitted
message of Randk, $h c ( x ) \stackrel { \mathrm { d e f } } { = } \sqrt { n / k } ( x _ { i } ) _ { i \in I } \in \mathbb { R } ^ { k }$ (so that $\mathcal { C } = h _ { \mathcal { C } } ^ { \top } h _ { \mathcal { C } }$ , just as $\mathrm { J } \mathrm { L } _ { S } = S ^ { \top } S )$ , does not preserve norms
with high probability. By linearity, the JL property for $h _ { \mathcal { C } }$ reduces to $\mathbb { P } \big [ \big | \| h _ { \mathcal { C } } ( x ) \| ^ { 2 } / \| x \| ^ { 2 } - 1 \big | \leq \varepsilon \big ] ^ { \cdot } \geq 1 - \delta$ for all
$x \in \mathbb { R } ^ { n } \setminus \{ 0 \}$ . For $x = e _ { 1 }$ the ratio is 0 with probability $\textstyle 1 - { \frac { k } { n } }$ , forcing $\begin{array} { r } { \delta \ge 1 - \frac { k } { n } } \end{array}$
Lemma 4 (Operator-norm bound [39])
For S as in Definitions 2 or 3, there exists a universal constant $c > 0$ , independent of n and $k ,$ such that,
with probability at least $1 - \delta$
$\| S \| \leq 1 + c \sqrt { \frac { n } { k } } + c \sqrt { \frac { \log ( 2 / \delta ) } { k } } .$ (5)
Consequently, on the same event, $\begin{array} { r } { \| \mathrm { J L } _ { S } ( x ) \| ^ { 2 } \leq \left( 1 + c \sqrt { \frac { n } { k } } + c \sqrt { \frac { \log ( 2 / \delta ) } { k } } \right) ^ { 4 } \| x \| ^ { 2 } } \end{array}$ for every $x \in \mathbb { R } ^ { n }$
The operator norm controls the lifted update, while the JL property controls the geometry of fixed vectors in the
compressed space; the high-probability analysis uses both.
```

## 2.3 Grad-BANT

Many Byzantine-resilient methods exploit gradient similarity to distinguish honest workers from adversarial ones. We introduce Grad-BANT (Algorithm 1), inspired by [13]. As in equation 2, it relies on a trial function $f _ { 0 }$ stored on the trusted server, whose gradient $\nabla f _ { 0 } ( x )$ is used to score the received messages; its accuracy with respect to the honest objective is specified in Assumption 3.

Algorithm 1: Grad-BANT   
1: Parameters: $x ^ { 0 } \in \mathbb { R } ^ { n } , \omega _ { i } ^ { - 1 } = 1 / M .$ stepsize $\gamma > 0$ , momentum $\beta \in ( 0 , 1 ]$ , smoothness constant $L > 0$   
2: for $t = 0 , 1 , \ldots , T - 1$ do   
3: Generate a fresh $S _ { t } ;$ server sends $x ^ { t }$ and the sketch seed to workers   
4: for all $i = 1 , \ldots , M$ in parallel do   
5: Honest worker: $g _ { i } ^ { t } = \nabla f _ { i } ( x ^ { t } )$ ; send $\boldsymbol y _ { i } ^ { t } = \boldsymbol S _ { t } \boldsymbol g _ { i } ^ { t }$ to server   
6: end for   
7: $\begin{array} { r } { u _ { i } ^ { t } = S _ { t } ^ { \top } y _ { i } ^ { t } ; \theta _ { i } ^ { t } = \left. S _ { t } \nabla f _ { 0 } ( x ^ { t } ) , y _ { i } ^ { t } \right. - \frac { L \gamma } { 2 } \| u _ { i } ^ { t } \| ^ { 2 } } \end{array}$   
8: if all $[ \theta _ { i } ^ { t } ] _ { 0 } = 0$ then   
9: $\begin{array} { r } { \omega _ { i } ^ { t } = ( 1 - \beta ) \omega _ { i } ^ { t - 1 } + \frac { \beta } { M } } \end{array}$   
10: else   
11: $\omega _ { i } ^ { t } = ( 1 - \beta ) \omega _ { i } ^ { t - 1 } + \beta \frac { [ \theta _ { i } ^ { t } ] _ { 0 } } { \Gamma ^ { M } ~ \lceil \theta ^ { t } \rceil _ { \alpha } }$   
$\textstyle \sum _ { j = 1 } ^ { M } [ \theta _ { j } ^ { t } ] _ { \mathrm { ( } }$   
12: end if   
13: $\begin{array} { r } { x ^ { t + 1 } = x ^ { t } - \gamma S _ { t } ^ { \top } \sum _ { i = 1 } ^ { M } \mathbb { I } _ { [ \theta _ { i } ^ { t } > 0 ] } \omega _ { i } ^ { t } y _ { i } ^ { t } } \end{array}$   
14: end for

At each iteration, an honest worker sends $y _ { i } ^ { t } = S _ { t } g _ { i } ^ { t }$ with $g _ { i } ^ { t } = \nabla f _ { i } ( x ^ { t } )$ , while a Byzantine worker may send any finite, measurable $\boldsymbol { y } _ { i } ^ { t } \in \mathbb { R } ^ { k }$ , possibly chosen after observing the current sketch and the honest messages; $u _ { i } ^ { t } = S _ { t } ^ { \top } y _ { i } ^ { t }$ denotes the lifted message. Each sketch is sampled independently of all data and previous randomness after $x ^ { t }$ is determined, so the honest gradients and the reference gradient are fixed conditional on the history before sampling $S _ { t }$

The trust score $\theta _ { i } ^ { t }$ measures how well the lifted message aligns with the trusted reference gradient $\nabla f _ { 0 } ( x ^ { t } )$ , while accounting for its norm: if $\begin{array} { r } { \left. \nabla f _ { 0 } ( x ^ { t } ) , u _ { i } ^ { t } \right. > \frac { L \gamma } { 2 } \| u _ { i } ^ { t } \| ^ { 2 } } \end{array}$ , the direction is suficiently aligned with the trusted descent direction and receives a positive weight; otherwise, it is suppressed by the indicator in the update. The scores are clipped by $[ x ] _ { 0 } \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { m a x } \{ x , 0 \}$ and normalized. If all scores are non-positive, we blend the previous weights with uniform weights (line 9) and leave the iterate unchanged. The momentum parameter $\beta$ smooths the evolution of the weights over time.

For honest workers, the server thus uses the compressed alignment terms $\left. S _ { t } \nabla f _ { 0 } ( x ^ { t } ) , S _ { t } g _ { i } ^ { t } \right.$ , which the JL property keeps close to the exact ones with high probability, while the norm penalty is evaluated on the actual lifted message. The aggregated update is lifted back to $\mathbb { R } ^ { n }$ via $S _ { t } ^ { \top }$ (line 13). Our analysis uses the aggregate honest alignment and the sum of honest squared norms; it does not require the compressed and uncompressed scores to select the same workers. We require the following similarity condition.

```latex
Assumption 2 (Data similarity)
The honest workers are said to satisfy $( \delta _ { 1 } , \delta _ { 2 } )$ -data similarity if there exist $\delta _ { 1 } , \delta _ { 2 } \geq 0$ such that
$\frac { 1 } { G } \sum _ { i \in \mathcal { G } } \left. \nabla f _ { i } ( x ) - \nabla f ( x ) \right. ^ { 2 } \leq \delta _ { 1 } + \delta _ { 2 } \left. \nabla f ( x ) \right. ^ { 2 } , \qquad \forall x \in \mathbb { R } ^ { n } .$ (6)
```

This condition bounds heterogeneity on average over the honest workers, which is weaker than imposing it on every

honest worker. Since f is their average, it also gives

$$
\frac { 1 } { G } \sum _ { i \in \mathcal { G } } \Vert \nabla f _ { i } ( x ) \Vert ^ { 2 } = \Vert \nabla f ( x ) \Vert ^ { 2 } + \frac { 1 } { G } \sum _ { i \in \mathcal { G } } \Vert \nabla f _ { i } ( x ) - \nabla f ( x ) \Vert ^ { 2 } \leq ( 1 + \delta _ { 2 } ) \Vert \nabla f ( x ) \Vert ^ { 2 } + \delta _ { 1 } .\tag{7}
$$

The next assumption quantifies how accurately the trusted reference represents the objective; $\eta _ { 0 }$ and $\eta _ { 1 }$ capture the mismatch between the trial and the true data distributions.

Assumption 3 (Reference accuracy)

There exist $\eta _ { 0 } \geq 0$ and $\eta _ { 1 } \in [ 0 , 1 )$ such that

$$
\begin{array} { r l r } { \| \nabla f _ { 0 } ( x ) - \nabla f ( x ) \| ^ { 2 } \leq \eta _ { 0 } + \eta _ { 1 } \| \nabla f ( x ) \| ^ { 2 } , } & { } & { \forall x \in \mathbb { R } ^ { n } . } \end{array}\tag{8}
$$

Theorem 1

Suppose that Assumptions 1, 2, and 3 hold. Let $T \geq 1$ be an integer, $\beta \in ( 0 , 1 ]$ , and $\varepsilon , \delta \in ( 0 , 1 )$ satisfy $d \ { \stackrel { \mathrm { d e f } } { = } } \ 1 - \eta _ { 1 } - \varepsilon ^ { 2 } > 0$ . Use fresh Gaussian or Rademacher JL matrices in Algorithm 1, and set

$$
\rho = \left( 1 + c \sqrt { \frac { n } { k } } + c \sqrt { \frac { \log ( 8 T / \delta ) } { k } } \right) ^ { 2 } , \qquad \Delta _ { 0 } = f _ { 0 } ( x ^ { 0 } ) - f _ { 0 } ^ { \mathrm { i n f } } ,\tag{9}
$$

where c is the constant in Lemma 4. Then, for any stepsize

$$
0 < \gamma \leq \frac { d } { 2 L \rho ( 1 + \varepsilon ) ( 1 + \delta _ { 2 } ) } ,\tag{10}
$$

after $T$ iterations we have

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \nabla f ( x ^ { t } ) \| ^ { 2 } \leq \frac { 4 M \Delta _ { 0 } } { \gamma \beta G d T } + \frac { 2 \eta _ { 0 } } { d } + \frac { 2 L \gamma \rho ( 1 + \varepsilon ) \delta _ { 1 } } { d } ,\tag{11}
$$

with probability at least $1 - \delta$ , provided that $k \geq C \varepsilon ^ { - 2 } \log ( 8 T / \delta )$ for a suficiently large universal constant $C > 0$

The proof is given in Appendix D. Neither convexity nor a small upper bound on $\delta _ { 2 }$ is required [14]. The reference error contributes an explicit residual term, heterogeneity afects both the admissible stepsize and the last term in equation 11, and the factor $M / ( \beta G )$ reflects the fraction of honest workers and the weight update. With $S _ { t } = I _ { n }$ $( \varepsilon = 0 , \rho = 1 )$ , the same analysis gives the uncompressed guarantee.

## 3 Eficient Implementation

Beyond communication, scalability in large-scale distributed learning is limited by the cost of storing and applying the compression operator: a dense Gaussian or Rademacher matrix $S \in \mathbb { R } ^ { k \times n }$ requires $O ( n k )$ memory and $O ( n k )$ operations per application, which is prohibitive for modern models. We therefore discuss structured and sparse JL constructions, whose geometric guarantees depend on the construction and its parameters, and their use within Grad-BANT.

## 3.1 Eficient JL Transforms

Dense JL matrices can be replaced by the SRHT $S _ { \mathrm { S R H T } } = \sqrt { n / k } R _ { T } H D$ , with a random diagonal sign matrix D, the normalized Walsh–Hadamard matrix H, and a sampler $R _ { T }$ of k coordinates, or by the related FJLT [40, 41, 42], both applied in $O ( n \log n )$ time, or by sparse sketches such as CountSketch [43, 44, 45], applied in $O ( n )$ time. All of them are generated from a random seed, so the server broadcasts only the seed of $S _ { t }$ . The appendix verifies unbiasedness and second-moment bounds for these sketches, which sufices for the expectation analysis (Theorem 8); the highprobability guarantee of Theorem 1 additionally requires the inner-product, Frobenius-norm, and operator-norm events of Appendix D.

## 3.2 Per-layer Sketching

Applying a single JL transform to the entire parameter vector may be impractical for large models, and even infeasible for large language models. A natural remedy is per-layer sketching, where each layer $x _ { w } \in \mathbb { R } ^ { n _ { w } }$ of $x = ( x _ { 1 } , \ldots , x _ { W } )$ is compressed by its own JL transform $S _ { w } \in \mathbb { R } ^ { k _ { w } \times n _ { w } } , \mathrm { i . e . , ~ } S = \mathrm { b l o c k d i a g } ( S _ { 1 } , \dots , S _ { W } )$ . If each layerwise transform preserves norms, so does the full block-diagonal transform.

Theorem 2 (Per-layer norm preservation)   
Let $x = ( x _ { 1 } , \ldots , x _ { W } ) \in \mathbb { R } ^ { n } \backslash \{ 0 \}$ , where $x _ { w } \in \mathbb { R } ^ { n _ { w } }$ and $\Sigma _ { w = 1 } ^ { W } n _ { w } = n$ Let $S =$ blockdiag $( S _ { 1 } , \ldots , S _ { W } )$ , where   
$S _ { w } \in \mathbb { R } ^ { k _ { w } \times n _ { w } }$ are random matrices defined on a common probability space. Suppose that, for each $w ,$ there   
exist parameters $\varepsilon _ { w } \ge 0$ and $\delta _ { w } \in [ 0 , 1 ]$ such that, for every fixed $z \in \mathbb { R } ^ { n _ { w } } , \mathbb { P } \left( \left| \| S _ { w } z \| ^ { 2 } - \| z \| ^ { 2 } \right| > \varepsilon _ { w } \| z \| ^ { 2 } \right) \leq$   
$\delta _ { w }$ . Then, for every fixed $x \in \mathbb { R } ^ { n } \setminus \{ 0 \}$   
$\mathbb { P } \left( \left| \| S x \| ^ { 2 } - \| x \| ^ { 2 } \right| > \left( \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) \varepsilon _ { w } \right) \| x \| ^ { 2 } \right) \leq \sum _ { w = 1 } ^ { W } \delta _ { w } ,$   
where $\alpha _ { w } ( x ) \stackrel { \mathrm { d e f } } { = } \| x _ { w } \| ^ { 2 } / \| x \| ^ { 2 } .$

If the layerwise distribution of gradient energy is known, the sketch dimensions can be allocated non-uniformly across layers.

Corollary 1 (Energy-aware allocation)   
Let $\varepsilon , \delta \in ( 0 , 1 )$ and use Gaussian or Rademacher sketches on the active layers, with dimensions $k _ { w } =$   
$\left\lceil C \frac { \log ( 2 W / \delta ) } { \varepsilon _ { w } ^ { 2 } } \right\rceil$ , where $C > 0$ is a suficiently large universal constant. Fix $x \in \mathbb { R } ^ { n } \setminus \{ 0 \}$ and let $\alpha _ { w } ( x ) =$   
$\dot { \| } \boldsymbol { x } _ { w } \| ^ { 2 } / \| \boldsymbol { x } \| ^ { 2 }$ For all active layers with $\alpha _ { w } ( x ) > 0$ , set   
ε α<sub>w</sub>(x)<sup>−1/3</sup> δ   
<sup>εw</sup> <sup>=</sup> P<sub>j:αj(x)>0</sub> α<sub>j</sub> (x)<sup>2/3 ,</sup> δ<sub>w</sub> = <sub>W</sub> . (12)   
Assume that $\varepsilon _ { w } < 1$ and $k _ { w } \leq n _ { w }$ for every active layer. Inactive layers are omitted, with $k _ { w } = 0$ Then,   
for this fixed vector x, the block-diagonal sketch satisfies an ε-distortion guarantee with probability at least   
$1 - \delta ,$ and the total number of transmitted coordinates scales as   
W X k<sub>w</sub> = Θ log(2W/δ) ε<sup>2</sup> X W αw(x)2/3! 3   
w=1 w=1

In this feasible regime, energy-aware allocation can reduce the total communication cost for the fixed vector compared to a uniform allocation.

## 3.3 Matrix and Tensor Extensions

Neural network parameters are naturally matrices or higher-order tensors, which suggests sketches that respect this structure. For $X \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ , instead of applying a dense sketch to $\mathrm { v e c } ( X )$ , one can use two smaller JL matrices $S _ { 1 } \in \mathbb { R } ^ { k _ { 1 } \times n _ { 1 } }$ and $S _ { 2 } \in \mathbb { R } ^ { k _ { 2 } \times n _ { 2 } }$ and set

$$
{ \mathcal { S } } ( X ) = S _ { 1 } X S _ { 2 } ^ { \top } \in \mathbb { R } ^ { k _ { 1 } \times k _ { 2 } } , \qquad \operatorname { v e c } ( S ( X ) ) = ( S _ { 2 } \otimes S _ { 1 } ) \operatorname { v e c } ( X ) .
$$

A dense sketch of $\mathrm { v e c } ( X ) \in \mathbb { R } ^ { n _ { 1 } n _ { 2 } }$ to dimension k requires $O ( n _ { 1 } n _ { 2 } k )$ memory, whereas the Kronecker-structured sketch stores only $S _ { 1 }$ and $S _ { 2 } , { \mathrm { i . e . , } } O ( n _ { 1 } k _ { 1 } + n _ { 2 } k _ { 2 } )$ memory, for a compressed dimension $k _ { 1 } k _ { 2 }$ . This dimension may not be optimal, but the savings in memory and implementation cost are essential for large-scale models. The same idea extends to higher-order tensors.

Let $S = S _ { d } \otimes S _ { d - 1 } \otimes . . . \otimes S _ { 1 }$ , where $S _ { j } \in \mathbb { R } ^ { k _ { j } \times n _ { j } }$ are independent Gaussian or Rademacher JL matrices, and suppose that $\begin{array} { r } { \operatorname* { m i n } _ { j \in [ d ] } k _ { j } = \Omega \left( \frac { d ^ { 2 } \log ( 2 { \dot { d } } / \delta ) } { { \varepsilon } ^ { 2 } } \right) } \end{array}$ Then, for every fixed tensor $X \in \mathbb { R } ^ { n _ { 1 } \times . . . \times n _ { d } }$ , with $s ( X )$ defined by $\operatorname { v e c } ( S ( X ) ) = S \operatorname { v e c } ( X )$

$$
\mathbb { P } \left( \Big | \| S ( X ) \| _ { F } ^ { 2 } - \| X \| _ { F } ^ { 2 } \Big | > \varepsilon \| X \| _ { F } ^ { 2 } \right) \leq \delta .
$$

As for per-layer sketches, their use in the high-probability convergence proof requires the theoretical justifying specified in Appendix D. Overall, dense Gaussian and Rademacher projections are useful for theory and small-scale experiments, whereas structured, sparse, per-layer, and Kronecker-based sketches reduce the cost of applying and storing the compression operator.

## 4 Experiments

## 4.1 Comparison of Compressors

In this section we compare several compressors along two axes: the time and memory required to compress and decompress the transmitted parameters and the extent to which they satisfy the JL-properties required by our Algorithm 1.

Table 1 reports the cost of applying each compressor to the parameters of a ResNet18 model. For RandK we treat all parameters as a single flattened vector, whereas for the JL-based compressors and PowerSGD we compres each layer separately. For SRFT and SRHT we consider both scenarios, since these methods can operate in either one without incurring a significant memory or time overhead compared to the remaining JL compressors. CountSketch is omitted from the experiments, as it is not readily supported by standard $\mathrm { P y }$ Torch tensor operations. Stored denotes the additional memory required to keep the compressor’s internal state (e.g. the JL matrix). All measurements are taken at a compression rate of 50%.

As the table shows, our JL variant is the fastest among all JL-based methods, and is outperformed only by RandK on GPU, which is unsurprising since that method merely selects a subset of coordinates. At the same time, in terms of peak memory and CPU time our method outperforms every compressor considered. Applying SRHT and SRFT to the full parameter vector ofers no practical advantage over layer-wise application. The efect on running time is marginal (a minor speedup on GPU and a 5–30% slowdown on CPU), whereas peak memory consumption increases by a factor of 3.3–3.6.

Table 2 compares several quantities that measure how well the JL properties are preserved by the diferent compression methods. Here $\Delta _ { \mathrm { i p } }$ and $\Delta _ { \theta }$ denote the quantities

$$
\Delta _ { \mathrm { i p } } = \frac { \left| \langle S u , S v \rangle - \langle u , v \rangle \right| } { \| u \| \| v \| } , \qquad \Delta _ { \theta } = \left| \angle ( S u , S v ) - \angle ( u , v ) \right| .
$$

Table 1: Cost of applying each compressor to ResNet-18 parameters at $k / n = 0 . 5 { \mathrm { : } }$ compression and decompression time, peak memory, memory for the compressor state (Stored) and per-client uplink message size. JL compressors and PowerSGD are applied layer-wise, Rand-k to the flattened parameter vector; <sup>†</sup> applied to the flattened parameter vector.
<table><tr><td>Compressor</td><td>GPU, ms</td><td>CPU, ms</td><td>Peak, MB</td><td>Stored, MB</td><td>Uplink, MB</td></tr><tr><td>Gaussian  ${ \mathrm { J L } } ,$  Kronecker (ours)</td><td>6.5</td><td>155</td><td>92.5</td><td>69.9</td><td>21.3</td></tr><tr><td>Rademacher JL, Kronecker (ours)</td><td>8.1</td><td>160</td><td>92.5</td><td>69.9</td><td>21.3</td></tr><tr><td>SRHT</td><td>55.0</td><td>718</td><td>181.2</td><td>117.9</td><td>21.3</td></tr><tr><td>SRFT</td><td>12.7</td><td>659</td><td>131.1</td><td>63.9</td><td>21.3</td></tr><tr><td>SRHT, whole vector†</td><td>30.6</td><td>927</td><td>644.3</td><td>106.6</td><td>21.3</td></tr><tr><td>SRFT, whole vector†</td><td>8.8</td><td>693</td><td>432.0</td><td>64.0</td><td>21.3</td></tr><tr><td>Rand-k</td><td>3.8</td><td>374</td><td>603.4</td><td>0.0</td><td>21.3</td></tr><tr><td>PowerSGD (rank 230)</td><td>174.6</td><td>195</td><td>94.7</td><td>18.7</td><td>21.3</td></tr><tr><td>Gaussian  $\mathrm { J L ^ { \dagger } } , \mathrm { d e n s e ^ { \frac { \dagger } { \hbar } } }$ </td><td> $1 2 \cdot 1 0 ^ { 3 }$ </td><td></td><td> $3 . 7 \cdot 1 0 ^ { 7 }$ </td><td> $3 . 7 \cdot 1 0 ^ { 7 }$ </td><td></td></tr></table>

$\Delta _ { \mathrm { i p } }$ measures the deviation of the inner product of the compressed and uncompressed vectors, whereas the $\Delta _ { \theta }$ measures the distortion of the angle between the vectors caused by compression.

Table 2: Distortion of inner products $( \Delta _ { \mathrm { i p } } )$ and angles $\left( \Delta _ { \theta } \right)$ under compression, mean ± std. (a) Random $4 0 9 6 \times 4 0 9 6 ;$ (b) stochastic ResNet-18 gradient vs. trust-set gradient. All compressors use $k / n = 0 . 5 ;$ for Adaptive JL, $k / n$ is set implicitly by the target ε from equation 12.
<table><tr><td></td><td></td><td colspan="3">(a) random matrix 4096 × 4096 20 draws of  $( S , u , v )$   $m e a n \pm s t d$  2</td><td colspan="3">(b) real ResNet-18 gradients 100 pairs, mean ± std</td></tr><tr><td>Compressor</td><td>Variant</td><td> $k / n$ </td><td> $\Delta _ { \mathrm { i p } } , 1 0 ^ { - 4 }$ </td><td> $\Delta _ { \theta } , \mathrm { d e g }$ </td><td> $k / n$ </td><td> $\Delta _ { \mathrm { i p } } , 1 0 ^ { - 4 }$ </td><td> $\Delta _ { \theta } , \mathrm { d e g }$ </td></tr><tr><td>Kronecker JL (ours)</td><td>Rad.</td><td>0.500</td><td> $5 . 0 \pm 4 . 5$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 9 0$ </td><td>0.500</td><td> $1 9 4 \pm 1 5 6$ </td><td> $1 . 1 \pm 1 . 7$ </td></tr><tr><td></td><td>Gauss.</td><td>0.500</td><td> $8 . 7 \pm 8 . 3$ </td><td> $0 . 0 1 7 \pm 0 . 0 1 3$ </td><td>0.500</td><td> $9 4 \pm 8 6$ </td><td> $0 . 8 \pm 0 . 7$ </td></tr><tr><td>SRHT</td><td>layer</td><td>0.500</td><td> $2 . 5 \pm 1 . 9$ </td><td> $0 . 0 0 6 0 \pm 0 . 0 0 5 6$ </td><td>0.500</td><td> $1 0 3 \pm 8 0$ </td><td> $0 . 5 5 \pm 0 . 4 3$ </td></tr><tr><td></td><td>whole</td><td></td><td></td><td></td><td>0.500</td><td> $3 0 2 \pm 2 5 1$ </td><td> $1 . 3 1 \pm 1 . 2 3$ </td></tr><tr><td>SRFT</td><td>layer</td><td>0.500</td><td> $2 . 6 \pm 1 . 7$ </td><td> $0 . 0 0 6 7 \pm 0 . 0 0 6 4$ </td><td>0.500</td><td> $1 2 0 \pm 1 2 5$ </td><td> $0 . 6 3 \pm 0 . 5 8$ </td></tr><tr><td></td><td>whole</td><td></td><td></td><td></td><td>0.500</td><td> $4 7 0 \pm 1 9 8$ </td><td> $2 . 1 \pm 1 . 1 1$ </td></tr><tr><td>Rand-k</td><td></td><td>0.500</td><td></td><td> $2 . 1 \pm 2 . 0 0 . 0 0 6 3 \pm 0 . 0 0 6 4$ </td><td>0.500</td><td> $2 0 0 0 \pm 8 0 0$ </td><td> $1 . 4 \pm 1 . 1$ </td></tr><tr><td>Adaptive JL</td><td>target ε = 0.05</td><td>1.00</td><td>0</td><td>0</td><td>1.00</td><td>0</td><td>0</td></tr><tr><td></td><td>target ε = 0.1</td><td>0.841</td><td> $3 . 3 \pm 2 . 5$ </td><td> $0 . 0 1 1 \pm 0 . 0 1 2$ </td><td>1.00</td><td>0</td><td>0</td></tr><tr><td></td><td>target ε = 0.2</td><td>0.0610</td><td> $1 3 \pm 1 1$ </td><td> $0 . 0 2 5 \pm 0 . 0 2 1$ </td><td>0.605</td><td> $3 9 \pm 3 8$ </td><td> $0 . 2 3 \pm 0 . 2 1$ </td></tr><tr><td></td><td>target  $\varepsilon = 0 . 3$ </td><td>0.0141</td><td> $1 9 \pm 1 2$ </td><td> $0 . 0 5 7 \pm 0 . 0 5 0$ </td><td>0.240</td><td> $9 6 \pm 7 7$ </td><td> $0 . 5 1 \pm 0 . 4 3$ </td></tr></table>

The table reports results for two setups: setup (a) is the compression of two $4 0 9 6 \times 4 0 9 6$ matrices, and setup (b) is the compression of an actual stochastic gradient of a ResNet18 model and its comparison against the gradient computed on the trust set, as used in Algorithm 1. For all compressors except AdaptiveJL the compression rate is fixed at 50%. For AdaptiveJL the compression rate is determined implicitly by the parameter ε. We therefore report results for several values of ε. A comparison of the compressors at other compression rates is provided in Appendix A.

As the table shows, RandK performs reasonably well in the first setup, which can be attributed to the random construction of the matrices and to the unbiasedness of the compressor. However, on real gradients the method distorts both the inner product and the angle quite severely. In contrast, the JL-based methods perform well in both scenarios. We also note separately that $\mathrm { S R F T }$ and SRHT applied to the gradient as a single vector yield the worst results among all JL-based methods considered, which together with Table 1, indicates that layer-wise compression is preferable.

## 4.2 Comparison with Byzantine Baselines

This section presents experiments comparing Algorithm 1 against a range of federated learning methods under various Byzantine attack scenarios. All results are reported in the tables below. Every experiment is run in the setting of training a ResNet18 model on the CIFAR-10 dataset. All methods are trained for $T = 2 0 0$ communication rounds with the data distributed uniformly across $M = 1 0$ clients. To compare methods with diferent compression levels, we measure communication in full gradient uploads: a compressed message of k coordinates counts as $k / n$ of a full gradient, where n is the number of model parameters, and uploads are summed over all M clients. After t rounds this gives

$$
G ( t ) = \sum _ { s = 1 } ^ { t } \sum _ { i = 1 } ^ { M } \frac { k _ { s , i } } { n } ,\tag{13}
$$

which reduces to $G ( t ) = t M k / n$ for a fixed compression ratio and to $G ( t ) = t M$ without compression. Over the full $T = 2 0 0$ rounds an uncompressed method thus performs 2000 full-gradient uploads, and a method with $k / n = 0 . 5$ performs 1000. Details on the attack scenarios and additional experiments are deferred to Appendix A.

Table 3: FedAvg and Grad-BANT with diferent compressors: (a) no attack, (b) label flipping with 60% Byzantine clients.
<table><tr><td>Method</td><td>Compressor</td><td> $k / n$ </td><td>Byz.</td><td>G→ 60</td><td>G→ 80</td><td>Best acc</td></tr><tr><td colspan="7">(a) no attack</td></tr><tr><td>FedAvg</td><td>Gaussian JL</td><td>0.1</td><td>0</td><td>23</td><td>52</td><td>0.880</td></tr><tr><td>FedAvg</td><td>Rand-k</td><td>0.1</td><td>0</td><td>15</td><td>39</td><td>0.897</td></tr><tr><td>FedAvg</td><td>SRHT</td><td>0.1</td><td>0</td><td>17</td><td>42</td><td>0.895</td></tr><tr><td>Grad-BANT</td><td>Gaussian JL</td><td>0.1</td><td>0</td><td>18</td><td>53</td><td>0.896</td></tr><tr><td>Grad-BANT</td><td>Rand-k</td><td>0.1</td><td>0</td><td>15</td><td>41</td><td>0.891</td></tr><tr><td>Grad-BANT</td><td>SRHT</td><td>0.1</td><td>0</td><td>16</td><td>42</td><td>0.893</td></tr><tr><td colspan="7">(b) label flipping</td></tr><tr><td>FedAvg</td><td>none</td><td>1.0</td><td>60%</td><td></td><td></td><td>0.276</td></tr><tr><td>FedAvg</td><td>Gaussian JL</td><td>0.1</td><td>60%</td><td></td><td></td><td>0.247</td></tr><tr><td>FedAvg</td><td>Rand-k</td><td>0.1</td><td>60%</td><td></td><td></td><td>0.315</td></tr><tr><td>FedAvg</td><td>SRHT</td><td>0.1</td><td>60%</td><td></td><td></td><td>0.334</td></tr><tr><td>Grad-BANT</td><td>Gaussian JL</td><td>0.1</td><td>60%</td><td>31</td><td>71</td><td>0.838</td></tr><tr><td>Grad-BANT</td><td>Adaptive JL</td><td>0.24</td><td>60%</td><td>25</td><td>69</td><td>0.833</td></tr><tr><td>Grad-BANT</td><td>SRHT</td><td>0.1</td><td>60%</td><td>26</td><td>78</td><td>0.830</td></tr><tr><td>Grad-BANT</td><td>Rand-k</td><td>0.1</td><td>60%</td><td>一</td><td></td><td>0.546</td></tr></table>

Table 4: Best test accuracy of Byzantine-robust methods with compression under label flipping (LF) and random gradients (RG) with 40% and 60% Byzantine clients. All methods use the $k / n = 0 . 5$
<table><tr><td>Method</td><td>LF 40%</td><td>LF 60%</td><td>RG 40%</td><td>RG 60%</td></tr><tr><td>RoSDHB</td><td>0.595</td><td>0.331</td><td>0.308</td><td>0.103</td></tr><tr><td>Byz-EF21</td><td>0.430</td><td>0.294</td><td>0.441</td><td>0.123</td></tr><tr><td>Byz-EF21-BC</td><td>0.429</td><td>0.265</td><td>0.431</td><td>0.103</td></tr><tr><td>Byz-VR-MARINA 2.0</td><td>0.555</td><td>0.159</td><td>0.571</td><td>0.123</td></tr><tr><td>Byz-DASHA-PAGE</td><td>0.565</td><td>0.152</td><td>0.455</td><td>0.115</td></tr><tr><td>Grad-BANT + JL (ours)</td><td>0.883</td><td>0.862</td><td>0.620</td><td>0.534</td></tr></table>

Table 3 reports the results of applying diferent compressors to FedAvg and GradBant, both in the attack-free setting and under the Label Flipping (LF) attack. For each configuration, the table reports the number of full-gradient transmissions required to reach a given value of the target metric, together with the best test accuracy attained. Columns G→60 and G→80 report $G ( t )$ from equation 13 at the first round t at which the test accuracy reaches 60% and 80%, respectively. For AdaptiveJL, $k / n$ denotes the average over rounds and clients.

In the attack-free setting, all compressors perform similarly for both algorithms. Once the attack is enabled, every compressor in the FedAvg setting fails to learn and shows almost no accuracy improvement throughout training. In contrast, for GradBant the properties of the compressor start to matter. All JL compressors which preserve inner products continue to work, whereas the RandK compressor is likewise compromised by the attack.

Table 4 compares Byzantine-robust methods with compression [46] under two diferent attacks. Unlike ours, the remaining methods do not use a trust set, and consequently cannot tolerate a Byzantine majority. When the number of Byzantine clients approaches one half, these methods yield results that lag far behind our algorithm.

Table 5: Comparison with Byzantine-robust baselines under diferent attacks (Byzantine fraction in the header): (a) best test accuracy; (b) full-gradient uploads G to reach 60% test accuracy. Baselines are uncompressed; Grad-BANT + JL uses k/n = 0.5. –: not reached.
<table><tr><td>Method</td><td>none</td><td>LF 60%</td><td>SF 60%</td><td>RG 40%</td><td>ALIE 60%</td><td>IPM 70%</td></tr><tr><td>(a) best accuracy</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedAvg (Adam)</td><td>0.917</td><td>0.276</td><td>0.331</td><td>0.132</td><td>0.145</td><td>0.100</td></tr><tr><td>CC</td><td>0.928</td><td>0.828</td><td>0.574</td><td>0.122</td><td>0.143</td><td>0.100</td></tr><tr><td>CC + bucketing</td><td>0.840</td><td>0.897</td><td>0.891</td><td>0.134</td><td>0.143</td><td>0.100</td></tr><tr><td>CC + FBM</td><td>0.924</td><td>0.855</td><td>0.892</td><td>0.116</td><td>0.184</td><td>0.100</td></tr><tr><td>Safeguard</td><td>0.921</td><td>0.104</td><td>0.858</td><td>0.132</td><td>0.113</td><td>0.137</td></tr><tr><td>FLTrust</td><td>0.767</td><td>0.704</td><td>0.594</td><td>0.100</td><td>0.100</td><td>0.101</td></tr><tr><td>RECESS</td><td>0.889</td><td>0.681</td><td>0.512</td><td>0.142</td><td>0.146</td><td>0.106</td></tr><tr><td>Zeno</td><td>0.919</td><td>0.361</td><td>0.859</td><td>0.105</td><td>0.101</td><td>0.124</td></tr><tr><td>TS-finetuned</td><td>0.920</td><td>0.884</td><td>0.877</td><td>0.889</td><td>0.903</td><td>0.630</td></tr><tr><td>TS-minimum-grad</td><td>0.922</td><td>0.887</td><td>0.861</td><td>0.901</td><td>0.882</td><td>0.677</td></tr><tr><td>TS-momentum</td><td>0.868</td><td>0.863</td><td>0.860</td><td>0.857</td><td>0.865</td><td>0.700</td></tr><tr><td>Grad-BANT + JL (ours), 50% compression</td><td>0.910</td><td>0.862</td><td>0.878</td><td>0.620</td><td>0.539</td><td>0.827</td></tr><tr><td>(b) full-gradient uploads G to 60%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedAvg (Adam)</td><td>160</td><td></td><td>850</td><td></td><td></td><td></td></tr><tr><td>CC</td><td>60</td><td>290</td><td></td><td></td><td></td><td></td></tr><tr><td>CC + bucketing</td><td>460</td><td>360</td><td>130</td><td></td><td></td><td></td></tr><tr><td>CC + FBM</td><td>60</td><td>140</td><td>130</td><td></td><td></td><td></td></tr><tr><td>Safeguard</td><td>140</td><td></td><td>170</td><td></td><td></td><td></td></tr><tr><td>FLTrust</td><td>840</td><td>1140</td><td></td><td></td><td></td><td></td></tr><tr><td>RECESS</td><td>150</td><td>610</td><td></td><td></td><td></td><td></td></tr><tr><td>Zeno</td><td>160</td><td></td><td>220</td><td></td><td></td><td></td></tr><tr><td>TS-finetuned</td><td>130</td><td>250</td><td>170</td><td>280</td><td>160</td><td>1160</td></tr><tr><td>TS-minimum-grad</td><td>160</td><td>160</td><td>200</td><td>160</td><td>150</td><td>570</td></tr><tr><td>TS-momentum</td><td>330</td><td>330</td><td>360</td><td>560</td><td>310</td><td>570</td></tr><tr><td>Grad-BANT + JL (ours), full-gradient uploads</td><td>80</td><td>120</td><td>120</td><td>140</td><td></td><td>110</td></tr></table>

Finally, Table 5 compares (a) the best test accuracy and (b) the number of full-gradient uploads G required to reach 60% accuracy for Byzantine-robust baselines without compression and for our algorithm with a compression rate of 50% under several attacks. The results show that GradBant requires fewer full-gradient transmissions than the remaining baselines to reach 60% accuracy. We note that under the ALIE attack our method does not reach this level of accuracy, plateauing at 53.9%. In terms of best accuracy, our method outperforms all baselines under the IPM attack. In the attack-free scenario and under SignFlip and LabelFlip our method stays close to the strongest baselines.

## 4.3 Heterogenious scenario

In this section, we consider a scenario with heterogeneous data distribution across clients. To model heterogeneity, we use a Dirichlet(α) distribution to partition the dataset samples among clients. This makes the clients heterogeneous both in local dataset size and in the label distribution of their samples. Table 6 reports the training results of our algorithm under four heterogeneity scenarios, both without any attack and under three diferent attacks.

Table 6: Best test accuracy of Grad-BANT + JL $( k / n = 0 . 5 )$ under heterogeneous data split by a Dirichlet distribution with concentration α (smaller α means stronger heterogeneity). Byzantine fractions: 60% for label and sign flipping, 70% for IPM.
<table><tr><td>Attack</td><td>IID  $\alpha = 1 . 0$ </td><td> $\alpha = 0 . 5$ </td><td> $\alpha = 0 . 2$ </td></tr><tr><td>No attack</td><td>0.910</td><td>0.896</td><td>0.872 0.842</td></tr><tr><td>Label flipping</td><td>0.862</td><td>0.857</td><td>0.782 0.742</td></tr><tr><td>Sign flipping</td><td>0.878</td><td>0.871</td><td>0.832 0.753</td></tr><tr><td>IPM</td><td>0.827</td><td>0.826</td><td>0.699 0.669</td></tr></table>

All results are reported at a 50% compression rate. As expected, the final model quality degrades as the degree of heterogeneity increases. However, our algorithm remains highly robust to it. In all scenarios, it loses at most 19% accuracy in the worst case and 7% on average relative to the uniform (IID) data distribution.

## 5 Conclusion

We studied Johnson–Lindenstrauss transforms as communication-eficient compressors for Byzantine-robust distributed optimization. Unlike standard sparsification, JL transforms preserve Euclidean geometry with high probability, which suits methods that weight workers by gradient alignment. For a JL-compressed variant of Grad-BANT, we proved a high-probability convergence guarantee under smoothness, average data similarity, and an explicit accuracy condition on the trusted reference gradient, without convexity or a small relative heterogeneity parameter. We also studied structured, sparse, per-layer, and Kronecker-structured sketches, which reduce the cost of storing and applying dense projections in large models; their moment bounds support the expectation analysis, while the high-probability guarantee additionally requires geometric concentration bounds. The experiments confirm that JL-based compression reduces communication while maintaining Byzantine robustness, suggesting that preserving geometric structure is an important design principle for compression in robust distributed learning.

## References

[1] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and trends® in machine learning, 14(1–2):1–210, 2021.

[2] Zhenheng Tang, Shaohuai Shi, Wei Wang, Bo Li, and Xiaowen Chu. Communication-eficient distributed deep learning: A comprehensive survey. arXiv preprint arXiv:2003.06307, 2020.

[3] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. arXiv preprint arXiv:1907.02189, 2019.

[4] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communicationeficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273–1282. PMLR, 2017.

[5] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communication-eficient sgd via gradient quantization and encoding. Advances in neural information processing systems, 30, 2017.

[6] Yue ${ \mathrm { Y u } } ,$ Jiaxiang Wu, and Longbo Huang. Double quantization for communication-eficient distributed optimization. Advances in neural information processing systems, 32, 2019.

[7] Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsification for communication-eficient distributed optimization. Advances in Neural Information Processing Systems, 31, 2018.

[8] Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi. Error feedback fixes signsgd and other gradient compression schemes. In International Conference on Machine Learning, pages 3252–3261. PMLR, 2019.

[9] Peter Richtárik, Igor Sokolov, and Ilyas Fatkhullin. Ef21: A new, simpler, theoretically better, and practically faster error feedback. Advances in Neural Information Processing Systems, 34:4384–4396, 2021.

[10] Konstantin Mishchenko, Eduard Gorbunov, Martin Takáč, and Peter Richtárik. Distributed learning with compressed gradient diferences. Optimization Methods and Software, pages 1–16, 2024.

[11] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Learning from history for byzantine robust optimization. In International conference on machine learning, pages 5311–5319. PMLR, 2021.

[12] Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. Fltrust: Byzantine-robust federated learning via trust bootstrapping. arXiv preprint arXiv:2012.13995, 2020.

[13] Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, Hui Li, and Xiaodong Lin. Recess vaccine for federated learning: Proactive defense against model poisoning attacks. Advances in Neural Information Processing Systems, 36:8702–8713, 2023.

[14] Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, and Aleksandr Beznosikov. Trial and trust: Addressing byzantine attacks with comprehensive defense strategy. arXiv preprint arXiv:2505.07614, 2025.

[15] William B Johnson, Joram Lindenstrauss, et al. Extensions of lipschitz mappings into a hilbert space. Contemporary mathematics, 26(189-206):1, 1984.

[16] Eyal Kushilevitz, Rafail Ostrovsky, and Yuval Rabani. Eficient search for approximate nearest neighbor in high dimensional spaces. In Proceedings of the thirtieth annual ACM symposium on Theory of computing, pages 614–623, 1998.

[17] Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, and Ludwig Schmidt. Practical and optima lsh for angular distance. Advances in neural information processing systems, 28, 2015.

[18] Jon M Kleinberg. Two algorithms for nearest-neighbor search in high dimensions. In Proceedings of the twenty-ninth annual ACM symposium on Theory of computing, pages 599–608, 1997.

[19] Christina Heinze, Brian McWilliams, and Nicolai Meinshausen. Dual-loco: Distributing statistical estimation using random projections. In Artificial intelligence and statistics, pages 875–883. PMLR, 2016.

[20] Gian-Andrea Thanei, Christina Heinze, and Nicolai Meinshausen. Random projections for large-scale regression. Big and Complex Data Analysis: Methodologies and Applications, pages 51–68, 2017.

[21] Bin Tang, Michael Shepherd, Malcolm I Heywood, and Xiao Luo. Comparing dimension reduction techniques for document clustering. In Advances in Artificial Intelligence: 18th Conference of the Canadian Society for Computational Studies of Intelligence, Canadian AI 2005, Victoria, Canada, May 9-11, 2005. Proceedings 18, pages 292–296. Springer, 2005.

[22] Christos Boutsidis, Anastasios Zouzias, and Petros Drineas. Random projections for k-means clustering. Advances in neural information processing systems, 23, 2010.

[23] Sanjoy Dasgupta. Learning mixtures of gaussians. In 40th Annual Symposium on Foundations of Computer Science (Cat. No. 99CB37039), pages 634–644. IEEE, 1999.

[24] Thierry Urruty, Chabane Djeraba, and Dan A Simovici. Clustering by random projections. In Advances in Data Mining. Theoretical Aspects and Applications: 7th Industrial Conference, ICDM 2007, Leipzig, Germany, July 14-18, 2007. Proceedings 7, pages 107–119. Springer, 2007.

[25] Reinhard Heckel, Michael Tschannen, and Helmut Bölcskei. Dimensionality-reduced subspace clustering. Information and Inference: A Journal of the IMA, 6(3):246–283, 2017.

[26] Xiao Guo, Yixuan Qiu, Hai Zhang, and Xiangyu Chang. Randomized spectral co-clustering for large-scale directed networks. arXiv preprint arXiv:2004.12164, 2020.

[27] Jalaj Upadhyay. Randomness eficient fast-johnson-lindenstrauss transform with applications in diferentia privacy and compressed sensing. arXiv preprint arXiv:1410.2470, 2014.

[28] Or Shefet. Diferentially private ordinary least squares. In International Conference on Machine Learning, pages 3105–3114. PMLR, 2017.

[29] David P Woodruf et al. Sketching as a tool for numerical linear algebra. Foundations and Trends® in Theoretical Computer Science, 10(1–2):1–157, 2014.

[30] Cameron Musco and Christopher Musco. Projection-cost-preserving sketches: Proof strategies and constructions. arXiv preprint arXiv:2004.08434, 2020.

[31] Tamas Sarlos. Improved approximation algorithms for large matrices via random projections. In 2006 47th annual IEEE symposium on foundations of computer science (FOCS’06), pages 143–152. IEEE, 2006.

[32] Zhao Song, Zheng Yu, and Lichen Zhang. Iterative sketching and its application to federated learning. 2021.

[33] Mayank Shrivastava, Berivan Isik, Qiaobo Li, Sanmi Koyejo, and Arindam Banerjee. Sketching for distributed deep learning: A sharper analysis. Advances in Neural Information Processing Systems, 37:6417–6447, 2024.

[34] Xinyan Dai, Xiao Yan, Kaiwen Zhou, Han Yang, Kelvin KW Ng, James Cheng, and Yu Fan. Hyper-sphere quantization: Communication-eficient sgd for federated learning. arXiv preprint arXiv:1911.04655, 2019.

[35] Jayadev Acharya, Chris De Sa, Dylan Foster, and Karthik Sridharan. Distributed learning with sublinear communication. In International Conference on Machine Learning, pages 40–50. PMLR, 2019.

[36] Sanjoy Dasgupta and Anupam Gupta. An elementary proof of a theorem of johnson and lindenstrauss. Random Structures & Algorithms, 22(1):60–65, 2003.

[37] Dimitris Achlioptas. Database-friendly random projections: Johnson-lindenstrauss with binary coins. Journal of computer and System Sciences, 66(4):671–687, 2003.

[38] Aleksandr Beznosikov, Samuel Horváth, Peter Richtárik, and Mher Safaryan. On biased compression for distributed learning. Journal of Machine Learning Research, 24(276):1–50, 2023.

[39] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2 edition, 2026.

[40] Nir Ailon and Bernard Chazelle. Approximate nearest neighbors and the fast johnson-lindenstrauss transform. In Proceedings of the Thirty-Eighth Annual ACM Symposium on Theory of Computing, pages 557–563, 2006.

[41] Joel A Tropp. Improved analysis of the subsampled randomized hadamard transform. Advances in Adaptive Data Analysis, 3(01n02):115–126, 2011.

[42] Nir Ailon and Edo Liberty. An almost optimal unrestricted fast johnson-lindenstrauss transform. ACM Transactions on Algorithms, 9(3):1–12, 2013.

[43] Moses Charikar, Kevin Chen, and Martin Farach-Colton. Finding frequent items in data streams. In International Colloquium on Automata, Languages, and Programming, pages 693–703. Springer, 2002.

[44] Kenneth L Clarkson and David P Woodruf. Low rank approximation and regression in input sparsity time. In Proceedings of the Forty-Fifth Annual ACM Symposium on Theory of Computing, pages 81–90, 2013.

[45] Daniel M Kane and Jelani Nelson. Sparser johnson-lindenstrauss transforms. Journal of the ACM, 61(1):1–23, 2014.

[46] Ahmad Rammal, Kaja Gruntkowska, Nikita Fedin, Eduard Gorbunov, and Peter Richtárik. Communication compression for byzantine robust learning: New eficient algorithms and improved rates. In International Conference on Artificial Intelligence and Statistics, pages 1207–1215. PMLR, 2024.

## Appendix

Supplementary Materials for On Johnson-Lindenstrauss Transforms for Byzantine-Robust Optimization

## A Experimental Details

## A.1 Protocol.

Hardware. All experiments were conducted on a single NVIDIA A100 GPU with 80 GB of memory. Running all experiments required approximately 200 GPU-hours in total.

Hyperparameters. The hyperparameters of each method were tuned via grid search. For each method, all tables report the results obtained with the configuration that achieved the highest validation accuracy.

Table 7: Byzantine attacks used in the experiments. Here g<sub>i</sub> denotes the stochastic gradient that an honest client i would send, g¯ and σ denote the coordinate-wise mean and standard deviation of the honest clients’ gradients, and C is the number of classes.
<table><tr><td>Attack</td><td>Byzantine behavior</td><td>Byz. fraction</td></tr><tr><td>Label flipping (LF)</td><td>Data poisoning: Byzantine clients compute gradients on their local data with each label y replaced by  $C - 1 - y .$ </td><td>40%, 60%</td></tr><tr><td>Sign flipping (SF)</td><td>Byzantine clients send the negated honest gradient  $- g _ { i }$  </td><td>60%</td></tr><tr><td>Random gradients (RG)</td><td>Byzantine clients send random vectors with i.i.d. Gaussian entries,  $\xi \sim$   $\dot { \mathcal { N } } ( 0 , s ^ { 2 } I )$ </td><td>40%, 60%</td></tr><tr><td>ALIE</td><td>Omniscient attack: Byzantine clients send  ${ \bar { g } } - z \sigma ,$  a small coordinate-wise perturbation of the honest mean that stays within the spread of honest gradients and thus evades robust aggregation rules.</td><td>60%</td></tr><tr><td>IPM</td><td>Omniscient attack: Byzantine clients send  $- \epsilon \bar { g }$  with  $\epsilon > 0 ,$  so that the inner product between the aggregated update and the true gradient be- comes negative.</td><td>70%</td></tr></table>

## A.2 Trial ablation

In this section, we study how the size and distribution of the trust set afect the performance of Algorithm 1. All experiments use $k / n = 0 . 5$ , label flipping with 60% Byzantine clients, IID data, and 100 training rounds. The results are summarized in Table 8.

Trust set size. Table 8 (a) shows that the size of the trust set has little efect on either the best test accuracy or the convergence speed. As the size increases from 100 to 2000 images, the best accuracy varies only between 0.838 and 0.844, and the number of rounds needed to reach 60% and 80% accuracy changes by at most 6 and 5 rounds, respectively.

Trust set distribution. Table 8 (b) considers a trust set of 1500 images whose distribution difers from uniform. Shifts in image appearance slow down training but do not prevent convergence. With grayscale images, reaching 60% and 80% accuracy takes 6 and 13 more rounds, respectively, while the best accuracy decreases only slightly, from 0.840 to 0.831. Strong augmentation has a larger efect. Reaching 60% accuracy takes almost three times as many rounds, 80% is never reached, and the best accuracy is 0.761. In both cases, training remains stable, and the final accuracy coincides with the best one. In contrast, class coverage is critical. When the trust set contains only 5 of the 10 classes, the best accuracy remains high (0.828), but training becomes unstable and the final accuracy drops to 0.636. With only 2 classes, the method fails completely: the accuracy never exceeds 0.16, which is close to random guessing.

Table 8: Efect of the trust set on Grad-BANT + JL: (a) size; (b) distribution shift, 1500 images. R→ 60 and R→ 80: rounds to reach 60% and 80% test accuracy; Best and Final: best and last-round accuracy; –: not reached. Label flipping, 60% Byzantine clients, $k / n = 0 . 5$ , IID data, 100 rounds.
<table><tr><td>Trust set R→ 60 R→ 80 Best</td></tr><tr><td>Final (a) size, all 10 classes</td></tr><tr><td>100 21 61 0.844 0.683</td></tr><tr><td>500 21 65 0.841 0.674</td></tr><tr><td>1000 23 64 0.839 0.622</td></tr><tr><td>1500 22 65 0.840 0.835</td></tr><tr><td>2000 27 66 0.838 0.838</td></tr><tr><td></td></tr><tr><td>(b) shift, 1500 images</td></tr><tr><td>none 22 65 0.840 0.835</td></tr><tr><td>grayscale 28 78 0.831 0.831</td></tr><tr><td>strong augmentation 63 0.761 0.761</td></tr><tr><td>5 of 10 classes 23 71 0.828 0.636</td></tr><tr><td>2 of 10 classes 0.160 0.102</td></tr></table>

Overall, Algorithm 1 is robust to the size of the trust set and to moderate shifts in its appearance. The key requirement is that the trust set covers all classes; its size mainly afects the stability of training in later rounds.

Training on the trust set alone. To check whether federated training is necessary when a trust set is available, we train ResNet-18 centrally on the trust set alone, using the same optimizer (Adam, learning rate 0.003), batch size (64), and augmentation as the clients. The budget is 12 400 optimizer steps, which matches the local computation of one client over 200 rounds. Table 9 shows that this baseline is far behind federated training. With 1500 images, the best accuracy is 0.609, whereas Algorithm 1 with a trust set of the same size reaches 0.913 within 100 rounds, even with 60% Byzantine clients (Table 8). The gap grows as the trust set shrinks: with 100 images, training on the trust set alone reaches at most 0.287, while the federated method still reaches 0.894. Even with 2000 images, the best accuracy is only 0.653. Moreover, the accuracy saturates early: between the 100th and 200th round-equivalents, it changes by less than 0.02 for all sizes, and the final accuracy is below the best one, which suggests overfitting to the small sample.

## A.3 Additional experiments

Table 10 extends the comparison of Table 2 to compression rates $k / n \in \{ 0 . 5 , 0 . 3 , 0 . 1 \}$ in the same setting: the distortion is measured between the stochastic ResNet-18 gradient and the trust-set gradient over 100 pairs.

As expected, the inner-product distortion $\Delta _ { \mathrm { i p } }$ increases monotonically for all compressors as the compression becomes stronger. Reducing $k / n$ from 0.5 to 0.1 increases $\Delta _ { \mathrm { i p } }$ by a factor of 3.0–5.2 for Kronecker JL and of about 5 for layer-wise SRHT and SRFT.

Kronecker JL remains competitive with layer-wise SRHT and SRFT at all compression rates: at every $k / n ,$ one of its variants attains the smallest mean $\Delta _ { \mathrm { i p } }$ among all compressors, although the diferences are within one standard deviation. The angle errors of all JL-type methods are of the same order, ranging from $0 . 5 5 ^ { \circ }$ at $k / n = 0 . 5$ to $4 . 2 ^ { \circ }$ at $k / n = 0 . 1$

Table 9: Training ResNet-18 on the trust set alone, without federation. @r: CIFAR-10 test accuracy after 62r optimizer steps, i.e., the computation of one client over r rounds. Epochs: passes over the trust set. Mean ± std over 3 seeds; each seed samples a diferent trust set. Federated training with the same trust set sizes reaches 0.89–0.92 (Table 8 (a)).
<table><tr><td></td><td>Trust set Epochs</td><td>@20</td><td>@50</td><td>@100</td><td>@200</td><td>Best</td></tr><tr><td>100</td><td>7936</td><td> $0 . 2 5 8 \pm 0 . 0 1 9$ </td><td> $0 . 2 6 0 \pm 0 . 0 1 4$ </td><td> $0 . 2 8 0 \pm 0 . 0 1 2$ </td><td> $0 . 2 7 0 \pm 0 . 0 1 5$ </td><td> $0 . 2 8 7 \pm 0 . 0 1 0$ </td></tr><tr><td>500</td><td>1587</td><td> $0 . 3 8 7 \pm 0 . 0 2 1$ </td><td> $0 . 4 0 7 \pm 0 . 0 2 7$ </td><td> $0 . 4 1 0 \pm 0 . 0 4 6$ </td><td> $0 . 4 2 8 \pm 0 . 0 2 4$ </td><td> $0 . 4 3 9 \pm 0 . 0 2 3$ </td></tr><tr><td>1000</td><td>794</td><td> $0 . 4 6 6 \pm 0 . 0 1 5$ </td><td> $0 . 4 9 6 \pm 0 . 0 1 1$ </td><td> $0 . 5 1 5 \pm 0 . 0 1 6$ </td><td> $0 . 5 0 1 \pm 0 . 0 0 3$ </td><td> $0 . 5 4 2 \pm 0 . 0 1 2$ </td></tr><tr><td>1500</td><td>529</td><td> $0 . 5 2 3 \pm 0 . 0 1 7$ </td><td> $0 . 5 5 4 \pm 0 . 0 2 1$ </td><td> $0 . 5 7 7 \pm 0 . 0 1 5$ </td><td> $0 . 5 7 8 \pm 0 . 0 0 8$ </td><td> $0 . 6 0 9 \pm 0 . 0 0 6$ </td></tr><tr><td>2000</td><td>397</td><td> $0 . 5 5 4 \pm 0 . 0 1 5$ </td><td> $0 . 6 0 7 \pm 0 . 0 1 2$ </td><td> $0 . 6 1 5 \pm 0 . 0 1 2$ </td><td> $0 . 6 2 8 \pm 0 . 0 1 1$ </td><td> $0 . 6 5 3 \pm 0 . 0 0 4$ </td></tr></table>

Applying SRHT and SRFT to the whole parameter vector does not reduce the distortion. At $k / n = 0 . 5$ , it increases $\Delta _ { \mathrm { i p } }$ by a factor of 2.9 for SRHT and 3.9 for SRFT; at smaller $k / n$ , the whole-vector variants are comparable to the layer-wise ones or worse by up to a factor of 1.5.

Rand-k yields the largest $\Delta _ { \mathrm { i p } }$ at every compression rate, 4–38 times larger than that of the other compressors, and its distortion grows the fastest as $k / n$ decreases. At the same time, its angle error remains comparable to that of the other methods.

Finally, Adaptive JL with target $\varepsilon = 0 . 3$ selects $k / n = 0 . 2 4 0$ and attains $\Delta _ { \mathrm { i p } }$ comparable to that of Gaussian Kronecker JL at $k / n = 0 . 5$ (96 vs. $9 4 \cdot 1 0 ^ { - 4 } )$ , i.e., with less than half the message size.

Table 11 shows that stronger compression substantially reduces the communication required to reach 80% test accuracy. Moderate compression does not slow down training: in all settings, the number of rounds needed to reach 80% is smallest at $k / n = 0 . 5$ . At $k / n = 0 . 1$ , more rounds are required, but each round is much cheaper, so the total trafic decreases by a factor of 2.8–4.0 compared to $k / n = 0 . 5$ and by a factor of 6.7–10.6 compared to $k / n = 0 . 9$ . At the same time, the best accuracy at $k / n = 0 . 1$ is at most 0.018 lower than at $k / n = 0 . 5$

Table 10: Distortion of inner products $( \Delta _ { \mathrm { i p } } )$ and angles $\left( \Delta _ { \theta } \right)$ under compression at diferent compression rates: stochastic ResNet-18 gradient vs. trust-set gradient, 100 pairs, mean $\pm \ \mathrm { s t d } .$ The rows for $k / n = 0 . 5$ and for Adaptive JL are repeated from Table 2. For Adaptive JL, $k / n = 0 . 2 4 0$ is set implicitly by the target $\varepsilon = 0 . 3$ from equation 12.
<table><tr><td>Compressor</td><td>Variant</td><td> $\Delta _ { \mathrm { i p } } , 1 0 ^ { - 4 }$ </td><td> $\Delta _ { \theta } , { \mathrm { d e g } }$ </td></tr><tr><td colspan="4"> $k / n = 0 . 5$ </td></tr><tr><td rowspan="3">Kronecker JL (ours) Rad.</td><td></td><td> $1 9 4 \pm 1 5 6$ </td><td> $1 . 1 \pm 1 . 7$ </td></tr><tr><td>Gauss.</td><td> $9 4 \pm 8 6$ </td><td> $0 . 8 \pm 0 . 7$ </td></tr><tr><td>layer whole</td><td> $1 0 3 \pm 8 0$ </td><td> $0 . 5 5 \pm 0 . 4 3$   $1 . 3 1 \pm 1 . 2 3$ </td></tr><tr><td rowspan="2">SRFT</td><td>layer</td><td> $3 0 2 \pm 2 5 1$   $1 2 0 \pm 1 2 5$ </td><td> $0 . 6 3 \pm 0 . 5 8$ </td></tr><tr><td>whole</td><td> $4 7 0 \pm 1 9 8$ </td><td> $2 . 1 \pm 1 . 1 1$ </td></tr><tr><td>Rand-k</td><td></td><td> $2 0 0 0 \pm 8 0 0$ </td><td> $1 . 4 \pm 1 . 1$ </td></tr><tr><td colspan="2"> $k / n = 0 . 3$ </td><td></td><td></td></tr><tr><td rowspan="2">Kronecker JL (ours) Rad.</td><td></td><td> $3 0 0 \pm 1 9 8$ </td><td> $1 . 6 3 \pm 0 . 9 8$ </td></tr><tr><td>Gauss. layer</td><td> $3 9 0 \pm 2 5 0$   $4 7 4 \pm 2 8 3$ </td><td> $1 . 6 9 \pm 1 . 3 5$   $1 . 7 2 \pm 0 . 9 6$ </td></tr><tr><td rowspan="2">SRHT SRFT</td><td>whole</td><td> $4 5 6 \pm 3 4 1$ </td><td> $2 . 6 3 \pm 2 . 1 1$ </td></tr><tr><td>layer whole</td><td> $5 3 4 \pm 3 1 3$ </td><td> $2 . 1 3 \pm 2 . 0 9$ </td></tr><tr><td colspan="2">Rand-k</td><td> $6 3 2 \pm 4 1 2$   $4 9 6 3 \pm 9 1 4$ </td><td> $1 . 8 7 \pm 1 . 3 1$   $1 . 5 7 \pm 1 . 1 6$ </td></tr><tr><td colspan="2"> $k / n = 0 . 1$ </td><td></td><td></td></tr><tr><td colspan="2">Kronecker JL (ours) Rad.</td><td> $5 9 0 \pm 4 9 0$ </td><td> $3 . 5 1 \pm 2 . 9 3$ </td></tr><tr><td rowspan="3">SRHT</td><td>Gauss.</td><td> $4 9 0 \pm 4 9 0$ </td><td> $4 . 1 7 \pm 3 . 9 2$ </td></tr><tr><td>layer</td><td> $5 1 2 \pm 4 8 1$ </td><td> $2 . 1 \pm 1 . 4$ </td></tr><tr><td>whole</td><td> $7 1 5 \pm 6 5 3$   $6 1 4 \pm 5 4 1$ </td><td> $4 . 1 5 \pm 3 . 3 1$   $3 . 0 4 \pm 2 . 9 8$ </td></tr><tr><td rowspan="2">SRFT</td><td>layer whole</td><td></td><td> $3 . 1 \pm 2 . 9$ </td></tr><tr><td></td><td> $8 9 1 \pm 7 5 3$   $1 8 5 0 0 \pm 7 9 0 0$ </td><td> $3 . 3 \pm 2 . 4$ </td></tr><tr><td>Rand-k</td><td></td><td></td><td></td></tr><tr><td colspan="2">adaptive  $k / n$ </td><td></td><td></td></tr><tr><td colspan="2">Adaptive JL target</td><td> $\varepsilon = 0 . 3$   $9 6 \pm 7 7$ </td><td> $0 . 5 1 \pm 0 . 4 3$ </td></tr></table>

Table 11: Communication eficiency of Grad-BANT + JL for diferent compression rates $k / n$ (IID data, 200 rounds, single seed): (a) rounds and (b) trafic in full-gradient uploads (a round costs $k / n$ units) to reach 80% test accuracy; (c) best test accuracy. Columns 0.9–0.5 use Rademacher JL, columns 0.3–0.1 use Gaussian $\mathrm { J L }$
<table><tr><td> $k / n$ </td><td>0.9</td><td>0.7</td><td>0.5</td><td>0.3</td><td>0.1</td></tr><tr><td>(a) rounds to 80% No attack</td><td>54</td><td>45</td><td>41</td><td>45</td><td>73</td></tr><tr><td>Label flipping 60% Sign flipping 60%</td><td>71 68</td><td>62 47 56</td><td>57 46 48</td><td>60 61 51</td><td>71 58 68</td></tr><tr><td colspan="6">IPM 70% 68 (b) traffic to 80%, full-gradient uploads No attack</td></tr><tr><td>Label flipping 60% Sign flipping 60% IPM 70%</td><td>48.6 63.9 61.2 61.2</td><td>31.5 43.4 32.9 39.2</td><td>20.5 28.5 23.0 24.0</td><td>13.5 18.0 18.3 15.3</td><td>7.3 7.1 5.8 6.8</td></tr><tr><td>(c) best accuracy in 200 rounds No attack Label flipping 60%</td><td>0.900</td><td>0.906</td><td>0.910</td><td>0.906</td><td>0.893</td></tr><tr><td>Sign flipping 60% IPM 70%</td><td>0.853 0.857 0.854</td><td>0.868 0.875 0.875</td><td>0.862 0.878 0.827</td><td>0.865 0.868 0.870</td><td>0.857 0.860 0.861</td></tr></table>

## B Technical Statements

Statement below is crucial for deriving high-probability convergence of proposed methods

Lemma 5

Suppose we have events $E _ { 1 } , E _ { 2 } , \ldots , E _ { n }$ that holds with following probabilities:

$$
\mathbb { P } \left[ E _ { i } \right] \geq 1 - \delta _ { i } ,
$$

Then,

$$
\mathbb { P } \left[ \bigcap _ { i = 1 } ^ { n } E _ { i } \right] \geq 1 - \sum _ { i = 1 } ^ { n } \delta _ { i }
$$

Proof.

$$
\mathbb { P } \left[ \bigcap _ { i = 1 } ^ { n } E _ { i } \right] = 1 - \mathbb { P } \left[ \bigcup _ { i = 1 } ^ { n } { \overline { { E _ { i } } } } \right] \geq 1 - \sum _ { i = 1 } ^ { n } \mathbb { P } \left[ { \overline { { E _ { i } } } } \right] \geq 1 - \sum _ { i = 1 } ^ { n } \delta _ { i }
$$

□

Next lemma will be useful throughout some theorems, allowing us to deal with norms:

Lemma 6 (Young’s inequality)

Let x, $y \in \mathbb { R } ^ { n }$ . Then, for any $\alpha > 0$ we have

$$
\langle x , y \rangle \leq { \frac { 1 } { 2 \alpha } } \| x \| ^ { 2 } + { \frac { \alpha } { 2 } } \| y \| ^ { 2 } .
$$

## C JL Statements

We start with the properties of a single sketch: inner-product preservation, moments of the lifted vector, and high-probability norm bounds. We then show how these properties extend to structured matrices, separate layers, and tensor modes. Throughout, Sx is the compressed vector and $S ^ { \top }$ Sx is the lifted vector used in the update. We write ∥ · ∥ for the Euclidean norm of a vector or the operator norm of a matrix, and $\| \cdot \| _ { F }$ for the Frobenius norm of a matrix or tensor.

## C.1 From norm preservation to inner products

```latex
Lemma 7 (Preservation of scalar product)
Let S be a linear JL transform with distortion ε and one-vector failure probability δ. For every fixed
$x , y \in \mathbb { R } ^ { n }$
$\begin{array} { r } { \mathbb { P } \left[ | \langle S x , S y \rangle - \langle x , y \rangle | \leq \varepsilon \| x \| \| y \| \right] \geq 1 - 2 \delta . } \end{array}$
```

Proof. The claim is immediate if either vector is zero. By linearity, it is enough to prove it for unit vectors. Apply the JL property to $x + y$ and $x - y$ . With probability at least 1 − 2δ, both squared norms are preserved within relative error ε. On this event, polarization and the parallelogram identity give

$$
\begin{array} { c } { \displaystyle { | \langle S x , S y \rangle - \langle x , y \rangle | \le \frac { \varepsilon } { 4 } \left( \| x + y \| ^ { 2 } + \| x - y \| ^ { 2 } \right) } } \\ { \displaystyle { = \frac { \varepsilon } { 2 } \left( \| x \| ^ { 2 } + \| y \| ^ { 2 } \right) = \varepsilon . } } \end{array}
$$

For general nonzero vectors, apply this bound to $x / \| x \|$ and $y / \| y \|$ , and multiply by ∥x∥∥y∥.

```latex
Lemma 8 (Simultaneous upper JL bound)
Let S be a linear JL transform with distortion ε and one-vector failure probability δ. For fixed vectors
$v _ { 1 } , \ldots , v _ { m } \in \mathbb { R } ^ { n }$
$\begin{array} { r } { \mathbb { P } \left[ \| S v _ { i } \| ^ { 2 } \leq ( 1 + \varepsilon ) \| v _ { i } \| ^ { 2 } \quad \mathrm { f o r ~ a l l ~ } i = 1 , \dots , m \right] \geq 1 - m \delta . } \end{array}$
```

Proof. Each upper bound fails with probability at most δ. Take the union bound over the m vectors.

## C.2 Dense sketches: moments and operator norms

The next identity gives the moments of both dense sketches and the sparse Gaussian construction below. It also allows an independent orthogonal transformation before sketching.

Lemma 9 (Moments of a normalized i.i.d. matrix)   
Let $A \in \mathbb { R } ^ { k \times n }$ have i.i.d. entries with   
$\mathbb { E } A _ { 1 1 } = 0 , \qquad \mathbb { E } A _ { 1 1 } ^ { 2 } = 1 , \qquad \mathbb { E } A _ { 1 1 } ^ { 4 } = \mu _ { 4 } < \infty .$   
Let $U \in \mathbb { R } ^ { n \times n }$ be orthogonal and independent of $A ; U$ may be deterministic or random. For $S = A U / { \sqrt { k } } ,$   
$\mathbb { E } [ S ^ { \top } S ] = I _ { n } , \qquad \mathbb { E } \| S ^ { \top } S x \| ^ { 2 } = \left( 1 + \frac { n - 2 + \mu _ { 4 } } { k } \right) \| x \| ^ { 2 } .$   
The second identity holds for every fixed $x \in \mathbb { R } ^ { n }$

Proof. First take $U = I _ { n }$ , and write the rows of A as column vectors $a _ { 1 } , \ldots , a _ { k }$ . Since $\mathbb { E } [ a _ { r } a _ { r } ^ { \top } ] = I _ { n }$

$$
\mathbb { E } [ S ^ { \top } S ] = \frac { 1 } { k } \sum _ { r = 1 } ^ { k } \mathbb { E } [ a _ { r } a _ { r } ^ { \top } ] = I _ { n } .
$$

For the second moment, expand

$$
\mathbb { E } \| A ^ { \top } A x \| ^ { 2 } = \sum _ { r , s = 1 } ^ { k } \mathbb { E } \left[ \langle a _ { r } , a _ { s } \rangle \langle a _ { r } , x \rangle \langle a _ { s } , x \rangle \right] .
$$

If $r \neq s ,$ conditioning on $a _ { r }$ shows that the corresponding expectation is $\mathbb { E } \langle a _ { r } , x \rangle ^ { 2 } = \| x \| ^ { 2 }$ . For a diagonal term, independence and centering of the entries remove all mixed terms, giving

$$
\mathbb { E } \left[ \| a _ { r } \| ^ { 2 } \langle a _ { r } , x \rangle ^ { 2 } \right] = \sum _ { j = 1 } ^ { n } \left( \sum _ { i \neq j } x _ { i } ^ { 2 } + \mu _ { 4 } x _ { j } ^ { 2 } \right)
$$

There are $k ( k - 1 )$ of-diagonal terms and k diagonal terms. Dividing their sum by $k ^ { 2 }$ yields

$$
\mathbb { E } \| S ^ { \top } S x \| ^ { 2 } = \frac { k ( k - 1 ) + k ( n - 1 + \mu _ { 4 } ) } { k ^ { 2 } } \| x \| ^ { 2 } = \left( 1 + \frac { n - 2 + \mu _ { 4 } } { k } \right) \| x \| ^ { 2 } .
$$

For general $U ,$ condition on $U .$ . Orthogonality reduces the calculation to the case just proved with x replaced by $U x$ , whose norm equals ∥x∥. Averaging over U proves both identities. □

## Lemma 10 (Gaussian and Rademacher moments)

Let $S \in \mathbb { R } ^ { k \times n }$ have independent entries, either $S _ { i j } \sim \mathcal { N } ( 0 , 1 / k )$ or $S _ { i j } \in \{ - 1 / \sqrt { k } , 1 / \sqrt { k } \}$ with equal probabilities. In both cases $\mathbb { E } [ S ^ { \top } S ] = I _ { n }$ , and for every fixed $x \in \mathbb { R } ^ { n }$

$$
\mathbb { E } \Vert S ^ { \top } S x \Vert ^ { 2 } = \left\{ \frac { n + k + 1 } { k } \Vert x \Vert ^ { 2 } , \mathrm { G a u s s i a n } , \right.
$$

Proof. Apply Lemma 9 with $U \ = \ I _ { n }$ . The fourth moment of a standard Gaussian entry is $^ { 3 , }$ and that of a Rademacher entry is 1. □

We next bound the operator norm. Every such bound also controls the reconstructed vector, since for every matrix S and every $x ,$

$$
\| S ^ { \top } S \| = \| S \| ^ { 2 } , \qquad \| S ^ { \top } S x \| ^ { 2 } \leq \| S \| ^ { 4 } \| x \| ^ { 2 } .
$$

The following standard estimate will be used for the Rademacher and sparse Gaussian constructions. Its subgaussian norm is defined, for a scalar X and a random vector $Z ,$ by

$$
\| X \| _ { \psi _ { 2 } } : = \operatorname* { i n f } \{ s > 0 : \mathbb { E } e ^ { X ^ { 2 } / s ^ { 2 } } \leq 2 \} , \qquad \| Z \| _ { \psi _ { 2 } } : = \operatorname* { s u p } _ { \| u \| = 1 } \| \langle Z , u \rangle \| _ { \psi _ { 2 } } .
$$

## Lemma 11 (Subgaussian operator-norm estimate $I ^ { \it 3 9 } I )$

Let $A \in \mathbb { R } ^ { m \times n }$ have independent, mean-zero, isotropic, subgaussian rows $A _ { 1 } , \ldots , A _ { m }$ , with $\| A _ { i } \| _ { \psi _ { 2 } } \leq K$ for all i. There is a universal constant $C > 0$ such that, for every $t \geq 0$

$$
\begin{array} { r } { \mathbb { P } \left[ \| A \| \le \sqrt { m } + C K ^ { 2 } ( \sqrt { n } + t ) \right] \ge 1 - 2 e ^ { - t ^ { 2 } } . } \end{array}
$$

## Lemma 12 (Gaussian and Rademacher operator norms)

There is a universal constant $C > 0$ such that, for a Gaussian or a Rademacher JL matrix $S \in \mathbb { R } ^ { k \times n }$ and every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
\| S \| \leq 1 + C \left( { \sqrt { \frac { n } { k } } } + { \sqrt { \frac { \log ( 2 / \delta ) } { k } } } \right) .
$$

Squaring the right-hand side bounds $\| S ^ { \top } S \|$ ; its fourth power times $\| { \boldsymbol { x } } \| ^ { 2 }$ bounds $\| S ^ { \top } S x \| ^ { 2 }$ simultaneously for all $x \in \mathbb { R } ^ { n }$

Proof. In the Gaussian case, write $S = A / \sqrt { k }$ with standard Gaussian entries in A. By Vershynin [39, Corollary 7.3.2], there is a universal constant $c _ { 0 } > 0$ such that, for every $t \geq 0$

$$
\begin{array} { r } { \mathbb { P } \left[ \| A \| \ge \sqrt { k } + \sqrt { n } + t \right] \le 2 e ^ { - c _ { 0 } t ^ { 2 } } . } \end{array}
$$

Taking $t = \sqrt { \log ( 2 / \delta ) / c _ { 0 } }$ and dividing by $\sqrt { k }$ gives, with probability at least $1 - \delta$

$$
\| S \| \leq 1 + \sqrt { \frac { n } { k } } + \frac { 1 } { \sqrt { c _ { 0 } } } \sqrt { \frac { \log ( 2 / \delta ) } { k } } ,
$$

which is the stated bound for any $C \geq \operatorname* { m a x } \{ 1 , c _ { 0 } ^ { - 1 / 2 } \}$

In the Rademacher case, write again $S = A / \sqrt { k }$ . Its independent rows are mean-zero and isotropic. For each unit vector u, Hoefding’s lemma gives

$$
{ \mathbb E } e ^ { \lambda \langle A _ { i } , u \rangle } \le e ^ { \lambda ^ { 2 } / 2 } \quad \mathrm { f o r ~ a l l ~ } \lambda \in { \mathbb R } ,
$$

so their subgaussian norms are bounded by a universal constant. Apply Lemma 11 with $m = k$ and $t = \sqrt { { \log ( 2 / \delta ) } }$ and divide by $\sqrt { k }$ . Enlarging C if necessary, one constant serves both constructions. The remaining claims follow from $\| S ^ { \top } S \| = \| S \| ^ { 2 }$ and $\| S ^ { \top } S x \| \leq \| S ^ { \top } S \| \| x \|$ □

## C.3 Preserving the total squared norm of several vectors

The convergence proof needs a bound on the sum of the squared norms of the compressed honest gradients. Applying a separate JL guarantee to every gradient would introduce a failure-probability factor equal to the number of honest workers. For Gaussian and Rademacher sketches, we can instead control this sum directly.

Place the vectors in the columns of a fixed matrix $V \in \mathbb { R } ^ { n \times N }$ . Then $\| V \| _ { F } ^ { 2 }$ is their total squared norm and $\| S V \| _ { F } ^ { 2 }$ is the corresponding quantity after compression. We first give an elementary concentration bound used in the Gaussian case.

```latex
Lemma 13 (Weighted chi-square concentration)
Let $a _ { 1 } , \ldots , a _ { r } \geq 0$ and let $Z _ { 1 } , \ldots , Z _ { r }$ be independent $\chi _ { k } ^ { 2 }$ random variables. For every $\varepsilon \in ( 0 , 1 )$
$\mathbb { P } \left[ \left| \frac { 1 } { k } \sum _ { i = 1 } ^ { r } a _ { i } Z _ { i } - \sum _ { i = 1 } ^ { r } a _ { i } \right| > \varepsilon \sum _ { i = 1 } ^ { r } a _ { i } \right] \le 2 e ^ { - k \varepsilon ^ { 2 } / 1 6 } .$
Proof. The claim is immediate if all weights vanish. Otherwise, divide the weights by their sum, so that
$\textstyle \sum _ { i } a _ { i } = 1$ For a single collection of independent standard Gaussian variables $g _ { i }$ , the Gaussian moment
generating function gives, for $0 \leq \lambda \leq 1 / 4$
$\log \mathbb { E } \exp \left( \lambda \sum _ { i } a _ { i } ( g _ { i } ^ { 2 } - 1 ) \right) = - \lambda - \frac { 1 } { 2 } \sum _ { i } \log ( 1 - 2 \lambda a _ { i } ) \leq 4 \lambda ^ { 2 } \sum _ { i } a _ { i } ^ { 2 } \leq 4 \lambda ^ { 2 } ,$
$\log \mathbb { E } \exp \left( - \lambda \sum _ { i } a _ { i } ( g _ { i } ^ { 2 } - 1 ) \right) = \lambda - \frac { 1 } { 2 } \sum _ { i } \log ( 1 + 2 \lambda a _ { i } ) \leq \lambda ^ { 2 } \sum _ { i } a _ { i } ^ { 2 } \leq \lambda ^ { 2 } .$
The second bound also holds for every $\lambda \geq 0$ . Write each $Z _ { i }$ as the sum of $k$ independent squared standard
Gaussian variables. Independence across these $k$ repetitions and the exponential Markov inequality give
upper and lower tail bounds
$e ^ { - k \lambda \varepsilon + 4 k \lambda ^ { 2 } } \quad { \mathrm { a n d } } \quad e ^ { - k \lambda \varepsilon + k \lambda ^ { 2 } } ,$
respectively. Taking $\lambda = \varepsilon / 8$ in the first bound and $\lambda = \varepsilon / 2$ in the second proves the result. □
```

Lemma 14 (Preservation of a fixed Frobenius norm)   
Let $S \in \mathbb { R } ^ { k \times n }$ be a Gaussian or Rademacher JL matrix. There is a universal constant $c _ { 1 } > 0$ such that, for   
every fixed matrix $V \in \mathbb { R } ^ { n \times N }$ and every $\varepsilon \in ( 0 , 1 )$ ，   
$\begin{array} { r } { \mathbb { P } [ \| S V \| _ { F } ^ { 2 } - \| V \| _ { F } ^ { 2 } | > \varepsilon \| V \| _ { F } ^ { 2 } ] \leq 2 e ^ { - c _ { 1 } k \varepsilon ^ { 2 } } . } \end{array}$   
In particular, the bound does not depend on the number of columns $N .$   
Proof. The case $V = 0$ is immediate.   
Gaussian sketches. Take a compact singular value decomposition of $V ,$ with nonzero singular values   
$\sigma _ { 1 } , \ldots , \sigma _ { r }$ Gaussian rotational invariance gives   
$\| S V \| _ { F } ^ { 2 } \stackrel { d } { = } \frac 1 k \sum _ { i = 1 } ^ { r } \sigma _ { i } ^ { 2 } Z _ { i } , \qquad \| V \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { r } \sigma _ { i } ^ { 2 } ,$   
where the $Z _ { i }$ are independent $\chi _ { k } ^ { 2 }$ variables. Apply Lemma 13 with weights $\sigma _ { i } ^ { 2 } .$   
Rademacher sketches. Let $\xi _ { 1 } , \ldots , \xi _ { k }$ be the rows of $\sqrt { k } S .$ , viewed as column vectors, and set $B = V V ^ { \top }$   
Then   
$k \| S V \| _ { F } ^ { 2 } = \sum _ { \ell = 1 } ^ { k } \xi _ { \ell } ^ { \top } B \xi _ { \ell } , \qquad \mathbb { E } \| S V \| _ { F } ^ { 2 } = \mathrm { t r } ( B ) = \| V \| _ { F } ^ { 2 } .$   
Apply the Hanson–Wright inequality to the concatenated vector of all kn independent signs and the block  
diagonal matrix with k copies of $B .$ For every $s > 0$ , this gives   
$\mathbb { P } [ \lvert k \rvert | S V \rVert _ { F } ^ { 2 } - k \rvert | V \rVert _ { F } ^ { 2 }  > s ] \le 2 \exp ( - c \operatorname* { m i n } \{ \frac { s ^ { 2 } } { k \lVert B \rVert _ { F } ^ { 2 } } , \frac { s } { \lVert B \rVert } \} )$

Since B is positive semidefinite, both $\| B \|$ and $\| B \| _ { F }$ are at most $\mathrm { t r } ( B ) = \| V \| _ { F } ^ { 2 }$ With $s = k \varepsilon \lVert V \rVert _ { F } ^ { 2 }$ and $0 < \varepsilon < 1$ , the minimum is at least $k \varepsilon ^ { 2 }$ Choosing a common constant for the two constructions completes the proof. □

This estimate concerns one fixed matrix V, rather than all matrices simultaneously. It also applies conditionally when $V$ is determined by the past and $S$ is sampled independently afterwards. Together with an operator-norm bound, it controls the lifted vectors through

$$
\| S ^ { \top } S V \| _ { F } ^ { 2 } \leq \| S \| ^ { 2 } \| S V \| _ { F } ^ { 2 } .
$$

The same Frobenius-norm estimate will allow us to compress a tensor one mode at a time below.

## C.4 Structured sketches

The following results give the same moment and operator-norm information for SRHT, the FJLT with a sparse Gaussian core, and CountSketch.

## Lemma 15 (SRHT)

Assume that n is a power of two and $1 \leq k \leq n$ Let H be the normalized Walsh–Hadamard matrix, let D have independent Rademacher entries on its diagonal, and let $T \subset [ n ]$ be a uniformly random k-element subset independent of $D .$ If $R _ { T }$ restricts a vector to its coordinates in $T$ , define

$$
S = \sqrt { \frac { n } { k } } R _ { T } H D .
$$

Then

$$
\mathbb { E } [ S ^ { \top } S ] = I _ { n } , \qquad \mathbb { E } \| S ^ { \top } S x \| ^ { 2 } = \frac { n } { k } \| x \| ^ { 2 } .
$$

The second identity holds for every fixed $x \in \mathbb { R } ^ { n }$ . Moreover, $\| S \| = { \sqrt { n / k } }$ almost surely.

Proof. The matrix $R _ { T } ^ { \top } R _ { T }$ is the coordinate projector onto T. Each coordinate is selected with probability $k / n$ , so

$$
\mathbb { E } [ R _ { T } ^ { \top } R _ { T } ] = \frac { k } { n } I _ { n } .
$$

Since $H D$ is orthogonal, conditioning on D gives

$$
\mathbb { E } [ S ^ { \top } S \mid D ] = \frac { n } { k } D H ^ { \top } \mathbb { E } [ R _ { T } ^ { \top } R _ { T } ] H D = I _ { n } .
$$

Also, $R _ { T } R _ { T } ^ { \top } = I _ { k }$ implies

$$
S S ^ { \top } = { \frac { n } { k } } I _ { k } , \qquad ( S ^ { \top } S ) ^ { 2 } = { \frac { n } { k } } S ^ { \top } S .
$$

The first identity gives $\| S \| = { \sqrt { n / k } }$ . The second and unbiasedness give

$$
\mathbb { E } \| S ^ { \top } S x \| ^ { 2 } = x ^ { \top } \mathbb { E } [ ( S ^ { \top } S ) ^ { 2 } ] x = \frac n k \| x \| ^ { 2 } .
$$

## Lemma 16 (FJLT with a sparse Gaussian core)

Assume that n is a power of two. Let H be the normalized Walsh–Hadamard matrix, let D be a diagonal Rademacher matrix, and let $P \in \mathbb { R } ^ { k \times n }$ have independent entries

$$
P _ { i j } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - q , } \\ { g _ { i j } / \sqrt { q } , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } q , } \end{array} \right. \qquad g _ { i j } \sim \mathcal { N } ( 0 , 1 ) , \qquad q \in ( 0 , 1 ] .
$$

Assume that all random choices are independent, and define $S = P H D / { \sqrt { k } } .$ Then

$$
\mathbb { E } [ S ^ { \top } S ] = I _ { n } , \qquad \mathbb { E } \| S ^ { \top } S x \| ^ { 2 } = \left( 1 + \frac { n - 2 + 3 / q } { k } \right) \| x \| ^ { 2 } \quad \mathrm { f o r ~ e v e r y ~ f i x e d ~ } x \in \mathbb { R } ^ { n } .
$$

Moreover, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
\left\| S \right\| \leq 1 + { \frac { C } { q } } \left( { \sqrt { \frac { n } { k } } } + { \sqrt { \frac { \log ( 2 / \delta ) } { k } } } \right) ,
$$

where $C > 0$ is universal.

Proof. The entries of P satisfy

$$
\mathbb { E } P _ { 1 1 } = 0 , \qquad \mathbb { E } P _ { 1 1 } ^ { 2 } = 1 , \qquad \mathbb { E } P _ { 1 1 } ^ { 4 } = 3 / q .
$$

The moment identities follow from Lemma 9 with $A = P$ and the independent orthogonal matrix $U = H D$ To bound the operator norm, observe that

$$
\mathbb { E } e ^ { \lambda P _ { 1 1 } } = 1 - q + q e ^ { \lambda ^ { 2 } / ( 2 q ) } \le e ^ { \lambda ^ { 2 } / ( 2 q ) } .
$$

By independence, for a row p of P and a unit vector u,

$$
\mathbb { E } e ^ { \lambda \langle p , u \rangle } \le \prod _ { j = 1 } ^ { n } e ^ { \lambda ^ { 2 } u _ { j } ^ { 2 } / ( 2 q ) } = e ^ { \lambda ^ { 2 } / ( 2 q ) } .
$$

Thus the rows are mean-zero and isotropic, with subgaussian norms at most $C / { \sqrt { q } }$ . Lemma 11, with $t = \sqrt { { \log ( 2 / \delta ) } }$ gives

$$
\| P \| \leq \sqrt { k } + \frac { C } { q } \left( \sqrt { n } + \sqrt { \log ( 2 / \delta ) } \right)
$$

with the claimed probability, after changing the universal constant. Finally, orthogonality of $H D$ gives $\| S \| =$ $\| P \| / { \sqrt { k } }$ □

## Lemma 17 (CountSketch)

Let $h ( 1 ) , \ldots , h ( n )$ be independent and uniform on [k], and let $\sigma ( 1 ) , \ldots , \sigma ( n )$ be independent Rademacher signs, independent of h. Define $S \in \mathbb { R } ^ { k \times n }$ by

$$
S _ { r j } = \sigma ( j ) \mathbf { 1 } _ { \{ h ( j ) = r \} } .
$$

Then

$$
\mathbb { E } [ S ^ { \top } S ] = I _ { n } , \qquad \mathbb { E } \| S ^ { \top } S x \| ^ { 2 } = \left( 1 + \frac { n - 1 } { k } \right) \| x \| ^ { 2 } \quad \mathrm { f o r ~ e v e r y ~ f i x e d ~ } x \in \mathbb { R } ^ { n } .
$$

If $N _ { r } = \# \{ j \in [ n ] : h ( j ) = r \}$ is the number of coordinates assigned to row $r ,$ then

$$
S S ^ { \top } = \mathrm { d i a g } ( N _ { 1 } , \ldots , N _ { k } ) , \qquad \| S \| ^ { 2 } = \operatorname* { m a x } _ { 1 \leq r \leq k } N _ { r } .
$$

Consequently, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta .$ ,

$$
\| S \| ^ { 2 } \leq { \frac { n } { k } } + \sqrt { { \frac { 2 n } { k } } \log { \frac { k } { \delta } } } + { \frac { 2 } { 3 } } \log { \frac { k } { \delta } } .
$$

Proof. Every column contains one nonzero entry of magnitude one. Hence $( S ^ { \top } S ) _ { j j } = 1$ , while for $i \neq j$

$$
( S ^ { \top } S ) _ { i j } = \sigma ( i ) \sigma ( j ) \mathbf { 1 } _ { \{ h ( i ) = h ( j ) \} } .
$$

Independent centered signs make the of-diagonal expectations zero, proving unbiasedness. To compute the second moment, write

$$
( S ^ { \top } S x - x ) _ { j } = \sigma ( j ) \sum _ { i \neq j } \sigma ( i ) \mathbf { 1 } _ { \{ h ( i ) = h ( j ) \} } x _ { i } .
$$

After squaring, the mixed terms vanish in expectation. Since each pair of coordinates collides with probability $1 / k$

$$
\mathbb { E } \| S ^ { \top } S x - x \| ^ { 2 } = \frac { 1 } { k } \sum _ { j = 1 } ^ { n } \sum _ { i \neq j } x _ { i } ^ { 2 } = \frac { n - 1 } { k } \| x \| ^ { 2 } .
$$

Unbiasedness now gives the claimed second moment.

Distinct rows of S have disjoint supports, and the squared norm of row $r$ is $N _ { r }$ . This proves the formula for $S S ^ { \top }$ and its operator norm. For each $r ,$ the count $N _ { r }$ is binomial with mean $n / k$ . Bernstein’s inequality gives, for every $u \geq 0 .$

$$
\mathbb { P } \left[ N _ { r } > \frac { n } { k } + \sqrt { \frac { 2 n u } { k } } + \frac { 2 u } { 3 } \right] \le e ^ { - u } .
$$

Take $u = \log ( k / \delta )$ and apply the union bound over the k rows.

## C.5 Per-layer sketching

We now combine the one-layer guaranties using the block-diagonal structure of the sketch. Write

$$
x = ( x _ { 1 } , \ldots , x _ { W } ) , \qquad x _ { w } \in \mathbb { R } ^ { n _ { w } } , \qquad n = \sum _ { w = 1 } ^ { W } n _ { w } , \qquad S = \mathrm { d i a g } ( S _ { 1 } , \ldots , S _ { W } ) ,
$$

where $S _ { w } \in \mathbb { R } ^ { k w \times n _ { w } }$ . The key identities are

$$
\| x \| ^ { 2 } = \sum _ { w = 1 } ^ { W } \| x _ { w } \| ^ { 2 } , \qquad \| S x \| ^ { 2 } = \sum _ { w = 1 } ^ { W } \| S _ { w } x _ { w } \| ^ { 2 } .
$$

$$
\boldsymbol { S } ^ { \intercal } \boldsymbol { S } = \mathrm { d i a g } ( S _ { 1 } ^ { \intercal } S _ { 1 } , \dots , S _ { W } ^ { \intercal } S _ { W } ) , \qquad \| \boldsymbol { S } ^ { \intercal } S \boldsymbol { x } \| ^ { 2 } = \sum _ { w = 1 } ^ { W } \| S _ { w } ^ { \intercal } S _ { w } x _ { w } \| ^ { 2 } .
$$

As in the main text, for $x \neq 0$ we use the energy fractions

$$
\alpha _ { w } ( x ) = \frac { \| x _ { w } \| ^ { 2 } } { \| x \| ^ { 2 } } , \qquad \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) = 1 .
$$

Thus, squared-norm bounds combine as energy-weighted averages, whereas operator-norm bounds combine by taking the largest layer bound.

## Theorem 4 (Per-layer JL properties)

Suppose that, for each w, there are $\varepsilon _ { w } \ge 0$ and $\delta _ { w } \in [ 0 , 1 ]$ such that, for every fixed $u \in \mathbb { R } ^ { n _ { w } }$ ，

$$
\begin{array} { r } { \mathbb { P } [ \| S _ { w } u \| ^ { 2 } - \| u \| ^ { 2 } | > \varepsilon _ { w } \| u \| ^ { 2 } ] \leq \delta _ { w } . } \end{array}
$$

Then, for every fixed $x \neq 0 .$

$$
\mathbb { P } \left[ | \| S x \| ^ { 2 } - \| x \| ^ { 2 } \big | > \left( \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) \varepsilon _ { w } \right) \| x \| ^ { 2 } \right] \leq \sum _ { w = 1 } ^ { W } \delta _ { w } .
$$

In particular, if max $_ { w } \varepsilon _ { w } \leq \varepsilon ,$ then S satisfies the $( \varepsilon , \sum _ { w } \delta _ { w } )$ distributional JL property. If the blocks are independent, both failure bounds can be replaced by $\begin{array} { r } { 1 - \prod _ { w = 1 } ^ { W } ( 1 - \delta _ { w } ) } \end{array}$

Proof. Fix $x \neq 0$ . With probability at least $1 - \textstyle \sum _ { w } \delta _ { w }$ , every layer satisfies $| \| S _ { w } x _ { w } \| ^ { 2 } - \| x _ { w } \| ^ { 2 } | \leq \varepsilon _ { w } \| x _ { w } \| ^ { 2 }$ . On this event,

$$
\big | \| S x \| ^ { 2 } - \| x \| ^ { 2 } \big | \leq \sum _ { w = 1 } ^ { W } \big | \| S _ { w } x _ { w } \| ^ { 2 } - \| x _ { w } \| ^ { 2 } \big | \leq \sum _ { w = 1 } ^ { W } \varepsilon _ { w } \| x _ { w } \| ^ { 2 } = \left( \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) \varepsilon _ { w } \right) \| x \| ^ { 2 } .
$$

The weighted distortion is at most $\operatorname* { m a x } _ { w } \varepsilon _ { w }$ , since the weights sum to one. For independent blocks, the layer success events are independent, so their intersection has probability at least $\prod _ { w } ( 1 - \delta _ { w } )$ □

The same block identities give the properties of the lifted map $S ^ { \top } S$ needed in the optimization analysis. None of the next three results requires independence between layers.

## Lemma 18 (Per-layer unbiasedness)

If $\mathbb { E } [ S _ { w } ^ { \top } S _ { w } ] = I _ { n _ { w } }$ for every layer, then

$$
\mathbb { E } [ S ^ { \top } S ] = I _ { n } , \qquad \mathbb { E } [ S ^ { \top } S x ] = x \quad { \mathrm { f o r ~ a l l ~ } } x \in \mathbb { R } ^ { n } .
$$

Proof. Take expectations in the block-diagonal identity for $S ^ { \top } S$

## Lemma 19 (Per-layer second moment)

Suppose, for each layer, there is $\omega _ { w } \ge 0$ such that

$$
\begin{array} { r } { \mathbb { E } \| S _ { w } ^ { \top } S _ { w } u \| ^ { 2 } \leq \omega _ { w } \| u \| ^ { 2 } \qquad \mathrm { f o r ~ e v e r y ~ } u \in \mathbb { R } ^ { n _ { w } } . } \end{array}
$$

Then, for every x $\neq 0$

$$
\mathbb { E } \| S ^ { \top } S x \| ^ { 2 } \leq \left( \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) \omega _ { w } \right) \| x \| ^ { 2 } \leq \left( \operatorname* { m a x } _ { w } \omega _ { w } \right) \| x \| ^ { 2 } .
$$

Proof. Using the sum of squared block norms and linearity of expectation gives

$$
\mathbb { E } \| S ^ { \top } S x \| ^ { 2 } = \sum _ { w = 1 } ^ { W } \mathbb { E } \| S _ { w } ^ { \top } S _ { w } x _ { w } \| ^ { 2 } \leq \sum _ { w = 1 } ^ { W } \omega _ { w } \| x _ { w } \| ^ { 2 } .
$$

Divide by $\| { \boldsymbol { x } } \| ^ { 2 }$ and use $\begin{array} { r } { \sum _ { w } \alpha _ { w } ( x ) = 1 } \end{array}$

```latex
Lemma 20 (Per-layer high-probability norm bound)
Suppose $\begin{array} { r } { \mathbb { P } [ \| S _ { w } \| ^ { 2 } \leq c _ { w } ] \geq 1 - \delta _ { w } } \end{array}$ for every layer. Then, with probability at least $\begin{array} { r } { 1 - \sum _ { w } \delta _ { w } , } \end{array}$
$\| S \| ^ { 2 } = \| S ^ { \top } S \| \leq \operatorname* { m a x } _ { w } c _ { w } ,$
and, on the same event, for every $x \neq 0 ,$
$\lVert S x \rVert ^ { 2 } \leq \left( \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) c _ { w } \right) \lVert x \rVert ^ { 2 } ,$
W
∥S⊤Sx∥2 ≤ X αw(x)c2w! ∥x∥2.
w=1
```

Proof. A union bound ensures that all inequalities $\| S _ { w } \| ^ { 2 } \leq c _ { w }$ hold on an event of the stated probability. On this event, block diagonality gives

$$
\| S \| ^ { 2 } = \| S ^ { \top } S \| = \operatorname* { m a x } _ { w } \| S _ { w } \| ^ { 2 } \leq \operatorname* { m a x } _ { w } c _ { w } .
$$

For any x, summing the layer bounds gives

$$
\| S x \| ^ { 2 } \leq \sum _ { w } c _ { w } \| x _ { w } \| ^ { 2 } , \qquad \| S ^ { \top } S x \| ^ { 2 } \leq \sum _ { w } \| S _ { w } \| ^ { 4 } \| x _ { w } \| ^ { 2 } \leq \sum _ { w } c _ { w } ^ { 2 } \| x _ { w } \| ^ { 2 } .
$$

Substituting $\| x _ { w } \| ^ { 2 } = \alpha _ { w } ( x ) \| x \| ^ { 2 }$ proves the claim.

Allocating the sketch dimensions. Theorem 4 shows that a layer containing little of a vector’s energy can tolerate a larger relative distortion. For Gaussian and Rademacher sketches, this allows a smaller sketch dimension, since $k _ { w }$ scales as $\varepsilon _ { w } ^ { - 2 }$ . The following allocation uses this weighted distortion budget for a fixed vector.

```latex
Corollary 2 (Energy-aware allocation)
Fix $x \neq 0$ and $\varepsilon , \delta \in ( 0 , 1 )$ . For every layer with $\alpha _ { w } ( x ) > 0$ , choose a Gaussian or Rademacher sketch with
$\varepsilon _ { w } = \frac { \varepsilon \alpha _ { w } ( x ) ^ { - 1 / 3 } } { \sum _ { j = 1 } ^ { W } \alpha _ { j } ( x ) ^ { 2 / 3 } } , \qquad \delta _ { w } = \frac { \delta } { W } , \qquad k _ { w } = \left\lceil C \frac { \log ( 2 W / \delta ) } { \varepsilon _ { w } ^ { 2 } } \right\rceil ,$
where $C > 0$ is a suficiently large universal constant. Assume that $\varepsilon _ { w } < 1$ and $k _ { w } \leq n _ { w }$ for all such layers.
Layers with $x _ { w } = 0$ are omitted, with $k _ { w } = 0$
Then, for this fixed vector,
$\begin{array} { r } { \mathbb { P } [ | | S x | | ^ { 2 } - | | x | | ^ { 2 } | \leq \varepsilon \| x \| ^ { 2 } ] \geq 1 - \delta , } \end{array}$
and the total number of transmitted coordinates satisfies
$\sum _ { w = 1 } ^ { W } k _ { w } = \Theta \left( \frac { \log ( 2 W / \delta ) } { \varepsilon ^ { 2 } } \left( \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) ^ { 2 / 3 } \right) ^ { 3 } \right) .$
```

Proof. The one-layer JL guarantees and a union bound give simultaneous success on all nonzero layers with prob ability at least $1 - \delta$ . Zero layers contribute no error for the fixed vector x. The total relative distortion on this event is at most

$$
\sum _ { w : \alpha _ { w } ( x ) > 0 } \alpha _ { w } ( x ) \varepsilon _ { w } = \varepsilon \frac { \sum _ { w : \alpha _ { w } ( x ) > 0 } \alpha _ { w } ( x ) ^ { 2 / 3 } } { \sum _ { j = 1 } ^ { W } \alpha _ { j } ( x ) ^ { 2 / 3 } } = \varepsilon ,
$$

as in Theorem 4.

To count coordinates, substitute the chosen distortions:

$$
\sum _ { w : \alpha _ { w } ( x ) > 0 } \frac { 1 } { \varepsilon _ { w } ^ { 2 } } = \frac { \left( \sum _ { j = 1 } ^ { W } \alpha _ { j } ( x ) ^ { 2 / 3 } \right) ^ { 2 } } { \varepsilon ^ { 2 } } \sum _ { w : \alpha _ { w } ( x ) > 0 } \alpha _ { w } ( x ) ^ { 2 / 3 } = \frac { 1 } { \varepsilon ^ { 2 } } \left( \sum _ { w = 1 } ^ { W } \alpha _ { w } ( x ) ^ { 2 / 3 } \right) ^ { 3 } .
$$

Since $0 < \varepsilon _ { w } < 1$ and $\log ( 2 W / \delta ) \geq \log 2$ , rounding each nonzero layer’s dimension changes it by at most a universal constant factor. Multiplying the last identity by log $( 2 W / \delta )$ therefore gives the stated total dimension. □

The allocation depends on the energy profile of x: its guarantee is for that fixed vector, with these dimensions chosen before the sketches are drawn. It applies only in the stated feasible regime $\varepsilon _ { w } < 1$ and $k _ { w } \leq n _ { w }$

## C.6 Kronecker sketches: compressing one tensor mode at a time

Per-layer sketches act on separate blocks. A Kronecker sketch instead applies several smaller matrices to diferent modes of the same tensor. We use the notation from the main text: for $X \in \mathbb { R } ^ { n _ { 1 } \times \cdots \times n _ { d } }$ and $S _ { j } \in \mathbb { R } ^ { k _ { j } \times n _ { j } }$ , define

$$
{ \cal S } ( X ) = X \times _ { 1 } S _ { 1 } \cdot \cdot \cdot \times _ { d } S _ { d } , \qquad { \cal S } = S _ { d } \otimes \cdot \cdot \cdot \otimes S _ { 1 } .
$$

Here $X \times _ { j } S _ { j }$ means multiplication along the jth mode. More explicitly, if $X _ { ( j ) }$ is the matrix formed by arranging the mode-j fibers of X as columns, then

$$
( X \times _ { j } S _ { j } ) _ { ( j ) } = S _ { j } X _ { ( j ) } , \qquad \operatorname { v e c } ( S ( X ) ) = S \operatorname { v e c } ( X ) .
$$

Both unfolding and vectorization preserve the Frobenius norm. Thus a single mode multiplication can be analyzed using a matrix Frobenius-norm bound. We prove this composition step once, then apply it to dense sketches and to general JL factors.

## Theorem 5 (Combining independent Frobenius-norm guarantees)

Let X be a fixed tensor and let $S _ { 1 } , \ldots , S _ { d }$ be independent random matrices. Fix $\varepsilon \in ( 0 , 1 )$ and $\delta _ { 1 } , \ldots , \delta _ { d } \in$ [0, 1]. Suppose that, for each $j$ and every fixed matrix V with $n _ { j }$ rows,

$$
\mathbb { P } [ \lvert \lvert S _ { j } V \rvert \rvert _ { F } ^ { 2 } - \lvert \lvert V \rvert \rvert _ { F } ^ { 2 }  > \frac { \varepsilon } { 4 d } \lvert \lvert V \rvert \rvert _ { F } ^ { 2 } ] \leq \delta _ { j } .
$$

Then

$$
\mathbb { P } \left[ \big | \| \boldsymbol { S } ( \boldsymbol { X } ) \| _ { F } ^ { 2 } - \| \boldsymbol { X } \| _ { F } ^ { 2 } \big | > \varepsilon \| \boldsymbol { X } \| _ { F } ^ { 2 } \right] \leq \sum _ { j = 1 } ^ { d } \delta _ { j } .
$$

Proof. Let $X ^ { ( 0 ) } = X$ and $X ^ { ( j ) } = X ^ { ( j - 1 ) } \times _ { j } S _ { j }$ denote the tensor after the first $j$ multiplications. Conditional on $S _ { 1 } , \ldots , S _ { j - 1 }$ , the tensor $X ^ { ( j - 1 ) }$ is fixed and $S _ { j }$ remains independent. The assumption applied to $( X ^ { ( j - 1 ) } ) _ { ( j ) }$ therefore gives

$$
\mathbb { P } \Bigg [ \Big | \| X ^ { ( j ) } \| _ { F } ^ { 2 } - \| X ^ { ( j - 1 ) } \| _ { F } ^ { 2 } \Big | > \frac { \varepsilon } { 4 d } \| X ^ { ( j - 1 ) } \| _ { F } ^ { 2 } ~ \Big | ~ S _ { 1 } , \ldots , S _ { j - 1 } \Bigg ] \leq \delta _ { j } .
$$

Taking expectations and a union bound, all d steps satisfy their bounds with probability at least $1 - \textstyle \sum _ { j } \delta _ { j }$ On this event,

$$
\left( 1 - \frac { \varepsilon } { 4 d } \right) ^ { d } \| X \| _ { F } ^ { 2 } \leq \| S ( X ) \| _ { F } ^ { 2 } \leq \left( 1 + \frac { \varepsilon } { 4 d } \right) ^ { d } \| X \| _ { F } ^ { 2 } .
$$

```latex
For $0 < \varepsilon < 1$ , Bernoulli’s inequality and $1 + u \leq e ^ { u }$ give
$\left( 1 - \frac { \varepsilon } { 4 d } \right) ^ { d } \geq 1 - \frac { \varepsilon } { 4 } \geq 1 - \varepsilon , \qquad \left( 1 + \frac { \varepsilon } { 4 d } \right) ^ { d } \leq e ^ { \varepsilon / 4 } \leq 1 + \varepsilon .$
This proves the claim. The proof also shows that it is enough to bound the conditional failure probability
at each step; a simultaneous guarantee over all possible intermediate tensors is not required. □
```

Theorem 6 (Gaussian and Rademacher Kronecker sketches)   
Let the factors $S _ { j } \in \mathbb { R } ^ { k _ { j } \times n _ { j } }$ be independent Gaussian or Rademacher JL matrices. There is a universal   
constant $c > 0$ such that, for every fixed tensor X and every $\varepsilon \in ( 0 , 1 )$   
$\mathbb { P } \left[ \left| \| S ( X ) \| _ { F } ^ { 2 } - \| X \| _ { F } ^ { 2 } \right| > \varepsilon \| X \| _ { F } ^ { 2 } \right] \le 2 \sum _ { j = 1 } ^ { d } e ^ { - c \varepsilon ^ { 2 } k _ { j } / d ^ { 2 } } \le 2 d e ^ { - c \varepsilon ^ { 2 } \operatorname* { m i n } _ { j } k _ { j } / d ^ { 2 } } .$   
Consequently, failure probability at most $\delta$ is ensured by   
$\operatorname* { m i n } _ { 1 \le j \le d } k _ { j } \ge C \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { 2 d } { \delta } ,$   
where $C > 0$ is a suficiently large universal constant. The same bounds hold for $\| S x \| ^ { 2 }$ with $x = \operatorname { v e c } ( X )$   
Proof. Lemma 14, with distortion $\varepsilon / ( 4 d )$ , bounds the failure probability at step j by $2 e ^ { - c _ { 1 } k _ { j } \varepsilon ^ { 2 } / ( 1 6 d ^ { 2 } ) }$ . Apply   
the conditioning argument of Theorem 5 and set $c = c _ { 1 } / 1 6$ . The dimension bound follows by requiring each   
of the d terms to be at most $\delta / d .$ Vectorization gives the final assertion. □

When only a fixed-vector JL guarantee is available. The preceding dense-sketch result uses the Frobeniusnorm estimate of Lemma 14, whose failure probability is independent of the matrix rank. For another construction, we can obtain a Frobenius-norm guarantee from fixed-vector bounds, at the cost of a rank factor.

```latex
Lemma 21 (From fixed vectors to a fixed matrix)
Fix $\varepsilon \in ( 0 , 1 )$ and $\delta \in [ 0 , 1 ]$ . Let $S \in \mathbb { R } ^ { k \times n }$ be a random matrix such that, for every fixed vector $u \in \mathbb { R } ^ { n }$
$\mathbb { P } \left[ \left| \| S u \| ^ { 2 } - \| u \| ^ { 2 } \right| > \varepsilon \| u \| ^ { 2 } \right] \leq \delta .$
Then, for every fixed matrix V with n rows,
$\begin{array} { r } { \mathbb { P } [ \| S V \| _ { F } ^ { 2 } - \| V \| _ { F } ^ { 2 } | > \varepsilon \| V \| _ { F } ^ { 2 } ] \leq \mathrm { r a n k } ( V ) \delta . } \end{array}$
Proof. The assertion is immediate for $V = 0 .$ Otherwise, let $u _ { i }$ be its left singular vectors and $\sigma _ { i } > 0$ its
nonzero singular values. Then
$\| S V \| _ { F } ^ { 2 } = \sum _ { i } \sigma _ { i } ^ { 2 } \| S u _ { i } \| ^ { 2 } , \qquad \| V \| _ { F } ^ { 2 } = \sum _ { i } \sigma _ { i } ^ { 2 } .$
There are rank(V) unit vectors $u _ { i } .$ A union bound ensures that all their squared norms are preserved
within ε with failure probability at most rank $( V ) \delta$ . Multiplying their bounds by $\sigma _ { i } ^ { 2 }$ and summing proves the
claim. □
```

Theorem 7 (Kronecker sketches from fixed-vector guarantees)   
Fix $ { \varepsilon } \in \left( 0 , 1 \right)$ and $\delta _ { 1 } , \hdots , \delta _ { d } \in [ 0 , 1 ]$ Let $X$ be fixed and let $S _ { 1 } , \ldots , S _ { d }$ be independent. Suppose that, for   
each $j$ and every fixed $u \in \mathbb { R } ^ { n _ { j } }$   
$\mathbb { P } \left[ \left| \| S _ { j } u \| ^ { 2 } - \| u \| ^ { 2 } \right| > \frac { \varepsilon } { 4 d } \| u \| ^ { 2 } \right] \leq \delta _ { j } .$   
Then   
$\mathbb { P } \left[ \left| \| S ( X ) \| _ { F } ^ { 2 } - \| X \| _ { F } ^ { 2 } \right| > \varepsilon \| X \| _ { F } ^ { 2 } \right] \leq \sum _ { j = 1 } ^ { d } \operatorname { r a n k } ( X _ { ( j ) } ) \delta _ { j } .$   
Proof. Use the intermediate tensors $X ^ { ( j ) }$ from Theorem 5. The first $j - 1$ mode multiplications only form   
linear combinations of columns in the mode-j unfolding. Hence   
$\operatorname { r a n k } \big ( ( X ^ { ( j - 1 ) } ) _ { ( j ) } \big ) \leq \operatorname { r a n k } ( X _ { ( j ) } ) .$   
Conditional on the first $j - 1$ factors, Lemma 21 therefore bounds the failure probability of step $j$ by   
rank $( X _ { ( j ) } ) \delta _ { j }$ . The same union bound and multiplication of stepwise norm bounds complete the proof. □   
The two composition theorems can be used at any desired distortion $\varepsilon \cdot$ evaluate each factor’s available failure   
bound at $\varepsilon / ( 4 d )$ and substitute it for $\delta _ { j }$ . For Gaussian and Rademacher factors, the direct Frobenius bound gives   
the stronger, rank-independent result above.   
The lifted operator. Finally, the algebra of Kronecker products transfers unbiasedness and operator-norm   
bounds from the factors to the full sketch.   
Lemma 22 (Unbiasedness and norm of Kronecker sketches)   
For independent factors $S _ { j } \in \mathbb { R } ^ { k _ { j } \times n _ { j } }$ , let $S = S _ { d } \otimes \cdot \cdot \cdot \otimes S _ { 1 } . \mathrm { ~ I f ~ } \mathbb { E } [ S _ { j } ^ { \top } S _ { j } ] = I _ { n _ { j } }$ for every $j ,$ then   
$\mathbb { E } [ S ^ { \top } S ] = I _ { n _ { 1 } \cdots n _ { d } } .$   
For every realization,   
$\| S \| = \prod _ { j = 1 } ^ { d } \| S _ { j } \| .$   
In particular, whenever $\| S _ { j } \| ^ { 2 } \leq c _ { j }$ for all $j ,$   
$\| S \| ^ { 2 } = \| S ^ { \top } S \| \leq \prod _ { j = 1 } ^ { d } c _ { j } , \qquad \| S ^ { \top } S x \| ^ { 2 } \leq \left( \prod _ { j = 1 } ^ { d } c _ { j } \right) ^ { 2 } \| x \| ^ { 2 }$ for all $x .$   
Proof. The mixed-product identity gives   
$S ^ { \top } S = ( S _ { d } ^ { \top } S _ { d } ) \otimes \cdots \otimes ( S _ { 1 } ^ { \top } S _ { 1 } ) .$   
Independence lets us take expectations factor by factor, proving unbiasedness. The singular values of a   
Kronecker product are products of the singular values of its factors, which proves the operator-norm identity.   
The remaining bounds follow from $\| S ^ { \top } S \| = \| S \| ^ { 2 }$ and $\| S ^ { \top } S x \| \leq \| S ^ { \top } S \| \| x \|$ □

These results separate the ingredients needed later: norm and inner-product preservation control geometry in the compressed space, while moments and operator norms control the lifted update. A structured construction can be used in the high-probability convergence theorem once its parameters provide all three sketch events specified in Appendix D.

## D Byzantine-robust Analysis

We prove convergence for fixed realized local and trusted objectives satisfying Assumptions 1, 2, and 3. The constants in these assumptions describe the optimization instance directly; no statistical model for the trusted dataset is imposed.

## D.1 Descent and convergence

Let $\mathcal { F } _ { t }$ denote the history before sampling $S _ { t } ,$ including the fixed objectives, the current iterate, and all previous randomness and messages, and write $\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t } ]$ . The gradients $g _ { 0 } ^ { t } = \nabla f _ { 0 } ( x ^ { t } ) , g ^ { t } = \nabla f ( x ^ { t } )$ , and $g _ { i } ^ { t } = \nabla f _ { i } ( x ^ { t } )$ for $i \in \mathcal G$ are $\mathcal { F } _ { t } { \mathrm { - m e a s u r a b l e } }$ . The sketch $S _ { t }$ is independent of $\mathcal { F } _ { t }$ , while Byzantine messages may depend on the current sketch and honest messages. No worker has access to future sketch randomness. All probabilities and expectations below keep the objectives and the initial iterate fixed.

We first derive an estimate for the scalar product that appears in both convergence proofs.

## Lemma 23

Let $x \in \mathbb { R } ^ { n }$ satisfy the reference-accuracy inequality in Assumption 3. Then, for every $\varepsilon \in [ 0 , 1 )$

$$
\langle \nabla f _ { 0 } ( x ) , \nabla f ( x ) \rangle - \varepsilon \| \nabla f _ { 0 } ( x ) \| \| \nabla f ( x ) \| \geq \frac { 1 - \eta _ { 1 } - \varepsilon ^ { 2 } } { 2 } \| \nabla f ( x ) \| ^ { 2 } - \frac { \eta _ { 0 } } { 2 } .\tag{14}
$$

Proof. Let $g _ { 0 } = \nabla f _ { 0 } ( x )$ and $g = \nabla f ( x )$ . By completing the square,

$$
\begin{array} { l l l } { \displaystyle \langle g _ { 0 } , g \rangle - \varepsilon \| g _ { 0 } \| \| g \| } & { = } & { \displaystyle \frac { 1 } { 2 } \left( \| g _ { 0 } \| - \varepsilon \| g \| \right) ^ { 2 } + \frac { 1 } { 2 } \left( ( 1 - \varepsilon ^ { 2 } ) \| g \| ^ { 2 } - \| g _ { 0 } - g \| ^ { 2 } \right) } \\ & { \ge } & { \displaystyle \frac { 1 - \eta _ { 1 } - \varepsilon ^ { 2 } } { 2 } \| g \| ^ { 2 } - \frac { \eta _ { 0 } } { 2 } , } \end{array}
$$

where the last inequality follows from Assumption 3.

Further, we establish the descent lemma used for convergence both in expectation and with high probability.

## Lemma 24

Let $f _ { 0 }$ have an L-Lipschitz gradient. Then every iteration of Algorithm 1 satisfies $f _ { 0 } ( x ^ { t + 1 } ) \leq f _ { 0 } ( x ^ { t } )$ and

$$
f _ { 0 } ( x ^ { t + 1 } ) \leq f _ { 0 } ( x ^ { t } ) - \frac { \gamma \beta } { M } \left. g _ { 0 } ^ { t } , \sum _ { i \in \mathcal { G } } u _ { i } ^ { t } \right. + \frac { L \gamma ^ { 2 } \beta } { 2 M } \sum _ { i \in \mathcal { G } } \| u _ { i } ^ { t } \| ^ { 2 } ,
$$

where $u _ { i } ^ { t } = S _ { t } ^ { \top } y _ { i } ^ { t }$ . For honest workers, $u _ { i } ^ { t } = S _ { t } ^ { \top } S _ { t } g _ { i } ^ { t }$

Proof. The initial weights are non-negative and sum to one, and both branches of the weight update preserve these properties. Set

$$
v ^ { t } = \sum _ { i = 1 } ^ { M } \omega _ { i } ^ { t } \mathbb { I } _ { [ \theta _ { i } ^ { t } > 0 ] } u _ { i } ^ { t } , \qquad q _ { i } ^ { t } = [ \theta _ { i } ^ { t } ] _ { 0 } , \qquad Q ^ { t } = \sum _ { i = 1 } ^ { M } q _ { i } ^ { t } .
$$

Applying convexity of the squared norm to the vectors $\mathbb { I } _ { [ \theta _ { i } ^ { t } > 0 ] } u _ { i } ^ { t }$ with weights $\omega _ { i } ^ { t }$ gives

$$
\| v ^ { t } \| ^ { 2 } \leq \sum _ { i = 1 } ^ { M } \omega _ { i } ^ { t } \mathbb { I } _ { [ \theta _ { i } ^ { t } > 0 ] } \| u _ { i } ^ { t } \| ^ { 2 } .
$$

Applying smoothness to the actual aggregated update, we obtain

$$
\begin{array} { r l } & { f _ { 0 } ( { x } ^ { t + 1 } ) \le f _ { 0 } ( { x } ^ { t } ) - \gamma \langle g _ { 0 } ^ { t } , { v } ^ { t } \rangle + \displaystyle \frac { L \gamma ^ { 2 } } { 2 } \| { v } ^ { t } \| ^ { 2 } \le f _ { 0 } ( { x } ^ { t } ) - \gamma \displaystyle \sum _ { i = 1 } ^ { M } \omega _ { i } ^ { t } \mathbb { I } _ { [ \theta _ { i } ^ { t } > 0 ] } \left( \langle g _ { 0 } ^ { t } , u _ { i } ^ { t } \rangle - \displaystyle \frac { L \gamma } { 2 } \| u _ { i } ^ { t } \| ^ { 2 } \right) } \\ & { \qquad = f _ { 0 } ( { x } ^ { t } ) - \gamma \displaystyle \sum _ { i = 1 } ^ { M } \omega _ { i } ^ { t } q _ { i } ^ { t } \le f _ { 0 } ( { x } ^ { t } ) . } \end{array}
$$

If $Q ^ { t } > 0$ , the weight update gives

$$
\sum _ { i = 1 } ^ { M } \omega _ { i } ^ { t } q _ { i } ^ { t } = ( 1 - \beta ) \sum _ { i = 1 } ^ { M } \omega _ { i } ^ { t - 1 } q _ { i } ^ { t } + \beta \frac { \sum _ { i = 1 } ^ { M } ( q _ { i } ^ { t } ) ^ { 2 } } { Q ^ { t } } \geq \beta \frac { \sum _ { i = 1 } ^ { M } ( q _ { i } ^ { t } ) ^ { 2 } } { Q ^ { t } } \geq \frac { \beta Q ^ { t } } { M } \geq \frac { \beta } { M } \sum _ { i \in \mathcal { G } } \theta _ { i } ^ { t } .
$$

The first inequality drops a non-negative term. The second follows from $\begin{array} { r } { ( Q ^ { t } ) ^ { 2 } \leq M \sum _ { i } ( q _ { i } ^ { t } ) ^ { 2 } } \end{array}$ , and the last uses $\begin{array} { r } { Q ^ { t } \geq \sum _ { i \in \mathcal { G } } q _ { i } ^ { t } \geq \sum _ { i \in \mathcal { G } } \theta _ { i } ^ { t } } \end{array}$ . If $Q ^ { t } = 0$ , then $v ^ { t } = 0$ and $\textstyle \sum _ { i \in { \mathcal { G } } } \theta _ { i } ^ { t } \leq 0$ . Thus, in both cases,

$$
f _ { 0 } ( x ^ { t + 1 } ) \leq f _ { 0 } ( x ^ { t } ) - \frac { \gamma \beta } { M } \sum _ { i \in \mathcal { G } } \theta _ { i } ^ { t } .
$$

Substituting $\begin{array} { r } { \theta _ { i } ^ { t } = \langle g _ { 0 } ^ { t } , u _ { i } ^ { t } \rangle - \frac { L \gamma } { 2 } \Vert u _ { i } ^ { t } \Vert ^ { 2 } } \end{array}$ proves the claimed bound. The argument uses smoothness of $f _ { 0 }$ and convexity of the squared norm, without assuming convexity of $f _ { 0 }$ □

We are now ready to present the convergence results. First, we derive convergence in expectation. To cover the diferent sketches mentioned above, we introduce two assumptions.

## Assumption 4 (Unbiasedness)

The lifted operator is conditionally unbiased if, for every $\mathcal { F } _ { t }$ -measurable vector $v \in \mathbb { R } ^ { n }$

$$
\mathbb { E } _ { t } [ S _ { t } ^ { \top } S _ { t } v ] = v .
$$

## Assumption 5 (Second moment boundedness in expectation)

There exists $\Omega \geq 1$ such that, for every $\mathcal { F } _ { t } .$ -measurable vector $v \in \mathbb { R } ^ { n }$ 2

$$
\mathbb { E } _ { t } \| S _ { t } ^ { \top } S _ { t } v \| ^ { 2 } \leq \Omega \| v \| ^ { 2 } .
$$

## Theorem 8

Suppose that Assumptions 1, 2, 3, 4, and 5 hold. Let $T \geq 1$ be an integer, $\beta \in ( 0 , 1 ] , \Delta _ { 0 } = f _ { 0 } ( x ^ { 0 } ) - f _ { 0 } ^ { \mathrm { i n f } }$ and choose $\gamma > 0$ such that

$$
D _ { \mathrm { E } } \stackrel { \mathrm { d e f } } { = } 1 - \eta _ { 1 } - L \gamma \Omega ( 1 + \delta _ { 2 } ) > 0 .
$$

Then, after T iterations of Algorithm 1,

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x ^ { t } ) \| ^ { 2 } \leq \frac { 2 M \Delta _ { 0 } } { \gamma \beta G D _ { \mathrm { E } } T } + \frac { \eta _ { 0 } + L \gamma \Omega \delta _ { 1 } } { D _ { \mathrm { E } } } .\tag{15}
$$

In particular, if $\begin{array} { r } { 0 < \gamma \leq \frac { 1 - \eta _ { 1 } } { 2 L \Omega ( 1 + \delta _ { 2 } ) } } \end{array}$ , then

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x ^ { t } ) \| ^ { 2 } \leq \frac { 4 M \Delta _ { 0 } } { \gamma \beta G ( 1 - \eta _ { 1 } ) T } + \frac { 2 \eta _ { 0 } } { 1 - \eta _ { 1 } } + \frac { 2 L \gamma \Omega \delta _ { 1 } } { 1 - \eta _ { 1 } } .\tag{16}
$$

Proof. Taking conditional expectation in Lemma 24 and using $\textstyle { \frac { 1 } { G } } \sum _ { i \in { \mathcal { G } } } g _ { i } ^ { t } = g ^ { t }$ , we obtain

$$
\begin{array} { r c l } { \displaystyle \mathbb { E } _ { t } f _ { 0 } ( x ^ { t + 1 } ) } & { \leq } & { f _ { 0 } ( x ^ { t } ) - \displaystyle \frac { \gamma \beta G } { M } \left. g _ { 0 } ^ { t } , g ^ { t } \right. + \displaystyle \frac { L \gamma ^ { 2 } \beta \Omega } { 2 M } \sum _ { i \in \mathcal { G } } \| g _ { i } ^ { t } \| ^ { 2 } } \\ & & { \leq } & { f _ { 0 } ( x ^ { t } ) - \displaystyle \frac { \gamma \beta G } { 2 M } \left( ( 1 - \eta _ { 1 } ) \| g ^ { t } \| ^ { 2 } - \eta _ { 0 } \right) + \displaystyle \frac { L \gamma ^ { 2 } \beta \Omega G } { 2 M } \left( ( 1 + \delta _ { 2 } ) \| g ^ { t } \| ^ { 2 } + \delta _ { 1 } \right) } \\ & { = } & { f _ { 0 } ( x ^ { t } ) - \displaystyle \frac { \gamma \beta G } { 2 M } \left( D _ { \mathrm { E } } \| g ^ { t } \| ^ { 2 } - \eta _ { 0 } - L \gamma \Omega \delta _ { 1 } \right) . } \end{array}
$$

Here, the second inequality uses Lemma 23 with $\varepsilon = 0$ and equation 7: Assumption 3 controls alignment, while Assumption 2 bounds the honest squared norms. Taking full expectation, summing over $t = 0 , \ldots , T -$ 1, and using $f _ { 0 } ( x ^ { T } ) \geq f _ { 0 } ^ { \mathrm { i n f } }$ gives equation 15. Under the displayed stepsize restriction, $D _ { \mathrm { E } } \geq ( 1 - \eta _ { 1 } ) / 2 ,$ which proves the last bound. □

For fresh Gaussian and Rademacher matrices, Lemmas 2 and 3 give $\Omega = ( n + k + 1 ) / k$ and $\Omega = ( n + k - 1 ) / k$ respectively. The expectation is over the subsequent algorithmic randomness for fixed objectives satisfying the stated assumptions.

We now derive convergence with high probability. The proof needs three events for each fresh sketch: preservation of one inner product, control of the total compressed honest energy, and an operator-norm bound.

## Theorem 9 (Theorem 1 from the main part)

Under the assumptions and notation of Theorem 1, except for its stepsize restriction, define

$$
c _ { \gamma } \stackrel { \mathrm { d e f } } { = } L \gamma \rho ( 1 + \varepsilon ) , \qquad D _ { \gamma } \stackrel { \mathrm { d e f } } { = } d - ( 1 + \delta _ { 2 } ) c _ { \gamma } .
$$

More generally, for every $\gamma > 0$ with $D _ { \gamma } > 0$ , we have

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \nabla f ( x ^ { t } ) \| ^ { 2 } \leq \frac { 2 M \Delta _ { 0 } } { \gamma \beta G D _ { \gamma } T } + \frac { \eta _ { 0 } + c _ { \gamma } \delta _ { 1 } } { D _ { \gamma } }\tag{17}
$$

with probability at least $1 - \delta .$ The stepsize in Theorem 1 gives equation 11.

Proof. Set $q = \delta / ( 4 T )$ and let $V _ { t } \in \mathbb { R } ^ { n \times G }$ be the matrix whose columns are the honest gradients $g _ { i } ^ { t } .$ . Conditional on $\mathcal { F } _ { t }$ , both $V _ { t }$ and the pair $( g _ { 0 } ^ { t } , g ^ { t } )$ are fixed. Lemma 14 gives

$$
\mathbb { P } \left[ \| S _ { t } V _ { t } \| _ { F } ^ { 2 } > ( 1 + \varepsilon ) \| V _ { t } \| _ { F } ^ { 2 } ~ | ~ \mathcal { F } _ { t } \right] \le 2 e ^ { - c _ { 1 } k \varepsilon ^ { 2 } } \le q
$$

when $k \geq C \varepsilon ^ { - 2 } \log ( 8 T / \delta )$ for a suficiently large universal constant C. The same lemma for a single column gives the fixed-vector JL bound with failure probability at most q. Applying Lemma 7 then gives

$$
\left| \left. S _ { t } g _ { 0 } ^ { t } , S _ { t } g ^ { t } \right. - \left. g _ { 0 } ^ { t } , g ^ { t } \right. \right| \leq \varepsilon \| g _ { 0 } ^ { t } \| \| g ^ { t } \|
$$

with conditional failure probability at most $2 q$ . Finally, Lemma 4 gives $\| S _ { t } \| ^ { 2 } \leq \rho$ with conditional failure probability at most $q .$ Thus, with conditional probability at least $1 - 4 q .$ all three events hold.

On their intersection, Lemma 23 implies

$$
\left. g _ { 0 } ^ { t } , \frac { 1 } { G } \sum _ { i \in \mathcal { G } } u _ { i } ^ { t } \right. \ = \ \left. S _ { t } g _ { 0 } ^ { t } , S _ { t } g ^ { t } \right. \ge \left. g _ { 0 } ^ { t } , g ^ { t } \right. - \varepsilon \| g _ { 0 } ^ { t } \| \| g ^ { t } \| \ge \frac { d } { 2 } \| g ^ { t } \| ^ { 2 } - \frac { \eta _ { 0 } } { 2 } .
$$

Moreover, using the operator norm and the Frobenius norm in this order,

$$
\begin{array} { l l l } { \displaystyle \sum _ { i \in \mathcal { G } } \| u _ { i } ^ { t } \| ^ { 2 } } & { = } & { \| S _ { t } ^ { \top } S _ { t } V _ { t } \| _ { F } ^ { 2 } \leq \| S _ { t } \| ^ { 2 } \| S _ { t } V _ { t } \| _ { F } ^ { 2 } \leq \rho ( 1 + \varepsilon ) \| V _ { t } \| _ { F } ^ { 2 } } \\ & & { \leq } & { \rho ( 1 + \varepsilon ) G \left( ( 1 + \delta _ { 2 } ) \| g ^ { t } \| ^ { 2 } + \delta _ { 1 } \right) . } \end{array}
$$

Substituting these two estimates into Lemma 24 yields

$$
\begin{array} { r c l } { f _ { 0 } ( x ^ { t + 1 } ) } & { \le } & { f _ { 0 } ( x ^ { t } ) - \displaystyle \frac { \gamma \beta G } { 2 M } \left( d \| g ^ { t } \| ^ { 2 } - \eta _ { 0 } \right) + \displaystyle \frac { L \gamma ^ { 2 } \beta \rho ( 1 + \varepsilon ) G } { 2 M } \left( ( 1 + \delta _ { 2 } ) \| g ^ { t } \| ^ { 2 } + \delta _ { 1 } \right) } \\ & { = } & { f _ { 0 } ( x ^ { t } ) - \displaystyle \frac { \gamma \beta G } { 2 M } \left( D _ { \gamma } \| g ^ { t } \| ^ { 2 } - \eta _ { 0 } - c _ { \gamma } \delta _ { 1 } \right) . } \end{array}
$$

Let $\mathcal { E } _ { t }$ be the intersection of the three sketch events at iteration t. Although the iterates are adaptive, their conditional failure probabilities are bounded for every history. Therefore,

$$
\mathbb { P } \left[ \bigcup _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { t } ^ { c } \right] \leq \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \mathbb { P } ( \mathcal { E } _ { t } ^ { c } \mid \mathcal { F } _ { t } ) \right] \leq 4 T q = \delta .
$$

On $\textstyle \bigcap _ { t = 0 } ^ { T - 1 } \mathcal E _ { t }$ , summing the descent inequality and using $f _ { 0 } ( x ^ { T } ) \geq f _ { 0 } ^ { \mathrm { i n f } }$ gives

$$
\frac { \gamma \beta G D _ { \gamma } } { 2 M } \sum _ { t = 0 } ^ { T - 1 } \| g ^ { t } \| ^ { 2 } \leq \Delta _ { 0 } + \frac { \gamma \beta G T } { 2 M } \left( \eta _ { 0 } + c _ { \gamma } \delta _ { 1 } \right) .
$$

Dividing by $\gamma \beta G D _ { \gamma } T / ( 2 M )$ proves equation 17. Finally, the stepsize in Theorem 1 gives $D _ { \gamma } \geq d / 2$ , which proves equation 11. □

The probability estimates involve only honest gradients and the trusted reference, all fixed before the fresh sketch is sampled. No geometric preservation event is imposed on Byzantine messages. The same proof applies to another sketch construction if it provides the three conditional events above with the stated failure probabilities and a suitable value of $\rho ;$ the required dimension must then be established for that construction.