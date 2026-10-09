# NP-Hardness of Minimizing Neurons in Two-Hidden-Layer ReLU Neural Networks

Sangrock Lee

Abstract—A fundamental question in neural network architecture optimization is whether the minimum hidden-neuron count required to approximate a target function within a prescribed tolerance can be computed efficiently. This paper resolves this question for two-hidden-layer ReLU networks under an L<sup>p</sup>(R<sup>d</sup>, R<sup>m</sup>) approximation constraint. For every fixed d ≥ 1, m ≥ 1, and $1 \leq p < \infty ,$ , we prove that computing the optimum exactly is NP-hard. The result holds even when the target is represented by a rational ReLU network whose realization is nonzero, componentwise nonnegative, compactly supported, globally Lipschitz, and continuous piecewise affine. The polynomial-time reduction from 3-SAT produces an architecture gap in which unsatisfiable formulas yield an optimum of zero, whereas satisfiable formulas yield an optimum of at least d + 2. The proof constructs compactly supported polyhedral frustum functions realized by two-hidden-layer ReLU networks and establishes the L<sup>p</sup>-density of finite linear combinations of box-frustum functions. The results offer theoretical justification for employing heuristic approximation methods in the design of ReLU neural networks, illustrating that attaining a minimal configuration within polynomial time is computationally unachievable.

Index Terms—NP-hardness, ReLU neural network, L<sup>p</sup> function, 3-SAT.

## 1 INTRODUCTION

O <sup>PTIMIZING</sup> <sup>the</sup> <sup>number</sup> <sup>of</sup> <sup>neurons</sup> <sup>in</sup> <sup>deep</sup> <sup>learning</sup> architectures is critical for balancing computational efficiency and model performance. Oversized networks increase the risk of overfitting, while undersized ones may fail to capture essential data structures [1]. Given the extensive use of feedforward neural networks in applications such as image classification [2], [3], natural language processing [4], [5], and scientific computing [6], [7], there is a need for methods to design minimal yet effective models [8], [9]. Despite efforts through analytic approaches, heuristic methods, and complexity analysis, determining the optimal number of parameters remains unresolved. Our research examines the fundamental computational hurdles in optimizing ReLU network structures and provides a complexity-theoretic rationale for approximation-based methods in network design by proving a worst-case NP-hardness result for exact hidden-neuron minimization.

Extensive work has sought analytical descriptions of neural network architectures that achieve universal function approximation. Cybenko [10] showed that a single-hiddenlayer network with sigmoid activation can approximate any continuous function on a compact domain arbitrarily well, provided that a sufficient finite number of hidden neurons is chosen according to the desired accuracy. Hornik et al. [11] extended this result to multilayer feedforward networks, establishing universality for Borel-measurable functions. Maiorov and Pinkus [12] showed that uniform approximation of any continuous function on a compact domain with a two-hidden-layer network requires a width that grows linearly with the input dimension. Their proof employs specially tailored sigmoidal activations; analytic, strictly monotone realizations of such functions are provided by Guliyev and Ismailov [13]. Because these activations are not standard choices like ReLU or tanh, they must be evaluated separately, which may limit practical applicability. Gripenberg [14] demonstrated that bounded width networks can achieve universality when depth is allowed to increase. Pinkus [15] surveys theoretical properties of multilayer feedforward neural networks, including universal approximation, derivative approximation, interpolation, and approximation bounds, with an emphasis on singlehidden-layer models and brief coverage of deeper architectures. Ismailov [16] reviews width-minimality results that depend on input dimension. In one dimension, with suitably chosen sigmoidal activations, a single hidden neuron can achieve uniform approximation, so the required width does not depend on target accuracy. For dimensions greater than one, for any fixed width, single-hidden-layer networks cannot approximate all continuous multivariate functions arbitrarily well, so the required width must increase as the accuracy requirement tightens. For two hidden layers and a computable sigmoidal activation, universality is attainable with width that grows linearly with the input dimension. Further studies [17], [18], [19] refined bounds on necessary network width for universal approximation. Park et al. [20] established minimal neuron requirements for ReLU networks for universal approximation in L<sup>p</sup> space when the layer depth is allowed to grow. Cai [21] generalized the findings of Park et al. [20] to arbitrary activation functions. These analytical insights have advanced understanding, yet the precise configurations of neural networks with universal approximation capability remain under active investigation.

In addressing the challenge of determining the optimal configuration of neural networks, heuristic approaches have played a vital role, as evidenced by Neural Architecture Search (NAS). Zoph et al. [22] introduced a scalable and transferable NAS for image classification, utilizing a controller to enhance the search process’s efficiency. Efficient

Neural Architecture Search (ENAS) [23] accelerates the NAS process by sharing parameters among child models, thereby reducing the computational resources required for architecture search. Differentiable Architecture Search (DARTS) [24] demonstrates that NAS can be conducted through the optimization of network architectures via gradient descent within a continuous search space. Alvarez and Salzmann<sup>´</sup> [25] introduced a method that dynamically optimizes the number of neurons in each layer during training by leveraging a group sparsity regularizer. Lee and Lee [26] presented NeuralScale, a framework employing iterative pruning and ’architecture descent’ to optimize neuron configurations, improving accuracy in architecture-fixed settings. Kapanova et al. [27] proposed an evolutionary algorithm for neural network architecture search, considering neurons, layers, weights, and activation functions. Although significant progress has been made, identifying the optimal network size remains a challenging and unresolved problem in NAS, highlighting the complexity and exploratory nature of current methodologies.

Different aspects of neural networks are examined through the lens of computational complexity, using networks with various widths and depths. Blum and Rivest [28] demonstrate that the process of training a neural network, characterized by a single hidden layer with merely two nodes employing a step activation function and a single-node output, is classified as an NP-complete problem. This finding underscores the inherent computational difficulties that exist even in the most basic configurations of neural networks. Furthermore, this outcome has been generalized to scenarios involving the training of neural networks with ReLU activation functions [29], [30]. Froese and Hertrich [31] have shown that the training of neural networks is NP-hard in fixed dimensions. Bertschinger et al. [32] established that training a neural network with a single hidden layer is ∃R-complete. Despite these insights, the complexity aspects of structural optimization in neural network models remain largely unexplored.

This paper establishes that exact minimization of the hidden-neuron count in two-hidden-layer ReLU networks is NP-hard for every fixed input dimension $d \geq 1 .$ , every fixed output dimension $m \geq 1 .$ , and every $1 \leq p < \infty$ . The result holds even for rational target networks whose realizations are nonzero, componentwise nonnegative, compactly supported, globally Lipschitz, and continuous piecewise affine. The reduction distinguishes instances with optimum zero from instances with optimum at least $d + { \hat { 2 } } $ . This result identifies a worst-case computational obstruction to exact architecture selection and provides a complexity-theoretic basis for heuristic, approximate, or restricted architectureselection methods for ReLU neural networks.

The remainder of this paper is organized as follows. Section II fixes the notation for ReLU neural networks, hidden-layer widths, total neuron counts, and bounded polytopes. Section III constructs compactly supported polyhedral frustum functions from bounded polytopes and shows that these functions are realized by two-hidden-layer ReLU networks. Section IV proves that finite linear combinations of compactly supported box-frustum functions are dense in $L ^ { p } ( \mathbb { R } ^ { \hat { d } } )$ and extends this approximation result to vector-valued functions. Section V defines the exact twohidden-layer neuron minimization problem, proves an $L ^ { p }$ degeneracy result for networks of width at most the input dimension, constructs a satisfiability-dependent target with a nonzero nonnegative baseline, and derives an architecture gap that implies NP-hardness. Section VI concludes with implications for exact architecture selection and heuristic network design.

## 2 PRELIMINARIES

Definition 1 (ReLU neural network). Let $\sigma : \mathbb { R } $ R be the ReLU activation function defined by

$$
\sigma ( t ) = \operatorname* { m a x } \{ 0 , t \} .
$$

A feedforward ReLU neural network with input dimension $d ,$ output dimension $m ,$ and L hidden layers is defined by

$$
\begin{array} { c } { { { \bf h } ^ { ( 0 ) } = { \bf x } , } } \\ { { { \bf h } ^ { ( l ) } = \sigma \left( W ^ { ( l ) } { \bf h } ^ { ( l - 1 ) } + { \bf b } ^ { ( l ) } \right) } } \end{array}
$$

$$
\begin{array} { c } { { f o r l = 1 , \ldots , L , a n d } } \\ { { { \ } } } \\ { { { \bf y } = W ^ { ( L + 1 ) } { \bf h } ^ { ( L ) } + { \bf b } ^ { ( L + 1 ) } . } } \end{array}
$$

Here $\mathbf { x } \in \mathbb { R } ^ { d } , \mathbf { y } \in \mathbb { R } ^ { m }$ , the ReLU activation is applied componentwise, $W ^ { ( l ) }$ is a real matrix, and $\mathbf { b } ^ { ( l ) }$ is a real vector. The width of the l-th hidden layer is the number of components of $\mathbf { h } ^ { ( l ) }$ . The total number of hidden neurons is the sum of the widths of all hidden layers.

Definition 2 (Bounded polytope). A bounded polytope in $\mathbb { R } ^ { d }$ is a nonempty bounded set of the form

$$
P = \bigcap _ { i = 1 } ^ { q } \left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \ell _ { i } ( \mathbf { x } ) \geq 0 \right\} ,
$$

where $q \in \mathbb { N }$ and

$$
\ell _ { i } ( { \boldsymbol { \mathbf { x } } } ) = { \boldsymbol { \mathbf { a } } } _ { i } ^ { \top } { \boldsymbol { \mathbf { x } } } + b _ { i }
$$

for some $\mathbf { a } _ { i } \in \mathbb { R } ^ { d }$ and $b _ { i } \in \mathbb { R }$

## 3 FORMATION OF FRUSTUM FUNCTIONS

A frustum is the portion of a solid geometry, such as a cone or pyramid, lying between two parallel cutting planes. Frustums are illustrated in Fig. 1. The construction below uses a ReLU network to produce a compactly supported polyhedral plateau function. The function is equal to one on a bounded polytope and vanishes outside a compact set. On the intermediate set where its values lie strictly between zero and one, the function is piecewise affine, so its graph consists of a flat plateau over the polytope together with polyhedral lateral faces. After the total violation function is introduced, this intermediate set is defined precisely as the transition region. The resulting function is called a frustum function.

It starts from an arbitrary nonempty bounded polytope

$$
P = \bigcap _ { i = 1 } ^ { q } \left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \ell _ { i } ( \mathbf { x } ) \geq 0 \right\} ,
$$

where

$$
\ell _ { i } ( { \mathbf { x } } ) = { \mathbf { a } } _ { i } ^ { \top } { \mathbf { x } } + b _ { i } .
$$

![](images/2b89a76e3b83542927b1566d2e82730a6e925ca2b1d3c389bd54e73a33c3a958.jpg)

![](images/2a05638a0b5b0294f1c5b43517d8f6c5862dc530566dac8533053f01acf102fa.jpg)  
(a)  
(b)  
Fig. 1. Illustration of frustums with polygonal bases. (a) Square-based frustum with parallel square top and bottom faces. (b) Hexagon-based frustum with parallel hexagonal bases connected by trapezoidal lateral faces.

Let $c _ { i } > 0$ for $i = 1 , \ldots , q ,$ , and write

$$
\mathbf { c } = ( c _ { 1 } , \hdots , c _ { q } ) .
$$

The first hidden layer computes the violation variables

$$
u _ { i } ( { \bf x } ) = \sigma ( - c _ { i } \ell _ { i } ( { \bf x } ) ) .
$$

Thus $u _ { i } ( { \bf x } ) = 0$ if the i-th inequality holds, and $u _ { i } ( { \bf x } ) > 0$ if the i-th inequality fails. Define the total violation

$$
V _ { P , \mathbf { c } } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { q } \sigma ( - c _ { i } \ell _ { i } ( \mathbf { x } ) ) .
$$

For $\rho > 0 ,$ , define

$$
\Theta _ { P , \rho , \mathbf { c } } ( \mathbf { x } ) = \sigma \left( 1 - \frac { 1 } { \rho } V _ { P , \mathbf { c } } ( \mathbf { x } ) \right) .
$$

Equivalently,

$$
\Theta _ { P , \rho , \mathbf { c } } ( \mathbf { x } ) = \sigma \left( 1 - \frac { 1 } { \rho } \sum _ { i = 1 } ^ { q } \sigma ( - c _ { i } \ell _ { i } ( \mathbf { x } ) ) \right) .
$$

The transition region is

$$
\Delta _ { P , \rho , \mathbf { c } } = \left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid 0 < V _ { P , \mathbf { c } } ( \mathbf { x } ) < \rho \right\} .
$$

On $P ,$ all inequalities are satisfied, so $V _ { P , { \bf c } } = 0$ and $\Theta _ { P , \rho , \mathbf { c } } =$ 1. On $\Delta _ { P , \rho , \mathbf { c } } ,$ the function is positive and piecewise affine. Whenever $V _ { P , { \bf c } } \ge \rho ,$ the function is zero. Hence

$$
\mathrm { s u p p } \Theta _ { P , \rho , \mathbf { c } } \subset \left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid V _ { P , \mathbf { c } } ( \mathbf { x } ) \leq \rho \right\} .
$$

This construction has a direct functional interpretation. The first hidden layer measures how much each defining inequality of $P$ is violated. Thus all violation variables are zero on ${ \dot { P } } .$ The total violation $V _ { P , \mathbf { c } }$ then determines the value of the scalar function $\Theta _ { P , \rho , \mathbf { c } }$ . Consequently, P is the unitlevel plateau of $\Theta _ { P , \rho , \mathbf { c } } ,$ points with $\bar { 0 } < V _ { P , { \bf c } } < \rho$ form the transition region, and points with $V _ { P , { \bf c } } \ge \rho$ are mapped to zero.

Lemma 3 (Polyhedral frustum realization). Let $P$ be a nonempty bounded polytope in $\mathbb { R } ^ { d }$ of the form

$$
P = \bigcap _ { i = 1 } ^ { q } \left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \ell _ { i } ( \mathbf { x } ) \geq 0 \right\} ,
$$

where

$$
\ell _ { i } ( { \boldsymbol { \mathbf { x } } } ) = { \boldsymbol { \mathbf { a } } } _ { i } ^ { \top } { \boldsymbol { \mathbf { x } } } + b _ { i } .
$$

Let $c _ { i } > 0 f o r i = 1 , \ldots , q ,$ let $\mathbf { c } = ( c _ { 1 } , \ldots , c _ { q } )$ , and let $\rho > 0 .$ Then

$$
0 \leq \Theta _ { P , \rho , \mathbf { c } } \leq 1 ,
$$

$$
\Theta _ { P , \rho , \mathbf { c } } = 1
$$

on $P ,$ and $\Theta _ { P , \rho , \mathbf { c } }$ is compactly supported. Moreover, $\Theta _ { P , \rho , \mathbf { c } }$ is realized by a two-hidden-layer ReLU network with $q$ neurons in the first hidden layer and one neuron in the second hidden layer.

Proof. For every $i ,$ the quantity $\sigma ( - c _ { i } \ell _ { i } ( { \bf x } ) )$ is nonnegative. Hence

$$
V _ { P , \mathbf { c } } ( \mathbf { x } ) \geq 0 ,
$$

and therefore

$$
1 - \frac { 1 } { \rho } V _ { P , \mathbf { c } } ( \mathbf { x } ) \leq 1 .
$$

Since ReLU is nonnegative and satisfies $\sigma ( t ) \leq 1$ whenever $t \leq 1 .$ , it follows that

$$
0 \leq \Theta _ { P , \rho , \mathbf { c } } ( \mathbf { x } ) \leq 1
$$

for every $\mathbf { x } \in \mathbb { R } ^ { d }$

If ${ \textbf { x } } \in { \textbf { \textit { P } } } ,$ , then $\ell _ { i } ( { \bf x } ) \geq 0$ for every i. Since $c _ { i } > 0 ,$ , we have

$$
\sigma ( - c _ { i } \ell _ { i } ( { \bf x } ) ) = 0
$$

for every i. Hence

$$
V _ { P , \mathbf { c } } ( \mathbf { x } ) = 0 ,
$$

and so

$$
\Theta _ { P , \rho , \mathbf { c } } ( \mathbf { x } ) = 1 .
$$

The network realization follows from the displayed formula. The first hidden layer computes the $q$ functions

$$
\sigma ( - c _ { i } \ell _ { i } ( { \bf x } ) )
$$

for $i = 1 , \ldots , q .$ The second hidden layer computes

$$
\sigma \left( 1 - \frac { 1 } { \rho } \sum _ { i = 1 } ^ { q } \sigma ( - c _ { i } \ell _ { i } ( { \bf x } ) ) \right) .
$$

The affine output layer is the identity.

It remains to prove compact support. Choose $\mathbf { x } _ { 0 } \in P .$ Since $P$ is bounded, it cannot contain a ray starting from $\mathbf { x } _ { 0 } .$ Hence there is no nonzero vector v such that

$$
\mathbf { a } _ { i } ^ { \top } \mathbf { v } \geq 0
$$

for every $i = 1 , \ldots , q .$ . For contradiction, assume that such a nonzero vector v exists, then for every $t \geq 0$ and every $i = 1 , \ldots , q ,$

$$
\ell _ { i } ( \mathbf { x } _ { 0 } + t \mathbf { v } ) = \ell _ { i } ( \mathbf { x } _ { 0 } ) + t \mathbf { a } _ { i } ^ { \top } \mathbf { v } \geq 0 .
$$

Thus the point ${ \bf x } _ { 0 } +$ tv satisfies every defining inequality of $P .$ Hence

$$
\left\{ \mathbf { x } _ { 0 } + t \mathbf { v } \mid t \geq 0 \right\} \subset P .
$$

Since $\mathbf { v } \neq 0 ,$

$$
\begin{array} { r } { \| \mathbf { x } _ { 0 } + t \mathbf { v } \| _ { 2 } \geq t \| \mathbf { v } \| _ { 2 } - \| \mathbf { x } _ { 0 } \| _ { 2 } , } \end{array}
$$

and the right-hand side becomes arbitrarily large as $t \to \infty$ Therefore $\scriptstyle { \dot { P } }$ would be unbounded, contradicting the boundedness of P. Let

$$
S = \left\{ \mathbf { v } \in \mathbb { R } ^ { d } \mid \| \mathbf { v } \| _ { 2 } = 1 \right\} .
$$

Fix $\mathbf { v } \in S .$ Then $\mathbf { v } \neq 0 .$ . If

$$
\mathbf { a } _ { i } ^ { \top } \mathbf { v } \geq 0
$$

for all $i = 1 , \ldots , q ,$ then v would be a nonzero vector with the property ruled out above. Hence these inequalities cannot all hold for this chosen direction v. Therefore there exists an index $i = i ( \mathbf { v } )$ such that

$$
\mathbf { a } _ { i } ^ { \top } \mathbf { v } < 0 .
$$

Define

$$
\gamma ( \mathbf { v } ) = \sum _ { j = 1 } ^ { q } \sigma ( - c _ { j } \mathbf { a } _ { j } ^ { \top } \mathbf { v } ) .
$$

For the index $i = i ( \mathbf { v } )$ found above, since $c _ { i } > 0 .$ , we have

$$
- c _ { i } \mathbf { a } _ { i } ^ { \top } \mathbf { v } > 0 .
$$

Thus

$$
\sigma ( - c _ { i } \mathbf { a } _ { i } ^ { \top } \mathbf { v } ) > 0 .
$$

Each summand in $\gamma ( \mathbf { v } )$ is a ReLU output, so no summand is negative. Hence this positive summand cannot be canceled, and therefore

$$
\gamma ( \mathbf { v } ) > 0 .
$$

Since the choice of $\mathbf { v } \in S$ was arbitrary, γ is positive on $S .$

The function γ is continuous because it is a finite sum of continuous functions. Since $S$ is compact, the minimum

$$
\gamma _ { 0 } = \operatorname* { m i n } _ { \mathbf { v } \in S } \gamma ( \mathbf { v } )
$$

exists. Since $\gamma$ is positive on $S ,$ it follows that $\gamma _ { 0 } ~ > ~ 0$ Therefore

$$
\gamma ( \mathbf { v } ) \geq \gamma _ { 0 }
$$

for $\mathbf { v } \in S .$ Let

$$
B = \sum _ { i = 1 } ^ { q } c _ { i } | b _ { i } | .
$$

For $\mathbf { x } \neq 0 ,$ , write $\mathbf { x } = t \mathbf { v } ,$ where $t = \| \mathbf { x } \| _ { 2 }$ and $\| \mathbf { v } \| _ { 2 } = 1$ Using the inequality $\sigma ( r - s ) \geq \sigma ( r ) - | s |$ , we obtain

$$
\begin{array} { l } { { \displaystyle V _ { P , { \bf c } } ( t { \bf v } ) = \sum _ { i = 1 } ^ { q } \sigma ( - c _ { i } ( t { \bf a } _ { i } ^ { \top } { \bf v } + b _ { i } ) ) } \ ~ } \\ { { \displaystyle = \sum _ { i = 1 } ^ { q } \sigma ( - \mathrm { l c } _ { i } { \bf a } _ { i } ^ { \top } { \bf v } - c _ { i } b _ { i } ) } \ ~ } \\ { { \displaystyle \geq \sum _ { i = 1 } ^ { q } \sigma ( - \mathrm { l c } _ { i } { \bf a } _ { i } ^ { \top } { \bf v } ) - \sum _ { i = 1 } ^ { q } c _ { i } \vert b _ { i } \vert } \ ~ } \\ { { \displaystyle ~ = t \sum _ { i = 1 } ^ { q } \sigma ( - c _ { i } { \bf a } _ { i } ^ { \top } { \bf v } ) - B } \ ~ } \\ { { \displaystyle \geq \mathrm { l } \gamma _ { 0 } - B } . } \end{array}
$$

If

$$
t > \frac { \rho + B } { \gamma _ { 0 } } ,
$$

then

$$
V _ { P , \mathbf { c } } ( t \mathbf { v } ) > \rho .
$$

For such $t ,$ the input to the outer ReLU in the definition of $\Theta _ { P , \rho , \mathbf { c } }$ is negative. Hence

$$
\begin{array} { r } { \Theta _ { P , \rho , \mathbf { c } } ( t \mathbf { v } ) = 0 . } \end{array}
$$

![](images/18788341e68097ea54e68b9e4cee9ea3389d79e7fea9e19ce320aca6e352a463.jpg)

Fig. 2. Transition region of a frustum function. The central set labeled $P$ is the bounded polytope, where the function equals one. The surrounding set labeled $\Delta _ { P , \rho , \mathsf { c } }$ represents the transition region, where $0 < V _ { P , \mathbf { c } } < \rho$ and the function decreases piecewise affinely from one to zero

Thus $\Theta _ { P , \rho , \mathbf { c } }$ vanishes outside the Euclidean ball

$$
\left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \| \mathbf { x } \| _ { 2 } \leq \frac { \rho + B } { \gamma _ { 0 } } \right\} .
$$

Therefore $\Theta _ { P , \rho , \mathbf { c } }$ is compactly supported.

□

Fig. 2 illustrates the transition region. The central set labeled P is the bounded polytope, where $V _ { P , { \bf c } } = 0$ and $\Theta _ { P , \rho , \mathbf { c } } = 1$ . The surrounding set labeled $\Delta _ { P , \rho , c }$ represents the transition region where $\begin{array} { r } { 0 < V _ { P , \mathbf { c } } < \rho . } \end{array}$

The constants $c _ { i }$ control the slopes of the lateral faces. Multiplying an inequality by $c _ { i } > 0$ does not change the polytope, because

$$
c _ { i } \ell _ { i } ( { \bf x } ) \geq 0
$$

is equivalent to

$$
\ell _ { i } ( \mathbf { x } ) \geq 0 .
$$

However, it changes how fast the frustum function decreases after the corresponding inequality is violated.

Let $I \subset \{ 1 , \ldots , q \}$ be an active violation set. On a region where

$$
\ell _ { i } ( { \bf x } ) < 0
$$

for $i \in I ,$

$$
\ell _ { i } ( \mathbf { x } ) > 0
$$

for $i \not \in I ,$ and

$$
0 < V _ { P , { \bf c } } ( { \bf x } ) < \rho ,
$$

the outer ReLU is active and the active set is fixed. On that region,

$$
\Theta _ { P , \rho , \mathbf { c } } ( \mathbf { x } ) = 1 + \frac { 1 } { \rho } \sum _ { i \in I } c _ { i } \ell _ { i } ( \mathbf { x } ) .
$$

Therefore

$$
\nabla \Theta _ { P , \rho , \mathbf { c } } ( \mathbf { x } ) = \frac { 1 } { \rho } \sum _ { i \in I } c _ { i } \mathbf { a } _ { i } .
$$

In particular, if only the i-th inequality is violated, then

$$
\nabla \Theta _ { P , \rho , \mathbf { c } } ( \mathbf { x } ) = \frac { c _ { i } } { \rho } \mathbf { a } _ { i } .
$$

Thus increasing $c _ { i }$ increases the magnitude of the corresponding transition slope, and decreasing $c _ { i }$ decreases it. $\mathrm { A t }$ points where the active violation set changes or where $V _ { P , { \bf c } } = \rho ,$ the function may have a kink. This behavior is illustrated in Fig. 3.

![](images/0206e22ae659cb1b1cac7bf9b795553e8e9ab200aba1c32d79ee1644aaab0da2.jpg)  
Fig. 3. Effect of scaling factors on a one-dimensional frustum function. The plateau is the interval $E _ { 1 }$ . The sloped parts are transition regions. Increasing $c _ { L }$ increases the magnitude of the left transition slope, while decreasing $c _ { R }$ decreases the magnitude of the right transition slope.

When

$$
P = \prod _ { j = 1 } ^ { d } [ \alpha _ { j } , \beta _ { j } ]
$$

is an axis-aligned bounded box and all scaling constants are equal to one, the frustum function takes the form

$$
\Theta _ { P , \rho } ( \mathbf { x } ) = \sigma \left( 1 - \frac { 1 } { \rho } \sum _ { j = 1 } ^ { d } ( \sigma ( \alpha _ { j } - x _ { j } ) + \sigma ( x _ { j } - \beta _ { j } ) ) \right) .
$$

This function equals one on $P$ and vanishes outside

$$
P ^ { \rho } = \prod _ { j = 1 } ^ { d } [ \alpha _ { j } - \rho , \beta _ { j } + \rho ] .
$$

Such a function is called a box-frustum function.

## 4 FUNCTION APPROXIMATION BY FRUSTUM FUNCTIONS

The previous section showed that frustum functions associated with nonempty bounded polytopes are realized by two-hidden-layer ReLU networks and are compactly supported. For the approximation argument, it is enough to use the special case where the polytope is an axis-aligned bounded box. This section proves that finite linear combinations of compactly supported box-frustum functions are dense in $L ^ { p } ( { \dot { \mathbb { R } } } ^ { d } )$ . The proof uses the standard density of finite linear combinations of characteristic functions of bounded boxes and replaces each box indicator by a boxfrustum function with a sufficiently thin transition region.

Lemma 4 (Frustum density). Let d ≥ 1, let $1 \leq p < \infty ,$ and let $f \in L ^ { p } ( \mathbb { R } ^ { d } )$ . For every $\varepsilon > 0 ,$ there exists a scalarvalued two-hidden-layer ReLU network ϕfrom $\mathbb { R } ^ { d }$ to R such that $\phi \in L ^ { p } (  { \mathbb { R } } ^ { d } )$ and

$$
\| f - \phi \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } < \varepsilon .
$$

Moreover, $\phi$ can be chosen as a finite linear combination of compactly supported box-frustum functions.

Proof. Since finite linear combinations of characteristic functions of bounded boxes are dense in $L ^ { p } ( \mathbb { R } ^ { d } )$ , there

exist bounded boxes $Q _ { 1 } , \ldots , Q _ { N } \subset \mathbb { R } ^ { d }$ and constants $c _ { 1 } , \ldots , c _ { N } \in \mathbb { R }$ such that

$$
g = \sum _ { k = 1 } ^ { N } c _ { k } \chi _ { Q _ { k } }
$$

satisfies

$$
\| f - g \| _ { L ^ { p } (  { \mathbb { R } } ^ { d } ) } < \frac { \varepsilon } { 2 } .
$$

Write

$$
Q _ { k } = \prod _ { i = 1 } ^ { d } [ a _ { k i } , b _ { k i } ] .
$$

For $\rho > 0 ,$ , define

$$
\theta _ { k , \rho } ( \mathbf { x } ) = \sigma \left( 1 - \frac { 1 } { \rho } \sum _ { i = 1 } ^ { d } \left( \sigma ( a _ { k i } - x _ { i } ) + \sigma ( x _ { i } - b _ { k i } ) \right) \right) .
$$

By Lemma $^ { 3 , }$ each $\theta _ { k , \rho }$ is a compactly supported twohidden-layer ReLU box-frustum function. Moreover,

$$
0 \leq \theta _ { k , \rho } \leq 1
$$

on $\mathbb { R } ^ { d } ,$

$$
\theta _ { k , \rho } = 1
$$

on $Q _ { k } ,$ and

$$
\theta _ { k , \rho } = 0
$$

outside

$$
Q _ { k } ^ { \rho } = \prod _ { i = 1 } ^ { d } [ a _ { k i } - \rho , b _ { k i } + \rho ] .
$$

Define

$$
\phi _ { \rho } = \sum _ { k = 1 } ^ { N } c _ { k } \theta _ { k , \rho } .
$$

This function is realized by one two-hidden-layer ReLU network by placing the subnetworks for $\theta _ { 1 , \rho } , \ldots , \theta _ { N , \rho }$ in parallel and combining their outputs in the affine output layer.

Since

$$
\mu ( Q _ { k } ^ { \rho } \backslash Q _ { k } )  0
$$

as $\rho \to 0 ,$ choose $\rho > 0$ such that

$$
N ^ { p - 1 } \sum _ { k = 1 } ^ { N } | c _ { k } | ^ { p } \mu ( Q _ { k } ^ { \rho } \setminus Q _ { k } ) < \left( \frac { \varepsilon } { 2 } \right) ^ { p } .
$$

Since $\theta _ { k , \rho } - \chi _ { Q }$ 2k vanishes outside $Q _ { k } ^ { \rho } \setminus Q _ { k }$ and has absolute value at most 1, the discrete Holder inequality gives

$$
| \phi _ { \rho } ( \mathbf { x } ) - g ( \mathbf { x } ) | ^ { p } \leq N ^ { p - 1 } \sum _ { k = 1 } ^ { N } | c _ { k } | ^ { p } | \theta _ { k , \rho } ( \mathbf { x } ) - \chi _ { Q _ { k } } ( \mathbf { x } ) | ^ { p } .
$$

Therefore

$$
\| \phi _ { \rho } - g \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } ^ { p } \leq N ^ { p - 1 } \sum _ { k = 1 } ^ { N } | c _ { k } | ^ { p } \mu ( Q _ { k } ^ { \rho } \setminus Q _ { k } ) < \left( \frac { \varepsilon } { 2 } \right) ^ { p } .
$$

Hence

$$
\| \phi _ { \rho } - g \| _ { L ^ { p } (  { \mathbb { R } } ^ { d } ) } < \frac { \varepsilon } { 2 } .
$$

By the triangle inequality,

$$
\| f - \phi _ { \rho } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } \leq \| f - g \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } + \| g - \phi _ { \rho } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } < \varepsilon .
$$

Setting $\phi = \phi _ { \rho }$ proves the claim.

Corollary 5 (Vector-valued frustum density). Let $d \ge 1$ let $m \geq 1$ , let $1 \ \leq \ p \ < \ \infty ,$ , and let $\textbf { f } = \mathbf { \widetilde { \Gamma } } ( f _ { 1 } , \dots , f _ { m } ) \ \in$ $L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } )$ , where

$$
\| \mathbf { f } \| _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } = \left( \int _ { \mathbb { R } ^ { d } } \sum _ { j = 1 } ^ { m } | f _ { j } ( \mathbf { x } ) | ^ { p } \mathrm { d } \mathbf { x } \right) ^ { 1 / p } .
$$

For every $\varepsilon > 0$ , there exists a two-hidden-layer ReLU network Φ from $\mathbb { R } ^ { d }$ to $\mathbb { R } ^ { m }$ such that

$$
\| \mathbf { f } - \Phi \| _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } < \varepsilon .
$$

Each coordinate of Φ can be chosen as a finite linear combination of compactly supported box-frustum functions.

Proof. For each $j = 1 , \dots , m ,$ Lemma 4 gives a scalar-valued two-hidden-layer ReLU network $\phi _ { j }$ such that

$$
\| f _ { j } - \phi _ { j } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } < \varepsilon m ^ { - 1 / p } .
$$

Place these scalar networks in parallel and let the affine output layer return

$$
\Phi = ( \phi _ { 1 } , \ldots , \phi _ { m } ) .
$$

Then

$$
\| \mathbf { f } - \Phi \| _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } ^ { p } = \sum _ { j = 1 } ^ { m } \| f _ { j } - \phi _ { j } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } ^ { p } < \sum _ { j = 1 } ^ { m } \varepsilon ^ { p } m ^ { - 1 } = \varepsilon ^ { p } .
$$

Taking the p-th root gives

$$
\| \mathbf { f } - \Phi \| _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } < \varepsilon .
$$

## 5 COMPUTATIONAL COMPLEXITY OF MINIMIZING NEURON NUMBERS IN RELU NETWORKS

Definition 6 (Two-hidden-layer neuron minimization). Fix integers $d \geq 1$ and $m \geq 1 ,$ , and fix $1 \leq p <$ ∞. For a measurable function

$$
\mathbf { g } = ( g _ { 1 } , \ldots , g _ { m } ) : \mathbb { R } ^ { d }  \mathbb { R } ^ { m } ,
$$

define

$$
\| \mathbf { g } \| _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } = \left( \int _ { \mathbb { R } ^ { d } } \sum _ { j = 1 } ^ { m } | g _ { j } ( \mathbf { x } ) | ^ { p } \mathrm { d } \mathbf { x } \right) ^ { 1 / p } .
$$

Let $\mathfrak { N } _ { N , d , m } ^ { ( 2 ) }$ be the class of functions from $\mathbb { R } ^ { d }$ to $\mathbb { R } ^ { m }$ realized by ReLU networks with two hidden layers and at most N hidden neurons in total. The parameters of an approximating network may be arbitrary real numbers. A network having a zero-width hidden layer is interpreted as a constant function.

For a rational ReLU target network

$$
\mathcal { S } : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }
$$

whose realization $\mathbf { f } _ { \mathcal { S } }$ belongs to $L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } )$ , and for a positive rational number $\varepsilon ,$ define

$$
\begin{array} { r l } & { \mathrm { O P T } _ { p , d , m } ( \mathcal { S } , \varepsilon ) = \operatorname* { m i n } \left\{ N \in \mathbb { N } _ { 0 } \middle | \exists \Phi \in \mathfrak { N } _ { N , d , m } ^ { ( 2 ) } \right. } \\ & { \left. s u c h t h a t \ \rVert \Phi - \mathbf { f } _ { S } \Vert _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } < \varepsilon \right\} . } \end{array}
$$

The minimum of the empty set is defined as $+ \infty$

Lemma 7 (Width-d $L ^ { p }$ degeneracy). Let $d \ge 1 , m \ge 1$ and $1 \leq p < \infty .$ . Let

$$
\Phi : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }
$$

be realized by a ReLU network in which every hidden layer has width at most d. If

$$
\Phi \in L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) ,
$$

then

$$
\Phi = 0 .
$$

Proof. For every output coordinate $j ,$ the scalar function $\Phi _ { j }$ is realized by a ReLU network whose hidden layers have width at most d. The scalar width-d degeneracy result states that such a network is either identically zero or does not belong to $L ^ { p } ( \mathbb { R } ^ { d } )$ [20, Appendix B.3]. Since

$$
\Phi _ { j } \in L ^ { p } ( \mathbb { R } ^ { d } ) ,
$$

it follows that

$$
\Phi _ { j } = 0
$$

for every $j = 1 , \ldots , m$ . Therefore

$$
\Phi = 0 .
$$

□

Lemma 8 (Minimum nonzero $L ^ { p }$ neuron count). Let $d \geq 1$ m $\geq 1$ and $1 \leq p < \infty$ . Let

$$
\Phi : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }
$$

be a nonzero function realized by a two-hidden-layer ReLU network. ${ \cal I } f$

$$
\Phi \in L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) ,
$$

then the network has at least $d + 2$ hidden neurons in total.

Proof. Let $n _ { 1 }$ and $n _ { 2 }$ denote the widths of the two hidden layers. If either width is zero, then Φ is constant. Since the only constant function in $L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } )$ is the zero function, this contradicts the assumption that Φ $\neq 0 .$ Hence

$$
n _ { 1 } \geq 1
$$

and

$$
n _ { 2 } \geq 1 .
$$

Suppose that

$$
n _ { 1 } + n _ { 2 } \leq d + 1 .
$$

Since both widths are positive,

$$
n _ { 1 } \leq d
$$

and

$$
n _ { 2 } \leq d .
$$

Thus, every hidden layer has width at most d. Lemma 7 gives

$$
\Phi = 0 ,
$$

which is a contradiction. Thus

$$
n _ { 1 } + n _ { 2 } \ge d + 2 .
$$

□

Lemma 9 (Frustum function satisfiability target). Fix $d \geq 1$ and $1 \leq p < \infty$ . Given a 3-CNFformula Ψ with $n \geq 1$ variables

and $s \geq 1$ clauses, one can construct in polynomial time a rational ReLU network whose realization

$$
f _ { \Psi , d } : \mathbb { R } ^ { d }  \mathbb { R }
$$

is nonnegative, compactly supported, globally Lipschitz, and continuous piecewise affine. Its support satisfies

$$
\operatorname { s u p p } f _ { \Psi , d } \subset [ 0 , 1 ] ^ { d } .
$$

With

$$
\eta = 2 ^ { - ( n + 4 ) }
$$

and

$$
\nu _ { \eta , d } = 2 \eta \left( \frac { 1 } { 2 } \right) ^ { d - 1 } ,
$$

the construction satisfies

$$
\Psi \not \in 3 \ – S A T \Longrightarrow f _ { \Psi , d } = 0
$$

and

$$
\Psi \in 3 \mathrm { - } S A T \Longrightarrow \| f _ { \Psi , d } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } \geq \nu _ { \eta , d } ^ { 1 / p } .
$$

$I f \Psi$ is satisfiable, there exists a measurable box of measure $\nu _ { \eta , d }$ on which

$$
f _ { \Psi , d } = 1 .
$$

Proof. Write the variables of Ψ as $X _ { 1 } , \ldots , X _ { n }$ and its clauses as $\dot { C } _ { 1 } , \dots , \dot { C } _ { s }$ . Define

$$
B _ { \eta } ( t ) = \frac { \sigma \left( t - \frac { 1 } { 2 } + \eta \right) - \sigma \left( t - \frac { 1 } { 2 } - \eta \right) } { 2 \eta } .
$$

For every $t \in \mathbb R$

$$
B _ { \eta } ( t ) = \left\{ \begin{array} { l l } { 0 } & { t \leq \frac { 1 } { 2 } - \eta , } \\ { \displaystyle \frac { t - \frac { 1 } { 2 } + \eta } { 2 \eta } } & { \frac { 1 } { 2 } - \eta \leq t \leq \frac { 1 } { 2 } + \eta , } \\ { 1 } & { t \geq \frac { 1 } { 2 } + \eta . } \end{array} \right.
$$

Thus

$$
0 \leq B _ { \eta } ( t ) \leq 1 .
$$

For $x \in \mathbb { R } ,$ , define

$$
r _ { 0 } ( x ) = \sigma ( x ) .
$$

For $i = 1 , \ldots , n ,$ define recursively

$$
z _ { i } ( x ) = B _ { \eta } ( r _ { i - 1 } ( x ) )
$$

and

$$
r _ { i } ( x ) = \sigma \left( 2 r _ { i - 1 } ( x ) - z _ { i } ( x ) \right) .
$$

For $x \in [ 0 , 1 ] ,$ , one has

$$
r _ { 0 } ( x ) = x \in [ 0 , 1 ] .
$$

We now show that if $r _ { i - 1 } ( x ) \in [ 0 , 1 ]$ , then $r _ { i } ( x ) \in [ 0 , 1 ]$ Indeed, suppose that $r \in [ 0 , 1 ]$ . If

$$
r \leq { \frac { 1 } { 2 } } - \eta ,
$$

then $B _ { \eta } ( r ) = 0$ and

$$
0 \leq 2 r \leq 1 - 2 \eta .
$$

If

$$
r \geq { \frac { 1 } { 2 } } + \eta ,
$$

then $B _ { \eta } ( r ) = 1$ and

$$
0 \leq 2 r - 1 \leq 1 .
$$

On the transition interval, the function

$$
2 r - B _ { \eta } ( r )
$$

is affine with slope

$$
2 - \frac { 1 } { 2 \eta } < 0 .
$$

Its maximum on that interval is attained at $\begin{array} { r } { r = \frac { 1 } { 2 } - \eta } \end{array}$ and equals $1 - 2 \eta$ . Applying ReLU therefore produces a value in $[ 0 , 1 ]$ . Consequently,

$$
r _ { i } ( x ) \in [ 0 , 1 ]
$$

and

$$
z _ { i } ( x ) \in [ 0 , 1 ]
$$

for every $x \in [ 0 , 1 ] .$

For $\dot { \bf z } = ( z _ { 1 } , \dots , z _ { n } ) \in [ 0 , 1 ] ^ { n }$ , assign a numerical value to each literal. The value of $X _ { i }$ is $z _ { i } ,$ and the value of its negation $\neg X _ { i }$ is $1 - z _ { i } .$ Thus,

$$
\ell _ { X _ { i } } ( \mathbf { z } ) = z _ { i }
$$

and

$$
\ell _ { \neg X _ { i } } ( \mathbf { z } ) = 1 - z _ { i } .
$$

If the literals of $C _ { j }$ are

$$
\lambda _ { j 1 } , \lambda _ { j 2 } , \lambda _ { j 3 } ,
$$

define

$$
d _ { j } ( \mathbf { z } ) = \sigma \left( 1 - \ell _ { \lambda _ { j 1 } } ( \mathbf { z } ) - \ell _ { \lambda _ { j 2 } } ( \mathbf { z } ) - \ell _ { \lambda _ { j 3 } } ( \mathbf { z } ) \right) .
$$

For a Boolean vector, $d _ { j }$ equals one exactly when $C _ { j }$ is false.

Define

$$
P ( \mathbf { z } ) = \sum _ { i = 1 } ^ { n } \left( z _ { i } - \sigma ( 2 z _ { i } - 1 ) \right) .
$$

Choose a nearest Boolean vector

$$
\bar { \mathbf { z } } \in \{ 0 , 1 \} ^ { n }
$$

coordinatewise. For every $i ,$

$$
z _ { i } - \sigma ( 2 z _ { i } - 1 ) = | z _ { i } - \bar { z } _ { i } | .
$$

Hence

$$
P ( \mathbf { z } ) = \sum _ { i = 1 } ^ { n } | z _ { i } - \bar { z } _ { i } | .
$$

Define

$$
S _ { \Psi } ( \mathbf { z } ) = 1 - \sum _ { j = 1 } ^ { s } d _ { j } ( \mathbf { z } ) - ( 3 s + 1 ) P ( \mathbf { z } )
$$

and

$$
G _ { \Psi } ( x ) = \sigma \left( S _ { \Psi } \left( z _ { 1 } ( x ) , \ldots , z _ { n } ( x ) \right) \right) .
$$

Since ReLU is 1-Lipschitz,

$$
| d _ { j } ( { \bf z } ) - d _ { j } ( \bar { \bf z } ) | \leq \sum _ { k = 1 } ^ { 3 } | \ell _ { \lambda _ { j k } } ( { \bf z } ) - \ell _ { \lambda _ { j k } } ( \bar { \bf z } ) | .
$$

Each literal is either $X _ { i } \ { \mathrm { o r } } \ { \neg } X _ { i } .$

For $X _ { i } ,$

$$
\vert \ell _ { X _ { i } } ( \mathbf { z } ) - \ell _ { X _ { i } } ( \bar { \mathbf { z } } ) \vert = \vert z _ { i } - \bar { z } _ { i } \vert .
$$

For $\neg X _ { i } ,$

$$
\begin{array} { c } { { \vert \ell _ { - X _ { i } } ( { \bf z } ) - \ell _ { - X _ { i } } ( { \bf \bar { z } } ) \vert = \vert ( 1 - z _ { i } ) - ( 1 - \bar { z } _ { i } ) \vert } } \\ { { { } = \vert z _ { i } - \bar { z } _ { i } \vert . } } \end{array}
$$

Since

$$
P ( \mathbf { z } ) = \sum _ { i = 1 } ^ { n } | z _ { i } - \bar { z } _ { i } | ,
$$

each literal difference is at most $P ( \mathbf { z } )$ . A clause has three literals, so the sum of its three literal differences is at most

$$
3 P ( \mathbf { z } ) .
$$

Therefore,

$$
| d _ { j } ( \mathbf { z } ) - d _ { j } ( \bar { \mathbf { z } } ) | \leq 3 P ( \mathbf { z } ) .
$$

Suppose that Ψ is unsatisfiable. Every Boolean vector falsifies at least one clause, so

$$
\sum _ { j = 1 } ^ { s } d _ { j } ( \bar { \mathbf { z } } ) \geq 1 .
$$

It follows that

$$
\sum _ { j = 1 } ^ { s } d _ { j } ( { \bf z } ) \geq \sum _ { j = 1 } ^ { s } d _ { j } ( \bar { \bf z } ) - 3 s P ( { \bf z } ) \geq 1 - 3 s P ( { \bf z } ) .
$$

Consequently,

$$
S _ { \Psi } ( \mathbf { z } ) \leq - P ( \mathbf { z } ) \leq 0 .
$$

Thus

$$
G _ { \Psi } ( x ) = 0
$$

for every $x \in [ 0 , 1 ]$

Suppose that Ψ is satisfiable. Let

$$
\mathbf { a } = ( a _ { 1 } , \ldots , a _ { n } ) \in \{ 0 , 1 \} ^ { n }
$$

be a satisfying assignment and define

$$
x _ { \mathbf { a } } = \sum _ { i = 1 } ^ { n } a _ { i } 2 ^ { - i } + 2 ^ { - ( n + 1 ) } .
$$

For $i = 1 , \ldots , n ,$ define

$$
\Delta _ { i } = 2 ^ { - ( n - i + 2 ) } .
$$

After the first i−1 digits have been extracted, the remainder at $x _ { a }$ is

$$
r _ { i - 1 } ( x _ { a } ) = \sum _ { k = i } ^ { n } a _ { k } 2 ^ { - ( k - i + 1 ) } + 2 ^ { - ( n - i + 2 ) } .
$$

If $a _ { i } = 1$ , then

$$
r _ { i - 1 } ( x _ { \mathbf { a } } ) \geq \frac { 1 } { 2 } + \Delta _ { i } .
$$

If $a _ { i } = 0 .$ , then

$$
r _ { i - 1 } ( x _ { \mathbf { a } } ) \leq \frac { 1 } { 2 } - \Delta _ { i } .
$$

Since

$$
\Delta _ { i } > \eta ,
$$

the remainder lies outside the transition interval of $B _ { \eta } .$ Hence

$$
z _ { i } ( x _ { \mathbf { a } } ) = a _ { i } .
$$

The update then removes the leading binary digit and gives the displayed remainder formula for the next index.

Let

$$
I _ { \mathbf { a } } = [ x _ { \mathbf { a } } - \eta , x _ { \mathbf { a } } + \eta ] .
$$

For any $x \in I _ { \mathbf { a } }$

$$
| r _ { 0 } ( x ) - r _ { 0 } ( x _ { \mathsf { a } } ) | \leq \eta .
$$

We have shown that $z _ { i } ( x _ { \mathbf { a } } ) = a _ { i } ,$ and we now prove by induction on i that

$$
z _ { i } ( x ) = z _ { i } ( x _ { \mathbf { a } } ) = a _ { i }
$$

for every $x \in I _ { \mathbf { a } }$ and every $i = 1 , \ldots , n$ . When $i = 1 ,$ , there is no index $t < i ,$ so the induction hypothesis is vacuously true. Suppose that

$$
z _ { t } ( x ) = z _ { t } ( x _ { \mathbf { a } } ) = a _ { t }
$$

for every $t < i .$ . Since ReLU is 1-Lipschitz,

$$
| r _ { t } ( x ) - r _ { t } ( x _ { a } ) | \leq 2 | r _ { t - 1 } ( x ) - r _ { t - 1 } ( x _ { a } ) | .
$$

Therefore

$$
| r _ { i - 1 } ( x ) - r _ { i - 1 } ( x _ { \mathfrak { a } } ) | \le 2 ^ { i - 1 } \eta .
$$

Moreover,

$$
2 ^ { i - 1 } \eta + \eta \le 2 ^ { i } \eta = \frac 1 4 \Delta _ { i } < \Delta _ { i } .
$$

If $a _ { i } = 1 .$ , then

$$
r _ { i - 1 } ( x ) > \frac { 1 } { 2 } + \eta .
$$

If $a _ { i } = 0 ,$ , then

$$
r _ { i - 1 } ( x ) < \frac { 1 } { 2 } - \eta .
$$

Thus

$$
z _ { i } ( x ) = a _ { i } .
$$

Induction gives

$$
z _ { i } ( x ) = a _ { i }
$$

for every $i = 1 , \ldots , n$ and every $x \in I _ { \mathbf { a } }$ . Since a satisfies Ψ,

$$
d _ { j } ( { \bf a } ) = 0
$$

for every $j ,$ and

$$
P ( \mathbf { a } ) = 0 .
$$

Consequently,

$$
G _ { \Psi } ( x ) = 1
$$

for every $x \in I _ { \mathsf { a } }$

Define

$$
E _ { \eta , d } = [ 4 \eta , 1 - 4 \eta ] \times \left[ \frac { 1 } { 4 } , \frac { 3 } { 4 } \right] ^ { d - 1 } .
$$

When $d = 1$ , this means

$$
E _ { \eta , 1 } = [ 4 \eta , 1 - 4 \eta ] .
$$

Define

$$
\begin{array} { c } { { F _ { \eta , d } ( { \bf x } ) = \sigma \left( 1 - \displaystyle \frac { 1 } { 4 \eta } \left( \sigma ( 4 \eta - x _ { 1 } ) + \sigma ( x _ { 1 } - 1 + 4 \eta ) \right. \right. } } \\ { { \displaystyle \left. \left. + \sum _ { j = 2 } ^ { d } \left( \sigma \left( \displaystyle \frac { 1 } { 4 } - x _ { j } \right) + \sigma \left( x _ { j } - \displaystyle \frac { 3 } { 4 } \right) \right) \right) \right) . } } \end{array}
$$

This box-frustum function equals one on $E _ { \eta , d } .$ . Its support satisfies

For nonnegative u and $v ,$ define

$$
\operatorname* { m n } ( u , v ) = \sigma \left( u - \sigma ( u - v ) \right) .
$$

Then

$$
\operatorname* { m n } ( u , v ) = \operatorname* { m i n } \{ u , v \} .
$$

Set

$$
f _ { \Psi , d } ( \mathbf { x } ) = \mathrm { m n } \left( G _ { \Psi } ( x _ { 1 } ) , F _ { \eta , d } ( \mathbf { x } ) \right) .
$$

It follows that

$$
f _ { \Psi , d } ( \mathbf x ) = \mathrm { m i n } \left\{ G _ { \Psi } ( x _ { 1 } ) , F _ { \eta , d } ( \mathbf x ) \right\} .
$$

In particular,

$$
\mathrm { s u p p } f _ { \Psi , d } \subset \mathrm { s u p p } F _ { \eta , d } \subset [ 0 , 1 ] ^ { d } .
$$

Suppose that $\Psi$ is unsatisfiable. If

$$
F _ { \eta , d } ( \mathbf { x } ) > 0 ,
$$

then $x _ { 1 } \in [ 0 , 1 ] ,$ , and hence

$$
G _ { \Psi } ( x _ { 1 } ) = 0 .
$$

If

$$
F _ { \eta , d } ( \mathbf { x } ) = 0 ,
$$

the minimum defining $f _ { \Psi , d }$ is also zero. Therefore

$$
f _ { \Psi , d } = 0
$$

on $\mathbb { R } ^ { d } .$

Suppose that $\Psi$ is satisfiable. Since

$$
2 ^ { - ( n + 1 ) } = 8 \eta ,
$$

the definition of $x _ { \mathbf { a } }$ gives

$$
x _ { \mathbf { a } } \in [ 8 \eta , 1 - 8 \eta ] .
$$

Consequently,

$$
I _ { \mathbf { a } } \subset [ 7 \eta , 1 - 7 \eta ] \subset [ 4 \eta , 1 - 4 \eta ] .
$$

Define

$$
R _ { \mathbf { a } } = I _ { \mathbf { a } } \times \left[ \frac { 1 } { 4 } , \frac { 3 } { 4 } \right] ^ { d - 1 } .
$$

When $d = 1$ , this means

$$
R _ { \mathbf { a } } = I _ { \mathbf { a } } .
$$

On $R _ { \mathbf { a } }$

$$
G _ { \Psi } ( x _ { 1 } ) = 1
$$

and

$$
F _ { \eta , d } ( \mathbf { x } ) = 1 .
$$

Therefore

$$
f _ { \Psi , d } = 1
$$

on $R _ { \mathbf { a } }$ . Its measure is

$$
\mu ( R _ { \mathbf { a } } ) = 2 \eta \left( \frac { 1 } { 2 } \right) ^ { d - 1 } = \nu _ { \eta , d } .
$$

It follows that

$$
\| f _ { \Psi , d } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } ^ { p } \geq \nu _ { \eta , d }
$$

and

$$
\| f _ { \Psi , d } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } \geq \nu _ { \eta , d } ^ { 1 / p } .
$$

Every function in the construction is obtained from rational affine maps and ReLU activations. Previously computed nonnegative quantities can be propagated through later layers by identity ReLU neurons. The sequential extraction network uses polynomially many neurons, and the clause computation and box-frustum construction add polynomially many neurons. The coefficients

$$
\frac { 1 } { 2 \eta } = 2 ^ { n + 3 }
$$

and

$$
\frac { 1 } { 4 \eta } = 2 ^ { n + 2 }
$$

have polynomial bit length. Thus a rational target network realizing $f _ { \Psi , d }$ can be constructed in polynomial time. Its realization is nonnegative, continuous piecewise affine, globally Lipschitz, and compactly supported. □

Theorem 10 (Architecture gap). Fix integers $d \geq 1$ and $m \geq$ 1, and $f i x \ 1 \leq \ p \ < \ \infty$ . Given a 3-CNF formula $\Psi ,$ one can construct in polynomial time a rational ReLU target network

$$
S _ { \Psi , d , m } : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }
$$

and a positive rational tolerance $\varepsilon _ { \Psi , d }$ satisfying

$$
\Psi \not \in 3 \not - S A T \Longrightarrow { \mathrm { O P T } } _ { p , d , m } \left( S _ { \Psi , d , m } , \varepsilon _ { \Psi , d } \right) = 0
$$

and

$$
\Psi \in 3 { \cal - S } A T \Longrightarrow \mathrm { O P T } _ { p , d , m } \left( S _ { \Psi , d , m } , \varepsilon _ { \Psi , d } \right) \geq d + 2 .
$$

Every constructed target is nonzero, componentwise nonnegative, compactly supported, globally Lipschitz, and continuous piecewise affine.

Consequently, the language

$$
\mathsf { L A R G E } _ { p , d , m } = \{ ( S , \varepsilon ) | \mathrm { O P T } _ { p , d , m } ( S , \varepsilon ) \geq d + 2 \}
$$

is NP-hard. Exact two-hidden-layer ReLU neuron minimization is NP-hardfor everyfixed input dimension $d \geq 1$ and every fixed output dimension $m \geq 1$

Proof. It is enough to reduce from the restriction of 3-SAT to formulas having at least one variable and one clause. This restriction remains NP-hard.

Let Ψ have $n \geq 1$ variables. Construct $f _ { \Psi , d } , \eta ,$ , and $\nu _ { \eta , d }$ according to Lemma 9. Thus

$$
\eta = 2 ^ { - ( n + 4 ) }
$$

and

$$
\nu _ { \eta , d } = 2 \eta \left( { \frac { 1 } { 2 } } \right) ^ { d - 1 } = 2 ^ { - n - d - 2 } .
$$

Define

$$
b _ { d } ( \mathbf { x } ) = \sigma \left( 1 - 4 \sum _ { j = 1 } ^ { d } \left( \sigma ( 2 - x _ { j } ) + \sigma \left( x _ { j } - \frac 9 4 \right) \right) \right) .
$$

This is the box-frustum function associated with

$$
Q _ { d } = \left[ 2 , \frac { 9 } { 4 } \right] ^ { d }
$$

and transition parameter $1 / 4 .$ It satisfies

$$
0 \leq b _ { d } \leq 1 ,
$$

equals one on $\mathrm { \Delta } Q \mathrm { \Delta } _ { d } ,$ and has a transition region where the total violation in its definition lies strictly between 0 and $1 / 4$ . If

$$
b _ { d } ( \mathbf { x } ) > 0 ,
$$

then every individual violation in its definition is less than $1 / 4 .$ . Consequently,

$$
{ \mathrm { s u p p } } b _ { d } \subset \left[ \frac { 7 } { 4 } , \frac { 5 } { 2 } \right] ^ { d } .
$$

Therefore

$$
\left\| b _ { d } \right\| _ { L ^ { p } \left( \mathbb { R } ^ { d } \right) } ^ { p } \leq \mu \left( \left[ { \frac { 7 } { 4 } } , { \frac { 5 } { 2 } } \right] ^ { d } \right) = \left( { \frac { 3 } { 4 } } \right) ^ { d } \leq 1 ,
$$

and hence

$$
\| b _ { d } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } \leq 1 .
$$

Set

$$
\varepsilon _ { \Psi , d } = \frac { \nu _ { \eta , d } } { 2 } = 2 ^ { - n - d - 3 }
$$

and

$$
\delta _ { \Psi , d } = \frac { \nu _ { \eta , d } } { 4 } = 2 ^ { - n - d - 4 } .
$$

Define

$$
\mathbf { F } _ { \Psi , d , m } ( \mathbf { x } ) = ( f _ { \Psi , d } ( \mathbf { x } ) + \delta _ { \Psi , d } b _ { d } ( \mathbf { x } ) , 0 , \ldots , 0 ) .
$$

When $m = 1$ , the trailing zero coordinates are absent.

Construct $\mathcal { S } _ { \Psi , d , m }$ by placing the networks for $f _ { \Psi , d }$ and $b _ { d }$ in parallel. If the two branches have different depths, identity ReLU neurons propagate the nonnegative output of the shorter branch through the remaining layers. The affine output layer returns

$$
f _ { \Psi , d } + \delta _ { \Psi , d } b _ { d }
$$

in its first coordinate and zero in every remaining coordinate.

The network $\mathcal { S } _ { \Psi , d , m }$ has rational parameters and polynomial size. The numbers

$$
\eta , \quad \nu _ { \eta , d } , \quad { \varepsilon } _ { \Psi , d } , \quad \delta _ { \Psi , d }
$$

have polynomial bit length. Since d and m are fixed, the baseline network has constant size relative to the size of Ψ.

The support of $f _ { \Psi , d }$ is contained in $[ 0 , 1 ] ^ { d }$ , while the support of $b _ { d }$ is contained in $[ 7 / 4 , 5 / 2 ] ^ { \bar { d } }$ . Thus the two supports are disjoint. In particular,

$$
f _ { \Psi , d } = 0
$$

and

$$
b _ { d } = 1
$$

on $Q _ { d } .$ Hence

$$
\mathbf { F } _ { \Psi , d , m } = \delta _ { \Psi , d } \mathbf { e } _ { 1 }
$$

on $Q _ { d } ,$ , where

$$
\mathbf { e } _ { 1 } = ( 1 , 0 , \hdots , 0 ) \in \mathbb { R } ^ { m }
$$

is the first standard basis vector. Since $\delta _ { \Psi , d } ~ > ~ 0 ,$ every constructed target is nonzero. The target is componentwise nonnegative, compactly supported, globally Lipschitz, and continuous piecewise affine.

Suppose that

$$
\Psi \not \in 3 { \cdot } { \mathrm { S A T } } .
$$

Lemma 9 gives

$$
f _ { \Psi , d } = 0 .
$$

Therefore

$$
\mathbf { F } _ { \Psi , d , m } = \delta _ { \Psi , d } b _ { d } \mathbf { e } _ { 1 } .
$$

Using

$$
\| b _ { d } \| _ { L ^ { p } ( \mathbb { R } ^ { d } ) } \leq 1 ,
$$

we obtain

$$
\| \mathbf { F } _ { \Psi , d , m } \| _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } \leq \delta _ { \Psi , d } = \frac { \nu _ { \eta , d } } { 4 } < \frac { \nu _ { \eta , d } } { 2 } = \varepsilon _ { \Psi , d } .
$$

Since the approximation error of the zero network satisfies

$$
\begin{array} { r } { \left. 0 - \mathbf { F } _ { \Psi , d , m } \right. _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } = \left. \mathbf { F } _ { \Psi , d , m } \right. _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } < \varepsilon _ { \Psi , d } , } \end{array}
$$

the zero network is feasible. Because the hidden-neuron count cannot be negative,

$$
\mathrm { O P T } _ { p , d , m } \left( S _ { \Psi , d , m } , \varepsilon _ { \Psi , d } \right) = 0 .
$$

Suppose that

$$
\Psi \in { 3 { \cdot } { \mathrm { S A T } } } .
$$

Let $R _ { \mathbf { a } }$ be the box supplied by Lemma 9. Then

$$
\mu ( R _ { \mathbf { a } } ) = \nu _ { \eta , d }
$$

and

$$
f _ { \Psi , d } = 1
$$

on $R _ { \mathbf { a } }$ . Since

$$
R _ { \mathbf { a } } \subset [ 0 , 1 ] ^ { d }
$$

and

$$
{ \mathrm { s u p p } } b _ { d } \subset \left[ \frac { 7 } { 4 } , \frac { 5 } { 2 } \right] ^ { d } ,
$$

one has

$$
b _ { d } = 0
$$

on $R _ { a }$ . Therefore

$$
\mathbf { F } _ { \Psi , d , m } = \mathbf { e } _ { 1 }
$$

on $R _ { \mathbf { a } }$

It follows that

$$
\begin{array} { r } { \left. \mathbf { F } _ { \Psi , d , m } \right. _ { L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) } ^ { p } \geq \nu _ { \eta , d } . } \end{array}
$$

Since

$$
0 < \nu _ { \eta , d } \leq 1
$$

and

$$
p \geq 1 ,
$$

we have

$$
\Vert \mathbf { F } _ { \Psi , d , m } \Vert _ { L ^ { p } \left( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } \right) } \geq \nu _ { \eta , d } ^ { 1 / p } \geq \nu _ { \eta , d } = 2 \varepsilon _ { \Psi , d } .
$$

Thus the zero network is not feasible and is therefore excluded from the set of feasible approximating networks.

Let Φ be any feasible two-hidden-layer ReLU network. Feasibility gives

$$
\Phi - { \bf F } _ { \Psi , d , m } \in L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) .
$$

Since

$$
\mathbf { F } _ { \Psi , d , m } \in L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) ,
$$

the triangle inequality gives

$$
\Phi \in L ^ { p } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { m } ) .
$$

The function Φ is nonzero because the zero function is not feasible. Lemma 8 shows that Φ has at least $d + 2$ hidden neurons. Therefore

$$
\operatorname { O P T } _ { p , d , m } \left( S _ { \Psi , d , m } , \varepsilon _ { \Psi , d } \right) \geq d + 2 .
$$

The optimum is finite. Indeed, every component of $\mathbf { F } _ { \Psi , d , m }$ belongs to $L ^ { p } ( \mathbb { R } ^ { d } )$ . Corollary 5 gives a two-hiddenlayer ReLU network whose approximation error is less than $\varepsilon _ { \Psi , d } .$

The construction of

$$
\mathcal { S } _ { \Psi , d , m }
$$

and

$$
\varepsilon _ { \Psi , d }
$$

is polynomial in the size of Ψ. We have proved

$$
\Psi \in { 3 \mathrm { - } S A T } \Longleftrightarrow ( S _ { \Psi , d , m } , \varepsilon _ { \Psi , d } ) \in { \mathsf { L A R G E } } _ { p , d , m } .
$$

This is a polynomial-time many-one reduction from 3-SAT. Therefore

$$
\mathsf { L A R G E } _ { p , d , m }
$$

is NP-hard. An exact minimization algorithm would decide this language by comparing the returned optimum with $d +$ 2. □

Theorem 11 (NP-hardness of exact neuron minimization). Fix integers $d \geq 1$ and $m \geq 1 .$ , and fix $1 \leq p <$ ∞. Computing

$$
\mathrm { O P T } _ { p , d , m } ( S , \varepsilon )
$$

exactly is NP-hard.

Proof. Given a 3-CNF formula Ψ, Theorem 10 constructs

$$
\left( S _ { \Psi , d , m } , \varepsilon _ { \Psi , d } \right)
$$

in polynomial time and guarantees

$$
\Psi \in 3 { \ - } { \cal S } \mathrm { { A T } } \Longleftrightarrow \mathrm { O P T } _ { p , d , m } \left( { \cal S } _ { \Psi , d , m } , \varepsilon _ { \Psi , d } \right) \ge d + 2 .
$$

Thus, if the optimum could be computed exactly in polynomial time, then 3-SAT could be decided in polynomial time by checking whether the optimum is at least $d + 2$ . Hence exact two-hidden-layer ReLU neuron minimization under the $L ^ { p }$ approximation constraint is NP-hard. □

## 6 CONCLUSION

This paper establishes that minimizing the total number of hidden neurons in two-hidden-layer ReLU networks under an $L ^ { p }$ approximation constraint is NP-hard for every fixed input dimension $d \ge 1$ , output dimension $m \geq 1 .$ , and $1 \leq p < \infty$ . The proof gives a polynomial-time reduction from 3-SAT using rational ReLU target networks whose realizations are nonzero, componentwise nonnegative, compactly supported, globally Lipschitz, and continuous piecewise affine. For an unsatisfiable formula, the zero network satisfies the approximation constraint and the optimum is zero. For a satisfiable formula, every feasible two-hiddenlayer ReLU network has at least $d + { \dot { 2 } }$ hidden neurons. This gap establishes a worst-case computational barrier to exact architecture selection and motivates the use of heuristic, approximate, or structurally restricted methods for ReLU network design.

## REFERENCES

[1] S. Geman, E. Bienenstock, and R. Doursat, “Neural networks and the bias/variance dilemma,” Neural Comput., vol. 4, no. 1, pp. 1–58, 1992.

[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in Adv. Neural Inf. Process. Syst., vol. 25, 2012. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.

[4] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa, “Natural language processing (almost) from scratch,” J. Mach. Learn. Res., vol. 12, pp. 2493–2537, 2011.

[5] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL–HLT. Minneapolis, Minnesota: Association for Computational Linguistics, 2019, pp. 4171–4186. [Online]. Available: https://aclanthology.org/N19-1423/

[6] J. Han, A. Jentzen, and W. E, “Solving high–dimensional partial differential equations using deep learning,” Proc. Natl. Acad. Sci. USA, vol. 115, no. 34, pp. 8505–8510, 2018. [Online]. Available: https://www.pnas.org/doi/abs/10.1073/pnas.1718942115

[7] M. Raissi, P. Perdikaris, and E. K. G. “Physics–informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” J. Comput. Phys., vol. 378, pp. 686– 707, 2019. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0021999118307125

[8] Y. LeCun, J. Denker, and S. Solla, “Optimal brain damage,” in Adv. Neural Inf. Process. Syst., D. Touretzky, Ed., vol. 2. Morgan–Kaufmann, 1989, pp. 598–605. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf

[9] J. Frankle and M. Carbin, “The lottery ticket hypothesis: Finding sparse, trainable neural networks,” in Proc. 7th Int. Conf. Learn. Represent. OpenReview.net, 2019. [Online]. Available: https://openreview.net/forum?id=rJl-b3RcF7

[10] G. Cybenko, “Approximation by superpositions of a sigmoidal function,” Math. Control Signals Syst., vol. 2, no. 4, pp. 303–314, 1989.

[11] K. Hornik, M. Stinchcombe, and H. White, “Multilayer feedforward networks are universal approximators,” Neural Netw., vol. 2, no. 5, pp. 359–366, 1989.

[12] V. Maiorov and A. Pinkus, “Lower bounds for approximation by mlp neural networks,” Neurocomput., vol. 25, no. 1, pp. 81–91, 1999.

[13] N. J. Guliyev and V. E. Ismailov, “Approximation capability of two hidden layer feedforward neural networks with fixed weights,” Neurocomput., vol. 316, pp. 262–269, 2018.

[14] G. Gripenberg, “Approximation by neural networks with a bounded number of nodes at each level,” J. Approx. Theory, vol. 122, no. 2, pp. 260–266, 2003.

[15] A. Pinkus, “Approximation theory of the mlp model in neural networks,” Acta Numer., vol. 8, pp. 143–195, 1999.

[16] V. E. Ismailov, Ridge Functions and Applications in Neural Networks. American Mathematical Society, 2021, vol. 263.

[17] Z. Lu, H. Pu, F. Wang, Z. Hu, and L. Wang, “The expressive power of neural networks: A view from the width,” Adv. Neural Inf. Process. Syst., vol. 30, 2017. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/2017/file/32cbf687880eb1674a07bf717761dd3a-Paper.pdf

[18] B. Hanin and M. Sellke, “Approximating continuous functions by relu nets of minimal width,” arXiv, Oct. 2017.

[19] P. Kidger and T. Lyons, “Universal approximation with deep narrow networks,” in Proc. 33rd Conf. Learn. Theory, ser. Proceedings of Machine Learning Research, J. Abernethy and S. Agarwal, Eds., vol. 125. PMLR, 09–12 Jul 2020, pp. 2306– 2327. [Online]. Available: https://proceedings.mlr.press/v125/ kidger20a.html

[20] S. Park, C. Yun, J. Lee, and J. Shin, “Minimum width for universal approximation,” in Proc. Int. Conf. Learn. Represent., 2021. [Online]. Available: https://openreview.net/forum?id=O-XJwyoIF-k

[21] Y. Cai, “Achieve the minimum width of neural networks for universal approximation,” in Proc. Int. Conf. Learn. Represent., 2023. [Online]. Available: https://openreview.net/forum?id= hfUJ4ShyDEU

[22] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, “Learning transferable architectures for scalable image recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2018, pp. 8697–8710.

[23] H. Pham, M. Guan, B. Zoph, Q. Le, and J. Dean, “Efficient neural architecture search via parameters sharing,” in Proc. Int. Conf. Mach. Learn., 2018, pp. 4095–4104.

[24] H. Liu, K. Simonyan, and Y. Yang, “Darts: Differentiable architecture search,” in Proc. Int. Conf. Learn. Represent., 2019. [Online]. Available: https://openreview.net/forum?id= S1eYHoC5FX

[25] J. M. Alvarez and M. Salzmann, “Learning the number of neurons in deep networks,” in Adv. Neural Inf. Process. Syst., 2016, pp. 2270– 2278.

[26] E. Lee and C. Y. Lee, “Neuralscale: Efficient scaling of neurons for resource-constrained deep neural networks,” Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 1475–1484, 2020.

[27] K. Kapanova, I. Dimov, and J. Sellier, “A genetic approach to automatic neural network architecture optimization,” Neural Comput. Appl., vol. 29, 2018.

[28] A. L. Blum and R. L. Rivest, “Training a 3-node neural network is np-complete,” Neural Netw., vol. 5, no. 1, pp. 117–127, 1992.

[29] D. Boob, S. S. Dey, and G. Lan, “Complexity of training relu neural network,” Discrete Optim., vol. 44, p. 100620, 2022.

[30] S. Goel, A. Klivans, P. Manurangsi, and D. Reichman, “Tight hardness results for training depth-2 relu networks,” in Proc. Innov. Theor. Comput. Sci., 2021, pp. 22:1–22:14.

[31] V. Froese and C. Hertrich, “Training neural networks is np-hard in fixed dimension,” in Proc. Adv. Neural Inf. Process. Syst., 2023. [Online]. Available: https://openreview.net/forum? id=VAQp2EnZeW

[32] D. Bertschinger, C. Hertrich, P. Jungeblut, T. Miltzow, and S. Weber, “Training fully connected neural networks is ∃Rcomplete,” in Proc. Adv. Neural Inf. Process. Syst., 2023. [Online]. Available: https://openreview.net/forum?id=H15KtcyHvn