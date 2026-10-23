# TRAINING MIXED-MONOTONE NEURAL NETWORKSFOR VERIFICATION WITHOUT SET PROPAGATION

Jonas Fingerhut, Lukas Koller, Matthias Althoff

Technical University of Munich

{jonas.fingerhut, lukas.koller, althoff}@tum.de

## ABSTRACT

Most formal verification approaches for neural networks conservatively enclose the output set of the neural network. However, computing a tight output enclosure is in general computationally hard. We sidestep this challenge with a novel neural network architecture that realizes mixed-monotone functions as the difference of monotone subnetworks. By utilizing the theory on mixed-monotone systems, this reduces the computation of a tight output enclosure from complicated set propagation to two simple forward passes of opposite corners of an input interval. Moreover, because the size of the resulting enclosure is a closed-form function of the weights of the neural network, we can directly minimize it during training. We prove a lower bound on the size of any such enclosure, given by the total variation of the represented function generalized to several inputs, which makes it possible to compare the tightness of the enclosure against the optimum. Our evaluation on regression and dynamical-system benchmarks demonstrates that our training reduces the enclosure size to within 0.5 % of the lower bound without sacrificing prediction accuracy. Computing our enclosure is orders of magnitude faster than CROWN and α-CROWN and on deep networks it is also tighter than zonotope propagation and CROWN, while its size exceeds that of α-CROWN by only 5.1 %.

## 1 INTRODUCTION

Formal verification of neural networks can decide whether the output of a neural network meets a given specification for all inputs bounded by a set. Especially when learned components are deployed in safety-critical control loops, for example in autonomous driving (Ye et al., 2022), formal verification is essential. The exact image of a neural network is in general non-convex, and complete verifiers based on SMT or mixed-integer programming scale poorly in the number of neurons (Katz et al., 2017). Practical tools therefore enclose that image by propagating sets layer by layer, e.g., using intervals (Gowal et al., 2019), zonotopes (Singh et al., 2018; Koller et al., 2026), linear relaxations (Zhang et al., 2018) or polynomial zonotopes (Kochdumper & Althoff, 2021; Kochdumper et al., 2023), adding an abstraction error at every nonlinear layer. That abstraction error accumulates with the depth of the neural network and with the size of the input set. The abstraction error can be reduced with additional computations, e.g., higher-order abstractions (Kochdumper et al., 2023; Ladner & Althoff, 2023) or branch-and-bound methods (Bunel et al., 2020), making it possible to trade off small abstraction errors against high computational cost.

Main idea We overcome this tradeoff by constructing the neural network as a mixed-monotone function, which we realize as the difference of two monotone subnetworks (Fig. 1). This allows us to use the theory on mixed-monotone systems to enclose the image of the neural network for an input interval without any set propagation. We only require two forward passes at opposite corners of the input interval, independent of the depth and width of the neural network and of the input dimension.

Moreover, we use the training of the neural network to explicitly optimize the tightness of the computed enclosure. Our results show that training reduces the enclosure size to within 0.5 % of a lower bound that we prove, without sacrificing prediction accuracy. The computation of our enclosure is (i) orders of magnitude faster than CROWN and α-CROWN and (ii) on deep neural networks it is also tighter than zonotope propagation and CROWN, while its size exceeds that of α-CROWN by only 5.1 %.

![](images/2e8adcd7d4082b9094cdda9f7b06441ba55e2f39cd289d31666c15fb3f19b440.jpg)  
Figure 1: Our neural network architecture is the difference of two monotone subnetworks Φ<sub>+</sub> and Φ<sub>−</sub>. Evaluating both subnetworks at the two opposite corners of the input interval yields an output enclosure without set propagation.

Related work Monotonicity of neural networks can be achieved using sign-constrained weights (Sill, 1997; Daniels & Velikova, 2010), lattice networks (You et al., 2017), or certification of monotonicity after training (Liu et al., 2020; Sivaraman et al., 2020); we build on the signconstrained neural networks of Runje & Shankaranarayana (2023). That literature is motivated by fairness, interpretability, and inductive bias; to the best of our knowledge, the enforced monotonicity has not been exploited for formal verification.

Mixed-monotone approaches combine a decomposition of the plant with a separately computed inclusion function for the network (Jafarpour et al., 2023), or decompose a given neural network (Meyer, 2022). In both cases, the decomposition is obtained after the neural network is fixed, and its tightness is inherited from the used verifier. Classically, finding tight mixed-monotone decompositions is hard and the known constructions require structural assumptions, such as a signstable Jacobian (Yang et al., 2019; Khajenejad & Yong, 2023). In contrast, our neural network has a decomposition function by construction, and we directly optimize the tightness of the result ing enclosure during training, because the size of the enclosure is differentiable with respect to the weights.

Since the tightness of the enclosure is part of our training objective, a related field is certified training (Gowal et al., 2019; Xu et al., 2020; Muller et al., 2023), where abstraction errors are implicitly¨ reduced by optimizing for robustness using sets that are propagated during training. Further, setbased training (Koller et al., 2025; Wendl et al., 2026) is even closer to our work, as it explicitly penalizes the size of the enclosure in the loss, but it still requires a differentiable set propagation in every training step, as required for any other certified training method.

## Contributions Our main contributions are:

• A neural network architecture for mixed-monotone functions, allowing one to compute an output enclosure using two simple forward passes.

• A formula for the smallest enclosure achievable by a difference of two monotone functions: we prove that its size equals the total variation of the represented function generalized to several inputs.

• A neural network training objective that directly minimizes the size of the enclosure.

• An evaluation on static analytic models and dynamical systems, including the enclosure of reachable sets over multiple time steps and comparisons with zonotope propagation, interval bound propagation (IBP), CROWN, and α-CROWN.

## 2 PRELIMINARIES

Notation We denote vectors with lowercase letters and matrices with uppercase letters. The i-th entry of a vector x is written as $x _ { ( i ) }$ and the entry in the i-th row and the $j \cdot$ -th column of a matrix $A \in \mathbb { R } ^ { n \times m }$ is written as $A _ { ( i , j ) }$ . For vectors $x , y \in \mathbb { R } ^ { n }$ of equal dimension, we write $x \leq y$ for the component-wise partial order, i.e., $x _ { ( i ) } \leq y _ { ( i ) }$ for all $i \in \{ 1 , 2 , \ldots , n \}$ . We write the matrix (with appropriate size) that contains only zeros or ones as 0 and 1. Given two matrices $A \in \mathbb { R } ^ { m \times n _ { 1 } }$ and $B \in \mathbb { R } ^ { m \times n _ { 2 } }$ , their (horizontal) concatenation is denoted by $[ A B ] \in \mathbb { R } ^ { m \times ( n _ { 1 } + n _ { 2 } ) }$ . The operation $| A |$ takes the elementwise absolute value of a matrix $A \in \dot { \mathbb { R } } ^ { n \times m }$ ; we use sign(·) likewise. Sets are denoted by uppercase calligraphic letters. The set of all natural numbers up to $\dot { n } \in$ N is written as $[ n ] = \{ 1 , 2 , \ldots , n \}$ . For $\bar { a , b \in \mathbb { R } ^ { m } }$ with $a \leq b ,$ , we write $[ a , b ] = \{ x \in \mathbb { R } ^ { \mathit { \bar { m } } } \mid a \leq x \leq b \}$ for the interval between a and b. For a nonempty bounded set ${ \mathcal { S } } \subseteq { \overline { { \mathbb { R } } } } ^ { m }$ , its interval hull $[ \underline { { s } } , \overline { { s } } ]$ is the smallest interval containing S, where $\underline { { s } } _ { ( i ) } = \operatorname* { i n f } _ { s \in \mathcal { S } } s _ { ( i ) }$ and $\overline { { { s } } } _ { ( i ) } = \operatorname* { s u p } _ { s \in { \mathcal { S } } } { s } _ { ( i ) }$ for all $i \in [ m ]$ . Similar to Althoff (2023, Sec. 3.1), we measure the size of $s$ by the sum of the edge lengths of its interval hull, $\| S \| _ { \mathrm { I } } = \mathbf { 1 } ^ { \top } ( \overline { { s } } - \underline { { s } } )$ , which is referred to as the interval norm.

## 2.1 MONOTONE FUNCTIONS AND MIXED-MONOTONE DECOMPOSITIONS

We utilize the theory of mixed monotonicity to construct neural networks that admit easy-tocompute and tight output enclosures. A monotone function returns its extremal values at two opposite corners of an input interval.

Definition 1 (Monotone function, (Coogan & Arcak, 2015, Def. 1)). Given an interval $\mathcal { X } _ { 0 } \subseteq \mathbb { R } ^ { n }$ , a function $f \colon \mathcal { X } _ { 0 }  \mathbb { R } ^ { m } \ i s$

(i) non-decreasing on $\begin{array} { r } { \mathcal { X } _ { 0 } \mathrm { ~ } i f \boldsymbol { x } \le x ^ { \prime } \implies f ( \boldsymbol { x } ) \le f ( x ^ { \prime } ) , f o r \mathit { a l l } \boldsymbol { x } , x ^ { \prime } \in \mathcal { X } _ { 0 } } \end{array}$ , and

(ii) non-increasing on $\begin{array} { r } { \mathcal { X } _ { 0 } \mathrm { ~ } i f \boldsymbol { x } \le x ^ { \prime } \implies f ( \boldsymbol { x } ) \ge f ( x ^ { \prime } ) , f o r a l l \boldsymbol { x } , x ^ { \prime } \in \mathcal { X } _ { 0 } . } \end{array}$

Throughout this work, a monotone (or monotonic) function or neural network refers to a nondecreasing one.

Mixed monotonicity generalizes monotonicity by allowing a function to be non-decreasing in some variables and non-increasing in others.

Proposition 1 (Enclosing a mixed-monotone function, (Coogan & Arcak, 2015, Def. 2)). For $\mathcal { X } _ { 0 } \subseteq$ $\mathbb { R } ^ { n }$ , let $f \colon \mathcal { X } _ { 0 }  \mathbb { R } ^ { m }$ . A function $d _ { f } \colon \mathcal { X } _ { 0 } \times \mathcal { X } _ { 0 } \to \mathbb { R } ^ { m }$ is a decomposition function for f iff for all $x _ { 1 } , x _ { 2 } \in \mathcal { X } _ { 0 }$

$( i ) d _ { f } ( x _ { 1 } , x _ { 1 } ) = f ( x _ { 1 } ) , ( i i ) d _ { f } ( \cdot , x _ { 2 } )$ is non-decreasing, (iii) $d _ { f } ( x _ { 1 } , \cdot )$ is non-increasing.

The function $f$ is called mixed-monotone $i f a$ decomposition function for $f$ exists. The image of f over any interval $\mathcal { X } = [ \underline { { x } } , \overline { { x } } ] \subseteq \mathcal { X } _ { 0 }$ can be enclosed by

$$
f ( \mathcal { X } ) \subseteq [ d _ { f } ( \underline { { x } } , \overline { { x } } ) , d _ { f } ( \overline { { x } } , \underline { { x } } ) ] .
$$

Decomposition functions are not unique (Coogan, 2020); hence, finding a tight decomposition is a central challenge. In this work, we build neural networks to be mixed-monotone functions and we directly optimize the tightness of output enclosures through neural network training.

A standard decomposition function is the difference of two monotone functions (Coogan & Arcak, 2015, Ex. 1), i.e., given the monotone functions $g$ and $h , f ( x ) = g ( x ) - h ( x )$ , is a mixed-monotone function with decomposition function

$$
d _ { f } ( x _ { 1 } , x _ { 2 } ) = g ( x _ { 1 } ) - h ( x _ { 2 } ) .\tag{1}
$$

We build our neural network architecture based on the decomposition in (1). Future work can explore more general forms of decomposition functions.

## 2.2 MONOTONIC NEURAL NETWORKS

Feed-forward networks $\Phi \colon \mathbb { R } ^ { n _ { 0 } }  \mathbb { R } ^ { n _ { \kappa } }$ are sequences of $\kappa \in \mathbb N$ layers. Each layer is either affine (linear layer) or an element-wise nonlinear activation function (nonlinear layer). In this work, we consider sign-constrained neural networks that can only represent monotonic functions.

Definition 2 (Sign-Constrained Neural Network, (Runje & Shankaranarayana, 2023, Def. 1)). For an input $x \in \mathbb { R } ^ { n _ { 0 } }$ , the output ofa sign-constrained neural network $y = \Phi ( x ) \in \mathbb { R } ^ { n _ { \kappa } } i s$

$$
h _ { 0 } = x , ~ h _ { k } = L _ { k } ( h _ { k - 1 } ) ~ f o r ~ k \in [ \kappa ] , ~ y = h _ { \kappa } ,
$$

where

$$
L _ { k } ( h _ { k - 1 } ) = \left\{ \begin{array} { l l } { | W _ { k } | h _ { k - 1 } + b _ { k } } & { i f t h e k \mathit { t h l a y e r i s l i n e a r } , } \\ { \phi _ { k } ( h _ { k - 1 } ) } & { o t h e r w i s e , } \end{array} \right.
$$

with weights $W _ { k } \in \mathbb { R } ^ { n _ { k } \times n _ { k - 1 } }$ , bias $b _ { k } \in \mathbb { R } ^ { n _ { k } }$ , and the (element-wise) nonlinear function $\phi _ { k }$

Any sign-constrained neural network represents a monotonic function (Runje & Shankaranarayana, 2023, Lem. 1). For the gradient computation during training, the weights remain unconstrained (Sec. A.4). The activation functions can be used to determine whether the function represented by a monotonic neural network is convex or concave (Runje & Shankaranarayana, 2023, Cor. 3).

## 2.3 PROBLEM STATEMENT

Given a neural network $\Phi \colon \mathbb { R } ^ { n _ { 0 } }  \mathbb { R } ^ { n _ { \kappa } }$ , an input set $\mathcal { X } \subset \mathbb { R } ^ { n _ { 0 } }$ , and an unsafe set $\mathcal { U } \subset \mathbb { R } ^ { n _ { \kappa } }$ , the goal of neural network verification is to prove that all inputs are safe, i.e., $\forall x \in { \mathcal { X } } \colon \Phi ( x ) \notin { \mathcal { U } }$ To this end, we compute an enclosure ${ \mathcal { V } } \supset \{ \Phi ( x ) \mid x \in { \dot { x } } \}$ of the output set and check whether $y \cap \mathcal { U } = \emptyset$ . We want to build a neural network architecture that allows an efficient computation of tight output enclosures to make neural network verification practical and scalable.

## 3 A MONOTONE-DIFFERENCE ARCHITECTURE

We propose a neural network architecture that realizes mixed-monotone functions using the difference of two monotone neural subnetworks: Let $\Phi _ { + } , \Phi _ { - } \colon \mathbb { R } ^ { n _ { 0 } } \ \to \ \mathbb { R } ^ { n _ { \kappa } }$ be monotonic neural networks, we construct a neural network Φ as

$$
\Phi = \Phi _ { + } - \Phi _ { - } .\tag{2}
$$

We use sign-constraints to realize the monotone neural subnetworks $\Phi _ { + } , \Phi _ { - } \ ( \mathrm { D e f . } \ 2 )$ . The neural network Φ is mixed-monotone by construction and we can use the decomposition function to enclose the image of the neural network.

Proposition 2. Given an input set $\mathcal { X } = [ \underline { { x } } , \overline { { x } } ] \subseteq \mathbb { R } ^ { n _ { 0 } }$ , a neural network $\Phi = \Phi _ { + } - \Phi _ { - } ,$ , which is the difference of two monotonic neural networks $\Phi _ { + } , \Phi _ { - }$ , we can enclose the output set of the neural network by

$$
{ \mathsf { e n c l o s e } } ( \Phi , \mathcal { X } ) = [ d _ { \Phi } ( \underline { { { x } } } , \overline { { { x } } } ) , d _ { \Phi } ( \overline { { { x } } } , \underline { { { x } } } ) ] \supseteq \Phi ( \mathcal { X } ) ,
$$

where

$$
d _ { \Phi } ( x _ { 1 } , x _ { 2 } ) = \Phi _ { + } ( x _ { 1 } ) - \Phi _ { - } ( x _ { 2 } ) .
$$

Proof. The neural network Φ is mixed monotone (Coogan & Arcak, 2015, Ex. 1), hence we can enclose the output using the decomposition function (Prop. 1). □

The computation of the enclosure only requires two forward passes through the neural network via the decomposition function, instead of requiring a complicated set propagation.

Computational Complexity Prop. 2 computes an enclosure with two forward passes, so that computing it is as expensive as a single point evaluation up to a constant factor, i.e., it takes $\mathcal { O } ( n _ { \mathrm { m a x } } ^ { 2 } \kappa )$ time, where $n _ { \mathrm { m a x } } = \mathrm { m a x } _ { k \in \{ 0 , \ldots , \kappa \} }$ n denotes the largest number of neurons of a layer, including the input layer. In contrast, propagating a zonotope through the same neural network takes $\bar { \mathcal { O } } \bar { ( n _ { \mathrm { m a x } } ^ { 2 } q \kappa ) }$ time, where the number of generators grows with the size of the neural network, i.e., $\begin{array} { r } { q \in \mathcal { O } ( n _ { 0 } + \sum _ { k \in [ \kappa ] } n _ { k } ) } \end{array}$ (Koller et al., 2025, Prop. 16). Linear relaxations, e.g., CROWN (Zhang et al., 2018) or DeepPoly (Singh et al., 2019), take $\mathcal { O } ( n _ { \operatorname* { m a x } } ^ { 3 } \kappa ^ { 2 } )$ time (Zhang et al., 2018, Sec. 3.1), as the bounds of each neuron are computed by a backward pass through all previous layers in $\mathcal { O } ( n _ { \mathrm { m a x } } ^ { 2 } \kappa )$ time (Singh et al., 2019, Sec. 2). Compared to both, our enclosure saves a factor that grows linearly with the number of neurons of the neural network.

![](images/4e8393b936e0f88dc3a4d83ab7e80ac97386b3c07bd6a09f00a7df9337d00217.jpg)  
Figure 2: The size of an enclosure depends on the decomposition: For the same function $f ,$ two decompositions yield enclosures of different sizes.

## 4 VERIFICATION-AWARE TRAINING

The decomposition of a mixed-monotone function is not unique, and different decompositions of the same function yield enclosures of different size. Fig. 2 compares two decompositions, which yield enclosures of different sizes. First, we use an example to demonstrate the non-uniqueness of a decomposition and its effects on the size of the enclosure, before we explain how we use neural network training to optimize the tightness of an enclosure.

Example 1 (Comparing Decompositions). Let $\mathcal { X } \ : = \ : [ 0 , 2 ]$ be the input interval and let $f ( x ) =$ $1 - | x - 1 |$ be a function (see $F i g . 2 ( a ) )$ . The exact image off (which generally is not computable) has size $l , i . e . , \| f ( \mathcal { X } ) \| _ { \mathrm { I } } = \| [ 0 , 1 ] \| _ { \mathrm { I } } = 1$

The function f can be represented by the mixed-monotone neural network $\Phi = \Phi _ { + } - \Phi _ { - }$ , with monotone subnetworks (see Fig. 2(b))

$$
\Phi _ { + } ( x ) = x ,
$$

$$
\Phi _ { - } ( x ) = 2 \operatorname* { m a x } ( x - 1 , 0 ) .\tag{3}
$$

Using Prop. 1, we can enclose the output set $o f \Phi$ by

$$
\mathsf { e n c l o s e } ( \Phi , \mathcal { X } ) = [ \Phi _ { + } ( 0 ) - \Phi _ { - } ( 2 ) , \Phi _ { + } ( 2 ) - \Phi _ { - } ( 0 ) ] = [ - 2 , 2 ] .
$$

The size ofthe enclosure is ∥enclose $( \Phi , { \mathcal { X } } ) \Vert _ { \mathrm { I } } = 4$

In contrast, another mixed-monotone neural network $\Phi ^ { \prime } = \Phi _ { + } ^ { \prime } - \Phi _ { - } ^ { \prime }$ can also represent the function $f$ with monotone subnetworks (see Fig. 2(c))

$$
\Phi _ { + } ^ { \prime } ( x ) = \operatorname* { m i n } ( x , 1 ) ,
$$

$$
\Phi _ { - } ^ { \prime } ( x ) = \operatorname* { m a x } ( x - 1 , 0 ) .\tag{4}
$$

The computed enclosure is smaller:

$$
{ \mathsf { e n c l o s e } } ( \Phi ^ { \prime } , \mathcal { X } ) = \left[ \Phi _ { + } ^ { \prime } ( 0 ) - \Phi _ { - } ^ { \prime } ( 2 ) , \Phi _ { + } ^ { \prime } ( 2 ) - \Phi _ { - } ^ { \prime } ( 0 ) \right] = [ - 1 , 1 ] ,
$$

which has size ∥enclose $( \Phi ^ { \prime } , \mathcal { X } ) \Vert _ { \mathrm { I } } = 2 .$

Training Loss Our goal is to directly optimize the tightness of the enclosure during training. To this end, we combine a standard regression loss (i.e., mean-squared error) with a penalty of the enclosure size for a fixed input perturbation radius $\epsilon \in \mathbb { R } ^ { n _ { 0 } }$ . For an input $x _ { j } ~ \in ~ \mathbb { R } ^ { n _ { 0 } }$ with target $t _ { j } \in \mathbb { R } ^ { n _ { \kappa } }$ , we compute the training loss

$$
L = \underbrace { | | \Phi ( x _ { j } ) - t _ { j } | | _ { 2 } ^ { 2 } } _ { \mathrm { R e g r e s s i o n L o s s } } + \beta \underbrace { | | { \bf e n c l o s e } ( \Phi , [ x _ { j } - \epsilon , x _ { j } + \epsilon ] ) | | _ { \mathrm { I } } } _ { \mathrm { E n c l o s u r e ~ S i z e } } .\tag{5}
$$

In principle, the training loss is similar to set-based training (Koller et al., 2025; Wendl et al., 2026): The training objective penalizes the size of an output enclosure. However, we do not require set propagations during training, which makes the training significantly more efficient.

Evaluating the training loss (5) only requires three forward passes, i.e., one for the regression loss and two for the enclosure (Prop. 2); hence, a training iteration takes $\mathcal { O } ( n _ { \mathrm { m a x } } ^ { 2 } \kappa )$ time, which is as expensive as standard point-based training up to a constant factor. In contrast, a set-based training iteration takes $\mathcal { O } ( n _ { \mathrm { m a x } } ^ { 2 } q \kappa )$ time (Koller et al., 2025, Sec. 4.4). Hence, our training saves the same factor q as our enclosure computation.

Subsequently, we derive a lower bound of the enclosure size, which motivates the training objective and serves as a reference to measure the tightness of computed enclosures.

Enclosure Size The size of an enclosure cannot be smaller than the total variation of the output of the neural network along chains in the input interval, which generalizes the total variation of a scalar function (Cohn, 1993, Sec. 4.4) to several inputs (Clarkson & Adams, 1933, Sec. 2, Def. A). A chain in an interval $\mathcal { X } \ = \ [ \underline { { x } } , \overline { { x } } ] \ \subseteq \ \mathbb { R } ^ { n _ { 0 } }$ is a tuple $( x _ { 0 } , \ldots , x _ { p } )$ of finitely many inputs with $\underline { { x } } \leq x _ { 0 } \leq x _ { 1 } \leq \cdot \cdot \cdot \leq x _ { p } \leq \overline { { x } }$ with respect to the component-wise partial order (Sec. 2), where the length $p \in$ N is arbitrary. The total variation of the output of a neural network Φ along chains in X is the vector $\nu _ { \Phi } ( \mathcal { X } ) \in \dot { \mathbb { R } } ^ { n _ { \kappa } }$ , whose i-th entry, for $i \in [ \bar { n } _ { \kappa } ]$ , is the largest summed change of the i-th output along a chain:

$$
\nu _ { \Phi } ( \mathcal { X } ) _ { ( i ) } = \operatorname* { s u p } \bigg \{ \sum _ { \ell = 1 } ^ { p } \bigl | \Phi ( x _ { \ell } ) _ { ( i ) } - \Phi ( x _ { \ell - 1 } ) _ { ( i ) } \bigr | \bigg | p \in \mathbb { N } , \underline { { x } } \le x _ { 0 } \le x _ { 1 } \le \cdots \le x _ { p } \le \overline { { x } } \bigg \} .\tag{6}
$$

Between two consecutive inputs of a chain, the output of Φ changes by at most the sum of the increases of $\Phi _ { + }$ and $\Phi _ { - } .$ , as neither subnetwork can decrease.

Proposition 3 (Smallest enclosure). Given an input interval $\mathcal { X } \ = \ [ \underline { { x } } , \overline { { x } } ] \ \subseteq \ \mathbb { R } ^ { n _ { 0 } }$ and a neural network $\Phi = \Phi _ { + } - \Phi _ { - }$ , where $\Phi _ { + } , \Phi _ { - }$ are non-decreasing on X $( D e f . \ I ) ,$ , it holds that

$$
\mathbf { 1 } ^ { \top } \nu _ { \Phi } ( { \boldsymbol { \chi } } ) \leq \| \mathbf { e n c l o s e } ( \Phi , { \boldsymbol { \chi } } ) \| _ { \mathrm { I } } .
$$

Moreover, there exist non-decreasingfunctions $\Phi _ { + } ^ { \prime } , \Phi _ { - } ^ { \prime } \colon \mathcal { X } \to \mathbb { R } ^ { n _ { \kappa } }$ with $\Phi _ { + } ^ { \prime } - \Phi _ { - } ^ { \prime } = \Phi$ on X, such that equality holdsfor $\Phi ^ { \prime } = \Phi _ { + } ^ { \prime } - \bar { \Phi } _ { - } ^ { \prime }$

Proof. See Sec. A.1.

The lower bound $\mathbf { 1 } ^ { \top } \nu _ { \Phi } ( \mathcal { X } )$ only depends on the function represented by the neural network and not on its decomposition.

## 5 EVALUATION

We have implemented our mixed-monotone neural network architecture in MATLAB using the CORA toolbox (Althoff, 2015). Our evaluation is structured as follows: We first assess prediction accuracy of mixed-monotone neural networks, next compare enclosure tightness and computational cost, and finally showcase the effectiveness of verification-aware training. Finally, we demonstrate the superior scalability of our approach with respect to the input dimension and number of neural network layers and additionally evaluate the enclosure tightness over multiple time steps of a dynamical system.

Benchmarks We evaluate our approach on two static analytic models, i.e., a scalar analytic model and a multi-input analytic model with a variable number of inputs, and on two dynamical systems (Pendulum and Tanks). The scalar analytic model has a closed-form total variation, which serves as an exact reference for the lower bound of Prop. 3. The two dynamical systems cover a cooperative system (Tanks), i.e., a system whose solutions preserve the partial order of the initial states for a constant input (Smith, 1995), and a non-cooperative system (Pendulum). In the multi-input analytic model, only two inputs enter nonlinearly, so that we can increase the input dimension without changing the nonlinearity of the model.

Metrics We report prediction accuracy using $R ^ { 2 }$ -scores. To measure the tightness of a computed enclosure, we report the ratio of its size to the size of the exact image, i.e., for an input set X and enclosure $\mathcal { V } \supseteq \Phi ( \mathcal { X } )$ , we report $\| { \mathcal { V } } \| _ { \mathrm { I } } / \| \Phi ( { \mathcal { X } } ) \| .$ <sub>I</sub>, which we refer to as the tightness of the enclosure. Since computing the exact image is generally intractable, we underapproximate its size using samples. Since $\hat { \| \Phi ( \mathcal { X } ) \| _ { \mathrm { I } } } \le \mathbf { 1 } ^ { \top } \nu _ { \Phi } ( \bar { \mathcal { X } } )$ by Props. 1 and 3, the tightness is also an upper bound of the ratio $\| \mathcal { V } \| _ { \mathrm { I } } / ( 1 ^ { \top } \nu _ { \Phi } \dot { ( } \mathcal { X } ) )$ of the enclosure size to the lower bound.

Table 1: Comparing prediction accuracy across different benchmarks shown as mean ± standard deviation over ten seeds.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">#Layers κ</td><td colspan="2">Accuracy ↑</td></tr><tr><td>Baseline</td><td>Ours</td></tr><tr><td rowspan="3">Scalar analytic</td><td>2</td><td> $0 . 9 8 3 7 \pm 0 . 0 0 1 8$ </td><td> $\mathbf { 0 . 9 8 5 8 \pm 0 . 0 0 0 9 }$ </td></tr><tr><td>4</td><td> $\mathbf { 0 . 9 9 9 8 \pm 0 . 0 0 0 1 }$ </td><td> $0 . 9 8 9 5 \pm 0 . 0 0 5 1$ </td></tr><tr><td>8</td><td> $\mathbf { 0 . 9 9 9 7 } \pm 0 . 0 0 0 3$ </td><td> $0 . 9 9 6 4 \pm 0 . 0 0 3 5$ </td></tr><tr><td rowspan="2">Pendulum</td><td>3</td><td> $\mathbf { 0 . 9 9 4 2 \pm 0 . 0 0 2 4 }$ </td><td> $0 . 9 8 7 8 \pm 0 . 0 0 7 1$ </td></tr><tr><td>8</td><td> $\mathbf { 0 . 9 9 3 1 } \pm 0 . 0 0 4 9$ </td><td> $0 . 9 8 6 8 \pm 0 . 0 0 9 8$ </td></tr><tr><td rowspan="2">Tanks</td><td>3</td><td> $\mathbf { 0 . 9 9 9 7 \pm 0 . 0 0 0 1 }$ </td><td> $0 . 9 9 9 3 \pm 0 . 0 0 0 4$ </td></tr><tr><td>8</td><td> ${ \bf 0 . 9 9 8 5 \pm 0 . 0 0 1 3 }$ </td><td> $0 . 9 9 8 4 \pm 0 . 0 0 1 2$ </td></tr></table>

## 5.1 PREDICTION ACCURACY

Our sign-constrained architecture maintains high prediction accuracy across the tested benchmarks and the number of neural network layers, showing that the monotonicity constraints do not prevent effective training. Tab. 1 compares our architecture with an unconstrained baseline under the same training protocol, with differences in mean $R ^ { 2 }$ of at most 0.011 across the reported configurations. Additional results for the scalar analytic model are provided in Tab. 7.

## 5.2 VERIFICATION TIGHTNESS AND RUNTIMES

In Tab. 2, we compare the tightness and runtimes of our enclosure (Ours) with interval bound propagation (IBP; Gowal et al., 2019), zonotope propagation in CORA (Althoff, 2015), CROWN (Zhang et al., 2018), and α-CROWN (Xu et al., 2021) for neural networks trained on the scalar analytic model; see Sec. A.6 for details and additional results. Since the methods are implemented in different toolboxes, auto LiRPA (PyTorch) and CORA (MATLAB), runtimes are reported per toolbox and relative to our enclosure within that toolbox to provide a fair comparison. At four layers, the enclosures of all methods exceed the size of the sampled image by at most 17 %, and our enclosure is as tight as zonotope propagation. At eight layers, our enclosure is tighter than zonotope propagation and CROWN, and its size exceeds that of α-CROWN by only 5.1 %, i.e., the tightness of our enclosure improves with depth while that of the set-propagation methods degrades. Computing our enclosure is 32× faster than zonotope propagation, 256× faster than CROWN, and 20 600× faster than α-CROWN, which additionally optimizes its bounds. Due to the monotone subnetworks, the IBP bounds coincide with ours, but our bounds are computed using two simple forward passes without any set propagation, which makes it 10× to 11× faster than IBP in auto LiRPA.

## 5.3 VERIFICATION-AWARE TRAINING AND THE LOWER BOUND

To evaluate the effectiveness of our verification-aware training, we vary the parameter $\beta$ to weight the width of the output set in the training loss (see (5)) and compare the accuracy and enclosure tightness. Tab. 3 shows the results. Using $\beta ,$ we can tradeoff accuracy against enclosure tightness: A small value increases accuracy, while a larger value favors enclosure tightness.

For $\beta \ge 0 . 0 5$ , the enclosure coincides with the sampled image on both benchmarks, i.e., the decomposition is already tight; a larger $\beta$ can then only reduce the enclosure size by reducing the variation of the function represented by the neural network, which decreases its accuracy.

Table 2: Comparing enclosure tightness and runtimes on the scalar analytic model shown as mean ± standard deviation over five seeds. The runtime is the time to enclose the output set for one input interval, reported as the minimum over repeated measurements (Sec. A.6). We evaluate our approach within CORA as well as within auto LiRPA.
<table><tr><td rowspan="2">#Layers κ</td><td rowspan="2">Framework</td><td rowspan="2">Method</td><td rowspan="2">Tightness  $\downarrow$ </td><td colspan="2">Runtime↓</td></tr><tr><td>[ms]</td><td>[relative to ours]</td></tr><tr><td rowspan="6">4</td><td rowspan="5">auto_LiRPA</td><td>Ours (exported)</td><td> $1 . 1 6 6 2 \pm 0 . 1 1 6 7$ </td><td>0.12</td><td>1×</td></tr><tr><td>IBP</td><td> $1 . 1 6 6 2 \pm 0 . 1 1 6 7$ </td><td>1.21</td><td>9.9×</td></tr><tr><td>CROWN</td><td> $1 . 1 2 6 4 \pm 0 . 0 3 7 1$ </td><td>13.7</td><td>112×</td></tr><tr><td>α-CROWN</td><td> $\mathbf { 1 . 0 0 2 4 } \pm 0 . 0 0 2 5$ </td><td>1070</td><td>8750×</td></tr><tr><td>Ours</td><td> $1 . 1 6 6 2 \pm 0 . 1 1 6 7$ </td><td>0.01</td><td>1×</td></tr><tr><td rowspan="3">CORA</td><td>Zonotope</td><td> $1 . 1 5 9 2 \pm 0 . 0 4 8 3$ </td><td>0.17</td><td>11.7×</td></tr><tr><td>Ours (exported)</td><td> $1 . 0 5 3 4 \pm 0 . 0 8 0 5$ </td><td>0.21</td><td>1×</td></tr><tr><td>IBP auto_LiRPA</td><td> $1 . 0 5 3 4 \pm 0 . 0 8 0 5$ </td><td>2.35</td><td>11.2×</td></tr><tr><td rowspan="3">8 CORA</td><td>CROWN α-CROWN</td><td> $1 . 2 6 4 0 \pm 0 . 0 6 0 8$ </td><td>53.8 4330</td><td>256×</td></tr><tr><td></td><td> $\mathbf { 1 . 0 0 2 2 \pm 0 . 0 0 1 2 }$ </td><td></td><td>20600×</td></tr><tr><td>Ours Zonotope</td><td> $1 . 0 5 3 4 \pm 0 . 0 8 0 5$   $1 . 4 4 0 9 \pm 0 . 0 6 5 5$ </td><td>0.02 0.69</td><td>1× 32×</td></tr></table>

Table 3: Comparing the accuracy and enclosure tightness for varying $\beta$ in the loss function (5) shown as mean ± standard deviation over three seeds.
<table><tr><td>Benchmark</td><td> $\beta$ </td><td>Accuracy ↑</td><td>Tightness↓</td></tr><tr><td rowspan="5">Scalar analytic</td><td>0</td><td> $0 . 9 8 4 6 \pm 0 . 0 0 0 7$ </td><td> $1 . 8 2 1 \pm 0 . 1 3 6$ </td></tr><tr><td>0.01</td><td> $0 . 9 8 5 7 \pm 0 . 0 0 3 4$ </td><td> $1 . 0 2 4 \pm 0 . 0 0 9$ </td></tr><tr><td>0.05</td><td> $\mathbf { 0 . 9 8 8 3 \pm 0 . 0 0 1 2 }$ </td><td> $1 . 0 0 4 \pm 0 . 0 0 3$ </td></tr><tr><td>0.2</td><td> $0 . 9 8 0 3 \pm 0 . 0 0 1 2$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td>1.0</td><td> $0 . 9 3 9 2 \pm 0 . 0 1 2 1$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan="5">Tanks</td><td>0</td><td> $\mathbf { 0 . 9 9 9 9 } \pm 0 . 0 0 0 0$ </td><td> $1 . 8 8 3 \pm 0 . 1 3 4$ </td></tr><tr><td>0.01</td><td> $0 . 9 9 9 8 \pm 0 . 0 0 0 1$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td>0.05</td><td> $0 . 9 9 8 5 \pm 0 . 0 0 0 6$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td>0.2</td><td> $0 . 9 8 5 2 \pm 0 . 0 0 3 5$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td>1.0</td><td> $0 . 8 6 4 6 \pm 0 . 0 1 2 1$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr></table>

Table 4: Comparing the enclosure tightness for different numbers of input dimensions shown as mean ± standard deviation over five seeds.
<table><tr><td rowspan="2">#Input Dims</td><td colspan="2">Tightness↓</td></tr><tr><td>Zonotope</td><td>Ours</td></tr><tr><td>2</td><td> $1 . 6 5 \pm 0 . 0 1$ </td><td> $\mathbf { 1 . 0 4 5 \pm 0 . 0 0 8 }$ </td></tr><tr><td>20</td><td> $1 . 7 7 \pm 0 . 0 9$ </td><td> $\mathbf { 1 . 0 7 3 \pm 0 . 0 0 9 }$ </td></tr><tr><td>100</td><td> $3 6 0 \pm 3 1$ </td><td> ${ \bf 1 . 1 6 7 \pm 0 . 0 3 9 }$ </td></tr></table>

Table 5: Comparing the enclosure tightness for different numbers of neural network layers shown as mean ± standard deviation over ten seeds. The crossover radius is reported as median (range) over seeds; “–” indicates no crossover.
<table><tr><td rowspan="2">#Layers κ</td><td colspan="2">Tightness ↓</td><td rowspan="2">Crossover Radius ↓ [median (range)]</td></tr><tr><td>Zonotope</td><td>Ours</td></tr><tr><td>2</td><td> ${ \bf 1 . 0 8 \pm 0 . 0 1 }$ </td><td> $5 . 9 2 \pm 0 . 7 9$ </td><td></td></tr><tr><td>3</td><td> ${ \bf 1 . 1 1 \pm 0 . 0 2 }$ </td><td> $2 . 4 7 \pm 0 . 7 2$ </td><td>1.00 (0.1–1.5)</td></tr><tr><td>4</td><td> ${ \bf 1 . 1 4 \pm 0 . 0 3 }$ </td><td> $1 . 5 6 \pm 0 . 3 7$ </td><td>0.20 (0.03–0.5)</td></tr><tr><td>6</td><td> $1 . 2 4 \pm 0 . 1 2$ </td><td> ${ \bf 1 . 1 3 \pm 0 . 1 2 }$ </td><td>0.025 (0.01–0.15)</td></tr><tr><td>8</td><td> $1 . 3 8 \pm 0 . 2 5$ </td><td> ${ \bf 1 . 1 9 \pm 0 . 1 8 }$ </td><td>0.04 (0.01–0.075)</td></tr></table>

## 5.4 SCALABILITY

Due to mixed-monotonicity of the neural network, our enclosure computation does not require to explicitly consider approximation errors. Set propagation approaches suffer from the wrapping ef fect of approximation errors across layers, which prevents the scalability of these approaches. We demonstrate the superior scalability of our approach across different dimensions: (i) the number of input dimensions, (ii) the number of neural network layers, and (iii) the number of time steps over which the neural network is applied iteratively.

Number of Input Dimensions We significantly increase the number of input dimensions of the multi-input analytic model and compare the enclosure tightness of our approach against the baseline using zonotope propagation. Tab. 4 shows the results: While the tightness of the baseline explodes for n = 100, our approach remains significantly tighter. Further results are provided in Tab. 9.

Number of Layers We compare in Tab. 5 the tightness of our approach against the baseline using zonotope propagation for different numbers of neural network layers at a fixed perturbation radius ϵ = 0.05. Additionally, we report the smallest input set radius (crossover radius) at which our enclosure is smaller than the corresponding zonotope enclosure. When increasing the number of layers, the crossover radius becomes smaller. Additional results are in Sec. A.7.

Multiple time steps Finally, we apply the neural network iteratively: we train a mixed-monotone neural network to predict the state of the Tanks system after one time step for a constant input and enclose the reachable set over several time steps by applying Prop. 2 repeatedly, where the enclosure of one time step is the input interval of the next (Sec. A.9). We compare against zonotope propagation through the unconstrained baseline trained on the same data. Fig. 3 shows the size of the enclosures over time: our enclosure stays close to the sampled reachable set, while the zonotope enclosure of the baseline diverges for κ = 8 layers.

## 6 CONCLUSION

We introduce a neural network architecture that realizes mixed-monotone functions as the difference of two monotone subnetworks. This architecture replaces set propagation by two forward passes of opposite corners of the input interval, which makes computing an output enclosure as cheap as a point evaluation. Because the size of this enclosure is a closed-form function of the weights, we can directly minimize it during training, which turns the classical problem of finding a tight mixedmonotone decomposition into a learning problem; the total variation of the represented function along chains provides a lower bound with which a learned decomposition can be compared. Our evaluation shows that (i) training reduces the enclosure size to within 0.5 % of this lower bound without sacrificing prediction accuracy, (ii) computing our enclosure is orders of magnitude faster than CROWN and α-CROWN, and (iii) on deep neural networks the enclosure is tighter than zonotope propagation and CROWN, while its size exceeds that of α-CROWN by only 5.1 %. The advantage grows where verification is hardest: with the depth of the neural network, with the number of input dimensions, and when the neural network is applied iteratively over multiple time steps.

![](images/61b609e566312fafc0776e09f2e55a982214d1efdd13e384c1b993a577cd7517.jpg)  
Figure 3: Size of the enclosure of the reachable set over time for the Tanks system, when the neural network is applied iteratively.

Limitations Our enclosure bounds the output of the neural network, not the output of the system that the neural network approximates. Due to our sign-constrained architecture, interval bound propagation returns the same enclosure; our advantages are that two forward passes suffice, which is 10× to 11× faster in our experiments, and that the size of the enclosure is directly minimized during training. We restrict the decomposition function to the difference of two monotone functions (1), to which the lower bound of Prop. 3 applies; more general decomposition functions can yield tighter enclosures and are future work. Finally, we only consider regression tasks, and our evaluation over multiple time steps is restricted to a cooperative system with a constant input.

## 7 AI USE STATEMENT

In this work, we used generative AI tools for drafting sections, identifying and summarizing relevant literature, assisting during the implementation and in the writing of proofs, editing to improve readability of the paper, and creating or modifying TikZ figures or images. We have reviewed all AI-assisted work to ensure its correctness. We have carefully checked the correctness of our AIassisted implementation and our reported results are reproducible using the included supplementary material. We have checked every claim and reference against its source.

## 8 REPRODUCIBILITY STATEMENT

All benchmark definitions and evaluation details are included in Sec. A.2 to A.4. Additionally, we include a reproducibility package as supplementary material with this submission, which contains all source code and allows the reader to reproduce all results contained in the paper.

## REFERENCES

Matthias Althoff. An introduction to CORA 2015. In Proc. of the Workshop on Applied Verification of Continuous and Hybrid Systems (ARCH), volume 34 of EPiC Series in Computing, pp. 120– 151, 2015.

Matthias Althoff. Checking and establishing reachset conformance in CORA 2023. In Proc. of the Workshop on Applied Verification of Continuous and Hybrid Systems (ARCH), volume 96 of EPiC Series in Computing, pp. 9–33, 2023.

Rudy Bunel, Ilker Turkaslan, Philip H. S. Torr, M. Pawan Kumar, Jingyue Lu, and Pushmeet Kohli. Branch and bound for piecewise linear neural network verification. Journal of Machine Learning Research, 21(42):1–39, 2020.

James A. Clarkson and Clifford R. Adams. On definitions of bounded variation for functions of two variables. Transactions ofthe American Mathematical Society, 35:824–854, 1933.

Donald L. Cohn. Measure Theory. Birkhauser, Boston, 1993.¨

Samuel Coogan. Mixed monotonicity for reachability and safety in dynamical systems. In IEEE Conference on Decision and Control (CDC), pp. 5074–5085, 2020.

Samuel Coogan and Murat Arcak. Efficient finite abstraction of mixed monotone systems. In Hybrid Systems: Computation and Control (HSCC), pp. 58–67, 2015.

Hennie Daniels and Marina Velikova. Monotone and partially monotone neural networks. IEEE Transactions on Neural Networks, 21(6):906–917, 2010.

Antoine Girard. Reachability of uncertain linear systems using zonotopes. In Hybrid Systems: Computation and Control (HSCC), pp. 291–305, 2005.

Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy A. Mann, and Pushmeet Kohli. Scalable verified training for provably robust image classification. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4841–4850, 2019.

Saber Jafarpour, Akash Harapanahalli, and Samuel Coogan. Interval reachability of nonlinear dynamical systems with neural network controllers. In Learning for Dynamics and Control Confer ence (L4DC), volume 211 of Proceedings ofMachine Learning Research, pp. 12–25, 2023.

Guy Katz, Clark Barrett, David L. Dill, Kyle Julian, and Mykel J. Kochenderfer. Reluplex: An efficient SMT solver for verifying deep neural networks. In Computer Aided Verification (CAV), pp. 97–117, 2017.

Mohammad Khajenejad and Sze Zheng Yong. Tight remainder-form decomposition functions with applications to constrained reachability and guaranteed state estimation. IEEE Transactions on Automatic Control, 68(12):7057–7072, 2023.

Ouail Kitouni, Niklas Nolte, and Mike Williams. Robust and provably monotonic networks. Machine Learning: Science and Technology, 4(3):035020, 2023.

Niklas Kochdumper and Matthias Althoff. Sparse polynomial zonotopes: A novel set representation for reachability analysis. IEEE Transactions on Automatic Control, 66(9):4043–4058, 2021.

Niklas Kochdumper, Christian Schilling, Matthias Althoff, and Stanley Bak. Open- and closed-loop neural network verification using polynomial zonotopes. In NASA Formal Methods (NFM), pp. 16–36, 2023.

Lukas Koller, Tobias Ladner, and Matthias Althoff. Set-based training for neural network verification. Transactions on Machine Learning Research, 2025.

Lukas Koller, Tobias Ladner, and Matthias Althoff. Out of the shadows: Exploring a latent space for neural network verification. In International Conference on Learning Representations (ICLR), 2026.

Tobias Ladner and Matthias Althoff. Automatic abstraction refinement in neural network verification using sensitivity analysis. In Hybrid Systems: Computation and Control (HSCC), 2023.

Xingchao Liu, Xing Han, Na Zhang, and Qiang Liu. Certified monotonic neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, 2020.

Pierre-Jean Meyer. Reachability analysis of neural networks using mixed monotonicity. IEEE Control Systems Letters, 6:3068–3073, 2022.

Mark Niklas Muller, Franziska Eckert, Marc Fischer, and Martin Vechev. Certified training: Small¨ boxes are all you need. In International Conference on Learning Representations (ICLR), 2023.

Davor Runje and Sharath M. Shankaranarayana. Constrained monotonic neural networks. In International Conference on Machine Learning (ICML), pp. 29338–29353, 2023.

Joseph Sill. Monotonic networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 10, 1997.

Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Puschel, and Martin Vechev. Fast¨ and effective robustness certification. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, 2018.

Gagandeep Singh, Timon Gehr, Markus Puschel, and Martin Vechev. An abstract domain for certify-¨ ing neural networks. Proceedings ofthe ACM on Programming Languages, 3(POPL):41:1–41:30, 2019.

Aishwarya Sivaraman, Golnoosh Farnadi, Todd Millstein, and Guy Van den Broeck. Counterexample-guided learning of monotonic neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, 2020.

Hal L. Smith. Monotone Dynamical Systems: An Introduction to the Theory of Competitive and Cooperative Systems, volume 41 of Mathematical Surveys and Monographs. American Mathematical Society, 1995.

Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U. Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulao, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard˜ interface for reinforcement learning environments. arXiv:2407.17032, 2024.

Manuel Wendl, Lukas Koller, Tobias Ladner, and Matthias Althoff. Training verifiably robust agents using set-based reinforcement learning. Transactions on Machine Learning Research, 2026.

Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, and Cho-Jui Hsieh. Automatic perturbation analysis for scalable certified robustness and beyond. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, 2020.

Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana, Xue Lin, and Cho-Jui Hsieh. Fast and complete: Enabling complete neural network verification with rapid and massively parallel incomplete verifiers. In International Conference on Learning Representations (ICLR), 2021.

Liren Yang, Oscar Mickelin, and Necmiye Ozay. On sufficient conditions for mixed monotonicity. IEEE Transactions on Automatic Control, 64(12):5080–5085, 2019.

Cunliang Ye, Yongfu Wang, Yunlong Wang, and Ming Tie. Steering angle prediction YOLOv5- based end-to-end adaptive neural network control for autonomous vehicles. Proceedings of the Institution of Mechanical Engineers, Part D: Journal ofAutomobile Engineering, 236(9):1991– 2011, 2022.

Seungil You, David Ding, Kevin Canini, Jan Pfeifer, and Maya Gupta. Deep lattice networks and partial monotonic functions. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustness certification with general activation functions. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, 2018.

## A APPENDIX

## A.1 PROOFS

Proposition 3. Given an input interval $\mathcal { X } = [ \underline { { x } } , \overline { { x } } ] \subseteq \mathbb { R } ^ { n _ { 0 } }$ and a neural network $\Phi = \Phi _ { + } - \Phi _ { - }$ where $\Phi _ { + } , \Phi $ are non-decreasing on $\mathcal { X } \ : ( D e f . \ : I )$ , it holds that

$$
\begin{array} { r } { \mathbf { 1 } ^ { \top } \nu _ { \Phi } ( \mathcal { X } ) \leq \| \mathbf { e n c l o s e } ( \Phi , \mathcal { X } ) \| _ { \mathrm { I } } . } \end{array}
$$

Moreover, there exist non-decreasingfunctions $\Phi _ { + } ^ { \prime } , \Phi _ { - } ^ { \prime } \colon \mathcal { X } \to \mathbb { R } ^ { n _ { \kappa } }$ with $\Phi _ { + } ^ { \prime } - \Phi _ { - } ^ { \prime } = \Phi o n \mathcal { X }$ , such that equality holdsfor $\Phi ^ { \prime } = \Phi _ { + } ^ { \prime } - \bar { \Phi } _ { - } ^ { \prime }$

Proof. We first compute the size of the enclosure enclose(Φ, X ), whose lower and upper bounds are $\Phi _ { + } ( \underline { { x } } ) - \Phi _ { - } ( \overline { { x } } )$ and $\Phi _ { + } ( \overline { { x } } ) - \Phi _ { - } ( \underline { { x } } )$ (Prop. 2). Since $x \leq \overline { { x } }$ and $\Phi _ { + } , \Phi$ are non-decreasing on X (Def. 1), the difference between the upper and the lower bound is

$$
\left( \Phi _ { + } ( \overline { { x } } ) - \Phi _ { - } ( \underline { { x } } ) \right) - \left( \Phi _ { + } ( \underline { { x } } ) - \Phi _ { - } ( \overline { { x } } ) \right) = \left( \Phi _ { + } ( \overline { { x } } ) - \Phi _ { + } ( \underline { { x } } ) \right) + \left( \Phi _ { - } ( \overline { { x } } ) - \Phi _ { - } ( \underline { { x } } ) \right) \geq \mathbf { 0 } .
$$

Hence, enclose $( \Phi , \mathcal { X } )$ is a non-empty interval, and its interval norm (Sec. 2) is

$$
| | \mathbf { e n c l o s e } ( \Phi , \mathcal { X } ) | | _ { \mathbf { I } } = \mathbf { 1 } ^ { \top } \big ( \Phi _ { + } ( \overline { { x } } ) - \Phi _ { + } ( \underline { { x } } ) + \Phi _ { - } ( \overline { { x } } ) - \Phi _ { - } ( \underline { { x } } ) \big ) .\tag{7}
$$

Next, we show that $\mathbf { 1 } ^ { \top } \nu _ { \Phi } ( { \boldsymbol { \chi } } ) \leq \| \mathtt { e n c l o s e } ( \Phi , { \boldsymbol { \chi } } ) \| _ { \mathrm { l } }$ . Let $( x _ { 0 } , \ldots , x _ { p } )$ be a chain in X. For all $\ell \in [ p ]$ , we have $x _ { \ell - 1 } \leq x _ { \ell } .$ , so that $\Phi _ { + } ( x _ { \ell } ) - \Phi _ { + } ( x _ { \ell - 1 } ) \geq \mathbf { 0 }$ and $\Phi _ { - } ( \dot { x _ { \ell } } ) - \Phi _ { - } ( x _ { \ell - 1 } ) \geq \mathbf { 0 } \left( \mathrm { D e f . ~ } 1 \right)$ With $\Phi = \Phi _ { + } - \Phi _ { - } \left( 2 \right)$ and the triangle inequality, we obtain

$$
| \Phi ( x _ { \ell } ) - \Phi ( x _ { \ell - 1 } ) | \leq \big ( \Phi _ { + } ( x _ { \ell } ) - \Phi _ { + } ( x _ { \ell - 1 } ) \big ) + \big ( \Phi _ { - } ( x _ { \ell } ) - \Phi _ { - } ( x _ { \ell - 1 } ) \big ) .\tag{8}
$$

We sum (8) over all $\ell \in [ p ]$ . On the right-hand side, all intermediate terms cancel; for $\Phi _ { + }$ , we obtain

$$
\sum _ { \ell = 1 } ^ { p } \bigl ( \Phi _ { + } ( x _ { \ell } ) - \Phi _ { + } ( x _ { \ell - 1 } ) \bigr ) = \Phi _ { + } ( x _ { p } ) - \Phi _ { + } ( x _ { 0 } ) \leq \Phi _ { + } ( \overline { { x } } ) - \Phi _ { + } ( \underline { { x } } ) ,
$$

where the inequality holds because $\underline { { x } } \leq x _ { 0 } , x _ { p } \leq \overline { { x } } .$ , and $\Phi _ { + }$ is non-decreasing on $x ;$ the same holds for $\Phi _ { - } .$ . Therefore,

$$
\sum _ { \ell = 1 } ^ { p } \lvert \Phi ( x _ { \ell } ) - \Phi ( x _ { \ell - 1 } ) \rvert \leq \Phi _ { + } ( \overline { { x } } ) - \Phi _ { + } ( \underline { { x } } ) + \Phi _ { - } ( \overline { { x } } ) - \Phi _ { - } ( \underline { { x } } ) .\tag{9}
$$

The right-hand side of (9) does not depend on the chain; hence, it is an upper bound of the supremum in $( 6 ) , \bar { \mathrm { i . e . , } } \nu _ { \Phi } ( { \mathcal X } ) \leq \Phi _ { + } ( \overline { { x } } ) - \Phi _ { + } ( \underline { { x } } ) \bar { + } \Phi _ { - } ( \overline { { x } } ) - \Phi _ { - } ( \underline { { x } } )$ . Multiplying both sides by $\mathbf { 1 } ^ { \top }$ and replacing the right-hand side by ∥enclose $( \Phi , { \mathcal { X } } ) \| _ { \mathrm { I } }$ using (7) yields $\mathbf { 1 } ^ { \top } \nu _ { \Phi } \mathbf { \dot { ( } } \mathcal { X ) } \leq |$ enclose $( \Phi , { \mathcal { X } } ) \| _ { \mathrm { I } }$

Finally, we show that the lower bound is attained by a suitable decomposition of Φ. For any nondecreasing functions $\Phi _ { + } ^ { \prime }$ and $\Phi _ { - } ^ { \prime }$ with $\Phi ^ { \prime } = \Phi _ { + } ^ { \prime } - \dot { \Phi } _ { - } ^ { \prime }$ , rearranging (7) gives

$$
\begin{array} { r } { \| \mathbf { e n c l o s e } ( \Phi ^ { \prime } , \mathcal { X } ) \| _ { \mathrm { I } } = \mathbf { 1 } ^ { \top } \Big ( \big ( \Phi _ { + } ^ { \prime } ( \overline { { x } } ) + \Phi _ { - } ^ { \prime } ( \overline { { x } } ) \big ) - \big ( \Phi _ { + } ^ { \prime } ( \underline { { x } } ) + \Phi _ { - } ^ { \prime } ( \underline { { x } } ) \big ) \Big ) . } \end{array}\tag{10}
$$

Hence, the size of the enclosure is determined by the values of the sum $\Phi _ { + } ^ { \prime } + \Phi _ { - } ^ { \prime }$ at the two corners of $\mathcal { X } .$ We therefore prescribe the difference and the sum of the two functions: the difference $\Phi _ { + } ^ { \prime } ( x ) - \Phi _ { - } ^ { \prime } ( x )$ must reproduce $\Phi ( x )$ , and the sum $\Phi _ { + } ^ { \prime } ( x ) + \Phi _ { - } ^ { \prime } ( x )$ should equal the variation of $\Phi$ accumulated from the lower corner x to $x , \mathrm { i . e . }$ ., for all ${ \dot { x } } \in { \mathcal { X } }$

$$
\Phi _ { + } ^ { \prime } ( x ) - \Phi _ { - } ^ { \prime } ( x ) = \Phi ( x ) ,
$$

$$
\Phi _ { + } ^ { \prime } ( x ) + \Phi _ { - } ^ { \prime } ( x ) = \nu _ { \Phi } ( [ \underline { { { x } } } , x ] ) .\tag{11}
$$

The variation $\nu _ { \Phi } ( \left[ \underline { { x } } , x \right] )$ in (11) is well defined: since $\underline { { x } } \le x \le \overline { { x } }$ , each chain in $[ \underline { { x } } , x ]$ is also a chain in $x ,$ so that $\mathbf { 0 } \leq \nu _ { \Phi } \bar { ( } [ \underline { { { x } } } , x ] ) \leq \nu _ { \Phi } ( \mathcal { X } )$ by (6), where $\nu _ { \Phi } ( \mathcal { X } )$ is finite due to (9). Adding the two equations in (11) eliminates $\Phi _ { - } ^ { \prime }$ , and subtracting the first from the second eliminates $\Phi _ { + } ^ { \prime }$

$$
2 \Phi _ { + } ^ { \prime } ( x ) = \underbrace { \bigl ( \Phi _ { + } ^ { \prime } ( x ) + \Phi _ { - } ^ { \prime } ( x ) \bigr ) } _ { = \nu _ { \Phi } ( [ \underline { { x } } , x ] ) } + \underbrace { \bigl ( \Phi _ { + } ^ { \prime } ( x ) - \Phi _ { - } ^ { \prime } ( x ) \bigr ) } _ { = \Phi ( x ) } = \nu _ { \Phi } ( [ \underline { { x } } , x ] ) + \Phi ( x ) ,
$$

$$
2 \Phi _ { - } ^ { \prime } ( x ) = \underbrace { \bigl ( \Phi _ { + } ^ { \prime } ( x ) + \Phi _ { - } ^ { \prime } ( x ) \bigr ) } _ { = \nu _ { \Phi } ( [ \underline { { x } } , x ] ) } - \underbrace { \bigl ( \Phi _ { + } ^ { \prime } ( x ) - \Phi _ { - } ^ { \prime } ( x ) \bigr ) } _ { = \Phi ( x ) } = \nu _ { \Phi } ( [ \underline { { x } } , x ] ) - \Phi ( x ) .
$$

Dividing by two yields the unique solution

$$
\begin{array} { r } { \Phi _ { + } ^ { \prime } ( x ) = \frac 1 2 \bigl ( \nu _ { \Phi } ( [ \underline { { x } } , x ] ) + \Phi ( x ) \bigr ) , \qquad \Phi _ { - } ^ { \prime } ( x ) = \frac 1 2 \bigl ( \nu _ { \Phi } ( [ \underline { { x } } , x ] ) - \Phi ( x ) \bigr ) . } \end{array}\tag{12}
$$

It remains to show that $\Phi _ { + } ^ { \prime }$ and $\Phi _ { - } ^ { \prime }$ are non-decreasing on X and that they attain the lower bound. To show that $\Phi _ { + } ^ { \prime }$ and $\Phi _ { - } ^ { \prime }$ are non-decreasing on $x ,$ let $x , x ^ { \prime } \in { \mathcal { X } }$ with $x \leq x ^ { \prime }$ . We first show that the accumulated variation grows at least by the change of Φ between x and $x ^ { \prime }$ . Let $( x _ { 0 } , \ldots , x _ { p } )$ be a chain in $[ \underline { { x } } , x ]$ . Since $x _ { p } \leq x \leq x ^ { \prime }$ , appending x and $\overline { { x ^ { \prime } } }$ yields the chain $( x _ { 0 } , \ldots , x _ { p } , x , x ^ { \prime } )$ in $[ \underline { { x } } , x ^ { \prime } ]$ whose summed change is bounded by the supremum in (6) for the interval $[ { \underline { { x } } } , x ^ { \prime } ] \colon$

$$
\sum _ { \ell = 1 } ^ { p } \lvert \Phi ( x _ { \ell } ) - \Phi ( x _ { \ell - 1 } ) \rvert + \lvert \Phi ( x ) - \Phi ( x _ { p } ) \rvert + \lvert \Phi ( x ^ { \prime } ) - \Phi ( x ) \rvert \leq \nu _ { \Phi } ( [ \underline { { x } } , x ^ { \prime } ] ) .
$$

Omitting the non-negative middle term and taking the supremum over all chains in $[ \underline { { x } } , x ]$ results in

$$
\nu _ { \Phi } ( [ \underline { { { x } } } , x ^ { \prime } ] ) - \nu _ { \Phi } ( [ \underline { { { x } } } , x ] ) \geq | \Phi ( x ^ { \prime } ) - \Phi ( x ) | .\tag{13}
$$

Subtracting (12) at x from (12) at $x ^ { \prime }$ and bounding the growth of the accumulated variation by (13) yields

$$
\begin{array} { r l } & { \Phi _ { + } ^ { \prime } ( x ^ { \prime } ) - \Phi _ { + } ^ { \prime } ( x ) = \frac { 1 } { 2 } \Bigl ( \underbrace { \nu _ { \Phi } ( [ \underline { { x } } , x ^ { \prime } ] ) - \nu _ { \Phi } ( [ \underline { { x } } , x ] ) } _ { \geq | \Phi ( x ^ { \prime } ) - \Phi ( x ) | \mathrm { b y ~ ( 1 3 ) } } + \bigl ( \Phi ( x ^ { \prime } ) - \Phi ( x ) \bigr ) \Bigr ) } \\ & { \qquad \geq \frac { 1 } { 2 } \Bigl ( | \Phi ( x ^ { \prime } ) - \Phi ( x ) | + \bigl ( \Phi ( x ^ { \prime } ) - \Phi ( x ) \bigr ) \Bigr ) \geq \bf { 0 } , } \end{array}
$$

$$
\begin{array} { r l } & { \Phi _ { - } ^ { \prime } ( x ^ { \prime } ) - \Phi _ { - } ^ { \prime } ( x ) = \frac { 1 } { 2 } \Bigl ( \underbrace { \nu _ { \Phi } ( [ \underline { { x } } , x ^ { \prime } ] ) - \nu _ { \Phi } ( [ \underline { { x } } , x ] ) } _ { \geq | \Phi ( x ^ { \prime } ) - \Phi ( x ) | \mathrm { b y ~ ( 1 3 ) } } - \bigl ( \Phi ( x ^ { \prime } ) - \Phi ( x ) \bigr ) \Bigr ) } \\ & { \qquad \geq \frac { 1 } { 2 } \Bigl ( | \Phi ( x ^ { \prime } ) - \Phi ( x ) | - \bigl ( \Phi ( x ^ { \prime } ) - \Phi ( x ) \bigr ) \Bigr ) \geq \bf { 0 } , } \end{array}
$$

where the last inequalities hold component-wise because $| a | + a \geq 0$ and $| a | - a \geq 0$ for every $a \in \mathbb { R }$

As $\Phi _ { + } ^ { \prime }$ and $\Phi _ { - } ^ { \prime }$ are non-decreasing on X, (10) holds for $\Phi ^ { \prime }$ , and we evaluate the sums at the two corners using (11). ${ \mathrm { A t ~ } } x = { \overline { { x } } }$ , the interval $[ \underline { { x } } , \overline { { x } } ]$ is $\mathcal { X }$ ${ \mathrm { A t ~ } } x = { \underline { { x } } } ,$ the interval $[ \underline { { x } } , \underline { { x } } ]$ only contains the point $\underline { { x } } ,$ so that every chain in it is constant, all summands in (6) are zero, and $\dot { \nu } _ { \Phi } ( [ \underline { { { \dot { x } } } } , \underline { { { x } } } ] ) = \mathbf { 0 }$ Hence,

$$
\begin{array} { r } { \| \mathsf { e n c l o s e } ( \Phi ^ { \prime } , \mathcal { X } ) \| _ { \mathrm { I } } = \mathbf { 1 } ^ { \top } \Big ( \underbrace { \Phi _ { + } ^ { \prime } ( \overline { { x } } ) + \Phi _ { - } ^ { \prime } ( \overline { { x } } ) } _ { = \nu _ { \Phi } ( [ \underline { { x } } , \overline { { x } } ] ) = \nu _ { \Phi } ( \mathcal { X } ) } - \underbrace { \big ( \Phi _ { + } ^ { \prime } ( \underline { { x } } ) + \Phi _ { - } ^ { \prime } ( \underline { { x } } ) \big ) } _ { = \nu _ { \Phi } ( [ \underline { { x } } , \underline { { x } } ] ) = \mathbf { 0 } } \Big ) = \mathbf { 1 } ^ { \top } \nu _ { \Phi } ( \mathcal { X } ) . } \end{array}
$$

Thus, $\Phi _ { + } ^ { \prime }$ and $\Phi _ { - } ^ { \prime }$ are non-decreasing on $x ,$ , satisfy $\Phi _ { + } ^ { \prime } - \Phi _ { - } ^ { \prime } = \Phi \mathrm { o n } \mathcal { X } .$ , and attain the lower bound, which proves the second statement. □

## A.2 BENCHMARK DEFINITIONS

In this subsection, we define the scalar analytic model, the two dynamical systems, and the multiinput analytic model.

Scalar analytic model The neural network is trained to approximate the model

$$
f ( x ) = \sin x + 0 . 3 \sin 3 x + 0 . 0 5 x\tag{14}
$$

on $[ - 3 , 3 ]$ . This model is not monotone, as it has six local extrema in $[ - 3 , 3 ]$

Dynamical systems A system ${ \dot { x } } = f ( x , u )$ is cooperative if it satisfies the Kamke–Muller condi-¨ tion (Smith, 1995)

$$
{ \frac { \partial f _ { ( i ) } } { \partial x _ { ( j ) } } } \geq 0 \ \forall i \neq j , \qquad { \frac { \partial f } { \partial u } } \geq 0 .\tag{15}
$$

Every cooperative system is also a monotone system in the sense that, for a constant input, all solutions starting in an interval $[ \underline { { x } } , \overline { { x } } ]$ are bounded by the two solutions starting in $\underline { { x } }$ and x (Smith, 1995).

The Pendulum system uses the dynamics of the Gymnasium environment Pendulum-v1 (Towers et al., 2024); due to the sin term, it violates (15) and is thus non-cooperative. The Tanks system consists of two cascaded tanks with the fill levels $x _ { ( 1 ) }$ and $x _ { ( 2 ) }$ and the inflow u:

$$
\begin{array} { l } { { \dot { x } _ { ( 1 ) } = - k _ { 1 } \sqrt { x _ { ( 1 ) } } + k _ { 4 } u , } } \\ { { \dot { x } _ { ( 2 ) } = k _ { 2 } \sqrt { x _ { ( 1 ) } } - k _ { 3 } \sqrt { x _ { ( 2 ) } } , } } \end{array}\tag{16}
$$

with $k _ { 1 } = 0 . 5 , k _ { 2 } = 0 . 4 , k _ { 3 } = 0 . 3 , k _ { 4 } = 0 . 1 2$ , and $u \in [ 0 , 1 0 ]$ . The system satisfies (15) and is thus cooperative. We integrate the system with a fourth-order Runge–Kutta scheme and use a time step of 4 s.

For the prediction accuracy (Tab. 1), the neural network is trained to predict the change of the state within one time step from the current state and the input; for the verification-aware training (Tab. 3) and the evaluation over multiple time steps (Sec. 5.4), which both use the Tanks system, it is trained to predict the next state. For these two evaluations, we predict the next state, because the Tanks system is cooperative, so that the map from the current state to the next state is monotone for a constant input, whereas the change of the state does not need to be monotone. For the Tanks system with eight layers, the tightness (Sec. 5) of our enclosure is 1.08 for the next state and 1.37 for the change of the state $( \epsilon = 0 . 0 5$ , ten seeds).

Multi-input analytic model The neural network is trained to approximate the model

$$
f ( x ) = \sin x _ { ( 1 ) } + 0 . 3 \sin 3 x _ { ( 2 ) } + { \frac { 0 . 0 5 } { n _ { 0 } } } \sum _ { i = 1 } ^ { n _ { 0 } } x _ { ( i ) }\tag{17}
$$

with $n _ { 0 }$ inputs in $[ - 3 , 3 ]$ , of which only $x _ { ( 1 ) }$ and $x _ { ( 2 ) }$ enter nonlinearly. We perturb either all inputs by the same radius, so that the size of the input set increases with $n _ { 0 }$ , or only $x _ { ( 1 ) }$ and $x _ { ( 2 ) }$ , so that the size of the input set does not depend on $n _ { 0 }$

## A.3 COMPUTATION OF THE TIGHTNESS

Since the exact image $\Phi ( \mathcal { X } )$ is not available, we evaluate the neural network at uniformly sampled inputs in X and at the corners of $\mathcal { X } \left( \mathrm { f o r } n _ { 0 } > 1 2 \right.$ , only at x and x). The resulting outputs are a subset of $\Phi ( \mathcal { X } )$ ; thus, their size is at most $\| \Phi ( { \mathcal { X } } ) \| _ { \mathrm { I } }$ . As we divide the size of the enclosure by the size of the sampled outputs, the reported tightness is an upper bound of the tightness with respect to the exact image. For the dynamical systems, the inputs and outputs of the neural network are standardized using the mean and the standard deviation of the training data; the perturbation radii refer to the standardized inputs, and the sizes of the enclosures and of the sampled outputs are computed in the original units of the outputs. For the static analytic models, the inputs and outputs are not rescaled. For each seed, we first average the sizes of the enclosures and the sizes of the sampled outputs over all input sets and then compute their ratio, so that input sets with a small image do not dominate the result; we report the mean and the standard deviation of this ratio over the seeds. We mark a configuration with † if the prediction accuracy of at least one seed is below $R ^ { 2 } = 0 . 9 ;$ we do not use marked configurations to compare architectures or verification methods. The numbers of input sets and seeds of each experiment are listed in Tab. 6, and the runtime measurements are described in Sec. A.6.

## A.4 EXPERIMENTAL DETAILS

Architectures Each of our sign-constrained subnetworks has 32 neurons per hidden layer, of which we assign one third to each of the activation functions ReL $\mathrm { U } ( x ) , - \mathrm { R e L } \bar { \mathrm { U } } ( - x )$ , and tanh(x). For multiple outputs, each output has its own pair of subnetworks. We compare our mixed-monotone neural networks with unconstrained neural networks (baseline). Additionally, we use the monotone Lipschitz-constrained neural networks of Kitouni et al. (2023) as subnetworks in $( 2 ) \ { \mathrm { ( L i p . - r e s . ) } }$ where each subnetwork has the form $x \mapsto \lambda \mathbf { 1 } ^ { \top } x + h ( x )$ with a neural network h whose Lipschitz constant with respect to the $\ell _ { 1 }$ -norm is at most λ, and we consider non-monotone neural networks that use the Lipschitz-constrained layers of Kitouni et al. (2023) without the linear function (Lip.). The number of parameters of each architecture differs from that of our neural networks by at most 2 %; to this end, we adapt the number of neurons per hidden layer of the baseline and of the non-monotone Lipschitz-constrained neural networks. Within each experiment, all architectures use the same data splits, optimizer settings, and number of epochs, which we did not tune per architecture.

Initialization We initialize each weight of a sign-constrained linear layer with the absolute value of a sample from a standard normal distribution, which we scale such that the expected value of each weight is $c / n _ { k - 1 }$ with $c = 1 . 4 .$ . In contrast to weights with zero mean, non-negative weights do not cancel each other in the weighted sum of a neuron: for weights that are independent of the inputs, the expected weighted sum of a neuron is $n _ { k - 1 }$ times the expected weight times the mean of its inputs. For the expected weight $c / n _ { k - 1 }$ , this expected value does not depend on $n _ { k - 1 }$ , whereas it is larger by the factor $\sqrt { n _ { k - 1 } }$ for the common scaling by $1 / \sqrt { n _ { k - 1 } }$ , which can accumulate over the layers. In our experiments, the magnitude of the outputs of all layers remains within $[ 0 . 3 , 6 ]$ for the evaluated numbers of layers. The training is sensitive to $c , \mathrm { e . g . }$ ., for $c = 5$ , the neural network does not fit the scalar analytic model.

Training We train all neural networks with Adam using a learning rate of $2 \cdot 1 0 ^ { - 3 }$ and a batch size of 64, without learning-rate decay and without early stopping; the number of epochs is listed in Tab. 6. For $\beta > 0$ , we increase the weight of the enclosure size in the training loss (5) in three stages of the training from 0 to $\beta / 2$ to $\beta ,$ so that the neural network fits the data before the enclosure is tightened; the state of the optimizer is kept between the stages. In each training iteration, we evaluate the regression loss for 64 training inputs and the enclosure size for 8 input sets with the radius $\epsilon = 0 . 2 ( \mathrm { S e c . } \mathrm { A } . 3 )$ , which are centered at randomly drawn training inputs and clipped to the training domain. The 64 training inputs and the 16 corners of the input sets are evaluated in a single forward pass of one batch. The input sets are drawn once per stage, resulting in $\lfloor N / 6 4 \rfloor \cdot 8$ different input sets for N training inputs.

Non-negative weights The sign-constrained subnetworks require non-negative weights in the forward pass (Def. 2). We use $| W |$ in the forward pass and optimize the unconstrained weights $W .$ , so that the gradient of the training loss follows from the chain rule:

$$
\frac { \partial L } { \partial W } = \frac { \partial L } { \partial | W | } \odot \mathrm { s i g n } ( W ) ,\tag{18}
$$

where ⊙ denotes the elementwise product; for entries that are zero, where the absolute value is not differentiable, we use +1, which is an element of the subdifferential [−1, 1] of the absolute value at zero; with the subgradient 0, a weight that is exactly zero would not be updated anymore. Hence, the weights used in the forward pass are non-negative throughout the training, which preserves the monotonicity of both subnetworks. An alternative is to project the weights onto the non-negative values during the training, i.e., $W \gets \operatorname* { m a x } ( W , \mathbf { 0 } )$ . In a preliminary experiment on the scalar analytic model, in which we applied this projection every 30 epochs, 32.3 % of the weights were exactly zero after 120 epochs and the prediction accuracy decreased from $R ^ { 2 } = 0 . 9 8$ to 0.76; therefore, we use the absolute value instead.

Settings of the experiments Tab. 6 lists the settings of all experiments. The neural networks for the comparison of verification methods (Tab. 2 and 8) and for the comparison across the number of layers (Tab. 5) are trained separately, both with $\beta = 0 ;$ ; therefore, the tightness of our enclosure in Tab. 2 and Tab. 5 differs for the same number of layers. The tightness of an enclosure is estimated with 800 samples per input set (2000 for the evaluation over multiple time steps). All random choices of an experiment, i.e., the dataset, its split, the initialization of the weights, the input sets, and the samples, are derived from a single seed. For the dynamical systems, we split the dataset by trajectories, so that the states of one trajectory are not distributed over the training, validation, and test data.

## A.5 ADDITIONAL RESULTS ON THE PREDICTION ACCURACY

Tab. 7 shows the prediction accuracy on the scalar analytic model for all tested numbers of layers and architectures. The baseline remains accurate for all numbers of layers, and the accuracy of our architecture differs from it by at most 0.01. In contrast, both Lipschitz-constrained architectures lose accuracy at six and eight layers when trained with the same settings: at eight layers, the monotone Lipschitz-constrained architecture reaches $R ^ { 2 } = 0 . 7 5 2 \pm 0 . 0 4 8$ and the non-monotone one $R ^ { 2 } =$ $0 . { \bar { 6 } } 3 5 \pm 0 . 0 7 4$

Table 6: Settings of the experiments.
<table><tr><td>Experiment</td><td>Benchmark</td><td>#Layers κ</td><td> $\beta$ </td><td>#Seeds</td><td>#Epochs</td><td>#Input Sets</td></tr><tr><td>Tab. 1 and 7</td><td>Scalar analytic</td><td>2, 3, 4, 6,8</td><td>0</td><td>10</td><td>120</td><td></td></tr><tr><td>Tab. 1</td><td>Pendulum, Tanks</td><td>3,8</td><td>0</td><td>10</td><td>100</td><td></td></tr><tr><td>Tab. 2 and 8</td><td>Scalar analytic</td><td>2, 3, 4, 6,8</td><td>0</td><td>5</td><td>120</td><td>100</td></tr><tr><td>Tab.3</td><td>Scalar analytic, Tanks</td><td>3</td><td>0-1</td><td>3</td><td>80</td><td>150</td></tr><tr><td>Tab. 4 and 9</td><td>Multi-input analytic</td><td>4</td><td>0.05</td><td>5</td><td>120</td><td>100</td></tr><tr><td>Tab.5</td><td>Scalar analytic</td><td>2,3,4, 6,8</td><td>0</td><td>10</td><td>120</td><td>150</td></tr><tr><td>Fig. 3</td><td>Tanks</td><td>3,8</td><td>0.05</td><td>3</td><td>100</td><td>20</td></tr></table>

Table 7: Comparing the prediction accuracy of different architectures on the scalar analytic model shown as mean ± standard deviation over ten seeds; bold font marks the best architecture per number of layers.
<table><tr><td rowspan="2">#Layers κ</td><td colspan="4">Architecture</td></tr><tr><td>Baseline (unconstr.)</td><td>Ours  $( \mathrm { s i g n - c o n s t r } )$ </td><td>Lip.-res. (monotone) Kitouni et al. (2023)</td><td>Lip. (non-monotone)</td></tr><tr><td>2</td><td> $0 . 9 8 4 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 9 8 6 \pm 0 . 0 0 1 }$ </td><td> $0 . 9 7 3 \pm 0 . 0 0 2$ </td><td> $0 . 9 7 2 \pm 0 . 0 0 2$ </td></tr><tr><td>3</td><td> ${ \bf 0 . 9 9 9 \pm 0 . 0 0 2 }$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 1$ </td></tr><tr><td>4</td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 5$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 1 \pm 0 . 0 0 1$ </td></tr><tr><td>6</td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 6$ </td><td> $0 . 8 7 6 \pm 0 . 0 2 2$ </td><td> $0 . 8 3 8 \pm 0 . 0 2 8$ </td></tr><tr><td>8</td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 6 \pm 0 . 0 0 4$ </td><td> $0 . 7 5 2 \pm 0 . 0 4 8$ </td><td> $0 . 6 3 5 \pm 0 . 0 7 4$ </td></tr></table>

## A.6 ADDITIONAL RESULTS ON THE COMPARISON OF VERIFICATION METHODS

Tab. 8 extends Tab. 2 to all tested numbers of layers. CROWN is tighter than our enclosure at two and three layers, and our enclosure is tighter at six and eight layers; at four layers, the difference is smaller than the standard deviation over the seeds. The tightness of α-CROWN, which optimizes its bounds with 20 iterations and a learning rate of 0.1, is at most 1.004 for all numbers of layers.

Interval bound propagation For our architecture, IBP returns the same enclosure as Prop. 2. IBP propagates an interval [l, u] through a linear layer with weights W and bias b to the interval $\lvert \mathrm { \dot { m a x } } ( \tilde { W } , { \mathbf 0 } ) l + \mathrm { m i n } ( W , { \mathbf 0 } ) \dot { u } + b , \mathrm { \dot { m a x } } ( W , { \mathbf 0 } ) u \dot { + } \mathrm { m i n } ( W , { \mathbf 0 } ) \dot { l } + b ]$ , where the maximum and the minimum are taken element-wise (Gowal et al., 2019). For non-negative weights, we have min ${ \bf \nabla } _ { ( W , { \bf 0 } ) } = { \bf 0 }$ , so that the resulting interval is $[ W l + b , W u + b ] , \mathrm { i . e . }$ ., the layer is evaluated at the lower and the upper bound of the input interval. A non-decreasing activation function maps an interval [l, u] to the interval between the function values at l and u. By induction over the layers, IBP thus propagates the input interval $[ \underline { { x } } , \overline { { x } } ]$ through $\Phi _ { + }$ and $\Phi _ { - } \ \mathrm { t o }$ the intervals $[ \Phi _ { + } ( \underline { { x } } ) , \Phi _ { + } ( \overline { { x } } ) ]$ and $[ \Phi _ { - } ( { \underline { { x } } } ) , \Phi _ { - } ( { \overline { { x } } } ) ]$ . Subtracting the second interval from the first one in interval arithmetic yields $[ \Phi _ { + } ( \underline { { { x } } } ) - \Phi _ { - } ( \underline { { { x } } } ) , \bar { \Phi } _ { + } ( \underline { { { x } } } ) - \Phi _ { - } ( \underline { { { x } } } ) ] \ = \ \mathrm { e n c l o s e } ( \Phi , \mathcal { X } )$ The equality holds because all weights of the subnetworks are non-negative: for the Lipschitz-constrained subnetworks of Kitouni et al. (2023), whose weights are not sign-constrained, the size of the IBP enclosure is larger than the size of our enclosure by a factor of 1.49 at two layers and 1.15 at eight layers $( \epsilon = 0 . 0 5$ , ten seeds). We verified the equality numerically: in CORA, the sizes of both enclosures agree up to the reported precision, and for the neural networks exported to auto LiRPA, the relative difference of the sizes is at most $1 . 1 \cdot 1 0 ^ { - 6 }$ with a median of $6 \cdot 1 0 ^ { - 8 }$ over all exported neural networks and tested perturbation radii.

Table 8: Comparing enclosure tightness and runtimes of all verification methods on the scalar analytic model at $\epsilon = 0 . 0 5$ for all tested numbers of layers. The tightness is shown as mean ± standard deviation over five seeds, and the runtimes are relative to our enclosure within the same toolbox; bold font marks the best tightness per number of layers.
<table><tr><td rowspan="2">#Layers κ</td><td colspan="4">Tightness ↓</td><td colspan="4">Runtime relative to ours ↓</td></tr><tr><td>Ours / IBP</td><td>Zonotope</td><td>CROWN</td><td>α-CROWN</td><td></td><td>IBP Zono. CROWN</td><td></td><td>α-CR.</td></tr><tr><td>2</td><td> $3 . 5 8 \pm 0 . 5 5$ </td><td></td><td>1.08 ± 0.01 1.07 ± 0.01</td><td> $\mathbf { 1 . 0 0 1 \pm 0 . 0 0 0 }$ </td><td>10.8</td><td>6.5</td><td>23.5</td><td>1490</td></tr><tr><td>3</td><td></td><td></td><td></td><td> $2 . 0 0 \pm 0 . 4 3 ~ 1 . 1 4 \pm 0 . 0 3 ~ 1 . 1 2 \pm 0 . 0 3 ~ 1 . 0 0 4 \pm 0 . 0 0 2$ </td><td>8.1</td><td>8.4</td><td>54.4</td><td>6130</td></tr><tr><td>4</td><td></td><td></td><td></td><td> $1 . 1 7 \pm 0 . 1 2 \ 1 . 1 6 \pm 0 . 0 5 \ 1 . 1 3 \pm 0 . 0 4 \ 1 . 0 0 2 \pm 0 . 0 0 2$ </td><td>9.9</td><td>11.7</td><td>112</td><td>8750</td></tr><tr><td>6</td><td></td><td></td><td>1.07 ± 0.08 1.25 ± 0.10 1.18 ± 0.04</td><td> $\mathbf { 1 . 0 0 2 \pm 0 . 0 0 1 }$ </td><td>10.2</td><td>19.6</td><td></td><td>193 12900</td></tr><tr><td>8</td><td></td><td></td><td>1.05 ± 0.08 1.44 ± 0.07 1.26 ± 0.06</td><td> $\mathbf { 1 . 0 0 2 \pm 0 . 0 0 1 }$ </td><td>11.2</td><td>31.8</td><td></td><td>256 20 600</td></tr></table>

Export to auto LiRPA We compute the bounds of IBP, CROWN, and α-CROWN with auto LiRPA v0.7.2 (PyTorch 2.11.0, Python 3.11), which requires a PyTorch model; therefore, we export the neural networks trained in CORA to PyTorch. The exported model reproduces every layer of the CORA neural network: the weights $W _ { k }$ of the linear layers enter the forward pass as $| W _ { k } | ,$ , and the nonlinear layers apply the three activation functions as described in Sec. A.4. The subtraction $\Phi = \Phi _ { + } - \Phi _ { - } \mathrm { \bf ~ \dot { i } s }$ implemented as a final linear layer with the fixed weights $[ I - I ]$ where I is the identity matrix, so that all methods bound the output of Φ instead of bounding the two subnetworks separately. The weights are transferred from the neural networks trained in CORA without retraining. For every exported neural network, we check that its outputs match the outputs of the CORA neural network for 512 uniformly sampled inputs and that IBP in auto LiRPA returns our enclosure.

Runtime measurements All neural networks are trained on a Linux cluster. The runtimes in CORA (Tab. 2 and 8 and Sec. A.8) are measured on an otherwise idle machine with an Intel Core Ultra 9 285K (24 cores, 62 GB) using MATLAB R2026a and CORA v2026.1.0. For each configuration, we measure the runtime for 200 input sets and report the minimum over 25 repetitions after one warm-up evaluation, as the mean is dominated by isolated outliers; the interquartile range is within $1 0 \%$ of the median for our enclosure and within 20 % for zonotope propagation. As the runtime does not depend on the values of the weights, we use randomly initialized neural networks for these measurements. The runtimes in auto LiRPA are measured on a different machine with an AMD Ryzen 7 Microsoft Surface Edition (Zen 2, 8 cores, 16 GB) and PyTorch restricted to eight threads; we report the minimum over ten repetitions after one warm-up evaluation and exclude the construction of the BoundedModule, which is only required once per neural network. Therefore, we only compare runtimes that are measured within the same toolbox. Although IBP computes the same bounds as our enclosure, IBP in auto LiRPA is eight to eleven times slower than our two forward passes for all tested numbers of layers and perturbation radii; our measurements do not separate the overhead of the toolbox from the cost of the bound computation.

## A.7 ADDITIONAL RESULTS ON THE NUMBER OF LAYERS

This subsection qualifies the decrease of the crossover radius with the number of layers in Tab. 5. Between six and eight layers, the medians of the crossover radius differ by one step of the tested perturbation radii, and the ranges of the crossover radius over the seeds overlap; hence, our measurements do not show a further decrease of the crossover radius from six to eight layers. At eight layers, the ratio of the size of our enclosure to the perturbation radius decreases from 1.448 to 1.219 over two orders of magnitude of the perturbation radius, while this ratio increases by a factor of twelve for zonotope propagation.

## A.8 ADDITIONAL RESULTS ON THE INPUT DIMENSION

Tab. 9 shows the enclosure tightness for all tested input dimensions and for both ways of perturbing the inputs (Sec. A.2). Our architecture with sign-constrained subnetworks (Ours) and with Lipschitzconstrained subnetworks (Lip.-res.) is enclosed using Prop. 2, and the baseline is enclosed using zonotope propagation. If only the first two inputs are perturbed, the interval norm of the input set is approximately 1.9 for all input dimensions, and the tightness of our enclosure remains below 1.05. If all inputs are perturbed, the interval norm of the input set and the tightness of all enclosures increase with the input dimension. In CORA, computing our enclosure takes 13 to $1 6 \mu \mathrm { s }$ per input set for all tested input dimensions, while zonotope propagation through the baseline takes 4.5 to 7.5 times as long (Sec. A.6).

Table 9: Comparing the enclosure tightness for different input dimensions $n _ { 0 }$ at $\epsilon = 0 . 5 .$ , when all inputs or only the first two inputs are perturbed, shown as mean ± standard deviation over five seeds; † marks configurations with $\mathbf { \hat { \mathit { R } } ^ { 2 } < 0 . 9 }$ (Sec. A.3).
<table><tr><td>Perturbed</td><td> $n _ { 0 }$ </td><td>Input Set Size  $\| { \mathcal { X } } \| _ { \mathrm { I } }$ </td><td>Ours</td><td>Lip.-res. Kitouni et al. (2023)</td><td>Zonotope</td></tr><tr><td rowspan="6">all</td><td>2</td><td>1.92</td><td> $1 . 0 4 5 \pm 0 . 0 0 8$ </td><td> $1 9 . 4 \pm 0 . 4$ </td><td> $1 . 6 5 \pm 0 . 0 1$ </td></tr><tr><td>5</td><td>4.82</td><td> $1 . 0 5 4 \pm 0 . 0 1 2$ </td><td> $3 8 . 2 \pm 3 . 2 $ </td><td> $1 . 5 9 \pm 0 . 0 5$ </td></tr><tr><td>10</td><td>9.56</td><td> $1 . 0 5 0 \pm 0 . 0 0 9$ </td><td> $8 6 . 5 ^ { \dagger } \pm 6 . 9$ </td><td> $1 . 6 6 \pm 0 . 0 8$ </td></tr><tr><td>20</td><td>19.18</td><td> $1 . 0 7 3 \pm 0 . 0 0 9$ </td><td> $1 8 1 ^ { \dagger } \pm 3 7$ </td><td> $1 . 7 7 \pm 0 . 0 9$ </td></tr><tr><td>50</td><td>47.90</td><td> $1 . 1 0 5 \pm 0 . 0 1 3$ </td><td> $3 7 8 ^ { \dag } \pm 3 8$ </td><td> $1 5 2 ^ { \dagger } \pm 1 2$ </td></tr><tr><td>100</td><td>95.91</td><td> $1 . 1 6 7 \pm 0 . 0 3 9$ </td><td> $8 6 2 ^ { \dagger } \pm 9 5$ </td><td> $3 6 0 ^ { \dagger } \pm 3 1$ </td></tr><tr><td rowspan="6">first two</td><td>2</td><td>1.92</td><td> $1 . 0 4 5 \pm 0 . 0 0 8$ </td><td> $1 9 . 4 \pm 0 . 4$ </td><td> $1 . 6 5 \pm 0 . 0 1$ </td></tr><tr><td>5</td><td>1.93</td><td> $1 . 0 4 7 \pm 0 . 0 1 1$ </td><td> $1 6 . 1 \pm 1 . 3$ </td><td> $1 . 5 4 \pm 0 . 0 5$ </td></tr><tr><td>10</td><td>1.90</td><td> $1 . 0 4 1 \pm 0 . 0 0 6$ </td><td> $1 9 . 3 ^ { \dagger } \pm 1 . 8$ </td><td> $1 . 5 5 \pm 0 . 0 7$ </td></tr><tr><td>20</td><td>1.91</td><td> $1 . 0 4 5 \pm 0 . 0 0 8$ </td><td> $2 0 . 3 ^ { \dagger } \pm 4 . 9$ </td><td> $1 . 5 2 \pm 0 . 0 5$ </td></tr><tr><td>50</td><td>1.91</td><td> $1 . 0 4 0 \pm 0 . 0 0 5$ </td><td> $1 6 . 8 ^ { \dagger } \pm 1 . 8$ </td><td> $2 . 4 2 ^ { \dagger } \pm 0 . 1 6$ </td></tr><tr><td>100</td><td>1.92</td><td> $1 . 0 2 8 \pm 0 . 0 0 7$ </td><td> $1 8 . 9 ^ { \dagger } \pm 2 . 0$ </td><td> $2 . 6 0 ^ { \dagger } \pm 0 . 0 6$ </td></tr></table>

## A.9 DETAILS ON THE EVALUATION OVER MULTIPLE TIME STEPS

This subsection provides the details of the evaluation over multiple time steps in Sec. 5.4 and examines whether the divergence of the zonotope enclosure is caused by the order reduction of the zonotopes. Starting from an initial input set $\mathcal { X } _ { 0 } = [ \underline { { x } } _ { 0 } , \overline { { x } } _ { 0 } ]$ ], we enclose the reachable set of the next time step using Prop. 2:

$$
\mathcal { X } _ { t + 1 } = \left[ \underline { { x } } _ { t + 1 } , \overline { { x } } _ { t + 1 } \right] = \big [ d _ { \Phi } ( \underline { { x } } _ { t } , \overline { { x } } _ { t } ) , d _ { \Phi } ( \overline { { x } } _ { t } , \underline { { x } } _ { t } ) \big ] ,\tag{19}
$$

where the input of the system is constant. As baseline, we propagate a zonotope through an unconstrained neural network that is trained on the same data: the output zonotope of one time step is used as the input set of the next time step without enclosing it by an interval, and its order is reduced to 20 after every time step using the method of Girard (2005). This evaluation therefore compares two combinations of neural network architecture and verification method; the comparison of verification methods on the same neural network is given in Tab. 2.

For each of the two numbers of layers $\kappa \in \{ 3 , 8 \}$ , we train three neural networks (seeds) and start from 20 initial input sets with a radius of 0.2 per neural network, which yields 60 runs per number of layers, each with at most 40 time steps and 2000 sampled trajectories. In every time step, we check that all samples are contained in all enclosures up to a tolerance of $1 0 ^ { - 6 }$ plus $\mathrm { i 0 ^ { - 3 } }$ times the largest edge length of the sampled reachable set; this check did not fail in any run. A run ends when the size of the sampled reachable set is below $2 \cdot 1 0 ^ { - 5 }$ , when the zonotope propagation fails or returns non-finite values, or after 40 time steps. Fig. 3 shows the median of the enclosure sizes over all runs in every time step. A run whose zonotope propagation failed or returned non-finite values is counted with an infinite size in all later time steps, so that the median remains finite as long as fewer than half of the runs have ended in this way; a run that ended because the sampled reachable set contracted is not counted in later time steps.

At three layers, all zonotope enclosures remain finite, and the tightness of the zonotope enclosure increases from 1.3 to 4.3 within 33 time steps with a maximum of 6.9. At eight layers, the median size of the zonotope enclosure exceeds $1 0 ^ { 6 }$ after eleven time steps and reaches $\overline { { 2 . 9 } } \cdot 1 \overline { { 0 } } ^ { 3 7 }$ after 30 time steps; in 26 of the 60 runs, the zonotope propagation fails after the size of the zonotope enclosure has exceeded $8 \cdot 1 0 ^ { 3 7 }$ or become non-finite, 41 runs exceed a size of $1 0 ^ { 3 }$ , and 19 runs stay below a size of 10. Within the evaluated time steps, the sampled reachable set contracts by more than four orders of magnitude.

To examine whether this divergence is caused by the order reduction, we repeat the evaluation for the baselines with eight layers of the Tanks and the Pendulum system with five initial input sets with the radius 0.2 per seed, 800 samples per input set, and the zonotope orders 10, 20, 50, 100, and 200. Increasing the order from 20 to 200 reduces the size of the zonotope enclosure after 20 time steps by about one order of magnitude, with little change between the orders 100 and 200. However, the median of the first time step at which the zonotope enclosure is more than 100 times larger than the sampled reachable set does not change for either system. These results only hold for the trained neural networks; enclosing the states of the physical system would additionally require taking the error of the model into account.