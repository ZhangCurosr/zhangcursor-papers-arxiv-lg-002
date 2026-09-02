# The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow

Raphaël Berthier

Sorbonne Université, Inria, Centre Inria de Sorbonne Université, Paris, France

September 2, 2026

## Abstract

The central flow of Cohen et al. (2025) is an empirically accurate continuous-time model of gradient descent at the edge of stability in deep learning, However, its derivation is heuristic. We propose a perturbative regime in which the central flow is the limit of gradient descent: we assume that the loss decomposes as $f = g + \varepsilon h ;$ in the limit $\varepsilon  0$ , the dynamics of gradient descent with learning rate η converge to the gradient flow of h constrained to the minimizers of g of sharpness at most $2 / \eta$ Our approach is formal rather than rigorous; it treats gradient descent as a singularly perturbed dynamical system in ε. Three timescales emerge: a fast timescale of oscillations along the sharpest direction, an intermediate timescale of the self-stabilization mechanism, and a slow timescale of the dynamics along the minimizers of g—the central flow. Using the method of multiple scales, a classical formal method from singular perturbation theory, we derive the expansion of the dynamics in ε: the central flow emerges as the leading-order term in the expansion, while the self-stabilization mechanism appears in the next-order term. We study this mechanism beyond previous analyses: with a single eigenvalue at the edge of stability, we compute the slow drift of the energy of the fluctuations; with several eigenvalues at the edge of stability, we derive the self-stabilization system and explain why fluctuations persist.

## Contents

1 Introduction 2   
2 Setting 4   
3 Convergence to the central flow 5   
4 Minimal example 8   
4.1 Introduction and experiments 8   
4.2 Method of multiple scales 9   
4.3 Energy variations in self-stabilization 15   
5 Valleys of codimension one 16   
6 Valleys of arbitrary codimension 18   
7 Related work 20   
8 Conclusion 21   
References 22

## 1 Introduction

Context. Gradient descent in deep learning still lacks theoretical understanding. Recent works have pointed out a surprising phenomenology. A large part of the training occurs at the so-called edge of stability [15, 16, 9]. In this regime, the sharpness of the loss—the maximal eigenvalue of the Hessian—fluctuates around the value $2 / \eta$ , where η is the learning rate. Above this $2 / \eta$ threshold, gradient descent on the local quadratic approximation of the loss develops diverging oscillations along the sharpest direction of the loss; it is thus remarkable that gradient descent on the loss itself does not diverge when fluctuations increase the sharpness above $2 / \eta$ . This is due to a so-called self-stabilization phenomenon [10]: the oscillations along the sharpest direction of the loss induce an efect in the local cubic approximation of the loss that pushes back the dynamics in the direction of sharpness reduction.

Overall, the resulting iterates form a rough path, both due to the oscillations along the sharpest direction and the fluctuations of the self-stabilization mechanism. This indicates that the optimization path cannot be approximated by a smooth continuous-time flow such as the gradient flow of the loss. This is a critical problem as the gradient flow is largely used to model gradient descent in deep learning.

Instead, one can expect a continuous-time approximation for the averaged trajectory, where the averaging is performed over the oscillations along the sharpest direction, and potentially over the fluctuations of the self-stabilization mechanism. Averaging over both, Cohen et al. [8] proposed the central flow: the gradient flow of the loss constrained to the stable region where the sharpness is below $2 / \eta$ . Through extensive numerical experiments, they have shown that the central flow is an excellent approximation of the averaged trajectory across a wide range of datasets and architectures.

Problem. However, the derivation of the central flow in [8] is heuristic. Several approximation steps are left informal, for instance using a local cubic approximation of the loss, or the meaning of the “local time-averaging” operator E. This raises the following question: is there a regime in which the central flow is the limit of gradient descent?

The regime $\eta  0$ of vanishing learning rate, used to derive the gradient flow, is not appropriate to answer this question. In this regime, the sharpness is always below $2 / \eta ,$ and thus edge of stability never occurs.

Contributions. In this work, we propose a perturbative regime—the sharp valley assumption—from which we derive the self-stabilization mechanism and the central flow. Mathematically, we assume that we run gradient descent with learning rate η on a loss function $f ( z ) = g ( z ) + \varepsilon h ( z )$ . The function g defines a valley: it has several minimizers and we denote $V = \{ z : g ( z ) = \operatorname* { m i n } g \}$ the valley of its minimizers. Further, $\varepsilon \ll 1$ is a small parameter and h is a perturbation that induces the dynamics along the valley.

Equivalently, gradient descent with learning rate $\varepsilon \eta$ on the loss function $\varepsilon ^ { - 1 } g + h$ gives the same sequence of iterates. The parameter $\varepsilon$ can thus be interpreted as the inverse sharpness of the valley.

The sharp valley picture has been present in several works [42, 15, 38, 22, 23, 34], as it is consistent with several empirical observations, such as the anti-alignment of successive iterates or the approximate low-rank structure of the Hessian (see Sec. 7 for a more complete discussion). A contribution of this work is to show that it is also consistent with the edge of stability, self-stabilization and the central flow.

Indeed, in the regime $\varepsilon  0$ , the central flow is the formal limit of gradient descent. More precisely, in this regime, gradient descent is a singularly perturbed dynamical system in which three timescales emerge: a fast timescale $t _ { 1 } = k$ , the iteration number $k ,$ which is the typical timescale of the oscillations along the sharpest direction; an intermediate timescale $t _ { 2 } = \varepsilon ^ { 1 / 2 } k$ , which is the typical timescale of the self-stabilization mechanism; and a slow timescale $t _ { 3 } ~ = ~ \varepsilon k$ , which is the typical timescale of the central flow—the evolution along the valley.

We then use the method of multiple scales [14] to derive an approximation of gradient descent that decouples the three timescales $t _ { 1 } , \ t _ { 2 }$ , and $t _ { 3 }$ . The leading term in the approximation is the central flow, which evolves on the slow timescale $t _ { 3 }$ . It consists in the gradient flow of $h$ in the valley, constrained to the stable region where the sharpness of $g$ is below $2 / \eta$ . The next-order term contains the oscillations along the sharpest direction as a function of $t _ { 1 }$ and the self-stabilization fluctuations as a function of $t _ { 2 }$ . The method of multiple scales is a formal method in the sense of asymptotic analysis: it produces the expansion by matching orders, without proving convergence or bounding the remainder. Our results should thus be interpreted as formal derivations, i.e., conjectures with strong supporting evidence.

To summarize, this work provides some justification of the central flow approximation by proposing a regime in which the approximation is tight and by providing an analytical tool—the method of multiple scales—to derive the approximation.

Further consequences. Beyond the clarification of previous works, the framework of this paper enables some new discussions.

We study the self-stabilization mechanism in more depth: we derive the self-stabilization system with multiple eigenvalues on the edge of stability (left open in [10]), and discuss the evolution of the self-stabilization fluctuations on the slow timescale $t _ { 3 }$ . With a single eigenvalue on the edge of stability, we compute the variations of the energy of the fluctuations; with multiple eigenvalues on the edge of stability, we explain the persistence of fluctuations.

Finally, we compare the central flow with competing continuous-time approximations of gradient descent [30, 27, 7]. We obtain that the central flow is a more coarse-grained approximation than the competing approximations, as it depends only on $t _ { 3 }$ , while the other approximations depend on $t _ { 2 }$ and $t _ { 3 }$

Outline. This paper is organized as follows. In Sec. 2, we introduce our mathematical setting, including the sharp valley assumption. The strongest result of this paper is provided in Sec. $6 ;$ however, for the sake of clarity and to enable additional discussions, previous sections provide simpler versions of the result. In Sec. 3, we present only the results on the convergence to the central flow. In Sec. 4, we include the discussion of the next-order term in the convergence, but in a minimal example. The simplicity in this example allows to study potential failures of the central flow approximation and the slow evolution of the self-stabilization loop. Further, in Sec. 5, we discuss the case of valleys of codimension one, which is more general than the minimal example but still simpler than the general case. In Sec. 6, we discuss the general case of valleys of arbitrary codimension. Finally, in Sec. 7, we discuss the connections to further related work, including the comparison with competing approxiamtions.

## 2 Setting

Let $f ( z ) , z \in Z$ , be a loss function, where $Z$ is a finite-dimensional vector space. We study the dynamics of gradient descent on $f$ with learning rate $\eta > 0$

$$
z ( k + 1 ) = z ( k ) - \eta \nabla f ( z ( k ) ) .
$$

We assume that the loss function $f$ can be decomposed as $f ( z ) = g ( z ) + \varepsilon h ( z )$ , where g defines a valley and $\varepsilon \ll 1$ is a small parameter. We denote $V = \{ z \in Z : g ( z ) = \operatorname* { m i n } g \}$ the valley of minimizers of $g .$

To simplify the analyses, we assume that the valley defined by $g$ is a linear space. More precisely, we assume there exists an orthogonal decomposition $Z = X \oplus ^ { \perp } Y$ , in which we decompose $z = ( x , y )$ , such that the valley defined by g is

$$
V = \{ ( 0 , y ) , y \in Y \} .
$$

In a more complex setting where V would be a manifold, the decomposition $z = ( x , y )$ would represent a local set of coordinates to parametrize the manifold V of minimizers of g.

Further, for all $z ,$ we denote $S ( z ) = \nabla _ { x } ^ { 2 } g ( z )$ the Hessian of $g$ in the sharp directions— those orthogonal to the valley. We assume that for all $( 0 , y ) \in V , S ( 0 , y ) \succ 0 :$ there is some curvature of $g$ in the sharp directions along the valley. Finally, we denote

$$
V _ { \mathrm { { S } } } = \{ z \in V : \eta S ( z ) \preccurlyeq 2 \mathrm { { I d } } _ { X } \}
$$

the stable valley of $^ { g , }$ in which the sharpness is below $2 / \eta$

Notations. When $\boldsymbol { y } \in \mathbb { R } ^ { d }$ is a vector, we denote $y [ 1 ] , \ldots , y [ d ]$ its components. Let A and B be Euclidean spaces. When $L : A  B$ is a linear operator, we denote $L ^ { * } : B  A$ its adjoint. We denote Sym A the set of symmetric linear operators on $A _ { i }$ , and $\operatorname { S y m } _ { + } A$ the set of symmetric positive semidefinite linear operators on $A .$ . We denote $\operatorname { I d } _ { A }$ the identity operator on A. When $C$ is a linear subspace of $A .$ , we denote $C ^ { \perp }$ its orthogonal complement, $P _ { C } : A  C$ the orthogonal projection onto $C ,$ and $\iota _ { C } : C \to A$ the canonical inclusion of C into A. Note that $\iota _ { C } = P _ { C } ^ { * }$

## 3 Convergence to the central flow

In this section, we state our convergence results to the central flow. More complete results, including next-order terms, are provided in the following sections.

As the gradient of $f = g + \varepsilon h$ in the valley is of order ε, the typical timescale of the dynamics of the iterates $z ( k ) = ( x ( k ) , y ( k ) )$ along the valley is $t _ { 3 } = \varepsilon k$ . Abusing notations, let us denote

$$
z ( t _ { 3 } ) = z \left( k = \lfloor t _ { 3 } / \varepsilon \rfloor \right) , \qquad x ( t _ { 3 } ) = x \left( k = \lfloor t _ { 3 } / \varepsilon \rfloor \right) , \qquad y ( t _ { 3 } ) = y \left( k = \lfloor t _ { 3 } / \varepsilon \rfloor \right) ,
$$

the time-rescaled iterates, that now depend on a continuous time parameter $t _ { 3 }$ . Note that the resulting function $z ( t _ { 3 } )$ depends on ε both through the rescaling of time and through the loss function. We now state convergence results to the central flow, obtained by the method of multiple scales.

Valleys of codimension one. For the sake of clarity, we start with the case where the valley is of codimension 1, i.e., the dimension of X is 1. Note that, in this case, $S ( z )$ is a scalar.

Result 3.1. Assume that X is of dimension 1. Further, assume that as $\varepsilon \ \to \ 0$ , the rescaled iterates $\boldsymbol { z } ( t _ { 3 } ) = ( \boldsymbol { x } ( t _ { 3 } ) , \boldsymbol { y } ( t _ { 3 } ) )$ converge to some function $z _ { 0 } ( t _ { 3 } ) = ( x _ { 0 } ( t _ { 3 } ) , y _ { 0 } ( t _ { 3 } ) )$ Moreover, assume that $z _ { \mathrm { 0 } }$ remains in the stable valley $V _ { \mathrm { S } } , i . e . , x _ { 0 } = 0$ and $\eta S ( 0 , y _ { 0 } ) \leqslant 2$

Then z<sub>0</sub> follows the gradient flow of h constrained to the stable valley $V _ { \mathrm { S } } , \ i . e .$ , there exists $\sigma ^ { 2 } = \sigma ^ { 2 } ( t _ { 3 } ) \geqslant 0$ such that

$$
\partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \nabla _ { y } h ( 0 , y _ { 0 } ) + \frac { \sigma ^ { 2 } } { 2 } \nabla _ { y } S ( 0 , y _ { 0 } ) \right] ,
$$

where for all $t _ { 3 } , \sigma ^ { 2 } ( t _ { 3 } ) = 0 \ o r \ \eta S ( 0 , y _ { 0 } ( t _ { 3 } ) ) = 2 .$

Again, the method of multiple scales is formal, thus the above result should not be interpreted as a mathematical theorem but rather as a conjecture with strong supporting evidence. All the results of this paper share this status. An introduction to the method of multiple scales is provided in Sec. 4.2.

The function $\sigma ^ { 2 }$ should be interpreted as the Lagrange multiplier associated with the constraint $\eta S ( 0 , y _ { 0 } ) \leqslant 2 \mathrm { : }$ it satisfies dual feasibility $\left( \sigma ^ { 2 } \geqslant 0 \right)$ and complementarity slackness with the constraint $( \sigma ^ { 2 } = 0$ or $\eta S ( 0 , y _ { 0 } ) = 2 )$ .

We illustrate the result in Fig. 1. We observe the edge of stability phenomenon, the self-stabilization fluctuations, and the convergence to the central flow $( x _ { 0 } , y _ { 0 } )$ as $\varepsilon \to 0$ . A more pedagogical example is provided in the next section.

We now discuss the assumptions of Result 3.1. The statement $x _ { 0 } = 0$ is not strictly needed and could instead be obtained as a consequence of the method of multiple scales. However, treating $x _ { 0 }$ in the method derivations results in more complicated computations, thus we prefer making the reasonable assumption that the limiting dynamics are in the valley V. Moreover, note that we could initialize x at a non-zero value, which would result in a relaxation of $x _ { 0 }$ to $0$ on the timescale $t _ { 1 }$ . This early part of the dynamics is not interesting for our study of the edge of stabili $\mathrm { t y } ,$ thus we do not include it here.

![](images/d79f691b024bcaac626eef54153fdb08f1c9052da0b992656c11a96eeb0884af.jpg)

![](images/b5c2140c31e36048099450aa80ebc38f2d0266ba05af5b0129cad0a39c20dc22.jpg)  
Figure 1: Iterates of gradient descent with learning rate $\eta = 0 . 3$ on $f = g + \varepsilon h$ , with $\varepsilon = 0 . 0 1 , 0 . 0 0 1 , 0 . 0 0 0 1$ . In this example, we take $Z = \mathbb { R } ^ { 3 } , X = \mathbb { R }$ and $Y = \mathbb { R } ^ { 2 }$ , thus the valley is of dimension 2 and codimension 1. Further, we take $\begin{array} { r } { g ( x , y ) = \frac { 1 } { 2 } ( \| y \| + 3 e ^ { x / 2 } ) x ^ { 2 } } \end{array}$ and $h ( x , y ) = - x - y [ 1 ]$ (where $y [ 1 ]$ denotes the first component of $y )$ . Note that, as required by Sec. 2, the valley of minimizers of g is $V = \{ ( 0 , y ) , y \in \mathbb { R } ^ { 2 } \}$ . Moreover, $S ( 0 , y ) = \| y \| + 3$ and thus $V _ { \mathrm { { S } } }$ is a closed disk: $V _ { \mathrm { { S } } } = \{ ( 0 , y ) , \| y \| \leqslant 2 / \eta - 3 \}$

On the contrary, we need the assumption $\eta S ( 0 , y _ { 0 } ) \leqslant 2$ to derive the result; we do not know how to obtain it as a by-product of the method. In fact, it is possible to build artificial examples where this condition breaks and the central flow prediction fails; this is discussed below in Remark 4.3.

Valleys of arbitrary codimension. We now turn to the more general case where the dimension of X is arbitrary. Recall that we denote $S ( z ) = \nabla _ { x } ^ { 2 } g ( z ) \in \operatorname { S y m } X$ . Further, we denote

$$
\begin{array} { c } { { D _ { y } S ( x , y ) : Y \to \operatorname { S y m } X } } \\ { { \delta y \mapsto D _ { y } S ( x , y ) [ \delta y ] } } \end{array}
$$

the diferential of S in the y variable, and consequently, $D _ { y } S ( x , y ) ^ { * } : \operatorname { S y m } X \to Y$ the adjoint of $D _ { y } S ( x , y )$ . Finally, we denote $\langle . , . \rangle$ the canonical inner product on Sym X.

Result 3.2. We no longer assume that X is of dimension 1. Further, assume that as $\varepsilon \ \to \ 0$ , the rescaled iterates $z ( t _ { 3 } ) = ( x ( t _ { 3 } ) , y ( t _ { 3 } ) )$ converge to some function $z _ { 0 } ( t _ { 3 } ) ~ =$ $( x _ { 0 } ( t _ { 3 } ) , y _ { 0 } ( t _ { 3 } ) )$ . Moreover, assume that z remains in the stable valley $V _ { \mathrm { S } } , i . e . , x _ { 0 } = 0$ and $\eta S ( 0 , y _ { 0 } ) \precsim 2 \mathrm { I d } _ { X }$

Then $z _ { \mathrm { 0 } }$ follows the gradient flow of h constrained to the stable valley $V _ { \mathrm { S } } , \ i . e .$ , there exists $\Sigma = \Sigma ( t _ { 3 } ) \in \mathrm { S y m } _ { + } X$ such that

$$
\partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \nabla _ { y } h ( 0 , y _ { 0 } ) + \frac { 1 } { 2 } D _ { y } S ( 0 , y _ { 0 } ) ^ { * } [ \Sigma ] \right] ,
$$

where for all $t _ { 3 } , \langle \Sigma ( t _ { 3 } ) , 2 \operatorname { I d } _ { X } - \eta S ( 0 , y _ { 0 } ( t _ { 3 } ) ) \rangle = 0 .$

Note that Result 3.2 is a generalization of Result 3.1. The function Σ should be interpreted as the Lagrange multiplier associated with the constraint $\eta S ( 0 , y _ { 0 } ) \precsim 2 \mathrm { I d } _ { X } \colon$ it satisfies dual feasibility $( \Sigma \in \operatorname { S y m } _ { + } X )$ and complementarity slackness with the constraint (⟨Σ, 2 Id $x - \eta S ( 0 , y _ { 0 } ) \rangle = 0 )$

We illustrate the result in Fig. 2.

![](images/3b280ff014b19119694a0ad81d16cb27d7526bc96b8484aa504483245771b97f.jpg)

![](images/14db106dba9ef5269f26b2f4d8a7169bd28909b9ce5246d28c4477ba07e8e168.jpg)

![](images/bc2c9250eccb53fe99bc72924aa022dc474cecf7c860bdcf07598bfcd16aab92.jpg)

![](images/ac6e180dbd24dd306c488143f351ab3fa9af20825224a0b95afc59807eeeb416.jpg)  
Figure 2: Iterates of gradient descent with learning rate $\eta = 0 . 3$ on $f = g + \varepsilon h$ , with $\varepsilon = 0 . 0 1$ . In this example, we take $Z = \mathbb { R } ^ { 5 }$ ， $X = \mathbb { R } ^ { 2 }$ and $Y = \mathbb { R } ^ { 3 }$ , thus the valley is of dimension 3 and codimension 2. Further, we take $\begin{array} { r } { g ( x , y ) = \frac { 1 } { 2 } x ^ { \top } H [ y ] x + \frac { 1 } { 2 } ( x [ 1 ] ^ { 3 } + x [ 2 ] ^ { 3 } ) } \end{array}$ with $H [ y ] = \left( \begin{array} { c c } { { y [ 1 ] } } & { { \frac { y [ 3 ] } { \sqrt { 2 } } } } \\ { { \frac { y [ 3 ] } { \sqrt { 2 } } } } & { { y [ 2 ] } } \end{array} \right)$ and $h ( x , y ) = - x [ 1 ] - 2 y [ 1 ] - y [ 2 ] - y [ 3 ]$ . Note that the assumptions of Sec. 2 are satisfied in the region where $H [ y ] + \mathrm { d i a g } ( x [ 1 ] , x [ 2 ] ) \succ 0$ (where the iterates remain). Moreover, $S ( 0 , y ) = H [ y ]$ and $V _ { \mathrm { { S } } } = \{ ( 0 , y ) , \eta H [ y ] \prec 2 \operatorname { I d } _ { 2 } \}$

The method of multiple scales also provides information on the oscillations and the self-stabilization fluctuations through the next-order term $z _ { 1 } = ( x _ { 1 } , y _ { 1 } )$ in the expansion as $\varepsilon \to 0$ . These complements are provided in Sec. 5 for valleys of codimension one and in Sec. 6 for the general case. These aspects require discussing in more detail how the method of multiple scales works. For pedagogical purposes, we start doing so on a minimal example in the next section.

## 4 Minimal example

## 4.1 Introduction and experiments

Consider the case $Z = \mathbb { R } ^ { 2 } , X = Y = \mathbb { R }$ , thus $z = ( x , y ) \in \mathbb { R } ^ { 2 }$ . Further, we consider the following choices determining the loss function $f = g + \varepsilon h ;$

$$
g ( x , y ) = \frac { 1 } { 2 } ( y + \zeta x ) x ^ { 2 } ,
$$

$$
h ( x , y ) = - x - y .\tag{4.1}
$$

Here, $\zeta \in \mathbb { R }$ is a parameter. This function g does not rigorously satisfy the valley structure required in Sec. 2. However, this is benign: below, we consider the function $g$ only in the region where $y + \zeta x > 0$ . In this region, the minimum of $g$ is 0 and its valley of minimizers is $V = \{ ( 0 , y ) , y > 0 \}$ . In the valley, h decreases as y increases, thus the gradient descent dynamics move along the valley in the direction of increasing y. However, the sharpness of $g$ in the valley is $S ( 0 , y ) = \partial _ { x } ^ { 2 } g ( 0 , y ) = y ,$ thus there is progressive sharpening, until the dynamics hit the edge of stability at $y = 2 / \eta$ . There are then self-stabilization fluctuations around the value $y = 2 / \eta$ . These phenomena are illustrated in Fig. 3.

![](images/41e2d7006339c56c36a0f664bc0de25b3f5ca1f2cb51c721fedd891cc044f5c3.jpg)

![](images/979f243def23939bd606763baa49501e5e6809828dfc714ccf176a9b63a1be5a.jpg)  
Figure 3: For $\varepsilon = 0 . 0 1$ ， $\zeta = 1$ and $g , h$ as provided in the minimal example of Eqs. (4.1), we plot the loss function $f = g + \varepsilon h$ (left) and the iterates of gradient descent on $f$ with learning rate $\eta = 0 . 3$ (right). The iterates stabilize on the edge of stability $y = 2 / \eta$ after some self-stabilization fluctuations.

The reader might object that the above example is not the simplest possible one with the same properties. For instance, one could consider $\begin{array} { r } { g ( x , y ) = \frac { 1 } { 2 } y x ^ { 2 } } \end{array}$ and $h ( x , y ) = - y$ However, it turns out that these choices would not lead to the exact same phenomenology, and the ones we make lead to a more faithful reproduction of the phenomenology of deep learning optimization. In short, the $x ^ { 3 }$ term in $g$ reproduces the damping of the selfstabilization loop, and the $- x$ term in h breaks the metastability in x. These aspects are discussed in Sec. 4.3 and Remark 4.3 respectively.

In Figure 4, we plot the iterates of gradient descent as a function of $t _ { 3 }$ , for various values of ε. We observe that the iterates converge to the central flow $( x _ { 0 } , y _ { 0 } ) { \mathrm { ~ a s ~ } } \varepsilon \to 0$ that is, to the gradient flow of h constrained to the stable valley $V _ { \mathrm { { S } } } = \{ ( 0 , y ) , y \leqslant 2 / \eta \}$

![](images/7e02213718231d4511b11e35bc3b479f3dbece92bdc3cbe15e23e16b6e00c622.jpg)

![](images/6946c9e485dd2499be7c50ab33362dbed8d4488cab1a831642f55ac990139b4e.jpg)  
Figure 4: Iterates of gradient descent on the minimal example of Eqs. (4.1) with learning rate $\eta = 0 . 3 , \zeta = 1$ and $\varepsilon = 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 }$

At the edge of stability, the oscillations of $x$ and the self-stabilization fluctuations appear in the next-order term of the approximation. $\mathrm { A s }$ will be confirmed by the method of multiple scales below, this next order term is of order $\varepsilon ^ { 1 / 2 }$ . In Figure $5 ,$ we plot the rescaled iterates $\varepsilon ^ { - 1 / 2 } ( x - x _ { 0 } , y - y _ { 0 } )$ as functions of the slow timescale $t _ { 3 } = \varepsilon k$ and the intermediate timescale $t _ { 2 } = \varepsilon ^ { 1 / 2 } k$ . In the limit $\varepsilon \to 0$ , we observe that the self-stabilization loop and its slow damping decouple on the timescales $t _ { 2 }$ and $t _ { 3 }$ . While the typical time of a self-stabilization fluctuation is $t _ { 2 }$ , the typical time of the damping of the self-stabilization loop is $t _ { 3 }$

## 4.2 Method of multiple scales

The goal of this section is to confirm and refine the empirical observations above using the method of multiple scales. It is a perturbation method: in our case, the perturbation parameter is $\varepsilon \ll 1$ , and it seeks an expansion of the form

$$
\begin{array} { l } { { x = x _ { 0 } + \varepsilon ^ { 1 / 2 } x _ { 1 } + \varepsilon x _ { 2 } + \dots , } } \\ { { \nonumber } } \\ { { y = y _ { 0 } + \varepsilon ^ { 1 / 2 } y _ { 1 } + \varepsilon y _ { 2 } + \dots . } } \end{array}\tag{4.2}
$$

The goal is to obtain expressions or evolution equations for the leading terms of this expansion. Here, the challenge is that the terms depend simultaneously on various timescales $t _ { 1 } = k , t _ { 2 } = \varepsilon ^ { 1 / 2 } k$ , and $t _ { 3 } = \varepsilon k$ , that depend themselves on ε. This is a typical situation where the method of multiple scales is useful.

![](images/7eb10f0bc810c74cc23ac9940d931d2826b4a47c91440310f63a2888e4c6c6c8.jpg)  
Figure 5: Rescaled iterates of gradient descent on the minimal example of Eqs. (4.1) with $\eta = 0 . 3 , \zeta = 1$ and $\varepsilon = 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 }$ . We denote $k _ { \mathrm { E o S } }$ the first iteration number for which $y > 2 / \eta \colon$ it encodes the beginning of the edge of stability. We denote $t _ { \mathrm { 2 , E o S } } = \varepsilon ^ { 1 / 2 } k _ { \mathrm { E o S } }$ and $t _ { 3 , \mathrm { E o S } } = \varepsilon k _ { \mathrm { E o S } }$ the corresponding rescaled times, and we start the plots at these times. We do not plot the trajectory with $\varepsilon = 1 0 ^ { - 5 }$ in the upper plots for readability.

Let us first give an overview of the method. The method postulates that the timescales $t _ { 1 } , t _ { 2 }$ , and $t _ { 3 }$ can be decoupled by treating them as independent parameters. In our case, $t _ { 1 }$ is a discrete parameter while $t _ { 2 }$ and $t _ { 3 }$ are continuous parameters. It then treats all functions of k as functions of the three independent timescales: $z = z ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) , \ x \ =$ $x ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) , y = y ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) , x _ { 0 } = x _ { 0 } ( t _ { 1 } , t _ { 2 } , t _ { 3 } )$ , etc. The diference equation $\Delta _ { k } z = z ( k +$ $1 ) - z ( k ) = - \eta \nabla f ( z )$ in $z ~ = ~ z ( k )$ is transformed into a partial diference-diferential equation in $z = z ( t _ { 1 } , t _ { 2 } , t _ { 3 } )$ . By identifying the diferent terms in the expansion of this equation, we infer properties for the terms $x _ { 0 } = x _ { 0 } ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) , y _ { 0 } = y _ { 0 } ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) , x _ { 1 } =$ $x _ { 1 } ( t _ { 1 } , t _ { 2 } , t _ { 3 } )$ , etc. The final approximation is provided by using the expansion of Eq. (4.2) and substituting the actual timescales $t _ { 1 } = k , t _ { 2 } = \varepsilon ^ { 1 / 2 } k$ , and $t _ { 3 } = \varepsilon k$

We obtain the following result for our minimal example.

Result 4.1. Assume that $( x _ { 0 } , y _ { 0 } )$ remains in the stable valley, $i . e . , x _ { 0 } = 0$ and ηy<sub>0</sub> $\leqslant 2$ Then $y _ { 0 } = y _ { 0 } ( t _ { 3 } ) , i . e . , y _ { 0 }$ does not depend on $t _ { 1 } \ o r \ t _ { 2 } ,$ and $y _ { 1 } = y _ { 1 } ( t _ { 2 } , t _ { 3 } ) , i . e . , y _ { 1 }$ does not depend on $t _ { 1 }$ . Further, we have the following:

(i) If $\eta y _ { 0 } < 2$ (relative interior of the stable valley), then $\partial _ { t _ { 3 } } y _ { 0 } = - \eta \partial _ { y } h ( x _ { 0 } , y _ { 0 } ) = \eta$

(ii) If $\eta y _ { 0 } = 2$ (edge of stability), then $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 }$ for some $a _ { 1 } = a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ . Further, we have the self-stabilization system

$$
\begin{array} { l } { { \partial _ { t _ { 2 } } a _ { 1 } = \eta a _ { 1 } y _ { 1 } , } } \\ { { \partial _ { t _ { 2 } } y _ { 1 } = \eta \left( 1 - \displaystyle \frac { a _ { 1 } ^ { 2 } } { 2 } \right) . } } \end{array}\tag{4.3}
$$

This result confirms that $( x _ { 0 } , y _ { 0 } )$ follows the central flow prediction. It moves in the stable valley $V _ { \mathrm { { S } } } = \{ ( 0 , y ) : \eta y \le 2 \}$ , following the constrained gradient flow of $h ,$ until it hits the constraint $\eta y _ { 0 } \leqslant 2$ . It then stays on the edge of stability threshold. This is consistent with the experiment of Fig. 4.

This result also gives information about the next-order term $( x _ { 1 } , y _ { 1 } )$ at the edge of stability. While $y _ { 1 }$ does not depend on $t _ { 1 } , \ x _ { 1 }$ oscillates in $t _ { 1 }$ , with amplitude $a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ Again, this is consistent with the experiments of Figs. 4 and 5. Moreover the joint evolution of $a _ { 1 }$ and $y _ { 1 }$ in $t _ { 2 }$ is provided by the system of ordinary diferential equations (ODEs) (4.3). This corresponds to the self-stabilization equations derived in [10].

As noted in [10], these self-stabilization dynamics preserve the energy

$$
E = \frac { y _ { 1 } ^ { 2 } } { 2 } + \frac { a _ { 1 } ^ { 2 } } { 4 } - \frac { 1 } { 2 } \log a _ { 1 } ^ { 2 } ,
$$

resulting in $( a _ { 1 } , y _ { 1 } )$ being periodic in $t _ { 2 }$ . As a consequence, $E = E ( t _ { 3 } )$ , i.e., E evolves only on the slow timescale $t _ { 3 }$ . This is consistent with the experiment of Fig. 5. The evolution of $E$ in $t _ { 3 }$ is discussed in Sec. 4.3.

We now turn to the derivation of Result 4.1. The situation in which we apply the method of multiple scales is rather sophisticated, due to the presence of three timescales (two is more common) and the presence of a discrete timescale $t _ { 1 }$ . For more pedagogical illustrations of the method, we recommend [14].

Derivation of Result 4.1. As written above, the method of multiple scales operates by identifying the several scales in the gradient descent iteration $\Delta _ { k } z = - \eta \nabla f ( z )$ . We thus expand both sides as functions of ε. For the right-hand side, we have

$$
\begin{array} { r l } & { - \eta \nabla f ( z ) = - \eta \nabla g ( z ) - \eta \varepsilon \nabla h ( z ) } \\ & { \qquad = - \eta \left[ \nabla g \left( z _ { 0 } + \varepsilon ^ { 1 / 2 } z _ { 1 } + \varepsilon z _ { 2 } + o ( \varepsilon ) \right) + \varepsilon \nabla h \left( z _ { 0 } + o ( 1 ) \right) \right] } \\ & { \qquad = - \eta \left[ \nabla g ( z _ { 0 } ) + \nabla ^ { 2 } g ( z _ { 0 } ) \left( \varepsilon ^ { 1 / 2 } z _ { 1 } + \varepsilon z _ { 2 } \right) + \frac { 1 } { 2 } \nabla ^ { 3 } g ( z _ { 0 } ) \left[ \varepsilon ^ { 1 / 2 } z _ { 1 } , \varepsilon ^ { 1 / 2 } z _ { 1 } \right] \right. } \\ & { \qquad \left. \qquad + \varepsilon \nabla h ( z _ { 0 } ) + o ( \varepsilon ) \right] } \\ & { \qquad = - \eta \nabla g ( z _ { 0 } ) - \varepsilon ^ { 1 / 2 } \eta \nabla ^ { 2 } g ( z _ { 0 } ) z _ { 1 } - \varepsilon \eta \left[ \nabla ^ { 2 } g ( z _ { 0 } ) z _ { 2 } + \frac { 1 } { 2 } \nabla ^ { 3 } g ( z _ { 0 } ) [ z _ { 1 } , z _ { 1 } ] + \nabla h ( z _ { 0 } ) \right] } \\ & { \qquad \quad \left. + o ( \varepsilon ) . \qquad \right. \qquad ( 4 . 4 } \end{array}
$$

Using the expressions for g and h of Eqs. (4.1) and the assumption $x _ { 0 } = 0$ , we obtain

$$
\begin{array} { l } { \displaystyle - \eta \partial _ { x } f ( x , y ) = - \varepsilon ^ { 1 / 2 } \eta y _ { 0 } x _ { 1 } - \varepsilon \eta \left[ y _ { 0 } x _ { 2 } + \frac 3 2 \zeta x _ { 1 } ^ { 2 } + x _ { 1 } y _ { 1 } - 1 \right] + o ( \varepsilon ) , } \\ { \displaystyle - \eta \partial _ { y } f ( x , y ) = - \varepsilon \eta \left[ \frac { x _ { 1 } ^ { 2 } } 2 - 1 \right] + o ( \varepsilon ) . } \end{array}\tag{4.5}
$$

The expansion of the left-hand side $\Delta _ { k } z$ is more subtle because it requires to express the discrete diference operator $\Delta _ { k }$ applied to a function $z = z ( t _ { 1 } , t _ { 2 } , t _ { 3 } )$ . As $t _ { 1 } , \ t _ { 2 } .$ , and $t _ { 3 }$ depend on $k .$ , the expression of $\Delta _ { k }$ depends on the discrete diference operator $\Delta _ { t _ { 1 } }$ and the partial derivatives $\partial _ { t _ { 2 } }$ and $\partial _ { t _ { 3 } }$

$$
\begin{array} { r l } & { \Delta _ { k } z = z ( t _ { 1 } + 1 , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon ) - z ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) } \\ & { \qquad = z ( t _ { 1 } + 1 , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon ) - z ( t _ { 1 } , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon ) } \\ & { \qquad + z ( t _ { 1 } , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon ) - z ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) } \\ & { \qquad = ( \Delta _ { t _ { 1 } } z ) ( t _ { 1 } , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon ) + z ( t _ { 1 } , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon ) - z ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) } \end{array}\tag{4.6}
$$

We first decompose

$$
z ( t _ { 1 } , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon ) - z ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) = \varepsilon ^ { 1 / 2 } \partial _ { t _ { 2 } } z + \varepsilon \partial _ { t _ { 3 } } z + \frac { \varepsilon } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z + o ( \varepsilon ) .
$$

Note that we do not write the evaluations of the partial derivatives at $( t _ { 1 } , t _ { 2 } , t _ { 3 } )$ for readability. Using a similar expansion for $( \Delta _ { t _ { 1 } } z ) ( t _ { 1 } , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon )$ , we obtain

$$
\begin{array} { c } { { \Delta _ { k } z = \Delta _ { t _ { 1 } } z + \varepsilon ^ { 1 / 2 } \partial _ { t _ { 2 } } z + \varepsilon \partial _ { t _ { 3 } } z + \displaystyle \frac { \varepsilon } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z + \varepsilon ^ { 1 / 2 } \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } z + \varepsilon \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z + \displaystyle \frac { \varepsilon } { 2 } \partial _ { t _ { 2 } } ^ { 2 } \Delta _ { t _ { 1 } } z + o ( \varepsilon ) } } \\ { { { } } } \\ { { = \Delta _ { t _ { 1 } } z + \varepsilon ^ { 1 / 2 } \left[ \partial _ { t _ { 2 } } z + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } z \right] + \varepsilon \left[ \partial _ { t _ { 3 } } z + \displaystyle \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z + \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z + \displaystyle \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } \Delta _ { t _ { 1 } } z \right] + o ( \varepsilon ) . } } \end{array}
$$

We combine the above decomposition with the expansion of $z$

$$
z = z _ { 0 } + \varepsilon ^ { 1 / 2 } z _ { 1 } + \varepsilon z _ { 2 } + o ( \varepsilon ) .
$$

We obtain

$$
\begin{array} { l } { \Delta _ { { k } } z = \Delta _ { { t } _ { 1 } } z _ { 0 } + \varepsilon ^ { 1 / 2 } \left[ \Delta _ { { t } _ { 1 } } z _ { 1 } + \partial _ { { t } _ { 2 } } z _ { 0 } + \partial _ { { t } _ { 2 } } \Delta _ { { t } _ { 1 } } z _ { 0 } \right] } \\ { \quad \quad \quad + \varepsilon \left[ \Delta _ { { t } _ { 1 } } z _ { 2 } + \partial _ { { t } _ { 2 } } z _ { 1 } + \partial _ { { t } _ { 2 } } \Delta _ { { t } _ { 1 } } z _ { 1 } + \partial _ { { t } _ { 3 } } z _ { 0 } + \displaystyle \frac { 1 } { 2 } \partial _ { { t } _ { 2 } } ^ { 2 } z _ { 0 } + \partial _ { { t } _ { 3 } } \Delta _ { { t } _ { 1 } } z _ { 0 } + \frac { 1 } { 2 } \partial _ { { t } _ { 2 } } ^ { 2 } \Delta _ { { t } _ { 1 } } z _ { 0 } \right] } \\ { \quad \quad \quad + o ( \varepsilon ) . } \end{array}\tag{4.7}
$$

Combining the expansions of Eqs. (4.5) and (4.7), we are ready to identify terms in the expansion of

$$
\Delta _ { k } x = - \eta \partial _ { x } f ( x , y ) ,\tag{4.8}
$$

$$
\Delta _ { k } y = - \eta \partial _ { y } f ( x , y ) .\tag{4.9}
$$

Order 1. Recall that, by assumption, $x _ { 0 } ~ = ~ 0 \quad$ . The x equation (4.8) does not give anything at order 1. The $y$ equation (4.9) gives $\Delta _ { t _ { 1 } } y _ { 0 } = 0$ , thus $y _ { 0 } = y _ { 0 } ( t _ { 2 } , t _ { 3 } )$ does not depend on $t _ { 1 }$ .

Order $\varepsilon ^ { 1 / 2 }$ . The x equation (4.8) gives

$$
\Delta _ { t _ { 1 } } x _ { 1 } = - \eta y _ { 0 } x _ { 1 } .
$$

Here, $y _ { 0 }$ should be interpreted as the sharpness of $g$ at $( x _ { 0 } , y _ { 0 } )$ ; this diference equation describes the linearized dynamics of x around $( x _ { 0 } , y _ { 0 } )$ . Its behavior depends on whether the edge of stability is reached. Fix $t _ { 2 }$ and $t _ { 3 }$

• If $\eta y _ { 0 } ( t _ { 2 } , t _ { 3 } ) < 2 , \mathrm { t h e n } x _ { 1 } = x _ { 1 } ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) \xrightarrow [ t _ { 1 } \to \infty ] { } 0 .$

• If $\eta y _ { 0 } ( t _ { 2 } , t _ { 3 } ) = 2$ , then $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ for some $a _ { 1 }$ that does not depend on $t _ { 1 }$ At the edge of stability, x<sub>1</sub> oscillates with an amplitude which is a function of $t _ { 2 }$ and $t _ { 3 }$ only.

We now turn to the y equation (4.9). It gives

$$
\Delta _ { t _ { 1 } } y _ { 1 } + \partial _ { t _ { 2 } } y _ { 0 } = 0 .
$$

Recall that $y _ { 0 }$ does not depend on $t _ { 1 }$ . As a consequence, $y _ { 1 }$ is an afine function of $t _ { 1 }$ with slope $- \partial _ { t _ { 2 } } y _ { 0 }$ . However, having $y _ { 1 }$ unbounded is not acceptable as it would imply that $\varepsilon ^ { 1 / 2 } y _ { 1 }$ eventually become of order 1, thus interfering with the leading order term y<sub>0</sub> in the expansion (4.2). In the method of multiple scales, such a term is called secular, and the method prescribes to build an approximation that forbids secular terms, to keep the reasoning consistent [14]. This prescription is the core mechanism of the method of multiple scales; we will use it repeatedly. Here, we can thus conclude that $\Delta _ { t _ { 1 } } y _ { 1 } = - \partial _ { t _ { 2 } } y _ { 0 } = 0$ . As a consequence, $y _ { 1 } = y _ { 1 } ( t _ { 2 } , t _ { 3 } )$ and $y _ { 0 } = y _ { 0 } ( t _ { 3 } )$

Order ε. The x equation (4.8) gives

$$
\Delta _ { t _ { 1 } } x _ { 2 } + \partial _ { t _ { 2 } } x _ { 1 } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } x _ { 1 } = - \eta \left[ y _ { 0 } x _ { 2 } + \frac { 3 } { 2 } \zeta x _ { 1 } ^ { 2 } + x _ { 1 } y _ { 1 } - 1 \right] .
$$

Here, we focus on understanding what happens at the edge of stability, i.e., when $\eta y _ { 0 } = 2$ We then know that $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ , and thus $\Delta _ { t _ { 1 } } x _ { 1 } = - 2 ( - 1 ) ^ { t _ { 1 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ . We obtain

$$
\Delta _ { t _ { 1 } } x _ { 2 } + 2 x _ { 2 } = ( - 1 ) ^ { t _ { 1 } } \partial _ { t _ { 2 } } a _ { 1 } - \eta \left[ { \frac { 3 } { 2 } } \zeta a _ { 1 } ^ { 2 } + ( - 1 ) ^ { t _ { 1 } } a _ { 1 } y _ { 1 } - 1 \right] .\tag{4.10}
$$

Consider the above equation as a diference equation for $x _ { 2 }$ in $t _ { 1 }$ , with fixed $t _ { 2 }$ and $t _ { 3 }$ . The right-hand side depends on $t _ { 1 }$ only through the $( - 1 ) ^ { t _ { 1 } }$ terms, as the terms $a _ { 1 }$ and $y _ { 1 }$ were shown to be independent of $t _ { 1 }$ . This suggests considering the following lemma.

Lemma 4.2. Let $\alpha , \beta \in \mathbb { R }$ . The solutions $\varphi = \varphi ( t _ { 1 } )$ of the diference equation

$$
\Delta _ { t _ { 1 } } \varphi + 2 \varphi = \alpha + \beta ( - 1 ) ^ { t _ { 1 } }
$$

are

$$
\varphi = { \frac { \alpha } { 2 } } + a ( - 1 ) ^ { t _ { 1 } } - \beta t _ { 1 } ( - 1 ) ^ { t _ { 1 } } , \qquad a \in \mathbb { R } .
$$

In particular, there exists a bounded solution if and only if $\beta = 0$

Proof. This lemma is a particular case of Lemma C.1 below.

Applying this lemma to Eq. (4.10), we obtain that we must have $\partial _ { t _ { 2 } } a _ { 1 } = \eta y _ { 1 } a _ { 1 }$ to avoid a secular term. The lemma also gives an expression for $x _ { 2 }$ , but it turns out that we do not need it in this section. This is typical of the method of multiple scales: we solve higher-order terms not always to include them in our approximation, but to obtain information about lower-order terms—here, the variations of $a _ { 1 }$ in $t _ { 2 }$

We now turn to the y equation (4.9). It gives

$$
\Delta _ { t _ { 1 } } y _ { 2 } + \partial _ { t _ { 2 } } y _ { 1 } + \partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \frac { x _ { 1 } ^ { 2 } } { 2 } - 1 \right] .
$$

As before, we consider this as a diference equation for $y _ { 2 }$ in $t _ { 1 }$

$$
\Delta _ { t _ { 1 } } y _ { 2 } = - \partial _ { t _ { 2 } } y _ { 1 } - \partial _ { t _ { 3 } } y _ { 0 } - \eta \left[ \frac { x _ { 1 } ^ { 2 } } { 2 } - 1 \right] .
$$

We distinguish whether we are at the edge of stability.

• If $\eta y _ { 0 } < 2$ , then $x _ { 1 } \to 0$ as $t _ { 1 } \to \infty$ , and all other quantities of the right-hand side are independent of $t _ { 1 }$ . To prevent a secular term, we must have

$$
- \partial _ { t _ { 2 } } y _ { 1 } - \partial _ { t _ { 3 } } y _ { 0 } + \eta = 0 .
$$

We consider this as an ODE for $y _ { 1 }$ in $t _ { 2 }$ . As $y _ { 0 }$ is independent of $t _ { 2 }$ , we have a secular term unless

$$
\partial _ { t _ { 3 } } y _ { 0 } = \eta .
$$

This finishes our derivation of item (i) of the result: in the relative interior of the stable region, the dynamics follow the gradient flow of $h$ constrained to the valley.

• If $\eta y _ { 0 } = 2$ , then we have $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ and $\partial _ { t _ { 3 } } y _ { 0 } = 0$ . We obtain

$$
\Delta _ { t _ { 1 } } y _ { 2 } = - \partial _ { t _ { 2 } } y _ { 1 } - \eta \left[ \frac { a _ { 1 } ^ { 2 } } { 2 } - 1 \right] .
$$

The right-hand side is independent of $t _ { 1 }$ . To avoid a secular term, we must have

$$
\partial _ { t _ { 2 } } y _ { 1 } = \eta \left( 1 - { \frac { a _ { 1 } ^ { 2 } } { 2 } } \right) .
$$

This completes the derivation of item (ii) of the result.

Remark 4.3. Without making the assumption ηy<sub>0</sub> $\leqslant 2$ , we would have considered a third case $\eta y _ { 0 } > 2$ in the derivations of order $\varepsilon ^ { 1 / 2 }$ . In this case, $x _ { 1 }$ develops a secular term (exponentially growing), unless $x _ { 1 } = 0$ . Thus, to avoid a secular term, we must have the dichotomy: ηy<sub>0</sub> $\leqslant 2$ or $x _ { 1 } = 0$ . It is not excluded that we leave the stable valley, however this can be done only in a way where the perturbation $x _ { 1 }$ of $x _ { 0 } = 0$ is exactly 0. This situation seems non-generic; in our case the term −x of $h$ maintains a small drift from $x = 0$ that prevents this situation. In Fig. 6, we observe that removing the term $- x$ from h results in the iterates leaving the stable valley for a significant slow time $t _ { 3 } ,$ thus breaking the central flow prediction. More generally, we conjecture that a perturbation in x (transversal gradient of $h ,$ noise, numerical errors, etc.) is necessary for the central flow prediction to hold.

## 4.3 Energy variations in self-stabilization

We have seen above that the self-stabilization dynamics preserve the energy

$$
E = E ( t _ { 3 } ) = \frac { y _ { 1 } ^ { 2 } } { 2 } + \frac { a _ { 1 } ^ { 2 } } { 4 } - \frac { 1 } { 2 } \log a _ { 1 } ^ { 2 } ,
$$

as a function of $t _ { 2 }$ . However, the energy still varies as a function $t _ { 3 } ,$ as observed in Figs. 4 and 5. The method of multiple scales allows us to study these variations.

Result 4.4. Under the assumptions of Result $4 . 1 ,$ at the edge of stability, we have

$$
\partial _ { t _ { 3 } } E = \eta ^ { 2 } ( 2 - 9 \zeta ^ { 2 } ) \left. y _ { 1 } ^ { 2 } \right. _ { t _ { 2 } } ,
$$

where $\left. y _ { 1 } ^ { 2 } \right. _ { t _ { 2 } } = \left. y _ { 1 } ^ { 2 } ( t _ { 2 } , t _ { 3 } ) \right. _ { t _ { 2 } }$ denotes the average of $y _ { 1 } ^ { 2 }$ over one period in $t _ { 2 }$

Note that $\left. y _ { 1 } ^ { 2 } \right. _ { t { \mathrm { 2 } } }$ is a function of E only, thus the above result is an autonomous ODE for E in $t _ { 3 }$ . If $\zeta ^ { 2 } ~ > ~ \frac { 2 } { 9 }$ , the energy decreases while if $\zeta ^ { 2 } < \frac { 2 } { 9 }$ , the energy increases, see Fig. 7. However, in the latter case, the energy does not diverge but saturates at a large value. In fact, E becomes so large that the scales of $\varepsilon ^ { 1 / 2 } a _ { 1 } , \varepsilon ^ { 1 / 2 } y _ { 1 }$ are no longer of order $\varepsilon ^ { 1 / 2 }$ , breaking the asymptotic expansion ansatz used to derive Result 4.4. Our derivations are then no longer predictive of the actual dynamics and more refined computations would be required to study the saturation of the energy.

![](images/263d44d67723d517b3aac3fa6359c182ff2cf253bda54ce96fa11cda096cdd09.jpg)

![](images/1077b72f5b3efe4e564089ae40bd37cfb4a031b1bfb68f405d8c14ef34b867cc.jpg)  
Figure 6: Iterates of gradient descent where $g$ is chosen as in Eqs. (4.1) and $h ( x , y ) = - x - y$ or $h ( x , y ) ~ = ~ - y$ . Here, $\eta \ : = \ : 0 . 3$ $\zeta ~ = ~ 1$ and $\varepsilon \ = \ 0 . 0 3$ . In the absence of the term $- x$ in $h$ , x becomes exponentially small in the first part of the dynamics, and thus it is possible to leave the stable valley, with the self-stabilization mechanism kicking in only after a significant slow time $t _ { 3 }$ . The self-stabilization mechanism is then so strong that the trajectory diverges.

To summarize, the self-stabilization system (4.3) and Result 4.4 should be understood as an analysis at energy levels of order 1. This analysis is most relevant for practice since neural networks seem to systematically dissipate energy when only one eigenvalue is on the edge of stability, see Fig. 8 of this paper or Figs. 5, 6, 10, 14, 15 in [8]. We leave it as an open problem to understand the structure of the loss of neural networks that causes this mechanism.

Analytically, studying the dependence of $E$ in $t _ { 3 }$ requires pushing the method of multiple scales to the order $\varepsilon ^ { 3 / 2 }$ . Indeed, as the reader might have noticed, the order 1 gives the dependence of $z _ { \mathrm { 0 } }$ in $t _ { 1 }$ ; the order $\varepsilon ^ { 1 / 2 }$ gives the dependencies of $z _ { 1 }$ in $t _ { 1 }$ and of $z _ { \mathrm { 0 } }$ in $t _ { 2 } ;$ the order ε gives the dependencies of $z _ { 2 }$ in $t _ { 1 }$ , of $z _ { 1 }$ in $t _ { 2 } .$ , and of $z _ { \mathrm { 0 } }$ in $t _ { 3 }$ . Continuing this pattern, the dependence of $z _ { 1 }$ (and thus of $E )$ in $t _ { 3 }$ is obtained at order $\varepsilon ^ { 3 / 2 }$ . This leads to rather tedious computations provided in Appendix A. We have done these derivations only in the minimal example of this section and leave the more general case of Sec. 5 for future work.

## 5 Valleys of codimension one

We now return to the general setup of Sec. 2. Here, we are interested in the fact that Y can be multidimensional, thus y<sub>0</sub> can move at the edge of stability. However, in this section, we still assume that $X = \mathbb { R }$ , so that $\nabla ^ { 2 } g ( 0 , y )$ has exactly one non-zero eigenvalue $S ( 0 , y ) = \partial _ { x } ^ { 2 } g ( 0 , y )$ . In Figure 1, we provide an example of such gradient descent dynamics where the valley is of dimension 2.

![](images/cdc0ff75d5b82c484da576392d00ff832113803340ed08bf46c8d098b6d6706c.jpg)  
(a) $\zeta = 0 . 5$

![](images/6e4429752f8ad0d342ef0a8a68801cb671314c85ca37aa33e2cbf0cd2cdb6956.jpg)  
(b) $\zeta = 0$  
Figure 7: Comparison between the empirical variation of the energy and the theoretical prediction of Result 4.4. This experiment is initialized at the edge of stability. Here, $\eta = 0 . 3$ and $\varepsilon = 1 0 ^ { - 5 }$ . More precisely, we define $y _ { 1 } ^ { \mathrm { e m p } } = \varepsilon ^ { - 1 / 2 } ( y - y _ { 0 } )$ and $a _ { 1 } ^ { \mathrm { e m p } } = \varepsilon ^ { - 1 / 2 } ( - 1 ) ^ { t _ { 1 } } ( x -$ $x _ { 0 } )$ . This defines an empirical energy $\bar { E ^ { \mathrm { e m p } } } = ( y _ { 1 } ^ { \mathrm { e m p } } ) ^ { 2 } / 2 + ( a _ { 1 } ^ { \mathrm { e m p } } ) ^ { 2 } / 4 - \log ( ( a _ { 1 } ^ { \mathrm { e m p } } ) ^ { 2 } ) / 2 ,$ that we compare against $\begin{array} { r } { E ^ { \mathrm { p r e d } } ( t _ { 3 } ) = E ^ { \mathrm { e m p } } ( 0 ) + \eta ^ { 2 } ( 2 - 9 \zeta ^ { 2 } ) \int _ { 0 } ^ { t _ { 3 } } ( y _ { 1 } ^ { \mathrm { e m p } } ) ^ { 2 } d s _ { 3 } } \end{array}$ . We denote $\begin{array} { r } { E _ { \operatorname* { m i n } } = \frac { 1 } { 2 } ( 1 - \log 2 ) } \end{array}$ the minimal possible energy.

We now state a general result for such valleys of codimension 1. This result is an extension of Result 3.1 and a generalization of Result 4.1.

Result 5.1. Assume that X is of dimension 1. Further, assume that $( x _ { 0 } , y _ { 0 } )$ remains in the stable valley, $i . e . , x _ { 0 } = 0$ and $\eta S ( 0 , y _ { 0 } ) \leqslant 2$ . Then $y _ { 0 } = y _ { 0 } ( t _ { 3 } )$ , and $y _ { 1 } = y _ { 1 } ( t _ { 2 } , t _ { 3 } )$ Further, we have the following:

(i) There exists $\sigma ^ { 2 } = \sigma ^ { 2 } ( t _ { 3 } ) \geqslant 0$ such that

$$
\partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \nabla _ { y } h ( 0 , y _ { 0 } ) + \frac { \sigma ^ { 2 } } { 2 } \nabla _ { y } S ( 0 , y _ { 0 } ) \right] ,
$$

where for all $t _ { 3 } , \sigma ^ { 2 } ( t _ { 3 } ) = 0 \ o r \ \eta S ( 0 , y _ { 0 } ( t _ { 3 } ) ) = 2$

(ii) If $\eta S ( 0 , y _ { 0 } ) = 2 ~ ( e d g e ~ o f ~ s t a b i l i t y )$ , then $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 }$ for some $a _ { 1 } = a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ Further, denote $\boldsymbol { u } _ { 1 } = \langle y _ { 1 } , \nabla _ { y } S ( 0 , y _ { 0 } ) \rangle$ . Then we have the self-stabilization system

$$
\partial _ { t _ { 2 } } a _ { 1 } = \eta a _ { 1 } u _ { 1 } ,
$$

$$
\partial _ { t _ { 2 } } u _ { 1 } = - \eta \left[ \langle \nabla _ { y } h ( 0 , y _ { 0 } ) , \nabla _ { y } S ( 0 , y _ { 0 } ) \rangle + \lVert \nabla _ { y } S ( 0 , y _ { 0 } ) \rVert ^ { 2 } \frac { a _ { 1 } ^ { 2 } } { 2 } \right] .\tag{5.1}
$$

Moreover, in that case, we have $\sigma ^ { 2 } = \langle a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } , \ i . e . \ \sigma ^ { 2 } ( t _ { 3 } )$ is the average of $a _ { 1 } ^ { 2 } ( t _ { 2 } , t _ { 3 } )$ over a period in $t _ { 2 }$

![](images/ae79e56173125db65f4e246124609587a0ca01c4027abc76a2cfbedf8d5ba14f.jpg)  
Figure 8: Evolution of the top eigenvalues of the Hessian along the training path of a neural network. The amplitude of the self-stabilization fluctuations decreases when only one eigenvalue is on the edge of stability. When a second eigenvalue reaches the edge of stability, the amplitude of the fluctuations stabilizes at a larger value, as expected from the analysis of Sec. 6. Here, we train a multilayer perceptron with two hidden layers of width 256 and GELU activation on a subset of CIFAR-10 with mean-squared error loss. The learning rate is $\eta = 0 . 0 3$

The derivation of this result is provided in Appendix B. Note that in the self-stabilization system (5.1), $\langle \nabla _ { y } h ( 0 , y _ { 0 } ) , \nabla _ { y } S ( 0 , y _ { 0 } ) \rangle$ and $\| \nabla _ { y } S ( 0 , y _ { 0 } ) \| ^ { 2 }$ are independent of $t _ { 2 }$ . Moreover, in a situation where the dynamics stay at the edge of stability, we must have $\langle \nabla _ { y } h ( 0 , y _ { 0 } ) , \nabla _ { y } S ( 0 , y _ { 0 } ) \rangle < 0 :$ the gradient flow on $h$ pushes towards the stability boundary. The self-stabilization system (5.1) is thus the same as the one of (4.3), up to a rescaling. A self-stabilization loop appears, preserving an energy $E = E ( t _ { 3 } )$ . For fixed $t _ { 3 } , ( a _ { 1 } , u _ { 1 } )$ is periodic in $t _ { 2 } .$ , thus the average $\langle a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } }$ over $t _ { 2 }$ is well-defined.

The derivation of the central flow in [8] already interpreted $\sigma ^ { 2 } = \mathbb { E } x ^ { 2 }$ , where E denotes “local time-averages”. Here, thanks to the timescale decoupling, we can be more precise: E corresponds to averaging over $t _ { 2 }$ , for fixed $t _ { 3 }$

## 6 Valleys of arbitrary codimension

This section tackles the case where the dimension of X is arbitrary.

Result 6.1. Assume that $( x _ { 0 } , y _ { 0 } )$ remains in the stable valley, i. $e _ { \cdot } , x _ { 0 } = 0$ and $\eta S ( 0 , y _ { 0 } ) \preceq$ $2 \operatorname { I d } _ { X }$ . Then $y _ { 0 } = y _ { 0 } ( t _ { 3 } )$ , and $y _ { 1 } = y _ { 1 } ( t _ { 2 } , t _ { 3 } )$ . Further, we have the following:

(i) There exists $\Sigma = \Sigma ( t _ { 3 } ) \in \mathrm { S y m } _ { + } X$ such that

$$
\partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \nabla _ { y } h ( 0 , y _ { 0 } ) + \frac { 1 } { 2 } D _ { y } S ( 0 , y _ { 0 } ) ^ { * } [ \Sigma ] \right] ,
$$

where for all $t _ { 3 } , \langle \Sigma ( t _ { 3 } ) , 2 \operatorname { I d } _ { X } - \eta S ( 0 , y _ { 0 } ) \rangle = 0$ (ii) Let $C = C ( t _ { 3 } ) = \ker ( 2 \operatorname { I d } _ { X } - \eta S ( 0 , y _ { 0 } ) ) \subset X$ denote the critical subspace. Recall that $\iota _ { C } : C \to X$ denotes the canonical inclusion, and $P _ { C } : X  C$ the orthogonal projection onto C. We have $P _ { C } x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 }$ for some $a _ { 1 } = a _ { 1 } ( t _ { 2 } , t _ { 3 } ) \in C$ and $P _ { C ^ { \perp } } x _ { 1 } \xrightarrow [ t _ { 1 } \to \infty ] { } 0$ . Further, denote

$$
\begin{array} { r } { \Psi = \Psi ( t _ { 3 } ) : Y \to \operatorname { S y m } C \quad } \\ { \qquad \quad \delta y \mapsto \Psi [ \delta y ] = P C D _ { y } S ( 0 , y _ { 0 } ) [ \delta y ] \iota _ { C } , } \end{array}
$$

and $U _ { 1 } = \Psi [ y _ { 1 } ] \in \operatorname { S y m } C$ . Then we have the self-stabilization system

$$
\begin{array} { l } { { \partial _ { t _ { 2 } } a _ { 1 } = \eta U _ { 1 } a _ { 1 } \mathrm { , } } } \\ { { \displaystyle \partial _ { t _ { 2 } } U _ { 1 } = - \eta \left[ \Psi [ \nabla _ { y } h ( 0 , y _ { 0 } ) ] + \frac { 1 } { 2 } \Psi \Psi ^ { * } [ a _ { 1 } a _ { 1 } ^ { \top } ] \right] } } \end{array}\tag{6.1}
$$

Moreover, provided that the average $\begin{array} { r } { \langle a _ { 1 } a _ { 1 } ^ { \top } \rangle _ { t _ { 2 } } = \operatorname* { l i m } _ { T _ { 2 }  \infty } \frac { 1 } { T _ { 2 } } \int _ { 0 } ^ { T _ { 2 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } ) a _ { 1 } ( t _ { 2 } , t _ { 3 } ) ^ { \top } \mathrm { d } t _ { 2 } } \end{array}$ is well-defined, we have $\Sigma = \iota _ { C } \langle a _ { 1 } a _ { 1 } ^ { \intercal } \rangle _ { t _ { 2 } } P _ { C }$

The derivation of this result is provided in Appendix C. The derivation of the central flow in [8] already interpreted $\Sigma = \mathbb { E } a a ^ { \top }$ , where E denotes “local time-averages”. Here, thanks to the timescale decoupling, we can be more precise: E corresponds to averaging over $t _ { 2 } ,$ for fixed $t _ { 3 }$

Result 6.1 is a generalization of Result 5.1. When only one eigenvalue of $S ( 0 , y _ { 0 } )$ has reached the edge of stability, we recover the self-stabilization system (5.1). Indeed, then C is of dimension 1, thus $a _ { 1 } \in C$ and $U _ { 1 } \in \mathrm { S y m } C$ are scalars. Moreover, $\Psi \Psi ^ { * } : \mathrm { S y m } C $ Sym C is a linear operator on a one-dimensional space, thus it is a scalar multiplication. Thus in that case, the self-stabilization system (6.1) reduces to the self-stabilization system (5.1). We have a periodic behavior in $t _ { 2 }$

When more eigenvalues enter the edge of stability, the behavior of the self-stabilization system (6.1) is much richer, see Fig. 2.

The self-stabilization system (5.1) admits the fixed point $\begin{array} { r } { u _ { 1 } = 0 , a _ { 1 } ^ { 2 } = - \frac { 2 \langle \nabla _ { y } h , \nabla _ { y } S \rangle } { \| \nabla _ { y } S \| ^ { 2 } } } \end{array}$ Meanwhile, finding a fixed point for the self-stabilization system (6.1) would require solving the equation $\Psi \Psi ^ { * } [ a _ { 1 } a _ { 1 } ^ { \top } ] = - 2 \Psi [ \nabla _ { y } h ( 0 , y _ { 0 } ) ]$ . As the right hand side is in im $\Psi = \operatorname { i m } \Psi \Psi ^ { * }$ there exists $\tilde { \Sigma } \in \mathrm { S y m } \bar { C }$ such that $\Psi \Psi ^ { * } [ \widetilde { \Sigma } ] = - 2 \Psi [ \nabla _ { y } h ( 0 , y _ { 0 } ) ]$ . However, it is non-generic that a rank-one solution $\widetilde { \Sigma }$ exists. Thus, generically, there is no fixed point, and to avoid a secular term, $a _ { 1 } a _ { 1 } ^ { \top }$ must evolve persistently so that its average value is $\widetilde { \Sigma } .$

This explains the observed behavior in Fig. 2. When only one eigenvalue is on the edge of stability, the self-stabilization system is observed to lose energy (on the $t _ { 3 }$ timescale), converging to its fixed point. However, when a second eigenvalue enters the edge of stability, there is a significant increase in the edge of stability fluctuations, and these fluctuations are persistent. This phenomenology, here illustrated on a toy example, has been observed systematically in deep learning experiments, see Fig. 8 of this paper or Figs. 5, 6, 10, 14, 15 in [8].

To finish, we now discuss the implications of Result 6.1 for the loss $f ( x , y )$ , as it is the most commonly observed quantity in practice.

Corollary 6.2. Under the assumptions of Result 6.1, we have

$$
f ( x , y ) = \operatorname* { m i n } g + \varepsilon f _ { 2 } + o ( \varepsilon )
$$

where

$$
f _ { 2 } \xrightarrow [ { t _ { 1 } \to \infty } ] { } h ( 0 , y _ { 0 } ) + \frac { \| a _ { 1 } \| ^ { 2 } } { \eta } .
$$

The derivation of this corollary is provided in Appendix D. The dominant non-constant term of the loss is of order ε. In the relative interior of the stable valley, this term depends only on $t _ { 3 }$ (up to a potential initial transient term in $t _ { 1 } )$ , through the component h of the loss. At the edge of stability, an additional increase $\| a _ { 1 } \| ^ { 2 } / \eta$ appears, depending on both $t _ { 2 }$ and $t _ { 3 }$ . In $t _ { 2 }$ , the self-stabilization fluctuations induce spikes in the loss. Averaging over these, and assuming that the average $\langle a _ { 1 } a _ { 1 } ^ { \top } \rangle _ { t _ { 2 } }$ is well-defined, we obtain

$$
\langle f _ { 2 } \rangle _ { t _ { 2 } } \xrightarrow [ t _ { 1 } \to \infty ] { } h ( 0 , y _ { 0 } ) + \frac { \mathrm { T r } \Sigma } { \eta } ,
$$

consistently with the result of [8, Sec. 3.2.3]. These results are illustrated in Fig. 9.

![](images/95e54d46d75ba8b4cfff7eedfbe8c5c0fe843a2b3bca1ca3f393d41b5d8438bb.jpg)

![](images/50549492585eec126d4c6b50dfa206f3e55831455334051e021bb5283b4a7297.jpg)

![](images/13f440f0f63b47bfd07fa46cb049784618307cbea35eeda74b97540acdd713b7.jpg)  
Figure 9: Rescaled loss $\begin{array} { r } { \varepsilon ^ { - 1 } \left( f ( x ( k ) , y ( k ) ) - \operatorname* { m i n } _ { ( x , y ) \in V _ { \mathrm { S } } } f ( x , y ) \right) } \end{array}$ along the iterates of gradient descent and its central flow prediction $h ( 0 , y _ { 0 } ) - \operatorname* { m i n } _ { ( x , y ) \in V _ { \mathrm { S } } } h ( x , y )$ min $+ \sigma ^ { 2 } / \eta .$ , in the same experiment as in Fig. 1. The loss progress made on h in the valley and the loss increase due to oscillations are both of the same order ε.

## 7 Related work

Sharp valley structure of the loss. Arguments for the sharp valley structure of the loss landscape are widespread in the literature. To give a few examples, this picture is consistent with plots of the loss along the sharpest direction of the Hessian around an iterate [15], plots of the loss along the segment between two consecutive iterates [42], and the anti-alignment of the gradients of consecutive iterates [42]. Along the training path, the loss Hessian exhibits an approximate low-rank structure [31, 32, 11, 28, 29, 4]. The gradients are mostly aligned with the sharpest directions of the Hessian [13, 19], however the component of the gradients in these dominant directions does not contribute to any loss decrease [15, 34], but to a sharpness reduction efect [33]. In a toy problem, the valley structure has been mathematically analyzed and the sharpest directions are associated with uncertain features of the problem [38]. Finally, given this accumulation of empirical evidence, the sharp valley structure has already been used as a working hypothesis in some works [22, 23].

Analyses of the edge of stability phenomenon. After the initial empirical observations of the edge of stability phenomenon [15, 16, 9], a large body of work has been devoted to explaining it [2, 3, 37, 10, 44, 24, 1, 6, 18, 35, 40, 39, 5, 36, 17, 12, 43, 21, 41, 20]. In contrast, our results assume that the leading-term of dynamics remains in the stable region, and study the consequences of this assumption. To the best of our knowledge, no previous work had provided additional justification for the central flow approximation [8].

From a modeling perspective, our work bears similarities with [25, 26] that study the dynamics of gradient descent around a valley of minimizers. They show that gradient descent implicitly performs a Riemannian gradient descent on the sharpness in the valley of minimizers, and describe the bifurcating dynamical system in the orthogonal direction. In their “subcritical regime”, their analysis corresponds to the case $h = 0$ in our work. However, in the absence of ∇h pushing towards the stability boundary, the edge of stability phenomenon is transient.

Comparison with other approximations. Several works [30, 27, 7] subsequent to the central flow [8] have proposed other averaged approximations of gradient descent, see Fig. 10 for a comparison. In addition to the slow dynamics in $t _ { 3 } .$ , these approximations attempt at capturing the self-stabilization dynamics in $t _ { 2 }$ . In fact, it is even explicit in the derivations of [30, 27] that they average only two subsequent iterates, thus over the timescale $t _ { 1 }$ only. All of these approximations report similar or slightly better accuracy than the central flow in approximating gradient descent. While qualitatively, these approximations correctly generate self-stabilization loops, these loops are often of in magnitude or in phase, leading to only minor quantitative improvement.

Overall, in the presence of the several timescales $t _ { 1 } , t _ { 2 } , t _ { 3 }$ , efective models can average at diferent levels, leading to diferent tradeofs between accuracy and conceptual simplicity. The central flow, with its constrained gradient flow formulation, is the simplest and coarsest version, averaging over $t _ { 1 }$ and $t _ { 2 }$ . The other approximations [30, 27, 7], averaging over $t _ { 1 }$ only, contain more information on the self-stabilization dynamics but have a more complex formulation. This paper advocates for seeking expansions of the dynamics, where each individual term is simple and interpretable, while their combination can have high accuracy.

## 8 Conclusion

The contributions of this paper are both conceptual and analytical. Conceptually, we clarify that the central flow and the self-stabilization mechanism are related to a sharp valley structure of the loss landscape. However, understanding why such a loss landscape structure appears when training deep learning models is still largely an open problem (see [38] for a first approach).

![](images/c759bec67bd49157b9a27ba96a5f96e4c04f197d1a8ff940591bef4d8635ce5d.jpg)  
Figure 10: Comparison of the central flow [8], the rod flow [30], the edge gradient descent [27] and the flow of [7] (labeled “free energy flow” here) in the setting of Fig. 3. The fact that the rod flow may fail to detect the start of the edge of stability phase had already been pointed out in [27, Fig. 6].

Analytically, we provide a framework for studying the dynamics of gradient descent on the edge of stability, based on the method of multiple scales. This framework could be improved by formalizing the derivations through more rigorous techniques of singular perturbation theory. Computing the efect of a curved valley V is also an interesting direction for future work. The works of [25, 26] provide interesting first elements in these directions.

The strength of the framework stems from its systematic nature. As a consequence, we plan to use it to analyze inertial and adaptive algorithms. We hope that understanding these complex dynamics across many optimizers will help to design new methods.

## Acknowledgements

The author thanks Victor Baillet, Léo Dana and Loucas Pillaud-Vivien for fruitful discussions. He also thanks the authors of [8] for making their code publicly available; in particular, this code was used to generate Figure 8.

AI-based tools were used for line completion in the text editor, for generating the plotting scripts that produce the figures from the data, and for proofreading. They were not used to generate the scientific content of the paper, to draft its prose beyond phraselevel completion, or to write the code implementing the experiments. The author has verified all AI-assisted output and takes full responsibility for the content of the paper.

The author acknowledges support from the ANR and the Ministère de l’Enseignement Supérieur et de la Recherche.

## References

[1] Atish Agarwala, Fabian Pedregosa, and Jefrey Pennington. Second-order regression models exhibit progressive sharpening to the edge of stability. In International Conference on Machine Learning, 2023.

[2] Kwangjun Ahn, Jingzhao Zhang, and Suvrit Sra. Understanding the unstable convergence of gradient descent. In International Conference on Machine Learning, 2022.

[3] Sanjeev Arora, Zhiyuan Li, and Abhishek Panigrahi. Understanding gradient descent on the edge of stability in deep learning. In International Conference on Machine Learning, 2022.

[4] Gerard Ben Arous, Reza Gheissari, Jiaoyang Huang, and Aukosh Jagannath. Highdimensional SGD aligns with emerging outlier eigenspaces. In International Conference on Learning Representations, 2024.

[5] Yuhang Cai, Jingfeng Wu, Song Mei, Michael Lindsey, and Peter Bartlett. Large stepsize gradient descent for non-homogeneous two-layer networks: Margin improvement and fast optimization. Advances in Neural Information Processing Systems, 2024.

[6] Lei Chen and Joan Bruna. Beyond the edge of stability via two-step gradient updates. In International Conference on Machine Learning, 2023.

[7] Antonin Chodron de Courcel. Gradient descent at the edge of stability: free energy model and kinetic description of the two-layer network. arXiv preprint arXiv:2606.05326, 2026.

[8] Jeremy Cohen, Alex Damian, Ameet Talwalkar, Zico Kolter, and Jason Lee. Understanding optimization in deep learning with central flows. In International Conference on Learning Representations, 2025.

[9] Jeremy Cohen, Simran Kaur, Yuanzhi Li, Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations, 2021.

[10] Alex Damian, Eshaan Nichani, and Jason Lee. Self-stabilization: The implicit bias of gradient descent at the edge of stability. In International Conference on Learning Representations, 2023.

[11] Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via Hessian eigenvalue density. In International Conference on Machine Learning, 2019.

[12] Avrajit Ghosh, Soo Min Kwon, Rongrong Wang, Saiprasad Ravishankar, and Qing Qu. Learning dynamics of deep linear networks beyond the edge of stability. International Conference on Learning Representations, 2025.

[13] Guy Gur-Ari, Daniel Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace. arXiv preprint arXiv:1812.04754, 2018.

[14] Mark Holmes. Introduction to perturbation methods. Springer Science & Business Media, 2012.

[15] Stanisław Jastrzębski, Zachary Kenton, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. On the relation between the sharpest directions of DNN loss and the SGD step length. In International Conference on Learning Representations, 2019.

[16] Stanislaw Jastrzębski, Maciej Szymczak, Stanislav Fort, Devansh Arpit, Jacek Tabor, Kyunghyun Cho, and Krzysztof Geras. The break-even point on optimization trajectories of deep neural networks. In International Conference on Learning Representations, 2020.

[17] Dayal Singh Kalra, Tianyu He, and Maissam Barkeshli. Universal sharpness dynamics in neural network training: Fixed point analysis, edge of stability, and route to chaos. In International Conference on Learning Representations, 2025.

[18] Itai Kreisler, Mor Shpigel Nacson, Daniel Soudry, and Yair Carmon. Gradient descent monotonically decreases the sharpness of gradient flow solutions in scalar networks and beyond. In International Conference on Machine Learning, 2023.

[19] Xinyan Li, Qilong Gu, Yingxue Zhou, Tiancong Chen, and Arindam Banerjee. Hessian based analysis of SGD for deep nets: Dynamics and generalization. In SIAM International Conference on Data Mining, 2020.

[20] Elon Litman. The origin of edge of stability. arXiv preprint arXiv:2604.20446, 2026.

[21] Liming Liu, Zixuan Zhang, Simon Du, and Tuo Zhao. A minimalist example of edgeof-stability and progressive sharpening. Advances in Neural Information Processing Systems, 2025.

[22] Yizhou Liu, Ziming Liu, and Jef Gore. Focus: First order concentrated updating scheme. arXiv preprint arXiv:2501.12243, 2025.

[23] Ziming Liu, Yizhou Liu, Jef Gore, and Max Tegmark. Neural thermodynamic laws for large language model training. arXiv preprint arXiv:2505.10559, 2025.

[24] Chao Ma, Daniel Kunin, Lei Wu, and Lexing Ying. Beyond the quadratic approximation: The multiscale structure of neural network loss landscapes. arXiv preprint arXiv:2204.11326, 2022.

[25] Lachlan MacDonald, Hancheng Min, Leandro Palma, Salma Tarmoun, Ziqing Xu, and René Vidal. Convergence rates for gradient descent on the edge of stability in overparametrised least squares. Advances in Neural Information Processing Systems, 2025.

[26] Lachlan MacDonald and René Vidal. Dynamics of gradient descent with large step size near a manifold of flat minima. arXiv preprint arXiv:2607.08380, 2026.

[27] Pierre Marion. Edge flow: A tractable and predictive continuous-time model for gradient descent at the edge of stability. arXiv preprint arXiv:2606.18080, 2026.

[28] Vardan Papyan. Measurements of three-level hierarchical structure in the outliers in the spectrum of deepnet Hessians. In International Conference on Machine Learning, 2019.

[29] Vardan Papyan. Traces of class/cross-class structure pervade deep learning spectra. Journal of Machine Learning Research, 21(252):1–64, 2020.

[30] Eric Regis and Sinho Chewi. Rod flow: A continuous-time model for gradient descent at the edge of stability. arXiv preprint arXiv:2602.01480, 2026.

[31] Levent Sagun, Leon Bottou, and Yann LeCun. Eigenvalues of the Hessian in deep learning: Singularity and beyond. arXiv preprint arXiv:1611.07476, 2016.

[32] Levent Sagun, Utku Evci, V. Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the Hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454, 2017.

[33] Jun-Ho So and Dongwook Shin. Mini-batch noise lowers sharpness via dominantsubspace fluctuations. In High-dimensional Learning Dynamics, 2026.

[34] Minhak Song, Kwangjun Ahn, and Chulhee Yun. Does SGD really happen in tiny subspaces? In International Conference on Learning Representations, 2025.

[35] Minhak Song and Chulhee Yun. Trajectory alignment: Understanding the edge of stability phenomenon via bifurcation theory. Advances in Neural Information Processing Systems, 2023.

[36] Yuqing Wang, Zhenghao Xu, Tuo Zhao, and Molei Tao. Good regularity creates large learning rate implicit biases: edge of stability, balancing, and catapult. Journal of Machine Learning Research, 26(273):1–68, 2025.

[37] Zixuan Wang, Zhouzi Li, and Jian Li. Analyzing sharpness along gd trajectory: Progressive sharpening and edge of stability. Advances in Neural Information Processing Systems, 2022.

[38] Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, and Tengyu Ma. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective. In International Conference on Learning Representations, 2025.

[39] Jingfeng Wu, Peter Bartlett, Matus Telgarsky, and Bin Yu. Large stepsize gradient descent for logistic loss: Non-monotonicity of the loss improves optimization eficiency. In Conference on Learning Theory, 2024.

[40] Jingfeng Wu, Vladimir Braverman, and Jason Lee. Implicit bias of gradient descent for logistic regression at the edge of stability. Advances in Neural Information Processing Systems, 2023.

[41] Jingfeng Wu, Pierre Marion, and Peter Bartlett. Large stepsizes accelerate gradient descent for regularized logistic regression. Advances in Neural Information Processing Systems, 2025.

[42] Chen Xing, Devansh Arpit, Christos Tsirigotis, and Yoshua Bengio. A walk with SGD. arXiv preprint arXiv:1802.08770, 2018.

[43] Geonhui Yoo, Minhak Song, and Chulhee Yun. Understanding sharpness dynamics in NN training with a minimalist example: The efects of dataset dificulty, depth, stochasticity, and more. arXiv preprint arXiv:2506.06940, 2025.

[44] Xingyu Zhu, Zixuan Wang, Xiang Wang, Mo Zhou, and Rong Ge. Understanding edge-of-stability training dynamics with a minimalist example. In International Conference on Learning Representations, 2023.

## Appendix

## A Derivation of Result 4.4

We compute the terms of order $\varepsilon ^ { 3 / 2 }$ in the equation $\Delta _ { k } z = - \eta \nabla f ( z )$ . For the right-hand side, we have

$$
\begin{array} { r l } { - \eta \nabla f ( x ) - } & { - \eta \nabla f ( x ) - } \\ & { = - \eta [ \nabla \delta ( \xi \circ \xi ^ { - 1 / 2 } - x \nabla \delta ( x ^ { 1 / 2 } )   } \\ & {   = - \eta [ \nabla \delta ( \xi \circ \xi ^ { - 1 / 2 } \circ + \xi ^ { - 2 / 2 } \circ + \xi ^ { \lambda / 2 } \circ + \delta ^ { \lambda / 2 } ) ] + \xi \nabla \delta ( \xi \circ \xi ^ { 1 / 2 } ) - \delta ( \xi ^ { \lambda / 2 } ) ] } \\ & { \qquad \quad = - \eta [ \nabla \delta ( \xi \circ \xi ^ { - 1 / 2 } \circ + \nabla ^ { \lambda / 2 } \delta ( x ^ { 1 / 2 } ) ) ( \xi ^ { 1 / 2 } \circ \xi \circ + \xi ^ { \lambda / 2 } \circ + \xi ^ { \lambda / 2 } \circ )  } \\ & { \qquad \quad + \frac { 1 } { 2 } \nabla \delta ^ { - 1 / 2 } ( \xi \circ \xi \circ \xi ) [ \xi ^ { 1 / 2 } \circ \xi \circ \xi ^ { - 1 / 2 } \circ \xi \circ \xi ^ { 1 / 2 } \circ + \xi \pi _ { 2 } ^ { \lambda } ] } \\ & { \qquad \quad + \frac { 1 } { 6 } \nabla \delta ^ { - 1 / 2 } ( \xi \circ \xi \circ \xi ) [ \xi ^ { 1 / 2 } \circ \xi _ { 1 } \xi ^ { 1 / 2 } \circ \xi _ { 1 } \xi ^ { 1 / 2 } \circ \xi _ { 1 }   } \\ & { \qquad  \textrm { i f } \frac { \xi } { \delta } \textbf { \textrm { B } } ] ( \xi \circ \xi ^ { 1 / 2 } \circ \xi \circ \xi ^ { 1 / 2 } \circ \xi \circ \xi ^ { 1 / 2 } \circ ) \textrm { d } \rho ( \xi ^ { \lambda / 2 } ) ] } \\ &  \qquad \quad = - \eta \nabla f ( x ) - \xi ^ { 1 / 2 } \eta \nabla \delta ^ { - 1 / 2 } [ \phi \circ \xi \circ \xi \circ \ \end{array}
$$

Using the expressions for g and h of Eqs. (4.1) and the assumption $x _ { 0 } = 0$ , we obtain

$$
\begin{array} { c } { { \displaystyle - \eta \partial _ { x } f ( x , y ) = - \varepsilon ^ { 1 / 2 } \eta y _ { 0 } x _ { 1 } - \varepsilon \eta \left[ y _ { 0 } x _ { 2 } + \frac 3 2 \zeta x _ { 1 } ^ { 2 } + x _ { 1 } y _ { 1 } - 1 \right] } } \\ { { { } } } \\ { { - \varepsilon ^ { 3 / 2 } \eta \left[ y _ { 0 } x _ { 3 } + 3 \zeta x _ { 1 } x _ { 2 } + x _ { 1 } y _ { 2 } + x _ { 2 } y _ { 1 } \right] + o ( \varepsilon ^ { 3 / 2 } ) , } } \\ { { { } } } \\ { { - \eta \partial _ { y } f ( x , y ) = - \varepsilon \eta \left[ \frac { x _ { 1 } ^ { 2 } } 2 - 1 \right] - \varepsilon ^ { 3 / 2 } \eta x _ { 1 } x _ { 2 } + o ( \varepsilon ^ { 3 / 2 } ) . } } \end{array}
$$

For the left-hand side, we first decompose

$$
\begin{array} { l } { { z ( t _ { 1 } , t _ { 2 } + { \varepsilon } ^ { 1 / 2 } , t _ { 3 } + { \varepsilon } ) - z ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) } } \\ { { \mathrm { ~ } } } \\ { { \displaystyle \quad = { \varepsilon } ^ { 1 / 2 } \partial _ { t _ { 2 } } z + { \varepsilon } \partial _ { t _ { 3 } } z + \frac { { \varepsilon } } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z + { \varepsilon } ^ { 3 / 2 } \partial _ { t _ { 2 } } \partial _ { t _ { 3 } } z + \frac { { \varepsilon } ^ { 3 / 2 } } { 6 } \partial _ { t _ { 2 } } ^ { 3 } z + o ( { \varepsilon } ^ { 3 / 2 } ) . } } \end{array}
$$

Using a similar expansion for $( \Delta _ { t _ { 1 } } z ) ( t _ { 1 } , t _ { 2 } + \varepsilon ^ { 1 / 2 } , t _ { 3 } + \varepsilon )$ and substituting in Eq. (4.6), we obtain

$$
\begin{array} { l } { { \Delta _ { k } z = \Delta _ { t _ { 1 } } z + \varepsilon ^ { 1 / 2 } \partial _ { t _ { 2 } } z + \varepsilon \partial _ { t _ { 3 } } z + \frac { \varepsilon } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z + \varepsilon ^ { 3 / 2 } \partial _ { t _ { 2 } } \partial _ { t _ { 3 } } z + \frac { \varepsilon ^ { 3 / 2 } } { 6 } \partial _ { t _ { 2 } } ^ { 3 } z } } \\ { { \qquad + \varepsilon ^ { 1 / 2 } \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } z + \varepsilon \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z + \frac { \varepsilon } { 2 } \partial _ { t _ { 2 } } ^ { 2 } \Delta _ { t _ { 1 } } z + \varepsilon ^ { 3 / 2 } \partial _ { t _ { 2 } } \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z + \frac { \varepsilon ^ { 3 / 2 } } { 6 } \partial _ { t _ { 2 } } ^ { 3 } \Delta _ { t _ { 1 } } z + o ( \varepsilon ^ { 3 / 2 } ) } } \\ { { \qquad = \Delta _ { t _ { 1 } } z + \varepsilon ^ { 1 / 2 } \left[ \partial _ { t _ { 2 } } z + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } z \right] + \varepsilon \left[ \partial _ { t _ { 3 } } z + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z + \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } \Delta _ { t _ { 1 } } z \right] } } \\   \qquad + \varepsilon ^ { 3 / 2 } \left[ \partial _ { t _ { 2 } } \partial _ { t _ { 3 } } z + \frac { 1 } { 6 } \partial _ { t _ { 2 } } ^ { 3 } z + \partial _ { t _ { 2 } } \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z + \frac { 1 } { 6 } \partial _ { t _ { 2 } } ^ { 3 } \Delta _ { t _ { 1 } } z \right] + o ( \varepsilon ^  3 / 2  \end{array}
$$

We combine the above decomposition with the expansion of z. We obtain

$$
\begin{array} { c } { { \Delta _ { k } z = \Delta _ { t _ { 1 } } z _ { 0 } + \varepsilon ^ { 1 / 2 } \left[ \Delta _ { t _ { 1 } } z _ { 1 } + \partial _ { t _ { 2 } } z _ { 0 } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } z _ { 0 } \right] } } \\ { { + \varepsilon \left[ \Delta _ { t _ { 1 } } z _ { 2 } + \partial _ { t _ { 2 } } z _ { 1 } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } z _ { 1 } + \partial _ { t _ { 3 } } z _ { 0 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z _ { 0 } + \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z _ { 0 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } \Delta _ { t _ { 1 } } z _ { 0 } \right] } } \\ { { + \varepsilon ^ { 3 / 2 } \bigg [ \Delta _ { t _ { 1 } } z _ { 3 } + \partial _ { t _ { 2 } } z _ { 2 } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } z _ { 2 } + \partial _ { t _ { 3 } } z _ { 1 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } z _ { 1 } + \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z _ { 1 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } \Delta _ { t _ { 1 } } z _ { 1 } } } \\ { { + \partial _ { t _ { 2 } } \partial _ { t _ { 3 } } z _ { 0 } + \frac { 1 } { 6 } \partial _ { t _ { 2 } } ^ { 3 } z _ { 0 } + \partial _ { t _ { 2 } } \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } z _ { 0 } + \frac { 1 } { 6 } \partial _ { t _ { 2 } } ^ { 3 } \Delta _ { t _ { 1 } } z _ { 0 } \bigg ] + o ( \varepsilon ^ { 3 / 2 } ) . } } \end{array}
$$

We are now ready to identify the terms of order $\varepsilon ^ { 3 / 2 }$ . From Eq. (4.8), we obtain

$$
\begin{array} { l } { { \displaystyle \Delta _ { t _ { 1 } } x _ { 3 } + \partial _ { t _ { 2 } } x _ { 2 } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } x _ { 2 } + \partial _ { t _ { 3 } } x _ { 1 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } x _ { 1 } + \partial _ { t _ { 3 } } \Delta _ { t _ { 1 } } x _ { 1 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } \Delta _ { t _ { 1 } } x _ { 1 } } } \\ { { \displaystyle \quad \quad = - \eta \left[ y _ { 0 } x _ { 3 } + 3 \zeta x _ { 1 } x _ { 2 } + x _ { 1 } y _ { 2 } + x _ { 2 } y _ { 1 } \right] . } } \end{array}
$$

Here, we are interested in the edge of stability phase where $\eta y _ { 0 } = 2 , x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ Moreover, applying Lemma 4.2 to Eq. (4.10), we have $\begin{array} { r } { x _ { 2 } = - \frac { \eta } { 2 } \left[ \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 \right] + ( - 1 ) ^ { t _ { 1 } } a _ { 2 } } \end{array}$ for some $a _ { 2 } = a _ { 2 } ( t _ { 2 } , t _ { 3 } )$ . This gives

$$
\begin{array} { r l } { \Delta _ { t _ { 1 } } x _ { 3 } + 2 x _ { 3 } = \frac { 3 \eta } { 4 } \zeta \partial _ { t _ { 3 } } \left( a _ { 1 } ^ { 2 } \right) - ( - 1 ) ^ { t _ { 1 } } \partial _ { t _ { 3 } } a _ { 2 } + 2 \left( - 1 \right) ^ { t _ { 1 } } \partial _ { t _ { 3 } } a _ { 2 } - \left( - 1 \right) ^ { t _ { 1 } } \partial _ { t _ { 3 } } a _ { 1 } } & { } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad - \frac { 1 } { 2 } \left( - 1 \right) ^ { t _ { 1 } } \partial _ { t _ { 3 } } ^ { t _ { 2 } } a _ { 1 } + 2 \left( - 1 \right) ^ { t _ { 1 } } \partial _ { t _ { 3 } } a _ { 1 } + \left( - 1 \right) ^ { t _ { 1 } } \partial _ { t _ { 3 } } ^ { t _ { 2 } } a _ { 1 } } & { } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\  = \frac { 3 \eta } { 4 } \zeta \partial _  t _ \end{array}
$$

The right hand side depends on $t _ { 1 }$ only through the $( - 1 ) ^ { t _ { 1 } }$ terms. By Lemma 4.2, to avoid a secular term, we must have

$$
\begin{array} { r } { \partial _ { t _ { 2 } } a _ { 2 } + \partial _ { t _ { 3 } } a _ { 1 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } a _ { 1 } = \eta \left( - 3 \zeta a _ { 1 } \frac { \eta } { 2 } \left[ \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 \right] + a _ { 1 } y _ { 2 } + a _ { 2 } y _ { 1 } \right) } \\ { = - \eta ^ { 2 } \frac { 3 } { 2 } \zeta a _ { 1 } \left[ \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 \right] + \eta \left( a _ { 1 } y _ { 2 } + a _ { 2 } y _ { 1 } \right) . } \end{array}\tag{A.1}
$$

We now identify terms of order $\varepsilon ^ { 3 / 2 }$ in the $y$ equation (4.9). Recall that $y _ { 0 } = 2 / \eta , y _ { 1 } =$ $y _ { 1 } ( t _ { 2 } , t _ { 3 } ) , y _ { 2 } = y _ { 2 } ( t _ { 2 } , t _ { 3 } )$ , leading to many simplifications. We obtain

$$
\Delta _ { t _ { 1 } } y _ { 3 } + \partial _ { t _ { 2 } } y _ { 2 } + \partial _ { t _ { 3 } } y _ { 1 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } y _ { 1 } = - \eta x _ { 1 } x _ { 2 } = - \eta ( - 1 ) ^ { t _ { 1 } } a _ { 1 } \left( - \frac { \eta } { 2 } \left[ \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 \right] + ( - 1 ) ^ { t _ { 1 } } a _ { 2 } \right) .
$$

We consider this as a diference equation for $y _ { 3 }$ in $t _ { 1 }$ .

$$
\Delta _ { t _ { 1 } } y _ { 3 } = - \partial _ { t _ { 2 } } y _ { 2 } - \partial _ { t _ { 3 } } y _ { 1 } - \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } y _ { 1 } - \eta ( - 1 ) ^ { t _ { 1 } } a _ { 1 } \left( - \frac { \eta } { 2 } \left[ \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 \right] + ( - 1 ) ^ { t _ { 1 } } a _ { 2 } \right) .
$$

The right hand side depends on $t _ { 1 }$ only through the $( - 1 ) ^ { t _ { 1 } }$ terms. To avoid a secular term, we must have

$$
\partial _ { t _ { 2 } } y _ { 2 } + \partial _ { t _ { 3 } } y _ { 1 } + \frac { 1 } { 2 } \partial _ { t _ { 2 } } ^ { 2 } y _ { 1 } = - \eta a _ { 1 } a _ { 2 } .\tag{A.2}
$$

Recall that we seek to compute

$$
\partial _ { t _ { 3 } } E = y _ { 1 } \partial _ { t _ { 3 } } y _ { 1 } + \left( { \frac { a _ { 1 } } { 2 } } - { \frac { 1 } { a _ { 1 } } } \right) \partial _ { t _ { 3 } } a _ { 1 } .
$$

This suggests considering the quantity $\begin{array} { r } { F = y _ { 1 } y _ { 2 } + \left( \frac { a _ { 1 } } { 2 } - \frac { 1 } { a _ { 1 } } \right) a _ { 2 } } \end{array}$ . Indeed, using Eqs. (A.1) and (A.2), we have

$$
\begin{array} { l } { { \partial _ { t _ { 2 } } F = ( \partial _ { t _ { 2 } } y _ { 1 } ) y _ { 2 } + y _ { 1 } ( \partial _ { t _ { 2 } } y _ { 2 } ) + ( \displaystyle \frac 1 2 + \frac 1 { a _ { 1 } ^ { 2 } } ) ( \partial _ { t _ { 2 } } a _ { 1 } ) a _ { 2 } + ( \displaystyle \frac { a _ { 1 } } { 2 } - \frac 1 { a _ { 1 } } ) ( \partial _ { t _ { 2 } } a _ { 2 } ) } } \\ { { \ = \eta ( 1 - \displaystyle \frac { a _ { 1 } ^ { 2 } } { 2 } ) y _ { 2 } - y _ { 1 } ( \partial _ { t _ { 3 } } y _ { 1 } + \frac 1 2 \partial _ { t _ { 2 } } ^ { 2 } y _ { 1 } + \eta a _ { 1 } a _ { 2 } ) + ( \displaystyle \frac 1 2 + \frac 1 { a _ { 1 } ^ { 2 } } ) ( \eta a _ { 1 } y _ { 1 } ) a _ { 2 } } } \\ { { \ ~ + ( \displaystyle \frac { a _ { 1 } } { 2 } - \frac 1 { a _ { 1 } } ) ( - \partial _ { t _ { 3 } } a _ { 1 } - \frac 1 2 \partial _ { t _ { 2 } } ^ { 2 } a _ { 1 } - \eta ^ { 2 } \displaystyle \frac 3 2 \zeta a _ { 1 } [ \displaystyle \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 ] + \eta ( a _ { 1 } y _ { 2 } + a _ { 2 } y _ { 1 } ) ) } } \\   \ = - \partial _ { t _ { 3 } } E - \displaystyle \frac { y _ { 1 } } { 2 } \partial _ { t _ { 2 } } ^ { 2 } y _ { 1 } - ( \displaystyle \frac { a _ { 1 } } { 2 } - \frac 1 { a _ { 1 } } ) \displaystyle \frac 1 2 \partial _ { t _ { 2 } } ^ { 2 } a _ { 1 } - \eta ^ { 2 } \displaystyle \frac 3 2 \zeta ( \displaystyle \frac { a _ { 1 } ^ { 2 } } { 2 } - 1 ) ( \displaystyle \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - \end{array}
$$

The right-hand side depends on t<sub>2</sub> only through the functions $a _ { 1 }$ and $y _ { 1 }$ , which are periodic in $t _ { 2 }$ . We then use the following lemma.

Lemma A.1. Let $t \mapsto \varphi ( t )$ be a periodic real function. Then $\varphi ( t )$ has a bounded primitive $i f$ and only if the average $o f \varphi$ is 0.

To avoid a secular term in $F$ (which would induce a secular term in $y _ { 2 } \ \mathrm { o r } \ a _ { 2 } )$ , we must have

$$
\partial _ { t _ { 3 } } E = \left. - \frac { y _ { 1 } } { 2 } \partial _ { t _ { 2 } } ^ { 2 } y _ { 1 } + \left( \frac { 1 } { 2 a _ { 1 } } - \frac { a _ { 1 } } { 4 } \right) \partial _ { t _ { 2 } } ^ { 2 } a _ { 1 } - \eta ^ { 2 } \frac { 3 } { 2 } \zeta \left( \frac { a _ { 1 } ^ { 2 } } { 2 } - 1 \right) \left( \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 \right) \right. _ { t _ { 2 } } .
$$

We now simplify this expression. We compute

$$
\begin{array} { l } { { \partial _ { t _ { 2 } } ^ { 2 } y _ { 1 } = \eta \partial _ { t _ { 2 } } \left( 1 - \frac { a _ { 1 } ^ { 2 } } { 2 } \right) = - \eta a _ { 1 } \partial _ { t _ { 2 } } a _ { 1 } = - \eta ^ { 2 } a _ { 1 } ^ { 2 } y _ { 1 } , } } \\ { { \partial _ { t _ { 2 } } ^ { 2 } a _ { 1 } = \eta \partial _ { t _ { 2 } } ( a _ { 1 } y _ { 1 } ) = \eta \left[ ( \partial _ { t _ { 2 } } a _ { 1 } ) y _ { 1 } + a _ { 1 } ( \partial _ { t _ { 2 } } y _ { 1 } ) \right] = \eta ^ { 2 } \left[ a _ { 1 } y _ { 1 } ^ { 2 } + a _ { 1 } \left( 1 - \frac { a _ { 1 } ^ { 2 } } { 2 } \right) \right] . } } \end{array}
$$

This gives

$$
\begin{array} { c l } { \displaystyle \partial _ { t s } E = \eta ^ { 2 }  \frac { a _ { 1 } ^ { 2 } } { 2 } y _ { 1 } ^ { 2 } + ( \frac { 1 } { 2 a _ { 1 } } - \frac { a _ { 1 } } { 4 } ) [ a _ { 1 } y _ { 1 } ^ { 2 } + a _ { 1 } - \frac { a _ { 1 } ^ { 3 } } { 2 } ] - \frac { 3 } { 2 } \zeta ( \frac { a _ { 1 } ^ { 2 } } { 2 } - 1 ) ( \frac { 3 } { 2 } \zeta a _ { 1 } ^ { 2 } - 1 )  _ { t _ { 2 } } } \\ { \displaystyle = \eta ^ { 2 }  \frac { a _ { 1 } ^ { 2 } } { 2 } y _ { 1 } ^ { 2 } + \frac { y _ { 1 } ^ { 2 } } { 2 } + \frac { 1 } { 2 } - \frac { a _ { 1 } ^ { 2 } } { 4 } - \frac { a _ { 1 } ^ { 2 } } { 4 } y _ { 1 } ^ { 2 } - \frac { a _ { 1 } ^ { 2 } } { 4 } + \frac { a _ { 1 } ^ { 4 } } { 8 } - \frac { 9 } { 8 } \zeta ^ { 2 } a _ { 1 } ^ { 4 } + \frac { 3 } { 4 } \zeta a _ { 1 } ^ { 2 } + \frac { 9 } { 4 } \zeta ^ { 2 } a _ { 1 } ^ { 2 } - \frac { 3 } { 2 } \zeta  _ { t _ { 2 } } } \\  \displaystyle = \eta ^ { 2 } ( \frac { \langle a _ { 1 } ^ { 2 } y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } } { 4 } + \frac { \langle y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } } { 2 } + \frac { 1 } { 2 } - \frac { 3 } { 2 } \zeta + ( - \frac { 1 } { 2 } + \frac { 3 \zeta } { 4 } + \frac { 9 \zeta ^ { 2 } } { 4 } ) \langle a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } + \frac { 1 } { 8 } ( 1 - 9 \zeta ^ { 2 } ) \langle a _ { 1 } ^ { 4 } \end{array}
$$

We now compute the average values that appear above. Note that, as $y _ { 1 }$ is periodic in $t _ { 2 }$

$$
0 = \langle \partial _ { t _ { 2 } } y _ { 1 } \rangle _ { t _ { 2 } } = \eta \left( 1 - \frac { \langle a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } } { 2 } \right) ,
$$

thus $\langle a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } = 2 .$ . Similarly,

$$
0 = \left. \partial _ { t _ { 2 } } ( y _ { 1 } ^ { 3 } ) \right. _ { t _ { 2 } } = 3 \left. y _ { 1 } ^ { 2 } \partial _ { t _ { 2 } } y _ { 1 } \right. _ { t _ { 2 } } = 3 \eta \left. y _ { 1 } ^ { 2 } \left( 1 - \frac { a _ { 1 } ^ { 2 } } { 2 } \right) \right. _ { t _ { 2 } } = 3 \eta \left( \left. y _ { 1 } ^ { 2 } \right. _ { t _ { 2 } } - \frac { \left. y _ { 1 } ^ { 2 } a _ { 1 } ^ { 2 } \right. _ { t _ { 2 } } } { 2 } \right) ,
$$

thus $\langle y _ { 1 } ^ { 2 } a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } = 2 \langle y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } }$ . Finally,

$$
\begin{array} { l } { { 0 = \left. \partial _ { t _ { 2 } } ( a _ { 1 } ^ { 2 } y _ { 1 } ) \right. _ { t _ { 2 } } = \eta \left. 2 a _ { 1 } ( \partial _ { t _ { 2 } } a _ { 1 } ) y _ { 1 } + a _ { 1 } ^ { 2 } \partial _ { t _ { 2 } } y _ { 1 } \right. _ { t _ { 2 } } = \eta \left. 2 a _ { 1 } ^ { 2 } y _ { 1 } ^ { 2 } + a _ { 1 } ^ { 2 } \left( 1 - \frac { a _ { 1 } ^ { 2 } } { 2 } \right) \right. _ { t _ { 2 } } } } \\ { { \ = \eta \left[ 2 \left. a _ { 1 } ^ { 2 } y _ { 1 } ^ { 2 } \right. _ { t _ { 2 } } + \left. a _ { 1 } ^ { 2 } \right. _ { t _ { 2 } } - \frac { \left. a _ { 1 } ^ { 4 } \right. _ { t _ { 2 } } } { 2 } \right] = \eta \left[ 4 \left. y _ { 1 } ^ { 2 } \right. _ { t _ { 2 } } + 2 - \frac { \left. a _ { 1 } ^ { 4 } \right. _ { t _ { 2 } } } { 2 } \right] , } } \end{array}
$$

thus $\langle a _ { 1 } ^ { 4 } \rangle _ { t _ { 2 } } = 4 + 8 \langle y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } }$ . Overall, we obtain

$$
\begin{array} { c } { { \partial _ { t 3 } E = \eta ^ { 2 } \left[ \displaystyle \frac { \langle y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } } { 2 } + \displaystyle \frac { \langle y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } } { 2 } + \displaystyle \frac { 1 } { 2 } - \frac { 3 } { 2 } \zeta - 1 + \frac { 3 \zeta } { 2 } + \frac { 9 \zeta ^ { 2 } } { 2 } + ( 1 - 9 \zeta ^ { 2 } ) \left( \frac { 1 } { 2 } + \langle y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } \right) \right] } } \\ { { = \eta ^ { 2 } ( 2 - 9 \zeta ^ { 2 } ) \langle y _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } . } } \end{array}
$$

Remark A.2. In the above derivations, ensuring that $\begin{array} { r } { F = y _ { 1 } y _ { 2 } + \left( \frac { a _ { 1 } } { 2 } - \frac { 1 } { a _ { 1 } } \right) a _ { 2 } } \end{array}$ is nonsecular sufices to compute the variations of the energy E. However, ensuring that $y _ { 2 }$ and $a _ { 2 }$ are both non-secular would require a condition on an independent combination of $y _ { 2 }$ and $a _ { 2 }$ . This additional condition would compute the phase shift of the self-stabilization fluctuations in the slow timescale $t _ { 3 }$ . This phase shift can be observed in Fig. 5, bottom plots.

## B Derivation of Result 5.1

As before, we identify terms in the expansion of

$$
\Delta _ { k } x = - \eta \partial _ { x } f ( x , y ) ,\tag{B.1}
$$

$$
\Delta _ { k } y = - \eta \nabla _ { y } f ( x , y ) .\tag{B.2}
$$

The expansion of the discrete diference operator $\Delta _ { k }$ is the same as in Eq. (4.7). The expansion of the right-hand side also remains unchanged until $\mathrm { E q . } \ ( 4 . 4 )$ . Recall that for all $y \in Y$ , we have $g ( 0 , y ) = \operatorname* { m i n } g$ . In particular, this implies that for all $p \geqslant 1 , \nabla _ { y } ^ { p } g ( 0 , y ) = 0$ Moreover, it also implies that for all $y \in Y , \partial _ { x } g ( 0 , y ) = 0$ . As a consequence, for all $p \geqslant 0$ $\nabla _ { y } ^ { p } \partial _ { x } g ( 0 , y ) = 0$ . This simplifies many terms in Eq. (4.4). Combining with the assumption $x _ { 0 } = 0$ and the notation $S ( 0 , y ) = \partial _ { x } ^ { 2 } g ( 0 , y )$ , we obtain

$$
\begin{array} { l } { \displaystyle - \eta \partial _ { x } f ( x , y ) = - \varepsilon ^ { 1 / 2 } \eta S ( 0 , y _ { 0 } ) x _ { 1 } } \\ { \displaystyle - \varepsilon \eta \left[ S ( 0 , y _ { 0 } ) x _ { 2 } + \frac { 1 } { 2 } \partial _ { x } S ( 0 , y _ { 0 } ) x _ { 1 } ^ { 2 } + x _ { 1 } \left. \nabla _ { y } S ( 0 , y _ { 0 } ) , y _ { 1 } \right. + \partial _ { x } h ( 0 , y _ { 0 } ) \right] } \\ { \displaystyle + o ( \varepsilon ) , } \\ { \displaystyle - \eta \nabla _ { y } f ( x , y ) = - \varepsilon \eta \left[ \frac { x _ { 1 } ^ { 2 } } { 2 } \nabla _ { y } S ( 0 , y _ { 0 } ) + \nabla _ { y } h ( 0 , y _ { 0 } ) \right] + o ( \varepsilon ) . } \end{array}\tag{B.3}
$$

In the derivations that follow, we do not write explicitly the evaluations at $( 0 , y _ { 0 } )$ anymore.   
We can now proceed to the identification of terms in the expansion of Eqs. (B.1)–(B.2).

Order 1. As in the derivation of Result 4.1, this order gives $y _ { 0 } = y _ { 0 } ( t _ { 2 } , t _ { 3 } )$

Order $\varepsilon ^ { 1 / 2 }$ . The x equation (B.1) gives

$$
\Delta _ { t _ { 1 } } x _ { 1 } = - \eta S x _ { 1 } .
$$

We distinguish whether we are at the edge of stability.

• If $\eta S < 2$ , recall that, by assumption from Sec. 2, we also have $\eta S > 0$ . Thus $x _ { 1 } \to 0$ as $t _ { 1 } \to \infty$

• If $\eta S = 2$ , then $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ for some a<sub>1</sub>.

As before, the y equation (B.2) gives $y _ { 1 } = y _ { 1 } ( t _ { 2 } , t _ { 3 } )$ and $y _ { 0 } = y _ { 0 } ( t _ { 3 } )$

Order ε. The x equation (B.1) gives

$$
\Delta _ { t _ { 1 } } x _ { 2 } + \partial _ { t _ { 2 } } x _ { 1 } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } x _ { 1 } = - \eta \left[ S x _ { 2 } + \frac { 1 } { 2 } ( \partial _ { x } S ) x _ { 1 } ^ { 2 } + x _ { 1 } \left. \nabla _ { y } S , y _ { 1 } \right. + \partial _ { x } h \right] .
$$

Again, we focus on understanding what happens at the edge of stability, i.e., when $\eta S = 2$ We obtain

$$
\Delta _ { t _ { 1 } } x _ { 2 } + 2 x _ { 2 } = ( - 1 ) ^ { t _ { 1 } } \partial _ { t _ { 2 } } a _ { 1 } - \eta \left[ \frac { 1 } { 2 } ( \partial _ { x } S ) a _ { 1 } ^ { 2 } + ( - 1 ) ^ { t _ { 1 } } a _ { 1 } \left. \nabla _ { y } S , y _ { 1 } \right. + \partial _ { x } h \right] .
$$

By Lemma 4.2, we must have $\partial _ { t _ { 2 } } a _ { 1 } = \eta a _ { 1 } \left. \nabla _ { y } S , y _ { 1 } \right.$ to avoid a secular term. We now turn to the $y$ equation (B.2). It gives

$$
\Delta _ { t _ { 1 } } y _ { 2 } + \partial _ { t _ { 2 } } y _ { 1 } + \partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \frac { x _ { 1 } ^ { 2 } } { 2 } \nabla _ { y } S + \nabla _ { y } h \right] .
$$

We distinguish whether we are at the edge of stability.

• If $\eta S < 2$ , then $x _ { 1 } \to 0$ as $t _ { 1 } \to \infty$ . To prevent $y _ { 2 }$ from developing a secular term on the timescale $t _ { 1 }$ , we must have $\partial _ { t _ { 2 } } y _ { 1 } + \partial _ { t _ { 3 } } y _ { 0 } = - \eta \nabla _ { y } h$ . Here, all quantities but $y _ { 1 }$ are independent of $t _ { 2 }$ . To prevent a secular term, we must have

$$
\partial _ { t _ { 3 } } y _ { 0 } = - \eta \nabla _ { y } h .
$$

In the relative interior of the stable region, the dynamics follow the gradient flow of $h$ constrained to the valley.

• If $\eta S = 2$ , then we have $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 } ( t _ { 2 } , t _ { 3 } )$ . We obtain

$$
\Delta _ { t _ { 1 } } y _ { 2 } = - \partial _ { t _ { 2 } } y _ { 1 } - \partial _ { t _ { 3 } } y _ { 0 } - \eta \left[ \frac { a _ { 1 } ^ { 2 } } { 2 } \nabla _ { y } S + \nabla _ { y } h \right] .
$$

The right-hand side is independent of $t _ { 1 }$ . To avoid a secular term, we must have

$$
\partial _ { t _ { 2 } } y _ { 1 } = - \partial _ { t _ { 3 } } y _ { 0 } - \eta \left[ \frac { a _ { 1 } ^ { 2 } } { 2 } \nabla _ { y } S + \nabla _ { y } h \right] .\tag{B.4}
$$

Diferentiating the edge of stability condition $\eta S ( 0 , y _ { 0 } ) = 2 $ with respect to $t _ { 3 } .$ we obtain $\langle \partial _ { t _ { 3 } } y _ { 0 } , \nabla _ { y } S \rangle = 0$ . This suggests taking the inner product of Eq. (B.4) with $\nabla _ { y } S$ . Recall that we denote $\boldsymbol { u } _ { 1 } = \langle y _ { 1 } , \nabla _ { y } S \rangle$ . We obtain

$$
\partial _ { t _ { 2 } } u _ { 1 } = - \eta \left[ \left\| \nabla _ { y } S \right\| ^ { 2 } \frac { a _ { 1 } ^ { 2 } } { 2 } + \left. \nabla _ { y } h , \nabla _ { y } S \right. \right] .
$$

This finishes the derivation of Eqs. (5.1).

As discussed below the result, the self-stabilization system (5.1) implies that $a _ { 1 }$ is periodic in $t _ { 2 }$ (for fixed $t _ { 3 } )$ . Thus in the right hand side of Eq. (B.4), all quantities are independent of $t _ { 2 }$ except for $a _ { 1 }$ which is periodic in $t _ { 2 }$

By Lemma $\mathrm { A . 1 } .$ to avoid $y _ { 1 }$ developing a secular term in $t _ { 2 }$ , we must have

$$
\partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \nabla _ { y } h + \frac { \langle a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } } { 2 } \nabla _ { y } S \right] .
$$

This completes the derivation of item (i) by taking $\sigma ^ { 2 } = \langle a _ { 1 } ^ { 2 } \rangle _ { t _ { 2 } } { \mathrm { ~ i f ~ } } \eta S = 2$ and $\sigma ^ { 2 } = 0$ if $\eta S < 2$ , and thus completes the derivation of the result.

## C Derivation of Result 6.1

As before, we identify terms in the expansion of

$$
\Delta _ { k } x = - \eta \nabla _ { x } f ( x , y ) ,\tag{C.1}
$$

$$
\Delta _ { k } y = - \eta \nabla _ { y } f ( x , y ) .\tag{C.2}
$$

The expansion of the right hand side is the same as in Eq. (B.3), except for a few modifications due to the fact that X is multidimensional:

$\partial _ { x }$ is replaced by $\nabla _ { x } .$

$S ( 0 , y ) = \partial _ { x } ^ { 2 } g ( 0 , y ) \in \mathbb { R }$ is replaced by $S ( 0 , y ) = \nabla _ { x } ^ { 2 } g ( 0 , y ) \in \mathrm { S y m } _ { + } X$

$\partial _ { x } S ( 0 , y ) \in \mathbb { R }$ is replaced by $D _ { x } S ( 0 , y ) : X \to \operatorname { S y m } X$

$\nabla _ { y } S ( 0 , y ) \in Y$ is replaced by $D _ { y } S ( 0 , y ) : Y \to \operatorname { S y m } X$

We obtain:

$$
\begin{array} { l } { \displaystyle - \eta \nabla _ { x } f ( x , y ) = - \varepsilon ^ { 1 / 2 } \eta S ( 0 , y _ { 0 } ) x _ { 1 } } \\ { \displaystyle - \varepsilon \eta \left[ S ( 0 , y _ { 0 } ) x _ { 2 } + \frac { 1 } { 2 } D _ { x } S ( 0 , y _ { 0 } ) [ x _ { 1 } ] x _ { 1 } + D _ { y } S ( 0 , y _ { 0 } ) [ y _ { 1 } ] x _ { 1 } + \nabla _ { x } h ( 0 , y _ { 0 } ) \right] } \\ { \displaystyle + o ( \varepsilon ) , } \\ { \displaystyle - \eta \nabla _ { y } f ( x , y ) = - \varepsilon \eta \left[ \frac { 1 } { 2 } D _ { y } S ( 0 , y _ { 0 } ) ^ { \ast } [ x _ { 1 } x _ { 1 } ^ { \top } ] + \nabla _ { y } h ( 0 , y _ { 0 } ) \right] + o ( \varepsilon ) . } \end{array}
$$

In the derivations that follow, we do not write explicitly the evaluations at $( 0 , y _ { 0 } )$ anymore.   
We can now proceed to the identification of terms in the expansion of Eqs. (C.1)–(C.2).

Order 1. As before, this order gives $y _ { 0 } = y _ { 0 } ( t _ { 2 } , t _ { 3 } )$

Order $\varepsilon ^ { 1 / 2 }$ . The x equation (C.1) gives

$$
\Delta _ { t _ { 1 } } x _ { 1 } = - \eta S x _ { 1 } .
$$

We distinguish directions depending on whether they are critical. Recall that $C = C ( t _ { 3 } ) =$ ker $( 2 \operatorname { I d } _ { X } - \eta S )$ denotes the critical subspace. We use a spectral decomposition of $\eta { \cal S }$ By assumption from Sec. 2, all eigenvalues are positive. The eigenvectors associated the eigenvalue 2 form a basis of $C _ { i }$ while the other eigenvectors are associated with eigenvalues in (0, 2) and form a basis of $C ^ { \perp }$

Denoting $x _ { 1 } ^ { \parallel } = P _ { C } x _ { 1 }$ and $x _ { 1 } ^ { \perp } = P _ { C ^ { \perp } } x _ { 1 }$ , this gives $x _ { 1 } ^ { \perp } \xrightarrow [ t _ { 1 } \to \infty ] { } 0$ , with $\begin{array} { r l r } {  { \sum _ { t _ { 1 } = 0 } ^ { + \infty } \| x _ { 1 } ^ { \perp } ( t _ { 1 } , t _ { 2 } , t _ { 3 } ) \| < } } \end{array}$ +∞ and $x _ { 1 } ^ { \parallel } = ( - 1 ) ^ { t _ { 1 } } a _ { 1 }$ for some $a _ { 1 } = a _ { 1 } ( t _ { 2 } , t _ { 3 } ) \in C$

As before, the y equation (C.2) gives $y _ { 1 } = y _ { 1 } ( t _ { 2 } , t _ { 3 } )$ and $y _ { 0 } = y _ { 0 } ( t _ { 3 } )$

Order ε. The x equation (C.1) gives

$$
\Delta _ { t _ { 1 } } x _ { 2 } + \partial _ { t _ { 2 } } x _ { 1 } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } x _ { 1 } = - \eta \left[ S x _ { 2 } + \frac { 1 } { 2 } D _ { x } S [ x _ { 1 } ] x _ { 1 } + D _ { y } S [ y _ { 1 } ] x _ { 1 } + \nabla _ { x } h \right] .
$$

We focus on the critical directions, i.e., we apply $P _ { C }$ to both sides of the equation. Denoting $x _ { 2 } ^ { \parallel } = P _ { C } x _ { 2 }$ , we obtain

$$
\Delta _ { t _ { 1 } } x _ { 2 } ^ { \parallel } + \partial _ { t _ { 2 } } x _ { 1 } ^ { \parallel } + \partial _ { t _ { 2 } } \Delta _ { t _ { 1 } } x _ { 1 } ^ { \parallel } = - \eta \left[ P _ { C } S x _ { 2 } + \frac { 1 } { 2 } P _ { C } D _ { x } S [ x _ { 1 } ] x _ { 1 } + P _ { C } D _ { y } S [ y _ { 1 } ] x _ { 1 } + P _ { C } \nabla _ { x } h \right] .
$$

Recall that $C$ is the eigenspace of $\eta S$ associated with the eigenvalue 2, thus $\eta P _ { C } S = 2 P _ { C }$ Moreover, we decompose $x _ { 1 } = \iota _ { C } x _ { 1 } ^ { \parallel } + \iota _ { C ^ { \perp } } x _ { 1 } ^ { \perp } = ( - 1 ) ^ { t _ { 1 } } \iota _ { C } a _ { 1 } + \iota _ { C ^ { \perp } } x _ { 1 } ^ { \perp }$ . This gives

$$
\begin{array} { l } { { \displaystyle \Delta _ { t _ { 1 } } x _ { 2 } ^ { \parallel } + 2 x _ { 2 } ^ { \parallel } = ( - 1 ) ^ { t _ { 1 } } \partial _ { t _ { 2 } } a _ { 1 } - \eta \biggl [ \frac 1 2 P _ { C } D _ { x } S [ \iota _ { C } a _ { 1 } ] \iota _ { C } a _ { 1 } + \frac 1 2 ( - 1 ) ^ { t _ { 1 } } P _ { C } D _ { x } S [ \iota _ { C } \bot x _ { 1 } ^ { \perp } ] \iota _ { C } a _ { 1 } } } \\ { { \displaystyle \qquad + \frac 1 2 ( - 1 ) ^ { t _ { 1 } } P _ { C } D _ { x } S [ \iota _ { C } a _ { 1 } ] \iota _ { C } \bot x _ { 1 } ^ { \perp } + \frac 1 2 P _ { C } D _ { x } S [ \iota _ { C } \bot x _ { 1 } ^ { \perp } ] \iota _ { C } \bot x _ { 1 } ^ { \perp } } } \\ { { \displaystyle \qquad + ( - 1 ) ^ { t _ { 1 } } P _ { C } D _ { y } S [ y _ { 1 } ] \iota _ { C } a _ { 1 } + P _ { C } D _ { y } S [ y _ { 1 } ] \iota _ { C } \bot x _ { 1 } ^ { \perp } + P _ { C } \nabla _ { x } h \biggr ] . } } \end{array}
$$

The right hand side depends on t<sub>1</sub> only through the $( - 1 ) ^ { t _ { 1 } }$ terms and $x _ { 1 } ^ { \perp }$ terms, but the latter are vanishing as $t _ { 1 } \to \infty$ , with $\textstyle \sum _ { t _ { 1 } = 0 } ^ { + \infty } \| x _ { 1 } ^ { \perp } \| < + \infty$ . This suggests studying the following generalization of Lemma 4.2.

Lemma C.1. Let $\alpha , \beta \in \mathbb { R }$ and $\gamma = \gamma ( t _ { 1 } ) \in \mathbb { R }$ such that $\textstyle \sum _ { t _ { 1 } = 0 } ^ { + \infty } | \gamma ( t _ { 1 } ) | < + \infty$ . The solutions $\varphi = \varphi ( t _ { 1 } )$ of the diference equation

$$
\Delta _ { t _ { 1 } } \varphi + 2 \varphi = \alpha + \beta ( - 1 ) ^ { t _ { 1 } } + \gamma ( t _ { 1 } )
$$

are

$$
\varphi = \frac { \alpha } { 2 } + a ( - 1 ) ^ { t _ { 1 } } - \beta t _ { 1 } ( - 1 ) ^ { t _ { 1 } } - \sum _ { s = 0 } ^ { t _ { 1 } - 1 } ( - 1 ) ^ { t _ { 1 } + s } \gamma ( s ) , \qquad a \in \mathbb { R } .
$$

In particular, there exists a bounded solution if and only $i f \beta = 0$

Proof. $\mathrm { A s \ ( - 1 ) } ^ { t _ { 1 } }$ is solution to the homogeneous equation, we seek solutions of the form $\varphi = A ( - 1 ) ^ { t _ { 1 } }$ for some $A = A ( t _ { 1 } )$ . We compute

$$
\begin{array} { r l } & { \Delta _ { t _ { 1 } } \varphi + 2 \varphi = \varphi ( t _ { 1 } + 1 ) - \varphi ( t _ { 1 } ) + 2 \varphi ( t _ { 1 } ) = \varphi ( t _ { 1 } + 1 ) + \varphi ( t _ { 1 } ) } \\ & { \qquad = A ( t _ { 1 } + 1 ) ( - 1 ) ^ { t _ { 1 } + 1 } + A ( t _ { 1 } ) ( - 1 ) ^ { t _ { 1 } } = - ( - 1 ) ^ { t _ { 1 } } \Delta _ { t _ { 1 } } A . } \end{array}
$$

The function $\varphi = A ( - 1 ) ^ { t _ { 1 } }$ solves the diference equation if and only if A solves

$$
\Delta _ { t _ { 1 } } A = - \alpha ( - 1 ) ^ { t _ { 1 } } - \beta - ( - 1 ) ^ { t _ { 1 } } \gamma ( t _ { 1 } ) .
$$

The discrete primitives are

$$
A = a + \frac { \alpha } { 2 } ( - 1 ) ^ { t _ { 1 } } - \beta t _ { 1 } - \sum _ { s = 0 } ^ { t _ { 1 } - 1 } ( - 1 ) ^ { s } \gamma ( s ) , \qquad a \in \mathbb { R } .
$$

This gives the result.

Applying Lemma C.1 to the equation above, we obtain that to avoid a secular term, we must have

$$
\partial _ { t _ { 2 } } a _ { 1 } = \eta P _ { C } D _ { y } S [ y _ { 1 } ] \iota _ { C } a _ { 1 } = \eta \Psi [ y _ { 1 } ] a _ { 1 } = \eta U _ { 1 } a _ { 1 } .
$$

We now turn to the $y$ equation (C.2). It gives

$$
\Delta _ { t _ { 1 } } y _ { 2 } + \partial _ { t _ { 2 } } y _ { 1 } + \partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \frac { 1 } { 2 } ( D _ { y } S ) ^ { * } [ x _ { 1 } x _ { 1 } ^ { \top } ] + \nabla _ { y } h \right] .
$$

Again, we decompose $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } \iota _ { C } a _ { 1 } + \iota _ { C ^ { \perp } } x _ { 1 } ^ { \perp }$

$$
\begin{array} { r l } & { \Delta _ { t _ { 1 } } y _ { 2 } + \partial _ { t _ { 2 } } y _ { 1 } + \partial _ { t _ { 3 } } y _ { 0 } } \\ & { \qquad = - \eta \biggl [ \frac { 1 } { 2 } ( D _ { y } S ) ^ { * } [ \iota _ { C } a _ { 1 } a _ { 1 } ^ { \top } P _ { C } ] + \frac { ( - 1 ) ^ { t _ { 1 } } } { 2 } ( D _ { y } S ) ^ { * } \left[ \iota _ { C } a _ { 1 } x _ { 1 } ^ { \bot \top } P _ { C ^ { \bot } } + \iota _ { C ^ { \bot } } x _ { 1 } ^ { \bot } a _ { 1 } ^ { \top } P _ { C } \right] } \\ & { \qquad + \frac { 1 } { 2 } ( D _ { y } S ) ^ { * } [ \iota _ { C ^ { \bot } } x _ { 1 } ^ { \bot } x _ { 1 } ^ { \bot \top } P _ { C ^ { \bot } } ] + \nabla _ { y } h \biggr ] . } \end{array}
$$

Here, the right hand side depends on $t _ { 1 }$ only through the $( - 1 ) ^ { t _ { 1 } }$ terms and $x _ { 1 } ^ { \perp }$ terms, but the latter are vanishing as $t _ { 1 } \to \infty$ . To avoid a secular term, we must have

$$
\partial _ { t _ { 2 } } y _ { 1 } + \partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ { \frac { 1 } { 2 } } ( D _ { y } S ) ^ { * } [ \iota _ { C } a _ { 1 } a _ { 1 } ^ { \top } P _ { C } ] + \nabla _ { y } h \right] .
$$

Note that for $M \in { \mathrm { S y } }$ m $C , \Psi ^ { * } [ M ] = ( D _ { y } S ) ^ { * } [ \iota _ { C } M P _ { C } ]$ . Thus we have

$$
\partial _ { t _ { 2 } } y _ { 1 } + \partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \frac { 1 } { 2 } \Psi ^ { * } [ a _ { 1 } a _ { 1 } ^ { \top } ] + \nabla _ { y } h \right] .\tag{C.3}
$$

Here, we want to use that the evolution of $S ( 0 , y _ { 0 } )$ is constrained in the critical subspace $C$ We use the following lemma.

Lemma C.2. Let $t \mapsto M ( t ) \succeq 0$ be a smooth curve of positive semidefinite matrices, $P =$ $P ( t )$ denote the orthogonal projection onto ker $M ( t )$ , and $\iota = \iota ( t ) = P ^ { * }$ the corresponding inclusion operator. Then, for all $t ,$ we have $P \partial _ { t } M \iota = 0$

Proof. Let $a \in$ ker $M ( t )$ . Then $a ^ { \top } M ( t ) a = 0$ . Moreover, for all $s ,$ we have $a ^ { \top } M ( s ) a \geqslant 0$ We thus have, for all $a \in$ ker $M ( t ) , \partial _ { t } ( a ^ { \top } M ( t ) a ) = a ^ { \top } \partial _ { t } M ( t ) a = 0$

Now, consider $a , b \in$ ker $M ( t )$ . Then, by polarization $( \partial _ { t } M ( t )$ is symmetric),

$$
2 a ^ { \mathsf { T } } \partial _ { t } M ( t ) b = ( a + b ) ^ { \top } \partial _ { t } M ( t ) ( a + b ) - a ^ { \top } \partial _ { t } M ( t ) a - b ^ { \top } \partial _ { t } M ( t ) b = 0 .
$$

Thus $P \partial _ { t } M \iota = 0$

Here, we apply the lemma to $M ( t _ { 3 } ) = 2 \operatorname { I d } _ { X } - \eta S ( 0 , y _ { 0 } ( t _ { 3 } ) )$ . We obtain

$$
0 = P _ { C } \partial _ { t _ { 3 } } S \iota _ { C } = P _ { C } D _ { y } S [ \partial _ { t _ { 3 } } y _ { 0 } ] \iota _ { C } = \Psi [ \partial _ { t _ { 3 } } y _ { 0 } ] .
$$

This suggests evaluating $\Psi$ at $\operatorname { E q }$ . (C.3). We obtain

$$
\partial _ { t _ { 2 } } U _ { 1 } = - \eta \left[ \frac { 1 } { 2 } \Psi \Psi ^ { * } [ a _ { 1 } a _ { 1 } ^ { \top } ] + \Psi [ \nabla _ { y } h ] \right] .
$$

This completes the derivation of the self-stabilization system (6.1).

We do not know whether the self-stabilization system induces periodic solutions in $t _ { 2 }$ when more than one eigenvalue is on the edge of stability. To give a meaning to the average $\langle . \rangle _ { t _ { 2 } }$ , we take the limit $T _ { 2 } \to \infty$ of the average over $[ 0 , T _ { 2 } ]$ (if the limit exists). For instance, to avoid a secular term, y<sub>1</sub> is uniformly bounded in $t _ { 2 } .$ thus we have

$$
\langle \partial _ { t _ { 2 } } y _ { 1 } \rangle _ { t _ { 2 } } = \operatorname* { l i m } _ { T _ { 2 } \to \infty } \frac { 1 } { T _ { 2 } } \int _ { 0 } ^ { T _ { 2 } } \partial _ { t _ { 2 } } y _ { 1 } \mathrm { d } t _ { 2 } = \operatorname* { l i m } _ { T _ { 2 } \to \infty } \frac { y _ { 1 } ( T _ { 2 } , t _ { 3 } ) - y _ { 1 } ( 0 , t _ { 3 } ) } { T _ { 2 } } = 0 .
$$

Taking the average of Eq. (C.3) in $t _ { 2 }$ , we obtain

$$
\partial _ { t _ { 3 } } y _ { 0 } = - \eta \left[ \nabla _ { y } h + \frac { 1 } { 2 } \langle \Psi ^ { * } [ a _ { 1 } a _ { 1 } ^ { \top } ] \rangle _ { t _ { 2 } } \right] .
$$

Provided that the average $\langle a _ { 1 } a _ { 1 } ^ { \top } \rangle _ { t _ { 2 } }$ is well-defined, we have

$$
\langle \Psi ^ { * } [ a _ { 1 } a _ { 1 } ^ { \top } ] \rangle _ { t _ { 2 } } = \Psi ^ { * } [ \langle a _ { 1 } a _ { 1 } ^ { \top } \rangle _ { t _ { 2 } } ] = ( D _ { y } S ) ^ { * } [ \iota _ { C } \langle a _ { 1 } a _ { 1 } ^ { \top } \rangle _ { t _ { 2 } } P _ { C } ] ,
$$

thus the result holds with $\Sigma = \iota _ { C } \langle a _ { 1 } a _ { 1 } ^ { \intercal } \rangle _ { t _ { 2 } } P _ { C }$ , which is complementary to $2 \operatorname { I d } _ { X } - \eta S$ (by definition of $C = \ker ( 2 \operatorname { I d } _ { X } - \eta S ) )$ . If we do not make the assumption that the average $\langle a _ { 1 } a _ { 1 } ^ { \top } \rangle _ { t _ { 2 } }$ is well-defined, we use that $a _ { 1 }$ is bounded in $t _ { 2 }$ (it is non-secular) and thus the averages $\begin{array} { r } { \frac { 1 } { T _ { 2 } } \int _ { 0 } ^ { T _ { 2 } } a _ { 1 } a _ { 1 } ^ { \top } \mathrm { d } t _ { 2 } } \end{array}$ have an accumulation point $\widetilde \Sigma \in \mathrm { S y m } _ { + } C$ as $T _ { 2 } \to \infty$ . We then have $\langle \Psi ^ { * } [ a _ { 1 } a _ { 1 } ^ { \top } ] \rangle _ { t _ { 2 } } = \Psi ^ { * } [ \widetilde { \Sigma } ]$ , and we can define $\Sigma = \iota _ { C } \widetilde { \Sigma } P _ { C } \in \mathrm { S y m } _ { + } X$ . This gives the derivation of item (i) and completes the derivation of Result 6.1.

## D Derivations of Corollary 6.2

We compute

$$
\begin{array} { r l } & { f ( x , y ) = g ( x , y ) + \varepsilon h ( x , y ) } \\ & { \qquad = g ( \varepsilon ^ { 1 / 2 } x _ { 1 } + \varepsilon x _ { 2 } + o ( \varepsilon ) , y _ { 0 } + \varepsilon ^ { 1 / 2 } y _ { 1 } + \varepsilon y _ { 2 } + o ( \varepsilon ) ) + \varepsilon h ( o ( 1 ) , y _ { 0 } + o ( 1 ) ) } \\ & { \qquad = g ( 0 , y _ { 0 } ) +  \nabla _ { x } g ( 0 , y _ { 0 } ) , \varepsilon ^ { 1 / 2 } x _ { 1 } + \varepsilon x _ { 2 }  +  \nabla _ { y } g ( 0 , y _ { 0 } ) , \varepsilon ^ { 1 / 2 } y _ { 1 } + \varepsilon y _ { 2 }  } \\ & { \qquad +  \frac { 1 } { 2 }  \varepsilon ^ { 1 / 2 } x _ { 1 } , \nabla _ { x } ^ { 2 } g ( 0 , y _ { 0 } ) \varepsilon ^ { 1 / 2 } x _ { 1 }  +  \varepsilon ^ { 1 / 2 } x _ { 1 } , \nabla _ { x } \nabla _ { y } g ( 0 , y _ { 0 } ) \varepsilon ^ { 1 / 2 } y _ { 1 }   } \\ & { \qquad + \frac { 1 } { 2 }  \varepsilon ^ { 1 / 2 } y _ { 1 } , \nabla _ { y } ^ { 2 } g ( 0 , y _ { 0 } ) \varepsilon ^ { 1 / 2 } y _ { 1 }  + \varepsilon h ( 0 , y _ { 0 } ) + o ( \varepsilon ) . } \end{array}
$$

As in the derivations above, we have $\nabla _ { x } g ( 0 , y _ { 0 } ) = 0 , \nabla _ { y } g ( 0 , y _ { 0 } ) = 0 , \nabla _ { x } ^ { 2 } g ( 0 , y _ { 0 } ) = S ( 0 , y _ { 0 } )$ $\nabla _ { x } \nabla _ { y } g ( 0 , y _ { 0 } ) = 0$ , and $\nabla _ { y } ^ { 2 } g ( 0 , y _ { 0 } ) = 0$ . Moreover, we have $g ( 0 , y _ { 0 } ) = \operatorname* { m i n } g .$ . We obtain

$$
f ( x , y ) = \operatorname * { m i n } g + \varepsilon f _ { 2 } + o ( \varepsilon ) , f _ { 2 } = h ( 0 , y _ { 0 } ) + \frac { 1 } { 2 } \langle x _ { 1 } , S ( 0 , y _ { 0 } ) x _ { 1 } \rangle .
$$

Here, we decompose $x _ { 1 } = ( - 1 ) ^ { t _ { 1 } } \iota _ { C } a _ { 1 } + \iota _ { C ^ { \perp } } x _ { 1 } ^ { \perp }$ . As $x _ { 1 } ^ { \perp } \to 0$ as $t _ { 1 } \to \infty$ , and C is the eigenspace of $S ( 0 , y _ { 0 } )$ associated to the eigenvalue $2 / \eta$ , we have

$$
f _ { 2 } \xrightarrow [ t _ { 1 } \to \infty ] { } h ( 0 , y _ { 0 } ) + \frac 1 2 \langle \iota _ { C } a _ { 1 } , S ( 0 , y _ { 0 } ) \iota _ { C } a _ { 1 } \rangle = h ( 0 , y _ { 0 } ) + \frac { \| a _ { 1 } \| ^ { 2 } } \eta .
$$