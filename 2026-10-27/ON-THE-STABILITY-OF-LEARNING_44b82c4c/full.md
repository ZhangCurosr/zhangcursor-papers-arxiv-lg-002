# ON THE STABILITY OF LEARNING

John J. Vastola

Department of Neurobiology, Harvard Medical School   
Department of Psychology and Center for Brain Science, Harvard University   
Kempner Institute for the Study of Natural and Artificial Intelligence, Harvard University   
john vastola@hms.harvard.edu

## ABSTRACT

Learning algorithms are usually analyzed under the assumption that informative gradients are continuously available. But this is not always true—learning signals in biological neural networks, for example, may be sparse, local, and only intermittently available at any given synapse. In this work, we show that if gradient information is noisy and only intermittently available, learning may be unstable. In particular, even if average learning dynamics converge to the correct optimum, variance in learning may be asymptotically infinite. We explore this point in a number of analytically tractable settings, including single-layer and multi-layer linear networks, and derive explicit conditions under which learning is stable. These conditions complement, but are conceptually distinct from, existing conditions related to the stability of stochastic gradient descent. Our work demonstrates that unbiased credit assignment is not enough for reliable learning; sufficiently frequent and informative feedback is also required, which yields an additional constraint on models of biologically plausible learning.

## 1 INTRODUCTION

The claim that the brain does not perform backpropagation (Rumelhart et al., 1986) rests in part on the idea that gradient information may be difficult to communicate. In particular, the gradient of an objective function with respect to any given synaptic weight depends on the synaptic weights of a large number of downstream neurons, and it is not obvious in general how such non-local information could be reliably communicated (Crick, 1989; Lillicrap et al., 2020). This issue has motivated a substantial body of work seeking alternative and more biologically plausible learning rules (Lillicrap et al., 2016; Scellier & Bengio, 2017), as well as schemes for approximately implementing backpropagation in specific contexts (Sacramento et al., 2018; Whittington & Bogacz, 2019; Millidge et al., 2022). It remains unknown if and how this information is communicated, but it is generally agreed that communicating it accurately would be difficult given locality constraints.

In this paper, we consider a related but conceptually distinct issue. If gradient information can be communicated, it is not at all obvious that it can be communicated reliably. It is plausible that a scheme which delivers the correct updates on average only delivers them occasionally, to different neurons at different times, and perhaps somewhat noise-corrupted. How do learning dynamics change when gradients are noisy and only intermittently available? We argue that learning changes substantially, and in fact can become unstable if updates are not delivered sufficiently frequently.

This idea is interesting for a few reasons. First, it implies that credit assignment which is correct on average is insufficient, since there remains the additional obstacle that weight updates cannot be too variable. Second, instability is not a generic consequence of noisy weight dynamics, but rather depends crucially on the multiplicative, sparse nature of the noise introduced by assuming intermittent gradient arrivals.

We show that the severity of this issue can be reduced by a few obvious noise mitigation strategies, but that the presence of the issue (i.e., learning becoming unstable if feedback is sufficiently sparse) survives. This problem exists in both linear and nonlinear networks; in shallow and deep networks; and after averaging parameters or recent gradients. The details of the issue change, but the core finding remains: an informative signal must arrive sufficiently often for learning to remain stable.

We proceed as follows. We begin by motivating the simplest version of our model. Next, by analyzing it in detail, we show that there exist gradient-variance-dependent stability issues that are exacerbated as informative feedback becomes less frequent. We then discuss simple examples of this issue near the fixed points of more complex networks, and the extent to which basic noise mitigation strategies address it.

## 2 SPARSE GRADIENT ARRIVALS CAN AMPLIFY ERRORS

![](images/a57a50b0e282fc51f4dd9d850ec7151fe7ce358df4feaa995327a39cef92ca70.jpg)

![](images/bc9f129acf3ab6787a1bca082f13bb76cea496376042a714eb06c1d002fdf811.jpg)

![](images/22cb2f4dbf268d63a2e13255bfbe3d4b0c212717601cd6dcbc38981175c94005.jpg)

![](images/99b50fc7d1a3d9153c012d74dfcdc48b37c837577b3de269a4224dc401dcc182.jpg)  
Figure 1: Sparse gradient arrivals: motivation and intuition. (a) A simple example of sparse gradient arrivals: a weight receives intermittent updates, in between which the regularization term produces relaxation toward zero. Three example stochastic simulations are shown. (b) Error as a function of number of arrivals in a toy model. If $s / \rho \ < \ 1$ , error always decreases (left); if $1 < s / \rho < 2$ , error alternates in sign but decreases in magnitude (center); and if $s / \rho > 2$ , learning becomes unstable, and error both alternates in sign and increases in magnitude (right).

Suppose we are interested in updating a set of weights $\boldsymbol { w } \in \mathbb { R } ^ { N }$ to optimize an objective

$$
\tilde { J } ( \pmb { w } ) : = J ( \pmb { w } ) + \frac { \gamma } { 2 } \| \pmb { w } \| _ { 2 } ^ { 2 }\tag{1}
$$

where $J$ is a task-related objective and the second term regularizes w. We explicitly separate $J$ and the regularization term because their biological interpretation in what follows is different. In the gradient-flow limit typical of many theoretical treatments of learning (Saxe et al., 2013; Bordelon et al., 2020; Domine et al., 2025), gradient descent involves updates of the form

$$
\frac { d } { d t } \pmb { w } = \eta \left[ - \frac { \partial J } { \partial \pmb { w } } - \gamma \pmb { w } \right]\tag{2}
$$

where $\eta > 0$ is the learning rate and $t \in [ 0 , \infty )$ is the amount of elapsed training time. We will investigate the consequences of assuming w only has intermittent access to the correct gradient. We do this as follows. We model sparse gradient arrivals via a Poisson process (Kingman, 1992): we assume that each synapse has a fixed probability per unit time of receiving a noise-corrupted version of the relevant gradient information. In doing things this way, we ignore potential temporal structure in gradient arrivals and correlations between the information arriving at different synapses; accounting for these factors will likely complexify the resulting model, and more importantly, make the associated stability issues worse rather than better.

Poisson gradient arrivals assumption. To model temporally sparse updates, we assume gradients are coupled to a Poisson process with mean arrival rate $\eta \rho ,$ , where $\rho > 0$ is the non-dimensionalized arrival rate. As $\rho$ increases, neurons tend to receive more gradient arrivals per learning time scale $1 / \eta .$ . If we denote a realization of this Poisson process by $\pmb { r } ( t ) \in \mathbb { R } ^ { N }$ , our model corresponds to the replacement

$$
- \frac { \partial J } { \partial w }  r ( t ) \odot \frac { 1 } { \rho } [ - \frac { \partial J } { \partial w } + \pmb { \xi } ]\tag{3}
$$

where $\odot$ denotes the Hadamard (or elementwise) product. The quantity $\pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { \mathrm { o b s } } ^ { 2 } I )$ represents observation noise. Our main motivation for including it is to illustrate how, unlike sparse gradient arrivals, this kind of noise is comparatively benign.

According to this prescription, weight changes due to gradient arrivals have mean and covariance

$$
\mathbb { E } [ \Delta \mathbf { { w } } ] = - \eta \Delta t \frac { \partial J } { \partial \mathbf { { w } } }
$$

$$
\mathrm { C o v } ( \Delta w _ { i } , \Delta w _ { j } ) = \frac { \eta \Delta t } { \rho } \delta _ { i j } \left[ \left( \frac { \partial J } { \partial w } \right) ^ { 2 } + \sigma _ { \mathrm { o b s } } ^ { 2 } \right] .\tag{4}
$$

Useful intuition. We can make a simple heuristic argument to explain why sparse gradient arrivals can destabilize learning. Consider gradient descent (Eq. 2) with only one weight and no regularization $( \gamma = 0 )$ on a mean-squared error objective

$$
J : = \frac { s } { 2 } ( w - w ^ { * } ) ^ { 2 } .\tag{5}
$$

For Poisson gradient arrivals, weights do not change until there is an arrival, so learning effectively occurs in discrete time. The error $e _ { n } : = w _ { n } - w ^ { * }$ after n gradient arrivals evolves according to the recurrence relation

$$
e _ { n + 1 } = \left( 1 - \frac { s } { \rho } \right) e _ { n } .\tag{6}
$$

These dynamics yield a geometric series with $e _ { n } = ( 1 - s / \rho ) ^ { n } e _ { 0 }$ which has three regimes. For a sufficiently large arrival rate $( s / \rho < 1 )$ , error consistently decreases and asymptotically approaches zero (Fig. 1b, left). For a moderately large arrival rate $( 1 < s / \rho < 2 )$ , weight dynamics overshoots its target and the error alternates between positive and negative; it still asymptotically approaches zero (Fig. 1b, center). If the arrival rate is too small $( s / \rho > 2 )$ , learning becomes unstable, and error both alternates in sign and increases in magnitude (Fig. 1b, right). This indicates that, at least for this toy model, the ratio $s / \rho$ between input variance and the gradient arrival rate controls whether learning is stable. Hence, the arrival rate $\rho$ must be larger than

$$
\boxed { \rho > \rho _ { c } : = \frac { s } { 2 } } .\tag{7}
$$

We will see that more complex models exhibit surprisingly similar behavior, and in particular lead to similar expressions for the critical arrival rate $\rho _ { c }$

## 3 INSTABILITY IN A LINEAR MODEL<sup>w</sup>

![](images/5109dfe779b5effee51ce3246bf8671a46ab88be28f10372266e0bfda00eff54.jpg)

![](images/3c7d457748193cb127cad0d7a72e3a39bfd56d74eb82bb99fc5683395efb92a6.jpg)

![](images/ed262c4b361fe0c4cc6e618c6d220b718d791e0220f447e6ef18105f744b01e8.jpg)

![](images/c7d4e32a73a4e0c9ebe50ce4c92b0e4c3bdb7306843679378c968303b2122a28.jpg)  
<sup><</sup> <sup><</sup> <sup>ρ</sup> Figure 2: Instability in a linear model. (a) Average dynamics are unconditionally stable and do not depend on the arrival rate $\rho .$ <sub>s</sub><sup>/ γ</sup>(b) Output variance converges more slowly as $\rho$ decreases and diverges to infinity when $\rho < \rho _ { c } .$ <(c) The steady state output variance diverges like $1 / ( \rho - \rho _ { c } )$ . (d) $\rho _ { c }$ decreases as $\gamma$ increases, so one way to improve stability is to increase the regularization strength.

In this section, we study instability due to sparse gradient arrivals in a tractable linear model, in part because we think it reveals the essence of the phenomenon. Consider an objective

$$
\tilde { J } = \mathbb { E } _ { x } \left\{ \frac { \| \boldsymbol { w } ^ { T } \boldsymbol { x } - ( \boldsymbol { w } ^ { * } ) ^ { T } \boldsymbol { x } \| _ { 2 } ^ { 2 } } { 2 } \right\} + \frac { \gamma } { 2 } \| \boldsymbol { w } \| _ { 2 } ^ { 2 } = \frac { 1 } { 2 } ( \boldsymbol { w } - \boldsymbol { w } ^ { * } ) ^ { T } \boldsymbol { S } ( \boldsymbol { w } - \boldsymbol { w } ^ { * } ) + \frac { \gamma } { 2 } \| \boldsymbol { w } \| _ { 2 } ^ { 2 }\tag{8}
$$

where the input x is assumed to have mean zero and covariance $\pmb { S } \in \mathbb { R } ^ { N \times N }$ . The negative gradient of J equals $\bar { \pmb { S } } ( \pmb { w } ^ { * } - \pmb { w } )$ , so our prescription yields dynamics

$$
\frac { d } { d t } \pmb { w } = \pmb { r } \odot \frac { 1 } { \rho } [ \pmb { S } ( \pmb { w } ^ { * } - \pmb { w } ) + \pmb { \xi } ] - \eta \gamma \pmb { w } .\tag{9}
$$

The average $\bar { \pmb w } : = \mathbb { E } [ \pmb { w } ]$ evolves according to standard-looking linear dynamics

$$
\frac { d } { d t } \bar { \pmb { w } } = \eta \left[ S ( \pmb { w } ^ { * } - \bar { \pmb { w } } ) - \gamma \bar { \pmb { w } } \right]\tag{10}
$$

which approaches the fixed point at $\pmb { w } ^ { \prime } : = ( \pmb { S } + \gamma \pmb { I } ) ^ { - 1 } \pmb { S } \pmb { w } ^ { \ast }$ exponentially quickly. On the other hand, the dynamics of the covariance matrix $\pmb { C } : = \mathbb { E } [ ( \pmb { w } - \pmb { \bar { w } } ) ( \mathbf { \bar { w } } - \pmb { \bar { w } } ) ^ { T } ]$ are substantially more interesting (see Appendix A):

$$
\left| \begin{array} { l } { { \frac { 1 } { \eta } \frac { d } { d t } C = - 2 \gamma C - C S - S C + \frac { 1 } { \rho } \sigma _ { \mathrm { o b s } } ^ { 2 } I + \frac { 1 } { \rho } \mathrm { d i a g } ( \bar { g } _ { i } ^ { 2 } + [ S C S ] _ { i i } ) } } \end{array} \right| .\tag{11}
$$

We can assess stability by determining the conditions under which lim $\scriptstyle 1 _ { t \to \infty } \operatorname { t r } [ C ( t ) ]$ is finite. While this is difficult to do analytically in general, it is possible to do if we make at least one additional assumption about S: either that (i) S is diagonal, or that (ii) S is circulant.

## 3.1 STABILITY GIVEN A DIAGONAL INPUT COVARIANCE

If S is diagonal, Eq. 11 simplifies since $[ S C S ] _ { i i } = S _ { i i } ^ { 2 } C _ { i i }$ . Let us simplify notation by writing $c _ { i } : = C _ { i i }$ and $s _ { i } : = S _ { i i }$ . The diagonal elements of $C$ then evolve according to

$$
\frac { 1 } { \eta } \frac { d } { d t } c _ { i } = \frac { 1 } { \rho } ( \sigma _ { \mathrm { o b s } } ^ { 2 } + \bar { g } _ { i } ^ { 2 } ) - \left( 2 \gamma c _ { i } + 2 s _ { i } - \frac { s _ { i } ^ { 2 } } { \rho } \right) c _ { i } .\tag{12}
$$

These dynamics are linear, and whether $c _ { i }$ converges to a stable value or diverges to infinity depends on the coefficient of $c _ { i ^ { - } }$ —that is, it must be negative. For learning to be stable, the coefficients of all $c _ { i }$ must be negative; hence, it must be true that

$$
\boxed { \rho > \rho _ { c } ^ { i } : = \frac { s _ { i } ^ { 2 } } { 2 ( s _ { i } + \gamma ) } }\tag{13}
$$

for all i. Equivalently, $\rho > \rho _ { c } : = \operatorname* { m a x } _ { i } \rho _ { c } ^ { i } .$ , which depends on the largest input variance $s _ { i } .$ Note that this condition reduces to the critical arrival rate associated with our naive model $\left( \mathrm { E q . ~ } 7 \right)$ when we take $\gamma  0$

As $\rho$ decreases, $c _ { i }$ converges increasingly slowly, and below $\rho _ { c }$ it becomes unstable (Fig. 2b). The steady state diagonal covariance elements diverge like $1 / ( \rho - \rho _ { c } ^ { i } )$ (Fig. 2c), or more quantitatively,

$$
c _ { i } = \frac { 1 } { \rho - \rho _ { c } ^ { i } } \frac { \sigma _ { \mathrm { o b s } } ^ { 2 } + \bar { g } _ { i } ^ { 2 } } { 2 ( s _ { i } + \gamma ) } .\tag{14}
$$

Our formula indicates that the critical arrival rate threshold $\rho _ { c }$ can be reduced in two ways: one can either decrease input variance or increase the regularization strength $\gamma \left( \mathrm { F i g . \ 2 d } \right)$ . The latter method recalls the standard bias-variance tradeoff (Bishop, 2006): we can decrease the (infinite) variance at the cost of increasing bias, since increasing γ makes w more different from the target weight $\boldsymbol { w } ^ { * }$

## 3.2 STABILITY GIVEN A CIRCULANT INPUT COVARIANCE

Suppose instead that S is circulant. Then S can be diagonalized via a discrete Fourier basis, i.e., $S \stackrel { \triangledown } { = } F \Lambda F ^ { \dagger }$ (see Appendix A). The arrival-related term can be written in terms of the covariance matrix in that basis, $\tilde { C } : = F ^ { \dagger } C F$ , since

$$
\frac { 1 } { N } \sum _ { i } [ S C S ] _ { i i } = \frac { 1 } { N } \sum _ { q } s _ { q } ^ { 2 } \tilde { C } _ { q q } .\tag{15}
$$

In this case, different ${ \tilde { C } } _ { q q }$ do not decouple, but it is still possible to straightforwardly characterize dynamics. After some algebra, we obtain a critical arrival rate

$$
\boxed { \rho > \rho _ { c } : = \cfrac { 1 } { 2 N } \sum _ { q } \frac { s _ { q } ^ { 2 } } { s _ { q } + \gamma } } .\tag{16}
$$

Unlike the previous result, this one involves the entire spectrum of $_ { s }$ (although it reduces to the previous one when all $s _ { q }$ are the same). Still, the result conveys the same qualitative message: $\rho _ { c }$ increases as the input variance increases, and decreases as the regularization strength γ increases.

![](images/9fdbdf4b170df5db7a98f748f8d50d5b36599cad0ffbd6a54e42df30015d09f1.jpg)  
b1ρ

![](images/eae17504eda87b8b17d7461aa29e461178ef49f93edea2851c3d66f608f48c33.jpg)

![](images/4185da75f52e7f68dab1c3eafcf06ee0692fc9a88b384116d2e1496bbe21e9cb.jpg)

![](images/4dadf0b5511b866fb7a2c9648cac502c5864befa7b31f78f321abea194f6aded.jpg)  
<sup><</sup> v<sup>a i</sup> <sup>< )</sup> t<sup>)</sup> 0<sup>)</sup> Figure 3: Instability in a two-layer network. (a) We trained a two-layer network to reproduce the <sup>ρ</sup> −<sup>11</sup> <sub>a</sub><sup>r</sup> <sub>v</sub><sup>a</sup> <sub>v</sub><sup>a</sup>output of an identical teacher network. To ensure our linear model provides a (locally) reasonable ρ<sup>ρ</sup> ) +<sup>1</sup>description of its dynamics, we initialized it near the teacher’s weights. (b) Normalized objective γ< a<sup>r w</sup>and output variance throughout learning. Note that for $\rho ~ < ~ \rho _ { c }$ (dark red), J blows up. For $\rho$ <sup>0</sup> ρ<sup>c</sup> 1somewhat above the threshold (red, orange), learning still looks unstable on this finite time scale. <sup>=</sup> w<sup>i</sup>Learning only appears stable well beyond the threshold (green). (c) Instantaneous growth rate of the <sup>ρ</sup>covariance dynamics near the fixed point. The black line indicates the linear model’s prediction and the red dots indicate numerical measurements. Below $\rho = \rho _ { c }$ , this local growth rate can be positive, which indicates instability.

## 4 INSTABILITY IN A NONLINEAR NETWORK

One virtue of our linear model analysis is that it implies sparse gradient arrivals pose problems for much more general networks. Near a stable fixed point $\pmb { w } ^ { * }$ of any well-behaved objective J,

$$
J ( w ) \approx J ( w ^ { * } ) + \frac { 1 } { 2 } ( w - w ^ { * } ) ^ { T } H ( w - w ^ { * } )\tag{17}
$$

where H denotes the (positive definite) Hessian of $J$ at the fixed point $\pmb { w } ^ { * }$ . Hence, gradient descent dynamics are approximately linear near this fixed point, and if we make the identification $S \to H$ they are well-described by our linear model (Eq. 9).

Example: a two-layer network. Although the above mathematical argument is fairly persuasive, it remains helpful to see an explicit nonlinear example where sparse gradient arrivals destabilize learning. For concreteness’ sake, we trained a simple two-layer network to reproduce the output of a teacher network with the same architecture (Fig. 3a). We initialized the student network near the teacher’s weights and assessed how changing the arrival rate $\rho$ affected learning. For simplicity and to speed up computation, only the hidden layer’s biases b and the output parameters v and c were trained; we found that this did not affect our conclusions, however.

We numerically computed the Hessian around the teacher network’s parameters in order to estimate the critical arrival rate $\rho _ { c }$ below which learning ought to be unstable. For $\rho < \rho _ { c } ,$ , we found that the loss function and output variance blew up as learning proceeded (Fig. 3b, c; the dark red line indicates $\rho = 0 . 9 8 \rho _ { c } )$ . In fact, we found that $\rho$ marginally above the critical arrival rate $( \rho = 1 . 2 \rho _ { c } ,$ red; $\rho ~ = ~ 1 . 3 \rho _ { c } .$ , orange) also exhibited unstable learning dynamics on the finite time scales we examined. Learning only became stable, with a smoothly-decreasing loss, when $\rho$ was much larger than $\rho _ { c } ( \rho = 2 \rho _ { c } , \mathrm { g r e e n } )$

We also tested whether the instantaneous change in the output variance (or the “local growth rate”), which is predicted by Eq. 11, empirically describes small-time changes in output variance. We find that this is indeed the case, and that the local growth rate is positive below $\rho = \rho _ { c } \left( \mathrm { F i g . } \ 3 \mathrm { c } \right)$

## 5 INSTABILITY IN A DEEP LINEAR NETWORK

Are deeper networks less stable in the sense we study here? To partially address this question in an analytically tractable setting, we consider an extremely simple deep linear network with $L \geq 1$ layers whose hidden ‘layers’ are just scalars:

$$
y = w _ { L } \cdot \cdot \cdot w _ { 1 } x\tag{18}
$$

where $w _ { i }$ is the scalar weight associated with the i-th ‘layer’, x is a scalar input, and y is a scalar output. Although deep linear networks are admittedly much simpler than their nonlinear counterparts,

![](images/23427e066e8734f1ff3f1c3819e3de44357aa3bdacc06905efa06684aa15f8d0.jpg)

![](images/87892ffc7e33785ce32dd996a4f275afc737f5dfa19b3a6846a2af788ba17493.jpg)  
g<sup>e</sup>Figure 4: Depth benignly modulates stability in a deep linear network. (a) $\rho _ { c }$ can increase, decrease, or remain constant with L depending on the value of the target $W _ { * }$ . In all cases, the difference between $\rho _ { c } ( L )$ and its asymptotic value $\rho _ { c } ( \infty )$ is not substantial. $\left( \mathbf { b } \right) \rho _ { c }$ is non-monotonic as a function of $\gamma ~ ( L ~ \bar { = } ~ 1 0 $ shown), but this is mainly because sufficiently high regularization destabilizes the nontrivial fixed point.

they have the virtue that they are analytically tractable, and they recapitulate a surprising amount of the phenomenology associated with more realistic networks (Saxe et al., 2013; Saxe et al., 2019; Domine et al., 2025).

Consider a teacher-student setting in which our goal is to match y to a target output $y _ { * } : = W _ { * } x :$

$$
J = \mathbb { E } _ { x } \left\{ \frac { ( y - y _ { * } ) ^ { 2 } } { 2 } \right\} = \frac { v } { 2 } ( w _ { 1 } \cdot \cdot \cdot w _ { L } - W _ { * } ) ^ { 2 }\tag{19}
$$

where we have assumed that x has mean zero and variance $v > 0 .$ Although the gradient descent learning dynamics of this network are generally nonlinear and complicated, we can analyze it around the symmetric fixed point with $w _ { 1 } = \cdot \cdot \cdot = w _ { L } = q$ . We will study the impact of depth on the stability of dynamics near this fixed point. Note that we are studying the fixed point of the regularized objective J<sup>˜</sup> (see Eq. 1), not J.

Define the quantity $h : = v q ^ { 2 L - 2 }$ . The Hessian H of J about the symmetric fixed point has two eigenvalues (see Appendix B):

$$
h _ { \perp } = \gamma ( \mathrm { w i t h \ m u l t i p l i c i t y } L - 1 )
$$

$$
h _ { \parallel } = \gamma + L ( h - \gamma ) ( \mathrm { w i t h } \mathrm { m u l t i p l i c i t y } 1 )\tag{20}
$$

Since H is circulant, we can immediately reuse the result from Sec. 3 to find that

$$
\boxed { \rho _ { c } ( L ) = \frac { 1 } { 2 L } \left[ \cfrac { h _ { \parallel } ^ { 2 } } { h _ { \parallel } + \gamma } + ( L - 1 ) \cfrac { h _ { \perp } ^ { 2 } } { h _ { \perp } + \gamma } \right] = \frac { 1 } { 2 L } \left[ \frac { [ \gamma + L ( h - \gamma ) ] ^ { 2 } } { L ( h - \gamma ) + 2 \gamma } + \frac { ( L - 1 ) } { 2 } \gamma \right] } .\tag{21}
$$

In the case of no regularization $( \gamma = 0 )$ , we have the simpler critical arrival rate

$$
\left| ~ \rho _ { c } ( L ) = \frac { h } { 2 } = \frac { v } { 2 } W _ { * } ^ { 2 - 2 / L } \right| .\tag{22}
$$

Interestingly, $\rho _ { c }$ only depends benignly on $L ,$ , and moreover has a well-behaved infinite-depth $( L $ $\infty )$ limit (Fig. 4a). There is an apparent instability as γ increases, since $\rho _ { c }$ diverges to infinity at a finite value of γ (Fig. 4b), but this is misleading; it occurs not because covariance dynamics become unstable, but because it is no longer optimal to match W<sub>∗</sub> when there is too much regularization.

At least in this case, depth does not exacerbate stability issues. In some sense this should be unsurprising, since a result that suggests $\rho _ { c }$ scales poorly with depth would seem inconsistent with biology, whose networks are often deep and highly recurrent; that is, learning is hard, but it should not be impossible.

## 1<sup>6</sup> <sup>POOLING</sup> <sup>PARAMETERS</sup> <sup>CAN</sup> <sup>IMPROVE</sup> <sup>STABILITY</sup>

One approach to improving stability is to average over parameters, since if nearby neurons have 1similar weights, it may make sense for them to share information (Fig. 5a). Spatially proximal neurons are known to communicate in a variety of non-synaptic ways, including through gap junctions (Shimizu & Stopfer, 2013) and extracellular vesicles (Nieves Torres & Lee, 2023), and although the computational role of these features is unknown, it is plausible that they share computationally relevant information, including regarding synaptic weights. Parameter-pooling inevitably increases bias, but this tradeoff may be worth it if learning is otherwise unstable.

![](images/e0d3a26de51cea3ff4de64f933ad9e23d9bbee57c7bb8e7d7a9e57d3b41d8d80.jpg)

![](images/ed2819d9520bbd0072f4c835618fcfdbb778177cc16905f763e38f4db91550db.jpg)

![](images/7acc96d5476a8f2288a43da198683db37f0bebdb809f6814de400dc5760a1400.jpg)

![](images/7eab4ae044dc602d7f307a0eb603cc68c6b90638c01a76714a97ac05145be42c.jpg)

![](images/faf1b3bd303a0e24486571fb3252c846a6005f979d519721ddd57927ab7132dd.jpg)  
Figure 5: Pooling parameters can improve stability. (a) If nearby synapses store similar information, neurons may benefit from sharing information with their spatial neighbors. Pictured is a line of synaptic weights; arrows indicate bidirectional information sharing. (b) The eigenvalues $d _ { q }$ of D are indexed by spatial frequencies $| q | ,$ , and $d _ { q }$ decreases as $| q |$ increases. (c) The q-dependence of $d _ { q }$ causes higher spatial frequencies of the fixed point to be damped, and the effect size increases as the amount of diffusion k increases (more purple  higher k). (d) One effect of this frequencydependence is to smooth the fixed point of the average dynamics. In the pictured example, w<sup>∗</sup> is highly oscillatory (black), but dynamics converge to a smoothed version of it (shades of purple). (e) $\rho _ { c }$ decreases as $k$ increases, so although parameter-pooling produces bias, it also improves stability.

Assume that weights are arranged along a one-dimensional ring, so that $w _ { i }$ is ‘near’ $w _ { i + 1 }$ and $w _ { i - 1 }$ . Also assume that S is circulant. Consider a slight modification of Eq. 9 which incorporates parameter sharing via a ‘diffusion’ matrix D which satisfies $[ D { \pmb w } ] _ { i } = w _ { i - 1 } - 2 w _ { i } + w _ { i + 1 } ;$

$$
\frac { d } { d t } { \pmb w } = { \pmb r } \odot \frac { 1 } { \rho } [ { \pmb S } ( { \pmb w } ^ { \ast } - { \pmb w } ) + { \pmb \xi } ] - \eta \gamma { \pmb w } + \eta k D { \pmb w } .\tag{23}
$$

Here, $k \geq 0$ <sup>1</sup>quantifies the strength of diffusion. This model reduces to the one studied in Sec. 3 when $k = 0$ , but otherwise behaves differently. Its average dynamics read

$$
\frac { 1 } { \eta } \frac { d } { d t } \pmb { w } = S ( \pmb { w } ^ { \ast } - \pmb { w } ) - \gamma \pmb { w } + k D \pmb { w }\tag{24}
$$

and, in the Fourier basis that diagonalizes D and S, the components of the fixed point are

$$
w _ { q } ^ { \mathrm { F P } } : = \frac { s _ { q } } { s _ { q } + \gamma - k d _ { q } } w _ { q } ^ { * } = B _ { q } w _ { q } ^ { * } .\tag{25}
$$

The prefactor $B _ { q } ,$ , which controls the asymptotic bias in learning, is highly spatial-frequencydependent: because $d _ { q }$ is more negative for large frequencies q (Fig. 5b), $B _ { q }$ is closer to zero for large $| q |$ , and the effect size increases as the diffusion strength k increases (Fig. 5c). This implies that what is learned is a spatially smoothed version of the target (Fig. 5d).

11Meanwhile, following algebra similar to what we encountered in Sec. 3 (see Appendix C), covari ance dynamics are stable provided that a stability condition

$$
\boxed { \rho > \rho _ { c } : = \frac { 1 } { 2 N } \sum _ { q } \frac { s _ { q } ^ { 2 } } { s _ { q } + \gamma - k d _ { q } } }\tag{26}
$$

holds. Since $\rho _ { c }$ decreases as k increases, pooling increases stability (Fig. 5e).

## 7 AVERAGING OVER GRADIENT ARRIVALS CAN IMPROVE STABILITY

Another plausible way to improve stability is to effectively reduce the variance of gradient arrivals by not learning directly from the available gradient signal, but from a temporal average of recent 11arrivals (Fig. 6a). We can concretely instantiate this idea by assuming that each parameter $w _ { i }$ has a

reduce noise by temporallyg<sup>k w∗</sup>i <sup>i</sup> |<sup>q</sup>| <sup>d</sup>q averaging recent gradientg<sub>eff</sub>

a<sub>∗</sub>

$$
g _ { e f f } \approx \frac { \pmb { g } ( t _ { 1 } ) + \ldots + \pmb { g } ( t _ { M } ) } { M }
$$

![](images/f53372a4936c736f4d97f52b721892af538a0490670b99943f51f83c035fb428.jpg)

![](images/1aabfbf91e97d75cd316c5e09130cc3f3da5ec9b36f3ced9b21b3d1796cc892d.jpg)

![](images/fe4a968742ed9ea2f0d1da114778501db4fb117afa50ea033cb685a1006a2198.jpg)  
Figure 6: Averaging over gradient arrivals can improve stability. (a) Intuitively, one can reduce variance by performing a temporal average over recent gradient arrivals. (b) Mean dynamics are linear but not monotonic. As the gradient-averaging time scale $\tau _ { z }$ increases, they converge more slowly. (black: $\tau _ { z } = 0 ;$ ; lightest green: $\tau _ { z } = 1 0 0 )$ (c, d) Increasing $\tau _ { z }$ improves stability since $\rho _ { c }$ decreases as long as $\gamma > 0$ . But if $\gamma = 0$ , there is no stability benefit.

matching accumulator variable $z _ { i }$ which leaky-integrates recent gradients on a time scale $\tau _ { z } > 0 ;$

$$
\begin{array} { l } { \displaystyle \frac { d } { d t } z = \boldsymbol { r } \odot \frac { 1 } { \eta \tau _ { z } \rho } [ S ( \boldsymbol { w ^ { * } } - \boldsymbol { w } ) + \xi ] - \frac { 1 } { \tau _ { z } } z } \\ { \displaystyle \frac { d } { d t } \boldsymbol { w } = \eta ( z - \gamma \boldsymbol { w } ) . } \end{array}\tag{27}
$$

By construction, the fixed point of average w dynamics is the same as before, although its transient behavior is slightly different (Fig. 6b). Significantly, though, the addition of the accumulator variable z greatly changes covariance dynamics. The covariances $Z : = \operatorname { C o v } ( z ) , X : = \operatorname { C o v } ( z , w )$ , and $\pmb { W } : = \mathbf { C o v } ( \pmb { w } )$ evolve according to the ODEs (see Appendix D)

$$
\begin{array} { l } { \displaystyle \frac { d } { d t } Z = - \frac { 2 } { \tau _ { z } } Z - \frac { 1 } { \tau _ { z } } ( X S + S X ^ { T } ) + \frac { 1 } { \eta \rho \tau _ { z } ^ { 2 } } \mathrm { d i a g } [ g _ { i } ( \bar { w } ) ^ { 2 } + [ S W S ] _ { i i } + \sigma _ { \mathrm { o b s } } ^ { 2 } ] } \\ { \displaystyle \frac { d } { d t } X = \eta Z - \left( \eta \gamma + \frac { 1 } { \tau _ { z } } \right) X - \frac { 1 } { \tau _ { z } } S W } \\ { \displaystyle \frac { d } { d t } W = \eta \left[ X + X ^ { T } - 2 \gamma W \right] . } \end{array}\tag{28}
$$

As before, it is much easier to analyze these dynamics if we make specific assumptions about S. We 1will again assume that S is either diagonal or circulant.

## 17.1 STABILITY GIVEN A DIAGONAL INPUT COVARIANCE

If S is diagonal, the dynamics of $( Z _ { i i } , X _ { i i } , W _ { i i } )$ for any fixed i are independent of those for $i ^ { \prime } \neq i ,$ and are characterized by $\mathbf { a \ 3 \times 3 }$ matrix

$$
\begin{array} { r } { M : = \left( \begin{array} { c c c } { - \frac { 2 } { \tau _ { z } } } & { - \frac { 2 s _ { i } } { \tau _ { z } } } & { \frac { s _ { i } ^ { 2 } } { \eta \rho \tau _ { z } ^ { 2 } } } \\ { \eta } & { - \left( \eta \gamma + \frac { 1 } { \tau _ { z } } \right) } & { - \frac { s _ { i } } { \tau _ { z } } } \\ { 0 } & { 2 \eta } & { - 2 \eta \gamma } \end{array} \right) \ . } \end{array}\tag{29}
$$

Covariance dynamics are stable provided that M has only negative eigenvalues, and this holds unless $\rho$ is sufficiently small. We can find where the critical value $\rho _ { c } ^ { i }$ is by computing where the determinant of M becomes zero. This yields a stability condition

$$
\boxed { \rho > \rho _ { c } ^ { i } : = \cfrac { s _ { i } ^ { 2 } } { 2 ( 1 + \eta \gamma \tau _ { z } ) ( s _ { i } + \gamma ) } }\tag{30}
$$

analogous to what we discovered before. It reduces to our earlier result (Eq. 13) in the $\tau _ { z } $ 0 limit, but otherwise produces a lower critical value of $\rho$ (Fig. 6c, d). Relatedly, we obtain steady state weight variances

$$
W _ { i i } = \frac { 1 } { \rho - \rho _ { c } ^ { i } } \frac { \sigma _ { \mathrm { o b s } } ^ { 2 } + \bar { g } _ { i } ^ { 2 } } { 2 ( s _ { i } + \gamma ) ( 1 + \gamma \eta \tau _ { z } ) }\tag{31}
$$

which again diverge like $1 / ( \rho - \rho _ { c } ^ { i } )$ . It is important to emphasize that this gain in stability does not come for free: increases in $\tau _ { z }$ can also greatly slow down the convergence of average learning dynamics (Fig. 6b, light green line).

## 7.2 STABILITY GIVEN A CIRCULANT INPUT COVARIANCE

For circulant S, after some algebra, we obtain a result

$$
\boxed { \rho > \frac { 1 } { 2 N ( 1 + \eta \gamma \tau _ { z } ) } \sum _ { q } \frac { s _ { q } ^ { 2 } } { s _ { q } + \gamma } }\tag{32}
$$

which is again similar to what we found before (c.f. Eq. 16), and reduces to it in the $\tau _ { z }  0$ limit.

## 8 DISCUSSION

Learning is difficult, and in this work we have identified and analyzed yet another reason why. It is not enough to perform accurate credit assignment on average if the relevant learning signal is highly variable, since this high variance can produce unstable learning. The source of this variance, sparse gradient arrivals, is distinct from other types of noise relevant to learning (e.g., noise due to using stochastic rather than full-batch gradient descent), and its effects are apparently more insidious.

Because any well-behaved network is approximately linear near a stable fixed point, one expects this issue to be generic rather than architecture-dependent. We considered three ways of addressing the problem—regularization, parameter-sharing, and gradient-averaging—and each reduces the severity of the issue but involves tradeoffs. Regularization and parameter-sharing reduce the (infinite) variance at the expense of bias; moreover, parameter-sharing may only be natural when there is specific (e.g., topographic) structure according to which spatial relationships correlate with relationships between synaptic weights. Gradient-averaging does not produce bias, but does slow down learning. It is plausible that all of these mechanisms, and perhaps others, are utilized by biological networks to avoid the kind of instability we study here.

Connection to analyses of stochastic gradient descent. Standard analyses of stochastic gradient descent (Wu et al., 2018; Ahn et al., 2022) involve a mathematical idea similar to the one we study here, whose content is especially similar to the naive model we considered in Sec. 2: if the learning rate or step size is taken to be too large, errors will accumulate and learning will become unstable. In other ways, the issue is qualitatively different. Our model is defined in continuous- rather than discrete-time, and moreover involves a nontrivial interaction between discrete and stochastic gradient arrivals on the one hand, and continuous relaxation (due to regularization) and accumulation (in the model considered in Sec. 7) on the other. Moreover, the biological interpretation of sparse arrivals—which can be viewed as related to error-communication difficulties rather than simply an overall learning rate—is conceptually distinct. Finally, average dynamics are unconditionally stable.

Speculation regarding biological means of variance reduction. If biological systems do indeed confront the problem of potential instability due to sparse gradient (or more generally, learning signal) arrivals, it is plausible that evolution has developed mechanisms to partially combat it. There is biological precedent for each of the three mechanisms—regularization, parameter-pooling, and gradient-averaging—we considered. It is known that the dendritic spines, which are sites of synaptic plasticity, turn over on relatively short time scales, e.g., on the order of one or two weeks in the hippocampus (Attardo et al., 2015). This continual ‘forgetting’ of synaptic weights can be modeled in the same way we have incorporated regularization. Our gradient-averaging model can be viewed as maintaining a kind of eligibility trace of learning-related information, and there is mounting evidence that such traces exist and play a major role in biological learning (He et al., 2015; Gerstner et al., 2018). Parameter-pooling has not been directly observed to our knowledge, but as we have mentioned, there are plausible means by which it could occur.

Limitations and future work. For clarity and tractability, we have mainly analyzed simple linear models here. More interesting behavior may be possible in less linear regimes, e.g., away from a stable fixed point. Another major assumption we made is that sparse gradients arrive independently at each synapse according to a Poisson process. In reality, the delivery of gradient information to any given neuron is likely correlated with the delivery of gradients to other neurons (e.g., because the information is coming from roughly the same source). Although details associated with this issue likely differ for more general models, our observations suggest that it is worth further study. Putative methods of credit assignment must produce both good average performance and stable variance.

## AI USE STATEMENT

In this work, we used generative AI tools for brainstorming, literature search, to assist with checking the mathematical correctness of our results, and to assist with code generation. The author has checked all text and results and takes full responsibility for them.

## REFERENCES

Kwangjun Ahn, Jingzhao Zhang, and Suvrit Sra. Understanding the unstable convergence of gradient descent. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pp. 247–257. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/ahn22a.html.

Alessio Attardo, James E. Fitzgerald, and Mark J. Schnitzer. Impermanence of dendritic spines in live adult CA1 hippocampus. Nature, 523(7562):592–596, July 2015. ISSN 1476-4687. doi: 10.1038/nature14467. URL https://doi.org/10.1038/nature14467.

Christopher M Bishop. Pattern recognition and machine learning. Springer, 2006.

Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In Hal Daume III and Aarti Singh (eds.),´ Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 1024–1034. PMLR, 13–18 Jul 2020. URL https: //proceedings.mlr.press/v119/bordelon20a.html.

Francis Crick. The recent excitement about neural networks. Nature, 337(6203):129–132, January 1989. ISSN 1476-4687. doi: 10.1038/337129a0. URL https://doi.org/10.1038/ 337129a0.

Clementine Domine, Nicolas Anguita, Alexandra M Proca, Lukas Braun, Daniel Kunin, Pedro Mediano, and Andrew Saxe. From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 102485–102536, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf.

Wulfram Gerstner, Marco Lehmann, Vasiliki Liakoni, Dane Corneil, and Johanni Brea. Eligibility Traces and Plasticity on Behavioral Time Scales: Experimental Support of NeoHebbian Three-Factor Learning Rules. Frontiers in Neural Circuits, Volume 12 - 2018, 2018. ISSN 1662-5110. doi: 10.3389/fncir.2018.00053. URL https://www.frontiersin.org/ journals/neural-circuits/articles/10.3389/fncir.2018.00053.

Kaiwen He, Marco Huertas, Su Z. Hong, XiaoXiu Tie, Johannes W. Hell, Harel Shouval, and Alfredo Kirkwood. Distinct Eligibility Traces for LTP and LTD in Cortical Synapses. Neuron, 88(3):528–538, November 2015. ISSN 0896-6273. doi: 10.1016/j.neuron.2015.09.037. URL https://doi.org/10.1016/j.neuron.2015.09.037.

John Frank Charles Kingman. Poisson processes, volume 3. Clarendon Press, 1992.

Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, and Colin J. Akerman. Random synaptic feedback weights support error backpropagation for deep learning. Nature Communications, 7 (1):13276, November 2016. ISSN 2041-1723. doi: 10.1038/ncomms13276. URL https: //doi.org/10.1038/ncomms13276.

Timothy P. Lillicrap, Adam Santoro, Luke Marris, Colin J. Akerman, and Geoffrey Hinton. Backpropagation and the brain. Nature Reviews Neuroscience, 21(6):335–346, June 2020. ISSN 1471-0048. doi: 10.1038/s41583-020-0277-3. URL https://doi.org/10.1038/ s41583-020-0277-3.

Beren Millidge, Alexander Tschantz, and Christopher L. Buckley. Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs. Neural Computation, 34(6):1329–1368, May 2022. ISSN 0899-7667. doi: 10.1162/neco a 01497. URL https://doi.org/10.1162/ neco\_a\_01497.

Damaris Nieves Torres and Sang H Lee. Inter-neuronal signaling mediated by small extracellular vesicles: wireless communication? Frontiers in Molecular Neuroscience, Volume 16 - 2023, 2023. ISSN 1662-5099. doi: 10.3389/fnmol.2023.1187300. URL https://www.frontiersin.org/journals/molecular-neuroscience/ articles/10.3389/fnmol.2023.1187300.

David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, October 1986. ISSN 1476-4687. doi: 10.1038/323533a0. URL https://doi.org/10.1038/323533a0.

Joao Sacramento, Rui Ponte Costa, Yoshua Bengio, and Walter Senn. Dendritic corti-˜ cal microcircuits approximate the backpropagation algorithm. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/ file/1dc3a89d0d440ba31729b0ba74b93a33-Paper.pdf.

Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv e-prints, art. arXiv:1312.6120, December 2013. doi: 10.48550/arXiv.1312.6120.

Andrew M. Saxe, James L. McClelland, and Surya Ganguli. A mathematical theory of semantic development in deep neural networks. Proceedings of the National Academy of Sciences, 116 (23):11537–11546, 2019. doi: 10.1073/pnas.1820226116. URL https://www.pnas.org/ doi/abs/10.1073/pnas.1820226116.

Benjamin Scellier and Yoshua Bengio. Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation. Frontiers in Computational Neuroscience, Volume 11 - 2017, 2017. ISSN 1662-5188. doi: 10.3389/fncom.2017.00024. URL https://www.frontiersin.org/journals/computational-neuroscience/ articles/10.3389/fncom.2017.00024.

Kazumichi Shimizu and Mark Stopfer. Gap junctions. Current Biology, 23(23):R1026–R1031, December 2013. ISSN 0960-9822. doi: 10.1016/j.cub.2013.10.067. URL https://doi. org/10.1016/j.cub.2013.10.067.

James C.R. Whittington and Rafal Bogacz. Theories of Error Back-Propagation in the Brain. Trends in Cognitive Sciences, 23(3):235–250, March 2019. ISSN 1364-6613. doi: 10.1016/j.tics.2018. 12.005. URL https://doi.org/10.1016/j.tics.2018.12.005.

Lei Wu, Chao Ma, and Weinan E. How SGD Selects the Global Minima in Overparameterized Learning: A Dynamical Stability Perspective. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/ file/6651526b6fb8f29a00507de6a49ce30f-Paper.pdf.

## A ANALYSIS OF LINEAR MODEL

In this appendix, we analyze the behavior of the linear model discussed in Sec. 3 of the main text. Recall that it has

$$
\frac { d } { d t } \pmb { w } = \pmb { r } \odot \frac { 1 } { \rho } [ \pmb { S } ( \pmb { w } ^ { * } - \pmb { w } ) + \pmb { \xi } ] - \eta \gamma \pmb { w }\tag{33}
$$

where r is distributed according to a Poisson arrival process with rate $\eta \rho$ and $\xi \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { o b s } } ^ { 2 } )$ is observation noise associated with a gradient arrival. The master equation that defines how the distribution $P ( { \boldsymbol { w } } | t )$ of weight values evolves in time is

$$
\frac { \partial P } { \partial t } = \eta \frac { \partial } { \partial w } \cdot [ \gamma w P ] + \eta \rho \sum _ { i } \int _ { - \infty } ^ { \infty } d \xi \mathcal { N } ( \xi ; 0 , \sigma _ { \mathrm { { o b s } } } ^ { 2 } ) ~ [ P ( w - ( 1 / \rho ) [ g _ { i } ( w ) + \xi ] e _ { i } | t ) - P ( w | t ) ]\tag{34}
$$

where $e _ { i }$ denotes the standard basis vector with a 1 as its i-th component and zeros elsewhere. $g _ { i } ( w )$   
is shorthand notation for the i-th component of $\pmb { S } ( \pmb { w } ^ { * } - \pmb { w } )$ , the true gradient.

## A.1 DERIVATION OF MEAN DYNAMICS

Let us first derive the dynamics of the mean weight $\bar { \pmb w } : = \mathbb { E } [ \pmb { w } ]$ . The master equation (Eq. 34) has the form of a Fokker-Planck equation plus the Poisson arrival term. The regularization term contributes the expected $- \eta \gamma \bar { \pmb w }$ . We can determine the contribution of the Poisson arrival term by computing its average over w. Note that

$$
\begin{array} { r l } & { \quad \eta \rho \displaystyle \int w \left\{ \int _ { - \infty } ^ { \infty } d \xi \mathcal { N } ( \xi ; 0 , \sigma _ { \mathrm { 0 b s } } ^ { 2 } ) \left[ P ( w - ( 1 / \rho ) [ g _ { i } ( w ) + \xi ] e _ { i } | t ) - P ( w | t ) \right] \right\} d w } \\ & { = \eta \rho \displaystyle \int \mathcal { N } ( \xi ; 0 , \sigma _ { \mathrm { o b s } } ^ { 2 } ) \left[ w P ( w - ( 1 / \rho ) [ g _ { i } ( w ) + \xi ] e _ { i } | t ) - w P ( w | t ) \right] d w } \\ & { = \eta \rho \displaystyle \int N ( \xi ; 0 , \sigma _ { \mathrm { o b s } } ^ { 2 } ) \left[ w + ( 1 / \rho ) ( g _ { i } ( w ) + \xi ) e _ { i } - w \right] P ( w | t ) d w } \\ & { = \eta \mathbb { E } [ g _ { i } ( w ) e _ { i } ] . } \end{array}\tag{35}
$$

Summing over $i ,$ we find that average dynamics are

$$
\boxed { \begin{array} { c } { \displaystyle \frac { d } { d t } \bar { \pmb { w } } = \eta \left[ { \pmb { S } } ( \pmb { w } ^ { * } - \bar { \pmb { w } } ) - \gamma \bar { \pmb { w } } \right] } \end{array} } ,\tag{36}
$$

i.e., they are unbiased.

## A.2 DERIVATION OF COVARIANCE DYNAMICS

Let us now derive the dynamics of the covariance matrix $C : = \mathrm { C o v } ( \pmb { w } )$ . Again, the regularization term contributes to its dynamics in the usual way; the hard part is to determine how the Poisson arrival term contributes.

The average of the Poisson arrival term times ww<sup>T</sup> equals

$$
\begin{array} { r l } & { \quad \eta \rho \displaystyle \int w w ^ { T } \left\{ \int _ { - \infty } ^ { \infty } d \xi \mathcal { N } ( \xi ; 0 , \sigma _ { \mathsf { o b s } } ^ { 2 } ) \left[ P ( w - ( 1 / \rho ) [ g _ { i } ( w ) + \xi ] e _ { i } | t ) - P ( w | t ) \right] \right\} d w } \\ & { = \eta \rho \displaystyle \int \mathcal { N } ( \xi ; 0 , \sigma _ { \mathsf { o b s } } ^ { 2 } ) \left[ w w ^ { T } P ( w - ( 1 / \rho ) [ g _ { i } ( w ) + \xi ] e _ { i } | t ) - w w ^ { T } P ( w | t ) \right] d w } \\ & { = \eta \rho \displaystyle \int \mathcal { N } ( \xi ; 0 , \sigma _ { \mathsf { o b s } } ^ { 2 } ) \left\{ \left[ w + ( 1 / \rho ) ( g _ { i } ( w ) + \xi ) e _ { i } \right] [ w + ( 1 / \rho ) ( g _ { i } ( w ) + \xi ) e _ { i } ] ^ { T } - w w ^ { T } \right\} P ( w | t ) } \\ & { = \eta \mathbb { E } [ ( w e _ { i } ^ { T } + e _ { i } w ^ { T } ) g _ { i } ( w ) + ( 1 / \rho ) ( g _ { i } ( w ) ^ { 2 } + \sigma _ { \mathsf { o b s } } ^ { 2 } ) e _ { i } e _ { i } ^ { T } ] . } \end{array}\tag{37}
$$

Summing over $i ,$ this becomes

$$
\begin{array} { r l } & { \quad \displaystyle \eta \sum _ { i } \mathbb { E } [ ( w e _ { i } ^ { T } + e _ { i } w ^ { T } ) g _ { i } ( w ) + ( 1 / \rho ) ( g _ { i } ( w ) ^ { 2 } + \sigma _ { \mathrm { o b s } } ^ { 2 } ) e _ { i } e _ { i } ^ { T } ] } \\ & { = \displaystyle \eta \mathbb { E } [ w ( w ^ { \ast } - w ) ^ { T } S + S ( w ^ { \ast } - w ) w ^ { T } + ( 1 / \rho ) \mathrm { d i a g } ( g _ { i } ( w ) ^ { 2 } + \sigma _ { \mathrm { o b s } } ^ { 2 } ) ] . } \end{array}\tag{38}
$$

Meanwhile, the time derivative of the mean product w¯w¯<sup>T</sup> equals

$$
\begin{array} { r } { \frac { d } { d t } ( \bar { \boldsymbol { w } } \bar { \boldsymbol { w } } ^ { T } ) = \left( \frac { d } { d t } \bar { \boldsymbol { w } } \right) \bar { \boldsymbol { w } } ^ { T } + \bar { \boldsymbol { w } } \left( \frac { d } { d t } \bar { \boldsymbol { w } } \right) ^ { T } } \\ { = \eta \left[ S ( \boldsymbol { w } ^ { * } - \bar { \boldsymbol { w } } ) - \gamma \bar { \boldsymbol { w } } \right] \bar { \boldsymbol { w } } ^ { T } + \eta \bar { \boldsymbol { w } } \left[ S ( \boldsymbol { w } ^ { * } - \bar { \boldsymbol { w } } ) - \gamma \bar { \boldsymbol { w } } \right] ^ { T } \mathrm { ~ . ~ } } \end{array}\tag{39}
$$

Putting these results together, we find that $C = \mathbb { E } [ { \pmb w w } ^ { T } ] - \bar { \pmb w } \bar { \pmb w } ^ { T }$ evolves according to

$$
\frac { 1 } { \eta } \frac { d } { d t } C = - 2 \gamma C - C S - S C + ( 1 / \rho ) \ \mathrm { d i a g } ( \mathbb { E } [ g _ { i } ( w ) ^ { 2 } ] + \sigma _ { \mathrm { o b s } } ^ { 2 } ) \ .\tag{40}
$$

This result can be slightly simplified using the fact that

$$
\mathbb { E } [ g _ { i } ( { \pmb w } ) ^ { 2 } ] = \bar { g } _ { i } ^ { 2 } + [ { \cal S } { \cal C } { \cal S } ] _ { i i } ,\tag{41}
$$

i.e., using the fact that the squared gradient can be written as a sum of ‘bias’ and ‘variance’ terms. Here $\bar { g } _ { i }$ is shorthand notation for $g _ { i } ( \bar { w } )$ ). We conclude that

$$
\left| \begin{array} { l } { { \frac { 1 } { \eta } \frac { d } { d t } C = - 2 \gamma C - C S - S C + \frac { 1 } { \rho } \sigma _ { \mathrm { o b s } } ^ { 2 } I + \frac { 1 } { \rho } \mathrm { d i a g } ( \bar { g } _ { i } ^ { 2 } + [ S C S ] _ { i i } ) } } \end{array} \right| .\tag{42}
$$

Analysis of Eq. 42 is cumbersome for general $S ,$ , but greatly simplifies for specific choices of $\pmb { S }$ . In the next two subsections, we perform a stability analysis of these covariance dynamics assuming S is either diagonal or circulant.

## A.3 STABILITY ANALYSIS FOR DIAGONAL S

Assume S is diagonal. Let $c _ { i } : = C _ { i i }$ and $s _ { i } : = S _ { i i }$ , and collect both into the vectors c and s. Eq. 42 implies that the diagonal entries of C satisfy

$$
\begin{array} { r } { \frac { 1 } { \eta } \frac { d } { d t } c = - 2 \gamma c - 2 S c + \frac { 1 } { \rho } \sigma _ { \mathrm { o b s } } ^ { 2 } \mathbf { 1 } + \frac { 1 } { \rho } \left[ \pmb { g } + S ^ { 2 } \pmb { c } \right] } \\ { = \frac { 1 } { \rho } [ \sigma _ { \mathrm { o b s } } ^ { 2 } \mathbf { 1 } + \pmb { g } ] - [ 2 \gamma \pmb { I } + 2 S - \frac { 1 } { \rho } S ^ { 2 } ] c } \end{array}\tag{43}
$$

where we use $\textbf {  { g } }$ to denote the vector whose components are $\bar { g } _ { i } ^ { 2 }$ and 1 to denote the vector whose components are all ones.

Stability. Because this ODE is linear, it is clear that stability requires that

$$
2 \gamma + 2 s _ { i } - \frac { 1 } { \rho } s _ { i } ^ { 2 } > 0\tag{44}
$$

for all i. Rearranging, this implies that we require

$$
\boxed { \rho > \rho _ { c } ^ { i } : = \frac { s _ { i } ^ { 2 } } { 2 ( s _ { i } + \gamma ) } }\tag{45}
$$

for all $i ,$ or equivalently that

$$
\boxed { \rho > \rho _ { c } : = \frac { s _ { \operatorname* { m a x } } ^ { 2 } } { 2 ( s _ { \operatorname* { m a x } } + \gamma ) } }\tag{46}
$$

where $s _ { \mathrm { m a x } }$ is the largest input variance $s _ { i }$

Steady state covariance. Setting the left-hand side of Eq. 43 to zero, we find that the steady state covariance is

$$
\begin{array}{c} \boxed { C _ { i i } = \frac { 1 } { \rho - \rho _ { c } ^ { i } } \frac { \sigma _ { \mathrm { { o b s } } } ^ { 2 } + g _ { i } ( \bar { w } ^ { * } ) ^ { 2 } } { 2 ( s _ { i } + \gamma ) } } \end{array} \biggr | ,\tag{47}
$$

where $\bar { \pmb { w } } ^ { * }$ denotes the fixed point of mean dynamics, provided that $\rho > \rho _ { c }$

## A.4 STABILITY ANALYSIS FOR CIRCULANT S

Assume S is circulant. Importantly, circulant matrices can be diagonalized via (discrete) Fourier basis functions. Concretely, let F denote the $N \times N$ matrix whose components are

$$
F _ { j q } = \frac { 1 } { \sqrt { N } } e ^ { i q j } .\tag{48}
$$

Then we can write $S = F \Lambda F ^ { \dagger }$ , where Λ is a diagonal matrix containing the eigenvalues of S. Moreover, if we define the transformed covariance $\tilde { C } : = F ^ { \dagger } C F$ , then

$$
\frac { 1 } { N } \sum _ { i } [ S C S ] _ { i i } = \frac { 1 } { N } \sum _ { q } s _ { q } ^ { 2 } { \tilde { C } } _ { q q } \ ,\tag{49}
$$

where $s _ { q }$ here denotes the q-th eigenvalue of S. The virtue of this result, and hence the reason the circulant assumption is useful, is that the above quantity is independent of the index i.

Combining this result with Eq. 42, we find that the diagonal entries ${ \tilde { C } } _ { q q }$ of the transformed covariance satisfy

$$
\frac { 1 } { \eta } \frac { d } { d t } { \tilde { C } } _ { q q } = - 2 \gamma { \tilde { C } } _ { q q } - 2 s _ { q } { \tilde { C } } _ { q q } + \frac { 1 } { \rho } \sigma _ { \mathrm { o b s } } ^ { 2 } + \frac { 1 } { \rho } \left( A _ { q q } + \frac { 1 } { N } \sum _ { q ^ { \prime } } s _ { q ^ { \prime } } ^ { 2 } { \tilde { C } } _ { q ^ { \prime } q ^ { \prime } } \right)\tag{50}
$$

where ${ \tilde { \cal A } } : = { \cal F } ^ { \dagger } \mathrm { d i a g } ( { \bar { g } } _ { i } ^ { 2 } ) { \cal F }$

Stability. Define $c _ { q } : = \tilde { C } _ { q q }$ and collect the $c _ { q }$ into the vector c. Also, collect the $\tilde { A } _ { q q }$ into a. Eq. 50 can be written in the form

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \eta } \frac { d } { d t } c = - 2 \gamma c - 2 \Lambda c + \frac { 1 } { \rho } \sigma _ { \mathrm { o b s } } ^ { 2 } \mathbf { 1 } + \frac { 1 } { \rho } \left( a + \frac { 1 } { N } \mathbf { 1 } \mathbf { 1 } ^ { T } \mathbf { \Lambda } ^ { 2 } c \right) } \\ { \displaystyle \qquad = \frac { 1 } { \rho } ( \sigma _ { \mathrm { o b s } } ^ { 2 } \mathbf { 1 } + a ) - \left[ 2 \gamma I + 2 \mathbf { \Lambda } - \frac { 1 } { \rho N } \mathbf { 1 } \mathbf { 1 } ^ { T } \mathbf { \Lambda } ^ { 2 } \right] c } \\ { \displaystyle \qquad = \frac { 1 } { \rho } ( \sigma _ { \mathrm { o b s } } ^ { 2 } \mathbf { 1 } + a ) - 2 \gamma \left[ I + ( 1 / \gamma ) \mathbf { \Lambda } \mathbf { 1 } - \frac { 1 } { 2 \rho N \gamma } \mathbf { 1 } \mathbf { 1 } ^ { T } \mathbf { \Lambda } ^ { 2 } \right] c . } \end{array}\tag{51}
$$

From this result, it is clear that the stability of c dynamics depends on the matrix

$$
M : = I + \frac { 1 } { \gamma } \mathbf { A } - \frac { 1 } { 2 \rho N \gamma } \mathbf { 1 1 } ^ { T } \mathbf { A } ^ { 2 } .\tag{52}
$$

This a diagonal matrix plus a rank-one perturbation. We can use the matrix determinant lemma to compute its determinant:

$$
\begin{array} { l } { \displaystyle \operatorname* { d e t } ( { \boldsymbol { M } } ) = \operatorname* { d e t } ( { \boldsymbol { I } } + \frac { 1 } { \gamma } \Lambda ) \left[ 1 - \frac { 1 } { 2 \rho N \gamma } \mathbf { 1 } ^ { T } \mathbf { \Lambda } ^ { 2 } ( { \boldsymbol { I } } + \frac { 1 } { \gamma } \Lambda ) ^ { - 1 } \mathbf { 1 } \right] } \\ { \displaystyle = \operatorname* { d e t } ( { \boldsymbol { I } } + \frac { 1 } { \gamma } \Lambda ) \left[ 1 - \frac { 1 } { 2 \rho N \gamma } \sum _ { q } \frac { s _ { q } ^ { 2 } } { s _ { q } / \gamma + 1 } \right] ~ . } \end{array}\tag{53}
$$

For sufficiently high $\rho ,$ it is clear that this determinant is positive, which indicates stability. We can identify the stability threshold by determining where the determinant is zero; since the eigenvalues of S are all positive, this can only happen when the bracketed factor equals zero.

We conclude that stability requires

$$
\boxed { \rho > \rho _ { c } : = \frac { 1 } { 2 N } \sum _ { q } \frac { s _ { q } ^ { 2 } } { s _ { q } + \gamma } }\tag{54}
$$

since beyond this value of $\rho ,$ the determinant is negative.

Steady state covariance. The above covariance dynamics can be rewritten as

$$
\frac { 1 } { \eta } \frac { d } { d t } c = \frac { 1 } { \rho } ( \sigma _ { \mathrm { \tiny { o b s } } } ^ { 2 } { \bf 1 } + a ) - 2 \gamma M c ,\tag{55}
$$

so its steady state occurs at

$$
c ^ { * } : = \frac { 1 } { 2 \rho \gamma } M ^ { - 1 } ( \sigma _ { \mathrm { o b s } } ^ { 2 } { \bf 1 } + a ) .\tag{56}
$$

This can be simplified if we write out $M ^ { - 1 }$ explicitly. By the Sherman-Morrison formula,

$$
\begin{array} { r } { M ^ { - 1 } = ( I + \Lambda / \gamma ) ^ { - 1 } + \frac { 1 } { 2 \rho N \gamma } \frac { ( I + \Lambda / \gamma ) ^ { - 1 } \mathbf { 1 } \mathbf { 1 } ^ { T } \Lambda ^ { 2 } ( I + \Lambda / \gamma ) ^ { - 1 } } { 1 - \frac { 1 } { 2 \rho N \gamma } \mathbf { 1 } ^ { T } \Lambda ^ { 2 } ( I + \Lambda / \gamma ) ^ { - 1 } \mathbf { 1 } } } \\ { = ( I + \Lambda / \gamma ) ^ { - 1 } + \frac { 1 } { 2 N \gamma } \frac { ( I + \Lambda / \gamma ) ^ { - 1 } \mathbf { 1 } \mathbf { 1 } ^ { T } \Lambda ^ { 2 } ( I + \Lambda / \gamma ) ^ { - 1 } } { \rho - \rho _ { c } } . } \end{array}\tag{57}
$$

Although the expression we obtain is more complicated than in the case of diagonal $S ,$ we still see a divergence that goes like $1 / ( \rho - \rho _ { c } )$ as $\rho$ approaches the critical arrival rate from above.

## B MORE DETAILS ON THE DEEP LINEAR NETWORK

In this appendix, we provide a more thorough account of the deep linear network analysis described in Sec. 5. We consider a scalar deep network with $L \geq 2$ ‘layers’:

$$
y = w _ { L } \cdot \cdot \cdot w _ { 1 } x .\tag{58}
$$

Here, $x$ is the input, $w _ { i }$ is the weight associated with the i-th ‘layer’, and $y$ is the output. x is assumed to have mean zero and variance v.

We assume this network is trained via gradient descent to match a target output $y _ { * } : = W _ { * } x$ which also depends linearly on its input. The corresponding objective equals

$$
\tilde { J } = J + \sum _ { i } \frac { \gamma } { 2 } w _ { i } ^ { 2 } = \frac { v } { 2 } ( w _ { 1 } \cdots w _ { L } - W _ { * } ) ^ { 2 } + \sum _ { i } \frac { \gamma } { 2 } w _ { i } ^ { 2 } .\tag{59}
$$

Fixed point assumption. Although learning dynamics are generally nonlinear, they are approximately linear near a fixed point. For simplicity, we will consider the symmetric fixed point with

$$
w _ { 1 } = \cdot \cdot \cdot = w _ { L } = q ,\tag{60}
$$

where $q$ is chosen to minimize the objective, i.e.,

$$
q : = \mathrm { a r g m i n } _ { q } \frac { v } { 2 } ( q ^ { L } - W _ { * } ) ^ { 2 } + \frac { L \gamma } { 2 } q ^ { 2 } .\tag{61}
$$

Since q minimizes $\tilde { J } , d \tilde { J } / d q = 0$ , so the condition

$$
\overline { { v ( W _ { * } - q ^ { L } ) q ^ { L - 2 } = \gamma } }\tag{62}
$$

holds.

Hessian near fixed point. To characterize linear dynamics near the symmetric fixed point, we need to compute the corresponding $L \times L$ Hessian matrix. Note that the first derivatives at an arbitrary point are

$$
\frac { \partial J } { \partial w _ { i } } = v ( w _ { 1 } \cdot \cdot \cdot w _ { L } - W _ { * } ) \frac { W } { w _ { i } }\tag{63}
$$

where we define $W : = w _ { 1 } \cdot \cdot \cdot w _ { L }$ . The Hessian’s components at an arbitrary point are

$$
\begin{array} { l } { { H _ { i i } = v \displaystyle \frac { W ^ { 2 } } { w _ { i } ^ { 2 } } } } \\ { { H _ { i j } = v \displaystyle \frac { W ^ { 2 } } { w _ { i } w _ { j } } + v ( W - W _ { * } ) \displaystyle \frac { W } { w _ { i } w _ { j } } . } } \end{array}\tag{64}
$$

At the symmetric fixed point,

$$
\begin{array} { l } { { H _ { i i } = v q ^ { 2 L - 2 } } } \\ { { \nonumber } } \\ { { H _ { i j } = v q ^ { 2 L - 2 } + v ( q ^ { L } - W _ { * } ) q ^ { L - 2 } . } } \end{array}\tag{65}
$$

To ease notation, define $h : = v q ^ { 2 L - 2 }$ . This result implies that we can write the Hessian matrix at the symmetric fixed point as

$$
\pmb { H } = \gamma \pmb { I } + ( h - \gamma ) \pmb { 1 } \pmb { 1 } ^ { T } \ ,\tag{66}
$$

i.e., it is a diagonal matrix plus a rank-one ‘correction’. We immediately learn that H has two eigenvalues: one associated with the correction, and another eigenvalue (with multiplicity $L - 1 )$ associated with the orthogonal directions:

$$
h _ { \perp } = \gamma
$$

$$
h _ { \parallel } = \gamma + L ( h - \gamma ) .\tag{67}
$$

Stability condition near symmetric fixed point. This Hessian is circulant, so we can reuse a linear model result (see Appendix A) to immediately learn that

$$
\boxed { \rho _ { c } ( L ) = \frac { 1 } { 2 L } \left[ \cfrac { h _ { \parallel } ^ { 2 } } { h _ { \parallel } + \gamma } + ( L - 1 ) \cfrac { h _ { \perp } ^ { 2 } } { h _ { \perp } + \gamma } \right] = \frac { 1 } { 2 L } \left[ \frac { [ \gamma + L ( h - \gamma ) ] ^ { 2 } } { L ( h - \gamma ) + 2 \gamma } + \frac { ( L - 1 ) } { 2 } \gamma \right] } .\tag{68}
$$

In the case of no regularization $( \gamma = 0 )$ , we have the simpler critical arrival rate

$$
\boxed { \rho _ { c } ( L ) = \frac { h } { 2 } = \frac { v } { 2 } W _ { * } ^ { 2 - 2 / L } }\tag{69}
$$

which asymptotically approaches $\rho _ { c } ( \infty ) : = ( v / 2 ) W _ { * } ^ { 2 }$ as $L \to \infty$

## C MORE DETAILS ON POOLING PARAMETERS

In this appendix, we derive results related to the parameter-pooling model discussed in Sec. 6. It is defined in almost the same way as our linear model (see Sec. 3 and Appendix A):

$$
\frac { d } { d t } { \pmb w } = { \pmb r } \odot \frac { 1 } { \rho } [ { \pmb S } ( { \pmb w } ^ { \ast } - { \pmb w } ) + { \pmb \xi } ] - \eta \gamma { \pmb w } + \eta k D { \pmb w } .\tag{70}
$$

The main difference is that there is an additional term ηkDw, where D is a ‘diffusion’ matrix that implements a sort of spatial parameter pooling. In particular, D is defined such that

$$
\sum _ { j } D _ { i j } w _ { j } : = w _ { i + 1 } - 2 w _ { i } + w _ { i - 1 }\tag{71}
$$

for all i. The interpretation is that information flows from nodes $i + 1$ and i 1 to node i, and from node i to nodes i + 1 and i 1.

For simplicity, we assume periodic boundary conditions, so that D is circulant rather than tridiagonal. Under this assumption, its eigenvalues are

$$
d _ { q } : = - 4 \sin ^ { 2 } \left( \frac { \pi q } { N } \right) \leq 0\tag{72}
$$

where $q = 0 , 1 , . . . , N - 1$ , and it can be diagonalized using discrete Fourier basis functions.

The derivation of mean and covariance dynamics for this model is almost the same as it was for the linear model studied in Appendix A; everything is the same up to the identification

$$
\gamma I  \gamma I - k D .\tag{73}
$$

For this reason, we omit the derivation and directly state the results. As in Appendix A, we assume that S is circulant for analytic tractability. This assumption is particularly important here because it implies that S and D commute.

Mean dynamics. Mean dynamics follow

$$
\boxed { \begin{array} { c } { \displaystyle \frac { d } { d t } \pmb { \bar { w } } = \eta \left[ \pmb { S } ( \pmb { w } ^ { * } - \pmb { \bar { w } } ) - ( \gamma \pmb { I } - k \pmb { D } ) \pmb { \bar { w } } \right] } \end{array} } .\tag{74}
$$

The corresponding fixed point is

$$
\bar { \pmb { w } } ^ { \mathrm { F P } } : = ( { \pmb S } + \gamma { \pmb I } - k { \pmb D } ) ^ { - 1 } { \pmb S } { \pmb w } ^ { \ast } .\tag{75}
$$

In the discrete Fourier basis, this becomes

$$
w _ { q } ^ { \mathrm { F P } } : = \frac { s _ { q } } { s _ { q } + \gamma - k d _ { q } } w _ { q } ^ { * } = B _ { q } w _ { q } ^ { * } .\tag{76}
$$

where $s _ { q }$ denotes the q-th eigenvalue of S and $d _ { q }$ denotes the q-th eigenvalue of D. As noted in the main text, the specific form of $d _ { q }$ makes the overall associated prefactor $B _ { q }$ fairly spatial-frequencydependent.

Covariance dynamics. Covariance dynamics follow

$$
\left| \frac { 1 } { \eta } \frac { d } { d t } C = - 2 \gamma C - C S + k C D + k D C - S C + \frac { 1 } { \rho } \sigma _ { \mathrm { o b s } } ^ { 2 } I + \frac { 1 } { \rho } \mathrm { d i a g } ( \bar { g } _ { i } ^ { 2 } + [ S C S ] _ { i i } ) \right| .\tag{77}
$$

Moreover, the diagonal elements of the discrete-Fourier-transformed matrix ${ \tilde { C } } : = F ^ { \dagger } C F$ (see Appendix A for more context) satisfy

$$
\frac { 1 } { \eta } \frac { d } { d t } { \tilde { C } } _ { q q } = - 2 \gamma { \tilde { C } } _ { q q } + 2 k d _ { q } { \tilde { C } } _ { q q } - 2 s _ { q } { \tilde { C } } _ { q q } + \frac { 1 } { \rho } { \sigma } _ { \mathrm { o b s } } ^ { 2 } + \frac { 1 } { \rho } \left( A _ { q q } + \frac { 1 } { N } \sum _ { q ^ { \prime } } s _ { q ^ { \prime } } ^ { 2 } { \tilde { C } } _ { q ^ { \prime } q ^ { \prime } } \right) .\tag{78}
$$

Following a slight modification of the argument from Appendix A, we obtain the stability condition

$$
\boxed { \rho > \frac { 1 } { 2 N } \sum _ { q } \frac { s _ { q } ^ { 2 } } { s _ { q } + \gamma - k d _ { q } } } .\tag{79}
$$

## D MORE DETAILS ON AVERAGING OVER GRADIENT ARRIVALS

In this appendix, we analyze the mean and covariance dynamics of the gradient-accumulation model discussed in Sec. 7 of the main text. It is defined similarly to the linear model (see Sec. 3 and Appendix A), but with a ‘gradient accumulation’ variable $z _ { i }$ for each weight $w _ { i }$ :

$$
\begin{array} { l } { \displaystyle \frac { d } { d t } z = \boldsymbol { r } \odot \frac { 1 } { \eta \tau _ { z } \rho } [ \boldsymbol { S } ( \boldsymbol { w } ^ { * } - \boldsymbol { w } ) + \boldsymbol { \xi } ] - \frac { 1 } { \tau _ { z } } z } \\ { \displaystyle \frac { d } { d t } \boldsymbol { w } = \eta ( z - \gamma \boldsymbol { w } ) . } \end{array}\tag{80}
$$

This model’s master equation has the form

$$
\begin{array} { l } { \displaystyle \frac { \partial P } { \partial t } = - \eta \frac { \partial } { \partial w } \cdot \left[ ( z - \gamma w ) P \right] + \frac { 1 } { \tau _ { z } } \frac { \partial } { \partial z } \cdot ( z P ) } \\ { \displaystyle + \ \eta \rho \sum _ { i } \int _ { - \infty } ^ { \infty } d \xi \mathcal { N } ( \xi ; 0 , \sigma _ { \mathrm { o b s } } ^ { 2 } ) \ \left[ P ( z - \frac { 1 } { \eta \rho \tau _ { z } } [ g _ { i } ( w ) + \xi ] e _ { i } , w | t ) - P ( z , w | t ) \right] . } \end{array}\tag{81}
$$

In this section, because there are multiple covariance matrices, we will use different notation than elsewhere. Let $Z : = \operatorname { C o v } ( z ) , X : = \operatorname { C o v } ( z , w )$ denote the cross-covariance, and $\pmb { W } : = \mathbf { C o v } ( \pmb { w } )$

The calculations required to treat this model largely resemble those from Appendix A, but are lengthier, and hence more tedious. Because they are straightforward, we skip a number of intermediate steps and mostly present the final results.

Mean dynamics. This model’s average dynamics are

$$
\begin{array} { l } { { \displaystyle { \frac { d } { d t } } z = \frac { 1 } { \tau _ { z } } S ( { \pmb w } ^ { * } - { \pmb \bar { w } } ) - \frac { 1 } { \tau _ { z } } \bar { z } } } \\ { { \displaystyle { \frac { d } { d t } } { \pmb \bar { w } } = \eta ( \bar { z } - \gamma { \pmb \bar { w } } ) . } } \end{array}\tag{82}
$$

In particular, the steady state value of w¯ is the same as it is in the linear model; this kind of gradient accumulation does not produce bias, at least asymptotically.

Covariance dynamics. After a tedious calculation, one can find that

$$
\begin{array} { l } { \displaystyle \frac { d } { d t } Z = - \frac { 2 } { \tau _ { z } } Z - \frac { 1 } { \tau _ { z } } ( X S + S X ^ { T } ) + \frac { 1 } { \eta \rho \tau _ { z } ^ { 2 } } \mathrm { d i a g } [ g _ { i } ( \bar { w } ) ^ { 2 } + [ S W S ] _ { i i } + \sigma _ { \mathrm { o b s } } ^ { 2 } ] } \\ { \displaystyle \frac { d } { d t } X = \eta Z - \left( \eta \gamma + \frac { 1 } { \tau _ { z } } \right) X - \frac { 1 } { \tau _ { z } } S W } \\ { \displaystyle \frac { d } { d t } W = \eta \left[ X + X ^ { T } - 2 \gamma W \right] . } \end{array}\tag{83}
$$

Diagonal S. Assume S is diagonal and let $Z _ { i } , X _ { i }$ , and $W _ { i }$ denote the diagonals of these covariance matrices. The diagonal entries have dynamics

$$
\begin{array} { l } { { \displaystyle \frac { d } { d t } Z _ { i } = - \frac { 2 } { \tau _ { z } } Z _ { i } - \frac { 2 } { \tau _ { z } } s _ { i } X _ { i } + \frac { 1 } { \eta \rho \tau _ { z } ^ { 2 } } [ g _ { i } ( \bar { w } ) ^ { 2 } + s _ { i } ^ { 2 } W _ { i } + \sigma _ { \mathrm { o b s . } } ^ { 2 } } } \\ { { \displaystyle \frac { d } { d t } X _ { i } = \eta Z _ { i } - \left( \eta \gamma + \frac { 1 } { \tau _ { z } } \right) X _ { i } - \frac { 1 } { \tau _ { z } } s _ { i } W _ { i } } } \\ { { \displaystyle \frac { d } { d t } W _ { i } = 2 \eta \left[ X _ { i } - \gamma W _ { i } \right] . } } \end{array}\tag{84}
$$

The stability of this three-dimensional system of ODEs is characterized by the $3 \times 3$ matrix

$$
\begin{array} { r } { M : = \left( \begin{array} { c c c } { - \frac { 2 } { \tau _ { z } } } & { - \frac { 2 s _ { i } } { \tau _ { z } } } & { \frac { s _ { i } ^ { 2 } } { \eta \rho \tau _ { z } ^ { 2 } } } \\ { \eta } & { - \left( \eta \gamma + \frac { 1 } { \tau _ { z } } \right) } & { - \frac { s _ { i } } { \tau _ { z } } } \\ { 0 } & { 2 \eta } & { - 2 \eta \gamma } \end{array} \right) \ . } \end{array}\tag{85}
$$

We can directly compute that its determinant is

$$
\begin{array} { c l l } { { } } & { { \displaystyle - \frac { 2 } { \tau _ { z } } \left[ 2 \eta \gamma ( \eta \gamma + 1 / \tau _ { z } ) + 2 \eta s _ { i } / \tau _ { z } \right] + \frac { 2 s _ { i } } { \tau _ { z } } \left[ - 2 \eta ^ { 2 } \gamma \right] + \frac { s _ { i } ^ { 2 } } { \eta \rho \tau _ { z } ^ { 2 } } \left[ 2 \eta ^ { 2 } \right] } }  \\ { { } } & { { = - \displaystyle \frac { 2 } { \tau _ { z } } \left[ 2 \eta \gamma ( \eta \gamma + 1 / \tau _ { z } ) + 2 \eta s _ { i } ( 1 / \tau _ { z } + \eta \gamma ) \right] + \frac { s _ { i } ^ { 2 } } { \eta \rho \tau _ { z } ^ { 2 } } \left[ 2 \eta ^ { 2 } \right] } } \\ { { } } & { { = \displaystyle - \frac { 4 \eta } { \tau _ { z } } ( \eta \gamma + 1 / \tau _ { z } ) \left[ \gamma + s _ { i } \right] + \frac { s _ { i } ^ { 2 } } { \eta \rho \tau _ { z } ^ { 2 } } \left[ 2 \eta ^ { 2 } \right] . } } \end{array}\tag{86}
$$

Stability requires that the determinant is negative. Hence, we need it to be true that

$$
2 ( 1 + \eta \gamma \tau _ { z } ) ( \gamma + s _ { i } ) - \frac { s _ { i } ^ { 2 } } { \rho } > 0 .\tag{87}
$$

Equivalently,

$$
\boxed { \rho > \rho _ { c } ^ { i } : = \frac { s _ { i } ^ { 2 } } { 2 ( 1 + \gamma \eta \tau _ { z } ) ( s _ { i } + \gamma ) } } .\tag{88}
$$

In the infinitely fast integration limit $( \tau _ { z }  0 )$ , this reduces to the linear model result (see Appendix A).

The corresponding steady state weight variances are

$$
\boxed { W _ { i i } = \frac { 1 } { \rho - \rho _ { c } ^ { i } } \frac { \sigma _ { \mathrm { o b s } } ^ { 2 } + \bar { g } _ { i } ^ { 2 } } { 2 ( s _ { i } + \gamma ) ( 1 + \gamma \eta \tau _ { z } ) } } \ .\tag{89}
$$

Interestingly, this expression is the same as what we found for the linear model, except that $\rho _ { c } ^ { i }$ has a different form.

Circulant S. If S is instead circulant, we must go through a lengthier version of the circulant algebra from Appendix A. In the end, we find almost the same results. Most importantly, we find a stability condition

$$
\boxed { \rho > \rho _ { c } : = \frac { 1 } { 2 N ( 1 + \eta \gamma \tau _ { z } ) } \sum _ { k } \frac { s _ { k } ^ { 2 } } { s _ { k } + \gamma } }\tag{90}
$$

which reduces to the corresponding linear model condition in the $\tau _ { z }  0$ limit.