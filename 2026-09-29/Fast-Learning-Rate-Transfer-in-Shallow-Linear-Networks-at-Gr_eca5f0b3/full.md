# Fast Learning Rate Transfer in Shallow Linear Networks at Growing Training Horizons

Mana Sakai<sup>1,2</sup> Masaaki Imaizumi<sup>1,2,3</sup>

<sup>1</sup>The University of Tokyo <sup>2</sup>RIKEN Center for Advanced Intelligence Project <sup>3</sup>Kyoto University

## Abstract

Hyperparameter transfer across model width can substantially reduce the cost of tuning large neural networks, but its behavior when the training horizon grows with width is not fully understood. Building on the framework of fast hyperparameter transfer (Ghosh et al., 2026), which formalizes when transfer is efective, we investigate conditions that ensure fast transfer in the growing-horizon regime. Specifically, we study learning-rate transfer in a shallow linear network with a single trainable hidden matrix, trained by full-batch gradient descent. Under additional spectral assumptions, our main results are threefold. (i) We prove fast learning-rate transfer as n, T → ∞ whenever $T = o ( { \sqrt { n } } )$ . (ii) We characterize the transfer rates through the finitewidth perturbation scale, the first-order sensitivities of the loss and its learning-rate derivative to finite-width perturbations, and the local loss curvature. (iii) We derive limiting distributions for the optimal learning rate and optimized loss, governed by fluctuations associated with the extreme eigenvalues of the data Gram matrix. These results clarify how spectral structure and local loss sensitivities govern learning-rate transfer at growing horizons.

## 1 Introduction

## 1.1 Background and Motivation

Tuning hyperparameters, such as learning rates, becomes increasingly costly as neural networks grow. Hyperparameter transfer seeks to reduce this cost by reusing hyperparameters tuned on small proxy models when training larger models. Building on µP (Yang and Hu, 2021), which specifies width-dependent initialization and learning-rate scalings that preserve nontrivial feature learning in the infinite-width limit, Yang et al. (2021) introduced µTransfer: tune hyperparameters on a small proxy model and transfer them to a larger model without retuning. Their experiments demonstrate that appropriately parameterized learning rates can remain efective across substantially diferent widths. Subsequent studies have examined the empirical robustness of transfer to architectural and optimizer choices (Lingle, 2024), as well as the role of layerwise learning-rate scaling across parameterizations (Lingle, 2024; Everett et al., 2024). These empirical successes motivate theoretical analysis of the reliable transfer of hyperparameters.

Ghosh et al. (2026) introduced the notion of fast hyperparameter transfer to formalize when transfer is efective. Their framework shows that what matters is whether the optimal hyperparameter converges suficiently fast relative to the rate at which model performance converges. Thus convergence of the optimal hyperparameter alone does not guarantee fast transfer. Relatedly, Hayou (2026) established convergence of the optimal learning rate for deep linear networks trained by full-batch gradient descent under µP, providing a theoretical basis for learning-rate transfer across width.

One major challenge in hyperparameter transfer is dealing with growing training horizons T, particularly for learning-rate transfer. Recent language-model scaling practices motivate increasing the number of training tokens alongside model size (Hofmann et al., 2022; Dey et al., 2025), which often increases the number of gradient updates. From a theoretical perspective, the joint scaling limit of width and training horizon is substantially more delicate than the fixed-horizon setting, since the limiting objective itself changes as T grows. Consequently, the theoretical understanding of this regime remains limited. An important exception is Wen et al. (2026), who establish fast learning-rate transfer at growing horizons in sketched linear regression. Motivated by these observations, we ask the following questions:

As model width and training horizon grow jointly, what conditions ensure fast learning-rate transfer, and which quantities govern whether such transfer occurs?

## 1.2 This Study

To answer these questions, we consider a linear network with a single trainable hidden matrix, frozen random input and output maps, and $\mu \mathrm { P }$ initialization. The network is trained by fullbatch gradient descent with a constant learning rate. For each horizon T, we compare the optimal learning rate and optimized training loss at width n with their infinite-width counterparts.

Our contributions are as follows:

• A suficient growth condition on the training horizon for fast transfer: For fixed sample size and input dimension, we prove fast transfer whenever $T \to \infty$ and $T = o ( { \sqrt { n } } )$ under additional spectral assumptions (Theorem 3.4; Corollary 3.5).

• Summary of transfer errors using loss-based quantities: In our setting, the rates of the three transfer errors can be summarized in terms of the finite-width perturbation scale and three horizon-dependent quantities: the first-order sensitivities of the loss value and its learning-rate derivative to finite-width perturbations, and the local curvature of the loss. This formulation makes explicit how these quantities combine to determine the transfer rates as the training horizon grows (Theorem 3.4; Section 4.2). It also helps interpret the failure of fast transfer in the fixed-horizon counterexample of Proposition 3.8.

• Limiting distributions of the optimal learning rate and the optimized loss: We derive limiting distributions of the optimal learning rate and optimized loss as the training horizon grows (Theorem 3.4). Both limits are linear functionals of the same two components of the perturbation, associated with the largest and smallest positive eigenvalues of the data Gram matrix.

Our analysis provides some insights. First, diferent spectra can lead to diferent transfer rates and suficient width–horizon scalings, since the loss-based quantities vary depending on the data spectrum as the horizon grows. This observation contrasts with Wen et al. (2026). Second, the leading loss-value and slope responses to the same finite-width perturbation need not be aligned. Consequently, some perturbation directions can change the optimized loss at leading order while producing no leading-order shift in the optimal learning rate. This illustrates how width dependence in the attained loss can difer from width dependence in hyperparameter selection. The observation is qualitatively consistent with the mechanism proposed by Ghosh et al. (2026), although our analysis concerns local finite-width perturbations rather than a decomposition of the optimization trajectory; Section 5 discusses this connection.

Additional related work is deferred to Appendix A.

## 2 Problem Setup and Transfer Framework

## 2.1 Model and exact loss dynamics

Fix positive integers d, m, training inputs $X ^ { \top } = [ x _ { 1 } \dots x _ { m } ]$ , and targets $y = ( y _ { 1 } , . . . , y _ { m } ) ^ { \intercal }$ Consider the linear network

$$
f _ { n } ( x ; W _ { 0 } , W _ { 1 } , V ) = V ^ { \top } W _ { 1 } W _ { 0 } x ,
$$

where $W _ { 0 } \in \mathbb { R } ^ { n \times d } , W _ { 1 } \in \mathbb { R } ^ { n \times n }$ , and $V \in \mathbb { R } ^ { n }$ . This model is similar to the one analyzed in Hayou (2026), except we have only one trainable weight $W _ { 1 }$ . The initialization is given entrywise as $W _ { 0 , i j } \sim \mathcal { N } ( 0 , d ^ { - 1 } ) , W _ { 1 , i j } ^ { ( 0 ) } \sim \mathcal { N } ( 0 , n ^ { - 1 } )$ , and $V _ { i } \sim \mathcal { N } ( 0 , n ^ { - 2 } )$ , following the scalings in $\mu \mathrm { P } .$ . All entries of $W _ { 0 } , W _ { 1 } ^ { ( 0 ) }$ , and $V$ are mutually independent. Define

$$
\begin{array} { r } { K = d ^ { - 1 } { X X ^ { \top } } \in \mathbb { R } ^ { m \times m } , \qquad A _ { n } = \| { V } \| ^ { 2 } H _ { n } H _ { n } ^ { \top } \in \mathbb { R } ^ { m \times m } , \qquad H _ { n } = { X W _ { 0 } ^ { \top } } \in \mathbb { R } ^ { m \times n } . } \end{array}
$$

The vector of predictions and the residual at step t are

$$
f _ { n } ^ { ( t ) } ( X ) = H _ { n } ( W _ { 1 } ^ { ( t ) } ) ^ { \top } V \in \mathbb { R } ^ { m } , \qquad \chi _ { n } ^ { ( t ) } = f _ { n } ^ { ( t ) } ( X ) - y \in \mathbb { R } ^ { m } .
$$

The loss is $\begin{array} { r } { { \cal { L } } _ { n } ( W _ { 1 } ) = ( 2 m ) ^ { - 1 } \| f _ { n } ( X ; W _ { 1 } ) - y \| ^ { 2 } } \end{array}$ , and for a fixed learning rate $\eta ,$ we train the model by full-batch gradient descent (GD) as

$$
\boldsymbol { W } _ { 1 } ^ { ( t + 1 ) } = \boldsymbol { W } _ { 1 } ^ { ( t ) } - \eta \nabla _ { \boldsymbol { W } _ { 1 } } \mathcal { L } _ { n } ( \boldsymbol { W } _ { 1 } ^ { ( t ) } ) \qquad ( t = 0 , \ldots , T - 1 ) .
$$

Since there is only one trainable weight, training this model with the GD update is equivalent to performing GD for linear regression on fixed random features.

For $A \in \mathbb { R } ^ { m \times m }$ and $\eta \in \mathbb { R }$ , define $B _ { A } ( \eta ) = I _ { m } - m ^ { - 1 } \eta A$ . We abbreviate $B _ { n } ( \eta ) : = B _ { A _ { n } } ( \eta )$ and $B _ { \infty } ( \eta ) : = B _ { K } ( \eta )$

Proposition 2.1 (Exact finite-width loss dynamics). For every integer $T \geq 1$ and every $\eta \in \mathbb { R }$ , we have $\chi _ { n } ^ { ( T ) } = B _ { n } ( \eta ) ^ { T } \chi _ { n } ^ { ( 0 ) }$ , and consequently,

$$
\phi _ { n , T } ( \eta ) : = \mathscr { L } _ { n } ( W _ { 1 } ^ { ( T ) } ) = ( 2 m ) ^ { - 1 } ( \chi _ { n } ^ { ( 0 ) } ) ^ { \top } B _ { n } ( \eta ) ^ { 2 T } \chi _ { n } ^ { ( 0 ) } .
$$

Proposition 2.1 turns learning-rate selection into the minimization of a scalar random loss curve. All dependence on the initialization enters through the fixed-dimensional pair $( \chi _ { n } ^ { ( 0 ) } , A _ { n } )$ , even though the parameter matrix grows with $n .$ This reduction separates the two ingredients of the analysis: the spectral geometry of the reference curve and the finite-width fluctuations around it.

## 2.2 Infinite-width objectives and spectral geometry

For each T, define the corresponding infinite-width objective by

$$
\phi _ { \infty , T } ( \eta ) : = ( 2 m ) ^ { - 1 } \| B _ { \infty } ( \eta ) ^ { T } y \| ^ { 2 } = ( 2 m ) ^ { - 1 } y ^ { \top } B _ { \infty } ( \eta ) ^ { 2 T } y .
$$

For every fixed T, Corollary C.1 shows that $\phi _ { n , T }$ converges to $\phi _ { \infty , T }$ uniformly on deterministic compact learning-rate intervals.

Consider the decomposition $\begin{array} { r } { K = \sum _ { j = 1 } ^ { r } \lambda _ { j } P _ { j } } \end{array}$ , where $\lambda _ { 1 } > \cdots > \lambda _ { r } > 0$ are the distinct positive eigenvalues and $P _ { j }$ are their orthogonal spectral projectors.<sup>1</sup> Let $P _ { 0 }$ denote the projection onto ker $( K )$ , and set $c _ { j } = \| P _ { j } y \| ^ { 2 }$ and $c _ { 0 } = \| P _ { 0 } y \| ^ { 2 }$ . Define $\mathcal { T } _ { y } = \{ j \in [ r ] : c _ { j } > 0 \}$ and we assume $\mathcal { I } _ { y } \neq \emptyset$ . Then we can express the infinite-width objective as

$$
\phi _ { \infty , T } ( \eta ) = \frac { c _ { 0 } } { 2 m } + \frac { 1 } { 2 m } \sum _ { j = 1 } ^ { r } c _ { j } \left( 1 - \frac { \eta \lambda _ { j } } { m } \right) ^ { 2 T } .
$$

Each active eigenspace contributes a term with contraction factor $1 - \eta \lambda _ { j } / m$ , while $c _ { 0 } / ( 2 m )$ is independent of the learning rate. Thus $\mathcal { T } _ { y }$ identifies the spectral modes relevant to learningrate selection.

## 2.3 Optimal learning rates and transfer quantities

Fix a deterministic compact interval $\mathcal { T } = [ \eta _ { \ell } , \eta _ { u } ]$ . For every $n \in \mathbb { N }$ and $T \geq 1$ , define

$$
\eta _ { n , T } = \operatorname* { m i n } \arg \operatorname* { m i n } _ { \eta \in \mathbb { Z } } \phi _ { n , T } ( \eta ) , \qquad \eta _ { \infty , T } = \operatorname* { m i n } \arg \operatorname* { m i n } _ { \eta \ge 0 } \phi _ { \infty , T } ( \eta ) .
$$

Since $\phi _ { n , T }$ is continuous and $\mathcal { T }$ is compact, arg mi $\iota _ { \eta \in \mathcal { T } } \phi _ { n , T } ( \eta )$ is a nonempty compact set, so $\eta _ { n , T }$ is well defined. Moreover, Corollary C.4 and Proposition C.5 imply that for suficiently large $n , ~ \eta _ { n , T }$ is almost surely the unique minimizer of $\phi _ { n , T }$ over $\mathcal { T }$ . When $K y \neq 0$ Corollary C.4 also implies that the minimizer of $\phi _ { \infty , T }$ is positive and unique over $[ 0 , \infty )$

Following Ghosh et al. (2026), we define the transfer quantities.

Definition 2.2. For each width n and training horizon $T ,$ define the loss gap $a _ { n , T }$ , hyperparameter gap $b _ { n , T }$ , and transfer suboptimality gap $c _ { n , T }$ by

$$
a _ { n , T } = \left| \phi _ { n , T } \bigl ( \eta _ { n , T } \bigr ) - \phi _ { \infty , T } \bigl ( \eta _ { \infty , T } \bigr ) \right| , \quad b _ { n , T } = \left| \eta _ { n , T } - \eta _ { \infty , T } \right| , \quad c _ { n , T } = \phi _ { \infty , T } \bigl ( \eta _ { n , T } \bigr ) - \phi _ { \infty , T } \bigl ( \eta _ { \infty , T } \bigr ) .
$$

We say the hyperparameter transfer is fast when $c _ { n , T } = o _ { p } ( a _ { n , T } )$ holds.

The loss gap measures how much optimized performance varies across width, whereas the transfer suboptimality gap measures the additional target loss due specifically to using the proxy-selected learning rate. The hyperparameter gap records the displacement between the two minimizers. Fast transfer means that the cost of not retuning is negligible relative to the width-induced discrepancy in optimized losses. These comparisons always use a common horizon $T _ { i }$ , including along a sequence $T = T _ { n } $ ∞; we do not transfer between models trained for diferent numbers of updates.

## 3 Main results

## 3.1 Long-horizon structure of the infinite-width objective

We now let $T = T _ { n } \to \infty$ . In this regime, the limiting objective itself changes with $n ,$ and its optimizer moves toward the large-T limit. Thus, we first identify the moving deterministic reference in Section 3.1, and then compare the finite-width problem with this reference.

The spectral representation of $\phi _ { \infty , T }$ in Section 2.2 shows that, for large $T _ { i }$ , the learningrate-dependent loss is governed by the largest active contraction magnitude, $\operatorname* { m a x } _ { j \in \mathcal { I } _ { y } } | 1 -$ $\eta \lambda _ { j } / m |$ . The long-horizon optimizer must therefore balance the two extreme active modes. The following theorem identifies this balance.

Theorem 3.1 (Active-spectrum limit of the optimal learning rate). In the decomposition of $\begin{array} { r } { K = \sum _ { i = 1 } ^ { r } \lambda _ { j } P _ { j } } \end{array}$ , define $\lambda _ { + } ^ { y } = \operatorname* { m a x } _ { j \in \mathcal { I } _ { y } } \lambda _ { j }$ and $\lambda _ { - } ^ { y } = \operatorname* { m i n } _ { j \in \mathcal { I } _ { y } } \lambda _ { j }$ . Suppose $\lambda _ { + } ^ { y } > \lambda _ { - } ^ { y } > 0$ . Then as $T \to \infty$ , we have

$$
\eta _ { \infty , T } \to \eta _ { * } ^ { y } : = 2 m / ( \lambda _ { + } ^ { y } + \lambda _ { - } ^ { y } ) .\tag{1}
$$

$I f \lambda _ { + } ^ { y } = \lambda _ { - } ^ { y } = \lambda$ , then we have $\eta _ { \infty , T } = m / \lambda$ for all $T \geq 1$

In Theorem 3.1, observe that the optimizer limit $\eta _ { * } ^ { y }$ remains strictly inside the activespectrum stability interval $( 0 , 2 m / \lambda + ^ { y } )$ . If $\lambda _ { + } ^ { y } = \lambda _ { 1 }$ , it also lies inside the stability interval of the full iteration. Indeed, if the top active eigenvalue equals $\lambda _ { 1 } = \lambda _ { \mathrm { m a x } } ( K )$ , then we have $\eta _ { \ast } ^ { y } = 2 m / ( \lambda _ { 1 } + \lambda _ { - } ^ { y } ) < 2 m / \lambda _ { 1 }$ . We also note that the limit in Theorem 3.1 is closely related to the classical optimal step size for Richardson iteration; see, e.g., Saad (2003, Section 4.2.1).

Theorem 3.1 concerns the deterministic objective, which involves only modes active in $y .$ At finite width, initialization and matrix perturbations can also activate modes absent from this reference objective. To keep the long-horizon comparison governed by the same spectral endpoints, we impose the following stronger condition.

Assumption 3.2 (Endpoint activity). In the decomposition of $\begin{array} { r } { K = \sum _ { j = 1 } ^ { r } \lambda _ { j } P _ { j } } \end{array}$ , suppose $r \geq 2$ . The largest and smallest positive eigenvalues $\lambda _ { 1 }$ and $\lambda _ { r }$ of K are simple and satisfy $\langle y , u _ { 1 } \rangle \neq 0$ and $\langle y , u _ { r } \rangle \ne 0$ , where $u _ { 1 }$ and $u _ { r }$ are unit eigenvectors satisfying $K u _ { 1 } = \lambda _ { 1 } u _ { 1 }$ and $K u _ { r } = \lambda _ { r } u _ { r }$ . Thus we have $\lambda _ { + } ^ { y } = \lambda _ { 1 }$ and $\lambda _ { - } ^ { y } = \lambda _ { r }$

Under Assumption 3.2, set $\eta _ { * } = 2 m / ( \lambda _ { 1 } + \lambda _ { r } ) = \eta _ { * } ^ { y }$ and $q _ { * } = ( \lambda _ { 1 } - \lambda _ { r } ) / ( \lambda _ { 1 } + \lambda _ { r } )$ . The endpoint contraction factors at $\eta _ { * }$ are $- q _ { * }$ and $q _ { * }$ . Proposition E.1 and Corollary E.2 give a more quantitative picture: $\eta _ { \infty , T } - \eta _ { * } = O ( T ^ { - 1 } )$ , i.e., the optimizer approaches a fixed location, while

$$
\phi _ { \infty , T } ( \eta _ { \infty , T } ) - c _ { 0 } / ( 2 m ) = \Theta ( q _ { * } ^ { 2 T } ) , \qquad \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) = \Theta ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } )
$$

so that the optimizable part of the loss and the local curvature shrink exponentially. The growing-horizon transfer analysis must track finite-width perturbations relative to these changing scales.

## 3.2 Fast learning-rate transfer at growing horizons

We now turn from the deterministic long-horizon geometry of the infinite-width objective to finite-width fluctuations. By Proposition 2.1, all randomness relevant to the finite-width loss curve is entirely captured by $( \bar { \chi _ { n } ^ { ( 0 ) } } , A _ { n } )$ . Thus, the first step is to characterize how this pair fluctuates around its infinite-width limit. The following theorem identifies the widthdependent fluctuation scale and its limiting random direction.

Lemma 3.3 (Joint initialization central limit theorem). Suppose m and d are fixed. $D e \mathrm { . }$ fine $\alpha \sim \mathcal { N } ( 0 , 2 )$ . Let $G$ be a symmetric Gaussian $d \times d$ matrix satisfying $\mathbb { E } G _ { r s } = 0$ and $\mathbb { E } ( G _ { r s } G _ { t u } ) = d ^ { - 2 } ( \delta _ { r t } \delta _ { s u } + \delta _ { r u } \delta _ { s t } )$ . Define $\Xi = \alpha K + X G X ^ { \top }$ , and $Z \sim { \mathcal { N } } ( 0 , K )$ . Suppose that $\alpha , G$ , and $Z$ are mutually independent. Then we have

$$
{ \sqrt { n } } ( \chi _ { n } ^ { ( 0 ) } + y , A _ { n } - K ) { \xrightarrow { d } } ( Z , \Xi ) .
$$

Lemma 3.3 shows that the primitive finite-width perturbation has scale $\epsilon _ { n } = n ^ { - 1 / 2 }$ . The efect of this perturbation on learning-rate transfer, however, also depends on the training horizon. As established above, the infinite-width loss and its local curvature vary substantially with $T ,$ hence the sensitivities of the loss value and its learning-rate derivative to finite-width perturbations have their own horizon-dependent scales. To separate these efects from the width-dependent fluctuation scale $\epsilon _ { n }$ , define

$$
\rho _ { L , T } = T q _ { * } ^ { 2 T - 1 } , \qquad \rho _ { G , T } = T ^ { 2 } q _ { * } ^ { 2 T - 2 } , \qquad \kappa _ { T } = T ^ { 2 } q _ { * } ^ { 2 T - 2 } .
$$

The scales $\rho _ { L , T }$ and $\rho _ { G , T }$ measure the first-order sensitivity of the loss and its learningrate derivative to finite-width perturbations, respectively. The scale $\kappa _ { T }$ measures the local curvature $\phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } )$ . Thus, $\epsilon _ { n }$ quantifies the magnitude of the finite-width perturbation, whereas $\rho _ { L , T } , \rho _ { G , T }$ , and $\kappa _ { T }$ capture how the loss response, loss-gradient response, and local curvature vary with the training horizon. The following theorem combines these scales to characterize the fluctuations of the optimal learning rate and optimized loss, and the resulting rates of the three transfer gaps.

Theorem 3.4 (Growing-horizon fluctuations and transfer rates; Theorems G.6, G.8, and G.10, extracted). Suppose m and d are fixed and $T = T _ { n }$ satisfies $T \to \infty$ and $T / \sqrt { n }  0$ . Suppose Assumption 3.2 holds, and suppose $0 < \eta _ { \ell } < \eta _ { * } < \eta _ { u } < 2 m / \lambda _ { 1 }$ . Set $L _ { * } = \log ( c _ { 1 } \lambda _ { 1 } / ( c _ { r } \lambda _ { r } ) )$ $a _ { 1 } ^ { * } = \exp ( - \lambda _ { 1 } L _ { * } / ( \lambda _ { 1 } + \lambda _ { r } ) )$ , and $a _ { r } ^ { * } = \exp ( \lambda _ { r } L _ { * } / ( \lambda _ { 1 } + \lambda _ { r } ) )$ . Then we have

$$
\begin{array} { r } { ( \epsilon _ { n } \rho _ { G , T } / \kappa _ { T } ) ^ { - 1 } ( \eta _ { n , T } - \eta _ { \infty , T } ) \overset { d } { \longrightarrow } \Omega _ { \eta , * } : = - 2 m ( \lambda _ { 1 } + \lambda _ { r } ) ^ { - 2 } ( u _ { 1 } ^ { \top } \Xi u _ { 1 } + u _ { r } ^ { \top } \Xi u _ { r } ) , } \end{array}
$$

$$
\begin{array} { r } { ( \epsilon _ { n } \rho _ { L , T } ) ^ { - 1 } ( \phi _ { n , T } ( \eta _ { n , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) ) \stackrel { d } { \longrightarrow } \Omega _ { L , * } : = \langle Q _ { L , * } , \Xi \rangle _ { \mathrm { F } } , } \end{array}
$$

where $\Xi$ is defined in Lemma 3.3 and $Q _ { L , }$ <sub>∗</sub> is defined by $Q _ { L , * } = m ^ { - 2 } \eta _ { * } \big ( c _ { 1 } a _ { 1 } ^ { * } u _ { 1 } u _ { 1 } ^ { \top } - c _ { r } a _ { r } ^ { * } u _ { r } u _ { r } ^ { \top } \big )$ Consequently, we have

$$
a _ { n , T } = \Theta _ { p } \big ( \epsilon _ { n } \rho _ { L , T } \big ) = \Theta _ { p } \bigg ( \frac { T q _ { * } ^ { 2 T - 1 } } { \sqrt { n } } \bigg ) , \qquad b _ { n , T } = \Theta _ { p } \bigg ( \frac { \epsilon _ { n } \rho _ { G , T } } { { \kappa _ { T } } } \bigg ) = \Theta _ { p } \bigg ( \frac { 1 } { \sqrt { n } } \bigg ) ,
$$

and

$$
c _ { n , T } = \Theta _ { p } \bigl ( \kappa _ { T } b _ { n , T } ^ { 2 } \bigr ) = \Theta _ { p } \biggl ( \frac { \epsilon _ { n } ^ { 2 } \rho _ { G , T } ^ { 2 } } { \kappa _ { T } } \biggr ) = \Theta _ { p } \biggl ( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 2 } } { n } \biggr ) .
$$

Corollary 3.5 (Fast transfer at growing horizons). Under the assumptions of Theorem ${ 3 . 4 } ,$ we have

$$
\frac { c _ { n , T } } { a _ { n , T } } = \Theta _ { p } \bigg ( \frac { \epsilon _ { n } ^ { 2 } \rho _ { G , T } ^ { 2 } / \kappa _ { T } } { \epsilon _ { n } \rho _ { L , T } } \bigg ) = \Theta _ { p } \bigg ( \frac { T } { q _ { * } \sqrt { n } } \bigg ) \xrightarrow { p } 0 .
$$

By Corollary 3.5, under the assumptions of Theorem 3.4, fast learning-rate transfer persists at growing training horizons when $T = o ( { \sqrt { n } } )$

## 3.3 A fixed-horizon benchmark

Having established our main growing-horizon result in Theorem 3.4, we now turn to the fixed-horizon regime. In particular, we verify that, under the nondegeneracy condition below, our model recovers the general rates conjectured by Wen et al. (2026) for fixed T: $a _ { n , T } =$ $\Theta _ { p } ( n ^ { - 1 / 2 } ) , b _ { n , T } = \Theta _ { p } ( n ^ { - 1 / 2 } )$ , and $c _ { n , T } = \Theta _ { p } ( n ^ { - 1 } )$ . These rates also provide a useful baseline for Theorem 3.4. When T is fixed, the horizon-dependent factors are absorbed into constants, whereas allowing $T$ to grow introduces nontrivial interactions between width and training horizon.

Theorem 3.6 (Fixed-horizon transfer rates). Fix $m , d , T \in \mathbb { N }$ , and suppose $| \mathcal { T } _ { y } | = \# \{ j \in$ $[ r ] : c _ { j } > 0 \} \geq 2 . { } ^ { 2 }$ Assume in addition that $\eta _ { \infty , T } \in \left( \eta _ { \ell } , \eta _ { u } \right)$ . Then for $\epsilon _ { n } = n ^ { - 1 / 2 }$ , we have

$$
a _ { n , T } = \Theta _ { p } ( \epsilon _ { n } ) , \qquad b _ { n , T } = \Theta _ { p } ( \epsilon _ { n } ) , \qquad c _ { n , T } = \Theta _ { p } ( \epsilon _ { n } ^ { 2 } ) .
$$

Corollary 3.7 (Fast transfer at a fixed horizon). Under the assumptions of Theorem ${ \it 3 . 6 , }$ we have $c _ { n , T } / a _ { n , T } = \Theta _ { p } ( \epsilon _ { n } ) \xrightarrow { p } 0$

Corollary 3.7 implies that, under the assumptions of Theorem 3.6, learning-rate transfer is fast at every fixed horizon T.

## 3.4 Counterexample to fast transfer

The preceding results establish fast transfer both at growing horizons under the conditions of Theorem 3.4 and, more broadly at fixed horizons, under the nondegeneracy condition of Theorem 3.6. Fast transfer, however, is not automatic even when the training horizon is fixed. We now exhibit a counterexample in which fast transfer fails.

Proposition 3.8 (Failure of fast transfer with a single positive eigenvalue). Fix integers $2 \leq m \leq d ,$ deterministic data $\boldsymbol { X } \in \mathbb { R } ^ { m \times d }$ satisfying $X X ^ { \top } = I _ { m }$ , and a nonzero $y \in \mathbb { R } ^ { m }$ Note that $| \mathcal { I } _ { y } | = 1$ holds in this case. Set $\eta _ { 0 } : = m d \in ( \eta _ { \ell } , \eta _ { u } )$ , and fix an integer $T \geq 1$ Then we have

$$
a _ { n , T } = \Theta _ { p } ( n ^ { - T } ) , \qquad b _ { n , T } = \Theta _ { p } ( n ^ { - 1 / 2 } ) , \qquad c _ { n , T } = \Theta _ { p } ( n ^ { - T } ) , \qquad c _ { n , T } / a _ { n , T } = \Theta _ { p } ( 1 ) ,
$$

thus fast transfer fails.

Proposition 3.8 shows that convergence of the optimal learning rate alone is not suficient for fast transfer. Indeed, although $b _ { n , T } = \Theta _ { p } ( n ^ { - 1 / 2 } )$ holds exactly as in Theorem 3.6, the loss gap and the transfer suboptimality gap both scale as $\Theta _ { p } ( n ^ { - T } )$ . Hence the transfer penalty is not asymptotically negligible relative to the finite-width loss discrepancy. The mechanism behind this failure is a degeneracy of the leading loss sensitivity; Section 5 revisits this example from the local perturbation perspective.

## 4 Perturbation structure of hyperparameter transfer

## 4.1 Model-specific perturbation limits for Theorem 3.4

In this section, assume the conditions of Theorem 3.4. The purpose of this section is modelspecific: we identify the first-order finite-width perturbations and their limiting distributions in our shallow linear model. Section 4.2 subsequently abstracts the rate-level consequences of these calculations into a general local perturbation formulation.

For symmetric $A \in \mathbb { R } _ { \mathrm { s y m } } ^ { m \times m }$ and $\chi \in \mathbb R ^ { m }$ , define the deterministic loss map and score

$$
\Phi _ { T } ( \eta ; \chi , A ) = ( 2 m ) ^ { - 1 } \chi ^ { \top } B _ { A } ( \eta ) ^ { 2 T } \chi , \qquad F _ { T } ( \eta ; \chi , A ) = \chi ^ { \top } A B _ { A } ( \eta ) ^ { 2 T - 1 } \chi .
$$

Thus we have $\phi _ { n , T } = \Phi _ { T } ( \cdot ; \chi _ { n } ^ { ( 0 ) } , A _ { n } )$ and $\phi _ { \infty , T } = \Phi _ { T } ( \cdot ; - y , K )$ . Define $F _ { n , T }$ and $F _ { \infty , T }$ by $F _ { n , T } = F _ { T } ( \cdot ; \chi _ { n } ^ { ( 0 ) } , A _ { n } )$ and $F _ { \infty , T } = F _ { T } ( \cdot ; - y , K )$ . Proposition C.3 gives $\partial _ { \eta } \Phi _ { T } = - ( T / m ^ { 2 } ) F _ { T }$ so interior minimizers of $\Phi _ { T }$ are roots of $F _ { T }$

Linearization and horizon dependence. Lemma 3.3 gives ${ \sqrt { n } } ( \chi _ { n } ^ { ( 0 ) } + y , A _ { n } - K ) \ { \xrightarrow { \ d } } \quad$ $( Z , \Xi )$ . Writing $D _ { ( \chi , A ) }$ for the Fr´echet derivative with respect to $( \chi , A )$ at $( \eta _ { \infty , T } ; - y , K )$ , and noting that Corollary C.4 implies $F _ { \infty , T } ( \eta _ { \infty , T } ) = 0$ , Lemmas C.6, G.5, and G.7 yield

$$
F _ { n , T } ( \eta _ { \infty , T } ) = D _ { ( \chi , A ) } F _ { T } [ \chi _ { n } ^ { ( 0 ) } + y , A _ { n } - K ] + O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 3 } / n ) ,\tag{2}
$$

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = D _ { ( \chi , A ) } \Phi _ { T } [ \chi _ { n } ^ { ( 0 ) } + y , A _ { n } - K ] + O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } / n ) .\tag{3}
$$

In particular, Lemma C.6 gives

$$
D _ { ( \chi , A ) } F _ { T } ( \eta _ { \infty , T } ; - y , K ) [ h , E ] = g _ { F , T } ^ { \top } h + \langle Q _ { F , T } , E \rangle _ { \mathrm { F } } ,
$$

$$
D _ { ( \chi , A ) } \Phi _ { T } ( \eta _ { \infty , T } ; - y , K ) [ h , E ] = g _ { L , T } ^ { \top } h + \langle Q _ { L , T } , E \rangle _ { \mathrm { F } } ,
$$

where $g _ { F , T } , Q _ { F , T } , g _ { L , T } , Q _ { L , T }$ are the fixed Fr´echet coeficients defined in (12), (13), (14), and (15). Since $( \chi _ { n } ^ { ( 0 ) } + y , A _ { n } - K )$ is $O _ { p } ( n ^ { - 1 / 2 } )$ , the size of the finite-width fluctuations is determined by the T-dependence of the Fr´echet coeficients. Lemma G.4 shows that on range(K), as $T  \infty , Q _ { F , T } / ( T q _ { * } ^ { 2 T - 2 } )$ converges to $Q _ { F , }$ <sub>∗</sub> defined in Lemma G.4, and $Q _ { L , T } / ( T q _ { * } ^ { 2 T - 1 } )$ converges to $Q _ { L , * }$ . Lemma G.4 also shows that on range(K), $g _ { F , T }$ and $g _ { L , T }$ are $o ( T q _ { * } ^ { 2 T - 2 } )$ and $o ( T q _ { * } ^ { 2 T - 1 } )$ , respectively. Thus, at the infinite-width optimizer, the leading finite-width perturbations come from $A _ { n } - K$ , rather than from the initial prediction $f _ { n } ^ { ( 0 ) } ( X ) = \chi _ { n } ^ { ( 0 ) } + y$

Combining these coeficient asymptotics with Lemma 3.3 and (2), we have $F _ { n , T } ( \eta _ { \infty , T } ) =$ $\langle Q _ { F , T } , A _ { n } - K \rangle _ { \mathrm { F } } + o _ { p } ( T q _ { * } ^ { 2 T - 2 } / \sqrt { n } )$ , and hence

$$
\sqrt { n } ( T q _ { * } ^ { 2 T - 2 } ) ^ { - 1 } F _ { n , T } ( \eta _ { \infty , T } ) \stackrel { d } { \longrightarrow } \langle Q _ { F , * } , \Xi \rangle _ { \mathrm { F } } = - 2 m ^ { - 1 } \eta _ { * } h _ { * } ( u _ { 1 } ^ { \top } \Xi u _ { 1 } + u _ { r } ^ { \top } \Xi u _ { r } ) ,\tag{4}
$$

where $h _ { * } : = c _ { 1 } \lambda _ { 1 } a _ { 1 } ^ { * }$ ; the equality $c _ { 1 } \lambda _ { 1 } a _ { 1 } ^ { * } = c _ { r } \lambda _ { r } a _ { r } ^ { * }$ is established in (47). Similarly, Lemma 3.3 and (3) give $\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = \langle Q _ { L , T } , A _ { n } - K \rangle _ { \mathrm { F } } + o _ { p } ( T q _ { * } ^ { 2 T - 1 } / \sqrt { n } )$ , so that

$$
\sqrt { n } ( T q _ { * } ^ { 2 T - 1 } ) ^ { - 1 } ( \phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) ) \stackrel { d } { \longrightarrow } \langle Q _ { L , * } , \Xi \rangle _ { \mathrm { F } } = \Omega _ { L , * } .\tag{5}
$$

From the perturbation limits to the distributional results. Lemma G.2 localizes the finite-width optimizer at $\eta _ { n , T } - \eta _ { \infty , T } = O _ { p } ( n ^ { - 1 / 2 } ) = o _ { p } ( T ^ { - 1 } )$ . Together with the local score-derivative asymptotics, the score limit (4) yields $\sqrt { n } ( \eta _ { n , T } - \eta _ { \infty , T } ) \stackrel { d } { \longrightarrow } \Omega _ { \eta , * }$ , as stated in Theorem G.6. Likewise, finite-width reoptimization changes the loss by $O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } / n ) =$ $o _ { p } ( T q _ { * } ^ { 2 T - 1 } / { \sqrt { n } } )$ , so the limit in (5) is unchanged when $\eta _ { \infty , T }$ is replaced by $\eta _ { n , T ; }$ yielding Theorem G.8. The remaining rate-level consequences of these perturbation estimates are organized abstractly in Section 4.2, while the complete model-specific arguments are given in Appendix G.

## 4.2 A local perturbation formulation for horizon-dependent transfer

Our fixed- and growing-horizon proofs can be organized through a local perturbation argument whose ingredients are closely related to earlier analyses of fast hyperparameter transfer. Ghosh et al. (2026) use local strong convexity to relate hyperparameter displacement to transfer suboptimality for a fixed limiting objective. In the growing-horizon setting, Wen et al. (2026) track horizon-dependent score perturbations, local curvature, optimizer localization, and finite-width loss fluctuations.

The purpose of this section is to recast these proof ingredients in a common horizondependent formulation that makes the relevant scales explicit. Section 4.1 provides the model-specific first-order perturbation estimates and limiting distributions for our shallow linear model; here we retain only the corresponding loss-response, loss-gradient-response, curvature, and localization scales in order to explain the resulting rates of $a _ { n , T } , \ b _ { n , T }$ , and $c _ { n , T }$ and the condition for fast transfer.

We consider a scalar hyperparameter $\eta ,$ and allow $T = T _ { n }$ to be fixed or diverging. Write

$$
\phi _ { n , T } ( \eta ) = \Phi _ { T } ( \eta ; \gamma _ { n } ) , \qquad \phi _ { \infty , T } ( \eta ) = \Phi _ { T } ( \eta ; \gamma _ { \infty } ) , \qquad \| \gamma _ { n } - \gamma _ { \infty } \| = O _ { p } ( \epsilon _ { n } ) ,
$$

where $\gamma _ { \infty }$ is deterministic and $\epsilon _ { n } \to 0$ . All stochastic orders below are along the chosen sequence $( n , T _ { n } )$ , and the scales $\rho _ { L , T } , \rho _ { G , T } , \kappa _ { T } , r _ { T }$ are deterministic and positive. Assume that the objectives are twice continuously diferentiable in $\eta$ near the search interval I, and that the derivative in $\gamma$ appearing below exists. Let $\eta _ { \infty , T }$ be a global minimizer of $\phi _ { \infty , T }$ lying in the interior of I, and let $\eta _ { n , T }$ minimize $\phi _ { n , T }$ over I.

Define the loss gradient $G _ { T } ( \eta ; \gamma ) : = \partial _ { \eta } \Phi _ { T } ( \eta ; \gamma )$ . At the moving infinite-width optimizer, suppose $G _ { T } ( \eta _ { \infty , T } ; \gamma _ { \infty } ) = 0$ and $G _ { T } ( \eta _ { \infty , T } ; \gamma _ { n } ) = O _ { p } ( \epsilon _ { n } \rho _ { G , T } )$ , and assume the first-order loss expansion

$$
\begin{array} { r l } & { d _ { n , T } : = \phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = D _ { \gamma } \Phi _ { T } ( \eta _ { \infty , T } ; \gamma _ { \infty } ) \big [ \gamma _ { n } - \gamma _ { \infty } \big ] + o _ { p } ( \epsilon _ { n } \rho _ { L , T } ) , } \\ & { | d _ { n , T } | = \Theta _ { p } \big ( \epsilon _ { n } \rho _ { L , T } \big ) . } \end{array}\tag{6}
$$

Here $\rho _ { L , T }$ and $\rho _ { G , T }$ describe the loss and loss-gradient responses to the finite-width perturbation, respectively. The lower bound in (6) holds, for example, if the normalized first-order loss term converges to a Gaussian with positive variance.

Set $\kappa _ { T } : = \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) > 0$ and $J _ { T } : = [ \eta _ { \infty , T } - r _ { T } , \eta _ { \infty , T } + r _ { T } ] \subset \mathrm { i n t } ( \mathbb { Z } )$ . Suppose first that optimizer localization $| \eta _ { n , T } - \eta _ { \infty , T } | = o _ { p } ( r _ { T } )$ has been established independently. The argument below is conditional on this localization; the scale comparison derived below is a consistency requirement for the local expansion. Writing $\begin{array} { r } { \underline { { \kappa } } _ { n , T } : = \operatorname* { i n f } _ { \eta \in J _ { T } } \partial _ { \eta } G _ { T } ( \eta ; \gamma _ { n } ) } \end{array}$ , assume $\mathrm P ( \underline { { \kappa } } _ { n , T } > 0 )  1 , \kappa _ { T } / \underline { { \kappa } } _ { n , T } = O _ { p } ( 1 )$ , and $\begin{array} { r } { \operatorname* { s u p } _ { \eta \in J _ { T } } \{ | \phi _ { n , T } ^ { \prime \prime } ( \eta ) | + | \phi _ { \infty , T } ^ { \prime \prime } ( \eta ) | \} = O _ { p } ( \kappa _ { T } ) } \end{array}$ . These conditions ensure positive finite-width curvature and uniform local control. With probability tending to one, both optimizers are interior, and the mean value theorem gives

$$
b _ { n , T } = | \eta _ { n , T } - \eta _ { \infty , T } | = | G _ { T } ( \eta _ { \infty , T } ; \gamma _ { n } ) / \partial _ { \eta } G _ { T } ( \tilde { \eta } _ { n , T } ; \gamma _ { n } ) | = O _ { p } ( \epsilon _ { n } \rho _ { G , T } / \kappa _ { T } ) ,
$$

where $\tilde { \eta } _ { n , T }$ lies between the two optimizers. The bound is compatible with the assumed localization whenever $\epsilon _ { n } \rho _ { G , T } / \kappa _ { T } = o ( r _ { T } )$

(a) Local loss surface  
![](images/c88018192dd7f879c8a6dd3d45b1ef605e87ef0e059a67d1fb5e82c1bd6ca709.jpg)

![](images/615a484a28f653d510695c5d3aad4cbe3569255fb1da24a32b7c95ed73b6cb66.jpg)

(c) Loss-gradient slices  
![](images/5fc987b108266b366140106bbee138144f214e67fe5a818f6ade4a05e7a7019c.jpg)  
Figure 1: Illustration of the growing-horizon rate decomposition. (a) The local loss surface $\Phi _ { T } ( \eta ; \gamma )$ . A perturbation along the γ-axis from $\gamma _ { \infty }$ to $\gamma _ { n } ,$ of size $O _ { p } ( \epsilon _ { n } )$ , afects both the loss value and its gradient with respect to η. (b) Loss slices at $\gamma _ { \infty }$ and $\gamma _ { n }$ . At $\eta _ { \infty , T } ,$ , the induced loss perturbation has scale $\Theta _ { p } \big ( \epsilon _ { n } \rho _ { L , T } \big )$ , which determines the leading scale of $a _ { n , T }$ . (c) Lossgradient slices. The finite-width perturbation changes the gradient at $\eta _ { \infty , T }$ by $O _ { p } ( \epsilon _ { n } \rho _ { G , T } )$ Local curvature of order $\kappa _ { T }$ converts this gradient perturbation into a hyperparameter gap $b _ { n , T }$

Taylor expansion of the infinite-width objective then yields

$$
c _ { n , T } = ( 1 / 2 ) \phi _ { \infty , T } ^ { \prime \prime } ( \bar { \eta } _ { n , T } ) ( \eta _ { n , T } - \eta _ { \infty , T } ) ^ { 2 } = O _ { p } ( \kappa _ { T } b _ { n , T } ^ { 2 } ) = O _ { p } ( \epsilon _ { n } ^ { 2 } \rho _ { G , T } ^ { 2 } / \kappa _ { T } ) ,
$$

where $\bar { \eta } _ { n , T }$ lies between the two optimizers. For fixed T with nondegenerate local curvature, this is the same local quadratic relation underlying $c _ { n } \ = \ \Theta _ { p } ( b _ { n } ^ { 2 } )$ in Ghosh et al. (2026, Proposition 1). Here we keep the factor $\kappa _ { T }$ explicit because the curvature may vanish as $T$ grows.

Similarly, the improvement from optimizing the finite-width objective satisfies $R _ { n , T } : =$ $\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { n , T } ( \eta _ { n , T } ) = O _ { p } ( \epsilon _ { n } ^ { 2 } \rho _ { G , T } ^ { 2 } / \kappa _ { T } )$ . Since $a _ { n , T } = | d _ { n , T } - R _ { n , T } |$ , a suficient condition for this optimization correction to be negligible relative to the first-order loss fluctuation is $\epsilon _ { n } \rho _ { G , T } ^ { 2 } / ( \kappa _ { T } \rho _ { L , T } ) \to 0$ . Under this condition, we have

$$
a _ { n , T } = \Theta _ { p } ( \epsilon _ { n } \rho _ { L , T } ) , \qquad c _ { n , T } / a _ { n , T } = O _ { p } \bigl ( \epsilon _ { n } \rho _ { G , T } ^ { 2 } / ( \kappa _ { T } \rho _ { L , T } ) \bigr ) = o _ { p } ( 1 ) .
$$

The lower bound in (6) is needed for this comparison; an $O _ { p }$ bound on the loss fluctuation alone does not sufice. Figure 1 illustrates these scales.

In our model, the correspondence is $( \gamma _ { n } , \gamma _ { \infty } ) = ( ( \chi _ { n } ^ { ( 0 ) } , A _ { n } ) , ( - y , K ) ) , G _ { T } = - ( T / m ^ { 2 } ) F _ { T } ,$ $\epsilon _ { n } = n ^ { - 1 / 2 } , \rho _ { L , T } \asymp T q _ { * } ^ { 2 T - 1 } , \rho _ { G , T } \asymp T ^ { 2 } q _ { * } ^ { 2 T - 2 } , \kappa _ { T } \asymp T ^ { 2 } q _ { * } ^ { 2 T - 2 }$ , and $r _ { T } \asymp T ^ { - 1 }$ . Consequently, we have $\epsilon _ { n } \rho _ { G , T } / ( \kappa _ { T } r _ { T } ) \asymp T / \sqrt { n }$ and $\epsilon _ { n } \rho _ { G , T } ^ { 2 } / ( \kappa _ { T } \rho _ { L , T } ) \asymp T / ( q _ { * } \sqrt { n } )$ . Since $q _ { * } > 0$ is fixed, $T =$ $o ( \sqrt { n } )$ makes both ratios vanish. Thus the general formulation recovers the rate conclusions of Theorem 3.4 and the fast-transfer conclusion of Corollary 3.5. The fixed-horizon proof of Theorem 3.6 uses the same local structure, with horizon-dependent scales fixed as $n \to \infty$ The model-specific estimates underlying these scales difer substantially from those in Wen et al. (2026). Here the perturbation is finite-dimensional and the relevant matrix powers admit uniform control near the optimum, whereas their source-capacity setting involves a spectrum accumulating at zero and an optimizer approaching the stability edge, requiring spectral-tail-dependent localization and fluctuation estimates.

## 5 Implications for mechanisms of learning-rate transfer

Relation to the top-k loss decomposition. Ghosh et al. (2026) decompose the linearized loss change along an optimization trajectory into a dominant top-k component and a residual component, and conjecture that fast transfer arises when the dominant component becomes rapidly width-stable while the width-sensitive residual has little efect on the hyperparameter optimum. In our model, the full-batch GD gradient with respect to $W _ { 1 }$ has rank at most one. Since the update is proportional to this gradient, the associated gradient–update alignment matrix also has rank at most one, so its stepwise linearized loss change is captured by $k = 1$ . This exact top-1 representation, however, is only a structural statement: it holds both in the fast-transfer regime of Theorem 3.4 and in Proposition 3.8, where $c _ { n , T } / a _ { n , T } = \Theta _ { p } ( 1 )$ for every fixed $T \geq 1$ . Thus, low-rank concentration of the stepwise linearized loss change alone does not determine the transfer rate; the local behavior of the finite-width perturbation around the infinite-width optimum also matters.

Section 4.2 characterizes this local behavior in terms of three scales: the loss-value response $\rho _ { L , T }$ , the hyperparameter-gradient response $\rho _ { G , T }$ , and the local curvature κ<sub>T</sub>. Under its assumptions, the leading loss perturbation has scale $\Theta _ { p } \big ( \epsilon _ { n } \rho _ { L , T } \big )$ , whereas both the transfer penalty and the improvement from finite-width reoptimization are bounded by $O _ { p } ( \epsilon _ { n } ^ { 2 } \rho _ { G , T } ^ { 2 } / \kappa _ { T } )$ . Hence fast transfer follows when $\epsilon _ { n } \rho _ { G , T } ^ { 2 } / ( \kappa _ { T } \rho _ { L , T } ) \to 0$ . The $T = 1$ case of Proposition 3.8 illustrates why the loss-response term is essential: in this counterexample, the first-order loss response vanishes, while the loss-gradient response remains nondegenerate and $\kappa _ { 1 } > 0$ , and $a _ { n , 1 }$ and $c _ { n , 1 }$ are both $\Theta _ { p } ( n ^ { - 1 } )$ . For fixed $T \geq 2$ , the first-order responses and the curvature all degenerate, placing the example in a higher-order regime outside the framework of Section 4.2.

Our growing-horizon limits also clarify the directional dependence of the loss-gradient and loss-value responses in Section 4.2. In Theorem 3.4, the normalized optimizer displacement and optimized-loss fluctuation converge in distribution to nonzero deterministic multiples of $\ell _ { \eta } ( \Xi )$ and $\ell _ { L } ( \Xi )$ , respectively, where $\ell _ { \eta } ( E ) : = u _ { 1 } ^ { \top } E u _ { 1 } + u _ { r } ^ { \top } E u _ { r }$ and $\ell _ { L } ( E ) : = c _ { 1 } a _ { 1 } ^ { * } u _ { 1 } ^ { \top } E u _ { 1 } -$ $c _ { r } a _ { r } ^ { * } u _ { r } ^ { \top } E u _ { r }$ . As shown in Section 4.1, these limits arise from $- ( T / m ^ { 2 } ) \langle Q _ { F , T } , A _ { n } - K \rangle _ { \mathrm { F } }$ and $\langle Q _ { L , T } , A _ { n } - K \rangle _ { \mathrm { F } } $ , which are the leading terms of the respective first-order responses at $\eta _ { \infty , T } ,$ whose scales are $\epsilon _ { n } \rho _ { G , T }$ and $\epsilon _ { n } \rho _ { L , T }$ . Although $\ell _ { \eta }$ and $\ell _ { L }$ depend on the same two endpoint coordinates, they are not proportional. Hence there are perturbation directions with a nonzero leading loss response but a vanishing leading loss-gradient response, and therefore no leading-order optimizer displacement along those directions. This is qualitatively consistent with the idea that width-dependent loss changes can have little efect on hyperparameter selection, as in the mechanism proposed by Ghosh et al. (2026); our argument concerns local responses to $A _ { n } - K$ , rather than a decomposition of the optimization trajectory.

Relation to spectral explanations of transfer. Our results show that, even in this solvable setting, the width consistency of a single leading eigenvalue does not determine the long-horizon optimal learning rate. Noci et al. (2024) study the width consistency of leading parameter-space Hessian eigenvalues, particularly sharpness. Our result highlights a distinction between the stability boundary and the optimal learning rate. Under Assumption 3.2, Theorem 3.1 implies $\eta _ { \infty , T }  2 m / ( \lambda _ { 1 } + \lambda _ { r } ) < 2 m / \lambda _ { 1 }$ as $T \to \infty$ . Thus the largest eigenvalue determines the stability boundary, whereas both spectral endpoints determine the long-horizon optimum.

## 6 Conclusion

We studied fast learning-rate transfer across width in a shallow linear network under $\mu \mathrm { P }$ at growing training horizons. Specifically, we characterized the growing-horizon infinitewidth optimizer through the active spectral endpoints and showed that, under endpoint activity and $T = o ( { \sqrt { n } } )$ , the three transfer quantities obey horizon-dependent rates that yield $c _ { n , T } = o _ { p } ( a _ { n , T } )$ . The local perturbation formulation organizes transfer in terms of the loss-value response, loss-gradient response, and local curvature, providing a sensitivity-based explanation for fast transfer and for the dependence of its rates on the spectral structure of the problem.

Our analysis is restricted to a solvable setting of linear networks with a single trainable linear layer. Extending the analysis to nonlinear or deep networks is therefore an important next step, as is studying transfer simultaneously across multiple scaling axes such as width and depth. Another limitation of the current framework is its reliance on nondegenerate firstorder sensitivities: the fixed-horizon counterexample shows that these first-order responses can vanish, requiring a higher-order analysis. Developing such a theory would help provide a clearer picture of the mechanisms governing hyperparameter transfer.

## Acknowledgments

We thank Denny Wu for valuable discussions. Mana Sakai was supported by RIKEN Junior Research Associate Program and JST FOREST (Grant No. JPMJFR216I). Masaaki Imaizumi was supported by JSPS KAKENHI (Grant No. 24K02904), JST CREST (Grant No. JPMJCR21D2), JST FOREST (Grant No. JPMJFR216I), and JST BOOST (Grant No. JPMJBY24A9).

## A Additional related work

Parameterization and the empirical scope of transfer. Successful transfer depends on more than the name of a parameterization. Lingle (2024) examines µ-transfer under practical training choices, while Everett et al. (2024) study alignment assumptions and optimizer-dependent scaling exponents. In particular, the latter show that alternative parameterizations can also support transfer when equipped with suitable layerwise learning rates. These findings concern the design and empirical efectiveness of scaling prescriptions. Our analysis instead fixes the initialization and update rule, and compares the resulting optimizer mismatch, optimized loss fluctuation, and transfer penalty.

Conditional separation-of-scales explanations. Hong and Wang (2025) study a separation between macroscopic and microscopic quantities under a width-dominance regime, with implications for early-stage hyperparameter selection. This provides a diferent perspective on the stability of tuning across scale. Our results directly analyze the exact terminal training loss in a specified finite-dimensional model and track the dependence of its local perturbation scales on a growing training horizon.

Transfer across depth, modules, and architectural changes. Depthwise transfer has been investigated through infinite-depth limits and residual-network dynamics (Yang et al., 2024; Bordelon et al., 2024), and through complete feature learning in deep Transformers (Dey et al., 2025). Mlodozeniec et al. (2026) further address transfer across modules, width, depth, batch size, and duration. Other work identifies architecture-specific scaling issues arising from large vocabularies or normalized Transformer parameterizations (Hayou and Liu, 2025; Shigida et al., 2026). These studies address scaling axes and training mechanisms outside the scope of our model.

Training schedules and scaling laws. Bordelon and Mori (2026) analyze terminal-loss optimization over learning-rate schedules in a random feature model. More broadly, empirical and dynamical scaling-law studies relate performance to model size, data, and training time (Hofmann et al., 2022; Bordelon et al., 2024). Our question is narrower: we transfer a constant learning rate across width while holding the source and target horizons equal, and quantify the resulting relative penalty.

Spectral explanations and sharpness. Noci et al. (2024) connect transfer to the width consistency of leading Hessian eigenvalues, while Lauditi et al. (2026) study spectral dynamics and feature learning in deep networks. These perspectives are related to the edge-ofstability behavior documented by Cohen et al. (2021). In our model, the largest eigenvalue determines the linear stability boundary, whereas the long-horizon optimal constant learning rate balances two active spectral endpoints.

![](images/b075bbd0cc7c28d39667d096a0dc9c7626c6048f4cd54bc509d8988cce7bc5c6.jpg)

![](images/d9d9e40f2c7397234a561cf0038c59c3d98977124b5e6993c1719b7382595c98.jpg)

![](images/2aa511c763a62b2c1bd5a2fca9cef1923a91f384e4c4ccb0a0c7ce9d3fbd5f4c.jpg)  
Figure 2: Numerical learning-rate transfer. (a) $a _ { n , T } , b _ { n , T } , c _ { n , T }$ at fixed $T \ = \ 1 0 .$ . (b) $\tilde { a } _ { n , T } , b _ { n , T } , \tilde { c } _ { n , T }$ at $T _ { n } = \lceil 2 n ^ { 0 . 1 5 } \rceil$ . (c) Per-initialization ratios $c _ { n , T } / a _ { n , T }$ , summarized by means (solid) and medians (dotted); the reference rates are $n ^ { - 1 / 2 }$ for fixed T and $T / ( q _ { * } { \sqrt { n } } )$ for growing T.

## B Numerical experiments

Setup. We use the model, initialization, and full-batch training rule of Section 2.1, with $m = d = 4$ and fixed data

$$
K = \mathrm { d i a g } ( 7 , 4 , 2 , 1 ) , \qquad X = 2 K ^ { 1 / 2 } , \qquad y = ( 1 , 1 , 1 , 1 ) ^ { \top } .
$$

The learning-rate interval is $I = [ 0 . 0 5 , 1 . 1 2 ]$ . Thus Assumption 3.2 holds with $\eta _ { * } = 1$ and $q _ { * } = 3 / 4$ . At each width $n \in \{ 2 ^ { 8 } , 2 ^ { 9 } , . . . , 2 ^ { 2 0 } \}$ , we draw 256 independent initializations per experiment. Figure 2 compares the fixed-horizon regime $T = 1 0$ with the growing-horizon regime $T _ { n } = \lceil 2 n ^ { 0 . 1 5 } \rceil$ , which ranges from 5 to 16.

Computation and estimation. For growing horizons, define

$$
\tilde { a } _ { n , T } = \frac { a _ { n , T } } { T q _ { * } ^ { 2 T - 1 } } , \qquad \tilde { c } _ { n , T } = \frac { c _ { n , T } } { T ^ { 2 } q _ { * } ^ { 2 T - 2 } } .
$$

We estimate unweighted least-squares slopes by regressing the log empirical means on log n over $n = 2 ^ { 1 3 } , \ldots , 2 ^ { 2 0 }$ : the largest eight widths for fixed T, and the subset satisfying $T _ { n } / \sqrt { n } \leq$ 0.1 for the growing schedule.

Results and implications. In panels (a) and (b) of Figure 2, all six slopes are close to the reference exponents $( - 1 / 2 , - 1 / 2 , - 1 )$ , consistent with Theorems 3.6 and 3.4. The overall decline of $c _ { n , T } / a _ { n , T }$ in panel (c) supports fast transfer along the tested schedule and illustrates the quadratic transfer penalty discussed in Section 4.2. Mean ratios are noisy and nonmonotone because small $a _ { n , T }$ values can dominate individual ratios; medians show a smoother decline. The theoretical rates are in probability and do not, by themselves, imply corresponding rates for expectations.

# C Basic properties and finite-width fluctuations

## C.1 Proof of Proposition 2.1

First, we compute

$$
\nabla _ { W _ { 1 } } \mathcal { L } _ { n } ( W _ { 1 } ^ { ( t ) } ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \chi _ { n , i } ^ { ( t ) } V H _ { n , i \cdot } = \frac { 1 } { m } V ( \chi _ { n } ^ { ( t ) } ) ^ { \top } H _ { n } .
$$

Thus we have $\begin{array} { r } { W _ { 1 } ^ { ( t + 1 ) } = W _ { 1 } ^ { ( t ) } - \frac { \eta } { m } V ( \chi _ { n } ^ { ( t ) } ) ^ { \top } H _ { n } } \end{array}$ , which implies

$$
f _ { n } ^ { ( t + 1 ) } ( X ) = H _ { n } ( W _ { 1 } ^ { ( t ) } ) ^ { \top } V - \frac { \eta } { m } H _ { n } H _ { n } ^ { \top } \chi _ { n } ^ { ( t ) } V ^ { \top } V = f _ { n } ^ { ( t ) } ( X ) - \frac { \eta } { m } A _ { n } \chi _ { n } ^ { ( t ) } .
$$

Subtracting y from both sides yields

$$
\begin{array} { r } { \chi _ { n } ^ { ( t + 1 ) } = B _ { n } ( \eta ) \chi _ { n } ^ { ( t ) } , } \end{array}
$$

which implies

$$
\begin{array} { r } { \chi _ { n } ^ { ( T ) } = B _ { n } ( \eta ) ^ { T } \chi _ { n } ^ { ( 0 ) } . } \end{array}
$$

Therefore we have

$$
\phi _ { n , T } ( \eta ) = \mathcal { L } _ { n } ( W _ { 1 } ^ { ( T ) } ) = \frac { 1 } { 2 m } \| \chi _ { n } ^ { ( T ) } \| ^ { 2 } = \frac { 1 } { 2 m } \left\| B _ { n } ( \eta ) ^ { T } \chi _ { n } ^ { ( 0 ) } \right\| ^ { 2 } = \frac { 1 } { 2 m } ( \chi _ { n } ^ { ( 0 ) } ) ^ { \top } B _ { n } ( \eta ) ^ { 2 T } \chi _ { n } ^ { ( 0 ) } .
$$

## C.2 Proof of Lemma 3.3

First, we show the weak convergence of the marginal distribution ${ \sqrt { n } } ( A _ { n } - K ) \ { \xrightarrow { d } } \equiv$ . Noting that $A _ { n } = ( n \| V \| ^ { 2 } ) X ( n ^ { - 1 } W _ { 0 } ^ { \top } W _ { 0 } ) X ^ { \top }$ holds, we decompose

$$
\begin{array} { l } { { A _ { n } - K = ( n \| V \| ^ { 2 } - 1 ) K + X ( n ^ { - 1 } W _ { 0 } ^ { \top } W _ { 0 } - d ^ { - 1 } I _ { d } ) X ^ { \top } } } \\ { { \phantom { A _ { n } - K } + ( n \| V \| ^ { 2 } - 1 ) X ( n ^ { - 1 } W _ { 0 } ^ { \top } W _ { 0 } - d ^ { - 1 } I _ { d } ) X ^ { \top } . } } \end{array}
$$

Define $\alpha _ { n } = \sqrt { n } ( n \| V \| ^ { 2 } - 1 )$ and $G _ { n } = \sqrt { n } ( n ^ { - 1 } W _ { 0 } ^ { \top } W _ { 0 } - d ^ { - 1 } I _ { d } )$ . Since the variables $n V _ { j }$ are i.i.d. $\mathcal { N } ( 0 , 1 )$ , the central limit theorem gives

$$
\alpha _ { n } = \frac { 1 } { \sqrt { n } } \sum _ { j = 1 } ^ { n } ( ( n V _ { j } ) ^ { 2 } - 1 ) \stackrel { \_ d } { \longrightarrow } \alpha , \qquad \alpha \sim N ( 0 , 2 ) .
$$

Next, noting that $( W _ { 0 , j } . ) ^ { \top } \sim N ( 0 , d ^ { - 1 } I _ { d } )$ independently, the multivariate central limit theorem gives

$$
G _ { n } = { \frac { 1 } { \sqrt { n } } } \sum _ { j = 1 } ^ { n } ( ( W _ { 0 , j \cdot } ) ^ { \top } W _ { 0 , j \cdot } - d ^ { - 1 } I _ { d } ) \xrightarrow { d } G ,
$$

where G is the Gaussian matrix specified in the statement. Since V and $W _ { 0 }$ are independent, $\alpha _ { n }$ and $G _ { n }$ are independent for every n, and hence the joint convergence in distribution holds:

$$
( \alpha _ { n } , G _ { n } ) \xrightarrow { d } ( \alpha , G ) .
$$

Combining the results, we have

$$
\sqrt { n } ( A _ { n } - K ) = \alpha _ { n } K + X G _ { n } X ^ { \top } + O _ { p } ( n ^ { - 1 / 2 } ) \stackrel { d } { \longrightarrow } \alpha K + X G X ^ { \top } = \Xi .\tag{7}
$$

Fix $u \in \mathbb { R } ^ { m }$ and $T \in \mathbb { R } ^ { m \times m }$ . Note that $\chi _ { n } ^ { ( 0 ) } + y = f _ { n } ^ { ( 0 ) } ( X ) = H _ { n } ( W _ { 1 } ^ { ( 0 ) } ) ^ { \top } V$ holds. Since $W _ { 1 } ^ { ( 0 ) }$ is independent of $( W _ { 0 } , V )$ , we have

$$
\sqrt { n } ( \chi _ { n } ^ { ( 0 ) } + y ) \mid W _ { 0 } , V \sim N ( 0 , A _ { n } ) ,\tag{8}
$$

which implies

$$
\mathbb { E } [ \exp ( i u ^ { \top } \sqrt { n } ( \chi _ { n } ^ { ( 0 ) } + y ) ) \mid W _ { 0 } , V ] = \exp \left( - \frac { 1 } { 2 } u ^ { \top } A _ { n } u \right) .
$$

Thus we have

$$
\begin{array} { r l } & { \mathbb { E } [ \exp ( i u ^ { \top } \sqrt { n } ( \chi _ { n } ^ { ( 0 ) } + y ) + i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } ) ] } \\ & { = \mathbb { E } ( \mathbb { E } [ \exp ( i u ^ { \top } \sqrt { n } ( \chi _ { n } ^ { ( 0 ) } + y ) + i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } ) \mid W _ { 0 } , V ] ) } \\ & { = \mathbb { E } \left[ \exp \left( - \frac { 1 } { 2 } u ^ { \top } A _ { n } u \right) \exp ( i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } ) \right] } \\ & { = \mathbb { E } \left[ \left( \exp \left( - \frac { 1 } { 2 } u ^ { \top } A _ { n } u \right) - \exp \left( - \frac { 1 } { 2 } u ^ { \top } K u \right) \right) \exp \left( i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } \right) \right] } \\ & { \quad + \exp \left( - \frac { 1 } { 2 } u ^ { \top } K u \right) \mathbb { E } [ \exp ( i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } ) ] . } \end{array}
$$

Note that (7) implies $A _ { n } \xrightarrow [ ] { p } K$ . Thus we compute the limit of the first term as

$$
\begin{array} { r l } & { \left| \mathbb { E } \left[ \left( \exp \left( - \frac { 1 } { 2 } u ^ { \top } A _ { n } u \right) - \exp \left( - \frac { 1 } { 2 } u ^ { \top } K u \right) \right) \exp \left( i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } \right) \right] \right| } \\ & { \leq \mathbb { E } \left| \exp \left( - \frac { 1 } { 2 } u ^ { \top } A _ { n } u \right) - \exp \left( - \frac { 1 } { 2 } u ^ { \top } K u \right) \right| } \\ & { \to 0 . } \end{array}
$$

For the second term, (7) implies

$$
\mathbb { E } [ \exp ( i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } ) ]  \mathbb { E } [ \exp ( i \langle T , \Xi \rangle _ { \mathrm { F } } ) ] .
$$

Hence the joint characteristic function satisfies

$$
\mathbb { E } [ \exp ( i u ^ { \top } \sqrt { n } ( \chi _ { n } ^ { ( 0 ) } + y ) + i \langle T , \sqrt { n } ( A _ { n } - K ) \rangle _ { \mathrm { F } } ) ]  \exp ( - \frac { 1 } { 2 } u ^ { \top } K u ) \mathbb { E } [ \exp ( i \langle T , \Xi \rangle _ { \mathrm { F } } ) ] ,
$$

which is the characteristic function of $( Z , \Xi )$ with $Z \sim N ( 0 , K )$ independent of $\Xi$ □

## C.3 Infinite-width loss for fixed horizon

Corollary C.1 (Infinite-width loss for fixed T). Fix $T \in \mathbb { N }$ and $\eta \in \mathbb { R }$ . Then

$$
\phi _ { n , T } ( \eta ) \stackrel { p } { \longrightarrow } \phi _ { \infty , T } ( \eta ) : = \frac { 1 } { 2 m } \left. B _ { \infty } ( \eta ) ^ { T } y \right. ^ { 2 } = \frac { 1 } { 2 m } y ^ { \top } B _ { \infty } ( \eta ) ^ { 2 T } y .
$$

The convergence is uniform on every deterministic compact interval of learning rates.

Proof. By Lemma 3.3, $( \chi _ { n } ^ { ( 0 ) } , A _ { n } ) \stackrel { p } { \longrightarrow } ( - y , K )$ . For fixed T, the map

$$
( \chi , A , \eta ) \longmapsto { \frac { 1 } { 2 m } } \chi ^ { \top } \left( I _ { m } - { \frac { \eta } { m } } A \right) ^ { 2 T } \chi
$$

is a polynomial in the entries of $( \chi , A , \eta )$ . Thus, the continuous mapping theorem proves $\phi _ { n , T } ( \eta ) \stackrel { p } { \longrightarrow } \phi _ { \infty , T } ( \eta )$ for every fixed $\eta .$ Furthermore, for fixed $T .$ both $\phi _ { n , T }$ and $\phi _ { \infty , T }$ are polynomials in η of degree at most $2 T$ . Their coeficients $c _ { n , 0 } , c _ { n , 1 } , \ldots , c _ { n , 2 T }$ are continuous functions of $( \chi _ { n } ^ { ( 0 ) } , A _ { n } )$ and converge in probability to the corresponding coeficients $c _ { \infty , 0 } , c _ { \infty , 1 } , \ldots , c _ { \infty , 2 T }$ determined by $( - y , K )$ . Thus, for any compact interval $[ \eta _ { l } , \eta _ { h } ]$ with $| \eta _ { l } | , | \eta _ { h } | \le C$ , we have

$$
\begin{array} { r l } & { \underset { \eta \in [ \eta _ { l } , \eta _ { h } ] } { \operatorname* { s u p } } | \phi _ { n , T } ( \eta ) - \phi _ { \infty , T } ( \eta ) | \le \underset { \eta \in [ \eta _ { l } , \eta _ { h } ] } { \operatorname* { s u p } } \sum _ { k = 0 } ^ { 2 T } | c _ { n , k } - c _ { \infty , k } | C ^ { k } } \\ & { \le \underset { k \in \{ 0 , 1 , \ldots , 2 T \} } { \operatorname* { m a x } } | c _ { n , k } - c _ { \infty , k } | \displaystyle \sum _ { k = 0 } ^ { 2 T } C ^ { k } \xrightarrow [ ] { p } 0 . } \end{array}
$$

## C.4 Range and null-space geometry

Lemma C.2. When $n \geq d _ { i }$ , we have

$$
\ker ( H _ { n } ^ { \top } ) \ { \stackrel { \mathrm { a . s . } } { = } } \ \ker ( A _ { n } ) \ { \stackrel { \mathrm { a . s . } } { = } } \ \ker ( K ) .
$$

Proof. We first show that ker $( H _ { n } ^ { \top } ) \ { \stackrel { \mathrm { a . s . } } { = } } \ \ker ( A _ { n } )$ holds. If $H _ { n } ^ { \top } z = 0$ , then $A _ { n } z = 0$ follows immediately. Thus, we have ker $( H _ { n } ^ { \top } ) \subset \ker ( A _ { n } )$ . On the other hand, if $A _ { n } z \ = \ 0$ , then $z ^ { \top } A _ { n } z = \| V \| ^ { 2 } \| H _ { n } ^ { \top } z \| ^ { 2 } = 0$ . Since $\| V \| ^ { 2 } > 0$ with probability one, $H _ { n } ^ { \top } z = 0$ almost surely. Thus, with probability one, we have ker $( A _ { n } ) \subset \ker ( H _ { n } ^ { \top } )$

When $n \geq d .$ , the Gaussian matrix $W _ { 0 }$ has rank d almost surely. Thus,

$$
\operatorname { r a n g e } ( H _ { n } ) = \operatorname { r a n g e } ( X W _ { 0 } ^ { \top } ) \overset { \mathrm { a . s . } } { = } \operatorname { r a n g e } ( X ) = \operatorname { r a n g e } ( K ) .
$$

This implies ker $( H _ { n } ^ { \top } ) = \operatorname { r a n g e } ( H _ { n } ) ^ { \bot } \stackrel { \mathrm { a . s . } } { = } \operatorname { r a n g e } ( K ) ^ { \bot } = \ker ( K )$

## C.5 Optimization landscape

Proposition C.3 (Derivatives of the finite-horizon objective). For every $T \geq 1$ , we have

$$
\frac { \partial } { \partial \eta } \Phi _ { T } ( \eta ; \chi , A ) = - \frac { T } { m ^ { 2 } } F _ { T } ( \eta ; \chi , A ) ,\tag{9}
$$

$$
\frac { \partial ^ { 2 } } { \partial \eta ^ { 2 } } \Phi _ { T } ( \eta ; \chi , A ) = \frac { T ( 2 T - 1 ) } { m ^ { 3 } } \chi ^ { \top } A ^ { 2 } B _ { A } ( \eta ) ^ { 2 T - 2 } \chi \ge 0 ,\tag{10}
$$

$$
\frac { \partial } { \partial \eta } F _ { T } ( \eta ; \chi , A ) = - \frac { 2 T - 1 } { m } \chi ^ { \top } A ^ { 2 } B _ { A } ( \eta ) ^ { 2 T - 2 } \chi \le 0 .\tag{11}
$$

Proof. Since the matrices A and $B _ { A } ( \eta )$ commute, we have

$$
\frac { d } { d \eta } B _ { A } ( \eta ) ^ { k } = - \frac { k } { m } A B _ { A } ( \eta ) ^ { k - 1 } .
$$

Thus

$$
\frac { \partial } { \partial \eta } \Phi _ { T } ( \eta ; \chi , A ) = - \frac { 1 } { 2 m } \chi ^ { \top } \frac { 2 T } { m } A B _ { A } ( \eta ) ^ { 2 T - 1 } \chi = - \frac { T } { m ^ { 2 } } F _ { T } ( \eta ; \chi , A ) ,
$$

$$
\frac { \partial } { \partial \eta } F _ { T } ( \eta ; \chi , A ) = - \chi ^ { \top } A \frac { 2 T - 1 } { m } A B _ { A } ( \eta ) ^ { 2 T - 2 } \chi = - \frac { 2 T - 1 } { m } \chi ^ { \top } A ^ { 2 } B _ { A } ( \eta ) ^ { 2 T - 2 } \chi .
$$

Corollary C.4 (Existence and uniqueness of the optimal learning rate). Suppose $A \in \mathbb { R } _ { \mathrm { s y m } } ^ { m \times m }$ $\chi \in \mathbb R ^ { m }$ , and $A \succeq 0$ . If $A \chi \neq 0$ , then $\eta \mapsto \Phi _ { T } ( \eta ; \chi , A )$ is strictly convex and coercive on R, that $i s , \Phi _ { T } ( \eta ; \chi , A )  \infty$ as $| \eta | \to \infty$ . Hence it has a unique global minimizer $\eta _ { T } ( \chi , A ) > 0$ characterized by $F _ { T } ( \eta _ { T } ( \chi , A ) ; \chi , A ) = 0$ . On a closed interval $[ a , b ]$ , the optimizer is $\eta _ { T } ( \chi , A )$ if $\eta _ { T } ( \chi , A ) ~ \in ~ [ a , b ]$ , and is the appropriate boundary point otherwise. If $A \chi = 0 ,$ then $\Phi _ { T } ( \eta ; \chi , A )$ is constant in $\eta ,$ and every $\eta \in \mathbb { R }$ is a global minimizer.

Proof. If $A \chi = 0$ , then $B _ { A } ( \eta ) \chi = \chi$ holds for every $\eta \in \mathbb { R }$ . Hence $\Phi _ { T } ( \eta ; \chi , A )$ is independent of $\eta ,$ , which proves the degenerate case.

In the remainder of the proof, suppose $A \chi \neq 0$ . Let $A = U \operatorname { d i a g } ( \mu _ { 1 } , \dots , \mu _ { m } ) U ^ { \top }$ be the eigendecomposition of $A ,$ and let $q _ { j } = ( U _ { \cdot j } ) ^ { \top } \chi$ . Define the active index set

$$
\mathcal { I } = \{ j \in [ m ] : \mu _ { j } > 0 \mathrm { ~ a n d ~ } q _ { j } \neq 0 \} .
$$

Since $\begin{array} { r } { A \chi = \sum _ { i = 1 } ^ { m } \mu _ { j } q _ { j } U _ { \cdot j } } \end{array}$ , the assumption $A \chi \neq 0$ is equivalent to $\mathcal { I } \neq \emptyset$

Proposition C.3 gives

$$
\Phi _ { T } ( \eta ; \chi , A ) = \frac { 1 } { 2 m } \sum _ { j = 1 } ^ { m } q _ { j } ^ { 2 } \left( 1 - \frac { \eta \mu _ { j } } { m } \right) ^ { 2 T } = \sum _ { j = 1 } ^ { m } g _ { j } ( \eta ) ,
$$

where we defined

$$
g _ { j } ( \eta ) = \frac { q _ { j } ^ { 2 } } { 2 m } \left( 1 - \frac { \eta \mu _ { j } } { m } \right) ^ { 2 T } .
$$

Here, each $g _ { j }$ corresponding to $j \notin \mathcal { I }$ is either identically zero or constant in η. For each $j \in \mathcal I$ , we have

$$
g _ { j } ^ { \prime } ( \eta ) = - \frac { T q _ { j } ^ { 2 } \mu _ { j } } { m ^ { 2 } } \left( 1 - \frac { \eta \mu _ { j } } { m } \right) ^ { 2 T - 1 } .
$$

The map $\eta \mapsto g _ { j } ^ { \prime } ( \eta )$ is strictly increasing; thus, $g _ { j }$ is strictly convex on $\mathbb { R }$ . Since $\mathcal { I }$ is nonempty, $\Phi _ { T } ( \eta ; \lambda , A )$ is strictly convex. Moreover, for any fixed $j \in \mathcal { I }$ 2

$$
g _ { j } ( \eta )  \infty \qquad ( | \eta |  \infty ) .
$$

Since $\Phi _ { T } ( \eta ; \chi , A ) \ge g _ { j } ( \eta )$ , it follows that

$$
\Phi _ { T } ( \eta ; \chi , A )  \infty \qquad ( | \eta |  \infty ) .
$$

Thus $\Phi _ { T } ( \cdot ; \chi , A )$ is coercive. In particular, there exists $R > 0$ such that

$$
\Phi _ { T } ( \eta ; \chi , A ) > \Phi _ { T } ( 0 ; \chi , A ) \qquad ( | \eta | \geq R ) .
$$

Since $\Phi _ { T } ( \cdot ; \chi , A )$ is continuous, it attains its minimum on the compact interval $[ - R , R ]$ . This minimum is also global, because every point outside $[ - R , R ]$ has objective value larger than $\Phi _ { T } ( 0 ; \chi , A )$ . Finally, strict convexity implies that the global minimizer is unique. We denote it by $\eta _ { T } ( \chi , A )$ . The first-order optimality condition and (9) give

$$
F _ { T } ( \eta _ { T } ( \chi , A ) ; \chi , A ) = 0 .
$$

We now verify the positivity of $\eta _ { T } ( \chi , A )$ . At $\eta = 0$ , (9) yields

$$
\Phi _ { T } ^ { \prime } ( 0 ; \chi , A ) = - \frac { T } { m ^ { 2 } } \chi ^ { \top } A \chi = - \frac { T } { m ^ { 2 } } \| A ^ { 1 / 2 } \chi \| ^ { 2 } .
$$

Note that $A \chi \neq 0$ implies $A ^ { 1 / 2 } \chi \neq 0$ . Consequently, we have

$$
\Phi _ { T } ^ { \prime } ( 0 ; \chi , A ) < 0 = \Phi _ { T } ^ { \prime } ( \eta _ { T } ( \chi , A ) ; \chi , A ) .
$$

Since $\Phi _ { T } ( \cdot ; \chi , A )$ is strictly convex, its derivative is strictly increasing. Thus we have $\eta _ { T } ( \chi , A ) > 0$ . Finally, consider minimization over a closed interval $[ a , b ]$ with $a < b$ . By Proposition C.3, we have

$$
F _ { T } ( \eta ; \chi , A ) = \sum _ { j \in \mathcal { I } } q _ { j } ^ { 2 } \mu _ { j } \left( 1 - \frac { \eta \mu _ { j } } { m } \right) ^ { 2 T - 1 } .
$$

For each $j \in \mathcal I$ , the corresponding summand is strictly decreasing in η. Hence $F _ { T }$ is strictly decreasing on R. Since it is also continuous, the stated boundary characterization follows.

Proposition C.5 (Almost-sure nondegeneracy of the finite-width optimization problem). Suppose $K \neq 0$ and $n \geq d .$ . Then we have $A _ { n } \chi _ { n } ^ { ( 0 ) } \neq 0$ almost surely. By Corollary $C . 4 ,$ for every $T \geq 1 , \phi _ { n , T }$ is almost surely strictly convex and has a unique global minimizer.

Proof. Let P denote the orthogonal projector onto range(K). Define $\begin{array} { r } { \mathcal { E } _ { n } = \{ \ker ( H _ { n } ^ { \top } ) = } \end{array}$ ker $( A _ { n } ) = \ker ( K ) ]$ . Then by Lemma C.2, the event ${ \mathcal { E } } _ { n }$ has probability one. On the event $\mathcal { E } _ { n } .$ we have $f _ { n } ^ { ( 0 ) } ( X ) \in \mathrm { r a n g e } ( K )$ , and hence $P f _ { n } ^ { ( 0 ) } ( X ) = f _ { n } ^ { ( 0 ) } ( X )$ . On the same event, $A _ { n } \chi _ { n } ^ { ( 0 ) } = 0$ is equivalent to $\chi _ { n } ^ { ( 0 ) } \in \ker ( A _ { n } ) = \operatorname { r a n g e } ( K ) ^ { \perp }$ , and therefore to $P \chi _ { n } ^ { ( 0 ) } = 0$ . Recalling the definition $\chi _ { n } ^ { ( 0 ) } = f _ { n } ^ { ( 0 ) } ( X ) - \stackrel { . } { y }$ , this is further equivalent to $f _ { n } ^ { ( 0 ) } ( X ) = P f _ { n } ^ { ( 0 ) } ( X ) = P y$ . Noting that $\mathrm { P } ( { \mathcal { E } } _ { n } ) = 1$ , we have

$$
\mathrm { P } ( A _ { n } \chi _ { n } ^ { ( 0 ) } = 0 ) = \mathrm { P } ( \{ A _ { n } \chi _ { n } ^ { ( 0 ) } = 0 \} \cap \mathcal { E } _ { n } ) = \mathrm { P } ( \{ f _ { n } ^ { ( 0 ) } ( X ) = P y \} \cap \mathcal { E } _ { n } ) = \mathrm { P } ( f _ { n } ^ { ( 0 ) } ( X ) = P y ) .
$$

It therefore sufices to show that $\mathrm { P } ( f _ { n } ^ { ( 0 ) } ( X ) = P y ) = 0$

By (8) in the proof of Lemma 3.3, we have

$$
f _ { n } ^ { ( 0 ) } ( X ) \mid W _ { 0 } , V \sim { \mathcal { N } } \left( 0 , { \frac { 1 } { n } } A _ { n } \right) .
$$

Since $K \neq 0$ , we can choose a fixed vector $u \in \mathrm { r a n g e } ( K ) \setminus \{ 0 \}$ . By Lemma C.2, we have u $\notin$ ker $\left( A _ { n } \right)$ almost surely. Since $A _ { n } \succeq 0$ , it follows that $u ^ { \top } A _ { n } u > 0$ almost surely. Hence,

$$
u ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) \mid W _ { 0 } , V \sim { \mathcal { N } } \left( 0 , { \frac { 1 } { n } } u ^ { \top } A _ { n } u \right)
$$

is a nondegenerate Gaussian random variable almost surely. Therefore,

$$
\operatorname { P } ( u ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) = u ^ { \top } P y ~ | ~ W _ { 0 } , V ) ~ { \stackrel { \mathrm { a . s . } } { = } } ~ 0 .
$$

Since $\{ f _ { n } ^ { ( 0 ) } ( X ) = P y \} \subset \{ u ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) = u ^ { \top } P y \}$ , we obtain

$$
\operatorname { P } ( f _ { n } ^ { ( 0 ) } ( X ) = P y ~ { \big | } ~ W _ { 0 } , V ) ~ { \stackrel { \mathrm { a . s . } } { = } } ~ 0 .
$$

Taking expectations yields $\mathrm { P } ( f _ { n } ^ { ( 0 ) } ( X ) = P y ) = 0$ as desired.

## C.6 First-order sensitivities of the score and the loss

Set $B _ { T } = B _ { \infty } ( \eta _ { \infty , T } )$ , and define

$$
g _ { F , T } : = - 2 K B _ { T } ^ { 2 T - 1 } y ,\tag{12}
$$

$$
Q _ { F , T } : = \frac { 1 } { 2 } \left( B _ { T } ^ { 2 T - 1 } y y ^ { \top } + y y ^ { \top } B _ { T } ^ { 2 T - 1 } \right) - \frac { \eta _ { \infty , T } } { 2 m } \sum _ { s = 0 } ^ { 2 T - 2 } ( B _ { T } ^ { 2 T - 2 - s } y y ^ { \top } K B _ { T } ^ { s } + B _ { T } ^ { s } K y y ^ { \top } B _ { T } ^ { 2 T - 2 - s } ) ,\tag{13}
$$

$$
g _ { L , T } : = - \frac { 1 } { m } B _ { T } ^ { 2 T } y ,\tag{14}
$$

$$
Q _ { L , T } : = - \frac { \eta _ { \infty , T } } { 2 m ^ { 2 } } \sum _ { s = 0 } ^ { 2 T - 1 } B _ { T } ^ { 2 T - 1 - s } y y ^ { \top } B _ { T } ^ { s } .\tag{15}
$$

Lemma C.6 (First-order sensitivities at the infinite-width optimum). For fixed $\eta ,$ regard $F _ { T } ( \eta ; \chi , A )$ and $\Phi _ { T } ( \eta ; \chi , A )$ as real-valued maps on $\mathbb { R } ^ { m } \times \mathbb { R } _ { \mathrm { s y m } } ^ { m \times m }$ , equipped with the product norm $\| ( h , E ) \| _ { \times } = \left( \| h \| ^ { 2 } + \| E \| _ { \mathrm { F } } ^ { 2 } \right) ^ { 1 / 2 }$ . Then for $h \in \mathbb { R } ^ { m }$ and $E \in \mathbb { R } _ { \mathrm { s y m } } ^ { m \times m }$ , we have

$$
\begin{array} { r } { D _ { ( \chi , A ) } F _ { T } ( \eta _ { \infty , T } ; - y , K ) [ h , E ] = g _ { F , T } ^ { \top } h + \langle Q _ { F , T } , E \rangle _ { \mathrm { F } } , } \\ { D _ { ( \chi , A ) } \Phi _ { T } ( \eta _ { \infty , T } ; - y , K ) [ h , E ] = g _ { L , T } ^ { \top } h + \langle Q _ { L , T } , E \rangle _ { \mathrm { F } } . } \end{array}
$$

Proof. The derivative of $B _ { A } ( \eta )$ with respect to $A$ in the direction $E$ is

$$
D _ { A } B _ { A } ( \eta ) [ E ] = - \frac { \eta } { m } E .
$$

We also note that by Lemma F.1, we have

$$
D _ { A } B _ { A } ( \eta ) ^ { k } [ E ] = - \frac { \eta } { m } \sum _ { s = 0 } ^ { k - 1 } B _ { A } ( \eta ) ^ { s } E B _ { A } ( \eta ) ^ { k - 1 - s } .\tag{16}
$$

We first consider the loss $\Phi _ { T } ( \eta ; \chi , A )$ . Since $B _ { A } ( \eta )$ is symmetric for symmetric A, diferentiation with respect to $\chi$ gives

$$
D _ { \chi } \Phi _ { T } ( \eta _ { \infty , T } ; - y , K ) [ h ] = - \frac { 1 } { m } y ^ { \top } B _ { T } ^ { 2 T } h = g _ { L , T } ^ { \top } h .
$$

Using (16) with $k = 2 T$ , we have

$$
D _ { A } \Phi _ { T } ( \eta _ { \infty , T } ; - y , K ) [ E ] = - \frac { \eta _ { \infty , T } } { 2 m ^ { 2 } } \sum _ { s = 0 } ^ { 2 T - 1 } y ^ { \top } B _ { T } ^ { s } E B _ { T } ^ { 2 T - 1 - s } y = \langle Q _ { L , T } , E \rangle _ { \mathrm { F } } .
$$

Combining the two partial derivatives gives the Fr´echet derivative with respect to the pair $( \chi , A )$ . <sup>3</sup> Next, since A commutes with $B _ { A } ( \eta )$ , the matrix $A B _ { A } ( \eta ) ^ { 2 T - 1 }$ is symmetric. Hence we have

$$
D _ { \chi } F _ { T } ( \eta _ { \infty , T } ; - y , K ) [ h ] = - 2 y ^ { \top } K B _ { T } ^ { 2 T - 1 } h = g _ { F , T } ^ { \top } h .
$$

For the derivative with respect to A, (16) gives

$$
D _ { A } F _ { T } ( \eta _ { \infty , T } ; - y , K ) [ E ] = y ^ { \top } E B _ { T } ^ { 2 T - 1 } y - \frac { \eta _ { \infty , T } } { m } \sum _ { s = 0 } ^ { 2 T - 2 } y ^ { \top } K B _ { T } ^ { s } E B _ { T } ^ { 2 T - 2 - s } y = \langle Q _ { F , T } , E \rangle _ { \mathrm { F } } .
$$

Combining the results completes the proof.

$$
B _ { K + E } ( \eta _ { \infty , T } ) ^ { 2 T } = B _ { T } ^ { 2 T } + D _ { A } B _ { K } ( \eta _ { \infty , T } ) ^ { 2 T } [ E ] + O ( \Vert E \Vert _ { \mathrm { F } } ^ { 2 } )
$$

# D Proof of the fixed-horizon transfer theorem

Throughout this section, T is fixed. Define

$$
r _ { j , T } = 1 - \frac { \eta _ { \infty , T } \lambda _ { j } } { m } \qquad ( j \in [ r ] ) .\tag{17}
$$

Lemma D.1. Under the assumptions of Theorem ${ \it 3 . 6 , }$ we have for every fixed T that

$$
F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) < 0 , \qquad \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) > 0 .
$$

Proof. Recall that $c _ { j } = \| P _ { j } y \| ^ { 2 }$ . Proposition C.3 implies

$$
- F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) = \frac { 2 T - 1 } { m } y ^ { \top } K ^ { 2 } B _ { \infty } ( \eta _ { \infty , T } ) ^ { 2 T - 2 } y = \frac { 2 T - 1 } { m } \sum _ { j = 1 } ^ { r } c _ { j } \lambda _ { j } ^ { 2 } r _ { j , T } ^ { 2 T - 2 } .
$$

If $T = 1$ , since $K y \neq 0$ , we have $y ^ { \top } K ^ { 2 } B _ { \infty } ( \eta _ { \infty , T } ) ^ { 2 T - 2 } y > 0$ . If $T \geq 2$ , since $| \mathcal { T } _ { y } | \geq 2$ , at least one of $r _ { j , T }$ is nonzero. Thus the preceding sum is strictly positive. Hence in all cases, we have $F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) < 0$ , and consequently, $\begin{array} { r } { \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) = - \frac { T } { m ^ { 2 } } F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) > 0 } \end{array}$ □

Lemma D.2. Since $\eta _ { \infty , T } \in \left( \eta _ { \ell } , \eta _ { u } \right)$ , there exists a deterministic $\delta > 0$ such that

$$
\begin{array} { r } { [ \eta _ { \infty , T } - \delta , \eta _ { \infty , T } + \delta ] \subset ( \eta _ { \ell } , \eta _ { u } ) . } \end{array}
$$

For such $\delta _ { \pm }$ , we have

$$
\begin{array} { r l r } & { } & { \underset { | \eta - \eta _ { \infty , T } | \leq \delta } { \operatorname* { s u p } } | F _ { n , T } ( \eta ) - F _ { \infty , T } ( \eta ) | = O _ { p } ( n ^ { - 1 / 2 } ) , } \\ & { } & { \underset { | \eta - \eta _ { \infty , T } | \leq \delta } { \operatorname* { s u p } } | F _ { n , T } ^ { \prime } ( \eta ) - F _ { \infty , T } ^ { \prime } ( \eta ) | = O _ { p } ( n ^ { - 1 / 2 } ) . } \end{array}
$$

Proof. For fixed $T ,$ , both $F _ { T } ( \eta ; \chi , A )$ and $\partial _ { \eta } F _ { T } ( \eta ; \chi , A )$ are polynomials in the entries of $( \eta , \chi , A )$ . Hence, on the compact interval $[ \eta _ { \infty , T } - \delta , \eta _ { \infty , T } + \delta ]$ , their derivatives with respect to $( \chi , A )$ are uniformly bounded on a fixed neighborhood of $( - y , K )$ . Define the product norm $\| ( \chi , A ) \| _ { \times } = \left( \| \chi \| ^ { 2 } + \| A \| _ { \mathrm { F } } ^ { 2 } \right) ^ { 1 / 2 }$ . Then Lemma 3.3 gives $\| ( f _ { n } ^ { ( 0 ) } ( X ) , E _ { n } ) \| _ { \times } = O _ { p } ( n ^ { - 1 / 2 } )$ where we defined $E _ { n } = A _ { n } - K$ . Thus, noting that $( \chi _ { n } ^ { ( 0 ) } , A _ { n } ) \stackrel { p } { \longrightarrow } ( - y , K )$ holds, we have

$$
\begin{array} { r l } & { \underset { | \eta - \eta _ { \infty } , { T } | \leq \delta } { \operatorname* { s u p } } | F _ { n , T } ( \eta ) - F _ { \infty , T } ( \eta ) | } \\ & { = \underset { | \eta - \eta _ { \infty } , { T } | \leq \delta } { \operatorname* { s u p } } \left| \int _ { 0 } ^ { 1 } D _ { \chi , A } F _ { T } ( \eta ; - y + s f _ { n } ^ { ( 0 ) } ( X ) , K + s E _ { n } ) [ f _ { n } ^ { ( 0 ) } ( X ) , E _ { n } ] d s \right| } \\ & { \leq \underset { | \eta - \eta _ { \infty } , { T } | \leq \delta } { \operatorname* { s u p } } \int _ { 0 } ^ { 1 } \| D _ { \chi , A } F _ { T } ( \eta ; - y + s f _ { n } ^ { ( 0 ) } ( X ) , K + s E _ { n } ) \| _ { \mathrm { o p } } \| ( f _ { n } ^ { ( 0 ) } ( X ) , E _ { n } ) \| _ { \times } d s } \\ & { = O _ { p } ( n ^ { - 1 / 2 } ) , } \end{array}
$$

and, similarly,

$$
\operatorname* { s u p } _ { | \eta - \eta _ { \infty } , \tau | \leq \delta } | F _ { n , T } ^ { \prime } ( \eta ) - F _ { \infty , T } ^ { \prime } ( \eta ) | = O _ { p } ( n ^ { - 1 / 2 } ) .
$$

Lemma D.3. Under the assumptions of Theorem 3.6, $\eta _ { n , T }$ lies in $( \eta _ { \ell } , \eta _ { u } )$ with probability tending to one, and we have

$$
\eta _ { n , T } - \eta _ { \infty , T } = O _ { p } ( n ^ { - 1 / 2 } ) .
$$

Proof. By Lemmas D.1, D.2, and continuity of $F _ { \infty , T } ^ { \prime } ,$ , there exists a deterministic $\gamma > 0$ such that

$$
\begin{array} { r } { \operatorname* { s u p } _ { | \eta - \eta _ { \infty , T } | \leq \delta } F _ { \infty , T } ^ { \prime } ( \eta ) \leq - 2 \gamma . } \end{array}
$$

Hence we have

$$
\operatorname { P } ( \operatorname* { s u p } _ { | \eta - \eta _ { \infty } , { \cal T } | \leq \delta } F _ { n , { \cal T } } ^ { \prime } ( \eta ) \leq - \gamma )  1 .
$$

Since Corollary C.4 implies $F _ { \infty , T } ( \eta _ { \infty , T } ) = 0$ , Lemma D.2 yields

$$
F _ { n , T } ( \eta _ { \infty , T } ) = O _ { p } ( n ^ { - 1 / 2 } ) .
$$

Thus, for fixed $\epsilon > 0$ , there exists $M _ { 0 } < \infty$ such that

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } \mathrm { P } \left( | F _ { n , T } ( \eta _ { \infty , T } ) | \leq \frac { M _ { 0 } } { \sqrt { n } } \right) \geq 1 - \epsilon .
$$

Choose $M > M _ { 0 } / \gamma$ . For all suficiently large n, we have $M / \sqrt { n } < \delta$ . On the intersection of the two preceding high-probability events, the mean value theorem gives

$$
\begin{array} { l l } { \displaystyle { F _ { n , T } \left( \eta _ { \infty , T } - \frac { M } { \sqrt { n } } \right) \ge \frac { - M _ { 0 } + \gamma M } { \sqrt { n } } > 0 , } } \\ { \displaystyle { F _ { n , T } \left( \eta _ { \infty , T } + \frac { M } { \sqrt { n } } \right) \le \frac { M _ { 0 } - \gamma M } { \sqrt { n } } < 0 . } } \end{array}
$$

Proposition C.5 and Corollary C.4 imply that, for all suficiently large $n , \ \phi _ { n , T }$ has almost surely a unique global minimizer characterized by the root of $F _ { n , T }$ . Therefore, on the same event, this root belongs to

$$
\left( \eta _ { \infty , T } - \frac { M } { \sqrt { n } } , \eta _ { \infty , T } + \frac { M } { \sqrt { n } } \right) \subset ( \eta _ { \ell } , \eta _ { u } ) .
$$

Thus, with probability at least $1 - \epsilon + o ( 1 )$ , we have

$$
| \eta _ { n , T } - \eta _ { \infty , T } | < \frac { M } { \sqrt { n } } .
$$

Since $\epsilon > 0$ was arbitrary, we obtain

$$
\eta _ { n , T } - \eta _ { \infty , T } = O _ { p } ( n ^ { - 1 / 2 } ) ,
$$

and the finite-width optimizer is interior with probability tending to one.

Theorem D.4 (Limit distribution of the optimal learning rate at fixed T). Under the assumptions of Theorem 3.6, we have

$$
\begin{array} { r } { \sqrt { n } ( \eta _ { n , T } - \eta _ { \infty , T } ) \stackrel { d } { \longrightarrow } \Omega _ { \eta , T } : = - \Omega _ { F , T } / F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) , } \end{array}
$$

where $\Omega _ { F , T } = g _ { F , T } ^ { \top } Z + \langle Q _ { F , T } , \Xi \rangle _ { F }$

Proof. By Lemma C.6, applied at $( \eta _ { \infty , T } , - y , K )$ , we have

$$
F _ { n , T } ( \eta _ { \infty , T } ) = g _ { F , T } ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { F , T } , A _ { n } - K \rangle _ { \mathrm { F } } + o _ { p } ( n ^ { - 1 / 2 } ) .
$$

Here we used $F _ { \infty , T } ( \eta _ { \infty , T } ) = 0$ by Corollary C.4. Combining this with Lemma 3.3 gives

$$
\sqrt { n } F _ { n , T } ( \eta _ { \infty , T } ) \stackrel { d } { \longrightarrow } \Omega _ { F , T } .\tag{18}
$$

By the mean value theorem, there exists a random $\tilde { \eta } _ { n , T }$ between $\eta _ { n , T }$ and $\eta _ { \infty , T }$ such that

$$
F _ { n , T } ( \eta _ { n , T } ) - F _ { n , T } ( \eta _ { \infty , T } ) = F _ { n , T } ^ { \prime } ( { \tilde { \eta } } _ { n , T } ) ( \eta _ { n , T } - \eta _ { \infty , T } ) .
$$

On the event $\{ \eta _ { n , T } \in \left( \eta _ { \ell } , \eta _ { u } \right) \}$ , by Corollary C.4, we have $F _ { n , T } ( \eta _ { n , T } ) = 0$ . Moreover, on the same event, Lemma D.2 implies $F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) = F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) + O _ { p } ( n ^ { - 1 / 2 } )$ . Thus by Lemma D.3, we have

$$
F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) = F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) + o _ { p } ( 1 )
$$

unconditionally. By Lemma D.1, this further implies that $F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) < 0$ holds with probability tending to one. Thus, on the intersection of this event and $\{ \eta _ { n , T } \in \left( \eta _ { \ell } , \eta _ { u } \right) \}$ , both of which have probability tending to one, we have

$$
\sqrt { n } ( \eta _ { n , T } - \eta _ { \infty , T } ) = - \frac { \sqrt { n } F _ { n , T } ( \eta _ { \infty , T } ) } { F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) } .
$$

Finally, by (18), we obtain

$$
\sqrt { n } ( \eta _ { n , T } - \eta _ { \infty , T } ) \stackrel { d } { \longrightarrow } - \frac { \Omega _ { F , T } } { F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) } = \Omega _ { \eta , T } .
$$

Theorem D.5 (Limit distribution of the optimal loss at fixed T). Under the assumptions of Theorem 3.6, we have

$$
\sqrt { n } \left( \phi _ { n , T } ( \eta _ { n , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) \right) \stackrel { d } { \longrightarrow } \Omega _ { L , T } : = g _ { L , T } ^ { \top } Z + \langle Q _ { L , T } , \Xi \rangle _ { \mathrm { F } } .
$$

Proof. By Lemma C.6, we have

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = g _ { L , T } ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { L , T } , A _ { n } - K \rangle _ { \mathrm { F } } + o _ { p } ( n ^ { - 1 / 2 } ) .
$$

Hence Lemma 3.3 yields

$$
\sqrt { n } ( \phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) ) \stackrel { d } { \longrightarrow } \Omega _ { L , T } .\tag{19}
$$

It remains to replace $\phi _ { n , T } ( \eta _ { \infty , T } )$ by $\phi _ { n , T } ( \eta _ { n , T } )$ . By Taylor’s theorem, there exists a random $\bar { \eta } _ { n , T }$ between $\eta _ { n , T }$ and $\eta _ { \infty , T }$ such that

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { n , T } ( \eta _ { n , T } ) = \phi _ { n , T } ^ { \prime } ( \eta _ { n , T } ) ( \eta _ { \infty , T } - \eta _ { n , T } ) + \frac { 1 } { 2 } \phi _ { n , T } ^ { \prime \prime } ( \bar { \eta } _ { n , T } ) ( \eta _ { \infty , T } - \eta _ { n , T } ) ^ { 2 }
$$

On the event $\{ \eta _ { n , T } \in \left( \eta _ { \ell } , \eta _ { u } \right) \}$ , Proposition C.3 gives $\phi _ { n , T } ^ { \prime } ( \eta _ { n , T } ) \ : = \ : 0$ . Thus, on the same event, we have

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { n , T } ( \eta _ { n , T } ) = \frac { 1 } { 2 } \phi _ { n , T } ^ { \prime \prime } ( \bar { \eta } _ { n , T } ) ( \eta _ { \infty , T } - \eta _ { n , T } ) ^ { 2 } .
$$

Since $T$ is fixed, (10), Lemma 3.3, and Lemma D.3 imply $\phi _ { n , T } ^ { \prime \prime } ( \bar { \eta } _ { n , T } ) = O _ { p } ( 1 )$ . Therefore, combining this with Lemma D.3, we have

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { n , T } ( \eta _ { n , T } ) = O _ { p } ( n ^ { - 1 } ) = o _ { p } ( n ^ { - 1 / 2 } )
$$

with probability tending to one. Together with (19), this yields

$$
\sqrt { n } ( \phi _ { n , T } ( \eta _ { n , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) ) \stackrel { d } { \longrightarrow } \Omega _ { L , T } .
$$

Lemma D.6. Under the assumptions of Theorem ${ \it 3 . 6 , }$ we have

$$
c _ { n , T } = \frac { \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) } { 2 } ( \eta _ { n , T } - \eta _ { \infty , T } ) ^ { 2 } ( 1 + o _ { p } ( 1 ) ) .
$$

Proof. We next expand the transfer gap. Since $\eta _ { \infty , T }$ is the unique global minimizer, Proposition C.3 gives $\phi _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) = 0$ . By Taylor’s theorem, there exists a random $\eta _ { n , T } ^ { \dag }$ between $\eta _ { n , T }$ and $\eta _ { \infty , T }$ such that

$$
c _ { n , T } = \frac { \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) } { 2 } ( \eta _ { n , T } - \eta _ { \infty , T } ) ^ { 2 } + \frac { \phi _ { \infty , T } ^ { \prime \prime \prime } ( \eta _ { n , T } ^ { \dag } ) } { 6 } ( \eta _ { n , T } - \eta _ { \infty , T } ) ^ { 3 } .
$$

For fixed T, by Lemma D.3, with probability tending to one, we have

$$
| \phi _ { \infty , T } ^ { \prime \prime \prime } ( \eta _ { n , T } ^ { \dag } ) | \leq \operatorname* { s u p } _ { \eta _ { \ell } \leq \eta \leq \eta _ { u } } | \phi _ { \infty , T } ^ { \prime \prime \prime } ( \eta ) | .
$$

Therefore, by Lemmas D.1 and D.3, we have

$$
c _ { n , T } = \frac { \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) } { 2 } ( \eta _ { n , T } - \eta _ { \infty , T } ) ^ { 2 } ( 1 + o _ { p } ( 1 ) ) .
$$

Proof of Theorem 3.6. We verify that the limiting Gaussian variables $\Omega _ { L , T }$ and $\Omega _ { \eta , T }$ of $a _ { n , T }$ and $b _ { n , T }$ are nondegenerate. We first consider $\Omega _ { \eta , T }$ . For $s > 0$ , direct substitution into the definition of $F _ { T }$ gives

$$
F _ { T } ( \eta _ { \infty , T } ; - y , s K ) = s F _ { \infty , T } ( s \eta _ { \infty , T } ) .
$$

Diferentiating both sides with respect to s at $s ~ = ~ 1$ , applying Lemma C.6, and using $F _ { \infty , T } ( \eta _ { \infty , T } ) = 0$ , we have

$$
\langle Q _ { F , T } , K \rangle _ { \mathrm { F } } = F _ { \infty , T } ( \eta _ { \infty , T } ) + \eta _ { \infty , T } F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) = \eta _ { \infty , T } F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) .
$$

Recall from Lemma 3.3 that $\Xi = \alpha K + X G X ^ { \top }$ with $\alpha \sim \mathcal { N } ( 0 , 2 )$ , where α is independent of G and Z. Noting that Corollary C.4 gives $\eta _ { \infty , T } > 0$ , the contribution of the αK component of $\Xi$ to $\Omega _ { \eta , T }$ is

$$
- \frac { \alpha \langle Q _ { F , T } , K \rangle _ { \mathrm { F } } } { F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) } = - \eta _ { \infty , T } \alpha \sim \mathcal { N } ( 0 , 2 \eta _ { \infty , T } ^ { 2 } ) .
$$

Since this component is independent of the remaining Gaussian terms, $\Omega _ { \eta , T }$ is nondegenerate. By Theorem D.4 and the continuous mapping theorem, this proves

$$
b _ { n , T } = \Theta _ { p } ( n ^ { - 1 / 2 } ) .
$$

Next, we consider $\Omega _ { L , T }$ . By the definition of $g _ { L , T }$ , we have

$$
g _ { L , T } ^ { \top } K g _ { L , T } = \frac { 1 } { m ^ { 2 } } \sum _ { j = 1 } ^ { r } c _ { j } \lambda _ { j } r _ { j , T } ^ { 4 T } ,
$$

where $r _ { j , T }$ are defined in (17). Under $| \mathcal { T } _ { y } | \geq 2$ , at least one of the $r _ { j , T }$ is nonzero. Hence we have $g _ { L , T } ^ { \top } K g _ { L , T } > 0$ . Since $Z \sim { \mathcal { N } } ( 0 , K )$ is independent of Ξ by Lemma 3.3, we have

$$
\mathrm { V a r } ( \Omega _ { L , T } ) = \mathrm { V a r } ( g _ { L , T } ^ { \top } Z ) + \mathrm { V a r } ( \langle Q _ { L , T } , \Xi \rangle _ { \mathrm { F } } ) = g _ { L , T } ^ { \top } K g _ { L , T } + \mathrm { V a r } ( \langle Q _ { L , T } , \Xi \rangle _ { \mathrm { F } } ) > 0 .
$$

Therefore, $\Omega _ { L , T }$ is also nondegenerate. By Theorem D.5 and the continuous mapping theorem, this proves

$$
a _ { n , T } = \Theta _ { p } ( n ^ { - 1 / 2 } ) .
$$

Finally, by Lemmas D.6 and D.1, we obtain

$$
c _ { n , T } = \Theta _ { p } ( b _ { n , T } ^ { 2 } ) = \Theta _ { p } ( n ^ { - 1 } ) .
$$

Proof of Corollary 3.7. By Theorem D.5 and the continuous mapping theorem, we have

$$
\sqrt { n } a _ { n , T } \xrightarrow { d } | \Omega _ { L , T } | ,
$$

where $\Omega _ { L , T }$ is a nondegenerate Gaussian random variable. Since $\mathrm { P } ( | \Omega _ { L , T } | = 0 ) = 0$ , another application of the continuous mapping theorem gives

$$
\frac { 1 } { \sqrt { n } a _ { n , T } } \xrightarrow { d } \frac { 1 } { | \Omega _ { L , T } | } .
$$

Therefore, we have

$$
\frac { 1 } { a _ { n , T } } = O _ { p } ( \sqrt { n } ) .
$$

Combining this with $c _ { n , T } = \Theta _ { p } ( n ^ { - 1 } )$ in Theorem 3.6 yields the claim.

## E Infinite-width large-horizon analysis

## E.1 Proof of Theorem 3.1

Note that we have

$$
\begin{array} { c } { \displaystyle \phi _ { \infty , T } ( \eta ) = \frac { 1 } { 2 m } \left\| \left( I _ { m } - \frac { \eta } { m } K \right) ^ { T } y \right\| ^ { 2 } = \frac { 1 } { 2 m } \left\| \left[ P _ { 0 } + \sum _ { j = 1 } ^ { r } \left( 1 - \frac { \eta \lambda _ { j } } { m } \right) P _ { j } \right] ^ { T } y \right\| ^ { 2 } } \\ { \displaystyle = \frac { 1 } { 2 m } \left\| P _ { 0 } y + \sum _ { j = 1 } ^ { r } \left( 1 - \frac { \eta \lambda _ { j } } { m } \right) ^ { T } P _ { j } y \right\| ^ { 2 } = \frac { c _ { 0 } } { 2 m } + \frac { 1 } { 2 m } \sum _ { j = 1 } ^ { r } c _ { j } \left( 1 - \frac { \eta \lambda _ { j } } { m } \right) ^ { 2 T } . } \end{array}
$$

Here, $c _ { 0 } / ( 2 m )$ is an irreducible training loss. No choice of learning rate or number of steps changes it. Define

$$
R _ { T } ( \eta ) : = ( 2 m \phi _ { \infty , T } ( \eta ) - c _ { 0 } ) ^ { 1 / ( 2 T ) } = \left( \sum _ { j = 1 } ^ { r } c _ { j } \left| 1 - \frac { \eta \lambda _ { j } } { m } \right| ^ { 2 T } \right) ^ { 1 / ( 2 T ) } .
$$

Then we have

$$
\arg \operatorname* { m i n } _ { \eta \ge 0 } \phi _ { \infty , T } ( \eta ) = \arg \operatorname* { m i n } _ { \eta \ge 0 } R _ { T } ( \eta ) .
$$

Define

$$
R _ { \infty } ( \eta ) : = \operatorname* { m a x } _ { j \in \mathcal { I } _ { y } } \left. 1 - \frac { \eta \lambda _ { j } } { m } \right. = \left. 1 - \frac { \eta \lambda _ { - } ^ { y } } { m } \right. \vee \left. 1 - \frac { \eta \lambda _ { + } ^ { y } } { m } \right. .
$$

Choose an index $j _ { * } \in \mathcal { I } _ { y }$ such that $| 1 - \eta \lambda _ { j _ { \ast } } / m | = R _ { \infty } ( \eta )$ . Then we have

$$
c _ { j _ { * } } R _ { \infty } ( \eta ) ^ { 2 T } \leq \sum _ { j \in \mathcal { I } _ { y } } c _ { j } \left| 1 - \frac { \eta \lambda _ { j } } m \right| ^ { 2 T } \leq \left( \sum _ { j = 1 } ^ { r } c _ { j } \right) R _ { \infty } ( \eta ) ^ { 2 T } ,
$$

which implies

$$
\operatorname* { m i n } _ { j \in \mathcal { I } _ { y } } c _ { j } ^ { 1 / ( 2 T ) } R _ { \infty } ( \eta ) \le R _ { T } ( \eta ) \le \left( \sum _ { j = 1 } ^ { r } c _ { j } \right) ^ { 1 / ( 2 T ) } R _ { \infty } ( \eta ) .
$$

Since $\mathcal { T } _ { y }$ is finite and all active coeficients are strictly positive, we have

$$
\operatorname* { m i n } _ { j \in \mathcal { I } _ { y } } c _ { j } ^ { 1 / ( 2 T ) } \to 1 , \qquad \left( \sum _ { j = 1 } ^ { r } c _ { j } \right) ^ { 1 / ( 2 T ) } \to 1 .
$$

Thus, on every compact interval I, we have

$$
\operatorname* { s u p } _ { \eta \in \mathcal { T } } | R _ { T } ( \eta ) - R _ { \infty } ( \eta ) | \leq ( | \operatorname* { m i n } _ { j \in \mathcal { T } _ { y } } c _ { j } ^ { 1 / ( 2 T ) } - 1 | \vee | ( \sum _ { j = 1 } ^ { r } c _ { j } ) ^ { 1 / ( 2 T ) } - 1 | ) \operatorname* { s u p } _ { \eta \in \mathcal { T } } | R _ { \infty } ( \eta ) |  0 .
$$

It remains to verify that the global minimizers of $R _ { T }$ are contained in a fixed compact interval. Fix any $j _ { 0 } \in \mathcal { I } _ { y }$ . Since $\eta _ { \infty , T }$ is a global minimizer over $[ 0 , \infty )$ , we have

$$
R _ { T } ( \eta _ { \infty , T } ) \leq R _ { T } ( 0 ) = \left( \sum _ { j \in \mathcal { I } _ { y } } c _ { j } \right) ^ { 1 / ( 2 T ) } .
$$

On the other hand, for any $\eta \geq 0$ , we have

$$
R _ { T } ( \eta ) \geq c _ { j _ { 0 } } ^ { 1 / ( 2 T ) } \left| 1 - \frac { \eta \lambda _ { j _ { 0 } } } { m } \right| .
$$

Therefore we have

$$
\left| 1 - \frac { \eta _ { \infty , T } \lambda _ { j _ { 0 } } } { m } \right| \le c _ { j _ { 0 } } ^ { - 1 / ( 2 T ) } R _ { T } ( \eta _ { \infty , T } ) \le \left( \frac { \sum _ { j \in \mathcal { I } _ { y } } c _ { j } } { c _ { j _ { 0 } } } \right) ^ { 1 / ( 2 T ) } \le \left( \frac { \sum _ { j \in \mathcal { I } _ { y } } c _ { j } } { c _ { j _ { 0 } } } \right) ^ { 1 / 2 } .
$$

Hence there exists a deterministic constant $M < \infty$ , independent of $T _ { i }$ such that $\eta _ { \infty , T } \in$ $[ 0 , M ]$ for every $T \geq 1$ . Enlarging M if necessary, we may also assume $\eta _ { * } ^ { y } \in [ 0 , M ]$ . Since $\begin{array} { r } { \operatorname* { s u p } _ { \eta \in [ 0 , M ] } | R _ { T } ( \eta ) - R _ { \infty } ( \eta ) | \to 0 } \end{array}$ holds and $R _ { \infty }$ has the unique minimizer $\eta _ { * } ^ { y }$ , we have

$$
\eta _ { \infty , T }  \eta _ { * } ^ { y } .
$$

If $\lambda _ { + } ^ { y } = \lambda _ { - } ^ { y } = \lambda$ , the loss is a positive multiple of $( 1 - \eta \lambda / m ) ^ { 2 T }$ plus a constant, whose unique minimizer is $m / \lambda$ □

## E.2 Expansion of the infinite-width optimizer and asymptotic loss

Proposition E.1 (First-order large-T expansion of the optimal learning rate). Suppose Assumption 3.2 holds and define

$$
\bar { \vartheta } : = \frac { 1 } { q _ { * } } \operatorname* { m a x } _ { 2 \leq j \leq r - 1 } \left| 1 - \frac { \eta _ { * } \lambda _ { j } } { m } \right| ,
$$

where max $\emptyset : = 0$ . Then $0 \leq \bar { \vartheta } < 1$ . Fix any constant ϑ satisfying $\bar { \vartheta } < \vartheta < 1$ . Then, as $T \to \infty$ , we have

$$
\eta _ { \infty , T } = \eta _ { * } - \frac { m q _ { * } } { \left( \lambda _ { 1 } + \lambda _ { r } \right) \left( 2 T - 1 \right) } L _ { * } + O ( T ^ { - 2 } ) + O \left( \frac { \vartheta ^ { 2 T } } { T } \right) .
$$

Proof. Near $\eta _ { * }$ , define the positive endpoint magnitudes

$$
r _ { 1 } ( \eta ) = \frac { \eta \lambda _ { 1 } } { m } - 1 , \qquad r _ { r } ( \eta ) = 1 - \frac { \eta \lambda _ { r } } { m } .
$$

Then we have $r _ { 1 } ( \eta _ { * } ) = r _ { r } ( \eta _ { * } ) = q _ { * } > 0$ . By the definition of ${ \bar { \vartheta } } .$ we have

$$
\operatorname* { m a x } _ { 2 \leq j \leq r - 1 } \frac { | 1 - \eta _ { * } \lambda _ { j } / m | } { r _ { 1 } ( \eta _ { * } ) } = \bar { \vartheta } < \vartheta .
$$

Since Theorem 3.1 gives $\eta _ { \infty , T } \to \eta _ { * }$ , for all suficiently large $T _ { \ast }$ , we have

$$
\operatorname* { m a x } _ { 2 \leq j \leq r - 1 } \frac { | 1 - \eta _ { \infty , T } \lambda _ { j } / m | } { r _ { 1 } ( \eta _ { \infty , T } ) } \leq \vartheta .\tag{20}
$$

By Corollary C.4, the score equation is given by

$$
\begin{array} { l } { { 0 = F _ { \infty , T } ( \eta _ { \infty , T } ) = y ^ { \top } K B _ { \infty } ( \eta _ { \infty , T } ) ^ { 2 T - 1 } y = y ^ { \top } K \left( I _ { m } - \frac { \eta _ { \infty , T } } { m } K \right) ^ { 2 T - 1 } y } } \\ { { = \displaystyle \sum _ { j = 1 } ^ { r } c _ { j } \lambda _ { j } \left( 1 - \frac { \lambda _ { j } \eta _ { \infty , T } } { m } \right) ^ { 2 T - 1 } } } \\ { { = - c _ { 1 } \lambda _ { 1 } ( r _ { 1 } ( \eta _ { \infty , T } ) ) ^ { 2 T - 1 } + c _ { r } \lambda _ { r } r _ { r } ( \eta _ { \infty , T } ) ^ { 2 T - 1 } + \displaystyle \sum _ { j = 2 } ^ { r - 1 } c _ { j } \lambda _ { j } \left( 1 - \frac { \lambda _ { j } \eta _ { \infty , T } } { m } \right) ^ { 2 T - 1 } . } } \end{array}
$$

Hence we have

$$
\vert c _ { r } \lambda _ { r } r _ { r } ( \eta _ { \infty , T } ) ^ { 2 T - 1 } - c _ { 1 } \lambda _ { 1 } r _ { 1 } ( \eta _ { \infty , T } ) ^ { 2 T - 1 } \vert \leq \sum _ { j = 2 } ^ { r - 1 } c _ { j } \lambda _ { j } \left. 1 - \frac { \lambda _ { j } \eta _ { \infty , T } } { m } \right. ^ { 2 T - 1 } .
$$

Using (20), we obtain

$$
\left| c _ { r } \lambda _ { r } \left( \frac { r _ { r } ( \eta _ { \infty , T } ) } { r _ { 1 } ( \eta _ { \infty , T } ) } \right) ^ { 2 T - 1 } - c _ { 1 } \lambda _ { 1 } \right| \leq \left( \sum _ { j = 2 } ^ { r - 1 } c _ { j } \lambda _ { j } \right) \vartheta ^ { 2 T - 1 } = O ( \vartheta ^ { 2 T - 1 } ) .
$$

Therefore, we have

$$
\left( \frac { r _ { r } ( \eta _ { \infty , T } ) } { r _ { 1 } ( \eta _ { \infty , T } ) } \right) ^ { 2 T - 1 } = \frac { c _ { 1 } \lambda _ { 1 } } { c _ { r } \lambda _ { r } } \left( 1 + { \cal O } ( \vartheta ^ { 2 T - 1 } ) \right) .
$$

Taking logarithms gives

$$
( 2 T - 1 ) \left( \log r _ { r } ( \eta _ { \infty , T } ) - \log r _ { 1 } ( \eta _ { \infty , T } ) \right) = L _ { * } + O ( \vartheta ^ { 2 T } ) .
$$

Writing $\eta _ { \infty , T } = \eta _ { * } + \delta$ , we have

$$
r _ { 1 } ( \eta _ { \infty , T } ) = q _ { * } + \frac { \lambda _ { 1 } \delta } { m } , \qquad r _ { r } ( \eta _ { \infty , T } ) = q _ { * } - \frac { \lambda _ { r } \delta } { m } .\tag{21}
$$

Thus, using $\log ( 1 + x ) = x + O ( x ^ { 2 } )$ , we obtain

$$
\log r _ { r } ( \eta _ { \infty , T } ) - \log r _ { 1 } ( \eta _ { \infty , T } ) = \log \left( \frac { 1 - \frac { \lambda _ { r } \delta } { m q _ { * } } } { 1 + \frac { \lambda _ { 1 } \delta } { m q _ { * } } } \right) = - \frac { \lambda _ { r } \delta } { m q _ { * } } - \frac { \lambda _ { 1 } \delta } { m q _ { * } } + O ( \delta ^ { 2 } ) = - \frac { \lambda _ { 1 } + \lambda _ { r } } { m q _ { * } } \delta + O ( \delta ^ { 2 } ) .
$$

Combining the results, we have

$$
L _ { * } + O ( \vartheta ^ { 2 T } ) = ( 2 T - 1 ) \left( - \frac { \lambda _ { 1 } + \lambda _ { r } } { m q _ { * } } \delta + O ( \delta ^ { 2 } ) \right) .
$$

Since the left-hand side is $O ( 1 )$ , we have $\delta = O ( T ^ { - 1 } )$ . Thus

$$
L _ { * } + O ( \vartheta ^ { 2 T } ) = - ( 2 T - 1 ) \frac { \lambda _ { 1 } + \lambda _ { r } } { m q _ { * } } \delta + O ( T ^ { - 1 } ) .
$$

Solving for $\delta ,$ we have

$$
\eta _ { \infty , T } - \eta _ { * } = \delta = - \frac { L _ { * } m q _ { * } } { ( \lambda _ { 1 } + \lambda _ { r } ) ( 2 T - 1 ) } + O ( T ^ { - 2 } ) + O \left( \frac { \vartheta ^ { 2 T } } { T } \right) .
$$

Corollary E.2 (Loss and curvature at the large-horizon optimum). Under the assumptions of Proposition $E . 1 ,$ we have

$$
\phi _ { \infty , T } ( \eta _ { \infty , T } ) - \frac { c _ { 0 } } { 2 m } = \Theta \left( q _ { * } ^ { 2 T } \right) ,
$$

$$
\left| F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) \right| = \Theta \left( T q _ { * } ^ { 2 T - 2 } \right) ,\tag{22}
$$

$$
\phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) = \Theta \left( T ^ { 2 } q _ { * } ^ { 2 T - 2 } \right) .\tag{23}
$$

Proof. Define $r _ { 1 } ( \eta )$ and $r _ { r } ( \eta )$ as in the proof of Proposition E.1. By (21) and Proposition E.1, we have

$$
r _ { r } ( \eta _ { \infty , T } ) = q _ { * } \left[ 1 + \frac { \lambda _ { r } } { \lambda _ { 1 } + \lambda _ { r } } \frac { L _ { * } } { 2 T - 1 } + O ( T ^ { - 2 } ) + O \left( \frac { \vartheta ^ { 2 T } } { T } \right) \right] ,
$$

$$
r _ { 1 } ( \eta _ { \infty , T } ) = q _ { * } \left[ 1 - \frac { \lambda _ { 1 } } { \lambda _ { 1 } + \lambda _ { r } } \frac { L _ { * } } { 2 T - 1 } + { \cal O } ( T ^ { - 2 } ) + { \cal O } \left( \frac { \vartheta ^ { 2 T } } { T } \right) \right] .
$$

In particular, both endpoint magnitudes are $q _ { * } ( 1 + O ( T ^ { - 1 } ) )$ . More precisely, for every fixed integer $\ell ,$ the expansion

$$
\left( 1 + { \frac { a } { 2 T - 1 } } + O ( T ^ { - 2 } ) \right) ^ { 2 T - \ell } = \exp ( a ) \left( 1 + O ( T ^ { - 1 } ) \right)
$$

gives

$$
r _ { r } ( \eta _ { \infty , T } ) ^ { 2 T - \ell } = q _ { * } ^ { 2 T - \ell } a _ { r } ^ { * } \left( 1 + O ( T ^ { - 1 } ) + O ( \vartheta ^ { 2 T } ) \right) ,\tag{24}
$$

$$
r _ { 1 } ( \eta _ { \infty , T } ) ^ { 2 T - \ell } = q _ { * } ^ { 2 T - \ell } a _ { 1 } ^ { * } \left( 1 + O ( T ^ { - 1 } ) + O ( \vartheta ^ { 2 T } ) \right) .\tag{25}
$$

We now control the remaining active modes. Since $\eta _ { \infty , T } \to \eta _ { \ L }$ <sub>∗</sub> by Theorem 3.1, we have

$$
\frac { 1 } { q _ { * } } \operatorname* { m a x } _ { 2 \le j \le r - 1 } \left| 1 - \frac { \eta _ { \infty , T } \lambda _ { j } } { m } \right| \to \bar { \vartheta } < 1 ,
$$

which implies, for suficiently large $T _ { \ast }$

$$
\operatorname* { m a x } _ { 2 \leq j \leq r - 1 } \left. 1 - \frac { \eta _ { \infty , T } \lambda _ { j } } { m } \right. \leq \vartheta q _ { * } .
$$

Hence we have

$$
\sum _ { j = 2 } ^ { r - 1 } c _ { j } \lambda _ { j } ^ { 2 } \left. 1 - \frac { \eta _ { \infty , T } \lambda _ { j } } { m } \right. ^ { 2 T - 2 } = O \left( ( \vartheta q _ { * } ) ^ { 2 T - 2 } \right) = o \left( q _ { * } ^ { 2 T - 2 } \right) .\tag{26}
$$

Using the spectral representation of the loss, we therefore have

$$
\phi _ { \infty , T } ( \eta _ { \infty , T } ) - \frac { c _ { 0 } } { 2 m } = \frac { 1 } { 2 m } \sum _ { j = 1 } ^ { r } c _ { j } \left( 1 - \frac { \eta _ { \infty , T } \lambda _ { j } } { m } \right) ^ { 2 T } = \frac { q _ { * } ^ { 2 T } } { 2 m } \left[ c _ { r } a _ { r } ^ { * } + c _ { 1 } a _ { 1 } ^ { * } + o ( 1 ) \right] = \Theta ( q _ { * } ^ { 2 T } ) ,
$$

where the last equality follows since $c _ { r } a _ { r } ^ { * } { + } c _ { 1 } a _ { 1 } ^ { * } { + } o ( 1 )$ converges to a strictly positive constant.

Next, recall that by Proposition C.3,

$$
F _ { \infty , T } ^ { \prime } ( \eta ) = - \frac { 2 T - 1 } { m } \sum _ { j = 1 } ^ { r } c _ { j } \lambda _ { j } ^ { 2 } \left( 1 - \frac { \eta \lambda _ { j } } { m } \right) ^ { 2 T - 2 } .
$$

Note that all summands in the preceding display are nonnegative before applying the leading minus sign. Applying (24), (25), and (26) gives

$$
F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) = - \frac { 2 T - 1 } { m } q _ { * } ^ { 2 T - 2 } \left[ c _ { r } \lambda _ { r } ^ { 2 } a _ { r } ^ { * } + c _ { 1 } \lambda _ { 1 } ^ { 2 } a _ { 1 } ^ { * } + o ( 1 ) \right] = - \Theta \left( T q _ { * } ^ { 2 T - 2 } \right) .
$$

The last equality follows since the limiting coeficient in square brackets is again strictly positive.

Finally, noting that

$$
\phi _ { \infty , T } ^ { \prime } ( \eta ) = - \frac { T } { m ^ { 2 } } F _ { \infty , T } ( \eta )
$$

holds by Proposition C.3, the curvature at the optimizer satisfies

$$
\phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) = - \frac { T } { m ^ { 2 } } F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) .
$$

Substituting the previous result yields

$$
\phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) = \frac { T } { m ^ { 2 } } \Theta \left( T q _ { * } ^ { 2 T - 2 } \right) = \Theta ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } ) .
$$

## E.3 Endpoint contractions and spectral separation

Lemma E.3 (Endpoint contraction asymptotics). For $r _ { j , T } , ~ ( j ~ \in ~ [ r ] )$ defined in (17), we have

$$
\frac { r _ { 1 , T } ^ { 2 T - 1 } } { q _ { * } ^ { 2 T - 1 } } \to - a _ { 1 } ^ { * } , \qquad \frac { r _ { r , T } ^ { 2 T - 1 } } { q _ { * } ^ { 2 T - 1 } } \to a _ { r } ^ { * } , \qquad \frac { r _ { 1 , T } ^ { 2 T - 2 } } { q _ { * } ^ { 2 T - 2 } } \to a _ { 1 } ^ { * } , \qquad \frac { r _ { r , T } ^ { 2 T - 2 } } { q _ { * } ^ { 2 T - 2 } } \to a _ { r } ^ { * } .
$$

Moreover, if $r > 2$ , there exists a constant $\vartheta _ { 0 } \in ( 0 , 1 )$ such that

$$
| r _ { j , T } | \leq \vartheta _ { 0 } q _ { * } \mathrm { ~ ( 2 } \leq j \leq r - 1 )
$$

for all suficiently large $T .$

Proof. Proposition E.1 gives

$$
\eta _ { \infty , T } - \eta _ { * } = - \frac { m q _ { * } L _ { * } } { ( \lambda _ { 1 } + \lambda _ { r } ) ( 2 T - 1 ) } + o ( T ^ { - 1 } ) .
$$

Since $1 - \eta _ { * } \lambda _ { 1 } / m = - q _ { * }$ and $1 - \eta _ { * } \lambda _ { r } / m = q _ { * }$ , we have

$$
\begin{array} { l } { { r _ { 1 , T } = - q _ { * } \left( 1 - \frac { \lambda _ { 1 } L _ { * } } { \left( \lambda _ { 1 } + \lambda _ { r } \right) ( 2 T - 1 ) } + o ( ( 2 T - 1 ) ^ { - 1 } ) \right) , } } \\ { { r _ { r , T } = q _ { * } \left( 1 + \frac { \lambda _ { r } L _ { * } } { \left( \lambda _ { 1 } + \lambda _ { r } \right) ( 2 T - 1 ) } + o ( ( 2 T - 1 ) ^ { - 1 } ) \right) . } } \end{array}
$$

Thus, noting that $2 T - 1$ is odd, we obtain

$$
\frac { r _ { 1 , T } ^ { 2 T - 1 } } { q _ { * } ^ { 2 T - 1 } } \to - a _ { 1 } ^ { * } , \qquad \frac { r _ { r , T } ^ { 2 T - 1 } } { q _ { * } ^ { 2 T - 1 } } \to a _ { r } ^ { * } , \qquad \frac { r _ { 1 , T } ^ { 2 T - 2 } } { q _ { * } ^ { 2 T - 2 } } \to a _ { 1 } ^ { * } , \qquad \frac { r _ { r , T } ^ { 2 T - 2 } } { q _ { * } ^ { 2 T - 2 } } \to a _ { r } ^ { * } .
$$

For an interior positive eigenvalue $\lambda _ { j } \in \left( \lambda _ { r } , \lambda _ { 1 } \right)$ , the definition of $\eta _ { * }$ gives

$$
- q _ { * } < 1 - \frac { \eta _ { * } \lambda _ { j } } { m } < q _ { * } .
$$

By Proposition E.1, we have

$$
\operatorname* { m a x } _ { 2 \leq j \leq r - 1 } \frac { | 1 - \eta _ { * } \lambda _ { j } / m | } { q _ { * } } = \bar { \vartheta } < 1 .
$$

Since Theorem 3.1 gives $\eta _ { \infty , T } \to \eta _ { * }$ , there exists $\vartheta _ { 0 } \in ( \bar { \vartheta } , 1 )$ such that, for all suficiently large $T ,$

$$
\operatorname* { m a x } _ { 2 \leq j \leq r - 1 } \frac { \vert r _ { j , T } \vert } { q _ { * } } = \operatorname* { m a x } _ { 2 \leq j \leq r - 1 } \frac { \vert 1 - \eta _ { \infty , T } \lambda _ { j } / m \vert } { q _ { * } } \leq \vartheta _ { 0 } .
$$

## F Matrix-power perturbation bounds

## F.1 Fr´echet expansion of matrix powers

Lemma F.1 (First-order expansion of matrix powers). Let $k \geq 1$ . Equip $\mathbb { R } ^ { p \times p }$ with any submultiplicative matrix norm ∥ · ∥, and define $\Psi _ { k } ( B ) = B ^ { k }$ as a map from this normed space into itself. Then $\Psi _ { k }$ is Fr´echet diferentiable at every $B _ { i }$ , and for every $C \in \mathbb { R } ^ { p \times p }$ , we have

$$
D \Psi _ { k } ( B ) [ C ] = \sum _ { s = 0 } ^ { k - 1 } B ^ { s } C B ^ { k - 1 - s } .
$$

Consequently, for every fixed C, we have

$$
( B + \epsilon C ) ^ { k } = B ^ { k } + \epsilon \sum _ { s = 0 } ^ { k - 1 } B ^ { s } C B ^ { k - 1 - s } + O ( \epsilon ^ { 2 } )
$$

as $\epsilon \to 0$ . No commutativity between B and C is required.

Proof. We first recall that for arbitrary square matrices X and $Y$ of the same size and every $k \geq 1$ , we have

$$
X ^ { k } - Y ^ { k } = \sum _ { s = 0 } ^ { k - 1 } X ^ { s } ( X - Y ) Y ^ { k - 1 - s } .
$$

Applying this identity with $X = B + E$ and $Y = B$ , we obtain

$$
\Psi _ { k } ( B + E ) - \Psi _ { k } ( B ) = ( B + E ) ^ { k } - B ^ { k } = \sum _ { s = 0 } ^ { k - 1 } ( B + E ) ^ { s } E B ^ { k - 1 - s } .
$$

Thus if we define

$$
R _ { k } ( B , E ) : = \sum _ { s = 0 } ^ { k - 1 } ( ( B + E ) ^ { s } - B ^ { s } ) E B ^ { k - 1 - s } ,
$$

we can write

$$
\Psi _ { k } ( B + E ) - \Psi _ { k } ( B ) = \sum _ { s = 0 } ^ { k - 1 } B ^ { s } E B ^ { k - 1 - s } + R _ { k } ( B , E ) .
$$

In $R _ { k } ( B , E )$ , the summand corresponding to $s = 0$ vanishes. For $s \geq 1$ , applying the same identity once more gives

$$
( B + E ) ^ { s } - B ^ { s } = \sum _ { j = 0 } ^ { s - 1 } ( B + E ) ^ { j } E B ^ { s - 1 - j } .
$$

Therefore, we have

$$
R _ { k } ( B , E ) = \sum _ { s = 1 } ^ { k - 1 } \sum _ { j = 0 } ^ { s - 1 } ( B + E ) ^ { j } E B ^ { s - 1 - j } E B ^ { k - 1 - s } .
$$

Let $\| \cdot \|$ be any submultiplicative matrix norm. Then for each $s \in \{ 1 , \ldots , k - 1 \}$ and $j \in \{ 0 , \ldots , s - 1 \}$ , we have

$$
\begin{array} { r l } & { \| ( B + E ) ^ { j } E B ^ { s - 1 - j } E B ^ { k - 1 - s } \| \leq \| B + E \| ^ { j } \| E \| \| B \| ^ { s - 1 - j } \| E \| \| B \| ^ { k - 1 - s } } \\ & { \qquad \leq ( \| B \| + \| E \| ) ^ { k - 2 } \| E \| ^ { 2 } . } \end{array}
$$

Since there are $k ( k - 1 ) / 2$ terms in the double sum, we have

$$
\| R _ { k } ( B , E ) \| \leq \frac { k ( k - 1 ) } { 2 } ( \| B \| + \| E \| ) ^ { k - 2 } \| E \| ^ { 2 } .
$$

Thus, we have

$$
\begin{array} { r l } & { \frac { \| \Psi _ { k } ( B + E ) - \Psi _ { k } ( B ) - \sum _ { s = 0 } ^ { k - 1 } B ^ { s } E B ^ { k - 1 - s } \| } { \| E \| } = \frac { \| R _ { k } ( B , E ) \| } { \| E \| } } \\ & { \leq \frac { k ( k - 1 ) } { 2 } ( \| B \| + \| E \| ) ^ { k - 2 } \| E \| \to 0 } \end{array}
$$

as $\| E \|  0$ . Since the map $\begin{array} { r } { E \mapsto \sum _ { s = 0 } ^ { k - 1 } B ^ { s } E B ^ { k - 1 - s } } \end{array}$ is linear and bounded, it is the Fr´echet derivative of $\Psi _ { k }$ at B. Therefore, we have

$$
D \Psi _ { k } ( B ) [ C ] = \sum _ { s = 0 } ^ { k - 1 } B ^ { s } C B ^ { k - 1 - s } .
$$

Finally, taking $E = \epsilon C$ for a fixed matrix C gives

$$
\begin{array} { l } { { ( B + \epsilon C ) ^ { k } = \Psi _ { k } ( B + \epsilon C ) = \Psi _ { k } ( B ) + D \Psi _ { k } ( B ) [ \epsilon C ] + R _ { k } ( B , \epsilon C ) } } \\ { { \displaystyle \qquad = B ^ { k } + \epsilon \sum _ { s = 0 } ^ { k - 1 } B ^ { s } C B ^ { k - 1 - s } + O ( \epsilon ^ { 2 } ) , } } \end{array}
$$

which proves the final assertion.

## F.2 Uniform noncommutative power bounds

Lemma F.2 (Uniform local power perturbation bounds). Let $K \succeq 0$ be a finite-dimensional symmetric positive semidefinite matrix and let P project onto range(K). Suppose there exists a deterministic sequence $q _ { T } > 0$ , bounded away from zero, such that

$$
\operatorname* { s u p } _ { \eta \in \mathcal { T } _ { T } } \| P B _ { K } ( \eta ) P \| _ { \mathrm { o p } } \leq q _ { T } , \qquad q _ { 0 } : = \operatorname* { i n f } _ { T } q _ { T } > 0 .
$$

Let $A = K + E \succeq 0$ be symmetric and have the same kernel as K. Then we have

$$
P A = A P , \qquad P K = K P , \qquad P E = E P .
$$

Suppose $T \| E \| _ { \mathrm { o p } } \leq c$ for some fixed c. Suppose $\mathrm { s u p } _ { T } \mathrm { s u p } _ { \eta \in \mathcal { I } _ { T } } \left| \eta \right| < \infty$ . Then, uniformly over $\eta \in \mathcal { I } _ { T }$ and $k \leq 2 T$ , we have

$$
\| P B _ { A } ( \eta ) ^ { k } P \| _ { \mathrm { o p } } \lesssim q _ { T } ^ { k } ,\tag{27}
$$

$$
\begin{array} { r } { \| P ( B _ { A } ( \eta ) ^ { k } - B _ { K } ( \eta ) ^ { k } ) P \| _ { \mathrm { o p } } \lesssim k q _ { T } ^ { k - 1 } \| E \| _ { \mathrm { o p } } . } \end{array}\tag{28}
$$

Proof. Define $\mathcal { H } = \mathrm { r a n g e } ( K ) = \ker ( K ) ^ { \perp }$ . Since A and K are symmetric and have the same kernel, we have

$$
\operatorname { r a n g e } ( A ) = \ker ( A ) ^ { \perp } = \ker ( K ) ^ { \perp } = \operatorname { r a n g e } ( K ) = \mathcal { H } .
$$

Consequently, H and $\mathcal { H } ^ { \perp }$ are invariant under A, K, and $E = A - K$ . Equivalently, we have<sup>4</sup>

$$
P A = A P , \qquad P K = K P , \qquad P E = E P ,
$$

and hence P also commutes with $B _ { A } ( \eta )$ and $B _ { K } ( \eta )$ . In particular, we have

$$
P B _ { A } ( \eta ) ^ { k } P = \left( P B _ { A } ( \eta ) P \right) ^ { k } , \qquad P B _ { K } ( \eta ) ^ { k } P = \left( P B _ { K } ( \eta ) P \right) ^ { k } .
$$

These identities hold as operators on H.

In the remainder of the proof, we regard these matrices as linear operators from H to itself. Fix $\eta \in \mathcal { I } _ { T }$ and define

$$
B _ { 0 } = { \cal P } B _ { K } ( \eta ) { \cal P } \big | _ { \mathcal { H } } = B _ { K } ( \eta ) \big | _ { \mathcal { H } } , \qquad \Delta = - \frac { \eta } { m } { \cal P } { \cal E } { \cal P } \big | _ { \mathcal { H } } = - \frac { \eta } { m } { \cal E } \big | _ { \mathcal { H } } .
$$

Then

$$
\left. P B _ { A } ( \eta ) P \right| _ { \mathcal { H } } = B _ { 0 } + \Delta .
$$

With respect to the orthogonal decomposition $\mathbb { R } ^ { m } = \mathcal { H } \oplus \mathcal { H } ^ { \perp }$ , the operators A and K have the block representations

$$
A = \left[ \begin{array} { l l } { A | _ { \mathcal { H } } } & { 0 } \\ { 0 } & { 0 } \end{array} \right] , \qquad K = \left[ \begin{array} { l l } { K | _ { \mathcal { H } } } & { 0 } \\ { 0 } & { 0 } \end{array} \right] .\tag{29}
$$

Consequently, we have

$$
B _ { A } ( \eta ) = \left[ { B _ { 0 } + \Delta \quad 0 } \atop { 0 } \right] , \qquad B _ { K } ( \eta ) = \left[ { B _ { 0 } \atop 0 } \ { \begin{array} { c } { { 0 } } \\ { { I _ { \mathcal { H } ^ { \bot } } } } \end{array} } \right] .\tag{30}
$$

By assumption, we have $\| B _ { 0 } \| _ { \mathrm { o p } } \leq q _ { T }$ . Set $\begin{array} { r } { M _ { \eta } = \operatorname* { s u p } _ { T \geq 1 } \operatorname* { s u p } _ { \eta \in { \mathcal { I } } _ { T } } | \eta | / m < \infty } \end{array}$ . Then it follows that $\| \Delta \| _ { \mathrm { o p } } \leq M _ { \eta } \| E \| _ { \mathrm { o p } } .$ . Since $T \| E \| _ { \mathrm { o p } } \leq c ,$ for $k \leq 2 T$ , we have

$$
\frac { k \| \Delta \| _ { \mathrm { o p } } } { q _ { T } } \leq \frac { 2 M _ { \eta } T \| E \| _ { \mathrm { o p } } } { q _ { T } } \leq \frac { 2 M _ { \eta } c } { q _ { 0 } } .
$$

Thus $k \| \Delta \| _ { \mathrm { o p } } / q _ { T }$ is bounded uniformly in $\eta , k ,$ and $T .$

<sup>4</sup>This equivalence can be proved as follows. For any $x \in \mathbb { R } ^ { m }$ , decompose $x = P x + ( I _ { m } - P ) x$ . Here, $P x \in \mathcal { H }$ and $( I _ { m } - P ) x \in \mathcal { H } ^ { \perp }$ . If H and $\mathcal { H } ^ { \perp }$ are invariant under A, we have $A P x \in { \mathcal { H } }$ and $A ( I _ { m } - P ) x \in \mathcal { H } ^ { \perp }$ Thus,

$$
P A x = P [ A P x + A ( I _ { m } - P ) x ] = P ( A P x ) + P [ A ( I _ { m } - P ) x ] = A P x + 0 .
$$

We first prove (27). For $s \leq k \leq 2 T$ , we have

$$
\Vert ( B _ { 0 } + \Delta ) ^ { s } \Vert _ { \mathrm { o p } } \leq ( \Vert B _ { 0 } \Vert _ { \mathrm { o p } } + \Vert \Delta \Vert _ { \mathrm { o p } } ) ^ { s } \leq ( q _ { T } + \Vert \Delta \Vert _ { \mathrm { o p } } ) ^ { s } = q _ { T } ^ { s } \left( 1 + \frac { \Vert \Delta \Vert _ { \mathrm { o p } } } { q _ { T } } \right) ^ { s }
$$

$$
\leq q _ { T } ^ { s } \exp \left( \frac { s \| \Delta \| _ { \mathrm { o p } } } { q _ { T } } \right) \leq q _ { T } ^ { s } \exp \left( \frac { 2 M _ { \eta } c } { q _ { 0 } } \right) \lesssim q _ { T } ^ { s } ,
$$

which proves (27). By (30), for every $k \geq 1$

$$
B _ { A } ( \eta ) ^ { k } - B _ { K } ( \eta ) ^ { k } = \left[ \begin{array} { c c } { ( B _ { 0 } + \Delta ) ^ { k } - B _ { 0 } ^ { k } } & { 0 } \\ { 0 } & { 0 } \end{array} \right] .
$$

Therefore,

$$
\begin{array} { r } { \left\| B _ { A } ( \eta ) ^ { k } - B _ { K } ( \eta ) ^ { k } \right\| _ { \mathrm { o p } , \mathbb { R } ^ { m } } = \left\| ( B _ { 0 } + \Delta ) ^ { k } - B _ { 0 } ^ { k } \right\| _ { \mathrm { o p } , \mathcal { H } } . } \end{array}
$$

Thus it is suficient to establish the desired power bounds on H. The telescoping identity $\begin{array} { r } { \Gamma ^ { k } - \Lambda ^ { k } = \sum _ { s = 0 } ^ { k - 1 } \Gamma ^ { s } ( \Gamma - \Lambda ) \Lambda ^ { k - 1 - s } } \end{array}$ implies

$$
( B _ { 0 } + \Delta ) ^ { k } - B _ { 0 } ^ { k } = \sum _ { s = 0 } ^ { k - 1 } ( B _ { 0 } + \Delta ) ^ { s } \Delta B _ { 0 } ^ { k - 1 - s } .
$$

Consequently, we have

$$
\begin{array} { r l r } {  { \| ( B _ { 0 } + \Delta ) ^ { k } - B _ { 0 } ^ { k } \| _ { \mathrm { o p } } \leq \sum _ { s = 0 } ^ { k - 1 } \| ( B _ { 0 } + \Delta ) ^ { s } \| _ { \mathrm { o p } } \| \Delta \| _ { \mathrm { o p } } \| B _ { 0 } ^ { k - 1 - s } \| _ { \mathrm { o p } } } } \\ & { } & { \lesssim \| \Delta \| _ { \mathrm { o p } } \sum _ { s = 0 } ^ { k - 1 } q _ { T } ^ { s } q _ { T } ^ { k - 1 - s } = k q _ { T } ^ { k - 1 } \| \Delta \| _ { \mathrm { o p } } \lesssim k q _ { T } ^ { k - 1 } \| E \| _ { \mathrm { o p } } , } \end{array}
$$

which implies (28).

## F.3 Application to the finite-width kernel

Lemma F.3 (Uniform finite-width contraction near the optimal learning rate). Suppose m and d are fixed and $T = T _ { n }$ satisfies $T \to \infty$ and $T / \sqrt { n }  0$ . Suppose Assumption 3.2 holds. Let $\begin{array} { r } { P = \sum _ { i = 1 } ^ { r } P _ { j } } \end{array}$ denote the orthogonal projector onto range(K). Then uniformly over $| \eta - \eta _ { \infty , T } | \leq \dot { C } / T$ and $0 \leq k \leq 2 T$ , we have

$$
\| P B _ { \infty } ( \eta ) ^ { k } P \| _ { \mathrm { o p } } \lesssim q _ { * } ^ { k } .
$$

For any $c > 0$ , define the event

$$
\displaystyle \mathcal E _ { n } = \{ \mathrm { k e r } ( A _ { n } ) = \mathrm { k e r } ( K ) , T \| A _ { n } - K \| _ { \mathrm { o p } } \leq c \} .
$$

Then we have $\mathrm { P } ( \mathcal { E } _ { n } )  1$ , and uniformly over $| \eta - \eta _ { \infty , T } | \leq C / T$ and $0 \leq k \leq 2 T$ , we have

$$
\begin{array} { r } { \| P B _ { n } ( \eta ) ^ { k } P \| _ { \mathrm { o p } } \lesssim q _ { * } ^ { k } , \qquad \| P \left( B _ { n } ( \eta ) ^ { k } - B _ { \infty } ( \eta ) ^ { k } \right) P \| _ { \mathrm { o p } } \lesssim k q _ { * } ^ { k - 1 } \| A _ { n } - K \| _ { \mathrm { o p } } } \end{array}
$$

on ${ \mathcal { E } } _ { n }$ . On this event, we also have

$$
P A _ { n } = A _ { n } P , \qquad P K = K P , \qquad P ( A _ { n } - K ) = ( A _ { n } - K ) P .
$$

Proof. We confirm that the assumptions of Lemma $\mathrm { F } . 2$ are satisfied with probability tending to one. First, we have

$$
\left\| P B _ { \infty } ( \eta ) P \right\| _ { \mathrm { o p } } = \left\| \sum _ { j = 1 } ^ { r } \left( 1 - \frac { \eta \lambda _ { j } } { m } \right) P _ { j } \right\| _ { \mathrm { o p } } = \operatorname* { m a x } _ { j \in [ r ] } \left| 1 - \frac { \eta \lambda _ { j } } { m } \right| = \left| 1 - \frac { \eta \lambda _ { 1 } } { m } \right| \vee \left| 1 - \frac { \eta \lambda _ { r } } { m } \right| .
$$

Since $\begin{array} { r } { \| P B _ { \infty } ( \eta _ { * } ) P \| _ { \mathrm { o p } } = \left| 1 - \frac { \eta _ { * } \lambda _ { 1 } } { m } \right| = \left| 1 - \frac { \eta _ { * } \lambda _ { r } } { m } \right| = q , } \end{array}$ <sub>∗</sub> holds, we have

$$
\left| \left\| P B _ { \infty } ( \eta ) P \right\| _ { \mathrm { o p } } - q _ { * } \right| \leq \left| \left| 1 - \frac { \eta \lambda _ { 1 } } { m } \right| - q _ { * } \right| \vee \left| \left| 1 - \frac { \eta \lambda _ { r } } { m } \right| - q _ { * } \right| \lesssim | \eta - \eta _ { * } | .
$$

Thus, there exists a constant $C _ { 1 }$ such that, uniformly over $| \eta - \eta _ { \infty , T } | \leq C / T$

$$
\| P B _ { \infty } ( \eta ) P \| _ { \mathrm { o p } } \leq q _ { * } + \frac { C _ { 1 } } { T } .\tag{31}
$$

This implies

$$
\| P B _ { \infty } ( \eta ) ^ { k } P \| _ { \mathrm { o p } } \le \| P B _ { \infty } ( \eta ) P \| _ { \mathrm { o p } } ^ { k } \le \left( q _ { * } + \frac { C _ { 1 } } { T } \right) ^ { k } \le q _ { * } ^ { k } \exp \left( \frac { k C _ { 1 } } { q _ { * } T } \right) \lesssim q _ { * } ^ { k } \qquad ( 0 \le k \le 2 T ) .
$$

By Lemma 3.3, we have

$$
\| A _ { n } - K \| _ { \mathrm { o p } } = O _ { p } ( n ^ { - 1 / 2 } ) , \qquad \| \chi _ { n } ^ { ( 0 ) } + y \| = O _ { p } ( n ^ { - 1 / 2 } ) .
$$

The assumption $T / \sqrt { n }  0$ therefore implies $T \| A _ { n } - K \| _ { \mathrm { o p } } = o _ { p } ( 1 )$ . Consequently, we have $\mathrm { P } ( \mathcal { E } _ { n } )  1$

Note that $\begin{array} { r } { \{ q _ { * } + \frac { C _ { 1 } } { T } \} _ { T } } \end{array}$ is bounded away from zero. Thus, by Lemma F.2, on the event ${ \mathcal { E } } _ { n }$ uniformly over $| \eta - \bar { \eta } _ { \infty , T } | \leq C / T$ and $0 \leq k \leq 2 T$ , we have

$$
\| P B _ { n } ( \eta ) ^ { k } P \| _ { \mathrm { o p } } \lesssim \biggl ( q _ { * } + \frac { C _ { 1 } } { T } \biggr ) ^ { k } = q _ { * } ^ { k } \left( 1 + \frac { C _ { 1 } } { q _ { * } T } \right) ^ { k } \leq q _ { * } ^ { k } \exp \left( \frac { C _ { 1 } k } { q _ { * } T } \right) \leq q _ { * } ^ { k } \exp \left( \frac { 2 C _ { 1 } } { q _ { * } } \right) \lesssim q _ { * } ^ { k } ,
$$

and similarly,

$$
\| P \left( B _ { n } ( \eta ) ^ { k } - B _ { \infty } ( \eta ) ^ { k } \right) P \| _ { \mathrm { o p } } \lesssim k \left( q _ { * } + \frac { C _ { 1 } } { T } \right) ^ { k - 1 } \| A _ { n } - K \| _ { \mathrm { o p } } \lesssim k q _ { * } ^ { k - 1 } \| A _ { n } - K \| _ { \mathrm { o p } } .
$$

On this event, we also have

$$
P A _ { n } = A _ { n } P , \qquad P K = K P , \qquad P ( A _ { n } - K ) = ( A _ { n } - K ) P
$$

by Lemma F.2.

## G Proof of the joint width-horizon transfer theorem

Throughout this section, assume the conditions of Theorem 3.4.

By assumption, we have $\begin{array} { r } { \eta _ { \ell } < \eta _ { * } < \eta _ { u } < \frac { 2 m } { \lambda _ { 1 } } } \end{array}$ . Since Proposition E.1 implies $\eta _ { \infty , T } =$ $\eta _ { * } + O ( T ^ { - 1 } )$ , there exists a constant $C > 0$ such that, for all suficiently large $T$

$$
\{ \eta : | \eta - \eta _ { \infty , T } | \leq C / T \} \subset ( \eta _ { \ell } , \eta _ { u } ) .
$$

Let $\begin{array} { r } { P = \sum _ { j = 1 } ^ { r } P _ { j } } \end{array}$ be the orthogonal projector onto range(K), and set $E _ { n } = A _ { n } - K$ By Lemma C.2, for all suficiently large n, we have ker $( A _ { n } ) \stackrel { \mathrm { a . s . } } { = } \ker ( K )$ and $f _ { n } ^ { ( 0 ) } ( X ) ~ \in ~$ $\operatorname { r a n g e } ( H _ { n } ) { \stackrel { \mathrm { a . s . } } { = } } \operatorname { r a n g e } ( K )$ . Consequently, we have

$$
P f _ { n } ^ { ( 0 ) } ( X ) \stackrel { \mathrm { a . s . } } { = } f _ { n } ^ { ( 0 ) } ( X ) , \qquad P E _ { n } \stackrel { \mathrm { a . s . } } { = } E _ { n } P \stackrel { \mathrm { a . s . } } { = } E _ { n } .\tag{32}
$$

## G.1 Uniform score approximation

Lemma G.1 (Uniform approximation of the score and its derivative). Note that Proposition E.1 places $\eta _ { \infty , T }$ in a C/T-neighborhood of η for some C. For this neighborhood, we have

$$
\begin{array} { r l r } & { } & { \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } | F _ { n , T } ( \eta ) - F _ { \infty , T } ( \eta ) | = O _ { p } \left( \frac { T q _ { * } ^ { 2 T - 2 } } { \sqrt { n } } \right) , } \\ & { } & { \underset { | \eta - \eta _ { \infty , T } | \leq C / T } { \operatorname* { s u p } } | F _ { n , T } ^ { \prime } ( \eta ) - F _ { \infty , T } ^ { \prime } ( \eta ) | = O _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 3 } } { \sqrt { n } } \right) . } \end{array}
$$

Proof. An $O _ { p } ( \cdot )$ bound established on an event whose probability tends to one also holds unconditionally. Thus, it sufices to show upper bounds for $\begin{array} { r } { \operatorname* { s u p } _ { | \eta - \eta _ { \infty , T } | \leq C / T } | F _ { n , T } ( \eta ) - F _ { \infty , T } ( \eta ) | } \end{array}$ and $\begin{array} { r } { \operatorname* { s u p } _ { | \eta - \eta _ { \infty , T } | \leq C / T } | F _ { n , T } ^ { \prime } ( \eta ) - F _ { \infty , T } ^ { \prime } ( \eta ) | } \end{array}$ | on the event ${ \mathcal { E } } _ { n }$ defined in Lemma F.3. We first consider the score. Set

$$
M _ { n , 2 T - 1 } ( \eta ) = A _ { n } B _ { n } ( \eta ) ^ { 2 T - 1 } , \qquad M _ { \infty , 2 T - 1 } ( \eta ) = K B _ { \infty } ( \eta ) ^ { 2 T - 1 } .
$$

On the event ${ \mathcal { E } } _ { n }$ , we have

$$
\begin{array} { r l } & { \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } | | P \left( M _ { n , 2 T - 1 } ( \eta ) - M _ { \infty , 2 T - 1 } ( \eta ) \right) P | | _ { \mathrm { c p } } } \\ & { = \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } | | P ( A _ { n } - K ) B _ { n } ( \eta ) ^ { 2 T - 1 } P + P K ( B _ { \infty } ( \eta ) ^ { 2 T - 1 } - B _ { \infty } ( \eta ) ^ { 2 T - 1 } ) P | | _ { \mathrm { c p } } } \\ & { \leq \| A _ { n } - K \| _ { \infty } \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } \| P B _ { n } ( \eta ) ^ { 2 T - 1 } P \| _ { \infty } } \\ & { \qquad + \| K \| _ { \infty } \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } \| P \left( B _ { n } ( \eta ) ^ { 2 T - 1 } - B _ { \infty } ( \eta ) ^ { 2 T - 1 } \right) P \| _ { \mathrm { c p } } } \\ & { = O _ { p } \left( \frac { Q _ { x } ^ { 2 T - 1 } } { \sqrt { n } } + \frac { T q _ { x } ^ { 2 T - 2 } } { \sqrt { n } } \right) } \\ & { = O _ { p } \left( \frac { T q _ { x } ^ { 2 T - 2 } } { \sqrt { n } } \right) . } \end{array}
$$

By Lemma F.3, on the same event, we have

$$
A _ { n } = P A _ { n } P , \qquad B _ { n } ( \eta ) P = P B _ { n } ( \eta ) .\tag{33}
$$

Therefore, for every integer $k \geq 0$ , we have

$$
A _ { n } B _ { n } ( \eta ) ^ { k } = P A _ { n } P B _ { n } ( \eta ) ^ { k } = P A _ { n } P B _ { n } ( \eta ) ^ { k } P = A _ { n } P B _ { n } ( \eta ) ^ { k } P .
$$

Consequently, we have

$$
\begin{array} { r l } { \underset { | \eta - \eta _ { \infty } , { \cal T } | \le C / T } { \operatorname* { s u p } } \| M _ { n , 2 T - 1 } ( \eta ) \| _ { \mathrm { o p } } \le \underset { | \eta - \eta _ { \infty } , { \cal T } | \le C / T } { \operatorname* { s u p } } \| A _ { n } \| _ { \mathrm { o p } } \| P B _ { n } ( \eta ) ^ { 2 T - 1 } P \| _ { \mathrm { o p } } } & { } \\ { \le q _ { \ast } ^ { 2 T - 1 } ( \| K \| _ { \mathrm { o p } } + \| A _ { n } - K \| _ { \mathrm { o p } } ) } & { } \\ { } & { = O _ { p } ( q _ { \ast } ^ { 2 T - 1 } ) . } \end{array}
$$

Since $A _ { n }$ commutes with $B _ { n } ( \eta )$ , the matrix $M _ { n , 2 T - 1 } ( \eta )$ is symmetric. Therefore,

$$
\begin{array} { r l } & { F _ { n , T } ( \eta ) - F _ { \infty , T } ( \eta ) = y ^ { \top } ( M _ { n , 2 T - 1 } ( \eta ) - M _ { \infty , 2 T - 1 } ( \eta ) ) y - 2 ( \chi _ { n } ^ { ( 0 ) } + y ) ^ { \top } M _ { n , 2 T - 1 } ( \eta ) y } \\ & { \qquad + \left( \chi _ { n } ^ { ( 0 ) } + y \right) ^ { \top } M _ { n , 2 T - 1 } ( \eta ) ( \chi _ { n } ^ { ( 0 ) } + y ) . } \end{array}
$$

On the event ${ \mathcal { E } } _ { n }$ , since both $M _ { n , 2 T - 1 } ( \eta )$ and $M _ { \infty , 2 T - 1 } ( \eta )$ vanish on $\ker ( K )$ , we may insert the projector $P$ in the first term:

$$
\begin{array} { r } { { y ^ { \top } } ( M _ { n , 2 T - 1 } ( \eta ) - M _ { \infty , 2 T - 1 } ( \eta ) ) y = { y ^ { \top } } P ( M _ { n , 2 T - 1 } ( \eta ) - M _ { \infty , 2 T - 1 } ( \eta ) ) P y . } \end{array}
$$

Thus, on the event ${ \mathcal { E } } _ { n }$ , we obtain

$$
\begin{array} { r l } { \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } | F _ { n , T } ( \eta ) - F _ { \infty , T } ( \eta ) | \leq \| y \| ^ { 2 } \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } \| P ( M _ { n , 2 T - 1 } ( \eta ) - M _ { \infty , 2 T - 1 } ( \eta ) ) P \| _ { \mathrm { o p } } } & { } \\ { + 2 \| \chi _ { n } ^ { ( 0 ) } + y \| \| y \| _ { | ^ { \infty } \eta _ { \infty } , \tau | \leq C / T } \| M _ { n , 2 T - 1 } ( \eta ) \| _ { \mathrm { o p } } } & { } \\ { + \| \chi _ { n } ^ { ( 0 ) } + y \| ^ { 2 } \underset { | \eta - \eta _ { \infty } , \tau | \leq C / T } { \operatorname* { s u p } } \| M _ { n , 2 T - 1 } ( \eta ) \| _ { \mathrm { o p } } } & { } \\ { = O _ { p } \left( \frac { T q _ { \infty } ^ { 2 T - 2 } } { \sqrt { n } } \right) + O _ { p } \left( \frac { q _ { \infty } ^ { 2 T - 1 } } { \sqrt { n } } \right) + O _ { p } \left( \frac { q _ { \infty } ^ { 2 T - 1 } } { n } \right) } & { } \\ { = O _ { p } \left( \frac { T q _ { \infty } ^ { 2 T - 2 } } { \sqrt { n } } \right) . } \end{array}
$$

We next consider the derivative of the score. Define

$$
R _ { n , 2 T - 2 } ( \eta ) : = A _ { n } ^ { 2 } B _ { n } ( \eta ) ^ { 2 T - 2 } , \qquad R _ { \infty , 2 T - 2 } ( \eta ) : = K ^ { 2 } B _ { \infty } ( \eta ) ^ { 2 T - 2 } .
$$

Then on the event ${ \mathcal { E } } _ { n }$ , we have by Lemma F.3 that

$$
\begin{array} { r l } & { \quad \underset { \{ u ^ { * } = \kappa , \eta \leq T \} } { \operatorname* { s u p } } \Vert P ( R _ { n , 2 : - ( \eta ) } - f _ { n } ) - R _ { \infty , \eta , z } \left[ \sigma ( \eta ) \right] P \Vert _ { \mathbf { p } } } \\ & { = \underset { \{ u ^ { * } = \kappa , \eta \leq T \} } { \operatorname* { s u p } } \Vert P ( A _ { n } ^ { 2 } , \kappa ^ { - 2 } ) B _ { n } ( u ) \Vert ^ { \mathcal { O } ^ { n - 2 } - p } P + P \kappa ^ { 2 } ( B _ { s } ( u ) ) ^ { \mathcal { O } ^ { n - 2 } - 2 } - B _ { \infty } ( u ) ^ { \mathcal { O } ^ { n - 2 } } ) P \Vert _ { \mathbf { x } } } \\ & { = \underset { \{ u ^ { * } = \kappa , \eta \leq T \} } { \operatorname* { s u p } } \Vert P ( \eta A _ { n } ^ { 2 } , u ) } \\ & { \leq \Vert A _ { n } ^ { 2 } - K ^ { 2 } \Vert _ { \infty } \underset { \{ u ^ { * } = \kappa , \eta \leq T \} } { \operatorname* { s u p } } \Vert P ( B _ { n } ( u ) ) ^ { \mathcal { H } ^ { 2 - p } } \Vert _ { \mathbf { g } } } \\ & { \qquad + \Vert R \Vert _ { \infty } ^ { 2 } \underset { \{ u ^ { * } = \kappa , \eta \leq W \} } { \operatorname* { s u p } } \Vert P \left( B _ { n } ( u ) ^ { \mathcal { O } ^ { n - 2 } } - B _ { \infty } ( u ) ^ { \mathcal { O } ^ { n - 2 } } \right) P \Vert _ { \mathbf { x } } } \\ & { \lesssim \Vert ( A _ { n } - K ) \vert _ { \mathcal { H } _ { \infty } ( + K } ( A _ { n } - K ) \Vert _ { \mathbf { g } ^ { \mathcal { O } ^ { n - 2 } } } + T q ^ { 2 \alpha } \underset { \{ u ^ { * } = \kappa \} } { \operatorname* { s u p } } \Vert _ { A _ { n } - K } \Vert _ { \mathbf { x } } } \\ &  \leq \Vert A _ { n } - K \Vert _ { \infty } ( \Vert A _ { n } \ \end{array}
$$

Moreover, (33) implies that, for every integer $k \geq 0$ 2

$$
A _ { n } ^ { 2 } B _ { n } ( \eta ) ^ { k } = P A _ { n } P A _ { n } P B _ { n } ( \eta ) ^ { k } = P A _ { n } ^ { 2 } P B _ { n } ( \eta ) ^ { k } = P A _ { n } ^ { 2 } P B _ { n } ( \eta ) ^ { k } P = A _ { n } ^ { 2 } P B _ { n } ( \eta ) ^ { k } P .
$$

Thus, we have

$$
\operatorname* { s u p } _ { | \eta - \eta _ { \infty } , r | \le C / T } \| R _ { n , 2 T - 2 } ( \eta ) \| _ { \mathrm { o p } } \le \operatorname* { s u p } _ { | \eta - \eta _ { \infty } , r | \le C / T } \| A _ { n } ^ { 2 } \| _ { \mathrm { o p } } \| P B _ { n } ( \eta ) ^ { 2 T - 2 } P \| _ { \mathrm { o p } } = O _ { p } ( q _ { * } ^ { 2 T - 2 } ) .
$$

Note that by Proposition C.3, we have

$$
\begin{array} { l } { { F _ { n , T } ^ { \prime } ( \eta ) - F _ { \infty , T } ^ { \prime } ( \eta ) } } \\ { { \displaystyle = - \frac { 2 T - 1 } { m } [ ( \chi _ { n } ^ { ( 0 ) } ) ^ { \top } R _ { n , 2 T - 2 } ( \eta ) \chi _ { n } ^ { ( 0 ) } - y ^ { \top } R _ { \infty , 2 T - 2 } ( \eta ) y ] } } \\ { { \displaystyle = - \frac { 2 T - 1 } { m } [ y ^ { \top } ( R _ { n , 2 T - 2 } ( \eta ) - R _ { \infty , 2 T - 2 } ( \eta ) ) y - 2 ( \chi _ { n } ^ { ( 0 ) } + y ) ^ { \top } R _ { n , 2 T - 2 } ( \eta ) y ] } } \\ { { \displaystyle ~ - \frac { 2 T - 1 } { m } ( \chi _ { n } ^ { ( 0 ) } + y ) ^ { \top } R _ { n , 2 T - 2 } ( \eta ) ( \chi _ { n } ^ { ( 0 ) } + y ) } . } \end{array}
$$

On the event ${ \mathcal { E } } _ { n }$ , since both $R _ { n , 2 T - 2 } ( \eta )$ and $R _ { \infty , 2 T - 2 } ( \eta )$ vanish on ker(K), we may insert the projector P in the first term:

$$
\begin{array} { r } { { y ^ { \top } } ( R _ { n , 2 T - 2 } ( \eta ) - R _ { \infty , 2 T - 2 } ( \eta ) ) y = { y ^ { \top } } P ( R _ { n , 2 T - 2 } ( \eta ) - R _ { \infty , 2 T - 2 } ( \eta ) ) P y . } \end{array}
$$

Thus, on the event ${ \mathcal { E } } _ { n }$ , we have

$$
\begin{array} { r l } & { \underset { | \eta - \eta _ { \infty } , \tau | \leq \bar { \tau } } { \operatorname* { s u p } } | F _ { \varepsilon , T } ^ { \varepsilon } ( \eta ) - F _ { \infty , T } ^ { \varepsilon } ( \eta ) | } \\ & { \leq \frac { 2 T - 1 } { m } \| g \| _ { \textnormal { \tiny { | | \eta - \eta _ { \infty } , \tau | \leq \bar { \tau } } } / \tau } ^ { 2 } \| P ( R _ { n , 2 : T - 2 } ( \eta ) - R _ { \infty , 2 : T - 2 } ( \eta ) ) P \| _ { \infty } } \\ & { \quad + \frac { 2 T - 1 } { m } 2 \| \chi _ { n } ^ { ( 0 ) } + \ y \| _ { 1 } \| g \| _ { \textnormal { \tiny { | | \eta - \eta _ { \infty } , \tau | \leq \bar { \tau } } / T } } | R _ { n , 2 : T - 2 } ( \eta ) \| _ { \infty } } \\ & { \quad + \frac { 2 T - 1 } { m } \| \chi _ { n } ^ { ( 0 ) } + y \| _ { \textnormal { \tiny { | | \eta - \eta _ { \infty } , \tau | \leq \bar { \tau } } / T } } ^ { 2 } \| R _ { n , 2 : T - 2 } ( \eta ) \| _ { \infty } } \\ & { = O _ { p } \left( \frac { T ^ { 2 } q _ { \infty } ^ { \varepsilon , - 3 } } { \sqrt { n } } \right) + O _ { p } \left( \frac { T q _ { \infty } ^ { \varepsilon , - 2 } } { \sqrt { n } } \right) + O _ { p } \left( \frac { T q _ { \infty } ^ { \varepsilon , - 2 } } { n } \right) } \\ & { = O _ { t } \left( \frac { T ^ { 2 } q _ { \infty } ^ { \varepsilon , - 3 } } { \sqrt { n } } \right) . } \end{array}
$$

## G.2 Localization of the finite-width optimizer

Lemma G.2 $( n ^ { - 1 / 2 } .$ -consistency of the finite-width optimizer). Under the assumptions of Theorem $\ 3 . 4 ,$ with probability tending to one, we have $\eta _ { n , T } \in \left( \eta _ { \ell } , \eta _ { u } \right)$ and $F _ { n , T } ( \eta _ { n , T } ) = 0$ Moreover, we have

$$
\eta _ { n , T } - \eta _ { \infty , T } = O _ { p } ( n ^ { - 1 / 2 } ) ,
$$

which implies

$$
| \eta _ { n , T } - \eta _ { \infty , T } | = o _ { p } ( T ^ { - 1 } ) .
$$

Proof. Define $s _ { T } = T q _ { * } ^ { 2 T - 2 }$ . By (22), there exist constants $0 < \underline { { C } } < \overline { { C } } < \infty$ such that

$$
\begin{array} { r } { C s _ { T } \leq - F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) \leq \overline { { C } } s _ { T } } \end{array}\tag{34}
$$

for all suficiently large T. Diferentiating (11) once more gives

$$
F _ { \infty , T } ^ { \prime \prime } ( \eta ) = \frac { ( 2 T - 1 ) ( 2 T - 2 ) } { m ^ { 2 } } y ^ { \top } K ^ { 3 } B _ { \infty } ( \eta ) ^ { 2 T - 3 } y .
$$

By Lemma F.3, uniformly over $| \eta - \eta _ { \infty , T } | \leq C / T$ , we have

$$
\begin{array} { r } { \| P B _ { \infty } ( \eta ) ^ { k } P \| _ { \mathrm { o p } } \lesssim q _ { * } ^ { k } \qquad ( 0 \leq k \leq 2 T ) , } \end{array}
$$

which implies

$$
\operatorname* { s u p } _ { | \eta - \eta _ { \infty , T } | \le C / T } | F _ { \infty , T } ^ { \prime \prime } ( \eta ) | \lesssim T ^ { 2 } \| P y \| ^ { 2 } \| K \| _ { \mathrm { o p } } ^ { 3 } \| P B _ { \infty } ( \eta ) ^ { 2 T - 3 } P \| _ { \mathrm { o p } } \lesssim T ^ { 2 } q _ { * } ^ { 2 T - 3 } .
$$

For every fixed $M > 0$ , the assumption $T / \sqrt { n }  0$ implies $M / \sqrt { n } \leq C / T$ for all suficiently large n. Hence we have

$$
\operatorname* { s u p } _ { | \eta - \eta _ { \infty } , \tau | \leq M / \sqrt { n } } | F _ { \infty , T } ^ { \prime } ( \eta ) - F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) | = O \left( \frac { M T ^ { 2 } q _ { * } ^ { 2 T - 3 } } { \sqrt { n } } \right) = o ( s _ { T } ) .\tag{35}
$$

Since Lemma G.1 gives

$$
\operatorname* { s u p } _ { | \eta - \eta _ { \infty } , \boldsymbol { T } | \leq M / \sqrt { n } } | F _ { n , T } ^ { \prime } ( \eta ) - F _ { \infty , T } ^ { \prime } ( \eta ) | = O _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 3 } } { \sqrt { n } } \right) = o _ { p } ( s _ { T } ) ,\tag{36}
$$

we have $\begin{array} { r } { \operatorname* { s u p } _ { | \eta - \eta _ { \infty , T } | \leq M / \sqrt { n } } F _ { n , T } ^ { \prime } ( \eta ) \ \leq \ F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) + o _ { p } ( s _ { T } ) } \end{array}$ . Combining this with (34), we obtain

$$
\mathrm { P } \left( \operatorname* { s u p } _ { | \eta - \eta _ { \infty } , \tau | \leq M / \sqrt { n } } F _ { n , T } ^ { \prime } ( \eta ) \leq - \frac { C } { 2 } s _ { T } \right) \to 1\tag{37}
$$

for every fixed $M > 0$

By Corollary C.4 and Theorem 3.1, we have $F _ { \infty , T } ( \eta _ { \infty , T } ) = 0$ . Thus the first bound in Lemma G.1 gives

$$
F _ { n , T } ( \eta _ { \infty , T } ) = O _ { p } \left( \frac { s _ { T } } { \sqrt { n } } \right) .
$$

Therefore, for any fixed $\epsilon > 0$ , there exists $M _ { 0 } < \infty$ such that

$$
\operatorname* { l i m i n f } _ { n \to \infty } \operatorname* { l i m } _ { } \operatorname* { i n f } _ { } \operatorname* { P } \left( | F _ { n , T } ( \eta _ { \infty , T } ) | \leq \frac { M _ { 0 } s _ { T } } { \sqrt { n } } \right) \geq 1 - \epsilon .\tag{38}
$$

Choose $M > 2 M _ { 0 } / \underline { { C } }$ . On the intersection of this event and the event in (37), the mean value theorem gives

$$
\begin{array} { l l } { \displaystyle { F _ { n , T } \left( \eta _ { \infty , T } - \frac { M } { \sqrt { n } } \right) \ge - \frac { M _ { 0 } s _ { T } } { \sqrt { n } } + \frac { C M s _ { T } } { 2 \sqrt { n } } > 0 , } } \\ { \displaystyle { F _ { n , T } \left( \eta _ { \infty , T } + \frac { M } { \sqrt { n } } \right) \le \frac { M _ { 0 } s _ { T } } { \sqrt { n } } - \frac { C M s _ { T } } { 2 \sqrt { n } } < 0 . } } \end{array}
$$

By Proposition C.3, $F _ { n , T }$ is continuous. On the event in (37), it is strictly decreasing throughout the interval $| \eta - \eta _ { \infty , T } | \leq M / \sqrt { n }$ . Hence the preceding sign inequalities and the intermediate value theorem imply that on the intersection of events in (37) and (38), $F _ { n , T }$ has a unique root in

$$
\left( \eta _ { \infty , T } - \frac { M } { \sqrt { n } } , \eta _ { \infty , T } + \frac { M } { \sqrt { n } } \right) \subset ( \eta _ { \ell } , \eta _ { u } ) .
$$

For all suficiently large $n ,$ Proposition C.5 and Corollary C.4 imply that $\phi _ { n , T }$ is almost surely strictly convex. Therefore this root is the unique minimizer of $\phi _ { n , T }$ over $\mathcal { Z }$ , namely $\eta _ { n , T }$ . Consequently, with probability at least $1 - \epsilon + o ( 1 )$ , we have

$$
| \eta _ { n , T } - \eta _ { \infty , T } | < \frac { M } { \sqrt { n } } .
$$

Since $\epsilon > 0$ was arbitrary, this proves that $\eta _ { n , T } \in \left( \eta _ { \ell } , \eta _ { u } \right)$ with probability tending to one and that $\eta _ { n , T } - \eta _ { \infty , T } = O _ { p } ( n ^ { - 1 / 2 } )$ □

## G.3 First-order perturbation expansion

We next derive the first-order triangular-array expansion of the score. Set

$$
B _ { T } = B _ { \infty } ( \eta _ { \infty , T } ) , \qquad \Delta _ { n } = - \frac { \eta _ { \infty , T } } { m } E _ { n } .
$$

Then we have

$$
B _ { n } ( \eta _ { \infty , T } ) = B _ { T } + \Delta _ { n } .
$$

For $1 \leq k \leq 2 T$ , define

$$
\mathcal { L } _ { k , n } = \sum _ { s = 0 } ^ { k - 1 } B _ { T } ^ { s } \Delta _ { n } B _ { T } ^ { k - 1 - s } , \qquad \mathcal { R } _ { k , n } = ( B _ { T } + \Delta _ { n } ) ^ { k } - B _ { T } ^ { k } - \mathcal { L } _ { k , n } .
$$

Then by Lemma F.1, we have

$$
{ \mathcal { L } } _ { k , n } = D \Psi _ { k } ( B _ { T } ) [ \Delta _ { n } ] , \qquad \Psi _ { k } ( B ) = B ^ { k } .
$$

Lemma G.3 (Linear and quadratic matrix-power perturbations). Let P denote the orthogonal projector onto range(K). Then we have

$$
\| P \mathcal { L } _ { k , n } P \| _ { \mathrm { o p } } = O _ { p } ( k q _ { * } ^ { k - 1 } \| E _ { n } \| _ { \mathrm { o p } } ) , \qquad \| P \mathcal { R } _ { k , n } P \| _ { \mathrm { o p } } = O _ { p } ( k ^ { 2 } q _ { * } ^ { k - 2 } \| E _ { n } \| _ { \mathrm { o p } } ^ { 2 } ) .
$$

Proof. Define the event ${ \mathcal { E } } _ { n }$ as in Lemma F.3. By Lemma F.3, we have $\mathrm { P } ( \mathcal { E } _ { n } )  1$ . An $O _ { p } ( \cdot )$ bound established on an event whose probability tends to one also holds unconditionally. Thus, it sufices to show that

$$
\begin{array} { r } { \| P \mathcal { L } _ { k , n } P \| _ { \mathrm { o p } } \lesssim k q _ { * } ^ { k - 1 } \| E _ { n } \| _ { \mathrm { o p } } , \qquad \| P \mathcal { R } _ { k , n } P \| _ { \mathrm { o p } } \lesssim k ^ { 2 } q _ { * } ^ { k - 2 } \| E _ { n } \| _ { \mathrm { o p } } ^ { 2 } } \end{array}
$$

holds on ${ \mathcal { E } } _ { n }$

First, note that Lemma F.3 gives

$$
\| P B _ { T } ^ { j } P \| _ { \mathrm { o p } } \lesssim q _ { * } ^ { j } \qquad ( 0 \leq j \leq 2 T ) .
$$

Since Proposition E.1 implies $\eta _ { \infty , T } = \eta _ { * } + O ( T ^ { - 1 } )$ , we have sup<sub>T</sub> $\eta _ { \infty , T } < \infty$ . Hence, by the definition of $\Delta _ { n }$ , we have

$$
\| \Delta _ { n } \| _ { \mathrm { o p } } \lesssim \| E _ { n } \| _ { \mathrm { o p } } .\tag{39}
$$

Therefore, we obtain

$$
\begin{array} { r l } & { \| P \mathcal { L } _ { k , n } P \| _ { \mathrm { o p } } \leq \displaystyle \sum _ { s = 0 } ^ { k - 1 } \| P ( B _ { T } ^ { s } \Delta _ { n } B _ { T } ^ { k - 1 - s } ) P \| _ { \mathrm { o p } } \leq \displaystyle \sum _ { s = 0 } ^ { k - 1 } \| P B _ { T } ^ { s } P \| _ { \mathrm { o p } } \| \Delta _ { n } \| _ { \mathrm { o p } } \| P B _ { T } ^ { k - 1 - s } P \| _ { \mathrm { o p } } } \\ & { \qquad \lesssim \displaystyle \sum _ { s = 0 } ^ { k - 1 } q _ { * } ^ { s } \| \Delta _ { n } \| _ { \mathrm { o p } } q _ { * } ^ { k - 1 - s } = k q _ { * } ^ { k - 1 } \| \Delta _ { n } \| _ { \mathrm { o p } } \lesssim k q _ { * } ^ { k - 1 } \| E _ { n } \| _ { \mathrm { o p } } . } \end{array}
$$

Moreover, since $B _ { T } + \Delta _ { n } = B _ { n } ( \eta _ { \infty , T } )$ , Lemma F.3 gives

$$
\| P ( B _ { T } + \Delta _ { n } ) ^ { j } P \| _ { \mathrm { o p } } \lesssim q _ { * } ^ { j }
$$

uniformly over $0 \leq j \leq 2 T$ on ${ \mathcal { E } } _ { n }$

For the remainder, the proof of Lemma F.1 gives the identity

$$
\mathcal { R } _ { k , n } = \sum _ { s = 1 } ^ { k - 1 } \sum _ { j = 0 } ^ { s - 1 } ( B _ { T } + \Delta _ { n } ) ^ { j } \Delta _ { n } B _ { T } ^ { s - 1 - j } \Delta _ { n } B _ { T } ^ { k - 1 - s }
$$

for $k \geq 2$ , while $\mathcal { R } _ { 1 , n } = 0$ . Hence, using the preceding projected power bounds and (39), we obtain

$$
\begin{array} { r l } {  { \| P \mathcal { R } _ { k , n } P \| _ { \mathrm { o p } } \leq \displaystyle \sum _ { s = 1 } ^ { k - 1 } \sum _ { j = 0 } ^ { s - 1 } \| P ( B _ { T } + \Delta _ { n } ) ^ { j } P \| _ { \mathrm { o p } } \| \Delta _ { n } \| _ { \mathrm { o p } } \| P B _ { T } ^ { s - 1 - j } P \| _ { \mathrm { o p } } \| \Delta _ { n } \| _ { \mathrm { o p } } \| P B _ { T } ^ { k - 1 - s } P \| _ { \mathrm { o p } } } } \\ & { \lesssim \displaystyle \sum _ { s = 1 } ^ { k - 1 } \displaystyle \sum _ { j = 0 } ^ { s - 1 } q _ { * } ^ { j } \| \Delta _ { n } \| _ { \mathrm { o p } } ^ { 2 } q _ { * } ^ { s - 1 - j } q _ { * } ^ { k - 1 - s } = \displaystyle \sum _ { s = 1 } ^ { k - 1 } s q _ { * } ^ { k - 2 } \| \Delta _ { n } \| _ { \mathrm { o p } } ^ { 2 } = \frac { k ( k - 1 ) } { 2 } q _ { * } ^ { k - 2 } \| \Delta _ { n } \| _ { \mathrm { o p } } ^ { 2 } } \\ & { \lesssim k ^ { 2 } q _ { * } ^ { k - 2 } \| E _ { n } \| _ { \mathrm { o p } } ^ { 2 } . } \end{array}
$$

## G.4 Limits of the first-order sensitivities of the loss and score

Lemma G.4 (Asymptotics of the local Fr´echet coeficients). We have

$$
\frac { P Q _ { F , T } P } { T q _ { * } ^ { 2 T - 2 } }  Q _ { F , * } : = - \frac { 2 \eta _ { * } h _ { * } } { m } ( u _ { 1 } u _ { 1 } ^ { \top } + u _ { r } u _ { r } ^ { \top } ) , \qquad \frac { P Q _ { L , T } P } { T q _ { * } ^ { 2 T - 1 } }  Q _ { L , * }
$$

in Frobenius norm, where $Q _ { L , }$ <sub>∗</sub> is defined by $Q _ { L , * } = m ^ { - 2 } \eta _ { * } \big ( c _ { 1 } a _ { 1 } ^ { * } u _ { 1 } u _ { 1 } ^ { \top } - c _ { r } a _ { r } ^ { * } u _ { r } u _ { r } ^ { \top } \big )$ . In addition, we have

$$
\frac { \| P g _ { F , T } \| } { T q _ { * } ^ { 2 T - 2 } }  0 , \qquad \frac { \| P g _ { L , T } \| } { T q _ { * } ^ { 2 T - 1 } }  0 .
$$

Proof. Set

$$
\Gamma _ { i j , T } ^ { ( k ) } : = \sum _ { s = 0 } ^ { k - 1 } r _ { i , T } ^ { k - 1 - s } r _ { j , T } ^ { s } = \left\{ \begin{array} { l l } { k r _ { i , T } ^ { k - 1 } } & { ( i = j ) , } \\ { ( r _ { i , T } ^ { k } - r _ { j , T } ^ { k } ) / ( r _ { i , T } - r _ { j , T } ) } & { ( i \ne j ) . } \end{array} \right.
$$

Since $\lambda _ { i } \neq \lambda _ { j }$ for $i \neq j$ and $\eta _ { \infty , T } \to \eta _ { * } > 0$ , the denominator $| r _ { i , T } - r _ { j , T } |$ is bounded away from zero for all suficiently large $T .$ . Directly projecting $Q _ { F , T }$ onto the two eigenspaces gives

$$
P _ { i } Q _ { F , T } P _ { j } = \left( \frac { r _ { i , T } ^ { 2 T - 1 } + r _ { j , T } ^ { 2 T - 1 } } { 2 } - \frac { \eta _ { \infty , T } ( \lambda _ { i } + \lambda _ { j } ) } { 2 m } \Gamma _ { i j , T } ^ { ( 2 T - 1 ) } \right) ( P _ { i } y ) ( P _ { j } y ) ^ { \top } .
$$

In particular, we have

$$
P _ { j } Q _ { F , T } P _ { j } = \left( r _ { j , T } ^ { 2 T - 1 } - \frac { \eta _ { \infty , T } ( 2 T - 1 ) \lambda _ { j } } { m } r _ { j , T } ^ { 2 T - 2 } \right) ( P _ { j } y ) ( P _ { j } y ) ^ { \top } .
$$

By Lemma E.3, for every $j = 2 , \ldots , r - 1$ , we have $P _ { j } Q _ { F , T } P _ { j } = o ( T q _ { * } ^ { 2 T - 2 } )$ . For $i \neq j$ noting that $| \Gamma _ { i j , T } ^ { ( 2 T - 1 ) } | \lesssim | r _ { i , T } | ^ { 2 T - 1 } + | r _ { j , T } | ^ { 2 T - 1 }$ , we have $P _ { i } Q _ { F , T } P _ { j } = O ( q _ { * } ^ { 2 T - 1 } ) = o ( T q _ { * } ^ { 2 T - 2 } )$ by Lemma E.3. By Assumption 3.2, we can write $( P _ { 1 } y ) ( P _ { 1 } y ) ^ { \top } = c _ { 1 } u _ { 1 } u _ { 1 } ^ { \top }$ and $( P _ { r } y ) ( P _ { r } y ) ^ { \top } =$ $c _ { r } u _ { r } u _ { r } ^ { \top }$ . Using Lemma E.3 and $\eta _ { \infty , T } \to \eta _ { * }$ , we have

$$
\begin{array} { r l } & { \frac { P _ { 1 } Q _ { F , T } P _ { 1 } } { T q _ { * } ^ { 2 T - 2 } } = \bigg ( r _ { 1 , T } ^ { 2 T - 1 } - \frac { \eta _ { \infty , T } ( 2 T - 1 ) \lambda _ { 1 } } { m } r _ { 1 , T } ^ { 2 T - 2 } \bigg ) \frac { ( P _ { 1 } y ) ( P _ { 1 } y ) ^ { \top } } { T q _ { * } ^ { 2 T - 2 } } } \\ & { \qquad - \frac { \eta _ { * } 2 \lambda _ { 1 } } { m } a _ { 1 } ^ { * } ( P _ { 1 } y ) ( P _ { 1 } y ) ^ { \top } = - \frac { 2 \eta _ { * } h _ { * } } { m } u _ { 1 } u _ { 1 } ^ { \top } , } \\ & { \frac { P _ { r } Q _ { F , T } P _ { r } } { T q _ { * } ^ { 2 T - 2 } } = \bigg ( r _ { r , T } ^ { 2 T - 1 } - \frac { \eta _ { \infty , T } ( 2 T - 1 ) \lambda _ { r } } { m } r _ { r , T } ^ { 2 T - 2 } \bigg ) \frac { ( P _ { r } y ) ( P _ { r } y ) ^ { \top } } { T q _ { * } ^ { 2 T - 2 } } } \\ & { \qquad - \frac { \eta _ { * } 2 \lambda _ { r } } { m } a _ { r } ^ { * } ( P _ { r } y ) ( P _ { r } y ) ^ { \top } = - \frac { 2 \eta _ { * } h _ { * } } { m } u _ { r } u _ { r } ^ { \top } . } \end{array}
$$

Thus we obtain

$$
\frac { P Q _ { F , T } P } { T q _ { * } ^ { 2 T - 2 } } = \sum _ { i = 1 } ^ { r } \sum _ { j = 1 } ^ { r } \frac { P _ { i } Q _ { F , T } P _ { j } } { T q _ { * } ^ { 2 T - 2 } } = \frac { P _ { 1 } Q _ { F , T } P _ { 1 } } { T q _ { * } ^ { 2 T - 2 } } + \frac { P _ { r } Q _ { F , T } P _ { r } } { T q _ { * } ^ { 2 T - 2 } } + o ( 1 )  Q _ { F , * } .
$$

The loss coeficient is handled in the same way. Projecting $Q _ { L , T }$ gives

$$
P _ { i } Q _ { L , T } P _ { j } = - \frac { \eta _ { \infty , T } } { 2 m ^ { 2 } } \Gamma _ { i j , T } ^ { ( 2 T ) } ( P _ { i } y ) ( P _ { j } y ) ^ { \top } .
$$

Thus all of-diagonal blocks and all interior diagonal blocks are $o ( T q _ { * } ^ { 2 T - 1 } )$ . At the two endpoints, we have by Lemma E.3 and $\eta _ { \infty , T } \to \eta _ { \ast }$ <sub>∗</sub> that

$$
\begin{array} { r l } & { \frac { P _ { 1 } Q _ { L , T } P _ { 1 } } { T q _ { * } ^ { 2 T - 1 } } = - \frac { \eta _ { \infty , T } 2 T r _ { 1 , T } ^ { 2 T - 1 } ( P _ { 1 } y ) ( P _ { 1 } y ) ^ { \top } } { 2 m ^ { 2 } T q _ { * } ^ { 2 T - 1 } } \to - \frac { \eta _ { * } } { m ^ { 2 } } ( - a _ { 1 } ^ { * } ) ( P _ { 1 } y ) ( P _ { 1 } y ) ^ { \top } = \frac { \eta _ { * } } { m ^ { 2 } } c _ { 1 } a _ { 1 } ^ { * } u _ { 1 } u _ { 1 } ^ { \top } , } \\ & { \frac { P _ { r } Q _ { L , T } P _ { r } } { T q _ { * } ^ { 2 T - 1 } } = - \frac { \eta _ { \infty , T } 2 T r _ { r , T } ^ { 2 T - 1 } ( P _ { r } y ) ( P _ { r } y ) ^ { \top } } { 2 m ^ { 2 } T q _ { * } ^ { 2 T - 1 } } \to - \frac { \eta _ { * } } { m ^ { 2 } } a _ { r } ^ { * } ( P _ { r } y ) ( P _ { r } y ) ^ { \top } = - \frac { \eta _ { * } } { m ^ { 2 } } c _ { r } a _ { r } ^ { * } u _ { r } u _ { r } ^ { \top } . } \end{array}
$$

Thus we obtain

$$
\frac { P Q _ { L , T } P } { T q _ { * } ^ { 2 T - 1 } } = \sum _ { i = 1 } ^ { r } \sum _ { j = 1 } ^ { r } \frac { P _ { i } Q _ { L , T } P _ { j } } { T q _ { * } ^ { 2 T - 1 } } = \frac { P _ { 1 } Q _ { L , T } P _ { 1 } } { T q _ { * } ^ { 2 T - 1 } } + \frac { P _ { r } Q _ { L , T } P _ { r } } { T q _ { * } ^ { 2 T - 1 } } + o ( 1 )  Q _ { L , * } .
$$

Finally, on range(K), Lemma E.3 gives

$$
\| P B _ { T } ^ { 2 T - 1 } P \| _ { \mathrm { o p } } = \left\| \sum _ { i = 1 } ^ { r } r _ { i , T } ^ { 2 T - 1 } P _ { i } \right\| \le \sum _ { i = 1 } ^ { r } | r _ { i , T } | ^ { 2 T - 1 } \lesssim q _ { * } ^ { 2 T - 1 } ,
$$

and similarly, $\| P B _ { T } ^ { 2 T } P \| _ { \mathrm { o p } } \lesssim q _ { * } ^ { 2 T }$ . Thus we have

$$
\frac { \| P g _ { F , T } \| } { T q _ { * } ^ { 2 T - 2 } }  0 , \qquad \frac { \| P g _ { L , T } \| } { T q _ { * } ^ { 2 T - 1 } }  0 .
$$

## G.5 Score linearization and weak convergence of the optimal learning rate

Lemma G.5 (Score linearization with an explicit remainder). Define

$$
\begin{array} { r l } & { S _ { 0 , n } : = K B _ { T } ^ { 2 T - 1 } , \qquad S _ { 1 , n } : = E _ { n } B _ { T } ^ { 2 T - 1 } + K \mathcal { L } _ { 2 T - 1 , n } , } \\ & { S _ { 2 , n } : = K \mathcal { R } _ { 2 T - 1 , n } + E _ { n } \mathcal { L } _ { 2 T - 1 , n } + E _ { n } \mathcal { R } _ { 2 T - 1 , n } , } \end{array}
$$

and

$$
\begin{array} { r } { R _ { F , n , T } : = - 2 f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( S _ { 1 , n } + S _ { 2 , n } ) y + f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( S _ { 0 , n } + S _ { 1 , n } + S _ { 2 , n } ) f _ { n } ^ { ( 0 ) } ( X ) + y ^ { \top } S _ { 2 , n } y . } \end{array}
$$

Then we have

$$
F _ { n , T } ( \eta _ { \infty , T } ) - F _ { \infty , T } ( \eta _ { \infty , T } ) = ( g _ { F , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { F , T } , E _ { n } \rangle _ { \mathrm { F } } + R _ { F , n , T }
$$

with

$$
| R _ { F , n , T } | = O _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 3 } } { n } \right) ,
$$

which consequently implies

$$
\frac { \sqrt { n } R _ { F , n , T } } { T q _ { * } ^ { 2 T - 2 } } = o _ { p } ( 1 ) .
$$

Proof. By Lemma C.2, we have $P E _ { n } = E _ { n } P = E _ { n }$ almost surely for all suficiently large n. Hence $P$ also commutes with $\Delta _ { n } , B _ { T } , { \mathcal { L } } _ { k , n } ,$ and $\mathcal { R } _ { k , n }$ almost surely, and all three matrices $S _ { 0 , n } , S _ { 1 , n } , S _ { 2 , n }$ vanish on ker $( K )$ . In addition, $A _ { n } B _ { n } ( { \ ' } \eta _ { \infty , T } ) ^ { 2 T - 1 }$ and $S _ { 0 , n }$ are symmetric. $S _ { 1 , n }$ is the Fr´echet derivative at $K$ , in the symmetric direction $E _ { n }$ , of the map

$$
\mathcal { F } ( A ) : = A \left( I _ { m } - \frac { \eta _ { \infty , T } } { m } A \right) ^ { 2 T - 1 } .
$$

Since $\mathcal { F } ( A )$ is symmetric whenever A is symmetric, and $K + t E _ { n }$ is symmetric for every $t \in \mathbb { R }$ the diference quotient $( { \mathcal { F } } ( K + t E _ { n } ) - { \mathcal { F } } ( K ) ) / t$ is symmetric. Passing to the limit $t  0$ therefore shows that $S _ { 1 , n } = D \mathcal { F } ( K ) [ E _ { n } ]$ is symmetric. Consequently, $S _ { 2 , n }$ is symmetric as well.

Note that we have

$$
A _ { n } B _ { n } ( \eta _ { \infty , T } ) ^ { 2 T - 1 } = S _ { 0 , n } + S _ { 1 , n } + S _ { 2 , n } .
$$

Thus, using $\chi _ { n } ^ { ( 0 ) } = - y + f _ { n } ^ { ( 0 ) } ( X )$ and $F _ { \infty , T } ( \eta _ { \infty , T } ) = y ^ { \top } S _ { 0 , n } y$ , we obtain

$$
\begin{array} { r l } & { F _ { n , T } ( \eta _ { \infty , T } ) - F _ { \infty , T } ( \eta _ { \infty , T } ) } \\ & { \mathrm { ~ \ } = - 2 f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } S _ { 0 , n } y + y ^ { \top } S _ { 1 , n } y - 2 f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( S _ { 1 , n } + S _ { 2 , n } ) y } \\ & { \qquad + f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( S _ { 0 , n } + S _ { 1 , n } + S _ { 2 , n } ) f _ { n } ^ { ( 0 ) } ( X ) + y ^ { \top } S _ { 2 , n } y } \\ & { \mathrm { ~ \ } = ( g _ { F , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + y ^ { \top } S _ { 1 , n } y + R _ { F , n , T } . } \end{array}
$$

Since the definitions of $\mathcal { L } _ { 2 T - 1 , n }$ and $\Delta _ { n }$ give

$$
y ^ { \top } S _ { 1 , n } y = y ^ { \top } E _ { n } B _ { T } ^ { 2 T - 1 } y - \frac { \eta _ { \infty , T } } { m } \sum _ { s = 0 } ^ { 2 T - 2 } y ^ { \top } K B _ { T } ^ { s } E _ { n } B _ { T } ^ { 2 T - 2 - s } y = \langle Q _ { F , T } , E _ { n } \rangle _ { \mathrm { F } } ,
$$

we have

$$
F _ { n , T } ( \eta _ { \infty , T } ) - F _ { \infty , T } ( \eta _ { \infty , T } ) = ( g _ { F , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { F , T } , E _ { n } \rangle _ { \mathrm { F } } + R _ { F , n , T } .
$$

We next bound the remainder. Lemma F.3 implies

$$
\| P S _ { 0 , n } P \| _ { \mathrm { o p } } = O _ { p } ( q _ { * } ^ { 2 T - 1 } ) .
$$

Lemmas F.3 and G.3 imply

$$
\| P S _ { 1 , n } P \| _ { \mathrm { o p } } = O _ { p } ( T q _ { * } ^ { 2 T - 2 } \| E _ { n } \| _ { \mathrm { o p } } ) .
$$

Lemma G.3 and $T \| E _ { n } \| _ { \mathrm { o p } } = o _ { p } ( 1 )$ imply

$$
\| P S _ { 2 , n } P \| _ { \mathrm { o p } } = O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 3 } \| E _ { n } \| _ { \mathrm { o p } } ^ { 2 } ) .
$$

Thus we have

$$
\| P S _ { 1 , n } P \| _ { \mathrm { o p } } + \| P S _ { 2 , n } P \| _ { \mathrm { o p } } = O _ { p } ( T q _ { * } ^ { 2 T - 2 } \| E _ { n } \| _ { \mathrm { o p } } )
$$

and

$$
\| P S _ { 0 , n } P \| _ { \mathrm { o p } } + \| P S _ { 1 , n } P \| _ { \mathrm { o p } } + \| P S _ { 2 , n } P \| _ { \mathrm { o p } } = O _ { p } ( q _ { * } ^ { 2 T - 1 } ) .
$$

By (32), we have $f _ { n } ^ { ( 0 ) } ( X ) \stackrel { \mathrm { a . s . } } { = } P f _ { n } ^ { ( 0 ) } ( X )$ and $S _ { j , n } \overset { \mathrm { a . s . } } { = } P S _ { j , n } P$ for $j = 0 , 1 , 2$ . Hence, the definition of $R _ { F , n , T }$ and the preceding bounds give

$$
\begin{array} { r l } & { | R _ { F , n , T } | } \\ & { \leq | 2 f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( S _ { 1 , n } + S _ { 2 , n } ) y | + | f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( S _ { 0 , n } + S _ { 1 , n } + S _ { 2 , n } ) f _ { n } ^ { ( 0 ) } ( X ) | + | y ^ { \top } S _ { 2 , n } y | } \\ & { = O _ { p } \left( \| f _ { n } ^ { ( 0 ) } ( X ) \| ( \| P S _ { 1 , n } P \| _ { \mathrm { o p } } + \| P S _ { 2 , n } P \| _ { \mathrm { o p } } ) + \| f _ { n } ^ { ( 0 ) } ( X ) \| ^ { 2 } \sum _ { j = 0 } ^ { 2 } \| P S _ { j , n } P \| _ { \mathrm { o p } } + \| P S _ { 2 , n } P \| _ { \mathrm { o p } } \right) } \\ & { = O _ { p } ( \| f _ { n } ^ { ( 0 ) } ( X ) \| T q _ { * } ^ { 2 T - 2 } \| E _ { n } \| _ { \mathrm { o p } } + \| f _ { n } ^ { ( 0 ) } ( X ) \| ^ { 2 } q _ { * } ^ { 2 T - 1 } + T ^ { 2 } q _ { * } ^ { 2 T - 3 } \| E _ { n } \| _ { \mathrm { o p } } ^ { 2 } ) } \\ & { = O _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 3 } } { n } \right) , } \end{array}
$$

where the last equality follows from $\| f _ { n } ^ { ( 0 ) } ( X ) \| = O _ { p } ( n ^ { - 1 / 2 } )$ and $\| E _ { n } \| _ { \mathrm { o p } } = O _ { p } ( n ^ { - 1 / 2 } )$ by Lemma 3.3. Consequently, we have

$$
\frac { \sqrt { n } | R _ { F , n , T } | } { T q _ { * } ^ { 2 T - 2 } } = O _ { p } \left( \frac { T } { q _ { * } \sqrt { n } } \right) = o _ { p } ( 1 ) .
$$

Theorem G.6 (Limit distribution of the optimal learning rate at growing T). Under the assumptions of Theorem $\it 3 . 4$ , the finite-width optimizer is interior with probability tending to one and we have

$$
\sqrt { n } ( \eta _ { n , T } - \eta _ { \infty , T } ) \stackrel { d } { \longrightarrow } \Omega _ { \eta , * } : = - \frac { 2 m } { ( \lambda _ { 1 } + \lambda _ { r } ) ^ { 2 } } ( u _ { 1 } ^ { \top } \Xi u _ { 1 } + u _ { r } ^ { \top } \Xi u _ { r } ) ,
$$

where $\Xi$ is defined in Lemma 3.3.

Proof. Note that by Corollary C.4, we have $F _ { \infty , T } ( \eta _ { \infty , T } ) = 0$ . Thus, Lemma G.5 implies

$$
\frac { \sqrt { n } F _ { n , T } ( \eta _ { \infty , T } ) } { T q _ { * } ^ { 2 T - 2 } } = \frac { \sqrt { n } [ ( g _ { F , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { F , T } , E _ { n } \rangle _ { \mathrm { F } } ] } { T q _ { * } ^ { 2 T - 2 } } + o _ { p } ( 1 ) .
$$

By (32) and Lemmas G.4 and 3.3, we have

$$
\frac { | \sqrt { n } ( g _ { F , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) | } { T q _ { * } ^ { 2 T - 2 } } | \stackrel { \mathrm { a . s . } } { = } | \frac { \sqrt { n } ( P g _ { F , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) } { T q _ { * } ^ { 2 T - 2 } } | \leq \frac { \| P g _ { F , T } \| } { T q _ { * } ^ { 2 T - 2 } } \| \sqrt { n } f _ { n } ^ { ( 0 ) } ( X ) \| = o _ { p } ( 1 ) .
$$

Moreover, since $E _ { n } = P E _ { n } P$ by (32), we have by Lemmas G.4 and 3.3 that

$$
\frac { \sqrt { n } \langle Q _ { F , T } , E _ { n } \rangle _ { \mathrm { F } } } { T q _ { * } ^ { 2 T - 2 } } = \left. \frac { P Q _ { F , T } P } { T q _ { * } ^ { 2 T - 2 } } , \sqrt { n } E _ { n } \right. _ { \mathrm { F } } \overset { d } { \longrightarrow } \langle Q _ { F , * } , \Xi \rangle _ { \mathrm { F } } = - \frac { 2 \eta _ { * } h _ { * } } { m } ( u _ { 1 } ^ { \top } \Xi u _ { 1 } + u _ { r } ^ { \top } \Xi u _ { r } ) .
$$

Therefore we have

$$
\frac { \sqrt { n } F _ { n , T } ( \eta _ { \infty , T } ) } { T q _ { * } ^ { 2 T - 2 } } = \frac { \sqrt { n } \langle Q _ { F , T } , E _ { n } \rangle _ { \mathrm { F } } } { T q _ { * } ^ { 2 T - 2 } } + o _ { p } ( 1 ) \overset { d } { \longrightarrow } - \frac { 2 \eta _ { * } h _ { * } } { m } ( u _ { 1 } ^ { \top } \Xi u _ { 1 } + u _ { r } ^ { \top } \Xi u _ { r } ) .\tag{40}
$$

By Proposition C.3, we can write

$$
F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) = - \frac { 2 T - 1 } { m } \sum _ { j = 1 } ^ { r } \lambda _ { j } ^ { 2 } \| P _ { j } y \| ^ { 2 } r _ { j , T } ^ { 2 T - 2 } .
$$

Thus, by Lemma E.3, we have

$$
\frac { F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) } { T q _ { * } ^ { 2 T - 2 } } = - \frac { 2 T - 1 } { m T } \sum _ { j = 1 } ^ { r } \lambda _ { j } ^ { 2 } \| P _ { j } y \| ^ { 2 } \frac { r _ { j , T } ^ { 2 T - 2 } } { q _ { * } ^ { 2 T - 2 } }  - \frac { 2 } { m } ( \lambda _ { 1 } ^ { 2 } \| P _ { 1 } y \| ^ { 2 } a _ { 1 } ^ { * } + \lambda _ { r } ^ { 2 } \| P _ { r } y \| ^ { 2 } a _ { r } ^ { * } ) .
$$

By Assumption 3.2, we have $\| \cal { P } _ { 1 } y \| ^ { 2 } = c _ { 1 }$ and $\| \boldsymbol { P } _ { r } \boldsymbol { y } \| ^ { 2 } = c _ { r }$ . Therefore we have

$$
\frac { F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) } { T q _ { * } ^ { 2 T - 2 } }  - \frac { 2 } { m } ( c _ { 1 } \lambda _ { 1 } ^ { 2 } a _ { 1 } ^ { * } + c _ { r } \lambda _ { r } ^ { 2 } a _ { r } ^ { * } ) = - \frac { 2 h _ { * } } { m } ( \lambda _ { 1 } + \lambda _ { r } ) < 0 .\tag{41}
$$

By the mean value theorem, there exists a random $\tilde { \eta } _ { n , T }$ between $\eta _ { n , T }$ and $\eta _ { \infty , T }$ such that

$$
F _ { n , T } ( \eta _ { n , T } ) - F _ { n , T } ( \eta _ { \infty , T } ) = F _ { n , T } ^ { \prime } ( { \tilde { \eta } } _ { n , T } ) ( \eta _ { n , T } - \eta _ { \infty , T } ) .
$$

Since $F _ { n , T } ( \eta _ { n , T } ) = 0$ on an event with probability tending to one, we have on the same event that

$$
- F _ { n , T } ( \eta _ { \infty , T } ) = F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) ( \eta _ { n , T } - \eta _ { \infty , T } ) .\tag{42}
$$

By Lemma G.2, we have $\tilde { \eta } _ { n , T } - \eta _ { \infty , T } = O _ { p } ( n ^ { - 1 / 2 } )$ . Hence (35) and (36) in the proof of Lemma G.2 imply

$$
F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) - F _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) = o _ { p } ( T q _ { * } ^ { 2 T - 2 } ) .\tag{43}
$$

By (41), this further yields

$$
\frac { F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) } { T q _ { * } ^ { 2 T - 2 } } = - \frac { 2 h _ { * } } { m } ( \lambda _ { 1 } + \lambda _ { r } ) + o _ { p } ( 1 ) ,
$$

thus $F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } )$ is negative with probability tending to one. Combining this with (42), we obtain, on an event with probability tending to one,

$$
\sqrt { n } ( \eta _ { n , T } - \eta _ { \infty , T } ) = - \frac { \sqrt { n } F _ { n , T } ( \eta _ { \infty , T } ) / T q _ { * } ^ { 2 T - 2 } } { F _ { n , T } ^ { \prime } ( \tilde { \eta } _ { n , T } ) / T q _ { * } ^ { 2 T - 2 } } .
$$

Therefore, (40), (41), and (43) yield

$$
\sqrt { n } \big ( \eta _ { n , T } - \eta _ { \infty , T } \big ) \stackrel { d } { \longrightarrow } - \frac { \eta _ { * } } { \lambda _ { 1 } + \lambda _ { r } } \big ( u _ { 1 } ^ { \top } \Xi u _ { 1 } + u _ { r } ^ { \top } \Xi u _ { r } \big ) = - \frac { 2 m } { \big ( \lambda _ { 1 } + \lambda _ { r } \big ) ^ { 2 } } \big ( u _ { 1 } ^ { \top } \Xi u _ { 1 } + u _ { r } ^ { \top } \Xi u _ { r } \big ) .
$$

## G.6 Fluctuations and weak convergence of the optimal loss

Lemma G.7 (Loss linearization with a quadratic remainder). Define

$$
R _ { L , n , T } : = \frac { 1 } { 2 m } [ y ^ { \top } \mathcal { R } _ { 2 T , n } y - 2 f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( \mathcal { L } _ { 2 T , n } + \mathcal { R } _ { 2 T , n } ) y + f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } B _ { n } ( \eta _ { \infty , T } ) ^ { 2 T } f _ { n } ^ { ( 0 ) } ( X ) ] .
$$

Then we have

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = ( g _ { L , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { L , T } , E _ { n } \rangle _ { \mathrm { F } } + R _ { L , n , T } .
$$

Thus

$$
| R _ { L , n , T } | = O _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 2 } } { n } \right) ,
$$

which implies

$$
\frac { \sqrt { n } R _ { L , n , T } } { T q _ { * } ^ { 2 T - 1 } } = o _ { p } ( 1 ) .
$$

Proof. Since $B _ { n } ( \eta _ { \infty , T } ) ^ { 2 T } = B _ { T } ^ { 2 T } + \mathcal { L } _ { 2 T , n } + \mathcal { R } _ { 2 T , n }$ , expanding $\chi _ { n } ^ { ( 0 ) } = - y + f _ { n } ^ { ( 0 ) } ( X )$ in the quadratic loss gives

$$
\begin{array} { l } { \displaystyle \phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = \frac { 1 } { 2 m } ( \chi _ { n } ^ { ( 0 ) } ) ^ { \top } B _ { n } ( \eta _ { \infty , T } ) ^ { 2 T } \chi _ { n } ^ { ( 0 ) } - \frac { 1 } { 2 m } y ^ { \top } B _ { T } ^ { 2 T } y } \\ { \displaystyle \qquad = - \frac { 1 } { m } f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } B _ { T } ^ { 2 T } y + \frac { 1 } { 2 m } y ^ { \top } \mathcal { L } _ { 2 T , n } y + R _ { L , n , T } } \\ { \displaystyle \qquad = \left( g _ { L , T } \right) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \frac { 1 } { 2 m } y ^ { \top } \mathcal { L } _ { 2 T , n } y + R _ { L , n , T } . } \end{array}
$$

By the definition of $\mathcal { L } _ { 2 T , n }$ , we have

$$
\frac { 1 } { 2 m } y ^ { \top } \mathcal { L } _ { 2 T , n } y = - \frac { \eta _ { \infty , T } } { 2 m ^ { 2 } } \sum _ { s = 0 } ^ { 2 T - 1 } y ^ { \top } B _ { T } ^ { s } E _ { n } B _ { T } ^ { 2 T - 1 - s } y = \langle Q _ { L , T } , E _ { n } \rangle _ { \mathrm { F } } ,
$$

which proves

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = ( g _ { L , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { L , T } , E _ { n } \rangle _ { \mathrm { F } } + R _ { L , n , T } .
$$

By Lemma G.3, we have

$$
\| P \mathcal { L } _ { 2 T , n } P \| _ { \mathrm { o p } } = O _ { p } ( T q _ { * } ^ { 2 T - 1 } \| E _ { n } \| _ { \mathrm { o p } } ) , \qquad \| P \mathcal { R } _ { 2 T , n } P \| _ { \mathrm { o p } } = O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } \| E _ { n } \| _ { \mathrm { o p } } ^ { 2 } ) .
$$

By Lemma F.3, we also have

$$
\| P B _ { n } ( \eta _ { \infty , T } ) ^ { 2 T } P \| _ { \mathrm { o p } } = O _ { p } ( q _ { * } ^ { 2 T } ) .
$$

By (32), we have

$$
f _ { n } ^ { ( 0 ) } ( X ) \stackrel { \mathrm { a . s . } } { = } P f _ { n } ^ { ( 0 ) } ( X ) , \qquad \mathcal { L } _ { 2 T , n } \stackrel { \mathrm { a . s . } } { = } P \mathcal { L } _ { 2 T , n } P , \qquad \mathcal { R } _ { 2 T , n } \stackrel { \mathrm { a . s . } } { = } P \mathcal { R } _ { 2 T , n } P .
$$

Lemma 3.3 then gives

$$
\begin{array} { r l } & { | R _ { L , n , T } | \lesssim | y ^ { \top } \mathcal { R } _ { 2 T , n } y | + | 2 f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } ( \mathcal { L } _ { 2 T , n } + \mathcal { R } _ { 2 T , n } ) y | + | f _ { n } ^ { ( 0 ) } ( X ) ^ { \top } B _ { n } ( \eta _ { \infty , T } ) ^ { 2 T } f _ { n } ^ { ( 0 ) } ( X ) | } \\ & { \qquad = O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } \| E _ { n } \| _ { \mathrm { o p } } ^ { 2 } + T q _ { * } ^ { 2 T - 1 } \| E _ { n } \| _ { \mathrm { o p } } \| f _ { n } ^ { ( 0 ) } ( X ) \| + q _ { * } ^ { 2 T } \| f _ { n } ^ { ( 0 ) } ( X ) \| ^ { 2 } ) } \\ & { \qquad = O _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 2 } } { n } \right) , } \end{array}
$$

and

$$
\frac { \sqrt { n } | R _ { L , n , T } | } { T q _ { * } ^ { 2 T - 1 } } = O _ { p } \left( \frac { T } { q _ { * } \sqrt { n } } \right) = o _ { p } ( 1 ) .
$$

Theorem G.8 (Limit distribution of the optimal loss at growing T). Under the assumptions of Theorem 3.4, we have

$$
\frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } ( \phi _ { n , T } ( \eta _ { n , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) ) \overset { d } { \longrightarrow } \Omega _ { L , * } = \langle Q _ { L , * } , \Xi \rangle _ { \mathrm { F } } .
$$

Proof. Lemma G.7 implies

$$
\frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } \bigl ( \phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) \bigr ) = \frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } \bigl [ \bigl ( g _ { L , T } \bigr ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) + \langle Q _ { L , T } , E _ { n } \rangle _ { \mathrm { F } } \bigr ] + o _ { p } ( 1 ) .
$$

By (32) and Lemmas G.4 and 3.3, we have

$$
\left| \frac { \sqrt { n } ( g _ { L , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) } { T q _ { * } ^ { 2 T - 1 } } \right| \overset { \mathrm { a . s . } } { = } \left| \frac { \sqrt { n } ( P g _ { L , T } ) ^ { \top } f _ { n } ^ { ( 0 ) } ( X ) } { T q _ { * } ^ { 2 T - 1 } } \right| \leq \frac { \| P g _ { L , T } \| } { T q _ { * } ^ { 2 T - 1 } } \| \sqrt { n } f _ { n } ^ { ( 0 ) } ( X ) \| = o _ { p } ( 1 ) .
$$

Moreover, since $E _ { n } = P E _ { n } P$ by (32), we have by Lemma G.4 that

$$
\frac { \sqrt { n } \langle Q _ { L , T } , E _ { n } \rangle _ { \mathrm { F } } } { T q _ { * } ^ { 2 T - 1 } } = \left. \frac { P Q _ { L , T } P } { T q _ { * } ^ { 2 T - 1 } } , \sqrt { n } E _ { n } \right. _ { \mathrm { F } } \xrightarrow [ ] d \langle Q _ { L , * } , \Xi \rangle _ { \mathrm { F } } .
$$

Thus we have

$$
\frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } ( \phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) ) \stackrel { d } { \longrightarrow } \langle Q _ { L , * } , \Xi \rangle _ { \mathrm { F } } .\tag{44}
$$

It remains to replace $\eta _ { \infty , T }$ by $\eta _ { n , T }$ in the finite-width loss. By Taylor’s theorem, there exists a random $\bar { \eta } _ { n , T }$ between $\eta _ { \infty , T }$ and $\eta _ { n , T }$ such that

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { n , T } ( \eta _ { n , T } ) = \phi _ { n , T } ^ { \prime } ( \eta _ { n , T } ) ( \eta _ { \infty , T } - \eta _ { n , T } ) + \frac { 1 } { 2 } \phi _ { n , T } ^ { \prime \prime } ( \bar { \eta } _ { n , T } ) ( \eta _ { \infty , T } - \eta _ { n , T } ) ^ { 2 } .
$$

By Lemma G.2, with probability tending to one, we have $\eta _ { n , T } \in \left( \eta _ { \ell } , \eta _ { u } \right)$ and $F _ { n , T } ( \eta _ { n , T } ) = 0$ Hence, on the same event, Proposition C.3 gives $\phi _ { n , T } ^ { \prime } ( \eta _ { n , T } ) = 0$ . Therefore, with probability tending to one, we have

$$
\phi _ { n , T } ( \eta _ { \infty , T } ) - \phi _ { n , T } ( \eta _ { n , T } ) = \frac { 1 } { 2 } \phi _ { n , T } ^ { \prime \prime } ( \bar { \eta } _ { n , T } ) ( \eta _ { \infty , T } - \eta _ { n , T } ) ^ { 2 } .\tag{45}
$$

By Lemma G.2, we have $| \bar { \eta } _ { n , T } - \eta _ { \infty , T } | = o _ { p } ( T ^ { - 1 } )$ . Moreover, (10) and Lemma C.2 yield

$$
\begin{array} { r l } & { \underset { | \eta - \eta _ { \infty , T } | \leq C / T } { \operatorname* { s u p } } | \phi _ { n , T } ^ { \prime \prime } ( \eta ) | = \frac { T ( 2 T - 1 ) } { m ^ { 3 } } \underset { | \eta - \eta _ { \infty , T } | \leq C / T } { \operatorname* { s u p } } | ( \chi _ { n } ^ { ( 0 ) } ) ^ { \top } A _ { n } ^ { 2 } B _ { n } ( \eta ) ^ { 2 T - 2 } \chi _ { n } ^ { ( 0 ) } | } \\ & { \stackrel { \mathrm { a . s . } } { = } \frac { T ( 2 T - 1 ) } { m ^ { 3 } } \underset { | \eta - \eta _ { \infty , T } | \leq C / T } { \operatorname* { s u p } } | ( P \chi _ { n } ^ { ( 0 ) } ) ^ { \top } ( P A _ { n } ^ { 2 } P ) ( P B _ { n } ( \eta ) ^ { 2 T - 2 } P ) ( P \chi _ { n } ^ { ( 0 ) } ) | } \\ & { \leq \frac { T ( 2 T - 1 ) } { m ^ { 3 } } \| P \chi _ { n } ^ { ( 0 ) } \| ^ { 2 } \| A _ { n } \| _ { \infty } ^ { 2 } \underset { | \eta - \eta _ { \infty , T } | \leq C / T } { \operatorname* { s u p } } \| P B _ { n } ( \eta ) ^ { 2 T - 2 } P \| _ { \mathrm { o p } } . } \end{array}
$$

Here, $\| P \chi _ { n } ^ { ( 0 ) } \| = O _ { p } ( 1 )$ and $\| A _ { n } \| _ { \mathrm { o p } } = O _ { p } ( 1 )$ hold by Lemma 3.3, while Lemma F.3 implies $\begin{array} { r } { \operatorname* { s u p } _ { | \eta - \eta _ { \infty , T } | \leq C / T } \| P B _ { n } ( \eta ) ^ { 2 T - 2 } P \| _ { \mathrm { o p } } = O _ { p } ( q _ { * } ^ { 2 T - 2 } ) } \end{array}$ . Thus we have

$$
\operatorname* { s u p } _ { | \eta - \eta _ { \infty , T } | \leq C / T } | \phi _ { n , T } ^ { \prime \prime } ( \eta ) | = O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } ) ,
$$

which implies $\phi _ { n , T } ^ { \prime \prime } ( \bar { \eta } _ { n , T } ) = O _ { p } ( T ^ { 2 } q _ { * } ^ { 2 T - 2 } )$ . Equation (45) and Lemma G.2 imply that

$$
\frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } ( \phi _ { n , T } ( \eta _ { n , T } ) - \phi _ { n , T } ( \eta _ { \infty , T } ) ) = O _ { p } \left( \frac { T } { q _ { * } \sqrt { n } } \right) = o _ { p } ( 1 ) .
$$

Therefore, by (44) and Slutsky’s theorem, we obtain

$$
\frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } ( \phi _ { n , T } ( \eta _ { n , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) ) \stackrel { d } { \longrightarrow } \langle Q _ { L , * } , \Xi \rangle _ { \mathrm { F } } .
$$

## G.7 Convergence rates of the optimal loss and the optimal learning rate

Lemma G.9 (Quadratic expansion of the transfer gap). Under the assumptions of Theorem ${ 3 . 4 } ,$ we have

$$
c _ { n , T } = \frac { \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) } { 2 } ( \eta _ { n , T } - \eta _ { \infty , T } ) ^ { 2 } ( 1 + o _ { p } ( 1 ) ) .
$$

Proof. Recall that $\eta _ { \infty , T } \ \in \ \left( \eta _ { \ell } , \eta _ { u } \right)$ for suficiently large T. Thus Corollary C.4 implies $\phi _ { \infty , T } ^ { \prime } ( \eta _ { \infty , T } ) = 0$ . By Taylor’s theorem, there exists a random $\eta _ { n , T } ^ { \dag }$ between $\eta _ { n , T }$ and $\eta _ { \infty , T }$ such that

$$
c _ { n , T } = \phi _ { \infty , T } ( \eta _ { n , T } ) - \phi _ { \infty , T } ( \eta _ { \infty , T } ) = \frac { \dot { \phi } _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) } { 2 } \big ( \eta _ { n , T } - \eta _ { \infty , T } \big ) ^ { 2 } + \frac { 1 } { 6 } \dot { \phi } _ { \infty , T } ^ { \prime \prime \prime } ( \eta _ { n , T } ^ { \dag } ) \big ( \eta _ { n , T } - \eta _ { \infty , T } \big ) ^ { 3 } \int _ { 0 } ^ { \infty } \frac { \ d \Theta } { \ d t } \left( \frac { \ d \Theta } { \ d t } - \frac { \ d \eta _ { \infty , T } } { \ d t } \right) d \eta _ { \infty , T }
$$

Diferentiating (10) gives

$$
\phi _ { \infty , T } ^ { \prime \prime \prime } ( \eta ) = - \frac { T ( 2 T - 1 ) ( 2 T - 2 ) } { m ^ { 4 } } y ^ { \top } K ^ { 3 } B _ { \infty } ( \eta ) ^ { 2 T - 3 } y .
$$

Lemma G.2 gives $| \eta _ { n , T } ^ { \dag } - \eta _ { \infty , T } | = O _ { p } ( n ^ { - 1 / 2 } ) = o _ { p } ( T ^ { - 1 } )$ . Therefore, with probability tending to one, $\eta _ { n , T } ^ { \dag }$ belongs to the $C / T .$ -neighborhood of $\eta _ { \infty , T }$ in Lemma F.3. Thus, Lemma F.3 yields

$$
| \phi _ { \infty , T } ^ { \prime \prime \prime } ( \eta _ { n , T } ^ { \dag } ) | = O _ { p } ( T ^ { 3 } q _ { * } ^ { 2 T - 3 } ) .
$$

Together with (23) and Lemma G.2, we have

$$
\left| \frac { \phi _ { \infty , T } ^ { \prime \prime \prime } ( \eta _ { n , T } ^ { \dagger } ) ( \eta _ { n , T } - \eta _ { \infty , T } ) / 6 } { \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) / 2 } \right| = O _ { p } \left( \frac { T ^ { 3 } q _ { * } ^ { 2 T - 3 } } { T ^ { 2 } q _ { * } ^ { 2 T - 2 } \sqrt { n } } \right) = O _ { p } \left( \frac { T } { q _ { * } \sqrt { n } } \right) = o _ { p } ( 1 ) ,
$$

which implies

$$
c _ { n , T } = \frac { \phi _ { \infty , T } ^ { \prime \prime } ( \eta _ { \infty , T } ) } { 2 } ( \eta _ { n , T } - \eta _ { \infty , T } ) ^ { 2 } ( 1 + o _ { p } ( 1 ) ) .
$$

Theorem G.10 (Rates of $a _ { n , T } , b _ { n , T }$ , and $c _ { n , T } )$ . Under the assumptions of Theorem $\ 3 . 4 ,$ we have

$$
a _ { n , T } = \Theta _ { p } \left( \frac { T q _ { * } ^ { 2 T - 1 } } { \sqrt { n } } \right) , \qquad b _ { n , T } = \Theta _ { p } ( n ^ { - 1 / 2 } ) , \qquad c _ { n , T } = \Theta _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 2 } } { n } \right) .
$$

Proof. We first show that the limit distribution of Theorem G.6 is nondegenerate. Recall from Lemma 3.3 that $\Xi = \alpha K + X G X ^ { \top }$ , where $\alpha \sim \mathcal { N } ( 0 , 2 )$ is independent of G. The contribution of $\alpha K$ to $\Omega _ { \eta , * }$ is

$$
- \frac { 2 m } { ( \lambda _ { 1 } + \lambda _ { r } ) ^ { 2 } } \alpha ( \lambda _ { 1 } + \lambda _ { r } ) = - \frac { 2 m } { \lambda _ { 1 } + \lambda _ { r } } \alpha = - \eta _ { * } \alpha \sim \mathcal N ( 0 , 2 \eta _ { * } ^ { 2 } ) ,
$$

which has strictly positive variance and is independent of the remaining Gaussian term. Hence $\Omega _ { \eta , * }$ <sub>∗</sub> is nondegenerate, and we have

$$
b _ { n , T } = \Theta _ { p } ( n ^ { - 1 / 2 } ) .\tag{46}
$$

We next show the nondegeneracy of the limit distribution of Theorem G.8. Since $c _ { r } \lambda _ { r } a _ { r } ^ { * } / ( c _ { 1 } \lambda _ { 1 } a _ { 1 } ^ { * } ) = c _ { r } \lambda _ { r } e ^ { L _ { * } } / ( c _ { 1 } \lambda _ { 1 } ) = 1$ , we have

$$
h _ { * } : = c _ { 1 } \lambda _ { 1 } a _ { 1 } ^ { * } = c _ { r } \lambda _ { r } a _ { r } ^ { * } > 0 .\tag{47}
$$

This implies

$$
\langle Q _ { L , * } , K \rangle _ { \mathrm { F } } = \frac { \eta _ { * } } { m ^ { 2 } } ( c _ { 1 } a _ { 1 } ^ { * } \lambda _ { 1 } - c _ { r } a _ { r } ^ { * } \lambda _ { r } ) = 0 .
$$

Thus we can write

$$
\Omega _ { L , * } = \langle Q _ { L , * } , X G X ^ { \top } \rangle _ { \mathrm { F } } = \langle X ^ { \top } Q _ { L , * } X , G \rangle _ { \mathrm { F } } .
$$

Recall from Lemma 3.3 that $\mathbb { E } ( G _ { r s } G _ { t u } ) \ : = \ : d ^ { - 2 } ( \delta _ { r t } \delta _ { s u } + \delta _ { r u } \delta _ { s t } )$ . Thus for a deterministic symmetric matrix S, we have

$$
\mathrm { V a r } ( \langle S , G \rangle _ { \mathrm { F } } ) = \sum _ { r , s , t , u } S _ { r s } S _ { t u } \mathbb { E } ( G _ { r s } G _ { t u } ) = 2 d ^ { - 2 } \| S \| _ { \mathrm { F } } ^ { 2 } .
$$

Therefore we have

$$
\begin{array} { r l } & { \mathrm { V a r } ( \Omega _ { L , * } ) = 2 d ^ { - 2 } \| X ^ { \top } Q _ { L , * } X \| _ { \mathrm { F } } ^ { 2 } = 2 \mathrm { t r } ( Q _ { L , * } K Q _ { L , * } K ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}
$$

Hence $\Omega _ { L , * }$ is a nondegenerate centered Gaussian random variable. Theorem G.8 therefore yields

$$
a _ { n , T } = \Theta _ { p } \left( \frac { T q _ { * } ^ { 2 T - 1 } } { \sqrt { n } } \right) .
$$

Finally, Lemma G.9, (23) and (46) imply

$$
c _ { n , T } = \Theta _ { p } \left( \frac { T ^ { 2 } q _ { * } ^ { 2 T - 2 } } { n } \right) .
$$

Proof of Corollary 3.5. By Theorems G.8, G.10, and the continuous mapping theorem, we have

$$
\frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } a _ { n , T } \xrightarrow { d } | \Omega _ { L , * } | ,
$$

where $\Omega _ { L , \ast }$ <sub>∗</sub> is a nondegenerate Gaussian random variable. Since $\mathrm { P } ( | \Omega _ { L , * } | = 0 ) = 0$ , applying the continuous mapping theorem once more gives

$$
\frac { T q _ { * } ^ { 2 T - 1 } } { \sqrt { n } a _ { n , T } } \xrightarrow { d } \frac { 1 } { | \Omega _ { L , * } | } ,
$$

and consequently,

$$
\frac { 1 } { a _ { n , T } } = O _ { p } \left( \frac { \sqrt { n } } { T q _ { * } ^ { 2 T - 1 } } \right) .
$$

Together with $c _ { n , T } = \Theta _ { p } \left( T ^ { 2 } q _ { * } ^ { 2 T - 2 } / n \right)$ , this yields

$$
\frac { c _ { n , T } } { a _ { n , T } } = \Theta _ { p } \left( \frac { T } { q _ { * } \sqrt { n } } \right) \xrightarrow [ ] { p } \Theta ,
$$

where the last convergence follows from $T / \sqrt { n }  0$ and the fact that $q _ { * } > 0$ is fixed. □

## H Counterexample to fast transfer

Proof of Proposition 3.8. Since $K = d ^ { - 1 } I _ { m }$ , Corollary C.1 gives

$$
\phi _ { \infty , T } ( \eta ) = \frac { \| y \| ^ { 2 } } { 2 m } \left( 1 - \frac { \eta } { \eta _ { 0 } } \right) ^ { 2 T } .\tag{48}
$$

Then we have

$$
\eta _ { \infty , T } = \eta _ { 0 } , \qquad \phi _ { \infty , T } ( \eta _ { \infty , T } ) = 0 .
$$

Define $\epsilon _ { n } : = n ^ { - 1 / 2 }$ and $M _ { n } : = { \sqrt { n } } ( d A _ { n } - I _ { m } )$ . Recall the Gaussian random variable α and the Gaussian matrix G from Lemma 3.3, and define $M = \alpha I _ { m } + d X G X ^ { \top }$ . Here, $\mathcal { G } : = d X G X ^ { \top }$ is a symmetric Gaussian matrix whose upper-triangular entries are independent and satisfy

$$
\begin{array} { r } { \mathcal { G } _ { i i } \sim \mathcal { N } ( 0 , 2 ) , \qquad \mathcal { G } _ { i j } \sim \mathcal { N } ( 0 , 1 ) \quad ( i < j ) . } \end{array}
$$

Then Lemma 3.3 and $X X ^ { \top } = I _ { m }$ imply

$$
( \chi _ { n } ^ { ( 0 ) } , M _ { n } ) \xrightarrow { d } ( - y , M ) .\tag{49}
$$

We first localize the finite-width optimizer without using positive curvature at the infinitewidth optimum. By Lemma C.2, we have ker $( A _ { n } ) \ { \stackrel { \mathrm { a . s . } } { = } } \ \ker ( K ) = \{ 0 \}$ . Since $A _ { n } \succeq 0$ , this implies $A _ { n } \succ 0$ almost surely. By Proposition C.5, we have $A _ { n } \chi _ { n } ^ { ( 0 ) } \neq 0$ almost surely, and consequently, $\chi _ { n } ^ { ( 0 ) } \neq 0$ almost surely. Suppose $\begin{array} { r } { A _ { n } = \sum _ { j } \lambda _ { j } ( A _ { n } ) v _ { n , j } v _ { n , j } ^ { \top } } \end{array}$ is the orthonormal eigendecomposition of $A _ { n }$ . Then the score is expressed as

$$
F _ { n } ( \eta ) = \sum _ { j } \lambda _ { j } ( A _ { n } ) \left( 1 - \frac { \eta } { m } \lambda _ { j } ( A _ { n } ) \right) ^ { 2 T - 1 } ( v _ { n , j } ^ { \top } \chi _ { n } ^ { ( 0 ) } ) ^ { 2 } .
$$

Substituting $\eta = m / \lambda _ { \mathrm { m a x } } ( A _ { n } )$ and $\eta = m / \lambda _ { \operatorname* { m i n } } ( A _ { n } )$ , we have

$$
F _ { n } \left( \frac { m } { \lambda _ { \operatorname* { m a x } } ( A _ { n } ) } \right) \geq 0 , \qquad F _ { n } \left( \frac { m } { \lambda _ { \operatorname* { m i n } } ( A _ { n } ) } \right) \leq 0 .
$$

Thus the score characterization in Corollary C.4 implies that the unconstrained optimizer $\hat { \eta } _ { n , T }$ satisfies

$$
\frac { m } { \lambda _ { \operatorname* { m a x } } ( A _ { n } ) } \leq \hat { \eta } _ { n , T } \leq \frac { m } { \lambda _ { \operatorname* { m i n } } ( A _ { n } ) } .
$$

Since $A _ { n } = d ^ { - 1 } ( I _ { m } + \epsilon _ { n } M _ { n } )$ , we have $\lambda _ { \operatorname* { m a x } } ( A _ { n } ) = d ^ { - 1 } [ 1 + \epsilon _ { n } \lambda _ { \operatorname* { m a x } } ( M _ { n } ) ]$ and $\lambda _ { \operatorname* { m i n } } ( A _ { n } ) =$ $d ^ { - 1 } [ 1 + \epsilon _ { n } \lambda _ { \operatorname* { m i n } } ( M _ { n } ) ]$ . Noting that $M _ { n } = O _ { p } ( 1 )$ , we have $m / \lambda _ { \mathrm { m a x } } ( A _ { n } ) = \eta _ { 0 } + O _ { p } ( \epsilon _ { n } )$ and $m / \lambda _ { \mathrm { m i n } } ( A _ { n } ) = \eta _ { 0 } + O _ { p } ( \epsilon _ { n } )$ . Therefore $\hat { \eta } _ { n , T }$ lies in $( \eta _ { \ell } , \eta _ { u } )$ with probability tending to one, and coincides with $\eta _ { n , T }$ . In particular, we have

$$
s _ { n , T } : = \epsilon _ { n } ^ { - 1 } \left( \frac { \eta _ { n , T } } { \eta _ { 0 } } - 1 \right) = O _ { p } ( 1 ) .\tag{50}
$$

For $s \in \mathbb { R }$ , the rescaled iteration matrix satisfies the exact identity

$$
B _ { n } ( \eta _ { 0 } ( 1 + \epsilon _ { n } s ) ) = - \epsilon _ { n } ( M _ { n } + s I _ { m } + \epsilon _ { n } s M _ { n } ) .
$$

Consequently, the rescaled finite-width objective is

$$
n ^ { T } \phi _ { n , T } ( \eta _ { 0 } ( 1 + \epsilon _ { n } s ) ) = \frac { 1 } { 2 m } ( \chi _ { n } ^ { ( 0 ) } ) ^ { \top } ( M _ { n } + s I _ { m } + \epsilon _ { n } s M _ { n } ) ^ { 2 T } \chi _ { n } ^ { ( 0 ) } .\tag{51}
$$

Define $\mathcal { H } _ { T } ( s ; D ) : = ( 2 m ) ^ { - 1 } y ^ { \top } ( D + s I _ { m } ) ^ { 2 T } y$ for symmetric matrix D. Since $T$ is fixed, the right-hand side of (51) is a polynomial in s whose coeficients converge jointly in distribution to those of $\mathcal { H } _ { T } ( s ; M )$ . Combining this with (49), for every $R < \infty$ , we have

$$
\Big ( M _ { n } , \chi _ { n } ^ { ( 0 ) } , n ^ { T } \phi _ { n , T } ( \eta _ { 0 } ( 1 + \epsilon _ { n } s ) ) \big | _ { [ - R , R ] } \Big ) \xrightarrow { d } \Big ( M , - y , \mathcal { H } _ { T } ( \cdot ; M ) \big | _ { [ - R , R ] } \Big )\tag{52}
$$

in $\mathbb { R } _ { \mathrm { s y m } } ^ { m \times m } \times \mathbb { R } ^ { m } \times C ( [ - R , R ] )$ , where $C ( [ - R , R ] )$ is equipped with the uniform norm.

For any symmetric $D _ { : }$ , let $\mu _ { 1 } ( D ) , \ldots , \mu _ { m } ( D )$ denote the eigenvalues of $D _ { : }$ , including zero and negative ones. Then a spectral decomposition expresses $\mathcal { H } _ { T } ( s ; D )$ as a nonnegative weighted sum of $( s + \mu _ { j } ( D ) ) ^ { 2 T }$ , with at least one strictly positive weight because $y \ne 0$ Moreover, for every $T \geq 1$ , the map $s \mapsto ( s + \lambda ) ^ { 2 T }$ is strictly convex and coercive. Hence $\mathcal { H } _ { T } ( \cdot ; D )$ is strictly convex and coercive, and therefore admits a unique minimizer $s _ { T } ( D )$ Since $\hat { \eta } _ { n , T }$ coincides with $\eta _ { n , T }$ with probability tending to one, the global minimizer of $s \mapsto$ $n ^ { T } \phi _ { n , T } ( \dot { \eta _ { 0 } } ( 1 + \epsilon _ { n } s ) )$ coincides with $s _ { n , T }$ on the same event. Thus the continuous mapping theorem and (52) yield

$$
( M _ { n } , \chi _ { n } ^ { ( 0 ) } , s _ { n , T } ) \stackrel { d } { \longrightarrow } ( M , - y , s _ { T } ( M ) ) .
$$

Since $\phi _ { \infty , T } ( \eta _ { 0 } ) = 0$ , we have $a _ { n , T } = \phi _ { n , T } ( \eta _ { n , T } )$ . Evaluating (51) at $s = s _ { n , T }$ therefore gives

$$
n ^ { T } a _ { n , T } \xrightarrow { d } L _ { T } : = \mathcal { H } _ { T } ( s _ { T } ( M ) ; M ) .
$$

By (48), we further obtain

$$
n ^ { T } c _ { n , T } = \frac { | | y | | ^ { 2 } } { 2 m } | s _ { n , T } | ^ { 2 T } \overset { d } { \longrightarrow } C _ { T } : = \frac { | | y | | ^ { 2 } } { 2 m } | s _ { T } ( M ) | ^ { 2 T } .
$$

These convergences are joint; hence,

$$
\left( \sqrt { n } \left( \frac { \eta _ { n , T } } { \eta _ { 0 } } - 1 \right) , n ^ { T } a _ { n , T } , n ^ { T } c _ { n , T } \right) \stackrel { d } { \longrightarrow } ( s _ { T } ( M ) , L _ { T } , C _ { T } ) .
$$

Observe that the original random initial prediction is retained in (51). Since $\chi _ { n } ^ { ( 0 ) } = - y +$ $O _ { p } ( n ^ { - 1 / 2 } )$ , replacing $\chi _ { n } ^ { ( 0 ) }$ by $- y$ changes the $n ^ { T }$ -rescaled local objective by only $O _ { p } ( n ^ { - 1 / 2 } )$ uniformly on compact sets of s. Hence the initialization fluctuation in $\chi _ { n } ^ { ( 0 ) }$ does not contribute to the leading $O _ { p } ( 1 )$ local limit.

It remains to establish nondegeneracy. If $L _ { T } = 0$ , then $( M + s _ { T } ( M ) I _ { m } ) ^ { T } y = 0$ . By symmetry of $M$ , this implies $( M + s _ { T } ( M ) I _ { m } ) y = 0$ , so y must be an eigenvector of $M$ , and hence of $\mathcal { G }$ . For a fixed nonzero y this event has probability zero when $m \geq 2 \colon$ writing $e = y / \| y \|$ , the vector $( I _ { m } - e e ^ { \top } ) \mathcal { G } e = P _ { e }$ ⊥Ge is a nondegenerate Gaussian vector on $e ^ { \perp }$ Thus $L _ { T } > 0$ almost surely.

We next show that $C _ { T } > 0$ almost surely. Since $M = \alpha I _ { m } + \mathcal { G }$ , for every $s \in \mathbb { R }$ we have $\mathcal { H } _ { T } ( s ; M ) = \mathcal { H } _ { T } ( s + \alpha ; \mathcal { G } )$ . Therefore, adding the scalar perturbation $\alpha I _ { m }$ only translates the minimizing argument, and hence we have

$$
s _ { T } ( M ) = s _ { T } ( { \mathcal G } ) - \alpha .
$$

Moreover, evaluating the objective at the minimizer gives

$$
L _ { T } = \mathcal { H } _ { T } ( s _ { T } ( M ) ; M ) = \mathcal { H } _ { T } ( s _ { T } ( \mathcal { G } ) ; \mathcal { G } ) .
$$

Thus $L _ { T }$ depends only on $\mathcal { G }$ , whereas

$$
C _ { T } = \frac { \Vert y \Vert ^ { 2 } } { 2 m } | s _ { T } ( \mathcal { G } ) - \alpha | ^ { 2 T } .
$$

Conditional on $\mathcal { G } _ { : }$ , the quantity $s _ { T } ( \mathcal G )$ is fixed and $\alpha \sim \mathcal { N } ( 0 , 2 )$ is independent of ${ \mathcal { G } } .$ . Hence we have $\mathrm { P } ( s _ { T } ( M ) = 0 \mid \mathcal { G } ) = 0 .$ , and therefore $C _ { T } > 0$ almost surely. Together with the preceding argument showing $L _ { T } > 0$ almost surely, this implies

$$
\mathrm { P } \left( 0 < \frac { C _ { T } } { L _ { T } } < \infty \right) = 1 .
$$

Finally, the continuous mapping theorem proves

$$
{ \frac { c _ { n , T } } { a _ { n , T } } } \ { \xrightarrow { d } } \ { \frac { C _ { T } } { L _ { T } } } = \Theta _ { p } ( 1 ) .
$$

Remark H.1 (Why the spectral condition matters). In this example, we have $B _ { \infty } ( \eta _ { 0 } ) =$ 0. Thus by Lemma C.6, the first-order loss sensitivity $D _ { ( \chi , A ) } \Phi _ { T } ( \eta _ { 0 } ; - y , K )$ vanishes, and the $n ^ { - 1 / 2 }$ optimal-loss fluctuation used in Corollary 3.7 is absent. Instead, both loss gaps are of order $n ^ { - T }$ . This is not a failure of learning-rate consistency or of absolute transfer accuracy: $b _ { n , T } \to 0$ and $c _ { n , T } \to 0$ in probability still hold. What fails is the relative criterion $c _ { n , T } = o _ { p } ( a _ { n , T } )$ . We also note that $D _ { ( \chi , A ) } F _ { T } ( \eta _ { 0 } ; - y , K )$ also vanishes when $T \geq 2$ while $D _ { ( \chi , A ) } F _ { 1 } ( \eta _ { 0 } ; - y , K ) [ h , E ] = - y ^ { \top } E y$

## References

Bordelon, B., A. Atanasov, and C. Pehlevan (2024, 21–27 Jul). A dynamical model of neural scaling laws. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp (Eds.), Proceedings of the 41st International Conference on Machine Learning, Volume 235 of Proceedings of Machine Learning Research, pp. 4345– 4382. PMLR.

Bordelon, B. and F. Mori (2026). Theory of optimal learning rate schedules and scaling laws for a random feature model. arXiv preprint arXiv:2602.04774 .

Bordelon, B., L. Noci, M. Li, B. Hanin, and C. Pehlevan (2024). Depthwise hyperparameter transfer in residual networks: Dynamics and scaling limit. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (Eds.), International Conference on Learning Representations, Volume 2024, pp. 22088–22127.

Cohen, J., S. Kaur, Y. Li, J. Z. Kolter, and A. Talwalkar (2021). Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations.

Dey, N., B. Zhang, L. Noci, M. Li, B. Bordelon, S. Bergsma, C. Pehlevan, B. Hanin, and J. Hestness (2025). Don’t be lazy: CompleteP enables compute-eficient deep transformers. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Eds.), Advances in Neural Information Processing Systems, Volume 38, Main Conference, pp. 137707–137739. Curran Associates, Inc.

Everett, K. E., L. Xiao, M. Wortsman, A. A. Alemi, R. Novak, P. J. Liu, I. Gur, J. Sohl-Dickstein, L. P. Kaelbling, J. Lee, and J. Pennington (2024, 21–27 Jul). Scaling exponents across parameterizations and optimizers. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp (Eds.), Proceedings of the 41st International Conference on Machine Learning, Volume 235 of Proceedings of Machine Learning Research, pp. 12666–12700. PMLR.

Ghosh, N., D. Wu, and A. Bietti (2026). Understanding the mechanisms of fast hyperparameter transfer. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust (Eds.), International Conference on Learning Representations, Volume 2026, pp. 65022–65063.

Hayou, S. (2026, 02–05 May). A proof of learning rate transfer under µP. In E. Khan, Y. Li, A. Solin, and A. Ramdas (Eds.), Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, Volume 300 of Proceedings of Machine Learning Research, pp. 3322–3330. PMLR.

Hayou, S. and L. Liu (2025). Optimal embedding learning rate in LLMs: The efect of vocabulary size. arXiv preprint arXiv:2506.15025 .

Hofmann, J., S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, O. Vinyals, J. Rae, and L. Sifre (2022). An empirical analysis of compute-optimal large language model training. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Advances in Neural Information Processing Systems, Volume 35, pp. 30016–30030. Curran Associates, Inc.

Hong, L. and Z. Wang (2025, 13–19 Jul). On the provable separation of scales in maximal update parameterization. In A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaf, and J. Zhu (Eds.), Proceedings of the 42nd International Conference on Machine Learning, Volume 267 of Proceedings of Machine Learning Research, pp. 23774–23786. PMLR.

Lauditi, C., C. Pehlevan, and B. Bordelon (2026). Spectral dynamics in deep networks: Feature learning, outlier escape, and learning rate transfer. arXiv preprint arXiv:2605.07870 .

Lingle, L. (2024). An empirical study of µP learning rate transfer. arXiv preprint arXiv:2404.05728 .

Mlodozeniec, B., P. Ablin, L. B´ethune, D. Busbridge, M. Klein, J. Ramapuram, and M. Cuturi (2026). Completed hyperparameter transfer across modules, width, depth, batch and duration. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust (Eds.), International Conference on Learning Representations, Volume 2026, pp. 45300–45325.

Noci, L., A. Meterez, T. Hofmann, and A. Orvieto (2024). Super consistency of neural network landscapes and learning rate transfer. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Advances in Neural Information Processing Systems, Volume 37, pp. 102696–102743. Curran Associates, Inc.

Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (Second ed.). Society for Industrial and Applied Mathematics.

Shigida, B., B. Hanin, and A. Gromov (2026). Learning rate transfer in Normalized Transformers. arXiv preprint arXiv:2604.27077.

Wen, G., A. Bietti, N. Ghosh, T. Misiakiewicz, and D. Wu (2026). Fast learning rate transfer for gradient descent in sketched linear regression. In High-dimensional Learning Dynamics 2026.

Yang, G., E. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao (2021). Tuning large neural networks via zero-shot hyperparameter transfer. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan (Eds.), Advances in Neural Information Processing Systems, Volume 34, pp. 17084–17097. Curran Associates, Inc.

Yang, G. and E. J. Hu (2021, 18–24 Jul). Tensor programs iv: Feature learning in infinitewidth neural networks. In M. Meila and T. Zhang (Eds.), Proceedings of the 38th International Conference on Machine Learning, Volume 139 of Proceedings of Machine Learning Research, pp. 11727–11737. PMLR.

Yang, G., D. Yu, C. Zhu, and S. Hayou (2024). Tensor programs vi: Feature learning in infinite depth neural networks. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (Eds.), International Conference on Learning Representations, Volume 2024, pp. 55099–55150.