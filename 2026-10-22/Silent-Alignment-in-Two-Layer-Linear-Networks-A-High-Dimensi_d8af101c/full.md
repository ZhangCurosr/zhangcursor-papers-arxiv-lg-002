# Silent Alignment in Two-Layer Linear Networks: A High-Dimensional Perspective

Simon Martin<sup>∗1,2</sup>, Francis Bach<sup>1</sup>, and Giulio Biroli<sup>2</sup>

<sup>1</sup>INRIA - École Normale Supérieure, PSL Research University, Paris, France. <sup>2</sup>Laboratoire de Physique de l’École Normale Supérieure, ENS, Université PSL, CNRS, Sorbonne Université, Université de Paris, F-75005 Paris, France.

## Abstract

We study gradient flow in a two-layer linear network in the high-dimensional regime where the input, inner and output dimensions diverge proportionally. Under whitened inputs and isotropic imbalance, we combine a closed-form solution of the dynamics with prior random matrix theory results to derive a high-dimensional limit for the dynamics. Our method yields exact asymptotic spectral distributions for the weight matrices at all times and characterizes the evolution of the learned subspace. In the small-initialization limit, we identify a silent alignment phenomenon: the singular directions of the network align on an order-one timescale, while the singular values and empirical loss remain at their initial values. Visible learning occurs later, on a logarithmic timescale, as singular values progressively escape zero and the loss decreases. This phase is characterized by a saddle-to-saddle dynamics, which becomes a continuous process of spectral learning in the high-dimensional limit.

## Contents

## 1 Introduction

2 Setting 4   
2.1 Notation and Conventions . 4   
2.2 Model . 5   
2.3 Main Assumptions 6   
2.4 Overview of the Results 7   
3 Finite-Dimensional Dynamics 8   
3.1 Closed-Form Solution 9   
3.2 The Prediction Matrix . 10   
3.3 Limit of the Dynamics . 11   
4 High-Dimensional Limit 12   
4.1 Stieltjes Transforms . 13   
4.2 Range-Projector Distances . . 15   
5 Vanishing Initialization and Silent Alignment 16   
5.1 Alignment Phase 16   
5.2 Learning Phase 20   
6 Conclusion and Perspectives 24   
Funding 25   
Acknowledgments 25   
Data Availability 25   
References 26   
A Prerequisite 32   
A.1 Random Matrix Theory 32   
A.2 Complex Analysis 33   
A.3 Self-Consistent Equations 34   
B Finite-Dimensional Dynamics 35   
B.1 Closed-Form Solution 35   
B.2 Block Expressions for the Solution 36   
B.3 SVD of the Prediction Matrix 38   
B.4 Convergence of the Dynamics 39   
C High-Dimensional Limit 45   
C.1 Random Matrix Results 45   
C.2 Stieltjes Transforms 51   
C.3 Two-Point Stieltjes Transforms 57   
C.4 Range-Projector Distances 57   
D Small-Initialization Limit 61   
D.1 Dynamics at Fixed Times 62   
D.2 Range Projectors . 64   
D.3 Top Singular Subspaces 69   
D.4 Saddle-to-Saddle Dynamics 75   
D.5 Spectrum in the Small-Initialization Limit . 78   
D.6 Loss Function . 80   
E Numerical Simulations 84

## 1 Introduction

Neural networks learn representations from data rather than relying only on fixed features (Bengio et al., 2013). Understanding this process requires describing how training selects useful directions, how those directions evolve, and when their evolution begins to improve prediction. The nature of this feature evolution, and the tools available to analyze it, depend on the scaling regime under consideration.

Regimes of feature learning. A line of work that has successfully characterized feature learning in neural networks is the study of single and multi-index models, associated with target functions depending on a low-dimensional feature space. Several works study how gradient-based methods recover the relevant directions in such nonlinear models (Damian et al., 2022; Bietti et al., 2022, 2025). When the number of samples grows proportionally with the input dimension, dynamical studies of finite-width shallow networks rely on low-dimensional order parameters to obtain phase diagrams and study signal recovery (Saad and Solla, 1995; Goldt et al., 2020; Refinetti et al., 2021; Arnaboldi et al., 2023; Montanari and Wang, 2026).

Going beyond these low-dimensional representations is still a challenge. Indeed, in large-width limits or random feature models, features often remain frozen during training (Rahimi and Recht, 2007; Jacot et al., 2018; Chizat et al., 2019; Mei and Montanari, 2022). In the same regime, mean-field and other large-width parameterizations allow for non-trivial feature evolution (Chizat and Bach, 2018; Yang and Hu, 2021; Bordelon and Pehlevan, 2022), but their nonlinear dynamics are more challenging to describe.

A complementary line of work studies joint limits in which the network width grows with the input dimension, allowing for feature learning with a macroscopic number of features. For quadratic neural networks, these limits have led to precise dynamical descriptions (Martin et al., 2024; Ben Arous et al., 2025; Martin et al., 2026), as for the closely related problem of matrix denoising (Bodin and Macris, 2023). Beyond quadratic networks, related scalings have been studied for deep linear networks (Bordelon and Pehlevan, 2025; Lauditi et al., 2026) and for two-layer networks after a few gradient steps (Ba et al., 2022; Moniri and Hassani, 2026). We focus throughout the paper on this proportional setting, often referred to as the extensive-width regime.

Linear networks and factorized models. Factorized linear models have motivated a long line of work, as they produce nonlinear but tractable dynamics that exhibit complex phenomena also observed in modern neural networks. These models include two-layer and deep linear networks, diagonal networks and low-rank matrix factorization. They have been used to study convergence and alignment of the successive layers (Bah et al., 2022; Ji and Telgarsky, 2019), conservation laws (Marcotte et al., 2023), and the implicit bias toward low-rank solutions (Gissin et al., 2020; Gidel et al., 2019; Gunasekar et al., 2017; Arora et al., 2019). Other works reveal saddle-to-saddle dynamics and successive low-rank or sparse learning (Jacot et al., 2021; Pesme and Flammarion, 2024; Li et al., 2021; Chou et al., 2024; Kunin et al., 2025; Min and Vidal, 2025). At finite stepsize, these models can also exhibit edge-of-stability and oscillatory regimes (Even et al., 2023; Ghosh et al., 2025).

In two-layer linear networks, training is known to be exactly tractable under structured initialization (Fukumizu, 1998; Dominé et al., 2023), leading to an Oja flow (Oja, 1982; Yan et al., 1994) that decouples along the singular values for a spectral initialization (Saxe et al., 2014; Tarmoun et al., 2021; Dominé et al., 2025). In the overparameterized setting, Varre et al.

(2023) studied the implicit bias of the gradient flow dynamics and showed that, for whitened inputs and a specific initialization, the singular vectors remain frozen during training, while singular values escape zero following a discrete jump process in the small-initialization regime (see also Gidel et al., 2019; Bach, 2026).

Small initialization can also produce an early phase in which the learned directions align while the weights remain small, in linear networks and matrix sensing (Stöger and Soltanolkotabi, 2021; Soltanolkotabi et al., 2023; Tu et al., 2024), as well as in ReLU networks (Min et al., 2024; Boursier and Flammarion, 2025). Close to our setting, Atanasov et al. (2022) called this phase silent alignment when the loss also stays at its initial value, and analyzed it in linear networks through the evolution of the neural tangent kernel. Dominé et al. (2023) derived it in a simple two-dimensional setting from an exact solution, Karkada et al. (2025) in symmetric matrix factorization, and Zhang et al. (2026) described a similar early phase in linear networks with vector outputs, through the dynamics linearized near initialization. In high dimension, Lampinen and Ganguli (2019) showed that the singular values are learned as a wave sweeping the spectrum, but only from an initialization already aligned with the data. A challenge we address here is to characterize the whole dynamics from a random initialization in the extensive-width regime, where a macroscopic number of directions take part in both the alignment and the learning stages.

Contributions. We consider a two-layer linear network trained by gradient flow on the square loss, with whitened inputs. Our contributions are twofold:

• High-dimensional dynamics. By combining a known closed-form solution of the dynamics with prior random matrix theory results, we obtain a high-dimensional description of the training trajectory in the extensive-width regime, which equivalently describes the factorization of a matrix of extensive rank. At all times, we characterize the spectral distribution of the prediction matrix and quantify how the learned subspace moves, through explicit limits for distances between range projectors. This allows us to follow feature learning throughout training.

• Silent alignment. We make the silent alignment mechanism explicit from a random initialization, both in finite dimension and in the high-dimensional regime where the network learns an extensive number of features. In the small-initialization limit, the singular directions of the prediction align on an order-one timescale, while its singular values and loss remain at their initial values. The singular values then escape progressively on a later timescale and the loss starts to decrease. In finite dimension, this leads to a saddle-to-saddle dynamics that becomes a continuous spectral learning process in the high-dimensional limit.

## 2 Setting

In this section, we introduce the model we are interested in, as well as our main assumptions.   
We then state an informal version of our main result.

## 2.1 Notation and Conventions

For a matrix $\mathbf { A } \in \mathbb { R } ^ { d \times m }$ , we denote by $\mathbf { A } ^ { \top }$ its transpose, $\operatorname { T r } ( \mathbf { A } )$ its trace, $\| \mathbf { A } \| _ { F } = \left[ \operatorname { T r } ( \mathbf { A A } ^ { \top } ) \right] ^ { 1 / 2 }$ its Frobenius norm, Im(A) its range and $\operatorname { K e r } ( \mathbf { A } )$ its kernel. We denote by ${ \cal S } _ { d } ( \mathbb { R } ) , { \cal S } _ { d } ^ { + } ( \mathbb { R } )$ and ${ \mathcal { S } } _ { d } ^ { + + } ( \mathbb { R } )$ the sets of $d \times d \operatorname { s y }$ mmetric, positive semidefinite (PSD) and positive definite matrices.

For A, $\mathbf { B } \in { \cal S } _ { d } ( \mathbb { R } )$ , we write $\mathbf { A } \succeq \mathbf { B }$ when A−B is PSD, and equivalently write $\mathbf { B } \preceq \mathbf { A }$ . Whenever $m \leq d ,$ , we define the Stiefel manifold:

$$
{ \mathrm { S t } } ( d , m ) = \Big \{ \mathbf { R } \in \mathbb { R } ^ { d \times m } , \mathbf { R } ^ { \top } \mathbf { R } = \mathbf { I } _ { m } \Big \} .
$$

Given E a Euclidean space and $L \colon E  \mathbb { R }$ continuously diferentiable, we let $\nabla L ( \mathbf { x } ) \in E$ be its gradient at $\mathbf { x } \in E$ . When L is a function of several variables, we denote by $\nabla _ { \mathbf x } L$ the gradient with respect to the variable x.

If $\mathbf { A } \in { \cal S } _ { d } ( \mathbb { R } )$ has spectral decomposition $\begin{array} { r } { \mathbf { A } = \sum _ { i } \lambda _ { i } \mathbf { q } _ { i } \mathbf { q } _ { i } ^ { \top } } \end{array}$ , then for a scalar function f we write $\begin{array} { r } { f ( \mathbf { A } ) = \sum _ { i } f ( \lambda _ { i } ) \mathbf { q } _ { i } \mathbf { q } _ { i } ^ { \top } } \end{array}$ . For a rectangular matrix M with singular value decomposition (SVD) $\begin{array} { r } { \mathbf { M } = \sum _ { i } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ , we similarly write $\begin{array} { r } { f ( \mathbf { M } ) = \sum _ { i } f ( \sigma _ { i } ) \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$

## 2.2 Model

In the following we consider a linear network of the form $\mathbf { x } \in \mathbb { R } ^ { d } \mapsto \mathbf { U } \mathbf { V } ^ { \top } \mathbf { x } \in \mathbb { R } ^ { p }$ , where $\mathbf { U } \in \mathbb { R } ^ { p \times m }$ and $\mathbf { V } \in \mathbb { R } ^ { d \times m }$ . From a dataset $\mathcal { D } _ { n } = \left\{ \left( \mathbf { x } _ { k } , \mathbf { y } _ { k } \right) \right\} _ { 1 \leq k \leq n }$ where ${ \bf x } _ { 1 } , \ldots , { \bf x } _ { n } \in \mathbb { R } ^ { d }$ and $\mathbf { y } _ { 1 } , \ldots , \mathbf { y } _ { n } \in \mathbb { R } ^ { p }$ , we aim at minimizing the loss function:

$$
\mathcal { L } ( \mathbf { U } , \mathbf { V } ) = \frac { 1 } { 2 n } \sum _ { k = 1 } ^ { n } \left\| \mathbf { U V } ^ { \top } \mathbf { x } _ { k } - \mathbf { y } _ { k } \right\| _ { 2 } ^ { 2 } .\tag{1}
$$

Up to an additive constant, we have:

$$
\mathcal { L } ( \mathbf { U } , \mathbf { V } ) = \frac { 1 } { 2 } \mathrm { T r } \big ( \mathbf { U } ^ { \top } \mathbf { U } \mathbf { V } ^ { \top } \Sigma \mathbf { V } \big ) - \mathrm { T r } \big ( \mathbf { U } \mathbf { V } ^ { \top } \mathbf { C } ^ { \top } \big ) ,
$$

with

$$
\mathbf { \Delta } \mathbf { \Sigma } \mathbf { \Sigma } = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \mathbf { x } _ { k } \mathbf { x } _ { k } ^ { \top } \in \mathcal { S } _ { d } ^ { + } ( \mathbb { R } ) , \qquad \mathbf { \Delta } \mathbf { C } = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \mathbf { y } _ { k } \mathbf { x } _ { k } ^ { \top } \in \mathbb { R } ^ { p \times d } .
$$

Σ is the second-moment matrix of the inputs $\left( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \right)$ and C is the input-output cross-moment of the dataset. We optimize the loss function L using gradient flow on the matrices $\mathbf { U } , \mathbf { V }$

$$
\dot { \mathbf { U } } ( t ) = - \nabla _ { \mathbf { U } } \mathcal { L } \big ( \mathbf { U } ( t ) , \mathbf { V } ( t ) \big ) , \qquad \dot { \mathbf { V } } ( t ) = - \nabla _ { \mathbf { V } } \mathcal { L } \big ( \mathbf { U } ( t ) , \mathbf { V } ( t ) \big ) .
$$

The dynamics then reads:

$$
\begin{array} { r l } & { \dot { \mathbf { U } } ( t ) = - \big ( \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top } \mathbf { \Sigma } \big ) - \mathbf { C } \big ) \mathbf { V } ( t ) , } \\ & { \dot { \mathbf { V } } ( t ) = - \big ( \Sigma \mathbf { V } ( t ) \mathbf { U } ( t ) ^ { \top } - \mathbf { C } ^ { \top } \big ) \mathbf { U } ( t ) . } \end{array}\tag{2}
$$

This leads to a nonlinear system of coupled matrix-diferential equations. It is known (Arora et al., 2018; Du et al., 2018) that the matrix Imb $\mathbf { \boldsymbol { \mathsf { \Lambda } } } ( t ) = \mathbf { \boldsymbol { \mathsf { V } } } ( t ) ^ { \top } \mathbf { \boldsymbol { \mathbf { V } } } ( t ) - \mathbf { \boldsymbol { \mathbf { U } } } ( t ) ^ { \top } \mathbf { \boldsymbol { \mathbf { U } } } ( t )$ remains constant along the flow and is therefore determined by the initialization $\mathbf { U } _ { 0 } : = \mathbf { U } ( t = 0 )$ and $\mathbf { V } _ { 0 } : = \mathbf { V } ( t = 0 )$ Several works (Dominé et al., 2023; Varre et al., 2023; Dominé et al., 2025) have studied the dynamics under balanced or structured imbalance assumptions. In what follows, we call Imb(t) the imbalance matrix.

## 2.3 Main Assumptions

In this paper, we work under the following assumptions:

## Assumption 1.

1.1. Whitened data. The second-moment matrix of the inputs satisfies $\pmb { \Sigma } = \mathbf { I } _ { d }$ . In particular, this implies the large-data regime $n \geq d .$

1.2. High-dimensional scaling. The inner and output dimensions scale proportionally to the input dimension d as it diverges to infinity:

$$
m _ { \stackrel { d  \infty } { d  \infty } } \kappa d , \qquad p _ { \stackrel { d  \infty } { d  \infty } } \tau d ,\tag{3}
$$

for some fixed constants $\kappa \in ( 0 , 1 )$ and $\tau > 0$

1.3. Cross-moment matrix. The empirical measure of the singular values of $\mathbf { C } \in \mathbb { R } ^ { p \times d }$

$$
\mu _ { C } ^ { d } = \frac { 1 } { \operatorname* { m i n } ( p , d ) } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( p , d ) } \delta _ { \sigma _ { i } } ,
$$

weakly converges as $d \to \infty$ to a compactly supported probability measure $\mu _ { C }$ . Moreover, $\mu _ { C }$ admits the decomposition:

$$
\mu _ { C } ( \mathrm { d } x ) = ( 1 - \kappa ^ { * } ) \delta _ { 0 } ( \mathrm { d } x ) + \kappa ^ { * } \phi _ { C } ( x ) \mathrm { d } x ,
$$

for $\kappa ^ { * } \in ( 0 , 1 ]$ . We assume that $\phi _ { C }$ is a bounded density with compact support bounded away from zero. We denote $x _ { 0 } =$ inf $\mathrm { S u p p } ( \phi _ { C } ) > 0$ . Additionally, there exist constants $0 < c < L$ independent of $d ,$ such that every non-zero singular value of C belongs to $[ c , L ]$ for all $d \geq 1$

1.4. Initialization. The dynamics (2) is initialized with $\mathbf { U } _ { 0 } = \mathbf { 0 }$ and $\mathbf { V } _ { 0 } = \sqrt { 2 \rho } \mathbf { R }$ , for some $\rho > 0$ and $\mathbf { R } \in \mathbb { R } ^ { d \times m }$ uniformly drawn on the Stiefel manifold $\mathrm { S t } ( d , m )$

We now briefly comment on these assumptions and give several consequences.

• Loss and generalization. Under Assumption 1.1, the empirical loss can be written, up to an additive constant, as

$$
\mathcal { L } ( \mathbf { U } , \mathbf { V } ) = \frac { 1 } { 2 } \| \mathbf { U } \mathbf { V } ^ { \top } - \mathbf { C } \| _ { F } ^ { 2 } + \mathrm { c o n s t a n t } .\tag{4}
$$

The constant is the part of the empirical loss that cannot be fitted by a linear map of the inputs. Thus, the squared Frobenius norm is twice the excess empirical loss. Additionally, our analysis studies how the network learns the dataset through $\mathbf { C } ,$ rather than its generalization performance. Studying generalization would require an additional model for the data beyond the empirical cross-moment matrix.

• Extensive-width regime. Together, Assumptions 1.2 and 1.3 define an extensive-width regime (Maillard et al., 2024; Martin et al., 2026): the input, inner, and output dimensions grow proportionally, and the cross-moment matrix C possesses a macroscopic number of non-zero singular directions to be learned.

• Overparameterized and underparameterized regimes. Assumption 1.3 implies that the rank of C (that we denote $m ^ { * }$ in Section 3) is asymptotically $\beta \kappa ^ { * } d$ , where $\beta = \operatorname* { m i n } ( 1 , \tau )$ . Throughout the paper, we define the terms underparameterized and overparameterized relative to this rank: in finite dimension they correspond to $m < m ^ { * }$ and $m > m ^ { * }$ , while in the high-dimensional limit they correspond to $\kappa < \beta \kappa ^ { * }$ and $\kappa > \beta \kappa ^ { * }$ respectively. We do not treat the critical high-dimensional case $\kappa = \beta \kappa ^ { * }$ , which would require a separate analysis.

• Structured initialization. Assumption 1.4 is similar to the one used by Varre et al. (2023), except that they work in the regime where $m \geq d .$ . This assumption implies that the imbalance matrix is given by $\mathbf { I m } \mathbf { b } ( t ) = 2 \rho \mathbf { I } _ { m }$ at all times. This setting of isotropic imbalance has also been studied by Tarmoun et al. (2021); Dominé et al. (2025).

• Matrix factorization. Under Assumption 1.1, the expression of the loss in equation (4) shows that the dynamics (2) is also the gradient flow of the asymmetric factorization of C, so that our results also apply to matrix factorization with a target of extensive rank.

## 2.4 Overview of the Results

The following informal theorem summarizes the main result of the paper: in the small-initialization limit, learning takes place on two distinct timescales.

Informal Theorem. Consider the gradient flow (2) under Assumption 1. In the high-dimensional limit and for small initialization $\rho ,$ the dynamics separates into two phases:

• Alignment phase. $A t \ t = \Theta ( 1 )$ as $\rho \to 0$ , the loss remains at its initial value and the prediction $\mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ remains near zero, but its singular subspaces become correlated with those of C.

• Learning phase. At $t = \Theta \big ( \log ( \rho ^ { - 1 } ) \big )$ , the singular values of $\mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ escape zero to match those of C. The loss progressively decreases, and after a time $t = s \log ( \rho ^ { - 1 } )$ , the prediction has recovered the singular values of C above the threshold

$$
x _ { s } = { \frac { 1 } { 2 s } } ,
$$

as long as the rank of the prediction allows it.

Figure 1 summarizes in more detail the contributions of the paper.

The two phases are visualized in Figure 2. The alignment error is the normalized right range-projector distance introduced in Section 4.2, while the loss is the normalized squared distance between $\mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ and C, equivalently the excess empirical loss by equation (4). On the timescale $t = \Theta ( 1 )$ , the finite-dimensional alignment curves collapse onto the $\rho \to 0$ prediction of Theorem 8, whereas the loss remains essentially constant. On the scale $t = s \log ( \rho ^ { - 1 } )$ , the loss curves collapse instead as $\rho \to 0$ , and their decay as a function of s is directly linked with the progressive learning of the singular values of C starting from the largest ones.

The following sections are dedicated to our results. They are organized as follows:

• In Section 3, we analyze the gradient flow dynamics in finite dimension. We recall an explicit closed-form solution for the dynamics (originally due to Fukumizu, 1998) and study its convergence at long times.

• In Section 4, we analyze the dynamics in the high-dimensional limit. We derive deterministic spectral distributions for the matrices associated with the dynamics and quantify the evolution of the learned subspace through projector distances.

• In Section 5, we consider the small-initialization limit $\rho \to 0$ and derive the two timescales of the dynamics, which correspond to a silent alignment phenomenon (Atanasov et al., 2022).

A few of our results already exist in the literature or are closely related to prior works. We recall them as they are essential for subsequent new results and lead to a more complete picture of the phenomena we describe. Whenever this is the case, we make it explicit and provide the relevant references.

![](images/cf943da34a74e27395ff4452a2d5003caae94475dca26a3028e48a8b1bb60bd9.jpg)  
Figure 1: Map of the results, with time increasing from left to right. The upper block gathers the results that hold for any $\rho > 0$ at all times. The two lower blocks consider the small-initialization limit $\rho \to 0$ , where the dynamics separates into an alignment stage and a learning stage. Both lead to the same long-time limit, which holds for any ρ (right). In each block, the top part gives the finite-dimensional result and the bottom part the high-dimensional limit $d \to \infty ,$ taken before $\rho \to 0$

The code for all simulations, figures and numerical checks, together with animations of the learning dynamics, is available in a GitHub repository. More detail on the numerical simulations can be found in Appendix E.

## 3 Finite-Dimensional Dynamics

In this section, we study the gradient flow dynamics (2) in finite dimension.

• In Section 3.1, we recall a closed-form solution for the gradient flow dynamics.

• In Section 3.2, we show that in our setting, the prediction matrix $\mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ can be understood from the Gram matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } , \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$

• In Section 3.3, we compute the long-time limit of the Gram and prediction matrices.

![](images/a3435e5847b003117bff63ae206b84d09399c0217baeb264fac87eb189b994e2.jpg)  
Figure 2: Colored curves are finite-dimensional gradient descent simulations of the dynamics (2) at positive $\rho ,$ averaged over 10 repetitions, with $d = 5 0 0$ , stepsize $\eta = 1 0 ^ { - 3 } , \tau = 1 , \kappa = 0 . 7 5$ and $\kappa ^ { * } = 0 . 5 5$ . Both quantities are divided by their initial values. $T o p { - } l e f t$ panel: alignment error (solid) on the timescale $t = \Theta ( 1 )$ for decreasing values of $\rho ,$ with the $\rho \to 0$ prediction of Theorem 8 in black. Dashed curves show the loss on the same timescale, which starts to decrease later as $\rho$ decreases and stays at its initial value as $\rho \to 0$ . Top-right panel: loss (solid) on the scale $s = t / \log ( \rho ^ { - 1 } )$ , with its $\rho \to 0$ limit from Theorem 13 in black. Dashed curves show the alignment error on this scale, whose $\rho \to 0$ limit vanishes for any $s > 0$ . Bottom-right panels: histograms of the singular values of $\mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ for $\rho = 1 0 ^ { - 8 }$ at the times $s _ { 1 } , s _ { 2 } , s _ { 3 }$ , normalized by the total number of singular values, compared with the target $\kappa ^ { * } \phi _ { C } \colon$ the singular values of C are learned from the largest ones downwards.

## 3.1 Closed-Form Solution

The first result we present is a closed-form solution for the dynamics, which shows that the gradient flow reduces to an Oja flow (Oja, 1982).

Proposition 1 (Oja flow representation, Fukumizu, 1998; Tarmoun et al., 2021). Consider the solution $\mathbf { U } ( t ) , \mathbf { V } ( t )$ of the gradient flow dynamics (2). Then, under Assumptions 1.1 and $1 . 4 \AA$ , the matrix

$$
\mathbf { W } ( t ) = { \binom { \mathbf { U } ( t ) } { \mathbf { V } ( t ) } } \in \mathbb { R } ^ { ( p + d ) \times m } ,
$$

is a solution of the matrix diferential equation:

$$
\dot { \mathbf { W } } ( t ) = - \frac { 1 } { 2 } \mathbf { W } ( t ) \mathbf { W } ( t ) ^ { \top } \mathbf { W } ( t ) + \mathbf { A } _ { \rho } \mathbf { W } ( t ) ,\tag{5}
$$

where

$$
\mathbf { A } _ { \rho } = \left( \begin{array} { l l } { - \rho \mathbf { I } _ { p } } & { \mathbf { C } } \\ { \mathbf { C } ^ { \top } } & { \rho \mathbf { I } _ { d } } \end{array} \right) \in S _ { p + d } ( \mathbb { R } ) .
$$

As a consequence, we have the closed-form solution for all $t \geq 0$

$$
\mathbf { W } ( t ) \mathbf { W } ( t ) ^ { \top } = e ^ { t \mathbf { A } _ { \rho } } \mathbf { W } _ { 0 } \left( \mathbf { I } _ { m } + \mathbf { W } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \mathbf { A } _ { \rho } } \mathrm { d } s \mathbf { W } _ { 0 } \right) ^ { - 1 } \mathbf { W } _ { 0 } ^ { \top } e ^ { t \mathbf { A } _ { \rho } } .\tag{6}
$$

Fukumizu (1998) obtained this result in the case of a balanced initialization, and Tarmoun et al. (2021); Dominé et al. (2025) extended it to an imbalance proportional to the identity. For completeness we prove this result in Appendix B.1. Additionally, in Appendix B.2, we derive explicit expressions for the Gram matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } , \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ and the prediction $\mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ from the above expression of $\mathbf W ( t ) \mathbf W ( t ) ^ { \top }$

The closed-form expression in equation (6) forms the basis of our analysis and allows us to study this dynamics in the high-dimensional limit, under the scaling of Assumption 1.2. This result turns the high-dimensional dynamics into a random matrix problem involving $\mathbf { W } _ { 0 }$ and $\mathbf { A } _ { \rho }$ The matrix $\mathbf { A } _ { \rho }$ can be interpreted as a ρ-deformation of the Hermitian dilation of C (see Tropp, 2015, Definition 2.1.5), therefore its spectral structure follows directly from that of C.

Equation (5) is an $\mathrm { O j a }$ flow. This nonlinear matrix dynamics is classical in finite dimension (Yan et al., 1994; Tarmoun et al., 2021; Tsuzuki and Ohki, 2025), and closely related flows appear in matrix denoising and quadratic neural networks, including in the extensive-width regime (Bodin and Macris, 2023; Martin et al., 2024, 2026). We derive new results for the present setting, and our techniques can also be applied to obtain a more precise understanding of the $\mathrm { O j a }$ flow in high dimension.

## 3.2 The Prediction Matrix

We are now interested in the prediction matrix $\mathbf { Z } ( t ) : = \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ , as the loss function in equation (1) only depends on U, V through Z. In the following, our analysis separates two aspects of learning:

• The recovery of feature directions, characterized by the singular vectors of $\mathbf { Z } ( t )$ and their overlap with those of C.

• The amplitudes associated with these directions, characterized by the singular values of $\mathbf { Z } ( t )$ In order to study these objects, we relate them to the symmetric matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ by exploiting the imbalance conservation.

Proposition 2 (Singular structure of the prediction). Consider the solution $\mathbf { U } ( t ) , \mathbf { V } ( t )$ of the gradient flow dynamics (2), and suppose Assumptions 1.1 and $1 . 4$ hold. Let $m ^ { * } = { \mathrm { r a n k } } ( \mathbf { C } ) \leq$ min $. ( p , d )$ . Then, with probability one (under the random initialization R), for all $t > 0 ,$

$$
\operatorname { r a n k } \mathbf { U } ( t ) = \operatorname* { m i n } ( m , m ^ { * } ) , \qquad \operatorname { r a n k } \mathbf { V } ( t ) = m , \qquad \operatorname { r a n k } \mathbf { Z } ( t ) = \operatorname * { m i n } ( m , m ^ { * } ) .\tag{7}
$$

Set $r _ { Z } = \operatorname* { m i n } ( m , m ^ { * } )$ . There exist eigendecompositions

$$
\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } = \sum _ { i = 1 } ^ { r _ { Z } } \lambda _ { i } ^ { U } ( t ) \widehat { \mathbf { u } } _ { i } ( t ) \widehat { \mathbf { u } } _ { i } ( t ) ^ { \top } , \qquad \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } = \sum _ { i = 1 } ^ { m } \lambda _ { i } ^ { V } ( t ) \widehat { \mathbf { v } } _ { i } ( t ) \widehat { \mathbf { v } } _ { i } ( t ) ^ { \top } ,
$$

where $\lambda _ { 1 } ^ { U } ( t ) \ge \cdots \ge \lambda _ { r z } ^ { U } ( t ) > 0$ and $\lambda _ { 1 } ^ { V } ( t ) \geq \cdots \geq \lambda _ { m } ^ { V } ( t ) > 0$ , such that $\lambda _ { i } ^ { V } ( t ) = \lambda _ { i } ^ { U } ( t ) + 2 \rho$ for $i \in \{ 1 , \ldots , r _ { Z } \}$ and $\bar { \lambda } _ { i } ^ { V } ( t ) = 2 \rho f o r i \in \{ r _ { Z } + 1 , \ldots , m \}$ . Finally, we have the singular value decomposition:

$$
\mathbf { Z } ( t ) = \sum _ { i = 1 } ^ { r _ { Z } } \sqrt { \lambda _ { i } ^ { U } ( t ) \big ( \lambda _ { i } ^ { U } ( t ) + 2 \rho \big ) } \widehat { \mathbf { u } } _ { i } ( t ) \widehat { \mathbf { v } } _ { i } ( t ) ^ { \top } .\tag{8}
$$

We prove this proposition in Appendix B.3. Note that a similar version of this result exists in the closely related work of Varre et al. (2023, Theorem C.11), in the case $m \geq d ,$ where singular vectors of the prediction remain frozen. In our setting (where $m < d )$ , the singular vectors evolve, leading to a genuine alignment phase.

This result is essential for our analysis as it shows that the singular values and singular vectors of $\mathbf { Z } ( t )$ can be analyzed through the eigenvalues and eigenvectors of the Gram matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ . As symmetric matrices are more easily analyzed in the high-dimensional limit (see Section 4), we focus on these two matrices to study the singular structure of $\mathbf { Z } ( t )$

## 3.3 Limit of the Dynamics

We now move on to the limit of the dynamics (2) as $t \to \infty$ . Thanks to Proposition 1, this limit can be characterized by studying the long-time behavior of the $\mathrm { O j a }$ flow.

Proposition 3 (Long-time limit). Let $\begin{array} { r } { \mathbf { C } = \sum _ { i = 1 } ^ { m ^ { * } } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ be a singular value decomposition for C, with $m ^ { * } \leq \operatorname* { m i n } ( p , d )$ and $\sigma _ { 1 } \geq . . . \geq \sigma _ { m ^ { * } } > 0$ . Whenever $m < m ^ { * }$ , assume in addition that $\sigma _ { m } > \sigma _ { m + 1 }$ . Then, under Assumptions 1.1 and $1 . 4 { : }$ with probability one (with respect to the randomness of R), we have the convergence:

$$
\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } \xrightarrow [ t  \infty ] { \operatorname* { m i n } ( m , m ^ { * } ) } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( m , m ^ { * } ) } ( \gamma _ { i } - \rho ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } ,\tag{9a}
$$

$$
\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } \xrightarrow [ t \to \infty ] { \operatorname* { m i n } ( m , m ^ { * } ) } ( \gamma _ { i } + \rho ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } + 2 \rho \mathbf { 1 } _ { m > m ^ { * } } \overline { { \Pi } } ,\tag{9b}
$$

$$
\mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top } \xrightarrow [ { t  \infty } ] { } \quad \sum _ { i = 1 } ^ { } \quad \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } ,\tag{9c}
$$

where $\gamma _ { i } = \sqrt { \sigma _ { i } ^ { 2 } + \rho ^ { 2 } }$ and Π is the orthogonal projection onto Im $( \mathbf { R } ) \cap \mathop { \mathrm { K e r } } ( \mathbf { C } ) \subset \mathbb { R } ^ { d }$ . In the case $m > m ^ { * }$ , with probability one, Π has rank $m - m ^ { * }$

Selection of the dominant eigenspace by the Oja flow is classical (Yan et al., 1994; Tsuzuki and Ohki, 2025). Our proof first identifies the leading eigenspace selected by the Oja flow dynamics (5). In the overparameterized regime, we isolate the invariant subspace inherited from the initialization, which leads to the projection Π. The three limits then follow by identifying the blocks of $\mathbf W ( t ) \mathbf W ( t ) ^ { \top }$

Interestingly, the presence of the projection Π (in the overparameterized case $m > m ^ { * } )$ reveals an initialization-dependent implicit bias. A related phenomenon was studied by Varre et al. (2023) in the regime $m \geq d .$ In our case, any projection of rank $m - m ^ { * }$ with range contained in Ker(C) would leave the loss and the prediction unchanged, but the dynamics selects a specific one from the initialization. The prediction $\mathbf { Z } = \mathbf { U } \mathbf { V } ^ { \top }$ is insensitive to this choice and selects the largest singular values of the cross-moment matrix C. The projection term in equation (9b) is associated with an uninformative part since its range is contained in Ker(C).

Apart from the projection Π, the limiting Gram matrices are spectral functions of $\mathbf { C C } ^ { \top }$ and $\mathbf { C } ^ { \top } \mathbf { C }$ . Comparing $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ to these reference matrices will then indicate how the dynamics correlates with the dataset, and more precisely the cross-moment matrix C.

![](images/dd2657c7fafa5c75de8834c543be9b5b71a198c68138ccb84a3d9ba5ce0d60e3.jpg)  
Figure 3: Colored curves and histograms are gradient descent simulations of the dynamics (2), with $d = 5 0 0 .$ , stepsize $\eta = 1 0 ^ { - 3 } , \tau = 1$ , and $\rho = 0 . 1$ , over 10 repetitions. Left panel: normalized distance to convergence $d ^ { - 1 } \| \mathbf { Z } ( t ) - \mathbf { Z } _ { \infty } \| _ { F } ^ { 2 }$ as a function of $t ,$ averaged over the repetitions, where $\mathbf { Z } _ { \infty }$ is the limit in (9c). Right panels: histogram of the non-zero singular values of $\mathbf { Z } ( t )$ at $t = 5 0 0$ for $\kappa = 0 . 5 5 , \kappa ^ { * } = 0 . 9$ (underparameterized regime) and $\kappa = 0 . 7 5 , \kappa ^ { * } = 0 . 5 5$ (overparameterized regime). The dashed curve is the density $\kappa ^ { * } \phi _ { C }$ , and the vertical line marks the rank-selection threshold $x _ { * }$ defined in Corollary 5.

Figure 3 illustrates the spectral selection mechanism of the dynamics from the point of view of the prediction. The left panel measures the distance to the limit $\mathbf { Z } _ { \infty }$ given by Proposition 3, and confirms convergence to zero. The right panels show which singular values survive at convergence: in the underparameterized regime only the top part of the spectrum of C is selected, while in the overparameterized regime the whole non-zero spectrum is recovered by the dynamics.

## 4 High-Dimensional Limit

This section describes the dynamics in the high-dimensional limit. The starting point of our analysis is the explicit solution in equation (6), from which we derive exact expressions for the Gram matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } \in S _ { p } ^ { + } ( \mathbb { R } )$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } \in \mathcal { S } _ { d } ^ { + } ( \mathbb { R } )$ (see Appendix B.2). Since these matrices are high-dimensional, we study them through their Stieltjes transforms, a central object in random matrix theory:

$$
\mathfrak { m } ^ { A } ( w ) = \frac { 1 } { d } \mathrm { T r } \Big ( \big ( \mathbf { A } - w \mathbf { I } _ { d } \big ) ^ { - 1 } \Big ) ,
$$

for a matrix $\mathbf { A } \in { \cal S } _ { d } ( \mathbb { R } )$ and $w \in \mathbb { C } \setminus \mathbb { R }$ . This object is particularly suited to analyze random matrices in high dimension: the Stieltjes transform is scalar-valued and uniquely determines the empirical spectral distribution of A. More details can be found in Appendix A.1.

Stieltjes transforms encode spectral distributions but contain no information about the eigenvectors of the underlying matrices. Since our work also studies the evolution of learned subspaces, we directly analyze range-projector distances in Section 4.2. A complementary approach is to introduce a generalized Stieltjes transform comparing the learned Gram matrices with the cross-moment Gram matrices $\mathbf { C C } ^ { \top }$ and $\mathbf { C } ^ { \top } \mathbf { C }$ . This quantity encodes eigenvector overlaps and can therefore track how the prediction learns the relevant subspace. We discuss this extension and derive its high-dimensional limit in Appendix C.3.

## 4.1 Stieltjes Transforms

In the following, we are interested in the Stieltjes transforms of the matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ ， $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$

$$
\mathsf { m } _ { t } ^ { U } ( w ) = \frac { 1 } { p } \mathrm { T r } \Big ( \big ( \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } - w \mathbf { I } _ { p } \big ) ^ { - 1 } \Big ) , \qquad \mathsf { m } _ { t } ^ { V } ( w ) = \frac { 1 } { d } \mathrm { T r } \Big ( \big ( \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } - w \mathbf { I } _ { d } \big ) ^ { - 1 } \Big ) .\tag{10}
$$

It is also possible to define a Stieltjes transform for the matrix $\mathbf { Z } ( t ) = \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ . Since it is rectangular, we instead consider the empirical measure associated with its singular values and the corresponding Stieltjes transform:

$$
\mathfrak { m } _ { t } ^ { Z } ( w ) = \frac { 1 } { \operatorname* { m i n } ( p , d ) } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( p , d ) } \frac { 1 } { \sigma _ { i } ( t ) - w } ,\tag{11}
$$

for $w \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ . Thus ${ \mathfrak { m } } _ { t } ^ { U }$ and ${ \mathfrak { m } } _ { t } ^ { V }$ describe the empirical eigenvalue distributions of the symmetric Gram matrices, whereas $\mathfrak { m } _ { t } ^ { Z }$ describes the empirical singular-value distribution of the prediction. Throughout the paper, we denote by $\mu _ { t } ^ { U } , \mu _ { t } ^ { V }$ and $\mu _ { t } ^ { Z }$ the empirical measures respectively associated with the $p$ eigenvalues of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ , the d eigenvalues of $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ and the min $( p , d )$ singular values of $\mathbf { Z } ( t )$ . The connection between these three measures is precisely the content of Proposition 2.

We now give deterministic limits for these transforms at any fixed training time.

Theorem 4 (Spectra at fixed time). Fix $t \geq 0$ . In the high-dimensional limit, under Assumption 1, with probability one, we have the convergence for all $w \in \mathbb { C } \backslash \mathbb { R } ^ { + }$

$$
\mathfrak { m } _ { t } ^ { U } ( w ) \xrightarrow [ d  \infty ] { } - \frac { 1 } { w } ( 1 - \frac { \kappa } { \tau } ) + \frac { \kappa } { \tau } g _ { t } ( w ) \int \frac { q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ( w ) \big ( r _ { t } ^ { \perp } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) ,\tag{12}
$$

$$
\mathfrak { m } _ { t } ^ { V } ( w ) \xrightarrow [ d  \infty ] { } - \frac { 1 - \kappa } { w } + \kappa g _ { t } ( w - 2 \rho ) \int \frac { q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ( w - 2 \rho ) \big ( r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) ,\tag{13}
$$

where $\gamma _ { x } = \sqrt { x ^ { 2 } + \rho ^ { 2 } } , \tilde { \mu } _ { C } = ( 1 - \beta ) \delta _ { 0 } + \beta \mu _ { C } \ w i t h \ \beta = \operatorname* { m i n } ( 1 , \tau ) , \ a n d , \ f o r \ \gamma \geq \rho ,$

$$
q _ { t } ( \gamma ) = 1 + \frac { \rho } { \gamma } \sinh ( 2 t \gamma ) + \frac { \rho ^ { 2 } } { \gamma ^ { 2 } } \bigl ( \cosh ( 2 t \gamma ) - 1 \bigr ) ,
$$

$$
r _ { t } ^ { \mathsf { L } } ( \gamma ) = 2 \rho \left( 1 - \frac { \rho ^ { 2 } } { \gamma ^ { 2 } } \right) \sinh ( t \gamma ) ^ { 2 } ,
$$

$$
r _ { t } ^ { \mathrm { R } } ( \gamma ) = 2 \rho \left( \cosh ( t \gamma ) + \frac \rho \gamma \sinh ( t \gamma ) \right) ^ { 2 } = r _ { t } ^ { \mathrm { L } } ( \gamma ) + 2 \rho q _ { t } ( \gamma ) .
$$

Moreover, $g _ { t }$ denotes the unique holomorphic solution on $\mathbb { C } \setminus \mathbb { R } ^ { + }$ of the equation:

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ( w ) \big ( r _ { t } ^ { \downarrow } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } ,\tag{14}
$$

for which $g _ { t } ( w ) > 0$ whenever $w < 0$

The limiting measures $\mu _ { t } ^ { U } , \mu _ { t } ^ { V }$ can be recovered from the Stieltjes–Perron inversion formula (see Theorem $ { 1 4 } )$ . Additionally, $\mu _ { t } ^ { Z }$ is related to $\mu _ { t } ^ { U }$ by

$$
\mu _ { t } ^ { Z } = \left( 1 - \frac { \tau } { \beta } \right) \delta _ { 0 } + \frac { \tau } { \beta } ( \theta _ { \rho } ) _ { \# } \mu _ { t } ^ { U } ,\tag{15}
$$

where $\theta _ { \rho } ( x ) = \sqrt { x ( x + 2 \rho ) }$ and $( \theta _ { \rho } ) _ { \# }$ denotes the pushforward by $\theta _ { \rho }$ (see Appendix A.1).

By Theorem 15, the convergence of these Stieltjes transforms is equivalent to the weak convergence of the corresponding spectral measures. The function $g _ { t }$ in Theorem 4 is computed self-consistently from an average over the distribution $\tilde { \mu } _ { C }$ through equation $( 1 4 )$ , as is usual in random matrix theory. Here $\tilde { \mu } _ { C }$ is the asymptotic spectral measure of $( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 } \in \mathcal { S } _ { d } ^ { + } ( \mathbb { R } )$

We prove this result in Appendix C.2.1. The main idea of the proof is to combine the explicit solution of Proposition 1 with a random matrix result (see Theorem 23) derived from a deterministic equivalent obtained by LeJeune et al. (2024).

Finally, we emphasize the very similar form between the expressions of $\mathfrak { m } _ { t } ^ { U } , \mathfrak { m } _ { t } ^ { V }$ . The main diferences originate from the diference in normalizations $( \mathbf { U } \in \mathbb { R } ^ { p \times m }$ while $\mathbf { V } \in \mathbb { R } ^ { d \times m } )$ , and the asymmetry of the initializations $( { \bf { U } } _ { 0 } = { \bf { 0 } }$ while $\mathbf { V } _ { 0 } = \sqrt { 2 \rho } \mathbf { R } )$ . The fact that the two Stieltjes transforms share the same self-consistent function g<sub>t</sub> evaluated at w and $w - 2 \rho$ is directly linked to the shift $2 \rho$ between the eigenvalues of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ established in Proposition 2. Additionally, we have the stronger identity:

$$
\mathfrak { m } _ { t } ^ { V } ( w ) = \tau \mathfrak { m } _ { t } ^ { U } ( w - 2 \rho ) + \frac { \kappa - \tau } { 2 \rho - w } - \frac { 1 - \kappa } { w } ,\tag{16}
$$

which can be derived directly from Proposition 2.

The equations in Theorem 4 can be solved numerically, and the limiting measures can be recovered by the Stieltjes–Perron inversion formula (see Theorem 14). In Figure 4, we focus on $\mu _ { t } ^ { Z }$ and compare the numerical integration of the above equations with numerical simulations of gradient descent with a small stepsize. At early times, the singular values of $\mathbf { Z } ( t )$ are still close to zero and as time increases, the spectral mass is transported toward the selected part of the spectrum of C. The density predicted by the above result matches this finite-dimensional evolution throughout the dynamics. Further details on how to simulate equations (12), (13) and (14) can be found in Appendix E.

In Appendix C.2.2, we show that taking $t = 0$ in the system of equations of Theorem 4 yields the spectral distribution chosen at initialization (Assumption 1.4). The next result gives the Stieltjes transforms in the limit $t \to \infty$

Corollary 5 (Spectra at long times). Consider the Stieltjes transforms in the high-dimensional limit $\mathfrak { m } _ { t } ^ { U } , \mathbf { \bar { m } } _ { t } ^ { V } , \mathbf { \bar { m } } _ { t } ^ { \bar { Z } }$ defined in Theorem 4. As $t \to \infty$ , we have the convergence, for every $w \in \mathbb { C } \backslash \mathbb { R } ^ { + }$

$$
\mathfrak { m } _ { t } ^ { U } ( w ) \xrightarrow [ t \to \infty ] { } - \frac { 1 } { w } \left( 1 - \frac { \operatorname* { m i n } ( \kappa , \beta \kappa ^ { * } ) } { \tau } \right) + \frac { \beta \kappa ^ { * } } { \tau } \int _ { x _ { * } } ^ { \infty } \frac { \phi _ { C } ( x ) \mathrm { d } x } { \gamma _ { x } - \rho - w } ,\tag{17}
$$

$$
\mathfrak { m } _ { t } ^ { V } ( w ) \xrightarrow [ t \to \infty ] { } - \frac { 1 - \kappa } { w } + \frac { ( \kappa - \beta \kappa ^ { * } ) ^ { + } } { 2 \rho - w } + \beta \kappa ^ { * } \int _ { x _ { * } } ^ { \infty } \frac { \phi _ { C } ( x ) \mathrm { d } x } { \gamma _ { x } + \rho - w } ,\tag{18}
$$

$$
\mathfrak { m } _ { t } ^ { Z } ( w ) \xrightarrow [ t \to \infty ] { } - \frac { 1 } { w } \left( 1 - \frac { \operatorname* { m i n } ( \kappa , \beta \kappa ^ { * } ) } { \beta } \right) + \kappa ^ { * } \int _ { x _ { * } } ^ { \infty } \frac { \phi _ { C } ( x ) \mathrm { d } x } { x - w } ,\tag{19}
$$

where, i $f \kappa > \beta \kappa ^ { * }$ , we use the convention $x _ { * } = x _ { 0 } =$ inf Supp ϕ<sub>C</sub>, and if $\kappa < \beta \kappa ^ { * } , x _ { * }$ is defined as

$$
\kappa = \beta \kappa ^ { * } \int _ { x _ { * } } ^ { \infty } \phi _ { C } ( x ) \mathrm { d } x .\tag{20}
$$

We prove this result in Appendix C.2.3. We then compare these formulas with the limiting spectral density found in Proposition 3 and show that they match. As a consequence, this shows that the high-dimensional limit $d \to \infty$ commutes with the long-time limit $t \to \infty$

![](images/1978e4e402a00dee7a8ec190bf35a42fb94bd09e6767dac29555550f4a6bffcf.jpg)  
Figure 4: Comparison between the empirical non-zero singular values of $\mathbf { Z } ( t )$ (purple histograms), obtained by gradient descent simulations of equation (2), and the high-dimensional density derived from equation (15) (purple curves). Gradient descent simulations are averaged over 20 repetitions, with $d = 5 0 0$ , stepsize $\eta = 1 0 ^ { - 3 }$ , τ = 1, κ = 0.55, $\kappa ^ { * } = 0 . 9 0$ , and $\rho = 0 . 1$ . Each panel corresponds to a diferent training time t. The dashed black curve is the density $\kappa ^ { * } \phi _ { C }$ , and the vertical line marks the threshold $x ,$ <sub>∗</sub> from Corollary 5.

In this result, $x _ { * }$ is the rank-selection threshold. In the underparameterized case, it selects a fraction κ of the mass of ${ \tilde { \mu } } _ { C } .$ , whereas in the overparameterized case, all the positive singular values are selected, and $x _ { * }$ can be chosen as the lower edge $x _ { 0 }$ of $\phi _ { C }$ . This precisely plays the same role as the selection of the m largest singular values of C in Proposition 3. If the underparameterized threshold is not unique, all solutions of equation (20) give the same limiting spectral distribution. Throughout the rest of the paper, we choose the largest such solution in the underparameterized case.

## 4.2 Range-Projector Distances

The previous results describe singular values but not singular vectors. We now study the left and right singular subspaces of $\mathbf { Z } ( t )$ , which correspond to the learned subspaces in output and input spaces. By Proposition 2, these are related to the ranges of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$

We define the orthogonal projections onto the left and right ranges of $\mathbf { Z } ( t )$

$$
\begin{array} { r } { \mathbf { P } ^ { \mathrm { L } } ( t ) = \mathrm { P r o j } _ { \mathrm { I m } \mathbf { Z } ( t ) } \in S _ { p } ( \mathbb { R } ) , \qquad \mathbf { P } ^ { \mathrm { R } } ( t ) = \mathrm { P r o j } _ { \mathrm { I m } ( \mathbf { Z } ( t ) ^ { \top } ) } \in S _ { d } ( \mathbb { R } ) . } \end{array}
$$

The main objects we consider are the squared distances

$$
\Delta ^ { \mathbf { L } } ( t ) ^ { 2 } = \frac { 1 } { d } \| \mathbf { P } ^ { \mathbf { L } } ( t ) - \mathbf { P } _ { \infty } ^ { \mathbf { L } } \| _ { F } ^ { 2 } , \qquad \Delta ^ { \mathsf { R } } ( t ) ^ { 2 } = \frac { 1 } { d } \| \mathbf { P } ^ { \mathsf { R } } ( t ) - \mathbf { P } _ { \infty } ^ { \mathsf { R } } \| _ { F } ^ { 2 } ,
$$

where $\mathbf { P } _ { \infty } ^ { \mathsf { L } } , \mathbf { P } _ { \infty } ^ { \mathsf { R } }$ are the $t \to \infty$ limits of $\mathbf { P } ^ { \mathsf { L } } ( t ) , \mathbf { P } ^ { \mathsf { R } } ( t )$ . Whenever $m < d ,$ the right range of $\mathbf { Z } ( t )$ does not span $\mathbb { R } ^ { d }$ (see Proposition 2), and its orientation can evolve during training. The same holds for the left range in the underparameterized case $m < m ^ { * }$ . The range projectors converge to those onto the selected singular spaces of C (see Proposition 3), so the distances quantify how the learned features align with the dataset.

Theorem 6 (High-dimensional range alignment). Under Assumption 1, fix $\rho > 0$ and $t > 0$ and assume that the singular values $o f \mathbf { C }$ satisfy $\sigma _ { m } > \sigma _ { m + 1 }$ for all d large enough whenever $m < m ^ { * }$ . For $\mathsf { a } = \mathsf { R }$ when $\kappa \in ( 0 , 1 )$ , and for $\mathsf { a } = \mathsf { L }$ whenever $\kappa \in ( 0 , \beta \kappa ^ { * } )$ , let g<sup>a</sup> be the unique positive solution of

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ^ { 2 } r _ { t } ^ { 2 } ( \gamma _ { x } ) } .
$$

Then the following convergence holds almost surely in the high-dimensional limit:

$$
\Delta ^ { \mathsf { a } } ( t ) ^ { 2 } \xrightarrow [ d  \infty ] { } \Delta _ { \rho } ^ { \mathsf { a } } ( t ) ^ { 2 } : = 2 ( 1 - \kappa ) \int _ { x _ { * } } ^ { \infty } \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { a } } r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) } .
$$

In the overparameterized case $\kappa > \beta \kappa ^ { * }$ , one has instead $\Delta _ { \rho } ^ { \mathsf { L } } ( t ) ^ { 2 } = 0$

We prove this theorem in Appendix C.4. This result therefore gives a simple expression of the range-projector distance in the high-dimensional limit. In the overparameterized regime, the fact that $\bar { \Delta _ { \rho } ^ { \mathsf { L } } } ( t ) ^ { 2 } = 0$ can be intuitively understood from the zero-initialization of U. Indeed, we have ${ \bf Z } ( 0 ) = { \bf \dot { 0 } }$ and:

$$
\begin{array} { r } { \dot { \mathbf Z } ( 0 ) = \mathbf { C } \mathbf { V } _ { 0 } \mathbf { V } _ { 0 } ^ { \top } = 2 \rho \mathbf { C } \mathbf { R } \mathbf { R } ^ { \top } . } \end{array}
$$

Thus, as soon as the dynamics starts, the range of $\mathbf { U } ( t )$ is generated by $\mathbf { C R R } ^ { \top }$ at leading order.   
For $m > m ^ { * }$ , its range is the full non-zero left singular space of C with probability one.

This result is later illustrated in Figure 5 when we consider the small-initialization limit $\rho \to 0$ This limit is the object of the next section.

## 5 Vanishing Initialization and Silent Alignment

We now consider the limit where the initialization scale ρ goes to zero. We show that at fixed $t > 0$ , the relevant singular spaces already start to align with those of C, while the singular values and the loss remain at their initial scale. We refer to this separation between early directional learning and later amplitude growth as silent alignment, by analogy with the kernel alignment studied by Atanasov et al. (2022). During the second phase, the singular values escape zero only on the timescale $t \asymp \log ( \rho ^ { - 1 } )$ ). On this scale, learning becomes visible through a spectral threshold moving across the singular values of C. This section derives these two timescales, and we refer to Figure 2 for a visual summary.

## 5.1 Alignment Phase

We first study the alignment phase at fixed time. We begin with a finite-dimensional intuition for the small-initialization limit $\rho \to 0$ , before turning to the high-dimensional results.

## 5.1.1 Finite-Dimensional Linearized Dynamics

We first consider the small-initialization limit in finite dimension and identify the normalization under which the matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } , \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ , and $\mathbf { Z } ( t ) = \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ admit non-trivial limits.

Proposition 7 (Small initialization at fixed time). Consider $t \geq 0$ fixed. As $\rho \to 0$ , under Assumptions 1.1 and 1.4, we have:

$$
\frac { 1 } { \rho } \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } \xrightarrow [ \rho  0 ] { } 2 \sinh ( t \mathbf { C } ) \mathbf { R } \mathbf { R } ^ { \top } \sinh ( t \mathbf { C } ) ^ { \top } ,\tag{21}
$$

$$
\frac { 1 } { \rho } { \bf V } ( t ) { \bf V } ( t ) ^ { \top } \xrightarrow [ \rho  0 ] { } 2 \cosh ( t ( { \bf C } ^ { \top } { \bf C } ) ^ { 1 / 2 } ) { \bf R } { \bf R } ^ { \top } \cosh ( t ( { \bf C } ^ { \top } { \bf C } ) ^ { 1 / 2 } ) ,\tag{22}
$$

$$
\frac { 1 } { \rho } \mathbf { Z } ( t ) \xrightarrow [ \rho  0 ] { } 2 \sinh ( t \mathbf { C } ) \mathbf { R } \mathbf { R } ^ { \top } \cosh ( t ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 } ) ,\tag{23}
$$

where sinh is applied to each singular value of C and cosh is applied to each eigenvalue of $( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$

This proposition is a straightforward consequence of the closed-form solution in Proposition 1. We prove this result in Appendix D.1. Additionally, we show that taking the $\rho \to 0$ limit in the Stieltjes transforms obtained in Theorem 4 leads to the high-dimensional spectral distribution of the matrices in Proposition 7.

These expressions correspond to a linearized regime, where the cubic term is subleading in the $\mathrm { O j a }$ flow dynamics (5). $\mathrm { A t } ~ t = \Theta ( 1 )$ the eigenvalues of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ are of order $\rho$ but are exponentially amplified by the matrix C. They become of order one at times of order $\log ( \rho ^ { - 1 } )$ . On the other hand, the corresponding eigenspaces are insensitive to this vanishing scale and can therefore correlate with those of C at finite time. This is the manifestation of silent alignment: directions are already being selected while the prediction has vanishing singular values and remains close to zero.

We emphasize that when linearizing the dynamics (2), the terms involving Σ are of higher order and vanish in the $\rho \to 0$ limit. This suggests the early alignment phenomenon we exhibit at $t = \Theta ( 1 )$ ) should hold without the whitening assumption (Assumption 1.1). However, as our results rely on the closed-form solution that requires this assumption, we keep it as a requirement for the rest of the section.

## 5.1.2 Range Projectors

We now study range projectors in the small-initialization limit. The distance derived in Theorem 6 has a non-trivial limit as $\rho \to 0$ for $t = \Theta ( 1 )$ , revealing the alignment of the range subspaces with those of C at this early timescale.

Theorem 8 (Range alignment at small initialization). Consider the setting of Theorem $\it 6 .$ Then, in the $\rho \to 0$ limit, with fixed $t > 0$ , we have:

$$
\Delta _ { \rho } ^ { \mathsf { a } } ( t ) ^ { 2 } \xrightarrow [ \rho \to 0 ] { } \Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 } : = 2 ( 1 - \kappa ) \int _ { x _ { * } } ^ { \infty } \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa b _ { t } ^ { \mathsf { a } } \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } ,
$$

for $\mathsf { a } = \mathsf { R }$ in all regimes, and $f o r \textsf { a } = \textsf { L }$ in the underparameterized regime $\kappa < \beta \kappa ^ { * }$ . In these regimes the map $t \mapsto \Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 }$ is strictly decreasing on $( 0 , \infty )$ . When $\kappa > \beta \kappa ^ { * }$ , one has instead $\Delta _ { 0 } ^ { \mathsf { L } } ( t ) ^ { 2 } = 0$ . Here

$$
\bar { r } _ { t } ^ { \mathsf { L } } ( x ) = \sinh ( t x ) ^ { 2 } , \qquad \bar { r } _ { t } ^ { \mathsf { R } } ( x ) = \cosh ( t x ) ^ { 2 } ,
$$

and $b _ { t } ^ { \mathsf { a } }$ is the unique positive solution of

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa b _ { t } ^ { 2 } \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } .\tag{24}
$$

We prove this result in Appendices D.2.1 and D.2.2. The above formula genuinely depends on t after taking the $\rho \to 0$ limit, and it decreases strictly with t: the alignment is already under way at any fixed time. The following theorem shows that this quantity vanishes as $t \to \infty :$ the prediction recovers the selected left and right singular spaces of C. The theorem also provides exact convergence rates.

Theorem 9 (Convergence rates for range alignment). Under Assumption 1, consider the limiting projector distances obtained in Theorem 8.

(i) Underparameterized case. Assume that $\kappa < \beta \kappa ^ { * }$ , that $\phi _ { C }$ is continuous at $x _ { * }$ , and that $\phi _ { C } ( x _ { * } ) > 0$ . Then, for a = L, R:

$$
\Delta _ { 0 } ^ { a } ( t ) ^ { 2 } \underset { t  \infty } { \sim } \beta \kappa ^ { * } \frac { \phi _ { C } ( x _ { * } ) } { t } \log ( 2 ) .
$$

(ii) Overparameterized case. Assume that $\kappa \in ( \beta \kappa ^ { * } , 1 )$ , let $x _ { 0 } =$ inf $\mathrm { S u p p } ( \phi _ { C } ) > 0$ , and assume that, for some $\phi _ { 0 } > 0$ and $\ell \in \mathbb { R } ^ { + }$

$$
\phi _ { C } ( x _ { 0 } + h ) \mathrel { \mathop { \sim } _ { h \to 0 ^ { + } } } \phi _ { 0 } h ^ { \ell } .
$$

Then, denoting by Γ the Gamma function:

$$
\Delta _ { 0 } ^ { \mathsf { R } } ( t ) ^ { 2 } \underset { t  \infty } { \sim } \frac { 8 \beta \kappa ^ { * } ( 1 - \kappa ) } { \kappa - \beta \kappa ^ { * } } \frac { \phi _ { 0 } \Gamma ( \ell + 1 ) } { ( 2 t ) ^ { \ell + 1 } } e ^ { - 2 x _ { 0 } t } .
$$

We prove these asymptotics in Appendix D.2.3.

The two regimes have diferent long-time behaviors. In the overparameterized case, the decay is exponentially fast and is controlled by the left edge $x _ { 0 }$ of the spectrum of $\mathbf { C } ,$ and the exponent ℓ only changes the polynomial prefactor. In the underparameterized case, the rank constraint selects the threshold $x _ { * }$ , and the singular values around this threshold produce the slower $1 / t$ relaxation. This is illustrated in Figure 5. The finite-dimensional simulations closely follow the high-dimensional prediction from Theorem 6, while the gray curves show the exact asymptotics from Theorem 9. The small discrepancies come from finite-dimensional efects, and the late plateaus in the middle and right panels are numerical precision floors.

## 5.1.3 Top Singular Subspaces

Range projectors only measure the learned subspace. We now ask whether the directions inside this subspace are learned in the correct order. To answer this, we compare the projectors onto the top singular spaces of $\mathbf { Z } ( t )$ and C. By Proposition 2, those of $\mathbf { Z } ( t )$ are also the top eigenspace projectors of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { \dot { V } } ( t ) \mathbf { V } ( t ) ^ { \top }$ . The next theorem treats a fixed rank in finite dimension and a rank proportional to d in the high-dimensional limit.

Theorem 10 (Alignment of leading singular subspaces). For $\mathsf { a } = \mathsf { L }$ , R and $k \leq \operatorname* { m i n } ( m , m ^ { * } )$ let $\mathbf { P } _ { k , \rho } ^ { \mathsf { a } } ( t )$ and $\mathbf { \phi } _ { \mathbf { k } } ^ { \mathbf { \sigma } } \mathbf { \Lambda } _ { \mathbf { k } } ^ { \mathbf { \sigma } }$ be the top-k left $( \mathsf { a } = \mathsf { L } )$ or right $\left( \mathsf { a } = \mathsf { R } \right)$ singular projectors of $\mathbf { Z } ( t )$ and C. Suppose Assumptions 1.1 and $1 . 4$ hold.

(i) Finite dimension. Assume that $\sigma _ { k } > \sigma _ { k + 1 }$ (with the convention $\sigma _ { m ^ { * } + 1 } = 0 )$ . Then, almost surely

$$
\operatorname* { l i m } _ { \rho \to 0 } \| \mathbf { P } _ { k , \rho } ^ { \mathsf { a } } ( t ) - \mathbf { P } _ { k } ^ { C , \mathsf { a } } \| _ { F } ^ { 2 } \underset { t \to \infty } { = } O \big ( e ^ { - 2 ( \sigma _ { k } - \sigma _ { k + 1 } ) t } \big ) .\tag{25}
$$

![](images/c1e82b32cacc3868ac2519ffb8dcd2df459bf7ff62c0859f49f9fdcf2be39053.jpg)  
Figure 5: Solid curves are finite-dimensional gradient descent simulations of the dynamics (2), averaged over 10 repetitions, with d = 500, stepsize $\eta = 1 0 ^ { - 3 } , \tau = 1$ , and $\rho = 1 0 ^ { - 5 }$ . Dashed curves are the high-dimensional predictions of Theorem 6, and gray curves are the long-time asymptotics from Theorem 9. Left panel: underparameterized regime, $\kappa = 0 . 5 5 , \kappa ^ { * } = 0 . 9 0$ showing the left and right distances. Middle and right panels: overparameterized regime, $\kappa = 0 . 7 5 , \kappa ^ { * } = 0 . 5 5$ with ϕ<sub>C</sub> proportional to $( x - 0 . 3 5 ) ^ { \ell }$ on [0.35, 2], for ℓ = 0 in the middle panel and $\ell = 1$ in the right panel. The late plateaus in these two panels are numerical precision floors.

(ii) High-dimensional limit. Assume moreover that Assumption 1 holds, and that $k \sim$ αd for some $\alpha \in ( 0 , \operatorname* { m i n } ( \kappa , \beta \kappa ^ { * } ) )$ . Let:

$$
x _ { \alpha } = \operatorname* { s u p } { \Big \{ } x > 0 , { \tilde { \mu } } _ { C } { \big ( } ( x , \infty ) { \big ) } = \alpha { \Big \} } .
$$

Consider:

$$
\Delta _ { \alpha } ^ { \mathsf { a } } ( t ) ^ { 2 } = \operatorname* { l i m } _ { \rho \to 0 } \operatorname* { s u p } _ { d \to \infty } \frac { 1 } { d } \big \| \mathbf { P } _ { k , \rho } ^ { \mathsf { a } } ( t ) - \mathbf { P } _ { k } ^ { C , \mathsf { a } } \big \| _ { F } ^ { 2 } .
$$

Then, with probability one:

– If there exists $\epsilon _ { \alpha } > 0$ such that $\tilde { \mu } _ { C } \big ( ( x _ { \alpha } - \epsilon _ { \alpha } , x _ { \alpha } ) \big ) = 0 , t h e$ n

$$
\Delta _ { \alpha } ^ { a } ( t ) ^ { 2 } \underset { t  \infty } { = } O ( e ^ { - 2 \epsilon _ { \alpha } t } ) .\tag{26}
$$

– In all cases,

$$
\Delta _ { \alpha } ^ { \mathsf { a } } ( t ) ^ { 2 } \underset { t  \infty } { = } O ( t ^ { - 1 } ) .\tag{27}
$$

We prove this theorem in Appendix D.3. The finite-dimensional result can be understood from Proposition 7: renormalizing the relevant matrices by $\rho$ allows us to observe how singular values are reordered at early timescales while remaining small. This ordering is exactly what makes the above singular vector convergence possible. The exponentially fast convergence is a direct consequence of the same proposition: for large t, a singular direction of C associated with a singular value σ is amplified by $e ^ { 2 \sigma t }$ . At the index k, its amplification relative to the lower modes is therefore of order $\bar { e ^ { 2 ( \sigma _ { k } - \sigma _ { k + 1 } ) t } }$ . This mechanism is the same as in filtered subspace iteration (Stewart, 1976; Saibaba, 2019), and is closely related in continuous time to the power flow (Absil et al., 2008). We also refer to Liu and Liu (2022) for similar rates in the convergence of the $\mathrm { O j a }$ flow.

The main additional content of Theorem 10 is its high-dimensional part. If a fixed $\mathrm { g a p }$ remains below the threshold $x _ { \alpha }$ , the exponentially fast convergence is preserved. On the other hand, when $\phi _ { C } ( x _ { \alpha } ) > 0$ , the gap $\sigma _ { k } - \sigma _ { k + 1 }$ typically vanishes in the high-dimensional limit: neighboring singular values accumulate around the threshold, and the convergence becomes a power law. This phenomenon and the associated asymptotics are very similar to Theorem 9 for the range projectors.

![](images/bcc51e71e859fd5873cc8fb278f4bdaeaca9009d1918e0ccc730e3302e58ee41.jpg)  
Figure 6: Colored curves are finite-dimensional gradient descent simulations of the dynamics (2), averaged over 10 repetitions, with $d = 5 0 0$ , stepsize $\eta = 1 0 ^ { - 3 } , \tau = 1 , \kappa = 0 . 5 5 , \kappa ^ { * } = 0 . 9 0$ $\alpha = 0 . 3 0$ , and decreasing $\rho .$ Each curve averages the left and right distances $d ^ { - 1 } \| \mathbf { P } _ { k , \rho } ^ { \mathsf { a } } ( t ) - \mathbf { P } _ { k } ^ { C , \mathsf { a } } \| _ { F } ^ { 2 }$ between the top-k singular projectors of $\mathbf { Z } ( t )$ and $\mathbf { C }$ , with $k = \alpha d$ . The black dashed curve is the $\rho \to 0$ limit of the same distance at $d = 5 0 0$ , computed from the linearized dynamics of Proposition 7. Gray dashed lines show the upper rates from Theorem 10, with prefactors chosen for visual comparison. Left panel: spectrum with a gap of width $\epsilon _ { \alpha } = 0 . 2$ below $x _ { \alpha }$ . Right panel: continuous spectrum with positive density at $x _ { \alpha }$ . Insets show the local spectral geometry at the cutof.

Figure 6 illustrates these two regimes. The black envelope is the $\rho \to 0$ projector distance computed from the linearized limit in Proposition 7. At finite $\rho ,$ the curves first follow this envelope and then leave it as they converge to zero. Decreasing $\rho$ delays this transition and reveals a longer part of the envelope. Gray dashed lines show the bounds obtained in the theorem. Although not tight, these bounds capture the essence of the phenomenon: a gap at the threshold $x _ { \alpha }$ induces exponentially fast convergence (left panel), while a continuous and positive density at the threshold degrades convergence into a power-law asymptotic (right panel).

## 5.2 Learning Phase

We now consider the learning phase. At the later timescale $t \asymp \log ( \rho ^ { - 1 } )$ , the singular vectors have already converged, while the singular values are learned progressively. We start by describing this phenomenon in finite dimension, and then study the high-dimensional limit.

## 5.2.1 Finite-Dimensional Saddle-to-Saddle Dynamics

We first describe the learning phase in finite dimension and recover a phenomenon of incremental learning, known as saddle-to-saddle dynamics.

Theorem 11 (Saddle-to-saddle dynamics). Let $s > 0$ , set $x _ { s } = ( 2 s ) ^ { - 1 }$ , and assume that no singular value of C equals $x _ { s }$ and that # $\left\{ i : \sigma _ { i } > x _ { s } \right\} \leq m$ . Then, with probability one with respect to the randomness of R, with $t _ { \rho } = s \log ( \rho ^ { - 1 } )$ , under Assumptions 1.1 and $1 . 4$ , the

following convergences hold:

$$
\begin{array} { r } { \begin{array} { r l } & { \mathbf { U } ( t _ { \rho } ) \mathbf { U } ( t _ { \rho } ) ^ { \top } \xrightarrow [ \rho  0 ] { } \mathcal { M } _ { s } \Big ( ( \mathbf { C C } ^ { \top } ) ^ { 1 / 2 } \Big ) , } \\ & { \mathbf { V } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top } \xrightarrow [ \rho  0 ] { } \mathcal { M } _ { s } \Big ( ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 } \Big ) , } \\ & { \qquad \mathbf { Z } ( t _ { \rho } ) \xrightarrow [ \rho  0 ] { } \mathcal { M } _ { s } ( \mathbf { C } ) , } \end{array} } \end{array}
$$

where $\mathcal { M } _ { s } ( x ) = x \mathbf { 1 } _ { x > x _ { s } }$ is applied spectrally to $( \mathbf { C C } ^ { \top } ) ^ { 1 / 2 }$ and $\left( \mathbf { C } ^ { \top } \mathbf { C } \right) ^ { 1 / 2 }$ , and pointwise to the singular values of C.

We prove this theorem in Appendix D.4. The proof reads the limits from the closed-form solution, and the only additional step checks that the intermediate limits are strict saddles of the factorized loss. This result is the saddle-to-saddle mechanism of Jacot et al. (2021), which has already been described in two-layer linear networks (Gidel et al., 2019; Varre et al., 2023; Bach, 2026). We recall it for completeness as we later show how this discrete process becomes continuous in the high-dimensional limit.

This result makes the spectral nature of the dynamics explicit. The mode associated with $\sigma _ { i }$ is activated when $s > ( 2 \sigma _ { i } ) ^ { - 1 }$ , so between two activation times the prediction is a truncated SVD of C, which is a saddle point of the loss. Near each saddle, the predictor appears frozen while the next mode grows from a size of order $\rho .$ The dynamics thus waits a time of order $\log ( \rho ^ { - 1 } )$ near each saddle, then moves to the next one in a time of order one.

At each saddle, the loss is carried by the singular values of C below $( 2 s ) ^ { - 1 }$ , as made precise in Theorem 13. Figure 7 shows this loss as a function of s for decreasing $\rho$ and two values of $d .$ In the variable $s ,$ the moves between saddles become sharper as $\rho$ decreases, which produces the steps of the figure.

![](images/7bb441a0fac0c7dab07cc4fed314b2e7f5b82c58d9b375b6a1726dd752c0b585.jpg)

![](images/16eb6496c3b018fe8ba70db9c3997591bd941954eb9c205f20dde3dcae56b900.jpg)  
Figure 7: Both panels show $d ^ { - 1 } \| \mathbf Z ( t ) - \mathbf C \| _ { F } ^ { 2 }$ against $s = t / \log ( \rho ^ { - 1 } )$ for decreasing $\rho ,$ with $d = 2 4$ on the left and $d = 9 6$ on the right. The black staircases are the corresponding finite-dimensional $\rho \to 0$ limits from Theorem 11. Each plateau is associated with a truncated SVD of $\mathbf { C } ,$ and each drop to the activation of another singular mode. Colored curves are computed from gradient descent simulations of the dynamics (2), with stepsize $\eta = 2 \times 1 0 ^ { - 3 }$ and parameters $\tau = 1$ $\kappa = 0 . 5 5$ , and $\kappa ^ { * } = 0 . 9 0$

## 5.2.2 Spectra at Logarithmic Times

We now consider the same logarithmic timescale after taking the high-dimensional limit first. The following theorem shows how the spectral mass progressively escapes zero to match the density ϕ .

Theorem 12 (Spectra at logarithmic times). Let $s > 0$ and $t _ { \rho } = s \log ( \rho ^ { - 1 } )$ . Denote $\begin{array} { r } { x _ { s } = \frac { 1 } { 2 s } } \end{array}$ . Consider the limiting Stieltjes transforms $\mathsf { i n } _ { t } ^ { U } , \mathsf { m } _ { t } ^ { V } , \mathsf { m } _ { t } ^ { Z }$ derived in Theorem 4. For every $w \in \bar { \mathbb { C } } \backslash \mathbb { R } ^ { + }$ , we have the limit:

$$
\mathfrak { m } _ { t _ { \rho } } ^ { U } ( w ) \xrightarrow [ \rho  0 ] { } - \frac { 1 } { w } ( 1 - \frac { 1 } { \tau } T _ { C } ( x _ { s } ) ) + \frac { \beta \kappa ^ { * } } { \tau } \int _ { x _ { s } } ^ { \infty } \frac { \phi _ { C } ( x ) \mathrm { d } x } { x - w } ,\tag{28}
$$

$$
\mathfrak { m } _ { t _ { \rho } } ^ { V } ( w ) \xrightarrow [ \rho  0 ] { } - \frac { 1 - T _ { C } ( x _ { s } ) } { w } + \beta \kappa ^ { * } \int _ { x _ { s } } ^ { \infty } \frac { \phi _ { C } ( x ) \mathrm { d } x } { x - w } ,\tag{29}
$$

$$
\mathfrak { m } _ { t _ { \rho } } ^ { Z } ( w ) \xrightarrow [ \rho  0 ] { } - \frac { 1 } { w } ( 1 - \frac { T _ { C } ( x _ { s } ) } { \beta } ) + \kappa ^ { * } \int _ { x _ { s } } ^ { \infty } \frac { \phi _ { C } ( x ) \mathrm { d } x } { x - w } ,\tag{30}
$$

where for $y > 0$

$$
T _ { C } ( y ) = \tilde { \mu } _ { C } ( ( y , \infty ) ) .
$$

This holds as long as $x _ { s } > x _ { * }$ , with $x _ { * }$ defined in Corollary 5.

For $x _ { s } > x _ { * } ,$ denote by $\mu _ { s }$ the weak limit of $\mu _ { t _ { \rho } ( s ) } ^ { Z }$ as $\rho \to 0$ , whose Stieltjes transform is given by equation (30). Then, the squared 2-Wasserstein distance between $\mu _ { s }$ and µ<sub>C</sub> is given by:

$$
W _ { 2 } ^ { 2 } ( \mu _ { s } , \mu _ { C } ) = \kappa ^ { * } \int _ { 0 } ^ { x _ { s } } x ^ { 2 } \phi _ { C } ( x ) \mathrm { d } x .\tag{31}
$$

We prove this in Appendix D.5. The Stieltjes transforms in Theorem 12 are already highdimensional limits: the limit $d \to \infty$ is taken first, followed by $\rho \to 0$ on the logarithmic timescale. The theorem is the high-dimensional counterpart of Theorem 11. It recovers the same threshold $x _ { s } = ( 2 s ) ^ { - 1 }$ , but the isolated finite-dimensional transitions now merge into a continuous spectral front. Indeed, a direction at spectral location $x > 0$ is amplified with magnitude $\rho e ^ { 2 x t }$ , which vanishes when $x < x _ { s }$ and diverges when $x > x _ { s }$ . As s increases, the front moves continuously from the right edge of the spectrum toward smaller singular values, similarly to the detection wave obtained by Lampinen and Ganguli (2019) from an initialization aligned with the data. Figure 8 follows this evolution through the positive density, the spectral mass at zero, and the corresponding loss.

It is interesting to remark that when considering the renormalized time variable $s ,$ the spectral front stops at the finite time $s _ { \mathrm { m a x } } = ( 2 x _ { * } ) ^ { - 1 }$ . We recall that $x _ { * }$ is the spectral cutof associated with the rank constraint of the prediction: it is either the left edge of the density $\phi _ { C }$ in the overparameterized case, or a larger threshold defined in equation (20) in the underparameterized case. In the latter case, for $s > s _ { \mathrm { m a x } }$ , the prediction is constrained by its rank and cannot learn more singular values.

The Wasserstein identity in equation (31) gives a simple interpretation of this spectral evolution. At the logarithmic timescale, the limiting prediction has already learned the singular values above $x _ { s }$ , and the remaining error is exactly the cost of transporting the extra mass at zero to the missing singular values below $x _ { s }$

![](images/6f287a33ce47ea7ad4d834a18e5c00acd2223cb106ab6762fe574ae4f36e76fd.jpg)  
Figure 8: Top panels: empirical positive singular-value densities of $\mathbf { Z } ( t )$ at four values of $s = t / \log ( \rho ^ { - 1 } )$ , compared with the $\rho \to 0$ limit from Theorem 12, the finite-ρ high-dimensional density from Theorem 4, and the target density $\kappa ^ { * } \phi _ { C }$ . The vertical lines mark the moving threshold $x _ { s }$ and the fixed rank-selection threshold $x _ { * }$ . Bottom ${ \it l e f t }$ panel: limiting mass at zero, made of a permanent atom due to the rank constraint and a waiting atom of singular values not learned yet, compared with the empirical mass below $x _ { s }$ . Bottom right panel: normalized prediction loss and its limit from Theorem 13. Orange dots and vertical lines identify the four selected values of s. The gradient descent simulations use $\rho = 1 0 ^ { - 4 0 } , d = 5 0 0$ , stepsize $\eta = 1 0 ^ { - 3 }$ $\tau = 1 , \kappa = 0 . 5 5$ , and $\kappa ^ { * } = 0 . 9 0$ , averaged over 10 repetitions.

## 5.2.3 Loss Function and Spectral Dynamics

To conclude our results in the small-initialization limit, we derive an exact expression of the normalized squared distance between the prediction $\mathbf { Z } ( t )$ and the cross-moment matrix C. Equation (4) shows that the distance $\| \mathbf { U V } ^ { \top } - \mathbf { C } \| _ { F } ^ { 2 }$ is twice the excess empirical loss. Therefore the study of this distance shows how the loss evolves during the dynamics.

Theorem 13 (Loss on the two timescales). Under Assumption 1, we have the almost-sure limits, for fixed $t \geq 0$ and $s > 0$

$$
\operatorname* { l i m } _ { \rho \to 0 } \operatorname* { l i m } _ { d \to \infty } \frac { 1 } { \operatorname* { m i n } ( p , d ) } \| \mathbf { Z } ( t ) - \mathbf { C } \| _ { F } ^ { 2 } = \kappa ^ { * } \int _ { 0 } ^ { \infty } x ^ { 2 } \phi _ { C } ( x ) \mathrm { d } x ,\tag{32}
$$

$$
\operatorname* { l i m } _ { \rho \to 0 } \operatorname* { l i m } _ { d \to \infty } \frac { 1 } { \operatorname* { m i n } ( p , d ) } \Big \| \mathbf { Z } ( s \log ( \rho ^ { - 1 } ) ) - \mathbf { C } \Big \| _ { F } ^ { 2 } = \kappa ^ { * } \int _ { 0 } ^ { x _ { s } } x ^ { 2 } \phi _ { C } ( x ) \mathrm { d } x ,\tag{33}
$$

whenever $x _ { s } = ( 2 s ) ^ { - 1 } > x _ { * }$

We prove this theorem in Appendix D.6. The proof computes separately the squared norm of the prediction and its inner product with C. This requires an additional random matrix result, given in Theorem 24.

At times $t = \Theta ( 1 )$ , the loss is the same as at initialization, although the singular spaces are already aligning, as shown in Theorem 8. It decreases only on the scale $t = s \log ( \rho ^ { - 1 } )$ , as the singular values of C above $x _ { s }$ are learned, and both phases are visible in Figures 2 and 8. On this scale, the limit in equation (33) coincides with the squared Wasserstein distance given in equation (31). This shows that the loss contains no alignment error, and is entirely carried by the singular values that have not yet been learned. We refer to Appendix D.6.4 for more details.

## 6 Conclusion and Perspectives

The main message of this paper is that learning directions and learning amplitudes can take place on diferent timescales. In two-layer linear networks with small initialization, the singular spaces of the predictor first align with those of C while the prediction and the loss remain at their initial values. The singular values grow only later, on a logarithmic timescale, following a threshold that moves down the spectrum of C. A training phase that looks inactive from the loss can therefore already contain substantial feature learning, and the loss alone is a poor clock for this process.

In the extensive-width regime, combining the closed-form solution with random matrix theory yields explicit limits, at all times, for the spectrum of the prediction and for the distances between the learned and target singular spaces. These limits describe the two successive phases, and show that the saddle-to-saddle transitions that occur in finite dimension merge into a continuous spectral front.

Training time as a rank constraint. At time $t = s \log ( \rho ^ { - 1 } )$ , the predictor keeps the singular values of C above $( 2 s ) ^ { - 1 }$ and discards the others, until the width becomes the limiting factor. Training time thus controls the efective rank of the predictor, while the width fixes its maximal rank. A similar efect is well known in linear regression, where gradient descent fits the strongest directions of the data first, so that early stopping acts as a spectral regularization (Yao et al., 2007; Ali et al., 2019). In our setting, the factorized dynamics makes this filter sharp, as each direction is either fully learned or not learned at all.

This ordering helps generalization only if the directions learned are associated with a genuine signal in the data rather than sampling noise. In a spiked model, where a strong enough low-rank signal produces outliers outside a random bulk (Baik et al., 2005; Benaych-Georges and Nadakuditi, 2011), the silent phase would first orient the network toward the informative directions, and the spectral front would learn their amplitudes before reaching the bulk, as found by Lampinen and Ganguli (2019). Combined with our dynamical description, such a model could indicate when to stop training, and how much of the noise enters the predictor before that time.

Beyond the exactly solvable setting. Our analysis relies on whitened inputs, two layers and an isotropic imbalance. Under these assumptions, the whole trajectory depends on a single fixed matrix C, whose spectrum fixes in advance the order in which directions are learned. For a general input covariance Σ, the dynamics depends on both Σ and C, whose singular directions difer in general. This spectral structure is then lost, and the learning order may depend on the trajectory itself.

Depth also afects the timescales: for a balanced mode of a depth-L network with $L > 2$ initialized at scale ${ \sqrt { \rho } } ,$ the dynamics of Saxe et al. (2014) gives a time of order $\rho ^ { - ( L - 2 ) / 2 }$ before the amplitude becomes macroscopic, so that the logarithmic timescale is specific to two layers. This analysis, however, starts from directions that are already aligned. Without this assumption, the predictor of a deep linear network and its singular values follow explicit equations for balanced initializations (Arora et al., 2018, 2019). For a single output, Atanasov et al. (2022) observed numerically that the alignment still precedes the growth of the amplitude, but no longer on a separate timescale. Characterizing rigorously the ordering of alignment and amplitude growth in deeper networks remains open.

The high-dimensional regime without a closed form. In both of the generalizations discussed above, the closed-form solution on which our high-dimensional analysis relies is no longer available. Whether the two phases identified here persist in the extensive-width regime is therefore an open question. Among high-dimensional tools, dynamical mean-field theory is a natural candidate: it reduces the training dynamics to a few correlation and response functions, and already describes deep linear networks whose width, input dimension and number of samples grow together (Bordelon and Pehlevan, 2025), including how outliers leave the bulk of the weight spectra (Lauditi et al., 2026). Extending these analyses to small initializations would be a natural way to address this question, with the two-layer linear network as an exactly solvable reference.

Gradient descent and learning schedules. From a practical viewpoint, early alignment may matter when training is tuned by monitoring the loss. At finite stepsize, the saddle-to-saddle dynamics survives for small steps (Gidel et al., 2019), whereas larger steps can change the implicit bias or create oscillations near the edge of stability (Even et al., 2023; Ghosh et al., 2025). A stepsize or schedule chosen from the loss alone may therefore perturb the alignment phase, which the loss does not see. Understanding how a learning schedule acts on each of the two timescales, preserving the alignment while accelerating the later growth of the amplitudes, would make this separation of timescales useful in practice.

## Funding

This work was supported by the French government, managed by the Agence Nationale de la Recherche, under the France 2030 program (PR[AI]RIE-PSAI) and the Investissements d’avenir program (PRAIRIE 3IA Institute) [ANR-23-IACL-0008 to F.B. and G.B., ANR-19-P3IA-0001].

## Acknowledgments

S.M. used large language models (Anthropic’s Claude Opus and OpenAI’s GPT-5.6) to improve the wording, find references, and help write and check the code for the numerical experiments. These models also helped review the proofs, identify gaps in the arguments, and occasionally find simpler approaches. For instance, they suggested that the high-dimensional part of Theorem 10 does not require an exact formula for the distance between singular subspaces. They also helped identify and understand the complex analysis tools used to prove convergence of Stieltjes transforms. We checked all results, proofs and code and take full responsibility for the content of this article.

## Data Availability

The code is available at https://github.com/simonmartin15/silent-alignment-linearnetworks. The data underlying this article are available in Zenodo, at https://doi.org/ 10.5281/zenodo.23435597.

## References

Absil, P.-A., Sepulchre, R., and Mahony, R. (2008). Continuous-time subspace flows related to the symmetric eigenproblem. Pacific Journal of Optimization, 4(2):179–194.

Ali, A., Kolter, J. Z., and Tibshirani, R. J. (2019). A continuous-time view of early stopping for least squares regression. In International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 1370–1378. PMLR.

Allez, R. and Bouchaud, J.-P. (2014). Eigenvector dynamics under free addition. Random Matrices: Theory and Applications, 3(3):1450010.

Anderson, G. W., Guionnet, A., and Zeitouni, O. (2010). An Introduction to Random Matrices, volume 118 of Cambridge Studies in Advanced Mathematics. Cambridge University Press.

Arnaboldi, L., Stephan, L., Krzakala, F., and Loureiro, B. (2023). From high-dimensional & mean-field dynamics to dimensionless ODEs: A unifying approach to SGD in two-layers networks. In Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 1199–1227. PMLR.

Arora, S., Cohen, N., and Hazan, E. (2018). On the optimization of deep networks: Implicit acceleration by overparameterization. In International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 244–253. PMLR.

Arora, S., Cohen, N., Hu, W., and Luo, Y. (2019). Implicit regularization in deep matrix factorization. In Advances in Neural Information Processing Systems, volume 32.

Atanasov, A., Bordelon, B., and Pehlevan, C. (2022). Neural networks as kernel learners: The silent alignment efect. In International Conference on Learning Representations.

Ba, J., Erdogdu, M. A., Suzuki, T., Wang, Z., Wu, D., and Yang, G. (2022). High-dimensional asymptotics of feature learning: How one gradient step improves the representation. In Advances in Neural Information Processing Systems, volume 35, pages 37932–37946.

Bach, F. (2026). Closed-form dynamics beyond quadratics. Blog post, https://francisbach. com/closed-form-dynamics/.

Bah, B., Rauhut, H., Terstiege, U., and Westdickenberg, M. (2022). Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers. Information and Inference: A Journal of the IMA, 11(1):307–353.

Bai, Z. and Silverstein, J. W. (2010). Spectral Analysis of Large Dimensional Random Matrices. Springer Series in Statistics. Springer, second edition.

Bai, Z. D. and Yin, Y. Q. (1993). Limit of the smallest eigenvalue of a large dimensional sample covariance matrix. The Annals of Probability, 21(3):1275–1294.

Baik, J., Ben Arous, G., and Péché, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. The Annals of Probability, 33(5):1643–1697.

Bak, J. and Newman, D. J. (2010). Complex Analysis. Undergraduate Texts in Mathematics. Springer, third edition.

Ben Arous, G., Erdogdu, M. A., Vural, N. M., and Wu, D. (2025). Learning quadratic neural

networks in high dimensions: SGD dynamics and scaling laws. In Advances in Neural Information Processing Systems, volume 38, pages 162874–162961.

Benaych-Georges, F. and Nadakuditi, R. R. (2011). The eigenvalues and eigenvectors of finite, low rank perturbations of large random matrices. Advances in Mathematics, 227(1):494–521.

Bengio, Y., Courville, A., and Vincent, P. (2013). Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8):1798– 1828.

Bietti, A., Bruna, J., and Pillaud-Vivien, L. (2025). On learning Gaussian multi-index models with gradient flow part I: General properties and two-timescale learning. Communications on Pure and Applied Mathematics, 78(12):2354–2435.

Bietti, A., Bruna, J., Sanford, C., and Song, M. J. (2022). Learning single-index models with shallow neural networks. In Advances in Neural Information Processing Systems, volume 35, pages 9768–9783.

Bodin, A. and Macris, N. (2023). Gradient flow on extensive-rank positive semi-definite matrix denoising. In 2023 IEEE Information Theory Workshop (ITW), pages 365–370.

Bordelon, B. and Pehlevan, C. (2022). Self-consistent dynamical field theory of kernel evolution in wide neural networks. In Advances in Neural Information Processing Systems, volume 35, pages 32240–32256.

Bordelon, B. and Pehlevan, C. (2025). Deep linear network training dynamics from random initialization: Data, width, depth, and hyperparameter transfer. In International Conference on Machine Learning.

Boursier, E. and Flammarion, N. (2025). Early alignment in two-layer networks training is a two-edged sword. Journal of Machine Learning Research, 26(183):1–75.

Bun, J., Bouchaud, J.-P., and Potters, M. (2018). Overlaps between eigenvectors of correlated random matrices. Physical Review E, 98(5):052145.

Chikuse, Y. (2003). Statistics on Special Manifolds, volume 174 of Lecture Notes in Statistics. Springer.

Chizat, L. and Bach, F. (2018). On the global convergence of gradient descent for overparameterized models using optimal transport. In Advances in Neural Information Processing Systems, volume 31.

Chizat, L., Oyallon, E., and Bach, F. (2019). On lazy training in diferentiable programming. In Advances in Neural Information Processing Systems, volume 32, pages 2937–2947.

Chou, H.-H., Gieshof, C., Maly, J., and Rauhut, H. (2024). Gradient descent for deep matrix factorization: Dynamics and implicit bias towards low rank. Applied and Computational Harmonic Analysis, 68:101595.

Damian, A., Lee, J., and Soltanolkotabi, M. (2022). Neural networks can learn representations with gradient descent. In Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 5413–5452. PMLR.

Dominé, C. C. J., Anguita, N., Proca, A. M., Braun, L., Kunin, D., Mediano, P. A. M., and

Saxe, A. M. (2025). From lazy to rich: Exact learning dynamics in deep linear networks. In International Conference on Learning Representations.

Dominé, C. C. J., Braun, L., Fitzgerald, J. E., and Saxe, A. M. (2023). Exact learning dynamics of deep linear networks with prior knowledge. Journal of Statistical Mechanics: Theory and Experiment, 2023(11):114004.

Du, S. S., Hu, W., and Lee, J. D. (2018). Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced. In Advances in Neural Information Processing Systems, volume 31.

Even, M., Pesme, S., Gunasekar, S., and Flammarion, N. (2023). (S)GD over diagonal linear networks: Implicit bias, large stepsizes and edge of stability. In Advances in Neural Information Processing Systems, volume 36, pages 29406–29448.

Fukumizu, K. (1998). Efect of batch learning in multilayer neural networks. In Proceedings of the Fifth International Conference on Neural Information Processing (ICONIP’98), pages 67–70.

Ghosh, A., Kwon, S. M., Wang, R., Ravishankar, S., and Qu, Q. (2025). Learning dynamics of deep matrix factorization beyond the edge of stability. In International Conference on Learning Representations.

Gidel, G., Bach, F., and Lacoste-Julien, S. (2019). Implicit regularization of discrete gradient dynamics in linear neural networks. In Advances in Neural Information Processing Systems, volume 32.

Gissin, D., Shalev-Shwartz, S., and Daniely, A. (2020). The implicit bias of depth: How incremental learning drives generalization. In International Conference on Learning Representations.

Goldt, S., Advani, M. S., Saxe, A. M., Krzakala, F., and Zdeborová, L. (2020). Dynamics of stochastic gradient descent for two-layer neural networks in the teacher–student setup. Journal of Statistical Mechanics: Theory and Experiment, 2020(12):124010.

Gunasekar, S., Woodworth, B. E., Bhojanapalli, S., Neyshabur, B., and Srebro, N. (2017). Implicit regularization in matrix factorization. In Advances in Neural Information Processing Systems, volume 30.

Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288.

Jacot, A., Gabriel, F., and Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31.

Jacot, A., Ged, F., Şimşek, B., Hongler, C., and Gabriel, F. (2021). Saddle-to-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity. arXiv preprint arXiv:2106.15933.

Ji, Z. and Telgarsky, M. (2019). Gradient descent aligns the layers of deep linear networks. In International Conference on Learning Representations.

Karkada, D., Simon, J. B., Bahri, Y., and DeWeese, M. R. (2025). Closed-form training dynamics

reveal learned features and linear structure in Word2Vec-like models. In Advances in Neural Information Processing Systems, volume 38, pages 4507–4540.

Kunin, D., Marchetti, G. L., Chen, F., Karkada, D., Simon, J. B., DeWeese, M. R., Ganguli, S., and Miolane, N. (2025). Alternating gradient flows: A theory of feature learning in two-layer neural networks. In Advances in Neural Information Processing Systems, volume 38, pages 4951–4998.

Lampinen, A. K. and Ganguli, S. (2019). An analytic theory of generalization dynamics and transfer learning in deep linear networks. In International Conference on Learning Representations.

Lauditi, C., Pehlevan, C., and Bordelon, B. (2026). Spectral dynamics in deep networks: Feature learning, outlier escape, and learning rate transfer. arXiv preprint arXiv:2605.07870.

Ledoit, O. and Péché, S. (2011). Eigenvectors of some large sample covariance matrix ensembles. Probability Theory and Related Fields, 151(1–2):233–264.

LeJeune, D., Patil, P., Javadi, H., Baraniuk, R. G., and Tibshirani, R. J. (2024). Asymptotics of the sketched pseudoinverse. SIAM Journal on Mathematics of Data Science, 6(1):199–225.

Lewis, A. S. (1995). The convex analysis of unitarily invariant matrix functions. Journal of Convex Analysis, 2(1–2):173–183.

Li, Z., Luo, Y., and Lyu, K. (2021). Towards resolving the implicit bias of gradient descent for matrix factorization: Greedy low-rank learning. In International Conference on Learning Representations.

Liu, J.-G. and Liu, Z. (2022). Convergence of Oja’s online principal component flow. arXiv preprint arXiv:2202.11308.

Maillard, A., Troiani, E., Martin, S., Zdeborová, L., and Krzakala, F. (2024). Bayes-optimal learning of an extensive-width neural network from quadratically many samples. In Advances in Neural Information Processing Systems, volume 37, pages 82085–82132.

Marcotte, S., Gribonval, R., and Peyré, G. (2023). Abide by the law and follow the flow: Conservation laws for gradient flows. In Advances in Neural Information Processing Systems, volume 36, pages 63210–63221.

Martin, S., Bach, F., and Biroli, G. (2024). On the impact of overparameterization on the training of a shallow neural network in high dimensions. In International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 3655–3663. PMLR.

Martin, S., Biroli, G., and Bach, F. (2026). High-dimensional analysis of gradient flow for extensive-width quadratic neural networks. Journal of Machine Learning Research, 27(136):1– 182.

Mei, S. and Montanari, A. (2022). The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics, 75(4):667–766.

Min, H., Mallada, E., and Vidal, R. (2024). Early neuron alignment in two-layer ReLU networks with small initialization. In International Conference on Learning Representations.

Min, H. and Vidal, R. (2025). Understanding incremental learning with closed-form solution to gradient flow on overparameterized matrix factorization. arXiv preprint arXiv:2508.20344.

Moniri, B. and Hassani, H. (2026). Feature learning in linear-width two-layer networks: Two vs. one step of gradient descent. arXiv preprint arXiv:2605.17767.

Montanari, A. and Wang, Z. (2026). Phase transitions for feature learning in neural networks. arXiv preprint arXiv:2602.01434.

Oja, E. (1982). Simplified neuron model as a principal component analyzer. Journal of Mathematical Biology, 15(3):267–273.

Pesme, S. and Flammarion, N. (2024). Saddle-to-saddle dynamics in diagonal linear networks. Journal of Statistical Mechanics: Theory and Experiment, 2024(10):104016.

Rahimi, A. and Recht, B. (2007). Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems, volume 20.

Refinetti, M., Goldt, S., Krzakala, F., and Zdeborová, L. (2021). Classifying high-dimensional Gaussian mixtures: Where kernel methods fail and neural networks succeed. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8936–8947. PMLR.

Saad, D. and Solla, S. A. (1995). On-line learning in soft committee machines. Physical Review E, 52(4):4225.

Saibaba, A. K. (2019). Randomized subspace iteration: Analysis of canonical angles and unitarily invariant norms. SIAM Journal on Matrix Analysis and Applications, 40(1):23–48.

Santambrogio, F. (2015). Optimal Transport for Applied Mathematicians: Calculus of Variations, PDEs, and Modeling. Birkhäuser, Cham.

Saxe, A. M., McClelland, J. L., and Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In International Conference on Learning Representations.

Schif, J. L. (1993). Normal Families. Universitext. Springer.

Smale, S. (1963). Stable manifolds for diferential equations and difeomorphisms. Annali della Scuola Normale Superiore di Pisa - Scienze Fisiche e Matematiche, Ser. 3, 17(1–2):97–116.

Soltanolkotabi, M., Stöger, D., and Xie, C. (2023). Implicit balancing and regularization: Generalization and convergence guarantees for overparameterized asymmetric matrix sensing. In Conference on Learning Theory, pages 5140–5142.

Stewart, G. W. (1976). Simultaneous iteration for computing invariant subspaces of non-Hermitian matrices. Numerische Mathematik, 25(2):123–136.

Stöger, D. and Soltanolkotabi, M. (2021). Small random initialization is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction. In Advances in Neural Information Processing Systems, volume 34, pages 23831–23843.

Tarmoun, S., França, G., Haefele, B. D., and Vidal, R. (2021). Understanding the dynamics of gradient flow in overparameterized linear models. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 10153–10161. PMLR.

Tropp, J. A. (2015). An introduction to matrix concentration inequalities. Foundations and Trends in Machine Learning, 8(1–2):1–230.

Tsuzuki, D. and Ohki, K. (2025). Global convergence of Oja’s component flow for general square matrices and its applications. arXiv preprint arXiv:2510.00801.

Tu, Z., Aranguri, S., and Jacot, A. (2024). Mixed dynamics in linear networks: Unifying the lazy and active regimes. In Advances in Neural Information Processing Systems, volume 37, pages 106059–106104.

Varre, A. V., Vladarean, M.-L., Pillaud-Vivien, L., and Flammarion, N. (2023). On the spectral bias of two-layer linear networks. In Advances in Neural Information Processing Systems, volume 36, pages 64380–64414.

Yan, W.-Y., Helmke, U., and Moore, J. B. (1994). Global analysis of Oja’s flow for neural networks. IEEE Transactions on Neural Networks, 5(5):674–683.

Yang, G. and Hu, E. J. (2021). Tensor Programs IV: Feature learning in infinite-width neural networks. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11727–11737. PMLR.

Yao, Y., Rosasco, L., and Caponnetto, A. (2007). On early stopping in gradient descent learning. Constructive Approximation, 26(2):289–315.

Yin, Y. Q., Bai, Z. D., and Krishnaiah, P. R. (1988). On the limit of the largest eigenvalue of the large dimensional sample covariance matrix. Probability Theory and Related Fields, 78(4):509–521.

Zhang, Y., Saxe, A., and Latham, P. E. (2026). Saddle-to-saddle dynamics explains a simplicity bias across neural network architectures. In International Conference on Learning Representations.

## A Prerequisite

## A.1 Random Matrix Theory

We provide some useful definitions in random matrix theory, and refer to Anderson et al. (2010);   
Bai and Silverstein (2010) for standard references.

Spectral measures. If $\mathbf { A } \in { \cal S } _ { d } ( \mathbb { R } )$ has eigenvalues $\lambda _ { 1 } , \ldots , \lambda _ { d } ,$ its empirical spectral measure is given by

$$
\mu ^ { A } = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \delta _ { { \lambda } _ { i } } ,
$$

where $\delta _ { \lambda }$ is the Dirac mass at $\lambda \in \mathbb { R }$ . For a rectangular matrix $\mathbf { B } \in \mathbb { R } ^ { p \times d }$ , we use the empirical measure of its singular values $\sigma _ { 1 } , \ldots , \sigma _ { \operatorname* { m i n } ( p , d ) }$

$$
\mu ^ { B } = \frac { 1 } { \operatorname* { m i n } ( p , d ) } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( p , d ) } \delta _ { \sigma _ { i } } .
$$

Additionally, we write $\mu _ { d } \Rightarrow \mu$ for weak convergence, meaning convergence against every bounded continuous test function.

Stieltjes transforms. The Stieltjes transform of a probability measure $\mu$ is:

$$
{ \mathfrak { m } } ^ { \mu } ( w ) = \int { \frac { \mu ( { \mathrm { d } } x ) } { x - w } } ,
$$

and is defined for $w \in \mathbb { C } \setminus \operatorname { S u p p } ( \mu )$ . For $\mu = \mu ^ { A }$ , we write m $\mathbf { \Omega } ^ { A } = \mathfrak { m } ^ { \mu ^ { A } }$ , so that, for a symmetric matrix $\mathbf { A } \in { \cal S } _ { d } ( \mathbb { R } )$ , the Stieltjes transform of its empirical spectral measure is

$$
\mathfrak { m } ^ { A } ( w ) = \frac { 1 } { d } \mathrm { T r } \Big ( \big ( \mathbf { A } - w \mathbf { I } _ { d } \big ) ^ { - 1 } \Big ) .
$$

The Stieltjes transform characterizes its associated probability measure. Let us denote by ℑ the imaginary part of complex numbers.

Theorem 14 (Stieltjes–Perron formula, Bai and Silverstein (2010), Theorem B.8). Let $\mu$ be a probability measure and $a < b$ such that $\mu$ does not have an atom at a or b. Then:

$$
\mu ( ( a , b ) ) = \operatorname* { l i m } _ { \eta \downarrow 0 } \frac { 1 } { \pi } \int _ { a } ^ { b } \mathbb { S } \mathfrak { m } ^ { \mu } ( x + \mathrm { i } \eta ) \mathrm { d } x .
$$

Additionally, pointwise convergence of a sequence of Stieltjes transforms is equivalent to weak convergence of the associated measures.

Theorem 15 (Anderson et al. (2010), Theorem 2.4.4). Given a sequence of probability measures $( \mu _ { d } ) _ { d \geq 1 }$ and a probability measure µ, $\mu _ { d } \Rightarrow \mu$ if and only if ${ \mathfrak { m } } ^ { \mu _ { d } } ( w ) \xrightarrow [ d  \infty ] { } { \mathfrak { m } } ^ { \mu } ( w )$ for every $w \in \mathbb { C } \setminus \mathbb { R }$

Therefore, the convergence results of the Stieltjes transforms in Theorems 4 and 12 and Corollary 5 are to be understood as the convergence of the underlying spectral measures.

Pushforwards. If θ is a measurable map and $\mu$ is a measure, $\theta _ { \# } \mu$ denotes the pushforward of $\mu$ by θ. It is characterized by the following identity, for every bounded continuous function $f \colon$

$$
\int f ( y ) ( \theta _ { \# } \mu ) ( \mathrm { d } y ) = \int f ( \theta ( x ) ) \mu ( \mathrm { d } x ) .
$$

## A.2 Complex Analysis

Stieltjes transforms are functions defined on the complex plane. The convergence results we provide when analyzing them rely on standard complex analysis results that we gather here.

Lemma A.1. Let m be the Stieltjes transform of a real probability measure supported on a closed set $S \subset \mathbb { R }$ . Then m is holomorphic on $\mathbb { C } \setminus S$ . Moreover, given a compact subset $K$ of $\mathbb { C } \setminus S$ , we have:

$$
\operatorname* { s u p } _ { w \in K } | \mathfrak { m } ( w ) | \leq \frac { 1 } { \mathrm { d i s t } ( S , K ) } < \infty .
$$

Proof. Since $| x - w | \geq$ dist $( w , S )$ , the function $w \mapsto ( x - w ) ^ { - 1 }$ is holomorphic on $\mathbb { C } \setminus S$ Diferentiating under the integral proves holomorphy. Then by the triangle inequality, for every w $\notin \boldsymbol { S }$ , we have $| { \mathfrak { m } } ( w ) | \leq \mathrm { d i s t } ( w , S ) ^ { - 1 } < \infty$ . Taking the supremum over $w \in K$ gives the claim. □

We now follow with the Vitali–Porter theorem.

Theorem 16 (Vitali–Porter theorem, Schif (1993), Section 2.4). Let $( f _ { n } ) _ { n \geq 1 }$ be a sequence of holomorphic functions on a complex domain Ω, such that for all compact subsets $K \subset \Omega$

$$
\operatorname* { s u p } _ { w \in K } \operatorname* { s u p } _ { n \geq 1 } | f _ { n } ( w ) | < \infty .
$$

A sequence satisfying this bound is called locally bounded. Assume that on a set $E \subset \Omega$ with an accumulation point in Ω, $f _ { n } ( w )$ converges as $n \to \infty$ . Then $f _ { n }$ converges uniformly on compact subsets of Ω to a holomorphic function.

We now mention the identity theorem, which gives global equality of two holomorphic functions matching on a smaller subset.

Theorem 17 (Identity theorem, Bak and Newman (2010), Theorem 6.9). Let f and g be holomorphic on a complex domain Ω. $I f = g$ on a subset $o f \Omega$ having an accumulation point in Ω, then $f = g$ on Ω.

These two theorems lead to a standard corollary that we use throughout the paper.

Corollary 18. Consider a sequence of holomorphic functions $( f _ { n } ) _ { n \geq 1 }$ satisfying the assumptions of Theorem 16. If f is a holomorphic function on Ω and $f _ { n } ( w ) \xrightarrow [ n \to \infty ] { } f ( w )$ for all $w \in E$ , then $f _ { n }$ converges to f uniformly on compact subsets of Ω.

Proof. By Theorem 16, the locally bounded sequence converges uniformly on compact subsets of Ω to some holomorphic function $\tilde { f } .$ . On $E ,$ the pointwise limit gives $\tilde { f } = f$ , hence Theorem 17 yields $\tilde { f } = f$ on all of Ω. □

Throughout the paper, we use this corollary to prove convergence of Stieltjes transforms. This result shows that it is enough to establish convergence on a smaller subset. Typically we use $\Omega = \mathbb { C } \backslash \mathbb { R } ^ { + }$ and $E = ( - \infty , 0 )$ . When the above sequence of functions is indexed by a continuous parameter, the above result is generalized by considering sequences of parameters converging to the desired limit.

## A.3 Self-Consistent Equations

Several proofs in the paper involve studying limits of functions computed self-consistently through an integral against a probability measure. Such equations are of the form:

$$
1 = \int \frac { \mu ( \mathrm { d } x ) } { 1 - \kappa + h _ { \lambda } a _ { \lambda } ( x ) } ,\tag{34}
$$

where $\mu$ is a probability measure, $a \gimel \geq 0$ , and $h _ { \lambda }$ is the solution of the previous equation. In the following, the parameter λ will be either the initialization scale $\rho \to 0$ or the time $t \to \infty$

Proposition 19. Let µ be a probability measure and $\kappa \in ( 0 , 1 )$ . Consider $\Lambda \subset \mathbb { R } ^ { + }$ with a limit point $\lambda _ { \infty } \in [ 0 , + \infty ]$ and, for every $\lambda \in \Lambda$ , a measurable function $a _ { \lambda } \colon \mathbb { R } \to \mathbb { R } ^ { + }$

## 1. Assume that

$$
\mu \Big ( \{ x \in \mathbb { R } , a _ { \lambda } ( x ) = 0 \} \Big ) < 1 - \kappa ,
$$

for every $\lambda \in \Lambda$ . Then equation (34) admits a unique solution $h _ { \lambda }$ in $( 0 , \infty )$

2. Let $c _ { \lambda } > 0$ be such that, for µ-almost every $x ,$

$$
c _ { \lambda } a _ { \lambda } ( x ) \xrightarrow [ \lambda \to \lambda _ { \infty } ] { } a _ { \infty } ( x ) ,
$$

where the limit can $b e + \infty$ . For $u > 0$ , define the limit function:

$$
\Phi ( u ) = \int \frac { \mu ( \mathrm { d } x ) } { 1 - \kappa + u a _ { \infty } ( x ) } .
$$

Then, consider the cases:

(i) $I f \Phi ( u ) = 1$ admits a unique solution in $( 0 , \infty )$ , let $u _ { \infty }$ be this solution.

(ii) $I f \Phi ( u ) < 1$ for every $u > 0$ , let $u _ { \infty } = 0$

(iii) $I f \Phi ( u ) > 1$ for every $u > 0$ , let $u _ { \infty } = \infty$

In these three cases, we have:

$$
\frac { h _ { \lambda } } { c _ { \lambda } } \xrightarrow [ \lambda  \lambda _ { \infty } ] { } u _ { \infty } .
$$

3. Moreover, if $u _ { \infty } \in ( 0 , \infty )$ and a family $( f _ { \lambda } ) _ { \lambda \in \Lambda }$ is uniformly bounded and converges to f µ-almost everywhere as $\lambda \to \lambda _ { \infty }$ , then

$$
\int \frac { f _ { \lambda } ( x ) } { 1 - \kappa + h _ { \lambda } a _ { \lambda } ( x ) } \mu ( \mathrm { d } x ) \xrightarrow [ \lambda  \lambda _ { \infty } ] { } \int \frac { f ( x ) } { 1 - \kappa + u _ { \infty } a _ { \infty } ( x ) } \mu ( \mathrm { d } x ) .\tag{35}
$$

Proof. For 1., let $c _ { \lambda } > 0$ be arbitrary, and use the scaling specified in 2. for the remaining claims. For $v \geq 0$ , we work directly with the function:

$$
\Phi _ { \lambda } ( v ) = \int \frac { \mu ( \mathrm { d } x ) } { 1 - \kappa + v c _ { \lambda } a _ { \lambda } ( x ) } .
$$

This function is continuous and decreasing, with

$$
\Phi _ { \lambda } ( 0 ) = \frac { 1 } { 1 - \kappa } > 1 , \qquad \operatorname * { l i m } _ { v  \infty } \Phi _ { \lambda } ( v ) = \frac { \mu ( a _ { \lambda } = 0 ) } { 1 - \kappa } < 1 .
$$

It is strictly decreasing because $a _ { \lambda } > 0$ on a set of positive measure. Hence the equation $\Phi _ { \lambda } ( v ) = 1$ has a unique solution. By equation (34), this solution is $u _ { \lambda } = h _ { \lambda } / c _ { \lambda }$ , which proves 1.

Let $v > 0$ be fixed. Then:

$$
\frac { 1 } { 1 - \kappa + v c _ { \lambda } a _ { \lambda } ( x ) } \xrightarrow [ { \lambda \to \lambda _ { \infty } } ] { } \frac { 1 } { 1 - \kappa + v a _ { \infty } ( x ) } ,
$$

µ-almost everywhere. Moreover, the integrand is upper-bounded by $( 1 - \kappa ) ^ { - 1 }$ , independently of $\lambda$ and x. Dominated convergence guarantees that for every $v > 0$

$$
\Phi _ { \lambda } ( v ) \xrightarrow [ \lambda \to \lambda _ { \infty } ] { } \Phi ( v ) .\tag{36}
$$

Let us first assume that $u _ { \infty } \in ( 0 , \infty )$ . Consider $\epsilon \in ( 0 , u _ { \infty } )$ . Since $\Phi$ is decreasing, we have:

$$
\Phi ( u _ { \infty } - \epsilon ) > 1 > \Phi ( u _ { \infty } + \epsilon ) .
$$

By the previous convergence, this also holds for $\Phi _ { \lambda }$ when λ is suficiently close to $\lambda _ { \infty }$ . Since $\Phi _ { \lambda } ( u _ { \lambda } ) = 1$ , monotonicity of $\Phi _ { \lambda }$ guarantees that $u _ { \infty } - \epsilon < u _ { \lambda } < u _ { \infty } + \epsilon$ . Hence $u _ { \lambda }$ converges to $u _ { \infty } ,$ proving point (i) of 2.

The same argument applies when $u _ { \infty } = 0$ or $u _ { \infty } = + \infty$ . In these two cases, for any $\epsilon > 0$ or $M > 0$ , respectively, one has $\Phi _ { \lambda } ( \epsilon ) < 1$ or $\Phi _ { \lambda } ( M ) > 1$ when λ is suficiently close to $\lambda _ { \infty }$ . The monotonicity of $\Phi _ { \lambda }$ then proves (ii) and (iii).

Finally, consider $( f _ { \lambda } ) _ { \lambda \in \Lambda }$ as in the proposition and $u _ { \infty } \in ( 0 , \infty )$ . We have the convergence µ-almost everywhere:

$$
\frac { f _ { \lambda } ( x ) } { 1 - \kappa + h _ { \lambda } a _ { \lambda } ( x ) } \xrightarrow [ \lambda  \lambda _ { \infty } ] { } \frac { f ( x ) } { 1 - \kappa + u _ { \infty } a _ { \infty } ( x ) } .
$$

Additionally,

$$
\left| \frac { f _ { \lambda } ( x ) } { 1 - \kappa + h _ { \lambda } a _ { \lambda } ( x ) } \right| \le \frac { 1 } { 1 - \kappa } \operatorname* { s u p } _ { \lambda \in \Lambda } \| f _ { \lambda } \| _ { \infty } .
$$

This bound is finite by assumption, therefore dominated convergence guarantees the result.

## B Finite-Dimensional Dynamics

In this section we prove the results stated in Section 3.

## B.1 Closed-Form Solution

We shall now prove Proposition 1. Starting from the dynamics (2) with $\pmb { \Sigma } = \mathbf { I } _ { d } .$ , we have:

$$
\begin{array} { r l } & { \dot { \mathbf { U } } ( t ) = - \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top } \mathbf { V } ( t ) + \mathbf { C V } ( t ) , } \\ & { \dot { \mathbf { V } } ( t ) = - \mathbf { V } ( t ) \mathbf { U } ( t ) ^ { \top } \mathbf { U } ( t ) + \mathbf { C } ^ { \top } \mathbf { U } ( t ) . } \end{array}
$$

Now, recall the definition of W(t). We have:

$$
\mathbf { W } ( t ) ^ { \top } \mathbf { W } ( t ) = \mathbf { V } ( t ) ^ { \top } \mathbf { V } ( t ) + \mathbf { U } ( t ) ^ { \top } \mathbf { U } ( t ) .
$$

Additionally, since the imbalance matrix Imb $\mathbf { \boldsymbol { \mathsf { \Lambda } } } ( t ) = \mathbf { \boldsymbol { \mathsf { V } } } ( t ) ^ { \top } \mathbf { \boldsymbol { \mathbf { V } } } ( t ) - \mathbf { \boldsymbol { \mathbf { U } } } ( t ) ^ { \top } \mathbf { \boldsymbol { \mathbf { U } } } ( t )$ remains constant along the dynamics, and equal to $2 \rho \mathbf { I } _ { m }$ as a consequence of Assumption 1.4, we have:

$$
\mathbf { V } ( t ) ^ { \top } \mathbf { V } ( t ) = \frac { \mathbf { W } ( t ) ^ { \top } \mathbf { W } ( t ) } { 2 } + \rho \mathbf { I } _ { m } , \qquad \mathbf { U } ( t ) ^ { \top } \mathbf { U } ( t ) = \frac { \mathbf { W } ( t ) ^ { \top } \mathbf { W } ( t ) } { 2 } - \rho \mathbf { I } _ { m } .
$$

Therefore:

$$
\begin{array} { r l } & { \dot { \mathbf { U } } ( t ) = - \frac { 1 } { 2 } \mathbf { U } ( t ) \mathbf { W } ( t ) ^ { \top } \mathbf { W } ( t ) - \rho \mathbf { U } ( t ) + \mathbf { C } \mathbf { V } ( t ) , } \\ & { \dot { \mathbf { V } } ( t ) = - \frac { 1 } { 2 } \mathbf { V } ( t ) \mathbf { W } ( t ) ^ { \top } \mathbf { W } ( t ) + \rho \mathbf { V } ( t ) + \mathbf { C } ^ { \top } \mathbf { U } ( t ) . } \end{array}
$$

Stacking the two equations gives the diferential equation (5) on $\mathbf { W } ( t )$ . Therefore ${ \bf Y } ( t ) =$ $\mathbf W ( t ) \mathbf W ( t ) ^ { \top }$ satisfies:

$$
\dot { \mathbf { Y } } ( t ) = \mathbf { A } _ { \rho } \mathbf { Y } ( t ) + \mathbf { Y } ( t ) \mathbf { A } _ { \rho } - \mathbf { Y } ( t ) ^ { 2 } .\tag{37}
$$

To prove equation (6), let ${ \bf Y } _ { 0 } ( t )$ be the solution displayed in this equation:

$$
\mathbf { Y } _ { 0 } ( t ) = e ^ { t \mathbf { A } _ { \rho } } \mathbf { W } _ { 0 } \mathbf { N } ( t ) ^ { - 1 } \mathbf { W } _ { 0 } ^ { \top } e ^ { t \mathbf { A } _ { \rho } } , \qquad \mathbf { N } ( t ) = \mathbf { I } _ { m } + \mathbf { W } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \mathbf { A } _ { \rho } } \mathrm { d } s \mathbf { W } _ { 0 } .
$$

Then $\mathbf { Y } _ { 0 } ( t = 0 ) = \mathbf { W } _ { 0 } \mathbf { W } _ { 0 } ^ { \top } = \mathbf { Y } ( t = 0 )$ and

$$
\dot { \mathbf { Y } } _ { 0 } ( t ) = \mathbf { A } _ { \rho } \mathbf { Y } _ { 0 } ( t ) + \mathbf { Y } _ { 0 } ( t ) \mathbf { A } _ { \rho } - e ^ { t \mathbf { A } _ { \rho } } \mathbf { W } _ { 0 } \mathbf { N } ( t ) ^ { - 1 } \dot { \mathbf { N } } ( t ) \mathbf { N } ( t ) ^ { - 1 } \mathbf { W } _ { 0 } ^ { \top } e ^ { t \mathbf { A } _ { \rho } } .
$$

Since $\dot { \mathbf { N } } ( t ) = \mathbf { W } _ { 0 } ^ { \top } e ^ { 2 t \mathbf { A } _ { \rho } } \mathbf { W } _ { 0 }$ , computing the last term gives equation (37) for ${ \bf Y } _ { 0 } ( t )$ . The proof follows by uniqueness of the solution given the initial condition.

## B.2 Block Expressions for the Solution

Recall that the matrices $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } , \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } , \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ can be recovered from equation (6). We exploit the block structure of $\mathbf { A } _ { \rho }$ and $\mathbf { W } _ { 0 }$ to obtain explicit expressions for these three matrices.

The key observation is that the matrix

$$
\mathbf { A } _ { \rho } = \left( \begin{array} { l l } { - \rho \mathbf { I } _ { p } } & { \mathbf { C } } \\ { \mathbf { C } ^ { \top } } & { \rho \mathbf { I } _ { d } } \end{array} \right) \in S _ { p + d } ( \mathbb { R } )
$$

satisfies the identity

$$
\mathbf { A } _ { \rho } ^ { 2 } = \binom { \mathbf { C C } ^ { \top } + \rho ^ { 2 } \mathbf { I } _ { p } } { \mathbf { 0 } } \mathbf { C } ^ { \top } \mathbf { C } + \rho ^ { 2 } \mathbf { I } _ { d } \bigg ) .\tag{38}
$$

Lemma B.1. Consider the setting and the assumptions of Proposition 1. Define $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ $\Gamma _ { \rho } = ( \mathbf { S } ^ { 2 } + \rho ^ { 2 } \mathbf { I } _ { d } ) ^ { 1 / 2 }$ , and $\gamma _ { \rho } ( x ) = \sqrt { x ^ { 2 } + \rho ^ { 2 } }$ . For $x \geq 0$ and $\gamma \geq \rho .$ , let

$$
\chi _ { t } ^ { U } ( x ) = \frac { x } { \gamma _ { \rho } ( x ) } \sinh { \big ( t \gamma _ { \rho } ( x ) \big ) } ,
$$

$$
\chi _ { t } ^ { V } ( x ) = \cosh \left( t \gamma _ { \rho } ( x ) \right) + \frac { \rho } { \gamma _ { \rho } ( x ) } \sinh \left( t \gamma _ { \rho } ( x ) \right) ,
$$

$$
q _ { t } ( \gamma ) = 1 + \frac { \rho } { \gamma } \sinh ( 2 t \gamma ) + \frac { \rho ^ { 2 } } { \gamma ^ { 2 } } \bigl ( \cosh ( 2 t \gamma ) - 1 \bigr ) .
$$

Then, we have the expressions, for all $t \geq 0$

$$
\begin{array} { r } { \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } \Big ( \mathbf { R } ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Big ) ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { C } ) ^ { \top } , } \end{array}
$$

$$
\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t } ^ { V } ( \mathbf { S } ) \mathbf { R } \Big ( \mathbf { R } ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Big ) ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t } ^ { V } ( \mathbf { S } ) ,
$$

$$
{ \bf U } ( t ) { \bf V } ( t ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t } ^ { U } ( { \bf C } ) { \bf R } \Big ( { \bf R } ^ { \top } q _ { t } ( { \bf r } _ { \rho } ) { \bf R } \Big ) ^ { - 1 } { \bf R } ^ { \top } \boldsymbol { \chi } _ { t } ^ { V } ( { \bf S } ) .
$$

Proof. We first compute $e ^ { t \mathbf { A } _ { \rho } }$ from ${ \bf A } _ { \rho } ^ { 2 }$ . We have, expanding the exponential and decomposing odd and even indices

$$
e ^ { t { \bf A } _ { \rho } } = \sum _ { k = 0 } ^ { \infty } \frac { t ^ { 2 k } } { ( 2 k ) ! } ( { \bf A } _ { \rho } ^ { 2 } ) ^ { k } + \sum _ { k = 0 } ^ { \infty } \frac { t ^ { 2 k + 1 } } { ( 2 k + 1 ) ! } { \bf A } _ { \rho } ( { \bf A } _ { \rho } ^ { 2 } ) ^ { k } .
$$

This leads to:

$$
e ^ { t \mathbf { A } _ { \rho } } = \cosh { ( t ( \mathbf { A } _ { \rho } ^ { 2 } ) ^ { 1 / 2 } ) } + \mathbf { A } _ { \rho } ( \mathbf { A } _ { \rho } ^ { 2 } ) ^ { - 1 / 2 } \sinh { ( t ( \mathbf { A } _ { \rho } ^ { 2 } ) ^ { 1 / 2 } ) } .
$$

Then, using the expression of $\mathbf { A } _ { \rho } ^ { 2 }$ and $\mathbf { W } _ { 0 }$ , we get:

$$
e ^ { t { \bf A } _ { \rho } } { \bf W } _ { 0 } = \sqrt { 2 \rho } \left( \begin{array} { c } { { { \bf C } { \bf \bf { T } } _ { \rho } ^ { - 1 } \sinh ( t { \bf { T } } _ { \rho } ) } } \\ { { \cosh ( t { \bf { T } } _ { \rho } ) + \rho { \bf { T } } _ { \rho } ^ { - 1 } \sinh ( t { \bf { T } } _ { \rho } ) } } \end{array} \right) { \bf R } = \sqrt { 2 \rho } \left( \begin{array} { c } { { \chi _ { t } ^ { U } ( { \bf { C } } ) } } \\ { { \chi _ { t } ^ { V } ( { \bf { S } } ) } } \end{array} \right) { \bf R } .\tag{39}
$$

The last equality follows directly from the SVD of C: if $\begin{array} { r } { { \mathbf C } = \sum _ { i } \sigma _ { i } { \mathbf u } _ { i } { \mathbf v } _ { i } ^ { \top } } \end{array}$ , then

$$
\mathbf { C } \Gamma _ { \rho } ^ { - 1 } \sinh ( t \mathbf { r } _ { \rho } ) = \sum _ { i } { \frac { \sigma _ { i } } { \sqrt { \sigma _ { i } ^ { 2 } + \rho ^ { 2 } } } } \sinh \left( t \sqrt { \sigma _ { i } ^ { 2 } + \rho ^ { 2 } } \right) \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } = \chi _ { t } ^ { U } ( \mathbf { C } ) .
$$

Now evaluating $e ^ { t \mathbf { A } _ { \rho } } \mathbf { W } _ { 0 }$ at $t = 2 s$ and contracting with $\mathbf { W } _ { 0 } .$ , we get:

$$
{ \bf W } _ { 0 } ^ { \top } e ^ { 2 s { \bf A } _ { \rho } } { \bf W } _ { 0 } = 2 \rho { \bf R } ^ { \top } \chi _ { 2 s } ^ { V } ( { \bf S } ) { \bf R } .
$$

Then, since ${ \mathbf { R } } ^ { \top } { \mathbf { R } } = { \mathbf { I } } _ { m }$

$$
\mathbf { I } _ { m } + \mathbf { W } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \mathbf { A } _ { \rho } } \mathrm { d } s \mathbf { W } _ { 0 } = \mathbf { R } ^ { \top } \left( \mathbf { I } _ { d } + 2 \rho \int _ { 0 } ^ { t } \chi _ { 2 s } ^ { V } ( \mathbf { S } ) \mathrm { d } s \right) \mathbf { R } .
$$

Now, we have, integrating $\chi _ { 2 s } ^ { V }$

$$
\int _ { 0 } ^ { t } \chi _ { 2 s } ^ { V } ( \mathbf { S } ) \mathrm { d } s = \frac { 1 } { 2 } \Gamma _ { \rho } ^ { - 1 } \sinh \left( 2 t \Gamma _ { \rho } \right) + \frac { \rho } { 2 } \Gamma _ { \rho } ^ { - 2 } \bigl ( \cosh \left( 2 t \Gamma _ { \rho } \right) - \mathbf { I } _ { d } \bigr ) .
$$

Therefore, we have:

$$
\mathbf I _ { m } + \mathbf { W } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \mathbf A _ { \rho } } \mathrm { d } s \mathbf { W } _ { 0 } = \mathbf { R } ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf { R } .
$$

Combining this identity with the solution in equation (6) and the expression of $e ^ { t \mathbf { A } _ { \rho } } \mathbf { W } _ { 0 }$ in equation (39), we get the desired results by identifying the three blocks of $\mathbf W ( t ) \mathbf W ( t ) ^ { \top }$ □

## B.3 SVD of the Prediction Matrix

Let us now prove Proposition 2. Before going on, we state a useful lemma describing the range of a random matrix R uniformly distributed on the Stiefel manifold St $( d , m )$

Lemma B.2. Let R be uniformly distributed on $\mathrm { S t } ( d , m )$ . Then, we have:

(i) Given $E \subset \mathbb { R } ^ { d }$ a deterministic subspace, we have dim $( \mathrm { I m } ( \mathbf { R } ) \cap E ) = \mathrm { m a x } ( m + \mathrm { d i m } ( E ) - d , 0 )$ almost surely.

(ii) Given $\left( \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { k } \right)$ a deterministic linearly independent family of $\mathbb { R } ^ { d }$ with $k \leq m$ , the family $( \mathbf { R } ^ { \top } \mathbf { v } _ { 1 } , \ldots , \mathbf { R } ^ { \top } \mathbf { v } _ { k } )$ is linearly independent almost surely.

Proof. By Chikuse (2003, Theorem 2.2.1), R has the same law as $\mathbf { G } ( \mathbf { G } ^ { \top } \mathbf { G } ) ^ { - 1 / 2 }$ , where $\mathbf { G } \in \mathbb { R } ^ { d ; }$ ×m has i.i.d. Gaussian entries. Both statements only depend on the law of R, so we may assume $\mathbf { R } = \mathbf { G } ( \mathbf { G } ^ { \top } \mathbf { G } ) ^ { - 1 / 2 }$ . For the first point, let $e = \dim ( E )$ , and let $\mathbf { V } _ { E } \in \mathbb { R } ^ { d \times ( d - e ) }$ be an orthonormal basis of $E ^ { \perp }$ . Then:

$$
\mathrm { d i m } \left( \mathrm { I m } ( \mathbf { R } ) \cap E \right) = m - \mathrm { r a n k } ( \mathbf { V } _ { E } ^ { \top } \mathbf { R } ) = m - \mathrm { r a n k } ( \mathbf { V } _ { E } ^ { \top } \mathbf { G } ) .
$$

Since $\mathbf { V } _ { E } ^ { \top } \mathbf { G }$ is a $( d - e ) \times m$ Gaussian matrix with i.i.d. entries, it has rank min $( d - e , m )$ almost surely. We then obtain (i).

Finally, let $\mathbf { V } = ( \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { k } ) \in \mathbb { R } ^ { d \times k }$ with full column rank and $k \leq m$ . Since $( \mathbf { G } ^ { \top } \mathbf { G } ) ^ { - 1 / 2 }$ is invertible,

$$
\operatorname { r a n k } ( \mathbf { R } ^ { \top } \mathbf { V } ) = \operatorname { r a n k } ( \mathbf { G } ^ { \top } \mathbf { V } ) .
$$

The matrix $\mathbf { G } ^ { \top } \mathbf { V }$ is an $m \times k$ Gaussian matrix whose m rows are i.i.d. $\mathcal { N } ( 0 , \mathbf { V } ^ { \top } \mathbf { V } )$ , hence it has rank k almost surely. Therefore $( \mathbf { R } ^ { \top } \mathbf { v } _ { 1 } , \ldots , \mathbf { R } ^ { \top } \mathbf { v } _ { k } )$ is linearly independent almost surely.

We are now ready to prove Proposition 2. The proof first proves the rank identities, and then the SVD for the prediction.

Step 1. Rank of the weights. We start by showing the rank identities (7) for $\mathbf { U } ( t ) , \mathbf { V } ( t )$ Recall that the imbalance matrix Imb $\mathbf { \boldsymbol { \mathsf { \Lambda } } } ( t ) = \mathbf { \boldsymbol { \mathsf { V } } } ( t ) ^ { \top } \mathbf { \boldsymbol { \mathbf { V } } } ( t ) - \mathbf { \boldsymbol { \mathbf { U } } } ( t ) ^ { \top } \mathbf { \boldsymbol { \mathbf { U } } } ( t )$ is constant along the dynamics, so that at all times, we have the identity:

$$
\mathbf { V } ( t ) ^ { \top } \mathbf { V } ( t ) = \mathbf { U } ( t ) ^ { \top } \mathbf { U } ( t ) + 2 \rho \mathbf { I } _ { m } \succeq 2 \rho \mathbf { I } _ { m } ,
$$

and therefore rank $\mathbf { V } ( t ) = m$ . For $\mathbf { U } ( t )$ , we go back to the expression of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ in Lemma B.1, and get:

$$
\operatorname { r a n k } \mathbf { U } ( t ) = \operatorname { r a n k } \big ( \chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } \big ) .
$$

Additionally:

$$
\chi _ { t } ^ { U } ( x ) = \frac { x } { \gamma _ { \rho } ( x ) } \sinh { \bigl ( t \gamma _ { \rho } ( x ) \bigr ) } ,
$$

so that $\chi _ { t } ^ { U } ( 0 ) = 0$ and $\chi _ { t } ^ { U } ( x ) > 0$ whenever $x > 0$ . Hence $\chi _ { t } ^ { U } ( \mathbf { C } )$ has the same kernel as C and the same rank $m ^ { * }$ . Therefore

$$
\mathrm { r a n k } \big ( \chi _ { t } ^ { U } ( { \mathbf { C } } ) { \mathbf { R } } \big ) = m - \mathrm { d i m } \big ( \mathrm { I m } ( { \mathbf { R } } ) \cap \mathrm { K e r } ( { \mathbf { C } } ) \big ) .
$$

By Lemma B.2, the dimension of this intersection is $\operatorname* { m a x } ( m - m ^ { * } , 0 )$ almost surely. We get that rank $\mathbf { U } ( t ) = \operatorname* { m i n } ( m , m ^ { * } )$ almost surely.

Step 2. SVD of the prediction. Let us now move to the SVD of the prediction. Since $\mathbf { V } ( t ) ^ { \top } \mathbf { V } ( t ) - \mathbf { U } ( t ) ^ { \top } \mathbf { U } ( t ) = 2 \rho \mathbf { I } _ { m }$ , the two matrices have a common orthonormal eigenbasis $\mathbf { o } _ { 1 } ( t ) , \ldots , \mathbf { o } _ { m } ( t )$ . Setting $r _ { Z } = \operatorname* { m i n } ( m , m ^ { * } )$ , we can choose compatible singular value decomposi-

$$
\mathbf { U } ( t ) = \sum _ { i = 1 } ^ { r _ { Z } } \sqrt { \lambda _ { i } ^ { U } ( t ) } \widehat { \mathbf { u } } _ { i } ( t ) \mathbf { o } _ { i } ( t ) ^ { \top } , \qquad \mathbf { V } ( t ) = \sum _ { i = 1 } ^ { m } \sqrt { \lambda _ { i } ^ { V } ( t ) } \widehat { \mathbf { v } } _ { i } ( t ) \mathbf { o } _ { i } ( t ) ^ { \top } .
$$

Order the eigenvalues of $\mathbf { U } ( t ) ^ { \top } \mathbf { U } ( t )$ decreasingly. The imbalance identity gives $\lambda _ { i } ^ { V } ( t ) = \lambda _ { i } ^ { U } ( t ) + 2 \rho$ for $i \in \{ 1 , \ldots , r _ { Z } \}$ and $\lambda _ { i } ^ { V } ( t ) = 2 \rho$ for $i > r _ { Z }$ . Hence the singular values of $\mathbf { V } ( t )$ are ordered in the same basis, including when some eigenvalues have multiplicities.

Computing $\mathbf { Z } ( t ) = \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top }$ now gives equation (8). Since the map $x \mapsto { \sqrt { x ( x + 2 \rho ) } }$ is increasing on $\mathbb { R } ^ { + }$ , this is an ordered singular value decomposition and rank $\mathbf { Z } ( t ) = r _ { Z }$

## B.4 Convergence of the Dynamics

We now prove Proposition 3. We first establish a general convergence result for the Oja flow in Appendix B.4.1, and then describe the spectral structure of $\mathbf { A } _ { \rho }$ in Appendix B.4.2. We finally isolate an invariant subspace that appears in the overparameterized case (Appendix B.4.3) and apply the general result in the overparameterized and underparameterized regimes (Appendices B.4.4 and B.4.5).

## B.4.1 Convergence of the Oja Flow

In the following, we consider the Oja flow dynamics:

$$
\dot { \mathbf H } ( t ) = ( \mathbf A - \mathbf H ( t ) \mathbf H ( t ) ^ { \top } ) \mathbf H ( t ) ,\tag{40}
$$

where $\mathbf { H } ( t ) \in \mathbb { R } ^ { D \times M }$ (with $M \leq D ) , \mathbf { A } \in { \mathcal { S } } _ { D } ( \mathbb { R } )$ . We are interested in the convergence of $\mathbf { H } ( t ) \mathbf { H } ( t ) ^ { \top }$ as $t \to \infty$ . The proof of Proposition 1 does not use the structure of $\mathbf { A } _ { \rho } ,$ so that applying it to $\mathbf { W } ( t ) = \sqrt { 2 } \mathbf { H } ( t )$ , which solves equation (5) with A instead of $\mathbf { A } _ { \rho } , \mathrm { g i v e s }$ for all $t \geq 0$ (see also Yan et al., 1994, Lemma 2.1):

$$
\mathbf { H } ( t ) \mathbf { H } ( t ) ^ { \top } = e ^ { t \mathbf { A } } \mathbf { H } _ { 0 } \left( \mathbf { I } _ { M } + 2 \mathbf { H } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \mathbf { A } } \mathrm { d } s \mathbf { H } _ { 0 } \right) ^ { - 1 } \mathbf { H } _ { 0 } ^ { \top } e ^ { t \mathbf { A } } ,\tag{41}
$$

where $\mathbf { H } _ { 0 } = \mathbf { H } ( t = 0 ) \in \mathbb { R } ^ { D \times M }$

For a closely related flow, Martin et al. (2024) characterize the limit through the local minimizers of the associated potential and a stable manifold argument (Smale, 1963). Here, we provide a proof by a direct analysis of the dynamics, which requires weaker assumptions on the initialization than the other route (see assumption (ii) in the following proposition).

Proposition 20. Consider the solution of the Oja flow dynamics (40). Let $\begin{array} { r } { \mathbf { A } = \sum _ { i = 1 } ^ { D } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } } \end{array}$ where $\lambda _ { 1 } \geq \cdots \geq \lambda _ { D }$ and $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { D }$ form an orthonormal basis $o f \mathbb { R } ^ { D }$ . Assume that:

(i) # $\{ i \in \{ 1 , \ldots , D \} , \lambda _ { i } > 0 \} \ge M$

(ii) The family $\big ( \mathbf { H } _ { 0 } ^ { \top } \mathbf { u } _ { 1 } , \ldots , \mathbf { H } _ { 0 } ^ { \top } \mathbf { u } _ { M } \big )$ is a basis of $\mathbb { R } ^ { M }$

(iii) Whenever $M < D$ , we have $\lambda _ { M } > \lambda _ { M + 1 }$

Then, we have the convergence:

$$
\mathbf { H } ( t ) \mathbf { H } ( t ) ^ { \top } \xrightarrow [ { t  \infty } ] { } \sum _ { i = 1 } ^ { M } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } .
$$

Proof. We start by briefly showing that we can consider without loss of generality that A is diagonal. The eigenvalue decomposition for A can be written $\mathbf { A } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top }$ , where Λ = $\mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { D } )$ and $\mathbf { U } \in O _ { D } ( \mathbb { R } )$ has columns $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { D }$ . Let $\tilde { \mathbf { H } } = \mathbf { U } ^ { \top } \mathbf { H }$ . Then, H<sup>˜</sup> is a solution of the diferential equation:

$$
\dot { \tilde { \mathbf { H } } } ( t ) = ( \mathbf { A } - \tilde { \mathbf { H } } ( t ) \tilde { \mathbf { H } } ( t ) ^ { \top } ) \tilde { \mathbf { H } } ( t ) .
$$

Then, the condition on $\mathbf { H } _ { 0 } ^ { \top } \mathbf { u } _ { 1 } , \ldots , \mathbf { H } _ { 0 } ^ { \top } \mathbf { u } _ { M }$ can be rephrased as the first M rows of $\tilde { \mathbf { H } } _ { 0 }$ forming a basis of $\mathbb { R } ^ { M }$ . In the following, if $M = D$ , the proof below applies with the lower blocks omitted.

Let us write:

$$
\begin{array} { r } { \pmb { \Lambda } = \left( \begin{array} { c c } { \pmb { \Lambda } _ { 1 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \pmb { \Lambda } _ { 2 } } \end{array} \right) , } \end{array}
$$

where $\pmb { \Lambda } _ { 1 } = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { M } ) \in \mathbb { R } ^ { M \times M }$ and $\pmb { \Lambda } _ { 2 } = \mathrm { d i a g } ( \lambda _ { M + 1 } , \dots , \lambda _ { D } ) \in \mathbb { R } ^ { ( D - M ) \times ( D - M ) }$ $\pmb { \Lambda } _ { 1 }$ contains the M largest eigenvalues (all positive) of A. Then, decompose:

$$
\tilde { \bf H } _ { 0 } = \binom { \bf S } { \tilde { \bf S } } ,
$$

where $\mathbf { S } \in \mathbb { R } ^ { M \times M }$ and $\tilde { \mathbf { S } } \in \mathbb { R } ^ { ( D - M ) \times M }$ . By assumption $\mathbf { S }$ is invertible. We now exploit the solution (41). We have:

$$
{ \bf I } _ { M } + 2 \tilde { \bf H } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \Lambda } \mathrm { d } s \tilde { \bf H } _ { 0 } = { \bf I } _ { M } + { \bf S } ^ { \top } \boldsymbol { \varphi } _ { t } ( { \bf A } _ { 1 } ) { \bf S } + \tilde { \bf S } ^ { \top } \boldsymbol { \varphi } _ { t } ( { \bf A } _ { 2 } ) \tilde { \bf S } ,
$$

where

$$
\varphi _ { t } ( \lambda ) = \frac { e ^ { 2 t \lambda } - 1 } { \lambda } ,
$$

and $\varphi _ { t } ( 0 ) = 2 t$ . Then $\varphi _ { t } ( \pmb { \Lambda } _ { 1 } )$ is positive definite and we can factorize:

$$
\begin{array} { r l r } {  { \mathbf { I } _ { M } + 2 \tilde { \mathbf { H } } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \mathbf { A } } \mathrm { d } s \tilde { \mathbf { H } } _ { 0 } } } \\ & { } & { = \mathbf { S } ^ { \top } \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { 1 / 2 } ( \mathbf { I } _ { M } + \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } \mathbf { S } ^ { - \top } \Big ( \mathbf { I } _ { M } + \tilde { \mathbf { S } } ^ { \top } \varphi _ { t } ( \mathbf { A } _ { 2 } ) \tilde { \mathbf { S } } \Big ) \mathbf { S } ^ { - 1 } \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } ) \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { 1 / 2 } \mathbf { S } . } \end{array}
$$

Moreover, since we assumed $\lambda _ { M } > \lambda _ { M + 1 }$ and $\lambda _ { 1 } \geq \cdots \geq \lambda _ { M } > 0$ , we have:

$$
\frac { \varphi _ { t } ( \lambda _ { j } ) } { \varphi _ { t } ( \lambda _ { i } ) } \xrightarrow [ t \to \infty ] 0 , \qquad \varphi _ { t } ( \lambda _ { i } ) \xrightarrow [ t \to \infty ] { } \infty ,
$$

for every $i \in \{ 1 , \ldots , M \}$ and $j \in \{ M + 1 , \ldots , D \}$ . Therefore, for $i , j \in \{ 1 , \ldots , M \}$ :

$$
\begin{array} { r l } & { \left( \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } \mathbf { S } ^ { - \top } \Big ( \mathbf { I } _ { M } + \tilde { \mathbf { S } } ^ { \top } \varphi _ { t } ( \mathbf { A } _ { 2 } ) \tilde { \mathbf { S } } \Big ) \mathbf { S } ^ { - 1 } \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } \right) _ { i j } } \\ & { \qquad = \frac { 1 } { \sqrt { \varphi _ { t } ( \lambda _ { i } ) \varphi _ { t } ( \lambda _ { j } ) } } \left( ( \mathbf { S } ^ { - \top } \mathbf { S } ^ { - 1 } ) _ { i j } + \displaystyle \sum _ { k = 1 } ^ { D - M } \mathbf { \Phi } ( \tilde { \mathbf { S } } \mathbf { S } ^ { - 1 } ) _ { k i } ( \tilde { \mathbf { S } } \mathbf { S } ^ { - 1 } ) _ { k j } \varphi _ { t } ( \lambda _ { M + k } ) \right) } \\ & { \qquad \xrightarrow [ t \to \infty ] 0 . } \end{array}
$$

Therefore:

$$
\mathbf { N } _ { 1 } ( t ) : = \mathbf { I } _ { M } + \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } \mathbf { S } ^ { - \top } \Big ( \mathbf { I } _ { M } + \tilde { \mathbf { S } } ^ { \top } \boldsymbol { \varphi } _ { t } ( \mathbf { A } _ { 2 } ) \tilde { \mathbf { S } } \Big ) \mathbf { S } ^ { - 1 } \boldsymbol { \varphi } _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } \xrightarrow [ t \to \infty ] { } \mathbf { I } _ { M } .
$$

Then, we have, using equation (41):

$$
\tilde { \mathbf { H } } ( t ) \tilde { \mathbf { H } } ( t ) ^ { \top } = \left( e ^ { t \Lambda _ { 1 } } \right) \varphi _ { t } ( \Lambda _ { 1 } ) ^ { - 1 / 2 } \mathbf { N } _ { 1 } ( t ) ^ { - 1 } \varphi _ { t } ( \Lambda _ { 1 } ) ^ { - 1 / 2 } \left( e ^ { t \Lambda _ { 1 } } \quad \mathbf { S } ^ { - \top } \tilde { \mathbf { S } } ^ { \top } e ^ { t \Lambda _ { 2 } } \right) .
$$

Now, we have:

$$
e ^ { t \mathbf { A } _ { 1 } } \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } \xrightarrow [ t \to \infty ] { } \mathbf { A } _ { 1 } ^ { 1 / 2 } , \qquad e ^ { t \mathbf { A } _ { 2 } } \tilde { \mathbf { S } } \mathbf { S } ^ { - 1 } \varphi _ { t } ( \mathbf { A } _ { 1 } ) ^ { - 1 / 2 } \xrightarrow [ t \to \infty ] { } \mathbf { 0 } .
$$

The first identity is a simple consequence of the expression of $\varphi _ { t } .$ , and the second can be shown by considering the coeficients of the matrix. Since $\mathbf { N } _ { 1 } ( t ) \xrightarrow [ { t \to \infty } ] { } \mathbf { I } _ { M }$ , we finally get:

$$
\tilde { \mathbf { H } } ( t ) \tilde { \mathbf { H } } ( t ) ^ { \top } \xrightarrow [ t  \infty ] { } ( \begin{array} { c c } { \mathbf { \Lambda } _ { 1 1 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { 0 } } \end{array} ) ,
$$

and therefore:

$$
\mathbf { H } ( t ) \mathbf { H } ( t ) ^ { \top } \xrightarrow [ { t  \infty } ] { } \mathbf { U } ( \begin{array} { l l } { \mathbf { \Lambda } _ { 1 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { 0 } } \end{array} ) \mathbf { U } ^ { \top } ,
$$

which is the desired limit.

## B.4.2 Spectral Structure of $\mathbf { A } _ { \rho }$

To apply Proposition 20, we now diagonalize the matrix $\mathbf { A } _ { \rho }$ from the singular value decomposition of C.

Proposition 21. Let $\begin{array} { r } { \mathbf { C } = \sum _ { i = 1 } ^ { m ^ { * } } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ be the SVD of C. Complete the families $\{ \mathbf { u } _ { i } \} _ { 1 \le i \le m ^ { * } }$ and $\{ \mathbf { v } _ { i } \} _ { 1 \leq i \leq m ^ { * } }$ into orthonormal bases of $\mathbb { R } ^ { p }$ and $\mathbb { R } ^ { d }$ . For $i \in \{ 1 , \ldots , m ^ { * } \}$ , define $\gamma _ { i } = \sqrt { \rho ^ { 2 } + \sigma _ { i } ^ { 2 } }$ and the vectors of $\mathbb { R } ^ { p + d }$

$$
\mathbf { z } _ { i } ^ { \pm } = \frac { 1 } { \sqrt { \sigma _ { i } ^ { 2 } + ( \rho \pm \gamma _ { i } ) ^ { 2 } } } \left( { \sigma _ { i } \mathbf { u } _ { i } } \right) , \quad \quad \quad \quad i \in \{ 1 , \ldots , m ^ { * } \} ,
$$

$$
{ \bf q } _ { j } ^ { - } = \left( { \bf u } _ { j } \right) ,
$$

$$
j \in \{ m ^ { * } + 1 , \ldots , p \} ,
$$

$$
{ \bf q } _ { k } ^ { + } = \left( \begin{array} { c c c } { { { \bf 0 } } } \\ { { { \bf v } _ { k } } } \end{array} \right) ,
$$

$$
k \in \{ m ^ { * } + 1 , \ldots , d \} .
$$

Then, we have the eigenvalue decomposition:

$$
\mathbf { A } _ { \rho } = \sum _ { i = 1 } ^ { m ^ { * } } \gamma _ { i } \mathbf { z } _ { i } ^ { + } ( \mathbf { z } _ { i } ^ { + } ) ^ { \top } - \sum _ { i = 1 } ^ { m ^ { * } } \gamma _ { i } \mathbf { z } _ { i } ^ { - } ( \mathbf { z } _ { i } ^ { - } ) ^ { \top } + \rho \sum _ { k = m ^ { * } + 1 } ^ { d } \mathbf { q } _ { k } ^ { + } ( \mathbf { q } _ { k } ^ { + } ) ^ { \top } - \rho \sum _ { j = m ^ { * } + 1 } ^ { p } \mathbf { q } _ { j } ^ { - } ( \mathbf { q } _ { j } ^ { - } ) ^ { \top } .
$$

Indeed, for $i \leq m ^ { * }$ , a direct computation gives ${ \bf A } _ { \rho } { \bf z } _ { i } ^ { \pm } = \pm \gamma _ { i } { \bf z } _ { i } ^ { \pm }$ , using $\sigma _ { i } ^ { 2 } = ( \gamma _ { i } - \rho ) ( \gamma _ { i } + \rho )$ Moreover $\mathbf { A } _ { \rho } \mathbf { q } _ { j } ^ { - } = - \rho \mathbf { q } _ { j } ^ { - }$ and $\mathbf { A } _ { \rho } \mathbf { q } _ { k } ^ { + } = \rho \mathbf { q } _ { k } ^ { + }$ . These $p + d$ vectors form an orthonormal basis, which proves the decomposition.

## B.4.3 Reduction in the Overparameterized Case

In the following we study the overparameterized case $m > m ^ { * }$ . We consider the notation of Proposition 1. Recall that the dynamics on $\mathbf { W } ( t )$ is initialized with:

$$
{ \bf W } _ { 0 } = \sqrt { 2 \rho } \left( { \bf 0 } \atop { \bf R } \right) ,
$$

where $\mathbf { R } \in \mathbb { R } ^ { d \times m }$ is such that ${ \mathbf { R } } ^ { \top } { \mathbf { R } } = { \mathbf { I } } _ { m }$ . In the following, we consider the subspace:

$$
E _ { 0 } = \binom { \mathbf { 0 } } { \mathrm { I m } ( \mathbf { R } ) \cap \mathrm { K e r } ( \mathbf { C } ) } \subset \mathbb { R } ^ { p + d } .\tag{42}
$$

Lemma B.3. Consider the notation of Proposition 1. Under Assumptions 1.1 and 1.4, with probability one, we have dim $( E _ { 0 } ) = \mathrm { m a x } ( m - m ^ { * } , 0 )$ . Moreover, for any $\mathbf { x } \in E _ { 0 }$ , we have $\mathbf W ( t ) \mathbf W ( t ) ^ { \top } \mathbf x = 2 \rho \mathbf x$ for all $t \geq 0$

Proof. The dimension follows from (i) in Lemma B.2, since rank $( \mathbf { C } ) = m ^ { * }$ . For $\mathbf { x } \in E _ { 0 }$ , one has $\mathbf { A } _ { \rho } \mathbf { x } = \rho \mathbf { x }$ and $\mathbf { W } _ { 0 } \mathbf { W } _ { 0 } ^ { \top } \mathbf { x } = 2 \rho \mathbf { x }$ . As a consequence,

$$
\left( \mathbf { I } _ { m } + \mathbf { W } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \mathbf { A } _ { \rho } } \mathrm { d } s \mathbf { W } _ { 0 } \right) \mathbf { W } _ { 0 } ^ { \top } \mathbf { x } = \left( 1 + 2 \rho \int _ { 0 } ^ { t } e ^ { 2 \rho s } \mathrm { d } s \right) \mathbf { W } _ { 0 } ^ { \top } \mathbf { x } = e ^ { 2 \rho t } \mathbf { W } _ { 0 } ^ { \top } \mathbf { x } .
$$

Applying the closed form (6) then gives the result.

From now on, we assume m $> m ^ { * }$ (overparameterized case), so that dim $\left( E _ { 0 } \right) = m - m ^ { * } > 0$ For $t > 0$ , the Woodbury identity applied to the solution in equation (6) gives

$$
\begin{array} { l } { { \displaystyle { \bf W } ( t ) { \bf W } ( t ) ^ { \top } = e ^ { t { \bf A } _ { \rho } } \Big ( { \bf F } ( t ) ^ { - 1 } - { \bf F } ( t ) ^ { - 1 } \big ( { \bf F } ( t ) ^ { - 1 } + { \bf W } _ { 0 } { \bf W } _ { 0 } ^ { \top } \big ) ^ { - 1 } { \bf F } ( t ) ^ { - 1 } \Big ) e ^ { t { \bf A } _ { \rho } } } , } \\ { { \displaystyle { \bf F } ( t ) = \int _ { 0 } ^ { t } e ^ { 2 s { \bf A } _ { \rho } } \mathrm { d } s , } } \end{array}
$$

where $\mathbf F ( t )$ is invertible. The space $E _ { 0 }$ and its orthogonal complement are invariant under both $\mathbf { A } _ { \rho }$ and $\mathbf { W } _ { 0 } \mathbf { W } _ { 0 } ^ { \top }$ . Thus, decomposing $\mathbb { R } ^ { p + d }$ along this orthogonal sum, we have:

$$
\mathbf { Y } ( t ) = 2 \rho \mathbf { I } \mathbf { I } + e ^ { t \tilde { \mathbf { A } } } \Big ( \tilde { \mathbf { F } } ( t ) ^ { - 1 } - \tilde { \mathbf { F } } ( t ) ^ { - 1 } \big ( \tilde { \mathbf { F } } ( t ) ^ { - 1 } + \tilde { \mathbf { Y } } _ { 0 } \big ) ^ { - 1 } \tilde { \mathbf { F } } ( t ) ^ { - 1 } \Big ) e ^ { t \tilde { \mathbf { A } } } ,\tag{43}
$$

where Π is the orthogonal projection onto $E _ { 0 }$ , and

$$
\tilde { \mathbf { Y } } _ { 0 } = \mathbf { H } ^ { \perp } \mathbf { W } _ { 0 } \mathbf { W } _ { 0 } ^ { \top } \mathbf { \Pi } \mathbf { \Pi } ^ { \perp } , \qquad \tilde { \mathbf { F } } ( t ) = \int _ { 0 } ^ { t } e ^ { 2 s \tilde { \mathbf { A } } } \mathrm { d } s , \qquad \tilde { \mathbf { A } } = \mathbf { \Pi } \mathbf { \Pi } ^ { \perp } \mathbf { A } _ { \rho } \mathbf { \Pi } \mathbf { \Pi } ^ { \perp } ,
$$

and $\Pi ^ { \perp } = \mathbf { I } _ { p + d } - \Pi$ is the orthogonal projection onto $E _ { 0 } ^ { \perp }$

Lemma B.4. Consider the setting of Lemma B.3. In the overparameterized case $m > m ^ { * }$ , with probability one, there exists $\tilde { \mathbf { R } } \in \mathbb { R } ^ { d \times m ^ { * } }$ such that $\tilde { \mathbf { R } } ^ { \top } \tilde { \mathbf { R } } = \mathbf { I } _ { m ^ { \ast } }$ , and:

$$
\tilde { \mathbf { Y } } _ { 0 } = 2 \rho \left( \begin{array} { c c } { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \tilde { \mathbf { R } } \tilde { \mathbf { R } } ^ { \top } } \end{array} \right) .
$$

Proof. Due to the form of $E _ { 0 }$ in equation (42), Π can be written as:

$$
\Pi = \left( { \bf 0 } \frac { { \bf 0 } } { { \bf 0 } } \right) ,\tag{44}
$$

where $\overline { { \Pi } }$ is the orthogonal projection onto Im $( \mathbf { R } ) \cap \mathop { \mathrm { K e r } } ( \mathbf { C } )$ . Then, the expression of $\mathbf { W } _ { 0 }$ guarantees that:

$$
\begin{array} { r l } { \tilde { \mathbf { Y } } _ { 0 } = 2 \rho \left( \begin{array} { c c } { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { ( \mathbf { I } _ { d } - \overline { { \Pi } } ) \mathbf { R } \mathbf { R } ^ { \top } ( \mathbf { I } _ { d } - \overline { { \Pi } } ) } \end{array} \right) . } \end{array}
$$

Since $\mathbf { R R } ^ { \top }$ is a projection and $\operatorname { I m } ( { \overline { { \mathbf { I I } } } } ) \subset \operatorname { I m } ( \mathbf { R } )$ , we have RR $^ { \top } \overline { { \mathbf { I I } } } = \overline { { \mathbf { I I } } }$ , and therefore:

$$
( \mathbf { I } _ { d } - \overline { { \mathbf { I } } } ) \mathbf { R } \mathbf { R } ^ { \top } ( \mathbf { I } _ { d } - \overline { { \mathbf { I } } } ) = \mathbf { R } \mathbf { R } ^ { \top } - \overline { { \mathbf { I } } } \overline { { \mathbf { I } } } .
$$

The matrix $\mathbf { R R } ^ { \top } - \overline { { \Pi } }$ is the orthogonal projection onto the orthogonal complement of $\operatorname { I m } ( \mathbf { R } ) \cap$ $\operatorname { K e r } ( \mathbf { C } )$ within Im(R). Its rank is

$$
m - \dim \bigl ( \operatorname { I m } ( \mathbf { R } ) \cap \operatorname { K e r } ( \mathbf { C } ) \bigr ) = m ^ { * } ,
$$

with probability one (from Lemma B.3). Therefore, we can write $\mathbf { R } \mathbf { R } ^ { \top } - \overline { { \mathbf { I } \mathbf { I } } } = \tilde { \mathbf { R } } \tilde { \mathbf { R } } ^ { \top }$ where $\tilde { \mathbf { R } } \in \bar { \mathbb { R } } ^ { d \times m ^ { * } }$ is such that $\tilde { \mathbf { R } } ^ { \top } \tilde { \mathbf { R } } = \mathbf { I } _ { m ^ { * } }$ . This concludes the proof. □

This lemma leads to:

$$
\begin{array} { r } { \tilde { \mathbf { Y } } _ { 0 } = \tilde { \mathbf { W } } _ { 0 } \tilde { \mathbf { W } } _ { 0 } ^ { \top } , \qquad \tilde { \mathbf { W } } _ { 0 } = \sqrt { 2 \rho } \left( \mathbf { 0 } } \\ { \tilde { \mathbf { R } } \right) \in \mathbb { R } ^ { ( p + d ) \times m ^ { * } } . } \end{array}
$$

Therefore, applying the Woodbury formula again to equation (43), we can write:

$$
\mathbf { Y } ( t ) = 2 \rho \mathbf { I } \mathbf { I } + e ^ { t \tilde { \mathbf { A } } } \tilde { \mathbf { W } } _ { 0 } \left( \mathbf { I } _ { m ^ { * } } + \tilde { \mathbf { W } } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \tilde { \mathbf { A } } } \mathrm { d } s \tilde { \mathbf { W } } _ { 0 } \right) ^ { - 1 } \tilde { \mathbf { W } } _ { 0 } ^ { \top } e ^ { t \tilde { \mathbf { A } } } .\tag{45}
$$

## B.4.4 Convergence in the Overparameterized Case

The goal is now to apply Proposition 20 to the matrix in equation (45). Up to the constant term $2 \rho \Pi$ , this matrix has the form of the solution of the Oja flow in equation (41) with an initialization $\tilde { \mathbf { W } } _ { 0 } \in \mathbb { R } ^ { ( p + d ) \times m ^ { * } }$ , which brings the case $m > m ^ { * }$ back to the case $m = m ^ { * }$ . This step is needed since, with $M = m$ , assumption (iii) of Proposition 20 fails for $\mathbf { A } _ { \rho } ,$ whose m-th and $( m + 1 ) { \cdot } \mathrm { t h }$ eigenvalues both equal ρ (see Proposition 21). To this end, we investigate how $\Pi ^ { \perp }$ acts on $\mathbf { A } _ { \rho }$ . Back to Proposition 21, we have, using the block structure of Π in equation (44):

$$
\begin{array} { r l r l } & { \mathbf { I I z } _ { i } ^ { \pm } = \displaystyle \frac { 1 } { \sqrt { \sigma _ { i } ^ { 2 } + ( \rho \pm \gamma _ { i } ) ^ { 2 } } } \left( { \displaystyle ( \rho \pm \gamma _ { i } ) \overline { { \mathbf { I } } } \mathbf { v } _ { i } } \right) , \quad } & & { i \in \{ 1 , \ldots , m ^ { * } \} , } \\ & { \mathbf { I I q } _ { j } ^ { - } = \mathbf { 0 } , \quad } & & { j \in \{ m ^ { * } + 1 , \ldots , p \} , } \\ & { \mathbf { I I q } _ { k } ^ { + } = \left( \displaystyle \frac { \mathbf { 0 } } { \mathbf { I I v } _ { k } } \right) , \quad } & & { k \in \{ m ^ { * } + 1 , \ldots , d \} . } \end{array}
$$

Since Im $( { \overline { { \mathbf { I I } } } } ) \subset \operatorname { K e r } ( \mathbf { C } )$ and $\mathbf { v } _ { i } \in \mathop { \mathrm { K e r } } ( \mathbf { C } ) ^ { \perp }$ for $i = 1 , \ldots , m ^ { * }$ , we have $\pmb { \Pi } \mathbf { z } _ { i } ^ { \pm } = \mathbf { 0 }$ . Therefore, only the vectors ${ \bf q } _ { k } ^ { + }$ for $k \in \{ m ^ { * } + 1 , \ldots , d \}$ are changed by $\Pi ^ { \perp }$ . Therefore:

$$
\mathbf { \Pi } ^ { \mathbf { T } ^ { \bot } } \mathbf { A } _ { \rho } \mathbf { \Pi } ^ { \bot } = \sum _ { i = 1 } ^ { m ^ { \star } } \gamma _ { i } \mathbf { z } _ { i } ^ { + } \left( \mathbf { z } _ { i } ^ { + } \right) ^ { \top } - \sum _ { i = 1 } ^ { m ^ { \star } } \gamma _ { i } \mathbf { z } _ { i } ^ { - } \left( \mathbf { z } _ { i } ^ { - } \right) ^ { \top } + \rho \sum _ { k = m ^ { \star } + 1 } ^ { d } \Pi ^ { \bot } \mathbf { q } _ { k } ^ { + } ( \mathbf { q } _ { k } ^ { + } ) ^ { \top } \mathbf { \Pi } ^ { \bot } - \rho \sum _ { j = m ^ { \star } + 1 } ^ { p } \mathbf { q } _ { j } ^ { - } \left( \mathbf { q } _ { j } ^ { - } \right) ^ { \top } .
$$

Apart from the third term, this is still an eigendecomposition for $\mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi \Pi } \mathbf { \Pi } \mathbf { \Pi \Pi } \mathbf { \Pi } \mathbf { \Pi \Pi } \mathbf { \Pi \Pi } \mathbf \mathbf { \Pi \Pi } \mathbf { \Pi \Pi } \mathbf \mathbf { \Pi \Pi } \mathbf \mathbf { \Pi \Pi \Pi } \mathbf \mathbf  \Pi \Pi \Pi \Pi \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \mathbf \Pi \Pi \mathbf \Pi \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \Pi \mathbf \Pi \Pi \Pi \Pi \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m m \ m \ m \ m \ m m \ m \ m m \ m \ m \ m m \ m m \ m \ m m m \ m m m \ m \ m m \ m m m m \ m m m m \ m \ m m \ m m m m \ m m m m m \ m m m \ m m m m m m \ m m m m m m \ m m m m m m m m m m m m \ m m m m m m m m m m m m m m m m m a n \ m m m m m m m m m m m m m m m m m a n \ m m m m m m m m a n m m m m m m m m m m m m m m m a$ . Indeed, ${ \bf z } _ { i } ^ { \pm }$ and $\mathbf { q } _ { j } ^ { - }$ are orthogonal to ${ \bf q } _ { k } ^ { + }$ and to $E _ { 0 }$ , hence to $\mathbf { \Pi } \mathbf { H } ^ { \perp } \mathbf { q } _ { k } ^ { + }$ . The third term is positive semidefinite and has operator norm at most $\rho \colon$

$$
\rho \lambda _ { \mathrm { m a x } } \left( \mathbf { I } ^ { \perp } \sum _ { k = m ^ { * } + 1 } ^ { d } \mathbf { q } _ { k } ^ { + } ( \mathbf { q } _ { k } ^ { + } ) ^ { \top } \mathbf { I } ^ { \perp } \right) \leq \rho \lambda _ { \mathrm { m a x } } \left( \sum _ { k = m ^ { * } + 1 } ^ { d } \mathbf { q } _ { k } ^ { + } ( \mathbf { q } _ { k } ^ { + } ) ^ { \top } \right) = \rho ,
$$

where we used that $\Pi ^ { \perp }$ is an orthogonal projection and $( \mathbf { q } _ { k } ^ { + } ) _ { m ^ { * } + 1 \leq k \leq d }$ is an orthonormal family. Therefore, every eigenvalue outside the span of $\mathbf { z } _ { 1 } ^ { + } , \ldots , \mathbf { z } _ { m ^ { * } } ^ { + }$ is at most $\rho .$ Since $\gamma _ { i } = \sqrt { \sigma _ { i } ^ { 2 } + \rho ^ { 2 } } >$ $\rho ,$ the top $m ^ { * }$ eigenvalues of $\tilde { \mathbf { A } }$ are $\gamma _ { 1 } , \ldots , \gamma _ { m ^ { * } }$ , with eigenvectors $\mathbf { z } _ { 1 } ^ { + } , \ldots , \mathbf { z } _ { m ^ { * } } ^ { + }$ . This verifies assumptions (i) and (iii) of Proposition 20.

Let us now verify assumption (ii) of Proposition 20. To do so, we need to prove that the family $\tilde { \mathbf { W } } _ { 0 } ^ { \top } \mathbf { z } _ { 1 } ^ { + } , \ldots , \tilde { \mathbf { W } } _ { 0 } ^ { \top } \mathbf { z } _ { m } ^ { + }$ <sub>∗</sub> forms a basis of $\mathbb { R } ^ { m ^ { * } }$ . Since $\tilde { \mathbf { W } } _ { 0 } ^ { \top } \tilde { \mathbf { W } } _ { 0 } = 2 \rho \mathbf { I } _ { m }$ ∗ is invertible, $\tilde { \mathbf { W } } _ { 0 }$ is injective, and it is therefore enough to show that the vectors $\tilde { \mathbf { W } } _ { 0 } \tilde { \mathbf { W } } _ { 0 } ^ { \top } \mathbf { z } _ { 1 } ^ { + } , \ldots , \tilde { \mathbf { W } } _ { 0 } \tilde { \mathbf { W } } _ { 0 } ^ { \top } \mathbf { z } _ { m } ^ { + }$ <sub>∗</sub> are linearly independent in $\mathbb { R } ^ { p + d }$ . We have:

$$
\tilde { \mathbf { W } } _ { 0 } \tilde { \mathbf { W } } _ { 0 } ^ { \top } \mathbf { z } _ { i } ^ { + } = \frac { 2 \rho ( \rho + \gamma _ { i } ) } { \sqrt { \sigma _ { i } ^ { 2 } + ( \rho + \gamma _ { i } ) ^ { 2 } } } \left( { \bf \tilde { R } } \tilde { \mathbf { R } } ^ { \top } \mathbf { v } _ { i } \right) .
$$

Since $\tilde { \mathbf { R } } \tilde { \mathbf { R } } ^ { \top } = \mathbf { R } \mathbf { R } ^ { \top } - \overline { { \mathbf { I } } }$ , and $\overline { { \mathbf { I I } } } \mathbf { v } _ { i } = \mathbf { 0 }$ for $i \in \{ 1 , \ldots , m ^ { * } \}$ }, we have R<sup>˜</sup> R<sup>˜</sup> $\mathbf { \bar { \Psi } } { \mathbf { v } } _ { i } = { \mathbf { R } } { \mathbf { R } } ^ { \top } { \mathbf { v } } _ { i }$ . Since $m > m ^ { * }$ , point (ii) in Lemma B.2 shows that $\mathbf R ^ { \top } \mathbf v _ { 1 } , \ldots , \mathbf R ^ { \top } \mathbf v _ { m ^ { * } }$ are linearly independent almost surely. Since R is injective on $\mathbb { R } ^ { m }$ , the family RR<sup>⊤</sup> $\mathbf { \bar { v } } _ { 1 } , \ldots , \mathbf { R R } ^ { \top } \mathbf { v } _ { m }$ ∗ is linearly independent in $\mathbb { R } ^ { d }$ , which proves the claim.

We are now ready to apply the result of Proposition 20. Back to equation (45), we let $\begin{array} { r } { \mathbf { H } _ { 0 } = \frac { 1 } { \sqrt { 2 } } \tilde { \mathbf { W } } _ { 0 } \in \mathbb { R } ^ { ( p + \bar { d } ) \times m ^ { * } } } \end{array}$ , so that:

$$
\mathbf { Y } ( t ) = 2 \rho \mathbf { I } \mathbf { I } + 2 e ^ { t \tilde { \mathbf { A } } } \mathbf { H } _ { 0 } \left( \mathbf { I } _ { m ^ { * } } + 2 \mathbf { H } _ { 0 } ^ { \top } \int _ { 0 } ^ { t } e ^ { 2 s \tilde { \mathbf { A } } } \mathrm { d } s \mathbf { H } _ { 0 } \right) ^ { - 1 } \mathbf { H } _ { 0 } ^ { \top } e ^ { t \tilde { \mathbf { A } } } .
$$

Given the eigendecomposition of $\tilde { \mathbf { A } }$ , we get the limit:

$$
\mathbf { Y } ( t ) \xrightarrow [ { t \to \infty } ] { } 2 \rho \mathbf { I I } + 2 \sum _ { i = 1 } ^ { m ^ { * } } \gamma _ { i } \mathbf { z } _ { i } ^ { + } ( \mathbf { z } _ { i } ^ { + } ) ^ { \top } .\tag{46}
$$

## B.4.5 Convergence in the Underparameterized Case

Whenever $m \leq m ^ { * } , E _ { 0 } = \{ \mathbf { 0 } \}$ with probability one, and we can directly apply Proposition 20. Using Proposition 21, the largest eigenvalues of $\mathbf { A } _ { \rho }$ are $\gamma _ { 1 } \geq \cdots \geq \gamma _ { m ^ { * } }$ . Thus there are at least m positive eigenvalues. If $m < m ^ { * }$ , the assumed gap $\sigma _ { m } > \sigma _ { m + 1 }$ gives $\gamma _ { m } > \gamma _ { m + 1 }$ , and if $m = m ^ { * }$ then $\gamma _ { m } > \rho ,$ and we have shown that the next eigenvalue is at most $\rho .$ This guarantees that assumptions (i), (iii) of Proposition 20 are verified.

For assumption (ii), we have the expression:

$$
\mathbf { W } _ { 0 } ^ { \top } \mathbf { z } _ { i } ^ { + } = \sqrt { \frac { 2 \rho ( \rho + \gamma _ { i } ) ^ { 2 } } { \sigma _ { i } ^ { 2 } + ( \rho + \gamma _ { i } ) ^ { 2 } } } \mathbf { R } ^ { \top } \mathbf { v } _ { i } .
$$

By point (ii) in Lemma B.2, with probability one, $\mathbf { R } ^ { \top } \mathbf { v } _ { 1 } , \ldots , \mathbf { R } ^ { \top } \mathbf { v } _ { m }$ is a basis of $\mathbb { R } ^ { m }$ . This confirms assumption (ii), and we have the limit:

$$
\mathbf { Y } ( t ) \xrightarrow [ { t \to \infty } ] { } 2 \sum _ { i = 1 } ^ { m } \gamma _ { i } \mathbf { z } _ { i } ^ { + } ( \mathbf { z } _ { i } ^ { + } ) ^ { \top } .\tag{47}
$$

## B.4.6 Identification of the Block Matrices

Recall that:

$$
\mathbf { Y } ( t ) = \left( \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } \mathbf { \Sigma } \mathbf { \mathbf { U } } ( t ) \mathbf { V } ( t ) ^ { \top } \right) .
$$

Due to the expression of ${ \mathbf z } _ { i } ^ { + }$ in Proposition 21, we have the identity:

$$
\mathbf { z } _ { i } ^ { + } ( \mathbf { z } _ { i } ^ { + } ) ^ { \top } = \frac { 1 } { 2 \gamma _ { i } } \left( \begin{array} { c c } { ( \gamma _ { i } - \rho ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } } & { \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \\ { \sigma _ { i } \mathbf { v } _ { i } \mathbf { u } _ { i } ^ { \top } } & { ( \gamma _ { i } + \rho ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array} \right) .
$$

Using the block structure of the projection Π in equation (44) and the limit of equations (46), (47), we end up with:

$$
\begin{array} { r l } & { \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } \xrightarrow [ t \to \infty ] { \operatorname* { m i n } ( m , m ^ { * } ) } ( \gamma _ { i } - \rho ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } , } \\ & { \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } \xrightarrow [ t \to \infty ] { \operatorname* { m i n } ( m , m ^ { * } ) } ( \gamma _ { i } + \rho ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } + 2 \rho \mathbf { 1 } _ { m > m ^ { * } } \overline { { \mathbf { \Pi } } } , } \\ & { \mathbf { U } ( t ) \mathbf { V } ( t ) ^ { \top } \xrightarrow [ t \to \infty ] { \operatorname* { m i n } ( m , m ^ { * } ) } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } . } \end{array}
$$

This is precisely Proposition 3.

## C High-Dimensional Limit

This appendix proves the high-dimensional results of Section 4. We start by establishing three random matrix convergences (Appendix C.1) and apply them to the Stieltjes transforms and their long-time limits (Appendix C.2). We then give an additional result for two-point Stieltjes transforms (Appendix C.3) and prove the range-projector formula (Appendix C.4).

## C.1 Random Matrix Results

In this section, we are interested in the high-dimensional limit of quantities of the form:

$$
T _ { d } ( a , b ) = \frac { 1 } { d } \mathrm { T r } \Big ( \mathbf { X } ^ { \top } b ( \mathbf { D } ) \mathbf { X } \big ( \mathbf { X } ^ { \top } a ( \mathbf { D } ) \mathbf { X } \big ) ^ { - 1 } \Big ) ,\tag{48}
$$

where X is uniformly drawn on $\mathrm { S t } ( d , m )$ , D is deterministic and symmetric, and the functions of D are applied spectrally. We prove three high-dimensional limits: we study the quantity (48) and derive its high-dimensional limit whenever a is real positive (Theorem 22) or when it is complex (Theorem 23), in which case the previous quantity has a Stieltjes-like structure. Finally, in Theorem 24, we derive the limit of a second-order trace quantity. All these results follow from the deterministic equivalents for sketched pseudoinverses of LeJeune et al. (2024).

Theorem 22. Let $\mathbf { X } \in \mathbb { R } ^ { d \times m }$ be uniformly distributed on $\mathrm { S t } ( d , m )$ . Let $\mathbf { D } \in { \cal { S } } _ { d } ( \mathbb { R } )$ be deterministic, with spectrum contained in a compact set $S \subset \mathbb { R }$ independent of $d ,$ and assume that its empirical spectral measure $\mu _ { d }$ converges weakly to a probability measure $\mu .$ Moreover, assume that $m / d \xrightarrow [ d  \infty ] { } \kappa \in ( 0 , 1 )$

Let $a : S \to \mathbb { R } ^ { + }$ and $b : S  \mathbb { C }$ be continuous. Assume that:

$$
\mu ( \{ x : a ( x ) > 0 \} ) > \kappa ,\tag{49}
$$

and that in $\dot { \bar { \cdot } } d \geq 1 \lambda _ { \operatorname* { m i n } } ^ { + } ( a ( \mathbf { D } ) ) > 0$ , where $\lambda _ { \operatorname* { m i n } } ^ { + }$ denotes the smallest non-zero eigenvalue. Let $h > 0$ be the unique solution of

$$
1 = \int \frac { \mu ( \mathrm { d } x ) } { 1 - \kappa + h a ( x ) } .\tag{50}
$$

Then, with probability one,

$$
\frac { 1 } { d } \mathrm { T r } ( { \bf X } ^ { \top } b ( { \bf D } ) { \bf X } ( { \bf X } ^ { \top } a ( { \bf D } ) { \bf X } ) ^ { - 1 } ) \xrightarrow [ d  \infty ] { } h \int \frac { b ( x ) } { 1 - \kappa + h a ( x ) } \mu ( \mathrm { d } x ) .\tag{51}
$$

Proof. First, the solution of equation (50) exists and is unique. Indeed, the right-hand side as a function of h is continuous and strictly decreasing on $\mathbb { R } ^ { + }$ , is equal to $( 1 - \kappa ) ^ { - 1 } > 1$ at $h = 0$ , and converges as $h \to \infty$ to $\mu ( \{ a = 0 \} ) / ( 1 - \kappa ) < 1$ by the assumption $\mu ( \{ a > 0 \} ) > \kappa$ . By linearity, it is enough to prove the convergence for real-valued b.

Consider $T _ { d } ( a , b )$ as defined in equation (48). We first show that X can be replaced by a Gaussian matrix. Let $\tilde { \textbf { G } } \in \mathbb { R } ^ { d \times m }$ have i.i.d. $\mathcal { N } ( 0 , 1 / m )$ entries, independent of X, and set $\mathbf { G } = \mathbf { X } ( \tilde { \mathbf { G } } ^ { \top } \tilde { \mathbf { G } } ) ^ { 1 / 2 }$ . Since $\tilde { \mathbf { G } } ( \tilde { \mathbf { G } } ^ { \top } \tilde { \mathbf { G } } ) ^ { - 1 / 2 }$ is uniform on $\mathrm { S t } ( d , m )$ (Chikuse, 2003, Theorem 2.2.1) and independent of $\tilde { \mathbf { G } } ^ { \top } \tilde { \mathbf { G } }$ , G has the same law as $\tilde { \mathbf { G } }$ , and $\mathbf { X } = \mathbf { \dot { G } } ( \mathbf { G } ^ { \top } \mathbf { G } ) ^ { - 1 / 2 }$ . Therefore:

$$
T _ { d } ( a , b ) = \frac { 1 } { d } \mathrm { T r } \Big ( { \bf G } ^ { \top } b ( { \bf D } ) { \bf G } \big ( { \bf G } ^ { \top } a ( { \bf D } ) { \bf G } \big ) ^ { - 1 } \Big ) .
$$

Since a is continuous, the set $\{ a > 0 \}$ is open, and weak convergence gives

$$
\operatorname* { l i m } _ { d \to \infty } \operatorname { i n f } _ { \mathbf { \alpha } } \frac { \operatorname { r a n k } ( a ( \mathbf { D } ) ) } { d } = \operatorname* { l i m } _ { d \to \infty } \operatorname* { i n f } _ { \mathbf { \alpha } } \mu _ { d } ( \{ a > 0 \} ) \geq \mu ( \{ a > 0 \} ) > \kappa .
$$

Therefore LeJeune et al. (2024, Theorem 4.1(ii)) applies at regularization parameter $\lambda = 0$ to the matrix $a ( \mathbf D )$ . Thus, in the sense of deterministic equivalents of LeJeune et al. (2024, Sec. 2),

$$
\mathbf { G } \big ( \mathbf { G } ^ { \top } { a } ( \mathbf { D } ) \mathbf { G } \big ) ^ { - 1 } \mathbf { G } ^ { \top } \simeq \big ( { a } ( \mathbf { D } ) + \nu _ { d } \mathbf { I } _ { d } \big ) ^ { - 1 } ,
$$

where $\nu _ { d } > 0$ is determined by

$$
\frac { m } { d } = \frac { 1 } { d } \mathrm { T r } \left( a ( { \bf D } ) { \left( a ( { \bf D } ) + \nu _ { d } { \bf I } _ { d } \right) } ^ { - 1 } \right) .\tag{52}
$$

By the definition of deterministic equivalents, the convergence is almost sure when testing along the matrix $d ^ { - 1 } b ( { \mathbf D } )$ , since

$$
\operatorname* { s u p } _ { d \geq 1 } { \frac { 1 } { d } } \mathrm { T r } \left| b ( \mathbf { D } ) \right| \leq \operatorname* { s u p } _ { x \in S } \left| b ( x ) \right| < \infty .
$$

Therefore, almost surely,

$$
\frac { 1 } { d } \mathrm { T r } \left( \mathbf { X } ^ { \top } b ( \mathbf { D } ) \mathbf { X } ( \mathbf { X } ^ { \top } a ( \mathbf { D } ) \mathbf { X } ) ^ { - 1 } \right) - \frac { 1 } { d } \mathrm { T r } \left( b ( \mathbf { D } ) \big ( a ( \mathbf { D } ) + \nu _ { d } \mathbf { I } _ { d } \big ) ^ { - 1 } \right) \xrightarrow [ d \to \infty ] { \mathrm { a . s . } } 0 .\tag{53}
$$

Now, the equation for $\nu _ { d }$ can be rewritten as

$$
\frac { m } { d } = \int \frac { a ( x ) } { a ( x ) + \nu _ { d } } \mu _ { d } ( \mathrm { d } x ) .
$$

Now it is clear that $\nu _ { d } > 0$ remains bounded in $( 0 , \infty )$ : if it had a subsequence diverging $\mathrm { t o } + \infty$ the RHS would converge to zero along this subsequence, leading to a contradiction. Additionally, $\nu _ { d }$ cannot have a subsequence going to zero. Otherwise, for any fixed $\nu > 0$ , we would have $\nu _ { d } < \nu$ along this subsequence for d large enough, hence:

$$
\frac { m } { d } \geq \int \frac { a ( x ) } { a ( x ) + \nu } \mu _ { d } ( \mathrm { d } x ) \xrightarrow [ d  \infty ] { } \int \frac { a ( x ) } { a ( x ) + \nu } \mu ( \mathrm { d } x ) ,
$$

and letting $\nu \to 0$ would give $\kappa \geq \mu ( \{ a > 0 \} )$ and contradict equation (49). Hence taking a convergent subsequence of $( \nu _ { d } ) _ { d \geq 1 }$ and using that $\mu _ { d } \Rightarrow \mu$ gives the equation:

$$
\kappa = \int \frac { a ( x ) } { a ( x ) + \nu } \mu ( \mathrm { d } x ) .\tag{54}
$$

The right-hand side is strictly decreasing in $\nu ,$ so this equation has a unique positive solution. Therefore $\nu _ { d }$ converges to $\nu .$ Setting $h = ( 1 - \kappa ) / \nu$ gives equation (50). Finally, as a consequence of the deterministic equivalent in equation (53), we have the almost sure convergence:

$$
\frac { 1 } { d } \mathrm { T r } \left( \mathbf { X } ^ { \top } b ( \mathbf { D } ) \mathbf { X } \big ( \mathbf { X } ^ { \top } a ( \mathbf { D } ) \mathbf { X } \big ) ^ { - 1 } \right) \xrightarrow [ d \to \infty ] { } \int \frac { b ( x ) } { a ( x ) + \nu } \mu ( \mathrm { d } x ) = h \int \frac { b ( x ) } { 1 - \kappa + h a ( x ) } \mu ( \mathrm { d } x ) ,
$$

which proves (51).

We shall now move on to the complex version of the previous result. Before stating it, we prove a first lemma that derives uniform bounds for traces of complex matrices.

Lemma C.1. Let $\mathbf { X } _ { 0 } \in \mathbb { R } ^ { d \times m }$ be such that $\mathbf { X } _ { 0 } ^ { \top } \mathbf { X } _ { 0 } = \mathbf { I } _ { m }$ . Let $\mathbf { D } \in { \mathcal { S } } _ { d } ( \mathbb { R } )$ have spectrum contained in a compact set $S \subset \mathbb { R }$ independent of d. Additionally, for $\zeta \in \mathbb { C } \backslash \mathbb { R } ^ { + }$ , consider $\begin{array} { r } { a _ { \zeta } ( x ) = q ( x ) - \zeta r ( x ) } \end{array}$ , where $q , r \colon S \to \mathbb { R } ^ { + }$ are continuous and $q ( x ) \geq \eta > 0$ on S. Then, given $K$ a compact set of $\mathbb { C } \setminus \mathbb { R } ^ { + }$ , we have:

$$
\operatorname* { s u p } _ { \zeta \in K } \operatorname* { s u p } _ { d \geq 1 } \frac { 1 } { d } \Big | \operatorname { T r } \Big ( \big ( \mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } \big ) ^ { - 1 } \Big ) \Big | < \infty , \qquad \operatorname* { s u p } _ { \zeta \in K } \operatorname* { s u p } _ { d \geq 1 } \frac { 1 } { d } \Big \| \big ( \mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } \big ) ^ { - 1 } \Big \| _ { F } ^ { 2 } < \infty ,
$$

where $\| \cdot \| _ { F }$ is the Frobenius norm on $\mathbb { C } ^ { m \times m }$

Proof. We let:

$$
{ \bf M } = { \bf X } _ { 0 } ^ { \top } q ( { \bf D } ) { \bf X } _ { 0 } , \qquad { \bf N } = { \bf M } ^ { - 1 / 2 } { \bf X } _ { 0 } ^ { \top } r ( { \bf D } ) { \bf X } _ { 0 } { \bf M } ^ { - 1 / 2 } .
$$

Both of these matrices belong to $S _ { m } ^ { + } ( \mathbb R )$ . Since $q \geq \eta$ on $S$ and $\mathbf { X } _ { 0 } ^ { \top } \mathbf { X } _ { 0 } = \mathbf { I } _ { m }$ , we have $\mathbf { M } \succeq \eta \mathbf { I } _ { m }$ Moreover $\mathbf { N } \succeq 0$ , because $r \geq 0$ on S. Additionally, for every $\zeta \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ , we have:

$$
\mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } = \mathbf { M } ^ { 1 / 2 } ( \mathbf { I } _ { m } - \zeta \mathbf { N } ) \mathbf { M } ^ { 1 / 2 } .
$$

Now, define:

$$
c _ { K } = \operatorname* { i n f } _ { \zeta \in K } \operatorname* { i n f } _ { \lambda \geq 0 } | 1 - \zeta \lambda | .
$$

Since K is compact in $\mathbb { C } \setminus \mathbb { R } ^ { + }$ , this constant is positive. Hence, for $\zeta \in K$ , we have:

$$
\| ( \mathbf { I } _ { m } - \zeta \mathbf { N } ) ^ { - 1 } \| _ { \mathrm { o p } } \leq c _ { K } ^ { - 1 } ,
$$

where $\| \cdot \| _ { \mathrm { o p } }$ is the operator norm. Therefore:

$$
\left\| { ( \mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } ) } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq \| \mathbf { M } ^ { - 1 / 2 } \| _ { \mathrm { o p } } ^ { 2 } \| ( \mathbf { I } _ { m } - \zeta \mathbf { N } ) ^ { - 1 } \| _ { \mathrm { o p } } \leq \frac { 1 } { \eta c _ { K } } .
$$

It follows that:

$$
\frac { 1 } { d } \left| \mathrm { T r } \Big ( \big ( \mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } \big ) ^ { - 1 } \Big ) \right| \leq \frac { m } { d } \Big \| \big ( \mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } \big ) ^ { - 1 } \Big \| _ { \mathrm { o p } } \leq \frac { 1 } { \eta c _ { K } } ,
$$

because $m \leq d$ . Similarly,

$$
\frac { 1 } { d } \Big \| \big ( \mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } \big ) ^ { - 1 } \Big \| _ { F } ^ { 2 } \leq \frac { m } { d } \Big \| \big ( \mathbf { X } _ { 0 } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } _ { 0 } \big ) ^ { - 1 } \Big \| _ { \mathrm { o p } } ^ { 2 } \leq \frac { 1 } { \eta ^ { 2 } c _ { K } ^ { 2 } } .
$$

Both bounds are uniform over $\zeta \in K , d \geq 1$ , and $\mathbf { X } _ { 0 }$ .

We now state the complex version of Theorem 22.

Theorem 23. Let $\mathbf { X } \in \mathbb { R } ^ { d \times m }$ be uniformly distributed on $\mathrm { S t } ( d , m )$ . Let $\mathbf { D } \in { \cal { S } } _ { d } ( \mathbb { R } )$ be deterministic, with spectrum contained in a compact set $S \subset \mathbb { R }$ independent of $d ,$ and assume that its empirical spectral measure $\mu _ { d }$ converges weakly to a probability measure $\mu .$ Moreover, assume that $m / d \xrightarrow [ d  \infty ] { } \kappa \in ( 0 , 1 )$

Let $q , r : S \to \mathbb { R } ^ { + }$ be continuous functions such that $q ( x ) \geq \eta > 0$ on $S$ . For $\zeta \in \mathbb { C } \backslash \mathbb { R } ^ { + }$ , set:

$$
\begin{array} { r } { a _ { \zeta } ( x ) = q ( x ) - \zeta r ( x ) . } \end{array}
$$

Then, there is a unique holomorphic function $h : \mathbb { C } \setminus \mathbb { R } ^ { + } \to \mathbb { C }$ such that $h ( \zeta ) > 0 \ f o r \ \zeta < 0$ and

$$
1 = \int \frac { \mu ( \mathrm { d } x ) } { 1 - \kappa + h ( \zeta ) a _ { \zeta } ( x ) } .\tag{55}
$$

Let $b : S  \mathbb { C }$ be continuous. Then, with probability one, for every compact set $K \subset \mathbb { C } \backslash \mathbb { R } ^ { + }$

$$
\operatorname* { s u p } _ { \zeta \in K } \left| \frac { 1 } { d } \mathrm { T r } \left( \mathbf { X } ^ { \top } b ( \mathbf { D } ) \mathbf { X } ( \mathbf { X } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X } ) ^ { - 1 } \right) - h ( \zeta ) \int \frac { b ( x ) } { 1 - \kappa + h ( \zeta ) a _ { \zeta } ( x ) } \mu ( \mathrm { d } x ) \right| \xrightarrow [ d \to \infty ] { 1 } 0 .\tag{56}
$$

Proof. Let us first prove the claim on the function $h ( \zeta )$ . For fixed $\zeta < 0$ , the function $a _ { \zeta }$ is positive and satisfies the assumptions of Theorem 22. Denoting by $h _ { + } ( \zeta )$ the corresponding positive solution of equation (55), Theorem 22 gives the desired convergence on the negative real axis for this fixed $\zeta$ and every fixed continuous $b .$ In particular, when $b = 1 , h _ { + } ( \zeta )$ is obtained as the limit of

$$
H _ { d } ( \zeta ) : = T _ { d } ( a _ { \zeta } , 1 ) = \frac { 1 } { d } \mathrm { T r } \Big ( \big ( { \bf X } ^ { \top } a _ { \zeta } ( { \bf D } ) { \bf X } \big ) ^ { - 1 } \Big ) ,
$$

where we exploited that $\mathbf { X } ^ { \top } \mathbf { X } = \mathbf { I } _ { m }$ . Then, for every compact set $K \subset \mathbb { C } \backslash \mathbb { R } ^ { + }$ , Lemma C.1 guarantees that:

$$
\operatorname* { s u p } _ { \zeta \in K } \operatorname* { s u p } _ { d \geq 1 } \left| H _ { d } ( \zeta ) \right| < \infty .
$$

Let $D ^ { - }$ be a dense countable subset of $( - \infty , 0 )$ . Since Theorem 22 guarantees convergence almost surely for each fixed $\zeta \in D ^ { - }$ , it holds simultaneously on $D ^ { - }$ on an event of probability one. On this event, Theorem 16 gives convergence of $H _ { d }$ to a holomorphic function h on $\mathbb { C } \setminus \mathbb { R } ^ { + }$ and $h = h _ { + }$ on $D ^ { - }$

Let us now show that h satisfies equation (55) everywhere. To do ${ \mathrm { s o } } ,$ we show that the RHS of equation (55) is holomorphic in ζ and use the identity theorem Theorem 17. Recall that ℑ denotes the imaginary part, and denote by z¯ the complex conjugate of $z \in \mathbb { C }$ . Let us consider $\zeta \in \mathbb { C }$ such that $\Im \zeta > 0$ . Then:

$$
\mathbb { S } H _ { d } ( \zeta ) = \frac { \mathbb { S } \zeta } { d } \mathrm { T r } \Big ( \big ( \mathbf { X } ^ { \top } { a } _ { \zeta } ( \mathbf { D } ) \mathbf { X } \big ) ^ { - 1 } \mathbf { X } ^ { \top } r ( \mathbf { D } ) \mathbf { X } \big ( \mathbf { X } ^ { \top } { a } _ { \widetilde { \zeta } } ( \mathbf { D } ) \mathbf { X } \big ) ^ { - 1 } \Big ) \geq 0 ,
$$

since $\mathbf { X } ^ { \top } r ( \mathbf { D } ) \mathbf { X } \succeq 0$ , and in the limit we also have $\Im h ( \zeta ) \geq 0$ . By the open mapping theorem (Bak and Newman, 2010, Theorem 7.1), either $\Im h ( \zeta ) > 0$ on the upper half-plane, or h is constant and real. This is the case if $r = 0$ µ-almost everywhere. This constant then can be found by taking the limit on the negative real axis, and therefore is positive. For $\Im \zeta < 0$ , since $H _ { d } ( \bar { \zeta } ) = \overline { { { H _ { d } ( \zeta ) } } }$ , we get by conjugation that either $\Im h ( \zeta ) < 0$ or h is a positive constant.

With these conclusions, let us study the integral in equation (55), applied to the function h. We first show that, for every compact set $K \subset \mathbb { C } \backslash \mathbb { R } ^ { + }$

$$
\operatorname* { i n f } _ { \zeta \in K } \operatorname* { i n f } _ { x \in S } \left| 1 - \kappa + h ( \zeta ) a _ { \zeta } ( x ) \right| > 0 .\tag{57}
$$

For $\zeta < 0$ , both $h ( \zeta )$ and $a _ { \zeta } ( x )$ are positive, so the denominator is positive. If $\Im \zeta \neq 0$ and the denominator vanished, then $h ( \zeta ) = - ( 1 - \kappa ) / ( q ( x ) - \zeta r ( x ) )$ . $\mathrm { I f } \ r ( x ) \ > \ 0$ , this implies $\Im h ( \zeta ) \Im \zeta < 0$ . If $r ( x ) = 0$ , then $h ( \zeta ) = - ( 1 - \kappa ) / q ( x )$ is a negative real number. Both cases are excluded, since we have shown that either $\Im h ( \zeta ) \Im \zeta > 0$ , or h is a positive constant. Thus the denominator never vanishes. Since K and S are both compact, it is easily seen that the uniform bound in equation (57) holds. Therefore, the function:

$$
\zeta \longmapsto \int { \frac { \mu ( \mathrm { d } x ) } { 1 - \kappa + h ( \zeta ) a _ { \zeta } ( x ) } }
$$

is holomorphic (since h is and the denominator is uniformly bounded away from zero) on $\mathbb { C } \backslash \mathbb { R } ^ { + }$ As it is equal to 1 on D<sup>−</sup>, the identity theorem (Theorem 17) shows that it is equal to 1 on $\mathbb { C } \backslash \mathbb { R } ^ { + }$ Uniqueness of the holomorphic solution on $\mathbb { C } \setminus \mathbb { R } ^ { + }$ follows from the uniqueness on $( - \infty , 0 )$ and the same theorem.

Let us now prove the convergence in equation (56). We start by fixing $b : S  \mathbb { C }$ continuous. As previously, since $D ^ { - }$ is countable, there is a probability one event for which $T _ { d } ( a _ { \zeta } , b )$ converges to its desired limit for all $\zeta \in D ^ { - }$ . Moreover, as a consequence of the Cauchy–Schwarz inequality and the fact that $\mathbf { X } ^ { \top } \mathbf { X } = \mathbf { I } _ { m }$ ，

$$
\begin{array} { r l } & { \left| T _ { d } ( a _ { \zeta } , b ) \right| = \frac { 1 } { d } \left| \mathrm { T r } \Big ( b ( { \bf D } ) { \bf X } ( { \bf X } ^ { \top } a _ { \zeta } ( { \bf D } ) { \bf X } ) ^ { - 1 } { \bf X } ^ { \top } \Big ) \right| } \\ & { \qquad \leq \sqrt { \frac { 1 } { d } \| b ( { \bf D } ) \| _ { F } ^ { 2 } } \sqrt { \frac { 1 } { d } \big \| \big ( { \bf X } ^ { \top } a _ { \zeta } ( { \bf D } ) { \bf X } \big ) ^ { - 1 } \big \| _ { F } ^ { 2 } } . } \end{array}
$$

Then, Lemma C.1 guarantees that $\zeta \mapsto T _ { d } ( a _ { \zeta } , b )$ is holomorphic and locally bounded on $\mathbb { C } \setminus \mathbb { R } ^ { + }$ since $\mathbf { X } ^ { \top } a _ { \zeta } ( \mathbf { D } ) \mathbf { X }$ is afine in $\zeta$ and invertible there. Additionally, equation (57) guarantees that the limiting function

$$
\zeta \longmapsto h ( \zeta ) \int \frac { b ( x ) } { 1 - \kappa + h ( \zeta ) a _ { \zeta } ( x ) } \mu ( \mathrm { d } x )
$$

is holomorphic on $\mathbb { C } \setminus \mathbb { R } ^ { + }$ . On the event of probability one chosen above, $T _ { d } ( a _ { \zeta } , b )$ converges to this function for every $\zeta \in D ^ { - }$ . Hence Corollary 18 gives uniform convergence on every compact subset of $\mathbb { C } \setminus \mathbb { R } ^ { + }$ , which proves equation (56). □

In the following theorem, we consider the case $r = 0$ , corresponding to the case where h is real, and derive the high-dimensional limit for a second-order version of the previous trace.

Theorem 24. Under the assumptions of Theorem 23, with $r = 0$ , let h be the unique positive solution of equation (55) and $b _ { 1 } , b _ { 2 } : S $ R be continuous, and define:

$$
\mathbf { H } = \mathbf { X } { \big ( } \mathbf { X } ^ { \top } q ( \mathbf { D } ) \mathbf { X } { \big ) } ^ { - 1 } \mathbf { X } ^ { \top } .
$$

Then:

$$
\begin{array} { r l } & { \frac { 1 } { d } \mathrm { T r } \Big ( b _ { 1 } ( { \bf D } ) { \bf H } b _ { 2 } ( { \bf D } ) { \bf H } \Big ) \xrightarrow { \mathrm { a . s . } } \frac { ( 1 - \kappa ) h } { J } \left( \int \frac { b _ { 1 } ( x ) } { \Delta ( x ) ^ { 2 } } \mu ( \mathrm { d } x ) \right) \left( \int \frac { b _ { 2 } ( x ) } { \Delta ( x ) ^ { 2 } } \mu ( \mathrm { d } x ) \right) } \\ & { \quad \quad \quad \quad \quad + h ^ { 2 } \int \frac { b _ { 1 } ( x ) b _ { 2 } ( x ) } { \Delta ( x ) ^ { 2 } } \mu ( \mathrm { d } x ) , } \end{array}\tag{58}
$$

where

$$
\Delta ( x ) = 1 - \kappa + h q ( x ) , J = \int \frac { q ( x ) } { \Delta ( x ) ^ { 2 } } \mu ( \mathrm { d } x ) .
$$

Proof. As in the proof of Theorem 22, we write $\mathbf { X } = \mathbf { G } ( \mathbf { G } ^ { \top } \mathbf { G } ) ^ { - 1 / 2 }$ , where $\mathbf { G } \in \mathbb { R } ^ { d \times m }$ has i.i.d. $\mathcal { N } ( 0 , 1 / m )$ entries. Hence H = $\mathbf { G } \big ( \mathbf { G } ^ { \top } q ( \mathbf { D } ) \mathbf { G } \big ) ^ { - 1 } \mathbf { G } ^ { \top }$ . Since $q ( \mathbf { D } ) \succeq \eta \mathbf { I } _ { d }$ has full rank and $\kappa < 1$ LeJeune et al. (2024, Theorem 4.8) applies with $\lambda = 0$ and gives the deterministic equivalent:

$$
\mathbf { H } b _ { 2 } ( \mathbf { D } ) \mathbf { H } \simeq \left( q ( \mathbf { D } ) + \nu _ { d } \mathbf { I } _ { d } \right) ^ { - 1 } \left( b _ { 2 } ( \mathbf { D } ) + \theta _ { d } \mathbf { I } _ { d } \right) \left( q ( \mathbf { D } ) + \nu _ { d } \mathbf { I } _ { d } \right) ^ { - 1 } .\tag{59}
$$

As this deterministic equivalent requires $b _ { 2 } ( { \bf D } )$ to be positive semidefinite, we prove it for $b _ { 2 } \geq 0$ and extend to a general continuous b<sub>2</sub> by decomposing $b _ { 2 } = \operatorname* { m a x } ( 0 , { b _ { 2 } } ) - \operatorname* { m a x } ( 0 , - { b _ { 2 } } )$ , and using the linearity of equation (58).

Recall that $\nu _ { d }$ is the unique positive solution of (52) and converges as $d \to \infty$ to $\nu ,$ solving (54). Moreover, $h = ( 1 - \kappa ) / \nu ,$ , and $\theta _ { d }$ is given by

$$
\theta _ { d } = \frac { \nu _ { d } \mathrm { T r } \Big ( \big ( q ( \mathbf { D } ) + \nu _ { d } \mathbf { I } _ { d } \big ) ^ { - 1 } b _ { 2 } \big ( \mathbf { D } \big ) \big ( q ( \mathbf { D } ) + \nu _ { d } \mathbf { I } _ { d } \big ) ^ { - 1 } \Big ) } { \mathrm { T r } \Big ( q ( \mathbf { D } ) \big ( q ( \mathbf { D } ) + \nu _ { d } \mathbf { I } _ { d } \big ) ^ { - 2 } \Big ) } .
$$

Now, we have the deterministic limits

$$
\begin{array} { r } { \frac { 1 } { d } \mathrm { T r } \Big ( \big ( q ( { \mathbf D } ) + \nu _ { d } { \mathbf I } _ { d } \big ) ^ { - 1 } b _ { 2 } ( { \mathbf D } ) \big ( q ( { \mathbf D } ) + \nu _ { d } { \mathbf I } _ { d } \big ) ^ { - 1 } \Big ) \xrightarrow [ d  \infty ] { } \int \frac { b _ { 2 } ( x ) } { \big ( q ( x ) + \nu ) ^ { 2 } } \mu ( \mathrm { d } x ) , } \\ { \frac { 1 } { d } \mathrm { T r } \Big ( q ( { \mathbf D } ) \big ( q ( { \mathbf D } ) + \nu _ { d } { \mathbf I } _ { d } \big ) ^ { - 2 } \Big ) \xrightarrow [ d  \infty ] { } \int \frac { q ( x ) } { \big ( q ( x ) + \nu \big ) ^ { 2 } } \mu ( \mathrm { d } x ) . } \end{array}
$$

The convergence is guaranteed since $\mu _ { d } \Rightarrow \mu$ and $\mu _ { d }$ is integrated against d-dependent functions that converge uniformly on S. Hence, $\theta _ { d }$ converges as $d \to \infty$ . Additionally, from the deterministic equivalent in equation (59), and the fact that

$$
\operatorname* { s u p } _ { d \geq 1 } \frac { 1 } { d } \mathrm { T r } \left| b _ { 1 } ( { \bf D } ) \right| \leq \operatorname* { s u p } _ { x \in S } \left| b _ { 1 } ( x ) \right| < \infty ,
$$

we have the almost sure limit

$$
\frac { 1 } { d } \mathrm { T r } \Big ( b _ { 1 } ( { \bf D } ) { \bf H } b _ { 2 } ( { \bf D } ) { \bf H } \Big ) - \frac { 1 } { d } \mathrm { T r } \Big ( b _ { 1 } ( { \bf D } ) \big ( b _ { 2 } ( { \bf D } ) + \theta _ { d } { \bf I } _ { d } \big ) \big ( q ( { \bf D } ) + \nu _ { d } { \bf I } _ { d } \big ) ^ { - 2 } \Big ) \xrightarrow [ d  \infty ] { \mathrm { a . s . } } 0 .
$$

To conclude, we have the limit for the deterministic trace:

$$
\frac { 1 } { d } \mathrm { T r } \Big ( b _ { 1 } ( { \bf D } ) \big ( b _ { 2 } ( { \bf D } ) + \theta _ { d } { \bf I } _ { d } \big ) \big ( q ( { \bf D } ) + \nu _ { d } { \bf I } _ { d } \big ) ^ { - 2 } \Big ) \xrightarrow [ d  \infty ] { b _ { 1 } ( x ) b _ { 2 } ( x ) } \mu ( \mathrm { d } x ) + \theta \int \frac { b _ { 1 } ( x ) } { ( q ( x ) + \nu ) ^ { 2 } } \mu ( \mathrm { d } x ) .
$$

Combining the relationship between ν and h with the previous equations gives the result.

## C.2 Stieltjes Transforms

In this section we study the Stieltjes transforms $\mathfrak { m } _ { t } ^ { U } , \mathfrak { m } _ { t } ^ { V } , \mathfrak { m } _ { t } ^ { Z }$ defined in equations (10), (11) and prove the results stated in Section 4.1.

## C.2.1 High-Dimensional Limit

We now prove Theorem 4 which derives the high-dimensional limit of the Stieltjes transforms.

Step 1. Stieltjes transforms of the Gram matrices. To start, we prove the high-dimensional limit for $\mathfrak { m } _ { t } ^ { U } , \bar { \mathfrak { m } _ { t } ^ { V } }$ . Let

$$
\mathbf { S } = \left( \mathbf { C } ^ { \top } \mathbf { C } \right) ^ { 1 / 2 } , \qquad \mathbf { T } _ { \rho } = \left( \mathbf { S } ^ { 2 } + \rho ^ { 2 } \mathbf { I } _ { d } \right) ^ { 1 / 2 } .
$$

We define:

$$
\begin{array} { r l r l } & { \mathbf { K } ( t ) = \mathbf { R } ^ { \top } q _ { t } ( \Gamma _ { \rho } ) \mathbf { R } } & { \qquad } & { \in \mathcal { S } _ { m } ^ { + + } ( \mathbb { R } ) , } \\ & { \mathbf { B } _ { U } ( t ) = \sqrt { 2 \rho } \chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } } & { \qquad } & { \in \mathbb { R } ^ { p \times m } , } \\ & { \mathbf { B } _ { V } ( t ) = \sqrt { 2 \rho } \chi _ { t } ^ { V } ( \mathbf { S } ) \mathbf { R } } & { \qquad } & { \in \mathbb { R } ^ { d \times m } , } \end{array}
$$

so that, as a consequence of Lemma B.1

$$
\begin{array} { r } { \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } = \mathbf { B } _ { U } ( t ) \mathbf { K } ( t ) ^ { - 1 } \mathbf { B } _ { U } ( t ) ^ { \top } , } \\ { \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } = \mathbf { B } _ { V } ( t ) \mathbf { K } ( t ) ^ { - 1 } \mathbf { B } _ { V } ( t ) ^ { \top } . } \end{array}
$$

Now using the Woodbury matrix identity, we have for $w \in \mathbb { C } \backslash$ R<sup>+</sup>,

$$
\begin{array} { r l } & { \big ( \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } - w \mathbf { I } _ { p } \big ) ^ { - 1 } = - \frac { 1 } { w } \mathbf { I } _ { p } - \frac { 1 } { w ^ { 2 } } \mathbf { B } _ { U } ( t ) \bigg ( \mathbf { K } ( t ) - \frac { 1 } { w } \mathbf { B } _ { U } ( t ) ^ { \top } \mathbf { B } _ { U } ( t ) \bigg ) ^ { - 1 } \mathbf { B } _ { U } ( t ) ^ { \top } , } \\ & { \big ( \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } - w \mathbf { I } _ { d } \big ) ^ { - 1 } = - \frac { 1 } { w } \mathbf { I } _ { d } - \frac { 1 } { w ^ { 2 } } \mathbf { B } _ { V } ( t ) \bigg ( \mathbf { K } ( t ) - \frac { 1 } { w } \mathbf { B } _ { V } ( t ) ^ { \top } \mathbf { B } _ { V } ( t ) \bigg ) ^ { - 1 } \mathbf { B } _ { V } ( t ) ^ { \top } . } \end{array}
$$

We now state a useful lemma.

Lemma C.2. Let $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ . For any $\psi , G \colon \mathbb { R }  \mathbb { R }$ such that $G ( 0 ) = 0$ , we have:

$$
G ( \mathbf { C } ) ^ { \top } \psi ( \mathbf { C } \mathbf { C } ^ { \top } ) G ( \mathbf { C } ) = G ( \mathbf { S } ) ^ { 2 } \psi ( \mathbf { S } ^ { 2 } ) .
$$

Proof. Write $\begin{array} { r } { \mathbf { C } = \sum _ { i = 1 } ^ { m ^ { * } } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ over the positive singular values. Then, we have

$$
G ( \mathbf { C } ) = \sum _ { i = 1 } ^ { m ^ { * } } G ( \sigma _ { i } ) \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } , \qquad \psi ( \mathbf { C } \mathbf { C } ^ { \top } ) = \sum _ { i } \psi ( \sigma _ { i } ^ { 2 } ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } + \psi ( 0 ) \mathbf { P } _ { \mathrm { K e r } } ,
$$

where $\mathbf { P } _ { \mathrm { K e r } }$ is the orthogonal projection onto $\mathrm { K e r } ( \mathbf { C C } ^ { \top } )$ . This projection vanishes after multiplication by $G ( \mathbf { C } )$ , which gives

$$
G ( \mathbf { C } ) ^ { \top } \psi ( \mathbf { C C } ^ { \top } ) G ( \mathbf { C } ) = \sum _ { i = 1 } ^ { m ^ { * } } G ( \sigma _ { i } ) ^ { 2 } \psi ( \sigma _ { i } ^ { 2 } ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } = G ( \mathbf { S } ) ^ { 2 } \psi ( \mathbf { S } ^ { 2 } ) .
$$

□

Therefore, since $\chi _ { t } ^ { U } ( 0 ) = 0$ , this gives:

$$
\begin{array} { r } { \mathbf { B } _ { U } ( t ) ^ { \top } \mathbf { B } _ { U } ( t ) = 2 \rho \mathbf { R } ^ { \top } \chi _ { t } ^ { U } ( \mathbf { S } ) ^ { 2 } \mathbf { R } , } \\ { \mathbf { B } _ { V } ( t ) ^ { \top } \mathbf { B } _ { V } ( t ) = 2 \rho \mathbf { R } ^ { \top } \chi _ { t } ^ { V } ( \mathbf { S } ) ^ { 2 } \mathbf { R } . } \end{array}
$$

Now, we have

$$
\begin{array} { r l } & { { \bf K } ( t ) - \frac { 1 } { w } { \bf B } _ { U } ( t ) ^ { \top } { \bf B } _ { U } ( t ) = { \bf R } ^ { \top } \left( q _ { t } ( { \bf r } _ { \rho } ) - \frac { 2 \rho } { w } \chi _ { t } ^ { U } ( { \bf S } ) ^ { 2 } \right) { \bf R } , } \\ & { { \bf K } ( t ) - \frac { 1 } { w } { \bf B } _ { V } ( t ) ^ { \top } { \bf B } _ { V } ( t ) = { \bf R } ^ { \top } \left( q _ { t } ( { \bf r } _ { \rho } ) - \frac { 2 \rho } { w } \chi _ { t } ^ { V } ( { \bf S } ) ^ { 2 } \right) { \bf R } . } \end{array}
$$

Everything is now expressed in terms of R and the deterministic symmetric matrix $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ Using the functions defined in Theorem 4, we have

$$
2 \rho \chi _ { t } ^ { U } ( \mathbf { S } ) ^ { 2 } = r _ { t } ^ { \mathrm { { L } } } ( { \mathbf { r } } _ { \rho } ) , ~ 2 \rho \chi _ { t } ^ { V } ( \mathbf { S } ) ^ { 2 } = r _ { t } ^ { \mathrm { { R } } } ( { \mathbf { r } } _ { \rho } ) .
$$

Then, putting everything together, we arrive at the expressions:

$$
\mathfrak { m } _ { t } ^ { U } ( w ) = - \frac { 1 } { w } - \frac { 1 } { w ^ { 2 } p } \mathrm { T r } \Bigg [ \left( \mathbf { R } ^ { \top } \left( q _ { t } ( \mathbf { r } _ { \rho } ) - \frac { 1 } { w } r _ { t } ^ { \bot } ( \mathbf { r } _ { \rho } ) \right) \mathbf { R } \right) ^ { - 1 } \mathbf { R } ^ { \top } r _ { t } ^ { \bot } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Bigg ] ,
$$

$$
\mathfrak { m } _ { t } ^ { V } ( w ) = - \frac { 1 } { w } - \frac { 1 } { w ^ { 2 } d } \mathrm { T r } \Bigg [ \left( \mathbf { R } ^ { \top } \left( q _ { t } ( \mathbf { r } _ { \rho } ) - \frac { 1 } { w } r _ { t } ^ { \mathsf { R } } ( \mathbf { r } _ { \rho } ) \right) \mathbf { R } \right) ^ { - 1 } \mathbf { R } ^ { \top } r _ { t } ^ { \mathsf { R } } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Bigg ] .
$$

The empirical spectral distribution of $( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ converges to ${ \tilde { \mu } } _ { C } = ( 1 - \beta ) \delta _ { 0 } + \beta \mu _ { C }$ , with $\beta = \operatorname* { m i n } ( 1 , \tau )$ , and $\tau = \mathrm { l i m } _ { d  \infty } p / d$ . To conclude, we apply Theorem 23, with $\zeta = w ^ { - 1 } \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ We get the almost sure limits:

$$
\begin{array} { r l } & { \mathfrak { m } _ { t } ^ { U } ( w ) \xrightarrow { \mathrm { \tiny ~ a . s . } } - \frac { 1 } { w } - \frac { 1 } { \tau w ^ { 2 } } h _ { t } ^ { \mathrm { L } } ( w ) \displaystyle \int \frac { r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) } { 1 - \kappa + h _ { t } ^ { \mathrm { L } } ( w ) \left( q _ { t } ( \gamma _ { x } ) - \frac { 1 } { w } r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) \right) } \tilde { \mu } _ { C } ( \mathrm { d } x ) , } \\ & { \mathfrak { m } _ { t } ^ { V } ( w ) \xrightarrow { \mathrm { \tiny ~ a . s . } } - \frac { 1 } { w } - \frac { 1 } { w ^ { 2 } } h _ { t } ^ { \mathrm { R } } ( w ) \displaystyle \int \frac { r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) } { 1 - \kappa + h _ { t } ^ { \mathrm { R } } ( w ) \left( q _ { t } ( \gamma _ { x } ) - \frac { 1 } { w } r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) \right) } \tilde { \mu } _ { C } ( \mathrm { d } x ) , } \end{array}
$$

where $\gamma _ { x } = \sqrt { x ^ { 2 } + \rho ^ { 2 } }$ and $h _ { t } ^ { \mathsf { L } } , h _ { t } ^ { \mathsf { R } }$ are the unique holomorphic functions on $\mathbb { C } \setminus \mathbb { R } ^ { + }$ , positive for $w < 0$ , solving

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + h _ { t } ^ { \mathrm { a } } ( w ) \left( q _ { t } ( \gamma _ { x } ) - \frac { 1 } { w } r _ { t } ^ { \mathrm { a } } ( \gamma _ { x } ) \right) } ,
$$

for $\mathsf { a } = \mathsf { L } , \mathsf { R } .$ . Finally, define

$$
g _ { t } ^ { \mathsf { L } } ( w ) = - \frac { 1 } { \kappa w } h _ { t } ^ { \mathsf { L } } ( w ) , \qquad g _ { t } ^ { \mathsf { R } } ( w ) = - \frac { 1 } { \kappa w } h _ { t } ^ { \mathsf { R } } ( w ) ,
$$

so that both functions are positive for $w < 0$ . Their self-consistent equations are

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { a } } ( w ) \big ( r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } .
$$

Then, using that:

$$
\frac { \kappa g _ { t } ^ { \mathrm { a } } ( w ) r _ { t } ^ { \mathrm { a } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathrm { a } } ( w ) \big ( r _ { t } ^ { \mathrm { a } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } = 1 - \frac { 1 - \kappa - \kappa w g _ { t } ^ { \mathrm { a } } ( w ) q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathrm { a } } ( w ) \big ( r _ { t } ^ { \mathrm { a } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } ,
$$

and the self-consistent equation for $g _ { t } ^ { \mathsf { a } }$ , we get

$$
\begin{array} { r l } & { \mathfrak { m } _ { t } ^ { U } ( w ) \xrightarrow { \mathrm { \tiny ~ a . s . } } - \frac { 1 } { w } \left( 1 - \frac { \kappa } { \tau } \right) + \frac { \kappa } { \tau } g _ { t } ^ { \perp } ( w ) \displaystyle \int \frac { q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathrm { L } } ( w ) \left( r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \right) } \tilde { \mu } _ { C } ( \mathrm { d } x ) , } \\ & { \mathfrak { m } _ { t } ^ { V } ( w ) \xrightarrow { \mathrm { \tiny ~ a . s . } } - \frac { 1 - \kappa } { w } + \kappa g _ { t } ^ { \mathrm { R } } ( w ) \displaystyle \int \frac { q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathrm { R } } ( w ) \left( r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \right) } \tilde { \mu } _ { C } ( \mathrm { d } x ) . } \end{array}
$$

To conclude, we use the identity:

$$
r _ { t } ^ { \mathsf { R } } ( \gamma ) = r _ { t } ^ { \mathsf { L } } ( \gamma ) + 2 \rho q _ { t } ( \gamma ) ,
$$

hence $g _ { t } ^ { \mathsf { R } }$ solves the equation

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathrm { R } } ( w ) \big ( r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) + ( 2 \rho - w ) q _ { t } ( \gamma _ { x } ) \big ) } .
$$

Both $g _ { t } ^ { \mathsf { R } } ( w )$ and $g _ { t } ^ { \mathsf { L } } ( w - 2 \rho )$ are positive for $w < 0$ . By uniqueness of the holomorphic solution with this property, we obtain

$$
g _ { t } ^ { \mathsf { R } } ( w ) = g _ { t } ^ { \mathsf { L } } ( w - 2 \rho ) .
$$

Writing $g _ { t } = g _ { t } ^ { \mathsf { L } }$ proves (12) and (13).

The identity (16) also follows directly from Proposition 2, where the non-zero eigenvalues of $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ are directly related to those of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ . Set $\kappa _ { d } = m / d$ and $\tau _ { d } = p / d .$ . For any test function $f ,$ we have the exact finite-dimensional relation

$$
\int f ( x ) \mu _ { t , d } ^ { V } ( \mathrm { d } x ) = ( 1 - \kappa _ { d } ) f ( 0 ) + ( \kappa _ { d } - \tau _ { d } ) f ( 2 \rho ) + \tau _ { d } \int f ( x + 2 \rho ) \mu _ { t , d } ^ { U } ( \mathrm { d } x ) .
$$

Taking $d \to \infty$ gives equation (16).

Step 2. Prediction matrix. Let $\lambda _ { 1 } ( t ) , \ldots , \lambda _ { p } ( t )$ be the full spectrum of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ , including zero eigenvalues, and set $s _ { i } ( t ) = \sqrt { \lambda _ { i } ( t ) ( \lambda _ { i } ( t ) + 2 \rho ) }$ . By Proposition 2, these give the singular values of $\mathbf { Z } ( t )$ after adjusting for the rectangular dimensions.

Since we consider the min $( p , d )$ singular values of $\mathbf { Z } ( t )$ , there are two cases:

$\operatorname { I f } p \leq d ,$ then the singular values of $\mathbf { Z } ( t )$ , denoted $\sigma _ { 1 } ( t ) , \ldots , \sigma _ { p } ( t )$ , are precisely $s _ { 1 } ( t ) , \ldots , s _ { p } ( t )$

• If $p > d , \sigma _ { 1 } ( t ) , . . . , \sigma _ { d } ( t )$ are obtained by taking $s _ { 1 } ( t ) , \ldots , s _ { p } ( t )$ (corresponding to the full spectrum of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } )$ and removing $p - d$ zero eigenvalues.

With $\beta _ { d } = \operatorname* { m i n } ( 1 , p / d )$ , this gives the exact finite-dimensional identity

$$
\int f ( x ) \mu _ { t , d } ^ { Z } ( \mathrm { d } x ) = \left( 1 - \frac { \tau _ { d } } { \beta _ { d } } \right) f ( 0 ) + \frac { \tau _ { d } } { \beta _ { d } } \int f \left( \sqrt { x ( x + 2 \rho ) } \right) \mu _ { t , d } ^ { U } ( \mathrm { d } x ) ,\tag{60}
$$

for every bounded continuous $f .$ Taking the high-dimensional limit and using the pushforward characterization in Appendix A.1 gives equation (15).

## C.2.2 Initialization

As a consistency check, we recompute from Theorem 4 the expression of ${ \mathfrak { m } } _ { t } ^ { U } ( w ) , { \mathfrak { m } } _ { t } ^ { V } ( w )$ at $t = 0$ leading to the spectral distributions of $\mathbf { U } ( 0 ) \mathbf { U } ( 0 ) ^ { \top }$ and $\mathbf { V } ( 0 ) \mathbf { V } ( 0 ) ^ { \top }$

Lemma C.3. At $t = 0$ , Theorem 4 yields

$$
\mathfrak { m } _ { 0 } ^ { U } ( w ) = - \frac { 1 } { w } , \qquad \mathfrak { m } _ { 0 } ^ { V } ( w ) = - \frac { 1 - \kappa } { w } + \frac { \kappa } { 2 \rho - w } .\tag{61}
$$

These are the Stieltjes transforms of the probability measures:

$$
\mu _ { 0 } ^ { U } = \delta _ { 0 } , ~ \mu _ { 0 } ^ { V } = ( 1 - \kappa ) \delta _ { 0 } + \kappa \delta _ { 2 \rho } .\tag{62}
$$

Proof. $\mathrm { A t } t = 0$ , we have $q _ { 0 } ( \gamma ) = 1$ and $r _ { 0 } ^ { \mathsf { L } } ( \gamma ) = 0$ . From equation (14), we then get $g _ { 0 } ( w ) = - w ^ { - 1 }$ and Theorem 4 gives the desired expressions. Their form identifies the corresponding Dirac masses. □

This result is consistent with Assumption 1.4, where we chose U to be initialized at zero, and $\mathbf { V } ( 0 ) \mathbf { V } ( 0 ) ^ { \top } = 2 \rho \mathbf { R } \mathbf { R } ^ { \top }$ , where $\mathbf { R R } ^ { \top }$ is an orthogonal projector of rank $m ,$ , asymptotically κd.

## C.2.3 Limiting Stieltjes Transforms

We now prove Corollary 5.

Step 1. Gram matrices. We start by proving the limit of ${ \mathfrak { m } } _ { t } ^ { U } , { \mathfrak { m } } _ { t } ^ { V }$ . We consider $\xi < 0$ , and then extend the convergence by using Corollary 18. Now, $g _ { t } ( \xi )$ is the unique positive solution of:

$$
1 = I _ { t } ( g ) : = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g \big ( r _ { t } ^ { \perp } ( \gamma _ { x } ) - \xi q _ { t } ( \gamma _ { x } ) \big ) } .\tag{63}
$$

$\mathrm { O n \mathbb { R } ^ { + } }$ , the map $g \mapsto I _ { t } ( g )$ is continuous and strictly decreasing from $( 1 - \kappa ) ^ { - 1 }$ to 0. Recall that $L$ is the uniform upper bound on the singular values of C in Assumption 1.3. A direct analysis gives, uniformly for $\gamma$ on $[ \rho , \sqrt { \rho ^ { 2 } + L ^ { 2 } } ]$ ，

$$
e ^ { - 2 t \gamma } q _ { t } ( \gamma ) \xrightarrow [ t \to \infty ] { } q _ { \infty } ( \gamma ) , \qquad e ^ { - 2 t \gamma } r _ { t } ^ { \downarrow } ( \gamma ) \xrightarrow [ t \to \infty ] { } r _ { \infty } ( \gamma ) ,
$$

where

$$
q _ { \infty } ( \gamma ) = { \frac { \rho } { 2 \gamma } } \left( 1 + { \frac { \rho } { \gamma } } \right) , r _ { \infty } ( \gamma ) = { \frac { \rho } { 2 } } \left( 1 - { \frac { \rho ^ { 2 } } { \gamma ^ { 2 } } } \right) .
$$

Denote $\begin{array} { r } { T _ { C } ( a ) = \beta \kappa ^ { * } \int _ { a } ^ { \infty } \phi _ { C } ( x ) \mathrm { d } x } \end{array}$ . We apply Proposition 19 with the quantities:

$$
h _ { t } = g _ { t } ( \xi ) , \qquad a _ { t } ( x ) = \kappa \big ( r _ { t } ^ { \perp } ( \gamma _ { x } ) - \xi q _ { t } ( \gamma _ { x } ) \big ) ,
$$

with parameter $t \to \infty$ . Consider $a > 0$ and choose $c _ { t } = e ^ { - 2 t \gamma _ { a } }$ . We have:

$$
c _ { t } a _ { t } ( x ) \xrightarrow [ { t \to \infty } ] { } \left\{ 0 , \begin{array} { l l } { x < a , } \\ { + \infty , } & { x > a . } \end{array} \right.
$$

Therefore, the limit function in Proposition 19 is given by:

$$
\Phi _ { a } ( u ) = \frac { 1 - T _ { C } ( a ) } { 1 - \kappa } ,
$$

for any u $> 0$ . Therefore, if $T _ { C } ( a ) > \kappa$ , then $\Phi _ { a } ( u ) < 1$ for all $u > 0 ,$ and 2. (ii) in Proposition 19 gives that $g _ { t } ( \xi ) e ^ { 2 t \gamma _ { a } } \longrightarrow 0$ . Likewise, if $T _ { C } ( a ) < \kappa$ , then 2. (iii) in Proposition 19 guarantees that $g _ { t } ( \xi ) e ^ { 2 t \gamma _ { a } } \longrightarrow + \infty$

In the underparameterized case, the asymptotic behavior of $g _ { t } ( \xi )$ is therefore determined by the level set $\{ x > 0 : T _ { C } ( x ) = \kappa \}$ . If this set contains an interval, $T _ { C }$ is constant on this interval and $\phi _ { C }$ vanishes almost everywhere on its interior. Therefore any choice of $x _ { * }$ in this set leads to the same limiting integrals.

In the overparameterized case $\kappa > \beta \kappa ^ { * }$ , we always have $T _ { C } ( a ) < \kappa$ . Then, we rather consider $a = 0$ , so that $c _ { t } = e ^ { - 2 \rho t }$ . In this case we have for $u > 0$ 2

$$
\Phi _ { 0 } ( u ) = \frac { 1 - \beta \kappa ^ { * } } { 1 - \kappa - \kappa \xi u } .
$$

Hence $\Phi _ { 0 } ( u ) = 1$ has a unique solution on $( 0 , \infty )$ . Applying 2. (i) in Proposition 19 guarantees the convergence:

$$
\kappa e ^ { 2 \rho t } g _ { t } ( \xi ) \xrightarrow [ t \to \infty ] { } - \frac { \kappa - \beta \kappa ^ { * } } { \xi } .\tag{64}
$$

Let us now derive the limit of the Stieltjes transforms. To do so, we consider:

$$
F _ { t } ( x , \xi ) = \frac { \kappa g _ { t } ( \xi ) q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ( \xi ) \big ( r _ { t } ^ { \perp } ( \gamma _ { x } ) - \xi q _ { t } ( \gamma _ { x } ) \big ) } ,
$$

and compute its pointwise limit. Due to the previous asymptotic behavior of $g _ { t } ( \xi )$ , we have for $x > 0 \colon$

$$
F _ { t } ( x , \xi ) \xrightarrow [ t \to \infty ] { } \left\{ \begin{array} { l l } { 0 , } & { T _ { C } ( x ) > \kappa , } \\ { \displaystyle { \frac { q _ { \infty } ( \gamma _ { x } ) } { r _ { \infty } ( \gamma _ { x } ) - \xi q _ { \infty } ( \gamma _ { x } ) } } , } & { T _ { C } ( x ) < \kappa . } \end{array} \right.\tag{65}
$$

Let us now treat the atom at $x = 0$ . In the underparameterized case, $T _ { C } ( 0 ) > \kappa .$ , so $F _ { t } ( 0 , \xi ) \to 0$ In the overparameterized case, using equation (64) leads to:

$$
F _ { t } ( 0 , \xi ) \xrightarrow [ t \to \infty ] { } - \frac { \kappa - \beta \kappa ^ { * } } { 1 - \beta \kappa ^ { * } } \frac 1 \xi .
$$

We now use dominated convergence to characterize the $t  \infty$ limit of $F _ { t } ( x , \xi )$ integrated against $\tilde { \mu } _ { C }$ . For $\xi < 0$ , the denominator in the expression of $F _ { t }$ is at least $- \xi \kappa g _ { t } ( \xi ) q _ { t } ( \gamma _ { x } )$ , so $0 \le F _ { t } ( x , \xi ) \le ( - \xi ) ^ { - 1 }$ uniformly in t and $x .$ Dominated convergence therefore gives

$$
\int F _ { t } ( x , \xi ) \tilde { \mu } _ { C } ( \mathrm { d } x ) \xrightarrow [ t  \infty ] { } - \frac { ( \kappa - \beta \kappa ^ { * } ) ^ { + } } { \xi } + \beta \kappa ^ { * } \int _ { 0 } ^ { \infty } \mathbf { 1 } _ { T _ { C } ( x ) < \kappa } \frac { q _ { \infty } ( \gamma _ { x } ) } { r _ { \infty } ( \gamma _ { x } ) - \xi q _ { \infty } ( \gamma _ { x } ) } \phi _ { C } ( x ) \mathrm { d } x .
$$

Let us now analyze the integral domains in both cases:

• In the overparameterized case, every $x > 0$ satisfies $T _ { C } ( x ) < \kappa ,$ so the integral covers the whole support of $\phi _ { C }$

• In the underparameterized case, we can replace $\{ T _ { C } ( x ) < \kappa \}$ by $\{ x > x _ { * } \}$ for any $x _ { * }$ such that $T _ { C } ( x _ { * } ) = \kappa$ , since $\phi _ { C }$ vanishes almost everywhere between any two such choices.

Finally, we use the following identity, for all $\gamma \geq \rho \colon$

$$
\frac { q _ { \infty } ( \gamma ) } { r _ { \infty } ( \gamma ) - \xi q _ { \infty } ( \gamma ) } = \frac { 1 } { \gamma - \rho - \xi } .
$$

We now consider $w < 0$ . Using the expression of ${ \mathfrak { m } } _ { t } ^ { U }$ in Theorem 4 and choosing $\xi = w$ directly yields equation (17). For ${ \mathfrak { m } } _ { t } ^ { V }$ , we choose $\xi = w - 2 \rho$ and use that $r _ { t } ^ { \mathsf { R } } = r _ { t } ^ { \mathsf { L } } + 2 \rho q _ { t }$ , so that the expression of ${ \mathfrak { m } } _ { t } ^ { V }$ in equation (13) writes:

$$
\mathfrak { m } _ { t } ^ { V } ( w ) = - \frac { 1 - \kappa } { w } + \int F _ { t } ( x , w - 2 \rho ) \tilde { \mu } _ { C } ( \mathrm { d } x ) ,
$$

and we obtain equation (18) taking $t \to \infty$ . Finally, using that both sides of equations $( 1 7 ) , ( 1 8 )$ are holomorphic on $\mathbb { C } \backslash \mathbb { R } ^ { + }$ , Corollary 18 and Lemma A.1 guarantee the result for all $w \in \mathbb { C } \backslash \mathbb { R } ^ { + }$

Step 2. Singular value distribution. Let us now derive equation (19). To do so, we use equation (15) that expresses $\mu _ { t } ^ { Z }$ in terms of $\mu _ { t } ^ { U }$ . Equation (17) identifies the limiting measure of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ as:

$$
\mu _ { \infty } ^ { U } ( \mathrm { d } x ) = \left( 1 - \frac { \operatorname* { m i n } ( \kappa , \beta \kappa ^ { * } ) } { \tau } \right) \delta _ { 0 } ( \mathrm { d } x ) + \frac { \beta \kappa ^ { * } } { \tau } ( \theta _ { \rho } ^ { - 1 } ) _ { \# } ( \mathbf { 1 } _ { x > x _ { * } } \phi _ { C } ( x ) \mathrm { d } x ) ,
$$

where $\theta _ { \rho } ( x ) = \sqrt { x ( x + 2 \rho ) }$ and $\theta _ { \rho } ^ { - 1 } ( x ) = \sqrt { \rho ^ { 2 } + x ^ { 2 } } - \rho$ on $\mathbb { R } ^ { + }$ . Since $\theta _ { \rho }$ is continuous, the pushforward relation (15) passes to the weak limit and leads to

$$
\mu _ { \infty } ^ { Z } ( \mathrm { d } x ) = \left( 1 - \frac { \operatorname* { m i n } ( \kappa , \beta \kappa ^ { * } ) } { \beta } \right) \delta _ { 0 } ( \mathrm { d } x ) + \kappa ^ { * } \mathbf { 1 } _ { x > x _ { * } } \phi _ { C } ( x ) \mathrm { d } x .
$$

Equation (19) follows by taking the Stieltjes transform of this measure.

Step 3. Comparison with finite dimension. We shall now relate Proposition 3 and Corollary 5. To do so, let $\sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { \operatorname* { m i n } ( p , d ) }$ be the singular values of $\mathbf { C } ,$ and set $r _ { d } = \operatorname* { m i n } ( m , m ^ { * } )$ The finite-dimensional limit selects the first $r _ { d }$ singular values. Under Assumption 1.3, we have $m ^ { * } / d  \beta \kappa ^ { * }$ . Combining weak convergence and the absence of an atom at $x _ { * }$ , we can show that the selection of the min $( m , m ^ { * } )$ top singular values of C is achieved by the threshold $x _ { * }$ in the high-dimensional limit. Then, if f is bounded continuous

$$
\operatorname* { l i m } _ { d \to \infty } \frac { 1 } { d } \sum _ { i = 1 } ^ { r _ { d } } f ( \sigma _ { i } ) = \beta \kappa ^ { * } \int _ { x _ { * } } ^ { \infty } f ( x ) \phi _ { C } ( x ) \mathrm { d } x .\tag{66}
$$

Starting from Proposition $^ { 3 , }$ the Stieltjes transforms of the limiting block matrices are

$$
\begin{array} { l } { { \displaystyle { \mathfrak { m } } ^ { U } ( w ) = \frac { 1 } { p } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( m , m ^ { * } ) } \frac { 1 } { \gamma _ { i } - \rho - w } - \frac { 1 } { w } \frac { p - \operatorname* { m i n } ( m , m ^ { * } ) } { p } , } \ ~ } \\ { { \displaystyle { \mathfrak { m } } ^ { V } ( w ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( m , m ^ { * } ) } \frac { 1 } { \gamma _ { i } + \rho - w } + \frac { \mathrm { r a n k } \overline { { \Pi } } } { d } \frac { 1 } { 2 \rho - w } - \frac { 1 } { w } \frac { d - \operatorname* { m i n } ( m , m ^ { * } ) - \mathrm { r a n k } \overline { { \Pi } } } { d } } , } \\ { { \displaystyle { \mathfrak { m } } ^ { Z } ( w ) = \frac { 1 } { \operatorname* { m i n } ( p , d ) } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( m , m ^ { * } ) } \frac { 1 } { \sigma _ { i } - w } - \frac { 1 } { w } \frac { \operatorname* { m i n } ( p , d ) - \operatorname* { m i n } ( m , m ^ { * } ) } { \operatorname* { m i n } ( p , d ) } , } } \end{array}
$$

where $\gamma _ { i } = \sqrt { \sigma _ { i } ^ { 2 } + \rho ^ { 2 } }$ . Using the asymptotic behavior of $m , m ^ { * } , p$ in the high-dimensional limit, along with rank $\overline { { \mathbf { I I } } } = ( m - m ^ { * } ) ^ { + }$ almost surely and equation (66), we exactly recover the expressions of Corollary 5.

## C.3 Two-Point Stieltjes Transforms

As mentioned in Section 4.1, the Stieltjes transform characterizes the spectrum of the associated matrix but does not give information on its eigenvectors. In order to characterize the eigenvectors of a random matrix in high dimension, one possibility is to introduce a reference matrix $\mathbf { A } _ { 0 } \in { \mathcal { S } } _ { d } ( \mathbb { R } )$ and compare the eigenvectors (or the projections onto the eigenspaces) of A and ${ \bf A } _ { 0 }$ . This can be achieved by considering:

$$
H ^ { A , A _ { 0 } } ( w , z ) = \frac { 1 } { d } \mathrm { T r } \Bigl ( \bigl ( \mathbf { A } - w \mathbf { I } _ { d } \bigr ) ^ { - 1 } \bigl ( \mathbf { A } _ { 0 } - z \mathbf { I } _ { d } \bigr ) ^ { - 1 } \Bigr ) .
$$

This object was studied by Ledoit and Péché (2011) for sample covariance matrices, and by Bun et al. (2018) for correlated random matrices, to analyze overlaps between eigenvectors (see also Allez and Bouchaud, 2014).

In this section, we study the two-point Stieltjes transforms:

$$
\begin{array} { l } { { \displaystyle H _ { t } ^ { U } ( w , z ) = \frac { 1 } { p } \mathrm { T r } \Big ( \big ( \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } - w \mathbf { I } _ { p } \big ) ^ { - 1 } \big ( \mathbf { C C } ^ { \top } - z \mathbf { I } _ { p } \big ) ^ { - 1 } \Big ) } , } \\ { { \displaystyle H _ { t } ^ { V } ( w , z ) = \frac { 1 } { d } \mathrm { T r } \Big ( \big ( \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } - w \mathbf { I } _ { d } \big ) ^ { - 1 } \big ( \mathbf { C } ^ { \top } \mathbf { C } - z \mathbf { I } _ { d } \big ) ^ { - 1 } \Big ) } , } \end{array}
$$

and provide a complementary high-dimensional limit for these functions. As shown in Proposition $^ { 3 , }$ in finite dimension, $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ respectively converge to spectral functions of $\mathbf { C C } ^ { \top }$ and $\mathbf { C } ^ { \top } \mathbf { C } \ ( \mathrm { u p }$ to the projection Π for $\mathbf { V } )$ . Therefore the choice of these two matrices as reference matrices for the two-point Stieltjes transform is relevant.

Claim 25. Fix $t \geq 0$ . In the high-dimensional limit, under Assumption $^ { 1 , }$ given $w , z \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ we have the almost sure convergence:

$$
\begin{array} { r l } & { H _ { t } ^ { U } ( w , z ) \xrightarrow [ d  \infty ] { } \frac { 1 } { w z } ( 1 - \frac { 1 } { \tau } ) - \frac { 1 } { \tau w } \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { x ^ { 2 } - z } \frac { 1 - \kappa - \kappa w g _ { t } ( w ) q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ( w ) ( r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) ) } , } \\ & { H _ { t } ^ { V } ( w , z ) \xrightarrow [ d  \infty ] { } - \frac { 1 } { w } \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { x ^ { 2 } - z } \frac { 1 - \kappa - \kappa w g _ { t } ( w - 2 \rho ) q _ { t } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ( w - 2 \rho ) ( r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) ) } , } \end{array}
$$

where the functions $q _ { t } , r _ { t } ^ { \mathsf { L } } , r _ { t } ^ { \mathsf { R } } , g _ { t }$ and the measure $\tilde { \mu } _ { C }$ are defined in Theorem $\it 4 .$

The proof is very similar to Appendix C.2.1, and mainly consists in applying Theorem 23 with the additional factor $( x ^ { 2 } - z ) ^ { - 1 }$ . As we do not use this result in the rest of the paper, we omit the remaining algebra.

## C.4 Range-Projector Distances

In the following we prove Theorem 6. Recall that we are interested in the distances:

$$
\Delta ^ { \mathbf { L } } ( t ) ^ { 2 } = \frac { 1 } { d } \| \mathbf { P } ^ { \mathbf { L } } ( t ) - \mathbf { P } _ { \infty } ^ { \mathbf { L } } \| _ { F } ^ { 2 } , \qquad \Delta ^ { \mathsf { R } } ( t ) ^ { 2 } = \frac { 1 } { d } \| \mathbf { P } ^ { \mathsf { R } } ( t ) - \mathbf { P } _ { \infty } ^ { \mathsf { R } } \| _ { F } ^ { 2 } ,
$$

where

$$
\begin{array} { r } { \mathbf { P } ^ { \mathrm { L } } ( t ) = \mathrm { P r o j } _ { \mathrm { I m } \mathbf { Z } ( t ) } \in \mathcal { S } _ { p } ( \mathbb { R } ) , \qquad \mathbf { P } ^ { \mathrm { R } } ( t ) = \mathrm { P r o j } _ { \mathrm { I m } \mathbf { Z } ( t ) ^ { \top } } \in \mathcal { S } _ { d } ( \mathbb { R } ) , } \end{array}
$$

are the projections onto the left and right ranges of $\mathbf { Z } ( t )$ , and $\begin{array} { r } { \mathbf { P } _ { \infty } ^ { \mathsf { L } } = \operatorname* { l i m } _ { t \to \infty } \mathbf { P } ^ { \mathsf { L } } ( t ) } \end{array}$ (and likewise for $\mathbf { P } _ { \infty } ^ { \mathsf { R } } )$

Step 1. Reduction to the Gram matrices. We start by relating the projections $\mathbf { P } ^ { \mathsf { L } } , \mathbf { P } ^ { \mathsf { R } }$ to the projections onto the ranges of $\mathbf { U } ( t ) , \mathbf { V } ( t )$ . Let us denote:

$$
\begin{array} { r } { \mathbf { P } ^ { U } ( t ) = \operatorname { P r o j } _ { \mathrm { I m } \mathbf { U } ( t ) } \in S _ { p } ( \mathbb { R } ) , \qquad \mathbf { P } ^ { V } ( t ) = \operatorname { P r o j } _ { \mathrm { I m } \mathbf { V } ( t ) } \in S _ { d } ( \mathbb { R } ) . } \end{array}
$$

Lemma C.4. Under the assumptions of Theorem 6, with probability one, we have for all $t > 0 \colon$

$$
\Delta ^ { \mathsf { L } } ( t ) ^ { 2 } = { \frac { 1 } { d } } \| \mathbf { P } ^ { U } ( t ) - \mathbf { P } _ { \infty } ^ { U } \| _ { F } ^ { 2 } , \qquad \Delta ^ { \mathsf { R } } ( t ) ^ { 2 } = { \frac { 1 } { d } } \| \mathbf { P } ^ { V } ( t ) - \mathbf { P } _ { \infty } ^ { V } \| _ { F } ^ { 2 } .\tag{67}
$$

Proof. We apply the result of Proposition 2 and use the associated notation. For $t > 0$ , with $r _ { Z } = \operatorname* { m i n } ( m , m ^ { * } )$ :

$$
\mathbf { Z } ( t ) = \sum _ { i = 1 } ^ { r _ { Z } } \sqrt { \lambda _ { i } ^ { U } ( t ) \big ( \lambda _ { i } ^ { U } ( t ) + 2 \rho \big ) } \widehat { \mathbf { u } } _ { i } ( t ) \widehat { \mathbf { v } } _ { i } ( t ) ^ { \top } .
$$

Therefore:

$$
\mathbf { P } ^ { \mathrm { L } } ( t ) = \sum _ { i = 1 } ^ { r _ { Z } } \widehat { \mathbf { u } } _ { i } ( t ) \widehat { \mathbf { u } } _ { i } ( t ) ^ { \top } , \qquad \mathbf { P } ^ { \mathrm { R } } ( t ) = \sum _ { i = 1 } ^ { r _ { Z } } \widehat { \mathbf { v } } _ { i } ( t ) \widehat { \mathbf { v } } _ { i } ( t ) ^ { \top } .
$$

Since $\widehat { \mathbf { u } } _ { 1 } ( t ) , \ldots , \widehat { \mathbf { u } } _ { r _ { Z } } ( t )$ are eigenvectors of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ associated with its non-zero eigenvalues, we have $\mathbf { P } ^ { \mathsf { L } } ( t ) = \mathbf { P } ^ { U } ( { \bar { t } } )$ . Since $\mathbf { V } ( t )$ is of rank m and $r _ { Z } = \operatorname* { m i n } ( m , m ^ { * } )$ , we also get $\mathbf { P } ^ { \mathsf { R } } ( t ) = \mathbf { P } ^ { V } ( t )$ for $m \leq m ^ { * }$ . When $m > m ^ { * }$ , the extra eigenvalues of $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ are $2 \rho ,$ and we have shown in Lemma B.3 that the associated eigenspace is the range of the projection Π (defined in equation (44)) onto Im(R) ∩ Ker(C). This space has dimension $m - m ^ { * }$ with probability one and is independent of time. In this case $\mathbf { P } ^ { V } ( t ) = \mathbf { P } ^ { \mathsf { R } } ( t ) + \overline { { \Pi } }$ , and Π cancels out in the expression of $\mathbf { P } ^ { V } ( \bar { t } ) - \mathbf { P } _ { \infty } ^ { V }$ . This concludes the proof. □

Step 2. Reduction to spectral overlaps. From now on, we work with the projectors $\mathbf { P } ^ { U } ( t ) , \mathbf { P } ^ { V } ( t )$ , and characterize their long-time limits.

Lemma C.5. Consider the notation and assumptions of Proposition 3. With probability one

$$
{ \bf P } _ { \infty } ^ { U } = \sum _ { i = 1 } ^ { r _ { Z } } { \bf u } _ { i } { \bf u } _ { i } ^ { \top } , \qquad { \bf P } _ { \infty } ^ { V } = \sum _ { i = 1 } ^ { r _ { Z } } { \bf v } _ { i } { \bf v } _ { i } ^ { \top } + { \bf 1 } _ { m > m ^ { * } } \overline { { \Pi } } ,
$$

where Π is the orthogonal projection onto $\operatorname { I m } ( \mathbf { R } ) \cap \operatorname { K e r } ( \mathbf { C } )$

Proof. The result follows directly from equations (9a) and (9b). In the overparameterized case, $\operatorname { I m } ( { \overline { { \Pi } } } ) \subset \operatorname { K e r } ( \mathbf { C } )$ is orthogonal to Span $\left( \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m ^ { * } } \right)$ , so the two projectors can be added.

We now consider the finite-dimensional projectors

$$
\mathbf { Q } ^ { \mathsf { L } } = \sum _ { i = 1 } ^ { r _ { Z } } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } , \qquad \mathbf { Q } ^ { \mathsf { R } } = \sum _ { i = 1 } ^ { r _ { Z } } \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } .\tag{68}
$$

By Proposition 2, rank $\mathbf { U } ( t ) = r _ { Z }$ and rank $\mathbf { V } ( t ) = m$ with probability one. Moreover, $\mathbf { P } ^ { V } ( t ) \overline { { \mathbf { I } } } \overline { { \mathbf { I } } } =$ Π at all times and rank $\overline { { \mathbf { I I } } } = ( m - m ^ { * } ) ^ { + }$ . Therefore, expanding the Frobenius norm in equation (67) and using Lemma C.5

$$
\Delta ^ { \mathsf { L } } ( t ) ^ { 2 } = \frac { 2 r _ { Z } } { d } - \frac { 2 } { d } \mathrm { T r } \big ( { \mathbf { P } } ^ { U } ( t ) { \mathbf { Q } } ^ { \mathsf { L } } \big ) , \qquad \Delta ^ { \mathsf { R } } ( t ) ^ { 2 } = \frac { 2 r _ { Z } } { d } - \frac { 2 } { d } \mathrm { T r } \big ( { \mathbf { P } } ^ { V } ( t ) { \mathbf { Q } } ^ { \mathsf { R } } \big ) .\tag{69}
$$

The rest of the proof computes the limit of the projector overlaps.

Step 3. Projection of the weight matrices. We now compute a more general form of projector overlaps.

Lemma C.6. Consider the notation and assumptions of Lemma B.1. Let $\psi : \mathbb { R }  \mathbb { R }$ be continuous. Then, for $\mathbf { V } ( t )$ , and for U(t) whenever m $\leq m ^ { * }$ and $t > 0$

$$
\frac { 1 } { d } \mathrm { T r } \left( \mathbf { P } ^ { U } ( t ) \psi ( \mathbf { C C } ^ { \top } ) \right) = \frac { 1 } { d } \mathrm { T r } \left( \mathbf { R } ^ { \top } \chi _ { t } ^ { U } ( \mathbf { S } ) ^ { 2 } \psi ( \mathbf { S } ^ { 2 } ) \mathbf { R } \left( \mathbf { R } ^ { \top } \chi _ { t } ^ { U } ( \mathbf { S } ) ^ { 2 } \mathbf { R } \right) ^ { - 1 } \right) ,
$$

$$
\frac { 1 } { d } \mathrm { T r } \left( \mathbf { P } ^ { V } ( t ) \psi ( \mathbf { C } ^ { \top } \mathbf { C } ) \right) = \frac { 1 } { d } \mathrm { T r } \left( \mathbf { R } ^ { \top } \chi _ { t } ^ { V } ( \mathbf { S } ) ^ { 2 } \psi ( \mathbf { S } ^ { 2 } ) \mathbf { R } \left( \mathbf { R } ^ { \top } \chi _ { t } ^ { V } ( \mathbf { S } ) ^ { 2 } \mathbf { R } \right) ^ { - 1 } \right) ,
$$

where $\chi _ { t } ^ { U } , \chi _ { t } ^ { V }$ are applied spectrally to S.

Proof. From Lemma B.1, the range of $\mathbf { V } ( t )$ is

$$
\operatorname { I m } \mathbf { V } ( t ) = \operatorname { I m } \left( \chi _ { t } ^ { V } ( \mathbf { S } ) \mathbf { R } \right) .
$$

Similarly, when $m \leq m ^ { * }$ , we have $\operatorname { K e r } ( \chi _ { t } ^ { U } ( \mathbf { C } ) ) = \operatorname { K e r } ( \mathbf { C } )$ . By Lemma B.2, $\operatorname { I m } ( \mathbf { R } ) \cap \operatorname { K e r } ( \mathbf { C } ) = \{ \mathbf { 0 } \}$ with probability one, so $\chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R }$ has full column rank and

$$
\operatorname { I m } \mathbf { U } ( t ) = \operatorname { I m } \left( \chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } \right) .
$$

Moreover, $\chi _ { t } ^ { V } ( \mathbf { S } )$ is invertible, so $\chi _ { t } ^ { V } ( \mathbf { S } ) \mathbf { R }$ also has full column rank. For any full-column-rank matrix $\mathbf { X } ,$ , the orthogonal projector onto its range is $\mathbf { X } ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { \top }$ . Applying this identity gives

$$
\begin{array} { r l } & { { \mathbf P } ^ { U } ( t ) = \chi _ { t } ^ { U } ( { \mathbf C } ) { \mathbf R } \Big ( { \mathbf R } ^ { \top } \chi _ { t } ^ { U } ( { \mathbf C } ) ^ { \top } \chi _ { t } ^ { U } ( { \mathbf C } ) { \mathbf R } \Big ) ^ { - 1 } { \mathbf R } ^ { \top } \chi _ { t } ^ { U } ( { \mathbf C } ) ^ { \top } , } \\ & { { \mathbf P } ^ { V } ( t ) = \chi _ { t } ^ { V } ( { \mathbf S } ) { \mathbf R } \Big ( { \mathbf R } ^ { \top } \chi _ { t } ^ { V } ( { \mathbf S } ) ^ { 2 } { \mathbf R } \Big ) ^ { - 1 } { \mathbf R } ^ { \top } \chi _ { t } ^ { V } ( { \mathbf S } ) . } \end{array}
$$

This directly implies the identity for V . For U, applying Lemma C.2 leads to:

$$
\begin{array} { r } { \chi _ { t } ^ { U } ( { \mathbf { C } } ) ^ { \top } \chi _ { t } ^ { U } ( { \mathbf { C } } ) = \chi _ { t } ^ { U } ( { \mathbf { S } } ) ^ { 2 } , \quad \quad } \\ { \chi _ { t } ^ { U } ( { \mathbf { C } } ) ^ { \top } \psi ( { \mathbf { C } } { \mathbf { C } } ^ { \top } ) \chi _ { t } ^ { U } ( { \mathbf { C } } ) = \chi _ { t } ^ { U } ( { \mathbf { S } } ) ^ { 2 } \psi ( { \mathbf { S } } ^ { 2 } ) , } \end{array}
$$

which concludes the proof of the lemma.

Step 4. High-dimensional limit. We apply Theorem 22 with $\mathbf { X } = \mathbf { R }$ and $\mathbf { D } = \mathbf { S }$ . Its empirical spectral measure converges weakly to $\tilde { \mu } _ { C }$ . Moreover, $\chi _ { t } ^ { V } ( x ) \geq 1$ , while $\chi _ { t } ^ { U } ( x ) > 0$ exactly when $x > 0$ . Hence, for U in the underparameterized case,

$$
\tilde { \mu } _ { C } \big ( \{ x \colon \chi _ { t } ^ { U } ( x ) > 0 \} \big ) = \tilde { \mu } _ { C } \big ( ( 0 , \infty ) \big ) = \beta \kappa ^ { * } > \kappa .
$$

For U and $V ,$ respectively, take $a ( x ) = \chi _ { t } ^ { U } ( x ) ^ { 2 } \geq 0 \mathrm { a n d } a ( x ) = \chi _ { t } ^ { V } ( x ) ^ { 2 } \geq 1$ , with $b ( x ) = a ( x ) \psi ( x ^ { 2 } )$ For U, Assumption 1.3 guarantees that $a ( \mathbf { S } )$ has its non-zero eigenvalues uniformly bounded away from zero. Thus, Theorem 22 gives the almost sure convergence

$$
\frac { 1 } { d } \mathrm { T r } \Big ( { \bf P } ^ { U } ( t ) \psi ( { \bf C C } ^ { \top } ) \Big ) \xrightarrow [ d \to \infty ] { } \lambda _ { t } ^ { \perp } \int \psi ( x ^ { 2 } ) \frac { \chi _ { t } ^ { U } ( x ) ^ { 2 } } { 1 - \kappa + \lambda _ { t } ^ { \perp } \chi _ { t } ^ { U } ( x ) ^ { 2 } } \tilde { \mu } _ { C } ( \mathrm { d } x ) ,
$$

$$
\frac { 1 } { d } \mathrm { T r } \Big ( { \bf P } ^ { V } ( t ) \psi ( { \bf C } ^ { \top } { \bf C } ) \Big ) \xrightarrow [ d  \infty ] { } \lambda _ { t } ^ { \sf R } \int \psi ( x ^ { 2 } ) \frac { \chi _ { t } ^ { V } ( x ) ^ { 2 } } { 1 - \kappa + \lambda _ { t } ^ { \sf R } \chi _ { t } ^ { V } ( x ) ^ { 2 } } \tilde { \mu } _ { C } ( \mathrm { d } x ) ,
$$

where $\lambda _ { t } ^ { \mathsf { L } } , \lambda _ { t } ^ { \mathsf { R } }$ solve

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \lambda _ { t } ^ { \mathrm { L } } \chi _ { t } ^ { U } ( x ) ^ { 2 } } , \qquad 1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \lambda _ { t } ^ { \mathrm { R } } \chi _ { t } ^ { V } ( x ) ^ { 2 } } .
$$

Using the notation of Theorem 4, we have

$$
2 \rho \chi _ { t } ^ { U } ( x ) ^ { 2 } = r _ { t } ^ { \mathsf { L } } ( \gamma _ { x } ) , 2 \rho \chi _ { t } ^ { V } ( x ) ^ { 2 } = r _ { t } ^ { \mathsf { R } } ( \gamma _ { x } ) .
$$

Writing $\lambda _ { t } ^ { a } = 2 \rho \kappa g _ { t } ^ { \mathsf { a } }$ , the two limits become

$$
\frac { 1 } { d } \mathrm { T r } \Big ( { \bf P } ^ { U } ( t ) \psi ( { \bf C C } ^ { \top } ) \Big ) \xrightarrow [ d  \infty ] { } \kappa g _ { t } ^ { \mathrm { L } } \int \psi ( x ^ { 2 } ) \frac { r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathrm { L } } r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) ,\tag{70}
$$

$$
\frac { 1 } { d } \mathrm { T r } \Big ( { \bf P } ^ { V } ( t ) \psi ( { \bf C } ^ { \top } { \bf C } ) \Big ) \xrightarrow [ d  \infty ] { } \kappa g _ { t } ^ { \mathsf { R } } \int \psi ( x ^ { 2 } ) \frac { r _ { t } ^ { \mathsf { R } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { R } } r _ { t } ^ { \mathsf { R } } ( \gamma _ { x } ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) .\tag{71}
$$

This holds for $\kappa \in ( 0 , 1 )$ in the case of $V ,$ and for $\kappa \in ( 0 , \beta \kappa ^ { * } )$ for $U .$ .

Step 5. Overparameterized case for U. Suppose $\kappa > \beta \kappa ^ { * }$ . Under Assumption 1.3, $m > m ^ { * }$ for all large d. We return to the range identity

$$
\operatorname { I m } \mathbf { U } ( t ) = \operatorname { I m } \left( \chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } \right) .
$$

The range of $\chi _ { t } ^ { U } ( { \bf C } ) { \bf R }$ is contained in Im(C). Moreover, since $\operatorname { K e r } ( \chi _ { t } ^ { U } ( \mathbf { C } ) ) = \operatorname { K e r } ( \mathbf { C } )$ and R is injective, we have:

$$
\dim \operatorname { K e r } \bigl ( \chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } \bigr ) = \dim \big ( \operatorname { I m } ( \mathbf { R } ) \cap \operatorname { K e r } ( \mathbf { C } ) \big ) .
$$

By Lemma B.2 this space has dimension $m - m ^ { * }$ almost surely, and therefore $\mathbf { U } ( t )$ has rank $m ^ { * }$ Since its range is included in the one of C, we have Im $\mathbf { U } ( t ) =$ Im C almost surely for all $t > 0$ Therefore $\mathbf { P } ^ { \cup } ( t ) = \mathbf { P } _ { \infty } ^ { U }$ , and the left range-projector distance is identically zero for every $t > 0 \colon$

$$
\Delta ^ { \mathsf { L } } ( t ) ^ { 2 } = 0 .
$$

Step 6. Approximation of the indicator. It remains to treat the underparameterized case for $U$ and both regimes for V . Let

$$
\psi _ { * } ( y ) = { \bf 1 } \{ \sqrt { y } > x _ { * } \} ,
$$

and define the projectors

$$
\mathbf { Q } _ { * } ^ { \mathsf { L } } = \psi _ { * } ( \mathbf { C C } ^ { \top } ) , \qquad \mathbf { Q } _ { * } ^ { \mathsf { R } } = \psi _ { * } ( \mathbf { C } ^ { \top } \mathbf { C } ) .
$$

Since the nonzero eigenvalues of $\mathbf { C C } ^ { \top }$ and $\mathbf { C } ^ { \top } \mathbf { C }$ are the squared nonzero singular values of C, these finite-dimensional projectors select the directions associated with the singular values of C above $x _ { * }$ , which is the threshold obtained in the high-dimensional limit. On the other hand, the projectors $\mathbf { Q } ^ { \mathrm { L } } , \mathbf { Q } ^ { \mathrm { R } }$ defined in equation (68) always select the top $r _ { d }$ directions. For $\mathsf { a } = \mathsf { L } , \mathsf { R }$ , the two projectors are diagonal in the same singular basis, and $\mathbf { Q } ^ { \mathsf { a } } - \mathbf { Q } _ { \ast } ^ { \mathsf { a } }$ is supported only on the directions selected by one cutof but not the other. By weak convergence and the absence of an atom at $x _ { * }$ , both ranks divided by d converge to min $( \kappa , \beta \kappa ^ { * } )$ . Therefore:

$$
\frac 1 d \mathrm { r a n k } ( { \bf Q } ^ { a } - { \bf Q } _ { * } ^ { a } ) \xrightarrow [ d  \infty ] { } 0 .
$$

As a consequence:

$$
\frac { 1 } { d } \left| \operatorname { T r } \left( \mathbf { P } ^ { U } ( t ) ( \mathbf { Q } ^ { \mathsf { L } } - \mathbf { Q } _ { * } ^ { \mathsf { L } } ) \right) \right| \underset { d \to \infty } { \longrightarrow } 0 , \qquad \quad \frac { 1 } { d } \left| \operatorname { T r } \left( \mathbf { P } ^ { V } ( t ) ( \mathbf { Q } ^ { \mathsf { R } } - \mathbf { Q } _ { * } ^ { \mathsf { R } } ) \right) \right| \underset { d \to \infty } { \longrightarrow } 0 .\tag{72}
$$

We now approximate $\psi _ { * }$ by a continuous function. Fix $0 < \epsilon < x _ { * } / 2$ and set

$$
W _ { \epsilon } = \{ x \ge 0 : | x - x _ { * } | \le 2 \epsilon \} .
$$

Choose a continuous $\psi _ { \epsilon }$ such that $0 \leq \psi _ { * } \leq \psi _ { \epsilon } \leq 1$ and such that $\psi _ { \epsilon } ( y ) = \psi _ { * } ( y )$ whenever $\sqrt { y } \notin W _ { \epsilon }$ . We would like to show that equations (70), (71) extend to $\psi _ { * }$ . We then study the LHS and RHS separately.

For the LHS, we have

$$
\frac { 1 } { d } \left| \mathrm { T r } \Big ( \mathbf { P } ^ { U } ( t ) ( \psi _ { \epsilon } - \psi _ { * } ) ( \mathbf { C C } ^ { \top } ) \Big ) \right| \leq \frac { 1 } { d } \mathrm { T r } \Big ( ( \psi _ { \epsilon } - \psi _ { * } ) ( \mathbf { C C } ^ { \top } ) \Big ) \leq \mu _ { d } ^ { S } ( W _ { \epsilon } ) ,\tag{73}
$$

where $\mu _ { d } ^ { S }$ is the empirical spectral measure of $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ . The same holds for V. Moreover, for the RHS, with $\mathsf { a } = \mathsf { L } , \mathsf { R } ;$

$$
0 \leq \frac { \kappa g _ { t } ^ { \mathsf { a } } r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { a } } r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) } \leq 1 ,
$$

so that

$$
\left| \kappa g _ { t } ^ { \mathsf { a } } \int \big ( \psi _ { \epsilon } ( x ^ { 2 } ) - \psi _ { * } ( x ^ { 2 } ) \big ) \frac { r _ { t } ^ { \mathsf { a } } \big ( \gamma _ { x } \big ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { a } } r _ { t } ^ { \mathsf { a } } \big ( \gamma _ { x } \big ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) \right| \le \tilde { \mu } _ { C } ( W _ { \epsilon } ) .\tag{74}
$$

Combining these estimates with the replacement of $\mathbf { Q } ^ { \lfloor }$ by $\mathbf { Q } _ { * } ^ { \mathsf { L } }$ gives

$$
\begin{array} { r l } & { | \frac { 1 } { d } \mathrm { T r } ( { \mathbf { P } } ^ { U } ( t ) { \mathbf { Q } } ^ { \mathtt { L } } ) - \kappa g _ { t } ^ { \mathtt { L } } \int _ { x _ { * } } ^ { \infty } \frac { r _ { t } ^ { \mathtt { L } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathtt { L } } r _ { t } ^ { \mathtt { L } } ( \gamma _ { x } ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) | } \\ & { \qquad \leq \frac { 1 } { d } | \mathrm { T r } ( { \mathbf { P } } ^ { U } ( t ) ( { \mathbf { Q } } ^ { \mathtt { L } } - { \mathbf { Q } } _ { * } ^ { \mathtt { L } } ) ) | + \frac { 1 } { d } | \mathrm { T r } \big ( { \mathbf { P } } ^ { U } ( t ) ( { \mathbf { Q } } _ { * } ^ { \mathtt { L } } - \psi _ { * } ( { \mathbf { C } } { \mathbf { C } } ^ { \top } ) ) \big ) | } \\ & { \qquad + | \frac { 1 } { d } \mathrm { T r } ( { \mathbf { P } } ^ { U } ( t ) \psi _ { * } ( { \mathbf { C } } { \mathbf { C } } ^ { \top } ) ) - \kappa g _ { t } ^ { \mathtt { L } } \int \psi _ { \epsilon } ( x ^ { 2 } ) \frac { r _ { t } ^ { \mathtt { L } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathtt { L } } r _ { t } ^ { \mathtt { L } } ( \gamma _ { x } ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) | } \\ &  \qquad + | \kappa g _ { t } ^ { \mathtt { L } } \int ( \psi _ { \epsilon } ( x ^ { 2 } ) - \psi _ { * } ( x ^ { 2 } ) ) \frac { r _ { t } ^ { \mathtt { L } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathtt { L } } r _ { t } ^ { \mathtt { L } } ( \gamma _ { x } ) } \tilde { \mu } _ { C } (  \end{array}
$$

Equation (72) guarantees that the first term goes to zero as $d \to \infty$ . The second term is bounded by $\mu _ { d } ^ { S } ( W _ { \epsilon } )$ thanks to equation (73), which converges to $\tilde { \mu } _ { C } ( W _ { \epsilon } )$ . Additionally, since $\psi _ { \epsilon }$ is continuous, equation (70) shows that the third term converges to zero as $d \to \infty$ . In the end, using equation (74) to bound the fourth term, we get that:

$$
\operatorname* { l i m } _ { d \to \infty } \left| \frac { 1 } { d } \mathrm { T r } ( \mathbf { P } ^ { U } ( t ) \mathbf { Q } ^ { \mathsf { L } } ) - \kappa g _ { t } ^ { \mathsf { L } } \int _ { x _ { * } } ^ { \infty } \frac { r _ { t } ^ { \mathsf { L } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { L } } r _ { t } ^ { \mathsf { L } } ( \gamma _ { x } ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) \right| \leq 2 \tilde { \mu } _ { C } ( W _ { \epsilon } ) .
$$

Since $x _ { * }$ is not an atom of ${ \tilde { \mu } } { C } ;$ , the right-hand side vanishes as $\epsilon \downarrow 0$ . The same argument, using equation (71), applies to V. Using equation (69), we exactly obtain the formula in Theorem $6 .$

## D Small-Initialization Limit

In this section we prove the results of Section 5.

## D.1 Dynamics at Fixed Times

## D.1.1 Expressions in Finite Dimension

We shall now prove Proposition 7. We start from the explicit expression of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ at finite $\rho$ obtained in Lemma B.1. These identities read:

$$
\begin{array} { r } { \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } \Big ( \mathbf { R } ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Big ) ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { C } ) ^ { \top } , } \\ { \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t } ^ { V } ( \mathbf { S } ) \mathbf { R } \Big ( \mathbf { R } ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Big ) ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t } ^ { V } ( \mathbf { S } ) , } \\ { \mathbf { Z } ( t ) = 2 \rho \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { C } ) \mathbf { R } \Big ( \mathbf { R } ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Big ) ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t } ^ { V } ( \mathbf { S } ) , } \end{array}
$$

where $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ and $\Gamma _ { \rho } = \left( \mathbf { S } ^ { 2 } + \rho ^ { 2 } \mathbf { I } _ { d } \right) ^ { 1 / 2 }$ . Taking the limit $\rho \to 0$ , we have for all $\sigma \geq 0$ $x \ge 0 ;$

$$
\chi _ { t } ^ { U } ( \sigma ) \xrightarrow [ \rho  0 ] { } \sinh ( t \sigma ) , \qquad \chi _ { t } ^ { V } ( \sigma ) \xrightarrow [ \rho  0 ] { } \cosh ( t \sigma ) , \qquad q _ { t } ( \gamma _ { x } ) \xrightarrow [ \rho  0 ] { } 1 .
$$

Taking these limits in the three equations, together with $\Gamma _ { \rho } \xrightarrow [ \rho  0 ] { } ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ , proves the result.

## D.1.2 High-Dimensional Limit

We now start from the result of Proposition 7 and prove the following result.

Proposition 26. Consider the expressions of $\rho ^ { - 1 } \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } , \rho ^ { - 1 } \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ in the $\rho \to 0$ limit given in Proposition 7. In the high-dimensional limit, under Assumption 1, their Stieltjes transforms converge almost surely to:

$$
\begin{array} { l } { { \varsigma _ { t } ^ { U } ( w ) = - \displaystyle \frac { 1 } { w } \left( 1 - \frac { \kappa } { \tau } \right) + \frac { \kappa } { \tau } h _ { t } ( w ) , } } \\ { { \varsigma _ { t } ^ { V } ( w ) = - \displaystyle \frac { 1 } { w } \big ( 1 - \kappa \big ) + \kappa h _ { t } ( w - 2 ) , } } \end{array}
$$

for $w \in \mathbb { C } \backslash \mathbb { R } ^ { + }$ , where $h _ { t }$ is the unique holomorphic solution such that $h _ { t } ( w ) > 0 \ f o r \ w < 0$ of

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa h _ { t } ( w ) \big ( 2 \bar { r } _ { t } ^ { \mathsf { L } } ( x ) - w \big ) } ,
$$

with

$$
\bar { r } _ { t } ^ { \mathsf { L } } ( x ) = \sinh ( t x ) ^ { 2 } .\tag{75}
$$

Proof. To start the proof, note that the $\rho \to 0$ limit of $\rho ^ { - 1 } \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ has the same non-zero eigenvalues as the matrix:

$$
2 \mathbf { R } ^ { \top } \sinh ( t \mathbf { C } ) ^ { \top } \sinh ( t \mathbf { C } ) \mathbf { R } \in \mathcal { S } _ { m } ^ { + } ( \mathbb { R } ) .
$$

Using Lemma C.2, we get:

$$
\sinh ( t \mathbf { C } ) ^ { \top } \sinh ( t \mathbf { C } ) = \sinh \Big ( t \big ( \mathbf { C } ^ { \top } \mathbf { C } \big ) ^ { 1 / 2 } \Big ) ^ { 2 } .
$$

Likewise, the non-zero eigenvalues of $\rho ^ { - 1 } \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ (in the $\rho \to 0$ limit) are the same as those of:

$$
2 \mathbf R ^ { \top } \cosh \big ( t ( \mathbf C ^ { \top } \mathbf C ) ^ { 1 / 2 } \big ) ^ { 2 } \mathbf R .
$$

This reformulation allows us to treat the cases of U and V jointly. As $d \to \infty ,$ the empirical spectral distribution of $\left( \mathbf { C } ^ { \top } \mathbf { C } \right) ^ { 1 / 2 }$ converges to $\tilde { \mu } _ { C }$ (defined in Theorem 4). Recall that $\bar { r } _ { t } ^ { \mathsf { R } } ( x ) =$ $\cosh ( t x ) ^ { 2 }$ , and set

$$
\begin{array} { r } { \Phi _ { t } ^ { U } = 2 \bar { r } _ { t } ^ { \mathrm { L } } \Big ( \big ( \mathbf { C } ^ { \top } \mathbf { C } \big ) ^ { 1 / 2 } \Big ) , \qquad \Phi _ { t } ^ { V } = 2 \bar { r } _ { t } ^ { \mathrm { R } } \Big ( \big ( \mathbf { C } ^ { \top } \mathbf { C } \big ) ^ { 1 / 2 } \Big ) . } \end{array}
$$

For $w \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ , we apply Theorem 23 with $\zeta = 1 / w , q = 1$ , and $r = 2 \bar { r } _ { t } ^ { \mathsf { a } }$ , where $\mathsf { a } = \mathsf { L } , \mathsf { R }$ corresponds respectively to $\bullet = U , V . \ \operatorname { I f } \ \mathfrak { h } _ { t } ^ { \bullet } ( \zeta )$ denotes the unique holomorphic solution of equation (55) for these choices, positive for $\zeta < 0$ , then the limiting Stieltjes transform of $\mathbf { R } ^ { \top } \Phi _ { t } ^ { \bullet } \mathbf { R }$ is

$$
h _ { t } ^ { \bullet } ( w ) = - \frac { 1 } { \kappa w } \mathfrak { h } _ { t } ^ { \bullet } ( 1 / w ) ,
$$

because

$$
\frac { 1 } { m } \mathrm { T r } \Big ( \big ( \mathbf { R } ^ { \top } \Phi _ { t } ^ { \bullet } \mathbf { R } - w \mathbf { I } _ { m } \big ) ^ { - 1 } \Big ) = - \frac { 1 } { w } \frac { 1 } { m } \mathrm { T r } \left( \Big ( \mathbf { R } ^ { \top } \Big ( \mathbf { I } _ { d } - \frac { 1 } { w } \Phi _ { t } ^ { \bullet } \Big ) \mathbf { R } \Big ) ^ { - 1 } \right) ,
$$

and ${ \mathbf { R } } ^ { \top } { \mathbf { R } } = { \mathbf { I } } _ { m }$ . Rewriting the self-consistent equation of Theorem 23 in terms of $h _ { t } ^ { \bullet } ( w )$ gives

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa h _ { t } ^ { \bullet } ( w ) \bigl ( 2 \bar { r } _ { t } ^ { \mathsf { a } } ( x ) - w \bigr ) } .
$$

For the case of $U .$ , set $h _ { t } = h _ { t } ^ { U }$ , which has the characterization stated in the proposition by Theorem 23. Since $2 \bar { r } _ { t } ^ { \mathsf { R } } = 2 \bar { r } _ { t } ^ { \mathsf { L } } + 2$ , the function $w \mapsto h _ { t } ( w - 2 )$ satisfies the equation for $V$ and is positive on $( - \infty , 0 )$ . The same characterization therefore gives $h _ { t } ^ { V } ( w ) = h _ { t } ( w - 2 )$ . We recover the Stieltjes transforms of $\rho ^ { - 1 } \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\rho ^ { - 1 } \mathbf { V } ( t ) \mathbf { V } ( t ) { \bar { \mathbf { \Lambda } } } ^ { \top }$ by considering the diferent numbers of zero eigenvalues when passing from the $m \times m$ matrices $\mathbf { R } ^ { \top } \Phi _ { t } ^ { \bullet } \mathbf { R }$ to the two Gram matrices. □

## D.1.3 Small-Initialization Limit

We shall now start from the expressions of the Stieltjes transform in Theorem 4 and take the small-initialization limit $\rho \to 0$ . To do so, we let $w \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ be fixed and consider the Stieltjes transforms $\mathrm { l i m } _ { \rho \to 0 } \rho \mathsf { m } _ { t } ^ { \bullet } ( \rho w )$ , which are the Stieltjes transforms of $\rho ^ { - 1 } \mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\rho ^ { - 1 } \mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ in the small-initialization limit. As shown in Proposition $^ { 7 , }$ these are the relevant matrices to study as $\rho \to 0$ while $t = \Theta ( 1 )$

Define $h _ { t , \rho } ^ { U } ( w ) = \rho g _ { t } ( \rho w )$ and $h _ { t , \rho } ^ { V } ( w ) = \rho g _ { t } ( \rho w - 2 \rho )$ . Their self-consistent equations are

$$
\begin{array} { r l } & { 1 = \displaystyle \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa h _ { t , \rho } ^ { U } ( w ) \left( \rho ^ { - 1 } r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \right) } , } \\ & { 1 = \displaystyle \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa h _ { t , \rho } ^ { V } ( w ) \left( \rho ^ { - 1 } r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \right) } . } \end{array}
$$

We now study the $\rho \to 0$ limit. We have

$$
\frac { 1 } { 2 \rho } r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) \xrightarrow [ \rho  0 ] { } \sinh ( t x ) ^ { 2 } , \qquad \xrightarrow [ 2 \rho ] { 1 } r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) \xrightarrow [ \rho  0 ] { } \cosh ( t x ) ^ { 2 } , \qquad q _ { t } ( \gamma _ { x } ) \xrightarrow [ \rho  0 ] { } 1 ,
$$

uniformly for $x \in \mathrm { S u p p } ( \tilde { \mu } _ { C } )$

By Corollary 18, it is enough to prove the convergence for $w < 0$ . To do so, we apply Proposition 19 with parameter $\rho \to 0 , c _ { \rho } = 1$ , and:

$$
h _ { \rho } = h _ { t , \rho } ^ { \bullet } ( w ) , \qquad \ \qquad a _ { \rho } ( x ) = \kappa \bigl ( \rho ^ { - 1 } r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \bigr ) ,
$$

where $\bullet = U , V$ and $\mathsf { a } = \mathsf { L } , \mathsf { R }$ . Then, we have, from the previous uniform asymptotics:

$$
a _ { \rho } ( x ) \xrightarrow [ { \rho \to 0 } ] { } \kappa \bigl ( 2 \bar { r } _ { t } ^ { \mathsf { a } } ( x ) - w \bigr ) ,
$$

which remains strictly positive since $w < 0 .$ . It is clear that the limiting function Φ is such that $\Phi ( u ) = 1$ admits a unique solution on $( 0 , \infty )$ . Applying $\mathbf { 2 . , \Psi } ( i )$ in Proposition 19 therefore gives the convergence of $h _ { t , \rho } ^ { \bullet } ( w )$ as $\rho \to 0$ to $h _ { t } ^ { \bullet } ( w )$ , which is characterized by:

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa h _ { t } ^ { \bullet } ( w ) ( 2 \bar { r } _ { t } ^ { \mathsf { a } } ( x ) - w ) } .\tag{76}
$$

To extend the convergence to $w \in \mathbb { C } \backslash \mathbb { R } ^ { + }$ , recall that $h _ { t , \rho } ^ { \bullet }$ and $h _ { t } ^ { \bullet }$ are rescaled Stieltjes transforms, hence Lemma A.1 and Corollary 18 guarantee the extension.

Finally, using the expressions of ${ \mathfrak { m } } _ { t } ^ { U } , { \mathfrak { m } } _ { t } ^ { V }$ in Theorem 4, we obtain:

$$
\operatorname* { l i m } _ { \rho \to 0 } \rho \mathsf { m } _ { t } ^ { \bullet } ( \rho w ) = \varsigma _ { t } ^ { \bullet } ( w ) .
$$

Together with Proposition 26, this shows that we recover the same eigenvalue distributions independently of the order in which the limits $\rho \to 0$ and $d \to \infty$ (while $t \geq 0$ is fixed) are taken.

## D.2 Range Projectors

We now study the range projectors in the small-initialization limit, proving Theorem 8 and then deriving the estimate of Theorem 9.

## D.2.1 Small Initialization

We prove the small-initialization limit in Theorem 8. Recall that $\Delta _ { \rho } ^ { \mathsf { a } } ( t ) ^ { 2 }$ denotes the almost sure high-dimensional limit of the range-projector distance at fixed initialization scale $\rho ,$ for $\mathsf { a } = \mathsf { L } , \mathsf { R }$ We start from the self-consistent equation for $g _ { t } ^ { \mathsf { a } }$ in Theorem 6:

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ^ { 2 } r _ { t } ^ { 2 } ( \gamma _ { x } ) } ,
$$

where $r _ { t } ^ { \mathsf { a } }$ is defined in Theorem 4 and $\gamma _ { x } = \sqrt { x ^ { 2 } + \rho ^ { 2 } }$ . For any $x \geq 0$ and $\mathsf { a } = \mathsf { L } , \mathsf { R }$ , we have the convergence

$$
\frac { 1 } { 2 \rho } r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) \xrightarrow [ \rho  0 ] { } \bar { r } _ { t } ^ { \mathsf { a } } ( x ) .
$$

We now apply Proposition 19 directly with parameter $\rho \to 0$ and

$$
h _ { \rho } = g _ { t , \rho } ^ { \mathsf { a } } , \qquad a _ { \rho } ( x ) = \kappa r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) , \qquad c _ { \rho } = { \frac { 1 } { 2 \rho } } ,
$$

where we made explicit the dependence of $g _ { t }$ with respect to $\rho .$ Additionally, $r _ { t } ^ { \mathsf { R } }$ does not vanish on the support of $\tilde { \mu } _ { C }$ and $r _ { t } ^ { \mathsf { L } } ( \gamma _ { x } )$ is only zero at $x = 0$ . In this case the zero set of $r _ { t } ^ { \mathsf { L } } ( \gamma _ { x } )$ has mass $1 - \beta \kappa ^ { * }$ for $\tilde { \mu } _ { C }$ . In the underparameterized regime we consider for L, this mass is smaller than $1 - \kappa ,$ so that Proposition 19 can be applied. The limit function in this proposition is:

$$
\Phi ( u ) = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa u \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } ,
$$

which admits a unique solution on $( 0 , \infty )$ . As a consequence of 2. (i) in Proposition 19, we have the convergence:

$$
2 \rho g _ { t , \rho } ^ { \mathsf { a } } \xrightarrow [ \rho  0 ] { } b _ { t } ^ { \mathsf { a } } ,
$$

the unique solution of equation (24).

Additionally, 3. in Proposition 19 gives the convergence of $\Delta _ { \rho } ^ { \mathsf { a } } ( t ) ^ { 2 }$ , with $f ( x ) = \mathbf { 1 } _ { x \geq x } ,$ in the $\rho \to 0$ limit, concluding the proof of Theorem 8.

## D.2.2 Monotonicity in Time

We prove that $t \mapsto \Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 }$ is strictly decreasing, for ${ \mathsf { a } } = { \mathsf { R } }$ , and for $\mathsf { a } = \mathsf { L }$ when $\kappa < \beta \kappa ^ { * }$ . For $t > 0$ , let

$$
f _ { t } ^ { \mathsf { a } } ( x ) = \frac { 1 } { 1 - \kappa + \kappa b _ { t } ^ { \mathsf { a } } \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } .
$$

By equation (24) and Theorem 8, we have

$$
\int f _ { t } ^ { \mathsf { a } } ( x ) \tilde { \mu } _ { C } ( \mathrm { d } x ) = 1 , \qquad \quad \Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 } = 2 ( 1 - \kappa ) \int _ { ( x _ { * } , \infty ) } f _ { t } ^ { \mathsf { a } } ( x ) \tilde { \mu } _ { C } ( \mathrm { d } x ) .\tag{77}
$$

We fix $0 < t < t ^ { \prime }$ and compare $f _ { t ^ { \prime } } ^ { \mathsf { a } }$ with $f _ { t } ^ { \mathsf { a } }$ . We first show that the ratio $x \mapsto \bar { r } _ { t ^ { \prime } } ^ { \mathsf { a } } ( x ) / \bar { r } _ { t } ^ { \mathsf { a } } ( x )$ is strictly increasing on $( 0 , \infty )$ . Its logarithmic derivative is

$$
\frac { \mathrm { d } } { \mathrm { d } x } \log \frac { \bar { r } _ { t ^ { \prime } } ^ { \mathsf { a } } ( x ) } { \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } = 2 \big ( h _ { x } ^ { \mathsf { a } } ( t ^ { \prime } ) - h _ { x } ^ { \mathsf { a } } ( t ) \big ) ,
$$

where $h _ { x } ^ { \mathsf { R } } ( u ) = u \operatorname { t a n h } ( u x )$ and $h _ { x } ^ { \mathsf { L } } ( u ) = u \coth ( u x )$ . In both cases $h _ { x } ^ { \mathsf { a } }$ is strictly increasing in $u > 0$ , since

$$
\frac { \mathrm { d } } { \mathrm { d } u } \big ( u \operatorname { t a n h } ( u x ) \big ) = \operatorname { t a n h } ( u x ) + \frac { u x } { \cosh ( u x ) ^ { 2 } } > 0 ,
$$

$$
\frac { \mathrm { d } } { \mathrm { d } u } \big ( u \coth ( u x ) \big ) = \frac { \sinh ( u x ) \cosh ( u x ) - u x } { \sinh ( u x ) ^ { 2 } } > 0 .
$$

Now, for $x > 0$ , we have $f _ { t ^ { \prime } } ^ { \mathsf { a } } ( x ) \geq f _ { t } ^ { \mathsf { a } } ( x )$ if and only if

$$
\frac { \bar { r } _ { t ^ { \prime } } ^ { \mathsf { a } } ( x ) } { \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } \leq \frac { b _ { t } ^ { \mathsf { a } } } { b _ { t ^ { \prime } } ^ { \mathsf { a } } } .
$$

Since we just showed that the left side is strictly increasing with respect to $x ,$ there exists $\bar { x } \in [ 0 , \infty ]$ such that $f _ { t ^ { \prime } } ^ { \mathsf { a } } > f _ { t } ^ { \mathsf { a } }$ on $( 0 , { \bar { x } } )$ , and $f _ { t ^ { \prime } } ^ { \tt a } < f _ { t } ^ { \tt a }$ on $( { \bar { x } } , \infty )$

Recall that $x _ { * } > 0$ and that $\tilde { \mu } _ { C } ( ( x _ { * } , \infty ) ) = \operatorname* { m i n } ( \kappa , \beta \kappa ^ { * } ) > 0$ . First, if $\bar { x } \leq x _ { * }$ , then $f _ { t ^ { \prime } } ^ { \tt a } < f _ { t } ^ { \tt a }$ on $( x _ { * } , \infty )$ , and equation (77) gives $\Delta _ { 0 } ^ { \mathsf { a } } ( t ^ { \prime } ) ^ { 2 } < \Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 }$

If $\bar { x } > x _ { * }$ , the first identity in equation (77), applied at t and $t ^ { \prime } ,$ gives

$$
\int _ { ( x _ { * } , \infty ) } \big ( f _ { t ^ { \prime } } ^ { \mathsf { a } } ( x ) - f _ { t } ^ { \mathsf { a } } ( x ) \big ) \tilde { \mu } _ { C } ( \mathrm { d } x ) = - \int _ { [ 0 , x _ { * } ] } \big ( f _ { t ^ { \prime } } ^ { \mathsf { a } } ( x ) - f _ { t } ^ { \mathsf { a } } ( x ) \big ) \tilde { \mu } _ { C } ( \mathrm { d } x ) .
$$

Since $[ 0 , x _ { * } ] \subset [ 0 , \bar { x } )$ , the integrand on the right is nonnegative. It is moreover strictly positive on a set of positive mass:

• If $\kappa < \beta \kappa ^ { * }$ , on $( 0 , x _ { * } ]$ , which has mass $\tilde { \mu } _ { C } ( ( 0 , x _ { * } ] ) = \beta \kappa ^ { * } - \kappa > 0$

$\mathrm { I f } ~ \kappa \geq \beta \kappa ^ { * }$ (so that $\mathsf { a } = \mathsf { R } )$ , at $x = 0$ , which has mass $\tilde { \mu } _ { C } ( \{ 0 \} ) = 1 - \beta \kappa ^ { * } \geq 1 - \kappa > 0$ . Indeed, $\bar { r } _ { t } ^ { \mathsf { R } } ( 0 ) = 1$ , so the ratio $\bar { r } _ { t ^ { \prime } } ^ { \mathsf { R } } / \bar { r } _ { t } ^ { \mathsf { R } }$ equals 1 at $x = 0$ and is larger than 1 on $( 0 , \infty )$ . Since $\bar { x } > 0$ this gives $b _ { t } ^ { \mathsf { R } } > b _ { t ^ { \prime } } ^ { \mathsf { R } }$ , and therefore:

$$
f _ { t ^ { \prime } } ^ { \mathrm { R } } ( 0 ) = \frac { 1 } { 1 - \kappa + \kappa b _ { t ^ { \prime } } ^ { \mathrm { R } } } > \frac { 1 } { 1 - \kappa + \kappa b _ { t } ^ { \mathrm { R } } } = f _ { t } ^ { \mathrm { R } } ( 0 ) .
$$

In both cases, the integral over $( x _ { * } , \infty )$ is negative, so that $\Delta _ { 0 } ^ { \mathsf { a } } ( t ^ { \prime } ) ^ { 2 } < \Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 }$

## D.2.3 Long Times

Let us prove Theorem 9 and derive the asymptotics for $\Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 }$ as $t \to \infty$

Step 1. Useful lemmas. We start with two elementary lemmas. The first one is a standard Laplace estimate at the left edge of the spectrum.

Lemma D.1. Let $\phi \colon \mathbb { R } ^ { + }  \mathbb { R } ^ { + }$ be bounded, measurable and compactly supported, and let $\alpha , x _ { 0 } > 0$ . Assume that:

$$
\phi ( x _ { 0 } + h ) \underset { h  0 ^ { + } } { \sim } \phi _ { 0 } h ^ { \ell } ,
$$

for some $\phi _ { 0 } > 0$ and $\ell \geq 0$ . Then:

$$
\int _ { x _ { 0 } } ^ { \infty } e ^ { - \alpha x t } \phi ( x ) \mathrm { d } x \underset { t  \infty } { \sim } \frac { \phi _ { 0 } \Gamma ( \ell + 1 ) } { \alpha ^ { \ell + 1 } t ^ { \ell + 1 } } e ^ { - \alpha x _ { 0 } t } .
$$

Proof. Let us perform the substitution $u = \alpha t ( x - x _ { 0 } )$ , so that:

$$
I ( t ) : = \int _ { x _ { 0 } } ^ { \infty } e ^ { - \alpha x t } \phi ( x ) \mathrm { d } x = \frac { e ^ { - \alpha x _ { 0 } t } } { \alpha t } \int _ { 0 } ^ { \infty } e ^ { - u } \phi \left( x _ { 0 } + \frac { u } { \alpha t } \right) \mathrm { d } u .
$$

Now, consider:

$$
\psi ( h ) = \frac { \phi ( x _ { 0 } + h ) } { h ^ { \ell } } ,
$$

so that $\psi ( h ) \xrightarrow [ h \to 0 ^ { + } ] { } \phi _ { 0 }$ . Moreover, $\psi$ is bounded on $\mathbb { R } ^ { + } ;$ it is bounded on some interval $( 0 , \delta ]$ by the previous limit, and by $\| \phi \| _ { \infty } \delta ^ { - \ell }$ on $[ \delta , \infty )$ . Therefore, we have:

$$
I ( t ) = \frac { e ^ { - \alpha x _ { 0 } t } } { \alpha ^ { \ell + 1 } t ^ { \ell + 1 } } \int _ { 0 } ^ { \infty } u ^ { \ell } e ^ { - u } \psi \left( \frac { u } { \alpha t } \right) \mathrm { d } u .
$$

Now, as a consequence of the dominated convergence theorem:

$$
\int _ { 0 } ^ { \infty } u ^ { \ell } e ^ { - u } \psi \left( \frac { u } { \alpha t } \right) \mathrm { d } u \xrightarrow [ t \to \infty ] { } \phi _ { 0 } \int _ { 0 } ^ { \infty } u ^ { \ell } e ^ { - u } \mathrm { d } u = \phi _ { 0 } \Gamma ( \ell + 1 ) ,
$$

which gives the result.

Lemma D.2. Let $c _ { t } > 0$ be such that log $c _ { t } = o ( t )$ as $t \to \infty$ , and let $\phi$ be bounded, compactly supported and continuous at $x _ { * }$ . Then:

$$
\int \left( { \frac { 1 } { 1 + c _ { t } e ^ { 2 t ( x - x _ { * } ) } } } - 1 _ { x < x _ { * } } \right) \phi ( x ) \mathrm { d } x = - { \frac { \phi ( x _ { * } ) } { 2 t } } \log c _ { t } + o \left( { \frac { 1 + | \log c _ { t } | } { t } } \right) .
$$

In particular, $\mathrm { i f } \ \phi ( x _ { * } ) > 0$ and the left-hand side is $o ( t ^ { - 1 } )$ , then $c _ { t } \xrightarrow [ t  \infty ] { } 1$ . In this case:

$$
\int _ { x _ { * } } ^ { \infty } \frac { \phi ( x ) \mathrm { d } x } { 1 + c _ { t } e ^ { 2 t ( x - x _ { * } ) } } \underset { t \to \infty } { \sim } \frac { \phi ( x _ { * } ) } { 2 t } \log ( 2 ) .
$$

Proof. We introduce:

$$
x _ { t } = x _ { * } - \frac { \log c _ { t } } { 2 t } ,
$$

so that $c _ { t } e ^ { 2 t ( x - x _ { * } ) } = e ^ { 2 t ( x - x _ { t } ) }$ and $x _ { t } \xrightarrow [ { t \to \infty } ] { } x _ { * }$ . We decompose:

$$
\begin{array} { r } { \displaystyle \int \left( \frac { 1 } { 1 + c _ { t } e ^ { 2 t ( x - x _ { * } ) } } - \mathbf { 1 } _ { x < x _ { * } } \right) \phi ( x ) \mathrm { d } x = \int \left( \frac { 1 } { 1 + e ^ { 2 t ( x - x _ { t } ) } } - \mathbf { 1 } _ { x < x _ { t } } \right) \phi ( x ) \mathrm { d } x } \\ { + \int \left( \mathbf { 1 } _ { x < x _ { t } } - \mathbf { 1 } _ { x < x _ { * } } \right) \phi ( x ) \mathrm { d } x . } \end{array}
$$

The first term is $o ( t ^ { - 1 } )$ . Indeed, with the substitution $x = x _ { t } + \xi t ^ { - 1 }$ , it becomes:

$$
\frac { 1 } { t } \int \left( \frac { 1 } { 1 + e ^ { 2 \xi } } - { \bf 1 } _ { \xi < 0 } \right) \phi \left( x _ { t } + \frac { \xi } { t } \right) \mathrm { d } \xi .
$$

The integral converges to zero by dominated convergence. For the second term, we have:

$$
\int \left( \mathbf { 1 } _ { x < x _ { t } } - \mathbf { 1 } _ { x < x _ { * } } \right) \phi ( x ) \mathrm { d } x = \int _ { x _ { * } } ^ { x _ { t } } \phi ( x ) \mathrm { d } x = ( x _ { t } - x _ { * } ) \phi ( x _ { * } ) + o ( | x _ { t } - x _ { * } | ) .
$$

Since $x _ { t } - x _ { * } = - ( 2 t ) ^ { - 1 } \log c _ { t }$ , this proves the first asymptotic. Additionally, if $\phi ( x _ { * } ) > 0$ and the left-hand side is $o ( t ^ { - 1 } )$ , then we must have log $c _ { t } = o ( 1 )$ , which implies $c _ { t } \xrightarrow [ t  \infty ] { } 1$

For the second asymptotic, we use the same technique. We get, after splitting the integral and introducing $x = x _ { t } + \xi t ^ { - 1 }$ , with $x _ { t }$ defined as above:

$$
\int _ { x _ { + } } ^ { \infty } \frac { \phi ( x ) \mathrm { d } x } { 1 + c _ { t } e ^ { 2 t ( x - x _ { + } ) } } = \frac { 1 } { t } \int _ { 0 } ^ { \infty } \phi \left( x _ { t } + \frac { \xi } { t } \right) \frac { \mathrm { d } \xi } { 1 + e ^ { 2 \xi } } + \frac { 1 } { t } \int \phi \left( x _ { t } + \frac { \xi } { t } \right) \frac { \mathrm { d } \xi } { 1 + e ^ { 2 \xi } } \left( \mathbf { 1 } _ { \xi > \frac { \log c _ { t } } { 2 } } - \mathbf { 1 } _ { \xi > 0 } \right) .
$$

The second term is $o ( t ^ { - 1 } )$ because log $c _ { t }  0$ and $\phi$ is bounded near $x _ { * }$ . For the first term, dominated convergence gives:

$$
\int _ { 0 } ^ { \infty } \phi \left( x _ { t } + \frac { \xi } { t } \right) \frac { \mathrm { d } \xi } { 1 + e ^ { 2 \xi } } \xrightarrow [ t \to \infty ] { \mathrm { d } \xi } \phi ( x _ { * } ) \int _ { 0 } ^ { \infty } \frac { \mathrm { d } \xi } { 1 + e ^ { 2 \xi } } = \frac { \phi ( x _ { * } ) } { 2 } \log ( 2 ) .
$$

Step 2. Self-consistent functions. We start by studying $b _ { t } ^ { \mathsf { a } }$ , for $\mathsf { a } = \mathsf { L } , \mathsf { R }$ . It is the solution of the self-consistent equation

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa b _ { t } ^ { 2 } \bar { r } _ { t } ^ { 2 } ( x ) } .\tag{78}
$$

We first determine its long-time behavior.

Lemma D.3. Consider the setting of Theorem 9, and $b _ { t } ^ { \mathsf { a } }$ solving equation (78). Then, as $t \to \infty ;$ • If $\kappa > \beta \kappa ^ { * }$ :

$$
b _ { t } ^ { \mathsf { R } } \xrightarrow [ t \to \infty ] { } 1 - \frac { \beta \kappa ^ { * } } { \kappa } .
$$

• If $\kappa < \beta \kappa ^ { * }$ , both $b _ { t } ^ { \mathsf { L } } , b _ { t } ^ { \mathsf { R } }$ converge to zero exponentially fast. Moreover, for all $y > 0$

$$
b _ { t } ^ { a } e ^ { 2 t y } \xrightarrow [ t \to \infty ] { } \left\{ \begin{array} { l l } { 0 , } & { y < x _ { * } , } \\ { + \infty , } & { y > x _ { * } , } \end{array} \right.
$$

where $x _ { * }$ is characterized by $\kappa = \beta \kappa ^ { * } \int _ { x _ { * } } ^ { \infty } \phi _ { C } ( x ) \mathrm { d } x$

We only need the asymptotic behavior of $b _ { t } ^ { \mathsf { R } }$ when $\kappa > \beta \kappa ^ { * }$ , since the left range-projector distance is then zero. The threshold $x _ { * }$ is unique under the assumptions of Theorem 9, since $\phi _ { C } ( x _ { * } ) > 0$

Proof. We start with the case $\kappa > \beta \kappa ^ { * }$ , and ${ \mathsf { a } } = { \mathsf { R } } .$ . We apply Proposition 19 directly with parameter $t \to \infty$ and

$$
h _ { t } = b _ { t } ^ { \mathsf { R } } , \qquad a _ { t } ( x ) = \kappa \bar { r } _ { t } ^ { \mathsf { R } } ( x ) ,
$$

and $c _ { t } = 1$ . As we have $\bar { r } _ { t } ^ { \mathsf { R } } ( x ) \xrightarrow [ t \to \infty ] { }$ +∞ for $x > 0$ and $\bar { r } _ { t } ^ { \mathsf { R } } ( 0 ) = 1$ , the limiting function in Proposition 19 is given by:

$$
\Phi ( u ) = \frac { 1 - \beta \kappa ^ { * } } { 1 - \kappa + \kappa u } .
$$

Therefore, $b _ { t } ^ { \mathsf { R } }$ converges to the desired limit in the overparameterized case by 2. (i) in Proposition 19.

Regarding the underparameterized case, consider $y > 0$ and again take

$$
h _ { t } = b _ { t } ^ { \mathsf { a } } , \qquad a _ { t } ( x ) = \kappa \bar { r } _ { t } ^ { \mathsf { a } } ( x ) ,
$$

with $c _ { t } = e ^ { - 2 t y }$ . We have:

$$
c _ { t } a _ { t } ( x ) \xrightarrow [ { t \to \infty } ] { } \left\{ { 0 , \atop + \infty } , \begin{array} { l } { x < y , } \\ { \qquad x > y . } \end{array}  \right.
$$

Hence the limit function in Proposition 19 is given by:

$$
\Phi _ { y } ( u ) = \frac { 1 - T _ { C } ( y ) } { 1 - \kappa } ,
$$

for every $u > 0$ . Since $x _ { * }$ is the only point satisfying $T _ { C } ( x _ { * } ) = \kappa ,$ , applying 2. (ii) and (iii) in Proposition 19 leads to the desired result for the underparameterized case. □

We now refine the asymptotic behavior of $b _ { t } ^ { \mathsf { a } }$ in the underparameterized case.

Lemma D.4. In the setting of Lemma D.3, with $\kappa < \beta \kappa ^ { * }$ , we have:

$$
b _ { t } ^ { \mathsf { a } } \underset { t  \infty } { \sim } \frac { 4 ( 1 - \kappa ) } { \kappa } e ^ { - 2 t x _ { * } } .
$$

Proof. We consider:

$$
c _ { t } ^ { \mathsf { a } } = { \frac { \kappa } { 4 ( 1 - \kappa ) } } b _ { t } ^ { \mathsf { a } } e ^ { 2 t x _ { * } } ,\tag{79}
$$

and fix $y \in ( 0 , x _ { * } )$ . By Lemma D.3, $b _ { t } ^ { \mathsf { a } } = o ( e ^ { - 2 y t } )$ . Hence the contribution of the atom at zero to equation (78) is

$$
\frac { 1 - \beta \kappa ^ { * } } { 1 - \kappa + \kappa b _ { t } ^ { \mathsf { a } } \bar { r } _ { t } ^ { \mathsf { a } } ( 0 ) } = \frac { 1 - \beta \kappa ^ { * } } { 1 - \kappa } + o ( e ^ { - 2 y t } ) .
$$

We also use the identity:

$$
1 = \frac { 1 - \beta \kappa ^ { * } } { 1 - \kappa } + \frac { \beta \kappa ^ { * } } { 1 - \kappa } \int _ { 0 } ^ { x _ { * } } \phi _ { C } ( x ) \mathrm { d } x ,
$$

which follows from the definition of $x _ { * }$ . Subtracting this identity from equation (78) gives:

$$
\int \left( \frac { 1 } { 1 + 4 c _ { t } ^ { \mathrm { a } } e ^ { - 2 t x _ { * } } \bar { r } _ { t } ^ { \mathrm { a } } ( x ) } - \mathbf { 1 } _ { x < x _ { * } } \right) \phi _ { C } ( x ) \mathrm { d } x = o ( e ^ { - 2 y t } ) .
$$

Now, we have:

$$
\left| \frac { 1 } { 1 + 4 c _ { t } ^ { \mathsf { a } } e ^ { - 2 t x _ { * } } \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } - \frac { 1 } { 1 + c _ { t } ^ { \mathsf { a } } e ^ { 2 t ( x - x _ { * } ) } } \right| \leq c _ { t } ^ { \mathsf { a } } e ^ { - 2 t x _ { * } } \left| 4 \bar { r } _ { t } ^ { \mathsf { a } } ( x ) - e ^ { 2 t x } \right| .
$$

Now, $4 \bar { r } _ { t } ^ { \mathsf { a } } ( x ) = \left( e ^ { t x } \mp e ^ { - t x } \right) ^ { 2 }$ , depending on $\mathsf { a } = \mathsf { L }$ , R. Hence $4 \bar { r } _ { t } ^ { \mathsf { a } } ( x ) - e ^ { 2 t x }$ remains bounded, and the right-hand side is $o ( e ^ { - 2 y t } )$ . Therefore,

$$
\int \left( \frac { 1 } { 1 + c _ { t } ^ { \mathsf { a } } e ^ { 2 t ( x - x _ { * } ) } } - \mathbf { 1 } _ { x < x _ { * } } \right) \phi _ { C } ( x ) \mathrm { d } x = o ( e ^ { - 2 y t } ) .
$$

Lemma D.3 guarantees that $\log ( c _ { t } ^ { \mathsf { a } } ) = o ( t )$ , hence by Lemma D.2, $c _ { t } ^ { \mathsf { a } } \xrightarrow [ t \to \infty ] { } 1$ , which concludes the proof. □

Step 3. Asymptotics for the distance. We now compute the asymptotic behavior of $\Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 }$ Let us start with the overparameterized case $\kappa > \beta \kappa ^ { * }$ . Since:

$$
\bar { r } _ { t } ^ { \mathsf { R } } ( x ) \underset { t \to \infty } { \sim } \frac { 1 } { 4 } e ^ { 2 t x }
$$

uniformly on the support of $\phi _ { C }$ , using the limit of $b _ { t } ^ { \mathsf { R } }$ in Lemma D.3 gives

$$
\Delta _ { 0 } ^ { \mathsf { R } } ( t ) ^ { 2 } \underset { t  \infty } { \sim } \frac { 8 \beta \kappa ^ { * } ( 1 - \kappa ) } { \kappa - \beta \kappa ^ { * } } \int e ^ { - 2 x t } \phi _ { C } ( x ) \mathrm { d } x ,
$$

and the asymptotic equivalent follows from Lemma D.1 with $\alpha = 2$

In the underparameterized case, Theorem 8 and equation (79) give

$$
\Delta _ { 0 } ^ { \mathsf { a } } ( t ) ^ { 2 } = 2 \beta \kappa ^ { * } \int _ { x _ { * } } ^ { \infty } \frac { \phi _ { C } ( x ) \mathrm { d } x } { 1 + 4 c _ { t } ^ { \mathsf { a } } e ^ { - 2 t x _ { * } } \bar { r } _ { t } ^ { \mathsf { a } } ( x ) } .
$$

We can replace 4e $- 2 t x _ { * } \bar { r } _ { t } ^ { \mathsf { a } } ( x )$ by $e ^ { 2 t ( x - x _ { * } ) }$ up to an exponentially vanishing correction. Since $c _ { t } ^ { \mathsf { a } }$ converges to 1, Lemma D.2 gives

$$
\Delta _ { 0 } ^ { a } ( t ) ^ { 2 } \underset { t  \infty } { \sim } \frac { \beta \kappa ^ { * } \phi _ { C } ( x _ { * } ) } { t } \log ( 2 ) ,
$$

which is the desired result.

## D.3 Top Singular Subspaces

In this section we prove Theorem 10. Unlike the range-projector distances, we do not compute an exact expression of the top singular vector distances in the high-dimensional limit. To achieve this, one could use the two-point Stieltjes transforms (see Claim 25 for an expression of their high-dimensional limit), which encode the necessary spectral overlaps. However, this route is technically involved and we do not pursue it here. Instead, we obtain direct and non-sharp convergence rates.

We start by giving some prior lemmas, then treat the finite-dimensional case, and finally the high-dimensional limit. We study the distance, for ${ \mathsf { a } } = { \mathsf { R } } .$ , L:

$$
\Delta _ { k , \rho } ^ { a } ( t ) ^ { 2 } = \frac { 1 } { d } \big \| \mathbf { P } _ { k , \rho } ^ { a } ( t ) - \mathbf { P } _ { k } ^ { C , { \mathsf { a } } } \big \| _ { F } ^ { 2 } ,
$$

and we recall that $\mathbf { P } _ { k , \rho } ^ { \ L } ( t ) , \mathbf { P } _ { k , \rho } ^ { \ R } ( t )$ project onto the top-k left and right singular vectors of the prediction $\mathbf { Z } ( t )$ , and likewise for $\mathbf { P } _ { k } ^ { C , \mathsf { L } } , \mathbf { P } _ { k } ^ { C , \mathsf { R } }$ with C.

Relation with subspace iteration. In the limit $\rho \to 0$ , the rescaled Gram matrices from Proposition 7 have the same non-zero eigenvalues as $2 \mathbf { M } ^ { \mathsf { a } } ( t )$ , with $\mathbf { M } ^ { \mathsf { a } } ( t ) = \mathbf { Y } _ { t } ^ { \mathsf { a } } ( \mathbf { Y } _ { t } ^ { \mathsf { a } } ) ^ { \top }$ , where

$$
\mathbf { Y } _ { t } ^ { \mathsf { L } } = \sinh ( t \mathbf { S } ) \mathbf { R } , \qquad \mathbf { Y } _ { t } ^ { \mathsf { R } } = \cosh ( t \mathbf { S } ) \mathbf { R } ,
$$

and $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ . This is the continuous version of a filtered subspace iteration (Stewart, 1976; Halko et al., 2011; Saibaba, 2019). Writing $f _ { t } ^ { \mathsf { L } } ( x ) = \sinh ( t x )$ and $f _ { t } ^ { \mathsf { R } } ( x ) = \cosh ( t x )$ , the ratio between the lower and upper components at the cutof k is:

$$
\frac { f _ { t } ^ { \mathsf { a } } ( \sigma _ { k + 1 } ) } { f _ { t } ^ { \mathsf { a } } ( \sigma _ { k } ) } = O \Bigl ( e ^ { - t ( \sigma _ { k } - \sigma _ { k + 1 } ) } \Bigr ) .
$$

This is the main idea behind the estimate given in equation (25). Our new approach concerns the proportional-rank limit $k \sim \alpha d .$ . In this case the bounds derive from the same idea as the finite-dimensional case, and the non-exponential asymptotic in equation (27) naturally arises from the fact that the gap $\sigma _ { k } - \sigma _ { k + 1 }$ vanishes as $d \to \infty$

Step 1. Gram matrices. We start to express the distances $\Delta _ { k , \rho } ^ { \mathsf { a } } ( t ) ^ { 2 }$ in terms of projections obtained from PSD matrices, related to those in Lemma B.1.

Lemma D.5. Consider the assumptions of Theorem 10. For $t > 0$ and whenever $k \leq \operatorname* { m i n } ( m , m ^ { * } )$

$$
\Delta _ { k , \rho } ^ { a } ( t ) ^ { 2 } = \frac { 1 } { d } \big \| \mathbf { Q } _ { k , \rho } ^ { a } ( t ) - \mathbf { P } _ { k } ^ { C , \mathsf { R } } \big \| _ { F } ^ { 2 } ,
$$

where $\mathbf { Q } _ { k , \rho } ^ { \mathsf { a } } ( t )$ is the projection onto the top-k eigenvectors of

$$
\mathbf { M } _ { \rho } ^ { \mathsf { a } } ( t ) = r _ { t } ^ { \mathsf { a } } ( \Gamma _ { \rho } ) ^ { 1 / 2 } \mathbf { R } \Big ( \mathbf { R } ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf { R } \Big ) ^ { - 1 } \mathbf { R } ^ { \top } r _ { t } ^ { \mathsf { a } } ( \Gamma _ { \rho } ) ^ { 1 / 2 } ,\tag{80}
$$

where $r _ { t } ^ { \mathsf { L } } , r _ { t } ^ { \mathsf { R } } , q _ { t }$ are defined in Theorem 4.

Proof. As a consequence of Proposition 2, we have the equalities:

$$
\begin{array} { r } { \mathbf { P } _ { k , \rho } ^ { \mathrm { L } } ( t ) = \mathbf { P } _ { k , \rho } ^ { U } ( t ) , \qquad \mathbf { P } _ { k , \rho } ^ { \mathrm { R } } ( t ) = \mathbf { P } _ { k , \rho } ^ { V } ( t ) , } \end{array}
$$

where $\mathbf { P } _ { k , \rho } ^ { U } ( t ) , \mathbf { P } _ { k , \rho } ^ { V } ( t )$ are the projectors onto the top-k eigenvectors of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top }$ This equality is guaranteed whenever $k \leq \operatorname* { m i n } ( m , m ^ { * } )$ . We now use the notation of Lemma B.1 and the identities:

$$
2 \rho \chi _ { t } ^ { U } ( x ) ^ { 2 } = r _ { t } ^ { \downarrow } ( \gamma _ { \rho } ( x ) ) , 2 \rho \chi _ { t } ^ { V } ( x ) ^ { 2 } = r _ { t } ^ { \mathsf { R } } ( \gamma _ { \rho } ( x ) ) ,
$$

where $\gamma _ { \rho } ( x ) = \sqrt { x ^ { 2 } + \rho ^ { 2 } }$ . Hence Lemma B.1 guarantees that $\mathbf { V } ( t ) \mathbf { V } ( t ) ^ { \top } = \mathbf { M } _ { \rho } ^ { \mathsf { R } } ( t )$ . This proves the claim for R.

For the left projection, we recall the singular value decomposition $\begin{array} { r } { \mathbf { C } = \sum _ { i = 1 } ^ { m ^ { * } } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ . We have:

$$
\mathbf { O } r _ { t } ^ { \mathsf { L } } ( \Gamma _ { \rho } ) ^ { 1 / 2 } = \sqrt { 2 \rho } \chi _ { t } ^ { U } ( \mathbf { C } ) , \qquad \mathbf { O } = \sum _ { i = 1 } ^ { m ^ { * } } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } ,
$$

and therefore $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top } = \mathbf { O } \mathbf { M } _ { \rho } ^ { \sf L } ( t ) \mathbf { O } ^ { \top }$ . Since $r _ { t } ^ { \mathrm { L } } ( \mathbf { T } _ { \rho } )$ vanishes on $\mathrm { K e r } ( \mathbf { C } )$ , the image of $\mathbf { M } _ { \rho } ^ { \mathrm { L } } ( t )$ is included in $\operatorname { S p a n } ( \mathbf { v } _ { 1 } , \dots , \mathbf { v } _ { m ^ { * } } )$ , where O is an isometry. Hence there is an exact correspondence between the top-k eigenvectors of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$ and those of $\mathbf { M } _ { \rho } ^ { \mathrm { L } } ( t )$ through O. Therefore,

$$
\mathbf { P } _ { k , \rho } ^ { U } ( t ) = \mathbf { O } \mathbf { Q } _ { k , \rho } ^ { \perp } ( t ) \mathbf { O } ^ { \top } , \qquad \mathbf { P } _ { k } ^ { C , \mathrm { L } } = \mathbf { O } \mathbf { P } _ { k } ^ { C , \mathrm { R } } \mathbf { O } ^ { \top } ,
$$

and

$$
\big \| \mathbf { P } _ { k , \rho } ^ { U } ( t ) - \mathbf { P } _ { k } ^ { C , \mathtt { L } } \big \| _ { F } ^ { 2 } = \Big \| \mathbf { O } \mathbf { Q } _ { k , \rho } ^ { \mathtt { L } } ( t ) \mathbf { O } ^ { \top } - \mathbf { O } \mathbf { P } _ { k } ^ { C , \mathtt { R } } \mathbf { O } ^ { \top } \Big \| _ { F } ^ { 2 } .
$$

Since $\mathbf { O } ^ { \top } \mathbf { O }$ is the orthogonal projection onto Span $\left( \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m ^ { * } } \right)$ and Im $\left( \mathbf { Q } _ { k , \rho } ^ { \mathsf { L } } ( t ) \right)$ , Im $( \mathbf { P } _ { k } ^ { C , \mathsf { R } } )$ are included in this space, the last quantity is equal to $\big \| \mathbf { Q } _ { k , \rho } ^ { \mathsf { L } } ( t ) - \mathbf { P } _ { k } ^ { C , \mathsf { R } } \big \| _ { F } ^ { 2 }$ , which concludes the proof. □

The proof of this lemma shows that $\mathbf { Q } _ { k , \rho } ^ { \mathsf { R } } ( t ) = \mathbf { P } _ { k , \rho } ^ { \mathsf { R } } ( t )$ , and that ${ \bf Q } _ { k , \rho } ^ { \lfloor } ( t )$ is obtained by transporting $\mathbf { P } _ { k , \rho } ^ { \ L } ( t )$ into the right singular basis of C. Hence both $\mathbf { Q } _ { k , \rho } ^ { \mathsf { L } } ( t ) , \dot { \mathbf { Q } } _ { k , \rho } ^ { \mathsf { R } } ( t )$ are compared with $\mathbf { P } _ { k } ^ { C , \mathsf { R } }$

Step 2. A projector distance bound. The next lemma gives the main bound used in the following. To do so, we define the quantity:

$$
\nu _ { \mathbf { P } } ( \mathbf { A } ) = \operatorname* { i n f } _ { \mathbf { u } \in \operatorname { I m } \mathbf { P } , \| \mathbf { u } \| = 1 } \| \mathbf { A u } \| ^ { 2 } .\tag{81}
$$

$\nu _ { \mathbf { P } } ( \mathbf { A } )$ is the smallest eigenvalue of $\mathbf { P A } ^ { \top } \mathbf { A P }$ restricted to the subspace $\operatorname { I m } ( \mathbf { P } )$ . It satisfies $\mathbf { u } ^ { \top } \mathbf { A } ^ { \top } \mathbf { A } \mathbf { u } \geq \nu _ { \mathbf { P } } ( \mathbf { A } ) \| \mathbf { u } \| ^ { 2 }$ for all $\mathbf { u } \in \operatorname { I m } ( \mathbf { P } )$

Lemma D.6. Let D, $\mathbf { H } \in { \mathcal { S } } _ { d } ^ { + } ( \mathbb { R } )$ , with $\mathbf { H } \succeq \mathbf { I } _ { d } .$ , let $\mathbf { R } \in \mathrm { S t } ( d , m )$ , and define

$$
\mathbf { M } = \mathbf { D } ^ { 1 / 2 } \mathbf { R } ( \mathbf { R } ^ { \top } \mathbf { H } \mathbf { R } ) ^ { - 1 } \mathbf { R } ^ { \top } \mathbf { D } ^ { 1 / 2 } .
$$

Let P be an orthogonal projector of rank $\ell \leq m$ that commutes with D, and $\mathbf { Q }$ be the projection onto the top-ℓ eigenvectors of M. Then:

$$
\left\| \mathbf { P } - \mathbf { Q } \right\| _ { F } ^ { 2 } \leq \frac { 2 \| \mathbf { H } \| _ { \mathrm { o p } } } { \nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) \nu _ { \mathbf { P } } ( \mathbf { D } ^ { 1 / 2 } ) } \mathrm { T r } \big ( ( \mathbf { I } _ { d } - \mathbf { P } ) \mathbf { D } \big ) ,\tag{82}
$$

whenever $\nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) , \nu _ { \mathbf { P } } ( \mathbf { D } ^ { 1 / 2 } )$ are non-zero.

Proof. Since P, Q are both projectors, it is clear that $\left\| \mathbf { P } - \mathbf { Q } \right\| _ { F } ^ { 2 } = 2 \mathrm { T r } \big ( ( \mathbf { I } _ { d } - \mathbf { P } ) \mathbf { Q } \big )$ . Now, let us denote by $\lambda _ { \ell } ( \mathbf { M } )$ the $\ell ^ { \mathrm { t h } }$ largest eigenvalue of M. On the image of Q, M has eigenvalues at least $\lambda _ { \ell } ( \mathbf { M } )$ , hence $\mathbf { M } \succeq \lambda _ { \ell } ( \mathbf { M } ) \mathbf { Q }$ . Additionally, ${ \bf M } \preceq { \bf D }$ since $\mathbf { R } ^ { \top } \mathbf { H } \mathbf { R } \succeq \mathbf { I } _ { m }$ and ${ \mathbf { R R } } ^ { \top } \preceq { \mathbf { I } } _ { d }$ . Putting these facts together, and using that ${ \mathbf I } _ { d } - { \mathbf P }$ is PSD, we obtain the bound:

$$
\left\| \mathbf { P } - \mathbf { Q } \right\| _ { F } ^ { 2 } \leq \frac { 2 } { \lambda _ { \ell } ( \mathbf { M } ) } \mathrm { T r } \big ( ( \mathbf { I } _ { d } - \mathbf { P } ) \mathbf { D } \big ) .\tag{83}
$$

It remains to lower-bound $\lambda _ { \ell } ( \mathbf { M } )$ . To do so, we use the Courant–Fischer min–max principle for eigenvalues of a symmetric matrix:

$$
\lambda _ { \ell } ( \mathbf { M } ) = \operatorname* { m a x } _ { \substack { \mathrm { d i m } \varepsilon = \ell \mathbf { y } \in \mathcal { E } , \| \mathbf { y } \| = 1 } } \mathbf { y } ^ { \top } \mathbf { M } \mathbf { y } \geq \operatorname* { m i n } _ { \mathbf { y } \in \mathrm { I m } ( \mathbf { P } ) , \| \mathbf { y } \| = 1 } \mathbf { y } ^ { \top } \mathbf { M } \mathbf { y } ,\tag{84}
$$

where we chose $\mathcal { E } = \mathrm { I m } ( \mathbf { P } )$ in the inequality. For $\mathbf { y } \in \operatorname { I m } ( \mathbf { P } )$ , let $\mathbf z = \mathbf D ^ { 1 / 2 } \mathbf y$ . Since P commutes with D, we still have $\mathbf { z } \in \operatorname { I m } ( \mathbf { P } )$ . Moreover, since ${ \mathbf { R } } ^ { \top } { \mathbf { R } } = { \mathbf { I } } _ { m }$

$$
\mathbf { R } ^ { \top } \mathbf { H } \mathbf { R } \preceq \| \mathbf { H } \| _ { \mathrm { o p } } \mathbf { I } _ { m } ,
$$

and therefore

$$
\mathbf { y } ^ { \top } \mathbf { M } \mathbf { y } \geq \frac { 1 } { \| \mathbf { H } \| _ { \mathrm { o p } } } \mathbf { z } ^ { \top } \mathbf { R } \mathbf { R } ^ { \top } \mathbf { z } \geq \frac { 1 } { \| \mathbf { H } \| _ { \mathrm { o p } } } \nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) \| \mathbf { z } \| ^ { 2 } \geq \frac { 1 } { \| \mathbf { H } \| _ { \mathrm { o p } } } \nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) \nu _ { \mathbf { P } } ( \mathbf { D } ^ { 1 / 2 } ) ,
$$

where we used the definition of $\nu _ { \mathbf { P } }$ in equation (81). Combining this lower bound on $\lambda _ { \ell } ( \mathbf { M } )$ with equation (83) proves the desired bound. □

We now apply this lemma to $\mathbf { M } _ { \rho } ^ { \mathsf { a } } ( t )$ defined in equation (80). More precisely, we choose:

$$
{ \bf H } = q _ { t } ( { \Gamma } _ { \rho } ) , \qquad { \bf D } = r _ { t } ^ { \mathrm { a } } ( { \Gamma } _ { \rho } ) , \qquad { \bf P } = { \bf P } _ { k } ^ { C , \mathrm { R } } .
$$

Recall that $\begin{array} { r } { \mathbf { \Gamma } \Gamma _ { \rho } = \left( \mathbf { C } ^ { \top } \mathbf { C } + \rho ^ { 2 } \mathbf { I } _ { d } \right) ^ { 1 / 2 } } \end{array}$ . Since $q _ { t } ( \gamma ) \geq 1$ for $\gamma \geq \rho ,$ we have indeed $\mathbf { H } \succeq \mathbf { I } _ { d }$ Additionally, $\Gamma _ { \rho }$ commutes with P. Therefore:

$$
\nu _ { \mathbf { P } } ( \mathbf { D } ^ { 1 / 2 } ) \geq \operatorname* { i n f } _ { x \in [ \sigma _ { k } , L ] } r _ { t } ^ { a } ( \gamma _ { \rho } ( x ) ) , \qquad \gamma _ { \rho } ( x ) = \sqrt { x ^ { 2 } + \rho ^ { 2 } } ,
$$

where we used that the singular values of C are uniformly bounded by $L$ by Assumption 1.3. In finite dimension, one can take $L = \sigma _ { 1 }$ . Additionally, P and D commute, so that:

$$
\mathrm { T r } \big ( ( \mathbf { I } _ { d } - \mathbf { P } ) \mathbf { D } \big ) = \sum _ { i = k + 1 } ^ { d } r _ { t } ^ { \mathsf { a } } ( \gamma _ { i } ) , \qquad \gamma _ { i } = \sqrt { \sigma _ { i } ^ { 2 } + \rho ^ { 2 } } .
$$

Finally, due to the expression of H:

$$
\| \mathbf { H } \| _ { \mathrm { o p } } \leq \operatorname* { s u p } _ { x \in [ 0 , L ] } q _ { t } ( \gamma _ { \rho } ( x ) ) .
$$

Therefore, combining Lemma D.5 and Lemma D.6, we get:

$$
\Delta _ { k , \rho } ^ { \mathsf { a } } ( t ) ^ { 2 } \leq \frac { 2 } { \nu _ { k } d } \left( \operatorname* { s u p } _ { x \in [ 0 , L ] } q _ { t } ( \gamma _ { \rho } ( x ) ) \right) \frac { \sum _ { i = k + 1 } ^ { d } r _ { t } ^ { \mathsf { a } } ( \gamma _ { i } ) } { \operatorname* { i n f } _ { x \in [ \sigma _ { k } , L ] } r _ { t } ^ { \mathsf { a } } ( \gamma _ { \rho } ( x ) ) } .\tag{85}
$$

We denoted $\nu _ { k } = \nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } )$ . This bound forms the basis of the finite and high-dimensional computations. As $\rho \to 0$ , we will show that the last ratio leads to the desired convergence rates: the numerator only concerns the singular values of C at most $\sigma _ { k + 1 }$ while the denominator only involves those greater than or equal to $\sigma _ { k }$

Step 3. Lower-bounding random projections. Before deriving the convergence rates, we show that $\nu _ { k }$ remains bounded away from zero. We provide a slightly more general version that we will need in the high-dimensional limit.

Lemma D.7. Let R be uniformly distributed on $\mathrm { S t } ( d , m )$ , and let $\mathbf { P }$ be an orthogonal projector of rank $\ell \leq m$ , independent of R. Then, with probability one

$$
\nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) > 0 .
$$

Moreover, if as $d \to \infty$ , we have $m \sim$ κd and $\ell \sim \alpha d ,$ where $0 < \alpha < \kappa < 1$ , then there exists a constant $C _ { \alpha , \kappa } > 0$ only depending on α, κ such that, with probability one, $\nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) > C _ { \alpha , \kappa }$ for d large enough.

Proof. As in the proof of Theorem 22, we write $\mathbf { R } = \mathbf { G } ( \mathbf { G } ^ { \top } \mathbf { G } ) ^ { - 1 / 2 }$ , where G has i.i.d. $\mathcal { N } ( 0 , 1 / m )$ entries and is independent of P. We can then choose a basis in which

$$
\mathbf { P } = { \binom { \mathbf { I } _ { \ell } } { \mathbf { 0 } } } \ \mathbf { 0 } ) , \qquad \mathbf { G } = { \binom { \mathbf { G } _ { 1 } } { \mathbf { G } _ { 2 } } } ,
$$

where $\mathbf { G } _ { 1 } \in \mathbb { R } ^ { \ell \times m }$ . Since $\nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } )$ is the smallest eigenvalue of $\mathbf { R R } ^ { \top }$ restricted to $\operatorname { I m } ( \mathbf { P } )$ , and $\mathbf { G } ^ { \top } \mathbf { G } \preceq \lambda _ { \operatorname* { m a x } } ( \mathbf { G } ^ { \top } \mathbf { G } ) \mathbf { I } _ { m }$ , we have

$$
\nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) = \lambda _ { \operatorname* { m i n } } \Bigl ( \mathbf { G } _ { 1 } ( \mathbf { G } ^ { \top } \mathbf { G } ) ^ { - 1 } \mathbf { G } _ { 1 } ^ { \top } \Bigr ) \geq \frac { \lambda _ { \operatorname* { m i n } } \bigl ( \mathbf { G } _ { 1 } \mathbf { G } _ { 1 } ^ { \top } \bigr ) } { \lambda _ { \operatorname* { m a x } } \bigl ( \mathbf { G } ^ { \top } \mathbf { G } \bigr ) } .
$$

Since $\ell \leq m$ , the Gaussian matrix $\mathbf { G } _ { 1 }$ has full row rank with probability one, so that $\nu _ { \mathbf { P } } ( \mathbf { R } ^ { \top } ) > 0$ In the high-dimensional limit, since $\alpha < \kappa ,$ the limits of the extreme eigenvalues of sample covariance matrices (Yin et al., 1988; Bai and Yin, 1993) give the almost sure convergence

$$
\lambda _ { \mathrm { m i n } } ( \mathbf { G } _ { 1 } \mathbf { G } _ { 1 } ^ { \top } ) \xrightarrow [ d  \infty ] { } ( 1 - \sqrt { \frac { \alpha } { \kappa } } ) ^ { 2 } , \qquad \lambda _ { \mathrm { m a x } } ( \mathbf { G } ^ { \top } \mathbf { G } ) \xrightarrow [ d  \infty ] { } ( 1 + \frac { 1 } { \sqrt { \kappa } } ) ^ { 2 } ,
$$

so that any $C _ { \alpha , \kappa }$ strictly smaller than the ratio of these limits works.

Step 4. The finite-dimensional case. We now prove equation (25). As $\rho \to 0$ , we have the uniform convergence on compact sets

$$
\frac { 1 } { 2 \rho } r _ { t } ^ { \mathrm { L } } \big ( \gamma _ { \rho } ( x ) \big ) \xrightarrow [ \rho  0 ] { } \bar { r } _ { t } ^ { \mathrm { L } } ( x ) , \qquad \xrightarrow [ 2 \rho ] { 1 } r _ { t } ^ { \mathrm { R } } \big ( \gamma _ { \rho } ( x ) \big ) \xrightarrow [ \rho  0 ] { } \bar { r } _ { t } ^ { \mathrm { R } } ( x ) , \qquad q _ { t } \big ( \gamma _ { \rho } ( x ) \big ) \xrightarrow [ \rho  0 ] { } 1 ,\tag{86}
$$

where

$$
\bar { r } _ { t } ^ { \mathsf { L } } ( x ) = \sinh ( t x ) ^ { 2 } , \qquad \bar { r } _ { t } ^ { \mathsf { R } } ( x ) = \cosh ( t x ) ^ { 2 } .
$$

Hence, equation (85) and the monotonicity of $\bar { r } _ { t } ^ { \mathsf { a } }$ give:

$$
\operatorname* { l i m } _ { \rho \to 0 } \Delta _ { k , \rho } ^ { \mathsf { a } } ( t ) ^ { 2 } \leq \frac { 2 ( d - k ) } { \nu _ { k } d } \frac { \bar { r } _ { t } ^ { \mathsf { a } } ( \sigma _ { k + 1 } ) } { \bar { r } _ { t } ^ { \mathsf { a } } ( \sigma _ { k } ) } .
$$

Since $\nu _ { k } > 0$ with probability one by Lemma D.7, and $\bar { r } _ { t } ^ { \mathsf { a } } ( x ) \underset { t \to \infty } { \sim } \frac { 1 } { 4 } e ^ { 2 t x }$ for $x > 0$ , we obtain equation (25).

Step 5. Intermediate projectors. Before proving the high-dimensional case, we introduce slightly diferent projectors than the top-k. Indeed, taking the high-dimensional limit in equation (85) would require knowing the limit of the $k ^ { \mathrm { t h } }$ and $( k + 1 ) ^ { \mathrm { t h } }$ singular values of C, which is a finer requirement than Assumption 1.3. However, the cutof $x _ { \alpha }$ defined in Theorem 10 asymptotically selects a fraction α of the singular directions.

Let us consider:

$$
\mathbf { P } _ { \alpha } ^ { C } = \sum _ { i = 1 } ^ { m ^ { * } } \mathbf { 1 } _ { \sigma _ { i } \geq x _ { \alpha } \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } } .
$$

Although $\sigma _ { k }$ can difer from $x _ { \alpha }$ in the high-dimensional limit, the characterization of $x _ { \alpha }$ guarantees that only $o ( d )$ singular values can remain in the gap between $\sigma _ { k }$ and $x _ { \alpha }$ . Let us denote by ℓ the

rank of $\mathbf { P } _ { \alpha } ^ { C }$ and consider $\mathbf { Q } _ { \alpha , \rho } ^ { \mathsf { a } } ( t )$ the projector onto the top-ℓ eigenspaces of $\mathbf { M } _ { \rho } ^ { \mathsf { a } } ( t )$ . Then $\ell / d$ converges toward $\tilde { \mu } _ { C } ( [ x _ { \alpha } , \infty ) ) = \alpha$ . As a consequence, since $k \sim \alpha d \colon$

$$
\frac { 1 } { d } \big \| \mathbf { P } _ { \alpha } ^ { C } - \mathbf { P } _ { k } ^ { C , \mathrm { R } } \big \| _ { F } ^ { 2 } = \frac { 1 } { d } \big | \mathrm { r a n k } ( \mathbf { P } _ { \alpha } ^ { C } ) - k \big | \xrightarrow [ d  \infty ] { } 0 .
$$

The same identity holds for $\mathbf { Q } _ { \alpha , \rho } ^ { \mathsf { a } } ( t )$ and $\mathbf { Q } _ { k , \rho } ^ { \mathsf { a } } ( t )$ . Now, as a consequence of the triangle inequality, we have:

$$
\frac { 1 } { \sqrt d } \Big \| \mathbf { Q } _ { k , \rho } ^ { \alpha } ( t ) - \mathbf { P } _ { k } ^ { C , \mathbb { R } } \Big \| _ { F } - \Big \| \mathbf { Q } _ { \alpha , \rho } ^ { \alpha } ( t ) - \mathbf { P } _ { \alpha } ^ { C } \Big \| _ { F } \Big | \leq \frac { 1 } { \sqrt d } \Big \| \mathbf { P } _ { \alpha } ^ { C } - \mathbf { P } _ { k } ^ { C , \mathbb { R } } \Big \| _ { F } + \frac { 1 } { \sqrt d } \| \mathbf { Q } _ { \alpha , \rho } ^ { \alpha } ( t ) - \mathbf { Q } _ { k , \rho } ^ { \alpha } ( t ) \| _ { F } .
$$

Therefore, Lemma D.5 guarantees that:

$$
\operatorname* { l i m } _ { d \to \infty } \Delta _ { k , \rho } ^ { a } ( t ) ^ { 2 } \leq \operatorname* { l i m } _ { d \to \infty } \frac { 1 } { d } \| \mathbf { Q } _ { \alpha , \rho } ^ { a } ( t ) - \mathbf { P } _ { \alpha } ^ { C } \| _ { F } ^ { 2 } .
$$

Applying again Lemma D.6 to these new projectors and using the same bounds as at the end of Step 2, we obtain a similar bound to the one in equation (85):

$$
\operatorname* { l i m } _ { d \to \infty } \operatorname* { s u p } _ { \mathbf { \Theta } } \Delta _ { k , \rho } ^ { \mathbf { a } } ( t ) ^ { 2 } \leq \operatorname* { l i m } _ { d \to \infty } \operatorname* { s u p } _ { \nu _ { \alpha } d } \frac { 2 } { \ L _ { x \in [ 0 , L ] } } q _ { t } ( \gamma _ { \rho } ( x ) ) ) \frac { \sum _ { i = 1 } ^ { d } \mathbf { 1 } _ { \sigma _ { i } < x _ { \alpha } r _ { t } ^ { \mathbf { a } } } ( \gamma _ { i } ) } { \operatorname* { i n f } _ { x \in [ x _ { \alpha } , L ] } r _ { t } ^ { \mathbf { a } } ( \gamma _ { \rho } ( x ) ) } .\tag{87}
$$

In this case $\nu _ { \alpha } = \nu _ { { \bf P } _ { \alpha } ^ { C } } ( { \bf R } ^ { \top } )$ (see equation (81)). This bound only involves the fixed cutof $x _ { \alpha }$ and does not require assumptions on the asymptotic behavior of the individual singular values of $\mathbf { C } .$

Step 6. High-dimensional case. For the high-dimensional limit, we take $d \to \infty$ first. Since the asymptotic spectral distribution of $( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { \top / 2 }$ is $\tilde { \mu } _ { C } .$ , and due to the characterization of $x _ { \alpha }$ we have the convergence:

$$
\frac { 1 } { d } \sum _ { i = 1 } ^ { d } \mathbf { 1 } _ { \sigma _ { i } < x _ { \alpha } } r _ { t } ^ { \mathbf { a } } ( \gamma _ { i } ) \xrightarrow [ d  \infty ] { } \int _ { 0 } ^ { x _ { \alpha } } r _ { t } ^ { \mathbf { a } } ( \gamma _ { \rho } ( x ) ) \tilde { \mu } _ { C } ( \mathrm { d } x ) .
$$

We apply the high-dimensional part of Lemma D.7 to $\nu _ { \alpha }$ , so that there exists a non-random constant $C _ { \alpha } > 0$ (independent of $d , \rho , t )$ such that with probability one, $\nu _ { \alpha } > C _ { \alpha }$ for d large enough (guaranteed by Step 3). Therefore, equation (87) guarantees that, with probability one:

$$
\operatorname* { l i m } _ { d \to \infty } \mathrm { s u p } \Delta _ { k , \rho } ^ { \mathrm { a } } ( t ) ^ { 2 } \leq \frac { 2 } { C _ { \alpha } } \operatorname* { s u p } _ { x \in [ 0 , L ] } q _ { t } ( \gamma _ { \rho } ( x ) ) \frac { \int _ { 0 } ^ { x _ { \alpha } } r _ { t } ^ { \mathrm { a } } ( \gamma _ { \rho } ( x ) ) \tilde { \mu } _ { C } ( \mathrm { d } x ) } { \operatorname* { i n f } _ { x \in [ x _ { \alpha } , L ] } r _ { t } ^ { \mathrm { a } } ( \gamma _ { \rho } ( x ) ) } .
$$

Now taking the $\rho \to 0$ limit, using the convergence in equation (86) and the monotonicity of $\bar { r } _ { t } ^ { \mathsf { a } }$ we get:

$$
\operatorname* { l i m } _ { \rho \to 0 } \operatorname* { s u p } _ { d \to \infty } \operatorname * { s u p } _ { } \Delta _ { k , \rho } ^ { \mathsf { a } } ( t ) ^ { 2 } \leq \frac 2 { C _ { \alpha } } \frac { \int _ { 0 } ^ { x _ { \alpha } } \bar { r } _ { t } ^ { \mathsf { a } } ( x ) \tilde { \mu } _ { C } ( \mathrm { d } x ) } { \bar { r } _ { t } ^ { \mathsf { a } } ( x _ { \alpha } ) } .
$$

Now, we have the bound, for all $t > 0$ and $x \in [ 0 , x _ { \alpha } ]$

$$
\frac { \sinh ( t x ) ^ { 2 } } { \sinh ( t x _ { \alpha } ) ^ { 2 } } \leq e ^ { - 2 t ( x _ { \alpha } - x ) } , \qquad \frac { \cosh ( t x ) ^ { 2 } } { \cosh ( t x _ { \alpha } ) ^ { 2 } } \leq 4 e ^ { - 2 t ( x _ { \alpha } - x ) } ,
$$

so that for some $\tilde { C } _ { \alpha } > 0 \colon$

$$
\operatorname* { l i m } _ { \rho \to 0 } \operatorname* { l i m } _ { d \to \infty } \Delta _ { k , \rho } ^ { a } ( t ) ^ { 2 } \leq \tilde { C } _ { \alpha } \int _ { 0 } ^ { x _ { \alpha } } e ^ { - 2 t ( x _ { \alpha } - x ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) .\tag{88}
$$

Whenever $\tilde { \mu } _ { C } ( ( x _ { \alpha } - \epsilon _ { \alpha } , x _ { \alpha } ) ) = 0$ , the integral is $O ( e ^ { - 2 \epsilon _ { \alpha } t } )$ , which proves the estimate in equation (26). In all cases, using the decomposition of $\tilde { \mu } _ { C }$ and the boundedness of $\phi _ { C }$ (Assumption 1.3), we have:

$$
\begin{array} { r l } & { \displaystyle \int _ { 0 } ^ { x _ { \alpha } } e ^ { - 2 t ( x _ { \alpha } - x ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) \le ( 1 - \beta \kappa ^ { * } ) e ^ { - 2 t x _ { \alpha } } + \beta \kappa ^ { * } \| \phi _ { C } \| _ { \infty } \displaystyle \int _ { 0 } ^ { x _ { \alpha } } e ^ { - 2 t ( x _ { \alpha } - x ) } \mathrm { d } x } \\ & { \qquad \le ( 1 - \beta \kappa ^ { * } ) e ^ { - 2 t x _ { \alpha } } + \frac { \beta \kappa ^ { * } \| \phi _ { C } \| _ { \infty } } { 2 t } . } \end{array}
$$

Together with equation (88), this gives the estimate in equation (27). This concludes the proof of Theorem 10.

## D.4 Saddle-to-Saddle Dynamics

We now prove Theorem 11. Set $t _ { \rho } = s \log ( \rho ^ { - 1 } )$ and let $\mathcal { T } _ { s }$ index the singular values selected at rescaled time s,

$$
{ \mathcal { T } } _ { s } = \{ i \leq m ^ { * } : \sigma _ { i } > ( 2 s ) ^ { - 1 } \} = \{ 1 , \ldots , k \} .
$$

By assumption, $k \leq m$ and no singular value lies at the threshold. We extend $\left( \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m ^ { * } } \right)$ into an orthonormal right singular basis of C and set $\sigma _ { i } = 0$ for $i > m ^ { * }$ . We consider the solutions given in Lemma B.1 and use the associated notation. For $i \in \mathcal { Z } _ { s }$ , we let

$$
d _ { \rho , i } = q _ { t _ { \rho } } ( \gamma _ { i } ) - 1 , \qquad { \bf D } _ { \rho } = \mathrm { d i a g } \bigl ( d _ { \rho , i } , i \in \mathcal { T } _ { s } \bigr ) \in \mathbb { R } ^ { k \times k } ,
$$

and

$$
\mathbf { R } _ { s } = \left( \begin{array} { c } { \mathbf { v } _ { 1 } ^ { \top } \mathbf { R } } \\ { \vdots } \\ { \mathbf { v } _ { k } ^ { \top } \mathbf { R } } \end{array} \right) \in \mathbb { R } ^ { k \times m } .
$$

Step 1. Scalar asymptotics. In the proof, we will use the following lemma.

Lemma D.8. Let $s > 0 , t _ { \rho } = s \log ( \rho ^ { - 1 } )$ and $x _ { s } = ( 2 s ) ^ { - 1 }$ . As $\rho \to 0 .$ , for $\mathsf { a } = \mathsf { L } , \mathsf { R }$ and $x > 0$ , we have:

$$
q _ { t _ { \rho } } ( \gamma _ { x } ) - 1 \underset { \rho  0 } { \sim } \frac { 1 } { 2 x } \rho ^ { 1 - 2 s x } , \qquad r _ { t _ { \rho } } ^ { \mathrm { a } } ( \gamma _ { x } ) \underset { \rho  0 } { \sim } \frac { 1 } { 2 } \rho ^ { 1 - 2 s x } .\tag{89}
$$

Therefore:

• For $0 \leq x < x _ { s } ,$ as $\rho \to 0 , q _ { t _ { \rho } } ( \gamma _ { x } )$ converges to 1 and both $r _ { t _ { \rho } } ^ { \mathsf { L } } ( \gamma _ { x } ) , r _ { t _ { \rho } } ^ { \mathsf { R } } ( \gamma _ { x } )$ converge to 0.

• For $x > x _ { s } , q _ { t _ { \rho } } ( \gamma _ { x } ) , r _ { t _ { \rho } } ^ { \mathsf { L } } ( \gamma _ { x } ) , r _ { t _ { \rho } } ^ { \mathsf { R } } ( \gamma _ { x } )$ diverge to +∞.

Proof. Let $x > 0$ . The formulas in Theorem 4 can be written as

$$
q _ { t _ { \rho } } ( \gamma _ { x } ) - 1 = \frac { \rho } { 2 \gamma _ { x } } ( \rho ^ { - 2 s \gamma _ { x } } - \rho ^ { 2 s \gamma _ { x } } ) + \frac { \rho ^ { 2 } } { 2 \gamma _ { x } ^ { 2 } } ( \rho ^ { - 2 s \gamma _ { x } } + \rho ^ { 2 s \gamma _ { x } } - 2 ) ,
$$

$$
r _ { t _ { \rho } } ^ { \perp } ( \gamma _ { x } ) = \frac { \rho } { 2 } \left( 1 - \frac { \rho ^ { 2 } } { \gamma _ { x } ^ { 2 } } \right) \big ( \rho ^ { - s \gamma _ { x } } - \rho ^ { s \gamma _ { x } } \big ) ^ { 2 } ,
$$

$$
r _ { t _ { \rho } } ^ { \mathsf { R } } ( \gamma _ { x } ) = \frac { \rho } { 2 } \left[ \left( 1 + \frac { \rho } { \gamma _ { x } } \right) \rho ^ { - s \gamma _ { x } } + \left( 1 - \frac { \rho } { \gamma _ { x } } \right) \rho ^ { s \gamma _ { x } } \right] ^ { 2 } .
$$

As $\rho \to 0$ we have $\gamma _ { x }  x$ and also $\rho ^ { \gamma _ { x } - x }  1$ , which leads to equation (89). The limits follow for $x > 0$ . For $x = 0$ , we have $q _ { t _ { \rho } } ( \rho ) = \rho ^ { - 2 s \rho } , r _ { t _ { \rho } } ^ { \perp } ( \rho ) = 0$ , and $r _ { t _ { \rho } } ^ { \mathsf { R } } ( \rho ) = 2 \rho ^ { 1 - 2 s \rho }$ , which gives the desired result. □

Therefore, for $i \in \mathcal { Z } _ { s }$ we have:

$$
d _ { \rho , i } \underset { \rho  0 } { \sim } \frac { 1 } { 2 \sigma _ { i } } \rho ^ { 1 - 2 s \sigma _ { i } } , \qquad \frac { \sqrt { 2 \rho } \chi _ { \rho } ^ { U } ( \sigma _ { i } ) } { \sqrt { d _ { \rho , i } } } \underset { \rho  0 } { \longrightarrow } \sqrt { \sigma _ { i } } , \qquad \frac { \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { i } ) } { \sqrt { d _ { \rho , i } } } \underset { \rho  0 } { \longrightarrow } \sqrt { \sigma _ { i } } .\tag{90}
$$

In particular $d _ { \rho , i } \to \infty$ for $i \in \mathcal { Z } _ { s }$ . For $i \notin \mathcal { I } _ { s }$ , we rather have $q _ { t _ { \rho } } ( \gamma _ { i } ) \to 1$ and

$$
\sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { i } ) \xrightarrow [ \rho  0 ] { } 0 , \qquad \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { i } ) \xrightarrow [ \rho  0 ] { } 0 .
$$

Whenever $\sigma _ { i } = 0$ , we have $q _ { t _ { \rho } } ( \rho ) = e ^ { 2 \rho t _ { \rho } } \to 1$ and $\chi _ { t _ { \rho } } ^ { V } ( 0 ) = e ^ { \rho t _ { \rho } }$ , therefore the previous statements also apply.

Step 2. Limit of the normalization matrix. Recall the expressions from Lemma B.1:

$$
\begin{array} { r l } & { \mathbf { U } ( t _ { \rho } ) \mathbf { U } ( t _ { \rho } ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t _ { \rho } } ^ { U } ( \mathbf { C } ) \mathbf { R } \mathbf { K } _ { \rho } ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t _ { \rho } } ^ { U } ( \mathbf { C } ) ^ { \top } , } \\ & { \mathbf { V } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t _ { \rho } } ^ { V } ( \mathbf { S } ) \mathbf { R } \mathbf { K } _ { \rho } ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t _ { \rho } } ^ { V } ( \mathbf { S } ) , } \\ & { \mathbf { U } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top } = 2 \rho \boldsymbol { \chi } _ { t _ { \rho } } ^ { U } ( \mathbf { C } ) \mathbf { R } \mathbf { K } _ { \rho } ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol { \chi } _ { t _ { \rho } } ^ { V } ( \mathbf { S } ) , } \end{array}\tag{91}
$$

where $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ , and $\mathbf { K } _ { \rho } = \mathbf { R } ^ { \top } q _ { t _ { \rho } } ( \mathbf { r } _ { \rho } ) \mathbf { R }$ . Then, we have, since $\mathbf { R } ^ { \top } \mathbf { R } = \mathbf { I } _ { m }$

$$
\mathbf { K } _ { \rho } = \mathbf { I } _ { m } + \mathbf { R } _ { s } ^ { \top } \mathbf { D } _ { \rho } \mathbf { R } _ { s } + \mathbf { R } ^ { \top } \sum _ { i \notin \mathcal { T } _ { s } } \big ( q _ { t _ { \rho } } ( \gamma _ { i } ) - 1 \big ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } \mathbf { R } .
$$

We call $\mathbf { E } _ { \rho }$ the last term, which converges to zero as $\rho \to 0$ . Now, we let:

$$
\mathbf { M } _ { \rho } = \mathbf { D } _ { \rho } ^ { 1 / 2 } \mathbf { R } _ { s } \mathbf { K } _ { \rho } ^ { - 1 } \mathbf { R } _ { s } ^ { \top } \mathbf { D } _ { \rho } ^ { 1 / 2 } = \mathbf { D } _ { \rho } ^ { 1 / 2 } \mathbf { R } _ { s } \Big ( \mathbf { I } _ { m } + \mathbf { R } _ { s } ^ { \top } \mathbf { D } _ { \rho } \mathbf { R } _ { s } + \mathbf { E } _ { \rho } \Big ) ^ { - 1 } \mathbf { R } _ { s } ^ { \top } \mathbf { D } _ { \rho } ^ { 1 / 2 } .
$$

Using the Woodbury formula, we have:

$$
\mathbf { M } _ { \rho } = \mathbf { I } _ { k } - \Big ( \mathbf { I } _ { k } + \mathbf { D } _ { \rho } ^ { 1 / 2 } \mathbf { R } _ { s } ( \mathbf { I } _ { m } + \mathbf { E } _ { \rho } ) ^ { - 1 } \mathbf { R } _ { s } ^ { \top } \mathbf { D } _ { \rho } ^ { 1 / 2 } \Big ) ^ { - 1 } .
$$

Since $\mathbf E _ { \rho } \to \mathbf 0$ , the matrix ${ \mathbf { I } _ { m } } + { \mathbf { E } _ { \rho } }$ is invertible for $\rho$ small enough. As $\rho \to 0$ , the matrix $\mathbf { R } _ { s } ( \mathbf { I } _ { m } + \mathbf { E } _ { \rho } ) ^ { - 1 } \mathbf { R } _ { s } ^ { \top }$ converges to ${ \bf R } _ { s } { \bf R } _ { s } ^ { \top }$ . By the definition of $\mathbf { R } _ { s }$ and Lemma B.2, $\mathbf { R } _ { s }$ has full row rank with probability one, so ${ \bf R } _ { s } { \bf R } _ { s } ^ { \top }$ is positive definite. Moreover, all diagonal coeficients of $\mathbf { D } _ { \rho }$ diverge by equation (90), so the inverse in the above equation tends to zero. Hence, we have $\mathbf { M } _ { \rho } \to \mathbf { I } _ { k }$ as $\rho \to 0$

Step 3. Limit of the dynamics. To conclude the proof, decompose:

$$
\begin{array} { r l } & { \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { U } ( \mathbf { C } ) = \displaystyle \sum _ { i \in \mathcal { L } _ { s } } \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { i } ) \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \displaystyle \sum _ { i \notin \mathcal { L } _ { s } } \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { i } ) \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } , } \\ & { \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { V } ( \mathbf { S } ) = \displaystyle \sum _ { i \in \mathcal { T } _ { s } } \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { i } ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } + \displaystyle \sum _ { i \notin \mathcal { T } _ { s } } \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { i } ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } . } \end{array}
$$

For $i \notin \mathcal { I } _ { s }$ , we have shown that $\sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { i } )$ and $\sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { i } )$ converge to zero as $\rho \to 0$ . Since ${ \bf K } _ { \rho } \succeq { \bf I } _ { m }$ , we get:

$$
\sum _ { i \notin \mathbb { Z } _ { s } } \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { i } ) \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } \mathbf { R } \mathbf { K } _ { \rho } ^ { - 1 / 2 } \xrightarrow [ \rho  0 ] { } \mathbf { 0 } ,
$$

$$
\sum _ { i \notin \mathbb { Z } _ { s } } \sqrt { 2 \rho } \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { i } ) \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } \mathbf { R } \mathbf { K } _ { \rho } ^ { - 1 / 2 } \xrightarrow [ \rho  0 ] { } \mathbf { 0 } .
$$

Therefore, when computing the limit of the matrices in equation (91), only the terms in $\mathcal { T } _ { s }$ remain, and we obtain:

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { \rho \to 0 } \mathbf { U } ( t _ { \rho } ) \mathbf { U } ( t _ { \rho } ) ^ { \top } = \displaystyle \operatorname* { l i m } _ { \rho \to 0 } \sum _ { i , j \in \mathbb { Z } _ { s } } \frac { 2 \rho \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { i } ) \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { j } ) } { \sqrt { d _ { \rho , i } d _ { \rho , j } } } \mathbf { M } _ { \rho } ^ { i j } \mathbf { u } _ { i } \mathbf { u } _ { j } ^ { \top } , } \\ { \displaystyle \operatorname* { l i m } _ { \rho \to 0 } \mathbf { V } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top } = \displaystyle \operatorname* { l i m } _ { \rho \to 0 } \sum _ { i , j \in \mathbb { Z } _ { s } } \frac { 2 \rho \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { i } ) \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { j } ) } { \sqrt { d _ { \rho , i } d _ { \rho , j } } } \mathbf { M } _ { \rho } ^ { i j } \mathbf { v } _ { i } \mathbf { v } _ { j } ^ { \top } , } \\ { \displaystyle \operatorname* { l i m } _ { \rho \to 0 } \mathbf { U } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top } = \displaystyle \operatorname* { l i m } _ { \rho \to 0 } \sum _ { i , j \in \mathbb { Z } _ { s } } \frac { 2 \rho \chi _ { t _ { \rho } } ^ { U } ( \sigma _ { i } ) \chi _ { t _ { \rho } } ^ { V } ( \sigma _ { j } ) } { \sqrt { d _ { \rho , i } d _ { \rho , j } } } \mathbf { M } _ { \rho } ^ { i j } \mathbf { u } _ { i } \mathbf { v } _ { j } ^ { \top } . } \end{array}
$$

Using the asymptotics in equation (90) and the fact that $\mathbf { M } _ { \rho }$ converges to $\mathbf { I } _ { k }$ , we get:

$$
\begin{array} { r l } & { \mathbf { U } ( t _ { \rho } ) \mathbf { U } ( t _ { \rho } ) ^ { \top } \xrightarrow [ \rho  0 ] { } \displaystyle \sum _ { i \in \mathcal { T } _ { s } } \sigma _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } , } \\ & { \mathbf { V } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top } \xrightarrow [ \rho  0 ] { } \displaystyle \sum _ { i \in \mathcal { T } _ { s } } \sigma _ { i } \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } , } \\ & { \mathbf { U } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top } \xrightarrow [ \rho  0 ] { } \displaystyle \sum _ { i \in \mathcal { T } _ { s } } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } . } \end{array}
$$

This is precisely the result of Theorem 11.

Step 4. Saddle-to-saddle interpretation. We finally make the saddle-to-saddle interpretation precise. Write

$$
\mathbf { P } _ { I } = ( \mathbf { u } _ { i } ) _ { i \in \mathcal { T } _ { s } } , \qquad \mathbf { Q } _ { I } = ( \mathbf { v } _ { i } ) _ { i \in \mathcal { T } _ { s } } , \qquad \Sigma _ { I } = \mathrm { d i a g } ( \sigma _ { i } , i \in \mathcal { T } _ { s } ) .
$$

Since $\mathbf { U } ( t _ { \rho } ) \mathbf { U } ( t _ { \rho } ) ^ { \top }$ and $\mathbf { V } ( t _ { \rho } ) \mathbf { V } ( t _ { \rho } ) ^ { \top }$ converge, the pair $( \mathbf { U } ( t _ { \rho } ) , \mathbf { V } ( t _ { \rho } ) )$ is bounded as $\rho \to 0$ . Let $\mathbf { \Pi } ( \mathbf { U } _ { s } , \mathbf { V } _ { s } )$ be any of its limit points, and $\mathbf { Z } _ { s } = \mathbf { U } _ { s } \mathbf { V } _ { s } ^ { \top }$ . By Theorem 11:

$$
\mathbf { U } _ { s } \mathbf { U } _ { s } ^ { \top } = \mathbf { P } _ { I } \boldsymbol { \Sigma } _ { I } \mathbf { P } _ { I } ^ { \top } , \qquad \mathbf { V } _ { s } \mathbf { V } _ { s } ^ { \top } = \mathbf { Q } _ { I } \boldsymbol { \Sigma } _ { I } \mathbf { Q } _ { I } ^ { \top } , \qquad \mathbf { U } _ { s } \mathbf { V } _ { s } ^ { \top } = \mathbf { P } _ { I } \boldsymbol { \Sigma } _ { I } \mathbf { Q } _ { I } ^ { \top } .
$$

As a consequence of the relationships for $\mathbf { U } _ { s } , \mathbf { V } _ { s }$ there exist $\mathbf { O } _ { U } , \mathbf { O } _ { V } \in \mathbb { R } ^ { k \times m }$ such that ${ \bf O } _ { U } { \bf O } _ { U } ^ { \top } =$ $\mathbf { I } _ { k } , \mathbf { O } _ { V } \mathbf { O } _ { V } ^ { \top } = \mathbf { I } _ { k }$ and we have ${ \bf U } _ { s } = { \bf P } _ { I } { \bf \Sigma } { \bf { \Sigma } } _ { I } ^ { 1 / 2 } { \bf O } _ { U }$ and ${ \bf V } _ { s } = { \bf Q } _ { I } { \bf \Sigma } { \bf \Sigma } _ { I } ^ { 1 / 2 } { \bf O } _ { V }$ . The above equation for $\mathbf { U } _ { s } \mathbf { V } _ { s } ^ { \top }$ implies that $\mathbf { O } _ { U } \mathbf { O } _ { V } ^ { \top } = \mathbf { I } _ { k }$ and therefore $\mathbf { O } _ { U } = \mathbf { O } _ { V }$ , which we denote by O from now on.

From these identities it easily follows that:

$$
\begin{array} { r } { ( \mathbf { Z } _ { s } - \mathbf { C } ) \mathbf { V } _ { s } = \mathbf { 0 } , \quad \quad \quad ( \mathbf { Z } _ { s } - \mathbf { C } ) ^ { \top } \mathbf { U } _ { s } = \mathbf { 0 } , } \end{array}
$$

since $\mathbf { Z } _ { s } - \mathbf { C }$ only contains singular directions of C orthogonal to ${ \bf P } _ { I }$ and $\mathbf { Q } _ { I }$ . Since the two previous matrices are precisely the gradient of the loss (with respect to U and V), the point $\mathbf { \Pi } ( \mathbf { U } _ { s } , \mathbf { V } _ { s } )$ is a critical point of the loss.

Let us now show this is a strict saddle. For $k < \operatorname* { m i n } ( m , m ^ { * } )$ , choose a direction that is not learned yet $j \notin \mathcal { T } _ { s }$ with $j \leq m ^ { * }$ , so that $\sigma _ { j } > 0$ , and a vector $\mathbf { e } \in \mathbb { R } ^ { m }$ of unit norm in $\operatorname { K e r } ( \mathbf { O } )$ ， and consider for some $\epsilon > 0$ and $\lambda \in \mathbb { R }$

$$
\mathbf { U } _ { \epsilon } = \mathbf { U } _ { s } + \epsilon \mathbf { u } _ { j } \mathbf { e } ^ { \top } , \mathbf { V } _ { \epsilon } = \mathbf { V } _ { s } + \lambda \epsilon \mathbf { v } _ { j } \mathbf { e } ^ { \top } .
$$

Then, we have

$$
\mathbf { U } _ { \epsilon } \mathbf { V } _ { \epsilon } ^ { \top } = \mathbf { Z } _ { s } + \lambda \epsilon ^ { 2 } \mathbf { u } _ { j } \mathbf { v } _ { j } ^ { \top } .
$$

Since the loss is given by $\begin{array} { r } { { \frac { 1 } { 2 } } \| \mathbf { U } \mathbf { V } ^ { \top } - \mathbf { C } \| _ { F } ^ { 2 } } \end{array}$ , up to an additive constant, this leads to

$$
\mathscr { L } ( \mathbf { U } _ { \epsilon } , \mathbf { V } _ { \epsilon } ) - \mathscr { L } ( \mathbf { U } _ { s } , \mathbf { V } _ { s } ) = \frac { 1 } { 2 } \Big ( ( \sigma _ { j } - \lambda \epsilon ^ { 2 } ) ^ { 2 } - \sigma _ { j } ^ { 2 } \Big ) = - \sigma _ { j } \lambda \epsilon ^ { 2 } + \frac { 1 } { 2 } \lambda ^ { 2 } \epsilon ^ { 4 } .
$$

If ϵ is small enough, the above term is negative for $\lambda > 0$ and positive for $\lambda ~ < ~ 0$ . Since $\epsilon \mapsto ( \mathbf { U } _ { \epsilon } , \mathbf { V } _ { \epsilon } )$ is a straight line through the critical point, the $\epsilon ^ { 2 }$ term shows that the Hessian of the loss has a negative eigenvalue. Hence every intermediate limiting point given in Theorem 11 is a strict saddle. Between two consecutive thresholds $( 2 \sigma _ { i } ) ^ { - 1 }$ , the selected index set $\mathcal { T } _ { s }$ is constant, and crossing a threshold moves the limit from one critical point to the next. This is the expected saddle-to-saddle dynamics.

## D.5 Spectrum in the Small-Initialization Limit

We shall now prove Theorem 12.

## D.5.1 Eigenvalues

Let us start with the limit for $\mathfrak { m } _ { t } ^ { U } , \mathfrak { m } _ { t } ^ { V }$ when $t = s \log ( \rho ^ { - 1 } )$ . For this proof only, write $g _ { t } ^ { \mathsf { L } } ( w ) = g _ { t } ( w )$ and $g _ { t } ^ { \mathsf { R } } ( w ) = g _ { t } ( w - 2 \rho )$ . Then, for $\mathsf { a } = \mathsf { L } , \mathsf { R }$

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { a } } ( w ) \big ( r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } .
$$

We now study the asymptotics as $\rho \to 0$ for $g _ { t } ^ { \mathsf { L } } , g _ { t } ^ { \mathsf { R } }$ , with $t _ { \rho } = s \log ( \rho ^ { - 1 } )$ and $x _ { s } = ( 2 s ) ^ { - 1 }$ . For clarity, we write $g _ { t _ { \rho } } ^ { \mathsf { a } } = g _ { \rho } ^ { \mathsf { a } }$ . We only consider $w < 0$ and will apply Corollary 18 to the Stieltjes transforms.

We apply Proposition 19 directly with parameter $\rho \to 0$ and:

$$
h _ { \rho } = g _ { \rho } ^ { \mathsf { a } } ( w ) , \qquad \quad \quad \quad a _ { \rho } ( x ) = \kappa \big ( r _ { t _ { \rho } } ^ { \mathsf { a } } ( \gamma _ { x } ) - w q _ { t _ { \rho } } ( \gamma _ { x } ) \big ) ,
$$

and $c _ { \rho } = 1$ . Then, Lemma D.8 gives:

$$
a _ { \rho } ( x ) \xrightarrow [ { \rho \to 0 } ] { } a _ { \infty } ( x ) = \left\{ \begin{array} { l l } { - \kappa w , } & { x < x _ { s } , } \\ { + \infty , } & { x > x _ { s } . } \end{array} \right.
$$

Hence the limit function in Proposition 19 is:

$$
\Phi ( u ) = \frac { 1 - T _ { C } ( x _ { s } ) } { 1 - \kappa - \kappa w u } .
$$

Since $x _ { s } > x _ { * }$ , we have $T _ { C } ( x _ { s } ) < \kappa ,$ and $\Phi ( u ) = 1$ has the unique solution on $( 0 , \infty )$

$$
u _ { \infty } = - \frac { \kappa - T _ { C } ( x _ { s } ) } { \kappa w } > 0 .
$$

Therefore, using 2. (i) in Proposition 19, $g _ { \rho } ^ { \mathsf { a } } ( w )$ converges to this value, which we denote by $g _ { 0 } ^ { \mathsf { a } } ( w )$

Let us now show the limit of the Stieltjes transforms when $w < 0$ . Back to the expression of the Stieltjes transforms ${ \mathfrak { m } } _ { t } ^ { U } , { \mathfrak { m } } _ { t } ^ { V }$ in equations (12), (13), we focus on the integral term:

$$
\kappa g _ { \rho } ^ { \mathsf { a } } ( w ) \int \frac { q _ { t _ { \rho } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { \rho } ^ { \mathsf { a } } ( w ) \big ( r _ { t _ { \rho } } ^ { \mathsf { a } } ( \gamma _ { x } ) - w q _ { t _ { \rho } } ( \gamma _ { x } ) \big ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) .
$$

The integrand satisfies the uniform bound

$$
0 \le \frac { \kappa g _ { \rho } ^ { \mathsf { a } } ( w ) q _ { t _ { \rho } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { \rho } ^ { \mathsf { a } } ( w ) \big ( r _ { t _ { \rho } } ^ { \mathsf { a } } ( \gamma _ { x } ) - w q _ { t _ { \rho } } ( \gamma _ { x } ) \big ) } \le - \frac { 1 } { w } .
$$

Moreover, for every $x \neq x _ { s }$ , Lemma D.8 shows that it converges pointwise to

$$
\frac { \kappa g _ { \rho } ^ { \mathrm { a } } ( w ) q _ { t _ { \rho } } ( \gamma _ { x } ) } { 1 - \kappa + \kappa g _ { \rho } ^ { \mathrm { a } } ( w ) \big ( r _ { t _ { \rho } } ^ { \mathrm { a } } ( \gamma _ { x } ) - w q _ { t _ { \rho } } ( \gamma _ { x } ) \big ) } \xrightarrow [ \rho  0 ] { } \{ \begin{array} { l l } { \displaystyle \frac { \kappa g _ { 0 } ^ { \mathrm { a } } ( w ) } { 1 - \kappa - \kappa w g _ { 0 } ^ { \mathrm { a } } ( w ) } , } & { x < x _ { s } , } \\ { \displaystyle \frac { 1 } { x - w } , } & { x > x _ { s } . } \end{array} 
$$

Since $\tilde { \mu } _ { C }$ has no atom at $x _ { s } .$ dominated convergence, the expression of $g _ { 0 } ^ { \mathsf { a } } ( w )$ , and Theorem 4 then give equations (28), (29) for $w < 0$

The extension to $w \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ is guaranteed by Corollary 18. Indeed, the RHS of equations (28), (29) are holomorphic on $\mathbb { C } \setminus \mathbb { R } ^ { + }$ and the LHS are Stieltjes transforms, hence Lemma A.1 guarantees uniform boundedness.

## D.5.2 Singular Values

Let us now prove equation (30). To do so, we recall equation (15) that links the empirical measure of the singular values of the prediction $\mathbf { Z } ( t )$ to the spectral measure of $\mathbf { U } ( t ) \mathbf { U } ( t ) ^ { \top }$

$$
\int f ( x ) \mu _ { t } ^ { Z } ( \mathrm { d } x ) = \left( 1 - \frac { \tau } { \beta } \right) f ( 0 ) + \frac { \tau } { \beta } \int f \left( \sqrt { x ( x + 2 \rho ) } \right) \mu _ { t } ^ { U } ( \mathrm { d } x ) ,
$$

for any bounded continuous test function f. Now, in the $\rho \to 0$ limit, we have ${ \sqrt { x ( x + 2 \rho ) } } \to x$ for $x \geq 0$ . One has to be careful since $\mu _ { t } ^ { U }$ depends on $\rho .$ For simplicity let us write $\mu _ { \rho } ^ { U } : = \mu _ { t _ { \rho } } ^ { U }$ and $\overline { { \mu } } ^ { U }$ its weak limit as $\rho \to 0$ . Let $\theta _ { \rho } ( x ) = \sqrt { x ( x + 2 \rho ) }$ . Then:

$$
\left| \int f \circ \theta _ { \rho } \mathrm { d } \mu _ { \rho } ^ { U } - \int f \mathrm { d } \overline { { \mu } } ^ { U } \right| \leq \left| \int f \circ \theta _ { \rho } \mathrm { d } \mu _ { \rho } ^ { U } - \int f \mathrm { d } \mu _ { \rho } ^ { U } \right| + \left| \int f \mathrm { d } \mu _ { \rho } ^ { U } - \int f \mathrm { d } \overline { { \mu } } ^ { U } \right| .\tag{92}
$$

We first prove the convergence for bounded Lipschitz functions. If $f$ is Lipschitz with constant $K .$ , then:

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { + } } \left. f ( \theta _ { \rho } ( x ) ) - f ( x ) \right. \leq K \operatorname* { s u p } _ { x \in \mathbb { R } ^ { + } } \left. \sqrt { x ( x + 2 \rho ) } - x \right. .
$$

Now, since we have, for all $x > 0$ and trivially at $x = 0$

$$
0 \leq { \sqrt { x ( x + 2 \rho ) } } - x = { \frac { 2 \rho x } { { \sqrt { x ( x + 2 \rho ) } } + x } } \leq \rho ,
$$

the previous supremum goes to zero, and the first term of equation (92) satisfies, since $\mu _ { \rho } ^ { U }$ is a probability measure supported on $\mathbb { R } ^ { + }$ :

$$
| \int f \circ \theta _ { \rho } \mathrm { d } \mu _ { \rho } ^ { U } - \int f \mathrm { d } \mu _ { \rho } ^ { U } | \leq \operatorname* { s u p } _ { x \in \mathbb { R } ^ { + } } | f ( \theta _ { \rho } ( x ) ) - f ( x ) | { \overrightarrow { \operatorname* { \rho } _ { \rho  0 } } } 0 .
$$

The convergence of the second term in equation (92) is a simple consequence of the weak convergence of $\mu _ { \rho } ^ { U }$ toward $\overline { { \mu } } ^ { U }$ . As a consequence, for all bounded Lipschitz functions:

$$
\int f ( x ) \mu _ { t _ { \rho } } ^ { Z } ( \mathrm { d } x ) \xrightarrow [ \rho  0 ] { } ( 1 - \frac { \tau } { \beta } ) f ( 0 ) + \frac { \tau } { \beta } \int f ( x ) \overline { { \mu } } ^ { U } ( \mathrm { d } x ) .
$$

This implies the weak convergence of $\mu _ { t _ { \rho } } ^ { Z }$ . Since $x \mapsto ( x - w ) ^ { - 1 }$ is bounded and Lipschitz on $\mathbb { R } ^ { + }$ for every $w \in \mathbb { C } \setminus \mathbb { R } ^ { + }$ , the Stieltjes transforms converge to:

$$
\mathfrak { m } _ { t _ { \rho } } ^ { Z } ( w ) \xrightarrow [ \rho  0 ] { } - \frac { 1 } { w } ( 1 - \frac { \tau } { \beta } ) + \frac { \tau } { \beta } \operatorname* { l i m } _ { \rho  0 } \mathfrak { m } _ { t _ { \rho } } ^ { U } ( w ) .
$$

This leads to equation (30).

## D.5.3 Wasserstein Distance

Let us now prove the expression of the Wasserstein distance in equation (31). Denote:

$$
\overline { { \mathfrak { m } } } _ { s } ( w ) = - \frac { 1 } { w } \left( 1 - \frac { T _ { C } ( x _ { s } ) } { \beta } \right) + \kappa ^ { * } \int _ { x _ { s } } ^ { \infty } \frac { \phi _ { C } ( x ) { \mathrm { d } } x } { x - w } .
$$

Therefore, the associated measure is

$$
\mu _ { s } ( \mathrm { d } x ) = \left( 1 - \frac { T _ { C } ( x _ { s } ) } { \beta } \right) \delta _ { 0 } ( \mathrm { d } x ) + \kappa ^ { * } \mathbf { 1 } _ { x > x _ { s } } \phi _ { C } ( x ) \mathrm { d } x .
$$

Recall the decomposition of $\mu _ { C }$ in Assumption 1.3. Then, the absolutely continuous part of $\mu _ { s }$ is exactly the restriction of the positive part of $\mu _ { C }$ to $( x _ { s } , \infty )$ , and the mass removed from $[ 0 , x _ { s } ]$ is added to the atom at zero. Hence, since the monotone map is optimal for the quadratic cost on the real line (Santambrogio, 2015, Section 2.2), the optimal transport from $\mu _ { C }$ to $\mu _ { s }$ is given by $\Theta _ { s } ( x ) = 0$ for $x \leq x _ { s }$ and $\Theta _ { s } ( x ) = x$ for x $> x _ { s }$ . Therefore:

$$
W _ { 2 } ^ { 2 } ( \mu _ { s } , \mu _ { C } ) = \kappa ^ { * } \int _ { 0 } ^ { x _ { s } } x ^ { 2 } \phi _ { C } ( x ) \mathrm { d } x .
$$

## D.6 Loss Function

We now prove Theorem 13, which computes the limit of the normalized squared distance:

$$
{ \frac { 1 } { \operatorname* { m i n } ( p , d ) } }  \\{ | \mathbf { Z } ( t ) - \mathbf { C } | \} _ { F } ^ { 2 } } .
$$

By equation (4), this quantity is proportional to the excess empirical loss. With our choices of scalings, the loss diverges as $d \to \infty$ , and to obtain a quantity of order one we need to rescale it by a factor proportional to d. In the proof it is convenient to first use the d-normalized distance

$$
{ \mathcal { D } } _ { d } ( t ) = { \frac { 1 } { d } } { \left\| { \mathbf { Z } } ( t ) - { \mathbf { C } } \right\| } _ { F } ^ { 2 } .
$$

Since min $( p , d ) / d  \beta$ , the normalization in Theorem 13 is recovered by multiplying the limit of $\mathcal { D } _ { d } ( t )$ by $\beta ^ { - 1 }$

## D.6.1 High-Dimensional Limit for the Distance

We start by proving the high-dimensional limit for the normalized distance $\mathcal { D } _ { d } ( t )$

Proposition 27. Using the notation of Theorem $^ { 4 , }$ in the high-dimensional limit, under $A s _ { - }$ sumption $\boldsymbol { \mathit { 1 } } , \mathcal { D } \boldsymbol { \mathit { d } } ( t )$ has the almost sure limit:

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { d \to \infty } \mathcal { D } _ { d } ( t ) = \int \left( \frac { \vartheta _ { t } \sqrt { r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) } } { 1 - \kappa + \vartheta _ { t } q _ { t } ( \gamma _ { x } ) } - x \right) ^ { 2 } \tilde { \mu } _ { C } ( \mathrm { d } x ) } \\ { \displaystyle \qquad + \frac { \vartheta _ { t } ( 1 - \kappa ) } { J _ { t } } \left( \int \frac { r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) \tilde { \mu } _ { C } ( \mathrm { d } x ) } { ( 1 - \kappa + \vartheta _ { t } q _ { t } ( \gamma _ { x } ) ) ^ { 2 } } \right) \left( \int \frac { r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) \tilde { \mu } _ { C } ( \mathrm { d } x ) } { ( 1 - \kappa + \vartheta _ { t } q _ { t } ( \gamma _ { x } ) ) ^ { 2 } } \right) , } \end{array}\tag{93}
$$

where $q _ { t } , r _ { t } ^ { \mathsf { L } } , r _ { t } ^ { \mathsf { R } }$ are defined in Theorem $ 4 , \vartheta _ { t }$ is the unique positive solution of the self-consistent equation

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \vartheta _ { t } q _ { t } ( \gamma _ { x } ) } ,\tag{94}
$$

and

$$
J _ { t } = \int \frac { q _ { t } ( \gamma _ { x } ) } { ( 1 - \kappa + \vartheta _ { t } q _ { t } ( \gamma _ { x } ) ) ^ { 2 } } \tilde { \mu } _ { C } ( \mathrm { d } x ) .\tag{95}
$$

Proof. Let $\mathbf { S } = ( \mathbf { C } ^ { \top } \mathbf { C } ) ^ { 1 / 2 }$ and $\Gamma _ { \rho } = ( \mathbf { S } ^ { 2 } + \rho ^ { 2 } \mathbf { I } _ { d } ) ^ { 1 / 2 }$ . From Lemma B.1, we have:

$$
\begin{array} { r } { \mathbf Z ( t ) = 2 \rho \boldsymbol \chi _ { t } ^ { U } ( \mathbf C ) \mathbf { R } \Big ( \mathbf R ^ { \top } q _ { t } ( \mathbf { r } _ { \rho } ) \mathbf R \Big ) ^ { - 1 } \mathbf { R } ^ { \top } \boldsymbol \chi _ { t } ^ { V } ( \mathbf S ) . } \end{array}
$$

Now, expanding the Frobenius squared distance in the expression of $\mathcal { D } _ { d } ( t )$ , we have the identity:

$$
\mathcal { D } _ { d } ( t ) = \frac { 4 \rho ^ { 2 } } { d } \mathrm { T r } \Big ( \chi _ { t } ^ { U } ( \mathbf { C } ) ^ { \top } \chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { H } _ { t } \chi _ { t } ^ { V } ( \mathbf { S } ) ^ { 2 } \mathbf { H } _ { t } \Big ) - \frac { 4 \rho } { d } \mathrm { T r } \Big ( \mathbf { C } ^ { \top } \chi _ { t } ^ { U } ( \mathbf { C } ) \mathbf { H } _ { t } \chi _ { t } ^ { V } ( \mathbf { S } ) \Big ) + \frac { 1 } { d } \mathrm { T r } \big ( \mathbf { C } ^ { \top } \mathbf { C } \big ) ,\tag{96}
$$

where

$$
\mathbf { H } _ { t } = \mathbf { R } { \Big ( } \mathbf { R } ^ { \top } q _ { t } ( \Gamma _ { \rho } ) \mathbf { R } { \Big ) } ^ { - 1 } \mathbf { R } ^ { \top } .
$$

Using Lemma C.2 and the notation of Theorem 4, we have

$$
\chi _ { t } ^ { U } ( { \mathbf { C } } ) ^ { \top } \chi _ { t } ^ { U } ( { \mathbf { C } } ) = \chi _ { t } ^ { U } ( { \mathbf { S } } ) ^ { 2 } = \frac { 1 } { 2 \rho } r _ { t } ^ { \bot } ( { \mathbf { \Gamma } } \Gamma _ { \rho } ) .
$$

Similarly, we have:

$$
\mathbf { C } ^ { \top } \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { C } ) = \mathbf { S } \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { S } ) = \frac { 1 } { \sqrt { 2 \rho } } \mathbf { S } r _ { t } ^ { \mathsf { L } } ( \mathbf { I } _ { \rho } ) ^ { 1 / 2 } .
$$

With these identities, we have expressed $\mathcal { D } _ { d } ( t )$ using traces of the matrix $\mathbf { H } _ { t }$ and spectral functions of $\mathbf { C } ^ { \top } \mathbf { C }$ . We now derive the limit of each term in equation (96). First of all, it is clear that:

$$
\frac { 1 } { d } \mathrm { T r } ( { \bf C } ^ { \top } { \bf C } ) \xrightarrow [ d  \infty ] { } \int x ^ { 2 } \tilde { \mu } _ { C } ( \mathrm { d } x ) .
$$

Then, we apply Theorem 22 to $\mathbf { D } = \mathbf { S }$ with $a ( x ) = q _ { t } ( \gamma _ { x } )$ and $b ( \boldsymbol { x } ) = x \sqrt { r _ { t } ^ { \mathsf { L } } ( \gamma _ { \boldsymbol { x } } ) r _ { t } ^ { \mathsf { R } } ( \gamma _ { \boldsymbol { x } } ) }$ . Its assumptions hold since $q _ { t } \geq 1$ . We obtain:

$$
\begin{array} { r l r } {  { \frac { 4 \rho } { d } \mathrm { T r } \Big ( \mathbf { C } ^ { \top } \boldsymbol { \chi } _ { t } ^ { U } ( \mathbf { C } ) \mathbf { H } _ { t } \boldsymbol { \chi } _ { t } ^ { V } ( \mathbf { S } ) \Big ) = \frac { 2 } { d } \mathrm { T r } \Big ( \mathbf { S } r _ { t } ^ { \bot } ( \mathbf { r } _ { \rho } ) ^ { 1 / 2 } r _ { t } ^ { \mathsf { R } } ( \mathbf { r } _ { \rho } ) ^ { 1 / 2 } \mathbf { H } _ { t } \Big ) } } \\ & { } & { \frac { \mathrm { a . s . } } { d \to \infty } \ 2 \vartheta _ { t } \int \frac { x \sqrt { r _ { t } ^ { \bot } ( \gamma _ { x } ) r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) } } { 1 - \kappa + \vartheta _ { t } q _ { t } ( \gamma _ { x } ) } \tilde { \mu } _ { C } ( \mathrm { d } x ) , } \end{array}
$$

where $\vartheta _ { t }$ solves the self-consistent equation (94). Finally, for the last term, we use Theorem 24:

$$
\begin{array} { r l } & { \frac { 4 \rho ^ { 2 } } { d } \mathrm { T r } \Big ( \chi _ { t } ^ { U } ( { \mathbf { C } } ) ^ { \top } \chi _ { t } ^ { U } ( { \mathbf { C } } ) \mathbf { H } _ { t } \chi _ { t } ^ { V } ( { \mathbf { S } } ) ^ { 2 } \mathbf { H } _ { t } \Big ) = \frac { 1 } { d } \mathrm { T r } \Big ( r _ { t } ^ { { \mathbf { L } } } ( { \mathbf { Y } } _ { \rho } ) \mathbf { H } _ { t } r _ { t } ^ { { \mathbf { R } } } ( { \mathbf { T } } _ { \rho } ) \mathbf { H } _ { t } \Big ) } \\ & { \xrightarrow [ d \to \infty ] { \mathrm { a . s . } } \frac { ( 1 - \kappa ) \vartheta _ { t } } { J _ { t } } \left( \int \frac { r _ { t } ^ { { \mathbf { L } } } ( \gamma _ { x } ) } { \Delta _ { t } ( \gamma _ { x } ) ^ { 2 } } \tilde { \mu } _ { C } ( \mathrm { d } x ) \right) \left( \int \frac { r _ { t } ^ { { \mathbf { R } } } ( \gamma _ { x } ) } { \Delta _ { t } ( \gamma _ { x } ) ^ { 2 } } \tilde { \mu } _ { C } ( \mathrm { d } x ) \right) + \vartheta _ { t } ^ { 2 } \int \frac { r _ { t } ^ { { \mathbf { L } } } ( \gamma _ { x } ) r _ { t } ^ { { \mathbf { R } } } ( \gamma _ { x } ) } { \Delta _ { t } ( \gamma _ { x } ) ^ { 2 } } \tilde { \mu } _ { C } ( \mathrm { d } x ) , } \end{array}
$$

where $\Delta _ { t } ( \gamma ) = 1 - \kappa + \vartheta _ { t } q _ { t } ( \gamma )$ and, consistently with Theorem 24, $J _ { t }$ is given by equation (95). Putting everything together, we obtain the expression in equation (93). □

## D.6.2 Small-Initialization Limit at Fixed Times

We now take the $\rho \to 0$ limit at fixed times. We denote $\vartheta _ { t , \rho } = \vartheta _ { t }$ to recall the dependence on $\rho .$ For $x \geq 0$ , the expressions in Theorem 4 guarantee that:

$$
q _ { t } ( \gamma _ { x } ) \underset { \rho  0 } { \longrightarrow } 1 , \qquad r _ { t } ^ { \perp } ( \gamma _ { x } ) \underset { \rho  0 } { \longrightarrow } 0 , \qquad r _ { t } ^ { \mathsf { R } } ( \gamma _ { x } ) \underset { \rho  0 } { \longrightarrow } 0 .
$$

For the convergence of $\vartheta _ { t , \rho }$ , we apply Proposition 19 directly with parameter $\rho \to 0$ and set:

$$
h _ { \rho } = \vartheta _ { t , \rho } , \qquad \quad a _ { \rho } ( x ) = q _ { t } ( \gamma _ { x } ) ,
$$

with $c _ { \rho } = 1$ . Due to the previous convergence of $q _ { t }$ , the limit function in Proposition 19 is given by:

$$
\Phi ( u ) = \frac { 1 } { 1 - \kappa + u } .
$$

The only solution to $\Phi ( u ) = 1$ is given by $u = \kappa ,$ , hence Proposition 19 guarantees that $\vartheta _ { t , \rho }$ converges to κ as $\rho \to 0$

A straightforward application of dominated convergence then guarantees that $J _ { t } \longrightarrow 1$ as $\rho \to 0$ , and therefore, taking the $\rho \to 0$ limit in equation (93), we obtain:

$$
\operatorname* { l i m } _ { \rho  0 } \operatorname* { l i m } _ { d  \infty } \mathcal { D } _ { d } ( t ) = \int x ^ { 2 } \tilde { \mu } _ { C } ( \mathrm { d } x ) .
$$

After multiplying by $\beta ^ { - 1 }$ , this gives equation (32). This is exactly the limit one would obtain when replacing $\mathbf { Z } ( t )$ by 0 in the definition of $\mathcal { D } _ { d } ( t )$

## D.6.3 Small-Initialization Limit at Log-Timescale

We now move on to the timescale $t _ { \rho } = s \log ( \rho ^ { - 1 } )$ . As previously, we let $x _ { s } = ( 2 s ) ^ { - 1 }$ and assume $x _ { s } > x _ { * }$ . Before deriving the limit of the loss, we start with an estimate in the small-initialization limit.

Lemma D.9. Let $q _ { t } , r _ { t } ^ { \mathsf { L } } , r _ { t } ^ { \mathsf { R } }$ be defined in Theorem 4. For every $\rho > 0 , t \geq 0$ , and $x \geq 0$ , with $\gamma _ { x } = \sqrt { x ^ { 2 } + \rho ^ { 2 } }$ , one has

$$
0 \leq \frac { r _ { t } ^ { \mathrm { L } } ( \gamma _ { x } ) } { q _ { t } ( \gamma _ { x } ) } \leq \gamma _ { x } - \rho , \qquad 0 \leq \frac { r _ { t } ^ { \mathrm { R } } ( \gamma _ { x } ) } { q _ { t } ( \gamma _ { x } ) } \leq \gamma _ { x } + \rho .\tag{97}
$$

Proof. For $\gamma > \rho ,$ the formulas in Theorem 4 give

$$
q _ { t } ( \gamma ) - \frac { r _ { t } ^ { \downarrow } ( \gamma ) } { \gamma - \rho } = 1 + \frac { \rho } { \gamma } \big ( 1 - e ^ { - 2 t \gamma } \big ) \geq 0 .
$$

The two bounds follow together with $r _ { t } ^ { \mathsf { R } } = r _ { t } ^ { \mathsf { L } } + 2 \rho q _ { t }$ . For $\gamma = \rho$ , that is $x = 0$ , we have $q _ { t } ( \rho ) = e ^ { 2 t \rho } , r _ { t } ^ { \mathsf { L } } ( \rho ) = 0$ and $r _ { t } ^ { \mathsf { R } } ( \rho ) = 2 \rho e ^ { 2 t \bar { \rho } }$ , thus the bounds also hold. □

We also recall Lemma D.8, which derives the limit of $q _ { t } , r _ { t } ^ { \mathsf { L } } , r _ { t } ^ { \mathsf { R } }$ at $t _ { \rho }$ when $\rho \to 0$ . Additionally, throughout the proof, we denote $\vartheta _ { \rho } = \vartheta _ { t _ { \rho } }$ to simplify notation.

Step 1. Asymptotic of the self-consistent function. Again, we apply Proposition 19 directly with parameter $\rho \to 0$ and:

$$
h _ { \rho } = \vartheta _ { \rho } , \qquad a _ { \rho } ( x ) = q _ { t _ { \rho } } ( \gamma _ { x } ) ,
$$

as well as $c _ { \rho } = 1$ . Then, Lemma D.8 shows that:

$$
a _ { \rho } ( x ) \xrightarrow [ \rho  0 ] { } \{ \begin{array} { l l } { 1 , } & { x < x _ { s } , } \\ { + \infty , } & { x > x _ { s } . } \end{array} 
$$

Hence the limit function in Proposition 19 is given by:

$$
\Phi ( u ) = \frac { 1 - T _ { C } ( x _ { s } ) } { 1 - \kappa + u } .
$$

Since $x _ { s } > x _ { * }$ , we have $T _ { C } ( x _ { s } ) < \kappa ,$ so that $\Phi ( u ) = 1$ has a unique solution on $( 0 , \infty )$ . By 2. (i) in Proposition 19 and the definition of $T _ { C }$ , we then have the convergence:

$$
\vartheta _ { \rho } \xrightarrow [ \rho  0 ] { } \kappa - \beta \kappa ^ { * } \int _ { x _ { s } } ^ { \infty } \phi _ { C } ( x ) \mathrm { d } x > 0 .
$$

Step 2. Asymptotics of the integrals. Let us start with the asymptotic behavior of $J _ { t _ { \rho } }$ Since $\vartheta _ { \rho }$ is bounded away from zero, the function

$$
x \mapsto \frac { q _ { t _ { \rho } } ( \gamma _ { x } ) } { ( 1 - \kappa + \vartheta _ { \rho } q _ { t _ { \rho } } ( \gamma _ { x } ) ) ^ { 2 } }
$$

is uniformly bounded. By Lemma D.8 and the preceding limit $\vartheta _ { \rho }  \kappa - T _ { C } ( x _ { s } )$ , it converges pointwise to $( 1 - T _ { C } ( x _ { s } ) ) ^ { - 2 }$ for $x < x _ { s }$ and to zero for $x > x _ { s }$ . Since $\tilde { \mu } _ { C }$ has no atom at $x _ { s }$ dominated convergence in equation (95) gives

$$
J _ { t _ { \rho } } \xrightarrow [ \rho  0 ] { } \frac { 1 } { 1 - T _ { C } ( x _ { s } ) } .
$$

Let us now consider the distance in equation (93). Starting with the second term, we write:

$$
\frac { r _ { t _ { \rho } } ^ { \mathsf { a } } ( \gamma _ { x } ) } { ( 1 - \kappa + \vartheta _ { \rho } q _ { t _ { \rho } } ( \gamma _ { x } ) ) ^ { 2 } } = \frac { r _ { t _ { \rho } } ^ { \mathsf { a } } ( \gamma _ { x } ) } { q _ { t _ { \rho } } ( \gamma _ { x } ) } \frac { q _ { t _ { \rho } } ( \gamma _ { x } ) } { ( 1 - \kappa + \vartheta _ { \rho } q _ { t _ { \rho } } ( \gamma _ { x } ) ) ^ { 2 } } .
$$

The bounds in equation (97) and the pointwise convergence obtained in Lemma D.8 allow us to apply dominated convergence, so that for $\mathsf { a } = \mathsf { L } , \mathsf { R }$

$$
\int \frac { r _ { t _ { \rho } } ^ { \mathsf { a } } ( \gamma _ { x } ) } { ( 1 - \kappa + \vartheta _ { \rho } q _ { t _ { \rho } } ( \gamma _ { x } ) ) ^ { 2 } } \tilde { \mu } _ { C } ( \mathrm { d } x ) \xrightarrow [ \rho  0 ] { } 0 .
$$

Therefore, the second term in equation (93) converges to zero.

Let us now consider the first term. We have the pointwise convergence as a consequence of Lemma D.8

$$
\frac { \vartheta _ { \rho } \sqrt { r _ { t _ { \rho } } ^ { \mathsf { L } } ( \gamma _ { x } ) r _ { t _ { \rho } } ^ { \mathsf { R } } ( \gamma _ { x } ) } } { 1 - \kappa + \vartheta _ { \rho } q _ { t _ { \rho } } ( \gamma _ { x } ) } \xrightarrow [ \rho  0 ] { } \{ 0 , \quad x < x _ { s } , 
$$

Additionally, Lemma D.9 guarantees that:

$$
\frac { \vartheta _ { \rho } \sqrt { r _ { t _ { \rho } } ^ { \mathrm { L } } ( \gamma _ { x } ) r _ { t _ { \rho } } ^ { \mathrm { R } } ( \gamma _ { x } ) } } { 1 - \kappa + \vartheta _ { \rho } q _ { t _ { \rho } } ( \gamma _ { x } ) } \leq \frac { \sqrt { r _ { t _ { \rho } } ^ { \mathrm { L } } ( \gamma _ { x } ) r _ { t _ { \rho } } ^ { \mathrm { R } } ( \gamma _ { x } ) } } { q _ { t _ { \rho } } ( \gamma _ { x } ) } \leq x .
$$

Then, dominated convergence gives:

$$
\operatorname* { l i m } _ { \rho \to 0 } \operatorname* { l i m } _ { d \to \infty } \mathcal { D } _ { d } ( t _ { \rho } ) = \int _ { 0 } ^ { x _ { s } } x ^ { 2 } \tilde { \mu } _ { C } ( \mathrm { d } x ) = \beta \kappa ^ { * } \int _ { x _ { 0 } } ^ { x _ { s } } x ^ { 2 } \phi _ { C } ( x ) \mathrm { d } x .
$$

After multiplying by $\beta ^ { - 1 }$ , this is precisely equation (33).

## D.6.4 Comparison with the Wasserstein Distance

We now explain why the following two quantities have the same limit on the timescale $t =$ $\mathrm { _ { \mathrm { 3 } } } \log ( \rho ^ { - 1 } )$ ), after taking the high-dimensional and small-initialization limits:

$$
\frac { 1 } { \operatorname* { m i n } ( p , d ) } \| \mathbf { Z } ( t ) - \mathbf { C } \| _ { F } ^ { 2 } , \qquad \frac { 1 } { \operatorname* { m i n } ( p , d ) } \sum _ { i = 1 } ^ { \operatorname* { m i n } ( p , d ) } \big ( \sigma _ { i } ( t ) - \sigma _ { i } \big ) ^ { 2 } ,\tag{98}
$$

where $\sigma _ { 1 } ( t ) , \ldots , \sigma _ { \operatorname* { m i n } ( p , d ) } ( t )$ and $\sigma _ { 1 } , \ldots , \sigma _ { \operatorname* { m i n } ( p , d ) }$ are the ordered singular values of $\mathbf { Z } ( t )$ and C. Proposition 28 (von Neumann’s trace inequality). Let $\mathbf { A } , \mathbf { B } \in \mathbb { R } ^ { p \times d }$ , and denote by $\pmb { \sigma } _ { A } , \pmb { \sigma } _ { B } \in \mathbb { R } ^ { \operatorname* { m i n } ( p , d ) }$ the ordered singular values of A, B. Then:

$$
\operatorname { T r } ( \mathbf { A } ^ { \top } \mathbf { B } ) \leq \sigma _ { A } ^ { \top } \pmb { \sigma } _ { B } ,
$$

and equality holds if and only if there exist $\mathbf { U } \in O _ { p } ( \mathbb { R } ) , \mathbf { V } \in O _ { d } ( \mathbb { R } )$ such that:

$$
\mathbf { A } = \mathbf { U } \mathrm { d i a g } ( \pmb { \sigma } _ { A } ) \mathbf { V } ^ { \top } , \qquad \mathbf { B } = \mathbf { U } \mathrm { d i a g } ( \pmb { \sigma } _ { B } ) \mathbf { V } ^ { \top } .
$$

A proof of this fact can be found in Lewis (1995, Theorem 2.1). The notation diag is understood as mapping $\pmb { \sigma } \in \mathbb { R } ^ { \operatorname* { m i n } ( p , d ) }$ into a rectangular diagonal matrix in $\mathbb { R } ^ { p \times d }$ with diagonal entries $\sigma _ { 1 } , \ldots , \sigma _ { \operatorname* { m i n } ( p , d ) }$ . Consequently, the equation:

$$
\left\| \mathbf { A } - \mathbf { B } \right\| _ { F } ^ { 2 } = \left\| \pmb { \sigma } _ { A } - \pmb { \sigma } _ { B } \right\| _ { 2 } ^ { 2 }
$$

is exactly the equality case in Proposition 28. The second quantity in equation (98) is the squared Wasserstein distance between the empirical singular-value distributions: the optimal transport matches the singular values in order. Thus, when the corresponding singular vectors are aligned, the Frobenius error equals this transport cost. In our setting, the equality of (31) and (33) shows that the contribution of misalignment vanishes asymptotically.

## E Numerical Simulations

This appendix describes the numerical methods used for the figures in the main text. The parameters specific to each experiment are given in the corresponding figure caption.

Finite-dimensional simulations. We simulate the dynamics (2) on $\mathbf { U } ( t ) , \mathbf { V } ( t )$ directly. For a given $d ,$ we fix the values of the parameters $\tau , \kappa , \kappa ^ { * }$ and set $p = \lfloor \tau d \rceil , m = \lfloor \kappa d \rceil$ , and $m ^ { * } = \lfloor \kappa ^ { * } \operatorname* { m i n } ( p , d ) \rceil$ . The non-zero singular values of C are sampled from the density $\phi _ { C }$ , and we take the corresponding singular vectors to be the canonical basis. This choice is without loss of generality for the observables considered here, by rotational invariance of the initialization.

The matrix $\mathbf { R } \in \mathrm { S t } ( d , m )$ is obtained by applying a QR decomposition to a standard Gaussian matrix, and we initialize

$$
{ \bf { U } } _ { 0 } = { \bf { 0 } } , \qquad { \bf { V } } _ { 0 } = \sqrt { 2 \rho } { \bf { R } } .
$$

We use explicit Euler steps with stepsize η:

$$
\begin{array} { r l } & { \mathbf { U } _ { n + 1 } = \mathbf { U } _ { n } + \eta \big ( \mathbf { C } - \mathbf { U } _ { n } \mathbf { V } _ { n } ^ { \top } \big ) \mathbf { V } _ { n } , } \\ & { \mathbf { V } _ { n + 1 } = \mathbf { V } _ { n } + \eta \big ( \mathbf { C } ^ { \top } - \mathbf { V } _ { n } \mathbf { U } _ { n } ^ { \top } \big ) \mathbf { U } _ { n } , } \end{array}
$$

and $t _ { n } = n \eta$ . These steps approximate the gradient flow (2) with an error of order $\eta .$ . In the linearized regime, a mode associated with a singular value σ grows like $( 1 + \eta \sigma ) ^ { t / \eta }$ instead of $e ^ { \sigma t }$ which changes its growth rate by a relative amount of order ησ. For $\eta = 1 0 ^ { - 3 }$ , this shifts the logarithmic timescales of Figures 7 and 8 by at most about 0.1%. As a check, rerunning two repetitions of Figure 8 with stepsize $\eta / 2$ changes the loss and the spectral mass below $x _ { s }$ by at most $2 \cdot 1 0 ^ { - 3 }$

When several values of $\rho$ are compared, we use the same realization of C and R. The curves are averaged over independent realizations, with the number of repetitions stated in each caption. Losses, singular values, and projector distances are computed from singular value decompositions at the saved times.

Self-consistent equations. We next describe how the high-dimensional curves are evaluated. Integrals with respect to the continuous part of $\tilde { \mu } _ { C }$ are approximated by Gauss–Legendre quadrature on the support of $\phi _ { C }$ and the atom at zero is added separately. For a spectral parameter $w = E + \mathrm { i } \varepsilon$ , the scalar self-consistent equation has the form

$$
1 = \int \frac { \tilde { \mu } _ { C } ( \mathrm { d } x ) } { 1 - \kappa + \kappa g _ { t } ^ { \mathsf { a } } ( w ) \big ( r _ { t } ^ { \mathsf { a } } ( \gamma _ { x } ) - w q _ { t } ( \gamma _ { x } ) \big ) } ,\tag{99}
$$

with $\mathsf { a } = \mathsf { L } , \mathsf { R }$ . After quadrature, equation (99) is a scalar complex equation. We solve it by damped Newton iterations, using the explicit derivative of its right-hand side. To select the correct branch, we first continue the positive solution from the negative real axis to the first value $E + \mathrm { i } \varepsilon$ . We then move along a grid of values of E and use the previous solution as the next initial value. We keep the solution whose imaginary part has the same sign as that of $w ,$ , and monitor the residual of the self-consistent equation.

Once $g _ { t } ^ { \mathsf { a } } ( w )$ is known, we evaluate the Stieltjes transforms in equations (12), (13). The regular part of the density is approximated through the Stieltjes–Perron formula (Theorem 14):

$$
\phi _ { t } ^ { \bullet } ( E ) \simeq \frac { 1 } { \pi } \mathbb { S } \mathfrak { m } _ { t } ^ { \bullet } ( E + \mathrm { i } \varepsilon ) ,
$$

for $\bullet = U , V$ . The parameter ε is kept small and fixed on each plot. Before comparing with histograms of non-zero eigenvalues, we subtract the known atoms at zero and, for $\mathbf { V V } ^ { \top }$ , at $2 \rho .$ and retain the continuous mass. The singular-value density of $\mathbf { Z }$ is then obtained from (15) by the pushforward $\theta _ { \rho } ( x ) = \sqrt { x ( x + 2 \rho ) }$

The scalar positive equations entering the range-projector formulas are solved by a bracketed root method, using the logarithm of the unknown to remain stable at long times. Finally, the small-initialization and logarithmic-time curves are evaluated directly from the formulas of Theorems 8, 12 and 13, using the same quadrature rule. No parameters are fitted to the finite-dimensional simulations.

Checks. We checked the numerical methods in several ways. At $d = 4 0$ , the Euler iterates stay within $1 . 2 \cdot 1 0 ^ { - 2 }$ (in relative Frobenius norm) of a fourth-order Runge–Kutta solution, which itself matches the closed form of Proposition 1 up to $1 0 ^ { - 8 }$ . Halving the stepsize $\eta$ shifts the activation times of Figure 7 by less than 0.1%. At fixed $\rho = 0 . 1$ , with the singular values of C set to the quantiles of $\phi _ { C }$ , the distance between the simulations and the formulas of Theorems 4 and 6 decreases with $d ,$ and is at most about $1 0 ^ { - 3 }$ at $d = 8 0 0$ , for $\tau \in \{ 0 . 7 , 1 , 1 . 4 \}$ Doubling the number of quadrature nodes changes the theoretical densities by less than 0.2% of their maximum. The code of the simulations, of the figures and of these checks is available at https: $: / / \mathrm { g } \mathrm { i }$ thub.com/simonmartin15/silent-alignment-linear-networks.