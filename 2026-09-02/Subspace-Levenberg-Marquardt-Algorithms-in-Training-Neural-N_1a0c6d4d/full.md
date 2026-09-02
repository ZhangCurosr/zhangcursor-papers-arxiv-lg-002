# Subspace Levenberg–Marquardt Algorithms in Training Neural Networks

M. Duc Hoang

MDHOANG@UCDAVIS.EDU

Department ofMathematics, University ofCalifornia, Davis, Davis, CA, USA

## Abstract

The Levenberg-Marquardt (LM) algorithm is a well-known second-order method for rapid convergence and strong robustness when training small- to medium-sized neural networks (NNs). However, its computational and memory costs increase significantly as the number of parameters in an NN grows. To address this limitation, subspace methods have been proposed, such as the Krylov subspace LM (KSLM) and the hybrid subspace LM (HSLM), making second-order algorithms more efficient. In this work, we evaluate the subspace Levenberg-Marquardt algorithms for regression and classification tasks in neural networks. We compare the performance of subspace LM variants with the classical LM method, as well as other popular first-order algorithms, such as stochastic gradient descent (SGD) and Adam.

## 1. Introduction

Feedforward multilayer perceptrons are widely used for supervised learning tasks such as regression and classification. Their performance depends in part on whether the network has sufficient capacity to represent the underlying structure of the problem. Classical results show that feedforward networks with sufficiently many hidden units can approximate broad classes of nonlinear functions [2, 6]. Approximation theory further shows that the network complexity required to achieve a prescribed accuracy can depend on properties of the target function, including its dimensionality and smoothness [11, 16]. Thus, more complex learning problems may require greater network capacity. In regression, this additional capacity allows the representation of richer nonlinear mappings, while in classification, increasing the number of parameters and layers can increase the capacity of the classifier [1]. However, larger networks also increase the number of trainable parameters and consequently make the optimization problem more computationally demanding.

First-order optimization methods, such as stochastic gradient descent (SGD) and Adam [7], are widely used for training neural networks due to their low memory requirements and relatively inexpensive iterations. However, their performance can depend strongly on hyperparameter choices, including the learning rate and momentum parameters. In addition, they may require a large number of iterations to achieve high accuracy, particularly when the optimization landscape is poorly conditioned. In contrast, second-order methods incorporate curvature information to determine more effective search directions and can achieve faster local convergence and greater robustness to problem scaling. For nonlinear least-squares training, the Levenberg–Marquardt (LM) method offers a practical alternative by using the Gauss–Newton approximation with adaptive damping and is particularly effective for small- to medium-sized networks [4]. However, the computational and memory costs associated with forming and solving the full LM system make it impractical as network size increases [15].

One approach for reducing these costs is to restrict the LM step to a low-dimensional subspace rather than solving the full-dimensional problem. Subspace LM methods retain important curvature information while reducing the size of the linear system that must be constructed and solved at each iteration. Subspace LM methods include Krylov subspace LM (KSLM) [12], which constructs a search space using Krylov directions. More recently, Hoang et al. introduced adaptive hybrid subspace LM (HSLM) [5], which combines multiple informative directions to improve the quality of the reduced-space approximation. These approaches provide a potential compromise between the inexpensive iterations of first-order methods and the rapid convergence of full LM.

In this work, we investigate the use of subspace LM algorithms for both regression and classification tasks in neural networks. We compare classical LM with its subspace variants, KSLM and HSLM, and evaluate their performance against widely used first-order optimizers such as SGD and Adam. Our study focuses on the trade-offs among convergence behavior, computational cost, and predictive performance to assess whether subspace formulations can extend the practical advantages of LM to larger neural-network optimization problems.

## 2. Levenberg–Marquardt and Subspace Methods for Neural Networks

We consider the standard Levenberg-Marquardt algorithm and its subspace KSLM and HSLM variants.

## 2.1. Levenberg–Marquardt Algorithm

Nonlinear least-squares problems are commonly formulated as

$$
\operatorname* { m i n } _ { x \in \mathbf { R } ^ { n } } \mathbf { F } ( x ) : = \operatorname* { m i n } _ { x \in \mathbf { R } ^ { n } } \frac { 1 } { 2 } \| \mathbf { r } ( x ) \| ^ { 2 } ,
$$

where $\mathbf { r } : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ is the residual vector. Let $\mathbf { J } \in \mathbb { R } ^ { m \times n }$ denote the Jacobian, then the Gauss– Newton method $[ 3 ]$ solves nonlinear least-squares problems at iteration k by linearizing the residual as $\mathbf { \nabla } \cdot ( x _ { k } + s _ { k } ) \approx \mathbf { r } ( x _ { k } ) + \mathbf { J } _ { k } s _ { k }$ , resulting in a linear least-squares subproblem

$$
\left( \mathbf { J } _ { k } ^ { \top } \mathbf { J } _ { k } \right) s _ { k } = - \mathbf { J } _ { k } ^ { \top } \mathbf { r } ( x _ { k } )
$$

where $s _ { k }$ is the update direction and $\mathbf { J } ^ { \top } \mathbf { J } \approx \mathbf { H }$ is the Gauss-Newton approximation to the Hessian. The Levenberg–Marquardt method can be viewed as a damped Gauss–Newton method for nonlinear least-squares problems [8, 10, 13]. The LM step $s _ { k }$ is obtained by solving

$$
\left( \mathbf { J } _ { k } ^ { \top } \mathbf { J } _ { k } + \mu _ { k } \mathbf { I } _ { n } \right) s _ { k } = - \mathbf { g } _ { k } ,\tag{1}
$$

where $\mathbf { g } _ { k } = \mathbf { J } _ { k } ^ { \top } \mathbf { r } ( x _ { k } )$ is the gradient and $\mu _ { k } > 0$ is a damping parameter adjusted at each iteration. When $\mu _ { k }$ is small, the method behaves similarly to Gauss–Newton, whereas for larger values the step approaches a scaled gradient-descent direction. This damping improves robustness when the Jacobian is ill-conditioned or the current iterate is far from a solution. See Appendix A for a detailed damping update strategy.

## 2.2. Krylov-Subspace Levenberg–Marquardt Algorithm

For large-scale nonlinear least-squares problems, solving the full LM system may become computationally expensive because forming and factorizing $\mathbf { J } _ { k } ^ { \top } \mathbf { J } _ { k } + \mu _ { k } \mathbf { I } _ { n }$ can require substantial memory and work. In a Krylov-subspace LM approach, the step direction $s _ { k }$ is restricted to a subspace of the form

$$
\begin{array} { r } { \mathcal { K } _ { p } ( { \bf B } _ { k } , { \bf g } _ { k } ) = \mathrm { s p a n } \{ { \bf g } _ { k } , { \bf B } _ { k } { \bf g } _ { k } , { \bf B } _ { k } ^ { 2 } { \bf g } _ { k } , \ldots , { \bf B } _ { k } ^ { p - 1 } { \bf g } _ { k } \} , \qquad { \bf B } _ { k } = { \bf J } _ { k } ^ { \top } { \bf J } _ { k } + \mu _ { k } { \bf I } _ { n } . } \end{array}
$$

Let $\mathbf { V } _ { k } \in \mathbb { R } ^ { n \times p }$ be a matrix whose columns form a basis of $\mathcal { K } _ { p } ( \mathbf { B } _ { k } , \mathbf { g } _ { k } )$ . The step is then written as $s _ { k } = \mathbf { V } _ { k } y _ { k }$ , and $y _ { k } \in \mathbb { R } ^ { p }$ is obtained from the reduced system

$$
\left( \mathbf { V } _ { k } ^ { \top } \mathbf { J } _ { k } ^ { \top } \mathbf { J } _ { k } \mathbf { V } _ { k } + \mu _ { k } \mathbf { I } _ { p } \right) y _ { k } = - \mathbf { V } _ { k } ^ { \top } \mathbf { g } _ { k } .
$$

Thus, instead of solving the full n-dimensional LM system, one solves a much smaller projected problem in the Krylov subspace. This approach can substantially reduce memory and computational requirements while retaining the LM damping mechanism, making it attractive for large neuralnetwork problems where the full LM system is too expensive to solve directly [12]. An additional advantage is that the Krylov subspace is invariant under the scalar shift introduced by the damping parameter [9]. For fixed $\mathbf { J } _ { k }$ and $\mathbf { g } _ { k }$

$$
\begin{array} { r } { \mathcal { K } _ { p } \left( \mathbf { J } _ { k } ^ { \top } \mathbf { J } _ { k } + \mu _ { k } \mathbf { I } _ { n } , \mathbf { g } _ { k } \right) = \mathcal { K } _ { p } \left( \mathbf { J } _ { k } ^ { \top } \mathbf { J } _ { k } , \mathbf { g } _ { k } \right) . } \end{array}\tag{2}
$$

Therefore, if several damping parameters are adjusted at the same iteration, the Krylov basis does not need to be recomputed. Thus, the basis can be reused while the $\mu _ { k }$ is adjusted, further reducing the cost of repeated LM step computations.

## 2.3. Hybrid Subspace Levenberg-Marquardt Algorithm

HSLM adaptively constructs a low-dimensional subspace from multiple sources of gradient and curvature information. Specifically, this subspace combines multiple sources of basis information, including

• the normalized gradient, which provides first-order descent information;

• recent optimization steps, which encode local optimization history;

• Krylov/Lanczos vectors, which approximate dominant curvature directions; and

• randomized Gauss-Newton-vector probes $( \mathbf { J } _ { k } ^ { \top } \mathbf { J } _ { k } ) w$ where $w \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { n } )$ , which broaden spectral coverage and improve robustness when the local curvature is noisy or heterogeneous.

Since the candidate pool combines complementary information, the resulting subspace may not fully capture the current descent direction. To assess its quality, a deterministic adequacy criterion is based on the fraction of gradient energy captured by the subspace

$$
\eta _ { k } ^ { \mathrm { s u b } } : = \frac { \| \mathbf V _ { k } ^ { \top } \mathbf g _ { k } \| ^ { 2 } } { \| \mathbf g _ { k } \| ^ { 2 } } \in [ 0 , 1 ] .\tag{3}
$$

The quantity $\eta _ { k } ^ { \mathrm { s u b } } \in [ 0 , 1 ]$ quantifies how much descent-relevant gradient information is captured by the current subspace, with larger values indicating greater adequacy. If $\eta _ { k } ^ { \mathrm { s u b } }$ drops below a prescribed threshold $\eta _ { \mathrm { m i n } } \in ( 0 , 1 ]$ , additional candidate directions are added and the adequacy test is repeated until the threshold is satisfied. Once the adequacy criterion is satisfied, the reduced basis is fixed and the Levenberg–Marquardt step is computed entirely within this subspace, reducing the original n-dimensional problem to a much smaller p-dimensional system.

Let $\mathbf { J } _ { k , p } = \mathbf { J } _ { k } \mathbf { V } _ { k }$ be the reduced Jacobian. Let the singular value decomposition of $\mathbf { J } _ { k , p }$ be

$$
{ \bf J } _ { k , p } = { \bf U } _ { k } \Sigma _ { k , p } { \bf Z } _ { k } ^ { \top } , \qquad { \bf \Sigma } \Sigma _ { k , p } = \mathrm { d i a g } ( \sigma _ { k , 1 } , \ldots , \sigma _ { k , p } ) .
$$

Writing the search direction as $s _ { k } = \mathbf { V } _ { k } \mathbf { Z } _ { k } y _ { k }$ , the resulting LM system can then be solved efficiently in the singular-vector coordinates as [5]

$$
\left( \boldsymbol { \Sigma } _ { k , p } ^ { 2 } + \mu _ { k } \mathbf { I } _ { p } \right) y _ { k } = - \boldsymbol { \Sigma } _ { k , p } \mathbf { U } _ { k } ^ { \top } \mathbf { r } _ { k } .
$$

This reduced formulation inherits the isotropic damping of the standard LM algorithm. However, because the singular values of the reduced Jacobian can vary substantially, treating all directions equally may lead to unnecessary damping in some directions and insufficient damping in others. To better accommodate ill-conditioning and variations in local curvature, the HSLM replace the isotropic damping matrix $\mathbf { I } _ { p }$ with a diagonal, curvature-dependent damping matrix $\mathbf { D } _ { k , p }$ . The resulting reduced system is

$$
\mathbf { B } _ { k , p } \mathbf { y } _ { k } = - \Sigma _ { k , p } \mathbf { U } _ { k } ^ { \top } \mathbf { r } _ { k } ,
$$

where

$$
\mathbf { B } _ { k , p } = \Sigma _ { k , p } ^ { 2 } + \mu _ { k } \mathbf { D } _ { k , p } , \qquad \mathbf { D } _ { k , p } = \mathrm { d i a g } ( d _ { k , 1 } , \dots , d _ { k , p } ) , \qquad d _ { k , i } = \operatorname* { m a x } ( \sigma _ { k , i } ^ { 2 } , \delta ) .
$$

The threshold $\delta > 0$ keeps the damping coefficient strictly positive, while singular value dependence provides stronger damping in directions with greater curvature.

Unlike the classical LM algorithm and KSLM, HSLM uses Armijo backtracking for step acceptance, with the damping parameter adjusted according to whether the step is accepted or rejected. The HSLM method generalizes subspace optimization by combining multiple sources of information within an adaptive low-dimensional space, while the deterministic adequacy strategy ensures that sufficient descent-relevant information is retained. This allows the method to reduce computational cost while preserving the main theoretical advantages of LM-type methods [5].

## 3. Evaluation and Comparison

We evaluate the optimization methods on two neural-network problems: a nonlinear regression problem and the 13-bit parity classification problem. In both experiments, we compare SGD, Adam, the classical LM method, KSLM, and HSLM. Each experiment is repeated over 100 independent trials using common initialization across methods. See Appendix A for more implementation details.

## 3.1. Damped Oscillation Regression Problem

All five methods successfully approximate the damped-oscillation function, $f ( x ) = 2 e ^ { - x ^ { 2 } } \cos ( 2 \pi x )$ (Figure 1A), but differ significantly in convergence speed and computational cost (Figure 1B and

Table 1). LM and KSLM converge in relatively few iterations, averaging $2 2 . 2 \pm 6 . 8$ , but require higher execution times of $8 3 . 0 \pm 2 6 . 2 \mathrm { ~ s ~ }$ and $1 3 2 . 8 \pm 4 0 . 5 \ \mathrm { s } .$ , respectively. While LM computes updates in the full parameter space, KSLM and HSLM restrict the step to lower-dimensional subspaces. KSLM retains fast LM-like convergence but incurs additional cost from constructing and expanding the Krylov subspace. HSLM achieves a better balance, requiring only $1 9 . 3 \pm 5 . 2$ iterations and $2 5 . 9 \pm 7 . 4 \mathrm { ~ s ~ }$ on average. In contrast, SGD and Adam require substantially more epochs before termination, averaging $1 5 5 . 0 \pm 5 5 . 1$ and $1 0 2 . 1 \pm 4 4 . 3$ , respectively.

![](images/f583f914f28f4a02b0e4a22e9abc12cab1e86e32ae6bd51c8e58ece1d7e9f509.jpg)

![](images/d4c229ba3d085bad316d058e896842646970bdc757e086c13881488a1c66f58f.jpg)  
Figure 1: Comparison of optimization methods on Damped Oscillation regression problem. A. Approximation of the target function obtained using SGD, Adam, LM, Krylov LM, and HSLM. B. Training mean squared error (MSE) over 100 epochs/iterations. See Appendix B1 for the complete training performance.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LM</td><td rowspan=1 colspan=1>KSLM</td><td rowspan=1 colspan=1>HSLM</td><td rowspan=1 colspan=1>SGD</td><td rowspan=1 colspan=1>Adam</td></tr><tr><td rowspan=1 colspan=1>Execution Time (s)</td><td rowspan=1 colspan=1> $8 3 . 0 \pm 2 6 . 2$ </td><td rowspan=1 colspan=1> $1 3 2 . 8 \pm 4 0 . 5$ </td><td rowspan=1 colspan=1> $2 5 . 9 \pm 7 . 4$ </td><td rowspan=1 colspan=1> $\overline { { 3 0 . 9 \pm 1 0 . 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 4 . 3 \pm 1 0 . 6 } }$ </td></tr><tr><td rowspan=1 colspan=1>Iteration/Epoch</td><td rowspan=1 colspan=1> $2 2 . 2 \pm 6 . 8$ </td><td rowspan=1 colspan=1> $2 2 . 2 \pm 6 . 8$ </td><td rowspan=1 colspan=1> $\overline { { 1 9 . 3 \pm 5 . 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 5 5 . 0 \pm 5 5 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 2 . 1 \pm 4 4 . 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>Training MSE $\overline { { \le \sigma _ { \mathrm { n o i s e } } ^ { 2 } } }$ (%)</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>0</td></tr></table>

Table 1: Summary of training performance for neural-network algorithms across 100 trials $( \mathrm { m e a n } \pm \mathrm { S D } )$

Moreover, all three LM-based methods achieved a 100% success rate in reaching the prescribed noise-variance threshold, $\sigma _ { \mathrm { n o i s e } } ^ { 2 } \approx 0 . 0 0 1 6$ , compared with 41% for SGD and 0% for Adam. By exploiting curvature information and adaptive damping, LM-based methods make stable, well-scaled updates near the minimum and more reliably approach the noise floor. In contrast, first-order methods may progress more slowly near the minimum. Among the LM-based methods, HSLM retains this fast and reliable convergence while substantially reducing the computational cost of classical LM and KSLM.

## 3.2. 13-Parity Classification Problem

For the 13-bit parity problem, all methods achieved high training and validation accuracy but differed substantially in convergence behavior (Figure 2 and Table 2). The LM-based methods reduced training and validation MSE faster than SGD and Adam (Figure 2A–B). HSLM required the fewest outer iterations among the LM-based methods, $3 7 . 0 \pm 2 7 . 2$ iterations on average, while SGD and Adam required substantially more epochs, $1 0 6 1 . 7 \pm 5 3 0 . 1$ and $3 3 3 . 3 \pm 4 5 8 . 5$

![](images/dc05c2881dea05088d2bcc7b0d5e725579465efcdf7b7796c5960241fe6ab284.jpg)

![](images/7b5ad4f47aa8161e6dd7996dbee71060842a7578cbf633fbb1042464fe2a2701.jpg)

![](images/c916d7e4f9ed798dcc39b8c324a6ec222b959a9c6ef2b4880f718a961caa6641.jpg)

![](images/dc2905f7f59b3916acfee2083b50241ff4907ddc63c94a2228740cfb4eae4b4a.jpg)

Figure 2: Training and validation performance of the optimization methods on the 13-parity classification problem over 100 epochs/iterations. A. Training MSE. B. Validation MSE. C. Training accuracy. D. Validation accuracy. See Appendix B2 for the complete training performance
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LM</td><td rowspan=1 colspan=1>KSLM</td><td rowspan=1 colspan=1>HSLM</td><td rowspan=1 colspan=1>SGD</td><td rowspan=1 colspan=1>Adam</td></tr><tr><td rowspan=1 colspan=1>Execution Time (s)</td><td rowspan=1 colspan=1> $5 . 8 \pm 4 . 1$ </td><td rowspan=1 colspan=1> $6 . 3 \pm 4 . 0$ </td><td rowspan=1 colspan=1> $3 . 7 \pm 3 . 8$ </td><td rowspan=1 colspan=1> $1 3 . 2 \pm 6 . 5$ </td><td rowspan=1 colspan=1> $5 . 7 \pm 7 . 8$ </td></tr><tr><td rowspan=1 colspan=1>Iteration/Epoch</td><td rowspan=1 colspan=1> $6 7 . 8 \pm 4 7 . 8$ </td><td rowspan=1 colspan=1> $\overline { { 7 1 . 1 \pm 4 6 . 3 } }$ </td><td rowspan=1 colspan=1> $3 7 . 0 \pm 2 7 . 2$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 6 1 . 7 \pm 5 3 0 . 1 } }$ </td><td rowspan=1 colspan=1> $3 3 3 . 3 \pm 4 5 8 . 5$ </td></tr><tr><td rowspan=1 colspan=1>Validation accuracy (%)</td><td rowspan=1 colspan=1> $\overline { { 9 7 . 7 \pm 7 . 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 9 7 . 9 \pm 5 . 1 } }$ </td><td rowspan=1 colspan=1> $9 9 . 6 \pm 0 . 3$ </td><td rowspan=1 colspan=1> $\overline { { 9 9 . 7 \pm 0 . 3 } }$ </td><td rowspan=1 colspan=1> $\overline { { 9 9 . 8 \pm 0 . 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>Convergence rate (%)</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>100</td></tr></table>

Table 2: Comparison of the training performance of the optimization methods over 100 trials $( \mathrm { m e a n } \pm \mathrm { S D } )$ . Convergence is defined as achieving a training MSE below 0.01 and a training accuracy above 0.99.

HSLM and Adam achieved a 100% convergence rate, compared with 92%, 89%, and 92% for LM, KSLM, and SGD, respectively. Adam obtained the highest validation accuracy $( 9 9 . 8 \pm 0 . 1 \% )$ , while HSLM achieved a comparable $9 9 . 6 \pm 0 . 3 \%$ with far fewer iterations. HSLM also remains effective with limited training data (see Appendix C), providing a strong balance of convergence speed, computational cost, and predictive accuracy.

## 4. Conclusion and Future Work

The numerical results show that the HSLM method achieves a favorable balance of convergence speed, computational efficiency, and solution accuracy for both regression and classification problems, suggesting that reduced-subspace approaches can preserve the fast convergence of LM-type methods while substantially reducing computational cost for neural-network training. However, larger datasets may increase the cost of forming and solving the associated subproblems, limiting scalability. Future work may explore broader classes of subspace methods that incorporate minibatch and matrix-free implementations, and investigate how different choices of subspace basis affect computational efficiency and convergence behavior.

## References

[1] P. L. Bartlett, N. Harvey, C. Liaw, and A. Mehrabian. Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research, 20(63):1–17, 2019.

[2] G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2:303–314, 1989. doi: 10.1007/BF02551274.

[3] J. E. Dennis, Jr. and Robert B. Schnabel. Numerical Methodsfor Unconstrained Optimization and Nonlinear Equations, volume 16 of Classics in Applied Mathematics. SIAM, Philadelphia, 1996. doi: 10.1137/1.9781611971200.

[4] M. T. Hagan and M. B. Menhaj. Training feedforward networks with the marquardt algorithm. IEEE Transactions on Neural Networks, 5(6):989–993, 1994. doi: 10.1109/72.329697.

[5] M. Duc Hoang and Timothy J. Lewis. Adaptive hybrid subspace levenberg–marquardt algorithm with adequacy monitor for large-scale least squares problems. arXiv preprint arXiv:2608.25524, 2026. doi: 10.48550/arXiv.2608.25524.

[6] K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989. doi: 10.1016/0893-6080(89)90020-8.

[7] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[8] K. Levenberg. A method for the solution of certain non-linear problems in least squares. Quarterly ofApplied Mathematics, 2(2):164–168, 1944. doi: 10.1090/QAM/10666.

[9] Y. Lin, D. O’Malley, and V. V. Vesselinov. A computationally efficient parallel levenberg– marquardt algorithm for highly parameterized inverse model analyses. Water Resources Research, 52(9):6948–6977, 2016. doi: 10.1002/2016WR019028.

[10] D. W. Marquardt. An algorithm for least-squares estimation of nonlinear parameters. Journal of the Society for Industrial and Applied Mathematics, 11(2):431–441, 1963. doi: 10.1137/ 0111030.

[11] H. N. Mhaskar. Neural networks for optimal approximation of smooth and analytic functions. Neural Computation, 8(1):164–177, 1996. doi: 10.1162/neco.1996.8.1.164.

[12] Eiji Mizutani and James W. Demmel. On iterative krylov-dogleg trust-region steps for solving neural networks nonlinear least squares problems. In Todd K. Leen, Thomas G. Dietterich, and Volker Tresp, editors, Advances in Neural Information Processing Systems 13, pages 605–611. MIT Press, 2001.

[13] Jorge J. More. The levenberg–marquardt algorithm: Implementation and theory. In G. A.´ Watson, editor, Numerical Analysis, volume 630 of Lecture Notes in Mathematics, pages 105– 116. Springer, Berlin, Heidelberg, 1978. doi: 10.1007/BFb0067700.

[14] Mark K. Transtrum, Benjamin B. Machta, and James P. Sethna. Geometry of nonlinear least squares with applications to sloppy models and optimization. Physical Review E, 83 (3):036701, 2011. doi: 10.1103/PhysRevE.83.036701.

[15] B. M. Wilamowski and H. Yu. Improved computation for levenberg–marquardt training. IEEE Transactions on Neural Networks, 21(6):930–937, 2010. doi: 10.1109/TNN.2010.2045657.

[16] D. Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks, 94:103–114, 2017. doi: 10.1016/j.neunet.2017.07.002.

## Appendix A. Experimental Setup

We evaluate five optimization methods—SGD, Adam, classical LM, Krylov-subspace LM, and HSLM—on nonlinear regression and 13-bit parity classification. Each experiment is repeated over 100 independent trials, with parameters initialized as $\theta _ { 0 } \sim \mathcal { U } [ - 1 , 1 ]$ . Within each trial, all methods use the same initial parameter vector to ensure a fair comparison. All experiments were implemented in Python using NumPy within Jupyter Notebook and executed on a Dell 14 Plus laptop equipped with an Intel Core Ultra 9 288V processor and 32 GB of RAM.

## A.1. Regression problem

For the regression task, we approximate the function

$$
f ( x ) = 2 e ^ { - x ^ { 2 } } \cos ( 2 \pi x ) , \qquad x \in [ - 2 , 2 ] .
$$

We generate 40,000 randomly sampled input points and corrupt the corresponding function values with additive Gaussian noise with a standard deviation equal to 5% of the standard deviation of the target value, $\sigma _ { y } .$

$$
y _ { i } = f ( x _ { i } ) + \sigma _ { \mathrm { n o i s e } } \epsilon _ { i } , \qquad \epsilon _ { i } \sim \mathcal { N } ( 0 , 1 ) , \qquad \sigma _ { \mathrm { n o i s e } } = 0 . 0 5 \sigma _ { y }
$$

The regression model is a fully connected multilayer perceptron with architecture $1 - 7 0 { - } 4 0 { - } 1$ The two hidden layers use the hyperbolic tangent (tanh) activation function, while the output layer is linear. Mean squared error (MSE) is used as the loss function. A single realization of the input data and Gaussian noise is generated and then kept fixed across all optimizers and all 100 trials.

## A.2. 13-bit parity problem

For the classification experiment, we consider the complete 13-bit parity problem, consisting of all $2 ^ { 1 3 } = 8 1 9 2$ binary input patterns. Both the inputs and target labels are represented using $\{ - 1 , 1 \}$ . We use a fully connected MLP with architecture $1 3 - 2 5 - 1 0 - 1$ , with tanh activation functions in both hidden layers and the output layer. The network is trained using MSE as the loss function. We split the full parity dataset using a stratified 90/10 training–validation split, fixed across optimization methods and all 100 trials.

## A.3. First-order methods

For SGD, we use a learning rate of 0.01, momentum of 0.9, and a mini-batch size of 64. For Adam, we use a learning rate of 0.003 and a mini-batch size of 64, with the standard momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ . Both SGD and Adam are allowed a maximum of 1500 epochs.

## A.4. LM-based methods

For classical LM, KSLM, and HSLM, the maximum number of outer iterations is set to 150. The initial damping parameter is $\mu _ { 0 } = 1 0$ . In this work, the damping parameter follows the delayedgratification strategy [14]:

$$
\mu _ { k + 1 } = \left\{ { \begin{array} { l l } { \mu _ { k } / \mu _ { \mathrm { d o w n } } , } & { { \mathrm { i f ~ s u c c e s s f u l } } } \\ { \mu _ { k } \cdot \mu _ { \mathrm { u p } } , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.
$$

This allows the LM method to move more quickly toward the Gauss–Newton regime when the local model is reliable, while shifting toward the gradient-descent regime when a step is unsuccessful. Following a successful step, the damping parameter is reduced according to $\displaystyle \mu _ { k + 1 } = \mu _ { k } / 2$ whereas following an unsuccessful step it is increased according to $\mu _ { k + 1 } = 5 \mu _ { k }$ . For Krylov LM, the maximum Krylov-subspace dimension is restricted to 5% of the full parameter dimension, and the Lanczos tolerance is set to $1 0 ^ { - 3 }$ . For HSLM, the initial random-probe subspace dimension is set to 1% of the full parameter dimension. The subspace adequacy threshold is set to $\eta _ { \mathrm { m i n } } = 0 . 9 9$ When the current subspace is inadequate, it is expanded using Lanczos directions corresponding to 2% of the parameter dimension and additional random Hessian-probe directions corresponding to 1% of the parameter dimension. The total subspace dimension is restricted to at most 10% of the full parameter dimension. See [5] for a more detailed implementation.

## A.5. Stopping criteria

For the noisy regression problem, the target training error is defined by the variance of noise, $\mathrm { M S E } \leq \sigma _ { \mathrm { n o i s e } } ^ { 2 }$ . For SGD and Adam, optimization is also terminated if the relative improvement in MSE remains below $1 0 ^ { - 5 }$ for 50 consecutive epochs or if the maximum epoch budget is reached. For the LM-based methods, optimization may additionally terminate when $\left\| J ^ { T } r \right\| _ { \infty } \leq 1 0 ^ { - 8 }$ , or when the step satisfies $\left\| s _ { k } \right\| _ { \infty } \leq 1 0 ^ { - 1 0 } ( 1 + \left\| \theta _ { k } \right\| _ { \infty } )$ , or when the maximum iteration budget is reached.

For the 13-bit parity problem, the training stopping criterion requires $\mathrm { M S E \le 1 0 ^ { - 2 } }$ together with above 99% training accuracy. We record the training and validation MSE, classification accuracy, number of iterations or epochs, and execution time for each optimizer over the 100 trials.

## Appendix B. Detailed Training Performance

## B.1. Damped Oscillation Regression Problem

![](images/22bfff54907697820542c569eec3e7463a83726112cdafa24f0fe3cf6184e03c.jpg)

![](images/c6d7563175c0db5b960317054970c5efc840b12d54da4a6da63a1367ba99fdad.jpg)  
Figure 3: Comparison of optimization methods on Damped Oscillation regression problem. A. Approximation of the target function obtained using SGD, Adam, LM, Krylov LM, and HSLM. B. Training mean squared error (MSE) vs epoch/iteration for optimization methods.

## B.2. 13-Parity Classification Problem

![](images/0f3ff113021ef382937fc9b3b5bf696be0ba3cabd537942a586eb0440949c7f5.jpg)

![](images/9bb5e1de083b57fb8eaa79e13b47c0698c1008bc820b1710b37d8b9f8a4e6d03.jpg)

![](images/34c7be79855d873d81743bcc54de4f155142c9fdfef8747e86eced9857606b73.jpg)

![](images/29f2f537da2c0b766c4779d767a7664ce7fe817b8462964cd5ce8435252c2b85.jpg)  
Figure 4: Training and validation performance of the optimization methods on the 13-parity classification problem. A. Training MSE. B. Validation MSE. C. Training accuracy. D. Validation accuracy.

## Appendix C. Further Experiments

To evaluate optimizer performance under limited-data conditions, we reduced the training/validation split to 30/70 and reran the algorithms on the 13-bit parity problem. The results show that the HSLM method maintains a stronger balance between convergence speed, computational efficiency, and predictive accuracy compared to other methods.

![](images/569496ee81c79790e9be22a2f847189c325836c10d650b13c43905779feca3a7.jpg)

![](images/7e6092421a2b3fd7311ec401717189c2671d2553fe4bafb8b88b0a12a61bb6b4.jpg)

![](images/20e3483e7279cac216fdbb50301a4c2ce1b3b8f53fa166b7846b7402aaf6221c.jpg)

![](images/86224e6840f091a3d699f21794be929c501f5e07624572ab2e3e023f03562efd.jpg)  
Figure 5: Training and validation performance of the optimization methods on the 13-parity classification problem. A. Training MSE. B. Validation MSE. C. Training accuracy. D. Validation accuracy.