# Scale-Invariant Empirical-Bayes Laplace Approximation for ReLU Networks

Shivam Pal<sup>1</sup> and Piyush Rai<sup>1,2</sup>

<sup>1</sup>Department of Computer Science and Engineering, IIT Kanpur <sup>2</sup>Department of Intelligent Systems, IIT Kanpur {pshivam,piyush}@cse.iitk.ac.in

## Abstract

ReLU networks are well-known to have continuous scale symmetries; infinitely many reparameterized weight vectors that represent the same function. Dinh et al. [2017] have shown that, by reparameterizing, the geometry of the loss landscape (e.g., curvature) of these networks can be changed arbitrarily without affecting the underlying function. Thus, any quantity that depends on the network’s geometry should be defined with care. In this paper, we show that the Empirical Bayes (EB) Laplace’s approximation is not well-defined for ReLU networks, mainly due to Hessian dependence in computing the marginal likelihood, which is not invariant under ReLU scale symmetries. Thus, we can get different prior precisions, posterior covariances, and predictive uncertainties for a network having identical predictions under ReLU scaling. We define family of prior precision fields indexed by γ and introduce γ-canonicalization, which first converts the network to its canonical representation by balancing the ratio of incoming to outgoing weights, and then fits the Empirical Bayes (EB) Laplace to the canonical representation. This procedure is equivalent to transporting the isotropic Gaussian prior from the canonical representation to the original coordinates, by producing a ReLUscale-compatible precision field. This makes the marginal likelihood invariant under ReLU scaling, while allowing γ to be selected by maximizing the marginal likelihood. The resulting EB Laplace approximation is well-defined on ReLU scaleequivalence classes and yields consistent prior precision selection and predictive uncertainties across functionally identical parameterizations.

## 1 Introduction

The Laplace’s approximation is a simple post-hoc method for approximate Bayesian inference in deep neural networks. It fits a local Gaussian posterior whose mean is a maximum-a-posteriori (MAP) solution, and covariance is determined by curvature and prior precision [MacKay, 1992, Ritter et al., 2018, Daxberger et al., 2021]. The prior precision is often selected using marginal likelihood, which can be estimated solely from training data and without any validation data. This is also known as empirical Bayes (EB) or type-II maximum likelihood learning. Recent progress in structured curvature approximations, linearized prediction, and software support [Ritter et al., 2018, Daxberger et al., 2021, Immer et al., 2021a] has made the EB Laplace approximation an attractive method for post-hoc uncertainty estimation and Bayesian model selection.

In this paper, we study the transformation properties of marginal likelihood under ReLU scale symmetries. For ReLU networks, the marginal likelihood is not well-defined on the learned predictor. Due to the positive homogeneity property of ReLU, multiplying the incoming and outgoing weights of a hidden unit by α > 0 and 1/α, respectively, makes the network function unchanged. These transformations generate ReLU scale orbits; continuous families of parameter vectors having the same function [Neyshabur et al., 2015a,b, Dinh et al., 2017]. Along such an orbit, the marginal likelihood is not invariant mainly due to the log-determinant of the curvature and the norm of the chosen orbit which do not transform compatibly. Consequently, standard EB Laplace can give different marginal likelihoods, prior precisions, and linearized predictive uncertainties to the parameterizations that represent exactly the same predictor.

We fix this problem by introducing γ-canonicalization where $\gamma \in [ 0 , 1 ]$ controls the choice of canonical representation. Our procedure first converts the network (MAP solution) to its canonical representation by balancing the ratio of incoming and outgoing weights of the hidden units of the network, then fitting the EB Laplace on the canonical representation. This generalizes the balanced and one-sided normalizations studied in scale-symmetric networks [Stock et al., 2019, Saul, 2023]. This procedure is equivalent to transporting the isotropic Gaussian prior from the canonical representation to the original coordinates, by producing a ReLU-scale-compatible precision field. We show that, for every $\gamma _ { \mathrm { : } }$ , the resulting generalized marginal likelihood is invariant on the ReLU scale orbit. Thus, it makes the EB Laplace approximation explicitly depend on the learned function and not on arbitrary parametrization of the function that is obtained due to training artifacts. γ is selected jointly with prior precision using generalized marginal likelihood.

Empirically, we further show that the standard EB Laplace varies substantially across functionally identical ReLU rescaling, while γ-canonicalized EB Laplace becomes invariant. We also find that γ-canonicalized EB Laplace gives better uncertainty estimates than standard Laplace, although our main guarantees are for the invariance.

## 2 Empirical-Bayes Laplace Approximation

Let $\mathbf { f } ( \cdot ; \mathbf { w } ) \colon \mathcal { X } \to \mathbb { R } ^ { C }$ be a neural network with parameters w $\mathbf { \Lambda } \in \mathbb { R } ^ { D } .$ , where $C$ is the number of output classes and D is the number of parameters. Let $\mathcal { D } = \{ ( x _ { n } , y _ { n } ) \} _ { n = 1 } ^ { N }$ denote the training data. We assume that a MAP solution wˆ is obtained by minimizing the regularized training loss, and log $p ( \mathcal { D } \mid \mathbf { w } )$ denotes the log-likelihood evaluated at w.

Laplace Approximation. The Laplace approximation replaces the intractable posterior $p ( \mathbf { w } \mid \mathcal { D } )$ by a Gaussian centered at wˆ . Given a likelihood-curvature approximation $\mathbf { C } _ { \hat { \mathbf { w } } } \in \mathbb { S } _ { + } ^ { D }$ and a prior precision $\lambda > 0 .$ , the approximate posterior is $q ( \mathbf { w } ) = \mathcal { N } ( \mathbf { w } \mid \hat { \mathbf { w } } , \pmb { \Sigma } _ { \hat { \mathbf { w } } } ) , \pmb { \Sigma } _ { \hat { \mathbf { w } } } = \left( \mathbf { C } _ { \hat { \mathbf { w } } } + \lambda \mathbf { I } \right) ^ { - 1 }$ . The exact Hessian is often not tractable, and also not guaranteed to be positive definite for neural networks. Common choices for curvature approximations are the generalized Gauss–Newton (GGN) matrix and the empirical Fisher matrix [MacKay, 1992, Ritter et al., 2018, Daxberger et al., 2021]. The GGN and Fisher are positive semi-definite by construction and are often preferred in practice. We will assume the GGN approximation of curvature throughout the paper.

Prior Precision and Empirical-Bayes Selection. The prior precision λ controls the scale of the Gaussian posterior and affects the uncertainty estimates. It can be selected using a validation set. However a common empirical-Bayes alternative is to select λ by maximizing the approximate marginal likelihood on the training data. Up to constants independent of $\lambda ,$ the log marginal likelihood is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t d } } ( \lambda ; \hat { \mathbf { w } } ) = \log p ( \mathcal { D } \mid \hat { \mathbf { w } } ) - \frac { 1 } { 2 } \log | \mathbf { C } _ { \hat { \mathbf { w } } } + \lambda \mathbf { I } | + \frac { D } { 2 } \log \lambda - \frac { \lambda } { 2 } \| \hat { \mathbf { w } } \| ^ { 2 } . } \end{array}\tag{1}
$$

Note that for deep neural networks, the marginal likelihood cannot be computed exactly. Eq. 1 is only an approximation, widely used in Bayesian deep learning [MacKay, 1992, Immer et al., 2021a,b, Antoran et al., 2022a]. The prior precision is then given by

$$
\lambda ^ { \star } ( \hat { \mathbf { w } } ) \in \mathop { \mathrm { a r g } } _ { \lambda > 0 } \operatorname* { m a x } _ { \lambda > 0 } ( \lambda ; \hat { \mathbf { w } } ) .\tag{2}
$$

Linearized Predictive Distribution. For a test input $x ,$ let $\mathbf { J } _ { \hat { \mathbf { w } } } ( x ) \in \mathbb { R } ^ { C \times D }$ denote the Jacobian of $\mathbf { f } ( \cdot ; \mathbf { w } )$ with respect to w, evaluated at wˆ . The Jacobian-linearized distribution over network outputs is Gaussian with mean $\pmb { \mu } ( x ) = \mathbf { f } ( x ; \hat { \mathbf { w } } )$ , and covariance $\mathbf { K } _ { \lambda } ( x ) = \mathbf { J } _ { \hat { \mathbf { w } } } ( x ) \pmb { \Sigma } _ { \hat { \mathbf { w } } } \mathbf { J } _ { \hat { \mathbf { w } } } ( x ) ^ { \top }$

## 3 Parameter-Space Symmetries in Neural Networks

A parameter-space symmetry of a neural network is a transformation of the network parameter that leaves the network function unchanged. Such symmetries matter for post-hoc Bayesian inference

because a quantity that changes under an exact function-preserving transformation is a property of the parameter representative, not of the learned function of the neural network.

Definition 3.1. A map $T \colon \mathbb { R } ^ { D }  \mathbb { R } ^ { D }$ is a parameter-space symmetry of $\mathbf { f } \left( \cdot ; \mathbf { w } \right) \mathrm { i f }$

$$
\mathbf { f } ( x ; T ( \mathbf { w } ) ) = \mathbf { f } ( x ; \mathbf { w } ) \qquad { \mathrm { f o r ~ a l l ~ } } x \in { \mathcal { X } } { \mathrm { ~ a n d ~ } } \mathbf { w } \in \mathbb { R } ^ { D } .
$$

Given a set of symmetries ${ \mathcal { G } } ,$ a scalar-valued function $\phi ( \mathbf { w } )$ is invariant under G if $\phi ( T ( \mathbf { w } ) ) = \phi ( \mathbf { w } )$ for all $T \in { \mathcal { G } }$ . A matrix-valued function $\Phi ( \mathbf { w } )$ is equivariant under G if it transforms predictably under each $T \in \mathcal G$ , meaning there exists a known map $\rho _ { T }$ such that $\Phi ( T ( \mathbf { w } ) ) \ = \ \bar { \rho } _ { T } ( \Phi ( \mathbf { w } ) )$ Invariance is the special case where $\rho _ { T }$ is the identity.

Neural networks have several exact parameter-space symmetries [Zhao et al., 2026]. For example, permutation symmetry, i.e., relabeling the hidden units of a layer and applying the corresponding permutation to adjacent weight matrices leaves the function unchanged [Hecht-Nielsen, 1990, Entezari et al., 2022, Ainsworth et al., 2023]. In this work, we focus only on continuous symmetry specific to ReLU networks.

## 3.1 ReLU Scale Orbits

ReLU networks have continuous exact symmetries arising due to their homogeneity property. For any $\alpha > 0 , \mathrm { R e L U } ( \alpha z ) = \alpha \mathrm { R e L U } ( z )$ . Thus, the scale of a hidden unit can be redistributed between its incoming and outgoing weights without changing the network function. Consider a feedforward ReLU network with weight matrix $\mathbf { W } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell } \times d _ { \ell - 1 } }$ and bias vector $\mathbf { b } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell } }$ at layer ℓ. For a hidden unit (ℓ, i) and a scalar $\alpha > 0 .$ , the rescaling

$$
\mathbf { W } ^ { ( \ell ) } [ i , : ]  \alpha \mathbf { W } ^ { ( \ell ) } [ i , : ] , \qquad \mathbf { b } ^ { ( \ell ) } [ i ]  \alpha \mathbf { b } ^ { ( \ell ) } [ i ] , \qquad \mathbf { W } ^ { ( \ell + 1 ) } [ : , i ]  \alpha ^ { - 1 } \mathbf { W } ^ { ( \ell + 1 ) } [ : , i ]\tag{3}
$$

leaves the network function unchanged. Composing this over all hidden units creates a group G of exact function-preserving transformations. Let $\textbf { w } \in \mathbb { R } ^ { D }$ be the flattened parameter and $\alpha = ( \alpha _ { \ell , i } ) > 0$ collects one positive scalar per hidden unit. Then, the corresponding transformation is $T _ { \alpha } ( \mathbf { w } ) = \mathbf { D } _ { \alpha } \mathbf { w }$ . Here, $\mathbf { D } _ { \alpha } ^ { \mathsf { ^ { - } } } \in \mathbb { R } ^ { D \times D }$ is a diagonal matrix corresponding to α. The ReLU scale orbit of w is the equivalence class $[ \mathbf { w } ] = \{ T _ { \alpha } ( \mathbf { w } ) \mid \alpha > 0 \}$ . Every element of [w] represents exactly the same function.

## 3.2 Transformation Laws under ReLU Rescaling

Let $\mathbf { v } = T _ { \alpha } ( \mathbf { w } ) = \mathbf { D } _ { \alpha } .$ w be a ReLU rescaling. By symmetry, $\mathbf { f } ( x ; \mathbf { v } ) = \mathbf { f } ( x ; \mathbf { w } )$ . Since ${ \pmb w } = { \pmb { \mathrm { D } } } _ { \alpha } ^ { - 1 } { \pmb v }$ and using the chain rule, the Jacobian transforms as follows.

$$
{ \bf J } _ { \pmb { v } } ( { \boldsymbol { x } } ) = \frac { \partial { \bf f } } { \partial \pmb { v } } = \frac { \partial { \bf f } } { \partial \pmb { w } } \frac { \partial \pmb { w } } { \partial \pmb { v } } = { \bf J } _ { \pmb { w } } ( { \boldsymbol { x } } ) { \bf D } _ { \alpha } ^ { - 1 } .\tag{4}
$$

The generalized Gauss–Newton (GGN) curvature matrix can be written as $\begin{array} { r l } { \mathbf { C } _ { v } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { n } \mathbf { J } _ { v } ^ { \mathbf { \overline { { \nu } } } } ( x _ { n } ) ^ { \top } \mathbf { A } _ { n } \mathbf { J } _ { v } ( x _ { n } ) } \end{array}$ . And this transforms by congruence,

$$
\begin{array} { r } { \mathbf { C } _ { v } = \mathbf { D } _ { \alpha } ^ { - 1 } \mathbf { C } _ { w } \mathbf { D } _ { \alpha } ^ { - 1 } . } \end{array}\tag{5}
$$

This congruence law also holds for standard KFAC curvature blocks under ReLU rescaling. We give the layerwise Kronecker-factor derivation in Appendix D.

## 4 Orbit Dependence of Standard Empirical-Bayes Laplace

Definition 4.1 (Orbit consistency). An empirical-Bayes Laplace procedure is orbit-consistent if, for every ReLU rescaling ${ \mathbf v } = T _ { \alpha } ( { \mathbf w } )$ , the marginal likelihood, selected prior precision, and covariance of the predictive distribution are the same on the ReLU-scale orbit.

$$
{ \mathcal { L } } ( \lambda ; { \mathbf { v } } ) = { \mathcal { L } } ( \lambda ; { \mathbf { w } } ) , \qquad \lambda ^ { \star } ( { \mathbf { v } } ) = \lambda ^ { \star } ( { \mathbf { w } } ) , \qquad { \mathbf { K } } _ { \lambda } ( { \mathbf { v } } ; x ) = { \mathbf { K } } _ { \lambda } ( { \mathbf { w } } ; x ) .\tag{6}
$$

Standard EB Laplace approximation is not orbit-consistent. Let $\mathbf { v } = T _ { \alpha } ( \mathbf { w } ) = \mathbf { D } _ { \alpha } \mathbf { w }$ be a ReLU rescaling. A sufficient condition for $\mathcal { L } _ { \mathrm { s t d } }$ to be orbit-consistent is that all four terms in (1) are invariant under this rescaling. The likelihood term satisfies this since the network function is unchanged, log $p ( \mathcal { D } \mid \mathbf { v } ) = \log p ( \mathcal { D } \mid \mathbf { w } )$ . However, the prior quadratic term satisfies $\| \mathbf { v } \| ^ { 2 } = \| \mathbf { D } _ { \alpha } \mathbf { w } \| ^ { 2 } \neq \| \mathbf { w } \| ^ { 2 }$ in general. And the log-determinant term is also not invariant. Although the curvature transforms by congruence (5), the isotropic component λI does not transform with it, so $\mathbf { C } _ { \mathbf { v } } + \lambda \mathbf { I }$ is not a congruence transform of $\mathbf { C } _ { \mathbf { w } } + \lambda \mathbf { I }$ . Thus, $\mathcal { L } _ { \mathrm { s t d } } ( \boldsymbol { \lambda } ; { \mathbf w } )$ is not constant on $\left[ \mathbf { w } \right]$ , and the prior precision ${ { \lambda } ^ { \star } } ( \mathbf { w } )$ is selected by maximizing marginal likelihood. As we show below, the selected prior precision can also vary along the orbit. A detailed derivation of proposition 4.2 is provided in Appendix A.

Proposition 4.2 (Orbit dependence of standard EB-Laplace). Consider the scalar ReLU regression model $f ( x ; a , c ) = c$ ReLU(ax) with $a , c > 0$ , Gaussian observation noise of variance ${ \bf \check { \sigma } } _ { y } ^ { 2 } ,$ , and inputs $x _ { n } > 0$ . The rescaling (a, c) 7→ (αa, c/α) for $\alpha > 0$ preserves the realized predictor. Let $\begin{array} { r } { S = \sum _ { n = 1 } ^ { N } x _ { n } ^ { 2 } , \kappa = S / \sigma _ { y } ^ { 2 } , } \end{array}$ , and $q ( \alpha ) = ( \alpha a ) ^ { 2 } + ( c / \alpha ) ^ { 2 }$ . The unique positive stationary point of $\mathcal { L } _ { \mathrm { s t d } } ( \lambda ; \alpha )$ in λ is

$$
\begin{array} { r } { \lambda ^ { \star } ( \alpha ) = \frac { - \kappa q ( \alpha ) + \sqrt { \kappa ^ { 2 } q ( \alpha ) ^ { 2 } + 4 \kappa } } { 2 } . } \end{array}\tag{7}
$$

Since $q ( \alpha )$ is non-constant on the scale orbit, $\lambda ^ { \star } ( \alpha )$ varies with α. Hence standard empirical-Bayes Laplace assigns different marginal likelihoods, selected prior precisions, and linearized predictive uncertainties to parameter vectors representing exactly the same function.

## 5 Scale-Consistent Laplace

The orbit-dependence of standard empirical-Bayes Laplace comes from a mismatch between curvature and the isotropic prior. Under a ReLU rescaling $\mathbf { v } = T _ { \alpha } ( \mathbf { w } ) = \mathbf { D } _ { \alpha } \mathbf { w }$ , Jacobian-based curvature transforms by congruence (5), whereas the standard prior precision λI is fixed in the original coordinates. In this section we identify the transformation law required for prior precision to be compatible with ReLU scale orbits, and prove the resulting invariance of marginal likelihood, selected prior precision and covariance of linearized prediction distribution.

## 5.1 Scale-Compatible Precision Fields

A scalar precision λ cannot make λI transform like curvature under arbitrary hidden-unit rescalings.   
We therefore allow the prior precision to depend on the MAP solution.

Definition 5.1 (Scale-compatible precision field). A map $\mathbf { w } \mapsto \pmb { \Lambda } ( \mathbf { w } ) \in \mathbb { S } _ { + + } ^ { D }$ is scale-compatible if, for every ReLU rescaling $\mathbf { v } = T _ { \alpha } ( \mathbf { w } ) = \mathbf { D } _ { \alpha } \mathbf { w }$

$$
\begin{array} { r } { \pmb { \Lambda } ( \mathbf { v } ) = \mathbf { D } _ { \alpha } ^ { - 1 } \pmb { \Lambda } ( \mathbf { w } ) \mathbf { D } _ { \alpha } ^ { - 1 } . } \end{array}\tag{8}
$$

Given such a field, define the generalized Laplace evidence

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g e n } } ( \mathbf { w } ) = \log p ( \mathcal { D } \mid \mathbf { w } ) - \frac { 1 } { 2 } \log \left| \mathbf { C } _ { \mathbf { w } } + \mathbf { A } ( \mathbf { w } ) \right| + \frac { 1 } { 2 } \log \left| \mathbf { A } ( \mathbf { w } ) \right| - \frac { 1 } { 2 } \mathbf { w } ^ { \top } \mathbf { A } ( \mathbf { w } ) \mathbf { w } . } \end{array}\tag{9}
$$

When $\pmb { \Lambda } ( \mathbf { w } ) = \lambda \mathbf { I }$ , this reduces to the standard isotropic objective in (1).

Theorem 5.2 (Evidence invariance). Let $\mathbf { v } = T _ { \alpha } ( \mathbf { w } ) = \mathbf { D } _ { \alpha } \mathbf { w }$ be a ReLU rescaling. Suppose that the curvature satisfies (5) and that Λ is scale-compatible. Then $\mathcal { L } _ { \mathrm { g e n } } ( \mathbf { v } ) = \mathcal { L } _ { \mathrm { g e n } } ( \mathbf { w } )$

Proof of the theorem 5.2 is given in Appendix B. If $\{ \pmb { \Lambda } _ { \eta } ( \mathbf { w } ) \} _ { \eta \in \mathcal { H } }$ is a family of scale-compatible precision fields, then $\mathcal { L } _ { \mathrm { g e n } } ( \mathbf { \bar { v } } ; \eta ) = \mathcal { L } _ { \mathrm { g e n } } ( \mathbf { w } ; \eta )$ for every $\eta .$ Therefore, the set of hyperparameters that are obtained by maximizing the marginal likelihood will also be constant on ReLU scale orbit.

## 5.2 Posterior Equivariance and Predictive Invariance

Scale compatibility also gives the correct transformation rule for the posterior covariance and the covariance of the predictive distribution under the Laplace approximation. Let posterior covariance be $\pmb { \Sigma _ { \mathbf { w } } } = ( \pmb { \mathrm { C } } _ { \mathbf { w } } + \pmb { \Lambda } ( \mathbf { w } ) ) ^ { - 1 }$ , and $\pmb { \Sigma _ { \mathbf { v } } } = ( \mathbf { C _ { \mathbf { v } } } + \pmb { \Lambda } ( \mathbf { v } ) ) ^ { - 1 }$ . Then $\pmb { \Sigma _ { v } } = \mathbf { D } _ { \alpha } \pmb { \Sigma _ { w } } \mathbf { D } _ { \alpha }$ . Thus, the posterior covariance is equivariant under scale-compatible precision field. This is the expected behavior under transformation. The induced Jacobian-linearized predictive covariance, however, is invariant.

Theorem 5.3 (Linearized predictive invariance). Let $\mathbf { v } \ = \ T _ { \alpha } ( \mathbf { w } ) \ = \ \mathbf { D } _ { \alpha } \mathbf { w }$ . Suppose that the curvature satisfies (5) and that Λ is scale-compatible. Then,for every test input x,

$$
\mathbf { f } ( x ; \mathbf { v } ) = \mathbf { f } ( x ; \mathbf { w } ) , \qquad \mathbf { J } _ { \mathbf { v } } ( x ) \pmb { \Sigma } _ { \mathbf { v } } \mathbf { J } _ { \mathbf { v } } ( x ) ^ { \top } = \mathbf { J } _ { \mathbf { w } } ( x ) \pmb { \Sigma } _ { \mathbf { w } } \mathbf { J } _ { \mathbf { w } } ( x ) ^ { \top } .\tag{10}
$$

Proof. The mean is invariant because v and w represent the same network function. For the covariance, using the transformation rule for Jacobian from Eq. $4 , \mathbf { J } _ { \mathbf { v } } ( x ) = \mathbf { J } _ { \mathbf { w } } ( x ) \mathbf { D } _ { \alpha } ^ { - 1 }$

$$
\begin{array} { r l } & { \mathbf { J } _ { \mathbf { v } } ( x ) \pmb { \Sigma } _ { \mathbf { v } } \mathbf { J } _ { \mathbf { v } } ( x ) ^ { \top } = \mathbf { J } _ { \mathbf { w } } ( x ) \mathbf { D } _ { \alpha } ^ { - 1 } \mathbf { D } _ { \alpha } \pmb { \Sigma } _ { \mathbf { w } } \mathbf { D } _ { \alpha } \mathbf { D } _ { \alpha } ^ { - 1 } \mathbf { J } _ { \mathbf { w } } ( x ) ^ { \top } } \\ & { \qquad = \mathbf { J } _ { \mathbf { w } } ( x ) \pmb { \Sigma } _ { \mathbf { w } } \mathbf { J } _ { \mathbf { w } } ( x ) ^ { \top } . } \end{array}
$$

## 5.3 Transported Prior from a Canonical Section

We have now seen that, to make the EB Laplace scale-consistent, the precision field should be scalecompatible. Here we show how to construct such a precision field. Let’s assume that $C _ { \gamma } \colon \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ selects a canonical representative $\bar { \mathbf { w } } = C _ { \gamma } ( \mathbf { w } )$ from a ReLU scale orbit and satisfies the following property

$$
C _ { \gamma } ( T _ { \alpha } ( \mathbf { w } ) ) = C _ { \gamma } ( \mathbf { w } ) \mathrm { f o r e v e r y R e L U r e s c a l i n g } T _ { \alpha } .\tag{11}
$$

The above property Eq. 11 is also called a section property and $C _ { \gamma }$ is called canonical section. We assume that ReLU scale orbit is non-degenerate and the scale displacement between w and $C _ { \gamma } ( \mathbf { w } )$ is well defined. Since w and $C _ { \gamma } ( \mathbf { w } )$ lie on the same orbit, there exists a positive diagonal matrix $ { \mathbf { D } } (  { \mathbf { w } } ; \gamma )$ such that $\mathbf { w } = \mathbf { D } ( \mathbf { w } ; \gamma ) C _ { \gamma } ( \mathbf { w } )$ . This matrix records the scale displacement from the canonical representative to the original representative. Using an isotropic Gaussian prior with precision λ at the canonical representative,

$$
\begin{array} { r } { p _ { \gamma } ( \bar { \mathbf { w } } ) \propto \exp \left( - \frac { \lambda } { 2 } \| \bar { \mathbf { w } } \| ^ { 2 } \right) , \qquad \bar { \mathbf { w } } = C _ { \gamma } ( \mathbf { w } ) . } \end{array}\tag{12}
$$

Using $\bar { \mathbf { w } } = \mathbf { D } ( \mathbf { w } ; \boldsymbol { \gamma } ) ^ { - 1 } \mathbf { w }$ , the canonical quadratic penalty becomes

$$
\begin{array} { r } { \frac \ { \lambda } { 2 } \| \bar { \mathbf { w } } \| ^ { 2 } = \frac { 1 } { 2 } \mathbf { w } ^ { \top } \mathbf { A } _ { \gamma } ( \mathbf { w } ; \lambda ) \mathbf { w } , \quad \mathrm { w h e r e } \quad \mathbf { A } _ { \gamma } ( \mathbf { w } ; \lambda ) = \lambda \mathbf { D } ( \mathbf { w } ; \gamma ) ^ { - 2 } . } \end{array}\tag{13}
$$

Thus, quadratic penalty of an isotropic Gaussian prior in canonical coordinates is equivalent to that induced by a representative-dependent (transported), diagonal precision in the original coordinates.

Proposition 5.4 (Scale compatibility of the transported prior). Suppose $C _ { \gamma }$ satisfies the section property (11). Then, for every $\lambda > 0$ , the transported precision field $\pmb { \Lambda } _ { \gamma } ( \mathbf { w } ; \lambda )$ defined by (13) is scale-compatible.

Corollary 5.5 (Orbit-consistent transported Laplace). Suppose $C _ { \gamma }$ satisfies the section property (11). Then,for everyfixed γ and $\lambda > 0$ , the generalized evidence and the Jacobian-linearized predictive distribution induced by $\boldsymbol { \Lambda } _ { \gamma } ( \mathbf { w } ; \lambda )$ are invariant under ReLU rescaling. Moreover, the evidencemaximizing set of prior precisions is the same for all representatives of a ReLU scale orbit.

The proof of the Proposition 5.4 is given in Appendix C. Combining Proposition 5.4 with Theorems 5.2 and 5.3 gives the section-level results. The transported-prior framework reduces the remaining problem to constructing a section $C _ { \gamma }$ satisfying Eq. (11). Next we give such a construction via γ-canonicalization.

## 6 γ-Canonicalization

Any canonical section $C _ { \gamma }$ satisfying (11) makes empirical-Bayes Laplace orbit-consistent. Here, we introduce a practical algorithm for selecting a canonical section. The construction is based on a family of gauge conditions that balance incoming and outgoing weight norms of the hidden units. First, we state the assumptions required for this construction and then give the iterative procedure for selecting the canonical representation.

## 6.1 Canonical Gauge Condition

A canonical section should not depend on the arbitrary representative returned by MAP training. ReLU scale orbits provide a natural way to define such a section. For each hidden unit, the scale symmetry redistributes the magnitude between the incoming and outgoing weights. Fixing the relative placement of this magnitude gives a gauge condition on the orbit. The parameter $\gamma \in [ 0 , 1 ]$ controls where the canonical representative lies between incoming-normalized, balanced, and outgoing-normalized representatives. This construction is closely related to equi-normalization and weight-balancing methods for scale-symmetric networks [Stock et al., 2019, Saul, 2023]. Without loss of generality we assume a feedforward ReLU network without any normalization layers. Let us define the magnitude of incoming weight of hidden unit $( \ell , i )$ by

Algorithm 1 γ-Canonicalization   
Require: parameters w, gauge index $\gamma \in [ 0 , 1 ] .$ , tolerance $\varepsilon > 0$   
Ensure: representative $\bar { \mathbf { w } } = C _ { \gamma } ( \mathbf { w } )$   
1: repeat   
2: for each hidden unit (ℓ, i) in sweep order do   
3: compute $s _ { i } ^ { ( \ell ) } ( \mathbf { w } ) > 0$ and $t _ { i } ^ { ( \ell ) } ( \mathbf { w } ) > 0$ ▷ Skip dead neuron   
4: $\alpha  ( t _ { i } ^ { ( \ell ) } ( \mathbf { w } ) ) ^ { \gamma } \big ( s _ { i } ^ { ( \ell ) } ( \mathbf { w } ) \big ) ^ { - ( 1 - \gamma ) }$   
5: $\mathbf { W } ^ { ( \ell ) } [ i , : ]  \alpha \mathbf { W } ^ { ( \ell ) } [ i , : ]$   
6: $b _ { i } ^ { ( \ell ) } \gets \alpha b _ { i } ^ { ( \ell ) }$   
7: $\mathbf { \Delta W } ^ { ( \ell + 1 ) } [ : , i ]  \alpha ^ { - 1 } \mathbf { W } ^ { ( \ell + 1 ) } [ : , i ]$   
8: end for   
9: until ma $\natural , i \mid \log \alpha _ { \ell , i } \mid < \varepsilon$   
10: return w

$$
s _ { i } ^ { ( \ell ) } ( \mathbf { w } ) = \left( \left\| \mathbf { W } ^ { ( \ell ) } [ i , : ] \right\| ^ { 2 } + \left| b _ { i } ^ { ( \ell ) } \right| ^ { 2 } \right) ^ { 1 / 2 }\tag{14}
$$

and the magnitude of outgoing weight by

$$
t _ { i } ^ { ( \ell ) } ( \mathbf { w } ) = \| \mathbf { W } ^ { ( \ell + 1 ) } [ : , i ] \| .\tag{15}
$$

Under the elementary rescaling (3) of the same unit $( \ell , i ) \ : \mathrm { b y } \ : \alpha > 0$ , with all other unit scales held fixed, these magnitudes transform as $s _ { i } ^ { ( \ell ) } \mapsto \alpha s _ { i } ^ { ( \ell ) } , t _ { i } ^ { ( \ell ) } \mapsto \alpha ^ { - 1 } t _ { i } ^ { ( \ell ) }$ . Thus the elementary rescaling changes the ratio between incoming and outgoing magnitudes while preserving their product. Across a multilayer network, the gauge equations are coupled because rescaling the current hidden units changes the adjacent hidden unit’s incoming and outgoing weight magnitudes. For $\gamma \in [ 0 , 1 ]$ , the γ-canonical gauge condition is given by

$$
\left( s _ { i } ^ { ( \ell ) } ( \bar { \mathbf { w } } ) \right) ^ { 1 - \gamma } = \big ( t _ { i } ^ { ( \ell ) } ( \bar { \mathbf { w } } ) \big ) ^ { \gamma } \qquad \mathrm { f o r e v e r y h i d d e n u n i t } ( \ell , i ) .\tag{16}
$$

For $\gamma = 0$ , the condition becomes $s ( \bar { \bf w } ) = 1$ , giving an incoming-normalized representative. For $\gamma = 1$ , it becomes $t ( \bar { \bf w } ) = 1$ , giving an outgoing-normalized representative. For $\gamma = 0 . 5$ , it becomes $s ( \bar { \bf w } ) = t ( \bar { \bf w } )$ , giving a balanced or minimum norm representative.

Assumption 6.1 (Canonical section). For each $\gamma \in [ 0 , 1 ]$ and each non-degenerate parameter vector, there exists a deterministic representative $\bar { \mathbf { w } } = C _ { \gamma } ( \mathbf { w } ) \in [ \mathbf { w } ]$ satisfying the gauge condition (16). The resulting map $C _ { \gamma }$ is a section of the ReLU scale quotient

$$
C _ { \gamma } ( T _ { \alpha } ( \mathbf { w } ) ) = C _ { \gamma } ( \mathbf { w } ) \qquad \mathrm { f o r e v e r y R e L U r e s c a l i n g } T _ { \alpha } .\tag{17}
$$

Assumption 6.1 requires the existence of a canonical representative. In our implementation, we compute $C _ { \gamma }$ by the iterative algorithm 1.

## 6.2 Iterative Construction

To compute the canonical representative, we repeatedly apply elementary rescalings. Holding all other unit scales fixed, the rescaling factor that enforces (16) for unit $( \ell , i )$ is

$$
\alpha _ { \ell , i } ( \mathbf { w } ; \gamma ) = \big ( t _ { i } ^ { ( \ell ) } ( \mathbf { w } ) \big ) ^ { \gamma } \big ( s _ { i } ^ { ( \ell ) } ( \mathbf { w } ) \big ) ^ { - ( 1 - \gamma ) } .\tag{18}
$$

Applying (3) with this factor sets the local residual $r _ { \ell , i } ( \mathbf { w } ; \gamma ) = ( 1 - \gamma ) \log s _ { i } ^ { ( \ell ) } ( \mathbf { w } ) - \gamma \log t _ { i } ^ { ( \ell ) } ( \mathbf { w } )$ to zero for that unit. Since the residuals are coupled across layers for interior values of $\gamma ,$ , we sweep over hidden units and repeat until the maximum update size is below a tolerance. We do not provide a proof of convergence for the γ-canonicalization algorithm. It is left for future work. For $\gamma = 0 . 5$ the convergence proof is given by [Saul, 2023]. In the experiments, convergence is assessed by the maximum gauge residual and by verifying that canonicalization preserves the network outputs to numerical precision. We find the algorithm always converges within at most 200 sweeps. We use a first-to-last sweep order for $\gamma < 1$ and a last-to-first sweep order for $\gamma = 1$ . The algorithm converges in a single sweep for $\gamma = 0$ and $\gamma = 1$

## 6.3 Selecting γ

The parameter $\gamma$ specifies where the isotropic prior is placed on the incoming and outgoing scale axis. It is therefore a prior-shape hyperparameter. Different values of γ correspond to different orbit-consistent priors. Since $\gamma$ indexes a prior family, it can be selected by the same marginal likelihood criterion used to select the prior precision. For a finite candidate set ${ \dot { \Gamma } } \subseteq [ 0 , 1 ]$ , we choose

$$
( \lambda ^ { \star } , \gamma ^ { \star } ) \in \underset { \lambda > 0 , \gamma \in \Gamma } { \arg \operatorname* { m a x } } \mathcal { L } _ { \mathrm { g e n } } ( \mathbf { w } ; \lambda , \gamma ) .\tag{19}
$$

By Corollary 5.5, under Assumption 6.1, the generalized evidence is invariant on $\left[ \mathbf { w } \right]$ for every fixed $( \lambda , \gamma )$ . Hence the maximizing set of $( \lambda , \gamma )$ values is the same for every representative of the ReLU scale orbit. The choice $\gamma = 0 . 5$ is geometrically natural and corresponds to the minimum norm canonical representation, However, it need not be optimal under the marginal likelihood. The evidence can prefer a prior placed closer to the incoming-normalized or outgoing-normalized representative. We therefore treat γ as a hyperparameter and select it from a grid search in the experiments.

## 7 Experiments

We evaluate $\gamma \cdot$ -canonicalization on both small and large network architectures, including ResNets. First, we perform an orbit-stress test on a Fashion-MNIST ReLU MLP to test whether the evidence, selected precision, and predictive uncertainty remain invariant across functionally identical ReLU rescaling. Next, we use a CIFAR-10 ReLU ConvNet with KFAC-Laplace to examine how $\gamma \mathrm { - }$ canonicalization affects predictive uncertainty in a convolution model. Although our theoretical guarantee is about invariance, in the next experiment we test whether $\gamma \cdot$ -canonicalization can improve uncertainty estimates when the MAP representative is poorly scaled.

Orbit-stress test. We first test orbit consistency by applying post-hoc Laplace to different parameterizations of the same trained ReLU predictor. We train a two-hidden-layer ReLU MLP on Fashion-MNIST and generate $K = 2 0$ function-preserving ReLU rescalings of the MAP solution by sampling unit-wise log $\alpha _ { i } \sim \mathcal { N } ( 0 , 1 )$ and applying the elementary rescaling in Eq. 3. All representatives preserve the original logits to numerical precision. We then fit full-network diagonal-GGN EB-Laplace to each representative, either in raw coordinates or after γ-canonicalization. Complete experimental details are given in Appendix E.

Table 1: Orbit stress test on a single trained ReLU predictor. We draw $K = 2 0$ function-preserving ReLU rescalings with log $\alpha _ { i } \sim \mathcal { N } ( 0 , 1 )$ and fit diagonal-GGN EB-Laplace to each representative. Values are mean ± standard deviation across orbit samples.
<table><tr><td>Method</td><td> $\log \lambda ^ { \star }$ </td><td>NLL</td><td>ECE</td><td> $\operatorname { A c c } .$ </td><td></td><td> $\log Z / 1 0 ^ { 4 }$ </td><td> $\| \boldsymbol { w } \| ^ { 2 }$ </td><td> $\operatorname* { m a x } { \| \Delta p \| _ { \infty } }$ </td></tr><tr><td></td><td></td><td>Standard EB 2.69±0.36 0.3906±0.0045 0.0682±0.0059</td><td></td><td> $0 . 8 7 1 8 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $- 8 . 8 0 { \pm } 0 . 8 9$ </td><td>2769±1016</td><td></td><td> $2 . 6 2 \times 1 0 ^ { - 1 }$ </td></tr><tr><td> $\gamma = 0 . 0 0$ </td><td> $5 . 0 7 { \pm } 0 $ </td><td>0.3646±0</td><td> $0 . 0 3 8 7 { \pm } 0$ </td><td> $0 . 8 7 2 5 { \pm } 0 $ </td><td> $- 4 . 7 2 \pm 0$ </td><td>156</td><td></td><td> $8 . 3 5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td> $\gamma = 0 . 2 5$ </td><td> $5 . 1 8 { \pm } 0 $ </td><td>0.3644±0</td><td> $0 . 0 3 8 9 { \pm } 0$ </td><td> $0 . 8 7 2 6 { \pm } 0$ </td><td> $- 4 . 7 0 \pm 0$ </td><td>140</td><td> $1 . 0 7 \times 1 0 ^ { - 6 }$ </td><td></td></tr><tr><td> $\gamma = 0 . 5 0$ </td><td> $5 . 2 9 { \pm 0 }$ </td><td>0.3672±0</td><td> $0 . 0 4 2 8 { \pm } 0$ </td><td> $0 . 8 7 2 4 { \pm } 0$ </td><td>-4.95±0</td><td>134</td><td></td><td> $8 . 3 5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td> $\gamma = 0 . 7 5$ </td><td> $5 . 3 5 { \pm 0 }$ </td><td>0.3723±0</td><td> $0 . 0 4 9 5 { \pm } 0 $ </td><td> $0 . 8 7 2 1 { \pm } 0$ </td><td> $- 5 . 4 2 \pm 0$ </td><td>139</td><td></td><td> $7 . 7 5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td> $\gamma = 1 . 0 0$ </td><td> $5 . 3 2 { \pm 0 }$ </td><td>0.3776±0</td><td> $0 . 0 5 6 2 { \pm } 0$ </td><td> $0 . 8 7 2 3 { \pm } 0$ </td><td> $- 5 . 9 4 \pm 0$ </td><td>156</td><td> $7 . 7 5 \times 1 0 ^ { - 7 }$ </td><td></td></tr></table>

Tables 1 and 2 show that standard EB-Laplace is strongly orbit-dependent. Functionally identical representatives give different selected precisions, marginal likelihoods, and predictive probabilities. In contrast, for every $\gamma ,$ canonicalized Laplace  
Table 2: Spread statistics of Orbit stress test.

removes this variation to numerical precision. The evidence selects $\gamma = 0 . 2 5$ , while the balanced representative $\gamma = 0 . 5$ gives the smallest parameter norm. However, it is not selected. It shows that generalized marginal likelihood is choosing a prior shape and not just minimizing the canonical parameter norm. Random function-preserving rescalings of the same MAP solution substantially change standard EB-Laplace, confirming that explicit canonicaliza-

<table><tr><td>Orbit statistic</td><td>Standard EB</td><td>γ-canonicalized EB</td></tr><tr><td>std(log λ*)</td><td>0.356</td><td>0</td></tr><tr><td> $\mathrm { r a n g e } ( \log \lambda ^ { \star } )$ </td><td>1.421</td><td>0</td></tr><tr><td>Spread factor of  $\lambda ^ { \star }$ </td><td>4.14×</td><td> $1 . 0 0 \times$ </td></tr><tr><td>range(log Z)</td><td> $3 . 4 7 \times 1 0 ^ { 4 } ~ \mathrm { n a t s }$ </td><td> $< 2 \times 1 0 ^ { - 2 }$  nats</td></tr><tr><td> $\mathrm { s t d ( N L L ) }$ </td><td>0.0045</td><td>0</td></tr><tr><td> $\mathrm { s t d } ( \mathrm { E C E } )$ </td><td>0.0059</td><td>0</td></tr><tr><td> $\mathrm { m a x } _ { k , k ^ { \prime } } \| p _ { k } - p _ { k ^ { \prime } } \| _ { \infty }$ </td><td> $2 . 6 2 \times 1 0 ^ { - 1 }$ </td><td> $\leq 1 . 0 7 \times { 1 0 } ^ { - 6 }$ </td></tr><tr><td>maxk  $\| f _ { k } - f _ { 0 } \| _ { \infty }$ </td><td> $7 . 6 3 \times 1 0 ^ { - 6 }$ </td><td> $\leq 9 . 5 4 \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $\operatorname { C o r r } ( \log \lambda ^ { \star } , \| w \| ^ { 2 } )$ </td><td> $- 0 . 9 7 1$ </td><td></td></tr></table>

tion is needed for orbit-consistency rather than relying on the scale representative returned by training.

CIFAR-10 ConvNet with KFAC-Laplace. We also evaluate γ-canonicalization with all-layer KFAC-Laplace on a plain ReLU ConvNet trained on CIFAR-10. KFAC curvature satisfies the same congruence transformation law under ReLU rescaling, as shown in Appendix D, so the scalecompatibility argument applies beyond diagonal curvature. Table 3 shows the results. Over three independently trained MAP seeds, standard KFAC-Laplace is strongly underconfident. Accuracy is unchanged, but the mean predictive confidence is only 0.526 for an accuracy of 0.774. Canonicalizing before fitting KFAC-Laplace, with γ selected by marginal likelihood within the canonicalized grid, selects $\gamma = 0 . 5$ in all three runs and substantially improves predictive quality. NLL decreases from 0.877 to 0.734 and ECE from 0.248 to 0.130, with essentially unchanged accuracy. Full experimental details and additional metrics are given in Appendix E.2.

Table 3: CIFAR-10 ConvNet with all-layer KFAC-Laplace. Mean ± std over three MAP seeds.
<table><tr><td>Method</td><td>Acc.</td><td>NLL</td><td>ECE</td><td>Conf.</td></tr><tr><td>Standard KFAC-LA</td><td> $0 . 7 7 4 \pm 0 . 0 0 3$ </td><td> $0 . 8 7 7 \pm 0 . 0 2 7$ </td><td> $0 . 2 4 8 \pm 0 . 0 1 9$ </td><td> $0 . 5 2 6 \pm 0 . 0 1 9$ </td></tr><tr><td>γ-canon KFAC-LA</td><td> $0 . 7 7 3 \pm 0 . 0 0 2$ </td><td> $0 . 7 3 4 \pm 0 . 0 0 3$ </td><td> $0 . 1 3 0 \pm 0 . 0 0 6$ </td><td> $0 . 6 4 3 \pm 0 . 0 0 4$ </td></tr></table>

CIFAR-100 ResNet with last-layer Laplace. ResNet networks do not have full ReLU-scale symmetry due to batch normalization layers. Thus, γ-canonicalization is not directly applicable. However, if we consider only the last-layer Laplace approximation, we can apply γ-canonicalization to get the canonical representation. Table 4 shows the results for this. For these networks, the trained representative is already close to the canonical representative, so canonicalization does not improve the predictive performance. However, the marginal likelihood and prior precision still change substantially under rescaling, and canonicalization removes this variation to zero. While normalization layers help stabilize the scale in these networks during training, relying on them is not ideal. Relying on BatchNorm or specific weight decay values to accidentally balance the network is a fragile strategy. If a practitioner changes the optimizer, drops weight decay, or uses a novel architecture without standard normalization, standard EB-Laplace can fail silently, producing miscalibrated uncertainties. Canonicalization explicitly removes this fragility and guarantees that the Laplace fit is robust regardless of the training dynamics. These results further demonstrate that canonicalization does not modify the predictions of an already well-balanced ResNet. Canonicalization yields behavior close to standard EB Laplace when the model is already well-scaled, and corrects the calibration issues in poorly scaled networks, as shown by the CIFAR-10 experiments 3.

Table 4: Orbit-stress tests on CIFAR-100 with ResNet-18, ResNet-50, and ResNet-101 using lastlayer EB-Laplace. For each MAP, we generate $K = 1 0$ function-preserving channel rescalings with independent log $\alpha _ { i } \sim \mathcal { N } ( 0 , 1 )$ . Values are mean ± standard deviation across orbit samples.
<table><tr><td>Model</td><td>Method</td><td> $\log \lambda ^ { \star }$ </td><td> $\log Z / 1 0 ^ { 3 }$ </td><td>Acc.</td><td>NLL</td><td>ECE</td><td>Conf.</td></tr><tr><td rowspan="9">ResNet-18</td><td>MAP</td><td></td><td>一</td><td>0.7873</td><td>0.8773</td><td>0.0977</td><td>0.8849</td></tr><tr><td>Standard EB (orig.)</td><td>3.92</td><td>-10.011</td><td>0.7794</td><td>0.7906</td><td>0.0314</td><td>0.7480</td></tr><tr><td>Standard EB</td><td> $2 . 8 9 \pm 0 . 1 3$ </td><td>-27.060 ± 1.307 0.7720 ± 0.0010 1.1478 ± 0.0487</td><td></td><td></td><td> $0 . 2 7 1 1 \pm 0 . 0 2 6 0$ </td><td>0.5010 ± 0.0252</td></tr><tr><td> $\gamma = 0 . 0 0$ </td><td> $2 . 5 4 \pm 0$ </td><td> $- 1 0 . 0 8 7 \pm 0$ </td><td> $0 . 7 7 9 3 \pm 0$ </td><td> $0 . 7 9 1 8 \pm 0$ </td><td> $0 . 0 3 2 7 \pm 0$ </td><td> $0 . 7 4 6 6 \pm 0$ </td></tr><tr><td> $\dot { \gamma } = 0 . 2 5$ </td><td> $3 . 0 4 \pm 0$ </td><td> $- 1 0 . 0 4 9 \pm 0$ </td><td> $0 . 7 7 9 4 \pm 0$ </td><td> $0 . 7 9 1 2 \pm 0$ </td><td> $0 . 0 3 2 1 \pm 0$ </td><td> $0 . 7 4 7 3 \pm 0$ </td></tr><tr><td> $\dot { \gamma } = 0 . 5 0$ </td><td> $3 . 5 4 \pm 0$ </td><td> $- 1 0 . 0 3 2 \pm 0$ </td><td> $0 . 7 7 9 4 \pm 0$ </td><td> $0 . 7 9 0 9 \pm 0$ </td><td> $0 . 0 3 1 8 \pm 0$ </td><td> $0 . 7 4 7 6 \pm 0$ </td></tr><tr><td> $\dot { \gamma } = 0 . 7 5$ </td><td> $4 . 0 4 \pm 0$ </td><td> $- 1 0 . 0 3 4 \pm 0$ </td><td> $0 . 7 7 9 4 \pm 0$ </td><td> $0 . 7 9 1 0 \pm 0$ </td><td> $0 . 0 3 1 8 \pm 0$ </td><td> $0 . 7 4 7 6 \pm 0$ </td></tr><tr><td> $\dot { \gamma } = 1 . 0 0$ </td><td> $4 . 5 3 \pm 0$ </td><td> $- 1 0 . 0 5 5 \pm 0$ </td><td> $0 . 7 7 9 3 \pm 0$ </td><td> $0 . 7 9 1 2 \pm 0$ </td><td> $0 . 0 3 1 9 \pm 0$ </td><td> $0 . 7 4 7 4 \pm 0$ </td></tr><tr><td>ResNet-50 MAP</td><td></td><td></td><td>0.7917</td><td>0.8801</td><td>0.1088</td><td>0.9004</td></tr><tr><td rowspan="9"></td><td>Standard EB (orig.)</td><td>4.18</td><td>-12.421 1</td><td>0.7870</td><td>0.7592</td><td>0.0341</td><td>0.8128</td></tr><tr><td>Standard EB</td><td></td><td></td><td></td><td></td><td>3.33 ± 0.08 −37.230 ± 1.199 0.7799 ± 0.0005 1.0358 ± 0.0208 0.2169 ± 0.0128 0.5631 ± 0.0126</td><td></td></tr><tr><td> $\gamma = 0 . 0 0$ </td><td> $4 . 2 2 \pm 0$ </td><td> $- 1 2 . 7 1 5 \pm 0$ </td><td> $0 . 7 8 7 1 \pm 0$ </td><td> $0 . 7 5 9 8 \pm 0$ </td><td> $0 . 0 3 1 8 \pm 0$ </td><td> $0 . 8 0 9 6 \pm 0$ </td></tr><tr><td> $\gamma = 0 . 2 5$ </td><td>3.97 ± 0</td><td> $- 1 2 . 5 0 7 \pm 0$ </td><td> $0 . 7 8 7 0 \pm 0$ </td><td> $0 . 7 5 9 4 \pm 0$ </td><td> $0 . 0 3 3 6 \pm 0$ </td><td> $0 . 8 1 1 7 \pm 0$ </td></tr><tr><td> $\dot { \gamma } = 0 . 5 0$ </td><td> $3 . 7 1 \pm 0$ </td><td> $- 1 2 . 4 3 2 \pm 0$ </td><td> $0 . 7 8 7 0 \pm 0$ </td><td> $0 . 7 5 9 3 \pm 0$ </td><td> $0 . 0 3 3 8 \pm 0$ </td><td> $0 . 8 1 2 5 \pm 0$ </td></tr><tr><td> $\dot { \gamma } = 0 . 7 5$ </td><td>3.45 ± 0</td><td> $- 1 2 . 4 8 5 \pm 0$ </td><td> $0 . 7 8 7 1 \pm 0$ </td><td> $0 . 7 5 9 5 \pm 0$ </td><td> $0 . 0 3 3 8 \pm 0$ </td><td> $0 . 8 1 2 1 \pm 0$ </td></tr><tr><td> $\gamma = 1 . 0 0$ </td><td>3.19 ± 0</td><td> $- 1 2 . 6 6 8 \pm 0$ </td><td> $0 . 7 8 7 1 \pm 0$ </td><td> $0 . 7 5 9 9 \pm 0$ </td><td> $0 . 0 3 2 5 \pm 0$ </td><td> $0 . 8 1 0 3 \pm 0$ </td></tr><tr><td>ResNet-101 MAP</td><td></td><td></td><td>0.8105</td><td>0.8285</td><td>0.1018</td><td>0.9122</td></tr><tr><td>Standard EB (orig.)</td><td>4.17</td><td>-11.020</td><td>0.8082</td><td>0.7065</td><td>0.0195</td><td>0.8152</td></tr><tr><td rowspan="7"></td><td>Standard EB</td><td>3.34 ± 0.09</td><td> −33.887 ± 1.141 0.8014 ± 0.0006 1.0727 ± 0.0245 0.2753 ± 0.0135 0.5261 ± 0.0135</td><td></td><td></td><td></td><td></td></tr><tr><td> $\gamma = 0 . 0 0$ </td><td>4.31 ± 0</td><td> $- 1 1 . 2 7 9 \pm 0$ </td><td> $0 . 8 0 8 3 \pm 0$ </td><td> $0 . 7 0 7 6 \pm 0$ </td><td> $0 . 0 1 8 7 \pm 0$ </td><td> $0 . 8 1 1 9 \pm 0$ </td></tr><tr><td> $\gamma = 0 . 2 5$ </td><td>3.98 ± 0</td><td> $- 1 1 . 0 7 5 \pm 0$ </td><td> $0 . 8 0 8 2 \pm 0$ </td><td>0.7067 ± 0</td><td>0.0192 ± 0</td><td> $0 . 8 1 4 5 \pm 0$ </td></tr><tr><td> $\gamma = 0 . 5 0$ </td><td>3.65 ± 0</td><td> $- 1 1 . 0 1 9 \pm 0$ </td><td> $0 . 8 0 8 3 \pm 0$ </td><td> $0 . 7 0 6 5 \pm 0$ </td><td>0.0196 ± 0</td><td> $0 . 8 1 5 1 \pm 0$ </td></tr><tr><td> $\gamma = 0 . 7 5$ </td><td>3.31 ± 0</td><td> $- 1 1 . 1 0 5 \pm 0$ </td><td>0.8081 ± 0</td><td> $0 . 7 0 7 1 \pm 0$ </td><td>0.0188 ± 0</td><td> $0 . 8 1 4 0 \pm 0$ </td></tr><tr><td> $\gamma = 1 . 0 0$ </td><td>2.96 ± 0</td><td> $- 1 1 . 3 4 2 \pm 0$ </td><td> $0 . 8 0 7 9 \pm 0$ </td><td> $0 . 7 0 8 5 \pm 0$ </td><td>0.0177 ± 0</td><td> $0 . 8 1 1 0 \pm 0$ </td></tr></table>

Table 5: Standard EB model selection on CIFAR-10. Evidence values are mean ± standard deviation over 20 rescalings. Selection rates are computed over all 400 pairs.
<table><tr><td rowspan="2">σ</td><td colspan="3">Standard EB evidence (103 nats)</td><td colspan="2">Width 128 selected (%)</td></tr><tr><td>Width 64</td><td>Width 128</td><td>Gap</td><td>Both at σ</td><td>Width 128 at 0.25</td></tr><tr><td>0.25</td><td> $- 1 1 0 . 6 \pm 0 . 8$ </td><td> $- 1 2 1 . 7 \pm 0 . 7$ </td><td>11.1</td><td>0.0</td><td>0.0</td></tr><tr><td>0.50</td><td> $- 1 1 9 . 3 \pm 2 . 9$ </td><td> $- 1 3 6 . 4 \pm 2 . 7$ </td><td>17.1</td><td>0.0</td><td>19.8</td></tr><tr><td>1.00</td><td> $- 1 6 0 . 6 \pm { 1 1 . 9 }$ </td><td> $- 2 1 4 . 3 \pm 1 6 . 9$ </td><td>53.7</td><td>0.2</td><td>100.0</td></tr></table>

Effect on model selection. We compare two MLPs trained on CIFAR-10, with widths 64 and 128. Marginal likelihood computed using canonical representation favors the width-64 model by $1 2 . 4 \times 1 0 ^ { 3 }$ nats. For each ReLU rescaling strength $\sigma \in \{ 0 . 2 5 , 0 . 5 , 1 . 0 \}$ (scale log $\alpha \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) \dot { ) }$ we create 20 versions of each model without changing its network predictions and refit standard EB-Laplace, and compare all 400 pairs. As Table 5 shows, the mean evidence gap changes from $1 1 . 1 \times 1 0 ^ { 3 }$ to $5 3 . 7 \times \mathrm { { \dot { 1 } } 0 ^ { 3 } }$ nats. Using the same rescaling strength for both models rarely changes their ranking, because the wider model loses more evidence in this example. However, when the width-128 model is kept at $\sigma = 0 . 2 5$ , increasing the rescaling strength of the width-64 model reverses the canonical ranking in 19.8% of comparisons at $\sigma = 0 . 5$ and 100% at $\sigma = 1 . 0$ . Thus, standard EB can change its model selection even though neither network’s predictions change. Canonicalization keeps the evidence gap and model ranking fixed across these rescalings.

## 8 Related Work

Laplace approximation and empirical-Bayes prior selection. The Laplace approximation is a classical tool for Bayesian model comparison, with early evidence-based treatments established in the literature [MacKay, 1992]. Subsequent work makes it practical in neural networks through structured curvature approximations and scalable software [Ritter et al., 2018, Daxberger et al., 2021]. Empirical-Bayes selection of prior precision by marginal-likelihood maximisation has also been previously developed [Immer et al., 2021a] and extended to linearized settings [Antoran et al., 2022a]. Our work does not propose a new curvature approximation or scalable implementation. It identifies a structural defect in the standard empirical-Bayes objective. It depends on the arbitrary parameter representative. We construct a corrected version that is well-defined on ReLU scale-equivalence classes.

Classical Laplace theory and Singular models. Standard Laplace and Bayesian information criterion (BIC) asymptotics assume local identifiability and nonsingular quadratic expansion around an isolated mode [Tierney and Kadane, 1986, Kass et al., 1990]. ReLU networks violate these likelihood-level assumptions because of scale symmetry. Moreover, for deep neural networks, the Hessian need not be positive definite. A Gaussian prior can make the posterior curvature positive definite, allowing a finite local Laplace approximation to be computed. However, the usual regularmodel asymptotic interpretation of such approximation is no longer valid. MacKay [1998] has shown that the accuracy of a Laplace approximation can depend on the chosen parameter basis. For non-identifiable models, the regular BIC expansion is invalid. The asymptotic expansion of the log marginal likelihood instead involves the log canonical threshold and its multiplicity, which replace the regular complexity term and may introduce an additional contribution [Watanabe, 2009]. Criteria such as wBIC [Watanabe, 2013] and sBIC [Drton and Plummer, 2017] are designed to account for this singular asymptotic behavior. We study a separate finite-sample defect of the empirical-Bayes Laplace surrogate used in practice. Our method does not correct the full singular asymptotics of neural-network marginal likelihoods. Rather, it makes that surrogate well-defined on ReLU scale orbit.

Scale symmetries and canonical representatives. Positive homogeneity of ReLU activations generates continuous families of parameter vectors representing the same function. These symmetries have been studied extensively in the context of norm-based capacity control and path-normalized optimization [Neyshabur et al., 2015a,b]. Furthermore, it has been shown that ReLU reparameterizations change sharpness measures without changing the represented function [Dinh et al., 2017], establishing that coordinate-dependent quantities are not intrinsic properties of the learned predictor. Our paper applies the same principle to empirical-Bayes Laplace. The evidence and selected prior precision should not depend on an arbitrary representative. To construct a canonical representative, we build on iterative balancing procedures that minimize the weight norm over a scale orbit [Stock et al., 2019, Saul, 2023]. Our γ-canonicalization generalizes these procedures. Rather than using balancing to improve optimization, our method uses it to define where an isotropic Gaussian prior is placed on each orbit, turning balancing into a mechanism for scale-consistent empirical-Bayes inference.

Reparameterization invariance in approximate Bayesian inference. The most directly related work studies reparameterization invariance in approximate Bayesian inference, showing that linearized Laplace behaves differently from sampled Laplace in directions that do not change the function [Roy et al., 2024]. A related argument suggests that Hessian-based quantities become inconsistent under reparameterization when the relevant metric structure is ignored [Kristiadi et al., 2023]. This issue has been addressed by proposing a Riemannian Laplace approximation that adapts the posterior geometry to the local shape of the log-posterior [Bergamin et al., 2023]. Our contribution is narrower and more specific than these geometric approaches. We focus exclusively on the exact continuous scale symmetry of ReLU networks and derive the precise transformation law the empirical-Bayes prior precision must satisfy for both the evidence and the linearized predictive distribution to be invariant on scale orbits. Antoran et al. [2022b] study Laplace inference in networks with normalisation layers and show that BatchNorm and related layers break the posterior-volume interpretation of the standard evidence, proposing a neural g-prior to address heterogeneous Jacobian scales. Our work addresses a different invariance failure in plain ReLU networks, where the empirical-Bayes evidence and selected prior precision depend on the arbitrary representative of a ReLU scale orbit.

Weight-space symmetries in Bayesian neural networks. Discrete weight-space symmetries, such as hidden-unit permutations, have received attention in Bayesian neural networks. Previous studies analyze how permutation symmetries affect posteriors and propose constructions that respect permutation invariance [Rossi et al., 2023, Gelberg et al., 2024]. Other works explore both permutation and scale symmetries in weight-space posteriors [Laurent et al., 2024]. Our work is complementary. We address the continuous ReLU scale symmetry rather than discrete permutation symmetry, and focus specifically on the empirical-Bayes evidence objective and prior precision selection rather than on the posterior geometry itself.

## 9 Limitations and future work

γ-canonicalization is only applicable to neural networks having activation functions with positive homogeneity property such as ReLU, leaky ReLU, PReLU, etc. Moreover, normalization layers such as batch normalization or layer normalization break the scale symmetry. Thus, γ-canonicalization will not work on the network having normalization layers. Extending our method to work on normalized networks would require deriving the appropriate symmetry group, canonical section, and transported prior for that architecture. We apply γ-canonicalization post-hoc after the network has been trained. It would be interesting to see how training dynamics change if we apply γ-canonicalization during training. We leave such investigations to future work.

## 10 Conclusion

Standard EB Laplace approximation for ReLU networks is not invariant under ReLU scale symmetry. It gives different marginal likelihoods, prior precisions, and predictive distributions for functionpreserving ReLU rescalings. We introduce γ-canonicalized EB Laplace approximation that fixes the ReLU scale dependence issue of standard EB Laplace. It selects a canonical representation and places an isotropic Gaussian prior on that, which is equivalent to transporting the isotropic Gaussian prior from the canonical representation to the original coordinates by producing a precision field. This precision field transforms compatibly with curvature, which makes the generalized marginal likelihood, prior precision, and predictive distribution invariant under ReLU rescaling. Moreover, Invariance of the marginal likelihood makes model selection via EB Laplace well-defined. Although our guarantee is for invariance, if the MAP solution has poor scale, γ-canonicalization can also give improved uncertainty estimates, as shown in our experiments.

## Acknowledgments and Disclosure of Funding

SP acknowledges support from Visvesvaraya PhD fellowship. PR acknowledges support from Google Research and Satya & Rao Remala Foundation.

## References

Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1019–1028. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/ dinh17b.html.

David J. C. MacKay. Bayesian interpolation. Neural Computation, 4(3):415–447, 05 1992. ISSN 0899-7667. doi: 10.1162/neco.1992.4.3.415. URL https://doi.org/10.1162/neco.1992.4. 3.415.

Hippolyt Ritter, Aleksandar Botev, and David Barber. A scalable laplace approximation for neural networks. In International Conference on Learning Representations, 2018. URL https:// openreview.net/forum?id=Skdvd2xAZ.

Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace redux - effortless bayesian deep learning. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=gDcaUj4Myhn.

Alexander Immer, Matthias Bauer, Vincent Fortuin, Gunnar Rätsch, and Mohammad Emtiyaz Khan. Scalable marginal likelihood estimation for model selection in deep learning. In ICML, pages 4563–4573, 2021a. URL http://proceedings.mlr.press/v139/immer21a.html.

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Peter Grünwald, Elad Hazan, and Satyen Kale, editors, Proceedings of The 28th Conference on Learning Theory, volume 40 of Proceedings of Machine Learning Research, pages 1376–1401, Paris, France, 03–06 Jul 2015a. PMLR. URL https://proceedings.mlr.press/ v40/Neyshabur15.html.

Behnam Neyshabur, Ruslan Salakhutdinov, and Nathan Srebro. Path-sgd: Path-normalized optimization in deep neural networks. In NIPS, pages 2422–2430, 2015b. URL http://papers.nips.cc/ paper/5797-path-sgd-path-normalized-optimization-in-deep-neural-networks.

Pierre Stock, Benjamin Graham, Rémi Gribonval, and Hervé Jégou. Equi-normalization of neural networks. In International Conference on Learning Representations, 2019. URL https:// openreview.net/forum?id=r1gEqiC9FX.

Lawrence K. Saul. Weight-balancing fixes and flows for deep learning. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id= uaHyXxyp2r.

Alexander Immer, Maciej Korzepa, and Matthias Bauer. Improving predictions of bayesian neural nets via local linearization. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 703–711. PMLR, 13–15 Apr 2021b. URL https://proceedings.mlr.press/v130/immer21a.html.

Javier Antoran, David Janz, James U Allingham, Erik Daxberger, Riccardo Rb Barbano, Eric Nalisnick, and Jose Miguel Hernandez-Lobato. Adapting the linearised Laplace model evidence for modern deep learning. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 796–821. PMLR, 17–23 Jul 2022a. URL https://proceedings.mlr.press/v162/antoran22a.html.

Bo Zhao, Robin Walters, and Rose Yu. Symmetry in neural network parameter spaces. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/ forum?id=jLpWq5QY6I.

Robert Hecht-Nielsen. On the algebraic structure of feedforward network weight spaces. In Rolf ECKMILLER, editor, Advanced Neural Computers, pages 129–135. North-Holland, Amsterdam, 1990. ISBN 978-0-444-88400-8. doi: https://doi.org/10.1016/B978-0-444-88400-8.50019-4. URL https://www.sciencedirect.com/science/article/pii/B9780444884008500194.

Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=dNigytemkL.

Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=CQsmMYmlP5T.

Luke Tierney and Joseph B. Kadane. Accurate approximations for posterior moments and marginal densities. Journal of the American Statistical Association, 81(393):82–86, 1986. doi: 10.1080/01621459.1986.10478240. URL https://www.tandfonline.com/doi/abs/ 10.1080/01621459.1986.10478240.

Robert E. Kass, Luke Tierney, and Joseph B. Kadane. The validity of posterior expansions based on laplace’s method. In S. Geisser, J. S. Hodges, S. J. Press, and A. Zellner, editors, Bayesian and Likelihood Methods in Statistics and Econometrics: Essays in Honor of George A. Barnard, pages 473–488. North-Holland, Amsterdam, 1990.

David J. C. MacKay. Choice of basis for laplace approximation. Machine Learning, 33(1):77–86, 1998. doi: 10.1023/A:1007558615313. URL https://doi.org.

Sumio Watanabe. Algebraic Geometry and Statistical Learning Theory. Cambridge University Press, USA, 2009. ISBN 0521864674.

Sumio Watanabe. A widely applicable bayesian information criterion. Journal ofMachine Learning Research, 14(27):867–897, 2013. URL http://jmlr.org/papers/v14/watanabe13a.html.

Mathias Drton and Martyn Plummer. A bayesian information criterion for singular models. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 79(2):323–380, 03 2017. ISSN 1369-7412. doi: 10.1111/rssb.12187. URL https://doi.org/10.1111/rssb.12187.

Hrittik Roy, Marco Miani, Carl Henrik Ek, Philipp Hennig, Marvin Pförtner, Lukas Tatzel, and Søren Hauberg. Reparameterization invariance in approximate bayesian inference. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https:// openreview.net/forum?id=204YOrDHny.

Agustinus Kristiadi, Felix Dangel, and Philipp Hennig. The geometry of neural nets’ parameter spaces under reparametrization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=vtLNwa6uX0.

Federico Bergamin, Pablo Moreno-Muñoz, Søren Hauberg, and Georgios Arvanitidis. Riemannian laplace approximations for bayesian neural networks. In Thirty-seventh Conference on Neural Infor mation Processing Systems, 2023. URL https://openreview.net/forum?id=bHS7qjLOAy.

Javier Antoran, James Urquhart Allingham, David Janz, Erik Daxberger, Eric Nalisnick, and José Miguel Hernández-Lobato. Linearised laplace inference in networks with normalisation layers and the neural g-prior. In Fourth Symposium on Advances in Approximate Bayesian Inference, 2022b. URL https://openreview.net/forum?id=uUH8x-h9zdB.

Simone Rossi, Ankit Singh, and Thomas Hannagan. On permutation symmetries in bayesian neural network posteriors: a variational perspective. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=kj33zJ9Vue.

Yoav Gelberg, Tycho F. A. van der Ouderaa, Mark van der Wilk, and Yarin Gal. Variational inference failures under model symmetries: Permutation invariant posteriors for bayesian neural networks. In Sharvaree Vadgama, Erik Bekkers, Alison Pouplin, Sekou-Oumar Kaba, Robin Walters, Hannah Lawrence, Tegan Emerson, Henry Kvinge, Jakub Tomczak, and Stephanie Jegelka, editors, Proceedings ofthe Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM), volume 251 of Proceedings ofMachine Learning Research, pages 233–248. PMLR, 29 Jul 2024. URL https://proceedings.mlr.press/v251/gelberg24a.html.

Olivier Laurent, Emanuel Aldea, and Gianni Franchi. A symmetry-aware exploration of bayesian neural network posteriors. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=FOSBQuXgAq.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017. URL https://arxiv.org/abs/1708.07747.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/guo17a. html.

## A Proof of Proposition 4.2(Scalar Orbit Dependence)

Model. Consider f(x; $a , c ) = c \mathrm { R e L U } ( a x )$ with $a , c > 0$ , Gaussian observation noise of variance $\sigma _ { y } ^ { 2 } .$ , and inputs $x _ { n } > 0$ . Since $a > 0$ ${ \mathrm { R e L U } } ( a x _ { n } ) = a x _ { n } , { \mathrm { s o ~ } } f ( x _ { n } ; a , c ) = a c x _ { n }$ . The rescaling $( \dot { a } , c ) \mapsto ( \alpha a , c / \alpha )$ preserves the realized predictor on the dataset for every $\alpha > 0$

Curvature. The Jacobian at $( a , c )$ for a single input $x _ { n }$ is $\mathbf { J } _ { n } ( a , c ) = ( c x _ { n } , a x _ { n } )$ . For Gaussian regression with noise variance $\begin{array} { r } { \dot { \sigma } _ { y } ^ { 2 } , } \end{array}$ the GGN matrix is

$$
\mathbf { G } ( a , c ) = \sum _ { n = 1 } ^ { N } { \frac { 1 } { \sigma _ { y } ^ { 2 } } } \mathbf { J } _ { n } ^ { \top } \mathbf { J } _ { n } = { \frac { S } { \sigma _ { y } ^ { 2 } } } \left( { c ^ { 2 } } _ { a c } a c \right) , \qquad S = \sum _ { n = 1 } ^ { N } x _ { n } ^ { 2 } .\tag{20}
$$

This matrix has rank one with nonzero eigenvalue $\mu ( \alpha ) = \kappa q ( \alpha )$ , where $\kappa = S / \sigma _ { y } ^ { 2 }$ and $q ( \alpha ) =$ $( \alpha a ) ^ { 2 } + ( c / \alpha ) ^ { 2 }$

Evidence along the orbit. Along the scale orbit, the likelihood is constant and $\| ( a ( \alpha ) , c ( \alpha ) ) \| ^ { 2 } =$ $q ( \alpha )$ . The eigenvalues of $\mathbf { G } ( \alpha ) + \lambda \mathbf { I }$ are λ and $\mu ( \alpha ) + \lambda , \operatorname { s o } | \mathbf { G } ( \alpha ) + \lambda \mathbf { I } | = \lambda \ddot { ( } \mu ( \alpha ) + \lambda )$ . Up to constants independent of λ and $\alpha .$ , the standard empirical-Bayes objective reduces to

$$
\mathcal { L } _ { \mathrm { s t d } } ( \lambda ; \alpha ) = \frac { 1 } { 2 } \log \frac { \lambda } { \kappa q ( \alpha ) + \lambda } - \frac { \lambda } { 2 } q ( \alpha ) .\tag{21}
$$

Empirical-Bayes optimum. Differentiating (21) with respect to $\lambda$ and setting the derivative to zero gives

$$
{ \frac { 1 } { \lambda } } - { \frac { 1 } { \kappa q ( \alpha ) + \lambda } } = q ( \alpha ) .\tag{22}
$$

Combining fractions,

$$
\frac { \kappa } { \lambda ( \kappa q ( \alpha ) + \lambda ) } = 1 .\tag{23}
$$

Since $q ( \alpha ) > 0$ , this simplifies to $\lambda ( \kappa q ( \alpha ) + \lambda ) = \kappa ,$ or equivalently

$$
\lambda ^ { 2 } + \kappa q ( \alpha ) \lambda - \kappa = 0 .\tag{24}
$$

The unique positive root is

$$
\lambda ^ { \star } ( \alpha ) = \frac { - \kappa q ( \alpha ) + \sqrt { \kappa ^ { 2 } q ( \alpha ) ^ { 2 } + 4 \kappa } } { 2 } .\tag{25}
$$

Non-invariance. The function $q ( \alpha ) = ( \alpha a ) ^ { 2 } + ( c / \alpha ) ^ { 2 }$ is strictly convex in log $\alpha ,$ minimized at $\alpha ^ { 2 } = c / a$ , and unbounded as $\alpha  0 \mathrm { o r } \alpha  \infty$ . Since $\lambda ^ { \star }$ is a strictly decreasing function of $q$ (its derivative with respect to q is negative for all $q > 0 ) , \lambda ^ { \star } ( \alpha )$ is non-constant along the orbit. Hence standard empirical-Bayes Laplace assigns different marginal likelihoods, selected prior precisions, and linearized predictive uncertainties to parameter vectors representing exactly the same predictor. □

## B Proof of theorem 5.2 (Evidence invariance)

Proof. The likelihood term is invariant because v and w represent the same predictor. For the posterior precision,

$$
\begin{array} { r } { \mathbf { C } _ { \mathbf { v } } + \pmb { \Lambda } ( \mathbf { v } ) = \mathbf { D } _ { \alpha } ^ { - 1 } \big ( \mathbf { C } _ { \mathbf { w } } + \pmb { \Lambda } ( \mathbf { w } ) \big ) \mathbf { D } _ { \alpha } ^ { - 1 } . } \end{array}
$$

Hence

$$
| \mathbf { C _ { v } } + \mathbf { \mathbf { \lambda } } ( \mathbf { v } ) | = | \mathbf { D } _ { \alpha } | ^ { - 2 } \left| \mathbf { C _ { w } } + \mathbf { \lambda } ( \mathbf { w } ) \right| .
$$

Similarly,

$$
| \mathbf { A } ( \mathbf { v } ) | = | \mathbf { D } _ { \alpha } | ^ { - 2 } | \mathbf { A } ( \mathbf { w } ) | .
$$

The two log-determinant changes therefore cancel in (9). Finally,

$$
\begin{array} { r l } & { \mathbf { v } ^ { \top } \mathbf { \boldsymbol { \Lambda } } ( \mathbf { v } ) \mathbf { v } = ( \mathbf { D } _ { \alpha } \mathbf { w } ) ^ { \top } \mathbf { D } _ { \alpha } ^ { - 1 } \mathbf { \boldsymbol { \Lambda } } ( \mathbf { w } ) \mathbf { D } _ { \alpha } ^ { - 1 } ( \mathbf { D } _ { \alpha } \mathbf { w } ) } \\ & { \qquad = \mathbf { w } ^ { \top } \mathbf { \boldsymbol { \Lambda } } ( \mathbf { w } ) \mathbf { w } . } \end{array}
$$

## C Proof of Proposition 5.4 (Scale Compatibility of the Transported Prior)

Proof. Let $\mathbf { v } = T _ { \alpha } ( \mathbf { w } ) = \mathbf { D } _ { \alpha } \mathbf { w }$ . By the section property, $C _ { \gamma } ( \mathbf { v } ) = C _ { \gamma } ( \mathbf { w } )$ . Using the construction in Section (5.3),

$$
\begin{array} { r } { \mathbf { v } = \mathbf { D } _ { \alpha } \mathbf { w } = \mathbf { D } _ { \alpha } \mathbf { D } ( \mathbf { w } ; \boldsymbol { \gamma } ) C _ { \boldsymbol { \gamma } } ( \mathbf { v } ) . } \end{array}
$$

Therefore

$$
\begin{array} { r } { \mathbf { D } ( \mathbf { v } ; \gamma ) = \mathbf { D } _ { \alpha } \mathbf { D } ( \mathbf { w } ; \gamma ) . } \end{array}
$$

Since these matrices are diagonal,

$$
\begin{array} { r l } & { \mathbf { \boldsymbol { \Lambda } } _ { \gamma } ( \mathbf { v } ; \lambda ) = \lambda \mathbf { \boldsymbol { D } } ( \mathbf { v } ; \gamma ) ^ { - 2 } } \\ & { \quad \quad \quad = \lambda \big ( \mathbf { \boldsymbol { D } } _ { \alpha } \mathbf { \boldsymbol { D } } ( \mathbf { w } ; \gamma ) \big ) ^ { - 2 } } \\ & { \quad \quad \quad = \mathbf { \boldsymbol { D } } _ { \alpha } ^ { - 1 } \big ( \lambda \mathbf { \boldsymbol { D } } ( \mathbf { w } ; \gamma ) ^ { - 2 } \big ) \mathbf { \boldsymbol { D } } _ { \alpha } ^ { - 1 } } \\ & { \quad \quad \quad = \mathbf { \boldsymbol { D } } _ { \alpha } ^ { - 1 } \mathbf { \boldsymbol { \Lambda } } _ { \gamma } ( \mathbf { w } ; \lambda ) \mathbf { \boldsymbol { D } } _ { \alpha } ^ { - 1 } . } \end{array}\tag{26}
$$

## D KFAC Congruence under ReLU Rescaling

We verify that standard KFAC curvature blocks obey the same congruence law as Jacobian-based curvature under the ReLU rescalings of Section 3.1. Consider an affine layer

$$
\begin{array} { r } { \mathbf { z } _ { \ell } = \mathbf { W } _ { \ell } \mathbf { a } _ { \ell - 1 } + \mathbf { b } _ { \ell } , \qquad \mathbf { a } _ { \ell } = \phi ( \mathbf { z } _ { \ell } ) . } \end{array}
$$

Let

$$
\mathbf { D } _ { \ell } = \mathrm { d i a g } ( \alpha _ { \ell , 1 } , . . . , \alpha _ { \ell , d _ { \ell } } )
$$

collect the positive hidden-unit scaling factors at layer ℓ, with $\mathbf { D } _ { 0 } = \mathbf { D } _ { L } = \mathbf { I }$ because input and output units are not rescaled. The ReLU parameter rescaling acts layerwise as

$$
\mathbf { W } _ { \ell } ^ { \prime } = \mathbf { D } _ { \ell } \mathbf { W } _ { \ell } \mathbf { D } _ { \ell - 1 } ^ { - 1 } , \qquad \mathbf { b } _ { \ell } ^ { \prime } = \mathbf { D } _ { \ell } \mathbf { b } _ { \ell } .
$$

This is the layerwise matrix form of the elementary hidden-unit rescaling in Eq. 3. With column-major vectorization,

$$
\begin{array} { r } { \mathrm { v e c } ( \mathbf { W } _ { \ell } ^ { \prime } ) = \mathbf { S } _ { \ell } \mathrm { v e c } ( \mathbf { W } _ { \ell } ) , \qquad \mathbf { S } _ { \ell } = \mathbf { D } _ { \ell - 1 } ^ { - 1 } \otimes \mathbf { D } _ { \ell } , } \end{array}
$$

so $\mathbf { S } _ { \ell }$ is the corresponding parameter-space block of the global rescaling matrix $\mathbf { D } _ { \alpha }$

The parameter rescaling induces the activation transformation

$$
\begin{array} { r } { \mathbf { z } _ { \ell } ^ { \prime } = \mathbf { D } _ { \ell } \mathbf { z } _ { \ell } , \qquad \mathbf { a } _ { \ell } ^ { \prime } = \mathbf { D } _ { \ell } \mathbf { a } _ { \ell } , } \end{array}
$$

where the second identity follows from positive homogeneity, $\phi ( \mathbf { D } _ { \ell } \mathbf { z } _ { \ell } ) = \mathbf { D } _ { \ell } \phi ( \mathbf { z } _ { \ell } )$ . For the loss derivative

$$
\pmb { \delta } _ { \ell } = \partial \mathcal { L } / \partial \mathbf { z } _ { \ell } ,
$$

the chain rule gives

$$
\begin{array} { r } { \delta _ { \ell } ^ { \prime } = \mathbf { D } _ { \ell } ^ { - 1 } \delta _ { \ell } . } \end{array}
$$

With column-major vectorization, the KFAC approximation to the weight block is

$$
\begin{array} { r } { \mathbf { C } _ { \ell } ^ { \mathrm { K F A C } } = \mathbf { A } _ { \ell - 1 } \otimes \mathbf { G } _ { \ell } , \qquad \mathbf { A } _ { \ell - 1 } = \mathbb { E } \left[ \mathbf { a } _ { \ell - 1 } \mathbf { a } _ { \ell - 1 } ^ { \top } \right] , \qquad \mathbf { G } _ { \ell } = \mathbb { E } \left[ \delta _ { \ell } \delta _ { \ell } ^ { \top } \right] . } \end{array}
$$

The transformations above imply

$$
\mathbf { A } _ { \ell - 1 } ^ { \prime } = \mathbf { D } _ { \ell - 1 } \mathbf { A } _ { \ell - 1 } \mathbf { D } _ { \ell - 1 } , \qquad \mathbf { G } _ { \ell } ^ { \prime } = \mathbf { D } _ { \ell } ^ { - 1 } \mathbf { G } _ { \ell } \mathbf { D } _ { \ell } ^ { - 1 } .
$$

Therefore, by the mixed-product property of the Kronecker product,

$$
\mathbf { C } _ { \ell } ^ { \mathrm { K F A C \prime } } = \mathbf { A } _ { \ell - 1 } ^ { \prime } \otimes \mathbf { G } _ { \ell } ^ { \prime } = ( \mathbf { D } _ { \ell - 1 } \otimes \mathbf { D } _ { \ell } ^ { - 1 } ) \mathbf { C } _ { \ell } ^ { \mathrm { K F A C } } ( \mathbf { D } _ { \ell - 1 } \otimes \mathbf { D } _ { \ell } ^ { - 1 } ) = \mathbf { S } _ { \ell } ^ { - 1 } \mathbf { C } _ { \ell } ^ { \mathrm { K F A C } } \mathbf { S } _ { \ell } ^ { - 1 } .
$$

This is the layerwise form of the global congruence law

$$
\begin{array} { r } { \mathbf { C } _ { v } = \mathbf { D } _ { \alpha } ^ { - 1 } \mathbf { C } _ { w } \mathbf { D } _ { \alpha } ^ { - 1 } . } \end{array}
$$

The KFAC factorization is compatible with the rescaling because the factorized expectation commutes with the induced activation and derivative transformations:

$$
\begin{array} { r } { \mathbb { E } \left[ \mathbf { a } _ { \ell - 1 } ^ { \prime } \mathbf { a } _ { \ell - 1 } ^ { \prime \top } \right] \otimes \mathbb { E } \left[ \pmb { \delta } _ { \ell } ^ { \prime } \pmb { \delta } _ { \ell } ^ { \prime \top } \right] = ( \mathbf { D } _ { \ell - 1 } \otimes \mathbf { D } _ { \ell } ^ { - 1 } ) \left( \mathbb { E } \left[ \mathbf { a } _ { \ell - 1 } \mathbf { a } _ { \ell - 1 } ^ { \top } \right] \otimes \mathbb { E } \left[ \delta _ { \ell } \pmb { \delta } _ { \ell } ^ { \top } \right] \right) ( \mathbf { D } _ { \ell - 1 } \otimes \mathbf { D } _ { \ell } ^ { - 1 } ) . } \end{array}
$$

Biases are handled by augmenting the activation vector,

$$
\begin{array} { r } { \bar { \mathbf { a } } _ { \ell - 1 } = ( \mathbf { a } _ { \ell - 1 } ^ { \top } , 1 ) ^ { \top } , \qquad \bar { \mathbf { D } } _ { \ell - 1 } = \mathrm { d i a g } ( \mathbf { D } _ { \ell - 1 } , 1 ) . } \end{array}
$$

The appended constant coordinate is left unscaled because it is not a hidden unit in the ReLU scale group, while the bias column itself scales with $\mathbf { D } _ { \ell } .$ Equivalently, for the augmented weight matrix $\mathbf { \bar { W } } _ { \ell } = [ \mathbf { W } _ { \ell } \mathbf { \Delta } \mathbf { b } _ { \ell } ]$

$$
\bar { \mathbf { W } } _ { \ell } ^ { \prime } = \mathbf { D } _ { \ell } \bar { \mathbf { W } } _ { \ell } \bar { \mathbf { D } } _ { \ell - 1 } ^ { - 1 } .
$$

The same congruence argument then applies to the augmented weight-bias block.

For convolutional layers in the standard unfolded-patch KFAC formulation, the proof carries over after replacing $\mathbf { a } _ { \ell - 1 }$ by the unfolded input patches and applying the corresponding channel scaling to each spatial patch coordinate.

## E Experimental Details

Our experiments are fully reproducible. All the experiments are conducted on a single NVIDIA A30 24GB GPU. Code is available in the supplementary material.

## E.1 Orbit Stress Test

Dataset and model We use Fashion-MNIST Xiao et al. [2017], with 60,000 training images and 10,000 test images across 10 classes. Images are normalized using the training-set mean 0.2860 and standard deviation 0.3530, and are flattened to vectors of dimension 784. The model is a two-hiddenlayer ReLU MLP with hidden dimensions [64, 64] and a linear output layer. It has no normalization layers.

MAP training The MAP solution is obtained by minimizing cross-entropy with $\ell _ { 2 }$ weight decay $1 0 ^ { - 4 }$ using Adam with learning rate $1 0 ^ { - 3 }$ , batch size 256, for 10 epochs. The resulting MAP model achieves 87.67% test accuracy.

Laplace approximation We use laplace-torch Daxberger et al. [2021]. All Laplace fits use all parameters, diagonal GGN curvature, and a scalar prior precision λ. The prior precision is initialized at 1.0 and optimized by 100 Adam steps with learning rate 0.1 on the Laplace marginal likelihood. For prediction, we use the Jacobian-linearized predictive distribution with the probit link approximation. Model computations use float32; γ-canonicalization is performed in float64 and cast back to float32 before fitting Laplace.

Orbit generation Starting from the MAP solution $\hat { w } .$ , we generate $K = 2 0$ function-preserving representatives. For each representative, we sample independent hidden-unit log-scales

$$
\log \alpha _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } { \mathcal { N } } ( 0 , \sigma ^ { 2 } ) , \quad \sigma = 1 . 0 ,
$$

and apply the elementary ReLU rescaling in Eq. 3 to every hidden unit. For every generated representative $v _ { k }$ , we verify

$$
\operatorname* { m a x } _ { x \in \mathcal { D } _ { \mathrm { t e s t } } } \left\| f ( x ; \hat { w } ) - f ( x ; v _ { k } ) \right\| _ { \infty } < 1 0 ^ { - 5 } .
$$

Thus all reported variation across orbit samples is due to the parameterization supplied to the Laplace approximation, not to any change in the predictor.

γ-canonicalization For each representative $v _ { k }$ and each $\gamma \in \{ 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 \}$ , we apply Algorithm 1 with tolerance $\varepsilon = \bar { 1 0 } ^ { - 6 }$ and a maximum budget of 200 sweeps. For interior values, all runs converged in fewer than 50 sweeps under the stopping criterion ma $\mathrm { x } _ { \ell , i } \vert \log \alpha _ { \ell , i } \vert < 1 0 ^ { - 6 }$ . The endpoints are computed by a single directional pass: forward incoming-normalization for $\gamma = 0$ and backward outgoing-normalization for $\gamma = 1$ . The canonicalization cost is much smaller than the cost of fitting the Laplace approximation.

Table 6: Standard deviations under rescaling and retraining. Rescaling preserves the network function while retraining produces different learned functions. Predictive metrics use standard EB-Laplace.
<table><tr><td>Quantity</td><td>Rescaling</td><td>Retraining</td></tr><tr><td>Log prior precision log λ*</td><td>0.356</td><td>0.016</td></tr><tr><td>Log evidence log Z</td><td>8900</td><td>884</td></tr><tr><td>NLL</td><td>0.0045</td><td>0.0088</td></tr><tr><td>ECE</td><td>0.0059</td><td>0.0046</td></tr><tr><td>Accuracy</td><td>0.0008</td><td>0.0040</td></tr><tr><td>Squared parameter norm  $\| \boldsymbol { w } \| _ { 2 } ^ { 2 }$ </td><td>1016.0</td><td>4.7</td></tr></table>

Metrics We report the following quantities. NLL is the mean negative log-likelihood of the Laplace predictive distribution on the test set. ECE is the expected calibration error with 20 equal-width confidence bins Guo et al. [2017]. The quantity log $\dot { Z }$ is the Laplace marginal likelihood at the selected $\lambda ^ { \star }$ . The norm $\| \boldsymbol { w } \| ^ { 2 }$ is the squared Euclidean norm of the parameter vector supplied to Laplace; for standard EB this is the raw (possibly rescaled) representative, and for γ-canonicalized EB it is the canonical representative. The statistic max $\| \Delta p \| _ { \infty }$ is the maximum absolute difference in predicted class probability between any two orbit samples, computed over all $\binom { K } { 2 }$ pairs, all test inputs, and all classes. The spread factor is exp max<sub>k</sub> log $\lambda _ { k } ^ { \star } - \operatorname* { m i n } _ { k } \log \lambda _ { k } ^ { \star } \big )$ .

## E.2 CIFAR-10 ConvNet with KFAC-Laplace

Model and training. We use a plain ReLU ConvNet without normalization layers. The channel widths are (64, 128, 256, 256). The network consists of four convolutional blocks followed by global average pooling and a linear classifier. Each convolutional block contains a $3 \times 3$ convolution followed by a ReLU nonlinearity; spatial downsampling is performed by avg pooling. The model is trained on CIFAR-10 using Adam with learning rate $1 \overline { { 0 } } ^ { - 3 }$ and weight decay $\mathrm { \dot { 1 } 0 ^ { - 5 } }$ for 50 epochs. We use a cosine learning-rate schedule.

Laplace approximation. We fit all-layer empirical-Bayes Laplace with KFAC curvature using laplace-torch. The prior precision is optimized by marginal likelihood using method $\mathbf { \tau } ^ { \mathsf { m a r g l i k } ^ { \mathsf { i } } }$ Predictions use the Jacobian-linearized predictive distribution with the probit link approximation. For the canonicalized variant, we first apply γ-canonicalization to the trained MAP solution and then fit the same KFAC-Laplace approximation. We evaluate $\gamma \in \{ 0 , 0 . 1 , 0 . 2 , \ldots , 1 . 0 \}$ and select γ by marginal likelihood within the canonicalized family. In all three seeds, the selected value is $\gamma = 0 . 5$

## F Within-orbit vs. between-solution variation

We compare variation between 20 function-preserving rescalings of one trained network $( \sigma = 1 )$ , and 20 independently trained networks with the same architecture and training procedure. We fit standard EB-Laplace to every rescaled or independently trained network. Table 6 shows that the standard deviation of log $\lambda ^ { \star }$ is approximately 22 times larger under rescaling than retraining. For log $Z ,$ it is approximately 10 times larger. Predictive metrics also vary under rescaling, although retraining produces larger variation in NLL and accuracy. Variation across training runs can reflect different learned functions. In contrast, variation under function-preserving rescaling arises entirely from the parameter scales supplied to the Laplace fit. These results show that arbitrary scale choices can affect the selected prior precision and evidence more than independent retraining.