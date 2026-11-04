# Variational Learning with Lie Group Exponential Families

Eren Mehmet Kıral RIKEN Center for Advanced Intelligence Project 1-4-1 Nihonbashi, Chuo-ku, Tokyo 103-0027, Japan

Thomas M¨ollenhof RIKEN Center for Advanced Intelligence Project 1-4-1 Nihonbashi, Chuo-ku, Tokyo 103-0027, Japan

Koichi Tojo Department of Mathematical Sciences, Tokai University, 4-1-1 Kitakaname, Hiratsuka-shi, Kanagawa 259-1292, Japan

Mohammad Emtiyaz Khan Department of Computer Science, Technische Universit¨at Darmstadt RIKEN Center for Advanced Intelligence Project Hochschulstraße 10, Darmstadt, 64289, Hessen, Germany

Kenichi Bannai

Department of Mathematics, Keio University, RIKEN Center for Advanced Intelligence Project 3-14-1 Hiyoshi, Kohoku-ku, Yokohama, 223-8522, Japan erenmehmetkiral@protonmail.com

thomas.moellenhoff@riken.jp

koichi.tojo@tokai.ac.jp

emtiyaz.khan@riken.jp

bannai@math.keio.ac.jp

## Abstract

Variational Bayesian learning for deep neural networks ofers many potential benefits. The dificulty in implementation arises from constraints on the parameters of the variational distributions, as well as the need for low-variance gradient estimators. Our recent work proposes Lie group methods for multiplicative or afine groups that naturally overcome such dificulties. Lie groups are essentially parameter spaces with a binary operator such as addition or multiplication, which allows for natural updates within the parameter space. In this article, we streamline and extend our Lie group framework to any general Lie group, and give an explicit relation to standard natural-gradient variational learning methods such as the Bayesian learning rule (BLR). In particular, we prove that linear approximations of the Lie group algorithm give generalizations of the BLR. The key ingredient is the harmonic exponential family, a general class of Lie group exponential families associated to representations of Lie groups. Our work provides a systematic method to construct from such representations eficiently implementable algorithms for variational learning.

Keywords: Lie groups, exponential families, optimization, Bayesian methods

## 1 Introduction

Variational learning for neural networks (Hinton and Van Camp, 1993; Graves, 2011) offers many potential benefits over standard empirical risk minimization and can be used to motivate various deep-learning algorithms (Khan and Rue, 2023). Some of the benefits of variational learning are improved regularization and reduced overfitting, as suggested by the minimum description length principle (Hinton and Van Camp, 1993; Graves, 2011; Blier and Ollivier, 2018; Gr¨unwald, 2007) and PAC-Bayesian bounds (Alquier, 2024). Depending on the form of the variational posterior, diferent deep learning algorithms can be obtained, for instance, Adam-like methods (Shen et al., 2024) for diagonal Gaussians, algorithms for binary neural networks for Bernoulli distributions (Meng et al., 2020) and biologically plausible multiplicative dynamics for log-normal distributions (Nishida et al., 2026). The Bayesian viewpoint also ofers a fruitful framework for thinking about model adaptation and model understanding, for example, federated and continual learning (Nguyen et al., 2018; Ashman et al., 2022), model merging (Matena and Rafel, 2022; Daheim et al., 2024) or sensitivity analysis to perturbations in both the model and data (Nickl et al., 2023).

The de-facto standard approach for variational learning in neural networks is to update the distribution parameters using gradient-based methods to leverage backpropagation training (Graves, 2011). Khan and Rue (2023), for example, use a natural gradient method on the natural parameters of an exponential family. This method has two main dificulties. First, one needs to satisfy constraints on the parameters of the distribution, for example, positivity of the covariance matrix in the case of Gaussians (Lin et al., 2020). Secondly, the objective functions involve integrals with respect to the variational distribution, and obtaining low-variance stochastic gradients is a major challenge. Reparametrization tricks provide such estimators, but it is not clear whether they are available for the variational family at hand and they have to be derived case by case (Mohamed et al., 2020).

Recently, Kıral et al. (2023) proposed a learning algorithm which addresses the above difficulties by parametrizing the probability distribution through Lie groups. The constraints on the parameters are naturally handled through the group’s closure and lower variance gradients are available by default as the Lie group automatically enables the reparametrization technique. Furthermore the homogeneity of the group allows transporting the gradients on the Lie group to a common tangent space thus allowing for a natural application of momentum, and such transports are compatible with the Fisher information metric which is invariant under reparametrizations.

In this work we establish the connection between the Lie group updates and existing methods on exponential families (EFs) such as the Bayesian Learning Rule (BLR) by Khan and Rue (2023). The Lie group update rule does not require the family to be an EF, however many useful families are EFs. EFs are especially well suited to the Bayesian framework, which relies on multiplication and division of distributions to add, remove or modify information. This corresponds to simple addition and subtraction on the natural parameters of the family, connecting the Bayesian framework to optimizing on natural parameters. EFs come with a rich structure, such as dually flat coordinate systems of natural and expectation parameters studied in information geometry (Amari, 2016). As EFs have many desirable properties, the aim of this paper is to further study the Lie group learning rule of Kıral et al. (2023) when specialized to EFs and to understand its relation to the Bayesian learning rule (Khan and Rue, 2023) which is derived for EFs and gives a unifying and simplifying view of many other algorithms.

Those EFs which also have a Lie group structure are families of distributions which combine the best of both worlds, and it should therefore not be surprising that most of the eponymous EFs named after a mathematician share this property. They have the above desirable properties of EFs, but the additional group structure gives easy reparametrization tricks and allows for updating without breaking constraints. Developing a recipe to construct exponential families on homogeneous spaces from real representations of Lie groups,

Tojo and Yoshino (2021) redefine the term harmonic exponential families subsuming the earlier definition coined by Cohen and Welling (2015) for homogeneous spaces of compact Lie groups via unitary representations. These families characterize all exponential families closed under a Lie group action. Using this characterization we establish a general and concrete connection between the Lie group update and BLR. Our main contribution is to prove that the linearization of the update proposed by Kıral et al. (2023) always recovers the Bayesian learning rule for exponential families with Lie group structure, see Theorem 8. Altogether, our work supplies the theoretical foundations for eficient algorithms for variational learning using Lie groups and exponential families.

## 2 Background on Variational Learning

We first introduce some background on variational learning methods, with particular focus on exponential families. In the following section we introduce the Lie group method of Kıral et al. (2023) to prepare for combining Lie groups with exponential families.

## 2.1 The Variational Learning Problem

Given a parametrized model (for example, a neural network) with parameter space $\Theta ,$ choose a base measure ν on Θ. For a standard neural network the parameter space $\Theta$ is either $\mathbb { R } ^ { P }$ or an open subset of $\mathbb { R } ^ { P }$ , and the base measure is generally taken as the Lebesgue measure. Denote the loss function by $\ell : \Theta \to \mathbb { R } _ { \geq 0 }$ . Instead of minimizing $\ell ( \pmb \theta )$ to arrive at a $\hat { \pmb { \theta } } \in \Theta$ , variational learning obtains a distribution on the parameters via the optimization,

$$
\widehat { q } \in \mathop { \mathrm { a r g } } _ { \boldsymbol { q } \in \mathcal { Q } } \operatorname* { m i n } _ { \boldsymbol { q } \in \mathcal { Q } } \mathcal { E } ( \boldsymbol { q } ) \qquad \mathrm { w h e r e } \qquad \mathcal { E } ( \boldsymbol { q } ) = \mathbb { E } _ { \boldsymbol { \theta } \sim \boldsymbol { q } } [ \ell ( \boldsymbol { \theta } ) ] - \tau \mathcal { H } _ { \boldsymbol { \nu } } ( \boldsymbol { q } ) .\tag{1}
$$

Here $\mathcal { Q }$ is a variational family of distributions on $\Theta$ and $\tau > 0$ is a temperature parameter. Assume members of $\mathcal { Q }$ to be of the form qdν for some density $q ,$ and by abuse of notation we often omit dν and simply refer to the distribution q—since the base measure will be fixed throughout, once chosen. We also assume that ℓ is ν-almost everywhere diferentiable and satisfies $\mathbb { E } _ { q } [ | \ell | ] < \infty$ for every $q \in \mathcal { Q }$ . The first is standard also in non-variational (pointbased) learning if applying methods such as gradient descent, and the second assumption holds as long as members of Q have rapid decay and ℓ has moderate growth<sup>1</sup>.

The term $\begin{array} { r } { \mathcal { H } _ { \nu } ( q ) = - \int _ { \Theta } q } \end{array}$ log qdν denotes the entropy of the measure qdν relative to the base measure ν. The base measure underlies the concept of entropy, see Fig. 1 for how the choice of base measure can afect the solution. Indeed, if we have two ways of writing the same measure $q _ { 1 } \mathrm { d } \nu = q _ { 2 } ( h \mathrm { d } \nu )$ their relative entropies are not the same and are related by the equality $\mathcal { H } _ { \nu } ( q _ { 1 } ) + \mathbb { E } _ { q _ { 1 } \mathrm { d } \nu } [ \log h ] = \mathcal { H } _ { h \nu } ( q _ { 2 } )$ . We introduce the required notions from measure theory later when we need them in Sec. 3.1.

The notion of relative entropy is exactly the KL-divergence $\begin{array} { r } { \mathcal { H } _ { \nu } ( q ) = - \operatorname { K L } ( q \mathrm { d } \nu \| \nu ) } \end{array}$ if ν is also a probability measure. The variational learning problem is sometimes (equivalently) introduced as $\mathcal { E } ( q ) = \mathbb { E } _ { q } [ \ell _ { \mathrm { d a t a } } ] + \tau \mathrm { K L } ( q | | p _ { \mathrm { p r i o r } } )$ , where the prior is chosen as $p _ { \mathrm { p r i o r } } \propto e ^ { - \frac { 1 } { \tau } \ell _ { \mathrm { r e g } } } \mathrm { d } \nu$ following a decomposition of the loss $\ell = \ell _ { \mathrm { d a t a } } + \ell _ { \mathrm { r e g } }$ into the data term and regularizer.

![](images/f397283dcbe6e894827b11ed0ba9fff47b29cc54b09631d962ff935ec97211ea.jpg)  
Figure 1: (Left) The distributions $q _ { 1 }$ and $q _ { 2 }$ are translates of each other and have the same entropy relative to the Lebesgue base measure $\mathrm { d } \nu ( \theta ) = \mathrm { d } \theta$ . Therefore an algorithm optimizing for Eq. 1 will not take entropy into account in deciding between them while finding ${ \hat { q } } .$ (Right) Two distributions on $\mathbb { R } _ { > 0 }$ are related by $\begin{array} { r } { q _ { 4 } ( \theta ) = \frac { 1 } { 2 } q _ { 3 } ( \frac { \theta } { 2 } ) } \end{array}$ . Relative to the Lebesgue measure, $q _ { 4 }$ is spread out more and has a higher entropy, however both measures have the same entropy relative to the base measure $\mathrm { d } \theta / \theta$ . In this case the densities are given by $\tilde { q } _ { j }$ with $\tilde { q } _ { j } ( \theta ) = \theta q _ { j } ( \theta )$ $( j = 3 , 4 )$ and $\begin{array} { r } { \tilde { q } _ { 4 } ( \theta ) = \tilde { q } _ { 3 } ( \frac { \theta } { 2 } ) } \end{array}$ . Therefore the solution to the Bayesian Learning problem in $\operatorname { E q } .$ . 1 depends on the base measure and it is the practitioner’s choice whether to bias the solutions $\hat { q }$ towards $q _ { 4 }$ $( \mathrm { d } \nu ( \theta ) = \mathrm { d } \theta )$ or not $\begin{array} { r } { ( \mathrm { d } \nu ( \theta ) = \frac { \mathrm { d } \theta } { \theta } ) } \end{array}$

In Eq. 1, the expectation term $\mathbb { E } _ { q } [ \ell ]$ pushes the optimization towards q that are more concentrated in regions of low loss. Antagonistically the negative entropy term $ { - } \mathcal { H } _ { \nu } ( q )$ pushes the optimization towards $q$ which are more spread out, exploring the parameter space. The objective $\mathcal { E }$ can thus be seen as an implementation of the exploration–exploitation trade-of in the parameter space. The temperature τ balances this trade-of. The entropy term is an additional regularization on top of what might already exist in $\ell ,$ selecting for models robust to larger parameter noise. These ideas have been studied from the minimum description length principle perspective (Hinton and Van Camp, 1993; Graves, 2011; Blier and Ollivier, 2018; Gr¨unwald, 2007) and through PAC-Bayesian bounds (Alquier, 2024).

If $\ell ( \pmb \theta )$ is the sum of negative log-likelihood and the negative log-prior, then Eq. 1 is equivalent (with $\tau = 1 )$ to minimizing the Kullback-Leibler divergence arg min $_ { \cdot q \in \mathcal { Q } } \operatorname { K L } ( q | | p )$ from the “Gibbs” posterior $p \propto p ( \mathcal { D } | \pmb { \theta } ) p _ { \mathrm { p r i o r } } ( \pmb { \theta } ) \propto e ^ { - \ell ( \pmb { \theta } ) }$ , leading to a Bayesian interpretation (Zellner, 1988). A general loss function ℓ leads to generalized posteriors, (Zhang, 1999; Catoni, 2007; Knoblauch et al., 2019; Bissiri et al., 2016).

The set Q over which we restrict Eq. 1 is also called the space of candidate posteriors. An important family of candidate posteriors $\mathcal { Q }$ are exponential families, which have many desirable properties. We introduce them in the following subsection, and subsequently the BLR algorithm solving for Eq. 1 with exponential families.

## 2.2 Exponential Families

First construct an exponential family of distributions on $\Theta .$ . For this purpose, fix a pair $( \nu , \mathbf { T } )$ where ν is a base measure on Θ and $\mathbf { T } : \Theta \to \mathbb { R } ^ { d }$ is a measurable function<sup>2</sup> with coordinate functions $\mathbf { T } ( \pmb \theta ) = ( T _ { 1 } ( \pmb \theta ) , T _ { 2 } ( \pmb \theta ) , \dots , T _ { d } ( \pmb \theta ) ) ^ { \top }$ . This T is called a suficient statistic. Given

T and ν as above, we construct the associated (maximal) exponential family as

$$
\begin{array} { r } { \mathrm { E F } _ { \nu } ( \mathbf { T } ) = \left\{ q _ { \lambda } \mathrm { d } \nu : q _ { \lambda } ( \theta ) = \exp \left( - \lambda ^ { \top } \mathbf { T } ( \theta ) - A ( \lambda ) \right) : \lambda \in \Omega \right\} , } \end{array}\tag{2}
$$

where $A = A _ { \nu , \textbf { T } } \colon \mathbb { R } ^ { d }  ( - \infty , \infty ]$ is the log partition function defined as

$$
A ( \boldsymbol { \lambda } ) = \log \int _ { \Theta } \exp \left( - \boldsymbol { \lambda } ^ { \top } \mathbf { T } ( \pmb { \theta } ) \right) \mathrm { d } \nu ( \pmb { \theta } )\tag{3}
$$

and $\Omega _ { \nu , \mathbf { T } } = \{ \lambda \in \mathbb { R } ^ { d } : A ( \pmb { \lambda } ) < \infty \}$ is the efective domain of A. The log-partition function $A ( \lambda )$ is convex and therefore the set $\boldsymbol { \Omega } = \Omega _ { \nu , \mathbf { T } }$ is convex as well.

The vectors λ are called the natural parameters of the distribution $q _ { \lambda } \mathrm { d } \nu$ . The coordinatefree notation is also revealing; consider $\mathbf { T } : \Theta \to \mathrm { V }$ where V is a finite dimensional real vector $\mathrm { s p a c e ^ { 3 } }$ and in this case λ is naturally an element of the dual space $\mathrm { V } ^ { \ast }$ . This coordinate-free notation lets us distinguish between the suficient statistics and natural parameters which are otherwise both simply elements of $\mathbb { R } ^ { d }$ . The linear dual space—denoted by ${ \mathrm { V } } ^ { * }$ —is the set of real valued linear functionals on V, which itself forms a vector space. The value of the linear functional is denoted by the inner product notation $\lambda ( \mathbf { v } ) = \langle \lambda , \mathbf { v } \rangle$ for $\mathbf { v } \in \mathrm { V }$ and $\lambda \in \mathrm { V } ^ { * }$ . This coincides with the coordinate-laden description above; upon choosing a basis of V and the corresponding dual basis in ${ \mathrm { V } } ^ { * }$ , both $\mathbf { T } ( \pmb \theta )$ and λ can be written as column vectors and the pairing is given by the dot product $\langle \mathbf { \lambda } \rangle , \mathbf { T } ( \pmb { \theta } ) \rangle = \lambda ^ { \top } \mathbf { T } ( \pmb { \theta } )$

The discussion above is a constructive approach to exponential families. Alternatively, we may as in Sec. C, from (Barndorf-Nielsen, 1978; Tojo and Yoshino, 2021, § 8.1; Def. 2.2), give criteria dubbing a set $\mathcal { P }$ of probability measures as an exponential family or not. Their approach does not commit to a base measure or suficient statistics, as the same set of probability distributions may be realized as an exponential family with two (or more) realizations given by $R = ( \nu , { \bf T } ) , R ^ { \prime } = ( \nu ^ { \prime } , { \bf T ^ { \prime } } )$ . It also allows for considering smaller families than all normalizable ones. Given a realization R of an exponential family ${ \mathcal { P } } _ { : }$ , denote the set of natural parameters corresponding to the family as $\Omega _ { R } ( \mathcal { P } ) \subseteq \Omega _ { R } \subseteq \mathrm { V } ^ { * }$ . We drop R from the notation if it is understood. It is worth going into some delicate technicalities about diferent realizations of a set of probability measures as an exponential family.

Example 1 Take $\mathcal { P }$ as the one-dimensional Gaussian distributions with fixed variance 1.   
This set of distributions can be realized as an exponential family in two ways.

Take $\mathrm { d } \nu ( x ) = \mathrm { d } x , i . e .$ the Lebesgue measure, and $\mathbf { T } ( x ) \ = \ ( x , { \textstyle \frac { 1 } { 2 } } x ^ { 2 } )$ as the suficient statistic $\mathbf { T } : \mathbb { R }  \mathbb { R } ^ { 2 } \cong \mathbf { V }$ . In this case the distribution $\begin{array} { r } { p = \frac { 1 } { \sqrt { 2 \pi } } \exp ( - \frac { 1 } { 2 } ( x - m ) ^ { 2 } ) \mathrm { d } x \in \mathcal { P } } \end{array}$ can be associated with the natural parameter $\lambda = ( - m , 1 ) \in \mathbb { R } ^ { 2 } \cong \mathrm { V } ^ { * }$ . This natural parameter space $\Omega _ { \mathrm { d } x , \mathbf { T } } ( \mathcal { P } )$ lies inside the set of all feasible natural parameters $\Omega _ { \nu , \mathbf { T } } = \{ ( \lambda _ { 1 } , \lambda _ { 2 } ) \in \mathbb { R } ^ { 2 }$ : $\lambda _ { 2 } > 0 \}$ which constructs the space of all 1-d Gaussians.

As a second realization we can take $\mathrm { d } \nu ^ { \prime } ( x ) = e ^ { - \frac { x ^ { 2 } } { 2 } }$ dx as the base measure and $\mathbf { T } ^ { \prime } : \mathbb { R } $ $\mathbb { R } \cong \mathrm { V }$ , with $\mathbf { T } ^ { \prime } ( x ) = x$ . In this case $\Omega _ { \nu ^ { \prime } , { \bf T ^ { \prime } } } ( \mathcal { P } ) = \mathbb { R } \cong \mathrm { V } ^ { * }$ . In this second realization $\mathcal { P }$ is a $f u l l$ exponential family (given $\nu ^ { \prime }$ and $\mathbf { T } ^ { \prime } )$ and furthermore it is a minimal realization of the exponential family.

The distributions $q _ { \lambda } \mathrm { d } \nu$ in $\mathrm { E F } _ { \nu } ( \pmb { T } )$ can be characterized as

$$
\arg \operatorname* { m a x } _ { \boldsymbol { q } } \mathcal { H } _ { \nu } ( \boldsymbol { q } ) \qquad \mathrm { s u b j e c t ~ t o } \qquad \mathbb { E } _ { \boldsymbol { q } \mathrm { d } \nu } [ \mathbf { T } ] = \mu ,
$$

where $\pmb { \mu } \in V$ is called the mean or the expectation parameter<sup>4</sup>. To be precise, given T, ν and $\mu ,$ if there is a $\lambda \in \Omega _ { \nu , \mathbf { T } }$ such that $\mathbb { E } _ { q _ { \lambda } } [ \mathbf { T } ] = \pmb { \mu }$ then any other distribution satisfying the same constraint has lower entropy. The proof in Cover and Thomas (2006, Theorem 12.1.1) extends to the general base-measure setting. Thus expectation parameters characterize elements of ${ \mathcal { P } } _ { : }$ each $\pmb { \mu }$ giving at most one distribution in $\mathcal { P } _ { \cdot }$ . This shows the conceptual centrality of exponential families, as they can be interpreted as implementing the principle of indiference (principle of maximum entropy) given the expectation information.

There are two canonical coordinatizations of ${ \mathcal { P } } ,$ the natural parameters $\pmb { \lambda } \in \Omega ( \mathcal { P } ) \subseteq$ $\mathrm { V } ^ { \ast }$ and the expectation parameters $\mu \in \mathrm { ~ V ~ }$ . If the functions $1 , T _ { 1 } ( \pmb \theta ) , \dots , T _ { d } ( \pmb \theta )$ are not linearly independent then multiple λ correspond to the same distribution. This can easily be fixed by omitting certain coordinates from $\mathbf { T } ,$ and let us assume that we have done that. Also not every choice of $\mu \in \mathrm { ~ V ~ }$ gives a valid distribution, for example the 1-d Gaussians (with non-fixed variance), have expectation parameters $\begin{array} { r } { \pmb { \mu } = ( m , \frac { 1 } { 2 } ( m ^ { 2 } + \sigma ^ { 2 } ) ) = } \end{array}$ $\mathbb { E } _ { x \sim \mathcal { N } ( m , \sigma ^ { 2 } ) } [ ( x , \frac { 1 } { 2 } x ^ { 2 } ) ]$ . So only those $\pmb { \mu } \in \mathrm { V }$ lying in the (strict) epigraph of the parabola $\textstyle y = { \frac { 1 } { 2 } } x ^ { 2 }$ are feasible expectation parameters corresponding to a natural parameter λ. If λ lies in the interior of $\Omega _ { \nu , { \bf T } }$ then the gradient of the log-partition function connects the two parametrizations as $\pmb { \mu } ( \pmb { \lambda } ) = \mathbb { E } _ { q _ { \pmb { \lambda } } } [ \mathbf { T } ] = - \nabla A ( \pmb { \lambda } )$ , which can be seen via direct calculation. A full exponential family P with a minimal realization $R = ( \nu , \mathbf { T } )$ is called regular if its set of natural parameters $\Omega _ { R } ( \mathcal { P } )$ is an open subset of ${ \mathrm { V } } ^ { * }$ , in this case $- \nabla _ { \lambda } A$ gives the change of variables formula between the two coordinates. We refer the interested reader to Amari (2016, Chapter 2) and Wainwright and Jordan (2008, Chapter 3).

If we select the candidate posteriors Q in $\operatorname { E q . }$ 1 as an exponential family then updating the distribution is the same as updating the natural parameter λ. In the following section we discuss a practical gradient-based algorithm updating λ, the BLR.

## 2.3 Variational Learning in Exponential Families

The Bayesian Learning Rule (BLR) by Khan and Rue (2023) is variational learning on an exponential family, in which E is minimized by Natural Gradient Descent (NGD) on λ. NGD means that the Fisher information metric is chosen as the Riemannian geometry on Ω(P) and the natural parameters are updated using Riemannian gradient descent.

Selecting an open exponential family as candidate posteriors, we minimize $\mathcal { E } ( \lambda ) : = \mathcal { E } ( q _ { \lambda } )$ with gradient descent on λ. Mention of gradients assumes a choice of metric, and if none is specified, it is implicitly the Euclidean geometry. Given a diferent geometry such as the Fisher information metric, the corresponding gradient is denoted by $\widetilde { \nabla } _ { \lambda } \mathcal { E } ,$ called the natural gradient. This vector is concretely computed by preconditioning the Euclidean gradient with the Fisher information matrix $F _ { \lambda }$

Altogether, this leads to the following update,

$$
\lambda ^ { \mathrm { u p d a t e d } } = \lambda - \alpha \widetilde { \nabla } _ { \lambda } \mathcal { E } ( \lambda ) = \lambda - \alpha \mathbf { F } _ { \lambda } ^ { - 1 } \mathrm { d } \mathcal { E } \big | _ { \lambda } ,\tag{4}
$$

for step size $\alpha > 0$ . Here $\mathrm { d } \mathcal { E } \big | _ { \lambda }$ refers to the diferential of E which naturally lives in the cotangent space $T _ { \lambda } ^ { * } \Omega ( \mathcal { P } ) \cong ( \mathrm { V } ^ { * } ) ^ { * } \cong \mathrm { V }$ . In the basis $\{ \mathrm { d } \lambda _ { i } \} _ { i }$ this diferential can be written as $\nabla _ { \lambda } \mathcal { E } ( \lambda )$ , a vector with partial derivatives $\frac { \partial \mathcal { E } } { \partial \lambda _ { i } }$ as coordinate values, and in coordinates the natural gradient is $\widetilde { \nabla } _ { \lambda } \mathcal { E } ( \lambda ) = \mathbf { F } _ { \lambda } ^ { - 1 } \nabla _ { \lambda } \mathcal { E } ( \lambda )$ . Geometrically speaking, the natural gradient is a tangent vector in $T _ { \lambda } \Omega ( \mathcal { P } ) \cong \mathrm { V } ^ { * }$ , and the inverse Fisher matrix can naturally be seen as a transformation mapping the diferential co-tangent vector back to the tangent space. More details on this geometric point of view can be found in Sec. C.2.

If the base measure ν in Eq. 1 coincides with that of the exponential family’s realization $\mathrm { d } \mathcal { H } _ { \nu } \vert _ { \lambda } ( q _ { \lambda } ) = - \mathbf { F } _ { \lambda } \lambda$ , then we get for $\tau = 1$

$$
\lambda ^ { \mathrm { u p d a t e d } } = ( 1 - \alpha ) \pmb { \lambda } + \alpha \mathbf { F } _ { \lambda } ^ { - 1 } \mathrm { d } \mathbb { E } _ { q _ { \lambda } } [ - \ell ( \pmb { \theta } ) ] \big | _ { \lambda } ,\tag{5}
$$

cf. BLR in Khan and Rue (2023, Eq. 3). Implementing the update in Eq. 5 comes with two dificulties. First, if the learning rate α is large the update may overshoot and leave the constraint set of admissible parameters $\lambda \in \Omega$ . Secondly, the formalism of exponential families by itself does not provide enough structure for eficiently calculating a low-variance stochastic approximation to the gradient of the expected loss. Where such calculation is tractable, we observe that a reparametrization trick is available because a Lie group acts on the parameters and the distributions. In the following section, we introduce the Lie Group Bayesian Learning Rule (LG-BLR) of Kıral et al. (2023), which addresses these issues.

## 3 The Lie Group Method of Kıral et al. (2023)

Estimating the derivative of the expected loss in Eq. 5 has garnered much interest, see Mohamed et al. (2020) for a survey of various methods. Pathwise gradient estimators are one low-variance stochastic estimation method. This technique requires that the distributions in the variational family come from diferentiable reparametrizations of a base noise, such as in the case of Gaussian distributions. Upon translations and scaling, the stochastic aspect of sampling from any Gaussian distribution can be relegated to sampling noise vectors from $\mathcal { N } ( \pmb { \theta } | \ \mathbf { 0 } , I )$ . This generalizes to all similar cases where a Lie group of transformations acts diferentiably on Θ and the distributions in Q are pushforwards of each other via these transformations. In many cases using the reparametrization method reduces the variance in stochastic estimation of the gradient of expectation in comparison with methods such as score function estimator, see Xu et al. (2019). This is the setting of LG-BLR (Kıral et al., 2023), which is a learning algorithm for the variational problem in Eq. 1. The purpose of this work is to show that in cases such as families of Gaussian or exponential distributions, where Q is both an exponential family and the Lie group setting applies, the linearization of the LG-BLR yields the BLR. This gives insights on BLR itself, which we mention later.

## 3.1 Notation for the Lie Group Update

The setup of Kıral et al. (2023) involves the notions of Lie groups, group actions, homogeneous spaces, and relatively invariant measures. For details about groups in the machine learning context (Bronstein et al., 2021, pp. 13, 40). For an introduction to measure theory in machine learning context, see the blog post (Escudero, 2024). Below is a short summary.

A group is a set G with a binary operation ∗ satisfying the properties of associativity, existence of an identity element and inverses. These mean: first, $g * ( h * k ) = ( g * h ) * k$ for all $g , h , k \in G ;$ second, there exists an identity element $e \in G$ such that e ∗ $g = g * e = g$ for all $g \in G ;$ finally, for any $g \in G$ there exists an inverse element which we denote by $g ^ { - 1 } \in G$ such that $g * g ^ { - 1 } = g ^ { - 1 } * g = e$ . A Lie group is also a smooth manifold, and both its binary group-operation and inversion are smooth. For groups written with a multiplicative notation, it is common to write $g h$ in place of $g * h$ . A smooth manifold is locally difeomorphic to Euclidean space, that is, there are infinitely diferentiable invertible mappings between local patches of G and an open subset of $\mathbb { R } ^ { m }$ , and m is called the dimension of the manifold. As an example the set of invertible matrices on n-dimensions is a group with respect to matrix multiplication, is a Lie group of dimension $n ^ { 2 }$

The tangent space of the Lie group at the identity is called a Lie algebra, denoted as ${ \mathfrak { g } } = T _ { e } G$ . It is a vector space, which also has a binary operation called the Lie bracket denoted by $[ X , Y ] ~ \in ~ { \mathfrak { g } }$ for $X , Y \ \in \ { \mathfrak { g } }$ . For matrix groups (i.e. subgroups of invertible $n \times n$ matrices) the Lie algebra is a vector space of square matrices and the Lie bracket is concretely given as $[ X , Y ] = X Y - Y X$

The diferential of the left-multiplication-by-g map $L _ { g } : x \mapsto$ gx on G provides a consistent way to transfer a direction vector $v \in { \mathfrak { g } }$ to a tangent vector $\mathrm { d } L _ { g } \big | _ { e } ( v ) \in T _ { g } G$ . In LG-BLR we make use of this, ensuring that all the gradient vectors end up in the same vector space g, so that we can accumulate past gradients via momentum<sup>5</sup>.

The right-multiplication-by-g map $R _ { g } : x \mapsto x g$ has the same efect, which is diferent from $L _ { g }$ if the group multiplication is not commutative. The diference between the left and right multiplication maps can be measured by the conjugation automorphisms, which for a given group element $g \in G$ is the map $x \mapsto g x g ^ { - 1 }$ . It is diferentiable and $e \mapsto e ,$ so its diferential at identity is a linear map $\operatorname { A d } _ { g } : { \mathfrak { g } } \to { \mathfrak { g } }$ . It is called the adjoint map. For commutative groups G, this map is always the identity. For matrix groups, $g \in G$ and $X \in { \mathfrak { g } }$ are both square matrices and $\operatorname { A d } _ { g } ( X ) = g X g ^ { - 1 } \in { \mathfrak { g } }$

Every Lie group comes with the structure of an exponential map, allowing us to wrap the tangent space at identity to the group manifold. The exponential exp $_ G : { \mathfrak { g } } \to G$ is a map defining paths—also called one parameter subgroups—in G by $\{ \gamma _ { X } ( t ) : = \exp ( t X )$ $t \in \mathbb { R } \}$ . These paths are closed under the group multiplcation and the direction at $t = 0$ is $\gamma _ { X } ^ { \prime } ( 0 ) = X$ . These are geodesics<sup>6</sup> on $G$ starting at the identity e, with initial velocity X. If the group is understood we may simply write $\exp _ { G } ( X ) = \exp ( X ) = e ^ { X }$ , dropping G from the notation. Given two directions $X , Y \in { \mathfrak { g } }$ such that $\left[ { \cal X } , { \cal Y } \right] = 0$ , the exponential map is multiplicative; $e ^ { X + Y } = e ^ { X } e ^ { Y }$ . In particular $e ^ { ( t + s ) X } = e ^ { t \bar { X } } e ^ { s X }$ , justifying why we call $\gamma _ { X }$ as one parameter subgroups. For matrix groups G, the Lie algebra g is a subspace of square matrices and the exponential map is given by $\begin{array} { r } { \exp ( X ) = \sum _ { n = 0 } ^ { \infty } \frac { X ^ { n } } { n ! } } \end{array}$

A group action on a set A is a map $G \times A \to A$ denoted by g·a satisfying $( g h ) \cdot a = g \cdot ( h \cdot a )$ and $e \cdot a = a$ for any $g , h \in G , a \in A$ and e as the identity element of G. The set $\{ g \cdot a : g \in G \} \subseteq A$ is called the orbit of a. A group action is transitive if there is some element $a _ { 0 } \in A$ such that any $a \in A$ is in the orbit of $a _ { 0 }$ . The stabilizer of an element $a _ { 0 }$ is the subgroup of G fixing $a _ { 0 } .$ , denoted by ${ \mathrm { S t a b } } _ { G } ( a _ { 0 } ) = \{ g \in G : g \cdot a _ { 0 } = a _ { 0 } \}$ . If transitive, the set A and the quotient set $G / H$ (where we call $H = \mathrm { S t a b } _ { \mathrm { G } } ( a _ { 0 } ) )$ can be identified by associating elements $a \in A$ with the group elements g that take a basis point $a _ { 0 }$ to a (modulo the group elements that fix $a _ { 0 } )$ . If A is a manifold, with such a transitive (and continuous) G-action, A is called a homogeneous space. Furthermore if A is a diferentiable manifold with a smooth group action, diferentiating the group action at the identity gives the infinitesimal group action, which we denote with the same symbol: $\begin{array} { r } { X \cdot a : = \frac { \mathrm { d } } { \mathrm { d } t } ( e ^ { t X } \cdot a ) \big | _ { t = 0 } . } \end{array}$

A representation of a group G is a group homomorphism $\pi : G \to \mathrm { G L } ( \mathrm { V } )$ for some vector space V. Given a basis of V this means associating matrices $\pi ( g )$ to elements of the group such that group operation corresponds to matrix multiplication: $\pi ( g _ { 1 } g _ { 2 } ) = \pi ( g _ { 1 } ) \pi ( g _ { 2 } )$ . This makes abstract groups concrete, since matrices are concrete objects. A faithful representation for example, i.e. one where π is injective, replicates the group structure of $G$ inside some $n \times n$ invertible matrices, where $n =$ dim V. Given a group representation $\pi$ on $\mathrm { V } ,$ there is a dual representation $\pi ^ { * } : G \to \mathrm { G L } ( \mathrm { V } ^ { * } )$ . The action of $\pi ^ { * } ( g )$ on a linear functional $\lambda \in \mathrm { V } ^ { * }$ is defined by $\langle \pi ^ { * } ( g ) \lambda , \mathbf { v } \rangle = \langle \lambda , \pi ( g ^ { - 1 } ) \mathbf { v } \rangle$ for every $\mathbf { v } \in \mathrm { V }$ . As matrices (in the dual basis of $\mathrm { V } ^ { * } ) \pi ^ { * } ( g )$ is given by the inverse transpose of $\pi ( g )$

A measure ν on A assigns a nonnegative number to members of a certain collection of subsets (called a sigma-algebra<sup>7</sup> elements of which in this context are called measurable sets) with countable additivity and $\nu ( \emptyset ) = 0$ . For example the Lebesgue measure assigns an interval $( a , b )$ its length $b - a .$ Given a map $\phi : A  B$ there is a measure $\phi _ { \sharp } \nu$ on B called the pushforward measure such that $( \phi _ { \sharp } \nu ) ( E ) = \nu ( \phi ^ { - 1 } ( E ) )$ for all measurable sets E. For this ϕ needs to be “measurable” which all continuous functions are if A also has a topology and the sigma-algebra is constructed from the open sets of A.

Assume the measure space A also has a measurable group action on it, then for each $g \in G .$ , the map $a \mapsto g \cdot a$ , yields the pushforward measure $g _ { \sharp } \nu .$ . Assume further that $\nu ( g E ) = \chi ( g ) \nu ( E )$ for some factor $\chi ( g )$ not depending on the measurable subset $E _ { \mathrm { { i } } }$ then we have $\begin{array} { r } { g _ { \sharp } \nu = \frac { 1 } { \chi ( g ) } \nu . } \end{array}$ In this case ν is called a relatively G-invariant measure. Such a measure is not guaranteed to exist, but in all important cases we present, it exists and can be explicitly written down. The multiplier $\chi : G \to \mathbb { R } _ { > 0 }$ is necessarily a group homomorphism, i.e. it satisfies $\chi ( g _ { 1 } g _ { 2 } ) = \chi ( g _ { 1 } ) \chi ( g _ { 2 } )$ for all $g _ { 1 } , g _ { 2 } \in G$ . We also introduce a bit of notation which is necessary later, and define the diferential of − log χ as ξ, i.e. $\xi : { \mathfrak { g } }  \mathbb { R }$ defined by $\begin{array} { r } { \xi ( X ) : = - \frac { \mathrm { d } } { \mathrm { d } t } \log ( \chi ( \exp ( t X ) ) ) \big | _ { t = 0 } . } \end{array}$

As an example, take $A = \mathbb { R } ^ { P }$ and ν the Lebesgue measure, and $G = \mathrm { A f f } _ { P } ( \mathbb { R } )$ with the action given by $( A , \mathbf { b } ) \cdot \mathbf { x } = A \mathbf { x } + \mathbf { b }$ . Then $( { \cal A } , { \bf b } ) _ { \sharp } \nu = | \operatorname* { d e t } ( { \cal A } ) | ^ { - 1 } \nu$ where the factor $\chi ( ( A , \mathbf { b } ) ) = | \operatorname* { d e t } ( A ) |$ is the Jacobian determinant coming from the change of variables formula. The Lie algebra aff consists of pairs $( X , x )$ with all $P \times P$ matrices X and $\pmb { x } \in \mathbb { R } ^ { P }$ . In this case $\begin{array} { r } { \xi ( ( X , \pmb { x } ) ) = - \frac { \mathrm { d } } { \mathrm { d } t } \log ( \mathrm { d e t } ( e ^ { t X } ) ) \big | _ { t = 0 } = - \frac { \mathrm { d } } { \mathrm { d } t } \operatorname { T r } ( t X ) \big | _ { t = 0 } = - \operatorname { T r } ( X ) } \end{array}$

In terms of change of variables this relative–G-invariance property of ν means that for any measurable function f on A we can write

$$
\int _ { A } f ( a ) \mathrm { d } \nu ( a ) = \int _ { A } f ( g ^ { - 1 } \cdot a ) \mathrm { d } ( g _ { \sharp } \nu ) ( a ) = \frac { 1 } { \chi ( g ) } \int _ { A } f ( g ^ { - 1 } \cdot a ) \mathrm { d } \nu ( a ) .
$$

This is very useful for calculating gradients of loss expectations in our LG-BLR setting.

## 3.2 The Lie Group Update Rule

Assume the setting above, i.e. let G be a Lie group acting smoothly and transitively on the space of parameters Θ of a machine learning model, and ν is a relatively invariant measure on Θ with multiplier $\chi .$ Further assume Θ is a diferentiable manifold, and without loss of generality we may assume it is embedded in some ambient Euclidean space R<sup>n</sup>. Fix a base distribution q<sub>0</sub>dν. The group action map $\theta \mapsto g \cdot \theta$ induces a pushforward on the distributions given as $g _ { \sharp } ( q _ { 0 } \mathrm { d } \nu ) = q ^ { g } \mathrm { d } \nu$ where

$$
q ^ { g } ( \pmb \theta ) = \frac { 1 } { \chi ( g ) } q _ { 0 } ( g ^ { - 1 } \pmb \theta )
$$

is the density of the pushforward distribution<sup>8</sup>. Sampling θ from $q ^ { g } \mathrm { d } \nu$ is equivalent to sampling $\theta _ { 0 } \sim q _ { 0 } \mathrm { d } \nu$ and applying the group action $\pmb { \theta } = \boldsymbol { g } \cdot \pmb { \theta } _ { 0 }$

In LG-BLR of Kıral et al. (2023), the family of candidate posteriors is chosen as

$$
\mathcal { Q } = \{ q ^ { g } \mathrm { d } \nu : g \in G \} .\tag{6}
$$

Members of $\mathcal { Q }$ are parametrized by Lie group elements, and optimizer steps update $g \in G$

![](images/02b85767552d3ea278a2e39a912343c1045531a8c3ca4980fd92405d4fef433e.jpg)  
Figure 2: The direction $Y \in { \mathfrak { g } }$ is carried to $T _ { g } G$ by the diferential of $L _ { g }$ and that is carried to the tangent space $T _ { q ^ { g } } \mathcal { Q }$ by $\left. \mathrm { d } \varphi \right| _ { g }$ . We solve for $Y = { \hat { Y } }$ so that this tangent vector on $\mathcal { Q }$ is in the direction of fastest ascent according to the Fisher metric.

For a step size $\alpha > 0$ , the update is given as

$$
\boldsymbol g ^ { \mathrm { u p d a t e d } } = \boldsymbol g \exp ( - \alpha \boldsymbol { \hat { Y } } )\tag{7}
$$

where $\hat { Y } \in { \mathfrak { g } }$ is in the direction<sup>9</sup> of fastest ascent for the objective $\mathcal { E } ( q )$ at $q = q ^ { g }$ . The notion of fastest ascent requires specifying a metric, and in accordance with the original BLR (Khan and Rue, 2023) we choose the Fisher information metric. To be precise, let B be the pullback of the Fisher information metric on Q to g, by $\varphi \circ L _ { g }$ , see Eq. 24 in Sec. A. Here $L _ { g } : G \to G$ is the left-multiplication-by-g map and $\varphi : G  \mathcal { Q }$ is the parametrization $\varphi ( g ) = q ^ { g }$ from the group to the densities. As the update direction choose a $\hat { Y } \in { \mathfrak { g } }$ satisfying

$$
B ( \hat { Y } , X ) = \mathrm { d } ( \mathcal { E } \circ \varphi \circ L _ { g } ) \big | _ { e } [ X ] \qquad \mathrm { ~ f o r ~ a l l ~ } \quad X \in { \mathfrak { g } } .\tag{8}
$$

![](images/334d6c0363b42f2ac3f053db1b54008f33e3467c6acc835573a5ca50451bbe47.jpg)  
Figure 3: Examples of Lie groups acting on $\Theta$ and distributions on $\Theta .$ (top-left) $\Theta = \mathbb { R }$ and $G = ( \mathbb { R } , + )$ , with action given by $g \cdot \theta = g + \theta .$ . The orbit of the pushforwards are various translations of $q _ { 0 }$ . (middle-left) $\Theta = \mathbb { R } _ { > 0 }$ and $G = ( \mathbb { R } _ { > 0 } , \times )$ with group action $g \cdot \theta = g \theta$ . Log-normal distributions with a fixed shape parameter, Rayleigh distributions, and exponential distributions can all be realized as Eq. 6 with this group action. (bottom-left) $\Theta = \mathbb { R }$ and $G = \mathrm { A f f } _ { 1 } ^ { + } ( \mathbb { R } ) = \{ ( A , b ) : A >$ $0 , b \in \mathbb { R } \}$ the afine group in one dimension, with the group action $( A , b ) \cdot \theta =$ $A \theta + b$ . If $q _ { 0 }$ is the unit normal distribution then $q ^ { ( A , b ) } = \mathcal { N } ( b , A ^ { 2 } )$ . (right) $\Theta = S ^ { 2 }$ and the special orthogonal group $G = \mathrm { S O } ( 3 , \mathbb { R } )$ acts on the sphere by rotations. von Mises–Fisher distributions with a fixed concentration parameter can be constructed as a single orbit under the pushforward action of this group.

One can show $\hat { Y } = s \hat { X }$ , where $\boldsymbol s = \mathrm { d } ( \mathcal { E } \circ \varphi \circ L _ { g } ) \big | _ { e } [ \hat { X } ]$ and where $\hat { X }$ solves

$$
{ \hat { X } } \in \operatorname { a r g } _ { X \in { \mathfrak { g } } } \operatorname { m a x } \ \mathrm { d } ( { \mathcal { E } } \circ \varphi \circ L _ { g } ) { \big | } _ { e } [ X ] \qquad { \mathrm { s u c h ~ t h a t } } \quad B ( X , X ) \leq 1
$$

$\mathrm { S o } .$ , in words, $\hat { X }$ is the direction of fastest ascent according to $B ,$ , and s is the maximal slope. This intuitive picture is valid even if our B has degenerate directions. The existence of $\hat { Y }$ requires that the diferential vanish on such directions, which it does.

The diferential at a given $X ~ \in ~ { \mathfrak { g } }$ can be explicitly calculated by perturbing $g$ to $g \exp ( t X )$ to get the real values $\begin{array} { r } { \mathrm { d } \big ( \mathcal { E } \circ \varphi \circ L _ { g } \big ) \big | _ { e } [ X ] = \frac { \mathrm { d } } { \mathrm { d } t } \mathcal { E } \left( q ^ { g \mathrm { e x p } ( t X ) } \right) \Big | _ { t = 0 } } \end{array}$ . This is the directional derivative of the objective in Eq. 1 in the direction of $X$ , (equivalently in the direction $\mathrm { d } L _ { g } \big | _ { e } [ X ] \in T _ { g } G$ at $g )$ . For our variational objective, it is calculated as,

$$
\frac { \mathrm { d } } { \mathrm { d } t } \mathscr { E } \big ( q ^ { g \exp ( t X ) } \big ) \bigg \vert _ { t = 0 } = \mathbb { E } _ { \pmb { \theta } \sim \ b { q } ^ { g } } \big [ \nabla \ell ( \pmb { \theta } ) ^ { \top } ( \mathrm { A d } _ { g } ( X ) \cdot \pmb { \theta } ) \big ] + \tau \xi ( X ) .\tag{9}
$$

We include this short calculation in Sec. A.1 for completeness, which is a straightforward application of the change of variables followed by diferentiating under the integral, aforded to us by our Lie group setting. The term ξ(X)—which we introduced in the previous subsection—arises as the diferential of the negative entropy term. The adjoint operator $\operatorname { A d } _ { g } : { \mathfrak { g } } \to { \mathfrak { g } }$ is the infinitesimal efect of conjugation as defined above, and can be concretely identified for each choice of Lie group G. The significance of Eq. 9 is that it expresses the diferential of the expected loss $\mathbb { E } _ { g } [ \ell ]$ with an expectation involving ∇ℓ. Thus, a perturbation in the distribution (a perturbation in $g )$ is expressed as a perturbation in the parameters θ.

In summary, we update the group element using the group multiplication and the exponential map, in the direction of the steepest slope. In LG-BLR the bilinear form $B ( \cdot , \cdot )$ on g determining this steepest slope is chosen as the pullback of the Fisher information metric on $\mathcal { Q } .$ This pullback is well defined, and is independent of the group element $g .$ As long as the Fisher metric is nonsingular Eq. 8 can be solved. It is possible that there are multiple solutions $\hat { Y }$ , but we can always choose one among them. More details on obtaining the update direction are in Sec. A.2. Furthermore we should point out that, the formalism of LG-BLR allows us to also simply start with a bilinear form B on g and that determines a left invariant metric on $G ,$ even if it doesn’t necessarily come from the Fisher metric on $\mathcal { Q } .$

## 3.3 On Applying Gradient Accumulation

The Lie group update rule is an optimization on a manifold, nonetheless we can accumulate past gradients as mentioned in Section 3.5 of Kıral et al. (2023). This is thanks to the homogeneity of the Lie group, since we could formulate the updates as multiplications by “small” (i.e. near identity) group elements given by directions at $e \in G$ . In other words, $\hat { Y } _ { t } \in \mathfrak { g } = T _ { e } G$ for all time steps t, regardless of the current position $g _ { t } \in G$ . So all the update vectors lie in the same vector space g.

This generalizes momentum in the standard gradient descent setup on the flat parameter space $\mathbb { R } ^ { P }$ . There as well, the gradient at time t lies in the corresponding tangent space, $( \nabla _ { \pmb { \theta } } \ell ) ( \pmb { \theta } _ { t } ) \in T _ { \pmb { \theta } _ { t } } \mathbb { R } ^ { P }$ , and an identification of all these tangent spaces is required for meaningfully operating with gradient vectors at multiple timepoints. The additive vector structure of $\mathbb { R } ^ { P }$ (as a special case of the group structure mentioned above) canonically identifies all these vector spaces with $\mathbb { R } ^ { P }$ . For a deeper discussion regarding the mathematical underpinnings of what makes momentum possible with manifold learning, see Sec. B.

Furthermore, optimizers such as RMSprop, Adam (Kingma and Ba, 2015), or Lion (Chen et al., 2023) not only accumulate past gradients, but also operate on the gradient vectors such as taking componentwise squares, adaptive componentwise scaling or taking signs. We are equally justified in doing such operations with LG-BLR in the Lie algebra g. The exponential map $\mathrm { e x p } _ { G }$ will still be able to retract the resulting vector onto the group. In Nishida et al. (2026) this idea was successfully applied to the multiplicative group, where a Lion-like update was applied. Roughly speaking, the update used in that paper was of the form $g _ { t + 1 } \gets g _ { t } \exp ( - \alpha \mathrm { s i g n } ( M _ { t } ) )$ where $M _ { t } \in { \mathfrak { g } }$ is the standard momentum accumulation. This greatly increased the stability of the multiplicative update rule.

In Sec. 5.4 we express this Lie group momentum in terms of natural gradients in BLR, in case the candidate posteriors are also an exponential family.

## 3.4 An Example: Afine Groups and Gaussians

For each Lie group, LG-BLR corresponds to a diferent update. For example let $G =$ $\mathrm { A f f } _ { P } ^ { + } ( \mathbb { R } )$ acting on $\Theta = \mathbb { R } ^ { P }$ Elements of $G$ are of the form $g = \left( A , \mathbf { b } \right)$ where $A$ is an invertible $P \times P$ matrix with positive determinant and b is a vector in $\mathbb { R } ^ { P }$ . The action is given by $( A , \mathbf { b } ) \cdot { \pmb \theta } = A { \pmb \theta } + \mathbf { b }$ . If ${ \bf q } _ { 0 } = \mathcal { N } ( \pmb { \theta } \mid \mathbf { 0 } , I )$ then, the orbit of $q _ { 0 }$ under the afine group action is the statistical manifold of all multivariate Gaussian distributions. Explicitly, $\boldsymbol { q } ^ { ( A , b ) } ( \pmb { \theta } ) = \mathcal { N } ( \pmb { \theta } | \pmb { b } , \boldsymbol { A } \boldsymbol { A } ^ { \top } )$ . The covariance matrix is $\Sigma = A A ^ { \top }$ , and the mean is $m = b$

The LG-BLR updates the group elements $^ { g , }$ along a tangent direction in g, which in turn gives an update in the Gaussian distribution $q ^ { g } \in \mathcal { Q }$ . Elements of the afine Lie algebra (denoted by $\mathfrak { a f f } _ { P } )$ can be written as $( X , \mathbf { x } )$ where X is a (not necessarily invertible) $P \times P$ matrix and $\mathbf { x } \in \mathbb { R } ^ { P }$ . The bilinear form on this tangent space obtained as the pullback of the Fisher Information metric on $\mathcal { Q }$ via $\left. \operatorname { d } ( \varphi \circ L _ { g } ) \right| _ { e }$ is independent of $g ,$ and is given by

$$
\langle ( X , \pmb { x } ) , ( Y , \pmb { y } ) \rangle = 2 \langle \mathrm { S y m } ( X ) , \mathrm { S y m } ( Y ) \rangle _ { \mathrm { F r o b } } + \mathbf { x } ^ { \top } \mathbf { y } .\tag{10}
$$

where $\operatorname { S y m } ( X ) = ( X + X ^ { \top } ) / 2$ , and the Frobenius inner product $\langle A , B \rangle _ { \mathrm { F r o b } } = \operatorname { T r } ( A ^ { \top } B )$ This symmetrization exists because the unit normal distribution $q _ { 0 }$ is invariant under rotations, and so $q ^ { g Q } = q ^ { g }$ for any orthogonal matrix $Q \in \mathrm { O } ( P , \mathbb { R } )$ . Therefore the bilinear form is degenerate in the direction of antisymmetric matrices.

The group can be represented as $( P + 1 ) \times ( P + 1 )$ matrices $\left[ \begin{array} { c } { A } \\ { \mathbf { 0 } ^ { \top } } \end{array} \right]$ and then the tangent matrices at identity are of the form $\left[ \begin{array} { l } { X } \\ { \mathbf { 0 } ^ { \top } } \end{array} \right]$ . The exponential map is given as

$$
\exp ( ( X , \mathbf { x } ) ) = \sum _ { n = 0 } ^ { \infty } { \frac { 1 } { n ! } } \left[ \mathbf { D ^ { \top } } \quad \mathbf { x } \right] ^ { n } = I + \sum _ { n = 1 } ^ { \infty } { \frac { 1 } { n ! } } \left[ X _ { \mathbf { 0 ^ { \top } } } ^ { n } \quad X _ { \mathbf { 0 } } ^ { n - 1 } \mathbf { x } \right] = \left( e ^ { X } , { \frac { e ^ { X } - I } { X } } \mathbf { x } \right) .\tag{11}
$$

Now we specialize the diferential of the objective functional in Eq. 9 to the case of the Afine group and Gaussians. The adjoint map for matrix groups can be calculated by conjugation

$$
\left[ { \begin{array} { c c } { A } & { b } \\ { 0 ^ { \top } } & { 1 } \end{array} } \right] \left[ { \begin{array} { c c } { X } & { x } \\ { 0 ^ { \top } } & { 0 } \end{array} } \right] \left[ { \begin{array} { c c } { A ^ { - 1 } } & { - A ^ { - 1 } b } \\ { 0 ^ { \top } } & { 1 } \end{array} } \right] = \left[ { \begin{array} { c c } { A X A ^ { - 1 } } & { A x - A X A ^ { - 1 } b } \\ { 0 ^ { \top } } & { 0 } \end{array} } \right]
$$

and thus for a $\pmb \theta$ sampled from $\boldsymbol { q } ^ { ( A , b ) }$ , or in other words for $\pmb { \theta } = A \pmb { \theta } _ { 0 } + \pmb { b }$ with $\pmb { \theta } _ { 0 } \sim q _ { 0 }$ 2

$$
\operatorname { A d } _ { ( A , b ) } ( X , \mathbf { x } ) \cdot \pmb \theta = A X A ^ { - 1 } ( \pmb \theta - \pmb b ) + A \pmb x = A X \pmb \theta _ { 0 } + A \pmb x .
$$

Therefore the diferential of E is given by

$$
\begin{array} { l } { \displaystyle \mathrm { d } ( \mathcal { E } \circ \varphi \circ L _ { g } ) \big | _ { ( I , 0 ) } [ ( X , x ) ] = \int _ { \Theta } ( \nabla \ell ( A \theta + b ) ) ^ { \top } ( A X \theta + A x ) q _ { 0 } ( \theta ) \mathrm { d } \nu ( \theta ) - \tau \mathrm { T r } ( X ) } \\ { \displaystyle \qquad = 2 \mathrm { T r } ( \mathrm { S y m } ( Y ) ^ { \top } \mathrm { S y m } ( X ) ) + \mathbf { y } ^ { \top } \mathbf { x } } \end{array}
$$

where

$$
\begin{array} { r } { \mathbf { y } = \boldsymbol { A } ^ { \top } \mathbb { E } _ { \pmb { \theta } \sim \boldsymbol { q } _ { 0 } } [ \nabla \ell ( \boldsymbol { A } \pmb { \theta } + \boldsymbol { b } ) ] , } \end{array}\tag{12}
$$

and

$$
\begin{array} { r } { Y = \frac { 1 } { 2 } A ^ { \top } \mathbb { E } _ { \pmb { \theta } \sim \ b { q } _ { 0 } } [ \nabla \ell ( A \pmb { \theta } + \pmb { b } ) \pmb { \theta } ^ { \top } ] - \frac { \tau } { 2 } I . } \end{array}\tag{13}
$$

![](images/61e0d0da52da53e263f1758ab96f5855ddbaf192548a06f5cdb8298a4616a2c5.jpg)  
Figure 4: The update vector is automatically modified so that the mean of the distribution takes larger steps in the direction of the uncertainty of the Gaussian, and smaller steps in direction of higher precision.

The inner product in Eq. 10 on ${ \mathfrak { a f f } } _ { P }$ induced via Eq. 24 from the Fisher information metric on Q gives this particular $( Y , \pmb { y } ) = ( \hat { Y } , \hat { \pmb { y } } )$ as the direction that solves Eq. 8.

In practice, we estimate the expectations using a finite number of MC-samples

$$
Y \approx \frac { 1 } { 2 K } \sum _ { k = 1 } ^ { K } A ^ { \top } \nabla \ell ( A \pmb { \theta } _ { k } + b ) \pmb { \theta } _ { k } ^ { \top } - \frac { \tau } { 2 } I , \quad \pmb { y } \approx \frac { 1 } { K } \sum _ { k = 1 } ^ { K } A ^ { \top } \nabla \ell ( A \pmb { \theta } _ { k } + b ) , \mathrm { ~ w h e r e ~ } \pmb { \theta } _ { k } \sim \mathcal { N } ( \mathbf { 0 } , I )
$$

and update the current group element $g = ( A , b )$ with these approximations as

$$
A ^ { \mathrm { u p d a t e d } } = A e ^ { - \alpha Y } , \qquad b ^ { \mathrm { u p d a t e d } } = b - A \frac { e ^ { - \alpha Y } - I } { - \alpha Y } \alpha y .
$$

Let us try to understand these updates intuitively. Up to a first order approximation in α the Gaussian’s mean vector b is updated as

$$
\begin{array} { r } { b ^ { \mathrm { u p d a t e d } } \approx \pmb { b } - \alpha A \pmb { y } = \pmb { b } - \alpha A A ^ { \top } \mathbb { E } _ { q ^ { ( A , b ) } } [ \nabla \ell ] = \pmb { b } - \alpha \pmb { \Sigma } \mathbb { E } _ { q ^ { ( A , b ) } } [ \nabla \ell ] . } \end{array}
$$

So instead of the (average) gradient the update vector is modified by Σ, which has the efect of taking longer steps in directions where there is more uncertainty, and smaller steps where the Gaussian has less uncertainty. For the A update, using integration by parts we rewrite

$$
\begin{array} { r } { Y = \frac { 1 } { 2 } A ^ { \top } \mathbb { E } _ { q ^ { ( A , b ) } } [ \nabla ^ { 2 } \ell ] A - \frac { \tau } { 2 } I . } \end{array}
$$

This is not necessarily useful practically, as the computation of the Hessian is expensive, and instead we use the form in Eq. 13. However for conceptual analysis this form is useful. We see that the precision matrix of the Gaussian is iteratively replaced by the expected Hessian. Indeed, the precision matrix $\mathbf { S } = \Sigma ^ { - 1 } = A ^ { - \top } A ^ { - 1 }$ , is updated as

$$
\mathbf { S } ^ { \mathrm { u p d a t e d } } = { A ^ { - \top } e ^ { \alpha ( Y + Y ^ { \top } ) } A ^ { - 1 } } = { A ^ { - \top } e ^ { \alpha ( A ^ { \top } \mathbb { E } _ { q } g [ \nabla ^ { 2 } \ell ] A - \tau I ) } A ^ { - 1 } } .
$$

The linear approximation to the exponential yields

$$
\begin{array} { r } { { \bf S } ^ { \mathrm { u p d a t e d } } \approx ( 1 - \alpha \tau ) { \bf S } + \alpha \mathbb { E } _ { q ^ { g } } [ \nabla ^ { 2 } \ell ] , } \end{array}
$$

which is identical with the precision update in Khan and Rue (2023, Eq. 12). An advantage of the Lie group parametrization in terms of A and b over the BLR is that it does not require any matrix inversions. Since S is related to the Hessian matrix, this enables a matrix-inverse-free second-order optimization. Such algorithms may be desirable for largescale deep learning (Lin et al., 2023), and our Lie group framework allows a natural way to derive these methods and also endow them with momentum.

We also note that, at this stage we get the second order modification to BLR’s precision matrix update—developed by Lin et al. (2020)—for free by expanding the exponential to the second order. Indeed (using the shorthand $\mathbf { E } : = \mathbb { E } _ { q ^ { g } } [ \nabla ^ { 2 } \ell ] )$ we see

$$
\begin{array} { r l } & { \mathbf { S } ^ { \mathrm { u p d a t e d } } = A ^ { - \top } e ^ { \alpha ( Y + Y ^ { \top } ) } A ^ { - 1 } = A ^ { - \top } e ^ { \alpha ( A ^ { \top } \mathbf { E } A - \tau I ) } A ^ { - 1 } } \\ & { \approx A ^ { - \top } \left( I + \alpha ( A ^ { \top } \mathbf { E } A - \tau I ) + \frac { \alpha ^ { 2 } } { 2 } \left( A ^ { \top } \mathbf { E } \Sigma \mathbf { E } A - 2 \tau A ^ { \top } \mathbf { E } A + \tau ^ { 2 } I \right) \right) A ^ { - 1 } } \\ & { = ( 1 - \alpha \tau ) \mathbf { S } + \alpha \mathbf { E } + \frac { \alpha ^ { 2 } } { 2 } \left( \mathbf { E } \Sigma \mathbf { E } - 2 \tau \mathbf { E } + \tau ^ { 2 } \mathbf { S } \right) } \end{array}
$$

which is identical $( \mathrm { w i t h } ~ \tau = 1 )$ to the precision update $\begin{array} { r } { \mathbf { S }  ( 1 - \alpha ) \mathbf { S } + \alpha \mathbf { E } + \frac { \alpha ^ { 2 } } { 2 } \hat { \mathbf { G } } \boldsymbol { \Sigma } \hat { \mathbf { G } } } \end{array}$ proposed in the citation above, where $\hat { \mathbf { G } } = \mathbf { S } - \mathbf { E }$ . The authors proposed this modification in order to preserve the positive definiteness constraint of the precision (equivalently the covariance) matrices throughout the optimization. The positivity follows since the second order truncation of the exponential is always positive definite, ensuring that the updates remain in Ω, which is not guaranteed by the linear approximation.

In the following sections, we will show that we recover the BLR in general, that is, that when we use the linear approximation for the exponential used in LG-BLR for candidate posterior families $\mathcal { Q }$ which are also open exponential families, then we get the BLR update. Our previous work Kıral et al. (2023) only proves this in particular cases, such as the Rayleigh and diagonal Gaussian family.

## 4 Lie Group Structure of Exponential Families

In Sec. 3.4 above we saw that the Gaussians conform to the Lie group setup. They are also an exponential family of distributions. In general, the pushforward of a distribution $q _ { \lambda }$ from an exponential family does not always correspond to another member $q _ { \lambda ^ { \prime } }$ for some other natural parameter $\lambda ^ { \prime } .$ . In harmonic exponential families—which we define below—we can always find such a $\lambda ^ { \prime }$ explicitly. We are interested in open exponential families that have a transitive Lie group action given by pushforwards as above, that is, when the BLR and the LG-BLR settings coincide. The work of Tojo and Yoshino (2021) allows us to characterize all such exponential families using harmonic exponential families. For the precise statement see Theorem 4. Using this concrete characterization we can express the Lie group updates in terms of updates in the natural parameters λ, with which we prove the connection between LG-BLR and BLR.

## 4.1 Harmonic Exponential Families

Setup 1 (Transitive group action on parameter manifold) Assume Θ is the measurable parameter manifold with a choice of a nonzero base measure ν. Assume a Lie group G acts transitively on $\Theta _ { i }$ , and ν is relatively invariant with respect to this action with multiplier $\chi : G \to \mathbb { R } _ { > 0 }$ . Because $G$ acts transitively, given any $\pmb { \theta } _ { 0 } \in \Theta$ we can identify $\Theta \cong G / H$ as spaces with a G-action, where $H = \mathrm { S t a b } _ { G } ( \pmb { \theta } _ { 0 } )$

The last sentence of the setup says that we can associate any point $\pmb { \theta } \in \Theta$ with a group element $g \in G$ that satisfies $\pmb { \theta } = \boldsymbol { g } \cdot \pmb { \theta } _ { 0 }$ . The associated group element can be replaced with any other coset representative $g h \in g H \in G / H$

What allows $( q _ { \lambda } ) ^ { g }$ to be another $q _ { \lambda ^ { \prime } }$ is a certain intertwining property of the suficient statistic $\mathbf { T } : \Theta \to \mathrm { V }$ with respect to the action of $g .$ . On the target $\mathrm { V } _ { : }$ , group elements act on vector spaces by representations $\pi : G \to \mathrm { G L } ( \mathrm { V } )$ . So we seek a special kind of suficient statistic such that ${ \bf T } ( g \cdot \pmb { \theta } )$ can be related to $\pi ( g ) \mathbf { T } ( \pmb { \theta } )$

The harmonic exponential family construction yields this kind of an intertwining map. Start with a group representation $\pi ,$ meaning $\pi ( g )$ are linear transformations on ${ \mathrm { V } } ,$ , i.e. matrices. Let $\alpha : G  \mathrm { V }$ be a 1-cocycle of $\pi ,$ meaning that

$$
{ \pmb { \alpha } } ( g h ) = \pi ( g ) { \pmb { \alpha } } ( h ) + { \pmb { \alpha } } ( g )\tag{14}
$$

for all $g , h \in G$ . Furthermore, if we assume ${ \pmb { \alpha } } | _ { H } \equiv 0$ then α induces a map $\mathbf { T } : \Theta \to \mathrm { V } .$ which also satisfies $\mathbf { T } ( g \cdot \pmb { \theta } ) = \pi ( g ) \mathbf { T } ( \pmb { \theta } ) +$ const (with the constant being precisely $\alpha ( g ) )$ . With such a suficient statistic T, elements of the exponential family $q _ { \lambda } \in \mathrm { E F } ( \mathbf { T } )$ satisfy

$$
\begin{array} { r l } & { q _ { \boldsymbol { \lambda } } ( g ^ { - 1 } \cdot \pmb { \theta } ) \propto \exp \left( - \langle \pmb { \lambda } , \mathbf { T } ( g ^ { - 1 } \cdot \pmb { \theta } ) \rangle \right) } \\ & { \qquad \propto \exp \left( - \langle \pmb { \lambda } , \pi ( g ) ^ { - 1 } \mathbf { T } ( \pmb { \theta } ) \rangle \right) } \\ & { \qquad = \exp \left( - \langle \pi ^ { * } ( g ) \pmb { \lambda } , \mathbf { T } ( \pmb { \theta } ) \rangle \right) \propto q _ { \pi ^ { * } ( g ) \boldsymbol { \lambda } } ( \pmb { \theta } ) . } \end{array}\tag{15}
$$

Here $\pi ^ { * } : G \to \mathrm { G L } ( \mathrm { V } ^ { * } )$ is the dual representation defined via $\langle \pi ^ { * } ( g ) \lambda , \mathbf { v } \rangle = \langle \lambda , \pi ( g ) ^ { - 1 } \mathbf { v } \rangle$ This makes the exponential family closed under the pushforward action. And the new natural parameter is given explicitly by the multiplication of the matrix $\pi ^ { * } ( g )$ with the old natural parameter vector λ. We collect the notation in the setup below.

Setup 2 (Group representation and cocycles) Let G be a connected Lie group and H a closed subgroup, V a finite dimensional real vector space, and $\pi : G \to \mathrm { G L } ( \mathrm { V } )$ a group representation. Let α be a 1-cocycle of $\pi \ ( i . e .$ . satisfying Eq. 14) vanishing on H. Using the notation from Setup $\quad 1 , \alpha : G \to \mathrm { V }$ induces a map $\mathbf { T } : \Theta \to \mathrm { V }$ using the identification of $\Theta \cong G / H$ as above.

We denote by $W _ { 0 } = W _ { 0 } ( G , H )$ the continuous group homomorphisms $\psi : G \to \mathbb { R }$ that vanish on H, that is $\psi ( g _ { 1 } g _ { 2 } ) = \psi ( g _ { 1 } ) + \psi ( g _ { 2 } )$ and $\psi ( h ) = 0$ for all $h \in H$ . These morphisms form a vector space. Furthermore they can be considered as real valued functions on Θ, satisfying $\psi ( g \pmb { \theta } ) = \psi ( g ) + \psi ( \pmb { \theta } )$

Definition 3 Assume the setup and the notation from setups 1 and 2. A harmonic exponential family on Θ is a family of distributions of the form

$$
q _ { \eta } ( \pmb { \theta } ) \propto \mathrm { e x p } \left( - \langle \lambda , \mathbf { T } ( \pmb { \theta } ) \rangle - \psi ( \pmb { \theta } ) \right)\tag{16}
$$

parametrized by $\pmb { \eta } = ( \lambda , \psi ) \in \ V ^ { * } \oplus W _ { 0 }$

The normalizable distributions among Eq. 16 have parameters $\pmb { \eta } = ( \lambda , \psi )$ satisfying $\lambda \in \Omega _ { e ^ { - \psi } \nu , { \bf T } } \subseteq \mathrm { V } ^ { * }$ . Under the group pushforward action $\psi$ stays fixed; if $q _ { \eta ^ { \prime } } = ( q _ { \eta } ) ^ { g }$ then $\pmb { \eta } ^ { \prime } = ( \pi ^ { \ast } ( g ) \lambda , \psi )$ . We will consider exponential families which are transitive under the group action (so that the LG-BLR updates can reach every member of the family). Thus, the homomorphism component of the parameter is fixed in our posterior candidates, and we simply write $q _ { \lambda }$ in place of $q _ { \eta } ,$ , dropping $\psi$ from the notation. Note also that, if desired, the homomorphisms $\psi : G \to \mathbb { R }$ can be subsumed in the base measure, i.e. $\nu ^ { \prime } = e ^ { - \psi } \nu .$ And $\nu ^ { \prime }$ is still a relatively invariant measure with respect to the action of $G ,$ with multiplier $\chi ^ { \prime } = e ^ { - \psi } \chi$ . Thus by only a change of base measure we assume from now on that $\psi \equiv 0$

## 4.2 Some Examples of Harmonic Exponential Families

For concreteness, we note how some well known exponential families are indeed realized as harmonic exponential families. The reader may skip these examples on a first reading.

Example 2 (Gamma Distributions) Gamma distributions are of the form

$$
\begin{array} { r } { f _ { \alpha , \lambda } ( x ) \mathrm { d } x \propto x ^ { \alpha - 1 } e ^ { - \lambda x } \mathrm { d } x = \exp ( - \lambda x + \alpha \log ( x ) ) \frac { \mathrm { d } x } { x } } \end{array}
$$

which is in the form of a harmonic exponential family with the invariant base measure $\mathrm { d } \nu ( x ) = \mathrm { d } x / x$ , the homomorphism $\psi ( x ) = - \alpha \log ( x )$ , and suficient statistic $T ( x ) = x$ Here the group is $G = \mathbb { R } _ { > 0 }$ under multiplication and $H \ : = \ : \{ 1 \}$ is the trivial subgroup. The representation is one dimensional with $\pi ( g ) = g .$ , and the relevant 1-cocycle we use is $\alpha ( g ) = g - 1$ . It satisfies $\pmb { \alpha } ( g h ) = g h - 1 = g ( h - 1 ) + ( g - 1 ) = \pi ( g ) \pmb { \alpha } ( h ) + \pmb { \alpha } ( g )$

Note that the group action fixes the shape parameter α, i.e. $( f _ { \alpha , \lambda } ) ^ { g } ( x ) \ = \ { \textstyle { \frac { 1 } { q } } } f _ { \alpha , \lambda } ( { \frac { x } { q } } ) \ =$ $f _ { \alpha , \frac { \lambda } { q } } ( x )$ . This amounts to the group action fixing the $\psi \in W _ { 0 }$ component of the harmonic exponential family parametrization.

Example 3 (Rayleigh Distribution) Let $G = \mathbb { R } _ { > 0 } , H = \{ 1 \}$ so that $\Theta \cong G / H \cong \mathbb { R } _ { > 0 }$ as well. Consider the 1 dimensional representation $\pi ( g ) = g ^ { 2 }$ The 1-cocycle $\alpha ( g ) =$ $( g ^ { 2 } - 1 ) / 2$ associated to $\pi$ vanishes on H. The measure $\nu = x \mathrm { d } x$ is relatively invariant with multiplier $\chi ( g ) = g ^ { 2 }$ . (Alternatively we could have fixed the homomorphism $\psi ( x ) = -$ log x and chosen $\nu = \mathrm { d } x . \nearrow$ With these choices, the resulting harmonic exponential family is

$$
q _ { \lambda } ( x ) \mathrm { d } \nu ( x ) = \lambda x e ^ { - \lambda \frac { x ^ { 2 } } { 2 } } \mathrm { d } x .
$$

The normalizable set of natural parameters are $\Omega _ { \nu , \alpha } = \{ \lambda \in \mathbb { R } : \lambda > 0 \}$ . The family is closed under the pushforward action by $^ { g , }$ as

$$
g _ { \sharp } ( q _ { \lambda } \mathrm { d } \nu ) = \frac { 1 } { \chi ( g ) } q _ { \lambda } ( g ^ { - 1 } x ) \mathrm { d } \nu ( x ) = \frac { x } { g ^ { 2 } } \lambda e ^ { - \lambda x ^ { 2 } / 2 g ^ { 2 } } \mathrm { d } x = q _ { \frac { \lambda } { g ^ { 2 } } } ( x ) \mathrm { d } \nu = q _ { \pi ^ { \ast } ( g ) \lambda } ( x ) \mathrm { d } \nu .
$$

Here the dual action on $\mathrm { V } ^ { \ast } \cong \mathbb { R }$ is given by the $1 \times 1$ matrix $\pi ^ { * } ( g ) = \pi ( g ^ { - 1 } ) ^ { \top } = 1 / g ^ { 2 }$

Example 4 (Poincar´e distribution) The following family is called the family of Poincar´e distributions on the upper half plane ${ \mathcal { H } } = \{ z = x + i y \mid x , y \in \mathbb { R } , y > 0 \}$

$$
\left\{ { \frac { D e ^ { 2 D } } { \pi } } \exp \left( - { \frac { a ( x ^ { 2 } + y ^ { 2 } ) + 2 b x + c } { y } } \right) { \frac { d x d y } { y ^ { 2 } } } \right\} _ { { \Big [ } { \frac { a ~ b } { b ~ c } } { \Big ] } \in \mathrm { S y m } ^ { + } ( 2 , \mathbb { R } ) } ,
$$

where $D = { \sqrt { \operatorname* { d e t } \left[ { \frac { a \ b } { b \ c } } \right] } } = { \sqrt { a c - b ^ { 2 } } }$ and $\mathrm { S y m } ^ { + } ( 2 , \mathbb { R } )$ denotes the set of positive definite symmetric matrices of size 2. This family is a harmonic exponential family on H. Let $G = \mathrm { { S L } _ { 2 } ( \mathbb { R } ) }$ and $H = \mathrm { S O _ { 2 } ( \mathbb { R } ) }$ . The Lie group $G$ acts transitively on the upper half plane $\mathcal { H } : = \{ z = x + i y \mid x , y \in \mathbb { R } , y > 0 \}$ by linear fractional transformation. Then we have $\operatorname { S t a b } _ { G } ( i ) = H$ . Therefore we can identify $\mathrm { S L _ { 2 } ( \mathbb { R } ) / S O _ { 2 } ( \mathbb { R } ) }$ with H via the following $G -$ equivariant map:

$$
\begin{array} { r } { \mathcal { H }  \mathrm { S L } _ { 2 } ( \mathbb { R } ) / \mathrm { S O } _ { 2 } ( \mathbb { R } ) , } \\ { z = x + i y \mapsto [ \sqrt { y } \quad \frac { x } { \sqrt { y } } ] ^ { } } \\ { 0 \quad \frac { 1 } { \sqrt { y } } \biggr ] } \end{array}
$$

Let $\pi \colon G \ \to \ \mathrm { G L } ( \mathrm { S y m } ( 2 , \mathbb { R } ) )$ (so $\mathrm { ~ V ~ } = \mathrm { { \ S y m } ( 2 , \mathbb { R } ) }$ is the three dimensional vector space of matrices) be the representation defined $b y \ \pi ( g ) S \ = \ g S g ^ { T } \ f o r S \ \in \ \mathrm { S y m } ( 2 , \mathbb { R } )$ , and $\alpha \colon G \to \mathrm { S y m } ( 2 , \mathbb { R } )$ the 1-cocycle given by ${ \pmb { \alpha } } ( g ) : = \pi ( g ) I _ { 2 } - I _ { 2 }$ , which satisfies ${ \pmb { \alpha } } | _ { H } = 0$ . We take an invariant measure $\textstyle { \frac { d { \bar { x } } d y } { y ^ { 2 } } }$ with respect to the $\operatorname { S L _ { 2 } } ( \mathbb { R } )$ -action, and $\psi \equiv 0$ (The homomorphism from $\operatorname { S L _ { 2 } } ( \mathbb { R } )$ to R must be zero). In this case $\mathrm { V } ^ { \ast }$ can also be identified with V and normalizability gives us positive definite symmetric matrices as $\operatorname { S y m } ^ { + } ( 2 , \mathbb { R } ) = \Omega \subset \operatorname { V } ^ { * }$

The resulting harmonic exponential family is the family of Poincar´e distributions. By the change of variable,

$$
\begin{array} { r l } & { \mathrm { S y m } ^ { + } ( 2 , \mathbb { R } ) \simeq \mathbb { R } _ { > 0 } \times \mathcal { H } , } \\ & { \quad \begin{array} { r l } { \binom { a } { b } } & { \mapsto ( \rho , \xi + i \eta ) : = \left( \sqrt { a c - b ^ { 2 } } , - \frac { b } { a } + i \frac { \sqrt { a c - b ^ { 2 } } } { a } \right) } \end{array} } \end{array}
$$

the Poincar´e distribution can be written as the following “normal-like” form

$$
\frac { \rho } { \pi } \exp \left( { - \frac { \rho } { y \eta } | z - w | ^ { 2 } } \right) \frac { d x d y } { y ^ { 2 } } ,
$$

where $\rho \in \mathbb { R } _ { > 0 } , \ w = \xi + i \eta \in \mathcal { H }$ . See Nielsen and Okamura $( 2 0 \% )$ , Tojo and Yoshino (2026b) for more information.

Example 5 Multivariate Gaussian distributions form a harmonic exponential family. In this case $G = \mathrm { A f f } _ { n } ^ { + } ( \mathbb { R } )$ , realized as square matrices of size $( n { + } 1 )$ of the form $\begin{array} { r } { g = \left( \begin{array} { c } { A } \\ { \mathbf { 0 } ^ { \top } } \end{array} \right) } \end{array}$ , for $A \in \operatorname { G L } _ { n } ( \mathbb { R } )$ with positive determinant and $\ b { b } \in \mathbb { R } ^ { n }$ . The subgroup $H \cong \operatorname { G L } _ { n } ^ { + } ( \mathbb { R } )$ is embedded as $\{ \big ( _ { \mathbf { 0 } } ^ { \mathrm { ~ \tiny ~ A ~ } } \mathbf { 0 } \big ) : A \in \mathrm { G L } _ { n } ^ { + } ( \mathbb { R } ) \}$ . The sample space is $\Theta : = G / H \cong \mathbb { R } ^ { n }$ , where from the quotient we can pick representatives with $A = I$ and identify any $\ b { b } \in \mathbb { R } ^ { n }$ with $\left\{ \left( { \begin{array} { c } { I } \\ { \mathbf { 0 } ^ { \top } } \end{array} } _ { 1 } ^ { b } \right) : b \in \mathbb { R } ^ { n } \right\}$ ， and vice versa. Set $\psi \equiv 0$

The action of $g \in G$ on $\pmb { \theta } \in \mathbb { R } ^ { n }$ is $g \cdot \pmb { \theta } = A \pmb { \theta } + b$ , which can also be realized as the matrix multiplication by viewing $\pmb \theta \in \mathbb { R } ^ { n }$ as an element of $G / H$

Let $\mathrm { V } = \mathrm { S y m } ( n + 1 , \mathbb { R } )$ , the vector space of $( n + 1 )$ -dimensional symmetric matrices. The (standard) Frobenius inner product on V is $\langle A , B \rangle _ { F r o b } = \mathrm { T r } ( A ^ { \top } B ) = \mathrm { T r } ( A B )$ . This is the same as the Euclidean inner product on the coordinates of the matrices, when flattened. Under this inner product the dual space can be given by the same symmetric matrix space

$$
\mathrm { V } ^ { \ast } \cong \operatorname { S y m } ( n + 1 , \mathbb { R } ) = \{ { \binom { \Lambda } { \lambda ^ { \top } } } \ \sp { \lambda }  c \in \mathbb { R } , \lambda \in \mathbb { R } ^ { n } , \Lambda \in \operatorname { S y m } ( n , \mathbb { R } ) \} .
$$

G acts on V by transpose conjugation, i.e. for symmetric matrices $S ~ \in ~ \mathrm { V }$ we have the representation $\pi ( g ) S = g S g ^ { \top }$ of $\mathrm { A f f } _ { n } ^ { + } ( \mathbb { R } )$ . The dual representation on $\mathrm { V } ^ { \ast }$ is given by $\pi ^ { * } ( g ) S = g ^ { - \top } S g ^ { - 1 }$ for S now viewed as an element $o f \mathrm { V } ^ { \ast }$

A 1-cocycle for this representation is $\pmb { \alpha } ( g ) : = ( g \pmb { e } ) ( g \pmb { e } ) ^ { \top } - e \pmb { e } ^ { \top }$ for e equal to the $( n + 1 ) ^ { t h }$ standard basis vector, i.e. $e = e _ { n + 1 } = { \binom { \mathbf { 0 } } { 1 } }$ . We encourage the reader to check the cocycle condition $\pmb { \alpha } ( g \cdot \pmb { \theta } ) = \pi ( g ) \pmb { \alpha } ( \pmb { \theta } ) + \pmb { \alpha } ( g )$ as well as that α vanishes on H. The suficient statistic coming from this 1-cocycle is therefore

$$
\begin{array} { r } { T _ { \alpha } ( \theta ) = \left( \begin{array} { l l } { \theta \theta ^ { \top } } & { \theta } \\ { \theta ^ { \top } } & { 0 } \end{array} \right) = \left( \begin{array} { l } { \theta } \\ { 1 } \end{array} \right) \left( \theta ^ { \top } } & { 1 \right) - \left( 0 \quad 0 \right) . } \end{array}
$$

The distributions in the harmonic exponential family generated from this data are

$$
\begin{array} { r } { q _ { \left( \underset { \vphantom { \ d \bigg ( } \lambda ^ { \top } } { \Lambda } \ \frac { \lambda } { c } \right) } ( \theta ) \propto \exp \left( - \left. \left( \underset { \vphantom { \ d \bigg ( } \lambda ^ { \top } } { \Lambda } \begin{array} { l l } { \lambda } \\ { c } \end{array} \right) , \begin{array} { l l } { \left( \theta \theta ^ { \top } \right)} & { \theta } \\ { \theta ^ { \top } } & { 0 } \end{array}  \right. _ { \scriptscriptstyle { F r o b } } \right) = \exp \left( - \theta ^ { \top } \Lambda \theta - 2 \lambda ^ { \top } \theta \right) . } \end{array}
$$

On the second equality we used the cyclic property of trace. The right hand side is normalizable if and only if Λ is positive definite (defining $\Omega _ { \mathrm { d } \theta , T } )$ . Furthermore this is precisely the multivariate Gaussian distribution $\mathcal { N } ( \pmb { \theta } | \mathbf { \mathscr { m } } , \Sigma )$ with $\begin{array} { r } { \Sigma = \frac { 1 } { 2 } \Lambda ^ { - 1 } } \end{array}$ and $m = - \Lambda ^ { - 1 } \lambda$

Note that c does not appear on the right hand side, so we do not have a minimal representation, this can be fixed by considering the subspace of V with lower right corner equal to 0. This subspace is preserved by $\pi ( g )$ with $g \in \mathrm { A f f } _ { n } ^ { + } ( \mathbb { R } )$

The dual representation $\pi ^ { * } ( g )$ can also be explicitly calculated by matrix multiplication

$$
\pi ^ { * } ( g ) \left( { \small \begin{array} { c c } { \Lambda } & { \lambda } \\ { \lambda ^ { \top } } & { 0 } \end{array} } \right) = g ^ { - \top } \left( { \small \begin{array} { c c } { \Lambda } & { \lambda } \\ { \lambda ^ { \top } } & { 0 } \end{array} } \right) g ^ { - 1 } = \left( { \small \begin{array} { c c } { A ^ { - \top } \Lambda A ^ { - 1 } } & { - A ^ { - \top } \Lambda A ^ { - 1 } b + A ^ { - \top } \lambda } \\ { - b ^ { \top } A ^ { - \top } \Lambda A ^ { - 1 } + \lambda ^ { \top } A ^ { - 1 } } & { } \end{array} } \right)
$$

where we do not need to keep track of the lower right entry. This is the action on the natural parameters, downstream of the afine action $\pmb { \theta } \mapsto \boldsymbol { A } \pmb { \theta } + \pmb { b }$ on the parameters. Particularly

$$
\Lambda ^ { n e w } = A ^ { - \top } \Lambda A ^ { - 1 } \quad \quad \quad a n d \quad \quad \quad \lambda ^ { n e w } = - \Lambda ^ { n e w } b + A ^ { - \top } \lambda
$$

from which we can write in the new Gaussian’s mean-covariance coordinates $\Sigma ^ { n e w } = A \Sigma A ^ { \top }$ and $m ^ { n e w } = A m + b$ . This sanity check confirms the mapping of natural parameters.

## 4.3 Characterizing Exponential Families Closed under a Group Action

The work of Tojo and Yoshino (2021) characterizes the exponential families closed under the pushforward action of a Lie group as harmonic exponential families.

Theorem 4 (Tojo & Yoshino) Assume Setup 1 and its notation. Also assume that Q is an exponential family on $\Theta$ which is closed under the pushforward action of G, and the action is transitive on $\mathcal { Q }$ . Then $\mathcal { Q }$ is a harmonic exponential family. In other words, if the harmonic exponential family is constructed from π, α in the notation of Setup ${ \mathcal { Q } } ,$ then $\mathcal { Q } = \{ q _ { \lambda } e ^ { - \psi } : \lambda \in \Omega ( \mathcal { Q } ) \}$ where

$$
\Omega ( \mathcal { Q } ) = \{ \pi ^ { * } ( g ) \lambda _ { 0 } : g \in G \} \subseteq \Omega\tag{17}
$$

for some $\lambda _ { 0 } \in \Omega = \Omega _ { \nu e ^ { - \psi } , { \bf T } } ;$ , and $\psi \in W _ { 0 }$ . Here T is obtained by factoring $\alpha : G  V$ through the canonical projection $G \twoheadrightarrow G / H \cong \Theta$ . The homomorphism $\psi : G \to \mathbb { R }$ is fixed throughout and can be subsumed in the base measure ν if desired.

Therefore we can concretely write the distributions in the intersection of the BLR setting of Khan and Rue (2023) and the LG-BLR setting of Kıral et al. (2023). Proof [Proof of Theorem 4] Take any $q _ { 0 } \in \mathcal { Q }$ By Lemma 5, there exist a relatively G-invariant measure ν on Θ, a finite dimensional representation π : $G  G L ( \mathrm { V } )$ and a 1- cocycle α of $\pi$ which vanishes on H (inducing $\pmb { T } : \Theta  \mathrm { V } )$ such that $q _ { 0 } = q _ { \lambda _ { 0 } } e ^ { - \psi }$ for some $\lambda _ { 0 } \in \Omega _ { \nu , T }$ and $\psi \in W _ { 0 }$ . The orbit of $\lambda _ { 0 }$ under $\pi ^ { * } ( g )$ forms $\Omega ( \mathcal { Q } )$ , and is contained in the full harmonic exponential family formed by π and α. The action of $g$ fixes $\psi$ and it can be subsumed in the measure ν.

Lemma 5 (Tojo and Yoshino (2026a, Theorem 2)) Assume the notation in Setup 1.   
Let ${ \mathcal { P } } \subset { \mathcal { R } } ( \Theta )$ be an exponential family on Θ, closed under the pushforward action of G.   
Here R(Θ) denotes the set of all Radon measures on Θ.

Then P is contained in a harmonic exponential family, with notation as in Setup 2. In particular, there exist $\pi \colon G \to \mathrm { G L } ( \mathrm { V } )$ and 1-cocycle α: $G  \mathrm { V }$ for π such that any $q _ { 0 } \in \mathcal { P }$ is of the form in Eq. 16.

The heuristic proof for the above theorem is as follows. Members of an exponential family are of the form $q _ { \lambda } \propto e ^ { - f _ { \lambda } }$ where $f _ { \lambda } ( { \pmb \theta } ) = \langle \lambda , { \bf T } ( { \pmb \theta } ) \rangle$ is in a finite dimensional function space $\mathcal { F }$ spanned by $T _ { 1 } ( \pmb \theta ) , T _ { 2 } ( \pmb \theta ) , \dots , T _ { d } ( \pmb \theta )$ and the constant function 1. Therefore if an exponential family is closed under the pushforward action of $g \in G$ , then space of functions $\mathcal { F }$ is closed under the action $( \pi ( g ) \cdot f ) ( \pmb { \theta } ) : = f ( g ^ { - 1 } \cdot \pmb { \theta } )$ . From this representation of G on $\mathcal { F }$ we construct and ${ \pmb { \alpha } } ( g ) = \pi ( g ) f _ { 0 } - f _ { 0 }$ for some nonconstant $f _ { 0 } \in \mathcal { F }$ . This is the heuristic behind why exponential families closed under the action of G admit such an algebraic structure.

## 4.4 Properties of Harmonic Exponential Families

Given T as above—related to a base measure ν, a representation $\pi$ and its 1-cocycle α— start with a probability distribution $q _ { \lambda _ { 0 } } \in \mathrm { E F } _ { \nu } ( \mathbf { T } )$ for $\lambda _ { 0 } \in \Omega$ . Given this distribution as the starting point (denoted by $q _ { 0 }$ in Section 3.2 above), form its G-orbit Q. Let ${ \boldsymbol { q } } _ { \lambda ^ { \prime } } ( \pmb { \theta } ) =$ $\begin{array} { r l } { ( q _ { \pmb { \lambda } _ { 0 } } ) ^ { g } ( \pmb { \theta } ) = \frac { 1 } { \chi ( g ) } q _ { \pmb { \lambda } _ { 0 } } ( g ^ { - 1 } \cdot \pmb { \theta } ) } \end{array}$ , then direct computation yields,

$$
\begin{array} { c } { { \lambda ^ { \prime } = \pi ^ { * } ( g ) \lambda _ { 0 } } } \\ { { \mu ^ { \prime } = \pi ( g ) \mu _ { 0 } + \alpha ( g ) } } \\ { { A ( \lambda ^ { \prime } ) = A ( \lambda _ { 0 } ) + \langle \lambda _ { 0 } , \alpha ( g ^ { - 1 } ) \rangle + \log \chi ( g ) } } \end{array}
$$

where $\pmb { \mu } ^ { \prime } , \pmb { \mu } _ { 0 } \in \mathrm { ~ V ~ }$ are the expectation parameters of $q _ { \lambda ^ { \prime } }$ and $q _ { \lambda _ { 0 } }$ . Note that the matrix multiplying the expectation parameter is dual (inverse transpose) to the one multiplying the natural parameter, so the parameters λ and µ are not only duals of one another in Fenchel sense but also in how they get transformed by a Lie group element.

The orbit $\Omega ( \mathcal { Q } ) : = \{ \pi ^ { * } ( g ) \lambda _ { 0 } : g \in G \} \subseteq \Omega$ of a given $\lambda _ { 0 } \in \Omega$ forms the candidate natural parameters to choose from. For BLR, since the update in Eq. 4 is linear in λ, in order to guarantee $\lambda ^ { \mathrm { u p d a t e d } } \in \Omega ( \mathcal { Q } )$ at least for some small enough step size, we need the orbit to be open (i.e. for Q to be an open exponential family), or be relatively open in an afine subset of Ω. For example, for normal distributions $\Omega = \{ \lambda = ( \lambda _ { 1 } , \lambda _ { 2 } ) | \lambda _ { 2 } > 0 \}$ is open, and the latter happens for fixed variance Gaussian distributions. In the latter case, as we have shown in Example 1, we can switch to a minimal realization $R _ { \mathrm { m i n } }$ and get an open $\Omega _ { R _ { \mathrm { m i n } } } ( \mathcal { Q } )$ . For LG-BLR, the openness of the parameter space is not required, since $\pi ^ { * } ( g ^ { \mathrm { u p d a t e d } } ) \lambda _ { 0 }$ remains on the orbit $\Omega ( \mathcal { Q } )$ , even for curved exponential families. We depicted one such illustrative natural parameter orbit above in Fig. 5 with the orange curve.

![](images/076f6bf4d07eb3fc76874a08cceb5eabc7fab229042a061ef318b6e84a5da132.jpg)  
Figure 5: The orbit $\Omega ( \mathcal { Q } )$ of $\lambda _ { 0 } \in \Omega$ under the action of $\pi ^ { * }$ stays in Ω.

![](images/0001bb13a6e75993654634a36c64be434b651293bfc281e2358a2643cd428198.jpg)  
Figure 6: Composition of maps that turn the variational inference problem into an optimization problem near the group identity element. We optimize $\mathcal { E } ( q _ { \pi ^ { * } ( g ) \lambda _ { 0 } } )$ by taking a small step $g \mapsto g \exp ( - \alpha Y )$ for some $Y \in { \mathfrak { g } }$ . We choose $Y = { \hat { Y } }$ as the direction of fastest ascent for $\mathcal { E } \circ ( \iota \circ \mathbf { N } ) \circ L _ { g }$ at e computed with respect to a bilinear form on ${ \mathfrak { g } } = T _ { e } G$ which is the pullback of the Fisher information metric.

The harmonic exponential family framework gives us a refinement of Figure 2 whereby we factor the parametrization $\varphi : G  \mathcal { Q }$ into $\varphi = \iota \circ \mathbf { N }$ . Here (having fixed a $\lambda _ { 0 } )$ we define the map $\mathbf { N } : G  \mathrm { V } ^ { * }$ as

$$
\mathbf { N } ( g ) : = \pi ^ { * } ( g ) \lambda _ { 0 } .\tag{18}
$$

This gives the natural parameter associated to a group element, and

$$
\begin{array} { r l } { \iota : \Omega _ { \nu , \mathbf { T } } \longrightarrow \mathrm { E F } _ { \nu } ( \mathbf { T } ) } \\ { \lambda \longmapsto q _ { \lambda } } \end{array}
$$

produces the distribution when given a natural parameter.

## 4.5 Lie Group Update on Harmonic Exponential Families

After establishing the explicit map between the natural parameters and the group elements, we can write the Lie group update in terms of λ. If we start at $\lambda = \mathbf { N } ( g )$ , applying N to both sides of the group update given by Eq. 7, we get the new natural parameters as

$$
\lambda ^ { \mathrm { u p d a t e d } } = \mathbf { N } ( g \exp ( - \alpha \hat { Y } ) ) ,\tag{19}
$$

where $\hat { Y }$ is given as above in $\operatorname { E q . 8 }$ , and $\alpha > 0$ is the step size.

As the main result of this paper, we show that for an open exponential family $\mathcal { Q }$ that is also in the LG-BLR setting, the “linearization” of the exponential map in Eq. 19 gives the BLR update in λ, i.e. it gives the update in Eq. 4.

As an aside, in Sec. $\mathrm { A . 3 }$ we see what happens in the case of linearizing Eq. 19 for curved exponential families. The LG-BLR setting does not require $\mathcal { Q }$ to be an open exponential family, it can be curved. In such a case, it turns out that the linearization of the λ update is the orthogonal projection (with respect to the Fisher metric at $\lambda \in \Omega )$ of the BLR update term onto the tangent space of $\Omega ( \mathcal { Q } )$ embedded in $\Omega .$

In the next section we precisely write what we mean by the term linear approximation to the exponential in Eq. 19.

## 5 BLR as a Linear Approximation of LG-BLR

## 5.1 The Linearization

Assume the notation in Setup 1, and that there is an open exponential family Q that is closed and in fact a single orbit under the Lie group’s pushforward action. In short we are in the joint setting where both BLR and LG-BLR are applicable. By Theorem 4 this family is a subset of a harmonic exponential family. Write this harmonic exponential family using the notation of Setup 2.

We now show that the linearization of Eq. 19 gives the BLR given by Eq. 4.

The representation $\pi ^ { * } : G \to \mathrm { G L } ( \mathrm { V } ^ { * } )$ is a homomorphism of Lie groups and its diferential (at identity) is a linear map $\mathrm { d } \pi ^ { * } : { \mathfrak { g } }  { \mathfrak { g l } } ( \mathrm { V } ^ { * } )$ . Here ${ \mathfrak { g l } } ( \mathrm { V } ^ { * } )$ is simply the space of all linear maps from ${ \mathrm { V } } ^ { * }$ to itself, i.e. in coordinates, all dim $\mathrm { \Omega _ { 1 } ( V ^ { \ast } ) \times d i m ( V ^ { \ast } ) }$ matrices. The exponential maps, the group homomorphism $\pi ^ { * }$ and its diferential $\operatorname { d } \pi ^ { * }$ intertwine as follows

$$
\pi ^ { * } ( \exp ( - \alpha Y ) ) = \exp \left( - \alpha ( \mathrm { d } \pi ^ { * } [ Y ] ) \right) .\tag{20}
$$

This is in fact true for all group homomorphisms, and not just $\pi ^ { * }$ .

The exponential map on the right is matrix exponentiation e $\operatorname { x p } : { \mathfrak { g l } } ( \operatorname { V } ^ { * } ) \to \operatorname { G L } ( \operatorname { V } ^ { * } )$ . It can be given as a Taylor series, and its linear approximation is

$$
\exp ( X ) = e ^ { X } = \sum _ { n = 0 } ^ { \infty } { \frac { X ^ { n } } { n ! } } \approx I + X .\tag{21}
$$

Using this as our linearization we have

$$
\begin{array} { r l } & { \lambda ^ { \mathrm { u p d a t e d } } = \mathbf { N } \left( g \exp ( - \alpha Y ) \right) = \pi ^ { * } ( g \exp ( - \alpha Y ) ) \lambda _ { 0 } = \pi ^ { * } ( g ) \pi ^ { * } ( \exp ( - \alpha Y ) ) \lambda _ { 0 } } \\ & { \qquad = \pi ^ { * } ( g ) e ^ { - \alpha \mathrm { d } \pi ^ { * } [ Y ] } \lambda _ { 0 } \approx \pi ^ { * } ( g ) \left( I - \alpha \mathrm { d } \pi ^ { * } [ Y ] \right) \lambda _ { 0 } . } \end{array}\tag{22}
$$

The first equality is from Eq. 19, and the second is the definition of N in Eq. 18. For the third equality we use that $\pi ^ { * }$ is a group homomorphism. The fourth equality is due to Eq. 20, and the final step is the approximation Eq. 21.

For now, we kept Y arbitrary. In the following subsections we show that if $Y = { \hat { Y } }$ is the direction of steepest ascent with respect to the Fisher metric, we recover the BLR in Eq. 4.

## 5.2 The Direction of Motion

Now let us connect $\hat { Y }$ with $\left. \mathrm { d } ( \mathcal { E } \circ \iota ) \right| _ { \lambda }$ . Recall that $\varphi = \iota \circ \mathbf { N }$ . We call $\left. \operatorname { d } ( \mathbf { N } \circ L _ { g } ) \right| _ { e } = R$ for ease of notation. $R : { \mathfrak { g } } \to T _ { \lambda } \Omega \cong \mathrm { V } ^ { * }$ is a linear map, and we have $\mathrm { d } ( \mathcal { E } \circ \iota \circ \mathbf { N } \circ L _ { g } ) \stackrel { } { = } \mathrm { d } ( \mathcal { E } \circ \iota ) \vert _ { \lambda } R$

Remark 6 In Sec. 2.3 above we made the identification $\mathcal { E } ( \lambda ) = \mathcal { E } ( q _ { \lambda } )$ which amounts to omitting ι. Therefore $\mathrm { d } \mathcal { E } \big | _ { \lambda }$ in $E q$ . 4 corresponds to $\left. \mathrm { i } ( \mathcal { E } \circ \iota ) \right| _ { \lambda }$ in our notation.

The linear map R also relates the bilinear form B on the Lie algebra g with the Fisher metric in the natural parameter space:

$$
B ( X , Y ) = \omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( R X , R Y ) { \mathrm { ~ f o r ~ a l l ~ } } X , Y \in { \mathfrak { g } } .
$$

Proposition 7 Assume that $\mathcal { Q }$ is an open exponential family and that the pushforward action of Lie group G is transitive on $\mathcal { Q }$ . Adopt the above notation. Then any $\hat { Y } \in { \mathfrak { g } } = T _ { e } G$ solving $B ( \hat { Y } , \cdot ) = \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \Delta } R$ satisfies

$$
\begin{array} { r } { R \hat { Y } = \hat { \mathbf { v } } = { F _ { \lambda } } ^ { - 1 } \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \lambda } = \widetilde \nabla \mathcal { E } ( q _ { \lambda } ) \big | _ { \lambda } . } \end{array}
$$

Proof It is possible, in general, that R does not map onto all of $\mathrm { V } ^ { \ast }$ , but to a subspace $\operatorname { I m } ( R ) = W \subseteq \mathrm { V } ^ { * }$ . This is not the case here, as we assume an open exponential family. Put $R \hat { Y } = \hat { \mathbf { v } }$ , and also writing $R X = \mathbf { v }$ (which runs over any $\mathbf { v } \in \mathrm { V } ^ { * }$ as R is onto), we have

$$
\omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( \hat { \mathbf { v } } , \mathbf { v } ) = \omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( R \hat { Y } , R X ) = { B } ( \hat { Y } , X ) = \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \lambda } [ R X ] = \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \lambda } [ \mathbf { v } ] .
$$

Since this should hold for every v and since $\omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( \hat { \mathbf { v } } , \mathbf { v } ) = \hat { \mathbf { v } } ^ { \top } F _ { \lambda } \mathbf { v }$ , we can see that

$$
F _ { \lambda } \hat { \mathbf v } = \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \lambda }
$$

and this proves the result.

## 5.3 Putting it all Together

Let us again write the map $R = \left. \mathrm { d } ( \mathbf { N } \circ L _ { g } ) \right| _ { e }$ , and express $R \hat { Y }$ in terms of its $\pi ^ { * }$ dependence,

$$
\begin{array} { c } { { \displaystyle { \cal R } \hat { \cal Y } = \mathrm { d } ( { \bf N } \circ L _ { g } ) \Big \vert _ { e } [ \hat { \cal Y } ] = \frac { \mathrm { d } } { \mathrm { d } t } \left( ( { \bf N } \circ L _ { g } ) ( \exp ( t \hat { \cal Y } ) ) \right) \Big \vert _ { t = 0 } = \left. \frac { \mathrm { d } } { \mathrm { d } t } \pi ^ { * } ( g \exp ( t \hat { \cal Y } ) ) \lambda _ { 0 } \right. _ { t = 0 } } } \\ { { \displaystyle ~ = \frac { \mathrm { d } } { \mathrm { d } t } \pi ^ { * } ( g ) \exp ( t ( \mathrm { d } \pi ^ { * } [ \hat { \cal Y } ] ) ) \lambda _ { 0 } \Big \vert _ { t = 0 } = \pi ^ { * } ( g ) \mathrm { d } \pi ^ { * } [ \hat { \cal Y } ] \lambda _ { 0 } } } \end{array}\tag{23}
$$

Using Eqs. 19 and 22 and the calculation in Eq. 23 above we express the linearization of the Lie group update rule as

$$
\begin{array} { r l } & { \lambda ^ { \mathrm { u p d a t e d } } = \mathbf { N } ( g \exp ( - \alpha \hat { Y } ) ) \approx \pi ^ { * } ( g ) ( I - \alpha ( \mathrm { d } \pi ^ { * } [ \hat { Y } ] ) ) \lambda _ { 0 } } \\ & { \qquad = \pi ^ { * } ( g ) \lambda _ { 0 } - \alpha \pi ^ { * } ( g ) ( \mathrm { d } \pi ^ { * } [ \hat { Y } ] ) \lambda _ { 0 } } \\ & { \qquad = \lambda - \alpha R \hat { Y } } \\ & { \qquad = \lambda - \alpha \widetilde { \nabla } _ { \lambda } \mathcal { E } ( q _ { \lambda } ) . } \end{array}
$$

In the final line we used Proposition 7.

This establishes the main theorem.

Theorem 8 Given a machine learning model with parameter space Θ and loss function $\ell ,$ assume that a Lie group G acts transitively and smoothly on Θ, as in Setup 1. Let Q be an open exponential family of distributions, on which G acts transitively by pushforwards. Therefore Q is included in a harmonic exponential family given by a group representation π and a 1-cocycle α as in Setup 2. In this setting both BLR given by Eq. 4 and LG-BLR given by Eq. 7 (or Eq. 19 in the natural parameters) exist.

Then, the linearization of the LG-BLR update in the step size is the BLR update.

## 5.4 A Final Remark on Momentum

Applying momentum, or any other optimization method that accumulates past gradients, either implicitly or explicitly requires a notion of transporting tangent vectors living in apriori diferent vector spaces. We discuss in Sec. B the role of flat afine connections and parallel transports in this regard at a general mathematical level.

In applying momentum in the Lie group setting, we accumulate the Lie algebra vectors $\hat { Y } _ { t } ,$ , which is diferent from accumulating the natural gradients when working with exponential families. Indeed from Proposition $^ { 7 }$ we see that $\left. \widetilde { \nabla } \mathcal { E } ( q _ { \lambda } ) \right| _ { \lambda _ { t } } = R _ { t } \hat { Y } _ { t }$ . Even though $R _ { t }$ is linear, it depends on the current point $g _ { t }$ and therefore accumulating the natural gradients themselves will result in a diferent vector. The Lie group momentum keeps track of and accumulates $\hat { Y } _ { t } = R _ { t } ^ { - 1 } \widetilde { \nabla } \mathcal { E } ( q _ { \lambda } ) \vert _ { \lambda _ { t } }$ , taking one of the preimages if $R _ { t }$ is not invertible (In the Gaussian case these $\hat { Y } _ { t }$ are given by the pairs $( Y _ { t } , \pmb { y } _ { t } )$ in Eqs. 12 and 13). Then if we do the linearization as above, the update at timestep T becomes for $0 < \beta < 1$

$$
\lambda _ { T + 1 } = \lambda _ { T } - \alpha R _ { T } \left( \sum _ { t = 1 } ^ { T } \beta ^ { T - t } ( 1 - \beta ) \hat { Y } _ { t } \right) .
$$

This kind of momentum corresponds to using the flat connection on the Lie group that assigns left-invariant vector fields as parallel. This particular Lie group connection is nice because under its parallel transports the length and the angle between the tangent vectors with respect to the Fisher information metric stay unchanged.

However we note that it is not the only kind of connection, and therefore not the only kind of momentum that can be applied on this exponential family. Directly accumulating the natural gradients $\widetilde { \nabla } _ { \lambda }$ corresponds to using the e-connection. We also note what the mixture connection corresponds to in Sec. B when learning on an exponential family. In all cases further stabilizing or adaptive operations can be applied to the accumulated momenta as well—as is done with, say, AdamW or Lion optimizers.

## 6 Discussion

In this work we have shown that LG-BLR is an extension of BLR in a very precise sense. We also introduced the notion of harmonic exponential families to the field of variational inference. These characterize the families of distributions with the desirable structure of a Lie group action, which we saw gives eficient gradient estimation and updates that always yield a valid (normalizable) distribution.

The LG-BLR is extensible, and promises many new capabilities. For example in Nishida et al. (2026) the authors take the LG-BLR for the group of positive reals under multiplication, the family of log-normal distributions, and extend the gradient accumulation method of Lion (Chen et al., 2023), adding regularizers relevant to the family and the group. The resulting optimization method performs stably under low precision floating point representations, and performs well compared to other multiplicative update algorithms. As another possible extension, the use of the Lie Group exponential is but one choice of a retraction, and can be replaced with easier to compute variants that can be obtained from the group exponential.

With this work, we hope that the reader has a better understanding of the LG-BLR both as a standalone method and in how it relates to the existing variational inference optimizer BLR.

## References

Pierre Alquier. User-friendly introduction to PAC-Bayes bounds. Foundations and Trends® in Machine Learning, 17(2):174–303, 2024.

Shun-Ichi Amari. Information Geometry and its Applications. Springer, 2016.

Matthew Ashman, Thang D Bui, Cuong V Nguyen, Stratis Markou, Adrian Weller, Siddharth Swaroop, and Richard E Turner. Partitioned variational inference: A framework for probabilistic federated learning. arXiv:2202.12275, 2022.

Ole Eiler Barndorf-Nielsen. Information and exponential families in statistical theory. Chichester: Wiley, 1978.

Pier Giovanni Bissiri, Chris C Holmes, and Stephen G Walker. A general framework for updating belief distributions. Journal of the Royal Statistical Society Series B: Statistical Methodology, 78(5):1103–1130, 2016.

L´eonard Blier and Yann Ollivier. The description length of deep learning models. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Michael M Bronstein, Joan Bruna, Taco Cohen, and Petar Veliˇckovi´c. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

Olivier Catoni. PAC-Bayesian Supervised Classification. Number 56. Institute of Mathematical Statistics Lecture Notes – Monograph Series, 2007.

Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Taco Cohen and Max Welling. Harmonic exponential families on manifolds. In International Conference on Machine Learning, pages 1757–1765. PMLR, 2015.

Thomas Cover and Joy Thomas. Elements of Information Theory. Wiley, 2nd edition, 2006.

Nico Daheim, Thomas M¨ollenhof, Edoardo M Ponti, Iryna Gurevych, and Mohammad Emtiyaz Khan. Model merging by uncertainty-based gradient matching. In International Conference on Learning Representations, volume 2024, pages 11836–11854, 2024.

Mauro Camara Escudero. Measure Theory for ML, AI and Difusion Models, 2024. URL https://maurocamaraescudero.netlify.app/post/ 2024-02-16-measure-theory-for-machine-learning-and-diffusion-models/. Accessed: Nov 26, 2025.

Alex Graves. Practical variational inference for neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2011.

Peter D Gr¨unwald. The minimum description length principle. MIT Press, 2007.

Gert Heckman. Introduction to Riemannian Geometry. https://www.math.ru.nl/ heckman/DiffGeom.pdf, 2019. Lecture notes, Radboud University Nijmegen.

Geofrey E Hinton and Drew Van Camp. Keeping the neural networks simple by minimizing the description length of the weights. In Conference on Learning Theory (COLT), 1993.

Mohammad Emtiyaz Khan and Haavard Rue. The Bayesian learning rule. J. Mach. Learn. Res. (JMLR), 24(281):1–46, 2023.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

Eren Mehmet Kıral, Thomas M¨ollenhof, and Mohammad Emtiyaz Khan. The Lie-Group Bayesian learning rule. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2023.

Jeremias Knoblauch, Jack Jewson, and Theodoros Damoulas. Generalized variational inference: Three arguments for deriving new posteriors. arXiv:1904.02063, 2019.

Wu Lin, Mark Schmidt, and Mohammad Emtiyaz Khan. Handling the positive-definite constraint in the Bayesian learning rule. In International Conference on Machine Learning (ICML), 2020.

Wu Lin, Valentin Duruisseaux, Melvin Leok, Frank Nielsen, Mohammad Emtiyaz Khan, and Mark Schmidt. Simplifying momentum-based positive-definite submanifold optimization with applications to deep learning. In International Conference on Machine Learning (ICML), 2023.

Michael S Matena and Colin A Rafel. Merging models with Fisher-weighted averaging. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Xiangming Meng, Roman Bachmann, and Mohammad Emtiyaz Khan. Training binary neural networks using the Bayesian learning rule. In International Conference on Machine Learning (ICML), 2020.

Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte Carlo gradient estimation in machine learning. J. Mach. Learn. Res. (JMLR), 21(132):1–62, 2020.

Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, and Richard E. Turner. Variational Continual Learning. ICLR, February 2018. URL https://openreview.net/forum?id= BkQqq0gRb.

Peter Nickl, Lu Xu, Dharmesh Tailor, Thomas M¨ollenhof, and Mohammad Emtiyaz Khan. The memory-perturbation equation: Understanding model’s sensitivity to data. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Frank Nielsen and Kazuki Okamura. Information measures and geometry of the hyperbolic exponential families of Poincar´e and hyperboloid distributions. Inf. Geom., 7:S943–S989, 2024. ISSN 2511-2481,2511-249X. doi: 10.1007/s41884-024-00158-9. URL https://doi. org/10.1007/s41884-024-00158-9.

Keigo Nishida, Eren Mehmet Kıral, Kenichi Bannai, Mohammad Emtiyaz Khan, and Thomas M¨ollenhof. Log-normal multiplicative dynamics for stable low-precision deep learning. In Forty-third International Conference on Machine Learning, 2026.

Yuesong Shen, Nico Daheim, Bai Cong, Peter Nickl, Gian Maria Marconi, Clement Bazan, Rio Yokota, Iryna Gurevych, Daniel Cremers, Mohammad Emtiyaz Khan, and Thomas M¨ollenhof. Variational learning is efective for large deep networks. In International Conference on Machine Learning (ICML), 2024.

Koichi Tojo and Taro Yoshino. Harmonic exponential families on homogeneous spaces. Information Geometry, 4(1):215–243, 2021.

Koichi Tojo and Taro Yoshino. Note on harmonic exponential families on homogeneous spaces. In Geometric science of information. Part III, volume 16035 of Lecture Notes in Comput. Sci., pages 246–253. Springer, Cham, 2026a.

Koichi Tojo and Taro Yoshino. The Fisher metric and the Amari–Chentsov tensor of the family of Poincar´e distributions. In Frank Nielsen and Fr´ed´eric Barbaresco, editors, Geometric Science of Information, pages 254–262, Cham, 2026b. Springer Nature Switzerland.

Martin J Wainwright and Michael I Jordan. Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning, 1(1–2):1–305, 2008.

Ming Xu, Matias Quiroz, Robert Kohn, and Scott A Sisson. Variance reduction properties of the reparameterization trick. In The 22nd international conference on artificial intelligence and statistics, pages 2711–2720. PMLR, 2019.

Arnold Zellner. Optimal information processing and Bayes’s theorem. The American Statistician, 42(4):278–280, 1988.

Tong Zhang. Theoretical analysis of a class of randomized regularization methods. In Conference on Learning Theory (COLT), 1999.

## Acknowledgments and Disclosure of Funding

This work is supported by the Bayes duality project, JST CREST Grant Number JP-MJCR2112. Eren Mehmet Kıral acknowledges the support of JSPS, KAKENHI Grant Number 26K14955 titled “Learning with Distributions and Lie Groups”. Thomas M¨ollenhof acknowledges the support of JSPS KAKENHI Grant Number 26H02541 titled “Variational Learning for Training Better AI Models at Lower Cost”.

## Author Contribution Statement

Authors list: Eren Mehmet Kıral (EMK), Thomas M¨ollenhof (TM), Koichi Tojo (KT), Mohammad Emtiyaz Khan (MEK), Kenichi Bannai (KB).

EMK led the project and wrote most of the manuscript; EMK and TM developed the LG-BLR update; EMK, TM and KT formulated the connection with harmonic exponential families; MEK provided insights on Bayesian Learning; and EMK, TM, KT, KB discussed the mathematical content and shaped the outline of the manuscript. MEK and KB provided the funding throughout the project. The final manuscript was approved by all authors.

## Appendix A. The Lie Group Update Rule

This appendix gives the mathematically careful solution to the constrained maximization problem in Eq. 8, with the bilinear form

$$
B ( X , Y ) : = \omega _ { \mathrm { F i s h e r } } ^ { q } ( \mathrm { d } ( \varphi \circ L _ { g } ) | _ { e } [ X ] , \mathrm { d } ( \varphi \circ L _ { g } ) | _ { e } [ Y ] ) \qquad \mathrm { f o r ~ } q = q ^ { g } .\tag{24}
$$

Because the Fisher metric is invariant under reparametrizations the right hand side is independent of the choice of g and therefore we are justified in not writing a g dependence on the left hand side. The bilinear form B, which is the pullback of the Fisher metric, is not necessarily nondegenerate, even though the Fisher metric is, i.e. the matrix representing B may be not invertible. But regardless, we will show below that the problem in Eq. 8 is solvable. In Kıral et al. (2023) this was not shown in full generality, where we provided solutions only for the groups that were presented.

## A.1 The Diferential of the Variational Objective

For the sake of completeness we calculate the diferential of E in Eq. 1 here. The entropy’s dependence on g is seen to be

$$
\begin{array} { r l } & { - \mathcal { H } _ { \nu } ( q ^ { g } ) = \displaystyle \int _ { \Theta } q ^ { g } ( \theta ) \log q ^ { g } ( \theta ) \mathrm { d } \nu ( \theta ) = \int _ { \Theta } \frac { 1 } { \chi ( g ) } q _ { 0 } ( g ^ { - 1 } \cdot \theta ) \log \big ( \frac { 1 } { \chi ( g ) } q _ { 0 } ( g ^ { - 1 } \cdot \theta ) \big ) \mathrm { d } \nu ( \theta ) } \\ & { \qquad = \displaystyle \int _ { \Theta } q _ { 0 } ( \theta ) \log \big ( \frac { 1 } { \chi ( g ) } q _ { 0 } ( \theta ) \big ) \mathrm { d } \nu ( \theta ) = - \log \chi ( g ) - \mathcal { H } _ { \nu } ( q _ { 0 } ) . } \end{array}
$$

Here the second line is obtained by a change of variables $\theta \mapsto g \cdot \theta$ . Now we can compute that $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \left( - \mathcal { H } _ { \nu } \big ( q ^ { g \mathrm { e x p } ( t X ) } \big ) \right) \big | _ { t = 0 } = - \frac { \mathrm { d } } { \mathrm { d } t } \log \chi ( \mathrm { e x p } ( t X ) ) = : \xi ( X ) } \end{array}$

Assume that Θ is embedded in the Euclidean space so that we can take its gradient ∇ℓ with respect to θ. The directional derivative of the expectation term can also be computed with a change of variables. First let us compute the expectation of the loss under the perturbed distribution

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { q ^ { g \ell \times \Psi ( L ) } } [ \ell ] = \int _ { \Theta } \ell ( \theta ) q ^ { g ^ { \mathrm { e x p } ( t X ) } } ( \theta ) \mathrm { d } \nu ( \theta ) } \\ { \displaystyle = \int _ { \Theta } \ell ( \theta ) \frac { 1 } { \chi ( g \exp ( t X ) ) } q _ { 0 } ( ( g \exp ( t X ) ) ^ { - 1 } \cdot \theta ) \mathrm { d } \nu ( \theta ) } \\ { \displaystyle = \int _ { \Theta } \ell ( \theta ) \frac { 1 } { \chi ( \exp ( t X ) ) } \frac { 1 } { \chi ( g ) } q _ { 0 } ( g ^ { - 1 } \big ( ( g \exp ( t X ) g ^ { - 1 } ) ^ { - 1 } \cdot \theta \big ) ) \mathrm { d } \nu ( \theta ) } \\ { \displaystyle = \int _ { \Theta } \ell ( \theta ) \frac { 1 } { \chi ( \exp ( t X ) ) } q ^ { g } \big ( ( g \exp ( t X ) g ^ { - 1 } ) ^ { - 1 } \cdot \theta \big ) \mathrm { d } \nu ( \theta ) } \\ { \displaystyle = \int _ { \Theta } \ell \big ( ( g \exp ( t X ) g ^ { - 1 } ) \cdot \theta \big ) q ^ { g } \big ( \theta \big ) \mathrm { d } \nu ( \theta ) \mathrm { d } \nu ( \theta ) . } \end{array}
$$

Where in the last line we make the change of variables $\pmb { \theta } \mapsto ( g \exp ( t X ) g ^ { - 1 } ) { \cdot } \pmb { \theta }$ . The multiplier χ exactly accounts for the Jacobian determinant in this change of variables formula. Notice that now the only t dependence is in the ℓ function and if we take the derivative at $t = 0$ 2

$$
\int _ { \Theta } \nabla \ell ( \pmb \theta ) ^ { \top } ( \mathrm { A d } _ { g } ( \boldsymbol X ) \cdot \pmb \theta ) \boldsymbol q ^ { g } ( \pmb \theta ) \mathrm { d } \nu ( \boldsymbol \theta ) .
$$

This is an expectation that can be approximated with stochastic sampling from $q ^ { g }$ , which in turn is equivalent to sampling $\theta _ { 0 } \sim q _ { 0 }$ and applying $g \cdot \pmb { \theta } _ { 0 }$

Here $\operatorname { A d } _ { g } ( X )$ is an element of the Lie algebra given by a diferential, and can be calculated exactly by $\begin{array} { r } { \operatorname { A d } _ { g } ( X ) : = \frac { \mathrm { d } } { \mathrm { d } t } ( g \exp ( t X ) g ^ { - 1 } ) \big \rvert _ { t = 0 } } \end{array}$ , and $\operatorname { A d } _ { g } ( X )$ · θ refers to the infinitesimal action of this Lie algebra vector on θ, defined as a limit, i.e. for $Z \in { \mathfrak { g } }$ the infinitesimal action is given as $\begin{array} { r } { Z \cdot \pmb { \theta } : = \frac { \mathrm { d } } { \mathrm { d } t } ( \exp ( t Z ) \cdot \pmb { \theta } ) \big | _ { t = 0 } \in T _ { \pmb { \theta } } \Theta } \end{array}$ . The gradient $\nabla \ell ( \theta )$ is actually coming from the cotangent vector in $T _ { \theta } ^ { * } \Theta$ so these two quantities can be paired. But as we assumed without loss of generality that Θ is embedded in $\mathbb { R } ^ { n }$ , the pairing is simply the dot product between two vectors in $\mathbb { R } ^ { n }$

## A.2 Solving for the Gradient

Let us show that we can always compute the gradient of $\mathcal { E } ( q ^ { g } )$ with respect to the positive semidefinite metric. With a slight abuse of notation, let $B , P = P _ { g }$ and F denote matrices

corresponding to the bilinear form in Eq. 24 on g, the map $\left. \operatorname { d } ( \varphi \circ L _ { g } ) \right| _ { e }$ and the Fisher metric $\omega _ { \mathrm { F i s h e r } } ^ { q ^ { g } }$ after a choice of bases in g and $T _ { q ^ { g } } \mathcal { Q }$ . Note that if Eq. 24 is written in coordinates, corresponds to the equality $B = P ^ { \top } F P$

Writing the optimization following Eq. 8 in coordinate form, the constrained optimization problem becomes

$$
\mathrm { d } \big ( \mathcal { E } \circ \varphi \circ L _ { g } \big ) \Big | _ { e } [ \mathbf { X } ] = \boldsymbol { \zeta } ^ { \top } P \mathbf { X } \qquad \mathrm { s u b j e c t ~ t o } \qquad \mathbf { X } ^ { T } B \mathbf { X } \leq 1 .
$$

where $\zeta$ is the column-vector form of the linear functional $ \mathcal { E }  _ { q ^ { g } } : T _ { q ^ { g } } \mathcal { Q }  \mathbb { R }$ consisting of partial derivatives in the given basis, and X is a column vector representation of a Lie algebra element in g.

Using the method of Lagrange multipliers, we solve

$$
\begin{array} { r l } { \underset { \mathbf { X } \in \mathfrak { g } } { \arg \operatorname* { m a x } } } & { { } \zeta ^ { \top } P \mathbf { X } + \beta \mathbf { X } ^ { \top } B \mathbf { X } . } \end{array}
$$

The first derivative in X yields the equality $\zeta ^ { \top } P + 2 \beta \mathbf { X } ^ { \top } B = 0$ or in other words

$$
2 \beta P ^ { \top } F P \mathbf { X } = - P ^ { \top } \boldsymbol { \zeta }
$$

which can be solved for all X satisfying $\begin{array} { r } { P \mathbf { X } = - \frac { 1 } { 2 \beta } F ^ { - 1 } \zeta } \end{array}$ . Since $P$ is the matrix of the diferential of $\varphi$ which parametrizes $\mathcal { Q } ,$ it is necessarily surjective, so we can surely solve for X, though not necessarily uniquely. This equation may have more than one solution if dim $\mathcal { Q } <$ dim g, and in that case we just pick any one of the solutions.

Even though in showing that there exists at least one solution to Eq. 8 we used the invertibility of $F ,$ , one does not need to invert the Fisher matrix to do this calculation. One can simply solve the system of equations $\begin{array} { r } { B \mathbf { X } = - \frac { 1 } { 2 \beta } P ^ { \top } \boldsymbol { \zeta } . } \end{array}$ , and $P ^ { \top } \zeta$ is the vector corresponding to $\operatorname { d } ( { \mathcal { E } } \circ \varphi \circ L _ { g } )$ after a choice of basis of g, and B needs to be calculated in a basis of g once and for all, it does not change throughout the optimization.

Furthermore, using the calculations in Sec. A.1 we can directly calculate the components of $P ^ { \top } \zeta$ (without calculating those of $\zeta )$ via the formula

$$
\frac { \mathrm { d } } { \mathrm { d } t } \mathscr { E } ( q ^ { g e ^ { t \mathbf { X } } } ) \Big | _ { t = 0 } = \mathbb { E } _ { q ^ { g } \mathrm { d } \nu } [ \left( \nabla _ { \theta } \ell ( \pmb { \theta } ) \right) ^ { \top } \left( \mathrm { A d } _ { g } ( \mathbf { X } ) \cdot \pmb { \theta } \right) ] + \tau \boldsymbol { \xi } ( \mathbf { X } )
$$

by substituting $\mathbf { X } = \mathbf { X } _ { i }$ , given a basis $\{ \mathbf { X } _ { i } \} _ { i }$ of the Lie algebra.

## A.3 Linearization of LG-BLR for Curved Exponential Families

For this subsection assume that the family Q is not necessarily open. It is a curved exponential family closed under the Lie group action. But the orbit of $\lambda _ { 0 }$ under $\pi ^ { * }$ doesn’t form an open set in $\Omega \subseteq \mathrm { V } ^ { * }$ as depicted in Fig. 5.

Here when we linearize the LG-BLR update, it is clear that the range of motion aforded to us by the Lie group Bayesian learning rule is restricted. Rather than being able to move in any which direction in $\Omega \subseteq \mathrm { V } ^ { * }$ , we can only move along $W = T _ { \lambda } \Omega ( \mathcal { Q } )$ , see Fig. 7.

![](images/bc0e7eed2c7737244d38de322e7adb60c855a554b11386c8a91aa1d57cbe4150.jpg)  
Figure 7: The Lie group BLR can only move along Q, but inside the general exponential family EF(α) the natural parameters can be updated in directions that have more degrees of freedom.

Proposition 9 Assume the setting and notation of Sec. 5.2, let Q be an exponential family where $\Omega ( \mathcal { Q } )$ is an (immersed) submanifold of Ω. Let $W : = T _ { \lambda } ( \Omega ( \mathcal { Q } ) ) \subseteq \ V ^ { * }$ , and let $\hat { Y } \in { \mathfrak { g } }$ be the update direction, satisfying $B ( \hat { Y } , X ) = \operatorname { d } ( \mathcal { E } \circ \varphi \circ L _ { g } ) \big | _ { e } [ X ]$ for all $X \in { \mathfrak { g } }$ . Then,

$$
R \hat { Y } = \operatorname { P r o j } _ { W } \widetilde { \nabla } _ { \lambda } \mathcal { E } ( q _ { \lambda } )
$$

where the orthogonal projection is taken with respect to the inner product given by the Fisher information metric $F _ { \lambda }$ . In particular $i f \ \mathcal { Q }$ is an open exponential family, we have $T _ { \lambda } \Omega ( \mathcal { Q } ) = \mathrm { V } ^ { \ast }$ in which case $\begin{array} { r } { R \hat { Y } = \widetilde \nabla _ { \lambda } \mathcal { E } ( q _ { \lambda } ) = { F } _ { \lambda } ^ { - 1 } \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \lambda } } \end{array}$

Proof First note that since N traces out the natural parameters in $\Omega ( \mathcal { Q } )$ , we have that the image of $R = \left. \mathrm { d } ( \mathbf { N } \circ L _ { g } ) \right| _ { e }$ is W. From the definition of B as a pullback, we get

$$
\mathrm { d } ( \mathcal { E } \circ \iota ) \big \vert _ { \lambda } [ R X ] = \mathrm { d } ( \mathcal { E } \circ \varphi \circ L _ { g } ) \big \vert _ { e } [ X ] = B ( \hat { Y } , X ) = \omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( R \hat { Y } , R X )
$$

So for all $\mathbf { w } \in W$ , the Lie algebra vector $\hat { Y }$ satisfies $\omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( R \hat { Y } , \mathbf { w } ) = \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \lambda } [ \mathbf { w } ]$ . Furthermore by the definition of natural gradient, for all $\mathbf { v } \in \mathrm { V } ^ { * }$ the equality

$$
\omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( \widetilde { \nabla } _ { \lambda } \mathcal { E } ( q _ { \lambda } ) , \mathbf { v } ) = \mathrm { d } ( \mathcal { E } \circ \iota ) \big | _ { \lambda } [ \mathbf { v } ]
$$

holds. This means that $\omega _ { \mathrm { F i s h e r } } ^ { \lambda } ( R \hat { Y } - \widetilde { \nabla } _ { \lambda } \mathcal { E } ( q _ { \lambda } ) , \mathbf { w } ) = 0$ for all $\mathbf { w } \in W$ , meaning that the diference is in $W ^ { \perp }$ , proving the proposition. ■

## Appendix B. Momentum and Connections

When using gradient methods on a diferentiable manifold M (including the standard parameter space $M = \Theta = \mathbb { R } ^ { P } )$ the act of applying momentum, or other acceleration techniques for optimization, involves accumulating past gradients. These gradients are on different tangent spaces; $\pmb { g } _ { \mathrm { p a s t } } \in T _ { p _ { \mathrm { p a s t } } } M$ vs. $\pmb { g } _ { \mathrm { n o w } } \in T _ { \pmb { p } _ { \mathrm { n o w } } } M$ . In order for this accumulation to make sense, we require a way to connect diferent tangent spaces, i.e. an afine connection.

An afine connection is defined as an operator taking in two vector fields $X , Y \in { \mathfrak { X } } ( M )$ and giving another vector field $Z = \nabla _ { X } Y$ satisfying certain conditions. $Z _ { p }$ —the resulting vector field’s value at a point p—is a measure of the change of $Y$ along X, infinitesimally at $\mathbf { \nabla } _ { \mathbf { p } } .$ . For our purposes of connecting diferent tangent spaces, however, the equivalent notion of parallel transport is more relevant.

Given an afine connection ∇ and a smooth path $\gamma : [ 0 , 1 ] \to M$ , the parallel transport along $\gamma$ is a linear map

$$
P _ { \gamma } : T _ { \gamma ( 0 ) } M \to T _ { \gamma ( 1 ) } M
$$

that carries the tangent vectors by solving a diferential equation, $\nabla _ { { \dot { \gamma } } ( t ) } V ( t ) = \mathbf { 0 }$ to be precise. The interpretation of the diferential equation is that according to the afine connection ∇, the vectors $V ( t )$ stay constant along the path $\gamma .$ Setting $V ( 0 ) = \pmb { v } \in T _ { \gamma ( 0 ) } M$ as the initial condition, we get a path of tangent vectors $V ( t )$ transporting v to $V ( \dot { 1 } ) \stackrel { \prime } { \in } T _ { \gamma ( 1 ) } M$ 2 which we define to be the parallel transport $P _ { \gamma } ( \pmb { v } )$

In general, parallel transports depend on the path γ along which vectors are transported. However if the curvature tensor<sup>10</sup> for the given connection is identically zero—in which case we call this a $f l a t$ connection—then for two paths with the same endpoints that can be deformed into each other their parallel transports are the same. This means that, unless there is a global topological obstruction such as in the handle of a cofee mug, for a flat connection we can talk about the parallel transport. See, e.g. p.36 in Heckman (2019).

For a Lie group G, we can declare left invariant vector fields to be parallel by setting $\nabla _ { X } Y = \mathbf { 0 } $ for $X , Y$ left invariant<sup>11</sup>. This is enough to define a flat afine connection. Flatness follows directly from the definition of the curvature tensor using that the Lie bracket of two left-invariant vector fields is also left-invariant. The parallel transports along all paths (locally in contractible patches) are given by the maps $\mathrm { d } L _ { g }$

In the LG-BLR we expressed our updates at any point g<sub>t</sub> by $g _ { t + 1 } = g _ { t } \exp ( - \eta \hat { Y } _ { t } )$ for an element $\hat { Y } _ { t } \in \mathfrak { g }$ . Given the flat connection defined in the above paragraph, the identification of the gradient at $g _ { t }$ to the vector $\hat { Y } _ { t }$ in the Lie algebra ${ \mathfrak { g } } = T _ { e } G$ corresponds to the parallel transport. As a side note, we remark that the retraction used in this update given by the exponential map is a geodesic for this connection.

The exponential families $\mathrm { E F } _ { \nu , { \bf T } }$ famously have two flat connections $\nabla ^ { e }$ and $\nabla ^ { m 1 2 }$ . The parallel transport with respect to the $\nabla ^ { e }$ connection corresponds to the flat vector space connection in the λ space. In other words for two natural parameters $\lambda _ { 1 } , \lambda _ { 2 } \in \Omega _ { \nu , { \bf T } } \subseteq \mathrm { V } ^ { * }$ the parallel transport given by the e-connection allows us to identify the spaces where the natural gradients $\widetilde { \nabla } _ { \lambda } \mathcal { E } ( \lambda _ { 1 } )$ and $\widetilde { \nabla } _ { \lambda } \mathcal { E } ( \lambda _ { 2 } )$ live, which are technically in diferent tangent spaces $T _ { \lambda _ { 1 } } \Omega$ and $T _ { \lambda _ { 2 } } \Omega$ . The concrete coordinates already realize them as vectors in $\mathrm { V } ^ { \ast }$ which coincides with the identification of the tangent spaces using the $\nabla ^ { e }$ parallel transport. The parallel transport given by the m-connection, on the other hand, identifies space of the gradients in the expectation parameters $\mu ,$ which coincides with the identification given by the concrete coordinatization $\mathrm { d } \mathcal { E } \big | _ { \lambda _ { 1 } } , \mathrm { d } \mathcal { E } \big | _ { \lambda _ { 2 } } \in \mathrm { V }$ . These two are related to each other by the Fisher metric $F _ { \lambda }$

Implementing the momentum as the Exponential Moving Average (EMA) of past gradients (parallel transported to the current point’s tangent space with respect to a chosen afine connection) we thus have three notions of such an EMA:

• For the e-connection the EMA is simply accumulating the natural gradients:

$$
\pmb { m } _ { t } = \beta \pmb { m } _ { t - 1 } + ( 1 - \beta ) \widetilde { \nabla } _ { \lambda } \mathcal { E } ( \pmb { \lambda } _ { t } ) .
$$

This accumulation is happening in the natural parameter space $\mathrm { V } ^ { \ast }$

• For the m-connection the parallel transport means that we apply an EMA on the ordinary gradients (not natural gradient), and the momentum in the natural parameter space is obtained using the Fisher information metric at the final (current) point:

$$
\pmb { n } _ { t } = \beta \pmb { n } _ { t - 1 } + ( 1 - \beta ) \mathrm { d } \pmb { \mathcal { E } } \big | _ { \pmb { \lambda } _ { t } } \qquad \pmb { m } _ { t } = F _ { \pmb { \lambda } _ { t } } ^ { - 1 } \pmb { n } _ { t }
$$

• For a harmonic exponential family, the statistical manifold can be parametrized by the Lie group and we can apply the left-invariant connection described above. This corresponds to an EMA of the update vectors $\hat { Y } _ { t } \in \mathfrak { g }$ . For an exponential family, the relation of such an averaging to calculations in the λ coordinates is described in more detail in Sec. 5.4 of the main text.

## Appendix C. Background on Exponential Families

## C.1 Non-constructive Definition of an Exponential Family

Let ${ \mathcal { R } } ( X )$ denote the set of all Radon measures on a manifold X.

Definition 10 (Exponential Family) A non-empty subset ${ \mathcal { P } } \subset { \mathcal { R } } ( X )$ consisting of probability measures is an exponential family on X if there exists a pair $( \nu , \mathbf { T } )$ satisfying

(i) $\nu \in { \mathcal { R } } ( X )$

(ii) $\mathbf { T } \colon X \to \mathrm { V }$ is a continuous map,

(iii) where V is a finite dimensional real vector space,

(iv) and for any probability distribution $p \in \mathcal { P } .$ , there exists $\lambda \in \mathrm { V } ^ { * }$ such that the Radon-Nikodym derivative of p with respect to ν is given by

$$
\frac { \mathrm { d } p } { \mathrm { d } \nu } ( \mathbf { x } ) = \exp ( - \left. \lambda , \mathbf { T } ( \mathbf { x } ) \right. - A ( \pmb { \lambda } ) ) ,\tag{25}
$$

$$
w h e r e ~ A ( \lambda ) : = \log \int _ { X } \exp ( - \left. \lambda , { \bf T } ( { \bf x } ) \right. ) d \nu ( { \bf x } ) \quad ( l o g - p a r t i t i o n ~ f u n c t i o n ) .\tag{26}
$$

Here, $\mathrm { V } ^ { \ast }$ denotes the dual space of V and $\langle \cdot , \cdot \rangle : \mathrm { V } ^ { * } \times \mathrm { V }  \mathbb R$ denotes the natural pairing. We call the pair $( \nu , \mathbf { T } )$ a realization $o f ~ \mathcal { P }$ . Moreover, if the dimension of V is minimal among all the realizations of P, the realization is said to be minimal.

Remark 11 The definition above mainly relies on (Barndorf-Nielsen, 1978, § 8.1). In (Barndorf-Nielsen, 1978, § 8.1) the notion of an exponential family is defined in the category of measurable spaces and measurable maps. On the other hand, we consider exponential families in the category of topological spaces and continuous maps in order to prove Lemma 5. In our setting, we do not need to assume that ν is σ-finite (see the appendix in Tojo and Yoshino (2021)).

Given an exponential family P, and a realization $R = ( \nu , \mathbf { T } )$ of P denote by $\Omega _ { R } ( \mathcal { P } ) \subseteq$ $\Omega _ { R } \subseteq \mathrm { V } ^ { * }$ the set of natural parameters λ that give elements of P. It is clearly possible that more than one λ yields the same distribution if the components $\{ 1 , T _ { 1 } ( \pmb \theta ) , \dots , T _ { d } ( \pmb \theta ) \}$ are not linearly independent functions (up to ν-almost everywhere equivalence). In such a case, clearly R is not a minimal realization of P since we can replace T by $\mathbf { T ^ { \prime } }$ by omitting certain dimensions so that the coordinate functions are linearly independent. Another source of non-minimality can occur if $\Omega _ { R } ( \mathcal { P } )$ lies in an afine space. Note first that the afine hull of these natural parameters of P is then of the form $\lambda _ { b } + \mathrm { W }$ for some subspace $\mathrm { ~ W ~ } \subseteq \mathrm { ~ V ~ } ^ { * }$ It is possible to change the realization $R ^ { \prime } = ( \nu e ^ { - \langle \lambda _ { b } , { \bf T } \rangle } , { \bf T } ^ { \prime } )$ where $\mathbf { T } ^ { \prime }$ is a subset of the linear combination of functions of T. In this case we get a minimal representation of the exponential family.

The log-partition function A is diferentiable in the interior of $\Omega _ { R } .$ and for $\pmb { \lambda } \in \Omega _ { R } ^ { \circ } \subseteq \Omega _ { R }$ in the interior, the negative gradient of the log partition function is the change of coordinates function from the natural parameters to the mean parameters

$$
\begin{array} { r l } { - \nabla A : \Omega ^ { \circ } \longrightarrow \mathrm { V } } & { } \\ { \lambda \longmapsto \mu : = \mathbb { E } _ { q _ { \lambda } } [ \mathbf { T } ] . } \end{array}
$$

Here the mean variable $\pmb { \mu } = \mathbb { E } _ { q _ { \lambda } } [ \mathbf { T } ] \in \mathrm { V }$ is another coordinatization of such distributions with $\lambda \in \Omega _ { R } ^ { \circ }$ . For a minimal realization this map from the interior is one-to-one, it is a change of coordinates.

## C.2 The Gradient vs the Diferential from a Geometric Point of View

There is a meaningful distinction between the gradient and the diferential of a function at a point. Let us take, for concreteness, the example of $\mathcal { E } : \Omega ( \mathcal { P } )  \mathbb { R }$ . The diferential of E at a λ is actually a linear map

$$
\mathrm { d } { \mathcal { E } } { \big \vert } _ { \lambda } : T _ { \lambda } \Omega ( { \mathcal { P } } ) \cong \mathrm { V } ^ { * }  T _ { { \mathcal { E } } ( \lambda ) } \mathbb { R } \cong \mathbb { R } .
$$

In other words $\mathrm { d } \mathscr { E } \bigr | _ { \lambda } \in ( \mathrm { V } ^ { * } ) ^ { * } \cong \mathrm { V }$ . This is not the same space where λ lives, it is its dual. In other words if λ are represented by column vectors, then the diferential $\mathrm { d } \mathcal { E } \vert _ { \lambda }$ is a row vector, representing a linear functional on the λ’s. It is a cotangent vector, and thus cannot be treated as a tangent direction to perform gradient descent on the manifold directly.

A trivial way to convert the diferential to a tangent vector in coordinates is to take the transpose of the row vector, turning it into a column vector. This corresponds to the special case of choosing a Euclidean metric in the following more general construction. A metric is a bilinear form on the tangent space. Again for concreteness consider the Fisher metric at a point $\lambda \in \Omega ( \mathcal { P } )$ . This is a bilinear form $F _ { \lambda } : \mathrm { V } ^ { * } \times \mathrm { V } ^ { * } \to \mathbb R$ which can also be seen as a linear function $F _ { \lambda } : \mathrm { V } ^ { * } \to \mathrm { V }$ upon fixing one of the variables (also called currying in computer science). If the metric is nondegenerate, i.e., if there are no nonzero vectors which are orthogonal to every other vector (including itself), then this map is invertible and $F _ { \lambda } ^ { - 1 }$ gives us a method to convert the diferentials (which are cotangent vectors at λ) to tangent vectors. In fact

$$
\operatorname* { a r g m a x } _ { \mathbf { v } \in T _ { \lambda } \Omega ( P ) \cong \mathrm { V ^ { \ast } } } \mathrm { d } \mathcal { E } \big | _ { \lambda } [ \mathbf { v } ] \qquad \mathrm { s u b j e c t ~ t o } \qquad \| \mathbf { v } \| _ { \mathrm { F i s h e r } } ^ { 2 } = \omega ^ { \mathrm { F i s h e r } } ( \mathbf { v } , \mathbf { v } ) = \mathbf { v } ^ { \top } F _ { \lambda } \mathbf { v } = 1 .
$$

is solved by $\mathbf { v } / \lVert \mathbf { v } \rVert _ { \mathrm { F i s h e r } }$ where $\begin{array} { r } { \mathbf { v } = F _ { \lambda } ^ { - 1 } \mathrm { d } \mathcal { E } \vert _ { \lambda } } \end{array}$

The Fisher metric at $\lambda \in \Omega ( \mathcal { P } )$ is a bilinear form on the tangent space $T _ { \lambda } \Omega ( \mathcal { P } ) \cong \mathrm { V } ^ { * }$ an analogue of the dot product on the tangent vectors. Concretely it is defined as

$$
\begin{array} { r } { \begin{array} { c } { \mathbf { F } _ { \lambda } : \mathrm { V } ^ { * } \times \mathrm { V } ^ { * } \longrightarrow \mathbb { R } } \\ { \mathbf { F } _ { \lambda } ( \mathbf { v } , \mathbf { w } ) = \mathbb { E } _ { \pmb { \theta } \sim q _ { \lambda } } [ ( \partial _ { \mathbf { v } } \log q _ { \lambda } ) ( \partial _ { \mathbf { w } } \log q _ { \lambda } ) ] . } \end{array} } \end{array}
$$

This is similar to what happens in the Lie algebra. With B on g we can rewrite Eq. 8 as finding a $\hat { Y } \in { \mathfrak { g } }$ that satisfies $B ( \hat { Y } , \cdot ) = \mathrm { d } ( \mathcal { E } \circ \varphi \circ L _ { g } ) \big | _ { e }$ . Here both sides are elements of ${ \mathfrak { g } } ^ { * }$ covectors taking in $X \in { \mathfrak { g } }$ and spitting out numbers. Such a $\hat { Y }$ can be found if the linear functional on the right hand side vanishes on the kernel of B. The equivalent presentation involving an optimization constrained to the unit ball also builds the intuition explaining the phrase “the direction of fastest ascent/descent”.