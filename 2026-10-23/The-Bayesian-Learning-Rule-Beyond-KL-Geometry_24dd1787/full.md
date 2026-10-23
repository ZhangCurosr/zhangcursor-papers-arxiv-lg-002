# The Bayesian Learning Rule Beyond KL Geometry

Sophia Sklaviadis<sup>∗,1</sup>, Wu Lin<sup>‡</sup>, André F. T. Martins<sup>∗</sup>, Mário A. T. Figueiredo<sup>∗</sup>, Thomas Möllenhoff<sup>†</sup>, Mohammad Emtiyaz Khan<sup>†,§</sup>

<sup>∗</sup>Instituto de Telecomunicações and Instituto Superior Técnico, Universidade de Lisboa, Portugal <sup>†</sup>RIKEN Center for Advanced Intelligence Project, Japan <sup>‡</sup>University of Central Florida, USA <sup>§</sup>TU Darmstadt and hessian.AI, Germany

<sup>1</sup>ssklaviadis@gmail.com

## Abstract

The Bayesian learning rule (BLR) exploits the relation between Bayesian updating and the Kullback-Leibler divergence (KLD) to derive efficient additive updates for exponential family variational posteriors. The KL geometry is powerful but restrictive: combined with posterior families with parameter-dependent support it can lead to undefined objectives. We ask which non-KL geometries preserve the additive BLR computation. Previous works that use f-divergences suggest that this is only possible with the KLD. Here, we show that this is not true. We show that additive information-geometric optimization algorithms can be derived for the class of mixed-space Bregman or Fenchel-Young discrepancies. These discrepancies induce generalized maximum-entropy families, dual coordinates, and generalized-BLR canonical gradient updates, recovering the classical BLR. The generalized BLR leads to new algorithms that cannot be derived from the KL-based BLR. For the Tsallis negentropy with quadratic statistics, the generalized BLR yields bounded-support and heavy-tailed q-Gaussian posteriors and an easy Newton-like implementation. Experiments up to CIFAR-100 illustrate controllable support and tail behavior and competitive performance.

## 1 Introduction

There is substantial work on generalizing Bayesian inference from the perspective of optimization: the variational posterior strikes a balance between information from the observations and adherence to the prior. Recently, Knoblauch et al. [2022] exploit the “modularity” of this optimization perspective, treating the variational family in which the posterior is sought, the loss connecting the posterior to the observations, and the divergence between the posterior and prior as independent design choices, ignoring formal interactions of these objects. In contrast, Khan and Rue [2023]’s Bayesian learning rule (BLR) is an iterated, conjugate inference extension of classical Bayes, preserving the standard couplings between the prior and entropic regularizer, and between prior and posterior families (Sec. 2). In the BLR, the updates stem from the interaction of Shannon entropy, KL geometry, and exponential-family posteriors, which is computationally powerful but restrictive: it ties the BLR to standard fixed-support exponential families. Posterior families with variable support or adaptive tail behavior fall outside the scope of the KL-based BLR. Here, we focus on a structural question raised by the BLR: are there generalizations of the KL-based variational objective that preserve the additive canonical-coordinate updates that free the BLR from its KL restriction?

![](images/08e17dc8b6f5dc5281c29b2ea75078a3d819252d5851c3d6525d2db1fb7181bf.jpg)  
Figure 1: Left: We generalize the BLR by exploiting the geometry of Fenchel-Young discrepancies. We recover as special cases the BLR and the learning algorithms derived from it, and derive new algorithms using bounded-support and heavy-tailed q-exponential families. Right: Unlike Gaussians, bounded-support q-Gaussians allow control of the support by varying q. Thus, we can fit posteriors that closely track the MAP. The illustrated loss is that of 2D Bayesian logistic regression [Murphy, 2012], each circle indicates the first standard deviation, and outer loss contours indicate higher loss values. The Gaussian (q = 1) fit is farther from the MAP because its full support assigns non-negligible mass to high-loss regions; bounded-support q-Gaussians truncate this exploration.

Our contributions are: (1) We identify mixed-space Bregman or Fenchel-Young discrepancies, as defined by Amari and Cichocki [2010], Martins et al. [2022] but only first connected to variational Bayes in Sklaviadis et al. [2025], as a class that preserves BLR-style additive canonical updates. (2) We propose a generalized BLR update beyond the KL geometry and show that the classical BLR is recovered as a special case (Sec. 3). (3) We instantiate the generalized BLR update for the Tsallis negative entropy, yielding parameter-dependent bounded-support and heavy-tailed q-Gaussian Newton-like variational learning methods (Sec. 4, App. F). Our new Newton-like optimizers require changing a few lines of code in state-of-the-art Improved Variational Online Newton (IVON) [Shen et al., 2024], a scalable Gaussian BLR-style optimizer for deep networks, to obtain heavy-tailed or bounded-support noise injection, maintain non-Gaussian posteriors, and target a generalized Bayesian objective using the maximum entropy (maxent) geometry associated with each objective. These optimizers enable generalized Bayesian learning beyond the KLD at scale (Sec. 5, App.s G, H).

By KL or mixed-space Bregman geometry we mean the convex-analytic structure defined by a reference score and convex function, which induce a mixed Bregman divergence: the score family and convex function determine the maxent posterior family whose convex conjugate determines dual coordinates, and a Hessian metric. Our terminology is motivated by the fact that a mixed Bregman divergence is the slack in a Fenchel-Young inequality.<sup>1</sup> Closest to our generalized BLR is Alquier [2021]’s mirror-descent update that optimizes the sum of an expected loss and a f-divergence. Only when the f-divergence coincides with the KLD is it possible to separate the f-divergence ratio into a difference of linear expectations. Uniquely, the KLD belongs both to the classes of f- and Bregman divergences [Amari, 2009]. The mixed-space Bregman divergence that we use recovers the KLD and BLR, but encompasses other divergences corresponding to generalized entropies. We optimize a generalized variational objective using dual coordinates of a relaxed class of reference scores in place of normalized priors, and generalized Fenchel posteriors in place of Gibbs posteriors. The resulting information-geometric algorithm has the same additive computation as the BLR, and admits a dual-space analogue of Bissiri et al. [2016]’s sequential coherence, unlike f-divergence updates.

Parameter-dependent bounded-support q-Gaussian posteriors allow the posterior mean to closely track MAP solutions (Figs. 1 and 2). By contrast, Gaussian variational means are far enough away from the MAP to avoid sampling parameter-space regions with high-loss. The heavy-tailed q-Gaussian posterior updates we derive are reparametrizations of Student-t distributions studied in robust Bayesian inference. Various aspects of robust Bayes can be reinterpreted in our framework, with the benefit of easy conjugate priors, but drawing these connections is beyond the scope of this paper; as is demonstrating state-of-the-art or scale. We show experiments up to CIFAR-100 (App. G), and defer large experiments, for example on quantization-aware training, pruning, and language model training with bounded variance to future work. We develop the theoretical framework and check that the proposed algorithms for variational training with controllable support and tail behaviors are competitive with their Gaussian IVON counterpart [Khan et al., 2018, Osawa et al., 2019, Shen et al., 2024], without additional tuning.

## 2 KLD Geometry and the BLR

The BLR relies on two levels of a common convex duality. At the density-space level, the Gibbs-Donsker-Varadhan variational formula is the Fenchel duality between the log-Laplace functional of a tilted prior, and the relative entropy with respect to that prior. Equivalently, if the prior is represented by a reference score, the KLD is the Fenchel-Young slack generated by the Shannon negentropy and that score. The density-space duality, restricted by finite moment constraints, results in the maximum entropy characterization of exponential families: with constant base measure, the exponential family $\pi _ { \lambda }$ with canonical parameter λ maximizes the Shannon entropy among all densities with the same moments, and with a non-constant base measure it maximizes the relative entropy with respect to that base measure. At the parametric level, restricting the score to the finite-dimensional linear class $\eta _ { \bf \lambda } ( \pmb { \theta } ) = \langle \lambda , { \bf T } ( \pmb { \theta } ) \rangle$ ⟩ induces the cumulant $A ( \lambda )$ . Under regularity and minimality, $\pmb { \mu } = \nabla A ( \pmb { \lambda } )$ , and $( \lambda , \mu )$ are the dual coordinates on which the additive BLR update is based.

Fix a measurable space $( \Theta , A )$ and a σ-finite reference measure h. Let $\mathcal { P }$ be a convex class of probability densities with respect to $h ,$ and write $\begin{array} { r } { \mathbb { E } _ { \pi } [ f ] = \int _ { \Theta } f ( \pmb { \theta } ) \pi ( \pmb { \theta } ) d h ( \pmb { \theta } ) } \end{array}$ . For the Shannon negentropy $\begin{array} { r } { \Omega _ { \mathrm { S h } } = - \mathcal { H } ( \pi ) = \dot { \int } _ { \Theta } \pi ( \pmb { \theta } ) \log \pi ( \pmb { \theta } ) d h ( \pmb { \theta } ) } \end{array}$ , the Fenchel conjugate over densities

$$
\Omega _ { \mathrm { S h } } ^ { * } ( \eta ) = \operatorname* { s u p } _ { \pi \in \mathcal { P } } \left\{ \mathbb { E } _ { \pi } [ \eta ( \pmb { \theta } ) ] - \Omega _ { \mathrm { S h } } ( \pi ) \right\} = \log \int _ { \Theta } e ^ { \eta ( \pmb { \theta } ) } d h ( \pmb { \theta } ) ,
$$

is log-normalization, whenever the integral is finite. The corresponding Fenchel-Young slack is

$$
\begin{array} { r } { \mathbb { D } _ { - \mathcal { H } } ( \eta , \pi ) = \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta ) + \Omega _ { \mathrm { S h } } ( \pi ) - \mathbb { E } _ { \pi } [ \eta ( \pmb { \theta } ) ] \geq 0 . } \end{array}
$$

The KLD can be derived as the Fenchel-Young slack generated by Shannon negentropy and a reference score. In particular, if a prior density is represented by score $\eta _ { 0 } .$ , as $\pi _ { 0 } ( \pmb { \theta } ) = \exp \left( \eta _ { 0 } ( \pmb { \theta } ) - \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } ) \right)$ , then log $\bar { \pi } _ { 0 } ( \pmb { \theta } ) = \eta _ { 0 } ( \pmb { \theta } ) - \Omega _ { \mathrm { S h } } ^ { * } ( \eta _ { 0 } )$ . Hence, for any density π,

$$
\begin{array} { l } { \displaystyle \mathbb { D } _ { - \mathcal { H } } ( \eta _ { 0 } , \pi ) = \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } ) + \int _ { \Theta } \pi ( \pmb { \theta } ) \log \pi ( \pmb { \theta } ) d h ( \pmb { \theta } ) - \mathbb { E } _ { \pi } [ \eta _ { 0 } ( \pmb { \theta } ) ] } \\ { \displaystyle \quad \quad = \int _ { \Theta } \pi ( \pmb { \theta } ) \left[ \log \pi ( \pmb { \theta } ) - \eta _ { 0 } ( \pmb { \theta } ) + \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } ) \right] d h ( \pmb { \theta } ) = \mathbb { D } _ { \mathrm { K L } } ( \pi \parallel \pi _ { 0 } ) . } \end{array}\tag{1}
$$

Additive constants in $\eta _ { 0 }$ do not matter, since they change $\Omega _ { \mathrm { S h } } ^ { * } ( \eta _ { 0 } )$ by the same constant.

The Gibbs or Donsker-Varadhan variational formula follows by applying the same slack to the tilted score $\eta _ { 0 } - \ell$ . The normalizer of the tilted prior is

$$
\mathcal { Z } = \int _ { \Theta } e ^ { - \ell ( \pmb { \theta } ) } \pi _ { 0 } ( \pmb { \theta } ) d h ( \pmb { \theta } ) = \exp \left( \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } - \ell ) - \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } ) \right) .
$$

Therefore, for every $\pi \in { \mathcal { P } }$ with $\pi \ll \pi _ { 0 }$

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol \pi } [ \ell ( \pmb \theta ) ] + \mathbb { D } _ { \mathsf { K L } } ( \boldsymbol \pi \| \boldsymbol \pi _ { 0 } ) = \mathbb { E } _ { \boldsymbol \pi } [ \ell ( \pmb \theta ) ] + \Omega _ { \mathrm { S h } } ( \boldsymbol \pi ) - \mathbb { E } _ { \boldsymbol \pi } [ \eta _ { 0 } ( \pmb \theta ) ] + \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } ) } \\ & { \qquad = - \log \mathcal { Z } + \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } - \ell ) + \Omega _ { \mathrm { S h } } ( \boldsymbol \pi ) - \mathbb { E } _ { \boldsymbol \pi } [ \eta _ { 0 } ( \pmb \theta ) - \ell ( \pmb \theta ) ] } \\ & { \qquad = - \log \mathcal { Z } + \mathbb { D } _ { - \mathcal H } ( \eta _ { 0 } - \ell , \boldsymbol \pi ) \ge - \log \mathcal { Z } . } \end{array}\tag{2}
$$

Taking the infimum over candidates π we get the Gibbs-Donsker-Varadhan variational formula

$$
- \log \mathcal { Z } = \operatorname* { i n f } _ { \pi \in \mathcal { P } : \pi \ll \pi _ { 0 } } \mathbb { E } _ { \pi } [ \ell ( \pmb { \theta } ) ] + \mathbb { D } _ { \mathrm { K L } } ( \pi \mid | \pi _ { 0 } ) .\tag{3}
$$

Conversely, sharpness of the Gibbs bound uniquely characterizes the KL regularizer (App. A). Equality in Eq. 3 holds when the Fenchel-Young slack in Eq. 2 vanishes, that is, when $\pi$ is the normalized density induced by the tilted score $\eta _ { 0 } - \ell ,$ , namely the Gibbs posterior:

$$
\pi _ { * } ( \pmb { \theta } ) = \exp \left( \eta _ { 0 } ( \pmb { \theta } ) - \ell ( \pmb { \theta } ) - \Omega _ { \mathrm { S h } } ^ { \ast } ( \eta _ { 0 } - \ell ) \right) = \frac { e ^ { - \ell ( \pmb { \theta } ) } \pi _ { 0 } ( \pmb { \theta } ) } { \mathcal { Z } } .
$$

This duality appears in several literatures. In statistical mechanics and information theory, freeenergy minimization and maximum Shannon entropy subject to moment constraints yield exponential families [Gibbs, 1902, Jaynes, 1957, Williams, 1980, Zellner, 1988]. In generalized Bayesian inference, the negative log-likelihood is replaced by a general loss while retaining the KLD penalty to the prior [Zhu et al., 2014, Bissiri et al., 2016, Ghosh and Basu, 2016, Jewson et al., 2018, Miller and Dunson, 2019]. In PAC-Bayesian learning theory, the KLD measures the complexity of moving from a fixed prior to a data-dependent posterior [Zhang, 1999, Catoni, 2007, Grünwald and Mehta, 2020]. In variational inference, the ELBO is equivalently a KLD-regularized expected-fit objective or a KL projection of the exact posterior onto a tractable family [Wainwright et al., 2008, Hoffman et al., 2013, Blei et al., 2017]. Instead of replacing the KLD in Eq. 3, we use the Fenchel-Young structure that generates the KLD as the template for the mixed-space Bregman construction in Sec. 3.<sup>2</sup>

## 2.1 Exponential Family Posteriors from Moment-Constrained Maxent

When the posterior belongs to a regular, minimal exponential family (expfam), through a constant base measure determined by finite moment constraints, the duality between log-normalization and negative entropy characterizes the posterior as the unique Shannon maxent distribution. To see this, first we fix a base measure $b ( \pmb \theta ) d \pmb \theta$ and a finite sufficient statistic

$$
\mathbf { T } ( \pmb { \theta } ) = \left( T _ { 1 } ( \pmb { \theta } ) , . . . , T _ { P } ( \pmb { \theta } ) \right) .\tag{4}
$$

For each $\boldsymbol { \lambda } \in \mathbb { R } ^ { P } ,$ the linear score $\eta _ { \Lambda } ( \pmb { \theta } ) = \langle \lambda , \mathbf { T } ( \pmb { \theta } ) \rangle$ defines an exponential tilt $e ^ { \eta _ { \mathsf { A } } ( \theta ) } b ( \theta ) d \theta$ which is not, in general, a probability measure. For $\lambda \in \Lambda = \{ \lambda \in \mathbb { R } ^ { P } : \int b ( \pmb { \theta } )$ exp $\left( \eta _ { \lambda } ( \pmb { \theta } ) \right) d \pmb { \theta } < \infty \}$ , the unique additive correction is the cumulant or log-partition function $\begin{array} { r } { A ( \boldsymbol { \lambda } ) = \log \int \dot { b } ( \pmb { \theta } ) \exp \left( \langle \boldsymbol { \bar { \lambda } } , \mathbf { T } ( \pmb { \theta } ) \rangle \right) d \pmb { \theta } } \end{array}$ , defined on Λ, and the density induced by h and T has the familiar exponential family form

$$
\pi _ { \lambda } ( \pmb { \theta } ) = b ( \pmb { \theta } ) \exp \left( \langle \pmb { \lambda } , \mathbf { T } ( \pmb { \theta } ) \rangle - A ( \pmb { \lambda } ) \right) .\tag{5}
$$

Note that λ is introduced as the coefficient of the finite-dimensional score, and $A ( \lambda )$ is derived as the log-normalizer of the corresponding exponential tilt, which is the measure-theoretic operation that uses the score $\eta _ { \lambda }$ to tilt the base measure, thereby characterizing the expfam form.<sup>3</sup>

Differentiating $\begin{array} { r } { \int \pi _ { \lambda } ( \pmb { \theta } ) d \pmb { \theta } = 1 } \end{array}$ with respect to λ we obtain the moment map $\nabla _ { \lambda } A ( \lambda ) = \operatorname { \mathbb { E } } _ { \pi _ { \lambda } } [ \overset {  } { \mathbf { T } } ( \pmb { \theta } ) ] = \pmb { \mu } ( \lambda )$ Thus, natural and expectation parameter form a Legendre dual coordinate system. For simplicity, we assume the base measure is constant. In this case, the member of the expfam with moment vector $\pmb { \mu } ( \lambda )$ is the unique Shannon entropy maximizer among all densities $\pi \in { \mathcal { P } }$ , that satisfy the same moment constraints (see App. B.1 for a proof).<sup>4</sup>

## 2.2 The BLR in Exponential Family Coordinates

Khan and Rue [2023] apply Eq. 3 to unify various learning algorithms under the Bayesian Learning Rule (BLR). The BLR is obtained by restricting the DV optimization in Eq. 3 to the set of regular, minimal exponential families $\mathcal { Q } \subseteq \mathcal { P }$ , and exploiting the $( \lambda , \mu )$ dual pair. Assuming, in addition, that the prior belongs to the same family, $\pi _ { 0 } ( \pmb { \theta } ) = b ( \pmb { \theta } )$ exp $( \langle \lambda _ { 0 } , \mathbf { T } ( \pmb \theta ) \rangle - A ( \lambda _ { 0 } ) )$ , and taking b to be constant for simplicity, $\mathbb { D } _ { - \mathcal { H } } ( \log \pi _ { 0 } , \pi _ { \lambda } ) = A ( \dot { \lambda } _ { 0 } ) + \dot { A } ^ { * } ( \pmb { \mu } ( \pmb { \lambda } ) ) - \langle \lambda _ { 0 } , \pmb { \mu } \rangle$ , where $A ^ { * }$ is the Legendre dual of A, and coincides with the negative Shannon entropy $A ^ { * } ( \dot { \pmb { \mu } } ( \lambda ) ) ^ { * } = - \mathcal { H } ( \pi _ { \lambda } )$ . Differentiating $\mathbb { D } _ { - \mathcal { H } } ( \log \pi _ { 0 } , \pi _ { \lambda } )$ with respect to the expectation parameter, $\nabla _ { \mu } \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \lambda } \| \pi _ { 0 } ) = \lambda - \lambda _ { 0 }$

For exponential families, the Fisher information matrix $\mathbf { F } ( \lambda )$ can be shown to equal $\nabla _ { \lambda } ^ { 2 } A ( \lambda )$ , which allows us to use the chain rule to express natural gradients $\widetilde { \nabla }$ as reparametrized regular gradients, $\widetilde { \nabla } \lambda = \mathbf { F } ( \lambda ) ^ { - 1 } \nabla \lambda = \nabla _ { \mu }$ . The expfam restriction renders the natural gradient of the KLD affine in λ, and the natural-gradient step into a mirror-descent step with an additive λ update

$$
\lambda  \lambda - \rho \nabla _ { \mu } ( \mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb \theta ) ] + \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \lambda } \| \pi _ { 0 } ) ) = ( 1 - \rho ) \lambda - \rho ( \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb \theta ) ] - \lambda _ { 0 } ) ,\tag{6}
$$

where $\rho$ is a step size or learning rate.

Under conjugacy, we can simplify further. There exists a natural parameter $\lambda ^ { \prime }$ such that

(7)

$$
\begin{array} { l } { \displaystyle \pi _ { \lambda ^ { \prime } } ( \pmb { \theta } ) = \frac { 1 } { \mathcal { Z } } \exp ( - \ell ( \pmb { \theta } ) ) \pi _ { 0 } ( \pmb { \theta } ) \Rightarrow \ell ( \pmb { \theta } ) = \log \pi _ { 0 } ( \pmb { \theta } ) - \log \pi _ { \lambda ^ { \prime } } ( \pmb { \theta } ) - \log \mathcal { Z } } \\ { \displaystyle \qquad = \langle \lambda _ { 0 } - \lambda ^ { \prime } , \mathbf { T } ( \pmb { \theta } ) \rangle + A ( \lambda ^ { \prime } ) - A ( \lambda _ { 0 } ) - \log \mathcal { Z } } \end{array}\tag{8}
$$

Thus, ${ \mathbb E } _ { \pi _ { \lambda } } \ell ( \pmb \theta ) = \langle \lambda _ { 0 } - \lambda ^ { \prime } , \pmb \mu ( \lambda ) \rangle + \mathrm { c o n s t . }$ , and $\nabla _ { \pmb { \mu } } \mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb { \theta } ) ] = \lambda _ { 0 } - \lambda ^ { \prime }$ , so Eq. 6 simplifies to <sup>5</sup>

$$
\begin{array} { r } { \lambda  ( 1 - \rho ) \lambda + \rho \lambda ^ { \prime } . } \end{array}
$$

For conjugate expfam models, a single BLR step coincides with classical Bayes’ rule. More generally, outside conjugacy, the BLR optimizes the Bayesian objective whose minimizer over $\mathcal { P }$ is the exact posterior. Many algorithms from optimization, statistics, and deep learning arise as special cases by varying the form of the exponential family. New algorithms have also been inspired by the general form of the BLR, including the Variational Online Newton algorithm that we extend in Sec. 4 and App. F to exemplify our proposed generalized BLR.

## 2.3 Limitations of the KLD and the BLR

KLD geometry has certain limitations. The KLD requires absolute continuity: if $\pi _ { \lambda }$ places mass outside the support of $\pi _ { 0 } ,$ then $\mathbb { D } _ { \mathrm { K L } } ( \pi _ { \lambda } \| \pi _ { 0 } ) = + \infty$ . Thus posterior families with bounded, parameterdependent support, and more generally families that cannot be represented as ordinary exponential families with fixed support, fall outside the scope of the KL-based BLR. Figs. 1 and 2 illustrate why bounded-support families are useful: they can avoid sampling high-loss regions and allow the variational mean to approach MAP solutions more closely than Gaussian approximations.

How then can we generalize the KLD so as to preserve the BLR’s efficient, additive updating? In Sec. 3 we show that Fenchel-Young discrepancies mediate exactly such a generalization. They yield generalized maxent families with well-defined dual parameters and generalized canonical gradients, allowing BLR-style updates under bounded, parameter-dependent support and beyond. Within the class of updates obtained by minimizing the sum of expected cumulative loss and an f-divergence, sequential coherence as defined by Bissiri et al. [2016] singles out the Kullback-Leibler divergence. In that sense, an f-divergence generalization of the KLD, like the $f .$ -divergence-regularized updates of Alquier [2021], cannot be used to define Bayes-like additive updates. Our mixed-space Bregman generalization of the KLD avoids this problem. The generalized BLR respects a dual-space analogue of sequential coherence: losses additively tilt the reference score, so processing losses sequentially or jointly yields the same tilted score. An interesting question that we hope to address in the future is whether mixed-space Bregman divergences uniquely recover BLR-style updates when we also wish to preserve the linear expectation of the loss term in Eq. 3.

## 3 Fenchel-Young BLR

To derive a generalized BLR beyond KL geometry, first we introduce mixed-space Bregman divergences to which we refer interchangeably as Fenchel-Young discrepancies (FYD). We call $\mathbb { D } _ { \Omega } ( \eta , \pi )$ a mixed-space Bregman because its two arguments belong to different spaces: η is a dual score, while π is a density. This generalization of the KLD forms the basis of our proposal and allows our construction to remain meaningful for generalized entropies that may be nonsmooth, extended-valued, or have parameter-dependent support. We show how FYDs lead to generalized maxent families, for which we can define dual coordinates. Finally, we derive a generalized variational objective and learning rule. FYDs are also studied as discriminative losses by Blondel et al. [2019], Martins et al. [2022], and their information-geometric properties are discussed in Amari and Cichocki [2010], Nielsen [2022]. Here, we use them in a novel way to generalize the Bayesian learning rule optimizing the generalized objective defined in Sklaviadis et al. [2025].

## 3.1 Generalizing the KLD to the Fenchel-Young Discrepancy

Fix a proper convex function $\Omega : \mathcal { P }  \mathbb { R } \cup \{ \infty \}$ on densities. The Fenchel conjugate of Ω

$$
\Omega ^ { * } ( \eta ) = \operatorname* { s u p } _ { \pi \in \mathcal { P } } \mathbb { E } _ { \pi } [ \eta ( \pmb { \theta } ) ] - \Omega ( \pi )
$$

acts on measurable scores η : Θ → R ∪ {±∞}, for which the pairing $\begin{array} { r } { \langle \eta , \pi \rangle = \mathbb { E } _ { \pi } [ \eta ( \pmb { \theta } ) ] = \int _ { \Theta } \eta ( \pmb { \theta } ) \pi ( \pmb { \theta } ) d h ( \pmb { \theta } ) } \end{array}$ is well defined. For any $\eta \in$ dom $\Omega ^ { * }$ and $\pi \in$ dom Ω with finite pairing, the Fenchel-Young discrepancy is [Martins et al., 2022, Amari and Cichocki, 2010]

$$
\mathbb { D } _ { \Omega } ( \eta , \pi ) = \Omega ^ { * } ( \eta ) + \Omega ( \pi ) - \mathbb { E } _ { \pi } [ \eta ( \pmb { \theta } ) ] .\tag{10}
$$

The first argument is a reference score, not necessarily the log-density of a normalized prior. Since every $\pi \in { \mathcal { P } }$ integrates to one, $\begin{array} { r } { \Omega ^ { * } ( \eta + c ) = \operatorname* { s u p } _ { \pi \in \mathcal { P } } \big \{ \dot { \mathbb { E } } _ { \pi } [ \eta ( \pmb { \theta } ) ] + c - \dot { \Omega } ( \pi ) \big \} = \Omega ^ { * } ( \eta ) + \dot { c } } \end{array}$ , and therefore $\mathbb { D } _ { \Omega } ( \eta + c , \bar { \pi } ) = \mathbb { D } _ { \Omega } ( \eta , \pi )$ . For general Ω, when the supremum defining $\Omega ^ { * } ( \eta )$ is attained, any maximizer $\pi _ { \eta } \in$ arg $\begin{array} { r } { \operatorname* { m a x } _ { \pi \in \mathcal { P } } \left\{ \mathbb { E } _ { \pi } [ \eta ( \pmb { \theta } ) ] - \Omega ( \pi ) \right\} } \end{array}$ } is a density induced by the score η. Equivalently, $\pi _ { \eta } \in \partial \Omega ^ { * } ( \eta )$ with respect to the $( \eta , \pi )$ pairing.

## 3.2 Generalizing the Exponential Family

Dual coordinates related to the FYD can be defined for a much larger class of distributions that the expfam, including deformed exponential families [Amari, 2016, Naudts, 2002]. Our generalization preserves the same logic we followed in Sec. 2.1 to derive exponential families and characterize their dual coordinates. We begin with a finite-dimensional $\eta _ { \lambda } .$ , thereby identify a convex potential $\Psi ^ { * } ( \lambda ) = \Omega ^ { * } ( \eta _ { \lambda } )$ , and only when the supremum defining $\Omega ^ { * } ( \eta _ { \lambda } )$ is attained by a density, do we recover the corresponding member $\pi _ { \lambda }$ of the induced maximum-entropy family. The generalized Fisher we use is distinct from both the invariant and conformally flat geometries of Amari et al. [2012], and it is a generalization of the metric associated with the derivation of deformed exponential families subject to linear (rather than escort) expectation constraints discussed in Naudts [2011].

Starting with a fixed measurable statistic $\mathbf { T } ( \pmb \theta )$ as in $\mathrm { E q . 4 }$ , define the moment vector $\pmb { \mu } = \mathbb { E } _ { \pi } [ \mathbf { T } ( \pmb { \theta } ) ]$ and associate with it the Lagrange multiplier $\mathbf { \boldsymbol { \lambda } } \mathbf { \boldsymbol { \lambda } } \in \mathbb { R } ^ { P }$ . The corresponding finite-dimensional score family is $\eta _ { \bf \lambda } ( \pmb { \theta } ) = \langle \lambda , { \bf T } ( \pmb { \theta } ) \rangle$ , defined up to an additive constant.<sup>6</sup> We define $\Psi$ to assign to each moment vector $\pmb { \mu }$ the optimal value of the corresponding constrained minimization problem

$$
\Psi ( \pmb \mu ) = \operatorname* { i n f } _ { \pi \in \mathcal { P } } \Omega ( \pi ) \mathrm { ~ s . t . ~ } \frac { \mathbb { E } } { \pi } [ \mathbf { T } ( \pmb \theta ) ] = \pmb \mu , \mathrm { ~ w i t h ~ d o m a i n ~ } \mathcal { M } = \left\{ \frac { \mathbb { E } } { \pi } [ \mathbf { T } ( \pmb \theta ) ] : \pi \in \mathcal { P } , \Omega ( \pi ) < \infty \right\} \mathrm { . ~ }\tag{11}
$$

We do not assume that the infimum in Eq. 11 is attained for every $\mu .$ When $\Omega = - { \mathcal { H } } .$ , Eq. 11 reduces to the classical Shannon maxent, and under the constant base measure convention of Sec. $3 . 1$ whenever the minimizer exists it is the expfam in Eq. 5 with natural parameter $\pmb { \lambda } = \nabla A ^ { * } ( \pmb { \mu } )$ . In that case, $\Psi ( \mu ) = A ^ { * } ( \mu )$ and $\Psi ^ { * } ( \lambda ) = A ( \lambda )$

Proposition 1. For every $\lambda \in \mathbb { R } ^ { P } , \Psi ^ { * } ( \lambda ) = \Omega ^ { * } ( \eta _ { \lambda } )$

Proof. By definition of the convex conjugate,

$$
\begin{array} { r l } & { \Psi ^ { * } ( \lambda ) = \underset { \mu \in \mathcal { M } } { \operatorname* { s u p } } \left. \left. \lambda , \mu \right. - \Psi ( \mu ) \right. = \underset { \mu \in \mathcal { M } } { \operatorname* { s u p } } ~ \underset { \pi \in \mathcal { P } : \mathbb { E } _ { \pi } \left[ \mathbf { T } \right] = \mu } { \operatorname* { s u p } } \left. \left. \lambda , \mu \right. - \Omega ( \pi ) \right. } \\ & { \quad \quad = \underset { \pi \in \mathcal { P } } { \operatorname* { s u p } } \left. \left. \lambda , \frac { \mathbb { E } \left[ \mathbf { T } ( \pmb \theta ) \right] } { \pi } \right. - \Omega ( \pi ) \right. = \underset { \pi \in \mathcal { P } } { \operatorname* { s u p } } \left. \frac { \mathbb { E } \left[ \eta _ { \lambda } ( \pmb \theta ) \right] - \Omega ( \pi ) } { \pi } \right. = \Omega ^ { * } ( \eta _ { \lambda } ) . ~ \bigtriangledown } \end{array}
$$

Prop. 1 identifies the finite-dimensional convex potential associated with the score family. However, it does not yet identify a family of densities, it only shows how the infinite dimensional moment problem projects to canonical coordinates. Next, we determine when a density obtained from a finite dimensional score solves the infinite dimensional problem in Eq. 11.

Proposition 2. Assume that for $\pmb { \lambda } \in \mathbb { R } ^ { P }$ the supremum defining $\Omega ^ { * } ( \eta _ { \lambda } )$ is attained by $\pi _ { \lambda } \in { \mathcal { P } }$ $\Omega ^ { * } ( \bar { \eta } _ { \lambda } ) = \mathbb { E } _ { \pi _ { \lambda } } [ \eta _ { \lambda } ( \pmb \theta ) ] - \Omega ( \bar { \pi _ { \lambda } } )$ . Define $\pmb { \mu } ( \lambda ) = \mathrm { \bar { E } } _ { \pi _ { \lambda } } [ \mathbf { T } ( \pmb { \theta } ) ]$ ]. Then $\pi _ { \lambda }$ minimizes $E q .$ 11.

Proof. By the optimality of $\pi _ { \lambda } \quad \mathrm { i n \quad }$ the definition of $\Omega ^ { * } ( \eta _ { \lambda } )$ , for every $\pi \in { \mathcal { P } }$ $\Omega ( \pi ) - \bar { \mathbb { E } } _ { \pi } [ \eta _ { \lambda } ( \pmb { \theta } ) ] \dot { \geq } \Omega ( \pi _ { \lambda } ) - \mathbb { E } _ { \pi _ { \lambda } } [ \eta _ { \lambda } ( \pmb { \theta } ) ]$ . Let π satisfy the same moment constraints as $\pi _ { \lambda }$ , namely $\mathbf { \mathbb { E } } _ { \pi } [ \mathbf { T } ( \pmb { \theta } ) ] = \pmb { \mu } ( \lambda )$ . Then

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol \pi } \big [ \eta _ { \lambda } ( \pmb \theta ) \big ] = \langle \lambda , \mathbb { E } _ { \boldsymbol \pi } \big [ \mathbf { T } ( \pmb \theta ) \big ] \rangle = \langle \lambda , \mathbb { E } _ { \boldsymbol \pi _ { \lambda } } \big [ \mathbf { T } ( \pmb \theta ) \big ] \rangle = \mathbb { E } _ { \boldsymbol \pi _ { \lambda } } \big [ \eta _ { \lambda } ( \pmb \theta ) \big ] . } \end{array}\tag{12}
$$

Hence $\Omega ( \pi ) \geq \Omega ( \pi _ { \lambda } )$ , which proves optimality. If Ω is strictly convex, this minimizer is unique.

In Prop. 2 the infinite dimensional optimizer is certified by a finite dimensional score lying in the span of the constraint functions. This motivates the induced family

$$
\mathcal { Q } _ { \Omega , { \bf T } } = \left\{ \pi _ { \lambda } \in \mathcal { P } : \pi _ { \lambda } \mathrm { a t t a i n s } \Omega ^ { * } ( \eta _ { \lambda } ) , \lambda \in \mathbb { R } ^ { P } \right\} .
$$

Props. 1 and $2$ are the only occasions where the full density-space duality is needed. Once the induced family $Q _ { \Omega , \mathbf { T } }$ and the potential $\Psi ^ { * }$ are identified, all subsequent geometry and updates are finite-dimensional in the dual coordinates $( \lambda , \mu )$ . From here on, $\pi _ { \lambda }$ denotes a member of $\mathcal { Q } _ { \Omega , \mathbf { T } }$ Whenever the maximizer is unique and the conjugate is differentiable at $\eta _ { \lambda }$ , we write $\pi _ { \lambda } = \nabla \Omega ^ { \ast } ( \eta _ { \lambda } )$ When $\Psi ^ { * }$ is differentiable at λ and the corresponding optimizer $\pi _ { \lambda }$ has finite moments,

$$
\pmb { \mu } ( \lambda ) = \nabla _ { \pmb { \lambda } } \Psi ^ { * } ( \pmb { \lambda } ) = \mathbb { E } _ { \pi _ { \lambda } } [ \mathbf { T } ( \pmb { \theta } ) ] .
$$

We further write $\lambda ( \mu ) = \nabla _ { \mu } \Psi ( \mu )$ . Thus $\boldsymbol { \lambda }$ and $\pmb { \mu }$ play the roles of canonical and expectation parameters, as in exponential families, but for the larger class of maxent families induced by Ω.

Finally, whenever $\pi _ { \lambda } \in \mathcal { Q } _ { \Omega , \mathbf { T } }$ solves Eq. 11 at $\mu ,$ , the FYD admits the mixed-space form

$$
\mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) = \Psi ^ { * } ( \lambda _ { 0 } ) + \Psi ( \pmb { \mu } ) - \langle \lambda _ { 0 } , \pmb { \mu } \rangle .
$$

Indeed, Prop. 1 gives $\Omega ^ { * } ( \eta _ { \lambda _ { 0 } } ) = \Psi ^ { * } ( \lambda _ { 0 } )$ , optimality of $\pi _ { \lambda }$ in Eq. 11 gives $\Omega ( { \pi } _ { \lambda } ) = \Psi ( { \mu } )$ , and $\mathbb { E } _ { \pi _ { \lambda } } [ \eta _ { \lambda _ { 0 } } ( \pmb { \theta } ) ] = \langle \bar { \lambda } _ { 0 } , \pmb { \mu } \rangle$ . This is directly analogous to the mixed-space Bregman form of the KLD in Eq. 1 and enables BLR-style additive updates in canonical coordinates. Importantly, the first argument is now a reference score $\eta _ { \mathsf { A } _ { 0 } }$ , not necessarily a normalized density. When $\Omega = - \mathcal { H }$ we recover the classical exponential family. When $\Omega = \Omega _ { 2 - q } ,$ the induced family is the deformed q-exponential (proven in App. B.2). In App. F.1, we show the quadratic statistic $\mathbf { T } ( \pmb \theta ) ^ { \top } = ( \pmb \theta , \pmb \theta \pmb \theta ^ { \top } )$ yields the bounded-support and heavy-tailed $q \mathrm { . }$ -Gaussian families used by the q-VON Algorithms 1 and 2.

## 3.3 Generalizing the Variational Objective

Using FYDs and the generalized maxent families, the generalized BLR objective

$$
\operatorname* { m i n } _ { \lambda \in \Lambda } \mathbb { E } _ { \theta \sim \pi _ { \lambda } } \left[ \ell ( \theta ) \right] + \mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) , \Lambda = \left\{ \lambda \in \mathbb { R } ^ { P } : \operatorname * { a r g m a x } _ { \pi \in \mathcal { P } } \left( \mathbb { E } _ { \pi } \left[ \eta _ { \lambda } ( \theta ) \right] - \Omega ( \pi ) \right) \neq \mathcal { O } \right\} ,\tag{13}
$$

recovers the BLR objective when $\Omega = - \mathcal { H }$ and $\eta _ { \lambda _ { 0 } } = \log \pi _ { 0 }$ . Other choices of Ω amount to new objectives. From the definition of FYDs we can write Eq. 13 as

$$
\operatorname* { m i n } _ { \lambda \in \Lambda } \mathbb { E } _ { \pmb { \theta } \sim \pi _ { \lambda } } \left[ \ell ( \pmb { \theta } ) - \eta _ { \lambda _ { 0 } } ( \pmb { \theta } ) \right] + \Omega ( \pi _ { \lambda } ) .\tag{14}
$$

This separation in terms of a linear expectation of $\eta _ { \lambda _ { 0 } }$ and a generalized negentropy of $\pi _ { \lambda }$ is not possible with f-divergences (except the KLD). Up to a π -independent scalar Eq. 13 is the FYD $\mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } - \ell , \pi _ { \lambda } )$ . Thus, observing a loss ℓ tilts the reference score from $\eta _ { \lambda _ { 0 } } \mathrm { t o } \eta _ { \lambda _ { 0 } } - \ell$ . Since losses compose associatively in dual space, processing losses sequentially or jointly results in the same score, preserving sequential coherence in score-space.

## 3.4 Generalizing the BLR

Theorem 1 (Fenchel-Young BLR). Let $\mathcal { Q } _ { \Omega , \mathbf { T } } , \Psi , \ \Psi ^ { * }$ , and the moment map $\pmb { \mu } ( \lambda ) = \mathbb { E } _ { \pi _ { \lambda } } [ \mathbf { T } ( \pmb { \theta } ) ]$ be as in Sec. 3.2 and Props. 1 and 2. Let $\Lambda _ { 0 }$ be an open coordinate chart of the induced family $\mathcal { Q } _ { \Omega , \mathbf { T } }$ . Assume that $\Psi ^ { * } \in \bar { C } ^ { 2 } ( \Lambda _ { 0 } )$ , and define a generalized Fisher<sup>7</sup> $\mathbf { F } ( \boldsymbol { \lambda } ) : = \mathbf { \bar { \nabla } } \nabla _ { \boldsymbol { \lambda } } ^ { 2 } \Psi ^ { * } ( \boldsymbol { \lambda } )$ invertible on $\Lambda _ { 0 } .$ . Assume also that the dual potential Ψ is differentiable on $\mu ( \Lambda _ { 0 } )$ , with $\nabla _ { \mu } \tilde { \Psi } ( \mu ( \lambda ) ) = \lambda$ . Fix $\lambda _ { 0 } \in \Lambda _ { 0 } ,$ and define $\bar { \ell } ( \pmb { \theta } ) = \ell ( \pmb { \theta } ) - \eta _ { \lambda _ { 0 } } ( \pmb { \theta } )$ . Assume that the map $\pmb { \mu } \mapsto \mathbb { E } _ { \pi _ { \lambda ( \pmb { \mu } ) } } [ \bar { \ell } ( \pmb { \theta } ) ]$ is differentiable on $\mu ( \Lambda _ { 0 } )$ . Then the generalized canonical gradient satisfies

```tcl
Algorithm 1 $q { \mathrm { - V O N } }$ (bounded-support $q < 1 )$ Algorithm 2 Student-t-VON (heavytailed $q > 1 )$
Require: Learning rates $\rho _ { t } ,$ , weight-decay $\overline { { \delta > 0 } }$ Require: Learning rates $\rho _ { t } ,$ , weight-decay $\overline { { \delta > 0 } }$
Require: Momentum $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ Require: Momentum $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$
Require: Hessian init $h _ { 0 } > 0 ,$ Require: Hessian init $h _ { 0 } > 0 ,$
Deformation index $q < 1$ Degrees of freedom $\nu > 2$
Compute: $\begin{array} { r } { \kappa ( q , D ) \gets \frac { R ^ { 2 } } { D + 2 ( 2 - q ) / ( 1 - q ) } } \end{array}$ Compute:
$\gamma  D + 2 / ( 1 - q )$ $\begin{array} { r } { \kappa _ { t }  \frac { \nu + D } { \nu } [ ( \nu + D ) ^ { \frac { D } { 2 } } \pi ^ { \frac { D } { 2 } } \frac { \Gamma ( \frac { \nu } { 2 } ) } { \Gamma ( \frac { \nu + D } { 2 } ) } ] ^ { \frac { 2 } { \nu } } } \end{array}$
Init: m ← (NN-weights), $\mathbf { h }  h _ { 0 } , \mathbf { g }  0 .$ Init: m $ \mathrm { ( N N \mathrm { - w e i g h t s ) } } .$ , h $ h _ { 0 } , \bar { \bf g }  0 ,$
$\zeta \gets N$ $\zeta \gets N$
Init: $\mathbf { s } \gets \zeta ( \mathbf { h } + \delta )$ (diagonal of S) Init: $\mathbf { s } \gets \zeta ( \mathbf { h } + \delta )$ (diagonal of S)
Init: $\pmb { \sigma } ^ { 2 }  ( s _ { 1 } \cdot . . . \cdot s _ { D } ) ^ { 1 / \gamma } / \mathbf { s }$ (scale) Init: $\pmb { \sigma } ^ { 2 }  \kappa _ { t } ( s _ { 1 } \cdot \ldots \cdot s _ { D } ) ^ { - 1 / \nu } / \mathbf { s }$ (scale)
1: for $t = 1 , 2 , \ldots$ do 1: for $t = 1 , 2 , \ldots$ do
2: $\widehat { \mathbf { g } } \gets \nabla _ { \pmb { \theta } } \ell ( \pmb { \theta } ) ,$ $\pmb \theta \sim \mathcal N _ { q } ( { \mathbf m } , \mathrm { d i a g } ( { \pmb \sigma } ^ { 2 } ) )$ 2: $\begin{array} { r } { \widehat { \mathbf { g } } \gets \nabla _ { \pmb { \theta } } \ell ( \pmb { \theta } ) , \qquad \pmb { \theta } \sim \mathcal { T } _ { \nu } ( \mathbf { m } , \mathrm { d i a g } ( \pmb { \sigma } ^ { 2 } ) ) } \end{array}$
3: hb $ \kappa ^ { - 1 } \widehat { \mathbf { g } } \cdot ( \pmb { \theta } - \mathbf { m } ) / \pmb { \sigma } _ { . } ^ { 2 }$ 3: $\widehat { \mathbf { h } } \gets \frac { \nu - 2 } { \nu } \widehat { \mathbf { g } } \cdot ( \pmb { \theta } - \mathbf { m } ) / \sigma ^ { 2 }$
(estimator for $\mathbb { E } _ { \tilde { \pi } } [ \mathrm { d i a g } ( \nabla ^ { 2 } \ell ) ] ~ )$ (estimator for $\mathbb { E } _ { \widetilde { \pi } } [ \mathrm { d i a g } ( \nabla ^ { 2 } \ell ) ] )$
4: $\mathbf { g }  \beta _ { 1 } \mathbf { g } + ( 1 - \mathbf { \bar { \beta } } _ { 1 } ) \mathbf { \widehat { g } }$ 4: $\mathbf { g }  \beta _ { 1 } \mathbf { g } + ( 1 - \mathbf { \bar { \beta } } _ { 1 } ) \mathbf { \widehat { g } }$
5: $\mathbf { h }  \beta _ { 2 } \mathbf { h } + ( 1 - \beta _ { 2 } ) \widehat \mathbf { h }$ 5: $\mathbf { h }  \beta _ { 2 } \mathbf { h } + ( 1 - \beta _ { 2 } ) \widehat \mathbf { h }$
6: $\mathbf { m } \gets \mathbf { m } - \rho _ { t } ( \mathbf { g } + \delta \mathbf { m } ) / ( \mathbf { h } + \delta )$ 6: $\mathbf { m } \gets \mathbf { m } - \rho _ { t } ( \mathbf { g } + \delta \mathbf { m } ) / ( \mathbf { h } + \delta )$
7: $\mathbf { s } \gets \zeta ( \mathbf { h } + \delta )$ 7: $\mathbf { s } \gets \zeta ( \mathbf { h } + \delta )$
8: $\pmb { \sigma } ^ { 2 }  ( s _ { 1 } \cdot . . . \cdot s _ { D } ) ^ { 1 / \gamma } / \mathbf { s }$ 8: $\pmb { \sigma } ^ { 2 }  \kappa _ { t } ( s _ { 1 } \cdot \ldots \cdot s _ { D } ) ^ { - 1 / \nu } / \mathbf { s }$
9: end for 9: end for
10: return $\mathbf { m } , \sigma ^ { 2 }$ 10: return m, $\sigma ^ { 2 }$
```

$$
\begin{array} { r } { \mathbf { F } ( \lambda ) ^ { - 1 } \nabla _ { \lambda } \left[ \mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb { \theta } ) ] + \mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) \right] = \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda ( \mu ) } } [ \bar { \ell } ( \pmb { \theta } ) ] \Big | _ { \mu = \mu ( \lambda ) } + \lambda . } \end{array}\tag{15}
$$

Consequently, the canonical-gradient step is

$$
\begin{array} { r } { \lambda _ { t + 1 }  \lambda _ { t } - \rho _ { t } \mathbf { F } ( \lambda _ { t } ) ^ { - 1 } \nabla _ { \lambda } ( \frac { \mathbb { E } [ \ell ( \pmb \theta ) ] + \mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) } { \pi _ { \lambda } } ) \Big | _ { \lambda _ { t } } = ( 1 - \rho _ { t } ) \lambda _ { t } - \rho _ { t } \nabla _ { \mu } \underset { \pi _ { \lambda ( \mu ) } } { \mathbb { E } } [ \bar { \ell } ( \pmb \theta ) ] \Big | _ { \mu _ { t } } . } \end{array}\tag{16}
$$

In particular, anyfixed point with $\rho _ { t } > 0$ satisfies $\begin{array} { r } { \pmb { \lambda } ^ { * } = - \nabla _ { \pmb { \mu } } \mathbb { E } _ { \pi _ { \pmb { \lambda } ( \pmb { \mu } ) } } [ \bar { \ell } ( \pmb { \theta } ) ] \bigg | _ { \pmb { \mu } = \pmb { \mu } ( \pmb { \lambda } ^ { * } ) } . } \end{array}$

The proof is in App. C. The canonical parameter at convergence is matched with the gradient of the loss. Analogous to the fixed-point interpretation of the BLR, this shows that generalized canonical gradients are intrinsic to the variational problem, rather than an optimization choice. The update in Eq. 16 is one of our main contributions. We call it the generalized BLR, as it applies to families of distributions larger than the exponential family (see Fig. 1). When $\Omega = - \mathcal { H }$ we recover the BLR. If $\eta _ { \lambda _ { 0 } } ( \pmb \theta ) - \ell ( \pmb \theta ) = \eta _ { \lambda ^ { \prime } } ( \pmb \theta ) + \bar { \mathrm { c o n s t } } ,$ ., then $\lambda _ { t + 1 } = ( 1 - \rho _ { t } ) \lambda _ { t } + \rho _ { t } \lambda ^ { \prime }$ , so a unit step recovers the generalized conjugate posterior induced by the tilted score.

## 4 Example: Tsallis Negentropy and q-VON

As our main example we replace the Shannon negentropy by the Tsallis negentropy $\Omega _ { 2 - q }$ and derive Algorithms 1 and 2 (see App. F for details), which we apply to CIFAR-100, CIFAR-10, MNIST (see App. G), and to Bayesian logistic regression on a UCI dataset (Fig. 2). The induced maxent family is the q-exponential family (see App. B.2 for a proof),

$$
\begin{array} { r } { \pi _ { \lambda } ( \pmb { \theta } ) = \mathrm { e x p } _ { q } \left( \lambda ^ { \top } \mathbf { T } ( \pmb { \theta } ) - A _ { 2 - q } ( \lambda ) \right) , } \end{array}\tag{17}
$$

where $\exp _ { q } ( u ) = [ 1 + ( 1 - q ) u ] _ { + } ^ { 1 / ( 1 - q ) }$ when $q \neq 1$ . We take $\Theta = \mathbb { R } ^ { D }$ , with Lebesgue reference measure, so $\pmb { \theta } \in \mathbb { R } ^ { D }$ . For the quadratic statistic $\mathbf { T } ( \pmb \theta ) ^ { \top } = ( \pmb \theta , \pmb \theta \pmb \theta ^ { \top } )$ with canonical parameter $\boldsymbol { \lambda } ^ { \top } = ( \mathbf { S } \mathbf { m } , - \frac { 1 } { 2 } \mathbf { S } )$ , this yields a location-scale $( \mathbf { m } , \pmb { \Sigma } )$ family of q-Gaussians. The quadratic geometry is unchanged, but q controls support and tail shapes: $q < 1$ gives parameter-dependent bounded-support Alg. 1, and $q > 1$ gives heavy-tailed Alg. 2 with $\nu = \bar { 2 } / ( q - 1 ) - D$ . In App. D we review the canonical, location-scale, and expectation parametrizations of bounded-support and heavy-tailed q-Gaussians, as well as a closed-form FYD for the heavy-tailed case, which we use to evaluate the objective optimized by Alg. 2. In App. E we derive the reparameterized gradient estimators underlying lines 2-3 of Algorithms 1 and 2.

Specializing the generalized BLR update (16) to quadratic families with expectation parameters $( \pmb { \mu } , \mathbf { M } ) = ( \mathbb { E } _ { \pi _ { \lambda } } [ \pmb { \theta } ] , \mathbb { E } _ { \pi _ { \lambda } } [ \pmb { \theta } \pmb { \theta } ^ { \top } ] )$ , we obtain Newton-like canonical updates. After converting gradients with respect to $( \mu , \mathbf { M } )$ to gradients with respect to (m, Σ) and applying Bonnet- and Price-type identities (App. E), both the bounded-support and heavy-tailed cases take the common form (App. F)

$$
\mathbf { S } _ { t + 1 } = ( 1 - \rho _ { t } ) \mathbf { S } _ { t } + \rho _ { t } \mathbb { E } _ { \widetilde { \pi } _ { t } ^ { 2 - q } } \left[ \nabla _ { \theta } ^ { 2 } \bar { \ell } ( \theta ) \right] , \quad \mathbf { m } _ { t + 1 } = \mathbf { m } _ { t } - \rho _ { t } \mathbf { S } _ { t + 1 } ^ { - 1 } \mathbb { E } _ { \pi _ { t } } \left[ \nabla _ { \theta } \bar { \ell } ( \theta ) \right] ,\tag{18}
$$

where $\tilde { \pi } ^ { 2 - q }$ denotes the $( 2 - q )$ -escort of π. The updates in Eq. 18 are strikingly similar in form to those of VON and IVON [Khan et al., 2018, Osawa et al., 2019, Shen et al., 2024]. The difference between $q < 1$ and $q > 1$ lies in the relation between (S, m) and the location-scale parametrizations, and in the corresponding escort reweighting appearing in the Price identity. The complete derivations, including the bounded-support and Student-t closed forms, and the escort-based Hessian identities are given in Apps. F, D and E. Algorithms 1 and 2 give practical diagonal implementations of these updates. Relative to IVON, only the sampling law, the constants relating S to the scale, and the corresponding diagonal Stein estimator change. As in IVON, we use minibatch stochastic gradients, a diagonal precision $\mathbf { S } = \mathrm { d i a g } ( \mathbf { s } )$ , the reparametrization ${ \mathbf s } = \zeta ( { \mathbf h } + \delta )$ , and momentum terms for stable deep-learning optimization. While the CIFAR-100 experiment we show in $\operatorname { A p p }$ . G is not a large-scale benchmark, there is no structural obstacle to scaling further, and applying $q { \mathrm { - V O N } }$ to transformer- or LLM-scale training is an empirical engineering question, which we leave to future work.

Mean-Field. In high-dimensional models we use a factorized mean-field q-VON-MF, where each parameter follows a 1D q-Gaussian. We replace Ω in Eq. 13 with a sum of 1D terms. The only change to Algorithms 1 and 2 is to use 1D constants (set $D \stackrel { - } { = } 1 )$ , and update each variance parameter separately rather than through the product $s _ { 1 } \cdots s _ { D } ( \mathrm { A p p . } \mathrm { F . } 2 )$ . This addresses the problem that finite covariance in high-dimensional settings forces heavy-tailed q-values to be close to 1, and that D can dominate the max support radius in bounded-support settings. An exciting future direction that our work opens is training deep networks whose parameters have variable levels of stochasticity; for example, each layer may be associated with distinct, learnable or tunable support- and tail-behaviors.

## 5 Experiments

![](images/90ce1bff6ca54845c7b200c45511d046ced67fa5add099faa16b7e5ed5c3f23b.jpg)  
Figure 2: Bounded-support, mean-field variational posterior (red, blue) means can approach the posterior mode of features $\mathsf { A 7 } _ { - } 0 , \mathsf { A 1 3 } _ { - } \mathsf { p }$ and A7\_ff, by changing the value of $q$ in Bayesian logistic regression on the UCI Credit Approval dataset. This is useful when posterior marginals (grey) are skewed, and thus the empirical mode is badly approximated by a Gaussian posterior (black).

We evaluate $q { \mathrm { - V O N } }$ on Bayesian logistic regression and image classification. Detailed experiment descriptions are in App. G, and results on varying q are in App. H. On Bayesian logistic regression, shrinking the max support radius by decreasing q < 1 lets the q-VON approximation closely track the MCMC posterior mode, in contrast to the Gaussian approximation (Fig. 2). On MNIST, CIFAR-10, and CIFAR-100 we compare q-VON and q-VON-MF against SGD, VSGD, IVON, and a mean-only baseline q-VSGD, using test accuracy, NLL, ECE, Brier score, and AUROC. Generally, q-VON improves over SGD and q-VSGD, showing that learning the posterior shape outperforms noise injection. The choice of tail behavior can trade off generalization and calibration: on MNIST, heavy-tailed q-VON has the highest accuracy while bounded-support q-VON has the best calibration. On CIFAR, q-VON-MF is competitive with (highly tuned) IVON without additional tuning. Our experiments demonstrate that generalized BLR updates can be implemented at the scale of modern neural-network optimizers while extending IVON to posterior families unavailable to the BLR.

## References

P. Alquier. Non-Exponentially Weighted Aggregation: Regret Bounds for Unbounded Loss Functions. In International Conference on Machine Learning (ICML), 2021.

S.-I. Amari. α-Divergence Is Unique, Belonging to both f-divergence and Bregman Divergence Classes. IEEE Transactions on Information Theory, 55(11):4925–4931, 2009.

S.-I. Amari. Information Geometry and its Applications, volume 194. Springer, 2016.

S.-I. Amari and A. Cichocki. Information geometry of divergence functions. Bulletin ofthe Polish academy ofsciences. Technical Sciences, 58(1):183–195, 2010.

S.-I. Amari and A. Ohara. Geometry of q-Exponential Family of Probability Distributions. Entropy, 13(6):1170–1185, 2011.

S.-I. Amari, A. Ohara, and H. Matsuzoe. Geometry of deformed exponential families: Invariant, dually-flat and conformal geometries. Physica A: Statistical Mechanics and its Applications, 391 (18):4308–4319, 2012.

P. G. Bissiri, C. C. Holmes, and S. G. Walker. A General Framework for Updating Belief Distributions. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78(5):1103–1130, 2016.

D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. Variational Inference: A Review for Statisticians. Journal ofthe American Statistical Association, 112(518):859–877, 2017.

M. Blondel, A. Martins, and V. Niculae. Learning classifiers with Fenchel-Young losses: Generalized entropies, margins, and algorithms. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 606–615. PMLR, 2019.

S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

O. Catoni. PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning. arXiv preprint arXiv:0712.0248, 2007.

A. Ghosh and A. Basu. Robust Bayes estimation using the density power divergence. Annals of the Institute ofStatistical Mathematics, 68(2):413–437, 2016.

J. W. Gibbs. Elementary Principles in Statistical Mechanics: Developed with Especial Reference to the Rational Foundations ofThermodynamics. C. Scribner’s sons, 1902.

P. D. Grünwald and N. A. Mehta. Fast Rates for General Unbounded Loss Functions: From ERM to Generalized Bayes. Journal ofMachine Learning Research, 21(56):1–80, 2020.

T. Guilmeau, E. Chouzenoux, and V. Elvira. Regularized rényi divergence minimization through bregman proximal gradient algorithms. Journal ofMachine Learning Research, 26(157):1–56, 2025. URL https://www.jmlr.org/papers/v26/23-0573.html.

C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning (ICML), 2017.

J. Hernandez-Lobato, Y. Li, M. Rowland, T. Bui, D. Hernández-Lobato, and R. Turner. Black-box α-divergence Minimization. In International conference on machine learning, pages 1511–1520. PMLR, 2016.

M. D. Hoffman, D. M. Blei, C. Wang, and J. Paisley. Stochastic Variational Inference. Journal of machine learning research, 2013.

P. Ishwar and P. Moulin. On the existence and characterization of the maxent distribution under general moment inequality constraints. IEEE transactions on information theory, 51(9):3322–3333, 2005.

E. T. Jaynes. Information Theory and Statistical Mechanics. Physical review, 106(4):620, 1957.

J. Jewson, J. Q. Smith, and C. Holmes. Principles of Bayesian Inference Using General Divergence Criteria. Entropy, 20(6):442, 2018.

M. Khan and W. Lin. Conjugate-Computation Variational Inference: Converting Variational Inference in Non-conjugate Models to Inferences in Conjugate Models. In Artificial Intelligence and Statistics, pages 878–887. PMLR, 2017.

M. E. Khan and H. Rue. The Bayesian Learning Rule. Journal of Machine Learning Research, 24 (281):1–46, 2023.

M. E. Khan, D. Nielsen, V. Tangkaratt, W. Lin, Y. Gal, and A. Srivastava. Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam. In International Conference on Machine Learning (ICML), 2018.

J. Knoblauch, J. Jewson, and T. Damoulas. An Optimization-centric View on Bayes’ Rule: Reviewing and Generalizing Variational Inference. Journal ofMachine Learning Research, 23(132):1–109, 2022.

Z. Landsman and J. Nešlehová. Stein’s lemma for elliptical random vectors. Journal ofMultivariate Analysis, 99(5):912–927, 2008.

Z. Landsman, S. Vanduffel, and J. Yao. Some Stein-type Inequalities for Multivariate Elliptical Distributions and Applications. Statistics & Probability Letters, 97:54–62, 2015.

A. Lasota and M. C. Mackey. Chaos, Fractals, and Noise: Stochastic Aspects ofDynamics, volume 97. Springer Science & Business Media, 2013.

Y. Li and R. E. Turner. Rényi Divergence Variational Inference. Advances in neural information processing systems, 29, 2016.

M. Lukashchuk, R. Trésor, W. W. Nuijten, I. Senoz, and B. de Vries. The Quotient Bayesian Learning Rule. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

A. F. Martins, M. Treviso, A. Farinhas, P. M. Aguiar, M. A. Figueiredo, M. Blondel, and V. Niculae. Sparse continuous distributions and Fenchel-Young losses. Journal of Machine Learning Research, 23(257):1–74, 2022.

H. Matsuzoe and A. Ohara. Geometry for q-exponential families. Recent progress in differential geometry and its related fields, pages 55–71, 2011.

J. W. Miller and D. B. Dunson. Robust Bayesian inference via coarsening. Journal ofthe American Statistical Association, 2019.

T. Minka et al. Divergence Measures and Message Passing. Technical report, Microsoft Research, 2005.

K. P. Murphy. Machine Learning: A Probabilistic Perspective. MIT press, 2012.

J. Naudts. Deformed exponentials and logarithms in generalized thermostatistics. Physica A: Statistical Mechanics and its Applications, 316(1-4):323–334, 2002.

J. Naudts. Generalised Thermostatistics. Springer Science & Business Media, 2011.

F. Nielsen. Statistical Divergences between Densities of Truncated Exponential Families with Nested Supports: Duo Bregman and duo Jensen divergences. Entropy, 24(3):421, 2022.

M. Opper and C. Archambeau. The Variational Gaussian Approximation Revisited. Neural computation, 21(3):786–792, 2009.

A. Orvieto, H. Kersting, F. Proske, F. Bach, and A. Lucchi. Anticorrelated Noise Injection for Improved Generalization. In International Conference on Machine Learning (ICML), 2022.

K. Osawa, S. Swaroop, M. E. Khan, A. Jain, R. Eschenhagen, R. E. Turner, and R. Yokota. Practical deep learning with Bayesian principles. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Y. Shen, N. Daheim, B. Cong, P. Nickl, G. M. Marconi, C. Bazan, R. Yokota, I. Gurevych, D. Cremers, M. E. Khan, and T. Möllenhoff. Variational Learning is Effective for Large Deep Networks. In International Conference on Machine Learning (ICML), 2024.

S. Sklaviadis, T. Möllenhoff, A. Martins, and M. Figueiredo. Fenchel-young Variational Learning, 2025. URL https://arxiv.org/abs/2502.10295. arXiv:2502.10295 [cs.LG].

S. Sklaviadis, T. Möllenhoff, M. A. T. Figueiredo, A. F. T. Martins, and M. E. Khan. A Stein Identity for q-Gaussians with Bounded Support. In Conference on Parsimony and Learning (CPAL), 2026.

H. Touchette. Legendre-Fenchel transforms in a nutshell. URL http://www. maths. qmul. ac. uk/ht/archive/lfth2. pdf, page 25, 2005.

M. J. Wainwright, M. I. Jordan, et al. Graphical Models, Exponential Families, and Variational Inference. Foundations and Trends in Machine Learning, 1(1–2):1–305, 2008.

P. M. Williams. Bayesian Conditionalisation and the Principle of Minimum Information. The British Journalfor the Philosophy ofScience, 31(2):131–144, 1980.

K. Wu and J. R. Gardner. Understanding Stochastic Natural Gradient Variational Inference. arXiv preprint arXiv:2406.01870, 2024.

A. Zellner. Optimal Information Processing and Bayes’s Theorem. The American Statistician, 42(4): 278–280, 1988.

T. Zhang. Theoretical Analysis of a Class of Randomized Regularization Methods. In Proceedings of the twelfth annual conference on Computational learning theory, pages 156–163, 1999.

J. Zhu, N. Chen, and E. P. Xing. Bayesian Inference with Posterior Regularization and Applications to Infinite Latent SVMs. The Journal ofMachine Learning Research, 15(1):1799–1847, 2014.

A The sharp Gibbs bound forces the KLD 14   
B Maxent 14   
B.1 Expfam as max Shannon entropy 14   
B.2 q-expfam as Max Tsallis (2 − q)-entropy 16   
B.3 Existence 17   
C Proof of Thm. 1 18   
D q-Gaussians: q-exponential Family Form to Location-Scale 18   
D.1 Canonical q-exponential form 18   
D.2 Radial laws and location-scale form 19   
D.3 Expectation parameters . 20   
D.4 FY Mixed-space Bregman between two heavy-tailed q-Gaussians . 21   
E Bonnet-, Price-, and Stein-type identities for q-Gaussians 21   
E.1 Bonnet-type identities . 22   
E.2 Price-type identities . 23   
F Tsallis Negentropy and q-VON 25   
F.1 q-Variational Online Newton 26   
F.2 Mean-Field Variants . 29   
G Experiments 29   
G.1 Bayesian logistic regression 29   
G.2 Image Classification with Neural Networks 29   
G.3 Empirical Verification of Convergence 31   
H Varying q 32   
H.1 1D Logistic Regression . 32   
H.2 MNIST 33

## A The sharp Gibbs bound forces the KLD

Proposition 3. Fix a prior density $\pi _ { 0 }$ with respect to $h .$ Suppose $\mathbb { D } ( \pi , \pi _ { 0 } )$ is a loss-independent functional such thatfor every measurable loss $\ell : \Theta \to \mathbb { R } \cup \{ + \infty \}$ with

$$
0 < \int e ^ { - \ell ( \pmb { \theta } ) } \pi _ { 0 } ( \pmb { \theta } ) d h ( \pmb { \theta } ) < \infty
$$

and every $\pi \in { \mathcal { P } } _ { : }$

$$
- \log \int e ^ { - \ell ( \pmb { \theta } ) } \pi _ { 0 } ( \pmb { \theta } ) d h ( \pmb { \theta } ) \leq \mathbb { E } _ { \pi } [ \ell ( \pmb { \theta } ) ] + \mathbb { D } ( \pi , \pi _ { 0 } ) ,
$$

with equality at the Gibbs density

$$
\pi _ { \ell } ( \pmb \theta ) = \frac { e ^ { - \ell ( \pmb \theta ) } \pi _ { 0 } ( \pmb \theta ) } { \int e ^ { - \ell ( \pmb \theta ^ { \prime } ) } \pi _ { 0 } ( \pmb \theta ^ { \prime } ) d h ( \pmb \theta ^ { \prime } ) } .
$$

Then for every $\pi \ll \pi _ { 0 }$ such that $\mathbb { D } _ { K L } ( \pi \Vert \pi _ { 0 } ) < \infty ,$ , defining

$$
\ell _ { \pi } ( \pmb \theta ) = - \log \frac { \pi ( \pmb \theta ) } { \pi _ { 0 } ( \pmb \theta ) } ,
$$

we have

$$
\mathbb { D } ( \pi , \pi _ { 0 } ) = \mathbb { D } _ { K L } ( \pi \| \pi _ { 0 } ) .
$$

Proof. We choose a loss whose Gibbs posterior is the candidate posterior we want, and then evaluate the variational identity there. Let $\ell = \ell _ { \pi }$ . Then

$$
\int e ^ { - \ell _ { \pi } ( \pmb { \theta } ) } \pi _ { 0 } ( \pmb { \theta } ) d h ( \pmb { \theta } ) = \int \pi ( \pmb { \theta } ) d h ( \pmb { \theta } ) = 1 ,
$$

so the left-hand side of the inequality is 0. Hence

$$
0 = \mathbb { E } _ { \boldsymbol \pi } \left[ - \log \frac { \pi ( \pmb \theta ) } { \pi _ { 0 } ( \pmb \theta ) } \right] + \mathbb { D } ( \boldsymbol \pi , \pi _ { 0 } ) = - \mathbb { D } _ { \mathrm { K L } } ( \boldsymbol \pi \parallel \pi _ { 0 } ) + \mathbb { D } ( \boldsymbol \pi , \pi _ { 0 } ) .
$$

□

## B Maxent

## B.1 Expfam as max Shannon entropy

The following proof is based on Lasota and Mackey [2013, Prop. 9.1.2].

Proposition 4. Let $( \Theta , A , h )$ be a measure space, and let

$$
\mathbf { T } ( \pmb { \theta } ) = ( T _ { 1 } ( \pmb { \theta } ) , . . . , T _ { P } ( \pmb { \theta } ) )
$$

be a measurable vector-valuedfunction. Assume there exist

$$
\pmb { \mu } = ( \mu _ { 1 } , . . . , \mu _ { P } ) \qquad \quad a n d \qquad \pmb { \lambda } = ( \lambda _ { 1 } , . . . , \lambda _ { P } )
$$

such that

$$
\mu _ { i } = \frac { \int _ { \Theta } T _ { i } ( \pmb { \theta } ) \exp \left( \langle \pmb { \lambda } , \mathbf { T } ( \pmb { \theta } ) \rangle \right) d h ( \pmb { \theta } ) } { \int _ { \Theta } \exp \left( \langle \pmb { \lambda } , \mathbf { T } ( \pmb { \theta } ) \rangle \right) d h ( \pmb { \theta } ) } , \qquad i = 1 , . . . , P ,
$$

and all of the integrals are finite. Then the maximum of the Shannon entropy

$$
\mathcal { H } ( \pi ) = - \int _ { \Theta } \pi ( \pmb { \theta } ) \log \pi ( \pmb { \theta } ) d h ( \pmb { \theta } )
$$

over all densities $\pi \in { \mathcal { P } }$ satisfying

$$
\mu _ { i } = \int _ { \Theta } T _ { i } ( \pmb \theta ) \pi ( \pmb \theta ) d h ( \pmb \theta ) , \qquad i = 1 , . . . , P ,
$$

is attained at

$$
\pi _ { \lambda } ( \pmb { \theta } ) = \frac { \exp \left( \langle \lambda , \mathbf { T } ( \pmb { \theta } ) \rangle \right) } { \int _ { \Theta } \exp \left( \langle \lambda , \mathbf { T } ( \pmb { \theta } ) \rangle \right) d h ( \pmb { \theta } ) } .
$$

Proof. Set

$$
Z = \int _ { \Theta } \exp \left( \left. \mathbf { \lambda } \mathbf { \lambda } \mathbf { \mathbf { \mathrm { , T } } } ( \theta ) \right. \right) d h ( \theta ) , \qquad \pi _ { \lambda } ( \theta ) = Z ^ { - 1 } \exp \left( \left. \mathbf { \lambda } \mathbf { \mathbf { \mathrm { , T } } } ( \theta ) \right. \right) .
$$

Let $\pi \in { \mathcal { P } }$ satisfy

$$
\mu _ { i } = \int _ { \Theta } T _ { i } ( \pmb \theta ) \pi ( \pmb \theta ) d h ( \pmb \theta ) , \qquad i = 1 , . . . , P .
$$

Then

$$
\begin{array} { r l r } {  { \mathcal H ( \pmb \pi ) = - \int _ { \Theta } \pi ( \pmb \theta ) \log \pi ( \pmb \theta ) d h ( \pmb \theta ) } } \\ & { } & { = - \int _ { \Theta } \pi ( \pmb \theta ) \log \pi _ { \lambda } ( \pmb \theta ) d h ( \pmb \theta ) - \int _ { \Theta } \pi ( \pmb \theta ) \log \frac { \pi ( \pmb \theta ) } { \pi _ { \lambda } ( \pmb \theta ) } d h ( \pmb \theta ) } \\ & { } & { \leq - \displaystyle \int _ { \Theta } \pi ( \pmb \theta ) \log \pi _ { \lambda } ( \pmb \theta ) d h ( \pmb \theta ) , } \end{array}
$$

since

$$
\int _ { \Theta } \pi ( \pmb \theta ) \log \frac { \pi ( \pmb \theta ) } { \pi _ { \lambda } ( \pmb \theta ) } d h ( \pmb \theta ) \geq 0 .
$$

Now

$$
\begin{array} { l } { \displaystyle - \int _ { \Theta } \pi ( \pmb { \theta } ) \log \pi _ { \lambda } ( \pmb { \theta } ) d h ( \pmb { \theta } ) = - \int _ { \Theta } \pi ( \pmb { \theta } ) \left( \langle \lambda , \mathbf { T } ( \pmb { \theta } ) \rangle - \log Z \right) d h ( \pmb { \theta } ) } \\ { \displaystyle \qquad = \log Z - \sum _ { i = 1 } ^ { P } \lambda _ { i } \int _ { \Theta } T _ { i } ( \pmb { \theta } ) \pi ( \pmb { \theta } ) d h ( \pmb { \theta } ) } \\ { \displaystyle \qquad = \log Z - \sum _ { i = 1 } ^ { P } \lambda _ { i } \mu _ { i } . } \end{array}
$$

Therefore,

$$
\mathcal { H } ( \pi ) \leq \log Z - \sum _ { i = 1 } ^ { P } \lambda _ { i } \mu _ { i } .
$$

It remains to show that $\pi _ { \lambda }$ attains this value. We compute

$$
\begin{array} { l } { \displaystyle \mathcal { H } ( \pi _ { \lambda } ) = - \int _ { \Theta } \pi _ { \lambda } ( \pmb { \theta } ) \log \pi _ { \lambda } ( \pmb { \theta } ) d h ( \pmb { \theta } ) } \\ { \displaystyle \quad = - \int _ { \Theta } \pi _ { \lambda } ( \pmb { \theta } ) \left( \langle \lambda , \mathbf { T } ( \pmb { \theta } ) \rangle - \log Z \right) d h ( \pmb { \theta } ) } \\ { \displaystyle \quad = \log Z - \sum _ { i = 1 } ^ { P } \lambda _ { i } \int _ { \Theta } T _ { i } ( \pmb { \theta } ) \pi _ { \lambda } ( \pmb { \theta } ) d h ( \pmb { \theta } ) . } \end{array}
$$

By the definition of $\pi _ { \lambda }$

$$
\int _ { \Theta } T _ { i } ( \pmb \theta ) \pi _ { \lambda } ( \pmb \theta ) d h ( \pmb \theta ) = \frac { \int _ { \Theta } T _ { i } ( \pmb \theta ) \exp \left( \langle \pmb \lambda , \mathbf T ( \pmb \theta ) \rangle \right) d h ( \pmb \theta ) } { \int _ { \Theta } \exp \left( \langle \pmb \lambda , \mathbf T ( \pmb \theta ) \rangle \right) d h ( \pmb \theta ) } = \mu _ { i } .
$$

Hence

$$
\mathcal { H } ( \pi _ { \lambda } ) = \log Z - \sum _ { i = 1 } ^ { P } \lambda _ { i } \mu _ { i } .
$$

So for every feasible $\pi$

$$
\begin{array} { r } { \mathcal { H } ( \pi ) \leq \mathcal { H } ( \pi _ { \lambda } ) , } \end{array}
$$

which proves that $\pi _ { \lambda }$ maximizes the entropy.

## B.2 q-expfam as Max Tsallis $( 2 - q )$ -entropy

Proposition 5. Let $( \Theta , A , h )$ be a measure space, and let

$$
\mathbf { T } ( \pmb { \theta } ) = ( T _ { 1 } ( \pmb { \theta } ) , . . . , T _ { P } ( \pmb { \theta } ) )
$$

be measurable. Fix $q < 2 , q \neq 1$ , and define

$$
\beta _ { q } ( t ) = \left\{ \begin{array} { l l } { \displaystyle t ^ { 2 - q } - t } \\ { ( 2 - q ) ( 1 - q ) } \\ { + \infty , } & { t < 0 } \end{array} \right. ,
$$

and

$$
\Omega _ { 2 - q } ( \pi ) = \int _ { \Theta } \beta _ { q } ( \pi ( \pmb { \theta } ) ) d h ( \pmb { \theta } ) .
$$

On the simplex of densities,

$$
\Omega _ { 2 - q } ( \pi ) = - { \frac { 1 } { 2 - q } } S _ { 2 - q } ( \pi ) , \qquad S _ { 2 - q } ( \pi ) = { \frac { 1 - \int _ { \Theta } \pi ( \theta ) ^ { 2 - q } d h ( \theta ) } { 1 - q } } .
$$

Assume there exist $\pmb { \mu } = ( \mu _ { 1 } , . . . , \mu _ { P } )$ and $\lambda = ( \lambda _ { 1 } , . . . , \lambda _ { P } )$ such that

$$
\pi _ { \lambda } ( \pmb { \theta } ) = \mathrm { e x p } _ { q } \left( \langle \pmb { \lambda } , \mathbf { T } ( \pmb { \theta } ) \rangle - A _ { 2 - q } ( \pmb { \lambda } ) \right)
$$

is a density, $\Omega _ { 2 - q } ( \pi _ { \lambda } ) < \infty ,$ , all the integrals below are finite, and

$$
\mu _ { i } = \int _ { \Theta } T _ { i } ( \pmb \theta ) \pi _ { \lambda } ( \pmb \theta ) d h ( \pmb \theta ) , \qquad i = 1 , . . . , P .
$$

Then $\pi _ { \lambda }$ minimizes $\Omega _ { 2 - q } ,$ , equivalently maximizes $\mathcal { S } _ { 2 - q } ,$ over all densities $\pi \in { \mathcal { P } }$ satisfying

$$
\mu _ { i } = \int _ { \Theta } T _ { i } ( \pmb \theta ) \pi ( \pmb \theta ) d h ( \pmb \theta ) , \qquad i = 1 , . . . , P .
$$

Proof. Set

$$
\eta _ { \lambda } ( \pmb \theta ) = \langle \lambda , \mathbf { T } ( \pmb \theta ) \rangle - A _ { 2 - q } ( \lambda ) .
$$

Let $\pi \in { \mathcal { P } }$ satisfy the same moment constraints as $\pi _ { \lambda }$

We claim that for h-a.e. θ,

$$
\eta _ { \mathsf { \pmb { \lambda } } } ( \pmb { \theta } ) + \frac { 1 } { 2 - q } \in \partial \beta _ { q } \left( \pi _ { \mathsf { \pmb { \lambda } } } ( \pmb { \theta } ) \right) .
$$

If $\pi _ { \lambda } ( \pmb { \theta } ) > 0$ , then

$$
\log _ { q } \pi _ { \lambda } ( \pmb { \theta } ) = \eta _ { \lambda } ( \pmb { \theta } ) ,
$$

and since

$$
\beta _ { q } ^ { \prime } ( t ) = \log _ { q } ( t ) + \frac { 1 } { 2 - q } , \qquad t > 0 ,
$$

it follows that

$$
\eta _ { \lambda } ( \pmb \theta ) + \frac { 1 } { 2 - q } = \beta _ { q } ^ { \prime } \left( \pi _ { \lambda } ( \pmb \theta ) \right) \in \partial \beta _ { q } \left( \pi _ { \lambda } ( \pmb \theta ) \right) .
$$

If $q < 1$ and $\pi _ { \lambda } ( \pmb { \theta } ) = 0$ , then by the definition of $\exp _ { q }$ we have

$$
\eta _ { \mathsf { \mathsf { A } } } ( \pmb { \theta } ) \leq - \frac { 1 } { 1 - q } .
$$

Also, the right derivative at 0 is

$$
\beta _ { q } ^ { \prime } ( 0 ) = - { \frac { 1 } { ( 1 - q ) ( 2 - q ) } } , \qquad \partial \beta _ { q } ( 0 ) = ( - \infty , \beta _ { q } ^ { \prime } ( 0 ) ] .
$$

Therefore

$$
\eta _ { \lambda } ( \pmb \theta ) + \frac { 1 } { 2 - q } \le - \frac { 1 } { 1 - q } + \frac { 1 } { 2 - q } = - \frac { 1 } { ( 1 - q ) ( 2 - q ) } = \beta _ { q } ^ { \prime } ( 0 ) ,
$$

and again

$$
\eta _ { \mathsf { \pmb { \lambda } } } ( \pmb { \theta } ) + \frac { 1 } { 2 - q } \in \partial \beta _ { q } \left( \pi _ { \mathsf { \pmb { \lambda } } } ( \pmb { \theta } ) \right) .
$$

Thus, by the subgradient inequality,

$$
\beta _ { q } ( { \boldsymbol \pi } ( { \pmb \theta } ) ) \geq \beta _ { q } ( { \boldsymbol \pi } _ { \lambda } ( { \pmb \theta } ) ) + \left( \eta _ { \lambda } ( { \pmb \theta } ) + \frac { 1 } { 2 - q } \right) ( { \boldsymbol \pi } ( { \pmb \theta } ) - { \boldsymbol \pi } _ { \lambda } ( { \pmb \theta } ) )
$$

for h-a.e. θ. Integrating we get

$$
\begin{array} { l } { \displaystyle \Omega _ { 2 - q } ( \pi ) - \Omega _ { 2 - q } ( \pi _ { \lambda } ) \geq \int _ { \Theta } \left( \eta _ { \lambda } ( \pmb \theta ) + \frac 1 { 2 - q } \right) ( \pi ( \pmb \theta ) - \pi _ { \lambda } ( \pmb \theta ) ) d h ( \pmb \theta ) } \\ { \displaystyle \qquad = \int _ { \Theta } \eta _ { \lambda } ( \pmb \theta ) ( \pi ( \pmb \theta ) - \pi _ { \lambda } ( \pmb \theta ) ) d h ( \pmb \theta ) , } \end{array}
$$

because both π and $\pi _ { \lambda }$ integrate to 1.

Expand $\eta _ { \lambda } \colon$

$$
\begin{array} { l } { \displaystyle \int _ { \Theta } \eta _ { \lambda } ( \theta ) \left( \pi ( \theta ) - \pi _ { \lambda } ( \theta ) \right) d h ( \theta ) } \\ { \displaystyle = \sum _ { i = 1 } ^ { P } \lambda _ { i } \left( \int _ { \Theta } T _ { i } ( \theta ) \pi ( \theta ) d h ( \theta ) - \int _ { \Theta } T _ { i } ( \theta ) \pi _ { \lambda } ( \theta ) d h ( \theta ) \right) - A _ { 2 - q } ( \lambda ) \left( \int _ { \Theta } \pi ( \theta ) d h ( \theta ) - \int _ { \Theta } \pi _ { \lambda } ( \theta ) d h ( \theta ) \right) = 0 , } \end{array}
$$

since π and $\pi _ { \lambda }$ have the same prescribed moments and both are densities. Hence

$$
\Omega _ { 2 - q } ( \pi ) \geq \Omega _ { 2 - q } ( \pi _ { \lambda } ) .
$$

When $q < 2 ,$

$$
\Omega _ { 2 - q } ( \pi ) = - \frac { 1 } { 2 - q } S _ { 2 - q } ( \pi ) ,
$$

and minimizing $\Omega _ { 2 - q }$ is equivalent to maximizing $S _ { 2 - q } .$

This certifies optimality of a feasible q-exponential candidate. What we have not proven is the existence of λ or of a feasible $\pi _ { \lambda }$

Remark. In general we assume

$$
q < \frac { D + 4 } { D + 2 } , \qquad q \neq 1 .
$$

Then $q < 1$ yields bounded-support q-Gaussians, while

$$
1 < q < \frac { D + 4 } { D + 2 }
$$

yields Student-t q-Gaussians with

$$
\nu = \frac { 2 } { q - 1 } - D > 2 .
$$

Under the parameter range imposed by finite covariance, $q = 2$ is excluded.

## B.3 Existence

Our abstract construction in Sec. 3.2 is stated on the canonical domain

$$
\Lambda _ { \Omega } = \{ \pmb { \lambda } \in \mathbb { R } ^ { P } : \Omega ^ { * } ( \eta _ { \lambda } ) \mathrm { { i s ~ a t t a i n e d ~ b y ~ a ~ u n i q u e ~ } } \pi _ { \lambda } \in \mathcal { P } \} .
$$

This conditional formulation is deliberate. A general existence theorem for arbitrary Ω would require additional infinite-dimensional assumptions on the support, the moment functions, and the regularizer. For the Shannon case, sufficient conditions of this kind are available in Ishwar and Moulin [2005]. For the Tsallis families used in Sec. F, we do not need such a general result because the optimizer is explicit: the bounded-support case is compactly supported, while the heavy-tailed case reduces to a Student-t family whose normalization and moment conditions can be checked directly from the closed form in App. D.

## C Proof of Thm. 1

To generalize Khan and Rue [2023] we derive a canonical gradient descent for optimizing Eq. 14. We use $\bar { \ell } ( \pmb { \theta } ) = \ell ( \pmb { \theta } ) - \eta _ { \lambda _ { 0 } } ( \pmb { \theta } )$ . By Prop. 1, the finite-dimensional potential $\Psi ^ { * }$ characterizes the geometry of the canonical parameter space. Thus, we define the generalized Fisher matrix,

$$
\mathbf { F } ( \pmb { \lambda } ) = \nabla _ { \pmb { \lambda } } ^ { 2 } \Psi ^ { * } ( \pmb { \lambda } ) = \nabla _ { \pmb { \lambda } } ^ { 2 } \Omega ^ { * } ( \eta _ { \pmb { \lambda } } ) = \frac { \partial \pmb { \mu } ( \pmb { \lambda } ) } { \partial \pmb { \lambda } } .\tag{19}
$$

and use the chain rule $\mathbf { F } ( \lambda ) ^ { - 1 } \nabla _ { \lambda } = \mathbf { F } ( \lambda ) ^ { - 1 } \mathbf { F } ( \lambda ) \nabla _ { \mu } = \nabla _ { \mu } ,$ , to obtain a generalized canonical gradient update of the same form as that of Khan and Rue [2023]. Whenever $\mathbf { F } ( \lambda )$ exists and is invertible, the Fenchel-Young BLR is

$$
\begin{array} { r } { \lambda _ { t + 1 }  \lambda _ { t } - \rho _ { t } \mathbf { F } ( \lambda _ { t } ) ^ { - 1 } \nabla _ { \lambda } ( \underline { { \mathbb { E } } } [ \bar { \ell } ( \pmb { \theta } ) ] + \Omega ( \pi _ { \lambda } ) ) \Big \vert _ { \lambda = \lambda _ { t } } = ( 1 - \rho _ { t } ) \lambda _ { t } - \rho _ { t } \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda } } [ \bar { \ell } ( \pmb { \theta } ) ] . } \end{array}\tag{20}
$$

The equality follows from the fact that $\nabla _ { \pmb { \mu } } \Omega ( \pi _ { \lambda } ) = \nabla _ { \pmb { \mu } } \Psi ( \pmb { \mu } ) = \lambda ( \mathrm { S e c . } \ 3 . 2 )$ , which results in the $( 1 - \rho _ { t } )$ factor. This characterizes the stationary condition of the generalized variational objective. At any fixed point $\lambda ^ { * }$ of the canonical update $\dot { \lambda ^ { * } } = - \nabla _ { \pmb { \mu } } \mathbb { E } _ { \pi _ { \lambda ^ { * } } } [ \bar { \ell } ( \pmb { \bar { \theta } } ) ]$

Proof of Thm. 1. By the definition of the Fenchel-Young discrepancy,

$$
\mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb { \theta } ) ] + \mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) = \Omega ^ { \ast } ( \eta _ { \lambda _ { 0 } } ) + \mathbb { E } _ { \pi _ { \lambda } } [ \bar { \ell } ( \pmb { \theta } ) ] + \Omega ( \pi _ { \lambda } ) .\tag{21}
$$

By Prop. $1 , \Omega ^ { * } ( \eta _ { \lambda _ { 0 } } ) = \Psi ^ { * } ( \lambda _ { 0 } )$ , which is constant with respect to λ. By Prop. 2, each $\pi _ { \lambda }$ solves the maxent problem at moment vector $\pmb { \mu } ( \lambda )$ . Hence $\Omega ( \pi _ { \lambda } ) = \Psi ( \mu ( \lambda ) )$ . Therefore, up to the additive constant $\Psi ^ { * } ( \lambda _ { 0 } )$

$$
\begin{array} { r } { \mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb { \theta } ) ] + \mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) = \mathbb { E } _ { \pi _ { \lambda } } [ \bar { \ell } ( \pmb { \theta } ) ] + \Psi ( \pmb { \mu } ( \lambda ) ) . } \end{array}
$$

Taking the gradient w.r.t. λ and applying the chain rule

$$
\begin{array} { r l r } & { } & { \nabla _ { \lambda } \left[ \mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb \theta ) ] + \mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) \right] = \displaystyle \frac { \partial \mu ( \lambda ) } { \partial \lambda } \left[ \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda ( \mu ) } } [ \bar { \ell } ( \pmb \theta ) ] + \nabla _ { \mu } \Psi ( \mu ) \right] _ { \mu = \mu ( \lambda ) } } \\ & { } & { = { \bf F } ( \lambda ) \left[ \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda ( \mu ) } } [ \bar { \ell } ( \pmb \theta ) ] + \nabla _ { \mu } \Psi ( \mu ) \right] _ { \mu = \mu ( \lambda ) } . } \end{array}\tag{22}
$$

Using the dual-coordinate relation $\nabla _ { \pmb { \mu } } \Psi ( \pmb { \mu } ( \pmb { \lambda } ) ) = \pmb { \lambda }$ and multiplying by $\mathbf { F } ( \lambda ) ^ { - 1 }$ we get

$$
\begin{array} { r } { \mathbf { F } ( \lambda ) ^ { - 1 } \nabla _ { \lambda } \left[ \mathbb { E } _ { \pi _ { \lambda } } [ \ell ( \pmb { \theta } ) ] + \mathbb { D } _ { \Omega } ( \eta _ { \lambda _ { 0 } } , \pi _ { \lambda } ) \right] = \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda ( \mu ) } } [ \bar { \ell } ( \pmb { \theta } ) ] \Big | _ { \mu = \mu ( \lambda ) } + \lambda . } \end{array}
$$

This proves Eq. 15. Substituting this identity into the canonical-gradient step gives

$$
\lambda _ { t + 1 } = \lambda _ { t } - \rho _ { t } \left[ \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda ( \mu ) } } [ \bar { \ell } ( \pmb { \theta } ) ] \bigg | _ { \mu = \mu ( \lambda _ { t } ) } + \lambda _ { t } \right] ,
$$

which rearranges to

$$
\lambda _ { t + 1 } = ( 1 - \rho _ { t } ) \lambda _ { t } - \rho _ { t } \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda ( \mu ) } } [ \bar { \ell } ( \pmb { \theta } ) ] \bigg | _ { \mu = \mu ( \lambda _ { t } ) } .
$$

The fixed-point condition follows by setting $\lambda _ { t + 1 } = \lambda _ { t } = \lambda ^ { * }$ with $\rho _ { t } > 0 .$

## D $q { \mathbf { - } } \mathbf { G }$ aussians: q-exponential Family Form to Location-Scale

## D.1 Canonical q-exponential form

We start from the q-exponential family canonical parametrization of $q \cdot$ -Gaussians (Eq. 82). To derive Alg. 1 and $\mathrm { A l g } . 2$ we use the following sufficient statistics and canonical parameters, which along with the deformation index q determine the q-Gaussian density

$$
{ \bf T } ( \theta ) = \binom { \theta } { \theta \theta ^ { \top } } , \qquad \lambda = \binom { { \bf S } { \bf m } } { - \frac { 1 } { 2 } { \bf S } } .\tag{23}
$$

Both bounded-support $( q < 1 )$ , and heavy-tailed densities $( q > 1 )$ can be written with subtractive normalization in q-exponential form

$$
\begin{array} { r l r } { \pi _ { \lambda } ( \theta ) = \exp _ { q } \left( \lambda ^ { \top } \mathbf { T } ( \theta ) - A _ { 2 - q } ( \lambda ) \right) , } & { } & { \lambda ^ { \top } \mathbf { T } ( \theta ) = - \frac { 1 } { 2 } ( \theta - \mathbf { m } ) ^ { \top } \mathbf { S } ( \theta - \mathbf { m } ) + \frac { 1 } { 2 } \mathbf { m } ^ { \top } \mathbf { S } \mathbf { m } , } \end{array}\tag{24}
$$

with constants absorbed into the deformed cumulant $A _ { 2 - q }$

When $q < 1$

$$
A _ { 2 - q } ( \mathbf { \boldsymbol { \lambda } } ) = \frac { 1 } { 2 } \mathbf { m } ^ { \top } \mathbf { S m } - \frac { 1 } { 2 } \left( R ( q , D ) ^ { 2 } | \mathbf { S } | ^ { 1 / \gamma } - \frac { 2 } { 1 - q } \right)\tag{25}
$$

with constant $\begin{array} { r } { \gamma = D + \frac { 2 } { 1 - q } } \end{array}$ , and where the max support radius $R$ has a closed form that we derive below.

When ${ \bf q } > { \bf 1 }$

$$
A _ { 2 - q } ( \mathbf { \lambda } \mathbf { \lambda } ) = \frac { 1 } { 2 } \mathbf { m } ^ { \top } \mathbf { S } \mathbf { m } + \frac { 1 } { q - 1 } \left( | \mathbf { S } | ^ { - 1 / \nu } \frac { \nu } { \nu + D } \kappa _ { t } ( \nu , D ) - 1 \right)\tag{26}
$$

where Γ is the Gamma function, and $\begin{array} { r } { \nu = \frac { 2 } { q - 1 } - D } \end{array}$ is the Student-t parameter as we show in more detail below where we also define constant $\kappa _ { t }$

## D.2 Radial laws and location-scale form

To derive the closed forms of the covariances of θ that we use to derive the Newton-like updates in Sec. F.1 for the different ranges of $q$ we need the following polar coordinate stochastic reparametrization. For feature dimension D we have

$$
r ^ { 2 } ( \theta ) = ( \theta - { \bf m } ) ^ { \top } { \boldsymbol \Sigma } ^ { - 1 } ( \theta - { \bf m } ) , \qquad \theta = { \bf m } + { \boldsymbol \Sigma } ^ { 1 / 2 } { \bf z } , \quad { \bf z } = { \boldsymbol \Sigma } ^ { - 1 / 2 } ( \theta - { \bf m } ) = r { \bf u } ,\tag{27}
$$

where $\mathbf { u } \sim \mathrm { U n i f } ( S ^ { D - 1 } )$ on the unit sphere surface $S ^ { D - 1 } = \lbrace \mathbf { u } : \mathbf { u } ^ { \top } \mathbf { u } = 1 \rbrace$ , and $r \geq 0$ is independent of u.

When $q < 1$ we use the determinant-rescaled elliptical scale matrix $\pmb { \Sigma } = | \mathbf { S } | ^ { 1 / \gamma } \mathbf { S } ^ { - 1 }$ to write $\pi _ { \lambda } ( \pmb \theta )$ in elliptical Pearson Type II, location-scale form, which is obtained from Eq. 24 and Eq. 25 by the definition of $\exp _ { q } { ( \mathrm { E q . } 8 3 ) }$ and some algebraic manipulation

$$
\pi _ { \lambda } ( \pmb \theta ) = | \pmb \Sigma | ^ { - 1 / 2 } \left[ \frac { 1 - q } { 2 } \left( R ^ { 2 } - r ^ { 2 } ( \pmb \theta ) \right) \right] _ { + } ^ { \frac { 1 } { 1 - q } } , \qquad \mathrm { s u p p } ( \pi _ { \lambda } ) = \{ \pmb \theta : r ^ { 2 } ( \pmb \theta ) < R ^ { 2 } \} .\tag{28}
$$

Evaluating the integral $\begin{array} { r } { \int _ { \mathbb { R } ^ { D } } \left[ \frac { 1 - q } { 2 } \left( R ^ { 2 } - \| { \mathbf z } \| ^ { 2 } \right) \right] _ { + } ^ { \frac { 1 } { 1 - q } } d { \mathbf z } \ = \ 1 } \end{array}$ in terms of the polar coordinate reparametrization from Eq. 27 yields the closed form of the max support radius $R ( q , D )$ )

$$
R ( q , D ) = \left[ \frac { \Gamma \left( \frac { D } { 2 } + \frac { 2 - q } { 1 - q } \right) } { \pi ^ { D / 2 } \Gamma \left( \frac { 2 - q } { 1 - q } \right) } \left( \frac { 2 } { 1 - q } \right) ^ { \frac { 1 } { 1 - q } } \right] ^ { \frac { 1 - q } { 2 + D ( 1 - q ) } } ,\tag{29}
$$

where Γ is the Gamma function.

The squared “radial fraction” follows a Beta distribution:

$$
\frac { r ^ { 2 } } { R ^ { 2 } ( q , D ) } \sim \mathrm { B e t a } \left( \frac { D } { 2 } , \frac { 2 - q } { 1 - q } \right) , \qquad r \in [ 0 , R ( q , D ) ] .\tag{30}
$$

Hence sampling can be implemented as $\begin{array} { r } { \mathbf { u } \sim \mathrm { U n i f } ( S ^ { D - 1 } ) , \ \frac { r ^ { 2 } } { R ^ { 2 } ( q , D ) } \sim \mathrm { B e t a } ( \frac { D } { 2 } , \frac { 2 - q } { 1 - q } ) , \mathbf { z }  r \mathbf { u } } \end{array}$ $\pmb { \theta } \gets \mathbf { m } + \pmb { \Sigma } ^ { 1 / 2 } \mathbf { z }$

When ${ \bf q } > { \bf 1 }$

$$
\nu = { \frac { 2 } { q - 1 } } - D \quad { \mathrm { a n d } } \quad\tag{31}
$$

$$
\Sigma = \kappa _ { t } | \mathbf { S } | ^ { - 1 / \nu } \mathbf { S } ^ { - 1 } \quad \mathrm { w h e r e } \quad \kappa _ { t } ( \nu , D ) = \frac { \nu + D } { \nu } \left[ ( \nu + D ) ^ { D / 2 } \pi ^ { D / 2 } \frac { \Gamma ( \nu / 2 ) } { \Gamma ( ( \nu + D ) / 2 ) } \right] ^ { 2 / \nu } ,\tag{32}
$$

From Eq. 24 and Eq. 26 by the definition of $\exp _ { q } \ ( \mathrm { E q . } \ 8 3 )$ and some algebraic manipulation we obtain the elliptical Pearson Type VII form of the density

$$
\pi _ { \lambda } ( \pmb \theta ) = \left( | \mathbf S | ^ { - 1 / \nu } \frac \nu { \nu + D } \kappa _ { t } + \frac { q - 1 } 2 ( \pmb \theta - \mathbf m ) ^ { \top } \mathbf S ( \pmb \theta - \mathbf m ) \right) ^ { - \frac 1 { q - 1 } } \propto \left( 1 + \frac 1 \nu r ^ { 2 } ( \pmb \theta ) \right) ^ { - ( \nu + D ) / 2 } .\tag{33}
$$

Finally, we can bring the density in Student-t form:

$$
{ \cal T } _ { \nu } ( \pmb \theta | \mathrm { \bf ~ m } , \pmb \Sigma ) = { \frac { 1 } { Z _ { \nu } | \pmb \Sigma | ^ { 1 / 2 } } } \left( 1 + { \textstyle { \frac { 1 } { \nu } } } r ^ { 2 } ( \pmb \theta ) \right) ^ { - ( \nu + D ) / 2 } , \qquad Z _ { \nu } = { \frac { \Gamma ( \nu / 2 ) ( \nu \pi ) ^ { D / 2 } } { \Gamma ( ( \nu + D ) / 2 ) } } ,\tag{34}
$$

since with $\Sigma$ as defined above, Eq. 33 is proportional to the standard Student-t kernel.

In standardized coordinates the Pearson Type VII density is:

$$
\pi ( \mathbf { z } ) = \frac { 1 } { Z _ { \nu } } \left( 1 + \frac { 1 } { \nu } \| \mathbf { z } \| ^ { 2 } \right) ^ { - ( \nu + D ) / 2 } , \qquad \mathbf { z } \in \mathbb { R } ^ { D } .\tag{35}
$$

Using again ${ \mathbf z } = r { \mathbf u }$ with Jacobian $d \mathbf { z } = r ^ { D - 1 }$ drdu and integrating out the angular part u, the induced density of the radius r is proportional to

$$
\begin{array} { r } { r ^ { D - 1 } \left( 1 + \frac { 1 } { \nu } r ^ { 2 } \right) ^ { - ( \nu + D ) / 2 } , \qquad r \in [ 0 , \infty ) . } \end{array}\tag{36}
$$

Equivalently, by the change of variables $r \mapsto r ^ { 2 } ( { \mathrm { i . e . ~ } } d r = { \textstyle { \frac { 1 } { 2 } } } ( r ^ { 2 } ) ^ { - 1 / 2 } d ( r ^ { 2 } ) )$ , the induced density of $r ^ { 2 }$ is proportional to

$$
\begin{array} { r } { ( r ^ { 2 } ) ^ { \frac { D } { 2 } - 1 } \left( 1 + \frac { 1 } { \nu } r ^ { 2 } \right) ^ { - ( \nu + D ) / 2 } , \qquad r ^ { 2 } \in ( 0 , \infty ) , } \end{array}\tag{37}
$$

which is a Beta-prime kernel. Hence,

$$
{ \frac { r ^ { 2 } } { \nu } } \sim \mathrm { B e t a P r i m e } \left( { \frac { D } { 2 } } , { \frac { \nu } { 2 } } \right) .\tag{38}
$$

A standard equivalent representation is the chi-square ratio or Fisher-F form:

$$
r ^ { 2 } \stackrel { d } { = } \nu \frac { \chi _ { D } ^ { 2 } } { \chi _ { \nu } ^ { 2 } } = D F _ { D , \nu } , \qquad r \in [ 0 , \infty ) ,\tag{39}
$$

with independent chi-squares in the first expression.

We can sample ${ \bf u } \sim \mathrm { U n i f } ( S ^ { D - 1 } ) , U \sim \chi _ { D } ^ { 2 } , V \sim \chi _ { \nu } ^ { 2 }$ independently, set $r \gets \sqrt { \nu U / V } , \mathbf { z } \gets r \mathbf { u }$ and finally $\pmb { \theta }  \mathbf { m } + \pmb { \Sigma } ^ { 1 / 2 } \mathbf { z }$ . Equivalently, sample $F \sim F _ { D , \nu }$ and set $r  \sqrt { D F }$

## D.3 Expectation parameters

To derive the general mixed Bregman form of the BLR we use linear moments under $\pi _ { \lambda } \colon$

$$
\pmb { \mu } = \mathbb { E } _ { \pi _ { \lambda } } [ \pmb { \theta } ] = \mathbf { m } , \qquad \mathbf { M } = \mathbb { E } _ { \pi _ { \lambda } } [ \pmb { \theta } \pmb { \theta } ^ { \top } ] = \mathbf { m } \mathbf { m } ^ { \top } + \mathrm { V a r } _ { \pi _ { \lambda } } ( \pmb { \theta } ) .\tag{40}
$$

For any elliptical $\pi _ { \lambda }$ with stochastic representation $\pmb { \theta } = \mathbf { m } + \pmb { \Sigma } ^ { 1 / 2 } ( r \mathbf { u } )$ , using $\pmb \theta - \pmb \mu = \pmb \Sigma ^ { 1 / 2 } ( r \mathbf u )$

$$
\operatorname { V a r } _ { \pi _ { \lambda } } ( \pmb { \theta } ) = \pmb { \Sigma } ^ { 1 / 2 } \mathbb { E } [ r ^ { 2 } ] \mathbb { E } [ \mathbf { u } \mathbf { u } ^ { \top } ] \pmb { \Sigma } ^ { 1 / 2 } = \pmb { \Sigma } ^ { 1 / 2 } \mathbb { E } [ r ^ { 2 } ] \frac { 1 } { D } \mathbf { I } \pmb { \Sigma } ^ { 1 / 2 } = \frac { \mathbb { E } [ r ^ { 2 } ] } { D } \pmb { \Sigma } .\tag{41}
$$

In particular, from Eq. 41 and the radial laws Eq. 30 and Eq. 39:

$$
q < 1 : { \mathrm { V a r } } _ { \pi _ { \lambda } } ( \theta ) = \kappa ( q , D ) \Sigma , \qquad q > 1 : { \mathrm { V a r } } _ { \pi _ { \lambda } } ( \theta ) = { \frac { \nu } { \nu - 2 } } \Sigma \quad ( \nu > 2 )\tag{42}
$$

with $\begin{array} { r } { \kappa = \frac { { R } ^ { 2 } } { D + \frac { 2 ( 2 - q ) } { 1 - q } } } \end{array}$ as in Alg. 1.

Escort reweightings. The Price- and Stein-type identities that can be derived for q-Gaussians naturally introduce a $( 2 - q )$ -escorts (see App. E for more details on related bibliography, and a proof of the heavy-tailed Price-type identity):

$$
\tilde { \pi } ^ { 2 - q } ( \pmb { \theta } ) \propto \pi ( \pmb { \theta } ) w _ { 2 - q } ( \pmb { \theta } ) ,\tag{43}
$$

with

$$
q < 1 : \quad w _ { 2 - q } ( \pmb \theta ) = R ^ { 2 } ( q , D ) - r ^ { 2 } ( \pmb \theta ) , \qquad q > 1 : \quad w _ { 2 - q } ( \pmb \theta ) = 1 + \frac { 1 } { \nu } r ^ { 2 } ( \pmb \theta ) .\tag{44}
$$

## D.4 FY Mixed-space Bregman between two heavy-tailed q-Gaussians

Analogously to the expression in [Martins et al., 2022, Prop. 18] we derive the FY mixed-space Bregman divergence for the $q > 1$ case. For $\mathbf { T } ( \pmb { \theta } ) = ( \pmb { \theta } , \pmb { \theta } \pmb { \theta } ^ { \top } )$ and $\boldsymbol { \lambda } = ( \mathbf { S } \mathbf { m } , - \frac { 1 } { 2 } \mathbf { S } )$ , we consider the quadratic form of a score $\eta _ { 0 } ( \pmb \theta ) = \lambda _ { 0 } ^ { \top } \mathbf { T } ( \pmb \theta )$ in the same linear family $\lambda _ { \mathrm { 0 } } = \binom { \bf S _ { \mathrm { 0 } } { \bf m } _ { \mathrm { 0 } } } { - \frac { 1 } { 2 } { \bf S } _ { \mathrm { 0 } } }$ . Up to an additive constant independent of $\pmb { \theta } \eta _ { 0 } ( \pmb { \theta } ) = - \frac { 1 } { 2 } ( \pmb { \theta } - \mathbf { m } _ { 0 } ) ^ { \top } \mathbf { S } _ { 0 } ( \pmb { \theta } - \mathbf { m } _ { 0 } )$ . Let $\pi _ { 0 }$ denote the q-Gaussian induced by this score, i.e.

$$
\pi _ { 0 } ( \pmb { \theta } ) = \mathcal { T } _ { \nu } ( \pmb { \theta } | \mathbf { m } _ { 0 } , \pmb { \Sigma } _ { 0 } ) , \qquad \pmb { \Sigma } _ { 0 } = \kappa _ { t } | \mathbf { S } _ { 0 } | ^ { - 1 / \nu } \mathbf { S } _ { 0 } ^ { - 1 } .\tag{45}
$$

The FY mixed-space Bregman divergence induced by $\Omega _ { 2 - q }$ is

$$
\mathrm { D } _ { \Omega _ { 2 - q } } ( \eta _ { 0 } , \pi ) = \Omega _ { 2 - q } ( \pi ) - \Omega _ { 2 - q } ( \pi _ { 0 } ) - \lambda _ { 0 } ^ { \top } \left( \pmb { \mu } ( \pi ) - \pmb { \mu } ( \pi _ { 0 } ) \right) ,\tag{46}
$$

where $\pmb { \mu } ( \pi ) = ( \mathbb { E } _ { \pi } [ \pmb { \theta } ] , \mathbb { E } _ { \pi } [ \pmb { \theta } \pmb { \theta } ^ { \top } ] )$ . Using $\mathbb { E } _ { \pi } [ \pmb { \theta } ] = \mathbf { m }$ and $\mathbb { E } _ { \pi } [ \pmb { \theta } \pmb { \theta } ^ { \top } ] = \mathbf { m } \mathbf { m } ^ { \top } + \mathrm { V a r } _ { \pi } ( \pmb { \theta } )$ , we get

$$
\begin{array} { r } { - \lambda _ { 0 } ^ { \top } ( \pmb { \mu } ( \pi ) - \pmb { \mu } ( \pi _ { 0 } ) ) = \frac { 1 } { 2 } ( \mathbf { m } - \mathbf { m } _ { 0 } ) ^ { \top } \mathbf { S } _ { 0 } ( \mathbf { m } - \mathbf { m } _ { 0 } ) } \\ { + \frac { 1 } { 2 } \mathrm { T r } \left( \mathbf { S } _ { 0 } \left( \mathrm { V a r } _ { \pi } ( \pmb { \theta } ) - \mathrm { V a r } _ { \pi _ { 0 } } ( \pmb { \theta } ) \right) \right) . } \end{array}\tag{47}
$$

For Student-t with $\begin{array} { r } { \nu > 2 , \mathrm { V a r } _ { \pi } ( \pmb { \theta } ) = \frac { \nu } { \nu - 2 } \pmb { \Sigma } } \end{array}$ , hence

$$
\frac { 1 } { 2 } \mathrm { T r } \left( \mathbf { S } _ { 0 } \left( \mathrm { V a r } _ { \pi } ( \pmb { \theta } ) - \mathrm { V a r } _ { \pi _ { 0 } } ( \pmb { \theta } ) \right) \right) = \frac { \kappa _ { t } \nu } { 2 ( \nu - 2 ) } \left( | \mathbf { S } | ^ { - 1 / \nu } \mathrm { T r } ( \mathbf { S } _ { 0 } \mathbf { S } ^ { - 1 } ) - D | \mathbf { S } _ { 0 } | ^ { - 1 / \nu } \right) .\tag{48}
$$

Using the deformed cumulant expression in Eq. 26 and $\begin{array} { r } { \Omega _ { 2 - q } ( \pi ) = \frac { 1 } { 2 - q } ( \lambda ^ { \top } \pmb { \mu } - A _ { 2 - q } ( \lambda ) ) } \end{array}$ , we obtain (up to a constant depending only on $( \nu , D ) ,$ 0

$$
\Omega _ { 2 - q } ( \pi ) = \mathrm { c o n s t . } ( \nu , D ) - { \frac { \kappa _ { t } \nu ( \nu + D ) } { 2 ( \nu - 2 ) } } | { \bf S } | ^ { - 1 / \nu } ,\tag{49}
$$

and the difference simplifies to

$$
\Omega _ { 2 - q } ( \pi ) - \Omega _ { 2 - q } ( \pi _ { 0 } ) = - \frac { \kappa _ { t } \nu ( \nu + D ) } { 2 ( \nu - 2 ) } \left( | \mathbf { S } | ^ { - 1 / \nu } - | \mathbf { S } _ { 0 } | ^ { - 1 / \nu } \right) .\tag{50}
$$

Putting everything together we get the heavy-tail, $q > 1$ FY mixed-space Bregman divergence

$$
\begin{array} { l } { { \displaystyle \mathrm { D } _ { \Omega _ { 2 - q } } ( \eta _ { 0 } , \pi ) = \frac { 1 } { 2 } ( \mathbf { m } - \mathbf { m } _ { 0 } ) ^ { \top } \mathbf { S } _ { 0 } ( \mathbf { m } - \mathbf { m } _ { 0 } ) } \ ~ } \\ { { \displaystyle ~ + \frac { \kappa _ { t } \nu } { 2 ( \nu - 2 ) } \left[ | \mathbf { S } | ^ { - 1 / \nu } \left( \mathrm { T r } ( \mathbf { S } _ { 0 } \mathbf { S } ^ { - 1 } ) - ( \nu + D ) \right) + \nu | \mathbf { S } _ { 0 } | ^ { - 1 / \nu } \right] } \ . } \end{array}\tag{51}
$$

## E Bonnet-, Price-, and Stein-type identities for q-Gaussians

In Alg. 1 and $\mathrm { A l g } . 2$ we replace the Gaussian samples of VON with elliptical q-Gaussian samples, using generalized Bonnet and Price identities related to appropriate elliptical Stein-type identities. The resulting q-VON algorithms retain VON’s form, differing by factors depending only on $( q , D )$ and closed-form escort reweighting derived from the dual; as $q \to 1$ we recover standard VON.

By generalizing the VON sampling step from a Gaussian to a q-Gaussian distribution we obtain a more general class of scalable $q { \mathrm { - V O N } }$ algorithms, which extend the BLR to optimize over a much wider class of distributions. We focus on variants of VON involving deformed exponential families with bounded, parameter-dependent support $( q < 1 )$ , and with heavy tails $( q > 1 )$ . Elliptical families include q-Gaussians, which are encompassed under Pearson Type II (bounded support) and Type VII (heavy-tailed) distributions for appropriate ranges of $q ,$ and provide bounded-support and heavy-tailed behaviors with tractable geometry and reparameterizations. Further, they admit Stein-type identities which lead to generalized Bonnet and Price theorems [Landsman and Nešlehová, 2008, Landsman et al., 2015, Matsuzoe and Ohara, 2011]. The references concerning the heavy-tailed cases prove Stein-type identities but do not apply them to obtain Bonnet- and Price-type theorems, which we prove below for reference. Related proofs can be found in Khan and Lin [2017], Lukashchuk et al.. A bounded-support q-Gaussian Stein identity is derived and used to prove Bonnet- and Price-type identities in Sklaviadis et al. [2026].

Here we review the gradient identities used to derive the $q { \mathrm { - V O N } }$ updates (Sec. F.1, Alg. 1 and $\mathrm { A l g } . 2 )$ . Throughout, $\pmb \theta \in \mathbb { R } ^ { D }$ is the random variable and $\pi _ { \lambda } ( \pmb \theta )$ is the variational posterior with canonical parameters $\boldsymbol { \lambda } = ( \mathbf { S } \mathbf { m } , - \frac { 1 } { 2 } \mathbf { S } )$ . We write the same distribution in elliptical location-scale form $\pi _ { \lambda } ( \theta ) = q _ { q } ( \theta \mid { \bf m } , \Sigma )$ , with $r ^ { 2 } ( \pmb { \theta } ) = ( \pmb { \theta } - \mathbf { m } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \pmb { \theta } - \mathbf { m } )$

## E.1 Bonnet-type identities

For both $q < 1$ and $q > 1$ , for differentiable $f$ with suitable integrability, the gradient with respect to the location parameter satisfies the Bonnet identity

$$
\nabla _ { \mathbf { m } } \mathbb { E } _ { \pi _ { \lambda } } \left[ f ( \pmb { \theta } ) \right] = \mathbb { E } _ { \pi _ { \lambda } } \left[ \nabla _ { \pmb { \theta } } f ( \pmb { \theta } ) \right] .\tag{52}
$$

## E.1.1 Proof of the $q > 1$ or Student-t Bonnet-type identity

For a proof of the $q < 1$ case see Sklaviadis et al. [2026].

Let $\pi _ { \lambda } ( \theta ) = \mathcal { T } _ { \nu } ( \theta \mid \mathbf { m } , \Sigma )$ with $\nu > 2$ and support R<sup>D</sup>, ${ } _ { . } { } ^ { D }$

$$
\pi _ { \lambda } ( \pmb { \theta } ) = \frac { 1 } { Z _ { \nu } | \pmb { \Sigma } | ^ { 1 / 2 } } \left( 1 + \frac { 1 } { \nu } r ^ { 2 } ( \pmb { \theta } ) \right) ^ { - ( \nu + D ) / 2 } .\tag{53}
$$

Assume $f : \mathbb { R } ^ { D }  \mathbb { R }$ is differentiable and integrable under $\pi _ { \lambda } .$ , and that boundary terms vanish (e.g.   
$f$ and $\nabla _ { \theta } f$ are bounded; it suffices that $\pi _ { \lambda } ( \pmb \theta )$ decays polynomially).

We show Eq. 52 holds componentwise. Fix $i \in \{ 1 , \ldots , D \}$ . By dominated convergence we may differentiate under the integral:

$$
\frac { \partial } { \partial \mathbf { m } _ { i } } \mathbb { E } _ { \boldsymbol { \pi } _ { \lambda } } [ f ( \pmb { \theta } ) ] = \int _ { \mathbb { R } ^ { D } } f ( \pmb { \theta } ) \frac { \partial } { \partial \mathbf { m } _ { i } } \pi _ { \lambda } ( \pmb { \theta } ) d \pmb { \theta } .\tag{54}
$$

Since $\pi _ { \lambda } ( \pmb \theta )$ depends on $( \theta , \mathbf { m } )$ only through $( \pmb \theta - \mathbf m )$ , we have the identity

$$
\frac { \partial } { \partial { \bf m } _ { i } } \pi _ { \lambda } ( \pmb \theta ) = - \frac { \partial } { \partial \pmb \theta _ { i } } \pi _ { \lambda } ( \pmb \theta ) .\tag{55}
$$

Therefore,

$$
\frac { \partial } { \partial \mathbf { m } _ { i } } \mathbb { E } _ { \pi _ { \lambda } } [ f ( \theta ) ] = - \int f ( \theta ) \frac { \partial } { \partial \theta _ { i } } \pi _ { \lambda } ( \theta ) d \theta = - \int \frac { \partial } { \partial \theta _ { i } } \left( \pi _ { \lambda } ( \theta ) f ( \theta ) \right) d \theta + \int \pi _ { \lambda } ( \theta ) \frac { \partial f ( \theta ) } { \partial \theta _ { i } } d \theta .\tag{56}
$$

The total-derivative term vanishes by the boundary assumption, and thus

$$
\frac { \partial } { \partial \mathbf { m } _ { i } } \mathbb { E } _ { \pi _ { \lambda } } [ f ( \pmb { \theta } ) ] = \mathbb { E } _ { \pi _ { \lambda } } \left[ \frac { \partial f ( \pmb { \theta } ) } { \partial \pmb { \theta } _ { i } } \right] .\tag{57}
$$

Stacking coordinates gives the vector form Eq. 52: $\nabla _ { \mathbf { m } } \mathbb { E } _ { \pi _ { \lambda } } [ f ( \pmb { \theta } ) ] = \mathbb { E } _ { \pi _ { \lambda } } [ \nabla _ { \pmb { \theta } } f ( \pmb { \theta } ) ]$

## E.2 Price-type identities

Let f be twice differentiable with $\nabla _ { \theta } f$ bounded and $\nabla _ { \theta } ^ { 2 } f$ integrable. Then the gradient with respect to the scale parameter admits a generalized Price identity of the form

$$
\nabla _ { \Sigma } \mathbb { E } _ { \pi _ { \lambda } } \left[ f ( \pmb { \theta } ) \right] = \frac { 1 } { 2 } \mathbb { E } _ { \pi _ { \lambda } } \left[ \tau _ { q } ( \pmb { \theta } ) \nabla _ { \pmb { \theta } } ^ { 2 } f ( \pmb { \theta } ) \right] ,\tag{58}
$$

where the scalar factor $\tau _ { q } ( \pmb { \theta } )$ depends on the value of $q \colon$

$$
\tau _ { q } ( \pmb \theta ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 - q } { 2 ( 2 - q ) } w _ { 2 - q } , } & { q < 1 , } \\ { \displaystyle \frac { \nu } { \nu + D - 2 } w _ { 2 - q } = \displaystyle \frac { \nu + r ^ { 2 } ( \pmb \theta ) } { \nu + D - 2 } , } & { q > 1 , } \end{array} \right. \mathrm { w h e r e }\tag{59}
$$

$$
\begin{array} { r } { w _ { 2 - q } ( \pmb \theta ) = \left\{ \begin{array} { l l } { R ^ { 2 } ( q , D ) - r ^ { 2 } ( \pmb \theta ) , } & { q < 1 , } \\ { 1 + \frac { 1 } { \nu } r ^ { 2 } ( \pmb \theta ) , } & { q > 1 . } \end{array} \right. } \end{array}\tag{60}
$$

Escort form. For both ranges of q

$$
\tilde { \pi } ^ { 2 - q } ( \pmb { \theta } ) \propto \pi _ { \lambda } ( \pmb { \theta } ) w _ { 2 - q } ( \pmb { \theta } ) .\tag{61}
$$

This means that Eq. 58 can be rewritten compactly in a form that resembles the Gaussian Stein identity as

$$
\nabla _ { \Sigma } \mathbb { E } _ { \pi _ { \lambda } } \left[ f ( \pmb \theta ) \right] = \frac { c _ { q } } { 2 } \mathbb { E } _ { \tilde { \pi } ^ { 2 - q } } \left[ \nabla _ { \pmb \theta } ^ { 2 } f ( \pmb \theta ) \right] ,\tag{62}
$$

with the constant

$$
c _ { q } = \mathbb { E } _ { \pi } \tau _ { q } ( \pmb \theta ) = \left\{ \begin{array} { l l } { \kappa ( q , D ) , } & { q < 1 , } \\ { \displaystyle \frac { \nu } { \nu - 2 } , } & { q > 1 ( \nu > 2 ) . } \end{array} \right.\tag{63}
$$

When $q \to 1$ we retrieve the Gaussian identity $\begin{array} { r } { \nabla _ { \Sigma } \mathbb { E } _ { \mathcal { N } ( \mathbf { m } , \Sigma ) } [ f ] = \frac { 1 } { 2 } \mathbb { E } [ \nabla _ { \theta } ^ { 2 } f ] } \end{array}$

## E.2.1 Diagonal Stein estimator

Eq. (62) shows that the gradient with respect to the scale parameter depends on the Hessian of $f$ averaged under the complementary escort $\tilde { \pi } ^ { 2 - q }$ . In our $q { \mathrm { - V O N } }$ algorithms we restrict to diagonal $\dot { \Sigma }$ and only require the diagonal of $\dot { \mathbb { E } } _ { \tilde { \pi } ^ { 2 - q } } [ \nabla _ { \theta } ^ { 2 } f ( \pmb { \theta } ) ]$ ]. To avoid explicit Hessian evaluations, we use a Stein identity to rewrite these escort Hessian diagonal entries in terms of first derivatives evaluated at samples $\theta \sim \pi _ { \lambda }$

Let $f : \mathbb { R } ^ { D } $ R be differentiable and integrable under $\pi _ { \lambda }$ . Stein-type identities take the following score form

$$
\mathbb { E } _ { \pi _ { \lambda } } \left[ \nabla _ { \pmb { \theta } } f ( \pmb { \theta } ) \right] = - \mathbb { E } _ { \pi _ { \lambda } } \left[ f ( \pmb { \theta } ) \nabla _ { \pmb { \theta } } \log \pi _ { \lambda } ( \pmb { \theta } ) \right] ,\tag{64}
$$

provided boundary terms vanish. For $q > 1$ (unbounded support) a Stein-type identity has been proven by Landsman et al. [2015]. For $q < 1$ the support is bounded but $\pi _ { \lambda } ( \pmb \theta )$ vanishes at the boundary, so a similar identity holds under mild regularity conditions on $f$ as shown in Sklaviadis et al. [2026].

For q-Gaussian densities with scale matrix Σ the scores can be written explicitly as

$$
\nabla _ { \pmb { \theta } } \log \pi _ { \pmb { \lambda } } ( \pmb { \theta } ) = - \frac { 1 } { ( 2 - q ) \tau _ { q } ( \pmb { \theta } ) } \pmb { \Sigma } ^ { - 1 } ( \pmb { \theta } - \mathbf { m } )\tag{65}
$$

Equivalently, and more conveniently for our purposes, these score functions imply an escort covariance form Stein identity:

$$
\begin{array} { r } { \mathbb { E } _ { \pi _ { \lambda } } \left[ ( \pmb { \theta } - \mathbf { m } ) f ( \pmb { \theta } ) \right] = c _ { q } \pmb { \Sigma } \mathbb { E } _ { \tilde { \pi } ^ { 2 - q } } \left[ \nabla _ { \pmb { \theta } } f ( \pmb { \theta } ) \right] . } \end{array}\tag{66}
$$

Algs. 1-2 only require an estimate of a diagonal $\mathbb { E } _ { \tilde { \pi } ^ { 2 - q } } [ \nabla _ { \theta } ^ { 2 } f ( \pmb { \theta } ) ]$ ]. Assume $\Sigma = \mathrm { d i a g } ( \sigma ^ { 2 } )$ and define $\widehat { \mathbf { g } } = \nabla _ { \pmb { \theta } } f ( \pmb { \theta } )$ . Then, componentwise,

$$
\mathbb { E } _ { \widetilde { \pi } ^ { 2 - q } } \left[ \frac { \partial ^ { 2 } f ( \pmb { \theta } ) } { \partial \pmb { \theta } _ { i } ^ { 2 } } \right] = \frac { 1 } { c _ { q } } \mathbb { E } _ { \pi _ { \lambda } } \left[ \frac { \partial f ( \pmb { \theta } ) } { \partial \pmb { \theta } _ { i } } \frac { \pmb { \theta } _ { i } - \mathbf { m } _ { i } } { \pmb { \sigma } _ { i } ^ { 2 } } \right] , \qquad i = 1 , \dots , D .\tag{67}
$$

Thus a single-sample unbiased estimator for the diagonal Hessian under the complementary escort is

$$
\widehat { \mathbf { h } } = \frac { 1 } { c _ { q } } \widehat { \mathbf { g } } \cdot ( \pmb { \theta } - \mathbf { m } ) / \pmb { \sigma } ^ { 2 } , \quad \pmb { \theta } \sim \pi _ { \lambda } ,\tag{68}
$$

which is exactly the form used in Algs. 1 and 2.

## E.2.2 Proof of the $q > 1$ Student-t Price-type identity

For a proof of the $q < 1$ Price-type identity see Sklaviadis et al. [2026].

Let $\pi _ { \lambda } ( \theta ) = \mathcal { T } _ { \nu } ( \theta \mid \mathbf { m } , \Sigma )$ with $\nu > 2 .$ . Assume $f$ is twice continuously differentiable, integrable under $\pi _ { \lambda }$ , and that boundary terms vanish for the integrations by parts below (it is sufficient that $f$ and $\nabla _ { \theta } f$ are bounded).

We prove the componentwise identity

$$
\frac { \partial } { \partial \Sigma _ { i j } } \mathbb { E } _ { \pi _ { \lambda } } [ f ( \pmb { \theta } ) ] = \frac { 1 } { 2 } \mathbb { E } _ { \pi _ { \lambda } } \left[ \frac { \nu + r ^ { 2 } ( \pmb { \theta } ) } { \nu + D - 2 } \frac { \partial ^ { 2 } f ( \pmb { \theta } ) } { \partial \pmb { \theta } _ { i } \partial \pmb { \theta _ { j } } } \right] , \qquad i , j \in \{ 1 , \dots , D \} ,\tag{69}
$$

which is equivalent to Eq. 58 for $q > 1$ since $\begin{array} { r } { \frac { \nu + r ^ { 2 } } { \nu + D - 2 } = \frac { \nu } { \nu + D - 2 } \left( 1 + \frac { 1 } { \nu } r ^ { 2 } \right) } \end{array}$

Derivative with respect to Σ of the expectation. By dominated convergence,

$$
\frac { \partial } { \partial \Sigma _ { i j } } \mathbb { E } _ { \pi _ { \lambda } } [ f ( \pmb { \theta } ) ] = \int f ( \pmb { \theta } ) \frac { \partial } { \partial \Sigma _ { i j } } \pi _ { \lambda } ( \pmb { \theta } ) d \pmb { \theta } = \mathbb { E } _ { \pi _ { \lambda } } \left[ f ( \pmb { \theta } ) \frac { \partial } { \partial \Sigma _ { i j } } \log \pi _ { \lambda } ( \pmb { \theta } ) \right] .\tag{70}
$$

Writing log $\begin{array} { r } { \pi _ { \lambda } ( \pmb { \theta } ) = \mathrm { c o n s t . } - \frac { 1 } { 2 } \log \left| \pmb { \Sigma } \right| - \frac { \nu + D } { 2 } \log \left( 1 + \frac { 1 } { \nu } r ^ { 2 } ( \pmb { \theta } ) \right) } \end{array}$ , and using $\begin{array} { r } { \frac { \partial } { \partial \Sigma _ { i j } } \log | \Sigma | = ( \Sigma ^ { - 1 } ) _ { j i } } \end{array}$ and $\begin{array} { r } { \frac { \partial } { \partial \sum _ { i j } } r ^ { 2 } ( \pmb { \theta } ) = - ( \pmb { \Sigma } ^ { - 1 } ( \pmb { \theta } - \mathbf { m } ) ( \pmb { \theta } - \mathbf { m } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ) _ { i j } } \end{array}$ , we obtain

$$
\frac { \partial } { \partial \Sigma _ { i j } } \log \pi _ { \boldsymbol { \lambda } } ( \pmb { \theta } ) = - \frac { 1 } { 2 } ( \pmb { \Sigma } ^ { - 1 } ) _ { i j } + \frac { \nu + D } { 2 ( \nu + r ^ { 2 } ( \pmb { \theta } ) ) } \left( \pmb { \Sigma } ^ { - 1 } ( \pmb { \theta } - \mathbf { m } ) ( \pmb { \theta } - \mathbf { m } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } \right) _ { i j } .\tag{71}
$$

Define $\mathbf { a } ( \pmb \theta ) = \pmb { \Sigma } ^ { - 1 } ( \pmb \theta - \mathbf { m } )$ so that $r ^ { 2 } ( \pmb { \theta } ) = ( \pmb { \theta } - \mathbf { m } ) ^ { \top } \mathbf { a } ( \pmb { \theta } )$ and $\left( \Sigma ^ { - 1 } ( \pmb { \theta } - \mathbf { m } ) ( \pmb { \theta } - \mathbf { m } ) ^ { \top } \Sigma ^ { - 1 } \right) _ { i j } =$ ${ \bf a } _ { i } ( \pmb { \theta } ) { \bf a } _ { j } ( \pmb { \theta } )$ . Then Eq. 71 becomes

$$
\frac { \partial } { \partial \Sigma _ { i j } } \log \pi _ { \lambda } ( \pmb \theta ) = - \frac { 1 } { 2 } ( \Sigma ^ { - 1 } ) _ { i j } + \frac { \nu + D } { 2 ( \nu + r ^ { 2 } ( \pmb \theta ) ) } \mathbf { a } _ { i } ( \pmb \theta ) \mathbf { a } _ { j } ( \pmb \theta ) .\tag{72}
$$

Therefore,

$$
\frac { \partial } { \partial \Sigma _ { i j } } \mathbb { E } _ { \pi _ { \lambda } } [ f ( \pmb { \theta } ) ] = \mathbb { E } _ { \pi _ { \lambda } } \left[ f ( \pmb { \theta } ) \left( - \frac { 1 } { 2 } ( \pmb { \Sigma } ^ { - 1 } ) _ { i j } + \frac { \nu + D } { 2 ( \nu + r ^ { 2 } ( \pmb { \theta } ) ) } \mathbf { a } _ { i } ( \pmb { \theta } ) \mathbf { a } _ { j } ( \pmb { \theta } ) \right) \right] .\tag{73}
$$

Integration by parts to express Eq. 73 in terms of the Hessian. We use the Student-t score:

$$
\frac { \partial } { \partial \pmb { \theta } _ { i } } \log \pi _ { \lambda } ( \pmb { \theta } ) = - \frac { \nu + D } { \nu + r ^ { 2 } ( \pmb { \theta } ) } \mathbf { a } _ { i } ( \pmb { \theta } ) .\tag{74}
$$

An important simplification is

$$
\frac { \partial } { \partial \pmb { \theta } _ { i } } \left( \pi _ { \lambda } ( \pmb { \theta } ) ( \nu + r ^ { 2 } ( \pmb { \theta } ) ) \right) = - ( \nu + D - 2 ) \pi _ { \lambda } ( \pmb { \theta } ) \mathbf { a } _ { i } ( \pmb { \theta } ) ,\tag{75}
$$

which follows from the product rule, Eq. 74, and $\begin{array} { r } { \frac { \partial } { \partial { \pmb \theta } _ { i } } r ^ { 2 } ( { \pmb \theta } ) = 2 { \bf a } _ { i } ( { \pmb \theta } ) } \end{array}$

Let $\begin{array} { r } { w ( \pmb { \theta } ) = \frac { \nu + r ^ { 2 } ( \pmb { \theta } ) } { \nu + D - 2 } } \end{array}$ . Then Eq. 75 is equivalent to

$$
\frac { \partial } { \partial \pmb { \theta } _ { i } } \left( \pi _ { \pmb { \lambda } } ( \pmb { \theta } ) w ( \pmb { \theta } ) \right) = - \pi _ { \pmb { \lambda } } ( \pmb { \theta } ) \mathbf { a } _ { i } ( \pmb { \theta } ) .\tag{76}
$$

Now integrate by parts twice:

$$
\mathbb { E } _ { \pi _ { \lambda } } \left[ w ( \theta ) \frac { \partial ^ { 2 } f ( \theta ) } { \partial \theta _ { i } \partial \theta _ { j } } \right] = \int \pi _ { \lambda } ( \theta ) w ( \theta ) \frac { \partial ^ { 2 } f ( \theta ) } { \partial \theta _ { i } \partial \theta _ { j } } d \theta = - \int \frac { \partial } { \partial \theta _ { i } } \left( \pi _ { \lambda } ( \theta ) w ( \theta ) \right) \frac { \partial f ( \theta ) } { \partial \theta _ { j } } d \theta
$$

$$
E \underline { { \underline { { q . } } } } \mathord { \stackrel { 7 6 } { = } } \int \pi _ { \lambda } ( \theta ) \mathbf { a } _ { i } ( \theta ) \frac { \partial f ( \theta ) } { \partial \theta _ { j } } d \theta = - \int \frac { \partial } { \partial \theta _ { j } } \left( \pi _ { \lambda } ( \theta ) \mathbf { a } _ { i } ( \theta ) \right) f ( \theta ) d \theta .\tag{77}
$$

We compute the derivative inside the last integral. Since $\begin{array} { r } { \frac { \partial } { \partial \pmb { \theta } _ { i } } \mathbf { a } _ { i } ( \pmb { \theta } ) = ( \pmb { \Sigma } ^ { - 1 } ) _ { i j } } \end{array}$ and $\begin{array} { r l } { \frac { \partial } { \partial \pmb { \theta } _ { j } } \pi _ { \lambda } ( \pmb { \theta } ) = } \end{array}$ $\begin{array} { r } { \pi _ { \lambda } ( \pmb { \theta } ) \frac { \partial } { \partial \pmb { \theta } _ { j } } \log \pi _ { \lambda } ( \pmb { \theta } ) } \end{array}$ , we get

$$
\frac { \partial } { \partial \pmb { \theta } _ { j } } \left( \pi _ { \lambda } ( \pmb { \theta } ) \mathbf { a } _ { i } ( \pmb { \theta } ) \right) = \pi _ { \lambda } ( \pmb { \theta } ) ( \pmb { \Sigma } ^ { - 1 } ) _ { i j } + \pi _ { \lambda } ( \pmb { \theta } ) \mathbf { a } _ { i } ( \pmb { \theta } ) \frac { \partial } { \partial \pmb { \theta } _ { j } } \log \pi _ { \lambda } ( \pmb { \theta } )\tag{78}
$$

Substituting into Eq. 77 yields

$$
\mathbb { E } _ { \pi _ { \lambda } } \left[ w ( \pmb \theta ) \frac { \partial ^ { 2 } f ( \pmb \theta ) } { \partial \pmb \theta _ { i } \partial \pmb \theta _ { j } } \right] = \mathbb { E } _ { \pi _ { \lambda } } \left[ f ( \pmb \theta ) \left( - ( \pmb \Sigma ^ { - 1 } ) _ { i j } + \frac { \nu + D } { \nu + r ^ { 2 } ( \pmb \theta ) } \mathbf { a } _ { i } ( \pmb \theta ) \mathbf { a } _ { j } ( \pmb \theta ) \right) \right] .\tag{79}
$$

Comparing Eq. 79 with Eq. 73 gives exactly the Price-type identity Eq. 69:

$$
\frac { \partial } { \partial \Sigma _ { i j } } \mathbb { E } _ { \pi _ { \lambda } } [ f ( \pmb { \theta } ) ] = \frac { 1 } { 2 } \mathbb { E } _ { \pi _ { \lambda } } \left[ \boldsymbol { w } ( \pmb { \theta } ) \frac { \partial ^ { 2 } f ( \pmb { \theta } ) } { \partial \pmb { \theta } _ { i } \partial \pmb { \theta } _ { j } } \right] ,\tag{80}
$$

which completes the proof.

## F Tsallis Negentropy and q-VON

As the main example we replace the Shannon negentropy by the Tsallis negative entropy $\Omega _ { 2 - q } ,$ which is maximized by q-exponential families (see App. Sec. B for a proof). When we use quadratic statistics the maxent family becomes a q-Gaussian, which has bounded support for $q < 1$ and heavy tails for $q > 1$ . Respectively, these lead to the $q { \mathrm { - V O N } }$ and Student-t-VON algorithms. Thus, the quadratic statistic remains the same, while the choice of Ω changes the support and tail behavior.

$$
\Omega _ { 2 - q } ( \pi ) = { \left\{ \begin{array} { l l } { \frac { \mathbb { E } _ { \pi } [ \pi ^ { 1 - q } - 1 ] } { \big ( ( 2 - q ) ^ { 2 } - ( 2 - q ) \big ) } } & { { \mathrm { i f ~ } } q \neq 1 , } \\ { \mathbb { E } _ { \pi } [ \log \pi ] } & { { \mathrm { i f ~ } } q = 1 , } \end{array} \right. }\tag{81}
$$

Using entropic index $2 - q .$ , and linear expectation constraints, the maxent problem in Eq. 11 is optimized by q-exponential families (see App. B.2 for a proof):<sup>8</sup>

$$
\pi _ { \lambda } ( \pmb \theta ) = \mathrm { e x p } _ { \boldsymbol q } ( \lambda ^ { \top } \mathbf { T } ( \pmb \theta ) - A _ { 2 - \boldsymbol q } ( \lambda ) ) ,\tag{82}
$$

where $A _ { 2 - q } ( \lambda )$ is the deformed cumulant (we give detailed closed form examples in App. D; see also Naudts [2011], Amari and Ohara [2011], Amari [2016], Martins et al. [2022]). The q-exp and q-log functions are defined as

$$
\exp _ { q } ( u ) = \left\{ \begin{array} { l l } { [ 1 + ( 1 - q ) u ] _ { + } ^ { 1 / ( 1 - q ) } } & { \mathrm { i f ~ } q \neq 1 } \\ { \exp u } & { \mathrm { i f ~ } q = 1 } \end{array} \right. , \qquad \log _ { q } ( u ) = \left\{ \begin{array} { l l } { \frac { u ^ { 1 - q } - 1 } { 1 - q } } & { \mathrm { i f ~ } q \neq 1 } \\ { \log u } & { \mathrm { i f ~ } q = 1 } \end{array} \right. ,\tag{83}
$$

and exponential families are recovered when $q = 1$

We will repeatedly use escort distributions associated with a density π. For any order $a > 0$ such that $\textstyle \int \pi ( \pmb { \theta } ) ^ { a } d \hat { \pmb { \theta } } < \infty$ , we define the escort of order a by

$$
\tilde { \pi } ^ { a } ( \pmb { \theta } ) = \frac { \pi ( \pmb { \theta } ) ^ { a } } { \int \pi ( \pmb { \theta } ) ^ { a } d \pmb { \theta } } .
$$

In the Tsallis setting below, the orders $a = q$ and $a = 2 - q$ arise naturally in the gradient identities and in the Newton-like updates.

When q-exponential families are derived by maximum Tsallis entropy subject to linear expectation constraints, $A _ { 2 - q }$ is related to Ω and $\Omega ^ { * }$ as follows [Martins et al., 2022, Prop.10, App. B]:

$$
\begin{array} { l } { { \Omega _ { 2 - q } ( \pi _ { \lambda } ) = \displaystyle \frac { 1 } { 2 - q } ( \lambda ^ { \top } \mu ( \lambda ) - A _ { 2 - q } ( \lambda ) ) } } \\ { { \Omega _ { 2 - q } ^ { * } ( \eta _ { \lambda } ) = ( 1 - q ) \Omega _ { 2 - q } ( \pi _ { \lambda } ) + A _ { 2 - q } ( \lambda ) = \displaystyle \frac { 1 } { 2 - q } \left( ( 1 - q ) \lambda ^ { \top } \mu ( \lambda ) + A _ { 2 - q } ( \lambda ) \right) , } } \end{array}\tag{84}
$$

where $\pmb { \mu } ( \lambda ) = \mathbb { E } _ { \pmb { \theta } \sim \pi _ { \lambda } } [ \mathbf { T } ( \pmb { \theta } ) ] = \nabla _ { \lambda } \Omega _ { 2 - q } ^ { * } ( \eta _ { \lambda } )$ , and

$$
\frac { \partial \pmb { \mu } ( \lambda ) } { \partial \lambda } = \nabla _ { \lambda } ^ { 2 } \Omega _ { 2 - q } ^ { * } ( \eta _ { \lambda } ) = : \mathbf { F } _ { 2 - q } ( \pmb { \lambda } ) .\tag{85}
$$

Eq. 85 is useful in invoking the chain rule to convert gradients into natural or canonical gradients and vice-versa, and the Hessian of $\Omega ^ { * }$ with respect to the canonical parameters $\mathbf { F } _ { 2 - q }$ recovers the Fisher matrix when $q = 1$

From Eq. 84 we obtain:

$$
\begin{array} { r l } & { \nabla _ { \mu } \Omega _ { 2 - q } ( \pi _ { \lambda } ) = \nabla _ { \mu } \mathbb { E } _ { \pi _ { \lambda } } \left[ \frac { 1 } { 2 - q } \log _ { q } \pi _ { \lambda } ( \theta ) \right] } \\ & { \phantom { \frac { 1 } { 2 - q } } = \frac { 1 } { 2 - q } \nabla _ { \mu } \int _ { \rho } \pi _ { \lambda } ( \theta ) [ \boldsymbol { \lambda } ^ { \top } \mathbf { T } ( \theta ) - A _ { 2 - q } ( \lambda ) ] } \\ & { \phantom { \frac { 1 } { 2 - q } } = \frac { 1 } { 2 - q } \nabla _ { \mu } [ \boldsymbol { \lambda } ^ { \top } \underbrace { \mathbb { E } _ { \pi _ { \lambda } } [ \mathbf { T } ( \theta ) ] } _ { \mu ( \lambda ) } - A _ { 2 - q } ( \lambda ) ] } \\ & { \phantom { \frac { 1 } { 2 - q } } = \frac { 1 } { 2 - q } \left[ \frac { \partial \lambda } { \partial \mu } \mu + \lambda - \frac { \partial \lambda } { \partial \mu } \frac { \partial A _ { 2 - q } ( \lambda ) } { \partial \lambda } \right] } \\ & { \phantom { \frac { 1 } { 2 - q } } = \frac { 1 } { 2 - q } \left[ \lambda + \mathbf { F } _ { 2 - q } ^ { - 1 } ( \lambda ) ( \mu - \widetilde { \mu } _ { q } ) \right] . } \end{array}
$$

We define $\widetilde { \pmb { \mu } } _ { q } = \nabla A _ { 2 - q } ( \pmb { \lambda } ) = \mathbb { E } _ { \widetilde { \pi } ^ { q } } [ \mathbf { T } ( \pmb { \theta } ) ]$ as the expectation with respect to the escort distribution $\tilde { \pi } ^ { q }$ of order $q ,$ which is defined as $\tilde { \pi } ^ { q } = { \pi ^ { q } } / \int { \pi ^ { q } d \theta }$ . Eq. 84 further implies that

$$
\pmb { \mu } - \tilde { \pmb { \mu } } _ { q } = ( 1 - q ) \mathbf { F } _ { 2 - q } \pmb { \lambda } ,\tag{86}
$$

and substituting above, it follows that when Ω is the Tsallis negentropy

$$
\nabla _ { \mu } \Omega _ { 2 - q } ( \pi _ { \lambda } ) = \frac { 1 } { 2 - q } ( \lambda + ( 1 - q ) \pmb { \lambda } ) = \lambda ,\tag{87}
$$

which we use below to derive Newton-like generalized BLR algorithms (Alg. 1, Alg. 2).

## F.1 q-Variational Online Newton

We now specialize the generalized BLR to the quadratic statistic $\mathbf { T } ( \pmb \theta ) ^ { \top } = ( \pmb \theta , \pmb \theta \pmb \theta ^ { \top } ) , \pmb \theta \in \mathbb { R } ^ { D }$ , which induces a location-scale family of q-Gaussians. The derivation in this subsection proceeds in three steps. First, we write the generic canonical update for this quadratic family in terms of the expectation parameters $( \mu , \mathbf { M } )$ . Second, we convert gradients with respect to $( \mu , \mathbf { M } )$ into gradients with respect to the location-scale parameters (m, Σ). Third, we specialize our result to bounded-support $q < 1$ and the heavy-tailed $q > 1$ cases, where Bonnet- and Price-type identities yield the Newton-like updates implemented in Alg. 1 and Alg. 2. The derivation follows immediately below the final algorithms, which we present first for orientation.

First, we connect the pseudocode to the general canonical update in Eq. 16. In both $q < 1$ and $q > 1$ cases we start from the same canonical update, and what changes is the relation between expectation parameters and location-scale parameters, as well as the corresponding stochastic identities used to evaluate the gradients. In Variational Online Newton (VON) gradients and Hessians are estimated by sampling from a Gaussian variational posterior Khan et al. [2018]. VON-style natural-gradient steps result in fast convergence and competitive test accuracy in large-scale deep learning applications [Shen et al., 2024]. Here, we generalize these updates to bounded-support (Alg. 1) and heavy-tailed (Alg. 2) q-Gaussian posteriors in a unified way. Remarkably, our generalizations require only small changes in the VON pseudo-code [Shen et al., 2024, Alg. 1], highlighted in blue.

Using the Tsallis negentropy $\Omega _ { 2 - q }$ in Eq. 16 and sufficient statistics $\mathbf { T } ( \pmb \theta ) ^ { \top } = ( \pmb \theta , \pmb \theta \pmb \theta ^ { \top } )$ we obtain $q \mathrm { - }$ Gaussian families with canonical parameters $\boldsymbol { \lambda } ^ { \top } = ( \mathbf { S } \mathbf { m } , - \frac { 1 } { 2 } \mathbf { S } )$ . q-Gaussians include both boundedsupport and heavy-tailed (Student-t) subfamilies. The sufficient statistics and canonical parameters determine the elliptical geometry (that is, the quadratic form), but the value of q determines the support and tail behavior. We derive the specific forms that Eq. 16 takes for $q < 1$ and $q > 1$ resulting respectively in Alg. 1 and $\mathrm { A l g } . 2$ , where in blue we highlight differences from the BLR’s Variational Online Newton form Khan et al. [2018] and IVON [Shen et al., 2024, Alg. 1]. Detailed background on q-Gaussians can be found in Appendix D.

We begin by writing Eq. 16 more explicitly as

$$
\lambda _ { t + 1 } = ( 1 - \rho _ { t } ) \lambda _ { t } - \rho _ { t } \left[ \nabla _ { \pmb { \mu } } \mathbb { E } _ { \pi _ { \lambda } } \bar { \ell } ( \pmb { \theta } ) \right] .\tag{88}
$$

For both $q < 1$ and $q > 1$ the expectation parameters are

$$
\left[ \overset { \pmb { \mu } } { \mathbf { M } } \right] = \left[ \begin{array} { c c } { \mathbb { E } _ { \pi _ { \lambda } } [ \pmb { \theta } ] } \\ { \mathbb { E } _ { \pi _ { \lambda } } [ \pmb { \theta } \pmb { \theta } ^ { \top } ] } \end{array} \right] = \left[ \begin{array} { c c } { \mathbf { m } } \\ { \mathrm { V a r } _ { \pi _ { \lambda } } ( \pmb { \theta } ) + \mathbf { m m } ^ { \top } } \end{array} \right] .\tag{89}
$$

Importantly, $\operatorname { V a r } ( \theta )$ has distinct forms for $q < 1$ and $q > 1$ , and it is not equal to the canonical parameter $\mathbf { S } ^ { - 1 }$ or the scale parameter $\Sigma ,$ unless $q = 1$ in which case $\mathbf { S } ^ { - 1 } = \Sigma = \mathrm { V a r } ( \pmb { \theta } )$

Chain rule. To use Bonnet- and Price-type stochastic gradient identities, we need a chain rule to express gradients with respect to expectation parameters $( \mu , \mathbf { M } )$ as gradients with respect location-scale parameters m $( \mu ) , \Sigma ( \mu , \mathbf { M } )$ . For any function $f ( \mathbf { m } ( \pmb { \mu } ) , \pmb { \Sigma } ( \pmb { \mu } , \mathbf { M } ) )$ , a straightforward generalization of the well known Gaussian chain rule [Opper and Archambeau, 2009, Khan and Lin, 2017] is

$$
\nabla _ { \mu } f \left( \mathbf { m } , \pmb { \Sigma } \right) = \nabla _ { \mathbf { m } } f \nabla _ { \mu } \mathbf { m } + \nabla _ { \pmb { \Sigma } } f \nabla _ { \mu } \pmb { \Sigma } ,\tag{90}
$$

$$
\nabla _ { \mathbf { M } } f \left( \mathbf { m } , \pmb { \Sigma } \right) = \nabla _ { \mathbf { m } } f \nabla _ { \mathbf { M } } \mathbf { m } + \nabla _ { \pmb { \Sigma } } f \nabla _ { \mathbf { M } } \pmb { \Sigma } .\tag{91}
$$

Bounded support. When $q \ < \ 1$ , bounded-support q-Gaussians have maximum support radius $R ( q , D )$ . The max support radius has a closed-form (reviewed in App. D), which avoids dependence on $\mathbf { S } ^ { - 1 }$ by expressing the scale matrix as a determinant reweighting $\pmb { \Sigma } = | \mathbf { S } | ^ { 1 / \gamma } \mathbf { S } ^ { - 1 }$ with constant $\begin{array} { r } { \gamma = D + \frac { \mathbf { \dot { 2 } } } { \mathbf { \Omega } _ { 1 - a } } } \end{array}$ From Eq. 89 we can show $\begin{array} { r c l } { \operatorname { V a r } _ { \pi _ { \lambda } } ( \pmb { \theta } ) } & { \stackrel { - } { = } } & { \kappa \bar { \Sigma } } \end{array}$ with constant $\kappa = R ^ { 2 } / ( D + 2 ( 2 - q ) / ( \dot { 1 } - q ) )$ (see App. D.3).

From Eq. 90 and $\partial \Sigma / \partial \pmb { \mu } = - 2 \kappa ^ { - 1 } \mathbf { m }$ , it follows that

$$
\nabla _ { \mu } f = \nabla _ { \mathbf { m } } f - \frac { 2 } { \kappa } \nabla _ { \Sigma } f \mathbf { m } ,\tag{92}
$$

and from Eq. 91 $\nabla _ { \mathbf { M } } \mathbf { m } = 0$ and ${ \partial \Sigma / \partial \mathbf { M } } = \kappa ^ { - 1 }$ that

$$
\nabla _ { \mathbf { M } } f = \kappa ^ { - 1 } \nabla _ { \Sigma } f .\tag{93}
$$

Setting $f = { \bar { \ell } } ,$ Eq. 88 becomes

$$
\begin{array} { r } { \lambda _ { t + 1 } = ( 1 - \rho _ { t } ) \lambda _ { t } - \rho _ { t } \left[ \nabla _ { \mathbf { m } } \underset { \theta \sim \pi _ { \lambda } } { \mathbb { E } } \bar { \ell } ( \pmb \theta ) - 2 \kappa ^ { - 1 } \left( \nabla _ { \pmb { \Sigma } } \underset { \theta \sim \pi _ { \lambda } } { \mathbb { E } } [ \bar { \ell } ( \pmb \theta ) ] \right) \mathbf { m } \right] . } \end{array}\tag{94}
$$

Rearranging Eq. 94, and using Price- and Bonnet-type identities for bounded-support $q \cdot$ -Gaussians, to push the gradients inside the expectations [Sklaviadis et al., 2026], we obtain the bounded-support Newton-like generalized BLR (Alg. 1):

$$
\mathbf { S } _ { t + 1 } = ( 1 - \rho _ { t } ) \mathbf { S } _ { t } + \rho _ { t } \frac { 1 } { \kappa D } \left( \mathbb { E } _ { \pi _ { t } } [ r ^ { 2 } ( \theta ) ] \mathbb { E } _ { \tilde { \pi } _ { t } ^ { 2 - q } } [ \nabla _ { \theta } ^ { 2 } \bar { \ell } ( \theta ) ] \right) = ( 1 - \rho _ { t } ) \mathbf { S } _ { t } + \rho _ { t } \mathbb { E } _ { \tilde { \pi } _ { t } ^ { 2 - q } } [ \nabla _ { \theta } ^ { 2 } \bar { \ell } ( \theta ) ]\tag{95}
$$

$$
\mathbf { m } _ { t + 1 } = \mathbf { m } _ { t } - \rho _ { t } \mathbf { S } _ { t + 1 } ^ { - 1 } \mathbb { E } _ { \pi _ { t } } [ \nabla \bar { \ell } ( \pmb { \theta } ) ] ,\tag{96}
$$

where $r ^ { 2 } ( \pmb { \theta } ) = ( \pmb { \theta } - \mathbf { m } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \pmb { \theta } - \mathbf { m } )$ and $\tilde { \pi } ^ { 2 - q }$ is the escort distribution of order $2 - q$ of $\pi _ { \lambda }$ The final form of $\operatorname { E q . 9 5 }$ follows by canceling the factor $\kappa ^ { - 1 } D$ with $\mathbb { E } _ { \pi } [ r ^ { 2 } ( \pmb { \theta } ) ]$ , and expressing the expectation with respect to the $( 2 - q )$ -escort.

Heavy tails. q-Gaussians with $q > 1$ resulting from Eq. 82 coincide with Student-t densities with degrees of freedom $\nu = 2 / ( q - 1 ) - D$ and unbounded support $\mathbb { R } ^ { D }$ (further details are reviewed in $ { \mathrm { A p p . } }  { \mathrm { D } } )$ . We assume $\nu > 2 \Longleftrightarrow 1 < q < ( D + 4 ) / ( D \bar { + } 2 )$ so that the covariance exists (and may optionally require $\nu >$ 4 if we need finite kurtosis). We write the Student-t scale matrix as $\mathbf { \Delta } \Sigma = \dot { \kappa _ { t } } | \mathbf { S } | ^ { - 1 / \nu } \mathbf { S } ^ { - \tilde { 1 } }$ with constant

$$
\kappa _ { t } ( \nu , D ) = \frac { \nu + D } { \nu } \left[ ( \nu + D ) ^ { D / 2 } \pi ^ { D / 2 } \frac { \Gamma ( \nu / 2 ) } { \Gamma ( ( \nu + D ) / 2 ) } \right] ^ { 2 / \nu } ,
$$

where Γ is the Gamma function. Using the Student-t parametrization $\pi _ { \lambda } ( \theta ) = \mathcal { T } _ { \nu } ( \theta \mid \mathbf { m } , \Sigma )$ , we can derive from Eq. 89 that $\operatorname { V a r } _ { \pi _ { \lambda } } ( \pmb { \theta } ) \stackrel { \bf { \textstyle = } } { = } \nu / ( \nu - 2 ) \pmb { \Sigma }$ (see App. D.3).

Thus, from Eq. 90 we obtain

$$
\nabla _ { \mu } f = \nabla _ { \mathbf { m } } f - 2 \frac { \nu - 2 } { \nu } \nabla _ { \Sigma } f \mathbf { m } ,\tag{97}
$$

and, from Eq. 91, since $\partial \Sigma / \partial \pmb { \mu } = - 2 / \nu ( \nu - 2 )$ )m and $\partial \boldsymbol { \Sigma } / \partial \mathbf { M } = ( \nu - 2 ) / \nu$

$$
\nabla _ { \mathbf { M } } f = \frac { \nu - 2 } { \nu } \nabla _ { \Sigma } f .\tag{98}
$$

Again, setting $f = { \bar { \ell } } ,$ the chain rule of Eq. 88 results in

$$
\begin{array} { r l } & { \lambda _ { t + 1 } = ( 1 - \rho _ { t } ) \lambda _ { t } } \\ & { \qquad - \rho _ { t } \left[ { \nabla } _ { \mathbf { m } } \mathbb { E } _ { \boldsymbol \pi } [ \bar { \ell } ( \pmb \theta ) ] - 2 \frac { \nu - 2 } { \nu } \left( { \nabla } _ { \Sigma } \mathbb { E } _ { \boldsymbol \pi } [ \bar { \ell } ( \pmb \theta ) ] \right) \mathbf { m } \right] . } \end{array}\tag{99}
$$

In $\mathrm { A p p }$ . E we prove the Student-t Bonnet-type identity $\nabla _ { \mathbf { m } } \mathbb { E } _ { \boldsymbol { \pi } } [ \bar { \ell } ( \pmb { \theta } ) ] = \mathbb { E } _ { \boldsymbol { \pi } } [ \nabla _ { \pmb { \theta } } \bar { \ell } ( \pmb { \theta } ) ]$ and the Student-t Price-type identity:

$$
\nabla _ { \Sigma } \mathbb { E } _ { \pi } [ \bar { \ell } ( \pmb \theta ) ] = \frac { \nu } { 2 \left( \nu + D - 2 \right) } \mathbb { E } _ { \pi } \left[ \left( 1 + \frac { 1 } { \nu } r ^ { 2 } ( \pmb \theta ) \right) \nabla _ { \pmb \theta } ^ { 2 } \bar { \ell } ( \pmb \theta ) \right] .\tag{100}
$$

Similarly to the bounded-support case, we can express the Price-type identity in terms of the $( 2 - q )$ escort

$$
\nabla _ { \pmb { \Sigma } } \mathbb { E } _ { \pi } [ \bar { \ell } ( \pmb { \theta } ) ] = \frac { \nu } { 2 ( \nu - 2 ) } \mathbb { E } _ { \widetilde { \pi } } \left[ \nabla _ { \pmb { \theta } } ^ { 2 } \bar { \ell } ( \pmb { \theta } ) \right] ,\tag{101}
$$

which is a neat fact that, as far as we know, has not been noticed before. Since $q = 1 + 2 / ( \nu + D )$ and $\pi ( \pmb { \theta } ) \propto \left( 1 + r ^ { 2 } ( \pmb { \theta } ) / \nu \right) ^ { - ( \nu + D ) / 2 }$ , with $r ^ { 2 }$ defined as in the bounded-support case, we can express the $( 2 - q )$ -escort as a reweighting of the base distribution π:

$$
\widetilde { \pi } ( \pmb { \theta } ) ^ { 2 - q } = \frac { \pi ( \pmb { \theta } ) ^ { 2 - q } } { \int \pi ( \pmb { \theta } ) ^ { 2 - q } d \pmb { \theta } } \propto \pi ( \pmb { \theta } ) \left( 1 + \frac { r ^ { 2 } } { \nu } ( \pmb { \theta } ) \right) .\tag{102}
$$

For $\nu > 2 ,$

$$
\mathbb { E } _ { \pi _ { \lambda } } [ r ^ { 2 } ( \pmb { \theta } ) ] = \frac { D \nu } { \nu - 2 } \Rightarrow \mathbb { E } _ { \pi _ { \lambda } } [ 1 + \frac { r ^ { 2 } } { \nu } ] = \frac { D + \nu - 2 } { \nu - 2 } ,
$$

and substituting into Eq. 100 yields $\operatorname { E q }$ . 101.

Thus, we obtain the Student-t Newton-like canonical parameter updates that are used in Alg. 2 by solving Eq. 99:

$$
\mathbf { S } _ { t + 1 } = ( 1 - \rho _ { t } ) \mathbf { S } _ { t } + \rho _ { t } \mathbb { E } _ { \tilde { \pi } _ { t } ^ { 2 - q } } \left[ \nabla _ { \pmb { \theta } } ^ { 2 } \bar { \ell } ( \pmb { \theta } ) \right] ,\tag{103}
$$

$$
\mathbf { m } _ { t + 1 } = \mathbf { m } _ { t } - \rho _ { t } \mathbf { S } _ { t + 1 } ^ { - 1 } \mathbb { E } _ { \boldsymbol { \pi } _ { t } } \left[ \nabla _ { \pmb { \theta } } \bar { \ell } ( \pmb { \theta } ) \right] .\tag{104}
$$

Implementation. In both the bounded-support and heavy-tailed cases we exploit escort distributions to mirror the familiar Gaussian stochastic gradient expressions in an elegant way, and to highlight the probabilistic structure that would otherwise be obscured by repeated integration by parts. For efficient sampling in $\mathrm { A l g . }$ 1 and Alg. 2, we express $\tilde { \pi } ^ { 2 - q }$ as a reweighting of $\pi _ { \lambda }$ . When $q > 1$ the $( 2 - q )$ -escort relatively upweights larger values of $r ( \pmb \theta )$ since typical samples dominate $\bar { \mathbb { E } } _ { \pi } [ \nabla \bar { \ell } ]$ and the mean update is mode-driven. The scale update in this case is more tail-aware since atypical large-radius samples influence the Hessian estimate. The combined effect is a conservative preconditioning because a larger effective curvature leads to smaller steps in m, but possibly higher variance in the curvature estimator. When $q < 1$ the $( 2 - q )$ )-escort concentrates on the interior of the support. Qualitatively, in this case, the mean is more sensitive to the bounded-support shape, while the scale update has lower variance, leading to a stable scale estimate, with bounded exploration due to the compact support.

We make additional adjustments to the generalized BLR $q { \mathrm { - V O N } }$ updates in Eqs. 103 and 104 and Eqs. 95 and 96, in order to easily implement them as optimizers in modern deep-learning frameworks. Our simplifications are analogous to Khan et al. [2018, App. D]’s and Shen et al. [2024]’s modifications of the BLR that lead to deep-learning-friendly VON and IVON algorithms. First, the loss function is reweighted with an inverse temperature $\zeta > 0$ and scaled by $1 / N$ , that is, we use $\begin{array} { r } { \big \zeta \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { i } ( { \pmb \theta } ) } \end{array}$ in Eq. 13. This reweighting allows us to use mini-batch stochastic gradients along with sampling to estimate the expectations (lines 2 of Alg. 1 and Alg. 2). Second, we use a diagonal $\mathbf { S } = \mathrm { d i a g } ( \mathbf { s } )$ and reparametrize it via ${ \mathbf s } = \zeta ( { \mathbf h } + \delta )$ (lines 7), where $\delta > 0$ is the $L _ { \mathrm { { 2 } ^ { - r e g u l a r i z e r } , } }$ $\begin{array} { r } { - \eta _ { 0 } ( { \pmb \theta } ) = \frac { \zeta \delta } { 2 } \| { \pmb \theta } \| ^ { 2 } } \end{array}$ in the generalized Bayesian objective of Eq. 13. The S-updates in Eqs. 95 and 103 are rewritten in terms of h in lines 5 of the algorithms, and separate learning rates $( 1 - \beta _ { 2 } )$ and $\alpha _ { t }$ are introduced in place of $\rho _ { t }$ . Similar to IVON [Shen et al., 2024], we add a gradient momentum in lines 4. This is not a direct consequence of the theory, but does not change the fixed-points of the methods and we found it improves results. The initial m is set to common initialization schemes, and $\mathbf { S } = \zeta ( h _ { 0 } + \delta ) \mathbf { I }$ where $h _ { 0 } > 0$ is a hyperparameter. To stabilize the algorithms, we ensure that the noise added to the parameters remains bounded by clipping the entries of s to a minimum $s _ { \operatorname* { m i n } } > 0$

## F.2 Mean-Field Variants

The constants $\kappa _ { t } ,$ , κ and $\gamma$ used in Alg. 1 and Alg. 2 depend on the dimensionality of the parameter θ. To further adapt the proposed $q { \mathrm { - V O N } }$ algorithms to high-dimensional problems, we also propose mean-field variants, where each parameter follows a one-dimensional $q { \mathrm { - } } \mathrm { G a u s s i a n } .$ . This corresponds to a factorized posterior, that is, $, \pi ( \pmb \theta ) = \pi ( \theta _ { 1 } ) \cdot \dots \cdot \pi ( \theta _ { D } )$ . Mean-field variants of Alg. 1 and $\mathrm { A l g } . 2$ are easily obtained when Ω in the generalized Bayesian variational problem Eq. 13 is replaced with the sum of one-dimensional functions on each parameter.

In practice, the only modification needed to Alg. 1 and Alg. 2 is to set $D = 1$ in the computation of the constants, and compute $\sigma ^ { 2 }$ in lines 8 individually for each s<sub>i</sub> rather than taking their product. The sampling step in lines 2 is also modified to sample from a one-dimensional q-Gaussian or Student-t distribution. In the remainder of the paper, we refer to the mean-field $q { \mathrm { - V O N } }$ variants as $q { \mathrm { - } } \mathrm { V O N - M F } .$

## G Experiments

## G.1 Bayesian logistic regression

We demonstrate how $q { \mathrm { - } } \mathsf { V O N - M F }$ with bounded support (Alg. 1, Sec. 4) can be used to approximate the posterior mode in Bayesian logistic regression on the UCI Credit Approval dataset (Fig. 2). Posterior marginals for each feature are computed with MCMC. We compare variational posterior approximations: mean-field Gaussian $( q = 1 )$ to bounded-support q-Gaussians with $q = - 2$ and $q = - 1 0$

## G.2 Image Classification with Neural Networks

We use q-VON (Alg. 1 and Alg. 2) and the mean-field variant q-VON-MF (described in Sec. 4) to train neural networks on the MNIST, CIFAR-10 and CIFAR-100 datasets. We compare with several baseline learning algorithms on test accuracy, test loss (NLL), expected calibration error Guo et al. [2017], Brier score and the AUROC metric introduced by Osawa et al. [2019]. The baselines are SGD (with momentum) applied to the standard empirical risk minimization objective to find the MAP, variational SGD (VSGD) which is SGD with Gaussian noise injection [Orvieto et al., 2022], and the Improved Variational Online Newton method (IVON) [Shen et al., 2024]. In all experiments, we closely follow the setup of Shen et al. [2024] and use their publicly available code. In particular, we use the same hyperparameters for $q { \mathrm { - V O N } }$ and q-VON-MF as IVON. IVON and VSGD target the standard Bayesian objective for Gaussian families with fixed isotropic variance and flexible diagonal variance respectively. Another baseline we consider is q-VSGD, which targets the generalized Bayesian objective but only learns the mean of the approximate posterior distribution. All algorithms have the same complexity and run-time, as each iteration is dominated by the cost of forward and backward propagation through the neural network. The results are in Table 2 for MNIST and Table 1 for CIFAR-10 and CIFAR-100. Below we summarize our main findings.

q-VON consistently improves over q-VSGD and SGD. Both on MNIST and CIFAR-10, we see that q-VON consistently improves over q-VSGD and SGD (MAP). This confirms that learning the shape of the approximate posterior rather than using a fixed shape improves both accuracy and calibration performance, and that the method is superior to pure noise injection approaches [Orvieto et al., 2022].

q-VON can exceed IVON’s performance. On the MNIST dataset, the heavy-tailed variant of q-VON $( q > 1 )$ gives better generalization performance than IVON, whereas the compact support variant $( q < 1 )$ gives better calibration. On CIFAR-10 and CIFAR-100, $q { \mathrm { - } } \mathbf { V O N - M F }$ reaches competitive performance to IVON, without tuning any of its hyperparameters.

Posterior sampling helps on the CIFAR datasets. For q-VON and IVON which estimate a distribution over weights, using the posterior predictive distribution generally boosts the performance.

Overall, the experiments demonstrate that the proposed q-VON algorithms and the mean field variants easily scale to large neural networks. The methods significantly improve over MAP estimation and reach competitive performance with the recent state-of-the-art method IVON [Shen et al., 2024]. We used the same setup and hyperparameters from [Shen et al., 2024] (which were tuned for Gaussian posteriors). Using IVON-tuned hyperparameters, q-VON is already competitive; dedicated tuning remains future work.

Tables 1 and 2 report results on deep learning with the proposed q-VON algorithms, as described in Sec. 4, App. F. q-VON $( q < 1 )$ denotes Alg. 1 with bounded-support Gaussian posterior and $q = 0 . 6$ $q \mathrm { - } \mathsf { V O N } \left( q > 1 \right)$ denotes Alg. 2 with Student-t posterior with $\nu = 5 \cdot 1 0 ^ { 5 }$ . q-VON-MF refers to the mean-field variant; each parameter has its own q-Gaussian $( q = 0 . 6$ for $q < 1$ and $\nu = 5$ for $q > 1 )$

Table 1: Results on convolutional neural networks. We evaluate various $q { \mathrm { - V O N } }$ variants using the same hyperparameters used in Shen et al. [2024] for IVON. Even without tuning, the results are comparable to IVON, with the heavy-tailed q-VON-MF reaching slightly higher AUROC scores.
<table><tr><td>Task</td><td>Method</td><td>Acc. (↑)</td><td>NLL (4)</td><td>ECE (↓)</td><td>Brier (↓)</td><td>AUROC (↑)</td></tr><tr><td rowspan="6">CIFAR-10 ResNet-20 270k params</td><td>SGD (MAP)</td><td> $9 1 . 2 \pm 0 . 1$ </td><td> $0 . 2 9 2 \pm 0 . 0 0 1$ </td><td> $3 . 7 \pm 0 . 1$ </td><td> $0 . 1 3 3 \pm 0 . 0 0 1$ </td><td> $9 1 . 9 \pm 0 . 1$ </td></tr><tr><td>VSGD</td><td> $9 1 . 9 \pm 0 . 0$ </td><td> $0 . 3 1 0 \pm 0 . 0 0 5$ </td><td> $4 . 5 \pm 0 . 1$ </td><td> $0 . 1 2 7 \pm 0 . 0 0 1$ </td><td> $9 2 . 1 \pm 0 . 1$ </td></tr><tr><td>IVON + sampling</td><td> $9 2 . 4 \pm 0 . 1$   ${ \bf 9 2 . 6 \pm 0 . 1 }$ </td><td> $0 . 2 5 9 \pm 0 . 0 0 6$   ${ \bf 0 . 2 2 1 } \pm 0 . 0 0 3$ </td><td> $3 . 5 \pm 0 . 2$   ${ \bf 0 . 8 \pm 0 . 1 }$ </td><td> $0 . 1 1 7 \pm 0 . 0 0 2$   ${ \bf 0 . 1 0 9 } \pm 0 . 0 0 2$ </td><td> $9 2 . 4 \pm 0 . 0$   $9 2 . 6 \pm 0 . 1$ </td></tr><tr><td> ${ \mathrm { q } } { \mathrm { - } } { \mathrm { V S G D } } \left( q < 1 \right)$ </td><td> $9 1 . 9 \pm 0 . 0$ </td><td> $0 . 3 0 9 \pm 0 . 0 0 8$ </td><td> $4 . 5 \pm 0 . 1$ </td><td> $0 . 1 2 8 \pm 0 . 0 0 2$ </td><td></td></tr><tr><td> ${ \mathsf { q } } { - } { \mathsf { V O N } } \left( q < 1 \right)$ </td><td> $9 2 . 3 \pm 0 . 2$ </td><td></td><td></td><td></td><td> $9 2 . 0 \pm 0 . 1$ </td></tr><tr><td>+ sampling  ${ \bf q } { \bf - V O N - M F } \left( q > 1 \right)$ </td><td> $9 2 . 4 \pm 0 . 2$   $9 2 . 4 \pm 0 . 0$ </td><td> $0 . 2 7 1 \pm 0 . 0 0 8$   $0 . 2 4 3 \pm 0 . 0 0 6$ </td><td> $3 . 9 \pm 0 . 2$   $2 . 6 \pm 0 . 2$ </td><td> $0 . 1 1 9 \pm 0 . 0 0 3$   $0 . 1 1 3 \pm 0 . 0 0 3$ </td><td> $9 2 . 6 \pm 0 . 2$   $\mathbf { 9 } 2 . 7 \pm 0 . 2$ </td></tr><tr><td rowspan="6">CIFAR-100 PreResNet-110</td><td>+sampling</td><td> $9 2 . 5 \pm 0 . 0$ </td><td> $0 . 2 7 0 \pm 0 . 0 0 3$   $0 . 2 2 9 \pm 0 . 0 0 2$ </td><td> $3 . 8 \pm 0 . 1$   $1 . 5 \pm 0 . 1$ </td><td> $0 . 1 1 8 \pm 0 . 0 0 1$   $0 . 1 1 0 \pm 0 . 0 0 1$ </td><td> $9 2 . 5 \pm 0 . 2$   $9 2 . 7 \pm 0 . 1 $ </td></tr><tr><td>SGD (MAP)</td><td> $7 3 . 0 \pm 0 . 2$ </td><td> $1 . 2 7 0 \pm 0 . 0 0 9$ </td><td> $1 4 . 1 \pm 0 . 1$ </td><td> $0 . 4 1 0 \pm 0 . 0 0 2$ </td><td> $8 6 . 1 \pm 0 . 1$ </td></tr><tr><td>VSGD</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IVON</td><td> $7 3 . 8 \pm 0 . 2$   $7 5 . 1 \pm 0 . 3$ </td><td> $1 . 3 1 9 \pm 0 . 0 0 9$ </td><td> $1 5 . 5 \pm 0 . 2$ </td><td> $0 . 4 1 0 \pm 0 . 0 0 3$ </td><td> $8 6 . 1 \pm 0 . 1$ </td></tr><tr><td>+ sampling</td><td> $7 5 . 7 \pm 0 . 2$ </td><td> $1 . 1 5 0 \pm 0 . 0 1 2$   $\mathbf { 0 . 8 8 0 \mathop { \pm } 0 . 0 0 6 }$ </td><td> $1 3 . 8 \pm 0 . 3$   $\mathbf { \ } 2 . \mathbf { 9 } \pm 0 . 2$ </td><td> $0 . 3 8 2 \pm 0 . 0 0 4$ </td><td> $8 6 . 8 \pm 0 . 2$ </td></tr><tr><td rowspan="4">4M params</td><td> ${ \mathrm { q } } { \mathrm { - } } { \mathrm { V S G D } } \left( q < 1 \right)$ </td><td></td><td></td><td></td><td> ${ \bf 0 . 3 3 6 \pm 0 . 0 0 3 }$ </td><td> $8 7 . 0 \pm 0 . 2$ </td></tr><tr><td> ${ \mathsf { q } } { - } { \mathsf { V O N } } \left( q < 1 \right)$ </td><td> $7 3 . 3 \pm 0 . 1$   $7 4 . 6 \pm 0 . 2$ </td><td> $1 . 3 2 6 \pm 0 . 0 0 1$   $1 . 2 1 6 \pm 0 . 0 0 8$ </td><td> $1 5 . 6 \pm 0 . 1$   $1 4 . 0 \pm 0 . 1$ </td><td> $0 . 4 1 6 \pm 0 . 0 0 2$ </td><td> $8 6 . 1 \pm 0 . 2$ </td></tr><tr><td>+ sampling</td><td> $7 4 . 7 \pm 0 . 2$ </td><td> $1 . 1 8 9 \pm 0 . 0 0 8$ </td><td> $1 3 . 3 \pm 0 . 1$ </td><td> $0 . 3 9 1 \pm 0 . 0 0 2$ </td><td> $8 6 . 3 \pm 0 . 1$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } \left( q > 1 \right)$   $7 4 . 9 \pm 0 . 1$   $7 5 . 4 \pm 0 . 1$ </td><td> $1 . 2 0 4 \pm 0 . 0 0 8$ </td><td></td><td> $1 4 . 4 \pm 0 . 1$ </td><td> $0 . 3 8 7 \pm 0 . 0 0 2$   $0 . 3 8 9 \pm 0 . 0 0 2$ </td><td> $8 6 . 4 \pm 0 . 1$   $8 6 . 7 \pm 0 . 1$ </td></tr></table>

Table 2: Additional results on MNIST. Heavy-tailed posteriors can improve generalization performance, whereas truncated ones can improve calibration.
<table><tr><td>Task</td><td>Method</td><td>Acc. (↑)</td><td>Test Loss (↓)</td><td>ECE (↓)</td><td>Brier (↓)</td><td>AUROC (↑)</td></tr><tr><td rowspan="10">MNIST MLP 550k params</td><td>SGD (MAP) VSGD</td><td> $9 8 . 6 9 \pm 0 . 0 1$   $9 8 . 7 5 \pm 0 . 0 3$ </td><td> $0 . 0 4 3 \pm 0 . 0 0 0$   $0 . 0 4 8 \pm 0 . 0 0 0$ </td><td> $0 . 3 7 \pm 0 . 0 2$   $0 . 6 0 \pm 0 . 0 2$ </td><td> $0 . 0 2 0 \pm 0 . 0 0 0$   $0 . 0 2 0 \pm 0 . 0 0 0$ </td><td> $9 7 . 8 7 \pm 0 . 0 7$ </td></tr><tr><td>IVON</td><td> $9 8 . 7 4 \pm 0 . 0 2$ </td><td> $0 . 0 4 8 \pm 0 . 0 0 1$ </td><td>0.60 ± 0.02</td><td> $0 . 0 2 0 \pm 0 . 0 0 0$ </td><td> $9 7 . 6 8 \pm 0 . 0 5$ </td></tr><tr><td>+ sampling</td><td> $9 8 . 7 2 \pm 0 . 0 1$ </td><td> $0 . 0 5 0 \pm 0 . 0 0 0$ </td><td>1.53 ± 0.02</td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 5 2 \pm 0 . 1 0$   $9 7 . 8 8 \pm 0 . 0 5$ </td></tr><tr><td> ${ \mathrm { q } } { \mathrm { - } } { \mathrm { V S G D } } \left( q < 1 \right)$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> ${ \sf q } { - } \mathrm { V O N } \left( q < 1 \right)$ </td><td> $9 8 . 6 2 \pm 0 . 0 2$  98.69 ± 0.02 0.042 ± 0.000</td><td> $0 . 0 4 5 \pm 0 . 0 0 1$ </td><td>0.46 ± 0.01</td><td>0.021 ± 0.000</td><td> $9 7 . 9 4 \pm 0 . 0 6$ </td></tr><tr><td>+ sampling</td><td></td><td>98.69 ± 0.02 0.042 ± 0.000</td><td>0.40 ± 0.03</td><td>3 0.020 ± 0.000</td><td> $9 8 . 0 4 \pm 0 . 0 5$ </td></tr><tr><td></td><td></td><td></td><td></td><td>) 0.40 ± 0.04 0.020 ± 0.000</td><td> $9 8 . 0 4 \pm 0 . 0 5$ </td></tr><tr><td> ${ \mathrm { q } } { \mathrm { - } } { \mathrm { V S G D } } \left( q > 1 \right)$ </td><td>98.59 ± 0.01</td><td>1 0.045 ± 0.000</td><td></td><td>) 0.48 ± 0.02 0.022 ± 0.000</td><td> $9 7 . 9 7 \pm 0 . 0 5$ </td></tr><tr><td> ${ \mathsf { q } } { - } { \mathsf { V O N } } \left( q > 1 \right)$ </td><td>98.81 ± 0.03</td><td>3 0.043 ± 0.000</td><td></td><td>) 0.60 ± 0.01 0.019 ± 0.000</td><td>97.84 ± 0.11</td></tr><tr><td>+ sampling</td><td> $9 8 . 7 6 \pm 0 . 0 2$ </td><td>0.042 ± 0.000</td><td>0.85 ± 0.01</td><td>0.020 ± 0.000</td><td> ${ \bf 9 8 . 0 5 \pm 0 . 0 9 }$ </td></tr><tr><td>q-VON-MF (q &lt; 1)</td><td>98.66 ± 0.03</td><td>0.052 ± 0.001</td><td></td><td>0.60 ± 0.02 0.021 ± 0.000</td><td></td></tr><tr><td>+ sampling</td><td></td><td></td><td></td><td></td><td> $9 7 . 1 9 \pm 0 . 1 6$ </td></tr><tr><td></td><td>98.62 ± 0.02</td><td>0.053 ± 0.001</td><td>1.60 ± 0.02</td><td>0.024 ± 0.000</td><td> $9 7 . 8 2 \pm 0 . 0 4$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } \left( q > 1 \right)$ </td><td>98.76 ± 0.02</td><td> $0 . 0 4 9 \pm 0 . 0 0 1$ </td><td></td><td>0.59 ± 0.02 0.020 ± 0.000</td><td> $9 7 . 1 4 \pm 0 . 0 7$ </td></tr><tr><td>+sampling</td><td> $9 8 . 6 7 \pm 0 . 0 1$ </td><td> $0 . 0 5 4 \pm 0 . 0 0 1$ </td><td></td><td>1.90 ± 0.02 0.023 ± 0.000</td><td> $9 7 . 8 8 \pm 0 . 0 5$ </td></tr></table>

## G.3 Empirical Verification of Convergence

A convergence analysis of BLR/VON-type methods in the generality considered here is still largely open. Existing results cover narrower settings: Wu and Gardner [2024] prove non-asymptotic rates for stochastic natural gradient (NG) VI in conjugate exponential family models and show that for non-conjugate log-concave likelihoods, the negative ELBO need not be convex in the expectationparameter coordinates used by canonical NGVI. Guilmeau et al. [2025] instead study Bregmanproximal algorithms for regularized Rényi-divergence minimization over exponential families, with convergence guarantees based on relative smoothness with respect to the log-partition geometry. These assumptions are different from ours: our geometry is induced by a density-space Fenchel-Young discrepancy, which defines the generalized maxent family and the corresponding BLR update. A convergence theorem for the Fenchel-Young BLR would be a substantial project in its own right and is outside the scope of this paper. Instead, we provide empirical evidence that the implemented generalized updates move toward lower values of their corresponding variational objectives.

We use 2D Bayesian logistic regression as a minimal nontrivial diagnostic, with bounded-support, Gaussian, and Student-t variants. The plots in Fig. 3 show slices of the optimizer-specific generalized objective over one mean-dispersion coordinate, with the remaining coordinate fixed at its final value. Each panel recomputes the objective associated with that posterior family and parametrization, because the Fenchel-Young discrepancy, sampling law, and scale-precision map change with q and with the full versus mean-field construction. All trajectories move toward low-objective regions and stabilize near the minimum.

2D logistic-regression objective slices and trajectories in (m<sub>1</sub>, v<sub>1</sub>)  
![](images/762bb23e22ab697a4da2024db86379e02d636911bdc67f872ded13e33c6d08ac.jpg)  
Figure 3: Optimizer-specific convergence on 2D Bayesian logistic regression. Each panel shows the objective associated with an optimizer and posterior family. The plots show slices of the optimizerspecific generalized objective over one mean-dispersion coordinate, with the remaining coordinate fixed at its final value. The trajectories consistently approach low-objective regions for boundedsupport, Gaussian, and Student-t variants, for both full and mean-field implementations. The same $\nu = 5$ in both Student-t panels corresponds to different $q \mathrm { ' s }$ because of the dependence on D.

## H Varying q

Different choices of q change the variational objective, the scale-precision map, and the stochastic Hessian estimator, so iteration counts across q’s are not directly comparable. Nevertheless, $q { \mathrm { - V O N } }$ retains the fast VON-style optimizer structure: a Newton-like mean update preconditioned by a learned diagonal precision, implemented with minibatch stochastic gradients, momentum, and Hessian-diagonal estimates rather than explicit second-order computations [Khan et al., 2018, Osawa et al., 2019, Shen et al., 2024]. Since $q { \mathrm { - V O N } }$ changes only the sampling law, the precision constants, and the corresponding diagonal Stein estimator, we expect similar fast behavior.

## H.1 1D Logistic Regression

In Fig. 4 we use a synthetic dataset of size $N = 2 5 6$ , with covariates $x _ { i } \sim \mathcal { N } ( 0 , 1 )$ , labels $y _ { i } \sim$ Bernoull $\mathrm { i } ( \sigma ( 2 x _ { i } ) )$ , and Gaussian prior w $\sim \mathcal { N } ( 0 , 0 . 1 )$ ). For each $q , q { \mathrm { - V O N } }$ is run for 3000 steps and repeated over 5 seeds. Points show the average posterior mean or variance, and error bars show 95% confidence intervals. Horizontal reference lines show the exact Bayesian posterior quantities for the ordinary Gaussian-prior model. Exact references are approximated by evaluating the negative log-posterior $\ell ( w ) + w ^ { \dot { 2 } } / 0 . 2$ on a grid.

![](images/9817aa7592f63a936a9be32a44a25de4aacc86f8f4cc904f0a07f6451e8f3ccb.jpg)

![](images/3c554dd1de61edf9914acd8f655d1a3498dc8728875f7e2e6eff18b664760c57.jpg)  
Figure 4: 1D Bayesian logistic regression q-sweep with q-VON.

## H.2 MNIST

In Table 3 we report the results of varying q on MNIST training with $q { \mathrm { - } } \mathsf { V O N - M F } ,$ again using the hyperparameters obtained from previous tuning with IVON.

Table 3: MNIST q-sweep for q-VON-MF. Results are mean $\pm \ : \mathrm { s . e . }$ over 5 seeds.
<table><tr><td>Task</td><td>Method</td><td>Acc. (↑)</td><td> $\mathrm { { T e s t L o s s } \left( \downarrow \right) }$ </td><td> $\mathrm { E C E } \left( \downarrow \right)$ </td><td>Brier (↓)</td><td>AUROC (↑)</td></tr><tr><td rowspan="15"></td><td> ${ \bf q } { \bf - V } { \bf O N - M F } \left( q = - 1 0 \right)$ </td><td> $8 1 . 4 2 \pm 1 5 . 7 7$ </td><td> $0 . 5 1 4 \pm 0 . 4 1 2$ </td><td> $1 . 3 5 \pm 0 . 5 9$ </td><td> $0 . 2 0 8 \pm 0 . 1 6 5$ </td><td> $8 8 . 5 5 \pm 6 . 7 9$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } ( q = - 8 )$ </td><td> $9 7 . 6 7 \pm 0 . 0 3$ </td><td> $0 . 0 8 6 \pm 0 . 0 0 1$ </td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $0 . 0 3 6 \pm 0 . 0 0 0$ </td><td> $9 6 . 2 6 \pm 0 . 1 8$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } ( q = - 6 )$ </td><td> $9 8 . 1 5 \pm 0 . 0 8$ </td><td> $0 . 0 7 0 \pm 0 . 0 0 2$ </td><td> $0 . 8 6 \pm 0 . 0 4$ </td><td> $0 . 0 2 9 \pm 0 . 0 0 1$ </td><td> $9 6 . 7 9 \pm 0 . 1 2$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } ( q = - 4 )$ </td><td> $9 8 . 6 6 \pm 0 . 0 3$ </td><td> $0 . 0 5 3 \pm 0 . 0 0 1$ </td><td> $0 . 7 2 \pm 0 . 0 1$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 6 7 \pm 0 . 1 2$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } ( q = - 2 )$ </td><td> ${ \bf 9 8 . 7 5 \pm 0 . 0 1 }$ </td><td> $0 . 0 4 8 \pm 0 . 0 0 1$ </td><td> $0 . 6 1 \pm 0 . 0 3$ </td><td> ${ \bf 0 . 0 2 0 } \pm 0 . 0 0 0$ </td><td> $9 7 . 5 2 \pm 0 . 0 5$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } ( q = - 1 )$ </td><td> $9 8 . 6 6 \pm 0 . 0 2$ </td><td> $0 . 0 4 7 \pm 0 . 0 0 0$ </td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 3 \pm 0 . 0 7$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } \left( q = 0 \right)$ </td><td> $9 8 . 6 0 \pm 0 . 0 1$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 2$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 8 \pm 0 . 0 5$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } \left( q = 0 . 4 \right)$ </td><td> $9 8 . 5 9 \pm 0 . 0 3$ </td><td> $0 . 0 4 7 \pm 0 . 0 0 1$ </td><td> $0 . 5 1 \pm 0 . 0 2$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 8 8 \pm 0 . 0 5$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } \left( q = 0 . 6 \right)$ </td><td> $9 8 . 5 6 \pm 0 . 0 2$ </td><td> $0 . 0 4 7 \pm 0 . 0 0 1$ </td><td> $0 . 5 0 \pm 0 . 0 2$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 1 \pm 0 . 0 3$ </td></tr><tr><td> ${ \bf q } { \bf - V O N - M F } ( q = 0 . 8 )$ </td><td> $9 8 . 5 7 \pm 0 . 0 2$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 4 8 \pm 0 . 0 2$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 1 \pm 0 . 0 5$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 0 . 9 5 \right)$ </td><td> $9 8 . 5 5 \pm 0 . 0 4$   $0 . 0 4 7 \pm 0 . 0 0 1$ </td><td></td><td> $0 . 4 8 \pm 0 . 0 3$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 8 \pm 0 . 0 6$ </td></tr><tr><td>MNIST MLP</td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 1 \right)$ </td><td> $9 8 . 5 7 \pm 0 . 0 1$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 1$ </td><td> ${ \bf 0 . 4 5 \pm 0 . 0 2 }$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 2 \pm 0 . 0 4$ </td></tr><tr><td>550k params</td><td> ${ \bf q } { \bf - V } { \bf O N - M F } \left( q = 1 . 0 0 0 0 0 4 \right)$ </td><td> $9 8 . 5 3 \pm 0 . 0 2$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 5 0 \pm 0 . 0 2$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 3 \pm 0 . 0 4$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V } { \bf O N - M F } \left( q = 1 . 0 2 \right)$ </td><td> $9 8 . 5 5 \pm 0 . 0 3$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 5 \pm 0 . 0 8$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 1 . 0 5 \right)$ </td><td> $9 8 . 5 3 \pm 0 . 0 1$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 4 9 \pm 0 . 0 1$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td></td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 1 . 1 \right)$ </td><td> $9 8 . 5 6 \pm 0 . 0 1$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 5 0 \pm 0 . 0 2$ </td><td></td><td> $9 8 . 0 5 \pm 0 . 0 7$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 1 . 2 \right)$ </td><td> $9 8 . 5 3 \pm 0 . 0 2$ </td><td></td><td></td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 3 \pm 0 . 0 3$ </td></tr><tr><td></td><td></td><td></td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 4 8 \pm 0 . 0 3$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 3 \pm 0 . 0 3$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 4 / 3 \right)$ </td><td> $9 8 . 5 1 \pm 0 . 0 1$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 4 7 \pm 0 . 0 2$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 5 \pm 0 . 0 4$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 1 . 5 \right)$ </td><td> $9 8 . 5 4 \pm 0 . 0 1$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 3$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 0 \pm 0 . 0 3$ </td></tr><tr><td></td><td> ${ \bf q } { - } \mathrm { V O N - M F } \left( q = 1 . 5 7 1 4 3 \right)$ </td><td> $9 8 . 5 5 \pm 0 . 0 2$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 9 \pm 0 . 0 4$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 1 . 6 \right)$ </td><td> $9 8 . 5 5 \pm 0 . 0 3$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 4 8 \pm 0 . 0 3$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 2 \pm 0 . 0 7$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V } { \bf O N - M F } \left( q = 1 . 6 2 5 \right)$ </td><td> $9 8 . 5 4 \pm 0 . 0 4$ </td><td> $0 . 0 4 7 \pm 0 . 0 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 4$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 0 1 \pm 0 . 0 6$ </td></tr><tr><td></td><td> ${ \bf q } { - } { \bf V } \mathrm { O N - M F } \left( q = 1 . 6 4 5 1 6 \right)$ </td><td> $9 8 . 5 5 \pm 0 . 0 3$ </td><td> $\mathbf { 0 . 0 4 6 \pm 0 . 0 0 0 }$ </td><td> $0 . 4 8 \pm 0 . 0 4$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> ${ \bf 9 8 . 0 7 \pm 0 . 0 6 }$ </td></tr><tr><td></td><td> ${ \bf q } { \bf - V O N - M F } \left( q = 1 . 6 5 5 7 4 \right)$ </td><td> $9 8 . 5 5 \pm 0 . 0 2$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 1$ </td><td> $0 . 4 6 \pm 0 . 0 3$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 3 \pm 0 . 0 5$ </td></tr><tr><td></td><td> ${ \bf q } { - } \mathrm { V O N - M F } \left( q = 1 . 6 6 4 4 5 \right)$ </td><td> $9 8 . 5 7 \pm 0 . 0 3$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 0$ </td><td> $0 . 4 7 \pm 0 . 0 3$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 0$ </td><td> $9 7 . 9 9 \pm 0 . 0 7$ </td></tr></table>