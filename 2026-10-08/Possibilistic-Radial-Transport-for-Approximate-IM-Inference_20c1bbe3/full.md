# Possibilistic Radial Transport for Approximate IM Inference

Jungeum Kim<sup>∗</sup> and Percy S. Zhai<sup>†</sup>

## Abstract

Probing the hypothesis space after seeing the data remains valid under possibilistic inferential models (IMs), provided the significance level stays fixed. The price is computation, as each plausibility is a supremum of the possibility contour over the hypothesis, and the contour itself is approximated at each queried parameter value. We propose a possibilistic radial transport, which hides the contour value of a parameter in the radius of its source point. When a transport that maximizes within-shell entropy is picked, sampling parameters covering a confidence cut becomes a matter of truncating the radius. We provide a deep learning algorithm that enforces the contour depth condition while maximizing the entropy within each shell. Our amortization makes coverage and power assessments of the learned approximation practical as well as predictive check of new datasets. We also use the sampler to construct a Bel–Pl spectrum for comparing and selecting interpretable hypotheses that satisfy a prescribed Bel–Pl decision criterion. In simulations the learned contours match or improve on ellipsoidal approximations to the cuts, while the coverage and power track the exact reference. Finally, we probe hypotheses about ovarian aging using synthetic AMH records, asking for each woman how many more years her median AMH level will remain above a specified reference value.

## 1 Introduction

After observing data, what can we say about uncertainty in an unknown parameter? Inferential models (IMs) unfold this story of uncertainty through belief and plausibility (Martin and Liu, 2015). Since the story is told after seeing the particular data in hand, it reads much like a Bayesian posterior, yet it requires no prior distribution. At the same time, its reliability is calibrated across repeated sampling, so that the resulting inference has frequentist validity. In this work, we focus on possibilistic IMs (Martin, 2026c), which build belief and plausibility on a single, intuitive possibility contour.

The contour is defined by $\pi _ { z } ( \theta ) = \mathsf { P } _ { \theta } \{ R ( Z , \theta ) \leq R ( z , \theta ) \}$ , the probability that a fresh dataset from θ ranks no higher than the observed one, where R is a ranking function such as the relative likelihood. It is typically estimated by standard Monte Carlo (MC), sampling many Z from $\mathsf { P } _ { \theta }$ . When the rank is costly to evaluate (for example, when it requires solving an MLE problem), so are the contour, plausibility, and belief. Our goal is to build a task-specific junior AI that can be reused on new datasets to rapidly approximate the IM contour and the resulting hypothesis tests.

While not easy to compute, the belief and plausibility (Bel–Pl) in IMs have an important property that makes them worth the efort. Denote them by Pl<sub>z</sub> and Bel<sub>z</sub>. A hypothesis H is rejected when Pl $_ { z } ( H ) = \operatorname* { s u p } _ { \theta \in H } \pi _ { z } ( \theta )$ is suficiently small and accepted when Bel $( H ) = 1 -$ $\mathrm { P l } _ { z } ( H ^ { c } )$ is suficiently large. This rule encodes that acceptance requires the corresponding $( 1 - \alpha )$ confidence region to lie inside H, while rejection requires it to be disjoint from H. The consequence is uniform validity (Cella and Martin, 2023, Corollary 2); if we fix α in advance and use the same valid contour, validity is preserved as we combine or change hypotheses about the same parameter. That is, we can even adjust a hypothesis after seeing the data, or choose a modified hypothesis after seeing the result of a previous test!

We find this inferential feature very appealing because it can ofer statistical frustration prevention for people who collect and analyze data. In a typical hypothesis test, we specify a hypothesis and α before observing the data, then perform the test once the data are available. Imagine spending time and money collecting data, only to find that the hypothesis does not receive a clear answer. Altering the data or the threshold to obtain a statistical rejection is common enough to have earned unflattering names such as p-hacking, N-hacking, and α-hacking (Simmons et al., 2011; Reinagel, 2023; Koning, 2023). On the other hand, under the uniform validity described above, we can keep α fixed and continue asking dif ferent questions by changing the hypothesis. Then, fast contour computation would make this flexible hypothesis probing much easier by helping to check various hypotheses in a reasonable amount of time.

Our pursuit of fast IM inference builds on the work of Martin (2026a), whose stitched contour approximation uses parameters drawn from an inner credal distribution. This replaces expensive data-space Monte Carlo, which requires normalization for each simulated dataset Z, with parameter-space sampling at the observed z. In its main implementation, however, the sampler is constructed by ellipsoidal approximations to contour level cuts, which we refer to as Martin’s IMMC method. This approximation is in line with a possibilistic Bernstein–von Mises theorem (Martin and Williams, 2025). For small $n _ { \mathrm { : } }$ , the α-cuts need not be close to ellipsoids. Proposition 1 illustrates how this departure from ellipsoidal geometry can afect the resulting plausibility assessments in a fixed parameterization. The plausibility of a hypothesis that the original IM could reject can become arbitrarily close to 1 under this ellipsoidal approximation, so a rejection that the original IM delivers is lost.

Here, we take a diferent route to pursue an eficient sampler that provides parameter samples for Martin’s stitched contour estimation. Our junior AI is a flexible deep neural sampler that transports the core of the source distribution into a region with high contour values. We seek a sampler within a class of transports that send a source point u to a parameter whose contour value is the source depth, $\pi _ { z } ( T _ { z } ( u ) ) = 1 - \| u \|$ . We establish the completeness of this class, that its pushforward distributions are exactly the inner credal distributions. With this “radial transport” formulation, the contour cuts now become an outcome of learning the sampler rather than a prerequisite for constructing it. Among the many ways of distributing probability within each contour shell, we choose the one that maximizes within-shell entropy, yielding the shell-uniform maximum-entropy inner credal (MIC) distribution. When the contour cuts are ellipsoidal and the same shell reference measures are used, this coincides with the inner probabilistic approximation used by Martin’s IMMC.

Another central word of our work is “amortized”. Rather than learning a sampler for a single fixed $z ,$ we seek an amortized transport that, once trained for a chosen likelihood model, can accommodate many datasets. We demonstrate this reuse in our ovarian aging application using synthetic AMH records, where the individuals are analyzed under a common likelihood model. The same trained sampler is reused across individuals to probe hypotheses about how long each person’s median AMH level remains above a specified reference value. Amortization also enables validation through extensive simulation-based coverage and power assessments before the sampler is used in practice. When a new dataset arrives, Bayesian-like predictive checks can provide a preliminary screen before using the learned sampler for inference.

With fast hypothesis probing enabled, the next question is which of the interpretable hypotheses supported by the same contour one should report. As a first step, we draw a Bel–Pl spectrum that shows, at each contour depth, the fraction of the shell occupied by a hypothesis. The spectrum gives us a glimpse of the hypothesis geometry hidden behind the Bel–Pl endpoints, and indicating the Bel or Pl location on the same plot adds extra informativeness. We use this information to compare and select interpretable hypotheses that satisfy a prescribed Bel–Pl decision criterion. Among hypotheses meeting the acceptance criterion, we favor those with higher mean contour depth while limiting the gap between their inner credal probability and belief.

Deep neural networks have been used to learn samplers for statistical inference. A prominent application is Bayesian posterior sampling, where neural samplers ofer an alternative to MCMC such as Wang and Ročková (2026) and Kim et al. (2025) to name a few. In generalized fiducial inference, Li and Hannig (2020) proposed a fiducial autoencoder combined with a rejection step analogous to approximate Bayesian computation. To our knowledge, ours is the first deep neural sampler for IM inference, building on the connection between IM contour approximation and parameter-space sampling established by Martin (2026a). On the inferential side, Grünwald (2024) and Koning (2023) use e-values to study data-dependent significance levels and post-hoc decision-making. Here, we keep α fixed and probe diferent hypotheses after observing the data.

In the current version of the manuscript, we assess the learned approximation empirically without establishing theoretical error bounds. Our experiments show comparable or improved contour accuracy relative to IMMC, rapid inference after training, coverage near the nominal level, and power and hypothesis selection consistent with the reference results. In our hypothesis-selection example, MIC mitigates belief underestimation and recovers the reference selection.

The paper is structured as follows. In Section 2, we briefly review possibilistic IMs to provide the context and a high-level description of our approach. The possibilistic radial transport is introduced in Section 3, together with our algorithm for training both nonamortized and amortized samplers. Section 4 discusses the Bel–Pl spectrum and hypothesis selection criteria. The numerical study in Section 5 demonstrates the faithfulness and eficiency of our methods in an ovarian aging analysis. The paper concludes with Section 6.

## 2 Possibilistic Inferential Model

In 1930, Fisher introduced the fiducial argument with an intriguing ambition of obtaining a distribution expressing uncertainty about θ without specifying a prior (Fisher, 1930). Many statistical minds have since pursued this ambition in diferent forms, including Dempster– Shafer theory (Shafer, 1976), generalized fiducial inference (Hannig et al., 2016), and inferential models (IM, Martin et al., 2010) to name a few. Here we focus on the possibilistic IMs, which integrate an imprecise probability over θ with frequentist validity (Martin, 2026c). Its core inferential framework is Bel-Pl through the imprecise probability.

The Bel–Pl representation used here has its roots in Dempster’s (1967) construction of lower and upper probabilities through multivalued mappings, which Shafer (1976) developed into a general theory of evidence. This construction assigns probability to sets of possible parameter values, allowing uncertainty to be expressed without selecting a single probability distribution on the parameter space. When these random sets are nested, the resulting belief function is consonant, and its plausibility is a possibility measure determined by a contour. The normalized likelihood was studied as such a contour (Shafer, 1982; Wasserman, 1990), but this representation alone did not guarantee frequentist calibration. In possibilistic IMs, frequentist validity is obtained by calibrating the (likelihood based) ranking function under the sampling model, which shares a similar idea with conformal prediction (Martin, 2026c).

## 2.1 The contour and Bel-Pl Inference

Let $( { \mathcal { Z } } , { \mathcal { Z } } )$ be a measurable observation space, and let $\{ P _ { \theta } : \theta \in \mathbb { T } \}$ be a statistical model, where each $P _ { \theta }$ is a probability measure on $( { \mathcal { Z } } , { \mathcal { Z } } )$ . Let $P _ { \theta } ^ { ( n ) }$ denote the distribution for n i.i.d. samples $Z _ { n }$ under parameter θ. The possibilistic IM contour is defined by

$$
\pi _ { z _ { n } } ( \theta ) : = P _ { \theta } ^ { ( n ) } \left( R ( Z _ { n } , \theta ) \leq R ( z _ { n } , \theta ) \right) , \ \theta \in \mathbb { T } ,\tag{2.1}
$$

where we consider the standard likelihood score $R ( z _ { n } , \theta ) = L _ { z _ { n } } ( \theta ) / \mathrm { s u p } _ { \nu } L _ { z _ { n } } ( \nu )$ as the rank. Intuitively, if $z _ { n }$ were generated from $P _ { \theta } ^ { ( n ) }$ , its rank among samples from $P _ { \theta } ^ { ( n ) }$ would be indistinguishable, and hence approximately uniform. In contrast, if $z _ { n }$ is unlikely under $P _ { \theta } ^ { ( n ) } , R ( z _ { n } , \theta )$ may become distinctively small. This uniformity gives the frequentist validity

$$
\operatorname* { s u p } _ { \theta \in \mathbb { T } } P _ { \theta } ^ { ( n ) } ( \pi _ { Z _ { n } } ( \theta ) \leq \alpha ) \leq \alpha , \quad \mathrm { f o r ~ a l l ~ } \alpha \in [ 0 , 1 ] .\tag{2.2}
$$

Then we can construct $1 0 0 ( 1 - \alpha ) \%$ confidence sets given by the α-cuts of the contour defined as $C _ { \alpha } ( z _ { n } ) = \{ \theta \in \mathbb { T } : \pi _ { z _ { n } } ( \theta ) \geq \alpha \}$ . For hypothesis test with $\alpha \in [ 0 , 0 . 5 ]$ , the rule is

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathrm { B e l } _ { z } ( H ) \geq 1 - \alpha , } & { \mathrm { a c c e p t ~ } H , } \\ { \mathrm { P l } _ { z } ( H ) \leq \alpha , } & { \mathrm { r e j e c t ~ } H , } \\ { \mathrm { o t h e r w i s e } , } & { \mathrm { w i t h h o l d ~ j u d g m e n t } . } \end{array} \right. } \end{array}\tag{2.3}
$$

Here, a low plausibility $\operatorname { P l } _ { z } ( H ) : = \operatorname* { s u p } _ { \theta \in H } \pi _ { z } ( \theta )$ indicates that all the points in H are not very compatible with the data, whereas a high belief, Bel $( H ) : = 1 - \operatorname* { s u p } _ { \theta \in H ^ { c } } \pi _ { z } ( \theta )$ indicates that H includes all the high compatibility parameter candidates. Importantly, because we must have max $\{ P l _ { z } ( H ) , P l _ { z } ( H ^ { c } ) \} = 1$ , we always have either $P l _ { z } ( H ) = 1$ or $B e l ( H ) = 0$ . Since then $P l _ { z } ( H ) \ge B e l ( H )$ , we know that H cannot be accepted if $P l _ { z } ( H ) \neq 1$ and likewise H cannot be rejected when $B e l ( H ) \neq 0$

## 2.2 Stitched Contour Approximation

The true contour $\pi _ { z _ { n } } ( \theta )$ is dificult to evaluate, and a standard approximation is a Monte Carlo (MC) based approximation $\begin{array} { r } { \widehat { \pi } _ { z _ { n } } ^ { \mathrm { M C } } ( \theta ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } \big ( R ( Z _ { n , j , \theta } , \theta ) \leq R ( z _ { n } , \theta ) \big ) } \end{array}$ , for $\theta \in \mathbb { T }$ (Martin, 2015). Then, constructing $C _ { \alpha }$ is particularly expensive as it involves computing $\widehat { \pi } _ { z _ { n } } ^ { \mathrm { M C } } ( \theta )$ for each $\theta$ (on a grid) in $\mathbb { T } ,$ , where for each $\theta$ extra sampling of $Z _ { n , j , \theta } \sim P _ { \theta } ^ { ( n ) }$ is conducted. Likewise, computing the supremum approximation for conducting the Pl/Bel test as in (2.3) is expensive.

Recent work of Martin (2026a) avoids repeating this data-space MC sampling by a parameter-space MC sampling reducing the computational cost significantly. Let $\mathcal { P } ( \mathbb { T } )$ denote the collection of probability measures on $( \mathbb { T } , B ( \mathbb { T } ) )$ , and denote a possibility measure $\Pi _ { z _ { n } } ( H ) = \operatorname* { s u p } _ { \theta \in H } \pi _ { z _ { n } } ( \theta )$ (same as $P l ( H ) )$ . He defines inner credal set as

$$
\mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z _ { n } } ) : = \{ Q \in \mathcal { P } ( \mathbb { T } ) : Q ( C _ { \alpha } ( z _ { n } ) ) = 1 - \alpha \mathrm { ~ f o r ~ a l l ~ } \alpha \in [ 0 , 1 ] \} .
$$

The definition implies that for $Q _ { z _ { n } } ^ { \mathrm { i n n } } \in \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z _ { n } } )$ , it is maximally consistent with the contour. Since $( \pi _ { z _ { n } } ) _ { \# } Q _ { z _ { n } } ^ { \mathrm { i n n } } = \mathrm { U n i f } ( 0 , 1 )$ , we emphasize that any $Q _ { z _ { n } } ^ { \mathrm { i n n } } \in \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z _ { n } } )$ has the calibration property

$$
Q _ { z _ { n } } ^ { \mathrm { i n n } } ( \pi _ { z _ { n } } ( \Theta ) \leq \alpha ) = \alpha , \ \alpha \in [ 0 , 1 ] .\tag{2.4}
$$

Although the validity in (2.2) and the calibration in (2.4) involve seemingly similar events, they refer to diferent sources of randomness. The former is a sampling-calibration property under $P _ { \theta } ^ { ( n ) }$ , with $\theta$ fixed and $Z _ { n }$ random; the latter is a depth-calibration property under $Q _ { z _ { n } } ^ { \mathrm { i n n } }$ , with $z _ { n }$ fixed and $\Theta$ random. Then his key innovation is to use this inner credal distribution as the sampler for the stitched contour estimate (Cella and Martin, 2025) as

$$
\rho _ { z _ { n } , Q } ( \theta ) = \mathbb { E } _ { \widetilde { \Theta } \sim Q } \left[ \mathbb { I } \{ r ( z _ { n } , \widetilde { \Theta } ) \leq r ( z _ { n } , \theta ) \} \right] .\tag{2.5}
$$

with the parameter sampling distribution $Q$ as $Q ^ { \mathrm { i n n } }$ . Here, $r$ is unnormalized rank since the normalizing constant canceled out as this formulation considers only one $z _ { n } .$ . Therefore, once having spent time to obtain $Q ^ { \mathrm { i n n } }$ , the empirical evaluation of (2.5) for various $\theta$ in query is inexpensive, especially by re-using the samples from $Q ^ { \mathrm { i n n } }$

Importantly, Corollary 1 of Martin (2026a) characterizes a $Q _ { z _ { n } } ^ { \mathrm { i n n } }$ through a shell level Markov mixture $Q _ { z _ { n } } ^ { \mathrm { i n n } } ~ = ~ \int _ { 0 } ^ { 1 } K ^ { \alpha } ( \cdot ) { \mathrm d } \alpha$ , where each $K ^ { \alpha }$ is supported on the boundary of $C _ { \alpha }$ . In this shell-mixture characterization, the α-cut boundaries therefore serve as the starting point, naturally suggesting to obtain samplers by learning these cut boundaries. Approximating $Q _ { z _ { n } } ^ { \mathrm { i n n } }$ , Martin (2026a) replaces the level set by an outer ellipsoid for each $\alpha \in [ 0 , 1 ]$ , and spread the weight by a uniform direction to the boundary. Note that such a

![](images/4e3fa3a3622bf601dfc19b34311df4962166bb642c3f280eb8ee963d54f9c214.jpg)  
Figure 1: The nested structure of our trained radial transport $T ^ { \mathrm { M I C } }$ . Transported samples from source balls of radius $1 - \alpha$ approximately fill the corresponding α-cuts. Black dotted lines indicate the estimated boundaries of the target cuts $C _ { \alpha }$

$Q _ { z _ { n } } ^ { \mathrm { i n n } }$ is not unique, and among them Martin seeks one that has a uniform distribution on each level-shell by a maximum entropy principle.

## 2.3 Our Approach: Possibilistic Radial Transport

The central theme of our work is viewing an inner credal distribution as a transport that preserves the contour depth. That is, when transporting the source $u ,$ the depth constraint is $\pi _ { z } ( T _ { z } ( u ) ) = 1 - \| u \|$ . Then we render learning the maximum entropy inner credal distribution as finding a calibrated transport that maximizes the entropy among the transported things from the same depth of the source. This frees us from the need of estimating the boundary of the α-cut of each level $\alpha \in [ 0 , 1 ]$ as our radial transport formulation does not explicitly invoke level sets. In fact, the shell information has not disappeared; it simply has a hiding place in the constraint linking source radius to contour depth. But then, any transport satisfying the depth constraint maps source samples with $\| U \| \le 1 - \alpha$ into the corresponding α-cut. That is, we use the transport to access the cuts rather than cuts as the pre-requisite for the credal distribution.

When we seek maximum entropy principle, we maximize the conditional entropy of the transported source given the source radius, integrated over radii. Any maximizing transport pushes the source distribution to the maximum-entropy inner credal distribution, $Q _ { z } ^ { \mathrm { M I C } }$

The hiding place, however, has its own computational challenge. Directly enforcing depth constraint during training would require expensive contour evaluations at each generated parameter value. We therefore replace the original contour with a diferentiable stitchedcontour approximation in the depth-matching loss. To keep this approximation faithful to the original contour, we add a faithfulness term that penalizes disagreement with contour targets precomputed at fixed training points. This allows us to train the transport without repeatedly evaluating the original contour. Figure 1 shows how the trained transports map source balls of radius $1 - \alpha$ to the corresponding α-cuts. The transported samples closely follow the geometry of the target cuts, although learning error prevents an exact match.

To amortize the sampler w.r.t. z as well, we need training target contour values for many datasets. Importantly, a likelihood normalizer depends on $Z$ , not on the $\theta$ at which the likelihood is evaluated. Therefore, the training data for approximating $\pi _ { z } ( \cdot )$ for a given z can be re-assembled to train an amortized sampler for approximating $\pi _ { Z } ( \cdot )$ for many diferent $Z$ (unknown at the time of training). Notice that Martin gives a direct sampler for nonelliptical star-convex cuts, which requires root-finding to locate a cut boundary for each sample (Martin, 2026a, Appendix C). Compared to this version, then, the IMMC algorithm itself can be seen as an amortization, as approximating $\pi _ { z } ( \theta )$ at each $\theta$ does not require optimization personalized to $\theta .$ In that sense our transport $T _ { z } ( u )$ for a fixed z can be seen as an already amortized version w.r.t. θ. In this regard, when we try to also amortize w.r.t. z, i.e., training $T ( u , Z )$ for $Z$ unseen at the time of training, it is a double amortization.

## 3 Possibilistic Radial Transport

In this section, we introduce the possibilistic radial transport, a map that hides the contour depth of a parameter in the radius of its source point. We establish that the pushforward distributions of this class are exactly the inner credal set $\mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ , and that sampling from an α-cut is recovered by simply truncating the source radius at $1 - \alpha$ . Among the transports in this class, we target the one that maximizes within-shell entropy. Then we develop a deep learning algorithm to find it and then amortize it across datasets.

## 3.1 Depth-Preserving Radial Representation

Our goal in this section is to build a flexible sampler for a distribution $Q \in \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ that is uniform within each contour shell. We seek an α-cut-free formulation of the sampler, which in turn lets us generate samples from the cuts.

We first define our target distribution. Consider the shell $L _ { z , a } = \{ \theta \in \mathbb { T } : \pi _ { z } ( \theta ) = a \}$ ， and consider the uniform distribution $\nu _ { z , a } = \mathrm { U n i f } ( L _ { z , a } )$ over that shell. Here, uniformity is defined relative to a declared shell reference measure. Assume that $a \mapsto \nu _ { z , a }$ is a measurable probability kernel. The maximum-entropy inner credal distribution is defined as

$$
Q _ { z } ^ { \mathrm { M I C } } ( B ) = \int _ { 0 } ^ { 1 } \nu _ { z , a } ( B ) \mathrm { d } a .\tag{3.1}
$$

Thus, $Q _ { z } ^ { \mathrm { M I C } }$ is a uniform mixture over contour depths and, conditionally on each depth, is uniform on the corresponding contour shell. With the same choice of shell reference measures, this coincides with Martin’s uniform-kernel inner probabilistic approximation.

The following proposition shows that replacing contour shells by the boundaries of enclosing ellipses can substantially alter plausibility assessments, even in the absence of numerical optimization and Monte Carlo errors. Let $Q _ { z } ^ { E }$ denote the population IMMC law obtained from the exact minimum-area MLE-centered enclosing ellipses, with uniform depths and uniform angular source directions. Recall the stitched contour $\rho _ { z , Q }$ defined in (2.5), and write $\rho _ { z } ^ { \mathrm { M I C } } = \rho _ { z , Q _ { z } ^ { \mathrm { M I C } } , } \rho _ { z } ^ { E } = \rho _ { z , Q _ { z } ^ { E } }$ . Also let $\operatorname { P l } _ { \pi _ { z } } ( H ) = \operatorname* { s u p } _ { \theta \in H } \pi _ { z } ( \theta )$ be the plausibility under the original IM contour, and $\begin{array} { r } { \mathrm { P l } _ { \rho _ { z } ^ { E } } ( H ) = \operatorname* { s u p } _ { \theta \in H } \rho _ { z } ^ { E } ( \theta ) } \end{array}$ be the plausibility under the stitched contour computed from $Q _ { z } ^ { E }$

Proposition 1 (Distortion from elliptical approximation). For every $\varepsilon \in ( 0 , 1 / 2 )$ , there exist a statistical model with parameter space $\mathbb { R } ^ { 2 }$ , an observation z, and a half-space H such that the exact IM contour $\pi _ { z }$ has compact, connected cuts with smooth boundaries at every depth $a \in ( 0 , 1 )$ , for which $\rho _ { z } ^ { \mathrm { M I C } } = \pi _ { z } , \mathrm { P l } _ { \pi _ { z } } ( H ) \leq \varepsilon$ , and yet $\operatorname { P l } _ { \rho _ { z } ^ { E } } ( H ) \geq 1 - \varepsilon$

The proof is provided in Appendix A.1. A hypothesis rejected by the original IM can have plausibility arbitrarily close to one under the ellipsoidal approximation. Thus, replacing the cuts by their enclosing ellipsoids may lead to an inferential conclusion diferent from that of the original IM, even when both the enclosing ellipses and the stitched contour are computed exactly. This motivates us to represent an inner credal distribution without first approximating its shells by a prescribed family of shapes.

We do this by tying $\mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ with a center-outward geometry by modeling through a transport from a source noise distribution. Note that although $C _ { \alpha }$ are nested by construction, they do not necessarily have a center-outward geometry. Assume that T is a Borel subset of $\mathbb { R } ^ { d }$ with $d \geq 2$ and that $\pi _ { z }$ is measurable.

Definition 1 (Inner Radial Transportation). Let $A \sim \mathrm { U n i f } ( 0 , 1 ) , V \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ , A ⊥ $V , U = ( 1 - A ) V .$ . Let $F _ { U }$ denote the distribution of U. Then $1 - \| U \| = A \sim \operatorname { U n i f } ( 0 , 1 )$ Define the class of inner radial transport as

$$
{ \mathcal { T } } _ { \mathrm { i n } } ( \pi _ { z } ) = \left\{ T : { \mathbb { B } } ^ { d } \to \mathbb { T } : T { \mathrm { ~ i s ~ m e a s u r a b l e ~ a n d ~ } } \pi _ { z } \{ T ( U ) \} = 1 - \| U \| , \ F _ { U } - a . s . \right\} .\tag{3.2}
$$

Interestingly, the pushforward distributions generated by ${ \mathcal { T } } _ { \mathrm { i n } } ( \pi _ { z } )$ form exactly the inner credal set, as the following theorem states. Furthermore, the cut-conditional distribution is recovered by transporting source draws conditional on $\| U \| \leq 1 - \alpha$ . This transport class also contains a map that generates the target $Q _ { z } ^ { \mathrm { M I C } }$

Theorem 1. The following properties of radial mapping holds:

(i) The pushforward distributions generated by inner radial transports form exactly the inner credal set $\{ T _ { \# } F _ { U } : T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } ) \} = \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$

(ii) For every $T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ and $\alpha \in ( 0 , 1 )$ , conditioning the source distribution on $\| U \| \leq$ $1 - \alpha$ yields the transported distribution conditioned on $C _ { \alpha } ( z )$ 2

$$
T _ { \# } \left[ F _ { U } \Big ( \cdot \vert \Vert U \Vert \leq 1 - \alpha \Big ) \right] = ( T _ { \# } F _ { U } ) \Big ( \cdot \vert C _ { \alpha } ( z ) \Big ) .
$$

In particular, there exists $T ^ { \mathrm { M I C } } \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ such that $( T ^ { \mathrm { M I C } } ) _ { \# } F _ { U } = Q _ { z } ^ { \mathrm { M I C } }$

The proof is provided in Appendix A.2. As a characterization of ${ \mathcal { C } } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ , Theorem 1 (i) is a transport counterpart to Martin’s Corollary 1. We note that to find $Q .$ , we no longer have to explicitly approximate the cut boundaries $\partial C _ { \alpha } ( z )$ first. Rather than $\partial C _ { \alpha } ( z )$ appearing explicitly, the cuts have a “hiding place” through the depth constraint $\pi _ { z } \{ T ( U ) \} \ : = : 1 - \| U \|$ . Thus, the radial representation has turned the task of directly constructing a shell mixture in $\mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ into one of finding a transport that obeys the depth constraint.

Now then, from this transport formulation, how can we pinpoint a $T ^ { \mathrm { M I C } }$ that pushes the source distribution forward to $Q ^ { \mathrm { M I C } } ?$ To this aim, we notice that the entropy maximization on each shell is possible while $\partial C _ { \alpha }$ is still hiding, through the source radius condition.

Theorem 2. Let $H _ { z , a }$ denote entropy relative to the declared shell reference measure. We set entropy to $- \infty$ when absolute continuity fails. Assume that this measure has finite, positive mass for almost every a. Also assume that $\begin{array} { r } { \int _ { 0 } ^ { 1 } \ d u \big | H _ { z , a } \big ( \nu _ { z , a } \big ) \big | } \end{array}$ da $< \infty$

(i) For every $T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ , the following entropy-gap identity holds:

$$
\int _ { 0 } ^ { 1 } \left[ H _ { z , a } ( \nu _ { z , a } ) - H _ { z , a } \left\{ T _ { \# } \left[ F _ { U } \Big ( \cdot \thinspace | \ \lVert U \rVert = 1 - a \Big ) \right] \right\} \right] \mathrm { d } a = \mathrm { K L } \left( T _ { \# } F _ { U } \lVert Q _ { z } ^ { \mathrm { M C } } \right) .
$$

(ii) For $T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ , the integrated conditional entropy is maximized if and only if the map sends the source distribution to $Q _ { z } ^ { \mathrm { M I C } }$ ，

$$
T \in \underset { T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } ) } { \mathrm { a r g } \operatorname* { m a x } } \int _ { 0 } ^ { 1 } H _ { z , a } \left\{ T _ { \# } \left[ F _ { U } \Big ( \cdot \vert \ : \| U \| = 1 - a \Big ) \right] \right\} \mathrm { d } a \Longleftrightarrow T _ { \# } F _ { U } = Q _ { z } ^ { \mathrm { M I C } } .\tag{3.3}
$$

The proof is provided in Appendix A.3. Theorem 2(ii) follows from the completeness in Theorem 1(i) and the entropy-gap identity in part (i), which shows that any departure from $Q _ { z } ^ { \mathrm { M I C } }$ within the inner credal set strictly lowers the integrated conditional entropy. Note that the maximizing transport in (3.3) need not be unique (e.g., the source’s rotation invariance). We take any maximizer and denote it as $T ^ { \mathrm { M I C } }$ . The problem is now reshaped to finding a transport that maximizes the integrated conditional entropy among those obey the depth constraint.

The hiding place for $\partial C _ { \alpha } ( z )$ is not cost free because enforcing $\pi _ { z } \{ T ( U ) \} = 1 - \| U \|$ is a serious headache in its own right. The depth constraint implies that we try to find $T ^ { \mathrm { M I C } }$ , we have to evaluate $\pi _ { z }$ after transporting every sample to make sure that a transport meets the restriction. Therefore, it is not that the challenge disappeared, but that it reshaped itself as another challenge. We think, however, that the challenge of evaluating $\pi _ { z }$ is manageable by using a proxy to bypass these repeated contour evaluations, which does not require an ellipsoidal approximation of the cuts.

Remark 1 (Kolmogorov−Smirnov diagnostics for depth calibration). Let $F _ { \pi , Q } ( \alpha ) = Q \{ \pi ( \Theta ) \leq$ $\alpha \}$ , and define $\begin{array} { r } { D _ { + } ( Q ) = \operatorname* { s u p } _ { \alpha \in ( 0 , 1 ) } [ F _ { \pi , Q } ( \alpha ) - \alpha ] _ { + } } \end{array}$ , and D<sub>−</sub> $\begin{array} { r } { ( Q ) = \operatorname* { s u p } _ { \alpha \in ( 0 , 1 ) } [ \alpha - F _ { \pi , Q } ( \alpha ) ] _ { + } } \end{array}$ These are the one-sided Kolmogorov–Smirnov (KS) discrepancies between the distribution of $\pi ( \Theta )$ under $Q$ and Unif(0, 1). Replacing $F _ { \pi , Q }$ with its empirical $\mathrm { C D F }$ yields the usual one-sample KS statistics. We report the two one-sided statistics separately to assess both the magnitude and direction of departures from uniform depth calibration.

Remark 2 (Warning for a misconception). We emphasize that $Q _ { z } ^ { \mathrm { M I C } }$ is not defined by normalizing the contour to make it a distribution. For example, when $\pi _ { z }$ is smooth with regular level sets, the coarea formula gives the density with respect to Lebesgue measure almost everywhere as $q _ { z } ^ { \mathrm { M I C } } ( \theta ) = \lVert \nabla \pi _ { z } ( \theta ) \rVert / \lvert \partial C _ { \pi _ { z } ( \theta ) } ( z ) \rvert . ^ { 1 }$ <sup>1</sup> Intuitively, the pdf of $Q _ { z } ^ { \mathrm { M I C } }$ would be higher where the contour slope relative to the level set size is larger and lower where this ratio is smaller.

## 3.2 Radial Transport Map Learning

In this section, we develop a learning algorithm for constructing an MIC sampler with target distribution $Q _ { z _ { n } } ^ { \mathrm { M I C } }$ . We now parameterize the transport as $T _ { \psi }$ , with $\psi \in \Psi$ , and replace the exact depth constraint in (3.2) by a scoring-rule penalty. We compare two distributions through using a scalar scoring rule $S ( p , q )$ , where $p$ denotes the target value and $q$ denotes the predicted value. This yields the following penalized population objective

$$
{ \mathcal L } ( \psi ) = - \int _ { 0 } ^ { 1 } H \big \{ Q _ { \psi } ( \cdot \mid A = \alpha ) \big \} d \alpha + \gamma _ { d } \mathbb { E } \Big [ S \Big ( A , \pi _ { z } \{ T _ { \psi } ( U ) \} \Big ) \Big ] ,\tag{3.4}
$$

for $U \sim F _ { U }$ , which is a sum of negative entropy with a depth-uniform penalty. Under exact feasibility, conditioning on $A = \alpha$ is equivalent to conditioning on $\pi _ { z } ( T _ { \psi } ( U ) ) = \alpha$ . When feasibility is only approximate, (3.4) is a penalized relaxation rather than an equivalent unconstrained formulation. In practice, we parametrize the transport $T _ { \psi }$ by using a deep neural network.

At each iteration, we simulate a mini-batch of transported particles, and then perform stochastic optimization using empirical approximations to (3.4). To address both diferentiability and computational eficiency, we will make extensive use of the (smoothed) stitched contour approximation. These approximations lead to the following empirical objective for stochastic optimization

$$
\widehat { \mathcal { L } } ( \psi ) = \gamma _ { h } \widehat { \mathcal { L } } _ { h } ( \psi ) + \widehat { \mathcal { L } } _ { d } ( \psi ) + \gamma _ { f } \widehat { \mathcal { L } } _ { \mathrm { f } } ( \psi ) ,
$$

where the last term stems from replacing the true contour with its stitched approximation. Here we weight the entropy term by $\gamma _ { h }$ rather than the depth term by $\gamma _ { d }$ as in (3.4); the two are equivalent up to overall scale $( \gamma _ { h } = 1 / \gamma _ { d } )$ , and we found it more natural in practice to tune the entropy weight directly. In what follows, we will define and describe each term.

The first term: negative entropy. The first term computes the negative entropy. Here, we adopt the Kozachenko–Leonenko entropy estimator (Kozachenko and Leonenko, 1987; Delattre and Fournier, 2017). Given samples $X _ { 1 } , \ldots , X _ { M }$ from an m-dimensional distribution, it is defined by

$$
\widehat { H } ( X _ { 1 : M } ) = \frac { m } { M } \sum _ { i = 1 } ^ { M } \log D _ { i } ^ { ( 1 ) } + C _ { m , M } ,\tag{3.5}
$$

where ${ D } _ { i } ^ { ( 1 ) }$ is the Euclidean distance from $X _ { i }$ to its 1-nearest neighbor among $\{ X _ { j } \} _ { j \neq i } ,$ and $C _ { m , M }$ is a bias-correcting constant depending only on $( m , M )$ . In our setting, we compute the entropy estimate among source points sharing the same radius, which has intrinsic dimension $m = d - 1$ . More precisely, let $A _ { i } \stackrel { \mathrm { i i d } } { \sim } \mathrm { U n i f } ( 0 , 1 )$ and $V _ { j } \overset { \mathrm { i i d } } { \sim } \operatorname { U n i f } ( \mathbb { S } ^ { d - 1 } )$ for $i = 1 , . . . , N$ and $j = 1 , . . . , M$ . Define $\widetilde { \Theta } _ { i j , \psi } : = T _ { \psi } ( ( 1 - A _ { i } ) V _ { j } )$ .

$$
\widehat { \mathcal { L } } _ { h } ( \psi ) : = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \widehat { H } ( \widetilde { \Theta } _ { i 1 , \psi } , \ldots , \widetilde { \Theta } _ { i M , \psi } ) .
$$

Thus, $\widehat { \mathcal { L } } _ { h } ( \psi )$ includes both the within-shell entropy estimation and the Monte Carlo integration over the shell level.

The second term: uniform regularization. The second term handles radial uniform regularization. For the scoring function $S ,$ we use the Brier-log composite score for S as $S ( p , q ) = ( p - q ) ^ { 2 } + \{ - p \log q - ( 1 - p ) \log ( 1 - q ) \}$ . The Brier part stabilizes global shape matching, while the log-score part penalizes errors near 0 and 1, which are important for tail contours. This score is minimized uniquely at $q = p$ . Note that any strictly proper scoring rule can be used for the excess scores.

Here, we must handle the issue that $\pi _ { z } ( \theta )$ may not be diferentiable w.r.t θ. For example, the simulator may not be written as a diferentiable pathwise map w.r.t. θ such as discrete sampling or hard thresholding. The cost of computing $\pi _ { z } ( T _ { \psi } ( U ) )$ for each iteration is another challenge. Therefore, in the radial-uniformity term of (3.4), we replace π by the stitched contour in (2.5). This is defined in our context as $\rho _ { \psi } ( \theta ) : = Q _ { \psi } \{ R ( \Theta ) \leq R ( \theta ) \}$ When estimating $\rho _ { \psi } ( \theta )$ , we use diferentiable, temperature-smoothed counterpart of the hard-indicator estimator, which is defined as

$$
\widehat { \rho } _ { \psi , \tau } ( \theta ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sigma \left( \frac { R ( \theta ) - R ( \widetilde { \Theta } _ { k } ^ { \mathrm { r e f } } ) } { \tau } \right) ,\tag{3.6}
$$

where $\widetilde { \Theta } _ { k } ^ { \mathrm { r e f } } \ = \ T _ { \psi } ( U _ { k } ^ { \mathrm { r e f } } )$ , U<sup>ref</sup><sub>k</sub> <sup>iid</sup>∼ F<sub>U</sub> , $\sigma ( x ) = ( 1 + e ^ { - x } ) ^ { - 1 }$ , and $\tau > 0$ is a temperature parameter. The resulting empirical radial-uniformity loss is

$$
\widehat { \mathcal { L } } _ { d } ( \psi ) = \frac { 1 } { N M } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } S \Bigl ( A _ { i } , \widehat { \rho } _ { \psi , \tau } ( \widetilde { \Theta } _ { i j } ) \Bigr ) .
$$

The stitched computation requires no additional simulated data $Z \sim P _ { \theta }$ . Consequently, this replacement not only provides a diferentiable surrogate but also substantially reduces the computational cost by avoiding repeated simulator draws and the associated normalization required to evaluate the original contour for every generated query particle.

The third term: fidelity of the stitched proxy. Since the depth constraint is enforced through the stitched approximation, it is very important that the stitched proxy has a high fidelity to the contour. Conceptually, we want to penalize $\mathbb { E } _ { \Theta \sim \operatorname { U n i f } ( \mathbb { T } ) } \left[ S ( \pi _ { z } ( \Theta ) , \rho _ { \psi } ( \Theta ) ) \right]$ 2 speaking in the population level. Note that $\pi _ { z } ( \Theta )$ does not depend on $T _ { \psi }$ . Therefore, before training, we construct the lookup table $\mathcal { D } _ { \mathrm { t a r g e t } } = \{ ( \vartheta _ { r } , \pi _ { z } ( \vartheta _ { r } ) ) \} _ { r = 1 } ^ { N _ { \mathrm { t a r g e t } } }$ , for $\vartheta _ { r } \stackrel { \mathrm { i i d } } { \sim } \mathrm { U n i f } ( \mathbb { T } )$ This one time construction helps to avoid repeatedly computing these expensive contour values during training. At each iteration, we sample a mini-batch $\{ ( \vartheta _ { \ell } , \pi _ { z } ( \vartheta _ { \ell } ) ) \} _ { \ell = 1 } ^ { N _ { \mathrm { f } } }$ from $\mathcal { D } _ { \mathrm { t a r g e t } }$ . As in the radial-uniformity loss, we use the diferentiable soft stitched estimate $\widehat { \rho } _ { \psi , \tau } ( \vartheta _ { \ell } )$ as in (3.6). The resulting empirical fit loss is

$$
\widehat { \mathcal { L } } _ { \mathrm { f } } ( \psi ) = \frac { 1 } { N _ { \mathrm { f } } } \sum _ { \ell = 1 } ^ { N _ { \mathrm { f } } } S ( \pi _ { z } ( \vartheta _ { \ell } ) , \widehat { \rho } _ { \psi , \tau } ( \vartheta _ { \ell } ) ) .
$$

![](images/a2f24f3a9a35f4ce856d1346cd222a8626b4a5eab14eca76fe25f3034735322a.jpg)  
Figure 2: (Example 1) Target contour $\pi$ (left), IMMC estimate (middle) and MIC estimate (right) for Gamma, Weibull and S-shear. White curves are the level sets $\alpha \in$ {0.05, 0.1, 0.2, 0.5} of each panel.

After training, transporting source draws satisfying $\| U \| \leq 1 - \alpha$ provides an approximate sampler for the cut-conditional MIC distribution, without further contour evaluations. That is, while training $Q _ { \psi }$ might be costly, once trained, (sub)sampling $C _ { \alpha }$ is cheap. The full algorithm for learning $Q ^ { \mathrm { M I C } }$ is presented in Algorithm 1 of Section B.

Remark 3. It is possible that the stitched contour approximates the original possibility contour very well. Suppose $R _ { z } ( \Theta )$ has a continuous distribution under $Q$ . Following Martin (2026b, Theorem 2) and using the probability integral transform, $\rho _ { Q } = \pi _ { z }$ if and only if $Q \in \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ and $\pi _ { z } = g _ { z } \circ R _ { z }$ for some nondecreasing $g _ { z }$ . This establishes structural compatibility of the objectives with an exact MIC transport that minimizing the transport loss while also minimizing the faithfulness loss at the same time is possible. If $\pi _ { z }$ is not a nondecreasing function of $R _ { z }$ , no stitched proxy based on $R _ { z }$ can reproduce it exactly, and the fidelity term may compete with the MIC objectives.

Example 1. Consider three toy examples of Gamma and Weibull (Martin, 2026a), and Shear. For Gamma, we model the rat survival times as $Z _ { i } \sim \mathrm { G a m m a } ( a , b )$ iid, with shape $a ~ > ~ 0$ , scale $b > 0$ , and log-parameter $\theta \ : = \ : ( \log a , \log b )$ used in Martin (2026a). For

Table 1: (Example 1) depth calibration, contour accuracy, and spatial coverage.
<table><tr><td>Model</td><td>Method</td><td> $D _ { + }$   $D _ { - }$ </td><td>MSE</td><td>max err. Spearman</td><td> $C _ { . 0 5 }$ </td></tr><tr><td rowspan="2"></td><td>Gamma IMMC</td><td>0.070</td><td>0.026 0.0002</td><td>0.116</td><td>0.988 0.983</td></tr><tr><td>MIC (ours)</td><td>0.075 0.063</td><td>0.0001</td><td>0.069 0.989</td><td>1.000</td></tr><tr><td rowspan="2">Weibull</td><td>IMMC</td><td>0.153 0.025</td><td>0.0021</td><td>0.159</td><td>0.987 0.982</td></tr><tr><td>MIC (ours)</td><td>0.136 0.056</td><td>0.0002</td><td>0.152 0.986</td><td>1.000</td></tr><tr><td rowspan="2">S-shear</td><td>IMMC</td><td>0.034</td><td>0.136 0.0016</td><td>0.110</td><td>1.000 0.543</td></tr><tr><td>MIC (ours)</td><td>0.057 0.049</td><td>0.0001</td><td>0.040</td><td>1.000 0.982</td></tr></table>

Weibull, we model the latent failure times as $Y _ { i } \sim \mathrm { { W e i b u l l } } ( k , \lambda )$ iid, with shape $k > 0$ scale $\lambda > 0$ , log-parameter $\theta \ : = \ : ( \log k , \log \lambda )$ , and observe right-censored data $( X _ { i } , \Delta _ { i } )$ 2 where $X _ { i } = \operatorname* { m i n } ( Y _ { i } , C _ { i } )$ and $\Delta _ { i } = I ( Y _ { i } > C _ { i } )$ . As a non-star-convex demonstration, we include the Shear model whose contour is exactly known. Let $Z _ { i } \stackrel { i i d } { \sim } N _ { 2 } ( m ( \theta ) , \sigma ^ { 2 } I _ { 2 } )$ for $i = 1 , . . . , 1 0$ with $\sigma = 2 . 5$ and $m ( \theta ) = ( \theta _ { 1 } , \theta _ { 2 } + 2 \sin ( 2 . 5 \theta _ { 1 } ) )$ . In $\eta = m ( \theta )$ this is a Gaussian location model with closed-form contour $\pi _ { z ^ { n } } ( \theta ) = \exp \{ - n \| \bar { z } - m ( \theta ) \| ^ { 2 } / ( 2 \sigma ^ { 2 } ) \}$ , whose α- cuts are discs in $\eta$ and bent bands in θ. We keep the focus on $\theta ,$ where the contour is non-star-convex.

Figure 2 shows that MIC’s reconstruction is visually comparable to that of IMMC for Gamma and more accurate for Weibull and S-shear, with tighter tail contours. Despite using ellipsoidal approximations, IMMC provides good reconstructions of non-elliptical contours away from the tails. The reconstructions are compared numerically in Table 1. For the uniform calibration check, we estimate $D _ { + }$ and D defined in Remark 1. A conservative $5 \%$ cutof for the two-sided KS diagnostic is approximately 0.096 (See, Section C.1 for details). We see that IMMC’s $D _ { + }$ and $D _ { - }$ <sub>−</sub> values are below this threshold for the Gamma model, indicating no detected departure from uniform calibration at this threshold. MIC’s deviations fall below this threshold in Gamma and S-shear. For contour quality, we see that both methods perform reasonably well in terms of MSE, maximum absolute error, and Spearman correlation between the reconstructed contour $\hat { \rho } _ { Q }$ and the target contour π. The coverage metric shows that MIC captures most of the grid-based approximation

to $C _ { . 0 5 }$ . This is supported by the scatter plot of $Q$ samples in Figure 12 in the Appendix.   
Implementation and evaluation details for MIC and IMMC are provided in Section C.1.

## 3.3 Amortization w.r.t. Data $z$

Here, we discuss how to learn a sampler $Q _ { z } ^ { \mathrm { M I C } } ( \theta )$ that is amortized not only w.r.t. θ but also w.r.t. the data z. That is, instead of learning a transport $T _ { z } ( u )$ for a fixed $z ,$ , we can learn $T ( u , z )$ , where $z$ is also an input. Once trained, the conditional transport can be reused across datasets without training a separate sampler for each dataset, facilitating repeatedinference tasks such as power and coverage estimation besides the contour estimation and α-cut sampling for the original $z _ { 0 }$ . It may bounce a new z if its predictive distribution behaves diferently from typical ones that are generated by the likelihood model we consider.

Amortizing our algorithm is not conceptually dificult. Once we have training dataset, the pair of $( z , \theta , \pi _ { z } ( \theta ) )$ for various z and $\theta$ is given, our original learning algorithm extends to the conditional transport with a slight tweak to handle various $Z$ as described in Algorithm 2. Once the amortized sampler is trained, we can empirically assess the coverage of its confidence regions using independent simulated datasets before inserting this amortized machine into the real world. We can also estimate power to understand the sensitivity of the tests of interest under the specified likelihood model.

While things sound simple, the cost of getting the training dataset is the overhead. Preparing the training data even for our standard MIC learning is computationally costly because it requires approximating the target $\pi _ { z } ( \theta )$ by MC. Therefore, thinking of obtaining the training data that will be used for even various z realizations sounds daunting. Thankfully, the good news is that the training data preparation for the standard MIC learning already computes the likelihood normalizing constants for any $Z$ that ever was sampled during the MC. For each simulated dataset $Z ,$ the likelihood normalizing constant depends only on $Z ,$ not on the parameter value $\theta$ at which the likelihood is evaluated. These resources can therefore be reused to construct training labels (contour values estimated through MC) across these datasets without additional simulation or MLE fitting. This reuse can substantially reduce the cost of generating the amortized training data, especially when likelihood normalization is the dominant cost. Utilizing this, an eficient training-data generation process is described in Algorithm 3.

A caveat is that the parameter box is chosen using an initial dataset $z _ { 0 }$ , so its suitability for amortized inference depends on whether it also covers the relevant contour regions for other datasets. When these regions are largely shared across datasets, as in the Sshear example, the same box can support amortized learning. However, when the feasible parameter region depends on z, the coverage of the box and the handling of support constraints would require additional care. Under a fixed computational budget, a wider box may accommodate more datasets but provide fewer training points with non-negligible contour values for each dataset. A narrower box concentrates training efort near the relevant parameter region for $z _ { 0 } .$ , at the cost of a more limited range of reuse.

When we do not want to extend our training parameter space too much, we can resort to an internal predictive check we explain below. This can be a useful optional screening before applying the learned sampler to a new dataset. A discrepancy can reflect limitations of the chosen parameter box, model misspecification, or inadequate generalization of the learned transport. While the check does not distinguish these causes, it flags cases requiring further investigation before interpreting the sampler’s output.

Our internal predictive check is a double generational predictive comparison to bounce a new dataset whose relationship with its predictive samples is atypical under our trained sampler. We simulate the first generation (children) similar to Bayesian predictive sampling, i.e. sample $\theta _ { i } \sim Q ( \theta \mid z _ { 0 } )$ and then sample $Z _ { i } \sim P _ { \theta _ { i } }$ . Then we generate the next generation (grandchildren) $Z _ { i j } \sim P _ { \theta _ { i j } }$ for $\theta _ { i j } \sim Q ( \theta \mid Z _ { i } ) , j = 1 , \ldots , J$ . Here, the premise is that, if our sampler is a perfect sampler and if $z _ { 0 }$ is from our specified likelihood model, the relationship between $z _ { 0 }$ and its children should be similar to the relationship between each $Z _ { i }$ and its children. This is simply because we know that $Z _ { i }$ genuinely come from the model. This gives us a basic screening check. We measure this generational relationship by the distance between $Z _ { i }$ and $Z _ { i j }$ , defining $d _ { i j } = d ( Z _ { i } , Z _ { i j } )$ . Here, since Z itself is a set of observations, we use the energy distance (Székely and Rizzo, 2013), which coincides with the squared MMD with kernel $k ( x , y ) = ( \| x \| + \| y \| - \| x - y \| ) / 2$ (Sejdinovic et al., 2013). It is widely used to test whether two samples come from the same distribution, and is zero if and only if the two distributions coincide. Then, for each i, we can make a KDE plot using $\{ d _ { i j } \} _ { j = 1 } ^ { J }$ . We can compare these KDE curves with the KDE curve of $d _ { 0 i } = d ( z _ { 0 } , Z _ { i } )$ $i = 1 , \dots , I$ . For example, we define a diagnostic score as the fraction of reference medians that are at least as large as the median distance from the observed data to its predictive samples. We use this comparison as a heuristic diagnostic.

![](images/72abcd956db37c6badec81e831f38eaa424077525772ab51d4c8d4c5bfbcbc4d.jpg)  
Figure 3: Contours of $\pi _ { z }$ for three data realizations z (cases 1–3). (a–c) Samples from a single amortized MIC sampler. (d–f) IMMC samples, with a separate IMMC approximation fitted for each dataset. (g) Contour approximation errors and depth-calibration statistics.

Example 2. In demonstrating amortized learning, we focus on S-shear models. This is because the true contour is available in closed form, allowing direct evaluation (not training) of the learned contour and transport. It also provides a challenging amortization problem because the target contour geometry varies substantially with the observed data z. In particular, its central ridge can tilt in opposite diagonal directions for diferent datasets. We use a single trained conditional transport $T ( u , z )$ throughout this experiment, while IMMC is fitted separately to each dataset. We center the transport at the MLE and condition its neural component on $( \sin ( b \bar { z } _ { 1 } ) , \cos ( b \bar { z } _ { 1 } ) )$ ), where $b = 2 . 5$ is the known shear frequency and $\bar { z } _ { 1 }$ is the sample mean of the first coordinate. This model-specific phase encoding reflects the changing shear geometry and allows us to retain a small network comparable to the non-amortized version. The implementation details are in Section C.2.

1. Contour estimation. Figure 3 compares the two samplers for three data realizations. The MIC samples follow the curved target contours despite their changing orientation and shape across datasets. The substantially smaller MSE values for MIC in Figure 3g implies that a single transport map $T ( u , z )$ maintains substantially smaller contour errors across these examples. Figure 13 in Appendix further shows that MIC is closer to the desired depth calibration.

2. Coverage and Power estimation. Table 2 summarizes coverage and eficiency at true parameters $\theta _ { \phi } = \left( \phi / 2 . 5 , - 2 \sin \phi \right)$ for $\phi \in \{ 0 , \pi / 2 , \pi \}$ . For each fixed $\phi ,$ we generate 200 independent datasets and evaluate coverage and the area of the 95% confidence region. At the nominal 95% level, MIC has empirical coverage of 96.5–98.0%. MIC therefore exhibits mild empirical conservatism at this level, with 95% confidence-region areas close to those of the exact reference. On the other hand, Figure 4 reports empirical rejection probabilities for testing $H _ { 0 } : \theta = ( 0 , 0 )$ at level $\alpha = 0 . 0 5$ under two families of alternatives, $\theta _ { \eta } ^ { ( 1 ) } = \left( \eta , - 2 \sin ( 2 . 5 \eta ) \right)$ and $\theta _ { \eta } ^ { ( 2 ) } = ( 0 , \eta )$ , which give $m ( \theta _ { \eta } ^ { ( 1 ) } ) = ( \eta , 0 )$ and $m ( \theta _ { \eta } ^ { ( 2 ) } ) = ( 0 , \eta )$ respectively. Posing an alternative hypothesis $\theta _ { a }$ as the ground truth, we can simulate data from $P _ { \theta _ { a } }$ many times and check how many times the hypothesis in test is rejected. Each rejection probability is estimated from 100 independent datasets. We can see that MIC closely tracks the rejection probabilities of the exact reference. Interestingly, MIC remains close to the reference even when the true parameter values lie outside the training box (those marked by open symbols). While this empirical generalization beyond the training parameter box is interesting, we do not establish a guarantee for such extrapolation.

3. Predictive Check. Figure 5 illustrates our predictive check with the amortized $Q ^ { \mathrm { M I C } }$ . We sample children and the grandchildren by using source draws satisfying $\| U \| \leq$ $1 - \alpha .$ , for $\alpha \in \{ 0 . 1 , 0 . 9 \}$ . The gray curves are the KDEs of $\{ \log d _ { i j } \} _ { j = 1 } ^ { 1 0 0 }$ for each $i ,$ providing an internal predictive reference, while the red curve is the KDE of $\{ \log d _ { 0 i } \} _ { i = 1 } ^ { 1 0 0 }$ , i.e. the log distances from the observed data $z _ { 0 }$ to its predictive samples. The reported $p$ is the fraction of gray medians that exceed the red median (diagnostic score). As Figure 5 shows, the scale inflation case provides a particularly flattering illustration for our predictive check as it is based on a distance (second column). Notice, however, that the mixture case also yields small diagnostic scores despite approximately matched marginal variances, illustrating a discrepancy beyond inflated marginal spread (third column).

Table 2: (Amortized S-shear) Repeated inference across datasets, Coverage entries are estimates [pointwise 95% Wilson CI]; area entries are medians [10th, 90th percentiles] across datasets. Reference is computed by using the exact, closed form.
<table><tr><td></td><td>Phase Method</td><td>90% coverage</td><td>95% coverage</td><td>99% coverage</td><td>95% area</td></tr><tr><td>0</td><td></td><td>Reference 0.920 [0.874, 0.950] 0.955 [0.917, 0.976] 0.995 [0.972, 0.999]</td><td></td><td></td><td>11.764 [11.764, 11.764]</td></tr><tr><td></td><td>MIC</td><td>0.945 [0.904, 0.969] 0.980 [0.950, 0.992] 0.995 [0.972, 0.999]</td><td></td><td></td><td>13.103 [12.410, 13.552]</td></tr><tr><td></td><td>IMMC</td><td></td><td></td><td></td><td>0.990 [0.964, 0.997] 0.995 [0.972, 0.999] 1.000 [0.981, 1.000] 79.417 [23.612, 136.648]</td></tr><tr><td> $\pi / 2$ </td><td></td><td>Reference 0.900 [0.851, 0.934] 0.950 [0.910, 0.973] 0.985 [0.957, 0.995]</td><td></td><td></td><td>11.764 [11.764, 11.764]</td></tr><tr><td></td><td>MIC</td><td>0.920 [0.874, 0.950] 0.965 [0.930, 0.983] 0.990 [0.964, 0.997] 13.029 [12.290, 13.501]</td><td></td><td></td><td></td></tr><tr><td></td><td>IMMC</td><td>0.980 [0.950, 0.992] 0.995 [0.972, 0.999] 1.000 [0.981, 1.000] 86.669 [23.751, 137.184]</td><td></td><td></td><td></td></tr><tr><td>π</td><td></td><td>Reference 0.890 [0.839, 0.926] 0.965 [0.930, 0.983] 0.990 [0.964, 0.997] 11.764 [11.764, 11.764]</td><td></td><td></td><td></td></tr><tr><td></td><td>MIC</td><td>0.900 [0.851, 0.934] 0.970 [0.936, 0.986] 1.000 [0.981, 1.000] 12.929 [12.168, 13.380]</td><td></td><td></td><td></td></tr><tr><td></td><td>IMMC</td><td>0.985 [0.957, 0.995] 0.995 [0.972, 0.999] 1.000 [0.981, 1.000] 76.473 [22.353, 138.914]</td><td></td><td></td><td></td></tr></table>

![](images/38f67669ef9bb2480270b50a9e4cdf6def69c2622a43964f03cbc3744ead997b.jpg)

![](images/96a6b3fe0d3f78383708c09817e0a20054769721e41f004c260f89d14fdcd154.jpg)  
Figure 4: Empirical power for testing $H _ { 0 } : \theta = ( 0 , 0 )$ in the S-shear model at $\alpha = 0 . 0 5$ We generate data from $\theta _ { \eta } = ( \eta , - 2 \sin ( 2 . 5 \eta ) )$ in the left panel and $\theta _ { \eta } = ( 0 , \eta )$ in the right panel, 100 times for each η. The empirical type-I error rate corresponds to $\eta = 0$

(a) Model draw, p = 0.38  
![](images/849cf09a5f7cb89cd836b2fc48afe9b270d87b82399dbccffae301cf53c81c89.jpg)

(b) Larger variance, p = 0.00  
![](images/81d06d375e3d9772b8bac2b29b5dc6bf08a15358988c7620ce3b512c69b2c9ff.jpg)  
(d) Model draw, p = 0.11

(c) Mixture, p = 0.00  
![](images/b6dbd40207c93865efdeb5cfcd1364d1d1904c1448a6a0bb918dccd84231efb8.jpg)  
(e) Larger variance, p = 0.00  
(f) Mixture, p = 0.01

![](images/19bda76a5a211f71f3d7ec6e699e8f44fdb3de1dfac55fa375a965b3314b5205.jpg)

![](images/b22bc204a5952406bb83f222e2c2565a2e0f062cb01caf4b0b4ff01de807db2e.jpg)

![](images/f5f95d228e7638d3c94fb1cd1d0b1b652d3b9a787679074068383682bb515546.jpg)  
Figure 5: Predictive screening for model-generated data (σ = 2.5; left), inflated deviations (×1.5; middle), and a Gaussian mixture with approximately matched marginal variances (right). All share the same sample mean and contour. Red: $z _ { 0 }$ to children (predictive) distances. Gray: children to grand children distances. p is a diagnostic score.

## 4 Hypothesis selection with the Bel–Pl Spectrum

Here, we explain how $Q _ { z } ^ { \mathrm { M I C } }$ helps us peek into the hypothesis geometry hidden behind the Bel–Pl endpoints. Our goal is to use this geometric information to select among interpretable hypotheses that satisfy a prescribed Bel–Pl decision criterion. Note that the test rule with Pl/Bel in (2.3) gives the equivalence $P _ { \theta } \{ \exists H \ni \theta : { \mathrm { P l } } _ { Z } ( H ) \leq \alpha \} = P _ { \theta } \{ \pi _ { Z } ( \theta ) \leq \alpha \}$ This equips the test with a uniform validity guarantee (Martin, 2026b; Cella and Martin, 2023),

$$
\operatorname* { s u p } _ { \theta \in \mathbb { T } } P _ { \theta } ^ { ( n ) } \bigl \{ P l _ { Z _ { n } } ( H ) \leq \alpha \mathrm { f o r ~ s o m e ~ m e a s u r a b l e ~ } H \ni \theta \bigr \} \leq \alpha .\tag{4.1}
$$

Given such a hypothesis probing or tuning is possible, if we are researchers across disciplines analyzing their data, we may be mindful of three things; 1) We may want the accepted hypothesis to be interpretable; an analyst may have a set of interpretable hypotheses, say $\xi > 0$ or $\theta _ { 1 } > \theta _ { 2 }$ , and may be interested in selecting among these interpretable hypotheses; 2) We may want the accepted hypothesis to be as informative as possible, for example, to have a small area. While $H = C _ { \alpha }$ would be the most eficient, but it may not be interpretable; and 3) When having a similarly sized hypothesis (or union of hypothesis), we may want one that has the largest mean conviction (contour) within the hypothesis.

![](images/ebdb3518821c6612ca0eb711eb8ac5c86b454d39bf34c9fb724b84c5ab84f954.jpg)  
Figure 6: The Bel-Pl spectrum for a hypothesis H.

Bel-Pl spectrum. Our sampler $Q _ { z } ^ { \mathrm { M I C } }$ can help to compute useful values for this process providing useful diagnostics values. The first is a diagnostics tool a Bel-Pl spectrum $( B e l ( H ) , \hat { g } _ { z , H } ( \cdot ) , \ P l ( H ) )$ . This is a useful diagnostics tool for understanding the area eficiency of a hypothesis in terms of contour level set occupancy. By this we can see the gap between the goal of accept/reject and the current hypothesis. We first define $g _ { z , H } : \alpha \mapsto [ 0 , 1 ]$ , a shellwise allocation function of H as follows. Assuming that the contour shell coincides with $\partial C _ { \alpha } ( z )$ , we can intuitively write that for any measurable set H,

$$
g _ { z , H } ( \alpha ) : = Q _ { z } ^ { \mathrm { M I C } } \{ \Theta \in H \mid \pi _ { z } ( \Theta ) = \alpha \} = \frac { | \partial C _ { \alpha } ( z ) \cap H | } { | \partial C _ { \alpha } ( z ) | } ,\tag{4.2}
$$

where | · | denotes the declared reference measure on the contour shell, and the conditional notation refers to the shell kernel $\mathrm { U n i f } ( L _ { z , \alpha } )$ introduced in Section $\mathrm { A ^ { 2 } }$ . A conceptual example of the Bel-Pl spectrum is illustrated in Figure 6. The Bel–Pl spectrum has a flat right tail at zero or one unless Bel<sub>z</sub>(H) = 0 and $\mathrm { P l } _ { z } ( H ) = 1$ simultaneously.<sup>3</sup>

Bloat and Boon. The profile gives the proportion of each shell occupied by H. In Figure 6(a-b), the orange shaded area measures its accumulated low-depth occupancy.

Moving from $\alpha = 1$ toward $\alpha = 0$ , the profile in Figure $\mathrm { 6 ( a ) }$ first departs from one at $\alpha ^ { * } = 1 { \mathrm { - B e l } } _ { z } ( H )$ , whereas the profile in Figure 6(b) first departs from zero at $\alpha ^ { * } = \operatorname { P l } _ { z } ( H )$ . In Figure $\mathrm { 6 ( a ) }$ , when the orange shaded area is large, it motivates searching for a tighter hypothesis within the interpretable class. A small shaded area, on the other hand, indicates little excess occupancy beyond $C _ { t } ( z )$ for $t = 1 - B e l ( \boldsymbol { H } )$ . When we are in a rejecting setting like in Figure $\mathrm { 6 ( b ) }$ , a larger orange shaded area below the error threshold means more options are rejected, making the test more informative. Since the profile measures proportions rather than absolute size, the area does not mean the actual size of $C _ { t } ( z )$ but the occupancy rate of H on each $\partial C _ { \alpha }$

Within-hypothesis mean contour depth While $Q _ { z } ^ { \mathrm { M I C } } ( H )$ measures the area under the spectrum, it leaves a vagueness as an eficiency metric: for example, consider two sets, $A = \{ \pi _ { z } ( \theta ) > 0 . 5 \}$ and $B = \{ \pi _ { z } ( \theta ) < 0 . 5 \}$ . They have the same probability $1 / 2$ w.r.t. $Q _ { z } ^ { \mathrm { M I C } }$ , while the contour gives more conviction to A. Handling this vagueness, we define the expected contour π value,

$$
\begin{array} { r } { \bar { \pi } _ { z } ( H ) : = E _ { \Theta \sim Q _ { z } ^ { \mathrm { M I C } } } \Big [ \pi _ { z } ( \Theta ) \mid \Theta \in H \Big ] = \frac { \int _ { H } \pi _ { z } ( \theta ) Q _ { z } ^ { \mathrm { M I C } } ( d \theta ) } { Q _ { z } ^ { \mathrm { M I C } } ( H ) } = \frac { \int _ { 0 } ^ { 1 } \alpha g _ { z , H } ( \alpha ) d \alpha } { \int _ { 0 } ^ { 1 } g _ { z , H } ( \alpha ) d \alpha } , } \end{array}\tag{4.3}
$$

with convention of $\bar { \pi } _ { z } ( H ) = 0$ when $Q _ { z } ^ { \mathrm { M I C } } ( H ) = 0$ . This weighted shell-occupancy profile mean reveals whether that occupancy is concentrated at high or low contour depths. For example, for above sets A and $B _ { : }$ , now we have $\bar { \pi } _ { z } ( A ) = 3 / 4$ and $\bar { \pi } _ { z } ( B ) = 1 / 4$ . Besides, dividing by $Q _ { z } ^ { \mathrm { M I C } } ( H )$ prevents a set from being favored simply because it has high mass. It is easy to show given $H _ { 1 }$ and $H _ { 2 }$ accepted at level $\alpha \in \mathsf { \Gamma } ( 0 , 1 / 2 )$ , if $H _ { 1 } ~ \subseteq ~ H _ { 2 }$ and $Q _ { z } ^ { \mathrm { M I C } } ( H _ { 2 } \setminus H _ { 1 } ) > 0$ , then $\bar { \pi } _ { z } ( H _ { 1 } ) > \bar { \pi } _ { z } ( H _ { 2 } )$ . This means when we have multiple candidate hypothesis that has inclusion relationship, a smaller set will be prioritized based on $\bar { \pi } _ { z }$

Hypothesis Selection or Rejection. Given surplus measures such as Bloat and Boon, along with the expected depth, there are many ways to formulate a hypothesis selection or composition rule. The optimal criterion would depend on the analyst’s specific context. Here, we suggest one of many possible selection rules. Putting these ingredients together,

![](images/15607a1e7eef904526def82127b8c1853fec8cccb17b4c3ece87d7d0a1266607.jpg)

![](images/0efd31d389149f209cb49c4774c329d45ce48a7a37177e8b1108708007656a29.jpg)

![](images/2617cd3c080417f86b410e2825bc03b89f5822e74d51bb1289eae497a1a74c2a.jpg)

![](images/d1eaa0b3293ee20abe4c3b2767c98d8188a31a64479c13c318ca292d10e8ff13.jpg)  
Figure 7: The primal candidate hypotheses in Example 3. Their 15 nonempty unions are considered as candidates.

we define the hypothesis selection rule

$$
\begin{array} { r l } { \widehat { H } ( z ) \in \underset { H \in \mathcal { H } } { \arg \operatorname* { m a x } } } & { \mathbb { E } _ { Q _ { z } ^ { \mathrm { M r c } } } \Big [ \pi _ { z } ( \Theta ) \mid \Theta \in H \Big ] } \\ & { \mathrm { s u b j e c t ~ t o ~ } \mathrm { B e l } _ { z } ( H ) \ge 1 - \alpha _ { 0 } \mathrm { ~ a n d ~ } Q _ { z } ^ { \mathrm { M r C } } ( H ) - B e l _ { z } ( H ) \le \beta , } \end{array}\tag{4.4}
$$

where H is a class of interpretable candidate hypotheses, $\alpha _ { 0 } \in ( 0 , 1 )$ is a fixed inferential error level, and $\beta \geq 0$ bounds the intrinsic bloat $Q _ { z } ^ { \mathrm { M I C } } ( H ) \mathrm { - B e l } _ { z } ( H )$ . Since Bel<sub>z</sub> $( H ) \ge 1 - \alpha _ { 0 }$ implies $g _ { z , H } ( \alpha ) = 1$ for almost every $\alpha \in ( \alpha _ { 0 } , 1 )$ , the optimization compares the contour depth and amount of the excess occupancy outside the shared core $C _ { z } ( \alpha _ { 0 } )$

In the rejection setting, we could seek to exclude more MIC mass, giving greater weight to lower contour values. For example, we could select

$$
\widehat { H } ( z ) \in \mathop { \mathrm { a r g } } _ { H \in \mathcal { H } } \operatorname* { m a x } _ { \begin{array} { l } { H } \end{array} } \ \int _ { H } \left\{ 1 - \pi _ { z } ( \theta ) \right\} Q _ { z } ^ { \mathrm { M I C } } ( d \theta ) \quad \mathrm { s u b j e c t ~ t o } \quad \mathrm { P l } _ { z } ( H ) \leq \alpha _ { 0 } ,\tag{4.5}
$$

which is equivalent to maximizing $Q _ { z } ^ { \mathrm { M I C } } ( H ) \mathbb { E } _ { Q _ { z } ^ { \mathrm { M I C } } } \bigl [ 1 - \pi _ { z } ( \Theta ) \ | \ \Theta \in H \bigr ]$

Example 3. We continue in the Shear data. The basis candidate hypotheses are $D _ { 1 } =$ $\{ | \theta _ { 2 } + 5 \theta _ { 1 } | \leq 5 \} , D _ { 2 } = \{ | \theta _ { 2 } + 2 \theta _ { 1 } | \leq 2 . 5 \} , D _ { 3 } = \{ | \theta _ { 2 } | \leq 3 . 5 \} , \mathrm { a n d } D _ { 4 } = \{ | \theta _ { 2 } + 2 \sin ( 2 . 5 \theta _ { 1 } ) | \leq 3 . 5 \} , \theta _ { 2 } = \{ | \theta _ { 2 } - 2 \theta _ { 1 } | \leq 3 . 5 \} , \theta _ { 3 } = \{ | \theta _ { 2 } - 2 \theta _ { 1 } | \} .$ 2.5} as in Figure 7. The 15 nonempty unions $H _ { S } = \bigcup _ { i \in S } D _ { i } , \emptyset \ne S \subseteq \{ 1 , 2 , 3 , 4 \}$ , are considered as candidates in the selection according to (4.4) with $\alpha _ { 0 } = 0 . 1 0$ and $\beta = 0 . 3 0$ The selection results are in Figure 8. As shown, the IMMC estimate of $B e l ( H _ { 4 } )$ (the orange star) is much smaller than that of MIC and the reference (the blue and black stars). While MIC and the reference selected $H _ { 4 }$ , IMMC selected $H _ { 1 , 2 , 3 , 4 } = \cup _ { i = 1 } ^ { 4 } H _ { i }$ as the most eficient based on (4.4).

![](images/0e17f0b04e85ea9653f2f157fac68c38d44c01b1a826392fa0dd0e83473dc840.jpg)  
Figure 8: The Bel-Pl spectrum comparison for the selected hypothesis. (a) MIC and the reference selected $H _ { 4 }$ , while IMMC estimates showed $H _ { 4 }$ as not meeting the constraint. (b) IMMC selected $H _ { 1 , 2 , 3 , 4 } = \cup _ { i = 1 } ^ { 4 } H _ { i }$ . Stars located at the estimated $1 - B e l ( H . )$

## 5 Application to Ovarian Aging

The real-life problem we consider here is ovarian aging. The number of women working with their own expertise has continued growing, far beyond what it was a century ago. As it has become common for the biological window for having children to overlap with the time needed to develop and establish a career, knowing how much ovarian reserve one has left can play a very important role in life planning (ASRM, 2024). Freezing eggs at a younger age and using them for IVF when desired is also widely considered, but the cost, time, concerns about side efects, and personal values mean that this is not an easily accessible path for everyone (ASRM, 2024).

Women are born with 1–2 million eggs, and this number declines over time, leaving only around 1,000 (about 0.1%) by menopause (ACOG and ASRM, 2014). But both the rate of loss and the number present at birth difer from person to person although there are patterns across age groups (Wallace and Kelsey, 2010; de Kat et al., 2016). AMH is a hormone released by small follicles in the ovaries that have just begun to grow, and is currently used as an indirect proxy for the remaining ovarian reserve (Steiner et al., 2017; ASRM, 2020).

We analyze synthetic records constructed to match the age-specific log-AMH curve and between-individual variation reported for the Doetinchem cohort in de Kat et al. (2016).

Our model is as follows. Let $Y _ { i j } ^ { * }$ be the latent log-AMH value measured for woman i at age $t _ { i j }$ . Then

$$
Y _ { i j } ^ { * } = f ( t _ { i j } + c _ { i } ) + a _ { i } + \varepsilon _ { i j } , \qquad \varepsilon _ { i j } \overset { \mathrm { i i d } } { \sim } N ( 0 , \sigma ^ { 2 } ) ,\tag{5.1}
$$

for $c _ { i } \ \in \ [ - 1 5 , 1 5 ]$ 2 $a _ { i } ~ \in ~ [ - 4 , 2 ]$ 2 $\sigma = 0 . 3$ , and $j = 1 , \dotsc , 4$ . Here, f is the population median log-AMH curve over age, and we take the curve estimated from a large cohort as given. There are two individual parameters in this model. The parameter $c _ { i }$ answers the question, “How many years ahead or behind is my AMH trajectory relative to the population reference?” The parameter $a _ { i }$ captures individual diferences in log-AMH levels after accounting for this age shift. AMH measurements have a detection limit, so we treat measurements below this limit as left-censored observations. See the appendix for details of the data, model, and parameter ranges, along with their rationale.

Our goal is to use IM to make inferences for each person under this model. In particular, we want to say how many more years this person’s median AMH will remain above a reference value b. Let $t _ { * }$ be the age at the last measurement. The hypothesis that “it will still be above b after k years” is

$$
H _ { k } : = \{ ( c , a ) : f ( t _ { * } + k + c ) + a \geq \log b \} , \qquad k = 0 , 1 , 2 , \ldots .\tag{5.2}
$$

Since f is decreasing, these hypotheses are nested as $H _ { 0 } \supseteq H _ { 1 } \supseteq H _ { 2 } \supseteq \cdots$ . The question then becomes, “Up to which k can we say this with confidence?” We report the largest k satisfying Bel<sub>z</sub> $\left( H _ { k } \right) \ge 1 - \alpha$ as “at least k years remain” for this person. Conversely, if there is a smallest k satisfying $\mathrm { P l } _ { z } ( H _ { k } ) \le \alpha$ , we report that “fewer than k years remain.” For values of k that meet neither condition, we withhold judgment. We set the reference value to $b = 0 . 9 4 6 ~ \mathrm { n g / m L }$ as justified in the appendix. We focus on the 160 women (hypothetical) whose MLEs imply a positive remaining time.

We wish that we could work with the real data in (de Kat et al., 2016) rather than the synthetic data constructed by digitizing their reported results. (The appendix also shows how our synthesized data shows similar summary behavior as the data reported in de Kat et al. (2016).) Thankfully, however, we know the true generating parameters as we use the synthetic data. This means that we also know each person’s true remaining duration (synthesized), thus enabling us to evaluate the empirical coverage.<sup>4</sup> We first fix $\alpha = 0 . 0 5$ and try the hypothesis probing that interests us most. In this context, this simply means finding the largest k satisfying $B e l ( H _ { k } ) \ge 0 . 9 5$ . For larger values of k, we can also reject those for which $P l ( H _ { k } ) \le 0 . 0 5$ . The results are in Figure 9 and Figure 14 in Appendix. As we can see, MIC inferred more additional time for more women (larger green area), and also ruled out a larger range of remaining durations (larger orange area), and higher consistency with the MC target. As shown in Figure 10, we can see that MIC comes closer to the nominal coverage level and gives narrower confidence intervals. As a result, the withhold rate also decreases. See Table 4 in Appendix for more detailed decision comparisons. For example, at $\alpha = 0 . 0 5$ , the withhold rate, averaged over the 160 women and all evaluated horizons, is 46.0% for MIC and 54.4% for IMMC. Across these horizons, no incorrect definitive judgments occur MIC, IMMC, or the MC reference.

![](images/9508d236d19e662c33c8d7ef4c87b19cb8ad9a299086f2d78f96ff51c359f3a5.jpg)  
Figure 9: Left: The true remaining years for which median AMH stays above the threshold. Center: Inference results from MIC. Right: Inference results from IMMC. A larger green area (closer to the truth) means more informative inference.

Finally, we look at the data for a few (hypothetical) women. Figure 16 in Appendix shows how various hypotheses were accepted or rejected for each woman (each row). As we can see in the left column, MIC’s α-cuts tend to be slightly tighter. Consequently, the right column shows cases where MIC can reject or accept hypotheses that IMMC cannot.

![](images/efb3f90e3a4eaa1afc9c86bbacac4c634e2b32ea9a14897012e583bbc4da8179.jpg)

![](images/8fa208e53e36aa1ae83a0bbfa1b00ce6748be30d71046d39a35bb6868528e671.jpg)

![](images/2c8a0f67ad7fb3830200d8f465ecdf9db3a957d0833177d64468f77431b0fd04.jpg)  
Figure 10: MIC yields tighter α-cuts. Left: Empirical coverage of $C _ { \alpha }$ for the true parameter θ across 160 synthetic women. Middle: The widths of the projected remaining-time intervals. Right: Fraction of women for whom judgment is withheld at each future horizon, with $\alpha = 0 . 0 5$

## 6 Concluding Remarks

We developed a radial transport formulation for learning an MIC sampler without first approximating contour-cut boundaries. Amortization allows this sampler to be reused across datasets, while the Bel–Pl spectrum uses its shellwise information to compare and select interpretable hypotheses. In our experiments, the learned contours and resulting coverage, power, and hypothesis selections are close to the reference results. In the synthetic AMH analysis, MIC gives decisions closer to the MC reference and fewer withheld judgments than the baseline.

There are several reasons why we call our method a “junior AI”. Large AI systems such as ChatGPT, Claude, and Gemini take substantial time and cost to train, but once trained, they produce answers relatively quickly. Likewise in our case, training the network can take time, but once well trained and validated, the network can be used repeatedly on new datasets without additional training. We call it “junior” because its competence is limited to a particular sample size n and a particular likelihood model.

This initial investment of training can pay of when many datasets or many hypotheses must be analyzed, as in automated scientific discovery. For example, suppose thousands of protein variants or vaccine candidates can be evaluated under a common model, and we want to know which of them satisfy a specified statistical criterion. If an amortized inferential machine is already available on the device or a remote server, each new measurement need not trigger another lengthy computation. Likewise, when there is little time to wait, such a junior AI may shine. For example, how long would people be willing to wait for a statistical assessment of the uncertainty behind an alert from a wearable device?

In our work, we established exact results for the radial representation of inner credal distributions and the maximum-entropy characterization (Theorems 1 and 2). On the other hand, our learned transport is an approximation, for which we do not establish finite-sample error bounds in the current version of the manuscript. Instead, amortization allows us to assess the learned sampler’s coverage and power through extensive simulation before it meets real data. We can flag a new dataset whose relationship with its predictive children is quite diferent from what may be thought as typical from the sampler’s perspective.

Using deep learning means there are hyperparameters to tune, such as the learning rate, optimizer, training schedule, and network architecture. While one may begin from our default settings, harder problems may call for diferent ones. In our algorithm, an important hyperparameter is $\gamma _ { h }$ , the weight of the entropy loss. While we fixed $\gamma _ { h } = 0 . 0 1$ across the examples for simple presentation, systematic selection of this weight may ofer further gains in calibration.

## References

American College of Obstetricians and Gynecologists Committee on Gynecologic Practice and Practice Committee of the American Society for Reproductive Medicine (2014). Female age-related fertility decline. Committee Opinion No. 589. Fertility and Sterility 101 (3), 633–634.

Cella, L. and R. Martin (2023). Possibility-theoretic statistical inference ofers performance and probativeness assurances. International Journal of Approximate Reasoning 163, 109060.

Cella, L. and R. Martin (2025). Computationally eficient variational-like approximations

of possibilistic inferential models. International Journal of Approximate Reasoning 186, 109506.

de Kat, A. C., Y. T. van der Schouw, M. J. C. Eijkemans, G. C. Herber-Gast, J. A. Visser, W. M. M. Verschuren, and F. J. M. Broekmans (2016). Back to the basics of ovarian aging: a population-based study on longitudinal anti-Müllerian hormone decline. BMC Medicine 14 (1), 151.

Delattre, S. and N. Fournier (2017). On the Kozachenko–Leonenko entropy estimator. Journal of Statistical Planning and Inference 185, 69–93.

Dempster, A. P. (1967). Upper and lower probabilities induced by a multivalued mapping. The Annals of Mathematical Statistics 38 (2), 325–339.

Ethics Committee of the American Society for Reproductive Medicine (2024). Planned oocyte cryopreservation to preserve future reproductive potential: an Ethics Committee opinion. Fertility and Sterility 121 (4), 604–612.

Ferraretti, A. P., A. La Marca, B. C. J. M. Fauser, B. Tarlatzis, G. Nargund, L. Gianaroli, and ESHRE working group on Poor Ovarian Response Definition (2011). ESHRE consensus on the definition of ‘poor response’ to ovarian stimulation for in vitro fertilization: the Bologna criteria. Human Reproduction 26(7), 1616–1624.

Fisher, R. A. (1930). Inverse probability. Mathematical Proceedings of the Cambridge Philosophical Society 26 (4), 528–535.

Grünwald, P. D. (2024). Beyond Neyman–Pearson: E-values enable hypothesis testing with a data-driven alpha. Proceedings of the National Academy of Sciences 121 (39), e2302098121.

Hannig, J., H. Iyer, R. C. S. Lai, and T. C. M. Lee (2016). Generalized fiducial inference: A review and new results. Journal of the American Statistical Association 111 (515), 1346–1361.

Kim, J., P. S. Zhai, and V. Ročková (2025). Deep generative quantile Bayes. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, Volume 258 of Proceedings of Machine Learning Research, pp. 4141–4149. PMLR.

Koning, N. W. (2023). Post-hoc α hypothesis testing and the post-hoc p-value. arXiv:2312.08040. Version 10, revised December 2, 2025.

Kozachenko, L. F. and N. N. Leonenko (1987). Sample estimate of the entropy of a random vector. Problems of Information Transmission 23 (2), 95–101.

Li, G. and J. Hannig (2020). Deep fiducial inference. Stat 9 (1), e308.

Martin, R. (2015). Plausibility functions and exact frequentist inference. Journal of the American Statistical Association 110 (512), 1552–1561.

Martin, R. (2026a). An eficient Monte Carlo method for valid prior-free possibilistic statistical inference. Journal of the American Statistical Association. Advance online publication.

Martin, R. (2026b). No-prior Bayes reIMagined: probabilistic approximations of possibilistic inferential models. Statistical Science. To appear, with discussion.

Martin, R. (2026c). Possibilistic inferential models: A review. Journal of the American Statistical Association 121 (553), 807–826.

Martin, R. and C. Liu (2015). Inferential models: reasoning with uncertainty. CRC Press.

Martin, R. and J. P. Williams (2025). Asymptotic eficiency of inferential models and a possibilistic Bernstein–von Mises theorem. International Journal of Approximate Reasoning 180, 109389.

Martin, R., J. Zhang, and C. Liu (2010). Dempster–Shafer theory and statistical inference with weak beliefs. Statistical Science 25 (1), 72–87.

Massart, P. (1990). The tight constant in the Dvoretzky–Kiefer–Wolfowitz inequality. The Annals of Probability 18 (3), 1269–1283.

Practice Committee of the American Society for Reproductive Medicine (2020). Testing and interpreting measures of ovarian reserve: a committee opinion. Fertility and Sterility 114 (6), 1151–1157.

Reinagel, P. (2023). Is N-Hacking Ever OK? The consequences of collecting more data in pursuit of statistical significance. PLOS Biology 21 (11), e3002345.

Sejdinovic, D., B. Sriperumbudur, A. Gretton, and K. Fukumizu (2013). Equivalence of distance-based and RKHS-based statistics in hypothesis testing. The Annals of Statistics 41 (5), 2263–2291.

Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton, NJ: Princeton University Press.

Shafer, G. (1982). Belief functions and parametric models. Journal of the Royal Statistical Society: Series B (Methodological) 44 (3), 322–339.

Simmons, J. P., L. D. Nelson, and U. Simonsohn (2011). False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows Presenting Anything as Significant. Psychological Science 22 (11), 1359–1366.

Steiner, A. Z., D. Pritchard, F. Z. Stanczyk, J. S. Kesner, J. W. Meadows, A. H. Herring, and D. D. Baird (2017). Association between biomarkers of ovarian reserve and infertility among older women of reproductive age. JAMA 318 (14), 1367–1376.

Székely, G. J. and M. L. Rizzo (2013). Energy statistics: A class of statistics based on distances. Journal of Statistical Planning and Inference 143 (8), 1249–1272.

Wallace, W. H. B. and T. W. Kelsey (2010). Human ovarian reserve from conception to the menopause. PLoS ONE 5 (1), e8772.

Wang, Y. and V. Ročková (2026). Generative Bayesian inference with GANs. Journal of Machine Learning Research 27 (29), 1–48.

Wasserman, L. A. (1990). Belief functions and statistical inference. Canadian Journal of Statistics 18 (3), 183–196.

## Appendix Table of Contents

A Proof of Theorems 35   
A.1 Proof of Proposition 1 36   
A.2 Proof of Theorem 1 38   
A.3 Proof of Theorem 2 . 40   
B Algorithms 41   
C Experimental Details 44   
C.1 Example 1 . 45   
C.2 Example 2 . 46   
C.3 Example 3 . 46   
C.4 Ovarian Aging Analysis 47   
D Additional Figures and Tables 50

## A Proof of Theorems

In the IM framework, the possibilistic quality of a parameter θ is measured through the IM contour $\pi _ { z _ { n } } ( \theta )$ . While membership in ${ \mathcal { C } } ^ { \mathrm { i n n } } ( \Pi )$ fixes the marginal distribution of contour depth, it leaves the conditional distributions within level sets unspecified.

For $\alpha \in ( 0 , 1 )$ , define a contour-based level set $L _ { z _ { n } , \alpha } : = \{ \theta \in \mathbb { T } : \pi _ { z _ { n } } ( \theta ) = \alpha \}$ . For $L _ { z _ { n } , \alpha }$ s.t. $0 < \mathrm { A r e a } ( L _ { z _ { n } , \alpha } ) < \infty , ^ { 5 }$ we define uniform distribution on $L _ { z _ { n } , \alpha } \colon$ For every Borel set $B \subseteq \mathbb { T }$ , define

$$
\mathrm { U n i f } ( L _ { z _ { n } , \alpha } ) ( B ) = \frac { \mathrm { A r e a } ( B \cap L _ { z _ { n } , \alpha } ) } { \mathrm { A r e a } ( L _ { z _ { n } , \alpha } ) } ,
$$

where uniformity and entropy are understood relative to a declared shell reference measure, taken here to be the Euclidean (d − 1)-dimensional surface measure in the chosen parameterization. Here now on, let $H _ { z _ { n } , \alpha }$ denote entropy with respect to Euclidean surface area on $L _ { z _ { n } , \alpha } { } ^ { 6 }$

## A.1 Proof of Proposition 1

Fix $\varepsilon \in ( 0 , 1 / 2 )$ . We first construct a specific counterexample, a model whose cuts become thin while their minimum-area MLE-centered enclosing ellipses remain fixed. For $\kappa > 0$ (to be chosen below), consider a single observation

$$
Z \sim N _ { 2 } ( m _ { \kappa } ( \theta ) , I _ { 2 } ) , \qquad \theta = ( x , y ) , \qquad m _ { \kappa } ( x , y ) = \bigl ( x , 2 ( 1 + \kappa x ^ { 2 } ) y \bigr ) .
$$

Note that $m _ { \kappa }$ is a smooth map of $\mathbb { R } ^ { 2 }$ onto itself. Thus $\widehat { \theta } _ { z } = m _ { \kappa } ^ { - 1 } ( z )$ , and the normalized likelihood is

$$
R _ { z } ( \theta ) = \exp \Bigl \{ - \frac { 1 } { 2 } \| z - m _ { \kappa } ( \theta ) \| ^ { 2 } \Bigr \} .
$$

Under $P _ { \theta }$ , we have $\lVert Z - m _ { \kappa } ( { \boldsymbol { \theta } } ) \rVert ^ { 2 } \sim \chi _ { 2 } ^ { 2 } .$ , so $R _ { Z } ( \theta ) \sim \mathrm { U n i f } ( 0 , 1 )$ . By (2.1), the exact IM contour thus satisfies $\pi _ { z } ( \theta ) = R _ { z } ( \theta )$ . Take $z = 0$ , for which $\widehat { \theta } _ { 0 } = 0$ . Then we can write the true contour as

$$
\pi _ { 0 } ( x , y ) = \exp \left[ - \frac { x ^ { 2 } + 4 ( 1 + \kappa x ^ { 2 } ) ^ { 2 } y ^ { 2 } } { 2 } \right] .
$$

For $a \in ( 0 , 1 )$ , let $r _ { a } = { \sqrt { - 2 \log a } }$ . The cut $C _ { a } ( 0 ) = m _ { \kappa } ^ { - 1 } \{ \eta : \| \eta \| \le r _ { a } \}$ is compact and connected, with a smooth boundary, due to the smoothness of the map $m _ { \kappa }$ . For each fixed $^ { a , }$ increasing κ makes $C _ { a } ( 0 )$ increasingly thin around the coordinate axes, while its four axis intercepts remain fixed. The cut therefore approaches a cross formed by two line segments along the coordinate axes.

We next show that the minimum-area MLE-centered enclosing ellipses remain fixed as κ increases. Since $1 + \kappa x ^ { 2 } \geq 1$ ，

$$
C _ { a } ( 0 ) \subseteq C _ { a } ^ { \mathrm { o u t } } ( 0 ) : = \{ ( x , y ) : x ^ { 2 } + 4 y ^ { 2 } \leq r _ { a } ^ { 2 } \} .
$$

The cut contains the four points $( \pm r _ { a } , 0 )$ and $( 0 , \pm r _ { a } / 2 )$ . Although the cut becomes increasingly thin, every enclosing ellipse must still contain these four fixed points. Yet these points determine the minimum-area MLE-centered enclosing ellipse. Any centered enclosing ellipse $\{ \theta : \theta ^ { \top } B \theta \leq 1 \}$ , with B symmetric positive definite, must therefore satisfy

$$
B _ { 1 1 } \leq r _ { a } ^ { - 2 } , \qquad B _ { 2 2 } \leq 4 r _ { a } ^ { - 2 } , \qquad \operatorname* { d e t } B \leq B _ { 1 1 } B _ { 2 2 } \leq 4 r _ { a } ^ { - 4 } .
$$

surface area,

$$
H _ { z _ { n } , \alpha } ( K ) = - \int _ { L _ { z _ { n } , \alpha } } \log \biggl ( \frac { d K } { d \mathrm { A r e a } } ( \theta ) \biggr ) K ( d \theta ) .
$$

Its area is at least $\pi r _ { a } ^ { 2 } / 2$ , which is attained by $C _ { a } ^ { \mathrm { o u t } } ( 0 )$ . Equality forces $B = \mathrm { d i a g } ( 1 , 4 ) / r _ { a } ^ { 2 }$ Thus $C _ { a } ^ { \mathrm { o u t } } ( 0 )$ is the unique minimum-area MLE-centered enclosing ellipse.

For independent $A \sim \mathrm { U n i f } ( 0 , 1 )$ and $V \sim \mathrm { U n i f } ( \mathbb { S } ^ { 1 } )$ , the population IMMC law is the distribution of $( r _ { A } V _ { 1 } , r _ { A } V _ { 2 } / 2 ) \stackrel { d } { = } ( X , Y / 2 )$ , with $X , Y \overset { \mathrm { i i d } } { \sim } N ( 0 , 1 )$ , and the equality in distribution follows from $r _ { A } ^ { 2 } \sim \chi _ { 2 } ^ { 2 } .$ . Therefore $Q _ { 0 } ^ { E } = N _ { 2 } ( 0 , \mathrm { d i a g } ( 1 , 1 / 4 ) )$ for every $\kappa > 0$

We now show that the population stitched contour based on $Q _ { \mathrm { 0 } } ^ { \mathrm { M I C } }$ recovers the exact IM contour. We use arclength as the shell reference measure. The smooth family of compact level curves defines a measurable probability kernel $a \mapsto \nu _ { 0 , a }$ . Note that each quadrant of $\partial C _ { a } ( 0 )$ is a monotone graph joining its two axis intercepts. Therefore

$$
4 r _ { a } \leq | \partial C _ { a } ( 0 ) | \leq 6 r _ { a } , \qquad \int _ { 0 } ^ { 1 } \bigl | \log | \partial C _ { a } ( 0 ) | \bigr | \mathrm { d } a < \infty ,
$$

where the integrability follows from $r _ { a } = { \sqrt { - 2 \log a } }$ . Thus the shell reference measure satisfies the entropy integrability condition in Theorem 2. By the definition of $Q _ { \mathrm { 0 } } ^ { \mathrm { M I C } }$ in $( 3 . 1 ) , \pi _ { 0 } ( \Theta ) \sim \mathrm { U n i f } ( 0 , 1 )$ under this law. Thus, for every contour depth $t \in ( 0 , 1 )$ , this distribution assigns probability t to parameter values whose exact IM contour values do not exceed t. Meanwhile, since $R _ { 0 } = \pi _ { 0 }$ , the population stitched contour satisfies

$$
\rho _ { 0 } ^ { \mathrm { M I C } } ( \theta ) = Q _ { 0 } ^ { \mathrm { M I C } } \{ \pi _ { 0 } ( \Theta ) \leq \pi _ { 0 } ( \theta ) \} = \pi _ { 0 } ( \theta ) .
$$

We finally construct a hypothesis whose plausibility is arbitrarily small under the exact IM contour but arbitrarily close to one under the population stitched contour based on $Q _ { 0 } ^ { E }$ Consider the half-space

$$
H = \{ ( x , y ) : y \geq r _ { \varepsilon } / 2 \} .
$$

Note that $\pi _ { 0 } ( x , y ) \leq e ^ { - 2 y ^ { 2 } } \leq \varepsilon \ \mathrm { o n }$ H, with equality at $( 0 , r _ { \varepsilon } / 2 )$ . Since $R _ { 0 } ~ = ~ \pi _ { 0 }$ , the population stitched contour is nondecreasing in $\pi _ { 0 } ( \theta )$ . Thus

$$
\operatorname { P l } _ { \pi _ { 0 } } ( H ) = \varepsilon , \qquad \operatorname { P l } _ { \rho _ { 0 } ^ { E } } ( H ) = Q _ { 0 } ^ { E } \{ \pi _ { 0 } ( \Theta ) \leq \varepsilon \} .
$$

As κ increases, $Q _ { 0 } ^ { E }$ remains fixed while the true contour cuts contract. Consequently, an increasing proportion of its draws have exact IM contour values below ε, making the stitched plausibility of H close to one. In fact, using the independence of X and Y, the bound $\operatorname* { P r } ( | Y | < b ) \leq \sqrt { 2 / \pi } b$ for $b \geq 0$ , and the bound $( 2 \pi ) ^ { - 1 / 2 }$ on the standard normal

density of X, we obtain

$$
\begin{array} { r l } & { 1 - { \mathrm { P l } } _ { \rho _ { 0 } ^ { E } } ( H ) = { \mathrm { P r } } \{ X ^ { 2 } + ( 1 + \kappa X ^ { 2 } ) ^ { 2 } Y ^ { 2 } < r _ { \varepsilon } ^ { 2 } \} } \\ & { \qquad \leq { \mathrm { P r } } \Big ( | Y | < \frac { r _ { \varepsilon } } { 1 + \kappa X ^ { 2 } } \Big ) } \\ & { \qquad \leq \sqrt { \frac { 2 } { \pi } } r _ { \varepsilon } \mathbb { E } \Big [ \frac { 1 } { 1 + \kappa X ^ { 2 } } \Big ] } \\ & { \qquad \leq \frac { r _ { \varepsilon } } { \pi } \int _ { \mathbb { R } } \frac { \mathrm { d } x } { 1 + \kappa x ^ { 2 } } = \frac { r _ { \varepsilon } } { \sqrt { \kappa } } . } \end{array}
$$

Choosing $\kappa \geq r _ { \varepsilon } ^ { 2 } / \varepsilon ^ { 2 } = ( - 2 \log \varepsilon ) / \varepsilon ^ { 2 }$ therefore gives $\operatorname { P l } _ { \rho _ { 0 } ^ { E } } ( H ) \geq 1 - \varepsilon$ . This proves the proposition.

## A.2 Proof of Theorem 1

We first show that every inner radial transport generates an inner credal distribution. With fixed $z ,$ for $T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ , the depth constraint gives

$$
\pi _ { z } \{ T ( U ) \} = 1 - \| U \| \sim \operatorname { U n i f } ( 0 , 1 ) .
$$

Thus $( T _ { \# } F _ { U } ) ( C _ { \alpha } ( z ) ) = 1 - \alpha$ for every $\alpha \in [ 0 , 1 ]$ . Moreover, for any nonempty Borel set $B \subseteq \mathbb { T }$ , the event $T ( U ) \in B$ implies $\pi _ { z } \{ T ( U ) \} \le \Pi _ { z } ( B )$ . Therefore

$$
( T _ { \# } F _ { U } ) ( B ) \leq \mathrm { P r } \big \{ \pi _ { z } \{ T ( U ) \} \leq \Pi _ { z } ( B ) \big \} = \Pi _ { z } ( B ) .
$$

The inequality also holds for $B = \varnothing$ . Hence $T _ { \# } F _ { U } \in \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$

Conversely, we show that every inner credal distribution can be generated by an inner radial transport. The source radius determines the contour depth, while the independent direction provides the randomness needed to sample within the corresponding shell. Take $Q \in \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ . By $( 2 . 4 ) , \pi _ { z } ( \Theta ) \sim \mathrm { U n i f } ( 0 , 1 )$ under $Q .$ . Since $\mathbb { T }$ is standard Borel and $\pi _ { z }$ is measurable, there is a regular conditional probability kernel $K ^ { a } = Q ( { } \cdot { } | \pi _ { z } ( \Theta ) = a )$ such that

$$
Q ( B ) = \int _ { 0 } ^ { 1 } K ^ { a } ( B ) \mathrm { d } a , \qquad K ^ { a } ( L _ { z , a } ) = 1 \quad \mathrm { f o r ~ a l m o s t ~ e v e r y ~ } a ,
$$

for every Borel set $B \subseteq \mathbb { T }$

We next construct a sampler for the conditional distributions $K ^ { a }$ . By the randomization lemma for probability kernels on standard Borel spaces, there exists a jointly measurable

map $G : [ 0 , 1 ] ^ { 2 } \to \mathbb { T }$ such that $G ( a , W ) \sim K ^ { a }$ whenever $W \sim \mathrm { U n i f } ( 0 , 1 )$ . Note that $d \geq 2$ ensures that the first coordinate $V _ { 1 }$ of $V \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ has a continuous CDF $F _ { V _ { 1 } }$ . Thus the source direction supplies the required uniform variable:

$$
W = F _ { V _ { 1 } } ( V _ { 1 } ) \sim \mathrm { U n i f } ( 0 , 1 ) , \qquad W \perp A .
$$

Fix any $\theta _ { * } \in \mathbb { T }$ and define

$$
\begin{array} { r } { T ( u ) = \left\{ \begin{array} { l l } { G ( 1 - \| u \| , F _ { V _ { 1 } } ( u _ { 1 } / \| u \| ) ) , } & { 0 < \| u \| < 1 , } \\ { \theta _ { * } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

This map is measurable, and the second case concerns only an $F _ { U } \mathrm { - n u l l }$ set. Since $U =$ $( 1 - A ) V$ , we have $T ( U ) = G ( A , W )$ almost surely. By the independence of A and $W$ , the conditional law of $T ( U )$ given $A = a$ is $K ^ { a }$ for almost every a. Therefore

$$
( T _ { \# } F _ { U } ) ( B ) = \int _ { 0 } ^ { 1 } K ^ { a } ( B ) \mathrm { d } a = Q ( B )
$$

for every Borel set $B \subseteq \mathbb { T }$ . Moreover, since $K ^ { a } ( L _ { z , a } ) = 1$ for almost every $a ,$

$$
\pi _ { z } \{ T ( U ) \} = A = 1 - \| U \| \quad F _ { U } { \mathrm { - a l m o s t ~ s u r e l y } } .
$$

Thus $T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ and $T _ { \# } F _ { U } = Q$ . Together with the first part of the proof, this establishes assertion (i).

We next show that conditioning the source radius produces the conditional distribution on a cut. For every $T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ and $\alpha \in ( 0 , 1 )$ , the depth constraint implies

$$
\{ \| U \| \leq 1 - \alpha \} = \{ \pi _ { z } ( T ( U ) ) \geq \alpha \} = \{ T ( U ) \in C _ { \alpha } ( z ) \} \quad \mathrm { u p ~ t o ~ a n ~ } F _ { U } \mathrm { - n u l l ~ s e t } .
$$

Thus restricting the source radius selects exactly the draws whose images lie in $C _ { \alpha } ( z )$ These events have probability $1 - \alpha > 0$ . For every Borel set $B \subseteq \mathbb { T }$ , we obtain

$$
\begin{array} { r l } { \left( T _ { \# } \left[ F _ { U } ( \cdot \vert \Vert U \Vert \leq 1 - \alpha ) \right] \right) ( B ) = \displaystyle \frac { \operatorname* { P r } \{ T ( U ) \in B , \Vert U \Vert \leq 1 - \alpha \} } { 1 - \alpha } } & { } \\ { = \displaystyle \frac { ( T _ { \# } F _ { U } ) ( B \cap C _ { \alpha } ( z ) ) } { 1 - \alpha } } & { } \\ { = ( T _ { \# } F _ { U } ) ( B \mid C _ { \alpha } ( z ) ) . } \end{array}
$$

This proves assertion (ii).

Finally, we apply the representation to the maximum-entropy inner credal distribution. The measurable probability kernel $a \mapsto \nu _ { z , a }$ makes $Q _ { z } ^ { \mathrm { M I C } }$ in (3.1) a probability measure. Since $\nu _ { z , a } ( L _ { z , a } ) = 1$ for almost every $^ { a , }$ for every Borel set $D \subseteq [ 0 , 1 ]$ 2

$$
Q _ { z } ^ { \mathrm { M I C } } \{ \pi _ { z } ( \Theta ) \in D \} = \int _ { 0 } ^ { 1 } \nu _ { z , a } \{ \pi _ { z } ( \Theta ) \in D \} \mathrm { d } a = \int _ { 0 } ^ { 1 } \mathbf { 1 } _ { D } ( a ) \mathrm { d } a .
$$

Thus $\pi _ { z } ( \Theta ) \sim \mathrm { U n i f } ( 0 , 1 )$ under $Q _ { z } ^ { \mathrm { M I C } }$ . The same cut-probability and domination arguments used above give $Q _ { z } ^ { \mathrm { M I C } } \in \mathcal { C } ^ { \mathrm { i n n } } ( \Pi _ { z } )$ . By (i), there exists $T ^ { \mathrm { M I C } } \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ such that $( T ^ { \mathrm { M I C } } ) _ { \# } F _ { U } = Q _ { z } ^ { \mathrm { M I C } }$ . This proves the final assertion.

## A.3 Proof of Theorem 2

We first show that fixing the source radius gives the conditional distribution on the corresponding contour shell. Fix $T \in \mathcal { T } _ { \mathrm { i n } } ( \pi _ { z } )$ . For $a \in ( 0 , 1 )$ , write

$$
K ^ { a } : = T _ { \# } \left[ F _ { U } ( \cdot \vert \Vert U \Vert = 1 - a ) \right] = \operatorname { L a w } \{ T ( ( 1 - a ) V ) \} .
$$

The independence of the source radius and direction gives this version of the regular conditional distribution, and $a \mapsto K ^ { a }$ is a measurable probability kernel. By the depth constraint and Fubini’s theorem, $K ^ { a } ( L _ { z , a } ) = 1$ for almost every a. Moreover,

$$
( T _ { \# } F _ { U } ) ( B ) = \int _ { 0 } ^ { 1 } K ^ { a } ( B ) \mathrm { d } a
$$

for every Borel set $B \subseteq \mathbb { T }$ . Thus $K ^ { a }$ is also a version of the conditional distribution of $T ( U )$ given $\pi _ { z } \{ T ( U ) \} = a$ . Likewise, by (3.1), $\nu _ { z , a }$ is the conditional distribution of $Q _ { z } ^ { \mathrm { M I C } }$ on the corresponding shell.

We next express the entropy gap on each shell as a KL divergence. Let $\mu _ { z , a }$ denote the declared shell reference measure. For almost every $^ { a , }$ this measure has finite, positive mass, so

$$
\nu _ { z , a } = \frac { \mu _ { z , a } } { \mu _ { z , a } ( L _ { z , a } ) } , \qquad H _ { z , a } ( \nu _ { z , a } ) = \log \mu _ { z , a } ( L _ { z , a } ) .
$$

If $K ^ { a } \ll \mu _ { z , a }$ , then

$$
\begin{array} { r l r } {  { H _ { z , a } ( \nu _ { z , a } ) - H _ { z , a } ( K ^ { a } ) = \log \mu _ { z , a } ( L _ { z , a } ) + \int _ { L _ { z , a } } \log ( \frac { \mathrm { d } K ^ { a } } { \mathrm { d } \mu _ { z , a } } ) \mathrm { d } K ^ { a } } } \\ & { } & { ~ = \int _ { L _ { z , a } } \log ( \frac { \mathrm { d } K ^ { a } } { \mathrm { d } \nu _ { z , a } } ) \mathrm { d } K ^ { a } } \\ & { } & { ~ = \mathrm { K L } ( K ^ { a } \| \nu _ { z , a } ) . ~ } \end{array}
$$

If $K ^ { a }$ is not much smaller than $\mu _ { z , a }$ , both sides equal $+ \infty$ under the stated entropy convention. Thus the identity holds in the extended-real sense, including when the KL divergence is infinite.

We now integrate over contour depths to obtain the entropy-gap identity. Both $T _ { \# } F _ { U }$ and $Q _ { z } ^ { \mathrm { M I C } }$ have the same uniform contour-depth marginal. Since contour depth is a measurable function of the parameter, the chain rule for relative entropy gives

$$
\begin{array} { r l r } {  { \mathrm { K L } ( T _ { \# } F _ { U } \| Q _ { z } ^ { \mathrm { M I C } } ) = \int _ { 0 } ^ { 1 } \mathrm { K L } ( K ^ { a } \| \nu _ { z , a } ) \mathrm { d } a } } \\ & { } & \\ & { } & { = \int _ { 0 } ^ { 1 } [ H _ { z , a } ( \nu _ { z , a } ) - H _ { z , a } ( K ^ { a } ) ] \mathrm { d } a . } \end{array}
$$

The common contour-depth marginal contributes zero to the KL divergence, so only the diferences within shells remain. The equality also holds when the KL divergence is infinite. Substituting the definition of $K ^ { a }$ proves assertion (i).

We finally show that the integrated conditional entropy is maximized exactly by transports that generate $Q _ { z } ^ { \mathrm { M I C } }$ . The integrability assumption makes R $^ { 1 } _ { ) } H _ { z , a } ( \nu _ { z , a } )$ da finite. Since $H _ { z , a } ( K ^ { a } ) \ \leq \ H _ { z , a } ( \nu _ { z , a } )$ almost everywhere, $\textstyle \int _ { 0 } ^ { 1 } H _ { z , a } ( K ^ { a } )$ da is well defined in $[ - \infty , \infty )$ Therefore assertion (i) yields

$$
\begin{array} { r l } {  { \int _ { 0 } ^ { 1 } H _ { z , a } ( K ^ { a } ) \mathrm { d } a = \int _ { 0 } ^ { 1 } H _ { z , a } ( \nu _ { z , a } ) \mathrm { d } a - \mathrm { K L } ( T _ { \# } F _ { U } \| Q _ { z } ^ { \mathrm { M I C } } ) } } \\ & { \qquad \leq \int _ { 0 } ^ { 1 } H _ { z , a } ( \nu _ { z , a } ) \mathrm { d } a . } \end{array}
$$

By Theorem 1, there exists an inner radial transport that generates $Q _ { z } ^ { \mathrm { M I C } }$ , so this upper bound is attained. Moreover, equality holds if and only if KL $( T _ { \# } F _ { U } \Vert Q _ { z } ^ { \mathrm { M I C } } ) = 0$ , or equivalently, $T _ { \# } F _ { U } = Q _ { z } ^ { \mathrm { M I C } }$ . This proves assertion (ii). Thus the maximizing distribution is uniquely $Q _ { z } ^ { \mathrm { M I C } }$ , although its corresponding transports need not be unique.

## B Algorithms

Here, we present our two main algorithms non-amortized version, amortized version, and the training data generating algorithm for the amortized training.

```latex
Algorithm 1 MIC Sampler Learning.
Input: data $z ,$ rank function $\overline { { R . } }$
Tuning parameters: loss weights $\gamma _ { h } , \gamma _ { f } ,$ and temperature $( \tau )$
Learnable objects: transport map $T _ { \psi } .$
Data: $D _ { \mathrm { t a r g e t } } = \{ ( \vartheta _ { i } , \pi ( \vartheta _ { i } ) ) \} _ { i = 1 } ^ { N _ { \mathrm { t a r g e t } } }$ $\vartheta _ { i } \sim$ Unif(T) computed for rank function $R .$
Transport Map Learning
for $t = 1 , . . . , N _ { \mathrm { i t e r } }$ do
Sample $A _ { i } \sim \mathrm { U n i f } ( 0 , 1 )$ and $V _ { j } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$
Define $U _ { i j } = ( 1 - A _ { i } ) V _ { j }$ for $i \in [ N ]$ and $j \in [ M ]$
Compute $\widetilde { \Theta } _ { i j } = T _ { \psi } ( U _ { i j } )$
Step 1. Sample particles $U _ { 1 } , \dots , U _ { K } \sim F _ { U }$ , and transport by $\widetilde { \Theta } _ { k } = T _ { \psi } ( U _ { k } )$
Compute empirical loss $\begin{array} { r } { \mathcal { L } _ { d } ( \psi ) = \frac { 1 } { M N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } S ( A _ { i } , \widehat { \rho } _ { t } ( \widetilde { \Theta } _ { i j } ) ) } \end{array}$ , where
$\widehat { \rho } _ { t } ( \widetilde { \Theta } _ { i j } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sigma \left( \frac { R ( \widetilde { \Theta } _ { i j } ) - R ( \widetilde { \Theta } _ { k } ) } { \tau } \right)$
Step 2. Compute a shell 1-nn distance by $\begin{array} { r } { R _ { i j } ^ { \psi } = \operatorname* { m i n } _ { t \neq j } \| \widetilde { \Theta } _ { i j } - \widetilde { \Theta } _ { i t } \| } \end{array}$
Compute empirical loss $\mathcal { L } _ { h } ( \psi ) = - \frac { d - 1 } { N M } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } \log R _ { i j } ^ { \psi } ,$
Step 3. Sample $( \vartheta _ { 1 } , \pi ( \vartheta _ { 1 } ) ) , . . . , ( \vartheta _ { K } , \pi ( \vartheta _ { K } ) )$ from $D _ { \mathrm { t a r g e t } }$ . Compute
$\mathcal { L } _ { \mathbf { f } } ( \psi ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } S \big ( \pi ( \vartheta _ { i } ) , \widehat { \rho } _ { t } ( \vartheta _ { i } ) \big ) , w h e r e \widehat { \rho } _ { t } ( \vartheta _ { i } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sigma \bigg ( \frac { R ( \vartheta _ { i } ) - R ( \widetilde { \Theta } _ { k } ) } { \tau } \bigg )$
Step 4. Aggregate the loss $\mathcal { L } ( \psi ) = \gamma _ { h } \mathcal { L } _ { h } ( \psi ) + \mathcal { L } _ { d } ( \psi ) + \gamma _ { f } \mathcal { L } _ { \mathrm { f } } ( \psi )$
Step 5. Update $T _ { \psi }$ based on $\nabla _ { \psi } \mathcal { L } ( \psi )$
Inference
Set $\widehat { Q } ^ { \mathrm { M I C } } : = Q _ { \widehat { \psi } }$ after training. Any of the followings can be conducted individually de
pending on the context:
1. Bayesian-like Sampling $\widetilde { \Theta } _ { 1 } , . . . , \widetilde { \Theta } _ { M } \sim \widehat { Q } ^ { \mathrm { M I C } }$ by $\widetilde { \Theta } _ { j } = T _ { \widehat { \psi } } ( U _ { j } )$ for $U _ { j } \sim F _ { U }$
2. $1 0 0 ( 1 - \alpha ) \%$ Credible Sampling $\widehat { C } _ { \alpha } = \{ T _ { \widehat { \psi } } ( U _ { j } ) , ~ U _ { j } \sim F _ { U } ( \cdot \vert ~ \Vert U \Vert \le 1 - \alpha ) \}$
3. Contour Approximation $\begin{array} { r } { \widehat { \rho } _ { M } ( \theta ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } \big ( R ( \widetilde { \Theta } _ { j } ) \leq R ( \theta ) \big ) } \end{array}$
```

Algorithm 2 Amortized MIC Sampler Learning   
Input: data z, rank function $\overline { { R . } }$   
Tuning parameters: loss weights $\gamma _ { h } , \gamma _ { f } .$ and temperature $( \tau )$   
Learnable objects: transport map $T _ { \psi } .$   
Data: $D _ { \mathrm { t a r g e t } } = \{ ( z _ { i } , \{ \vartheta _ { i b } , \pi _ { z _ { i } } ( \vartheta _ { i b } ) \} _ { b = 1 } ^ { N } ) \} _ { i = 1 } ^ { N } , \vartheta _ { i } \sim$ Unif(T) computed for rank function $R .$   
Transport Map Learning   
for $t = 1 , . . . , N _ { \mathrm { i t e r } }$ do   
Sample ${ \bigl ( } z _ { l } , \vartheta _ { l 1 } , \pi _ { z _ { l } } { \bigl ( } \vartheta _ { l 1 } { \bigr ) } { \bigr ) } , . . . , { \bigl ( } z _ { l } , \vartheta _ { l B } , \pi _ { z _ { l } } { \bigl ( } \vartheta _ { l B } { \bigr ) } { \bigr ) }$ from $D _ { \mathrm { t a r g e t } }$ for $l = 1 , . . . , L .$   
Sample $A _ { i } \sim \mathrm { U n i f } ( 0 , 1 )$ and $V _ { j } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$   
Define $U _ { i j } = ( 1 - A _ { i } ) V _ { j }$ for $i \in [ N ]$ and $j \in [ M ]$   
Compute $\tilde { \Theta } _ { i j l } = T _ { \psi } ( U _ { i j } , z _ { l } )$   
Step 1. Sample particles $U _ { 1 } , \dots , U _ { K } \sim F _ { U } ,$ and transport by $\widetilde { \Theta } _ { k l } = T _ { \psi } ( U _ { k } , z _ { l } )$   
Compute empirical loss $\begin{array} { r } { \mathcal { L } _ { d } ( \boldsymbol { \psi } , z _ { l } ) = \frac { 1 } { M N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } S ( A _ { i } , \widehat { \rho } _ { z _ { l } } ( \widetilde { \Theta } _ { i j l } ) ) } \end{array}$ , where   
$\widehat { \rho } _ { z _ { l } } ( \widetilde { \Theta } _ { i j l } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sigma \left( \frac { R ( \widetilde { \Theta } _ { i j l } , z _ { l } ) - R ( \widetilde { \Theta } _ { k l } , z _ { l } ) } { \tau } \right)$   
Step 2. Compute a shell 1-nn distance by $\begin{array} { r } { R _ { i j l } ^ { \psi } = \operatorname* { m i n } _ { t \neq j } \| \widetilde { \Theta } _ { i j l } - \widetilde { \Theta } _ { i t l } \| } \end{array}$ for each $( i , l )$   
Compute empirical loss:   
$\mathcal { L } _ { h } ( \psi , z _ { l } ) = - \frac { d - 1 } { N M } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } \log R _ { i j l } ^ { \psi } ,$   
Step 3. Compute $\begin{array} { r } { \mathcal { L } _ { \mathrm { f } } ( \psi , z _ { l } ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } S ( \pi _ { z _ { l } } ( \vartheta _ { l b } ) , \widehat { \rho } _ { z _ { l } } ( \vartheta _ { l b } ) ) } \end{array}$ , where   
$\widehat { \rho } _ { z _ { l } } ( \vartheta _ { l b } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sigma \bigg ( \frac { R ( \vartheta _ { l b } , z _ { l } ) - R ( \widetilde { \Theta } _ { k l } , z _ { l } ) } { \tau } \bigg )$   
Step 4. Aggregate the loss $\begin{array} { r } { \mathcal L ( \psi ) = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \left( \gamma _ { h } \mathcal L _ { h } ( \psi , z _ { l } ) + \mathcal L _ { d } ( \psi , z _ { l } ) + \gamma _ { f } \mathcal L _ { \mathrm { f } } ( \psi , z _ { l } ) \right) } \end{array}$   
Step 5. Update $T _ { \psi }$ based on $\nabla _ { \psi } \mathcal { L } ( \psi )$   
Inference   
Set $\widehat { Q } _ { z } ^ { \mathrm { M I C } } : = Q _ { \widehat { \psi } , z }$ after training. Any of the followings can be conducted individually   
depending on the context. For the test point $z _ { 0 }$   
1. Bayesian-like Sampling $\widetilde { \Theta } _ { 1 } , . . . , \widetilde { \Theta } _ { M } \sim \widehat { Q } ^ { \mathrm { M I C } }$ by $\widetilde { \Theta } _ { j } = T _ { \widehat { \psi } } ( U _ { j } , z _ { 0 } )$ for $U _ { j } \sim F _ { U }$   
2. $1 0 0 ( 1 - \alpha ) \%$ Credible Sampling $\widehat { C } _ { \alpha } = \{ T _ { \widehat { \psi } } ( U _ { j } , z _ { 0 } ) , ~ U _ { j } \sim F _ { U } ( \cdot \vert ~ \Vert U \Vert \le 1 - \alpha ) \}$   
3. Contour Approximation $\begin{array} { r } { \widehat { \rho } _ { z _ { 0 } } ( \theta ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } \big ( R ( \widetilde { \Theta } _ { j } , z _ { 0 } ) \leq R ( \theta , z _ { 0 } ) \big ) } \end{array}$

```latex
Algorithm 3 Data Generation for Amortized MIC Sampler Learning
Input: data $z _ { 0 }$ , likelihood function $L , \#$ of parameter samples N, and MC step size K.
Return: $D _ { \mathrm { t a r g e t } } = \left\{ \left( Z _ { \theta _ { i } , k } , \theta _ { j } , \widehat { \pi } _ { Z _ { \theta _ { i } , k } } ( \theta _ { j } ) \right) : i , j \in [ N ] , \ j \neq i , \ k \in [ K ] \right\}$
Data Generation
Define a parameter support (box) T based on $z _ { 0 } .$
Sample $\theta _ { 1 } , \dots , \theta _ { N } \sim \mathrm { U n i f } ( \mathbb { T } )$
For each $i \in [ N ]$ , sample $Z _ { \theta _ { i } , 1 } , \ldots , Z _ { \theta _ { i } , K } \stackrel { \mathrm { i i d } } { \sim } P _ { \theta _ { i } }$ , and compute $\widehat { \theta } ^ { \mathrm { M L E } } ( Z _ { \theta _ { i } , k } ) , k = 1 , \ldots , K$
For each $i , k ,$ define $C _ { i , k } = L \Bigl ( \widehat { \theta } ^ { \mathrm { M L E } } ( Z _ { \theta _ { i } , k } ) , Z _ { \theta _ { i } , k } \Bigr )$ and $R ( \cdot , Z _ { \theta _ { i } , k } ) = L ( \cdot , Z _ { \theta _ { i } , k } ) / C _ { i , k }$
For each $j \in [ N ]$ , define the empirical rank CDF $\begin{array} { r } { \widehat { F } _ { j } ( r ) = K ^ { - 1 } \sum _ { k ^ { \prime } = 1 } ^ { K } \mathbb { I } \left\{ R ( \theta _ { j } , Z _ { \theta _ { j } , k ^ { \prime } } ) \leq r \right\} } \end{array}$
For each $Z _ { \theta _ { i } , k }$ and each $j \neq i ,$ compute $\widehat { \pi } _ { Z _ { \theta _ { i } , k } } ( \theta _ { j } ) = \widehat { F } _ { j } \left\{ R ( \theta _ { j } , Z _ { \theta _ { i } , k } ) \right\}$
Return $D _ { \mathrm { t a r g e t } } = \left\{ \left( Z _ { \theta _ { i } , k } , \theta _ { j } , \widehat { \pi } _ { Z _ { \theta _ { i } , k } } ( \theta _ { j } ) \right) : i , j \in [ N ] , \ j \neq i , \ k \in [ K ] \right\}$
```

## C Experimental Details

We train with Adam and a decaying learning rate initially set to 0.01, using $\gamma _ { h } = 0 . 0 1$ and $\gamma _ { f } = 1$ . We anneal τ to decrease to 0.05. All neural networks use two hidden layers of width 64 with SiLU activations, denoted by MLP. This plain architecture was a deliberate choice so that the performance cannot be attributed to network design. For example, in Examples 1 and 3 (non-amortized versions), the network is not even given MLE information, so the network also needs to find the mode (MLE) during training. Consequently, it took quite a bit of time given the simplicity of the data. A well-designed network, for example, one centered at the MLE such as $T ( u ) \ = \ \hat { \theta } _ { z } ^ { \mathrm { M L E } } + W u + \| u \| \mathrm { M L P } ( u , 1 - \| u \| )$ , where W is trainable, can make learning much faster, with 15-20 times fewer training steps.

For amortized experiments (Example 2 and the ovarian application), to reduce the complexity of learning, we give the MLE information and use standardized parameter coordinates for shell entropy. Amortized samplers use $T ( u , z ) = \hat { \theta } _ { z } ^ { \mathrm { M L E } } + \mathrm { M L P } \big ( [ u , \ 1 \ -$ $\| u \| , ~ e ( z ) ] )$ , where $e ( z )$ is a model-specific encoder of the data, and the output is applied in logit coordinates when the parameter box is constrained.

For computation time, the S-shear model is a good example because its MLE is available in closed form. Once the training data is given, MIC training on a single RTX 3060 GPU takes 8.2 s per 1,000 steps for the non-amortized sampler and 18.5 s for the amortized sampler. Drawing 1,000 samples takes 0.29 ms and 0.16 ms, respectively, on the GPU.

IMMC fitting takes 0.09 s per dataset, and drawing 1,000 samples takes 0.39 ms, both on the CPU. That is, once training or fitting is complete, the sampling times for MIC and IMMC are comparable. For models requiring expensive numerical MLE computation, a major computational cost for both methods is preparing the likelihood-ratio values needed for training or fitting.

## C.1 Example 1

We use the same Gamma and censored-Weibull data as Martin (2026a), which have 20 observations for Gamma and 26 observations including 14 censored times $( \Delta _ { i } = 1 )$ for Weibull. Weibull simulations use the estimated censoring distribution obtained by Kaplan– Meier estimation. For S-shear, we set $n = 1 0 , \sigma = 2 . 5$ , and the observed data are simulated at $\theta \ : = \ : ( 0 , 0 )$ , with $\bar { z } = ( - 0 . 0 1 6 , 0 . 3 6 1 )$ ). The training and evaluation boxes are $[ 1 . 0 0 2 , 3 . 1 9 7 ] \times [ 1 . 5 0 6 , 3 . 7 7 1 ]$ for Gamma, $[ - 2 , 2 ] \times [ 6$ , 10] for Weibull, and $[ - 2 . 6 9 9 3 , 2 . 6 9 9 3 ] \times$ [−4.6993, 4.6993] for S-shear.

MIC is trained for 300,000 optimization steps, using 1,000 uniformly sampled parameter targets per model. Training contour values use 50 Monte Carlo datasets per target for Gamma and Weibull, with likelihood maxima approximated over the target parameter set. IMMC uses the likelihood rank with 100 α levels and 500 Monte Carlo draws per level. For S-shear, both methods use the exact contour for training dataset. For all three datasets, both methods are fitted to the same observed data.

Evaluation uses a separate $5 0 \times 5 0$ grid over each box, with target contours computed using 500 Monte Carlo datasets per point and MLE normalization for Gamma and Weibull, and exactly for S-shear. Each reconstructed contour $\hat { \rho } _ { Q }$ uses $N _ { Q } = 1 0 , 0 0 0$ sampler draws. MSE and maximum absolute error use the full grid. On the other hand, Spearman correlation uses points with $\pi > 0 . 0 0 5$ . The diagnostics $D _ { + }$ and D<sub>−</sub> in Remark 1 use 200 sampler draws, giving a 5% two-sided DKW (Dvoretzky-Kiefer-Wolfowitz) reference threshold of $\sqrt { \log ( 2 / 0 . 0 5 ) / ( 2 \cdot 2 0 0 ) } \approx 0 . 0 9 6$ (Massart, 1990). The $C _ { . 0 5 }$ coverage is the fraction of grid cells with $\pi \geq 0 . 0 5$ occupied by at least one of the $N _ { Q }$ draws. Draws are assigned to their nearest grid point, where those outside the box are excluded.

## C.2 Example 2

We use the same S-shear model and training box as in Example 1. The training bank contains 100 datasets simulated at each of 1,000 uniformly sampled parameters, totaling 100,000 datasets. The phase encoding used is $e ( z ) = ( \sin ( 2 . 5 \bar { z } _ { 1 } ) , \cos ( 2 . 5 \bar { z } _ { 1 } ) )$ . We trained for 3,000 steps. The trained transport is fixed for all evaluations, whereas IMMC is fitted separately to each dataset. Evaluation datasets are independent of the training bank and shared across methods. Both fitted contours use 5,000 reference draws per dataset; the reference contour is exact. The three cases in Figure 3 have sample means $\left( - 0 . 7 5 1 , - 0 . 5 4 3 \right)$ 2 $\left( - 0 . 3 7 7 , 0 . 3 7 1 \right)$ , and (1.126, −0.668). Their grid errors use a $1 4 0 \times 1 4 0 ~ \mathrm { g r i d }$ . Estimating $D _ { \pm }$ uses exact contour values at 12,000 sampler draws.

Coverage uses 200 datasets per phase, and power uses 100 per alternative. Table 2 reports pointwise 95% Wilson intervals for coverage and medians with 10th/90th percentiles for region area. Areas are computed analytically: if $\ell _ { \alpha }$ is the $( \lfloor \alpha M \rfloor + 1 )$ -th smallest relative log-likelihood among the $M = 5 { , } 0 0 0$ reference draws, the fitted region has area $- 2 \pi \sigma ^ { 2 } \operatorname* { m i n } ( \ell _ { \alpha } , 0 ) / n$ . The exact area is $- 2 \pi \sigma ^ { 2 } \log ( \alpha ) / n$ , since m has unit Jacobian.

For the predictive check, we use the S-shear data from Example 1. The larger-variance data are generated as $\bar { z } + 1 . 5 ( z _ { i } - \bar { z } )$ . For the mixture data, we draw five observations from each of two Gaussian components with means $\pm ( 2 . 2 9 , 2 . 2 9 )$ and covariance $I _ { 2 } .$ , then recenter the pooled sample to z¯. The equally weighted mixture has marginal variances $2 . 2 9 ^ { 2 } + 1 \approx \sigma ^ { 2 }$ . All three datasets share the same sample mean and hence the same exact contour.

## C.3 Example 3

We use the third dataset of Example 2, where the fixed data sample z was such as $\bar { z } ~ = ~ ( 1 . 1 2 6 , - 0 . 6 6 8 )$ . MIC is trained the same as in Example 1. IMMC uses the settings of Example 1 on the same data. The Reference uses the exact contour and uniform Euclidean arclength on each level curve in $\theta$ coordinates. Numerical integration uses 200 equal depth bins, three quadrature levels per bin, and 2,048 arclength-equispaced points per level. The fraction of points in H gives the spectrum, from which $Q ( H )$ and mean depth are integrated. For reference belief and plausibility, we compute the supremum by direct numerical maximization since the exact contour is available in closed form.

MIC and IMMC each use 200,000 sampler draws and an independent set of 5,000 reference draws for the stitched contour. $Q ( H )$ and mean depth are estimated by the sample fraction in H and the average stitched depth among those samples; the spectrum uses 200 depth bins. Belief and plausibility are computed from stitched-depth maxima over $H ^ { c }$ and H, using a common pool consisting of both methods’ sampler draws, a $2 0 0 \times 2 0 0$ grid over the box, and the MLE. The selection rule is applied to all 15 candidate unions with $\alpha _ { 0 } = 0 . 1 0$ and $\beta = 0 . 3 0$

## C.4 Ovarian Aging Analysis

Model. The detection limit of the AMH assay used here is $d = 0 . 0 0 1 8 ~ \mathrm { n g / m L }$ de Kat et al. (2016), and values below it are recorded only as “below d.” What we actually observe is $\delta _ { i j } = \mathbf { 1 } \{ Y _ { i j } ^ { * } < \log d \} , Y _ { i j } = \operatorname* { m a x } ( Y _ { i j } ^ { * } , \log d )$ and a censored measurement enters the likelihood through a probability rather than a density. Writing $\mu _ { i j } ( c , a ) = f ( t _ { i j } + c ) + a$ the likelihood for woman i is

$$
L _ { i } ( c , a ) = \prod _ { j : \delta _ { i j } = 0 } \frac { 1 } { \sigma } \phi \left( \frac { Y _ { i j } - \mu _ { i j } ( c , a ) } { \sigma } \right) \prod _ { j : \delta _ { i j } = 1 } \Phi \left( \frac { \log d - \mu _ { i j } ( c , a ) } { \sigma } \right)\tag{C.1}
$$

A censored observation tells us only that “this person’s latent measurement was below the detection limit at this age.” When f is decreasing, with a fixed, this observation’s likelihood contribution increases as c increases (as the ovarian clock moves further ahead) and decreases as c decreases. Once the latent mean is suficiently below the detection limit, this contribution approaches 1 and does not distinguish well how much further below the limit it is.

Rationale for $b = 0 . 9 4 6 ~ \mathrm { n g / m L }$ . This is almost the same as the value of the population curve at age 40 $( e ^ { f ( 4 0 ) } \approx 0 . 9 6 ~ \mathrm { n g / m L } )$ , and is a familiar value because it falls within the AMH cut-of range $( 0 . 5 { - } 1 . 1 \ \mathrm { n g / m L } )$ of the Bologna criteria widely used to predict poor response in IVF (Ferraretti et al., 2011). This is an analytical reference value, however, rather than a clinical decision threshold, and is unrelated to the chance of natural conception. We consider k from 0 to 30 years in one-year increments.

![](images/8162e80594299d3e41b648ebe7c65f7842a61cbf71f47ed52abda5e6fdbde835.jpg)

![](images/209444c93d56de381896b47ca40b46f41c8d523fa5e13ea55f853f47a4873ddb.jpg)

![](images/b8734ec49a69187484ada5c79c681b7a940a21ae1c0f10f966b61342bec953d3.jpg)

![](images/59898e3511efd4419e06e5a1dc13dc0a980ee476e966e6bdd9f22cb0280f8247.jpg)  
Figure 11: The comparison between the original and the synthetic trajectory. Top: The original, copied from de Kat et al. (2016). Bottom: From our synthesized data.

Data. Since individual records from de Kat et al. (2016) are not publicly available, we constructed synthetic longitudinal AMH records using their reported age-specific quantile curves and summary statistics. We represented the population reference curve f by shapepreserving interpolation and drew $c _ { i }$ and $a _ { i }$ independently from mean-zero skew-normal distributions. The curve and distribution parameters were calibrated jointly to broadly reproduce the reported age-specific quantiles and proportions of measurements below the detection limit. Measurements were then generated from (5.1) and censored at the detection limit d. We followed the cohort design in de Kat et al. (2016), matching the reported initial age-group proportions, five-year visit spacing, and participant counts across successive rounds. Of the 3,326 synthetic women, 371 had at least four measurements between ages 20 and 45; we used their first four measurements $( n = 4 )$ . Figure 11 compares the agespecific AMH distribution in the full synthetic cohort with the published curves. In our analysis, we focus on the 160 women out of the 371, whose MLEs imply a positive remaining time.

Computational details. The encoder $e ( z )$ is simply the MLE (rescaled to $[ - 1 , 1 ] ^ { 2 } )$ concatenated with the data z itself; the first-visit age $( t _ { i 1 } - 2 5 ) / 5$ , the four log-AMH measurements standardized by the mean and standard deviation of the training bank (censored values enter as log d), and the four censoring indicators $\delta _ { i j }$ . IMMC also receives these variables. The training bank contains 1,000 parameters sampled uniformly from $[ - 1 5 , 1 5 ] \times [ - 4 , 2 ]$ and 200 datasets per parameter. We train for 40,000 steps. IMMC is fitted separately to each woman. Both methods use 10,000 draws for stitched contours. The MC reference uses 500 simulated datasets at each point of a $1 5 1 \times 6 1$ grid over the parameter box, preserving each woman’s visit ages and applying the censoring rule. Relative-likelihood ranks use a numerically computed box-constrained MLE for each simulated dataset. The reference contour is bilinearly interpolated onto a 301×121 grid; When taking supremum to compute Bel and Pl, the candidate theta pool includes the MLE and points immediately on either side of each hypothesis boundary besides the grid.

## D Additional Figures and Tables

This section provides additional supportive results.

Table 3: (Example 3) Selected unions among all 15 candidates, with $\widehat { \mathrm { B e l } } ( H ) \geq 0 . 9 0$ and ${ \widehat { Q } } ( H ) - { \widehat { \mathrm { B e l } } } ( H ) \leq 0 . 3 0$ . Gap denotes ${ \widehat { Q } } ( H ) - { \widehat { \mathrm { B e l } } } ( H )$ , † indicates infeasibility, and ⋆ marks the selected hypothesis. MIC and Reference selected $H _ { 4 }$ and IMMC selected $H _ { 1 , 2 , 3 , 4 }$
<table><tr><td>Method</td><td> $H _ { S }$ </td><td>Bel</td><td>Pl</td><td> $Q ( H )$ </td><td> $\mathrm { G a p }$ </td><td>π</td></tr><tr><td>Reference</td><td> $H _ { 4 }$ </td><td>0.93172</td><td>1.00000</td><td>0.98540</td><td>0.05368</td><td>0.50702*</td></tr><tr><td>MIC</td><td> $H _ { 4 }$ </td><td>0.93140</td><td>1.00000</td><td>0.95814</td><td>0.02674</td><td>0.51230*</td></tr><tr><td>IMMC</td><td> $H _ { 4 }$ </td><td>0.79780</td><td>1.00000</td><td>0.83044</td><td>0.03264</td><td>0.58206†</td></tr><tr><td>Reference</td><td> $H _ { 1 , 2 , 3 , 4 }$ </td><td>0.99873</td><td>1.00000</td><td>0.99998</td><td>0.00125</td><td>0.50001</td></tr><tr><td>MIC</td><td> $H _ { 1 , 2 , 3 , 4 }$ </td><td>0.99980</td><td>1.00000</td><td>1.00000</td><td>0.00020</td><td>0.49190</td></tr><tr><td>IMMC</td><td> $H _ { 1 , 2 , 3 , 4 }$ </td><td>0.90160</td><td>1.00000</td><td>0.96544</td><td>0.06384</td><td>0.51622*</td></tr></table>

![](images/593d2fd634d02423442e9464936507a5d483e057cb23a372fbd3b053d64c7c4a.jpg)  
Figure 12: (Example 1) Samples from MIC (blue) and IMMC (red) generated with the various source level α. The black dotted lines are estimated target $C _ { \alpha } .$

![](images/27058220146a063130209751fe8b7183ef2b8e6615257e959a21a00eb872a905.jpg)

![](images/765682831eaf4fa0fa046f4cf026a970d1ff481a4a9bca9ad67034b07e4a62c9.jpg)

![](images/f9a475b2814306fa7cf52864dff935c38fcd8c48b6123baf673cc25a2d002fee.jpg)

(d)  
![](images/e1421ca4dade43822e8b305be0da746c8482c2c95b09212571afb3564b9ba857.jpg)

![](images/3d8a6b35b05ba9b6759b5a4ebd8d10b71c9d8e405da79446e1fd672c399d63bf.jpg)

![](images/85be05b2f409a5ed4c9147136a0b87cd64935d50164fbc1d51f7a59ba1c72aeb.jpg)  
Figure 13: (Example 2) The source radius to contour depth alignment (top) and the credal calibration (bottom). The contour depth is computed based on the known true contour and the stitched estimation.

![](images/aa17841e975ec350a68e3c56f60ceb56037d5959ab437cb15975f6be0b989442.jpg)  
Figure 14: (Ovarian Application) Comparison to the MC reference decisions. MIC mostly aligns to the MC reference.

Table 4: (Ovarian Application) Hypothesis decisions at $\alpha = 0 . 0 5$ for the 160 women of the primary cohort. An error is an incorrect acceptance or rejection of $H _ { k }$ . In the $^ { 6 6 } \mathrm { A l l }$ horizons” row, the withhold rate is averaged over the full grid $k = 1 , \ldots , 3 0$ , and the error count records women with at least one error anywhere on that grid. MC ref. is the direct Monte Carlo contour on the parameter grid.
<table><tr><td colspan="3">Withhold rate (%)</td><td colspan="3">Women with an error</td></tr><tr><td>Horizon (years)</td><td>MIC</td><td>IMMC MC ref.</td><td>MIC</td><td>IMMC</td><td>MC ref.</td></tr><tr><td>1</td><td>68.8</td><td>91.2</td><td>68.8</td><td>0 0</td><td>0</td></tr><tr><td>3</td><td>86.9</td><td>100.0</td><td>86.9</td><td>0 0</td><td>0</td></tr><tr><td>5</td><td>88.1</td><td>96.2</td><td>89.4</td><td>0 0</td><td>0</td></tr><tr><td>10</td><td>83.1</td><td>88.1</td><td>76.2</td><td>0 0</td><td>0</td></tr><tr><td>All horizons</td><td>46.0</td><td>54.4</td><td>43.9</td><td>0 0</td><td>0</td></tr></table>

(a) woman A (case 14), p = 0.65  
![](images/c473a549a3277e531a8597ab7ed4ec1806d792d3187b7bbf8b338cedb26c150c.jpg)  
(d) woman A (case 14), p = 0.73

(b) woman B (case 12), p = 0.97  
![](images/de04b6ba00db3f0b2f45f0496e7f9abc5688397a24793e2b0772f4b51ea6c50f.jpg)  
(e) woman B (case 12), p = 0.98

(c) woman C (case 194), p = 0.72  
![](images/9147c3bd0159b7d9237627ba3cad66b5a0d492ab19ce6cf116cdf828f1578182.jpg)  
(f) woman C (case 194), p = 0.65

![](images/e23b311895eb462db2f0807362b1559d077149474bdec47c3829e693e229a89a.jpg)

![](images/4a4346a78199835710f51cb3fdb77c98051421167b165331da4a61303797fc18.jpg)

![](images/25d9124f923948bcaeaef6bdd877d3805328faa0eedb06c7ac0a890583a32afd.jpg)  
Figure 15: (Ovarian Application) Predictive check result for three chosen (simulated) women. It gives an illustration of the predictive check when the data is well explained by the likelihood model and when the amortized sampler is well-trained. Their diagnostics value is far from 0 and the red line (self-children relationship) resembles the other gray lines (children-grand-children relationship).

![](images/6b154bfe9b2b91d77cd54974dedb6507f64e18de5d7ffb89c4d5a6b54fb524c2.jpg)

![](images/c510abc6c01081b670dc7607dc124f40c7f290603d10c89c2089a732e7f87981.jpg)  
Verdicts at alpha=0.05; true remaining time 9.1 y

![](images/d3a652b076841d0e156a5a0bdf66067f770db7e5cc074ce00aa17f315996d025.jpg)

![](images/37969d2c2ad2abf80c376ba0239532f8d5084e996a3c92befc6a98950e220e58.jpg)

![](images/c0acb48f4151dc1ef85d0b858cc1bfc0aa1954117ef9e466e13e11677486e59d.jpg)

Verdicts at alpha=0.05; true remaining time 6.8 y  
![](images/a1d2ace95053fb4877e203c7e31462f240e780b42923defa3d3faa793599a9c2.jpg)  
Figure 16: (Ovarian Application) A few chosen individual examples. Left: Multiple hypotheses and estimated contours. Right: Bel and Pl values showing acceptance/rejection with α = 0.05.