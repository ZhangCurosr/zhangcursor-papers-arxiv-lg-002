# When Does Training on synthetic data provably work?

Leroy Odunlami

Tobias Riedlinger

Hanno Gottschalk

Institute of Mathematics, Technische Universität Berlin Berlin, Germany

## Abstract

High-quality training data are a major bottleneck in machine learning, motivating increasing use of generated data. We study when a binary classifier trained on synthetic data transfers successfully to a real distribution, under a shared conditional label distribution. First, we derive finite-sample excess-error bounds when the real covariate distribution is absolutely continuous with respect to the synthetic distribution under a controlled density ratio. Second, we look at the partially and fully singular case, where the real distribution contains mass outside the synthetic distribution. In this scenario, if the hypothesis class is suficiently expressive, synthetic risk alone cannot uniformly control real-world error: a classifier can be arbitrarily close to optimal on synthetic data while approaching worst-case error on the uncovered real region, even when the distributions are arbitrarily close in Wasserstein distance. For a restricted function class, where the hypotheses and the log-odds of the conditional label probability satisfy a Lipschitz condition, we recover a bound in terms of the probability that a real sample lies outside the reach of the synthetic training data, restoring a transfer guarantee even in the fully singular case. We confirm our results experimentally in controlled low-dimensional settings as well as on an application-scale dataset: Using 32,200 images generated as counterparts to real ImageNet images from four diferent state-of-the art image generators, we show the existence of an adversarial classifier that exhibits up to 0.5% error rate on the synthetic domain while degenerating to 99.25% error on the real domain. At the same time, across a diverse collection of ordinarily trained architectures, synthetic and real test errors remain positively correlated. These results distinguish the worst possible case from typical empirical behavior. Our results suggest that synthetic data should be budgeted at very high sample counts and models trained thereon should be rigorously validated on real data.

## 1 Introduction

Obtaining high-quality training data in suficient quantities is one of the biggest challenges during the development of better machine learning models and can be prohibitively expensive for many use-cases Richter et al. (2016),Cordts et al. (2016). Also, labeling through human workers which is required for many applications is error-prone Northcutt et al. (2021) due to the tedious and time-consuming nature of the task. Rapid progress in generative modeling raises the question whether artificially generated (synthetic) data can replace real data in training, without compromising safety Bauer et al. (2024), Lu et al. (2025). Other potential use cases include addressing class imbalances by augmenting underrepresented categories Sampath et al. (2021), as well as enhancing privacy by substituting personal and sensitive data with synthetic alternatives. The use of synthetic data, however, introduces a domain gap Schwonberg et al. (2023) that must be carefully evaluated in order to determine whether supplementation with synthetic data is a viable approach for a specific task. Despite the surge in synthetic data for training, existing theory focuses on conditions under which synthetic training succeeds, while the question of what can go wrong has received little attention. In order to close this gap in the literature, we will model the situation in the framework of probability theory in order to derive excess-error bounds. We investigate a binary classification problem, where covariates are drawn from a real distribution $P _ { X }$ and a synthetic distribution $Q _ { X }$ . Labels on the other hand will be drawn according to a shared conditional label distribution (concept map), yielding joint distributions � and �. Based on this view point, we identify 3 diferent scenarios that govern synthetic training: The absolutely continuous case, the fully singular case and the mixed case. A Hölder-type bound on the Radon-Nikodym derivative allows us to transfer rates of convergence from the classical case to the absolutely continuous case. In the mixed case, where the synthetic distribution covers the real one only partially, synthetic training leaves the uncovered region unconstrained: the empirical synthetic risk cannot distinguish the Bayes rule from an adversarial one. In the singular case, this construction afects all of $P _ { X }$ and there exists a model that has optimal accuracy under $Q$ and worst possible accuracy under �. This negative result relies on the existence of a hypothesis that switches abruptly between the two supports. A Lipschitz condition on the model class and on the conditional label distribution rules such hypotheses ou and restores a guarantee. Our contributions are the following:

1. We show that if $P _ { X }$ is absolutely continuous with respect to $Q _ { X }$ with density ratio in $L ^ { s } ( Q _ { X } )$ , classical statistical learning rates transfer to synthetic training (Theorem 3).

2. We show that otherwise, under the Lebesgue decomposition theorem, suficiently rich classifiers can achieve �-small error under � while being �-close to worst possible accuracy on the singular component of $P _ { X }$ (Theorem 4).

3. In the fully singular case, we prove the existence of an adversarial classifier which is equal to the Bayes classifier on the support of $Q _ { X }$ and opposite to it on the support of $P _ { X }$ (Corollary 5). Further, we show that Lipschitz constraints on the model class and the concept map bound the excess-error by the probability that a real sample falls outside the reach of the synthetic support (Theorem 7).

4. We observe the above phenomena numerically in both a controlled low-dimensional setting and a realistic high-dimensional setting and provide empirical evidence that partial training success in the sense of Theorem 4 is typical.

The paper is structured as follows: Section 2 gives an overview of related literature. In Section 3 we derive convergence rates for classifiers trained on synthetic data where possible and identify scenarios where this is not possible. Section 4 demonstrates our findings experimentally in controlled low-dimensional settings as well as using state-of-the-art generative models.

## 2 Related Work

Synthetic training data is used across disciplines. Difusion-generated images improve ImageNet classification (Azizi et al., 2023), models trained on ImageNet replicas perform competitively (Sariyildiz et al., 2023), brain MRI segmentation networks can be trained on synthetic scans alone (Billot et al., 2023), and small language models trained on synthetic text data match larger ones on coding benchmarks (Gunasekar et al., 2024). A smaller line of research examines the risks: recursive training on model outputs erases the tails of the original distribution (Shumailov et al., 2024).

The theoretical underpinnings of training under distributional mismatch are rooted in the covariate-shift literature. Shimodaira (2000) shows that density-ratio reweighting is asymptotically optimal under model misspecification. Such reweighting is well-defined only when the test distribution is absolutely continuous with respect to the training distribution, and breaks down otherwise. David et al. (2010) prove impossibility results showing that the covariate-shift assumption alone is insuficient for guaranteeing learning on the target domain. Ben-David et al. (2010) derive the canonical domain-adaptation bound, decomposing target error into source error and an ℋΔℋ-divergence term, refined by Mansour et al. (2009) into the loss-aware discrepancy distance. Chang et al. (2019) bound the real-data risk of a classifier trained on GAN-synthetic samples, decomposing it into the synthetic risk, an ideal-joint-hypothesis error, and a refined hypothesis-pair divergence $d _ { h \Delta h ^ { * } }$ . Xu et al. (2023) bound the synthetic-versus-real generalization gap by feature-distribution discrepancy, conditional-label approximation error, and hypothesis-class misspecification. More recently, Nguyen et al. (2025) bound the real risk of a predictor trained on real and synthetic samples combined, via

output-space discrepancy, local-robustness and a concentration term. Shidani et al. (2026) derive a stability based bound for mixtures in which the excess risk depends on the Wasserstein-2 distance, and use it to characterize an optimal synthetic-to-real ratio, under a Lipschitz hypothesis class. Exact asymptotics are available in restricted settings (El Firdoussi et al., 2025). Unlike prior works whose bounds are mostly phrased in terms of a distance between the synthetic and real distributions, our focus is to show that for unrestricted hypothesis classes vanishing Wasserstein distance is not suficient for convergence, i.e. the erro on the real domain does not converge uniformly.

## 3 Theoretical Results

## 3.1 Notation and Assumptions

Data and distributions. Let $P _ { X }$ and $Q _ { X }$ be probability distributions on $( \mathcal { X } , \mathcal { B } ( \mathcal { X } ) )$ , where $\mathcal { X } \subset \mathbb { R } ^ { d }$ is compact and $\mathcal { B } ( \mathcal { X } )$ denotes the Borel �-algebra on �. We think of � as the data space, and of $P _ { X }$ and $Q _ { X }$ as the true distribution and the generative model. We pair � with the labels $\mathcal { Y } = \{ 0 , 1 \}$ and assume a true conditional distribution $p ( \cdot | x )$ on �, with $\eta ( \boldsymbol { x } ) : = p ( 1 | \boldsymbol { x } )$ . The joint distributions � and � on $\mathcal { X } \times$ � pair each covariate distribution (i.e. distribution on �) with this shared conditional, i.e. for $C \in { \mathcal { B } } ( { \mathcal { X } } \times$ �), $\begin{array} { r } { P ( C ) = \int _ { \mathcal { X } } \sum _ { \boldsymbol { u } \in \mathcal { Y } } \mathbf { 1 } _ { C } ( \boldsymbol { x } , \boldsymbol { y } ) p ( \boldsymbol { y } | \boldsymbol { x } ) P _ { X } ( \mathrm { d } \boldsymbol { x } ) } \end{array}$ , and the same analogously for � with $Q _ { X }$ in place of $P _ { X }$ . The synthetic training set $V _ { n }$ consists of the first � elements of an i.i.d. sequence $\left( X _ { i } , Y _ { i } \right) _ { i \in \mathbb { N } } \sim Q$

Model. We model $p ( y | x )$ by $p _ { \theta } ( y | x ) = y \left( \sigma \circ f _ { \theta } \right) ( x ) + ( 1 - y ) [ 1 - ( \sigma \circ f _ { \theta } ) ( x ) ]$ , with the sigmoid $\sigma ( t ) =$ $\exp ( t ) / ( 1 + \exp ( t ) )$ and its inverse �����. The model is given by a compact $\Theta \subset \mathbb { R } ^ { m }$ and a map $\Phi : \Theta \times$ $\mathcal { X } $ ℝ with $\left\| \Phi \right\| _ { \infty } \le B < \infty .$ , continuous in the first and measurable in the second argument. We write $f _ { \theta } : =$ $\Phi ( \theta , \cdot ) , \mathcal { F } : = \{ f _ { \theta } \mid \theta \in \Theta \} , \gamma : = \sigma ( - B ) \in ( 0 , 1 / 2 )$ and $\eta _ { \theta } ( x ) : = p _ { \theta } ( 1 | x )$

Classifiers and risk. For a measurable classifier $h : \mathcal { X }  \{ 0 , 1 \}$ the classification error (or error rate) is $\operatorname { e r r } _ { P } ( h ) : = P ( \{ h ( X ) \neq Y \} )$ , minimized by the Bayes-optimal classifier $y ^ { * } ( x ) \in \arg$ max $_ y p ( y | x )$ , which need not be of the form $\hat { y } _ { \theta }$ and is not necessarily unique. Our goal is to train $\hat { y } _ { \theta } ( x ) = 1 \big \{ \eta _ { \theta } ( x ) > \frac { 1 } { 2 } \big \}$ by minimizing the empirical risk $\begin{array} { r } { \hat { R } ( \theta , V _ { n } ) = \frac { 1 } { n } \sum _ { ( x , y ) \in V _ { n } } \ell ( \theta , x , y ) } \end{array}$ over $\theta \in \Theta$ , for the negative log-likelihood $\ell ( \theta , x , y ) = - \log p _ { \theta } ( y | x )$ with population risk ${ \dot { R _ { Q } } } ( \theta ) \ddot { : = } \mathbb { E } _ { Q } [ \ell ( \theta , X , Y ) ]$ . We write $\hat { \theta } _ { n } \in \mathrm { a r g }$ mi $\boldsymbol { \mathbf { \rho } } _ { 1 } _ { \theta \in \Theta } \hat { R } ( \theta , V _ { n } )$ $\theta ^ { \ast } \in$ arg mi $1 _ { \theta \in \Theta } R _ { \scriptscriptstyle - Q } ( \theta )$ and $\varepsilon _ { \mathcal { F } } : = \mathbb { E } _ { X \sim Q _ { x } } [ \mathrm { K L } _ { X } ( p ^ { \cdot } \| p _ { \theta ^ { * } } ) ]$ for the approximation floor of $\mathcal { F }$ . Both are attained, since $\begin{array} { r } { \ell ( \theta , x , y ) \in \left[ \dot { \log \frac { 1 } { 1 - \gamma } } , \log \frac { 1 } { \gamma } \right] } \end{array}$ is continuous in �, so $R _ { Q }$ and ${ \hat { R } } ( \cdot , V _ { n } )$ are continuous on the compact Θ.

Divergences and complexity. For distributions $\mu , \nu$ on � with $\mu \ll \nu$ we write $\operatorname { K L } ( \mu \parallel \nu ) : =$ $\begin{array} { r } { \sum _ { u } \mu ( y ) \log ( \mu ( y ) / \nu ( y ) ) } \end{array}$ and $\begin{array} { r } { H ( \mu , \nu ) : = - \sum _ { u } \mu ( y ) \log \nu ( y ) } \end{array}$ with $H ( \mu ) : = H ( \mu , \mu )$ and abbreviate $\tilde { \mathrm { K L } _ { x } } ( \mu \parallel \nu ) : = \mathrm { K L } ( \mu ( \cdot \left| x \right) \parallel \nu ( \cdot \left| x \right) )$ , likewise $H _ { x }$ . Since $\mathcal { Y } = \{ 0 , 1 \}$ , $\begin{array} { r } { \mathrm { K L } _ { x } ( p \parallel p _ { \theta } ) = \mathrm { K L } \big ( \mathrm { B e r n } _ { \eta ( x ) } \parallel \mathrm { B e r n } _ { \eta _ { \theta } ( x ) } \big ) } \end{array}$ for the Bernoulli distribution $\operatorname { B e r n } _ { a }$ with parameter �. For $\mu , \nu$ on $\mathcal { X }$ we use the total variation distance $\mathrm { T V } ( \mu , \nu ) = \operatorname* { s u p } _ { A \in { \mathcal { B } } ( { \mathcal { X } } ) } | \mu ( A ) - \nu ( A )$ | and the Wasserstein-1 distance $\begin{array} { r } { W _ { 1 } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Gamma ( \mu , \nu ) } \mathbb { E } _ { ( x , x ^ { \prime } ) \sim \pi } [ d ( x , x ^ { \prime } ) ] } \end{array}$ where $\Gamma ( \mu , \nu )$ is the set of couplings of � and � and � the Euclidean metric. Finally, Dudley’s entropy integral for the loss class $\mathcal { L } = \{ - \log p _ { \theta } ( \cdot \mid \cdot ) \mid \theta \in \Theta \}$ is $\begin{array} { r } { \mathcal { J } ( \mathcal { L } ) = \int _ { 0 } ^ { \log ( ( 1 - \gamma ) / \gamma ) } \sqrt { \log N ( t , \mathcal { L } , \| \cdot \| _ { \infty } ) } \mathrm { d } t } \end{array}$ , where � denotes the covering number. We assume $\mathcal { J } ( \mathcal { L } ) < \infty$ throughout, Remark 2 verifies this for deep ReLU networks.

Assumptions. Each of our results relies on a subset of the following assumptions:

$\begin{array} { r l r l } { \left( L _ { p } \right) } & { { } } & { } & { { } p ( y | x ) } \end{array}$ is Lipschitz in � with constant $L _ { p } .$

(�<sup>∗</sup>) $f ^ { * } : = 1 \circ \mathfrak { g i t } \circ \eta$ is Lipschitz with constant $L ^ { * } .$

(�) $\exists \kappa > 0$ with $p ( y | x ) > \kappa$ for all $x , y .$ . Set $F : = \log ( ( 1 - \kappa ) / \kappa )$

$( L ^ { \infty } )$ $P _ { X } \ll Q _ { X }$ and $\left\| \mathrm { d } P _ { X } / \mathrm { d } Q _ { X } \right\| _ { L ^ { \infty } ( \mathcal { X } , Q _ { X } ) }$ is finite.

(�<sup>�</sup>) $P _ { X } \ll Q _ { X }$ and $\left\| \mathrm { d } P _ { X } / \mathrm { d } Q _ { X } \right\| _ { L ^ { s } ( \mathcal { X } , Q _ { X } ) }$ is finite, $s \in ( 1 , \infty )$

$\left( L _ { f } \right)$ Every $f \in \mathcal F$ is Lipschitz with $L _ { f } : = \operatorname* { s u p } _ { f \in { \mathcal { F } } } \operatorname { L i p } ( f ) < \infty .$

## 3.2 Classical Setting

The following analysis is organized into 3 cases according to the relationship between the real and synthetic distribution: absolute continuity $( P \ll Q )$ , partial absolute continuity $( P \ll Q )$ and the singular case $( P \perp$ �), cf. Figure 1. As a baseline, we will first look at the standard supervised learning case, where training and test data are sourced from the same distribution. Rates of this type are known in the literature under varying sets of assumptions, cf. Bartlett et al. (2006); Zhang et al. (2024). Here, we prove a bound on the excess classification error that holds for every model class in the sense specified above, with no additional architectural assumptions. Throughout, $\lesssim$ denotes inequality up to a multiplicative constant.

![](images/a1162551f27a34ce30082daf61d8f6ca1b7c75482f072662873749d3ad9f8287.jpg)  
Figure 1: Example for the 3 possible cases (left to right): Absolute continuous case, mixed case and singular case

Theorem 1 (Convergence rate under matched distribution). For any $\delta \in ( 0 , 1 )$ it holds that $\mathrm { e r r } _ { Q } \big ( \hat { y } _ { \hat { \theta } _ { n } } \big ) -$ $\mathrm { e r r } _ { Q } ( y ^ { * } ) \lesssim \sqrt { \varepsilon _ { \mathcal { F } } + r _ { n } ( \delta ) }$ with $r _ { n } ( \delta ) = \Big [ \mathcal { J } ( \mathcal { L } ) + \log ( 1 / \gamma ) \big ( 1 + \sqrt { \log ( 2 / \delta ) } \big ) \Big ] / \sqrt { n }$ with probability at least $^ { 1 - }$ �. (Proof in Appendix A.1)

Remark 2. The quantities in Theorem 1 can be made more concrete for specific architectures. For instance under (�) for truncated deep ReLU networks of depth � and weight bound $W _ { ; }$ , one obtains $\varepsilon _ { \mathcal { F } } \leq \mathrm { d i s t } _ { \infty } ^ { 2 } ( f ^ { * } , \mathcal { F } ) / ( 1 6 \kappa ( 1 - \kappa ) )$ with $f ^ { * } : = 1 \circ \mathbf { g } \mathbf { i } \ t \circ \boldsymbol \eta$ and $\mathcal { J } ( \mathcal { L } ) \lesssim L \sqrt { \log W }$ . (Statement and proof in $\mathrm { A p - }$ pendix A.2)

We now turn to the setting of two diferent distributions $P _ { X }$ and $Q _ { X }$ . In general, the generator will cover a part of the real distribution and miss the rest. Formally, Lebesgue’s decomposition splits $P _ { X }$ into an absolutely continuous and a singular part with regard to $Q _ { X }$ , i.e. $P _ { X } = P _ { X } ^ { a } + P _ { X } ^ { s }$ with $P _ { X } ^ { a } \ll Q _ { X }$ and $P _ { X } ^ { s } \perp Q _ { X }$ . Moreover, there is a partition $\mathcal { X } = M _ { a } \sqcup M _ { s }$ with $Q _ { X } ( M _ { s } ) = 0$ such that $P _ { X } ^ { a } = P _ { X } ( \cdot \cap M _ { a } )$ and $P _ { X } ^ { s } = P _ { X } ( \cdot \cap M _ { s } )$ . Let $P ^ { a } : = P ( \cdot \cap ( M _ { a } \times \mathcal { Y } ) )$ and $P ^ { s } : = P ( \cdot \cap ( M _ { s } \times \mathcal { Y } ) )$ be the corresponding parts of the joint distribution, so that $P = P ^ { a } + P ^ { s }$ . The target error splits¹ accordingly into er $\dot { \tau } _ { P } = \mathrm { e r r } _ { P ^ { a } } + \mathrm { e r r } _ { P ^ { s } }$ . The following sections investigate the diferent scenarios: $P = P ^ { a }$ , i.e. the generator captures the whole real distribution, $P = P ^ { s }$ , i.e. the generator misses the entire distribution and the general case $P = P ^ { a } + P ^ { s }$ where $P ^ { a } , P ^ { s } \ne 0$ . We will see that in the absolutely continuous case (Section 3.3), the derived results carry over. In the non-absolutely continuous case (Sections 3.4 and 3.5), the results do not carry over absent further assumptions.

## 3.3 Case 1: � is absolutely continuous with regard to $Q$

Theorem 3 (Transfer of rates). For each $\delta \in ( 0 , 1 )$ with probability at least $1 - \delta .$

(i) Under $( L ^ { \infty } )$ , it holds that err $_ P \left( \hat { y } _ { \hat { \theta } _ { \ast } } \right) - \mathrm { e r r } _ { P } ( y ^ { \ast } ) \lesssim \sqrt { \varepsilon _ { \mathcal { F } } + r _ { n } ( \delta ) }$

(ii) Under (�<sup>�</sup>), it holds that er $\dot { \cdot } _ { P } \Big ( \hat { y } _ { \hat { \theta } _ { \ast } } \Big ) - \mathrm { e r r } _ { P } ( y ^ { \ast } ) \lesssim ( \varepsilon _ { \mathcal { F } } + r _ { n } ( \delta ) ) ^ { \frac { 1 } { 2 t } } ,$

where $\begin{array} { r } { \frac { 1 } { s } + \frac { 1 } { t } = 1 \ a n d \ r _ { n } ( \delta ) : = \left[ \mathcal { J } ( \mathcal { L } ) + \mathrm { l o g } ( 1 / \gamma ) \Big ( 1 + \sqrt { \log ( 2 / \delta ) } \Big ) \right] / \sqrt { n } . \ \left( P r o o f \ i n \ S e c t i o n \ A . \mathcal { S } \right) } \end{array}$

This shows that under above regularity assumptions the rates do carry cover. Regarding the plausibility of these assumptions, absolute continuity fails as soon as $Q _ { X }$ assigns zero probability to an event where $P _ { X }$ assigns positive probability. This can easily happen, e.g. if the supports of both distributions have diferent intrinsic dimension. For generative models such as variational autoencoders (Kingma & Welling, 2022) and generative adversarial networks (Goodfellow et al., 2014), which map a low-dimensional latent space into the data space, the synthetic support will most likely have a smaller dimension compared to the real one. For models such as normalizing flows (Rezende & Mohamed, 2015), neural ODEs (Chen et al., 2018), difusion models (Ho et al., 2020), and flow matching (Lipman et al., 2023), whose induced distributions may have positive density over the entire space, the situation is less clear. In either case, the absolutely continuous case can hardly be verified in realistic settings. Next, we will look at the general case where $P$ is partially absolutely continuous, partially singular with regard to $Q$

## 3.4 Case 2: � splits into an absolutely continuous and a singular part

Theorem 4 (Higher expressivity of $\mathcal { F }$ leads to higher worst-case error). Let $\varepsilon > 0$ and set $\tilde { y } : = y ^ { * } \mathbf { 1 } _ { M _ { a } } +$ $\hat { y } _ { \mathrm { a d v } } \mathbf { 1 } _ { M _ { \mathrm { s } } }$ with $\begin{array} { r } { \hat { y } _ { \mathrm { a d v } } : = \mathbf { 1 } \big \{ \eta < \frac { 1 } { 2 } \big \} } \end{array}$ . Suppose there exists $\theta \in \Theta$ with $P _ { X } ( \hat { y } _ { \theta } \neq \tilde { y } ) \leq \varepsilon$ and $Q _ { X } ( \hat { y } _ { \theta } \neq \tilde { y } ) \leq \varepsilon$ . Then $\mathrm { e r r } _ { Q } ( \hat { y } _ { \theta } ) - \mathrm { e r r } _ { Q } ( y ^ { \ast } ) \leq \varepsilon$ , but er $\dot { \mathbf { \sigma } } _ { P ^ { s } } ( \hat { y } _ { \theta } ) \geq \operatorname* { m a x } _ { h } \operatorname { e r r } _ { P ^ { s } } ( h ) - \varepsilon$ the maximum being over all measurable classi' fiers ℎ. (Proof in Section $A . 4 )$

Theorem 4 shows the more expressive our function class is with regard to our learning target the higher the worst-case error. The assumptions $P _ { X } ( \hat { y } _ { \theta } \neq \tilde { y } ) \leq \varepsilon$ and $Q _ { X } ( \hat { y } _ { \theta } \neq \tilde { y } ) \leq \varepsilon$ encode the expressivity of the model class. They are strictly weaker than a universal approximation property since they require �-approximation of the given target �̃ only rather than a whole function space. The theorem does not assert that empirical risk minimization will select such a parameter. Minimizing the synthetic risk however, cannot rule out this scenario, not even with an infinite amount of synthetic training data. Since the adversarial and the Bayes optimal behavior difer only on a $Q _ { X } \mathrm { - n u l l }$ set, no criterion computable from synthetic data distinguishes them. Which minimizer a concrete training pipeline returns is determined by inductive biases in the model class such as architecture, initialization as well as optimization algorithm that are independent of the probability distributions of the data.

## 3.5 Case 3: � is completely singular to �

We now consider the case of completely singular distributions. Under $P _ { X } \perp Q _ { X }$ the synthetic distribution leaves no part of the target covered, so the adversarial construction of Theorem 4 afects all of � rather than a part of it. This is due to the freedom of the hypothesis class to behave arbitrarily where no synthetic data lies. No criterion computable from synthetic data can exclude such a classifier, so without further conditions on the hypothesis space or the optimization algorithm, no uniform control over the target error is possible in this regime.

Corollary 5 (Adversarial classifier). Assume $P _ { X } \perp Q _ { X }$ in addition to the hypotheses of Theorem $\it 4 .$ Then there exists $\theta \in \Theta$ with $\mathrm { e r r } _ { Q } ( \hat { y } _ { \theta } ) - \mathrm { e r r } _ { Q } ( y ^ { \ast } ) \leq \varepsilon ,$ but $\mathrm { e r r } _ { P } ( \hat { y } _ { \theta } ) \geq 1 - \mathrm { e r r } _ { P } ( y ^ { \ast } ) - \varepsilon$

Proof. We have $P ^ { a } = 0$ and $P ^ { s } = P$ . Hence, $\mathrm { e r r } _ { P } = \mathrm { e r r } _ { P ^ { s } }$ , and Theorem 4 yields $\theta \in \Theta$ with the stated bound under � and $\operatorname { e r r } _ { P } ( { \hat { y } } _ { \theta } ) \geq \operatorname* { m a x } _ { h } \operatorname { e r r } _ { P } ( h ) - \varepsilon$ . As shown in the proof of Theorem 4, m $\begin{array} { r } { \mathrm { { 1 a x } } _ { h } \mathrm { { e r r } } _ { P } ( h ) = } \end{array}$ $\mathbb { E } _ { P } \left[ \operatorname* { m a x } _ { y } p ( y | X ) \right]$ ]. Since $\mathcal { Y } = \{ 0 , 1 \}$ , we have ma $\begin{array} { r } { \mathrm { x } _ { y } p ( y | x ) + \operatorname* { m i n } _ { y } p ( y | x ) = 1 } \end{array}$ for every �, and $\mathrm { e r r } _ { P } ( y ^ { \ast } ) =$ $\mathbb { E } _ { P } \left[ \operatorname* { m i n } _ { y } { \bar { p ( y | X ) } } \right]$ , so ma $\mathrm { x } _ { h } \mathrm { e r r } _ { P } ( h ) = 1 - \mathrm { e r r } _ { P } ( y ^ { \ast } )$ □

For illustrative purposes, consider the case where the Bayes error is zero, i.e. perfect classification is possible. Then the equation instantiates to err $\dot { \mathbf { \xi } } _ { P } ( \hat { y } _ { \theta } ) \geq 1 - \varepsilon , \mathrm { i . e }$ . the error rate comes arbitrarily close to 1, which is the worst case possible. As already in the previous case, this result rules out uniform consistency under synthetic training. Implications of this will be discussed in section Section 5. Existing bounds are often phrased in terms of Wasserstein distance between the real and the synthetic distribution. A divergence however, e.g. Wasserstein distance may be arbitrarily small even for mutually singular distributions, in which case our negative results apply. Whether the supports are truly disjoint is hard to verify. One can however translate the abstract singularity condition into a more intuitive proxy criterion which can be tested in practice, see Section 4.3.

Remark 6. $P _ { X }$ and $Q _ { X }$ are mutually singular if and only if there exists a measurable $D : \mathcal { X }  \{ 0 , 1 \}$ that separates the two distributions perfectly at the population level, i.e. $P _ { X } ( D = 1 ) = 1$ and $Q _ { X } ( D = 1 ) = 0$

Proof. If $P _ { X }$ and $Q _ { X }$ are mutually singular, there exists $A \in { \mathcal { B } } ( { \mathcal { X } } )$ with $P _ { X } ( A ) = 1$ and $Q _ { X } ( A ) = 0 ,$ , and $D =$ ${ \mathbf { 1 } } _ { A }$ separates perfectly. Conversely, given such a �, the set $A : = D ^ { - 1 } ( \{ 1 \} )$ ) satisfies $P _ { X } ( A ) = 1$ and $Q _ { X } ( A ) =$ 0, which is the definition of mutual singularity. □

Although this is merely a proxy criterion and not a mathematically rigorous check of singularity, it gives us an idea to which extent the distributions appear to be singular from the viewpoint of the neural network. Constraining the hypothesis class through a Lipschitz condition restores upper bounds on the classification error. The intuition behind this construction is simple: functions that cannot change arbitrarily fast will assume similar values on nearby regions.

![](images/40c8c4c5870ce4c2b5cbcfe105b707f00ccf7ab0c4dca8194ffabef848deb8ba.jpg)

![](images/de2a4d527c8bd4fba3d723bce83f21a3816ee9c36ef4281d8979fc501fc41513.jpg)  
Figure 2: Real (left) and synthetic (middle) distribution and convergence of error rates (right). The left and middle plot show a true (blue) distribution with ring-shaped support and a synthetic (red) distribution with disk-shaped support covering the real one. The right plot shows the achieved classification error on a (real/blue) test data set after a full training run on a training set of size n.

Theorem 7 (Lipschitz bound). Assume $\left( \kappa \right) , \left( L _ { f } \right)$ and (�<sup>∗</sup>), and let $y ^ { * } ( x ) = 1 \{ f ^ { * } ( x ) > 0 \}$ . Define $r _ { \theta } ( x ) : = \operatorname * { m i n } \bigr \{ | f _ { \theta } ( x ) | / L _ { f } , | f ^ { * } ( x ) | / L ^ { * } \bigr \} , S : = \{ x \in \operatorname * { s u p p } Q _ { X } | \hat { y } _ { \theta } ( x ) = y ^ { * } ( x ) \}$ and $\begin{array} { r } { U _ { \theta } : = \bigcup _ { x \in S } B ( x , r _ { \theta } ( x ) ) } \end{array}$ Then $\mathrm { e r r } _ { P } ( \hat { y } _ { \boldsymbol { \theta } } ) - \mathrm { e r r } _ { P } ( y ^ { * } ) \leq P _ { X } ( X \notin U _ { \boldsymbol { \theta } } )$ ). In particular, if supp $P _ { X } \subseteq U _ { \theta }$ , then $\mathrm { e r r } _ { P } ( \hat { y } _ { \theta } ) = \mathrm { e r r } _ { P } ( \bar { y } ^ { * } )$ . (Proof in Appendix A.6)

Remark 8. For a bound that is in spirit more similar to Theorem 1 and Theorem 3, cf. Appendix B.3.

## 4 Experiments

We will now check our most important theoretical results through a series of experiments. First, we use simple low-dimensional distributions allowing us to control all relevant quantities. This way, we ensure that all assumptions of the theorems are met and that the underlying efects can be observed. We then scale the experiments up to a realistic synthetic dataset using multiple diferent state-of-the-art generative image models. Our results show that in practice the disjoint support scenario is relevant and synthetically trained models need to be rigorously validated against real data, especially for safety-relevant applications.

## 4.1 Absolutely continuous case

Theorem 3 established a transfer of convergence rates up to a multiplicative factor in the case where $P _ { X } \ll$ $Q _ { X }$ . To validate this in a controlled setting, we drew samples from uniform distributions on the disk and on an annulus contained therein, cf. Figure 2. Subsequently, we trained Multilayer Perceptrons (MLPs) on real and synthetic data and evaluated on a held-out real dataset. The lines in Figure 2 (right) show the median over 100 seeds and the band the interquartile range. For more details confer Appendix B.1.

The plot shows that both models converge at a comparable rate. The real median appears to converge slightly faster, which is to be expected, but the efect is not strongly visible against the seed noise. What’s more distinct is that the synthetic model’s variance is much higher at lower sample size. This is unsurprising given that by coincidence the coverage of the annulus by red samples can be very sparse. Hence, it makes sense that the performance of a model trained on any given low sample count synthetic training set would strongly depend on chance.

Both efects, larger variance at small training set sizes as well slower convergence speed suggest that synthetic data should be budgeted at a much higher sample count than its real counterpart. This makes intuitive sense given that a higher synthetic sample count increases the probability that some of the samples will resemble real data closely enough. Inflating the size of a synthetic training set in order to compensate for these efects should be unproblematic as the appeal of synthetic data is exactly its low marginal cost and better availability. It is however an efect worth accounting for.

![](images/a0ebe833b624e2113221f9aa8c9be37009929e7053704bd8817908eb3dc6db99.jpg)  
Figure 3: Datasets (left) and error rates vs. shift distance (right).

## 4.2 Mixed absolutely continuous and singular case

Theorem 4 and Corollary 5 show that adversarial networks exist, not that training finds them. To measure both the typical and the worst case, we start from identical distributions and gradually move the synthetic support away from the real one, so that the test error can be monitored as a function of displacement alone, cf. Figure 3. We use two paths: vertical displacement, across the decision boundary, and horizontal displacement, along it.

Along the vertical path the error leaves the Bayes rate steeply and approaches random guessing almost as a step function: as the synthetic disk moves up, class A disappears from the training set and the model has no chance of recovering the concept map. Near shift 0.225 the spread is remarkable, with the best runs still at the Bayes error and the worst already at chance. The horizontal path keeps the classes balanced, and accordingly the synthetic median stays well below chance up to the 25th percentile even at full disjointness. The variance across runs nonetheless remains large, even in this benign setting with fixed label map, balanced classes and a linear decision boundary.

The takeaway is that success of synthetic training is possible but depends on the arrangement of the supports relative to the concept map, which in practice is unknown, and that convergence on synthetic data is uninformative thereof. Finally, the worst case is easy to exhibit: in the disjoint horizontal configuration we add real samples with inverted labels to the synthetic training set. The resulting classifier lands within 0.17 percentage points of the Bayes error on synthetic and 0.16 points of the adversarial error on real data. We repeat this construction on realistic data in the next section. For our experimental validation of Theorem 7, see Appendix B.4.

## 4.3 Realistic synthetic data

In order to move from controlled model distributions to real-world distributions, we created a dataset using 4 modern generative models: FLUX.2 [dev] (Black Forest Labs), Stable Difusion 3.5 Large (Stability AI), Qwen-Image (Alibaba), HunyuanImage 3.0 (Tencent). The purpose of this dataset is to isolate the domain gap by equating the image contents semantically. We therefore recreate an existing set of real images rather than generating new scenes from scratch, We selected eight ImageNet categories, four types of big cat and four types of bear, with 1,300 images each, yielding 10,400 real images in total, cf. Figure 4. Each image was captioned by the vision-language model Qwen3'VL'8B'Instruct, and the resulting captions were used as prompts for the generators. Every generator reproduced the full set of 10,400 images, using about 1,075 GPU hours (Nvidia A100 80GB) with the exception of Hunyuan Image 3.0, for which we restricted the generation to a subset of 1000 images due to its high computational cost. Per-generator comparisons involving Hunyuan Image 3.0 should be read with care, since its reduced sample count might be a confounding factor.

![](images/925d08658d4415bd8179d2a7f83f7725dbead2ab9d47475199a3eb6651ecd53a.jpg)  
Figure 4: We used 4 state-of-the-art generative models to recreate 8 complete ImageNet classes. Descriptions of the real images were created using a vision-language model and used as prompts for the generative models. In total, 32,200 images were created, see Section 4.3.

We ran a total of 3 experiments on this data, aimed at Remark 6 and Corollary 5 as well as to relate the typical synthetic error to its real counterpart. Remark 6 noted the equivalence of the existence of a perfect real vs synthetic classifier and mutual singularity of the underlying distributions. In order to assess to what extent this is relevant in practice, we trained a model on synthetic vs real classification. For every generator, we computed the classification accuracy within every ImageNet class and then took the average across all classes. As architecture, we choose V-JEPA ViT-H/16 Bardes et al. (2024), because its training set does not include ImageNet so data leakage is ruled out. The results are depicted in Figure 5 (left). As evident from the histogram, the classifier is able to separate real and synthetic domain at a very high confidence level. Although the classification accuracy is not 100% it still suggests that synthetic data does not truly capture the real distributions support. This is surprising, given that many of the generative images are photorealistic. In contrast to the human eye, it is very easy for a deep neural network to tell real and synthetic images apart, even with state-of-the-art generative models.

Repeating the experiment around the adversarial classifier from Section 4.2 we trained an adversarial classifier on correctly labeled synthetic and inversely labeled real data. The results are depicted in Figure 5 (right). Even though the true theoretical worst case of 100% real error and 0% synthetic error are not reached, this still confirms the existence of strongly adversarial networks. The poor performance on the real domain would be invisible to any model selection process depending only on synthetic data. Note that these models would almost certainly never arise during a normal training run. This merely demonstrates existence. If $P _ { X } \ll Q _ { X }$ , such a network should not persist as the synthetic sample size grows because the correct and the inverted labels would conflict, so no model could achieve low synthetic and high real error simultaneously. This strengthens the evidence that the supports of real and synthetic data might be mostly separated in practice, even with modern generative models. This stresses the importance of rigorous validation with real data.

![](images/72f941fe5c7c63e2b3a2326d0d5d8ae49e50a8449326163b394085965c06c2bf.jpg)

![](images/7db123841ae8079f0406a9f615429577a67b2bf293e768a5705202bf81256f45.jpg)

![](images/477c89c9ee2d1b820401a972b4e93fa3f1f7645d44e6b43a923a3a4352cdee30.jpg)  
Figure 5: The left plot shows the real vs synthetic classification accuracy of a V-JEPA based classifier applied to our dataset. The middle plot shows existence of adversarial classifiers. The right plot shows correlation of real and synthetic error.

To conclude on an optimistic note, in practice the real and synthetic error appear to be somewhat predictive of one another. The worst case which is the adversarial example is possible but unlikely and this seems to hold irrespective of the specific architecture. To check this we tested a zoo of 90 diferent architectures from 14 families. The tested models include variants of vanilla CNNs Lecun et al. (1998), ResNet He et al. (2016) and Inception Szegedy et al. (2015), where hyperparameters like kernel size or number of layers were varied, cf. Appendix B.2 All models were trained on synthetic data alone and evaluated on held-out test sets like before. Within each generator the two errors are strongly linearly related $( r ^ { 2 }$ up to 0.92 (Qwen Image), � = 90 each). That means that in practice for modern generative models the synthetic error at least directionally indicates the magnitude of the real test error. Given that our prior experiments all provided evidence for the fact that synthetic and real data probably have mostly separated supports this fact cannot be explained through the data alone. This is only explainable due to inductive biases of deep neural networks.

## 5 Discussion

In this paper, we have proposed a theoretical framework for supervised learning under a domain gap between a synthetic and real covariate distribution. Our analysis has revealed three qualitative regimes distinguished by the relationship between the measures: absolute continuity, partial and full singularity. While our positive result in the absolutely continuous setting shows that learning rates directly transfer from synthetic to real data, the underlying assumptions cannot be rigorously verified in practice. At the same time, our experiments provide evidence that real and synthetic images can be separated with almost perfect accuracy, which suggests that the (partially) singular regime is more likely to hold in practice for the four tested generators. In this scenario, the domain gap persists even in the large-sample limit. Despite the lack of provable guarantees in this case, we typically observe a linear correlation between real and synthetic test error. Our experiments further inform how data should be allocated. Because synthetic training exhibit higher variance and slower convergence, it is advisable to use substantially larger sample sizes than real data. Real data, meanwhile, plays an irreplaceable role in validation and model selection, since synthetic test error is an unreliable proxy for real performance. Why the typical case is substantially better behaved than the worst case remains an open question. It connects to the more general question of why neural networks generalize far better than worst-case theory predicts, and deserves further investigation.

## Acknowledgement

The research leading to these results is funded by the German Federal Ministry for Economic Afairs and Energy within the project “NXT GEN AI METHODS – Generative Methoden für Perzeption, Prädiktion und Planung”. The authors would like to thank the consortium for the successful cooperation.

## References

Shekoofeh Azizi, Simon Kornblith, Chitwan Saharia, Mohammad Norouzi, and David J. Fleet. Synthetic data from difusion models improves imagenet classification. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=DlRsoxjyPm.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video, 2024. URL https:// arxiv.org/abs/2404.08471.

Peter L. Bartlett, Michael I. Jordan, and Jon D. McAulife. Convexity, classification, and risk bounds. Journal of the American Statistical Association, 101(473):138–156, 2006.

André Bauer, Simon Trapp, Michael Stenger, Robert Leppich, Samuel Kounev, Mark Leznik, Kyle Chard, and Ian Foster. Comprehensive exploration of synthetic data generation: A survey, 2024. URL https://arxiv.org/abs/2401. 02524.

S. Ben-David, J. Blitzer, and K. et al. Crammer. A theory of learning from diferent domains. Machine Learning, 79:151–175, 2010. doi: 10.1007/s10994-009-5152-4. URL https://doi.org/10.1007/s10994-009-5152-4.

Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher, Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, and Juan Eugenio Iglesias. Synthseg: Segmentation of brain mri scans of any contrast and resolution without retraining. Medical Image Analysis, 86:102789, 2023. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2023. 102789. URL https://www.sciencedirect.com/science/article/pii/S1361841523000506.

Fu-Chieh Chang, Hao-Jen Wang, Chun-Nan Chou, and Edward Y. Chang. G2R bound: A generalization bound for supervised learning from GAN-synthetic data. arXiv preprint arXiv:1905.12313, 2019. URL https://arxiv.org/abs/ 1905.12313.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary diferential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https:// proceedings.neurips.cc/paper\_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf.

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3213–3223, 2016. doi: 10.1109/ CVPR.2016.350.

Shai Ben David, Tyler Lu, Teresa Luu, and David Pal. Impossibility theorems for domain adaptation. In Yee Whye Teh and Mike Titterington (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pp. 129–136, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010. PMLR. URL https://proceedings.mlr.press/v9/david10a.html.

Luc Devroye, László Györfi, and Gábor Lugosi. A Probabilistic Theory of Pattern Recognition. Stochastic Modelling and Applied Probability. Springer New Yorks, NY, 1996. ISBN 978-0-387-94618-4. doi: 10.1007/978-1-4612-0711-5.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. URL https://arxiv.org/abs/2010. 11929.

Aymane El Firdoussi, Mohamed El Amine Seddik, Soufiane Hayou, Reda Alami, Ahmed Alzubaidi, and Hakim Hacid. Maximizing the potential of synthetic data: Insights from random matrix theory. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 46780–46815, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/7416573f05b50beac6d0aef3abc805c0-Paper-Conference. pdf.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger (eds.), Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014. URL https://proceedings.neurips.cc/paper\_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Cesar Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Conti Kaufmann, Gustavo Henrique de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Behl, Xin Wang, Sebastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need, 2024. URL https://openreview.net/forum?id=Fq8tKtjACC.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016. doi: 10.1109/ CVPR.2016.90.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/4c5bcfec 8584af0d967f1ab10179ca4b-Paper.pdf.

Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7132–7141, 2018. doi: 10.1109/CVPR.2018.00745.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2261–2269, 2017. doi: 10.1109/CVPR.2017.243.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. URL https://arxiv.org/abs/1312.6114.

Y. Lecun, L. Bottou, Y. Bengio, and P. Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi: 10.1109/5.726791.

Duo Li, Jie Hu, Changhu Wang, Xiangtai Li, Qi She, Lei Zhu, Tong Zhang, and Qifeng Chen. Involution: Inverting the inherence of convolution for visual recognition. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12316–12325, 2021. doi: 10.1109/CVPR46437.2021.01214.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=PqvMRDCJT9t.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9992–10002, 2021. doi: 10.1109/ICCV48922.2021.00986.

Yingzhou Lu, Lulu Chen, Yuanyuan Zhang, Minjie Shen, Huazheng Wang, Xiao Wang, Capucine van Rechem, Tianfan Fu, and Wenqi Wei. Machine learning for synthetic data generation: A review, 2025. URL https://arxiv. org/abs/2302.04062.

Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Domain adaptation: Learning bounds and algorithms. In Proceedings of The 22nd Annual Conference on Learning Theory (COLT 2009), Montréal, Canada, 2009. URL http://www.cs.nyu.edu/\~mohri/postscript/nadap.pdf.

Lan-Cuong Nguyen, Quan Nguyen-Tri, Bang Tran Khanh, Dung D. Le, Long Tran-Thanh, and Khoat Than. Provably improving generalization of few-shot models with synthetic data. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 46178–46195. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/nguyen25h.html.

Curtis Northcutt, Anish Athalye, and Jonas Mueller. Pervasive label errors in test sets destabilize machine learning benchmarks. In J. Vanschoren and S. Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/ paper\_files/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf.

Yongming Rao, Wenliang Zhao, Zheng Zhu, Jie Zhou, and Jiwen Lu. Gfnet: Global filter networks for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):10960–10973, 2023. doi: 10.1109/TPAMI.2023.3263824.

Ievgen Redko, Amaury Habrard, and Marc Sebban. Theoretical analysis of domain adaptation with optimal transport. In Machine Learning and Knowledge Discovery in Databases (ECML PKDD), pp. 737–753. Springer, 2017.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 1530–1538, Lille, France, 07–09 Jul 2015. PMLR. URL https://proceedings.mlr. press/v37/rezende15.html.

Stephan R. Richter, Vibhav Vineet, Stefan Roth, and Vladlen Koltun. Playing for data: Ground truth from computer games. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (eds.), Computer Vision – ECCV 2016, pp. 102–118, Cham, 2016. Springer International Publishing. ISBN 978-3-319-46475-6.

David E. Rumelhart, Geofrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986. ISSN 1476-4687. doi: 10.1038/323533a0. URL https://doi.org/10.1038/ 323533a0.

Vignesh Sampath, Iñaki Maurtua, Juan José Aguilar Martín, and Aitor Gutierrez. A survey on generative adversarial networks for imbalance problems in computer vision tasks. Journal of Big Data, 8(1), 2021. ISSN 2196-1115. doi: 10.1186/s40537-021-00414-0. URL https://doi.org/10.1186/s40537-021-00414-0.

Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4510–4520, 2018. doi: 10.1109/CVPR.2018.00474.

Mert Bulent Sariyildiz, Karteek Alahari, Diane Larlus, and Yannis Kalantidis. Fake it till you make it: Learning transferable representations from synthetic imagenet clones. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8011–8021, 2023. doi: 10.1109/CVPR52729.2023.00774.

Manuel Schwonberg, Joshua Niemeijer, Jan-Aike Termöhlen, Nico M Schmidt, Hanno Gottschalk, Tim Fingscheidt, et al. Survey on unsupervised domain adaptation for semantic segmentation for visual perception in automated driving. IEEE Access, 11:54296–54336, 2023.

Jian Shen, Yanru Qu, Weinan Zhang, and Yong Yu. Wasserstein distance guided representation learning for domain adaptation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

Amitis Shidani, Tyler Farghly, Yang Sun, Habib Ganjgahi, and George Deligiannidis. Beyond real data: Synthetic data through the lens of regularization, 2026. URL https://arxiv.org/abs/2510.08095.

Hidetoshi Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90(2):227–244, 2000. ISSN 0378-3758. doi: https://doi.org/10.1016/ S0378-3758(00)00115-4. URL https://www.sciencedirect.com/science/article/pii/S0378375800001154.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. Ai models collapse when trained on recursively generated data. Nature, 631:755–759, 2024. doi: 10.1038/s41586-024-07566-y. URL https://www.nature.com/articles/s41586-024-07566-y.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1–9, 2015. doi: 10.1109/CVPR.2015.7298594.

Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. Mlp-mixer: An all-mlp architecture for vision. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 24261–24272. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89 fe-Paper.pdf.

Martin J. Wainwright. High'Dimensional Statistics: A Non'Asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2019.

Saining Xie, Alexander Kirillov, Ross Girshick, and Kaiming He. Exploring randomly wired neural networks for image recognition. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1284–1293, 2019. doi: 10.1109/ICCV.2019.00137.

Shirong Xu, Will Wei Sun, and Guang Cheng. Utility theory of synthetic data generation. arXiv preprint arXiv:2305.10015, 2023. URL https://arxiv.org/abs/2305.10015.

Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions, 2016. URL https://arxiv.org/ abs/1511.07122.

Zihan Zhang, Lei Shi, and Ding-Xuan Zhou. Classification with deep neural networks and logistic loss. Journal of Machine Learning Research, 25(125):1–117, 2024. URL http://jmlr.org/papers/v25/22-0049.html.

## A Supplementary Proofs

## A.1 Proof of Theorem 1

Proof of theorem 1. We will write $R = R _ { Q }$ for this proof. We will first bound the expected KL divergence. It can be rewritten in terms of the risk functional as

$$
\begin{array} { r l } { \mathbb { E } _ { Q _ { X } } [ \mathrm { K L } _ { X } ( p \| \ p _ { \theta } ) ] = \mathbb { E } _ { Q _ { X } } [ H _ { X } ( p , p _ { \theta } ) ] - \mathbb { E } _ { Q _ { X } } [ H _ { X } ( p ) ] } \\ & { = \mathbb { E } _ { Q _ { X } } [ - p ( 0 | X ) \log p _ { \theta } ( 0 | X ) - p ( 1 | X ) \log p _ { \theta } ( 1 | X ) ] - \mathbb { E } _ { Q _ { X } } [ H _ { X } ( p ) ] } \\ & { = \mathbb { E } _ { Q } [ \ell ( \theta , X , Y ) ] - \mathbb { E } _ { Q _ { X } } [ H _ { X } ( p ) ] } \\ & { = R ( \theta ) - \mathbb { E } _ { Q _ { X } } [ H _ { X } ( p ) ] } \\ & { = \big [ R ( \theta ^ { * } ) - \mathbb { E } _ { Q _ { X } } [ H _ { X } ( p ) ] \big ] + [ R ( \theta ) - R ( \theta ^ { * } ) ] } \\ & { = \mathbb { E } _ { Q _ { X } } [ \mathrm { K L } _ { X } ( p \| \ p _ { \theta ^ { * } } ) ] + [ R ( \theta ) - R ( \theta ^ { * } ) ] } \\ & { = \underbrace { \mathbb { E } _ { \mathcal { F } } } _ { \Im \Re } + [ R ( \theta ) - R ( \theta ^ { * } ) ] } \\ & { \xrightarrow [ \Im \Re ] { = 0 } \left( R ( \theta ) - R ( \theta ^ { * } ) \right] } \end{array}
$$

It remains to bound $R { \bigl ( } { \hat { \theta } } _ { n } { \bigr ) } - R ( \theta ^ { * } )$ . Consider the following decomposition:

$$
\begin{array} { r } { R \big ( \widehat { \theta } _ { n } \big ) - R ( \theta ^ { * } ) = \underbrace { \Big \{ R \big ( \widehat { \theta } _ { n } \big ) - \widehat { R } \big ( \widehat { \theta } _ { n } , V _ { n } \big ) \Big \} } _ { T _ { 1 } } + \underbrace { \Big \{ \widehat { R } \big ( \widehat { \theta } _ { n } , V _ { n } \big ) - \widehat { R } ( \theta ^ { * } , V _ { n } ) \Big \} } _ { T _ { 2 } \le 0 } + \underbrace { \Big \{ \widehat { R } ( \theta ^ { * } , V _ { n } ) - R ( \theta ^ { * } ) \Big \} } _ { T _ { 3 } } . } \end{array}
$$

Using Lemma 11 this will give us a bound of the form

$$
\mathrm { e r r } _ { Q } \left( \hat { y } _ { \hat { \theta } _ { n } } \right) - \mathrm { e r r } _ { Q } ( y ^ { \ast } ) \lesssim \sqrt { ( \varepsilon _ { \mathcal { F } } + T _ { 1 } + T _ { 3 } ) } .\tag{1}
$$

We have $\begin{array} { r } { T _ { 1 } = R \big ( \widehat \theta _ { n } \big ) - \widehat { R } \big ( \widehat \theta _ { n } , V _ { n } \big ) \leq \operatorname* { s u p } _ { \theta \in \Theta } \big | R ( \theta ) - \widehat { R } ( \theta , V _ { n } ) \big | = \left\| Q _ { n } - Q \right\| _ { \mathcal { L } } , } \end{array}$ where $\begin{array} { r } { Q _ { n } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { ( X _ { i } , Y _ { i } ) } } \end{array}$ is the empirical measure of $V _ { n }$ and $\begin{array} { r } { \| \mu \| _ { \mathcal { L } } : = \operatorname* { s u p } _ { \theta \in \Theta } \left| \int \ell ( \theta , \cdot ) \mathrm { d } \mu \right| } \end{array}$ . Since $\theta \mapsto { \bar { \ell } } ( \theta , x , y )$ is continuous and Θ is compact, all suprema over Θ in this proof coincide with suprema over a countable dense subset and are therefore measurable. By Theorem 4.10 of Wainwright (2019) applied to $\mathcal { L }$ with uniform bound $b = \log ( 1 / \gamma )$ for any $\epsilon \geq 0$ we have

$$
\left\| Q _ { n } - Q \right\| _ { \mathcal { L } } \leq 2 \mathcal { R } _ { n } ( \mathcal { L } ) + \epsilon\tag{2}
$$

with �-probability at least $\begin{array} { r } { 1 - \exp \left( - \frac { n \epsilon ^ { 2 } } { 2 b ^ { 2 } } \right) } \end{array}$ , where $\mathcal { R } _ { n } ( \mathcal { L } )$ is the Rademacher complexity

$$
\mathcal { R } _ { n } ( \mathcal { L } ) = \mathbb { E } _ { V _ { n } , \varepsilon } \left[ \operatorname* { s u p } _ { \theta \in \Theta } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \varepsilon _ { i } \ell ( \theta , x _ { i } , y _ { i } ) \right| \right] ,
$$

and $\varepsilon _ { i }$ are i.i.d. Rademacher variables independent of $V _ { n }$ . Setting exp $\begin{array} { r } { \left( - \frac { n \epsilon ^ { 2 } } { 2 b ^ { 2 } } \right) = \delta / 2 } \end{array}$ and solving for � gives $\epsilon =$ log $( 1 / \gamma ) \sqrt { 2 \log ( 2 / \delta ) / n }$ . To bound $\mathscr { R } _ { n } ( \mathcal { L } )$ , we apply Theorem 5.22 of Wainwright (2019), which establishes that for any zero-mean sub-Gaussian process $\left\{ X _ { \theta } \right\} _ { \theta \in \mathbb { T } }$ with respect to a pseudometric $\rho _ { X }$ , and any $\alpha \in$ $[ 0 , D ]$ , it holds that:

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { \theta , \tilde { \theta } \in \mathbb { T } } \left( X _ { \theta } - X _ { \tilde { \theta } } \right) \right] \leq 2 \mathbb { E } \left[ \operatorname* { s u p } _ { \vartheta , \vartheta ^ { \prime } \in \mathbb { T } } \left( X _ { \vartheta } - X _ { \vartheta ^ { \prime } } \right) \right] + 3 2 \mathcal { J } ( \alpha / 4 ; D ) ,\tag{3}
$$

where $D : = \dim ( \mathbb { T } )$ is the $\rho _ { X } { \mathrm { - d i a m e t e r } }$ and $\alpha / 4$ and � denote the integration limits. The construction that follows is carried out for a fixed realisation of $V _ { n }$ , so that loss values are deterministic and only � is random. The expectation over $V _ { n }$ is taken at the end. Define the process

$$
Z _ { \theta } : = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \varepsilon _ { i } \ell ( \theta , x _ { i } , y _ { i } ) ,
$$

which is zero-mean since $\mathbb { E } [ \varepsilon _ { i } ] = 0$ . Define the empirical $L ^ { 2 } .$ -pseudometric on Θ by

$$
\rho _ { X } ( \theta , \theta ^ { \prime } ) : = \| \ell _ { \theta } - \ell _ { \theta ^ { \prime } } \| _ { L ^ { 2 } ( Q _ { n } ) } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \ell ( \theta , x _ { i } , y _ { i } ) - \ell ( \theta ^ { \prime } , x _ { i } , y _ { i } ) \right) ^ { 2 } } .
$$

This is only a pseudometric, since distinct parameters can induce identical losses on the sample. We verify that $\{ Z _ { \theta } \}$ is sub-Gaussian with respect to $\rho _ { X }$ . Writing $a _ { i } : = \ell ( \theta , x _ { i } , y _ { i } ) - \ell ( \theta ^ { \prime } , x _ { i } , y _ { i } )$ , the $a _ { i }$ are deterministic for fixed $V _ { n } .$ , so $\varepsilon _ { i } a _ { i } \in [ - | a _ { i } | , + | a _ { i } ] ]$ is a deterministic bound and Hoefding’s lemma gives $\mathbb { E } _ { \varepsilon } \big [ e ^ { t \varepsilon _ { i } a _ { i } } \big ] \leq$ exp $\left( t ^ { 2 } a _ { i } ^ { 2 } / 2 \right)$ . By independence of the $\varepsilon _ { i } .$

$$
\mathbb { E } _ { \varepsilon } \left[ e ^ { \lambda ( Z _ { \theta } - Z _ { \theta ^ { \prime } } ) } \right] = \prod _ { i = 1 } ^ { n } \mathbb { E } _ { \varepsilon } \left[ e ^ { \frac { \lambda } { \sqrt { n } } \varepsilon _ { i } a _ { i } } \right] \le \prod _ { i = 1 } ^ { n } e ^ { \frac { \lambda ^ { 2 } a _ { i } ^ { 2 } } { 2 n } } = e ^ { \frac { \lambda ^ { 2 } \rho _ { X } ^ { 2 } \left( \theta , \theta ^ { \prime } \right) } { 2 } } .
$$

Hence $\{ Z _ { \theta } : \theta \in \Theta \}$ is sub-Gaussian with respect to $\rho _ { X }$ in the sense of Definition 5.16 of Wainwright (2019). The diameter satisfies $\begin{array} { r } { D : = \operatorname* { s u p } _ { \theta , \theta ^ { \prime } \in \Theta } \rho _ { X } ( \theta , \theta ^ { \prime } ) \leq \operatorname* { s u p } _ { \theta , \theta ^ { \prime } } \left. \ell _ { \theta } - \ell _ { \theta ^ { \prime } } \right. _ { \infty } \leq \log ( ( 1 - \gamma ) / \gamma ) } \end{array}$ . Since $\rho _ { X } ( \theta , \theta ^ { \prime } ) \leq$ $\left\| \ell _ { \theta } - \ell _ { \theta ^ { \prime } } \right\| _ { \infty }$ , every �-cover of $\mathcal { L }$ in $\left\| \cdot \right\| _ { \infty }$ induces a �-cover of Θ in $\rho _ { X }$ , so the $\rho _ { X }$ -covering numbers are bounded by $N ( t , \check { \mathcal { L } } , \| \cdot \| _ { \infty } )$ . By Wainwright (2019) Theorem 5.22 with $\alpha = 0$ , the pairwise supremum is therefore controlled by the Dudley integral from Section 3:

$$
\mathbb { E } _ { \varepsilon } \left[ \operatorname* { s u p } _ { \theta , \theta ^ { \prime } \in \Theta } ( Z _ { \theta } - Z _ { \theta ^ { \prime } } ) \right] \leq 3 2 \mathcal { J } ( \mathcal { L } ) .
$$

The first term in the r.h.s. of equation 3 vanishes at $\alpha = 0$ . Although $\rho _ { X }$ is only a pseudometric, $\rho _ { X } ( \theta , \theta ^ { \prime } ) =$ 0 forces $\ell ( \theta , x _ { i } , y _ { i } ) = \ell ( \theta ^ { \prime } , x _ { i } , y _ { i } )$ for every �, hence $Z _ { \theta } - Z _ { \theta ^ { \prime } } = 0$ identically. The upper integration limit exceeds the true diameter $D \leq \log ( ( 1 - \gamma ) / \gamma )$ , but the integrand vanishes for radii larger than �. Theorem 5.22 bounds the pairwise supremum, whereas $\mathcal { R } _ { n } ( \mathcal { L } )$ involves $\mathbb { E } [ \operatorname { s u p } _ { \theta } \lvert Z _ { \theta } ] \rvert$ . To pass between them, fix any $\theta _ { 0 } \in \Theta$ and write $\begin{array} { r } { S : = \operatorname* { s u p } _ { \theta ^ { \prime } , \theta ^ { \prime \prime } \in \Theta } ( Z _ { \theta ^ { \prime } } - Z _ { \theta ^ { \prime \prime } } ) } \end{array}$ . For every �,

$$
Z _ { \theta } = Z _ { \theta _ { 0 } } + \left( Z _ { \theta } - Z _ { \theta _ { 0 } } \right) \le | Z _ { \theta _ { 0 } } | + S \quad \mathrm { a n d } \quad - Z _ { \theta } = - Z _ { \theta _ { 0 } } + \left( Z _ { \theta _ { 0 } } - Z _ { \theta } \right) \le | Z _ { \theta _ { 0 } } | + S ,
$$

hence $| Z _ { \theta } | \leq | Z _ { \theta _ { \mathsf { n } } } | + S$ , and therefore

$$
\begin{array} { r l } & { \mathcal { R } _ { n } ( \mathcal { L } ) = \frac { 1 } { \sqrt { n } } \mathbb { E } _ { \nu _ { \alpha } } \bigg [ \frac { \mathrm { s u p } } { \mathrm { e } } Z _ { \nu } \bigg [ z _ { \alpha } \log Z _ { \nu } \bigg ] } \\ & { \qquad \leq \frac { 1 } { \sqrt { n } } \mathbb { E } _ { \nu _ { \alpha } } \bigg [ \mathbb { Z } _ { \nu _ { \alpha } } \big ] + \frac { \mathrm { s u p } ( Z _ { \nu } - Z _ { \nu } ) } { \nu _ { \beta } } \bigg ] \qquad \quad \mathrm { ( u s i a g l e ~ i n s t . , ~ s e ~ t h e o r i s e ) } } \\ & { \qquad - \frac { 1 } { \sqrt { n } } \bigg ( \mathbb { E } _ { \nu _ { \alpha } , \nu _ { \alpha } } | \mathcal { L } _ { \nu _ { \alpha } } \big ) + \mathbb { E } _ { \nu _ { \alpha } , \nu _ { \alpha } } \bigg [ \frac { \mathrm { s u p } ( Z _ { \nu } - Z _ { \nu } ) } { \nu _ { \alpha } \nu _ { \alpha } } \bigg ] \bigg ) \qquad \mathrm { ( l i n e a r i t y ) } } \\ & { \qquad \leq \frac { 1 } { \sqrt { n } } \bigg ( \mathbb { E } _ { \nu _ { \alpha } } \bigg [ \big ( \mathbb { E } _ { \nu _ { \alpha } } Z _ { \nu } ^ { \nu } \big ) ^ { 1 / 2 } \bigg ) ^ { 1 / 2 } + 3 2 \mathcal { L } ( \mathcal { L } ) \bigg ) \qquad \quad \mathrm { ( l e n o r s t . ~ T h ~ m ~ i > 2 ~ n ~ t a ~ - n ~ ) } } \\ & { \qquad - \sqrt { n } \bigg ( \mathbb { E } _ { \nu _ { \alpha } } \bigg [ \big ( \mathbb { E } _ { \nu _ { \alpha } } \big [ z _ { \nu _ { \alpha } } \big ] _ { \nu _ { \alpha } } \big ) + 3 2 \mathcal { L } ( \mathcal { L } ) \bigg ) \qquad \quad \mathrm { ( E } _ { \nu _ { \alpha } } [ \mathcal { L } _ { \nu _ { \alpha } } ] = \delta _ { \nu _ { \alpha } } \bigg ) } \\ &  \qquad \leq  \end{array}
$$

where the fourth line uses $\begin{array} { r } { \mathbb { E } _ { \varepsilon } \mid Z _ { \theta _ { 0 } } \mid \leq \left( \mathbb { E } _ { \varepsilon } Z _ { \theta _ { 0 } } ^ { 2 } \right) ^ { 1 / 2 } = \left. \ell _ { \theta _ { 0 } } \right. _ { L ^ { 2 } ( Q _ { n } ) } \leq \left. \ell _ { \theta _ { 0 } } \right. _ { \infty } } \end{array}$ together with the Theorem 5.22 bound above, and the last uses $\left\| \ell _ { \theta _ { 0 } } \right\| _ { \infty } \leq b$ . Therefore with probability at least $1 - \delta / 2 , ( 2 )$ yields that $T _ { 1 }$ is bounded by

$$
T _ { 1 } \leq \frac { 2 b + 6 4 \mathcal { J } ( \mathcal { L } ) } { \sqrt { n } } + \log \left( \frac { 1 } { \gamma } \right) \sqrt { \frac { 2 \log ( 2 / \delta ) } { n } } .
$$

We will now bound $T _ { 3 }$ . Since $\theta ^ { * }$ is non-random, $\begin{array} { r } { T _ { 3 } = \hat { R } ( \theta ^ { * } , V _ { n } ) - R ( \theta ^ { * } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( \theta ^ { * } , X _ { i } , Y _ { i } ) - } \end{array}$ $\mathbb { E } [ \ell ( \theta ^ { * } , X , Y ) ]$ is an average of i.i.d. centered random variables, each bounded in $[ \log ( 1 / ( \bar { 1 } - \gamma ) ) , \log ( 1 / \gamma ) ]$ with range $\log ( ( 1 - \gamma ) / \gamma )$ . By Hoefding’s inequality, $Q ^ { \otimes n } ( T _ { 3 } \geq t ) \leq \exp \{ - 2 n t ^ { 2 } / \log ^ { 2 } ( ( 1 - \gamma ) / \gamma ) \}$ . Setting $\exp \{ - 2 n t ^ { 2 } / \log ^ { 2 } ( ( 1 - \gamma ) / \gamma ) \} = \delta / 2$ and solving for � gives $t = \log ( ( 1 - \gamma ) / \gamma ) \sqrt { \log ( 2 / \delta ) / 2 n }$ . Therefore with probability at least $1 - \delta / 2 ,$

$$
T _ { 3 } \leq \log \left( \frac { 1 - \gamma } { \gamma } \right) \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .
$$

By a union bound, both bounds hold simultaneously with probability at least $1 - \delta$ over $V _ { n } \sim Q ^ { \otimes n }$ . Since $T _ { 2 } \leq 0 ,$ , we have $R \big ( \hat { \theta } _ { n } \big ) - R ( \theta ^ { * } ) \leq T _ { 1 } + T _ { 3 }$ , hence

$$
\begin{array} { r l r } {  { R \big ( \hat { \theta } _ { n } \big ) - R ( \theta ^ { * } ) \leq \frac { 2 b + 6 4 \mathcal { J } ( \mathcal { L } ) } { \sqrt { n } } + \log \Big ( \frac { 1 } { \gamma } \Big ) \sqrt { \frac { 2 \log ( 2 / \delta ) } { n } } + \log \Big ( \frac { 1 - \gamma } { \gamma } \Big ) \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } } } \\ & { } & { \leq \frac { 2 \log ( 1 / \gamma ) } { \sqrt { n } } + \frac { 6 4 \mathcal { J } ( \mathcal { L } ) } { \sqrt { n } } + \sqrt { 2 } \log \Big ( \frac { 1 } { \gamma } \Big ) \frac { \sqrt { \log ( 2 / \delta ) } } { \sqrt { n } } + \frac { 1 } { \sqrt { 2 } } \log \Big ( \frac { 1 } { \gamma } \Big ) \frac { \sqrt { \log ( 2 / \delta ) } } { \sqrt { n } } } \\ & { } & { \lesssim \frac { \mathcal { J } ( \mathcal { L } ) } { \sqrt { n } } + \frac { \log ( 1 / \gamma ) } { \sqrt { n } } + \log \Big ( \frac { 1 } { \gamma } \Big ) \frac { \sqrt { \log ( 2 / \delta ) } } { \sqrt { n } } } \\ & { } & { = \frac { \mathcal { J } ( \mathcal { L } ) + \log ( 1 / \gamma ) \big ( 1 + \sqrt { \log ( 2 / \delta ) } \big ) } { \sqrt { n } } , } \end{array}
$$

Together with the decomposition $\mathbb { E } _ { Q _ { X } } \big [ \mathrm { K L } _ { X } \big ( p \mathrm { ~ \| ~ } p _ { \hat { \theta } _ { n } } \big ) \big ] = \varepsilon _ { \mathcal { F } } + \big ( R \big ( \hat { \theta } _ { n } \big ) - R ( \theta ^ { * } ) \big )$ from the beginning of the proof, and applying Lemma 11 then gives the result. In both bounds, $\lesssim$ hides only universal constants, independent of $n , \delta ,$ and $\gamma .$ □

## A.2 Concrete rate for fully-connected truncated deep ReLU networks

Remark 9 (Explicit rate). Assume (�) and $f ^ { * } : = 1 \circ \mathfrak { g i t } \circ \eta$ . Choose the bound of the model class as $B : = F =$ log $( 1 - \kappa ) / \kappa )$ , so that $\gamma = \sigma ( - B ) = \kappa$ . Let $g _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R }$ be a fully connected ReLU network, cf. Equation (5) of depth � and hidden width $N : = d + 2$ , with parameters � ranging over $\Theta : = [ - W , W ] ^ { m }$ for a weight bound $W \geq 2$ and � the number of parameters, and let $\mathcal { F } : = \{ T _ { B } \circ g _ { \theta } : \theta \in \Theta \}$ and $T _ { B } ( t ) : = \operatorname* { m a x } \{ - B , \operatorname* { m i n } \{ B , t \} \}$ truncation with �. Then ℱ is a model class in the sense of Section 3 with bound $B ,$ and

$$
\varepsilon _ { \mathcal { F } } \leq \frac { \operatorname* { i n f } _ { f \in \mathcal { F } } \left\| f ^ { * } - f \right\| _ { \infty } ^ { 2 } } { 1 6 \kappa ( 1 - \kappa ) } , \qquad \mathcal { J } ( \mathcal { L } ) \lesssim L \sqrt { \log W } .
$$

Proof. Each $f _ { \theta } = T _ { B } \circ g _ { \theta }$ is continuous with $\left\| f _ { \theta } \right\| _ { \infty } \leq B$ , so $\mathcal { F }$ is a model class with bound $B ,$ and $\gamma = \sigma ( - B ) = \kappa . \mathrm { ~ B y ~ } ( \kappa ) , f ^ { * } = 1 0 \mathrm { g i t } \circ \eta$ satisfies $\begin{array} { r } { \left. f ^ { * } \right. _ { \infty } \leq B } \end{array}$ . Throughout, $\begin{array} { r } { r _ { \mathcal { X } } : = \operatorname* { s u p } _ { x \in \mathcal { X } } \left\| x \right\| _ { \infty } \vee 1 < \infty } \end{array}$ by compactness of �, and $C , C ^ { \prime } , \ldots$ denote constants depending only on � and $r _ { \mathcal { X } }$

Step 1: the approximation floor. For $\mu \ll \nu$ on a discrete space, log $t \leq t - 1$ with $t = \mu ( \boldsymbol { y } ) / \nu ( \boldsymbol { y } )$ gives

$$
\begin{array} { r l } { \mathrm { K L } ( \mu \mid \nu ) = \displaystyle \sum _ { y } \mu ( y ) \log \frac { \mu ( y ) } { \nu ( y ) } } \\ { \leq \displaystyle \sum _ { x } \mu ( y ) \left( \frac { \mu ( y ) } { \nu ( y ) } - 1 \right) } \\ { = \displaystyle \sum _ { y } \frac { \mu ( y ) ^ { 2 } } { \nu ( y ) } - \sum _ { y } \mu ( y ) } \\ { = \displaystyle \sum _ { x } \frac { \mu ( y ) ^ { 2 } } { \nu ( y ) } - 2 \sum _ { y } \mu ( y ) + \sum _ { x } \nu ( y ) } \\ { = \displaystyle \sum _ { x } \frac { \mu ( y ) ^ { 2 } - 2 \mu ( y ) \nu ( y ) + \nu ( y ) ^ { 2 } } { \nu ( y ) } } \\ { = \displaystyle \sum _ { x } \frac { \left( \mu ( y ) - \nu ( y ) \right) ^ { 2 } } { \nu ( y ) } - \nu ^ { 2 } ( \mu ( y ) \nu ) . } \end{array}
$$

Writing $\eta _ { f } : = \sigma \circ f ;$ which takes values in $[ \kappa , 1 - \kappa ]$ whenever $\| f \| _ { \infty } \leq B$

$$
\begin{array} { r l } & { \chi ^ { 2 } \big ( \mathrm { B e r n } _ { \eta ( \infty ) } \big | \mathrm { B e r n } _ { \eta ( \infty ) } \big ) = \frac { \big ( \eta ( x ) - \eta _ { f } ( x ) \big ) ^ { 2 } } { \eta _ { f } ( x ) } + \frac { \big ( ( 1 - \eta ( x ) ) - \big ( 1 - \eta _ { f } ( x ) \big ) \big ) ^ { 2 } } { 1 - \eta _ { f } ( x ) } } \\ & { \qquad = ( \eta ( x ) - \eta _ { f } ( x ) ) ^ { 2 } \bigg ( \frac { 1 } { \eta _ { f } ( x ) } + \frac { 1 } { 1 - \eta _ { f } ( x ) } \bigg ) } \\ & { \qquad = ( \eta ( x ) - \eta _ { f } ( x ) ) ^ { 2 } \frac { \big ( 1 - \eta _ { f } ( x ) \big ) + \eta _ { f } ( x ) } { \eta _ { f } ( x ) \big ( 1 - \eta _ { f } ( x ) \big ) } } \\ & { \qquad = \frac { \big ( \eta ( x ) - \eta _ { f } ( x ) \big ) ^ { 2 } } { \eta _ { f } ( x ) \big ( 1 - \eta _ { f } ( x ) \big ) } } \\ & { \qquad = \frac { \big ( \eta ( x ) - \eta _ { f } ( x ) \big ) ^ { 2 } } { \eta _ { f } ( x ) \big ( 1 - \eta _ { f } ( x ) \big ) } } \\ & { \qquad \leq \frac { ( \eta ( x ) - \eta _ { f } ( x ) ) ^ { 2 } } { \kappa ( 1 - \kappa ) } . } \end{array}
$$

Since sup<sub>�</sub> $\begin{array} { r } { \sigma ^ { \prime } ( t ) = \operatorname* { s u p } _ { t } \sigma ( t ) ( 1 - \sigma ( t ) ) \leq \frac { 1 } { 4 } } \end{array}$ , we have $\begin{array} { r } { | \eta ( x ) - \eta _ { f } ( x ) | \leq \frac { 1 } { 4 } \| f ^ { * } - f \| _ { \infty } } \end{array}$ , hence for every $x \in \mathscr { X }$

$$
\mathrm { K L } _ { x } { \bigl ( } p \parallel p _ { f } { \bigr ) } \leq { \frac { \left\| f ^ { * } - f \right\| _ { \infty } ^ { 2 } } { 1 6 \kappa ( 1 - \kappa ) } } .\tag{4}
$$

Since $R _ { Q } ( \theta ) = \mathbb { E } _ { Q _ { \mathrm { \tiny ~ Y } } } [ H _ { X } ( p ) ] + \mathbb { E } _ { Q _ { \mathrm { \tiny ~ Y } } } [ \mathrm { K L } _ { X } ( p \mathrm { \ } \| \ p _ { \theta } ) ]$ and the first term does not depend on $\theta , \theta ^ { * }$ also minimizes $\mathbb { E } _ { Q _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ]$ . Hence $\varepsilon _ { \mathcal { F } } \leq \bar { \mathbb { E } } _ { Q _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ]$ for all $\theta \in \Theta$ , and taking the infimum over $f \in \mathcal F$ in equation 4 gives the stated bound.

Step 2: the entropy integral. Write

$$
g _ { \theta } = A _ { \theta } ^ { ( L + 1 ) } \circ \mathrm { R e L U } \circ \cdots \circ \mathrm { R e L U } \circ A _ { \theta } ^ { ( 1 ) }\tag{5}
$$

with $A _ { \theta } ^ { ( l ) } ( z ) = W ^ { ( l ) } z + b ^ { ( l ) }$ , and for $f \in \mathcal F$ let $\ell _ { f } ( x , y ) : = - \log p _ { f } ( y | x )$ with $p _ { f } ( y | x ) : = y \eta _ { f } ( x ) + ( 1 - y ) \big ( 1 -$ $\eta _ { f } ( x ) )$ . Both loss maps $t \mapsto - \log \sigma ( t )$ and $t \mapsto - \log ( 1 - \sigma ( t ) )$ have derivatives bounded by 1 in absolute value, so $\left| \ell _ { f } ( x , y ) - \ell _ { f ^ { \prime } } ( x , y ) \right| \leq \left\| f - f ^ { \prime } \right\| _ { \infty }$

$$
\begin{array} { r l } & { | \ell _ { f } ( x , y ) - \ell _ { f ^ { \prime } } ( x , y ) | = \left| \log p _ { f ^ { \prime } } ( y | x ) - \log p _ { f } ( y | x ) \right| } \\ & { \qquad = \left\{ | \log \sigma ( f ^ { \prime } ( x ) ) - \log \sigma ( f ( x ) ) | , \right. \qquad y = 1 } \\ & { \qquad \left. | \log ( 1 - \sigma ( f ^ { \prime } ( x ) ) ) - \log ( 1 - \sigma ( f ( x ) ) ) | , y = 0 \right. } \\ & { \qquad \le | f ( x ) - f ^ { \prime } ( x ) | \le \| f - f ^ { \prime } \| _ { \infty } , } \end{array}
$$

and therefore $N ( t , \mathcal { L } , \| \cdot \| _ { \infty } ) \leq N ( t , \mathcal { F } , \| \cdot \| _ { \infty } )$

The hidden representations are uniformly bounded. With $h ^ { ( 0 ) } : = x$ and $h ^ { ( l ) } : = \operatorname { R e L U } \Bigl ( A _ { \theta } ^ { ( l ) } \bigl ( h ^ { ( l - 1 ) } \bigr ) \Bigr )$ , every row of $W ^ { ( l ) }$ has at most $N$ entries, all parameters are bounded by $W$ , and ReLU does not increase absolute values, so $\left\| h ^ { ( l ) } \right\| _ { \infty } \le N W \left\| h ^ { ( l - 1 ) } \right\| _ { \infty } + \mathbf { \hat { W } }$ . Using $W \le N W r _ { \mathcal { X } }$ (as $N , r _ { \mathcal { X } } \geq 1 )$ , induction over � gives

$$
\left\| h ^ { ( l ) } \right\| _ { \infty } \le ( 2 N W ) ^ { l } r _ { \mathcal { X } } = : M _ { l } , \qquad l = 0 , . . . , L .
$$

Next, $\theta \mapsto f _ { \theta }$ is Lipschitz. Let $\begin{array} { r } { \left. \theta - \theta ^ { \prime } \right. _ { \infty } \leq \tau } \end{array}$ and interpolate through hybrids $\boldsymbol { \theta } = \boldsymbol { \theta } ^ { ( 0 ) } , . . . , \boldsymbol { \theta } ^ { ( L + 1 ) } = \boldsymbol { \theta } ^ { \prime }$ , where $\theta ^ { ( k ) }$ carries the parameters of $\theta ^ { \prime }$ in layers $1 , \ldots$ � and those of $\theta$ elsewhere; all hybrids lie in $\Theta .$ , so the bounds $M _ { l }$ apply. Consecutive hybrids $\theta ^ { ( l - 1 ) }$ and $\theta ^ { ( l ) }$ difer only in layer $l ,$ where they use the parameters $\left( W ^ { ( l ) } , b ^ { ( l ) } \right)$ of $\theta$ and $\left( W ^ { \prime ( l ) } , b ^ { \prime ( l ) } \right)$ of $\theta ^ { \prime }$ , respectively. Both apply layer � to the same input $h : = h ^ { ( l - 1 ) }$ with $\| h \| _ { \infty } \leq M _ { l - 1 }$ Since all entries difer by at most � and each row of $W ^ { ( l ) }$ has at most $N$ entries, for every output coordinate $i ,$

$$
\begin{array} { r l } & { \Big | \big ( W ^ { \prime ( l ) } h + b ^ { \prime ( l ) } \big ) _ { i } - \big ( W ^ { ( l ) } h + b ^ { ( l ) } \big ) _ { i } \Big | = \Bigg | \displaystyle \sum _ { j } \Big ( W _ { i j } ^ { \prime ( l ) } - W _ { i j } ^ { ( l ) } \Big ) h _ { j } + \Big ( b _ { i } ^ { \prime ( l ) } - b _ { i } ^ { ( l ) } \Big ) \Bigg | } \\ & { \qquad \le \displaystyle \sum _ { j } \Big | W _ { i j } ^ { \prime ( l ) } - W _ { i j } ^ { ( l ) } \Big | \left| h _ { j } \right| + \left| b _ { i } ^ { \prime ( l ) } - b _ { i } ^ { ( l ) } \right| } \\ & { \qquad \le N \tau M _ { l - 1 } + \tau = ( N M _ { l - 1 } + 1 ) \tau . } \end{array}
$$

Hence the outputs of layer � difer by at most $( N M _ { l - 1 } + 1 ) \tau \mathrm { i n } \| \cdot \| _ { \infty }$ . Each of the $L + 1 - l$ subsequent afine maps multiplies this diference by at most $N W$ , while ReLU and $T _ { B }$ are 1-Lipschitz and do not increase it. Summing over the $L + 1$ layers and using $N M _ { l - 1 } + 1 \le 2 N ( 2 N W ) ^ { l - 1 } r _ { \mathcal { X } }$ gives

$$
\begin{array} { r l r l } { \Delta _ { t } } & { A _ { 1 , t } } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } \| _ { L ^ { \infty } } ^ { 2 } , } & { S _ { i } } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { - \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } & { \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , } \\ & { } &  \sum _ { i = 1 } ^ { \infty } \| B _ { i } ^ { \top } \| _ { L ^ { \infty } } ^ { 2 } , \end{array}\tag{6}
$$

By Wainwright (2019, Example 5.8), log $N ( \delta ; [ - 1 , 1 ] ^ { m } , \| \cdot \| _ { \infty } ) \leq m \log ( 1 + 2 / \delta )$ . Since $\Theta = W [ - 1 , 1 ] ^ { m }$ , taking $\delta = t / ( W \Lambda )$ gives

$$
N ( t / \Lambda ; \Theta , \| \cdot \| _ { \infty } ) \leq \biggl ( 1 + \frac { 2 W \Lambda } { t } \biggr ) ^ { m } ,
$$

and by the Lipschitz bound the image of such a cover under $\theta \mapsto f _ { \theta }$ is a �-cover of $\mathcal { F }$ , because every $\theta \in \Theta$ has a cover element $\theta _ { i }$ with $\left\| \theta - \theta _ { i } \right\| _ { \infty } \leq t / \Lambda$ , so that equation 6 gives $\left\| f _ { \theta } - f _ { \theta _ { i } } \right\| _ { \infty } \leq \Lambda \cdot t / \Lambda = t$

Since each of the $L + 1$ afine maps has at most $N ( N + 1 )$ parameters,

$$
m \le N ( N + 1 ) ( L + 1 ) \le 2 N ( N + 1 ) L = \underbrace { 2 N ( N + 1 ) } _ { = : C _ { 1 } } L \qquad ( L + 1 \le 2 L ) .
$$

Moreover,

$$
\begin{array} { r l r l } & { \log ( 1 + 2 W \Lambda ) \leq \log ( 3 W \Lambda ) } & & { ( W \Lambda \geq 1 ) } \\ & { } & & { \qquad = \log 3 + \log W + \log L + ( L + 1 ) \log C + ( L + 1 ) \log W \quad ( \Lambda = L ( C W ) ^ { L + 1 } ) } \\ & { } & & { \qquad \leq \log 3 + \log W + L + 2 L \log C + 2 L \log W \quad } & & { ( \log L \leq L , L + 1 \leq 2 L ) } \\ & { } & & { \qquad \leq L \log 3 + L \log W + L + 2 L \log C + 2 L \log W \quad } & & { ( L \geq 1 ) } \\ & { } & & { \qquad = ( \log 3 + 1 + 2 \log C ) L + 3 L \log W } \\ & { } & & { \qquad \leq ( \log 3 + 1 + 2 \log C ) \frac { L \log W } { \log 2 } + 3 L \log W \quad } & & { ( \log W \geq \log 2 ) } \\ & { } & & { \qquad = \underbrace { \left( \frac { \log 3 + 1 + 2 \log C } { \log 2 } + 3 \right) } _ { = \in \mathcal { L } _ { 0 } } L \log W . } \end{array}
$$

Hence

$$
\begin{array} { r l r l } & { \log N ( t , \mathcal { F } , \| \cdot \| _ { \infty } ) \leq m \log ( 1 + 2 W \Lambda / t ) } & & { \mathrm { ( c o v e r i n g ~ b o u n d ) } } \\ & { \leq m ( \log ( 1 + 2 W \Lambda ) + ( \log ( 1 / t ) \vee 0 ) ) } & & { \mathrm { ( l o g ( 1 + } a / t ) \leq \log ( 1 + a ) + ( \log ( 1 / t ) \vee 0 ) ) } \\ & { \leq C _ { 1 } L ( C _ { 2 } L \log W + ( \log ( 1 / t ) \vee 0 ) ) } & & { \mathrm { ( b o u n d s ~ o n ~ m ~ a n d l o g ( 1 + 2 W \Lambda ) ) } } \\ & { = C _ { 1 } C _ { 2 } L ^ { 2 } \log W + C _ { 1 } L \left( \log ( 1 / t ) \vee 0 \right) } \\ & { \leq \underbrace { C _ { 1 } \operatorname* { m a x } \{ C _ { 2 } , 1 \} } _ { = C _ { 3 } } ( L ^ { 2 } \log W + L ( \log ( 1 / t ) \vee 0 ) ) . } \end{array}
$$

Finally, since $\gamma = \kappa .$ the upper integration limit in $\mathcal { J } ( \mathcal { L } )$ is $B ,$ and

$$
\begin{array} { r l r l } { \sum \Biggl ( \gamma \leq \gamma - \gamma \Biggr ) _ { i } ^ { T } \sqrt { \gamma \zeta ( \eta , t , \zeta , \eta ) } \mathcal { L } ^ { \zeta } } & { } & & { } \\ & { + \int _ { 0 } ^ { t } \sqrt { \gamma \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ & { + \int _ { 0 } ^ { t } \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } & { \mathrm { B y s } , \zeta ( \eta , t , \zeta , \eta ) , } \\ & { \leq \int _ { 0 } ^ { t } \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ & { - \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ & { \leq \sqrt { \eta } \int _ { 0 } ^ { t } \sqrt { \gamma \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ & { } & & { \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ & { } & & { \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ & { } & & { \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ & { } & & { \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm { I } _ { \leq \eta } ^ { T } \zeta ( \eta , t , \zeta , \eta ) , } \\ &  + \sqrt { \zeta ( \eta , t , \zeta , \eta ) } \mathrm  I \end{array}
$$

## A.3 Proof of Theorem 3

Proof. (i) By the change-of-measure formula, we have

$$
\mathbb { E } _ { P _ { X } } \left[ \mathrm { K L } _ { X } \left( p \parallel p _ { \hat { \theta } _ { n } } \right) \right] = \mathbb { E } _ { Q _ { X } } \left[ \mathrm { K L } _ { X } \left( p \parallel p _ { \hat { \theta } _ { n } } \right) { \frac { \mathrm { d } P _ { X } } { \mathrm { d } Q _ { X } } } \right] \leq \left. { \frac { \mathrm { d } P _ { X } } { \mathrm { d } Q _ { X } } } \right. _ { L ^ { \infty } ( \mathcal { X } , Q _ { X } ) } \mathbb { E } _ { X \sim Q _ { X } } \left[ \mathrm { K L } _ { X } \left( p \parallel p _ { \hat { \theta } _ { n } } \right) \right] .
$$

Application of Theorem 1 yields the assertion.

(ii) For fixed but arbitrary $x \in \mathscr { X }$ , we have $\begin{array} { r } { \mathrm { K L } _ { x } \bigl ( p | | p _ { \hat { \theta } _ { n } } \bigr ) \leq \left| p ( 0 | x ) \log \frac { p ( 0 | x ) } { p _ { \hat { \theta } _ { n } } ( 0 | x ) } + p ( 1 | x ) \log \frac { p ( 1 | x ) } { p _ { \hat { \theta } _ { n } } ( 1 | x ) } \right| \leq \frac { 2 } { e } + } \end{array}$ $H _ { x } \big ( p | | p _ { \hat { \theta } _ { \infty } } \big ) < \infty$ , because $\operatorname* { m a x } _ { t \in [ 0 , 1 ] } | t \log t | = 1 / e$ and the cross-entropy term is bounded by $\log ( 1 / \gamma )$ , since $\begin{array} { r } { \| f _ { \theta } \| _ { \infty } \leq \tilde { B } } \end{array}$ . Now,

$$
\begin{array} { r l } & { \mathbb { E } _ { P _ { x } } \left[ { \mathrm { K L } } _ { X } \left( p \mid | p _ { \hat { \theta } _ { n } } \right) \right] \leq \mathbb { E } _ { \varnothing _ { X } } \left[ \left. \frac { \mathrm { d } P _ { X } } { \mathrm { d } Q _ { X } } { \mathrm { K L } } \left( p \mid | p _ { \hat { \theta } _ { n } } \right) \right. \right] = \left\| \frac { \mathrm { d } P _ { X } } { \mathrm { d } Q _ { X } } { \mathrm { K L } } _ { X } \left( p \mid | p _ { \hat { \theta } _ { n } } \right) \right\| _ { { \mathrm { L } } ^ { 1 } ( Q _ { X } ) } } \\ & { \qquad \leq \left\| \frac { \mathrm { d } P _ { X } } { \mathrm { d } Q _ { X } } \right\| _ { { \mathrm { L } } ^ { s } ( Q _ { X } ) } \left\| { \mathrm { K L } } _ { X } \left( p \mid | p _ { \hat { \theta } _ { n } } \right) \right\| _ { { \mathrm { L } } ^ { t } ( Q _ { X } ) } = \left\| \frac { \mathrm { d } P _ { X } } { \mathrm { d } Q _ { X } } \right\| _ { { \mathrm { L } } ^ { s } ( Q _ { X } ) } \mathbb { E } _ { Q _ { X } } \left[ { \mathrm { K L } } _ { X } ^ { t } \left( p \mid | p _ { \hat { \theta } _ { n } } \right) \right] ^ { 1 / t } , } \end{array}
$$

where for $f _ { n } \leq C , \ \mathbb { E } ^ { 1 / t } [ f _ { n } | ^ { t } ] = \mathbb { E } ^ { 1 / t } [ f _ { n } | ^ { 1 + ( t - 1 ) } ] = \mathbb { E } ^ { 1 / t } [ f _ { n } | \ | f _ { n } | ^ { t - 1 } ] \leq C ^ { ( t - 1 ) / t } \mathbb { E } ^ { 1 / t } [ f _ { n } ]$ . Now the assertion follows from an application of Theorem 1 and Lemma 11. □

## A.4 Proof of Theorem 4

Proof. Note that $\tilde { y }$ is measurable. Write $D : = \{ x \in \mathcal { X } : \hat { y } _ { \theta } ( x ) \neq \tilde { y } ( x ) \}$

Let � be a finite measure on $\mathcal X \times \mathcal Y$ with covariate marginal $\nu _ { X }$ . The events $\{ \hat { y } _ { \theta } ( X ) \neq Y \}$ and $\{ \tilde { y } ( X ) \neq Y \}$ coincide on $D ^ { c } \times \mathcal { Y }$ , so their symmetric diference is contained in $D \times \mathcal { Y }$ . Since $\nu ( E ) \leq \nu ( E ^ { \prime } ) + \nu ( E \setminus E ^ { \prime } )$ for any measurable $E , E ^ { \prime }$ , and symmetrically,

$$
| \mathrm { e r r } _ { \nu } ( \hat { y } _ { \theta } ) - \mathrm { e r r } _ { \nu } ( \tilde { y } ) | \leq \nu ( D \times \mathcal { Y } ) = \nu _ { X } ( D ) .\tag{7}
$$

As $P _ { X } ^ { a }$ and $P _ { X } ^ { s }$ are restrictions of $P _ { X }$ , we have $\nu _ { X } ( D ) \leq \varepsilon$ for $\nu \in \{ Q , P ^ { a } , P ^ { s } \}$

By construction �̃ and $y ^ { * }$ difer at most on $M _ { s }$ , and $Q _ { X } ( M _ { s } ) = 0 , \mathrm { s o } \tilde { y } = y ^ { * }$ holds $Q _ { X } { \mathrm { - a . e } }$ . Since $\mathrm { e r r } _ { Q }$ depends on a classifier only through its values $Q _ { X } \mathrm { - a . e . }$ ., this gives $\mathrm { e r r } _ { Q } ( \tilde { y } ) = \mathrm { e r r } _ { Q } ( y ^ { \ast } )$ . Combining with equation $7$ for $\nu = Q$ and $Q _ { X } ( D ) \leq \varepsilon _ { * }$

$$
\operatorname { e r r } _ { Q } ( { \hat { y } } _ { \theta } ) \leq \operatorname { e r r } _ { Q } ( { \tilde { y } } ) + Q _ { X } ( D ) \leq \operatorname { e r r } _ { Q } ( y ^ { * } ) + \varepsilon ,
$$

which is the first claim.

For every measurable $h ,$ using $\mathcal { Y } = \{ 0 , 1 \}$

$$
\operatorname { e r r } _ { P ^ { s } } ( h ) = \int _ { M _ { s } } \sum _ { y \in \mathcal { Y } } \mathbf { 1 } \{ h ( x ) \neq y \} p ( y | x ) P _ { X } ( \mathrm { d } x ) = \int _ { M _ { s } } p ( 1 - h ( x ) | x ) P _ { X } ( \mathrm { d } x ) .
$$

The integrand is at most $\operatorname* { m a x } _ { y } p ( y | x )$ for every �, so $\begin{array} { r } { \operatorname { e r r } _ { P ^ { s } } ( h ) \leq \int _ { M _ { \circ } } \operatorname* { m a x } _ { y } p ( y | x ) P _ { X } ( \mathrm { d } x ) } \end{array}$ for every ℎ. Since $\tilde { y } = \hat { y } _ { \mathrm { a d v } } \ \mathrm { o n } \ M _ { s }$ , the integrand equals ma $\tau _ { y } p ( y | x )$ for $h = \tilde { y } ,$ so this upper bound is attained and $\mathrm { e r r } _ { P ^ { s } } ( \tilde { y } ) = \mathrm { m a x } _ { h } \mathrm { e r r } _ { P ^ { s } } ( h )$ . Equation 7 applied to $\nu = P ^ { s }$ gives $\vert \mathrm { e r r } _ { P ^ { s } } ( \hat { y } _ { \theta } ) - \mathrm { e r r } _ { P ^ { s } } ( \tilde { y } ) \vert \leq P _ { X } ^ { s } ( D )$ , and dropping the absolute value yields $\mathrm { e r r } _ { P ^ { s } } ( \hat { y } _ { \theta } ) \geq \mathrm { e r r } _ { P ^ { s } } ( \tilde { y } ) - P _ { X } ^ { s } ( D )$ . Since $P _ { X } ^ { s }$ is a restriction of $P _ { X } ,$ , we have $P _ { X } ^ { s } ( D ) \leq$ $P _ { X } ( D ) \leq \varepsilon$ , and therefore

$$
\operatorname { e r r } _ { P ^ { s } } ( \hat { y } _ { \theta } ) \geq \operatorname { e r r } _ { P ^ { s } } ( \tilde { y } ) - \varepsilon = \operatorname* { m a x } _ { h } \operatorname { e r r } _ { P ^ { s } } ( h ) - \varepsilon ,
$$

which is the second claim.

## A.5 Proof of Lemma 10

Lemma 10. Assume $( \kappa ) , ~ ( L _ { p } )$ and $\left( L _ { f } \right)$ . Then, for every $\theta \in \Theta$

$$
\mathbb { E } _ { X \sim { P } _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ] \le \mathbb { E } _ { X \sim { Q } _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ] + \big [ L _ { f } + ( B + F ) L _ { p } \big ] W _ { 1 } ( { P } _ { X } , { Q } _ { X } ) .
$$

Proof of Lemma 10. Set $\eta ( \boldsymbol { x } ) : = p ( 1 | \boldsymbol { x } )$ and define $G ( a , t ) : = \mathrm { K L } \big ( \mathrm { B e r n } _ { a } \big \| \mathrm { B e r n } _ { \sigma ( t ) } \big )$ . Using $\log \sigma ( t ) = t -$ $\log ( 1 + e ^ { t } )$ and log $\begin{array} { r } { \mathrm { ~ ; ~ } ( 1 - \sigma ( t ) ) = - \log ( 1 + e ^ { t } ) } \end{array}$ , we obtain

$$
\begin{array} { l } { G ( a , t ) : = \mathrm { K L } \Big ( \mathrm { B e r n } _ { a } \left. \mathrm { B e r n } _ { \sigma ( t ) } \right) } \\ { \qquad = a \log \left( \displaystyle \frac { a } { \sigma ( t ) } \right) + ( 1 - a ) \log \left( \displaystyle \frac { 1 - a } { 1 - \sigma ( t ) } \right) } \\ { \qquad = a \log a - a \log \sigma ( t ) + ( 1 - a ) \log ( 1 - a ) - ( 1 - a ) \log ( 1 - \sigma ( t ) ) } \\ { \qquad = a \log a - a \big ( t - \log ( 1 + e ^ { t } ) \big ) + ( 1 - a ) \log ( 1 - a ) - ( 1 - a ) \big ( - \log ( 1 + e ^ { t } ) \big ) } \\ { \qquad = a \log a - a t + a \log ( 1 + e ^ { t } ) + ( 1 - a ) \log ( 1 - a ) + ( 1 - a ) \log ( 1 + e ^ { t } ) } \\ { \qquad = a \log a + ( 1 - a ) \log ( 1 - a ) + \log ( 1 + e ^ { t } ) - a t . } \end{array}
$$

Its partial derivatives are

$$
\partial _ { t } G ( a , t ) = \sigma ( t ) - a , \qquad \partial _ { a } G ( a , t ) = \log \biggl ( \frac { a } { 1 - a } \biggr ) - t .
$$

By $( \kappa ) , \eta ( x ) \in [ \kappa , 1 - \kappa ]$ , and $| f _ { \theta } ( x ) | \le B$ by definition of the model class. Therefore, with $\sigma ( t ) , a \leq 1$ and $t = f _ { \boldsymbol { \theta } } ( { \boldsymbol { x } } ) \leq B$

$$
| \partial _ { t } G ( a , t ) | \leq 1 , \qquad | \partial _ { a } G ( a , t ) | \leq F + B .
$$

For any $x , x ^ { \prime } \in \mathcal { X }$ , the mean value theorem gives

$$
\begin{array} { r l } & { \left| \mathrm { K L } _ { x } ( p \parallel p _ { \theta } ) - \mathrm { K L } _ { x ^ { \prime } } ( p \parallel p _ { \theta } ) \right| } \\ & { \quad = | G ( \eta ( x ) , f _ { \theta } ( x ) ) - G ( \eta ( x ^ { \prime } ) , f _ { \theta } ( x ^ { \prime } ) ) | } \\ & { \quad \le | f _ { \theta } ( x ) - f _ { \theta } ( x ^ { \prime } ) | + ( B + F ) | \eta ( x ) - \eta ( x ^ { \prime } ) | } \\ & { \quad \le \left[ L _ { f } + ( B + F ) L _ { p } \right] d ( x , x ^ { \prime } ) . } \end{array}
$$

Hence $x \mapsto \mathrm { K L } _ { x } ( p \parallel p _ { \theta } )$ is Lipschitz with constant at most $L _ { f } + ( B + F ) L _ { p }$ . Set

$$
L : = L _ { f } + ( B + F ) L _ { p }
$$

and let $\pi \in \Gamma ( P _ { X } , Q _ { X } )$ be any coupling. Since the marginals of � are $P _ { X }$ and $Q _ { X }$ , respectively,

$$
\begin{array} { r l } & { \left| \mathbb E _ { X \sim P _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ] - \mathbb E _ { X \sim Q _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ] \right| } \\ & { = \left| \int _ { \mathcal X \times \mathcal X } ( \mathrm { K L } _ { x } ( p \parallel p _ { \theta } ) - \mathrm { K L } _ { x ^ { \prime } } ( p \parallel p _ { \theta } ) ) \mathrm d \pi ( x , x ^ { \prime } ) \right| } \\ & { \le \displaystyle \int _ { \mathcal X \times \mathcal X } | \mathrm { K L } _ { x } ( p \parallel p _ { \theta } ) - \mathrm { K L } _ { x ^ { \prime } } ( p \parallel p _ { \theta } ) | \mathrm d \pi ( x , x ^ { \prime } ) } \\ & { \le L \int _ { \mathcal X \times \mathcal X } d ( x , x ^ { \prime } ) \mathrm d \pi ( x , x ^ { \prime } ) . } \end{array}
$$

Taking the infimum over all $\pi \in \Gamma ( P _ { X } , Q _ { X } )$ yields

$$
\left| \mathbb { E } _ { X \sim { P } _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ] - \mathbb { E } _ { X \sim { Q } _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ] \right| \le \left[ L _ { f } + ( B + F ) L _ { p } \right] W _ { 1 } ( P _ { X } , Q _ { X } ) .
$$

The claimed one-sided inequality follows.

Lemma 11. Let � be a probability distribution on $\mathcal X \times \mathcal Y$ with conditional � and covariate marginal $\nu _ { X }$ . For any $\theta \in \Theta$ and the Bayes classifier $y ^ { * }$ $\operatorname { e r r } _ { \nu } ( \hat { y } _ { \theta } ) - \operatorname { e r r } _ { \nu } ( y ^ { * } ) \leq \sqrt { 2 \mathbb { E } _ { \nu _ { X } } [ \mathrm { K L } _ { X } ( p \parallel p _ { \theta } ) ] }$

Proof. By Theorem 2.2 of Devroye et al. (1996), it holds that $\operatorname { r r } _ { \nu } ( \hat { y } _ { \theta } ) - \operatorname { e r r } _ { \nu } ( y ^ { \ast } ) \leq 2 \mathbb { E } _ { \nu _ { x } } \ | \eta ( X ) - \eta _ { \theta } ( X ) |$ For two Bernoulli distributions we have $\mathrm { T V } \big ( \mathrm { B e r n } _ { \eta ( x ) } , \mathrm { B e r n } _ { \eta _ { \theta } ( x ) } \big ) = | \eta ( x ) - \eta _ { \theta } ( x )$ |. By taking the expectation and applying Pinsker’s and Jensen’s inequalities, we obtain the result. □

## A.6 Proof of Theorem 7

Proof. By $( \kappa ) , f ^ { * }$ is bounded. Let $x \in U _ { \theta }$ . By definition of $U _ { \theta }$ as a union of balls, there is $x ^ { \prime } \in S$ with $x \in$ $B ( x ^ { \prime } , r _ { \theta } ( x ^ { \prime } ) )$ , i.e.

$$
\| x - x ^ { \prime } \| < r _ { \theta } ( x ^ { \prime } ) = \operatorname* { m i n } \Bigg \{ \frac { | f _ { \theta } ( x ^ { \prime } ) | } { L _ { f } } , \frac { | f ^ { * } ( x ^ { \prime } ) | } { L ^ { * } } \Bigg \} .
$$

Since $x ^ { \prime } \in S$ , we have $\hat { y } _ { \theta } ( x ^ { \prime } ) = y ^ { * } ( x ^ { \prime } ) . \mathrm { ~ B y ~ } \big ( L _ { f } \big ) , | f _ { \theta } ( x ) - f _ { \theta } ( x ^ { \prime } ) | \leq L _ { f } \| x - x ^ { \prime } \| < | f _ { \theta } ( x ^ { \prime } ) |$ , so $f _ { \theta } ( x )$ has the same strict sign as $f _ { \theta } ( x ^ { \prime } )$ and hence $\hat { y } _ { \boldsymbol { \theta } } ( x ) = \hat { y } _ { \boldsymbol { \theta } } ( x ^ { \prime } ) . \mathrm { ~ B y ~ } ( L ^ { * } ) , | f ^ { * } ( x ) ^ { * } - f ^ { * } ( x ^ { \prime } ) | \leq L ^ { * } \| x - x ^ { \prime } \| < | f ^ { * } ( x ^ { \prime } )$ |, so $f ^ { * } ( x )$ has the same strict sign as $f ^ { * } ( x ^ { \prime } )$ and hence $y ^ { * } ( x ) = y ^ { * } ( x ^ { \prime } )$ . Together,

$$
\hat { y } _ { \boldsymbol \theta } ( \boldsymbol x ) = \hat { y } _ { \boldsymbol \theta } ( \boldsymbol x ^ { \prime } ) = y ^ { * } ( \boldsymbol x ^ { \prime } ) = y ^ { * } ( \boldsymbol x ) ,
$$

so $\{ \hat { y } _ { \theta } \neq y ^ { * } \} \subseteq \mathcal { X } \setminus U _ { \theta }$ . Since ${ \bf 1 } \{ \hat { y } _ { \theta } ( X ) \neq Y \} \leq { \bf 1 } \{ y ^ { * } ( X ) \neq Y \} + { \bf 1 } \{ \hat { y } _ { \theta } ( X ) \neq y ^ { * } ( X ) \}$ , taking expectations under $P$ yields

$$
\begin{array} { r } { \operatorname { e r r } _ { P } ( \hat { y } _ { \theta } ) \leq \operatorname { e r r } _ { P } ( y ^ { * } ) + P _ { X } ( \hat { y } _ { \theta } ( X ) \neq y ^ { * } ( X ) ) \leq \operatorname { e r r } _ { P } ( y ^ { * } ) + P _ { X } ( X \notin U _ { \theta } ) . } \end{array}
$$

If supp $P _ { X } \subseteq U _ { \theta }$ , the right-hand side vanishes.

## B Miscellaneous

## B.1 Additional Information on Section 4.1

The networks trained on real and on synthetic data were trained with 100 seeds each. The conditional label distribution is shared by both marginals and is modelled as $p ( 1 | ( x _ { 1 } , x _ { 2 } ) ) = \operatorname* { m i n } \{ \operatorname* { m a x } \{ \sigma ( x _ { 2 } ) , \kappa \} , 1 -$ $\kappa \}$ with $\kappa = 0 . 0 5$ . We trained a MLP with two hidden layers of width 64 and ReLU activations, mapping to a single logit $z ( x )$ , which is bounded via composition with $f ( x ) = B \operatorname { t a n h } ( z ( x ) / B )$ with $B = \log ( ( 1 -$ $\kappa ) / \kappa )$ . Assumptions $( L ^ { \infty } )$ and (�) are satisfied by construction. One model was trained on real (blue), one on synthetic samples (red) and both were evaluated against the same real test set.

## B.2 Details on Correlation Experiment

The tested models include variants of vanilla CNNs Lecun et al. (1998), ResNet He et al. (2016), DenseNet Huang et al. (2017), MobileNet Sandler et al. (2018), Inception Szegedy et al. (2015), squeeze-excite CNN Hu et al. (2018), dilated CNN Yu & Koltun (2016), ViT-small Dosovitskiy et al. (2021), MLP Rumelhart et al. (1986), MLP-Mixer Tolstikhin et al. (2021), random-wired networks Xie et al. (2019), spectral mixers Rao et al. (2023), Swin-inspired windowed attention (no window shifting) Liu et al. (2021) and involution networks Li et al. (2021), where hyper parameters like kernel size or number of layers were varied. All networks are trained from scratch on $3 \times 2 2 4 \times 2 2 4$ inputs. Depth, width, kernel size, patch size and attention head count (if applicable) were drawn randomly and each instance also draws one activation from {ReLU, GELU, SiLU, leaky ReLU, Mish, ELU}, applied uniformly throughout that network. The windowed-attention variant applies local window attention at a single resolution, without shifted windows, hierarchical patch merging, or relative position bias, so it is not a proper Swin Transformer. The spectral mixer takes only the global learnable frequency-domain filter from GFNet.

## B.3 Alternative Lipschitz bound

In order to obtain a bound that is in spirit more similar to Theorem 1 and Theorem 3 one can take the following angle.

Theorem 12 (Convergence under a Lipschitz constraint). Assume $\left( \kappa \right) , \left( L _ { f } \right)$ and $\left( L _ { p } \right)$ , and set $L : = L _ { f } +$ $( B + F ) L _ { p }$ . Then for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ over $V _ { n } \sim Q ^ { \otimes n }$

$$
\begin{array} { r } { \mathrm { e r r } _ { P } \Big ( \widehat { y } _ { \widehat { \theta } _ { n } } \Big ) - \mathrm { e r r } _ { P } ( y ^ { * } ) \lesssim \sqrt { \varepsilon _ { \mathcal { F } } + r _ { n } ( \delta ) + L W _ { 1 } ( P _ { X } , Q _ { X } ) } , } \end{array}
$$

with $r _ { n } ( \delta )$ as in Theorem 3.

Proof. By Lemma 11, we have $\mathrm { e r r } _ { P } \big ( \hat { y } _ { \hat { \theta } _ { n } } \big ) - \mathrm { e r r } _ { \frac { P } { \rho } } ( y ^ { * } ) \lesssim \big ( \mathbb { E } _ { P _ { X } } \big [ \mathrm { K L } _ { X } \big ( p \mathrm { ~ \textubscript ~ { \textmu ~ } } \big \| ~ p _ { \hat { \theta } _ { n } } \big ) \big ] \big ) ^ { \frac { 1 } { 2 } }$ . By Lemma 10, applied for fixed $V _ { n } ,$ one has $\mathbb { E } _ { P _ { X } } \left[ \mathrm { K L } _ { X } \Big ( p \normalfont { \parallel } p _ { \hat { \theta } _ { n } } \Big ) \right] ^ { * } \leq \mathbb { E } _ { Q _ { X } } \left[ \mathrm { K L } _ { X } \Big ( p \normalfont { \parallel } p _ { \hat { \theta } _ { n } } \Big ) \right] + L W _ { 1 } ( P _ { X } ^ { * , * } \dot { Q } _ { X } )$ , and by Theorem 1, with probability at least $1 - \delta$ over $V _ { n } \sim \mathcal { \dot { Q } } ^ { \smash { \circ } { \tilde { \otimes } } n } , \mathbb { E } _ { Q _ { X } } \bigl [ \mathbf { \bar { K } } \mathbf { L } _ { X } \bigl ( p \mathrm { ~ \\\\\\\lVert ~ } p _ { \hat { \theta } _ { n } } \bigr ) \bigr ] \lesssim \varepsilon _ { \mathcal { F } } + r _ { n } ( \delta )$ . Combining the three displays yields the stated inequality. □

Bounds of this form appear in domain adaptation, where the target risk of a Lipschitz hypothesis is controlled by the source risk plus a Wasserstein term (Redko et al., 2017; Shen et al., 2018, Lemma 1). These results bound an expected loss between two hypotheses, whereas we need the conditional Kullback–Leibler risk, and Redko et al. (2017) measure the Wasserstein cost in an RKHS rather than the Euclidean metric. It must be noted however that $\varepsilon _ { \mathcal { F } }$ and � are inversely related: As � decreases $\varepsilon _ { \mathcal { F } }$ increases. Therefore, the bound might be U-shaped. Also the Jensen and Pinsker step are very pessimistic so that in practice the bound in Theorem 7 should be preferred.

## B.4 Experimental validation of Theorem 7

For this experiment, we sampled � uniformly from the unit disk and � uniformly from the concentric annulus $1 < \| x \| \leq 1 . 1$ , which is disjoint from the disk. The dataset is depicted in Figure 6 (left). The shared concept map was modeled as $\eta ( x ) = \sigma ( 4 x _ { 2 } )$ clipped to [0.05, 0.95]. As neural network architecture we chose a 2-32-32-1 ReLU MLP. We enforce a Lipschitz bound � on the network by projecting the singular values of each of the three weight matrices onto $\left[ 0 , L ^ { 1 / 3 } \right]$ after every optimisation step, for $L \in$ $\{ 0 . 5 , 1 , 2 , 4 , 8 , 1 6 , 6 4 , \infty \}$ . For each �, we train with Adam (learning rate $1 0 ^ { - 2 }$ , 2000 full-batch steps) on a weighted binary cross-entropy over 1000 synthetic points with labels drawn from � and 1000 real points with adversarially flipped labels, weighting the real points by $\lambda \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 1 , 0 . 3 , 1 \}$ . Among all checkpoints (every 40 steps) whose synthetic excess error is at $\varepsilon = 0 . 0 5$ , we report the largest real excess error, i.e. the worst-case model that is still accurate on �. We compare it to the bound given by the real mass not covered by the specified open balls around correctly classified synthetic points, estimated with 4000 real and 8000 synthetic samples. Excess errors are measured against Monte Carlo estimates of the Bayes error from 20,000 samples.

![](images/7036526ded7eb2462b0fdeea9a7cdea75d35025192f7bf96e997f3998f06ec75.jpg)  
Figure 6: Dataset (left) and excess error against Lipschitz constant, cf. Appendix B.4

![](images/91065023e234f5437a30db33cd9bf9d12ca9357f360fa40d3e4ba2aaa0dff943.jpg)