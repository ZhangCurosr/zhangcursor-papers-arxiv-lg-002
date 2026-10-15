# On Inherent Privacy of Posterior Sampling: A Unified Rényi-Divergence Framework

Ruyu Zhou Department of Computer Science and Statistics University of Rhode Island ruyu.zhou@uri.edu

Fang Liu   
Department of ACMS   
University of Notre Dame   
fliu2@nd.edu

## Abstract

Differential Privacy (DP) provides a rigorous framework for quantifying privacy guarantees. While most methods achieve DP by injecting calibrated random noise, the intrinsic randomness of certain procedures such as sampling can yield DP guarantees “for free”. In this work, we develop a unified Rényi divergence framework to characterize the inherent DP guarantees of sampling from a Bayesian posterior distribution. We show that the privacy loss incurred by One-Posterior-Sampling (OPS) is governed by posterior exponential moments of the single-record loglikelihood ratio, and establish explicit DP guarantees under uniformly bounded, sub-Gaussian, or sub-exponential tail regimes. We apply the framework to representative Bayesian models, including categorical likelihoods with arbitrary priors, Gaussian-Gaussian conjugate pair, generalized linear models from the exponential family and linear regression with Gaussian likelihood, both with Gaussian priors on the regression coefficients. Our results elucidate sample size, model structure, neighboring relations in DP (bounded or unbounded), and how prior if applicable, jointly determine the inherent DP guarantees of OPS. Furthermore, our results recover the existing DP guarantees for OPS as special cases – often with tighter privacy loss bounds – and substantially broaden the class of Bayesian models for which the inherent DP guarantees of OPS can be rigorously characterized.

## 1 Introduction

Background and Motivation. Differential Privacy (DP) has emerged as the gold standard for providing rigorous privacy guarantees when releasing information from sensitive databases since its introduction in 2006 [Dwork et al., 2006a,b]. By bounding the change in a randomized algorithm’s output distribution induced by any single individual’s data, classical DP notion ensures that no individual can exert a disproportionate influence on the outcome. However, the worst-case nature of pure DP can be overly conservative for practical applications. This has motivated several influential relaxations that enable more refined privacy accounting, such as Rényi DP [Mironov, 2017], zeroconcentrated DP [Bun and Steinke, 2016], Gaussian DP [Dong et al., 2022], among others.

While a target DP level is typically achieved by explicitly injecting calibrated noise – a paradigm underpinning most DP methods – many learning and analytical procedures are intrinsically stochastic. Accounting for the privacy protection provided by such inherent stochasticity would potentially help meet DP requirements while reducing, or even eliminating, the need for external perturbation, thereby better preserving model utility. Bayesian posterior sampling naturally falls into this category and offers a principled setting for exploiting this form of “free” privacy. In particular, a single posterior draw θ defines a complete randomized predictor or synthetic-data generator; any number of predictions or synthetic records computed from θ<sup>∗</sup> inherit its privacy guarantee by post-processing, without composition.

Despite growing interest in Bayesian methods under DP, existing guarantees for One-Posterior-Sampling (OPS) remain limited (see Related Work): many require uniformly bounded likelihood ratios, bounded sample or parameter spaces, or modified (pseudo-)posterior mechanisms; or yield loose privacy loss bounds. Recent work on posterior sampling under contamination and pseudoposterior mechanisms shows that this problem remains active and practically relevant. We revisit OPS and develop sharper DP guarantees under weaker and more broadly applicable conditions.

## Our Contributions.

• We establish an exact characterization of the Rényi divergence between the posteriors formulated over two neighboring datasets, showing that the privacy loss of OPS is fully determined by two specific exponential moments of the single-record log-likelihood ratio.

• We derive explicit DP guarantees for OPS under three log-likelihood ratio regimes: uniformly bounded, sub-Gaussian, and sub-exponential. These results recover existing OPS DP guarantees as special cases and are compatible with various DP notions.

• We apply the framework to a broader class of Bayesian models than previously studied. The resulting bounds show how the prior, sample size, model structure, and neighboring relation jointly determine the inherent DP guarantees of OPS; in particular, under unbounded neighboring, the two directions of the Rényi divergence can differ even in their range of finite orders.

Related Work. Chaudhuri and Mishra [2006] initiated the idea that random sampling itself can provide DP guarantees. Building on this insight, Li et al. [2012] showed that random sampling before k-anonymization [Sweeney, 2002] yields approximate DP. Subsequent work has mainly followed two directions: 1) studying DP guarantee induced by algorithmic randomness, including uncertainty and noise in control systems [Giraldo et al., 2017] and stochastic generative models such as GANs [Lin et al., 2021]; 2) studying the privacy amplification effect, where subsampling randomness amplifies the privacy guarantees of a base mechanism. Foundational results include Balle et al. [2018] and survey by Steinke [2022], with recent extensions to multistage sampling schemes [Zhao et al., 2024].

Privacy guarantees from posterior sampling in the Bayesian framework have also garnered significant research attention, especially for OPS. Early work Mir [2013], Dimitrakakis et al. [2014, 2017], Wang et al. [2015] established pure DP guarantees for general OPS by assuming uniformly bounded log-likelihood ratio between neighboring datasets – an assumption that is often too restrictive. The resulting DP guarantees scale with the assumed uniform bound and can be conservative due to triangle-inequality-based arguments. Subsequent work examined OPS in specific Bayesian models, including discrete Bayesian networks [Zhang et al., 2016] and exponential family models with conjugate priors [Foulds et al., 2016]. The work most closely related to ours is Geumlek et al. [2017], which studies OPS under Rényi DP [Mironov, 2017], but is limited to exponential family and Bayesian logistic regression. In contrast, our work develops a unified Rényi DP characterization for OPS that applies beyond these settings, and recover their results as special cases.

Beyond OPS, intrinsic DP guarantees have also been studied for related posterior-sampling-based inference procedures, including collapsed Gibbs sampling for latent Dirichlet allocation [Zhao et al., 2020, Blei et al., 2003] and stochastic gradient MCMC [Li et al., 2019]. Related directions include combining OPS with β divergences to obtain bounded sensitivity [Jewson et al., 2023], per-instance DP analyses for linear regression [Minami et al., 2016, Wang, 2019], and posterior sampling under contamination models without boundedness assumptions [Hu et al., 2024]. Our work differs by focusing on unmodified OPS and deriving sharper privacy-loss guarantees under weaker assumptions.

## 2 Preliminaries

Definition 1 (Differential Privacy (DP)). [Dwork et al., 2006a] A randomized mechanism M is said to satisfy (ε, δ)-DP iffor all pairs ofneighboring datasets $( { \bf Y } , { \bf Y ^ { \prime } } )$ differing by one record, denoted by $d ( \dot { \mathbf { Y } } , \dot { \mathbf { Y } } ^ { \prime } ) \dot { = } 1$ , andfor all subsets $\bar { S } \subseteq i m a g e ( \mathcal { M } )$ , it holds that

$$
\operatorname* { P r } ( \mathcal { M } ( \mathbf { Y } ) \in \mathcal { S } ) \leq e ^ { \varepsilon } \operatorname* { P r } ( \mathcal { M } ( \mathbf { Y } ^ { \prime } ) \in \mathcal { S } ) + \delta .\tag{1}
$$

$d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1$ refers to either bounded neighboring [Kifer and Machanavajjhala, 2011], where Y becomes $\mathbf { \bar { Y } ^ { \prime } }$ by substituting exactly one record, or unbounded neighboring, where Y becomes $\mathbf { Y } ^ { \prime }$ by

adding or removing a single record. $\varepsilon > 0$ and $\delta \geq 0$ in Eq. (1) are privacy loss parameters, with smaller values indicating stronger privacy protection. When $\mathbf { \bar { \boldsymbol { \delta } } } = 0 , ( \bar { \varepsilon , } \delta ) – \mathbf { D \bar { \mathbf { P } } }$ becomes pure ε-DP.

DP can also be characterized using a privacy loss random variable that quantifies the privacy loss incurred by releasing a particular output (Definition 2).

Definition 2 (Privacy Loss Random Variable). [Bun and Steinke, 2016, Sommer et al., 2018] For a randomized mechanism $\mathcal { M } ,$ let $o \in$ image(M) be any potential output ofM and let $\mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } \in \mathcal { X }$ be two inputs. The privacy loss random variable $\mathcal { L }$ associated with releasing o is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { M } ( \mathbf { X } _ { 0 } ) } \| \mathcal { M } ( \mathbf { X } _ { 1 } ) \big ( o \big ) = \log \bigl ( \operatorname* { P r } ( \mathcal { M } ( \mathbf { X } _ { 0 } ) = o ) / \operatorname* { P r } ( \mathcal { M } ( \mathbf { X } _ { 1 } ) = o ) \bigr ) . } \end{array}\tag{2}
$$

For any two neighboring datasets Y and $\mathbf { Y } ^ { \prime }$ , a mechanism $\mathcal { M }$ satisfies pure $\boldsymbol { \varepsilon } { \mathrm { - D P } }$ if and only if $\mathrm { P r } \big ( \mathcal { L } _ { \mathcal { M } ( \mathbf { Y } ) \mid \mid \mathcal { M } ( \mathbf { Y ^ { \prime } } ) } > \varepsilon \big ) = 0$ , whereas $( \varepsilon , \delta ) – \mathrm { D P }$ is equivalent, up to a small loss in parameters, to the requirement that $\operatorname* { P r } \bigl ( \mathcal L _ { \mathcal { M } ( \mathbf { Y } ) \| \mathcal { M } ( \mathbf { Y } ^ { \prime } ) } > \varepsilon \bigr ) \leq \delta .$ . A relaxation of DP based on the Rényi divergence is:

Definition 3 (Rényi Differential Privacy (RDP)). [Mironov, 2017] A randomized mechanism M is said to have $( \alpha , \varepsilon ) – R D P$ iffor all pairs ofneighboring datasets $\mathbf { Y } , \mathbf { Y } ^ { \prime } $ , it holds that

$$
D _ { \alpha } ( \mathcal { M } ( \mathbf { Y } ) \| \mathcal { M } ( \mathbf { Y } ^ { \prime } ) ) \leq \varepsilon ,\tag{3}
$$

where $D _ { \alpha } ( P \| Q )$ is the Rényi divergence [Rényi, 1961] of order $\alpha > 1$ between distributions $P , Q .$

$$
D _ { \alpha } ( P \| Q ) = ( \alpha - 1 ) ^ { - 1 } \log \mathbb { E } _ { x \sim Q } \left( P / Q \right) ^ { \alpha } = ( \alpha - 1 ) ^ { - 1 } \log \mathbb { E } _ { x \sim P } \left( P / Q \right) ^ { \alpha - 1 } .\tag{4}
$$

Beyond defining RDP, Rényi divergence is also linked to several popular DP notions, such as $( \varepsilon , \delta )$ $\mathrm { D P } , \rho$ zero-concentrated ${ \sf D P } \left( \mathrm { z C D P } \right)$ , and Gaussian DP (GDP), in the sense that they can be expressed via Rényi divergence. The following proposition summarizes these relationships.

Proposition 1 (Relations between Rényi divergence and DP notions). Let M be a randomized mechanism, and $\mathbf { Y } , \mathbf { Y ^ { \prime } }$ be any pair of neighboring datasets satisfying $d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1$ . Then,

• M is $o f \varepsilon { - } D P$ if and only i $^ { \mathrm { \prime } } D _ { \infty } ( { \cal M } ( { \bf Y } ) \| { \cal M } ( { \bf Y } ^ { \prime } ) ) \le \varepsilon [ { \cal M } i r o n o \nu , 2 { \cal O } I 7 ] ^ { 1 }$

• M is $o f \rho { - } z C D P$ if and only $i f \ \forall \alpha \in ( 1 , \infty ) , D _ { \alpha } ( \mathcal { M } ( \mathbf { Y } ) \| \mathcal { M } ( \mathbf { Y } ^ { \prime } ) ) \le \rho \alpha$ [Bun and Steinke, 2016].

• If M is of $( \alpha , \varepsilon ) – R D P ,$ it is also $o f \left( \varepsilon + ( \alpha - 1 ) ^ { - 1 } \log { 1 / \delta } , \delta \right) \ – D P f o r \forall \ 0 < \delta < 1$ [Mironov, 2017].

$H \mathcal M$ is of $\cdot { \bf \Pi } _ { \rho - z C D P }$ it is also $o f \sqrt { 2 \rho } { \cdot } G D P .$ If M is of $\mathsf { \Pi } \mu { - } G D P ,$ it is also of $\dot { \mu } ^ { 2 } / 2 \ – z C D P$ [Dong et al., 2022].

## 3 A Unified Rényi Framework for Privacy Loss from Posterior Sampling

We develop a unified Rényi-divergence framework for characterizing the inherent privacy loss of posterior sampling, based on a decomposition of the privacy loss random variable between the posterior distributions formulated over two neighboring datasets.

## 3.1 Privacy Loss Decomposition for Neighboring Posteriors

Let $\mathbf { y } \in \mathcal { V }$ denote a single observation, its distribution is $p ( \mathbf { y } \vert \pmb { \theta } )$ with parameter $\theta \in \Theta$ , and $\pi ( \theta )$ is the prior distribution. Given a dataset $\mathbf { Y } = \{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { n }$ and assuming independence across the n observations, the posterior distribution of θ is

$$
\pi ( \pmb { \theta } | \mathbf { Y } ) = \frac { \pi ( \pmb { \theta } ) \prod _ { i = 1 } ^ { n } p ( \mathbf { y } _ { i } | \pmb { \theta } ) } { \int \prod _ { i = 1 } ^ { n } p ( \mathbf { y } _ { i } | \pmb { \theta } ) \pi ( \pmb { \theta } ) d \pmb { \theta } } = \frac { \pi ( \pmb { \theta } ) \ell ( \pmb { \theta } ; \mathbf { Y } ) } { p ( \mathbf { Y } ) } ,\tag{5}
$$

where $\begin{array} { r } { \ell ( \pmb { \theta } ; \mathbf { Y } ) \triangleq \prod _ { i = 1 } ^ { n } p ( \mathbf { y } _ { i } | \pmb { \theta } ) } \end{array}$ denotes the likelihood function and $\begin{array} { r } { p ( \mathbf { Y } ) = \int \ell ( \pmb { \theta } ; \mathbf { Y } ) \pi ( \pmb { \theta } ) d \pmb { \theta } } \end{array}$ is the normalizing constant. Consider any pair of neighboring datasets $d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1$ with corresponding posterior distributions, $\pi ( \pmb \theta | \mathbf Y )$ and $\mathbf { \bar { \theta } } \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } )$ , which we refer to as neighboring posteriors. The privacy loss random variable $c ( \pmb \theta )$ associated with releasing one posterior sample θ is

$$
c ( { \pmb \theta } ) = \log \left( \frac { \pi ( { \pmb \theta } | { \bf Y } ) } { \pi ( { \pmb \theta } | { \bf Y } ^ { \prime } ) } \right) = \log \left( \frac { \pi ( { \pmb \theta } ) ^ { 2 } \ell ( { \pmb \theta } ; { \bf Y } ) } { p ( { \bf Y } ) } \cdot \frac { p ( { \bf Y } ^ { \prime } ) } { \pi ( { \pmb \theta } ) \ell ( { \pmb \theta } ; { \bf Y } ^ { \prime } ) } \right) \mathrm { ~ p e r ~ E q . ~ } ( 5 )
$$

$$
= \underbrace { \log \left( \frac { \ell ( \pmb { \theta } ; \mathbf { Y } ) } { \ell ( \pmb { \theta } ; \mathbf { Y } ^ { \prime } ) } \right) } _ { \triangleq \mathbf { \Lambda } _ { c _ { 1 } ( \pmb { \theta } ) } } + \underbrace { \log \left( \frac { p ( \mathbf { Y } ^ { \prime } ) } { p ( \mathbf { Y } ) } \right) } _ { \triangleq \mathbf { \Lambda } _ { c _ { 2 } } } = c _ { 1 } ( \pmb { \theta } ) + c _ { 2 } .\tag{6}
$$

Here, $c _ { 1 } ( \pmb \theta )$ is the pointwise log-likelihood ratio evaluated at $\theta ,$ , while $c _ { 2 }$ represents the marginal log-likelihood ratio obtained after integrating out $\theta . \ c _ { 1 } ( \theta )$ and $c _ { 2 }$ are related; in particular, $c _ { 2 }$ can be expressed as the logarithm of an exponential moment of $c _ { 1 } ( \pmb \theta )$ under the posterior $\pi ( \pmb \theta | \mathbf Y )$ . This relation is formalized in Lemma 2, with the proof is provided in Appendix $\mathrm { A . 1 }$

Lemma 2 (Marginal-Pointwise identity). c = log $\mathbb { E } _ { \pmb { \theta } \sim \pi ( \pmb { \theta } | \mathbf { Y } ) } \left[ \exp \left( - c _ { 1 } ( \pmb { \theta } ) \right) \right]$ ] . Specifically, if there existfinite constants $a , b \in \mathbb { R }$ with $a < b$ such that $c _ { 1 } ( \pmb \theta ) \in [ a , b ] f o r \forall \pmb \theta \in \Theta$ , then $c _ { 2 } \in [ - b , - a ]$

## 3.2 Rényi Divergence Characterization

Building on the decomposition in Eq. (6), we establish an exact expression for the Rényi divergence of any order $\alpha > 1$ between neighboring posteriors in Theorem 3. The proof is provided in Appendix A.2.

Theorem 3 (Exact characterization of Rényi divergence between neighboring posteriors). Let $P =$ $\pi ( \pmb { \theta } | \mathbf { Y } )$ and $Q = \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } )$ be a pair ofneighboring posteriors. With $c _ { 1 } ( \pmb \theta )$ and $c _ { 2 }$ defined in Eq. (6), the Rényi divergence oforder $\alpha > 1$ between $P$ and $Q$ is

$$
D _ { \alpha } ( P \| Q ) = c _ { 2 } + ( \alpha - 1 ) ^ { - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim P } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \pmb { \theta } ) } \right]\tag{7}
$$

$$
= \log \left[ \left( \mathbb { E } _ { \theta \sim P } \left[ e ^ { - c _ { 1 } ( \theta ) } \right] \right) \cdot \left( \mathbb { E } _ { \theta \sim P } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \theta ) } \right] \right) ^ { \frac { 1 } { \alpha - 1 } } \right] p e r L e m m a \ : 2 .\tag{8}
$$

Eq. (8) suggests that the Rényi divergence of order $\alpha - D _ { \alpha } ( P \| Q )$ – between neighboring posteriors is fully determined by exponential moments $\mathbb { E } _ { \pmb { \theta } \sim P } [ e ^ { t c _ { 1 } ( \pmb { \theta } ) } ]$ , evaluated at $t = - 1$ and $t = \alpha - 1$ , of the log-likelihood ratio $c _ { 1 } ( \pmb \theta )$ under $P = \pi ( \pmb \theta | \mathbf { Y } )$ . This establishes Theorem 3 as the foundational result to build a unified framework for analyzing the inherent privacy guarantees of OPS.

Remark 4 (Reverse direction). Theorem 3 holds for any ordered pair of neighboring posteriors. Swapping the roles ofY and $\mathbf { Y ^ { \prime } }$ , let $\begin{array} { r } { \tilde { c } _ { 1 } ( \pmb { \theta } ) \triangleq \log \frac { \ell ( \pmb { \theta } ; \mathbf { Y } ^ { \prime } ) } { \ell ( \pmb { \theta } ; \mathbf { Y } ) } = - c _ { 1 } ( \pmb { \theta } ) } \end{array}$ ; then

$$
\begin{array} { r } { D _ { \alpha } ( Q \| P ) = \log \mathbb { E } _ { \theta \sim Q } \left[ e ^ { - \tilde { c } _ { 1 } ( \theta ) } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim Q } \left[ e ^ { ( \alpha - 1 ) \tilde { c } _ { 1 } ( \theta ) } \right] , } \end{array}\tag{9}
$$

with expectations taken under $Q = \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } )$ . Since $d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1 \Leftrightarrow d ( \mathbf { Y } ^ { \prime } , \mathbf { Y } ) = 1 , ( \alpha , \varepsilon ) { \cdot } R D P$ requires $\begin{array} { r } { \operatorname* { s u p } _ { ( \mathbf { Y } , \mathbf { Y ^ { \prime } } ) : d ( \mathbf { Y } , \mathbf { Y ^ { \prime } } ) = 1 } \operatorname* { m a x } \{ D _ { \alpha } ( \{ P \| Q \} , D _ { \alpha } ( Q \| P ) \} \le \varepsilon . } \end{array}$

As Eqs. (7), (8) and (9) depend on the specific neighboring pair $\mathbf { Y } , \mathbf { Y } ^ { \prime }$ under consideration, the task of deriving DP guarantees for OPS reduces to obtaining data-independent upper bounds for these exponential moments: of $c _ { 1 } ( \pmb \theta )$ under $\pi ( \pmb \theta | \mathbf Y )$ and of $\tilde { c } _ { 1 } ( \pmb \theta )$ under $\pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } )$ . To this end, we derive a series of results in Sections 4 to 6 under progressively weaker regularity conditions on $c _ { 1 } ( \pmb \theta )$ . These results significantly expand the class of Bayesian models for which explicit upper bounds on $D _ { \alpha }$ in both directions can be obtained, while also tightening the bounds available in the existing literature. Furthermore, the derived upper bounds on $\bar { D _ { \alpha } }$ can be interpreted under various DP notions, either through definitions or via well-established conversion results summarized in Proposition 1.

## 4 DP Guarantees of OPS with Uniformly Bounded Log-Likelihood Ratio

Under uniform boundedness of $c _ { 1 } ( \pmb \theta )$ , we derive explicit DP guarantees for OPS in Corollaries 5 and 6 based on Theorem 3, and demonstrate their application to categorical likelihoods in Example 7. Corollary 5 (Attainable RDP guarantees of OPS with uniformly bounded log-likelihood ratio). Suppose that $\dot { c } _ { 1 } ( \pmb \theta )$ is uniformly bounded, that is, there exist finite constants $a , \bar { b } \in$ R with $a < b$ such that $c _ { 1 } ( \pmb { \theta } ) \in [ a , b ]$ for all $\theta \in \Theta$ . Define $\Delta \triangleq b - a$ and $A \triangleq e ^ { - \Delta } \in ( 0 , 1 ) , B \triangleq e ^ { ( \alpha - 1 ) \Delta } > 1$ Then, for every $\alpha > 1$

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P | | Q ) , D _ { \alpha } ( Q | | P ) \} \leq \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) \triangleq \log \bigg ( \frac { ( \alpha - 1 ) ( B - A ) } { \alpha ( B - 1 ) } \bigg ) + \frac { 1 } { \alpha - 1 } \log \bigg ( \frac { B - A } { \alpha ( 1 - A ) } \bigg )\tag{10}
$$

and releasing OPS satisfies $( \alpha , \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) )$ -RDP.

The full proof is in Appendix A.3. We provide the proof sketch below. First, we show that $D _ { \alpha } ( P | | Q )$ is monotonically increasing with respect to the convex order of $c _ { 1 } ( \pmb \theta )$ ; Second, among all random variables supported on $[ a , b ]$ , the endpoint-supported binary random variable $X ^ { * } -$ defined by $\operatorname* { P r } ( X ^ { * } = a ) = 1 - p$ and $\operatorname* { P r } ( X ^ { * } = b ) = p .$ , with $\mathbb { E } ( X ^ { \ast } ) = \mathbb { E } [ c _ { 1 } ( \pmb { \theta } ) ]$ – attains the maximal convex order. Evaluating $D _ { \alpha } ( P | | Q )$ at $X ^ { * }$ thus yields an analytical upper bound for a given pair of neighboring datasets, parameterized by $p .$ . Finally, by maximizing this neighboring pair-specific upper bound over $p \in [ 0 , 1 ]$ ], we obtain the data-independent upper bound $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$

The bound is sharp over all random variables supported on $[ a , b ]$ , though not necessarily realized by a valid neighboring posterior pair. Since $\tilde { c } _ { 1 } ( \pmb \theta ) = - c _ { 1 } ( \pmb \theta ) \in [ - b , - a ]$ has the same range $\Delta ,$ , the same bound holds for $D _ { \alpha } ( Q \| P )$ by Remark 4. Corollary 5 reveals two insightful properties of $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ in Eq. (10) for a fixed $\alpha > 1$ . First, shift-invariance property: $\varepsilon _ { \mathrm { u b } } ( \alpha , \bar { \Delta } )$ depends solely on the range $\Delta = b - a$ , rather than the specific values of a and $b ,$ implying that a reparameterization that only shifts $c _ { 1 } ( \pmb \theta )$ does not affect the inherent DP guarantee offered by OPS. Second, monotonic increase of $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ with $\Delta \colon$ larger values of $\Delta$ lead to weaker privacy guarantees, which aligns with the intuition that greater dissimilarity between the two neighboring posteriors implies weaker intrinsic privacy protection provided by OPS.

Corollary 6 below recovers the existing pure-DP guarantees for the symmetric case $a = - b ( \mathbf { e . g . }$ Thm. 2 in Dimitrakakis et al. [2014], Thm. 8 in Dimitrakakis et al. [2017], and Thm. 1 in Wang et al. [2015]) as a special case; the proof is in Appendix $\mathrm { A . 4 }$

Corollary 6 (Pure-DP guarantee). Under the conditions in Corollary 5, for every $\alpha > 1$

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq b - a = \Delta = \operatorname* { s u p } _ { \alpha > 1 } \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) ,\tag{11}
$$

implying OPS satisfies pure ${ \varepsilon } { - } D P$ with $\varepsilon = \Delta$

While Corollary 6 recovers the pure-DP guarantee, Corollary 5 is strictly more informative in two respects. $F i r s t .$ it addresses the attainability gap in the non-degenerate case of $a < b$ . While $\Delta$ is a valid upper bound on $D _ { \alpha } ( P | | Q )$ , it is generally unattainable at finite α except $a = b$ . Specifically, attaining $\Delta = b - a$ would require simultaneously that $( \alpha - 1 ) ^ { - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim P } \overline { { [ e ^ { ( \alpha - 1 ) c _ { 1 } ( \pmb { \theta } ) } ] } } = b$ and $c _ { 2 } = - a$ . However, per Lemma $2 , c _ { 2 }$ decreases monotonically with $c _ { 1 } ( \pmb \theta )$ : the condition $c _ { 2 } = - a$ implies $c _ { 1 } ( \pmb { \theta } ) = a$ almost surely, whereas the condition $( \alpha - 1 ) ^ { - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim P } \big [ e ^ { ( \alpha - 1 ) c _ { 1 } ( \pmb { \theta } ) } \big ] = b$ requires $c _ { 1 } ( \pmb \theta ) = b$ almost surely. These two conditions cannot hold simultaneously unless $b = a .$ Corollary 5 resolves this attainability gap by explicitly accounting for the interdependence of $c _ { 1 } ( \pmb \theta )$ and $c _ { 2 }$

Second, $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ in Eq. (10) captures the full spectrum of Rényi divergence across all $\alpha > 1$ whereas Corollary 6 and prior works provide only the static, worst-case bound corresponding to the limit $\alpha \to \infty$ . This order-dependent profile in Corollary 5 thus enables sharper RDP accounting for composition. For fixed $\Delta$ , expanding $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ via Taylor expansion as $\alpha \to \infty$ gives $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) =$ $\Delta - \dot { \alpha } ^ { - 1 } \log ( \alpha ( 1 - e ^ { - \Delta } ) ) \dot { + } \mathcal { O } ( \alpha ^ { - \top } )$ , that is, $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ converges to the pure-DP bound $\Delta$ at the rate of $\Theta ( \log \alpha / \alpha )$ (see Appendix A.4 for the proof).

Example 7 (Categorical likelihood with arbitrary prior). Let $Y = \{ y _ { i } \} _ { i = 1 } ^ { n }$ be a dataset, where each $y _ { i } \in \{ 1 , \ldots , K \}$ is a categorical outcome. Suppose $y _ { i } | \pmb \theta \stackrel { i . i . d } { \sim }$ Categorical(θ), where $\pmb { \theta } = ( \theta _ { 1 } , \dots , \theta _ { K } )$ with $\textstyle \sum _ { j = 1 } ^ { K } \theta _ { j } = 1$ and $\theta _ { j } = \mathrm { P r } ( y _ { i } = j )$ . Let $\begin{array} { r } { s _ { j } ( Y ) = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ y _ { i } = j \} } \end{array}$ denote the j-th category count.

(1) Bounded neighboring $( Y , Y ^ { \prime } ) .$ : there exist distinct indices $j \neq k f o r j , k \in \{ 1 , \ldots , K \}$ such that $s _ { j } ( Y ) = s _ { j } ( Y ^ { \prime } ) + 1 , s _ { k } ( Y ) = s _ { k } ( Y ^ { \prime } ) - 1$ , and $s _ { \ell } ( Y ) = s _ { \ell } ( Y ^ { \prime } ) f o r { a l l } \ell \not \in \{ j , k \}$

$$
c _ { 1 , \mathrm { b o u n d e d } } ( \theta ) = \log \frac { \ell ( \theta ; Y ) } { \ell ( \theta ; Y ^ { \prime } ) } = \sum _ { \ell = 1 } ^ { K } \left( s _ { \ell } ( Y ) - s _ { \ell } ( Y ^ { \prime } ) \right) \log \theta _ { \ell } = \log \frac { \theta _ { j } } { \theta _ { k } } ,
$$

$$
D _ { \alpha , \mathrm { b o u n d e d } } \left( P \| Q \right) \leq \operatorname* { m a x } _ { j \neq k } \left\{ \log \mathbb { E } _ { \theta \sim P } \left[ \theta _ { k } / \theta _ { j } \right] + \left( \alpha - 1 \right) ^ { - 1 } \log \mathbb { E } _ { \theta \sim P } \left[ \left( \theta _ { j } / \theta _ { k } \right) ^ { \alpha - 1 } \right] \right\} .\tag{12}
$$

(2) Unbounded neighboring $( Y , Y ^ { \prime } ) { \mathrm { : } }$ : there exists a unique index $j \in \{ 1 , \ldots , K \}$ such that $s _ { j } ( Y ) =$ $s _ { j } ( Y ^ { \prime } ) + 1 , a n d s _ { \ell } ( Y ) = s _ { \ell } ( Y ^ { \prime } ) f o r a l l \ell \neq j$

$$
c _ { 1 , \mathrm { u n b o u n d e d } } ( \pmb \theta ) = \log \frac { \ell ( \pmb \theta ; Y ) } { \ell ( \pmb \theta ; Y ^ { \prime } ) } = \sum _ { \ell = 1 } ^ { K } \left( s _ { \ell } ( Y ) - s _ { \ell } ( Y ^ { \prime } ) \right) \log \theta _ { \ell } = \log \theta _ { j } ,
$$

$$
D _ { \alpha , \mathrm { u n b o u n d e d } } ( P \| Q ) \leq \operatorname* { m a x } _ { j \in \{ 1 , \dots , K \} } \Big \{ \log \mathbb { E } _ { \theta \sim P } \big [ \theta _ { j } ^ { - 1 } \big ] + ( \alpha - 1 ) ^ { - 1 } \log \mathbb { E } _ { \theta \sim P } \big [ \theta _ { j } ^ { \alpha - 1 } \big ] \Big \} .\tag{13}
$$

Eqs. (12) and (13) are direct applications of Theorem 3; the proof is in Appendix A.7. Since $c _ { 1 , \mathrm { b o u n d e d } } ( \pmb { \theta } )$ and $c _ { 1 , \mathrm { u n b o u n d e d } } ( \pmb { \theta } )$ diverge as $\theta _ { i } \to 0$ , we restrict the parameter space to obtain explicit DP guarantees. Specifically, suppose $\mathsf { \bar { \theta } } _ { j } \in [ \eta , 1 - \eta ]$ for some $\eta \in ( 0 , 1 / 2 )$ and all $j ;$ then $\begin{array} { r } { c _ { 1 , \mathrm { b o u n d e d } } ( \pmb { \theta } ) = \log ( \theta _ { j } / \theta _ { k } ) \in [ - \log \frac { 1 - \eta } { \eta } , \log \frac { \tilde { 1 } - \eta } { \eta } ] } \end{array}$ and $c _ { 1 , \mathrm { u n b o u n d e d } } ( \pmb { \theta } ) = \log \theta _ { j } \in [ \log \eta , \log ( 1 - \eta ) ]$ with ranges $\begin{array} { r } { \Delta _ { \mathrm { b o u n d e d } } = 2 \log { \frac { 1 - \eta } { \eta } } } \end{array}$ and $\begin{array} { r } { \dot { \Delta } _ { \mathrm { u n b o u n d e d } } = \log \frac { 1 - \eta } { \eta } } \end{array}$ , respectively. Hence the bounded neighboring dominates with $\begin{array} { r } { \Delta = 2 \operatorname* { l o g } \frac { 1 - \eta } { \eta } } \end{array}$ . Some concrete examples of the inherent RDP guarantees for OPS when $\alpha = 2$ (where Eq. (10) reduces to $\varepsilon _ { \mathrm { u b } } ( 2 , \Delta ) = \log ( [ 1 + \cosh ( \Delta ) ] / 2 )$ with $\begin{array} { r } { \Delta = 2 \log \frac { 1 - \eta } { \eta } ) } \end{array}$ and $\alpha = \infty$ are given below:

• If $\eta = 0 . 0 1$ , Corollary 5 yields $( 2 , \sim 7 . 8 0 4 )$ -RDP, and Corollary 6 yields $( \infty , \sim 9 . 1 9 0 )$ -RDP.

$\mathrm { I f } \ \eta = 0 . 1$ , Corollary 5 yields (2, ∼ 3.033)-RDP, and Corollary 6 yields $( \infty , \sim 4 . 3 9 4 ) – \mathrm { R D P }$

• $\mathrm { f } \ \eta = 0 . 3$ , Corollary 5 yields (2, ∼ 0.646)-RDP, and Corollary 6 yields $( \infty , \sim 1 . 6 9 5 ) – \mathrm { R D P } .$

We remark that Eqs. (12) and (13) can be evaluated exactly for specific choices of the prior to produce distribution-specific bounds. Examples include the Bernoulli-Beta pair (Appendix A.7.3) and Categorical-Dirichlet pair (Appendix A.7).

## 5 DP Guarantees of OPS with sub-Gaussian Log-Likelihood Ratio

In this section, we establish explicit DP guarantees for OPS under a weaker assumption than the uniform boundedness assumption in Section 4, that ${ \mathrm { i s } } ,$ when $c _ { 1 } ( \pmb \theta )$ is sub-Gaussian. The main result (Corollary 8) applies to a broader class of Bayesian models than those covered in Corollary 5.

Corollary 8 (RDP guarantees of OPS with sub-Gaussian log-likelihood ratio). Suppose that there exists a variance proxy $\sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } \in ( 0 , \infty )$ such that $f o r \forall t \in \mathbb { R } ,$ , it holds that log $\mathbb { E } _ { \pmb { \theta } \sim P } \big ( \bar { e } ^ { t ( c _ { 1 } ( \pmb { \theta } ) - \mu _ { \mathbf { Y } } , \mathbf { Y } ^ { \prime } ) } \big ) \leq$ $t ^ { 2 } \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } / 2$ where $\mu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } { = } \mathbb { E } _ { \pmb { \theta } \sim P } \left[ c _ { 1 } ( \pmb { \theta } ) \right]$ ]. Define

$$
\begin{array} { r } { \bar { \sigma } ^ { 2 } \triangleq \operatorname* { s u p } _ { ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) : d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1 } \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } , } \end{array}\tag{14}
$$

where the supremum is over all ordered neighboring pairs. Then,for every $\alpha > 1$

$$
\begin{array} { r } { \operatorname* { m a x } \{ D _ { \alpha } ( P | | Q ) , D _ { \alpha } ( Q | | P ) \} \leq \varepsilon _ { \mathrm { s u b G a u s } } ( \alpha , \bar { \sigma } ^ { 2 } ) \triangleq \alpha \bar { \sigma } ^ { 2 } / 2 , } \end{array}\tag{15}
$$

and releasing OPS satisfies $( \alpha , \varepsilon _ { \mathrm { s u b G a u s } } ( \alpha , \bar { \sigma } ^ { 2 } ) ) – R D P .$

The proof is in Appendix A.5. Unlike the uniformly bounded case in Corollary $5 , \varepsilon _ { \mathrm { s u b G a u s } } ( \alpha , \bar { \sigma } ^ { 2 } )$ in Eq. (15) scales linearly with α and diverges as $\alpha \to \infty$ . Therefore, the sub-Gaussian regime does not yield pure DP guarantees for OPS. Nevertheless, this linear dependence on α aligns naturally with other relaxed DP notions. Specifically, since Eq. (15) holds for all $\alpha \in ( 1 , \infty )$ , OPS satisfies $\rho { \mathrm { - } } \mathbf { Z } \mathbf { C D P }$ with $\rho = \bar { \sigma } ^ { 2 } / 2 , \mu { \cdot } \mathrm { G D P }$ with $\mu = \bar { \sigma }$ and $\begin{array} { r } { ( \alpha \bar { \sigma } ^ { 2 } / 2 + \frac { \log ( 1 / \delta ) } { \alpha - 1 } , \delta ) { \ - \mathrm { D P } } } \end{array}$ for $\forall 0 < \delta < 1$ via Proposition 1. Furthermore, by Corollary 3 in Mironov [2017], the Gaussian mechanism with scale parameter $\sigma ^ { 2 }$ satisfies $( \alpha , \alpha / ( 2 \sigma ^ { 2 } ) ) – \mathrm { R D P } .$ This facilitates an intuitive interpretation of Corollary 8:

Claim: When $c _ { 1 } ( \pmb \theta )$ is sub-Gaussian, the inherent DP guarantees of OPS are conceptually equivalent to applying the Gaussian mechanism with scale parameter $\bar { \sigma } ^ { - 2 }$ defined in Eq. (14).

We apply Corollary 8 and Theorem 3 to Generalized Linear Models (GLMs) with Gaussian priors (Section 5.1), and to Gaussian likelihood (Section 5.2), including the Gaussian-Gaussian conjugate model and Bayesian linear regression with Gaussian priors on regression coefficients. Our results show that the choice of neighboring relation can fundamentally affect the tail behavior of $c _ { 1 } ( \pmb \theta )$ leading to qualitatively different inherent privacy guarantees.

## 5.1 Generalized Linear Models with Gaussian Prior

Let each data record be $\left( \mathbf { x } _ { i } , y _ { i } \right)$ , where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ is the covariate vector and $y _ { i } \in \mathcal { V }$ is the response. Let $\mathbf { Y } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ denote the dataset. Conditioned on $\mathbf { x } _ { i }$ and the parameter $\beta \in \mathbb { R } ^ { d }$ , the response $y _ { i }$ follows an exponential family distribution: $p ( y _ { i } | \mathbf { x } _ { i } , \beta ) = \exp ( y _ { i } \mathbf { x } _ { i } ^ { \top } \beta - \psi ( \mathbf { x } _ { i } ^ { \top } \beta ) + h ( y _ { i } ) )$ ), where $\psi ( \cdot )$ is the log-partition function and $\exp ( h ( y _ { i } ) )$ is the base measure (which may include a known dispersion parameter, such as $\sigma ^ { 2 }$ in Gaussian likelihood). We show that, under a bounded record-level score condition, $c _ { 1 } ( \beta )$ for GLMs with a Gaussian prior on $\beta$ is sub-Gaussian under the posterior $\pi ( \cdot | \mathbf { Y } )$ . This yields explicit OPS privacy guarantees for both bounded and unbounded neighboring relations, as formalized in Proposition 9. The proof is in Appendix A.8.

Proposition 9 (GLMs with Gaussian prior). Suppose there exists afinite constant $L > 0$ such that for every $\left( \mathbf { x } _ { i } , y _ { i } \right)$ and every $\begin{array} { r } { \beta , \| \nabla _ { \beta } \log p ( y _ { i } | \mathbf { x } _ { i } , \bar { \beta } ) \| \le L , } \end{array}$ and impose a Gaussian prior on $\beta ,$ , i.e. $\boldsymbol { \pi } ( \beta ) \mathop { = } \mathcal { N } ( 0 , \tau ^ { 2 } \mathbf { I } _ { d } )$ . Then:

(1) Bounded neighboring $( { \bf Y } , { \bf Y ^ { \prime } } ) \colon c _ { 1 } ( \beta )$ is sub-Gaussian under $\pi ( \beta | \mathbf { Y } )$ with variance proxy $\sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } \leq 4 L ^ { 2 } \tau ^ { 2 }$ for every ordered neighboring pair and, per Corollary 8,

$$
\begin{array} { r } { \operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \le \varepsilon _ { \mathrm { G L M , b o u n d e d } } ( \alpha ) \triangleq \alpha \bar { \sigma } ^ { 2 } / 2 \le 2 \alpha L ^ { 2 } \tau ^ { 2 } . } \end{array}\tag{16}
$$

(2) Unbounded neighboring $( { \bf Y } , { \bf Y } ^ { \prime } ) \colon c _ { 1 } ( \beta )$ is sub-Gaussian under $\pi ( \beta | \mathbf { Y } )$ with variance proxy $\sigma _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } \leq L ^ { 2 } \tau ^ { 2 }$ for every ordered neighboring pair and, per Corollary 8,

$$
\begin{array} { r } { \operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \varepsilon _ { \mathrm { G L M , u n b o u n d e d } } ( \alpha ) \stackrel { \Delta } { = } \alpha \bar { \sigma } ^ { 2 } / 2 \leq \alpha L ^ { 2 } \tau ^ { 2 } / 2 . } \end{array}\tag{17}
$$

Eqs. (16) and (17) taken together, releasing OPS satisfies $( \alpha , 2 \alpha L ^ { 2 } \tau ^ { 2 } )$ -RDP.

For GLMs with Gaussian priors, as in Example 7, the bounded neighboring relation yields a larger worst-case Rényi divergence, specifically incurring a 4-fold increase relative to the unbounded case. Under bounded neighboring, OPS in this setting can also be viewed as a regularized exponential mechanism, and the bound in Eq. (16) matches the strongly log-concave Lipschitz-perturbation analysis of Gopi et al. [2022]. However, our formulation distinguishes itself by further accommodating unbounded neighboring – yielding Eq.(17) – and by embedding the GLM case within a broader framework that supports the non-log-concave-tail analyses developed in Section 5.2.

Proposition 9 shows how the bounded-score constant $L ,$ prior variance $\tau ^ { 2 }$ , and neighboring relations jointly determine the inherent DP guarantee of OPS from GLMs with Gaussian prior. First, ε in Eqs. (16) and (17) scale quadratically with $L .$ Thus, records with large score norm can substantially weaken the intrinsic DP guarantee offered by OPS, consistent with the common practice of data clipping and normalization in DP procedures. Second, ε<sub>GLM</sub> grows linearly with the prior variance $\tau ^ { 2 } . \mathrm { \ r A }$ larger $\tau ^ { 2 }$ corresponds to a less informative prior, making the posterior more dominated by the likelihood formulated from the private data and thus weakening the inherent DP guarantees of OPS.

The bound $\varepsilon _ { \mathrm { G L M } }$ does not explicitly depend on the data size $n ,$ as a consequence of the worstcase uniform bound used in our proof. Specifically, we control the variance proxy $\sigma _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 }$ via a uniform score bound $L$ and the strong log-concavity of the posterior induced by the Gaussian prior, whose curvature governed by $\tau ^ { 2 }$ . Under these conditions, the only curvature guaranteed uniformly over all datasets and $\beta$ values comes from the prior. Deriving a sharper explicit n-dependent guarantee is possible but would require showing that the likelihood contributes a nontrivial curvature, such as by establishing an n-dependent lower bound on the Hessian of the negative log-posterior $\begin{array} { r } { { \cal { H } } ( \beta ) { = } { \dot { \sum _ { i = 1 } ^ { n } } } \psi ^ { \prime \prime } ( { \bf { x } } _ { i } ^ { \top } { \tilde { \beta } } ) { \bf { x } } _ { i } { \bf { x } } _ { i } ^ { \top } { + } \dot { \tau } ^ { - 2 } I _ { d } } \end{array}$ . However, the likelihood curvature $\psi ^ { \prime \prime } ( \mathbf { x } _ { i } ^ { \top } \mathbf { \bar { \boldsymbol { \beta } } } )$ can be arbitrarily close to 0 in the worst case: in logistic regression, $\psi ^ { \prime \prime } ( \eta ) { = } \sigma ( \eta ) ( 1 - \sigma ( \eta ) )$ with $\sigma ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ vanishes as $| { \bf x } _ { i } ^ { \top } \beta |  \infty$ . The likelihood thus provides no guaranteed curvature in general, leaving the Gaussian prior as the only universal source of strong convexity $H ( \beta ) \succeq \tau ^ { - 2 } \check { I _ { d } }$ . For instance, if $\textstyle \sum _ { i = 1 } ^ { n } \psi ^ { \prime \prime } ( \bar { \mathbf { x } _ { i } ^ { \intercal } } \beta ) \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \intercal } \ \succeq \ \bar { k } ( n ) \mathbf { I } _ { d }$ uniformly over all admissible datasets and $\beta$ for some $k ( n )$ increasing in $n ,$ then $\tau ^ { - 2 }$ can be replaced by $\tau ^ { - 2 } + k ( n )$ , yielding an n-dependent guarantee for a more restricted model class. We leave developing sharper, n-dependent DP guarantees under additional curvature or design assumptions for future investigation.

Example 10 (Logistic regression with Gaussian prior). Logistic regression is a special case ofthe GLM above with binary response $y _ { i } \in \{ 0 , 1 \}$ }, log-partitionfunction $\bar { \psi } ( t ) = \log ( 1 + e ^ { t } )$ and $h ( \cdot ) = 0$ Suppose $\| \mathbf { x } _ { i } \| \leq R < \infty \dot { f o r } \forall i ,$ then the condition in Proposition 9 holds with $L = R$

(1) Bounded neighboring $( { \bf Y } , { \bf Y ^ { \prime } } )$ : WLOG, assume $( \mathbf { Y } , \mathbf { Y } ^ { \prime } )$ differ in the n-th observation.

$$
\begin{array} { r l } & { \operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \varepsilon _ { \mathrm { l o g i s t i c , b o u n d e d } } ( \alpha ) } \\ & { \qquad = \left\{ \begin{array} { l l } { \alpha R ^ { 2 } \tau ^ { 2 } / 2 } & { i f \mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } ( a n d t h u s y _ { n } \neq y _ { n } ^ { \prime } ) } \\ { 2 \alpha R ^ { 2 } \tau ^ { 2 } } & { i f \mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime } ( y _ { n } \neq y _ { n } ^ { \prime } o r y _ { n } = y _ { n } ^ { \prime } ) } \end{array} \right. . } \end{array}\tag{18}
$$

(2) Unbounded neighboring $( { \bf Y } , { \bf Y ^ { \prime } } )$ : Letting $L = R$ in Eq. (17) yields

$$
\begin{array} { r } { \operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \varepsilon _ { \mathrm { l o g i s t i c , u n b o u n d e d } } ( \alpha ) = \alpha R ^ { 2 } \tau ^ { 2 } / 2 . } \end{array}\tag{19}
$$

Example 10 suggests that replacing a binary label $y _ { n }$ incurs the same worst-case privacy loss as removing an entire record $\left( \mathbf { x } _ { n } , y _ { n } \right)$ while replacing the covariate vector ${ \bf x } _ { n }$ results in a variance proxy 4-fold that of a record removal. It is worth noting that Theorem 16 in Geumlek et al. [2017] examines the same Bayesian logistic regression setting. However, their $( \alpha , 2 \alpha R ^ { 2 } \tau ^ { 2 } ) – \mathbf { R D P }$ guarantee for OPS is restricted to the bounded setting with $\mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } .$ , whereas our results cover all neighboring scenarios; in their setting $( \mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } )$ , our bound $\alpha R ^ { 2 } \tau ^ { 2 } / 2$ reduces theirs by 75%, because label replacement cancels the log-partition terms so that $c _ { 1 } ( \beta )$ is R-Lipschitz rather than 2R-Lipschitz.

## 5.2 Gaussian likelihood with Gaussian prior & linear regression with Gaussian prior

We analyze two foundational Bayesian models not previously studied for the inherent DP of OPS: (1) univariate Gaussian likelihood with a Gaussian prior for its mean and known variance (Proposition 11); (2) linear regression with a multivariate Gaussian prior for its coefficients and known variance (Proposition 12). Notably, both models exhibit fundamentally different privacy profiles depending on neighboring relation: $c _ { 1 }$ is sub-Gaussian under bounded neighboring, but transitions to a non-sub-Gaussian regime under unbounded neighboring; moreover, the two directions of the Rényi divergence also differ in their range of finiteness.

Proposition 11 (Gaussian-Gaussian conjugate pair with known variance). Let $\mathbf { Y } = \{ y _ { i } \} _ { i = 1 } ^ { n }$ be the dataset, where $y _ { i } | \theta \stackrel { i . i . d } { \sim } \mathcal { N } ( \theta , \sigma ^ { 2 } )$ with known variance $\sigma ^ { 2 }$ . With Gaussian prior on $\theta \colon \pi ( \theta ) =$ $\mathcal { N } ( \mu _ { 0 } , \tau ^ { 2 } )$ , the posterior is $\theta | \dot { Y } \sim \dot { \mathcal { N } } ( \mu _ { Y } , v _ { Y } ^ { 2 } )$ , where $v _ { Y } ^ { 2 } = \sigma ^ { 2 } / \mu$ and $\begin{array} { r } { \mu _ { Y } = \mu _ { 0 } + \sum _ { i = 1 } ^ { n } ( y _ { i } - \dot { \mu _ { 0 } } ) / \omega , } \end{array}$ with $\omega = n + \sigma ^ { 2 } / \tau ^ { 2 }$

(1) Bounded neighboring $( { \bf Y } , { \bf Y ^ { \prime } } )$ : WLOG, assume Y, Y<sup>′</sup> differ in their n-th record $y _ { n }$ and $y _ { n } ^ { \prime } .$ Then $c _ { 1 } ( \theta )$ is Gaussian with variance $\sigma _ { Y , Y ^ { \prime } } ^ { 2 } = ( y _ { n } - y _ { n } ^ { \prime } ) ^ { 2 } / ( \stackrel { . . . } { \sigma } ^ { 2 } \omega )$ . Ifall records lie in an interval oflength 2kσ for some $k > 0 ,$ , then $\sigma _ { Y , Y ^ { \prime } } ^ { 2 } \leq \bar { \sigma } ^ { 2 } = 4 k ^ { 2 } / \omega$ , and per Corollary 8

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \varepsilon _ { \mathrm { G G , b o u n d e d } } ( \alpha , \omega ) \triangleq \alpha { \bar { \sigma } } ^ { 2 } / 2 = 2 k ^ { 2 } \alpha / \omega .\tag{20}
$$

(2) Unbounded neighboring $( { \bf Y } , { \bf Y ^ { \prime } } )$ : WLOG, assume $\mathbf { Y } ^ { \prime }$ is obtained by removing the n-th record $y _ { n }$ in Y. Then $c _ { 1 } ( \theta )$ is not sub-Gaussian; nevertheless, each term in the exact expression of the Rényi divergence $D _ { \alpha } ( P \| Q )$ in Eq. (8) can be analytically derived using the momentgeneratingfunctions (MGFs) ofthe non-central $\chi ^ { 2 }$ distribution with one degree offreedom and non-centrality parameter $( \mu _ { Y } - y _ { n } ) ^ { 2 } / v _ { Y } ^ { 2 }$ . Specifically,

$$
D _ { \alpha } ( P \| Q ) = A ( \alpha , \omega ) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } B ( \alpha , \omega )\tag{21}
$$

$$
A ( \alpha , \omega ) = - \frac { 1 } { 2 } \log \left( 1 - \omega ^ { - 1 } \right) - \frac { \log \left( 1 + \frac { \alpha - 1 } { \omega } \right) } { 2 ( \alpha - 1 ) } , \quad B ( \alpha , \omega ) = \left[ \frac { 1 } { 1 - \omega ^ { - 1 } } - \frac { 1 } { 1 + \frac { \alpha - 1 } { \omega } } \right] ,
$$

$$
\begin{array} { r } { D _ { \alpha } ( Q \| P ) = \left\{ \begin{array} { l l } { \widetilde { A } ( \alpha , \omega ) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \widetilde { B } ( \alpha , \omega ) } & { i f 1 < \alpha < \omega , } \\ { \infty } & { i f \alpha \ge \omega , } \end{array} \right. } \end{array}\tag{22}
$$

$$
\tilde { A } ( \alpha , \omega ) = - \frac { 1 } { 2 } \log \frac { \omega } { \omega - 1 } - \frac { 1 } { 2 ( \alpha - 1 ) } \log \frac { \omega - \alpha } { \omega - 1 } , \quad \tilde { B } ( \alpha , \omega ) = \frac { \alpha \omega } { ( \omega - 1 ) ( \omega - \alpha ) } .
$$

If for all i, $| y _ { i } - \mu _ { Y } | \leq k ^ { * } \sigma$ for some $k ^ { * } > 0 ,$ , then for $1 < \alpha < \omega$

$$
\begin{array} { r l } & { \operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \varepsilon _ { \mathrm { G G , u n b o u n d e d } } ( \alpha , \omega ) } \\ & { \qquad = \operatorname* { m a x } \Big \{ A ( \alpha , \omega ) + \frac { k ^ { * 2 } } { 2 } B ( \alpha , \omega ) , \ \widetilde { A } ( \alpha , \omega ) + \frac { k ^ { * 2 } } { 2 } \widetilde { B } ( \alpha , \omega ) \Big \} . } \end{array}\tag{23}
$$

The proof is in Appendix A.9. To our knowledge, Proposition 11 gives the first quantitative characterization of how sample size n, prior variance $\tau ^ { 2 }$ , and neighboring relation jointly determine the inherent DP guarantees of OPS in the Gaussian-Gaussian models. The key takeaways are:

Prior regularization. In both neighboring relations, the influence of the prior is characterized by the precision term $\omega = n + \sigma ^ { 2 } / \tau ^ { 2 }$ , where the ratio $\sigma ^ { 2 } / \tau ^ { 2 }$ can be interpreted as the “effective sample size” contributed by the prior. A more informative prior (smaller $\tau ^ { 2 } )$ increases $\omega ,$ leading to smaller values for both $\varepsilon _ { \mathbf { G G } }$ <sub>, bounded</sub> and $\varepsilon _ { \mathrm { G G } }$ <sub>unbounded</sub>, thereby tightening the privacy loss bounds.

Bounded neighboring. The worst-case privacy loss of OPS is controlled by the maximum difference between possible replacements, as shown in $\operatorname { E q . } \left( 2 0 \right)$ . Specifically, when the substituted records are identical $( \mathrm { i } . \mathrm { e } . , y _ { n } = y _ { n } ^ { \prime } )$ , both directions vanish. Moreover, for fixed $\alpha ,$ k and $\sigma ^ { 2 } , \varepsilon _ { \mathrm { G G , b o u n d e d } } ( \alpha , \omega ) =$ $2 k ^ { 2 } \alpha / \omega \xrightarrow { } 0$ as n → ∞ or $\tau ^ { 2 } \to 0$ . This indicates that OPS can achieve arbitrarily strong privacy through increasing the data size or decreasing the prior variance.

Unbounded neighboring. $D _ { \alpha } ( P \| Q )$ in Eq. (21) decomposes into a structural scale-shift term $A ( \alpha , \omega )$ , and a data-dependent location-shift term $( \mu _ { Y } - y _ { n } ) ^ { 2 } B ( \alpha , \omega ) / ( 2 \sigma ^ { 2 } )$ . The lower bound of $D _ { \alpha } ( P \| Q )$ is attained when the removed record $y _ { n }$ coincides with the original posterior mean $\mu _ { Y }$ . In this case, while the removal does not shift the posterior mean, it reduces the posterior precision. This change in the "sharpness" of the distribution is a detectable signal of the record’s presence. Thus, $D _ { \alpha } ( P \| Q )$ reduces to a strictly positive baseline $A ( \alpha , \omega ) \sim$ – the minimum unavoidable privacy loss incurred by removal; likewise, $D _ { \alpha } ( Q \| P )$ reduces to $\bar { A } ( \alpha , \omega ) > 0$ for addition (Claim A.9).

The two-sided upper bound is in Eq. (23), where $k ^ { * }$ represents the maximum deviation of a potentially removed record from $\mu _ { Y }$ , scaled by $\sigma . \ \mathrm { A s } \ n \to \infty , \omega \to \infty ; A , \widetilde { A } = { \mathcal { O } } ( \omega ^ { - 2 } )$ and $B , \widetilde { B } = \mathcal { O } ( \omega ^ { - 1 } )$ , so $\varepsilon _ { \mathrm { G G , u n b o u n d e d } } ( \alpha , \omega )$ vanishes and the admissible range $1 < \alpha < \omega$ expands. However, the two directions behave differently: $D _ { \alpha } ( P \| Q )$ (removal) is finite for all $\alpha > 1$ , whereas $D _ { \alpha } ( Q \| P )$ (addition) diverges for $\alpha \geq \omega$ , since removing a record inflates the posterior variance so that $\pi ( \theta | { \bf \ddot { Y } } ^ { \prime } )$ has heavier tails than $\pi ( \theta | \mathbf { Y } )$ . Consequently, OPS does not satisfy pure DP under unbounded neighboring, and the neighboring relation affects not only the magnitude of the privacy loss but also the range of Rényi orders over which a finite guarantee exists.

Proposition 12 (Linear regression with Gaussian prior). Let each data record be $\left( \mathbf { x } _ { i } , y _ { i } \right)$ , where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ is the covariate vector, $y _ { i } \in \mathbb R$ is the response, and $\mathbf { \Phi } _ { J i } | \mathbf { x } _ { i } , \beta \sim { \mathcal { N } } ( \mathbf { x } _ { i } ^ { \top } \beta , \sigma ^ { 2 } )$ with known variance $\sigma ^ { 2 }$ . Let $\mathbf { Y } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ be the dataset, $\mathbf { x } \in \mathbb { R } ^ { n \times d }$ be the design matrix, $\mathbf { y } \in \mathbb { R } ^ { n }$ be the response vector. With Gaussian prior $\boldsymbol { \pi } ( \beta ) \mathop { = } \mathcal { N } ( 0 , \tau ^ { 2 } \mathbf { I } _ { d } )$ , the posterior $\beta | { \bf Y } \sim \mathcal { N } ( \pmb { \mu _ { \mathrm { Y } } } , \pmb { \Sigma _ { \mathrm { Y } } } )$ , where $\Sigma _ { \mathbf { Y } } ^ { - 1 } = { \sigma } ^ { - 2 } { \mathbf { x } } ^ { \top } { \mathbf { x } } + { \sigma } ^ { - 2 } { \mathbf { I } } _ { d }$ and $\mu _ { \mathbf { Y } } { = } \Sigma _ { \mathbf { Y } } \left( \sigma ^ { - 2 } { \mathbf { x } } ^ { \top } \mathbf { y } \right)$ . Assume $\left\| \mathbf { x } _ { i } \right\| \leq R < \infty f o t$ all i.

(1) Bounded neighboring $( \mathbf { Y } , \mathbf { Y } ^ { \prime } )$ with $\mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } .$ : WLOG, suppose the n-th record $\left( \mathbf { x } _ { n } , y _ { n } \right)$ in Y is replaced by $\left( \mathbf { x } _ { n } , y _ { n } ^ { \prime } \right)$ in $\mathbf { \widetilde { Y } ^ { \prime } } ,$ , where $y _ { n } \ne y _ { n } ^ { \prime }$ . Then $c _ { 1 } ( \beta )$ is Gaussian under $\pi ( \beta | \mathbf { Y } )$ with variance $\begin{array} { r } { \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } = ( y _ { n } - y _ { n } ^ { \prime } ) ^ { 2 } \mathbf { x } _ { n } ^ { \top } \Sigma _ { \mathbf { Y } } \mathbf { x } _ { n } / \sigma ^ { 4 } . \ I f \left| y _ { n } - y _ { n } ^ { \prime } \right| \leq \frac { \gamma } { 2 } } \end{array}$ kσ for some $k > 0$ , then $\sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } \leq \bar { \sigma } ^ { 2 } = 4 k ^ { 2 } \tau ^ { 2 } R ^ { 2 } / \sigma ^ { 2 }$ , and per Corollary 8

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \varepsilon _ { \mathrm { L R , b o u n d e d } } ( \alpha ) \triangleq \alpha { \bar { \sigma } } ^ { 2 } / 2 = 2 \alpha k ^ { 2 } \tau ^ { 2 } R ^ { 2 } / \sigma ^ { 2 } .\tag{24}
$$

(2) Unbounded neighboring $( { \bf Y } , { \bf Y ^ { \prime } } )$ : WLOG, suppose $\mathbf { Y } ^ { \prime }$ is obtained by removing the n-th record $\left( \mathbf { x } _ { n } , y _ { n } \right)$ in Y. $c _ { 1 } ( \beta )$ is not sub-Gaussian under $\pi ( \beta | \mathbf { Y } )$ ). Define $\omega _ { n } = \sigma ^ { 2 } / ( \mathbf { x } _ { n } ^ { \top } \Sigma _ { \mathbf { Y } } \mathbf { x } _ { n } )$ . With $A , B , { \widetilde { A } } , { \widetilde { B } }$ defined in Proposition 11,

$$
D _ { \alpha } ( P \| Q ) = A ( \alpha , \omega _ { n } ) + \frac { ( \mathbf { x } _ { n } ^ { \top } \pmb { \mu } _ { \mathbf { Y } } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } B ( \alpha , \omega _ { n } ) ,\tag{25}
$$

$$
\begin{array} { r } { D _ { \alpha } ( Q \| P ) = \left\{ \begin{array} { l l } { \widetilde { A } ( \alpha , \omega _ { n } ) + \frac { ( \mathbf { x } _ { n } ^ { \top } \mu \mathbf { x } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \widetilde { B } ( \alpha , \omega _ { n } ) , } & { i f 1 < \alpha < \omega _ { n } , } \\ { \infty , } & { i f \alpha \geq \omega _ { n } . } \end{array} \right. } \end{array}\tag{26}
$$

Since $\omega _ { n } \geq \omega _ { \operatorname* { m i n } } \triangleq 1 + \sigma ^ { 2 } / ( \tau ^ { 2 } R ^ { 2 } )$ for every admissible pair, iffor all $i , | \mathbf { x } _ { i } ^ { \top } \pmb { \mu } _ { \mathbf { Y } } - y _ { i } | \leq k ^ { * } c$ for some $k ^ { * } > 0 ,$ , then for $1 < \alpha < \omega _ { \mathrm { { m i n } } }$

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \le \varepsilon _ { \mathrm { L R , u n b o u n d e d } } ( \alpha ) = \varepsilon _ { \mathrm { G G , u n b o u n d e d } } ( \alpha , \omega _ { \mathrm { m i n } } ) .\tag{27}
$$

Proposition 12 extends the results in Proposition 11 to high-dimensional settings. In the bounded case, the privacy loss is determined by the difference between the replacement records and the posterior precision, and scales linearly with α. In the unbounded case, the data geometry determines the inherent DP guarantees of OPS through $\omega _ { n } = \sigma ^ { 2 } / ( \mathbf { x } _ { n } ^ { \top } \Sigma _ { \mathbf { Y } } \mathbf { x } _ { n } )$ . Specifically, $A ( \alpha , \omega _ { n } )$ is governed by the influence of the removed record on the estimate; high-influential points have low $\omega _ { n }$ and may substantially reduce the posterior precision, creating a detectable signal even when the posterior mean is unchanged. As in Proposition 11, the reverse direction is finite only for $\alpha \ < \ \omega _ { n } ;$ thus high-influence points, with small $\omega _ { n }$ , both enlarge the privacy loss and shrink the range of Rényi orders admitting a finite guarantee.

## 6 Discussion

This work advances our understanding of the inherent privacy guarantees of Bayesian posterior sampling in a unified Rényi-divergence framework. Our results demonstrate that the intrinsic stochasticity of OPS provides meaningful privacy protection – shaped by the interplay among prior information, data size, model structure, and neighboring relationships – even without explicit noise injection. By offering explicit and interpretable privacy characterizations across diverse Bayesian models, our analysis clarifies when and how OPS can serve as a reliable DP mechanism. Our results offer a principled basis for reducing external noise injection and thereby improving the utility of released statistical outputs while maintaining formal privacy protection. The work also offers insights on designing more transparent and effective private data analysis methods in sensitive applications.

There are several interesting directions for future work. For example, the sub-Gaussian regime examined in Section 5 can be further relaxed to sub-exponential log-likelihood ratios, thereby extending the framework to settings where both uniform boundedness and sub-Gaussianity fail.

Corollary 13 (RDP guarantees of OPS with sub-exponential log-likelihood ratio). Suppose that there exist $\nu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } \in ( 0 , \infty )$ and $b { \bf { Y } } , { \bf { Y } } ^ { \prime } \in ( 0 , \infty )$ such that $f o r \forall t \in \mathbb { R }$ satisfying $| t | < b _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { - 1 }$ , it holds that log $\begin{array} { r } { \mathbb { E } _ { \pmb { \theta } \sim P } \left( e ^ { t \left( c _ { 1 } ( \pmb { \theta } ) - \pmb { \mu } _ { \mathbf { Y } } , \mathbf { \mathbf { y } } ^ { \prime } \right) } \right) \leq \frac { t ^ { 2 } \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 ( 1 - b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } | t | ) } } \end{array}$ where $\mu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } = \mathbb { E } _ { \pmb { \theta } \sim P } \left[ c _ { 1 } ( \pmb { \theta } ) \right]$ . Define

$$
\varepsilon _ { \mathrm { s u b E x p } } ( \alpha , b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } , \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } ) \triangleq \frac { \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 } \cdot \frac { \alpha ( 1 - 2 b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ) + 2 b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } } { ( 1 - b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ) \left( 1 - b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ( \alpha - 1 ) \right) } .\tag{28}
$$

When $b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } < \operatorname* { m i n } \{ ( \alpha - 1 ) ^ { - 1 } , 1 \}$ for all ordered neighboring pairs,

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P | | Q ) , D _ { \alpha } ( Q | | P ) \} \leq \operatorname* { s u p } _ { ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) : d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1 } \varepsilon _ { \mathrm { s u b E x p } } ( \alpha , b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } , \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } ) ,\tag{29}
$$

and releasing OPS satisfies $( \alpha , \mathrm { s u p } _ { ( \mathbf { Y } , \mathbf { Y ^ { \prime } } ) } \varepsilon _ { \mathrm { s u b E x p } } ( \alpha , b _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } , \nu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } ) ) { \cdot } R D P .$

The proof is in Appendix A.6. While $\varepsilon _ { \mathrm { s u b E x p } } ( \alpha , b _ { { \bf Y } , { \bf Y ^ { \prime } } } , \nu _ { { \bf Y } , { \bf Y ^ { \prime } } } ^ { 2 } )$ is tight for fixed sub-exponential parameters $\left( \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } , b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } \right)$ , deriving these parameters for a specific model often introduces looseness. For example, in the unbounded case of the Gaussian-Gaussian pair (Proposition 11), the log-likelihood ratio satisfies the sub-exponential condition as shown in Appendix A.9, but the derived parameters rely on upper bounds and are thus sub-optimal. In contrast, directly exploiting the exact MGFs of a non-central $\chi ^ { 2 }$ distribution and calculating $D _ { \alpha } ( P \| Q )$ via Theorem 3 yields sharper bounds. This illustrates the potential gap between general tail-based bounds and model-specific exact characterizations.

Despite this limitation, Corollary 13 remains a broadly applicable theoretical tool for analyzing inherent DP guarantees of OPS when both uniform boundedness and sub-Gaussianity fail. Remark 14 demonstrates this for linear regression under bounded-neighboring with covariate replacement, a case excluded from Proposition 12 because $c _ { 1 } ( \beta )$ contains a non-sub-Gaussian quadratic term.

Remark 14 (Linear regression with Gaussian prior, bounded neighboring with $\mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime } ) .$ . When $\mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime } , c _ { 1 } ( \beta )$ contains both affine and quadratic terms in $\beta$ and is generally not sub-Gaussian under $\pi ( \beta | \mathbf { Y } )$ . Suppose $\| \mathbf { x } _ { i } \| \leq R < \bar { \infty }$ and $| y _ { i } | \le M _ { y } < \infty$ for ∀ i, and assume the scale restriction $\sigma ^ { 2 } > 4 \tau ^ { 2 } R ^ { 2 }$ , then $c _ { 1 } ( \beta )$ is sub-exponential with parameters

$$
b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } = \frac { 4 \tau ^ { 2 } R ^ { 2 } } { \sigma ^ { 2 } } , \quad \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } \le \frac { 8 M _ { y } ^ { 2 } R ^ { 2 } \tau ^ { 2 } } { \sigma ^ { 4 } } + \frac { 1 6 \tau ^ { 4 } R ^ { 4 } } { \sigma ^ { 4 } } + \frac { 8 n ^ { 2 } \tau ^ { 6 } M _ { y } ^ { 2 } R ^ { 6 } } { \sigma ^ { 8 } } ,\tag{30}
$$

which can be plugged into Corollary 13for explicit RDP bound.

The proof is in Appendix A.10.1. The parameters in Eq. (30) also illustrate the conservatism that can arise when deriving model-specific sub-exponential parameters for Corollary 13: the Cauchy-Schwarz decomposition, operator norm bounds, and row-wise boundedness assumptions all introduce looseness, leading to a restricted valid range of α and an $n ^ { 2 } .$ -dependent variance proxy. Developing tighter techniques for estimating $\left( \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } , \bar { b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } } \right)$ , especially for complex models where exact MGFs are analytically intractable, is a promising direction for future work. Other future directions include leveraging OPS as a principled DP mechanism in applications such as privacy-preserving synthetic data generation, Markov chain Monte Carlo algorithms, and variational inference, with the goal of optimizing the trade-off between privacy guarantees and statistical utility.

## References

Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Annual international conference on the theory and applications ofcryptographic techniques, pages 486–503. Springer, 2006a.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pages 265–284. Springer, 2006b.

Ilya Mironov. Rényi differential privacy. In 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pages 263–275. IEEE, 2017.

Mark Bun and Thomas Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, pages 635–658, 2016.

Jinshuo Dong, Aaron Roth, and Weijie J Su. Gaussian differential privacy. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1):3–37, 2022.

Kamalika Chaudhuri and Nina Mishra. When random sampling preserves privacy. In Annual International Cryptology Conference, pages 198–213. Springer, 2006.

Ninghui Li, Wahbeh Qardaji, and Dong Su. On sampling, anonymization, and differential privacy or, k-anonymization meets differential privacy. In Proceedings of the 7th ACM Symposium on Information, Computer and Communications Security, pages 32–33, 2012.

Latanya Sweeney. k-anonymity: A model for protecting privacy. International Journal ofUncertainty, Fuzziness and Knowledge-Based Systems, 10(05):557–570, 2002.

Jairo Giraldo, Alvaro Cardenas, and Murat Kantarcioglu. Security and privacy trade-offs in cps by leveraging inherent differential privacy. In 2017 IEEE Conference on Control Technology and Applications (CCTA), pages 1313–1318. IEEE, 2017.

Zinan Lin, Vyas Sekar, and Giulia Fanti. On the privacy properties of gan-generated samples. In International Conference on Artificial Intelligence and Statistics, pages 1522–1530. PMLR, 2021.

Borja Balle, Gilles Barthe, and Marco Gaboardi. Privacy amplification by subsampling: Tight analyses via couplings and divergences. Advances in neural information processing systems, 31, 2018.

Thomas Steinke. Composition of differential privacy & privacy amplification by subsampling. arXiv preprint arXiv:2210.00597, 2022.

Xingyuan Zhao, Ruyu Zhou, and Fang Liu. Enhancing trade-offs in privacy, utility, and computational efficiency through multistage sampling technique (must). IEEE Transactions on Privacy, 2024.

Darakhshan J Mir. Differential privacy: an exploration of the privacy-utility landscape. Rutgers The State University of New Jersey, School of Graduate Studies, 2013.

Christos Dimitrakakis, Blaine Nelson, Aikaterini Mitrokotsa, and Benjamin IP Rubinstein. Robust and private bayesian inference. In International Conference on Algorithmic Learning Theory, pages 291–305. Springer, 2014.

Christos Dimitrakakis, Blaine Nelson, Zuhe Zhang, Aikaterini Mitrokotsa, and Benjamin IP Rubinstein. Differential privacy for bayesian inference through posterior sampling. Journal ofmachine learning research, 18(11):1–39, 2017.

Yu-Xiang Wang, Stephen Fienberg, and Alex Smola. Privacy for free: Posterior sampling and stochastic gradient monte carlo. In International Conference on Machine Learning, pages 2493– 2502, 2015.

Zuhe Zhang, Benjamin IP Rubinstein, and Christos Dimitrakakis. On the differential privacy of bayesian inference. In Thirtieth AAAI Conference on Artificial Intelligence, 2016.

James Foulds, Joseph Geumlek, Max Welling, and Kamalika Chaudhuri. On the theory and practice of privacy-preserving bayesian data analysis. pages 227–236, 2016.

Joseph Geumlek, Shuang Song, and Kamalika Chaudhuri. Renyi differential privacy mechanisms for posterior sampling. In Advances in Neural Information Processing Systems, pages 5289–5298, 2017.

Fangyuan Zhao, Xuebin Ren, Shusen Yang, Qing Han, Peng Zhao, and Xinyu Yang. Latent dirichlet allocation model training with differential privacy. IEEE Transactions on Information Forensics and Security, 16:1290–1305, 2020.

David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. Journal ofmachine Learning research, 3(Jan):993–1022, 2003.

Bai Li, Changyou Chen, Hao Liu, and Lawrence Carin. On connecting stochastic gradient mcmc and differential privacy. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 557–566. PMLR, 2019.

Jack E Jewson, Sahra Ghalebikesabi, and Chris C Holmes. Differentially private statistical inference through \β-divergence one posterior sampling. Advances in Neural Information Processing Systems, 36:76974–77001, 2023.

Kentaro Minami, Hiromi Arai, Issei Sato, and Hiroshi Nakagawa. Differential privacy without sensitivity. In NIPS, pages 956–964, 2016.

Yu-Xiang Wang. Per-instance differential privacy. Journal of Privacy and Confidentiality, 9(1), 2019.

Shenggang Hu, Louis Aslett, Hongsheng Dai, Murray Pollock, and Gareth O Roberts. Privacy guarantees in posterior sampling under contamination. arXiv preprint arXiv:2403.07772, 2024.

Daniel Kifer and Ashwin Machanavajjhala. No free lunch in data privacy. In Proceedings ofthe 2011 ACM SIGMOD International Conference on Management ofData, SIGMOD ’11, page 193–204, New York, NY, USA, 2011. Association for Computing Machinery. ISBN 9781450306614. doi: 10.1145/1989323.1989345. URL https://doi.org/10.1145/1989323.1989345.

David Sommer, Sebastian Meiser, and Esfandiar Mohammadi. Privacy loss classes: The central limit theorem in differential privacy. Cryptology ePrint Archive, 2018.

Alfréd Rényi. On measures of entropy and information. In Proceedings of the fourth Berkeley symposium on mathematical statistics and probability, volume 1: contributions to the theory of statistics, volume 4, pages 547–562. University of California Press, 1961.

Sivakanth Gopi, Yin Tat Lee, and Daogao Liu. Private convex optimization via exponential mechanism. In Conference on Learning Theory, pages 1948–1989. PMLR, 2022.

Moshe Shaked and J George Shanthikumar. Stochastic orders. Springer, 2007.

Michel Ledoux. The concentration ofmeasure phenomenon. Number 89. American Mathematical Soc., 2001.

## Appendix Contents

A.1 Proof of Lemma 2 13   
A.2 Proof of Theorem 3 14   
A.3 Proof of Corollary 5 14   
A.4 Proof of Corollary 6 16   
A.5 Proof of Corollary 8 17   
A.6 Proof of Corollary 13 18   
A.7 Proof of Example 7 for Categorical data 18   
A.7.1 bounded neighboring 19   
A.7.2 unbounded neighboring 19   
A.7.3 Proof of Bernoulli data 20   
A.8 Proof of Proposition 9 for Generalized Linear Models 23   
A.8.1 bounded neighboring 23   
A.8.2 unbounded neighboring 24   
A.8.3 Proof of Example 10 for Logistic Regression . 24   
A.9 Proof of Proposition 11 for Gaussian-Gaussian conjugate pair 26   
A.9.1 bounded neighboring 26   
A.9.2 unbounded neighboring 27   
A.10 Proof of Proposition 12 for Linear Regression 32   
A.10.1 bounded neighboring 32   
A.10.2 unbounded neighboring 36

## A.1 Proof of Lemma 2

Proof. The posterior of parameter θ given data Y is

$$
\pi ( \pmb { \theta } | \mathbf { Y } ) = \frac { \pi ( \pmb { \theta } ) \prod _ { i = 1 } ^ { n } p ( \mathbf { y } _ { i } | \pmb { \theta } ) } { \int \prod _ { i = 1 } ^ { n } p ( \mathbf { y } _ { i } | \pmb { \theta } ) \pi ( \pmb { \theta } ) d \pmb { \theta } } \overset { \triangle } { = } \frac { \pi ( \pmb { \theta } ) \ell ( \pmb { \theta } ; \mathbf { Y } ) } { \int \ell ( \pmb { \theta } ; \mathbf { Y } ) \pi ( \pmb { \theta } ) d \pmb { \theta } } = \frac { \pi ( \pmb { \theta } ) \ell ( \pmb { \theta } ; \mathbf { Y } ) } { p ( \mathbf { Y } ) } .\tag{A.1}
$$

For any pair of neighboring datasets Y and $\mathbf { Y } ^ { \prime }$ where $d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1$ , the privacy loss for releasing posterior sample $c ( \pmb \theta )$ per Eq (2) can be decomposed as:

$$
c ( \pmb \theta ) = \log \left( \frac { \pi ( \pmb \theta | \mathbf Y ) } { \pi ( \pmb \theta | \mathbf Y ^ { \prime } ) } \right) = \underbrace { \log \left( \frac { \ell ( \pmb \theta ; \mathbf Y ) } { \ell ( \pmb \theta ; \mathbf Y ^ { \prime } ) } \right) } _ { \triangleq \mathrm { \Lambda } _ { c _ { 1 } } ( \pmb \theta ) } + \underbrace { \log \left( \frac { p ( \mathbf Y ^ { \prime } ) } { p ( \mathbf Y ) } \right) } _ { \triangleq \mathrm { \Lambda } _ { c _ { 2 } } } .\tag{A.2}
$$

Note that $p ( \mathbf { Y ^ { \prime } } )$

$$
\begin{array} { r l } & { = \displaystyle \int \ell ( \theta ; \mathbf { Y } ^ { \prime } ) \pi ( \theta ) d \theta = \int \frac { \ell ( \theta ; \mathbf { Y } ^ { \prime } ) } { \ell ( \theta ; \mathbf { Y } ) } \ell ( \theta ; \mathbf { Y } ) \pi ( \theta ) d \theta = \int \frac { \ell ( \theta ; \mathbf { Y } ^ { \prime } ) } { \ell ( \theta ; \mathbf { Y } ) } \pi ( \theta | \mathbf { Y } ) p ( \mathbf { Y } ) d \theta \quad \mathrm { p e r ~ } \mathbf { E q } . } \\ & { = p ( \mathbf { Y } ) \int \frac { \ell ( \theta ; \mathbf { Y } ^ { \prime } ) } { \ell ( \theta ; \mathbf { Y } ) } \pi ( \theta | \mathbf { Y } ) d \theta } \end{array}\tag{A.1}
$$

$$
\Rightarrow \ \frac { p ( { \bf Y } ^ { \prime } ) } { p ( { \bf Y } ) } = \int \frac { \ell ( \theta ; { \bf Y } ^ { \prime } ) } { \ell ( \theta ; { \bf Y } ) } \pi ( \theta | { \bf Y } ) d \theta = \int e ^ { - c _ { 1 } ( \theta ) } \pi ( \theta | { \bf Y } ) d \theta = \mathbb { E } _ { \theta \sim \pi ( \cdot | { \bf Y } ) } \left[ e ^ { - c _ { 1 } ( \theta ) } \right] .\tag{A.3}
$$

Since $c _ { 2 } = \log ( p ( \mathbf { Y } ^ { \prime } ) / p ( \mathbf { Y } ) )$ , we have

$$
c _ { 2 } = \log \left( \frac { p ( \mathbf { Y } ^ { \prime } ) } { p ( \mathbf { Y } ) } \right) = \log \left( \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { - c _ { 1 } ( \pmb { \theta } ) } \right] \right) .\tag{A.4}
$$

## A.2 Proof of Theorem 3

Proof. Per the definition of Rényi divergence in Eq (4),

$$
D _ { \alpha } ( P | | Q ) = { \frac { 1 } { \alpha - 1 } } \log \mathbb { E } _ { x \sim Q } \left( { \frac { P ( x ) } { Q ( x ) } } \right) ^ { \alpha } = { \frac { 1 } { \alpha - 1 } } \log \mathbb { E } _ { x \sim P } \left( { \frac { P ( x ) } { Q ( x ) } } \right) ^ { \alpha - 1 } .\tag{A.5}
$$

For any neighboring datasets Y and $\mathbf { Y } ^ { \prime } .$ , we can calculate the Rényi divergence between the corresponding neighboring posterior distributions $\pi ( \pmb \theta | \mathbf Y )$ and $\pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } )$ as below

$$
\begin{array} { r l } { D _ { \boldsymbol { \alpha } } \big ( \pi ( \pmb { \theta } | \mathbf { Y } ) \big | \big | \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } ) \big ) = \displaystyle \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ \left( \frac { \pi ( \pmb { \theta } | \mathbf { Y } ) } { \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } ) } \right) ^ { \alpha - 1 } \right] } & { } \\ { \displaystyle } & { = \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) c ( \pmb { \theta } ) } \right] } \\ { \displaystyle } & { = \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) ( c _ { 1 } ( \pmb { \theta } ) + c _ { 2 } ) } \right] ~ \mathrm { p e r ~ E q ~ ( A . 2 ) } } \\ { \displaystyle } & { = c _ { 2 } + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \pmb { \theta } ) } \right] } \end{array}\tag{A.6}
$$

Plug Eq. (A.4) into Eq. (A.6),

$$
D _ { \alpha } ( \pi ( \theta | \mathbf { Y } ) | | \pi ( \theta | \mathbf { Y } ^ { \prime } ) ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { - c _ { 1 } ( \theta ) } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \theta ) } \right]\tag{A.7}
$$

$$
= \log \left[ \left( \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { - c _ { 1 } ( \pmb { \theta } ) } \right] \right) \left( \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) \cdot c _ { 1 } ( \pmb { \theta } ) } \right] \right) ^ { \frac { 1 } { \alpha - 1 } } \right] .\tag{A.8}
$$

## A.3 Proof of Corollary 5

Proof. Assume $c _ { 1 } ( \pmb \theta )$ is uniformly bounded, i.e. there exist finite constants $a , b \in \mathbb { R }$ with $a < b$ such that $\forall \pmb \theta \in \Theta , c _ { 1 } ( \pmb \theta ) \in [ a , b ]$ . Thus $- c _ { 1 } ( \pmb \theta ) \in [ - b , - a ]$ and per monotonicity.

$$
c _ { 2 } = \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { - c _ { 1 } ( \pmb { \theta } ) } \right] \in [ - b , - a ]\tag{A.9}
$$

$$
\frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \pmb { \theta } ) } \right] \in [ a , b ] .\tag{A.10}
$$

Taken together with the decomposition in Eq. (A.6), we obtain

$$
D _ { \alpha } \big ( \pi ( \pmb { \theta } | \mathbf { Y } ) \vert \vert \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } ) \big ) = c _ { 2 } + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot \vert \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) \cdot c _ { 1 } ( \pmb { \theta } ) } \right] \leq b - a .\tag{A.11}
$$

Note that to attain the equality in Eq. (A.11) of the upper bound $b - a$ would require simultaneously $c _ { 1 } ( \pmb \theta ) = a$ (so that $c _ { 2 } = - a )$ and $c _ { 1 } ( \pmb \theta ) = b$ (so that $\begin{array} { r } { \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \pmb { \theta } ) } \right] = b ) } \end{array}$ almost surely, which is impossible unless $a = b .$ Therefore, this bound is not tight and unattainable.

To obtain an attainable upper bound of $D _ { \alpha } ( \pi ( \pmb { \theta } | \mathbf { Y } ) | | \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } ) )$ , it suffices to maximize the two exponential moments of $c _ { 1 } ( \pmb \theta )$ involved in Eq. (A.11), subject to the constraint $c _ { 1 } ( \pmb { \theta } ) \in [ a , b ]$ . This maximization can be achieved using convex-order arguments, as in [Shaked and Shanthikumar, 2007]. WLOG, let X be any random variable uniformly bounded with $[ a , b ]$ , and $X ^ { * }$ denote a binary random variable supported on $\{ a , b \}$ as below, with the same mean as $\bar { X } \colon$

$$
X ^ { * } = { \left\{ \begin{array} { l l } { a , } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } } 1 - p } \\ { b , } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } } p } \end{array} \right. } ;\tag{A.12}
$$

that is, $\operatorname { \mathbb { E } } ( X ) = \operatorname { \mathbb { E } } ( X ^ { * } ) = a ( 1 - p ) + b p$ for some $p \in [ 0 , 1 ]$ . For any convex function $\phi$ and any $x \in [ a , b ]$ , convexity implies the chord bound:

$$
\phi ( x ) ~ \leq ~ \lambda ( x ) \phi ( b ) + ( 1 - \lambda ( x ) ) \phi ( a ) ~ \forall x \in [ a , b ]\tag{A.13}
$$

Let $\textstyle \lambda ( x ) = { \frac { x - a } { b - a } }$ and take expectation of $\phi ( x )$ over its distribution, we have:

$$
\begin{array} { r } { \mathbb { E } [ \phi ( X ) ] \leq p \phi ( b ) + ( 1 - p ) \phi ( a ) = \mathbb { E } [ \phi ( X ^ { \ast } ) ] . } \end{array}\tag{A.14}
$$

Lemma A.1 (Theorem 3.A.7 in Shaked and Shanthikumar [2007]). Let X and Y be two random variables with the samefinite means. Then $X \leq _ { \mathrm { c x } } Y$ if and only if,

$$
\begin{array} { r } { \mathbb { E } ( \phi ( X ) ) \leq \mathbb { E } ( \phi ( Y ) ) \qquad f o r a n y c o n \nu e x f u n c t i o n \phi . } \end{array}\tag{A.15}
$$

Hence, per the Lemma above, we have $X \le _ { \mathrm { c x } } X ^ { * }$ . Since X can be any random variable uniformly bounded with $[ a , b ] , X ^ { * }$ is maximal in convex order among all such bounded random variables.

Also, since the exponential function $e ^ { x }$ is convex, and the Rényi divergence functional $D _ { \alpha }$ in $\operatorname { E q . } \left( 8 \right)$ is fully determined by the exponential moments of $c _ { 1 } ( \pmb \theta )$ , it follows that $D _ { \alpha }$ is monotone with respect to the convex order. Setting $X = c _ { 1 } ( \pmb \theta )$ , we therefore obtain $c _ { 1 } ( \theta ) \leq _ { \mathrm { c x } } X ^ { * }$ , which implies that the $D _ { \alpha } ( \pi ( \pmb { \theta } | \mathbf { Y } ) | | \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } ) )$ ) is maximized when $c _ { 1 } ( \pmb \theta )$ coincides with the endpoint-supported binary random variable ${ \dot { X } } ^ { * }$ as defined in Eq. (A.12). Then per Eq. (A.7)

$$
\begin{array} { r l } & { \quad D _ { \alpha } \big ( \pi ( \theta | \mathbf { Y } ) \big ) | \pi ( \theta | \mathbf { Y } ^ { \prime } ) \rangle } \\ & { = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { - c _ { 1 } ( \theta ) } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | \mathbf { Y } ) } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \theta ) } \right] } \\ & { \le \log \mathbb { E } \left[ e ^ { - X ^ { * } } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } \left[ e ^ { ( \alpha - 1 ) X ^ { * } } \right] \quad \mathrm { p e r ~ c o n v e x ~ o r d e r ~ d o m i n a n c e } } \\ & { = \log \big ( ( 1 - p ) e ^ { - \alpha } + p e ^ { - b } \big ) + \frac { 1 } { \alpha - 1 } \log \Big ( ( 1 - p ) e ^ { ( \alpha - 1 ) a } + p e ^ { ( \alpha - 1 ) b } \Big ) } \\ & { = \log \Big ( 1 - p + p e ^ { - ( b - a ) } \Big ) + \frac { 1 } { \alpha - 1 } \log \Big ( 1 - p + p e ^ { ( \alpha - 1 ) ( b - a ) } \Big ) } \\ & { \triangleq D _ { \alpha } ( p ) } \end{array}\tag{A.16}
$$

(A.17)

For a fixed pair of neighboring datasets $\mathbf { Y }$ and $\mathbf { Y } ^ { \prime }$ , in order to align the means of $c _ { 1 } ( \pmb \theta )$ and $X ^ { * }$ , one should require,

$$
\begin{array} { r l } & { \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \big [ c _ { 1 } ( \pmb { \theta } ) \big ] = a ( 1 - p ) + b p } \\ & { \Rightarrow \hat { p } = \frac { \mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } \big [ c _ { 1 } ( \pmb { \theta } ) \big ] - a } { b - a } , } \end{array}\tag{A.18}
$$

However, since $\mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | \mathbf { Y } ) } [ c _ { 1 } ( \pmb { \theta } ) ]$ may take any value in $[ a , b ]$ across neighboring dataset pairs, the associated $\hat { p }$ ranges over [0, 1]. Thus, an upper bound on $D _ { \alpha } ( \pi ( \pmb { \theta } | \mathbf { Y } ) | | \pi ( \pmb { \theta } | \mathbf { Y } ^ { \prime } ) )$ that is valid over all neighboring datasets need to be obtained by maximizing $D _ { \alpha } ( p )$ in Eq. (A.16) over $p \in [ 0 , 1 ]$

Let $\Delta = b - a$ and define $A = e ^ { - \Delta } \in ( 0 , 1 ) , B = e ^ { ( \alpha - 1 ) \Delta } > 1$ . To maximize $D _ { \alpha } ( p )$ over $p \in [ 0 , 1 ]$ take the first derivative on both sides with respect to $p \mathrm { : }$

$$
D _ { \alpha } ^ { \prime } ( p ) = \frac { d D _ { \alpha } ( p ) } { d p } = \frac { A - 1 } { 1 - p + p A } + \frac { 1 } { \alpha - 1 } \cdot \frac { B - 1 } { 1 - p + p B } .\tag{A.19}
$$

Setting $D _ { \alpha } ^ { \prime } ( p ) = 0$ , we have

$$
\begin{array} { r l } {  { p ^ { * } = \arg \operatorname* { m a x } _ { p \in [ 0 , 1 ] } D _ { \alpha } ( p ) } } \\ & { = \frac { ( B - 1 ) - ( \alpha - 1 ) ( 1 - A ) } { \alpha ( 1 - A ) ( B - 1 ) } } \\ & { = \frac { \big ( e ^ { ( \alpha - 1 ) \Delta } - 1 \big ) - ( \alpha - 1 ) \big ( 1 - e ^ { - \Delta } \big ) } { \alpha \big ( 1 - e ^ { - \Delta } \big ) \big ( e ^ { ( \alpha - 1 ) \Delta } - 1 \big ) } . } \end{array}\tag{A.20}
$$

Therefore,

$$
\begin{array} { l } { { \displaystyle { D _ { \alpha } \leq \log \left( 1 - p ^ { * } + p ^ { * } e ^ { - \Delta } \right) + \frac { 1 } { \alpha - 1 } \log \left( 1 - p ^ { * } + p ^ { * } e ^ { ( \alpha - 1 ) \Delta } \right) } } } \\ { { \displaystyle ~ = \log \left( 1 - \frac { ( B - 1 ) - ( \alpha - 1 ) ( 1 - A ) } { \alpha ( 1 - A ) ( B - 1 ) } + \frac { ( B - 1 ) - ( \alpha - 1 ) ( 1 - A ) } { \alpha ( 1 - A ) ( B - 1 ) } A \right) } } \\ { { \displaystyle ~ + \frac { 1 } { \alpha - 1 } \log \left( 1 - \frac { ( B - 1 ) - ( \alpha - 1 ) ( 1 - A ) } { \alpha ( 1 - A ) ( B - 1 ) } + \frac { ( B - 1 ) - ( \alpha - 1 ) ( 1 - A ) } { \alpha ( 1 - A ) ( B - 1 ) } B \right) } } \end{array}
$$

$$
\begin{array} { r l } & { = \log \left( \frac { \alpha ( \Delta \ J - A ) ( B - 1 ) - ( \Delta \ J - A ) ( B - 1 ) + ( \alpha - 1 ) ( \Delta \ J - A ) - ( \alpha - 1 ) ( \lambda \ J - A ) } { \alpha ( \Delta \ J - A ) ( B - 1 ) } \right) } \\ & { \quad + \frac { 1 } { \alpha - 1 } \log \left( \frac { \alpha ( 1 - A ) ( B - 1 ) - ( B - 1 ) + B ( B - 1 ) - ( \alpha - 1 ) ( 1 - A ) ( B - 1 ) } { \alpha ( 1 - A ) ( B - 1 ) } \right) } \\ & { = \log \left( \frac { ( \alpha - 1 ) ( B - A ) } { \alpha ( B - 1 ) } \right) + \frac { 1 } { \alpha - 1 } \log \left( \frac { B - A } { \alpha ( 1 - A ) } \right) \triangleq \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) . } \end{array}\tag{21}
$$

Reverse direction. If $c _ { 1 } ( \pmb { \theta } ) \in [ a , b ]$ for all $\theta \in \Theta$ , then $\tilde { c } _ { 1 } ( \pmb \theta ) = - c _ { 1 } ( \pmb \theta ) \in [ - b , - a ]$ for all $\theta \in \Theta$ an interval of the same width $\Delta = b - a$ . The argument above uses only the support width of the log-likelihood ratio and holds under an arbitrary posterior. Applying it to $\tilde { c } _ { 1 } ( \pmb \theta )$ under $Q = \pi ( \theta | \mathbf { Y } ^ { \prime } )$ via Eq. (9) yields $D _ { \alpha } ( Q \| P ) \leq \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ . Hence max{ $\begin{array} { r } { D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \le \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) } \end{array}$ , and the $( \alpha , \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) )$ -RDP guarantee holds in both directions.

Claim A.2. For $\forall \alpha > 1$ , and ${ ' f o r } \forall \Delta _ { 2 } \geq \Delta _ { 1 } \geq 0 ,$ , we have $0 \le \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta _ { 2 } ) - \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta _ { 1 } ) \le \Delta _ { 2 } - \Delta _ { 1 }$

Proof. Take the first derivative of Eq. (A.16) with respect to $\Delta$

$$
\partial _ { \Delta } D _ { \alpha } ( p , \Delta ) = - \frac { p e ^ { - \Delta } } { 1 - p + p e ^ { - \Delta } } + \frac { p e ^ { ( \alpha - 1 ) \Delta } } { 1 - p + p e ^ { ( \alpha - 1 ) \Delta } } = h \left( e ^ { ( \alpha - 1 ) \Delta } \right) - h \left( e ^ { - \Delta } \right) ,\tag{A.22}
$$

where $\begin{array} { r } { h ( u ) = \frac { p u } { 1 - p + p u } } \end{array}$ for $u > 0$ . Since

$$
\frac { \partial h ( u ) } { \partial u } = \frac { p ( 1 - p + p u ) - p ^ { 2 } u } { ( 1 - p + p u ) ^ { 2 } } = \frac { p ( 1 - p ) } { ( 1 - p + p u ) ^ { 2 } } \geq 0 ,\tag{A.23}
$$

$h ( u )$ is non-decreasing in u. Per $\alpha > 1$ and $\Delta \ge 0 , e ^ { ( \alpha - 1 ) \Delta } \ge e ^ { - \Delta }$ , then $\partial _ { \Delta } D _ { \alpha } ( p , \Delta ) \geq 0$ and the equality holds only in the degenerate cases $p = 0 , p = 1$ , or $\Delta = 0$

Since the pointwise maximum preserves order, for any $\Delta _ { 2 } \geq \Delta _ { 1 }$ and $\forall p \in [ 0 , 1 ] , \forall \alpha > 1$

$$
0 \le D _ { \alpha } ( p , \Delta _ { 2 } ) - D _ { \alpha } ( p , \Delta _ { 1 } ) \le \Delta _ { 2 } - \Delta _ { 1 }
$$

Taking the maximum over p on both sides

$$
\begin{array} { r } { \Rightarrow 0 \leq \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta _ { 2 } ) - \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta _ { 1 } ) \leq \Delta _ { 2 } - \Delta _ { 1 } . } \end{array}\tag{A.24}
$$

We can conclude that $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ is 1-Lipschitz in $\Delta$ . In other words, $\varepsilon _ { \mathrm { u b } } ( \alpha , \Delta )$ grows at most linearly with slope at most 1.

## A.4 Proof of Corollary 6

Claim A.3. Forfixed $\Delta ,$ as $\alpha  \infty , \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) = \Delta - \alpha ^ { - 1 } \log ( \alpha ( 1 - e ^ { - \Delta } ) ) + \mathcal { O } ( \alpha ^ { - 1 } )$

Proof. $A = e ^ { - \Delta } \in ( 0 , 1 )$ . As $\alpha  \infty , B = e ^ { ( \alpha - 1 ) \Delta }  \infty$ . Thus

$$
{ \frac { B - A } { B - 1 } } = 1 + { \frac { 1 - A } { B - 1 } } = 1 + { \mathcal { O } } ( B ^ { - 1 } ) = 1 + { \mathcal { O } } ( e ^ { - ( \alpha - 1 ) \Delta } )\tag{A.25}
$$

Plug Eq. (A.25) into the first term in Eq. (A.21)

$$
\begin{array} { r l r } {  { \log ( \frac { ( \alpha - 1 ) ( B - A ) } { \alpha ( B - 1 ) } ) = \log ( \frac { \alpha - 1 } { \alpha } ) + \log ( \frac { B - A } { B - 1 } ) } } \\ & { } & { = \log ( 1 - \alpha ^ { - 1 } ) + \log ( 1 + \mathcal { O } ( e ^ { - ( \alpha - 1 ) \Delta } ) ) } \\ & { } & { = - \alpha ^ { - 1 } + \mathcal { O } ( \alpha ^ { - 2 } ) + \mathcal { O } ( e ^ { - ( \alpha - 1 ) \Delta } ) \quad \mathrm { a s ~ } \alpha \to \infty } \end{array}\tag{A.26}
$$

Since $B - A = e ^ { ( \alpha - 1 ) \Delta } - e ^ { - \Delta } = e ^ { - \Delta } \left( e ^ { \alpha \Delta } - 1 \right)$

$$
{ \frac { 1 } { \alpha - 1 } } \log \left( { \frac { B - A } { \alpha ( 1 - A ) } } \right) = { \frac { 1 } { \alpha - 1 } } { \Big [ } \log ( B - A ) - \log ( \alpha ( 1 - A ) ) { \Big ] }
$$

$$
\begin{array} { l } { \displaystyle = \frac { 1 } { \alpha - 1 } \Big [ - \Delta + \log \big ( e ^ { \alpha \Delta } - 1 \big ) - \log ( \alpha ( 1 - A ) ) \Big ] } \\ { \displaystyle = \frac { 1 } { \alpha - 1 } \Big [ - \Delta + \alpha \Delta + \log \big ( 1 - e ^ { - \alpha \Delta } \big ) - \log ( \alpha ( 1 - A ) ) \Big ] } \\ { \displaystyle = \Delta - \frac { 1 } { \alpha - 1 } \log ( \alpha ( 1 - A ) ) + \frac { \log \big ( 1 - e ^ { - \alpha \Delta } \big ) } { \alpha - 1 } } \end{array}
$$

Plug Eqs. (A.27) and (A.26) in Eq. (A.21), then

$$
\begin{array} { l } { \varepsilon _ { \mathrm { u b } } ( \alpha , \Delta ) = \displaystyle \frac { 1 } { \alpha - 1 } \log \left( \frac { B - A } { \alpha ( 1 - A ) } \right) + \log \left( \frac { ( \alpha - 1 ) ( B - A ) } { \alpha ( B - 1 ) } \right) } \\ { = \Delta - \displaystyle \frac { 1 } { \alpha - 1 } \log ( \alpha ( 1 - A ) ) + \frac { \log \left( 1 - e ^ { - \alpha c \Delta } \right) } { \alpha - 1 } - \alpha ^ { - 1 } + \mathcal { O } ( \alpha ^ { - 2 } ) + \underbrace { \mathcal { O } \left( e ^ { - ( \alpha - 1 ) \Delta } \right) } _ { \mathrm { d e m u c h e ~ i n e s ~ \alpha ~ ^ 2 ~ } } } \\ { \mathrm { s i n c e ~ } \displaystyle \frac { 1 } { \alpha - 1 } = \frac { 1 } { \alpha } + \mathcal { O } ( \alpha ^ { - 2 } ) \mathrm { a n d ~ l o g ~ ( 1 - ~ } e ^ { - \alpha \Delta } ) \approx e ^ { - \alpha \Delta } \quad \mathrm { f o r ~ u n t h i c i n t y ~ l u g e c o ~ } } \\ { = \Delta - \left( \frac { 1 } { \alpha } + \mathcal { O } ( \alpha ^ { - 2 } ) \right) \cdot \log ( \alpha ( 1 - A ) ) + \mathcal { O } \left( \frac { e ^ { - \alpha \Delta } } { \alpha } \right) - \alpha ^ { - 1 } + \mathcal { O } ( \alpha ^ { - 2 } ) } \\ { \mathrm { s i n c e ~ } \displaystyle \frac { e ^ { - \alpha \Delta } } { \alpha } \leq \frac { 1 } { \alpha ^ { 2 } } \leq \frac { \log \alpha } { \alpha ^ { 2 } } \leq \frac { 1 } { \alpha } \leq \frac { \log \alpha } { \alpha } \quad \mathrm { f o r ~ u n f i c i n t y ~ l u g e ~ } } \\ { = \Delta - \displaystyle \frac { 1 } { \alpha } \log ( \alpha ( 1 - e ^ { - \Delta } ) ) + \mathcal { O } \left( \alpha ^ { - 1 } \right) } \end{array}\tag{A.27}
$$

Similar to Appendix $\mathrm { A } . 3 , \tilde { c } _ { 1 } ( \pmb \theta ) \in [ - b , - a ]$ under Q, we also have $D _ { \alpha } ( Q \| P ) \leq b - a = \Delta$ for every $\alpha > 1$ . Hence max $\{ D _ { \infty } ( P \| Q ) , \dot { D } _ { \infty } ( Q \| \dot { P } ) \} \le \Delta$ , and the pure $\Delta { \bf - D P }$ conclusion of Corollary 5 is two-sided. □

## A.5 Proof of Corollary 8

Proof. Step 1: a fixed ordered pair. Fix an ordered neighboring pair $( { \bf Y } , { \bf Y ^ { \prime } } )$ , and let $P = \pi ( \cdot | \mathbf { Y } )$ $Q = \pi ( \cdot | \dot { \mathbf { Y } } ^ { \prime } )$ . Suppose $c _ { 1 } ( \pmb { \theta } ) \overset { \cdot } { = } \log \{ \ell ( \pmb { \theta } ; \mathbf { Y } ) / \ell ( \pmb { \theta } ; \mathbf { Y } ^ { \prime } ) \}$ is sub-Gaussian under P, i.e., there exists $\sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } \in ( 0 , \infty )$ such that for all $t \in \mathbb { R }$

$$
\log \mathbb { E } _ { \pmb { \theta } \sim P } \Big [ e ^ { t ( c _ { 1 } ( \pmb { \theta } ) - \mu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ) } \Big ] \leq \frac { t ^ { 2 } \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 } , \qquad \mu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } \triangleq \mathbb { E } _ { \pmb { \theta } \sim P } [ c _ { 1 } ( \pmb { \theta } ) ] .\tag{A.29}
$$

Setting $t = \alpha - 1$ and $t = - 1$ in Eq. (A.29), respectively, bounds the two exponential moments in Eq. (A.7):

$$
\begin{array} { l } { { \displaystyle \log \mathbb { E } _ { \pmb { \theta } \sim P } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \pmb { \theta } ) } \right] \leq ( \alpha - 1 ) \mu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } + \frac { ( \alpha - 1 ) ^ { 2 } \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 } } , \ } \\ { { \displaystyle c _ { 2 } = \log \mathbb { E } _ { \pmb { \theta } \sim P } \left[ e ^ { - c _ { 1 } ( \pmb { \theta } ) } \right] \leq - \mu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } + \frac { \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 } } . \ } \end{array}
$$

Plugging both into Eq. (A.6),

$$
\begin{array} { r l } & { D _ { \alpha } \big ( P \| Q \big ) = c _ { 2 } + \displaystyle \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim P } \big [ e ^ { ( \alpha - 1 ) c _ { 1 } ( \theta ) } \big ] } \\ & { \qquad \quad \le - \mu \mathbf { Y } , \mathbf { Y } ^ { \prime } + \displaystyle \frac { \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 } + \mu \mathbf { Y } , \mathbf { Y } ^ { \prime } + \displaystyle \frac { ( \alpha - 1 ) \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 } = \displaystyle \frac { \alpha } { 2 } \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } . } \end{array}\tag{A.30}
$$

Step 2: uniform, two-sided bound. Applying Step 1 to the reversed pair $( { \bf Y ^ { \prime } } , { \bf Y } )$ gives $D _ { \alpha } ( Q \| P ) \leq$ $\frac { \alpha } { 2 } \sigma _ { \mathbf { Y ^ { \prime } } , \mathbf { Y } } ^ { 2 }$ , where, by definition, $\sigma _ { \mathbf { Y ^ { \prime } } , \mathbf { Y } } ^ { 2 }$ is the variance proxy of $\tilde { c } _ { 1 } ( \pmb \theta ) = - c _ { 1 } ( \pmb \theta )$ under Q. Since $( { \bf Y ^ { \prime } } , { \bf Y } )$ is an ordered neighboring pair whenever $( { \bf Y } , { \bf Y ^ { \prime } } )$ is, both $\sigma _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 }$ and $\sigma _ { \mathbf { Y ^ { \prime } } , \mathbf { Y } } ^ { 2 }$ are bounded by

$$
{ \bar { \sigma } } ^ { 2 } \triangleq \operatorname* { s u p } _ { ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) : d ( \mathbf { Y } , \mathbf { Y } ^ { \prime } ) = 1 } \sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } ,\tag{A.31}
$$

where the supremum is over all ordered neighboring pairs in the admissible data domain. Hence, by Eq. (A.30),

$$
\operatorname* { s u p } _ { ( \mathbf { Y } , \mathbf { Y ^ { \prime } } ) : d ( \mathbf { Y } , \mathbf { Y ^ { \prime } } ) = 1 } \operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \frac { \alpha \bar { \sigma } ^ { 2 } } { 2 } \triangleq \varepsilon _ { \mathrm { s u b G a u s } } ( \alpha , \bar { \sigma } ^ { 2 } ) ,\tag{A.32}
$$

which completes the proof.

## A.6 Proof of Corollary 13

Proof. If assume $c _ { 1 } ( \pmb \theta )$ is sub-Exponential under $\pi ( \cdot | \mathbf { Y } )$ , i.e. there exist $\nu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } \in \mathbf { \Gamma } ( 0 , \infty )$ and $b { \bf { \bf { Y } } } , { \bf { Y } } ^ { \prime } \in ( 0 , \infty )$ such that for all $t \in \mathbb { R }$ satisfying $| t | < 1 / b _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } }$ , it holds that

$$
\log \mathbb { E } _ { \pmb { \theta } \sim P } \left( e ^ { t ( c _ { 1 } ( \pmb { \theta } ) - \mu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ) } \right) \leq \frac { t ^ { 2 } \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 ( 1 - b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } | t | ) } ;\tag{A.33}
$$

Then, by setting $t \ : \mathrm { a s } - 1$ and $\alpha - 1$ in $\mathrm { E q . } \left( \mathrm { A . } 3 3 \right)$ respectively, the required exponential moments of $c _ { 1 } ( \pmb \theta )$ under $\pi ( \cdot | \mathbf { Y } )$ in $\mathrm { E q . } \left( \mathrm { A . } 7 \right)$ can be derived as:

when $t = \alpha - 1$ , we have:

$$
\log \mathbb { E } _ { \theta \sim \pi ( \cdot | \mathbf { Y } ) } \left( e ^ { ( \alpha - 1 ) \cdot c _ { 1 } ( \theta ) } \right) \leq ( \alpha - 1 ) \mu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } + \frac { ( \alpha - 1 ) ^ { 2 } \nu _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } } { 2 ( 1 - b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ( \alpha - 1 ) ) } ,\tag{A.34}
$$

when $t = - 1$ , we have:

$$
c _ { 2 } = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | \mathbf { Y } ) } [ e ^ { - c _ { 1 } ( \theta ) } ] \leq - \mu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } + \frac { \nu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } } { 2 ( 1 - b _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ) } .\tag{A.35}
$$

Plugging the above two into Eq. (A.6), under $b _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } < 1$ and $b _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ( \alpha - 1 ) < 1$

$$
\begin{array} { r l } { D _ { \alpha } ( \pi ( \theta | { \bf Y } ) ) \| \pi ( \theta | { \bf Y } ^ { \prime } ) ) = c _ { 2 } + \displaystyle \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot , { \bf Y } ) } \left[ e ^ { ( \alpha - 1 ) \cdot c _ { 1 } ( \theta ) } \right] } & { } \\ { \le - \mu { \bf Y } , { \bf Y } ^ { \prime } + \displaystyle \frac { \nu _ { { \bf Y } , { \bf Y } ^ { \prime } } ^ { 2 } } { 2 \left( 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } \right) } + \displaystyle \frac { 1 } { \alpha - 1 } \Big [ \big ( \alpha - 1 ) \mu { \bf Y } , { \bf Y } ^ { \prime } + \displaystyle \frac { \big ( \alpha - 1 \big ) ^ { 2 } \nu _ { { \bf Y } , { \bf Y } ^ { \prime } } ^ { 2 } } { 2 \big ( 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } ( \alpha - 1 ) \big ) } \Big ] } & { } \\ { = \displaystyle \frac { \nu _ { { \bf Y } , { \bf Y } ^ { \prime } } ^ { 2 } } { 2 \big ( 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } \big ) } + \frac { \big ( \alpha - 1 \big ) \nu _ { { \bf Y } , { \bf Y } ^ { \prime } } ^ { 2 } } { 2 \big ( 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } ( \alpha - 1 ) \big ) } } & { } \\  = \displaystyle \frac { \nu _ { { \bf Y } , { \bf Y } ^ { \prime } } ^ { 2 } } { 2 } \cdot \frac { 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } ( \alpha - 1 ) + \big ( \alpha - 1 \big ) \big ( 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } \big ) }  \big ( 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } \big ) \big ( 1 - b _ { { \bf Y } , { \bf Y } ^ { \prime } } ( \alpha - 1 ) \big ) \end{array}\tag{A.36}
$$

Applying the above to the reversed pair $( { \bf Y ^ { \prime } } , { \bf Y } )$ , i.e., to $\tilde { c } _ { 1 } ( \pmb \theta ) = - c _ { 1 } ( \pmb \theta )$ under $\pi ( \cdot | \mathbf { Y } ^ { \prime } )$ , gives $\begin{array} { r } { \hat { D _ { \alpha } } \big ( \bar { \boldsymbol { \pi } } ( \pmb { \theta } | \mathbf { Y } ^ { \prime } ) \| \boldsymbol { \pi } ( \pmb { \theta } | \mathbf { Y } ) \big ) \leq \varepsilon _ { \mathrm { s u b E x p } } ( \alpha , \hat { b } _ { \mathbf { Y } ^ { \prime } , \mathbf { Y } } , \nu _ { \mathbf { Y } ^ { \prime } , \mathbf { Y } } ^ { 2 } ) } \end{array}$ , provided $b _ { \mathbf { Y ^ { \prime } } , \mathbf { Y } } < \operatorname* { m i n } \{ ( \alpha - 1 ) ^ { - 1 } , 1 \}$ . Since $( { \bf Y ^ { \prime } } , { \bf Y } )$ is an ordered neighboring pair whenever $( { \bf Y } , { \bf Y ^ { \prime } } )$ is, if $b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } < \operatorname* { m i n } \{ ( \alpha - 1 ) ^ { - 1 } , 1 \}$ for all ordered neighboring pairs, then

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \operatorname* { s u p } _ { ( \mathbf { Y } , \mathbf { Y ^ { \prime } } ) : d ( \mathbf { Y } , \mathbf { Y ^ { \prime } } ) = 1 } \varepsilon _ { \mathrm { s u b E x p } } ( \alpha , b _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } , \nu _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } )\tag{A.37}
$$

uniformly over all neighboring pairs, which completes the proof.

## A.7 Proof of Example 7 for Categorical data

Proof. Let $y _ { i } \in \{ 1 , \ldots , K \}$ denote a categorical outcome with K categories. Let $\pmb \theta = ( \theta _ { 1 } , \dots , \theta _ { K } )$ where $\textstyle \sum _ { j = 1 } ^ { K } \theta _ { j } \ = \ 1$ and $\theta _ { j } = \operatorname* { P r } ( y _ { i } = j )$ . Assume $y _ { i } | \pmb \theta \sim \mathrm { C a t e g o r i c a l } ( \pmb \theta )$ independently for $i = 1 , \ldots , n ,$ , and the likelihood of the dataset $\begin{array} { r } { Y = \{ y _ { 1 } , \dots , y _ { n } \} \mathrm { ~ i s ~ } \ell ( \pmb { \theta } ; \mathbf { Y } ) = \prod _ { j = 1 } ^ { K } \theta _ { j } ^ { s _ { j } ( Y ) } } \end{array}$ , where $\begin{array} { r } { s _ { j } ( Y ) = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ y _ { i } = j \} } \end{array}$ is the category counts.

## A.7.1 bounded neighboring

WLOG, two neighboring datasets Y and $Y ^ { \prime }$ are both of size n and differ in exactly one observation. Specifically, there exist distinct indices j ̸= k and $j , k \in \{ 1 , \dots , K \}$ such that $\begin{array} { r } { \dot { s _ { j } } ( Y ) = s _ { j } ( Y ^ { \prime } ) + } \end{array}$ $\bar { 1 , s _ { k } ( Y ) } \overset { \cdot } { = } s _ { k } ( Y ^ { \prime } ) - 1$ , and $s _ { \ell } ( Y ) = s _ { \ell } \bar { ( Y ^ { \prime } ) }$ for all $\ell \not \in \{ j , k \}$

$$
c _ { 1 } ( \pmb \theta ) = \log \frac { \ell ( \pmb \theta ; Y ) } { \ell ( \pmb \theta ; Y ^ { \prime } ) } = \sum _ { \ell = 1 } ^ { K } \left( s _ { \ell } ( Y ) - s _ { \ell } ( Y ^ { \prime } ) \right) \log \theta _ { \ell } = \log \frac { \theta _ { j } } { \theta _ { k } }\tag{A.38}
$$

$$
\Rightarrow D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( j k ) } ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \frac { \theta _ { k } } { \theta _ { j } } \right] + \frac { 1 } { \alpha - 1 } \left( \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \left( \frac { \theta _ { j } } { \theta _ { k } } \right) ^ { \alpha - 1 } \right] \right) .\tag{A.39}
$$

Finally, to obtain an RDP guarantee that holds for all neighboring datasets under bounded neighboring, one takes the maximum over all ordered pairs $( j , k )$ corresponding to a single-record change:

$$
D _ { \alpha , \mathrm { b o u n d e d } } ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) ) = \operatorname* { m a x } _ { \substack { j , k \in \{ 1 , \dots , K \} , j \neq k } } \bigg \{ D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( j k ) } \big ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \big ) \bigg \} ,\tag{A.40}
$$

Since the maximum ranges over all ordered category pairs $( j , k )$ and all datasets $Y$ in the domain, the reverse ordering $( Y ^ { \prime } , Y )$ , which is itself a bounded-neighboring pair with $j$ and k exchanged and posterior $\pi ( \cdot | Y ^ { \prime } )$ , is already included.

Dirichlet prior. Assume a Dirichlet prior: $\pi ( \pmb \theta ) = \mathrm { D i r } ( \alpha _ { 1 } , \dots , \alpha _ { K } )$ . Then $\pmb \theta | Y \sim \operatorname { D i r } ( \beta _ { 1 } , \dots , \beta _ { K } )$ where $\beta _ { j } = \alpha _ { j } + s _ { j } ( Y )$ for $j \in \{ 1 , \ldots , K \}$ }. We can calculate the moment below:

$$
\mathbb { E } _ { \pmb { \theta } \sim \pi ( \cdot | Y ) } \left[ \frac { \theta _ { k } } { \theta _ { j } } \right] = \frac { \beta _ { k } } { \beta _ { j } - 1 } = \frac { \alpha _ { k } + s _ { k } ( Y ) } { \alpha _ { j } + s _ { j } ( Y ) - 1 } , \mathrm { ~ i f ~ } \beta _ { j } > 1\tag{A.41}
$$

$$
\mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \left( \frac { \theta _ { j } } { \theta _ { k } } \right) ^ { \alpha - 1 } \right] = \frac { \Gamma ( \beta _ { j } + \alpha - 1 ) } { \Gamma ( \beta _ { j } ) } \cdot \frac { \Gamma ( \beta _ { k } - \alpha + 1 ) } { \Gamma ( \beta _ { k } ) } , \mathrm { ~ i f ~ } \beta _ { k } > \alpha - 1\tag{A.42}
$$

Therefore, plug Eqns. (A.42)(A.41) into Eq (A.39), we can derive the closed-form expression for Rényi Divergence:

$$
D _ { \alpha , { \mathrm { b o u n d e d } } } ^ { ( j k ) } \big ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \big ) = \log \frac { \beta _ { k } } { \beta _ { j } - 1 } + \frac { 1 } { \alpha - 1 } \left[ \log \left( \frac { \Gamma ( \beta _ { j } - 1 + \alpha ) } { \Gamma ( \beta _ { j } ) } \cdot \frac { \Gamma ( \beta _ { k } + 1 - \alpha ) } { \Gamma ( \beta _ { k } ) } \right) \right] .
$$

Finally, with $\alpha < 1 + \operatorname* { m i n } _ { k } \beta _ { k }$

$$
D _ { \alpha , \mathrm { b o u n d e d } } ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) ) = \operatorname* { m a x } _ { j , k \in \{ 1 , \ldots , K \} , j \neq k } \Big \{ D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( j k ) } \big ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \big ) \Big \} .\tag{A.43}
$$

Special case. $\mathrm { I f } \exists \ \eta \in ( 0 , 1 / 2 )$ such that $\begin{array} { r } { \forall j \in \{ 1 , \ldots , K \} , \theta _ { j } \in [ \eta , 1 - \eta ] , \mathtt { t h e n } | c _ { 1 } ( \theta ) | \leq \log ( \frac { 1 - \eta } { \eta } ) } \end{array}$

## A.7.2 unbounded neighboring

WLOG, let dataset Y has size n and dataset $Y ^ { \prime }$ is of size $n - 1$ by removing the n-th record $y _ { n } \in \{ 1 , \ldots , K \}$ . Specifically, there exist one index $j \in \{ 1 , \ldots , K \}$ such that $s _ { j } ( \mathbf { \bar { \boldsymbol { Y } } } ) = s _ { j } ( \boldsymbol { Y } ^ { \prime } ) + 1$ and $s _ { \ell } \dot { ( Y ) } = s _ { \ell } \dot { ( Y ^ { \prime } ) }$ for all $\ell \neq j$

$$
c _ { 1 } ( \pmb \theta ) = \log \frac { \ell ( \pmb \theta ; Y ) } { \ell ( \pmb \theta ; Y ^ { \prime } ) } = \sum _ { \ell = 1 } ^ { K } \left( s _ { \ell } ( Y ) - s _ { \ell } ( Y ^ { \prime } ) \right) \log \theta _ { \ell } = \log \theta _ { j }\tag{A.44}
$$

$$
\Rightarrow D _ { \alpha , \mathfrak { u n b o u n d e d } } ^ { ( j ) } ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \big [ \theta _ { j } ^ { - 1 } \big ] + \frac { 1 } { \alpha - 1 } \left( \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \big [ \theta _ { j } ^ { \alpha - 1 } \big ] \right) .\tag{A.45}
$$

For the reverse direction, $\tilde { c } _ { 1 } ( \pmb { \theta } ) = - c _ { 1 } ( \pmb { \theta } ) = - \log \theta _ { j }$ , and by Eq. (9),

$$
D _ { \alpha , { \mathrm { u n b o u n d e d } } } ^ { ( j ) } ( \pi ( \cdot | Y ^ { \prime } ) \| \pi ( \cdot | Y ) ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ^ { \prime } ) } [ \theta _ { j } ] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ^ { \prime } ) } \left[ \theta _ { j } ^ { - ( \alpha - 1 ) } \right] .\tag{A.46}
$$

Finally, to obtain an RDP guarantee that holds for all neighboring datasets under unbounded neighboring, one takes the maximum over all $j$ :

$$
\begin{array} { r l r } & { } & { D _ { \alpha , \mathrm { u n b o u n d e d } } ( \pi ( \cdot | Y ) \parallel \pi ( \cdot | Y ^ { \prime } ) ) = \underset { j \in \{ 1 , \dots , K \} } { \operatorname* { m a x } } \underset { \substack { \theta \in \{ 1 , \dots , K \} } } { \operatorname* { m a x } } \ 6 \big [ D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( j ) } ( \pi ( \cdot | Y ) \parallel \pi ( \cdot | Y ^ { \prime } ) \big ) , } \\ & { } & { D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( j ) } ( \pi ( \cdot | Y ^ { \prime } ) \parallel \pi ( \cdot | Y ) ) \Big \} , } \end{array}
$$

Dirichlet prior. Assume a Dirichlet prior: $\begin{array} { r c l } { { \pi ( \pmb \theta ) } } & { { = } } & { { \mathrm { D i r } ( \alpha _ { 1 } , \ldots , \alpha _ { K } ) } } \end{array}$ . Then $\theta | Y \sim$ Dir $( \beta _ { 1 } , \dots , \beta _ { K } )$ , where $\beta _ { j } ~ = ~ \alpha _ { j } + s _ { j } ( \bar { Y } )$ for $j \in \{ 1 , \ldots , K \}$ . And the marginal $\dot { \theta _ { j } } | Y \sim$ Bet $\begin{array} { r } { \iota ( \beta _ { j } , \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } - \beta _ { j } ) } \end{array}$ , based on which, we can calculate the moments below:

$$
\underline { { \mathbb { E } } } _ { \pmb { \theta } \sim \pi ( \cdot | Y ) } \left[ \theta _ { j } ^ { - 1 } \right] = \frac { \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } - 1 } { \beta _ { j } - 1 } \mathrm { i f } \beta _ { j } > 1\tag{A.47}
$$

$$
\mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \big [ \theta _ { j } ^ { \alpha - 1 } \big ] = \frac { \Gamma ( \beta _ { j } + \alpha - 1 ) } { \Gamma ( \beta _ { j } ) } \cdot \frac { \Gamma ( \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } ) } { \Gamma ( \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } + \alpha - 1 ) } .\tag{A.48}
$$

Therefore, plug Eqns. (A.48)(A.47) into Eq (A.45), we can derive the closed-form expression for Rényi Divergence:

$$
D _ { \alpha , { \mathrm { u n b o u n d e d } } } ^ { ( j ) } ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) ) = \log \frac { \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } - 1 } { \beta _ { j } - 1 } + \frac { 1 } { \alpha - 1 } \log \left[ \frac { \Gamma ( \beta _ { j } + \alpha - 1 ) } { \Gamma ( \beta _ { j } ) } \cdot \frac { \Gamma ( \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } ) } { \Gamma ( \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } + \alpha - 1 ) } \right] .\tag{A.49}
$$

Since $s _ { j } ( Y ) \geq 1 , \beta _ { j } \geq \alpha _ { j } + 1 > 1$ , so this direction is finite for every $\alpha > 1$ . For the reverse direction, $\theta _ { j } | Y ^ { \prime } \sim$ Beta $( \beta _ { j } - 1 , \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } - \beta _ { j } )$ , when $1 < \alpha < \beta _ { j }$

$$
D _ { \alpha , { \mathrm { u n b o u n d e d } } } ^ { ( j ) } ( \pi ( \cdot | Y ^ { \prime } ) \| \pi ( \cdot | Y ) ) = \log \frac { \beta _ { j } - 1 } { \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } - 1 } + \frac { 1 } { \alpha - 1 } \log \left[ \frac { \Gamma ( \beta _ { j } - \alpha ) } { \Gamma ( \beta _ { j } - 1 ) } \cdot \frac { \Gamma ( \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } - 1 ) } { \Gamma ( \sum _ { \ell = 1 } ^ { K } \beta _ { \ell } - \alpha ) } \right] ,\tag{A.50}
$$

but $= \infty$ for $\alpha \geq \beta _ { j }$ . Since $\beta _ { j } \ge \alpha _ { j } + 1$ with equality when $s _ { j } ( Y ) = 1$ , the two-sided divergence is finite uniformly over all unbounded neighboring pairs if and only if $\begin{array} { r } { \alpha < 1 + \operatorname* { m i n } _ { j } \alpha _ { j } ; } \end{array}$ hence OPS with a Dirichlet prior does not satisfy pure DP under unbounded neighboring.

Special Case. If ∃ $\eta \in ( 0 , 1 / 2 )$ such that $\forall j \in \{ 1 , \dots , K \} , \theta _ { j } \in [ \eta , 1 - \eta ]$ , then $\begin{array} { r } { | c _ { 1 } ( \pmb { \theta } ) | \leq \log ( \frac { 1 } { \eta } ) } \end{array}$ In the reverse direction, $\tilde { c } _ { 1 } ( \pmb { \theta } ) = - \log \theta _ { j }$ has a range of the same width, so Corollary 5 applies in both directions.

## A.7.3 Proof of Bernoulli data

Proof. Assume observed dataset $Y = \{ y _ { i } \} _ { i = 1 } ^ { n }$ , where $y _ { i } | \theta \stackrel { \mathrm { i . i . d } } { \sim }$ Bernoulli(θ) with $\theta = \operatorname* { P r } ( y _ { i } = 1 )$ and $1 - \theta = \mathrm { P r } ( y _ { i } = 0 )$

## A.7.3.1 bounded neighboring

WLOG, suppose neighboring datasets $Y , Y ^ { \prime }$ , both of size $n ,$ differ in their n-th record $y _ { n }$ and $y _ { n } ^ { \prime }$ Then the log-likelihood ratio can be derived as

$$
\begin{array} { r l } & { \ell ( \theta ; Y ) = \theta ^ { y _ { n } } + \sum _ { i = 1 } ^ { n - 1 } y _ { i } } \\ & { \ell ( \theta ; Y ^ { \prime } ) = \theta ^ { y _ { n } ^ { \prime } + \sum _ { i = 1 } ^ { n - 1 } y _ { i } } ( 1 - \theta ) ^ { n - y _ { n } ^ { \prime } - \sum _ { i = 1 } ^ { n - 1 } y _ { i } } } \\ & { \Rightarrow c _ { 1 } ( \theta ) = \log \frac { \ell ( \theta ; Y ) } { \ell ( \theta ; Y ^ { \prime } ) } = ( y _ { n } - y _ { n } ^ { \prime } ) \log \bigg ( \cfrac \theta { 1 - \theta } \bigg ) = \left\{ \begin{array} { l l } { \log \bigg ( \frac \theta { 1 - \theta } \bigg ) } & { \mathrm { i f ~ } y _ { n } - y _ { n } ^ { \prime } = 1 } \\ { - \log \bigg ( \frac \theta { 1 - \theta } \bigg ) } & { \mathrm { i f ~ } y _ { n } - y _ { n } ^ { \prime } = - 1 } \end{array} \right. } \end{array}\tag{A.51}
$$

$$
\begin{array} { r } { \operatorname { I f } y _ { n } - y _ { n } ^ { \prime } = 1 \operatorname { w i t h } \mathbb { E } _ { \theta \sim \pi ( \cdot | \mathbf { Y } ) } \big [ \frac { 1 - \theta } { \theta } \big ] < \infty \mathrm { a n d } \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \bigg [ \Big ( \frac { \theta } { 1 - \theta } \Big ) ^ { \alpha - 1 } \bigg ] < \infty \mathrm { . ~ } } \end{array}
$$

$$
D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( + ) } \left( \pi ( \cdot | Y ) \parallel \pi ( \cdot | Y ^ { \prime } ) \right) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \frac { 1 - \theta } { \theta } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \left( \frac { \theta } { 1 - \theta } \right) ^ { \alpha - 1 } \right] .
$$

$$
\begin{array} { r } { \mathrm { I f } y _ { n } - y _ { n } ^ { \prime } = - 1 \operatorname { w i t h } \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \Big [ \frac { \theta } { 1 - \theta } \Big ] < \infty \mathrm { a n d } \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \Big [ \big ( \frac { 1 - \theta } { \theta } \big ) ^ { \alpha - 1 } \Big ] < \infty \mathrm { . ~ } } \end{array}
$$

$$
D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( - ) } \left( \pi ( \cdot | Y ) \parallel \pi ( \cdot | Y ^ { \prime } ) \right) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \frac { \theta } { 1 - \theta } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \left( \frac { 1 - \theta } { \theta } \right) ^ { \alpha - 1 } \right] .
$$

Without additional structure on the posterior, the worst-case Rényi divergence over bounded neighboring is given by

$$
D _ { \alpha , \mathrm { b o u n d e d } } \left( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \right) \leq \operatorname* { m a x } \{ D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( + ) } , D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( - ) } \}
$$

The reverse ordering $( Y ^ { \prime } , Y )$ is itself a bounded-neighboring pair with $y _ { n } ^ { \prime } - y _ { n } = \mp 1$ and posterior $\pi ( \cdot | Y ^ { \prime } )$ , i.e., it is of the form $D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( \mp ) }$ evaluated at the dataset $Y ^ { \prime }$ . Hence taking the maximum over $D ^ { ( + ) }$ and $D ^ { ( - ) }$ and over all datasets in the domain already covers the reverse direction.

Special case. If $\theta \in [ \eta , 1 - \eta ]$ for some constant $\eta \in ( 0 , 1 / 2 )$ . Since $\log ( \frac { \theta } { 1 - \theta } )$ is monotonically increasing, $\begin{array} { r } { | c _ { 1 } ( \theta ) | \le \log ( \frac { 1 - \eta } { \eta } ) } \end{array}$ uniformly.

Beta prior. ${ \mathrm { I f } } \pi ( \theta ) = { \mathrm { B e t a } } ( a , b )$ , and define $\textstyle s = \sum _ { i = 1 } ^ { n } y _ { i }$ , then $\pi ( \theta | Y ) = \operatorname { B e t a } ( A , B )$ , where $A = a + s , B = b + n - s .$

If $y _ { n } - y _ { n } ^ { \prime } = 1$ and $A > 1 , B > \alpha - 1$

$$
\begin{array} { r l } & { \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \displaystyle \frac { 1 - \theta } { \theta } \right] = \frac { B } { A - 1 } } \\ & { \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \left( \frac { \theta } { 1 - \theta } \right) ^ { \alpha - 1 } \right] = \frac { \mathrm { B e t a } \left( A - 1 + \alpha , B + 1 - \alpha \right) } { \mathrm { B e t a } \left( A , B \right) } = \frac { \Gamma ( A + \alpha - 1 ) } { \Gamma ( A ) } \cdot \frac { \Gamma ( B - ( \alpha - 1 ) ) } { \Gamma ( B ) } } \\ & { \Rightarrow D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( + ) } \left( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \right) = \log \left[ \frac { B } { A - 1 } \right] + \frac { 1 } { \alpha - 1 } \log \left[ \frac { \Gamma ( A + \alpha - 1 ) } { \Gamma ( A ) } \cdot \frac { \Gamma ( B - ( \alpha - 1 ) ) } { \Gamma ( B ) } \right] . } \end{array}\tag{A.52}
$$

$$
\mathrm { I f } \ y _ { n } - y _ { n } ^ { \prime } = - 1 , \mathrm { a n d } \ B > 1 , A > \alpha - 1
$$

$$
\begin{array} { r l } & { \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \bigg [ \displaystyle \frac { \theta } { 1 - \theta } \bigg ] = \frac { A } { B - 1 } } \\ & { \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \bigg [ \bigg ( \frac { 1 - \theta } { \theta } \bigg ) ^ { \alpha - 1 } \bigg ] = \frac { \mathrm { B e t a } ( A + 1 - \alpha , B - 1 + \alpha ) } { \mathrm { B e t a } ( A , B ) } = \frac { \Gamma ( A - ( \alpha - 1 ) ) } { \Gamma ( A ) } \cdot \frac { \Gamma ( B + ( \alpha - 1 ) ) } { \Gamma ( B ) } } \\ & { \Rightarrow D _ { \alpha , \mathrm { b o u n d e d } } ^ { ( - ) } ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) ) = \log \bigg [ \frac { A } { B - 1 } \bigg ] + \frac { 1 } { \alpha - 1 } \log \bigg [ \frac { \Gamma ( A - ( \alpha - 1 ) ) } { \Gamma ( A ) } \cdot \frac { \Gamma ( B + ( \alpha - 1 ) ) } { \Gamma ( B ) } \bigg ] } \end{array}\tag{A.53}
$$

Remark A.4. To ensurefinite Beta-moment, $\alpha < \alpha _ { \operatorname* { m a x } } = 1 + \operatorname* { m i n } \{ a + s , \ b + n - s \}$

## A.7.3.2 unbounded neighboring

WLOG, suppose neighboring datasets $Y , Y ^ { \prime }$ , where Y is of size n and $Y ^ { \prime }$ is of size $n - 1$ by removing the n-th record $y _ { n }$ . Then similarly, the log-likelihood ratio can be derived as

$$
\ell ( \theta ; Y ) = \theta ^ { y _ { n } + \sum _ { i = 1 } ^ { n - 1 } y _ { i } } ( 1 - \theta ) ^ { n - y _ { n } - \sum _ { i = 1 } ^ { n - 1 } y _ { i } }
$$

$$
\ell ( \theta ; Y ^ { \prime } ) = \theta ^ { \sum _ { i = 1 } ^ { n - 1 } y _ { i } } ( 1 - \theta ) ^ { n - 1 - \sum _ { i = 1 } ^ { n - 1 } y _ { i } }
$$

$$
\Rightarrow c _ { 1 } ( \theta ) = \log { \frac { \ell ( \theta ; Y ) } { \ell ( \theta ; Y ^ { \prime } ) } } = y _ { n } \log \theta + ( 1 - y _ { n } ) \log ( 1 - \theta ) = { \left\{ \log \theta \begin{array} { l l } { \log \theta } & { { \mathrm { i f ~ } } y _ { n } = 1 } \\ { \log \left( 1 - \theta \right) } & { { \mathrm { i f ~ } } y _ { n } = 0 } \end{array} \right. }\tag{A.54}
$$

If y<sub>n</sub> = 1,

$$
D _ { \alpha , \operatorname { u n b o u n d e d } } ^ { ( 1 ) } \big ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \big ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \theta ^ { - 1 } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \theta ^ { \alpha - 1 } \right] .
$$

If $y _ { n } = 0 ,$

$$
D _ { \alpha , \mathfrak { m b o u n d e d } } ^ { ( 0 ) } ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ ( 1 - \theta ) ^ { - 1 } \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \left( 1 - \theta \right) ^ { \alpha - 1 } \right] .
$$

For the reverse direction, $\tilde { c } _ { 1 } ( \theta ) = - c _ { 1 } ( \theta )$ and the expectations are taken under $\pi ( \cdot | Y ^ { \prime } )$

$$
\begin{array} { r l } & { D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( 1 ) } \big ( \pi ( \cdot | Y ^ { \prime } ) \| \pi ( \cdot | Y ) \big ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ^ { \prime } ) } \left[ \theta \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ^ { \prime } ) } \left[ \theta ^ { - ( \alpha - 1 ) } \right] , } \\ & { D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( 0 ) } \big ( \pi ( \cdot | Y ^ { \prime } ) \| \pi ( \cdot | Y ) \big ) = \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ^ { \prime } ) } \left[ 1 - \theta \right] + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { \theta \sim \pi ( \cdot | Y ^ { \prime } ) } \left[ ( 1 - \theta ) ^ { - ( \alpha - 1 ) } \right] . } \end{array}
$$

Without additional structure on the posterior, the worst-case Rényi divergence over unbounded neighboring is given by

$$
D _ { \alpha , \mathrm { u n b o u n d e d } } \leq \operatorname* { m a x } _ { k \in \{ 0 , 1 \} } \operatorname* { m a x } \left\{ D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( k ) } \big ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \big ) , D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( k ) } \big ( \pi ( \cdot | Y ^ { \prime } ) \| \pi ( \cdot | Y ) \big ) \right\}
$$

Special case: $\mathrm { I f } \theta \in [ \eta , 1 - \eta ]$ for some $\eta \in ( 0 , 1 / 2 ) . \left| c _ { 1 } ( \theta ) \right| \le \operatorname* { m a x } \{ | \log \eta | , | \log ( 1 - \eta ) | \} = - \log \eta$ uniformly. In the reverse direction, $\tilde { c } _ { 1 } ( \theta ) = - c _ { 1 } ( \theta )$ satisfies the same bound, so Corollary 5 applies in both directions.

Beta prior. If $\pi ( \theta ) = \operatorname { B e t a } ( a , b )$ , and define $\textstyle s = \sum _ { i = 1 } ^ { n } y _ { i }$ , then $\pi ( \theta | Y ) = \operatorname { B e t a } ( A , B )$ , where $A = a + s , B = b + n - s$ . Then

If $y _ { n } = 1$ and A > 1,

$$
\mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \left[ \theta ^ { - 1 } \right] = \frac { A + B - 1 } { A - 1 } ;\tag{A.55}
$$

$$
\mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \big [ \theta ^ { \alpha - 1 } \big ] = \frac { \mathrm { B e t a } ( A - 1 + \alpha , B ) } { \mathrm { B e t a } ( A , B ) } = \frac { \Gamma ( A + \alpha - 1 ) \Gamma ( A + B ) } { \Gamma ( A ) \Gamma ( A + B + \alpha - 1 ) }\tag{A.56}
$$

$$
\Rightarrow D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( 1 ) } \bigl ( \pi ( \cdot | Y ) | | \pi ( \cdot | Y ^ { \prime } ) \bigr ) = \log \left( \frac { A + B - 1 } { A - 1 } \right) + \frac { 1 } { \alpha - 1 } \log \left( \frac { \Gamma ( A + \alpha - 1 ) \Gamma ( A + B ) } { \Gamma ( A ) \Gamma ( A + B + \alpha - 1 ) } \right) .\tag{A.57}
$$

If y<sub>n</sub> = 0, and B > 1,

$$
\mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \big [ ( 1 - \theta ) ^ { - 1 } \big ] = \frac { A + B - 1 } { B - 1 } ;\tag{A.58}
$$

$$
\mathbb { E } _ { \theta \sim \pi ( \cdot | Y ) } \Big [ \big ( 1 - \theta ) ^ { \alpha - 1 } \Big ] = \frac { \mathrm { B e t a } ( A , B - 1 + \alpha ) } { \mathrm { B e t a } ( A , B ) } = \frac { \Gamma ( A + B ) \Gamma ( B + \alpha - 1 ) } { \Gamma ( A + B + \alpha - 1 ) \Gamma ( B ) }\tag{A.59}
$$

$$
\Rightarrow D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( 0 ) } \big ( \pi ( \cdot | Y ) \| \pi ( \cdot | Y ^ { \prime } ) \big ) = \log \left( \frac { A + B - 1 } { B - 1 } \right) + \frac { 1 } { \alpha - 1 } \log \left( \frac { \Gamma ( B + \alpha - 1 ) \Gamma ( A + B ) } { \Gamma ( B ) \Gamma ( A + B + \alpha - 1 ) } \right) .\tag{A.60}
$$

$$
\mathrm { F o r t h e ~ r e v e r s e ~ d i r e c t i o n , i f ~ } y _ { n } = 1 , \mathrm { t h e n } \pi ( \theta | Y ^ { \prime } ) = \mathrm { B e t a } ( A - 1 , B ) \mathrm { ~ a n d ~ i f ~ } 1 < \alpha < A ,
$$

$$
D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( 1 ) } ( \pi ( \cdot | Y ^ { \prime } ) | | \pi ( \cdot | Y ) ) = \log \left( \frac { A - 1 } { A + B - 1 } \right) + \frac { 1 } { \alpha - 1 } \log \left( \frac { \Gamma ( A - \alpha ) \Gamma ( A + B - 1 ) } { \Gamma ( A - 1 ) \Gamma ( A + B - \alpha ) } \right) ;\tag{A.61}
$$

if y = 0, then π(θ|Y<sup>′</sup>) = Beta(A, B − 1) and if $1 < \alpha < B$

$$
D _ { \alpha , \mathrm { u n b o u n d e d } } ^ { ( 0 ) } ( \pi ( \cdot | Y ^ { \prime } ) | | \pi ( \cdot | Y ) ) = \log \left( \frac { B - 1 } { A + B - 1 } \right) + \frac { 1 } { \alpha - 1 } \log \left( \frac { \Gamma ( B - \alpha ) \Gamma ( A + B - 1 ) } { \Gamma ( B - 1 ) \Gamma ( A + B - \alpha ) } \right) .\tag{A.62}
$$

Both equal ∞ when α exceeds the stated range.

Remark A.5. Theforward direction under unbounded neighboring imposes no upper bound on the Rényi order $\alpha ,$ , but the reverse direction is finite only for $\alpha < A ( i f y _ { n } = 1 )$ or $\alpha < B ( i f y _ { n } = 0 )$ Since $A = a + s \geq a + 1$ when $y _ { n } = 1$ and $B = b + n - s \geq b + 1$ when $y _ { n } = 0 ,$ , with equality attainable, the two-sided Rényi divergence is finite uniformly over all unbounded neighboring pairs if and only $i f \alpha < 1 + \operatorname* { m i n } \{ a , \dot { b } \}$ . In particular, OPS with a Beta prior does not satisfy pure DP under unbounded neighboring.

□

## A.8 Proof of Proposition 9 for Generalized Linear Models

Proof. We consider generalized linear models (GLMs) with canonical link. Each record is $\left( \mathbf { x } _ { i } , y _ { i } \right)$ where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ and $y _ { i } \in \mathcal { V } \subseteq \mathbb { R }$ . The dataset is $\mathbf { Y } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , and the model parameter is $\beta \in \mathbb { R } ^ { d }$ . Conditioned on $\mathbf { x } _ { i }$ and $\beta ,$ the response $y _ { i }$ follows an exponential family distribution as below:

$$
p ( y _ { i } | \mathbf { x } _ { i } , \beta ) = \exp \left( y _ { i } \mathbf { x } _ { i } ^ { \top } \beta - \psi ( \mathbf { x } _ { i } ^ { \top } \beta ) + h ( y _ { i } ) \right) ,\tag{A.63}
$$

where $\psi ( \cdot )$ is the log-partition function and $h ( \cdot )$ does not depend on $\beta .$ . Then the log-likelihood is

$$
\log p ( y _ { i } | \mathbf { x } _ { i } , \beta ) = y _ { i } \mathbf { x } _ { i } ^ { \top } \beta - \psi ( \mathbf { x } _ { i } ^ { \top } \beta ) + h ( y _ { i } )\tag{A.64}
$$

$$
\Rightarrow \ell ( \boldsymbol { \beta } | \mathbf { Y } ) = \sum _ { i = 1 } ^ { n } \log p ( y _ { i } | \mathbf { x } _ { i } , \boldsymbol { \beta } ) = \sum _ { i = 1 } ^ { n } \left( y _ { i } \mathbf { x } _ { i } ^ { \top } \boldsymbol { \beta } - \psi ( \mathbf { x } _ { i } ^ { \top } \boldsymbol { \beta } ) \right) + \sum _ { i = 1 } ^ { n } h ( y _ { i } ) .\tag{A.65}
$$

Suppose there exists a finite constant $L \ > \ 0$ such that for every $\left( \mathbf { x } _ { i } , y _ { i } \right)$ and every $\beta ,$ $\| \nabla _ { \beta }$ log $p ( y _ { i } | \mathbf { x } _ { i } , \beta ) \| \ \leq \ L$ . If imposing a Gaussian prior on $\beta$ , i.e. $\bar { { \pi } ( \beta ) } = \bar { \mathcal { N } } ( 0 , \tau ^ { 2 } \mathbf { I } _ { d } )$ , then the posterior distribution can be derived as

$$
\pi ( \beta | \mathbf { Y } ) \propto \exp \left( \sum _ { i = 1 } ^ { n } \bigl ( y _ { i } \mathbf { x } _ { i } ^ { \top } \beta - \psi ( \mathbf { x } _ { i } ^ { \top } \beta ) \bigr ) - \frac { \| \beta \| _ { 2 } ^ { 2 } } { 2 \tau ^ { 2 } } \right) .\tag{A.66}
$$

## A.8.1 bounded neighboring

Let Y and $\mathbf { Y } ^ { \prime }$ differ in exactly one record, i.e. ${ \bf z } _ { n } = ( { \bf x } _ { n } , y _ { n } )$ in Y is replaced by $\mathbf z _ { n } ^ { \prime } = ( \mathbf x _ { n } ^ { \prime } , y _ { n } ^ { \prime } )$ in $\mathbf { Y } ^ { \prime }$ , then

$$
\begin{array} { r l } & { c _ { 1 } ( \boldsymbol { \beta } ) = \log \frac { \ell ( \boldsymbol { \beta } ; \mathbf { Y } ) } { \ell ( \boldsymbol { \beta } ; \mathbf { Y } ^ { \prime } ) } = \log p ( y _ { n } | \mathbf { x } _ { n } , \boldsymbol { \beta } ) - \log p ( y _ { n } ^ { \prime } | \mathbf { x } _ { n } ^ { \prime } , \boldsymbol { \beta } ) } \\ & { \phantom { c _ { 1 } ( \boldsymbol { \beta } ) = \log p ( y _ { n } ^ { \prime } | \mathbf { x } _ { n } ^ { \prime } , \boldsymbol { \beta } ) = \log p ( y _ { n } ^ { \prime } | \mathbf { x } _ { n } ^ { \prime } , \boldsymbol { \beta } ) = \log p ( y _ { n } ^ { \prime } | \mathbf { x } _ { n } ^ { \prime \top } \boldsymbol { \beta } - \psi ( \mathbf { x } _ { n } ^ { \prime \top } \boldsymbol { \beta } ) + h ( y _ { n } ^ { \prime } ) ) } } \\ & { \phantom { c _ { 2 } ( \boldsymbol { \beta } ) = \log p ( y _ { n } ^ { \prime } | \mathbf { x } _ { n } ^ { \prime \top } ) = \log p ( y _ { n } ^ { \prime } | \mathbf { x } _ { n } ^ { \prime \top } \boldsymbol { \beta } - \left( \psi ( \mathbf { x } _ { n } ^ { \top } \boldsymbol { \beta } ) - \psi ( \mathbf { x } _ { n } ^ { \prime \top } \boldsymbol { \beta } ) \right) + h ( y _ { n } ) - h ( y _ { n } ^ { \prime } ) } } \end{array}\tag{A.67}
$$

First, we prove $c _ { 1 } ( \beta )$ is Lipschitz.

$$
\nabla _ { \beta } c _ { 1 } ( \beta ) = y _ { n } \mathbf { x } _ { n } - y _ { n } ^ { \prime } \mathbf { x } _ { n } ^ { \prime } - \left( \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \top } \beta ) \mathbf { x } _ { n } - \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) \mathbf { x } _ { n } ^ { \prime } \right)\tag{A.68}
$$

$$
\begin{array} { r l } { \Rightarrow \| \nabla _ { \beta } c _ { 1 } ( \beta ) \| = \Big \| y _ { n } \mathbf { x } _ { n } - y _ { n } ^ { \prime } \mathbf { x } _ { n } ^ { \prime } - \big ( \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \top } \beta ) \mathbf { x } _ { n } - \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) \mathbf { x } _ { n } ^ { \prime } \big ) \Big \| } & { } \\ { = \Big \| \big ( y _ { n } - \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \top } \beta ) \big ) \mathbf { x } _ { n } - \big ( y _ { n } ^ { \prime } - \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) \big ) \mathbf { x } _ { n } ^ { \prime } \Big \| } & { } \\ { \leq | y _ { n } - \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \top } \beta ) | \| \mathbf { x } _ { n } \| + | y _ { n } ^ { \prime } - \psi ^ { \prime } ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) | \| \mathbf { x } _ { n } ^ { \prime } \| \leq 2 L } \end{array}\tag{A.69}
$$

Hence, $c _ { 1 } ( \beta )$ is Lipschitz continuous with constant 2L. Then, we continue to prove the log-concavity of the posterior $\pi ( \bar { \boldsymbol { \beta } } | \mathbf { Y } )$ .

$$
\log \pi ( \beta | \mathbf { Y } ) = \sum _ { i = 1 } ^ { n } \bigl ( y _ { i } \mathbf { x } _ { i } ^ { \top } \beta - \psi ( \mathbf { x } _ { i } ^ { \top } \beta ) \bigr ) - \frac { \| \beta \| _ { 2 } ^ { 2 } } { 2 \tau ^ { 2 } } + \mathrm { c o n s t a n t }\tag{A.70}
$$

$$
\Rightarrow - \nabla _ { \beta } ^ { 2 } \log \pi ( \beta | \mathbf { Y } ) = \sum _ { i = 1 } ^ { n } \psi ^ { \prime \prime } ( \mathbf { x } _ { i } ^ { \top } \beta ) \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \top } + \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \succeq \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \quad \mathrm { p e r ~ c o n v e x i t y ~ o f } \ \psi ( \cdot ) .\tag{A.71}
$$

So $\pi ( \beta | \mathbf { Y } )$ is m-strongly log-concave with $m = \tau ^ { - 2 }$

Lemma A.6 (Log-Sobolev Inequality [Ledoux, 2001]). Let π be a probability measure on $\mathbb { R } ^ { d }$ with density π $\mathbf { \boldsymbol { \cdot } } ( \mathbf { \boldsymbol { x } } ) \propto \bar { e } ^ { - U ( \mathbf { \boldsymbol { x } } ) }$ , where U is twice continuously differentiable and m-strongly convex, i.e., $\nabla ^ { 2 } U ( x ) \succeq m I _ { d , }$ for all x. Then π satisfies a logarithmic Sobolev inequality (LSI) with constant $1 / m \colon$

$$
\mathrm { E n t } _ { \pi } ( g ^ { 2 } ) \leq \frac 2 m \mathbb { E } _ { \pi } \left[ \lVert \nabla g \rVert ^ { 2 } \right] , \quad f o r a l l s m o o t h g .
$$

Lemma A.7 (Herbst’s Theorem [Ledoux, 2001]). Let π be a probability measure satisfying a LSI with constant $C _ { \mathrm { L S I } }$ . Thenfor any L-Lipschitzfunction f,

$$
\log \mathbb { E } _ { \pi } \left[ \exp \bigl ( t ( f - \mathbb { E } _ { \pi } f ) \bigr ) \right] \leq \frac { C _ { \mathrm { L S I } } L ^ { 2 } t ^ { 2 } } { 2 } , \quad \forall t \in \mathbb { R } .
$$

In particular, f is sub-Gaussian under π with variance proxy $C _ { \mathrm { L S I } } L ^ { 2 }$

Per above lemmas, $c _ { 1 } ( \beta )$ is sub-Gaussian under $\pi ( \cdot | \mathbf { Y } )$ with

$$
\sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } \leq \frac { ( 2 L ) ^ { 2 } } { m } = 4 L ^ { 2 } \tau ^ { 2 } .\tag{A.72}
$$

The same argument applies to the reversed pair $( \mathbf { Y } ^ { \prime } , \mathbf { Y } ) \colon \tilde { c } _ { 1 } ( \boldsymbol { \beta } ) = - c _ { 1 } ( \boldsymbol { \beta } )$ has the same Lipschitz constant 2L, and $\pi ( \bar { \beta } | { \bf \bar { Y } } ^ { \prime } )$ is also $\tau ^ { - 2 }$ -strongly log-concave since $\mathbf { Y } ^ { \prime }$ also has n records. Hence $\sigma _ { \mathbf { Y } ^ { \prime } , \mathbf { Y } } ^ { 2 } \le 4 L ^ { 2 } \tau ^ { 2 }$ , and $\mathring { \sigma } ^ { 2 } \leq 4 L ^ { 2 } \tau ^ { 2 }$ . Plug into Corollary 8

$$
\operatorname* { m a x } \{ D _ { \alpha } ( \pi ( \beta | \mathbf { Y } ) \| \pi ( \beta | \mathbf { Y } ^ { \prime } ) ) , D _ { \alpha } ( \pi ( \beta | \mathbf { Y } ^ { \prime } ) \| \pi ( \beta | \mathbf { Y } ) ) \} \le \frac { \alpha \bar { \sigma } ^ { 2 } } { 2 } \le 2 \alpha L ^ { 2 } \tau ^ { 2 }\tag{A.73}
$$

## A.8.2 unbounded neighboring

Let Y has size n and $\mathbf { Y } ^ { \prime }$ is of size $n - 1$ by removing the n-th record ${ \bf z } _ { n } = ( { \bf x } _ { n } , y _ { n } )$ in Y. Then

$$
c _ { 1 } ( \beta ) = \log \frac { \ell ( \beta ; \mathbf { Y } ) } { \ell ( \beta ; \mathbf { Y } ^ { \prime } ) } = \log p ( y _ { n } | _ { \mathbf { x } _ { n } } , \beta ) = y _ { n } \mathbf { x } _ { n } ^ { \top } \beta - \psi ( \mathbf { x } _ { n } ^ { \top } \beta ) + h ( y _ { n } ) .\tag{A.74}
$$

Similarly, we prove $c _ { 1 } ( \beta )$ is Lipschitz.

$$
\begin{array} { r l } & { \quad \nabla _ { \boldsymbol { \beta } } c _ { 1 } ( \boldsymbol { \beta } ) = y _ { n } \mathbf x _ { n } - \psi ^ { \prime } ( \mathbf x _ { n } ^ { \top } \boldsymbol { \beta } ) \mathbf x _ { n } } \\ & { \Rightarrow \| \nabla _ { \boldsymbol { \beta } } c _ { 1 } ( \boldsymbol { \beta } ) \| = \left\| \left( y _ { n } - \psi ^ { \prime } ( \mathbf x _ { n } ^ { \top } \boldsymbol { \beta } ) \right) \mathbf x _ { n } \right\| \leq L . } \end{array}\tag{A.75}
$$

Combined with log-concavity, $c _ { 1 } ( \beta )$ is sub-Gaussian under $\pi ( \cdot | \mathbf { Y } )$ with

$$
\sigma _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } \leq \frac { L ^ { 2 } } { m } \leq L ^ { 2 } \tau ^ { 2 } .\tag{A.76}
$$

For the reversed pair $( { \bf Y ^ { \prime } } , { \bf Y } ) , { \tilde { c } } _ { 1 } ( \beta ) = - \log p ( y _ { n } | { \bf x } _ { n } , \beta )$ has the same gradient norm bound L, and

$$
- \nabla _ { \beta } ^ { 2 } \log \pi ( \beta | \mathbf { Y } ^ { \prime } ) = \sum _ { i = 1 } ^ { n - 1 } \psi ^ { \prime \prime } ( \mathbf { x } _ { i } ^ { \top } \beta ) \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \top } + \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \succeq \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } ,\tag{A.77}
$$

where the empty sum equals 0 when $n = 1$ . Hence $\pi ( \beta | { \bf Y } ^ { \prime } )$ is also $\tau ^ { - 2 }$ -strongly log-concave, $\sigma _ { \mathbf { Y } ^ { \prime } , \mathbf { Y } } ^ { 2 } \le L ^ { 2 } \tau ^ { \frac { 1 } { 2 } }$ , and $\bar { \sigma } ^ { 2 } \leq L ^ { 2 } \tau ^ { 2 }$ . Plug into Corollary 8

$$
\operatorname* { m a x } \{ D _ { \alpha } ( \pi ( \beta | \mathbf { Y } ) \| \pi ( \beta | \mathbf { Y } ^ { \prime } ) ) , D _ { \alpha } ( \pi ( \beta | \mathbf { Y } ^ { \prime } ) \| \pi ( \beta | \mathbf { Y } ) ) \} \le \frac { \alpha { \bar { \sigma } } ^ { 2 } } { 2 } \le \frac { \alpha L ^ { 2 } \tau ^ { 2 } } { 2 } .\tag{A.78}
$$

## A.8.3 Proof of Example 10 for Logistic Regression

## A.8.3.1 bounded neighboring

Proof. For logistic regression we have $\psi ( t ) = \log ( 1 + e ^ { t } )$ , and $h ( \cdot ) = 0$ , plug into Eq. (A.67):

$$
c _ { 1 } ( \beta ) = y _ { n } \mathbf { x } _ { n } ^ { \top } \beta - y _ { n } ^ { \prime } \mathbf { x } _ { n } ^ { \prime \top } \beta - \left( \log ( 1 + e ^ { \mathbf { x } _ { n } ^ { \top } \beta } ) - \log ( 1 + e ^ { \mathbf { x } _ { n } ^ { \prime \top } \beta } ) \right)\tag{A.79}
$$

First, we prove $c _ { 1 } ( \beta )$ is Lipschitz. Recall $\| \mathbf { x } \| \leq R$ and let $\begin{array} { r } { \sigma ( t ) = \frac { 1 } { 1 + e ^ { - t } } } \end{array}$ , given $| \sigma ( \cdot ) | < 1$ and $\begin{array} { r } { \frac { d \log ( 1 + e ^ { t } ) } { d t } = \sigma ( t ) } \end{array}$ , we can have

$$
\begin{array} { r l } { \nabla _ { \beta } c _ { 1 } ( \beta ) = y _ { n } \mathbf { x } _ { n } - y _ { n } ^ { \prime } \mathbf { x } _ { n } ^ { \prime } - \Big ( \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) \mathbf { x } _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) \mathbf { x } _ { n } ^ { \prime } \Big ) } & { } \\ { \mathrm { ~ } } & { \mathrm { ~ i f ~ } \mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } \mathrm { ~ a n d ~ } y _ { n } \neq y _ { n } ^ { \prime } } \\ { = \left\{ \begin{array} { l l } { y _ { n } \mathbf { x } _ { n } - y _ { n } \mathbf { x } _ { n } ^ { \prime } - \Big ( \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) \mathbf { x } _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) \mathbf { x } _ { n } ^ { \prime } \Big ) } & { \mathrm { ~ i f ~ } \mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime } \mathrm { ~ a n d ~ } y _ { n } = y _ { n } ^ { \prime } } \\ { y _ { n } \mathbf { x } _ { n } - y _ { n } ^ { \prime } \mathbf { x } _ { n } ^ { \prime } - \Big ( \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) \mathbf { x } _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) \mathbf { x } _ { n } ^ { \prime } \Big ) } & { \mathrm { ~ i f ~ } \mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime } \mathrm { ~ a n d ~ } y _ { n } \neq y _ { n } ^ { \prime } } \\ { 0 } & { \mathrm { ~ i f ~ } \mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } \mathrm { ~ a n d ~ } y _ { n } = y _ { n } ^ { \prime } } \end{array} \right. . } \end{array}
$$

Therefore, when ${ \bf x } _ { n } = { \bf x } _ { n } ^ { \prime }$ , the worst case is indeed flipping the label, then $\| \nabla _ { \beta } c _ { 1 } ( \beta ) \| = \| ( y _ { n } -$ $y _ { n } ^ { \prime } ) \mathbf { x } _ { n } \| \leq \| \mathbf { x } _ { n } \| \leq R$ . Otherwise,

$$
\begin{array} { r l } & { \| \nabla _ { \beta } c _ { 1 } ( \beta ) \| \leq \| y _ { n } \mathbf { x } _ { n } - y _ { n } ^ { \prime } \mathbf { x } _ { n } ^ { \prime } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) \mathbf { x } _ { n } + \sigma ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) \mathbf { x } _ { n } ^ { \prime } \| } \\ & { \qquad \leq \| ( y _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) ) \mathbf { x } _ { n } - ( y _ { n } ^ { \prime } - \sigma ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) ) \mathbf { x } _ { n } ^ { \prime } \| } \\ & { \qquad \leq \| ( y _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) ) \mathbf { x } _ { n } \| + \| ( y _ { n } ^ { \prime } - \sigma ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) ) \mathbf { x } _ { n } ^ { \prime } \| } \\ & { \qquad \leq \underbrace { | y _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) | } _ { \leq 1 } \cdot \| \mathbf { x } _ { n } \| + \underbrace { | y _ { n } ^ { \prime } - \sigma ( \mathbf { x } _ { n } ^ { \prime \top } \beta ) | } _ { \leq 1 } \cdot \| \mathbf { x } _ { n } ^ { \prime } \| \leq 2 R . } \end{array}\tag{A.80}
$$

Now we have proved that $c _ { 1 } ( \beta )$ is Lipschitz continuous with constant $L = R$ or $L = 2 R$ . Then, we similarly prove the log-concavity of the posterior $\pi ( \cdot | \mathbf { Y } )$

$$
\begin{array} { c } { \displaystyle \log \pi ( \beta | \mathbf { Y } ) = \displaystyle \sum _ { i = 1 } ^ { n } \bigl ( y _ { i } \mathbf { x } _ { i } ^ { \top } \boldsymbol { \beta } - \log ( 1 + e ^ { \mathbf { x } _ { i } ^ { \top } \boldsymbol { \beta } } ) \bigr ) - \frac { \| \boldsymbol { \beta } \| _ { 2 } ^ { 2 } } { 2 \tau ^ { 2 } } + \mathrm { c o n s t a n t } } \\ { \displaystyle \Rightarrow - \nabla _ { \boldsymbol { \beta } } ^ { 2 } \log \pi ( \boldsymbol { \beta } | \mathbf { Y } ) = \displaystyle \sum _ { i = 1 } ^ { n } \sigma ( \mathbf { x } _ { i } ^ { \top } \boldsymbol { \beta } ) ( 1 - \sigma ( \mathbf { x } _ { i } ^ { \top } \boldsymbol { \beta } ) ) \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \top } + \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \succeq \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \mathrm { f o r } \forall \boldsymbol { \beta } . } \end{array}\tag{A.81}
$$

So $\pi ( \beta | \mathbf { Y } )$ is m-strongly log-concave with $m = \tau ^ { - 2 }$ . Therefore,

$$
\sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } \leq { \frac { L ^ { 2 } } { m } } \leq { \left\{ \begin{array} { l l } { R ^ { 2 } \tau ^ { 2 } } & { { \mathrm { ~ i f ~ } } \mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } } \\ { ( 2 R ) ^ { 2 } \tau ^ { 2 } = 4 R ^ { 2 } \tau ^ { 2 } } & { { \mathrm { ~ i f ~ } } \mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime } } \end{array} \right. } .\tag{A.82}
$$

For the reversed pair $( { \bf Y ^ { \prime } } , { \bf Y } )$ , the Lipschitz constants $R ( \operatorname { i f } \mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime } )$ and $2 R ( \mathrm { i f } \ \mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime } )$ are symmetric in the two records, and $\pi ( \bar { \boldsymbol { \beta } } | \mathbf { Y } ^ { \prime } )$ is also τ<sup>−2</sup>-strongly log-concave, so the same bounds hold for $\sigma _ { \mathbf { Y ^ { \prime } } , \mathbf { Y } } ^ { 2 }$ . Plug into Corollary 8

$$
\operatorname* { m a x } \{ D _ { \alpha } ( \pi ( \beta | { \bf Y } ) \| \pi ( \beta | { \bf Y } ^ { \prime } ) ) , D _ { \alpha } ( \pi ( \beta | { \bf Y } ^ { \prime } ) \| \pi ( \beta | { \bf Y } ) ) \} \le \frac { \alpha \bar { \sigma } ^ { 2 } } { 2 } \le \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \alpha R ^ { 2 } \tau ^ { 2 } } & { \mathrm { ~ i f ~ } { \bf x } _ { n } = { \bf x } _ { n } ^ { \prime } } \\ { 2 \alpha R ^ { 2 } \tau ^ { 2 } } & { \mathrm { ~ i f ~ } { \bf x } _ { n } \ne { \bf x } _ { n } ^ { \prime } } \end{array} \right. .\tag{A.83}
$$

where $\bar { \sigma } ^ { 2 }$ is taken over the corresponding class of neighboring pairs.

For comparison, Theorem 16 in [Geumlek et al., 2017] gives $( \alpha , 2 \alpha R ^ { 2 } \tau ^ { 2 } )$ -RDP for the replacement case with ${ \bf x } _ { n } = { \bf x } _ { n } ^ { \prime }$ ; our bound for this case is $\scriptstyle \alpha R ^ { 2 } \tau ^ { 2 } / 2$ □

## A.8.3.2 unbounded neighboring

Proof. Now we have $\psi ( t ) = \log ( 1 + e ^ { t } )$ , and $h ( \cdot ) = 0$ . Similarly, we prove $c _ { 1 } ( \beta )$ is Lipschitz first as what we did in bounded neighboring. Recall $\| \mathbf { x } \| \leq R$ and let $\begin{array} { r } { \sigma ( t ) \ = \ \frac { 1 } { 1 + e ^ { - t } } } \end{array}$ , given $y \in \{ 0 , 1 \} , | \sigma ( t ) | < 1$ for $\forall t \in \mathbb { R }$ and $\begin{array} { r } { \frac { d \log ( 1 + e ^ { t } ) } { d t } = \sigma ( t ) } \end{array}$ , we have

$$
\begin{array} { r l } & { \quad \nabla _ { \beta } c _ { 1 } ( \beta ) = y _ { n } \mathbf { x } _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) \mathbf { x } _ { n } \quad \mathrm { p e r ~ E q . ~ } ( \mathbf { A . } 7 4 ) } \\ & { \Rightarrow \| \nabla _ { \beta } c _ { 1 } ( \beta ) \| = \| ( y _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) ) \mathbf { x } _ { n } \| \leq \underbrace { | y _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) | } _ { \leq 1 } \| \mathbf { x } _ { n } \| \leq R } \end{array}\tag{A.84}
$$

Now we have proved that $c _ { 1 } ( \beta )$ is Lipschitz continuous with constant $L = R$ . Given $\pi ( \beta | \mathbf { Y } )$ is m-strongly log-concave with $\dot { m } = \tau ^ { - \dot { 2 } }$ and the Lemmas, $c _ { 1 } ( \beta )$ is sub-Gaussian under $\pi ( \cdot | \mathbf { \dot { Y } } )$ with

$$
\sigma _ { \mathbf { Y } , \mathbf { Y ^ { \prime } } } ^ { 2 } \leq \frac { L ^ { 2 } } { m } \leq R ^ { 2 } \tau ^ { 2 }\tag{A.85}
$$

For the reversed pair $( \mathbf { Y } ^ { \prime } , \mathbf { Y } ) , \| \nabla _ { \beta } \widetilde { c } _ { 1 } ( \beta ) \| = | y _ { n } - \sigma ( \mathbf { x } _ { n } ^ { \top } \beta ) | \| \mathbf { x } _ { n } \| \le R ,$ and $\pi ( \beta | \mathbf { Y } ^ { \prime } )$ is $\tau ^ { - 2 } .$ strongly log-concave by the argument for general GLMs above. Hence $\sigma _ { \mathbf { Y } ^ { \prime } , \mathbf { Y } } ^ { 2 } \le R ^ { 2 } \tau ^ { 2 }$ and $\bar { \sigma } ^ { 2 } \leq$ $R ^ { 2 } \tau ^ { 2 }$ . Plug into Corollary 8

$$
\operatorname* { m a x } \{ D _ { \alpha } ( \pi ( \beta | \mathbf { Y } ) \| \pi ( \beta | \mathbf { Y } ^ { \prime } ) ) , D _ { \alpha } ( \pi ( \beta | \mathbf { Y } ^ { \prime } ) \| \pi ( \beta | \mathbf { Y } ) ) \} \le \frac { \alpha \bar { \sigma } ^ { 2 } } { 2 } \le \frac { \alpha R ^ { 2 } \tau ^ { 2 } } { 2 }\tag{A.86}
$$

## A.9 Proof of Proposition 11 for Gaussian-Gaussian conjugate pair

Proof. Assume observed dataset $Y = \{ y _ { i } \} _ { i = 1 } ^ { n }$ , where $y _ { i } | \theta \stackrel { \mathrm { i . i . d } } { \sim } \mathcal { N } ( \theta , \sigma ^ { 2 } )$ and $\sigma ^ { 2 }$ is known. If imposing a Gaussian prior on θ, i.e. $\pi ( \theta ) = \bar { \mathcal { N } } ( \mu _ { 0 } , \tau ^ { 2 } )$ , then per conjugacy, the posterior $\theta | Y \sim$ $\mathcal { N } ( \mu _ { Y } , \bar { v _ { Y } ^ { 2 } } )$ , where

$$
v _ { Y } ^ { 2 } = \left( { \frac { n } { \sigma ^ { 2 } } } + { \frac { 1 } { \tau ^ { 2 } } } \right) ^ { - 1 } , \qquad \mu _ { Y } = v _ { Y } ^ { 2 } \left( { \frac { 1 } { \sigma ^ { 2 } } } \sum _ { i = 1 } ^ { n } y _ { i } + { \frac { \mu _ { 0 } } { \tau ^ { 2 } } } \right) .\tag{A.87}
$$

## A.9.1 bounded neighboring

WLOG, suppose neighboring datasets $Y , Y ^ { \prime }$ , both of size $n ,$ differ in their n-th record $y _ { n }$ and $y _ { n } ^ { \prime }$ Then the log-likelihood ratio $c _ { 1 } ( \theta )$ can be derived as

$$
\ell ( \theta ; Y ) = ( 2 \pi \sigma ^ { 2 } ) ^ { - n / 2 } \exp \left( - \frac { ( y _ { n } - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } - \sum _ { i = 1 } ^ { n - 1 } \frac { ( y _ { i } - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ;\tag{A.88}
$$

$$
\ell ( \theta ; Y ^ { \prime } ) = ( 2 \pi \sigma ^ { 2 } ) ^ { - n / 2 } \exp \left( - { \frac { ( y _ { n } ^ { \prime } - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } } - \sum _ { i = 1 } ^ { n - 1 } { \frac { ( y _ { i } - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } } \right)\tag{A.89}
$$

$$
\begin{array} { c } { { \Rightarrow \ c _ { 1 } ( \theta ) = \log \displaystyle \frac { \ell ( \theta ; Y ) } { \ell ( \theta ; Y ^ { \prime } ) } = \displaystyle \frac { ( y _ { n } ^ { \prime } - \theta ) ^ { 2 } - ( y _ { n } - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } } } \\ { { \qquad = \displaystyle \frac { ( y _ { n } ^ { \prime 2 } - y _ { n } ^ { 2 } ) - 2 \theta ( y _ { n } ^ { \prime } - y _ { n } ) } { 2 \sigma ^ { 2 } } \triangleq K + \displaystyle \frac { \Delta y } { \sigma ^ { 2 } } \theta , } } \end{array}\tag{A.90}
$$

where $\begin{array} { r } { K = \frac { ( y _ { n } ^ { \prime 2 } - y _ { n } ^ { 2 } ) } { 2 \sigma ^ { 2 } } } \end{array}$ and $\Delta y = y _ { n } - y _ { n } ^ { \prime }$ . Per the affine invariance of Gaussian distributions

$$
c _ { 1 } ( \theta ) \sim \mathcal { N } \left( K + \frac { \Delta y } { \sigma ^ { 2 } } \mu _ { Y } , \left( \frac { \Delta y } { \sigma ^ { 2 } } \right) ^ { 2 } v _ { Y } ^ { 2 } \right) .\tag{A.91}
$$

Since the moment generating function (MGF) for ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ is $\exp ( t \mu + \sigma ^ { 2 } t ^ { 2 } / 2 )$

$$
\begin{array} { r l } & { \log \mathbb { E } _ { \theta \sim \pi ( \theta \mid Y ) } \left( e ^ { t ( c _ { 1 } ( \theta ) - \mathbb { E } _ { \theta \sim \pi ( \theta \mid Y ) } [ c _ { 1 } ( \theta ) ] ) } \right) } \\ & { = \frac { t ^ { 2 } \left( \Delta y / \sigma ^ { 2 } \right) ^ { 2 } v _ { Y } ^ { 2 } } { 2 } = \frac { t ^ { 2 } } { 2 } \left( \frac { ( y _ { n } - y _ { n } ^ { \prime } ) ^ { 2 } } { \sigma ^ { 4 } } \left( \frac { n } { \sigma ^ { 2 } } + \frac { 1 } { \tau ^ { 2 } } \right) ^ { - 1 } \right) \triangleq \frac { t ^ { 2 } \sigma _ { Y , Y ^ { \prime } } ^ { 2 } } { 2 } . } \end{array}\tag{A.92}
$$

Then we conclude that $c _ { 1 } ( \theta )$ is sub-Gaussian with $\begin{array} { r } { \sigma _ { Y , Y ^ { \prime } } ^ { 2 } = \frac { ( y _ { n } - y _ { n } ^ { \prime } ) ^ { 2 } } { \sigma ^ { 4 } } \left( \frac { n } { \sigma ^ { 2 } } + \frac { 1 } { \tau ^ { 2 } } \right) ^ { - 1 } } \end{array}$

To obtain $\begin{array} { r } { \bar { \sigma } ^ { 2 } = \operatorname* { s u p } _ { ( Y , Y ^ { \prime } ) : d ( Y , Y ^ { \prime } ) = 1 } \sigma _ { Y , Y ^ { \prime } } ^ { 2 } } \end{array}$ , we need to assume a bounded data domain, i.e. $\exists c \in$ $\mathbb { R } , k > 0$ such that $y \in [ c - k \sigma , c + k \sigma ]$ for every record, then $| y _ { n } - y _ { n } ^ { \prime } | \leq 2 k \sigma$ , then

$$
\sigma _ { Y , Y ^ { \prime } } ^ { 2 } \leq \frac { ( 2 k \sigma ) ^ { 2 } } { \sigma ^ { 4 } } \left( \frac { n } { \sigma ^ { 2 } } + \frac { 1 } { \tau ^ { 2 } } \right) ^ { - 1 } = \frac { 4 k ^ { 2 } } { n + \frac { \sigma ^ { 2 } } { \tau ^ { 2 } } } = \bar { \sigma } ^ { 2 } .\tag{A.93}
$$

For the reversed pair $( Y ^ { \prime } , Y ) , \widetilde { c } _ { 1 } ( \theta ) = - c _ { 1 } ( \theta )$ is Gaussian under $\pi ( \theta | Y ^ { \prime } )$ with variance $( y _ { n } \mathrm { ~ - ~ }$ $y _ { n } ^ { \prime } ) ^ { 2 } v _ { Y ^ { \prime } } ^ { 2 } / \sigma ^ { 4 } ;$ since $Y ^ { \prime }$ also has n records, $v _ { Y ^ { \prime } } ^ { 2 } = v _ { Y } ^ { 2 }$ , so the same bound holds for $\sigma _ { Y ^ { \prime } , Y } ^ { 2 }$ . Plug Eq. (A.93) into Corollary 8,

$$
\operatorname* { m a x } \{ D _ { \alpha } ( \pi ( \theta | Y ) \| \pi ( \theta | Y ^ { \prime } ) ) , D _ { \alpha } ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) ) \} \leq \frac { \alpha { \bar { \sigma } } ^ { 2 } } { 2 } \leq \frac { 2 k ^ { 2 } \alpha } { n + \frac { \sigma ^ { 2 } } { \tau ^ { 2 } } } .\tag{A.94}
$$

## A.9.2 unbounded neighboring

Forward direction Let Y has size n and $Y ^ { \prime }$ is of size $n - 1$ by removing the n-th record $y _ { n }$ in $Y$ Then

$$
c _ { 1 } ( \theta ) = \log \frac { \ell ( \theta ; Y ) } { \ell ( \theta ; Y ^ { \prime } ) } = \frac { - ( y _ { n } - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } - \frac 1 2 \log ( 2 \pi \sigma ^ { 2 } ) \triangleq \frac { - Z ^ { 2 } } { 2 \sigma ^ { 2 } } - \frac 1 2 \log ( 2 \pi \sigma ^ { 2 } ) .\tag{A.95}
$$

where $Z = \theta - y _ { n }$ . Under the posterior $\theta | Y \sim \mathcal { N } ( \mu _ { Y } , v _ { Y } ^ { 2 } )$ , then $Z | Y \sim \mathcal { N } ( \mu _ { Y } - y _ { n } , v _ { Y } ^ { 2 } )$ , and $Z ^ { 2 } / v _ { Y } ^ { 2 }$ follows a noncentral chi-square distribution with degree of freedom 1, that is

$$
\frac { Z ^ { 2 } } { v _ { Y } ^ { 2 } } \sim \chi _ { k } ^ { 2 } ( \lambda ) , \quad \mathrm { w i t h } \ k = 1 , \lambda = \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { v _ { Y } ^ { 2 } } .\tag{A.96}
$$

We note that $c _ { 1 } ( \theta )$ in Eq. (A.95) doesn’t fall into the sub-Gaussian regime because $c _ { 1 } ( \theta )$ is a negative quadratic function of a Gaussian random variable, so its exponential moment $\mathbb { E } _ { P } [ \dot { e } ^ { t c _ { 1 } ( \theta ) } ]$ is finite only for |t| in a bounded neighborhood of zero (due to the noncentral $\chi ^ { 2 }$ tail of $Z ^ { 2 } )$ , rather than for all $t \in \mathbb { R }$ as required by the sub-Gaussian condition. So there is no finite $\sigma _ { Y , Y ^ { \prime } } ^ { 2 }$

But we can still leverage properties of $\chi ^ { 2 }$ distribution to explicitly calculate each term in Eq. (7). First, the expectation of $Z ^ { \bar { 2 } }$ is

$$
\mathbb { E } _ { \theta \sim \pi ( \theta | Y ) } \left[ Z ^ { 2 } \right] = v _ { Y } ^ { 2 } ( k + \lambda ) = v _ { Y } ^ { 2 } \left( 1 + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { v _ { Y } ^ { 2 } } \right) = v _ { Y } ^ { 2 } + ( \mu _ { Y } - y _ { n } ) ^ { 2 } .\tag{A.97}
$$

Then, recall the moment generating function of $\chi _ { k } ^ { 2 } ( \lambda )$ is $e ^ { \lambda t / ( 1 - 2 t ) } ( 1 - 2 t ) ^ { - k / 2 }$ for $t < 1 / 2$ , hence

$$
\begin{array} { r l r } {  { \mathbb { E } _ { \theta \sim \pi ( \theta | Y ) } [ e ^ { t Z ^ { 2 } / v _ { Y } ^ { 2 } } ] = \exp ( \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { v _ { Y } ^ { 2 } } \cdot \frac { t } { 1 - 2 t } ) ( 1 - 2 t ) ^ { - 1 / 2 } } } \\ & { \Rightarrow \mathbb { E } _ { \theta \sim \pi ( \theta | Y ) } [ e ^ { t Z ^ { 2 } } ] = \exp ( \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { v _ { Y } ^ { 2 } } \cdot \frac { v _ { Y } ^ { 2 } t } { 1 - 2 v _ { Y } ^ { 2 } t } ) ( 1 - 2 t v _ { Y } ^ { 2 } ) ^ { - 1 / 2 } } \\ & { } & { = ( 1 - 2 v _ { Y } ^ { 2 } t ) ^ { - 1 / 2 } \exp ( \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } t } { 1 - 2 v _ { Y } ^ { 2 } t } ) , \qquad t < \frac { 1 } { 2 v _ { Y } ^ { 2 } } . } \end{array}\tag{A.98}
$$

Now we have two approaches to explicitly upper bound $D _ { \alpha } ( \pi ( \theta | Y ) \| \pi ( \theta | Y ^ { \prime } ) )$

1) Derive sub-exponential parameters $b _ { Y } , \nu _ { Y }$ for $c _ { 1 } ( \theta )$ and apply Corollary 13. Since the derivation of $b _ { Y } , \nu _ { Y }$ is not tight, the resulting RDP bound is generally not tight;

2) Directly calculate each term in Eq. (A.6) using the closed-form moment generating function in Eq. (A.98). This approach yields an exact expression.

Method 1: sub-exponential parameters.<sup>2</sup> Our target is to identify $b _ { Y } , \nu _ { Y } ^ { 2 }$ such that

$$
\log \mathbb { E } _ { \theta \sim P } \left( e ^ { t \left( c _ { 1 } ( \theta ) - \mathbb { E } _ { P } \left[ c _ { 1 } ( \theta ) \right] \right) } \right) \leq \frac { t ^ { 2 } \nu _ { Y } ^ { 2 } } { 2 ( 1 - b _ { Y } | t | ) }\tag{A.99}
$$

for $| t | < 1 / b _ { Y }$ , with $c _ { 1 } ( \theta )$ defined in Eq. (A.95).

$$
\begin{array} { l } { { \displaystyle c _ { 1 } ( \theta ) - \mathbb { E } _ { P } \left[ c _ { 1 } ( \theta ) \right] = \frac { - Z ^ { 2 } } { 2 \sigma ^ { 2 } } - \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) - \mathbb { E } _ { P } \left( \frac { - Z ^ { 2 } } { 2 \sigma ^ { 2 } } - \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) \right) } } \\ { { \displaystyle \qquad = - \frac { 1 } { 2 \sigma ^ { 2 } } \left( Z ^ { 2 } - \mathbb { E } _ { P } ( Z ^ { 2 } ) \right) } } \end{array}\tag{A.100}
$$

So we only need to examine $\bigl ( Z ^ { 2 } - \mathbb { E } _ { P } ( Z ^ { 2 } ) \bigr )$

$$
\begin{array} { r l } & { \log \mathbb { E } _ { P } \left[ e ^ { t ( \mathbb { Z } ^ { 2 } - \mathbb { E } _ { P } ( \mathbb { Z } ^ { 2 } ) ) } \right] } \\ & { = \log \mathbb { E } _ { P } \left[ e ^ { t ( \mathbb { Z } ^ { 2 } - v _ { Y } ^ { 2 } - ( \mu _ { Y } - y _ { n } ) ^ { 2 } ) } \right] \qquad \mathrm { p e r ~ E q . ~ ( A . 9 7 ) } } \\ & { = \log \mathbb { E } _ { P } \left[ e ^ { t \mathbb { Z } ^ { 2 } } \right] - \left( v _ { Y } ^ { 2 } + ( \mu _ { Y } - y _ { n } ) ^ { 2 } \right) t } \\ & { = - \displaystyle \frac { 1 } { 2 } \log ( 1 - 2 v _ { Y } ^ { 2 } t ) - v _ { Y } ^ { 2 } t + ( \mu _ { Y } - y _ { n } ) ^ { 2 } t \left( \frac { 1 } { 1 - 2 v _ { Y } ^ { 2 } t } - 1 \right) \qquad \mathrm { p e r ~ E q . ~ ( A . 9 8 ) } } \\ & { = \displaystyle \frac { 1 } { 2 } \left[ - \log ( 1 - x ) - x \right] + ( \mu _ { Y } - y _ { n } ) ^ { 2 } t \left( \frac { 1 } { 1 - x } - 1 \right) \qquad \mathrm { l e t } x = 2 v _ { Y } ^ { 2 } t . } \end{array}\tag{A.101}
$$

Since Eq. (A.98) requires $t < 1 / ( 2 v _ { Y } ^ { 2 } ) , x = 2 v _ { Y } ^ { 2 } t < 1$ . To ensure Eq. $\begin{array} { r } { ( \mathrm { A } . 1 0 1 ) { \le } \frac { t ^ { 2 } \nu _ { Y } ^ { \prime 2 } } { 2 ( 1 - b _ { Y } ^ { \prime } | t | ) } } \end{array}$ for some $b _ { Y } ^ { \prime } , \nu _ { Y } ^ { \prime }$ when $| t | < 1 / b _ { Y } ^ { \prime }$ , we need some inequalities below for $| x | < 1$

$$
- \log ( 1 - x ) - x \leq { \frac { x ^ { 2 } } { 2 ( 1 - | x | ) } } , \qquad { \frac { 1 } { 1 - x } } - 1 = { \frac { x } { 1 - x } } \leq { \frac { | x | } { 1 - | x | } } .\tag{A.102}
$$

Plug the inequalities above into Eq. (A.101):

$$
\begin{array} { l } { \displaystyle \mathrm { E q . ~ } ( \mathrm { A . } 1 0 1 ) \leq \frac { 1 } { 2 } \cdot \frac { x ^ { 2 } } { 2 ( 1 - | x | ) } + ( \mu _ { Y } - y _ { n } ) ^ { 2 } | t | \cdot \frac { | x | } { 1 - | x | } } \\ { \displaystyle = \frac { 1 } { 2 } \cdot \frac { 4 v _ { Y } ^ { 4 } t ^ { 2 } } { 2 ( 1 - 2 v _ { Y } ^ { 2 } | t | ) } + ( \mu _ { Y } - y _ { n } ) ^ { 2 } | t | \cdot \frac { 2 v _ { Y } ^ { 2 } | t | } { 1 - 2 v _ { Y } ^ { 2 } | t | } } \\ { \displaystyle = \frac { v _ { Y } ^ { 4 } t ^ { 2 } } { 1 - 2 v _ { Y } ^ { 2 } | t | } + \frac { 2 v _ { Y } ^ { 2 } ( \mu _ { Y } - y _ { n } ) ^ { 2 } t ^ { 2 } } { 1 - 2 v _ { Y } ^ { 2 } | t | } } \\ { \displaystyle = \frac { 2 \left( v _ { Y } ^ { 2 } + 2 ( \mu _ { Y } - y _ { n } ) ^ { 2 } \right) v _ { Y } ^ { 2 } t ^ { 2 } } { 2 ( 1 - 2 v _ { Y } ^ { 2 } | t | ) } \triangleq \frac { t ^ { 2 } \nu _ { Y } ^ { \prime 2 } } { 2 ( 1 - b _ { Y } ^ { \prime } | t | ) } . } \end{array}\tag{A.103}
$$

Therefore $b _ { Y } ^ { \prime } = 2 v _ { Y } ^ { 2 }$ and $\nu _ { Y } ^ { \prime 2 } = 2 \left( v _ { Y } ^ { 2 } + 2 ( \mu _ { Y } - y _ { n } ) ^ { 2 } \right) v _ { Y } ^ { 2 }$

Recall that $b _ { Y } ^ { \prime } , \nu _ { Y } ^ { \prime 2 }$ are for $\begin{array} { r l r } {  { \big ( Z ^ { 2 } - \mathbb { E } _ { P } \big ( Z ^ { 2 } \big ) \big ) } } \end{array}$ ), while $\begin{array} { r } { c _ { 1 } ( \theta ) - \mathbb { E } _ { P } \left[ c _ { 1 } ( \theta ) \right] = - \frac { 1 } { 2 \sigma ^ { 2 } } \left( Z ^ { 2 } - \mathbb { E } _ { P } \left( Z ^ { 2 } \right) \right) } \end{array}$ , hence $b _ { Y } = b _ { Y } ^ { \prime } / 2 \sigma ^ { 2 }$ and $\nu _ { Y } ^ { 2 } = \nu _ { Y } ^ { \prime 2 } / 4 \sigma ^ { 4 }$ . Plug in Corollary 13:

$$
\begin{array} { r l } & { D _ { \alpha } ( P | | Q ) \leq \frac { \nu _ { Y } ^ { 2 } } { 2 } \frac { \alpha ( 1 - 2 b _ { Y } ) + 2 b _ { Y } } { ( 1 - b _ { Y } ) \left( 1 - b _ { Y } ( \alpha - 1 ) \right) } } \\ & { \qquad = \frac { \nu _ { Y } ^ { 2 } \left( v _ { Y } ^ { 2 } + 2 ( \mu _ { Y } - y _ { n } ) ^ { 2 } \right) } { 4 \sigma ^ { 4 } } \frac { \alpha \left( 1 - \frac { 2 v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } \right) + \frac { 2 v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } } { \left( 1 - \frac { v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } \right) \left( 1 - \frac { v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } ( \alpha - 1 ) \right) } } \\ & { \qquad = \frac { v _ { Y } ^ { 2 } \left( v _ { Y } ^ { 2 } + 2 ( \mu _ { Y } - y _ { n } ) ^ { 2 } \right) } { 4 \sigma ^ { 2 } } \cdot \frac { \alpha \sigma ^ { 2 } - 2 v _ { Y } ^ { 2 } \left( \alpha - 1 \right) } { \left( \sigma ^ { 2 } - v _ { Y } ^ { 2 } \right) \left( \sigma ^ { 2 } - v _ { Y } ^ { 2 } \left( \alpha - 1 \right) \right) } , } \end{array}\tag{A.104}
$$

where $v _ { Y } ^ { 2 }$ and $\mu _ { Y }$ are defined in Eq. (A.87).

Method 2: exact moment calculation For the first term in $\operatorname { E q . } \ ( \mathbf { A . } 6 )$ , per the $c _ { 1 } ( \theta )$ defined in Eq. (A.95), and the moment generating function of $Z ^ { 2 }$ derived in Eq. (A.98)

$$
\begin{array} { r l r } {  { - c _ { 1 } ( \theta ) = \frac { Z ^ { 2 } } { 2 \sigma ^ { 2 } } + \frac 1 2 \log ( 2 \pi \sigma ^ { 2 } ) } } \\ & { } & { \ \mathbb { E } _ { P } [ e ^ { - c _ { 1 } ( \theta ) } ] = \mathbb { E } _ { P } [ \exp ( \frac { Z ^ { 2 } } { 2 \sigma ^ { 2 } } + \frac 1 2 \log ( 2 \pi \sigma ^ { 2 } ) ) ] } \\ & { } & { = ( 2 \pi \sigma ^ { 2 } ) ^ { \frac 1 2 } \mathbb { E } _ { P } [ \exp ( \frac { Z ^ { 2 } } { 2 \sigma ^ { 2 } } ) ] } \\ & { } & { = ( 2 \pi \sigma ^ { 2 } ) ^ { \frac 1 2 } ( 1 - 2 v _ { Y } ^ { 2 } \frac { 1 } { 2 \sigma ^ { 2 } } ) ^ { - \frac 1 2 } \exp ( \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } \frac { 1 } { 2 \sigma ^ { 2 } } } { 1 - 2 v _ { Y } ^ { 2 } \frac { 1 } { 2 \sigma ^ { 2 } } } ) \mathrm { p l u g } t = \frac { 1 } { 2 \sigma ^ { 2 } } \mathrm { i n E q . } \mathrm { ( A . 9 8 ) } } \end{array}
$$

$$
= ( 2 \pi \sigma ^ { 2 } ) ^ { \frac { 1 } { 2 } } \left( 1 - \frac { v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - \frac { 1 } { 2 } } \exp \left( \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - v _ { Y } ^ { 2 } ) } \right) .\tag{A.105}
$$

This requires $\begin{array} { r } { t = \frac { 1 } { 2 \sigma ^ { 2 } } < \frac { 1 } { 2 v _ { \mathrm { v } } ^ { 2 } } } \end{array}$ , equivalently $v _ { Y } ^ { 2 } < \sigma ^ { 2 }$ , recall $\begin{array} { r } { v _ { Y } ^ { 2 } = \left( \frac { n } { \sigma ^ { 2 } } + \frac { 1 } { \tau ^ { 2 } } \right) ^ { - 1 } } \end{array}$ , this holds for $n \geq 1$ in the Gaussian-Gaussian conjugate model. Then per Lemma 2

$$
c _ { 2 } = \log \mathbb { E } _ { P } { \big [ } e ^ { - c _ { 1 } ( \theta ) } { \big ] } = { \frac { 1 } { 2 } } \log ( 2 \pi \sigma ^ { 2 } ) - { \frac { 1 } { 2 } } \log \left( 1 - { \frac { v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } } \right) + { \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - v _ { Y } ^ { 2 } ) } } .\tag{A.106}
$$

Next, for the second term in Eq. (A.6), we also use the moment generating function for $Z ^ { 2 }$ in Eq. (A.98) to exactly calculate the (α − 1) moment for $c _ { 1 } ( \theta )$

$$
\begin{array} { r l } { ( \alpha - 1 ) c _ { 1 } ( \theta ) = - \frac { \alpha - 1 } { 2 \sigma ^ { 2 } } Z ^ { 2 } - \frac { \alpha - 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) , } & { } \\ { \mathbb { E } _ { P } [ e ^ { ( \alpha - 1 ) c _ { 1 } ( \theta ) } ] = \mathbb { E } _ { P } \left[ \exp \left( - \frac { \alpha - 1 } { 2 \sigma ^ { 2 } } Z ^ { 2 } - \frac { \alpha - 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) \right) \right] } & { } \\ { = ( 2 \pi \sigma ^ { 2 } ) ^ { - \frac { \alpha - 1 } { 2 } } \mathbb { E } _ { P } \left[ \exp \left( - \frac { \alpha - 1 } { 2 \sigma ^ { 2 } } Z ^ { 2 } \right) \right] \ \mathrm { p l u g } \ t = \frac { - ( \alpha - 1 ) } { 2 \sigma ^ { 2 } } \ \mathrm { i n } \ \mathbf { E } \mathbf { q } _ { \cdot } ( \Lambda . 9 8 ) } & { } \\ { = ( 2 \pi \sigma ^ { 2 } ) ^ { - \frac { \alpha - 1 } { 2 } } \left( 1 - 2 v _ { Y } ^ { 2 } \left( \frac { 1 - \alpha } { 2 \sigma ^ { 2 } } \right) \right) ^ { - \frac { 1 } { 2 } } \exp \left( \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } \left( \frac { 1 - \alpha } { 2 \sigma ^ { 2 } } \right) } { 1 - 2 v _ { Y } ^ { 2 } \left( \frac { 1 - \alpha } { 2 \sigma ^ { 2 } } \right) } \right) } & { } \\  = ( 2 \pi \sigma ^ { 2 } ) ^ { - \frac { \alpha - 1 } { 2 } } \left( 1 + v _ { Y } ^ { 2 } \left( \frac { \alpha - 1 } { \sigma ^ { 2 } } \right) \right) ^ { - \frac { 1 } { 2 } } \exp \left( \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } ( 1 - \alpha ) } { 2 ( \sigma ^ { 2 } + ( \alpha - 1 ) v _ { Y } ^ { 2 } ) } \right) \ . \hfill \mathrm { ( A . } \end{array}\tag{107}
$$

This requires $\begin{array} { r } { t = \frac { - ( \alpha - 1 ) } { 2 \sigma ^ { 2 } } < \frac { 1 } { 2 v _ { \mathrm { v } } ^ { 2 } } } \end{array}$ . Since $\begin{array} { r } { \alpha > 1 , \frac { - ( \alpha - 1 ) } { 2 \sigma ^ { 2 } } < 0 } \end{array}$ , so this requirement always holds. Next, the 2nd term in $\operatorname { E q . } \left( \mathbf { A . 6 } \right)$ can be calculated as

$$
\begin{array} { l } { { \displaystyle \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { P } \left[ e ^ { ( \alpha - 1 ) c _ { 1 } ( \theta ) } \right] } \ ~ } \\ { = \ - \ \displaystyle \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) - \displaystyle \frac { 1 } { 2 ( \alpha - 1 ) } \log \left( 1 + { v } _ { Y } ^ { 2 } \left( \displaystyle \frac { \alpha - 1 } { \sigma ^ { 2 } } \right) \right) - \displaystyle \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } + ( \alpha - 1 ) v _ { Y } ^ { 2 } ) } . } \end{array}\tag{A.108}
$$

Plug $c _ { 2 }$ in Eq. (A.106) and $\begin{array} { r } { \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { P } \left[ e ^ { ( \alpha - 1 ) \cdot c _ { 1 } ( \theta ) } \right] } \end{array}$ in Eq. (A.108) into Eq. (A.6), we have

$$
\begin{array} { l } { { \displaystyle { D _ { \alpha } ( \pi ( \theta | Y ) | | \pi ( \theta | Y ^ { \prime } ) ) } } } \\ { { \displaystyle = c _ { 2 } + \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { P } \left[ e ^ { ( \alpha - 1 ) \cdot c _ { 1 } ( \theta ) } \right] } } \\ { { \displaystyle = \frac { 1 } { \frac { 2 } { \pi } \log ( 2 \pi \widehat { \sigma } ^ { 2 } ) } - \frac { 1 } { 2 } \log \left( 1 - \frac { v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } \right) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - v _ { Y } ^ { 2 } ) } } } \\ { { \displaystyle ~ - \frac { 1 } { 2 } \log ( 2 \widehat { \pi \sigma ^ { 2 } } ) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \left( 1 + v _ { Y } ^ { 2 } \left( \frac { \alpha - 1 } { \sigma ^ { 2 } } \right) \right) - \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } + ( \alpha - 1 ) v _ { Y } ^ { 2 } ) } } } \\ { { \displaystyle = ~ - \frac { 1 } { 2 } \log \left( 1 - \frac { v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } \right) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \left( 1 + v _ { Y } ^ { 2 } \left( \frac { \alpha - 1 } { \sigma ^ { 2 } } \right) \right) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - v _ { Y } ^ { 2 } ) } - \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } + ( \alpha - 1 ) v _ { Y } ^ { 2 } ) } } . } \end{array}
$$

Recall

$$
v _ { Y } ^ { 2 } = \left( { \frac { n } { \sigma ^ { 2 } } } + { \frac { 1 } { \tau ^ { 2 } } } \right) ^ { - 1 } , \qquad \mu _ { Y } = v _ { Y } ^ { 2 } \left( { \frac { 1 } { \sigma ^ { 2 } } } \sum _ { i = 1 } ^ { n } y _ { i } + { \frac { \mu _ { 0 } } { \tau ^ { 2 } } } \right) .\tag{A.109}
$$

Then define $\begin{array} { r } { \omega = n + \frac { \sigma ^ { 2 } } { \tau ^ { 2 } } } \end{array}$ and assume $y _ { n } \in \left[ \mu _ { Y } - k ^ { * } \sigma , \mu _ { Y } + k ^ { * } \sigma \right]$

$$
\begin{array} { c l } { { } } & { { \displaystyle D _ { \alpha } ( \pi ( \theta | Y ) | | \pi ( \theta | Y ^ { \prime } ) ) } } \\ { { } } & { { = \displaystyle - \frac { 1 } { 2 } \log \left( 1 - \frac { v _ { Y } ^ { 2 } } { \sigma ^ { 2 } } \right) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \left( 1 + v _ { Y } ^ { 2 } \left( \frac { \alpha - 1 } { \sigma ^ { 2 } } \right) \right) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - v _ { Y } ^ { 2 } ) } - \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } + ( \alpha - 1 ) v _ { Y } ^ { 2 } ) } } } \\ { { } } & { { = \displaystyle - \frac { 1 } { 2 } \log \left( 1 - \omega ^ { - 1 } \right) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \left( 1 + \frac { \alpha - 1 } { \omega } \right) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \left[ \frac { 1 } { 1 - \omega ^ { - 1 } } - \frac { 1 } { 1 + \frac { \alpha - 1 } { \omega } } \right] } } \end{array}
$$

$$
\begin{array} { l } { \displaystyle \triangleq A ( \alpha , \omega ) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } B ( \alpha , \omega ) } \\ { \displaystyle \le A ( \alpha , \omega ) + \frac { { k ^ { * } } ^ { 2 } } { 2 } B ( \alpha , \omega ) } \end{array}\tag{A.110}
$$

Since $\begin{array} { r } { \mu _ { Y } = \left( \frac { \sigma ^ { 2 } } { \tau ^ { 2 } } \mu _ { 0 } + \sum _ { i = 1 } ^ { n } y _ { i } \right) / \omega } \end{array}$ is a convex combination of $\mu _ { 0 }$ and $\{ y _ { i } \}$ , if all records and $\mu _ { 0 }$ lie in $[ c - M , c + M ]$ , then $\left| \mu _ { Y } - y _ { n } \right| \leq 2 M , \mathrm { i . e . , } k ^ { \ast } = 2 M / \sigma$ is a data-independent choice.

Claim A.8. For all $\alpha > 1$ and $\omega > 1$ , we have $A ( \alpha , \omega ) > 0$ and $B ( \alpha , \omega ) > 0$ . Furthermore, holding αfixed, both $A ( \alpha , \omega )$ and $B ( \alpha , \omega )$ decrease monotonically as ω increases.

Proof. Assume $n \geq 1$ , then $\omega = n + \sigma ^ { 2 } / \tau ^ { 2 } > 1$ . Let $x = \omega ^ { - 1 }$ and $t = \alpha - 1$ , then $x \in ( 0 , 1 )$ and $t > 0$ , and

$$
A ( \alpha , \omega ) = - \frac { 1 } { 2 } \log { ( 1 - x ) } - \frac { 1 } { 2 t } \log { ( 1 + x t ) } \triangleq A ( x ) .\tag{A.111}
$$

Thus to prove $A ( \alpha , \omega ) > 0$ is equivalent to prove

$$
- \log { ( 1 - x ) } > \frac { 1 } { t } \log { ( 1 + x t ) } .\tag{A.112}
$$

Since $- \log ( 1 - x ) > x$ for $x \in ( 0 , 1 )$ , and log $\begin{array} { r } { ( 1 + x t ) < x t \Rightarrow t ^ { - 1 } \log ( 1 + x t ) < x , } \end{array}$

$$
- \log { ( 1 - x ) } > x > \frac { 1 } { t } \log { ( 1 + x t ) } ,\tag{A.113}
$$

which completes the proof of strictly positivity of $A ( \alpha , \omega )$ . Then, take derivative of x on $A ( x )$

$$
A ^ { \prime } ( x ) = \frac { 1 } { 2 ( 1 - x ) } - \frac { 1 } { 2 ( 1 + x t ) } > 0 ,\tag{A.114}
$$

per $x \in ( 0 , 1 )$ and $t > 0$ . Thus,

$$
\frac { \partial A } { \partial \omega } = A ^ { \prime } ( x ) \frac { d x } { d \omega } = - \omega ^ { 2 } A ^ { \prime } ( x ) < 0 .\tag{A.115}
$$

Furthermore, take derivative of x on $\begin{array} { r } { B ( x ) = \frac { 1 } { 1 - x } - \frac { 1 } { 1 + x t } } \end{array}$

$$
B ^ { \prime } ( x ) = \frac { 1 } { ( 1 - x ) ^ { 2 } } + \frac { t } { ( 1 + x t ) ^ { 2 } } > 0 \quad \Rightarrow \frac { \partial B } { \partial \omega } < 0\tag{A.116}
$$

thus, both $A ( \alpha , \omega )$ and $B ( \alpha , \omega )$ decrease monotonically as ω increases.

Reverse direction We next analyze the reverse Rényi divergence $D _ { \alpha } ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) )$ , where $Y$ has size n and $Y ^ { \prime }$ is of size $n - 1$ , obtained by removing the n-th record $y _ { n }$ from Y. Define

$$
\begin{array} { l } { { \displaystyle { \widetilde { c } } _ { 1 } ( \theta ) \triangleq \log { \displaystyle { \frac { \ell ( \theta ; Y ^ { \prime } ) } { \ell ( \theta ; Y ) } } } = - \log p ( y _ { n } | \theta ) = { \displaystyle { \frac { ( y _ { n } - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } } } + \frac 1 2 \log ( 2 \pi \sigma ^ { 2 } ) \triangleq \frac { Z ^ { \prime 2 } } { 2 \sigma ^ { 2 } } + \frac 1 2 \log ( 2 \pi \sigma ^ { 2 } ) } } \\ { { \displaystyle \quad \quad = - c _ { 1 } ( \theta ) , \qquad \quad \quad ( \Delta . \mathbb { Z } ) = \frac { \chi ^ { \prime } } { \sigma ^ { 2 } } } . } \end{array}\tag{17}
$$

where $Z ^ { \prime } = \theta - y _ { n }$ . Under the posterior $\theta | Y ^ { \prime } \sim \mathcal { N } ( \mu _ { Y ^ { \prime } } , v _ { Y ^ { \prime } } ^ { 2 } ) , Z ^ { \prime } | Y ^ { \prime } \sim \mathcal { N } ( \mu _ { Y ^ { \prime } } - y _ { n } , v _ { Y ^ { \prime } } ^ { 2 } )$ , and $Z ^ { \prime 2 } / v _ { Y } ^ { 2 }$ <sub>′</sub> follows a noncentral chi-square distribution with one degree of freedom:

$$
\frac { Z ^ { \prime 2 } } { v _ { Y ^ { \prime } } ^ { 2 } } \sim \chi _ { 1 } ^ { 2 } ( \lambda ^ { \prime } ) , \qquad \lambda ^ { \prime } = \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { v _ { Y ^ { \prime } } ^ { 2 } } .\tag{A.118}
$$

Recall that the MGF of $\chi _ { k } ^ { 2 } ( \lambda )$ is exp $\begin{array} { r } { \left( \frac { \lambda t } { 1 - 2 t } \right) ( 1 - 2 t ) ^ { - k / 2 } } \end{array}$ for $\begin{array} { r } { t < \frac { 1 } { 2 } } \end{array}$ . Therefore,

$$
\mathbb { E } _ { \theta \sim \pi ( \theta | Y ^ { \prime } ) } \left[ e ^ { t Z ^ { \prime 2 } } \right] = \left( 1 - 2 v _ { Y ^ { \prime } } ^ { 2 } t \right) ^ { - 1 / 2 } \exp \left( \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } t } { 1 - 2 v _ { Y ^ { \prime } } ^ { 2 } t } \right) , \qquad t < \frac { 1 } { 2 v _ { Y ^ { \prime } } ^ { 2 } } .\tag{A.119}
$$

We now directly calculate the two terms in $\operatorname { E q . } \left( \mathbf { A . 6 } \right)$ using the closed-form MGF in Eq. (A.119). For the first term, per Eq. $\begin{array} { r } { ( \mathrm { A } . 1 1 7 ) , - \widetilde { c } _ { 1 } ( \theta ) = - \frac { Z ^ { \prime 2 } } { 2 \sigma ^ { 2 } } - \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) } \end{array}$ , and hence

$$
\mathbb { E } _ { \theta \sim \pi ( \theta | Y ^ { \prime } ) } \big [ e ^ { - \widetilde { c } _ { 1 } ( \theta ) } \big ] = ( 2 \pi \sigma ^ { 2 } ) ^ { - 1 / 2 } \mathbb { E } _ { \theta \sim \pi ( \theta | Y ^ { \prime } ) } \Big [ \exp \Big ( - \frac { Z ^ { \prime 2 } } { 2 \sigma ^ { 2 } } \Big ) \Big ]
$$

$$
= ( 2 \pi \sigma ^ { 2 } ) ^ { - 1 / 2 } \Big ( 1 + \frac { v _ { Y ^ { \prime } } ^ { 2 } } { \sigma ^ { 2 } } \Big ) ^ { - 1 / 2 } \exp \Big ( - \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } + v _ { Y ^ { \prime } } ^ { 2 } ) } \Big ) ,\tag{A.120}
$$

where we plug $t = - 1 / ( 2 \sigma ^ { 2 } )$ into Eq. (A.119). Since $t < 0$ , the MGF condition always holds. Therefore, per Lemma 2,

$$
\widetilde { c } _ { 2 } = \log \mathbb { E } _ { \theta \sim \pi ( \theta | Y ^ { \prime } ) } \big [ e ^ { - \widetilde { c } _ { 1 } ( \theta ) } \big ] = - \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) - \frac { 1 } { 2 } \log \Big ( 1 + \frac { v _ { Y ^ { \prime } } ^ { 2 } } { \sigma ^ { 2 } } \Big ) - \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } + v _ { Y ^ { \prime } } ^ { 2 } ) } .\tag{A.121}
$$

Next, for the second term in Eq. (A.6), $\begin{array} { r } { ( \alpha - 1 ) \widetilde c _ { 1 } ( \theta ) = \frac { \alpha - 1 } { 2 \sigma ^ { 2 } } Z ^ { \prime 2 } + \frac { \alpha - 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) } \end{array}$ . Thus,

$$
\begin{array} { r l } & { \mathbb { E } _ { \theta \sim \pi ( \theta | Y ^ { \prime } ) } \big [ e ^ { ( \alpha - 1 ) \widetilde { c } _ { 1 } ( \theta ) } \big ] = ( 2 \pi \sigma ^ { 2 } ) ^ { ( \alpha - 1 ) / 2 } \mathbb { E } _ { \theta \sim \pi ( \theta | Y ^ { \prime } ) } \Big [ \exp \Big ( \frac { \alpha - 1 } { 2 \sigma ^ { 2 } } Z ^ { \prime 2 } \Big ) \Big ] } \\ & { \qquad = ( 2 \pi \sigma ^ { 2 } ) ^ { ( \alpha - 1 ) / 2 } \Big ( 1 - \frac { ( \alpha - 1 ) v _ { Y ^ { \prime } } ^ { 2 } } { \sigma ^ { 2 } } \Big ) ^ { - 1 / 2 } \exp \Big ( \frac { ( \alpha - 1 ) ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - ( \alpha - 1 ) v _ { Y ^ { \prime } } ^ { 2 } ) } \Big ) , } \end{array}\tag{A.122}
$$

where we plug $\textstyle t = { \frac { \alpha - 1 } { 2 \sigma ^ { 2 } } }$ into Eq. (A.119). This requires

$$
\frac { \alpha - 1 } { 2 \sigma ^ { 2 } } < \frac { 1 } { 2 v _ { Y ^ { \prime } } ^ { 2 } } , \mathrm { o r e q u i v a l e n t l y } ( \alpha - 1 ) v _ { Y ^ { \prime } } ^ { 2 } < \sigma ^ { 2 } .\tag{A.123}
$$

Consequently,

$$
\begin{array} { c } { { \displaystyle \frac 1 { \alpha - 1 } \log \mathbb E _ { \theta \sim \pi ( \theta \mid Y ^ { \prime } ) } \left[ e ^ { ( \alpha - 1 ) \tilde { c } _ { 1 } ( \theta ) } \right] = \displaystyle \frac 1 2 \log ( 2 \pi \sigma ^ { 2 } ) - \displaystyle \frac 1 { 2 ( \alpha - 1 ) } \log \left( 1 - \displaystyle \frac { ( \alpha - 1 ) v _ { Y ^ { \prime } } ^ { 2 } } { \sigma ^ { 2 } } \right) } } \\ { { + \displaystyle \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - ( \alpha - 1 ) v _ { Y ^ { \prime } } ^ { 2 } ) } . } } \end{array}\tag{A.124}
$$

Plugging Eqs. (A.121) and (A.124) into Eq. (A.6), the ${ \scriptstyle { \frac { 1 } { 2 } } } \log ( 2 \pi \sigma ^ { 2 } )$ terms cancel, and we obtain

$$
\begin{array} { l }  { \displaystyle { D _ { \alpha } ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) ) = - \frac { 1 } { 2 } \log \left( 1 + \frac { { v _ { Y ^ { \prime } } ^ { 2 } } } { \sigma ^ { 2 } } \right) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \left( 1 - \frac { ( \alpha - 1 ) v _ { Y ^ { \prime } } ^ { 2 } } { \sigma ^ { 2 } } \right) } } \\ { { + \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } - ( \alpha - 1 ) v _ { Y ^ { \prime } } ^ { 2 } ) } - \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { 2 ( \sigma ^ { 2 } + v _ { Y ^ { \prime } } ^ { 2 } ) } . } } \end{array}\tag{A.125}
$$

Recall that $\begin{array} { r } { v _ { Y ^ { \prime } } ^ { 2 } = \left( \frac { n - 1 } { \sigma ^ { 2 } } + \frac { 1 } { \tau ^ { 2 } } \right) ^ { - 1 } = \frac { \sigma ^ { 2 } } { \omega - 1 } } \end{array}$ with $\textstyle \omega = n + { \frac { \sigma ^ { 2 } } { \tau ^ { 2 } } }$ . Then the finiteness condition in Eq. (A.123) becomes

$$
\frac { \alpha - 1 } { \omega - 1 } < 1 \iff 1 < \alpha < \omega .\tag{A.126}
$$

For $1 < \alpha < \omega$ , Eq. (A.125) becomes

$$
\begin{array} { l } { \displaystyle { D _ { \alpha } \big ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) \big ) = - \frac { 1 } { 2 } \log \Big ( \frac { \omega } { \omega - 1 } \Big ) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \Big ( \frac { \omega - \alpha } { \omega - 1 } \Big ) } } \\ { \displaystyle { + \frac { ( \mu _ { Y ^ { \prime } } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \Big [ \frac { 1 } { 1 - \frac { \alpha - 1 } { \omega - 1 } } - \frac { 1 } { 1 + \frac { 1 } { \omega - 1 } } \Big ] . } } \end{array}\tag{A.127}
$$

The posterior means satisfy

$$
\omega \mu _ { Y } = ( \omega - 1 ) \mu _ { Y ^ { \prime } } + y _ { n } \Rightarrow \mu _ { Y ^ { \prime } } - y _ { n } = { \frac { \omega } { \omega - 1 } } ( \mu _ { Y } - y _ { n } ) .\tag{A.128}
$$

Therefore, for $1 < \alpha < \omega$

$$
\begin{array} { r l } & { D _ { \alpha } ( \pi ( \theta | Y ^ { \prime } ) | | \pi ( \theta | Y ) ) = - \displaystyle \frac { 1 } { 2 } \log \Big ( \displaystyle \frac { \omega } { \omega - 1 } \Big ) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \Big ( \displaystyle \frac { \omega - \alpha } { \omega - 1 } \Big ) + \frac { ( \mu _ { Y } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \frac { \alpha \omega } { ( \omega - 1 ) ( \omega - \alpha ) } } \\ & { \quad \quad \quad \triangleq \widetilde { A } ( \alpha , \omega ) + \frac { \left( \mu _ { Y } - y _ { n } \right) ^ { 2 } } { 2 \sigma ^ { 2 } } \widetilde { B } ( \alpha , \omega ) , \quad \quad \quad \quad ( \mathbf { A } . 1 2 9 ) } \end{array}
$$

where

$$
\widetilde { A } ( \alpha , \omega ) \triangleq - \frac { 1 } { 2 } \log \Big ( \frac { \omega } { \omega - 1 } \Big ) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \Big ( \frac { \omega - \alpha } { \omega - 1 } \Big ) ,
$$

$$
\widetilde { B } ( \alpha , \omega ) \triangleq \frac { \alpha \omega } { ( \omega - 1 ) ( \omega - \alpha ) } .\tag{A.130}
$$

For comparison, the corresponding quantities for the forward direction in Eq. (A.110) are

$$
\begin{array} { l } { \displaystyle { A ( \alpha , \omega ) = - \frac { 1 } { 2 } \log \Big ( \frac { \omega - 1 } { \omega } \Big ) - \frac { 1 } { 2 ( \alpha - 1 ) } \log \Big ( 1 + \frac { \alpha - 1 } { \omega } \Big ) , } } \\ { \displaystyle { B ( \alpha , \omega ) = \frac { \alpha \omega } { ( \omega - 1 ) ( \omega + \alpha - 1 ) } . } } \end{array}
$$

If $| \mu _ { Y } - y _ { n } | \leq k ^ { * } \sigma$ , then for every $1 < \alpha < \omega _ { : }$

$$
D _ { \alpha } ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) ) \le \widetilde { A } ( \alpha , \omega ) + \frac { k ^ { * 2 } } { 2 } \widetilde { B } ( \alpha , \omega ) .\tag{A.131}
$$

For $\alpha \geq \omega ,$ , the MGF in Eq. (A.122) is not finite, and therefore

$$
D _ { \alpha } ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) ) = \infty , \quad \alpha \geq \omega ; \qquad \mathrm { i n ~ p a r t i c u l a r } , D _ { \infty } ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) ) = \infty .\tag{A.132}
$$

Claim A.9. For all $1 < \alpha < \omega , \widetilde { A } ( \alpha , \omega ) > 0$ and $\widetilde { B } ( \alpha , \omega ) > 0 .$ . Holding α fixed, both decrease monotonically in ω on $( \alpha , \infty )$ , and $\tilde { A } ( \alpha , \omega )  \infty a s \omega \downarrow \alpha .$

Proof. Let $u = ( \omega - 1 ) ^ { - 1 } { \mathrm { a n d } } t = \alpha - 1 > 0$ . The condition $1 < \alpha < \omega$ is equivalent to $0 < t u < 1$ and $\omega / ( \omega - 1 ) = 1 + u , ( \omega - \alpha ) / ( \omega - 1 ) = 1 - t u$ . Hence $\begin{array} { r } { \widetilde { A } = - \frac { 1 } { 2 } \log ( 1 + u ) - \frac { 1 } { 2 t } \log ( 1 - t u ) } \end{array}$ and $\begin{array} { r } { \widetilde B = \frac { \alpha u ( 1 + u ) } { 1 - t u } } \end{array}$ . Since $- \log ( 1 - y ) > y$ for $y \in ( 0 , 1 )$ and $\log ( 1 + u ) < u$ for $u > 0$ , we have $\begin{array} { r } { - \frac { 1 } { t } \log ( 1 - t u ) > u > \log ( 1 + u ) , \mathrm { { s o } } \widetilde { A } > 0 ; \widetilde { B } > 0 } \end{array}$ is immediate. Further, $\begin{array} { r } { \frac { \partial \widetilde { A } } { \partial u } = \frac { 1 } { 2 } \big ( \frac { 1 } { 1 - t u } - \frac { 1 } { 1 + u } \big ) > } \end{array}$ $0 ,$ and $\widetilde { B }$ is increasing in u because its numerator increases and its positive denominator decreases in u. Since u is decreasing in $\omega ,$ both $\widetilde { A }$ and $\widetilde { B }$ decrease in ω. Finally, tu ↑ 1 as $\omega \downarrow \alpha ,$ , so $- \log ( 1 - t u ) \to \infty$ □

Two-sided guarantee. Combining Eqs. (A.110) and (A.131), if $| \mu _ { Y } - y _ { n } | \leq k ^ { * } \sigma$ , then for every $1 < \alpha < \omega _ { : }$

$$
\begin{array} { r l } & { \quad \operatorname* { m a x } \{ D _ { \alpha } ( \pi ( \theta | Y ) \| \pi ( \theta | Y ^ { \prime } ) ) , D _ { \alpha } ( \pi ( \theta | Y ^ { \prime } ) \| \pi ( \theta | Y ) ) \} } \\ & { \le \operatorname* { m a x } \Big \{ A ( \alpha , \omega ) + \displaystyle \frac { k ^ { * 2 } } { 2 } B ( \alpha , \omega ) , \ : \widetilde { A } ( \alpha , \omega ) + \displaystyle \frac { k ^ { * 2 } } { 2 } \widetilde { B } ( \alpha , \omega ) \Big \} , } \end{array}\tag{A.133}
$$

so OPS satisfies RDP of this order only for $1 < \alpha < \omega$ , and does not satisfy pure DP under unbounded neighboring. Note that $\widetilde { B } ( \alpha , \omega ) > B ( \alpha , \omega )$ since $\omega - \alpha < \omega + \alpha - 1 . \mathrm { A s } \omega = n + \sigma ^ { 2 } / \tau ^ { 2 }$ depends on $n ,$ a guarantee holding uniformly over all $n \geq 1$ is obtained by evaluating at $\omega _ { \mathrm { m i n } } \stackrel { \cdot } { = } 1 + \stackrel { \cdot } { \sigma ^ { 2 } } / \tau ^ { 2 }$ and is finite only for $1 < \alpha < 1 + \dot { \sigma } ^ { 2 } / \tau ^ { 2 }$ □

## A.10 Proof of Proposition 12 for Linear Regression

Proof. Let $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ be covariates and $y _ { i } \in \mathbb { R }$ be response for the i-th individual. Given the dataset $\mathbf { Y } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , let $\mathbf { x } \in \mathbb { R } ^ { n \times d }$ be the design matrix, and $\mathbf { y } \in \mathbb { R } ^ { n }$ be the response vector. The response follows a Gaussian distribution: $y _ { i } | \mathbf { x } _ { i } , \beta \sim \mathcal { N } ( \mathbf { x } _ { i } ^ { \top } \beta , \sigma ^ { 2 } )$ . Impose a Gaussian prior on regression coefficient $\beta , \mathrm { i . e . } \ \pi ( \beta ) = \mathcal { N } ( 0 , \tau ^ { 2 } \mathbf { I } _ { d } )$ , then the posterior is $\beta | { \bf \bar { Y } } = \mathcal { N } ( \mu _ { \bf Y } , \Sigma _ { \bf Y } )$ , where $\begin{array} { r } { \Sigma _ { \mathbf { Y } } = \Big ( \frac { 1 } { \sigma ^ { 2 } } \mathbf { X } ^ { \top } \mathbf { x } + \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \Big ) ^ { - 1 } , \mathrm { a n d } \mu _ { \mathbf { Y } } = \Sigma _ { \mathbf { Y } } \left( \frac { 1 } { \sigma ^ { 2 } } \mathbf { X } ^ { \top } \mathbf { y } \right) } \end{array}$

## A.10.1 bounded neighboring

Let Y and $\mathbf { Y } ^ { \prime }$ differ in exactly one record, i.e. ${ \bf z } _ { n } = ( { \bf x } _ { n } , y _ { n } )$ in Y is replaced by $\mathbf z _ { n } ^ { \prime } = ( \mathbf x _ { n } ^ { \prime } , y _ { n } ^ { \prime } )$ in $\mathbf { Y } ^ { \prime }$ , then

$$
\begin{array} { r } { c _ { 1 } ( \beta ) = \log \frac { \ell ( \beta ; \mathbf { Y } ) } { \ell ( \beta ; \mathbf { Y } ^ { \prime } ) } = \log p ( y _ { n } | \mathbf { x } _ { n } , \beta ) - \log p ( y _ { n } ^ { \prime } | \mathbf { x } _ { n } ^ { \prime } , \beta ) } \\ { = - \frac { ( y _ { n } - \mathbf { x } _ { n } ^ { \top } \beta ) ^ { 2 } - ( y _ { n } ^ { \prime } - \mathbf { x } _ { n } ^ { \prime \top } \beta ) ^ { 2 } } { 2 \sigma ^ { 2 } } . } \end{array}\tag{A.134}
$$

Case 1: $\operatorname { I f } \mathbf { x } _ { n } = \mathbf { x } _ { n } ^ { \prime }$ , then

$$
c _ { 1 } ( { \boldsymbol { \beta } } ) = - { \frac { ( y _ { n } ^ { 2 } - y _ { n } ^ { \prime 2 } ) - 2 ( y _ { n } - y _ { n } ^ { \prime } ) \mathbf { x } _ { n } ^ { \top } { \boldsymbol { \beta } } } { 2 \sigma ^ { 2 } } } .\tag{A.135}
$$

Since $\beta | \mathbf { Y } = \mathcal { N } ( \mu _ { \mathbf { Y } } , \Sigma _ { \mathbf { Y } } )$ and per the linear transformation of Gaussian random variable, we have $\mathbf { x } _ { n } ^ { \top } \beta \vert \mathbf { Y } \sim { \mathcal { N } } ( \mathbf { x } _ { n } ^ { \top } { \pmb { \mu } } _ { \mathbf { Y } } , \mathbf { x } _ { n } ^ { \top } { \pmb { \Sigma } } _ { \mathbf { Y } } \mathbf { x } _ { n } )$ . Therefore, $c _ { 1 } ( \beta )$ under $P = \pi ( \cdot | \mathbf { Y } )$ is Gaussian with variance

$$
\sigma _ { \mathbf { Y } , \mathbf { Y } ^ { \prime } } ^ { 2 } = \operatorname { V a r } _ { \beta \sim \pi ( \cdot | \mathbf { Y } ) } [ c _ { 1 } ( \beta ) ] = \frac { ( y _ { n } - y _ { n } ^ { \prime } ) ^ { 2 } } { \sigma ^ { 4 } } \mathbf { x } _ { n } ^ { \top } \boldsymbol { \Sigma } _ { \mathbf { Y } } \mathbf { x } _ { n } .\tag{A.136}
$$

Since $\mathbf { x } ^ { \top } \mathbf { x }$ is positive semi-definite, $\begin{array} { r } { \Sigma _ { \mathbf { Y } } = \left( \frac { 1 } { \sigma ^ { 2 } } \mathbf { x } ^ { \top } \mathbf { x } + \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \right) ^ { - 1 } \preceq \left( \frac { 1 } { \tau ^ { 2 } } \mathbf { I } _ { d } \right) ^ { - 1 } = \tau ^ { 2 } \mathbf { I } _ { d } . } \end{array}$ If additionally assume $| y _ { n } - y _ { n } ^ { \prime } | \leq 2 k \sigma$ , then

$$
\begin{array} { l } { { \displaystyle \sigma _ { { \bf Y } , { \bf Y } ^ { \prime } } ^ { 2 } = \frac { ( y _ { n } - y _ { n } ^ { \prime } ) ^ { 2 } } { \sigma ^ { 4 } } { \bf x } _ { n } ^ { \top } \Sigma _ { { \bf Y } } { \bf x } _ { n } } } \\ { { \displaystyle \quad \quad \leq \frac { 4 k ^ { 2 } \sigma ^ { 2 } } { \sigma ^ { 4 } } { \bf x } _ { n } ^ { \top } ( \tau ^ { 2 } { \bf I } _ { d } ) { \bf x } _ { n } = \frac { 4 k ^ { 2 } } { \sigma ^ { 2 } } \tau ^ { 2 } \| { \bf x } _ { n } \| ^ { 2 } \leq \frac { 4 k ^ { 2 } \tau ^ { 2 } R ^ { 2 } } { \sigma ^ { 2 } } \triangleq \bar { \sigma } ^ { 2 } } . }  \end{array}\tag{A.137}
$$

For the reversed pair $( \mathbf { Y } ^ { \prime } , \mathbf { Y } ) , \tilde { c } _ { 1 } ( \beta ) = - c _ { 1 } ( \beta )$ is Gaussian under $\pi ( \beta | \mathbf { Y } ^ { \prime } )$ with variance $( y _ { n } \mathrm { ~ - ~ }$ $y _ { n } ^ { \prime } ) ^ { 2 } { \bf x } _ { n } ^ { \top } \Sigma _ { { \bf Y ^ { \prime } } } { \bf x } _ { n } / \bar { \sigma ^ { 4 } } ;$ ; since $\Sigma _ { \mathbf { Y } ^ { \prime } } \preceq \tau ^ { 2 } \mathbf { I } _ { d } .$ the same bound holds for $\sigma _ { \mathbf { Y ^ { \prime } } , \mathbf { Y } } ^ { 2 }$ . Per Corollary 8,

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \frac { \alpha \bar { \sigma } ^ { 2 } } { 2 } \leq \frac { 2 \alpha k ^ { 2 } \tau ^ { 2 } R ^ { 2 } } { \sigma ^ { 2 } } .\tag{A.138}
$$

Case 2: When $\mathbf { x } _ { n } \neq \mathbf { x } _ { n } ^ { \prime }$ , the log-likelihood ratio decomposes as $c _ { 1 } ( \beta ) = L ( \beta ) + Q ( \beta )$ , where

$$
L ( \boldsymbol { \beta } ) = - \frac { ( y _ { n } ^ { 2 } - y _ { n } ^ { \prime 2 } ) - 2 ( y _ { n } \mathbf x _ { n } - y _ { n } ^ { \prime } \mathbf x _ { n } ^ { \prime } ) ^ { \top } \boldsymbol { \beta } } { 2 \sigma ^ { 2 } } ,\tag{A.139}
$$

$$
Q ( \beta ) = - \frac { \beta ^ { \top } A \beta } { 2 \sigma ^ { 2 } } , \quad A \triangleq \mathbf { x } _ { n } \mathbf { x } _ { n } ^ { \top } - \mathbf { x } _ { n } ^ { \prime } \mathbf { x } _ { n } ^ { \prime \top } .\tag{A.140}
$$

Since A may be indefinite, $Q ( \beta )$ is a linear combination of non-central $\chi ^ { 2 }$ random variables with potentially negative coefficients under $\beta | \mathbf { Y } \sim { \mathcal { N } } ( \mu _ { \mathbf { Y } } , \Sigma _ { \mathbf { Y } } ) , \operatorname { s o } c _ { 1 } ( \beta )$ is generally not sub-Gaussian. We show that it satisfies a sub-exponential MGF bound under additional boundedness assumptions, by bounding the MGF of each part separately and combining via Cauchy-Schwarz. Let $\mu _ { L } = \mathbb { E } [ L ( \beta ) ]$ $\mu _ { Q } = \mathbb { E } [ Q ( \beta ) ]$ ], and $\mu _ { \mathbf { Y } } ^ { c _ { 1 } } = \bar { \mu } _ { L } + \bar { \mu } _ { Q } { } ^ { 3 }$

Step 1: MGF of the linear part. Since $L ( \beta )$ is affine in $\beta$ and $\beta | \mathbf { Y } \sim \mathcal { N } ( \mu _ { \mathbf { Y } } , \Sigma _ { \mathbf { Y } } )$ , L(β) is Gaussian under the posterior. Using $\Sigma _ { \mathbf { Y } } \preceq \tau ^ { 2 } \mathbf { I } _ { d } , | y _ { i } | \leq M _ { y }$ , and $\left\| \mathbf { x } _ { i } \right\| \leq R :$

$$
\begin{array} { r } { \mathrm { V a r } [ L ( \beta ) ] = \frac { \left( y _ { n } { \mathbf { x } } _ { n } - y _ { n } ^ { \prime } { \mathbf { x } } _ { n } ^ { \prime } \right) ^ { \top } \Sigma _ { \mathbf { Y } } \left( y _ { n } { \mathbf { x } } _ { n } - y _ { n } ^ { \prime } { \mathbf { x } } _ { n } ^ { \prime } \right) } { \sigma ^ { 4 } } \leq \frac { \| y _ { n } { \mathbf { x } } _ { n } - y _ { n } ^ { \prime } { \mathbf { x } } _ { n } ^ { \prime } \| ^ { 2 } \tau ^ { 2 } } { \sigma ^ { 4 } } } \\ { \leq \frac { 4 M _ { y } ^ { 2 } R ^ { 2 } \tau ^ { 2 } } { \sigma ^ { 4 } } \triangleq \bar { \sigma } _ { L } ^ { 2 } . } \end{array}\tag{A.141}
$$

Therefore $L ( \beta )$ is sub-Gaussian with variance proxy $\bar { \sigma } _ { L } ^ { 2 }$ and for all $t \in \mathbb { R }$

$$
\log \mathbb { E } \left[ e ^ { t ( L ( \beta ) - \mu _ { L } ) } \right] \leq \frac { t ^ { 2 } \bar { \sigma } _ { L } ^ { 2 } } { 2 } .\tag{A.142}
$$

Step 2: Centered MGF of the quadratic part. We compute $\mu _ { Q }$ exactly. For $\beta \sim \mathcal { N } ( \mu _ { \mathrm { Y } } , \Sigma _ { \mathrm { Y } } ) ;$

$$
\mu _ { Q } = - \frac { 1 } { 2 \sigma ^ { 2 } } \mathbb { E } [ \beta ^ { \top } A \beta ] = - \frac { 1 } { 2 \sigma ^ { 2 } } \left( \mathrm { t r } ( A \Sigma _ { \mathbf { Y } } ) + \mu _ { \mathbf { Y } } ^ { \top } A \mu _ { \mathbf { Y } } \right) .\tag{A.143}
$$

Define the symmetrized matrix $\tilde { A } \triangleq \Sigma _ { \mathbf { Y } } ^ { 1 / 2 } A \Sigma _ { \mathbf { Y } } ^ { 1 / 2 }$ . By Sylvester’s determinant identity:

$$
\operatorname* { d e t } \left( \mathbf { I } + { \frac { t } { \sigma ^ { 2 } } } \Sigma _ { \mathbf { Y } } A \right) = \operatorname* { d e t } \left( \mathbf { I } + { \frac { t } { \sigma ^ { 2 } } } \Sigma _ { \mathbf { Y } } ^ { 1 / 2 } A \Sigma _ { \mathbf { Y } } ^ { 1 / 2 } \right) = \operatorname* { d e t } \left( \mathbf { I } + { \frac { t } { \sigma ^ { 2 } } } { \tilde { A } } \right) ,\tag{A.144}
$$

so the positivity condition $\begin{array} { r } { \mathbf { I } + \frac { t } { \sigma ^ { 2 } } \tilde { \boldsymbol { A } } \succ 0 } \end{array}$ controls both the determinant and the existence of the MGF, while the inverse term is retained in the nonsymmetric form $\begin{array} { r } { ( \mathbf { I } + \frac { t } { \sigma ^ { 2 } } \Sigma _ { \mathbf { Y } } A ) ^ { - 1 } } \end{array}$ , which is algebraically equivalent. Since $\tilde { A }$ is symmetric:

$$
\begin{array} { r } { \| \tilde { A } \| _ { \mathrm { o p } } = \| \Sigma _ { \mathbf { Y } } ^ { 1 / 2 } A \Sigma _ { \mathbf { Y } } ^ { 1 / 2 } \| _ { \mathrm { o p } } \leq \| \Sigma _ { \mathbf { Y } } \| _ { \mathrm { o p } } \| A \| _ { \mathrm { o p } } \leq \tau ^ { 2 } ( \| \mathbf { x } _ { n } \| ^ { 2 } + \| \mathbf { x } _ { n } ^ { \prime } \| ^ { 2 } ) \leq 2 \tau ^ { 2 } R ^ { 2 } , } \end{array}\tag{A.145}
$$

so the positivity condition holds for $| t | < b _ { Q } ^ { - 1 }$ where $b _ { Q } \triangleq 2 \tau ^ { 2 } R ^ { 2 } / \sigma ^ { 2 }$ . The raw MGF of $Q ( \beta )$ is:

$$
\mathbb { E } \left[ e ^ { t Q ( \beta ) } \right] = \frac { \exp \left( - \frac { t } { 2 \sigma ^ { 2 } } \pmb { \mu } _ { \mathbf { Y } } ^ { \top } A \left( \mathbf { I } + \frac { t } { \sigma ^ { 2 } } \Sigma _ { \mathbf { Y } } A \right) ^ { - 1 } \pmb { \mu } _ { \mathbf { Y } } \right) } { \sqrt { \operatorname* { d e t } \left( \mathbf { I } + \frac { t } { \sigma ^ { 2 } } \tilde { A } \right) } } .\tag{A.146}
$$

Taking the logarithm of Eq. (A.146) and centering by $\mu _ { Q }$ via Eq. (A.143):

$$
\begin{array} { l } { { \displaystyle \log \mathbb { E } \left[ e ^ { t ( Q ( \beta ) - \mu _ { Q } ) } \right] = - \frac { 1 } { 2 } \log \operatorname* { d e t } \left( { \bf I } + \frac { t } { \sigma ^ { 2 } } { \tilde { A } } \right) - \frac { t } { 2 \sigma ^ { 2 } } \mu _ { \mathrm { Y } } ^ { \top } A \left( { \bf I } + \frac { t } { \sigma ^ { 2 } } \Sigma _ { \bf Y } A \right) ^ { - 1 } \mu _ { \mathrm { Y } } } } \\ { { \displaystyle ~ + \frac { t } { 2 \sigma ^ { 2 } } \left( \mathrm { t r } ( A \Sigma _ { \bf Y } ) + \mu _ { \mathrm { Y } } ^ { \top } A \mu _ { \mathrm { Y } } \right) } . } \end{array}\tag{A.147}
$$

We bound each term in Eq. (A.147) separately.

Term 1 (log-determinant). Applying the scalar inequality $- \log ( 1 + x ) \leq - x + x ^ { 2 } / ( 1 - | x | )$ (valid for $| x | < 1 )$ eigenvalue-by-eigenvalue to $\tilde { A }$ , and using $\| t \tilde { A } / \sigma ^ { 2 } \| _ { \mathrm { o p } } \leq b _ { Q } | t | < 1 \colon$

$$
- \frac { 1 } { 2 } \log \operatorname * { d e t } \biggl ( { \bf I } + \frac { t } { \sigma ^ { 2 } } { \tilde { A } } \biggr ) \leq - \frac { t } { 2 \sigma ^ { 2 } } \mathrm { t r } ( { \tilde { A } } ) + \frac { t ^ { 2 } \mathrm { t r } ( { \tilde { A } } ^ { 2 } ) } { 2 \sigma ^ { 4 } ( 1 - b _ { Q } | t | ) } ,\tag{A.148}
$$

where $\mathrm { t r } ( \tilde { A } ) = \mathrm { t r } ( \Sigma _ { \mathbf { Y } } A )$

Term 2 (quadratic numerator). Since $\begin{array} { r } { ( \mathbf { I } + \frac { t } { \sigma ^ { 2 } } \Sigma _ { \mathbf { Y } } A ) ^ { - 1 } } \end{array}$ is not symmetric in general, we use operator norms directly. Note first that

$$
\left\| \frac { t } { \sigma ^ { 2 } } \Sigma _ { \mathbf { Y } } A \right\| _ { \mathrm { o p } } \leq \frac { | t | } { \sigma ^ { 2 } } \| \Sigma _ { \mathbf { Y } } \| _ { \mathrm { o p } } \| A \| _ { \mathrm { o p } } \leq b _ { Q } | t | < 1 ,\tag{A.149}
$$

so the resolvent bound gives

$$
\left. \left( \mathbf { I } + { \frac { t } { \sigma ^ { 2 } } } \Sigma _ { \mathbf { Y } } A \right) ^ { - 1 } \right. _ { \mathrm { o p } } \leq { \frac { 1 } { 1 - b _ { Q } | t | } } .\tag{A.150}
$$

Writing $\begin{array} { r } { ( { \bf I } + \frac { t } { \sigma ^ { 2 } } \Sigma { \bf \underline { { Y } } } A ) ^ { - 1 } = { \bf I } - \frac { t } { \sigma ^ { 2 } } \Sigma { \bf \underline { { Y } } } A ( { \bf I } + \frac { t } { \sigma ^ { 2 } } \Sigma { \bf \underline { { Y } } } A ) ^ { - 1 } } \end{array}$ and applying Eq. (A.150):

$$
\begin{array} { r l } & { \quad - \displaystyle \frac { t } { 2 \sigma ^ { 2 } } \mu _ { \mathbf { Y } } ^ { \top } A \left( \mathbf { I } + \frac { t } { \sigma ^ { 2 } } \Sigma _ { \mathbf { Y } } A \right) ^ { - 1 } \mu _ { \mathbf { Y } } } \\ & { = \displaystyle - \frac { t } { 2 \sigma ^ { 2 } } \mu _ { \mathbf { Y } } ^ { \top } A \mu _ { \mathbf { Y } } + \frac { t ^ { 2 } } { 2 \sigma ^ { 4 } } \mu _ { \mathbf { Y } } ^ { \top } A \Sigma _ { \mathbf { Y } } A \left( \mathbf { I } + \frac { t } { \sigma ^ { 2 } } \Sigma _ { \mathbf { Y } } A \right) ^ { - 1 } \mu _ { \mathbf { Y } } } \\ & { \leq \displaystyle - \frac { t } { 2 \sigma ^ { 2 } } \mu _ { \mathbf { Y } } ^ { \top } A \mu _ { \mathbf { Y } } + \frac { t ^ { 2 } \| \mu _ { \mathbf { Y } } \| ^ { 2 } \| A \| _ { \mathrm { o p } } ^ { 2 } \| \Sigma _ { \mathbf { Y } } \| _ { \mathrm { o p } } } { 2 \sigma ^ { 4 } ( 1 - b _ { Q } | t | ) } , } \end{array}\tag{A.151}
$$

where the last step uses $\begin{array} { r } { \big | \mu _ { { \bf Y } } ^ { \top } A \Sigma _ { { \bf Y } } A ( { \bf I } + \frac { t } { \sigma ^ { 2 } } \Sigma _ { { \bf Y } } A ) ^ { - 1 } \mu _ { { \bf Y } } \big | \le \| { \pmb \mu _ { \bf Y } } \| ^ { 2 } \| A \| _ { \mathrm { o p } } ^ { 2 } \| \Sigma _ { { \bf Y } } \| _ { \mathrm { o p } } / ( 1 - b _ { Q } ) ^ { 2 } , } \end{array}$ |t|) via Eq. (A.150).

Term 3 (centering).

$$
{ \frac { t } { 2 \sigma ^ { 2 } } } \left( \mathrm { t r } ( A \Sigma _ { \mathbf { Y } } ) + { \pmb { \mu } { \Sigma } } _ { \mathbf { Y } } ^ { \top } A { \pmb { \mu } } _ { \mathbf { Y } } \right) = { \frac { t } { 2 \sigma ^ { 2 } } } \mathrm { t r } ( { \tilde { A } } ) + { \frac { t } { 2 \sigma ^ { 2 } } } { \pmb { \mu } } _ { \mathbf { Y } } ^ { \top } A { \pmb { \mu } } _ { \mathbf { Y } } .\tag{A.152}
$$

Combining Terms 1–3, the linear terms $- \frac { t } { 2 \sigma ^ { 2 } } \mathrm { t r } ( \tilde { A } )$ and $- \frac { t } { 2 \sigma ^ { 2 } } \mu _ { \mathbf { Y } } ^ { \top } A \mu _ { \mathbf { Y } }$ cancel exactly with Eq. (A.152), giving:

$$
\log \mathbb { E } \left[ e ^ { t ( Q ( \beta ) - \mu _ { Q } ) } \right] \leq \frac { t ^ { 2 } } { 2 ( 1 - b _ { Q } | t | ) } \left( \frac { \mathrm { t r } ( \tilde { A } ^ { 2 } ) } { \sigma ^ { 4 } } + \frac { \| \mu _ { \mathbf { Y } } \| ^ { 2 } \| A \| _ { \mathrm { o p } } ^ { 2 } \| \Sigma _ { \mathbf { Y } } \| _ { \mathrm { o p } } } { \sigma ^ { 4 } } \right) \triangleq \frac { t ^ { 2 } \nu _ { Q } ^ { 2 } } { 2 ( 1 - b _ { Q } | t | ) } .\tag{A.153}
$$

Bounding each term using $\Sigma _ { \mathbf { Y } } \preceq \tau ^ { 2 } \mathbf { I } _ { d } , \| A \| _ { F } ^ { 2 } \leq 8 R ^ { 4 } , \| A \| _ { \mathrm { o p } } \leq 2 R ^ { 2 }$ , and $\| \pmb { \mu } _ { \mathbf { Y } } \| \le n \tau ^ { 2 } M _ { y } R / \sigma ^ { 2 } $

$$
\mathrm { t r } ( \tilde { A } ^ { 2 } ) = \mathrm { t r } ( \Sigma _ { \mathbf { Y } } A \Sigma _ { \mathbf { Y } } A ) \leq \tau ^ { 4 } \| A \| _ { F } ^ { 2 } \leq 8 \tau ^ { 4 } R ^ { 4 } ,\tag{A.154}
$$

$$
\| \pmb { \mu _ { \mathbf { Y } } } \| ^ { 2 } \| \boldsymbol { A } \| _ { \mathrm { o p } } ^ { 2 } \| \Sigma _ { \mathbf { Y } } \| _ { \mathrm { o p } } \leq \frac { n ^ { 2 } \tau ^ { 4 } M _ { y } ^ { 2 } R ^ { 2 } } { \sigma ^ { 4 } } \cdot 4 R ^ { 4 } \cdot \tau ^ { 2 } = \frac { 4 n ^ { 2 } \tau ^ { 6 } M _ { y } ^ { 2 } R ^ { 6 } } { \sigma ^ { 4 } } ,\tag{A.155}
$$

so that

$$
\nu _ { Q } ^ { 2 } \leq \frac { 8 \tau ^ { 4 } R ^ { 4 } } { \sigma ^ { 4 } } + \frac { 4 n ^ { 2 } \tau ^ { 6 } M _ { y } ^ { 2 } R ^ { 6 } } { \sigma ^ { 8 } } .\tag{A.156}
$$

Step 3: Combining via Cauchy-Schwarz. Since $L ( \beta )$ and $Q ( \beta )$ both depend on $\beta ,$ , they are not independent. Applying Cauchy-Schwarz:

$$
\begin{array} { r l r } {  { \log \mathbb { E } \Big [ e ^ { t ( c _ { 1 } ( \beta ) - \mu _ { \mathrm { Y } } ^ { c _ { 1 } } ) } \Big ] \leq \frac { 1 } { 2 } \log \mathbb { E } \Big [ e ^ { 2 t ( L - \mu _ { L } ) } \Big ] + \frac { 1 } { 2 } \log \mathbb { E } \Big [ e ^ { 2 t ( Q - \mu _ { Q } ) } \Big ] } } \\ & { } & { \leq \frac { ( 2 t ) ^ { 2 } \bar { \sigma } _ { L } ^ { 2 } } { 4 } + \frac { ( 2 t ) ^ { 2 } \nu _ { Q } ^ { 2 } } { 4 ( 1 - b _ { Q } | 2 t | ) } } \\ & { } & { = t ^ { 2 } \bar { \sigma } _ { L } ^ { 2 } + \frac { t ^ { 2 } \nu _ { Q } ^ { 2 } } { 1 - 2 b _ { Q } | t | } } \\ & { } & { \leq \frac { t ^ { 2 } ( \bar { \sigma } _ { L } ^ { 2 } + \nu _ { Q } ^ { 2 } ) } { 1 - 2 b _ { Q } | t | } \triangleq \frac { t ^ { 2 } \nu _ { \mathrm { Y } } ^ { 2 } / 2 } { 1 - b _ { \mathrm { Y } } | t | } , } \end{array}\tag{A.157}
$$

where

$$
\nu _ { \mathbf { Y } } ^ { 2 } \triangleq 2 ( \bar { \sigma } _ { L } ^ { 2 } + \nu _ { Q } ^ { 2 } ) \leq \frac { 8 M _ { y } ^ { 2 } R ^ { 2 } \tau ^ { 2 } } { \sigma ^ { 4 } } + \frac { 1 6 \tau ^ { 4 } R ^ { 4 } } { \sigma ^ { 4 } } + \frac { 8 n ^ { 2 } \tau ^ { 6 } M _ { y } ^ { 2 } R ^ { 6 } } { \sigma ^ { 8 } } ,\tag{A.158}
$$

$$
b _ { \mathbf { Y } } \triangleq 2 b _ { Q } = \frac { 4 \tau ^ { 2 } R ^ { 2 } } { \sigma ^ { 2 } } .\tag{A.159}
$$

This establishes that $c _ { 1 } ( \beta )$ satisfies a sub-exponential MGF bound under $\pi ( \beta | \mathbf { Y } )$ with parameters $\left( \nu _ { \mathbf { Y } } ^ { 2 } , b _ { \mathbf { Y } } \right)$ in the sense of Eq. (A.33).

For the reversed pair $( \mathbf { Y } ^ { \prime } , \mathbf { Y } ) , \tilde { c } _ { 1 } ( \boldsymbol { \beta } ) = - c _ { 1 } ( \boldsymbol { \beta } ) = - L ( \boldsymbol { \beta } ) - Q ( \boldsymbol { \beta } )$ , i.e., $( L , Q , A )$ is replaced by $( - L , - Q , - A )$ . Steps 1–3 use only $\Sigma \preceq \tilde { \tau } ^ { 2 } \mathbf { I } _ { d } , \| \mathbf { x } _ { i } \| \leq R , | \tilde { y _ { i } | } \leq M _ { y } , \| \pmb { \mu } \| \leq \tilde { n } \tau ^ { 2 } \tilde { M _ { y } } R / \sigma ^ { 2 }$ $\| { - } \hat { A } \| _ { \mathrm { o p } } = \| A \| _ { \mathrm { o p } }$ and $\lVert - A \rVert _ { F } = \lVert \dot { A } \rVert _ { F } .$ , all of which also hold under $\pi ( \beta | { \bf Y } ^ { \prime } )$ since $\overset { \vartriangle } { \mathbf { Y } ^ { \prime } }$ also has n records. Hence $\tilde { c } _ { 1 } ( \beta )$ satisfies the same sub-exponential bound with parameters $\left( \nu _ { \mathbf { Y } } ^ { 2 } , b _ { \mathbf { Y } } \right)$ in Eqs. (A.158)–(A.159).

Step 4: Applying Corollary 13. Corollary 13 requires $b _ { \mathbf { Y } } < \operatorname* { m i n } \{ ( \alpha - 1 ) ^ { - 1 } , 1 \}$ . The condition $b \mathbf { r } < 1$ is an additional scale restriction requiring:

$$
\frac { 4 \tau ^ { 2 } R ^ { 2 } } { \sigma ^ { 2 } } < 1 \Longleftrightarrow \sigma ^ { 2 } > 4 \tau ^ { 2 } R ^ { 2 } .\tag{A.160}
$$

Under Eq. (A.160) and $\alpha < 1 + \sigma ^ { 2 } / ( 4 \tau ^ { 2 } R ^ { 2 } )$ , substituting $\left( \nu _ { \mathbf { Y } } ^ { 2 } , b _ { \mathbf { Y } } \right)$ into Corollary 13:

$$
\operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} \leq \varepsilon _ { \mathrm { L R , C a s e 2 } } ( \alpha , b _ { \mathbf { Y } } , \nu _ { \mathbf { Y } } ^ { 2 } ) \triangleq \frac { \nu _ { \mathbf { Y } } ^ { 2 } } { 2 } \cdot \frac { \alpha ( 1 - 2 b _ { \mathbf { Y } } ) + 2 b _ { \mathbf { Y } } } { ( 1 - b _ { \mathbf { Y } } ) \left( 1 - b _ { \mathbf { Y } } ( \alpha - 1 ) \right) } \cdot\tag{A.161}
$$

Remark A.10. Three sources of looseness affect Eq. (A.161). First, the Cauchy-Schwarz step doubles b , restricting the valid range of α. Second, the operator norm bound $\| \tilde { A } \| _ { \mathrm { o p } } \leq 2 \tau ^ { 2 } R ^ { 2 }$ ignores data-dependent tightening of $\Sigma _ { \mathbf { Y } }$ due to the likelihood. Third, the n<sup>2</sup>-dependent term in $\nu _ { \mathbf { Y } } ^ { 2 }$ arises from bounding $\| \mu _ { \mathbf { Y } } \|$ using only row-wise boundedness; under a direct bound $\| \mu _ { \mathbf { Y } } \| \le C f o r$ some constant C, this term becomes n-independent. Despite this looseness, Eq. (A.161) yields an explicit finite-order RDP bound for OPS in the bounded-neighboring linear-regression setting with covariate replacement, and illustrates how the sub-exponentialframework ofCorollary 13 extends beyond the sub-Gaussian cases treated in the main text.

## A.10.2 unbounded neighboring

Let Y has size n and $\mathbf { Y } ^ { \prime }$ is of size $n - 1$ by removing the n-th record ${ \bf z } _ { n } = ( { \bf x } _ { n } , y _ { n } )$ in Y. Then

$$
c _ { 1 } ( { \boldsymbol { \beta } } ) = \log { \frac { \ell ( { \boldsymbol { \beta } } ; \mathbf { Y } ) } { \ell ( { \boldsymbol { \beta } } ; \mathbf { Y } ^ { \prime } ) } } = \log p ( y _ { n } | \mathbf { x } _ { n } , { \boldsymbol { \beta } } ) = - { \frac { ( y _ { n } - \mathbf { x } _ { n } ^ { \top } { \boldsymbol { \beta } } ) ^ { 2 } } { 2 \sigma ^ { 2 } } } - { \frac { 1 } { 2 } } \log ( 2 \pi \sigma ^ { 2 } ) .
$$

This expression is structurally identical to Eq. (A.95) by setting $Z = y _ { n } - \mathbf { x } _ { n } ^ { \top } { \boldsymbol { \beta } } .$ , where

$$
Z | \mathbf { Y } \sim { \mathcal { N } } \left( y _ { n } - \mathbf { x } _ { n } ^ { \top } { \pmb { \mu } } _ { \mathbf { Y } } , \mathbf { x } _ { n } ^ { \top } { \pmb { \Sigma } } _ { \mathbf { Y } } \mathbf { x } _ { n } \right)
$$

In Proposition $1 1 , \omega = \sigma ^ { 2 } / v _ { Y } ^ { 2 }$ , where $v _ { Y } ^ { 2 }$ is the posterior variance of $\theta - y _ { n }$ ; here the corresponding quantity is the posterior variance of $Z | \mathbf { Y }$ . Thus, the corresponding $\omega _ { n }$ here is $\begin{array} { r } { \omega _ { n } = \frac { \sigma ^ { 2 } } { \mathbf { x } _ { n } ^ { \top } \Sigma \mathbf { y } \mathbf { x } _ { n } } } \end{array}$ , plug in Eq. (A.110)

$$
\begin{array} { r l r } {  { D _ { \alpha } ( P \| Q ) = \underbrace { - \frac { \log ( 1 - \omega _ { n } ^ { - 1 } ) } { 2 } - \frac { \log ( 1 + \frac { \alpha - 1 } { \omega _ { n } } ) } { 2 ( \alpha - 1 ) } } _ { A ( \alpha , \omega _ { n } ) } + \frac { ( \mathbf { x } _ { n } ^ { \top } \boldsymbol { \mu } _ { \mathbf { Y } } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \underbrace { [ \frac { 1 } { 1 - \omega _ { n } ^ { - 1 } } - \frac { 1 } { 1 + \frac { \alpha - 1 } { \omega _ { n } } } ] } _ { B ( \alpha , \omega _ { n } ) } } } \\ & { } & { \leq A ( \alpha , \omega _ { n } ) + \frac { k ^ { * 2 } } { 2 } B ( \alpha , \omega _ { n } ) , \qquad ( \mathrm { A } . \mathrm { I } \omega , \omega _ { n } ) } \end{array}\tag{62}
$$

if $| \mathbf { x } _ { n } ^ { \top } \pmb { \mu } _ { \mathbf { Y } } - y _ { n } | \leq k ^ { * } \sigma$ for some $k ^ { * } > 0$

Reverse direction. We now bound $D _ { \alpha } ( \pi ( \beta | \mathbf { Y } ^ { \prime } ) | | \pi ( \beta | \mathbf { Y } ) )$ . The posterior under $\mathbf { Y ^ { \prime } }$ is $\beta | \mathbf { Y } ^ { \prime }$ ∼ $\mathcal { N } ( \mu _ { \mathbf { Y ^ { \prime } } } , \Sigma _ { \mathbf { Y ^ { \prime } } } )$ with $\Sigma _ { { \mathbf { Y } } ^ { \prime } } ^ { - 1 } = \Sigma _ { { \mathbf { Y } } } ^ { - 1 } - \sigma ^ { - 2 } { \mathbf { x } } _ { n } { \mathbf { x } } _ { n } ^ { \top }$ . Let $\begin{array} { r } { v _ { n } ^ { \prime 2 } \triangleq \mathbf { x } _ { n } ^ { \top } \Sigma _ { \mathbf { Y } ^ { \prime } } \mathbf { x } _ { n } } \end{array}$ and

$$
\begin{array} { l } { { \displaystyle \tilde { c } _ { 1 } ( \beta ) = \log \frac { \ell ( \beta ; \mathbf { Y } ^ { \prime } ) } { \ell ( \beta ; \mathbf { Y } ) } = \frac { ( y _ { n } - \mathbf { x } _ { n } ^ { \top } \beta ) ^ { 2 } } { 2 \sigma ^ { 2 } } + \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) , } } \\ { { \displaystyle Z ^ { \prime } \triangleq y _ { n } - \mathbf { x } _ { n } ^ { \top } \beta \vert \mathbf { Y } ^ { \prime } \sim \mathcal { N } ( y _ { n } - \mathbf { x } _ { n } ^ { \top } \pmb \mu _ { \mathbf { Y } ^ { \prime } } , v _ { n } ^ { \prime 2 } ) } . } \end{array}\tag{A.163}
$$

This is structurally identical to Eq. (A.117) with $\mu _ { Y ^ { \prime } }$ replaced by $\mathbf { x } _ { n } ^ { \top } \pmb { \mu } _ { \mathbf { Y } ^ { \prime } }$ and $v _ { Y } ^ { 2 }$ <sub>′</sub> replaced by $v _ { n } ^ { \prime 2 } .$ Hence, by Eq. (A.125), $D _ { \alpha } ( Q \| P ) ^ { - } < \infty$ if and only if $( \bar { \alpha } - 1 ) v _ { n } ^ { \prime 2 } < \bar { \sigma } ^ { 2 }$ . By the Sherman–Morrison formula,

$$
\Sigma _ { \mathbf { Y } } = \Sigma _ { \mathbf { Y ^ { \prime } } } - \frac { \Sigma _ { \mathbf { Y ^ { \prime } } } \mathbf { X } _ { n } \mathbf { X } _ { n } ^ { \top } \Sigma _ { \mathbf { Y ^ { \prime } } } } { \sigma ^ { 2 } + v _ { n } ^ { \prime 2 } } \Rightarrow \mathbf { x } _ { n } ^ { \top } \Sigma _ { \mathbf { Y } } \mathbf { x } _ { n } = \frac { \sigma ^ { 2 } v _ { n } ^ { \prime 2 } } { \sigma ^ { 2 } + v _ { n } ^ { \prime 2 } } \Rightarrow \omega _ { n } = 1 + \frac { \sigma ^ { 2 } } { v _ { n } ^ { \prime 2 } } ,
$$

$$
\mathrm { i . e . } \ \frac { v _ { n } ^ { \prime 2 } } { \sigma ^ { 2 } } = \frac { 1 } { \omega _ { n } - 1 } .
$$

Moreover, $\begin{array} { r } { \mu _ { \mathbf { Y } } = \Sigma _ { \mathbf { Y } } \big ( \Sigma _ { \mathbf { Y ^ { \prime } } } ^ { - 1 } \mu _ { \mathbf { Y ^ { \prime } } } + \sigma ^ { - 2 } \mathbf { x } _ { n } y _ { n } \big ) \mathrm { a n d } \mathbf { x } _ { n } ^ { \top } \Sigma _ { \mathbf { Y } } \Sigma _ { \mathbf { Y ^ { \prime } } } ^ { - 1 } = \frac { \sigma ^ { 2 } } { \sigma ^ { 2 } + v _ { n } ^ { \prime 2 } } \mathbf { x } _ { n } ^ { \top } , \mathbf { s } { \mathrm { 0 } } } \end{array}$ o

$$
\mathbf { x } _ { n } ^ { \top } \pmb { \mu } _ { \mathbf { Y } } = \frac { \sigma ^ { 2 } } { \sigma ^ { 2 } + v _ { n } ^ { \prime 2 } } \mathbf { x } _ { n } ^ { \top } \pmb { \mu } _ { \mathbf { Y } ^ { \prime } } + \frac { v _ { n } ^ { \prime 2 } } { \sigma ^ { 2 } + v _ { n } ^ { \prime 2 } } y _ { n } \Rightarrow \mathbf { x } _ { n } ^ { \top } \pmb { \mu } _ { \mathbf { Y } ^ { \prime } } - y _ { n } = \frac { \omega _ { n } } { \omega _ { n } - 1 } ( \mathbf { x } _ { n } ^ { \top } \pmb { \mu } _ { \mathbf { Y } } - y _ { n } ) .\tag{A.164}
$$

These coincide with $v _ { Y ^ { \prime } } ^ { 2 } = \sigma ^ { 2 } / ( \omega - 1 )$ and Eq. (A.128) with $\omega$ replaced by $\omega _ { n }$ . Repeating the derivation of Eq. $( \mathrm { A } . 1 2 { \bar { 9 } } )$ , when $1 < \alpha < \omega _ { n }$

$$
D _ { \alpha } ( Q \| P ) = \widetilde { A } ( \alpha , \omega _ { n } ) + \frac { ( \mathbf { x } _ { n } ^ { \top } \mu _ { \mathbf { Y } } - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \widetilde { B } ( \alpha , \omega _ { n } ) \leq \widetilde { A } ( \alpha , \omega _ { n } ) + \frac { k ^ { * 2 } } { 2 } \widetilde { B } ( \alpha , \omega _ { n } ) ,\tag{A.165}
$$

and $D _ { \alpha } ( Q \| P ) = \infty$ for $\alpha \geq \omega _ { n }$

Two-sided, data-independent guarantee. Since $\Sigma _ { \mathbf { Y } ^ { \prime } } \preceq \tau ^ { 2 } \mathbf { I } _ { d }$ and $\| \mathbf { x } _ { n } \| \leq R , v _ { n } ^ { \prime 2 } \leq \tau ^ { 2 } R ^ { 2 }$ , hence $\omega _ { n } \geq \omega _ { \operatorname* { m i n } } \triangleq 1 + \sigma ^ { 2 } / ( \tau ^ { 2 } R ^ { 2 } )$ for every admissible neighboring pair, with equality, e.g., when $n = 1$ and $\| \mathbf { x } _ { 1 } \| = R$ . By the monotonicity of $A , B$ and $\tilde { A } , \tilde { B }$ in $\omega$ (Claim A.9 and A.8), for every $1 < \alpha < \omega _ { \mathrm { { m i n } } }$

$$
\begin{array} { r l } & { \quad \operatorname* { m a x } \{ D _ { \alpha } ( P \| Q ) , D _ { \alpha } ( Q \| P ) \} } \\ & { \le \operatorname* { m a x } \Big \{ A ( \alpha , \omega _ { \operatorname* { m i n } } ) + \cfrac { { k ^ { * 2 } } } { 2 } B ( \alpha , \omega _ { \operatorname* { m i n } } ) , \ \widetilde { A } ( \alpha , \omega _ { \operatorname* { m i n } } ) + \cfrac { { k ^ { * 2 } } } { 2 } \widetilde { B } ( \alpha , \omega _ { \operatorname* { m i n } } ) \Big \} , } \end{array}\tag{A.166}
$$

whereas for $\alpha \geq \omega _ { \mathrm { m i n } }$ there exist admissible neighboring pairs with $D _ { \alpha } ( Q \| P ) = \infty$ . In particular, OPS does not satisfy pure DP in this setting.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the paper’s theoretical contributions Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: The paper discusses limitations in Section 6, including the dependence of the guarantees on model assumptions, tail conditions on the log-likelihood ratio, and the looseness of general sub-exponential bounds relative to model-specific exact calculations.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: The paper is theoretical. Each theoretical result states its assumptions, and proofs are provided either in the main text or in the appendix.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [N/A]

Justification: The paper does not report empirical experiments. The main contributions are theoretical.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [N/A]

Justification: The paper does not use datasets or code to support experimental claims. The work is theoretical.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [N/A]

Justification: The paper does not include experiments. The work is theoretical.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [N/A]

Justification: The paper does not include experiments. The work is theoretical.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [N/A]

Justification: The paper does not include experiments. The work is theoretical.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research is theoretical and concerns privacy guarantees for Bayesian posterior sampling.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Potential positive impacts include improved understanding of privacypreserving Bayesian posterior sampling. Such understanding can help design mechanisms that preserve more utility in released statistical outputs while exploiting the inherent privacy provided by posterior randomness. potential limitations include the risk of generalizing the guarantees outside the stated assumptions.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release models, datasets, or tools with high misuse risk. Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [N/A]

Justification: The paper does not use existing datasets, code, models, or other external assets.Related Prior work is listed in the References Section.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not introduce or release new datasets, codebases, models, benchmarks, or other assets.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve human-subject research.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The core method development does not involve LLMs as an important, original, or non-standard component.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.