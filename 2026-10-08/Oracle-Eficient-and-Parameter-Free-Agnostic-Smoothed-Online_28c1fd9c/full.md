# Oracle-Eficient and Parameter-Free Agnostic Smoothed Online Learning

Sasha Voitovych MIT voitovyc@mit.edu

Adam Block Columbia University adam.block@columbia.edu

Alexander Rakhlin   
MIT   
rakhlin@mit.edu

Abhishek Shetty Georgia Tech shetty@mit.edu

## Abstract

Online learning is an attractive framework in many domains because it permits well-defined learning even when data are dependent or chosen adversarially. This generality, however, comes at a steep price, introducing significant statistical and computational barriers. Recently, smoothed online learning has emerged as a promising framework that interpolates between the fully adversarial and fully stochastic settings by assuming that the conditional law of each covariate has density at most 1/σ with respect to some fixed base measure µ, and it is known to match the statistical and computational guarantees of classical learning while still allowing for much of the flexibility of online learning. However, existing oracle-eficient algorithms require either (i) sampling access to the base measure µ or (ii) labels that are perfectly predicted by a fixed hypothesis. Both assumptions limit the applicability of these algorithms, in contrast to statistical learning, where empirical risk minimization (ERM) learns eficiently in the agnostic setting without any knowledge of the data distribution. We show that neither assumption is necessary, giving the first oracle-eficient algorithm that achieves sublinear regret in the agnostic setting without knowledge of µ. Our algorithm, based on Gaussian Follow-The-Perturbed-Leader, is parameter-free: it requires no knowledge of µ, the smoothing parameter σ, or the horizon T, and it achieves regret $\widetilde { O } ( d \sqrt { T / \sigma } )$ for binary classes of VC dimension d with a single call to an ERM oracle per round, which is optimal up to a d factor. En route to establishing the regret bound, we introduce several new techniques that may be of independent interest.

## 1 Introduction

Learning from independent data without a priori knowledge of the generating process is the archetypal paradigm in machine learning (Valiant, 1984): a learner that observes a training sample drawn from an unknown distribution can generalize to new examples from the same distribution. Its main appeal is that it removes the need to model nature’s distribution, which may be highly complex and need not belong to a known parametric family. Beyond the statistical benefits of assuming independence, there are computational advantages: near-optimal learning can be achieved with a simple algorithm, empirical risk minimization (ERM), which simply optimizes its predictions on historical data (Vapnik and Chervonenkis, 1971; Blumer et al., 1989).

In many modern applications, however, the independence assumption is dificult to justify, let alone verify. Data often arrive sequentially, and the data-generating process may even react to the learner’s own predictions. Many modern data modalities, such as text, video, and state trajectories of physical systems (e.g., in robotics and control), exhibit strong temporal dependencies. In interactive decision making, such as bandits and reinforcement learning, the learner’s own past decisions determine which data it collects (Lattimore and Szepesv´ari, 2020; Foster et al., 2021). Financial markets are another example: prices are strongly dependent over time, and the predictions of trading systems influence the actions of market participants, which in turn shape the data those systems observe. Deployed recommendation systems exhibit similar feedback loops: the system’s recommendations influence users’ behavior, which in turn shapes the data the system collects (Chaney et al., 2018; Jiang et al., 2019). In such settings, guarantees premised on independence need not hold, and algorithms developed assuming independence may not learn, necessitating a theory of learning that makes minimal assumptions on how the data are generated.

Online learning (Littlestone, 1988) provides a promise of such a theory: it dispenses with the independence assumption altogether and permits learning even from dependent, possibly adversarial data. Formally, online learning is a repeated game between a learner and nature: in each round t, the learner observes a covariate $X _ { t }$ , predicts a label $\hat { Y } _ { t } ,$ observes the true label $Y _ { t } ,$ , and sufers loss $\ell ( \hat { Y } _ { t } , Y _ { t } )$ . The learner’s goal is to minimize its regret with respect to a fixed comparator class ${ \mathcal { F } } \mathrm { : }$

$$
\mathrm { R e g } _ { T } = \sum _ { t = 1 } ^ { T } \ell ( \hat { Y } _ { t } , Y _ { t } ) - \operatorname* { i n f } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) .\tag{1}
$$

Importantly, the data-generating process may be arbitrary; in particular, it may adapt to the learner’s past actions. This generality, however, comes at both a statistical and a computational price.

Learnability in online binary classification is characterized by the Littlestone dimension, which may be infinite even for simple classes such as linear threshold functions (Littlestone, 1988; Ben-David et al., 2009). The situation is further exacerbated by computational barriers. A natural computational framework is the ERM oracle model, designed to capture the eficacy of optimization in modern practice. When queried, an ERM oracle for $\mathcal { F }$ returns a predictor $f \in { \mathcal { F } }$ that minimizes the empirical loss on the given dataset. This oracle forms the basic computational primitive of statistical learning and is a practically meaningful abstraction, since ERM is routinely solved, at least approximately, with of-the-shelf tools such as stochastic gradient descent. Ideally, then, one would like online learning algorithms to be oracle-eficient, i.e., run eficiently when given query access to an ERM oracle. Without further assumptions in the sequential setting, however, Hazan and Koren (2016) showed that this goal is unattainable: even for a finite comparator class, access to an ERM oracle does not make online learning computationally tractable. In other words, in the absence of independence, generalization and learning do not reduce to optimization.

In order to circumvent these barriers, there has been substantial efort in understanding smoothed online learning, which provides a way to interpolate between the stochastic and adversarial regimes (Rakhlin et al., 2011; Haghtalab et al., 2020; Block et al., 2022, 2024b; Blanchard, 2025; Blanchard et al., 2026). Intuitively, smoothness prevents the adversary from concentrating probability mass on small sets, while still permitting strong temporal dependencies. Formally, letting $p _ { t }$ denote the law of $X _ { t }$ conditional on the history, we require that there exist a single probability measure $\mu ,$ referred to as the base measure, such that in every round t,

$$
\left. \frac { \mathrm { d } p _ { t } } { \mathrm { d } \mu } \right. _ { \infty } \leq \frac { 1 } { \sigma } ,
$$

where the smoothness parameter $\sigma \in ( 0 , 1 ]$ controls the power of the adversary. At $\sigma = 1$ , the covariates are drawn independently from $\mu$ in every round; as $\sigma \downarrow 0 .$ , the constraint becomes increasingly permissive. This relaxation removes both statistical and computational barriers: an extensive literature shows that VC classes are learnable in the smoothed setting for every $\sigma > 0$ and, moreover, eficiently so in the ERM-oracle model (Haghtalab et al., 2020; Block et al., 2022; Haghtalab et al., 2022). Further, from a statistical perspective, Blanchard et al. (2026) showed that a technical generalization of the smoothed model essentially captures learnability for general sequences of distributions.

However, most oracle-eficient algorithms in this setting assume that $\mu$ is known to the learner and rely crucially on the ability to sample from it. In practice, this amounts to an additional modeling assumption on nature, and the choice of any particular $\mu$ is often hard to justify. A notable exception is the work of Block et al. (2024b), which shows that ERM itself achieves sublinear regret in the realizable setting, where the data are labeled by some fixed $f ^ { \star } \in { \mathcal { F } } , \operatorname { i . e . , } Y _ { t } = f ^ { \star } ( X _ { t } )$ 1 Whether eficient learning is possible in the more general agnostic setting has remained open. This is in contradistinction to classical statistical learning theory, where a single algorithm (ERM) succeeds in both the realizable and agnostic settings while being entirely parameter-free and requiring no knowledge of the data distribution. Thus, in an attempt to relax the independence assumptions on nature, we have lost precisely what makes statistical learning appealing: the learner never needs to model nature explicitly.

Main results. In this work, we give the first oracle-eficient algorithm that achieves sublinear regret in general agnostic smoothed online learning without the knowledge of the base measure. Surprisingly, this guarantee is achieved by a natural algorithm: follow-the-perturbed-leader (FTPL) with Gaussian perturbations, which is entirely parameter-free. It is well known since Hannan (1957) that ERM can fail on agnostic data even when the covariate domain has size 1 (which forces $\sigma = 1 )$ . Intuitively, this failure stems from a lack of stability: ERM is sensitive to individual data points, and its solution can change drastically in response to a single label flip. A standard way to stabilize ERM is to regularize it, either with an explicit potential, as in follow-the-regularized-leader (FTRL) (Shalev-Shwartz and Singer, 2007; Abernethy et al., 2008; McMahan, 2017), or with a random perturbation of the objective, as in FTPL (Hannan, 1957; Kalai and Vempala, 2005; Hutter and Poland, 2005; Devroye et al., 2013). Motivated by the algorithm in Block et al. (2022), we show that natural Gaussian perturbations sufice to achieve sublinear regret. In particular, for binary classification, we show that FTPL with Gaussian perturbations attains

$$
\mathbb { E } \mathrm { R e g } _ { T } \leq \widetilde { O } \left( d \sqrt { \frac { T } { \sigma } } \right)\tag{2}
$$

simultaneously for all $\mu$ and $\sigma .$ . This guarantee has optimal dependence on $T$ and $\sigma ,$ and is statistically suboptimal only by a factor of $\sqrt { d }$ (Blanchard, 2025). We also note that in the realizable case the $\sqrt { d }$ factor can be removed via an $L ^ { \star } { \mathrm { - r e g r e t } }$ bound (see Section D). This matches the ERM guarantee of Block et al. (2024b) in the realizable case and is optimal up to logarithmic factors (Blanchard, 2025, Theorem 3).

We now define our Gaussian FTPL learner formally. Let ${ \mathcal { F } } \subseteq \{ \pm 1 \} ^ { \mathcal { X } }$ be a comparator class. In round $t ,$ the algorithm samples a fresh Gaussian vector $G ^ { ( t ) } = ( G _ { 1 } ^ { ( t ) } , \dots , G _ { t - 1 } ^ { ( t ) } ) \sim \mathcal { N } ( 0 , I _ { t - 1 } )$ and computes

$$
\hat { f } _ { t } \in \underset { f \in \mathcal { F } } { \arg \operatorname* { m i n } } \sum _ { s = 1 } ^ { t - 1 } \left[ \ell ( f ( X _ { s } ) , Y _ { s } ) + G _ { s } ^ { ( t ) } f ( X _ { s } ) \right] ,\tag{3}
$$

where we let $\ell ( \hat { y } , y ) = - y \hat { y }$ in the binary case. We use a weighted ERM oracle (Section 2 gives the precise model). This algorithm is completely parameter-free: it requires no knowledge of the base measure $\mu ,$ the smoothness parameter $\sigma ,$ or the horizon $T .$ . It thus addresses precisely the limitation of prior work discussed above, achieving oracle-eficient learning adaptively for every $\mu$ and $\sigma$

Our algorithm and guarantee extend to classes $\mathcal { F } \subseteq [ - 1 , 1 ] ^ { \mathcal { X } }$ and convex 1-Lipschitz losses by adding an averaging step in each round (Section 3). In terms of the fat-shattering dimension fat ${ \mathfrak { s } } ( { \mathcal { F } } )$ (Section 2), the resulting bound is $\widetilde { O } ( \dot { d } \sqrt { T / \sigma } )$ for parametric classes with fa $\mathrm { t } _ { \varepsilon } ( \mathcal { F } ) \lesssim d \log ( 1 / \varepsilon )$ and ${ \widetilde O } _ { \alpha } ( T ^ { ( 2 \alpha + 1 ) / ( 2 \alpha + 2 ) } / \sqrt { \sigma } )$ for nonparametric classes with $\mathrm { f a t } _ { \varepsilon } ( \mathcal { F } ) \lesssim \varepsilon ^ { - \alpha }$ , matching the leading exponents of T and $1 / \sigma$ in Blanchard (2025, Theorem 4).

Technical overview. We sketch the analysis for the binary case, which already contains the main ideas for the general real-valued setting. A central tool in the analysis of FTPL algorithms is stability (Kalai and Vempala, 2005): intuitively, if the perturbations have controlled magnitude and the algorithm’s predictions are stable, a regret bound follows. However, we find it dificult to analyze the stability of Eq. (3) directly; instead, we are able to show that the predictions of Eq. (3) are close to predictions of an auxiliary algorithm that is itself stable.

In more detail, our proof proceeds in two steps: (i) we first compare the predictions of our FTPL forecaster to those of an auxiliary algorithm, exponential weights over a cover $\mathcal { G }$ of $\mathcal { F }$ (Vovk, 1990; Littlestone and Warmuth, 1994; Freund and Schapire, 1997); (ii) we then directly bound the regret of the auxiliary algorithm. We emphasize that the auxiliary algorithm need not be eficient, as it is used only in the analysis.

Step 1: Comparison with the auxiliary algorithm. Formally, let $\tilde { f } _ { t } \in \mathrm { c o n v } ( \mathcal { G } )$ be defined as follows:

$$
\tilde { f } _ { t } : = \sum _ { f \in \mathcal { G } } w _ { t } ( f \mid G ^ { ( t ) } ) f , \quad \mathrm { w i t h } \quad w _ { t } ( f \mid G ^ { ( t ) } ) \propto \exp \left( - \sum _ { s < t } [ \ell ( f ( X _ { s } ) , Y _ { s } ) + G _ { s } ^ { ( t ) } f ( X _ { s } ) ] \right) ,\tag{4}
$$

such that the weights $w _ { t } ( f \mid G ^ { ( t ) } )$ sum to one over all $f \in { \mathcal { G } }$ . Our first technical contribution reduces the regret of $\hat { f } _ { t }$ to that of $\tilde { f } _ { t }$ . More precisely, letting $\mathrm { R e g } ^ { \mathrm { E W } }$ denote the regret of the learner in Eq. (4), we show that, as long as $\mathcal { G }$ approximates $\mathcal { F }$ suficiently well (see Section 4.2 for details),

$$
\mathbb { E } \mathrm { R e g } _ { T } - \mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { E W } } \leq \widetilde { O } \left( d \sqrt { \frac { T } { \sigma } } \right) .\tag{5}
$$

The proof of this reduction has two steps. First, we compare the predictions of $\hat { f } _ { t }$ and $\tilde { f } _ { t }$ on past covariates $X _ { 1 : t - 1 }$ , showing that for every $1 \leq t \leq T$

$$
\mathbb { E } \sum _ { s = 1 } ^ { t - 1 } ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { s } ) ) ^ { 2 } \lesssim d ^ { 2 } \log ^ { 2 } ( 2 T / \sigma ) .\tag{6}
$$

This follows by expressing the prediction vectors of $\hat { f } _ { t }$ and $\tilde { f } _ { t }$ as gradients of certain potentials and using Stein’s lemma (Stein, 1981; Abernethy et al., 2014) to relate diferences in predictions to diferences in potentials.

Next, we use an improved version of the surprise lemma of Block et al. (2024b) to translate Eq. (6) into closeness of predictions on the current covariate $X _ { t }$ . Specifically, adapting arguments from Xie et al. (2022); Block et al. (2024b) to convex hulls of VC classes, we prove that for any sequence of predictable functions $h _ { t } \in \mathrm { c o n v } ( \pm \mathcal { F } )$ ,

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } | h _ { t } ( X _ { t } ) | \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } + \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) } .
$$

Intuitively, this shows that the value of a predictable test function $h _ { t }$ at a new covariate $X _ { t }$ is controlled by its cumulative second moments on past covariates; informally, values of $| h _ { t } ( X _ { t } )$ that are surprisingly large relative to the history cannot occur too often. Applied to diferences $h _ { t } = ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ) / 2$ and combined with Eq. (6), this shows that predictions $\hat { f } _ { t } ( X _ { t } )$ and $\tilde { f _ { t } } ( X _ { t } )$ are close on average. This proves Eq. (5) since the loss is 1-Lipschitz.

Step 2: Bounding the regret of the auxiliary algorithm. Thus, we have reduced our problem to bounding the regret of the exponential weights algorithm in Eq. (4). Here, a central dificulty is that, while typical worst-case bounds on the regret of exponential weights require learning rate $\eta \propto 1 / \sqrt { T }$ , the algorithm described by Eq. (4) uses a constant learning rate $( \mathrm { i . e . } \ \eta = 1 )$ for which these bounds are vacuous (Cesa-Bianchi and Lugosi, 2006). We show that despite this apparent hurdle, the Gaussian perturbations themselves provide the necessary stability to ensure low regret. Precisely, we show that the predictions of exponential weights with Gaussian perturbations have low cumulative variance:

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G ^ { ( t ) } ) } ( f ( X _ { t } ) ) \leq \widetilde { O } \left( \sqrt { \frac { T ( \log | \mathcal { G } | + d ) } { \sigma } } \right) .
$$

Combining this with the standard variance-based analysis of exponential weights (Cesa-Bianchi et al., 2007; De Rooij et al., 2014) yields a corresponding regret bound:

$$
\mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { E W } } \lesssim d \log ( 2 T / \sigma ) + \sum _ { t = 1 } ^ { T } \mathbb { E } \mathrm { V a r } _ { f \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( f ( X _ { t } ) ) \leq \widetilde O \left( \sqrt { \frac { T ( \log \vert \mathcal G \vert + d ) } { \sigma } } \right) .
$$

By controlling the size of the cover (Proposition 4.2) and using Eq. (5), we obtain Eq. (2).

Conventions and notation. We write log for the natural logarithm. For an integer $n ,$ we let [n] denote the set $\{ 1 , \ldots , n \}$ . Throughout, $C , c > 0$ denote suficiently large and suficiently small universal constants, respectively. We additionally use standard non-asymptotic big-O notation as shorthand: we write $f \le O ( g )$ or $f \lesssim g$ to mean that $f \leq C g$ for a suficiently large universal constant C. We use Oe to suppress logarithmic factors in $d , T .$ , and $1 / \sigma$ . We write $Z _ { a : b }$ for the sequence $( Z _ { a } , Z _ { a + 1 } , \ldots , Z _ { b } )$ . For a class H of functions, we write conv(H) to denote the pointwise closure of the convex hull of H and let $\operatorname { c o n v } ( \pm \mathcal { H } ) : = \operatorname { c o n v } ( \mathcal { H } \cup - \mathcal { H } )$

## 2 Preliminaries and Related Work

We now introduce the necessary technical definitions and notation for our analysis, before moving on to briefly discuss related work.

Online learning and smoothness. We reserve X to refer to the space of covariates, and d to refer to the VC dimension of the comparator class ${ \mathcal F } .$ For convenience, we assume binary labels are in $\{ \pm 1 \}$ . For a sequence of covariates $X _ { t }$ , labels $Y _ { t } ,$ and predictions $\hat { Y } _ { t }$ , let $\mathcal { H } _ { t } = \sigma ( ( X _ { s } , Y _ { s } , \hat { Y } _ { s } ) _ { s \in [ t ] } )$ be the σ-algebra induced by the learner’s observations up to round t, and write $p _ { t } = \operatorname { L a w } ( X _ { t } \mid { \mathcal { H } } _ { t - 1 } )$ For a fixed probability measure $\mu$ and $\sigma \in ( 0 , 1 ]$ , the covariate process is σ-smooth with respect to $\mu$ if, almost surely, for all $t , p _ { t }$ is absolutely continuous w.r.t. $\mu$ and

$$
\left. \frac { \mathrm { d } p _ { t } } { \mathrm { d } \mu } \right. _ { \infty } \leq \frac { 1 } { \sigma } .
$$

Complexity measures. Let ${ \mathcal { F } } \subseteq \{ \pm 1 \} ^ { \mathcal { X } }$ . We say that $\mathcal { F }$ shatters a set $\{ x _ { 1 } , \ldots , x _ { n } \} \subseteq { \mathcal { X } }$ if for every $y _ { 1 : n } \in \{ \pm 1 \} ^ { n }$ there exists $f \in { \mathcal { F } }$ with $f ( x _ { i } ) = y _ { i }$ for all $i \in [ n ]$ . The VC dimension of $\mathcal { F }$ is the largest integer d such that $\mathcal { F }$ shatters some set of size d (or ∞ if no such largest d exists) (Vapnik and Chervonenkis, 1971).

For real-valued classes ${ \mathcal { F } } \subseteq [ - 1 , 1 ] ^ { \mathcal { X } }$ and a scale $\varepsilon \ > \ 0 .$ , we say that $\mathcal { F } \ \varepsilon { - } s h a t t e r s$ a set $\{ x _ { 1 } , \ldots , x _ { n } \} \subseteq { \mathcal { X } }$ if there exist witnesses $y _ { 1 : n } ~ \in ~ \mathbb { R } ^ { n }$ such that for every $\sigma _ { 1 : n } ~ \in ~ \{ \pm 1 \} ^ { n }$ there exists $f \in { \mathcal { F } }$ with

$$
\sigma _ { i } \left( f ( x _ { i } ) - y _ { i } \right) \geq \varepsilon \qquad { \mathrm { f o r ~ a l l ~ } } i \in [ n ] .
$$

The fat-shattering dimension of F at scale ε, denoted fa ${ , \mathrm { t } _ { \varepsilon } ( \mathcal { F } ) }$ , is the largest n such that F ε-shatters some set of size n (Kearns and Schapire, 1994; Alon et al., 1997). The map $\varepsilon \mapsto \mathrm { f a t } _ { \varepsilon } ( \mathcal { F } )$ is nonincreasing, and for binary classes $\mathcal { F } \subseteq \{ \pm 1 \} ^ { \mathcal { X } } , \mathrm { f a t } _ { \varepsilon } ( \mathcal { F } )$ is equal to the VC dimension of the class for all $\varepsilon \in ( 0 , 1 ]$

ERM oracle model. Following Hazan and Koren (2016) and the literature on oracle-eficient smoothed online learning (Block et al., 2022; Haghtalab et al., 2022), we assume access to a weighted agnostic ERM oracle. In particular, for any n covariates $X _ { 1 : n } ,$ , a sequence $\ell _ { i } \colon [ - 1 , 1 ] \to [ - 1 , 1 ]$ of convex functions, each given either by label loss (that is, $\ell _ { i } ( \cdot ) = \ell ( \cdot , y _ { i } )$ for some label $y _ { i } )$ or a linear function, and weights $w _ { i } \geq 0$ , we assume we can compute a solution to the following optimization problem:

$$
\underset { f \in \mathcal { F } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { n } w _ { i } \ell _ { i } \big ( f ( X _ { i } ) \big ) .
$$

We note that weights in our analysis satisfy $\| w \| _ { \infty } \lesssim \sqrt { \log ( T ) }$ w.h.p.

Related work. Due to its robustness and fundamental nature, online learning has been extensively studied over the past decades with a relatively complete understanding of its theoretical foundations (Freund and Schapire, 1997; Ben-David et al., 2009; Cesa-Bianchi and Lugosi, 2006; Shalev-Shwartz, 2012; Rakhlin et al., 2011; Block et al., 2021) and computational properties (Hazan and Koren, 2016; Kalai and Vempala, 2005; Abernethy et al., 2014; Rakhlin et al., 2012). On the other hand, because of the strong statistical and computational lower bounds, more recently a number of works have explored online learning under various relaxations and assumptions to achieve tractable algorithms and improved performance guarantees (Rakhlin et al., 2011). Of particular relevance are prior works (Haghtalab et al., 2020; Block et al., 2022; Haghtalab et al., 2022; Block and Polyanskiy, 2023; Block et al., 2024a,b) that focus on the statistical and computational advantages of smoothed online learning under the assumption that the base measure is known, as well as applications to other domains like game theory (Daskalakis et al., 2024). Other work sought to extend these results, either by taking advantage of additional structure for specific base measures (Block et al., 2023b,a; Block and Simchowitz, 2022), or considering more general loss functions (Bhatt et al., 2023).

In all of the above work relevant to smoothed online learning, some knowledge of the base measure was required, which limits the applicability of the results. A separate line of work has sought to develop methods that do not rely on knowing the base measure, thereby broadening the scope and applicability of smoothed online learning techniques (Wu et al., 2023, 2024), although the former work focuses on oracle-ineficient algorithms under substantially stronger assumptions whereas the latter provides oracle-eficient solutions assuming the features are independent. In Block et al. (2024b), the authors demonstrate that it is possible to achieve strong performance guarantees even without knowledge of the base measure using just empirical risk minimization, but they require the labels to be well-specified in the sense that their conditional means can be perfectly predicted by an element of the comparator class. On the other end of the spectrum, Blanchard (2025) consider the agnostic setting where no such well-specification assumption is made, but do not consider oracle-eficient algorithms. Unlike these prior works, ours is the first to propose an oracle-eficient algorithm (similar in spirit to one in Block et al. (2022)) that is capable of achieving sublinear regret under an arbitrary unknown base measure; moreover, our algorithm’s regret scales optimally in both time and smoothness, losing only a $\sqrt { d }$ factor compared to the optimal rate.

## 3 Main Results

The main results of our paper are two oracle-eficient regret bounds, for the binary and the realvalued loss cases. We begin with the binary case. Recall that Gaussian FTPL as defined in Eq. (3), in each round $t ,$ computes:

$$
\hat { f } _ { t } \in \underset { f \in \mathcal { F } } { \arg \operatorname* { m i n } } \sum _ { s = 1 } ^ { t - 1 } \left[ \ell ( f ( X _ { s } ) , Y _ { s } ) + G _ { s } ^ { ( t ) } f ( X _ { s } ) \right] ,
$$

and predicts $\hat { Y } _ { t } = \hat { f } _ { t } ( X _ { t } )$ . Note that the above can be implemented using one call to the weighted ERM oracle. For this algorithm, we obtain the following regret guarantee.

Theorem 3.1. Consider a binary class $\mathcal { F } \colon \mathcal { X } \to \{ \pm 1 \}$ of VC dimension $d \geq 1$ , and let $\ell ( \hat { y } , y ) = - \hat { y } y$ be the binary loss. For every horizon $T \geq 1$ and every covariate process that is σ-smooth with respect to a fixed $\mu ,$ the learner in (3) satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } \lesssim d \log ^ { 2 } ( 2 T / \sigma ) \sqrt { \frac { T } { \sigma } } .
$$

A complete proof of the above statement is given in Section B. Note that the regret bound above is obtained in a completely parameter-free manner. In particular, the learner in (3) requires no knowledge of the base measure $\mu ,$ the smoothness parameter $\sigma ,$ or the horizon $T .$

Extending the above guarantee to convex Lipschitz losses requires an additional averaging step, which, intuitively, helps reduce the variance of the predictor. Formally, in round t, our learner resamples the noise $G ^ { ( t ) }$ independently t times, and computes (conditionally) independent solutions $\hat { f } _ { t } ^ { ( i ) } , i \in [ t ]$ to the optimization problem in Eq. (3). The prediction is then aggregated in the following way:

$$
\hat { Y } _ { t } = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \hat { f } _ { t } ^ { ( i ) } ( X _ { t } ) .\tag{7}
$$

The above can be recognized as a Monte-Carlo estimate of the expectation $\mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } )$ . By Jensen’s inequality $\ell ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) , Y _ { t } ) \leq \mathbb { E } _ { G ^ { ( t ) } } \ell ( \hat { f } _ { t } ( X _ { t } ) , Y _ { t } )$ , so, up to the Monte Carlo error, the averaged prediction incurs no more loss than a single Gaussian FTPL prediction in conditional expectation.

Theorem 3.2. Consider a class $\mathcal { F } \colon \mathcal { X }  [ - 1 , 1 ]$ and let $\ell ( \hat { y } , y )$ be convex and 1-Lipschitz in its first argument, and suppose the covariate process is σ-smooth with respect to a fixed $\mu .$ Then, the learner in (7) satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } \lesssim \operatorname* { i n f } _ { 0 < \varepsilon \leq 1 / 1 6 } \left\{ \left( 1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) + \sqrt { T } \varepsilon \right) \sqrt { \frac { T } { \sigma } } \log ^ { 3 } \left( \frac { 2 T } { \sigma \varepsilon } \right) \right\} ,\tag{8}
$$

where c is a universal constant.

A complete proof of the above statement is given in Section C. Theorem 3.2 gives explicit rates for two important families of function classes. For parametric classes, i.e., those satisfying $\mathrm { f a t } _ { \varepsilon } ( \mathcal { F } ) \lesssim d \log ( 1 / \varepsilon )$ at suficiently small scales, we have

$$
\mathbb { E } \mathrm { R e g } _ { T } \leq \widetilde { O } \left( d \sqrt { \frac { T } { \sigma } } \right) .
$$

As in binary classification, the dependence on d is suboptimal by a factor of ${ \sqrt { d } } .$ , up to logarithmic factors (Blanchard, 2025). For nonparametric classes, i.e., those satisfying $\mathrm { f a t } _ { \varepsilon } ( \mathcal { F } ) \lesssim \varepsilon ^ { - \alpha } , \alpha > 0$ choosing $\varepsilon \asymp _ { \alpha } T ^ { - 1 / ( 2 \alpha + 2 ) }$ gives

$$
\mathbb { E } \mathrm { R e g } _ { T } \leq \widetilde { O } _ { \alpha } \left( \frac { T ^ { ( 2 \alpha + 1 ) / ( 2 \alpha + 2 ) } } { \sqrt { \sigma } } \right) .
$$

For fixed $\sigma ,$ this is sublinear for every $\alpha > 0$ . It matches the leading horizon and smoothness exponents in Blanchard (2025, Theorem 4), which achieves this guarantee with an ineficient algorithm. Notably, our result applies to regret as defined in Eq. (1), while Blanchard (2025) only controls pseudo-regret (where comparator is fixed before the draw of the data). Compared with the i.i.d. setting, where the optimal regret rate scales as $T ^ { \frac { 1 } { 2 } \vee \frac { \alpha - 1 } { \alpha } }$ (Mendelson and Vershynin, 2003), up to logarithmic factors, our bound has a larger exponent of T. Nevertheless, sublinear regret remains possible for every $\alpha > 0$ . We thus present a framework for smoothed online learning that recovers the qualitative picture of statistical learnability, while retaining oracle eficiency and requiring no knowledge of the base measure.

## 4 Analysis Techniques

We give an overview of the proof of Theorem 3.1 in the binary case, as it is simpler and still contains the key ideas needed for the general case. The complete proofs for the binary case are deferred to Section B, and the proofs for the real-valued case are contained in Appendix C. Our proof has three ingredients: a sharper surprise lemma (Section 4.1), a reduction to exponential weights (Section 4.2), and a regret bound for Gaussian-perturbed exponential weights (Section 4.3).

## 4.1 Surprise Lemma

For smoothed online learning algorithms that aim to extrapolate behavior over the past into the future, surprise lemmas have proven useful (Block et al., 2024b; Blanchard, 2025). Intuitively, for an arbitrary sequence of (predictable) test functions $\{ h _ { t } \}$ , a surprise lemma quantifies for how many t the value $h _ { t } ( X _ { t } )$ can be significantly larger than the past values $h _ { t } ( X _ { 1 } ) , \ldots , h _ { t } ( X _ { t - 1 } )$ . Formally, we prove the following version of the surprise lemma, which controls expected absolute values of test functions as a function of their past second moments.

Lemma 4.1. Let $\mathcal { F }$ be a class of VC dimension d and let $\{ h _ { t } \} _ { t \in [ T ] }$ be a predictable sequence of functions with $h _ { t } \in \mathrm { c o n v } ( \pm \mathcal { F } )$ . Then

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } \left| h _ { t } ( X _ { t } ) \right| \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } } + \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } \left( \frac { 2 T } { \sigma } \right) .\tag{9}
$$

The above is proven in two steps. We first use a result from Xie et al. (2022, Section 3.2, equations $\left( 3 \right) - \left( 5 \right) )$ ), which establishes an analogous estimate where the second moments are evaluated on a tangent sequence $\bar { X } _ { 1 : T }$ as opposed to realized covariates $X _ { 1 : T }$ , where a (decoupled) tangent sequence $\bar { X } _ { 1 : T }$ is defined by drawing independent samples $\bar { X } _ { t } \mid \mathcal { H } _ { t - 1 } \sim p _ { t }$ for each t. Compared to the analogous statement in Block et al. (2024b, Lemma 12), which involves first moments of the test functions, keeping second moments allows for a sharper transfer from past to current covariates. The next step is to extend the norm comparison inequality of (Block et al., 2024b, Theorem 10) to convex hulls of VC classes, for which we utilize Maurey’s empirical method (Pisier, 1981). It is important to allow $h _ { t } \in \mathrm { c o n v } ( \pm \mathcal { F } )$ as we wish to apply Lemma 4.1 to functions of predictions of exponential weights, which live in the convex hull of the class. The full proof is in Appendix B.1.

## 4.2 Reduction to exponential weights

In this section, we show that predictions of our Gaussian FTPL algorithm are close to predictions of an auxiliary exponential weights algorithm that is run over the cover $\mathcal { G }$ of ${ \mathcal F } .$ . Prior work establishes existence of covers of controlled size that approximate $\mathcal { F }$ well along a draw of a smoothed covariate trajectory, using the coupling lemma (Haghtalab et al., 2024).

Proposition 4.2. Let $\mathcal { F } \colon \mathcal { X } \to \{ \pm 1 \}$ be a binary class with VC dimension d. For every probability measure $\mu ,$ horizon $T \geq 1$ , and $\sigma \in ( 0 , 1 ]$ , there exists a cover ${ \mathcal { G } } \subseteq { \mathcal { F } }$ such that log $| \mathcal { G } | \lesssim d \log ( 2 T / \sigma )$ and, for every covariate process that is σ-smooth with respect to $\mu$

$$
\mathbb { P } \left( \operatorname* { s u p } _ { f \in \mathcal { F } } \operatorname* { i n f } _ { g \in \mathcal { G } } \sum _ { s \leq T } \mathbb { 1 } \left\{ f ( X _ { s } ) \neq g ( X _ { s } ) \right\} > C d \log ( 2 T / \sigma ) \right) \leq \frac { 1 } { T } ,
$$

where $C > 0$ is a universal constant.

Recall that in Eq. (4) we define an auxiliary exponential weights learner using the cover $\mathcal { G }$ from Proposition 4.2:

$$
\tilde { f } _ { t } : = \sum _ { f \in \mathcal { G } } w _ { t } ( f \mid G ^ { ( t ) } ) f , \quad \mathrm { w i t h } \quad w _ { t } ( f \mid G ^ { ( t ) } ) \propto \exp \left( - \sum _ { s < t } [ \ell ( f ( X _ { s } ) , Y _ { s } ) + G _ { s } ^ { ( t ) } f ( X _ { s } ) ] \right) .
$$

We show that predictions of such an algorithm are close on average to the predictions of our Gaussian FTPL learner as in Eq. (3).

Lemma 4.3 (Reduction to exponential weights). Consider a binary function class $\mathcal { F } \colon \mathcal { X } \to \{ \pm 1 \}$ of VC dimension d. For $\hat { f } _ { t }$ and $\tilde { f } _ { t }$ as defined in Eqs. (3) and $( 4 )$ , respectively, we have the following bound:

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } | \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) | \lesssim d \log ^ { 2 } ( 2 T / \sigma ) \sqrt { \frac { T } { \sigma } } .\tag{10}
$$

Since binary loss $\ell ( \hat { y } , y ) = - \hat { y } y$ is Lipschitz, the above immediately implies

$$
\mathbb { E } \mathrm { R e g } _ { T } - \mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { E W } } \lesssim d \log ^ { 2 } ( 2 T / \sigma ) \sqrt { \frac { T } { \sigma } } .
$$

We prove Eq. (10) in two steps. First, we show that $\hat { f } _ { t }$ and $\tilde { f } _ { t }$ agree on past covariates (when averaged over their Gaussian randomness). Then, using predictability of $\hat { f } _ { t }$ and $\tilde { f } _ { t }$ , we leverage the surprise lemma (Lemma 4.1) to transfer that guarantee into closeness on current covariates. In particular, we obtain the following estimate.

Lemma 4.4. For every $t \leq T$

$$
\mathbb { E } \sum _ { s < t } \Big ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { s } ) \Big ) ^ { 2 } \lesssim d ^ { 2 } \log ^ { 2 } ( 2 T / \sigma ) .
$$

It shows that the Gaussian-averaged predictions $\mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t }$ and $\mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t }$ closely agree on the past data points in expected squared norm.

Proof sketch of Lemma $4 { \cdot } 4 .$ . The case $t = 1$ is immediate. For $t \geq 2$ , fix the history $( X _ { 1 : t - 1 } , Y _ { 1 : t - 1 } )$ write $G = G ^ { ( t ) }$ , and introduce two potentials:

$$
\Phi _ { t } ( G ) = \operatorname* { m a x } _ { f \in \mathcal { F } } \sum _ { s < t } ( Y _ { s } - G _ { s } ) f ( X _ { s } ) ,
$$

$$
\widetilde { \Phi } _ { t } ( G ) = \log \sum _ { f \in \mathcal { G } } \exp \left( \sum _ { s < t } ( Y _ { s } - G _ { s } ) f ( X _ { s } ) \right) = \operatorname* { m a x } _ { w \in \Delta ( \mathcal { G } ) } \left[ \mathbb { E } _ { f \sim w } \sum _ { s < t } ( Y _ { s } - G _ { s } ) f ( X _ { s } ) + H ( w ) \right] ,
$$

where $\begin{array} { r } { H ( w ) = \sum _ { f \in { \mathcal { G } } } w ( f ) \log ( 1 / w ( f ) ) } \end{array}$ denotes entropy. Then, almost surely,<sup>2</sup> we have $\hat { f } _ { t } ( X _ { s } ) =$ $ - \partial _ { G _ { s } } \Phi _ { t } ( G )$ and $\tilde { f } _ { t } ( X _ { s } ) = - \partial _ { G _ { s } } \widetilde { \Phi } _ { t } ( G )$ for every $s < t .$ . These identities follow from Danskin’s theorem for the finite maximum (Bertsekas, 2016, Appendix B), and using the fact that $w _ { t }$ is the maximizer in the variational formula for $\tilde { \Phi } _ { t } ;$ in the context of online learning, this idea also appears in Abernethy et al. (2014). Thus,

$$
\begin{array} { r l } { \displaystyle \sum _ { s < \ell } ( \mathbb { E } _ { G } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G } \hat { f } _ { t } ( X _ { s } ) ) ^ { 2 } = \left\| \mathbb { E } _ { G } \nabla _ { G } ( \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) ) \right\| _ { 2 } ^ { 2 } } & { } \\ & { \overset { ( a ) } { = } \left\| \mathbb { E } _ { G } [ G ( \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) ) ] \right\| _ { 2 } ^ { 2 } } \\ & { = \left\| \mathbb { E } _ { G } \left[ G \left( \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) - \mathbb { E } _ { G } [ \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) ] \right) \right] \right\| _ { 2 } ^ { 2 } } \\ & { = \displaystyle \operatorname* { s u p } _ { \| u \| _ { 2 } = 1 } \mathrm { C o v } _ { G } ( u ^ { T } G , \Phi _ { t } - \widetilde { \Phi } _ { t } ) ^ { 2 } } \\ & { \overset { ( b ) } { \leq } \displaystyle \operatorname* { s u p } _ { \| u \| _ { 2 } = 1 } \frac { \mathbb { E } _ { G } [ ( u ^ { T } G ) ^ { 2 } ] \operatorname { V a r } _ { G } ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) } { \| u \| _ { 2 } } } \\ & { = \operatorname* { s u r c } _ { G } ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) , } \end{array}
$$

where (a) uses Stein’s lemma (Lemma A.4) and (b) uses the Cauchy–Schwarz inequality. It remains to control $\Phi _ { t } - \widetilde { \Phi } _ { t }$ . The two potentials difer in that $( 1 ) \Phi _ { t }$ maximizes over $\mathcal { F }$ rather than $\mathcal { G } _ { : }$ which Proposition 4.2 controls, and $( 2 ) \ \widetilde { \Phi } _ { t }$ adds an entropy term, which costs at most log |G|. Combining these, we obtain:

$$
\begin{array} { r } { \mathrm { V a r } _ { G } ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) \le \mathbb { E } _ { G } | \widetilde { \Phi } _ { t } - \Phi _ { t } | ^ { 2 } \lesssim d ^ { 2 } \log ^ { 2 } ( 2 T / \sigma ) + \log ^ { 2 } | \mathcal { G } | \lesssim d ^ { 2 } \log ^ { 2 } ( 2 T / \sigma ) , } \end{array}
$$

as desired.

An application of the surprise lemma (Lemma 4.1) to Lemma 4.4 finishes the proof of Lemma 4.3. The full proof is in Appendix B.3.

## 4.3 Regret of Gaussian-perturbed exponential weights

It remains to bound the regret of the exponential weights algorithm. In particular, we show the following upper bound.

Lemma 4.5. The regret of the exponential weights algorithm in (4) satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { E W } } \lesssim \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) .\tag{11}
$$

The proof has three steps. First, we adapt the variance-based analysis of exponential weights (Cesa-Bianchi et al., 2007, Lemmas 3 and 4), using the Bernstein bound of De Rooij et al. (2014, Lemma 4) to handle the Gaussian-perturbed losses. This results in the following claim.

Proposition 4.6. Let ${ \mathcal { G } } \subseteq { \mathcal { F } }$ be a nonempty finite class, and let $\ell ( \hat { y } , y ) = - y \hat { y }$ be the binary loss. For every data sequence, the predictor $\tilde { f } _ { t }$ defined in (4) satisfies the following inequality,

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \ell ( \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \operatorname* { m i n } _ { f \in \mathcal { G } } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \lesssim \log | \mathcal { G } | + \sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w t ( \cdot | G ^ { ( t ) } ) } ( f ( X _ { t } ) ) .\tag{12}
$$

By Proposition 4.2, the comparator losses over the entire class $\mathcal { F }$ and over the cover $\mathcal { G }$ are close, so it remains to bound the cumulative variance on the right-hand side of (12). We now show that the Gaussian perturbations alone keep these variances small on the covariates from the past, even though the learning rate is constant.

Lemma 4.7. Let $\mathcal { G }$ be a nonempty finite class of functions $\mathcal { X }  [ - 1 , 1 ]$ and let ℓ be any loss. For every $t \leq T$ and every fixed history $( X _ { s } , Y _ { s } ) _ { s < t }$ , the weights in (4) satisfy

$$
\sum _ { s < t } \Big ( \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( f ( X _ { s } ) ) \Big ) ^ { 2 } \lesssim \log ( 2 \vert \mathcal { G } \vert ) .\tag{13}
$$

The proof, given in Appendix B.4, combines Stein’s lemma (Lemma A.4) with a Gaussian maximal inequality. Finally, we transfer this bound to the current covariates. For binary predictions, the variance can be written as

$$
\frac 1 2 \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( f ( x ) ) = \mathbb { E } _ { G ^ { ( t ) } } \sum _ { f , g \in \mathcal { G } } w _ { t } ( f \mid G ^ { ( t ) } ) w _ { t } ( g \mid G ^ { ( t ) } ) \mathbb { 1 } \{ f ( x ) \neq g ( x ) \} .
$$

This is a predictable convex mixture over the disagreement class $\{ \mathbb { 1 } \{ f \neq g \} : f , g \in { \mathcal { F } } \}$ , whose VC dimension is $O ( d )$ (see Appendix B.4). Applying Lemma 4.1 to this mixture and summing (13) over $t \leq T$ gives

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G ^ { ( t ) } ) } ( f ( X _ { t } ) ) \lesssim \sqrt { \frac { T \log | \mathcal { G } | \log ( 2 T ) } { \sigma } } + \sqrt { \frac { ( d + 1 ) T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) .
$$

Substituting into (12) and adding the cover approximation cost gives Lemma 4.5. Finally, Theorem 3.1 follows by combining Lemmas 4.3 and 4.5.

## 5 Conclusion and Discussion

Our results show oracle-eficient learning from smooth data is possible even when the data are agnostic and the base measure unknown. Moreover, the algorithm that we propose is close to the best that could be hoped for: entirely parameter free, requiring no knowledge of the base measure, the smoothness parameter, or the horizon, and conceptually simple. Modulo the remaining $\sqrt { d }$ gap between that which we have shown is achievable by our algorithm and the statistical optimum, our work essentially closes the question of oracle-eficient learning from smooth data.

Concurrent work. In independent and concurrent work, Buzaglo and Hazan (2026) establish an oracle-eficient regret bound of $\widetilde { O } ( \sqrt { d T } )$ for binary classification with i.i.d. covariates and adversarial labels (which corresponds to the $\sigma = 1$ case of our setting). They analyze the same form of Gaussian FTPL, with a horizon-dependent perturbation scale. Their analysis controls the mass of disagreement regions of nearly leading hypotheses (that is, the sets of covariates on which at least two near-leaders disagree) through a generalization argument that relies heavily on the i.i.d. assumption. General σ-smoothed processes require additional care, since it still permits intricate dependencies between covariates, which prevents such a generalization argument to extend directly. A natural attempt would be to apply our surprise lemma (Lemma 4.1) to the disagreement indicators. However, the family of disagreement regions can have infinite VC dimension even when the underlying hypothesis class has VC dimension one (e.g., for the “star class” (Hanneke, 2024)).

## Statement on the use of AI

In this work, we used generative AI tools to help develop theoretical arguments, formulate and refine mathematical claims and hypotheses, propose proof ingredients, write and revise proofs, and interpret theoretical results. Over July and August of 2026, the authors established a $\widetilde { \cal O } ( \overline { { { T ^ { 3 / 4 } d ^ { 1 / 4 } } } } / \sqrt { \sigma } )$ regret bound for an adaptation of the algorithm of Wu et al. (2024), extending its original guarantee for i.i.d. covariates $( \sigma = 1 )$ to the general σ-smooth setting. The proof analyzed the stability of Rademacher potentials with perturbations on subsampled past covariates, using an informationtheoretic decoupling argument to control the efect of dependencies among covariates. The proof was developed and refined through iterative work between the authors and GPT 5.6 Sol. On September 6, 2026, GPT 6 Astra subsequently proposed the modification to Gaussian FTPL used in the present manuscript. The corresponding proof was completed through further iterations with the model. We additionally used GPT 6 Astra, GPT 5.6 Sol, and Claude Opus 5.5 to identify and compare related literature, draft and edit manuscript text, improve exposition and organization, and edit LaTeX. All mathematical claims in the present manuscript were independently verified by the authors. The authors take responsibility for the final manuscript, including all text, claims, and proofs produced with AI assistance.

## Acknowledgments

The authors acknowledge support from the NSF through awards DMS-2031883 and PHY-2019786, the DARPA AIQ program, and AFOSR FA9550-25-1-0375. SV acknowledges support from Amazon AI Research Innovation Fellowship.

## References

Jacob Abernethy, Chansoo Lee, Abhinav Sinha, and Ambuj Tewari. Online linear optimization via smoothing. In Conference on learning theory, pages 807–823. PMLR, 2014.

Jacob D Abernethy, Elad Hazan, and Alexander Rakhlin. Competing in the dark: An eficient algorithm for bandit linear optimization. In Conference on Learning Theory, 2008.

Noga Alon, Shai Ben-David, Nicolo Cesa-Bianchi, and David Haussler. Scale-sensitive dimensions, uniform convergence, and learnability. Journal of the ACM (JACM), 44(4):615–631, 1997.

Ingo Alth¨ofer. On sparse approximations to randomized strategies and convex combinations. Linear Algebra and its Applications, 199:339–355, 1994.

Shai Ben-David, D´avid P´al, and Shai Shalev-Shwartz. Agnostic online learning. In Conference on Learning Theory, 2009.

Dimitri P. Bertsekas. Nonlinear Programming. Athena Scientific, 3 edition, 2016.

Alankrita Bhatt, Nika Haghtalab, and Abhishek Shetty. Smoothed analysis of sequential probability assignment. Advances in Neural Information Processing Systems, 36:79808–79831, 2023.

Mo¨ıse Blanchard. Agnostic smoothed online learning. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing, pages 1997–2006, 2025.

Mo¨ıse Blanchard, Abhishek Shetty, and Alexander Rakhlin. Characterizing online and private learnability under distributional constraints via generalized smoothness. In Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 729–759. PMLR, 2026.

Adam Block and Yury Polyanskiy. The sample complexity of approximate rejection sampling with applications to smoothed online learning. In The Thirty Sixth Annual Conference on Learning Theory, pages 228–273. PMLR, 2023.

Adam Block and Max Simchowitz. Eficient and near-optimal smoothed online learning for generalized linear functions. Advances in neural information processing systems, 35:7477–7489, 2022.

Adam Block, Yuval Dagan, and Alexander Rakhlin. Majorizing measures, sequential complexities, and online learning. In Proceedings of Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 587–590. PMLR, 2021.

Adam Block, Yuval Dagan, Noah Golowich, and Alexander Rakhlin. Smoothed online learning is as easy as statistical learning. In Conference on Learning Theory, pages 1716–1786. PMLR, 2022.

Adam Block, Max Simchowitz, and Alexander Rakhlin. Oracle-eficient smoothed online learning for piecewise continuous decision making. In The Thirty Sixth Annual Conference on Learning Theory, pages 1618–1665. PMLR, 2023a.

Adam Block, Max Simchowitz, and Russ Tedrake. Smoothed online learning for prediction in piecewise afine systems. Advances in Neural Information Processing Systems, 36:41663–41674, 2023b.

Adam Block, Mark Bun, Rathin Desai, Abhishek Shetty, and Zhiwei S Wu. Oracle-eficient diferentially private learning with public data. Advances in Neural Information Processing Systems, 37:113191–113233, 2024a.

Adam Block, Alexander Rakhlin, and Abhishek Shetty. On the performance of empirical risk minimization with smoothed data. In The Thirty Seventh Annual Conference on Learning Theory, pages 596–629. PMLR, 2024b.

Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K Warmuth. Learnability and the Vapnik-Chervonenkis dimension. Journal of the ACM (JACM), 36(4):929–965, 1989.

St´ephane Boucheron, G´abor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 02 2013. doi: 10.1093/acprof: oso/9780199535255.001.0001.

Gon Buzaglo and Elad Hazan. Oracle-eficient online classification with stochastic inputs and adversarial outputs. arXiv preprint arXiv:2609.33760, 2026.

Nicolo Cesa-Bianchi and G´abor Lugosi. Prediction, learning, and games. Cambridge university press Cambridge, 2006.

Nicolo Cesa-Bianchi, Yishay Mansour, and Gilles Stoltz. Improved second-order bounds for prediction with expert advice. Machine Learning, 2007.

Allison JB Chaney, Brandon M Stewart, and Barbara E Engelhardt. How algorithmic confounding in recommendation systems increases homogeneity and decreases utility. In Proceedings of the 12th ACM conference on recommender systems, pages 224–232, 2018.

Constantinos Daskalakis, Noah Golowich, Nika Haghtalab, and Abhishek Shetty. Smooth Nash Equilibria: Algorithms and Complexity. In 15th Innovations in Theoretical Computer Science Conference (ITCS 2024), volume 287 of Leibniz International Proceedings in Informatics (LIPIcs), pages 37:1–37:22, 2024.

Steven De Rooij, Tim Van Erven, Peter D Gr¨unwald, and Wouter M Koolen. Follow the leader if you can, hedge if you must. The Journal of Machine Learning Research, 15(1):1281–1316, 2014.

Luc Devroye, G´abor Lugosi, and Gergely Neu. Prediction by random-walk perturbation. In Conference on Learning Theory, pages 460–473. PMLR, 2013.

Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.

Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1):119–139, 1997.

Nika Haghtalab, Tim Roughgarden, and Abhishek Shetty. Smoothed analysis of online and diferentially private learning. Advances in Neural Information Processing Systems, 33:9203–9215, 2020.

Nika Haghtalab, Yanjun Han, Abhishek Shetty, and Kunhe Yang. Oracle-eficient online learning for smoothed adversaries. Advances in Neural Information Processing Systems, 35:4072–4084, 2022.

Nika Haghtalab, Tim Roughgarden, and Abhishek Shetty. Smoothed analysis with adaptive adversaries. Journal of the ACM, 71(3):1–34, 2024.

James Hannan. Approximation to Bayes risk in repeated play. In Contributions to the Theory of Games, Volume III, Annals of Mathematics Studies. Princeton University Press, 1957.

Steve Hanneke. The star number and eluder dimension: Elementary observations about the dimensions of disagreement. In The Thirty Seventh Annual Conference on Learning Theory, pages 2308–2359. PMLR, 2024.

David Haussler. Sphere packing numbers for subsets of the boolean n-cube with bounded Vapnik-Chervonenkis dimension. Journal of Combinatorial Theory, Series A, 69(2):217–232, 1995.

Elad Hazan and Tomer Koren. The computational power of optimization in online learning. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pages 128–141, 2016.

Marcus Hutter and Jan Poland. Adaptive online prediction by following the perturbed leader. Journal of Machine Learning Research, 6(22):639–660, 2005.

Ray Jiang, Silvia Chiappa, Tor Lattimore, Andr´as Gy¨orgy, and Pushmeet Kohli. Degenerate feedback loops in recommender systems. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pages 383–390, 2019.

Adam Kalai and Santosh Vempala. Eficient algorithms for online decision problems. Journal of Computer and System Sciences, 71(3):291–307, 2005.

Michael J Kearns and Robert E Schapire. Eficient distribution-free learning of probabilistic concepts. Journal of Computer and System Sciences, 48(3):464–497, 1994.

Tor Lattimore and Csaba Szepesv´ari. Bandit Algorithms. Cambridge University Press, 2020.

Tengyuan Liang, Alexander Rakhlin, and Karthik Sridharan. Learning with square loss: Localization through ofset rademacher complexity. In Conference on Learning Theory, pages 1260–1285. PMLR, 2015.

Nick Littlestone. Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm. Machine learning, 2(4):285–318, 1988.

Nick Littlestone and Manfred K Warmuth. The weighted majority algorithm. Information and computation, 108(2):212–261, 1994.

H Brendan McMahan. A survey of algorithms and analysis for adaptive online learning. Journal of Machine Learning Research, 18(90):1–50, 2017.

Shahar Mendelson and Roman Vershynin. Entropy and the combinatorial dimension. Inventiones mathematicae, 152(1):37–55, 2003.

Gilles Pisier. Remarques sur un r´esultat non publi´e de B. Maurey. S´eminaire d’analyse fonctionnelle, pages 1–12, 1981.

Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Online learning: Stochastic, constrained, and smoothed adversaries. Advances in neural information processing systems, 24, 2011.

Sasha Rakhlin, Ohad Shamir, and Karthik Sridharan. Relax and randomize: From value to algorithms. Advances in Neural Information Processing Systems, 25, 2012.

Norbert Sauer. On the density of families of sets. Journal of Combinatorial Theory, Series A, 13(1): 145–147, 1972.

Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends in Machine Learning, 4(2):107–194, 2012. ISSN 1935-8237. doi: 10.1561/2200000018.

Shai Shalev-Shwartz and Yoram Singer. A primal-dual perspective of online learning algorithms. Machine Learning, 2007.

Saharon Shelah. A combinatorial problem; stability and order for models and theories in infinitary languages. Pacific Journal of Mathematics, 41(1):247–261, 1972.

Charles M Stein. Estimation of the mean of a multivariate normal distribution. The annals of Statistics, pages 1135–1151, 1981.

L. G. Valiant. A theory of the learnable. Communications of the ACM, 27(11):1134–1142, 1984.

V. N. Vapnik and A. Ya. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability & Its Applications, 16(2):264–280, 1971. doi: 10.1137/1116025.

Volodimir G Vovk. Aggregating strategies. In Proceedings of the third annual workshop on Computational learning theory, 1990.

Changlong Wu, Ananth Grama, and Wojciech Szpankowski. Online learning in dynamically changing environments. In Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 325–358. PMLR, 2023.

Changlong Wu, Jin Sima, and Wojciech Szpankowski. Oracle-eficient hybrid online learning with unknown distribution. In The Thirty Seventh Annual Conference on Learning Theory, pages 4992–5018. PMLR, 2024.

Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. arXiv preprint arXiv:2210.04157, 2022.

## A Additional preliminaries

We collect preliminaries for smoothed online learning, variance-based regret bounds for exponential weights, Gaussian integration by parts, and Maurey’s empirical method.

## A.1 Covers under smoothed processes

VC and fat-shattering dimensions are intimately related to covers. In a proposition below, we collect bounds that control sizes of covers in terms of class’s VC or fat-shattering dimension. The bounds on projections follow from Sauer (1972); Shelah (1972); Alon et al. (1997). $L ^ { 2 } ( \mu )$ versions follow from Haussler (1995); Mendelson and Vershynin (2003).

Proposition A.1. There are universal constants $0 < c < 1$ and $C > 1$ such that the following hold for every $\mathcal { F } \colon \mathcal { X }  [ - 1 , 1 ]$ , and $\varepsilon \in ( 0 , 1 ]$

(i) For any set of n points $z _ { 1 : n }$ , there exist a finite cover $\mathcal { G } \subset \mathcal { F }$ such that:

$$
\log | \mathcal { G } | \leq C \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } ( C n / \varepsilon ) , \qquad \operatorname* { s u p } _ { f \in \mathcal { F } } \operatorname* { m i n } _ { g \in \mathcal { G } } \operatorname* { m a x } _ { i \in [ n ] } | f ( z _ { i } ) - g ( z _ { i } ) | \leq \varepsilon .
$$

$I f { \mathcal { F } }$ is binary with VC dimension $d ,$ the cover can be chosen with log $| { \mathcal { G } } | \leq d \log ( C n )$ and, for every $f \in { \mathcal { F } }$ , there exists $g \in { \mathcal { G } }$ such that $f ( z _ { i } ) = g ( z _ { i } )$ for all $i \in [ n ]$

(ii) For every probability measure $\mu$ on $\mathcal { X } _ { i }$ , there is a finite subset $\mathcal { G } \subset \mathcal { F }$ such that

$$
\log | \mathcal { G } | \leq C \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ( C / \varepsilon ) , \qquad \operatorname* { s u p } _ { f \in \mathcal { F } } \operatorname* { m i n } _ { g \in \mathcal { G } } \| f - g \| _ { L ^ { 2 } ( \mu ) } \leq \varepsilon .
$$

$I f { \mathcal { F } }$ is binary, we have log $| \mathcal { G } | \leq C d \log ( C / \varepsilon )$ and the same approximation guarantee holds.

It is well-established in the smoothed online learning literature that covers w.r.t. $L ^ { 1 } ( \mu )$ norm provide approximation guarantees along the draw of a smoothed trajectory (Haghtalab et al., 2024). In particular, the following proposition follows using the coupling construction of Haghtalab et al. (2024, Theorem 2.1 and Appendix B.2).

Proposition 4.2. Let $\mathcal { F } \colon \mathcal { X } \to \{ \pm 1 \}$ be a binary class with VC dimension d. For every probability measure $\mu ,$ horizon $T \geq 1$ , and $\sigma \in ( 0 , 1 ]$ , there exists a cover ${ \mathcal { G } } \subseteq { \mathcal { F } }$ such that log $| \mathcal { G } | \lesssim d \log ( 2 T / \sigma )$ and, for every covariate process that is σ-smooth with respect to $\mu$

$$
\mathbb { P } \left( \operatorname* { s u p } _ { f \in \mathcal { F } } \operatorname* { i n f } _ { g \in \mathcal { G } } \sum _ { s \leq T } \mathbb { 1 } \left\{ f ( X _ { s } ) \neq g ( X _ { s } ) \right\} > C d \log ( 2 T / \sigma ) \right) \leq \frac { 1 } { T } ,
$$

where $C > 0$ is a universal constant.

Proof of Proposition 4.2. Set $\delta = 1 / ( 2 T )$ and $n = \lceil \sigma ^ { - 1 } \log ( T / \delta ) \rceil$ ⌉. The recursive coupling construction of Haghtalab et al. (2024, Theorem 2.1 and Appendix B.2) implies existence of a coupling between the smooth covariate sequence $X _ { 1 : T }$ and an array $Z _ { 1 : T , 1 : n } ,$ drawn i.i.d. from $\mu ,$ so that

$$
\mathbb { P } \left[ X _ { t } \in \{ Z _ { t , 1 } , . . . , Z _ { t , n } \} \mathrm { ~ f o r ~ e v e r y ~ } t \leq T \right] \geq 1 - T ( 1 - \sigma ) ^ { n } \geq 1 - \delta .\tag{14}
$$

On the displayed event, for any pair of functions $f , g ,$ , we can upper bound the discrepancy between $f$ and g on $X _ { 1 : T }$ as:

$$
\sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ f ( X _ { t } ) \neq g ( X _ { t } ) \right\} \leq \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \left\{ f ( Z _ { t , i } ) \neq g ( Z _ { t , i } ) \right\} .
$$

From part (ii) of Proposition A.1, there exist a cover $\mathcal { G } \subset \mathcal { F }$ with the following property:

$$
\log | \mathcal { G } | \lesssim d \log ( 2 T / \sigma ) , \qquad \forall f \in \mathcal { F } \exists g \in \mathcal { G } : \quad \mathbb { E } _ { Z \sim \mu } | f ( Z ) - g ( Z ) | \le \frac { \sigma } { T } .\tag{15}
$$

Now, consider the disagreement class

$$
\mathcal { H } : = \mathcal { F } \Delta \mathcal { F } = \{ x \mapsto \mathbb { 1 } \{ f ( x ) \neq g ( x ) \} : f , g \in \mathcal { F } \} .
$$

The VC dimension of such a class is in $O ( d )$ Blumer et al. (1989). Results for norm comparison inequalities for VC classes (Boucheron et al., 2013, Exercise 12.4) thus imply that, with probability at least $1 - \delta$ over the i.i.d. array $Z _ { 1 : T , 1 : n }$

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } \left[ \frac { 1 } { T n } \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { n } h ( Z _ { t , i } ) - 2 \mathbb { E } _ { \mu } h \right] \lesssim \frac { d \log ( 2 T n ) + \log ( 1 / \delta ) } { T n } .
$$

Enumerate $\mathcal { G } = \{ g _ { 1 } , \dotsc , g _ { | \mathcal { G } | } \}$ and, for each $f \in { \mathcal { F } }$ , choose $\pi ( f ) \in [ | { \mathcal { G } } | ]$ so that $g _ { \pi ( f ) }$ satisfies Eq. (15). On the preceding event,

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \left\{ f ( Z _ { t , i } ) \neq g _ { \pi ( f ) } ( Z _ { t , i } ) \right\} \lesssim \sigma n + d \log ( 2 T n ) + \log ( 1 / \delta ) \lesssim d \log ( 2 T / \sigma ) ,
$$

where the last inequality uses $\sigma n \le \log ( 2 T ^ { 2 } ) + 1 , \log ( 2 T n ) \le \log ( 2 T / \sigma )$ , and $d \geq 1$ . Thus,

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \operatorname* { i n f } _ { g \in \mathcal { Q } } \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ f ( Z _ { t , i } ) \neq g ( Z _ { t , i } ) \} \lesssim \sigma n + d \log ( 2 T n ) + \log ( 1 / \delta ) \lesssim d \log ( 2 T / \sigma ) .
$$

Intersecting this high probability event with one in Eq. (14), we obtain with probability $1 - 2 \delta =$ $1 - 1 / T$

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \operatorname* { i n f } _ { g \in \mathcal { Q } } \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ f ( X _ { t } ) \neq g ( X _ { t } ) \right\} \lesssim \sigma n + d \log ( 2 T n ) + \log ( 1 / \delta ) \lesssim d \log ( 2 T / \sigma ) .
$$

This concludes the proof.

For real-valued classes, the above can be extended to bounds that depend on the fat-shattering dimension of the class.

Proposition A.2. Let $\mathcal { F } \subseteq [ - 1 , 1 ] ^ { \mathcal { X } }$ . There are universal constants $c , C > 0$ such that the following hold. For every µ, horizon $T \geq 1 , \sigma \in ( 0 , 1 ]$ , and $0 < \varepsilon \le 1 / 1 6$ , there exist a finite subset ${ \mathcal { G } } = \{ g _ { 1 } , \dotsc , g _ { | { \mathcal { G } } | } \} \subseteq { \mathcal { F } }$ and a map π : ${ \mathcal { F } } \to [ | { \mathcal { G } } | ]$ with $g _ { \pi ( f ) } = f f o r f \in \mathcal { G }$ , such that

$$
\log | \mathcal { G } | \lesssim \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon }\tag{16}
$$

and, for every covariate process that is σ-smooth with respect to $\mu ,$ with probability at least $1 - 1 / T$ we have

$$
\operatorname* { s u p } _ { f \in { \mathcal { F } } } \sum _ { s = 1 } ^ { T } \mathbb { 1 } \{ | f ( X _ { s } ) - g _ { \pi ( f ) } ( X _ { s } ) | > 2 \varepsilon \} \lesssim \mathrm { f a t } _ { c \varepsilon } ( { \mathcal { F } } ) \log ^ { 2 } { \frac { C T } { \sigma \varepsilon } } .\tag{17}
$$

Proof. We use a double-sampling trick. First, let

$$
\delta = 1 / T , \quad n = \left\lceil \sigma ^ { - 1 } \log ( 2 T / \delta ) \right\rceil , \quad N = T n , \quad k = \left\lceil C \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon } \right\rceil .
$$

Let $Z _ { 1 : N }$ and $Z _ { 1 : N } ^ { \prime }$ be independent i.i.d. samples from $\mu .$ . We claim that

$$
\mathbb { P } \left[ \operatorname* { s u p } _ { \substack { f , g \in \mathcal { F } , } \atop \operatorname* { m a x } _ { j \leq N } | f ( Z _ { j } ^ { \prime } ) - g ( Z _ { j } ^ { \prime } ) | \leq \varepsilon } \sum _ { j = 1 } ^ { N } \mathbb { 1 } \left\{ | f ( Z _ { j } ) - g ( Z _ { j } ) | > 2 \varepsilon \right\} \geq k \right] \leq \frac { \delta } { 2 } .\tag{18}
$$

Condition on the unordered multiset $\{ Z _ { j } , Z _ { i } ^ { \prime } \} , j \in [ N ]$ . Choose an $\varepsilon / 4 -$ cover ${ \mathcal { G } } ^ { \prime } \subseteq { \mathcal { F } }$ on the 2N points $\{ Z _ { j } , Z _ { i } ^ { \prime } \} , j \le N$ using Proposition $\mathrm { A . 1 ( i ) }$ , so that $\mathcal { G } ^ { \prime }$ depends only on the multiset of 2N points, but not on their ordering. By Proposition $\mathrm { A . 1 ( i ) }$ , its cardinality satisfies

$$
\log | \mathcal { G } ^ { \prime } | \lesssim \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } ( C N / \varepsilon ) \lesssim \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon } .
$$

If $f , g$ satisfy the event in (18), choose functions $h , h ^ { \prime } \in \mathcal { G } ^ { \prime }$ from the cover such that $| h - f | \leq \varepsilon / 4$ and $| h ^ { \prime } - g | \leq \varepsilon / 4$ on all 2N points. Then, on the event Eq. (18), we have

$$
\operatorname* { m a x } _ { j \leq N } | h ( Z _ { j } ^ { \prime } ) - h ^ { \prime } ( Z _ { j } ^ { \prime } ) | \leq \frac { 3 \varepsilon } { 2 } , \qquad \sum _ { j = 1 } ^ { N } \mathbb { 1 } \left\{ | h ( Z _ { j } ) - h ^ { \prime } ( Z _ { j } ) | > \frac { 3 \varepsilon } { 2 } \right\} \geq k .
$$

Fix $h , h ^ { \prime } \in \mathcal { G } ^ { \prime }$ . Now, recall that so far we have conditioned on the multiset of 2N observations, but not on their ordering. Call a sample $Z \in \{ Z _ { j } , Z _ { i } ^ { \prime } \} _ { j \leq N }$ good if $| h ( Z ) - h ^ { \prime } ( Z ) | \le 3 \varepsilon / 2$ , and call it bad otherwise. Then, on the above event, all bad samples are from $Z _ { 1 : N }$ , and there are at least k of them. By exchangeability, the probability that all $\geq k$ bad samples are assigned to $Z _ { 1 : N }$ is at most $2 ^ { - k }$ . Thus, conditional on the multiset of 2N points, we have for any fixed pair $h , h ^ { \prime } \colon$

$$
\mathbb { P } [ \operatorname* { m a x } _ { j \leq N } | h ( Z _ { j } ^ { \prime } ) - h ^ { \prime } ( Z _ { j } ^ { \prime } ) | \leq \frac { 3 \varepsilon } { 2 } , \quad \sum _ { j = 1 } ^ { N } \mathbb { 1 } \{ | h ( Z _ { j } ) - h ^ { \prime } ( Z _ { j } ) | > \frac { 3 \varepsilon } { 2 } \} \geq k | \{ Z _ { j } , Z _ { j } ^ { \prime } \} _ { j \leq N } ] \leq 2 ^ { - k } .
$$

A union bound over $h , h ^ { \prime } \in \mathcal { G } ^ { \prime }$ therefore bounds the conditional probability in (18) by

$$
| \mathcal { G } ^ { \prime } | ^ { 2 } 2 ^ { - k } = \exp \left( 2 \log | \mathcal { G } ^ { \prime } | - k \log 2 \right) \leq \frac { \delta } { 2 } ,
$$

where the last inequality follows from the cardinality bound and $\log ( 2 / \delta ) \lesssim \log ( 2 T )$ , for $C$ suficiently large. This proves (18).

Now, to construct the desired cover, note that Eq. (18) implies existence of at least one fixed tuple $z _ { 1 : N } ^ { \prime }$ such that, when $Z _ { 1 : N } \sim \mu$ , we have

$$
\mathbb { P } \left[ \operatorname* { s u p } _ { \substack { f , g \in \mathcal { F } , } \atop \operatorname* { m a x } _ { j \leq N } | f ( z _ { j } ^ { \prime } ) - g ( z _ { j } ^ { \prime } ) | \leq \varepsilon } \sum _ { j = 1 } ^ { N } \mathbb { 1 } \{ | f ( Z _ { j } ) - g ( Z _ { j } ) | > 2 \varepsilon \} \geq k \right] \leq \frac { \delta } { 2 } .
$$

Construct an ε-cover $\mathcal { G }$ on $z _ { 1 : N } ^ { \prime }$ using Proposition $\mathrm { { A . 1 } ( i ) }$ , and choose, for each $f \in { \mathcal { F } }$ , a cover element $g _ { \pi ( f ) } \in \mathcal { G }$ such that

$$
\operatorname* { m a x } _ { j \le N } | f ( z _ { j } ^ { \prime } ) - g _ { \pi ( f ) } ( z _ { j } ^ { \prime } ) | \le \varepsilon ,
$$

taking $g _ { \pi ( f ) } = f$ for $f \in { \mathcal { G } }$ . Proposition $\mathrm { { A . 1 } ( i ) }$ gives (16). By (18), with probability at least $1 - \delta / 2$ over a fresh i.i.d. sample $Z _ { 1 : N }$

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \sum _ { j = 1 } ^ { N } \mathbb { 1 } \{ | f ( Z _ { j } ) - g _ { \pi ( f ) } ( Z _ { j } ) | > 2 \varepsilon \} < k .
$$

The conclusion follows from (14): with probability at least $1 - \delta / 2 ,$ , we can couple $X _ { 1 : T }$ to $Z _ { 1 : N }$ such that $X _ { 1 : T } \subset Z _ { 1 : N }$ . Applying union bound gives:

$$
\operatorname* { s u p } _ { f \in { \mathcal { F } } } \sum _ { s = 1 } ^ { T } \mathbb { 1 } \{ | f ( X _ { s } ) - g _ { \pi ( f ) } ( X _ { s } ) | > 2 \varepsilon \} \lesssim \mathrm { f a t } _ { c \varepsilon } ( { \mathcal { F } } ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon } ,
$$

with probability $1 - \delta .$ , as desired.

## A.2 Variance-based regret bounds

Variance-based bounds are a standard tool for analyzing exponential weights (Cesa-Bianchi et al., 2007). The following is a one-round Bernstein bound of De Rooij et al. (2014, Lemma 4), which will be useful in our analysis of exponential weights.

Proposition A.3. Let $\mathcal { G }$ be a nonempty finite class, let w $\in \Delta ( \mathcal { G } )$ be a set of weights, and let $\ell \colon \mathcal { G }  \mathbb { R }$ be a loss function. Suppose,

$$
\operatorname* { m a x } _ { f \in \mathcal { G } } \ell ( f ) - \operatorname* { m i n } _ { f \in \mathcal { G } } \ell ( f ) \leq s ,
$$

that is, s upper bounds the range of the loss. Then, for every $\eta > 0$

$$
\mathbb { E } _ { f \sim w } \ell ( f ) + \frac { 1 } { \eta } \log \sum _ { f \in \mathcal { G } } w ( f ) \exp \left( - \eta \ell ( f ) \right) \leq \frac { \exp \left( \eta s \right) - \eta s - 1 } { \eta s ^ { 2 } } \operatorname { V a r } _ { f \sim w } ( \ell ( f ) ) ,
$$

where the coeficient is interpreted as $\eta / 2$ when $s = 0$

## A.3 Gaussian geometry

The following identity is due to Stein (1981). Intuitively, it will allow us to relate gradients of certain potentials to their correlations with the Gaussians.

Lemma A.4 (Stein’s lemma). Let $G = ( G _ { 1 } , \ldots , G _ { n } )$ have independent standard Gaussian coordinates, and let $\Psi : \mathbb { R } ^ { n }  \mathbb { R }$ be Lipschitz. Then, for every $s \leq n$ 2

$$
\mathbb { E } _ { G } \partial _ { G _ { s } } \Psi ( G ) = \mathbb { E } _ { G } [ G _ { s } \Psi ( G ) ] .
$$

To control the aforementioned correlations, the following two assertions will be useful.

Lemma A.5. Let $G = ( G _ { 1 } , \ldots , G _ { n } )$ have independent standard Gaussian coordinates.

(i) For every potential $\Psi : \mathbb { R } ^ { n }  \mathbb { R }$ with $\mathbb { E } _ { G } [ \Psi ( G ) ^ { 2 } ] < \infty$ , we have $\begin{array} { r } { \sum _ { s \leq n } \left( \mathbb { E } _ { G } [ G _ { s } \Psi ( G ) ] \right) ^ { 2 } \leq } \end{array}$ $\operatorname { V a r } _ { G } ( \Psi ( G ) )$

(ii) Consider a finite class $\mathcal { H } \colon \mathcal { X } \to [ - 1 , 1 ]$ and any $h \colon \mathcal { X } \times \mathbb { R } ^ { n }  [ - 1 , 1 ]$ such that $h ( \cdot , G ) \in \mathrm { c o n v } ( \mathcal { H } )$ almost surely. Then, for any $z _ { 1 : n } \in \mathcal X ^ { n }$

$$
\sum _ { i \leq n } { ( \mathbb { E } _ { G } [ G _ { i } h ( z _ { i } , G ) ] ) ^ { 2 } } \leq 2 \log | \mathcal { H } | .
$$

Proof. First, note that we may assume without loss of generality that $\mathbb { E } \Psi ( G ) = 0$ . Then, (i) can be recognized as an instance of Bessel inequality for $L ^ { 2 }$ space. Indeed, let $\gamma$ be the standard Gaussian measure on R. For $f , g \in L ^ { 2 } ( \gamma ^ { n } )$ , let

$$
\langle f , g \rangle _ { L ^ { 2 } ( \gamma ^ { n } ) } : = \mathbb { E } _ { G \sim \gamma ^ { n } } f ( G ) g ( G ) .
$$

Let $e _ { i } \colon x \mapsto x _ { i }$ be the ith coordinate function; then, $\{ e _ { i } \}$ are orthonormal in this space, and we have

$$
\begin{array} { r l } { \displaystyle \sum _ { s \leq n } \left( \mathbb { E } _ { G } [ G _ { s } \Psi ( G ) ] \right) ^ { 2 } = \displaystyle \sum _ { s \leq n } \left( \langle e _ { s } , \Psi \rangle _ { L ^ { 2 } ( \gamma ^ { n } ) } \right) ^ { 2 } } & { } \\ { \leq \| \Psi \| _ { L ^ { 2 } ( \gamma ^ { n } ) } ^ { 2 } } & { } \\ { } & { = \mathbb { E } _ { G } \Psi ( G ) ^ { 2 } = \mathrm { V a r } _ { G } ( \Psi ( G ) ) , } \end{array}
$$

where the inequality is Bessel’s inequality and the last equality uses $\mathbb { E } _ { G } \Psi ( G ) = 0$ . This concludes the proof of part (i).

For (ii), let $\alpha _ { s } = \mathbb { E } _ { G } [ G _ { s } h ( z _ { s } , G ) ]$ for $s \leq n .$ and assume that $\alpha \neq 0$ , as otherwise there is nothing to prove. For a fixed $G ,$ the map $\begin{array} { r } { h \mapsto \sum _ { s < n } \alpha _ { s } G _ { s } h ( z _ { s } ) } \end{array}$ is linear, so its value at $h ( \cdot , G ) \in \mathrm { c o n v } ( \mathcal { H } )$ is at most its maximum over $\mathcal { H } .$ . Thus,

$$
\| { \boldsymbol { \alpha } } \| _ { 2 } ^ { 2 } = \mathbb { E } _ { G } \sum _ { s \leq n } { \alpha } _ { s } G _ { s } h ( z _ { s } , G ) \leq \mathbb { E } _ { G } \operatorname* { m a x } _ { h \in \mathcal { H } } \sum _ { s \leq n } { \alpha } _ { s } G _ { s } h ( z _ { s } ) .
$$

For each $h \in \mathcal H$ , the sum inside the maximum is a centered Gaussian with variance $\begin{array} { r } { \sum _ { s < n } \alpha _ { s } ^ { 2 } h ( z _ { s } ) ^ { 2 } \le } \end{array}$ $\| \alpha \| _ { 2 } ^ { 2 }$ . Thus, for every $\lambda > 0$ , Jensen’s inequality and the Gaussian moment-generating function give

$$
\begin{array} { r l r } {  { \mathbb { E } _ { G } \operatorname* { m a x } _ { h \in \mathcal { H } } \sum _ { s \leq n } \alpha _ { s } G _ { s } h ( z _ { s } ) \leq \frac { 1 } { \lambda } \log \sum _ { h \in \mathcal { H } } \mathbb { E } _ { G } \exp ( \lambda \sum _ { s \leq n } \alpha _ { s } G _ { s } h ( z _ { s } ) ) } } \\ & { } & { \leq \frac { \log | \mathcal { H } | } { \lambda } + \frac { \lambda \| \alpha \| _ { 2 } ^ { 2 } } { 2 } . } \end{array}
$$

Taking $\lambda = \sqrt { 2 \log | \mathcal { H } | } / \| \alpha \| _ { 2 }$ gives $\| \alpha \| _ { 2 } ^ { 2 } \leq \sqrt { 2 \log | \mathcal { H } | } \| \alpha \| _ { 2 }$ . Dividing by $\lVert \alpha \rVert _ { 2 }$ and squaring concludes the proof. □

## A.4 Convex hull approximations

We use the following consequence of Maurey’s empirical method (Pisier, 1981), in the $\ell _ { \infty }$ form due to Alth¨ofer (1994).

Proposition A.6 (Maurey’s empirical method). Let $\mathcal { F } \subseteq [ - 1 , 1 ] ^ { \mathcal { X } }$ . For any $h \in \mathrm { c o n v } ( \pm \mathcal { F } )$ , integers $m , K \ge 1$ , and points $Z _ { 1 } , \dots , Z _ { m } \in \mathcal { X }$ , there exist functions $f _ { 1 } , \ldots , f _ { K } \in \pm \mathcal { F }$ such that

$$
\operatorname* { m a x } _ { s \in [ m ] } \left| h ( Z _ { s } ) - \frac { 1 } { K } \sum _ { i = 1 } ^ { K } f _ { i } ( Z _ { s } ) \right| \leq \sqrt { \frac { 2 \log ( 2 m ) } { K } } .
$$

## A.5 Log-partition functions and their derivatives

Lemma A.7 (Derivatives of log-partition functions). Let $\psi _ { 1 } , \ldots , \psi _ { N } : \mathbb { R } ^ { n } \to$ R be Lipschitz, and define

$$
\Psi ( \theta ) = \log \sum _ { i = 1 } ^ { N } \exp \left( \psi _ { i } ( \theta ) \right) , \qquad w _ { i } ( \theta ) = \frac { \exp \left( \psi _ { i } ( \theta ) \right) } { \sum _ { j = 1 } ^ { N } \exp \left( \psi _ { j } ( \theta ) \right) } .
$$

Then, for almost every $\theta \in \mathbb { R } ^ { n }$ , every $r \leq n$ , and every $h \in \mathbb { R } ^ { N }$ ，

$$
\partial _ { \theta _ { r } } \Psi ( \theta ) = \mathbb { E } _ { i \sim w ( \theta ) } \partial _ { \theta _ { r } } \psi _ { i } ( \theta ) , \qquad \partial _ { \theta _ { r } } \mathbb { E } _ { i \sim w ( \theta ) } h _ { i } = \mathrm { C o v } _ { i \sim w ( \theta ) } \left( h _ { i } , \partial _ { \theta _ { r } } \psi _ { i } ( \theta ) \right) .
$$

In particular, $i f \psi _ { i } ( \theta ) = a _ { i } + \langle \theta , z _ { i } \rangle$ for some $a _ { i } \in \mathbb { R }$ and $z _ { i } \in \mathbb { R } ^ { n }$ , then for every $\theta \in \mathbb { R } ^ { n }$

$$
\partial _ { \theta _ { r } } \Psi ( \theta ) = \mathbb { E } _ { i \sim w ( \theta ) } ( z _ { i } ) _ { r } , \qquad \partial _ { \theta _ { r } } \mathbb { E } _ { i \sim w ( \theta ) } h _ { i } = \operatorname { C o v } _ { i \sim w ( \theta ) } \left( h _ { i } , ( z _ { i } ) _ { r } \right) .
$$

Proof. By Rademacher’s theorem, almost every θ is a point of diferentiability of all $\psi _ { 1 } , \ldots , \psi _ { N }$ . At such θ, the chain rule gives the first identity and

$$
\partial _ { \theta _ { r } } w _ { i } ( \theta ) = w _ { i } ( \theta ) \left( \partial _ { \theta _ { r } } \psi _ { i } ( \theta ) - \mathbb { E } _ { j \sim w ( \theta ) } \partial _ { \theta _ { r } } \psi _ { j } ( \theta ) \right) .
$$

Multiplying by $h _ { i }$ and summing over i gives the second identity. In the linear case, each $\psi _ { i }$ is diferentiable everywhere with $\partial _ { \theta _ { r } } \psi _ { i } ( \theta ) = ( z _ { i } ) _ { \ i }$ , so substituting into the two identities gives the stated result. □

## B Proofs for the binary case

The binary proof has three ingredients: a surprise lemma using second moments, a reduction to exponential weights, and a regret bound for the exponential-weights reference.

## B.1 Surprise lemma

We first bound the cumulative expected values of predictable test functions by their historical second moments on a tangent sequence. A norm comparison then replaces the tangent moments by their realized counterparts.

## B.1.1 Surprise lemma with tangent sequence

The following lemma follows from the change-of-measure argument of Xie et al. (2022, Section 3.2, equations (3)–(5)). We restate the result in our notation, and provide a proof for completeness.

Lemma B.1. For predictable test functions $h _ { t } \colon \mathcal { X } \to [ 0 , 1 ]$ , we have

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } h _ { t } ( X _ { t } ) \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( \bar { X } _ { s } ) ^ { 2 } } + \frac { \log ( 2 T ) } { \sigma } .\tag{19}
$$

Proof. We begin by letting

$$
a _ { t } ( x ) = \sigma \frac { \mathrm { d } p _ { t } } { \mathrm { d } \mu } ( x ) \in [ 0 , 1 ] .
$$

Similar to Xie et al. (2022), for each $x \in \mathcal { X }$ let

$$
\tau ( x ) : = \operatorname* { m i n } \left\{ t \in [ T ] \colon \sum _ { s \leq t } a _ { s } ( x ) \geq 1 \right\}
$$

be the index of the first round in which the cumulative scaled density at x reaches 1, with $\tau ( x ) = T$ if the set is empty. Intuitively, we can partition the surprise contribution $\mathbb { E } h _ { t } ( X _ { t } )$ in each round into x’s that are still in the “burn-in” period $( t \leq \tau ( x ) )$ and those that have left the burn-in phase. Observe that the burn-in contribution can be upper bounded as follows:

$$
\frac { 1 } { \sigma } \int \sum _ { t = 1 } ^ { \tau ( x ) } a _ { t } ( x ) h _ { t } ( x ) \mathrm { d } \mu ( x ) \leq \frac { 1 } { \sigma } + \frac { 1 } { \sigma } \int \sum _ { t = 1 } ^ { \tau ( x ) - 1 } a _ { t } ( x ) h _ { t } ( x ) \mathrm { d } \mu ( x ) \leq \frac { 2 } { \sigma } ,\tag{20}
$$

where we used $| h _ { t } | \leq 1$ . It remains to bound the contribution of those covariates that have left the burn-in period. From Xie et al. (2022, Lemma 4), we have with probability 1:

$$
\sum _ { t = 1 } ^ { T } \frac { a _ { t } ( x ) } { 1 + \sum _ { s < t } a _ { s } ( x ) } \lesssim \log ( 2 T ) .
$$

On the event $\textstyle \{ \sum _ { s < t } a _ { s } ( x ) \geq 1 \}$ , we also have

$$
\frac { a _ { t } ( x ) ^ { 2 } } { \sum _ { s < t } a _ { s } ( x ) } \leq \frac { 2 a _ { t } ( x ) } { 1 + \sum _ { s < t } a _ { s } ( x ) } .
$$

By Cauchy–Schwarz, the contribution after burn-in is bounded by

$$
\begin{array} { r l } & { \displaystyle \frac 1 \sigma \mathbb E \int _ { t = \tau ( x ^ { \prime } ) \mid 1 } ^ { \gamma } a _ { t } ( x ) h _ { t } ( x ) \mathrm { d } \mu ( x ) } \\ & { \quad \le \left( \frac 1 \sigma \mathbb E \int _ { t = \tau ( x ^ { \prime } ) + 1 } ^ { \frac T } \frac { a _ { t } ( x ) ^ { 2 } } { \sum _ { s \in t } a _ { s } ( x ) } \mathrm { d } \mu ( x ) \right) ^ { 1 / 2 } \left( \frac 1 \sigma \mathbb E \int _ { t = 1 } ^ { \frac T } \left( \sum _ { s \in \mathcal E } a _ { s } ( x ) \right) h _ { t } ( x ) ^ { 2 } \mathrm { d } \mu ( x ) \right) ^ { 1 / 2 } } \\ & { \quad \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } } \frac { 1 } { \sigma } \mathbb E \int _ { \frac { t = 1 } { \tau } } ^ { \frac T } \left( \sum _ { s \in \mathcal E } a _ { s } ( x ) \right) h _ { t } ( x ) ^ { 2 } \mathrm { d } \mu ( x ) } \\ & { \quad = \sqrt { \frac { \log ( 2 T ) } { \sigma } } \frac { 1 } { \varepsilon - 1 } \sum _ { s \in \mathcal E } \mathbb E _ { p _ { s } } h _ { t } ^ { 2 } . } \end{array}
$$

Combining the above with Eq. (20), we obtain:

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } h _ { t } ( X _ { t } ) \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( \bar { X } _ { s } ) ^ { 2 } } + \frac { \log ( 2 T ) } { \sigma } ,
$$

as desired.

For the same predictable functions $0 \leq h _ { t } \leq 1$ , Block et al. (2024b, Lemma 12) gives, up to universal constants,

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } h _ { t } ( X _ { t } ) \lesssim \frac { \log ( 2 T / \sigma ) } { \sigma } + \log ( 2 T / \sigma ) \sqrt { \frac { T } { \sigma } \mathbb { E } \sum _ { t = 1 } ^ { T } \frac { 1 } { t } \sum _ { s < t } \mathbb { E } _ { p _ { s } } h _ { t } } .
$$

Since $h _ { t } ^ { 2 } \leq h _ { t }$ and $t \leq T$

$$
\mathbb { E } \sum _ { \mathit { t } = 1 } ^ { \mathit { T } } \sum _ { s < { \mathit { t } } } \mathbb { E } _ { p _ { s } } h _ { { \mathit { t } } } ^ { 2 } \leq T \mathbb { E } \sum _ { \mathit { t } = 1 } ^ { \mathit { T } } { \frac { 1 } { \mathit { t } } } \sum _ { s < { \mathit { t } } } \mathbb { E } _ { p _ { s } } h _ { { \mathit { t } } } .
$$

Thus (19) also implies this first-moment bound in Block et al. (2024b).

## B.1.2 Comparison inequality for signed convex hulls

We next compare tangent and realized second moments uniformly for elements of signed convex hulls.

Lemma B.2. Under the same assumptions as in Lemma 4.1, we have

$$
\mathbb { E } \operatorname* { s u p } _ { h \in \mathrm { c o n v } ( \pm \mathcal { F } ) } \left\{ \sum _ { t \leq T } \mathbb { E } _ { \bar { X } _ { t } } h ^ { 2 } ( \bar { X } _ { t } ) - 1 2 \sum _ { t \leq T } h ( X _ { t } ) ^ { 2 } \right\} \lesssim d \log ^ { 3 } ( 2 T / \sigma ) ,
$$

Proof of Lemma B.2. In the proof, we first reduce the claim to comparison for binary function class using layer-cake identity for expectations and Maurey’s empirical method (Proposition A.6). Then, the proof is concluded by invoking Block et al. (2024b, Theorem 10). To start, using the layer-cake identity, we have

$$
\sum _ { s = 1 } ^ { T } \mathbb { E } _ { \bar { X } _ { s } } h ^ { 2 } ( \bar { X } _ { s } ) \leq 1 + \int _ { 1 / \sqrt { T } } ^ { 1 } 2 \lambda \sum _ { s = 1 } ^ { T } \mathbb { P } [ | h ( \bar { X } _ { s } ) | \geq \lambda ] \mathrm { d } \lambda .
$$

Also, note that we have:

$$
\sum _ { s = 1 } ^ { T } h ( X _ { s } ) ^ { 2 } \geq \int _ { 1 / \sqrt { T } } ^ { 1 } \frac { \lambda } { 2 } \sum _ { s = 1 } ^ { T } \mathbb { 1 } \{ | h ( X _ { s } ) | \geq \lambda / 2 \} \mathrm { d } \lambda .
$$

Thus,

$$
\begin{array} { r l r } {  { \sum _ { s = 1 } ^ { T } \mathbb { E } _ { \bar { X } _ { s } } h ^ { 2 } ( \bar { X } _ { s } ) - 1 2 \sum _ { s = 1 } ^ { T } h ( X _ { s } ) ^ { 2 } } } \\ & { } & { \leq 1 + \displaystyle \int _ { 1 / \sqrt { T } } ^ { 1 } 2 \lambda \{ \sum _ { s = 1 } ^ { T } \mathbb { P } [ | h ( \bar { X } _ { s } ) | \geq \lambda ] - 3 \sum _ { s = 1 } ^ { T } \mathbb { 1 } \{ | h ( X _ { s } ) | \geq \lambda / 2 \} \} \mathrm { d } \lambda . } \end{array}\tag{21}
$$

For $1 / \sqrt { T } \leq \lambda \leq 1$ , take $K = \lceil 3 2 \lambda ^ { - 2 } \log ( 4 T ) \rceil$ , and define

$$
\mathcal { F } _ { \lambda } = \left\{ x \mapsto \mathbb { 1 } \left\{ \left| \frac { 1 } { K } \sum _ { i = 1 } ^ { K } f _ { i } ( x ) \right| \geq \frac { 3 \lambda } { 4 } \right\} : f _ { 1 } , \ldots , f _ { K } \in \pm \mathcal { F } \right\} ,
$$

to be the class obtained by averaging K functions from ${ \mathcal F } .$ . Applying Proposition $\mathrm { A . 6 }$ to the $2 T$ points $X _ { 1 : T } , \bar { X } _ { 1 : T }$ , for every h ∈ conv(±F) there exists $g \in { \mathcal { F } } _ { \lambda }$ such that

$$
\mathbb { 1 } \left\{ | h ( x ) | \geq \lambda \right\} \leq g ( x ) \leq \mathbb { 1 } \left\{ | h ( x ) | \geq \lambda / 2 \right\}
$$

for every $x \in \{ X _ { s } , \bar { X } _ { s } : s \leq T \}$ . Set $m = \lceil 2 T \log ( T ) / \sigma \rceil$ and let $Z _ { 1 : m } \sim \mu ^ { m }$ . Using Block et al. (2024b, Theorem 10) and a bound on Wills functional using growth function of the class, we have:

$$
\begin{array} { r l } & { \mathbb { E } \underset { h \in \mathrm { c o u v } ( \pm \mathcal { F } ) } { \operatorname* { s u p } } \{ \underset { s = 1 } { T } \mathbb { P } [ | h ( \bar { X } _ { s } ) | \geq \lambda ] - 3 \sum _ { s = 1 } ^ { T } \mathbb { 1 } \{ | h ( X _ { s } ) | \geq \lambda / 2 \} \} } \\ & { \quad \leq \mathbb { E } \underset { h \in \mathrm { c o u v } ( \pm \mathcal { F } ) } { \operatorname* { s u p } } \{ \underset { s = 1 } { T } \{ \underset { s = 1 } { T }  \Vert h ( \bar { X } _ { s } ) | \geq \lambda \} - 3 \underset { s = 1 } { T } \{ | h ( X _ { s } ) | \geq \lambda / 2 \} \} } \\ & { \quad \leq \mathbb { E } \underset { g \in \mathcal { F } _ { \lambda } } { \operatorname* { s u p } } \underset { s = 1 } { \overset { T } { \sum } } ( g ( \bar { X } _ { s } ) - 3 g ( X _ { s } ) ) } \\ & { \quad \lesssim 1 + \log \underset { \bar { Z } _ { 1 : m } } { \operatorname* { s u p } } | \mathcal { F } _ { \lambda } | _ { Z _ { 1 : m } } \big | . } \end{array}\tag{22}
$$

We now bound the growth function of $\mathcal { F } _ { \lambda }$ . On any m points $Z _ { 1 } , \ldots , Z _ { m }$ , the Sauer–Shelah lemma (part (i) of Proposition A.1) gives

$$
\begin{array} { r } { \log | \pm \mathcal { F } | _ { Z _ { 1 : m } } | \le C d \log ( C m ) . } \end{array}
$$

Thus, via a simple counting argument, we have:

$$
\log \left| \mathcal { F } _ { \lambda } \right| _ { Z _ { 1 : m } } \bigr | \le C K d \log ( C m ) .
$$

Thus, we have

$$
1 + \log \operatorname* { s u p } _ { Z _ { 1 : m } } \left| \left. \mathcal { F } _ { \lambda } \right| _ { Z _ { 1 : m } } \right| \lesssim d K \log ( 2 m ) \lesssim d \lambda ^ { - 2 } \log ^ { 2 } ( 2 T / \sigma ) .
$$

Plugging this into Eqs. (21) and (22) gives

$$
\begin{array} { r l } & { \mathbb { E } \underset { h \in \mathrm { c o n v } ( \pm \mathcal { F } ) } { \operatorname* { s u p } } \left\{ \underset { s = 1 } { \overset { T } { \sum } } \mathbb { E } _ { \bar { X } _ { s } } h ^ { 2 } ( \bar { X } _ { s } ) - 1 2 \underset { s = 1 } { \overset { T } { \sum } } h ( X _ { s } ) ^ { 2 } \right\} } \\ & { \quad \lesssim 1 + \displaystyle \int _ { 1 / \sqrt { T } } ^ { 1 } 2 \lambda \left( d \lambda ^ { - 2 } \log ^ { 2 } ( 2 T / \sigma ) \right) \mathrm { d } \lambda \lesssim d \log ^ { 3 } ( 2 T / \sigma ) , } \end{array}
$$

as desired.

## B.1.3 Proof of the surprise lemma

It remains to combine the change-of-measure bound in Lemma B.1 with the uniform norm comparison in Lemma B.2.

Lemma 4.1. Let $\mathcal { F }$ be a class of VC dimension d and let $\{ h _ { t } \} _ { t \in [ T ] }$ be a predictable sequence of functions with $h _ { t } \in \mathrm { c o n v } ( \pm \mathcal { F } )$ . Then

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } \left| h _ { t } ( X _ { t } ) \right| \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } + \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } \left( \frac { 2 T } { \sigma } \right) .\tag{9}
$$

Proof of Lemma $4 . 1 .$ If $\sigma T < 1$ , the right-hand side of (9) is at least a constant multiple of $T ,$ , so the claim follows from $| h _ { t } | \leq 1$ . Otherwise, apply Lemma B.1 to $| h _ { t } |$ and the uniform bound in Lemma B.2 to $h _ { t }$ on each prefix $X _ { 1 : t - 1 }$ . This gives

$$
\begin{array} { r l r } {  { \mathbb { E } \sum _ { t = 1 } ^ { T } | h _ { t } ( X _ { t } ) | \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } ( \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } + d T \log ^ { 3 } ( 2 T / \sigma ) ) } + \frac { \log ( 2 T ) } { \sigma } } } \\ & { } & { \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } + \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) , } } \end{array}
$$

where the last step uses $\sigma T \geq 1 , d \geq 1$ , and $\log ( 2 T ) \leq \log ( 2 T / \sigma )$

The proofs of Lemmas B.2 and 4.1 use the VC dimension only to count the restrictions of $\pm \mathcal { F }$ to finitely many points. The same argument therefore applies to finite classes of real-valued functions, which we use in Appendix C.

Corollary B.3. Let $\mathcal { F } \subseteq [ - 1 , 1 ] ^ { \mathcal { X } }$ be a nonempty finite class, and let $\{ h _ { t } \} _ { t \in [ T ] }$ be a predictable sequence of functions with $h _ { t } \in \mathrm { c o n v } ( \pm \mathcal { F } )$ . Then,

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } | h _ { t } ( X _ { t } ) | \lesssim \sqrt { \frac { \log ( 2 T ) } { \sigma } } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } + \sqrt { \frac { \log ( 2 | \mathcal { F } | ) T } { \sigma } } \log ^ { 2 } \left( \frac { 2 T } { \sigma } \right) .
$$

## B.2 Comparing predictions on the past

We begin by proving that predictions of FTPL Eq. (3) and exponential weights Eq. (4) in round t are close on past covariates $X _ { s }$ with $s < t .$

Lemma 4.4. For every $t \leq T$

$$
\mathbb { E } \sum _ { s < t } \Big ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { s } ) \Big ) ^ { 2 } \lesssim d ^ { 2 } \log ^ { 2 } ( 2 T / \sigma ) .
$$

Proof of Lemma $4 { \cdot } 4 .$ Fix the history $\left( X _ { 1 : t - 1 } , Y _ { 1 : t - 1 } \right)$ and let $G : = G ^ { ( t ) }$ to lighten the notational load. Use the same potentials as in the proof sketch:

$$
\Phi _ { t } ( G ) = \operatorname* { m a x } _ { f \in \mathcal { F } } \sum _ { s < t } ( Y _ { s } - G _ { s } ) f ( X _ { s } ) ,
$$

$$
\begin{array} { r l } {  { \widetilde { \Phi } _ { t } ( G ) = \log \sum _ { f \in \mathcal { G } } \exp ( \sum _ { s < t } ( Y _ { s } - G _ { s } ) f ( X _ { s } ) ) } } \\ & { = \underbracket { \operatorname* { m a x } _ { w \in \Delta ( \mathcal { G } ) } [ \mathbb { E } _ { f \sim w } \sum _ { s < t } ( Y _ { s } - G _ { s } ) f ( X _ { s } ) + H ( w ) ] } _ { w \in \Delta ( \mathcal { G } ) } . } \end{array}
$$

$w _ { t } ( \cdot \mid G )$ from (4) maximize the latter potential in its variational form. Note that, by absolute continuity of the Gaussians, the set of all functions that maximize the former or the latter potential share predictions on every past covariate $X _ { s }$ . Thus, for every $s < t$ and almost every $G ,$ , we may apply Danskin’s theorem for the finite maximum (Bertsekas, 2016, Appendix B). Combining it with Lemma $\mathrm { A . 7 }$ , diferentiating the two potentials gives:

$$
\partial _ { G _ { s } } \Phi _ { t } ( G ) = - \hat { f } _ { t } ( X _ { s } ) , \qquad \partial _ { G _ { s } } \widetilde { \Phi } _ { t } ( G ) = - \tilde { f } _ { t } ( X _ { s } ) .
$$

Both potentials are Lipschitz functions of G, so Stein’s lemma (Lemma A.4) yields

$$
\sum _ { s < t } \left( \mathbb { E } _ { G } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G } \tilde { f } _ { t } ( X _ { s } ) \right) ^ { 2 } = \Big \lVert \mathbb { E } _ { G } \nabla _ { G } \big ( \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) \big ) \Big \rVert _ { 2 } ^ { 2 } = \Big \lVert \mathbb { E } _ { G } [ G ( \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) ) ] \Big \rVert _ { 2 } ^ { 2 } .
$$

Applying part (i) of Lemma A.5 further gives:

$$
\sum _ { s < t } \Big ( \mathbb { E } _ { G } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G } \tilde { f } _ { t } ( X _ { s } ) \Big ) ^ { 2 } \leq \mathbb { E } _ { G } ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) ^ { 2 } .\tag{23}
$$

Now, let

$$
\Phi _ { t } ^ { \dag } ( G ) = \operatorname* { m a x } _ { f \in \mathcal { G } } \sum _ { s < t } ( Y _ { s } - G _ { s } ) f ( X _ { s } ) ,
$$

be the unregularized potential over the cover $\mathcal { G } .$ . Since entropy is upper bounded by log |G|, we have:

$$
0 \leq \widetilde { \Phi } _ { t } ( G ) - \Phi _ { t } ^ { \dagger } ( G ) \leq \log | \mathcal { G } | \lesssim d \log ( 2 T / \sigma ) .\tag{24}
$$

By Proposition 4.2, with probability at least $1 - 1 / T$ over the history,

$$
\operatorname* { s u p } _ { f \in { \mathcal { F } } } \operatorname* { i n f } _ { g \in { \mathcal { G } } } \sum _ { s < t } \mathbb { 1 } \{ f ( X _ { s } ) \neq g ( X _ { s } ) \} \leq C d \log ( 2 T / \sigma ) .
$$

On the above event, using $Y _ { t } \in \{ \pm 1 \}$ , we have:

$$
0 \leq \Phi _ { t } ( G ) - \Phi _ { t } ^ { \dag } ( G ) \leq 2 C d \log ( 2 T / \sigma ) \qquad \quad  \\  + \operatorname* { m a x } _ { \substack { f \in \mathscr { F } , \ q \in \mathscr { G } : \ } } \left| \sum _ { s < t } G _ { s } ( f ( X _ { s } ) - g ( X _ { s } ) ) \right| .\tag{25}
$$

For each pair in the maximum, its Gaussian sum has variance

$$
\operatorname { V a r } _ { G } \left( \sum _ { s < t } G _ { s } ( f ( X _ { s } ) - g ( X _ { s } ) ) \right) = 4 \sum _ { s < t } 1 \left\{ f ( X _ { s } ) \neq g ( X _ { s } ) \right\} \leq 4 C d \log ( 2 T / \sigma ) .
$$

By Blumer et al. (1989), the diference class has VC dimension in $O ( d )$ . Thus, by Sauer–Shelah lemma (Proposition A.1), the maximum in Eq. (25) is taken over at most exp(Cd log(Ct)) functions. Applying Gaussian tail inequality with a union bound then gives:

$$
\mathbb { E } _ { G } \left[ \operatorname* { m a x } _ { \substack { { f \in \mathscr { F } , \ g \in \mathscr { Q } _ { \cdot } } } } \left| \sum _ { s < t } G _ { s } ( f ( X _ { s } ) - g ( X _ { s } ) ) \right| ^ { 2 } \right]
$$

It follows that $\begin{array} { r } { \mathbb { E } _ { G } ( \Phi _ { t } - \Phi _ { t } ^ { \dagger } ) ^ { 2 } \lesssim d ^ { 2 } \log ^ { 2 } ( 2 T / \sigma ) } \end{array}$ . Combining this with Eq. (24), we have:

$$
\begin{array} { r l } & { \mathbb { E } _ { G } ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) ^ { 2 } \leq \mathbb { E } _ { G } ( \Phi _ { t } - \Phi _ { t } ^ { \dagger } ) ^ { 2 } + \mathbb { E } _ { G } ( \widetilde { \Phi } _ { t } - \Phi _ { t } ^ { \dagger } ) ^ { 2 } } \\ & { \qquad \lesssim d ^ { 2 } \log ^ { 2 } ( 2 T / \sigma ) . } \end{array}
$$

Noting that the failure event of Proposition 4.2 contributes at most $O ( 1 )$ to the expected squared prediction discrepancy concludes the proof. □

## B.3 Proof of the reduction to exponential weights

We now apply the surprise lemma Lemma 4.1 to the diference of predictions.

Lemma 4.3 (Reduction to exponential weights). Consider a binary function class $\mathcal { F } \colon \mathcal { X } \to \{ \pm 1 \}$ of VC dimension d. For $\hat { f } _ { t }$ and $\tilde { f } _ { t }$ as defined in Eqs. (3) and (4), respectively, we have the following bound:

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } | \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) | \lesssim d \log ^ { 2 } ( 2 T / \sigma ) \sqrt { \frac { T } { \sigma } } .\tag{10}
$$

Proof of Lemma 4.3. Apply Lemma 4.1 to the sequence

$$
h _ { t } = \frac { 1 } { 2 } \left( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } \right) , \qquad t \in [ T ] .
$$

Note that both $\mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t }$ and $\mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t }$ are elements of conv(±F) and are predictable. Thus, using Lemma 4.1 gives

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } \Big | \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) \Big | \lesssim \left( \displaystyle \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { s < t } \Big ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { s } ) \Big ) ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \qquad + \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) . } \end{array}
$$

By Lemma 4.4, the expected double sum is in $O ( d ^ { 2 } T \log ^ { 2 } ( 2 T / \sigma ) )$ . Substituting this bound and using $\log ( 2 T ) \leq \log ( 2 T / \sigma )$ gives

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } \Big | \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) \Big | \lesssim d \log ^ { 2 } ( 2 T / \sigma ) \sqrt { \frac { T } { \sigma } } .
$$

This proves (10).

## B.4 Regret of Gaussian-perturbed exponential weights

We first prove Proposition 4.6, which reduces the regret bound to controlling cumulative prediction variance. We then prove Lemma 4.7, which bounds the variances on the historical covariates, and combine it with the surprise lemma to prove Lemma 4.5.

Proposition 4.6. Let ${ \mathcal { G } } \subseteq { \mathcal { F } }$ be a nonempty finite class, and let $\ell ( \hat { y } , y ) = - y \hat { y }$ be the binary loss. For every data sequence, the predictor $\tilde { f } _ { t }$ defined in (4) satisfies the following inequality,

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \ell ( \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \operatorname* { m i n } _ { f \in \mathcal { G } } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \lesssim \log | \mathcal { G } | + \sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w t ( \cdot | G ^ { ( t ) } ) } ( f ( X _ { t } ) ) .\tag{12}
$$

Proposition 4.6 is the special case of the following proposition in which ${ \mathcal { G } } \subseteq { \mathcal { F } }$ consists of binary functions and $\ell ( \hat { y } , y ) = - y \hat { y }$ , which is linear, and hence convex and 1-Lipschitz, in its first argument. We use the general form in Appendix C.4.

Proposition B.4. Let G be a nonempty finite class of functions $\mathcal { X }  [ - 1 , 1 ]$ , and let $\ell ( \hat { y } , y )$ be convex and 1-Lipschitz in its first argument. For every data sequence, the predictor $\tilde { f } _ { t }$ defined in (4) satisfies the following inequality,

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \ell ( \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \operatorname* { m i n } _ { f \in \mathcal { G } } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \lesssim \log | \mathcal { G } | + \sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( f ( X _ { t } ) ) .
$$

Proof of Proposition $B . 4 .$ . Recall the log-partition potential from the proof of Lemma 4.4, written for a general loss for each t:

$$
\widetilde { \Phi } _ { t } ( G ^ { ( t ) } ) = \log \sum _ { f \in \mathcal { G } } \exp \left( - \sum _ { s < t } \Big [ \ell ( f ( X _ { s } ) , Y _ { s } ) + G _ { s } ^ { ( t ) } f ( X _ { s } ) \Big ] \right) ,
$$

For a fixed round t, we have the following identity:

$$
\widetilde { \Phi } _ { t + 1 } ( G ^ { ( t + 1 ) } ) - \widetilde { \Phi } _ { t } ( G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) = \log \sum _ { f \in \mathcal { G } } w _ { t } ( f \mid G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) \exp \left( - \ell ( f ( X _ { t } ) , Y _ { t } ) - G _ { t } ^ { ( t + 1 ) } f ( X _ { t } ) \right) .
$$

Apply Proposition $\mathrm { A . 3 }$ to these weights and the losses $\ell ( f ( X _ { t } ) , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } f ( X _ { t } )$ at learning rate 1. Their range is at most $s _ { t } : = 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | )$ , and

$$
\begin{array} { r } { \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) } ( \ell ( f ( X _ { t } ) , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } f ( X _ { t } ) ) \le ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) ^ { 2 } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) } ( f ( X _ { t } ) ) , } \end{array}
$$

since $y \mapsto \ell ( y , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } y { \mathrm { ~ i s ~ } } ( 1 + | G _ { t } ^ { ( t + 1 ) } | )$ )-Lipschitz on $[ - 1 , 1 ]$ . Thus, cancelling $s _ { t } ^ { 2 }$ from numerator and the denominator in Proposition A.3, we have

$$
\begin{array} { r l } {  { \sum _ { f \in \mathcal { G } } w _ { t } ( f \mid G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) [ \ell ( f ( X _ { t } ) , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } f ( X _ { t } ) ] } } \\ & { \leq \widetilde { \Phi } _ { t } ( G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) - \widetilde { \Phi } _ { t + 1 } ( G ^ { ( t + 1 ) } ) } \\ & { \quad + \frac { \exp \big ( 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) \big ) - 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) - 1 } { 4 } \mathrm { V a r } _ { f \sim w _ { t } ( \cdot \vert G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) } ( f ( X _ { t } ) ) . } \end{array}
$$

This inequality also holds when the loss range is zero. Note that the last coordinate $G _ { t } ^ { ( t + 1 ) }$ is centered and independent of the first t − 1 coordinates. In particular,

$$
\begin{array} { r l } & { \mathbb { E } _ { G _ { t } ^ { ( t + 1 ) } } \frac { \exp { \left( 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) \right) } - 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) - 1 } { 4 } } \\ & { \quad \leq \frac { \exp ( 2 ) } { 4 } \mathbb { E } _ { G _ { t } ^ { ( t + 1 ) } } \left( \exp { \left( 2 G _ { t } ^ { ( t + 1 ) } \right) } + \exp { \left( - 2 G _ { t } ^ { ( t + 1 ) } \right) } \right) = \frac { \exp ( 4 ) } { 2 } . } \end{array}
$$

Now, $G _ { 1 : t - 1 } ^ { ( t + 1 ) }$ has the same law as $G ^ { ( t ) }$ conditionally on the history up to round t. Thus, using convexity of the loss,

$$
\mathbb { E } _ { G ^ { ( t ) } } \ell ( \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) \le \mathbb { E } _ { G ^ { ( t ) } } \widetilde { \Phi } _ { t } ( G ^ { ( t ) } ) - \mathbb { E } _ { G ^ { ( t + 1 ) } } \widetilde { \Phi } _ { t + 1 } ( G ^ { ( t + 1 ) } ) + \frac { \exp ( 4 ) } { 2 } \mathbb { E } _ { G ^ { ( t ) } } \mathrm { V a r } _ { f \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( f ( X _ { t } ) ) .\tag{26}
$$

Note that we have $\widetilde { \Phi } _ { 1 } = \log | \mathcal { G } |$ , and

$$
\begin{array} { r l } & { \mathbb { E } _ { G ^ { ( T + 1 ) } } \widetilde { \Phi } _ { T + 1 } ( G ^ { ( T + 1 ) } ) \geq \underset { f \in \mathcal { G } } { \operatorname* { m a x } } \mathbb { E } _ { G ^ { ( T + 1 ) } } \left[ - \underset { s = 1 } { \overset { T } { \sum } } \Big ( \ell ( f ( X _ { s } ) , Y _ { s } ) + G _ { s } ^ { ( T + 1 ) } f ( X _ { s } ) \Big ) \right] } \\ & { \quad \quad \quad = - \underset { f \in \mathcal { G } } { \operatorname* { m i n } } \underset { s = 1 } { \overset { T } { \sum } } \ell ( f ( X _ { s } ) , Y _ { s } ) , } \end{array}
$$

by Jensen’s inequality. Combining this with Eq. (26) and telescoping gives the desired result.

Lemma 4.7. Let $\mathcal { G }$ be a nonempty finite class of functions $\mathcal { X }  [ - 1 , 1 ]$ and let ℓ be any loss. For every $t \leq T$ and every fixed history $( X _ { s } , Y _ { s } ) _ { s < t } ,$ , the weights in (4) satisfy

$$
\sum _ { s < t } \left( \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( f ( X _ { s } ) ) \right) ^ { 2 } \lesssim \log ( 2 \vert \mathcal { G } \vert ) .\tag{13}
$$

Proof of Lemma $4 . 7 .$ Write $G = G ^ { ( t ) }$ throughout this proof for notational brevity. From Lemma A.7, we have

$$
\partial _ { G _ { s } } \tilde { f } _ { t } ( X _ { s } ) = - \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G ) } ( f ( X _ { s } ) ) .
$$

Applying Lemma A.4 therefore gives:

$$
\begin{array} { r } { \mathbb { E } _ { G } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G ) } \bigl ( f ( X _ { s } ) \bigr ) = - \mathbb { E } _ { G } \left[ G _ { s } \tilde { f } _ { t } ( X _ { s } ) \right] . } \end{array}
$$

To bound all these correlations together, we use part (ii) of Lemma A.5, which gives:

$$
\sum _ { s < t } \big ( \mathbb { E } _ { G } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G ) } ( f ( X _ { s } ) ) \big ) ^ { 2 } = \sum _ { s < t } \Big ( \mathbb { E } _ { G } \left[ G _ { s } \tilde { f } _ { t } ( X _ { s } ) \right] \Big ) ^ { 2 } \lesssim \log 2 | \mathcal { G } | ,
$$

thus, concluding the proof.

Lemma 4.5. The regret of the exponential weights algorithm in (4) satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { E W } } \lesssim \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) .\tag{11}
$$

Proof of Lemma 4.5. Apply Proposition 4.6 to the cover from Proposition 4.2. The diference in the comparator terms over $\mathcal { G }$ and $\mathcal { F }$ is upper bounded as:

$$
\begin{array} { r l } & { 0 \le \underset { g \in \mathcal { G } } { \operatorname* { m i n } } \displaystyle \sum _ { t = 1 } ^ { T } \ell ( g ( X _ { t } ) , Y _ { t } ) - \underset { f \in \mathcal { F } } { \operatorname* { i n f } } \displaystyle \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) } \\ & { \leq 2 \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \underset { g \in \mathcal { G } } { \operatorname* { i n f } } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ f ( X _ { t } ) \neq g ( X _ { t } ) \} . } \end{array}
$$

By Proposition 4.2, the expectation of this diference is at most $C d \log ( 2 T / \sigma ) + 2$ . Thus, applying this estimate to (12), we have:

$$
\mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { E W } } \lesssim \log | \mathcal { G } | + \mathbb { E } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G ^ { ( t ) } ) } ( f ( X _ { t } ) ) + C d \log ( 2 T / \sigma ) + 2 .\tag{27}
$$

The above reduces the proof to bounding the cumulative prediction variance. We know from Lemma 4.7 that variance of $f \sim w _ { t }$ on the past covariates $X _ { s }$ with $s < t$ can be suitably upper bounded. To upper bound the above, we will show that variance terms can be expressed as predictable test functions of a VC class, and apply the surprise lemma Lemma 4.1. Indeed, for every t and $x \in \mathcal { X }$ , we have

$$
\frac 1 2 \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( f ( x ) ) = \mathbb { E } _ { G ^ { ( t ) } } \sum _ { f , g \in \mathcal { G } } w _ { t } ( f \mid G ^ { ( t ) } ) w _ { t } ( g \mid G ^ { ( t ) } ) \mathbb { 1 } \{ f ( x ) \neq g ( x ) \} .
$$

This is a predictable test-function that belongs to the convex hull of the disagreement class:

$$
\left\{ x \mapsto \mathbb { 1 } \left\{ f ( x ) \neq g ( x ) \right\} : f , g \in { \mathcal { F } } \right\} .
$$

From Blumer et al. (1989), its VC dimension is in $O ( d )$ . Thus, using Lemma 4.7 and applying Lemma 4.1 to the sequence of variances gives

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } \operatorname { V a r } _ { f \sim w _ { t } ( \cdot | G ^ { ( t ) } ) } ( f ( X _ { t } ) ) \lesssim \sqrt { \frac { T \log | \mathcal { G } | \log ( 2 T ) } { \sigma } } + \sqrt { \frac { ( d + 1 ) T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) .
$$

Plugging this into Eq. (27) yields:

$$
\mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { E W } } \lesssim \log | \mathcal { G } | + \sqrt { \frac { T \log | \mathcal { G } | \log ( 2 T ) } { \sigma } } + \sqrt { \frac { ( d + 1 ) T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) + d \log ( 2 T / \sigma ) .
$$

To obtain (11), we use Proposition 4.2 and note that $\mathrm { R e g } _ { T } ^ { \mathrm { E W } } \leq 2 T$ with probability 1, and thus we can absorb the last term by $\sqrt { T \cdot d \log ( 2 T / \sigma ) }$ . This concludes the proof. □

Theorem 3.1 follows by combining Lemma 4.3 and Lemma 4.5.

## C Proofs for the real-valued loss case

The real-valued proof has the same three ingredients as the binary proof: a surprise lemma, a reduction to an exponential-weights reference algorithm that runs over a finite cover, and a regret bound for exponential weights.

First, we change slightly the way the reference exponential algorithm is defined. Let $\mathcal { G }$ be the cover from Proposition A.2 and, for each $f ,$ let $g _ { \pi ( f ) }$ be the element of the cover that corresponds to $f .$ . One can consider a partition of $\mathcal { F }$ into cells where each cell contains all functions $f$ that share a corresponding element $g _ { \pi ( f ) }$ . Within each cell, we consider clipping of functions towards their corresponding cover center; in particular, for a fixed valued $\varepsilon < 1 / 1 6$ , and for $f \in { \mathcal { F } }$ , define

$$
\bar { f } ( x ) = g _ { \pi ( f ) } ( x ) + \operatorname* { m a x } \{ - 2 \varepsilon , \operatorname* { m i n } \{ 2 \varepsilon , f ( x ) - g _ { \pi ( f ) } ( x ) \} \} .
$$

We then modify exponential weights by assigning each $g \in { \mathcal { G } }$ a weight determined by the smallest loss attained by a clipped function in its cell. Formally, we let

$$
\Phi _ { t , g } ( G ) = \operatorname* { s u p } _ { f \in \mathcal { F } : g _ { \pi ( f ) } = g } \left\{ - \sum _ { s < t } \left[ \ell ( \bar { f } ( X _ { s } ) , Y _ { s } ) + G _ { s } \bar { f } ( X _ { s } ) \right] \right\} , \qquad \widetilde { \Phi } _ { t } ( G ) = \log \sum _ { g \in \mathcal { G } } \exp \left( \Phi _ { t , g } ( G ) \right)\tag{28}
$$

Then, define:

$$
w _ { t } ( g \mid G ) = \exp \left( \Phi _ { t , g } ( G ) - \widetilde { \Phi } _ { t } ( G ) \right) , \qquad \widetilde { f } _ { t } = \sum _ { g \in \mathcal { G } } w _ { t } ( g \mid G ) g .\tag{29}
$$

Intuitively, computing the averaging weights using clipped functions as opposed to the centers $g$ accomplishes two goals. First, by selecting the minimizer within each cell, we improve the comparison between exponential weights and FTPL. Second, by introducing clipping, we ensure that the cell potentials are not too far from the perturbed scores of the cover centers $\mathcal { G }$ . Formally, we establish the following guarantee for the clipping we use.

Lemma C.1. Let $\mathcal { G }$ and the chosen cover elements $g _ { \pi ( f ) }$ be as in Proposition $A . { \mathcal { Q } } ,$ and define <sup>¯</sup>f for $f \in { \mathcal { F } }$ as above.

(i) For every $f \in { \mathcal { F } }$ and $x \in \mathcal { X } _ { : }$ , we have $\bar { f } ( x ) \in [ - 1 , 1 ] , | \bar { f } ( x ) - g _ { \pi ( f ) } ( x ) | \leq 2 \varepsilon$ , and

$$
| \bar { f } ( x ) - f ( x ) | \leq 2 \mathbb { 1 } \{ | f ( x ) - g _ { \pi ( f ) } ( x ) | > 2 \varepsilon \} .
$$

(ii) With probability at least $1 - 1 / T _ { \ast }$

$$
\operatorname* { s u p } _ { f \in { \mathcal { F } } } \sum _ { s = 1 } ^ { T } | f ( X _ { s } ) - { \bar { f } } ( X _ { s } ) | \lesssim \mathrm { f a t } _ { c \varepsilon } ( { \mathcal { F } } ) \log ^ { 2 } { \frac { C T } { \sigma \varepsilon } } .
$$

Proof. For part (i), $\bar { f } - g _ { \pi ( f ) }$ is $f - g _ { \pi ( f ) }$ clipped to $[ - 2 \varepsilon , 2 \varepsilon ]$ . Hence ${ \bar { f } } ( x )$ lies between $g _ { \pi ( f ) } ( x )$ and $f ( x )$ , so it belongs to $[ - 1 , 1 ]$ and satisfies $| \bar { f } ( x ) - f ( x ) | \leq | f ( x ) - g _ { \pi ( f ) } ( x ) | \leq 2$ . It equals $f ( x )$ whenever $| f ( x ) - g _ { \pi ( f ) } ( x ) | \leq 2 \varepsilon$ . For part (ii), part (i) gives $| f ( X _ { s } ) - { \bar { f } } ( X _ { s } ) | \leq 2 \mathbb { 1 } \left\{ | f ( X _ { s } ) - g _ { \pi ( f ) } ( X _ { s } ) | > 2 \varepsilon \right\}$ Summing over $s \leq T$ and applying (17) proves the claim. □

## C.1 Finite-scale surprise bounds

We begin by adapting our surprise bound argument for real-valued classes.

Lemma C.2. Let $0 < \varepsilon \le 1 / 1 6$ , and let $\{ h _ { t } \} _ { t \in [ T ] }$ be a predictable sequence of functions, each of which is a convex combination or a probability mixture of elements of $\pm \mathcal { F }$ . Then

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } | h _ { t } ( X _ { t } ) | \lesssim T \varepsilon \sqrt { \frac { \log ( 2 T ) } { \sigma } } + \sqrt { \frac { \log ( 2 T ) } { \sigma } } \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } + \sqrt { \frac { ( 1 + \mathrm { f a t } _ { c s } ( \mathcal { F } ) ) T } { \sigma } } \log ^ { 5 / 2 } \frac { C T } { \sigma \varepsilon } .
$$

Compared with Lemma 4.1, the VC dimension is replaced by the fat-shattering dimension at scale cε, at the price of the additive term $T \varepsilon { \sqrt { \log ( 2 T ) / \sigma } }$ . This term pays for replacing each function by its chosen cover element, which is within 2ε of it except at a few covariates.

Proof of Lemma C.2. Throughout this proof, suprema over h range over conv $( \pm { \mathcal { F } } )$ . We first establish the following analogue of Lemma B.2:

$$
\mathbb { E } \operatorname* { s u p } _ { h } \left\{ \sum _ { s = 1 } ^ { T } \mathbb { E } _ { \bar { X } _ { s } } h ( \bar { X } _ { s } ) ^ { 2 } - 1 2 \sum _ { s = 1 } ^ { T } h ( X _ { s } ) ^ { 2 } \right\} \lesssim T \varepsilon ^ { 2 } + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 4 } \frac { C T } { \sigma \varepsilon } .\tag{30}
$$

As in the proof of Lemma B.2, we use the layer-cake identity and Maurey’s empirical method (Proposition $\mathrm { A . 6 } )$ to reduce the claim to comparison for a binary function class, after which we apply Block et al. (2024b, Theorem 10). To start, using the layer-cake identity, we have

$$
\begin{array} { r l } {  { \sum _ { s = 1 } ^ { T } \mathbb { E } _ { \bar { X } _ { s } } h ( \bar { X } _ { s } ) ^ { 2 } \le 4 T \varepsilon ^ { 2 } + \int _ { 2 \varepsilon } ^ { 1 } 2 \lambda \sum _ { s = 1 } ^ { T } \mathbb { P } [ | h ( \bar { X } _ { s } ) | \ge \lambda ] \mathrm { d } \lambda . } } \end{array}
$$

Also, note that we have:

$$
\sum _ { s = 1 } ^ { T } h ( X _ { s } ) ^ { 2 } \geq \int _ { 2 \varepsilon } ^ { 1 } \frac { \lambda } { 2 } \sum _ { s = 1 } ^ { T } \mathbb { 1 } \left\{ | h ( X _ { s } ) | \geq \lambda / 2 \right\} \mathrm { d } \lambda .
$$

Thus,

$$
\sum _ { s = 1 } ^ { T } \mathbb { E } _ { \bar { X } _ { s } } h ( \bar { X } _ { s } ) ^ { 2 } - 1 2 \sum _ { s = 1 } ^ { T } h ( X _ { s } ) ^ { 2 }
$$

$$
\leq 4 T \varepsilon ^ { 2 } + \int _ { 2 \varepsilon } ^ { 1 } 2 \lambda \left\{ \sum _ { s = 1 } ^ { T } \mathbb { P } [ | h ( \bar { X } _ { s } ) | \geq \lambda ] - 3 \sum _ { s = 1 } ^ { T } \mathbb { 1 } \left\{ | h ( X _ { s } ) | \geq \lambda / 2 \right\} \right\} \mathrm { d } \lambda .\tag{31}
$$

For $2 \varepsilon \le \lambda \le 1$ , apply Proposition $\mathrm { A . 2 }$ at scale $\lambda / 1 6$ and horizon $T$ to obtain a fixed finite class $\mathcal { G } _ { \lambda } \subseteq \mathcal { F }$ and a choice of cover element $g _ { \pi ( f ) }$ for every $f \in { \mathcal { F } }$ . Then

$$
\log | \mathcal { G } _ { \lambda } | \lesssim \mathrm { f a t } _ { c \lambda } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \lambda } .
$$

The tangent sequence is also σ-smooth with respect to $\mu .$ . Thus, applying Proposition $\mathrm { A . 2 }$ to both sequences and taking a union bound, with probability at least $1 - 2 / T$ we have

$$
\operatorname* { s u p } _ { f \in { \mathcal F } } \sum _ { s = 1 } ^ { T } \left( \mathbb { 1 } \{ | f ( X _ { s } ) - g _ { \pi ( f ) } ( X _ { s } ) | > \lambda / 8 \} + \mathbb { 1 } \{ | f ( \bar { X } _ { s } ) - g _ { \pi ( f ) } ( \bar { X } _ { s } ) | > \lambda / 8 \} \right) \lesssim \mathrm { f a t } _ { c \lambda } ( { \mathcal F } ) \log ^ { 2 } \frac { C T } { \sigma \lambda } .\tag{32}
$$

Take $K = \lceil C \lambda ^ { - 2 } \log ( 4 T ) \rceil$ and define the fixed binary class

$$
\mathcal { F } _ { \lambda } = \left\{ x \mapsto \mathbb { 1 } \left\{ \left| \frac { 1 } { K } \sum _ { i = 1 } ^ { K } f _ { i } ( x ) \right| \geq \frac { 3 \lambda } { 4 } \right\} : f _ { 1 } , \ldots , f _ { K } \in \pm \mathcal { G } _ { \lambda } \right\} .
$$

Applying Proposition A.6 to the 2T points $X _ { 1 : T } , \bar { X } _ { 1 : T }$ , for C large enough and every $h \in \mathrm { c o n v } ( \pm \mathcal { F } )$ there exist $f _ { 1 } , \ldots , f _ { K } \in \pm \mathcal { F }$ such that

$$
\operatorname* { m a x } _ { x \in \{ X _ { s } , \bar { X } _ { s } , s \in [ T ] \} } \left| h ( x ) - \frac { 1 } { K } \sum _ { i = 1 } ^ { K } f _ { i } ( x ) \right| \leq \lambda / 8 .
$$

For each $f _ { i } ,$ consider the corresponding element $g _ { i } : = g _ { \pi ( f _ { i } ) }$ of the cover. On the event in $\operatorname { E q . }$ (32), we have

$$
\begin{array} { r l r } {  { \sum _ { s = 1 } ^ { T } \bigg ( \mathbb { 1 } \{ \underset { i \leq K } { \operatorname* { m a x } } | f _ { i } ( X _ { s } ) - g _ { i } ( X _ { s } ) | > \lambda / 8 \} + \mathbb { 1 } \{ \underset { i \leq K } { \operatorname* { m a x } } | f _ { i } ( \bar { X } _ { s } ) - g _ { i } ( \bar { X } _ { s } ) | > \lambda / 8 \} \bigg ) } \quad } & { } & \\ & { \lesssim K \mathrm { f a t } _ { c \lambda } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \lambda } . } & { } & \end{array}
$$

Call $x \in \{ X _ { s } , \bar { X } _ { s } , s \in [ T ] \}$ good if $| f _ { i } ( x ) - g _ { i } ( x ) | \le \lambda / 8$ , and call it bad otherwise. Let $\bar { g } =$ $K ^ { - 1 } \textstyle \sum _ { i = 1 } ^ { K } { g _ { i } }$ . For any good $x ,$ the Maurey approximation and the cover guarantee imply

$$
\left| h ( x ) - \bar { g } ( x ) \right| \leq \left| h ( x ) - \frac { 1 } { K } \sum _ { i = 1 } ^ { K } f _ { i } ( x ) \right| + \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left| f _ { i } ( x ) - g _ { i } ( x ) \right| \leq \frac { \lambda } { 8 } + \frac { \lambda } { 8 } = \frac { \lambda } { 4 } .
$$

Consequently, $g = \mathbb { 1 } \{ | \bar { g } | \ge 3 \lambda / 4 \} \in \mathcal { F } _ { \lambda }$ satisfies,

$$
\mathbb { 1 } \left\{ | h ( x ) | \geq \lambda \right\} \leq g ( x ) \leq \mathbb { 1 } \left\{ | h ( x ) | \geq \lambda / 2 \right\}
$$

at all but $C K \mathrm { f a t } _ { c \lambda } ( \mathcal { F } ) \log ^ { 2 } ( C T / ( \sigma \lambda ) )$ points among the two sequences. These inequalities allow us to replace the two threshold indicators by g, on all but $C K \mathrm { f a t } _ { c \lambda } ( \mathcal { F } ) \log ^ { 2 } ( C T / ( \sigma \lambda ) )$ bad points. This gives:

$$
\mathbb { E } \operatorname* { s u p } _ { h } \left\{ \sum _ { s = 1 } ^ { T } \mathbb { P } [ | h ( \bar { X } _ { s } ) | \ge \lambda ] - 3 \sum _ { s = 1 } ^ { T } \mathbb { 1 } \left\{ | h ( X _ { s } ) | \ge \lambda / 2 \right\} \right\}
$$

$$
\begin{array} { r l } & { \le \mathbb { E } \underset { h } { \operatorname* { s u p } } \left\{ \displaystyle \sum _ { s = 1 } ^ { T } \mathbb { 1 } \{ | h ( \bar { X } _ { s } ) | \ge \lambda \} - 3 \sum _ { s = 1 } ^ { T } \mathbb { 1 } \{ | h ( X _ { s } ) | \ge \lambda / 2 \} \right\} } \\ & { \lesssim K \mathrm { f a t } _ { c \lambda } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \lambda } + \mathbb { E } \underset { g \in \mathcal { F } _ { \lambda } } { \operatorname* { s u p } } \displaystyle \sum _ { s = 1 } ^ { T } \big ( g ( \bar { X } _ { s } ) - 3 g ( X _ { s } ) \big ) } \\ & { \lesssim K \mathrm { f a t } _ { c \lambda } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \lambda } + \log | \mathcal { F } _ { \lambda } | . } \end{array}\tag{33}
$$

The last inequality applies Block et al. (2024b, Theorem 10) to the fixed binary class $\mathcal { F } _ { \lambda }$ , whose Wills functional is bounded by its cardinality.

A counting argument gives $| \mathcal { F } _ { \lambda } | \le ( 2 | \mathcal { G } _ { \lambda } | ) ^ { K }$ . Using the bound on $\left| \mathcal { G } _ { \lambda } \right|$ and $\mathrm { f a t } _ { c \lambda } ( { \mathcal F } ) \le \mathrm { f a t } _ { c \varepsilon } ( { \mathcal F } )$ for $\lambda \geq 2 \varepsilon$ , the last line of (33) is at most

$$
C K \mathrm { f a t } _ { c \lambda } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \lambda } \lesssim \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \lambda ^ { - 2 } \log ^ { 3 } \frac { C T } { \sigma \lambda } .
$$

Combining Eqs. (31) and (33), taking the supremum inside the integral, and then taking expectation gives

$$
\begin{array} { r l r } {  { \mathbb { E } \operatorname* { s u p } _ { h } \{ \sum _ { s = 1 } ^ { T } \mathbb { E } _ { \bar { X } _ { s } } h ( \bar { X } _ { s } ) ^ { 2 } - 1 2 \sum _ { s = 1 } ^ { T } h ( X _ { s } ) ^ { 2 } \} \lesssim T \varepsilon ^ { 2 } + \int _ { 2 \varepsilon } ^ { 1 } 2 \lambda \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \lambda ^ { - 2 } \log ^ { 3 } \frac { C T } { \sigma \lambda } \mathrm { d } \lambda } } \\ & { } & { \lesssim T \varepsilon ^ { 2 } + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 4 } \frac { C T } { \sigma \varepsilon } , } \end{array}
$$

where the last step uses $\textstyle \int _ { \gamma _ { \varepsilon } } ^ { 1 } \lambda ^ { - 1 } \mathrm { d } \lambda \leq \log ( 1 / \varepsilon )$ . This proves (30).

If $\sigma T < 1$ , the claim follows from $| h _ { t } | \leq 1$ , so assume $\sigma T \geq 1$ . Apply Lemma B.1 to $| h _ { t } |$ and (30) on each prefix $X _ { 1 : t - 1 }$ . Summing the prefix approximation terms gives $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( t - 1 ) \varepsilon ^ { 2 } \overset { \cdot } { \leq } T ^ { 2 } \varepsilon ^ { 2 } } \end{array}$ , so

$$
\begin{array} { r } { \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } | h _ { t } ( X _ { t } ) | \lesssim \frac { \log ( 2 T ) } { \sigma } + \left[ \frac { \log ( 2 T ) } { \sigma } \left( \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } + T ^ { 2 } \varepsilon ^ { 2 } + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) T \log ^ { 4 } \frac { C T } { \sigma \varepsilon } \right) \right] ^ { 1 / 2 } } \\ { \lesssim T \varepsilon \sqrt { \frac { \log ( 2 T ) } { \sigma } } + \sqrt { \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } ) ^ { 2 } } + \sqrt { \frac { \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) T } { \sigma } } \log ^ { 5 / 2 } \frac { C T } { \sigma \varepsilon } + \frac { \log ( 2 T ) } { \sigma } , } \end{array}
$$

where the last step uses ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ and log $\cdot ( 2 T ) \leq \log ( C T / ( \sigma \varepsilon ) )$ . Since $\sigma T \geq 1$ , we have $1 / \sigma \le \sqrt { T / \sigma }$ , so the final term is absorbed into the preceding one.

## C.2 Proof of the prediction comparison on the past

We begin by proving that predictions of FTPL Eq. (3) and the reference from Eq. (29) in round t are close on past covariates $X _ { s }$ with $s < t .$

Lemma C.3. For every $t \leq T$

$$
\mathbb E \sum _ { s < t } \left( \mathbb E _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { s } ) - \mathbb E _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { s } ) \right) ^ { 2 } \lesssim \mathrm { f a t } _ { c \varepsilon } ( \mathcal F ) ^ { 2 } \log ^ { 5 } \frac { C T } { \sigma \varepsilon } + T \varepsilon ^ { 2 } + 1 .
$$

Proof of Lemma C.3. Fix the history $( X _ { 1 : t - 1 } , Y _ { 1 : t - 1 } )$ and let $G : = G ^ { ( t ) }$ to lighten the notational load. The case $t = 1$ is immediate, so suppose that $t \geq 2$ . Recall the definition of potentials from Eq. (28):

$$
\begin{array} { l } { \displaystyle \Phi _ { t } ( G ) = \displaystyle \operatorname* { s u p } _ { f \in \mathcal { F } } \left\{ - \sum _ { s < t } \left[ \ell ( f ( X _ { s } ) , Y _ { s } ) + G _ { s } f ( X _ { s } ) \right] \right\} , } \\ { \displaystyle \widetilde { \Phi } _ { t } ( G ) = \log \sum _ { g \in \mathcal { G } } \exp \left( \Phi _ { t , g } ( G ) \right) } \\ { \displaystyle \quad \quad = \operatorname* { m a x } _ { w \in \Delta ( \mathcal { G } ) } \left[ \mathbb { E } _ { g \sim w } \Phi _ { t , g } ( G ) + H ( w ) \right] , } \end{array}
$$

with $\Phi _ { t , g }$ as in Eq. (28). Note that $w _ { t } ( \cdot \mid G )$ from Eq. (29) maximize the latter potential in its variational form. Note that, by absolute continuity of the Gaussians, the set of all functions that maximize the former potential share predictions on the past covariates $X _ { s } .$ . Diferentiating the two potentials therefore gives, for every $s < t$ and almost every $G ,$ by Danskin’s theorem (Bertsekas, 2016, Appendix B) and Lemma $\mathrm { A . 7 } \mathrm { : }$

$$
\partial _ { G _ { s } } \Phi _ { t } ( G ) = - \hat { f } _ { t } ( X _ { s } ) , \qquad \partial _ { G _ { s } } \widetilde { \Phi } _ { t } ( G ) = \sum _ { g \in \mathcal { G } } w _ { t } ( g \mid G ) \partial _ { G _ { s } } \Phi _ { t , g } ( G ) .
$$

Applying Danskin’s theorem to $\Phi _ { t , g }$ , and using the definition of clipped functions ${ \bar { f } } ,$ we have:

$$
| \partial _ { G _ { s } } \Phi _ { t , g } ( G ) + g ( X _ { s } ) | \leq 2 \varepsilon ,\tag{34}
$$

and thus,

$$
\Bigl | \partial _ { G _ { s } } \widetilde { \Phi } _ { t } ( G ) + \widetilde { f } _ { t } ( X _ { s } ) \Bigr | \le 2 \varepsilon .\tag{35}
$$

Both potentials are Lipschitz. Applying Stein’s lemma (Lemma A.4) and then Lemma $\mathrm { A . 5 ( i ) }$ gives:

$$
\begin{array} { r } { \left\| \mathbb { E } _ { G } \nabla _ { G } ( \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) ) \right\| _ { 2 } ^ { 2 } \leq \mathbb { E } _ { G } ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) ^ { 2 } . } \end{array}
$$

Combining this with (35) yields

$$
\sum _ { s < t } \Big ( \mathbb { E } _ { G } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G } \tilde { f } _ { t } ( X _ { s } ) \Big ) ^ { 2 } \leq 2 \mathbb { E } _ { G } \big ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) ^ { 2 } + 8 ( t - 1 ) \varepsilon ^ { 2 } .\tag{36}
$$

It remains to bound the diference of the potentials. Since the entropy of any distribution on $\mathcal { G }$ is at most log |G|, the variational form of $\widetilde { \Phi } _ { t }$ gives

$$
\operatorname* { m a x } _ { g \in \mathcal { G } } \Phi _ { t , g } ( G ) \leq \widetilde { \Phi } _ { t } ( G ) \leq \operatorname* { m a x } _ { g \in \mathcal { G } } \Phi _ { t , g } ( G ) + \log | \mathcal { G } | .
$$

The cells $\{ f \in \mathcal { F } : g _ { \pi ( f ) } = g \}$ partition ${ \mathcal { F } } ,$ , so $\operatorname* { m a x } _ { g \in { \mathcal { G } } } \Phi _ { t , g } ( G )$ is the supremum defining $\Phi _ { t } ( G )$ with each $f$ replaced by ${ \bar { f } } .$ . For each $s < t ,$ , the map $y \mapsto \ell ( y , Y _ { s } ) + G _ { s } y$ is $\left( 1 + | G _ { s } | \right)$ -Lipschitz on [−1, 1]. Hence, on the event of Lemma C.1(ii), which has probability at least $1 - 1 / T$ over the history, using (16) to bound log |G|,

$$
\begin{array} { r l } & { \left| \Phi _ { t } ( G ) - \widetilde { \Phi } _ { t } ( G ) \right| \leq \displaystyle \log | \mathcal { G } | + \left( 1 + \operatorname* { m a x } _ { s < t } | G _ { s } | \right) \displaystyle \operatorname* { s u p } _ { f \in \mathcal { F } } \sum _ { s < t } | f ( X _ { s } ) - \bar { f } ( X _ { s } ) | } \\ & { \qquad \lesssim \left( 1 + \operatorname* { m a x } _ { s < t } | G _ { s } | \right) \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon } . } \end{array}
$$

Since $\begin{array} { r } { \mathbb { E } _ { G } ( 1 + \operatorname* { m a x } _ { s < t } | G _ { s } | ) ^ { 2 } \lesssim \log ( 2 T ) } \end{array}$ , it follows that

$$
\mathbb { E } _ { G } ( \Phi _ { t } - \widetilde { \Phi } _ { t } ) ^ { 2 } \lesssim \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) ^ { 2 } \log ^ { 5 } \frac { C T } { \sigma \varepsilon } .
$$

Plugging this into Eq. (36) gives the claimed bound on the above event. The complement of this high probability event contributes at most O(1). This concludes the proof. □

## C.3 Proof of the reduction to exponential weights

We now apply the surprise lemma Lemma C.2 to the diference of predictions.

Lemma C.4. For $\tilde { f } _ { t }$ as defined in Eq. (29), we have the following bound:

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } \bigg | \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) \bigg | \lesssim \left[ \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \sqrt { \frac { T } { \sigma } } + \frac { T \varepsilon } { \sqrt { \sigma } } \right] \log ^ { 3 } \frac { C T } { \sigma \varepsilon } + \frac { \log ( 2 T ) } { \sigma } .\tag{37}
$$

Proof of Lemma C.4. Apply Lemma C.2 to the sequence

$$
h _ { t } = \frac { 1 } { 2 } \left( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } \right) , \qquad t \in [ T ] .
$$

Note that both $\mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t }$ and $\mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t }$ are probability mixtures of elements of $\mathcal { F }$ and are predictable. Thus, $h _ { t }$ is a probability mixture of elements of $\pm \mathcal { F }$ , and using Lemma C.2 gives

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } \Big | \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) \Big | } \\ & { \quad \lesssim T  { \varepsilon } \sqrt { \frac { \log ( 2 T ) } { \sigma } } + \left( \frac { \log ( 2 T ) } { \sigma } \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { s < t } \Big ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { s } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { s } ) \Big ) ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \quad \quad + \sqrt { \frac { \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) T } { \sigma } } \log ^ { 5 / 2 } \frac { C T } { \sigma \varepsilon } . } \end{array}
$$

By Lemma C.3, the expected double sum is in $O ( T \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) ^ { 2 } \log ^ { 5 } ( C T / ( \sigma \varepsilon ) ) + T ^ { 2 } \varepsilon ^ { 2 } + T )$ . Substituting this bound and using $\log ( 2 T ) \leq \log ( C T / ( \sigma \varepsilon ) )$ gives

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } \bigg | \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) - \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) \bigg | \lesssim \left[ \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \sqrt { \frac { T } { \sigma } } + \frac { T \varepsilon } { \sqrt { \sigma } } \right] \log ^ { 3 } \frac { C T } { \sigma \varepsilon } + \frac { \log ( 2 T ) } { \sigma } .
$$

This proves (37).

## C.4 Regret of Gaussian-perturbed exponential weights

We first prove Proposition C.5, which reduces the regret bound to controlling cumulative prediction variance. We then prove Lemma C.6, which bounds the variances on the historical covariates, and combine it with the surprise lemma to prove Lemma C.7.

Proposition C.5. For every data sequence, the predictor $\tilde { f } _ { t }$ defined in $E q$ . (29) satisfies the following inequality,

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \ell ( \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \operatorname* { i n f } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \ell ( \bar { f } ( X _ { t } ) , Y _ { t } ) } } \\ & { } & { \lesssim \log | \mathcal { G } | + T \varepsilon + \sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot | G ^ { ( t ) } ) } \big ( g ( X _ { t } ) \big ) . } \end{array}\tag{38}
$$

Proof of Proposition C.5. We follow the proof of Proposition B.4, with the log-partition potential from Eq. (28)

$$
\widetilde { \Phi } _ { t } ( G ^ { ( t ) } ) = \log \sum _ { g \in \mathcal { G } } \exp \left( \Phi _ { t , g } ( G ^ { ( t ) } ) \right) , \qquad 1 \leq t \leq T + 1 .
$$

For a fixed round t, we have the following identity:

$$
\widetilde { \Phi } _ { t + 1 } ( G ^ { ( t + 1 ) } ) - \widetilde { \Phi } _ { t } ( G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) = \log \sum _ { g \in \mathcal { G } } w _ { t } ( g \mid G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) \exp \left( \Phi _ { t + 1 , g } ( G ^ { ( t + 1 ) } ) - \Phi _ { t , g } ( G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) \right) .
$$

By Lemma C.1(i), the potential increments satisfy

$$
\left| \Phi _ { t , g } ( G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) - \Phi _ { t + 1 , g } ( G ^ { ( t + 1 ) } ) - \ell ( g ( X _ { t } ) , Y _ { t } ) - G _ { t } ^ { ( t + 1 ) } g ( X _ { t } ) \right| \leq 2 \varepsilon ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) .
$$

Consequently,

$$
\begin{array} { r l } {  { \widetilde { \Phi } _ { t + 1 } ( G ^ { ( t + 1 ) } ) - \widetilde { \Phi } _ { t } ( G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) \le \log \sum _ { g \in \mathcal { G } } w _ { t } ( g \mid G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) \exp ( - \ell ( g ( X _ { t } ) , Y _ { t } ) - G _ { t } ^ { ( t + 1 ) } g ( X _ { t } ) ) } } \\ & { \qquad + 2 \varepsilon ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) . } \end{array}
$$

Apply Proposition A.3 to these weights and the losses $\ell ( g ( X _ { t } ) , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } g ( X _ { t } )$ at learning rate 1. Their range is at most $s _ { t } : = 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | )$ , and

$$
\begin{array} { r } { \operatorname { V a r } _ { g \sim w _ { t } ( \cdot | G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) } ( \ell ( g ( X _ { t } ) , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } g ( X _ { t } ) ) \leq ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) ^ { 2 } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot | G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) } ( g ( X _ { t } ) ) , } \end{array}
$$

since $u \mapsto \ell ( u , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } u$ is $( 1 + | G _ { t } ^ { ( t + 1 ) } | ) \ – \mathrm { I }$ Lipschitz. Thus, cancelling $s _ { t } ^ { 2 }$ from numerator and the denominator in Proposition $\mathrm { A . 3 }$ , we have

$$
\begin{array} { r l } {  { \sum _ { g \in \mathcal { G } } w _ { t } ( g \mid G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) [ \ell ( g ( X _ { t } ) , Y _ { t } ) + G _ { t } ^ { ( t + 1 ) } g ( X _ { t } ) ] } } \\ & { \leq \widetilde { \Phi } _ { t } ( G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) - \widetilde { \Phi } _ { t + 1 } ( G ^ { ( t + 1 ) } ) + 2 \varepsilon ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) } \\ & { \quad + \frac { \exp \Big ( 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) \Big ) - 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) - 1 } { 4 } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot \vert G _ { 1 : t - 1 } ^ { ( t + 1 ) } ) } ( g ( X _ { t } ) ) . } \end{array}
$$

Note that the last coordinate $G _ { t } ^ { ( t + 1 ) }$ is centered and independent of the first $t - 1$ coordinates. In particular,

$$
\begin{array} { r l } & { \mathbb { E } _ { G _ { t } ^ { ( t + 1 ) } } \frac { \exp { \left( 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) \right) } - 2 ( 1 + | G _ { t } ^ { ( t + 1 ) } | ) - 1 } { 4 } } \\ & { \quad \leq \frac { \exp ( 2 ) } { 4 } \mathbb { E } _ { G _ { t } ^ { ( t + 1 ) } } \left( \exp { \left( 2 G _ { t } ^ { ( t + 1 ) } \right) } + \exp { \left( - 2 G _ { t } ^ { ( t + 1 ) } \right) } \right) = \frac { \exp ( 4 ) } { 2 } . } \end{array}
$$

Now, $G _ { 1 : t - 1 } ^ { ( t + 1 ) }$ has the same law as $G ^ { ( t ) }$ conditionally on the history up to round t. Thus, using convexity of the loss and E $| G _ { t } ^ { ( t + 1 ) } | \leq 1$

$$
\begin{array} { r l } & { \ell ( \mathbb { E } _ { G ^ { ( t ) } } \widetilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) \le \mathbb { E } _ { G ^ { ( t ) } } \widetilde { \Phi } _ { t } ( G ^ { ( t ) } ) - \mathbb { E } _ { G ^ { ( t + 1 ) } } \widetilde { \Phi } _ { t + 1 } ( G ^ { ( t + 1 ) } ) } \\ & { \qquad + \ : 4 \varepsilon + \frac { \exp ( 4 ) } { 2 } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( g ( X _ { t } ) ) . } \end{array}\tag{39}
$$

Note that we have $\widetilde { \Phi } _ { 1 } = \log | \mathcal { G } |$ , and

$$
\begin{array} { r l } & { \mathbb { E } _ { G ^ { ( T + 1 ) } } \widetilde { \Phi } _ { T + 1 } ( G ^ { ( T + 1 ) } ) \geq \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \mathbb { E } _ { G ^ { ( T + 1 ) } } \left[ - \displaystyle \sum _ { s = 1 } ^ { T } \Big ( \ell ( \bar { f } ( X _ { s } ) , Y _ { s } ) + G _ { s } ^ { ( T + 1 ) } \bar { f } ( X _ { s } ) \Big ) \right] } \\ & { \quad \quad \quad = - \underset { f \in \mathcal { F } } { \operatorname* { i n f } } \displaystyle \sum _ { s = 1 } ^ { T } \ell ( \bar { f } ( X _ { s } ) , Y _ { s } ) , } \end{array}
$$

by Jensen’s inequality. Combining this with $\operatorname { E q . }$ (39) and telescoping gives the desired result.

Lemma C.6. For every $t \leq T$ and every fixed history $( X _ { s } , Y _ { s } ) _ { s < t }$ , the weights from $E q .$ (29) satisfy

$$
\sum _ { s < t } \Big ( \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } \big ( g ( X _ { s } ) \big ) \Big ) ^ { 2 } \lesssim \log ( 2 \vert \mathcal { G } \vert ) + T \varepsilon ^ { 2 } .
$$

Proof of Lemma C.6. Write $G = G ^ { ( t ) }$ throughout this proof for notational brevity. From Lemma A.7, we have

$$
\partial _ { G _ { s } } \tilde { f } _ { t } ( X _ { s } ) = \mathrm { C o v } _ { g \sim w _ { t } ( \cdot | G ) } ( g ( X _ { s } ) , \partial _ { G _ { s } } \Phi _ { t , g } ( G ) ) ,
$$

Using Eq. (34), we have

$$
\begin{array} { r } { \Bigl | \partial _ { G _ { s } } \tilde { f } _ { t } ( X _ { s } ) + \operatorname { V a r } _ { g \sim w _ { t } ( \cdot | G ) } ( g ( X _ { s } ) ) \Bigr | \le 2 \varepsilon . } \end{array}
$$

After applying Lemma A.4, we thus get:

$$
\begin{array} { r } { \mathbb { E } _ { G } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot | G ) } ( g ( X _ { s } ) ) \leq - \mathbb { E } _ { G } \left[ G _ { s } \tilde { f } _ { t } ( X _ { s } ) \right] + 2 \varepsilon . } \end{array}
$$

To bound the sum of correlations, we invoke part (ii) of Lemma A.5. This gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { s < t } \left( \mathbb { E } _ { G } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot \vert G ) } ( g ( X _ { s } ) ) \right) ^ { 2 } \lesssim \sum _ { s < t } \left( \mathbb { E } _ { G } \left[ G _ { s } \tilde { f } _ { t } ( X _ { s } ) \right] \right) ^ { 2 } + \varepsilon ^ { 2 } t } \\ & { \qquad \lesssim \log ( 2 \vert \mathcal { G } \vert ) + \varepsilon ^ { 2 } t , } \end{array}
$$

which concludes the proof.

Finally, combining the preceding bounds, we obtain the following result.

Lemma C.7. The regret of the modified exponential weights algorithm from Eq. (29) satisfies

$$
\begin{array} { r l } & { \mathbb E \left[ \displaystyle \sum _ { t = 1 } ^ { T } \ell ( \mathbb E _ { G ^ { ( t ) } } \widetilde f _ { t } ( X _ { t } ) , Y _ { t } ) - \displaystyle \operatorname* { i n f } _ { f \in \mathcal F } \displaystyle \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \right] } \\ & { \quad \lesssim \mathrm { f a t } _ { c \varepsilon } ( \mathcal F ) \log ^ { 2 } \displaystyle \frac { C T } { \sigma \varepsilon } + T \varepsilon + \sqrt { \frac { T } { \sigma } } \left[ \sqrt { \log ( 2 | \mathcal G | ) } \log ^ { 2 } \displaystyle \frac { 2 T } { \sigma } + \sqrt { T \log ( 2 T ) } \varepsilon \right] + \displaystyle \frac { \log ( 2 T ) } { \sigma } . } \end{array}
$$

Proof of Lemma C.7. Apply Proposition C.5 to the cover from Proposition A.2. The diference in the comparator terms over the clipped functions and $\mathcal { F }$ is upper bounded as:

$$
\left| \operatorname* { i n f } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \ell ( \bar { f } ( X _ { t } ) , Y _ { t } ) - \operatorname* { i n f } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \right| \leq \operatorname* { s u p } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } | \bar { f } ( X _ { t } ) - f ( X _ { t } ) | ,
$$

since $\ell ( \cdot , Y _ { t } )$ is 1-Lipschitz. The right-hand side is at most $2 T ,$ so, by Lemma C.1(ii), its expectation is at most $C \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } ( C T / ( \sigma \varepsilon ) ) + O ( 1 )$ . Thus, applying this estimate to (38), we have:

$$
\begin{array} { r l } & { \mathbb E \left[ \displaystyle \sum _ { t = 1 } ^ { T } \ell ( \mathbb E _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \displaystyle \operatorname* { i n f } _ { f \in \mathcal F } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \right] } \\ & { \quad \lesssim \log | \mathcal G | + T \varepsilon + \mathbb E \displaystyle \sum _ { t = 1 } ^ { T } \mathbb E _ { G ^ { ( t ) } } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot \vert G ^ { ( t ) } ) } ( g ( X _ { t } ) ) + \mathrm { f a t } _ { c \varepsilon } ( \mathcal F ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon } . } \end{array}\tag{40}
$$

The above reduces the proof to bounding the cumulative prediction variance. We know from Lemma C.6 that the variance of $g ( X _ { s } )$ under $g \sim w _ { t }$ on the past covariates $X _ { s } , s < t$ , can be suitably upper bounded. To upper bound the above, we will show that variance terms can be expressed as predictable test functions in the convex hull of a finite class, and apply Corollary B.3. Indeed, for every t and $x \in \mathcal { X }$ , we have

$$
\frac { 1 } { 2 } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { g \sim w _ { t } ( \cdot | G ^ { ( t ) } ) } ( g ( x ) ) = \mathbb { E } _ { G ^ { ( t ) } } \sum _ { g , g ^ { \prime } \in \mathcal { G } } w _ { t } ( g \mid G ^ { ( t ) } ) w _ { t } ( g ^ { \prime } \mid G ^ { ( t ) } ) \frac { ( g ( x ) - g ^ { \prime } ( x ) ) ^ { 2 } } { 4 } .
$$

This is a predictable test-function that belongs to the convex hull of the squared-diference class:

$$
\left\{ x \mapsto ( g ( x ) - g ^ { \prime } ( x ) ) ^ { 2 } / 4 : g , g ^ { \prime } \in \mathcal { G } \right\} .
$$

It has at most $| \mathcal { G } | ^ { 2 }$ elements. $\operatorname { I f } | { \mathcal { G } } | = 1$ , all variances vanish; otherwise, using Lemma C.6 and applying Corollary B.3 to the sequence of variances, with the finite class from above in place of $\mathcal { F }$ gives

$$
\begin{array} { r l } {  { \mathbb { E } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \operatorname { V a r } _ { g \sim w t ^ { ( \cdot | G ^ { ( t ) } ) } } ( g ( X _ { t } ) ) } } \\ & { \lesssim \sqrt { \frac { T \log ( 2 T ) } { \sigma } ( \log ( 2 | \mathcal { G } | ) + T \varepsilon ^ { 2 } ) } + \sqrt { \frac { T \log | \mathcal { G } | } { \sigma } } \log ^ { 2 } \frac { 2 T } { \sigma } + \frac { \log ( 2 T ) } { \sigma } . } \end{array}
$$

Plugging this into Eq. (40) yields:

$$
\begin{array} { r l } & { \mathbb E \left[ \displaystyle \sum _ { t = 1 } ^ { T } \ell ( \mathbb E _ { G ^ { ( t ) } } \widetilde f _ { t } ( X _ { t } ) , Y _ { t } ) - \displaystyle \operatorname* { i n f } _ { f \in \mathcal F } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \right] } \\ & { \quad \lesssim \log | \mathcal G | + T \varepsilon + \sqrt { \frac { T \log ( 2 T ) } { \sigma } \left( \log ( 2 | \mathcal G | ) + T \varepsilon ^ { 2 } \right) } } \\ & { \quad \quad + \sqrt { \frac { T \log | \mathcal G | } { \sigma } } \log ^ { 2 } \frac { 2 T } { \sigma } + \frac { \log ( 2 T ) } { \sigma } + \mathrm { f a t } _ { c \varepsilon } ( \mathcal F ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon } . } \end{array}
$$

To obtain the stated regret bound, we use Proposition A.2 to bound log |G| and use ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } } .$ This concludes the proof. □

## C.5 Proof of the real-valued theorem

The preceding bounds control the Gaussian mean of FTPL. It remains to account for the finite average in (7) and combine the prediction and reference-regret bounds.

Theorem 3.2. Consider a class $\mathcal { F } \colon \mathcal { X }  [ - 1 , 1 ]$ and let $\ell ( \hat { y } , y )$ be convex and 1-Lipschitz in its first argument, and suppose the covariate process is σ-smooth with respect to a fixed $\mu .$ Then, the learner in (7) satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } \lesssim \operatorname* { i n f } _ { 0 < \varepsilon \leq 1 / 1 6 } \left\{ \left( 1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) + \sqrt { T } \varepsilon \right) \sqrt { \frac { T } { \sigma } } \log ^ { 3 } \left( \frac { 2 T } { \sigma \varepsilon } \right) \right\} ,\tag{8}
$$

where c is a universal constant.

Proof of Theorem 3.2. Fix $0 < \varepsilon \le 1 / 1 6$ with $\mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) < \infty . \mathrm { ~ I f ~ } 1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) > T$ or $T < 1 / \sigma$ , then 2T is at most a constant times the expression inside the infimum in (8), so the bound at this ε follows from $\mathrm { R e g } _ { T } \leq 2 T$ . Otherwise, $1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \leq T$ and $T \geq 1 / \sigma$ , and we construct the fixed cover and the modified exponential weights algorithm from Eq. (29).

Conditional on $\mathcal { H } _ { t - 1 } , X _ { t } , Y _ { t }$ , the average of the t independent oracle predictions has variance at most $1 / t$ about $\mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } )$ . Cauchy–Schwarz and Lipschitzness give

$$
\begin{array} { r } { \mathbb { E } \left[ \ell ( \hat { Y } _ { t } , Y _ { t } ) - \ell ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) , Y _ { t } ) \mid \mathcal { H } _ { t - 1 } , X _ { t } , Y _ { t } \right] \leq t ^ { - 1 / 2 } . } \end{array}
$$

Summing over t and using $\begin{array} { r } { \sum _ { t \leq T } t ^ { - 1 / 2 } \leq 2 \sqrt { T } } \end{array}$ , we obtain

$$
\mathbb { E } \mathrm { R e g } _ { T } \leq 2 \sqrt { T } + \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \ell ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \operatorname* { i n f } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \right] .\tag{41}
$$

We decompose the regret of the Gaussian mean on the right-hand side as

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \ell ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \operatorname* { i n f } _ { f \in \mathcal { F } } \displaystyle \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) = \sum _ { t = 1 } ^ { T } \left[ \ell ( \mathbb { E } _ { G ^ { ( t ) } } \hat { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \ell ( \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) \right] } \\ & { \quad \quad \quad \quad \quad + \left[ \displaystyle \sum _ { t = 1 } ^ { T } \ell ( \mathbb { E } _ { G ^ { ( t ) } } \tilde { f } _ { t } ( X _ { t } ) , Y _ { t } ) - \operatorname* { i n f } _ { f \in \mathcal { F } } \displaystyle \sum _ { t = 1 } ^ { T } \ell ( f ( X _ { t } ) , Y _ { t } ) \right] . } \end{array}
$$

Since the loss is 1-Lipschitz, the first sum is bounded by Lemma C.4 in expectation. Lemma C.7 bounds the expectation of the second bracket. Plugging these bounds into (41), and using (16) to bound $\sqrt { \log ( 2 | \mathcal { G } | ) } \lesssim \sqrt { 1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) } \log ( C T / ( \sigma \varepsilon ) )$ and lo $\lceil 2 T / \sigma ) \leq \log ( C T / ( \sigma \varepsilon ) )$ , yields

$$
\begin{array} { r l } & { \mathbb { E } \mathrm { R e } _ { T } \lesssim \sqrt { T } + \left[ ( 1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) ) \sqrt { \frac { T } { \sigma } } + \frac { T \varepsilon } { \sqrt { \sigma } } \right] \log ^ { 3 } \frac { C T } { \sigma \varepsilon } + \frac { \log ( 2 T ) } { \sigma } } \\ & { \quad \quad \quad \quad + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) \log ^ { 2 } \frac { C T } { \sigma \varepsilon } + T \varepsilon } \\ & { \quad \quad \quad \quad + \sqrt { \frac { T } { \sigma } } \left[ \sqrt { 1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) } \log ^ { 3 } \frac { C T } { \sigma \varepsilon } + \sqrt { T \log ( 2 T ) } \varepsilon \right] . } \end{array}
$$

To obtain (8), note that each term on the right-hand side is at most a constant times

$$
\left( 1 + \mathrm { f a t } _ { c \varepsilon } ( \mathcal { F } ) + \sqrt { T } \varepsilon \right) \sqrt { \frac { T } { \sigma } } \log ^ { 3 } \frac { C T } { \sigma \varepsilon } .
$$

For the term $\log ( 2 T ) / \sigma ,$ , this uses $T \geq 1 / \sigma$ , which gives $1 / \sigma \leq \sqrt { T / \sigma } ;$ for the remaining terms, it uses $\sigma \leq 1$ and $\log ( C T / ( \sigma \varepsilon ) ) \geq 1$ . Moreover, $T / ( \sigma \varepsilon ) \geq 1 6$ gives $\log ( C T / ( \sigma \varepsilon ) ) \lesssim \log ( 2 T / ( \sigma \varepsilon ) )$ . The learner does not depend on $\varepsilon ,$ so taking the infimum over ε proves (8). □

## D $L ^ { \star } .$ -dependent regret

In this section, we demonstrate that Gaussian FTPL is capable of achieving a regret bound that adapts to the loss of the best comparator in binary classification. In particular, we derive bound similar to Theorem 3.1 in terms of the best comparator’s number of mistakes, denoted by:

$$
L ^ { \star } = \operatorname* { m i n } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ f ( X _ { t } ) \neq Y _ { t } \} .
$$

Theorem D.1. Consider a binary class $\mathcal { F } \colon \mathcal { X } \to \{ \pm 1 \}$ ofVC dimension d $\geq 1$ , and let $\ell ( \hat { y } , y ) = - \hat { y } y$ be the binary loss. For every horizon $T \geq 1$ and every covariate process that is σ-smooth with respect to a fixed $\mu ,$ the learner in (3) satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } \lesssim \sqrt { \frac { T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) \left( \sqrt { d ^ { 2 } \wedge \mathbb { E } L ^ { \star } } + \sqrt { d } \right) .\tag{42}
$$

The learner does not require knowledge of $L ^ { \star }$ . In particular, in the realizable case, $L ^ { \star } = 0$ almost surely, thus, Gaussian FTPL recovers the bound

$$
\mathbb { E } \mathrm { R e g } _ { T } \lesssim \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) ,
$$

which, up to logarithmic factors, is optimal in the realizable case (Blanchard, 2025). Note that, in order to establish Theorem D.1, it sufices to prove:

$$
\mathbb { E } \mathrm { R e g } _ { T } \lesssim \sqrt { \frac { T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) ( \sqrt { \mathbb { E } L ^ { \star } } + \sqrt { d } ) ,
$$

since the second point in the minimum is given by Theorem 3.1. The proof of the above statement is shorter than Theorem 3.1 and is more reminiscent of the proof for ERM (Block et al., 2024b). We first bound the expected number of mistakes that Gaussian FTPL makes on the past covariates.

Lemma D.2. Let ${ \mathcal { F } } \subseteq \{ \pm 1 \} ^ { \mathcal { X } }$ have VC dimension $d \geq 1$ . Fix a history $( X _ { 1 : t - 1 } , Y _ { 1 : t - 1 } )$ , and write $G = G ^ { ( t ) }$ . The forecaster in (3) satisfies

$$
\mathbb { E } _ { G } \sum _ { s < t } { \mathbb { 1 } \{ { \hat { f } } _ { t } ( X _ { s } ) \neq Y _ { s } \} } \le 2 \operatorname* { m i n } _ { f \in { \mathcal { F } } } \sum _ { s < t } { \mathbb { 1 } \{ f ( X _ { s } ) \neq Y _ { s } \} } + 2 d \log ( 2 t ) .\tag{43}
$$

Proof. The claim is immediate for $t = 1$ . Fix $t \geq 2$ and $g \in { \mathcal { F } }$ . All expectations are over $G ,$ with the history held fixed. Since $( f ( X _ { s } ) - Y _ { s } ) ^ { 2 } = 2 ( 1 - Y _ { s } f ( X _ { s } ) )$ , optimality of $\hat { f } _ { t }$ in (3) gives the basic inequality

$$
\sum _ { s < t } ( \hat { f } _ { t } ( X _ { s } ) - Y _ { s } ) ^ { 2 } \leq \sum _ { s < t } ( g ( X _ { s } ) - Y _ { s } ) ^ { 2 } + 2 \sum _ { s < t } G _ { s } ( g ( X _ { s } ) - \hat { f } _ { t } ( X _ { s } ) ) .
$$

Split $g ( X _ { s } ) - \hat { f } _ { t } ( X _ { s } ) = \left( g ( X _ { s } ) - Y _ { s } \right) + \left( Y _ { s } - \hat { f } _ { t } ( X _ { s } ) \right)$ and subtract half the learner’s squared error. Taking expectations yields:

$$
\frac 1 2 \mathbb { E } _ { G } \sum _ { s < t } ( \hat { f } _ { t } ( X _ { s } ) - Y _ { s } ) ^ { 2 } \leq \sum _ { s < t } ( g ( X _ { s } ) - Y _ { s } ) ^ { 2 }
$$

$$
+ \mathbb { E } _ { G } \operatorname* { m a x } _ { f \in \mathcal { F } } \left\{ 2 \sum _ { s < t } G _ { s } ( Y _ { s } - f ( X _ { s } ) ) - \frac { 1 } { 2 } \sum _ { s < t } ( f ( X _ { s } ) - Y _ { s } ) ^ { 2 } \right\} .
$$

The last term can be interpreted as an ofset Gaussian complexity (Liang et al., 2015). From Gaussian MGF, we have:

$$
\mathbb { E } _ { G } \exp \left( \frac { 1 } { 2 } \sum _ { s < t } G _ { s } ( Y _ { s } - f ( X _ { s } ) ) - \frac { 1 } { 8 } \sum _ { s < t } ( f ( X _ { s } ) - Y _ { s } ) ^ { 2 } \right) = 1 .
$$

Applying Jensen’s inequality conditional on the values $X _ { 1 : t - 1 }$ and summing over the distinct prediction vectors on these covariates gives

$$
\begin{array} { r l } & { \mathbb { E } _ { G } \underset { f \in \mathcal { F } } { \operatorname* { m a x } } \left\{ 2 \displaystyle \sum _ { s \in \mathcal { L } } G _ { s } ( Y _ { s } - f ( X _ { s } ) ) - \frac { 1 } { 2 } \displaystyle \sum _ { s < t } ( f ( X _ { s } ) - Y _ { s } ) ^ { 2 } \right\} } \\ & { \quad \le 4 \log \mathbb { E } _ { G } \exp \left( \underset { f \in \mathcal { F } } { \operatorname* { m a x } } \left\{ \frac { 1 } { 2 } \displaystyle \sum _ { s < t } G _ { s } ( Y _ { s } - f ( X _ { s } ) ) - \frac { 1 } { 8 } \displaystyle \sum _ { s < t } ( f ( X _ { s } ) - Y _ { s } ) ^ { 2 } \right\} \right) } \\ & { \quad \le 4 \log \mathbb { E } _ { G } \underset { v \in \mathcal { F } | _ { X _ { 1 : t - 1 } } } { \sum } \exp \left\{ \frac { 1 } { 2 } \displaystyle \sum _ { s < t } G _ { s } ( Y _ { s } - v _ { s } ) - \frac { 1 } { 8 } \displaystyle \sum _ { s < t } ( v _ { s } - Y _ { s } ) ^ { 2 } \right\} } \\ & { \quad \le 4 \log \left| \mathcal { F } | _ { X _ { 1 : t - 1 } } \right| \le 4 d \log ( 2 t ) , } \end{array}
$$

where the last inequality follows from Sauer–Shelah (part (i) of Proposition A.1). Substituting this bound and using $( f ( X _ { s } ) - Y _ { s } ) ^ { 2 } = 4 \mathbb { 1 } \{ f ( X _ { s } ) \neq Y _ { s } \}$ gives

$$
\mathbb { E } _ { G } \sum _ { s < t } \mathbb { 1 } \left\{ { \hat { f } } _ { t } ( X _ { s } ) \neq Y _ { s } \right\} \leq 2 \sum _ { s < t } \mathbb { 1 } \left\{ g ( X _ { s } ) \neq Y _ { s } \right\} + 2 d \log ( 2 t ) .
$$

Minimizing over $g \in { \mathcal { F } }$ proves the claim.

The proof of the regret bound is concluded by applying the surprise lemma Lemma 4.1 to translate Lemma D.2 into a guarantee on the current prediction loss.

Proof of Theorem D.1. The idea is to apply Lemma 4.1 to the process $( X _ { t } , Y _ { t } )$ . First, note that conditional law of $( X _ { t } , Y _ { t } )$ given $\mathcal { H } _ { t - 1 }$ is $\sigma / 2 \AA$ -smooth with respect to $\mu \otimes { \mathbf u } { \boldsymbol { \mathrm { n i f } } } ( \{ \pm 1 \} )$ . Let

$$
h _ { t } ( x , y ) : = \mathbb { E } _ { G ^ { ( t ) } } \frac { 1 - y \hat { f } _ { t } ( x ) } { 2 } = \mathbb { E } _ { G ^ { ( t ) } } \mathbb { 1 } \{ \hat { f } _ { t } ( x ) \neq y \} .
$$

Note that $h _ { t }$ is a predictable test function which belongs to the signed convex hull of the class

$$
\{ ( x , y ) \mapsto y f ( x ) : f \in { \mathcal { F } } \} \cup \{ 1 \} ,
$$

whose VC dimension is in $O ( d )$ . Since $h _ { t } ^ { 2 } \leq h _ { t }$ , we have

$$
\begin{array} { r l r } {  { \mathbb { E } \sum _ { t = 1 } ^ { T } \sum _ { s < t } h _ { t } ( X _ { s } , Y _ { s } ) ^ { 2 } \le \mathbb { E } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { G ^ { ( t ) } } \sum _ { s < t } { \mathbb { 1 } \{ \hat { f } _ { t } ( X _ { s } ) \neq Y _ { s } \} } } } \\ & { } & { \lesssim T \mathbb { E } L ^ { \star } + d T \log ( 2 T ) , ~ } \end{array}\tag{44}
$$

where the second transition used Lemma D.2. Therefore, using Lemma 4.1, we have

$$
\begin{array} { r l r } {  { \mathbb { E } \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ \hat { f } _ { t } ( X _ { t } ) \neq Y _ { t } \} = \mathbb { E } \sum _ { t = 1 } ^ { T } h _ { t } ( X _ { t } , Y _ { t } ) } } \\ & { } & { \lesssim \sqrt { \frac { T \log ( 2 T ) } { \sigma } ( \mathbb { E } L ^ { \star } + d \log ( 2 T ) ) } + \sqrt { \frac { d T } { \sigma } \log ^ { 2 } ( 2 T / \sigma ) } } \\ & { } & { \lesssim \sqrt { \frac { T \log ( 2 T ) } { \sigma } \mathbb { E } L ^ { \star } } + \sqrt { \frac { d T } { \sigma } } \log ^ { 2 } ( 2 T / \sigma ) . ~ } \end{array}
$$

The first equality uses the conditional independence of $( X _ { t } , Y _ { t } )$ and $G ^ { ( t ) }$ given $\mathcal { H } _ { t - 1 }$ . Since $\mathrm { R e g } _ { T }$ is at most a factor of 2 times number of mistakes, the inequality above concludes the proof. □