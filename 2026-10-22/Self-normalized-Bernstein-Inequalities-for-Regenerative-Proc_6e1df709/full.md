# Self-normalized Bernstein Inequalities for Regenerative Processes and Applications to Learning Without Independence

Imon Banerjee<sup>\*1</sup> and Pierre Alquier<sup>∗†2</sup>

<sup>1</sup>University of Chicago, Booth School of Business <sup>2</sup>ESSEC Business School, Asia-Pacific campus, Singapore

## Abstract

We prove a self-normalized Bernstein inequality for bounded functions of a regenerative process with independent, identically distributed marked cycles of finite mean length. The bound uses observed cycle lengths and retains explicit terms for truncation and the cycle ending after the observation horizon. Selecting a tuning parameter from a finite deterministic collection gives a version of Bernstein inequality where all terms are empirical but one, related to the tails of the regeneration times. We then study this term in many regimes: it turns out that it is of second-order in many cases of interest. This is important for applications in machine learning: we prove generalization error bounds for classification with non-i.i.d., regenerative data. These bounds can be used to define learning objectives: by minimizing the first-order term only in the bound, we define agnostic classifiers that still satisfy oracle inequalities. We also show that the bound on the average deviation provided by our results is minimax-optimal in the sample size and in the expected regeneration time. Finally, we combine the Bernstein inequality with a regenerative strong approximation to obtain a Gaussian approximation using an observable block variance estimator.

Keywords: Regenerative processes; empirical Bernstein inequalities; PAC-Bayes bounds; Gaussian approximation.

## 1 Introduction

Bernstein inequalities are fundamental tools in statistics and machine learning. By controlling deviations of empirical averages from their expectations through a variance term, they support confidence intervals, model selection and guarantees on the performance of learning algorithms [6, 14]. To use such guarantees in practice, the quantities appearing in the bound must be available from the observations or specified in advance. A bound involving unknown features of the underlying distribution cannot be evaluated directly and may also be difficult to use as a learning criterion. This motivates empirical, or agnostic, bounds, which can be evaluated without knowing the data-generating distribution. For independent and identically distributed (i.i.d.) bounded observations, empirical Bernstein inequalities replace the population variance with an observable estimate and an explicit correction [31]. These ideas also extend to PAC–Bayes guarantees for learning [45].

For time series, temporal dependence creates an additional obstacle. Classical extensions of Bernstein inequalities typically involve mixing coefficients or, for Markov chains, a spectral or pseudo-spectral gap [39, 42]. These quantities describe the strength of dependence but are generally unknown, so the resulting bounds cannot be evaluated directly from a single observed trajectory. Although they provide useful rates, their dependence on unknown parameters limits their direct application to statistical inference and the design of learning criteria. Estimating these parameters is one possible response. Here, we develop bounds that use observed regeneration times to account for temporal dependence. In this case, the only non-empirical term is in many regimes a second-order term. In machine learning, we show that ignoring these terms lead to agnostic classification rules who satisfy strong oracle inequalities.

We study bounded losses along a regenerative process. Regeneration divides a trajectory into cycles, allowing dependence within each cycle while providing independence between cycles. We assume that the marked cycles are i.i.d., have finite mean length and are independent of the initial delay. This framework includes regenerative Markov chains [11] as well as processes whose observations are not Markovian.

Two features require care: (i) a bounded loss can have an unbounded cycle reward, and (ii) the number of completed cycles is random. We truncate the centred cycle rewards according to cycle length and construct an exponential supermartingale, which we stop at the first cycle ending after time n. Removing that cycle yields the self-normalized Bernstein inequality in Theorem 3.1. Selecting its tuning parameter from a finite deterministic collection gives the empirical Bernstein bound in Corollary 3.1. Its quadratic term is a sum of observed squared cycle lengths below the truncation level, common to all bounded losses. The bound also retains the observed contribution of longer cycles, a correction for the cycle crossing the observation horizon, and an explicit population tail expectation arising from truncation. Thus, the dependence on the cycle structure is expressed through observable quantities and a separate tail remainder, without estimating mixing coefficients or a spectral gap.

The main application we have in mind is generalization error bounds in machine learning. Here, we use our version of Bernstein inequality to prove PAC-Bayes generalization bounds for classifiers learned on non-i.i.d., regenerative data. The associated Gibbs posterior satisfies oracle inequalities both in probability and in expectation.

We then study the optimality of the scale of our bounds. Under a normalized moment condition of order greater than two on the cycle lengths, and starting at a regeneration time, we obtain a fixed-confidence estimation error of order $\sqrt { m / n }$ , where m is the mean cycle length. For integer m, a matching lower bound establishes the minimax optimality of this dependence in Theorem 6.1. The effective number of independent observations is therefore of order $n / m$ in this comparison. Integrating the Bernstein inequality also gives expected deviation bounds, which we specialize under exponential and polynomial regeneration tails.

Finally, we connect the concentration bounds to asymptotic inference. Under the moment and initialization assumptions used for the minimax analysis, for a fixed loss and regenerative law with positive asymptotic variance, we use the Bernstein inequality to control empirical centering in an observable block variance estimator. Combining this control with the regenerative strong approximation of Bashtova and Shashkin [10] yields a self-normalized Gaussian approximation and studentized asymptotic normality.

The rest of the paper is organised as follows. Subsection 1.1 reviews the related literature, and Section 2 introduces the model and notation. Section 3 establishes the Bernstein and expected deviation bounds. While the first-order terms in these bounds are empirical, one of the second order terms is more problematic. In Section 4, we control this term in various regimes. Section 5 presents the PAC-Bayes and oracle inequalities. In particular, we show that ignoring the problematic terms to define a learning objective does not affect the properties of machine learning algorithms. Section 6 shows that confidence intervals derived from our version of Bernstein inequality are minimax optimal. Section 7 studies the self-normalized Gaussian approximation. Sections 8 contains the proofs of all the results in the paper.

## 1.1 Background and Related Literature

Concentration inequalities for regenerative processes use independence between cycles while allowing dependence within each cycle. We refer to the monograph by Bertail and Clemenc¸on [´ 11] for detailed literature survey of regenerative Markov chains. For a positive recurrent Markov chain with an accessible atom, Bertail and Clemenc¸on [´ 12] derive tail inequalities under finite second moments of the cycle lengths and centered rewards, with positive reward variance. Their bounds use truncated population moments and retain probabilities of long cycles and large rewards. Finally, [16] derive Rosenthal-type moment bounds and Bernstein-type inequalities for additive functionals of geometrically ergodic Markov chains, with leading terms governed by the asymptotic variance and explicit dependence on the mixing time in the remainder terms.

Empirical-process results use regeneration to control a class of functions. For Markov chains with a stationary law, small-set minorization and exponential moment conditions on the initial and subsequent regeneration waiting times, Adamczak [2] proves concentration for bounded centered additive functionals. Under one-step minorization, they also control suprema over countable, uniformly bounded, centered classes, retaining their expected supremum and a population bound on the variances of cycle rewards. For positive Harris recurrent, aperiodic chains with one-step minorization, Bertail and Portier [13] develop block Rademacher complexity bounds for uniformly bounded VC classes. Their bounds combine a uni form second-moment bound for block rewards with either a cycle moment of order greater than one or an exponential cycle moment. When combined with [2], this leads to novel Dvoretzky-Kiefer-Wolfowitz (DKW) type inequalities for exponentially regenerating Markov chains [7, 9].

Concentration bounds can also retain random quantities from the trajectory. Under geometric drift, onestep minorization on Lyapunov sublevel sets, and well-defined conditional second moments, Wintenberger [46] proves an exponential inequality for path functionals satisfying a weighted bounded-difference condition. Its quadratic normalizer combines observed values of the Lyapunov function with transition-dependent conditional moments and coefficients. More directly, Jerison [24] gives a data-dependent Dvoretzky–Kiefer– Wolfowitz inequality, uniform over thresholds of one real-valued observable. They assume i.i.d. regeneration cycles, initialization at regeneration, identifiable boundaries, and a known exponential bound on the tail of the cycle length. The leading term uses observed cycle lengths, while a correction retains the tail constants. The sample contains a deterministic number of cycles, so its observation horizon is random.

The theory of generalization for machine learning algorithms was essentially developed for independent observations, see Bach [6], Mohri et al. [38] for an introduction. However, there were notable attempts to extend some results to the setting of time series and stochastic processes, under various mixing and weakdependence assumptions [1, 20, 28–30, 33, 34, 36, 37, 43, 44, 44, 50]. Unfortunately, while the theory in the i.i.d. case allows to define to distribution free, agnostic criteria for training and model selection, all the results in the dependent case require some prior knowledge on the strength of the temporal dependence. A class of generalization bounds known as PAC-Bayes bounds, introduced by [32] in the i.i.d. setting, which can be used both as criteria to optimize when training algorithms, and to derive powerful oracle inequalities [4, 15, 23]. PAC-Bayes bounds were also extended to beyond independent observations: Alquier and Wintenberger [5], Banerjee et al. [8], Eringis et al. [18], Fard et al. [19], Ralaivola et al. [40]. These results suffer the same limitation.

This led to a trend in the recent literature, which aims at estimating the unknown mixing coefficients involved in these bounds [21, 27, 47, 49]. For Markov chains, [26, 48] studied the estimation of the pseudospectral gap (in the case of finite, and compact continuous state-space, respectively). Together with Paulin’s inequality [39], this leads to agnostic deviation inequalities [25, 26]. Our results go beyond these works in many ways. First, out bound holds for general regenerative process, including non-Markovian ones. Then, even in the Markov case with a discrete state-space, our results are not limited to a finite state-space as in [25, 48]: they can accommodate countable infinite state-spaces. Finally, our inequalities do not require the estimation of any mixing coefficient or spectral gap, which makes them easier to use in practice. It might seems that a drawback of our approach is that only the first order terms in our inequalities are empirical. In other words, we can derive asymptotic empirical confidence intervals from our bounds, but the nonasymptotic coverage is lost. We point out that this is also the case in [26]. In particular, this does not prevent from building efficient, agnostic learning rules.

## 2 Statement of the problem and notations

We summarize important notations in Table 1. Let $\mathbb { N } : = \{ 1 , 2 , . . . \}$ and $ { \mathbb { N } } _ { 0 } : = \{ 0 , 1 , 2 , . . . \}$ . Let $Z =$ $( Z _ { t } ) _ { t \in \mathbb { N } }$ take values in a measurable space $( \boldsymbol { \ Z } , \boldsymbol { \mathcal { Z } } )$ , and let $( { \mathcal { G } } _ { t } ) _ { t \in  { \mathbb { N } } _ { 0 } }$ be a filtration to which $Z$ is adapted. This filtration includes the variables used to identify regeneration times, such as the auxiliary marks in a split Markov chain.

Definition 2.1. The process $Z$ is regenerative if there are almost surely finite, integer-valued stopping times

$$
0 \leq \tau _ { 1 } < \tau _ { 2 } < \cdots
$$

such that, for every $j \in \mathbb N$ , the shifted process $( Z _ { \tau _ { j } + t } ) _ { t \in \mathbb { N } }$ is independent of $\mathcal { G } _ { \tau _ { j } }$ and has a law that does not depend on $j .$ . We denote the common post-regeneration law by $\mathbb { P } _ { A }$ and its expectation by $\mathbb { E } _ { A }$

As previously mentioned, all Harris recurrent Markov chains can acquire a regenerative structure through Nummelin splitting [11]. Furthermore, if $X _ { t }$ takes values in a measurable space $\mathsf { X }$ and $g : { \mathsf { X } } \to { \mathsf { Y } }$ is a fixed measurable map into a measurable space ${ \mathsf { Y } } ,$ , then $( X _ { t } , g ( X _ { t } ) )$ has the same regeneration times as $X$ . Non-Markovian regenerative models are popular in queueing and storage models [35, Appendix C].

For $j \in \mathbb N$ , define the cycle length and block by

$$
\Delta _ { j } : = \tau _ { j + 1 } - \tau _ { j } , \qquad B _ { j } : = ( Z _ { \tau _ { j } + 1 } , \ldots , Z _ { \tau _ { j + 1 } } ) .
$$

Throughout the paper, we assume that the marked cycles $( \Delta _ { j } , B _ { j } ) _ { j \in \mathbb { N } }$ are i.i.d. with their common regenerative law, have positive integer lengths, and are independent of $\tau _ { 1 }$ . We also assume

$$
\mathbb { E } _ { A } [ \Delta _ { 1 } ] < \infty .
$$

We use $\mathbb { P } _ { A }$ and $\mathbb { E } _ { A }$ for the joint post-regeneration law and expectation, including the cycle marks. We observe the regeneration boundaries—i.e. we know the exact regeneration times—as well as the process. These marked-cycle assumptions specify the independence used in the results below; the definition of the post-regeneration observation process alone does not impose a common rule for marking the next cycle.

The process may start from an arbitrary initial distribution. Define its equilibrium occupation measure by, for any $C \in { \mathcal { Z } }$

$$
\bar { \pi } ( C ) : = \frac { \mathbb { E } _ { A } \left[ \sum _ { t = \tau _ { 1 } + 1 } ^ { \tau _ { 2 } } \mathbf { 1 } _ { \{ Z _ { t } \in C \} } \right] } { \mathbb { E } _ { A } [ \Delta _ { 1 } ] } .
$$

Given a (measurable) function $f : Z \to [ 0 , 1 ]$ , our objective is to provide empirical confidence bounds on

$$
\bar { \pi } ( f ) = \int f ( z ) \bar { \pi } ( d z ) ,
$$

from the observation of a single trajectory $\left( Z _ { 1 } , \ldots , Z _ { n } \right)$ only. By this, we mean that the bounds should depend only on $\left( Z _ { 1 } , \ldots , Z _ { n } \right)$ , not on unknown parameters of the distribution of $Z .$ As our applications will

be mostly in machine learning, we will from work with loss function $f ( z ) = \ell _ { \theta } ( z )$ , parametrized by $\theta \in \Theta$ with $0 \leq f \theta ( z ) \leq 1$ . In this case, $\bar { \pi } ( f )$ is usually written as a risk function

$$
R ( \theta ) : = \int \ell _ { \theta } ( z ) \bar { \pi } ( d z ) , \qquad \theta \in \Theta
$$

in the machine learning literature. Until Section 5, the parameter θ will be fixed and thus play little role. However, from Section 5, we will discuss algorithms aiming at minimizing the risk $R ( \theta )$ . While classical approaches rely on the empirical risk

$$
\hat { R } ( \theta ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell _ { \theta } ( Z _ { i } ) ,
$$

the regenerative structure suggests a different appraoch. For $j \in \mathbb N$ and $\theta \in \Theta$ , define the block loss and centered discrepancy by

$$
L _ { j , \theta } : = \sum _ { t = \tau _ { j } + 1 } ^ { \tau _ { j + 1 } } \ell _ { \theta } ( Z _ { t } ) , \qquad \xi _ { j , \theta } : = \Delta _ { j } R ( \theta ) - L _ { j , \theta } .
$$

The regenerative form of $\mathrm { K a c } ^ { \prime } \mathbf { s }$ lemma [35, Theorem 10.4.9] gives the risk identity below. This equality centers the block discrepancies and underlies our bounds (for example, it enables the exponential supermartingale construction in Lemma 8.1):

$$
R ( \theta ) = \frac { \mathbb { E } _ { A } [ L _ { 1 , \theta } ] } { \mathbb { E } _ { A } [ \Delta _ { 1 } ] } , \qquad \mathbb { E } _ { A } [ \xi _ { j , \theta } ] = 0 , \qquad | \xi _ { j , \theta } | \le \Delta _ { j } .
$$

Given the sample size $n \in \mathbb { N }$ , let

$$
M _ { n } : = \operatorname* { m i n } \{ m \in \mathbb { N } : \tau _ { m + 1 } > n \} .
$$

This is the index of the first cycle whose end is after time n. The blocks $B _ { 1 } , \ldots , B _ { M _ { n } - 1 }$ are complete. If $\tau _ { 1 } \leq n ,$ then we also have

$$
M _ { n } = \operatorname* { m a x } \{ m \in \mathbb { N } : \tau _ { m } \leq n \} .
$$

Let $\mathcal { F } _ { 0 } : = \sigma ( \tau _ { 1 } )$ and $\mathcal { F } _ { m } : = \sigma ( \tau _ { 1 } , B _ { 1 } , \ldots , B _ { m } )$ for $m \in \mathbb { N }$ . Define

$$
D _ { n } : = \tau _ { M _ { n } } - \tau _ { 1 } = \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } .
$$

On $\{ D _ { n } > 0 \}$ , the completed-block empirical risk is

$$
\widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) : = \frac { 1 } { D _ { n } } \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \theta } .
$$

We set it to zero when $D _ { n } \ = \ 0$ , except where we explicitly specify a different convention. The length $n - D _ { n }$ is the part of the sample outside the completed cycles.

As mentioned above, a difficulty here is that a bounded loss can have an unbounded block sum because cycle lengths can be unbounded. Various techniques were introduce to prove PAC and PAC-Bayes bounds for unbounded losses. One of the simplest idea is truncation, as used by Alquier [3], and also sufficient in our context. We compare the average of the $\xi _ { j , \theta }$ to an average of truncated variables $Y _ { j , \theta } : = \xi _ { j , \theta } \mathbf { 1 } _ { \{ \Delta _ { j } \leq u _ { n } \} }$ for some finite deterministic truncation level $u _ { n } \in ( 0 , \infty )$ . Put, for any $u > 0$

$$
\beta ( u ) : = \mathbb { E } _ { A } [ \Delta _ { 1 } \mathbf { 1 } _ { \{ \Delta _ { 1 } > u \} } ] .
$$

Finally, put

$$
q ( x ) : = - \log ( 1 - x ) - x , \qquad x \in [ 0 , 1 ) .
$$

It will be convenient to extend $q ( x ) = 0 \mathrm { f o r } x > 1$

Table 1: Notation used throughout the paper.
<table><tr><td>Notation</td><td>Definition</td><td>Notation</td><td>Definition</td></tr><tr><td> $Z _ { t }$ </td><td>Observation at time t.</td><td> $\mathcal { G } _ { t }$ </td><td>Filtration including observations and regeneration marks.</td></tr><tr><td> $\tau _ { j }$ </td><td>The j-th regeneration time.</td><td> $\mathbb { P } _ { A } , \mathbb { E } _ { A }$ </td><td>Common post-regeneration law and expectation.</td></tr><tr><td> $\Delta _ { j }$ </td><td>Block length  $\tau _ { j + 1 } - \tau _ { j }$ </td><td> $B _ { j }$ </td><td>Block  $( Z _ { \tau _ { j } + 1 } , \dotsc , Z _ { \tau _ { j + 1 } } )$ </td></tr><tr><td> $\Theta , { \mathcal { P } } ( \Theta )$ </td><td>Parameter space and its probability measures.</td><td> $\mu , \rho$ </td><td>Prior and generic posterior on Θ.</td></tr><tr><td> $\ell _ { \theta }$ </td><td>Loss associated with  $\theta ,$  taking values in [0, 1].</td><td></td><td>Equilibrium occupation measure.</td></tr><tr><td> $R ( \theta )$ </td><td>Long-run risk  $\textstyle \int \ell _ { \theta } ( z ) { \bar { \pi } } ( d z )$ </td><td> $L _ { j , \theta }$ </td><td>Cumulative loss in block  $j .$ </td></tr><tr><td> $\xi _ { j , \theta }$ </td><td>Centered discrepancy  $\Delta _ { j } R ( \theta ) - L _ { j , \theta } .$ </td><td>n</td><td>Length of the observed sample path.</td></tr><tr><td> $M _ { n }$ </td><td>Index of the first cycle ending after n.</td><td> $\mathcal { F } _ { m }$ </td><td>Block filtration  $\sigma ( \tau _ { 1 } , B _ { 1 } , \dots , B _ { m } )$ </td></tr><tr><td> $D _ { n }$ </td><td>Total length of the completed blocks.</td><td> ${ \widehat { R } } _ { n } ^ { \mathrm { r e g } } ( \theta )$ </td><td>Empirical risk over the completed blocks.</td></tr><tr><td> $u _ { n }$ </td><td>Deterministic truncation level.</td><td> $\beta ( u )$ </td><td>Tail contribution  $\mathbb { E } _ { A } \big [ \Delta _ { 1 } \mathbf { 1 } _ { \{ \Delta _ { 1 } > u \} } \big ] .$ </td></tr><tr><td> $Y _ { j , \theta }$ </td><td>Truncated discrepancy  $\xi _ { j , \theta } \mathbf { 1 } _ { \{ \Delta _ { j } \leq u _ { n } \} }$ </td><td> $q ( x )$ </td><td> $- \log ( 1 - x ) - x ,$  for  $x \in [ 0 , 1 )$ </td></tr></table>

## 3 Self-normalized Bernstein inequalities for regenerative processes

We begin with the statement of our self-normalized Bernstein inequality.

Theorem 3.1. Fix $\theta \in \Theta$ , deterministic values $u _ { n } > 0$ and $0 < \lambda < n / u _ { n }$ , and $\delta \in ( 0 , 1 )$ . With probability at least $1 - \delta ,$ on $\{ D _ { n } > 0 \}$ ,

$$
\begin{array} { l } { \displaystyle \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \leq \frac { n } { \lambda D _ { n } } \Biggl [ \log \frac { 2 } { \delta } + M _ { n } \log \left( 1 + \frac { \lambda \beta ( u _ { n } ) } { n } \right) } \\ { \displaystyle ~ + \sum _ { j = 1 } ^ { M _ { n } - 1 } \mathbf { 1 } _ { \{ \Delta _ { j } \leq u _ { n } \} ^ { q } } \left( \frac { \lambda \Delta _ { j } } { n } \right) - \log \left( 1 - \frac { \lambda u _ { n } } { n } \right) \Biggr ] } \\ { \displaystyle ~ + \frac { 1 } { D _ { n } } \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } \mathbf { 1 } _ { \{ \Delta _ { j } > u _ { n } \} } . } \end{array}\tag{3.1}
$$

By convention, we will consider that the bound (and all its corollaries below) are defined as $+ \infty$ when $D _ { n } = 0$

Remark 3.1. The only nonempirical term in (3.1) is

$$
\beta ( u _ { n } ) = \mathbb { E } _ { A } \left[ \Delta _ { 1 } \mathbf { 1 } _ { \{ \Delta _ { 1 } > u _ { n } \} } \right] .
$$

Under exponential tails, the choices of $u _ { n }$ and λ in Section 4 make its contribution to the expected deviation bound $\bar { O ( ( \log n ) / n ) } = o ( n ^ { - 1 / 2 } )$ , smaller than the leading $n ^ { - 1 / 2 }$ term. This need not hold under polynomial tails with exponent $1 < \alpha < 2$ . Moreover, it doesn’t depend on θ. Thus, in practical machine learning applications where we will use this inequality to define learning objectives, it is actually safe to ignore this term.

Theorem 3.1 uses a deterministic tuning parameter λ. We note that such a parameter is common in PAC-Bayes bounds, where it is actually useful to keep this parameter free [4]. However, we show here that we can optimize this parameter to get our first-order empirical version of Bernstein inequality. Define

$$
J _ { n } : = 2 \left( 1 + \lceil \log _ { 2 } \operatorname* { m a x } \{ 1 , n / u _ { n } \} \right] ) .
$$

Corollary 3.1. Fix $\theta \in \Theta$ , a deterministic $u _ { n } > 0 ,$ , and $\delta \in ( 0 , 1 )$ . Put

$$
H _ { n , u } : = \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } ^ { 2 } \mathbf { 1 } _ { \left\{ \Delta _ { j } \leq u _ { n } \right\} } , \qquad T _ { n , u } : = \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } \mathbf { 1 } _ { \left\{ \Delta _ { j } > u _ { n } \right\} } ,
$$

Then, with probability at least $1 - \delta ,$ , on $\{ D _ { n } > 0 \}$

$$
\Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | \leq \frac { 3 \sqrt { H _ { n , u } \log ( J _ { n } / \delta ) } + 2 u _ { n } [ \log ( J _ { n } / \delta ) + 1 ] } { D _ { n } } + \frac { M _ { n } \beta ( u _ { n } ) + T _ { n , u } } { D _ { n } } .\tag{3.2}
$$

We next integrate the deviation inequality. The resulting bound separates the truncated second moment, the tail contribution, and the expected length outside the completed cycles. This allows to understand the order of magnitude of the empirical deviations above.

Theorem 3.2. Fix $n \geq 1 , \theta \in \Theta$ , and deterministic values $u _ { n } > 0$ and $0 < \lambda \leq n / ( 2 u _ { n } )$ . Set $\widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) = 0$ when $D _ { n } = 0$ , and write, for any $u > 0$

$$
v ( u ) : = \mathbb { E } _ { A } \left[ \Delta _ { 1 } ^ { 2 } \mathbf { 1 } _ { \{ \Delta _ { 1 } \leq u \} } \right] .
$$

Then

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | \right] \leq \displaystyle \frac { 1 + \log 2 } { \lambda } + \frac { \lambda \mathbb { E } [ M _ { n } ] } { n ^ { 2 } } v ( u _ { n } ) } \\ & { \qquad + \displaystyle \frac { 2 \mathbb { E } [ M _ { n } ] } { n } \beta ( u _ { n } ) + \frac { 2 u _ { n } } { n } + \frac { n - \mathbb { E } [ D _ { n } ] } { n } . } \end{array}\tag{3.3}
$$

In the next section, we will make all the terms in $( 3 . 3 ) \longrightarrow [ \mathbb { E } [ D _ { n } ] , \mathbb { E } [ M _ { n } ] , \beta ( u _ { n } )$ and $v ( u _ { n } )$ —explicit under various assumptions on the regeneration. In particular, the ratio $r _ { n } = { \frac { \mathbb { E } [ M _ { n } ] } { n } } $ is a key quantity. As a proof of concept, let us just explicit these when the data process is made of blocks of length at most $k ,$ independent of each other. That is, $1 \leq \Delta _ { j } = \tau _ { j + 1 } - \tau _ { j } \leq k$ . Also suppose $\tau _ { 1 } = 1$ . This includes i.i.d. data for $k = 1$ . We always have $M _ { n } \leq n + 1 .$ , so $r _ { n } \leq ( n + 1 ) / n \leq 2$ , moreover, in this case, $D _ { n } \leq 2 k$ $v ( u _ { n } ) \leq k ^ { 2 }$ and thus (3.3) gives

$$
\mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq \frac { 1 + \log 2 } { \lambda } + \frac { 2 \lambda k ^ { 2 } } { n } + 4 \beta ( u _ { n } ) + \frac { 2 u _ { n } } { n } + \frac { 2 k } { n } .
$$

We can simply put $u _ { n } = k$ to get $\beta ( u _ { n } ) = 0$ , and an exact minimization in λ gives $\lambda = ( 1 / k ) \sqrt { n / [ 2 ( 1 + \log 2 ) ] }$ which satisfies $0 \leq \lambda \leq n / ( 2 u _ { n } ) = n / ( 2 k )$ for n large enough, in this case:

$$
\mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq 2 \sqrt { \frac { 2 k ( 1 + \log 2 ) } { n } } + \frac { 4 k } { n } .
$$

More generally, in situations where $\Delta _ { j }$ is unbounded but has light enough tails, we expect to recover $\mathbb { E } ( M _ { n } ) = { \mathcal { O } } ( n ) , \mathbb { E } ( D _ { n } ) = { \mathcal { O } } ( 1 )$ and, for $u _ { n } \to \infty$ well chosen, $\beta ( u _ { n } ) = o ( 1 / \sqrt { n } )$ and $u _ { n } / n = o ( 1 / { \sqrt { n } } )$

## 4 Control of the term $\beta ( u _ { n } )$ in various regimes

This section shows how assumptions on regeneration lengths yield explicit bounds and convergence rates for the expected estimation error. We first describe how Markov chains give rise to regenerative blocks, then specialize Theorem 3.2 to independent observations and to regeneration lengths with exponential or polynomial tails. From now, we keep the notation

$$
r _ { n } : = \mathbb { E } [ M _ { n } ] / n \leq ( n + 1 ) / n .
$$

## 4.1 Regeneration for Markov chains

A Markov chain regenerates at successive returns to an accessible atom [35, Chapter 5].

Definition 4.1. A ψ-irreducible, aperiodic Markov chain $X = ( X _ { m } ) _ { m \in  { \mathbb { N } } _ { 0 } }$ with transition kernel P is atomic if there is a measurable set A, called an atom, with $\psi ( A ) > 0$ , such that

$$
P ( x , \cdot ) = P ( y , \cdot ) , \qquad x , y \in A .
$$

Remark 4.1. In a countable irreducible state space, any singleton may serve as an atom.

Remark 4.2. For a chain without an accessible atom, Nummelin splitting constructs one from a small-set minorization. Suppose that an accessible set $S$ satisfies, for some $r \in  { \mathbb { N } } , \varepsilon _ { S } \in ( 0 , 1 )$ , and probability measure $\Gamma _ { S } ,$

$$
P ^ { r } ( x , C ) \geq \varepsilon _ { S } \Gamma _ { S } ( C ) , \qquad x \in S ,
$$

for every measurable C. For $r \ = \ 1$ , introduce independent Bernoulli variables $( Y _ { m } ) _ { m \geq 1 }$ with success probability $\varepsilon _ { S } .$ . At a visit $X _ { m } \in S$ , draw $X _ { m + 1 }$ from Γ<sub>S</sub> $i f Y _ { m } = 1$ , and otherwisefrom the residual kernel

$$
\frac { P ( X _ { m } , \cdot ) - \varepsilon _ { S } \Gamma _ { S } ( \cdot ) } { 1 - \varepsilon _ { S } } .
$$

This preserves the transition kernel of X, and the split chain $Z _ { m } = ( X _ { m } , Y _ { m } )$ has atom $A = S \times \{ 1 \}$ . For $r > 1$ , the construction applies to the r-step skeleton; applying the bounds to all intermediate observations requires a separately verified i.i.d. cycle construction [11, Section 1.2.3]. The problem is that, we need to know S and $\varepsilon _ { S }$ to use this in practice. We refer the reader to Bertail and Clemenc¸on [´ 11, p. 13] who propose a procedure to estimate such a set S.

Define the successive visits to the atom by

$$
\rho _ { A } ( 1 ) : = \operatorname* { i n f } \{ m \geq 1 : Z _ { m } \in A \} , \mathrm { a n d , f o r } j \geq 2 , \rho _ { A } ( j ) : = \operatorname* { i n f } \{ m > \rho _ { A } ( j - 1 ) : Z _ { m } \in A \} .
$$

Assuming these times are almost surely finite, the strong Markov property gives i.i.d. blocks

$$
B _ { j } = ( X _ { \rho _ { A } ( j ) + 1 } , \ldots , X _ { \rho _ { A } ( j + 1 ) } ) , \qquad j \geq 1 ,
$$

with i.i.d. lengths $\rho _ { A } ( j + 1 ) - \rho _ { A } ( j )$ . We require their common mean to be finite, as in Section 2.

## 4.2 Expected deviations under regeneration-length assumptions

We now make the tail term $\beta ( u _ { n } )$ explicit under exponential and polynomial assumptions on the regeneration lengths, and we specify a truncation level. While this is essentially all we need for the applications in machine learning in Section 5, we also apply Theorem 3.2, and minimize the resulting bound over the admissible values of λ to make the order of magnitude of the bounds explicit. The proofs use only the marked-cycle assumptions, so the same conclusions apply to the non-Markovian example in the next subsection.

Throughout this subsection, fix $\theta \in \Theta$ , assume $\tau _ { 1 } = 0$ (which always holds if we are ready to throw away an initial segment of the observed trajectory). We study exponential regeneration tails in a first time, and polynomial ones in a second time.

Corollary 4.1. Suppose that, for some $C _ { \mathrm { e } } , \kappa > 0$

$$
\begin{array} { r } { \mathbb { P } _ { A } ( \Delta _ { 1 } > t ) \le C _ { \mathrm { e } } e ^ { - \kappa t } , \qquad t \ge 1 . } \end{array}
$$

For every integer $u \geq 1$

$$
\beta ( u ) \leq C _ { \mathrm { e } } e ^ { - \kappa u } \left( u + \frac { 1 } { 1 - e ^ { - \kappa } } \right) .\tag{4.1}
$$

Moreover, the second moment and the expected renewal age satisfy

$$
v _ { 2 } : = \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] \leq 1 + \frac { C _ { \mathrm { e } } e ^ { - \kappa } ( 3 - e ^ { - \kappa } ) } { ( 1 - e ^ { - \kappa } ) ^ { 2 } } , \qquad \mathbb { E } [ n - D _ { n } ] \leq \frac { C _ { \mathrm { e } } e ^ { - \kappa } } { ( 1 - e ^ { - \kappa } ) ^ { 2 } } .
$$

For $n \geq 1$ , choose

$$
u _ { n } : = \operatorname* { m a x } \left\{ 1 , \left\lceil { \frac { \log n } { \kappa } } \right\rceil \right\} .
$$

Then

$$
\beta ( u _ { n } ) \le \frac { C _ { \mathrm { e } } } { n } \left( u _ { n } + \frac { 1 } { 1 - e ^ { - \kappa } } \right) \le \frac { C _ { \mathrm { e } } } { n } \left( \frac { \log n } { \kappa } + 1 + \frac { 1 } { 1 - e ^ { - \kappa } } \right) .\tag{4.2}
$$

For every deterministic $0 < \lambda \leq n / ( 2 u _ { n } )$

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq \frac { 1 + \log 2 } { \lambda } + \frac { r _ { n } \lambda v _ { 2 } } { n } } \\ { \displaystyle \qquad + 2 r _ { n } C _ { \mathrm { e } } e ^ { - \kappa u _ { n } } \left( u _ { n } + \frac { 1 } { 1 - e ^ { - \kappa } } \right) } \\ { \displaystyle \qquad + \frac { 2 u _ { n } } { n } + \frac { C _ { \mathrm { e } } e ^ { - \kappa } } { n ( 1 - e ^ { - \kappa } ) ^ { 2 } } . } \end{array}\tag{4.3}
$$

The right-hand side is minimized over this interval by

$$
\lambda _ { n } : = \operatorname* { m i n } \left\{ \sqrt { \frac { ( 1 + \log 2 ) n } { r _ { n } v _ { 2 } } } , \frac { n } { 2 u _ { n } } \right\} .\tag{4.4}
$$

This choice yields the explicit bound

$$
\begin{array} { r l r } & { } & { \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq 2 \sqrt { \frac { ( 1 + \log 2 ) r _ { n } v _ { 2 } } { n } } + \frac { 2 ( 2 + \log 2 ) u _ { n } } { n } } \\ & { } & { \quad \quad \quad + \frac { 2 r _ { n } C _ { \mathrm { e } } } { n } \left( u _ { n } + \frac { 1 } { 1 - e ^ { - \kappa } } \right) + \frac { C _ { \mathrm { e } } e ^ { - \kappa } } { n ( 1 - e ^ { - \kappa } ) ^ { 2 } } . } \end{array}\tag{4.5}
$$

For a fixed regenerative law, $\lambda _ { n } \sim \sqrt { ( 1 + \log { 2 } ) m n / v _ { 2 } } ,$ , and

$$
\mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq \left( 2 \sqrt { \frac { ( 1 + \log 2 ) v _ { 2 } } { m } } + o ( 1 ) \right) n ^ { - 1 / 2 } + O \left( \frac { \log n } { n } \right) .\tag{4.6}
$$

Here $\beta ( u _ { n } ) = O ( ( \log n ) / n ) = o ( n ^ { - 1 / 2 } )$ . Thus the contribution $2 r _ { n } \beta ( u _ { n } )$ , the truncation correction $2 u _ { n } / n$ , and the expected renewal age divided by n are all second order relative to the $n ^ { - 1 / 2 }$ deviation rate.

Under polynomial tails, the truncated second moment and the role of the tail contribution depend on whether the tail exponent is above, equal to, or below two.

Corollary 4.2. Suppose that, for some $C _ { \mathrm { p } } > 0$ and $\alpha > 1$

$$
\begin{array} { r } { \mathbb { P } _ { A } ( \Delta _ { 1 } > t ) \le C _ { \mathrm { p } } t ^ { - \alpha } , \qquad t \ge 1 . } \end{array}
$$

Put $B _ { \alpha } : = \alpha C _ { \mathrm { p } } / ( \alpha - 1 )$ . For every $u \geq 1$

$$
\beta ( u ) \leq B _ { \alpha } u ^ { 1 - \alpha } .\tag{4.7}
$$

For $n \geq 1$ , choose $u _ { n } : = \lceil n ^ { 1 / \alpha } \rceil$ . Then

$$
\beta ( u _ { n } ) \leq B _ { \alpha } u _ { n } ^ { 1 - \alpha } \leq B _ { \alpha } n ^ { - ( \alpha - 1 ) / \alpha } .\tag{4.8}
$$

For $u \geq 1$ , define

$$
V _ { \alpha } ( u ) : = \left\{ \begin{array} { l l } { \displaystyle 1 + \frac { 2 C _ { \mathrm { p } } } { \alpha - 2 } , } & { \displaystyle \alpha > 2 , } \\ { \displaystyle 1 + 2 C _ { \mathrm { p } } \log u , } & { \displaystyle \alpha = 2 , } \\ { \displaystyle \left( 1 + \frac { 2 C _ { \mathrm { p } } } { 2 - \alpha } \right) u ^ { 2 - \alpha } , } & { \displaystyle 1 < \alpha < 2 , } \end{array} \right.
$$

and put

$$
G _ { \alpha } ( n ) : = \left\{ \begin{array} { l l } { \displaystyle C _ { \mathrm { p } } \left( 1 + \frac { 1 } { \alpha - 2 } \right) , } & { \displaystyle \alpha > 2 , } \\ { \displaystyle C _ { \mathrm { p } } ( 1 + \log n ) , } & { \displaystyle \alpha = 2 , } \\ { \displaystyle C _ { \mathrm { p } } \left( 1 + \frac { n ^ { 2 - \alpha } } { 2 - \alpha } \right) , } & { \displaystyle 1 < \alpha < 2 . } \end{array} \right.
$$

These quantities satisfy $v ( u ) \leq V _ { \alpha } ( u )$ and $\mathbb { E } [ n - D _ { n } ] \leq G _ { \alpha } ( n )$ . For every deterministic $0 < \lambda \leq n / ( 2 u _ { n } )$

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | \right] \leq \frac { 1 + \log 2 } { \lambda } + \frac { r _ { n } \lambda } { n } V _ { \alpha } ( u _ { n } ) } \\ & { \qquad + \ : 2 r _ { n } B _ { \alpha } u _ { n } ^ { 1 - \alpha } + \frac { 2 u _ { n } + G _ { \alpha } ( n ) } { n } . } \end{array}\tag{4.9}
$$

The right-hand side is minimized over this interval by

$$
\lambda _ { n } : = \operatorname* { m i n } \left\{ \sqrt { \frac { ( 1 + \log 2 ) n } { r _ { n } V _ { \alpha } ( u _ { n } ) } } , \frac { n } { 2 u _ { n } } \right\} .\tag{4.10}
$$

This choice yields

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq 2 \sqrt { \frac { ( 1 + \log 2 ) r _ { n } V _ { \alpha } ( u _ { n } ) } { n } } + \frac { 2 ( 2 + \log 2 ) u _ { n } } { n } } \\ & { \qquad + 2 r _ { n } B _ { \alpha } u _ { n } ^ { 1 - \alpha } + \frac { G _ { \alpha } ( n ) } { n } . } \end{array}\tag{4.11}
$$

Keeping the regenerative lawfixed as $n \to \infty$ , we obtain

$$
\mathbb { E } \left[ \Big \vert R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big \vert \right] = O ( n ^ { - 1 / 2 } ) ,
$$

$$
\alpha > 2 ,\tag{4.12}
$$

$$
\mathbb { E } \left[ \Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | \right] = O \left( \sqrt { \frac { \log n } { n } } \right) ,
$$

$$
\alpha = 2 ,\tag{4.13}
$$

$$
\begin{array} { r } { \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] = O \left( n ^ { - ( \alpha - 1 ) / \alpha } \right) , } \end{array}
$$

$$
1 < \alpha < 2 .\tag{4.14}
$$

The constants in these order terms may depend on thefixed regenerative law, $C _ { \mathrm { p } } ,$ , and $\alpha .$

Remark 4.3. For $\alpha > 2 , ( 4 . 8 )$ gives $\beta ( u _ { n } ) = o ( n ^ { - 1 / 2 } )$ , so the tail contribution is second order. At $\alpha = 2 ,$ , it gives $\beta ( u _ { n } ) = O ( n ^ { - 1 / 2 } )$ , which is still smaller than the general rate $\sqrt { ( \log n ) / n }$ , but only by a logarithmic factor.

For $1 < \alpha < 2$ , the available bound on $\beta ( u _ { n } )$ has the same order $\cdot n ^ { - ( \alpha - 1 ) / \alpha }$ as the optimized quadratic and truncation terms. The tail contribution is therefore no longer necessarily second order. In particular, if $\mathbb { P } _ { A } ( \Delta _ { 1 } > t ) \sim c t ^ { - \alpha }$ for some $c > 0 ,$ , then

$$
\beta ( u _ { n } ) \sim \frac { \alpha c } { \alpha - 1 } n ^ { - ( \alpha - 1 ) / \alpha } .
$$

Since $r _ { n }  1 / m ,$ , its contribution $2 r _ { n } \beta ( u _ { n } )$ is then of leading order in (4.11). Consequently, it cannot in general be omitted when assessing the leading magnitude of the bound. The polynomial upper-tail assumption alone does not force this behaviour, since it also allows more rapidly decaying tails.

Remark 4.4. The tail assumption with $\alpha = 2$ does not imply $v _ { 2 } : = \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] < \infty .$ . If this second moment is finite, $V _ { 2 } ( u _ { n } )$ may be replaced by $v _ { 2 }$ in (4.9), (4.10), and (4.11). With the same choice $u _ { n } = \lceil \sqrt { n } \rceil$ , this improves the rate to $O ( n ^ { - 1 / 2 } )$ . In this case the actual tail term satisfies $\beta ( u _ { n } ) = o ( n ^ { - 1 / 2 } )$ ; the coarser upper bound in (4.8) need not reflect this improvement. The correction $u _ { n } / n ,$ however, remains of order $n ^ { - 1 / 2 }$

## 4.3 A non-Markovian example

We next apply the same calculation to a queue whose length process is not Markovian. Returns to the empty state give independent cycles. After verifying exponential tails for their lengths, we use Corollary 4.1 to obtain explicit tail and expected deviation bounds.

Example: a Geo/D/1 queue. Let $( I _ { t } ) _ { t \in  { \mathbb { N } } _ { 0 } }$ be independent $\operatorname { B e r n o u l l i } ( p )$ arrival indicators, where $p \in$ $( 0 , 1 / 2 )$ . Start the queue empty, with $Q _ { 0 } = 0$ , and let $Q _ { t } \in \mathbb { N } _ { 0 }$ be the number of customers immediately before the arrival in slot t. An arriving customer joins the queue at the beginning of the slot. The server then provides one slot of service, in order of arrival and without preemption. Each customer requires two service slots, and departures occur at the end of a slot. A customer arriving to an empty system starts service in that slot; after a departure, the next waiting customer starts service in the following slot.

Define the successive empty observations by

$$
\tau _ { 1 } : = 0 , \qquad \tau _ { j + 1 } : = \operatorname* { i n f } \{ t > \tau _ { j } : Q _ { t } = 0 \} , \qquad j \in \mathbb { N } .
$$

An empty slot with no arrival gives a cycle of length one; an arrival to the empty system starts a busy-period cycle. We use the same block convention as in Section 2:

$$
\Delta _ { j } : = \tau _ { j + 1 } - \tau _ { j } , \qquad B _ { j } : = ( Q _ { \tau _ { j } + 1 } , \dots , Q _ { \tau _ { j + 1 } } ) .
$$

Corollary 4.3. The queue-length process $( Q _ { t } ) _ { t \in  { \mathbb { N } } _ { 0 } }$ is not Markov with respect to its natural filtration, but the marked cycles $( \Delta _ { j } , B _ { j } ) _ { j \in \mathbb { N } }$ are i.i.d. We define $\begin{array} { r } { \kappa _ { 0 } : = - \frac { 1 } { 2 } \log ( 4 p ( 1 - p ) ) , C _ { 0 } : = e ^ { 2 \kappa _ { 0 } } } \end{array}$ . Then $\kappa _ { 0 } > 0$ and

$$
\mathbb { P } ( \Delta _ { 1 } > t ) \le C _ { 0 } e ^ { - \kappa _ { 0 } t } , \qquad t \ge 0 .
$$

In particular, all regeneration times are almost surely finite, the cycle lengths have finite moments of every order, and

$$
m : = \mathbb { E } [ \Delta _ { 1 } ] = \frac { 1 - p } { 1 - 2 p } , \qquad v _ { 2 } : = \mathbb { E } [ \Delta _ { 1 } ^ { 2 } ] = \frac { ( 1 - p ) ( 1 - 2 p + 4 p ^ { 2 } ) } { ( 1 - 2 p ) ^ { 3 } } .\tag{4.15}
$$

For every integer u $\geq 1$ ,

$$
\beta ( u ) \leq C _ { 0 } e ^ { - \kappa _ { 0 } u } \left( u + \frac { 1 } { 1 - e ^ { - \kappa _ { 0 } } } \right) .\tag{4.16}
$$

For $n \geq 1$ , choose

$$
u _ { n } : = \operatorname* { m a x } \left\{ 1 , \left\lceil { \frac { \log n } { \kappa _ { 0 } } } \right\rceil \right\} .
$$

Then

$$
\beta ( u _ { n } ) \le \frac { C _ { 0 } } { n } \left( u _ { n } + \frac { 1 } { 1 - e ^ { - \kappa _ { 0 } } } \right) \le \frac { C _ { 0 } } { n } \left( \frac { \log n } { \kappa _ { 0 } } + 1 + \frac { 1 } { 1 - e ^ { - \kappa _ { 0 } } } \right) .\tag{4.17}
$$

Fix $\theta \in \Theta ,$ , take $Z _ { t } = Q _ { t } ) .$ For every deterministic $0 < \lambda \leq n / ( 2 u _ { n } )$

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq \frac { 1 + \log 2 } { \lambda } + \frac { r _ { n } \lambda v _ { 2 } } { n } } \\ { \displaystyle \qquad + 2 r _ { n } C _ { 0 } e ^ { - \kappa _ { 0 } u _ { n } } \left( u _ { n } + \frac { 1 } { 1 - e ^ { - \kappa _ { 0 } } } \right) } \\ { \displaystyle \qquad + \frac { 2 u _ { n } } { n } + \frac { C _ { 0 } e ^ { - \kappa _ { 0 } } } { n ( 1 - e ^ { - \kappa _ { 0 } } ) ^ { 2 } } . } \end{array}\tag{4.18}
$$

The right-hand side is minimized over this interval by

$$
\lambda _ { n } : = \operatorname* { m i n } \left\{ \sqrt { \frac { ( 1 + \log 2 ) n } { r _ { n } v _ { 2 } } } , \frac { n } { 2 u _ { n } } \right\} .\tag{4.19}
$$

This choice yields

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq 2 \sqrt { \frac { ( 1 + \log 2 ) r _ { n } v _ { 2 } } { n } } + \frac { 2 ( 2 + \log 2 ) u _ { n } } { n } } \\ & { \quad \quad \quad \quad + \frac { 2 r _ { n } C _ { 0 } } { n } \left( u _ { n } + \frac { 1 } { 1 - e ^ { - \kappa _ { 0 } } } \right) + \frac { C _ { 0 } e ^ { - \kappa _ { 0 } } } { n ( 1 - e ^ { - \kappa _ { 0 } } ) ^ { 2 } } . } \end{array}\tag{4.20}
$$

For fixed $p , r _ { n } \to ( 1 - 2 p ) / ( 1 - p )$ , and hence

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq \left( \frac { 2 \sqrt { ( 1 + \log 2 ) ( 1 - 2 p + 4 p ^ { 2 } ) } } { 1 - 2 p } + o ( 1 ) \right) n ^ { - 1 / 2 } } \\ & { \qquad + O \left( \frac { \log n } { n } \right) . } \end{array}\tag{4.21}
$$

The constants in the order terms depend only on $p .$

Thus this non-Markovian process has an expected deviation bound of order $n ^ { - 1 / 2 }$ , with the explicit tail contribution $\beta ( u _ { n } ) = O ( ( \log n ) / n )$ again of second order. Since $v _ { 2 }$ is explicit in $p ,$ replacing $r _ { n }$ by $( n + 1 ) / n$ in the finite-sample bound and its minimizing choice of $\lambda$ also gives a version involving only $p$ and n.

## 5 Applications to classification with PAC-Bayes bounds

The preceding bounds concern a fixed parameter. We now consider the situation where $\ell _ { \theta } ( z )$ measures the classification error of a parameter θ on a data z. In this case $R ( \theta )$ is the generalization risk. The objective is to build a classifier whose classification error is as close as possible to in $\mathbf { f } _ { \theta \in \Theta } R ( \theta )$ . PAC-Bayes bounds consider randomized classification procedure, that is, sample-dependent probability distributions over Θ. This requires a measurable space structure. $\mathrm { { S o . } }$ from now, $( \Theta , \tau )$ is a measurable parameter space and let ${ \mathcal { P } } ( \Theta )$ will denote the set of its probability measures. Assume that

$$
( \theta , z ) \longmapsto \ell _ { \theta } ( z ) , \qquad \Theta \times { \cal Z } \longrightarrow [ 0 , 1 ] ,
$$

is jointly measurable. For the learning results, fix a prior $\mu \in { \mathcal P } ( \Theta )$ independently of the observations. We use $\rho \in { \mathcal { P } } ( \Theta )$ for a posterior distribution; a posterior chosen from the data is a measurable probability kernel. Write $\operatorname { K L } ( \rho \| \mu )$ for the relative entropy or Kullback-Leibler divergence:

$$
{ \mathrm { K L } } ( \rho | | \mu ) : = \int _ { \Theta } \log \left( { \frac { d \rho } { d \mu } } \right) d \rho \quad { \mathrm { i f ~ } } \rho \ll \mu ,
$$

and set it to +∞ otherwise.

## 5.1 PAC-Bayes bounds

Theorem 5.1. For every deterministic

$$
0 < \lambda < { \frac { n } { u _ { n } } }
$$

and every $\delta \in ( 0 , 1 )$ , with probbility at least $1 - \delta$ over the observed sample $( Z _ { 1 } , \ldots , Z _ { n } )$ , simultaneously for every $\rho \in \mathcal { P } ( \Theta )$

$$
\begin{array} { r l r } {  { \mathbb { E } _ { \theta \sim \rho } R ( \theta ) \le \mathbb { E } _ { \theta \sim \rho } \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) } } \\ & { } & { + \frac { n } { \lambda D _ { n } } \Bigg [ \mathrm { K L } ( \rho \| \mu ) + \log ( \frac { 1 } { \delta } ) + M _ { n } \log ( 1 + \frac { \lambda \beta ( u _ { n } ) } { n } ) } \\ & { } & { + \displaystyle \sum _ { j = 1 } ^ { M _ { n } - 1 } \mathbf { 1 } _ { \{ \Delta _ { j } \le u _ { n } \} } q ( \frac { \lambda \Delta _ { j } } { n } ) - \log ( 1 - \frac { \lambda u _ { n } } { n } ) \Bigg ] + \frac { 1 } { D _ { n } } \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } \mathbf { 1 } _ { \{ \Delta _ { j } > u _ { n } \} } . } \end{array}\tag{5.1}
$$

The right-hand side can be used as a learning objective. The minimization of PAC-Bayes bounds with respect to $\rho$ leads to randomized classifiers $\hat { \rho } _ { \lambda }$ that satisfy oracle inequalities [4, 15]. In practice, these classifiers also have very good performance, we refer the reader to [17] for application to image classification with deep neural networks.

The important point is that the minimization in $\rho$ does not involve any term depending on distribution of the tails of the $\Delta _ { j } \mathrm { ^ { * } s }$ . Actually, it involves only observable (empirical) quantities:

$$
\widehat { \rho } _ { \lambda } : = \arg \operatorname* { m i n } _ { \rho \in \mathcal { P } ( \Theta ) } \left\{ \mathbb { E } _ { \theta \sim \rho } \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) + \frac { n \mathrm { K L } ( \rho \| \mu ) } { \lambda D _ { n } } \right\} .
$$

In the previous attempts on empirical PAC-Bayes bounds for non i.i.d. data [25], the minimization in $\rho$ actually involved non-empirical terms, that were estimated.

From Section 2.1.2 page 20 of [4], we have an explicit form for $\hat { \rho } _ { \lambda }$ , given by its density in $\theta \in \Theta$ with respect to $\mu { : }$

$$
\frac { d \widehat { \rho } _ { \lambda } } { d \mu } ( \theta ) = \frac { \displaystyle \exp \left\{ - \frac { \lambda } { n } \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \theta } \right\} } { \displaystyle \int _ { \Theta } \exp \left\{ - \frac { \lambda } { n } \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \vartheta } \right\} \mu ( d \vartheta ) } .\tag{5.2}
$$

Note, however, that the randomized estimator depends on λ. Thus, we still have to tune $\lambda ,$ but this time without using any information on the data-generating process.

## 5.2 Oracle risk bound in expectation

In order to allow an adaptive choice for λ, we will propose an optimization over a finite grid $\Lambda _ { n }$ (the same strategy was used to derive Corollary 3.1 from Theorem 3.1). Thus, let $\Lambda _ { n } \subset ( 0 , n / u _ { n } )$ be non-empty, finite, with $K _ { n } = | \Lambda _ { n } |$ and $\lambda _ { + } : = \operatorname* { m a x } \Lambda _ { n }$

For each $\lambda \in \Lambda _ { n }$ , keep the Gibbs posterior $\widehat { \rho } _ { \lambda }$ from (5.2), and put

$$
\begin{array} { l } { \displaystyle \widehat { \mathcal { C } } _ { n } ( \lambda ) : = \mathbb { E } _ { \theta \sim \widehat { \rho } _ { \lambda } } \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \theta } } \\ { \displaystyle \qquad + \frac { n } { \lambda } \left\{ \mathrm { K L } ( \widehat { \rho } _ { \lambda } \| \mu ) + p _ { n } ( \lambda ) + Q _ { n } ( \lambda ) \right\} + T _ { n , u } } \end{array}
$$

where

$$
p _ { n } ( \lambda ) : = \log \left( \frac { K _ { n } \lambda _ { + } } { \lambda } \right) , Q _ { n } ( \lambda ) : = \sum _ { 1 \leq j < M _ { n } } q \left( \frac { \lambda \Delta _ { j } } { n } \right) - \log \left( 1 - \frac { \lambda u _ { n } } { n } \right) ,
$$

and $T _ { n , u }$ is defined in Corollary 3.1.

On $\{ D _ { n } > 0 \}$ , dividing this criterion by $D _ { n }$ gives the empirical part of the PAC-Bayes bound, with the grid penalty included and the term involving $\beta ( u _ { n } )$ omitted. The term $T _ { n , u }$ does not depend on $\lambda ,$ so it can also be omitted when minimizing. Choose

$$
\widehat { \lambda } \in \arg \operatorname* { m i n } _ { \lambda \in \Lambda _ { n } } \widehat { \mathcal { C } } _ { n } ( \lambda ) ,\tag{5.3}
$$

breaking ties by taking the smallest minimizer (this definition also applies when $D _ { n } = 0$ , in which case our convention $R _ { n } ^ { \mathrm { r e g } } ( \theta ) = 0$ implies ${ \hat { \rho } } _ { \lambda } \mu \ ]$ .

Theorem 5.2. For every finite deterministic grid $\Lambda _ { n } \subset ( 0 , n / u _ { n } )$ as above, $i f \operatorname { \mathbb { E } } [ D _ { n } ] > 0$ , the selected posterior satisfies

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathbb { E } _ { \theta \sim \hat { \rho } _ { \hat { \lambda } } } R ( \theta ) \right] } \\ & { \leq \underset { \lambda \in \Lambda _ { n } } { \operatorname* { i n f } } \left\{ \underset { \rho \ll \mu } { \operatorname* { i n f } } \left[ \mathbb { E } _ { \theta \sim \rho } R ( \theta ) + \frac { 2 n } { \lambda \mathbb { E } [ D _ { n } ] } \mathrm { K L } ( \rho \| \mu ) \right] + \frac { n \{ p _ { n } ( \lambda ) + 2 \mathbb { E } [ Q _ { n } ( \lambda ) ] \} } { \lambda \mathbb { E } [ D _ { n } ] } \right\} } \\ & { \quad + \frac { 2 \mathbb { E } [ M _ { n } ] \beta ( u _ { n } ) + 2 \mathbb { E } [ T _ { n , u } ] } { \mathbb { E } [ D _ { n } ] } + \frac { n } { \lambda + \mathbb { E } [ D _ { n } ] } + \frac { n - \mathbb { E } [ D _ { n } ] } { n } . } \end{array}\tag{5.4}
$$

The penalty $p _ { n } ( \lambda )$ comes from a union bound over the grid; its form bounds the additional expectation remainder by $n / ( \lambda _ { + } \mathbb { E } [ D _ { n } ] )$ . The inequality

$$
\frac { n } { \lambda } M _ { n } \log \left( 1 + \frac { \lambda \beta ( u _ { n } ) } { n } \right) \leq M _ { n } \beta ( u _ { n } )
$$

provides a common upper bound for the omitted term, independently of λ. Thus the selection rule requires no information about the regeneration tails.

A classical assumption in Bayesian statistics and PAC-Bayes bounds is the prior mass condition. Write

$$
R _ { * } : = \operatorname* { i n f } _ { \theta \in \Theta } R ( \theta ) , \qquad V ( \varepsilon ) : = \{ \theta \in \Theta : R ( \theta ) - R _ { * } \leq \varepsilon \} ,
$$

and assume that there are $c \in ( 0 , 1 )$ and $d \geq 1$ such that, for every $\varepsilon \in ( 0 , 1 )$ ,

$$
\mu ( V ( \varepsilon ) ) \geq c \varepsilon ^ { d } .\tag{5.5}
$$

Let $\mu _ { V ( \varepsilon ) }$ denote the prior conditioned on $V ( \varepsilon )$ . Then

$$
\begin{array} { r } { \mathbb { E } _ { \theta \sim \mu _ { V ( \varepsilon ) } } R ( \theta ) \leq R _ { * } + \varepsilon , \qquad \mathrm { K L } ( \mu _ { V ( \varepsilon ) } | | \mu ) = - \log \mu ( V ( \varepsilon ) ) \leq \log ( 1 / c ) + d \log ( 1 / \varepsilon ) . } \end{array}
$$

While we an optimize in ε, we can obtain an explicit rate with the simple choice $\varepsilon = 1 / n$ . For $n \geq 2$ , it yields

$$
\operatorname* { i n f } _ { \rho \ll \mu } \left\{ \mathbb { E } _ { \theta \sim \rho } R ( \theta ) + \frac { 2 n } { \lambda \mathbb { E } [ D _ { n } ] } \mathrm { K L } ( \rho \| \mu ) \right\} \leq R _ { * } + \frac { 1 } { n } + \frac { 2 n \{ d \log n + \log ( 1 / c ) \} } { \lambda \mathbb { E } [ D _ { n } ] } .\tag{5.6}
$$

These choices of comparison posteriors are only used in the analysis; neither c nor d is needed to compute $\widehat { \lambda } .$

We illustrate this result under exponential regeneration tails with unknown parameters.

Corollary 5.1. In the setting of Theorem 5.2, assume $\tau _ { 1 } = 0$ , the prior mass condition (5.5) with $0 < c \leq 1$ and d $\geq 1$ , and,for some unknown $C _ { \mathrm { e } } , \kappa > 0 ,$ , for any $t \geq 1$

$$
\mathbb { P } _ { A } ( \Delta _ { 1 } > t ) \le C _ { \mathrm { e } } e ^ { - \kappa t } , .
$$

For $n \geq 2 ,$ , choose

$$
u _ { n } : = \left\lceil ( \log n ) ^ { 2 } \right\rceil , \qquad K _ { n } : = 1 + \left\lceil \log _ { 2 } n \right\rceil , \qquad \Lambda _ { n } : = \left\{ \frac { n } { 2 ^ { j + 1 } u _ { n } } : j = 0 , \ldots , K _ { n } - 1 \right\} ,\tag{5.7}
$$

and select $\widehat { \lambda } b y$ (5.3). Thus $\lambda _ { + } = n / ( 2 u _ { n } )$ and $p _ { n } ( n / ( 2 ^ { j + 1 } u _ { n } ) ) = \log ( K _ { n } 2 ^ { j } ) . P u t$

$$
a _ { n } : = 2 d \log n + 2 \log ( 1 / c ) + \log ( 2 n K _ { n } ) , \qquad v _ { 2 } : = \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] .
$$

Then

$$
\begin{array} { r l r } & { } & { \mathbb { E } \left[ \mathbb { E } _ { \theta \sim \hat { \rho } _ { \hat { \lambda } } } R ( \theta ) \right] \le R _ { * } + \displaystyle \frac { 1 } { n } + \frac { n } { \mathbb { E } [ D _ { n } ] } \left[ 3 \sqrt { \frac { r _ { n } v _ { 2 } a _ { n } } { n } } + \frac { ( 2 a _ { n } + 6 ) u _ { n } } { n } \right. } \\ & { } & { \left. ~ + 4 r _ { n } C _ { \mathrm { e } } e ^ { - \kappa u _ { n } } \left( u _ { n } + \displaystyle \frac { 1 } { 1 - e ^ { - \kappa } } \right) \right] + \displaystyle \frac { C _ { \mathrm { e } } e ^ { - \kappa } } { n ( 1 - e ^ { - \kappa } ) ^ { 2 } } . } \end{array}\tag{5.8}
$$

For a fixed regenerative law and fixed $c , d ,$ this gives

$$
\mathbb { E } \left[ \mathbb { E } _ { \theta \sim \hat { \rho } _ { \hat { \lambda } } } R ( \theta ) \right] - R _ { * } = O \left( \sqrt { \frac { d \log n } { n } } \right) + O \left( \frac { d ( \log n ) ^ { 3 } } { n } \right) .\tag{5.9}
$$

The truncation level, the grid, and the selection criterion depend on neither c, d nor the unknown parameters ofthe regeneration tails.

Thus, we recover the same rate $O ( { \sqrt { \frac { d \log n } { n } } } )$ as in learning from i.i.d. data. The $2 ^ { j + 1 }$ in the construction of the grid was suggested by [15]. The universal choice $u _ { n } = \lceil ( \log n ) ^ { 2 } \rceil$ makes $\beta ( u _ { n } )$ smaller than every fixed power of $n ^ { - 1 }$ under exponential tails. The explicit terms after the square root in (5.8) are therefore second order for fixed $c , d .$ In particular, omitting $\beta ( u _ { n } )$ from the criterion does not affect the leading rate.

## 6 Minimax upper and lower bounds

In this section, we discuss the optimality of Theorem 3.1 and its corollaries. While the order $1 / \sqrt { n }$ is expected, we will show that Theorem 3.1 leads to an optimal dependence in the ratio $m / n$ , where $m : =$ $\mathbb { E } _ { A } [ \Delta _ { 1 } ]$ , under a moment assumption on the tail. Note that, in this optimality consideration, we are no longer concerned with empirical bounds: we will show that, even a statistician who would know m and the constants in the moment assumptions would not be able to improve the order of magnitude of the bounds derived from Theorem 3.1.

First, we derive an upper bound on the estimation error from Theorem 3.1 then we will compare it with a minimax lower bound. We state a proposition for a fixed tuning parameter; choosing that parameter gives the upper inequality in Theorem 6.1.

Throughout this section, we still assume that $\tau _ { 1 } = 0$ , and suppose that, for some $\varepsilon > 0$ and $C \geq 1$

$$
\begin{array} { r } { \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 + \varepsilon } ] \leq C m ^ { 2 + \varepsilon } . } \end{array}\tag{6.1}
$$

For the upper bound, choose the truncation level

$$
u _ { n } = m \left( { \frac { n } { m } } \right) ^ { 1 / ( 2 + \varepsilon ) } .
$$

The moment condition controls both the quadratic term and the discarded tail. It also ensures that completed cycles cover at least half of the observation horizon on the event used in the next proposition.

Proposition 6.1. Fix $\theta \in \Theta , \delta \in ( 0 , 1 ) , n \geq 4 8 C m / \delta ,$ and a deterministic $0 < \lambda \leq n / ( 2 u _ { n } )$ . There is an event ofprobability at least $1 - \delta$ on which $D _ { n } \geq n / 2$ and

$$
\left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \leq \frac { 2 } { \lambda } \log \frac { 8 } { \delta } + \frac { 2 4 C \lambda m } { \delta n } + \left( \frac { 4 8 C } { \delta } + 4 \right) \left( \frac { m } { n } \right) ^ { ( 1 + \varepsilon ) / ( 2 + \varepsilon ) } .\tag{6.2}
$$

Choosing λ balances the reciprocal and linear terms in (6.2). We compare the resulting order with what any estimator can achieve. Consider the scalar model with $\Theta = \{ \theta _ { 0 } \} , X _ { t } = 0 , Y _ { t } \in [ 0 , 1 ]$ , and $\ell _ { \theta _ { 0 } } ( Z _ { t } ) = Y _ { t }$ Let $\mathfrak { P } ( m , C , \varepsilon )$ denote the class of laws of i.i.d. regenerative blocks with positive integer lengths of mean m satisfying (6.1), with observations starting at a regeneration time. Define the long-run risk under P by

$$
R _ { P } ( \theta _ { 0 } ) : = \frac { \mathbb { E } _ { A , P } \Big [ \sum _ { t = 1 } ^ { \Delta _ { 1 } } Y _ { t } \Big ] } { m } .
$$

For $\delta \in ( 0 , 1 )$ , define the minimax estimation error by

$$
r _ { n , \delta } ( m , C , \varepsilon ) : = \operatorname* { i n f } _ { \tilde { R } _ { n } } \operatorname* { i n f } \Big \{ r \geq 0 : \operatorname* { s u p } _ { P \in \mathfrak { P } ( m , C , \varepsilon ) } \mathbb { P } _ { A , P } \big ( | \widetilde { R } _ { n } - R _ { P } ( \theta _ { 0 } ) | > r \big ) \leq \delta \Big \} ,
$$

where the first infimum is over real-valued measurable estimators based on the observations and observed regeneration boundaries. For the upper bound, use the completed-cycle empirical risk, setting it to $1 / 2$ when $D _ { n } = 0$ . Write

$$
C _ { C , \delta } : = 8 \sqrt { \frac { 3 C } { \delta } \log \frac { 8 } { \delta } } + 4 \log \frac { 8 } { \delta } + \frac { 4 8 C } { \delta } + 4 .
$$

Theorem 6.1. Assume that $m \in \mathbb { N } .$ . Then, for every $\delta \in ( 0 , 3 / 8 )$ and $n \geq 4 8 C m / \delta _ { \mathrm { { i } } }$

$$
\frac { 1 } { 1 6 \sqrt { 2 } } \sqrt { \frac { m } { n } } \leq r _ { n , \delta } ( m , C , \varepsilon ) \leq C _ { C , \delta } \sqrt { \frac { m } { n } } .\tag{6.3}
$$

Thus, for each fixed $\delta \in ( 0 , 3 / 8 )$ , the minimax estimation error is oforder ${ \sqrt { m / n } } .$

## 7 Self-normalized Gaussian approximation

The moment assumption (6.1) also allows us to estimate the variance of Gaussian fluctuations from the completed blocks. We use the self-normalized Bernstein inequality in Theorem 3.1 to control the empirical centering in this estimator. We then combine this control with the regenerative strong approximation of Bashtova and Shashkin [10, Theorem 1].

Keep $m : = \mathbb { E } _ { A } [ \Delta _ { 1 } ]$ , fix $\theta \in \Theta$ , and put $p : = 2 + \varepsilon$ . Define the completed-block and full-path centered sums by

$$
S _ { n , \theta } ^ { \mathrm { r e g } } : = \sum _ { j = 1 } ^ { M _ { n } - 1 } \xi _ { j , \theta } , \qquad S _ { n , \theta } ^ { \mathrm { f u l l } } : = \sum _ { t = 1 } ^ { n } \{ R ( \theta ) - \ell _ { \theta } ( Z _ { t } ) \} .
$$

Both sums are zero at $n = 0$ . On $\{ D _ { n } > 0 \}$ , define the observable quadratic sum and variance estimator by

$$
\widehat { Q } _ { n , \theta } : = \sum _ { j = 1 } ^ { M _ { n } - 1 } \left\{ L _ { j , \theta } - \Delta _ { j } \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right\} ^ { 2 } , \qquad \widehat { \sigma } _ { n , \theta } ^ { 2 } : = \frac { \widehat { Q } _ { n , \theta } } { D _ { n } } .
$$

Set both quantities to zero when $D _ { n } = 0$

Theorem 7.1. Assume that $\tau _ { 1 } = 0$ , that the regenerative blocks are i.i.d. with positive integer lengths, and that (6.1) holdsfor some $\varepsilon > 0$ and $C \geq 1$ . Keep the regenerative law and θ fixed as $n  \infty ,$ , and suppose that

$$
v _ { \theta } ^ { 2 } : = \mathbb { E } _ { A } [ \xi _ { 1 , \theta } ^ { 2 } ] > 0 , \qquad \sigma _ { \theta } ^ { 2 } : = \frac { v _ { \theta } ^ { 2 } } { m } .
$$

There is a realization of the joint law of the losses and regeneration boundaries, together with a standard Brownian motion $W _ { \theta } ,$ such that, almost surely,

$$
\operatorname* { m a x } _ { 0 \leq k \leq n } \left| S _ { k , \theta } ^ { \mathrm { r e g } } - \sigma _ { \theta } W _ { \theta } ( k ) \right| = o ( n ^ { 1 / p } ) .\tag{7.1}
$$

Moreover,

$$
\widehat { \sigma } _ { n , \theta } ^ { 2 } - \sigma _ { \theta } ^ { 2 } = \left\{ { O \big ( } n ^ { 2 / p - 1 } { \big ) } , \begin{array} { l l } { 2 < p < 4 , } \\ { O \Big ( \sqrt { \log ( n + 2 ) / n } \Big ) , } & { p \geq 4 . } \end{array}  \right.\tag{7.2}
$$

In particular, $D _ { n }$ and $\widehat { Q } _ { n , \theta }$ are positive eventually. Then,

$$
\operatorname* { m a x } _ { 0 \leq k \leq n } \left| \frac { S _ { k , \theta } ^ { \mathrm { r e g } } } { \widehat \sigma _ { n , \theta } \sqrt { n } } - \frac { W _ { \theta } ( k ) } { \sqrt { n } } \right| = o ( n ^ { 1 / p - 1 / 2 } ) ,\tag{7.3}
$$

$$
\operatorname* { s u p } _ { 0 \leq t \leq 1 } \left| \frac { S _ { \lfloor n t \rfloor , \theta } ^ { \mathrm { r e g } } } { \sqrt { \widehat Q _ { n , \theta } } } - \frac { W _ { \theta } ( n t ) } { \sqrt n } \right| = o ( n ^ { 1 / p - 1 / 2 } ) ,\tag{7.4}
$$

and conclusions similar to (7.1),(7.3), and (7.4) hold for $S ^ { \mathrm { f u l l } }$ as well. Consequently,

$$
\frac { \sqrt { D _ { n } } \{ R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \} } { \widehat { \sigma } _ { n , \theta } } = \frac { W _ { \theta } ( n ) } { \sqrt { n } } + o ( n ^ { 1 / p - 1 / 2 } ) \stackrel { d } { \longrightarrow } \mathcal { N } ( 0 , 1 ) .\tag{7.5}
$$

For both $H = \mathrm { f u l l }$ and $H = \deg _ { \mathrm { {  s } } }$ , the processes t 7→ $S _ { \lfloor n t \rfloor , \theta } ^ { H } / \sqrt { \hat { Q } _ { n , \theta } }$ converge in distribution to standard Brownian motion in $D [ 0 , 1 ]$ with the usual Skorokhod topology.

## 8 Proofs

## 8.1 Proof of Theorem 3.1

For deterministic $s \in ( 0 , 1 / u _ { n } )$ , define

$$
\begin{array} { r } { \Phi _ { s } ( x ) : = s x - \log ( 1 + s x ) , \qquad x > - 1 / s , } \end{array}
$$

and, for $m \in  { \mathbb { N } } _ { 0 }$

$$
\mathcal { M } _ { m , \theta } ( s ) : = \exp \left\{ s \sum _ { j = 1 } ^ { m } Y _ { j , \theta } - \sum _ { j = 1 } ^ { m } \Phi _ { s } ( Y _ { j , \theta } ) - m \log ( 1 + s \beta ( u _ { n } ) ) \right\}
$$

where we remind that

$$
Y _ { j , \theta } : = \xi _ { j , \theta } \mathbf { 1 } _ { \left\{ \Delta _ { j } \leq u _ { n } \right\} } .
$$

Lemma 8.1. Fix $\theta \in \Theta$ and deterministic $s \in ( 0 , 1 / u _ { n } )$ . Then $( \mathcal { M } _ { m , \theta } ( s ) ) _ { m \geq 0 }$ is a nonnegative supermartingale with respect to $( \mathcal { F } _ { m } ) _ { m \geq 0 }$ , with $\mathcal { M } _ { 0 , \theta } ( s ) = 1$ . In particular,

$$
\mathbb { E } [ \mathcal { M } _ { M _ { n } , \theta } ( s ) ] \leq 1 .\tag{8.1}
$$

Proof of Lemma 8.1. We first bound the mean introduced by truncation, then verify the supermartingale property and apply optional stopping at $M _ { n }$ , the index of the first cycle whose end is after time n. Since the untruncated reward is centered and bounded by the block length,

$$
\begin{array} { r l } & { \mathbb { E } _ { A } [ Y _ { j , \theta } ] = - \mathbb { E } _ { A } \left[ \xi _ { j , \theta } \mathbf { 1 } _ { \{ \Delta _ { j } > u _ { n } \} } \right] } \\ & { \quad \quad \quad \leq \mathbb { E } _ { A } \left[ \Delta _ { j } \mathbf { 1 } _ { \{ \Delta _ { j } > u _ { n } \} } \right] = \beta ( u _ { n } ) . } \end{array}\tag{8.2}
$$

The admissible range of s ensures that

$$
| Y _ { j , \theta } | \le u _ { n } , \qquad 1 + s Y _ { j , \theta } > 0 .
$$

Thus $\Phi _ { s } ( Y _ { j , \theta } )$ is well defined. Recall from the definition of Φ that

$$
e ^ { s x - \Phi _ { s } ( x ) } = 1 + s x .
$$

Independence of the marked cycles from the preceding block history, together with (8.2), now gives

$$
\begin{array} { r l r } {  { \mathbb { E } [ \exp \{ s Y _ { j , \theta } - \Phi _ { s } ( Y _ { j , \theta } ) - \log ( 1 + s \beta ( u _ { n } ) ) \} | \mathcal { F } _ { j - 1 } ] } } \\ & { } & { = \frac { 1 + s \mathbb { E } [ Y _ { j , \theta } ~ | ~ \mathcal { F } _ { j - 1 } ] } { 1 + s \beta ( u _ { n } ) } \le 1 . } \end{array}
$$

Multiplying by the preceding value of $\displaystyle \mathcal { M } _ { j - 1 , \theta } ( s )$ proves that the process is a nonnegative supermartingale starting at one.

It remains to check that optional stopping applies. The random index $M _ { n }$ is a stopping time because

$$
\left\{ M _ { n } \leq m \right\} = \left\{ \tau _ { m + 1 } > n \right\} \in \mathcal { F } _ { m } .
$$

Positive integer cycle lengths also imply

$$
M _ { n } \leq n + 1 .
$$

The bounded optional stopping theorem for supermartingales [22, Chapter 10, Section 7, p. 495] therefore yields

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { M } _ { M _ { n } , \theta } ( s ) ] \leq \mathbb { E } [ \mathcal { M } _ { 0 , \theta } ( s ) ] = 1 , } \end{array}
$$

which is (8.1).

We are now ready to prove the theorem.

Proof of Theorem 3.1. We apply the stopped exponential bound to both signs of the truncated reward. We then remove the first cycle ending after time n and restore the rewards discarded by truncation. We work on $\{ D _ { n } > 0 \}$ , where the empirical risk is an average over completed cycles.

Set $s : = \lambda / n \in ( 0 , 1 / u _ { n } )$ . Replacing $\ell _ { \theta }$ by $1 - \ell _ { \theta }$ changes the sign of its centered block reward. Lemma 8.1, Markov’s inequality, and a union bound therefore give an event of probability at least $1 - \delta$ on which, for each $a \in \{ - 1 , 1 \}$ ,

$$
s \sum _ { j = 1 } ^ { M _ { n } } a Y _ { j , \theta } \leq \sum _ { j = 1 } ^ { M _ { n } } \Phi _ { s } ( a Y _ { j , \theta } ) + M _ { n } \log ( 1 + s \beta ( u _ { n } ) ) + \log \frac { 2 } { \delta } .
$$

For $b \geq 0$ with $s b < 1$ , the definition of $\Phi _ { s }$ gives

$$
\Phi _ { s } ( x ) \leq q ( s b ) , \qquad x \in [ - b , b ] .
$$

Thus the completed cycles satisfy, for both signs,

$$
\Phi _ { s } ( a Y _ { j , \theta } ) \le \mathbf { 1 } _ { \{ \Delta _ { j } \le u _ { n } \} } q ( s \Delta _ { j } ) , \qquad j < M _ { n } .
$$

For the additional cycle, truncation gives

$$
| Y _ { M _ { n } , \theta } | \leq u _ { n } , \qquad \Phi _ { s } ( a Y _ { M _ { n } , \theta } ) \leq q ( s u _ { n } ) .
$$

Removing its reward costs at most $s u _ { n }$ . Since

$$
q ( s u _ { n } ) + s u _ { n } = - \log ( 1 - s u _ { n } ) ,
$$

the two signed inequalities imply

$$
\begin{array} { l } { { \displaystyle s \left| \sum _ { j = 1 } ^ { M _ { n } - 1 } Y _ { j , \theta } \right| \le \log \frac { 2 } { \delta } + M _ { n } \log ( 1 + s \beta ( u _ { n } ) ) } } \\ { { \displaystyle \qquad + \sum _ { j = 1 } ^ { M _ { n } - 1 } \mathbf { 1 } _ { \{ \Delta _ { j } \le u _ { n } \} } q ( s \Delta _ { j } ) - \log ( 1 - s u _ { n } ) } . } \end{array}
$$

To restore the untruncated rewards, use

$$
\begin{array} { r l r } {  { D _ { n } \Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | = | \sum _ { j = 1 } ^ { M _ { n } - 1 } \xi _ { j , \theta } | } } \\ & { } & { \leq | \sum _ { j = 1 } ^ { M _ { n } - 1 } Y _ { j , \theta } | + \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } \mathbf { 1 } _ { \{ \Delta _ { j } > u _ { n } \} } . } \end{array}
$$

Substitute the bound for the truncated sum and divide by $D _ { n }$ to obtain (3.1). The convention at $D _ { n } = 0$ covers the remaining case. □

## 8.2 Proof of Corollary 3.1

Proof of Corollary 3.1. We first prove an inequality that holds simultaneously for finitely many deterministic values of the tuning parameter. We can then choose one of these values using the observed block lengths. The case $D _ { n } = 0$ is covered by convention, so assume $D _ { n } > 0$

Consider the values

$$
s _ { j } : = \frac { 2 ^ { - ( j + 1 ) } } { u _ { n } } , \qquad 0 \le j < \frac { J _ { n } } { 2 } .
$$

For each of these values, apply Theorem 3.1 with $\lambda = n s _ { j }$ , and replacing δ by $\frac { 2 \delta } { J _ { n } }$ . There are $J _ { n } / 2$ such values, so a union bound gives an event of probability at least $1 - \delta$ on which all the resulting inequalities hold. On this event, the elementary bounds

$$
q ( x ) \leq x ^ { 2 } , \qquad - \log ( 1 - x ) \leq 2 x \quad ( 0 \leq x \leq 1 / 2 ) , \qquad \log ( 1 + y ) \leq y \quad ( y \geq 0 )
$$

give, simultaneously for every $s _ { j }$

$$
D _ { n } \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \leq \frac { \log ( J _ { n } / \delta ) } { s _ { j } } + s _ { j } H _ { n , u } + M _ { n } \beta ( u _ { n } ) + 2 u _ { n } + T _ { n , u } .
$$

It remains to choose one of the values $s _ { j }$ so that the sum of the first two terms is close to its minimum. If s could be chosen freely subject to $s \leq 1 / ( 2 u _ { n } )$ , the minimizing value would be

$$
s _ { * } : = \operatorname* { m i n } \left\{ \sqrt { \frac { \log ( J _ { n } / \delta ) } { H _ { n , u } } } , \frac { 1 } { 2 u _ { n } } \right\} .
$$

When $H _ { n , u } = 0$ , take $s _ { * } = 1 / ( 2 u _ { n } )$ . The smallest value in our collection is no greater than $s _ { * }$ . To see this, note that

$$
H _ { n , u } \leq D _ { n } ^ { 2 } \leq n ^ { 2 } , \qquad \log ( J _ { n } / \delta ) \geq \log 2 .
$$

If $n / u _ { n } \ge 1$ , the smallest value consequently satisfies

$$
s _ { J _ { n } / 2 - 1 } \leq \frac { 1 } { 2 n } < \frac { \sqrt { \log ( J _ { n } / \delta ) } } { n } \leq s _ { * }
$$

whenever the square-root term determines $s _ { * }$ . If the minimum is attained at the largest allowed value, then $s _ { * } = s _ { 0 } . \mathrm { I f } n / u _ { n } < 1$ , then $J _ { n } = 2$ , and the same bounds show that $s _ { * } = s _ { 0 }$ also in this case.

When the square-root term is smaller than $1 / ( 2 u _ { n } )$ , successive values $s _ { j }$ differ by a factor of two, so there is an index $j$ such that

$$
\frac { s _ { * } } { 2 } \leq s _ { j } \leq s _ { * } .
$$

Then

$$
\frac { \log ( J _ { n } / \delta ) } { s _ { j } } + s _ { j } H _ { n , u } \leq 3 \sqrt { H _ { n , u } \log ( J _ { n } / \delta ) } .
$$

When $s _ { * } = 1 / ( 2 u _ { n } )$ , choose $s _ { 0 } = 1 / ( 2 u _ { n } )$ . In this case,

$$
\frac { H _ { n , u } } { 2 u _ { n } } \leq \sqrt { H _ { n , u } \log ( J _ { n } / \delta ) } ,
$$

and hence

$$
\frac { \log ( J _ { n } / \delta ) } { s _ { 0 } } + s _ { 0 } H _ { n , u } \leq 2 u _ { n } \log ( J _ { n } / \delta ) + \sqrt { H _ { n , u } \log ( J _ { n } / \delta ) } .
$$

Substituting either estimate into the inequality that holds for every $s _ { j } ,$ and dividing by $D _ { n }$ , proves (3.2).

## 8.3 Proof of Theorem 3.2

Proof of Theorem 3.2. We integrate the Bernstein tail bound before estimating its random remainder. This first controls the deviation weighted by $D _ { n } ;$ the length outside the completed cycles accounts for removing that weight. For the calculations below, write

$$
e _ { n } : = \Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | , \qquad s : = \frac { \lambda } { n } ,
$$

and denote the logarithmic correction by

$$
\begin{array} { l } { { \displaystyle C _ { n } ( s ) : = { M _ { n } } \log ( 1 + s \beta ( u _ { n } ) ) } \ ~ } \\ { { \displaystyle M _ { n } - 1 } \ ~ } \\ { { \displaystyle ~ + ~ \sum _ { j = 1 } ^ { M _ { n } } \mathbf { 1 } _ { \{ \Delta { j } \leq u _ { n } } \} q ( s \Delta _ { j } ) - \log ( 1 - s u _ { n } ) } . } \end{array}
$$

We use $H _ { n , u }$ and $T _ { n , u }$ as defined in Corollary 3.1. All quantities are integrable. The number of blocks is at most $n + 1$ , the completed lengths and tail sum are at most n, and $s u _ { n } \leq 1 / 2$

For every $t > 0$ , apply Theorem 3.1 with $\delta = e ^ { - t }$ . It gives

$$
\mathbb { P } \left( s D _ { n } e _ { n } - C _ { n } ( s ) - s T _ { n , u } > \log 2 + t \right) \leq e ^ { - t } .
$$

This statement also holds when $D _ { n } = 0$ . Since there are no completed blocks, the expression on the left of the threshold is − ${ \cal C } _ { n } ( s ) \le 0$ . Integrating its positive part yields

$$
\mathbb { E } \left[ ( s D _ { n } e _ { n } - C _ { n } ( s ) - s T _ { n , u } - \log 2 ) _ { + } \right] \leq \int _ { 0 } ^ { \infty } e ^ { - t } d t = 1 .
$$

Therefore

$$
\begin{array} { r } { s \mathbb { E } [ D _ { n } e _ { n } ] \leq 1 + \log 2 + \mathbb { E } [ C _ { n } ( s ) ] + s \mathbb { E } [ T _ { n , u } ] . } \end{array}
$$

Since the loss and empirical risk lie in [0, 1],

$$
e _ { n } = { \frac { D _ { n } } { n } } e _ { n } + \left( 1 - { \frac { D _ { n } } { n } } \right) e _ { n } \leq { \frac { D _ { n } } { n } } e _ { n } + { \frac { n - D _ { n } } { n } } .
$$

Taking expectations and using $s n = \lambda$ , we obtain

$$
\mathbb { E } [ e _ { n } ] \leq \frac { 1 + \log { 2 } + \mathbb { E } [ C _ { n } ( s ) ] } { \lambda } + \frac { \mathbb { E } [ T _ { n , u } ] } { n } + \frac { n - \mathbb { E } [ D _ { n } ] } { n } .\tag{8.3}
$$

We bound the sum over completed cycles by including cycle $M _ { n }$ , the first cycle whose end is after time $n .$ The enlarged sum ends at a stopping time, so we can use the following expectation identity. For any measurable $f : \mathbb { N } \to [ 0 , \infty )$ with $\mathbb { E } _ { A } [ f ( \Delta _ { 1 } ) ] < \infty$ , independence and $\{ M _ { n } \geq j \} \in \mathcal { F } _ { j - 1 }$ give

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \sum _ { j = 1 } ^ { M _ { n } } f ( \Delta _ { j } ) \right] = \sum _ { j = 1 } ^ { n + 1 } \mathbb { E } \left[ \mathbf { 1 } _ { \{ M _ { n } \geq j \} } f ( \Delta _ { j } ) \right] } & { } \\ { \displaystyle } & { = \sum _ { j = 1 } ^ { n + 1 } \mathbb { P } ( M _ { n } \geq j ) \mathbb { E } _ { A } [ f ( \Delta _ { 1 } ) ] } \\ { \displaystyle } & { = \mathbb { E } [ M _ { n } ] \mathbb { E } _ { A } [ f ( \Delta _ { 1 } ) ] . } \end{array}
$$

Apply this identity to the truncated squared length and the discarded length. It follows that

$$
\mathbb { E } [ H _ { n , u } ] \leq \mathbb { E } [ M _ { n } ] v ( u _ { n } ) ,\tag{8.4}
$$

$$
\mathbb { E } [ T _ { n , u } ] \leq \mathbb { E } [ M _ { n } ] \beta ( u _ { n } ) .\tag{8.5}
$$

For the logarithmic terms, the admissible range gives

$$
\begin{array} { r l } { q ( x ) \le x ^ { 2 } , } & { { } \quad 0 \le x \le 1 / 2 , } \\ { - \log ( 1 - x ) \le 2 x , } & { { } \quad 0 \le x \le 1 / 2 . } \end{array}
$$

Together with $\log ( 1 + x ) \leq x$ for nonnegative x, these bounds imply

$$
\begin{array} { r l } & { \mathbb { E } [ C _ { n } ( s ) ] \le s \beta ( u _ { n } ) \mathbb { E } [ M _ { n } ] + s ^ { 2 } \mathbb { E } [ H _ { n , u } ] + 2 s u _ { n } } \\ & { \quad \quad \le s \beta ( u _ { n } ) \mathbb { E } [ M _ { n } ] + s ^ { 2 } v ( u _ { n } ) \mathbb { E } [ M _ { n } ] + 2 s u _ { n } . } \end{array}
$$

Substituting this estimate and (8.5) into (8.3), and replacing s by $\lambda / n ,$ , proves (3.3).

## 8.4 Proof of Corollary 4.1 and Corollary 4.2

For both proofs, we will use the following notations:

$$
S _ { 0 } : = 0 , \qquad S _ { k } : = \sum _ { j = 1 } ^ { k } \Delta _ { j } \quad ( k \in \mathbb { N } ) , \qquad N _ { n } : = \operatorname* { m a x } \{ k \ge 0 : S _ { k } \le n \} .
$$

Then $M _ { n } = N _ { n } + 1 , D _ { n } = S _ { N _ { n } }$ , and

$$
A _ { n } : = n - D _ { n } , \qquad m : = \mathbb { E } _ { A } [ \Delta _ { 1 } ] .
$$

Finally, recall that

$$
r _ { n } = { \frac { \mathbb { E } [ M _ { n } ] } { n } } .
$$

Since the lengths are positive integers, $M _ { n } \leq n + 1$ and $r _ { n } \leq ( n + 1 ) / n \leq 2$ . For a fixed regenerative law, $r _ { n }  1 / m$ and $\mathbb { E } [ D _ { n } ] / n  1$

Proof of Corollary 4.1. We first bound the length-dependent quantities in Theorem 3.2, then minimize the resulting bound in λ.

For every integer $u \geq 1$ , the tail-sum identity gives

$$
\begin{array} { r l } {  { \beta ( u ) = u \mathbb { P } _ { A } ( \Delta _ { 1 } > u ) + \sum _ { k = u } ^ { \infty } \mathbb { P } _ { A } ( \Delta _ { 1 } > k ) } } \\ & { \le C _ { \mathrm { e } } u e ^ { - \kappa u } + C _ { \mathrm { e } } \sum _ { k = u } ^ { \infty } e ^ { - \kappa k } } \\ & { = C _ { \mathrm { e } } e ^ { - \kappa u } ( u + \frac { 1 } { 1 - e ^ { - \kappa } } ) , } \end{array}
$$

which proves (4.1). The prescribed truncation level satisfies

$$
e ^ { - \kappa u _ { n } } \leq \frac { 1 } { n } , \qquad 1 \leq u _ { n } \leq 1 + \frac { \log n } { \kappa } ,
$$

and therefore gives both inequalities in (4.2). The second-moment tail-sum identity similarly gives

$$
\begin{array} { l } { \displaystyle { v _ { 2 } = 1 + \sum _ { k = 1 } ^ { \infty } ( 2 k + 1 ) \mathbb { P } _ { A } ( \Delta _ { 1 } > k ) } } \\ { \displaystyle { \le 1 + C _ { \mathrm { e } } \sum _ { k = 1 } ^ { \infty } ( 2 k + 1 ) e ^ { - \kappa k } = 1 + \frac { C _ { \mathrm { e } } e ^ { - \kappa } ( 3 - e ^ { - \kappa } ) } { ( 1 - e ^ { - \kappa } ) ^ { 2 } } . } } \end{array}
$$

In particular, $v ( u _ { n } ) \leq v _ { 2 } < \infty$ . Theorem 3.2 now gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq \displaystyle \frac { 1 + \log 2 } { \lambda } + \displaystyle \frac { r _ { n } \lambda v _ { 2 } } { n } + 2 r _ { n } \beta ( u _ { n } ) } \\ & { \qquad \displaystyle + \displaystyle \frac { 2 u _ { n } } { n } + \displaystyle \frac { \mathbb { E } [ A _ { n } ] } { n } . } \end{array}\tag{8.6}
$$

To bound the renewal age, define

$$
U ( s ) : = \sum _ { k \geq 0 } \mathbb { P } ( S _ { k } = s ) , \qquad s \in \mathbb { N } _ { 0 } .
$$

The renewal epochs are strictly increasing, so $U ( s ) \leq 1$ . If $A _ { n } \ \geq \ t ,$ the last renewal occurs at some $s \leq n - t .$ , and the following cycle ends after n. Independence of that next cycle gives

$$
\begin{array} { r l } {  { \mathbb { P } ( A _ { n } \geq t ) = \sum _ { s = 0 } ^ { n - t } U ( s ) \mathbb { P } _ { A } ( \Delta _ { 1 } > n - s ) } \quad } & { } \\ & { \leq \displaystyle \sum _ { r = t } ^ { n } \mathbb { P } _ { A } ( \Delta _ { 1 } > r ) , \qquad 1 \leq t \leq n . } \end{array}\tag{8.7}
$$

Summing over t and changing the order of the finite sums yields

$$
\begin{array} { r l r } {  { \mathbb { E } [ A _ { n } ] \le \sum _ { r = 1 } ^ { n } r \mathbb { P } _ { A } ( \Delta _ { 1 } > r ) } } \\ & { } & { \le C _ { \mathrm { e } } \sum _ { r = 1 } ^ { \infty } r e ^ { - \kappa r } = \frac { C _ { \mathrm { e } } e ^ { - \kappa } } { ( 1 - e ^ { - \kappa } ) ^ { 2 } } . } \end{array}\tag{8.8}
$$

Substituting (4.1) and (8.8) into (8.6) proves (4.3).

For the optimization, let $a , b , c > 0$ . Differentiating $a / \lambda + b \lambda$ shows that its minimizer on $0 < \lambda \leq \mathfrak { c }$ c is $\lambda _ { * } = \operatorname* { m i n } \{ \sqrt { a / b } , c \}$ , and

$$
\operatorname* { m i n } _ { 0 < \lambda \leq c } \left( { \frac { a } { \lambda } } + b \lambda \right) = { \binom { 2 { \sqrt { a b } } , } { a / c + b c , } } \quad { \sqrt { a / b } } \leq c , \quad \leq 2 { \sqrt { a b } } + { \frac { a } { c } } .\tag{8.9}
$$

Indeed, in the second case $b c < { \sqrt { a b } }$ . Taking

$$
a = 1 + \log 2 , \qquad b = { \frac { r _ { n } v _ { 2 } } { n } } , \qquad c = { \frac { n } { 2 u _ { n } } }
$$

proves that (4.4) is the exact minimizer of (4.3). Moreover,

$$
\frac { 1 + \log 2 } { \lambda _ { n } } + \frac { r _ { n } \lambda _ { n } v _ { 2 } } { n } \leq 2 \sqrt { \frac { ( 1 + \log 2 ) r _ { n } v _ { 2 } } { n } } + \frac { 2 ( 1 + \log 2 ) u _ { n } } { n } .
$$

Combining this inequality with $e ^ { - \kappa u _ { n } } \leq 1 / n$ proves (4.5).

Finally, we record the renewal limits, which also apply under the polynomial assumption because they require only finite mean cycle length. The strong law gives $S _ { k } / k $ m almost surely. Since

$$
S _ { N _ { n } } \leq n < S _ { N _ { n } + 1 }
$$

and $N _ { n }  \infty$ , it follows that

$$
\frac { N _ { n } } { n } \longrightarrow \frac { 1 } { m } , \qquad \frac { M _ { n } } { n } \longrightarrow \frac { 1 } { m } , \qquad \frac { D _ { n } } { n } \longrightarrow 1 \qquad \mathrm { a l m o s t s u r e l y } .
$$

The bounds $M _ { n } / n \leq 2$ and $D _ { n } / n \leq 1$ permit dominated convergence, so

$$
r _ { n } \longrightarrow \frac { 1 } { m } , \qquad \frac { \mathbb { E } [ D _ { n } ] } { n } \longrightarrow 1 .
$$

As $u _ { n } = O ( \log n )$ , the constraint on $\lambda _ { n }$ is eventually inactive and $\lambda _ { n } \sim \sqrt { ( 1 + \log { 2 } ) m n / v _ { 2 } }$ . In (4.5), the square-root term has the leading constant stated in (4.6), and all the other terms are $O ( ( \log n ) / n )$ □

Proof of Corollary 4.2. For every real $u \geq 1$ , the tail-integral identity gives

$$
\begin{array} { r l } & { \displaystyle \beta ( u ) = u \mathbb { P } _ { A } ( \Delta _ { 1 } > u ) + \int _ { u } ^ { \infty } \mathbb { P } _ { A } ( \Delta _ { 1 } > t ) d t } \\ & { \quad \quad \quad \leq C _ { \mathrm { p } } u ^ { 1 - \alpha } + C _ { \mathrm { p } } \int _ { u } ^ { \infty } t ^ { - \alpha } d t = \frac { \alpha C _ { \mathrm { p } } } { \alpha - 1 } u ^ { 1 - \alpha } . } \end{array}
$$

This proves (4.7). Since $u _ { n } \ge n ^ { 1 / \alpha }$ and $1 - \alpha < 0$ , it also proves (4.8).

For the truncated second moment, Tonelli’s theorem yields

$$
\begin{array} { r l r } & { } & { v ( u ) \le \mathbb { E } _ { A } [ ( \Delta _ { 1 } \wedge u ) ^ { 2 } ] = 2 \displaystyle \int _ { 0 } ^ { u } t \mathbb { P } _ { A } ( \Delta _ { 1 } > t ) d t } \\ & { } & { \qquad \le 1 + 2 C _ { \mathrm { p } } \displaystyle \int _ { 1 } ^ { u } t ^ { 1 - \alpha } d t . } \end{array}
$$

For $\alpha > 2 .$ , the last integral is at most $1 / ( \alpha - 2 )$ ; for $\alpha = 2 ,$ , it is log $u ;$ and for $1 < \alpha < 2 ,$ it is at most $u ^ { 2 - \alpha } / ( 2 - \alpha )$ . Using $u ^ { 2 - \alpha } \geq 1$ in the last case proves

$$
v ( u ) \leq V _ { \alpha } ( u ) .
$$

Consequently, Theorem 3.2 gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left| R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \right| \right] \leq \frac { 1 + \log 2 } { \lambda } + \frac { r _ { n } \lambda } { n } V _ { \alpha } ( u _ { n } ) + 2 r _ { n } \beta ( u _ { n } ) } \\ { \displaystyle \qquad + \frac { 2 u _ { n } } { n } + \frac { \mathbb { E } [ A _ { n } ] } { n } . } \end{array}\tag{8.10}
$$

The renewal-age argument in (8.7) applies without any exponential tail assumption. Thus

$$
\mathbb { E } [ A _ { n } ] \leq \sum _ { r = 1 } ^ { n } r \mathbb { P } _ { A } ( \Delta _ { 1 } > r ) \leq C _ { \mathrm { p } } \sum _ { r = 1 } ^ { n } r ^ { 1 - \alpha } .
$$

The decreasing summand can be bounded by an integral:

$$
\sum _ { r = 1 } ^ { n } r ^ { 1 - \alpha } \leq \left\{ \begin{array} { l l } { 1 + \displaystyle \int _ { 1 } ^ { \infty } t ^ { 1 - \alpha } d t = 1 + \displaystyle \frac { 1 } { \alpha - 2 } , } & { \alpha > 2 , } \\ { 1 + \log n , } & { \alpha = 2 , } \\ { 1 + \displaystyle \frac { n ^ { 2 - \alpha } } { 2 - \alpha } , } & { 1 < \alpha < 2 . } \end{array} \right.
$$

It follows that

$$
\mathbb { E } [ A _ { n } ] \leq G _ { \alpha } ( n ) .
$$

Substitution into (8.10) proves (4.9).

Apply (8.9) with

$$
a = 1 + \log 2 , \quad \quad b = { \frac { r _ { n } V _ { \alpha } ( u _ { n } ) } { n } } , \quad \quad c = { \frac { n } { 2 u _ { n } } } .
$$

This proves that (4.10) minimizes (4.9), and gives

$$
\frac { 1 + \log 2 } { \lambda _ { n } } + \frac { r _ { n } \lambda _ { n } V _ { \alpha } ( u _ { n } ) } { n } \leq 2 \sqrt { \frac { ( 1 + \log 2 ) r _ { n } V _ { \alpha } ( u _ { n } ) } { n } } + \frac { 2 ( 1 + \log 2 ) u _ { n } } { n } .
$$

Adding the remaining terms proves (4.11).

For the rates, the truncation level satisfies

$$
n ^ { 1 / \alpha } \leq u _ { n } \leq 2 n ^ { 1 / \alpha } , \qquad u _ { n } ^ { 1 - \alpha } \leq n ^ { - ( \alpha - 1 ) / \alpha } , \qquad \frac { u _ { n } } { n } \leq 2 n ^ { - ( \alpha - 1 ) / \alpha } .
$$

Since $r _ { n } \le 2$ , the definitions of $V _ { \alpha }$ and $G _ { \alpha }$ give

$$
\frac { V _ { \alpha } ( u _ { n } ) } { n } = \left\{ \begin{array} { l l } { O ( n ^ { - 1 } ) , } & { \alpha > 2 , } \\ { O ( ( \log n ) / n ) , } & { \alpha = 2 , } \\ { O \big ( n ^ { - 2 ( \alpha - 1 ) / \alpha } \big ) , } & { 1 < \alpha < 2 , } \end{array} \right.
$$

and

$$
\frac { G _ { \alpha } ( n ) } { n } = \left\{ \begin{array} { l l } { O ( n ^ { - 1 } ) , } & { \alpha > 2 , } \\ { O ( ( \log n ) / n ) , } & { \alpha = 2 , } \\ { O ( n ^ { 1 - \alpha } ) , } & { 1 < \alpha < 2 . } \end{array} \right.
$$

When $\alpha > 2$ , the square-root term in (4.11) is $O ( n ^ { - 1 / 2 } )$ , and the other terms are smaller because $( \alpha -$ $1 ) / \alpha > 1 / 2$ . When $\alpha = 2$ , the square-root term is $O ( { \sqrt { ( \log n ) / n } } )$ , the tail and truncation terms are $O ( n ^ { - 1 / 2 } )$ , and the age term is $O ( ( \log n ) / n )$ . When $1 < \alpha < 2$ , the square-root, tail, and truncation terms are all ${ \cal O } ^ { ' } ( n ^ { - ( \alpha - 1 ) / \alpha } )$ , while

$$
n ^ { 1 - \alpha } = o \big ( n ^ { - ( \alpha - 1 ) / \alpha } \big ) .
$$

This proves (4.12)–(4.14).

For the remarks, if $\mathbb { P } _ { A } ( \Delta _ { 1 } > t ) \sim c t ^ { - \alpha }$ , the same tail-integral identity gives

$$
\beta ( u ) \sim \frac { \alpha c } { \alpha - 1 } u ^ { 1 - \alpha } .
$$

Together with $u _ { n } \sim n ^ { 1 / \alpha }$ and $r _ { n }  1 / m$ , this proves that the tail contribution can have leading order when $1 < \alpha < 2 . { \mathrm { ~ A t ~ } } \alpha = 2$ , if $v _ { 2 } < \infty$ , replacing $V _ { 2 } ( u _ { n } )$ by $v _ { 2 }$ makes the square-root term $O ( n ^ { - 1 / 2 } )$ , and all remaining terms are of this order or smaller. Moreover,

$$
u \beta ( u ) \leq \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } \mathbf { 1 } _ { \{ \Delta _ { 1 } > u \} } ] \longrightarrow 0 ,
$$

by integrability of $\Delta _ { 1 } ^ { 2 }$ . Hence $\beta ( u _ { n } ) = o ( n ^ { - 1 / 2 } )$ for $u _ { n } = \lceil \sqrt { n } \rceil$ , as asserted.

## 8.5 Proof of Corollary 4.3

Proof. We verify non-Markovianity, regeneration, and the cycle-length bounds, then apply Corollary 4.1. The histories

$$
( Q _ { 0 } , Q _ { 1 } , Q _ { 2 } ) = ( 0 , 0 , 1 ) , \qquad ( Q _ { 0 } , Q _ { 1 } , Q _ { 2 } ) = ( 0 , 1 , 1 )
$$

have probabilities $( 1 - p ) p$ and $p ^ { 2 } .$ , respectively. In the first history, the customer has one service slot remaining. In the second, the first customer has departed and the remaining customer’s service has not started. Consequently,

$$
\begin{array} { r l } & { \mathbb { P } ( Q _ { 3 } = 0 \mid Q _ { 0 } = 0 , Q _ { 1 } = 0 , Q _ { 2 } = 1 ) = 1 - p , } \\ & { \mathbb { P } ( Q _ { 3 } = 0 \mid Q _ { 0 } = 0 , Q _ { 1 } = 1 , Q _ { 2 } = 1 ) = 0 . } \end{array}
$$

Both histories end at $Q _ { 2 } = 1 , \operatorname { s o } \left( Q _ { t } \right)$ is not Markov with respect to its natural filtration.

Each $\tau _ { j }$ is a stopping time for $\sigma ( Q _ { 0 } , \ldots , Q _ { t } )$ and for $\mathcal G _ { t } : = \sigma ( I _ { 0 } , . . . , I _ { t - 1 } )$ . At an empty epoch, the queue’s future is determined by the unused arrivals $( I _ { \tau _ { j } + r } ) _ { r \in  { \mathbb { N } } _ { 0 } }$ , which are independent of $\mathcal { G } _ { \tau _ { j } }$ . The same service rule applies at every empty state, so the marked cycles are i.i.d. once their almost sure finiteness is established.

If $I _ { 0 } = 0 ,$ , then $\Delta _ { 1 } = 1$ . Conditional on $I _ { 0 } = 1$ , let $T \in \mathbb { N } \cup \{ \infty \}$ be the number of customers served before the queue next becomes empty. The cycle lasts $2 T$ slots. Define the potential offspring counts by

$$
X _ { 1 } : = I _ { 1 } , \qquad X _ { i } : = I _ { 2 i - 2 } + I _ { 2 i - 1 } , \qquad i \ge 2 .
$$

While the queue is busy, these count the new arrivals during each successive service. The first-slot arrival is the initial customer itself. Conditional on $I _ { 0 } = 1$ , these counts are independent, with $X _ { 1 } \sim \mathrm { B e r n o u l l i } ( p )$ and $X _ { i } \sim$ Binomial $. ( 2 , p )$ for $i \geq 2$ . The unused arrivals define the counts even after the busy period ends.

For an independent $V \sim$ Bernoulli(p), put $X _ { 1 } ^ { \prime } : = X _ { 1 } + V$ and $X _ { i } ^ { \prime } : = X _ { i }$ for $i \geq 2 .$ . The resulting counts are i.i.d. Binomial $( 2 , p )$ . Their Galton–Watson total progeny $T ^ { \prime }$ dominates $T ,$ and is finite almost surely because the offspring mean is $2 p < 1$ . If $T ^ { \prime } \geq k$ , the first $k - 1$ customers have at least $k - 1$ offspring. Therefore, for each integer $k \geq 1$ , the binomial Chernoff bound gives

$$
\begin{array} { r l } & { \mathbb { P } ( T \geq k \mid I _ { 0 } = 1 ) \leq \mathbb { P } ( T ^ { \prime } \geq k ) } \\ & { \qquad \leq \mathbb { P } \mathrm { ( B i n o m i a l ( 2 ( k - 1 ) } , p ) \geq k - 1 ) \leq e ^ { - 2 ( k - 1 ) \kappa _ { 0 } } . } \end{array}
$$

For $k = 1$ , the last bound is simply one. Taking $k = \lfloor t / 2 \rfloor + 1$ yields

$$
\mathbb { P } ( 2 T > t | I _ { 0 } = 1 ) \le e ^ { 2 \kappa _ { 0 } } e ^ { - \kappa _ { 0 } t } , \qquad t \ge 0 .
$$

For $t \geq 1$ , only $I _ { 0 } = 1$ contributes to $\mathbb { P } ( \Delta _ { 1 } > t )$ . For $0 \leq t < 1 , \mathbb { P } ( \Delta _ { 1 } > t ) = 1 \leq e ^ { 2 \kappa _ { 0 } } e ^ { - \kappa _ { 0 } t }$ . This proves the stated exponential tail, finiteness of every cycle-length moment, and almost sure finiteness of all regeneration times.

To calculate the first two moments, let $d : = 1 - 2 p$ , and let H be the total progeny of a Galton–Watson process with offspring $X \sim { \mathrm { B i n o m i a l } } ( 2 , p )$ . Its branching recursion

$$
H \ { \stackrel { d } { = } } \ 1 + \sum _ { i = 1 } ^ { X } H _ { i } ,
$$

where the $H _ { i }$ are independent copies of H, independent of X, gives

$$
\mathbb { E } [ H ] = 1 + 2 p \mathbb { E } [ H ] = \frac { 1 } { d } .
$$

Taking second moments in the same recursion and using $\mathbb { E } [ X ( X - 1 ) ] = 2 p ^ { 2 }$ , we obtain

$$
d \mathbb { E } [ H ^ { 2 } ] = 1 + 4 p \mathbb { E } [ H ] + 2 p ^ { 2 } ( \mathbb { E } [ H ] ) ^ { 2 } , \qquad \mathbb { E } [ H ^ { 2 } ] = \frac { 1 - 2 p ^ { 2 } } { d ^ { 3 } } .
$$

The initial customer has only $J \sim$ Bernoulli(p) offspring, so, conditional on $I _ { 0 } = 1 , T$ has the distribution of $1 + J H$ , with J and H independent. Hence

$$
\mathbb { E } [ T \mid I _ { 0 } = 1 ] = 1 + \frac { p } { d } = \frac { 1 - p } { d } ,
$$

and

$$
\mathbb { E } [ T ^ { 2 } \mid I _ { 0 } = 1 ] = 1 + { \frac { 2 p } { d } } + { \frac { p ( 1 - 2 p ^ { 2 } ) } { d ^ { 3 } } } = { \frac { ( 1 - p ) ( 1 - 2 p + 2 p ^ { 2 } ) } { d ^ { 3 } } } .
$$

Since $\Delta _ { 1 } = 1$ on $\{ I _ { 0 } = 0 \}$ and $\Delta _ { 1 } = 2 T$ on $\{ I _ { 0 } = 1 \}$ ,

$$
\begin{array} { r l } & { \mathbb { E } [ \Delta _ { 1 } ] = ( 1 - p ) + 2 p \mathbb { E } [ T \mid I _ { 0 } = 1 ] , } \\ & { \mathbb { E } [ \Delta _ { 1 } ^ { 2 } ] = ( 1 - p ) + 4 p \mathbb { E } [ T ^ { 2 } \mid I _ { 0 } = 1 ] . } \end{array}
$$

Substitution and simplification prove (4.15).

All the assumptions of Corollary 4.1 now hold with $C _ { \mathrm { e } } = C _ { 0 } , \kappa = \kappa _ { 0 }$ , and $\tau _ { 1 } = 0$ . Its tail bounds give (4.16) and (4.17), and its expected deviation and optimization statements give (4.18)–(4.20). Finally, $r _ { n }  1 / m$ and

$$
\frac { v _ { 2 } } { m } = \frac { 1 - 2 p + 4 p ^ { 2 } } { ( 1 - 2 p ) ^ { 2 } } ,
$$

so its asymptotic bound becomes (4.21).

## 8.6 Proof of Theorem 5.1

Lemma 8.2. For every deterministic $s \in ( 0 , 1 / u _ { n } )$ and every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ simultaneously for every $\rho \ll \mu ,$

$$
\begin{array} { r l r } {  { s \mathbb { E } _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } } Y _ { j , \theta } \le \mathbb { E } _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } } \Phi _ { s } ( Y _ { j , \theta } ) } } \\ & { } & { \quad + M _ { n } \log ( 1 + s \beta ( u _ { n } ) ) + \mathrm { K L } ( \rho \| \mu ) + \log ( \frac { 1 } { \delta } ) . } \end{array}\tag{8.11}
$$

ProofofLemma 8.2. We average the stopped exponential bound under the prior. Markov’s inequality then gives a common probability event, and the Donsker–Varadhan formula extends the bound to every posterior. This is the standard PAC–Bayes argument; see the proof of Theorem 2.1 in [4].

Lemma 8.1 and Tonelli’s theorem give

$$
\mathbb { E } \left[ \int _ { \Theta } \mathcal { M } _ { M _ { n } , \theta } ( s ) \mu ( d \theta ) \right] \leq 1 .
$$

Hence, with probability at least $1 - \delta .$

$$
\int _ { \Theta } \mathcal { M } _ { M _ { n } , \theta } ( s ) \mu ( d \theta ) \leq \delta ^ { - 1 } .
$$

On this event, the Donsker–Varadhan formula gives, simultaneously for every $\rho \ll \mu ,$

$$
\begin{array} { r l r } {  { \mathbb { E } _ { \theta \sim \rho } \log \mathcal { M } _ { M _ { n } , \theta } ( s ) \le \mathrm { K L } ( \rho \| \mu ) + \log \int _ { \Theta } \mathcal { M } _ { M _ { n } , \theta } ( s ) \mu ( d \theta ) } } \\ & { } & { \le \mathrm { K L } ( \rho \| \mu ) + \log ( 1 / \delta ) . } \end{array}
$$

To identify the desired inequality, expand the logarithm:

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { \theta \sim \rho } \log \mathcal { M } _ { M _ { n } , \theta } ( s ) = s \mathbb { E } _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } } Y _ { j , \theta } - \mathbb { E } _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } } \Phi _ { s } ( Y _ { j , \theta } ) } } \\ { { - M _ { n } \log ( 1 + s \beta ( u _ { n } ) ) . } } \end{array}
$$

Substituting this identity in the preceding bound and rearranging proves (8.11).

Theorem 5.1. We apply the stopped PAC–Bayes inequality, restrict its sum to the cycles completed by time $n ,$ and restore the losses excluded by truncation. The case $D _ { n } = 0$ is covered by convention. Throughout the proof, set

$$
s : = \frac { \lambda } { n } , \qquad 0 < s < \frac { 1 } { u _ { n } } .
$$

Assume that the event given in Lemma 8.2, which has probability at least $1 - \delta .$ , holds. Its sum includes cycle $M _ { n }$ , the first cycle whose end is after time $n .$ . We bound its contribution before restricting the sum to the completed cycles. The correction for each completed block satisfies

$$
\Phi _ { s } ( Y _ { j , \theta } ) \leq \mathbf { 1 } _ { \{ \Delta _ { j } \leq u _ { n } \} } q ( s \Delta _ { j } ) , \qquad j < M _ { n } .
$$

For cycle $M _ { n }$ , truncation gives

$$
| Y _ { M _ { n } , \theta } | \leq u _ { n } , \qquad \Phi _ { s } ( Y _ { M _ { n } , \theta } ) \leq q ( s u _ { n } ) .
$$

In particular, removing its reward costs at most $u _ { n }$

$$
\sum _ { j = 1 } ^ { M _ { n } - 1 } Y _ { j , \theta } = \sum _ { j = 1 } ^ { M _ { n } } Y _ { j , \theta } - Y _ { M _ { n } , \theta } \leq \sum _ { j = 1 } ^ { M _ { n } } Y _ { j , \theta } + u _ { n } .
$$

Subtracting this cycle’s reward and bounding its correction therefore costs at most

$$
u _ { n } + \frac { q ( s u _ { n } ) } { s } = - \frac { \log ( 1 - s u _ { n } ) } { s } .
$$

Combining these estimates with (8.11) yields, simultaneously for every $\rho \ll \mu ,$

$$
\begin{array} { r l r } {  { \mathbb { E } _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } - 1 } Y _ { j , \theta } \le \frac { 1 } { s } \Bigg [ \mathrm { K L } ( \rho \| \mu ) + \log \frac { 1 } { \delta } + M _ { n } \log ( 1 + s \beta ( u _ { n } ) ) } } \\ & { } & { + \sum _ { j = 1 } ^ { M _ { n } - 1 } \mathbf { 1 } _ { \{ \Delta _ { j } \le u _ { n } \} } q ( s \Delta _ { j } ) - \log ( 1 - s u _ { n } ) \Bigg ] . } \end{array}\tag{8.12}
$$

Since $| \xi _ { j , \theta } | \le \Delta _ { j }$

$$
\begin{array} { r } {  { \sum _ { j = 1 } ^ { M _ { n } - 1 } \xi _ { j , \theta } = \sum _ { j = 1 } ^ { M _ { n } - 1 } Y _ { j , \theta } + \sum _ { j = 1 } ^ { M _ { n } - 1 } \xi _ { j , \theta } \mathbf { 1 } _ { \{ \Delta _ { j } > u _ { n } \} } } \quad } \\ { \le \sum _ { j = 1 } ^ { M _ { n } - 1 } Y _ { j , \theta } + \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } \mathbf { 1 } _ { \{ \Delta _ { j } > u _ { n } \} } . } \end{array}
$$

The completed discrepancies equal the empirical risk error multiplied by the observed length:

$$
\sum _ { j = 1 } ^ { M _ { n } - 1 } \xi _ { j , \theta } = D _ { n } \{ R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \} .
$$

Average over $\rho ,$ substitute (8.12), and divide by $D _ { n }$ . Recalling $s = \lambda / n$ gives (5.1) on the same event, simultaneously for all $\rho \ll \mu$ □

## 8.7 Proof of Theorem 5.2

Proof of Theorem 5.2. We first control the forward deviation simultaneously over the grid with a common integrable remainder. This allows us to select $\lambda$ from the data before comparing with a fixed posterior and a fixed grid point.

For $\lambda \in \Lambda _ { n }$ , set $s = \lambda / n$ , and write

$$
C _ { n } ( s ) : = M _ { n } \log ( 1 + s \beta ( u _ { n } ) ) + Q _ { n } ( \lambda ) , \qquad Z _ { \lambda } : = \int _ { \Theta } \mathcal { M } _ { M _ { n } , \theta } ( s ) \mu ( d \theta ) .
$$

Lemma 8.1 and Tonelli’s theorem imply $\mathbb { E } [ Z _ { \lambda } ] \le 1$ . As in the proof of Theorem 5.1, removing the cycle ending after time n gives, pathwise for every θ,

$$
s \sum _ { j = 1 } ^ { M _ { n } - 1 } Y _ { j , \theta } - C _ { n } ( s ) \leq \log \mathcal { M } _ { M _ { n } , \theta } ( s ) .\tag{8.13}
$$

Indeed, the completed-cycle corrections are bounded by the sum in $Q _ { n } ( \lambda )$ , while

$$
s Y _ { M _ { n } , \theta } - \Phi _ { s } ( Y _ { M _ { n } , \theta } ) \geq - s u _ { n } - q ( s u _ { n } ) = \log ( 1 - s u _ { n } ) .
$$

The Donsker–Varadhan formula, followed by restoration of the truncated rewards, therefore gives, for every $\rho \ll \mu$ with finite KL divergence,

$$
\begin{array} { c l } { { } } & { { \displaystyle { D _ { n } \mathbb E } _ { \theta \sim \rho } R ( \theta ) \le \mathbb E _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \theta } + \frac n \lambda \left\{ { \mathrm { K L } } ( \rho \| \mu ) + Q _ { n } ( \lambda ) + \log Z _ { \lambda } \right\} } } \\ { { } } & { { \displaystyle ~ + ~ M _ { n } \beta ( u _ { n } ) + T _ { n , u } . } } \end{array}
$$

Here we used log $( 1 + x ) \leq x$ to bound the population tail term independently of $\lambda .$

Define $x _ { + } : = \operatorname* { m a x } \{ x , 0 \}$ and

$$
W _ { n } : = \operatorname* { m a x } _ { \lambda \in \Lambda _ { n } } \frac { n } { \lambda } \{ \log Z _ { \lambda } - p _ { n } ( \lambda ) \} _ { + } .
$$

Markov’s inequality gives, for $t \geq 0$

$$
\mathbb { P } \{ \log Z _ { \lambda } > p _ { n } ( \lambda ) + t \} \le e ^ { - p _ { n } ( \lambda ) - t } .
$$

Integrating this inequality and bounding the maximum by a sum yields

$$
\mathbb { E } [ W _ { n } ] \le \sum _ { \lambda \in \Lambda _ { n } } \frac { n } { \lambda } e ^ { - p _ { n } ( \lambda ) } = \frac { n } { \lambda _ { + } } .\tag{8.14}
$$

Consequently, simultaneously for every grid point and posterior,

$$
\begin{array} { c l } { { } } & { { \displaystyle { D _ { n } \mathbb E } _ { \theta \sim \rho } R ( \theta ) \le \mathbb E _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \theta } + \frac n \lambda \left\{ { \mathrm { K L } } ( \rho \| \mu ) + p _ { n } ( \lambda ) + Q _ { n } ( \lambda ) \right\} } } \\ { { } } & { { \displaystyle ~ + M _ { n } \beta ( u _ { n } ) + T _ { n , u } + W _ { n } . } } \end{array}\tag{8.15}
$$

The quantities used below are integrable for the Gibbs posteriors and for each fixed comparator with finite KL divergence: $M _ { n } \leq n + 1 , \lambda u _ { n } / n < 1$ , and comparison with $\mu$ in the Gibbs optimization gives

$$
0 \leq \mathrm { K L } ( \widehat { \rho } _ { \lambda } \| \mu ) \leq \frac { \lambda D _ { n } } { n } \leq \lambda .
$$

When $D _ { n } = 0$ , the Gibbs posterior is $\mu .$

Apply (8.15) to $\lambda = \dot { \lambda }$ and $\rho = { \widehat { \rho } } _ { \widehat { \lambda } } .$ . By (5.3),

$$
D _ { n } \mathbb { E } _ { \theta \sim \widehat { \rho } _ { \widehat { \lambda } } } R ( \theta ) \leq \widehat { \mathcal { C } } _ { n } ( \widehat { \lambda } ) + M _ { n } \beta ( u _ { n } ) + W _ { n } \leq \widehat { \mathcal { C } } _ { n } ( \lambda ) + M _ { n } \beta ( u _ { n } ) + W _ { n }
$$

for every fixed $\lambda \in \Lambda _ { n }$ . The Gibbs variational identity further gives, for every fixed $\rho \ll \mu$ with finite KL divergence,

$$
\begin{array} { r l r } {  { D _ { n } \mathbb { E } _ { \theta \sim \hat { \rho } _ { \hat { \lambda } } } R ( \theta ) \le \mathbb { E } _ { \theta \sim \rho } \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \theta } + \frac { n } { \lambda } \{ { \mathrm { K L } } ( \rho \| \mu ) + p _ { n } ( \lambda ) + Q _ { n } ( \lambda ) \} } } \\ & { } & { + M _ { n } \beta ( u _ { n } ) + T _ { n , u } + W _ { n } . } \end{array}\tag{8.16}
$$

This argument uses no division by $D _ { n } .$ , so it also covers $D _ { n } = 0 .$

For the reverse comparison, apply (8.13) and the Donsker–Varadhan formula to the complementary loss $1 - \ell _ { \theta } .$ , keeping λ and $\rho$ fixed. The corresponding exponential integral $Z _ { \lambda } ^ { - }$ satisfies $\mathbb { E } [ Z _ { \lambda } ^ { - } ] \le 1$ , and hence E[log $Z _ { \lambda } ^ { - } ] ~ \le ~ 0$ by Jensen’s inequality. The corrections $Q _ { n } ( \lambda ) , M _ { n } \beta ( \stackrel { . . . } { u } _ { n } )$ , and $T _ { n , u }$ are unchanged. It follows that

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathbb { E } _ { \theta \sim \rho } \displaystyle \sum _ { j = 1 } ^ { M _ { n } - 1 } L _ { j , \theta } \right] \leq \mathbb { E } [ D _ { n } ] \mathbb { E } _ { \theta \sim \rho } R ( \theta ) } \\ & { \qquad + \displaystyle \frac { n } { \lambda } \left\{ \mathrm { K L } ( \rho | | \mu ) + \mathbb { E } [ Q _ { n } ( \lambda ) ] \right\} + \mathbb { E } [ M _ { n } ] \beta ( u _ { n } ) + \mathbb { E } [ T _ { n , u } ] . } \end{array}\tag{8.17}
$$

Taking expectations in (8.16) and using (8.14) and (8.17) gives

$$
\begin{array} { r l } & { \mathbb E \left[ D _ { n } \mathbb E _ { \theta \sim \widehat { \rho } _ { \widehat { \lambda } } } R ( \theta ) \right] } \\ & { \leq \mathbb E [ D _ { n } ] \mathbb E _ { \theta \sim \rho } R ( \theta ) + \displaystyle \frac { n } { \lambda } \left\{ 2 \mathrm { K L } ( \rho \| \mu ) + p _ { n } ( \lambda ) + 2 \mathbb E [ Q _ { n } ( \lambda ) ] \right\} } \\ & { \quad + \ 2 \mathbb E [ M _ { n } ] \beta ( u _ { n } ) + 2 \mathbb E [ T _ { n , u } ] + \displaystyle \frac { n } { \lambda _ { + } } . } \end{array}\tag{8.18}
$$

Finally, put $X _ { n } = \mathbb { E } _ { \theta \sim \widehat { \rho } _ { \widehat { \lambda } } } R ( \theta )$ . Since $0 \leq X _ { n } \leq 1$ and $0 \leq D _ { n } \leq n .$

$$
\mathbb { E } [ X _ { n } ] = { \frac { 1 } { n } } \mathbb { E } [ D _ { n } X _ { n } ] + \mathbb { E } \left[ \left( 1 - { \frac { D _ { n } } { n } } \right) X _ { n } \right] \leq { \frac { \mathbb { E } [ D _ { n } X _ { n } ] } { \mathbb { E } [ D _ { n } ] } } + { \frac { n - \mathbb { E } [ D _ { n } ] } { n } } .
$$

Substitute (8.18) and take the infimum over fixed $\rho \ll \mu$ and $\lambda \in \Lambda _ { n }$ . This proves (5.4).

## 8.8 Proof of Corollary 5.1

ProofofCorollary 5.1. Every grid point satisfies $\lambda u _ { n } / n \leq 1 / 2$ , so

$$
Q _ { n } ( \lambda ) \leq \frac { \lambda ^ { 2 } } { n ^ { 2 } } H _ { n , u } + \frac { 2 \lambda u _ { n } } { n } ,
$$

using $q ( x ) \leq x ^ { 2 }$ and $- \log ( 1 - x ) \leq 2 x$ on $[ 0 , 1 / 2 ]$ . The stopped-sum bounds (8.4) and (8.5) give

$$
\mathbb { E } [ H _ { n , u } ] \leq \mathbb { E } [ M _ { n } ] v ( u _ { n } ) , \qquad \mathbb { E } [ T _ { n , u } ] \leq \mathbb { E } [ M _ { n } ] \beta ( u _ { n } ) .
$$

Moreover, for the grid in (5.7),

$$
p _ { n } ( \lambda ) \leq \log K _ { n } + ( K _ { n } - 1 ) \log 2 \leq \log ( 2 n K _ { n } ) , \qquad { \frac { 1 } { \lambda _ { + } } } = { \frac { 2 u _ { n } } { n } } .
$$

Apply Theorem 5.2 and (5.6). With

$$
b _ { n } : = \frac { 2 r _ { n } v ( u _ { n } ) } { n } ,
$$

we obtain

$$
\begin{array} { r l } & { \mathbb E \left[ \mathbb E _ { \theta \sim \widehat { \rho _ { \widehat \lambda } } } R ( \theta ) \right] - R _ { * } } \\ & { \leq \frac 1 n + \frac { n } { \mathbb E [ D _ { n } ] } \left[ \operatorname* { m i n } _ { \lambda \in \Lambda _ { n } } \left\{ \frac { a _ { n } } { \lambda } + b _ { n } \lambda \right\} + \frac { 6 u _ { n } } { n } + 4 r _ { n } \beta ( u _ { n } ) \right] + \frac { n - \mathbb E [ D _ { n } ] } { n } . } \end{array}\tag{8.19}
$$

The term $6 u _ { n } / n$ consists of $4 u _ { n } / n$ from the two crossing-cycle corrections and $2 u _ { n } / n$ from the grid expectation remainder.

We next bound the minimum over the grid. If $b _ { n } = 0$ , choose $\lambda = \lambda _ { + }$ . Otherwise, the unconstrained minimizer is $\lambda _ { * } = \sqrt { a _ { n } / b _ { n } }$ . Since $r _ { n } \le 2 , v ( u _ { n } ) \le u _ { n } ^ { 2 }$ , and $a _ { n } > 1$

$$
\lambda _ { * } \geq { \frac { \sqrt { a _ { n } n } } { 2 u _ { n } } } \geq { \frac { 1 } { 2 u _ { n } } } \geq \operatorname* { m i n } \Lambda _ { n } .
$$

If $\lambda _ { * } \leq \lambda _ { + }$ , choose the grid point nearest to $\lambda _ { * }$ on the logarithmic scale. Adjacent grid points differ by a factor of two, so

$$
2 ^ { - 1 / 2 } \leq \frac { \lambda } { \lambda _ { * } } \leq 2 ^ { 1 / 2 } , \qquad \frac { a _ { n } } { \lambda } + b _ { n } \lambda \leq \frac { 3 } { \sqrt { 2 } } \sqrt { a _ { n } b _ { n } } .
$$

If $\lambda _ { * } > \lambda _ { + }$ , choose $\lambda = \lambda _ { + }$ and use $b _ { n } \lambda _ { + } \leq { \sqrt { a _ { n } b _ { n } } }$ . In both cases, and also when $b _ { n } = 0$

$$
\operatorname* { m i n } _ { \lambda \in \Lambda _ { n } } \left\{ \frac { a _ { n } } { \lambda } + b _ { n } \lambda \right\} \leq \frac { 3 } { \sqrt { 2 } } \sqrt { a _ { n } b _ { n } } + \frac { a _ { n } } { \lambda _ { + } } \leq 3 \sqrt { \frac { r _ { n } v _ { 2 } a _ { n } } { n } } + \frac { 2 a _ { n } u _ { n } } { n } .
$$

Only this comparison value uses the unknown quantities $a _ { n } , b _ { n } ;$ ; the implemented choice remains the empirical minimizer $\widehat { \lambda }$

Corollary 4.1 supplies

$$
\beta ( u _ { n } ) \le C _ { \mathrm { e } } e ^ { - \kappa u _ { n } } \left( u _ { n } + \frac 1 { 1 - e ^ { - \kappa } } \right) , \qquad n - \mathbb { E } [ D _ { n } ] \le \frac { C _ { \mathrm { e } } e ^ { - \kappa } } { ( 1 - e ^ { - \kappa } ) ^ { 2 } } .
$$

Insert these inequalities and the grid bound into (8.19) to obtain (5.8).

For fixed c, d and a fixed regenerative law, $u _ { n } = O ( ( \log n ) ^ { 2 } ) , K _ { n } = O ( \log n )$ , and $a _ { n } = O ( d \log n )$ Also $n / \mathbb { E } [ D _ { n } ] = 1 + O ( n ^ { - 1 } ) , r _ { n } \to 1 / m$ , and $v _ { 2 } < \infty$ . Finally,

$$
e ^ { - \kappa u _ { n } } \leq e ^ { - \kappa ( \log n ) ^ { 2 } } = n ^ { - \kappa \log n } ,
$$

so $\beta ( u _ { n } ) = o ( n ^ { - a } )$ for every fixed $a > 0$ . The square-root term has order ${ \sqrt { d \log n / n } }$ , the truncation and grid corrections have order $d ( \log n ) ^ { 3 } / n$ , and the renewal-age term is $O ( n ^ { - 1 } )$ . This proves (5.9) and the stated second-order property. □

## 8.9 Proof of Proposition 6.1

ProofofProposition 6.1. We first control the number and lengths of the regeneration cycles. These bounds give an event on which $D _ { n } \geq n / 2$ . On that event, we estimate each term of Theorem 3.1.

The moment assumption and $C \geq 1$ imply

$$
\mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] \leq C m ^ { 2 } , \qquad \beta ( u _ { n } ) \leq \frac { C m ^ { 2 + \varepsilon } } { u _ { n } ^ { 1 + \varepsilon } } .
$$

Since $M _ { n }$ is a bounded stopping time, Wald’s identity [22, Chapter 10, Section 14.2] gives

$$
\mathbb { E } \left[ \sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } \right] = m \mathbb { E } [ M _ { n } ] ,
$$

$$
\mathbb { E } \left[ \sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } ^ { 2 } \right] = \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] \mathbb { E } [ M _ { n } ] .
$$

Every cycle before $M _ { n }$ ends by time n, whereas cycle $M _ { n }$ ends after time n. Thus

$$
\sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } = D _ { n } + \Delta _ { M _ { n } } \leq n + \Delta _ { M _ { n } } , \qquad \Delta _ { M _ { n } } ^ { 2 } \leq \sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } ^ { 2 } .
$$

Cauchy–Schwarz consequently gives

$$
\begin{array} { r l } & { m \mathbb { E } [ M _ { n } ] \leq n + \sqrt { \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] \mathbb { E } [ M _ { n } ] } } \\ & { \qquad \leq n + \displaystyle \frac { m } { 2 } \mathbb { E } [ M _ { n } ] + \frac { \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] } { 2 m } . } \end{array}
$$

Rearranging and using $n \geq 4 8 C m / \delta \geq C m$ , we obtain

$$
\mathbb { E } [ M _ { n } ] \leq \frac { 2 n } { m } + \frac { \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] } { m ^ { 2 } } \leq \frac { 3 n } { m } .
$$

The same stopping-time identity now yields

$$
\begin{array} { c } { { \displaystyle \mathbb { E } \left[ \sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } ^ { 2 } \right] \le 3 C m n , } } \\ { { \displaystyle \mathbb { E } \left[ \sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } \mathbf { 1 } _ { \left\{ \Delta _ { j } > u _ { n } \right\} } \right] \le \frac { 3 n \beta ( u _ { n } ) } { m } . } } \end{array}
$$

Apply Markov’s inequality separately to the three quantities. Each resulting bound holds with probability at least $1 - \delta / 4$ . A union bound gives an event of probability at least $1 - 3 \delta / 4$ on which

$$
\begin{array} { c } { { \displaystyle M _ { n } \leq \frac { 1 2 n } { m \delta } , } } \\ { { \displaystyle \sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } ^ { 2 } \leq \frac { 1 2 C m n } { \delta } , } } \\ { { \displaystyle \sum _ { j = 1 } ^ { M _ { n } } \Delta _ { j } { \bf 1 } _ { \{ \Delta _ { j } > u _ { n } \} } \leq \frac { 1 2 n \beta ( u _ { n } ) } { m \delta } . } } \end{array}\tag{8.20}
$$

If $\beta ( u _ { n } ) = 0$ , the third sum is zero almost surely. On this event, the length of cycle $M _ { n }$ satisfies

$$
\Delta _ { M _ { n } } ^ { 2 } \leq \frac { 1 2 C m n } { \delta } \leq \frac { n ^ { 2 } } { 4 } .
$$

Since cycle $M _ { n }$ ends after time $n ,$

$$
D _ { n } + \Delta _ { M _ { n } } > n , ~ D _ { n } \geq { \frac { n } { 2 } } .
$$

We now apply Theorem 3.1 with $\delta$ replaced by $\delta / 4$ . Its event and the bounds in (8.20) hold together with probability at least $1 - \delta$ . We estimate the terms on this intersection. The first term is bounded by

$$
\frac { n } { \lambda D _ { n } } \log \frac { 8 } { \delta } \leq \frac { 2 } { \lambda } \log \frac { 8 } { \delta } .
$$

For the block count and quadratic terms, use log $( 1 + x ) \leq x$ and $q ( x ) \leq x ^ { 2 }$ to obtain

$$
\begin{array} { r } { \displaystyle \frac { n M _ { n } } { \lambda D _ { n } } \log \left( 1 + \frac { \lambda \beta ( u _ { n } ) } { n } \right) \leq \frac { 2 4 \beta ( u _ { n } ) } { m \delta } , } \\ { \displaystyle \frac { n } { \lambda D _ { n } } \sum _ { j = 1 } ^ { M _ { n } - 1 } \mathbf { 1 } _ { \{ \Delta _ { j } \leq u _ { n } \} } q \left( \frac { \lambda \Delta _ { j } } { n } \right) \leq \frac { 2 4 C \lambda m } { \delta n } . } \end{array}
$$

Finally, we bound the correction for this additional cycle and the losses discarded by truncation:

$$
- \frac { n } { \lambda D _ { n } } \log \left( 1 - \frac { \lambda u _ { n } } { n } \right) \leq \frac { 4 u _ { n } } { n } ,
$$

$$
\frac { 1 } { D _ { n } } \sum _ { j = 1 } ^ { M _ { n } - 1 } \Delta _ { j } \mathbf { 1 } _ { \left\{ \Delta _ { j } > u _ { n } \right\} } \leq \frac { 2 4 \beta ( u _ { n } ) } { m \delta } .
$$

Here the logarithmic and quadratic estimates are valid because $\lambda u _ { n } / n \le 1 / 2$ . The chosen truncation level gives

$$
{ \frac { u _ { n } } { n } } = \left( { \frac { m } { n } } \right) ^ { ( 1 + \varepsilon ) / ( 2 + \varepsilon ) } , \qquad { \frac { \beta ( u _ { n } ) } { m } } \leq C \left( { \frac { m } { n } } \right) ^ { ( 1 + \varepsilon ) / ( 2 + \varepsilon ) } .
$$

Adding the five estimates proves

$$
\begin{array} { r l r } {  { \Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | \le \frac { 2 } { \lambda } \log \frac { 8 } { \delta } + \frac { 2 4 C \lambda m } { \delta n } } } \\ & { } & { + ( \frac { 4 8 C } { \delta } + 4 ) ( \frac { m } { n } ) ^ { ( 1 + \varepsilon ) / ( 2 + \varepsilon ) } . } \end{array}
$$

This is (6.2).

□

## 8.10 Proof of Theorem 6.1

Theorem 6.1. We obtain the upper bound by choosing λ in Proposition 6.1. For the lower bound, we use deterministic cycle lengths and Bernoulli losses that are constant within each cycle. We then compare both bounds over $\mathfrak { P } ( m , C , \varepsilon )$

Upper bound. Fix $\theta \in \Theta$ and choose the deterministic value

$$
\lambda : = \operatorname* { m i n } \left\{ \sqrt { \frac { \delta n } { 1 2 C m } \log \frac { 8 } { \delta } } , \frac { n } { 2 u _ { n } } \right\} .
$$

When the first expression is at most $n / ( 2 u _ { n } )$ , the first two terms of (6.2) have sum

$$
8 \sqrt { \frac { 3 C m } { \delta n } \log \frac { 8 } { \delta } } .
$$

Otherwise, $\lambda = n / ( 2 u _ { n } )$ , and the reciprocal term equals

$$
{ \frac { 4 u _ { n } } { n } } \log { \frac { 8 } { \delta } } ,
$$

while the linear term is no larger than at the unconstrained minimizer. Consequently, this choice satisfies

$$
\frac { 2 } { \lambda } \log \frac 8 \delta + \frac { 2 4 C \lambda m } { \delta n } \leq 8 \sqrt { \frac { 3 C m } { \delta n } \log \frac 8 \delta } + \frac { 4 u _ { n } } { n } \log \frac 8 \delta .
$$

Since $n \geq m$ and $( 1 + \varepsilon ) / ( 2 + \varepsilon ) > 1 / 2$

$$
{ \frac { u _ { n } } { n } } = \left( { \frac { m } { n } } \right) ^ { ( 1 + \varepsilon ) / ( 2 + \varepsilon ) } \leq { \sqrt { \frac { m } { n } } } .
$$

Adding the remaining term of (6.2) therefore gives

$$
\begin{array} { l } { \displaystyle \Big | R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \Big | \leq \left[ 8 \sqrt { \frac { 3 C } { \delta } \log \frac { 8 } { \delta } } + 4 \log \frac { 8 } { \delta } + \frac { 4 8 C } { \delta } + 4 \right] \sqrt { \frac { m } { n } } } \\ { \displaystyle = C _ { C , \delta } \sqrt { \frac { m } { n } } . } \end{array}\tag{8.21}
$$

This inequality holds on the event from Proposition 6.1, which has probability at least $1 - \delta$ and satisfies $D _ { n } \geq n / 2$

Lower bound. Suppose that $m \in \mathbb { N }$ and $n \geq m$ . Fix $p \in [ 0 , 1 ]$ and restrict to cycles of deterministic length $\Delta _ { j } = m$ . In cycle $j ,$ , let all observations share an independent Bernoulli label:

$$
Y _ { t } = U _ { j } \quad \mathrm { f o r } \tau _ { j } < t \leq \tau _ { j + 1 } , \qquad U _ { j } \sim \mathrm { B e r n o u l l i } ( p ) .
$$

Denote this regenerative law by $P _ { p }$ . Its risk and length moment are

$$
R _ { P _ { p } } ( \theta _ { 0 } ) = p , \qquad \mathbb { E } _ { A , P _ { p } } [ \Delta _ { 1 } ^ { 2 + \varepsilon } ] = m ^ { 2 + \varepsilon } \leq C m ^ { 2 + \varepsilon } ,
$$

so it belongs to $\mathfrak { P } ( m , C , \varepsilon )$ . The observed path, including its possibly incomplete last cycle, determines exactly

$$
k _ { n } : = \left\lceil { \frac { n } { m } } \right\rceil
$$

independent cycle labels. Conversely, these labels determine the path; the regeneration boundaries are deterministic. Thus estimating the long-run risk in this submodel is equivalent to estimating $p$ from $k _ { n }$ independent Bernoulli observations.

Choose two possible values of $p \mathrm { : }$

$$
h _ { n } : = \frac { 1 } { 1 6 \sqrt { k _ { n } } } , \qquad p _ { - } : = \frac 1 2 - h _ { n } , \qquad p _ { + } : = \frac 1 2 + h _ { n } .
$$

Let $P ^ { ( n ) }$ and $P _ { + } ^ { ( n ) }$ denote their laws on the observed sample and regeneration boundaries. The reduction to independent Bernoulli observations gives

$$
\begin{array} { l } { { \displaystyle \mathrm { K L } ( P _ { - } ^ { ( n ) } \| P _ { + } ^ { ( n ) } ) = k _ { n } \mathrm { K L } \big ( \mathrm { B e r n o u l l i } ( p _ { - } ) \| \mathrm { B e r n o u l l i } ( p _ { + } ) \big ) } } \\ { ~ = 2 k _ { n } h _ { n } \log \left( \frac { 1 + 2 h _ { n } } { 1 - 2 h _ { n } } \right) . } \end{array}
$$

Using

$$
\log \left( { \frac { 1 + x } { 1 - x } } \right) \leq { \frac { 2 x } { 1 - x } } , \qquad 0 \leq x < 1 ,
$$

and $h _ { n } \leq 1 / 1 6$ , we obtain

$$
\mathrm { K L } ( P _ { - } ^ { ( n ) } \| P _ { + } ^ { ( n ) } ) \leq \frac { 8 k _ { n } h _ { n } ^ { 2 } } { 1 - 2 h _ { n } } \leq 3 2 k _ { n } h _ { n } ^ { 2 } = \frac { 1 } { 8 } .
$$

Pinsker’s inequality consequently yields

$$
\mathrm { T V } ( P _ { - } ^ { ( n ) } , P _ { + } ^ { ( n ) } ) \leq \sqrt { \frac { 1 } { 2 } \mathrm { K L } ( P _ { - } ^ { ( n ) } \| P _ { + } ^ { ( n ) } ) } \leq \frac { 1 } { 4 } .
$$

To turn this testing bound into an estimation bound, let any estimator ${ \widetilde { R } } _ { n }$ define the test

$$
\varphi _ { n } : = \mathbf { 1 } _ { \{ \widetilde { R } _ { n } \geq 1 / 2 \} } .
$$

A testing error implies an estimation error of at least $h _ { n } .$ , since

$$
\begin{array} { r } { \{ \varphi _ { n } = 1 \} \subseteq \{ | \widetilde { R } _ { n } - p _ { - } | \geq h _ { n } \} , } \end{array}
$$

$$
\{ \varphi _ { n } = 0 \} \subseteq \{ | { \widetilde { R } } _ { n } - p _ { + } | \geq h _ { n } \} .
$$

It follows that

$$
\begin{array} { r l } & { \underset { \sigma \in \{ - , + \} } { \operatorname* { m a x } } P _ { \sigma } ^ { ( n ) } ( | \widetilde { R } _ { n } - p _ { \sigma } | \geq h _ { n } ) \geq \frac { 1 } { 2 } \left\{ P _ { - } ^ { ( n ) } ( \varphi _ { n } = 1 ) + P _ { + } ^ { ( n ) } ( \varphi _ { n } = 0 ) \right\} } \\ & { \qquad \quad \geq \frac { 1 - \mathrm { T V } ( P _ { - } ^ { ( n ) } , P _ { + } ^ { ( n ) } ) } { 2 } } \\ & { \qquad \quad \geq \frac { 3 } { 8 } . } \end{array}
$$

For $n \geq m$

$$
k _ { n } \leq \frac { 2 n } { m } , \qquad h _ { n } \geq \frac { 1 } { 1 6 \sqrt { 2 } } \sqrt { \frac { m } { n } } .
$$

The two laws lie in the model class, and the estimator was arbitrary. Hence

$$
\operatorname* { i n f } _ { \widetilde { R } _ { n } } \operatorname* { s u p } _ { P \in \mathfrak { P } ( m , C , \varepsilon ) } \mathbb { P } _ { A , P } \left( | \widetilde { R } _ { n } - R _ { P } ( \theta _ { 0 } ) | \geq \frac { 1 } { 1 6 \sqrt { 2 } } \sqrt { \frac { m } { n } } \right) \geq \frac { 3 } { 8 } .\tag{8.22}
$$

Minimax bounds. Apply (8.21) with $\theta = \theta _ { 0 }$ to each law in $\mathfrak { P } ( m , C , \varepsilon )$ . Its constants are the same for every law in this class. Setting the empirical risk to $1 / 2$ when $D _ { n } = 0$ does not affect the event where $D _ { n } \geq n / 2 .$ so

$$
\operatorname* { s u p } _ { P \in \mathfrak { P } ( m , C , \varepsilon ) } \mathbb { P } _ { A , P } \left( | \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta _ { 0 } ) - R _ { P } ( \theta _ { 0 } ) | > C _ { C , \delta } \sqrt { \frac { m } { n } } \right) \leq \delta .
$$

This proves the upper inequality in (6.3). For the lower inequality, every threshold satisfying

$$
0 \leq r < \frac { 1 } { 1 6 \sqrt { 2 } } \sqrt { \frac { m } { n } }
$$

also satisfies, by (8.22),

$$
\operatorname* { i n f } _ { \widetilde { R } _ { n } } \operatorname* { s u p } _ { P \in \mathfrak { P } ( m , C , \varepsilon ) } \mathbb { P } _ { A , P } ( | \widetilde { R } _ { n } - R _ { P } ( \theta _ { 0 } ) | > r ) \ge \frac { 3 } { 8 } > \delta .
$$

Such a threshold cannot satisfy the probability requirement in the definition of $r _ { n , \delta } ( m , C , \varepsilon )$ . Hence the lower inequality in (6.3) follows. □

## 8.11 Proof of Theorem 7.1

ProofofTheorem 7.1. We first couple the centered sums with Brownian motion. We then use the Bernstein inequality to control empirical centering, prove the variance estimator’s convergence rate, and show that replacing the true variance by its estimate preserves the approximation. All limits below are almost sure unless stated otherwise. Recall that

$$
p = 2 + \varepsilon , \qquad N _ { n } = M _ { n } - 1 .
$$

We use the strong law, the Marcinkiewicz–Zygmund strong law, and the law of the iterated logarithm in the forms given in [22, Sections 6.6, 6.7 and 8.1].

Observe by Tonelli’s theorem that for every $b > 0$

$$
\begin{array} { r l r } {  { \sum _ { j = 1 } ^ { \infty } \mathbb { P } _ { A } ( \Delta _ { j } > b j ^ { 1 / p } ) = \mathbb { E } _ { A } [ \sum _ { j = 1 } ^ { \infty } \mathbf { 1 } _ { \{ \Delta _ { 1 } ^ { p } > b ^ { p } j \} } ] } } \\ & { } & { \leq b ^ { - p } \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { p } ] < \infty . } \end{array}
$$

Then, by Borel–Cantelli lemma, applied to positive rational values of b, implies

$$
\Delta _ { j } = o ( j ^ { 1 / p } ) .
$$

Taking a maximum over a fixed initial segment and its complement, and using $M _ { n } \leq n + 1$ , therefore yields

$$
\operatorname* { m a x } _ { j \leq n + 1 } \Delta _ { j } = o ( n ^ { 1 / p } ) , \qquad 0 \leq n - D _ { n } < \Delta _ { M _ { n } } = o ( n ^ { 1 / p } ) .\tag{8.23}
$$

Further recall the inequalities

$$
D _ { n } \leq n < D _ { n } + \Delta _ { M _ { n } } ;
$$

which, along with the strong law of large numbers

$$
\frac { N _ { n } } { n } \longrightarrow \frac { 1 } { m } , \qquad \frac { D _ { n } } { n } \longrightarrow 1 .
$$

We next check the hypotheses of Bashtova and Shashkin [10, Theorem 1]. Extend $S _ { n , \theta } ^ { \mathrm { f u l l } }$ to continuous time by

$$
A _ { \theta } ( t ) : = S _ { \lfloor t \rfloor , \theta } ^ { \mathrm { f u l l } } , \qquad t \geq 0 .
$$

This step process is right-continuous and separable. Its cycle paths and lengths are i.i.d., since each is a measurable function of one regenerative block. The cycle reward has mean zero, and the absolute increment at any time within a cycle is bounded by its length:

$$
\mathbb { E } _ { A } [ \xi _ { 1 , \theta } ] = 0 , \qquad \operatorname* { s u p } _ { 0 \leq t \leq \Delta _ { j } } | A _ { \theta } ( \tau _ { j } + t ) - A _ { \theta } ( \tau _ { j } ) | \leq \Delta _ { j } .
$$

This verifies both moment hypotheses of Bashtova and Shashkin [10, Theorem 1, p. 2]. The variance per unit time in that theorem is

$$
\frac { \mathbb { E } _ { A } [ \xi _ { 1 , \theta } ^ { 2 } ] } { m } = \sigma _ { \theta } ^ { 2 } .
$$

It follows that the full-path sum admits a Brownian coupling satisfying

$$
\operatorname* { m a x } _ { 0 \le k \le n } \left| S _ { k , \theta } ^ { \mathrm { f u l l } } - \sigma _ { \theta } W _ { \theta } ( k ) \right| = o ( n ^ { 1 / p } ) .
$$

We retain the joint law of the loss paths and regeneration times in this realization. Observe that removing the incomplete cycle changes the sum by at most the longest cycle, because

$$
S _ { k , \theta } ^ { \mathrm { r e g } } = S _ { D _ { k } , \theta } ^ { \mathrm { f u l l } } .
$$

Consequently,

$$
\operatorname* { m a x } _ { 0 \leq k \leq n } \left. S _ { k , \theta } ^ { \mathrm { f u l l } } - S _ { k , \theta } ^ { \mathrm { r e g } } \right. \leq \operatorname* { m a x } _ { j \leq n + 1 } \Delta _ { j } = o ( n ^ { 1 / p } ) .
$$

The triangle inequality proves (7.1) for both sums with the same Brownian motion.

The variance estimator centers each block at the empirical risk. We use Theorem 3.1 to control the difference between this empirical risk and $R ( \theta )$ , which will then bound the change in the quadratic sum. Keep the truncation level in the statement and choose

$$
u _ { n } = m ( n / m ) ^ { 1 / p } , \qquad \delta _ { n } = ( n + 2 ) ^ { - 2 } , \qquad \lambda _ { n } = \operatorname* { m i n } \left\{ \sqrt { \frac { n } { m } \log ( n + 2 ) } , \frac { n } { 2 u _ { n } } \right\} .
$$

These choices satisfy the admissibility condition for every n. Moreover,

$$
{ \sqrt { { \frac { n } { m } } \log ( n + 2 ) } } { \frac { u _ { n } } { n } } = { \sqrt { \log ( n + 2 ) } } \left( { \frac { m } { n } } \right) ^ { 1 / 2 - 1 / p } \longrightarrow 0 ,
$$

so the constraint on $\lambda _ { n }$ is eventually inactive.

For each $n ,$ apply inequality (3.1) in Theorem 3.1 with $\lambda = \lambda _ { n } , \delta = \delta _ { n } .$ and the truncation level $u _ { n }$ chosen above. The resulting bound holds with probability at least $1 - \delta _ { n }$ . Since $\textstyle \sum _ { n } \delta _ { n } < \infty$ , Borel–Cantell implies that the bound holds for all sufficiently large n, almost surely. Also, (8.23) implies

$$
\frac { \operatorname* { m a x } _ { j \le M _ { n } } \Delta _ { j } } { u _ { n } } \longrightarrow 0 .
$$

Hence all completed cycles eventually satisfy $\Delta _ { j } \leq u _ { n }$ . The truncation indicators in (3.1) then equal one, and its sum over longer completed cycles is zero. The first inequality in the calculation below is exactly the application of Theorem 3.1 with the stated parameters. The second inequality uses only

$$
\begin{array} { r l } {  { \Big | \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) - R ( \theta ) \Big | \overset { \mathrm { T h e o r e m 3 . 1 } } { \leq } \frac { n } { \lambda _ { n } D _ { n } } \Bigg [ \log \frac { 2 } { \delta _ { n } } + M _ { n } \log \Big ( 1 + \frac { \lambda _ { n } \beta ( u _ { n } ) } { n } \Big ) } \quad } & { } \\ & { \qquad + \displaystyle \sum _ { j = 1 } ^ { N _ { n } } q ( \frac { \lambda _ { n } \Delta _ { j } } { n } ) - \log ( 1 - \frac { \lambda _ { n } u _ { n } } { n } ) \bigg ] } \\ & { \leq \frac { n \log ( 2 / \delta _ { n } ) } { \lambda _ { n } D _ { n } } + \frac { M _ { n } } { D _ { n } } \beta ( u _ { n } ) } \\ & { \qquad + \frac { \lambda _ { n } } { n D _ { n } } \displaystyle \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } ^ { 2 } + \frac { 2 u _ { n } } { D _ { n } } . } \end{array}\tag{8.24}
$$

The strong law and the moment assumption give:

$$
\frac { M _ { n } } { D _ { n } } \longrightarrow \frac { 1 } { m } , \qquad \frac { 1 } { D _ { n } } \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } ^ { 2 } \longrightarrow \frac { \mathbb { E } _ { A } [ \Delta _ { 1 } ^ { 2 } ] } { m } \le C ^ { 2 / p } m ,
$$

and

$$
\log ( 2 / \delta _ { n } ) \leq 3 \log ( n + 2 ) , \qquad { \frac { \beta ( u _ { n } ) } { m } } \leq C \left( { \frac { m } { n } } \right) ^ { 1 - 1 / p } , \qquad { \frac { u _ { n } } { n } } = \left( { \frac { m } { n } } \right) ^ { 1 - 1 / p } .
$$

In particular, the following bounds hold eventually:

$$
D _ { n } \geq n / 2 , \qquad M _ { n } / D _ { n } \leq 2 / m , \qquad \frac { 1 } { D _ { n } } \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } ^ { 2 } \leq 2 C ^ { 2 / p } m .
$$

Substituting these bounds into the final inequality in (8.24) gives

$$
\begin{array} { r l } & { \Big | \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) - R ( \theta ) \Big | \leq ( 6 + 2 C ^ { 2 / p } ) \sqrt { \frac { m \log ( n + 2 ) } { n } } + ( 2 C + 4 ) \left( \frac { m } { n } \right) ^ { 1 - 1 / p } } \\ & { \qquad = O \left( \sqrt { \frac { \log ( n + 2 ) } { n } } \right) . } \end{array}\tag{8.25}
$$

Here the law is fixed and $1 - 1 / p > 1 / 2$ . We use the resulting almost-sure estimate (8.25) below to control the change in the quadratic sum caused by empirical centering.

First consider the quadratic sum centered at the true risk. Define

$$
Q _ { n , \theta } : = \sum _ { j = 1 } ^ { N _ { n } } \xi _ { j , \theta } ^ { 2 } , \qquad e _ { n , \theta } : = \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) - R ( \theta ) .
$$

The law of the iterated logarithm for the cycle lengths, evaluated at $N _ { n } \sim n / m$ , gives

$$
D _ { n } - m N _ { n } = O \Big ( \sqrt { n \log \log ( n + e ^ { e } ) } \Big ) , \qquad \frac { N _ { n } } { D _ { n } } - \frac { 1 } { m } = O \left( \sqrt { \frac { \log \log ( n + e ^ { e } ) } { n } } \right) .
$$

Both bounds also hold when the length variance is zero. For $2 < p < 4$ , the centered variables $\xi _ { j , \theta } ^ { 2 } - v _ { \theta } ^ { 2 }$ have finite $p / 2$ -moment, so the Marcinkiewicz–Zygmund strong law applies. For $p \geq 4 .$ , they have finite variance, so the law of the iterated logarithm applies instead. Together, these give

$$
Q _ { n , \theta } - v _ { \theta } ^ { 2 } N _ { n } = \left\{ \begin{array} { l l } { o ( n ^ { 2 / p } ) , } & { 2 < p < 4 , } \\ { O \left( \sqrt { n \log \log ( n + e ^ { e } ) } \right) , } & { p \geq 4 . } \end{array} \right.
$$

The exact decomposition

$$
\frac { Q _ { n , \theta } } { D _ { n } } - \sigma _ { \theta } ^ { 2 } = \frac { Q _ { n , \theta } - v _ { \theta } ^ { 2 } N _ { n } } { D _ { n } } + v _ { \theta } ^ { 2 } \left( \frac { N _ { n } } { D _ { n } } - \frac { 1 } { m } \right)
$$

therefore yields

$$
\frac { Q _ { n , \theta } } { D _ { n } } - \sigma _ { \theta } ^ { 2 } = \left\{ { O \big ( n ^ { 2 / p - 1 } \big ) , } \right. \ 2 < p < 4 ,
$$

In the first case, the renewal-count error is smaller than the stated rate because

$$
n ^ { 1 / 2 - 2 / p } \sqrt { \log \log ( n + e ^ { e } ) } \longrightarrow 0 .
$$

It remains to replace the true-risk centering by empirical centering. For each block,

$$
\Delta _ { j } \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) - L _ { j , \theta } = \xi _ { j , \theta } + \Delta _ { j } e _ { n , \theta } .
$$

Squaring and summing gives

$$
\widehat { Q } _ { n , \theta } - Q _ { n , \theta } = 2 e _ { n , \theta } \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } \xi _ { j , \theta } + e _ { n , \theta } ^ { 2 } \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } ^ { 2 } .
$$

The block bound and the strong law imply

$$
| \xi _ { j , \theta } | \leq \Delta _ { j } , \qquad \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } ^ { 2 } = O ( n ) .
$$

We now substitute the estimate obtained from Theorem 3.1 into this variance calculation. Since $e _ { n , \theta }$ is the empirical risk error, (8.25) says that, almost surely,

$$
| e _ { n , \theta } | = O \left( \sqrt { \frac { \log ( n + 2 ) } { n } } \right) , \qquad e _ { n , \theta } ^ { 2 } = O \left( \frac { \log ( n + 2 ) } { n } \right) .
$$

Inserting these two bounds and $\begin{array} { r } { \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } ^ { 2 } = O ( n ) } \end{array}$ into the preceding expansion gives

$$
\begin{array} { c l } { \displaystyle | \widehat { Q } _ { n , \theta } - Q _ { n , \theta } | \leq ( 2 | e _ { n , \theta } | + e _ { n , \theta } ^ { 2 } ) \sum _ { j = 1 } ^ { N _ { n } } \Delta _ { j } ^ { 2 } } \\ { \displaystyle { \stackrel { ( 8 , 2 5 ) } { = } O \left( n \sqrt { \frac { \log \left( n + 2 \right) } { n } } + \log \left( n + 2 \right) \right) } } \\ { \displaystyle { = O \left( \sqrt { n \log \left( n + 2 \right) } \right) . } } \end{array}
$$

Dividing by $D _ { n } \sim n$ proves (7.2). For $2 < p < 4$ , this extra error is negligible because

$$
n ^ { 1 / 2 - 2 / p } \sqrt { \log ( n + 2 ) } \longrightarrow 0 .
$$

In particular,

$$
\widehat { \sigma } _ { n , \theta } \longrightarrow \sigma _ { \theta } > 0 .
$$

The normalization error is the variance-estimation error multiplied by the maximum Brownian fluctuation. For $T > 0$ and $x > 0$ , the exponential inequality for Brownian motion [41, Chapter II, Proposition 1.8, p. 55], applied to $W _ { \theta }$ and $- W _ { \theta }$ , gives

$$
\mathbb { P } \left( \operatorname* { s u p } _ { 0 \leq t \leq T } | W _ { \theta } ( t ) | > x \right) \leq 4 e ^ { - x ^ { 2 } / ( 2 T ) } .
$$

Apply this at dyadic times with

$$
T = 2 ^ { r } , \qquad x = 3 { \sqrt { 2 ^ { r } \log r } } , \qquad r \in \{ 2 , 3 , \ldots \} .
$$

The resulting probabilities are summable. Borel–Cantelli and comparison with the nearest dyadic time imply

$$
{ \frac { \operatorname* { m a x } _ { 0 \leq k \leq n } | W _ { \theta } ( k ) | } { \sqrt { n } } } = O { \Bigl ( } { \sqrt { \log \log ( n + e ^ { e } ) } } { \Bigr ) } ~ .
$$

For either centered sum, the triangle inequality separates the coupling and variance-estimation errors:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { 0 \le k \le n } \left. \frac { S _ { k , \theta } ^ { H } } { \widehat { \sigma } _ { n , \theta } \sqrt { n } } - \frac { W _ { \theta } ( k ) } { \sqrt { n } } \right. } & { \leq \frac { \operatorname* { m a x } _ { 0 \le k \le n } | S _ { k , \theta } ^ { H } - \sigma _ { \theta } W _ { \theta } ( k ) | } { \widehat { \sigma } _ { n , \theta } \sqrt { n } } } \\ & { \qquad + \left. \frac { \sigma _ { \theta } } { \widehat { \sigma } _ { n , \theta } } - 1 \right. \frac { \operatorname* { m a x } _ { 0 \le k \le n } | W _ { \theta } ( k ) | } { \sqrt { n } } . } \end{array}
$$

The strong approximation (7.1) bounds the first term by $o ( n ^ { 1 / p - 1 / 2 } )$ . Positivity of the limiting variance and (7.2) bound the second term by

$$
\left\{ \begin{array} { l l } { o \big ( n ^ { 2 / p - 1 } \big ) O \left( \sqrt { \log \log ( n + e ^ { e } ) } \right) , } & { 2 < p < 4 , } \\ { O \left( n ^ { - 1 / 2 } \sqrt { \log ( n + 2 ) \log \log ( n + e ^ { e } ) } \right) , } & { p \geq 4 . } \end{array} \right.
$$

In both cases, division by $n ^ { 1 / p - 1 / 2 }$ makes the bound tend to zero. This proves (7.3).

Next replace the sample-size normalization by the observed quadratic sum. The relation

$$
\sqrt { \widehat { Q } _ { n , \theta } } = \widehat { \sigma } _ { n , \theta } \sqrt { D _ { n } }
$$

changes the normalized sum by the factor $\sqrt { n / D _ { n } }$ . By (8.23),

$$
\left| { \sqrt { \frac { n } { D _ { n } } } } - 1 \right| = { \frac { n - D _ { n } } { { \sqrt { D _ { n } } } ( { \sqrt { n } } + { \sqrt { D _ { n } } } ) } } = o ( n ^ { 1 / p - 1 } ) .
$$

The additional error is therefore bounded by

$$
\begin{array} { c l l } { \sqrt { \displaystyle \frac { n } { D _ { n } } } o ( n ^ { 1 / p - 1 / 2 } ) + \left| \sqrt { \displaystyle \frac { n } { D _ { n } } } - 1 \right| \frac { \operatorname* { m a x } _ { 0 \leq k \leq n } | W _ { \theta } ( k ) | } { \sqrt { n } } } \\ { = o ( n ^ { 1 / p - 1 / 2 } ) . } \end{array}
$$

Finally, pass from integer times to all $t \in [ 0 , 1 ]$ . The same exponential inequality on each unit interval and a union bound yield

$$
\begin{array} { r l } & { \mathbb P \left( \underset { 0 \leq k \leq n } { \operatorname* { m a x } } \underset { 0 \leq u \leq 1 } { \operatorname* { s u p } } | W _ { \theta } ( k + u ) - W _ { \theta } ( k ) | > 3 \sqrt { \log ( n + 2 ) } \right) } \\ & { \quad \quad \leq 4 ( n + 1 ) ( n + 2 ) ^ { - 9 / 2 } . } \end{array}
$$

Using Borel–Cantelli lemma then gives the uniform interpolation bound

$$
\operatorname* { s u p } _ { 0 \leq t \leq 1 } \frac { | W _ { \theta } ( \lfloor n t \rfloor ) - W _ { \theta } ( n t ) | } { \sqrt { n } } = O \left( \sqrt { \frac { \log ( n + 2 ) } { n } } \right) = o ( n ^ { 1 / p - 1 / 2 } ) .
$$

This proves (7.4). At t = 1, the identities

$$
S _ { n , \theta } ^ { \mathrm { r e g } } = D _ { n } \{ R ( \theta ) - \widehat { R } _ { n } ^ { \mathrm { r e g } } ( \theta ) \} , \qquad \frac { W _ { \theta } ( n ) } { \sqrt { n } } \sim { \mathcal N } ( 0 , 1 )
$$

give (7.5) by Slutsky’s theorem. Brownian scaling makes the comparison process

$$
t \longmapsto { \frac { W _ { \theta } ( n t ) } { \sqrt { n } } }
$$

a standard Brownian motion in distribution for each n. The vanishing uniform approximation error therefore also proves the stated convergence in $D [ 0 , 1 ]$ □

## References

[1] Baptiste Abeles, Eugenio Clerico, and Gergely Neu. Generalization bounds for mixing processes via delayed online-to-PAC conversions. arXiv preprint arXiv:2406.12600, 2024.

[2] Radosław Adamczak. A tail inequality for suprema of unbounded empirical processes with applications to Markov chains. Electronic Journal of Probability, 13:1000–1034, 2008. doi: 10.1214/EJP. v13-521.

[3] Pierre Alquier. PAC-Bayesian bounds for randomized empirical risk minimizers. Mathematical Methods ofStatistics, 17(4):279–304, 2008. doi: 10.3103/S1066530708040017.

[4] Pierre Alquier. User-friendly Introduction to PAC-Bayes Bounds. Foundations and Trends in Machine Learning, 17(2):174–303, jan 2024. ISSN 1935-8237. doi: 10.1561/2200000100. URL https: //doi.org/10.1561/2200000100.

[5] Pierre Alquier and Olivier Wintenberger. Model selection for weakly dependent time series forecasting. Bernoulli, 18(3):883 – 913, 2012.

[6] Francis Bach. Learning theory from first principles. MIT press Cambridge, 2024.

[7] Imon Banerjee and Sayak Chakrabarty. Small resamples, sharp guarantees: Convergence rates for resampled studentized quantile estimators. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 62434–62482. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-2090.

[8] Imon Banerjee, Vinayak A. Rao, and Harsha Honnappa. PAC-Bayes bounds on variational tempered posteriors for Markov models. Entropy, 23(3), 2021.

[9] Imon Banerjee, Jiaqi Lei, and Sanjay Mehrotra. Nonparametric multi change point detection for Markov chains via adaptive clustering. In Proceedings of the 29th International Confer ence on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 2224–2232. PMLR, 2026. URL https://proceedings.mlr.press/v300/ banerjee26c.html.

[10] Elena Bashtova and Alexey Shashkin. Strong Gaussian approximation for cumulative processes with heavy tails. arXiv:2007.15481v1 [math.PR], 2020. URL https://arxiv.org/abs/2007. 15481v1.

[11] Patrice Bertail and Stephan Cl´ emenc¸on. Regeneration-based statistics for harris recurrent markov´ chains. In Patrice Bertail, Philippe Soulier, and Paul Doukhan, editors, Dependence in Probability and Statistics, volume 187 of Lecture Notes in Statistics, pages 3–54. Springer, New York, NY, 2006. doi: 10.1007/0-387-36062-X 1.

[12] Patrice Bertail and Stephan Cl ´ emenc¸on. Sharp bounds for the tails of functionals of Markov chains.´ Theory ofProbability and Its Applications, 54(3):505–515, 2010. doi: 10.1137/S0040585X97984401.

[13] Patrice Bertail and Franc¸ois Portier. Rademacher complexity for Markov chains: Applications to kernel smoothing and Metropolis–Hastings. Bernoulli, 25(4B):3912–3938, 2019. doi: 10.3150/19-BEJ1115.

[14] S. Boucheron, G. Lugosi, and P. Massart. Concentration inequalities. Oxford University Press, 2013.

[15] Olivier Catoni. PAC-Bayesian supervised classification: The thermodynamics of statistical learning. Institute of Mathematical Statistics Lecture Notes – Monograph Series, 56. Institute of Mathematical Statistics, Beachwood, OH, 2007.

[16] Alain Durmus, Eric Moulines, Alexey Naumov, Sergey Samsonov, and Marina Sheshukova. Rosenthal-type inequalities for linear statistics of Markov chains. arXiv preprint arXiv:2303.05838, 2023.

[17] G. K. Dziugaite and D. M. Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2017.

[18] Deividas Eringis, John-Josef Leth, Zheng-Hua Tan, Rafal Wisniewski, Alireza Fakhrizadeh Esfahani, and Mihaly Petreczky. PAC-Bayesian theory for stochastic LTI systems. In 2021 60th IEEE Conference on Decision and Control (CDC), IEEE Conference on Decision and Control. Proceedings, pages 6626– 6633. IEEE (Institute of Electrical and Electronics Engineers), 2021.

[19] Mahdi Milani Fard, Joelle Pineau, and Csaba Szepesvari. PAC-Bayesian policy evaluation for rein-´ forcement learning. In Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence, pages 195–202, 2011.

[20] David Gamarnik. Extension of the PAC framework to finite and countable Markov chains. In Proceedings of the twelfth annual conference on Computational learning theory, pages 308–317, 1999.

[21] Steffen Grunew ¨ alder and Azadeh Khaleghi. Estimating the mixing coefficients of geometrically er-¨ godic Markov processes. arXiv:2402.07296, 2024.

[22] Allan Gut. Probability: a graduate course. Springer Texts in Statistics. Springer, 2005. doi: 10.1007/ b138932.

[23] Fredrik Hellstrom, Giuseppe Durisi, Benjamin Guedj, and Maxim Raginsky. Generalization bounds:¨ Perspectives from information theory and PAC-Bayes. Foundations and Trends in Machine Learning, 18(1):1–223, 2025.

[24] Daniel Jerison. A data-dependent DKW inequality for regenerative Markov chains. arXiv preprint arXiv:2606.30866v2, 2026.

[25] Vahe Karagulyan and Pierre Alquier. Empirical PAC-Bayes bounds for Markov chains. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 361–369. PMLR, 2026. URL https://proceedings. mlr.press/v300/karagulyan26a.html.

[26] Vahe Karagulyan, Mikołaj Kasprzak, and Pierre Alquier. Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds. Preprint arXiv:2610.20007, 2026.

[27] Azadeh Khaleghi and Gabor Lugosi. Inferring the mixing properties of a stationary ergodic process from a single sample-path. IEEE Transactions on Information Theory, 69(6):4014–4026, 2023.

[28] Vitaly Kuznetsov and Mehryar Mohri. Learning theory and algorithms for forecasting non-stationary time series. Advances in neural information processing systems, 28, 2015.

[29] Vitaly Kuznetsov and Mehryar Mohri. Generalization bounds for non-stationary mixing processes. Machine Learning, 106(1):93–117, 2017.

[30] Vitaly Kuznetsov and Mehryar Mohri. Discrepancy-based theory and algorithms for forecasting nonstationary time series. Annals ofMathematics and Artificial Intelligence, 88(4):367–399, 2020.

[31] Andreas Maurer and Massimiliano Pontil. Empirical Bernstein bounds and sample variance penalization. arXiv preprint arXiv:0907.3740, 2009.

[32] David A McAllester. Some PAC-Bayesian theorems. In Proceedings of the eleventh annual conference on Computational learning theory, pages 230–234, 1998.

[33] Daniel J McDonald, Cosma Rohilla Shalizi, and Mark Schervish. Nonparametric risk bounds for time-series forecasting. Journal of Machine Learning Research, 18(32):1–40, 2017.

[34] Ron Meir. Nonparametric time series prediction through adaptive model selection. Machine learning, 39(1):5–34, 2000.

[35] Sean P Meyn and Richard L Tweedie. Markov chains and stochastic stability. Springer Science & Business Media, 2012.

[36] Dharmendra S Modha and Elias Masry. Memory-universal prediction of stationary random processes. IEEE transactions on information theory, 44(1):117–133, 2002.

[37] Mehryar Mohri and Afshin Rostamizadeh. Stability bounds for stationary ϕ-mixing and β-mixing processes. Journal of Machine Learning Research, 11(26):789–814, 2010.

[38] Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning, volume 2. MIT press Cambridge, 2012.

[39] Daniel Paulin. Concentration inequalities for Markov chains by Marton couplings and spectral methods. Electronic Journal ofProbability, 20(none):1 – 32, 2015.

[40] Liva Ralaivola, Marie Szafranski, and Guillaume Stempfel. Chromatic PAC-Bayes bounds for non-iid data: Applications to ranking and stationary β-mixing processes. The Journal of Machine Learning Research, 11:1927–1956, 2010.

[41] Daniel Revuz and Marc Yor. Continuous Martingales and Brownian Motion, volume 293 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 3 edition, 1999. doi: 10.1007/978-3-662-06400-9.

[42] Paul-Marie Samson. Concentration of measure inequalities for Markov chains and ϕ-mixing processes. The Annals of Probability, 28(1):416–461, 2000.

[43] Cosma Shalizi and Aryeh Kontorovich. Predictive PAC learning and process decompositions. Advances in neural information processing systems, 26, 2013.

[44] Ingo Steinwart, Don Hush, and Clint Scovel. Learning from dependent observations. Journal of Multivariate Analysis, 100(1):175–194, 2009.

[45] Ilya O. Tolstikhin and Yevgeny Seldin. PAC-Bayes-empirical-Bernstein inequality. In Advances in Neural Information Processing Systems, volume 26, pages 109–117, 2013.

[46] Olivier Wintenberger. Exponential inequalities for unbounded functions of geometrically ergodic Markov chains: Applications to quantitative error bounds for regenerative Metropolis algorithms. Statistics, 51(1):222–234, 2017. doi: 10.1080/02331888.2016.1268205.

[47] Geoffrey Wolfer and Pierre Alquier. Optimistic estimation of convergence in Markov chains with the average-mixing time. arXiv preprint arXiv:2402.10506, 2024.

[48] Geoffrey Wolfer and Aryeh Kontorovich. Improved estimation of relaxation time in nonreversible Markov chains. Annals ofApplied Probability, 34(1A):249–276, 2024.

[49] Senhan Yao. Strongly consistent estimation of the extended ℓ<sub>1</sub>-sum of ϕ-mixing coefficients from a single trajectory. arXiv preprint arXiv:2609.26232, 2026.

[50] Bin Yu. Rates of convergence for empirical processes of stationary β-mixing sequences. The Annals of Statistics, 22(1):94–116, 1994.