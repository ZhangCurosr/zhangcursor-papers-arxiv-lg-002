# Nonpreemptive Scheduling While Learning Context-Dependent Service Rates

Wansoo Choi<sup>1</sup> Seoungbin Bae<sup>2</sup> Dabeen Lee<sup>1</sup>

<sup>1</sup>Department of Mathematical Sciences, Seoul National University <sup>2</sup>Department of Industrial & Systems Engineering, KAIST choiws0624@snu.ac.kr, sbbae31@kaist.ac.kr, dabeenl@snu.ac.kr

## Abstract

We study nonpreemptive contextual queueing bandits in a single-server system. Each job is represented by a d-dimensional context vector; in each round, a job may arrive with its context drawn from an unknown distribution D, and its departure probability is determined by a logistic model of that context vector with an unknown parameter θ<sup>∗</sup>. The server learns from service outcomes while deciding which waiting job to serve and whether to idle, aiming to minimize queuelength regret, the gap between its expected terminal queue length and the minimum achievable by an admissible policy. Once selected, a job must be served until completion, and we refer to this as the nonpreemptive setting. A central challenge is that, even with full model knowledge, the optimal policy cannot in general be characterized by a simple myopic rule, since the optimal action can change with the remaining horizon at the same queue state. Nevertheless, when the model and horizon are known, the optimal action can be obtained through a finite-horizon Bellman recursion. Motivated by this, we propose Learn–Clear–Plan (LCP), which estimates the system and uses the resulting Bellman recursion to make horizon-dependent decisions. LCP achieves $\widetilde { O } ( \sqrt { d / T } )$ queue-length regret, while a lower-bound construction gives $\Omega ( \operatorname* { m i n } \{ 1 / \sqrt { d } , \sqrt { d / T } \} )$ regret for every learning policy on some instance, establishing optimality up to polylogarithmic factors when $T \geq d ^ { 2 }$ . When the horizon is unknown, no horizon-independent policy achieves vanishing regret against the finite-horizon optimum. We therefore use SEPT, the policy that serves a waiting job with the highest probability of departure, as a fixed reference, and suggest an estimated-SEPT algorithm that achieves a tracking error of $\widetilde O ( \sqrt { d / t } )$ without knowing the model.

## 1 Introduction

Queueing systems are central to cloud computing, online service platforms, communication networks, and other applications in which heterogeneous jobs compete for limited service capacity (Neely, 2010). In many such systems, a job’s completion probability depends on observable features, while the corresponding service model is initially unknown. A scheduler must therefore learn context-dependent service rates from departure feedback while simultaneously controlling congestion. Queueing bandits formalize this learning-while-scheduling problem, and queue-length regret measures the excess number of unfinished jobs relative to an optimal policy with full knowledge of the service model (Krishnasamy et al., 2021; Stahlbuhk et al., 2021).

Contextual queueing bandits extend this framework to job-specific features. Under the assumption that the contexts of arriving jobs are drawn independently from a fixed distribution, Bae et al. (2026) introduce a preemptive contextual queueing bandit model, in which the scheduler may reconsider the entire queue and select a diferent job–server pair in the next round after an unsuccessful service attempt. They obtain $\widetilde { O } ( T ^ { - 1 / 4 } )$ queue-length regret in this setting. For the same stochastic-context setting, Bae and Lee (2026) subsequently propose the $\mathrm { C Q B - } \eta { - } 2$ algorithm and improve the upper bound to $\widetilde { O } ( T ^ { - 1 / 2 } )$ . They further show that every learning algorithm incurs $\Omega ( T ^ { - 1 / 2 } )$ regret on some instance, showing that this dependence on T is optimal up to logarithmic factors.

Table 1: Summary of main results. Known/unknown indicates whether the server knows the horizon; IA/WC denote the idling-allowed/work-conserving settings, and SEPT denotes the shortestexpected-processing-time policy. Logarithmic factors and fixed model constants are suppressed.
<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Benchmark / Objective</td><td rowspan=1 colspan=1>Result</td><td rowspan=1 colspan=1>Note</td></tr><tr><td rowspan=1 colspan=1>Known T, IA</td><td rowspan=1 colspan=1>Regret against optimal IA</td><td rowspan=1 colspan=1> $\widetilde { O } ( \sqrt { d / T } )$ </td><td rowspan=1 colspan=1>Algorithm 1,Theorem 1</td></tr><tr><td rowspan=1 colspan=1>Known T, IA/WC</td><td rowspan=1 colspan=1>Regret against optimum</td><td rowspan=1 colspan=1> $\overline { { \Omega ( \operatorname* { m i n } \{ d ^ { - 1 / 2 } , \sqrt { d / T } \} ) } }$ </td><td rowspan=1 colspan=1>Theorem 2</td></tr><tr><td rowspan=1 colspan=1>Unknown, IA/WC</td><td rowspan=1 colspan=1>Vanishing regret againstfinite-horizon optimum</td><td rowspan=1 colspan=1>Impossible: $\mathrm { l i m \ s u p } _ { t \to \infty } R _ { t } ^ { 5 } ( \pi ) > 0$ </td><td rowspan=1 colspan=1>Theorem 3</td></tr><tr><td rowspan=1 colspan=1>Unknown, WC</td><td rowspan=1 colspan=1>Tracking true SEPT</td><td rowspan=1 colspan=1> $\widetilde { O } ( \sqrt { d / t } )$ </td><td rowspan=1 colspan=1>Algorithm 2,Theorem 4</td></tr></table>

This preemptive abstraction facilitates modeling bandit feedback, characterizing optimal policies, and analyzing regret, but it excludes systems in which an ongoing computation, repair, or communication session cannot be interrupted, or in which preemption and migration are prohibitively costly. To capture such service commitments, we study a contextual queueing bandit model with a nonpreemptive setting, where once a job starts service, the server must continue processing it until completion. This change substantially complicates optimal scheduling. In the preemptive setting, an optimal policy is work-conserving, i.e., not idle, and selects an available job–server pair with the highest departure probability (Bae et al., 2026). However, under nonpreemption, the optimal action can depend on the current queue state, remaining horizon, and arriving job’s distribution even in the single-server case (see Section 3 for a detailed discussion). So, the optimal policy need not admit such a simple myopic characterization as in the preemptive setting, making learning and planning fundamentally intertwined. To study these dificulties in their most basic form, we focus first on the single-server setting. Our main results for known and unknown horizons are summarized in Table 1.

Known horizon: Learn–Clear–Plan algorithm and its optimality For a known horizon T, we compare with the optimal finite-horizon policy that may idle. Our Learn–Clear–Plan (LCP) algorithm estimates the departure probability function of jobs and their departure probability distribution, waits until the queue empties, and then follows an empirical finite-horizon Bellman policy (the policy that follows from the Bellman recursion, which is calculated by using the estimated function and distribution). Under some conditions related to job arrivals, Theorem 1 shows that it achieves $\widetilde { O } ( \sqrt { d / T } )$ as a regret upper bound. In particular, unlike the preemptive results of Bae et al. (2026); Bae and Lee (2026), this guarantee does not require a lower-eigenvalue bound on the context covariance matrix. We complement this result with the lower bounds in Theorem 2: for fixed model and trafic parameters, every learning policy incurs Ω(min $\{ d ^ { - 1 / 2 } , \sqrt { d / T } \}$ ) regret against the optimum policy on some instance, regardless of whether idling is permitted or not. Hence, when $T \geq d ^ { 2 }$ , our upper bound under fixed horizon is optimal in its dependence on d and T

up to logarithms.

Unknown horizon: Impossibility and SEPT tracking. When the horizon is unknown, Theorem 3 shows that there is no horizon-free policy such that the t-th round queue-length regret converges to 0 as $t \to \infty$ , even when the service model is known. We therefore use $\mathrm { S E P T }$ , the policy that serves a waiting job with the highest departure probability whenever the server is free, as a fixed horizon-independent reference policy. Our estimated-SEPT algorithm updates its logistic estimator only between busy periods and follows the resulting priority rule during each busy period. By Theorem 4, its tracking error satisfies $\mathcal { T } _ { t } ^ { \mathrm { S E P T } } = \widetilde { O } ( \sqrt { d / t } )$ for fixed model and trafic parameters. Thus, our known-horizon result concerns regret relative to a finite-horizon optimum, whereas our unknown-horizon result concerns tracking a fixed reference policy.

## 2 Problem Setting

We study the nonpreemptive contextual queueing bandit problem in a discrete-time system consisting of a single queue and a single server. Each job is represented by a context vector in $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ which determines its service-completion probability. We study both a known-horizon setting, where the server knows the evaluation horizon $T \in \mathbb { N }$ and may design its policy accordingly, and an unknown-horizon setting, where the server does not know the evaluation horizon and must use a horizon-independent policy (we call this an anytime policy). Let $Q _ { t }$ denote the number of unfinished jobs present at the beginning of round t, and call it the queue length at round t. We assume that the system is initially empty, so $Q _ { 1 } = 0$ . A key feature of our model is the nonpreemptive setting. This means that once the server selects a job, it must continue serving that job in every subsequent round until the job departs or the horizon ends. We call a job that is currently in service a job in service, and call all other unfinished jobs waiting jobs. The server is called free when no job is in service and the queue is not empty, and when it is free, it may start one waiting job or remain idle, meaning that it performs no service in that round. If the queue is empty, it idles necessarily.

Each round consists of a service stage (departure process) followed by an arrival stage (arrival process). If there is a job in service at the beginning of round t, the server must serve that job again. Otherwise, it may start one waiting job of its choice or idle. Here, the server can observe whether the served job is completed. Let $D _ { t } \in \{ 0 , 1 \}$ be the departure indicator in round t, with $D _ { t } = 0$ if the server idles. If a job with context x is served in round $t ,$ its departure indicator has distribution Bern(p(x)), where $p ( x ) : = \mu ( x ^ { \top } \theta ^ { * } )$ is called the success probability function and $\mu ( z ) : = ( 1 + e ^ { - z } ) ^ { - 1 }$ Here, $\theta ^ { * } \in \mathbb { R } ^ { d }$ is called the parameter vector. Conditional on the contexts, departure indicators are independent across jobs and rounds. Equivalently, if $G ( x )$ is the number of service attempts required to complete a job with context $x ,$ then $G ( x ) \sim \operatorname { G e o m } ( p ( x ) )$ and $\mathbb { E } [ G ( x ) ] = 1 / p ( x )$ holds. After the service stage, a new job arrives with indicator $A _ { t } \sim \operatorname { B e r n } ( \lambda )$ . Conditional on $A _ { t } = 1$ , draw $X _ { t } \sim \mathcal { D }$ from a context distribution D independently. The arriving job becomes available for service in round $t + 1$ , and we denote $X _ { t }$ as the context of the job that arrived in round t. Then the queue length evolves as $Q _ { t + 1 } = ( Q _ { t } - D _ { t } ) ^ { + } + A _ { t }$ , where $( q ) ^ { + } : = \operatorname* { m a x } \{ 0 , q \}$

Admissible policies may randomize, although the algorithms proposed in this paper are deterministic conditional on the observed history. We consider two problem settings. In the idling-allowed (IA) setting, policies are allowed to idle whenever the server is free. In the work-conserving (WC) setting, policies must start a waiting job whenever the server is free and the queue is nonempty. We refer to policies admissible in these settings as IA and WC policies, respectively. Thus, every

WC policy is also an IA policy. For a horizon $T \in \mathbb { N }$ , denote $J _ { T } ^ { \pi } : = \mathbb { E } _ { \pi } [ Q _ { T + 1 } ]$ as the expected final queue length when the server follows policy π. Let $V _ { T } ^ { \mathrm { I A } }$ and $\overset { \cdot \mathrm { ~ \sum ~ } } { V _ { T } ^ { \mathrm { W C } } }$ denote the minimum expected final queue lengths over all IA and WC policies, respectively, with knowledge of $T , \theta ^ { * }$ , and $\mathcal { D }$ . We define the corresponding regrets as $R _ { T } ^ { \mathrm { I A } } ( \pi ) : = J _ { T } ^ { \pi } - \bar { V } _ { T } ^ { \mathrm { I A } } , R _ { T } ^ { \mathrm { W C } } ( \pi ) : = J _ { T } ^ { \pi } - \bar { V } _ { T } ^ { \mathrm { W C } }$

Assumption 1. The sequence $( A _ { t } , X _ { t } ) _ { t \geq 1 }$ is i.i.d., with $A _ { t }$ independent of $X _ { t }$ . The server knows $\lambda \in ( 0 , 1 ) , S > 0$ , and $0 < p _ { - } < p _ { + } < 1$ , while $\theta ^ { * }$ and D are unknown. Moreover, $\| X \| _ { 2 } \leq 1$ almost surely, $\| \theta ^ { * } \| _ { 2 } \leq S , p ( X ) \in [ p _ { - } , p _ { + } ]$ almost surely, and $\rho : = \lambda \mathbb { E } _ { X \sim { \mathcal { D } } } \left[ 1 / p ( X ) \right] ] < 1$ holds.

In Assumption 1, except for the final condition $\rho < 1$ , these are nothing but the standard boundedness and stochastic-context assumptions for logistic bandit models (Bae et al., 2026; Bae and Lee, 2026). The condition $\rho < 1$ plays a crucial role in our analysis, as it is a key ingredient in establishing the workload properties in Lemma 1, which are repeatedly used throughout the paper. See Section 3.2 and Appendix C for its interpretation and detailed use. It also serves as a replacement for the trafic-slack condition used in the prior work.

## 3 Main Dificulties and new techniques

In this section, we highlight the main dificulty arising from the nonpreemptive setting and introduce a new tool, which we call workload, to address this dificulty.

## 3.1 Dificulties of nonpreemptive setting

A central dificulty in nonpreemptive contextual queueing bandits is the loss of the simple optimal scheduling rule available in the preemptive setting. In the preemptive model of Bae et al. (2026); Bae and Lee (2026), an optimal policy is work-conserving and selects a waiting job with the highest departure probability when the server is free. Under nonpreemption, this myopic rule need not be optimal, as the following counterexamples show. Suppose that a free server has one waiting job with departure probability $p _ { s }$ . With one round remaining, serving it immediately is trivially optimal. However, with two rounds remaining, the advantage in expected departures of idling first and then serving the best available job, compared with serving immediately, is at least the following value.

$$
p _ { s } ^ { 2 } - p _ { s } ( 1 + \lambda ) + \lambda \left( \mathbb { E } [ \operatorname* { m a x } \{ p _ { s } , p ( X ) \} ] - p _ { s } \mathbb { E } [ p ( X ) ] \right) .\tag{1}
$$

A detailed proof is included in Appendix G.1.

We can construct a two-context instance satisfying Assumption 1 such that the quantity (1) is positive, as shown in Appendix G.1. This counterexample shows that a work-conserving policy is not always optimal. Furthermore, even in the WC setting, choosing a job with the highest departure probability is not always optimal, as shown in the following proposition.

Proposition 1. There is a stable three-context instance with $\begin{array} { r } { \lambda = \frac { 1 } { 2 } } \end{array}$ and departure probabilities $\begin{array} { r } { p _ { A } = \frac { 7 } { 2 0 } , p _ { B } = \frac { 1 3 } { 2 0 } } \end{array}$ , and $\begin{array} { r } { p _ { F } = \frac { 1 9 } { 2 0 } } \end{array}$ such that when the server is free and exactly one A job and one B job are waiting, serving B first is the unique optimal action with one round remaining. With fourteen rounds remaining, serving A first is the unique optimal action. This instance satisfies Assumption 1.

The proof of this proposition will be given in Appendix G.2. Hence, in the nonpreemptive setting, there is no universal horizon-independent priority rule, and we need to know the remaining horizon, the current queue, and the future-arrival distribution to decide which action is optimal in each round. Thus, this explains that determining an optimal action requires solving the finite-horizon Bellman recursion at each round.

## 3.2 A New Technique: Workload-Based Arrival Analysis

In the nonpreemptive setting, the optimal policy depends on the queue state and remaining horizon rather than a simple myopic rule, making departure-based regret analysis dificult. From this viewpoint, we shift our attention to the arrival process, whose law is much simpler and independent of the scheduling policy. For each arriving job with context $X _ { t } .$ , independently draw its required service time $G ( X _ { t } ) \sim \operatorname { G e o m } ( p ( X _ { t } ) )$ , and define the workload $W _ { t }$ as the sum of the remaining required service times of all jobs present at the beginning of round t. Under any WC policy, the following recursion holds.

$$
W _ { t + 1 } = ( W _ { t } - 1 ) ^ { + } + A _ { t } G ( X _ { t } )\tag{2}
$$

Thus, under a common coupling of arrivals, arrived jobs, and their required service times, the workload is independent of the service policy and order, satisfies $Q _ { t } \leq W _ { t }$ , and vanishes if $Q _ { t } = 0$ Thus, any two WC queues $Q _ { t } ^ { ( 1 ) } , Q _ { t } ^ { ( 2 ) }$ coupled as above share the same workload process $( W _ { t } ) _ { t \geq 1 }$ become empty at the same times, and satisfy $| Q _ { t } ^ { ( 1 ) } - Q _ { t } ^ { ( 2 ) } | \leq W _ { t }$ for all t. Furthermore, with this definition, $\rho = \lambda \mathbb { E } [ 1 / p ( X ) ] = \lambda \mathbb { E } _ { X \sim { \mathcal { D } } , G ( X ) \sim \operatorname { G e o m } ( p ( X ) ) } [ G ( X ) ]$ is precisely the expected amount of workload brought into the system by arrivals in each round.

We now introduce the following basic workload properties. The constants in the following are formally defined in Appendix C, and the properties are proved in Appendices C.1 and C.3.

Lemma 1. Under Assumption 1, there exist positive constants $r \in ( 0 , 1 ) , \psi ( r ) \in ( 0 , 1 ) , M _ { G } ( r )$ $K _ { r }$ and $C _ { W } ( r )$ such that the WC workload process started from $W _ { 1 } = 0$ satisfies $\mathbb { E } e ^ { r W _ { t } } \leq K _ { r }$ and $\mathbb { E } W _ { t } \le C _ { W } ( r )$ for every $t \geq 1$ . Moreover, for $u \geq 1$ , let $\tau _ { 0 } : =$ inf $\left\{ j \ge 0 : W _ { u + j } = 0 \right\}$ . Then,

$$
\mathbb { E } \left[ W _ { u + h } \mathbf { 1 } \{ \tau _ { 0 } > h \} \vert W _ { u } \right] \le \frac { e ^ { r W _ { u } } } { r } \psi ( r ) ^ { h }\tag{3}
$$

for every $h \geq 0$ . Hence, E $[ W _ { u + h } \mathbf { 1 } \{ \tau _ { 0 } > h \} ] \le ( K _ { r } / r ) \psi ( r ) ^ { h }$ . We refer to these two inequalities as the workload tail bounds. Lastly, $\mathbb { E } [ W _ { t + 1 } - W _ { t } \mid W _ { t } = w ] = \rho - 1 < 0$ for every $w \ge 1$ , so the workload recursion admits a stationary distribution W that stochastically dominates $W _ { t } ~ f o r$ every $t \geq 1$

Before concluding this subsection, we introduce the notion of a busy period, which will be used later in our SEPT analysis. Under a WC policy, a busy period is defined as the sequence of consecutive service rounds that begins when the server first serves a job after the system has become empty and ends in the first round whose service stage leaves the system empty, before the arrival at the end of that round. Let $\tau _ { \mathrm { b p } }$ denote its length.

## 4 Known Horizon: Upper and Lower bounds of Regret

We now consider the known-horizon IA setting and present the Learn-Clear-Plan (LCP) algorithm, which achieves $R _ { T } ^ { \mathrm { I A } } = \widetilde { O } ( \sqrt { d / ( \lambda T ) } )$ . Before presenting LCP, we define the estimators used by the algorithm and provide high-probability upper bounds, and introduce the concept of Bellman policy.

## 4.1 Estimating Departure Probabilities and Their Distribution

Index jobs by their arrival order. Let $X _ { i }$ be the context of job i and Y<sub>i</sub> the departure indicator from its first service attempt. Then $( X _ { i } , Y _ { i } ) _ { i \geq 1 }$ is an i.i.d. sequence with $Y _ { i } \mid X _ { i } \sim \operatorname { \mathrm { B e r n } } ( p ( X _ { i } ) )$ ). Using the first n pairs, we estimate $\theta ^ { * }$ by $\widehat { \theta } _ { n }$ through empirical risk minimization, and use this estimate to construct a predictor ${ \widehat { p } } _ { n }$ of $p .$ For $y \in \{ 0 , 1 \}$ , define $\ell _ { \theta } ( x , y ) : = \log ( 1 + e ^ { x ^ { \top } \theta } ) - y x ^ { \top } \theta$ , and let

$$
\widehat { \theta } _ { n } \in \mathop { \mathrm { a r g } \operatorname* { m i n } } _ { \| \theta \| _ { 2 } \leq S } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell _ { \theta } ( X _ { i } , Y _ { i } ) , \qquad \widehat { p } _ { n } ( x ) : = \mathrm { p r o j } _ { [ p _ { - } , p _ { + } ] } \Big ( \mu ( x ^ { \top } \widehat { \theta } _ { n } ) \Big ) .\tag{4}
$$

We next estimate the distribution of future jobs’ departure probabilities. Using the contexts from pairs $( X _ { i } , Y _ { i } ) _ { i = n + 1 } ^ { 2 n }$ and ${ \widehat { p } } _ { n } .$ , we define the empirical distribution $\begin{array} { r } { \widehat { F } _ { n } : = \frac { 1 } { n } \sum _ { i = n + 1 } ^ { \overline { { 2 } } n } \delta _ { \widehat { p } _ { n } ( X _ { i } ) } } \end{array}$ . This distribution estimates $F _ { \widehat { p } _ { n } } : = \operatorname { L a w } ( \widehat { p } _ { n } ( X ) )$ , where $X \sim \mathcal { D }$ is independent of the training block.

To quantify the estimation errors, for any predictor $q ~ : ~ \mathbb { R } ^ { d } \  \ [ p _ { - } , p _ { + } ]$ , define $\mathcal { E } ( q ) : =$ $\mathbb { E } _ { X \sim { \mathcal { D } } } | q ( X ) - p ( X ) |$ and denote $W _ { 1 }$ as the 1-Wasserstein distance. Let $\bar { S } : = \operatorname* { m a x } \{ 1 , S \}$ . For $n \geq 5$ , define $\epsilon _ { n } ( \delta ) : =$ min(1, $\sqrt { ( 8 e ^ { \bar { S } } \left( d \log ( 3 2 \bar { S } n ) + \log ( 1 / \delta ) \right) + 1 ) / 2 n } )$ and set $\epsilon _ { n } ( \delta ) = 1$ for $n < 5$ We further define $\eta _ { n } ( \delta ) : = ( p _ { + } - p _ { - } ) \sqrt { \log ( 2 / \delta ) / 2 n }$

Lemma 2. For every $n \geq 1$ and $\delta \in ( 0 , 1 )$ , each of the following bounds holds with probability at least $1 - \delta$

$$
\begin{array} { r } { \mathcal { E } ( \widehat { p } _ { n } ) \leq \epsilon _ { n } ( \delta ) , \qquad W _ { 1 } ( \widehat { F } _ { n } , F _ { \widehat { p } _ { n } } ) \leq \eta _ { n } ( \delta ) . } \end{array}\tag{5}
$$

The proofs of the two inequalities are provided in Appendices D.1 and D.3. We refer to them as concentration inequalities.

## 4.2 Bellman Policy for the Known-Horizon IA Setting

We now explain how to characterize an optimal policy in the known-horizon IA setting through a Bellman recursion when p and D are known. We first represent the current state mathematically and then present the corresponding finite-horizon Bellman recursion. We first consider a future-job success-probability distribution F with finite support. Let $r _ { 1 } , \ldots , r _ { m }$ be the distinct values consisting of the support of F together with the success probabilities of all currently present jobs, and let $\nu _ { j } = F ( \{ r _ { j } \} )$ ; thus $\nu _ { j } = 0$ for a value that appears only among the currently present jobs. Represent the queue at the beginning of a round by a state $( c , a )$ , where $c \in \mathbb { Z } _ { > 0 } ^ { m } , c _ { j }$ is the number of waiting jobs with success probability $r _ { j } .$ , and $a \in \{ 0 , 1 , \ldots , m \}$ identifies the job type in service; $a = 0$ means that the server is free. The job in service is excluded from c and is represented by a. Let $V _ { h } ( c , a )$ be the minimum expected terminal queue length with h rounds remaining and starting state $( c , a )$ , so that $\begin{array} { r } { V _ { 0 } ( c , a ) = \sum _ { j } c _ { j } + \mathbf { 1 } \{ a \neq 0 \} } \end{array}$ , the number of unfinished jobs. Writing $e _ { j }$ for the j-th standard basis vector, define the arrival operator, which averages over the arrival at the end of the current round, by $( A f ) ( c , a ) = ( 1 - \lambda ) f ( c , a ) + \lambda \textstyle \sum _ { i = 1 } ^ { m } \nu _ { j } f ( c + e _ { j } , a )$

If $a \neq 0$ , nonpreemption forces the continuation of service, giving

$$
V _ { h } ( c , a ) = r _ { a } ( \varLambda V _ { h - 1 } ) ( c , 0 ) + ( 1 - r _ { a } ) ( \varLambda V _ { h - 1 } ) ( c , a ) .\tag{6}
$$

If the server is free, it may idle or start a waiting job:

$$
\begin{array} { r l } { } & { V _ { h } ( c , 0 ) } \\ { } & { = \operatorname* { m i n } \Bigl \{ ( A V _ { h - 1 } ) ( c , 0 ) , \underset { j : c _ { j } > 0 } { \operatorname* { m i n } } \bigl [ r _ { j } ( A V _ { h - 1 } ) ( c - e _ { j } , 0 ) + ( 1 - r _ { j } ) ( A V _ { h - 1 } ) ( c - e _ { j } , j ) \bigr ] \Bigr \} . } \end{array}\tag{7}
$$

Here, the first term in $( 7 )$ corresponds to idling, and each term in the inner minimum corresponds to starting a type-j job. The inner minimum is omitted when $c = 0$ . Each term is the expected optimal continuation value resulting from the corresponding action. Hence, if the first term attains the minimum, idling is optimal, while if the term indexed by $j$ reaches the minimum, starting a type-j job is optimal. Repeating such minimizing choices whenever the server is free yields an optimal finite-horizon policy. For a general $F _ { ; }$ an analogous recursion represents the waiting jobs by a multiset of success probabilities and replaces the arrival sum by an integral with respect to $F .$

We call a policy that makes these minimizing choices at every free-server state a Bellman policy for the corresponding model and horizon. The true Bellman policy based on $( p , F _ { p } )$ uses $p ( x )$ as a success probability function and $F _ { p } = \operatorname { L a w } ( p ( X ) )$ , $X \sim \mathcal { D }$ , for future-arrivals success-probability distribution. Thus, a true Bellman policy is an optimal IA policy. In particular, starting from an empty queue with $T$ rounds remaining, it attains $V _ { T } ^ { \mathrm { I A } }$ . Suppose now that the true model is $( p , F _ { p } )$ but the server has only an estimated success-probability function $q$ and an estimated future-job distribution $\widehat F .$ . The estimated Bellman policy based on $( q , { \widehat { F } } )$ treats $( q , { \widehat { F } } )$ as the true model and follows the minimizing actions of the corresponding finite-horizon Bellman recursion.

## 4.3 The Learn-Clear-Plan Algorithm

We now introduce the Learn–Clear–Plan (LCP) algorithm, which first estimates the unknown model, clears the queue, and then follows the estimated Bellman policy based on that estimated model. Learn. Choose a planning-window length $H < T$ , let $L : = T - H$ , and set $n : = \lfloor \lambda L / 8 \rfloor$ . During the first L rounds, whenever the server is free, it starts the earliest-arriving waiting job; we refer to this rule as FCFS policy, meaning that first-come, first-served. When job i with context $X _ { i }$ first begins service, the server retains only the departure indicator $Y _ { i }$ from its first service attempt. If at least 2n samples are collected, the first n pairs $( X _ { i } , Y _ { i } )$ are used to construct ${ \widehat { p } } _ { n }$ , while the contexts of the next n jobs are used to construct ${ \widehat { F } } _ { n }$ . Separating the samples into two blocks ensures that, conditional on ${ \widehat { p } } _ { n }$ , the contexts in the second block remain i.i.d.

Clear. If the queue remains nonempty after round L, the server continues serving jobs according to FCFS without idling until it reaches a round that begins with an empty queue. This provides an empty initial state for the subsequent planning phase.

Plan. Once the queue has been cleared, the server follows the estimated-model Bellman policy based on $( \widehat { p } _ { n } , \widehat { F } _ { n } )$ for the current queue state and the remaining number of rounds. Importantly, the current queue need not consist only of jobs whose estimated success probabilities lie in the support of ${ \widehat { F } } _ { n }$ . The distribution ${ \widehat { F } } _ { n }$ describes only the success probabilities of future arrivals assumed in the Bellman calculation; the jobs already present in the queue are part of the given current state. Thus, if a job with context $X$ arrives in the actual system with ${ \widehat { p } } _ { n } ( X )$ outside the support of ${ \widehat { F } } _ { n }$ during the Plan phase, it is simply included in the current state as an additional type with zero future-arrival mass. This leaves ${ \widehat { F } } _ { n }$ unchanged while allowing the job to be considered among the available actions in subsequent Bellman calculations.

Algorithm 1 Learn–Clear–Plan   
Require: horizon $T ,$ planning-window length $H < T$   
1: $L \gets T - H , n \gets \lfloor \lambda L / 8 \rfloor$   
2: $Z \gets ( ) , \operatorname* { P L A N } \gets 0$   
3: for $t = 1 , \dots , T$ do   
4: if $t > L , Q _ { t } = 0 ,$ and $\mathrm { P L A N } = 0$ then   
5: if $| Z | \geq$ 2n and $n \geq 1$ then   
6: compute ${ \widehat { p } } _ { n }$ from the first n pairs in $Z$   
7: $\begin{array} { r } { \widehat { F } _ { n } \gets \frac { 1 } { n } \sum _ { i = n + 1 } ^ { 2 n } \delta _ { \widehat { p } _ { n } ( X _ { i } ) } } \end{array}$   
8: else   
9: $\widehat { p } _ { n } ( x ) \gets ( p _ { - } + p _ { + } ) / 2 , \widehat { F } _ { n } \gets \delta _ { ( p _ { - } + p _ { + } ) / 2 }$   
10: end if   
11: $\mathrm { P L A N }  1$   
12: end if   
13: if $\mathrm { P L A N } = 1$ then   
14: take an optimal action from the Bellman recursion with $T - t + 1$ rounds under $( \widehat { p } _ { n } , \widehat { F } _ { n } )$   
15: else   
16: continue the job in service; otherwise serve FCFS, or idle if $Q _ { t } = 0$   
17: end if   
18: observe $D _ { t }$   
19: if job i begins service in round t then   
20: set $Y _ { i } \gets D _ { t }$ and append $( X _ { i } , Y _ { i } )$ to Z   
21: end if   
22: observe $A _ { t }$ and, if $A _ { t } = 1$ , the arriving context $X _ { t }$   
23: end for

## 4.4 Upper bound of regret

Theorem 1. Under Assumption 1, let $1 \leq H < T , L = T - H , n = \lfloor \lambda L / 8 \rfloor \geq 1$ , and $\delta _ { T } = T ^ { - 4 }$ $L e t \ \pi ^ { \mathrm { L C P } }$ be the policy in Algorithm 1. Then

$$
\begin{array} { r l } & { R _ { T } ^ { \mathrm { I A } } ( \pi ^ { \mathrm { L C P } } ) \leq \displaystyle \frac { K _ { r } } { r } \psi ( r ) ^ { H } + H \left( e ^ { - \lambda L / 8 } + K _ { r } e ^ { - r \lambda L / 4 } + 2 \delta _ { T } \right) } \\ & { \qquad + \displaystyle \frac { 2 H ^ { 2 } } { p _ { - } } \epsilon _ { n } ( \delta _ { T } ) + \frac { ( H - 1 ) H ( H + 1 ) ( H + 2 ) } { 1 2 } \eta _ { n } ( \delta _ { T } ) . } \end{array}\tag{8}
$$

Here, $r , \psi ( r ) , K _ { r }$ are constants as given in Lemma 1. For $H = \lceil \log ^ { 2 } ( e T ) \rceil$ and suficiently large $T _ { i }$ $R _ { T } ^ { \mathrm { I A } } ( \pi ^ { \mathrm { L C P } } ) = \widetilde { O } ( \sqrt { d / ( \lambda T ) } )$ for fixed model and trafic parameters. The total number of Bellman recursion terms evaluated over all decision epochs in the Plan phase is at most $\exp ( O ( \log ^ { 3 } T ) )$ .

Sketch of Proof. Consider the event that Clear occurs before the end of the horizon, at least 2n jobs have begun service by round $L ,$ and both bounds in Lemma 2 hold. The complementary event contributes the first two terms of Equation (8) by concentration inequalities and the workload tail bound in Lemma 1. In this event, Plan starts from an empty queue with $h \leq H$ rounds remaining. Let $q = \widehat { p } _ { n }$ and let πb be the estimated Bellman policy based on $( q , \widehat { F } _ { n } )$ . For an empty-start h-round system, let $J _ { h } ^ { r } ( \pi )$ denote the expected terminal queue length under the success probability function $r ,$ and let $V _ { h } ^ { r }$ be its optimal value among all IA policies. We decompose regret for this event as follows.

$$
J _ { h } ^ { p } ( \widehat \pi ) - V _ { h } ^ { p } = \left( J _ { h } ^ { p } ( \widehat \pi ) - J _ { h } ^ { q } ( \widehat \pi ) \right) + \left( J _ { h } ^ { q } ( \widehat \pi ) - V _ { h } ^ { q } \right) + \left( V _ { h } ^ { q } - V _ { h } ^ { p } \right) .
$$

The first and third terms compare models that difer in their success-probability functions, whereas the middle term captures the suboptimality under $( q , F _ { q } )$ of the Bellman policy computed using $( q , \widehat { F } _ { n } )$ . We now invoke Equations (32) and (33) to bound these terms. The first inequality below bounds the diference in the expected terminal queue length when the same policy is used under $p$ and $q$ in terms of $\mathcal { E } ( q )$ , while the second bounds the loss from using the Bellman policy computed with ${ \widehat { F } } _ { n }$ instead of $F _ { q } ,$ with q fixed, in terms of $W _ { 1 } ( F _ { q } , \widehat { F } _ { n } )$

$$
\vert J _ { h } ^ { p } ( \pi ) - J _ { h } ^ { q } ( \pi ) \vert \leq \frac { h ^ { 2 } } { p _ { - } } \mathcal { E } ( q ) , \qquad J _ { h } ^ { q } ( \widehat { \pi } ) - V _ { h } ^ { q } \leq \frac { ( h - 1 ) h ( h + 1 ) ( h + 2 ) } { 1 2 } W _ { 1 } ( F _ { q } , \widehat { F } _ { n } ) .\tag{9}
$$

Applying the first inequality to πb and to each of the optimal policies under p and q bounds the first and third terms by $2 H ^ { 2 } \epsilon _ { n } ( \delta _ { T } ) / p _ { - }$ , which yields the third term in Theorem 1. The second inequality, together with Lemma 2, bounds the middle term by the final term in Equation (8). The complete proof appears in Appendix E.

## 4.5 Lower Bounds of Regret

We next establish a worst-case regret lower bound for every learning policy, including randomized policies. We construct a family of hard instances with dummy, baseline, and candidate jobs that share the same context distribution D and model constants $\lambda , S , p _ { - } , p _ { + }$ , and difer only in $\theta ^ { * }$ . These instances are statistically hard to distinguish and are chosen so that, if the final round begins with the server free and exactly one baseline and one candidate waiting, either job selection is suboptimal on some instance, with a departure-probability (regret) gap of Ω(min $\{ 1 / \sqrt { d } , \sqrt { d / T } \} )$ ).

For WC policies, we define an event H that, followed by the arrival of a candidate, produces this final decision state. Stationary workload domination bounds the probability of this event away from zero, and standard testing arguments imply that, on some instance, the learner takes a suboptimal final action with suficiently high probability, yielding the desired regret lower bound. The IA proof follows the same strategy, but requires an additional contradiction argument because the event H requires the server to start a waiting job when it becomes free, while an IA policy may instead choose to idle. The complete proofs appear in Appendix F.

Theorem 2. For every $d \geq 3$ , there exist a context distribution D on $\mathbb { R } ^ { d }$ and constants $\lambda \in ( 0 , 1 )$ ， $S > 0$ , and $0 < p _ { - } < p _ { + } < 1$ with the following property. For every $T \geq 4$ , every setting ${ \mathsf { S } } \in \{ { \mathrm { I A } } , { \mathrm { W C } } \}$ , and every learning policy π in that setting, including randomized policies, there exists a parameter vector $\theta ^ { * }$ such that $( \mathcal { D } , \theta ^ { * } )$ satisfies Assumption 1 and

$$
R _ { T } ^ { \mathsf { S } } ( \pi ) \ge c _ { \mathsf { S } } \operatorname* { m i n } \left( 1 / { \sqrt { d } } , { \sqrt { d / T } } \right) ,\tag{10}
$$

even $i f \mathcal { D }$ is known to the server. The constants $c _ { \mathrm { I A } } , c _ { \mathrm { W C } } > 0$ and the model constants can be chosen independently of d and T.

For $T \geq d ^ { 2 }$ , the IA lower bound is $\Omega ( { \sqrt { d / T } } )$ , matching the upper LCP bound up to polylogarithmic factors for the fixed model and the trafic parameters. Thus, LCP achieves the optimal worst-case regret rate up to these factors in the known-horizon IA setting.

## 5 Unknown Horizon: Impossibility and SEPT Tracking

## 5.1 Nonexistence of an Anytime Policy with Vanishing Regret

A natural question in the unknown-horizon setting is whether an anytime IA or WC policy π can satisfy $R _ { t } ^ { \mathrm { I A } } ( \pi )  0$ (resp. $R _ { t } ^ { \mathrm { W C } } ( \pi )  0 )$ as $t \to \infty$ . However, the two counterexamples in Equation (1) and Proposition 1 show that this is impossible for both settings, even when $\theta ^ { * }$ and $\mathcal { D }$ are known.

Theorem 3. For each setting ${ \mathsf { S } } \in \{ { \mathrm { I A } } , { \mathrm { W C } } \}$ , there exists a fixed instance satisfying Assumption $^ { 1 , }$ with two contexts for IA and three for WC, such that every anytime S policy $\pi _ { ; }$ , even with full knowledge of $\theta ^ { * }$ and D, satisfies

$$
\operatorname* { l i m } _ { t \to \infty } { R } _ { t } ^ { { \mathsf { S } } } ( \pi ) > 0 .\tag{11}
$$

We present only the proof idea for the WC setting here. The IA case is proved similarly using the queue state in Equation (1), for which the optimal action depends on the remaining horizon. For WC, we use the state in Proposition 1, where the server is free with exactly one A job and one B job waiting. Using workload domination, we show that this state occurs at the beginning of the round $s ,$ for every $s \geq 5$ , with probability bounded away from zero. In this event, starting B is uniquely optimal if the evaluation horizon is $s ,$ whereas starting $A$ is uniquely optimal if the evaluation horizon is $s + 1 3$ . Thus, neither choosing A nor choosing B is optimal for both evaluation horizons, and each suboptimal choice incurs a fixed positive regret gap. Therefore, for any randomized action distribution over A and $B ,$ at least one of the regrets at horizons s and $s + 1 3$ is bounded below by a positive constant. This contradicts the convergence of regret to zero. The complete WC proof appears in Appendix G.3, while the complete IA proof appears in Appendix G.1 within the same appendix section.

Thus, in both IA and WC settings, no anytime policy, including randomized policies, can achieve vanishing regret against the finite-horizon optimum, i.e., $R _ { t } ^ { \mathsf { S } } ( \pi ) \to 0$ as $t \to \infty$ . Moreover, in the free-server states of these counterexamples, no action is optimal for all horizons: each choice is suboptimal for some horizon.

## 5.2 SEPT policy and estimated SEPT Algorithm

Because the optimal action at a given state can vary with the evaluation horizon, and no anytime policy achieves vanishing regret against the finite-horizon optimum, we adopt the shortest-expectedprocessing-time (SEPT) policy as an anytime reference policy. SEPT is the WC policy that, whenever the server is free, starts a waiting job with the highest success probability, breaking ties by arrival order. Since we cannot follow the true SEPT policy because $p$ is unknown, our goal is to construct an anytime policy whose expected terminal queue length approaches that of true SEPT as the evaluation horizon grows, without assuming that SEPT is optimal in the present nonpreemptive setting. To this end, we introduce the estimated-SEPT algorithm, a WC policy that updates an estimate $\widehat { p }$ only at the end of each busy period, keeps it fixed throughout the next busy period, and uses it in place of $p$ to apply the same priority rule as SEPT. Updating $\widehat { p }$ only between busy periods ensures that the predictor used within a busy period depends only on observations preceding that busy period and is therefore independent of that busy period’s arriving job contexts. Let $\mathbf { \mathcal { J } } _ { t } ^ { \mathrm { a l g } }$ and $J _ { t } ^ { \mathrm { S E P T } }$ denote the expected final queue lengths after t rounds under estimated-SEPT and true SEPT, respectively, and define $\mathcal { T } _ { t } ^ { \mathrm { S E P T } } : = | J _ { t } ^ { \mathrm { a l g } } - J _ { t } ^ { \mathrm { S E P T } } |$ . For $t \geq 6 ,$ put $u = \lfloor t / 2 \rfloor , m _ { t } = \lfloor \lambda u / 2 \rfloor$ , and $\delta _ { t } = t ^ { - 4 }$ . If $m _ { t } \geq 5 , \mathrm { s e t } \ \bar { \epsilon } _ { t } = \operatorname* { m i n } \left( 1 , \sqrt { ( 8 e ^ { \bar { S } } \left( d \log ( 3 2 \bar { S } t ) + \log ( \pi ^ { 2 } t ^ { 2 } / 6 \delta _ { t } ) \right) + 1 ) / 2 m _ { t } } \right) , b _ { t } = e ^ { - \lambda u / 8 } + \delta _ { t }$

Algorithm 2 Busy-period estimated SEPT   
1: $Z \gets ( ) , \widehat { p } ( x ) \gets ( p _ { - } + p _ { + } ) / 2$   
2: for $t = 1 , 2 , \dots$ . do   
3: if a job is in service then   
4: continue serving that job   
5: else if $Q _ { t } > 0$ then   
6: serve the earliest arrival in arg max<sub>j waiting</sub> $\widehat { p } ( X _ { j } )$   
7: else   
8: idle   
9: end if   
10: observe $D _ { t } ;$ if job i starts service, set $Y _ { i } \gets D _ { t }$ and append $( X _ { i } , Y _ { i } )$ to $Z$   
11: if the queue is empty after service and before the arrival, and $| Z | \ge 1$ then   
12: recompute $\widehat { p }$ using Equation (4) from all pairs in $Z$   
13: end if   
14: observe $A _ { t }$ and, if $A _ { t } = 1$ , the arriving context $X _ { t }$   
15: end for

Theorem 4. Under Assumption 1, if $m _ { t } \geq 5$ , then Algorithm 2 satisfies

$$
\mathcal { T } _ { t } ^ { \mathrm { S E P T } } \leq \Phi _ { r } ( \bar { \epsilon } _ { t } ) + \frac { M _ { G } ( r ) } { r ( 1 - \psi ( r ) ) } \psi ( r ) ^ { u } + \frac { \sqrt { K _ { r } b _ { t } } } { r } .\tag{12}
$$

Here, $\Phi _ { r } ( \epsilon )$ is the function defined on $\epsilon \in ( 0 , 1 ]$ and satisfies $\Phi _ { r } ( \epsilon ) = O ( \epsilon ( 1 + \log ^ { 3 } ( 1 / \epsilon ) ) )$ , and $r , \psi ( r ) , M _ { G } ( r ) , K _ { \eta }$ are constants as given in Lemma 1. Consequently, for fixed model and trafic parameters, $\mathcal { T } _ { t } ^ { \mathrm { S E P T } } = \widetilde { O } ( \sqrt { d / ( \lambda t ) } )$ ). $I f m _ { t } < 5$ , the uniform bound ${ \mathcal { T } } _ { t } ^ { \mathrm { S E P T } } \leq C _ { W } ( r )$ holds.

The proof uses ideas similar to those in the LCP analysis. Under common arrivals, contexts, and service times, estimated-SEPT and true SEPT have identical workload processes and busy period boundaries, so only the last busy period can afect the terminal queue-length diference. We decompose the tracking error according to its starting round. For recent busy periods with small prediction error, Lemma 15 together with the workload tail bound gives the first term in Equation (12); old busy periods are controlled directly by the workload tail bound, giving the second. Finally, Lemma 13 controls the contribution of recent busy periods with large prediction error by the third term. The complete proof appears in Appendix H.

## 6 Conclusion and Future Work

We studied contextual queueing bandits under nonpreemptive service. For a known horizon, our LCP algorithm achieves $\widetilde { O } ( \sqrt { d / ( \lambda T ) } )$ regret against the IA optimum, matching our lower bound up to logarithmic factors for $T \geq d ^ { 2 }$ . For an unknown horizon, we show that no anytime policy achieves vanishing regret against the finite-horizon IA or WC optimum, and that our estimated-SEPT algorithm tracks true SEPT at rate $\widetilde { O } ( \sqrt { d / ( \lambda t ) } ) ,$ ).

Extending these results to multiple servers is challenging because the workload contributed by an arriving job depends on which server will eventually process it in the future. Thus, unlike in the single-server setting, the workload is policy-dependent, and our common-workload and common-empty-time coupling arguments no longer apply directly. Nevertheless, the techniques developed in this work, together with the multi-server analysis of Bae and Lee (2026), suggest that under appropriate assumptions it may be possible to obtain upper and lower regret bounds of a similar form in the multi-server nonpreemptive setting. Establishing such results is an interesting direction for future work.

## References

Marc Abeille, Louis Faury, and Clément Calauzènes. Instance-wise minimax-optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence and Statistics, pages 3691–3699. PMLR, 2021.

Seoungbin Bae and Dabeen Lee. Algorithm for contextual queueing bandits with rate-optimal queue length regret. arXiv preprint arXiv:2606.09668, 2026.

Seoungbin Bae, Garyeong Kang, and Dabeen Lee. Queue length regret bounds for contextual queueing bandits. arXiv preprint arXiv:2601.19300, 2026.

Tuhinangshu Choudhury, Gauri Joshi, Weina Wang, and Sanjay Shakkottai. Job dispatching policies for queueing systems with unknown service rates. In Proceedings of the Twenty-second International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing, pages 181–190, 2021.

Aryeh Dvoretzky, Jack Kiefer, and Jacob Wolfowitz. Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator. The Annals of Mathematical Statistics, pages 642–669, 1956.

Louis Faury, Marc Abeille, Clément Calauzènes, and Olivier Fercoq. Improved optimistic algorithms for logistic bandits. In International Conference on Machine Learning, pages 3052–3060. PMLR, 2020.

Sarah Filippi, Olivier Cappe, Aurélien Garivier, and Csaba Szepesvári. Parametric bandits: The generalized linear case. Advances in neural information processing systems, 23, 2010.

Daniel Freund, Thodoris Lykouris, and Wentao Weng. Eficient decentralized multi-agent learning in asymmetric queuing systems. In Conference on Learning Theory, pages 4080–4084. PMLR, 2022.

Wei-Kang Hsu, Jiaming Xu, Xiaojun Lin, and Mark R Bell. Integrated online learning and adaptive control in queueing systems with uncertain payofs. Operations Research, 70(2):1166–1181, 2022.

Jiatai Huang, Leana Golubchik, and Longbo Huang. When lyapunov drift based queue scheduling meets adversarial bandit learning. IEEE/ACM Transactions on Networking, 32(4):3034–3044, 2024.

Jung-hun Kim and Min-hwan Oh. Queueing matching bandits with preference feedback. Advances in Neural Information Processing Systems, 37:64649–64702, 2024.

Subhashini Krishnasamy, Rajat Sen, Ramesh Johari, and Sanjay Shakkottai. Regret of queueing bandits. Advances in Neural Information Processing Systems, 29, 2016.

Subhashini Krishnasamy, Rajat Sen, Ramesh Johari, and Sanjay Shakkottai. Learning unknown service rates in queues: A multiarmed bandit approach. Operations research, 69(1):315–330, 2021.

Lihong Li, Yu Lu, and Dengyong Zhou. Provably optimal algorithms for generalized linear contextual bandits. In International Conference on Machine Learning, pages 2071–2080. PMLR, 2017.

Qingkai Liang and Eytan Modiano. Minimizing queue length regret under adversarial network models. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 2(1):1–32, 2018.

Nishant Mehta. Fast rates with high probability in exp-concave statistical learning. In Artificial Intelligence and Statistics, pages 1085–1093. PMLR, 2017.

Nadav Merlis, Hugo Richard, Flore Sentenac, Corentin Odic, Mathieu Molina, and Vianney Perchet. On preemption and learning in stochastic scheduling. In International Conference on Machine Learning, pages 24478–24516. PMLR, 2023.

Michael Neely. Stochastic network optimization with application to communication and queueing systems. Morgan & Claypool Publishers, 2010.

Rhonda Righter and J George Shanthikumar. Scheduling multiclass single server queueing systems to stochastically maximize the number of successful departures. Probability in the Engineering and Informational Sciences, 3(3):323–333, 1989.

Flore Sentenac, Etienne Boursier, and Vianney Perchet. Decentralized learning in online queuing systems. Advances in Neural Information Processing Systems, 34:18501–18512, 2021.

Thomas Stahlbuhk, Brooke Shrader, and Eytan Modiano. Learning algorithms for minimizing queue length regret. IEEE Transactions on Information Theory, 67(3):1759–1781, 2021.

Richard R Weber, Pravin Varaiya, and Jean Walrand. Scheduling jobs with stochastically ordered processing times on parallel machines to minimize expected flowtime. Journal of Applied Probability, 23(3):841–847, 1986.

Zixian Yang, R Srikant, and Lei Ying. Learning while scheduling in multi-server systems with unknown statistics: Maxweight with discounted ucb. In International Conference on Artificial Intelligence and Statistics, pages 4275–4312. PMLR, 2023.

Yueyang Zhong, John R Birge, and Amy R Ward. Learning to schedule in multiclass many-server queues with abandonment. Operations Research, 73(6):3085–3103, 2025.

## A Appendix Overview

The appendix is organized as follows.

1. Appendix B provides additional discussion of related work.

2. Appendix C proves the workload bounds in Lemma 1 and establishes additional workload and busy-period properties used throughout the subsequent analyses.

3. Appendix D proves the departure-probability and distribution estimation bounds in Lemma 2, together with a simultaneous prediction bound.

4. Appendix E gives the complete proof of the known-horizon upper bound in Theorem 1, together with the supporting lemmas used in the analysis.

5. Appendix F constructs the hard family of instances and proves the regret lower bounds in Theorem 2 for both the WC and IA settings.

6. Appendix G proves Theorem 3, namely, that for both ${ \mathsf { S } } \in \{ { \mathrm { I A } } , { \mathrm { W C } } \}$ , there is no anytime S policy π such that $R _ { t } ^ { \mathsf { S } } ( \pi ) \to 0$ as $t \to \infty$ , even when the model is fully known.

7. Finally, Appendix H gives the complete proof of the tracking guarantee for the estimated-SEPT algorithm in Theorem 4, together with the supporting lemmas used in the analysis.

## B Additional Related Work

Queueing bandits and learning while scheduling. Queueing bandits study learning-whilescheduling problems in which unknown service rates must be learned while controlling queue lengths (Krishnasamy et al., 2016, 2021). Queue-length regret has subsequently been studied under several queueing architectures and information structures (Stahlbuhk et al., 2021; Liang and Modiano, 2018). Related work considers job dispatching with unknown service rates (Choudhury et al., 2021), decentralized learning in queueing systems (Sentenac et al., 2021; Freund et al., 2022), integrated online learning and queue control (Hsu et al., 2022), and learning-enhanced MaxWeighttype scheduling in multi-server systems (Yang et al., 2023). Other work combines Lyapunov-drift scheduling with adversarial bandit feedback (Huang et al., 2024). Most of these models do not simultaneously incorporate job-specific contextual service probabilities and nonpreemptive service commitments, which are the two central features of our setting.

Contextual queueing and logistic bandits. Context-aware queueing models allow service outcomes to depend on job or queue features. Kim and Oh (2024) study a multi-class multi-server queueing-matching problem with feature-dependent multinomial-logit service preferences. The closest models to ours are the contextual queueing bandits of Bae et al. (2026); Bae and Lee (2026). In the stochastic-context setting most directly comparable to ours, jobs arrive with random contexts, and in each round the scheduler selects a job–server pair whose departure probability is determined by a logistic model. The objective is to minimize the expected queue length at a terminal horizon, or equivalently to control queue-length regret relative to a known-model oracle. Their setting is preemptive: after an unsuccessful service attempt, the scheduler may select a diferent job–server pair in the next round. Consequently, when the service model is known, an optimal policy is work-conserving and selects, in each nonempty round, an available job–server pair with the largest departure probability.

Our nonpreemptive model difers in that, once a job–server pair is selected, the same pair must be served until the job departs. Hence the current decision constrains future actions, and the finite-horizon optimal policy need not admit the same myopic characterization. In addition, the stochastic-context guarantees of Bae et al. (2026); Bae and Lee (2026) use a pointwise traficslackness condition and a positive lower bound on a feature second-moment matrix, whereas our main single-server analysis uses the workload condition $\lambda \mathbb { E } [ 1 / p ( X ) ] < 1$ and does not require such a covariance lower bound.

Our logistic service model is also related to generalized linear and logistic bandits (Filippi et al., 2010; Li et al., 2017; Faury et al., 2020; Abeille et al., 2021).

Nonpreemptive stochastic scheduling and learning. Nonpreemptive scheduling has been extensively studied when service-time distributions are known. For fixed collections of jobs, $\mathrm { S E P T }$ and related priority rules are optimal under suitable stochastic-order assumptions for classical objectives such as expected flow time (Weber et al., 1986), while other priority rules can maximize the number of successful completions before a due date under additional structural conditions (Righter and Shanthikumar, 1989). More recently, learning has been incorporated into stochastic scheduling. Merlis et al. (2023) study single-machine scheduling when job-type duration distributions are initially unknown, considering both nonpreemptive and preemptive settings and comparing their learning costs. Zhong et al. (2025) study scheduling with unknown model primitives in multiclass many-server queues with abandonment.

Our setting difers from these works in several respects. Jobs arrive stochastically over time and carry individual contextual features, their service-completion probabilities are governed by an unknown logistic model, and the objective is the expected queue length at a finite terminal horizon. Moreover, under nonpreemption, each selected job occupies the server until completion, so the efect of a scheduling decision persists across multiple rounds and interacts with future arrivals. Consequently, neither the classical SEPT optimality results nor existing learning-to-schedule guarantees directly characterize the finite-horizon benchmark studied here.

## C Workload properties and stationary workload

This section proves the properties stated in Lemma 1 and establishes several additional properties of the stationary workload associated with the workload recursion. These results will be used later as key ingredients in the regret lower-bound analyses.

We begin by defining some constants introduced in Lemma 1. For $r > 0$ such that $( 1 - p _ { - } ) e ^ { r } < 1$ -， define

$$
M _ { G } ( r ) : = \mathbb { E } e ^ { r G ( X ) } = \mathbb { E } \frac { p ( X ) e ^ { r } } { 1 - ( 1 - p ( X ) ) e ^ { r } } , \qquad \psi ( r ) : = e ^ { - r } \big ( 1 - \lambda + \lambda M _ { G } ( r ) \big ) .
$$

Since $\psi ( 0 ) = 1$ and $\psi ^ { \prime } ( 0 ) = \rho - 1 < 0$ , there exists $r > 0$ satisfying both $\psi ( r ) < 1$ and $( 1 - p _ { - } ) e ^ { r } < 1$ Fix any such r and define

$$
K _ { r } : = \frac { 1 - \lambda + \lambda M _ { G } ( r ) } { 1 - \psi ( r ) } , \qquad C _ { W } ( r ) : = \frac { 1 } { r } \log K _ { r } .
$$

These five constants: $r \in ( 0 , 1 ) , \psi ( r ) \in ( 0 , 1 ) , M _ { G } ( r ) , K _ { r }$ and $C _ { W } ( r )$ are what we introduced in Lemma 1.

For convenience, we divide the conclusions of Lemma 1 into three items for the purposes of this appendix. We refer to the bounds $\mathbb { E } e ^ { r W _ { t } } \leq K _ { \tau }$ and $\mathbb { E } W _ { t } \le C _ { W } ( r )$ for every $t \geq 1$ as Item 1: uniformly bounded expected workload. We refer to the conditional workload bound in Equation (3), together with its unconditional consequence $\mathbb { E } [ W _ { u + h } \mathbf { 1 } \{ \tau _ { 0 } > h \} ] \le ( K _ { r } / r ) \psi ( r ) ^ { h }$ , as Item 2: workload tail bound. Finally, we refer to the existence of a stationary workload distribution $W$ that stochastically dominates $W _ { t }$ for every $t \geq 1$ as Item 3: stationary workload domination.

## C.1 Proofs of Items 1 and 2 of Lemma 1

Proof. Item 1: Uniformly bounded expected workload.

Let $Z _ { t } = A _ { t } G _ { t }$ be the workload brought by the arrival at the end of round $t ,$ with $Z _ { t } = 0$ when $A _ { t } = 0$ . Conditional on $A _ { t } = 1$ and $X _ { t } ,$ we have $Z _ { t } = G _ { t } \sim \operatorname { G e o m } ( p ( X _ { t } ) )$ . Since the arrival indicator is independent of the context and the latent service requirement,

$$
B _ { r } : = \mathbb { E } e ^ { r Z _ { t } } = 1 - \lambda + \lambda \mathbb { E } _ { X \sim D } e ^ { r G ( X ) } = 1 - \lambda + \lambda M _ { G } ( r ) .
$$

Here, $G ( X )$ is a random variable that satisfies $G ( X ) \sim \operatorname { G e o m } ( p ( X ) )$

If $W _ { t } = w \ge 1$ , the workload is reduced by one during round t and increased by $Z _ { t }$ at the end of the round. Hence

$$
\mathbb E ( e ^ { r W _ { t + 1 } } \mid W _ { t } = w ) = \mathbb E e ^ { r ( w - 1 + Z _ { t } ) } = e ^ { r ( w - 1 ) } B _ { r } = e ^ { r w } \psi ( r ) .
$$

If $W _ { t } = 0$ , no work is served and $W _ { t + 1 } = Z _ { t }$ . Splitting according to whether $W _ { t }$ is zero therefore gives

$$
\begin{array} { r l } & { \mathbb E e ^ { r W _ { t + 1 } } = \mathbb E \left[ e ^ { r W _ { t } } e ^ { - r } \mathbb E ( e ^ { r Z _ { t } } \mid W _ { t } ) \mathbf 1 \{ W _ { t } > 0 \} \right] + \mathbb E \left[ \mathbb E ( e ^ { r Z _ { t } } \mid W _ { t } ) \mathbf 1 \{ W _ { t } = 0 \} \right] } \\ & { \qquad = \psi ( r ) \mathbb E \left( e ^ { r W _ { t } } \mathbf 1 \{ W _ { t } > 0 \} \right) + B _ { r } \mathbb P ( W _ { t } = 0 ) } \\ & { \qquad \le \psi ( r ) \mathbb E e ^ { r W _ { t } } + B _ { r } . } \end{array}
$$

Starting from $W _ { 1 } = 0$ and applying this inequality successively,

$$
\begin{array} { l } { \displaystyle \mathbb { E } e ^ { r W _ { t } } \leq \psi ( r ) ^ { t - 1 } \mathbb { E } e ^ { r W _ { 1 } } + B _ { r } \sum _ { j = 0 } ^ { t - 2 } \psi ( r ) ^ { j } } \\ { = \psi ( r ) ^ { t - 1 } + B _ { r } \frac { 1 - \psi ( r ) ^ { t - 1 } } { 1 - \psi ( r ) } } \\ { = K _ { r } + \psi ( r ) ^ { t - 1 } ( 1 - K _ { r } ) \leq K _ { r } . } \end{array}
$$

Here we used $K _ { r } = B _ { r } / ( 1 - \psi ( r ) )$ , and the last inequality holds because $B _ { r } \geq 1$ and $0 < \psi ( r ) < 1$ so $K _ { r } = B _ { r } / ( 1 - \psi ( r ) ) \geq 1$ . Using Jensen’s inequality for the function $x \mapsto e ^ { r x }$ then yields

$$
\mathbb { E } W _ { t } \leq \frac { 1 } { r } \log \mathbb { E } e ^ { r W _ { t } } \leq \frac { 1 } { r } \log K _ { r } = C _ { W } ( r ) .
$$

Item 2: Workload tail bound.

Fix a positive integer u and let $\tau _ { 0 } = \operatorname* { i n f } \{ h \geq 0 : W _ { u + h } = 0 \}$ . On $\{ \tau _ { 0 } > h \}$ the workload is positive

through the first h updates. Since the set of events $\{ \tau _ { 0 } > h \}$ is equal to $\{ \tau _ { 0 } > h - 1 \} \cap \{ W _ { u + h } > 0 \}$ , we can get the following recurrence inequality.

$$
\begin{array} { r l } & { \mathbb { E } \left( e ^ { r W _ { u + h } } \mathbf { 1 } \{ \tau _ { 0 } > h \} \mid W _ { u } \right) } \\ & { \quad = \mathbb { E } \left[ \mathbf { 1 } \{ \tau _ { 0 } > h - 1 \} \mathbb { E } \left( e ^ { r W _ { u + h } } \mathbf { 1 } \{ W _ { u + h } > 0 \} \mid W _ { u + h - 1 } \right) \middle | W _ { u } \right] } \\ & { \quad \le \mathbb { E } \left[ \mathbb { E } \left( e ^ { r W _ { u + h } } \mid W _ { u + h - 1 } \right) \mathbf { 1 } \{ \tau _ { 0 } > h - 1 \} \middle | W _ { u } \right] } \\ & { \quad \le \psi ( r ) \mathbb { E } \left( e ^ { r W _ { u + h - 1 } } \mathbf { 1 } \{ \tau _ { 0 } > h - 1 \} \mid W _ { u } \right) } \end{array}
$$

Applying this inequality successively, we get

$$
\mathbb { E } \left( e ^ { r W _ { u + h } } \mathbf { 1 } \big \{ \tau _ { 0 } > h \big \} \mid W _ { u } \right) \le \mathbb { E } \left( e ^ { r W _ { u } } \mathbf { 1 } \big \{ \tau _ { 0 } > 0 \big \} \mid W _ { u } \right) \psi ( r ) ^ { h } \le e ^ { r W _ { u } } \psi ( r ) ^ { h } .
$$

Because $w \leq e ^ { r w } / r$ for $w \geq 0$ , we obtain

$$
\mathbb { E } \left( W _ { u + h } \mathbf { 1 } \{ \tau _ { 0 } > h \} \mid W _ { u } \right) \le \frac { e ^ { r W _ { u } } } { r } \psi ( r ) ^ { h } ,
$$

which is Equation (3). Finally, by taking the expectation of $W _ { u }$ , we obtain a useful inequality like below.

$$
\mathbb { E } \left( W _ { u + h } \mathbf { 1 } \big \{ \tau _ { 0 } > h \big \} \right) = \mathbb { E } \left[ \mathbb { E } \left( W _ { u + h } \mathbf { 1 } \big \{ \tau _ { 0 } > h \big \} \ | \ W _ { u } \right) \right] \leq \frac { \mathbb { E } ( e ^ { r W _ { u } } ) } { r } \psi ( r ) ^ { h } \leq \frac { K _ { r } } { r } \psi ( r ) ^ { h }\tag{13}
$$

## C.2 Upper bound of long busy period probability

We now show that the probability that a busy period lasts for more than h rounds is upper bounded by a function that decays exponentially in h. This requires a slightly diferent stopped workload because an arrival in the round that empties the system right before that arrival belongs to the next busy period. Suppose a busy period starts in round s (so in round $s - 1$ , the queue becomes empty after arrival and a new job arrives immediately later), and let G be the service time of its first job (which arrived at the end of round $s - 1 )$ . Set $\widetilde { W } _ { 0 } = G$ and, for $h \geq 0$ , define

$$
\widetilde { W } _ { h + 1 } = \left\{ \begin{array} { l l } { \widetilde { W } _ { h } - 1 + Z _ { s + h } , } & { \widetilde { W } _ { h } \geq 2 , } \\ { 0 , } & { \widetilde { W } _ { h } \leq 1 . } \end{array} \right.
$$

Thus $\widetilde { W } _ { h }$ is the amount of work belonging to this busy period after h service rounds (in other words, at the beginning of round $s + h )$ , and it is easy to check that $\widetilde { W } _ { h }$ remains zero after it hits zero for the first time. So if we define $\tau _ { \mathrm { b p } }$ as the smallest h such that the queue $\widetilde { W } _ { h }$ hits zero, then $\{ \tau _ { \mathrm { b p } } > h \} = \{ \widetilde { W } _ { h } > 0 \}$ holds. In particular, when $\widetilde { W } _ { h } = 1$ , the last unit is served in round $s + h$ and the process is stopped just before the arrival at the end of that round is added.

If $\tilde { W } _ { h } = w \ge 2$ , the same calculation as above gives

$$
\mathbb { E } \left( e ^ { r \widetilde { W } _ { h + 1 } } \mathbf { 1 } \{ \widetilde { W } _ { h + 1 } > 0 \} \mid \widetilde { W } _ { h } = w \right) = e ^ { r w } \psi ( r ) .
$$

If $w \leq 1$ , the left-hand side is zero. Using the same inductive argument that we used to prove Equation (3), we get

$$
\mathbb { E } \left( e ^ { r \widetilde { W } _ { h } } \mathbf { 1 } \{ \tau _ { \mathrm { b p } } > h \} \mid G \right) \le e ^ { r G } \psi ( r ) ^ { h } .\tag{14}
$$

Here, we used $W _ { 0 } = G$ . Since $e ^ { r \widetilde { W } _ { h } } \geq 1$ on $\{ \tau _ { \mathrm { b p } } > h \}$ 2

$$
\mathbb { P } ( \tau _ { \mathrm { b p } } > h \mid G ) \le e ^ { r G } \psi ( r ) ^ { h } .
$$

Averaging over G and using $\mathbb { E } e ^ { r G } = M _ { G } ( r )$ proves the desired result.

## C.3 Proof of Item 3 of Lemma 1 and Stationary Workload Identities

We now prove the existence of a stationary workload, which we refer to above as Item 3 of Lemma 1, using the assumption $\rho < 1$ . We then establish several properties of the stationary workload. All of these results are summarized in the following lemma.

Lemma 3. The recursion in Equation (2) admits a stationary distribution. Let W denote a random variable with this stationary distribution, and call it the stationary workload. Then the process started from zero and satisfying Equation (2) is stochastically dominated by it. Let $A \sim \mathrm { B e r n } ( \lambda )$ and, independently of A, let $X \sim \mathcal { D }$ and $G \mid X \sim \operatorname { G e o m } ( p ( X ) )$ . With $Z = A G$ 2

$$
\mathbb { P } ( W > 0 ) = \rho , \qquad \mathbb { E } W = \frac { \mathbb { E } Z ^ { 2 } + \rho - 2 \rho ^ { 2 } } { 2 ( 1 - \rho ) } .\tag{15}
$$

Proof. For every integer $w \ge 1$ , workload satisfies the following negative-drift inequality.

$$
\begin{array} { r l } & { \mathbb { E } [ W _ { t + 1 } - W _ { t } \mid W _ { t } = w ] = \mathbb { E } [ A _ { t } G ( X _ { t } ) ] - 1 } \\ & { \phantom { \mathbb { E } [ W _ { t + 1 } - W _ { t } \mid W _ { t } = w ] } = \lambda \mathbb { E } _ { X \sim \mathcal { D } , G ( X ) \sim \operatorname { G e o m } ( ( p ( X ) ) ) } [ G ( X ) ] - 1 } \\ & { \phantom { \mathbb { E } [ W _ { t + 1 } - W _ { t } \mid W _ { t } = w ] } = \lambda \mathbb { E } _ { X \sim \mathcal { D } } [ 1 / p ( X ) ] - 1 = \rho - 1 < 0 . } \end{array}
$$

From every finite state, a suficiently long run without arrivals reaches zero. The countable-state Foster–Lyapunov criterion therefore gives positive recurrence. The exponential drift in the proof of Lemma 1 gives a finite stationary exponential moment. Because w $) \mapsto ( w - 1 ) ^ { + } + Z _ { t }$ is nondecreasing, coupling a zero-started process and a stationary process with the same increments gives stochastic domination.

For the stationary workload W, let $W ^ { \prime } = ( W - 1 ) ^ { + } + Z$ , where $Z$ is independent of $W$ . Since $W ^ { \prime } \overset { d } { = } W$ by the definition of the stationary workload, taking expectations of $W , W ^ { \prime }$ gives

$$
0 = \mathbb { E } ( W ^ { \prime } - W ) = - \mathbb { P } ( W > 0 ) + \mathbb { E } Z ,
$$

and hence $\mathbb P ( W > 0 ) = \mathbb E Z = \rho$ . Moreover, since $( W - 1 ) ^ { + } = W - \mathbf { 1 } \{ W \geq 1 \}$ and $( ( W - 1 ) ^ { + } ) ^ { 2 } =$ $W ^ { 2 } - 2 W + \mathbf { 1 } \{ W \geq 1 \}$ ,

$$
\begin{array} { r } { { \mathbb { E } } ( W - 1 ) ^ { + } = { \mathbb { E } } W - \rho , \qquad { \mathbb { E } } ( ( W - 1 ) ^ { + } ) ^ { 2 } = { \mathbb { E } } W ^ { 2 } - 2 { \mathbb { E } } W + \rho . } \end{array}
$$

Squaring $W ^ { \prime } = ( W - 1 ) ^ { + } + Z$ , taking expectations, and using the independence of $W$ and $Z$ gives

$$
\begin{array} { r l } & { \mathbb { E } W ^ { 2 } = \mathbb { E } ( ( W - 1 ) ^ { + } ) ^ { 2 } + 2 \mathbb { E } ( ( W - 1 ) ^ { + } ) \mathbb { E } Z + \mathbb { E } Z ^ { 2 } } \\ & { \qquad = \mathbb { E } W ^ { 2 } - 2 \mathbb { E } W + \rho + 2 \rho ( \mathbb { E } W - \rho ) + \mathbb { E } Z ^ { 2 } . } \end{array}
$$

Canceling $\mathbb { E } W ^ { 2 }$ from both sides and collecting the terms in EW gives

$$
2 ( 1 - \rho ) \mathbb { E } W = \mathbb { E } Z ^ { 2 } + \rho - 2 \rho ^ { 2 } .
$$

Division by $2 ( 1 - \rho ) > 0$ proves Equation (15).

## D Estimating Departure Probabilities

This section proves the two bounds in Lemma 2. These inequalities will serve as key ingredients in the regret analysis.

## D.1 Proof of first item in Lemma 2

Proof. Let $z = x ^ { \top } \theta .$ . The gradient and Hessian of the logistic loss are

$$
\nabla \ell _ { \boldsymbol { \theta } } ( x , y ) = ( \mu ( z ) - y ) x , \qquad \nabla ^ { 2 } \ell _ { \boldsymbol { \theta } } ( x , y ) = \mu ^ { \prime } ( z ) x x ^ { \top } .
$$

Because $| \mu ( z ) - y | \leq 1$ and $\| x \| _ { 2 } \leq 1 , \| \nabla \ell _ { \theta } ( x , y ) \| _ { 2 } \leq 1$ . Thus the loss is 1-Lipschitz in $\theta .$ The parameter ball has diameter at most $2 S \le 2 \bar { S }$ , so the mean-value theorem gives

$$
| \ell _ { \theta } ( x , y ) - \ell _ { \theta ^ { \prime } } ( x , y ) | \leq 2 \bar { S } \qquad \mathrm { f o r ~ a l l ~ } \| \theta \| _ { 2 } , \| \theta ^ { \prime } \| _ { 2 } \leq S .
$$

We next verify exp-concavity. Since $| z | \leq S \leq \bar { S }$ , if $y = 0$ , then

$$
\frac { \mu ^ { \prime } ( z ) } { \mu ( z ) ^ { 2 } } = \frac { 1 - \mu ( z ) } { \mu ( z ) } = e ^ { - z } \geq e ^ { - \bar { S } } ;
$$

if $y = 1$ , then

$$
{ \frac { \mu ^ { \prime } ( z ) } { ( 1 - \mu ( z ) ) ^ { 2 } } } = { \frac { \mu ( z ) } { 1 - \mu ( z ) } } = e ^ { z } \geq e ^ { - { \bar { S } } } .
$$

In either case,

$$
\begin{array} { r } { \nabla ^ { 2 } \ell _ { \theta } ( x , y ) \succeq e ^ { - \bar { S } } \nabla \ell _ { \theta } ( x , y ) \nabla \ell _ { \theta } ( x , y ) ^ { \top } , } \end{array}
$$

which is the second-order characterization of $e ^ { - \bar { S } _ { - } } \mathrm { e x p - c o n c a v i t y }$ on the convex parameter ball.

Before applying the excess-risk theorem, observe that $\theta ^ { * }$ minimizes the population logistic risk over the parameter ball. Indeed, for every θ in this ball, since $Y \sim \operatorname { B e r n } ( p ( X ) )$ conditional on the context $X$ , the following equality about KL divergence holds.

$$
\begin{array} { r l } { \mathbb { E } \ell _ { \theta } ( X , Y ) - \mathbb { E } \ell _ { \theta ^ { * } } ( X , Y ) = \left( \mathbb { E } _ { X } \left[ \log \left( 1 + e ^ { X ^ { \top } \theta } \right) \right] - \mathbb { E } \left[ p ( X ) X ^ { \top } \theta \right] \right) } & { } \\ & { \phantom { = } - \left( \mathbb { E } _ { X } \left[ \log \left( 1 + e ^ { X ^ { \top } \theta ^ { * } } \right) \right] - \mathbb { E } \left[ p ( X ) X ^ { \top } \theta ^ { * } \right] \right) } \\ & { = \mathbb { E } _ { X } \left[ \log \left( \frac { 1 + e ^ { X ^ { \top } \theta } } { 1 + e ^ { X ^ { \top } \theta ^ { * } } } \right) - p ( X ) \log \left( \frac { e ^ { X ^ { \top } \theta } } { e ^ { X ^ { \top } \theta ^ { * } } } \right) \right] } \\ & { = \mathbb { E } _ { X } \left[ p ( X ) \log \left( \frac { 1 + e ^ { X ^ { \top } \theta } } { 1 + e ^ { X ^ { \top } \theta ^ { * } } } \right) + ( 1 - p ( X ) ) \log \left( \frac { 1 + e ^ { X ^ { \top } \theta } } { 1 + e ^ { X ^ { \top } \theta ^ { * } } } \right) \right] } \\ & { = \mathbb { E } _ { X } \left[ p ( X ) \log \left( \frac { p ( X ) } { \mu ( X ^ { \top } \theta ) } \right) + ( 1 - p ( X ) ) \log \left( \frac { 1 - p ( X ) } { 1 - \mu ( X ^ { \top } \theta ) } \right) \right] } \\ & { = \mathbb { E } _ { X } D _ { \mathrm { K L } } \left( \mathrm { R e r m } ( p ( X ) ) \right) \left. \mathrm { R e r m } ( \mu ( X ^ { \top } \theta ) ) \right. \geq 0 } \end{array}\tag{16}
$$

Since we estimate $\theta ^ { * }$ by $\widehat { \theta } _ { n }$ by using empirical risk minimization, we can apply Theorem 1 of Mehta (2017). In the notation of that theorem, the feasible set is a d-dimensional closed ball of radius S to which the unknown vector $\theta ^ { * }$ belongs, so its diameter is $2 S \le 2 \bar { S }$ . For every $( x , y )$ , the loss function $\ell _ { \boldsymbol { \theta } } ( x , y )$ viewed as a function of the variable θ has Lipschitz constant 1 as shown above. Its exp-concavity parameter is $e ^ { - \bar { S } }$ , and for every $( x , y ) , \theta , \theta ^ { * } , | \ell _ { \theta } ( x , y ) - \ell _ { \theta ^ { * } } ( x , y ) | \leq | \theta - \theta ^ { * } | \leq 2 \bar { S }$ holds because the loss function has Lipschitz constant 1. The theorem contains the factor $2 \bar { S } \vee e ^ { \bar { S } }$ , which equals $e ^ { \bar { S } }$ because $e ^ { \bar { S } } \geq 2 \bar { S }$ for $\bar { S } \geq 1$ . Its logarithmic term becomes log $\left( 1 6 \cdot 1 \cdot 2 \bar { S } \cdot n \right) = \log ( 3 2 \bar { S } n )$ Consequently, for every $n \geq 5$ , with probability at least $1 - \delta$

$$
\mathbb { E } \ell _ { \widehat { \theta _ { n } } } ( X , Y ) - \mathbb { E } \ell _ { \theta ^ { * } } ( X , Y ) \leq \frac { 8 e ^ { \bar { S } } \left( d \log ( 3 2 \bar { S } n ) + \log ( 1 / \delta ) \right) + 1 } { n } .\tag{17}
$$

By Equation (16), the left-hand side of Equation (17) equals the expected Bernoulli KL divergence over all $X \sim \mathcal { D }$ . Pinsker’s inequality gives $| p - q | \leq { \sqrt { D _ { \mathrm { K L } } ( \mathrm { B e r n } ( p ) \| \mathrm { B e r n } ( q ) ) / 2 } }$ . Applying Pinsker’s inequality pointwise for each X and then using the Cauchy-Schwarz inequality, we get the following inequality.

$$
\begin{array} { r l } & { \mathbb { E } _ { X } | p ( X ) - \mu ( X ^ { \top } \widehat \theta _ { n } ) | \leq \mathbb { E } _ { X } \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } ( \mathrm { B e r n } ( p ( X ) ) \| \mathrm { B e r n } ( \mu ( X ^ { \top } \widehat \theta _ { n } ) ) ) } } \\ & { \qquad \leq \sqrt { \mathbb { E } _ { X } [ \frac { 1 } { 2 } D _ { \mathrm { K L } } ( \mathrm { B e r n } ( p ( X ) ) \| \mathrm { B e r n } ( \mu ( X ^ { \top } \widehat \theta _ { n } ) ) ) ] \cdot \mathbb { E } _ { X } [ 1 ] } } \\ & { \qquad = \sqrt { \frac { 1 } { 2 } \mathbb { E } _ { X } D _ { \mathrm { K L } } ( \mathrm { B e r n } ( p ( X ) ) \| \mathrm { B e r n } ( \mu ( X ^ { \top } \widehat \theta _ { n } ) ) ) } } \\ & { \qquad = \sqrt { \frac { 1 } { 2 } ( \mathbb { E } \ell _ { \widehat \theta _ { n } } ( X , Y ) - \mathbb { E } \ell _ { \theta ^ { * } } ( X , Y ) ) } } \end{array}
$$

Projection onto $[ p _ { - } , p _ { + } ]$ cannot increase the absolute error because $p ( X )$ already lies in this interval. So $| p ( X ) - \widehat { p } ( \bar { X } ) | \leq | p ( X ) - \mu ( X ^ { \top } \widehat { \theta } _ { n } ) |$ always holds by the definition of ${ \widehat { p } } ( X )$ , and this gives $\begin{array} { r } { \mathbb { E } _ { X } | p ( X ) - \widehat { p } ( X ) | \leq \sqrt { \frac { 1 } { 2 } \left( \mathbb { E } \ell _ { \widehat { \theta _ { n } } } ( X , Y ) - \mathbb { E } \ell _ { \theta ^ { * } } ( X , Y ) \right) } } \end{array}$ by the above inequality. Substituting Equation (17) proves desired result. For $n < 5$ , the definition gives $\epsilon _ { n } ( \delta ) = 1$ , while $| { \widehat { p } } _ { n } ( X ) - p ( X ) | \leq$ 1. So this shows first item of Lemma 2 holds for every $n \geq 1$ □

## D.2 A simultaneous prefix bound

Corollary 1. For $\delta \in ( 0 , 1 )$ , define $\delta _ { n } = 6 \delta / ( \pi ^ { 2 } n ^ { 2 } )$ . With probability at least $1 - \delta$ , simultaneously for every $n \geq 1$

$$
\mathcal { E } ( \widehat { p } _ { n } ) \leq \epsilon _ { n } ( \delta _ { n } ) .
$$

Proof. For each fixed $n , { \mathcal { E } } ( { \widehat { p } } _ { n } ) \leq \epsilon _ { n } ( \delta _ { n } )$ fails with probability at most $\delta _ { n }$ by first item of Lemma 2. So the probability that there exists a natural number such that $\begin{array} { r } { \mathcal { E } ( \widehat { p } _ { n } ) \leq \epsilon _ { n } ( \delta _ { n } ) \mathrm { ~ d ~ } \sum _ { n > 1 } \delta _ { n } = \delta } \end{array}$ , a union bound proves the result. No stopping-time conditioning is used: all estimators are defined on the same infinite i.i.d. sequence of job-indexed observations $( X _ { i } , Y _ { i } ) _ { i \geq 1 }$ □

## D.3 Proof of second item in Lemma 2

Conditional on ${ \widehat { p } } _ { n }$ , the variables ${ \widehat { p } } _ { n } ( X _ { 1 } ^ { \prime } ) , \dots , { \widehat { p } } _ { n } ( X _ { n } ^ { \prime } )$ are i.i.d. with law $F _ { \widehat { p } _ { n } }$ and support in $[ p _ { - } , p _ { + } ]$ The Dvoretzky–Kiefer–Wolfowitz inequality (Dvoretzky et al., 1956) gives, with probability at least

$$
1 - \delta ,
$$

$$
\operatorname* { s u p } _ { z } | \widehat { F } _ { n } ( ( - \infty , z ] ) - F _ { \widehat { p } _ { n } } ( ( - \infty , z ] ) | \leq \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .
$$

For probability distributions supported on $[ p _ { - } , p _ { + } ]$ , the one-dimensional identity for $W _ { 1 }$ proves second item of Lemma 2 as follows.

$$
\begin{array} { r l r } {  { W _ { 1 } ( \widehat { F } _ { n } , F _ { \widehat { p } _ { n } } ) = \int _ { p _ { - } } ^ { p _ { + } } | \widehat { F } _ { n } ( ( - \infty , z ] ) - F _ { \widehat { p } _ { n } } ( ( - \infty , z ] ) | d z } } \\ & { } & \\ & { } & { \leq ( p _ { + } - p _ { - } ) \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } , } \end{array}
$$

Furthermore, couple ${ \widehat { p } } _ { n } ( X )$ and $p ( X )$ using the same independent $X \sim \mathcal { D }$ and applying the definition of Wasserstein distance, we get following inequality.

$$
W _ { 1 } ( F _ { \widehat { p } _ { n } } , F _ { p } ) = \operatorname* { i n f } _ { ( P , \widehat { P } ) \mathrm { ~ c o u p l i n g } } \mathbb { E } \big [ | P - \widehat { P } | \big ] \leq \mathbb { E } | p ( X ) - \widehat { p } _ { n } ( X ) | = \mathcal { E } ( \widehat { p } _ { n } ) .
$$

## E Known-Horizon Analysis

We now prove Theorem 1. The proof uses three ingredients; Lemma 4, Lemma 5, and Lemma 6. The first is used to compare the empirical Bellman model with the true system. The second ensures that the learning phase can collect a suficient amount of data to make an empirical distribution and an estimated probability function, with high probability. The third one compares the minimum expected queue length of a queue that starts with fewer remaining rounds to the minimum expected queue length of the T-round benchmark, and this allows us to calculate the regret upper bound correctly.

The first ingredient is the planning-transfer lemma. If the system assumes its unknown probability function and distribution to be $q : \mathbb { R } ^ { d }  [ p _ { - } , p _ { + } ]$ and $\widehat F$ and uses the Bellman recursion that follows $( q , { \widehat { F } } )$ (for convenience of exposition, we write this Bellman recursion as the empirical Bellman recursion) to find out the optimal decision in each round, this empirical Bellman recursion difers from the true optimal system (the system that makes its decision according to the true Bellman recursion) in two ways. First, for the probability function, it replaces $p ( X )$ by $q ( X )$ for every job. Second, for the distribution, it uses the distribution $\widehat F$ which is an estimated distribution of $F _ { q } = \operatorname { L a w } ( q ( X ) )$ , although the real distribution of departure probabilities of jobs is $F _ { p } = \operatorname { L a w } ( p ( X ) )$ ). By focusing on these probability/distribution diferences, we can find an upper bound of regret between a queue that follows the empirical Bellman recursion and a queue that follows the true Bellman recursion (i.e. a queue with the smallest expected queue length). Here, we upper bound the prediction error of $p , q$ by the $L ^ { 1 } ( D )$ distance of $p , q ,$ , and the distributional error by the Wasserstein distance between $F _ { q } , \widehat { F }$

For an h-round system that follows the probability function $q$ and starts empty, let $J _ { h } ^ { p } ( \pi )$ be the expected terminal queue under policy $\pi ,$ and let $V _ { h } ^ { p }$ be its minimum over the relevant policy class in the given setting. Then the following lemma holds for the estimated probability function $q$ and empirical distribution $\widehat F$

Lemma 4 (Planning transfer). Fix $q : \mathbb { R } ^ { d }  [ p _ { - } , p _ { + } ]$ before an h-round process starts with an empty queue. Let $F _ { q } = \operatorname { L a w } ( q ( X ) )$ , let $\widehat F$ be a distribution on $[ p _ { - } , p _ { + } ]$ , and let πb be an optimal policy following the Bellman recursion defined by $( q , { \widehat { F } } )$ . When πb is run in the true contextual model,

$$
J _ { h } ^ { p } ( \widehat { \pi } ) - V _ { h } ^ { p } \leq \frac { 2 h ^ { 2 } } { p _ { - } } \mathbb { E } | q ( X ) - p ( X ) | + \frac { ( h - 1 ) h ( h + 1 ) ( h + 2 ) } { 1 2 } W _ { 1 } ( F _ { q } , \widehat { F } ) .\tag{18}
$$

The result holds both in the idling-allowed setting and the work-conserving setting.

The second ingredient ensures that the FCFS learning phase has observed a complete prefix of jobs in arrival order. This matters because the statistical events are defined on the job-indexed sequence $( X _ { i } , Y _ { i } ) _ { i \geq 1 }$ , independently of the random time at which the algorithm happens to have served enough jobs.

Lemma 5 (Sample availability). Let $N _ { L }$ be the number of jobs that have started service by the end of round L, and let $n = \lfloor \lambda L / 8 \rfloor$ . Under FCFS,

$$
\mathbb { P } ( N _ { L } < 2 n ) \le e ^ { - \lambda L / 8 } + K _ { r } e ^ { - r \lambda L / 4 } .\tag{19}
$$

On $\{ N _ { L } \ge 2 n \}$ , the first 2n observations $( X _ { i } , Y _ { i } )$ recorded by the algorithm correspond to the first 2n jobs in arrival order.

The final ingredient is a monotonicity property of the finite-horizon empty-state value.

Lemma 6 (Empty-state horizon monotonicity). The minimum terminal cost over policies that may idle is nondecreasing when jobs are added to the initial state. In particular, for systems that start empty,

$$
V _ { h } ^ { \mathrm { I A } } \leq V _ { h + 1 } ^ { \mathrm { I A } } .\tag{20}
$$

The proofs of the above three lemmas will appear after the proof of Theorem 1.

## E.1 Proof of Theorem 1

Now, we start to prove Theorem 1.

Proof. Index the jobs in arrival order and let $( X _ { i } , Y _ { i } ) _ { i \geq 1 }$ denote their contexts and first-round departure indicators. This sequence is i.i.d. and does not depend on when the jobs begin service. Define

$$
{ \mathcal { G } } _ { 1 } = \{ N _ { L } \geq 2 n \} ,\tag{21}
$$

$$
\mathcal { G } _ { 2 } = \{ \mathcal { E } ( \widehat { p } _ { n } ) \leq \epsilon _ { n } ( \delta _ { T } ) \} ,\tag{22}
$$

$$
\mathcal { G } _ { 3 } = \{ W _ { 1 } ( \widehat { F } _ { n } , F _ { \widehat { p } _ { n } } ) \leq \eta _ { n } ( \delta _ { T } ) \} .\tag{23}
$$

The estimator in $\mathcal { G } _ { 2 }$ uses the first n observations, whereas $\mathcal { G } _ { 3 }$ uses the estimated probability function and contexts of jobs $n + 1 , \ldots , 2 n$ . Neither event is conditioned on $\mathcal { G } _ { 1 }$ . Hence, Lemmas 2 and 5 and a union bound give

$$
\begin{array} { r } { \mathbb { P } ( ( \mathcal { G } _ { 1 } \cap \mathcal { G } _ { 2 } \cap \mathcal { G } _ { 3 } ) ^ { c } ) \le \mathbb { P } ( \mathcal { G } _ { 1 } ^ { c } ) + \mathbb { P } ( \mathcal { G } _ { 2 } ^ { c } ) + \mathbb { P } ( \mathcal { G } _ { 3 } ^ { c } ) \le e ^ { - \lambda L / 8 } + K _ { r } e ^ { - r \lambda L / 4 } + 2 \delta _ { T } . } \end{array}\tag{24}
$$

Let $\mathcal { G } = \mathcal { G } _ { 1 } \cap \mathcal { G } _ { 2 } \cap \mathcal { G } _ { 3 }$ and define the first empty round in the planning window by

$$
\tau = \operatorname* { m i n } \{ s \in \{ L + 1 , \ldots , T \} : Q _ { s } = 0 \} ,
$$

with $\tau = \infty$ when the set is empty (i.e. after starting round $L ,$ the first round in which the queue hits zero at the end of that round is round $T ,$ or the queue never becomes empty at the end of the remaining rounds). On $\{ \tau \leq T \}$ , define $h = T - \tau + 1 \leq H$ . Conditional on the history at the beginning of round $\tau ,$ the queue is empty. Thus, on $\mathcal { G } \cap \{ \tau \leq T \}$ , the arrivals, contexts, and departure indicators from round $\tau$ onward are independent of the first $2 n$ job-indexed observations used for estimation, since those $2 n$ jobs have already completed before the start of round $\tau .$

On $\mathcal { G } \cap \{ \tau \leq T \}$ , set $q = \widehat { p } _ { n }$ . After the queue hits zero at the beginning of round $\tau .$ , the final queue length $Q _ { T + 1 }$ is only related to the arrival, departure and job selections in rounds $\tau , . . . , T$ , and all the previous ones are completely independent of it. Since the Bellman policy used after the start of $\tau$ is optimal under $( q , \widehat { F } _ { n } )$ , applying Lemma 4 and then Lemma 6 shows that the conditional terminal cost $J _ { h } ^ { p } ( \widehat { \pi } )$ is at most

$$
\begin{array} { r l } { \displaystyle V _ { h } ^ { \mathrm { I A } } + \frac { 2 h ^ { 2 } } { p _ { - } } \epsilon _ { n } ( \delta _ { T } ) + \frac { ( h - 1 ) h ( h + 1 ) ( h + 2 ) } { 1 2 } \eta _ { n } ( \delta _ { T } ) \le V _ { T } ^ { \mathrm { I A } } + \frac { 2 H ^ { 2 } } { p _ { - } } \epsilon _ { n } ( \delta _ { T } ) } & { } \\ { \displaystyle + \frac { ( H - 1 ) H ( H + 1 ) ( H + 2 ) } { 1 2 } \eta _ { n } ( \delta _ { T } ) . } \end{array}\tag{25}
$$

On $\mathcal { G } ^ { c } \cap \{ \tau \leq T \}$ , the queue is empty at the start of round $\tau ,$ and hence there are only $h \leq H$ rounds remaining. During these remaining rounds, at most H arrivals can occur, so $Q _ { T + 1 } \leq H$ regardless of the planning model.

It remains to control the event $\{ \tau = \infty \}$ . Before planning, Algorithm 1 is work conserving and, in the event $\{ \tau = \infty \}$ , the queue remains non-empty at the beginning of rounds $L + 1 , . . . , T ;$ our queue must follow a work-conserving policy starting from round $L + 1$ to round T. Recall $Q _ { s } = 0$ if and only if $W _ { s } = 0$ in a work-conserving policy. If we define $\tau _ { 0 } = \operatorname* { i n f } \{ j \ge 0 : W _ { L + 1 + j } = 0 \}$ , on $\{ \tau = \infty , W _ { T + 1 } > 0 \}$ , the workload has not reached zero in any of the first H updates, and hence $\tau _ { 0 } > H$ . On $\{ \tau = \infty , W _ { T + 1 } = 0 \}$ , the terminal queue is zero. Consequently, Equation (3) gives

$$
\begin{array} { r l } { \mathbb { E } \left( Q _ { T + 1 } \mathbf { 1 } \{ \tau = \infty \} \right) } & { \leq \mathbb { E } \left( W _ { T + 1 } \mathbf { 1 } \{ \tau = \infty \} \right) } \\ & { = \mathbb { E } \left( W _ { T + 1 } \mathbf { 1 } \{ \tau = \infty , W _ { T + 1 } > 0 \} \right) } \\ & { \leq \mathbb { E } \left( W _ { T + 1 } \mathbf { 1 } \{ \tau _ { 0 } > H \} \right) } \\ & { \leq \displaystyle \frac { \psi ( r ) ^ { H } } { r } \mathbb { E } e ^ { r W _ { L + 1 } } \leq \frac { K _ { r } } { r } \psi ( r ) ^ { H } . } \end{array}\tag{26}
$$

For clarity, let

$$
\Gamma _ { H } = \frac { 2 H ^ { 2 } } { p _ { - } } \epsilon _ { n } ( \delta _ { T } ) + \frac { ( H - 1 ) H ( H + 1 ) ( H + 2 ) } { 1 2 } \eta _ { n } ( \delta _ { T } ) .
$$

For an event $E _ { \mathrm { { i } } }$ define the event-restricted terminal cost

$$
J _ { T } ^ { \pi } ( E ) : = \mathbb { E } \big [ Q _ { T + 1 } ^ { \pi } \mathbf { 1 } _ { E } \big ] .
$$

Then we have

$$
J _ { T } ^ { \mathrm { \pi ^ { L C P } } } = J _ { T } ^ { \mathrm { \pi ^ { L C P } } } ( \mathcal { G } \cap \{ \tau \leq T \} ) + J _ { T } ^ { \pi ^ { \mathrm { L C P } } } ( \mathcal { G } ^ { c } \cap \{ \tau \leq T \} ) + J _ { T } ^ { \pi ^ { \mathrm { L C P } } } ( \{ \tau = \infty \} ) .
$$

We bound these three terms separately by the properties we’ve shown above.

For each event $E \in \mathcal { G } \cap \{ \tau \leq T \}$ , we define $J _ { h } ^ { p } ( \widehat { \pi } , E )$ (here $h = T - \tau + 1 )$ as the conditional terminal cost of the queue when the first $1 , . . . , \tau - 1$ rounds have happened as E. In other words, $J _ { h } ^ { p } ( \widehat { \pi } , E ) = \mathbb { E } \left( Q _ { T + 1 } ^ { \pi ^ { \mathrm { L C P } } } \mid E \right)$ . The queue has been cleared by the beginning of round $\tau ,$ so the planning phase starts from an empty queue with at most H rounds remaining. So by Equation (25), $J _ { h } ^ { p } ( \widehat { \pi } , E ) \le V _ { T } ^ { \mathrm { I A } } + \Gamma _ { H }$ holds. Hence,

$$
\begin{array} { r l } & { J _ { T } ^ { \pi ^ { \mathrm { L C P } } } ( \mathcal G \cap \left\{ \tau \leq T \right\} ) = \mathbb E \left[ \mathbb E \left( Q _ { T + 1 } ^ { \pi ^ { \mathrm { L C P } } } \mid E \right) \mathbf 1 ( E \in \mathcal G \cap \left\{ \tau \leq T \right\} ) \right] } \\ & { \qquad = \mathbb E \left[ J _ { h } ^ { p } ( \widehat \pi , E ) \mathbf 1 ( E \in \mathcal G \cap \left\{ \tau \leq T \right\} ) \right] } \\ & { \qquad \leq \mathbb E \left[ \left( V _ { T } ^ { \mathrm { I A } } + \Gamma _ { H } \right) \mathbf 1 ( E \in \mathcal G \cap \left\{ \tau \leq T \right\} ) \right] } \\ & { \qquad = \left( V _ { T } ^ { \mathrm { I A } } + \Gamma _ { H } \right) \mathbb P ( \mathcal G \cap \left\{ \tau \leq T \right\} ) . } \end{array}
$$

For $\mathcal { G } ^ { c } \cap \{ \tau \leq T \}$ , as we’ve shown that $Q _ { T + 1 } ^ { \pi ^ { \mathrm { L C P } } } \leq H$ holds regardless of the planning model, we can get the following inequality directly.

$$
J _ { T } ^ { \pi ^ { \mathrm { L C P } } } ( \mathcal { G } ^ { c } \cap \{ \tau \leq T \} ) \leq H \mathbb { P } ( \mathcal { G } ^ { c } \cap \{ \tau \leq T \} ) .
$$

Finally, on $\{ \tau = \infty \}$ , the following inequality holds by Equation (26).

$$
J _ { T } ^ { \pi ^ { \mathrm { L C P } } } ( \{ \tau = \infty \} ) \le \frac { K _ { r } } r \psi ( r ) ^ { H } .
$$

Combining the three bounds gives

$$
\begin{array} { r l } & { J _ { T } ^ { \pi ^ { \mathrm { L C P } } } \leq \left( V _ { T } ^ { \mathrm { I A } } + \Gamma _ { H } \right) \mathbb { P } ( \mathcal { G } \cap \{ \tau \leq T \} ) + H \mathbb { P } ( \mathcal { G } ^ { c } \cap \{ \tau \leq T \} ) + \displaystyle \frac { K _ { r } } { r } \psi ( r ) ^ { H } . } \\ & { \qquad \leq V _ { T } ^ { \mathrm { I A } } + \Gamma _ { H } + H \mathbb { P } ( \mathcal { G } ^ { c } ) + \displaystyle \frac { K _ { r } } { r } \psi ( r ) ^ { H } . } \end{array}
$$

Subtracting $V _ { T } ^ { \mathrm { I A } }$ and using $V _ { T } ^ { \mathrm { I A } } \geq 0$ yields

$$
\begin{array} { r l } & { R _ { T } ^ { \mathrm { I A } } ( \pi ^ { \mathrm { L C P } } ) = J _ { T } ^ { \pi ^ { \mathrm { L C P } } } - V _ { T } ^ { \mathrm { I A } } } \\ & { \qquad \le \Gamma _ { H } + H \mathbb { P } ( \mathcal { G } ^ { c } ) + \displaystyle \frac { K _ { r } } { r } \psi ( r ) ^ { H } . } \end{array}
$$

Substituting Equation (24) proves Equation (8).

For $H \ = \ \lceil \log ^ { 2 } ( e T ) \rceil$ and suficiently large T, $L \ge T / 2$ and $n \ge \lambda T / 3 2$ . Hence $\epsilon _ { n } ( \delta _ { T } ) =$ $\widetilde { O } ( { \sqrt { d / ( \lambda T ) } } )$ and $\eta _ { n } ( \delta _ { T } ) = { \widetilde O } ( ( \lambda T ) ^ { - 1 / 2 } )$ The factors $H ^ { 2 }$ and $( H \mathrm { ~ - ~ } 1 ) H ( H + 1 ) ( H + 2 )$ are polylogarithmic. Since $\psi ( r ) < 1$ , the workload and sampling-tail terms are smaller than any inverse polynomial in $T$ . This proves the fact that regret is $\widetilde { O } ( { \sqrt { d / ( \lambda T ) } } )$ . An upper bound on the total number of computations required to evaluate the Bellman recursions during the planning stage is given in Appendix E.2. □

## E.2 Total Bellman Computation Cost during the Plan Phase

We now bound the total number of Bellman terms evaluated during the planning stage. Suppose that the planning stage starts with h rounds remaining and with empirical distribution ${ \widehat { F } } _ { n } .$ , where $h \leq H = T - L$ . During planning, the algorithm recomputes the empirical Bellman recursion from the current state. For $s = 1 , \ldots , h$ , let $B _ { s }$ be the number of elementary Bellman terms evaluated when there are s rounds remaining and the server is free. We bound $\textstyle \sum _ { s = 1 } ^ { h } B _ { s }$

It is useful to view one evaluation of the Bellman recursion as a finite directed acyclic computation graph (DAG). A node at depth k represents a reachable state with $s - k$ rounds remaining, and directed edges point from a state to the one-step successor states appearing on the right-hand side of the Bellman recursion (here, Bellman recursion means the recursion formulas of $V _ { k } ( c , a ) , ( A V _ { k } ) ( c , a ) )$ The terminal nodes, corresponding to states of 0 rounds remaining, have values given directly by the terminal queue length and therefore do not require any Bellman update. Thus $B _ { s }$ is bounded by the number of outgoing edges from nonterminal nodes in this computation graph.

Since the planning stage starts from an empty queue, at the beginning of a round with s rounds remaining, at most $h - s$ jobs have arrived during the planning stage. Thus the system contains at most $h - s$ jobs. Also, at most $h - s$ fitted departure-probability labels outside the original support of ${ \widehat { F } } _ { n }$ can have appeared as present-only labels. These labels are added to the finite label set used in the Bellman recursion and are assigned zero future-arrival mass, so the distribution ${ \widehat { F } } _ { n }$ itself does not change. Therefore, if m denotes the size of the finite label set used in this Bellman computation, then

$$
m \leq n + h - s .
$$

In this Bellman computation, future arrivals are assumed to have departure-probability labels in this fixed finite label set. Hence every reachable state can be represented as $( c , a )$ , where $c \in \mathbb { Z } _ { > 0 } ^ { m }$ is the count vector and $a \in \{ 0 , 1 , \ldots , m \}$ denotes the label of the job in service, with $a = 0$ representing a free server. After k further rounds of the recursion, $k = 0 , \ldots , s - 1$ , the total number of jobs present is at most $h - s + k$ . Thus the number of possible count vectors at that depth is at most

$$
\# \{ c \in \mathbb { Z } _ { \geq 0 } ^ { m } : \| c \| _ { 1 } \leq h - s + k \} = { \binom { m + h - s + k } { m } } .
$$

For each count vector, the service label a has at most $m + 1$ possible values. Therefore the number of nonterminal reachable states in this Bellman computation is at most

$$
( m + 1 ) \sum _ { k = 0 } ^ { s - 1 } { \binom { m + h - s + k } { m } } \leq ( m + 1 ) { \binom { m + h } { m + 1 } } ,
$$

where the last inequality follows from the hockey-stick identity.

For each nonterminal reachable state, evaluating its value requires at most $m + 1$ values of the form $\mathcal { A } V$ at the next step, and each such value depends on at most $m + 1$ next-stage values of V . Thus, in the corresponding computation graph, each nonterminal node has outdegree at most $( m + 1 ) ^ { 2 }$ . Hence,

$$
B _ { s } \leq ( m + 1 ) ^ { 3 } { \binom { m + h } { m + 1 } } .
$$

Using $m \leq n + h - s$ , we obtain

$$
B _ { s } \leq ( n + h - s + 1 ) ^ { 3 } { \binom { n + 2 h - s } { h - 1 } } \leq ( n + h + 1 ) ^ { 3 } { \binom { n + 2 h - s } { h - 1 } }
$$

Consequently,

$$
\begin{array} { c } { \displaystyle \sum _ { s = 1 } ^ { h } B _ { s } \leq ( n + h + 1 ) ^ { 3 } \sum _ { s = 1 } ^ { h } \binom { n + 2 h - s } { h - 1 } } \\ { \leq ( n + h + 1 ) ^ { 3 } \binom { n + 2 h } { h } . } \end{array}\tag{27}
$$

Since $h \leq H$ , the total number of elementary Bellman terms evaluated during the whole planning stage is at most

$$
( n + H + 1 ) ^ { 3 } { \binom { n + 2 H } { H } } .
$$

When $n = \Theta ( T )$ and $H = \Theta ( \log ^ { 2 } T )$ , the logarithm of this bound is $O ( H \log T ) = O ( \log ^ { 3 } T )$

## E.3 Proof of Lemma 4

We first compare two Bellman recursions that may difer in two ways: the two systems may assign diferent departure probabilities to the same present jobs, and the departure-probability distributions of future arrivals may be diferent. This slightly more general comparison will be used later with suitable specializations.

Let $V _ { k } ^ { F } ( s )$ be the minimum expected terminal queue length from state s with k rounds remaining, when the departure probability of each future arrival is drawn from F. Equivalently, $V _ { k } ^ { F } ( s )$ is the value function obtained from the Bellman recursion for the model that uses the departure probabilities recorded in the current state for present jobs and draws the departure probability of each future arrival from F. Let $Q _ { k } ^ { F } ( s , a )$ be the corresponding one-step value obtained by taking an admissible action a in state s and then acting optimally. Here admissibility includes the nonpreemptive constraint: if a job is currently in service, the only admissible action is to continue serving that job. For a policy π and an empty initial state, write $J _ { h } ^ { F } ( \pi )$ for its expected terminal queue length, and write $V _ { h } ^ { F }$ for the corresponding minimum over the relevant policy class.

We compare two states s and se whose currently present jobs are paired, and whose jobs in service are paired as well. A paired present job may carry diferent departure-probability labels in the two states: we denote its label by $p _ { j }$ in s and by $\widetilde { p } _ { j }$ in se. Define

$$
\Delta ( s , \widetilde { s } ) : = \sum _ { j \mathrm { ~ p r e s e n t } } | p _ { j } - \widetilde { p } _ { j } | .
$$

Lemma 7. Let F and $\widetilde { F }$ be distributions of the departure probabilities of future jobs, with $W _ { 1 } ( F , \widetilde { F } ) =$ η. For states s and se described above and every $k \geq 0$

$$
| V _ { k } ^ { F } ( s ) - V _ { k } ^ { \widetilde F } ( \widetilde s ) | \le \frac { k ( k + 1 ) } { 2 } \Delta ( s , \widetilde s ) + \frac { ( k - 1 ) k ( k + 1 ) } { 6 } \eta .\tag{28}
$$

The bound holds both when policies may idle and when they must be work conserving.

Proof. For convenience of explanation, we define $\begin{array} { r } { a _ { k } : = \frac { k ( k + 1 ) } { 2 } , g _ { k } : = \frac { ( k - 1 ) k ( k + 1 ) } { 6 } } \end{array}$ . We will prove our lemma by induction on k. For $k = 0$ , both terminal values equal the common number of jobs in the two states, so Equation (28) holds because $a _ { 0 } = g _ { 0 } = 0$ . Assume it holds with $k - 1$ rounds remaining.

Take any action that is admissible in both states. If the action serves a job which is in both of $s , { \widetilde s } ,$ let $p$ and $\widetilde { p }$ be its departure probabilities and put $\delta = | p - \widetilde { p } |$ . Use a common uniform random variable $U \sim \mathrm { U n i f } [ 0 , 1 ]$ to generate the two departure indicators. If the selected job has departure probabilities $p$ and $\widetilde { p }$ in the systems started with $s , { \widetilde s }$ respectively, set

$$
D = { \bf 1 } \{ U \leq p \} , \qquad \widetilde { D } = { \bf 1 } \{ U \leq \widetilde { p } \} .
$$

Then $D \sim \operatorname { B e r n } ( p ) , { \tilde { D } } \sim \operatorname { B e r n } ( { \tilde { p } } )$ , and

$$
\mathbb { P } ( D \neq \widetilde { D } ) = | p - \widetilde { p } | = \delta .
$$

Use the same arrival indicator $A \sim \mathrm { B e r n } ( \lambda )$ in both systems. Conditional on $A = 1$ , choose an optimal coupling $( P , \tilde { P } )$ of $F$ and $\widetilde { F }$ such that

$$
\begin{array} { r } { \mathbb { E } | P - \widetilde { P } | = W _ { 1 } ( F , \widetilde { F } ) = \eta . } \end{array}
$$

Existence of this coupling comes from the definition of Wasserstein distance. The two arriving jobs are then paired and assigned departure-probability labels $P$ and ${ \cal \tilde { P } } ,$ respectively. If $A = 0$ , no new job is added to either system.

First, we see the case when the two departure indicators agree, i.e. both 0 or both 1. In both cases, the successor states then contain the same set of jobs, so we can use the induction hypothesis in both cases because they have the same set of jobs. If both indicators are 1, its contribution δ is removed from $\Delta ( s , \widetilde s )$ ; if both indicators are 0, that contribution is retained. A new arrival adds $| P - \widetilde { P } |$ in both cases. In either case, the expected sum of the probability diferences in the successor states is at most $\Delta ( s , \widetilde { s } ) + \eta$ . The induction hypothesis therefore bounds the expected diference of their continuation values by

$$
a _ { k - 1 } ( \Delta ( s , \widetilde { s } ) + \eta ) + g _ { k - 1 } \eta = a _ { k - 1 } \Delta ( s , \widetilde { s } ) + ( a _ { k - 1 } + g _ { k - 1 } ) \eta .\tag{29}
$$

We now see the case when the two departure indicators are diferent. WLOG let $p \leq \widetilde { p } ,$ so in the queue started with state s the departure indicator was 1, but in the queue started with state s the departure indicator was 0. On their disagreement event, one successor state has exactly one more job than the other. Over the remaining $k - 1$ rounds, the terminal queue diference caused by adding or deleting one job in the starting queue is at most k: the initial job-count diference is one, and the two systems can difer by at most one departure in each remaining round. The disagreement event has probability $\delta ,$ so its contribution is at most kδ. Combining this contribution with Equation (29) and using $\delta \le \Delta ( s , \widetilde s )$ gives

$$
| Q _ { k } ^ { F } ( s , a ) - Q _ { k } ^ { \widetilde { F } } ( \widetilde { s } , a ) | \leq ( a _ { k - 1 } \Delta ( s , \widetilde { s } ) + ( a _ { k - 1 } + g _ { k - 1 } ) \eta ) + k \Delta ( s , \widetilde { s } )\tag{30}
$$

$$
= a _ { k } \Delta ( s , \widetilde { s } ) + g _ { k } \eta ,\tag{31}
$$

where $a _ { k } = a _ { k - 1 } + k$ and $g _ { k } = g _ { k - 1 } + a _ { k - 1 }$ . If the common action is to idle, no departure indicator is generated, and at the beginning of the next round the two queues have the same set of jobs. So by using the induction hypothesis again, it’s not dificult to show the same bound holds without the kδ term.

Let $a ^ { * }$ minimize $Q _ { k } ^ { F } ( s , a )$ and let $\widetilde { a } ^ { * }$ minimize $Q _ { k } ^ { \widetilde { F } } ( \widetilde { s } , a )$ (so that $Q _ { k } ^ { F } ( s , a ^ { * } ) = V _ { k } ^ { F } ( s ) , Q _ { k } ^ { \widetilde F } ( \widetilde s , a ) =$ $V _ { k } ^ { \widetilde { F } } ( \widetilde { s } ) \big >$ . Of course these $a ^ { * } , \tilde { a } ^ { * }$ denote the job in service when the server is not free in the starting state s. Applying Equation (31) to $\widetilde { a } ^ { * }$ gives

$$
V _ { k } ^ { F } ( s ) - V _ { k } ^ { \widetilde { F } } ( \widetilde { s } ) \le Q _ { k } ^ { F } ( s , \widetilde { a } ^ { * } ) - Q _ { k } ^ { \widetilde { F } } ( \widetilde { s } , \widetilde { a } ^ { * } ) \le a _ { k } \Delta ( s , \widetilde { s } ) + g _ { k } \eta .
$$

Applying it to $a ^ { * }$ with the two models interchanged gives the reverse inequality. This proves Equation (28). □

Setting $s = { \widetilde s }$ (so all the probabilities are equal in the initial queue) in Lemma 7 gives

$$
| V _ { k } ^ { F } ( s ) - V _ { k } ^ { \widetilde { F } } ( s ) | \le g _ { k } \eta .
$$

The common-action bound Equation (31), again with $s = { \widetilde { s } } ,$ gives $| Q _ { k } ^ { F } ( s , a ) - Q _ { k } ^ { \widetilde F } ( s , a ) | \le g _ { k } \eta _ { \gamma }$ for every admissible action a. Let $\widetilde { \pi } _ { k } ( s )$ be an optimal action with k rounds remaining under $\widetilde { F }$ (so we can think that $\widetilde { \pi } _ { k } ( s )$ is the optimal decision according to the Bellman recursion defined by the starting queue $s ,$ the given success probability for each job, and the future success-probability distribution $\widetilde { F } )$ . Of course it denotes the job in service when the server is not free in the starting state s. Since $Q _ { k } ^ { \widetilde { F } } ( s , \widetilde { \pi } _ { k } ( s ) ) = V _ { k } ^ { \widetilde { F } } ( s )$

$$
\begin{array} { r l } & { Q _ { k } ^ { F } \big ( s , \widetilde { \pi } _ { k } ( s ) \big ) - V _ { k } ^ { F } ( s ) = Q _ { k } ^ { F } \big ( s , \widetilde { \pi } _ { k } ( s ) \big ) - Q _ { k } ^ { \widetilde { F } } \big ( s , \widetilde { \pi } _ { k } ( s ) \big ) } \\ & { \qquad + V _ { k } ^ { \widetilde { F } } \big ( s \big ) - V _ { k } ^ { F } \big ( s \big ) } \\ & { \qquad \leq 2 g _ { k } \eta . } \end{array}
$$

Let $S _ { k }$ be the state with k rounds remaining when $\widetilde { \pi }$ (recall that this policy selects its decision according to the Bellman recursion, which is defined by the given success probability for each job, and the future success-probability distribution $\widetilde { F } )$ is run in the model with future-job distribution F. By the definition of $Q _ { k } ^ { F }$

$$
\begin{array} { r } { Q _ { k } ^ { F } ( S _ { k } , \pi _ { k } ( S _ { k } ) ) = \operatorname { \mathbb { E } } \left( V _ { k - 1 } ^ { F } ( S _ { k - 1 } ) \mid S _ { k } \right) . } \end{array}
$$

So $\begin{array} { r } { \mathbb { E } \left( Q _ { k } ^ { F } ( S _ { k } , \widetilde { \pi } _ { k } ( S _ { k } ) ) \right) = \mathbb { E } \left[ \mathbb { E } \left( V _ { k - 1 } ^ { F } ( S _ { k - 1 } ) \mid S _ { k } \right) \right] = \mathbb { E } \left( V _ { k - 1 } ^ { F } ( S _ { k - 1 } ) \right) } \end{array}$ holds. And also by the definition and assumptions of Lemma 4, $S _ { h } = \mathcal { O } , \mathbb { E } \left( V _ { h } ^ { F } ( \mathcal { O } ) \right) = V _ { h } ^ { F }$ and $J _ { h } ^ { F } ( \widetilde { \pi } ) \ = \ \mathbb { E } ( S _ { 0 } ) \ =$ $\mathbb { E } \left( V _ { 0 } ^ { F } ( S _ { 0 } ) \right)$ hold. Taking expectations and summing over $k = h , h - 1 , \ldots , 1$ telescopes:

$$
\begin{array} { r l } & { { \mathcal { J } _ { h } ^ { F } } ( \widetilde { \pi } ) - { V _ { h } ^ { F } } = \displaystyle \sum _ { k = 1 } ^ { h } \Big ( \mathbb { E } \left( { V } _ { k - 1 } ^ { F } ( S _ { k - 1 } ) \right) - \mathbb { E } \left( { V } _ { k } ^ { F } ( S _ { k } ) \right) \Big ) } \\ & { \qquad = \displaystyle \sum _ { k = 1 } ^ { h } \Big ( \mathbb { E } \left( { Q } _ { k } ^ { F } ( S _ { k } , \widetilde { \pi } _ { k } ( S _ { k } ) ) \right) - \mathbb { E } \left( { V } _ { k } ^ { F } ( S _ { k } ) \right) \Big ) } \\ & { \qquad = \displaystyle \sum _ { k = 1 } ^ { h } \mathbb { E } \left( { Q } _ { k } ^ { F } ( S _ { k } , \widetilde { \pi } _ { k } ( S _ { k } ) ) - { V } _ { k } ^ { F } ( S _ { k } ) \right) } \\ & { \qquad \leq 2 \eta \displaystyle \sum _ { k = 1 } ^ { h } g _ { k } = 2 \eta \left( \displaystyle \sum _ { 4 } ^ { h + 2 } \right) = \displaystyle \frac { ( h - 1 ) h ( h + 1 ) ( h + 2 ) } { 1 2 } \eta . } \end{array}\tag{32}
$$

We next control the change from fitted to true departure probabilities for the same contexts.

Lemma 8. For $p , q \in [ p _ { - } , p _ { + } ]$ , there is a coupling of $G _ { p } \sim \operatorname { G e o m } ( p )$ and $G _ { q } \sim \operatorname { G e o m } ( q )$ such that

$$
\mathbb { P } ( G _ { p } \neq G _ { q } ) = \frac { | p - q | } { \operatorname* { m a x } ( p , q ) } \leq \frac { | p - q | } { p _ { - } } .
$$

Proof. Let $U _ { 1 } , U _ { 2 } , . . .$ . be i.i.d. uniform random variables and set $G _ { p } \ = \ \operatorname* { i n f } \{ g \ : \ U _ { g } \ \leq \ p \}$ and $G _ { q } = \operatorname* { i n f } \{ g : U _ { g } \leq q \}$ . If $p \leq q$ , then at the first time g such that $U _ { g } \leq q$ holds, the service times agree exactly when $U _ { g } \leq p ,$ so this event has conditional probability $p / q$ . Thus $\mathbb { P } ( G _ { p } \neq G _ { q } ) =$ $1 - p / q = ( q - p ) / q$ . The other case is symmetric. □

For this part, $J _ { h } ^ { q } ( \pi )$ and $V _ { h } ^ { q }$ denote the cost of a fixed policy $\pi$ and the optimal value when the same contexts are used but every departure probability $p ( X )$ is replaced by $q ( X )$ . Fix $q$ before an h-round process starts empty, and use the same arrivals and contexts in the true and fitted systems (i.e. systems using the same distribution of context vectors, but diferent departure probability functions). Couple the two service times of each arriving job using Lemma 8. Define $\mathcal { T } _ { h }$ as the set of jobs arriving during these h rounds, and define H as the set of events such that there is at least one job such that it started to be served in both queues during the h rounds, but the service times of that job in the two queues were diferent. Then, conditional on their contexts and $\mathcal { T } _ { h }$ , each job $X _ { i } \in \mathcal { T } _ { h }$ has service time disagreement probability at most $\frac { | p ( X _ { i } ) - q ( X _ { i } ) | } { p _ { - } }$ . So a union bound gives

$$
\mathbb { P } ( \mathcal { H } \mid \mathcal { T } _ { h } ) \leq \frac { 1 } { p _ { - } } \sum _ { i \in \mathcal { T } _ { h } } | p ( X _ { i } ) - q ( X _ { i } ) | .
$$

Because $q$ is fixed before the process begins and each arriving context has distribution $\mathcal { D } _ { : }$

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \sum _ { i \in \mathcal { I } _ { h } } | p ( X _ { i } ) - q ( X _ { i } ) | = \mathbb { E } \left[ | \mathbb { Z } _ { h } | \cdot \mathbb { E } _ { X \sim \mathcal { D } } | p ( X ) - q ( X ) | \right] } & { } \\ { = \lambda h \mathbb { E } _ { X \sim \mathcal { D } } | p ( X ) - q ( X ) | \leq h \mathcal { E } ( q ) . } \end{array}
$$

Thus the probability that there exists a job that is served in both queues during the h rounds and the service time of that job is diferent in the two queues is at most $h \mathcal { E } ( q ) / p .$ <sub>−</sub>. If every pair of service times agrees, the two systems have identical histories and take identical actions under the same nonanticipating policy. On the disagreement event, each terminal queue contains at most the $h$ jobs that could have arrived, so the final queue length diference is at most h. Therefore, for every fixed policy π,

$$
| J _ { h } ^ { p } ( \pi ) - J _ { h } ^ { q } ( \pi ) | \leq \frac { h ^ { 2 } } { p _ { - } } \mathcal { E } ( q ) .\tag{33}
$$

Let $\pi _ { p }$ and $\pi _ { q }$ be optimal in the true and fitted systems, respectively (so $J _ { h } ^ { p } ( \pi _ { p } ) = V _ { h } ^ { p } , J _ { h } ^ { q } ( \pi _ { q } ) = V _ { h } ^ { q } )$ Applying Equation (33) to these two policies gives

$$
\begin{array} { l } { { V _ { h } ^ { p } - V _ { h } ^ { q } \le J _ { h } ^ { p } ( \pi _ { q } ) - J _ { h } ^ { q } ( \pi _ { q } ) \le \displaystyle \frac { h ^ { 2 } } { p _ { - } } \mathcal { E } ( q ) , } } \\ { { V _ { h } ^ { q } - V _ { h } ^ { p } \le J _ { h } ^ { q } ( \pi _ { p } ) - J _ { h } ^ { p } ( \pi _ { p } ) \le \displaystyle \frac { h ^ { 2 } } { p _ { - } } \mathcal { E } ( q ) . } } \end{array}
$$

Hence $| V _ { h } ^ { p } - V _ { h } ^ { q } | \leq h ^ { 2 } \mathcal { E } ( q ) / p$

Finally, compare the true model that follows $( p , F _ { p } )$ with the intermediate model $( q , F _ { q } )$ and then with the empirical model $( q , { \widehat { F } } )$ :

$$
J _ { h } ^ { p } ( \widehat \pi ) - V _ { h } ^ { p } = \left( J _ { h } ^ { p } ( \widehat \pi ) - J _ { h } ^ { q } ( \widehat \pi ) \right) + \left( J _ { h } ^ { q } ( \widehat \pi ) - V _ { h } ^ { q } \right) + \left( V _ { h } ^ { q } - V _ { h } ^ { p } \right) .
$$

The first and third terms are bounded by Equation (33); Equation (32) bounds the middle term with $\eta = W _ { 1 } ( F _ { q } , \widehat { F } )$ . Adding the three bounds gives Equation (18).

## E.4 Proof of Lemma 5

Proof. Let $\begin{array} { r } { M _ { L } = \sum _ { t = 1 } ^ { L } A _ { t } } \end{array}$ be the number of arrivals through round L. Since $M _ { L } \sim$ Binomial $( L , \lambda )$ and $\mathbb { E } ( M _ { L } ) = \lambda L$ , the multiplicative Chernof bound with relative deviation $1 / 2$ gives $\mathbb { P } ( M _ { L } <$ $\lambda L / 2 ) \leq e ^ { - \lambda L / 8 }$ . Also, 2n $\leq \lambda L / 4$ by definition. Therefore, on $\left\{ M _ { L } \ge \lambda L / 2 , N _ { L } < 2 n \right\}$ , the number of jobs that have arrived but not begun service by the end of round $L { \mathrm { i s } } M _ { L } { - } N _ { L } > { \lambda } L / 2 { - } 2 n \geq { \lambda } L / 4$ So more than $\lambda L / 4$ jobs have arrived but not begun service by the end of round L. Each such job contributes at least one unit to $W _ { L + 1 }$ , so $W _ { L + 1 } > \lambda L / 4$ . Markov’s inequality applied to $e ^ { r W _ { L + 1 } }$ and Lemma 1 give

$$
\begin{array} { r l } & { \mathbb { P } ( M _ { L } \geq \lambda L / 2 , N _ { L } < 2 n ) \leq \mathbb { P } ( W _ { L + 1 } > \lambda L / 4 ) } \\ & { \qquad = \mathbb { P } ( e ^ { r W _ { L + 1 } } > e ^ { r \lambda L / 4 } ) } \\ & { \qquad \leq \frac { \mathbb { E } ( e ^ { r W _ { L + 1 } } ) } { e ^ { r \lambda L / 4 } } \leq K _ { r } e ^ { - r \lambda L / 4 } . } \end{array}\tag{34}
$$

So by a union bound, we can get the desired inequality as follows.

$$
\begin{array} { r l } & { \mathbb { P } ( N _ { L } < 2 n ) = \mathbb { P } ( M _ { L } \geq \lambda L / 2 , N _ { L } < 2 n ) + \mathbb { P } ( M _ { L } < \lambda L / 2 , N _ { L } < 2 n ) } \\ & { \qquad \leq \mathbb { P } ( M _ { L } \geq \lambda L / 2 , N _ { L } < 2 n ) + \mathbb { P } ( M _ { L } < \lambda L / 2 ) } \\ & { \qquad \leq K _ { r } e ^ { - r \lambda L / 4 } + e ^ { - \lambda L / 8 } } \end{array}\tag{35}
$$

Under FCFS, jobs begin service in arrival order. Hence, whenever $N _ { L } \ge 2 n$ , the first $2 n$ recorded observations are $( X _ { i } , Y _ { i } ) _ { i = 1 } ^ { 2 n }$ □

## E.5 Proof of Lemma 6

Proof. Let s be an initial queue and let $s ^ { + }$ be the queue that contains all jobs of s together with some additional jobs. Fix a policy $\pi ^ { + }$ for $s ^ { + }$ . We construct a policy π for s by maintaining a virtual copy of the larger system, and show that by this algorithm its expected terminal queue length is not bigger than that of $\pi ^ { + }$ started from $s ^ { + }$

Couple the two systems by the same future arrivals and by the same service times for every common job. Whenever $\pi ^ { + }$ selects a common job, the smaller system selects the same job; whenever $\pi ^ { + }$ selects one of the additional jobs, the smaller system idles. Idling is admissible in the policy class considered here. By induction over the number of rounds, we can directly prove that every job in the smaller system is also present in the virtual larger system, and the algorithm always works well in every round. Hence the terminal queue under the constructed policy is not bigger than the terminal queue under $\pi ^ { + }$ . This shows that, if we take an infimum of the expected queue length (at the end) of the queue started with s among all possible policies, it is not bigger than the expected queue length of the queue started with $s ^ { + }$ and using policy $\phi ^ { + }$ . Now, taking an infimum on both sides of the inequality over all possible policies $\pi ^ { + }$ , we can prove monotonicity in the initial state.

Now start an $( h + 1 )$ )-round system with an empty queue. No job is available for service in its first round, and the server must idle. After the arrival at the end of that round, the queue state S is either empty or contains one waiting job. Since what we’ve proved above implies $V _ { h } ^ { \mathrm { I A } } ( S ) \geq V _ { h } ^ { \mathrm { I A } } ( \emptyset ) = V _ { h } ^ { \mathrm { I A } }$ the Bellman recursion and the initial-state monotonicity give

$$
V _ { h + 1 } ^ { \mathrm { I A } } = \mathbb { E } V _ { h } ^ { \mathrm { I A } } ( S ) \geq \mathbb { E } V _ { h } ^ { \mathrm { I A } } ( \emptyset ) = V _ { h } ^ { \mathrm { I A } } .
$$

## F Lower Bounds

This section constructs the common family of instances and proves Theorem 2. For work-conserving policies, the last round reduces to choosing the faster of two waiting jobs. For policies that may idle, we first show that any policy with small regret reaches the same decision state with probability bounded away from zero.

## F.1 The hard family

We first define the distribution D, job context vectors, probability constants $( \lambda , p _ { - } , p _ { + } )$ , and parameter vector $\theta ^ { * }$ that will be used to make a lower bound of regret as follows.

1. Job types and distribution $\mathcal { D } \colon$

Let $m = d - 2$ , and let there be d types of jobs in our distribution: a dummy job $C _ { D }$ , a baseline job $C _ { 0 }$ , and m candidate jobs $C _ { 1 } , \ldots , C _ { m }$ . Conditional on an arrival, let’s define each job’s arrival probability

$$
\mathbb { P } ( C _ { D } ) = \nu _ { D } , \quad \mathbb { P } ( C _ { 0 } ) = \nu _ { 0 } , \quad \mathbb { P } ( C _ { j } ) = \frac { \nu _ { C } } { m } .
$$

where $\nu _ { D } , \nu _ { 0 } , \nu _ { C }$ are three nonnegative real numbers that sum to 1. So $\mathcal { D }$ is the distribution that consists of jobs $C _ { D } , C _ { 0 } , C _ { 1 } , \ldots , C _ { m }$ with arrival probability as above.

2. Job context vectors and probability constants:

For the d-dimensional standard basis set $( e _ { j } ) _ { j = 1 } ^ { d }$ , define

$$
x _ { D } = e _ { 1 } , \qquad x _ { 0 } = { \frac { e _ { 2 } } { \sqrt { 2 } } } , \qquad x _ { C _ { j } } = { \frac { e _ { 2 } + e _ { j + 2 } } { \sqrt { 2 } } } .
$$

Choose fixed $z _ { D } , z _ { 0 } , a _ { 0 } , S \ ( a _ { 0 } , S > 0 )$ such that $z _ { D } > z _ { 0 } + a _ { 0 }$ and $S ^ { 2 } > z _ { D } ^ { 2 } + 2 z _ { 0 } ^ { 2 }$ , and set

$$
c _ { S } = \sqrt { \frac { S ^ { 2 } - z _ { D } ^ { 2 } - 2 z _ { 0 } ^ { 2 } } { 2 } } , ~ \ell _ { 0 } = \operatorname* { m i n } _ { | z - z _ { 0 } | \leq a _ { 0 } } \mu ^ { \prime } ( z ) > 0 .
$$

For example, we can choose $\begin{array} { r } { z _ { D } = \frac { S } { 2 } , z _ { 0 } = - \frac { S } { 2 } , a _ { 0 } = 0 . 9 9 S . } \end{array}$

Now, we choose fixed bounds $p _ { - } \leq \mu ( z _ { 0 } - a _ { 0 } )$ and $p _ { + } \geq \mu ( z _ { D } )$ with $0 < p _ { - } < p _ { + } < 1$ , and a fixed $\lambda > 0$ small enough that the following inequality holds.

$$
\lambda < p _ { - } , \qquad \frac { \lambda ( 1 - \lambda ) } { p _ { - } - \lambda } \leq \frac { ( 1 - \mu ( z _ { D } ) ) ^ { 4 } } { 1 6 }\tag{36}
$$

We can always choose such $\lambda > 0$ since the functions $\begin{array} { r } { \lambda \mapsto \lambda , \lambda \mapsto \frac { \lambda ( 1 - \lambda ) } { p _ { - } - \lambda } } \end{array}$ (defined on $0 < \lambda < p _ { - } )$ go to 0 as $\lambda \to 0 ^ { + }$

## 3. Parameter vector $\theta ^ { * } { \mathrm { : } }$

For fixed constants $\alpha > 0$ that will be determined later, let $\begin{array} { r } { a _ { T } = \operatorname* { m i n } \left( a _ { 0 } , \frac { c _ { S } } { \sqrt { m } } , \alpha \sqrt { \frac { m } { T } } \right) } \end{array}$ . Next, for each $\omega \in \{ - 1 , + 1 \} ^ { m }$ , set the parameter vector corresponding to ω as

$$
\theta _ { \omega } = ( z _ { D } , \sqrt { 2 } z _ { 0 } , \sqrt { 2 } \omega _ { 1 } a _ { T } , \ldots , \sqrt { 2 } \omega _ { m } a _ { T } ) .
$$

Then the dummy job has departure probability $p _ { D } = \mu ( z _ { D } )$ , the baseline job has departure probability $p _ { 0 } = \mu ( z _ { 0 } )$ , and the candidate job $C _ { j }$ has departure probability $p _ { C _ { j } } = \mu ( z _ { 0 } + \omega _ { j } a _ { T } )$

The following lemma shows that the above definitions satisfy Assumption 1.

Lemma 9. The above construction satisfies Assumption 1 for every d, T.

Proof. Since $\begin{array} { r } { a _ { T } \leq \frac { c _ { S } } { \sqrt { m } } } \end{array}$ by definition and $w _ { i } ^ { 2 } = 1$ for all $i \in [ m ]$

$$
\begin{array} { c } { \| \theta _ { \omega } \| _ { 2 } ^ { 2 } = z _ { D } ^ { 2 } + 2 z _ { 0 } ^ { 2 } + 2 ( w _ { 1 } ^ { 2 } + \ldots + w _ { m } ^ { 2 } ) a _ { T } ^ { 2 } } \\ { = z _ { D } ^ { 2 } + 2 z _ { 0 } ^ { 2 } + 2 m a _ { T } ^ { 2 } } \\ { \leq z _ { D } ^ { 2 } + 2 z _ { 0 } ^ { 2 } + 2 c _ { S } ^ { 2 } = S ^ { 2 } . } \end{array}
$$

Since $a _ { T } ~ \le ~ a _ { 0 }$ by definition, $z _ { 0 } + w _ { j } a _ { T } \in \{ z _ { 0 } - a _ { T } , z _ { 0 } + a _ { T } \} \subset [ z _ { 0 } - a _ { 0 } , z _ { D } ]$ holds. Also $z _ { 0 } , z _ { D } \in [ z _ { 0 } - a _ { 0 } , z _ { D } ]$ , so every job’s departure probability lies in $[ \mu ( z _ { 0 } - a _ { 0 } ) , \mu ( z _ { D } ) ] \subset [ p _ { - } , p _ { + } ]$ Finally,

$$
\rho _ { \omega } = \lambda \mathbb { E } _ { \omega } \frac { 1 } { p ( X ) } \le \frac { \lambda } { p _ { - } } < 1 .
$$

## F.2 Information in the observed history

Fix an adaptive learning policy π. For each $\omega \in \{ - 1 , + 1 \} ^ { m }$ , let $P _ { \omega } ^ { T }$ denote the probability law induced by running π for $T$ rounds on the instance with parameter $\theta _ { \omega }$ . This law is defined on the complete interaction history through round $T _ { i }$ including all arrival indicators and contexts, the learner’s internal randomization and resulting actions, and all observed departure indicators. Thus, for any event E determined by the interaction through round $T , P _ { \omega } ^ { T } ( \mathcal { E } )$ is the probability that E occurs under parameter $\theta _ { \omega }$ and policy π. We suppress the dependence of $P _ { \omega } ^ { T }$ on the fixed policy π in the notation.

Let $\boldsymbol { \omega } ^ { ( j ) }$ be the vector obtained by changing the sign of coordinate $\textit { j } \left( \omega _ { j } \right)$ of $\omega ,$ and let $N _ { j } ( T )$ be the number of rounds through T in which a candidate-j job is served.

Lemma 10. For every adaptive nonpreemptive policy, including policies that may idle,

$$
D _ { \mathrm { K L } } ( P _ { \omega } ^ { T } \Vert P _ { \omega ^ { ( j ) } } ^ { T } ) \leq \frac { a _ { T } ^ { 2 } } { 2 } \mathbb { E } _ { \omega } N _ { j } ( T ) \leq \frac { \lambda \nu _ { C } T a _ { T } ^ { 2 } } { 2 m p _ { - } } ,\tag{37}
$$

$$
\mathrm { T V } ( P _ { \omega } ^ { T } , P _ { \omega ^ { ( j ) } } ^ { T } ) \leq a _ { T } \sqrt { \frac { \lambda \nu _ { C } T } { 4 m p _ { - } } } \leq \frac { \alpha } { 2 } \sqrt { \frac { \lambda \nu _ { C } } { p _ { - } } } .\tag{38}
$$

Proof. Only the departure indicators observed while serving candidate-j jobs have diferent conditional distributions under $\omega$ and $\boldsymbol { \omega } ^ { ( j ) }$ . The two neighboring logits are $z _ { 0 } + \omega _ { j } a _ { T }$ and $z _ { 0 } - \omega _ { j } a _ { T }$ If we define $A ( z ) = \log ( 1 + e ^ { z } )$ , then $A ^ { \prime } ( z ) = \mu ( z )$ and $D _ { \mathrm { K L } } \left( \mathrm { B e r n } ( \mu ( u ) ) \| \mathrm { B e r n } ( \mu ( v ) ) \right) = A ( v ) -$ $A ( u ) - A ^ { \prime } ( u ) ( v - u )$ hold, and the following inequality can be obtained from Taylor’s theorem.

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } \left( \mathrm { B e r n } ( \mu ( z _ { 0 } + \omega _ { j } a _ { T } ) ) \right) \parallel \mathrm { B e r n } ( \mu ( z _ { 0 } - \omega _ { j } a _ { T } ) ) ) } \\ & { = A ( z _ { 0 } + \omega _ { j } a _ { T } ) - A ( z _ { 0 } - \omega _ { j } a _ { T } ) - 2 A ^ { \prime } ( z _ { 0 } - \omega _ { j } a _ { T } ) \omega _ { j } a _ { T } } \\ & { = \cfrac { 1 } { 2 } A ^ { \prime \prime } ( \varepsilon ) ( 2 \omega a _ { T } ) ^ { 2 } \leq 2 \left( \underset { z } { \operatorname* { s u p } } \cfrac { e ^ { z } } { ( 1 + e ^ { z } ) ^ { 2 } } \right) a _ { T } ^ { 2 } \leq \frac { a _ { T } ^ { 2 } } { 2 } } \end{array}
$$

Here, ε is some real number in $( z _ { 0 } - \omega _ { j } a _ { T } , z _ { 0 } + \omega _ { j } a _ { T } )$ obtained by second order Taylor’s theorem, and we used $\begin{array} { r } { A ^ { \prime \prime } ( \varepsilon ) = \frac { e ^ { \varepsilon } } { ( 1 + e ^ { \varepsilon } ) ^ { 2 } } \leq \operatorname* { s u p } _ { z } \frac { \check { e } ^ { z } } { ( 1 + e ^ { z } ) ^ { 2 } } = \frac { 1 } { 4 } } \end{array}$

Let $I _ { t } ^ { ( j ) }$ indicate that a candidate-j job is served in round t. Let $\mathcal { F } _ { t } ^ { - }$ denote the information available immediately before the departure indicator $D _ { t }$ is observed; this includes the complete history through round $t - 1$ , as well as the learner’s randomization and resulting action in round t. In particular, $I _ { t } ^ { ( j ) }$ is $\mathcal { F } _ { t } ^ { - }$ -measurable.

Conditional on $\mathcal { F } _ { t } ^ { - }$ , the two neighboring instances have identical departure-indicator laws when $I _ { t } ^ { ( j ) } = 0$ . When $I _ { t } ^ { ( j ) } = 1$ , the conditional laws of $D _ { t }$ under $\omega$ and $\boldsymbol { \omega } ^ { ( j ) }$ are, respectively,

$$
\mathrm { B e r n } ( \mu ( z _ { 0 } + \omega _ { j } a _ { T } ) ) \quad \mathrm { a n d } \quad \mathrm { B e r n } ( \mu ( z _ { 0 } - \omega _ { j } a _ { T } ) ) .
$$

All other conditional kernels, including those governing arrivals, contexts, learner randomization, and actions, are identical under the two instances. Therefore, the KL chain rule gives

$$
\begin{array} { r l } & { \displaystyle { \cal D } _ { \mathrm { K L } } ( P _ { \omega } ^ { T } | | P _ { \omega ( \bar { \jmath } ) } ^ { T } ) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \omega } \left[ I _ { t } ^ { ( \bar { \jmath } ) } { \cal D } _ { \mathrm { K L } } \left( \mathrm { R e r n } ( \mu ( z _ { 0 } + \omega _ { j } a _ { T } ) ) \right) | \mathrm { R e r n } ( \mu ( z _ { 0 } - \omega _ { j } a _ { T } ) ) ) \right] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad = \left( \displaystyle { \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \omega } } { \cal I } _ { t } ^ { ( \bar { \jmath } ) } \right) { \cal D } _ { \mathrm { K L } } \left( \mathrm { B e r n } ( \mu ( z _ { 0 } + \omega _ { j } a _ { T } ) ) \right) \| \mathrm { B e r n } ( \mu ( z _ { 0 } - \omega _ { j } a _ { T } ) ) ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \left( \mathbb { E } _ { \omega } \displaystyle { \sum _ { t = 1 } ^ { T } } I _ { t } ^ { ( \bar { \jmath } ) } \right) { \cal D } _ { \mathrm { K L } } \left( \mathrm { B e r n } ( \mu ( z _ { 0 } + \omega _ { j } a _ { T } ) ) \right) \| \mathrm { B e r n } ( \mu ( z _ { 0 } - \omega _ { j } a _ { T } ) ) ) } \\ & { \quad \quad \quad \quad \quad \quad = E _ { \omega } \mathcal { N } _ { j } ( T ) \cdot { \cal D } _ { \mathrm { K L } } \left( \mathrm { B e r n } ( \mu ( z _ { 0 } + \omega _ { j } a _ { T } ) ) \right) \| \mathrm { B e r n } ( \mu ( z _ { 0 } - \omega _ { j } a _ { T } ) ) ) } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

where $\begin{array} { r } { N _ { j } ( T ) = \sum _ { t = 1 } ^ { T } I _ { t } ^ { ( j ) } } \end{array}$ . This proves the first inequality in Equation (37).

Pre-sample the geometric service time of every candidate-j job. Regardless of the scheduling decisions, the number of rounds that one specific candidate-j job is served by time $T$ cannot exceed its service time. Therefore $\begin{array} { r } { N _ { j } ( T ) \le \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ A _ { t } = 1 , X _ { t } = C _ { j } \} G _ { t } } \end{array}$ , and the following holds.

$$
\begin{array} { r l } & { \mathbb { E } _ { \omega } N _ { j } ( T ) \leq \mathbb { E } _ { \omega } \left( \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ A _ { t } = 1 , X _ { t } = C _ { j } \} G _ { t } \right) } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \omega } \left( \mathbf { 1 } \{ A _ { t } = 1 , X _ { t } = C _ { j } \} G _ { t } \right) } \\ & { \qquad = T \cdot \mathbb { E } _ { \omega } \left( \mathbf { 1 } \{ A _ { t } = 1 , X _ { t } = C _ { j } \} G _ { t } \right) } \\ & { \qquad = T \cdot \lambda \frac { \nu _ { C } } { m } \mathbb { E } _ { \omega } ( G _ { t } \mid X _ { t } = C _ { j } ) } \\ & { \qquad = \displaystyle \frac { \lambda \nu _ { C } T } { m p ( X _ { C _ { j } } ) } \leq \frac { \lambda \nu _ { C } T } { m p _ { - } } } \end{array}
$$

This proves the second inequality in Equation (37).

For Equation (38), using Pinsker’s inequality $\mathrm { T V } ( P , Q ) \leq { \sqrt { { \frac { 1 } { 2 } } D _ { \mathrm { K L } } ( P \| Q ) } }$ gives the first bound in Equation (38), and $a _ { T } \leq \alpha \sqrt { m / T }$ gives the second. □

![](images/c1d3dc4bbdfd14569794054b629c591200da834af0a82210c5c3f74a815c3fe3.jpg)  
Figure 1: The final four-round construction used in the lower-bound proof. Solid vertical lines mark round boundaries, and dashed lines separate service from the end-of-round arrival. The shaded region corresponds to $\mathcal { H } ;$ the candidate-j arrival is appended afterward.

For later use, let $P _ { \omega } ^ { t } , P _ { \omega } ^ { t , \mathrm { p r e } }$ denote the law of the history up to the end of round t, and the law of the history after service in round t and before the arrival at the end of that round. This history is a measurable function of the complete observations through round $t ,$ so the definition of total variance between two distributions $P _ { \omega } ^ { t } , P _ { \omega ^ { ( j ) } } ^ { t }$ gives

$$
\begin{array} { r } { \mathrm { T V } ( P _ { \omega } ^ { t , \mathrm { p r e } } , P _ { \omega ^ { ( j ) } } ^ { t , \mathrm { p r e } } ) \leq \mathrm { T V } ( P _ { \omega } ^ { t } , P _ { \omega ^ { ( j ) } } ^ { t } ) . } \end{array}\tag{39}
$$

Define

$$
\Delta _ { 0 } = 1 - { \frac { \lambda } { p _ { - } } } , \qquad \zeta = \lambda ^ { 3 } \nu _ { D } \nu _ { 0 } \nu _ { C } p _ { D } ( 1 - p _ { D } ) , \qquad g _ { D } = ( 1 - \lambda ) ^ { 2 } p _ { D } ( 1 - p _ { D } ) ^ { 2 } .\tag{40}
$$

It’s easy to check that these three constants are all positive. Choose the fixed $\alpha > 0$ which is introduced in Appendix F.1 so that following inequality holds.

$$
\alpha \leq \sqrt { \frac { p _ { - } } { \lambda \nu _ { C } } } \operatorname* { m i n } \left( \frac { \zeta \Delta _ { 0 } } { 2 } , \frac { 3 \zeta } { 1 6 } \right)
$$

This condition was chosen to make the inequality $\begin{array} { r } { \frac { \alpha } { 2 } \sqrt { \frac { \lambda \nu _ { C } } { p _ { - } } } \leq \operatorname* { m i n } \left( \frac { \zeta \Delta _ { 0 } } { 4 } , \frac { 3 \zeta } { 3 2 } \right) } \end{array}$ hold. So by this condition, $\begin{array} { r } { \mathrm { T V } ( P _ { \omega } ^ { T } , P _ { \omega ^ { ( j ) } } ^ { T } ) \leq \frac { \zeta \Delta _ { 0 } } { 4 } , \frac { 3 \zeta } { 3 2 } } \end{array}$ holds.

## F.3 Proof of Theorem 2 for WC case

Proof. Under every work-conserving policy, the workload started with $W _ { 1 } = 0$ and satisfies Equation (2) is stochastically dominated by stationary workload(Lemma 1). By Lemma 3, for every ω,

$$
\mathbb { P } _ { \omega } ( W _ { T - 3 } = 0 ) \ge \mathbb { P } _ { \omega } ( W = 0 ) = 1 - \rho _ { \omega } \ge \Delta _ { 0 }\tag{41}
$$

where $W$ denotes the stationary distribution’s workload. The above inequality holds because the zero-started workload $W _ { t }$ is stochastically dominated by the stationary workload W for every $t \geq 1$

Starting from this event $\{ W _ { T - 3 } = 0 \}$ , consider the following arrivals and departures: First, the server must idle since $W _ { T - 3 } = 0$ , and then a dummy arrives at the end of round $T - 3$ . The server starts to process that dummy job in round $T - 2$ , but it fails to succeed. Next, a baseline job arrives at the end of round $T - 2$ . In round $T - 1$ , the server must do the failed dummy job, and it succeeds with it in round $T - 1$ . Let $\mathcal { H }$ denote this event before the arrival at the end of round $T - 1$ . The above Figure 1 illustrates this event. By independence of events,

$$
\begin{array} { r l } & { \mathbb { P } _ { \omega } ( \mathcal { H } ) = \mathbb { P } _ { \omega } ( W _ { T - 3 } = 0 ) \cdot ( \lambda \nu _ { D } ) \cdot ( 1 - p _ { D } ) \cdot ( \lambda \nu _ { 0 } ) \cdot p _ { D } } \\ & { \qquad \ge \Delta _ { 0 } \lambda ^ { 2 } \nu _ { D } \nu _ { 0 } p _ { D } ( 1 - p _ { D } ) = \cfrac { \zeta \Delta _ { 0 } } { \lambda \nu _ { C } } . } \end{array}\tag{42}
$$

Conditional on $\mathcal { H } .$ , candidate $C _ { j }$ arrives at the end of round $T - 1$ with probability $\lambda \nu _ { C } / m$ The server is free in round $T$ and must choose between $C _ { 0 }$ and $C _ { j } . \mathrm { ~ I f ~ } \omega _ { j } = + 1 , C _ { j }$ has the higher departure probability; if $\omega _ { j } = - 1 , C _ { 0 }$ does. In either case, the mean-value theorem gives

$$
| p _ { C _ { j } } - p _ { 0 } | = | \mu ( z _ { 0 } + \omega _ { j } a _ { T } ) - \mu ( z _ { 0 } ) | \geq \ell _ { 0 } a _ { T } .
$$

Thus, serving the slower job rather than the faster one increases the expected terminal queue length by $| p _ { C _ { j } } - p _ { 0 } | \geq \ell _ { 0 } a _ { T }$

Fix j and pair an instance $\omega$ with $\boldsymbol { \omega } ^ { ( j ) }$ . Given the history just before the arrival at the end of round $T - 1$ , let the Markov kernel $K _ { j }$ append a candidate-j arrival and then apply the learner’s randomized decision in round $T .$ Write $\overline { { P } } _ { \omega , j } = P _ { \omega } ^ { T - 1 , \mathrm { p r e } } K _ { j }$ , and let $\boldsymbol { S } _ { j } ( \omega )$ be the event that, conditional on H and on $C _ { j }$ arriving at the end of round $T - 1$ , the decision selects the optimal job between $C _ { 0 }$ and $C _ { j }$ in round $T$ for the parameter vector $\theta _ { \omega }$ . So if $\omega _ { j } = 1$ then $S _ { j } ( \omega )$ is the event that the decision selects $C _ { j }$ , and if $\omega _ { j } = - 1$ then $S _ { j } ( \omega )$ is the event that the decision selects $C _ { 0 }$ This directly shows that, within this conditional experiment, $S _ { j } ( \omega ) ^ { c } = S _ { j } ( \omega ^ { ( j ) } )$ since $[ \omega ^ { ( j ) } ] _ { j } = - \omega _ { j }$ By these definitions, data processing for $K _ { j }$ , followed by Equation (39), gives

$$
\begin{array} { r l } & { \overline { { P } } _ { \omega , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ) ^ { c } ) + \overline { { P } } _ { \omega ^ { ( j ) } , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ^ { ( j ) } ) ^ { c } ) } \\ & { \quad = \overline { { P } } _ { \omega , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ) ^ { c } ) + \overline { { P } } _ { \omega ( j ) , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ) ) } \\ & { \quad = \overline { { P } } _ { \omega , j } ( \mathcal { H } ) - \overline { { P } } _ { \omega , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ) ) + \overline { { P } } _ { \omega ^ { ( j ) } , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ) ) } \\ & { \quad = \overline { { P } } _ { \omega , j } ( \mathcal { H } ) - \left( \overline { { P } } _ { \omega , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ) ) - \overline { { P } } _ { \omega ^ { ( j ) } , j } ( \mathcal { H } , \mathcal { S } _ { j } ( \omega ) ) \right) } \\ & { \quad \geq \mathbb { P } _ { \omega } ( \mathcal { H } ) - \mathrm { T V } ( \overline { { P } } _ { \omega , j } , \overline { { P } } _ { \omega ^ { ( j ) } , j } ) } \\ & { \quad \geq \mathbb { P } _ { \omega } ( \mathcal { H } ) - \mathrm { T V } ( \boldsymbol { P } _ { \omega ^ { ( - 1 , \mathrm { p r e } } } ^ { T - 1 , \mathrm { p r e } } , \boldsymbol { P } _ { \omega ^ { ( j ) } } ^ { T - 1 , \mathrm { p r e } } ) } \\ & { \quad \geq \mathbb { P } _ { \omega } ( \mathcal { H } ) - \mathrm { T V } ( \boldsymbol { P } _ { \omega ^ { ( - 1 ) } , \boldsymbol { P } _ { \omega ^ { ( j ) } } ^ { T - 1 } } ) . } \end{array}
$$

In more detail, the first inequality follows from the definition of total variation, while the second follows from the data-processing inequality $\mathrm { T V } ( P K , Q K ) \le \mathrm { T V } ( P , Q )$ , which holds for any probability measures P and $Q$ on the same measurable space and any common Markov kernel K from that space to another measurable space. We also used the identity $\overline { { P } } _ { \omega , j } ( \mathcal { H } ) = P _ { \omega } ( \mathcal { H } )$ , which holds because $K _ { j }$ leaves the pre-arrival history unchanged.

In the original experiment, a candidate-j arrival occurs with probability $\lambda \nu _ { C } / m$ , independently of the history immediately before that arrival; conditional on this arrival, the action law is the one produced by $K _ { j }$ . Multiplying by the arrival probability and applying Equations (38) and (42), the paired probability of a wrong final action is at least

$$
\begin{array} { r l } & { \displaystyle \frac { \lambda \nu _ { C } } { m } \left( \overline { { P } } _ { \omega , j } ( \mathcal { H } , S _ { j } ( \omega ) ^ { c } ) + \overline { { P } } _ { \omega ^ { ( j ) } , j } ( \mathcal { H } , S _ { j } ( \omega ^ { ( j ) } ) ^ { c } ) \right) \geq \frac { \lambda \nu _ { C } } { m } \left( \mathbb { P } _ { \omega } ( \mathcal { H } ) - \mathrm { T V } ( P _ { \omega } ^ { T - 1 } , P _ { \omega ^ { ( j ) } } ^ { T - 1 } ) \right) } \\ & { \qquad \geq \frac { \lambda \nu _ { C } } { m } \left( \frac { \zeta \Delta _ { 0 } } { \lambda \nu _ { C } } - \frac { \zeta \Delta _ { 0 } } { 4 } \right) } \\ & { \qquad = \frac { \zeta \Delta _ { 0 } } { m } \left( 1 - \frac { \lambda \nu _ { C } } { 4 } \right) \geq \frac { 3 \zeta \Delta _ { 0 } } { 4 m } . } \end{array}\tag{43}
$$

Now let $\begin{array} { r } { p _ { \mathrm { e r r } } ( \omega ) = \sum _ { j = 1 } ^ { m } \frac { \lambda \nu _ { C } } { m } \overline { { P } } _ { \omega , j } ( \mathcal { H } , S _ { j } ( \omega ) ^ { c } ) } \end{array}$ . So this is the probability that, for the system with parameter vector $\theta _ { \omega }$ , the event H occurs, a candidate job arrives at the end of round $T - 1$ , and the system chooses the slower job in the final round. Now, by a double-counting argument, the following equalities and inequality hold. More specifically, the third equality follows by applying the double-counting argument to the vertices and edges of the m-dimensional hypercube, whose vertex set is $\{ - 1 , 1 \} ^ { m }$

$$
\begin{array} { r l } { \mathbb { E } _ { \mathrm { s e c i l u t } ( [ \mathbf { \bar { f } } \cdot \mathbf { \Phi } , 1 ] : | \mathbf { \bar { n } } ) } | \rho _ { m r } ( \omega ) | = \displaystyle \frac { 1 } { 2 ^ { m } } \sum _ { \alpha \in \{ - 1 , 1 \} } \sum _ { m \in \Gamma ( \omega ) } \mathbf { \bar { n } } _ { r m } ( \omega ) } & { } \\ & { = \displaystyle \frac { 1 } { 2 ^ { m } } \sum _ { \alpha \in \{ - 1 , 1 \} } \sum _ { n = j = 1 \atop n \atop n \in \mathbb { Z } } \displaystyle \sum _ { m \leq i \in \mathbb { Z } } \frac { \lambda \nu _ { C } } { m } \overline { { p } } _ { \alpha , j } ( \boldsymbol { n } , S _ { j } ( \omega ) ^ { c } ) } \\ & { = \displaystyle \frac { 1 } { 2 ^ { m } } \sum _ { \alpha \in \{ - 1 , 1 \} } \sum _ { m = \pm \infty + \infty } \frac { \lambda \nu _ { C } } { m } \left( \overline { { p } } _ { \alpha , j } ( \mathcal { M } , S _ { j } ( \omega ) ^ { c } ) + \overline { { p } } _ { \alpha ^ { ( j ) } , j } ( \mathcal { M } , S _ { j } ( \omega ^ { ( j ) } ) ^ { c } ) \right) } \\ & { \geq \displaystyle \frac { 1 } { 2 ^ { m } } \sum _ { \alpha \in \{ - 1 , 1 \} } \sum _ { n = \pm \infty + \infty } \frac { \lambda \nu _ { C } } { 4 m } } \\ & { = \displaystyle \frac { 3 ( \Delta _ { \alpha } ) \omega } { 4 m } \frac { \omega ^ { 2 m } } { \omega ^ { 2 m } ( \omega - i - 1 ) m } \frac { 3 ( \Delta _ { \alpha } ) } { m } } \\ & { = \displaystyle \frac { 3 ( \Delta _ { \alpha } ) } { 4 m } \frac { \omega ^ { 2 m } } { \omega ^ { 2 m } } = \frac { 3 ( \Delta _ { \alpha } ) } { 4 m } } \end{array}\tag{44}
$$

Therefore $\begin{array} { r } { \operatorname* { m a x } _ { \omega \in \{ - 1 , 1 \} ^ { m } } p _ { \mathrm { e r r } } ( \omega ) \geq \mathbb { E } _ { \omega \sim \mathrm { U n i f } ( \{ - 1 , 1 \} ^ { m } ) } [ p _ { \mathrm { e r r } } ( \omega ) ] \geq \frac { 3 \zeta \Delta _ { 0 } } { 8 } } \end{array}$ holds, so if we choose $\omega _ { 0 } \in$ arg ma $\mathrm { X } _ { \omega \in \{ - 1 , 1 \} ^ { m } } p _ { \mathrm { e r r } } ( \omega )$ , then $\begin{array} { r } { p _ { \mathrm { e r r } } ( \omega _ { 0 } ) \ge \frac { 3 \zeta \Delta _ { 0 } } { 8 } } \end{array}$ . Changing only the final action to the better member of $\{ C _ { 0 } , C _ { j } \}$ is a feasible work-conserving comparison policy. Thus the queue-length regret for the instance with parameter vector $\theta _ { \omega _ { 0 } }$ is at least $\begin{array} { r } { | p _ { 0 } - p _ { C _ { j } } | \cdot p _ { \mathrm { e r r } } ( \omega _ { 0 } ) \geq \ell _ { 0 } a _ { T } \cdot \frac { 3 \zeta \Delta _ { 0 } } { 8 } } \end{array}$ , and this instance has regret at least this value.

So this means that for any fixed policy π, there must be some $\omega \in \{ - 1 , 1 \} ^ { m }$ such that, if the unknown parameter vector were actually $\theta _ { \omega } .$ , then the regret would be at least $\ell _ { 0 } a _ { T } \cdot \frac { 3 \zeta \Delta _ { 0 } } { 8 }$ . Since this works for every work-conserving policy $\pi .$ , this shows that whenever we use a work-conserving policy for T rounds, there exist a distribution $\mathcal { D }$ and a parameter vector $\theta ^ { * }$ such that the regret is bounded below by $\ell _ { 0 } a _ { T } \cdot \frac { 3 \zeta \Delta _ { 0 } } { 8 }$

Finally, define $b _ { d , T } = d ^ { - 1 / 2 } \wedge \sqrt { d / T }$ . Since $b _ { d , T } \leq d ^ { - 1 / 2 } \leq 1 / \sqrt { 3 }$ and $\textstyle { \frac { d } { 3 } } \leq m = d - 2 < d$ hold for $d \geq 3$

$$
a _ { 0 } \geq \sqrt { 3 } a _ { 0 } b _ { d , T } , \qquad \frac { c _ { S } } { \sqrt { m } } > \frac { c _ { S } } { \sqrt { d } } \geq c _ { S } b _ { d , T } .
$$

If $T \geq d ^ { 2 }$ , then $b _ { d , T } = \sqrt { d / T }$ and $\sqrt { m / T } \geq b _ { d , T } / \sqrt { 3 }$ . If $T < d ^ { 2 }$ , then $b _ { d , T } = d ^ { - 1 / 2 }$ and $\sqrt { m / T } >$ $\sqrt { m } / d \geq 1 / \sqrt { 3 d } = b _ { d , T } / \sqrt { 3 }$ . Consequently,

$$
a _ { T } \geq \operatorname* { m i n } \left( \sqrt { 3 } a _ { 0 } , c _ { S } , \frac { \alpha } { \sqrt { 3 } } \right) b _ { d , T } .
$$

Therefore, the theorem follows with

$$
c _ { \mathrm { W C } } = \frac { 3 \ell _ { 0 } \zeta \Delta _ { 0 } } { 8 } \operatorname* { m i n } \left( \sqrt { 3 } a _ { 0 } , c _ { S } , \frac { \alpha } { \sqrt { 3 } } \right) ,
$$

and this lower bound is $\Omega \left( \operatorname* { m i n } { \left\{ { \frac { 1 } { \sqrt { d } } } , { \sqrt { \frac { d } { T } } } \right\} } \right)$

## F.4 Two facts for policies that may idle

Lemma 11. Under $\lambda < p _ { - }$ 2

$$
\operatorname* { s u p } _ { T } V _ { T } ^ { \mathrm { I A } } \leq \operatorname* { s u p } _ { T } V _ { T } ^ { \mathrm { W C } } \leq \frac { \lambda ( 1 - \lambda ) } { p _ { - } - \lambda } .
$$

Proof. Run FCFS. In every round with a job in service, couple its departure indicator with a Bernoulli- ${ } ^ { - p _ { - } }$ <sub>−</sub> indicator so that a departure in the comparison system also occurs in the true system. The true queue is then stochastically dominated by the late-arrival $\mathrm { G e o / G e o / 1 }$ queue

$$
\widetilde { Q } _ { t + 1 } = \Big ( \widetilde { Q } _ { t } - S _ { t } \Big ) ^ { + } + A _ { t } , \qquad \mathbb { P } ( S _ { t } = 1 \mid \widetilde { Q } _ { t } > 0 ) = p _ { - } .
$$

Its drift outside zero is $\lambda - p _ { - } < 0$ , so it has a stationary version and its zero-started process is stochastically dominated by the stationary version’s queue. Let $( \widetilde { Q } , S , A )$ have the stationary one-round distribution. Taking expectations in the recursion gives

$$
\begin{array} { r l } & { 0 = \mathbb { E } \left[ \left( ( \widetilde Q - S ) ^ { + } + A \right) \right] - \mathbb { E } \widetilde Q } \\ & { \quad = \mathbb { E } \left[ ( A - S ) \cdot \mathbf { 1 } \left( \widetilde Q > 0 \right) \right] + \mathbb { E } \left[ A \cdot \mathbf { 1 } ( \widetilde Q = 0 ) \right] } \\ & { \quad = \mathbb { E } A - \mathbb { E } \left[ S \cdot \mathbf { 1 } \left( \widetilde Q > 0 \right) \right] } \\ & { \quad = \lambda - p _ { - } \mathbb { P } ( \widetilde Q > 0 ) } \end{array}\tag{45}
$$

and therefore $\mathbb { E } \left[ S \cdot { \mathbf { 1 } } \left( \widetilde { Q } > 0 \right) \right] = \mathbb { E } A = \lambda$ and $\mathbb { P } ( \tilde { Q } > 0 ) = \lambda / p .$ <sub>−</sub>. Since $\mathbb { E } ( S \mid \tilde { Q } ) = \mathbb { P } ( S _ { t } = 1 \mid$ $\widetilde { Q } _ { t } > 0 ) = p .$ <sub>−</sub> for $\widetilde Q > 0$

$$
\begin{array} { r l } & { \mathbb { E } ( \widetilde { Q } S ) = \mathbb { E } \left[ \widetilde { Q } S \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] + \mathbb { E } \left[ \widetilde { Q } S \cdot \mathbf { 1 } ( \widetilde { Q } = 0 ) \right] } \\ & { \qquad = \mathbb { E } \left[ \widetilde { Q } S \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] } \\ & { \qquad = \mathbb { E } \left[ \mathbb { E } ( S \mid \widetilde { Q } ) \widetilde { Q } \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] } \\ & { \qquad = p _ { - } \mathbb { E } \left[ \widetilde { Q } \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] = p _ { - } \mathbb { E } \widetilde { Q } } \end{array}\tag{46}
$$

Moreover, A is independent of $( \widetilde { Q } , S ) , A ^ { 2 } = A$ , and $S ^ { 2 } = S$ . Expanding the square of the stationary recursion gives

$$
\begin{array} { r l } & { 0 = \mathbb { E } \left[ \left( ( \widetilde { Q } - S ) ^ { + } + A \right) ^ { 2 } \right] - \mathbb { E } \widetilde { Q } ^ { 2 } } \\ & { \quad = \mathbb { E } \left[ S \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] + \mathbb { E } A - 2 \mathbb { E } \left[ \widetilde { Q } S \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] + 2 \mathbb { E } \left[ \widetilde { Q } A \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] - 2 \mathbb { E } \left[ S A \cdot \mathbf { 1 } ( \widetilde { Q } > 0 ) \right] } \\ & { \quad = 2 \lambda - 2 p _ { - } \mathbb { E } \widetilde { Q } + 2 \lambda \mathbb { E } \widetilde { Q } - 2 \lambda ^ { 2 } } \\ & { \quad = - 2 ( p _ { - } - \lambda ) \mathbb { E } \widetilde { Q } + 2 \lambda ( 1 - \lambda ) . } \end{array}
$$

Thus $\begin{array} { r } { \mathbb { E } \tilde { Q } = \frac { \lambda ( 1 - \lambda ) } { p _ { - } - \lambda } } \end{array}$ , and this shows that the expected final queue length of the FCFS policy is at most $\frac { \lambda ( 1 - \lambda ) } { p _ { -- } - \lambda }$ by stochastic dominance. FCFS is feasible for both benchmark classes, so the result is proved. □

Lemma 12. Suppose the server is free, only a dummy job with departure probability $p _ { D }$ is waiting, three rounds remain, and every present or future job has departure probability at most $p _ { D }$ . For every policy that idles in the first round, there is a policy that serves the dummy immediately and has expected terminal queue smaller by at least $g _ { D }$

Proof. We first establish a two-round comparison. Suppose a job with departure probability q is waiting and every other present or future job has departure probability at most $q .$ If this job is served immediately, let M be the expected highest departure probability available in the final round, conditional on its departure in the first round. The expected number of departures is

$$
q + ( 1 - q ) q + q M = 2 q - q ^ { 2 } + q M .
$$

Now suppose another waiting job has departure probability $p \leq q$ and is served first. If it departs, the $q$ job is available in the last round; if it remains, nonpreemption requires its service to continue. The expected number of departures is therefore

$$
p + ( 1 - p ) p + p q = 2 p - p ^ { 2 } + p q .
$$

Because the $p$ job remains available after a first-round departure of the q job, $M \geq p$ . Subtracting the two expressions gives

$$
( 2 q - q ^ { 2 } + q M ) - ( 2 p - p ^ { 2 } + p q ) = ( q - p ) ( 2 - q - p ) + q ( M - p ) \ge 0 .
$$

Serving the $q$ job also dominates idling: after idling, at most one departure can occur, with probability at most $q ,$ whereas $2 q - q ^ { 2 } + q M \ge q$

Return to the three-round state in the lemma and condition on the arrivals and their contexts during the next two rounds. Compare the best continuation after serving the dummy in the first round with the best continuation after idling. If the dummy departs in the first round, the former system begins the second round with the same arrivals and without the dummy. Because policies in this comparison may idle, deleting one job cannot increase the minimum expected terminal queue: the system without the dummy can simulate a policy for the larger system and omit every action assigned to the deleted job. If the dummy does not depart, geometric memorylessness gives it a fresh ${ \mathrm { G e o m } } ( p _ { D } )$ residual service time. The two systems then begin the second round with the same jobs, except that immediate service is committed to the dummy whereas the idling policy may choose its next action. The dummy has the highest departure probability, so the two-round comparison above shows that continuing its service is optimal. Immediate service in the first round is therefore no worse for every fixed realization of the arrivals and contexts.

It remains to lower-bound the strict improvement. Condition on no arrival at the ends of the first two rounds; this event has probability $( 1 - \lambda ) ^ { 2 }$ . Apart from an arrival at the end of the third round, which contributes equally under both first actions, the dummy is the only job. If it is served immediately, it remains at the terminal time with probability $( 1 - p _ { D } ) ^ { 3 }$ . After idling in the first round, even the best continuation can serve it in only two rounds, so it remains with probability $( 1 - p _ { D } ) ^ { 2 }$ . The conditional reduction in expected terminal queue is therefore

$$
( 1 - p _ { D } ) ^ { 2 } - ( 1 - p _ { D } ) ^ { 3 } = p _ { D } ( 1 - p _ { D } ) ^ { 2 } .
$$

The comparison is weakly favorable on every other realization of the arrivals. Averaging thus gives the unconditional improvement

$$
( 1 - \lambda ) ^ { 2 } ( 1 - p _ { D } ) ^ { 2 } p _ { D } = g _ { D } ,
$$

so immediate service reduces the expected terminal queue by at least $g _ { D }$

## F.5 Proof of Theorem 2 for IA case

Proof. Define

$$
c _ { 0 } = \operatorname* { m i n } \left( \frac { ( 1 - p _ { D } ) ^ { 4 } } { 1 6 a _ { 0 } } , \frac { 3 \lambda \nu _ { D } g _ { D } } { 3 2 a _ { 0 } } , \frac { 1 1 \ell _ { 0 } \zeta } { 3 2 } \right) .
$$

Suppose for a contradiction that there was a policy $\pi$ (which can be idle) that satisfies ma $\mathrm { x } _ { \omega } R _ { T } ^ { \mathrm { I A } } ( \pi , \omega ) ~ < ~ c _ { 0 } a _ { T }$ . Here, $R _ { T } ^ { \mathrm { I A } } ( \pi , \omega )$ is the regret between policy π and the optimum policy among all idling-allowed policies when the unknown parameter vector was $\theta _ { \omega }$

All departure probabilities are at most $p _ { D }$ by definition. Conditional on $Q _ { T - 3 } ^ { \pi } > 0$ , the probability that no job departs in rounds $T - 3 , T - 2 , T - 1 , T$ is at least $( 1 - p _ { D } ) ^ { 4 }$ , regardless of the policy’s actions. Here we used the fact that every job has departure failure probability at least $1 - p _ { D }$ , and idling can be viewed as having a departure failure probability of 1. On this event the terminal queue is nonempty, and hence

$$
J _ { T } ^ { \pi } = \mathbb { E } \left[ Q _ { T + 1 } ^ { \pi } \cdot { \bf 1 } \left( Q _ { T - 3 } ^ { \pi } > 0 , \mathrm { n o ~ d e p a r t u r e ~ o n ~ l a s t ~ 4 ~ r o u n d s } \right) \right] \geq ( 1 - p _ { D } ) ^ { 4 } \mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } > 0 )
$$

when our parameter vector was $\theta _ { \omega }$ . The assumed regret bound gives $J _ { T } ^ { \pi } = V _ { T } ^ { \mathrm { I A } } + R _ { T } ^ { \mathrm { I A } } ( \pi ) ~ <$ $V _ { T } ^ { \mathrm { I A } } + c _ { 0 } a _ { T }$ . By Lemma 11 and Equation (36), $V _ { T } ^ { \mathrm { I A } } \leq ( 1 - p _ { D } ) ^ { 4 } / 1 6$ , and the definition of $c _ { 0 }$ gives c<sub>0</sub>a<sub>T</sub> $\leq ( 1 - p _ { D } ) ^ { 4 } a _ { T } / 1 6 a _ { 0 } \leq ( 1 - p _ { D } ) ^ { 4 } / 1 6$ . Combining these inequalities with the preceding lower bound yields

$$
\mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } > 0 ) < \frac { 1 } { 8 } .\tag{47}
$$

On $\{ Q _ { T - 3 } ^ { \pi } = 0 \}$ , consider the event that a dummy arrives at the end of round $T - 3$ . Let $\mathcal { T }$ be the event that the learner idles rather than serving this dummy in round $T - 2$ . By Lemma 12, changing only this idle action gives a feasible policy (call it $\pi ^ { * } )$ whose expected terminal cost is smaller by at least $g _ { D } \mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = 1 , X _ { T - 3 } = C _ { D } , \mathbb { Z } )$ . Since $V _ { T } ^ { \mathrm { I A } }$ is the minimum over all policies that may idle,

$$
g _ { D } \mathbb { P } _ { \omega } \big ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = 1 , X _ { T - 3 } = C _ { D } , \mathbb { Z } \big ) \le J _ { T } ^ { \pi } - J _ { T } ^ { \pi ^ { * } } \le J _ { T } ^ { \pi } - V _ { T } ^ { \mathrm { I A } } = R _ { T } ^ { \mathrm { I A } } ( \pi ) < c _ { 0 } a _ { T }
$$

and this gives $\begin{array} { r } { \mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = 1 , X _ { T - 3 } = C _ { D } , \mathcal { Z } ) \le \frac { c _ { 0 } a _ { T } } { a _ { D } } \le { \frac { 3 \lambda \nu _ { D } } { 3 2 } } . } \end{array}$ Now we find a lower bound of $\mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = \overset { \sim } { 1 } , X _ { T - 3 } = C _ { D } )$ to get a lower bound of $\mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = 1 , X _ { T - 3 } = C _ { D } , \mathcal { T } ^ { c } )$ . The dummy arrival at the end of round $T - 3$ is independent of $Q _ { T - 3 } ^ { \pi } .$ So by Equation $( 4 7 ) , \mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = 1 , X _ { T - 3 } = C _ { D } ) = \mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } =$ 0) · $\begin{array} { r } { \lambda \nu _ { D } > \frac { 7 \lambda \nu _ { D } } { 8 } } \end{array}$ , and $\begin{array} { r } { \mathbb { P } _ { \omega } \big ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = 1 , X _ { T - 3 } = C _ { D } , \mathcal { T } ^ { c } \big ) > \frac { 7 \lambda \nu _ { D } } { 8 } - \frac { 3 \lambda \nu _ { D } } { 3 2 } = \frac { 2 5 \lambda \nu _ { D } } { 3 2 } } \end{array}$ . This shows that the probability that the following statements all hold is larger than $2 5 \lambda \nu _ { D } / 3 2 \colon Q _ { T - 3 } ^ { \pi } = 0 .$ , a dummy job arrives at the end of round $T - 3$ , and the learner starts the dummy in round $T - 2$

Next intersect this event with the event that the dummy fails to depart in round $T - 2$ , and a baseline job arrives in round $T - 2$ , followed by departure of the dummy in round $T - 1$ . Let H denote the resulting event before the arrival at the end of round $T - 1$ . Independence gives

$$
\begin{array} { l } { \mathbb { P } _ { \omega } ( \mathcal { H } ) = \mathbb { P } _ { \omega } ( Q _ { T - 3 } ^ { \pi } = 0 , A _ { T - 3 } = 1 , X _ { T - 3 } = C _ { D } , \mathcal { T } ^ { c } ) \cdot ( 1 - p _ { D } ) \cdot \lambda \nu _ { 0 } \cdot p _ { D } } \\ { > \displaystyle \frac { 2 5 } { 3 2 } \lambda ^ { 2 } \nu _ { D } \nu _ { 0 } p _ { D } ( 1 - p _ { D } ) = \frac { 2 5 \zeta } { 3 2 \lambda \nu _ { C } } . } \end{array}\tag{48}
$$

Pair neighboring instances $\omega , \omega ^ { ( j ) }$ and use the same kernel $K _ { j }$ and event $S _ { j } ( \omega )$ as in the workconserving proof. For each $\omega , S _ { j } ( \omega ) ^ { c }$ is a wrong decision and includes selecting the smaller probability job or idling at round $T$ . Data processing through $K _ { j }$ , Equation (39), and Equations (38) and (48) show that their paired wrong-action probability, including the independent candidate arrival, is at least

$$
\frac { \lambda \nu _ { C } } { m } \left( \mathbb { P } _ { \omega } ( \mathcal { H } ) - \mathrm { T V } ( P _ { \omega } ^ { T - 1 } , P _ { \omega ( \mathbb { J } ) } ^ { T - 1 } ) \right) \ge \frac { \lambda \nu _ { C } } { m } \left( \frac { 2 5 \zeta } { 3 2 \lambda \nu _ { C } } - \frac { 3 \zeta } { 3 2 } \right) = \frac { 2 5 \zeta } { 3 2 m } - \frac { 3 \zeta \lambda \nu _ { C } } { 3 2 m } \ge \frac { 1 1 \zeta } { 1 6 m } .
$$

Using the double counting method on the m-dimensional hypercube to get an average of $p _ { \mathrm { e r r } } ( \omega )$ as shown in the work-conserving proof gives an average wrong-action probability at least $\frac { 1 1 \zeta } { 3 2 } . \mathrm { ~ A ~ }$ wrong choice costs $| p _ { 0 } ( \omega ) - p _ { C _ { i } } ( \omega ) | \geq \ell _ { 0 } a _ { T }$ for the case of selecting a small probability job in round $T _ { i }$ and max $\{ p _ { 0 } ( \omega ) , p _ { C _ { j } } ( \omega ) \} \ge \lvert p _ { 0 } ( \omega ) - p _ { C _ { j } } ( \omega ) \rvert \ge \ell _ { 0 } a _ { T }$ for the case of idling in round T (here, $p _ { 0 } ( \omega ) , p _ { C _ { j } } ( \omega )$ mean $p _ { 0 } , p _ { C _ { j } }$ for the parameter vector $\theta _ { \omega } )$ . Both cases have wrong choice costs of at least $\ell _ { 0 } a _ { T }$ Thus there must be some $\omega \in \{ - 1 , 1 \} ^ { m }$ such that $\begin{array} { r } { R _ { T } ^ { \mathrm { I A } } ( \pi , \omega ) \geq \frac { 1 1 \ell _ { 0 } \zeta a _ { T } } { 3 2 } \geq c _ { 0 } a _ { T } } \end{array}$ , a contradiction. Using the final bound on $a T$ in the proof of Theorem 2 for WC case, the theorem holds with

$$
c _ { \mathrm { I A } } = c _ { 0 } \operatorname* { m i n } \left( { \sqrt { 3 } } a _ { 0 } , c _ { S } , { \frac { \alpha } { \sqrt { 3 } } } \right) .
$$

## G Benchmark Structure

We first prove the impossibility results for policies that may idle and for work-conserving policies.   
We then compare the two finite-horizon benchmark values.

## G.1 Proof of Theorem 3 for IA case

Proof. Let there be two kinds of jobs: a slow one and a fast one, which will be denoted as $s , f$ respectively. Define the distribution D so that a slow job comes with probability $\nu = 1 0 ^ { - 4 }$ , and a fast job comes with probability $1 - \nu .$ . Choose

$$
p _ { s } = 0 . 0 1 , \qquad p _ { f } = 0 . 5 , \qquad \lambda = 0 . 0 2 0 5 .
$$

as the success probabilities for each job, and the arrival probability. Use contexts $e _ { 1 } = ( 1 , 0 ) , e _ { 2 } =$ $( 0 , 1 )$ and parameter vector $\theta ^ { * } = ( \mathrm { l o g i t } ( p _ { s } )$ , logit $\left( p _ { f } \right) )$ ). Finally, properly choose fixed constants $S , p _ { - }$ and $p _ { + }$ so that $| \theta ^ { * } | < S , p _ { - } \le 0 . 0 1 , p _ { + } \ge 0 . 5$ . By this definition, its load is

$$
\rho = \lambda \left( \frac { \nu } { p _ { s } } + \frac { 1 - \nu } { p _ { f } } \right) \approx 0 . 0 4 1 2 < 1 .
$$

For the stationary workload increment $Z = A G$ in Lemma $3 .$

$$
\mathbb { E } Z ^ { 2 } = \lambda \mathbb { E } _ { X } \left[ \mathbb { E } _ { g \sim \mathrm { G e o m } ( p ( X ) ) } \left( g ^ { 2 } \right) \right] = \lambda \left( \nu \frac { 2 - p _ { s } } { p _ { s } ^ { 2 } } + ( 1 - \nu ) \frac { 2 - p _ { f } } { p _ { f } ^ { 2 } } \right) \approx 0 . 1 6 3 8 .
$$

Here, we used the fact that $\begin{array} { r } { \mathbb { E } _ { g \sim \mathrm { G e o m } ( p ) } \left( g ^ { 2 } \right) = \frac { 2 - p } { p ^ { 2 } } } \end{array}$ holds. Substituting this value and value of $\rho$ into Equation (15) gives $\mathbb { E } W \approx 0 . 1 0 5 1 < 0 . 2$ . The zero-started workload is stochastically dominated by the stationary distribution workload. Moreover, every work-conserving policy satisfies $Q _ { t + 1 } \leq W _ { t + 1 } ,$ and every such policy is feasible when idling is allowed. Therefore $V _ { t } ^ { \mathrm { I A } } \leq ^ { ^ { \circ } } V _ { t } ^ { \mathrm { W C } } \leq \mathbb { E } W _ { t + 1 } \leq \mathbb { E } W < 0 . 2$ for every t.

We will show $\begin{array} { r } { \operatorname* { s u p } _ { i > t } \{ R _ { i } ^ { \mathrm { I A } } ( \pi ) \} \geq \operatorname* { m i n } \{ \frac { \lambda \nu } { 4 } p _ { s } , \frac { \lambda \nu } { 4 } g , 0 . 3 \} } \end{array}$ holds for all $t \geq 1$ . Here $g = p _ { s } ^ { 2 } - p _ { s } ( 1 +$ $\lambda ) + \lambda ( 1 - p _ { s } ) \mathbb { E } p ( X )$ ≈ $4 . 1 5 0 6 \times 1 0 ^ { - 5 } > 0$ , and this fact shows that lim sup $R _ { t } ^ { \mathrm { { I A } } } ( \pi )$ is at least min $\{ \frac { \lambda \nu } { 4 } p _ { s } , \frac { \lambda \nu } { 4 } g , 0 . 3 \}$ , which is a fixed positive constant.

For simplicity, let C = min $\{ \frac { \lambda \nu } { 4 } p _ { s } , \frac { \lambda \nu } { 4 } g , 0 . 3 \}$ and suppose, for a contradiction, there exists a $t _ { 0 } \geq 1$ such that s $\mathrm { 4 p } _ { i \geq t _ { 0 } } \{ R _ { i } ^ { \mathrm { I A } } ( \pi ) \} < C _ { 0 }$ holds. So $R _ { t } ^ { \mathrm { I A } } ( \pi ) < C _ { 0 }$ for all $t \geq t _ { 0 }$ . Since $J _ { t } ^ { \pi } = V _ { t } ^ { \mathrm { I A } } + R _ { t } ^ { \mathrm { I A } } ( \pi )$ and $V _ { t } ^ { \mathrm { I A } } < 0 . 2$ for all t, we have $\mathbb { E } _ { \pi } Q _ { t + 1 } < 0 . 2 + C _ { 0 } \le 0 . 5$ for all $t \geq t _ { 0 }$ . Markov’s inequality for the nonnegative integer $Q _ { t + 1 }$ gives $\mathbb { P } _ { \pi } ( Q _ { t + 1 } = 0 ) = 1 - \mathbb { P } _ { \pi } ( Q _ { t + 1 } \geq 1 ) \geq 1 - \mathbb { E } _ { \pi } Q _ { t + 1 } > 1 / 2$ . Now fix $t \geq t _ { 0 }$ . Then the arrival at the end of round $t + 1$ is independent of the event $\{ Q _ { t + 1 } = 0 \}$ , which means the queue became empty at the beginning of round $t + 1$ . Hence, with probability greater than $\lambda \nu / 2$ , the system is empty at the beginning of round $t + 1$ and a slow job arrives at the end of that round. At the beginning of round $t + 2$ , the server is free and exactly one slow job is waiting.

If the policy idles in this state, changing only this action to service reduces the expected terminal queue with horizon $t + 2$ by $p _ { s } .$ , since the expected number of departures difers by $p _ { s }$ in the final round. Now evaluate the same decision with horizon $t + 3 ,$ , so that two rounds remain. If the policy serves immediately, then the server has two choices, idle or work, in the final round $t + 3$ if the server is free and the queue is not zero at the beginning of round $t + 3$ . In that case, choosing work in round $t + 3$ makes the expected number of departures maximal, so in these two rounds the maximum of the expected number of departures is

$$
\begin{array} { r } { p _ { s } ( 1 + \lambda \mathbb { E } p ( X ) ) + ( 1 - p _ { s } ) p _ { s } = 2 p _ { s } - p _ { s } ^ { 2 } + \lambda p _ { s } \mathbb { E } p ( X ) . } \end{array}
$$

If it follows the policy that idles in round $t + 2$ and serves the fastest job available in the last round $t + 3$ , the expected number of departures is

$$
( 1 - \lambda ) p _ { s } + \lambda \mathbb { E } \operatorname* { m a x } \{ p _ { s } , p ( X ) \} = ( 1 - \lambda ) p _ { s } + \lambda \mathbb { E } p ( X ) ,
$$

where the equality uses $p ( X ) \geq p _ { s }$ in this instance. Thus idling first has an expected-departure advantage at least

$$
g = p _ { s } ^ { 2 } - p _ { s } ( 1 + \lambda ) + \lambda ( 1 - p _ { s } ) \mathbb { E } p ( X ) \approx 4 . 1 5 0 6 \times 1 0 ^ { - 5 } > 0 .
$$

So the expectation of queue length decreases by at least g if we change our decision to idle in round $t + 2$ , rather than serve a slow job.

Let H denote the event defined above: $Q _ { T + 1 } = 0$ and a slow job arrived at the end of round $t + 1$ , so the server is free and the slow job is the only waiting job at the beginning of round $t + 2$ Let $\mathbf { H } _ { t + 1 }$ denote the random history through the end of round $t + 1$ , and, for each realization ${ \mathfrak { h } } \in { \mathcal { H } }$ define

$$
a ( \mathfrak { h } ) : = \mathbb { P } _ { \pi } \big ( \pi \mathrm { ~ s e r v e s ~ t h e ~ s l o w ~ j o b ~ i n ~ r o u n d ~ } t + 2 | \mathbf { H } _ { t + 1 } = \mathfrak { h } \big ) .
$$

Because π is a horizon-free policy, it uses the same conditional randomized decision rule after the same history h, regardless of whether the evaluation horizon is $t + 2$ or $t + 3$ (this is because the server doesn’t know the horizon). A comparison policy that changes only this action is feasible because idling is allowed. Therefore, applying the preceding comparison argument conditionally on each history ${ \mathfrak { h } } \in { \mathcal { H } }$ and then averaging over the common distribution of $\mathbf { H } _ { t + 1 }$ gives

$$
R _ { t + 2 } ^ { \mathrm { I A } } ( \pi ) + R _ { t + 3 } ^ { \mathrm { I A } } ( \pi ) \geq \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { H } } \left( ( 1 - a ( \mathbf { H } _ { t + 1 } ) ) p _ { s } + a ( \mathbf { H } _ { t + 1 } ) g \right) \right] .
$$

Since $( 1 - a ) p _ { s } + a g \ge \operatorname* { m i n } \{ p _ { s } , g \}$ for every $a \in [ 0 , 1 ]$ , and since $\mathbb { P } ( \mathcal { H } ) \ge \lambda \nu / 2$ , it follows that

$$
R _ { t + 2 } ^ { \mathrm { I A } } ( \pi ) + R _ { t + 3 } ^ { \mathrm { I A } } ( \pi ) \geq \mathbb { P } ( \mathcal { H } ) \operatorname* { m i n } \{ p _ { s } , g \} \geq \frac { \lambda \nu } { 2 } \operatorname* { m i n } \{ p _ { s } , g \} > 0 .
$$

But by assumption, since $t \geq t _ { 0 }$ the following inequality holds.

$$
R _ { t + 2 } ^ { \mathrm { I A } } ( \pi ) + R _ { t + 3 } ^ { \mathrm { I A } } ( \pi ) < 2 C _ { 0 } = 2 \operatorname* { m i n } \{ \frac { \lambda \nu } { 4 } p _ { s } , \frac { \lambda \nu } { 4 } g , 0 . 3 \} \le \frac { \lambda \nu } { 2 } \operatorname* { m i n } \{ p _ { s } , g \}
$$

This gives a contradiction; therefore we have proved that $\begin{array} { r } { \operatorname* { s u p } _ { i > t } \{ R _ { i } ^ { \mathrm { I A } } ( \pi ) \} \geq \operatorname* { m i n } \{ \frac { \lambda \nu } { 4 } p _ { s } , \frac { \lambda \nu } { 4 } g , 0 . 3 \} } \end{array}$ holds. Since $g < p _ { s }$ and $\begin{array} { r } { \frac { \lambda \nu } { 4 } p _ { s } < \frac { 1 } { 4 } < 0 . 3 } \end{array}$ hold in the above setting, min $\begin{array} { r } { \{ \frac { \lambda \nu } { 4 } p _ { s } , \frac { \lambda \nu } { 4 } g , 0 . 3 \} = \frac { \lambda \nu } { 4 } g } \end{array}$ holds and we can simplify the result as follows: for every $t \geq 1$

$$
\operatorname* { s u p } _ { i \geq t } \{ R _ { i } ^ { \mathrm { I A } } ( \pi ) \} \geq \frac { \lambda \nu } { 4 } g \approx 2 . 1 2 7 2 \times 1 0 ^ { - 1 1 } .
$$

## G.2 Exact certificate for Proposition 1

Define three jobs as A, B, F, and set their arrival probabilities $\nu _ { A } = 1 / 4 , \nu _ { B } = 1 / 1 0 0$ , and $\nu _ { F } = 3 7 / 5 0$ Together, we set their departure probabilities as $p _ { A } = 7 / 2 0 , p _ { B } = 1 3 / 2 0 , p _ { F } = 1 9 / 2 0$ and $\lambda = 1 / 2$ These arrival probabilities sum to one, and give

$$
\rho = \frac { 1 } { 2 } \left( \frac { \nu _ { A } } { p _ { A } } + \frac { \nu _ { B } } { p _ { B } } + \frac { \nu _ { F } } { p _ { F } } \right) = \frac { 6 5 2 1 } { 8 6 4 5 } < 1 .
$$

Let $c = ( c _ { A } , c _ { B } , c _ { F } )$ record the waiting job counts, and let $a \in \{ 0 , A , B , F \}$ denote the type in service; $a = 0$ means that the server is free. The job in service is not included in $c ,$ and $e _ { i }$ increments the waiting count of type i. For a function $f ,$ recall the definition of the arrival operator which is defined in Section 4.2.

$$
( A f ) ( c , a ) = ( 1 - \lambda ) f ( c , a ) + \lambda \sum _ { i \in \{ A , B , F \} } \nu _ { i } f ( c + e _ { i } , a ) .
$$

Now we will use the Bellman recursion formula which was first shown in Section 4.2. Let $V _ { h } ( c , a )$ be the minimum expected terminal queue with h rounds remaining. Its boundary value is $V _ { 0 } ( c , a ) =$ $c _ { A } + c _ { B } + c _ { F } + 1 \{ a \neq 0 \}$ , which denotes the number of remaining jobs (waiting or in service) in the queue. A type-a job in service forces the recursion

$$
V _ { h } ( c , a ) = p _ { a } ( { \cal A } V _ { h - 1 } ) ( c , 0 ) + ( 1 - p _ { a } ) ( { \cal A } V _ { h - 1 } ) ( c , a ) .\tag{49}
$$

If the server is free and $c \neq 0 ,$ , define the value of starting type i for i with $c _ { i } > 0$ by

$$
Q _ { h } ( i ; c ) = p _ { i } ( { \cal A } V _ { h - 1 } ) ( c - e _ { i } , 0 ) + ( 1 - p _ { i } ) ( { \cal A } V _ { h - 1 } ) ( c - e _ { i } , i )
$$

Work conservation gives

$$
V _ { h } ( c , 0 ) = \operatorname* { m i n } _ { i : c _ { i } > 0 } Q _ { h } ( i ; c ) .\tag{50}
$$

When $c = 0 , V _ { h } ( 0 , 0 ) = ( A V _ { h - 1 } ) ( 0 , 0 )$ . For each fixed finite horizon h, these recursions visit only finitely many states because the distribution has finite support. Moreover, since all probability values are rational, every Bellman term can be computed exactly using rational arithmetic.

For the remainder of this subsection, abbreviate $Q _ { h } ( i ) = Q _ { h } ( i ; e _ { A } + e _ { B } )$ . Thus, $Q _ { h } ( i )$ denotes the expected terminal queue length when the system starts with h rounds remaining, a free server, and the initial queue A, B, and job i is selected for service in the first round. By calculating the Bellman recursion, we can show that the one-round action values are $Q _ { 1 } ( A ) = 4 3 / 2 0$ and $Q _ { 1 } ( B ) = 3 7 / 2 0$ , so B is uniquely optimal with gap 3/10. Iterating Equations (49) and (50) to $h = 1 4$ gives following values of $Q _ { 1 4 } ( A ) , Q _ { 1 4 } ( B )$ . Here, both of $Q _ { 1 4 } ( A ) , Q _ { 1 4 } ( B )$ are rational.

$$
\begin{array} { c } { { Q _ { 1 4 } ( A ) \approx 1 . 6 3 7 5 , \qquad Q _ { 1 4 } ( B ) \approx 1 . 6 3 9 6 , } } \\ { { Q _ { 1 4 } ( B ) - Q _ { 1 4 } ( A ) \approx 0 . 0 0 2 1 > 0 } } \end{array}\tag{51}
$$

Thus A is uniquely optimal. Taking contexts $e _ { A } , e _ { B } , e _ { F }$ and $\theta ^ { * } = ( \log \mathrm { i t } p _ { A }$ , logit p<sub>B</sub> , logit $p _ { F } )$ gives a three-dimensional logistic representation with finite S. The accompanying script evaluates the same recursion using exact fractions and asserts both action inequalities.

The preceding example further shows that, in a finite-horizon, work-conserving, nonpreemptive system, a policy that selects a job with the highest departure probability whenever the server becomes free need not be optimal. Indeed, when only one round remains, serving B, which has the higher departure probability, maximizes the expected number of departures and hence minimizes the expected terminal queue length. In contrast, when 14 rounds remain, serving A, which has the lower departure probability, results in a larger expected number of departures and hence a smaller expected terminal queue length. Thus, the optimal decision may depend on the number of remaining rounds and need not coincide with the myopic maximum-departure-probability rule. This is one of the fundamental distinctions between the preemptive and nonpreemptive settings: in the finite-horizon, work-conserving preemptive setting, it is well known that an optimal policy selects a currently available job–server pair with the highest departure probability in every round. See Bae and Lee (2026) for more detail.

## G.3 Proof of Theorem 3 for WC case

Proof. Use the instance from Proposition 1. By Equation (41), the zero-started common workload satisfies $\mathbb { P } ( W _ { t } = 0 ) \ge 1 - \rho$ for every t. Fix $s \geq 5$ , and starting from the event $W _ { s - 4 } = 0$ , consider the following event: an F job arrives at the end of round $s - 4 ;$ it remains in service in round $s - 3$ and an A job arrives; it remains in service in round $s - 2$ and a B job arrives; and it departs in round $s - 1$ , with no arrival at the end of that round. The F job is the only job present when its service begins, and nonpreemption fixes all three service decisions. At the beginning of round $s ,$ the server is free with exactly one A and one B job waiting. Define the above event as ${ \mathcal { H } } _ { \mathrm { W C } }$ Independence of the arrivals and departure indicators gives the lower bound of the probability that this kind of event happens.

$$
\begin{array} { r l } & { \mathbb { P } ( \mathcal { H } _ { \mathrm { W C } } ) = \mathbb { P } ( W _ { s - 4 } = 0 ) \cdot \lambda \nu _ { F } \cdot ( 1 - p _ { F } ) \cdot \lambda \nu _ { A } \cdot ( 1 - p _ { F } ) \cdot \lambda \nu _ { B } \cdot p _ { F } \cdot ( 1 - \lambda ) } \\ & { \qquad \ge ( 1 - \rho ) \lambda ^ { 3 } ( 1 - \lambda ) \nu _ { F } \nu _ { A } \nu _ { B } p _ { F } ( 1 - p _ { F } ) ^ { 2 } } \end{array}\tag{52}
$$

Here, we define $\begin{array} { r } { \zeta _ { \mathrm { W C } } = ( 1 - \rho ) \lambda ^ { 3 } ( 1 - \lambda ) \nu _ { F } \nu _ { A } \nu _ { B } p _ { F } ( 1 - p _ { F } ) ^ { 2 } \approx 6 . 7 4 6 9 \times 1 0 ^ { - 8 } . } \end{array}$

Let ${ \mathbf { H } } _ { s - 1 }$ denote the random history through the end of round $s - 1$ . For each realization $\mathfrak { h } \in \mathcal { H } _ { \mathrm { W C } }$ , define

$$
\alpha _ { s } ( \mathfrak { h } ) : = \mathbb { P } _ { \pi } \left( \pi \ \mathrm { s t a r t s } \ B \ \mathrm { i n \ r o u n d } \ s | \mathbf { H } _ { s - 1 } = \mathfrak { h } \right) .
$$

Because $\pi$ is horizon-free, it uses the same conditional randomized decision rule after the same history h, regardless of whether the terminal horizon is s or $s + 1 3$

With terminal horizon s, one round remains after each history $\mathfrak { h } \in \mathcal { H } _ { \mathrm { W C } }$ . A comparison policy that changes only this last action from A to B is feasible and work-conserving. Therefore

$$
R _ { s } ^ { \mathrm { W C } } ( \pi ) \geq \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { H } _ { \mathrm { W C } } } \left( 1 - \alpha _ { s } ( \mathbf { H } _ { s - 1 } ) \right) ( p _ { B } - p _ { A } ) \right] .
$$

Now evaluate the same infinite policy at horizon $s + 1 3$ . Fourteen rounds remain after the same history. If the policy starts $B ,$ even its best continuation has value $Q _ { 1 4 } ( B )$ , whereas a comparison policy that starts A and then continues optimally has value $Q _ { 1 4 } ( A )$ . Hence

$$
R _ { s + 1 3 } ^ { \mathrm { W C } } ( \pi ) \geq \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { H } _ { \mathrm { W C } } } \alpha _ { s } ( \mathbf { H } _ { s - 1 } ) \big ( Q _ { 1 4 } ( B ) - Q _ { 1 4 } ( A ) \big ) \right] .
$$

Adding the two inequalities and using $( 1 - \alpha ) x + \alpha y \geq \operatorname* { m i n } \{ x , y \}$ for every $\alpha \in [ 0 , 1 ]$ and $x , y > 0$ gives

$$
\begin{array} { r l } & { R _ { s } ^ { \mathrm { W C } } ( \pi ) + R _ { s + 1 3 } ^ { \mathrm { W C } } ( \pi ) \geq \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { H } _ { \mathrm { W C } } } \left( \left( 1 - \alpha _ { s } ( \mathbf { H } _ { s - 1 } ) \right) ( p _ { B } - p _ { A } ) + \alpha _ { s } ( \mathbf { H } _ { s - 1 } ) \left( Q _ { 1 4 } ( B ) - Q _ { 1 4 } ( A ) \right) \right) \right] } \\ & { \qquad \geq \mathbb { P } ( \mathcal { H } _ { \mathrm { W C } } ) \operatorname* { m i n } \left\{ p _ { B } - p _ { A } , Q _ { 1 4 } ( B ) - Q _ { 1 4 } ( A ) \right\} } \\ & { \qquad \geq \zeta _ { \mathrm { W C } } \operatorname* { m i n } \left\{ p _ { B } - p _ { A } , Q _ { 1 4 } ( B ) - Q _ { 1 4 } ( A ) \right\} \approx 1 . 4 2 5 \times 1 0 ^ { - 1 0 } . } \end{array}\tag{53}
$$

So this inequality shows that for $s \geq 5$

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { i \geq s } R _ { s } ^ { \mathrm { W C } } ( \pi ) \geq \frac { 1 } { 2 } \left( R _ { s } ^ { \mathrm { W C } } ( \pi ) + R _ { s + 1 3 } ^ { \mathrm { W C } } ( \pi ) \right) } \\ { \displaystyle \qquad \geq \frac { 1 } { 2 } \zeta _ { \mathrm { W C } } \operatorname* { m i n } \left\{ p _ { B } - p _ { A } , Q _ { 1 4 } ( B ) - Q _ { 1 4 } ( A ) \right\} } \\ { \displaystyle \qquad \approx 7 . 1 2 5 \times 1 0 ^ { - 1 1 } . } \end{array}
$$

## G.4 A comparison between the benchmark classes

Theorem 5. For every $T \geq 1 , 0 \leq V _ { T } ^ { \mathrm { W C } } - V _ { T } ^ { \mathrm { I A } } \leq C _ { W } ( r )$ . For any policy π, let $\begin{array} { r } { D _ { T } ^ { \pi } : = \mathbb { E } _ { \pi } \sum _ { t = 1 } ^ { T } D _ { t } = } \end{array}$ $\lambda T - J _ { T } ^ { \pi }$ be its expected number of departures. In particular, $D _ { T } ^ { \mathrm { I A } } = \lambda T - V _ { T } ^ { \mathrm { I A } }$ and $\mathsf { \bar { \Pi } } _ { T } ^ { \mathrm { W C } } = \lambda T - V _ { T } ^ { \mathrm { W C } }$ Then

$$
D _ { T } ^ { \mathrm { W C } } \geq \left( 1 - \frac { C _ { W } ( r ) } { \lambda T } \right) ^ { + } D _ { T } ^ { \mathrm { I A } } .
$$

Moreover, a work-conserving learner with regret $R _ { T } ^ { \mathrm { W C } } ( \pi )$ satisfies

$$
D _ { T } ^ { \pi } \geq \left( 1 - \frac { C _ { W } ( r ) + R _ { T } ^ { \mathrm { W C } } ( \pi ) } { \lambda T } \right) ^ { + } D _ { T } ^ { \mathrm { I A } } .
$$

Proof. Every work-conserving policy is also feasible when idling is allowed, so $V _ { T } ^ { \mathrm { I A } } \leq V _ { T } ^ { \mathrm { W C } }$ . Under a work-conserving policy, $Q _ { T + 1 } \leq W _ { T + 1 }$ pathwise, and Lemma 1 gives $V _ { T } ^ { \mathrm { W C } } \leq \dot { C } _ { W } ( r )$ . This proves the additive comparison.

Summing $Q _ { t + 1 } = Q _ { t } - D _ { t } + A _ { t }$ from $t = 1$ to $T$ and using $Q _ { 1 } = 0$ gives

$$
Q _ { T + 1 } = \sum _ { t = 1 } ^ { T } A _ { t } - \sum _ { t = 1 } ^ { T } D _ { t } .
$$

Taking expectations and using $\mathbb { E } A _ { t } = \lambda$ yields $D _ { T } ^ { \pi } = \lambda T - J _ { T } ^ { \pi }$ . Hence $D _ { T } ^ { \mathrm { W C } } \geq \lambda T - C _ { W } ( r )$ and $D _ { T } ^ { \mathrm { I A } } \leq \lambda T$ . If $\lambda T \le C _ { W } ( r )$ , the first multiplicative bound has a zero right-hand side. Otherwise,

$$
D _ { T } ^ { \mathrm { W C } } \geq \lambda T - C _ { W } ( r ) \geq \left( 1 - { \frac { C _ { W } ( r ) } { \lambda T } } \right) D _ { T } ^ { \mathrm { I A } } .
$$

Finally, $J _ { T } ^ { \pi } \le V _ { T } ^ { \mathrm { W C } } + R _ { T } ^ { \mathrm { W C } } ( \pi ) \le C _ { W } ( r ) + R _ { T } ^ { \mathrm { W C } } ( \pi )$ , and the same two cases prove the learner bound. □

## H Anytime SEPT Tracking

We first state the prediction guarantee and the busy-period bound used in the proof of Theorem 4. We then prove the theorem and establish each supporting result. The two sigma-fields below record the order of service, estimation, and arrivals within a round.

Let $\mathcal { F } _ { t }$ be the σ-algebra generated by all information about the queue, the job in service, the observations, and the learner’s randomization available at the beginning of round t. The scheduling decision is $\mathcal { F } _ { t } .$ -measurable. Let $\mathcal { G } _ { t }$ be the σ-algebra obtained by additionally including $D _ { t }$ and any estimator update made after service in round $t ,$ but not $( A _ { t } , X _ { t } )$ . Under Assumption 1, $( A _ { t } , X _ { t } )$ is independent of $\mathcal { G } _ { t }$ , with $A _ { t } \sim \operatorname { B e r n } ( \lambda )$ and $X _ { t } \sim \mathcal { D }$ independent of $A _ { t }$

The following three lemmas are the main ingredients in the proof of Theorem 4. First, Lemma 13 shows that, with high probability, every suficiently recent busy period uses a predictor with small prediction error. Recall that $\begin{array} { r } { u = \left\lfloor \frac { t } { 2 } \right\rfloor , m _ { t } = \left\lfloor \frac { \lambda u } { 2 } \right\rfloor , b _ { t } = e ^ { - \lambda u / 8 } + \frac { 1 } { t ^ { 4 } } } \end{array}$

Lemma 13. Fix $t \geq 6$ such that $m _ { t } \geq 5$ . Then there exists an event $B _ { t }$ with $\mathbb { P } ( B _ { t } ) \le b _ { t }$ such that, on $B _ { t } ^ { c }$ , every predictor used in a busy period beginning in a round s with $u + 2 \leq s \leq t$ satisfies $\mathcal { E } ( \widehat { p } ) \leq \bar { \epsilon } _ { t }$

For $s \geq 2 .$ , let $ { \boldsymbol { S } } _ { s }$ denote the event that the system is empty immediately after service in round $s - 1$ and that an arrival occurs at the end of that round. Thus, on $S _ { s } ,$ , a new busy period begins in round s. The next Lemma 14 shows that, under two work-conserving policies, the contribution to the expected terminal queue-length diference from a busy period beginning in round s and remaining active up to round t decays exponentially in its age $h = t - s + 1$

Lemma 14. Let $h = t - s + 1$ , and let $C _ { s }$ be the event that a busy period begins in round s and remains active after service in round $t ;$ equivalently, if we denote $\tau _ { \mathrm { b p } }$ as the length of the busy period starting at round s conditional on the event $S _ { s } , C _ { s }$ is defined as $S _ { s } \cap \{ \tau _ { \mathrm { b p } } > h \}$ . Assign each job the same service time under two work-conserving ordering rules. If their queue lengths after round t are $Q _ { t + 1 } ^ { ( 1 ) }$ and $Q _ { t + 1 } ^ { ( 2 ) }$ , define $Z _ { s , t } = Q _ { t + 1 } ^ { ( 1 ) } - Q _ { t + 1 } ^ { ( 2 ) }$ . Then

$$
\mathbb { E } \big ( | Z _ { s , t } | \mathbf { 1 } \{ C _ { s } \} \big ) \le \mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ C _ { s } \} \big ) \le \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { h } .\tag{54}
$$

We refer to the above inequality as the busy-period tail bound.

Finally, Lemma 15 shows that if a predictor $q ,$ fixed before a busy period begins, approximates the true departure-probability function p with small prediction error, then $q { \mathrm { - S E P T } }$ and true SEPT have close expected terminal queue lengths when both policies are evaluated under the true model $p .$ Here, $q { \mathrm { - S E P T } }$ is the priority policy that, whenever the server becomes free, selects a waiting job maximizing $q ( X )$ , thereby treating $q$ as the departure-probability function. For a priority rule $\pi _ { \ i }$ $J _ { h , \mathrm { s t o p } } ^ { p } ( \pi )$ denotes the expected queue length after h rounds when the actual departure probability of a job with context X is $p ( X )$ and the jobs are selected according to π, with the comparison stopped as soon as service leaves no unfinished job. Thus, $J _ { h , \mathrm { s t o p } } ^ { p } ( q \mathrm { - S E P T } )$ is the expected terminal queue length when jobs are ranked using q, while their actual departures are governed by $p .$

Lemma 15. Suppose ${ \mathcal { E } } ( q ) = \mathbb { E } | q ( X ) - p ( X ) | = \epsilon$ . Run a busy period for h rounds, beginning with the first round in which its initial job is served, and stop both comparison systems as soon as service leaves no unfinished job. Then

$$
\left| J _ { h , \mathrm { s t o p } } ^ { p } ( q - S E P T ) - J _ { h , \mathrm { s t o p } } ^ { p } ( p - S E P T ) \right| \leq \frac { 2 ( h + 1 ) ^ { 2 } } { p _ { - } } \epsilon .\tag{55}
$$

We conclude this introductory part preceding the main proofs by introducing the function $\Phi _ { r } ( \epsilon )$ which appears in the statement of Theorem 4. For $\epsilon \in ( 0 , 1 ]$ , define

$$
\Phi _ { r } ( \epsilon ) = \sum _ { a = 0 } ^ { \infty } \operatorname* { m i n } \left( \frac { 2 ( a + 2 ) ^ { 2 } } { p _ { - } } \epsilon , \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { a + 1 } \right) .\tag{56}
$$

Here, $M _ { G } ( r )$ and $\psi ( r )$ are constants defined at the beginning of Appendix C. Splitting the series at the smallest a for which $\psi ( r ) ^ { a } \leq \varepsilon$ and bounding the terms appropriately, we obtain $\Phi _ { r } ( \varepsilon ) =$ $O \left( \varepsilon \left( 1 + \log ^ { 3 } ( 1 / \varepsilon ) \right) \right)$ , where the constant depends only on $r , p _ { - } , M _ { G } ( r )$ , and $\psi ( r )$ . A detailed proof of this asymptotic bound is provided in Appendix H.6.

## H.1 Proof of Theorem 4

Proof. Run Algorithm 2 and true SEPT with the same arrivals and contexts, and assign each job the same geometric service time in the two systems. Both policies are work conserving, so Equation (2)

gives the same workload path. In particular, the two systems become empty after service in exactly the same rounds and therefore have the same busy-period boundaries (i.e. the starting/ending times of busy periods are all the same in both queues).

For $s \in \{ 2 , \ldots , t \}$ , let $q _ { s }$ denote the predictor held by the learner immediately after service and any estimator update in round $s - 1$ . On $S _ { s }$ , this predictor is fixed before the context of the job initiating the new busy period is observed and remains fixed throughout that busy period. Starting with this initial job which arrived at the end of round $s - 1$ , compare $q _ { s } – \mathrm { S E P T }$ and true SEPT using the same subsequent arrivals, contexts, and service times. Stop both comparison systems as soon as service leaves no unfinished job. With $h = t - s + 1$ , let $Z _ { s , t }$ be the diference between their queue lengths after these h rounds, where both stopped queue lengths are defined to be zero if the busy period has already ended.

By definition, on $\mathcal { S } _ { s } \ \backslash \mathcal { C } _ { s }$ , the busy period that begins in round s ends by the completion of service in round t. The two stopped queue lengths are therefore both zero, and hence $Z _ { s , t } = 0$ Consequently, E $( Z _ { s , t } \mathbf { 1 } \{ S _ { s } \} ) = \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ \mathcal { C } _ { s } \} \right)$ for every s.

Moreover, only the last busy period occurring during the first t rounds can contribute to the terminal queue-length diference. If no busy period remains active after service in round t, the two coupled systems have the same terminal queue length. It is therefore suficient to restrict attention to the event $\textstyle \bigcup _ { s = 2 } ^ { t } { \mathcal { C } } _ { s }$ . Since the events $\mathcal { C } _ { 2 } , \ldots , \mathcal { C } _ { t }$ are pairwise disjoint and $Z _ { s , t }$ agrees with the actual terminal queue-length diference on $\mathcal { C } _ { s } ,$ , we obtain

$$
\begin{array} { r l } & { J _ { t } ^ { \mathrm { a l g } } - J _ { t } ^ { \mathrm { S E P T } } = \mathbb { E } \left( \left( Q _ { t + 1 } ^ { \mathrm { a l g } } - Q _ { t + 1 } ^ { \mathrm { S E P T } } \right) \mathbf { 1 } \{ \bigcup _ { s = 2 } ^ { t } \mathcal { C } _ { s } \} \right) } \\ & { \qquad = \displaystyle \sum _ { s = 2 } ^ { t } \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ \mathcal { C } _ { s } \} \right) = \displaystyle \sum _ { s = 2 } ^ { t } \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ \mathcal { S } _ { s } \} \right) . } \end{array}\tag{57}
$$

A busy period beginning in round $t + 1$ need not be included, since its initial job has not yet received service and is present in both terminal queues.

For the statistical comparison, define $E _ { s } = \{ \mathcal { E } ( q _ { s } ) \leq \bar { \epsilon } _ { t } \}$ and let $\mathcal { H } _ { s } = \mathcal { G } _ { s - 1 } \vee \sigma ( A _ { s - 1 } )$ . The predictor $q _ { s }$ and the events $\boldsymbol { S _ { s } }$ and $E _ { s }$ are $\mathcal { H } _ { s }$ -measurable. Conditional on $\mathcal { H } _ { s }$ , the predictor $q _ { s }$ is fixed. Moreover, on the event $\mathcal { S } _ { s } ,$ the context of the initial job of the busy period beginning in round s has distribution D, and the subsequent arrival process is independent of $\mathcal { H } _ { s }$

Recall that $Z _ { s , t }$ is defined as the diference between the terminal queue lengths in the stopped h-round comparison of $q _ { s } – \mathrm { S E P T }$ and true SEPT, where $h = t - s + 1$ . Therefore, conditional on $\mathcal { H } _ { s } .$ on the event $\boldsymbol { S _ { s } }$ , its conditional expectation is precisely the diference between the two expected terminal queue lengths appearing in Lemma 15:

$$
\begin{array} { r } { \mathbb { E } \left( Z _ { s , t } \mid \mathcal { H } _ { s } \right) = J _ { h , \mathrm { s t o p } } ^ { p } \left( q _ { s } \mathrm { - S E P T } \right) - J _ { h , \mathrm { s t o p } } ^ { p } \left( p \mathrm { - S E P T } \right) . } \end{array}
$$

Consequently, Lemma 15 applies conditionally and gives, on $S _ { s }$ ,

$$
| \mathbb { E } \left( Z _ { s , t } \mid \mathcal { H } _ { s } \right) | \leq \frac { 2 ( h + 1 ) ^ { 2 } } { p _ { - } } \mathcal { E } ( q _ { s } ) .
$$

In particular, on $S _ { s } \cap E _ { s }$

$$
\big \vert \mathbb { E } \left( Z _ { s , t } \mid \mathcal { H } _ { s } \right) \big \vert \leq \frac { 2 ( h + 1 ) ^ { 2 } } { p _ { - } } \bar { \epsilon } _ { t } .
$$

Since $S _ { s } \cap E _ { s }$ is $\mathcal { H } _ { s } .$ -measurable, the tower property then gives

$$
\begin{array} { r l } & { \left| \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ S _ { s } \cap E _ { s } \} \right) \right| \le \mathbb { E } \left[ \mathbf { 1 } \{ S _ { s } \cap E _ { s } \} \left| \mathbb { E } \left( Z _ { s , t } \mid \mathcal { H } _ { s } \right) \right| \right] } \\ & { \qquad \le \displaystyle \frac { 2 ( h + 1 ) ^ { 2 } } { p _ { - } } \bar { \epsilon } _ { t } . } \end{array}
$$

On $\mathcal { S } _ { s } \backslash C _ { s }$ , the busy period has ended by round t, so the stopped discrepancy $Z _ { s , t }$ is zero. It follows from Lemma 14 that

$$
\begin{array} { r l } {  { \big \vert \mathbb { E } \big ( Z _ { s , t } \mathbf { 1 } \{ \boldsymbol { S _ { s } } \cap E _ { s } \} \big ) \big \vert = \big \vert \mathbb { E } \big ( Z _ { s , t } \mathbf { 1 } \{ \mathcal { C } _ { s } \cap E _ { s } \} \big ) \big \vert } \quad } & { } \\ & { \leq \mathbb { E } \big ( \vert Z _ { s , t } \vert \mathbf { 1 } \{ C _ { s } \cap E _ { s } \} \big ) } \\ & { \leq \mathbb { E } \big ( \vert Z _ { s , t } \vert \mathbf { 1 } \{ C _ { s } \} \big ) \leq \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { h } . } \end{array}
$$

Combining the preceding two bounds yields

$$
\big | \mathbb { E } \big ( Z _ { s , t } \mathbf { 1 } \{ { \mathcal { S } } _ { s } \cap E _ { s } \} \big ) \big | \leq \operatorname* { m i n } \left\{ \frac { 2 ( h + 1 ) ^ { 2 } } { p _ { - } } \bar { \epsilon } _ { t } , \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { h } \right\} .\tag{58}
$$

For $s \in \{ u + 2 , \ldots , t \}$ , set $a = h - 1 = t - s$ . Summing equation 58 over these starting rounds gives

$$
\sum _ { s = u + 2 } ^ { t } | \mathbb { E } ( Z _ { s , t } \mathbf { 1 } \{ S _ { s } \cap E _ { s } \} ) | \leq \sum _ { a = 0 } ^ { t - u - 2 } \operatorname* { m i n } \left\{ \frac { 2 ( a + 2 ) ^ { 2 } } { p _ { - } } \bar { \epsilon } _ { t } , \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { a + 1 } \right\} \leq \Phi _ { r } ( \bar { \epsilon } _ { t } ) .\tag{59}
$$

By Lemma 13, on $B _ { t } ^ { c } ,$ , every predictor used in a busy period beginning in a round $s \in \{ u { + } 2 , \ldots , t \}$ satisfies $\mathcal { E } ( q _ { s } ) \leq \bar { \epsilon } _ { t }$ . Thus

$$
\begin{array} { r } { \mathcal { C } _ { s } \cap E _ { s } ^ { c } \subseteq S _ { s } \cap E _ { s } ^ { c } \subseteq B _ { t } , \qquad u + 2 \leq s \leq t . } \end{array}
$$

Since all events $C _ { s }$ are disjoint, ${ { C } _ { s } } \cap { { E } _ { s } ^ { c } } \left( u + 2 \le s \le t \right)$ are all disjoint in the event $B _ { t }$ , so $\bigcup _ { s = u + 2 } ^ { t } C _ { s } \cap E _ { s } ^ { c } \subset B _ { t }$ holds. Using this fact and the fact that $Z _ { s , t } = 0$ in the event $\mathcal { S } _ { s } \ : \backslash \ : \mathcal { C } _ { s }$ , we can derive the following inequality.

$$
\begin{array} { r l } { \displaystyle \sum _ { s = u + 2 } ^ { t } \big \vert \mathbb { E } \big ( Z _ { s , t } \mathbf { 1 } \{ \mathcal { S } _ { s } \cap E _ { s } ^ { c } \} \big ) \big \vert = \displaystyle \sum _ { s = u + 2 } ^ { t } \big \vert \mathbb { E } \big ( Z _ { s , t } \mathbf { 1 } \{ \mathcal { C } _ { s } \cap E _ { s } ^ { c } \} \big ) \big \vert \leq \displaystyle \sum _ { s = u + 2 } ^ { t } \mathbb { E } \big ( \big \vert Z _ { s , t } \big \vert \mathbf { 1 } \{ \mathcal { C } _ { s } \cap E _ { s } ^ { c } \} \big ) } & { } \\ { \leq \displaystyle \sum _ { s = u + 2 } ^ { t } \mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ \mathcal { C } _ { s } \cap E _ { s } ^ { c } \} \big ) = \mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ \begin{array} { c } { t } \\ { \bigcup _ { s = u + 2 } } \end{array} \mathcal { C } _ { s } \cap E _ { s } ^ { c } \} \big ) } & { } \\ { \leq \mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ B _ { t } \} \big ) } \end{array}\tag{60}
$$

Moreover, $( r w ) ^ { 2 } \leq e ^ { r w }$ for every $w \geq 0$ , so Lemma 1 gives

$$
\mathbb { E } W _ { t + 1 } ^ { 2 } \le \frac { 1 } { r ^ { 2 } } \mathbb { E } e ^ { r W _ { t + 1 } } \le \frac { K _ { r } } { r ^ { 2 } } .
$$

By using the Cauchy–Schwarz inequality for $W _ { t + 1 }$ and $\mathbf { 1 } \{ B _ { t } \}$ , and the inequality in Lemma 13, we get the following upper bound for Equation (60).

$$
\mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ B _ { t } \} \big ) \le \sqrt { \mathbb { E } W _ { t + 1 } ^ { 2 } \mathbb { P } ( B _ { t } ) } \le \frac { \sqrt { K _ { r } b _ { t } } } { r }
$$

Finally, for $C _ { s }$ with $2 \leq s \leq u + 1$ , by applying Lemma 14 we can get $\begin{array} { r } { \mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ C _ { s } \} \big ) \le \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { h } } \end{array}$ where $h = t - s + 1$ . Summing this inequality for $s = 2 , . . . , u + 1$ , we get the following upper bound.

$$
\begin{array} { r l } { \displaystyle \sum _ { s = 2 } ^ { u + 1 } \left| \mathbb { E } \big ( Z _ { s , t } \mathbf { 1 } \big \{ \mathcal { S } _ { s } \big \} \big ) \right| = \displaystyle \sum _ { s = 2 } ^ { u + 1 } \left| \mathbb { E } \big ( Z _ { s , t } \mathbf { 1 } \big \{ \mathcal { C } _ { s } \big \} \big ) \right| \leq \displaystyle \sum _ { s = 2 } ^ { u + 1 } \mathbb { E } \big ( | Z _ { s , t } | \mathbf { 1 } \big \{ \mathcal { C } _ { s } \big \} \big ) \leq \displaystyle \sum _ { s = 2 } ^ { u + 1 } \mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \big \{ \mathcal { C } _ { s } \big \} \big ) } & { } \\ { \leq \displaystyle \frac { M _ { G } ( r ) } { r } \displaystyle \sum _ { h = t - u } ^ { t - 1 } \psi ( r ) ^ { h } \leq \displaystyle \frac { M _ { G } ( r ) } { r ( 1 - \psi ( r ) ) } \psi ( r ) ^ { t - u } \leq \frac { M _ { G } ( r ) } { r ( 1 - \psi ( r ) ) } \psi ( r ) ^ { u } } & { } \end{array}\tag{61}
$$

Here, we used the fact $t - u \geq u$

Combining the three preceding estimates Equation (59), Equation (60), Equation (61) with equation $5 7$ gives

$$
\begin{array} { r l } & { | J _ { t } ^ { \mathrm { a l g } } - J _ { t } ^ { \mathrm { S E P T } } | = \displaystyle \left| \sum _ { s = 2 } ^ { t } \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ S _ { s } \} \right) \right| } \\ & { \qquad \leq \displaystyle \sum _ { s = 2 } ^ { u + 1 } | \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ S _ { s } \} \right) | + \displaystyle \sum _ { s = u + 2 } ^ { t } | \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ S _ { s } \cap E _ { s } \} \right) | + \displaystyle \sum _ { s = u + 2 } ^ { t } | \mathbb { E } \left( Z _ { s , t } \mathbf { 1 } \{ S _ { s } \cap E _ { s } ^ { c } \} \right) | } \\ & { \qquad \leq \displaystyle \frac { M _ { G } ( r ) } { r ( 1 - \psi ( r ) ) } \psi ( r ) ^ { u } + \Phi _ { r } ( \bar { \epsilon } _ { t } ) + \frac { \sqrt { K _ { r } b _ { t } } } { r } } \end{array}\tag{62}
$$

Therefore we can obtain Equation (12).

For fixed model and trafic parameters,

$$
m _ { t } = \Theta ( \lambda t ) \qquad \mathrm { a n d } \qquad \bar { \epsilon } _ { t } = \widetilde { O } \left( \sqrt { \frac { d } { \lambda t } } \right) .
$$

The series calculation in Appendix H.6 therefore gives the stated rate. If $m _ { t } < 5$ , both policies are work conserving, so Lemma 1 implies

$$
0 \leq J _ { t } ^ { \mathrm { a l g } } , J _ { t } ^ { \mathrm { S E P T } } \leq C _ { W } ( r ) ,
$$

and hence

$$
\mathcal { T } _ { t } ^ { \mathrm { S E P T } } = \vert J _ { t } ^ { \mathrm { a l g } } - J _ { t } ^ { \mathrm { S E P T } } \vert \leq C _ { W } ( r ) .
$$

## H.2 Proof of Lemma 13

Proof. Suppose that a busy period begins in round s. Its first job arrived at the end of round $s - 1$ The predictor used during this busy period was computed after service in round $s - 1$ and before that arrival. Because the queue was empty immediately after service in round $s - 1$ , every job that arrived by the end of round $s - 2$ had already departed, and $( X _ { i } , Y _ { i } )$ had been recorded for each such job. The predictor is therefore fitted to a complete prefix of the i.i.d. job-indexed sequence and is fixed before the context of the initial job is drawn at the end of round $s - 1$

Suppose that $u + 2 \leq s \leq t$ . Since $u \leq s - 2$ , the data used by the predictor contain all $\begin{array} { r } { M _ { u } = \sum _ { j = 1 } ^ { u } A _ { j } } \end{array}$ jobs arriving through round u. Since $m _ { t } = \lfloor \lambda u / 2 \rfloor$ and $\mathbb { E } M _ { u } = \lambda u$ , a multiplicative Chernof bound with relative deviation $1 / 2$ gives

$$
\mathbb { P } ( M _ { u } < m _ { t } ) \le \mathbb { P } \left( M _ { u } < \frac { \lambda u } { 2 } \right) \le e ^ { - \lambda u / 8 } .
$$

Let $\Omega _ { t }$ be the simultaneous prediction event supplied by Corollary 1 with failure probability $t ^ { - 4 }$ Define $B _ { t } = \{ M _ { u } < m _ { t } \} \cup \Omega _ { t } ^ { c }$ . Then

$$
\mathbb { P } ( B _ { t } ) \le \mathbb { P } ( \{ M _ { u } < m _ { t } \} ) + \mathbb { P } ( \Omega _ { t } ^ { c } ) \le e ^ { - \lambda u / 8 } + \frac { 1 } { t ^ { 4 } } = b _ { t } .
$$

On $B _ { t } ^ { c } .$ consider a predictor used in a busy period beginning in a round $s \in \{ u + 2 , \ldots , t \}$ . If N denotes the number of observations used to construct this predictor, then $m _ { t } \le N \le s - 2 < t$ holds, and Corollary 1 gives $\begin{array} { r } { \mathcal { E } ( \widehat { p } _ { N } ) \leq \epsilon _ { N } \left( \frac { 6 } { \pi ^ { 2 } N ^ { 2 } t ^ { 4 } } \right) } \end{array}$ . Using $m _ { t } \le N \le t$ , we obtain

$$
\begin{array} { r l } & { \mathcal { E } ( \widehat { p } _ { N } ) \leq \operatorname* { m i n } \left( 1 , \sqrt { \frac { 8 e ^ { \bar { S } } \left( d \log ( 3 2 \bar { S } N ) + \log ( \pi ^ { 2 } N ^ { 2 } t ^ { 4 } / 6 ) \right) + 1 } { 2 N } } \right) } \\ & { \qquad \leq \operatorname* { m i n } \left( 1 , \sqrt { \frac { 8 e ^ { \bar { S } } \left( d \log ( 3 2 \bar { S } t ) + \log ( \pi ^ { 2 } t ^ { 6 } / 6 ) \right) + 1 } { 2 m _ { t } } } \right) } \\ & { \qquad = \bar { \epsilon } _ { t } . } \end{array}
$$

This proves the claim.

## H.3 Proof of Lemma 14

Proof. Couple the two systems so that their arrival indicators, arriving contexts, and job-specific service times agree. Since both policies are work conserving, their workload processes coincide. Consequently, their busy periods begin and end in the same rounds.

Let G be the service time of the job initiating the busy period in round $s ,$ and let $\widetilde { W } _ { h }$ be the stopped workload associated with this busy period after h service rounds. On $C _ { s } , W _ { t + 1 } = \widetilde { W } _ { h }$ and $\tau _ { \mathrm { b p } } > h$ hold. Therefore,

$$
\mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ C _ { s } \} \big ) = \mathbb { P } ( S _ { s } ) \mathbb { E } \left( \widetilde { W } _ { h } \mathbf { 1 } \{ \tau _ { \mathrm { b p } } > h \} \middle | S _ { s } \right) .
$$

Conditional on $ { \boldsymbol { S } } _ { s }$ and $G ,$ , the future workload increments have the same distribution as those in the stopped busy-period process. Using $w \leq e ^ { r w } / r$ and Equation (14), we obtain

$$
\mathbb { E } \left( \widetilde { W } _ { h } \mathbf { 1 } \big \{ \tau _ { \mathrm { b p } } > h \big \} \Big | \mathcal { S } _ { s } , G \right) \leq \frac { 1 } { r } \mathbb { E } \left( e ^ { r \widetilde { W } _ { h } } \mathbf { 1 } \big \{ \tau _ { \mathrm { b p } } > h \big \} \Big | \mathcal { S } _ { s } , G \right) \leq \frac { e ^ { r G } } { r } \psi ( r ) ^ { h } .
$$

The event $ { \boldsymbol { S } } _ { s }$ is determined before the context of the initial job is observed. Hence the conditional distribution of G given $ { \boldsymbol { S } } _ { s }$ is the usual mixture $G \mid X \sim \operatorname { G e o m } ( p ( X ) )$ , and

$$
\mathbb { E } \big ( e ^ { r G } \mid \mathcal { S } _ { s } \big ) = \mathbb { E } _ { X } \left[ \frac { p ( X ) e ^ { r } } { 1 - ( 1 - p ( X ) ) e ^ { r } } \right] = M _ { G } ( r ) .
$$

It follows that

$$
\begin{array} { r l r } {  { \mathbb { E } \big ( W _ { t + 1 } \mathbf { 1 } \{ C _ { s } \} \big ) \le \mathbb { P } ( S _ { s } ) \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { h } } } \\ & { } & { \displaystyle \le \frac { M _ { G } ( r ) } { r } \psi ( r ) ^ { h } . } \end{array}
$$

Finally, on $C _ { s }$ , both terminal queue lengths lie in $[ 0 , W _ { t + 1 } ]$ . Therefore,

$$
| Z _ { s , t } | = | Q _ { t + 1 } ^ { ( 1 ) } - Q _ { t + 1 } ^ { ( 2 ) } | \leq W _ { t + 1 } .
$$

This proves Equation (54).

## H.4 Sensitivity of SEPT to departure probabilities

Lemma 16. Fix N jobs, their arrival rounds, a horizon, and a deterministic tie-breaking rule. For $v \in [ p _ { - } , p _ { + } ] ^ { N }$ , let $\mathcal { V } ( v )$ be the expected terminal queue length when job i has service time Geom(v<sub>i</sub>) and the jobs are served in decreasing order of $v _ { i }$ . Then

$$
| \mathcal { V } ( v ) - \mathcal { V } ( w ) | \leq \frac { N } { p _ { - } } \| v - w \| _ { 1 } .\tag{63}
$$

The same conclusion holds if the process is stopped when the queue first becomes empty.

Proof. First suppose that the coordinates of v and w induce the same strict order. The two systems then use the same priority rule. Couple their service times coordinatewise as in Lemma 8. A union bound gives

$$
\mathbb { P } ( \mathrm { s o m e ~ p a i r ~ o f ~ s e r v i c e ~ t i m e s ~ d i f f e r s } ) \le \frac { 1 } { p _ { - } } \sum _ { i = 1 } ^ { N } | v _ { i } - w _ { i } | .
$$

If every pair of service times agrees, the two schedules and their terminal queues agree. Since each terminal queue contains at most N jobs, multiplying the disagreement probability by N proves Equation (63) whenever v and w induce the same strict order.

Next, suppose that $v _ { i } = v _ { j }$ . Jobs i and j then have identical service-time distributions. If at most one of them is waiting, their relative priority is irrelevant. If both are waiting, exchanging their labels together with their i.i.d. geometric service times leaves the law of the queue unchanged. Repeating this argument after the first departure, or equivalently proceeding by induction on the number of remaining rounds, shows that either relative order gives the same expected terminal queue length. Thus V is continuous when two coordinates cross.

For vectors v and w with distinct coordinates, divide the line segment joining them at the finitely many points at which two coordinates become equal. On each resulting subsegment, the coordinate order is fixed, so the preceding coupling argument applies. Summing the resulting bounds proves Equation (63), since the sum of the $\ell _ { 1 } { \mathrm { - l e n g t h s } }$ of the subsegments equals $\lVert v - w \rVert _ { 1 }$ . General v, w follow by continuity. Stopping the process when the queue first becomes empty does not afect the label-exchange argument, so the same proof applies to the stopped process. □

## H.5 Proof of Lemma 15

Proof. Condition on the context of the initial job and on all subsequent arrival indicators and contexts in the h-round comparison. Stop the process as soon as service leaves no unfinished job. Let N be the number of jobs described by these variables. Then $N \leq h + 1$ , and these variables are independent of $q ,$ which was fixed before the context of the initial job was drawn. Write

$$
p _ { i } = p ( X _ { i } ) , q _ { i } = q ( X _ { i } ) .
$$

Let $J _ { p } ( q { \mathrm { - } } \mathrm { S E P T } )$ be the conditional expected terminal queue length when job i has departure probability $p _ { i }$ and priority $q _ { i }$ . Compare this system with the intermediate system in which job i has both departure probability and priority $q _ { i }$ . We have

$$
J _ { p } ( q \mathrm { - S E P T } ) - J _ { p } ( p \mathrm { - S E P T } ) = \left( J _ { p } ( q \mathrm { - S E P T } ) - J _ { q } ( q \mathrm { - S E P T } ) \right) + \left( \mathcal { V } ( q ) - \mathcal { V } ( p ) \right) .\tag{64}
$$

The priority order is fixed in the first diference. Coordinatewise geometric coupling therefore bounds its absolute value by

$$
\frac { N } { p _ { - } } \sum _ { i = 1 } ^ { N } | p _ { i } - q _ { i } | .
$$

The second diference is bounded by the same quantity using Lemma 16. Hence

$$
| J _ { p } ( q \mathrm { - S E P T } ) - J _ { p } ( p \mathrm { - S E P T } ) | \leq \frac { 2 N } { p - } \sum _ { i = 1 } ^ { N } | p _ { i } - q _ { i } | .
$$

Averaging over the arrival variables and contexts gives

$$
\mathbb { E } \sum _ { i = 1 } ^ { N } \left| p _ { i } - q _ { i } \right| \leq ( h + 1 ) \mathcal { E } ( q ) .
$$

Since $N \leq h + 1$ , we can show the desired result.

## H.6 Asymptotic dependence of $\Phi _ { r } ( \epsilon )$ on ϵ

For $\epsilon \in ( 0 , 1 ]$ , let

$$
L _ { \epsilon } = \left\lceil \frac { \log ( 1 / \epsilon ) } { \log ( 1 / \psi ( r ) ) } \right\rceil .
$$

By definition, $\psi ( r ) ^ { L _ { \epsilon } } \leq \epsilon$ . Bounding the terms indexed by $a \in \{ 0 , \ldots , L _ { \epsilon } \}$ by their statistical bound and the remaining terms by their busy-period tail bound gives

$$
\begin{array} { l } { \displaystyle \Phi _ { r } ( \epsilon ) \le \frac { 2 \epsilon } { p _ { - } } \sum _ { a = 0 } ^ { L _ { \epsilon } } ( a + 2 ) ^ { 2 } + \frac { M _ { G } ( r ) } { r } \sum _ { a = L _ { \epsilon } + 1 } ^ { \infty } \psi ( r ) ^ { a + 1 } } \\ { \displaystyle < \frac { 2 \epsilon ( L _ { \epsilon } + 2 ) ( L _ { \epsilon } + 3 ) ( 2 L _ { \epsilon } + 5 ) } { 6 p _ { - } } + \frac { M _ { G } ( r ) } { r ( 1 - \psi ( r ) ) } \psi ( r ) ^ { L _ { \epsilon } + 2 } } \\ { \displaystyle \le \frac { 2 \epsilon ( L _ { \epsilon } + 3 ) ^ { 3 } } { 3 p _ { - } } + \frac { M _ { G } ( r ) } { r ( 1 - \psi ( r ) ) } \epsilon . } \end{array}
$$

Since

$$
L _ { \epsilon } \le 1 + \frac { \log ( 1 / \epsilon ) } { \log ( 1 / \psi ( r ) ) } ,
$$

we conclude that

$$
\Phi _ { r } ( \epsilon ) = O \left( \epsilon ( 1 + \log ^ { 3 } ( 1 / \epsilon ) ) \right) ,
$$

with the dependence stated after Equation (56).

## H.7 Average tracking and a uniformly random horizon

From Theorem 4 and $\textstyle \sum _ { t = 1 } ^ { T } t ^ { - 1 / 2 } \leq 2 { \sqrt { T } }$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { T } _ { t } ^ { \mathrm { S E P T } } = \widetilde { O } \left( \sqrt { \frac { d } { \lambda T } } \right) .
$$

If τ is independent of the queueing process and uniformly distributed on [T], then conditioning on τ gives

$$
\mathbb { E } Q _ { \tau + 1 } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } Q _ { t + 1 } .
$$

Thus, the same rate compares the learner and SEPT when the evaluation horizon is chosen independently and uniformly from [T].

## Use of Generative-AI Tools

The authors formulated the research problem and used GPT-5.6 Sol to explore possible solutions alongside their own investigation. The model proposed an alternative approach and generated initial theoretical results and proof drafts. The authors checked and revised these drafts, correcting errors and filling gaps in the arguments. In particular, the authors corrected and refined the proofs of the lower bound in Theorem 2 and the impossibility of an anytime policy with vanishing regret under an unknown horizon in Theorem 3. GPT-5.6 Sol was also used for language editing and manuscript polishing. The authors take responsibility for the correctness and presentation of the final work.