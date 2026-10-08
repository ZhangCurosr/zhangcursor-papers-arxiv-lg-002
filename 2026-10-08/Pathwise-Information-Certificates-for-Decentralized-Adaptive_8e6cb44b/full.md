# Pathwise Information Certificates for Decentralized Adaptive Sensing

Theodoros Tsiligkaridis MIT Lincoln Laboratory

## Abstract

We study decentralized adaptive sensing, where multiple agents choose measurements from evolving local beliefs while exchanging information over a communication graph. We ask whether the measurements actually selected by an adaptive policy have collected enough evidence to distinguish the true target from every plausible alternative. We develop a pathwise certificate based on the R´enyi–Chernof information accumulated along the realized sensing trajectory. It yields nonasymptotic MAP-error bounds and an anytime, network-wide stopping rule for arbitrary history-dependent sensing policies, while separating accumulated statistical information from a bounded network-mixing transient. Linear growth of the information against the least-resolved competitor implies exponential decay of MAP and squared-localization error. A classical pairwise KL converse, specialized to the adaptive decentralized transcript, shows that insuficient information on any pair prevents a positive uniform error exponent, confirming the hardest competitor as a fundamental bottleneck. Across policies, graph topologies, sensor profiles, and seeds, the worst-competitor score correlates more strongly with localization speed than an average-pair proxy in both 1D (r = 0.89 versus 0.40) and structured 2D sensing (r = 0.77 versus 0.48). Our results provide a practical way to certify and diagnose adaptive multiagent sensing systems using the evidence they actually collect.

## 1 INTRODUCTION

Adaptive information acquisition arises when sensing is itself a decision, from active perception and environmental monitoring to sequential diagnosis (Bajcsy et al., 2018; Singh et al., 2009; Gorry and Barnett, 1968). In a decentralized system, performance depends on both which evidence is acquired and how that evidence propagates through the network. The key dificulty arises when sensing decisions depend on evolving local beliefs: the observation kernel changes with the realized history, so a useful reliability guarantee should remain valid for data-dependent policies and stopping times.

Existing theories address important pieces of this problem separately. Distributed social-learning analyses typically assume fixed local observation laws and quantify how statistical information and graph topology drive learning (Nedi´c et al., 2017; Lalitha et al., 2018; Shahrampour et al., 2016; Wu and Uribe, 2025). Controlled sensing and active testing analyze designed action policies, action frequencies, error exponents, or sequential tests (Chernof, 1959; Nitinawarat et al., 2013; Lalitha and Javidi, 2017; Rangi et al., 2021; Hsu and Wang, 2025). Classical testing already identifies hard alternatives as rate bottlenecks; our focus is different: given the trajectory produced by an adaptive decentralized policy, we ask whether the actions it actually selected accumulated enough evidence against every alternative to certify reliable localization.

Our certificate accumulates pairwise Chernof information from the queries actually selected. A measurement contributes only to hypothesis pairs that it separates, with the size of that contribution determined jointly by the sensor model and stationary network. The least accumulated evidence over competing hypotheses controls the finite-time and anytime guarantees, while network disagreement appears as a bounded transient (Figure 1).

The construction is not tied to a particular sensing policy or target geometry. It evaluates the information induced by the realized actions of an arbitrary historydependent policy, so it can be applied post hoc to heuristic, posterior-driven, or learned sensing policies without changing how those policies are trained. We use binary queries because they make pairwise distinguishability especially transparent, and illustrate the theory with ordered 1D thresholds and a structurally diferent 2D half-plane library. Appendix D extends the same construction to bounded action-dependent observation kernels beyond the binary symmetric channel model.

![](images/6ea6c7b163cdc2230bfd055cd8f2db2d758c6b6d17ed1c689cea61d4e1740d34.jpg)  
Figure 1: Decentralized adaptive sensing and the pathwise certificate. Agents select historydependent queries from evolving local beliefs, observe noisy responses, and pool beliefs over a communication graph. Each separating query contributes pairwise Chernof information $\mathsf { J } _ { j } ( s v _ { j } )$ , which depends on the sensor model and the agent’s stationary network influence. Accumulating this evidence along the realized trajectory yields the information process $\mathcal { T } _ { t } ( m ; s )$ , which drives finite-time and anytime reliability guarantees against the hardest competitor.

## Contributions.

1. Pathwise finite-time and anytime reliability. We accumulate pairwise Chernof information along the realized adaptive trajectory, with each separating measurement contributing an amount that depends on its sensor model and stationary graph influence. This yields node-wise finite-horizon MAP bounds and a network-wide anytime certificate for arbitrary history-dependent policies, with network disagreement appearing only as a bounded transient.

2. Hardest-competitor rates with a classical converse. Linear growth of the least-resolved competitor’s realized information yields exponential MAP and squared-localization-loss rates. The hard-pair principle and two-point lower-bound machinery are classical; our contribution is to connect them to the same realized adaptive decentralized transcript used by the certificate, and to show why average pair coverage can hide an unresolved alternative.

3. Policy-agnostic certification with optional guaranteed coverage. The results apply to randomized, heuristic, posterior-driven, or learned policies. A certified-exploration wrapper separately guarantees worst-pair coverage while leaving all other rounds to the base policy.

4. Empirical tests in 1D and 2D. Across policies, graphs, sensors, and seeds, we test exponential localization, graph efects, and the hardest-competitor hypothesis in ordered 1D and structured 2D. In both, worst-competitor information is more strongly associated with localization speed than an average-pair proxy.

## 2 RELATED WORK

Adaptive querying and controlled sensing. Noisy twenty questions and probabilistic bisection adapt queries to reduce uncertainty (Horstein, 1963; Jedynak et al., 2012; Waeber et al., 2013; Chung et al., 2018; Zhou and Hero, 2021); decentralized variants combine active queries with belief sharing (Tsiligkaridis et al., 2014, 2015; Casta˜n´on et al., 2017; Tsiligkaridis and Tsiligkaridis, 2017; Tsiligkaridis, 2018). Controlled sensing instead studies policy design and error exponents (Chernof, 1959; Nitinawarat et al., 2013; Hsu and Wang, 2025). We certify the evidence collected by a realized adaptive trajectory rather than optimize a policy class.

Decentralized learning and active testing over networks. Log-linear social learning gives consistency and finite-time or topology-dependent guarantees under fixed observation models (Jadbabaie et al., 2012; Rahimian and Jadbabaie, 2015; Nedi´c et al., 2017; Lalitha et al., 2018; Shahrampour et al., 2016; Wu and Uribe, 2025). Active network testing permits selected sensing actions (Lalitha and Javidi, 2017; Rangi et al., 2021), including learned policies (Szostak and Cohen, 2024). Prior combinations of decentralization and active sensing analyze specific strategies or asymptotic characteristics; ours conditions on the complete realized action trajectory and is nonasymptotic and anytimevalid (Table 1).

To our knowledge, prior work does not jointly provide finite-time and anytime-valid guarantees that certify whether the measurements actually selected by an arbitrary decentralized, history-dependent sensing policy have accumulated suficient evidence for reliable localization.

## 3 DECENTRALIZED ADAPTIVE QUERYING

Let $\boldsymbol { \Theta } = \{ \theta _ { 1 } , \ldots , \theta _ { M } \}$ be a finite hypothesis set and let $m ^ { \star } \in [ M ]$ denote the unknown target. For localization, $\theta _ { m }$ may lie in any metric space, including $\mathbb { R } ^ { d } \colon$ the theory uses only the induced hypothesis-indexed observation laws. Agent j selects $q _ { j , t } : [ M ] \to \{ 0 , 1 \}$ from its local history $\mathcal { F } _ { j , t } ^ { \mathrm { l o c } }$ . For analysis, $\mathcal { F } _ { t }$ denotes the global filtration after the round-t queries (and their policy randomization) are selected but before the next observations, with $\mathcal { F } _ { j , t } ^ { \mathrm { l o c } } \subseteq \mathcal { F } _ { t }$ . Then

Table 1: Positioning of our work. The novelty of our guarantee is the conjunction of decentralization and adaptive sensing with a finite-time, anytime-valid certificate for the realized trajectory of an otherwise arbitrary policy.
<table><tr><td>Work</td><td></td><td></td><td>Decentralized Adaptive sensing Arbitrary-policy cert. Realized-path cert. Finite-time Anytime stop</td><td></td><td></td><td></td></tr><tr><td>Nedić et al. (2017)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Nitinawarat et al. (2013)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tsiligkaridis et al. (2015)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lalitha and Javidi (2017)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Rangi et al. (2021)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>This work</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

$$
\begin{array} { r l } & { Y _ { j , t + 1 } = q _ { j , t } ( m ^ { \star } ) \oplus W _ { j , t + 1 } , } \\ & { W _ { j , t + 1 } \sim \mathrm { B e r n o u l l i } ( \epsilon _ { j } ) , \qquad 0 < \epsilon _ { j } < \frac { 1 } { 2 } , } \end{array}\tag{1}
$$

and conditional on $\mathcal { F } _ { t }$ the channel noises are independent across agents and time. The local likelihood used in the Bayes update is

$$
\ell _ { j , t } ( y \mid m ) = \left\{ { 1 - \epsilon _ { j } , } \atop { \epsilon _ { j } , } \right. \ y \not = q _ { j , t } ( m ) ,  \qquad y \in \{ 0 , 1 \} .\tag{2}
$$

Let $\mu _ { i , t } ( m )$ denote agent i’s posterior belief in hypothesis m at round t. Agents start from a common fullsupport prior $\mu _ { i , 0 } = \pi _ { 0 }$ . Let $A \in \mathbb { R } ^ { N \times N }$ be primitive (irreducible and aperiodic) and row stochastic, with stationary left eigenvector $\pmb { v } ^ { \top } \pmb { A } = \pmb { v } ^ { \top } , \pmb { v } > 0 , \pmb { v } ^ { \top } \pmb { 1 } = 1$ Each round consists of a local Bayes update followed by log-linear pooling:

$$
\begin{array} { r l } & { \widetilde { \mu } _ { j , t + 1 } ( m ) = \frac { \mu _ { j , t } ( m ) \ell _ { j , t } ( Y _ { j , t + 1 } \mid m ) } { \sum _ { u } \mu _ { j , t } ( u ) \ell _ { j , t } ( Y _ { j , t + 1 } \mid u ) } , } \\ & { \mu _ { i , t + 1 } ( m ) = \frac { \prod _ { j } \widetilde { \mu } _ { j , t + 1 } ( m ) ^ { A _ { i j } } } { \sum _ { u } \prod _ { j } \widetilde { \mu } _ { j , t + 1 } ( u ) ^ { A _ { i j } } } . } \end{array}\tag{3}
$$

Thus each round has a simple “local evidence, then consensus” structure: each agent first performs a Bayesian update and then averages log-beliefs through the communication network. This geometric pooling rule is standard in distributed non-Bayesian/social learning (Jadbabaie et al., 2012; Rahimian and Jadbabaie, 2015; Nedi´c et al., 2017; Lalitha et al., 2018). Agent i predicts

$$
\widehat { m } _ { i , t } \in \mathop { \mathrm { a r g } } _ { m } \operatorname* { m a x } _ { \mu _ { i , t } ( m ) . }
$$

Query policies. In the experiments, we evaluate posterior bisection, Thompson-style sampling, random querying, and exploration+bisection in 1D and 2D. Exact policy definitions are given in Appendix G. The theory does not assume any of these policies.

Complexity. The Bayes update costs $O ( M )$ per agent and pooling costs $O ( M | E | )$ network-wide per round. Ordered-threshold selection is $O ( M ) ;$ ; naive scoring of a finite library Q is $O ( M | \mathcal { Q } | )$ per agent. Explicit anytime tracking additionally stores $O ( M ^ { 2 } )$ pair scores and costs $O ( N M ^ { 2 } )$ per round naively; structured query families can reduce this overhead. We implement belief updates in log space.

## 4 RELIABILITY CERTIFICATES FROM REALIZED INFORMATION

Notation and accumulated evidence. Fix a true target $m ^ { \star }$ and a false target m $\not = m ^ { \star }$ . Two things determine whether a measurement is useful for a pair of hypotheses: whether the query separates that pair, and how informative the corresponding sensor observation is. A query contributes evidence for this pair only when the two targets predict diferent noiseless answers, so define

$$
I _ { j , t } ( m , m ^ { \star } ) : = \mathbf { 1 } \{ q _ { j , t } ( m ) \neq q _ { j , t } ( m ^ { \star } ) \} .
$$

For a separating query, agent j must distinguish the two binary symmetric channel (BSC) response laws with crossover probability $\epsilon _ { j }$ . For $\alpha \in [ 0 , 1 ]$ , define its pairwise information contribution

$$
\mathsf { J } _ { j } ( \alpha ) : = - \log \left[ ( 1 - \epsilon _ { j } ) ^ { 1 - \alpha } \epsilon _ { j } ^ { \alpha } + \epsilon _ { j } ^ { 1 - \alpha } ( 1 - \epsilon _ { j } ) ^ { \alpha } \right] .\tag{4}
$$

It is the negative log Chernof coeficient (equivalently, a scaled R´enyi divergence) between the two BSC output laws (Chernof, 1952; R´enyi, 1961; van Erven and Harremo¨es, 2014); larger values mean that one separating observation is more informative. Let

$$
\begin{array} { c } { \displaystyle \mathcal { S } : = \{ s > 0 : s v _ { j } \leq 1 \forall j \} , } \\ { \displaystyle \mathcal { T } _ { t } ( m ; s ) : = \sum _ { \tau < t } \sum _ { j = 1 } ^ { N } \mathsf { J } _ { j } ( s v _ { j } ) I _ { j , \tau } ( m , m ^ { \star } ) . } \end{array}\tag{5}
$$

The set $\boldsymbol { S } = ( 0 , 1 / \operatorname* { m a x } _ { j } v _ { j } ]$ is always nonempty because $s = 1$ is admissible. Relative to a balanced graph $( v _ { j } = 1 / N )$ , strong stationary imbalance narrows this interval and can prevent the terms $s v _ { j }$ from being simultaneously near their most informative values. Appendix C quantifies the resulting certified-information loss. $\mathcal { T } _ { t } ( m ; s )$ is the evidence accumulated against competitor m along the realized queries, and s is an analysis parameter rather than part of the sensing policy. For the anytime guarantee s is fixed in advance; only the post-hoc diagnostic optimizes it after a rollout. For the finite-time network correction, define

$$
b _ { j } : = \log \frac { 1 - \epsilon _ { j } } { \epsilon _ { j } } , \qquad \Delta _ { i } : = \sum _ { k \geq 1 } \sum _ { j = 1 } ^ { N } | ( A ^ { k } ) _ { i j } - v _ { j } | b _ { j } ,\tag{6}
$$

and the prior factor $K _ { \pi _ { 0 } } ( m ^ { \star } , s ) \qquad : =$ $\smash { \sum _ { m \neq m ^ { \star } } ( \pi _ { 0 } ( m ) / \pi _ { 0 } ( m ^ { \star } ) ) ^ { s } }$ For the network-wide anytime result below, we will use the corresponding worst-case quantities

$$
\Delta _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i } \Delta _ { i } , \qquad \mathcal { K } _ { \operatorname* { m a x } } ( s ) : = \operatorname* { m a x } _ { m ^ { \prime } } \mathcal { K } _ { \pi _ { 0 } } ( m ^ { \prime } , s ) .
$$

For a uniform prior, $\begin{array} { r } { K _ { \pi _ { 0 } } ( m ^ { \star } , s ) = K _ { \operatorname* { m a x } } ( s ) = M - 1 } \end{array}$ Primitivity of A implies $\Delta _ { i } < \infty$ because $A ^ { k }$ converges geometrically to $\bar { 1 } \bar { v } ^ { \top }$ (Seneta, 2006; Levin and Peres, 2017).

Ofline-computable certificate constants. The admissible set $s ,$ stationary weights $_ { v , }$ information functions $\mathsf { J } _ { j }$ , mixing corrections $\Delta _ { i }$ and $\Delta _ { \mathrm { m a x } } .$ , and prior factors $\kappa _ { \pi _ { 0 } }$ and $\boldsymbol { \kappa } _ { \mathrm { m a x } }$ depend only on the known prior, communication graph, and sensor models and can therefore be precomputed before deployment. During operation, the anytime certificate only needs to update the pairwise information accumulated by the realized queries.

## 4.1 Finite-time reliability

For any chosen evidence threshold $a _ { m , t }$ , the error probability is bounded by the probability that the policy has not yet accumulated that evidence plus an exponentially small residual due to noise and network disagreement.

Theorem 1 (Finite-time adaptive-query localization). Fix agent $i ,$ true target $m ^ { \star } , s \in { \cal S }$ , and deterministic levels $a _ { m , t } \geq 0$ . For any history-dependent query policy,

$$
\begin{array} { r l } & { \mathbb { P } _ { m ^ { \star } } \bigl ( \widehat { m } _ { i , t } \neq m ^ { \star } \bigr ) \leq \displaystyle \sum _ { m \neq m ^ { \star } } \Bigg [ \mathbb { P } _ { m ^ { \star } } \bigl ( \mathcal { T } _ { t } ( m ; s ) < a _ { m , t } \bigr ) } \\ & { \qquad + \left( \displaystyle \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } \right) ^ { s } e ^ { s \Delta _ { i } - a _ { m , t } } \Bigg ] . } \end{array}\tag{7}
$$

The first term measures whether the policy has collected enough evidence against $m ;$ the second is the residual efect of observation noise and network disagreement. Thus the bound separates information acquisition from decentralized communication transients.

Proof idea. The full proof is in Appendix A. Projecting posterior log-odds onto v removes the repeated geometric-pooling dynamics. Conditional on the history, the selected queries are fixed, and the fractional moment of each likelihood ratio is exactly $\exp [ - \mathrm { J } _ { j } ( s v _ { j } ) ]$ when the query separates the pair and 1 otherwise. Multiplying the projected odds by $\exp [ { \mathcal { I } } _ { t } ( m ; s ) ]$ therefore cancels this conditional drift and produces a nonnegative martingale. The bounded diference between projected and node-wise log-odds contributes the factor $e ^ { s \Delta _ { i } }$

For geometrically mixing $A , \Delta _ { i }$ is finite; Appendix A.7 gives explicit geometric and spectral bounds. Thus poorer mixing changes the finite-time ofset rather than which competitors are informative.

## 4.2 Anytime network-wide reliability

Theorem 1 is truth-indexed and therefore most natural for fixed-horizon analysis. A truth-independent stopping certificate cannot depend on the unknown target, so we replace truth-versus-competitor quantities by the minimum evidence over all candidate pairs. For distinct targets $m , m ^ { \prime }$ , write $I _ { j , t } ( m , m ^ { \prime } ) : = \mathbf { 1 } \{ q _ { j , t } ( m ) \neq $ $q _ { j , t } ( m ^ { \prime } ) \big \}$ and define the counterparts

$$
\begin{array} { l } { { \displaystyle { \mathcal { T } } _ { t } ( m , m ^ { \prime } ; s ) : = \sum _ { \tau < t } \sum _ { j } \mathsf { J } _ { j } ( s v _ { j } ) I _ { j , \tau } ( m , m ^ { \prime } ) } , } \\ { { \displaystyle { \mathcal { T } } _ { t } ( s ) : = \operatorname* { m i n } _ { m \neq m ^ { \prime } } { \mathcal { T } } _ { t } ( m , m ^ { \prime } ; s ) } . } \end{array}\tag{8}
$$

Theorem 2 (Anytime network-wide certificate). For fixed $s \in S$ and $\delta \in ( 0 , 1 )$ , define

$$
\begin{array} { l } { { c _ { \delta } ( s ) : = s \Delta _ { \operatorname* { m a x } } + \log \frac { N { / K } _ { { \operatorname* { m a x } } } ( s ) } { \delta } , } } \\ { { \tau _ { \delta } ( s ) : = \operatorname* { i n f } \{ t : \underline { { { \mathcal { T } } } } _ { t } ( s ) \geq c _ { \delta } ( s ) \} . } } \end{array}\tag{9}
$$

Then, for every true target $m ^ { \star }$

$$
\mathbb { P } _ { m ^ { \star } } ( \exists i , \ \exists t \geq \tau _ { \delta } ( s ) : \widehat { m } _ { i , t } \neq m ^ { \star } ) \leq \delta .
$$

The rule stops only after every target pair has enough evidence, so validity does not depend on knowing the target or fixing a horizon. Computing $\underline { { \mathcal { T } } } _ { t } ( s )$ requires the network-wide selected-query log: Eq. (3) alone does not make it locally available at each node. Deployment therefore assumes either a centralized logger/monitor or an auxiliary dissemination/aggregation protocol for query identities; the theorem is agnostic to that protocol. If $\underline { { \mathcal { T } } } _ { t } ( s ) \geq \gamma t - B$ , then

$$
\tau _ { \delta } ( s ) \leq \left\lceil \frac { B + s \Delta _ { \operatorname* { m a x } } + \log ( N \mathcal { K } _ { \operatorname* { m a x } } ( s ) / \delta ) } { \gamma } \right\rceil .
$$

The fixed s is a one-dimensional ofline design choice: for certified exploration one may maximize $W ( s )$ in

Eq. (17); more generally, if a lower bound $\underline { { \mathcal { T } } } _ { t } ( s ) \geq $ $\gamma ( s ) t - B ( s )$ is available, one may minimize the displayed stopping-time bound over $s \in S$ before monitoring begins. Data-dependent online tuning of s would require a separate time-uniform argument.

Three uses. Theorem 1 gives fixed-horizon bounds, Theorem 2 gives a truth-independent stopping condition, and Eq. (18) gives a truth-aware post-hoc policy diagnostic. Only the last optimizes s after the rollout.

## 4.3 Exponential rates and why the hardest competitor matters

Define the rate $\begin{array} { r } { E _ { i } : = \operatorname* { l i m } \operatorname* { i n f } _ { t  \infty } - t ^ { - 1 } \log \mathbb { P } _ { m ^ { \star } } ( \widehat { m } _ { i , t } \neq } \end{array}$ $m ^ { \star } )$

Corollary 1 (Stochastic information growth implies exponential learning). If for some fixed $s \in \mathcal { S } , \gamma , \kappa > 0$ finite B, and constants $C _ { m }$

$$
\mathbb { P } _ { m ^ { \star } } ( \mathcal { T } _ { t } ( m ; s ) < \gamma t - B ) \leq C _ { m } e ^ { - \kappa t }
$$

for every false m and all suficiently large t, then $E _ { i } \ge \mathrm { m i n } \{ \gamma , \kappa \}$ . If the information lower bound holds deterministically, $E _ { i } \geq \gamma$

Here γ is the information-growth rate and κ controls its lower-tail reliability; the slower mechanism limits the guaranteed exponent.

Corollary 2 (MAP and squared localization loss). Equip the finite target set with a metric d and assume distinct targets, so $\begin{array} { r } { d _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { m \neq m ^ { \prime } } d ( \theta _ { m } , \theta _ { m ^ { \prime } } ) > 0 } \end{array}$ With $D _ { \Theta } : = \operatorname* { m a x } _ { m , m ^ { \prime } } d ( \theta _ { m } , \theta _ { m ^ { \prime } } )$ 2

$$
\begin{array} { r l } & { d _ { \operatorname* { m i n } } ^ { 2 } \mathbb { P } _ { m ^ { \star } } ( \widehat { m } _ { i , t } \neq m ^ { \star } ) \leq \mathbb { E } _ { m ^ { \star } } [ d ( \widehat { \theta } _ { i , t } , \theta _ { m ^ { \star } } ) ^ { 2 } ] } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } \leq D _ { \Theta } ^ { 2 } \mathbb { P } _ { m ^ { \star } } ( \widehat { m } _ { i , t } \neq m ^ { \star } ) . } \end{array}\tag{10}
$$

Hence MAP error and squared metric localization loss have the same exponent.

The hard-pair/minimax principle is classical (Chernof, 1952, 1959; Nitinawarat et al., 2013; Bretagnolle and Huber, 1979; Tsybakov, 2009). Here the unresolved pair is identified by the realized decentralized action trajectory. The next results show that an average can hide such a pair and that pairwise distinguishability is necessary for a uniform exponent.

Proposition 1 (Average pair coverage can hide localization failure). For any $M \geq 4$ ordered targets with thresholds $q _ { k } ( m ) = 1 \{ m \leq k \}$ , consider synchronized threshold schedules in which all agents use the same threshold on a given round. Let $k _ { \star } = \lfloor M / 2 \rfloor$ . Repeating only q<sub>k</sub> separates

$$
\phi _ { \mathrm { o n e } } ( M ) : = \frac { k _ { \star } ( M - k _ { \star } ) } { \binom { M } { 2 } }\tag{11}
$$

of all unordered pairs on every round, yet at least two hypotheses on the same side are never distinguished, so for every horizon T,

$$
\operatorname* { s u p } _ { m ^ { \star } } \mathbb { P } _ { m ^ { \star } } ( \widehat { m } _ { T } \neq m ^ { \star } ) \geq \frac { 1 } { 2 } .
$$

By contrast, a balanced cycle through all $M - 1$ thresholds has worst-pair separation frequency $1 / ( M - 1 )$ and average pair-separation fraction

$$
\phi _ { \mathrm { c y c } } ( M ) = \frac { 1 } { ( M - 1 ) { \binom { M } { 2 } } } \sum _ { k = 1 } ^ { M - 1 } k ( M - k ) = \frac { M + 1 } { 3 ( M - 1 ) } .\tag{12}
$$

For every $M \ge 4 , \phi _ { \mathrm { o n e } } ( M ) > \phi _ { \mathrm { c y c } } ( M )$ , although the cyclic schedule repeatedly separates every pair and therefore admits a positive certified exponent under Proposition 3.

This is why our empirical diagnostic uses the minimum over competitors: an average information score can look favorable even when one unresolved alternative still dominates localization error. The average-pair quantity is used in the experiments as an ablation of the minimum-over-competitors. For $M = 3 2 .$ , Proposition 1 gives the 0.516 versus 0.355 comparison used in the 1D experiment. Appendix A.10 gives the proof.

Classical pairwise converse. Let $\mathsf { P } _ { m } ^ { T }$ be the law of the complete adaptive transcript through round T under hypothesis m. The following is a specialization of the standard Bretagnolle–Huber/Le Cam two-point argument (Bretagnolle and Huber, 1979; Tsybakov, 2009).

Proposition 2 (Pairwise KL converse for the adaptive transcript). For any estimator,

$$
\operatorname* { s u p } _ { u \in [ M ] } \mathbb { P } _ { u } ( \widehat { m } _ { T } \neq u ) \geq \frac { 1 } { 4 } \exp \left[ - \operatorname* { m i n } _ { m \neq m ^ { \prime } } D _ { \mathrm { K L } } ( \mathsf { P } _ { m } ^ { T } \| \mathsf { P } _ { m ^ { \prime } } ^ { T } ) \right] .\tag{13}
$$

For the BSC query model and a common historydependent policy kernel,

$$
\begin{array} { r l r } {  { D _ { \mathrm { K L } } ( \mathsf { P } _ { m } ^ { T } \| \mathsf { P } _ { m ^ { \prime } } ^ { T } ) = \mathbb { E } _ { m } [ \sum _ { t < T } \sum _ { j } d _ { j } \mathbf { 1 } \{ q _ { j , t } ( m ) \neq q _ { j , t } ( m ^ { \prime } ) \} ] , } } \\ & { } & { d _ { j } = ( 1 - 2 \epsilon _ { j } ) \log \frac { 1 - \epsilon _ { j } } { \epsilon _ { j } } . \qquad ( 1 4 ) } \end{array}
$$

Hence sublinear expected information for even one pair rules out a positive uniform localization-error exponent.

Together with the finite-time upper bound, this gives a two-sided message: resolving the hardest remaining competitor is suficient for certification and also necessary for any method to achieve a positive uniform error exponent.

Beyond binary observations. The proof extends beyond BSC likelihoods. For a general actiondependent observation kernel, Eq. (4) is replaced by the corresponding Chernof coeficient between the two action-conditioned output laws. Under uniformly bounded pairwise log-likelihood ratios, the finite-time and anytime results remain valid; Appendix D gives the construction and bounded continuous example.

## 5 CERTIFIED EXPLORATION FOR ARBITRARY ADAPTIVE POLICIES

The certificate diagnoses realized evidence but does not force pairwise exploration. To guarantee coverage while otherwise preserving the base policy, let $\bar { \mathcal { Q } } _ { \mathrm { c e r t } } = \{ q ^ { ( 1 ) } , \dots , q ^ { ( L ) } \}$ be a synchronized finite separating family with minimum response-code distance

$$
d _ { \mathcal { Q } } : = \operatorname* { m i n } _ { m \neq m ^ { \prime } } \sum _ { \ell = 1 } ^ { L } \mathbf { 1 } \{ q ^ { ( \ell ) } ( m ) \neq q ^ { ( \ell ) } ( m ^ { \prime } ) \} .\tag{15}
$$

Because $\mathcal { Q } _ { \mathrm { c e r t } }$ is separating, $d _ { \mathcal { Q } } \geq 1$ . With $W ( s ) : =$ $\textstyle \sum _ { j } { \mathsf { J } } _ { j } ( s v _ { j } )$ , after $C _ { t }$ complete certified cycles every pair has received at least $C _ { t } d _ { \mathcal { Q } } W ( s )$ information; adaptive rounds can only add nonnegative information.

Proposition 3 (Certified exploration from a separating query family). For every $s \in S$

$$
\begin{array} { r } { \mathbb { P } _ { m ^ { \star } } \left( \widehat { m } _ { i , t } \neq m ^ { \star } \right) \leq K _ { \pi _ { 0 } } ( m ^ { \star } , s ) \exp ( s \Delta _ { i } - C _ { t } d _ { \mathcal { Q } } W ( s ) ) . } \end{array}\tag{16}
$$

If certified rounds occupy asymptotic fraction η and $C _ { t } \geq \eta t / L - c _ { 0 }$ , then

$$
E _ { i } \geq \frac { \eta d _ { \mathcal { Q } } } { L } \operatorname* { s u p } _ { s \in \mathcal { S } } W ( s ) .\tag{17}
$$

For ordered 1D thresholds, $L = M - 1$ and $d _ { \mathcal { Q } } = 1 ;$ a balanced threshold cycle is max–min optimal within synchronized threshold exploration (Appendix A.13). For an $S \times S$ grid, the S−1 vertical and S−1 horizontal cuts form a separating family with $L = 2 ( S - 1 )$ and $d _ { \mathcal { Q } } ~ = ~ 1$ Diagonal half-planes can still be used on adaptive rounds.

For rollouts with known ground truth we report the post-hoc worst-competitor information score

$$
{ \widehat \gamma } _ { T } ^ { \mathrm { d i a g } } : = \operatorname* { s u p } _ { s \in { \mathcal { S } } } \frac { 1 } { T } \operatorname* { m i n } _ { m \neq m ^ { \star } } { \mathcal { I } } _ { T } ( m ; s ) .\tag{18}
$$

Because s is optimized after observing the rollout, this is a ranking diagnostic, not the fixed-s anytime certificate in Theorem 2.

![](images/641f7e18a7407e4323f1063dc31f56892aa352d8c31c147363f46459272c15ab.jpg)  
Figure 2: 1D exponential localization across graph geometries. With the sensing policy fixed, network-average MAP error decays approximately exponentially before the Monte Carlo floor, with graphdependent finite-horizon rates. Shading denotes pointwise 95% confidence intervals across independent trajectories.

## 6 EXPERIMENTS

We test three predictions in ordered 1D and nonordered 2D queries: approximately exponential localization under linear information growth, graph-dependent finite-horizon behavior, and stronger association with the least-resolved competitor than with aggregate information. Rates follow the deterministic pre-floor fitting rule in Appendix G. Uncertainty uses independent trajectories after within-trajectory agent averaging; pooled correlations use 10,000 paired seed-cluster-bootstrap replicates. Curve intervals are computed on the original scale, with log-axis clipping only for display.

## 6.1 1D ordered thresholds: convergence, graph efects, and the bottleneck

Exponential convergence across graphs. We first fix exploration+bisection and vary only the communication graph $( M = 3 2 , N = 1 0$ , homogeneous ϵ = 0.25, 5,000 trials, 80 rounds). Before the Monte Carlo floor, MAP error is approximately log-linear with fitted exponents 0.4288 (complete), 0.4035 (ring), and 0.3384 (directed); the corresponding post-hoc hardest-competitor scores are 0.5380, 0.5295, and 0.5230. Since $\Delta _ { i }$ is an additive finite-time ofset, it does not change the asymptotic exponent. The observed finite-window slope differences can therefore reflect pre-asymptotic mixing efects and/or graph-dependent lower-tail behavior of realized information, rather than a change of asymptotic rate caused by $\Delta _ { i }$ itself. Figure 2 shows the curves; the squared-loss companion is in the appendix.

Worst-competitor study. We next vary policy, graph, sensor profile, and seed over 240 configurations. The deterministic fit rule returns a resolvable pre-floor exponent for 210; the remaining 30 are reported as $\mathrm { n / a }$ rather than assigned an imputed rate. Thus all 1D correlation statements are explicitly conditional on this 210-configuration subset. On it, the post-hoc hardestcompetitor score in Eq. (18) has Pearson $r _ { \mathrm { w o r s t } } = 0 . 8 9 1$ (95% seed-cluster-bootstrap CI [0.850, 0.925]) with the empirical localization exponent. As an ablation, we average the same pairwise information over all pairs,

$$
\begin{array} { r l } & { \overline { { \mathcal { Z } } } _ { T } ( s ) : = \binom { M } { 2 } ^ { - 1 } \displaystyle \sum _ { m < m ^ { \prime } } \mathcal { T } _ { T } ( m , m ^ { \prime } ; s ) , } \\ & { \qquad \widehat { \gamma } _ { T } ^ { \mathrm { a v g } } : = \displaystyle \operatorname* { s u p } _ { s \in \mathcal { S } } \frac { \overline { { \mathcal { T } } } _ { T } ( s ) } { T } , } \end{array}\tag{19}
$$

which gives $r _ { \mathrm { a v g } } = 0 . 4 0 1 ~ ( 9 5 \% ~ \mathrm { C I } ~ [ 0 . 3 5 9 , 0 . 4 3 5 ] )$ . The paired diference is $\Delta r = 0 . 4 9 0$ (95% CI [0.446, 0.527]), and the hardest-score regression slope is 0.549 (95% CI [0.524, 0.578]). Figure 3 summarizes this pooled linear association. Because points cluster by policy, these Pearson statistics should not be read as a within-policy rank-ordering guarantee.

## 6.2 2D spatial queries: policy behavior and the bottleneck

Ordered thresholds have special one-dimensional structure, so we next ask whether the same information bottleneck persists when neither the targets nor the admissible queries are linearly ordered. The target set is an $8 \times 8$ grid $( M = 6 4 )$ , and the adaptive library contains 42 vertical, horizontal, and diagonal half-plane queries. The certified family contains the 14 vertical/horizontal cuts and separates every pair. Posterior bisection chooses, from this finite spatial library, the query whose one-side posterior mass is closest to $1 / 2$ . Unlike Fig. 2, which isolates network efects in 1D with the sensing policy fixed, here we first illustrate policy-dependent localization on a fixed complete graph and then test the information bottleneck across policies, graphs, sensors, and seeds.

Policy-dependent 2D localization. Figure 4 shows network-average MAP error for posterior bisection, Thompson-style sampling, random spatial querying, and exploration+bisection on the complete graph. The four policies exhibit distinct pre-floor localization behavior. Random spatial querying is particularly diagnostic: its average-pair score is 0.44, whereas its hardest-competitor score and fitted exponent are both about 0.09. Thus aggregate pairwise coverage can appear substantial even when one poorly resolved alternative controls the observed localization rate.

2D worst-competitor study. We repeat the pooled design over four policies, three communication graphs, two sensor profiles, and ten seed blocks, giving 240 configurations. All 240 have valid pre-floor fits. The hardest-competitor score has $r _ { \mathrm { w o r s t } } = 0 . 7 6 8$ (95% CI [0.761, 0.777]), while the same average-pair proxy gives $r _ { \mathrm { a v g } } ~ = ~ 0 . 4 8 4$ (95% CI [0.474, 0.494]). Their paired diference is $\Delta r = 0 . 2 8 5$ (95% CI [0.270, 0.299]), and the hardest-score regression slope is 0.773 (95% CI [0.736, 0.811]). Graph efects persist in the pooled sweep, with mean exponents 0.44, 0.33, and 0.28 for complete, ring, and directed graphs, respectively. The same bottleneck behavior therefore persists after replacing ordered thresholds with a non-ordered spatial query library.

## 7 LIMITATIONS

The theory assumes a finite hypothesis class, fixed primitive graph, common full-support prior, and known conditionally independent sensors. The fixed graph is structural because the proof uses one stationary left eigenvector; time-varying graphs need a productmixing argument. Bounded pairwise log-likelihood ratios exclude Gaussian tails directly. Online stopping also needs global query logging, naive $O ( M ^ { 2 } )$ pair tracking, and a pre-fixed s. Learned-policy exponents further require a lower-tail information-growth guarantee. Continuous hypotheses, misspecification, and these deployment extensions remain open.

## 8 CONCLUSION

We introduced a pathwise information certificate for decentralized adaptive sensing that evaluates the evidence actually collected by an arbitrary history-dependent policy. The same realized-information process yields fixed-horizon MAP bounds, an anytime network-wide stopping rule, and, under linear information growth, exponential localization rates. A pairwise KL converse explains why the hardest remaining alternative is the relevant bottleneck. Across the tested 1D and 2D query families, worst-competitor information has a stronger pooled association with localization speed than average-pair information, supporting the certificate as a reliability tool and finite-trajectory diagnostic.

![](images/b89ebfc914cf3ea5f6afc0018eebbf219a78657e89b4048d2bc75803876e8395.jpg)

![](images/050522b70629dc3e793fcaef9ad74f86540c89344e5c9552d775c667972635c3.jpg)  
Figure 3: 1D hardest-competitor bottleneck. Across policies, graph topologies, sensor profiles, and seed blocks, empirical localization speed correlates more strongly with the worst-competitor information score than with the average-pair proxy. The paired diference is $\Delta r = 0 . 4 9 0$ with 95% CI [0.446, 0.527].

![](images/5ad857e6d8152713964eb4a3b4c15d57eccdd25e55fae8d27ee7a857ed8cf387.jpg)  
Figure 4: Policy-dependent localization in 2D. Network-average MAP error for posterior bisection, Thompson-style sampling, random spatial querying, and exploration+bisection on the 8×8 spatial grid with a complete communication graph. Shading denotes pointwise 95% confidence intervals across independent Monte Carlo trajectories.

(a)  
![](images/45770dff49a421eaa7ce89c25a3ce642510e183267c0fe1af28fdf2e2ba5d9ba.jpg)

(b)  
![](images/50c2f386622c793909cad31b8b3a51cde7d96191c305020cfe8fc6513c415ecf.jpg)  
Figure 5: 2D hardest-competitor bottleneck. Across 240 policy–graph–sensor–seed configurations on an $8 \times 8$ spatial grid, worst-competitor information correlates more strongly with empirical localization speed than the average-pair proxy. The paired diference is $\Delta r = 0 . 2 8 5$ with 95% CI [0.270, 0.299].

## ACKNOWLEDGMENTS

Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the oficial policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

## AI USE STATEMENT

In this work, we used generative AI tools to polish the writing (language, grammatical refinement), and debug experimental code. We did not use AI for developing research ideas, designing methodology, proving mathematical statements, conducting experiments, or analyzing results. We have reviewed all AI-assisted work to ensure accuracy and faithful representation of our claims. All mathematical and algorithmic claims, experimental findings, and paper contents were conceived and verified by the authors. We take responsibility for the final content of this work.

## References

Bajcsy, R., Aloimonos, Y., and Tsotsos, J. K. (2018). Revisiting active perception. Autonomous Robots, 42(2):177–196.

Bhattacharyya, A. (1943). On a measure of divergence between two statistical populations defined by their

probability distributions. Bulletin of the Calcutta Mathematical Society, 35:99–109.

Bretagnolle, J. and Huber, C. (1979). Estimation des densit´es: risque minimax. Zeitschrift f¨ur Wahrscheinlichkeitstheorie und Verwandte Gebiete, 47(2):119– 137.

Casta˜n´on, D. A., Tsiligkaridis, T., and Hero, A. O. (2017). Corrections to “on decentralized estimation with active queries”. IEEE Transactions on Signal Processing, 65(18):4971–4972.

Chernof, H. (1952). A measure of asymptotic eficiency for tests of a hypothesis based on the sum of observations. The Annals of Mathematical Statistics, 23(4):493–507.

Chernof, H. (1959). Sequential design of experiments. The Annals of Mathematical Statistics, 30(3):755– 770.

Chung, H. W., Sadler, B. M., Zheng, L., and Hero, A. O. (2018). Unequal error protection querying policies for the noisy 20 questions problem. IEEE Transactions on Information Theory, 64(2):1105–1131.

Gorry, G. A. and Barnett, G. O. (1968). Sequential diagnosis by computer. JAMA, 205(12):849–854.

Horstein, M. (1963). Sequential transmission using noiseless feedback. IEEE Transactions on Information Theory, 9(3):136–143.

Howard, S. R., Ramdas, A., McAulife, J., and Sekhon, J. (2020). Time-uniform chernof bounds via nonnegative supermartingales. Probability Surveys, 17:257– 317.

Hsu, C.-Y. and Wang, I.-H. (2025). Tradeofs among action taking policies matter in active sequential multi-hypothesis testing: The optimal error exponent region. IEEE Transactions on Information Theory, 71(9):6546–6565.

Huang, B. and Wang, I.-H. (2025). On the price of decentralization in decentralized detection. IEEE Transactions on Information Theory, 71(4):2341– 2359.

Jadbabaie, A., Molavi, P., Sandroni, A., and Tahbaz-Salehi, A. (2012). Non-Bayesian social learning. Games and Economic Behavior, 76(1):210–225.

Jedynak, B., Frazier, P. I., and Sznitman, R. (2012). Twenty questions with noise: Bayes optimal policies for entropy loss. Journal of Applied Probability, 49(1):114–136.

Lalitha, A. and Javidi, T. (2017). Learning via active hypothesis testing over networks. In 2017 IEEE Information Theory Workshop (ITW), pages 374– 378.

Lalitha, A., Javidi, T., and Sarwate, A. D. (2018). Social learning and distributed hypothesis testing. IEEE Transactions on Information Theory, 64(9):6161–6179.

Levin, D. A. and Peres, Y. (2017). Markov Chains and Mixing Times. American Mathematical Society, 2 edition. With contributions by Elizabeth L. Wilmer.

Nedi´c, A., Olshevsky, A., and Uribe, C. A. (2017). Fast convergence rates for distributed non-Bayesian learning. IEEE Transactions on Automatic Control, 62(11):5538–5553.

Nitinawarat, S., Atia, G. K., and Veeravalli, V. V. (2013). Controlled sensing for multihypothesis testing. IEEE Transactions on Automatic Control, 58(10):2451–2464.

Rahimian, M. A. and Jadbabaie, A. (2015). Learning without recall: A case for log-linear learning. IFAC-PapersOnLine, 48(22):46–51.

Rangi, A., Franceschetti, M., and Marano, S. (2021). Distributed Chernof test: Optimal decision systems over networks. IEEE Transactions on Information Theory, 67(4):2399–2425.

R´enyi, A. (1961). On measures of entropy and information. In Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Contributions to the Theory of Statistics, pages 547–561. University of California Press.

Seneta, E. (2006). Non-negative Matrices and Markov Chains. Springer, 2 edition.

Shahrampour, S., Rakhlin, A., and Jadbabaie, A. (2016). Distributed detection: Finite-time analysis and impact of network topology. IEEE Transactions on Automatic Control, 61(11):3256–3268.

Singh, A., Krause, A., Guestrin, C., and Kaiser, W. J. (2009). Eficient informative sensing using multiple robots. Journal of Artificial Intelligence Research, 34:707–755.

Szostak, H. and Cohen, K. (2024). Deep multi-agent reinforcement learning for decentralized active hypothesis testing. IEEE Access, 12:130444–130459.

Tsiligkaridis, A. and Tsiligkaridis, T. (2017). Distributed probabilistic bisection search using social learning. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4069–4073.

Tsiligkaridis, T. (2018). Decentralized adaptive search using the noisy 20 questions framework in timevarying networks. Signal Processing, 142:330–339.

Tsiligkaridis, T., Sadler, B. M., and Hero, A. O. (2014). Collaborative 20 questions for target local-

ization. IEEE Transactions on Information Theory, 60(4):2233–2252.

Tsiligkaridis, T., Sadler, B. M., and Hero, A. O. (2015). On decentralized estimation with active queries. IEEE Transactions on Signal Processing, 63(10):2610–2622.

Tsybakov, A. B. (2009). Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, New York.

van Erven, T. and Harremo¨es, P. (2014). R´enyi divergence and Kullback–Leibler divergence. IEEE Transactions on Information Theory, 60(7):3797–3820.

Ville, J. (1939). Etude critique de la notion de collectif <sup>´</sup> . Gauthier-Villars, Paris.

Waeber, R., Frazier, P. I., and Henderson, S. G. (2013). Bisection search with noisy responses. SIAM Journal on Control and Optimization, 51(3):2261–2279.

Wu, B. and Uribe, C. A. (2025). Frequentist guarantees of distributed (non)-Bayesian inference. Journal of Machine Learning Research, 26(168):1–65.

Zhou, L. and Hero, A. O. (2021). Achievable resolution limits for the noisy adaptive 20 questions problem. In 2021 IEEE International Symposium on Information Theory (ISIT), pages 784–789.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes] The decentralized adaptive querying model, BSC observation model, local Bayes update, and geometric pooling rule are fully specified in Section 3, with the Bayes/socialpooling update in Eq. (3); the controlledobservation extension is summarized in Section 4 and derived in Appendix D.

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes] Per-round computational complexity is given in Section 3, and the stoppingtime/sample-complexity consequence is derived after Theorem 2 in Section 4.2.

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Yes] Anonymized code is included in the supplementary material.

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. [Yes] Primitive/stationary graph, common full-support prior, conditionally independent sensing noise, and (for the controlled-observation extension summarized in Section 4 and derived in Appendix D) bounded log-likelihood ratios are all stated explicitly.

(b) Complete proofs of all theoretical results. [Yes] Full proofs of the finite-time and anytime certificates, the stochastic-rate and MAP/MSE corollaries, the aggregateinformation counterexample, the pairwise minimax converse, and the certifiedexploration results are given in Appendix A; the explicit mixing calculation is included there as well.

(c) Clear explanations of any assumptions. [Yes] Each theorem is followed by an interpretive paragraph relating assumptions to the decentralized adaptive-querying setting (e.g., the discussion after Theorem 1 and Proposition 3).

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Yes] The supplementary material includes the code and reproduction instructions, with graph construction, sensor, and seed configuration specified.

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [Yes] Simulation parameters (grid geometry, agents N, BSC noise ϵ, query library, trial counts, round horizons, and seeds) are specified per experiment in Appendix G.

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Yes] We clearly define all empirical statistics and uncertainty estimates. MAP-error and localization-loss curves report pointwise 95% intervals across independent Monte Carlo trajectories, with agents averaged within each trajectory. The 1D code uses Student-t critical values (normal fallback), while the 2D plotting code uses mean ±1.96 standard errors; all intervals are computed on the original linear scale and any log-plot clipping is display-only. For both pooled ten-seed bottleneck studies, we report 95% seed-cluster-bootstrap confidence intervals for Pearson correlation and the regression slope using 10,000 bootstrap replicates; the worst-competitor versus average-pair comparison additionally reports a paired seed-clusterbootstrap confidence interval for ∆r.

(d) A description of the computing infrastructure used (e.g., type of GPUs, internal cluster, or cloud provider). [Yes] Experiments used CPU execution on a single 4-core workstation.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator if your work uses existing assets. [Not Applicable] The paper does not build on external datasets, pretrained models, or third-party code assets beyond standard numerical libraries.

(b) The license information of the assets, if applicable. [Not Applicable] No external licensed assets are used.

(c) New assets either in the supplemental material or as a URL, if applicable. [Yes] Simulation code implementing the query policies, graph constructions, and figure-generation scripts is released as a new asset in the supplementary material.

(d) Information about consent from data providers/curators. [Not Applicable] All data are synthetically generated via controlled simulation; no external data providers are involved.

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Not Applicable] The work involves no human subject data, PII, or sensitive content.

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. [Not Applicable] No human subjects or crowdsourcing were involved in this work.

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable] No human subjects research was conducted.

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable] No participant compensation was involved.

## A COMPLETE TECHNICAL PROOFS

## A.1 Posterior-odds recursion with a common full-support prior

Fix a false hypothesis $m \neq m ^ { \star }$ and define the posterior odds and log-odds

$$
R _ { i , t } ( m ) : = \frac { \mu _ { i , t } ( m ) } { \mu _ { i , t } ( m ^ { \star } ) } , \qquad r _ { i , t } ( m ) : = \log R _ { i , t } ( m ) .
$$

For agent $j ,$ define the one-step likelihood ratio

$$
L _ { j , t } ( m ) : = \frac { \ell _ { j , t } ( Y _ { j , t + 1 } \mid m ) } { \ell _ { j , t } ( Y _ { j , t + 1 } \mid m ^ { \star } ) } .
$$

We first derive the recursion used throughout the proofs. From the local Bayes update,

$$
\frac { \widetilde { \mu } _ { j , t + 1 } ( m ) } { \widetilde { \mu } _ { j , t + 1 } ( m ^ { \star } ) } = \frac { \mu _ { j , t } ( m ) \ell _ { j , t } ( Y _ { j , t + 1 } \mid m ) } { \mu _ { j , t } ( m ^ { \star } ) \ell _ { j , t } ( Y _ { j , t + 1 } \mid m ^ { \star } ) } = R _ { j , t } ( m ) L _ { j , t } ( m ) ,
$$

because the local normalizing denominator cancels. Taking the ratio of the two pooled beliefs in (3) similarly cancels the social-pooling normalizer, giving

$$
R _ { i , t + 1 } ( m ) = \prod _ { j = 1 } ^ { N } [ R _ { j , t } ( m ) L _ { j , t } ( m ) ] ^ { A _ { i j } } .
$$

Taking logarithms and defining $\pmb { r } _ { t } ( m ) = ( r _ { 1 , t } ( m ) , \dots , r _ { N , t } ( m ) ) ^ { \top }$ and $\lambda _ { t } ( m ) = ( \log L _ { 1 , t } ( m ) , \dots , \log L _ { N , t } ( m ) ) ^ { \top }$ yields

$$
r _ { t + 1 } ( m ) = A \big ( r _ { t } ( m ) + \lambda _ { t } ( m ) \big ) .\tag{20}
$$

Because all agents use the same prior,

$$
r _ { 0 } ( m ) = \log \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } { \bf 1 } .
$$

## A.2 Fractional-moment martingale

Define the stationary geometric posterior odds

$$
G _ { t } ( m ) : = \prod _ { i = 1 } ^ { N } R _ { i , t } ( m ) ^ { v _ { i } } = \exp \bigl ( { v ^ { \top } } { r _ { t } ( m ) } \bigr ) .
$$

Premultiplying (20) by $\pmb { v } ^ { \top }$ and using $\pmb { v } ^ { \top } A = \pmb { v } ^ { \top }$ gives

$$
\boldsymbol { v } ^ { \top } \boldsymbol { r } _ { t + 1 } ( m ) = \boldsymbol { v } ^ { \top } \boldsymbol { r } _ { t } ( m ) + \sum _ { j = 1 } ^ { N } v _ { j } \log L _ { j , t } ( m ) ,
$$

and therefore

$$
G _ { t + 1 } ( \boldsymbol { m } ) = G _ { t } ( \boldsymbol { m } ) \prod _ { j = 1 } ^ { N } L _ { j , t } ( \boldsymbol { m } ) ^ { v _ { j } } .\tag{21}
$$

Fix $s \in S ,$ , so $\alpha _ { j } : = s v _ { j } \in [ 0 , 1 ]$ for every agent. Conditional on $\mathcal { F } _ { t } ,$ every query selected at round t is fixed. If $q _ { j , t } ( m ) = q _ { j , t } ( m ^ { \star } )$ , the two hypotheses induce the same BSC output law, hence $L _ { j , t } ( m ) = 1$ almost surely and $\mathbb { E } _ { m ^ { \star } } [ L _ { j , t } ( m ) ^ { \alpha _ { j } } \mid \mathcal { F } _ { t } ] = 1$

If the query separates the pair, then under the true hypothesis $m ^ { \star }$ the likelihood ratio equals

$$
L _ { j , t } ( m ) = \left\{ { \epsilon } _ { j } / ( 1 - \epsilon _ { j } ) , \mathrm { ~ w i t h ~ p r o b a b i l i t y ~ } 1 - \epsilon _ { j } , \right.
$$

Consequently,

$$
\mathbb { E } _ { m ^ { \star } } [ L _ { j , t } ( m ) ^ { \alpha _ { j } } \mid \mathcal { F } _ { t } ] = ( 1 - \epsilon _ { j } ) ^ { 1 - \alpha _ { j } } \epsilon _ { j } ^ { \alpha _ { j } } + \epsilon _ { j } ^ { 1 - \alpha _ { j } } ( 1 - \epsilon _ { j } ) ^ { \alpha _ { j } } = e ^ { - \mathsf { J } _ { j } ( \alpha _ { j } ) } .
$$

Combining the separating and nonseparating cases $\mathrm { g i }$ ves

$$
\mathbb { E } _ { m ^ { \star } } \left[ L _ { j , t } ( m ) ^ { s v _ { j } } \mid \mathcal { F } _ { t } \right] = \exp [ -  { \mathrm { J } _ { j } } ( s v _ { j } ) I _ { j , t } ( m , m ^ { \star } ) ] .
$$

The BSC noises are conditionally independent across agents, so raising (21) to the power s and taking conditional expectation yields

$$
\mathbb { E } _ { m ^ { \star } } [ G _ { t + 1 } ( m ) ^ { s } \mid \mathcal { F } _ { t } ] = G _ { t } ( m ) ^ { s } \exp \left[ - \sum _ { j = 1 } ^ { N } \mathsf { J } _ { j } ( s v _ { j } ) I _ { j , t } ( m , m ^ { \star } ) \right] .
$$

Because

$$
\mathcal { T } _ { t + 1 } ( m ; s ) = \mathcal { T } _ { t } ( m ; s ) + \sum _ { j = 1 } ^ { N } \mathsf { J } _ { j } ( s v _ { j } ) I _ { j , t } ( m , m ^ { \star } ) ,
$$

the process

$$
Z _ { t } ( m ; s ) : = G _ { t } ( m ) ^ { s } \exp \bigl (  { \mathcal { T } } _ { t } ( m ; s ) \bigr )\tag{22}
$$

satisfies

$$
\mathbb { E } _ { m ^ { \star } } \big [ Z _ { t + 1 } ( m ; s ) \mid \mathcal { F } _ { t } \big ] = Z _ { t } ( m ; s ) .
$$

Thus $\{ Z _ { t } ( m ; s ) \} _ { t \geq 0 }$ is a nonnegative martingale. $\mathrm { A t                       } t = 0 \mathrm { . }$ , since $\textstyle \sum _ { i } v _ { i } = 1$ 2

$$
G _ { 0 } ( m ) = \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } ,
$$

so for every $t ,$

$$
\mathbb { E } _ { m ^ { \star } } Z _ { t } ( m ; s ) = Z _ { 0 } ( m ; s ) = \left( { \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } } \right) ^ { s } .\tag{23}
$$

## A.3 Network disagreement is a bounded transient

Unrolling (20) gives

$$
r _ { t } ( m ) = A ^ { t } r _ { 0 } ( m ) + \sum _ { \tau = 0 } ^ { t - 1 } A ^ { t - \tau } \lambda _ { \tau } ( m ) .
$$

The common-prior vector is proportional to 1. Since $A ^ { t } \mathbf { 1 } = \mathbf { 1 }$ and $\pmb { v } ^ { \top } \mathbf { 1 } = 1$ , its contribution is identical in $r _ { i , t } ( m )$ and $\pmb { v } ^ { \top } \pmb { r } _ { t } ( m )$ and therefore cancels. Hence

$$
r _ { i , t } ( m ) - v ^ { \top } r _ { t } ( m ) = \sum _ { \tau = 0 } ^ { t - 1 } \sum _ { j = 1 } ^ { N } \big ( ( A ^ { t - \tau } ) _ { i j } - v _ { j } \big ) \lambda _ { j , \tau } ( m ) .
$$

For the BSC model,

$$
| \lambda _ { j , \tau } ( m ) | = | \log L _ { j , \tau } ( m ) | \leq \log \frac { 1 - \epsilon _ { j } } { \epsilon _ { j } } = b _ { j } .
$$

Therefore, by the triangle inequality,

$$
\left| r _ { i , t } ( \boldsymbol { m } ) - \boldsymbol { v } ^ { \top } \boldsymbol { r } _ { t } ( \boldsymbol { m } ) \right| \leq \sum _ { \tau = 0 } ^ { t - 1 } \sum _ { j = 1 } ^ { N } \left| ( A ^ { t - \tau } ) _ { i j } - v _ { j } \right| b _ { j } .
$$

Letting $k = t - \tau$ and enlarging the finite sum to the infinite series in (6),

$$
\begin{array} { r } { \left| r _ { i , t } ( m ) - \pmb { v } ^ { \top } \pmb { r } _ { t } ( m ) \right| \leq \Delta _ { i } . } \end{array}\tag{24}
$$

Because $v ^ { \top } r _ { t } ( m ) = \log G _ { t } ( m )$ , the upper side of (24) implies

$$
r _ { i , t } ( m ) \leq \log G _ { t } ( m ) + \Delta _ { i } .
$$

Exponentiating and raising to $s > 0$ gives

$$
R _ { i , t } ( m ) ^ { s } \leq e ^ { s \Delta _ { i } } G _ { t } ( m ) ^ { s } .\tag{25}
$$

## A.4 Proof of Theorem 1

Fix i, t, and the true hypothesis $m ^ { \star }$ . If the MAP estimate is incorrect, then at least one false hypothesis has posterior mass at least that of the truth. Therefore

$$
\{ \hat { m } _ { i , t } \neq m ^ { \star } \} \subseteq \bigcup _ { m \neq m ^ { \star } } \{ R _ { i , t } ( m ) \geq 1 \} ,
$$

and the union bound gives

$$
\mathbb { P } _ { m ^ { \star } } ( \widehat { m } _ { i , t } \neq m ^ { \star } ) \leq \sum _ { m \neq m ^ { \star } } \mathbb { P } _ { m ^ { \star } } ( R _ { i , t } ( m ) \geq 1 ) .\tag{26}
$$

Now fix one false m. Split the event according to whether enough information has accumulated:

$$
\begin{array} { r } { \mathbb { P } _ { m ^ { \star } } \big ( R _ { i , t } ( m ) \geq 1 \big ) \leq \mathbb { P } _ { m ^ { \star } } \big ( \mathcal { Z } _ { t } ( m ; s ) < a _ { m , t } \big ) + \mathbb { P } _ { m ^ { \star } } \big ( R _ { i , t } ( m ) \geq 1 , \ \mathcal { Z } _ { t } ( m ; s ) \geq a _ { m , t } \big ) . } \end{array}
$$

On the second event, $R _ { i , t } ( m ) ^ { s } \geq 1 $ , so

$$
\mathbb { P } _ { m ^ { \star } } \bigl ( R _ { i , t } ( m ) \geq 1 , ~ \mathcal { Z } _ { t } ( m ; s ) \geq a _ { m , t } \bigr ) \leq \mathbb { E } _ { m ^ { \star } } \bigl [ R _ { i , t } ( m ) ^ { s } \mathbf { 1 } \{ \mathcal { Z } _ { t } ( m ; s ) \geq a _ { m , t } \} \bigr ] .
$$

Using (25),

$$
\mathbb { E } _ { m ^ { \star } } [ R _ { i , t } ( m ) ^ { s } \mathbf { 1 } \{ { \mathcal { T } } _ { t } \geq a _ { m , t } \} ] \leq e ^ { s \Delta _ { i } } \mathbb { E } _ { m ^ { \star } } [ G _ { t } ( m ) ^ { s } \mathbf { 1 } \{ { \mathcal { T } } _ { t } \geq a _ { m , t } \} ] .
$$

From (22), $G _ { t } ( m ) ^ { s } = e ^ { - \mathscr { T } _ { t } ( m ; s ) } Z _ { t } ( m ; s )$ . Hence

$$
\begin{array} { r l } & { e ^ { s \Delta _ { i } } \mathbb { E } _ { m ^ { \star } } \bigl [ G _ { t } ( m ) ^ { s } \mathbf { 1 } \{ { \mathcal { T } } _ { t } \geq a _ { m , t } \} \bigr ] } \\ & { \qquad = e ^ { s \Delta _ { i } } \mathbb { E } _ { m ^ { \star } } \Bigl [ e ^ { - { \mathcal { T } } _ { t } ( m ; s ) } Z _ { t } ( m ; s ) \mathbf { 1 } \{ { \mathcal { T } } _ { t } \geq a _ { m , t } \} \Bigr ] } \\ & { \qquad \leq e ^ { s \Delta _ { i } - a _ { m , t } } \mathbb { E } _ { m ^ { \star } } Z _ { t } ( m ; s ) . } \end{array}
$$

Applying (23) gives

$$
\mathbb { P } _ { m ^ { \star } } \left( R _ { i , t } ( m ) \geq 1 \right) \leq \mathbb { P } _ { m ^ { \star } } \left( \mathcal { T } _ { t } ( m ; s ) < a _ { m , t } \right) + \left( \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } \right) ^ { s } e ^ { s \Delta _ { i } - a _ { m , t } } .
$$

Summing this inequality over m ̸= $m ^ { \star }$ in (26) proves (7).

## A.5 Proof of the anytime-valid network-wide certificate

We first prove a pairwise time-uniform statement. Fix agent $i ,$ false hypothesis $m \neq m ^ { \star }$ , and a constant $c \geq 0$ . If at some time t both $R _ { i , t } ( m ) \geq 1$ and $\begin{array} { r } { \mathcal { T } _ { t } ( m ; s ) \ge c , } \end{array}$ then (25) implies

$$
1 \leq R _ { i , t } ( m ) ^ { s } \leq e ^ { s \Delta _ { i } } G _ { t } ( m ) ^ { s } .
$$

Therefore $G _ { t } ( m ) ^ { s } \geq e ^ { - s \Delta _ { i } }$ . Multiplying by $e ^ { \mathcal { I } _ { t } ( m ; s ) }$ shows

$$
Z _ { t } ( m ; s ) = G _ { t } ( m ) ^ { s } e ^ { { \mathcal { T } } _ { t } ( m ; s ) } \geq e ^ { c - s \Delta _ { i } } .
$$

Hence the event

$$
\{ \exists t \geq 0 : \ R _ { i , t } ( m ) \geq 1 , \ { \mathcal { T } } _ { t } ( m ; s ) \geq c \}
$$

is contained in

$$
\left\{ \operatorname* { s u p } _ { t \geq 0 } Z _ { t } ( m ; s ) \geq e ^ { c - s \Delta _ { i } } \right\} .
$$

Since $Z _ { t } ( m ; s )$ is a nonnegative martingale, Ville’s inequality (Ville, 1939; Howard et al., 2020) gives

$$
\mathbb { P } _ { m ^ { \star } } \biggl ( \operatorname* { s u p } _ { t \geq 0 } Z _ { t } ( m ; s ) \geq a \biggr ) \leq \frac { \mathbb { E } _ { m ^ { \star } } Z _ { 0 } ( m ; s ) } { a } \qquad ( a > 0 ) .
$$

Taking $a = e ^ { c - s \Delta _ { i } }$ and using (23),

$$
\mathbb { P } _ { m ^ { \star } } ( \exists t \geq 0 : ~ R _ { i , t } ( m ) \geq 1 , ~ \mathcal { Z } _ { t } ( m ; s ) \geq c ) \leq \left( \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } \right) ^ { s } e ^ { s \Delta _ { i } - c } .\tag{27}
$$

We next pass from one false hypothesis and one agent to all of them. If agent i makes an incorrect MAP decision at time t, then some false m satisfies $R _ { i , t } ( m ) \geq 1$ . Moreover, by definition of (8),

$$
\mathbb { Z } _ { t } ( s ) \geq c \quad \Longrightarrow \quad \mathbb { Z } _ { t } ( m , m ^ { \star } ; s ) \geq c \quad { \mathrm { f o r ~ e v e r y ~ } } m \neq m ^ { \star } .
$$

The true-vs-false pairwise quantity $\mathcal { T } _ { t } ( m , m ^ { \star } ; s )$ is exactly the $\mathcal { T } _ { t } ( m ; s )$ used in the martingale proof. Consequently, a union bound over agents and false hypotheses, followed by (27), yields

$$
\begin{array} { r l } & { \mathbb { P } _ { m ^ { \star } } \bigl ( \exists i , \exists t : \widehat { m } _ { i , t } \neq m ^ { \star } , \mathbb { Z } _ { t } ( s ) \geq c \bigr ) } \\ & { \qquad \leq \displaystyle \sum _ { i = 1 } ^ { N } \displaystyle \sum _ { m \neq m ^ { \star } } \left( \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } \right) ^ { s } e ^ { s \Delta _ { i } - c } } \\ & { \qquad \leq N { \mathcal K } _ { \operatorname* { m a x } } ( s ) e ^ { s \Delta _ { \operatorname* { m a x } } - c } . } \end{array}
$$

Set $c = c _ { \delta } ( s )$ from (9). The final expression equals δ.

It remains only to connect this event to the stopping time. Every term in (8) is nonnegative, so $\underline { { \mathcal { T } } } _ { t } ( s )$ is nondecreasing in t. Therefore, on $\{ \tau _ { \delta } ( s ) < \infty \}$ ,

$$
t \geq \tau _ { \delta } ( s ) \quad \implies \quad \underline { { { \mathcal { Z } } } } _ { t } ( s ) \geq c _ { \delta } ( s ) .
$$

Thus

$$
\{ \exists i , \exists t \geq \tau _ { \delta } ( s ) : \widehat { m } _ { i , t } \neq m ^ { \star } \} \subseteq \{ \exists i , \exists t : \widehat { m } _ { i , t } \neq m ^ { \star } , \mathbb { Z } _ { t } ( s ) \geq c _ { \delta } ( s ) \} ,
$$

whose probability is at most δ. This proves Theorem 2.

## A.6 Derivation of the anytime sample-complexity consequence

Assume $\underline { { \mathcal { T } } } _ { t } ( s ) \geq \gamma t - B$ for every t. By the definition of $\tau _ { \delta } ( s )$ in (9), it is enough to find an integer t for which

$$
\gamma t - B \geq c _ { \delta } ( s ) .
$$

Indeed, the assumed information-growth inequality then implies

$$
\underline { { { \mathcal { T } } } } _ { t } ( s ) \geq \gamma t - B \geq c _ { \delta } ( s ) ,
$$

so the stopping condition has been met no later than time t. Solving the first inequality gives

$$
t \geq \frac { B + c _ { \delta } ( s ) } { \gamma } .
$$

Substituting the definition of $c _ { \delta } ( s )$ from (9),

$$
t \geq \frac { B + s \Delta _ { \operatorname* { m a x } } + \log ( N \mathcal { K } _ { \operatorname* { m a x } } ( s ) / \delta ) } { \gamma } .
$$

Taking the smallest integer satisfying this inequality yields

$$
\tau _ { \delta } ( s ) \leq \left\lceil \frac { B + s \Delta _ { \operatorname* { m a x } } + \log ( N \mathcal { K } _ { \operatorname* { m a x } } ( s ) / \delta ) } { \gamma } \right\rceil ,
$$

which is the sample-complexity statement given after Theorem 2.

## A.7 Explicit mixing bound

For a finite primitive stochastic matrix, Perron–Frobenius theory gives a simple eigenvalue at 1 and places every other eigenvalue strictly inside the unit disk (Seneta, 2006). Consequently, the powers converge geometrically to the stationary projection: there exist $C _ { A } < \infty$ and $\rho _ { A } < 1$ such that

$$
\operatorname* { m a x } _ { i } \left\| e _ { i } ^ { \top } A ^ { k } - { \pmb v } ^ { \top } \right\| _ { 1 } \leq C _ { A } \rho _ { A } ^ { k } .
$$

If a nonsymmetric A has nontrivial Jordan blocks, the spectral expansion can contain a polynomial factor $k ^ { r - 1 } \lambda ^ { k } ;$ for any $\rho _ { A }$ strictly larger than the largest nonunit eigenvalue modulus, that polynomial factor is absorbed into a finite constant $C _ { A }$ . Thus the displayed geometric bound still holds. This is the standard finite-state geometric-ergodicity property of an irreducible aperiodic chain (Levin and Peres, 2017).

By definition of $\Delta _ { i }$ and $b _ { j } \leq b _ { \operatorname* { m a x } }$

$$
\Delta _ { i } \leq b _ { \operatorname* { m a x } } \sum _ { k = 1 } ^ { \infty } \big \| e _ { i } ^ { \top } A ^ { k } - \pmb { v } ^ { \top } \big \| _ { 1 } \leq b _ { \operatorname* { m a x } } C _ { A } \sum _ { k = 1 } ^ { \infty } \rho _ { A } ^ { k } = b _ { \operatorname* { m a x } } C _ { A } \frac { \rho _ { A } } { 1 - \rho _ { A } } ,
$$

which proves the stated mixing bound.

Now suppose A is symmetric and doubly stochastic. Then ${ \pmb v } = { \bf 1 } / N$ . Define $\lambda _ { \star } : = \operatorname* { m a x } _ { \ell \geq 2 } | \lambda _ { \ell } ( A ) | < 1$ , the largest absolute nonunit eigenvalue. The component orthogonal to 1 contracts in Euclidean norm by at most ${ \lambda } _ { \star } ^ { k }$ . Hence

$$
\left\| e _ { i } ^ { \top } A ^ { k } - { \pmb v } ^ { \top } \right\| _ { 2 } \leq \lambda _ { \star } ^ { k } \left\| e _ { i } - \frac { \bf 1 } { N } \right\| _ { 2 } = \lambda _ { \star } ^ { k } \sqrt { 1 - \frac { 1 } { N } } .
$$

Using $\| x \| _ { 1 } \leq \sqrt { N } \| x \| _ { 2 }$ 2

$$
\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\| e _ { i } ^ { \top } A ^ { k } - { \pmb v } ^ { \top } \aftergroup \egroup \right\| _ { 1 } \leq \sqrt { N - 1 } \lambda _ { \star } ^ { k } . } \end{array}
$$

Summing the geometric series gives

$$
\Delta _ { i } \le b _ { \mathrm { { m a x } } } \sqrt { N - 1 } \frac { \lambda _ { \star } } { 1 - \lambda _ { \star } } .
$$

## A.8 Proof of Corollary 1

For all suficiently large t such that $\gamma t - B \geq 0$ , set $a _ { m , t } = \gamma t - B$ for every false m in Theorem 1. Then

$$
\begin{array} { r l } {  { \mathbb { P } _ { m ^ { \star } } ( \widehat { m } _ { i , t } \neq m ^ { \star } ) \leq \sum _ { m \neq m ^ { \star } } C _ { m } e ^ { - \kappa t } + \sum _ { m \neq m ^ { \star } } ( \frac { \pi _ { 0 } ( m ) } { \pi _ { 0 } ( m ^ { \star } ) } ) ^ { s } e ^ { s \Delta _ { i } - ( \gamma t - B ) } } } \\ & { = ( \sum _ { m \neq m ^ { \star } } C _ { m } ) e ^ { - \kappa t } + K _ { \pi _ { 0 } } ( m ^ { \star } , s ) e ^ { s \Delta _ { i } + B } e ^ { - \gamma t } . } \end{array}
$$

This finite-t bound immediately yields the rate claim. If $c _ { 1 } , c _ { 2 } > 0$ , then $c _ { 1 } e ^ { - \kappa t } + c _ { 2 } e ^ { - \gamma t }$ has asymptotic exponential rate min $\{ \kappa , \gamma \}$ , because the slower-decaying exponential dominates the sum up to multiplicative constants. Therefore $E _ { i } \ge \mathrm { m i n } \{ \gamma , \kappa \}$ . If the information lower bound holds deterministically, the lower-tail probabilities are zero and the $e ^ { - \kappa t }$ term disappears. □

## A.9 Proof of Corollary 2

On the event $\{ \widehat { m } _ { i , t } \neq m ^ { \star } \}$ , the estimated and true hypotheses are distinct, so by definition of the minimum separation in the chosen metric,

$$
d ( { \widehat { \theta } } _ { i , t } , \theta _ { m ^ { \star } } ) \geq d _ { \operatorname* { m i n } } .
$$

For every outcome, both points belong to the finite target set, so

$$
d ( \widehat { \theta } _ { i , t } , \theta _ { m ^ { \star } } ) \leq D _ { \Theta } .
$$

Combining the two observations pointwise,

$$
d _ { \operatorname* { m i n } } ^ { 2 } \mathbf { 1 } \{ \widehat { m } _ { i , t } \neq m ^ { \star } \} \leq d ( \widehat { \theta } _ { i , t } , \theta _ { m ^ { \star } } ) ^ { 2 } \leq D _ { \Theta } ^ { 2 } \mathbf { 1 } \{ \widehat { m } _ { i , t } \neq m ^ { \star } \} .
$$

Taking expectations proves (10). Multiplying the finite-time MAP upper bound in Theorem 1 by $D _ { \Theta } ^ { 2 }$ gives the stated squared-metric upper bound. Under the stated assumption $d _ { \operatorname* { m i n } } > 0$ , both sandwich constants are fixed and positive, so multiplication by them does not change lim in $\boldsymbol { \mathrm { I } } _ { t \to \infty } - t ^ { - 1 } \log ( \cdot )$ . Hence MAP error and squared-metric localization loss have the same exponent. If $d _ { \operatorname* { m i n } } = 0$ , only the upper-bound implication is asserted. □

## A.10 Proof of Proposition 1

Let $k _ { \star } = \lfloor M / 2 \rfloor$ and let every agent repeat the single threshold $q _ { k _ { \star } }$ . Any two hypotheses on the same side of the cut generate the same noiseless query bit at every round and therefore induce exactly the same complete transcript distribution. Because $M \geq 4$ , at least one side contains two distinct hypotheses $m , m ^ { \prime } .$ . If $P$ denotes their common transcript law, the disjoint events $\{ \widehat { m } _ { T } = m \}$ and $\{ \widehat { m } _ { T } = m ^ { \prime } \}$ satisfy

$$
P ( \widehat { m } _ { T } = m ) + P ( \widehat { m } _ { T } = m ^ { \prime } ) \leq 1 .
$$

Hence their two error probabilities sum to at least one, so one is at least $1 / 2 ,$ proving the minimax lower bound.   
The cut separates exactly $k _ { \star } ( M - k _ { \star } )$ unordered pairs, which gives Eq. (11).

Now cycle once through $q _ { 1 } , \ldots , q _ { M - 1 }$ . For threshold k, exactly $k ( M - k )$ unordered pairs lie on opposite sides. Therefore

$$
\sum _ { k = 1 } ^ { M - 1 } k ( M - k ) = \frac { M ( M - 1 ) ( M + 1 ) } { 6 } ,
$$

and dividing by the $M - 1$ rounds in a cycle and by $\binom { M } { 2 }$ gives

$$
\phi _ { \mathrm { c y c } } ( M ) = \frac { M + 1 } { 3 ( M - 1 ) } .
$$

Every adjacent pair $( k , k + 1 )$ is separated by exactly one threshold in a complete cycle, so the worst-pair frequency is $1 / ( M - 1 )$ ; nonadjacent pairs are separated more often.

It remains to compare the average pair fractions. If M is even, $k _ { \star } = M / 2$ and

$$
\phi _ { \mathrm { o n e } } ( M ) = \frac { M } { 2 ( M - 1 ) } > \frac { M + 1 } { 3 ( M - 1 ) } = \phi _ { \mathrm { c y c } } ( M )
$$

for $M > 2$ . If M is odd, $k _ { \star } = ( M - 1 ) / 2$ and

$$
\phi _ { \mathrm { o n e } } ( M ) = \frac { M + 1 } { 2 M } > \frac { M + 1 } { 3 ( M - 1 ) } = \phi _ { \mathrm { c y c } } ( M )
$$

for $M > 3$ . Thus the strict inequality holds for every $M \geq 4$ . The balanced cyclic schedule nevertheless has positive worst-pair coverage, so Proposition 3 gives a positive certified exponent when the cycle is repeated.

## A.11 Proof of Proposition 2

Fix two distinct hypotheses $m , m ^ { \prime }$ and let $\mathsf { P } _ { m } ^ { T }$ and $\mathsf { P } _ { m ^ { \prime } } ^ { T }$ denote the laws of the complete transcript through round T. For any estimator $\widehat { m } _ { T }$ , define $A : = \{ \widehat { m } _ { T } = m \}$ . Then

$$
\begin{array} { r } { { \mathbb P } _ { m } ( \widehat { m } _ { T } \neq m ) + { \mathbb P } _ { m ^ { \prime } } ( \widehat { m } _ { T } \neq m ^ { \prime } ) \geq { \mathbb P } _ { m } ^ { T } ( A ^ { c } ) + { \mathbb P } _ { m ^ { \prime } } ^ { T } ( A ) . } \end{array}
$$

The Bretagnolle–Huber inequality (Bretagnolle and Huber, 1979) gives

$$
\mathsf { P } _ { m } ^ { T } ( A ^ { c } ) + \mathsf { P } _ { m ^ { \prime } } ^ { T } ( A ) \geq \frac { 1 } { 2 } \exp \left[ { - D _ { \mathrm { K L } } \big ( \mathsf { P } _ { m } ^ { T } \| \mathsf { P } _ { m ^ { \prime } } ^ { T } \big ) } \right] .
$$

Therefore at least one of the two hypothesis-specific error probabilities is at least one quarter of the exponential term:

$$
\operatorname* { m a x } \{ \mathbb { P } _ { m } ( \widehat { m } _ { T } \neq m ) , \mathbb { P } _ { m ^ { \prime } } ( \widehat { m } _ { T } \neq m ^ { \prime } ) \} \geq \frac { 1 } { 4 } e ^ { - D _ { \mathrm { K L } } ( \mathsf { P } _ { m } ^ { T } \| \mathsf { P } _ { m ^ { \prime } } ^ { T } ) } .\tag{28}
$$

It remains to identify the transcript divergence under an adaptive policy. Let $p _ { j , t } ( \cdot \mid m , \mathcal { F } _ { t } )$ ) denote agent $j ^ { \prime } { : }$ s conditional observation law under hypothesis m after the history-dependent action at round t has been selected. Apply the KL chain rule sequentially to policy randomization, selected actions, observations, and communicated messages. Conditional on a realized history, the policy uses the same action-selection kernel under m and $m ^ { \prime }$ , so the action-selection term contributes zero KL divergence. Likewise, communicated messages and belief updates are generated by hypothesis-independent kernels once the realized history is fixed. The only nonzero contribution therefore comes from the observation kernels. Conditional independence across agents gives

$$
D _ { \mathrm { K L } } ( \mathsf { P } _ { m } ^ { T } | | \mathsf { P } _ { m ^ { \prime } } ^ { T } ) = \mathbb { E } _ { m } \left[ \sum _ { t = 0 } ^ { T - 1 } \sum _ { j = 1 } ^ { N } D _ { \mathrm { K L } } ( p _ { j , t } ( \cdot \mid m , \mathcal { F } _ { t } ) | | p _ { j , t } ( \cdot \mid m ^ { \prime } , \mathcal { F } _ { t } ) ) \right] .\tag{29}
$$

For the BSC query model, a nonseparating query induces identical output laws and hence zero KL divergence. A separating query compares Bernoulli $( 1 - \epsilon _ { j } )$ with Bernoul $\mathrm { i } ( \epsilon _ { j } )$ up to label exchange, giving

$$
D _ { \mathrm { K L } } = ( 1 - \epsilon _ { j } ) \log \frac { 1 - \epsilon _ { j } } { \epsilon _ { j } } + \epsilon _ { j } \log \frac { \epsilon _ { j } } { 1 - \epsilon _ { j } } = ( 1 - 2 \epsilon _ { j } ) \log \frac { 1 - \epsilon _ { j } } { \epsilon _ { j } } = : d _ { j } .
$$

This proves (14). Finally, since (28) holds for every pair, choose the pair minimizing the transcript KL divergence to obtain (13). If this minimum is $o ( T )$ , dividing the negative log of the lower bound by $T$ shows that the uniform error probability cannot have a positive exponential decay rate. □

## A.12 Proof of Proposition 3

Fix two distinct hypotheses $m , m ^ { \prime }$ . By definition of $d _ { \mathcal { Q } }$ in (15), one complete cycle through $\mathcal { Q } _ { \mathrm { c e r t } }$ contains at least $d _ { \mathcal { Q } }$ queries that separate this pair. On every synchronized certified round using such a query, all agents separate the pair, so the pairwise information increases by

$$
\sum _ { j = 1 } ^ { N } \mathsf { J } _ { j } ( s v _ { j } ) = W ( s ) .
$$

After $C _ { t }$ complete cycles,

$$
{ \mathcal { T } } _ { t } ( m , m ^ { \prime } ; s ) \geq C _ { t } d _ { \mathcal { Q } } W ( s ) .
$$

In particular, for the true target $m ^ { \star }$ and every false $m$

$$
\begin{array} { r } { \mathcal { T } _ { t } ( m ; s ) \geq C _ { t } d _ { \mathcal { Q } } W ( s ) . } \end{array}
$$

All noncertified rounds contribute additional nonnegative information, regardless of the adaptive base policy, so discarding them preserves the lower bound.

Set $a _ { m , t } = C _ { t } d _ { \mathcal { Q } } W ( s )$ in Theorem 1. The information-shortfall probability is then zero, and

$$
\begin{array} { r } { \mathbb { P } _ { m ^ { \star } } \left( \widehat { m } _ { i , t } \neq m ^ { \star } \right) \leq K _ { \pi _ { 0 } } ( m ^ { \star } , s ) \exp ( s \Delta _ { i } - C _ { t } d _ { \mathcal { Q } } W ( s ) ) , } \end{array}
$$

which is (16). Corollary 2 gives the corresponding squared-metric localization-loss bound.

If $C _ { t } \geq \eta t / L - c _ { 0 }$ , then

$$
\mathbb { P } _ { m ^ { \star } } \left( \widehat { m } _ { i , t } \neq m ^ { \star } \right) \leq K _ { \pi _ { 0 } } ( m ^ { \star } , s ) \exp ( s \Delta _ { i } + c _ { 0 } d _ { \mathcal { Q } } W ( s ) ) \exp \left( - \frac { \eta d _ { \mathcal { Q } } W ( s ) } { L } t \right) .
$$

The first factor is independent of $t ,$ so

$$
E _ { i } \ge \frac { \eta d _ { \mathcal { Q } } W ( s ) } { L } .
$$

Optimizing over $s \in S$ proves (17); equality of the MAP and squared-metric localization-loss exponents follows from Corollary 2. □

## A.13 Max–min threshold exploration

Proposition 4 (Max–min threshold exploration). For H synchronized certified ordered-threshold rounds with threshold counts $n _ { k } ( H )$

$$
\operatorname* { m i n } _ { k } n _ { k } ( H ) W ( s ) \leq \frac { H } { M - 1 } W ( s ) ,
$$

and a balanced cycle attains this rate asymptotically.

Proof. Fix H synchronized certified threshold rounds. For each boundary $k \in \{ 1 , \ldots , M - 1 \}$ , let $n _ { k } ( H )$ denote how many of those rounds use threshold $q _ { k }$ . Exactly one threshold separates the adjacent pair $( k , k + 1 )$ : if the cut lies below k, both hypotheses are on the same side; if it lies above k, they are again on the same side; and the cut k places k and $k + 1$ on opposite sides. Therefore the certified information supplied to adjacent pair $( k , k + 1 )$ is exactly

$$
n _ { k } ( H ) W ( s ) .
$$

Every certified round uses exactly one threshold, so

$$
\sum _ { k = 1 } ^ { M - 1 } n _ { k } ( H ) = H .
$$

The minimum of $M - 1$ nonnegative numbers cannot exceed their average. Hence

$$
\operatorname* { m i n } _ { k } n _ { k } ( H ) \leq { \frac { 1 } { M - 1 } } \sum _ { k = 1 } ^ { M - 1 } n _ { k } ( H ) = { \frac { H } { M - 1 } } .
$$

Multiplying by $W ( s )$ proves the claimed upper bound.

Now consider a balanced cyclic schedule. After H rounds, each threshold has been used either $\lfloor H / ( M - 1 ) \rfloor$ or $\lceil H / ( M - 1 ) \rceil$ times. Thus

$$
\frac { 1 } { H } \operatorname* { m i n } _ { k } n _ { k } ( H ) W ( s ) \longrightarrow \frac { W ( s ) } { M - 1 } \qquad \mathrm { a s } ~ H \to \infty .
$$

This matches the upper bound, proving max–min optimality of the exploration-only certified rate within the synchronized ordered-threshold class. The result makes no claim about the extra information generated on adaptive noncertified rounds. □

## B QUERY CODING AND SEPARATING FAMILIES

The main reliability theory permits any binary subset query $q : [ M ]  \{ 0 , 1 \}$ , and Proposition 3 expresses certified exploration through an arbitrary separating family. This appendix gives the corresponding coding interpretation and clarifies how the 1D and 2D schedules relate to unrestricted binary queries.

Lemma 1 (Shortest unrestricted separating family). A family of L unrestricted binary queries can separate all M hypotheses only $i f L \geq \lceil \log _ { 2 } M \rceil$ . This lower bound is achievable.

Proof. Associate each hypothesis m with the binary response vector

$$
c ( m ) : = ( q _ { 1 } ( m ) , \ldots , q _ { L } ( m ) ) \in \{ 0 , 1 \} ^ { L } .
$$

Two hypotheses are separated by at least one query if and only if their response vectors difer. Hence a separating family requires M distinct vectors in a set containing only $2 ^ { \check { L } }$ vectors. Therefore $M \leq 2 ^ { L }$ and $L \geq \lceil \log _ { 2 } M \rceil$

Conversely, let $L = \lceil \log _ { 2 } M \rceil$ and assign each hypothesis a distinct binary string of length L. For coordinate $\ell ,$ define $q _ { \ell } ( m )$ to be the ℓth bit of the string assigned to m. Distinct hypotheses difer in at least one coordinate, so this family separates every pair. □

The minimum Hamming distance of the response vectors is exactly $d _ { \mathcal { Q } }$ in (15), so Proposition 3 can be read as a response-code guarantee: longer minimum code distance gives more certified pairwise information per cycle. The ordered 1D threshold code has length $M - 1$ and minimum distance one. For the $S \times S$ spatial experiment, the axial code formed by $S - 1$ vertical and $S - 1$ horizontal cuts has length $2 ( S - 1 )$ and minimum distance one. These concrete geometric libraries are not cardinality-optimal among unrestricted subset queries; they are chosen because they correspond to interpretable sensing actions.

## C STATIONARY-IMBALANCE COMPARISON

Define $\begin{array} { r } { W _ { \mathrm { s t d } } ^ { * } = \operatorname* { s u p } _ { s : s v _ { i } \leq 1 } \sum _ { j } \mathsf { J } _ { j } ( s v _ { j } ) } \end{array}$ . Under inverse-stationary innovation weights $\beta _ { j } = 1 / v _ { j }$ (Huang and Wang, 2025), define $\begin{array} { r } { W _ { \mathrm { c o r r } } ^ { * } = \bar { \mathrm { s u p } } _ { 0 < s \leq 1 } \sum _ { j } \mathsf { J } _ { j } ( s ) } \end{array}$ . Then $\begin{array} { r } { W _ { \mathrm { s t d } } ^ { * } \leq W _ { \mathrm { c o r r } } ^ { * } = \sum _ { j } \mathsf { J } _ { j } ( 1 / 2 ) } \end{array}$ , with equality if and only if $v _ { j } = 1 / N$ for all j. This compares the leading certified per-cycle information; the correction may still enlarge the finite-time mixing prefactor.

Proof. For $0 < \epsilon _ { j } < 1 / 2$ , the BSC Chernof function $\mathsf { J } _ { j } ( \alpha )$ is symmetric around $1 / 2$ and uniquely maximized at $\alpha = 1 / 2$ . Therefore, for every admissible s,

$$
\sum _ { j } \mathsf { J } _ { j } ( s v _ { j } ) \leq \sum _ { j } \mathsf { J } _ { j } ( 1 / 2 ) ,\tag{30}
$$

so $W _ { \mathrm { s t d } } ^ { * } \leq \sum _ { i } { \mathsf { J } } _ { j } ( 1 / 2 )$ . Under $\beta _ { j } = 1 / v _ { j }$ , the admissibility condition becomes $0 < s \le 1$ and each argument $\begin{array} { r } { s v _ { j } \beta _ { j } = s , } \end{array}$ hence

$$
W _ { \mathrm { c o r r } } ^ { * } = \operatorname* { s u p } _ { 0 < s \leq 1 } \sum _ { j } \mathsf { J } _ { j } ( s ) = \sum _ { j } \mathsf { J } _ { j } ( 1 / 2 ) .\tag{31}
$$

Equality for the standard rule requires $s v _ { j } = 1 / 2$ for every $j ,$ which is possible exactly when all $v _ { j }$ are equal; since they sum to one, $v _ { j } = 1 / N$ □

## D GENERAL CONTROLLED OBSERVATION MODELS: DERIVATION

This appendix gives the controlled-observation derivation summarized at the end of Section 4. The BSC structure makes the certificate especially interpretable, but the martingale proof is more general. Suppose query/action $q _ { j , t }$ induces, conditional on $\mathcal { F } _ { t }$ , an observation density or mass function $p _ { j , t } ( y \mid m , \mathcal { F } _ { t } )$ under hypothesis $m ,$ dominated by a common measure; as in the main text, we suppress $\mathcal { F } _ { t }$ in formulas when unambiguous. Define

$$
L _ { j , t } ( \boldsymbol { m } ) = \frac { p _ { j , t } ( Y _ { j , t + 1 } \mid \boldsymbol { m } ) } { p _ { j , t } ( Y _ { j , t + 1 } \mid \boldsymbol { m } ^ { \star } ) } .
$$

and, for $\alpha \in [ 0 , 1 ]$ ，

$$
\rfloor _ { j , t } ^ { m , m ^ { \star } } ( \alpha ) = - \log \int p _ { j , t } ( y \mid m ^ { \star } ) ^ { 1 - \alpha } p _ { j , t } ( y \mid m ) ^ { \alpha } d \nu ( y ) .\tag{32}
$$

Conditional on $\mathcal { F } _ { t } .$

$$
\mathbb { E } _ { m ^ { \star } } \bigl [ L _ { j , t } ( m ) ^ { \alpha } \mid \mathcal { F } _ { t } \bigr ] = e ^ { - \boldsymbol { \mathrm { J } } _ { j , t } ^ { m , m ^ { \star } } ( \alpha ) } .
$$

Hence the same fractional-moment argument applies with

$$
\mathcal { T } _ { t } ^ { \mathrm { g e n } } ( m ; s ) = \sum _ { \tau = 0 } ^ { t - 1 } \sum _ { j = 1 } ^ { N } \mathsf { J } _ { j , \tau } ^ { m , m ^ { \star } } ( s v _ { j } ) .
$$

If the truth-indexed log-likelihood ratios satisfy the uniform bound | log $L _ { j , t } ( m ) | \leq b _ { j }$ almost surely for all relevant $m , t .$ , then the same $\Delta _ { i }$ construction applies and Theorem 1, the explicit mixing bound, Corollary 1, and the finite-time squared-localization argument hold verbatim after replacing $\mathcal { T } _ { t }$ by $\mathcal { T } _ { t } ^ { \mathrm { g e n } }$

For the anytime result, fix any ordered pair m $\neq m ^ { \prime }$ and repeat the same construction with $m ^ { \prime }$ treated as the truth. Define

$$
L _ { j , t } ^ { m , m ^ { \prime } } : = \frac { p _ { j , t } ( Y _ { j , t + 1 } \mid m , \mathcal { F } _ { t } ) } { p _ { j , t } ( Y _ { j , t + 1 } \mid m ^ { \prime } , \mathcal { F } _ { t } ) } .
$$

For $\alpha \in [ 0 , 1 ]$ , let

$$
\begin{array} { r } { \int _ { j , t } ^ { m , m ^ { \prime } } ( \alpha ) : = - \log \displaystyle \int p _ { j , t } ( y \mid m ^ { \prime } , \mathcal { F } _ { t } ) ^ { 1 - \alpha } p _ { j , t } ( y \mid m , \mathcal { F } _ { t } ) ^ { \alpha } d \nu ( y ) , } \end{array}
$$

and define

$$
\mathcal { T } _ { t } ^ { \mathrm { g e n } } ( m , m ^ { \prime } ; s ) : = \sum _ { \tau < t } \sum _ { j } \ J _ { j , \tau } ^ { m , m ^ { \prime } } ( s v _ { j } ) ,\tag{33}
$$

$$
\underline { { { \mathcal Z } } } _ { t } ^ { \mathrm { g e n } } ( s ) : = \operatorname* { m i n } _ { m \neq m ^ { \prime } } \ : { \mathcal T } _ { t } ^ { \mathrm { g e n } } ( m , m ^ { \prime } ; s ) .\tag{34}
$$

Conditional on $\mathcal { F } _ { t }$

$$
\mathbb { E } _ { m ^ { \prime } } [ ( L _ { j , t } ^ { m , m ^ { \prime } } ) ^ { \alpha } \mid \mathcal { F } _ { t } ] = \exp [ - \mathrm { J } _ { j , t } ^ { m , m ^ { \prime } } ( \alpha ) ] .
$$

If | log ${ \cal L } _ { j , t } ^ { m , m ^ { \prime } } | \leq b _ { j }$ almost surely uniformly over all ordered pairs, histories, and selected actions, define the stationary geometric pairwise odds

$$
G _ { t } ^ { m , m ^ { \prime } } : = \prod _ { i = 1 } ^ { N } \left( \frac { \mu _ { i , t } ( m ) } { \mu _ { i , t } ( m ^ { \prime } ) } \right) ^ { v _ { i } } .
$$

Then the corresponding pairwise process

$$
Z _ { t } ^ { m , m ^ { \prime } } ( s ) = \left( G _ { t } ^ { m , m ^ { \prime } } \right) ^ { s } \exp \bigl ( \mathcal { T } _ { t } ^ { \mathrm { g e n } } ( m , m ^ { \prime } ; s ) \bigr )
$$

is a nonnegative martingale and the same bounded network-disagreement argument applies. Replacing $\underline { { \mathcal { T } } } _ { t } ( s )$ in the proof of Theorem 2 by $\underline { { \mathcal { T } } } _ { t } ^ { \mathrm { g e n } } ( s )$ therefore gives the same Ville/union-bound guarantee. This establishes the stated general-observation anytime extension.

For the BSC query model, (32) is zero when a query does not separate the pair and equals $\mathsf { J } _ { j } ( \alpha )$ when it does, recovering (5). This connects the certificate directly to the controlled-sensing formulation of action-dependent observation laws (Nitinawarat et al., 2013).

## E GENERALIZED INNOVATION WEIGHTS

For completeness, consider

$$
\widetilde { \mu } _ { j , t + 1 } ( m ) \propto \mu _ { j , t } ( m ) \ell _ { j , t } ( Y _ { j , t + 1 } \mid m ) ^ { \beta _ { j } } , \qquad \beta _ { j } > 0 .
$$

The ratio recursion becomes

$$
\boldsymbol { r } _ { t + 1 } = A \left( \boldsymbol { r } _ { t } + \boldsymbol { \xi } _ { t } \right) , \qquad \boldsymbol { \xi } _ { j , t } = \beta _ { j } \log L _ { j , t } .
$$

Every proof above is unchanged after replacing $\mathsf { J } _ { j } ( s v _ { j } )$ by $\mathsf { J } _ { j } ( s v _ { j } \beta _ { j } )$ , restricting s to $s v _ { j } \beta _ { j } \leq 1$ , and replacing $b _ { j }$ in $\Delta _ { i }$ by $\beta _ { j } b _ { j }$ . With $\beta _ { j } = 1 / v _ { j }$ , the leading certified information becomes $\textstyle \sum _ { j } \operatorname { J } _ { j } ( s )$ and is maximized at $s = 1 / 2$ This inverse-stationary correction was introduced for fixed-observation decentralized detection by Huang and Wang (2025); Appendix C states its consequence for the adaptive-query certified exploration floor.

## F USING THE CERTIFICATE WITH LEARNED QUERY POLICIES

The theorem can be used as a policy diagnostic without modifying a learned policy. In simulated or supervised evaluation, where the ground truth is known, log every chosen query and compute

$$
{ \widehat { \gamma } } _ { T } ( s ) = { \frac { 1 } { T } } \operatorname* { m i n } _ { m \neq m ^ { \star } } { \mathcal { I } } _ { T } ( m ; s ) .
$$

For a fixed, pre-specified s, a larger $\widehat { \gamma } _ { T } ( s )$ means that the policy has accumulated evidence more uniformly against its hardest competitor. If s is optimized after the rollout, the resulting quantity is the post-hoc diagnostic score in (18), not an anytime-valid certificate by itself. Corollary 1 shows what additional statement is needed to convert this diagnostic into an unconditional exponent guarantee: an exponentially decaying lower tail for the information process. This distinction is especially relevant for learned decentralized active-sensing policies such as those studied empirically in Szostak and Cohen (2024).

## G ADDITIONAL EXPERIMENTAL DETAILS

All experiments used CPU execution on a single 4-core workstation.

Rate fitting and plotted uncertainty. For 1D runs, let $e _ { t }$ be network-average MAP error. We first retain $2 \leq t \leq \operatorname* { m a x } \{ 1 0 , \lfloor 0 . 8 5 T \rfloor \}$ with $0 . 7 5 / ( n _ { \mathrm { t r i a l } } N ) < e _ { t } < 0 . 8 ;$ if fewer than five points remain, we fall back to positive points with $1 \leq t$ and $e _ { t } < 0 . 8$ . We fit log e<sub>t</sub> on the last contiguous retained segment and return $\mathrm { n / a }$ if fewer than three points remain. For 2D runs we retain $t \ge 4 , e _ { t } > \operatorname* { m a x } \{ 3 / ( n _ { \mathrm { t r i a l } } N ) , 1 0 ^ { - 8 } \}$ , and $e _ { t } \leq 0 . 3 0 ;$ if fewer than five points remain, we use positive points with $t \geq 4$ and $e _ { t } < 0 . 8 .$ , again requiring five points.

All squared-loss curves use the squared distance of each agent’s MAP estimate from the true target, averaged over agents within a trajectory and then over independent trajectories; they are not posterior Bayes risk. For 1D curves, pointwise intervals use Student-t critical values (normal fallback); the 2D plotting code uses mean $\pm 1 . 9 6$ standard errors. These intervals are computed on the linear scale. Log plots use small display floors only for zero means or nonpositive interval endpoints $\mathrm { { \bar { ( e . g . , 1 0 ^ { - 1 2 } } } }$ in the 1D MSE panel and $1 0 ^ { - 1 4 }$ for the 2D band), so values below the Monte Carlo resolution are not interpreted as measured errors.

Graph-geometry study. Figure 2 uses $M = 3 2 , N = 1 0$ , homogeneous BSC crossover probability $\epsilon = 0 . 2 5$ 5,000 Monte Carlo trials, 80 rounds, and exploration+bisection with one deterministic exploration round every five rounds. Fits use only positive-error points before the Monte Carlo floor.

<table><tr><td>Graph</td><td>Fitted exponent</td><td>Information score</td></tr><tr><td>Complete</td><td>0.4288</td><td>0.5380</td></tr><tr><td>Ring</td><td>0.4035</td><td>0.5295</td></tr><tr><td>Directed</td><td>0.3384</td><td>0.5230</td></tr></table>

For completeness, Appendix H repeats the graph-geometry experiment with posterior bisection, Thompson-style posterior sampling, and random thresholds. Their fitted exponent $/$ information-score pairs are: posterior bisection—complete 0.5283/0.6758, ring $\mathrm { n } / \mathrm { a } / 0$ .6648, directed 0.4194/0.6566; Thompson—complete 0.3412/0.5780, ring 0.3176/0.5726, directed 0.3146/0.5707; random—complete 0.0558/0.0417, ring 0.0541/0.0417, directed 0.0516/0.0415. $\mathrm { { ^ { * } n } / a ^ { \prime } } $ denotes a run for which the predefined positive-error pre-floor fitting rule did not return a stable exponent.

Policy comparison, Fig. 6a. The common setting is $M = 3 2 , N = 1 0$ , homogeneous $\epsilon = 0 . 2 5$ , 1,500 trials, 80 rounds, and the complete graph. For reproducibility, the Thompson-style rule samples $\widetilde { m } _ { j , t } \sim \mu _ { j , t }$ and chooses

$$
k _ { j , t } ^ { \mathrm { T S } } \in \underset { 1 \leq k \leq M - 1 } { \arg \operatorname* { m a x } } \sum _ { m = 1 } ^ { M } \mu _ { j , t } ( m ) \mathbf { 1 } \{ q _ { k } ( m ) \neq q _ { k } ( \widetilde { m } _ { j , t } ) \} ,
$$

with random tie-breaking.

<table><tr><td>Policy</td><td>Fitted exponent</td><td>Information score</td></tr><tr><td>Posterior bisection</td><td>0.4487</td><td>0.6756</td></tr><tr><td>Thompson-style sampling</td><td> $\mathrm { n / a }$ </td><td>0.5773</td></tr><tr><td>Random thresholds</td><td>0.0580</td><td>0.0413</td></tr><tr><td>Exploration+bisection</td><td>0.4418</td><td>0.5378</td></tr></table>

The Thompson curve reaches the empirical floor before the fitting rule identifies a stable pre-floor linear region, so we do not assign it a panel-(a) exponent.

Pooled bottleneck study, Fig. 3. The ten-seed study spans four policies, three communication graphs, and homogeneous/heterogeneous sensor profiles, giving 240 configurations; 210 satisfy the deterministic rate-fit rule above. The other 30 receive $\mathrm { n / a }$ because no valid pre-floor fitting segment is available; no exponent is imputed for them. Correlations and regressions are therefore explicitly conditional on the 210-configuration subset rather than the full design. All correlations and regression quantities use 10,000 seed-cluster-bootstrap replicates with whole seed blocks resampled jointly. The numerical results are reported once in Section 6.1 and annotated in Fig. 3.

Directed heterogeneous-sensor study. Figure 8 uses the deliberately imbalanced five-agent directed graph, 10,000 Monte Carlo trials, 100 rounds, and exploration+bisection. The stationary distribution is

$$
( 0 . 2 1 7 8 , 0 . 2 3 6 4 , 0 . 2 3 1 2 , 0 . 2 0 8 1 , 0 . 1 0 6 5 ) ,
$$

the BSC crossover probabilities are

and the corresponding midpoint Chernof/Bhattacharyya information values (Bhattacharyya, 1943) are

$$
( 0 . 2 2 3 1 , 0 . 0 4 7 2 , 0 . 1 1 3 1 , 0 . 4 1 3 3 , 0 . 8 3 0 4 ) .
$$

For exploration+bisection, the fitted observable-tail exponents are 0.4637 (standard update) and 0.4633 (inversestationary update), with raw post-hoc information-score values 0.5407 and 0.1590. We do not compare these posthoc score magnitudes directly because innovation reweighting changes the fractional-moment parameterization.

The additional policy variants in Appendix H give the following exponent / information-score pairs for standard versus inverse-stationary updates: posterior bisection $\left( \mathrm { n / a / 0 . 6 0 4 0 } \right)$ versus $( \mathrm { n / a / 0 . 1 5 9 4 } )$ ; Thompson 0.3770/0.4429 versus 0.5061/0.1283; random $0 . 0 4 7 8 / 0 . 0 3 8 6$ versus $0 . 0 5 0 2 / 0 . 0 4 3 5 ;$ and exploration+bisection 0.4637/0.5407 versus 0.4633/0.1590. These policy-dependent finite-horizon outcomes are why the main text treats Experiment 3 as a mechanism stress test rather than evidence that inverse-stationary weighting uniformly improves empirica convergence speed.

Worst-competitor versus average-pair proxy. For every valid pooled configuration, the simulation records both the truth-indexed worst-competitor diagnostic score $\widehat { \gamma } _ { T } ^ { \mathrm { d i a g } }$ from (18) and the all-pairs aggregate rate in (19). Both correlations are computed on the identical 210 configurations. Resampling whole seed blocks preserves the paired design and avoids treating the multiple policy–graph–sensor configurations associated with one seed as independent bootstrap units; Section 6.1 reports the resulting statistics.

Continuous-observation study. Conditional on query bit $b \in \{ 0 , 1 \}$ , the external-validity experiment uses

$$
p _ { j } ( y \mid b ) = \frac { \kappa _ { j } } { 2 \sinh \kappa _ { j } } \exp \bigl ( ( 2 b - 1 ) \kappa _ { j } y \bigr ) , \qquad - 1 \leq y \leq 1 ,\tag{35}
$$

whose log-likelihood ratio obeys $| \log [ p _ { j } ( y \mid 1 ) / p _ { j } ( y \mid 0 ) ] | \le 2 \kappa _ { j }$ . For a separating query,

$$
\mathrm { J } _ { j } ^ { \mathrm { c o n t } } ( \alpha ) = - \log \left[ \frac { \sinh ( ( 2 \alpha - 1 ) \kappa _ { j } ) } { ( 2 \alpha - 1 ) \sinh \kappa _ { j } } \right] ,\tag{36}
$$

with continuous limit $\mathrm { J } _ { j } ^ { \mathrm { c o n t } } ( 1 / 2 ) = \log ( \sinh \kappa _ { j } / \kappa _ { j } )$ . We choose $\kappa _ { j }$ numerically so that

$$
\begin{array} { r } { \mathrm { J } _ { j } ^ { \mathrm { c o n t } } ( 1 / 2 ) = - \log ( 2 \sqrt { \epsilon _ { j } ( 1 - \epsilon _ { j } ) } ) , } \end{array}\tag{37}
$$

matching the midpoint information of the reference BSC. The study uses $M = 3 2$ $N = 1 0$ , 5,000 trials, 80 rounds, exploration+bisection with one deterministic exploration round every five rounds, and the same complete/ring/directed graph constructions as Fig. 2. The simulator uses the exact likelihood in the Bayes update and the exact increment in (36). The fitted exponent $/$ worst-score pairs are complete 0.4248/0.5372, ring $\mathrm { n / a / 0 . 5 2 8 5 }$ , and directed $\mathrm { n / a / 0 . 5 2 2 1 }$ ; the latter two reach zero Monte Carlo error before the predefined positive-error fitting rule identifies a stable pre-floor linear region.

Structured 2D spatial-query experiment. We use an $8 \times 8$ grid $( M = 6 4 )$ , $N = 1 0$ , 100 rounds, and a 42-query library of vertical, horizontal, and $\pm 4 5 ^ { \circ }$ half-planes. The certified family comprises the 14 axial cuts. On the complete graph with homogeneous $\epsilon = 0 . 2 5$ and 1,500 trials per policy, the exponent $/$ worst-score $/$ averagepair-score triples are posterior bisection $0 . 7 5 7 1 / 0 . 5 5 1 0 / 0 . 5 2 4 5$ , Thompson-style sampling 0.2953/0.4767/0.4857, random spatial queries 0.0914/0.0928/0.4393, and exploration+bisection 0.4660/0.4673/0.5275. All final empirical MAP errors are zero at round 100.

The pooled 2D bottleneck study is run with 500 trials per configuration over four policies, three graphs, homogeneous/heterogeneous sensor profiles, and ten seed blocks (240 configurations); the heterogeneous profile spans BSC crossover probabilities 0.13–0.35. All 240 configurations yield valid pre-floor fits. With 10,000 paired seed-cluster-bootstrap replicates, $r _ { \mathrm { w o r s t } } = 0 . 7 6 8$ (95% CI [0.761, 0.777]), $r _ { \mathrm { a v g } } = 0$ .484 (95% CI [0.474, 0.494]), $\Delta r = 0 . 2 8 5$ (95% CI [0.270, 0.299]), and the worst-score regression slope is 0.773 (95% CI [0.736, 0.811]).

Tail fitting. The exact 1D/2D fitting rules are stated at the start of Appendix $\mathrm { G } ;$ zero-error rounds are never fitted. Resolving deeper tails would require more independent trials or a dedicated rare-event estimator, so display-floor values are not treated as observations.

![](images/0b8ca8b23bb29259874fe368920c3b0192882fc88489d77b471bfe51ec3d5ca3.jpg)  
(a) Query-policy comparison on the common 1D problem.

Bounded continuous observations: exponential localization (95% CI)  
![](images/8171f42e50186d08ea6f6fddff6bbf133b396120a06df55dc648b1688de8a102.jpg)  
(b) Information-matched bounded continuous sensor.

Figure 6: Additional theory-facing checks. Left: four query policies on the common 1D problem. Right: the bounded continuous-observation model, which exercises the general controlled-observation construction.

![](images/8db0b9872f0e492e8b932c518eeddc58ce22223613cad92d31b929cfc7a83aa5.jpg)  
(a) Network-average MAP error.

2D spatial queries: squared localization error (95% CI)  
![](images/614e1f420936dd0391052af695e5ec9ee39706a35d7879f1c1a066ffc9c56be4.jpg)  
(b) Squared Euclidean localization loss.  
Figure 7: Structured 2D localization curves. Four query policies on the $8 \times 8$ spatial grid; shading denotes pointwise 95% normal-approximation confidence intervals over independent trajectories.

## H ADDITIONAL QUERY-POLICY EXPERIMENTS

This appendix collects companion localization-loss plots, additional query-policy curves, the bounded continuousobservation study, and directed-influence robustness checks. The results are mechanism checks rather than claims of policy optimality.

![](images/714aa733e0f8d5efee41d4fc6fbae030456e48d6dd6d8a48f94e3e171a488124.jpg)  
Figure 8: Directed stationary-influence stress test. Sensor information is deliberately anti-aligned with stationary network influence, exposing the coupling represented by $\mathsf { J } _ { j } ( s v _ { j } )$

![](images/273f2128c1e8e8b7c72eb5adbd5b3774d1f1f9d617978c69f6246871f7b96e6b.jpg)

BSC observations: exponential localization (95% CI)  
![](images/3bc28b515e09d8ba75d6088c00149d8b3dbb9b9d24134af1ba36d8e9af987ceb.jpg)  
(b) Posterior bisection: MAP error.

(a) Exploration+bisection: MSE (companion to Fig. 2).  
BSC observations: exponential localization (95% CI)  
![](images/728ae82ab1e240fbb746f912f62e816c8a6cba6e393bf8982c4e78f90e710b05.jpg)  
(c) Thompson-style sampling: MAP error.

BSC observations: exponential localization (95% CI)  
![](images/650c2d03cb5fee9368588d943593e30af1e507ba418f389c975fc674a2ef21c5.jpg)  
(d) Random thresholds: MAP error.  
Figure 9: Graph-geometry robustness and MSE companion. The MSE panel for the main exploration+bisection study and MAP-error results for the remaining query policies. Full MAP/MSE outputs are included with the released figure scripts.

Directed imbalance changes how sensor information is used (95% CI)  
![](images/3cb58da4b8e2e9d9c53d419b23694c7049fbefaa85cabd98d061d6882a24e43a.jpg)  
(a) Exploration+bisection.

Directed imbalance changes how sensor information is used (95% CI)  
![](images/745cd32ad9b4ee230e8997013d8e05a137f4fd4419eec5c2ce01c04738b76b2f.jpg)  
(b) Posterior bisection.

Directed imbalance changes how sensor information is used (95% CI)  
![](images/75a6ec56cc2e33cf9400a74a262bed617489972c79629a61897a99d2dbcf9419.jpg)  
(c) Thompson-style sampling.

Directed imbalance changes how sensor information is used (95% CI)  
![](images/7b79da42ad6e4fb5cba5bc2a8ca453569943c257b8eb4945f139f94228d11bd1.jpg)  
(d) Random thresholds.

Figure 10: Directed-network update comparisons. Standard versus inverse-stationary updating under four query policies, all using the same anti-aligned sensor-quality/stationary-influence profile shown in Fig. 8.