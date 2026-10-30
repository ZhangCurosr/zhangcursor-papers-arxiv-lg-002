# Optimal Contextual Pricing in Non-Stationary Environments

Yiding Feng<sup>∗</sup> Alireza Rezaeimoghadam<sup>†</sup>

## Abstract

We study contextual dynamic pricing in a non-stationary environment. In each round, a seller observes a buyer’s context, posts a price, and receives only binary purchase feedback. The buyer’s value is linear in the context and an unobserved feature vector that may change over time, subject to a known total variation budget V. We measure dynamic regret against the first-best benchmark that extracts each buyer’s full value.

For identical contexts, we develop a deterministic algorithm that combines binary-search exploration, lower-envelope pricing, and two complementary restart mechanisms. It achieves regret $O ( ( \sqrt { T V } + 1 ) \log T )$ . With heterogeneous contexts, a new geometric dificulty arises: small feature changes can leave a relaxed feasible set narrow but far from the current feature vector. To address this dificulty, we introduce a multiscale shallow-pricing algorithm that maintains relaxed feasible sets and ellipsoidal approximations at several variation scales. Larger scales provide reliable localization certificates, while the smallest active scale determines the exploitation price. The algorithm achieves dynamic regret $O ( ( d ^ { 2 } + d ^ { 3 / 2 } \sqrt { T V } ) \log ^ { 2 } T )$ . Both guarantees match the identical-context lower bound Ω( TV + log log T) up to logarithmic factors.

## 1 Introduction

Online sellers repeatedly make pricing decisions for buyers with diferent observable characteristics and changing willingness to pay. The economic scale of these decisions is substantial: U.S. consumers spent \$257.8 billion online during November and December 2025, including \$44.2 billion during the five-day Cyber Week (Adobe, 2026). Observable context can help a seller tailor prices to individual transactions, but the relationship between context and willingness to pay may change with seasonality, preferences, and market conditions. Learning this relationship is particularly challenging because pricing feedback is censored: a purchase or rejection reveals only whether the buyer’s value exceeds the posted price. An efective pricing algorithm must therefore learn from limited feedback while adapting to a changing environment.

We study contextual dynamic pricing over T rounds. In each round, a new buyer arrives with an observable d-dimensional context and an unobserved d-dimensional feature vector. The buyer’s value is the inner product of these two vectors. Contexts are nonnegative unit vectors, and hidden features are nonnegative vectors with Euclidean norm at most one, so values lie between zero and one. After observing the context, the seller posts a price and observes only whether the buyer purchases. Contexts may vary arbitrarily across rounds. The hidden features may also change, subject to a known upper bound V on the sum of their Euclidean changes between consecutive rounds. Our objective is to minimize dynamic regret relative to the first-best benchmark: the total revenue of an oracle that knows each buyer’s value and posts that value as the price. This benchmark extracts the full surplus in every round and is stronger than comparison with the best single fixed price in hindsight (Kleinberg and Leighton, 2003).

Our problem builds on the study of online posted-price learning initiated by Kleinberg and Leighton (2003) and connects two lines of dynamic pricing research. Stationary contextual pricing studies heterogeneous contexts with a fixed hidden parameter (Paes Leme and Schneider, 2018; Cohen et al., 2020; Liu et al., 2021), whereas Paes Leme et al. (2021) study non-contextual pricing with changing values under bounds on their per-round changes. We consider heterogeneous contexts together with changing hidden features under a known aggregate variation budget. Non-stationary contextual pricing has also been studied under stochastic valuation noise and demand models (Javanmard, 2017; Baby et al., 2023); our focus is the noiseless linear-valuation model and regret against the first-best benchmark. (A comprehensive discussion on the further related work can be found in Section 1.2.) This leads to the following question:

What is the optimal dynamic regret in this setting, and what additional dificulties arise from the interaction between contextual heterogeneity and non-stationarity?

## 1.1 Our Results and Techniques

We develop a deterministic contextual pricing algorithm with dynamic regret $O \left( ( d ^ { 2 } + d ^ { 3 / 2 } { \sqrt { T V } } ) \log ^ { 2 } T \right)$ Together with a lower bound of $\Omega ( \sqrt { T V } + \log \log T )$ , this establishes optimality up to logarithmic factors in T for fixed d among deterministic algorithms. The guarantee holds for every variation budget $V \in [ 0 , T ]$ , including the stationary case. Our algorithm combines periodic and inconsistency restarts with a multiscale extension of shallow pricing.

We first develop the restart mechanism in the identical-context environment (Section 3). In this setting, binary search followed by lower-envelope pricing provides a simple exploration-thenexploitation procedure. Non-stationarity creates two distinct dificulties: an increase in values can leave an accepted price persistently too low, whereas even a small decrease can turn a previously accepted price into a rejection and cause a large revenue loss. We address these dificulties through periodic restarts and immediate restarts after rejected exploitation prices. A price margin controls the frequency of the latter by ensuring that each such rejection requires suficient variation. This yields dynamic regret $O ( ( \sqrt { T V } + 1 ) \log T )$

Extending this argument to heterogeneous contexts requires additional geometric reasoning. A narrow interval directly localizes a scalar value, but localization in one context direction need not provide useful information in another. We therefore build on the shallow-pricing procedure of Cohen et al. (2020), which uses ellipsoidal approximations to guide exploration. To accommodate non-stationarity, we relax the halfspace constraints induced by purchase feedback. However, using a single relaxation scale introduces a new dificulty: a relaxed feasible set can become narrow while excluding the current hidden parameter. Nearly parallel context directions can amplify small parameter changes into a large localization error. Our Example 4.1 shows why the identical-context exploitation argument does not extend directly.

Our main algorithm resolves this dificulty by maintaining relaxed feasible sets and enclosing ellipsoids at multiple dyadic scales (Section 4). It explores whenever an active scale indicates substantial uncertainty and otherwise posts the lower-envelope price of the smallest active feasible set. The key observation is that a scale at least as large as the within-batch parameter diameter retains every hidden parameter in that batch. Its ellipsoid therefore provides a reliable certificate of localization. Nesting of the feasible sets then controls the regret of accepted exploitation prices, even when the smallest active set excludes the current parameter. Rejected exploitation prices trigger immediate restarts. Combining this multiscale argument with geometric bounds on exploration and variation-based bounds on the number of restarts yields our dynamic regret guarantee.

## 1.2 Further Related Work

We discuss additional connections to online pricing, contextual search, and learning in non-stationary environments. The main distinctions concern the valuation model, the measure of non-stationarity, the information available to the seller, and the revenue benchmark.

Online posted pricing and demand learning. Kleinberg and Leighton (2003) study online posted-price auctions under identical, random, and worst-case valuations. For an unknown identical value, they establish the optimal Θ(log log T) regret. In this special case, the best fixed price and the first-best benchmark coincide. For arbitrary valuation sequences, their comparison is instead with the best fixed price in hindsight. A complementary operations research literature studies the exploration–exploitation tradeof in learning an unknown demand function. For example, Besbes and Zeevi (2009) develop pricing policies and regret bounds relative to a seller who knows the demand function, in a setting with finite inventory. Our model has no intertemporal inventory constraint and compares revenue with the sum of buyers’ realized values. Thus, diferences in regret rates across these settings reflect diferences in both information and benchmarks.

Stationary contextual pricing and geometric search. Cohen et al. (2020) study featurebased pricing with an unknown linear valuation model and develop ellipsoid-based algorithms with regret polynomial in d and logarithmic in T. They also introduce the robust ShallowPricing procedure, which relaxes feedback constraints to accommodate small valuation noise. Our algorithms use this geometric primitive. Related geometric methods appear in multidimensional binary search (Lobel et al., 2018) and contextual search via intrinsic volumes (Paes Leme and Schneider,

2018). Liu et al. (2021) obtain an O(d log log T + d log d) regret bound for stationary contextual pricing under their normalization and develop extensions to more general contextual search problems. These works exploit a fixed underlying parameter or hypothesis. In our setting, feedback constraints collected at diferent times may correspond to diferent hidden parameters. Consequently, shrinking an uncertainty set need not localize the current parameter. Extension models have been studied in Krishnamurthy et al. (2021); Lykouris et al. (2025); Gaucher et al. (2025); Feng et al. (2026a,b); Cosson et al. (2026). Our contribution is to control this loss of consistency while retaining the exploration guarantees of geometric search. Multiscale methods themselves also appear in prior contextual search algorithms, including Liu et al. (2021); here, the scales represent diferent tolerances for parameter drift.

Contextual pricing with stochastic valuation noise. A substantial literature assumes that valuations consist of a linear contextual component and an additional random shock. Javanmard and Nazerzadeh (2019) study high-dimensional pricing with sparse parameters and develop regularized likelihood methods. Xu and Wang (2021) establish logarithmic regret guarantees with a known noise distribution, including for adversarial contexts. When the noise distribution is unknown, Xu and Wang (2022) distinguish learning a linear pricing policy from learning a linear valuation model with agnostic noise, while Luo et al. (2022) combine parameter exploration with a UCB procedure for learning the noise distribution and maximizing revenue. More recently, Xu and Wang (2026) study unknown noise distributions that induce discontinuous demand, allowing jumps and atoms under stochastic, well-conditioned contexts. These models introduce statistical uncertainty beyond the unknown linear parameter. Their distribution-aware oracle generally knows the demand model but does not observe each buyer’s realized noise before pricing. Our noiseless model instead permits an oracle that knows the current parameter and context to extract the buyer’s full value. Accordingly, the associated regret guarantees address diferent benchmarks and cannot be compared solely by their dependence on T.

Non-contextual pricing against a moving value. The closest predecessor to our identicalcontext setting is Paes Leme et al. (2021), who study changing scalar values with binary purchase feedback and the same first-best benchmark. They analyze adversarial evolution under pointwise bounds on consecutive value changes, as well as stochastic evolution. Their adversarial revenue guarantees cover a fixed changing rate and unknown, non-increasing sequences of changing-rate bounds. The monotonicity condition concerns these bounds, not the values themselves. Their algorithms combine localization with randomized checking rounds to detect discrepancies between the current value and the maintained interval. Our formulation assumes a known aggregate variation budget V and permits this variation to be distributed arbitrarily over time. Our identical-context algorithm uses periodic restarts and deterministic inconsistency restarts, together with a price margin that controls their frequency. It achieves $O ( ( \sqrt { T V } + 1 ) \log T )$ regret and supplies the restart mechanism used in our contextual algorithm. The two formulations also difer in prior information: knowing the aggregate budget is an assumption of our algorithm, whereas their unknown-rate algorithms adapt without this information.

Non-stationary pricing with stochastic demand. Non-stationary demand learning has also been studied beyond the moving-value model. Keskin and Zeevi (2017) analyze pricing with an unknown changing demand model and characterize regret under variation constraints. In the contextual setting, Javanmard (2017) studies a linear valuation model with random shocks and time-varying coeficients, using projected stochastic gradient methods. Their guarantees cover both adversarial and stochastic context sequences and explicitly account for changes in the coeficients. Baby et al. (2023) study non-stationary contextual pricing with safety constraints. With known strictly log-concave market noise, they derive dynamic regret guarantees through proper online learning with exp-concave losses and adapt to an unknown parameter variation budget. Their benchmark is the sequence of optimal expected-revenue prices under the current demand model. These papers already address contextual heterogeneity together with non-stationarity. Our focus is the noiseless linear-valuation model, an aggregate Euclidean variation budget, and regret against full surplus, without assuming the smooth demand probabilities used in their statistical analyses.

Change detection provides another approach to non-stationary pricing. Zhao et al. (2026) develop a policy based on online change-point detection for high-dimensional generalized linear demand models, with guarantees that account for sparsity and the number of changes. Jiang and Zhao (2026) develop multiscale change-point detection that adapts to both piecewise stationarity and a design-adjusted parameter variation budget, without knowing these measures in advance. Their model uses stochastic contexts and generalized linear demand, and their oracle maximizes expected revenue. Their scales organize statistical detection over time. In our algorithm, scales index simultaneous relaxations of the same geometric feedback constraints and determine whether localization is reliable enough for exploitation. This distinction also clarifies that our contribution is not the general use of restarts or multiple scales in non-stationary pricing.

Non-stationary bandits and restart methods. Besbes et al. (2014) study stochastic multiarmed bandits under a variation budget on mean rewards and establish near-optimal dynamic regret through periodic restarts of Exp3. Cheung et al. (2019) develop sliding-window methods and a bandit-over-bandit approach for non-stationary linear bandits. Chen et al. (2019) give an adaptive contextual bandit algorithm that uses replay phases to detect changes without prior knowledge of the amount of non-stationarity. These works motivate forgetting outdated observations and combining exploration with change detection. However, their variation measures and reward structures difer from ours. Even with a fixed context, an arbitrarily small decrease in the hidden parameter can change a purchase into a rejection at a fixed positive price, causing a constant revenue drop. Thus, a small parameter variation budget does not imply a comparably small uniform variation budget for revenue functions. Generic reward-variation guarantees therefore do not directly yield our regret bound. Our analysis instead uses the geometry of purchase feedback and the asymmetric consequences of underpricing and overpricing.

Dynamic regret and online convex optimization. Dynamic regret against a changing comparator is a classical objective in online optimization. Zinkevich (2003) analyzes online convex optimization with moving comparators, and Besbes et al. (2015) study stochastic optimization under variation budgets on the objective functions. Baby and Wang (2021) establish optimal dynamic regret guarantees for exp-concave online learning when improper predictions are allowed. The pricing results of Baby et al. (2023) connect this literature to constrained estimation and stochastic demand. Our revenue loss is discontinuous and is not a convex function of the posted price, so these guarantees do not apply directly. Nevertheless, the distinction between comparator movement, objective variation, and parameter variation is relevant here: our budget constrains hidden features, while contexts may vary arbitrarily. In particular, even when V = 0, both buyer values and first-best prices may change across rounds.

Corruption-robust contextual search. Krishnamurthy et al. (2023) study contextual search when some responses are adversarially corrupted and design algorithms whose performance degrades with the number of corruptions. Paes Leme et al. (2022) develop density-based methods for adversarially noisy contextual search, obtaining guarantees for symmetric and tolerance-based losses. These works share the concern that intersecting all observed constraints can eliminate the correct hypothesis. The source of inconsistency difers in our model: feedback is accurate for the current buyer, but the hidden parameter itself moves. A single persistent parameter shift may make many subsequent observations inconsistent with an earlier parameter while contributing only once to our variation budget. Moreover, a small symmetric prediction error does not ensure small pricing regret, because even slight overpricing can eliminate the entire revenue from a round. Thus, corruption budgets and search-loss guarantees require additional arguments before they can be used for our objective.

## 2 Preliminaries

We introduce the contextual dynamic pricing problem in a non-stationary environment.

Environment. There is a single seller who interacts with buyers over T discrete rounds. In each round $t \in [ T ] \ \triangleq \ \{ 1 , 2 , \dots , T \}$ , a new buyer, indexed by t, arrives. Each buyer t has a public d-dimensional context vector $u _ { t } \in [ 0 , 1 ] ^ { d }$ with $^ { 1 } \parallel u _ { t } \parallel = 1$ and a private d-dimensional feature vector $\theta _ { t } \geq 0 , \| \theta _ { t } \| \leq 1$ . After observing the public context $u _ { t }$ , the seller posts a take-it-or-leave-it price $p _ { t } \in \ [ 0 , 1 ]$ for a single product. Buyer $t \mathrm { { } ^ { \circ } s }$ value $( { \mathrm { i . e . } }$ , willingness to $\mathrm { p a y } )$ takes the linear form $v _ { t } \triangleq \langle \theta _ { t } , u _ { t } \rangle \in [ 0 , 1 ]$ . Thus, buyer t purchases the product at price $p _ { t }$ if and only if her value is at least the price, $\mathrm { i . e . , } \ v _ { t } \geq p _ { t }$ . The seller observes only whether the buyer purchases the product and does not observe her value. (We denote the purchase feedback by $f _ { t } \triangleq \mathbb { I } \{ v _ { t } \geq p _ { t } \} \in \{ 0 , 1 \} . )$ The seller’s goal is to maximize her revenue, i.e., the total payments from all $T$ buyers. We assume $T \geq 2$ . No stochastic assumptions are imposed on the context or feature sequences.

Dynamic regret minimization. We evaluate an online learning algorithm by its regret relative to the first-best benchmark (i.e., full surplus), $\begin{array} { r } { \sum _ { t \in \left[ T \right] } v _ { t } = \sum _ { t \in \left[ T \right] } \left. \theta _ { t } , u _ { t } \right. } \end{array}$ . This benchmark is achieved by posting the full-surplus-extraction price $\boldsymbol { p } _ { t } ^ { * } = \boldsymbol { v } _ { t }$ for each buyer t. Specifically, given a sequence of T buyers described by $\{ u _ { t } , \theta _ { t } \} _ { t \in [ T ] }$ , the dynamic regret of an algorithm Alg is

$$
\mathrm { R E G R E T } [ \mathrm { A L G } ] \triangleq \sum _ { t \in [ T ] } \langle \theta _ { t } , u _ { t } \rangle - \sum _ { t \in [ T ] } p _ { t } \cdot \mathbb { I } \{ \langle \theta _ { t } , u _ { t } \rangle \geq p _ { t } \} ,
$$

where $\{ p _ { t } \} _ { t \in [ T ] }$ are the prices posted by Alg.

Non-stationarity upper bound. We measure the non-stationarity of the unknown feature sequence $\{ \theta _ { t } \} _ { t \in [ T ] } = ( \theta _ { t } ) _ { t \in [ T ] }$ by its total variation, $\begin{array} { r } { \mathrm { T V } ( \{ \theta _ { t } \} _ { t \in [ T ] } ) \stackrel { \triangle } { = } \sum _ { t \in [ T - 1 ] } \| \theta _ { t + 1 } - \theta _ { t } \| } \end{array}$ . Although the feature sequence $\{ \theta _ { t } \} _ { t \in [ T ] }$ remains unobserved by the seller throughout the $T$ rounds, we assume that the seller knows an upper bound V on its total variation in advance, i.e., $\mathrm { T V } \big ( \{ \theta _ { t } \} _ { t \in [ T ] } \big ) \leq V$ We focus on $V \in [ 0 , T ]$ ; larger variation budgets admit the trivial regret guarantee $T .$ . In the special case of zero non-stationarity, i.e., $V = 0$ , our problem reduces to the classical stationary contextual dynamic pricing problem studied in Kleinberg and Leighton (2003); Paes Leme and Schneider (2018); Cohen et al. (2020); Liu et al. (2021).

## 3 Warm-Up: Optimal Dynamic Pricing for Identical Contexts

We begin with the identical-context environment, where all buyers have the same public context, denoted by u. This section serves as a warm-up for the more general environment with heterogeneous contexts: its algorithmic ingredients and regret analysis closely parallel, and lay the groundwork for, the design and analysis of our main algorithm in Section 4.

Although the public contexts are identical, the private features $\{ \theta _ { t } \} _ { t \in [ T ] }$ may still vary across buyers. We can therefore view this setting as a pricing problem without public contexts, in which buyers have potentially diferent private values $\{ v _ { t } \} _ { t \in [ T ] }$ . For ease of presentation, we omit $\{ u _ { t } , \theta _ { t } \} _ { t \in [ T ] }$ in this section and work directly with the value sequence $\{ v _ { t } \} _ { t \in [ T ] }$ . Under the normalization in our model, $v _ { t } ~ \in ~ [ 0 , 1 ]$ for every buyer t. Moreover, since identical context u is unit-length $( \mathrm { i . e . , ~ } \| u \| = 1 )$ , we have $\begin{array} { r } { \sum _ { t \in [ T - 1 ] } | v _ { t + 1 } - v _ { t } | \leq \sum _ { t \in [ T - 1 ] } \| \theta _ { t + 1 } - \theta _ { t } \| \leq V } \end{array}$ . We consider non-stationarity upper bounds $V \in [ 0 , T ]$

Below, we present the algorithm for this environment and its dynamic regret guarantee. We then conclude the section with a sketch of the regret analysis.

Algorithm overview. For stationary online learning problems, a standard approach is explorethen-exploit (ETE), also commonly called explore-then-commit in the bandit literature (Lattimore and Szepesvári, 2020, Chapter 6). When all buyers have the same value, this approach can be implemented by first conducting binary search for $\lceil \log _ { 2 } T \rceil$ rounds to locate the value within an interval of width at most $1 / T$ . This exploration phase incurs at most $\lceil \log _ { 2 } T \rceil$ regret. During the subsequent exploitation phase, posting the lower endpoint of this interval in every remaining round incurs at most O(1) additional regret.<sup>2</sup>

For non-stationary environments, a standard approach is to restart a learning algorithm periodically. For example, in non-stationary multi-armed bandit problems, Besbes et al. (2014) show that periodically restarting Exp3 achieves near-optimal dynamic regret under a variation-budget constraint. However, the revenue structure of dynamic pricing creates an additional dificulty. An arbitrarily small decrease in a buyer’s value can reduce the revenue from a previously optimal price to zero: if the seller continues posting $p _ { t + 1 } = v _ { t }$ but $v _ { t + 1 } < v _ { t }$ , the buyer rejects the ofer. Thus, even an arbitrarily small amount of variation can cause constant regret in every subsequent round until the next periodic restart, provided the buyer’s value remains bounded away from zero.<sup>3</sup>

Motivated by this dificulty, our algorithm supplements periodic restarting with an inconsistency restart. Informally, if a buyer rejects a price that the algorithm posts during exploitation, the algorithm immediately restarts ETE. However, inconsistency restarting alone cannot guarantee sublinear regret either. To see this, consider a sequence in which all buyers during exploration have the same value. Immediately after exploration ends, the value increases by V and remains at this higher level for the rest of the horizon, where $V > 0$ is chosen so that all values remain in [0, 1]. No inconsistency restart occurs because buyers continue accepting the exploitation price. The regret in each exploitation round is at least V, leading to total regret of Ω(VT) when exploration takes O(log T) rounds. This regret is linear in T for any fixed admissible $V > 0$

Our algorithm therefore combines periodic and inconsistency restarts. We call the consecutive rounds from initialization or a restart until the next restart a batch. Periodic restarting is controlled by a window-size parameter $a \in [ T ]$ , which specifies the maximum number of rounds in a batch. This parameter balances the regret caused by non-stationarity during exploitation against the exploration cost of frequent periodic restarts.

Algorithm 1 Pricing for the identical-context environment   
1: Input: $T \in \mathbb { N } , V \in [ 0 , T ]$   
2: Parameters: $\delta \gets \operatorname* { m a x } \{ 1 / T , \sqrt { V / T } \} , a \gets \operatorname* { m i n } \{ T , \lceil \sqrt { T / V } \rceil \}$ ▷ set $a \gets T$ when $V = 0$   
3: t ← 1, i ← 0   
4: while $t \leq T$ do   
5: $i  i + 1 , \tau $ min $\{ T , t + a _ { . } - 1 \} , [ p _ { i } ^ { \mathrm { m i n } } , p _ { i } ^ { \mathrm { m a x } } ] \gets [ 0 , 1 ]$   
6: while $t \leq \tau$ and $p _ { i } ^ { \operatorname* { m a x } } - p _ { i } ^ { \operatorname* { m i n } } > \delta$ do   
7: Post $p _ { t } \gets ( p _ { i } ^ { \operatorname* { m i n } } + p _ { i } ^ { \operatorname* { m a x } } ) / 2 ;$ observe $f _ { t }$   
8: if $f _ { t } = 1 ,$ set $p _ { i } ^ { \mathrm { m i n } } \gets p _ { t } ;$ else set $p _ { i } ^ { \operatorname* { m a x } } \gets p _ { t }$   
9: t ← t + 1   
10: end while   
11: $\underline { { p } } _ { i } \gets \operatorname* { m a x } \{ 0 , p _ { i } ^ { \mathrm { m i n } } - \delta \}$   
12: <sup>i</sup> while $t \leq \tau$ do   
13: Post $p _ { t } \gets \underline { { p } } _ { i } ;$ ; observe $f _ { t } ;$ set $t \gets t + 1$   
14: if $f _ { t - 1 } = 0 ,$ break ▷ inconsistency restart   
15: end while   
16: end while

Inconsistency restarting is controlled by a price-margin parameter $\delta \in ( 0 , 1 ]$ . This parameter determines both the target interval width during exploration and the price posted during exploitation. Specifically, once binary search produces an interval of width at most $\delta ,$ the algorithm posts its lower endpoint minus $\delta ,$ clipped at zero, during the subsequent exploitation phase. We call this price the lower-envelope price (LEP). Intuitively, δ balances the regret from conservative pricing during exploitation against the exploration cost of frequent inconsistency restarts. The formal algorithm is given in Algorithm 1.

Regret guarantee and its analysis. We now state the regret guarantee in Theorem 3.1.

Theorem 3.1. In the identical-context environment, given any non-stationarity upper bound $V \in$ $[ 0 , T ]$ , set $a =$ min $\{ T , \lceil \sqrt { T / V } \rceil \}$ and $\delta = \operatorname* { m a x } \{ 1 / T , \sqrt { V / T } \} ^ { 4 }$ . Then the dynamic regret of Algorithm 1 is at most $O ( \sqrt { T V } \log T + \log T )$ . Moreover, for every $V \in [ 0 , T ]$ and every deterministic online algorithm, there exists an admissible instance with total variation at most V on which the algorithm incurs dynamic regret of at least $\Omega ( \sqrt { T V } + \log \log T )$

Combining the upper and lower bounds in Theorem 3.1, the regret guarantee of Algorithm 1 is optimal among deterministic online algorithms, up to logarithmic factors in $T ,$ for every $V \in$ $[ 0 , T ]$ . The lower bound combines the variation-dependent term $\Omega ( { \sqrt { T V } } )$ , obtained by adapting the construction in Paes Leme et al. (2021, Theorem 4.2), with the stationary lower bound Ω(log log T) from Kleinberg and Leighton (2003). The former considers the stronger restriction $| v _ { t + 1 } - v _ { t } | \leq V / T$ on changes between consecutive rounds,<sup>5</sup> while the latter applies when all buyers have the same value. We conclude this section with a sketch of the upper-bound proof. The full proof appears in Appendix B.

To bound the dynamic regret of Algorithm 1, we bound the regret from exploration and exploitation separately. During exploration, binary search takes at most $\lceil \log _ { 2 } ( 1 / \delta ) \rceil$ rounds per batch. Since each round incurs at most one unit of regret, the total exploration regret is at most $m \lceil \log _ { 2 } ( 1 / \delta ) \rceil$ , where m is the number of batches. Periodic restarts contribute at most $T / a$ completed batches, and each inconsistency restart requires at least $\delta$ variation within its batch. Hence, $m \leq 1 + T / a + V / \delta = O ( \sqrt { T V } + 1 )$ . Since $\delta \ge 1 / T$ , the total exploration regret is therefore at most $O ( ( \sqrt { T V } + 1 )$ log $T ) = O ( { \sqrt { T V } }$ log $T + \log T )$

To analyze exploitation, let $B _ { i } \subseteq [ T ]$ denote the rounds in batch i, and define its value range as $\Delta _ { i } \triangleq \operatorname* { m a x } _ { s , t \in B _ { i } } \left| v _ { s } - v _ { t } \right|$ . These within-batch value ranges satisfy $\begin{array} { r } { \sum _ { i \in [ m ] } \Delta _ { i } \leq V } \end{array}$ . Each accepted exploitation round in batch i incurs regret of at most $2 \delta + \Delta _ { i }$ . Each rejected exploitation round incurs at most one unit of regret and immediately ends the batch, so there is at most one such round per batch. Since each batch contains at most a rounds, the total exploitation regret is at most $\begin{array} { r } { 2 T \delta + \sum _ { i \in [ m ] } | B _ { i } | \Delta _ { i } + m \leq 2 T \delta + a V + m = O ( \sqrt { T V } + 1 ) } \end{array}$ . Combining the exploration and exploitation bounds proves the stated upper bound.

## 4 Optimal Dynamic Pricing using Multi-Scale Shallow Pricing

In this section, we turn to the more general environment in which buyers have heterogeneous d-dimensional context vectors. We present our main result: an online learning algorithm whose dynamic regret has optimal dependence on T and the non-stationarity upper bound V , up to logarithmic and dimension-dependent factors. Our algorithm (Algorithm 2) and its regret analysis extend the approach developed for the identical-context environment in Section 3.

Compared with the identical-context setting, our algorithm and its analysis introduce two main technical ingredients.

The first ingredient is to replace binary-search exploration with shallow pricing, introduced by Cohen et al. (2020) for stationary contextual dynamic pricing, while retaining lower-envelope pricing (LEP) for exploitation. With identical contexts, uncertainty concerns a single scalar value and can be represented by an interval. With heterogeneous contexts, uncertainty concerns a vector, and learning its projection in one context direction need not provide accurate information in another. Shallow pricing maintains an ellipsoidal approximation of the feasible region and uses its width in the current context direction to determine whether further exploration is needed. Thus, exploration and exploitation may be interleaved as diferent contexts arrive.

The second ingredient is to maintain multiple copies of the geometric learning procedure, each with a price-margin parameter λ drawn from a dyadic set $\Lambda = \{ \delta , 2 \delta , \ldots , 2 ^ { L } \delta \}$ , whose largest scale covers the diameter of the feature domain. This multiscale construction addresses a dificulty arising from the combination of heterogeneous contexts and non-stationary features when $d \geq 2 .$ As illustrated in Example 4.1, a relaxed feasible set can become narrow in the current context direction while remaining far from the current hidden feature. Maintaining multiple scales ensures that a larger-scale set can still provide a valid certificate of localization. This particular dificulty does not arise in either the stationary contextual setting of Cohen et al. (2020) or the non-stationary identical-context setting of Section 3, where a single price-margin scale sufices for the respective regret guarantees.

Shallow pricing with LEP. Our contextual algorithms build on the geometric approach of Cohen et al. (2020): each purchase or rejection provides a halfspace constraint on the hidden feature vector. For a nonempty compact convex set $K \subseteq \mathbb { R } ^ { d }$ , define its width in context direction u as

$$
\begin{array} { r } { \mathrm { w i d t h } ( K , u ) \triangleq \operatorname* { m a x } _ { x \in K } \left. x , u \right. - \operatorname* { m i n } _ { x \in K } \left. x , u \right. . } \end{array}
$$

At scale $\lambda ,$ we maintain a relaxed feasible set $S _ { t } ^ { ( \lambda ) }$ and an enclosing ellipsoid $\mathcal { E } _ { t } ^ { ( \lambda ) }$ , where

$$
\begin{array} { r } { \mathcal { E } _ { t } ^ { ( \lambda ) } \triangleq \left\{ x \in \mathbb { R } ^ { d } : ( x - c _ { t } ^ { ( \lambda ) } ) ^ { \top } ( A _ { t } ^ { ( \lambda ) } ) ^ { - 1 } ( x - c _ { t } ^ { ( \lambda ) } ) \leq 1 \right\} , \mathrm { a n d } \mathrm { w i d t h } ( \mathcal { E } _ { t } ^ { ( \lambda ) } , u _ { t } ) = 2 \sqrt { u _ { t } ^ { \top } A _ { t } ^ { ( \lambda ) } u _ { t } } . } \end{array}
$$

Here, $c _ { t } ^ { ( \lambda ) }$ is the center and $A _ { t } ^ { ( \lambda ) }$ is symmetric positive definite. The feasible set stores the feedback constraints and determines the exploitation price; the ellipsoid determines whether further exploration is needed. We initialize the feasible set and its enclosing ellipsoid as $S _ { 0 } = \{ x \in \mathbb { R } ^ { d } : x \geq$ $0 , \ \| x \| \leq 1 \}$ and $\mathcal { E } _ { 0 } = \{ x \in \mathbb { R } ^ { d } : \| x \| \leq 1 \}$ , respectively.

The ShallowPricing procedure explores at scale λ only when $\mathrm { w i d t h } ( { \mathcal { E } } _ { t } ^ { ( \lambda ) } , u _ { t } ) > 4 d \lambda$ . It proposes the midpoint price $\widetilde { p } _ { t } = \left. c _ { t } ^ { ( \lambda ) } , u _ { t } \right.$ ; our algorithm posts $p _ { t } =$ min $\{ 1 , \operatorname* { m a x } \{ 0 , \widetilde { p } _ { t } \} \}$ . After observing feedback $f _ { t } = \mathbb { I } \{ \mathrm { B u y e r ~ } t $ accept price $p _ { t } \}$ , we form the relaxed halfspace

$$
H _ { t } ^ { ( \lambda ) } \triangleq \left\{ \{ x \in \mathbb { R } ^ { d } : \langle x , u _ { t } \rangle \leq p _ { t } + \lambda \} , \quad f _ { t } = 0 , \right.
$$

Every observation updates the relaxed feasible sets using the actual price and feedback: $S _ { t + 1 } ^ { ( \lambda ) } =$ $S _ { t } ^ { ( \lambda ) } \cap H _ { t } ^ { ( \lambda ) }$ . On an exploration round, only the selected ellipsoid is updated, using the inferred midpoint feedback $\widetilde { f } _ { t } = \overline { { f _ { t } \mathbb { I } } } \{ \widetilde { p } _ { t } \leq 1 \}$ . The operator ShallowUpdat $\mathrm { e } ( \mathcal { E } , u ; \lambda , \widetilde { f } )$ denotes the standard shallow update at the unclipped midpoint, with acceptance if $\widetilde f = 1$ and rejection if $\widetilde f = 0$ . For $\mathit { \Delta } \widetilde { p } _ { t } < 0$ , the posted price is necessarily accepted, and its relaxed constraint implies the midpointbased constraint. For $\widetilde { p } _ { t } \ > \ 1$ , the midpoint would necessarily be rejected, and every $x \in S _ { 0 }$ satisfies $\langle x , u _ { t } \rangle \leq 1 < \widetilde { p } _ { t } + \lambda$ . Thus, clipping preserves the ellipsoidal outer approximation and the exploration guarantee. The update formulas are given in Appendix A.

For a nonempty compact feasible set $S ,$ its lower-envelope price in context direction u is

$$
\begin{array} { r } { \mathrm { L E P } ( S , u ) \triangleq \operatorname* { m a x } \left\{ 0 , \operatorname* { m i n } _ { x \in S } \left. x , u \right. \right\} . } \end{array}
$$

This generalizes lower-endpoint pricing in the identical-context setting. Since $S _ { t } ^ { ( \lambda ) } \subseteq \mathcal { E } _ { t } ^ { ( \lambda ) }$ , the ellipsoid also certifies that width $( S _ { t } ^ { ( \lambda ) } , u _ { t } ) \le \mathrm { w i d t h } ( \mathcal { E } _ { t } ^ { ( \lambda ) } , u _ { t } )$

Identical-context approach does not extend directly. A natural extension of the identicalcontext algorithm fixes a single price-margin parameter $\delta$ and maintains a relaxed feasible set $S _ { t } ^ { ( \delta ) }$ together with an enclosing ellipsoid $\mathcal { E } _ { t } ^ { ( \delta ) }$ . In each round, the algorithm explores using Shallow-Pricing if width $( \mathcal { E } _ { t } ^ { ( \delta ) } , u _ { t } ) > 4 d \delta ;$ otherwise, it posts $p _ { t } = \mathrm { L E P } ( S _ { t } ^ { ( \delta ) } , u _ { t } )$ . Every observation updates the relaxed feasible set, while only exploration rounds update the ellipsoid. The algorithm restarts when the window of a rounds expires, the relaxed feasible set becomes empty, or an exploitation price is rejected.

A crucial dificulty arises from using a single price-margin parameter. In the identical-context setting, each accepted exploitation round in batch $B _ { i }$ incurs regret of at most $2 \delta + \Delta _ { i }$ , where $\Delta _ { i }$ is the within-batch value range. With heterogeneous contexts, the analogous quantity is the withinbatch parameter diameter, $\Delta _ { i } \ \triangleq \operatorname* { m a x } _ { s , t \in B _ { i } } \Vert \theta _ { s } - \theta _ { t } \Vert$ . If $\Delta _ { i } \ \leq \ \delta$ , every parameter in the batch satisfies all δ-relaxed feedback constraints. Thus, whenever the ellipsoid is suficiently narrow for exploitation, the regret is at most 4dδ. However, when $\Delta _ { i } > \delta$ , even by only a constant factor, small parameter changes can cause a large localization error, as the following example illustrates. Appendix C provides a formal analysis of this single-scale extension. Optimizing the parameters in the resulting upper bound yields a dynamic regret guarantee of $O ( ( d ^ { 2 } + d ^ { 3 / 2 } V ^ { 1 / 3 } T ^ { 2 / 3 } ) \log T )$ . For fixed dimension, a matching lower bound up to logarithmic factors shows that this suboptimal rate is unavoidable under the selected tuning.

Example 4.1 (A narrow but misplaced relaxed feasible set). Let $d = 2 , f x \delta \in ( 0 , 1 / 1 6 ]$ , and write $q = \sqrt { 1 + \delta ^ { 2 } }$ . Initialize the relaxed feasible set to $S _ { 0 } = \{ x \in \mathbb { R } ^ { d } : x \geq 0 , ~ \| x \| \leq 1 \}$ . Consider two price queries within a batch.<sup>6</sup>

In round 1, let $u _ { 1 } = ( 0 , 1 ) , \theta _ { 1 } = ( 1 / 2 , 1 / 4 + \delta )$ , and $p _ { 1 } = 1 / 4 + \delta$ . The buyer accepts because $v _ { 1 } = p _ { 1 }$ , giving the relaxed constraint $x _ { 2 } \geq 1 / 4 .$ In round 2, let $u _ { 2 } = ( \delta , 1 ) / q , \theta _ { 2 } = ( 1 / 2 , 1 / 4 - 2 \delta )$ ， and $p _ { 2 } = ( 1 / 4 + \delta ^ { 2 } ) / q - \delta$ . The value is $v _ { 2 } = ( 1 / 4 - 3 \delta / 2 ) / q$ , so $p _ { 2 } - v _ { 2 } = \delta ( \delta + 3 / 2 - q ) / q > 0$ . The buyer therefore rejects, giving the relaxed constraint δx<sub>1</sub> $+ x _ { 2 } \le 1 / 4 + \delta ^ { 2 }$ . After these observations, the relaxed feasible set is $S ^ { ( \delta ) } = S _ { 0 } \cap \{ x \in \mathbb { R } ^ { d } : x _ { 2 } \geq 1 / 4 , \ \delta x _ { 1 } + x _ { 2 } \leq 1 / 4 + \delta ^ { 2 } \}$

From round 3 onward, let $u _ { t } = ( 1 , 0 )$ and $\theta _ { t } = ( 1 / 2 , 1 / 4 + \delta )$ . All contexts, parameters, and prescribed prices satisfy the model assumptions. The parameter diameter is 3δ, and the total parameter variation is 6δ. Yet the two relaxed constraints imply $0 \leq x _ { 1 } \leq \delta$ for every $x \in S ^ { ( \delta ) }$ . Since both $( 0 , 1 / 4 )$ and $( \delta , 1 / 4 )$ belong to this set, width $( S ^ { ( \delta ) } , u _ { t } ) = \delta$

For every round $t \geq 3$ , the true value is $1 / 2$ , whereas $\mathrm { L E P } ( S ^ { ( \delta ) } , u _ { t } ) = 0$ . This price is accepted and incurs regret $1 / 2$ . Its relaxed feedback constraint, $x _ { 1 } \geq - \delta$ , leaves the feasible set unchanged. Thus, if LEP continues to be used without a restart, every subsequent round incurs regret $1 / 2$ despite the parameter diameter being only 3δ.

The example shows that a small parameter diameter and a small directional width of the relaxed feasible set do not, by themselves, imply a regret bound of $O ( \Delta _ { i } + \delta )$ on accepted exploitation rounds. Here, $\Delta _ { i } + \delta = 4 \delta$ , while the regret remains $1 / 2$ as $\delta$ tends to zero. The two initial contexts are nearly parallel. Combining their feedback constraints requires dividing by $\delta ,$ which can amplify an $O ( \delta )$ change in one coordinate into a constant localization error in another. With identical contexts, all constraints concern the same scalar value, so this amplification does not occur.

This example illustrates a limitation of reasoning from the relaxed feasible set alone. It does not establish that the single-scale algorithm generates this history: the prescribed queries need not be its midpoint prices, and a narrow feasible set need not have a narrow enclosing ellipsoid. Accordingly, the example explains the obstacle to extending the one-dimensional exploitation argument, without proving a regret lower bound for every single-scale tuning.

This example motivates maintaining several relaxation scales, as we explain after introducing the multiscale algorithm.

Algorithm for general environment: multi-scale shallow pricing. Our algorithm retains the two restart mechanisms from the identical-context setting: periodic restarts controlled by a window-size parameter a and inconsistency restarts triggered by rejected exploitation prices. We again call the consecutive rounds between restarts a batch. Within each batch, the algorithm combines shallow-pricing exploration with lower-envelope pricing (LEP), using multiple relaxation scales to account for the unknown amount of parameter variation.

The algorithm is parameterized by a base price-margin parameter $\delta ,$ which determines the smallest relaxation scale. We maintain the dyadic scale set $\Lambda = \{ \delta , 2 \delta , \ldots , 2 ^ { L } \delta \}$ , where L is the smallest nonnegative integer such that $2 ^ { L } \delta \geq \sqrt { 2 }$ , the diameter of the feature domain for $d \geq 2$ Each scale $\lambda \in \Lambda$ maintains a relaxed feasible set $S _ { t } ^ { ( \lambda ) }$ and an enclosing ellipsoid $\mathcal { E } _ { t } ^ { ( \lambda ) }$ . At the beginning of each batch, all feasible sets and ellipsoids are reset to their initial states. Each scale represents a diferent allowance for parameter variation within the batch. A scale is active if its relaxed feasible set is nonempty; the largest scale always remains active. We initialize $S _ { 0 } = \{ x \in$ $\mathbb { R } ^ { d } : x \geq 0 , \ \| x \| \leq 1 \}$ and $\mathcal { E } _ { 0 } = \{ x \in \mathbb { R } ^ { d } : \| x \| \leq 1 \}$

For exploration, the algorithm checks the ellipsoid width in the current context direction at every active scale. If any active scale λ satisfies width $( \mathcal { E } _ { t } ^ { ( \lambda ) } , u _ { t } ) \ > \ 4 d \lambda$ , the algorithm selects the smallest such scale and posts the corresponding shallow-pricing price. The resulting feedback updates the relaxed feasible sets at all scales, using their respective margins, while only the ellipsoid at the selected scale receives a shallow-cut update. Since a new context may reveal uncertainty in a diferent direction, exploration and exploitation can be interleaved within a batch.

For exploitation, the algorithm requires every active scale to satisfy its width condition. It then posts the lower-envelope price computed from the smallest active scale. This choice uses the nesting of the relaxed feasible sets: a smaller scale gives a smaller feasible set and therefore a weakly higher lower-envelope price. Meanwhile, a suficiently large scale still contains the current hidden feature vector and certifies that the remaining uncertainty is small. Together, these properties control the regret on accepted exploitation rounds even when the smallest active set does not contain the current parameter.

As in the identical-context setting, the algorithm restarts after at most a rounds or immediately after an exploitation price is rejected. Periodic restarts limit the regret from changes that remain undetected because buyers continue accepting the posted prices. An inconsistency restart responds to overpricing and certifies more than δ parameter variation within the batch, allowing its frequency to be controlled by V. The window-size parameter a balances the cost of retaining outdated information against the exploration cost of frequent restarts, while the base margin $\delta$ balances pricing accuracy against exploration and inconsistency-restart costs. The formal algorithm is given in Algorithm 2.

Before stating the dynamic regret guarantee, we revisit Example 4.1 to explain how multiple scales address the illustrated dificulty.

Insert the first two observations from Example 4.1 into the relaxed feasible sets at every scale. At scale $\delta ,$ the resulting set is narrow in direction $u _ { t } = ( 1 , 0 )$ but excludes the current parameter $\theta _ { t } ~ = ~ ( 1 / 2 , 1 / 4 + \delta )$ . At scale $4 \delta ,$ the current parameter remains feasible because the withinbatch parameter diameter is only 3δ. This larger-scale set also contains $( 0 , 1 / 4 )$ , so its width in direction $u _ { t }$ is at least $1 / 2$ . For $\delta < 1 / 6 4$ , this exceeds the exploration threshold $4 d ( 4 \delta ) = 3 2 \delta$ . Any enclosing ellipsoid therefore also violates the width condition, and the multiscale rule would explore after this feedback history. The larger scale provides a reliable certificate of localization, while the smallest active scale determines the exploitation price. To see why this combination sufices, consider a batch $B _ { i }$ with parameter diameter $\Delta _ { i } \triangleq \operatorname* { m a x } _ { s , t \in B _ { i } } \Vert \theta _ { s } - \theta _ { t } \Vert$ . Let $\lambda _ { i }$ be the smallest maintained scale satisfying $\lambda _ { i } \geq \Delta _ { i }$ . Then $\lambda _ { i } \leq 2 \operatorname* { m a x } \{ \Delta _ { i } , \delta \}$ , and $\theta _ { t } \in S _ { t } ^ { ( \lambda _ { i } ) }$ throughout the batch. Thus, scale $\lambda _ { i }$ remains active. Whenever the algorithm exploits, every active scale, including $\lambda _ { i } ,$ satisfies its ellipsoid width condition. Let $\lambda _ { t } ^ { \operatorname* { m i n } }$ denote the smallest active scale. Since $\lambda _ { t } ^ { \mathrm { m i n } } \leq \lambda _ { i } ,$ nesting gives $S _ { t } ^ { ( \lambda _ { t } ^ { \operatorname* { m i n } } ) } \subseteq S _ { t } ^ { ( \lambda _ { i } ) }$ . Consequently, the posted price satisfies $p _ { t } = \mathrm { L E P } ( S _ { t } ^ { ( \lambda _ { t } ^ { \mathrm { m i n } } ) } , u _ { t } ) \ge$ $\mathrm { L E P } ( S _ { t } ^ { ( \lambda _ { i } ) } , u _ { t } )$ . On each accepted exploitation round, $v _ { t } - p _ { t } \leq \mathrm { w i d t h } ( S _ { t } ^ { ( \lambda _ { i } ) } , u _ { t } ) \leq \mathrm { w i d t h } ( \mathcal { E } _ { t } ^ { ( \lambda _ { i } ) } , u _ { t } ) \leq$

Algorithm 2 Multiscale shallow pricing with LEP   
1: Input: $T \in \mathbb { N } , d \in \mathbb { N }$ with $d \ge 2 , V \in [ 0 , T ]$   
2: Parameters: $\delta \gets$ max $\{ 1 / T , \sqrt { d V / T } \}$ , a ← min $\{ T , \lceil \sqrt { d T / V } \rceil \}$ ▷ set $a \gets T$ when $V = 0$   
3: $\Lambda \gets \{ \delta , 2 \delta , \ldots , 2 ^ { L } \delta \}$ , where L is the smallest nonnegative integer with $2 ^ { L } \delta \geq \sqrt { 2 }$   
4: $t \gets 1 , i \gets 0$   
5: while $t \leq T$ do   
6: $i  i + 1 , \tau  \operatorname* { m i n } \{ T , t + a - 1 \}$   
7: For all $\lambda \in \Lambda ,$ , set $S _ { t } ^ { ( \lambda ) } \gets S _ { 0 }$ and $\mathcal { E } _ { t } ^ { ( \lambda ) }  \mathcal { E } _ { 0 }$   
8: while $t \leq \tau$ do   
9: Observe $u _ { t } ;$ set $\Lambda _ { t }  \{ \lambda \in \Lambda : S _ { t } ^ { ( \lambda ) } \neq \varnothing \}$   
10: $\mathcal { X } _ { t } \gets \{ \lambda \in \Lambda _ { t } : \mathrm { w i d t h } ( \mathcal { E } _ { t } ^ { ( \lambda ) } , u _ { t } ) > 4 d \lambda \}$   
11: if $\boldsymbol { \mathcal { X } } _ { t } \neq \boldsymbol { \mathcal { O } }$ then   
12: $\lambda _ { t } \gets \operatorname* { m i n } { \mathcal { X } _ { t } , \widetilde { p } _ { t } } \gets \Big \langle c _ { t } ^ { ( \lambda _ { t } ) } , u _ { t } \Big \rangle , p _ { t } \gets \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , \widetilde { p } _ { t } \} \}$   
13: else   
14: $\lambda _ { t } ^ { \operatorname* { m i n } } \gets \operatorname* { m i n } \Lambda _ { t } , p _ { t } \gets \mathrm { L E P } ( S _ { t } ^ { ( \lambda _ { t } ^ { \operatorname* { m i n } } ) } , u _ { t } )$   
15: end if   
16: Post $p _ { t }$ and observe f   
17: For all $\lambda \in \Lambda ,$ set $\tilde { S _ { t + 1 } ^ { ( \lambda ) } } \gets S _ { t } ^ { ( \lambda ) } \cap H _ { t } ^ { ( \lambda ) }$ and $\mathcal { E } _ { t + 1 } ^ { ( \lambda ) }  \mathcal { E } _ { t } ^ { ( \lambda ) }$   
18: if $\underset { \sim } { \mathcal { X } } _ { t } \neq \emptyset$ then   
19: $\widetilde { f } _ { t } \gets f _ { t } \mathbb { I } \{ \widetilde { p } _ { t } \leq 1 \}$   
20: $\mathcal { E } _ { t + 1 } ^ { ( \lambda _ { t } ) } \gets \mathrm { S h a l l o w U p d a t e } ( \mathcal { E } _ { t } ^ { ( \lambda _ { t } ) } , u _ { t } ; \lambda _ { t } , \widetilde { f } _ { t } )$   
21: end if   
22: $t \gets t + 1$   
23: if $\mathcal { X } _ { t - 1 } = \emptyset$ and $f _ { t - 1 } = 0 ,$ break ▷ inconsistency restart   
24: end while   
25: end while

$4 d \lambda _ { i } \leq 8 d \operatorname* { m a x } \{ \Delta _ { i } , \delta \}$ . Multiple scales therefore recover a regret bound controlled by the parameter diameter and the base margin, even when the smallest active set excludes the current parameter. A misplaced smaller-scale set may also cause overpricing. The resulting rejection immediately triggers an inconsistency restart, and these rejected rounds are accounted for separately in the regret analysis. The bound above applies to accepted exploitation rounds without requiring the smallest active set to contain the current parameter.

Regret guarantee and its analysis. We now state the dynamic regret guarantee of Algorithm 2.

Theorem 4.2. In the contextual environment, given any non-stationarity upper bound $V \in [ 0 , T ]$ set $a = \operatorname* { m i n } \{ T , \lceil \sqrt { d T / V } \rceil \}$ and $\delta = \operatorname* { m a x } \{ 1 / T , \sqrt { d V / T } \}$ with the convention $a = T$ when $V = 0$ Then the dynamic regret of Algorithm 2 is at most $\dot { O } ( ( d ^ { 2 } + d ^ { 3 / 2 } \sqrt { T V } ) \log ^ { 2 } T )$

Moreover, for every $V \in [ 0 , T ]$ and every deterministic online algorithm, there exists an admissible instance with total variation at most V on which the algorithm incurs dynamic regret of at least $\Omega ( \sqrt { T V } +$ log log T).

Combining the upper and lower bounds in Theorem 4.2, the regret guarantee of Algorithm 2 is optimal among deterministic online algorithms, up to logarithmic factors in $T ,$ for every $V \in [ 0 , T ]$ when d is fixed. The term $O ( d ^ { 2 } \log ^ { 2 } T )$ accounts for the initial exploration cost and ensures that the guarantee also covers the stationary case $V = 0$ . The remaining term scales as $\sqrt { T V }$ , matching the variation-dependent lower bound up to logarithmic and dimension-dependent factors. We do not claim that the dependence on d is optimal.

The lower bound follows directly from Theorem 3.1, since the identical-context environment is a special case of the contextual environment. Specifically, any scalar value sequence $\{ v _ { t } \} _ { t \in [ T ] }$ can be embedded by setting $u _ { t } = ( 1 , 0 , \ldots , 0 )$ and $\theta _ { t } = ( v _ { t } , 0 , \ldots , 0 )$ . For the upper bound, it sufices to analyze $V \leq T / d$ . When $V > T / d ,$ the stated bound follows from the trivial regret bound $T _ { i }$ , since $d ^ { 3 / 2 } { \sqrt { T V } } \geq T$ . We next sketch the analysis for the nontrivial regime; the full proof is deferred to the Appendix D.

Consider the regime $V \leq T / d .$ . Let $B _ { 1 } , \ldots , B _ { m }$ denote the batches, and define their parameter diameters by $\Delta _ { i } \triangleq \operatorname* { m a x } _ { s , t \in B _ { i } } \Vert \theta _ { s } - \theta _ { t } \Vert$ . Since the batches are disjoint intervals, $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \Delta _ { i } \leq V } \end{array}$ . We bound the regret from exploration and exploitation separately.

We first bound the number of batches. Periodic restarts contribute at most $T / a$ completed batches. Every rejected exploitation round certifies more than $\delta$ parameter variation within its batch: otherwise, the current parameter would remain feasible at scale $\delta ,$ and the smallest-scale LEP could not exceed the current value. Thus, there are at most $V / \delta$ inconsistency restarts, giving m $\leq 1 + T / a + V / \delta = O ( 1 + \sqrt { T V / d } )$

For exploration, the shallow-pricing exploration bound gives at most $O ( d ^ { 2 } \log T )$ exploration rounds per scale in each batch. Since $\delta \ge 1 / T$ , the number of maintained scales is ${ \cal O } ( \log T )$ . Each exploration round incurs at most one unit of regret, so the total exploration regret is at most $O ( m d ^ { 2 } \log ^ { 2 } T ) = O ( ( d ^ { 2 } + d ^ { 3 / 2 } \sqrt { T V } ) \log ^ { 2 } T )$

For exploitation, let $\lambda _ { i }$ be the smallest maintained scale satisfying $\lambda _ { i } \geq \Delta _ { i }$ . As established above, this scale remains active throughout batch $B _ { i } { \mathrm { . } }$ and each accepted exploitation round incurs regret of at most $4 d \lambda _ { i } \leq 8 d \operatorname* { m a x } \{ \Delta _ { i } , \delta \}$ . Using $| B _ { i } | \le a$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \Delta _ { i } \leq V } \end{array}$ , the total regret from accepted exploitation rounds is at most $8 d T \delta + 8 d \sum _ { i = 1 } ^ { m } | B _ { i } | \Delta _ { i } \leq 8 d T \delta + 8 d a V = O ( d + d ^ { 3 / 2 } \sqrt { T V } )$ Each rejected exploitation round incurs at most one unit of regret, and there are at most $V / \delta$ such rounds. Their total regret is therefore at most $V / \delta \leq \sqrt { T V / d }$

Combining these bounds yields Regret $[ \mathrm { A L G } ] = O ( ( d ^ { 2 } + d ^ { 3 / 2 } \sqrt { T V } ) \log ^ { 2 } T )$ , as claimed.

## AI Use Statement

Generative AI tools were used as an assistive tool during the development of this work, including for improving exposition and organization, checking notation and internal consistency, discussing proof presentation and examples, and assisting with literature exploration. All research ideas, mathematical claims, proofs, and cited references included in the final manuscript were reviewed and verified by the authors. The authors take full responsibility for the final content of the paper.

## References

Adobe (2026). Adobe: Holiday shopping season drove a record \$257.8 billion online with consumers embracing generative AI tools.

Baby, D. and Wang, Y.-X. (2021). Optimal dynamic regret in exp-concave online learning. In Proceedings of the 34th Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 359–409.

Baby, D., Xu, J., and Wang, Y.-X. (2023). Non-stationary contextual pricing with safety constraints. Transactions on Machine Learning Research.

Besbes, O., Gur, Y., and Zeevi, A. (2014). Stochastic multi-armed-bandit problem with nonstationary rewards. In Advances in Neural Information Processing Systems, volume 27.

Besbes, O., Gur, Y., and Zeevi, A. (2015). Non-stationary stochastic optimization. Operations Research, 63(5):1227–1244.

Besbes, O. and Zeevi, A. (2009). Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations Research, 57(6):1407–1420.

Chen, Y., Lee, C.-W., Luo, H., and Wei, C.-Y. (2019). A new algorithm for non-stationary contextual bandits: Eficient, optimal and parameter-free. In Proceedings of the 32nd Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 696–726.

Cheung, W. C., Simchi-Levi, D., and Zhu, R. (2019). Learning to optimize under non-stationarity. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 1079–1087.

Cohen, M. C., Lobel, I., and Paes Leme, R. (2020). Feature-based dynamic pricing. Management Science, 66(11):4921–4943.

Cosson, R., Fusco, F., Gupta, A., Leonardi, S., Leme, R. P., and Russo, M. (2026). Contextual online bilateral trade. CoRR, abs/2602.12903.

Feng, Y., Ma, M., Peng, B., and Wan, Z. (2026a). Contextual search in principal-agent games: The curse of degeneracy. In Proceedings of the 2026 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1666–1723.

Feng, Y., Ma, M., Peng, B., and Wan, Z. (2026b). Searching for optimal prices in two-sided markets. CoRR, abs/2602.11691.

Gaucher, S., Bernasconi, M., Castiglioni, M., Celli, A., and Perchet, V. (2025). Feature-based online bilateral trade. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Javanmard, A. (2017). Perishability of data: Dynamic pricing under varying-coeficient models. Journal of Machine Learning Research, 18(53):1–31.

Javanmard, A. and Nazerzadeh, H. (2019). Dynamic pricing in high-dimensions. Journal of Machine Learning Research, 20(9):1–49.

Jiang, F. and Zhao, Z. (2026). On non-stationary dynamic pricing: Adaptivity and optimality. arXiv preprint arXiv:2607.24115.

Keskin, N. B. and Zeevi, A. (2017). Chasing demand: Learning and earning in a changing environment. Mathematics of Operations Research, 42(2):277–307.

Kleinberg, R. and Leighton, T. (2003). The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In Proceedings of the 44th Annual IEEE Symposium on Foundations of Computer Science, pages 594–605.

Krishnamurthy, A., Lykouris, T., Podimata, C., and Schapire, R. (2021). Contextual search in the presence of irrational agents. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, STOC 2021, pages 910–918, New York, NY, USA. Association for Computing Machinery.

Krishnamurthy, A., Lykouris, T., Podimata, C., and Schapire, R. E. (2023). Contextual search in the presence of adversarial corruptions. Operations Research, 71(4):1120–1135.

Lattimore, T. and Szepesvári, C. (2020). Bandit Algorithms. Cambridge University Press.

Liu, A., Paes Leme, R., and Schneider, J. (2021). Optimal contextual pricing and extensions. In Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms, pages 1059–1078.

Lobel, I., Paes Leme, R., and Vladu, A. (2018). Multidimensional binary search for contextual decision-making. Operations Research, 66(5):1346–1361.

Luo, Y., Sun, W. W., and Liu, Y. (2022). Contextual dynamic pricing with unknown noise: Explorethen-UCB strategy and improved regrets. In Advances in Neural Information Processing Systems, volume 35, pages 37445–37457.

Lykouris, T., Nietert, S., Okoroafor, P., Podimata, C., and Zimmert, J. (2025). Contextual dynamic pricing with heterogeneous buyers. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025.

Paes Leme, R., Podimata, C., and Schneider, J. (2022). Corruption-robust contextual search through density updates. In Proceedings of the 35th Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 3504–3505.

Paes Leme, R. and Schneider, J. (2018). Contextual search via intrinsic volumes. In Proceedings of the 59th Annual IEEE Symposium on Foundations of Computer Science, pages 268–282.

Paes Leme, R., Sivan, B., Teng, Y., and Worah, P. (2021). Learning to price against a moving target. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 6223–6232.

Xu, J. and Wang, Y.-X. (2021). Logarithmic regret in feature-based dynamic pricing. In Advances in Neural Information Processing Systems, volume 34, pages 13898–13910.

Xu, J. and Wang, Y.-X. (2022). Towards agnostic feature-based dynamic pricing: Linear policies vs linear valuation with unknown noise. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pages 9643–9662.

Xu, J. and Wang, Y.-X. (2026). Optimal contextual pricing under agnostic non-Lipschitz demand. arXiv preprint arXiv:2605.05609.

Zhao, Z., Jiang, F., Yu, Y., and Chen, X. (2026). High-dimensional dynamic pricing under nonstationarity: Learning and earning with change-point detection. Management Science. Articles in Advance.

Zinkevich, M. (2003). Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th International Conference on Machine Learning. AAAI Press.

## A Technical Background: Geometric Search and Shallow Pricing

## A.1 Background: geometric search in feature-based pricing

Our contextual algorithms build on the geometric viewpoint introduced in feature-based dynamic pricing by Cohen et al. (2020). In that model, each arriving product is described by an observed feature vector, the buyer’s valuation is linear in an unknown parameter, and the seller observes only binary accept/reject feedback after posting a price. Thus, each observation induces a halfspace constraint on the unknown parameter. Pricing can therefore be viewed as a geometric search problem: the seller maintains a region of parameters that remain consistent with past feedback, and each new context direction asks for the range of possible valuations induced by this region.

For a convex set $K \subseteq \mathbb { R } ^ { d }$ and a context direction u, define the directional width of K along u as

$$
{ \mathrm { w i d t h } } ( K , u ) : = \operatorname* { m a x } _ { x \in K } \left. x , u \right. - \operatorname* { m i n } _ { x \in K } \left. x , u \right. .
$$

This is the length of the interval of valuations that are still possible in context u when the hidden parameter is only known to lie in K. Equivalently, if

$$
b ^ { - } ( K , u ) : = \operatorname* { m i n } _ { x \in K } \left. x , u \right. , \qquad b ^ { + } ( K , u ) : = \operatorname* { m a x } _ { x \in K } \left. x , u \right. ,
$$

then

$$
{ \mathrm { w i d t h } } ( K , u ) = b ^ { + } ( K , u ) - b ^ { - } ( K , u ) .
$$

The algorithmic primitive in Cohen et al. (2020) is EllipsoidPricing. Instead of working directly with a possibly complicated polyhedral uncertainty set, the algorithm maintains an ellipsoidal outer approximation

$$
\mathcal { E } _ { t } = \mathcal { E } ( A _ { t } , c _ { t } ) : = \left\{ \boldsymbol { x } \in \mathbb { R } ^ { d } : ( \boldsymbol { x } - \boldsymbol { c } _ { t } ) ^ { \top } \boldsymbol { A } _ { t } ^ { - 1 } ( \boldsymbol { x } - \boldsymbol { c } _ { t } ) \leq 1 \right\} ,
$$

where $c _ { t }$ is the center and $A _ { t }$ is a positive definite matrix. When this primitive is run at scale λ in our algorithms, we write these quantities as $E _ { t } ^ { ( \lambda ) } , c _ { t } ^ { ( \lambda ) }$ , and $A _ { t } ^ { ( \lambda ) }$ . For a context $u _ { t } .$ , the minimum and maximum possible valuations over this ellipsoid have the closed forms

$$
b _ { t } ^ { - } = \left. c _ { t } , u _ { t } \right. - \sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } , \qquad b _ { t } ^ { + } = \left. c _ { t } , u _ { t } \right. + \sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } .
$$

Thus

$$
\mathrm { w i d t h } ( \mathcal { E } _ { t } , u _ { t } ) = b _ { t } ^ { + } - b _ { t } ^ { - } = 2 \sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } .
$$

Given a target accuracy $\eta > 0$ , EllipsoidPricing behaves as follows. If

$$
b _ { t } ^ { + } - b _ { t } ^ { - } \leq \eta ,
$$

the algorithm exploits by posting the lower endpoint

$$
p _ { t } = b _ { t } ^ { - } .
$$

If instead

$$
b _ { t } ^ { + } - b _ { t } ^ { - } > \eta ,
$$

the algorithm explores by posting the midpoint price

$$
p _ { t } = \frac { b _ { t } ^ { - } + b _ { t } ^ { + } } { 2 } = \left. c _ { t } , u _ { t } \right. .
$$

An acceptance, $f _ { t } = 1$ , implies the central halfspace constraint

$$
\langle x , u _ { t } \rangle \geq p _ { t } ,
$$

while a rejection, $f _ { t } = 0$ , implies

$$
\langle x , u _ { t } \rangle \leq p _ { t } .
$$

The next ellipsoid is chosen as the minimum-volume ellipsoid containing the corresponding halfellipsoid.

More explicitly, define

$$
q _ { t } : = \frac { A _ { t } u _ { t } } { \sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } } .
$$

For an acceptance, the central-cut update is

$$
c _ { t + 1 } = c _ { t } + \frac { 1 } { d + 1 } q _ { t } ,
$$

and for a rejection it is

$$
c _ { t + 1 } = c _ { t } - \frac { 1 } { d + 1 } q _ { t } .
$$

In both cases,

$$
A _ { t + 1 } = \frac { d ^ { 2 } } { d ^ { 2 } - 1 } \left( A _ { t } - \frac { 2 } { d + 1 } q _ { t } q _ { t } ^ { \top } \right) .
$$

The key fact behind the exploration bound is that each central cut decreases the ellipsoid volume by a constant factor, while exploration occurs only when the current direction has large ellipsoida width.

The robust primitive we use is ShallowPricing, also from Cohen et al. (2020). It modifies the central-cut update by adding a safety bufer to the inferred halfspace. At scale $\lambda > 0$ , let

$$
\widetilde { p } _ { t } : = \frac { b _ { t } ^ { - } + b _ { t } ^ { + } } { 2 } = \langle c _ { t } , u _ { t } \rangle
$$

denote the standard, unclipped midpoint proposal. Under the standard ShallowPricing update, an acceptance produces the shifted constraint

$$
\langle x , u _ { t } \rangle \geq \widetilde { p } _ { t } - \lambda ,
$$

and a rejection produces the shifted constraint

$$
\left. x , u _ { t } \right. \leq \widetilde p _ { t } + \lambda .
$$

Thus the update no longer cuts the ellipsoid through its center; it is a shallow cut.

The shallow-cut update can also be written in closed form. Let

$$
\alpha _ { t } : = - \frac { \lambda } { \sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } } .
$$

The quantity $\alpha _ { t }$ is the signed depth of the cut. When the algorithm explores only if

$$
\mathrm { w i d t h } ( \mathcal { E } _ { t } , u _ { t } ) > 4 d \lambda ,
$$

we have

$$
\sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } > 2 d \lambda ,
$$

and hence

$$
\alpha _ { t } > - \frac { 1 } { 2 d } .
$$

This is the regime in which the shallow-cut ellipsoid update is valid.

For an acceptance, the shallow-cut center update is

$$
c _ { t + 1 } = c _ { t } + \frac { 1 + d \alpha _ { t } } { d + 1 } q _ { t } ,
$$

whereas for a rejection,

$$
c _ { t + 1 } = c _ { t } - \frac { 1 + d \alpha _ { t } } { d + 1 } q _ { t } .
$$

In both cases, the shape matrix is updated as

$$
A _ { t + 1 } = \frac { d ^ { 2 } } { d ^ { 2 } - 1 } ( 1 - \alpha _ { t } ^ { 2 } ) \left( A _ { t } - \frac { 2 ( 1 + d \alpha _ { t } ) } { ( d + 1 ) ( 1 + \alpha _ { t } ) } q _ { t } q _ { t } ^ { \top } \right) .
$$

Clipping the exploration price to [0, 1]. Our algorithms post

$$
p _ { t } : = \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , \widetilde { p } _ { t } \} \} .
$$

Let

$$
\widetilde { f } _ { t } : = f _ { t } \mathbf { 1 } \{ \widetilde { p } _ { t } \leq 1 \}
$$

denote the inferred feedback at the unclipped midpoint. We denote the corresponding ellipsoid update by

$$
E _ { t + 1 } = \operatorname { S h a l l o w U p d a t e } ( E _ { t } , u _ { t } ; \lambda , \widetilde { f } _ { t } ) .
$$

The relaxed feasible set is updated using the actual posted price $p _ { t }$ and observed feedback $f _ { t } .$ , while the ellipsoid is updated using the standard shallow cut associated with $\widetilde { p } _ { t }$ and $\widetilde { f } _ { t }$

If $0 ~ \le ~ \widetilde { p } _ { t } ~ \le ~ 1$ , then $p _ { t } \ = \ \widetilde { p _ { t } }$ and $\widetilde { f } _ { t } ~ = ~ f _ { t }$ , so the standard shallow-pricing update applies unchanged.

Suppose instead that $\mathit { \Delta } \widetilde { p } _ { t } < 0$ . Since realized valuations are nonnegative, $p _ { t } = 0$ is necessarily accepted, and the actual λ-relaxed feedback constraint is

$$
\langle x , u _ { t } \rangle \geq - \lambda .
$$

Since $\begin{array} { r } { \tilde { p } _ { t } < 0 . } \end{array}$

$$
\{ x : \langle x , u _ { t } \rangle \geq - \lambda \} \subseteq \{ x : \langle x , u _ { t } \rangle \geq \widetilde { p } _ { t } - \lambda \} .
$$

Thus the actual relaxed constraint is stronger than the standard acceptance constraint associated with $\widetilde { p } _ { t }$

Finally, suppose that $\widetilde { p } _ { t } > 1$ . Then $\widetilde f _ { t } = 0$ . Since every $x \in S _ { 0 }$ satisfies $\langle x , u _ { t } \rangle \leq 1$ , we have

$$
\left. x , u _ { t } \right. \leq 1 < \widetilde p _ { t } + \lambda .
$$

Hence the updated relaxed feasible set is contained in the standard rejection halfspace

$$
\{ x : \langle x , u _ { t } \rangle \leq \widetilde p _ { t } + \lambda \} ,
$$

regardless of the feedback observed at the clipped price $p _ { t } = 1$

Therefore, in all cases, the standard shallow cut associated with $( \widetilde { p } _ { t } , \widetilde { f } _ { t } )$ contains the updated relaxed feasible set. Thus clipping the posted price to [0, 1] preserves the ellipsoidal outerapproximation invariant and the usual shallow-cut exploration guarantee.

In our non-stationary setting, the relaxation parameter λ has a diferent interpretation from statistical noise: it is a robustness margin that allows past feedback constraints to remain compatible with a parameter that may have moved within the current batch. The single-scale baseline uses one fixed radius $\lambda = \delta$ , while our final algorithm maintains a dyadic hierarchy of such radii. The ShallowPricing primitive itself is not our contribution; our use of it together with scale-dependent relaxed feasible sets and LEP pricing is described in Section 4.

## A.2 Relaxed feasible sets versus ellipsoidal outer approximations

In our non-stationary algorithms, the relaxed feasible set and its ellipsoidal outer approximation have distinct roles. At each relaxation scale λ we maintain

$$
S _ { t } ^ { ( \lambda ) } \subseteq \mathcal { E } _ { t } ^ { ( \lambda ) } .
$$

The relaxed feasible set is the intersection of the shifted feedback halfspaces and is therefore the object used for LEP pricing. The ellipsoid is used only to decide whether the current context direction is suficiently uncertain to require exploration and, on such a round, to perform the shallow-cut update. In particular,

$$
\mathrm { w i d t h } ( S _ { t } ^ { ( \lambda ) } , u _ { t } ) \leq \mathrm { w i d t h } ( \mathcal { E } _ { t } ^ { ( \lambda ) } , u _ { t } ) .
$$

On a multiscale exploration round at scale $\lambda _ { t } ,$ , only $\mathcal { E } _ { t } ^ { ( \lambda _ { t } ) }$ is shallow-updated. This is because the ellipsoid update uses ShallowUpdate with the unclipped midpoint $\widetilde { p } _ { t }$ and its inferred feedback $\widetilde { f } _ { t }$ . Since $\widetilde { p } _ { t }$ is centered with respect to $E _ { t } ^ { ( \lambda _ { t } ) }$ , the shallow-cut depth guarantee above applies directly at that scale. For every other scale λ, leaving the ellipsoid unchanged preserves the outerapproximation invariant because

$$
S _ { t + 1 } ^ { ( \lambda ) } \subseteq S _ { t } ^ { ( \lambda ) } \subseteq \mathcal { E } _ { t } ^ { ( \lambda ) } = \mathcal { E } _ { t + 1 } ^ { ( \lambda ) } .
$$

Throughout the remainder of the analysis, let

$$
R : = { \sqrt { 2 } } ,
$$

which upper-bounds the diameter of the initial feasible set $S _ { 0 }$

Lemma A.1 (Shallow-cut exploration bound). Fix a relaxation scale $\lambda > 0$ and an exploration threshold η satisfying

$$
4 d \lambda \leq \eta < 2 0 R ( d + 1 ) .
$$

If an ellipsoid is updated only on rounds for which

$$
\mathrm { w i d t h } ( E _ { t } , u _ { t } ) > \eta ,
$$

using the standard λ-shifted shallow cut associated with the unclipped midpoint proposal, then within a fixed batch the number of such updates is at most

$$
2 0 d ^ { 2 } \ln \left( \frac { 2 0 R ( d + 1 ) } { \eta } \right) .
$$

Proof. Inspecting the proof of Theorem 3 of Cohen et al. (2020), the exploration-count argument uses $\eta \geq 4 d \lambda$ to ensure that the shallow-cut depth satisfies $\alpha _ { t } \geq - 1 / ( 2 d )$ , and then gives

$$
N \leq 2 0 d ^ { 2 } \ln \left( \frac { 2 0 R ( d + 1 ) } { \eta } \right) .
$$

The clipping argument above ensures that the ellipsoid update used here is the same valid shallow update at the unclipped midpoint. □

## B Identical Context Upper Bound

## B.1 Complete proof of Theorem 3.1

Illustration. During exploration, binary search progressively narrows $[ p _ { i } ^ { \operatorname* { m i n } } , p _ { i } ^ { \operatorname* { m a x } } ]$ according to the observed accept/reject feedback. The final endpoints retain accepted/rejected witness rounds that are used in the regret analysis below. Once its width is at most $\delta ,$ the algorithm switches to LEP with a margin and repeatedly posts $\underline { { p } } _ { i } = \operatorname* { m a x } \{ p _ { i } ^ { \mathrm { m i n } } - \delta , 0 \}$ . The margin makes a rejection informative: if this LEP price is rejected, the valuation must have decreased suficiently relative to an earlier exploration round, and the resulting reset can be charged to the total variation budget. The hard window-size reset, in contrast, limits the duration of undetectable upward movements. These are precisely the two ingredients used in the regret analysis below.

Proof. Let

$$
B _ { 1 } = \{ 1 , \ldots , t _ { 1 } \} , \quad B _ { 2 } = \{ t _ { 1 } + 1 , \ldots , t _ { 2 } \} , \quad \ldots , \quad B _ { r } = \{ t _ { r - 1 } + 1 , \ldots , T \}
$$

denote the batches generated by the algorithm, i.e., consecutive rounds between resets. Let $R _ { i }$ denote the regret incurred in batch $B _ { i }$ , and let $E _ { i } \subseteq B _ { i }$ denote its exploration rounds. By construction,

$$
| B _ { i } | \leq a .
$$

Also define the variation diameter of batch $B _ { i }$ by

$$
\Delta _ { i } : = \operatorname* { m a x } _ { j , k \in B _ { i } } | v _ { j } - v _ { k } | .
$$

For every exploitation round $j \in B _ { i } \setminus E _ { i }$ , the algorithm posts the same LEP price with margin,

$$
p _ { j } = \underline { { p } } _ { i } = \operatorname* { m a x } \{ p _ { i } ^ { \operatorname* { m i n } } - \delta , 0 \} .
$$

For any batch that reaches exploitation, let

$$
L _ { i } : = p _ { i } ^ { \operatorname * { m i n } } , \qquad U _ { i } : = p _ { i } ^ { \operatorname * { m a x } }
$$

denote the final binary-search interval, so that

$$
U _ { i } - L _ { i } \leq \delta .
$$

We first bound the regret on an accepted LEP round $j \in B _ { i } \setminus E _ { i }$ . Recall that

$$
p _ { j } = \underline { { p } } _ { i } = \operatorname* { m a x } \{ L _ { i } - \delta , 0 \} .
$$

If $U _ { i } < 1$ , then the final upper endpoint must have been created by a rejected binary-search query. Hence there exists an exploration round $t _ { 1 } \in E _ { i }$ such that

$$
v _ { t _ { 1 } } < U _ { i } .
$$

Therefore, by the definition of $\Delta _ { i }$

$$
v _ { j } \leq v _ { t _ { 1 } } + \Delta _ { i } < U _ { i } + \Delta _ { i } \leq L _ { i } + \delta + \Delta _ { i } .
$$

Consequently,

$$
v _ { j } - p _ { i } \leq \Delta _ { i } + 2 \delta .
$$

Indeed, if $L _ { i } \geq \delta$ , then $p _ { i } = L _ { i } - \delta$ and the claim follows directly; while if $L _ { i } < \delta ,$ , then $p _ { i } = 0$ and

$$
v _ { j } < L _ { i } + \delta + \Delta _ { i } < \Delta _ { i } + 2 \delta .
$$

If instead $U _ { i } = 1$ , then $U _ { i } - L _ { i } \leq \delta$ implies $L _ { i } \ge 1 - \delta$ . Hence

$$
p _ { i } = \operatorname* { m a x } \{ L _ { i } - \delta , 0 \} \geq 1 - 2 \delta ,
$$

and since $v _ { j } \leq 1$

$$
v _ { j } - p _ { i } \le 2 \delta \le \Delta _ { i } + 2 \delta .
$$

Thus every accepted $\mathrm { L E P }$ round incurs regret at most $\Delta _ { i } + 2 \delta$

The regret of batch $B _ { i }$ can be decomposed as

$$
\begin{array} { r l } & { \displaystyle R _ { i } = \sum _ { j \in B _ { i } } \Big ( v _ { j } - \mathbb { Z } \{ p _ { j } \leq v _ { j } \} p _ { j } \Big ) } \\ & { \quad = \sum _ { j \in E _ { i } } \Big ( v _ { j } - \mathbb { Z } \{ p _ { j } \leq v _ { j } \} p _ { j } \Big ) + \sum _ { j \in B _ { i } \backslash E _ { i } } \Big ( v _ { j } - \mathbb { Z } \{ p _ { j } \leq v _ { j } \} p _ { j } \Big ) } \\ & { \quad \leq | E _ { i } | + | B _ { i } | \big ( \Delta _ { i } + 2 \delta \big ) + 1 } \\ & { \quad \leq O ( \log ( T ) + | B _ { i } | \delta + a \Delta _ { i } ) . } \end{array}
$$

Here we used that binary search requires at most log(T) exploration rounds, and that on every exploitation round before a reset the regret is at most $\Delta _ { i } + 2 \delta$

Summing over all batches gives

$$
\begin{array} { r } { \mathcal { R } _ { T } = \displaystyle \sum _ { i = 1 } ^ { r } R _ { i } \leq \sum _ { i = 1 } ^ { r } O \left( | B _ { i } | \delta + a \Delta _ { i } + \log ( T ) \right) } \\ { = O ( T \delta + a \sum _ { i = 1 } ^ { r } \Delta _ { i } + r \log ( T ) ) . } \end{array}
$$

Since the batches are disjoint intervals,

$$
\sum _ { i = 1 } ^ { r } \Delta _ { i } \leq \mathrm { T V } ( \mathbf { v } ) \leq V .
$$

It therefore remains to bound the number of batches.

Lemma B.1. The two reset mechanisms imply

$$
r \leq \frac { V } { \delta } + \frac { T } { a } + 1 .
$$

Proof. Every batch $B _ { i }$ ends for one of the following reasons:

• the window-size cap is reached, i.e., $| B _ { i } | = a ;$

$i = r ,$ so this is the final incomplete batch;

• an inconsistency occurs on an LEP round and triggers an immediate reset.

There are at most $T / a$ batches of the first type and at most one batch of the second type. We now show that there are at most $V / \delta$ inconsistency resets.

Consider a batch $B _ { i }$ that ends because an LEP price is rejected at round $t _ { i } .$ . Since the algorithm resets only after the binary-search phase, this means that LEP has become overpricing:

$$
v _ { t _ { i } } < p _ { t _ { i } } = \underline { { p } } _ { i } .
$$

Moreover, an inconsistency can only occur when $\underline { { p } } _ { i } > 0 \underline { { \ : } }$ , so

$$
\underline { { { p } } } _ { i } = p _ { i } ^ { \mathrm { m i n } } - \delta .
$$

Hence

$$
v _ { t _ { i } } < p _ { i } ^ { \mathrm { m i n } } - \delta .
$$

During binary search, the lower endpoint $p _ { i } ^ { \mathrm { m i n } }$ can increase only after an accepted query. Therefore, there exists an exploration round $j \in E _ { i }$ at which a price equal to the resulting lower endpoint was accepted, so

$$
p _ { i } ^ { \mathrm { m i n } } \leq v _ { j } .
$$

Consequently,

$$
v _ { t _ { i } } + \delta < p _ { i } ^ { \operatorname * { m i n } } \leq v _ { j } ,
$$

and therefore

$$
\delta < | v _ { j } - v _ { t _ { i } } | \leq \Delta _ { i } .
$$

In particular,

$$
\delta < \sum _ { k = t _ { i - 1 } + 1 } ^ { t _ { i } - 1 } | v _ { k + 1 } - v _ { k } | .
$$

Thus every inconsistency reset can be charged to more than $\delta$ total variation inside its batch. Since these batches are disjoint and

$$
\mathrm { T V } ( \mathbf { v } ) \leq V ,
$$

there can be at most $V / \delta$ such resets. Combining the three cases gives

$$
r \leq \frac { T } { a } + \frac { V } { \delta } + 1 .
$$

Using Lemma B.1, we obtain

$$
\mathcal { R } _ { T } \leq O ( T \delta + a V + \left( \frac { T } { a } + \frac { V } { \delta } + 1 \right) \log ( T ) ) .
$$

For notational simplicity, in the tuning calculation below we ignore the ceiling in the definition of a. This afects the bounds only by an absolute constant factor: for $x \geq 1$ ，

$$
x \leq \lceil x \rceil \leq 2 x ,
$$

and the additional truncation by T preserves this constant-factor relation. Thus replacing min $\{ T , \lceil \sqrt { T / V } \rceil \}$ by min $\{ T , { \sqrt { T / V } } \}$ does not afect the stated big-O regret bound. Setting

$$
a = \operatorname * { m i n } \{ T , \sqrt { \frac { T } { V } } \} , \qquad \delta = \operatorname * { m a x } \{ \frac { 1 } { T } , \sqrt { \frac { V } { T } } \} ,
$$

gives

$$
\begin{array} { r l r } {  { \mathcal { R } _ { T } \le O ( T \delta + a V + ( \frac { T } { a } + \frac { V } { \delta } ) \log ( T ) + \log ( T ) ) } } \\ & { } & { ~ = O ( \sqrt { T V } \log ( T ) + \log ( T ) ) . ~ } \end{array}
$$

for the case $\begin{array} { r } { V > \frac { 1 } { T } } \end{array}$ and when $V \leq \frac { 1 } { T }$ we get

$$
\begin{array} { l } { { \displaystyle \mathcal { R } _ { T } \leq O ( T \delta + a V + \left( \frac { T } { a } + \frac { V } { \delta } \right) \log ( T ) + \log ( T ) ) } } \\ { { \displaystyle \qquad = O ( 1 + \log ( T ) ) = O ( \log ( T ) ) . } } \end{array}
$$

This proves the claimed dynamic-regret bound.

Remark B.2. Alternatively for $V > 0$ choosing

$$
a = \operatorname* { m i n } \left\{ T , \sqrt { \frac { T \log T } { V } } \right\} , \qquad \delta = \operatorname* { m a x } \left\{ \frac { 1 } { T } , \sqrt { \frac { V \log T } { T } } \right\} ,
$$

in the bound above yields the slightly sharper guarantee

$$
R E G R E T [ A L G ] = O \left( { \sqrt { T V \log T } } + \log T \right) .
$$

For $V = 0$ , we take $a = T$ and $\delta = 1 / T$

Algorithm 3 Single-scale shallow pricing with LEP   
1: Input: T ∈ N, d ∈ N with $d \ge 2 , V \in [ 0 , T ]$   
2: Parameters: $\delta \gets \operatorname* { m a x } \{ 1 / T , d ^ { 1 / 2 } ( V / \dot { T } ) ^ { 1 / 3 } \}$ , a ← min{T, ⌈d<sup>1/2</sup>(T/V)<sup>1/3</sup>⌉}, η ← 4dδ ▷ set a ← T when V = 0   
3: t ← 1, i ← 0   
4: while $t \leq T$ do   
5: $i  i + 1 , \tau $ min $\{ T , t + a - 1 \}$   
6: Set $S _ { t } \gets S _ { 0 } , S _ { t } ^ { + } \gets \mathsf { \bar { { S } } _ { 0 } } .$ and $\mathcal { E } _ { t }  \mathcal { E } _ { 0 }$   
7: while $t \leq \tau$ do   
8: if $S _ { t } ^ { + } = \emptyset ,$ , break ▷ inconsistency restart   
9: Observe u<sub>t</sub>   
10: if width $( \mathcal { E } _ { t } , u _ { t } ) > \eta$ then   
11: $\widetilde { p } _ { t } \gets \langle c _ { t } , u _ { t } \rangle$ , pt ← min{1, max{0, pet}}   
12: else   
13: $p _ { t } \gets \mathrm { L E P } ( S _ { t } ^ { + } , u _ { t } )$   
14: end if   
15: Post $p _ { t }$ and observe $f _ { t }$   
16: Construct $H _ { t }$ and $H _ { t } ^ { + }$ according to $p _ { t }$ and $f _ { t }$   
17: $S _ { t + 1 } \gets S _ { t } \cap H _ { t } , S _ { t + 1 } ^ { + }  S _ { t } ^ { + } \cap H _ { t } ^ { + }$ , and $\mathcal { E } _ { t + 1 }  \mathcal { E } _ { t }$   
18: if width $( \mathcal { E } _ { t } , u _ { t } ) > \eta$ then   
19: $\widetilde { f _ { t } }  f _ { t } \mathbb { I } \{ \widetilde { p _ { t } } \leq 1 \}$   
20: $\mathscr { E } _ { t + 1 } \gets$ ShallowUpdate $( \mathcal { E } _ { t } , u _ { t } ; \delta , \widetilde { f _ { t } } )$   
21: end if   
22: $t \gets t + 1$   
23: if width $\boldsymbol { \mathrm { \ell } } ( \mathcal { E } _ { t - 1 } , u _ { t - 1 } ) \leq \boldsymbol { \eta }$ and $f _ { t - 1 } = 0 ,$ , break ▷ inconsistency restart   
24: end while   
25: end while

## C Single-Scale Contextual Baseline

This appendix gives the complete single-scale contextual algorithm and its analysis. As discussed in Section 4, this is the most direct geometric extension of the identical-context warm-up: it maintains a single relaxed feasible set at radius $\delta ,$ uses shallow pricing with the midpoint proposal clipped to [0, 1], and exploits using LEP.

Within a batch, let $S _ { t }$ denote the set of parameters consistent with the exact feedback observed so far. In parallel, let $S _ { t } ^ { + } = S _ { t } ^ { ( \delta ) }$ denote the δ-relaxed feasible set. After posting $p _ { t }$ in context $u _ { t } .$ define

$$
H _ { t } = \left\{ \begin{array} { l l } { \{ x : \langle x , u _ { t } \rangle \leq p _ { t } \} , \quad f _ { t } = 0 , } \\ { \{ x : \langle x , u _ { t } \rangle \geq p _ { t } \} , \quad f _ { t } = 1 , } \end{array} \right. \qquad H _ { t } ^ { + } = \left\{ \begin{array} { l l } { \{ x : \langle x , u _ { t } \rangle \leq p _ { t } + \delta \} , \quad f _ { t } = 0 , } \\ { \{ x : \langle x , u _ { t } \rangle \geq p _ { t } - \delta \} , \quad f _ { t } = 1 . } \end{array} \right.
$$

The feasible sets are updated according to

$$
S _ { t + 1 } = S _ { t } \cap H _ { t } , \qquad S _ { t + 1 } ^ { + } = S _ { t } ^ { + } \cap H _ { t } ^ { + } .
$$

The algorithm also maintains an ellipsoidal outer approximation

$$
S _ { t } ^ { + } \subseteq \mathcal { E } _ { t } .
$$

As in the multiscale algorithm, the relaxed feasible set determines the LEP price, while the ellipsoid determines whether exploration is necessary.

The empty-set restart in Algorithm 3 is a technical form of inconsistency reset: if the relaxed feedback constraints become mutually infeasible, the current batch is discarded. As shown below, such a restart, like a rejected LEP round, can be charged to at least δ variation within the batch.

Theorem C.1 (Single-scale contextual regret). For $4 d \delta \leq \eta < 2 0 R ( d + 1 )$ , Algorithm 3 has dynamic regret

$$
\mathcal { R } _ { T } \leq O ( d ^ { 2 } \log \frac { 2 0 R ( d + 1 ) } { \eta } ) \left( \frac { T } { a } + \frac { V } { \delta } + 1 \right) + T \eta + a \frac { V } { \delta } .
$$

with the convention $a = T$ when $V = 0$ . With the parameter choices

$$
a = \operatorname* { m i n } \{ T , \lceil d ^ { 1 / 2 } \left( \frac { T } { V } \right) ^ { 1 / 3 } \rceil \} , \qquad \delta = \operatorname* { m a x } \{ \frac { 1 } { T } , d ^ { 1 / 2 } \left( \frac { V } { T } \right) ^ { 1 / 3 } \} , \qquad \eta = 4 d \delta ,
$$

the regret is

$$
\mathcal { R } _ { T } \leq O \Big ( ( d ^ { 2 } + d ^ { 3 / 2 } V ^ { 1 / 3 } T ^ { 2 / 3 } ) \log ( T ) \Big ) .
$$

## C.1 Shallow-pricing exploration bound

Lemma C.2 (Shallow-pricing exploration bound). Suppose $4 d \delta \leq \eta < 2 0 R ( d + 1 )$ . Within any fixed batch, the number of exploration rounds in Algorithm 3 is at most

$$
O \left( d ^ { 2 } \log \frac { 2 0 R ( d + 1 ) } { \eta } \right) .
$$

Proof. Consider an exploration round t. By Algorithm 3,

$$
\mathrm { w i d t h } ( \mathcal { E } _ { t } , u _ { t } ) > \eta .
$$

Writing

$$
\mathcal { E } _ { t } = E ( A _ { t } , c _ { t } ) ,
$$

we have

$$
\mathrm { w i d t h } ( \mathcal { E } _ { t } , u _ { t } ) = 2 \sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } ,
$$

and therefore

$$
\sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } > \frac { \eta } { 2 } .
$$

Let

$$
\widetilde { p } _ { t } : = \left. c _ { t } , u _ { t } \right. .
$$

The posted exploration price is

$$
p _ { t } = \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , \widetilde { p } _ { t } \} \} ,
$$

and the ellipsoid is updated using

$$
\begin{array} { r } { \widetilde { f } _ { t } = f _ { t } \mathbb { I } \{ \widetilde { p } _ { t } \leq 1 \} . } \end{array}
$$

By Appendix A, this clipping preserves the ellipsoidal outer-approximation invariant and the validity of the standard shallow update.

The normalized depth of this shallow cut satisfies

$$
\frac { \delta } { \sqrt { u _ { t } ^ { \top } A _ { t } u _ { t } } } < \frac { 2 \delta } { \eta } \leq \frac { 1 } { 2 d } .
$$

Thus every exploration round uses a valid shallow ellipsoid update, and the standard shallow-cut potential argument recalled in Appendix A gives the displayed bound. □

## C.2 Reset count

Lemma C.3 (Reset count). The number of non-window resets in Algorithm 3 is at most $V / \delta .$ Consequently, the total number of batches satisfies

$$
r \leq 1 + \frac { T } { a } + \frac { V } { \delta } .
$$

Proof. First consider a rejected LEP round t. Rejection means

$$
\left. \theta _ { t } , u _ { t } \right. < p _ { t } .
$$

Since the realized valuation is nonnegative, this implies $p _ { t } ~ > ~ 0$ . Hence the clipping at zero is inactive and

$$
p _ { t } = \mathrm { L E P } ( S _ { t } ^ { + } , u _ { t } ) = \operatorname* { m i n } _ { x \in S _ { t } ^ { + } } \left. x , u _ { t } \right. .
$$

It follows that

$$
\theta _ { t } \notin S _ { t } ^ { + } .
$$

Since $S _ { t } ^ { + }$ is the intersection of the $\delta \mathrm { - }$ relaxed feedback halfspaces generated earlier in the same batch, there exists some previous round $j < t$ such that

$$
\theta _ { t } \notin H _ { j } ^ { + } .
$$

At round $j ,$ however, $\theta _ { j }$ satisfied the corresponding exact halfspace $H _ { j }$

If $f _ { j } = 0$ , then

$$
\langle \theta _ { j } , u _ { j } \rangle \leq p _ { j } , \qquad \langle \theta _ { t } , u _ { j } \rangle > p _ { j } + \delta ,
$$

and hence, using $\| u _ { j } \| = 1$ 2

$$
\begin{array} { r } { \| \theta _ { t } - \theta _ { j } \| \geq \langle \theta _ { t } - \theta _ { j } , u _ { j } \rangle > \delta . } \end{array}
$$

If $f _ { j } = 1$ , then

$$
\langle \theta _ { j } , u _ { j } \rangle \geq p _ { j } , \qquad \langle \theta _ { t } , u _ { j } \rangle < p _ { j } - \delta ,
$$

and similarly

$$
\lVert { \boldsymbol { \theta } } _ { t } - { \boldsymbol { \theta } } _ { j } \rVert > \delta .
$$

Thus every rejected-LEP reset certifies more than $\delta$ variation within the batch.

The same conclusion holds if a batch is restarted because $S _ { t } ^ { + } = \emptyset$ . Suppose that $S _ { t } ^ { + } = \varnothing$ at the beginning of round t. If the variation diameter of the completed batch were at most $\delta ,$ then for every earlier round $j$ in that batch, $\theta _ { t - 1 }$ would satisfy the δ-relaxed feedback halfspace generated at round $j .$ . It also satisfies the relaxed halfspace generated at round $t - 1$ . Hence $\theta _ { t - 1 } \in S _ { t } ^ { + }$ contradicting $S _ { t } ^ { + } = \emptyset$

Since diferent reset batches are disjoint intervals and the total variation budget satisfies

$$
\mathrm { T V } ( \Theta ) \leq V ,
$$

there can be at most $V / \delta$ non-window resets. The hard window cap contributes at most $T / a$ additional batches, plus one final incomplete batch. □

## C.3 Proof of Theorem C.1

Proof. Let $B _ { i }$ denote the i-th batch and define its variation diameter by

$$
\Delta _ { i } : = \operatorname* { m a x } _ { j , k \in B _ { i } } \left\| \theta _ { j } - \theta _ { k } \right\| .
$$

We split the batches into good batches, for which $\Delta _ { i } \leq \delta$ , and bad batches, for which $\Delta _ { i } > \delta _ { }$

First consider a good batch. For every round t in this batch and every previous round $j < t$ in the same batch, the exact feedback halfspace generated at $j$ contains $\theta _ { j }$ . Since

$$
\lVert { \boldsymbol { \theta } } _ { t } - { \boldsymbol { \theta } } _ { j } \rVert \leq \Delta _ { i } \leq \delta ,
$$

the current parameter $\theta _ { t }$ satisfies the corresponding δ-relaxed halfspace. Therefore

$$
\theta _ { t } \in S _ { t } ^ { + }
$$

throughout the batch meaning that every LEP round in a good batch is accepted

$$
p _ { t } \le \langle \theta _ { t } , u _ { t } \rangle
$$

Now consider an LEP exploitation round t in this good batch. Since the round is not exploratory,

$$
\mathrm { w i d t h } ( \mathcal { E } _ { t } , u _ { t } ) \leq \eta .
$$

Because

$$
S _ { t } ^ { + } \subseteq \mathcal { E } _ { t } ,
$$

we have

$$
\mathrm { w i d t h } ( S _ { t } ^ { + } , u _ { t } ) \leq \mathrm { w i d t h } ( \mathcal { E } _ { t } , u _ { t } ) \leq \eta .
$$

The LEP price is

$$
p _ { t } \ge \operatorname* { m i n } _ { x \in S _ { t } ^ { + } } \left. x , u _ { t } \right. .
$$

Since $\theta _ { t } \in S _ { t } ^ { + }$ and the realized valuation is nonnegative, the clipped LEP price cannot exceed the current valuation, and hence

$$
\begin{array} { c l l } { \left. \theta _ { t } , u _ { t } \right. - \displaystyle { p _ { t } } \leq \operatorname* { m a x } _ { x \in S _ { t } ^ { + } } \left. x , u _ { t } \right. - \displaystyle { \operatorname* { m i n } _ { x \in S _ { t } ^ { + } } \left. x , u _ { t } \right. } } \\ { \displaystyle { = \mathrm { w i d t h } ( S _ { t } ^ { + } , u _ { t } ) \leq \eta . } } \end{array}
$$

Thus every accepted LEP round in a good batch incurs regret at most $\eta .$

Since exploration prices are clipped to [0, 1], exploration rounds incur regret at most 1. By Lemma C.2, each batch contains only ${ \cal \tilde { O } } ( d ^ { 2 } )$ exploration rounds. Hence every good batch $B _ { i }$ satisfies

$$
R _ { i } \leq O ( d ^ { 2 } \log \frac { 2 0 R ( d + 1 ) } { \eta } ) + | B _ { i } | \eta .
$$

For a bad batch, we use the crude bound

$$
R _ { i } \leq | B _ { i } | \leq a .
$$

Since every bad batch satisfies $\Delta _ { i } > \delta$ and the batches are disjoint, the number of bad batches is at most $V / \delta$

By Lemma C.3, the total number of batches is at most

$$
r \leq 1 + \frac { T } { a } + \frac { V } { \delta } .
$$

Therefore

$$
\mathcal { R } _ { T } \leq O ( d ^ { 2 } \log \frac { 2 0 R ( d + 1 ) } { \eta } ) \left( \frac { T } { a } + \frac { V } { \delta } + 1 \right) + T \eta + a \frac { V } { \delta } .
$$

For notational simplicity, we ignore the ceiling in a in the tuning calculation below. Since the unrounded quantity is at least one, this changes the bounds only by an absolute constant factor. With

$$
a = \operatorname* { m i n } \{ T , d ^ { 1 / 2 } \left( \frac { T } { V } \right) ^ { 1 / 3 } \} , \qquad \delta = \operatorname* { m a x } \{ \frac { 1 } { T } , d ^ { 1 / 2 } \left( \frac { V } { T } \right) ^ { 1 / 3 } \} , \qquad \eta = 4 d \delta ,
$$

If $\eta \geq 2 0 R ( d { + } 1 )$ , the claimed tuned bound follows from the trivial regret bound Regret $| \mathrm { A L G } ] \leq$ $T ,$ since

$$
d ^ { 3 / 2 } V ^ { 1 / 3 } T ^ { 2 / 3 } = \frac { \eta T } { 4 } \geq 5 R ( d + 1 ) T .
$$

Hence it sufices to consider $\eta < 2 0 R ( d + 1 )$ , where Lemma C.2 applies.

One may introduce

$$
Q : = V ^ { 1 / 3 } T ^ { 2 / 3 } ,
$$

and see that

$$
{ \frac { T } { a } } \leq 1 + { \frac { Q } { \sqrt { d } } } , \qquad { \frac { V } { \delta } } \leq { \frac { Q } { \sqrt { d } } } ,
$$

Then

$$
T \eta \leq 4 d + 4 d ^ { 3 / 2 } Q , \qquad a \frac { V } { \delta } \leq Q ,
$$

Overall proving

$$
\mathcal { R } _ { T } \leq O \Bigl ( ( d ^ { 2 } + d ^ { 3 / 2 } V ^ { 1 / 3 } T ^ { 2 / 3 } ) \log ( T ) \Bigr ) ,\tag{1}
$$

as claimed.

We next show that the selected single-scale tuning can incur $\Omega ( V ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret even on a stationary instance.

Proposition C.4 (Lower bound for the selected single-scale tuning). Consider the single-scale shallow pricing algorithm in dimension $d = 2$ , with parameters

$$
\delta = \operatorname* { m a x } \left\{ \frac { 1 } { T } , \sqrt { 2 } \left( \frac { V } { T } \right) ^ { 1 / 3 } \right\} , \qquad a = \operatorname* { m i n } \left\{ T , \left\lceil \sqrt { 2 } \left( \frac { T } { V } \right) ^ { 1 / 3 } \right\rceil \right\} , \qquad \eta = 8 \delta ,
$$

where $V \in ( 0 , T ]$ . At each restart, initialize the relaxed feasible set to

$$
S _ { 0 } = \{ x \in \mathbb { R } ^ { 2 } : x \geq 0 , ~ \| x \| \leq 1 \}
$$

and the ellipsoid to

$$
\mathcal { E } _ { 0 } = \{ x \in \mathbb { R } ^ { 2 } : \| x \| \leq 1 \} .
$$

Suppose each batch contains at most a rounds.

There exists a stationary instance with identical contexts on which the algorithm incurs dynamic regret of at least

$$
{ \frac { 3 } { 8 { \sqrt { 2 } } } } V ^ { 1 / 3 } T ^ { 2 / 3 } .
$$

The same lower bound holds when exploration prices are clipped to [0, 1].

Proof. For every round $t \in [ T ]$ , set

$$
u _ { t } = u = \left( { \frac { 1 } { \sqrt 2 } } , { \frac { 1 } { \sqrt 2 } } \right) , \qquad \theta _ { t } = \theta = \left( { \frac { 3 } { 4 \sqrt 2 } } , { \frac { 3 } { 4 \sqrt 2 } } \right) .
$$

These contexts and parameters satisfy the model assumptions, and every buyer has value

$$
v _ { t } = \langle \theta , u \rangle = { \frac { 3 } { 4 } } .
$$

Moreover, the parameter sequence is stationary, so its total variation is zero and hence does not exceed $V .$

Consider the first round of any batch. If the algorithm explores, it posts the midpoint of the projection of ${ \mathcal { E } } _ { 0 }$ onto u, namely

$$
\left. 0 , u \right. = 0 .
$$

This price is accepted and incurs regret $3 / 4$ . If the algorithm instead exploits, it posts the lowerenvelope price of $S _ { 0 }$ in direction $u ,$ which is

$$
\operatorname* { m a x } \left\{ 0 , \operatorname* { m i n } _ { x \in S _ { 0 } } \left. x , u \right. \right\} = 0 .
$$

This also incurs regret $3 / 4$ . Thus, regardless of whether the algorithm initially explores or exploits, the first round of each batch incurs $3 / 4$ regret. Clipping exploration prices to [0, 1] does not change this conclusion, since the initial exploration price is 0.

Let r denote the number of batches. Since each batch contains at most a rounds,

$$
r \geq { \frac { T } { a } } .
$$

All other rounds incur nonnegative regret, so

$$
\mathrm { R E G R E T } [ \mathrm { A L G } ] \geq \frac { 3 r } { 4 } \geq \frac { 3 T } { 4 a } .
$$

Since $V \leq T$

$$
{ \sqrt { 2 } } \left( { \frac { T } { V } } \right) ^ { 1 / 3 } \geq { \sqrt { 2 } } > 1 .
$$

Therefore,

$$
a \leq \left\lceil { \sqrt { 2 } } \left( { \frac { T } { V } } \right) ^ { 1 / 3 } \right\rceil \leq 2 { \sqrt { 2 } } \left( { \frac { T } { V } } \right) ^ { 1 / 3 } .
$$

Consequently,

$$
\mathrm { R E G R E T } [ \mathrm { A L G } ] \geq \frac { 3 } { 8 \sqrt { 2 } } V ^ { 1 / 3 } T ^ { 2 / 3 } ,
$$

as claimed.

This lower bound matches the $V ^ { 1 / 3 } T ^ { 2 / 3 }$ dependence of the stated upper bound for the exact tuning of Algorithm 3, up to logarithmic factors when the dimension is fixed.

In particular, when $V = 1$ , this tuning incurs $\Omega ( T ^ { 2 / 3 } )$ regret even on a stationary instance. The result concerns the selected parameter choice; it does not rule out improved guarantees under a diferent tuning of the single-scale algorithm. □

## D Multiscale Contextual Analysis

This appendix gives the complete analysis of Algorithm 2. We first establish the nesting and containment properties of the relaxed feasible sets, then bound the exploration complexity and LEP regret, and finally combine these ingredients to prove Theorem 4.2.

## D.1 Structural properties of the relaxed sets

Recall the dyadic scale set

$$
\Lambda = \{ \delta , 2 \delta , \ldots , 2 ^ { L } \delta \} ,
$$

where $2 ^ { L } \delta \geq R .$

Lemma D.1 (Nesting across scales). $I f \lambda \leq \lambda ^ { \prime }$ , then for every round t within a batch,

$$
S _ { t } ^ { ( \lambda ) } \subseteq S _ { t } ^ { ( \lambda ^ { \prime } ) } .
$$

Consequently, the active set

$$
\Lambda _ { t } = \{ \lambda \in \Lambda : S _ { t } ^ { ( \lambda ) } \neq \emptyset \}
$$

is a sufix of Λ.

Proof. For each observed feedback sign, increasing the relaxation radius only enlarges the corresponding halfspace:

$$
H _ { j } ^ { ( \lambda ) } \subseteq H _ { j } ^ { ( \lambda ^ { \prime } ) } \qquad \mathrm { w h e n e v e r ~ } \lambda \leq \lambda ^ { \prime } .
$$

Intersecting these halfspaces over previous rounds preserves the inclusion. The sufix property follows immediately. □

Lemma D.2 (Existence of an active scale). For every round t within a batch,

$$
\Lambda _ { t } \neq \varnothing .
$$

In particular,

$$
\lambda _ { t } ^ { \mathrm { m i n } } = \operatorname* { m i n } \Lambda _ { t }
$$

is well-defined.

Proof. Let

$$
\lambda _ { \operatorname* { m a x } } = 2 ^ { L } \delta \geq R .
$$

Fix any current round t and any previous round $j < t$ in the same batch. The parameter $\theta _ { j }$ satisfies the exact feedback halfspace generated at round j. Since both $\theta _ { j }$ and $\theta _ { t }$ lie in $S _ { 0 }$

$$
\begin{array} { r } { \| \theta _ { t } - \theta _ { j } \| \leq \mathrm { d i a m } ( S _ { 0 } ) \leq R \leq \lambda _ { \operatorname* { m a x } } . } \end{array}
$$

Using $\| u _ { j } \| = 1$ , the current parameter $\theta _ { t }$ therefore satisfies the $\lambda _ { \operatorname* { m a x } } .$ -relaxed version of the feedback halfspace from round $j .$ . This holds for every previous round, so

$$
\theta _ { t } \in S _ { t } ^ { ( \lambda _ { \operatorname* { m a x } } ) } .
$$

Hence the largest scale is active.

## D.2 Containment at the retrospective right scale

For a batch $B _ { i }$ define its variation diameter by

$$
\Delta _ { i } : = \operatorname* { m a x } _ { j , k \in B _ { i } } \left\| \theta _ { j } - \theta _ { k } \right\| ,
$$

and define the retrospective right scale

$$
\lambda _ { i } : = \operatorname* { m i n } \{ \lambda \in \Lambda : \lambda \geq \Delta _ { i } \} .
$$

This scale exists because

$$
\Delta _ { i } \leq \dim ( S _ { 0 } ) \leq R .
$$

Lemma D.3 (Multiscale containment). For every $t \in B _ { i }$ and every $s \in B _ { i }$ with $s \leq t _ { i }$

$$
\theta _ { s } \in S _ { t } ^ { ( \lambda _ { i } ) } .
$$

Moreover,

$$
\lambda _ { i } \leq 2 \operatorname* { m a x } \{ \Delta _ { i } , \delta \} .
$$

Proof. Fix $s \leq t$ and a previous round $j < t$ in the same batch. If $f _ { j } = 0$ , then

$$
\langle \theta _ { j } , u _ { j } \rangle \leq p _ { j } .
$$

Because

$$
\begin{array} { r } { \| { \theta } _ { s } - { \theta } _ { j } \| \leq \Delta _ { i } \leq \lambda _ { i } , } \end{array}
$$

we obtain

$$
\begin{array} { r } { \langle \theta _ { s } , u _ { j } \rangle \le \langle \theta _ { j } , u _ { j } \rangle + \| \theta _ { s } - \theta _ { j } \| \le p _ { j } + \lambda _ { i } . } \end{array}
$$

Thus $\theta _ { s }$ satisfies $H _ { i } ^ { ( \lambda _ { i } ) }$

The case $f _ { j } = 1$ is symmetric:

$$
\begin{array} { r } { \langle \theta _ { s } , u _ { j } \rangle \geq \langle \theta _ { j } , u _ { j } \rangle - \| \theta _ { s } - \theta _ { j } \| \geq p _ { j } - \lambda _ { i } . } \end{array}
$$

Hence $\theta _ { s }$ satisfies every relaxed halfspace defining $S _ { t } ^ { ( \lambda _ { i } ) }$

The dyadic definition of $\lambda _ { i }$ gives

$$
\lambda _ { i } < 2 \Delta _ { i } \qquad \mathrm { w h e n e v e r } \ \Delta _ { i } \geq \delta ,
$$

while

$$
\lambda _ { i } = \delta \qquad \mathrm { w h e n } ~ \Delta _ { i } < \delta .
$$

Therefore

$$
\lambda _ { i } \leq 2 \operatorname* { m a x } \{ \Delta _ { i } , \delta \} .
$$

Corollary D.4 (The right scale is active). For every $t \in B _ { i }$

$$
\lambda _ { i } \in \Lambda _ { t } \qquad a n d ~ h e n c e \qquad \lambda _ { t } ^ { \operatorname* { m i n } } \leq \lambda _ { i } .
$$

Proof. Lemma D.3 gives

$$
\theta _ { t } \in S _ { t } ^ { ( \lambda _ { i } ) } ,
$$

so the set is nonempty and $\lambda _ { i }$ is active. The second claim follows from the definition of $\lambda _ { t } ^ { \operatorname* { m i n } }$

## D.3 Exploration complexity

Lemma D.5 (Multiscale exploration bound). Suppose $V \leq T / d$ . Within any fixed batch, $A l g o \cdot$ rithm 2 performs at most

$$
2 0 d ^ { 2 } ( L + 1 ) \ln \biggl ( \frac { 5 R ( d + 1 ) } { d \delta } \biggr )
$$

exploration rounds.

Proof. Fix a scale $\lambda \in \Lambda$ . The algorithm selects this scale only when it is active and

$$
\mathrm { w i d t h } ( { \mathcal E } _ { t } ^ { ( \lambda ) } , u _ { t } ) > 4 d \lambda .
$$

Whenever it is selected, only $\mathcal { E } _ { t } ^ { ( \lambda ) }$ is shallow-updated using the standard λ-shifted shallow cut associated with the unclipped midpoint proposal. By Appendix A, this update remains a valid outer approximation when the posted exploration price is clipped at $[ 0 , 1 ]$ . Since $V \leq T / d .$ , we have $\delta \leq 1 < R$ . By the minimality of L, every $\lambda \in \Lambda$ satisfies $\lambda < 2 R$ . Hence

$$
4 d \lambda < 8 d R < 2 0 R ( d + 1 ) ,
$$

so Lemma A.1 applies with $\eta = 4 d \lambda$ , bounding the number of times this scale can be selected by

$$
2 0 d ^ { 2 } \ln \left( \frac { 2 0 R ( d + 1 ) } { 4 d \lambda } \right) \leq 2 0 d ^ { 2 } \ln \left( \frac { 5 R ( d + 1 ) } { d \delta } \right) ,
$$

where we used $\lambda \geq \delta .$ . Summing over the $L + 1$ dyadic scales proves the claim.

## D.4 Accepted-LEP regret

Lemma D.6 (Accepted LEP regret at the right scale). Consider an accepted LEP round t in batch $B _ { i }$ . Then

$$
\langle \theta _ { t } , u _ { t } \rangle - p _ { t } \leq 8 d \operatorname* { m a x } \{ \Delta _ { i } , \delta \} .
$$

Proof. By Corollary D.4, $\lambda _ { i }$ is active and

$$
\lambda _ { t } ^ { \mathrm { m i n } } \leq \lambda _ { i } .
$$

Since round t is an LEP round, no active scale violates its width threshold. In particular,

$$
\mathrm { w i d t h } ( { \mathcal E } _ { t } ^ { ( \lambda _ { i } ) } , u _ { t } ) \le 4 d \lambda _ { i } .
$$

Using

$$
{ S } _ { t } ^ { ( \lambda _ { i } ) } \subseteq { \mathcal { E } _ { t } ^ { ( \lambda _ { i } ) } } ,
$$

we also have

$$
\mathrm { w i d t h } ( S _ { t } ^ { ( \lambda _ { i } ) } , u _ { t } ) \leq 4 d \lambda _ { i } .
$$

By Lemma D.1,

$$
S _ { t } ^ { ( \lambda _ { t } ^ { \operatorname* { m i n } } ) } \subseteq S _ { t } ^ { ( \lambda _ { i } ) } .
$$

Therefore, by monotonicity of $z \mapsto \operatorname* { m a x } \{ 0 , z \}$ ，

$$
\begin{array} { r l } & { p _ { t } = \mathrm { L E P } \big ( S _ { t } ^ { ( \lambda _ { t } ^ { \mathrm { m i n } } ) } , u _ { t } \big ) } \\ & { \quad \ge \mathrm { L E P } \big ( S _ { t } ^ { ( \lambda _ { i } ) } , u _ { t } \big ) } \\ & { \quad \ge \displaystyle \operatorname* { m i n } _ { x \in S _ { t } ^ { ( \lambda _ { i } ) } } \langle x , u _ { t } \rangle . } \end{array}
$$

Lemma D.3 gives

$$
\theta _ { t } \in S _ { t } ^ { ( \lambda _ { i } ) } .
$$

Since the LEP round is accepted,

$$
\begin{array} { r l } & { \langle \theta _ { t } , u _ { t } \rangle - p _ { t } \leq \langle \theta _ { t } , u _ { t } \rangle - \underset { x \in S _ { t } ^ { ( \lambda _ { i } ) } } { \mathrm { m i n } } \langle x , u _ { t } \rangle } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \leq \mathrm { w i d t h } ( S _ { t } ^ { ( \lambda _ { i } ) } , u _ { t } ) } \\ & { \quad \quad \quad \quad \leq 4 d \lambda _ { i } } \\ & { \quad \quad \quad \quad \quad \quad \leq 8 d \operatorname* { m a x } \{ \Delta _ { i } , \delta \} . } \end{array}
$$

## D.5 Reset count

Lemma D.7 (Multiscale reset count). The number of LEP-rejection resets in Algorithm 2 is at most

$$
{ } _ { \frac { V } { \delta } . } ^ { V }
$$

Consequently, the total number of batches satisfies

$$
r \leq 1 + \frac { T } { a } + \frac { V } { \delta } .
$$

Proof. Consider a rejected LEP round t. Rejection means

$$
\left. \theta _ { t } , u _ { t } \right. < p _ { t } .
$$

Since the realized valuation is nonnegative, we must have $p _ { t } > 0$ . Hence the clipping at zero is inactive, and therefore

$$
p _ { t } = \mathrm { L E P } \big ( S _ { t } ^ { ( \lambda _ { t } ^ { \mathrm { m i n } } ) } , u _ { t } \big ) = \operatorname* { m i n } _ { x \in S _ { t } ^ { ( \lambda _ { t } ^ { \mathrm { m i n } } ) } } \left. x , u _ { t } \right. .
$$

It follows that

$$
{ \boldsymbol { \theta } } _ { t } \not \in { \cal S } _ { t } ^ { ( \lambda _ { t } ^ { \operatorname* { m i n } } ) } .
$$

Hence $\theta _ { t }$ violates some earlier $\lambda _ { t } ^ { \operatorname* { m i n } }$ -relaxed feedback halfspace generated at a round $j$ in the same batch, while $\theta _ { j }$ satisfied the corresponding exact halfspace.

If $f _ { j } = 0$ , then

$$
\begin{array} { r } { \langle \theta _ { j } , u _ { j } \rangle \le p _ { j } , \qquad \langle \theta _ { t } , u _ { j } \rangle > p _ { j } + \lambda _ { t } ^ { \operatorname* { m i n } } , } \end{array}
$$

so

$$
\| \theta _ { t } - \theta _ { j } \| > \lambda _ { t } ^ { \operatorname* { m i n } } \geq \delta .
$$

The case $f _ { j } = 1$ is symmetric. Thus every LEP-rejection reset certifies more than $\delta$ variation within its batch.

Since reset batches are disjoint and the total variation budget is at most $V ,$ there can be at most $V / \delta$ such resets. Window-size resets contribute at most $T / a$ additional batches, plus one final batch. □

## D.6 Proof of Theorem 4.2

Proof. If $V > T / d _ { \ast }$ the claimed bound follows from the trivial regret bound Regret $[ \mathrm { A L G } ] \leq T$ since

$$
d ^ { 3 / 2 } { \sqrt { T V } } \geq T .
$$

Hence, for the remainder of the proof, it sufices to consider $V \leq T / d .$ . Let

$$
B _ { 1 } , \ldots , B _ { r }
$$

be the batches. Since the batches are disjoint intervals,

$$
\sum _ { i = 1 } ^ { r } \Delta _ { i } \leq \mathrm { T V } ( \Theta ) \leq V .
$$

We decompose regret into exploration rounds, accepted LEP rounds, and rejected LEP rounds.

Since exploration prices are nonnegative and realized valuations lie in [0, 1], each exploration round incurs regret at most 1. Then by Lemma D.5, exploration contributes at most

$$
2 0 d ^ { 2 } ( L + 1 ) \ln \left( \frac { 5 R ( d + 1 ) } { d \delta } \right) r .
$$

For accepted LEP rounds in batch $B _ { i }$ , Lemma D.6 gives per-round regret at most

$$
8 d \operatorname* { m a x } \{ \Delta _ { i } , \delta \} \leq 8 d ( \Delta _ { i } + \delta ) .
$$

Since

$$
| B _ { i } | \leq a ,
$$

summing over batches yields

$$
\begin{array} { l } { \mathrm { a c c e p t e d \mathrm { - } L E P ~ r e g r e t } \leq 8 d a \displaystyle \sum _ { i = 1 } ^ { r } \Delta _ { i } + 8 d \delta \sum _ { i = 1 } ^ { r } | B _ { i } | } \\ { \leq 8 d a V + 8 d \delta T . } \end{array}
$$

By Lemma D.7, there are at most $V / \delta$ rejected LEP rounds, each with regret at most 1. Moreover,

$$
r \leq 1 + { \frac { T } { a } } + { \frac { V } { \delta } } .
$$

Therefore

$$
\begin{array} { c } { { \displaystyle \mathcal { R } _ { T } \leq 2 0 d ^ { 2 } ( L + 1 ) \ln \bigg ( \frac { 5 R ( d + 1 ) } { d \delta } \bigg ) \left( 1 + \frac { T } { a } + \frac { V } { \delta } \right) } } \\ { { \displaystyle + 8 d a V + 8 d \delta T + \frac { V } { \delta } . } } \end{array}
$$

For notational simplicity, in the tuning calculation below we ignore the ceiling in the definition of a. This afects the bounds only by an absolute constant factor: for $x \geq 1$ ，

$$
x \leq \lceil x \rceil \leq 2 x ,
$$

and the additional truncation by $T$ preserves this constant-factor relation. Thus replacing min $\{ T , \lceil \sqrt { d T / V } \rceil \}$ by min $\{ T , \sqrt { d T / V } \}$ does not afect the stated big-O regret bound. Choose

$$
a = \operatorname * { m i n } \{ T , \sqrt { \frac { d T } { V } } \} , \qquad \delta = \operatorname * { m a x } \{ \frac { 1 } { T } , \sqrt { \frac { d V } { T } } \} .
$$

Then

$$
{ \frac { T } { a } } \leq 1 + { \sqrt { \frac { V T } { d } } } , \qquad { \frac { V } { \delta } } \leq { \sqrt { \frac { V T } { d } } } ,
$$

and

$$
d a V \le d ^ { 3 / 2 } \sqrt { V T } , \qquad d \delta T \le d + d ^ { 3 / 2 } \sqrt { V T } .
$$

Finally,

$$
L + 1 = O \left( 1 + \log { \frac { R } { \delta } } \right) ,
$$

Hence

$$
\mathcal { R } _ { T } = O \Big ( ( d ^ { 2 } + d ^ { 3 / 2 } \sqrt { V T } ) \log ^ { 2 } ( T ) \Big ) .
$$