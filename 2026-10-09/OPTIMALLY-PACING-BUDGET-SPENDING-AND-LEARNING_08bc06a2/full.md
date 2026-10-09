# OPTIMALLY PACING BUDGET SPENDING AND LEARNING

Mark Braverman<sup>∗1,2</sup>, Jingyi Liu<sup>∗1</sup>, Jieming Mao<sup>2</sup>, Jon Schneider<sup>2</sup>, Eric Xue<sup>∗1</sup>

<sup>1</sup>Department of Computer Science, Princeton University <sup>2</sup>Google Research

## ABSTRACT

We establish near-optimal regret bounds for budget-constrained online learning against arbitrary classes of budget-pacing experts in the adversarial setting. In particular, given any class of F experts and a candidate budget pacing schedule, we provide a full-information <sub>algorithm which obtains regret O(D</sub>√<sub>log F +</sub> √<sub>T log F) against all experts whose cumula-</sub> tive spending stays within distance D of this schedule, matching lower bounds established by Braverman et al. (2025).

We additionally show that our technique extends to various problems in online resource allocation, where the learner gets to see the rewards and costs of the current options available to them, and establish O(D log F) regret bounds when fractional allocation is allowed. <sub>This is the first algorithm we are aware of which can achieve o(</sub>√<sub>T) guarantees for such</sub> tasks.

Keywords Online Learning with Resource Constraints, Online Resource Allocation, Spend-or-Save Dilemma

## 1 Introduction

Consider first the classical full-information online learning problem with experts. There are N actions, F experts and T rounds in total. In each round, the learner gets an advice distribution over the set of N actions from each expert, and then they choose an action to take. Each action is associated with some bounded reward, unknown to the learner prior to decision. In the adversarial model, the reward can be chosen by an adaptive adversary who might know the learner’s algorithm and the transcript of play but does not know the learner’s private randomness in the current round. The learner’s performance is measured in terms of their regret against the best expert, i.e. the difference between the cumulative rewards earned by the best expert and the cumulative reward earned by the learner. This setting is well understood and the celebrated Multiplicative Weights Update (MWU) algorithm is known to achieve an optimal regret guarantee of O( T log F)[Cesa-Bianchi and Lugosi, 2006].

In many real-world applications, such as repeated auctions with budgets, drone control, and resource allocation, playing an action not only yields a reward but also incurs some cost. For example, in a repeated auction, the action may be a bid, the reward may correspond to winning an item, and the cost is the resulting payment. This motivates the study of online learning problems with resource constraints (OLRC) [Stradi et al., 2025], where each action additionally consumes a bounded resource cost, and the learner is endowed with a fixed budget of this resource and cannot overspend. Once the budget is depleted, the learner must stop and only take the null action going forward, which gives zero reward and incurs zero cost. The learner’s goal is thus to minimize regret against the best expert while maintaining budget feasibility.

In contrast to the online learning problem without resource constraints, where taking any action in the current round does not affect the learner’s ability to take any action in a future round, the online learning problem with resource constraints forces a learner to only choose the null action once their budget is depleted, which introduces a temporal coupling of the learner’s decision sets across the time horizon. Moreover, different experts may have very different spending patterns over time. The learner therefore cannot freely change its mixture over experts across rounds: even if every expert is individually budget-feasible, switching among them can cause the learner itself to violate the budget constraint. These differences with the classical setting highlight the difficulty in designing an optimal learning algorithm for the setting with resource constraints: one must simultaneously 1) learn which action is better and 2) decide what mixture of experts to follow that will guarantee both no-regret and budget feasibility.

In fact, this problem is already non-trivial even without the first difficulty. In the online resource allocation (ORA) setting [Stradi et al., 2025, Balseiro et al., 2023], the current round’s reward and resource costs of each action (and thereby of following each expert) are known to the learner before they need to make a decision, so the learner doesn’t need to $\mathrm { \ " { g u e s s } " }$ which action/expert gets a better reward or a better reward to cost ratio for the current round. Thus ORA removes uncertainty about the current round, but not the intertemporal allocation problem: the learner still does not know whether budget spent today should instead be saved for better opportunities in the future.

The second difficulty is therefore best summarized by the “spend-or-save" dilemma, present in both the ORA setting [Balseiro and Gur, 2019] and the OLRC setting [Immorlica et al., 2022]. In the OLRC setting, it shows that the classical benchmark of competing against a strategy that plays a fixed distribution until the budget is depleted is no longer possible, and any algorithm that satisfies the budget constraint must incur a worst-case regret of $\Omega ( T )$ . Braverman et al. [2025] characterized the learnability of the OLRC problem with experts in terms of the variability of their hindsight spending patterns: sublinear regret is attainable if and only if the experts are close to a common subpacing spending pattern in unnormalized Earth $\mathbf { M o v e r { \bar { s } } }$ Distance (EMD), with EMD radius $o ( T ^ { 2 } )$ . Their lower bound construction is, in spirit, very close to the original “spend-or-save" dilemma, where even with two experts, if their spending patterns $c , c ^ { \prime } \in [ 0 , 1 ] ^ { T }$ differ by $\mathrm { \dot { E M D } } ( c , c ^ { \prime } ) = \Omega ( T ^ { 2 } )$ (for example, one expert spends all budget $B = { \bar { T } } { \bar { / 2 } }$ in the first half, and the other expert spends all budget $B = T / 2$ in the second half), then any algorithm must incur a worst-case regret of $\Omega ( \bar { T } )$ on some instantiation of reward and cost sequence and expert advice distributions. However, for general $\mathrm { \ E M D } ( c , c ^ { \prime } )$ , their lower bound construction only gives a bound of $\Omega ( \mathrm { E M D } ( c , c ^ { \prime } ) / T )$ , which is below their algorithmic upper bound of $\tilde { O } ( \sqrt { \mathrm { E M D } ( c , c ^ { \prime } ) } + \sqrt { T \log F } )$ whenever $\mathrm { E M D } ( c , c ^ { \prime } ) = o ( T ^ { 2 } )$

Thus the “spend-or-save" dilemma and prior work on learnable benchmarks highlight that the key to quantifying learnability is through controlling the variability of the experts’ spending patterns. Yet, to the best of our knowledge, existing results that propose new benchmarks do not have matching upper and lower bounds [Braverman et al., 2025, Stradi et al., 2025]. This motivates us to ask the question: can we tightly characterize the regret rate in learning settings with resource constraints?

## 1.1 A Toy Example: Alternating Online Knapsack

The above question is already non-trivial in very simple settings. To see this, consider the following toy problem. You have an incoming stream of $T$ balls whose colors alternate between red and blue (so every odd numbered ball is red, and every even numbered ball is blue). The learner processes the balls one by one: in round $t ,$ they observe the reward $r _ { t } \in [ 0 , 1 ]$ of ball t and have to decide what fraction of it to pick up. The learner can pick up at most $T / 2$ balls and wants low regret relative to either picking all the red balls or picking all the blue balls. How small can this regret be?

Note that this maps cleanly onto the online resource allocation (ORA) problem defined above: each ball has unit cost, and each color of ball (red and blue) corresponds to a different expert. The red expert has spending pattern $( 1 , 0 , 1 , 0 , \ldots )$ , the blue expert has spending pattern $( 0 , 1 , 0 , 1 , \ldots { \hat { ) } }$ , and they are both within EMD distance $O ( T )$ of the uniform pacing pattern $( 1 / 2 , \bar { 1 / 2 } , \ldots )$ . The existing results of Braverman et al. [2025] yield an $O ( \sqrt { \mathrm { E M D } } ) = O ( \sqrt { T } )$ regret algorithm against these experts, and indeed a simple variant of MWU (treating each pair of balls as a single round with a red arm and a blue arm) achieves this regret bound, even when the reward is not observable until the end of the round. On the other hand, the best lower bound construction only gives a lower bound of $\Omega ( \mathrm { E M D } / T ) = \Omega ( 1 )$

We show that in this case the lower bound is tight: there exists an algorithm that achieves constant regret for this problem. The key idea is somewhat different from the usual Lagrangian treatment of a budget constraint. A standard approach would attach a dual price to deviation from the fixed pacing budget and optimize reward minus priced resource consumption. But regret is measured relative to one of the two experts, not relative to this fixed pacing baseline. Our algorithm instead measures both reward regret and spending discrepancy relative to the same mixture of the red and blue experts. It then chooses the current action by solving a small minimax problem over this comparator mixture and a resource-price variable. This lets a single potential control both quantities and, in this example, yields constant regret.

More generally, we can get similar results for the problem where there are alternating blocks of D red balls and D blue balls, for a total of $T$ rounds. Here the EMD-based regret bounds give a regret of $O \Big ( D \cdot \sqrt { T / ( 2 D ) } \Big ) = O ( \sqrt { D T } )$ . But the above technique establishes a regret bound of $O ( D + { \sqrt { T } } )$ in the OLRC setting, and a regret bound of only $O ( D )$ in the ORA setting. Note that for all $D = o ( T )$ , this is asymptotically better than the best previously known algorithm for both the OLRC and ORA settings.

## 1.2 Our Results

More generally, we answer the question of how to characterize instance-optimal regret rates in a general class of ORA and OLRC problems. In particular, we show that instead of earth mover’s distance, the more relevant quantity is the largest difference in experts’ cumulative spending over any prefix ofthe time horizon. Formally, consider the following distance between two spending patterns $c , c ^ { \prime } \in [ 0 , 1 ] ^ { T }$

$$
\operatorname { K o l } ( c , c ^ { \prime } ) : = \operatorname* { m a x } _ { t \in [ T ] } \left| \sum _ { s = 1 } ^ { t } ( c _ { s } - c _ { s } ^ { \prime } ) \right| .
$$

We call this quantity the Kolmogorov distance. This captures directly the feature exploited by the lower-bound construction in the spend or save dilemma: if two spending patterns are far apart in earth mover’s distance, then their cumulative spending must differ substantially at some point in time, and it is this cumulativespending gap that drives the resulting spend-or-save lower bound construction. In particular, we will show that across multiple different settings the Kolmogorov distance between spending patterns of experts characterizes the optimal regret achievable. For example, in the toy example above, the Kolmogorov distance between the red spending pattern and the blue spending pattern is 1, explaining the constant regret algorithm.

We therefore fix a baseline spending pattern b and a radius $D ,$ and compete against experts whose spending patterns remain within Kolmogorov distance D of $b ^ { 1 }$ . We call such experts D-pacing. In our main theorem, we show that we can establish a regret bound of either $O ( D { \sqrt { \log F } } )$ (in the ORA setting) or $O ( ( { \sqrt { T } } + D ) { \sqrt { \log F } } )$ (in the OLRC setting) against the performance of the best of these pacing experts, when allocation is allowed to be fractional:

Theorem 1.1 (Theorem 4.6 (simplified)). Fix a time horizon T, a baseline spending pattern $b ,$ and a radius D. Assume there exists a constant $\alpha > 0$ such that, for all times t and actions $^ { a , }$ we have a bounded reward-to-cost ratio<sup>2</sup>

$$
r _ { t } ( a ) \leq \alpha c _ { t } ( a ) .
$$

Suppose there are $F$ experts, each ofwhich is D-paced with respect to the same baseline spending pattern b.

Then there exist (fractional $p l a y ) ^ { 3 }$ online algorithms with the following guarantees:

$$
\bullet \ : o R A \colon \mathrm { R e g } _ { T } = O \bigl ( D \sqrt { \log F } \bigr ) .
$$

$$
\bullet \bullet d R C \colon \operatorname { R e g } _ { T } = O { \Big ( } ( { \sqrt { T } } + D ) { \sqrt { \log F } } { \Big ) } .
$$

These guarantees are optimal up to constant factors. In Section 6, we show an $\Omega ( D { \sqrt { \log F } } )$ lower bound even in the ORA setting. Together with the classical $\Omega ( { \sqrt { T \log F } } )$ experts lower bound, this also yields an $\Omega ( \operatorname* { m a x } \{ D , \sqrt { T } \} \sqrt { \log F } )$ lower bound in the OLRC setting, matching our upper bound up to constant factors.

The algorithms of Theorem 1.1 are flexible and can be extended in many ways: to algorithms with sampled play, to sets of experts which are subpacing<sup>4</sup>, and to settings with multiple resources. We provide a full summary of guarantees in the single-resource setting in Section $\mathbf { A } ;$ the corresponding multiple-resource guarantees are given in Theorem 4.6 and Section 5.

## 1.3 Related Work

## Online Resource Allocation (ORA) & Online Learning with Resource Constraints (OLRC)

In the ORA setting, the typical benchmark is the best-in-hindsight dynamic strategy that satisfies the global budget constraint[Balseiro et al., 2023], and in the OLRC setting, the typical benchmark is the set of strategies that play the same distribution every round until they run out of budget [Immorlica et al., 2022]. In the OLRC setting, when the learner only gets bandit feedback, this is often referred to as the Bandits with Knapsacks problem, initiated by Badanidiyuru et al. [2013] for the stochastic environment and Immorlica et al. [2022] for the adversarial environment. When the rewards and costs are adversarially chosen (the main focus of our paper) there is a $\Omega ( T )$ lower bound in both the ORA and OLRC settings [Balseiro and Gur, 2019, Immorlica et al., 2022]. Thus prior work mostly focuses on achieving competitive ratio guarantees for these settings [Immorlica et al., 2022, Castiglioni et al., 2022a,b, 2024, Balseiro et al., 2023]. Exceptions to these are works that focus on alternative benchmarks [Stradi et al., 2025, Braverman et al., 2025] and works that derive a regret guarantee as a function of some nonstationarity measure of the environment [Slivkins et al., 2023, Liu et al., 2022, Fikioris and Tardos, 2023]. Jenatton et al. [2016] study dynamic regret guarantees for more general non-additive objective functions in the 1-lookahead setting (which is similar to our ORA setting), where the objective can encode a penalty term for long-term constraint violations. Our benchmarks are incomparable: their benchmark is the optimal dynamic sequence for a soft penalized objective, whereas we compete against a set of D-pacing experts under a hard budget constraint.

A recurring theme in both the ORA setting and the OLRC setting is the use of the online primal-dual framework or Lagrangian-based algorithm for developing algorithms [Balseiro et al., 2023, Immorlica et al., 2022, Stradi et al., 2025, Braverman et al., 2025]. The budget constraint is Lagrangified by a dual multiplier, and the primal player makes decisions based on this Lagrangified objective, while the dual player updates the dual multiplier separately, which serves as a penalty for deviations from a uniform per-round budget [Balseiro et al., 2023] or a spending plan [Stradi et al., 2025]. Our algorithm, interestingly, does not follow the same primal-dual template. Instead of using an alternating primal step and dual step, our algorithm optimizes the primal variable and dual variable jointly according to some potential function, which is what allows us to get a T-independent bound in the ORA setting and a decoupled $O ( D + { \sqrt { T } } )$ bound in the OLRC setting against D-pacing experts.

Online Convex Optimization with Constraints There is a vast literature on online convex optimization problems with long-term constraints [Mannor et al., 2009, Mahdavi et al., 2012]. At round t, the learner makes a decision $x _ { t } \in \mathcal { X }$ and gets the full-information feedback of a convex cost function $f _ { t }$ and a constraint function $g _ { t } \leq 0$ . Different from the OLRC setting, which requires hard budget feasibility, this setting allows the algorithm to have constraint violations, and the goal is to minimize regret against some strategy $x ^ { * }$ (with certain feasibility assumptions) and the cumulative constraint violation separately. Mannor et al. [2009] provides an impossibility result for obtaining a vanishing bound in both objectives when $x ^ { * }$ is only required to satisfy $\begin{array} { r } { \sum _ { t } g _ { t } ( \dot { x } ^ { * } ) \le 0 } \end{array}$ . Follow-up work often restricts the benchmark strategy to satisfy $g _ { t } ( x ^ { * } ) \stackrel { } { \leq } 0 , \hat { \forall } t \in [ T ]$ [Neely and Yu, 2017, Sun et al., 2017, Chen and Giannakis, 2018, Sinha and Vaze, 2024], which is analogous to a strict requirement for the benchmark strategies to follow a given spending plan in our setting. A slightly more general benchmark is considered by Liakopoulos et al. [2019], where the benchmark strategy only needs to be feasible for K-sized windows. The objectives of this line of work differ from ours, as it does not hard-stop the algorithm when there is a constraint violation, so the results are incomparable.

## Bidding in Repeated Auctions

Repeated auctions with budget constraints form a canonical single-resource instance of OLRC. Extensive literature employs pacing multipliers to dynamically shade bids and regulate expenditure [Borgs et al., 2007, Balseiro and Gur, 2019, Gaitonde et al., 2023]. While the standard benchmark is the best-in-hindsight multiplier, vanishing regret against this baseline is unattainable in the adversarial setting due to the spend-orsave dilemma [Gaitonde et al., 2023]. Consequently, recent works Gaitonde et al. [2023]; Lucier et al. [2024] consider more relaxed benchmarks—the best sequence of perfect pacing multipliers, where each perfect pacing multiplier makes sure the budget (or ROI) constraint is tight in expectation in that round and achieve no-regret guarantees as a function of the path length of the perfect pacing multipliers. Our benchmark extends beyond strategies that need to satisfy per-round budget and scales smoothly with the variability of spending patterns in the benchmark class.

## 1.4 Comparisons with Prior Work

Braverman et al. [2025] study the adversarial Bandits with Knapsacks setting and they introduce a new benchmark class consisting of experts with spending patterns within some radius of a baseline spending pattern $b ,$ in Earth Mover’s Distance. More precisely, for any two spending patterns $c , c ^ { \prime } \in [ 0 , \stackrel { \cdot } { 1 } ] ^ { T }$ , the unnormalized Earth Mover’s Distance between them is EMD $\begin{array} { r } { \ u { \mathrm { \Sigma } } ^ { \mathrm { \Lambda } } ( c , c ^ { \prime } ) : = \sum _ { t = 1 } ^ { T } \left| \sum _ { s = 1 } ^ { t } ( c _ { s } - c _ { s } ^ { \prime } ) \right| } \end{array}$ . The regret bound they achieve against a benchmark class with radius EMD is $\tilde { O } ( \sqrt { \mathrm { E M D } } + \sqrt { T \log F } )$ . In comparison, we introduce the Kolmogorov distance $\begin{array} { r } { \mathrm { K o l } ( c , c ^ { \prime } ) : = \operatorname* { m a x } _ { t \in [ T ] } \left| \sum _ { s = 1 } ^ { t } ( c _ { s } - c _ { s } ^ { \prime } ) \right| } \end{array}$ . In the full-information setting, we achieve a regret bound of $O ( D { \sqrt { \log F } } + { \sqrt { T \log F } } )$ when the Kolmogorov radius is $D .$ Since a Kolmogorov radius $D$ implies an Earth Mover’s Distance of $\Omega \dot { ( } D ^ { 2 } )$ for spending patterns of the same total budget spending, we have $D = O ( { \sqrt { \mathrm { E M D } } } )$ . Hence, our guarantees recover the dependence on EMD from prior work in the full-information setting, up to a $\sqrt { \log { \bar { F } } }$ factor, while potentially giving strictly smaller regret when the maximum cumulative-spending deviation is much smaller than EMD.

Stradi et al. [2025] study both the ORA and OLRC settings with adversarial rewards and costs and a target spending plan, but our results differ in key ways. In the ORA setting (with sampled play), they achieve a dynamic regret of ${ \tilde { O } } ( { \scriptstyle { \frac { 1 } { \rho _ { \operatorname* { m i n } } } } } { \sqrt { T } } )$ against the best dynamic strategy that spends at most the per-round budget of the spending plan, where $\rho _ { \mathrm { m i n } }$ is the minimum per-round budget of the spending plan. In the OLRC setting (with sampled play), they only benchmark against fixed distributions (that has per-round spending bounded above by the per-round spending of the budget plan), instead of dynamic strategies like ours, and achieve a regret of the same order $\begin{array} { r } { \tilde { O } ( \frac { 1 } { \rho _ { \mathrm { m i n } } } \sqrt { T } ) } \end{array}$ . Our benchmark differs from theirs as we study dynamic strategies in both settings of a set of D-pacing experts, and instead of strict adherence to the spending plan, we only require the experts to approximate the plan in Kolmogorov distance (i.e. the maximum absolute cumulative difference in spending). A more robust version considered in Stradi et al. [2025] does allow strategies to satisfy the spending plan up to some error terms in each round, but the additional regret term depends linearly on the cumulative error. This is a looser parameter than the Kolmogorov distance we propose — for example, a strategy that spends its budget $B = T / 2$ uniformly over even rounds (1 per even round) is very close to a pacing baseline that spends uniformly $1 / 2$ each round in Kolmogorov distance, but is very far when measured by cumulative error, since it overspends by a constant in $\Omega ( \bar { T } )$ rounds. Our paper thus proposes a more refined metric, and we also show that this metric tightly characterizes the regret rate in both the ORA and OLRC settings.

## 2 Models and Preliminaries

## 2.1 Problem Setup

Consider a setting with m resources, a set of N actions, denoted A, and a set of F experts, denoted ${ \mathcal F } .$ . The learner has a budget $B _ { i }$ in each resource $i \in [ m ]$ . Fix a time horizon $T \geq 1$ . At each round $t \in [ T ]$ , the learner receives an advice from each expert and picks an action $a _ { t } \in \mathcal A$ . Each action $a \in { \mathcal { A } }$ obtains some bounded reward $r _ { t } ( a ) \in [ 0 , 1 ]$ and incurs a set of bounded costs $c _ { t } ^ { i } ( a ) \in [ 0 , 1 ]$ for each resource $i \in [ m ]$ . The rewards and costs are determined by an adaptive adversary. Let $r _ { t } : = ( r _ { t } ( a ) ) _ { a \in \mathcal { A } }$ and $c _ { t } ^ { i } : = ( c _ { t } ^ { i } ( a ) ) _ { a \in \mathcal { A } }$ . The learner’s goal is to maximize the total sum of rewards $\textstyle \sum _ { t \in [ T ] } r _ { t } ( a _ { t } )$ from the selected actions subject to the budget constraint of each resource $\begin{array} { r } { \sum _ { t \in [ T ] } c _ { t } ^ { i } ( a _ { t } ) \leq B _ { i } , \forall i \in [ m ] . ^ { 5 } } \end{array}$

We study two standard information structures. In the online resource allocation (ORA) regime, the round-t reward and cost vectors $( r _ { t } , c _ { t } ^ { 1 } , \ldots , c _ { t } ^ { m } )$ are revealed before the learner chooses its round-t distribution. In the online learning with resource constraints (OLRC) regime, the learner chooses its round-t distribution before observing these vectors. The OLRC regime additionally splits into the full-information setting and the bandit setting. In the full-information setting, the complete vectors $( r _ { t } , c _ { t } ^ { 1 } , \ldots , c _ { t } ^ { m } )$ are revealed after the decision; in the bandit setting, only the reward and costs associated with the action $a _ { t }$ the learner chooses is revealed, i.e. $( r _ { t } ( a _ { t } ) , c _ { t } ^ { 1 } ( a _ { t } ) , \overline { { \cdot \cdot \cdot , \dot { c } _ { t } ^ { m } ( a _ { t } ) } } )$ . In the paper, we also sometimes refer to the two regimes as with pre-decision feedback, corresponding to ORA, and without pre-decision feedback, corresponding to OLRC. We formally state the sequence of events in both regimes below.

Our main analysis first considers the fractional relaxation, in which choosing $p _ { t } \in \Delta ( \mathcal { A } )$ yields deterministic reward $r _ { t } ( p _ { t } ) : = \langle p _ { t } , r _ { t } \rangle$ and deterministic resource costs $c _ { t } ^ { i } ( p _ { t } ) : = { \langle p _ { t } , \bar { c } _ { t } ^ { i } \rangle }$ . We call this the fractional play. Here the learner only needs to satisfy $\begin{array} { r } { \sum _ { t \in [ T ] } c _ { t } ^ { i } ( p _ { t } ) \leq B _ { i } , \forall i \in [ m ] } \end{array}$ , which we callfractional budget feasibility. In Section 5.3, we return to the sampled play setting, where the learner draws $a _ { t } \sim p _ { t }$ and needs to satisfy $\begin{array} { r } { \sum _ { t \in [ T ] } c _ { t } ^ { i } ( a _ { t } ) \leq B _ { i } , \forall i \in [ m ] } \end{array}$ . Our results focus on the full-information feedback setting, since it already captures the key challenge of temporal coupling arising from the budget constraints.

Online resource allocation (ORA) At round $t = 1 , \dots , T$

1. The learner receives an advice distribution $q _ { t } ^ { f } \in \Delta ( \mathcal { A } )$ from each expert $f \in { \mathcal { F } }$

2. For each action $a \in { \mathcal { A } } .$ , the adaptive adversary decides on a reward $r _ { t } ( a ) \in [ 0 , 1 ]$ and resource costs $c _ { t } ^ { i } ( a ) \in [ 0 , 1 ] , \forall i \in [ m ]$

3. The learner observes the full vectors $( r _ { t } , c _ { t } ^ { 1 } , \ldots , c _ { t } ^ { m } )$

4. The learner chooses a distribution $p _ { t } \in \Delta ( \mathcal { A } )$

5. In fractional $p l a y$ , the learner earns reward $r _ { t } ( p _ { t } )$ and costs $c _ { t } ^ { i } ( p _ { t } ) , \forall i \in [ m ]$ deterministically. In sampled play, the learner samples an action $a _ { t } \sim p _ { t }$ to take and earns reward $r _ { t } ( a _ { t } )$ and costs $c _ { t } ^ { i } ( \bar { a _ { t } } ) , \forall i \in [ m ]$

Online learning with resource constraints (OLRC) At round $t = 1 , \dots , T$

1. The learner receives an advice distribution $q _ { t } ^ { f } \in \Delta ( \mathcal { A } )$ from each expert $f \in { \mathcal { F } }$

2. The learner chooses a distribution $p _ { t } \in \Delta ( \mathcal { A } )$ , observable by the adversary.

3. For each action $a \in { \mathcal { A } } .$ , the adaptive adversary decides on a reward $r _ { t } ( a ) \in [ 0 , 1 ]$ and resource costs $c _ { t } ^ { i } ( a ) \in [ 0 , 1 ] , \forall i \in [ m ]$ , not yet observable by the learner.

4. In fractional $p l a y$ , the learner earns reward $r _ { t } ( p _ { t } )$ and costs $c _ { t } ^ { i } ( p _ { t } ) , \forall i \in [ m ]$ deterministically. In sampled play, the learner samples an action $a _ { t } \sim p _ { t }$ to take and earns reward $r _ { t } ( a _ { t } )$ and costs $c _ { t } ^ { i } ( \bar { a _ { t } } ) , \forall i \in [ m ]$ . In the full-information setting, the learner observes the full vectors $( r _ { t } , c _ { t } ^ { 1 } , \ldots , c _ { t } ^ { m } )$ In the bandit setting, the learner only observes the reward and costs of the selected action, i.e. $( r _ { t } ( a _ { t } ) , c _ { t } ^ { 1 } ( a _ { t } ) , \ldots , \bar { c } _ { t } ^ { m } ( a _ { t } ) )$

We assume there is a null action $\perp \in { \mathcal { A } }$ with $r _ { t } ( \perp ) = 0$ and $c _ { t } ^ { i } ( \perp ) = 0$ for every $i \in [ m ]$ and $t \in [ T ]$ . The learner is required to satisfy the hard budget constraints. Our algorithms enforce feasibility conservatively using a hard stop: before round t, if the cumulative spending on some resource i plus the maximum possible one-round cost would exceed $B _ { i }$ , the learner plays ⊥ from round t onward.

For $p \in \Delta ( \mathcal { A } )$ , write $r _ { t } ( p ) : = \langle p , r _ { t } \rangle$ and $c _ { t } ^ { i } ( p ) : = \langle p , c _ { t } ^ { i } \rangle$ , and define $r _ { t } ^ { f } : = r _ { t } ( q _ { t } ^ { f } )$ and $c _ { t } ^ { f , i } : = c _ { t } ^ { i } ( q _ { t } ^ { f } )$

We make the following assumption about the conversion of costs to rewards:

Assumption 2.1. For every resource i, there exist some constant reward-to-cost ratio $\alpha _ { i } > 0$ , residuals $\varepsilon _ { t } ^ { i } \geq 0$ , and some constant $V _ { i } \geq 0$ such that

$$
\begin{array} { r } { r _ { t } ( a ) \leq \alpha _ { i } c _ { t } ^ { i } ( a ) + \varepsilon _ { t } ^ { i } \quad \forall a \in \mathcal { A } , t \in [ T ] , \mathrm { ~ a n d ~ } \sum _ { t = 1 } ^ { T } \varepsilon _ { t } ^ { i } \leq V _ { i } . } \end{array}
$$

## 2.2 Subpacing Experts and New Regret Benchmark

Consider two spending patterns of the same resource $c = ( c _ { t } ) _ { t \in [ T ] } \in [ 0 , 1 ] ^ { T }$ and $c ^ { \prime } = ( c _ { t } ^ { \prime } ) _ { t \in [ T ] } \in [ 0 , 1 ] ^ { T }$ . We say the Kolmogorov distance<sup>6</sup> between them is

$$
\operatorname { K o l } ( c , c ^ { \prime } ) : = \operatorname* { m a x } _ { t \in [ T ] } \left| \sum _ { s = 1 } ^ { t } ( c _ { s } - c _ { s } ^ { \prime } ) \right| .
$$

Suppose $\mathrm { K o l } ( c , c ^ { \prime } ) \le D$ , then we say c is D-pacing with respect to $c ^ { \prime }$ (and $c ^ { \prime }$ is also D-pacing with respect to c). We also allow for a more permissive definition, which takes into account the fact that c and $c ^ { \prime }$ might have very different total spending: we say c is D-subpacing with respect to $c ^ { \prime }$ if

$$
\exists ( d _ { t } ) _ { t \in [ T ] } \mathrm { ~ s . t . ~ } d _ { t } \in [ 0 , c _ { t } ^ { \prime } ] \forall t \in [ T ] \mathrm { ~ a n d ~ K o l } ( c , d = ( d _ { t } ) _ { t \in [ T ] } ) \leq D .
$$

Fix a vector of baseline spending patterns $\pmb { b } = ( b ^ { i } ) _ { i \in [ m ] }$ , where $b ^ { i } = ( b _ { t } ^ { i } ) _ { t \in [ T ] } \in [ 0 , 1 ] ^ { T }$ and $\textstyle \sum _ { t = 1 } ^ { T } b _ { t } ^ { i } = B _ { i }$ for every resource $i \in [ m ]$ . We say an expert $f$ is $\pmb { { \cal D } } = ( D _ { i } ) _ { i \in [ m ] }$ -subpacing with respect to b if $c ^ { f , i } = ( c _ { t } ^ { f , i } ) _ { t \in [ T ] }$ is $D _ { i } { \mathrm { - s u b p a c i n g } }$ with respect to $b ^ { i } .$ , for all resource $i \in [ m ]$

Now consider a set of experts ${ \mathcal F } .$ . Let ${ \mathcal { F } } ( D , b ) : = \{ f \in { \mathcal { F } } \mid$ f is D-subpacing w.r.t. b} be the set of D-subpacing experts in $\mathcal { F }$ with respect to a prespecified baseline b. Our regret benchmark is

$$
\operatorname { O P T } _ { D , b , \mathcal { F } } : = \operatorname* { m a x } _ { f \in \mathcal { F } ( D , b ) } \sum _ { t = 1 } ^ { T } r _ { t } ^ { f }
$$

and the regret of a strategy $\mathbf { x } : = ( x _ { t } ) _ { t } \in \Delta ( \mathcal { A } ) ^ { T }$ against $\mathrm { O P T } _ { D , b , \mathcal { F } }$ is

$$
\mathrm { R e g } _ { \mathrm { O P T } _ { D , b , \mathcal { F } } } ( \mathbf { x } ) : = \mathrm { O P T } _ { D , b , \mathcal { F } } - \sum _ { t = 1 } ^ { T } r _ { t } ( x _ { t } ) .
$$

## 3 A Warm-up Example

Before we present the full algorithm, we give some intuition in a simpler pre-decision-feedback (ORA) setting with two experts, two actions, one resource, fractional play, and a perfectly pacing baseline. In each round, the learner observes the current reward and cost before choosing its action.<sup>7</sup>

## 3.1 The Alternating Red and Blue Balls Problem

Fix $D \in \mathbb { Z } ^ { + }$ . Consider the setting where a sequence of D red balls is followed by a sequence of D blue balls, with this pattern repeated alternately for a total of T balls and $T = 2 k D$ for some $k \in \mathbb { Z } ^ { + }$ . In each round, a value $r _ { t }$ of the current ball is revealed, and the cost for each ball is always 1. The two actions are “picking the ball" and the null action ⊥. The red expert always advises picking the red balls, and the blue expert always advises picking the blue balls. The only thing that is hidden from the learner at round t is the reward of future balls arriving at rounds $> t .$ . The budget of the learner is $T / 2$ (so she can pick $T / 2$ units of balls in total) and for simplicity, we assume the learner can pick fractionally.

The baseline spending pattern b for this problem is the perfectly pacing spending pattern, i.e. $b \ =$ $\{ 1 / 2 , . . . , 1 / 2 \} \stackrel { \cdot } { \in } [ \stackrel { \cdot } { 0 } , \stackrel { \cdot } { 1 } ] ^ { T }$ By definition, the red expert R has a spending pattern of $\begin{array} { r l } { c ^ { R } } & { { } = } \end{array}$ $( 1 , \ldots , 1 , 0 , \ldots , 0 \ldots )$ , and the blue expert B has a spending pattern of $c ^ { B } = ( 0 , \ldots , 0 , 1 , \ldots , 1 \ldots )$ . Since the alternating monochromatic blocks have size D each, we have $\operatorname { K o l } ( c ^ { R } , b ) \leq D / 2$ and $\operatorname { K o l } ( c ^ { B } , b ) \leq D / 2$ We note that the red expert’s reward at t is $r _ { t } ^ { R } = r _ { t } \mathbf { 1 } \{ t$ is a Red ball} and the blue expert’s reward is $r _ { t } ^ { B } = r _ { t } \mathbf { 1 } \cdot$ {t is a Blue ball}. The learner is competing against the benchmark $\mathrm { O P T } _ { \mathit { D } , b , \mathcal { F } }$ where $\mathcal { F } = \{ R , B \}$ Thus

$$
\begin{array} { r } { \operatorname { O P T } _ { D , b , \mathcal { F } } = \operatorname* { m a x } \left\{ \sum _ { t \in [ T ] } r _ { t } ^ { R } , \sum _ { t \in [ T ] } r _ { t } ^ { B } \right\} . } \end{array}
$$

Let $x _ { t } \in [ 0 , 1 ]$ denote the fraction picked by the learner in round t, i.e $[ x _ { t } , 1 - x _ { t } ]$ is the distribution the learner chooses over the two actions. Then the learner’s regret for using the strategy $\mathbf { x } ~ = ~ ( x _ { t } ) _ { t \in [ T ] }$ is $\begin{array} { r } { \mathrm { R e g } _ { \mathrm { O P T } _ { D , b , \mathcal { F } } } ( \mathbf { x } ) : = \mathbf { O P T } _ { D , b , \mathcal { F } } - \sum _ { t = 1 } ^ { T } r _ { t } x _ { t } } \end{array}$

## 3.2 An Optimal Algorithm for the Alternating Red and Blue Balls Problem

In the following section, we design an algorithm with a regret guarantee of $O ( D )$ and is $O ( D )$ -pacing against b. The regret guarantee is in fact optimal, by the lower bound in Theorem 6.1, since $\operatorname { K o l } ( c ^ { R } , c ^ { B } ) = \Theta ( D )$

Step 1. Reduction to D subproblems. In fact, we can reduce the problem to an even simpler problem, where the length of each monochromatic block is 1, and the baseline spending pattern is still the perfectly pacing spending pattern. We do so by putting the arriving red and blue balls into D separate queues. For each ball that arrives, if it is the $j \mathrm { - t h }$ ball in a monochromatic block, then we put it in the j-th queue. Thus each queue becomes a subproblem with $D ^ { \prime } = 1$ , with both experts being constant-pacing with respect to the perfectly pacing baseline. It suffices to show that for each subproblem, the algorithm has a regret guarantee of $O ( 1 )$ and a pacing guarantee of $O ( 1 )$ against the perfectly pacing baseline.

Step 2. Solving the Subproblem with $D ^ { \prime } = 1$ . We now focus on a single queue and, for notational simplicity, relabel its horizon as T. Thus the balls alternate between red and blue, with a red ball followed by a blue ball, T is even, and the learner has budget $T / 2$ . At each round $t \in [ T ]$ , the learner observes the current reward $r _ { t } \in [ 0 , 1 ]$ and then chooses a fraction $x _ { t } \in [ 0 , 1 ]$ of the current ball to eat. The two benchmark experts eat all red balls and all blue balls, respectively. The baseline spending pattern is uniform $1 / 2$ each round. This feedback model lies naturally between two extremes:

• If, upon the arrival of each red ball, the learner knew the rewards of both that red ball and the following blue ball before choosing what fraction to eat, then it could simply eat whichever of the two has a larger reward. It would consume exactly one unit per red–blue pair and obtain reward $\textstyle \sum _ { k = 1 } ^ { T / 2 } \operatorname* { m a x } \{ r _ { 2 k - 1 } , r _ { 2 k } \}$ , which is at least the reward of either the red expert or the blue expert. Hence it would have nonpositive regret, satisfy the budget exactly, and have only constant cumulative spending deviation from the uniform spending rate at any point.

• In contrast, if the learner had to choose $x _ { t }$ before observing the current reward $r _ { t }$ , the classical online-learning lower bound yields worst-case regret of $\Omega ( { \sqrt { T } } ) . ^ { 8 }$

Our setting lies between these two cases: the learner knows the reward of the current ball before deciding how much to eat, but has no information about future rewards. We show that the ability to know the current round’s reward is already sufficient to achieve constant regret while maintaining constant pacing.

Our goal is two-fold: minimize regret against both experts and minimize the Kolmogorov distance from the baseline. Since we need to prove worst-case guarantees for both objectives, we can assign a dual variable to each objective and view the problem as a zero-sum game between the learner and an adversary (separate from the adversarial environment), where the adversary picks the “worst" dual variables that punishes the learner when regret against any expert is high or when the algorithm is deviating from the baseline by too much, and the learner needs to make decisions that avoid a high punishment. The payoff of this hypothetical zero-sum game needs to be carefully designed and merges the two objectives into a single function of the learner and the adversary’s decisions.

Let us begin by defining some notations. Let $a _ { t } ^ { R } = \mathbf { 1 } \{ t \mathrm { i s } \mathrm { R e d } \}$ and $a _ { t } ^ { B } = \mathbf { 1 } \{ t { \mathrm { i s } } \mathbf { B } \mathrm { l u e } \}$ be the fraction that the Red expert picks and the fraction that the Blue expert picks at round t, respectively. Let $R _ { t } ^ { R } =$ $\begin{array} { r } { \sum _ { s < t } r _ { s } ( a _ { s } ^ { R } - x _ { s } ) } \end{array}$ and $\begin{array} { r } { R _ { t } ^ { B } = \sum _ { s < t } r _ { s } ( a _ { s } ^ { B } - x _ { s } ) } \end{array}$ denote the learner’s regret against the Red expert and the Blue experts, at the end of round t. We write the regret vector as $R _ { t } = ( R _ { t } ^ { R } , R _ { t } ^ { B } )$ . We parameterize a distribution over the two experts by $p ( z ) ~ = ~ \bigl ( \frac { 1 } { 2 } + z , \frac { 1 } { 2 } - z \bigr )$ , where $z \in [ - \frac { 1 } { 2 } , \frac { 1 } { 2 } ]$ . On round t, the corresponding distribution eats $a _ { t } ( z ) \ = \ \langle p ( z ) , ( a _ { t } ^ { R } , a _ { t } ^ { B } ) \rangle$ , and $\langle R _ { t } , p ( z ) \rangle$ is the learner’s regret to the distribution $p ( z )$ . Let $\begin{array} { r } { Y _ { t } = \sum _ { s < t } \left( x _ { s } - \frac { \mathrm { i } } { 2 } \right) } \end{array}$ be the learner’s cumulative overspending from the pacing baseline, and let $\begin{array} { r } { E _ { t } ( z ) = \sum _ { s < t } \bigl ( \bar { x _ { s } } - a _ { s } ( z ) \bigr ) } \end{array}$ be the learner’s overspending against the distribution $p ( z )$

Suppose on round t the learner chooses $x \in [ 0 , 1 ]$ , while an adversary chooses the dual variables $z \in$ $[ - \bar { 1 } / 2 , 1 / 2 ]$ and a dual resource price $v \in \mathbb { R }$ . One might come up with the following payoff function to merge the two objectives:

$$
\langle R _ { t - 1 } , p ( z ) \rangle + v Y _ { t - 1 } + r _ { t } \big ( a _ { t } ( z ) - x \big ) + v \big ( x - 1 / 2 \big ) .
$$

The first term is the learner’s regret against the adversarially chosen distribution $p ( z )$ before round $t ,$ and $v Y _ { t - 1 }$ prices its cumulative resource deviation against the pacing baseline. The difficulty is that these two terms compare the learner to different reference points: regret is measured against the mixture $p ( z )$ , while resource deviation is measured against the fixed baseline. We instead measure both quantities relative to the same mixture $p ( z )$ . This alignment is crucial: if the learner chooses the spending level prescribed by $p ( z )$ then both the instantaneous regret and resource-deviation terms vanish simultaneously. We therefore use:

$$
\langle R _ { t - 1 } , p ( z ) \rangle + v E _ { t - 1 } ( z ) + r _ { t } { \big ( } a _ { t } ( z ) - x { \big ) } + v { \big ( } x - a _ { t } ( z ) { \big ) } ,
$$

where the dual variable v penalizes the deviation from the cumulative spending of the mixture $p ( z )$ instead of the cumulative spending of the pacing baseline.

Finally, since the payoff function contains nonlinear terms involving the dual variables, we need to regularize both dual variables to apply Sion’s minimax theorem [Sion, 1958]. Define

$$
F _ { t } ( x , z , v ) = \langle R _ { t - 1 } , p ( z ) \rangle + v E _ { t - 1 } ( z ) + r _ { t } { \big ( } a _ { t } ( z ) - x { \big ) } + v { \big ( } x - a _ { t } ( z ) { \big ) } - { \frac { 1 } { 2 } } v ^ { 2 } - z ^ { 2 } .\tag{1}
$$

Let $\begin{array} { r } { S _ { t } \ = \ \sum _ { s < t } \left( \mathbf { 1 } \{ s \ \mathrm { i s } \ \mathrm { R e d } \} - \mathbf { 1 } \{ s \ \mathrm { i s } \ \mathrm { B l u e } \} \right) } \end{array}$ . Then, $| S _ { t } | \le 1$ , and we have $E _ { t - 1 } ( z ) + x - a _ { t } ( z ) =$ $\begin{array} { r } { Y _ { t - 1 } + x - \frac { 1 } { 2 } = S _ { t } z } \end{array}$ , so the only nonlinear dependence on $( z , v )$ is

$$
- v S _ { t } z - \frac { 1 } { 2 } v ^ { 2 } - z ^ { 2 } .
$$

Its Hessian is

$$
\left( \begin{array} { l l } { - 2 } & { - S _ { t } } \\ { - S _ { t } } & { - 1 } \end{array} \right) ,
$$

which is negative definite because $| S _ { t } | \le 1$ . Hence, $F _ { t }$ is convex in x and jointly concave in $( z , v )$ , and Sion’s minimax theorem applies.

Now, we are ready to define the algorithm, which simply solves a minmax problem in each round, shown in Algorithm 1. It is worth noting that the algorithm is agnostic of the time horizon, and both the regret guarantee and the pacing guarantee do not depend on $T .$

Algorithm 1: Sequential online selection for alternating red and blue balls.

```latex
for $t = 1 , \dots , T$ do
Observe the current reward $r _ { t }$
Define $\begin{array} { r } { F _ { t } ( x , z , v ) = \langle R _ { t - 1 } , p ( z ) \rangle + v E _ { t - 1 } ( z ) + r _ { t } \big ( a _ { t } ( z ) - x \big ) + v \big ( x - a _ { t } ( z ) \big ) - \frac { 1 } { 2 } v ^ { 2 } - z ^ { 2 } } \end{array}$
Choose
$x _ { t } \in \arg \operatorname* { m i n } _ { x \in [ 0 , 1 ] } \operatorname* { m a x } _ { | z | \leq 1 / 2 } F _ { t } ( x , z , v )$
end
```

Theorem 3.1. Algorithm 1 has constant regret and is constant-pacing with respect to the baseline $b =$ $( 1 / 2 ) _ { t \in [ T ] } , i . e .$

$$
\operatorname* { m a x } \{ R _ { t } ^ { R } , R _ { t } ^ { B } \} = O ( 1 ) \ a n d \ \operatorname { K o l } ( ( x _ { t } ) _ { t \in [ T ] } , b ) = O ( 1 ) .
$$

Proof. Since $F _ { t } ( x , z , v )$ is affine in x and jointly concave in $( z , v )$ , Sion’s minimax theorem gives

$$
W _ { t } = \operatorname* { m i n } _ { x \in [ 0 , 1 ] } \operatorname* { m a x } _ { | z | \leq 1 / 2 } F _ { t } ( x , z , v ) = \operatorname* { m a x } _ { | z | \leq 1 / 2 } \operatorname* { m i n } _ { x \in [ 0 , 1 ] } F _ { t } ( x , z , v ) .
$$

Define potential $\begin{array} { r } { \Phi _ { t } ( z , v ) = \langle R _ { t } , p ( z ) \rangle + v E _ { t } ( z ) - \frac { 1 } { 2 } v ^ { 2 } - z ^ { 2 } } \end{array}$ . Thus $\Phi _ { t } ( z , v ) = F ( x _ { t } , z , v )$ and we have

$$
W _ { t } = \operatorname* { m a x } _ { | z | \leq 1 / 2 } \Phi _ { t } ( z , v ) = \operatorname* { m a x } _ { | z | \leq 1 / 2 } \left. R _ { t } , p ( z ) \right. + \frac { 1 } { 2 } E _ { t } ( z ) ^ { 2 } - z ^ { 2 } ,
$$

by choosing the optimal $v ^ { * } = E _ { t } ( z )$ . We note that

$$
\begin{array} { c l l } { F _ { t } ( x , z , v ) = \langle R _ { t - 1 } , p ( z ) \rangle + v E _ { t - 1 } ( z ) + r _ { t } \big ( a _ { t } ( z ) - x \big ) + v \big ( x - a _ { t } ( z ) \big ) - \frac { 1 } { 2 } v ^ { 2 } - z ^ { 2 } . } \\ { \ } & { } \\ { = \langle R _ { t - 1 } , p ( z ) \rangle + v E _ { t - 1 } ( z ) + ( r _ { t } - v ) \big ( a _ { t } ( z ) - x \big ) - \frac { 1 } { 2 } v ^ { 2 } - z ^ { 2 } } \end{array}
$$

This shows that $\begin{array} { r } { F _ { t } ( x = a _ { t } ( z ) , z , v ) = \langle R _ { t - 1 } , p ( z ) \rangle + v E _ { t - 1 } ( z ) - \frac { 1 } { 2 } v ^ { 2 } - z ^ { 2 } = \Phi _ { t - 1 } ( z , v ) } \end{array}$ . Thus we have

$$
W _ { t } = \operatorname* { m a x } _ { | z | \leq 1 / 2 } \operatorname* { m i n } _ { x \in [ 0 , 1 ] } F _ { t } ( x , z , v ) \leq \operatorname* { m a x } _ { | z | \leq 1 / 2 } F _ { t } ( x = a _ { t } ( z ) , z , v ) = \operatorname* { m a x } _ { | z | \leq 1 / 2 } \Phi _ { t - 1 } ( z , v ) = W _ { t - 1 } .
$$

Since $\begin{array} { r } { W _ { 0 } = \operatorname* { m a x } _ { | z | \leq 1 / 2 } - z ^ { 2 } = 0 } \end{array}$ , induction on t gives $W _ { t } \leq 0$ . This means

$$
\operatorname* { m a x } _ { | z | \leq 1 / 2 } \langle R _ { t } , p ( z ) \rangle + \frac { 1 } { 2 } E _ { t } ( z ) ^ { 2 } - z ^ { 2 } \leq 0
$$

Evaluating at $z = \pm \frac { 1 } { 2 }$ gives $\begin{array} { r } { R _ { t } ^ { R } + \frac { 1 } { 2 } E _ { t } ( 1 / 2 ) ^ { 2 } \leq \frac { 1 } { 4 } } \end{array}$ and $R _ { t } ^ { B } + \textstyle { \frac { 1 } { 2 } } E _ { t } ( - 1 / 2 ) ^ { 2 } \leq \textstyle { \frac { 1 } { 4 } }$ . Thus $R _ { t } ^ { R } , R _ { t } ^ { B } \leq 1 / 4$ proving the regret guarantee.

For the pacing guarantee, it suffices to show $| Y _ { t } | = O ( 1 ) , \forall t \in [ T ]$ . Choose $( z _ { t } , v _ { t } )$ such that $( x _ { t } , z _ { t } , v _ { t } )$ is a saddle point of $F _ { t } ( x , z , v )$ . Optimality in v gives $v _ { t } = E _ { t } ( z _ { t } ) = Y _ { t } - S _ { t } z _ { t }$ , and hence $Y _ { t } = v _ { t } + S _ { t } z _ { t }$ . Since $x \mapsto \bar { F _ { t } } ( x , z _ { t } , v _ { t } )$ is affine with slope $v _ { t } - r _ { t }$ , optimality in x implies that if $x _ { t } = 0$ , then $v _ { t } \geq r _ { t } \geq 0$ ; if $0 < x _ { t } < 1$ , then $v _ { t } = r _ { t } \in [ 0 , 1 ]$ ; and if $x _ { t } = 1$ , then $v _ { t } \leq r _ { t } \leq 1$

We claim inductively that $- 1 / 2 \le Y _ { t } \le 3 / 2$ . The claim holds for $Y _ { 0 } ~ = ~ 0$ . If $x _ { t } = 0$ , then $v _ { t } \geq 0$ gives $Y _ { t } = v _ { t } + S _ { t } z _ { t } \ge - 1 / 2$ since $| z _ { t } | \le 1 / 2$ , and $| S _ { t } | \le 1$ , while $Y _ { t } = Y _ { t - 1 } - 1 / 2 \le 1$ by induction. If $0 < x _ { t } < 1$ , then $Y _ { t } = \stackrel { \cdot } { r } _ { t } + S _ { t } \dot { z } _ { t } \in [ - \mathrm { 1 } / 2 , \mathrm { 3 } / 2 ]$ , since $r _ { t } ~ \in ~ [ 0 , 1 ]$ . If $x _ { t } = 1$ , then $v _ { t } \ \leq \ 1$ gives $Y _ { t } = v _ { t } + S _ { t } z _ { t } \le 3 / 2$ , while $Y _ { t } = Y _ { t - 1 } + \mathrm { \bar { 1 } } / 2 \geq 0$ by induction. Thus we have $- 1 / 2 \le Y _ { t } \le 3 / 2$ for all $t \in \left\lceil T \right\rceil$ . If exact budget feasibility is desired, truncating the path once the budget $T / \dot { 2 }$ is exhausted changes both the regret and pacing guarantees by only an additional ${ \bf \hat { O } } ( 1 )$ □

To solve the original problem with alternating D-length blocks of red balls and blue balls, we run Algorithm 1 for each queue. Since the algorithm for the subproblems has constant regret and is constant-pacing, it is easy to see that the resulting algorithm has at most ${ \hat { O } } ( D )$ regret and is $O ( D )$ -pacing.

## 4 Near-Optimal Algorithm for Learning with Pacing Experts

The algorithm in Section 3 is in fact generalizable to any finite set of experts and multiple resources. We study both the ORA regime and the OLRC regime. Our algorithm for the first regime generalizes the toy algorithm in Section $^ { 3 , }$ whereas the algorithm for the latter regime requires an additional stability analysis. We first assume the learner can play fractionally, i.e. deterministically playing a distribution $p _ { t } \in \Delta ( \mathcal { A } )$ and get deterministic reward $r _ { t } ( p _ { t } )$ and costs $c _ { t } ^ { i } ( p _ { t } )$ ). We then remove this assumption and analyze sampled actions in Section 5.3.

## 4.1 Notation

Throughout, an unqualified sum $\textstyle \sum _ { t }$ denotes the full-horizon sum $\scriptstyle \sum _ { t = 1 } ^ { T }$ . Fix a baseline spending pattern for each resource $i \in [ m ]$ $b ^ { i } = ( b _ { t } ^ { i } ) _ { t \in [ T ] } \in [ 0 , 1 ] ^ { T }$ and $\textstyle \sum _ { t = 1 } ^ { T } b _ { t } ^ { i } = B _ { i }$ . Every expert is initially assumed $D _ { i } { \mathrm { - p a c i n g } }$ in resource i:

$$
S _ { t } ^ { f , i } : = \sum _ { s \leq t } ( c _ { s } ^ { f , i } - b _ { s } ^ { i } ) , \qquad | S _ { t } ^ { f , i } | \leq D _ { i } \quad \forall f , i , t .\tag{2}
$$

Recall that a cost path $( c _ { t } ) _ { t }$ is called D-subpacing with respect to a baseline $( b _ { t } ) _ { \mathrm { : } }$ if there exists a witness $d _ { t } \in [ 0 , b _ { t } ]$ such that

$$
\left| \sum _ { s \leq t } ( c _ { s } - d _ { s } ) \right| \leq D \qquad \forall t \in [ T ] .\tag{3}
$$

Thus, D-pacing is the special case witnessed by $d _ { t } = b _ { t }$

Define

$$
b _ { \operatorname* { m i n } } ^ { i } : = \operatorname* { m i n } _ { t } b _ { t } ^ { i } , \qquad \beta _ { i } : = 1 / b _ { \operatorname* { m i n } } ^ { i } , \qquad \eta _ { i } : = \operatorname* { m i n } \{ \alpha _ { i } , \beta _ { i } \}\tag{4}
$$

where recall $\alpha _ { i }$ is the reward-to-cost ratio of resource i in Assumption 2.1. In words, $b _ { \operatorname* { m i n } } ^ { i }$ denotes the smallest term in the baseline spending pattern $b _ { i }$ and $\beta _ { i } : = 1 / b _ { \mathrm { m i n } } ^ { i }$ its reciprocal (with $\beta _ { i } = + \infty \mathrm { i f } b _ { \mathrm { m i n } } ^ { i } = 0 )$ . The quantity $\eta _ { i }$ is the cheaper terminal price: additional rewards accrued by an expert after the algorithm exhausts some resource i and hard stops at time τ can be priced either by the additional cost incurred by an expert at rate $\alpha _ { i }$ or, when $b _ { \operatorname* { m i n } } ^ { i } > 0$ , by the remaining cost of the baseline at rate $\beta _ { i }$ , since $T - \tau \leq \bar { \beta _ { i } } \sum _ { t > \tau } \dot { b } _ { t } ^ { i }$ and $\begin{array} { r } { \sum _ { t > \tau } r _ { t } ^ { f } \le T - \tau } \end{array}$ . Since $\alpha _ { i } > 0$ and $\beta _ { i } \in ( 0 , + \infty ]$ , every $\eta _ { i }$ is strictly positive.

We henceforth assume that $D _ { i } > 0$ for at least one resource i. If $D _ { i } = 0$ for every resource, then every paced expert has $c _ { t } ^ { f , i } = b _ { t } ^ { i }$ on every round (and every 0-subpacing expert has $c _ { t } ^ { f , i } \leq b _ { t } ^ { i } )$ , so the learning component reduces to the standard learning-with-expert-advice problem under the corresponding feedback model; we omit this degenerate case.

For any distribution over experts $w \in \Delta ( \mathcal { F } )$ , define

$$
\begin{array} { r } { \pi _ { t } ( w ) : = \sum _ { f } w _ { f } q _ { t } ^ { f } , \qquad r _ { t } ( w ) : = \sum _ { f } w _ { f } r _ { t } ^ { f } , \qquad c _ { t } ^ { i } ( w ) : = \sum _ { f } w _ { f } c _ { t } ^ { f , i } . } \end{array}
$$

That is, $\pi _ { t } ( w )$ is the probability distribution over actions induced by the distribution w over experts, while $r _ { t } ( w )$ and $c _ { t } ^ { i } ( w )$ are the expected reward and expected cost in resource i, respectively, of choosing an action based on this distribution. (By a slight abuse of notation, we use $r _ { t } ( \cdot )$ and $c _ { t } ^ { i } ( \bar { \cdot } )$ to denote the expected reward and cost for both distributions over actions and distributions over experts, with the intended domain clear from context.)

The algorithm uses a subroutine called a controller. At each time step, the controller chooses a distribution over experts $x _ { t } \in \Delta ( \mathcal { F } )$ . The algorithm then plays the distribution over actions $p _ { t } = \pi _ { t } ( x _ { t } )$ (or ⊥ if the controller has exceeded its budget).

Define

$$
R _ { t } ^ { f } : = \sum _ { s \leq t } ( r _ { s } ^ { f } - r _ { s } ( p _ { s } ) ) , \qquad Y _ { t } ^ { i } : = \sum _ { s \leq t } ( c _ { s } ^ { i } ( p _ { s } ) - b _ { s } ^ { i } ) , \qquad E _ { t } ^ { i } ( w ) : = Y _ { t } ^ { i } - \sum _ { f } w _ { f } S _ { t } ^ { f , i } .\tag{5}
$$

$R _ { t } ^ { f }$ denotes the controller’s regret against expert $f , Y _ { t } ^ { i }$ its “overspending” (which can be negative) compared to the baseline spending pattern, and $E _ { t } ^ { i }$ the difference in overspending between the controller and the expected overspending of the distribution w over experts: since $\begin{array} { r } { \sum _ { f } w _ { f } = 1 } \end{array}$ , the baseline terms cancel and $\begin{array} { r } { E _ { t } ^ { i } ( w ) = \sum _ { s < t } ( c _ { s } ^ { i } ( p _ { s } ) - c _ { s } ^ { i } ( w ) ) } \end{array}$ . When $p _ { t } = \pi _ { t } ( x _ { t } )$ , we also have $\begin{array} { r } { E _ { t } ^ { i } ( w ) = \sum _ { s < t } ( c _ { s } ^ { i } ( x _ { s } ) - c _ { s } ^ { i } ( w ) ) } \end{array}$ . We reserve $R _ { t } ^ { f }$ for the regret coordinate used in the potential. In statements of regret guarantees, we write

$$
\mathrm { R e g } _ { t } ^ { f } : = \sum _ { s \leq t } ( r _ { s } ^ { f } - r _ { s } ( p _ { s } ) ) .\tag{6}
$$

Algorithmic resource prices. Let $\lambda = ( \lambda _ { i } ) _ { i \in [ m ] } \in \mathbb { R } _ { > 0 } ^ { m }$ be prices chosen by the algorithm. These are distinct from the structural conversion parameters $\alpha _ { i }$ in Assumption 2.1. Write

$$
D _ { \lambda , \infty } : = \operatorname* { m a x } _ { i \in [ m ] } \lambda _ { i } D _ { i } , \qquad \lambda _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i \in [ m ] } \lambda _ { i } .\tag{7}
$$

Let $v = ( v _ { 0 } , v _ { 1 } , \ldots , v _ { m } ) \in \Delta _ { m + 1 }$ be a distribution over the m resources and a slack coordinate. Fix $v \in \Delta _ { m + 1 } , w \in \Delta ( \mathcal { F } )$ . We say the resource penalty of coordinate 0 is 0, while the resource penalty of coordinate i is $\lambda _ { i } v _ { i } E _ { t } ^ { i } ( w )$ . Intuitively, this term penalizes deviation of the algorithm’s spending from the spending of the expert mixture w, scaled by the chosen price $\lambda _ { i }$ and the probability mass v<sub>i</sub>.

Potential function. We use the shifted negative entropies as regularizers for $w \in \Delta ( \mathcal { F } )$ and $v \in \Delta _ { m + 1 } ;$

$$
\begin{array} { l l } { \Psi _ { x } ( w ) : = \displaystyle \sum _ { f } w _ { f } \log w _ { f } + \log F = D _ { \mathrm { K L } } ( w \| U _ { F } ) } \\ { \Psi _ { v } ( v ) : = \displaystyle \sum _ { i = 0 } ^ { m } v _ { i } \log v _ { i } + \log ( m + 1 ) = D _ { \mathrm { K L } } ( v \| U _ { m + 1 } ) . } \end{array}
$$

where $U _ { k }$ denotes the uniform distribution over k elements. Then $\Psi _ { x } ( w ) \in [ 0 , \log F ]$ and $\Psi _ { v } ( v ) \ \in$ $[ 0 , \log ( m + 1 ) ]$

Choose regularization coefficients $\rho _ { x } , \rho _ { v } > 0$ to be defined later. We define the potential function and its maximum as

$$
\Phi _ { t } ^ { \lambda } ( w , v ) : = \sum _ { f } w _ { f } R _ { t } ^ { f } + \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } E _ { t } ^ { i } ( w ) - \rho _ { x } \Psi _ { x } ^ { } ( w ) - \rho _ { v } \Psi _ { v } ^ { } ( v ) , \qquad M _ { t } ^ { \lambda } : = \operatorname* { m a x } _ { w , v } \Phi _ { t } ^ { \lambda } ( w , v ) ,\tag{8}
$$

Here and throughout, the maximum is taken over $( w , v ) \in \Delta ( \mathcal { F } ) \times \Delta _ { m + 1 }$ , and we omit this domain when it is clear from context. The maximum potential ${ M } _ { t } ^ { \lambda }$ provides a useful upper bound on the combined regret and resource penalty against an arbitrary expert mixture w and resource distribution $v . ^ { 9 }$

To define how the potential function changes after a new decision $x \in \Delta ( \mathcal { F } )$ , we let

$$
F _ { t } ^ { \lambda } ( x , w , v ) : = \Phi _ { t - 1 } ^ { \lambda } ( w , v ) + r _ { t } ( w ) - r _ { t } ( x ) + \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } \big ( c _ { t } ^ { i } ( x ) - c _ { t } ^ { i } ( w ) \big ) .
$$

Hard stop. Let $\begin{array} { r } { C _ { t - 1 } ^ { i } : = \sum _ { s \leq t - 1 } c _ { s } ^ { i } ( p _ { s } ) } \end{array}$ denote the cumulative spending of the controller. Before round $t ,$ if the hard stop has already taken effect, or if $C _ { t - 1 } ^ { i } + 1 > B _ { i }$ for some i, output $\perp$ from round t onward. Otherwise, run the controllers outlined in Section 4.3. Since every one-round cost is at most 1, this guarantees fractional budget feasibility $\begin{array} { r } { \sum _ { t } c _ { t } ^ { i } ( p _ { t } ) \leq B _ { i } } \end{array}$

## 4.2 Key Curvature Lemma

The potential function involves three nonlinear terms: $- \rho _ { x } \Psi _ { x } ( w ) , - \rho _ { v } \Psi _ { v } ( v )$ , and $\scriptstyle \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } E _ { t } ^ { i } ( w )$ . The first two terms are shifted entropy terms that provide strong concavity in w and $v ,$ respectively, in $L ^ { 1 } .$ -norm. The third term is problematic because it involves products of v and w and the nonlinearity can erase the concavity in w and $v .$ However, with appropriately chosen parameters $\rho _ { x } , \rho _ { v } ,$ , we show that the potential function satisfies the following concavity properties.

Lemma 4.1. Assume

$$
\rho _ { x } \rho _ { v } \geq 4 D _ { \lambda , \infty } ^ { 2 } .\tag{9}
$$

Then for every $t , \Phi _ { t } ^ { \lambda }$ is jointly concave in $( w , v )$ , and $F _ { t } ^ { \lambda } ( x , \cdot , \cdot )$ is jointly concave in $( w , v )$ for everyfixed $x \in \Delta ( { \mathcal { F } } )$

Moreover, $i f ( w ^ { * } , v ^ { * } )$ maximizes $\Phi _ { t } ^ { \lambda }$ , then both $w ^ { * }$ and $v ^ { * }$ havefull support and,for everyfeasible $( w , v )$

$$
\Phi _ { t } ^ { \lambda } ( w , v ) \leq M _ { t } ^ { \lambda } - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| w ^ { * } ) - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v ^ { * } ) .\tag{10}
$$

Proof. Consider interior points $( w , v )$ and $( w ^ { \prime } , v ^ { \prime } ) = ( w + \Delta w , v + \Delta v )$ . To show $\Phi _ { t } ^ { \lambda }$ is jointly concave in $( w , v )$ , it suffices to show

$$
\Phi _ { t } ^ { \lambda } ( w + \Delta w , v + \Delta v ) - \Phi _ { t } ^ { \lambda } ( w , v ) \leq \left. \nabla \Phi _ { t } ^ { \lambda } ( w , v ) , ( \Delta w , \Delta v ) \right. .
$$

We have

$$
\begin{array} { l } { \displaystyle \Phi _ { t } ^ { \lambda } ( w + \Delta w , v + \Delta v ) - \Phi _ { t } ^ { \lambda } ( w , v ) } \\ { \displaystyle = \langle \Delta w , R _ { t } \rangle + \sum _ { i = 1 } ^ { m } \lambda _ { i } \Delta v _ { i } Y _ { t } ^ { i } - \sum _ { i = 1 } ^ { m } \lambda _ { i } \Delta v _ { i } \langle w , S _ { t } ^ { i } \rangle - \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } \langle \Delta w , S _ { t } ^ { i } \rangle } \\ { \displaystyle \quad - \sum _ { i = 1 } ^ { m } \lambda _ { i } \Delta v _ { i } \langle \Delta w , S _ { t } ^ { i } \rangle - \rho _ { x } ( \Psi _ { x } ( w + \Delta w ) - \Psi _ { x } ( w ) ) - \rho _ { v } ( \Psi _ { v } ( v + \Delta v ) - \Psi ( v ) ) } \\ { \displaystyle = \Big \langle \nabla \Phi _ { t } ^ { \lambda } ( w , v ) , ( \Delta w , \Delta v ) \Big \rangle - \sum _ { i = 1 } ^ { m } \lambda _ { i } \Delta v _ { i } \langle \Delta w , S _ { t } ^ { i } \rangle - \rho _ { x } D _ { \mathrm { K L } } ( w + \Delta w \| w ) - \rho _ { v } D _ { \mathrm { K L } } ( v + \Delta v \| v ) . } \end{array}
$$

where we use the fact that Ψ<sub>x</sub>(w + ∆w) − Ψ<sub>x</sub>(w) = ∇Ψ<sub>x</sub>(w)∆w + D<sub>KL</sub>(w + ∆w∥w) and Ψ<sub>v</sub>(v + ∆v) − Ψ<sub>v</sub>(v) = ∇Ψ<sub>v</sub>(v)∆v + D<sub>KL</sub>(v + ∆v∥v).

We now bound the first nonlinear term, which crucially relies on the pacing property of the experts:

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { m } \lambda _ { i } \Delta v _ { i }  \Delta w , S _ { t } ^ { i }  \Bigg | \leq D _ { \lambda , \infty } \| \Delta v \| _ { 1 } \| \Delta w \| _ { 1 } } } & { ( \| S _ { t } ^ { i } \| _ { \infty } \leq D _ { i } ) } \\ & { } & { \leq \frac { \rho _ { x } } { 4 } \| \Delta w \| _ { 1 } ^ { 2 } + \frac { D _ { \lambda , \infty } ^ { 2 } } { \rho _ { x } } \| \Delta v \| _ { 1 } ^ { 2 } } \\ & { } & { \leq \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w ^ { \prime } \| w ) + \frac { 2 D _ { \lambda , \infty } ^ { 2 } } { \rho _ { x } } D _ { \mathrm { K L } } ( v ^ { \prime } \| v ) } \\ & { } & { ( \mathrm { P i n s k e r ' s ~ I n e q u a l i t y : ~ } D _ { \mathrm { K L } } ( P \| Q ) \geq \frac { \| P - Q \| _ { 1 } ^ { 2 } } { 2 } ) } \\ & { } & { \leq \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w ^ { \prime } \| w ) + \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v ^ { \prime } \| v ) } \qquad ( \rho _ { x } \rho _ { v } \geq 4 D _ { \lambda , \infty } ^ { 2 } )  \end{array}
$$

This implies

$$
\Phi _ { t } ^ { \lambda } ( w ^ { \prime } , v ^ { \prime } ) - \Phi _ { t } ^ { \lambda } ( w , v ) \leq \Big \langle \nabla \Phi _ { t } ^ { \lambda } ( w , v ) , ( \Delta w , \Delta v ) \Big \rangle - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w ^ { \prime } \| w ) - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v ^ { \prime } \| v ) .\tag{11}
$$

Thus joint concavity of $\Phi _ { t } ^ { \lambda }$ follows because $D _ { \mathrm { K L } } ( w ^ { \prime } | | w ) \geq 0$ and $D _ { \mathrm { K L } } ( v ^ { \prime } | | v ) \geq 0$ . The joint concavity of $F _ { t } ^ { \lambda } ( x , \cdot , \cdot )$ can be derived analogously because the nonlinear terms are the same in both cases.

Now, when $( w ^ { * } , v ^ { * } )$ maximizes $\Phi _ { t } ^ { \lambda } , w ^ { * }$ and $v ^ { * }$ must have full support: consider $w ^ { * }$ and suppose it has a zero coordinate. Then shifting ε mass from a positive coordinate to a 0 coordinate of $w ^ { * }$ changes the objective by $\rho _ { x } \varepsilon \log ( 1 / \varepsilon ) + O ( \varepsilon ) > 0$ , for ε sufficiently small, which contradicts the optimality of $w ^ { * }$ . The argument for $v ^ { * }$ is analogous. Additionally, $\big \langle \nabla \Phi _ { t } ^ { \lambda } ( w ^ { * } , v ^ { * } ) , ( w - w ^ { * } , v - v ^ { * } ) \big \rangle \leq 0 , \forall ( w , v ) \in \Delta ( \mathcal { F } ) \times \Delta _ { m + 1 }$ by first-order necessary optimality condition for $\Phi _ { t } ^ { \lambda }$ over the convex domain.

Thus by (11), we have for every feasible $( w , v )$

$$
\begin{array} { r l } & { \Phi _ { t } ^ { \lambda } ( w , v ) - \Phi _ { t } ^ { \lambda } ( w ^ { * } , v ^ { * } ) \leq \Big \langle \nabla \Phi _ { t } ^ { \lambda } ( w ^ { * } , v ^ { * } ) , ( w - w ^ { * } , v - v ^ { * } ) \Big \rangle - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| w ^ { * } ) - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v ^ { * } ) } \\ & { \qquad \leq - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| w ^ { * } ) - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v ^ { * } ) } \end{array}
$$

Substituting $M _ { t } ^ { \lambda } = \Phi _ { t } ^ { \lambda } ( w ^ { * } , v ^ { * } )$ finishes the proof.

## 4.3 Core Algorithms

In this section, we give the core algorithms for both the ORA and OLRC regimes. These algorithms guarantee optimal regret bounds against D-pacing experts without exceeding the budgets B, where $D = ( D _ { i } ) _ { i \in [ m ] }$ and $B = ( B _ { i } ) _ { i \in [ m ] }$ . Section 5 then describes how to extend the algorithm to compete against D-subpacing experts, to be subpacing itself, and to convert the fractional guarantees to sampled play with pathwise budget feasibility and high-probability regret guarantees.

With pre-decision feedback. In the ORA regime, the core algorithm, stated in Algorithm 2, follows the recommendations of the following controller, as long as it is not about to exceed any budget $B _ { i } \colon$ : in each round, after observing the current reward and cost vectors, the controller recommends

$$
x _ { t } \in \arg \operatorname* { m i n } _ { x \in \Delta ( \mathcal { F } ) } \operatorname* { m a x } _ { w \in \Delta ( \mathcal { F } ) , v \in \Delta _ { m + 1 } } F _ { t } ^ { \lambda } ( x , w , v ) ,\tag{12}
$$

and the algorithm plays $p _ { t } = \pi _ { t } ( x _ { t } ) ( { \mathrm { o r } } \bot { \mathrm { i f } } C _ { t - 1 } ^ { i } > B _ { i } - 1$ for some resource $i )$

```latex
Algorithm 2: Core ORA Algorithm
Let $\lambda _ { i } = \eta _ { i } = \operatorname* { m i n } \{ \alpha _ { i } , \beta _ { i } \}$ where
$\alpha _ { i }$ is the reward-to-cost ratio of resource $i$
$\beta _ { i } = 1 / \operatorname* { m i n } _ { t } b _ { t } ^ { i }$ is the reciprocal of the smallest term in the baseline spending pattern $b ^ { i }$
for $t = 1 , \dots , T$ do
If $\begin{array} { r } { C _ { t - 1 } ^ { i } = \sum _ { s < t } c _ { s } ^ { i } ( p _ { s } ) > B _ { i } - 1 } \end{array}$ for some resource i, break and play $\perp$ from round t onward
Otherwise, observe the current round’s rewards vector $r _ { t }$ and cost vectors $( c _ { t } ^ { i } ) _ { i \in [ m ] }$
Define
$\begin{array} { r } { F _ { t } ^ { \lambda } ( x , w , v ) : = \Phi _ { t - 1 } ^ { \lambda } ( w , v ) + r _ { t } ( w ) - r _ { t } ( x ) + \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } \big ( c _ { t } ^ { i } ( x ) - c _ { t } ^ { i } ( w ) \big ) } \end{array}$
where $\begin{array} { r } { \Phi _ { t - 1 } ^ { \lambda } ( w , v ) : = \sum _ { f } w _ { f } R _ { t - 1 } ^ { f } + \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } E _ { t - 1 } ^ { i } ( w ) - \rho _ { x } \Psi _ { x } ( w ) - \rho _ { v } \Psi _ { v } ( v ) } \end{array}$
Choose
$\begin{array} { r } { x _ { t } \in \arg \operatorname* { m i n } _ { x \in \Delta ( \mathcal { F } ) } \operatorname* { m a x } _ { w \in \Delta ( \mathcal { F } ) , v \in \Delta _ { m + 1 } } F _ { t } ^ { \lambda } ( x , w , v ) } \end{array}$
Play $p _ { t } = \pi _ { t } ( x _ { t } )$ and receive fractional reward $r _ { t } ( p _ { t } )$ and fractional costs $c _ { t } ^ { i } ( p _ { t } )$
end
```

Intuitively, $F _ { t } ^ { \lambda }$ is the hypothetical potential function at t if the algorithm were to play x in round $t ,$ and the controller recommends the $x _ { t }$ that minimizes the potential against the worst expert mixture and resource distribution. This generalizes the decision rule for the alternating red and blue ball problem in Section 3.2.

Under the regularization condition in Lemma 4.1, this controller guarantees that the potential does not increase over time.

Lemma 4.2. Under (9), before the hard stop the controller with pre-decision feedback satisfies

$$
M _ { t } ^ { \lambda } \leq M _ { t - 1 } ^ { \lambda } .
$$

Proof. For fixed $( w , v )$ , the potential $F _ { t } ^ { \lambda }$ is affine in $x ,$ while Lemma 4.1 shows that, under its regularization condition, it is jointly concave in $( w , v )$ for fixed x. Hence, Sion’s minimax theorem applies. Moreover, $F _ { t } ^ { \lambda } ( w , w , v ) = \Phi _ { t - 1 } ^ { \lambda } ( w , v )$ and $\Phi _ { t } ^ { \lambda } ( w , v ) = F _ { t } ^ { \lambda } ( x _ { t } , w , v )$ , so

Without pre-decision feedback. In this regime, the controller does not have access to the current round’s rewards and costs in the OLRC regime, so it cannot recommend the distribution x over experts that minimizes the worst-case potential ma $\mathfrak { c } _ { w , v } \breve { F _ { t } ^ { \lambda } } ( x , w , v )$ . Instead, the controller simply recommends the output of Followthe-Regularized-Leader (FTRL) on the regularized objective max<sub>v</sub> $\boldsymbol { \Phi } _ { t - 1 } ^ { \lambda } ( \cdot , \boldsymbol { v } )$ . As before, the algorithm, stated in Algorithm $^ { 3 , }$ follows the controller’s recommendation, as long as it is not about to exceed any budget $B _ { i }$ . More specifically, the controller chooses

$$
( x _ { t } , v _ { t } ) \in \arg \operatorname* { m a x } _ { w , v } \Phi _ { t - 1 } ^ { \lambda } ( w , v ) ,\tag{13}
$$

and the algorithm plays $p _ { t } = \pi _ { t } ( x _ { t } ) ( { \mathrm { o r } } \bot { \mathrm { i f } } C _ { t - 1 } ^ { i } > B _ { i } - 1$ for some resource $i )$ . Similar to how the stability of FTRL bounds its regret in the standard online learning setting, we will see that the stability of the controller bounds the increments of the potential ${ M } _ { t } ^ { \lambda }$

```latex
Algorithm 3: Core OLRC Algorithm
Let $\lambda _ { i } = \eta _ { i } =$ min $\{ \alpha _ { i } , \beta _ { i } \}$ where
$\alpha _ { i }$ is the reward-to-cost ratio of resource $i$
$\beta _ { i } = 1 / \operatorname* { m i n } _ { t } b _ { t } ^ { i }$ is the reciprocal of the smallest term in the baseline spending pattern $b ^ { i }$
for $t = 1 , \dots , T$ do
If $\begin{array} { r } { C _ { t - 1 } ^ { i } = \sum _ { s < t } c _ { s } ^ { i } ( p _ { s } ) > B _ { i } - 1 } \end{array}$ for some resource $i ,$ break and play $\perp$ from round t onward
Otherwise, define
$\begin{array} { r } { \Phi _ { t - 1 } ^ { \lambda } ( w , v ) : = \sum _ { f } w _ { f } R _ { t - 1 } ^ { f } + \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } E _ { t - 1 } ^ { i } ( w ) - \rho _ { x } \Psi _ { x } ( w ) - \rho _ { v } \Psi _ { v } ( v ) } \end{array}$
Let
$\begin{array} { r } { ( x _ { t } , v _ { t } ) \in \arg \operatorname* { m a x } _ { w \in \Delta ( \mathcal { F } ) , v \in \Delta _ { m + 1 } } \Phi _ { t - 1 } ^ { \lambda } ( w , v ) } \end{array}$
Play $p _ { t } = \pi _ { t } ( x _ { t } )$ and receive fractional reward $r _ { t } ( p _ { t } )$ and fractional costs $c _ { t } ^ { i } ( p _ { t } )$
Observe reward vector $r _ { t }$ and cost vectors $( c _ { t } ^ { i } ) _ { i \in [ m ] }$
end
```

Lemma 4.3. Under (9), before the hard stop the controller without pre-decision feedback satisfies

$$
M _ { t } ^ { \lambda } - M _ { t - 1 } ^ { \lambda } \leq \frac { ( 1 + \lambda _ { \operatorname* { m a x } } ) ^ { 2 } } { \rho _ { x } } .
$$

Proof. Before the hard stop, the additional algorithm reward in round t is $r _ { t } ( x _ { t } )$ and the additional overspending with respect to expert mixture w is $\begin{array} { r } { \hat { E _ { t } ^ { i } } ( w ) - E _ { t - 1 } ^ { i } ( w ) = c _ { t } ^ { i } ( x _ { t } ) - c _ { t } ^ { i } ( w ) = \langle c _ { t } ^ { i } , x _ { t } - w \rangle } \end{array}$ . Hence

$$
\Phi _ { t } ^ { \lambda } ( w , v ) - \Phi _ { t - 1 } ^ { \lambda } ( w , v ) = \left. r _ { t } - \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } c _ { t } ^ { i } , w - x _ { t } \right. \leq ( 1 + \lambda _ { \operatorname* { m a x } } ) \| w - x _ { t } \| _ { 1 } .
$$

Since $( x _ { t } , v _ { t } )$ maximizes $\Phi _ { t - 1 } ^ { \lambda }$ , we have

$$
\begin{array} { r l r } { \Phi _ { t } ^ { \lambda } ( w , v ) - M _ { t - 1 } ^ { \lambda } = \Phi _ { t } ^ { \lambda } ( w , v ) - \Phi _ { t - 1 } ^ { \lambda } ( w , v ) + \Phi _ { t - 1 } ^ { \lambda } ( w , v ) - M _ { t - 1 } ^ { \lambda } } \\ & { } & { \qquad \leq ( 1 + \lambda _ { \operatorname* { m a x } } ) \| w - x _ { t } \| _ { 1 } - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| x _ { t } ) - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v _ { t } ) \qquad \mathrm { ( b y ~ L e m m a ~ 4 . 1 ) } } \\ & { } & { \qquad \leq ( 1 + \lambda _ { \operatorname* { m a x } } ) \| w - x _ { t } \| _ { 1 } - \frac { \rho _ { x } } { 4 } \| w - x _ { t } \| _ { 1 } ^ { 2 } \qquad \mathrm { ( P i n s k e r ' s ~ I n e q u a l i t y ) } } \\ & { } & { \qquad \leq \frac { ( 1 + \lambda _ { \operatorname* { m a x } } ) ^ { 2 } } { \rho _ { x } } } \end{array}
$$

where the last line follows by optimizing the quadratic function in $\lVert \boldsymbol { w } - \boldsymbol { x } _ { t } \rVert _ { 1 }$ .

## 4.4 Completing the Analysis

We first derive a lemma for bounding the rewards and costs of an expert after the algorithm hard stops. Let $\begin{array} { r } { A _ { t } ^ { f , j } : = \sum _ { s \leq t } c _ { s } ^ { f , j } } \end{array}$ be cumulative spending of expert $f$ in resource $j$ by the end of round t and $\begin{array} { r } { B _ { t } ^ { j } : = \sum _ { s \leq t } b _ { s } ^ { j } } \end{array}$ be the cumulative baseline spending in resource $j$ by the end of round t. Recall that $\begin{array} { r } { C _ { t } ^ { j } = \sum _ { s < t } c _ { s } ^ { j } ( p _ { s } ) } \end{array}$ is the analogous term for algorithm spending.

Lemma 4.4. Suppose that by the end ofround $\tau ,$ there exists some resource $j$ such that $C _ { \tau } ^ { j } \geq B _ { j } - h _ { j }$ for some $h _ { j } \geq 0$ . Then

$$
\sum _ { t > \tau } c _ { t } ^ { f , j } \leq E _ { \tau } ^ { j } ( e _ { f } ) + h _ { j } + D _ { j } ,\tag{14}
$$

Learning with Constraint

$$
\sum _ { t > \tau } r _ { t } ^ { f } \leq \alpha _ { j } \big ( E _ { \tau } ^ { j } ( e _ { f } ) + h _ { j } + D _ { j } \big ) + \sum _ { t > \tau } \varepsilon _ { t } ^ { j } ,\tag{15}
$$

$$
I f \beta _ { j } < \infty , \sum _ { t > \tau } r _ { t } ^ { f } \leq \beta _ { j } \big ( E _ { \tau } ^ { j } ( e _ { f } ) + h _ { j } + D _ { j } \big ) .\tag{16}
$$

Proof. Since each expert is D-pacing with respect to the baselines, we have $A _ { T } ^ { f , j } \leq B _ { j } + D _ { j }$ . Additionally, by definition, $E _ { \tau } ^ { j } ( e _ { f } ) = C _ { \tau } ^ { j } - A _ { \tau } ^ { f , j }$ . Thus

$$
\begin{array} { r l r } {  { \sum _ { t > \tau } c _ { t } ^ { f , j } = A _ { T } ^ { f , j } - A _ { \tau } ^ { f , j } \le B _ { j } + D _ { j } - \big ( C _ { \tau } ^ { j } - E _ { \tau } ^ { j } ( e _ { f } ) \big ) } } \\ & { } & \\ & { } & { \le \big ( B _ { j } - C _ { \tau } ^ { j } \big ) + E _ { \tau } ^ { j } ( e _ { f } ) + D _ { j } } \\ & { } & { \le h _ { j } + E _ { \tau } ^ { j } ( e _ { f } ) + D _ { j } . } \end{array}
$$

By Assumption 2.1, we have $r _ { t } ^ { f } \le \alpha _ { j } c _ { t } ^ { f , j } + \varepsilon _ { t } ^ { j }$ . Summing this over t together with (14) gives (15).

Additionally, we have

$$
\begin{array} { r l } {  { \sum _ { t > \tau } b _ { t } ^ { j } = B _ { j } - B _ { \tau } ^ { j } } } \\ & { = ( B _ { j } - C _ { \tau } ^ { j } ) + ( C _ { \tau } ^ { j } - B _ { \tau } ^ { j } ) } \\ & { \le h _ { j } + ( C _ { \tau } ^ { j } - A _ { \tau } ^ { f , j } ) + ( A _ { \tau } ^ { f , j } - B _ { \tau } ^ { j } ) } \\ & { \le h _ { j } + E _ { \tau } ^ { j } ( e _ { f } ) + D _ { j } . } \end{array}
$$

$$
\begin{array} { r } { ( \sum _ { t = 1 } ^ { T } b _ { t } ^ { j } = B _ { j } ) } \end{array}
$$

$$
( A _ { \tau } ^ { f , j } - B _ { \tau } ^ { j } \leq D _ { j } )
$$

$\mathrm { I f } \ \beta _ { j } < \infty ,$ , then $\begin{array} { r } { \sum _ { t > \tau } r _ { t } ^ { f } \le T - \tau \le \beta _ { j } \sum _ { t > \tau } b _ { t } ^ { j } } \end{array}$ , this gives (16).

Theorem 4.5. Assume $\lambda _ { i } \geq \eta _ { i }$ for every resource and choose $\rho _ { x } , \rho _ { v } > 0 \ s a t i s f y i n g \ ( 9 )$ . Then both algorithms are budget feasible via fractional play. Moreover, for every expert f simultaneously,

$$
\mathrm { R e g } _ { T } ^ { f } \leq \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) + \mathsf { T a i l } , \qquad w i t h p r e - d e c i s i o n f e e d b a c k ,\tag{17}
$$

$$
\mathrm { R e g } _ { T } ^ { f } \leq \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) + \frac { T ( 1 + \lambda _ { \operatorname* { m a x } } ) ^ { 2 } } { \rho _ { x } } + \mathsf { T a i l } , w i t h o u t p r e - d e c i s i o n f e e d b a c k ,\tag{18}
$$

where

$$
\mathsf { T a i l } : = \operatorname* { m a x } _ { i } \left\{ \eta _ { i } ( 1 + D _ { i } ) + \mathbf { 1 } \{ \alpha _ { i } < \beta _ { i } \} V _ { i } \right\} .\tag{19}
$$

Proof. The stopping rule gives fractional budget feasibility. Let

$$
\mathsf { S t a b } _ { s } : = \left\{ \sum _ { t \leq s } ^ { 0 , } \frac { \mathsf { w i t h \ p r e \mathrm { - } d e c i s i o n \ f e e d b a c k } , } { \rho _ { x } } \right.
$$

Since $M _ { 0 } ^ { \lambda } = 0$ , Lemma 4.2 and Lemma 4.3 imply $M _ { s } ^ { \lambda } \le \mathsf { S t a b } _ { s }$ for every round s before the algorithm hard stops.

Recall the potential function at T is $\begin{array} { r } { \Phi _ { T } ^ { \lambda } ( w , v ) : = \sum _ { f } w _ { f } R _ { T } ^ { f } + \sum _ { i = 1 } ^ { m } \lambda _ { i } v _ { i } E _ { T } ^ { i } ( w ) - \rho _ { x } \Psi _ { x } ( w ) - \rho _ { v } \Psi _ { v } ( v ) } \end{array}$ . If no hard stop occurs, evaluating the potential function at $( e _ { f } , e _ { 0 } )$ gives

$$
R _ { T } ^ { f } + 0 - \rho _ { x } \log F - \rho _ { v } \log ( m + 1 ) \leq 5 \mathsf { t a b } _ { T } ,
$$

which gives the desired bounds.

Otherwise let resource $j$ trigger the hard stop at the end of round τ. Then $B _ { j } - C _ { \tau } ^ { j } < 1$ . Set $s _ { j } : = \eta _ { j } / \lambda _ { j } \in$ [0, 1] and evaluate the prefix potential at $\left( e _ { f } , ( 1 - s _ { j } ) e _ { 0 } + s _ { j } e _ { j } \right)$ . The resource term is $\lambda _ { j } s _ { j } E _ { \tau } ^ { j } ( e _ { f } ) =$ $\eta _ { j } E _ { \tau } ^ { j } ( e _ { f } )$ , while Ψ<sub>x</sub>(e<sub>f</sub>) = log F and $\Psi _ { v } ( ( 1 - s _ { j } ) e _ { 0 } + s _ { j } e _ { j } ) \leq \log ( m + 1 )$ . Therefore

$$
R _ { \tau } ^ { f } + \eta _ { j } E _ { \tau } ^ { j } ( e _ { f } ) - \rho _ { x } \log F - \rho _ { v } \log ( m + 1 ) \leq 5 \mathsf { t a b } _ { \tau } .
$$

Since $B _ { j } - C _ { \tau } ^ { j } < 1$ , Lemma 4.4 applies with $h _ { j } = 1$ . If $\alpha _ { j } < \beta _ { j }$ , use (15); then $\eta _ { j } = \alpha _ { j }$ and the suffix residual is at most $V _ { j } . \mathrm { I f } \alpha _ { j } \geq \beta _ { j }$ , use (16); then $\eta _ { j } = \beta _ { j }$ . Thus in both cases

$$
\sum _ { t > \tau } r _ { t } ^ { f } \leq \eta _ { j } \big ( E _ { \tau } ^ { j } ( e _ { f } ) + 1 + D _ { j } \big ) + { \bf 1 } \big \{ \alpha _ { j } < \beta _ { j } \big \} V _ { j } \leq \eta _ { j } E _ { \tau } ^ { j } ( e _ { f } ) + \sf { T a i l } .
$$

Thus we have

$$
\mathrm { R e g } _ { T } ^ { f } \leq R _ { \tau } ^ { f } + \sum _ { t > \tau } r _ { t } ^ { f } \leq 5 \mathsf { t a b } _ { \tau } + \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) + \mathsf { T a i l }
$$

For the core algorithm, we use $\lambda _ { i } = \eta _ { i } = \operatorname* { m i n } \{ \alpha _ { i } , \beta _ { i } \}$ . Define

$$
D _ { \eta , \infty } : = \operatorname* { m a x } _ { i } \eta _ { i } D _ { i } , \qquad \eta _ { \mathrm { m a x } } : = \operatorname* { m a x } _ { i } \eta _ { i } .
$$

Theorem 4.6 (Main core guarantee). Choose $\lambda _ { i } = \eta _ { i } , \forall i \in [ m ]$ . Let $\rho _ { x } , \rho _ { v } > 0$ satisfy $\rho _ { x } \rho _ { v } \geq 4 D _ { \eta , \infty } ^ { 2 }$ . Then both algorithms are budgetfeasible viafractional play, andfor every expert f simultaneously,

$$
\mathrm { R e g } _ { T } ^ { f } \leq \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) + { \sf T a i l } ,\tag{20}
$$

$$
\mathrm { R e g } _ { T } ^ { f } \leq \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) + \frac { T ( 1 + \eta _ { \operatorname* { m a x } } ) ^ { 2 } } { \rho _ { x } } + \mathsf { T a i l } , w i t h o u t p r e - d e c i s i o n f e e d b a c k ,\tag{21}
$$

where

$$
\mathsf { T a i l } : = \operatorname* { m a x } _ { i } \left\{ \eta _ { i } ( 1 + D _ { i } ) + \mathbf { 1 } \{ \alpha _ { i } < \beta _ { i } \} V _ { i } \right\} .
$$

Optimizing the regularization parameters gives

$$
\mathrm { R e g } _ { T } ^ { f } \leq \mathsf { T a i l } + O \left( D _ { \eta , \infty } \sqrt { \log { F \log ( m + 1 ) } } \right) ,
$$

with pre-decision feedback,

(22)

$$
\mathrm { R e g } _ { T } ^ { f } \le \mathsf { T a i l } + O \left( ( 1 + \eta _ { \operatorname* { m a x } } ) \sqrt { T \log F } + D _ { \eta , \infty } \sqrt { \log F \log ( m + 1 ) } \right) , w i t h o u t p r e - d e c i s i o n f e d b a c k .\tag{23}
$$

Proof. The untuned regret guarantees follow from Theorem 4.5 with $\lambda = \eta$ . For the optimized guarantees, take $\dot { \rho } _ { v } = 4 D _ { \eta , \infty } ^ { 2 } / \rho _ { x }$ . By the standing assumption that $D _ { i } > 0$ for some i and the positivity of every η , one has $D _ { \eta , \infty } > 0$ . With pre-decision feedback, minimizing

$$
\rho _ { x } \log { F } + \frac { 4 D _ { \eta , \infty } ^ { 2 } \log ( m + 1 ) } { \rho _ { x } }
$$

gives $O \left( D _ { \eta , \infty } { \sqrt { \log F \log ( m + 1 ) } } \right)$ . Without pre-decision feedback, minimizing

$$
\rho _ { x } \log { F } + \frac { 4 D _ { \eta , \infty } ^ { 2 } \log ( m + 1 ) + T ( 1 + \eta _ { \operatorname* { m a x } } ) ^ { 2 } } { \rho _ { x } } ,
$$

and using ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ give (23).

## 5 Extensions

The following three extensions are modular and may be composed in any combination. The first enlarges the comparator class from pacing to subpacing experts, the second additionally requires the learner itself to be subpacing, and the third converts any resulting fractional controller to sampled play with pathwise budget feasibility and high-probability guarantees, under full-information. We first give a characterization of subpacing used by the first two extensions.

The key part of the characterization is that if a spending pattern is D-subpacing with respect to a baseline, then it cannot overspend the baseline by more than D on any prefix, or by more than 2D over any interval. This allows us to convert a subpacing spending pattern into a pacing one by artificially raising its spending when it falls too far below the baseline, without causing the resulting path to exceed the baseline by more than D on any prefix.

Additionally, we show a downward closure property of the set of D-subpacing spending patterns.

Lemma 5.1 (Subpacing characterization and downward closure). Let $\begin{array} { r } { A _ { t } : = \sum _ { s \leq t } c _ { s } } \end{array}$ and $\begin{array} { r } { B _ { t } : = \sum _ { s \leq t } b _ { s } } \end{array}$ with $A _ { 0 } = B _ { 0 } = 0$ . The path c is D-subpacing with respect to b if and only if

$$
A _ { t } - B _ { t } \leq D \quad \forall t , \qquad \left( A _ { t } - A _ { s } \right) - \left( B _ { t } - B _ { s } \right) \leq 2 D \quad \forall 0 \leq s < t \leq T .\tag{24}
$$

Consequently, $i f 0 \le c _ { t } \le \bar { c } _ { t }$ pointwise and c¯ is D-subpacing with respect to $b ,$ then c is also D-subpacing.

Proof. Suppose first that c is D-subpacing with respect to b and $d _ { t } \in [ 0 , b _ { t } ]$ is a subpacing witness. Let $\begin{array} { r } { X _ { t } : = \sum _ { s < t } d _ { s } } \end{array}$ . Since $X _ { t } \le B _ { t }$ and $\left\lceil A _ { t } - \bar { X } _ { t } \right\rceil \leq D , \bar { A } _ { t } - B _ { t } \leq A _ { t } - \dot { X _ { t } } \leq \bar { D }$ . For $0 \leq s < t$

$$
\begin{array} { c } { { ( A _ { t } - A _ { s } ) - ( B _ { t } - B _ { s } ) \leq ( A _ { t } - A _ { s } ) - ( X _ { t } - X _ { s } ) } } \\ { { = ( A _ { t } - X _ { t } ) - ( A _ { s } - X _ { s } ) \leq 2 D , } } \end{array}
$$

which proves necessity. The first inequality follows from the fact that

$$
X _ { t } - X _ { s } = \sum _ { \tau = s + 1 } ^ { t } d _ { \tau } \leq \sum _ { \tau = s + 1 } ^ { t } b _ { \tau } = B _ { t } - B _ { s } .
$$

Conversely, assume (24). Let

$$
K _ { t } : = \operatorname* { m a x } _ { 0 \leq s \leq t } ( B _ { s } - A _ { s } - D ) _ { + }
$$

be the maximum prefix underspending of c with respect to the baseline beyond D by the end of round t.   
Define $X _ { t } : = B _ { t } \bar { - } K _ { t }$ and $d _ { t } : \bar { = } X _ { t } - X _ { t - 1 }$ . We will show that d is a subpacing witness of c.

Since $K _ { t } \ge 0 , X _ { t } \le B _ { t }$ . We claim that $| A _ { t } - X _ { t } | \leq D$ . For the lower bound, $K _ { t } \ge ( B _ { t } - A _ { t } - D ) .$ <sub>+</sub> implies $A _ { t } - B _ { t } + K _ { t } \geq - D , \operatorname { i . e . } A _ { t } - X _ { t } \geq - D$ . For the upper bound, if $K _ { t } = 0$ , then $A _ { t } - X _ { t } = A _ { t } - B _ { t } \leq D$ If $K _ { t } > 0$ , choose $s \leq t$ attaining the maximum, so $\bar { K _ { t } } = B _ { s } - A _ { s } - D$ . If $s = t$ , then $A _ { t } - X _ { t } = - D$ . If $s < t$ , the second inequality in (24) gives

$$
A _ { t } - X _ { t } = A _ { t } - B _ { t } + B _ { s } - A _ { s } - D = ( A _ { t } - A _ { s } ) - ( B _ { t } - B _ { s } ) - D \le D .
$$

Thus, $| A _ { t } - X _ { t } | \leq D$ for every t.

It remains to show that $d _ { t } = X _ { t } - X _ { t - 1 }$ lies in $[ 0 , b _ { t } ]$ . Since $K _ { t }$ is nondecreasing, $d _ { t } = b _ { t } - ( K _ { t } - K _ { t - 1 } ) \leq b _ { t }$ If $K _ { t } = K _ { t - 1 }$ , then $d _ { t } = b _ { t } \geq 0$ . Otherwise the new maximum is attained at t, so $K _ { t } = B _ { t } - A _ { t } - D$ and $K _ { t - 1 } \geq B _ { t - 1 } - A _ { t - 1 } - D$ . Hence $K _ { t } - K _ { t - 1 } \le b _ { t } - c _ { t }$ , which implies $d _ { t } \geq c _ { t } \geq 0$ . Therefore $\left( d _ { t } \right)$ is a valid witness and c is D-subpacing.

For downward closure, let $\begin{array} { r } { \bar { A } _ { t } : = \sum _ { s < t } \bar { c } _ { s } } \end{array}$ . Pointwise domination implies $A _ { t } - B _ { t } \leq \bar { A } _ { t } - B _ { t }$ and, for every $s < t , ( A _ { t } - A _ { s } ) - ( B _ { t } - B _ { s } ) \leq ( \bar { A } _ { t } - \bar { A } _ { s } ) - ( B _ { t } - B _ { s } )$ . Thus if c¯ satisfies (24), then so does $^ { c , }$ and the characterization proves the claim. □

## 5.1 Extension 1: Subpacing Experts

Note that the core algorithm cannot handle subpacing experts because these experts may underspend too much and cause the potential $\Phi _ { t } ^ { \lambda }$ to become non-concave in $( w , v )$ . To extend the core algorithm to handle subpacing experts, we artificially raise the spending of these experts so that their cumulative (artificial) spending in each resource i is never $D _ { i }$ below the baseline’s spending in this resource. We then feed each expert’s artificial costs into the core algorithms instead of their true costs.

At a high level, we inflate an expert’s spending in resource i in the following way. The first time t an expert $f$ spends less than the baseline by $D _ { i } .$ that is,

$$
S _ { t } ^ { f , i } = \sum _ { \tau = 1 } ^ { t } ( c _ { \tau } ^ { f , i } - b _ { \tau } ^ { i } ) < - D _ { i } \quad \mathrm { a n d } \quad S _ { s } ^ { f , i } = \sum _ { \tau = 1 } ^ { s } ( c _ { \tau } ^ { f , i } - b _ { \tau } ^ { i } ) \geq - D _ { i } \forall s < t ,
$$

we increase the costs faced by that expert so that her spending relative to the baseline does not decrease further and remains $- D _ { i }$ away from the baseline’s spending. That is, we inflate her costs so that her (artificial) spending is

$$
\bar { c } _ { t } ^ { f , i } : = b _ { t } ^ { i } - S _ { t - 1 } ^ { f , i } - D _ { i } .
$$

Note that

$$
S _ { t - 1 } ^ { f , i } + ( \bar { c } _ { t } ^ { f , i } - b _ { t } ^ { i } ) = - D _ { i } .
$$

Let

$$
\hat { S } _ { t } ^ { f , i } : = S _ { t - 1 } ^ { f , i } + ( \hat { c } _ { t } ^ { f , i } - b _ { t } ^ { i } ) .
$$

If the expert continues to underspend in the next round, that is, $c _ { t + 1 } ^ { f , i } < b _ { t + 1 } ^ { i }$ , then we continue to inflate her costs so that her (artificial) spending solves

$$
\bar { S } _ { t } ^ { f , i } + ( \bar { c } _ { t + 1 } ^ { f , i } - b _ { t + 1 } ^ { i } ) = - D _ { i } .
$$

Equivalently, $\bar { c } _ { t + 1 } ^ { f , i } : = b _ { t + 1 } ^ { i }$ since $\bar { S } _ { t } ^ { f , i } = - D _ { i }$ . Otherwise, $c _ { t + 1 } ^ { f , i } \geq b _ { t + 1 } ^ { i }$ , and $\bar { c } _ { t + 1 } ^ { f , i } : = c _ { t + 1 } ^ { f , i }$ , that is, we stop inflating the costs faced by the expert.

$\operatorname { I f } ,$ in a later round $t ^ { \prime } > t ,$ the expert’s (artificial) spending again falls below the baseline’s spending by more that $D _ { i }$ , that is,

$$
\bar { S } _ { t ^ { \prime } - 1 } ^ { f , i } + ( c _ { t ^ { \prime } } ^ { f , i } - b _ { t ^ { \prime } } ^ { i } ) < - D _ { i } \quad \mathrm { b u t } \quad \bar { S } _ { s } ^ { f , i } \geq - D _ { i } \forall s < t ^ { \prime } ,
$$

we once again artificially raise costs so that the difference $\bar { S } _ { t ^ { \prime } } ^ { f , i }$ never goes below $- D _ { i }$ . This process repeats as necessary.

In general, the artificial spending is given by

$$
\bar { S } _ { 0 } ^ { f , i } : = 0 , \quad \bar { c } _ { t } ^ { f , i } : = \left\{ \begin{array} { l l } { c _ { t } ^ { f , i } } & { \bar { S } _ { t - 1 } ^ { f , i } + ( c _ { t } ^ { f , i } - b _ { t } ^ { i } ) \geq - D _ { i } } \\ { b _ { t } ^ { i } - \bar { S } _ { t - 1 } ^ { f , i } - D _ { i } } & { \mathrm { o . w . } } \end{array} \right. , \quad \bar { S } _ { t } ^ { f , i } : = \bar { S } _ { t - 1 } ^ { f , i } + ( \bar { c } _ { t } ^ { f , i } - b _ { t } ^ { i } )
$$

Equivalently,

$$
\bar { S } _ { 0 } ^ { f , i } : = 0 , \quad \delta _ { t } ^ { f , i } : = \left( - D _ { i } - \big ( \bar { S } _ { t - 1 } ^ { f , i } + c _ { t } ^ { f , i } - b _ { t } ^ { i } \big ) \right) _ { + } , \quad \bar { c } _ { t } ^ { f , i } : = c _ { t } ^ { f , i } + \delta _ { t } ^ { f , i } , \quad \bar { S } _ { t } ^ { f , i } : = \bar { S } _ { t - 1 } ^ { f , i } + \big ( \bar { c } _ { t } ^ { f , i } - b _ { t } ^ { i } \big )
$$

The main concern with artificially inflating costs is that the expert’s artificial spending may exceed the baseline by more than $D _ { i }$ at a later time. But note that for the expert’s spending to go from more than $D _ { i }$ below the baseline’s to more than $D _ { i }$ above, the increase in the difference in spending must exceed $2 D _ { i }$ . This is exactly what Lemma 5.1 precludes.

Lemma 5.2. Suppose each expert’s spending pattern in resource i is only $D _ { i }$ -subpacing. Define

$$
\bar { S } _ { 0 } ^ { f , i } : = 0 , \quad \delta _ { t } ^ { f , i } : = ( - D _ { i } - ( \bar { S } _ { t - 1 } ^ { f , i } + c _ { t } ^ { f , i } - b _ { t } ^ { i } ) ) _ { + } , \quad \bar { c } _ { t } ^ { f , i } : = c _ { t } ^ { f , i } + \delta _ { t } ^ { f , i } , \quad \bar { S } _ { t } ^ { f , i } : = \bar { S } _ { t - 1 } ^ { f , i } + ( \bar { c } _ { t } ^ { f , i } - b _ { t } ^ { i } ) .
$$

Then,for every expert $f ,$ resource i and round t,

$$
0 \leq c _ { t } ^ { f , i } \leq \bar { c } _ { t } ^ { f , i } \leq 1 , \qquad | \bar { S } _ { t } ^ { f , i } | \leq D _ { i } , \qquad r _ { t } ^ { f } \leq \alpha _ { i } \bar { c } _ { t } ^ { f , i } + \varepsilon _ { t } ^ { i } .
$$

That is, replacing $c _ { t } ^ { f , i }$ by $\bar { c } _ { t } ^ { f , i }$ produces a paced instance with the same baseline spending pattern $b _ { i } ,$ budget $B _ { i } ,$ , Kolmogorov bound $D _ { i } ,$ reward-to-cost ratio $\alpha _ { i } ,$ , residuals $( \varepsilon _ { t } ^ { i } )$ <sub>t</sub>, and violation budgets $V _ { i } .$

Proof. Fix an expert-resource pair $( f , i )$ and suppress these indices. Write $\begin{array} { r } { A _ { t } : = \sum _ { s \leq t } c _ { s } , B _ { t } : = \sum _ { s \leq t } b _ { s } } \end{array}$ and $\begin{array} { r } { K _ { t } : = \sum _ { s \leq t } \delta _ { s } } \end{array}$ . We show that

$$
\bar { S } _ { t } = A _ { t } - B _ { t } + K _ { t } \quad \mathrm { a n d } \quad K _ { t } = \left( \operatorname* { m a x } _ { 0 \leq s \leq t } ( B _ { s } - A _ { s } ) - D _ { i } \right) _ { + }
$$

by induction. If these equalities hold in round $t - 1$ , then the update gives

$$
\begin{array} { r l } {  { K _ { t } = K _ { t - 1 } + ( - D _ { i } - ( \bar { S } _ { t - 1 } ^ { f , i } + c _ { t } ^ { f , i } - b _ { t } ^ { i } ) ) _ { + } } } \\ & { = K _ { t - 1 } + ( B _ { t } - A _ { t } - K _ { t - 1 } - D _ { i } ) _ { + } } \\ & { = \operatorname* { m a x } \{ K _ { t - 1 } , B _ { t } - A _ { t } - D _ { i } \} } \\ & { = \bigg ( \operatorname* { m a x } _ { 0 \leq s \leq t } ( B _ { s } - A _ { s } ) - D _ { i } \bigg ) _ { + } } \end{array}
$$

$$
K _ { t }
$$

$$
\delta _ { t } )\tag{IH}
$$

Meanwhile,

$$
\begin{array} { r l } & { \bar { S } _ { t } = \bar { S } _ { t - 1 } + ( \bar { c } _ { t } - b _ { t } ) } \\ & { \quad = A _ { t - 1 } - B _ { t - 1 } + K _ { t - 1 } + ( c _ { t } + \delta _ { t } - b _ { t } ) } \\ & { \quad = A _ { t } - B _ { t } + K _ { t } } \end{array}
$$

$$
\begin{array} { r l r } & { } & { ( \mathrm { d e f n i t i o n o f } \ \bar { S } _ { t } ) } \\ & { } & { ( \mathrm { I H } ; \mathrm { d e f n i t i o n o f } \ \bar { c } _ { t } ) } \\ & { } & { \mathrm { ( d e f i n i t i o n s ~ o f } \ A _ { t } , B _ { t } , \mathrm { a n d } \ K _ { t } ) } \end{array}
$$

We now prove that the expert’s artificial spending pattern is $D _ { i } { \mathrm { - p a c e d } } ,$ , i.e. $| \bar { S } _ { t } | \le D _ { i }$ for every prefix. Since the original spending pattern $S _ { t } = \bar { \boldsymbol { A } } _ { t } - \bar { \boldsymbol { B } } _ { t }$ is D<sub>i</sub>-subpacing, Lemma 5.1 gives $A _ { t } - B _ { t } \le D _ { i }$ and $( A _ { t } - \bar { A _ { s } } ) - ( \bar { B } _ { t } - \bar { B _ { s } } \bar { ) } \leq 2 D _ { i }$ for all $0 \leq s < t .$ . For the lower bound, $\bar { K } _ { t } \ge B _ { t } - A _ { t } - D _ { i }$ hence, $\bar { S } _ { t } = \dot { A } _ { t } - \dot { B } _ { t } + K _ { t } \ge - D _ { i }$ . For the upper bound, if ma $\mathrm { x } _ { s < t } ( B _ { s } - A _ { s } ) \leq D _ { i }$ , then $K _ { t } = 0$ and $\bar { S } _ { t } = A _ { t } - B _ { t } \le D _ { i }$ . Otherwise choose $s \leq t$ attaining ma $\mathsf { z } _ { r \leq t } ( B _ { r } - \mathsf { \bar { A } } _ { r } ) > D _ { i }$ . Then $K _ { t } = B _ { s } - A _ { s } - D _ { i }$ If $s = t ,$ this gives $\bar { S } _ { t } = - D _ { i }$ . If $s < t$ , then

$$
\begin{array} { r } { \bar { S } _ { t } = ( A _ { t } - A _ { s } ) - ( B _ { t } - B _ { s } ) - D _ { i } \leq 2 D _ { i } - D _ { i } = D _ { i } . } \end{array}
$$

Thus, $| \bar { S } _ { t } | \le D _ { i }$ for every t.

By construction, $\delta _ { t } \geq 0 ,$ , so $\bar { c } _ { t } = c _ { t } + \delta _ { t } \ge c _ { t } \ge 0$ . Also, $\delta _ { t } = K _ { t } - K _ { t - 1 }$ . If $K _ { t } = K _ { t - 1 }$ , then $\bar { c } _ { t } = c _ { t } \le 1$ If $K _ { t } > K _ { t - 1 }$ , then $K _ { t } = B _ { t } - A _ { t } - D _ { i }$ while $K _ { t - 1 } \geq B _ { t - 1 } - A _ { t - 1 } - D _ { i }$ , and therefore,

$$
0 < \delta _ { t } = K _ { t } - K _ { t - 1 } \leq b _ { t } - c _ { t } .
$$

Hence $\bar { c } _ { t } = c _ { t } + \delta _ { t } \leq b _ { t } \leq 1$ . Finally, monotonicity of costs implies that $r _ { t } ^ { f } \le \alpha _ { i } c _ { t } ^ { f , i } + \varepsilon _ { t } ^ { i } \le \alpha _ { i } \bar { c } _ { t } ^ { f , i } + \varepsilon _ { t } ^ { i }$ .

Theorem 5.3. Suppose each expert’s spending pattern in resource i is only $D _ { i }$ -subpacing. Define

$$
\bar { S } _ { 0 } ^ { f , i } : = 0 , \quad \delta _ { t } ^ { f , i } : = ( ( b _ { t } ^ { i } - c _ { t } ^ { f , i } ) - \bar { S } _ { t - 1 } ^ { f , i } - D _ { i } ) _ { + } , \quad \bar { c } _ { t } ^ { f , i } : = c _ { t } ^ { f , i } + \delta _ { t } ^ { f , i } , \quad \bar { S } _ { t } ^ { f , i } : = \bar { S } _ { t - 1 } ^ { f , i } + ( \bar { c } _ { t } ^ { f , i } - b _ { t } ^ { i } ) .
$$

Algorithms 2 and 3 with $c _ { t } ^ { f , i }$ replaced by $\bar { c } _ { t } ^ { f , i }$ everywhere<sup>10</sup> and the same choice ofparameters as in Theorem 4.6 are budgetfeasible viafractional play and satisfy

$$
\begin{array} { r l } & { \mathrm { R e g } _ { T } ^ { f } \leq \mathsf { T a i l } + O \left( D _ { \eta , \infty } \sqrt { \log { F } \log ( m + 1 ) } \right) \qquad } & { w i t h p r e - d e c i s i o n f e e d b a c k } \\ & { \mathrm { R e g } _ { T } ^ { f } \leq \mathsf { T a i l } + O \left( ( 1 + \eta _ { \operatorname* { m a x } } ) \sqrt { T \log { F } } + D _ { \eta , \infty } \sqrt { \log { F } \log ( m + 1 ) } \right) \quad w i t h o u t p r e - d e c i s i o n f e e d b a c k } \end{array}
$$

for every expert f simultaneously, where

$$
\mathsf { T a i l } : = \operatorname* { m a x } _ { i } \left\{ \eta _ { i } \left( 1 + D _ { i } \right) + \mathbf { 1 } \{ \alpha _ { i } < \beta _ { i } \} V _ { i } \right\} .
$$

Moreover, ifthe algorithms are subpacing with respect to the artificial costs and some Kolmogorov bound, then they are also subpacing with respect to the original cost and the same Kolmogorov bound.

Proof. Fractional budget feasibility and the regret guarantees follow from Lemma 5.2: the latter are immediate from the reduction implied by Lemma 5.2, while the hard stop condition with respect to the artificial costs implies that $\begin{array} { r } { \sum _ { t } \bar { c } _ { t } ^ { i } ( x _ { t } ) \dot { \mathbf { \Psi } } \leq B _ { i } } \end{array}$ , so point-wise domination of original costs by artificial costs (from Lemma 5.2) implies that

$$
\sum _ { t } c _ { t } ^ { i } ( x _ { t } ) \leq \sum _ { t } \bar { c } _ { t } ^ { i } ( x _ { t } ) \leq B _ { i }
$$

Point-wise domination and Lemma 5.1 then imply the final part of the result.

## 5.2 Extension 2: Subpacing Algorithm

The spending of the core algorithms may not be $D _ { i } { \mathrm { - s u b p a c i n g } }$ with respect to the baseline spending pattern because the potential sums the regret term and an overspending penalty term: if the core algorithms accumulate large negative regret with respect to an expert mixture w, then they may exceed the spending of that mixture by a large margin.

To prevent this from happening, we force the core algorithms to play the null action if their spending exceeds the baseline by too much. The resulting loss in reward from not following the controller’s recommendation is then bounded using the reward-to-cost ratio $\alpha _ { i } .$ . This is why we choose $\lambda _ { i } = \alpha _ { i }$ instead of $\eta _ { i }$ , since $\beta _ { i }$ offers no control over the rewards missed because of overspending too much compared to the baseline, which can occur in rounds before the hard-stop.

Additionally, for the algorithms to be subpacing, we also need to prevent the algorithms from underspending the baseline by a lot, followed by spending a lot in a short interval, which violates subpacing but does not show up as cumulative overspending. This is achieved by adjusting the baseline so it lowers if the algorithms underspend by too much.

The subpacing algorithm for the ORA setting is shown in Algorithm 4 and the algorithm for the OLRC setting is shown in Algorithm 5.

Write

$$
D _ { \alpha , \infty } : = \operatorname* { m a x } _ { i \in [ m ] } \alpha _ { i } D _ { i } , \qquad \alpha _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i \in [ m ] } \alpha _ { i } .
$$

The core variables $Y _ { t } ^ { i }$ and $E _ { t } ^ { i }$ from Section 4 retain their original definitions. For this extension, we introduce a clipped cumulative deviation $Y _ { t } ^ { i , \mathrm { s p } }$ defined as

$$
Y _ { 0 } ^ { i , \mathrm { s p } } : = 0 \qquad Y _ { t } ^ { i , \mathrm { s p } } = \operatorname* { m a x } \{ - W _ { i } , Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i } \}
$$

for some $W _ { i } > 0$ chosen later and define

$$
E _ { t } ^ { i , \mathrm { s p } } ( w ) : = Y _ { t } ^ { i , \mathrm { s p } } - \sum _ { f } w _ { f } S _ { t } ^ { f , i } .
$$

Analogously, define

$$
\Phi _ { t } ^ { \mathrm { s p } } ( w , v ) : = \sum _ { f } w _ { f } R _ { t } ^ { f } + \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } E _ { t } ^ { i , \mathrm { s p } } ( w ) - \rho _ { x } \Psi _ { x } ( w ) - \rho _ { v } \Psi _ { v } ( v ) , \qquad M _ { t } ^ { \mathrm { s p } } : = \operatorname* { m a x } _ { w , v } \Phi _ { t } ^ { \mathrm { s p } } ( w , v ) ,\tag{25}
$$

where, as in Section 4, all maxima over $( w , v )$ are over $\Delta ( \mathcal { F } ) \times \Delta _ { m + 1 }$ . For the pre-decision controller, define

$$
F _ { t } ^ { \mathrm { r a w } } ( x , w , v ) : = \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) + r _ { t } ( w ) - r _ { t } ( x ) + \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } \big ( c _ { t } ^ { i } ( x ) - c _ { t } ^ { i } ( w ) \big ) .\tag{26}
$$

For common parameters $\rho _ { x } , \rho _ { v } > 0$ , set

$$
W _ { i } : = 1 + D _ { i } + \frac { 2 + \rho _ { v } \log ( 4 m T ) } { \alpha _ { i } } , \qquad i \in [ m ] .\tag{27}
$$

We call round t safe if $Y _ { t - 1 } ^ { i , \mathrm { s p } } \leq W _ { i } - 1$ for every $i \in [ m ]$ , and unsafe otherwise. Before the hard stop, the learner plays the controller’s recommendation $p _ { t } = \pi _ { t } ( x _ { t } )$ on a safe round and the null action $p _ { t } = \perp$ on an

```latex
Algorithm 4: Subpacing algorithm extension with pre-decision feedback.
Initialize $R _ { 0 } ^ { f } = 0 , S _ { 0 } ^ { f , i } = 0 .$ , and $Y _ { 0 } ^ { i , \mathrm { s p } } = 0$ for every $f , i ;$
for $t = 1 , \ldots , T$ do
Receive the expert advice $q _ { t } ^ { f }$ and observe $( r _ { t } , c _ { t } ^ { 1 } , \hdots , c _ { t } ^ { m } ) ;$
if the hard stop is active, or $C _ { t - 1 } ^ { i } + 1 > B _ { i }$ for some i then
Activate the hard stop and set $p _ { t } = \perp ;$
else
if $Y _ { t - 1 } ^ { i , \mathrm { s p } } \leq W _ { i } - 1$ for every $i \in [ m ]$ then
Choose $x _ { t } \in$ arg min $\begin{array} { r } { \mathbf { \Phi } _ { x \in \Delta ( \mathcal { F } ) } ^ { * } \mathrm { m a x } _ { w , v } F _ { t } ^ { \mathrm { r a w } } ( x , w , v ) } \end{array}$
Set $p _ { t } = \pi _ { t } ( x _ { t } )$ ;
else
Set $p _ { t } = \bot ;$
end
end
Update, for every $f , i ,$
$R _ { t } ^ { f } = R _ { t - 1 } ^ { f } + r _ { t } ^ { f } - r _ { t } ( p _ { t } ) ,$ $S _ { t } ^ { f , i } = S _ { t - 1 } ^ { f , i } + c _ { t } ^ { f , i } - b _ { t } ^ { i } ,$
$Y _ { t } ^ { i , \mathrm { s p } } = \operatorname* { m a x } \{ - W _ { i } , Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i } \}$
end
```

unsafe round. Once the hard stop takes effect, the learner plays $p _ { t } = \perp$ forever. $Y _ { t } ^ { i , \mathrm { s p } }$ is always updated using the actual fractional cost $c _ { t } ^ { i } ( p _ { t } )$

Lemma 5.4. Let $p \in \Delta _ { d }$ have full support, let $s \in \mathbb { R } ^ { d }$ , and let $z > 0$ . Then

$$
\operatorname* { s u p } _ { w \in \Delta _ { d } } \{ \langle w , s \rangle - z D _ { \mathrm { K L } } ( w \| p ) \} = z \log \left( \sum _ { k = 1 } ^ { d } p _ { k } e ^ { s _ { k } / z } \right) .\tag{28}
$$

The unique maximizer is

$$
w _ { k } ^ { * } = \frac { p _ { k } e ^ { s _ { k } / z } } { \sum _ { j = 1 } ^ { d } p _ { j } e ^ { s _ { j } / z } } .\tag{29}
$$

Proof. Let $\begin{array} { r } { Z : = \sum _ { k = 1 } ^ { d } p _ { k } e ^ { s _ { k } / z } } \end{array}$ and $p _ { k } ^ { * } : = p _ { k } e ^ { s _ { k } / z } / Z$ . For every $w \in \Delta _ { d }$

$$
D _ { \mathrm { K L } } ( w \| p ^ { * } ) - D _ { \mathrm { K L } } ( w \| p ) = \sum _ { k = 1 } ^ { d } w \log \frac { p _ { k } } { p _ { k } ^ { * } } = - \frac { 1 } { z } \langle w , s \rangle + \log Z .
$$

Therefore

$$
\begin{array} { r } { \langle w , s \rangle - z D _ { \mathrm { K L } } ( w \| p ) = z \log Z - z D _ { \mathrm { K L } } ( w \| p ^ { * } ) . } \end{array}
$$

The result follows from nonnegativity of KL divergence, with equality if and only if $w = p ^ { * }$

Theorem 5.5. Assume the experts are D<sub>i</sub>-paced in every resource i. Let $\rho _ { x } , \rho _ { v } > 0$ satisfy

$$
\rho _ { x } \rho _ { v } \geq 4 D _ { \alpha , \infty } ^ { 2 }\tag{30}
$$

and

$$
W _ { i } = 1 + D _ { i } + \frac { 2 + \rho _ { v } \log ( 4 m T ) } { \alpha _ { i } } , \qquad i \in [ m ] .\tag{31}
$$

With pre-decision feedback, run Algorithm 4; without pre-decision feedback, run Algorithm 5.

Algorithm 5: Subpacing algorithm extension without pre-decision feedback.   
Initialize $R _ { 0 } ^ { f } = 0 , S _ { 0 } ^ { f , i } = 0 .$ , and $Y _ { 0 } ^ { i , \mathrm { s p } } = 0$ for every $f , i ;$   
for $t = 1 , \ldots , T$ do   
Receive the expert advice $q _ { t } ^ { f } ;$   
if the hard stop is active, or $C _ { t - 1 } ^ { i } + 1 > B _ { i }$ for some i then   
Activate the hard stop and set $p _ { t } = \perp ;$   
else   
Choose $( x _ { t } , v _ { t } ) \in$ arg max ${ } _ { w , v } \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v )$   
if $Y _ { t - 1 } ^ { i , \mathrm { s p } } \leq W _ { i } - 1$ for every $i \in [ m ]$ then   
Set $p _ { t } = \pi _ { t } ( x _ { t } ) ;$   
else   
Set $p _ { t } = \bot ;$   
end   
end   
Observe $( r _ { t } , c _ { t } ^ { 1 } , \ldots , c _ { t } ^ { m } )$   
Update, for every $f , i ,$   
$R _ { t } ^ { f } = R _ { t - 1 } ^ { f } + r _ { t } ^ { f } - r _ { t } ( p _ { t } ) ,$ $S _ { t } ^ { f , i } = S _ { t - 1 } ^ { f , i } + c _ { t } ^ { f , i } - b _ { t } ^ { i } ,$   
$Y _ { t } ^ { i , \mathrm { s p } } = \operatorname* { m a x } \{ - W _ { i } , Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i } \}$   
end

Define $\begin{array} { r } { V _ { \operatorname* { m i x } } = \sum _ { i = 1 } ^ { m } V _ { i } . } \end{array}$ . In either regime, the actualfractional cost path satisfies every hard budget and is W<sub>i</sub>-subpacing in every resource i. Moreover, simultaneously for every expert f, Algorithm 4 satisfies

$$
\mathrm { R e g } _ { T } ^ { f } \le \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) + \operatorname* { m a x } _ { i } \eta _ { i } ( 1 + D _ { i } ) + V _ { \operatorname* { m i x } } + \frac { 1 + \alpha _ { \operatorname* { m a x } } } { 4 } ,\tag{32}
$$

and Algorithm 5 satisfies

$$
\mathrm { R e g } _ { T } ^ { f } \leq \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) + \frac { T ( 1 + \alpha _ { \operatorname* { m a x } } ) ^ { 2 } } { \rho _ { x } } + \operatorname* { m a x } _ { i } \eta _ { i } ( 1 + D _ { i } ) + V _ { \operatorname* { m i x } } + \frac { 1 + \alpha _ { \operatorname* { m a x } } } { 4 } .\tag{33}
$$

Optimizing $\rho _ { x } , \rho _ { v }$ for regret bounds gives

$$
\mathrm { R e g } _ { T } ^ { f } = \widetilde { O } _ { m } \Big ( D _ { \alpha , \infty } \sqrt { \log { F } } + \alpha _ { \mathrm { m a x } } + V _ { \mathrm { m i x } } \Big )
$$

and

$$
W _ { i } = \widetilde { O } _ { m , T } \bigg ( 1 + D _ { i } + \frac { 1 } { \alpha _ { i } } + \frac { D _ { \alpha , \infty } \sqrt { \log { F } } } { \alpha _ { i } } \bigg ) .
$$

with pre-decision feedback, and

$$
\mathrm { R e g } _ { T } ^ { f } = \widetilde { O } _ { m } \Big ( \big ( ( 1 + \alpha _ { \mathrm { m a x } } ) \sqrt { T } + D _ { \alpha , \infty } \big ) \sqrt { \log { F } } + V _ { \mathrm { m i x } } \Big )
$$

and

$$
W _ { i } = \widetilde { O } _ { m , T } \left( 1 + D _ { i } + \frac { 1 } { \alpha _ { i } } + \frac { D _ { \alpha , \infty } ^ { 2 } \sqrt { \log { F } } } { \alpha _ { i } { \left( D _ { \alpha , \infty } + { \left( 1 + \alpha _ { \operatorname* { m a x } } \right) } \sqrt { T } \right) } } \right) .
$$

without pre-decisionfeedback. The notations $\widetilde { O } _ { m } ( \cdot )$ and $\tilde { O } _ { T , m } ( \cdot )$ suppress polylogarithmic factors in their subscripts.

Proof. We first prove a subpacing witness, then bound the growth of the potential, and finally convert the prefix potential bound into full-horizon regret.

Subpacing witness. We claim that

$$
- W _ { i } \leq Y _ { t } ^ { i , \mathrm { s p } } \leq W _ { i } \qquad \mathrm { f o r } \mathrm { e v e r y } i , t .\tag{34}
$$

The lower bound follows directly from the definition. For the upper bound, we prove by induction. If round t is safe, then $Y _ { t - 1 } ^ { i , \mathrm { s p } } \leq W _ { i } - 1$ , and therefore

$$
Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i } \leq Y _ { t - 1 } ^ { i , \mathrm { s p } } + 1 \leq W _ { i } .
$$

If round t is unsafe, then $c _ { t } ^ { i } ( p _ { t } ) = 0$ , so

$$
Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i } = Y _ { t - 1 } ^ { i , \mathrm { s p } } - b _ { t } ^ { i } \leq Y _ { t - 1 } ^ { i , \mathrm { s p } } \leq W _ { i } ,
$$

where the last inequality is the induction hypothesis. Thus (34) holds.

For each $i , t ,$ define

$$
k _ { t } ^ { i } : = \left( - W _ { i } - \left( Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i } \right) \right) _ { + } , \qquad d _ { t } ^ { i } : = b _ { t } ^ { i } - k _ { t } ^ { i } .
$$

Intuitively, $k _ { t } ^ { i }$ is how much is clipped by the lower bound $\mathrm { o f } - W _ { i }$ during round $t ,$ and we adjust the baseline downward by this clipped amount. We claim that for every resource $i , ( d _ { t } ^ { i } ) _ { t }$ is a subpacing witness of the algorithm, i.e. the algorithm is $W _ { i }$ -pacing with respect to $d ^ { \bar { i } }$

Since $Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i } \geq - W _ { i } - b _ { t } ^ { i }$ , we have $0 \leq k _ { t } ^ { i } \leq b _ { t } ^ { i }$ and therefore $d _ { t } ^ { i } \in [ 0 , b _ { t } ^ { i } ]$ . This definition of $Y _ { t } ^ { i , \mathrm { s p } }$ implies

$$
Y _ { t } ^ { i , \mathrm { s p } } = Y _ { t - 1 } ^ { i , \mathrm { s p } } + c _ { t } ^ { i } ( p _ { t } ) - d _ { t } ^ { i } ,
$$

and hence

$$
Y _ { t } ^ { i , \mathrm { s p } } = \sum _ { s \leq t } \bigl ( c _ { s } ^ { i } ( p _ { s } ) - d _ { s } ^ { i } \bigr ) .\tag{35}
$$

Together with (34), this proves directly that the actual fractional resource-i cost path $( c _ { t } ^ { i } ( p _ { t } ) )$ <sub>t</sub> is W<sub>i</sub>-subpacing.   
The hard stop gives fractional budget feasibility.

Bounding potential difference due to clipping. On a round before the hard stop, let $\Phi _ { t } ^ { \operatorname { r a w } }$ denote the potential obtained after updating $R _ { t } ^ { f }$ and $S _ { t } ^ { \bar { f } , \bar { i } }$ and replacing $Y _ { t } ^ { i , \mathrm { s p } }$ by its unclipped value $Y _ { t } ^ { i , \mathrm { { r a w } } } : = Y _ { t - 1 } ^ { i , \mathrm { { s p } } } +$ $c _ { t } ^ { i } ( p _ { t } ) - b _ { t } ^ { i }$ . More formally, define

$$
\Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) : = \sum _ { f } w _ { f } R _ { t } ^ { f } + \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } E _ { t } ^ { i , \mathrm { r a w } } ( w ) - \rho _ { x } \Psi _ { x } ( w ) - \rho _ { v } \Psi _ { v } ( v ) ,
$$

where

$$
E _ { t } ^ { i , \mathrm { r a w } } ( w ) : = Y _ { t } ^ { i , \mathrm { r a w } } - \langle w , S _ { t } ^ { i } \rangle .
$$

Let

$$
M _ { t } ^ { \mathrm { r a w } } : = \operatorname* { m a x } _ { w , v } \Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) .
$$

We want to bound the potential increase caused by clipping in a round before the hard stop. Fix w and suppose $Y _ { t } ^ { i , \mathrm { s p } }$ was clipped from $Y _ { t } ^ { i , \mathrm { r a w } } \leq - W _ { i }$ to $Y _ { t } ^ { i , \mathrm { s p } } = - W _ { i }$ at the end of round t. Applying Lemma 5.4 to the v-maximization part of the potential function gives

$$
\operatorname* { s u p } _ { v \in \Delta _ { m + 1 } } \left. \left. v , \left( 0 , \alpha \odot E _ { t } ^ { \mathrm { s p } } \right) \right. - \rho _ { v } D _ { \mathrm { K L } } ( v \| U _ { m + 1 } ) \right. = \rho _ { v } \log \left( \frac { 1 } { m + 1 } + \sum _ { j = 1 } ^ { m } \frac { 1 } { m + 1 } e ^ { \alpha _ { j } E _ { t } ^ { j , \mathrm { s p } } / \rho _ { v } } \right) .
$$

where $\alpha \odot E _ { t } ^ { \mathrm { s p } }$ is the pointwise product of $( \alpha _ { i } ) _ { i = 1 } ^ { m }$ and $( E _ { t } ^ { i , \mathrm { s p } } ) _ { i = 1 } ^ { m }$ and we add a 0 coordinate in front. To isolate the effect of each coordinate of $\alpha \odot { \dot { E } } _ { t } ^ { \mathrm { s p } }$ , we define

$$
V ( s ) = \operatorname* { s u p } _ { v \in \Delta _ { m + 1 } } \{ \langle v , ( 0 , s ) \rangle - \rho _ { v } D _ { \mathrm { K L } } ( v \| U _ { m + 1 } ) \} = \rho _ { v } \log \left( \frac { 1 } { m + 1 } + \sum _ { j = 1 } ^ { m } \frac { 1 } { m + 1 } e ^ { s _ { j } / \rho _ { v } } \right)
$$

for $s \in \mathbb { R } ^ { m }$ . Then for the clipped coordinate $i \in [ m ]$ , we have

$$
\frac { \partial V ( s ) } { \partial s _ { i } } = \frac { e ^ { s _ { i } / \rho _ { v } } } { 1 + \sum _ { j = 1 } ^ { m } e ^ { s _ { j } / \rho _ { v } } } \leq e ^ { s _ { i } / \rho _ { v } } .
$$

Since $Y _ { t } ^ { i , \mathrm { r a w } } \leq - W _ { i }$ , for any $y \in [ Y _ { t } ^ { i , \operatorname { r a w } } , - W _ { i } ]$ , we have

$$
\alpha _ { i } \big ( y - \langle w , S _ { t } ^ { i } \rangle \big ) \leq \alpha _ { i } \big ( - W _ { i } - \langle w , S _ { t } ^ { i } \rangle \big ) \leq \alpha _ { i } ( - W _ { i } + D _ { i } ) \leq - \rho _ { v } \log ( 4 m T ) .
$$

Thus

$$
\frac { \partial V ( s ) } { \partial s _ { i } } \leq \frac { 1 } { 4 m T }
$$

for $s _ { i } = \alpha _ { i } \big ( y - \langle w , S _ { t } ^ { i } \rangle \big )$ and $y \in [ Y _ { t } ^ { i , \operatorname { r a w } } , - W _ { i } ]$ . This implies clipping the coordinate i affects $V ( s )$ by at most

$$
\frac { 1 } { 4 m T } \alpha _ { i } k _ { t } ^ { i } \leq \frac { 1 } { 4 m T } \alpha _ { i } b _ { t } ^ { i } \leq \frac { \alpha _ { i } } { 4 m T } .
$$

We apply the same argument for each clipped coordinate one by one, and the resulting change in $V ( s )$ is bounded by

$$
V ( \alpha \odot ( Y _ { t } ^ { \mathrm { s p } } - \langle w , S _ { t } \rangle ) - V ( \alpha \odot ( Y _ { t } ^ { \mathrm { r a w } } - \langle w , S _ { t } \rangle ) \le \frac { \alpha _ { \operatorname* { m a x } } } { 4 T } .
$$

on each round. Since this is true for any w and

$$
{ M } _ { t } ^ { \mathrm { s p } } = \operatorname* { m a x } _ { w } \sum _ { f } w _ { f } R _ { t } ^ { f } - \rho _ { x } \Psi _ { x } ( w ) + V ( \alpha \odot ( Y _ { t } ^ { \mathrm { s p } } - \langle w , S _ { t } \rangle )
$$

$$
{ M } _ { t } ^ { \mathrm { { r a w } } } = \operatorname* { m a x } _ { w } \sum _ { f } { w } _ { f } { R } _ { t } ^ { f } - \rho _ { x } \Psi _ { x } ( w ) + { V } ( \alpha \odot ( Y _ { t } ^ { \mathrm { { r a w } } } - \langle w , S _ { t } \rangle ) ,
$$

we conclude

$$
M _ { t } ^ { \mathrm { s p } } - M _ { t } ^ { \mathrm { r a w } } \leq \frac { \alpha _ { \mathrm { m a x } } } { 4 T }\tag{36}
$$

Potential growth without pre-decision feedback. Suppose first that round t is safe and the hard stop is inactive. Then $p _ { t } = \pi _ { t } ( x _ { t } )$ , so $r _ { t } ( p _ { t } ) = r _ { t } ( x _ { t } )$ and $c _ { t } ^ { i } ( p _ { t } ) { \bar { = } } c _ { t } ^ { i } ( x _ { t } )$ . Before clipping in round $t ,$ we have

$$
\Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) - \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) = \left. r _ { t } - \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } c _ { t } ^ { i } , w - x _ { t } \right. \leq ( 1 + \alpha _ { \operatorname* { m a x } } ) \| w - x _ { t } \| _ { 1 } .
$$

Since changing from $Y _ { t } ^ { i }$ to $Y _ { t } ^ { i , \mathrm { s p } }$ in the potential function only changes the term $\alpha _ { i } v _ { i } Y _ { t } ^ { i , \mathrm { s p } }$ , which is affine in $v ,$ Lemma 4.1 still holds for this modified potential function with $\lambda _ { i } = \alpha _ { i }$ , provided that the regularization parameters satisfy (30). In particular, since $( x _ { t } , v _ { t } )$ maximizes $\Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v )$ , then for every feasible $( w , v )$

$$
\Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) \leq M _ { t - 1 } ^ { \mathrm { s p } } - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| x _ { t } ) - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v _ { t } ) .\tag{37}
$$

Thus

$$
\begin{array} { r l } & { \Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) - M _ { t - 1 } ^ { \mathrm { s p } } = \Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) - \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) + \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) - M _ { t - 1 } ^ { \mathrm { s p } } } \\ & { \qquad \leq ( 1 + \alpha _ { \mathrm { m a x } } ) \| w - x _ { t } \| _ { 1 } - \frac { \rho _ { x } } { 4 } \| w - x _ { t } \| _ { 1 } ^ { 2 } . } \end{array}
$$

where the last line follows from Pinsker’s Inequality and non-negativity of KL divergence.

Optimizing over $\| w - x _ { t } \| _ { 1 } \ge 0$ yields

$$
M _ { t } ^ { \mathrm { r a w } } - M _ { t - 1 } ^ { \mathrm { s p } } \leq \frac { ( 1 + \alpha _ { \mathrm { m a x } } ) ^ { 2 } } { \rho _ { x } } .\tag{38}
$$

Now suppose round t is unsafe. Choose $j$ with $Y _ { t - 1 } ^ { j , \mathrm { s p } } > W _ { j } - 1$ . Since $\langle x _ { t } , S _ { t - 1 } ^ { j } \rangle \leq D _ { j }$

$$
\alpha _ { j } E _ { t - 1 } ^ { j , \mathrm { s p } } ( x _ { t } ) > \alpha _ { j } ( W _ { j } - 1 - D _ { j } ) = 2 + \rho _ { v } \log ( 4 m T ) .\tag{39}
$$

Because $( x _ { t } , v _ { t } )$ maximizes $\Phi _ { t - 1 } ^ { \mathrm { s p } }$ , fixing $x _ { t }$ shows that $v _ { t }$ maximizes the v-dependent terms. Lemma 5.4 therefore gives

$$
v _ { t , 0 } = \frac { 1 } { 1 + \sum _ { i = 1 } ^ { m } \exp ( \alpha _ { i } E _ { t - 1 } ^ { i , \mathrm { s p } } ( x _ { t } ) / \rho _ { v } ) } \leq \frac { e ^ { - 2 / \rho _ { v } } } { 4 m T } .\tag{40}
$$

On an unsafe round before the hard stop, $p _ { t } = \perp$ , so $r _ { t } ( p _ { t } ) = 0$ and $c _ { t } ^ { i } ( p _ { t } ) = 0$ for every i. In this case,

$$
\Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) - \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) = \left. r _ { t } - \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } c _ { t } ^ { i } , w \right. .\tag{41}
$$

Let $\epsilon _ { t } ^ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i } \epsilon _ { t } ^ { i }$ . For every $( w , v )$ , Assumption 2.1 implies $r _ { t } ( w ) \leq \alpha _ { i } c _ { t } ^ { i } ( w ) + \epsilon _ { t } ^ { \mathrm { m a x } }$ . Then multiplying this by $v _ { i }$ and summing over $i \in [ m ]$ gives

$$
( 1 - v _ { 0 } ) r _ { t } ( w ) \leq \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } c _ { t } ^ { i } ( w ) + ( 1 - v _ { 0 } ) \epsilon _ { t } ^ { \operatorname* { m a x } } .
$$

Since $r _ { t } ( w ) \leq 1$ , we have

$$
\begin{array} { r } { r _ { t } ( w ) - \displaystyle \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } c _ { t } ^ { i } ( w ) \leq v _ { 0 } + \epsilon _ { t } ^ { \operatorname* { m a x } } } \\ { \implies \Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) - \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) \leq v _ { 0 } + \epsilon _ { t } ^ { \operatorname* { m a x } } . } \end{array}\tag{42}
$$

(43)

Combining (37) and (43) gives

$$
\begin{array} { r l } & { \Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) - M _ { t - 1 } ^ { \mathrm { s p } } = \Phi _ { t } ^ { \mathrm { r a w } } ( w , v ) - \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) + \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) - M _ { t - 1 } ^ { \mathrm { s p } } } \\ & { \qquad \le v _ { 0 } + \epsilon _ { t } ^ { \mathrm { m a x } } - \cfrac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| x _ { t } ) - \cfrac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v _ { t } ) } \end{array}
$$

Optimizing over $( w , v )$ gives

$$
\begin{array} { r l r } & { } & { M _ { t } ^ { \mathrm { r a w } } - M _ { t - 1 } ^ { \mathrm { s p } } \leq \epsilon _ { t } ^ { \mathrm { m a x } } + \underset { w } { \operatorname* { s u p } } \left\{ - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| x _ { t } ) \right\} } \\ & { } & { + \underset { v } { \operatorname* { s u p } } \left\{ v _ { 0 } - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v _ { t } ) \right\} . } \end{array}
$$

The first supremum is zero. Lemma 4.1 guarantees that $v _ { t }$ has full support, so Lemma 5.4 applies to the second supremum with $p = v _ { t } , s = e _ { 0 }$ , and $z = \rho _ { v } / 2$ . This gives

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { v } \left\{ v _ { 0 } - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v _ { t } ) \right\} = \frac { \rho _ { v } } { 2 } \log \left( 1 + v _ { t , 0 } ( e ^ { 2 / \rho _ { v } } - 1 ) \right) } & { } \\ { \displaystyle \leq \frac { \rho _ { v } } { 8 m T } ( 1 - e ^ { - 2 / \rho _ { v } } ) } & { } \\ { \displaystyle \leq \frac { 1 } { 4 m T } . } \end{array}\tag{by (40)}
$$

$$
( 1 - e ^ { - x } \leq x )
$$

Thus every unsafe round satisfies

$$
M _ { t } ^ { \mathrm { r a w } } - M _ { t - 1 } ^ { \mathrm { s p } } \leq \epsilon _ { t } ^ { \mathrm { m a x } } + \frac { 1 } { 4 m T } .\tag{44}
$$

Since $M _ { 0 } ^ { \mathrm { s p } } = 0$ , summing (38), (44), and (36) through any prefix $\tau$ before the hard stop gives

$$
M _ { \tau } ^ { \mathrm { s p } } \leq \frac { T ( 1 + \alpha _ { \mathrm { m a x } } ) ^ { 2 } } { \rho _ { x } } + \sum _ { t < \tau } \epsilon _ { t } ^ { \mathrm { m a x } } + \frac { 1 + \alpha _ { \mathrm { m a x } } } { 4 } .\tag{45}
$$

Potential growth with pre-decision feedback. Recall that $F _ { t } ^ { \mathrm { r a w } } ( x , w , v ) = \Phi _ { t - 1 } ^ { \mathrm { s p } } ( w , v ) + r _ { t } ( w ) - r _ { t } ( x ) +$ $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \alpha _ { i } v _ { i } \big ( c _ { t } ^ { i } ( x ) - c _ { t } ^ { i } ( w ) \big ) } \end{array}$ . On a safe round, $F _ { t } ^ { \mathrm { r a w } }$ is affine in x and jointly concave in $( w , v )$ for fixed x by Lemma $4 . 1$ , since changing from $Y _ { t } ^ { i }$ to $Y _ { t } ^ { i , \mathrm { s p } }$ in the potential function only changes the term $\alpha _ { i } v _ { i } Y _ { t } ^ { i , \mathrm { s p } }$ which is affine in v, and the regularization parameters satisfy (30). Sion’s minimax theorem gives

$$
\begin{array} { r l } & { M _ { t } ^ { \mathrm { r a w } } = \underset { x } { \mathrm { m i n } } \underset { w , v } { \mathrm { m a x } } F _ { t } ^ { \mathrm { r a w } } ( x , w , v ) = \underset { w , v } { \mathrm { m a x } } \underset { x } { \mathrm { m i n } } F _ { t } ^ { \mathrm { r a w } } ( x , w , v ) } \\ & { \qquad \le \underset { w , v } { \mathrm { m a x } } F _ { t } ^ { \mathrm { r a w } } ( w , w , v ) = M _ { t - 1 } ^ { \mathrm { s p } } . } \end{array}
$$

where the first equality follows from the choice of $x _ { t }$ in Algorithm 4. Thus a safe round causes no raw increase.

Now suppose round t is unsafe and choose $j$ with $Y _ { t - 1 } ^ { j , \mathrm { s p } } > W _ { j } - 1$ . For every w,

$$
\alpha _ { j } E _ { t - 1 } ^ { j , \mathrm { s p } } ( w ) > \alpha _ { j } ( W _ { j } - 1 - D _ { j } ) = 2 + \rho _ { v } \log ( 4 m T ) .\tag{46}
$$

Since the action chosen is the same in both regimes during an unsafe round, we can reuse the argument from the no-pre-decision feedback regime. Let $( w _ { t } ^ { \star } , v _ { t } ^ { \star } )$ maximize $\Phi _ { t - 1 } ^ { \mathrm { s p } }$ . In the no-pre-decision regime this is the pair chosen by the controller, while in the pre-decision regime it is used only for the analysis. Then we get

$$
\begin{array} { r l } & { M _ { t } ^ { \mathrm { r a w } } - M _ { t - 1 } ^ { \mathrm { s p } } \leq \epsilon _ { t } ^ { \mathrm { m a x } } + \underset { w } { \operatorname* { s u p } } \left\{ - \frac { \rho _ { x } } { 2 } D _ { \mathrm { K L } } ( w \| w _ { t } ^ { * } ) \right\} + \underset { v } { \operatorname* { s u p } } \left\{ v _ { 0 } - \frac { \rho _ { v } } { 2 } D _ { \mathrm { K L } } ( v \| v _ { t } ^ { * } ) \right\} } \\ & { \quad \quad \quad \leq \epsilon _ { t } ^ { \mathrm { m a x } } + \frac { 1 } { 4 m T } . } \end{array}\tag{47}
$$

following the same argument as the other regime.

Summing (47) and (36) through any prefix $\tau$ before the hard stop gives

$$
M _ { \tau } ^ { \mathrm { s p } } \leq \sum _ { t \leq \tau } \epsilon _ { t } ^ { \mathrm { m a x } } + \frac { 1 + \alpha _ { \mathrm { m a x } } } { 4 } .\tag{48}
$$

From the prefix potential to regret. This part of the proof basically repeats the logic in Section 4.4 by noting that Lemma 4.4 still holds if we have $\bar { E _ { \tau } ^ { j , \mathrm { s p } } } ( e _ { f } ) \geq \bar { C } _ { \tau } ^ { j } - A _ { \tau } ^ { f , j }$ (instead of having $\bar { E } _ { \tau } ^ { j } ( e _ { f } ) = C _ { \tau } ^ { j } - A _ { \tau } ^ { f , j }$ as in the core algorithm) and setting $h _ { j } = 1$ . For completion, we expand out the proofs.

If the hard stop never occurs, then by definition $\mathrm { R e g } _ { T } ^ { f } = R _ { T } ^ { f }$ . Evaluating (25) at $( e _ { f } , e _ { 0 } )$ gives

$$
\mathrm { R e g } _ { T } ^ { f } \leq M _ { T } ^ { \mathrm { s p } } + \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) .
$$

Now suppose the hard stop first takes effect on round $\tau + 1$ , and let resource $j$ trigger it. Thus $\tau$ is the last round before the hard stop, and by definition

$$
\mathrm { R e g } _ { \tau } ^ { f } = R _ { \tau } ^ { f } , \qquad C _ { \tau } ^ { j } = \sum _ { t \le \tau } c _ { t } ^ { j } ( p _ { t } ) .
$$

Recall

$$
A _ { t } ^ { f , j } : = \sum _ { s \leq t } c _ { s } ^ { f , j } , \qquad B _ { t } ^ { j } : = \sum _ { s \leq t } b _ { s } ^ { j } .
$$

By (35),

$$
\begin{array} { r l } & { E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } ) = Y _ { \tau } ^ { j , \mathrm { s p } } - S _ { \tau } ^ { f , j } } \\ & { \qquad = C _ { \tau } ^ { j } - \displaystyle \sum _ { t \leq \tau } d _ { t } ^ { j } - ( A _ { \tau } ^ { f , j } - B _ { \tau } ^ { j } ) } \\ & { \qquad \geq C _ { \tau } ^ { j } - A _ { \tau } ^ { f , j } . } \end{array}
$$

$$
( d _ { t } ^ { j } \leq b _ { t } ^ { j } )
$$

Since resource j triggers the stop on round $\tau + 1$ , the stopping rule gives $B _ { j } - C _ { \tau } ^ { j } < 1$ , while pacing gives $A _ { T } ^ { f , j } \leq B _ { j } + D _ { j }$ . Therefore

$$
\begin{array} { r l r } {  { \sum _ { t > \tau } c _ { t } ^ { f , j } = A _ { T } ^ { f , j } - A _ { \tau } ^ { f , j } } } \\ & { } & { \leq ( B _ { j } - C _ { \tau } ^ { j } ) + ( C _ { \tau } ^ { j } - A _ { \tau } ^ { f , j } ) + D _ { j } } \\ & { } & { < 1 + E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } ) + D _ { j } . } \end{array}\tag{49}
$$

By Assumption 2.1,

$$
\sum _ { t > \tau } r _ { t } ^ { f } \leq \alpha _ { j } \big ( E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } ) + 1 + D _ { j } \big ) + \sum _ { t > \tau } \epsilon _ { t } ^ { j } .\tag{50}
$$

If $\beta _ { j } < \infty$ , then

$$
\begin{array} { r l } {  { \sum _ { t > \tau } b _ { t } ^ { j } = B _ { j } - B _ { \tau } ^ { j } } } \\ & { = ( B _ { j } - C _ { \tau } ^ { j } ) + ( C _ { \tau } ^ { j } - A _ { \tau } ^ { f , j } ) + ( A _ { \tau } ^ { f , j } - B _ { \tau } ^ { j } ) } \\ & { < 1 + E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } ) + D _ { j } . } \end{array}
$$

Since $\begin{array} { r } { \sum _ { t > \tau } r _ { t } ^ { f } \le T - \tau \le \beta _ { j } \sum _ { t > \tau } b _ { t } ^ { j } } \end{array}$ , we also have

$$
\sum _ { t > \tau } r _ { t } ^ { f } \le \beta _ { j } \bigl ( E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } ) + 1 + D _ { j } \bigr ) .\tag{51}
$$

Combining (50) and (51),

$$
\sum _ { t > \tau } r _ { t } ^ { f } \leq \eta _ { j } \big ( E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } ) + 1 + D _ { j } \big ) + \mathbf { 1 } \{ \alpha _ { j } < \beta _ { j } \} \sum _ { t > \tau } \epsilon _ { t } ^ { j } .\tag{52}
$$

Set $s : = \eta _ { j } / \alpha _ { j } \leq 1$ and evaluate the prefix potential at

$$
\begin{array} { r } { ( w , v ) = ( e _ { f } , ( 1 - s ) e _ { 0 } + s e _ { j } ) . } \end{array}
$$

Using $\alpha _ { j } s = \eta _ { j } , \Psi _ { x } ( e _ { f } ) =$ log F, and $\Psi _ { v } ( ( 1 - s ) e _ { 0 } + s e _ { j } ) \leq \log ( m + 1 )$ gives

$$
\mathrm { R e g } _ { \tau } ^ { f } \le M _ { \tau } ^ { \mathrm { s p } } + \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) - \eta _ { j } E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } ) .\tag{53}
$$

After the hard stop the learner obtains zero reward, so

$$
\mathrm { R e g } _ { T } ^ { f } = \mathrm { R e g } _ { \tau } ^ { f } + \sum _ { t > \tau } r _ { t } ^ { f } .
$$

Adding (52) and (53) cancels the $E _ { \tau } ^ { j , \mathrm { s p } } ( e _ { f } )$ terms. Moreover,

$$
\sum _ { t \leq \tau } \epsilon _ { t } ^ { \operatorname* { m a x } } + \mathbf { 1 } \{ \alpha _ { j } < \beta _ { j } \} \sum _ { t > \tau } \epsilon _ { t } ^ { j } \leq \sum _ { t = 1 } ^ { T } \epsilon _ { t } ^ { \operatorname* { m a x } } \leq V _ { \operatorname* { m i x } } .
$$

Substituting (48) or (45) yields (32) and (33).

For the optimized rates, take $\rho _ { v } = 4 D _ { \alpha , \infty } ^ { 2 } / \rho _ { x }$ . With pre-decision feedback, minimizing

$$
\rho _ { x } \log { F } + \frac { 4 D _ { \alpha , \infty } ^ { 2 } \log ( m + 1 ) } { \rho _ { x } }
$$

gives the stated bound. Without pre-decision feedback, minimize

$$
\rho _ { x } \log { F } + \frac { 4 D _ { \alpha , \infty } ^ { 2 } \log ( m + 1 ) + T ( 1 + \alpha _ { \operatorname* { m a x } } ) ^ { 2 } } { \rho _ { x } } ,
$$

and use ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } } .$

Composing Extension 1 and 2. We briefly explain why this extension composes with the online-raising reduction of Extension 1. Let $\bar { c } _ { t } ^ { f , i }$ denote the raised expert costs, and define

$$
\bar { S } _ { t } ^ { f , i } : = \sum _ { s \leq t } ( \bar { c } _ { s } ^ { f , i } - b _ { s } ^ { i } ) , \qquad \bar { c } _ { t } ^ { i } ( w ) : = \sum _ { f } w _ { f } \bar { c } _ { t } ^ { f , i } .
$$

Apply the same controller to the raised instance: replace every occurrence of $c _ { t } ^ { i } ( w )$ in the analytical controller by $\hat { c } _ { t } ^ { i } ( w )$ and every $S _ { t } ^ { f , i }$ in the comparison state by $\bar { S } _ { t } ^ { f , i }$ . On a safe round before the hard stop, we update $Y _ { t } ^ { i , \mathrm { s p } }$ using cost $\hat { c } _ { t } ^ { i } ( x _ { t } ) ;$ ; on an unsafe round or after the hard stop, it is updated using zero cost. The reward coordinate remains the true reward $r _ { t } ( p _ { t } )$ , since raising changes only costs.

For budget accounting, define the raised cumulative cost

$$
\bar { C } _ { t - 1 } ^ { i } : = \sum _ { \tiny \begin{array} { c } { { s < t : s \mathrm { ~ i s ~ s a f e ~ a n d } } } \\ { { s \mathrm { ~ p r e c e d e s ~ t h e ~ h a r d ~ s t o p ~ } } } \end{array} } \bar { c } _ { s } ^ { i } ( x _ { s } ) ,
$$

and trigger the hard stop whenever $\bar { C } _ { t - 1 } ^ { i } + 1 > B _ { i }$ for some $i .$ The physical distribution is still $p _ { t } = \pi _ { t } ( x _ { t } )$ on a safe round before the hard stop and $p _ { t } = \perp$ otherwise. Hence, on every round,

$$
0 \leq c _ { t } ^ { i } ( p _ { t } ) \leq { \left\{ \begin{array} { l l } { { \bar { c } } _ { t } ^ { i } ( x _ { t } ) , } & { { \mathrm { i f ~ } } t { \mathrm { ~ i s ~ s a f e ~ a n d ~ p r e c e d e s ~ t h e ~ h a r d ~ s t o p , } } } \\ { 0 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

Since the raised experts are $D _ { i } { \mathrm { - p a c e d } } .$ , lie in [0, 1], and satisfy the same soft reward-to-cost condition, the preceding potential and terminal arguments apply verbatim to the raised accounting path. They show that this raised path is $W _ { i }$ -subpacing and respects the raised hard budget. The true fractional cost path is pointwise dominated by the raised accounting path. Hence true budget feasibility follows from the raised hard budget, and Lemma 5.1 transfers the $W _ { i }$ -subpacing guarantee to the true costs.

## 5.3 Extension 3: sampled play with pathwise budgets and high-probability guarantees

In this section, we show a straightforward extension to the sampling setting under full-information, where the algorithm gets realized rewards $r _ { t } ( a )$ from a draw $a \sim p _ { t }$ and incurs realized cost $c _ { t } ^ { i } ( a )$ . The algorithm hard stops when the realized costs exceed $B _ { i } - 1$ for some resource i. We use martingale concentration to show a high probability regret bound. If we additionally compose with Extension 2, then we also show a high probability subpacing bound.

Sampling the core fractional algorithm. Let $p _ { t } \in \Delta ( A )$ denote the round-t distribution produced by the core fractional algorithm, where recall that hard stop was determined by the algorithm’s fractional/expected spending. The sampled learner maintains an additional realized hard stop. Let $\widehat { c } _ { t } ^ { i } : = c _ { t } ^ { i } ( a _ { t } )$ and

$$
\widehat { C } _ { t - 1 } ^ { i } : = \sum _ { s < t } \widehat { c } _ { s } ^ { i }
$$

be its realized cumulative cost in resource i. Before round t, if the realized hard stop is already active, or if $\widehat { C } _ { t - 1 } ^ { i } + 1 > B _ { i }$ for some $i ,$ the learner activates the realized hard stop and sets $a _ { t } : = \perp$ . Otherwise it samples $a _ { t } \sim p _ { t }$ . The underlying fractional controller is still run as in Section 4. The realized hard stop only truncates the sampled play and does not alter $p _ { t }$ or the fractional controller’s state.

Let $\chi _ { t } \in \{ 0 , 1 \}$ indicate that round t has not entered the realized hard stop yet, so that $\chi _ { t } = 1$ exactly when $a _ { t }$ is sampled from $p _ { t }$ . Define

$$
\begin{array} { r } { \widehat { r } _ { t } : = r _ { t } ( a _ { t } ) , \qquad \widehat { c } _ { t } ^ { i } : = c _ { t } ^ { i } ( a _ { t } ) , \qquad \mu _ { t } ^ { r } : = \chi _ { t } r _ { t } ( p _ { t } ) , \qquad \mu _ { t } ^ { i } : = \chi _ { t } c _ { t } ^ { i } ( p _ { t } ) . } \end{array}
$$

Thus $\mu _ { t } ^ { r }$ and $\mu _ { t } ^ { i }$ are the expected reward and resource-i cost of the sampled learner conditioned on all the information before its private round-t draw (with both equal to zero when the realized hard stop is active). For $V , L \geq 0$ , define

$$
\mathrm { F r } ( V , L ) : = \frac { L } { 3 } + \sqrt { 2 V L + \frac { L ^ { 2 } } { 9 } } ,
$$

and set

$$
\boldsymbol { Q } _ { r } : = \operatorname* { m i n } \left\{ \boldsymbol { T } , \operatorname* { m i n } _ { i \in [ m ] } ( \alpha _ { i } B _ { i } + V _ { i } ) \right\} , \qquad \boldsymbol { Q } _ { i } : = \operatorname* { m i n } \{ B _ { i } , \boldsymbol { T } \} ,
$$

$$
\Gamma _ { r } ( \delta ) : = \mathrm { F r } \left( Q _ { r } , \log \frac { 2 } { \delta } \right) , \qquad \Gamma _ { i } ( \delta ) : = \mathrm { F r } \left( Q _ { i } , \log \frac { 4 m } { \delta } \right) .
$$

In particular,

$$
\Gamma _ { r } ( \delta ) = O \left( \sqrt { Q _ { r } \log \frac { 2 } { \delta } } + \log \frac { 2 } { \delta } \right) , \qquad \Gamma _ { i } ( \delta ) = O \left( \sqrt { Q _ { i } \log \frac { 4 m } { \delta } } + \log \frac { 4 m } { \delta } \right) .
$$

We use the following Freedman inequality [Freedman, 1975]: if $( Z _ { t } ) _ { t \leq T }$ is a martingale-difference sequence with $| Z _ { t } | \le 1$ and

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } [ Z _ { t } ^ { 2 } \mid { \mathcal { H } } _ { t } ] \leq V
$$

almost surely, then

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { s \leq T } \sum _ { t \leq s } Z _ { t } > \operatorname { F r } ( V , L ) \right) \leq e ^ { - L } .
$$

The same statement applies to the lower tail by replacing $Z _ { t }$ with $- Z _ { t }$

Lemma 5.6. For every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ simultaneouslyfor every prefix $s \leq T$ and every resource $i \in [ m ]$

$$
\sum _ { t \leq s } ( \mu _ { t } ^ { r } - \widehat { r } _ { t } ) \leq \Gamma _ { r } ( \delta ) , \qquad \left| \sum _ { t \leq s } ( \widehat { c } _ { t } ^ { i } - \mu _ { t } ^ { i } ) \right| \leq \Gamma _ { i } ( \delta ) .\tag{54}
$$

Proof. Let $\mathcal { H } _ { t }$ contain the history, the fractional proposal $p _ { t } .$ , the indicator $\chi _ { t }$ , and the complete round-t reward and cost vectors immediately before the private draw. In both feedback regimes, the adversary has fixed the round-t reward and cost vectors before this draw. Hence

$$
\mathbb { E } [ \widehat { r } _ { t } \mid \mathcal { H } _ { t } ] = \mu _ { t } ^ { r } , \qquad \mathbb { E } [ \widehat { c } _ { t } ^ { i } \mid \mathcal { H } _ { t } ] = \mu _ { t } ^ { i } .
$$

Therefore

$$
Z _ { t } ^ { r } : = \widehat { r } _ { t } - \mu _ { t } ^ { r } , \qquad Z _ { t } ^ { i } : = \widehat { c } _ { t } ^ { i } - \mu _ { t } ^ { i }
$$

are martingale differences with absolute value at most one. Since rewards and costs lie in [0, 1], we have

$$
\mathbb { E } [ ( Z _ { t } ^ { r } ) ^ { 2 } \mid { \mathcal { H } } _ { t } ] = { \mathrm { V a r } } ( \widehat { r } _ { t } \mid { \mathcal { H } } _ { t } ) \leq E [ ( \widehat { r } _ { t } ) ^ { 2 } \mid { \mathcal { H } } _ { t } ] \leq E [ ( \widehat { r } _ { t } ) \mid { \mathcal { H } } _ { t } ] = \mu _ { t } ^ { r }
$$

and similarly, $\mathbb { E } [ ( Z _ { t } ^ { i } ) ^ { 2 } \mid \mathcal { H } _ { t } ] \leq \mu _ { t } ^ { i }$ . The fractional hard stop gives $\begin{array} { r } { \sum _ { t } c _ { t } ^ { i } ( p _ { t } ) \leq B _ { i } } \end{array}$ , so

$$
\sum _ { t } \mu _ { t } ^ { i } \le \operatorname* { m i n } \{ B _ { i } , T \} = Q _ { i } .
$$

Moreover, by Assumption 2.1, for every resource $i ,$

$$
\sum _ { t } \mu _ { t } ^ { r } \le \sum _ { t } \chi _ { t } \bigl ( \alpha _ { i } c _ { t } ^ { i } ( p _ { t } ) + \epsilon _ { t } ^ { i } \bigr ) \le \alpha _ { i } B _ { i } + V _ { i } ,
$$

and trivially $\textstyle \sum _ { t } \mu _ { t } ^ { r } \leq T$ . Thus the reward predictable variation is at most $Q _ { r }$

Apply Freedman’s Inequality $t 0 - Z _ { t } ^ { r }$ with $L = \log ( 2 / \delta )$ and to each of $Z _ { t } ^ { i }$ and $- Z _ { t } ^ { i }$ with $L = \log ( 4 m / \delta )$ A union bound over the resulting 1 + 2m events proves the claim. □

Lemma 5.7. Condition on (54). Let τ be the last round before either thefractional hard stop or the realized hard stop first takes effect. $H \tau < T$ , then some resource j satisfies

$$
B _ { j } - C _ { \tau } ^ { j } < 1 + \Gamma _ { j } ( \delta ) .\tag{55}
$$

Consequently, for every expert $f ,$

$$
\sum _ { t > \tau } r _ { t } ^ { f } \leq \eta _ { j } \big ( E _ { \tau } ^ { j } ( e _ { f } ) + 1 + D _ { j } + \Gamma _ { j } ( \delta ) \big ) + \mathbf { 1 } \{ \alpha _ { j } < \beta _ { j } \} \sum _ { t > \tau } \epsilon _ { t } ^ { j } .\tag{56}
$$

Proof. Recall $\begin{array} { r } { C _ { s } ^ { i } = \sum _ { t < s } c _ { t } ^ { i } ( p _ { t } ) } \end{array}$ . If the fractional hard stop is first, then its stopping rule gives $C _ { \tau } ^ { j } + 1 > B _ { j }$ for some resource $j$ . Then (56) holds even without the concentration term $\Gamma _ { j } ( \delta )$ by applying Lemma 4.4 with $h _ { j } = 1$

If the realized hard stop is first, then for some resource $\begin{array} { r } { j , \sum _ { t \leq \tau } \widehat { c } _ { t } ^ { j } + 1 > B _ { j } } \end{array}$ . Both stops are inactive through round $\tau _ { \ast }$ , so $\chi _ { t } = 1$ for all $t \leq \tau$ and therefore $\mu _ { t } ^ { j } = c _ { t } ^ { j } ( p _ { t } )$ on this prefix. Thus we have

$$
C _ { \tau } ^ { j } = \sum _ { t \leq \tau } \mu _ { t } ^ { j } \geq \sum _ { t \leq \tau } \widehat { c } _ { t } ^ { j } - \Gamma _ { j } ( \delta ) > B _ { j } - 1 - \Gamma _ { j } ( \delta ) ,
$$

which proves (55). Applying Lemma 4.4 with $h _ { j } = 1 + \Gamma _ { j } ( \delta )$ gives (56).

Theorem 5.8. Let $\mathrm { R e g } _ { \mathrm { f r a c } }$ denote the deterministicfractional regret boundfrom Theorem 4.6for the chosen feedback regime. The sampled learner satisfies every hard budget on every realization. Moreover, with probability at least $1 - \delta ,$ , simultaneouslyfor every expert $f ,$

$$
\widehat { \mathrm { R e g } } _ { T } ^ { f } : = \sum _ { t = 1 } ^ { T } r _ { t } ^ { f } - \sum _ { t = 1 } ^ { T } \widehat { r } _ { t } \leq \mathrm { R e g } _ { \mathrm { f r a c } } + \Gamma _ { r } ( \delta ) + \operatorname* { m a x } _ { i \in [ m ] } \eta _ { i } \Gamma _ { i } ( \delta ) .\tag{57}
$$

Proof. The realized hard stop guarantees pathwise budget feasibility. Condition on (54), and let τ be the last round before either stop first takes effect, with $\tau = T$ if neither stop occurs. Since both stops are inactive through τ, we have $\mu _ { t } ^ { r } = r _ { t } ( p _ { t } )$ for every $t \leq \tau$ . Hence

$$
\sum _ { t \leq \tau } ( r _ { t } ^ { f } - \widehat { r } _ { t } ) = \sum _ { t \leq \tau } ( r _ { t } ^ { f } - r _ { t } ( p _ { t } ) ) + \sum _ { t \leq \tau } ( r _ { t } ( p _ { t } ) - \widehat { r } _ { t } ) \leq R _ { \tau } ^ { f } + \Gamma _ { r } ( \delta ) .\tag{58}
$$

$\mathrm { I f } \tau = T$ , combining (58) with Theorem 4.6 proves (57).

Now suppose $\tau < T$ , and let $j$ be the resource from Lemma 5.7. The prefix-potential argument in the proof of Theorem 4.5 is unchanged through round τ . In particular, evaluating the prefix potential at $\bar { ( } e _ { f } , ( 1 - s _ { j } ) e _ { 0 } + s _ { j } e _ { j } )$ with $s _ { j } = \eta _ { j } / \lambda _ { j }$ gives

$$
R _ { \tau } ^ { f } + \eta _ { j } E _ { \tau } ^ { j } ( e _ { f } ) \leq \mathrm { S t a b } _ { \tau } + \rho _ { x } \log F + \rho _ { v } \log ( m + 1 ) ,\tag{59}
$$

where $\mathrm { S t a b } _ { \tau } = 0$ with pre-decision feedback and

$$
\mathrm { S t a b } _ { \tau } \leq \frac { T ( 1 + \eta _ { \mathrm { m a x } } ) ^ { 2 } } { \rho _ { x } }
$$

without pre-decision feedback. Combining (56),(58) and (59) yields

$$
\widehat { \mathrm { R e g } } _ { T } ^ { f } \leq \mathrm { R e g } _ { \mathrm { f r a c } } + \Gamma _ { r } ( \delta ) + \eta _ { j } \Gamma _ { j } ( \delta ) ,
$$

Since $\eta _ { j } \Gamma _ { j } ( \delta ) \leq \operatorname* { m a x } _ { i } \eta _ { i } \Gamma _ { i } ( \delta )$ , (57) follows.

We now state a corollary of composing three extensions.

Corollary 5.9. Let $\mathrm { R e g } _ { \mathrm { f r a c } }$ denote the deterministicfractional regret bound ofthe chosen base algorithm: Theorem 4.6for the core algorithm, or Theorem 5.5 when Extension 2 is used. Applying Extension 1 does not change this deterministic bound. Then the sampled learner satisfies every hard budget on every realization and, with probability at least $1 - \delta ,$ simultaneously for every expert $f ,$

$$
\widehat { \mathrm { R e g } } _ { T } ^ { f } \leq \mathrm { R e g } _ { \mathrm { f r a c } } + \Gamma _ { r } ( \delta ) + \operatorname* { m a x } _ { i \in [ m ] } \eta _ { i } \Gamma _ { i } ( \delta ) .
$$

IfExtension 2 is used, then on the same event the realized resource-i cost path is $( W _ { i } + \Gamma _ { i } ( \delta ) )$ )-subpacingfor every $i \in [ m ]$

Proofsketch. Extension 1 raises each $D _ { i } .$ -subpacing expert to a virtual $D _ { i } { \mathrm { - } } { \mathrm { p a c e d } }$ expert without changing any parameters entering the regret bound. Run Extension 2 on this raised instance. The algorithm is W<sub>i</sub>-subpacing and budget-feasible, and since the true costs are pointwise dominated by the raised costs, downward closure transfers both properties to the true fractional path. Finally, Extension 3 samples this fractional path with a realized hard stop. The realized hard stop guarantees budget feasibility pathwise. Conditioned on (54), sampling increases the fractional regret bound by at most $\Gamma _ { r } ( \delta ) + \mathrm { m a x } _ { i \in [ m ] } \eta _ { i } \Gamma _ { i } ( \delta )$ by the same argument as in the proof of Theorem 5.8 and observing that the claims of Lemma 4.4 still hold with Extension 2 (See Section 5.2). Sampling also preserves learner subpacing up to the concentration radius $\Gamma _ { i } ( \delta )$ : since $0 \leq \mu _ { t } ^ { i } \leq c _ { t } ^ { i } ( p _ { t } )$ pointwise and $( c _ { t } ^ { i } ( p _ { t } ) ) _ { t \leq T }$ is $W _ { i } .$ -subpacing, Theorem 5.5 implies that $( \mu _ { t } ^ { i } ) _ { t \leq T }$ is $W _ { i }$ -subpacing. Let $d _ { t } ^ { i } \in [ 0 , b _ { t } ^ { i } ]$ be a corresponding witness. On the event (54), for every prefix $s ,$

$$
\left| \sum _ { t \leq s } ( \widehat { c } _ { t } ^ { i } - d _ { t } ^ { i } ) \right| \leq \left| \sum _ { t \leq s } ( \mu _ { t } ^ { i } - d _ { t } ^ { i } ) \right| + \left| \sum _ { t \leq s } ( \widehat { c } _ { t } ^ { i } - \mu _ { t } ^ { i } ) \right| \leq W _ { i } + \Gamma _ { i } ( \delta ) .
$$

Thus the realized sampled resource-i path is $( W _ { i } + \Gamma _ { i } ( \delta ) )$ -subpacing.

## 6 Lower Bounds

In this section, we show that the regret of any algorithm against a set of experts must depend on the Kolmogorov distance between their spending patterns, regardless of whether the algorithm has pre-decision feedback. Theorem 6.1 states that for any spending patterns that are D apart and for any algorithm, there exists an instance involving only two actions and two experts on which the algorithm incurs $\Omega ( D )$ regret. Theorem 6.2 states that for any D and for any number of experts $F ,$ there exists a single instance on which any algorithm incurs $\Omega ( D \sqrt { \log { F } } )$ regret. Both results hold even in the ORA regime. The difference between the two results, outside of the number of experts, is that the first result holds for any choice of spending patterns, while the construction in the second result chooses a specific baseline spending pattern. The constructions for both results can be found in Braverman et al. [2025], but we outline them here to be self-contained.

Theorem 6.1 (Braverman et al. [2025]). For any two spending patterns s, $\pmb { s } ^ { \prime } \in [ 0 , 1 ] ^ { T }$ such that $\textstyle \sum _ { t } s _ { t } =$ $\textstyle \sum _ { t } s _ { t } ^ { \prime } = B$ andfor any algorithm that spends at most B in expectation, there exists a set ofexperts $\mathcal { F }$ and a sequence ofrewards and costs such that the algorithm incurs expected regret at least $\operatorname { K o l } ( s , s ^ { \prime } ) / 4$ . This lower bound holds with $| \mathcal { F } | = 2$ and even ifthe algorithm can observe the rewards and costs ofthe current round before making its decision.

The main idea behind Theorem 6.1 is the spend-or-save dilemma: if there exists a time $\tau$ when the cumulative spending of two spending patterns differ by D, then the expert that spent more before $\tau$ does better in a world where the most valuable rewards come before τ, while the expert that spent less before $\tau$ does better in a world where the most valuable reward come after τ. If the rewards and costs up to time $\tau$ are identical between the two worlds (so an algorithm cannot distinguish between the two worlds), then the algorithm once again faces a spend-or-save dilemma proportional to the difference in spending between the two experts at time $\tau .$

Proof. Let $D = \operatorname { K o l } ( s , s ^ { \prime } )$ and $\textstyle S _ { \tau } = \sum _ { t = 1 } ^ { \tau } s _ { t }$ . Define $S _ { \tau } ^ { \prime }$ similarly for $\pmb { s } ^ { \prime } .$ . The construction consists of two actions $\mathcal { A } = \{ \top , \bot \}$ and two experts $\overline { { \mathcal { F } } } = \{ \pmb { x } , \pmb { x } ^ { \prime } \}$ where

$$
x _ { t } ( \top ) = s _ { t } \quad { \mathrm { a n d } } \quad x _ { t } ( \bot ) = 1 - s _ { t }
$$

$$
x _ { t } ^ { \prime } ( \top ) = s _ { t } ^ { \prime } \quad \mathrm { a n d } \quad x _ { t } ^ { \prime } ( \bot ) = 1 - s _ { t } ^ { \prime } .
$$

Let τ denote a round where $| S _ { \tau } - S _ { \tau } ^ { \prime } | = D$ , and assume without loss of generality that $S _ { \tau } - S _ { \tau } ^ { \prime } = D$ . Note that $\tau \leq T - D$ since

$$
\begin{array} { r } { D = \sum _ { t = 1 } ^ { \tau } ( s _ { t } - s _ { t } ^ { \prime } ) \leq B - \sum _ { t = 1 } ^ { \tau } s _ { t } ^ { \prime } = \sum _ { t = \tau + 1 } ^ { T } s _ { t } ^ { \prime } \leq T - \tau . } \end{array}
$$

Define two sequences of rewards $\boldsymbol { r } , \boldsymbol { r } ^ { \prime }$ and costs $\displaystyle c , c ^ { \prime } .$

$$
r _ { t } ( \perp ) = r _ { t } ^ { \prime } ( \perp ) = c _ { t } ( \perp ) = c _ { t } ^ { \prime } ( \perp ) = 0
$$

$$
r _ { t } ( \top ) = { \left\{ \begin{array} { l l } { 1 / 2 } & { t \leq \tau } \\ { 0 } & { t > \tau } \end{array} \right. }
$$

$$
r _ { t } ^ { \prime } ( \top ) = \left\{ { 1 } / { 2 } \right. \ t \leq \tau
$$

$$
c _ { t } ( \top ) = c _ { t } ^ { \prime } ( \top ) = 1
$$

Note that because the first τ rounds are identical between the two sequences, the expected spending of the algorithm is the same against both sequences by round τ. Let $B _ { \tau }$ denote this quantity. Since the algorithm spends at most B in expectation, we have that the expected reward of the algorithm against the first sequence is exactly $B _ { \tau } / 2$ , while its expected reward against the second sequence is at most $B _ { \tau } / \bar { 2 + } ( B - B _ { \tau } ) = \bar { B ^ { } } - B _ { \tau } / 2$ Meanwhile, the two experts obtain exactly $S _ { \tau } / 2$ and $S _ { \tau } ^ { \prime } / 2$ against the first sequence and exactly $B - S _ { \tau } / 2$ and $B - S _ { \tau } ^ { \prime } / 2$ against the second sequence. If the algorithm has expected regret at most $R$ on both sequences, then

$$
2 R \geq \left( S _ { \tau } / 2 - B _ { \tau } / 2 \right) + \left( ( B - S _ { \tau } ^ { \prime } / 2 ) - ( B - B _ { \tau } / 2 ) \right) = ( S _ { \tau } - S _ { \tau } ^ { \prime } ) / 2 = D / 2 .
$$

Theorem 6.2 (Braverman et al. [2025]). Let $B = T / 2 .$ . For all $D \geq 1$ and $F \geq 2$ satisfying $T \geq 2 D$ log $F ,$ there exists a baseline spending pattern, a set ofF experts that are D-pacing with respect to this baseline spending pattern, and a sequence of (random) rewards and costs such that the expected regret of any budgetfeasiblefractional-play algorithm is at least $\Omega ( D { \sqrt { \log F } } )$ . This lower bound holds even ifthe algorithm can observe the rewards and costs ofthe current round before making its decision.

ProofSketch. We outline the construction and provide a sketch for why it witnesses the desired lower bound. See Braverman et al. [2025] for the full proof.<sup>11</sup>

For simplicity, assume $F$ is a power of 2 and $D \in \mathbb { Z } ^ { + }$ . This is without loss up to constant factors. The construction consists of two non-null actions (so three actions total) and repeats a spend-or-save dilemma in each of log F windows of length 2D. Each window $i \in \{ 1 , \ldots , \log F \}$ is split into two halves. The first action yields reward $a _ { i } \in [ 0 , 1 ]$ in each round in the first half and 0 reward in each round in the second half. The second action does the opposite: it yields 0 reward in each round in the first half but $b _ { i } \in [ 0 , 1 ]$ reward in each round in the second half. The first action incurs cost $2 B / T = 1$ when it yields reward $a _ { i }$ and incurs 0 cost otherwise. The second action incurs cost $2 B / T = 1$ when it yields reward $b _ { i }$ and incurs 0 cost otherwise. The set of experts is indexed by $\{ 0 , 1 \} ^ { \log F }$ where a 0 in coordinate i means that the expert chooses the first action in the first half (and the null action in the second half) and a 1 means that the expert chooses the second action in the second half (and the null action in the first half). If there are any rounds remaining after log F windows, the reward and cost of each non-null action is 1 and $B / T = 1 / 2$ , respectively, and each expert gets reward 1 in each remaining round. This means that any money saved during the log $F$ windows by the algorithm cannot be spent to catch up to any expert during this time period. Let the baseline spending pattern be the uniform pacing spending pattern, i.e. spending $\bar { B / T }$ per round. Then each expert is indeed D-pacing against this baseline spending pattern: the cumulative deviation is at most $D$ at any timestep. Let $B _ { 0 } = \supset \log F$ be the cumulative spending of the uniformly pacing spending pattern during the log $\scriptstyle { \dot { F } }$ windows.

The rewards $a _ { 1 } , b _ { 1 } , \dotsc , a _ { \log F } , b _ { \log F }$ are given by an unbiased random walk with step size $\varepsilon = 1 / \sqrt { \log F }$ that starts at $a _ { 1 } = 1 / 2$ and terminates when it hits either 0 or 1. The expected reward of the best expert is

$$
\begin{array} { r } { \mathrm { O P T } = D \cdot \mathbb { E } [ \sum _ { i = 1 } ^ { \log F } \operatorname* { m a x } \{ a _ { i } , b _ { i } \} ] + ( T - 2 D \log F ) . } \end{array}
$$

Meanwhile, let $C$ denote the actual cumulative spending of the algorithm within the log F windows. Then its expected total reward at most

$$
\begin{array} { r l } & { \quad a _ { 1 } B _ { 0 } + \mathbb { E } [ ( C - B _ { 0 } ) _ { + } ] + \mathbb { E } [ \operatorname* { m i n } \{ T - 2 D \log F , 2 ( B - C ) \} ] } \\ & { = D \cdot \mathbb { E } \left[ \displaystyle \sum _ { i = 1 } ^ { \log F } \frac { a _ { i } + b _ { i } } { 2 } \right] + \mathbb { E } [ ( C - B _ { 0 } ) _ { + } ] + \mathbb { E } [ \operatorname* { m i n } \{ T - 2 D \log F , 2 ( B - C ) \} ] } \\ & { \le D \cdot \mathbb { E } \left[ \displaystyle \sum _ { i = 1 } ^ { \log F } \frac { a _ { i } + b _ { i } } { 2 } \right] + ( T - 2 D \log F ) . } \end{array}
$$

The key observation behind this fact is that for $\tau \leq 2 D$ log F,

$$
\begin{array} { r } { Q _ { \tau } = \left( B _ { 0 } - \sum _ { t = 1 } ^ { \tau } c _ { t } ( x _ { t } ) \right) \cdot \left( a _ { \lceil \tau / 2 D \rceil } \cdot \mathbb { 1 } ( \tau \ \mathrm { f i r s t } \mathrm { \ h a l f } ) + b _ { \lceil \tau / 2 D \rceil } \cdot \mathbb { 1 } ( \tau \ \mathrm { s e c o n d } \mathrm { h a l f } ) ) \right) + \sum _ { t = 1 } ^ { \tau } r _ { t } ( x _ { t } ) } \end{array}
$$

remains a martingale, even when $x _ { t }$ is allowed to depend on $r _ { t } \mathrm { : }$ the future value of money remains equal in expectation to its current value, even if the algorithm knows the current round’s reward.

The difference in reward is then

$$
\begin{array} { r } { \Omega ( D ) \cdot \mathbb { E } [ \sum _ { i = 1 } ^ { \log F } | a _ { i } - b _ { i } | ] } \end{array}
$$

Note that with step size $\varepsilon = \Theta ( 1 / \sqrt { \log F } )$ , the random walk does not terminate after 2 log F steps with constant probability. It follows that any algorithm’s regret is $\Omega ( D { \sqrt { \log F } } )$ . □

Together with the standard no-regret lower bound, we get that in the OLRC setting, no learner can guarantee regret o(max $\{ D , T ^ { 1 / 2 } \} \cdot \sqrt { \log F } )$

Theorem 6.3 (Braverman et al. [2025]). For all $D \geq 1$ and $F \geq 2$ satisfying $T \geq 2 D$ log F, there exists a budget B, a baseline spending pattern, a set of F experts that are D-pacing with respect to this baseline spending pattern, and a sequence of (random) rewards and costs such that the expected regret of any budget-feasiblefractional-play learning algorithm is at least $\Omega ( \operatorname* { m a x } \{ D , T ^ { 1 / 2 } \} \cdot \sqrt { \log F } )$

AI Disclosure: The authors used OpenAI’s ChatGPT extensively in the development of this work. The authors independently came up with an equivalent algorithm for the warm-up example in Section 3, and ChatGPT found an equivalent formulation of the algorithm which is generalizable to broader settings. In particular, the main algorithms and analysis in Section 4.3 and Section 5 presented in the paper were generated and written through interactions with ChatGPT. ChatGPT was also used to investigate related literature and improve the writing. The authors subsequently checked and verified all the mathematical claims, algorithms, proofs, and references appearing in the submitted manuscript and take full responsibility for their correctness and originality.

## References

Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins. Bandits with knapsacks. In Proceedings ofthe 2013 IEEE 54th Annual Symposium on Foundations ofComputer Science, FOCS ’13, page 207–216, USA, 2013. IEEE Computer Society. ISBN 9780769551357. doi: 10.1109/FOCS.2013.30. URL https://doi.org/10.1109/FOCS.2013.30.

Santiago R. Balseiro and Yonatan Gur. Learning in repeated auctions with budgets: Regret minimization and equilibrium. Manage. Sci., 65(9):3952–3968, September 2019. ISSN 0025-1909. doi: 10.1287/mnsc.2018. 3174. URL https://doi.org/10.1287/mnsc.2018.3174.

Santiago R. Balseiro, Haihao Lu, and Vahab Mirrokni. The best of many worlds: Dual mirror descent for online allocation problems. Operations Research, 71(1):101–119, 2023. doi: 10.1287/opre.2021.2242. URL https://doi.org/10.1287/opre.2021.2242.

Christian Borgs, Jennifer Chayes, Nicole Immorlica, Kamal Jain, Omid Etesami, and Mohammad Mahdian. Dynamics of bid optimization in online advertisement auctions. In Proceedings of the 16th International Conference on World Wide Web, WWW ’07, page 531–540, New York, NY, USA, 2007. Association for Computing Machinery. ISBN 9781595936547. doi: 10.1145/1242572.1242644. URL https: //doi.org/10.1145/1242572.1242644.

Mark Braverman, Jingyi Liu, Jieming Mao, Jon Schneider, and Eric Xue. A new benchmark for online learning with budget-balancing constraints, 2025. URL https://arxiv.org/abs/2503.14796.

Matteo Castiglioni, Andrea Celli, and Christian Kroer. Online learning with knapsacks: the best of both worlds. In International Conference on Machine Learning, pages 2767–2783. PMLR, 2022a.

Matteo Castiglioni, Andrea Celli, Alberto Marchesi, Giulia Romano, and Nicola Gatti. A unifying framework for online optimization with long-term constraints. Advances in Neural Information Processing Systems, 35:33589–33602, 2022b.

Matteo Castiglioni, Andrea Celli, and Christian Kroer. Online learning under budget and roi constraints via weak adaptivity. In Forty-first International Conference on Machine Learning, 2024.

Nicolo Cesa-Bianchi and Gabor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

Tianyi Chen and Georgios B Giannakis. Bandit convex optimization for scalable and dynamic iot management. IEEE Internet of Things Journal, 6(1):1276–1286, 2018.

Giannis Fikioris and Éva Tardos. Approximately stationary bandits with knapsacks. In Gergely Neu and Lorenzo Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 3758–3782. PMLR, 12–15 Jul 2023. URL https://proceedings.mlr.press/v195/fikioris23a.html.

David A Freedman. On tail probabilities for martingales. the Annals of Probability, pages 100–118, 1975.

Jason Gaitonde, Yingkai Li, Bar Light, Brendan Lucier, and Aleksandrs Slivkins. Budget Pacing in Repeated Auctions: Regret and Efficiency Without Convergence. In Yael Tauman Kalai, editor, 14th Innovations in Theoretical Computer Science Conference (ITCS 2023), volume 251 of Leibniz International Proceedings in Informatics (LIPIcs), pages 52:1–52:1, Dagstuhl, Germany, 2023. Schloss Dagstuhl – Leibniz-Zentrum für Informatik. ISBN 978-3-95977-263-1. doi: 10.4230/LIPIcs.ITCS.2023.52. URL https://drops. dagstuhl.de/entities/document/10.4230/LIPIcs.ITCS.2023.52.

Alison L. Gibbs and Francis Edward Su. On choosing and bounding probability metrics. International Statistical Review / Revue Internationale de Statistique, 70(3):419–435, 2002. ISSN 03067734, 17515823. URL http://www.jstor.org/stable/1403865.

Nicole Immorlica, Karthik Sankararaman, Robert Schapire, and Aleksandrs Slivkins. Adversarial bandits with knapsacks. J. ACM, 69(6), November 2022. ISSN 0004-5411. doi: 10.1145/3557045. URL https://doi.org/10.1145/3557045.

Rodolphe Jenatton, Jim Huang, Dominik Csiba, and Cedric Archambeau. Online optimization and regret guarantees for non-additive long-term constraints, 2016. URL https://arxiv.org/abs/1602. 05394.

Nikolaos Liakopoulos, Apostolos Destounis, Georgios Paschos, Thrasyvoulos Spyropoulos, and Panayotis Mertikopoulos. Cautious regret minimization: Online optimization with long-term budget constraints. In International Conference on Machine Learning, pages 3944–3952. PMLR, 2019.

Shang Liu, Jiashuo Jiang, and Xiaocheng Li. Non-stationary bandits with knapsacks. Advances in Neural Information Processing Systems, 35:16522–16532, 2022.

Brendan Lucier, Sarath Pattathil, Aleksandrs Slivkins, and Mengxiao Zhang. Autobidders with budget and roi constraints: Efficiency, regret, and pacing dynamics. In Shipra Agrawal and Aaron Roth, editors, Proceedings ofThirty Seventh Conference on Learning Theory, volume 247 of Proceedings ofMachine Learning Research, pages 3642–3643. PMLR, 30 Jun–03 Jul 2024. URL https://proceedings. mlr.press/v247/lucier24a.html.

Mehrdad Mahdavi, Rong Jin, and Tianbao Yang. Trading regret for efficiency: Online convex optimization with long term constraints. Journal of Machine Learning Research, 13(81):2503–2528, 2012. URL http://jmlr.org/papers/v13/mahdavi12a.html.

Shie Mannor, John N Tsitsiklis, and Jia Yuan Yu. Online learning with sample path constraints. Journal of Machine Learning Research, 10(3), 2009.

Michael J Neely and Hao Yu. Online convex optimization with time-varying constraints. arXiv preprint arXiv:1702.04783, 2017.

Abhishek Sinha and Rahul Vaze. Optimal algorithms for online convex optimization with adversarial constraints. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=TxffvJMnBy.

Maurice Sion. On general minimax theorems. Pacific J. Math., 8(4):171–176, 1958. URL http://dml. mathdoc.fr/item/1103040253.

Aleksandrs Slivkins, Karthik Abinav Sankararaman, and Dylan J Foster. Contextual bandits with packing and covering constraints: A modular lagrangian approach via regression. In The Thirty Sixth Annual Conference on Learning Theory, pages 4633–4656. PMLR, 2023.

Francesco Emanuele Stradi, Matteo Castiglioni, Alberto Marchesi, Nicola Gatti, and Christian Kroer. Noregret learning under adversarial resource constraints: A spending plan is all you need! In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 130556–130604. Curran Associates, Inc., 2025. doi: 10.52202/085713-4351. URL https://proceedings.neurips.cc/paper\_files/ paper/2025/file/bd5c3c51db72a6614bb71ce5318a78d0-Paper-Conference.pdf.

Wen Sun, Debadeepta Dey, and Ashish Kapoor. Safety-aware algorithms for adversarial contextual bandit. In International Conference on Machine Learning, pages 3280–3288. PMLR, 2017.

## A Summary of Guarantees for One Resource

Suppose there is a single resource, and suppose the parameters α and V in Assumption 2.1 are fixed constants. Let D be the pacing or subpacing width of the experts and assume $D \geq 1 , F \geq { \bar { 2 } }$ , and $T \geq 1$ . Let B be the budget. Write $\mathrm { R e g } _ { T } ^ { f }$ for regret against expert f and $\mathrm { R e g } _ { T } : = \operatorname* { m a x } _ { f } \mathrm { R e g } _ { T } ^ { f }$ . The notation $\widetilde { O } _ { T }$ only suppresses polylogarithmic factors in $\mathbf { \bar { \boldsymbol { T } } }$ , so all dependence on F remains explicit. Under these simplifications, the main guarantees are:

• Fractional play, D-pacing/subpacing experts. Algorithm 2 and Algorithm 3 (and optionally with Extension 1 in Section 5.1) satisfy fractional budget feasibility, and simultaneously for every expert $f ,$

$$
\mathrm { R e g } _ { T } ^ { f } = O \Big ( D \sqrt { \log F } \Big )
$$

with pre-decision feedback,

and

$$
\mathrm { R e g } _ { T } ^ { f } = O \Big ( ( \sqrt { T } + D ) \sqrt { \log F } \Big )
$$

without pre-decision feedback.

Note that the same rates hold for D-pacing and D-subpacing experts, by Theorem 5.3.

• Fractional play, subpacing learning algorithm. Algorithm 2 and Algorithm 3 with Extension 2 in Section 5.2 satisfy fractional budget feasibility and achieve

$$
\mathrm { R e g } _ { T } ^ { f } = O \Big ( D \sqrt { \log F } \Big )
$$

with pre-decision feedback,

and

$$
\mathrm { R e g } _ { T } ^ { f } = O \Bigl ( ( \sqrt { T } + D ) \sqrt { \log F } \Bigr )
$$

without pre-decision feedback.

Their fractional spending path can be made W-subpacing, with

$$
W = \widetilde { O } _ { T } \Big ( D \sqrt { \log F } \Big )
$$

with pre-decision feedback,

and

$$
W = \widetilde { O } _ { T } \bigg ( D + \frac { D ^ { 2 } \sqrt { \log { F } } } { \sqrt { T } + D } \bigg )
$$

without pre-decision feedback

• Sampled play under full-information feedback. Algorithm 2 and Algorithm 3 with Extension 3 in Section 5.3 satisfy the budget constraint with sampled actions, and with probability at least $1 - \delta .$ , it adds at most

$$
O \left( \sqrt { \left( 1 + \operatorname* { m i n } \{ B , T \} \right) \log \frac { 1 } { \delta } } + \log \frac { 1 } { \delta } \right)
$$

to the fractional regret bounds. If the fractional path is $W$ -subpacing, the realized sampled path has the same $W$ plus a deviation of this order.