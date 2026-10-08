# Online Resource Allocation with an Endogenous Markov State: Fewer LP Solves Earn More

Zhaohua Chen

Taobao & Tmall Group of Alibaba, Renmin University of China chenzhaohua.czh@alibaba-inc.com

## Abstract

We study finite-horizon online resource allocation with i.i.d. requests and an endogenous Markov state on a finite state space: each action afects the transition of the state that governs future rewards and resource consumption. In this problem, a transient fluid LP benchmark upper bounds the expected reward of every nonanticipating policy, while a stationary LP supplies randomized state-dependent controls. We assume that the stationary LP has a unique optimum and identify primal nondegeneracy and irreducibility of the optimal induced kernel as important regularity conditions in this framework. With a known request prior, we show that, under nondegeneracy and irreducibility, both frequent and infrequent re-solving attain O(1) regret. However, under a degenerate optimum, irreducibility yields the sharp worst-case $\Theta ( { \sqrt { T } } )$ rate for infrequent re-solving, while frequent re-solving can incur Ω(T) regret. Thus, more frequent optimization can perform asymptotically worse. With an unknown request prior, we develop a three-phase U-shaped infrequent re-solving policy that coordinates learning and inventory correction with O(log log T) LP solves. When the optimal induced kernel is irreducible and the algorithm is given the optimal target state class and a constant-cost entrance policy, it attains O(1) regret under nondegeneracy and $O ( \sqrt { T } )$ regret under degeneracy. Without the target-class information, linear minimax regret is unavoidable. Numerical experiments further illustrate the instability of round-by-round re-solving relative to epoch-wise infrequent re-solving, show that thresholding greatly mitigates its loss, and find that infrequent schemes remain dominant under both known and estimated priors.

## 1 Introduction

Consider the following scenario. A service provider that uses large models repeatedly receives requests from users throughout the day. Because its computing resources are limited, it has a daily token budget and a daily running-time budget for each cluster. Upon receiving each request, it decides which model should handle the request and on which cluster the task should run. Diferent models require diferent numbers of tokens. The reward that the service provider receives depends on the quality of the final output and the running time of the task.

This problem is, in general, an online resource allocation problem. However, it involves additional factors. The running time of a task depends strongly on the state of the cluster on which it is deployed. A lightly loaded cluster runs at its normal speed but operates significantly more slowly under a heavy load. Therefore, the cluster states afect both resource consumption and reward. In turn, the service provider’s action and the current cluster states determine the next cluster states, either deterministically or randomly. This state factor is not captured by the classical model of online resource allocation.

More generally, rewards and resource consumption are often afected by an endogenous environmental “state” that evolves according to the chosen action. Suppose an algorithm engineer maintains a retrieval-augmented generation (RAG) knowledge base subject to a fixed token budget. For each query, she must decide whether to update the knowledge base using additional tokens before answering or to answer directly. Without an update, the token cost is lower, but the response quality may be worse when the state indicates that the knowledge base is stale. As another example, a retailer may be building a long-term relationship with a customer. Its reward from serving the customer’s request depends on how much the customer trusts the retailer. This trust evolves with the quality of the service that the customer receives.

To capture these real-world scenarios, we augment the online resource allocation model with an endogenous Markov state. To explain our model, we first recall the online resource allocation problem.

In the online resource allocation framework, there are T rounds and m resources. The initial total resource inventory is $B ^ { 1 } = \rho \cdot T$ , where $\rho \in \mathbb { R } _ { + + } ^ { m }$ is a constant vector independent of $T .$ . In each round $t ,$ the decision maker observes an i.i.d. request $J ^ { t }$ and chooses an action $A ^ { t }$ . This action produces a reward and consumes resources, as captured by

$$
R ^ { t } = r ( J ^ { t } , A ^ { t } ) \geq 0 , \qquad C ^ { t } = c ( J ^ { t } , A ^ { t } ) \geq 0 .
$$

The decision maker’s goal is to maximize the expected total reward subject to the initial resource constraint, that is,

$$
\operatorname* { m a x } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } R ^ { t } \right] , \qquad \mathrm { s . t . } \sum _ { t = 1 } ^ { T } C ^ { t } \leq \rho \cdot T .
$$

To ensure feasibility, we assume that there is a null action that always produces zero reward and consumes no resources, regardless of the request.

We extend this classical model as follows: in each round t, the decision maker observes the current state $S ^ { t }$ before observing the request. The reward and resource consumption under action

$A ^ { t }$ are determined by $S ^ { t }$ and $J ^ { t } ,$ as follows:

$$
R ^ { t } = r ( S ^ { t } , J ^ { t } , A ^ { t } ) , \qquad C ^ { t } = c ( S ^ { t } , J ^ { t } , A ^ { t } ) .
$$

Meanwhile, action $A ^ { t }$ causes a Markov state transition from $S ^ { t }$ to $S ^ { t + 1 }$ under the transition kernel

$$
S ^ { t + 1 } \sim P ( \cdot \mid S ^ { t } , A ^ { t } ) .
$$

We note that the null action produces zero reward and consumption for every state and request but can still induce a state transition. All other components of the problem are identical to those of the classical online resource allocation problem.

Our goal is to develop algorithms that perform well in this online decision-making problem.

## 1.1 Main contributions

Benchmarks and regularity. To evaluate policies for our online resource allocation problem with a Markov state, we consider two closely related benchmarks. The first is a transient fluid LP, which provides an upper bound on the total expected reward of any nonanticipating policy. However, this benchmark is “asymmetric” because the state in the initial round is fixed, which makes the benchmark dificult to use directly to guide an online policy. Therefore, we also consider a stationary fluid LP benchmark whose value is smaller than that of the transient fluid LP by at most a constant. Like its counterpart in the original online resource allocation problem, this LP directly induces a randomized policy in each round by using normalized variables from its optimal solution as action probabilities.

Using the fluid LP benchmark means that regret on the order of $\sqrt { T }$ is unavoidable in the worst case even for state-free online resource allocation. Nevertheless, this choice is necessary: the hindsight benchmark is strictly stronger in the state-free case but is unsuitable in the Markov-state setting. Without a state, the number of arrivals of each request type determines the hindsight benchmark. With a state, the order in which requests arrive also matters because it afects how the state evolves. Incorporating this order, however, would make the benchmark unattainably large.

With a fluid benchmark, regularity conditions are crucial to algorithmic performance. We consider three regularity conditions. First, we impose the following condition throughout the paper.

• The stationary LP has a unique optimum.

On the algorithmic side, assuming a unique optimum avoids technical complications. Without uniqueness, algorithms based on the stationary LP must specify which optimizer to use, especially to ensure consistency across solves. To avoid these complications, we focus on the unique-optimum case.

The other two conditions are as follows:

• The stationary LP has a nondegenerate optimum.

• The optimal induced kernel—the state transition probability matrix on the optimal target state class—is irreducible, and there is an entrance policy that guarantees entry into that class in a constant expected number of rounds and with constant expected resource consumption.

The first of these two conditions has already proved important in online resource allocation, separating O(1) regret from Θ( T) regret (Proposition 5.1). The second condition is specific to the Markov-state setting. It states that under the optimal fluid solution, all states that should be reached can be reached. As we will show, this condition is both necessary and suficient for the gap between the LP benchmark and the optimal policy to be sublinear (Proposition 5.4).

Frequent versus infrequent re-solving under a known request prior. In the known-requestprior case, we examine two algorithms: frequent re-solving and infrequent re-solving. Variants of these algorithms have already been studied in the state-free setting, and we extend them to the Markov-state environment. After entering the optimal target state class, frequent re-solving (FR) generally re-solves the stationary LP using the current average remaining inventory in each round and uses the resulting randomized control in that round. By contrast, infrequent re-solving (IR) solves this stationary LP only at carefully scheduled rounds, for a total of O(log log T) re-solves. Previous work on these two methods (Bumpensanti and Wang, 2020; Li et al., 2026; Chen et al., 2024a) suggests the following observations:

O1. When the stationary LP has a unique nondegenerate optimum, FR has been shown to have O(1) regret. IR also appears to have O(1) regret, but FR performs better than IR.

<sub>O2. When the stationary LP has a unique degenerate optimum, FR has been shown to have Θ(</sub>√<sub>T)</sub> regret, while IR also appears to have Θ( T) regret.

O3. When the same thresholding/argmax operation is applied to both heuristics, the variant of FR appears to perform better than the corresponding variant of IR.

In this work, we revisit these observations for the more general problem with a Markov state. Table 1 presents our theoretical results:

R1. When the stationary LP has a unique nondegenerate optimum, both FR and IR have O(1) regret, while IR can perform better than FR.

R2. When the stationary LP has a unique degenerate optimum, the worst-case regret of IR is <sub>Θ(</sub>√<sub>T), while that of FR is Θ(T).</sub>

R3. Although the thresholded variant FRT can perform much better than FR, our numerical experiments show that it can still perform worse than IRT.

Thus, the comparison between frequent and infrequent re-solving can reverse in the presence of a Markov state, as a concrete support to the conceptual result of Secomandi (2008), which indicates that re-solving may not always lead to an improvement.

Why can IR perform better than FR in the presence of a Markov state, contrary to the pattern observed in the state-free setting? Our counterexamples suggest the following mechanism. A policy may move the system to a temporary state because it also prescribes a recovery action there. After the system has moved, however, frequent re-solving can produce a new fluid optimizer that assigns zero stationary probability to that state and omits the recovery action. Under infrequent re-solving, holding the policy fixed within an epoch preserves the sequence of decisions that makes the initial action worthwhile. Thus, with a dynamically evolving state, retaining an epoch-wise stationary policy for some time can allow the fluid-LP objective to be achieved asymptotically, whereas changing the policy too frequently can be harmful.

Table 1: Known-prior regret relative to the transient fluid benchmark.
<table><tr><td rowspan="2">Policy</td><td colspan="2">Optimal induced kernel irreducible</td><td>Optimal induced kernel reducible</td></tr><tr><td>unique, non-degen.</td><td>unique, degen.</td><td>unique, non-degen.</td></tr><tr><td>FR</td><td>O(1)</td><td>Θ(T)</td><td>Θ(T)</td></tr><tr><td>IR</td><td>O(1)</td><td>Θ(√T)</td><td>Θ(T)</td></tr><tr><td>Best policy</td><td>0(1)</td><td>Θ(√T)</td><td>Θ(T)</td></tr></table>

Note. The lower bounds, and hence the Θ(·) entries, are worst-case orders over the corresponding classes of instances.

Table 2: Unknown-prior regret relative to the transient fluid benchmark.
<table><tr><td>Policy</td><td colspan="2">Known optimal target state class</td><td>Unknown optimal target state class</td></tr><tr><td></td><td>unique, non-degen.</td><td>unique, degen.</td><td>unique, degen.</td></tr><tr><td>UIR-UP</td><td>0(1)</td><td> $\Theta ( { \sqrt { T } } )$ </td><td></td></tr><tr><td>Best policy</td><td>O(1)</td><td> $\Theta ( { \sqrt { T } } )$ </td><td>Θ(T)</td></tr></table>

Note. The lower bounds, and hence the Θ(·) entries, are worst-case orders over the corresponding classes of instances.

Three-phase infrequent re-solving under an unknown request prior. In the unknownrequest-prior case, under the assumption that the optimal target state class is known, we propose a three-phase infrequent re-solving policy. The three phases serve diferent purposes. The first phase, with O(log log log T) re-solves, aims to correct inaccurate early prior estimates quickly when few request samples are available. This phase lasts only $O ( \log ^ { 2 } T )$ rounds. The second phase, which starts at the end of the first phase and ends at the midpoint of the horizon, balances learning error and inventory consumption. For this phase, we design a new schedule that difers from those in previous work and guarantees $O ( \sqrt { T } )$ regret in the degenerate case relative to the fluid benchmark. This schedule also has O(log log T) re-solves. The construction of this schedule may be of independent interest. Informally, it satisfies

$$
\sum _ { k = 0 } ^ { K - 1 } \frac { t _ { k + 1 } - t _ { k } } { \sqrt { t _ { k } } } = O ( \sqrt { T } ) ,
$$

where $K = O ( \log \log T )$ is the number of re-solves, $t _ { 0 } = \Theta ( \log ^ { 2 } T )$ , and $t _ { K } = \Theta ( T )$ . The last phase mirrors the designs of Bumpensanti and Wang (2020); Li et al. (2026) by re-solving increasingly frequently near the end, when inventory errors become costly.

Our policy, which we call U-shaped IR under unknown priors (UIR-UP), has the following

optimal regret guarantees:

1. When the stationary LP has a unique and nondegenerate optimum, it has O(1) regret.

2. When the stationary LP has a unique and degenerate optimum, it has $O ( \sqrt { T } )$ regret.

Knowledge of the optimal target state class is required by our UIR-UP policy and is necessary for achieving sublinear regret. To show this, we consider two request distributions with diferent optimal target state classes that cannot be distinguished from observations suficiently early. Therefore, without knowing either the optimal target state class or the request prior, any policy incurs linear regret on at least one of these two instances. All theoretical results for the unknown-request-prior case are presented in Table 2.

In our numerical experiments, we compare four policies in the unknown-request-prior case: UIR-UP; FR-UP, a direct extension of FR to the unknown-prior setting; UIRT-UP; and FRT-UP. The last two are the thresholded variants of UIR-UP and FR-UP, respectively. Our empirical results are similar to those in the known-prior setting: FR-UP can still incur linear regret, and although thresholding prevents this trend, FRT-UP still performs worse than UIR-UP and UIRT-UP in most cases.

To further demonstrate the importance of re-solving frequency, we modify FR and FR-UP so that, after the first round, each policy independently chooses whether to re-solve the stationary LP in every round, with a constant probability. Our results show that, when the optimum is nondegenerate, algorithmic performance improves as this re-solving probability decreases, regardless of whether the request prior is known. However, this modification does not afect the regret order of FR or FR-UP, demonstrating the importance of the re-solving schedules used by IR and UIR-UP.

## 1.2 Related work

The extensive literature on online resource allocation under the small-item and long-horizon assumptions has developed along several major methodological lines.

One important line consists of primal-dual methods, motivated in part by the success of gradientbased optimization in machine learning. These methods learn or adjust shadow prices and then use them to make online allocation decisions (Agrawal et al., 2014; Li et al., 2020; Sun et al., 2020). These methods are often modular and can accommodate more general action spaces, including continuous ones (Jiang et al., 2020; Balseiro et al., 2023; Castiglioni et al., 2022), as well as more restrictive feedback models, such as bandit feedback (Han et al., 2023; Slivkins et al., 2024). Many early and broadly applicable methods, however, incur an $O ( \sqrt { T } )$ gap from optimal performance. More recent work obtains logarithmic or constant regret under diferent regularity assumptions (L and Ye, 2022; Bray, 2025; Ma et al., 2025; Gao et al., 2026; Gupta, 2024; He et al., 2025).

Our work belongs to the line of re-solving methods. These methods periodically recompute a deterministic fluid problem and implement the resulting control until the next update. For example, Reiman and Wang (2008) show that a single re-solve can already yield $o ( \sqrt { T } )$ regret. A central question in this line of work concerns regularity conditions. For example, Jasin and Kumar (2012, 2013) establish that frequent re-solving achieves constant regret under nondegeneracy assumptions. Later work studies how re-solving algorithms behave without these assumptions and how they can be enhanced to perform well in their absence. Regarding the first issue, Chen et al. (2024a) show that frequent re-solving still achieves $\tilde { O } ( \sqrt { T } )$ regret when nondegeneracy does not hold, and Balseiro et al. (2024) study a wide range of regularity conditions that yield constant, logarithmic, or square-root regret. Regarding the second issue, which is most closely related to our work, Bumpensanti and Wang (2020); Li et al. (2026) develop infrequent re-solving, which solves the LP according to a carefully designed schedule rather than in every round. They also study thresholding and argmax operations that post-process the LP solutions and substantially improve the performance of frequent and infrequent re-solving. However, their results and observations suggest that, under the same post-processing rule (none, thresholding, or argmax), frequent re-solving still performs better than infrequent re-solving. Our results show that this comparison can reverse in the presence of a Markov state, which also support the result of Secomandi (2008) – this work shows abstractly that re-solving a mathematical-programming-based control algorithm need not improve performance when sequential consistency fails. We mention that re-solving methods are also used for continuous request and action spaces (Maglaras and Meissner, 2006; Jasin, 2014, 2015; Arlotto and Xie, 2020; Jiang and Zhang, 2020; Miao and Wang, 2025; Wang and Wang, 2022; Besbes et al., 2025) and for contextual bandits and decision-making with knapsack constraints (Wu et al., 2015; Chen et al., 2024b) in machine learning.

A closely related recent line uses compensated coupling. These methods compare an online trajectory with an ofline or fluid benchmark by assigning a marginal value to the loss from each online decision (Vera and Banerjee, 2021; Vera et al., 2021; Jiang et al., 2025; Banerjee and Freund, 2020, 2025). Other relevant approaches include budget-ratio policies (Arlotto and Gurvich, 2019; Vera et al., 2025). Compensated-coupling analyses yield O(1) or O(log T) regret, particularly for discrete request and action spaces. These settings substantially overlap with those studied using re-solving methods.

Markov dependence. Recent work incorporates Markov dependence into online resource allocation models in several ways. Most extensions use an exogenous chain to model demand. In Farias and Van Roy (2007), an observed demand mode evolves as a Markov chain and determines fare-product arrival probabilities; before each arrival, the firm chooses which feasible products to ofer. In Jiang (2026), the observed state follows a known time-inhomogeneous Markov chain and determines the arriving customer’s type. Li et al. (2025) use a two-timescale model: an observed Markov chain moves once per stage, and its current state determines the distribution of the product requests arriving within that stage. In all three models, allocation decisions deplete cumulative inventory but do not afect the demand-modulating chain. Our requests are i.i.d. and observed before the decision; instead, the chosen action controls the state transition, and the resulting state afects later rewards and resource consumption.

The models of Brown and Zhang (2022) and Brown and Zhang (2025) are closer to ours because their actions also afect future states. Brown and Zhang (2022) consider many project-specific controlled Markov states together with a common exogenous Markov signal that can afect rewards, transitions, resource use, and available capacity. Brown and Zhang (2025) consider many independent, possibly heterogeneous, controlled Markov subproblems without that common signal. In both models, the decision maker acts on all subproblems in parallel, and shared resource constraints couple those decisions within each period. $\mathrm { O u r }$ model instead has one persistent controlled state, one i.i.d. request per period, and cumulative inventory that can be allocated over the horizon. Their large-system scaling increases the number of subproblems and the available per-period capacity, whereas our asymptotic parameter is the horizon. Accordingly, their fluid state describes a cross-section of many simultaneous subproblems, while our stationary occupation-measure LP describes time allocation along one controlled trajectory.

## 2 Model and benchmarks

We study an online decision-making problem. There are $T$ rounds and m resources. The initial resource inventory is $B ^ { 1 } = \rho \cdot T$ , where $\rho \in \mathbb { R } _ { + + } ^ { m }$ is a constant vector independent of $T .$ . In each round t, the decision maker first observes the current state $S ^ { t } \in S$ and a request $J ^ { t } \in \mathcal { I }$ . We assume that the state space S is finite and that requests $J ^ { 1 } , J ^ { 2 } , \ldots$ . are i.i.d. with prior distribution

$$
p = ( p _ { j } ) _ { j \in \mathcal { I } } , \qquad p _ { j } > 0 , \qquad \sum _ { j \in \mathcal { I } } p _ { j } = 1 .
$$

For the first part of our results, we assume that this prior distribution is known before any decision is made; we then extend the analysis to the unknown-prior case in Section 6.

Given these observations, the decision maker chooses an action $A ^ { t }$ from a finite action space ${ \mathcal { A } } ,$ producing reward

$$
R ^ { t } = r ( S ^ { t } , J ^ { t } , A ^ { t } ) \in [ 0 , \overline { { r } } ] ,
$$

and a resource-consumption vector

$$
C ^ { t } = c ( S ^ { t } , J ^ { t } , A ^ { t } ) \in [ 0 , \bar { c } ] ^ { m } .
$$

The deterministic reward and resource-consumption functions

$$
r : \mathcal { S } \times \mathcal { I } \times \mathcal { A } \to \mathbb { R } _ { + } , \qquad c : \mathcal { S } \times \mathcal { I } \times \mathcal { A } \to \mathbb { R } _ { + } ^ { m }
$$

are known. To ensure that the decision-making problem is feasible, there is a null action $a ^ { 0 } \in { \mathcal { A } }$ such that

$$
r ( s , j , a ^ { 0 } ) = 0 , \quad c ( s , j , a ^ { 0 } ) = 0 , \quad \forall s \in \mathcal { S } , j \in \mathcal { I } ,
$$

while every non-null action has a nonzero resource-consumption vector:

$$
a \neq a ^ { 0 } \implies c ( s , j , a ) \neq 0 \quad \forall ( s , j ) \in \mathcal { S } \times \mathcal { I } .
$$

The chosen action $A ^ { t }$ also induces a Markov state transition from $S ^ { t }$ to $S ^ { t + 1 }$ according to the known transition kernel

$$
S ^ { t + 1 } \sim P ( \cdot \mid S ^ { t } , A ^ { t } ) .
$$

In fact, we will show in Section 6.3 that, when the request prior is unknown, knowledge of the

optimal target state class is necessary to achieve good performance relative to the benchmark introduced below. The state transition is independent of the request. We assume that the initial state $S ^ { 1 }$ is fixed.

After the action in round t, the remaining resource inventory is updated to

$$
\boldsymbol { B } ^ { t + 1 } = \boldsymbol { B } ^ { t } - \boldsymbol { C } ^ { t } .
$$

The procedure stops after the first round in which any resource is depleted or after round $T _ { \ast }$ whichever comes first. Thus, the stopping time is

$$
\tau : = \operatorname* { m i n } \left\{ \operatorname* { i n f } \left\{ t \in [ T ] : \operatorname* { m i n } _ { i \in [ m ] } B _ { i } ^ { t + 1 } \leq 0 \right\} , T \right\} .\tag{1}
$$

Equivalently, the decision maker always chooses the null action after $\tau .$

Let $\mathcal { F } ^ { t }$ be the history generated by all observations and randomizations strictly before the arrival of request $J ^ { t } ,$ , together with the current state $S ^ { t }$ and inventory $B ^ { t }$ . A nonanticipating policy $\pi$ chooses action $A ^ { t }$ based only on $\mathcal { F } ^ { t }$ and $J ^ { t }$ . Its value is the expected total reward, including the reward in the final round:

$$
V _ { T } ^ { \pi } ( T \rho , s ^ { 1 } ) : = \mathbb { E } ^ { \pi } [ \mathcal { R } _ { T } ^ { \pi } ] , \qquad \mathcal { R } _ { T } ^ { \pi } : = \sum _ { t = 1 } ^ { \tau } R ^ { t } .
$$

The optimal nonanticipating value, that is, the optimal total expected reward among all nonanticipating policies, is

$$
V _ { T } ^ { * } ( T \rho , s ^ { 1 } ) : = \operatorname* { s u p } _ { \mathrm { \tiny ~ n o n a n t i c i p a t i n g ~ } \pi } V _ { T } ^ { \pi } ( T \rho , s ^ { 1 } ) .
$$

## 2.1 The transient fluid LP upper bound

We formulate an upper-bound benchmark for a generic subproblem with $H \geq 1$ remaining rounds, current inventory $B \in \mathbb { R } _ { + } ^ { m }$ , and current state $s ^ { 0 } \in S$

For $s \in \mathcal S , j \in \mathcal I$ , and $a \in { \mathcal { A } }$ , let

$X _ { s j a } \ge 0$ be the expected number of nonterminal rounds with state s, request $j ,$ and action $a ;$

$Y _ { s j a } \ge 0$ be the probability that the terminal round has state s, request $j ,$ , and action $a ;$

$Q _ { s } \geq 0$ be the expected number of active rounds spent in state s, including the terminal round.

(TF2)

Define $\Phi _ { H } ( B , s ^ { 0 } )$ as the optimal value of the following LP:

$$
\operatorname* { m a x } _ { X , Y , Q \geq 0 } \quad \sum _ { s \in S } \sum _ { j \in \mathcal { I } } \sum _ { a \in \mathcal { A } } r ( s , j , a ) \big ( X _ { s j a } + Y _ { s j a } \big )
$$

subject to $\sum _ { a \in \mathcal { A } } \left( X _ { s j a } + Y _ { s j a } \right) = p _ { j } Q _ { s } ,$

$$
s \in { \mathcal { S } } , \ j \in { \mathcal { I } } ,\tag{TF1}
$$

$$
Q _ { s ^ { \prime } } = { \bf 1 } \{ s ^ { \prime } = s ^ { 0 } \} + \sum _ { s \in \cal S } \sum _ { j \in { \mathcal I } } \sum _ { a \in { \mathcal A } } P ( s ^ { \prime } \mid s , a ) X _ { s j a } , ~ s ^ { \prime } \in \cal S ,\tag{2}
$$

$$
\sum _ { s \in S } Q _ { s } \le H , \qquad ( \mathrm { T F 3 } )
$$

$$
\sum _ { s \in S } \sum _ { j \in \mathcal { I } } \sum _ { a \in \mathcal { A } } c _ { i } ( s , j , a ) X _ { s j a } \leq B _ { i } ,
$$

$$
i \in [ m ] .\tag{TF4}
$$

The terminal variables Y contribute reward without consuming fluid capacity. This relaxation accounts for the convention that only the resource consumption incurred strictly before the terminal action must fit within the initial inventory.

The following lemma shows that this LP gives an upper bound for every subproblem.

Lemma 2.1. For every $H \geq 1 , B \in \mathbb { R } _ { + } ^ { m }$ , and $s ^ { 0 } \in S$

$$
V _ { H } ^ { * } ( B , s ^ { 0 } ) \leq \Phi _ { H } ( B , s ^ { 0 } ) .
$$

In particular, we have $V _ { T } ^ { \ast } ( T \rho , s ^ { 1 } ) \leq \Phi _ { T } ( T \rho , s ^ { 1 } )$

Accordingly, we define the regret of a nonanticipating policy $\pi$ as the gap between the value of the transient fluid LP and the policy’s value:

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \pi } ( T \rho , s ^ { 1 } ) .
$$

## 2.2 The stationary LP

Although the transient fluid LP gives an upper bound, it is dificult to use because of the asymmetry between the variables $\{ X _ { s j a } \}$ and $\{ Y _ { s j a } \}$ . We next define the stationary LP for any per-period capacity vector $\beta \in \mathbb { R } _ { + } ^ { m }$ . This LP extends the fluid LP widely studied in the online resource allocation literature (e.g., Jasin and Kumar (2012); Jasin (2015); Bumpensanti and Wang (2020)) to the Markov-state setting. In this LP, $q _ { s }$ is the stationary probability of state s, while $z _ { s j a }$ is the joint stationary probability of state s, request $j ,$ and action a, or equivalently, the per-round stationary flow through that state-request-action triple.

$$
{ \begin{array} { r l r l } { G ( \beta ) : = { \underset { z , q \geq 0 } { \operatorname* { m a x } } } } & { \displaystyle \sum _ { s \in \mathcal { S } } \displaystyle \sum _ { j \in \mathcal { I } } \sum _ { \alpha \in \mathcal { A } } { r } ( s , j , a ) z _ { s j a } } \\ { { \mathrm { s u b j e c t ~ t o } } } & { \displaystyle \sum _ { \alpha \in \mathcal { A } } z _ { s j a } = p _ { j } q _ { s } , } & { s \in { \mathcal { S } } , ~ j \in \mathcal { I } , } \\ & { } & { \displaystyle { q _ { s } } = \sum _ { s \in \mathcal { S } } \displaystyle \sum _ { j \in \mathcal { I } } \sum _ { \alpha \in \mathcal { A } } P ( s ^ { \prime } \mid s , a ) z _ { s j a } , ~ s ^ { \prime } \in { \mathcal { S } } , ~ } & { { \mathrm { ( S F 2 ) } } } \\ & { } & { \displaystyle \sum _ { s \in \mathcal { S } } q _ { s } = 1 , ~ { \mathrm { ( S F 3 ) } } } \\ & { } & { \displaystyle \sum _ { s \in \mathcal { S } } \displaystyle \sum _ { j \in \mathcal { I } } \sum _ { \alpha \in \mathcal { A } } c _ { i } ( s , j , a ) z _ { s j a } \leq \beta _ { i } , ~ } & { i \in [ m ] , ~ } & { { \mathrm { ( S F 4 ) } } } \end{array} }\tag{SF1}
$$

(3)

First, the LP above is always feasible: choose the null action and any stationary distribution of the transition matrix induced by that action. Second, an optimal solution $( z ( \beta ) , q ( \beta ) )$ naturally induces the following randomized policy at every state s for which $q _ { s } ( \beta ) > 0 \colon$

$$
\pi _ { \beta } ( a \mid s , j ) : = \frac { z _ { s j a } ( \beta ) } { p _ { j } q _ { s } ( \beta ) } .\tag{4}
$$

Constraint (SF1) ensures that these probabilities sum to one. Furthermore, the induced Markov state transition matrix is

$$
K _ { \beta } ( s , s ^ { \prime } ) : = \sum _ { j \in \mathcal { I } } p _ { j } \sum _ { a \in \mathcal { A } } \pi _ { \beta } ( a \mid s , j ) P ( s ^ { \prime } \mid s , a ) .\tag{5}
$$

Constraint (SF2) implies that $q ( \beta )$ is stationary for $K _ { \beta }$

However, this solution does not directly determine an action rule at a state with $q _ { s } ( \beta ) = 0$ . We therefore consider the natural extension

$$
\pi _ { \beta } ( a ^ { 0 } \mid s , j ) = 1 , \qquad q _ { s } ( \beta ) = 0 .\tag{6}
$$

This extension always chooses the null action in this case. We call it the null-completion convention. This is actually a very natural convention—when the process is in a state that should not be entered, the most natural choice is to remain inactive, consuming no resources and gaining no reward until the process is driven out of the state. Our algorithms below use this convention at states assigned zero stationary probability.

Finally, we show that $\Phi _ { T } ( T \rho , s ^ { 1 } )$ exceeds $T G ( \rho )$ by at most a constant. Therefore, up to an additive constant, we can use the more convenient $T G ( \rho )$ as a proxy for the transient fluid LP upper bound in the subsequent analysis, since our algorithms also rely on this homogeneous core LP.

Lemma 2.2. For every $T _ { i }$

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) \leq T G ( \rho ) + O ( 1 ) .\tag{7}
$$

## 2.3 Why not the hindsight optimum?

Previous work on online resource allocation has obtained universal constant-regret results by using the hindsight optimum as the benchmark (Bumpensanti and Wang, 2020; Li et al., 2026). In the state-free setting, the hindsight optimum is the expected optimal reward attainable by a policy, possibly an anticipatory one, that knows the realization of all future requests. This benchmark is naturally an upper bound on the value of any nonanticipating policy. In the state-free setting, it is also tighter than the stationary LP: the gap between $T G ( \rho )$ and the hindsight benchmark can be as large as $\Theta ( { \sqrt { T } } )$ when $G ( \rho )$ has a degenerate optimum, as discussed in Section 3. As a direct consequence, adopting $T G ( \rho )$ as the benchmark inevitably produces an $\Omega ( { \sqrt { T } } )$ regret lower bound in certain settings.

So why not use the hindsight optimum in our problem? The main reason is that, in the state-free case, the order in which requests arrive does not afect the hindsight optimum as long as the number of arrivals of each request type is fixed. Thus, for each realization, the only decision variables are the numbers of times that each action is taken for each request type. This property does not hold in our Markov-state setting: the order of requests and its interaction with the chosen actions afect the random state transitions and, consequently, resource consumption and reward collection.

Consequently, the randomness of the state transitions must be incorporated into the definition of the hindsight optimum, but the resulting benchmark is too strong. Chen et al. (2024b) show that such an “optimum” can exceed the value of the optimal nonanticipating policy by $\Theta ( T )$ . Therefore, we use the transient fluid LP as a conservative benchmark and the stationary LP as its analytically convenient proxy, and we regard finding a tighter benchmark as an important direction for future research.

## 3 Regularity assumptions

This section states the conditions that govern the benchmarks introduced above and the algorithms analyzed below. These conditions concern the stationary LP at the nominal per-period capacity vector $\rho .$

Throughout this work, we assume that the stationary LP has a unique optimum.

Assumption 3.1 (Unique optimality). The stationary $L P G ( \rho )$ has a unique optimal solution.

Our next condition concerns primal nondegeneracy in the fixed standard-form representation of the stationary LP.

Assumption 3.2 (Nondegeneracy). The optimal solution of the stationary $L P G ( \rho )$ is nondegenerate.

Together, Assumptions 3.1 and 3.2 imply local stability. Suppose they hold, and let the unique optimal solution be $( z ^ { * } , q ^ { * } )$ . Define its resource slacks, which we call the nominal slacks, by

$$
\sigma _ { i } ^ { * } : = \rho _ { i } - \sum _ { s , j , a } c _ { i } ( s , j , a ) z _ { s j a } ^ { * } , \qquad i \in [ m ] ,
$$

and let

$$
\begin{array} { r } { I ^ { * } : = \{ i : \sigma _ { i } ^ { * } = 0 \} , \qquad \mathcal { R } ^ { * } : = \{ s : q _ { s } ^ { * } > 0 \} , \qquad \mathcal { E } ^ { * } : = \{ ( s , j , a ) : z _ { s j a } ^ { * } > 0 \} . } \end{array}\tag{8}
$$

Thus $I ^ { * }$ is the set of nominally binding resources, $\mathcal { R } ^ { * }$ is the optimal target state class (equivalently, the active-state set of the nominal optimum), and $\mathcal { E } ^ { * }$ is the set of active decisions.

To state the stability properties, we first define a one-sided neighborhood for any scalar $r > 0 \AA$

$$
\mathcal { N } _ { r } : = \left. \beta \in \mathbb { R } _ { + } ^ { m } : \begin{array} { l } { | \beta _ { i } - \rho _ { i } | < r , i \in I ^ { * } ; } \\ { \beta _ { i } > \rho _ { i } - r , i \notin I ^ { * } } \end{array} \right. .
$$

The corresponding one-sided distance from a capacity ratio $\beta$ to $\rho$ is

$$
\Delta ( \beta ) : = \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { i \in I ^ { * } } | \beta _ { i } - \rho _ { i } | , \operatorname* { m a x } _ { i \notin I ^ { * } } ( \rho _ { i } - \beta _ { i } ) _ { + } \right\} .\tag{9}
$$

Here and below, a maximum over an empty resource index set is interpreted as zero. Equivalently, $\beta \in \mathcal N _ { r }$ if and only if $\Delta ( \beta ) < r$ . For the binding coordinates, define

$$
\mathcal { B } _ { r } : = \left\{ \theta \in \mathbb { R } ^ { | I ^ { * } | } : | \theta _ { i } - \rho _ { i } | \leq r ~ \mathrm { f o r } ~ \mathrm { e v e r y } ~ i \in I ^ { * } \right\} .
$$

We have the following result.

Proposition 3.3. Under Assumptions 3.1 and 3.2, there exist constants $\delta , \eta > 0$ , with $\delta \ <$ min $\cdot i \in [ m ] ^ { \rho _ { i } }$ , such that, for every $\beta \in \mathcal { N } _ { 3 \delta }$ ，

(i) the unique optimal solution $( z ( \beta ) , q ( \beta ) )$ of $G ( \beta )$ has binding-resource set $I ^ { * }$ , active-state set $\mathcal { R } ^ { * }$ , and active-decision set $\mathcal { E } ^ { * }$ ;

(ii) every nonzero coordinate of $( z ( \beta ) , q ( \beta ) )$ and every non-binding resource slack of $G ( \beta )$ is at least η;

(iii) every coordinate $o f z ( \beta )$ and $q ( \beta )$ is afine in $\beta _ { I ^ { * } }$ and does not depend on $\beta _ { i } ~ f o r ~ i \notin I ^ { * }$

Another important condition specific to the Markov-state setting is the following.

Assumption 3.4 (Irreducibility of the optimal induced kernel). The optimal induced kernel, namely, the state transition probability matrix on $\mathcal { R } ^ { * }$ induced by the optimal policy, is defined by

$$
K ^ { * } ( s , s ^ { \prime } ) : = \sum _ { j } p _ { j } \sum _ { a } \pi _ { \rho } ( a \mid s , j ) P ( s ^ { \prime } \mid s , a ) , \qquad s , s ^ { \prime } \in \mathcal { R } ^ { * } .
$$

This matrix is irreducible on $\mathcal { R } ^ { * }$

This assumption concerns movement within the optimal target state class $\mathcal { R } ^ { * }$ : once the process is in $\mathcal { R } ^ { * }$ , every state in the class communicates with every other under $\pi _ { \rho }$ . We also have the following local properties.

Proposition 3.5. Under Assumptions 3.1, 3.2, and 3.4, there exists a constant $\delta > 0$ , with $\delta < \mathrm { m i n } _ { i \in [ m ] } \rho _ { i }$ , such that the following hold for every $\beta \in \mathcal { N } _ { 3 \delta }$

(i) $\mathcal { R } ^ { * }$ is closed under $K _ { \beta }$ .

(ii) Restricted to $\mathcal { R } ^ { * } , q ( \beta ) , \pi _ { \beta }$ , and $K _ { \beta }$ , viewed as functions of $\beta _ { I ^ { * } }$ , are uniformly bounded and Lipschitz on the closed box $B _ { 3 \delta }$

(iii) $K _ { \beta }$ is irreducible on $\mathcal { R } ^ { * }$

The following result shows that, under nondegeneracy, the optimal target state class is the entire state space; hence, the optimal induced kernel is a transition matrix on the entire state space.

Proposition 3.6. Under Assumption 3.2,

$$
q _ { s } ^ { * } > 0 , \forall s \in \mathcal { S } ; \quad \mathcal { R } ^ { * } = \mathcal { S } .
$$

Thus the initial state $s ^ { 1 }$ belongs to $\mathcal { R } ^ { * }$ under nondegeneracy, so entrance is immediate. However, under degeneracy, reachability of $\mathcal { R } ^ { * }$ is not automatic, so we impose it separately.

Assumption 3.7 (Constant-cost entrance policy). There exists a fixed nonanticipating entrance policy $\pi ^ { \mathrm { { e n t } } }$ , independent of T, such that, for the corresponding infinite-resource process,

$$
\tau ^ { \mathrm { e n t } } : = \operatorname* { i n f } \{ t \geq 1 : S ^ { t } \in \mathcal { R } ^ { * } \}
$$

is almost surely finite and

$$
\mathbb { E } [ \tau ^ { \mathrm { e n t } } ] = O ( 1 ) , \quad \mathbb { E } \left[ \sum _ { { t = 1 } } ^ { \tau ^ { \mathrm { e n t } } - 1 } \| c ( S ^ { t } , J ^ { t } , A ^ { t } ) \| _ { 1 } \right] = O ( 1 ) .\tag{10}
$$

If $s ^ { 1 } \in \mathcal { R } ^ { * }$ , we take $\tau ^ { \mathrm { e n t } } = 1$ and the entrance phase is empty.

Assumption 3.7 is automatic under nondegeneracy by Proposition 3.6: the entrance phase is empty. In the degenerate case, we focus on entrance-augmented algorithms. They first execute the policy $\pi ^ { \mathrm { e n t } }$ specified by Assumption 3.7 until the beginning of the first round $\tau ^ { \mathrm { e n t } }$ for which $S ^ { \tau ^ { \mathrm { e n t } } } \in \mathcal { R } ^ { * }$ . If the horizon ends or a resource is exhausted before such a round, the process terminates according to the original model.

## 4 Frequent and infrequent re-solving under regularity

In this section, we study two algorithms that re-solve the stationary LP at specified rounds using the average remaining inventory. As their names suggest, the main diference between them is their re-solving frequency. Both algorithms have been studied extensively in the online resource allocation literature and can achieve constant-regret guarantees under regularity assumptions (Assumptions 3.1 and 3.2) (Bumpensanti and Wang, 2020; Chen et al., 2024a). We extend these results to the Markov-state setting by further imposing the irreducibility assumption (Assumption 3.4).

Algorithm 1 Frequent re-solving (FR)   
1: Run $\pi ^ { \mathrm { e n t } }$ until successful entrance at the beginning of round $\tau ^ { \mathrm { e n t } }$ , or until the process stops.   
2: if the process stops before successful entrance then   
3: return   
4: for $t = \tau ^ { \mathrm { e n t } } , \dots , T$ do   
5: Set $H ^ { t } \gets T - t + 1$ and $\beta ^ { t } \gets B ^ { t } / H ^ { t } .$   
6: Solve $G ( \beta ^ { t } )$ and select $( z ( \beta ^ { t } ) , q ( \beta ^ { t } ) ) \gets \mathfrak { S } ( \beta ^ { t } )$   
7: Observe request $J ^ { t }$ and the current state $S ^ { t }$   
8: if $q _ { S ^ { t } } ( \beta ^ { t } ) = 0$ then   
9: Set $A ^ { t }  a ^ { 0 }$   
10: else   
11: Draw $A ^ { t } \sim \pi _ { \beta ^ { t } } ( \cdot \mid S ^ { t } , J ^ { t } )$ using (4).   
12: Execute $A ^ { t } { \mathrm { . } }$ earn reward $R ^ { t } .$ , and incur consumption $C ^ { t } .$   
13: Set $B ^ { t + 1 }  B ^ { t } - C ^ { t }$ and generate $S ^ { t + 1 }$ according to $P ( \cdot \mid S ^ { t } , A ^ { t } )$   
14: if min $\cdot i \in [ m ] ^ { B _ { i } ^ { t + 1 } } \leq 0$ then   
15: return

## 4.1 Algorithms

We first specify how an LP optimizer is selected. Although $G ( \rho )$ has a unique optimizer under Assumption 3.1, the LPs evaluated at the perturbed capacity ratios encountered by FR and IR may have several optimizers. Fix once and for all a deterministic measurable selector

$$
{ \mathfrak { S } } : \beta \longmapsto x ( \beta ) = ( z ( \beta ) , q ( \beta ) )
$$

such that $x ( \beta )$ is an optimal solution of $G ( \beta )$ for every $\beta ;$ for example, one may use lexicographic tie-breaking with a fixed ordering of all LP variables. Both FR and IR use this same selector whenever they solve the stationary LP. Thus the same LP input always yields the same selected optimizer. The word “deterministic” applies only to this tie-breaking rule: after the optimizer is selected, the algorithms may still randomize their actions according to the induced policy in (4). We impose no continuity requirement on S.

The frequent re-solving (FR) algorithm is given in Algorithm 1. After entering the optimal target state class $\mathcal { R } ^ { * }$ , it solves the stationary LP using the current average remaining inventory in every round.

The infrequent re-solving algorithm (Algorithm 2) reduces the number of $\mathrm { L P }$ solves by holding each selected policy fixed until the next scheduled update. The algorithm is parameterized by $\alpha \in ( 1 / 2 , 1 )$ . Upon successful entry at the beginning of round $\tau ^ { \mathrm { e n t } }$ , let

$$
h _ { 0 } = H ^ { \mathrm { e n t } } : = T - \tau ^ { \mathrm { e n t } } + 1 .\tag{11}
$$

Define recursively

$$
h _ { k + 1 } : = \operatorname* { m a x } \left\{ 1 , \operatorname* { m i n } \left\{ \left\lceil h _ { k } ^ { \alpha } \right\rceil , h _ { k } - 1 \right\} \right\} \quad \mathrm { w h i l e } \ h _ { k } > 1 .\tag{12}
$$

Algorithm 2 Infrequent re-solving (IR)   
Require: $\alpha \in ( 1 / 2 , 1 )$   
1: Solve $G ( \rho )$ once and select $( z ( \rho ) , q ( \rho ) ) \gets \mathfrak { S } ( \rho )$   
2: Let $\pi _ { \rho }$ be the induced policy, with the null-completion convention.   
3: Run $\pi ^ { \mathrm { e n t } }$ until successful entrance at the beginning of round $\tau ^ { \mathrm { e n t } }$ , or until the process stops.   
4: if the process stops before successful entrance then   
5: return   
6: Construct $h _ { 0 } , \ldots , h _ { K }$ and $t _ { 0 } , \dots , t _ { K }$ using (11)–(13). Set $t _ { K + 1 } \gets T + 1 .$   
7: for $k = 0 , \ldots , K$ do   
8: if $k = 0$ then   
9: Set $\pi ^ { 0 }  \pi _ { \rho } .$ ▷ No post-entry solve.   
10: else   
11: Set $\beta ^ { k } \gets B ^ { t _ { k } } / h _ { k }$   
12: Solve $G ( \beta ^ { k } )$ and select $( z ( \beta ^ { k } ) , q ( \beta ^ { k } ) ) \gets \mathfrak { S } ( \beta ^ { k } )$   
13: Set $\pi ^ { k }  \pi _ { \beta ^ { k } }$ and apply the null-completion convention in (6).   
14: for $t = t _ { k } , \ldots , t _ { k + 1 } - 1$ do   
15: Observe request J<sup>t</sup> and the current state $S ^ { t } .$   
16: Draw $A ^ { t } \sim \pi ^ { k } ( \cdot \mid S ^ { t } , J ^ { t } )$   
17: Execute $A ^ { t } ,$ earn reward $R ^ { t } { \mathrm { . } }$ , and incur consumption $C ^ { t }$   
18: Set $B ^ { t + 1 }  B ^ { t } - C ^ { t }$ and generate $S ^ { t + 1 }$ according to $P ( \cdot \mid S ^ { t } , A ^ { t } )$   
19: if min $\boldsymbol { \mathrm { i } } \in [ m ] B _ { i } ^ { t + 1 } \leq 0$ then   
20: return

Let K be the first index with $h _ { K } = 1$ , and set

$$
t _ { k } = \tau ^ { \mathrm { e n t } } + h _ { 0 } - h _ { k } , \qquad k = 0 , 1 , \ldots , K .\tag{13}
$$

For $k < K$ , epoch k consists of rounds $t _ { k } , \ldots , t _ { k + 1 } - 1$ and has length $\Lambda _ { k } : = h _ { k } - h _ { k + 1 }$

The IR algorithm solves the stationary LP initially and re-solves it in rounds $t _ { 1 } , \dots , t _ { K } ;$ the solution associated with each epoch determines the policy used throughout that epoch. Note that the algorithm does not re-solve in round $t _ { 0 } = \tau ^ { \mathrm { e n t } }$ . It is straightforward to verify that the IR algorithm performs O(log log T) stationary-LP solves.

## 4.2 Performance under regularity

We now present constant-regret results for these two algorithms under regularity assumptions (Assumptions 3.1, 3.2, and 3.4). Note that under Assumption 3.2, entrance is immediate because $\mathcal { R } ^ { \ast } = \mathcal { S }$ , so Assumption 3.7 is not required.

Theorem 4.1 (Constant regret of frequent and infrequent re-solving under regularity). Suppose Assumptions 3.1, 3.2, and 3.4 hold. Then, for any fixed $\alpha \in ( 1 / 2 , 1 )$ 2

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { F R } } ( T \rho , s ^ { 1 } ) = O ( 1 ) , \quad \Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { I R } } ( T \rho , s ^ { 1 } ) = O ( 1 ) .
$$

Consequently,

$$
V _ { T } ^ { * } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { F R } } ( T \rho , s ^ { 1 } ) = O ( 1 ) , \quad V _ { T } ^ { * } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { I R } } ( T \rho , s ^ { 1 } ) = O ( 1 ) .
$$

## 5 Incomplete regularity

In this section, we separately examine how primal nondegeneracy and irreducibility of the optimal induced kernel afect the algorithms’ performance.

## 5.1 A unique degenerate optimum

Optimal policy: square-root lower bound. We start with the case in which the stationary LP $G ( \rho )$ has a unique but degenerate optimum. In this case, Bumpensanti and Wang (2020) established an $\Omega ( { \sqrt { T } } )$ gap between $T G ( \rho )$ and the optimal online value in a single-state setting. In the single-state setting, $T G ( \rho )$ is exactly the transient fluid LP, so the lower bound directly applies.

Proposition 5.1 (Single-state square-root lower bound under unique degeneracy, from Bumpensanti and Wang (2020)). There is a one-state, one-resource instance with a unique degenerate nominal stationary optimum such that

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { * } ( T \rho , s ^ { 1 } ) = T G ( \rho ) - V _ { T } ^ { * } ( T \rho , s ^ { 1 } ) = \Theta ( \sqrt { T } ) .
$$

IR: square-root upper bound. When the optimum is unique but degenerate, we can show that the IR algorithm achieves $O ( \sqrt { T } )$ regret, provided that the optimal induced kernel is irreducible and a constant-cost entrance policy exists.

Theorem 5.2 (Square-root regret of IR under a unique nominal optimum). Suppose Assumptions 3.1, 3.4, and 3.7 hold. Then, for any fixed $\alpha \in ( 1 / 2 , 1 )$ ,

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { I R } } ( T \rho , s ^ { 1 } ) = O ( \sqrt { T } ) .\tag{14}
$$

Consequently,

$$
V _ { T } ^ { \ast } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { I R } } ( T \rho , s ^ { 1 } ) = O ( \sqrt { T } ) .\tag{15}
$$

Together with Proposition 5.1, this result establishes worst-case regret of order $\Theta ( { \sqrt { T } } )$ for IR under Assumptions 3.1, 3.4, and 3.7.

FR: linear lower bound. For FR, however, the following result shows that the algorithm does not guarantee sublinear regret when the stationary LP has a unique but degenerate optimum, in contrast to the state-free setting.

Proposition 5.3. There is a one-resource, one-request-type instance with a unique degenerate nominal stationary optimum that satisfies Assumptions 3.4 and 3.7 such that

$$
V _ { T } ^ { * } ( T \rho , A ) - V _ { T } ^ { \mathrm { F R } } ( T \rho , A ) = \Omega ( T ) \quad a n d \quad \Phi _ { T } ( T \rho , A ) - V _ { T } ^ { \mathrm { F R } } ( T \rho , A ) = \Omega ( T ) .\tag{16}
$$

## 5.2 Failure of irreducibility of the optimal induced kernel

In this subsection, we further show that Assumption 3.4, that is, irreducibility of the optimal induced kernel, is necessary for any algorithm to attain a universal sublinear-regret guarantee. The intuition is that if the optimal induced kernel is reducible, then no algorithm can asymptotically implement the stationary solution: either the realized state-action frequencies do not match the stationary optimum, or some states in the optimal target state class cannot be reached.

Proposition 5.4. There is an instance satisfying Assumptions 3.1, 3.2, and 3.7, but not Assumption $\ 3 . 4 \ i$ such that, for every $T \geq 2$ , every nonanticipating policy π satisfies

$$
T G ( \rho ) - V _ { T } ^ { \pi } ( T \rho , s ^ { 1 } ) \geq \frac { T } { 4 } , \quad \Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \pi } ( T \rho , s ^ { 1 } ) \geq \frac { T } { 4 } .\tag{17}
$$

## 6 Unknown request prior

In this section, we extend our analysis to the incomplete-information setting in which the request distribution $p$ is unknown at the outset and must be learned online. The resulting statistical estimation error afects control because the empirical prior enters the fluid $\operatorname { L P } ;$ under degeneracy, a small perturbation can change the selected action support. To make the request prior explicit, we write $G ^ { p ^ { \prime } } ( \beta )$ for the stationary $\mathrm { L P }$ in (3) with $p$ replaced by $p ^ { \prime }$ and $\Phi _ { H } ^ { p ^ { \prime } } ( B , s )$ for the corresponding transient LP in (2). In particular, $G = G ^ { p }$ and $\Phi _ { H } = \Phi _ { H } ^ { p }$

This section gives two complementary results. First, suppose that the optimal induced kernel satisfies Assumption 3.4 and that both the true optimal target state class $\mathcal { R } ^ { * }$ and an entrance policy satisfying Assumption 3.7 are supplied to the algorithm. We propose an infrequent empirica re-solving policy that adapts automatically to degeneracy: it has $O ( 1 )$ regret when the nominal optimum is unique and nondegenerate (Assumptions 3.1 and 3.2) and $O ( \sqrt { T } )$ regret under uniqueness alone, which allows degeneracy (Assumption 3.1). The algorithm also uses O(log log T) LP solves in either case. Second, if $\mathcal { R } ^ { * }$ itself is unknown, learning is impossible even under irreducibility of the optimal induced kernel (Assumption 3.4) and constant-cost entrance (Assumption 3.7): there are two priors satisfying both conditions and nominal uniqueness such that, for every horizon and every prior-uninformed policy, the maximum regret across the two instances is bounded below by a positive constant times $T - 1$ . This lower bound highlights the necessity of knowing the optimal target state class $\mathcal { R } ^ { * }$ in our problem.

## 6.1 Terminology and preliminaries

We first introduce some terminology and notation for our algorithm. Throughout this subsection, we suppose that the optimal target state class $\mathcal { R } ^ { * }$ is known. In fact, we will show that this condition is necessary for any policy under the unknown request prior to attain sublinear regret.

Safe policy. For every $s \in \mathcal { R } ^ { * }$ and $j \in \mathcal { I }$ , fix a safe action $a ^ { \mathrm { s a f e } } ( s , j )$ satisfying

$$
P ( { \mathcal { R } } ^ { * } \mid s , a ^ { \mathrm { s a f e } } ( s , j ) ) = 1 .
$$

Such an action exists: nominal request balance and closure of $\mathcal { R } ^ { * }$ imply that at least one nominally used action has this property. Because $P$ and $\mathcal { R } ^ { * }$ are known, a safe action can be selected without knowing $p .$ Let $\pi ^ { \mathrm { s a f e } }$ denote the deterministic policy that selects $a ^ { \mathrm { s a f e } } ( s , j )$

Empirical estimation. Like FR and IR, the following algorithm first executes the entrance policy $\pi ^ { \mathrm { { e n t } } }$ . Suppose that the entrance policy succeeds at the beginning of round $\tau ^ { \mathrm { e n t } }$ , and let $H ^ { \mathrm { e n t } }$ denote the number of remaining rounds. A re-solving step taken at the beginning of round $\tau ^ { \mathrm { e n t } } + n$ uses the following Laplace-smoothed estimate:

$$
\widehat { p } ^ { n } ( j ) : = \frac { N _ { j } ( n ) + 1 } { n + | \mathcal { I } | } , \qquad N _ { j } ( n ) : = \sum _ { t = \tau ^ { \mathrm { e n t } } } ^ { \tau ^ { \mathrm { e n t } } + n - 1 } \mathbf { 1 } \{ J ^ { t } = j \} , \qquad n \geq 0 .\tag{18}
$$

In the above, $N _ { j } ( n )$ is the number of times that request $j$ has arrived up to round $\tau ^ { \mathrm { e n t } } + n$ (not including that round) since entrance.

Induced and core-projected policies. For a prior $p ^ { \prime }$ and capacity ratio $\beta ,$ let $G ^ { p ^ { \prime } , \mathcal { R } ^ { * } } ( \beta )$ denote the stationary LP in (3) with $p$ replaced by $p ^ { \prime }$ , subject to the following additional restrictions:

$$
q _ { s } = 0 , \qquad z _ { s j a } = 0 , \qquad s \notin \mathcal { R } ^ { * } .
$$

If $x = ( z , q )$ solves a stationary LP with prior $p ^ { \prime } .$ , define its induced policy wherever the denominator is positive by

$$
\pi _ { x } ^ { p ^ { \prime } } ( a \mid s , j ) : = \frac { z _ { s j a } } { p _ { j } ^ { \prime } q _ { s } } = \frac { z _ { s j a } } { \sum _ { a ^ { \prime } } z _ { s j a ^ { \prime } } } .\tag{19}
$$

Next, define the set of actions that keep the known optimal target state class closed:

$$
\begin{array} { r } { \mathcal { A } _ { \mathcal { R } ^ { * } } ( s ) : = \{ a \in \mathcal { A } : P ( \mathcal { R } ^ { * } \mid s , a ) = 1 \} , \qquad s \in \mathcal { R } ^ { * } . } \end{array}
$$

For any stationary LP solution $x = ( z , q )$ , define its core projection, whenever the denominator below is positive, by

$$
\Pi _ { \mathcal { R } ^ { * } } ( x ) ( a \mid s , j ) : = \frac { z _ { s j a } \mathbf { 1 } \{ a \in \mathcal { A } _ { \mathcal { R } ^ { * } } ( s ) \} } { \sum _ { a ^ { \prime } \in \mathcal { A } _ { \mathcal { R } ^ { * } } ( s ) } z _ { s j a ^ { \prime } } } , \qquad s \in \mathcal { R } ^ { * } , \ j \in \mathcal { I } .\tag{20}
$$

Thus, the projected policy can be implemented entirely within the supplied optimal target state class $\mathcal { R } ^ { * }$

Three-phase re-solving schedule. The re-solving schedule is divided into three phases. In the early phase, we fix any constant $n _ { 0 } \geq 4$ and define

$$
\tau ^ { \mathrm { { l r n } } } : = \left\lceil ( \log H ^ { \mathrm { { e n t } } } ) ^ { 2 } \right\rceil .
$$

Starting with $r _ { 0 } = n _ { 0 }$ , define recursively

$$
r _ { k + 1 } : = \operatorname* { m i n } \{ \lceil r _ { k } ^ { 2 } \rceil , \tau ^ { \mathrm { l r n } } \} \quad \mathrm { w h i l e } \ r _ { k } < \tau ^ { \mathrm { l r n } } .\tag{21}
$$

When $n _ { 0 } \leq \tau ^ { \mathrm { { l r n } } }$ , let E be the first index such that $r _ { E } = \tau ^ { \mathrm { l r n } }$

In the middle phase, let

$$
\tau ^ { \mathrm { h a l } } : = \left\lfloor \frac { H ^ { \mathrm { e n t } } } { 2 } \right\rfloor .
$$

When $r _ { 0 } < \tau ^ { \mathrm { l r n } } < \tau ^ { \mathrm { h a l } }$ , set

$$
K : = \operatorname* { m i n } \left\{ \tau ^ { \mathrm { h a l } } - \tau ^ { \mathrm { l r n } } , \operatorname* { m a x } \left\{ 1 , \left\lceil \log _ { 2 } \log \frac { \tau ^ { \mathrm { h a l } } } { \tau ^ { \mathrm { l r n } } } \right\rceil + 2 \right\} \right\} ,\tag{22}
$$

and choose the integer re-solving times as an optimizer of

$$
\operatorname* { m i n } _ { \tau ^ { \mathrm { l r n } } = t _ { 0 } < t _ { 1 } < \cdots < t _ { K } = \tau ^ { \mathrm { h a l } } } \sum _ { k = 0 } ^ { K - 1 } \frac { t _ { k + 1 } - t _ { k } } { \sqrt { t _ { k } } } ,\tag{23}
$$

where the minimization is over integer partitions.

For the late phase, set

$$
h _ { 0 } = { H ^ { \mathrm { e n t } } - \tau ^ { \mathrm { h a l } } } = \left\lceil \frac { H ^ { \mathrm { e n t } } } { 2 } \right\rceil , \qquad h _ { k + 1 } = \operatorname* { m a x } \left\{ 1 , \operatorname* { m i n } \left\{ \left\lceil h _ { k } ^ { \alpha } \right\rceil , h _ { k } - 1 \right\} \right\} ,\tag{24}
$$

Apply this recursion while $h _ { k } > 1$ . Let $L$ be the first index with $h _ { L } = 1$ , and append the terminal endpoint $h _ { L + 1 } = 0$

## 6.2 U-shaped infrequent re-solving

Algorithm 3 presents our policy. As described above, it executes three re-solving phases. The first two phases allow the estimated prior to guide the decision rule early, and the final phase resembles the original infrequent re-solving scheme. UIR-UP requires O(log log T) LP solves.

Our main result on the UIR-UP policy is the following.

Theorem 6.1 (Order-optimal regret of UIR-UP). Suppose Assumptions 3.1, 3.4, and 3.7 hold. Then UIR-UP satisfies

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { U I R - U P } } ( T \rho , s ^ { 1 } ) = O ( \sqrt { T } ) .\tag{25}
$$

$H ,$ in addition, Assumption 3.2 holds, then UIR-UP satisfies

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \mathrm { U I R - U P } } ( T \rho , s ^ { 1 } ) = O ( 1 ) .\tag{26}
$$

```latex
Algorithm 3 U-shaped infrequent re-solving with an unknown prior (UIR-UP)
Require: $\mathcal { R } ^ { * } , n _ { 0 } \geq 4 ,$ and $\alpha \in ( 1 / 2 , 1 )$
1: Run $\pi ^ { \mathrm { { e n t } } }$ until entrance occurs at the beginning of round $\tau ^ { \mathrm { e n t } }$ , or until the process stops.
2: if the process stops before entrance then
3: return
4: Compute $\tau ^ { \mathrm { { l r n } } }$ and $\tau ^ { \mathrm { h a l } } .$
5: if $n _ { 0 } < \tau ^ { \mathrm { l r n } } < \tau ^ { \mathrm { h a l } }$ does not hold then
6: Execute $\pi ^ { \mathrm { s a f e } }$ during rounds $\tau ^ { \mathrm { e n t } } , \dots , T .$
7: return
8: Execute $\pi ^ { \mathrm { s a f e } }$ during rounds $\tau ^ { \mathrm { e n t } } , \dots , \tau ^ { \mathrm { e n t } } + n _ { 0 } - 1$ ▷ Accumulate samples.
9: Construct $r _ { 0 } < \cdots < r _ { E } = \tau ^ { \mathrm { l r n } }$ using (21). ▷ Phase 1: infrequent re-solving.
10: for $k = 0 , \ldots , E - 1$ do
11: Set $t \gets \tau ^ { \mathrm { e n t } } + r _ { k }$ and solve $G ^ { \widehat { p ^ { r } k } , \mathcal { R } ^ { * } } ( B ^ { t } / ( H ^ { \mathrm { e n t } } - r _ { k } ) )$ and select $x ^ { k }$ if the LP is feasible.
12: if the LP is infeasible or $q _ { S ^ { t } } ^ { k } = 0$ then
13: Set $\pi ^ { k }  \pi ^ { \mathrm { s a f e } }$
14: else
15: Set $\pi ^ { k }  \pi _ { x ^ { k } } ^ { \widehat { p } ^ { r _ { k } } }$ using (19), and use $\pi ^ { \mathrm { s a f e } }$ wherever the denominator in that formula is zero.
16: Execute $\pi ^ { k }$ during rounds $\tau ^ { \mathrm { e n t } } + r _ { k } , \dots , \tau ^ { \mathrm { e n t } } + r _ { k + 1 } - 1$
17: Choose $\tau ^ { \mathrm { l r n } } = t _ { 0 } < \cdots < t _ { K } = \tau ^ { \mathrm { h a l } }$ using (22)–(23). ▷ Phase 2: infrequent re-solving.
18: for $k = 0 , \ldots , K - 1$ do
19: Set $t \gets \tau ^ { \mathrm { e n t } } + t _ { k }$ and solve $G ^ { \widehat { p } ^ { t _ { k } } } ( B ^ { t } / ( H ^ { \mathrm { e n t } } - t _ { k } ) )$ and select $x ^ { k }$
20: if $\Pi _ { { \mathcal { R } } ^ { * } } ( x ^ { k } )$ is defined for every $( s , j ) \in \mathcal { R } ^ { * } \times \mathcal { I }$ then
21: Set $\begin{array} { r } { \dot { \pi } ^ { k }  \Pi _ { \mathcal { R } ^ { * } } ( x ^ { k } ) } \end{array}$
22: else
23: Set $\pi ^ { k }  \pi ^ { \mathrm { s a f e } } .$
24: Execute $\pi ^ { k }$ during rounds $\tau ^ { \mathrm { e n t } } + t _ { k } , \ldots , \tau ^ { \mathrm { e n t } } + t _ { k + 1 } - 1 .$
25: Construct $h _ { 0 } , \ldots , h _ { L + 1 }$ using (24). ▷ Phase 3: infrequent re-solving.
26: for $k = 0 , \ldots , L$ do
27: Set $t  T + 1 - h _ { k }$
28: Solve $G ^ { \widehat { p } ^ { H ^ { \mathrm { e n t } } } - h _ { k } } \left( B ^ { t } / h _ { k } \right)$ and select $x ^ { k } .$
29: if $\Pi _ { { \mathcal { R } } ^ { * } } ( x ^ { k } )$ is defined for every $( s , j ) \in \mathcal { R } ^ { * } \times \mathcal { I }$ then
30: Set $\begin{array} { r } { \dot { \pi } ^ { k }  \Pi _ { \mathcal { R } ^ { * } } ( x ^ { k } ) } \end{array}$
31: else
32: Set $\pi ^ { k }  \pi ^ { \mathrm { s a f e } } .$
33: Execute $\pi ^ { k }$ during rounds $T + 1 - h _ { k } , \dots , T - h _ { k + 1 }$
```

## 6.3 Unknown optimal target state class $\mathcal { R } ^ { \ast }$ : a linear information-theoretic regret lower bound

Finally, the supplied optimal target state class and entrance policy play a substantive role in our problem. If the optimal target state class $\mathcal { R } ^ { * }$ depends on the unknown prior and the process must irreversibly enter a class before enough requests can be observed, even sublinear regret is impossible.

Proposition 6.2. Fix any constant $\varepsilon \in ( 0 , 1 )$ . There are two fixed request priors $p ^ { + }$ and $p ^ { - }$ such that, under each prior, the nominal stationary fluid LP has a unique degenerate optimum, Assumption 3.4 holds, and Assumption 3.7 holds with a one-round entrance policy. For every integer horizon $T \geq 1$ and every policy π that knows neither the true prior nor its optimal target state class,

$$
\operatorname* { m a x } _ { p ^ { \prime } \in \{ p ^ { + } , p ^ { - } \} } \left\{ V _ { T } ^ { \ast , p ^ { \prime } } ( T \rho , s ^ { 1 } ) - V _ { T } ^ { \pi , p ^ { \prime } } ( T \rho , s ^ { 1 } ) \right\} \geq \frac { 1 - \varepsilon } { 8 } ( T - 1 ) .
$$

Table 3: Algorithms used in the experiments.
<table><tr><td>Algorithm</td><td>Prior</td><td>Re-solving</td><td>Description</td></tr><tr><td>FR</td><td>known</td><td>Θ(T)</td><td>Algorithm 1.</td></tr><tr><td>IR</td><td>known</td><td>Θ(log log T)</td><td>Algorithm 2.</td></tr><tr><td>FRT</td><td>known</td><td>Θ(T)</td><td>FR with thresholding.</td></tr><tr><td>IRT</td><td>known</td><td>Θ(log log T)</td><td>IR with thresholding.</td></tr><tr><td>FR-UP</td><td>unknown</td><td>Θ(T)</td><td>FR with the request prior replaced by  ${ \widehat { p } } ^ { n }$ </td></tr><tr><td>UIR-UP</td><td>unknown</td><td>Θ(log log T), three phases</td><td>Algorithm 3.</td></tr><tr><td>FRT-UP</td><td>unknown</td><td>Θ(T)</td><td>FR-UP with thresholding.</td></tr><tr><td>UIRT-UP</td><td>unknown</td><td>Θ(log log T), three phases</td><td>UIR-UP with thresholding,</td></tr></table>

## 7 Numerical experiments

In this section, we study how the algorithms analyzed in this work perform on a concrete problem instance as the capacity ratio of its first resource crosses a boundary at which the unique optimum becomes degenerate. We also examine re-solving policies with thresholding modifications inspired by Bumpensanti and Wang (2020); Li et al. (2026). We start by introducing the algorithms we consider.

## 7.1 Algorithms

Table 3 summarizes the algorithms examined in this section. We provide further details below.

Infrequent re-solving schedules. All infrequent re-solving algorithms use $\alpha = 0 . 7 ;$ for the unknown-prior algorithms, this parameter applies in phase 3.

Empirical prior estimation. All algorithms in the unknown-prior setting use the Laplacesmoothed estimate after n requests have been observed:

$$
\widehat { p } ^ { n } ( j ) = \frac { N _ { j } ( n ) + 1 } { n + | \mathcal { I } | } .
$$

Sparse-schedule implementation. For UIR-UP and UIRT-UP, the sparse learning phase uses the explicit rounded construction in the proof of Theorem E.2. Specifically, when $\tau ^ { \mathrm { h a l } } / \tau ^ { \mathrm { l r n } } \geq \exp ( 1 )$ , the implementation sets $K = \lfloor \log _ { 2 } \log ( \tau ^ { \mathrm { h a l } } / \tau ^ { \mathrm { l r n } } ) \rfloor + 2$ , generates the real-valued times in Equation (E.4), rounds them up, and removes repeated times. The resulting schedule $\tau ^ { \mathrm { l r n } } = t _ { 0 } < \cdot \cdot \cdot < t _ { K } = \tau ^ { \mathrm { h a l } }$ satisfies

$$
\sum _ { k = 0 } ^ { K - 1 } \frac { t _ { k + 1 } - t _ { k } } { \sqrt { t _ { k } } } \leq 9 \sqrt { \tau ^ { \mathrm { h a l } } } , \qquad K = O ( \log \log T ) .
$$

These are the sparse-phase scheduling properties used in the proof of Theorem 6.1; hence, this construction preserves the asymptotic guarantees for unthresholded UIR-UP.

Table 4: Primitive data of the experimental environment. Each entry in the last column is (next state; reward; resource-consumption vector).
<table><tr><td>State</td><td>Action</td><td>Outcome for request  $j$ </td></tr><tr><td>A</td><td>0</td><td>(B; 0; (0, 0))</td></tr><tr><td>A</td><td>u</td><td> $( B ; 1 ; ( 3 / 2 , 1 / 2 ) )$ </td></tr><tr><td>A</td><td>v</td><td> $( B ; 1 + b _ { j } ; ( 5 / 2 , 3 / 2 ) )$ </td></tr><tr><td>B</td><td>0</td><td>(A; 0; (0, 0))</td></tr><tr><td>B</td><td>u</td><td> $\left( A ; 1 ; \left( 1 / 2 , 3 / 2 \right) \right)$ </td></tr><tr><td>B</td><td>u</td><td> $( C ; 1 - d _ { j } ; ( 1 / 2 , 3 / 2 ) )$ </td></tr><tr><td>C</td><td>0</td><td>(C; 0; (0, 0))</td></tr><tr><td>C</td><td>u</td><td> $\left( A ; 3 / 4 ; ( 1 / 2 , 1 / 2 ) \right)$ </td></tr><tr><td>C</td><td>v</td><td>(C; 0; (1, 1))</td></tr></table>

Thresholding convention. At every re-solving step with $H > 1$ , FRT and FRT-UP set $\epsilon =$ $H ^ { 3 ( 1 - 2 \alpha ) / 8 } = H ^ { - 0 . 1 5 }$ . For IRT and UIRT-UP, let $h _ { k }$ be the remaining horizon at the start of epoch $k ,$ that is, at the k-th re-solving step overall. The threshold is fixed at

$$
\epsilon _ { k } = h _ { k } ^ { 3 ( 1 - 2 \alpha ) / 8 } = h _ { k } ^ { - 0 . 1 5 }
$$

for all rounds until the next re-solving step; the policy is thresholded once at the start of the epoch and then frozen. Thresholding is disabled only for the last solve, when the remaining horizon is one. For each state–request pair, if an action has probability greater than $1 - \epsilon ,$ the procedure deterministically selects an action with the largest probability; otherwise, it sets probabilities below ϵ to zero and renormalizes the remaining probabilities. If no mass remains, it again selects an action with the largest original probability. Ties are broken in favor of the first action in the fixed action order. The sufix “T” in FRT, IRT, FRT-UP, and UIRT-UP denotes this thresholding operation.

## 7.2 Environment

Instance. In our experimental instance, there are three states $\boldsymbol { \mathcal { S } } = \{ A , B , C \}$ and three request types with respective probabilities

$$
p = \left( \frac { 1 } { 2 } , \frac { 1 } { 3 } , \frac { 1 } { 6 } \right) .
$$

There are two resources and three actions $\{ 0 , u , v \}$ , where 0 is the null action. The initial state is A, and the per-period resource vector is $\rho = ( \rho _ { 1 } , 1 )$ . Let

$$
b = \left( { \frac { 1 } { 2 } } , { \frac { 1 } { 3 } } , { \frac { 1 } { 4 } } \right) , \qquad d = \left( { \frac { 1 } { 2 4 } } , { \frac { 1 } { 1 2 } } , { \frac { 1 } { 8 } } \right) .
$$

For request $j ,$ the deterministic next state, reward, and two-dimensional resource-consumption vector are given in Table 4.

The only primitive varied across experimental instances is $\rho _ { 1 }$ . In a neighborhood of $\rho _ { 1 } = 1$ , the

optimal homogeneous core has state masses and value

$$
\begin{array} { r } { \big ( q _ { A } , q _ { B } , q _ { C } ; G ( \rho ) \big ) = \left\{ \begin{array} { l l } { \big ( \rho _ { 1 } - \frac { 1 } { 2 } , \rho _ { 1 } - \frac { 1 } { 2 } , 2 ( 1 - \rho _ { 1 } ) ; 1 - \frac { 7 } { 1 2 } ( 1 - \rho _ { 1 } ) \big ) , } & { \rho _ { 1 } < 1 , } \\ { \big ( \frac { 1 } { 2 } , \frac { 1 } { 2 } , 0 ; 1 \big ) , } & { \rho _ { 1 } \ge 1 . } \end{array} \right. } \end{array}
$$

At $\rho _ { 1 } = 1$ , the optimizer is unique and degenerate, and the optimal target state class changes under an arbitrarily small downward perturbation. Nearby values of $\rho _ { 1 }$ above one also have unique degenerate optima with the same two-state optimal target state class. At ${ \rho _ { 1 } } = \mathrm { { 0 } } \mathrm { { . 9 9 5 } }$ , the optimizer is nondegenerate, its optimal target state class is $\{ A , B , C \}$ , and its optimal induced kernel is irreducible. For the unknown-prior infrequent policies, the supplied optimal target state class is $\{ A , B , C \}$ when $\rho _ { 1 } < 1$ and $\{ A , B \}$ when $\rho _ { 1 } \geq 1$ . In all cases, the entrance policy is empty.

Experimental settings. For every trial, we compute the empirical regret against the transient fluid LP benchmark,

$$
\Phi _ { T } ( T \rho , s ^ { 1 } ) - \sum _ { t = 1 } ^ { \tau } R ^ { t } ,
$$

where $\tau$ is the stopping time in (1). Reported bands are two-sided 95% Student’s t confidence intervals for the mean regret. The infrequent methods IR, IRT, UIR-UP, and UIRT-UP use 1,000 independent trials per configuration, whereas the more expensive round-by-round methods FR, FRT, FR-UP, and FRT-UP use 200.

We conduct horizon sweeps over

$$
T \in \{ 5 0 0 , 1 0 0 0 , 2 0 0 0 , 4 0 0 0 , 8 0 0 0 , 1 6 0 0 0 , 3 2 0 0 0 \}
$$

at both ${ \rho _ { 1 } } = \mathrm { { 0 } } \mathrm { { . 9 9 5 } }$ and $\rho _ { 1 } = 1$

The capacity-ratio sweep fixes $T = 2 0 0 0$ and uses

$$
\begin{array} { r } { \rho _ { 1 } \in \{ 0 . 9 9 5 , 0 . 9 9 7 , 0 . 9 9 8 , 0 . 9 9 9 , 1 , 1 . 0 0 1 , 1 . 0 0 2 , 1 . 0 0 3 , 1 . 0 0 5 \} . } \end{array}
$$

## 7.3 Known request prior

Figure 1 compares the horizon sweeps at two nearby capacity ratios: ${ \rho _ { 1 } } = \mathrm { { 0 } } \mathrm { { . 9 9 5 } }$ , where the unique optimum is nondegenerate, and $\rho _ { 1 } = 1$ , where it is degenerate. At the nondegenerate point, the mean regret of FR ranges from 75.185 to 107.435 over the tested horizons, while that of FRT remains between 6.136 and 6.982 over the full sweep. The behavior of unthresholded FR is consistent with Theorem 4.1, while thresholding further enhances performance, as reported by Bumpensanti and Wang (2020) in the state-free setting. At $T = 3 2 , 0 0 0$ , IR has mean regret 5.535 with a 95% confidence interval of [1.542, 9.527], and IRT has mean regret 6.009 with a 95% confidence interval of [5.445, 6.573]. In this case, the infrequent re-solving algorithms have a small advantage over the frequent re-solving algorithms.

This gap is much larger at the unique degenerate point, where the mean regret of FR increases from 233.852 at $T = 5 0 0$ to 15,285.072 at $T = 3 2 { , } 0 0 0$ . Its ratio of mean regret to horizon stays between approximately 0.468 and 0.521 over the sweep, illustrating the linear-loss phenomenon in Proposition 5.3. Thresholding removes most of this loss: the mean regret of FRT remains between 6.920 and 7.391. Nevertheless, infrequent re-solving policies still perform better: the mean regret of IR remains between 1.704 and 2.706. However, introducing thresholding appears to weaken IR: the mean regret of IRT ranges from 2.079 to 3.518. These results suggest that re-solving too frequently may not be optimal in online resource allocation with a Markov state, contrary to the findings for the state-free problem reported by Li et al. (2026).

![](images/d4fcf2a17f6c12fe54bd50a8bd293836e17e63dbb4d60b378b482b15ce62dd58.jpg)  
(a) Unique nondegenerate optimum, ${ \rho _ { 1 } } = \mathrm { { 0 } } \mathrm { { . 9 9 5 } }$

![](images/bbcf1c24618fc1eaf1d036f19441e8d6ecdf3db351d2449c494c7492a08eb15d.jpg)  
(b) Unique degenerate optimum, $\rho _ { 1 } = 1$  
Figure 1: Mean regret against the transient fluid LP with a known request prior.

Table 5: Mean regret against the transient fluid LP at $T = 3 2 , 0 0 0$ . Parentheses give the number of trials.
<table><tr><td></td><td colspan="2">Known prior</td><td colspan="2">Unknown prior</td></tr><tr><td>ρ1</td><td>FR (200)</td><td>IR (1,000)</td><td>FR-UP (200)</td><td>UIR-UP (1,000)</td></tr><tr><td>0.995</td><td>100.725</td><td>5.535</td><td>87.568</td><td>1.947</td></tr><tr><td>1</td><td>15,285.072</td><td>1.704</td><td>15,266.652</td><td>1.000</td></tr><tr><td></td><td>FRT (200)</td><td>IRT (1,000)</td><td>FRT-UP (200)</td><td>UIRT-UP (1,000)</td></tr><tr><td>0.995</td><td>6.854</td><td>6.009</td><td>6.303</td><td>8.001</td></tr><tr><td>1</td><td>7.230</td><td>2.351</td><td>6.250</td><td>1.000</td></tr></table>

## 7.4 Unknown request prior

Figure 2 gives the corresponding horizon sweeps when the policies estimate the request prior. At ${ \rho _ { 1 } } = \mathrm { { 0 } } \mathrm { { . 9 9 5 } }$ , the mean regret of FR-UP ranges from 75.704 to 102.245, while that of FRT-UP remains between 5.720 and 6.303 over the full sweep. At $T = 3 2 , 0 0 0$ , UIR-UP has mean regret 1.947 and UIRT-UP has mean regret 8.001. The regrets of FRT-UP, UIR-UP, and UIRT-UP remain far below that of FR-UP throughout the sweep.

At $\rho _ { 1 } = 1$ $\mathrm { F R - U P }$ exhibits the same linear-loss pattern as FR, reaching a mean regret of

![](images/60d10f3d21bd1f21f079b328954b01b548447d82b351dd78a754042ecd3a247c.jpg)  
(a) Unique nondegenerate optimum, ${ \rho _ { 1 } } = \mathrm { { 0 } } \mathrm { { . 9 9 5 } }$

![](images/1c3a4b305ef00f80824955423f9aba1debbbbd73499a75518cb0028631801085.jpg)  
(b) Unique degenerate optimum, $\rho _ { 1 } = 1$  
Figure 2: Mean regret against the transient fluid LP with an unknown request prior.

Table 6: Mean regret against the transient fluid LP near the capacity-ratio boundary at $T = 2 0 0 0$
<table><tr><td rowspan="2"> $\rho _ { 1 }$ </td><td colspan="4">Known prior</td><td colspan="4">Unknown prior</td></tr><tr><td>FR</td><td>IR</td><td>FRT</td><td>IRT</td><td>FR-UP</td><td>UIR-UP</td><td>FRT-UP</td><td>UIRT-UP</td></tr><tr><td>0.998</td><td>203.055</td><td>2.467</td><td>6.794</td><td>2.209</td><td>207.775</td><td>1.485</td><td>5.971</td><td>1.512</td></tr><tr><td>0.999</td><td>318.835</td><td>2.623</td><td>6.901</td><td>1.934</td><td>307.075</td><td>1.457</td><td>5.734</td><td>1.278</td></tr><tr><td>1.000</td><td>1,016.932</td><td>2.090</td><td>6.860</td><td>2.219</td><td>1,011.322</td><td>1.000</td><td>6.360</td><td>1.000</td></tr><tr><td>1.001</td><td>1.000</td><td>1.090</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.081</td><td>1.000</td><td>1.000</td></tr><tr><td>1.002</td><td>1.000</td><td>1.102</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.087</td><td>1.000</td><td>1.000</td></tr></table>

15,266.652 at $T = 3 2 { , } 0 0 0$ with a 95% confidence interval of $[ 1 3 , 9 1 9 . 1 9 0 , 1 6 , 6 1 4 . 1 1 4 ]$ . The mean regret of FRT-UP remains between 5.999 and 6.320 over the horizon sweep. UIR-UP and UIRT-UP have zero empirical variance at each recorded horizon, suggesting that they consistently attain the optimal reward in this instance.

These observations match those in the known-prior case, implying that prior estimation is not the dominant dificulty. Table 5 compares all eight policies at $T = 3 2 { , } 0 0 0$

## 7.5 Perturbing the capacity ratio across the degeneracy boundary

Figure 3 reports the capacity-ratio sweep, which holds $T = 2 0 0 0$ fixed and varies $\rho _ { 1 }$ across the boundary at 1; the left panel shows the known-prior policies, and the right panel shows the unknownprior policies. This provides a direct numerical illustration of support instability under perturbations near the unique degenerate point. For known-prior FR, mean regret rises from 87.255 at ${ \rho _ { 1 } } = \mathrm { { 0 } } \mathrm { { . 9 9 5 } }$ to 318.835 at 0.999, jumps to 1,016.932 at 1, and drops to 1.000 at 1.001. FR-UP follows the same pattern: 307.075 at 0.999, 1,011.322 at 1, and 1.000 at 1.001. A change of only 0.001 in $\rho _ { 1 }$ on either side of the boundary produces a large change in the observed performance of unthresholded round-by-round re-solving.

Thresholding attenuates the boundary spike: at $\rho _ { 1 } = 1$ , FRT and $\mathrm { F R T - U P }$ have mean regrets of

![](images/51ee4a3b82bd6aa7510323cc8a71cfc1019ba045e24c94fb6506838fac783393.jpg)  
(a) Known request prior.

![](images/b531ba58a8a9e533ffb1ed32e6564f9baa6e1d213d18f10bde73299fa14fc2a0.jpg)  
(b) Unknown request prior.  
Figure 3: Mean regret against the transient fluid LP, with 95% confidence intervals, as a function of the first resource’s capacity ratio $\rho _ { 1 }$ at $T = 2 0 0 0$ . All other model primitives are unchanged.

6.860 and 6.360. Their means stay between 5.441 and 7.133 across all tested capacity ratios at or below one and equal 1.000 above one. For policies using infrequent schedules, regret remains low when $\rho _ { 1 } \in [ 0 . 9 9 5 , 1 . 0 0 5 ]$

## 7.6 Lowering the re-solving frequency

To isolate the efect of re-solving frequency, we introduce FR-p and $\mathrm { F R - U P - } p .$ Each policy solves the LP in the first round and, before every later round, independently re-solves with probability $p ;$ otherwise, it retains the most recently computed stationary policy. Thus $p = 1$ recovers FR or FR-UP pathwise. We test $p \in \{ 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 \}$ using 200 trials for each configuration and compare the resulting performance with the existing results for $p = 1$ and for the corresponding infrequent policy, IR or UIR-UP.

Figures 4a and 4b show a consistent ordering of the sample mean regrets by re-solving probability. At the nondegenerate point, the mean regret at $T = 3 2 { , } 0 0 0$ decreases from 100.725 for $p = 1$ to 16.098 for $p = 0 . 2$ , while IR has mean regret 5.535. This ordering of the sample means holds at every tested horizon. At the degenerate point, reducing $p$ lowers the loss substantially but does not prevent it from growing on the scale of the horizon. At $T = 3 2 { , } 0 0 0$ , the mean regrets for $p = 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 ,$ 1 are, respectively, 2,540.116, 6,021.859, 8,299.999, 11,809.823, and 15,285.072, whereas IR has mean regret 1.704.

Figures 4c and 4d give the same comparison when the request prior is estimated. The results follow the same pattern. At the nondegenerate point, the sample mean regrets are ordered by p at every tested horizon. At $T = 3 2 , 0 0 0$ , their values are 17.806, 25.826, 33.965, 56.300, and 87.568 for $p = 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1$ , respectively, compared with 1.947 for UIR-UP. At the degenerate point, the mean regrets at $T = 3 2 { , } 0 0 0$ are 1,809.039, 4,777.806, 8,323.301, 10,979.652, and 15,266.652 for the same five values of $p ,$ while $\mathrm { U I R - U P }$ has mean regret 1.000.

Together, these experiments show that using a lower constant re-solving probability can improve

![](images/40f9fed706e0bab600b1c5051abcc7fec4392c9c763855b5ba76f680705ce71a.jpg)  
(a) Known request prior, unique nondegenerate optimum, $\rho _ { 1 } = 0 . 9 9 5 .$

![](images/4a222bc9d47ca9bc47eac82face44350200b536641df1badac0b348a5333d4f8.jpg)

(b) Known request prior, unique degenerate optimum, ρ<sub>1</sub> = 1.  
![](images/08453e5140db308dd4e4bdb1695a6918da46f0b193d9b146bb40fae3fd1f1ce2.jpg)

$$
{ \rho _ { 1 } } = \mathrm { { 0 } } { . 9 9 5 }
$$

![](images/e1bd927780b6dde3322c6b7b9c1fb291fa6f64d7084e572183732ea99a33eb13.jpg)  
(d) Unknown request prior, unique degenerate optimum, $\rho _ { 1 } = 1$  
Figure 4: Mean regret for constant-frequency re-solving.

the performance of a re-solving algorithm, but this adjustment does not attain the performance order of IR and UIR-UP. This observation supports the importance of a carefully designed LP-solving schedule for infrequent re-solving.

## 8 Conclusion

In this work, we introduce a practically motivated extension of the well-studied online resource allocation problem in which a controlled state evolves according to action-dependent Markov transitions. For this problem, we first develop a stationary LP that extends the fluid benchmark widely used in online resource allocation. In the known-request-prior setting, we consider frequent re-solving policies, which re-solve the LP in every round, and infrequent re-solving policies, which re-solve it in O(log log T) rounds. When the stationary LP has a unique, primal-nondegenerate optimum and the optimal induced kernel is irreducible, both algorithms achieve O(1) regret. However, without nondegeneracy, the infrequent re-solving policy exhibits greater robustness: it still achieves $O ( \sqrt { T } )$ regret, while the frequent re-solving algorithm can incur Θ(T) regret. This result contrasts sharply with results for the original online resource allocation problem showing that frequent LP re-solving is usually more robust. The main underlying mechanism is that too many LP solves can drive the policy away from the stationary optimum and prevent it from returning. In the unknown-prior setting, we further develop a three-phase infrequent re-solving schedule that guarantees order-optimal regret under diferent levels of regularity. The resulting algorithm also solves the LP O(log log T) times in total. Our numerical studies further reveal the advantage of infrequent over frequent re-solving schemes for this problem.

## Acknowledgement

The analysis of the algorithms and the numerical experiments were developed through extensive interactions with ChatGPT. The author has verified the correctness of the proofs and codes and made a thorough re-organization of the results, and the author assumes full responsibility for the contents.

## References

Shipra Agrawal, Zizhuo Wang, and Yinyu Ye. A dynamic near-optimal algorithm for online linear programming. Operations Research, 62(4):876–890, 2014. URL https://doi.org/10.1287/op re.2014.1289.

Alessandro Arlotto and Itai Gurvich. Uniformly bounded regret in the multisecretary problem. Stochastic Systems, 9(3):231–260, 2019.

Alessandro Arlotto and Xinchang Xie. Logarithmic regret in the dynamic and stochastic knapsack problem with equal rewards. Stochastic Systems, 10(2):170–191, 2020.

Santiago R. Balseiro, Haihao Lu, and Vahab Mirrokni. The best of many worlds: Dual mirror descent for online allocation problems. Operations Research, 71(1):101–119, 2023. URL https: //doi.org/10.1287/opre.2021.2242.

Santiago R. Balseiro, Omar Besbes, and Dana Pizarro. Survey of dynamic resource-constrained reward collection problems: Unified model and analysis. Operations Research, 72(5):2168–2189, 2024. doi: 10.1287/opre.2023.2441. URL https://doi.org/10.1287/opre.2023.2441.

Siddhartha Banerjee and Daniel Freund. Uniform loss algorithms for online stochastic decisionmaking with applications to bin packing. In Abstracts of the 2020 SIGMETRICS/Performance Joint International Conference on Measurement and Modeling of Computer Systems, pages 1–2. Association for Computing Machinery, 2020. doi: 10.1145/3393691.3394224. URL https://doi.org/10.1145/3393691.3394224.

Siddhartha Banerjee and Daniel Freund. Good prophets know when the end is near. Management Science, 71(6):4877–4894, 2025. doi: 10.1287/mnsc.2023.04307. URL https://doi.org/10.128 7/mnsc.2023.04307.

Dimitris Bertsimas and John N. Tsitsiklis. Introduction to linear optimization, volume 6. Athena Scientific, Belmont, MA, 1997.

Omar Besbes, Yash Kanoria, and Akshit Kumar. Dynamic resource allocation: Algorithmic design principles and spectrum of achievable performances. Operations Research, 73(3):1273–1288, 2025. doi: 10.1287/opre.2022.0504. URL https://doi.org/10.1287/opre.2022.0504.

Robert L. Bray. Logarithmic regret in multisecretary and online linear programs with continuous valuations. Operations Research, 73(4):2188–2203, 2025. doi: 10.1287/opre.2022.0036. URL https://doi.org/10.1287/opre.2022.0036.

David B. Brown and Jingwei Zhang. Dynamic programs with shared resources and signals: Dynamic fluid policies and asymptotic optimality. Operations Research, 70(5):3015–3033, 2022. URL https://doi.org/10.1287/opre.2021.2181.

David B. Brown and Jingwei Zhang. Fluid policies, reoptimization, and performance guarantees in dynamic resource allocation. Operations Research, 73(2):1029–1045, 2025. URL https: //doi.org/10.1287/opre.2022.0601.

Pornpawee Bumpensanti and He Wang. A re-solving heuristic with uniformly bounded loss for network revenue management. Management Science, 66(7):2993–3009, 2020. URL https: //doi.org/10.1287/mnsc.2019.3365. Online appendix: https://pubsonline.informs.org /doi/suppl/10.1287/mnsc.2019.3365/suppl\_file/mnsc.2019.3365.sm1.pdf.

Matteo Castiglioni, Andrea Celli, and Christian Kroer. Online learning with knapsacks: the best of both worlds. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 2767–2783. PMLR, 2022. URL https://proceedings.mlr.press/v162/castiglioni22a.html.

Guanting Chen, Xiaocheng Li, and Yinyu Ye. An improved analysis of LP-based control for revenue management. Operations Research, 72(3):1124–1138, 2024a. URL https://doi.org/10.1287/ opre.2022.2358.

Zhaohua Chen, Rui Ai, Mingwei Yang, Yuqi Pan, Chang Wang, and Xiaotie Deng. Contextual decision-making with knapsacks beyond the worst case. In Advances in Neural Information Processing Systems 37, 2024b. URL https://doi.org/10.52202/079017-2798.

Vivek F. Farias and Benjamin Van Roy. An approximate dynamic programming approach to network revenue management. Working paper, 23 April, 2007. URL https://web.mit.edu/vivekf/www /papers/ADP-rm-07-03.pdf.

David A. Freedman. On tail probabilities for martingales. The Annals of Probability, 3(1):100–118, 1975. doi: 10.1214/aop/1176996452. URL https://doi.org/10.1214/aop/1176996452.

Wenzhi Gao, Dongdong Ge, Chunlin Sun, Chenyu Xue, and Yinyu Ye. Beyond $\mathcal { O } ( \sqrt { T } )$ regret: Decoupling learning and decision making in online linear programming. Operations Research, 74 (4):1932–1944, 2026. doi: 10.1287/opre.2024.1575. URL https://doi.org/10.1287/opre.2024. 1575.

Varun Gupta. Greedy algorithm for multiway matching with bounded regret. Operations Research, 72(3):1139–1155, 2024. doi: 10.1287/opre.2022.2400. URL https://doi.org/10.1287/opre.2 022.2400.

Yuxuan Han, Jialin Zeng, Yang Wang, Yang Xiang, and Jiheng Zhang. Optimal contextual bandits with knapsacks under realizability via regression oracles. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5011–5035, 2023. URL https://proceedings.mlr.press/v206/han23b.html.

Siqi He, Yehua Wei, Jiaming Xu, and Sophie H. Yu. Online resource allocation without re-solving: The efectiveness of primal-dual policies. SSRN working paper 5133857, revised 11 September 2025, 2025. URL https://doi.org/10.2139/ssrn.5133857.

Stefanus Jasin. Reoptimization and self-adjusting price control for network revenue management. Operations Research, 62(5):1168–1178, 2014. URL https://doi.org/10.1287/opre.2014.1297.

Stefanus Jasin. Performance of an LP-based control for revenue management with unknown demand parameters. Operations Research, 63(4):909–915, 2015. URL https://doi.org/10.1287/opre .2015.1390.

Stefanus Jasin and Sunil Kumar. A re-solving heuristic with bounded revenue loss for network revenue management with customer choice. Mathematics of Operations Research, 37(2):313–345, 2012. URL https://doi.org/10.1287/moor.1120.0537.

Stefanus Jasin and Sunil Kumar. Analysis of deterministic LP-based booking limit and bid price controls for revenue management. Operations Research, 61(6):1312–1320, 2013. URL https://doi.org/10.1287/opre.2013.1216.

Jiashuo Jiang. Constant approximation for network revenue management with Markovian-correlated customer arrivals. In Web and Internet Economics—21st International Conference, WINE 2025, Proceedings, volume 16266 of Lecture Notes in Computer Science, pages 704–746. Springer, 2026. doi: 10.1007/978-3-032-18660-7\_38. URL https://doi.org/10.1007/978-3-032-18660-7\_38.

Jiashuo Jiang and Jiawei Zhang. Online resource allocation with stochastic resource consumption, 2020. URL https://arxiv.org/abs/2012.07933. arXiv preprint arXiv:2012.07933.

Jiashuo Jiang, Xiaocheng Li, and Jiawei Zhang. Online stochastic optimization with Wassersteinbased non-stationarity, 2020. URL https://arxiv.org/abs/2012.06961. arXiv preprint arXiv:2012.06961.

Jiashuo Jiang, Will Ma, and Jiawei Zhang. Degeneracy is OK: Logarithmic regret for network revenue management with indiscrete distributions. Operations Research, 73(6):3405–3420, 2025. doi: 10.1287/opre.2022.0641. URL https://doi.org/10.1287/opre.2022.0641.

Guokai Li, Zizhuo Wang, and Jingwei Zhang. Infrequent resolving algorithm for online linear programming, 2026. URL https://arxiv.org/abs/2408.00465v7. arXiv preprint arXiv:2408.00465v7, revised 1 September 2026.

Weiyuan Li, Paat Rusmevichientong, Huseyin Topaloglu, and Jingwei Zhang. History-dependent fluid approximations and performance guarantees for revenue management with Markov-modulated demands. SSRN working paper 5107682, 21 January 2025, 2025. URL https://doi.org/10.2 139/ssrn.5107682.

Xiaocheng Li and Yinyu Ye. Online linear programming: Dual convergence, new algorithms, and regret bounds. Operations Research, 70(5):2948–2966, 2022. URL https://doi.org/10.1287/ opre.2021.2164.

Xiaocheng Li, Chunlin Sun, and Yinyu Ye. Simple and fast algorithm for binary integer and online linear programming. In Advances in Neural Information Processing Systems 33, pages 9412–9421, 2020.

Wanteng Ma, Ying Cao, Danny H. K. Tsang, and Dong Xia. Optimal regularized online allocation by adaptive re-solving. Operations Research, 73(4):2079–2096, 2025. URL https://doi.org/10 .1287/opre.2022.0486.

Constantinos Maglaras and Joern Meissner. Dynamic pricing strategies for multiproduct revenue management problems. Manufacturing & Service Operations Management, 8(2):136–148, 2006. URL https://doi.org/10.1287/msom.1060.0105.

Sentao Miao and Yining Wang. Network revenue management with nonparametric demand learning: √<sub>T-regret</sub> <sub>and</sub> <sub>polynomial</sub> <sub>dimension</sub> <sub>dependency.</sub> <sub>Mathematics</sub> <sub>of</sub> <sub>Operations</sub> <sub>Research,</sub> <sub>2025.</sub> doi: 10.1287/moor.2022.0086. URL https://doi.org/10.1287/moor.2022.0086. Articles in Advance.

Martin I. Reiman and Qiong Wang. An asymptotically optimal policy for a quantity-based network revenue management problem. Mathematics of Operations Research, 33(2):257–282, 2008. URL https://doi.org/10.1287/moor.1070.0288.

Sébastien Roch. Modern Discrete Probability: An Essential Toolkit. Cambridge University Press, Cambridge, 2024. doi: 10.1017/9781009305129. URL https://doi.org/10.1017/9781009305 129.

Nicola Secomandi. An analysis of the control-algorithm re-solving issue in inventory and revenue management. Manufacturing & Service Operations Management, 10(3):468–483, 2008. URL https://doi.org/10.1287/msom.1070.0184.

Aleksandrs Slivkins, Xingyu Zhou, Karthik Abinav Sankararaman, and Dylan J. Foster. Contextual bandits with packing and covering constraints: A modular Lagrangian approach via regression. Journal of Machine Learning Research, 25(394):1–37, 2024. URL https://jmlr.org/papers/v2 5/24-1220.html.

Rui Sun, Xinshang Wang, and Zijie Zhou. Near-optimal primal-dual algorithms for quantity-based network revenue management, 2020. URL https://arxiv.org/abs/2011.06327. arXiv preprint arXiv:2011.06327.

Aad W. van der Vaart and Jon A. Wellner. Weak Convergence and Empirical Processes: With Applications to Statistics. Springer Series in Statistics. Springer-Verlag, New York, 1996. doi: 10.1007/978-1-4757-2545-2. URL https://doi.org/10.1007/978-1-4757-2545-2.

Alberto Vera and Siddhartha Banerjee. The Bayesian prophet: A low-regret framework for online decision making. Management Science, 67(3):1368–1391, 2021. URL https://doi.org/10.128 7/mnsc.2020.3624.

Alberto Vera, Siddhartha Banerjee, and Itai Gurvich. Online allocation and pricing: Constant regret via Bellman inequalities. Operations Research, 69(3):821–840, 2021. URL https://doi.or g/10.1287/opre.2020.2061.

Alberto Vera, Alessandro Arlotto, Itai Gurvich, and Eli Levin. Dynamic resource allocation: The geometry and robustness of constant regret. Mathematics of Operations Research, 50(4):2834–2872, 2025. doi: 10.1287/moor.2020.0334. URL https://doi.org/10.1287/moor.2020.0334.

Yining Wang and He Wang. Constant regret resolving heuristics for price-based revenue management. Operations Research, 70(6):3538–3557, 2022. doi: 10.1287/opre.2021.2219. URL https: //doi.org/10.1287/opre.2021.2219.

Huasen Wu, Rayadurgam Srikant, Xin Liu, and Chong Jiang. Algorithms with logarithmic or sublinear regret for constrained contextual bandits. In Advances in Neural Information Processing Systems 28, 2015.

## A Proofs for Section 2

## A.1 Proof of Lemma 2.1

Fix an arbitrary nonanticipating policy $\pi$ for the H-round subproblem, and let $\tau \leq H$ be its stopping time. Define

$$
X _ { s j a } ^ { \pi } : = \mathbb { E } ^ { \pi } \left[ \sum _ { t = 1 } ^ { \tau - 1 } \mathbf { 1 } \{ S ^ { t } = s , J ^ { t } = j , A ^ { t } = a \} \right] ,
$$

$$
Y _ { s j a } ^ { \pi } : = \mathbb { P } ^ { \pi } ( S ^ { \tau } = s , J ^ { \tau } = j , A ^ { \tau } = a ) , \qquad e _ { s } ^ { \pi } : = \mathbb { E } ^ { \pi } \left[ \sum _ { t = 1 } ^ { \tau } \mathbf { 1 } \{ S ^ { t } = s \} \right] .
$$

We verify the four constraint families separately.

Request balance. For each t, the event $\{ t \le \tau , S ^ { t } = s \}$ is determined before $J ^ { t }$ is drawn: it depends only on whether a resource was exhausted in an earlier round and on the current state. Therefore

$$
\begin{array} { r } { \mathbb { E } ^ { \pi } [ { \mathbf 1 } \{ t \leq \tau , S ^ { t } = s , J ^ { t } = j \} ] = p _ { j } \mathbb { E } ^ { \pi } [ { \mathbf 1 } \{ t \leq \tau , S ^ { t } = s \} ] . } \end{array}
$$

Summing over $t = 1 , \dots , H$ gives

$$
\sum _ { a } ( X _ { s j a } ^ { \pi } + Y _ { s j a } ^ { \pi } ) = p _ { j } Q _ { s } ^ { \pi } ,
$$

which is (TF1).

State flow. Every visit to $s ^ { \prime }$ is either the initial visit or follows a round that is strictly before the terminal round. Hence

$$
\begin{array} { l } { { \displaystyle Q _ { s ^ { \prime } } ^ { \pi } = \mathbf { 1 } \{ s ^ { \prime } = s ^ { 0 } \} + \mathbb { E } ^ { \pi } \left[ \sum _ { t = 1 } ^ { \tau - 1 } \mathbf { 1 } \{ S ^ { t + 1 } = s ^ { \prime } \} \right] } } \\ { { \displaystyle ~ = \mathbf { 1 } \{ s ^ { \prime } = s ^ { 0 } \} + \sum _ { s , j , a } P ( s ^ { \prime } \mid s , a ) X _ { s j a } ^ { \pi } } , } \end{array}
$$

where the second equality follows by conditioning on $( S ^ { t } , J ^ { t } , A ^ { t } )$ . This is (TF2).

Horizon. By definition,

$$
\sum _ { s } Q _ { s } ^ { \pi } = \mathbb { E } ^ { \pi } [ \tau ] \leq H ,
$$

which is (TF3).

Resources before the terminal action. Pathwise,

$$
\sum _ { t = 1 } ^ { \tau - 1 } c _ { i } ( S ^ { t } , J ^ { t } , A ^ { t } ) = B _ { i } - B _ { i } ^ { \tau } \leq B _ { i } .
$$

The last inequality holds because no earlier round has exhausted resource $i ;$ when $\tau = 1$ , the left-hand sum is empty and the same inequality is immediate. Taking expectations gives (TF4).

Finally,

$$
\sum _ { s , j , a } r ( s , j , a ) ( X _ { s j a } ^ { \pi } + Y _ { s j a } ^ { \pi } ) = \mathbb { E } ^ { \pi } \left[ \sum _ { t = 1 } ^ { \tau } R ^ { t } \right] .
$$

Thus, for every nonanticipating policy, these expected state-request-action and state-visit counts form a feasible transient-LP solution with the same objective value. Maximizing over π proves the result.

## A.2 Proof of Lemma 2.2

Consider the dual of the stationary homogeneous core LP:

$$
\begin{array} { r } { \begin{array} { c l l } { \displaystyle \operatorname* { m i n } _ { \lambda , \gamma , h , u } } & { \lambda ^ { \top } \beta + \gamma } \\ { \mathrm { s u b j e c t ~ t o } } & { u _ { s j } \ge r ( s , j , a ) - \lambda ^ { \top } c ( s , j , a ) + \displaystyle \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) h _ { s ^ { \prime } } , } & { s , j , a , } \\ & { h _ { s } + \gamma \ge \displaystyle \sum _ { j \in \mathcal { I } } p _ { j } u _ { s j } , } & { s \in \mathcal { S } , } \\ & { \lambda \ge 0 . } \end{array} } \end{array}\tag{SD1}
$$

(A.1)

(SD2)

The null action implies $\gamma \geq 0$ for every feasible solution of the stationary dual. Indeed, (SD1) with $a = a ^ { 0 }$ , together with the fact that the null action has zero reward and zero consumption, gives

$$
u _ { s j } \geq \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ^ { 0 } ) h _ { s ^ { \prime } } .
$$

Averaging this inequality over $j$ and using (SD2) yields

$$
h _ { s } + \gamma \geq \sum _ { s ^ { \prime } \in S } { \cal P } ( s ^ { \prime } \mid s , a ^ { 0 } ) h _ { s ^ { \prime } } .
$$

Let $\mu$ be any stationary distribution of the null-action transition matrix; one exists because $s$ is finite. Multiply the last inequality by $\mu _ { s }$ and sum over s. Stationarity cancels the state-bias terms and gives $\gamma \geq 0$

The dual of the transient LP (2) is

$$
\begin{array} { r l l } { \underset { \lambda , \gamma , h , u } { \operatorname* { m i n } } } & { \lambda ^ { \top } B + \gamma H + h _ { s ^ { 0 } } } \\ { \mathrm { s u b j e c t ~ t o } } & { u _ { s j } \geq r ( s , j , a ) , } & { s , j , a , \quad \mathrm { ~ ( T D 1 ) } } \\ & { u _ { s j } \geq r ( s , j , a ) - \lambda ^ { \top } c ( s , j , a ) + \displaystyle { \sum _ { s ^ { \prime } } } P ( s ^ { \prime } \mid s , a ) h _ { s ^ { \prime } } , } & { s , j , a , \quad \mathrm { ~ ( T D 2 ) } } \\ & { h _ { s } + \gamma \geq \displaystyle { \sum _ { j } } p _ { j } u _ { s j } , } & { s , \quad \mathrm { ~ ( T D 3 ) } } \\ & { \quad \quad \lambda \geq 0 , \quad \gamma \geq 0 . } \end{array}\tag{A.2}
$$

Let $( \lambda ^ { * } , \gamma ^ { * } , h ^ { * } , u ^ { * } )$ be an optimal solution of Equation $( \mathrm { A . 1 } )$ at $\beta = \rho ,$ , normalized by $h _ { s ^ { 1 } } ^ { * } = 0$ Define

$$
\kappa : = \operatorname* { m a x } _ { s , j , a } \bigl ( r ( s , j , a ) - u _ { s j } ^ { * } \bigr ) _ { + } .
$$

Increase every component of $u ^ { * }$ and $h ^ { * }$ by $\kappa ,$ and leave $\lambda ^ { * } , \gamma ^ { * }$ unchanged. By definition of $\kappa , ~ ( \mathrm { T D 1 } )$ holds. Since every row of $\textstyle P ( \cdot \mid s , a )$ sums to one, adding the same constant to both u and $h$ preserves (TD2). Since $\textstyle \sum _ { j } p _ { j } = 1$ , it also preserves (TD3). Moreover, $\gamma ^ { * } \geq 0$ . Hence this shifted solution is feasible for Equation (A.2), and its objective at $B = T \rho , H = T , s ^ { 0 } = s ^ { 1 }$ is

$$
\begin{array} { r } { T \big ( ( \lambda ^ { * } ) ^ { \top } \rho + \gamma ^ { * } \big ) + h _ { s ^ { 1 } } ^ { * } + \kappa = T G ( \rho ) + \kappa . } \end{array}
$$

Weak duality proves the lemma.

## B Proofs for Section 3

## B.1 Proof of Proposition 3.3

Proof. Consider an LP in the standard maximization form:

$$
\operatorname* { m a x } \left\{ d ^ { \top } x : A x = b , \ x \geq 0 \right\} ,
$$

and a nonsingular basis $A ^ { * }$ of A. Let $N ^ { * }$ be the columns of A outside $A ^ { * }$ , and define the reduced costs by

$$
\overline { { { d } } } _ { N ^ { \ast } } : = d _ { N ^ { \ast } } - d _ { A ^ { \ast } } ^ { \top } ( A ^ { \ast } ) ^ { - 1 } N ^ { \ast } .
$$

We first recall an important result from Bertsimas and Tsitsiklis (1997):

Proposition B.1 (From Bertsimas and Tsitsiklis (1997)). The following hold:

• $I f$ x is the unique and nondegenerate optimal solution and $A ^ { * }$ is its basis, then $\overline { { d } } _ { N ^ { * } } < 0$

$I f x$ is a feasible solution with basis A<sup>∗</sup>, $x _ { A ^ { * } } > 0$ , and $\overline { { d } } _ { N ^ { * } } < 0$ , then x is the unique and nondegenerate optimal solution.

Using Proposition $_ { \mathrm { B . 1 , } }$ we prove our results as follows. First, add a slack $\sigma _ { i } \geq 0$ to each resource constraint in (SF4), retain every resource row, and delete a fixed set of redundant rows among (SF1)–(SF3). Every resource row can be retained because the column of $\sigma _ { i }$ is nonzero only in that row. With

$$
x = ( z , q , \sigma ) ,
$$

the resulting full-row-rank standard form is

$$
\operatorname* { m a x } \left\{ d ^ { \top } x : A x = b ( \beta ) , \ x \geq 0 \right\} .\tag{B.1}
$$

Notice that only the resource entries of the right-hand side depend on $\beta ,$ and they are afine in $\beta .$ In fact, if $e _ { i }$ is the unit vector corresponding to resource row $i ,$ then

$$
b ( \beta ) = b ( \rho ) + \sum _ { i \in [ m ] } ( \beta _ { i } - \rho _ { i } ) e _ { i } .
$$

Now let $A ^ { * }$ be the basis for the optimal solution $x ^ { * } ( \rho )$ under $\beta = \rho _ { ; }$ , and let $N ^ { * }$ be the remaining columns of A. Therefore,

$$
x _ { A ^ { * } } ^ { * } ( \rho ) = ( A ^ { * } ) ^ { - 1 } b ( \rho ) > 0 , \quad x _ { N ^ { * } } ^ { * } ( \rho ) = 0 .
$$

Also, Proposition B.1 and the assumed uniqueness and nondegeneracy give $\bar { d } _ { N ^ { * } } < 0$

Define $x ( \beta )$ by

$$
x _ { A ^ { \ast } } ( \beta ) : = ( A ^ { \ast } ) ^ { - 1 } b ( \beta ) , \quad x _ { N ^ { \ast } } ( \beta ) : = 0 .
$$

Since $( A ^ { * } ) ^ { - 1 } b ( \rho ) > 0$ , there exist suficiently small constants $\delta , \eta > 0$ such that $x _ { A ^ { \ast } } ( \beta ) \geq \eta$ whenever $\beta \in \mathcal { N } _ { 3 \delta }$ . By Proposition B.1, $x ( \beta )$ is the unique and nondegenerate optimal solution to (B.1) and

has the same basis $A ^ { * }$

Further, for $i \notin I ^ { * }$ , the column of the basic slack $\sigma _ { i }$ is $e _ { \sigma _ { i } }$ . Hence

$$
( A ^ { * } ) ^ { - 1 } e _ { i } = e _ { \sigma _ { i } } ,
$$

and

$$
\begin{array} { l } { { \displaystyle x _ { A ^ { * } } ( \beta ) = ( A ^ { * } ) ^ { - 1 } b ( \beta ) } } \\ { { \displaystyle \quad = ( A ^ { * } ) ^ { - 1 } \big ( b ( \rho ) + \sum _ { i \in [ m ] } ( \beta _ { i } - \rho _ { i } ) e _ { i } \big ) } } \\ { { \displaystyle \quad = x _ { A ^ { * } } ^ { * } ( \rho ) + \sum _ { i \in I ^ { * } } ( \beta _ { i } - \rho _ { i } ) \big ( A ^ { * } \big ) ^ { - 1 } e _ { i } + \sum _ { i \notin I ^ { * } } ( \beta _ { i } - \rho _ { i } ) e _ { \sigma _ { i } } . } } \end{array}
$$

Therefore, the coordinates of $x _ { A ^ { * } } ( \beta )$ that do not correspond to resource slacks (i.e., those corresponding to states and decisions) do not depend on $\beta _ { i }$ for $i \notin I ^ { * }$ □

## B.2 Proof of Proposition 3.5

Proof. Closure. Fix $s ^ { \prime } \notin \mathcal { R } ^ { * }$ . Local support stability gives $q _ { s ^ { \prime } } ( \beta ) = 0$ . The state-flow equality is

$$
0 = q _ { s ^ { \prime } } ( \beta ) = \\\sum _ { s , j , a } P ( s ^ { \prime } \mid s , a ) z _ { s j a } ( \beta ) .
$$

Every summand is nonnegative. Therefore, whenever $z _ { s j a } ( \beta ) > 0$ , one must have $P ( s ^ { \prime } \mid s , a ) = 0$ Averaging these transition probabilities under $\pi _ { \beta }$ shows $K _ { \beta } ( s , s ^ { \prime } ) = 0$ for every $s \in \mathcal { R } ^ { * }$

Lipschitz dependence. By Proposition 3.3, the fixed-basis stationary LP solution is afine in $\beta _ { I ^ { * } }$ . On $\mathcal { R } ^ { * }$ , every $q _ { s } ( \beta )$ is at least some positive constant, so division in (4) preserves Lipschitz continuity. (5) then gives the same property for $K _ { \beta }$ . Uniform boundedness is immediate.

Irreducibility. Finally, irreducibility within $\mathcal { R } ^ { * }$ follows by observing that $K ^ { * } ( s , s ^ { \prime } ) > 0$ if and only if $K _ { \beta } ( s , s ^ { \prime } ) > 0$ □

## B.3 Proof of Proposition 3.6

Proof. Consider the fixed standard form (B.1), which retains all variables $( z , q , \sigma )$ and deletes only redundant equality rows. Let $A ^ { * }$ be the nominal basis matrix. It is nonsingular, and Assumption 3.2 ensures that every basic variable is strictly positive.

Suppose that $q _ { s } ^ { * } = 0$ for some $s \in { \mathcal { S } }$ . By (SF1) in (3),

$$
\sum _ { a \in \mathcal { A } } z _ { s j a } ^ { * } = p _ { j } q _ { s } ^ { * } = 0 \qquad \mathrm { f o r ~ e v e r y ~ } j \in \mathcal { I } .
$$

Nonnegativity implies $z _ { s j a } ^ { * } = 0$ for every $j \in \mathcal { I }$ and $a \in { \mathcal { A } }$ . Thus $q _ { s }$ and all coordinates $z _ { s j a }$ indexed by this state are nonbasic.

Fix any $j \in \mathcal I$ and consider the coeficient row of the equality

$$
\sum _ { a \in \mathcal { A } } z _ { s j a } - p _ { j } q _ { s } = 0 .
$$

This row is nonzero, but all its nonzero coeficients belong to nonbasic columns, so its restriction to the basis columns is zero. Thus the basis matrix contains an all-zero row, contradicting its nonsingularity. Hence $q _ { s } ^ { * } > 0$ for every $s \in { \mathcal { S } }$ , and (8) gives ${ \mathcal { R } } ^ { * } = { \mathcal { S } }$ □

## C Proof of Theorem 4.1

## C.1 Analytical setup

This appendix supplies the stopped comparisons and supporting analysis for the algorithms in Section 4.

To start with, we mention that the null-completion convention is never invoked along a regular path in the proof of Theorem 4.1. Nevertheless, it is essential to the unique-degenerate lower-bound construction in Proposition 5.3 for frequent re-solving.

For the upper-bound proof, choose an integer cutof $H _ { \perp }$ satisfying

$$
H _ { \perp } : = \left\lceil 1 + \operatorname* { m a x } \left\{ \frac { \overline { c } } { \operatorname* { m i n } _ { i } ( \rho _ { i } - \delta ) } , \frac { \overline { c } + \operatorname* { m a x } _ { i } \rho _ { i } } { \delta } + 1 \right\} \right\rceil .\tag{C.1}
$$

Analytical-check convention. Throughout the appendices, a check means a history-measurable test that can terminate a proof-only comparison. We use five distinct types.

1. The locality check is applied at a resolving time, before an action, and tests membership of the current resolving parameter in the neighborhood specified at that use.

2. The target-state check, also applied at a resolving time before an action, tests whether the current state belongs to $\mathcal { R } ^ { * }$

3. The resource near-depletion check is applied before each contemplated action and fires when mi $\begin{array} { r } { \mathrm { n } _ { i \in [ m ] } B _ { i } ^ { t } \leq \bar { c } . } \end{array}$

4. The epoch-movement check is applied after a reached epoch endpoint and before the checks for the next epoch; it compares the old-to-new capacity movement with the bufer specified there.

5. Finally, the prior-error check, used only with an unknown request prior, is applied at the start of an empirical epoch before an action and compares the empirical-prior error with the tolerance specified there.

Each use below states the relevant neighborhood, tolerance, or bufer explicitly.

On a successful entrance, call the entrance regular when $\beta ^ { \mathrm { e n t } } \in \mathcal { N } _ { \delta }$ and irregular otherwise. Define FR<sup>g</sup> by coupling it to FR until the first post-entry round at which the pre-action locality check $\beta ^ { t } \in \mathcal { N } _ { \delta }$ fails or the deterministic cutof $H ^ { t } \leq H _ { \perp }$ is reached, and using the null action thereafter; it also truncates immediately after an irregular entrance. The comparison then satisfies the pathwise inequality

$$
\mathcal { R } _ { T } ^ { \mathrm { F R } } \geq \mathcal { R } _ { T } ^ { \mathrm { F R ^ { g } } } .\tag{C.2}
$$

Thus, every upper bound proved for the guarded comparison transfers directly to FR.

Before its analytical truncation, Equation (C.1) implies $B _ { i } ^ { t } > \bar { c }$ for every resource. Upon successful entry, $S ^ { \tau ^ { \mathrm { e n t } } } \in \mathcal { R } ^ { * }$ , and Proposition 3.5 ensures that the optimal target state class $\mathcal { R } ^ { * }$ is closed under every locally induced kernel. Hence the denominator in Equation (4) is uniformly positive along the guarded comparison. The second term in Equation (C.1) also provides a one-step bufer: if $\beta ^ { t } \in \mathcal { N } _ { \delta }$ and $H ^ { t } > H _ { \perp }$ , then $\beta ^ { t + 1 } \in \mathcal { N } _ { 2 \delta }$

Similarly, define $\mathrm { I R ^ { g } }$ by coupling it to IR and switching permanently to the null action after an irregular entrance. After a regular entrance, the switch occurs at the first resolving round at which the pre-action locality check $\beta ^ { t } \in \mathcal { N } _ { \delta }$ fails, the first round at which the pre-action resource near-depletion check min $\mathsf { \Omega } _ { i \in [ m ] } B _ { i } ^ { t } \le \bar { c }$ fires, or the first round at which the deterministic cutof $H ^ { t } \leq H _ { \perp }$ is reached. The comparison also pathwise satisfies

$$
\mathcal { R } _ { T } ^ { \mathrm { I R } } \geq \mathcal { R } _ { T } ^ { \mathrm { I R ^ { g } } } .\tag{C.3}
$$

## C.2 Common analytical ingredients and proofs

## C.2.1 The dual Bellman potential

Choose an arbitrary optimal dual solution of $G ( \rho )$ ，

$$
( \lambda ^ { * } , \gamma ^ { * } , h ^ { * } , u ^ { * } ) ,
$$

and use the dual shift invariance to normalize $h _ { s ^ { 1 } } ^ { * } = 0$ . Complementary slackness gives

$$
\begin{array} { l l } { { } } & { { \displaystyle \lambda _ { i } ^ { * } = 0 , \qquad } } & { { \qquad i \notin I ^ { * } , } } \\ { { } } & { { \displaystyle u _ { s j } ^ { * } = r ( s , j , a ) - ( \lambda ^ { * } ) ^ { \top } c ( s , j , a ) + \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) h _ { s ^ { \prime } } ^ { * } , \qquad } } & { { \qquad ( s , j , a ) \in \mathcal { E } ^ { * } , } } \end{array}\tag{C.4}
$$

$$
{ h _ { s } ^ { * } + \gamma ^ { * } = \sum _ { j } p _ { j } u _ { s j } ^ { * } , }
$$

$$
s \in \mathcal { R } ^ { * } .\tag{C.5}
$$

For every action with $\pi _ { \beta ^ { \prime } } ( a \mid s , j ) > 0$ , one has $z _ { s j a } ( \beta ^ { \prime } ) > 0$ . Proposition 3.3 therefore implies $( s , j , a ) \in \mathcal { E } ^ { * }$ . Nominal complementary slackness (C.4) gives

$$
r ( s , j , a ) - ( \lambda ^ { * } ) ^ { \top } c ( s , j , a ) + \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) h _ { s ^ { \prime } } ^ { * } = u _ { s j } ^ { * } .
$$

Average this equality over $A \sim \pi _ { \beta ^ { \prime } } ( \cdot \mid s , j )$ . The right-hand side remains $u _ { s j } ^ { * }$ because it does not depend on a. Then average over $J \sim p .$ . Since $s \in \mathcal { R } ^ { * }$ , (C.5) gives

$$
\mathbb { E } \left[ R - ( \lambda ^ { * } ) ^ { \top } c ( s , J , A ) + h _ { S ^ { \prime } } ^ { * } \mid S = s \right] = \sum _ { j } p _ { j } u _ { s j } ^ { * } = h _ { s } ^ { * } + \gamma ^ { * } .\tag{C.6}
$$

Now define

$$
\begin{array} { r } { F ( H , B , s ) : = ( \lambda ^ { * } ) ^ { \top } B + \gamma ^ { * } H + h _ { s } ^ { * } . } \end{array}
$$

We directly obtain the following lemma.

Lemma C.1. Fix $\beta ^ { \prime } \in \mathcal { N } _ { 3 \delta }$ and $s \in \mathcal { R } ^ { * }$ . Suppose that one round uses the policy $\pi _ { \beta ^ { \prime } } { } _ { i }$ , regardless of the current inventory ratio. Then, for every $H \geq 1$ and $B \in \mathbb { R } ^ { m }$ ,

$$
\begin{array} { r } { \mathbb { E } \left[ R + F ( H - 1 , B - c ( s , J , A ) , S ^ { \prime } ) \ | \ S = s , B , H \right] = F ( H , B , s ) , } \end{array}\tag{C.7}
$$

where $J \sim p , A \sim \pi _ { \beta ^ { \prime } } ( \cdot \mid s , J ) $ , and $S ^ { \prime } \sim P ( \cdot \mid s , A )$

Proof. Adding $( \lambda ^ { * } ) ^ { \top } B + \gamma ^ { * } ( H - 1 )$ to both sides of (C.6) gives (C.7).

At the initial point, strong duality and $h _ { s ^ { 1 } } ^ { * } = 0$ give

$$
F ( T , T \rho , s ^ { 1 } ) = T ( ( \lambda ^ { * } ) ^ { \top } \rho + \gamma ^ { * } ) = T G ( \rho ) .\tag{C.8}
$$

For any history on which entrance succeeds, recall that $\tau ^ { \mathrm { e n t } }$ is the first round t such that $S ^ { t } \in \mathcal { R } ^ { * }$ ， and define

$$
N ^ { \mathrm { e n t } } : = \tau ^ { \mathrm { e n t } } - 1 , \quad H ^ { \mathrm { e n t } } : = T - N ^ { \mathrm { e n t } } ;
$$

$$
\begin{array} { c c } { C ^ { \mathrm { e n t } } : = \displaystyle \sum _ { t = 1 } ^ { N ^ { \mathrm { e n t } } } c ( S ^ { t } , J ^ { t } , A ^ { t } ) , \quad B ^ { \mathrm { e n t } } : = B ^ { 1 } - C ^ { \mathrm { e n t } } , \quad R ^ { \mathrm { e n t } } : = \displaystyle \sum _ { t = 1 } ^ { N ^ { \mathrm { e n t } } } r ( S ^ { t } , J ^ { t } , A ^ { t } ) ; } \\ { S ^ { \mathrm { e n t } } : = S ^ { \tau ^ { \mathrm { e n t } } } , \quad X ^ { \mathrm { e n t } } : = B ^ { \mathrm { e n t } } - H ^ { \mathrm { e n t } } \rho . } \end{array}
$$

For any inventory deviation $x \in \mathbb { R } ^ { m }$ , define the one-sided norm

$$
\mathcal { D } ( x ) : = \sum _ { i \in I ^ { * } } | x _ { i } | + \sum _ { i \notin I ^ { * } } ( - x _ { i } ) _ { + } .\tag{C.9}
$$

The next lemma shows that the entrance imbalance afects every performance bound by at most a constant.

Lemma C.2. Under Assumption 3.7,

• let ${ \mathcal { E } } _ { T }$ be the event that $N ^ { \mathrm { e n t } } < T$ and $C _ { i } ^ { \mathrm { e n t } } < T \rho _ { i }$ for every resource. Then

$$
T \mathbb { P } ( \mathcal { E } _ { T } ^ { \mathrm { c } } ) = O ( 1 ) ,\tag{C.10}
$$

$$
\mathbb { E } \big [ \lVert X ^ { \mathrm { e n t } } \rVert _ { 1 } { \mathbf { 1 } } \{ \mathcal { E } _ { T } \} \big ] = O ( 1 ) , \quad \mathbb { E } [ { \mathcal { D } } ( X ^ { \mathrm { e n t } } ) { \mathbf { 1 } } \{ \mathcal { E } _ { T } \} ] = O ( 1 ) ;\tag{C.11}
$$

• on ${ { \mathcal { E } } _ { T } }$ ,

$$
\begin{array} { r l r } {  { T G ( \rho ) - R ^ { \mathrm { e n t } } - F ( H ^ { \mathrm { e n t } } , B ^ { \mathrm { e n t } } , S ^ { \mathrm { e n t } } ) } } \\ & { \leq ( \| \lambda ^ { * } \| _ { \infty } + 1 ) \| C ^ { \mathrm { e n t } } \| _ { 1 } + ( \| \rho \| _ { 1 } + | \gamma ^ { * } | ) N ^ { \mathrm { e n t } } + \operatorname* { m a x } _ { s } | h _ { s } ^ { * } | , } & \end{array}\tag{C.12}
$$

and the expectation of the right-hand side is O(1);

• for every fixed $r > 0$

$$
\mathbb { E } \left[ H ^ { \mathrm { e n t } } \mathbf { 1 } \{ \mathcal { E } _ { T } , { B } ^ { \mathrm { e n t } } / H ^ { \mathrm { e n t } } \not \in \mathcal { N } _ { r } \} \right] \leq \frac { 1 } { r } \mathbb { E } [ \mathcal { D } ( X ^ { \mathrm { e n t } } ) \mathbf { 1 } \{ \mathcal { E } _ { T } \} ] = O ( 1 ) .\tag{C.13}
$$

Proof. Markov’s inequality and Assumption 3.7 give

$$
\mathbb { P } ( N ^ { \mathrm { e n t } } \geq T ) \leq \frac { \mathbb { E } N ^ { \mathrm { e n t } } } { T } , \qquad \mathbb { P } ( C _ { i } ^ { \mathrm { e n t } } \geq T \rho _ { i } ) \leq \frac { \mathbb { E } C _ { i } ^ { \mathrm { e n t } } } { T \rho _ { i } } .
$$

A union bound proves (C.10). Interpret the entrance quantities on the infinite-resource trajectory

from Assumption 3.7. On ${ \mathcal { E } } _ { T }$ , we know that

$$
X ^ { \mathrm { e n t } } = ( T \rho - C ^ { \mathrm { e n t } } ) - ( T - N ^ { \mathrm { e n t } } ) \rho = N ^ { \mathrm { e n t } } \rho - C ^ { \mathrm { e n t } } .
$$

Consequently, using componentwise nonnegativity of consumption,

$$
\begin{array} { r l r } {  { \mathbb { E } \big [ \| X ^ { \mathrm { e n t } } \| _ { 1 } { \mathbf { 1 } } \{ \mathcal { E } _ { T } \} \big ] \leq \| \rho \| _ { 1 } \mathbb { E } [ N ^ { \mathrm { e n t } } ] + \mathbb { E } [ \| C ^ { \mathrm { e n t } } \| _ { 1 } ] } } \\ & { } & { = \| \rho \| _ { 1 } \mathbb { E } [ \tau ^ { \mathrm { e n t } } - 1 ] + \mathbb { E } [ \sum _ { t = 1 } ^ { \tau ^ { \mathrm { e n t } } - 1 } \| c ( S ^ { t } , J ^ { t } , A ^ { t } ) \| _ { 1 } ] = O ( 1 ) , } \end{array}
$$

where the indicator was dropped when obtaining the first upper bound and the final equality follows from (10). Moreover,

$$
\mathcal { D } ( \boldsymbol { x } ) \leq \| \boldsymbol { x } \| _ { 1 } , \qquad \boldsymbol { x } \in \mathbb { R } ^ { m } ,
$$

directly from (C.9). This proves (C.11). Direct substitution into the definition of $F$ gives

$$
T G ( \rho ) - R ^ { \mathrm { e n t } } - F ( H ^ { \mathrm { e n t } } , B ^ { \mathrm { e n t } } , S ^ { \mathrm { e n t } } ) = ( \lambda ^ { * } ) ^ { \top } C ^ { \mathrm { e n t } } + \gamma ^ { * } N ^ { \mathrm { e n t } } - h _ { S ^ { \mathrm { e n t } } } ^ { * } - R ^ { \mathrm { e n t } } .
$$

This expression is bounded above by the right-hand side of (C.12), whose expectation is finite because rewards are nonnegative and the entrance bounds hold. Finally, if $B ^ { \mathrm { e n t } } / H ^ { \mathrm { e n t } } \notin { \mathcal { N } } _ { r }$ , then either $\lvert X _ { i } ^ { \mathrm { e n t } } \rvert \ge r H ^ { \mathrm { e n t } }$ for some $i \in I ^ { * }$ or $- X _ { i } ^ { \mathrm { e n t } } \geq r H ^ { \mathrm { e n t } }$ for some i /∈ $I ^ { * }$ . Hence $r H ^ { \mathrm { e n t } } \le { \mathcal { D } } ( X ^ { \mathrm { e n t } } )$ on that event, proving (C.13). □

## C.3 Common resource averages and Poisson equations

For each resource $i ,$ define

$$
g _ { i } ( \beta , s ) : = \sum _ { j } p _ { j } \sum _ { a } \pi _ { \beta } ( a \mid s , j ) c _ { i } ( s , j , a ) ,
$$

and

$$
\overline { { c } } _ { i } ( \beta ) : = \sum _ { s } q _ { s } ( \beta ) g _ { i } ( \beta , s ) = \sum _ { s , j , a } c _ { i } ( s , j , a ) z _ { s j a } ( \beta ) .\tag{C.14}
$$

Proposition 3.3 implies that when $\beta \in \mathcal { N } _ { 3 \delta }$ 2

$$
\begin{array} { r } { \overline { c } _ { i } ( \beta ) = \beta _ { i } \quad ( i \in I ^ { * } ) , \qquad \overline { c } _ { i } ( \beta ) < \beta _ { i } - \eta \quad ( i \notin I ^ { * } ) . } \end{array}\tag{C.15}
$$

Now let $K _ { \beta }$ and $q _ { \beta }$ denote the kernel and stationary distribution on $\mathcal { R } ^ { * }$ corresponding to those binding coordinates, and set

$$
\Pi _ { \beta } : = \mathbf { 1 } q _ { \beta } ^ { \top } .
$$

These objects are independent of the inactive capacity coordinates by Proposition 3.3, and $K _ { \beta }$ is irreducible by Proposition 3.5. We claim that

$$
I - K _ { \beta } + \Pi _ { \beta }
$$

is nonsingular. Indeed, suppose that $x \in \mathbb { R } ^ { | \mathcal { R } ^ { * } | }$ belongs to its null space. Premultiplying by $q _ { \beta } ^ { \top }$ and using stationarity $q _ { \beta } ^ { \top } K _ { \beta } = q _ { \beta } ^ { \top }$ and $q _ { \beta } ^ { \top } \mathbf { 1 } = 1$ gives

$$
0 = q _ { \beta } ^ { \top } ( I - K _ { \beta } + \Pi _ { \beta } ) x = q _ { \beta } ^ { \top } x .
$$

Hence $\Pi _ { \beta } x = \mathbf { 1 } q _ { \beta } ^ { \top } x = 0$ , and the null-space equation reduces to $( I - K _ { \beta } ) x = 0$ . For a finite irreducible stochastic matrix, the right eigenspace associated with eigenvalue one is span{1}. Thus $x = c \mathbf { 1 }$ for some scalar c, and the centering equality above gives $c = 0$ . The null space is therefore trivial, so the matrix is nonsingular. Define

$$
Z _ { \beta } : = ( I - K _ { \beta } + \Pi _ { \beta } ) ^ { - 1 } .
$$

More generally, if $q _ { \beta } ^ { \top } x = 0$ and $x ^ { \prime } = Z _ { \beta } x$ , then premultiplying $( I - K _ { \beta } + \Pi _ { \beta } ) x ^ { \prime } = x$ by $q _ { \beta } ^ { \top }$ gives $\Pi _ { \beta } x ^ { \prime } = 0$ and $( I - K _ { \beta } ) x ^ { \prime } = x$ . Conversely, every centered solution of this last equation solves the invertible system with the stationary projector added, so the centered solution is unique.

For $\beta \in \mathcal { N } _ { 3 \delta }$ , the vector $g _ { i } ( \beta , \cdot ) - \overline { { c } } _ { i } ( \beta ) \mathbf { 1 }$ has zero $q ( \beta )$ -mean by (C.14). Consequently, $Z _ { \beta _ { I } }$ <sub>∗</sub> maps this vector to the unique centered solution of the Poisson equation. We denote that solution by $v _ { i } ( \beta , \cdot )$ ; equivalently, it is the unique solution on $\mathcal { R } ^ { * }$ of

$$
\begin{array} { r l } & { v _ { i } ( \beta , s ) - \displaystyle \sum _ { s ^ { \prime } \in { \mathcal R } ^ { * } } K _ { \beta } ( s , s ^ { \prime } ) v _ { i } ( \beta , s ^ { \prime } ) = g _ { i } ( \beta , s ) - \overline { { c } } _ { i } ( \beta ) , } \\ & { \qquad \displaystyle \sum _ { s \in { \mathcal R } ^ { * } } q _ { s } ( \beta ) v _ { i } ( \beta , s ) = 0 . } \end{array}
$$

Lemma C.3. There are finite constants $V , C _ { \mathrm { P o i s } }$ such that, for every resource $i ,$ every state $s \in \mathcal { R } ^ { * }$ and all $\beta , \beta ^ { \prime } \in \mathcal { N } _ { 2 \delta }$ ,

$$
\begin{array} { r } { | v _ { i } ( { \boldsymbol { \beta } } , s ) | \leq V , \qquad | v _ { i } ( { \boldsymbol { \beta } } , s ) - v _ { i } ( { \boldsymbol { \beta } } ^ { \prime } , s ) | \leq C _ { \mathrm { P o i s } } \| \beta _ { I ^ { * } } - \beta _ { I ^ { * } } ^ { \prime } \| _ { \infty } . } \end{array}
$$

Proof. Only the binding coordinates afect $q , \pi , K , g _ { i }$ , and $\overline { { c } } _ { i }$ . Consider the compact box $\boldsymbol { B } _ { 2 \delta }$ defined in Section 3. It lies strictly inside the basis-stability region $\left| \beta _ { i } - \rho _ { i } \right| < 3 \delta$

For $\beta \in B _ { 2 \delta }$ , consider the corresponding restrictions to $\mathcal { R } ^ { * }$ and the fundamental matrix defined above. Proposition 3.5 ensures irreducibility throughout the box. The matrix depends continuously on $\beta _ { i }$ , so compactness gives

$$
\operatorname* { s u p } _ { \beta \in B _ { 2 \delta } } \| Z _ { \beta } \| < \infty .\tag{C.16}
$$

The centered solution is

$$
v _ { i } ( \beta ) = Z _ { \beta } \big ( g _ { i } ( \beta ) - \overline { { c } } _ { i } ( \beta ) { \bf 1 } \big ) .
$$

The right-hand side is uniformly bounded, proving the first claim.

For the second claim, the fixed-basis formulas make $q , \pi , K , g _ { i }$ , and $\overline { { c } } _ { i }$ Lipschitz on the box. The inverse identity

$$
Z _ { \beta } - Z _ { \beta ^ { \prime } } = Z _ { \beta } \big [ ( K _ { \beta } - K _ { \beta ^ { \prime } } ) - ( \Pi _ { \beta } - \Pi _ { \beta ^ { \prime } } ) \big ] Z _ { \beta ^ { \prime } }
$$

combined with (C.16) makes $Z _ { \beta }$ Lipschitz as well. Substitution into the formula for $v _ { i }$ proves the

stated bound. Since inactive coordinates do not afect these objects, the result holds on the full one-sided neighborhood. □

## C.4 Frequent re-solving analysis and proof

Throughout this section, we analyze the policy FR<sup>g</sup> defined in Appendix C.1. The implemented FR policy dominates this stopped policy pathwise by (C.2), so the resulting regret upper bound transfers to FR.

## C.4.1 Time-varying martingale decomposition and exact ratio recursion

We call a round regular if it occurs before analytical truncation, its state belongs to $\mathcal { R } ^ { * }$ , its remaining horizon exceeds $H _ { \perp }$ , and the remaining average inventory of the round belongs to $\mathcal { N } _ { \delta }$ . In a regular FR round, write

$$
C ^ { t } : = c ( S ^ { t } , J ^ { t } , A ^ { t } ) , \qquad C _ { i } ^ { t } : = c _ { i } ( S ^ { t } , J ^ { t } , A ^ { t } ) .
$$

Define the martingale diference

$$
\begin{array} { l } { { \displaystyle \xi _ { i } ^ { t + 1 } : = C _ { i } ^ { t } - g _ { i } ( \beta ^ { t } , S ^ { t } ) + v _ { i } ( \beta ^ { t } , S ^ { t + 1 } ) } } \\ { { \displaystyle ~ - \sum _ { s ^ { \prime } } K _ { \beta ^ { t } } ( S ^ { t } , s ^ { \prime } ) v _ { i } ( \beta ^ { t } , s ^ { \prime } ) . } } \end{array}
$$

We thus have by definition that

$$
\mathbb { E } [ \xi _ { i } ^ { t + 1 } \mid \mathcal { F } ^ { t } ] = 0 .
$$

Moreover, $| \xi _ { i } ^ { t + 1 } | \leq \bar { c } + 2 V = O ( 1 )$ . The Poisson equation gives the exact identity

$$
C _ { i } ^ { t } - \overline { { c } } _ { i } ( \beta ^ { t } ) = \xi _ { i } ^ { t + 1 } + v _ { i } ( \beta ^ { t } , S ^ { t } ) - v _ { i } ( \beta ^ { t } , S ^ { t + 1 } ) .\tag{C.17}
$$

Since $B ^ { t + 1 } = B ^ { t } - C ^ { t }$ and $H ^ { t + 1 } = H ^ { t } - 1$

$$
\beta _ { i } ^ { t + 1 } - \beta _ { i } ^ { t } = - \frac { C _ { i } ^ { t } - \beta _ { i } ^ { t } } { H ^ { t } - 1 } .\tag{C.18}
$$

Define

$$
W _ { i } ^ { t } : = \beta _ { i } ^ { t } - \rho _ { i } - \frac { v _ { i } ( \beta ^ { t } , S ^ { t } ) } { H ^ { t } } .
$$

Lemma C.4. When $\beta ^ { t } \in \mathcal { N } _ { \delta } , H ^ { t } > H _ { \perp } , \beta ^ { t + 1 } \in \mathcal { N } _ { 2 \delta }$ . Further, for every resource $i ,$

$$
W _ { i } ^ { t + 1 } - W _ { i } ^ { t } = - \frac { \xi _ { i } ^ { t + 1 } } { H ^ { t } - 1 } - \frac { \overline { { c } } _ { i } ( \beta ^ { t } ) - \beta _ { i } ^ { t } } { H ^ { t } - 1 } + \mathcal { R } _ { i } ^ { t + 1 } ,\tag{C.19}
$$

where

$$
\mathcal { R } _ { i } ^ { t + 1 } = \frac { v _ { i } ( \beta ^ { t } , S ^ { t + 1 } ) - v _ { i } ( \beta ^ { t + 1 } , S ^ { t + 1 } ) } { H ^ { t } - 1 } - \frac { v _ { i } ( \beta ^ { t } , S ^ { t } ) } { H ^ { t } ( H ^ { t } - 1 ) } ,\tag{C.20}
$$

$$
| \mathcal { R } _ { i } ^ { t + 1 } | = O \big ( ( H ^ { t } - 1 ) ^ { - 2 } \big ) .\tag{C.21}
$$

Proof. For $i \in I ^ { * }$ and $\beta ^ { t } \in \mathcal { N } _ { \delta }$

$$
\vert \beta _ { i } ^ { t + 1 } - \beta _ { i } ^ { t } \vert \leq \frac { C _ { i } ^ { t } + \beta _ { i } ^ { t } } { H ^ { t } - 1 } \leq \frac { \overline { { c } } + \rho _ { i } + \delta } { H ^ { t } - 1 } \leq \frac { \overline { { c } } + \operatorname* { m a x } _ { i } \rho _ { i } + \delta } { H ^ { t } - 1 } < \delta .
$$

For $i \notin I ^ { * }$ , only the lower boundary matters and

$$
\beta _ { i } ^ { t + 1 } \geq \beta _ { i } ^ { t } - \frac { C _ { i } ^ { t } } { H ^ { t } - 1 } > \rho _ { i } - \delta - \frac { \overline { { c } } } { H ^ { t } - 1 } > \rho _ { i } - 2 \delta .
$$

Thus $\beta ^ { t + 1 } \in \mathcal { N } _ { 2 \delta }$

Substitute (C.17) into (C.18), and then subtract the two Poisson-correction terms in the definition of W. Direct cancellation gives (C.19) and (C.20).

The preceding bound and Lemma C.3 imply

$$
\vert v _ { i } ( \beta ^ { t } , S ^ { t + 1 } ) - v _ { i } ( \beta ^ { t + 1 } , S ^ { t + 1 } ) \vert \leq \frac { C _ { \mathrm { P o i s } } ( \overline { c } + \operatorname* { m a x } _ { i } \rho _ { i } + \delta ) } { H ^ { t } - 1 } .
$$

Also $| v _ { i } ( \beta ^ { t } , S ^ { t } ) | \le V$ . Therefore (C.20) gives

$$
| \mathcal { R } _ { i } ^ { t + 1 } | \leq \frac { C _ { \mathrm { P o i s } } ( \overline { { c } } + \operatorname* { m a x } _ { i } \rho _ { i } + \delta ) } { ( H ^ { t } - 1 ) ^ { 2 } } + \frac { V } { H ^ { t } ( H ^ { t } - 1 ) } = O \Big ( ( H ^ { t } - 1 ) ^ { - 2 } \Big ) .
$$

## C.4.2 An exponential exit bound

We use the following standard maximal form of Freedman’s inequality.

Fact C.5 (Freedman’s inequality, from Freedman (1975)). Let $( M ^ { n } , { \mathcal { F } } ^ { n } )$ be $a$ martingale with $M ^ { 0 } = 0$ , increments $| M ^ { n } - M ^ { n - 1 } | \leq b$ , and accumulated quadratic variation

$$
V ^ { n } : = \sum _ { r = 1 } ^ { n } \mathbb { E } [ ( M ^ { r } - M ^ { r - 1 } ) ^ { 2 } \mid \mathcal { F } ^ { r - 1 } ] .
$$

Then, $f o r \ x , v > 0$

$$
\mathbb { P } \left( \operatorname* { m a x } _ { r \leq n } M ^ { r } \geq x , \ V ^ { n } \leq v \right) \leq \exp \left( - \frac { x ^ { 2 } } { 2 ( v + b x / 3 ) } \right) .
$$

Applying the result to M and −M gives the corresponding two-sided bound with a factor $2 .$

Fix a post-entry stopping time $t _ { 0 }$ with $S ^ { t _ { 0 } } \in \mathcal { R } ^ { * }$ and define

$$
\begin{array} { r } { \sigma _ { N } ( t _ { 0 } ) : = \operatorname* { i n f } \{ t \ge t _ { 0 } : H ^ { t } > H _ { \perp } \mathrm { ~ a n d ~ } \beta ^ { t } \notin \mathcal { N } _ { \delta } \} , } \end{array}
$$

with the convention inf $\mathcal { O } = \infty$ . Under the hypotheses of Lemma C.6, the stopped quantities in its proof use only regular rounds before $\sigma _ { N } ( t _ { 0 } )$ and above the fixed cutof, so every Poisson term is well defined by the one-step bufer.

Lemma C.6. Conditional on any history for which $\mathrm { F R ^ { g } }$ has not been analytically truncated before $t _ { 0 }$ and

$$
S ^ { t _ { 0 } } \in \mathcal { R } ^ { * } , \qquad H ^ { t _ { 0 } } > H _ { \perp } , \qquad \beta ^ { t _ { 0 } } \in \mathcal { N } _ { \delta / 4 } ,
$$

the following bound holds conditionally on $\mathcal { F } ^ { t _ { 0 } }$ for every integer $h \geq H _ { \perp }$

$$
\begin{array} { r } { \mathbb { P } ( \sigma _ { N } ( t _ { 0 } ) < \infty , \ H ^ { \sigma _ { N } ( t _ { 0 } ) } \geq h \mid \mathcal { F } ^ { t _ { 0 } } ) \leq \exp ( - \Omega ( h ) ) . } \end{array}\tag{C.22}
$$

Proof. Condition on a history in $\mathcal { F } ^ { t _ { 0 } }$ satisfying the stated hypotheses. The guarded comparison follows FR from $t _ { 0 }$ until its first subsequent neighborhood exit or fixed cutof, so Lemma C.4 applies to every increment used below. Fix a resource i and an integer $h \geq H _ { \perp }$ . If $h > H ^ { t _ { 0 } }$ , the event in (C.22) is empty, so assume $h \leq H ^ { t _ { 0 } }$ . Rounds whose increments can afect a state with remaining horizon at least h satisfy $H ^ { t } - 1 \geq h$ . Define the stopped martingale, for $n = t _ { 0 } - 1 , t _ { 0 } , \ldots , T - h$ ，

$$
M _ { i } ^ { n , ( h ) } : = \sum _ { \begin{array} { c } { { t = t _ { 0 } } } \end{array} } ^ { n \wedge ( \sigma _ { N } ( t _ { 0 } ) - 1 ) } { \frac { \xi _ { i } ^ { t + 1 } } { H ^ { t } - 1 } } .
$$

With respect to the shifted filtration $( \mathcal { F } ^ { n + 1 } )$ , this is a martingale because $\mathbf { 1 } \{ t _ { 0 } \leq t < \sigma _ { N } ( t _ { 0 } ) \}$ is $\mathcal { F } ^ { t }$ -measurable. Its increments are $O ( h ^ { - 1 } )$ , and its accumulated quadratic variation is $O ( h ^ { - 1 } )$ by computation. By (C.21), the cumulative remainder over the same rounds satisfies

$$
\sum _ { t = t _ { 0 } } ^ { ( T - h ) \wedge ( \sigma _ { N } ( t _ { 0 } ) - 1 ) } | \mathcal { R } _ { i } ^ { t + 1 } | = O ( h ^ { - 1 } ) .\tag{C.23}
$$

Finally,

$$
| W _ { i } ^ { t } - ( \beta _ { i } ^ { t } - \rho _ { i } ) | \leq \frac { V } { H ^ { t } } .\tag{C.24}
$$

For all suficiently large $h ,$ the Poisson correction, which is at most $2 V / h$ , and the $O ( h ^ { - 1 } )$ accumulated remainder sum to at most $\delta / 4$ . We first treat such $h .$

Binding resource. For $i \in I ^ { * }$ , (C.15) makes the second term in (C.19) zero. If this coordinate exits while $H ^ { \sigma _ { N } ( t _ { 0 } ) } \geq h$ , then

$$
| \beta _ { i } ^ { \sigma _ { N } ( t _ { 0 } ) } - \rho _ { i } | \geq \delta .
$$

Summing (C.19) from $t _ { 0 }$ to $\sigma _ { N } ( t _ { 0 } ) - 1$ and then using (C.24) and (C.23) together with the regularstart condition gives

$$
| W _ { i } ^ { t _ { 0 } } | \leq \frac \delta 4 + \frac { V } { H ^ { t _ { 0 } } }
$$

for binding coordinates. For all suficiently large $h ,$ the initial ofset, terminal Poisson correction, and accumulated remainder together use at most $\delta / 2$ of the available δ margin. Hence an exit with remaining horizon at least h implies

$$
\operatorname* { m a x } _ { n \leq T - h } | M _ { i } ^ { n , ( h ) } | \geq \frac \delta 2 .
$$

Fact C.5, with $x = \delta / 2$ and the preceding increment and variation bounds, gives

P(binding coordinate i exits while $H \geq h ) \leq \exp ( - \Omega ( h ) )$

Non-binding resource. For $i \notin I ^ { * }$ , only a lower-bound exit is possible. By (C.15),

$$
- \frac { \overline { { c } } _ { i } ( \beta ^ { t } ) - \beta _ { i } ^ { t } } { H ^ { t } - 1 } \geq \frac { \eta } { H ^ { t } - 1 } \geq 0 .
$$

Thus, after summing (C.19), dropping this nonnegative drift can only decrease the lower bound on $\boldsymbol { W } _ { i } ^ { t }$ :

$$
W _ { i } ^ { t } \geq W _ { i } ^ { t _ { 0 } } - M _ { i } ^ { t - 1 , ( h ) } - O ( h ^ { - 1 } ) .
$$

For all suficiently large h, if $\beta _ { i } ^ { t } \le \rho _ { i } - \delta$ and $H ^ { t } \geq h$ , then (C.24) implies

$$
M _ { i } ^ { t - 1 , ( h ) } \geq \frac { \delta } { 2 } .
$$

The one-sided Freedman bound gives

P(non-binding coordinate i exits while $H \geq h ) \leq \exp ( - \Omega ( h ) )$

A union bound over the finite resource set proves (C.22) for all suficiently large h. The finitely many remaining integers $h \geq H _ { \perp }$ are absorbed into the multiplicative constant implicit in (C.22).

## C.4.3 Finishing the proof for frequent re-solving

By Proposition 3.6, $s ^ { 1 } \in \mathcal { R } ^ { * }$ and the entrance phase is empty. Let

$$
\sigma : = \operatorname* { i n f } \{ t \geq 1 : H ^ { t } \leq H _ { \bot } \ \mathrm { o r } \ \beta ^ { t } \notin \mathcal { N } _ { \delta } \}
$$

be the first round in which the guarded comparison is truncated. Before $\sigma ,$ the current parameter lies in $\mathcal { N } _ { \delta }$ , and Proposition 3.5 keeps the state in $\mathcal { R } ^ { * }$ . Consequently, Lemma C.1 makes the rewardplus-potential increments martingale diferences up to σ. Conditional finite-horizon telescoping from 1 gives

$$
\begin{array} { r } { F ( H ^ { 1 } , B ^ { 1 } , S ^ { 1 } ) - V _ { T } ^ { \mathrm { F R } ^ { \mathrm { g } } } \leq \mathbb { E } \left[ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \mid \mathcal { F } ^ { \tau ^ { \mathrm { e n t } } } \right] . } \end{array}
$$

Whether the cutof or a neighborhood exit is reached, the preceding regular round and the one-step bufer give $\beta ^ { \sigma } \in \mathcal { N } _ { 2 \delta }$ . Thus the inventory is at most $H ^ { \sigma } ( \rho _ { i } + 2 \delta )$ for every $i \in I ^ { * }$ , while $\lambda _ { i } ^ { * } = 0$ for $i \notin I ^ { * }$ . The definition of $F$ therefore gives

$$
F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \leq C ( 1 + H ^ { \sigma } ) .
$$

At the cutof, $H ^ { \sigma } \leq H _ { \perp }$ , so this is a deterministic constant. On a neighborhood exit, the tail-sum

formula and Lemma C.6 give the required integrable bound: conditionally on ${ \mathcal { F } } ^ { 1 }$

$$
\mathbb { E } [ H ^ { \sigma } \mathbf { 1 } \{ \mathrm { n e i g h b o r h o o d ~ e x i t } \} \mid \mathcal { F } ^ { 1 } ] \leq H _ { \perp } + \sum _ { h = H _ { \perp } + 1 } ^ { \infty } \exp ( - \Omega ( h ) ) = O ( 1 ) ,
$$

uniformly over all regular entry histories. Consequently,

$$
\mathbb { E } \left[ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \right] = O ( 1 ) .
$$

Hence

$$
T G ( \rho ) - V _ { T } ^ { \mathrm { F R } ^ { \mathrm { g } } } = O ( 1 ) .
$$

## C.5 Infrequent re-solving analysis and proof

This section analyzes $\mathrm { I R ^ { g } }$ from Appendix C.1. By (C.3), the implemented guard-free IR policy earns at least as much reward on every coupled sample path, so all upper bounds transfer to IR.

## C.5.1 Uniform concentration within a frozen epoch

We use the following maximal bounded-increment inequality.

Fact C.7 (Maximal Azuma–Hoefding inequality, from Roch (2024)). $I f ( M ^ { \ell } ) _ { \ell = 0 } ^ { \Lambda }$ is a martingale with $M ^ { 0 } = 0$ and $| M ^ { \ell } - M ^ { \ell - 1 } | \le b$ almost surely, then

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 0 \leq \ell \leq \Lambda } | M ^ { \ell } | \geq x \right) \leq 2 \exp \left( - \frac { x ^ { 2 } } { 2 \Lambda b ^ { 2 } } \right) .
$$

Fix $\beta \in \mathcal { N } _ { 2 \delta }$ and run the stationary randomized policy $\pi _ { \beta }$ for Λ rounds from an arbitrary initial state in $\mathcal { R } ^ { * }$ . We have the following lemma. Here P denotes probability under this frozen policy; its stationary mean consumption of resource i is the already defined quantity $\overline { { c } } _ { i } ( \beta )$

Lemma C.8. For every resource i and $x \ge$ max{4V, 1},

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq \ell \leq \Lambda } \left| \sum _ { r = 1 } ^ { \ell } ( C _ { i } ^ { r } - \bar { c } _ { i } ( \beta ) ) \right| \geq x \right) \leq 2 \exp \left( - \frac { x ^ { 2 } } { 8 \Lambda ( \bar { c } + 2 V ) ^ { 2 } } \right) .
$$

Proof. Use the Poisson solution $v _ { i } ( \beta , \cdot )$ . Under the frozen policy, similarly define

$$
\xi _ { i } ^ { r + 1 } : = C _ { i } ^ { r } - g _ { i } ( \beta , S ^ { r } ) + v _ { i } ( \beta , S ^ { r + 1 } ) - \sum _ { s ^ { \prime } } K _ { \beta } ( S ^ { r } , s ^ { \prime } ) v _ { i } ( \beta , s ^ { \prime } ) .
$$

The conditional law of the request, action, and next state gives

$$
\mathbb { E } [ \xi _ { i } ^ { r + 1 } \mid \mathcal { F } ^ { r } ] = 0 , \qquad | \xi _ { i } ^ { r + 1 } | \leq \bar { c } + 2 V .
$$

The Poisson equation yields

$$
C _ { i } ^ { r } - \overline { { c } } _ { i } ( \beta ) = \xi _ { i } ^ { r + 1 } + v _ { i } ( \beta , S ^ { r } ) - v _ { i } ( \beta , S ^ { r + 1 } ) .\tag{C.25}
$$

Summing to a prefix ℓ gives

$$
\sum _ { r = 1 } ^ { \ell } ( C _ { i } ^ { r } - \overline { { c } } _ { i } ( \beta ) ) = \sum _ { r = 1 } ^ { \ell } \xi _ { i } ^ { r + 1 } + v _ { i } ( \beta , S ^ { 1 } ) - v _ { i } ( \beta , S ^ { \ell + 1 } ) .
$$

The last two terms have total absolute value at most 2V. Therefore, for $x \ge 4 V$ , a centered-cost deviation of size x implies a martingale deviation of size at least $x / 2$ . Fact C.7 gives

$$
\mathbb { P } \left( \operatorname* { m a x } _ { \ell \leq \Lambda } \left| \sum _ { r = 1 } ^ { \ell } ( C _ { i } ^ { r } - \overline { { c } } _ { i } ( \beta ) ) \right| \geq x \right) \leq 2 \exp \left( - \frac { x ^ { 2 } } { 8 \Lambda ( \overline { { c } } + 2 V ) ^ { 2 } } \right) .
$$

## C.5.2 Stability of infrequent resolving points

Choose κ such that

$$
0 < \kappa < 1 - \frac { 1 } { 2 \alpha } , \qquad \zeta : = 2 \alpha ( 1 - \kappa ) - 1 > 0 .\tag{C.26}
$$

For every epoch with $h _ { k + 1 } > H _ { \perp }$ , define

$$
d _ { k } : = h _ { k + 1 } ^ { - \kappa } .
$$

In the following, we temporarily write $\beta ^ { t _ { k } }$ as $\beta ^ { k }$ for brevity. Notice that the entrance policy is not invoked when the stationary LP has a unique nondegenerate optimum according to Proposition 3.6.

The IR epoch analysis uses two of the checks in the convention of Appendix C.1. The resource near-depletion check is the pre-action test

$$
\operatorname* { m i n } _ { i \in [ m ] } B _ { i } ^ { t } \leq \bar { c } .
$$

The guarded comparison stops before taking the next action when this check fires. It is not the actual depletion rule (1): since every coordinate of one-period consumption is at most ${ \overline { { c } } } ,$ actual depletion cannot occur before the near-depletion check fires.

For an epoch k satisfying the hypotheses below, condition on $\mathcal { F } ^ { t _ { k } }$ and couple the guarded comparison to a $f u l l$ frozen continuation that uses $\pi _ { \beta ^ { k } }$ for all $\Lambda _ { k }$ scheduled rounds. Use the same requests, action randomizations, and transition randomizations until the resource near-depletion check fires. Denote the hypothetical costs by $\widetilde { C } ^ { t }$ and define the hypothetical endpoint ratio by

$$
\widetilde { \beta } ^ { k + 1 } : = \beta ^ { k } + \frac { \Lambda _ { k } \beta ^ { k } - \sum _ { t = t _ { k } } ^ { t _ { k + 1 } - 1 } \widetilde { C } ^ { t } } { h _ { k + 1 } } .\tag{C.27}
$$

At the hypothetical endpoint, the epoch-movement check fails if

$$
\left| \widetilde { \beta } _ { i } ^ { k + 1 } - \beta _ { i } ^ { k } \right| > d _ { k } \quad \mathrm { f o r ~ s o m e ~ } i \in I ^ { * } , \quad \mathrm { o r } \quad \widetilde { \beta } _ { i } ^ { k + 1 } < \beta _ { i } ^ { k } - d _ { k } \quad \mathrm { f o r ~ s o m e ~ } i \notin I ^ { * } .
$$

Let $\mathcal { E } _ { k }$ be the event that either the epoch-movement check fails or the resource near-depletion check

fires during epoch k. Thus the two checks are distinct: the first controls movement of the endpoint ratio, while the second prevents resource depletion within an epoch.

Lemma C.9. Suppose the guarded comparison reaches $t _ { k }$ without truncation, uses the frozen policy $\pi _ { \beta ^ { k } }$ during epoch k, and satisfies $\beta ^ { k } \in \mathcal { N } _ { \delta }$ and $h _ { k + 1 } > H _ { \perp }$ . Then

$$
\mathbb { P } ( \mathcal { E } _ { k } \mid \mathcal { F } ^ { t _ { k } } ) \leq \exp ( - \Omega ( h _ { k } ^ { \zeta } ) ) .
$$

Proof. Condition on $\mathcal { F } ^ { t _ { k } }$ and use the full frozen continuation defined above. Until the resource neardepletion check fires, the actual and hypothetical trajectories agree. Hence, if neither component of $\mathcal { E } _ { k }$ occurs, the guarded comparison reaches the endpoint and its endpoint ratio equals $\widetilde { \beta } ^ { k + 1 }$

Binding coordinates. For $i \in I ^ { * } , \overline { { c } } _ { i } ( \beta ^ { k } ) = \beta _ { i } ^ { k }$ . Hence

$$
\widetilde { \beta } _ { i } ^ { k + 1 } - \beta _ { i } ^ { k } = - \frac { 1 } { h _ { k + 1 } } \sum _ { t = t _ { k } } ^ { t _ { k + 1 } - 1 } ( \widetilde { C } _ { i } ^ { t } - \bar { c } _ { i } ( \beta ^ { k } ) ) .
$$

Applying Lemma C.8 with

$$
x = d _ { k } h _ { k + 1 } = h _ { k + 1 } ^ { 1 - \kappa }
$$

gives

$$
\mathbb { P } \left( | \widetilde { \beta } _ { i } ^ { k + 1 } - \beta _ { i } ^ { k } | > d _ { k } \ | \ \mathcal { F } ^ { t _ { k } } \right) \leq \exp \left( - \Omega \left( \frac { h _ { k + 1 } ^ { 2 ( 1 - \kappa ) } } { \Lambda _ { k } } \right) \right) .
$$

Because $h _ { k + 1 } \geq h _ { k } ^ { \alpha }$ and $\Lambda _ { k } \leq h _ { k }$ , this is $\exp ( - \Omega ( h _ { k } ^ { \zeta } ) )$ .

Non-binding coordinates. For $i \notin { I ^ { * } , \beta _ { i } ^ { k } - \bar { c } _ { i } ( \beta ^ { k } ) } \ge \eta$ . (C.27) shows that

$$
\widetilde { \beta } _ { i } ^ { k + 1 } < \beta _ { i } ^ { k } - d _ { k }
$$

requires

$$
\sum _ { t = t _ { k } } ^ { t _ { k + 1 } - 1 } ( \widetilde { C } _ { i } ^ { t } - \overline { { c } } _ { i } ( \beta ^ { k } ) ) > d _ { k } h _ { k + 1 } + \eta \Lambda _ { k } .
$$

This threshold is at least $d _ { k } h _ { k + 1 }$ , so the same exponential estimate applies.

Resource near-depletion. Let

$$
\underline { { \beta } } : = \operatorname* { m i n } _ { i } ( \rho _ { i } - \delta ) > 0 .
$$

After a hypothetical prefix of length $\ell \leq \Lambda _ { k }$ , the deterministic inventory reference centered at the stationary mean satisfies

$$
\begin{array} { r l } { h _ { k } \beta _ { i } ^ { k } - \ell \overline { { c } } _ { i } ( \beta ^ { k } ) \geq ( h _ { k } - \ell ) \beta _ { i } ^ { k } } & { { } } \\ { \geq h _ { k + 1 } \underline { { \beta } } . } \end{array}
$$

The first inequality uses $\overline { { c } } _ { i } ( \beta ^ { k } ) \le \beta _ { i } ^ { k }$ , with equality for binding resources and strict slack for nonbinding resources. For each fixed prefix length, the conditional expected inventory of the hypothetical frozen-policy process can difer from this reference by at most 2V because of the Poisson boundary terms in (C.25). The pathwise near-depletion comparison below uses the stationary-centered cost sum, whose concentration bound already includes those terms. If the resource near-depletion check fires after ℓ preceding actions, the coupled hypothetical inventory at that prefix is at most c. Therefore

$$
\sum _ { r = 0 } ^ { \ell - 1 } ( \widetilde { C } _ { i } ^ { t _ { k } + r } - \bar { c } _ { i } ( \beta ^ { k } ) ) \geq h _ { k + 1 } \underline { { \beta } } - \bar { c }
$$

for some resource i. Enlarge $H _ { \perp }$ so that this threshold is at least $( \beta / 2 ) h _ { k + 1 }$ and at least max $\{ 4 V , 1 \}$ . Lemma C.8 then gives

$$
\mathbb { P } ( \mathrm { r e s o u r c e ~ n e a r - d e p l e t i o n ~ i n ~ e p o c h ~ } k \mid \mathcal { F } ^ { t _ { k } } ) \leq C \exp \left( - c \frac { h _ { k + 1 } ^ { 2 } } { \Lambda _ { k } } \right) \leq C \exp ( - c h _ { k } ^ { 2 \alpha - 1 } ) .
$$

Since $\zeta < 2 \alpha - 1$ , a union bound over the two components of $\mathcal { E } _ { k }$ and the finitely many resources proves the lemma. □

In the following lemma, with minor abuse of notation, we use K to denote the first schedule index with $h _ { K } \leq H _ { \perp }$ . Recall that it originally refers to the index of the final re-solving epoch of the non-truncated IR.

Lemma C.10. The cutof $H _ { \perp }$ can be enlarged so that, for every $T _ { \cdot }$

$$
\sum _ { k = 0 } ^ { K - 2 } d _ { k } < \frac { \delta } { 2 } ,\tag{C.28}
$$

with an empty sum when $K \leq 1$ . More generally, fix any $k _ { 0 } \in \{ 0 , \ldots , K - 1 \} . \ I f \ \beta ^ { k _ { 0 } } \in \mathcal { N } _ { \delta / 4 }$ and none of the events $\mathcal { E } _ { k }$ defined before Lemma $C . 9$ occurs for $k = k _ { 0 } , \ldots , K - 2$ , then every resolving ratio $\beta ^ { k } , k _ { 0 } \le k \le K - 1$ , lies in $\mathcal { N } _ { \delta }$

Proof. Choose any number $\alpha ^ { \prime }$ with $\alpha < \alpha ^ { \prime } < 1$ . Enlarge $H _ { \perp }$ so that

$$
\lceil h ^ { \alpha } \rceil \leq h ^ { \alpha ^ { \prime } } \qquad ( h > H _ { \bot } ) .\tag{C.29}
$$

If $K \leq 1$ , (C.28) is an empty sum and the propagation assertion is immediate. For the movement-sum argument, assume $K \geq 2$ . For $k \le K - 2$ , both $h _ { k }$ and $h _ { k + 1 }$ exceed $H _ { \perp }$ , and (C.29) implies

$$
h _ { k } ^ { - \kappa } \leq h _ { k + 1 } ^ { - \kappa / \alpha ^ { \prime } } .
$$

Let $x : = h _ { K - 1 } ^ { - \kappa } \leq H _ { \perp } ^ { - \kappa } < 1$ . Reading the sequence backward gives

$$
h _ { K - 2 } ^ { - \kappa } \leq x ^ { 1 / \alpha ^ { \prime } } , \quad h _ { K - 3 } ^ { - \kappa } \leq x ^ { 1 / ( \alpha ^ { \prime } ) ^ { 2 } } , \quad \ldots .
$$

The ratio of each successive term in this backward list to the preceding one is at most $x ^ { 1 / \alpha ^ { \prime } - 1 } \leq$ $H _ { \mathrm { ~ | ~ } } ^ { - \kappa ( 1 / \alpha ^ { \prime } - 1 ) }$ . Enlarge $H _ { \perp }$ so that this bound is at most $1 / 2$ . Then

$$
\sum _ { k = 0 } ^ { K - 2 } d _ { k } = \sum _ { r = 1 } ^ { K - 1 } h _ { r } ^ { - \kappa } \leq 2 x \leq 2 H _ { \perp } ^ { - \kappa } .
$$

A final enlargement makes this smaller than $\delta / 2$

If $\beta ^ { k _ { 0 } } \in \mathcal { N } _ { \delta / 4 }$ , then, on the complement of all bad events, every binding coordinate changes by at most $d _ { k }$ in absolute value during epoch $k ,$ and every non-binding coordinate decreases by at most $d _ { k }$ . The tail sum $\textstyle \sum _ { k = k _ { 0 } } ^ { K - 2 } d _ { k }$ is bounded by the full sum in (C.28). Induction therefore gives

$$
| \beta _ { i } ^ { k } - \rho _ { i } | < 3 \delta / 4 \quad ( i \in I ^ { * } ) , \qquad \beta _ { i } ^ { k } > \rho _ { i } - 3 \delta / 4 \quad ( i \notin I ^ { * } )
$$

for $k _ { 0 } \leq k \leq K - 1$ . Hence all these average capacities lie in $\mathcal { N } _ { \delta }$

## C.5.3 Finishing the proof for infrequent re-solving

Proof. By Proposition 3.6, $s ^ { 1 } \in \mathcal { R } ^ { * }$ and the entrance phase is empty. First handle $h _ { 1 } \leq H _ { \perp }$ . By the clipped recursion, this implies $h _ { 0 } \leq \operatorname* { m a x } \{ H _ { \perp } + 1 , H _ { \perp } ^ { 1 / \alpha } \}$ , a fixed constant. The inventory at entry is at most $h _ { 0 } ( \rho _ { i } + \delta / 4 )$ for every $i \in I ^ { * }$ and can only decrease. Since $\lambda _ { i } ^ { * } = 0$ for $i \notin I ^ { * }$ , the definition of F therefore bounds every later positive potential by $C ( 1 + h _ { 0 } ) = O ( 1 )$ , so nonnegative continuation rewards sufice for this case. We may therefore assume $h _ { 1 } > H _ { \perp }$ , and hence $K \geq 2$

At $t _ { 0 } = 1$ , the epoch-0 ratio is $B ^ { 1 } / h _ { 0 } = \rho ,$ and epoch 0 uses the corresponding frozen policy $\pi _ { \rho }$ Thus Lemma C.9 applies also to epoch 0. For $k = 0 , \ldots , K - 2$ , use the bad event $\mathcal { E } _ { k }$ defined before Lemma C.9, and define

$$
N : = \operatorname* { i n f } \{ k \in \{ 0 , \ldots , K - 2 \} : \mathcal { E } _ { k } \mathrm { ~ o c c u r s } \} ,
$$

with $N = \infty$ if no such event occurs. Let $\sigma$ be the round in which the guarded comparison is first truncated by the locality, resource near-depletion, or cutof rule specified in Appendix C.1. Every pre-σ action in epoch 0 uses $\pi _ { \rho } ,$ and every later pre-σ action uses a frozen policy $\pi _ { \beta ^ { k } }$ with parameter in ${ \mathcal { N } } _ { \delta } ;$ Proposition 3.5 keeps the state in $\mathcal { R } ^ { * }$ . Consequently, Lemma C.1 makes the reward-plus-potential increments martingale diferences up to $\sigma .$ Finite-horizon telescoping from round 1, followed by the zero-reward null actions after truncation and deletion of the negative part of the terminal potential, gives

$$
\begin{array} { r } { F ( H ^ { 1 } , B ^ { 1 } , S ^ { 1 } ) - V _ { T } ^ { \mathrm { I R } ^ { \mathrm { g } } } \leq \mathbb E \left[ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \right] . } \end{array}
$$

On $\{ N = \infty \}$ , Lemma C.10 with $k _ { 0 } = 0$ shows that all subsequent pre-cutof resolving ratios remain stable and the last epoch before the proof cutof begins with $h _ { K - 1 } \leq H _ { \perp } ^ { 1 / \alpha }$ . At the start of that last epoch, the inventory is at most $h _ { K - 1 } ( \rho _ { i } + \delta )$ for every $i \in I ^ { * }$ and can only decrease. Since $\lambda _ { i } ^ { * } = 0$ for $i \notin I ^ { * }$ , the definition of $F$ and the displayed bound on $h _ { K - 1 }$ give a deterministic constant bound on the residual at truncation. On $\{ N = k \}$ , no earlier bad event occurred, so $\beta ^ { k } \in \mathcal { N } _ { \delta }$ , and the same argument from the regular start of epoch k gives

$$
F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \leq C ( 1 + h _ { k } ) .
$$

Applying Lemma C.9 conditionally on $\mathcal { F } ^ { t _ { k } }$ and then using the tower property gives

$$
\mathbb { P } ( N = k ) \le \exp \bigl ( - \Omega ( h _ { k } ^ { \zeta } ) \bigr ) .
$$

Therefore,

$$
\mathbb { E } [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } ] \leq C + C \sum _ { k = 0 } ^ { K - 2 } ( 1 + h _ { k } ) \exp ( - c h _ { k } ^ { \zeta } ) .
$$

The last sum is uniformly bounded in $T .$ . Indeed, before the cutof the sequence $\left( h _ { k } \right)$ is a strictly decreasing sequence of distinct integers larger than $H _ { \perp }$ . Hence

$$
\sum _ { k = 0 } ^ { K - 2 } ( 1 + h _ { k } ) \exp ( - c h _ { k } ^ { \zeta } ) \leq \sum _ { h = H _ { \perp } + 1 } ^ { \infty } ( 1 + h ) \exp ( - c h ^ { \zeta } ) < \infty ,
$$

where the right-hand side depends only on the fixed instance constants, $H _ { \perp }$ , and $\zeta > 0$ . Thus

$$
\mathbb { E } [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } ] = O ( 1 ) .
$$

Hence

$$
T G ( \rho ) - V _ { T } ^ { \mathrm { I R ^ { g } } } = O ( 1 ) .
$$

## D Incomplete regularity analysis and proofs

## D.1 Proof of Theorem 5.2

Degeneracy creates two dificulties that were absent from the constant-regret proof. First, a nearby LP may have several optimizers, so there need not be a locally fixed basis. Second, a nearby optimizer may assign positive stationary state probability to states outside the optimal target state class. The analysis below handles these dificulties separately: uniqueness gives a selector-uniform Lipschitz bound on every nearby optimizer, and stationary domination converts the small total stationary probability outside the optimal target state class into a small finite-time probability of visiting those states.

Use the unique-optimum notation ${ { x } ^ { * } } , { { I } ^ { * } } , { { \mathcal { R } } ^ { * } } , { { \mathcal { E } } ^ { * } } , { { \sigma } ^ { * } }$ , and $\Delta$ introduced in (8) and (9), together with the one-sided inventory-deviation norm D defined in (C.9).

The proof of Theorem 5.2 uses the following lemmas. Lemma D.1 is the only step that uses nominal uniqueness; it replaces local basis stability by an upper-Lipschitz estimate. Lemma D.2 shows that states outside the optimal target state class remain rare throughout a frozen epoch. Lemma D.3 then gives a maximal additive-functional estimate without requiring the perturbed stationary state distribution to be supported on the optimal target state class. Finally, Lemma D.4 propagates inventory deviations through a fixed number of epochs, by which time the remaining horizon is $O ( \sqrt { T } )$ . The theorem then follows from one stopped dual-potential identity.

For brevity, we define $\mathcal { O } : = \mathcal { S } \setminus \mathcal { R } ^ { * }$ in this proof.

## D.1.1 Lipschitz continuity of optimizers

Lemma D.1. There is a radius $\delta _ { 0 } > 0$ such that, whenever $\Delta ( \beta ) \leq \delta _ { 0 }$ , every optimizer $x ( \beta )$ of $G ( \beta )$ satisfies

$$
\| x ( \beta ) - x ^ { * } \| _ { 1 } = O ( \Delta ( \beta ) ) .\tag{D.1}
$$

In particular, there is $\eta > 0$ such that every selected local optimizer satisfies

$$
q _ { s } ( \beta ) \geq \eta ,
$$

$$
s \in \mathcal { R } ^ { * } ,\tag{D.2}
$$

$$
\sum _ { s \in \mathcal { O } } q _ { s } ( \beta ) = O ( \Delta ( \beta ) ) ,\tag{D.3}
$$

$$
\| \pi _ { \beta } ( \cdot \mid s , j ) - \pi _ { \rho } ( \cdot \mid s , j ) \| _ { 1 } = O ( \Delta ( \beta ) ) ,
$$

$$
s \in { \mathcal { R } } ^ { * } , ~ j \in { \mathcal { I } } .\tag{D.4}
$$

Proof. For a stationary flow $x = ( z , q )$ , with minor abuse of notation, abbreviate its reward and its rate of using resource i by

$$
\overline { { r } } ( x ) : = \sum _ { s , j , a } r ( s , j , a ) z _ { s j a } , \qquad \overline { { c } } _ { i } ( x ) : = \sum _ { s , j , a } c _ { i } ( s , j , a ) z _ { s j a } .
$$

Also set $\overline { { r } } ^ { * } : = \overline { { r } } ( x ^ { * } )$

Step 1: one fixed comparison polytope. Let ${ \mathcal { P } } ^ { + }$ be the fixed polytope of nonnegative stationary

flows satisfying (SF1)–(SF3) and only the nominally binding resource constraints

$$
\begin{array} { r } { \overline { c } _ { i } ( x ) \leq \rho _ { i } , \qquad i \in I ^ { * } ; } \end{array}
$$

the inactive resource constraints are omitted. This polytope is compact: (SF1)–(SF3) imply $\begin{array} { r } { \sum _ { s , j , a } z _ { s j a } = \sum _ { s } q _ { s } = 1 } \end{array}$

We first show that $x ^ { * }$ is the unique reward maximizer over ${ \mathcal { P } } ^ { + }$ . Take any $y \in { \mathcal { P } } ^ { + }$ with $y \neq x ^ { * }$ Since there are finitely many inactive resources and each has nominal slack $\sigma _ { i } ^ { * } > 0$ , one can choose a single $0 < \varepsilon < 1$ so small that

$$
\varepsilon \big ( \overline { { c } } _ { i } ( y ) - \overline { { c } } _ { i } ( x ^ { * } ) \big ) _ { + } \leq \sigma _ { i } ^ { * } , \qquad i \notin I ^ { * } .\tag{D.5}
$$

The mixture

$$
x _ { \varepsilon } : = ( 1 - \varepsilon ) x ^ { * } + \varepsilon y
$$

is then feasible for the nominal LP. The binding constraints hold by convexity because both endpoints satisfy them, while for $i \notin I ^ { * }$ 2

$$
\bar { c } _ { i } ( x _ { \varepsilon } ) \leq \bar { c } _ { i } ( x ^ { * } ) + \varepsilon \big ( \bar { c } _ { i } ( y ) - \bar { c } _ { i } ( x ^ { * } ) \big ) _ { + } \leq \rho _ { i }
$$

by (D.5). Consequently, $\overline { { r } } ( y ) > \overline { { r } } ( x ^ { * } )$ would produce a strictly better nominal feasible solution, and equality would produce the distinct nominal optimizer $x _ { \varepsilon }$ . Both contradict nominal uniqueness. Thus

$$
\begin{array} { r } { \overline { { r } } ( y ) < \overline { { r } } ( x ^ { * } ) \qquad \mathrm { f o r ~ e v e r y } ~ y \in \mathcal { P } ^ { + } \setminus \{ x ^ { * } \} . } \end{array}
$$

A unique maximizer of a linear objective over a polytope is a vertex. Because ${ \mathcal { P } } ^ { + }$ has finitely many vertices, the strict separation above can be made quantitative. If ${ \mathcal { P } } ^ { + } \neq \{ x ^ { * } \}$ , define

$$
\chi : = \operatorname* { m i n } _ { \begin{array} { c } { v \in \mathrm { v e r t } ( \mathcal { P } ^ { + } ) } \\ { v \neq x ^ { * } } \end{array} } \frac { \overline { { r } } ^ { * } - \overline { { r } } ( v ) } { \| v - x ^ { * } \| _ { 1 } } > 0 .
$$

If $\begin{array} { r } { y = \sum _ { v } \lambda _ { v } v \in \mathcal { P } ^ { + } } \end{array}$ is any vertex decomposition, then

$$
\begin{array} { r l } & { \overline { { r } } ^ { * } - \overline { { r } } ( y ) = \displaystyle \sum _ { v } \lambda _ { v } \big ( \overline { { r } } ^ { * } - \overline { { r } } ( v ) \big ) } \\ & { \qquad \quad \geq \chi \displaystyle \sum _ { v } \lambda _ { v } \| v - x ^ { * } \| _ { 1 } } \\ & { \qquad \quad \geq \chi \| y - x ^ { * } \| _ { 1 } . } \end{array}\tag{D.6}
$$

Step 2: move a nearby optimizer into the fixed polytope. Set

$$
d : = \Delta ( \beta ) , \qquad \underline { { \rho } } : = \operatorname* { m i n } _ { i \in [ m ] } \rho _ { i } > 0 , \qquad \theta : = d / \underline { { \rho } } .
$$

We will shrink $\delta _ { 0 }$ a few times. First choose it so that $\delta _ { 0 } \le \rho / 2$ and $\delta _ { 0 } \leq \sigma _ { i } ^ { * } / 2$ for every $i \notin I ^ { * }$ and suppose $d \leq \delta _ { 0 }$ . The second condition is vacuous if every resource is binding. To make the

comparison flow explicit, let $\mu$ be any stationary distribution of the finite-state transition matrix induced by the null action, and define $x ^ { \mathrm { n u l l } } = ( z ^ { 0 } , q ^ { 0 } )$ by

$$
q _ { s } ^ { 0 } = \mu _ { s } , \qquad z _ { s j a } ^ { 0 } = p _ { j } \mu _ { s } { \bf 1 } \{ a = a ^ { 0 } \} .
$$

Then $x ^ { \mathrm { n u l l } }$ satisfies (SF1)–(SF3) and has zero reward and zero resource use.

First mix the nominal optimizer with $x ^ { \mathrm { n u l l } }$

$$
\widetilde { \boldsymbol { x } } : = ( 1 - \theta ) \boldsymbol { x } ^ { * } + \theta \boldsymbol { x } ^ { \mathrm { n u l l } } .
$$

This flow is feasible at $\beta .$ For a binding resource,

$$
\overline { { c } } _ { i } ( \widetilde { x } ) = ( 1 - \theta ) \rho _ { i } \leq \rho _ { i } - d \leq \beta _ { i } ,
$$

because $\theta \rho _ { i } \geq d .$ . For an inactive resource,

$$
\begin{array} { r } { \overline { { c } } _ { i } ( \widetilde { x } ) \leq \rho _ { i } - \sigma _ { i } ^ { * } \leq \rho _ { i } - d \leq \beta _ { i } . } \end{array}
$$

Thus every optimizer $x ( \beta )$ satisfies

$$
\begin{array} { r } { \overline { r } ( x ( \beta ) ) \geq \overline { r } ( \widetilde x ) = ( 1 - \theta ) \overline { r } ^ { * } . } \end{array}\tag{D.7}
$$

Now apply the same mixture to an arbitrary perturbed optimizer:

$$
\widehat { x } : = ( 1 - \theta ) x ( \beta ) + \theta x ^ { \mathrm { n u l l } } .
$$

For every $i \in I ^ { * }$ , feasibility at $\beta$ and the definition of $d$ give

$$
\overline { { c } } _ { i } ( \widehat { x } ) \leq ( 1 - \theta ) ( \rho _ { i } + d ) = \rho _ { i } + ( d - \theta \rho _ { i } ) - \theta d \leq \rho _ { i } ,
$$

where the last inequality again uses $\theta \rho _ { i } \geq d .$ . Hence $\widehat { \boldsymbol { x } } \in \mathcal { P } ^ { + }$ . By (D.7),

$$
\begin{array} { r } { \overline { r } ( \widehat x ) = ( 1 - \theta ) \overline { r } ( x ( \beta ) ) \geq ( 1 - \theta ) ^ { 2 } \overline { r } ^ { * } . } \end{array}
$$

Since $\widehat { \boldsymbol { x } } \in \mathcal { P } ^ { + }$ and $x ^ { * }$ maximizes reward there,

$$
0 \leq \bar { r } ^ { * } - \bar { r } ( \widehat { x } ) \leq \big ( 1 - ( 1 - \theta ) ^ { 2 } \big ) \bar { r } ^ { * } \leq 2 { \theta } \bar { r } ^ { * } .
$$

Therefore, if ${ \mathcal { P } } ^ { + } \neq \{ x ^ { * } \}$ , the sharpness bound (D.6) yields

$$
\| \widehat { x } - x ^ { * } \| _ { 1 } \leq \frac { 2 \overline { r } ^ { * } } { \chi } \theta .\tag{D.8}
$$

If $\mathcal { P } ^ { + } = \{ x ^ { * } \}$ , the left-hand side is instead zero.

Finally, every stationary flow $( z , q )$ has $\| z \| _ { 1 } = \| q \| _ { 1 } = 1$ . Thus, for the concatenated $\ell _ { 1 } { \mathrm { - n o r m } }$

used in the lemma, every stationary flow has norm two, and

$$
\| x ( \beta ) - \widehat { x } \| _ { 1 } = \theta \| x ( \beta ) - x ^ { \mathrm { n u l l } } \| _ { 1 } \leq 4 \theta .
$$

Combining this estimate with (D.8) gives, in the nonsingleton case,

$$
\| x ( \beta ) - x ^ { * } \| _ { 1 } \leq \left( 4 + \frac { 2 \overline { { r } } ^ { * } } { \chi } \right) \frac { d } { \underline { { \rho } } } .
$$

In the singleton case the same bound holds with the second term in parentheses omitted. Thus there is a constant $K < \infty$ , depending only on the nominal problem, such that

$$
\| x ( \beta ) - x ^ { * } \| _ { 1 } \leq K d .\tag{D.9}
$$

This proves (D.1), uniformly over every choice of the perturbed optimizer.

Step 3: state masses and policies. The set $\mathcal { R } ^ { * }$ is nonempty because the nominal state masses sum to one. Define

$$
q _ { \mathrm { m i n } } ^ { * } : = \operatorname* { m i n } _ { s \in \mathcal { R } ^ { * } } q _ { s } ^ { * } > 0 , \qquad \eta : = q _ { \mathrm { m i n } } ^ { * } / 2 .
$$

Shrink $\delta _ { 0 }$ once more so that $K \delta _ { 0 } \leq q _ { \mathrm { m i n } } ^ { * } / 2$ . Then, for every $s \in \mathcal { R } ^ { * }$ , (D.9) gives

$$
q _ { s } ( \beta ) \geq q _ { s } ^ { * } - K d \geq q _ { \operatorname* { m i n } } ^ { * } / 2 = \eta ,
$$

proving (D.2). On the other hand, $q _ { s } ^ { * } = 0$ for $s \in \mathcal { O }$ , so

$$
\sum _ { s \in \mathcal { O } } q _ { s } ( \beta ) \leq \| q ( \beta ) - q ^ { * } \| _ { 1 } \leq K d ,
$$

which proves (D.3).

It remains to compare the induced policies. Set $p _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { j \in \mathcal { I } } p _ { j } > 0$ . Fix $s \in \mathcal { R } ^ { * }$ and $j \in \mathcal { I }$ and abbreviate, for the following calculation,

$$
z _ { a } = z _ { s j a } ( \beta ) , \quad z _ { a } ^ { * } = z _ { s j a } ^ { * } , \quad q = q _ { s } ( \beta ) , \quad q ^ { * } = q _ { s } ^ { * } , \quad p = p _ { j } .
$$

Since $q \geq \eta$ and $\textstyle \sum _ { a } z _ { a } ^ { * } = p q ^ { * }$ by (SF1),

$$
\begin{array} { r l } & { \| \pi _ { \beta } ( \cdot \mid s , j ) - \pi _ { \rho } ( \cdot \mid s , j ) \| _ { 1 } = \displaystyle \sum _ { a } \left| \frac { z _ { a } } { p q } - \frac { z _ { a } ^ { * } } { p q ^ { * } } \right| } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \leq \displaystyle \frac { \sum _ { a } | z _ { a } - z _ { a } ^ { * } | } { p q } + \frac { \sum _ { a } z _ { a } ^ { * } } { p } \left| \frac { 1 } { q } - \frac { 1 } { q ^ { * } } \right| } \\ & { \quad \quad \quad \quad \quad \quad = \displaystyle \frac { \sum _ { a } | z _ { a } - z _ { a } ^ { * } | } { p q } + \frac { | q - q ^ { * } | } { q } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \leq \left( \displaystyle \frac { 1 } { p _ { \operatorname* { m i n } } \eta } + \displaystyle \frac { 1 } { \eta } \right) K d . } \end{array}
$$

This is (D.4). Every constant above depends only on the fixed nominal problem, so all bounds are

uniform over every optimizer.

## D.1.2 Rare opportunity to escape the optimal target state class

Lemma D.2. Fix $\beta$ with $\Delta ( \beta ) \le \delta _ { 0 }$ , and run the frozen selected policy $\pi _ { \beta }$ . If the initial state distribution $\mu$ is supported on $\mathcal { R } ^ { * }$ , then for every $n \geq 0$ 2

$$
\mathbb { P } ( S ^ { n } \in { \mathcal { O } } ) = O ( \Delta ( \beta ) ) .
$$

Consequently, for every $\Lambda \geq 1$

$$
\mathbb { E } \left[ \sum _ { n = 0 } ^ { \Lambda - 1 } \mathbf { 1 } \{ S ^ { n } \in { \mathcal { O } } \} \right] = O ( \Lambda \Delta ( \beta ) ) .\tag{D.10}
$$

Proof. Let $K _ { \beta }$ be the kernel induced by the frozen policy and let $q ( \beta )$ be its stationary stateprobability vector supplied by the selected fluid optimizer. Since $\mu$ is supported on $\mathcal { R } ^ { * }$ and $q _ { s } ( \beta ) \geq \eta$ there,

$$
\mu \leq \eta ^ { - 1 } q ( \beta ) .
$$

This is a componentwise inequality of nonnegative row vectors. Multiplication by the nonnegative matrix $K _ { \beta } ^ { n }$ preserves it, and stationarity gives $q ( \beta ) K _ { \beta } ^ { n } = q ( \beta )$ . Therefore

$$
\mathbb { P } ( S ^ { n } \in \mathcal { O } ) = \mu K _ { \beta } ^ { n } ( \mathcal { O } ) \le \eta ^ { - 1 } q ( \beta ) ( \mathcal { O } ) = O ( \Delta ( \beta ) )
$$

by (D.3). Summing over n proves the second claim.

## D.1.3 Bounding the accumulated gap

Lemma D.3. Let ${ \mathcal { F } } ^ { n }$ contain the full history before the request in post-entry round $n .$ . Let $Y ^ { n }$ be a uniformly bounded, ${ \mathcal { F } } ^ { n + 1 }$ -measurable one-period quantity under a frozen policy $\pi _ { \beta }$ . Assume there is a state function $f _ { \beta }$ such that, for every $n ,$

$$
\mathbb { E } [ Y ^ { n } \mid { \mathcal { F } } ^ { n } ] = f _ { \beta } ( S ^ { n } ) .
$$

In particular, current-round reward and resource consumption satisfy this full-history conditionalmean condition. Suppose that there is a bounded nominal function f<sup>∗</sup> on $\mathcal { R } ^ { * }$ such that, uniformly over $s \in \mathcal { R } ^ { * }$ ，

$$
| f _ { \beta } ( s ) - f ^ { * } ( s ) | = O ( \Delta ( \beta ) ) .\tag{D.11}
$$

Set $\begin{array} { r } { \overline { f } ^ { * } : = \sum _ { s \in \mathcal { R } ^ { * } } q _ { s } ^ { * } f ^ { * } ( s ) } \end{array}$ . If the epoch starts in $\mathcal { R } ^ { * }$ , then

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 1 \leq n \leq \Lambda } \left| \sum _ { r = 0 } ^ { n - 1 } ( Y ^ { r } - \overline { { f } } ^ { * } ) \right| \right] = O \left( \sqrt { \Lambda } + \Lambda \Delta ( \beta ) + 1 \right) .\tag{D.12}
$$

Proof. If $\Delta ( \beta ) > \delta _ { 0 }$ , uniform boundedness of $Y ^ { n }$ and $\overline { { f } } ^ { * }$ bounds the left-hand side of (D.12) by

$O ( \Lambda ) = O ( \Lambda \Delta ( \beta ) )$ . It therefore sufices to consider $\Delta ( \beta ) \leq \delta _ { 0 }$ , where the local estimates below apply.

Step 1: use a corrector for the nominal chain. Assumption 3.4 makes the optimal induced kernel $K ^ { * }$ irreducible on $\mathcal { R } ^ { * }$ . Therefore the centered Poisson equation

$$
( I - K ^ { * } ) v = f ^ { * } - { \overline { { f } } } ^ { * } \mathbf { 1 } , \qquad ( q ^ { * } ) ^ { \top } v = 0
$$

has a bounded solution. Extend v by zero on $\mathcal { O } .$

Step 2: isolate a bounded-increment martingale. Define

$$
\begin{array} { r } { M ^ { r + 1 } : = Y ^ { r } - f _ { \beta } ( S ^ { r } ) + v ( S ^ { r + 1 } ) - ( K _ { \beta } v ) ( S ^ { r } ) . } \end{array}
$$

Then $\mathbb { E } [ M ^ { r + 1 } \mid \mathcal { F } ^ { r } ] = 0$ , and the increments are uniformly bounded. Doob’s square-integrable maximal inequality gives

$$
\mathbb { E } \operatorname* { m a x } _ { n \leq \Lambda } \left| \sum _ { r = 0 } ^ { n - 1 } M ^ { r + 1 } \right| = O ( { \sqrt { \Lambda } } ) .\tag{D.13}
$$

Step 3: bound the remainder. Set

$$
\begin{array} { r } { e _ { \beta } ( s ) : = f _ { \beta } ( s ) - \overline { { f } } ^ { * } - v ( s ) + ( K _ { \beta } v ) ( s ) . } \end{array}
$$

For $s \in \mathcal { R } ^ { * }$ , the nominal Poisson equation gives

$$
e _ { \beta } ( s ) = [ f _ { \beta } ( s ) - f ^ { * } ( s ) ] + [ ( K _ { \beta } - K ^ { * } ) v ] ( s ) .
$$

The first term is $O ( \Delta ( \beta ) )$ by (D.11); the second is of the same order by (D.4). Hence

$$
| e _ { \beta } ( s ) | = O ( \Delta ( \beta ) ) , \qquad s \in \mathcal { R } ^ { * } .
$$

On ${ \mathcal { O } } _ { : }$ , boundedness of $Y ^ { n }$ and v gives $| e _ { \beta } ( s ) | = O ( 1 )$ . The exact pathwise decomposition is

$$
\begin{array} { l } { { \displaystyle \sum _ { r = 0 } ^ { n - 1 } ( Y ^ { r } - \overline { { f } } ^ { * } ) = \sum _ { r = 0 \atop r } ^ { n - 1 } M ^ { r + 1 } + v ( S ^ { 0 } ) - v ( S ^ { n } ) } } \\ { { \displaystyle \qquad + \sum _ { r = 0 } ^ { n - 1 } e _ { \beta } ( S ^ { r } ) . } } \end{array}
$$

Consequently,

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { n \leq \Lambda } \left. \sum _ { r = 0 } ^ { n - 1 } ( Y ^ { r } - \overline { f } ^ { * } ) \right. \leq \displaystyle \operatorname* { m a x } _ { n \leq \Lambda } \left. \sum _ { r = 0 } ^ { n - 1 } M ^ { r + 1 } \right. + 2 \| v \| _ { \infty } } & { } \\ { \displaystyle + O ( \Lambda \Delta ( \beta ) ) + O \left( \sum _ { r = 0 } ^ { \Lambda - 1 } \mathbf { 1 } \{ S ^ { r } \in \mathcal O \} \right) . } \end{array}
$$

Taking expectations and applying (D.10) and (D.13) proves (D.12).

## D.1.4 Stopping control

Before proceeding to the next lemma, we make the following definitions. The successful-entrance event is

$$
\mathcal { E } _ { T } : = \{ N ^ { \mathrm { e n t } } < T \} \cap \bigcap _ { i \in [ m ] } \{ C _ { i } ^ { \mathrm { e n t } } < T \rho _ { i } \} .
$$

Thus, on $\mathcal { E } _ { T }$ , at least one round and a strictly positive amount of every resource remain at entrance. On this event set

$$
t _ { 0 } : = \tau ^ { \mathrm { e n t } } , \qquad h _ { 0 } : = H ^ { \mathrm { e n t } } , \qquad \beta ^ { 0 } : = \frac { B ^ { \mathrm { e n t } } } { h _ { 0 } } .
$$

The value of $\beta ^ { 0 }$ of $\mathcal { E } _ { T }$ is immaterial; set it equal to $\rho$ there. On $\mathcal { E } _ { T } ,$ restart the clipped power schedule (12) from $h _ { 0 }$ . With minor abuse of notation, let K be the first schedule index for which $h _ { K } \leq H _ { \perp }$ . Set

$$
J _ { \operatorname* { m a x } } : = \operatorname* { m i n } \{ k \geq 1 : \alpha ^ { k } \leq 1 / 2 \} , \qquad J : = \operatorname* { m i n } \{ K , J _ { \operatorname* { m a x } } \} .
$$

The restarted schedule and J are determined by the entrance history, whereas $J _ { \mathrm { m a x } }$ is a deterministic constant.

Define the regular-entry event by

$$
\mathcal { G } _ { 0 } : = \mathcal { E } _ { T } \cap \{ \Delta ( \beta ^ { 0 } ) < \delta _ { 0 } \} .
$$

On $\mathcal { G } _ { 0 }$ , let $\sigma$ be the first of the following times:

(a) a resolving time $t _ { k } , 1 \le k < J ,$ at which either the pre-action locality check $\Delta ( \beta ^ { k } ) < \delta _ { 0 }$ or the pre-action target-state check $S ^ { t _ { k } } \in \mathcal { R } ^ { * }$ fails;

(b) the first pre-action round t in an epoch $k < J$ at which the resource near-depletion check $\mathrm { m i n } _ { i \in [ m ] } B _ { i } ^ { t } \leq \bar { c }$ fires;

(c) the scheduled cutof $t _ { J } .$ , which is fixed at entry.

For each fixed $0 \leq k \leq J _ { \operatorname* { m a x } }$ , define $\mathcal { R } _ { k }$ as follows. On $\mathcal { G } _ { 0 } \cap \left\{ k \le J \right\}$ , it is the event $\{ \sigma \geq t _ { k } \}$ ; of that event, set $\mathbf { 1 } \{ \mathcal { R } _ { k } \} = 0$ . Thus $\mathcal { R } _ { k }$ means that the analysis reaches $t _ { k }$ before applying the checks there.

For every epoch $k < J$ that begins with a local parameter and a state in $\mathcal { R } ^ { * }$ , couple the actual path to its full frozen continuation: this continuation uses the policy assigned to epoch k for all $\Lambda _ { k }$ scheduled rounds, even if the resource near-depletion check would stop the actual path. Let $C _ { k , i } ( r )$ be its cumulative consumption of resource i in the first r rounds, write $C _ { k } ( \boldsymbol { r } )$ for the vector of these totals, and let $S ^ { t _ { k } }$ <sup>+r,hyp</sup> be its state after those rounds. The actual and hypothetical paths agree until resource near-depletion activation. At the hypothetical endpoint define

$$
X ^ { k + 1 , \mathrm { h y p } } : = B ^ { t _ { k } } - C _ { k } ( \Lambda _ { k } ) - h _ { k + 1 } \rho , \qquad D _ { k + 1 } ^ { \mathrm { h y p } } : = \mathcal { D } ( X ^ { k + 1 , \mathrm { h y p } } ) .\tag{D.14}
$$

All full-epoch conditional estimates below concern this continuation and do not condition on the actual path surviving the epoch.

At entrance, define on all of ${ \mathcal { E } } _ { T }$

$$
X ^ { 0 } : = { \cal B } ^ { \mathrm { e n t } } - h _ { 0 } \rho = X ^ { \mathrm { e n t } } , \qquad D _ { 0 } : = { \mathcal { D } } ( X ^ { 0 } ) ,
$$

and set both quantities to zero of $\mathcal { E } _ { T }$ . For $1 \leq k \leq J _ { \operatorname* { m a x } }$ , define

$$
X ^ { k } : = \left\{ \begin{array} { l l } { B ^ { t _ { k } } - h _ { k } \rho , } & { \mathrm { o n } ~ \mathcal { R } _ { k } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \qquad D _ { k } : = \mathcal { D } ( X ^ { k } ) .
$$

Whenever the corresponding ratio and deviation are defined—on ${ \mathcal { E } } _ { T }$ for $k = 0$ and on $\mathcal { R } _ { k }$ for $k \geq 1 \mathrm { - w e }$ have

$$
h _ { k } \Delta ( \beta ^ { k } ) \leq D _ { k } .\tag{D.15}
$$

Indeed, after multiplying by $h _ { k }$ , the distance on the left is the maximum of the relevant coordinate deviations, whereas the norm on the right is their sum. Any other quantity multiplied by a reached-event indicator is assigned zero of that event, and an empty epoch sum is zero.

Lemma D.4. For every fixed $0 \leq k \leq J _ { \operatorname* { m a x } }$ ，

$$
\mathbb { E } [ D _ { k } { \bf 1 } \{ \mathcal { R } _ { k } \} ] = O ( \sqrt { T } ) .\tag{D.16}
$$

Moreover,

$$
\mathbb { E } \left[ \sum _ { k = 0 } ^ { J - 1 } D _ { k } \mathbf { 1 } \{ \mathcal { R } _ { k } , ~ \Delta ( \beta ^ { k } ) < \delta _ { 0 } , ~ S ^ { t _ { k } } \in \mathcal { R } ^ { * } \} \right] = O ( \sqrt { T } ) ,\tag{D.17}
$$

$$
\mathbb { E } \left[ \left( H ^ { \sigma } + \sum _ { i \in I ^ { * } } B _ { i } ^ { \sigma } \right) \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} \right] = O ( \sqrt { T } ) .\tag{D.18}
$$

In particular, after multiplication by $\mathbf { 1 } \{ \mathcal { G } _ { 0 } \}$ , both the remaining horizon and the total binding-resource inventory at the analytical stop have expectations of order $O ( \sqrt { T } )$

Proof. If $J = 0 ,$ , then $\sigma = \tau ^ { \mathrm { e n t } }$ and $h _ { 0 } \leq H _ { \perp }$ . On $\mathcal { G } _ { 0 }$ , each binding inventory is at most $( \rho _ { i } + \delta _ { 0 } ) h _ { 0 }$ so the terminal envelope on these histories is $O ( 1 )$ and the epoch sum is empty. Moreover, $\mathbb { E } [ D _ { 0 } \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} ] = O ( 1 )$ by Lemma C.2. In what follows, we therefore consider epochs only on histories with $J \geq 1$ ; all sums and recursions include the corresponding reach indicators.

Step 1: the schedule reaches the square-root horizon after a fixed number of epochs. At every schedule index $k \leq J ,$ the ceiling recursion gives

$$
h _ { k } = O ( T ^ { \alpha ^ { k } } + 1 ) .
$$

If $J = K$ , then $h _ { J } = h _ { K } \leq H _ { \perp }$ . Otherwise $\alpha ^ { J } \leq 1 / 2$ . In either case,

$$
h _ { J } = O ( { \sqrt { T } } ) .\tag{D.19}
$$

Moreover, J is bounded by a constant depending only on $\alpha ,$ even when the fixed cutof is reached

first.

Step 2: maximal resource deviations within one epoch. Epoch 0 is protected and uses the nominal policy $\pi _ { \rho } ,$ not the selector at $\beta ^ { 0 }$ . The nominal Poisson decomposition gives, conditional on the entrance history,

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { n \leq \Lambda _ { 0 } } \left. C _ { 0 , i } ( n ) - n \overline { { c } } _ { i } ( \rho ) \right. \mid \mathcal { F } ^ { t _ { 0 } } \right] = O ( \sqrt { h _ { 0 } } + 1 ) .\tag{D.20}
$$

For $k \geq 1$ , suppose the analysis has reached $t _ { k }$ and passed the checks at $t _ { k }$ . Lemma D.3, applied to the one-period consumption of resource i, gives

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { n \leq \Lambda _ { k } } | C _ { k , i } ( n ) - n \overline { { c } } _ { i } ( \rho ) | \ : | \ : \mathcal { F } ^ { t _ { k } } \right] = O ( \sqrt { h _ { k } } + D _ { k } + 1 ) .\tag{D.21}
$$

Indeed, the conditional mean on $\mathcal { R } ^ { * }$ difers from the nominal mean by $O ( \Delta ( \beta ^ { k } ) )$ by (D.4), and $\Lambda _ { k } \Delta ( \beta ^ { k } ) \le h _ { k } \Delta ( \beta ^ { k } ) \le D _ { k }$

Step 3: propagate the one-sided inventory deviation. For $i \in I ^ { * } , { \bar { c } } _ { i } ( \rho ) = \rho _ { i }$ , and at a hypothetical full-epoch endpoint,

$$
X _ { i } ^ { k + 1 , \mathrm { h y p } } = X _ { i } ^ { k } - \left[ C _ { k , i } ( \Lambda _ { k } ) - \Lambda _ { k } \rho _ { i } \right] .
$$

For $i \notin I ^ { * } , \overline { { c } } _ { i } ( \rho ) = \rho _ { i } - \sigma _ { i } ^ { * }$ , and

$$
X _ { i } ^ { k + 1 , \mathrm { h y p } } = X _ { i } ^ { k } + \Lambda _ { k } \sigma _ { i } ^ { * } - [ C _ { k , i } ( \Lambda _ { k } ) - \Lambda _ { k } \overline { { c } } _ { i } ( \rho ) ] .
$$

The favorable drift $\Lambda _ { k } \sigma _ { i } ^ { * }$ can only reduce the negative part, so

$$
\begin{array} { r } { ( - X _ { i } ^ { k + 1 , \mathrm { h y p } } ) _ { + } \leq ( - X _ { i } ^ { k } ) _ { + } + | C _ { k , i } ( \Lambda _ { k } ) - \Lambda _ { k } \overline { { c } } _ { i } ( \rho ) | . } \end{array}
$$

Taking the one-sided norm specified in (D.14), summing over resources, and using (D.20) in epoch 0 gives

$$
\mathbb { E } [ D _ { 1 } ^ { \mathrm { h y p } } \mid \mathcal { F } ^ { t _ { 0 } } ] = O ( D _ { 0 } + \sqrt { h _ { 0 } } + 1 ) .
$$

For $1 \leq k < J$ , (D.21) similarly gives

$$
\mathbb { E } [ D _ { k + 1 } ^ { \mathrm { h y p } } \mid \mathcal { F } ^ { t _ { k } } ] = O ( D _ { k } + \sqrt { h _ { k } } + 1 ) .
$$

On $\mathcal { G } _ { 0 } .$

$$
X ^ { 0 } = X ^ { \mathrm { e n t } } = N ^ { \mathrm { e n t } } \rho - C ^ { \mathrm { e n t } } ,
$$

so Lemma C.2 gives

$$
\mathbb { E } [ D _ { 0 } \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} ] = O ( 1 ) .
$$

If the analysis reaches $t _ { k + 1 }$ , then epoch k did not stop for resource near-depletion, so the actual and hypothetical endpoint inventories coincide. Consequently,

$$
\begin{array} { r l } & { \mathbb { E } \big [ D _ { k + 1 } \mathbf { 1 } \big \{ \mathcal { R } _ { k + 1 } \big \} \big ] \leq \mathbb { E } \big [ D _ { k + 1 } ^ { \mathrm { h y p } } \mathbf { 1 } \big \{ \mathcal { R } _ { k } , k < J , \Delta ( \beta ^ { k } ) < \delta _ { 0 } , S ^ { t _ { k } } \in \mathcal { R } ^ { * } \big \} \big ] } \\ & { \qquad \leq O \left( \mathbb { E } \big [ D _ { k } \mathbf { 1 } \big \{ \mathcal { R } _ { k } \big \} \big ] + \sqrt { T } + 1 \right) . } \end{array}
$$

Here the start-event indicator is measurable at $t _ { k }$ , and $\sqrt { h _ { k } } \leq \sqrt { T }$ on that event. Iterating over the deterministic index range $0 \leq k < J _ { \operatorname* { m a x } }$ , with the reach indicators equal to zero after the realized cutof, proves (D.16). Summing over the same fixed set of indices proves (D.17).

Step 4: charge the analytical stopping events. At every reached resolving point,

$$
h _ { k } \Delta ( \beta ^ { k } ) \leq D _ { k } .
$$

Therefore

$$
h _ { k } \mathbf { 1 } \{ \mathcal { R } _ { k } , \Delta ( \boldsymbol { \beta } ^ { k } ) \geq \delta _ { 0 } \} \leq \delta _ { 0 } ^ { - 1 } D _ { k } \mathbf { 1 } \{ \mathcal { R } _ { k } \} ,
$$

whose expectation is $O ( \sqrt { T } )$ by (D.16).

Closure of the optimal target state class under the optimal induced kernel gives

$$
\mathbb { P } ( S ^ { t _ { 1 } } \notin \mathcal { R } ^ { * } , \mathcal { R } _ { 1 } ) = 0 .
$$

For $2 \leq k < J$ , Lemma D.2, applied to the preceding frozen epoch, gives

$$
\mathbb { P } ( S ^ { t _ { k } , \mathrm { h y p } } \notin \mathcal { R } ^ { * } \mid \mathcal { F } ^ { t _ { k - 1 } } ) = O ( \Delta ( \beta ^ { k - 1 } ) )
$$

on every reached regular start of epoch $k - 1$ with $k - 1 < J .$ . On $\mathcal { R } _ { k }$ , the actual and hypothetical endpoints coincide; this inclusion, rather than conditioning on survival, transfers the estimate to the reached endpoint. Since $h _ { k } \leq h _ { k - 1 }$ ，

$$
\begin{array} { r l } & { \mathbb { E } \Big [ h _ { k } { \mathbf { 1 } } \{ { \mathcal { R } } _ { k } , S ^ { t _ { k } } \notin { \mathcal { R } } ^ { * } \} \Big ] \leq O \left( \mathbb { E } \Big [ h _ { k - 1 } \Delta ( \beta ^ { k - 1 } ) { \mathbf { 1 } } \{ { \mathcal { R } } _ { k - 1 } , k - 1 < J , \Delta ( \beta ^ { k - 1 } ) < \delta _ { 0 } , S ^ { t _ { k - 1 } } \in { \mathcal { R } } ^ { * } \} \Big ] \right) } \\ & { \qquad = O \left( \mathbb { E } [ D _ { k - 1 } { \mathbf { 1 } } \{ { \mathcal { R } } _ { k - 1 } \} ] \right) = O ( \sqrt { T } ) . } \end{array}
$$

Now suppose resource near-depletion activates after n completed actions of a regular epoch $k < J$ . There are then $h _ { k } - n$ rounds remaining. For $i \in I ^ { * }$ ,

$$
B _ { i } ^ { t _ { k } + n } = ( h _ { k } - n ) \rho _ { i } + X _ { i } ^ { k } - [ C _ { k , i } ( n ) - n \rho _ { i } ] ,
$$

and for $i \notin I ^ { * }$ ，

$$
B _ { i } ^ { t _ { k } + n } = ( h _ { k } - n ) \rho _ { i } + X _ { i } ^ { k } + n \sigma _ { i } ^ { * } - [ C _ { k , i } ( n ) - n \overline { { c } } _ { i } ( \rho ) ] .
$$

If some inventory is at most c, then in either case

$$
( h _ { k } - n ) \operatorname* { m i n } _ { i } \rho _ { i } \leq \overline { c } + D _ { k } + \sum _ { i } \operatorname* { m a x } _ { r \leq \Lambda _ { k } } | C _ { k , i } ( r ) - r \overline { c } _ { i } ( \rho ) | .\tag{D.22}
$$

Multiply (D.22) by the event that resource near-depletion first activates in this epoch. This event is contained in its reached regular-start event; because the right-hand side is nonnegative, it can be bounded using that start event before taking conditional expectations. For $k = 0$ , the maximal terms are controlled by (D.20); for $k \geq 1$ they are controlled by (D.21). Together with (D.16), this bounds the expected remaining horizon at a resource near-depletion stop by $O ( \sqrt { T } )$ Summing over at most $J _ { \mathrm { m a x } }$ epochs does not change the order. If the analysis instead reaches $t _ { J }$ (D.19) gives $H ^ { \sigma } = h _ { J } = O ( { \sqrt { T } } )$ . Combining the ratio, outside-state, resource near-depletion, and scheduled-cutof bounds yields

$$
\mathbb { E } [ H ^ { \sigma } \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} ] = O ( \sqrt { T } ) .
$$

It remains to control binding inventories at σ. At a resolving stop $t _ { k }$

$$
B _ { i } ^ { t _ { k } } = h _ { k } \rho _ { i } + X _ { i } ^ { k } \leq h _ { k } \rho _ { i } + D _ { k } , \qquad i \in I ^ { * } .
$$

The preceding event-specific bounds control $h _ { k }$ , while (D.16) controls $D _ { k }$ . At a resource neardepletion stop in epoch k,

$$
B _ { i } ^ { \sigma } \leq H ^ { \sigma } \rho _ { i } + | X _ { i } ^ { k } | + \operatorname* { m a x } _ { r \leq \Lambda _ { k } } | C _ { k , i } ( r ) - r \rho _ { i } | , \qquad i \in I ^ { * } .
$$

The resource near-depletion estimate, (D.16), and the appropriate maximal resource estimate control the three terms. Summing over the fixed set of binding resources proves (D.18). □

## D.1.5 Finishing the proof

Proof. We proceed in three steps.

Step 1: entrance paths and the stopped comparison. Use the entrance event, initial postentry ratio and deviation, regular-entry event, and analytical stop defined in the setup preceding Lemma D.4. Lemma C.2 gives

$$
\mathbb { E } [ D _ { 0 } \mathbf { 1 } \{ \mathcal { E } _ { T } \} ] = O ( 1 ) , \qquad T \mathbb { P } ( \mathcal { E } _ { T } ^ { \mathrm { c } } ) = O ( 1 ) .\tag{D.23}
$$

It also shows that, on $\mathcal { E } _ { T }$ , the entrance discrepancy

$$
Z _ { \mathrm { e n t } } : = T G ( \rho ) - R ^ { \mathrm { e n t } } - F ( h _ { 0 } , B ^ { \mathrm { e n t } } , S ^ { \mathrm { e n t } } )
$$

is bounded above by a nonnegative random variable with $O ( 1 )$ expectation. In particular,

$$
\mathbb { E } [ Z _ { \mathrm { e n t } } ^ { + } \mathbf { 1 } \{ \mathcal { E } _ { T } \} ] = O ( 1 ) .\tag{D.24}
$$

On ${ \mathcal { E } } _ { T } \backslash { \mathcal { G } } _ { 0 }$ , the distance–deviation relation (D.15) gives

$$
h _ { 0 } \leq \frac { D _ { 0 } } { \delta _ { 0 } } .
$$

Since $T = N ^ { \mathrm { e n t } } + h _ { 0 }$ after successful entrance, Assumption 3.7 and (D.23) imply

$$
\begin{array} { r l r } {  { \mathbb { E } [ T \mathbf { 1 } \{ \mathcal { G } _ { 0 } ^ { \mathrm { c } } \} ] \leq T \mathbb { P } ( \mathcal { E } _ { T } ^ { \mathrm { c } } ) + \mathbb { E } [ N ^ { \mathrm { e n t } } ] + \frac { 1 } { \delta _ { 0 } } \mathbb { E } [ D _ { 0 } \mathbf { 1 } \{ \mathcal { E } _ { T } \} ] } } \\ & { } & { = O ( 1 ) . } \end{array}\tag{D.25}
$$

On $\mathcal { G } _ { 0 }$ , use the analytical stop controlled by Lemma D.4. Its resource near-depletion check is applied before an action when some inventory coordinate is at most ${ \overline { { c } } } ,$ so resource depletion cannot precede σ. Define a comparison policy as follows: after regular entry it agrees with IR up to σ and then always uses the null action; after a successful but nonregular entry it uses the null action immediately. On entrance-failure paths it simply follows IR until the process stops. Actual IR dominates this comparison pathwise, because the policies agree before analytical truncation and every later reward of IR is nonnegative. For notational completeness, set $\sigma = \tau ^ { \mathrm { e n t } }$ on ${ \mathcal { E } } _ { T } \setminus { \mathcal { G } } _ { 0 }$ . On entrance-failure paths, any post-entry random variable below is assigned an arbitrary value, say zero; such variables only occur multiplied by $\mathbf { 1 } \{ \mathcal { G } _ { 0 } \}$

Step 2: bound the accumulated dual slack. Define the nonnegative nominal dual action and state slacks by

$$
\begin{array} { c l c r } { { \ell _ { s j a } : = { u _ { s j } ^ { * } } - \left[ r ( s , j , a ) - ( \lambda ^ { * } ) ^ { \top } c ( s , j , a ) + \displaystyle \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) h _ { s ^ { \prime } } ^ { * } \right] , } } \\ { { \ell _ { s } : = { h _ { s } ^ { * } } + \gamma ^ { * } - \displaystyle \sum _ { j } p _ { j } u _ { s j } ^ { * } . } } \end{array}
$$

Under a selected policy at parameter $\beta ,$ its expected one-period slack at state s is

$$
\ell _ { \beta } ( s ) : = \ell _ { s } + \sum _ { j } p _ { j } \sum _ { a } \pi _ { \beta } ( a \mid s , j ) \ell _ { s j a } .
$$

If $s \in \mathcal { R } ^ { * }$ , complementary slackness makes the state slack zero and makes every action used by the nominal policy tight. Hence

$$
\begin{array} { c } { { 0 \leq \ell _ { \beta } ( s ) = \displaystyle \sum _ { j } p _ { j } \sum _ { a } ( \pi _ { \beta } ( a \mid s , j ) - \pi _ { \rho } ( a \mid s , j ) ) \ell _ { s j a } } } \\ { { { } } } \\ { { { } = O ( \Delta ( \beta ) ) , \qquad s \in \mathcal { R } ^ { * } , } } \end{array}
$$

where the final estimate follows from (D.4) and the finiteness of the action-slack collection. For $s \in \mathcal { O }$ , all slacks are bounded, so $\ell _ { \beta } ( s ) = O ( 1 )$ . Combining the two cases gives the pointwise bound

$$
\ell _ { \beta } ( s ) = O ( \Delta ( \beta ) + { \bf 1 } \{ s \in \mathcal O \} ) .\tag{D.26}
$$

Consider a reached epoch $1 \leq k < J$ whose resolving-point checks pass, and use the hypothetical full frozen continuation defined in the setup before Lemma D.4. Summing (D.26) and then applying Lemma D.2 give

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. \underset { t = t _ { k } } { \overset { t _ { k + 1 } - 1 } { \sum } } \ \ell _ { \beta ^ { k } } ( S ^ { t , \mathrm { h y p } } ) \ \right| \mathcal { F } ^ { t _ { k } } \right] } \\ & { \qquad \leq O \left( \Lambda _ { k } \Delta ( \beta ^ { k } ) + \mathbb { E } \left[ \left. \underset { t = t _ { k } } { \overset { t _ { k + 1 } - 1 } { \sum } } { \mathbf 1 } \{ S ^ { t , \mathrm { h y p } } \in \mathcal { O } \} \right| \ \mathcal { F } ^ { t _ { k } } \right] \right) } \\ & { \qquad \leq O \left( \Lambda _ { k } \Delta ( \beta ^ { k } ) \right) \leq O ( D _ { k } ) . } \end{array}\tag{D.27}
$$

The last inequality uses $\Lambda _ { k } \leq h _ { k }$ and (D.15). If resource near-depletion stops the actual epoch early,

its realized loss is a prefix of this nonnegative full-epoch loss, so the same estimate applies.

The protected epoch 0 has zero dual slack. Indeed, the entrance state is in $\mathcal { R } ^ { * }$ , stationarity of $q ^ { * }$ implies that $\mathcal { R } ^ { * }$ is closed under the nominal policy, and complementary slackness makes both the state slack and every used action slack zero there. Set $\ell ^ { t } = 0$ in that epoch; in a later epoch $k ,$ set $\ell ^ { t } = \ell _ { \beta ^ { k } } ( S ^ { t } )$ . Thus (D.27) and (D.17) yield

$$
\mathbb { E } \left[ \sum _ { t = \tau ^ { \mathrm { e n t } } } ^ { \sigma - 1 } \ell ^ { t } \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} \right] = O ( \sqrt { T } ) .\tag{D.28}
$$

Step 3: finish by the stopped-potential reduction. Direct substitution of the slack definitions into the nominal dual constraints gives, on $\mathcal { G } _ { 0 }$ and at every round before $\sigma _ { : }$

$$
\begin{array} { r } { \mathbb { E } \left[ R ^ { t } + F ( H ^ { t } - 1 , B ^ { t + 1 } , S ^ { t + 1 } ) ~ \Big | ~ \mathcal { F } ^ { t } \right] = F ( H ^ { t } , B ^ { t } , S ^ { t } ) - \ell ^ { t } . } \end{array}
$$

This includes the protected epoch because its slack is zero. Let

$$
R _ { \mathrm { p o s t } } : = \sum _ { t = \tau ^ { \mathrm { e n t } } } ^ { \sigma - 1 } R ^ { t }
$$

be the stopped comparison’s post-entry reward. Since $\sigma \leq T + 1$ is bounded, conditional finitehorizon telescoping from $\tau ^ { \mathrm { e n t } }$ , followed by the zero-reward null actions after truncation and deletion of the negative part of the terminal potential, gives, on $\mathcal { G } _ { 0 }$

$$
\begin{array} { r l r } {  { F ( h _ { 0 } , B ^ { \mathrm { e n t } } , S ^ { \mathrm { e n t } } ) - \mathbb { E } [ R _ { \mathrm { p o s t } } \ \lvert \ \mathcal { F } ^ { \tau ^ { \mathrm { e n t } } } ] } } \\ & { } & { \leq \mathbb { E } [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } + \sum _ { t = \tau ^ { \mathrm { e n t } } } ^ { \sigma - 1 } \ell ^ { t } \ \bigg \rvert \ \mathcal { F } ^ { \tau ^ { \mathrm { e n t } } } ] . } \end{array}\tag{D.29}
$$

For completeness, complementary slackness gives $\lambda _ { i } ^ { * } = 0$ for $i \notin I ^ { * }$ . Hence the positive terminal potential satisfies

$$
F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \leq \sum _ { i \in I ^ { * } } \lambda _ { i } ^ { * } B _ { i } ^ { \sigma } + | \gamma ^ { * } | H ^ { \sigma } + \| h ^ { * } \| _ { \infty } .
$$

The terminal-envelope estimate (D.18) therefore gives

$$
\mathbb { E } \left[ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} \right] \leq O ( \sqrt { T } ) .\tag{D.30}
$$

The event $\mathcal { G } _ { 0 }$ is measurable at entrance. Multiplying (D.29) by its indicator, taking expectations, and using (D.30) and (D.28) give

$$
\mathbb { E } \left[ \left( F ( h _ { 0 } , B ^ { \mathrm { e n t } } , S ^ { \mathrm { e n t } } ) - R _ { \mathrm { p o s t } } \right) \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} \right] \leq O ( \sqrt { T } ) .\tag{D.31}
$$

Finally, the stopped comparison earns $R ^ { \mathrm { e n t } } + R _ { \mathrm { p o s t } }$ on $\mathcal { G } _ { 0 }$ , earns a nonnegative reward on its

complement, and is dominated pathwise by IR. Therefore

$$
\begin{array} { r l } & { T G ( \rho ) - V _ { T } ^ { \mathrm { I R } } ( T \rho , s ^ { 1 } ) } \\ & { \quad \leq \mathbb { E } \left[ ( T G ( \rho ) - R ^ { \mathrm { e n t } } - R _ { \mathrm { p o s t } } ) \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} \right] + G ( \rho ) \mathbb { E } [ T \mathbf { 1 } \{ \mathcal { G } _ { 0 } ^ { \mathrm { c } } \} ] } \\ & { \quad = \mathbb { E } [ Z _ { \mathrm { e n t } } \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} ] + \mathbb { E } \left[ ( F ( h _ { 0 } , B ^ { \mathrm { e n t } } , S ^ { \mathrm { e n t } } ) - R _ { \mathrm { p o s t } } ) \mathbf { 1 } \{ \mathcal { G } _ { 0 } \} \right] + G ( \rho ) \mathbb { E } [ T \mathbf { 1 } \{ \mathcal { G } _ { 0 } ^ { \mathrm { c } } \} ] } \\ & { \quad \leq O ( \sqrt { T } ) , } \end{array}
$$

where the last line follows from (D.24), (D.31), and (D.25); here we also used $\mathcal { G } _ { 0 } \subseteq \mathcal { E } _ { T }$ to bound the first expectation by (D.24).

Hence the stationary-benchmark gap is $O ( \sqrt { T } )$ . Lemma 2.2 and $V _ { T } ^ { * } \le \Phi _ { T } ( T \rho , s ^ { 1 } )$ now give (14) and (15). □

## D.2 Proof of Proposition 5.3

Proof. There is one request type, one resource, and three states

$$
{ \cal S } = \{ A , B , C \} .
$$

The initial state is A, the initial inventory is $B ^ { 1 } = T$ , and $\rho = 1$ . The common action set contains the null action $a ^ { 0 }$ and non-null actions $a , e , r .$ . The relevant state-action pairs are

<table><tr><td>state</td><td>action</td><td>reward</td><td>cost</td><td>next state</td></tr><tr><td>A</td><td>a</td><td>1</td><td>3/2</td><td>B</td></tr><tr><td>B</td><td>a</td><td>1</td><td>1/2</td><td>A</td></tr><tr><td>B</td><td>e</td><td>1</td><td>1/2</td><td>C</td></tr><tr><td>C</td><td>r</td><td>3/4</td><td>1/2</td><td>A.</td></tr></table>

The null action has zero reward and cost and transitions according to

$$
A \ { \stackrel { a ^ { 0 } } { \longrightarrow } } \ B , \qquad B \ { \stackrel { a ^ { 0 } } { \longrightarrow } } \ A , \qquad C \ { \stackrel { a ^ { 0 } } { \longrightarrow } } \ C .
$$

Every unspecified non-null state-action pair has reward zero, positive cost, and the same transition as the null action in that state, so it is weakly dominated by the null action and can be omitted when computing the fluid value. This omission does not create a uniqueness issue. Indeed, whenever the reduced optimizer identified below is capacity-binding, a full optimizer cannot place positive mass on an unspecified action: replacing that mass by the same mass on the null action preserves reward and state flow and strictly lowers resource use. The resulting point would have to be the unique reduced optimizer, which binds the resource constraint, making the original point infeasible. For ratios above one, uniqueness instead follows directly from the equality case in the reward bound.

Reduction of the stationary LP. Let x be the stationary state-request-action mass of action e in state B, let $n _ { C }$ be the null-action mass in state C, and let $n _ { A } , n _ { B }$ be the null-action masses in

states A, B. State flow gives

$$
z _ { C , r } = x , \qquad q _ { A } = q _ { B } = q : = \frac { 1 - x - n _ { C } } { 2 } , \qquad q _ { C } = x + n _ { C } .
$$

The remaining relevant stationary state-request-action masses are

$$
z _ { A , a } = q - n _ { A } , \qquad z _ { B , a } = q - x - n _ { B } , \qquad z _ { B , e } = x .
$$

Accordingly, the stationary reward and resource consumption are

$$
\begin{array} { c } { { \displaystyle { \overline { { r } } = ( q - n _ { A } ) + ( q - x - n _ { B } ) + x + \frac { 3 } { 4 } x = 1 - \frac { 1 } { 4 } x - n _ { C } - n _ { A } - n _ { B } , \qquad ( \mathrm { I } - \mathrm { I } ) , } } } \\ { { \displaystyle { \sum _ { s , j , a } c ( s , j , a ) z _ { s j a } = \frac { 3 } { 2 } ( q - n _ { A } ) + \frac { 1 } { 2 } ( q - x - n _ { B } ) + \frac { 1 } { 2 } x + \frac { 1 } { 2 } x = 1 - \frac { 1 } { 2 } x - n _ { C } - \frac { 3 } { 2 } n _ { A } - \frac { 1 } { 2 } n _ { B } . } } } \end{array}\tag{D.32}
$$

For a capacity ratio $\beta = 1 - d .$ resource feasibility is equivalent to

$$
{ \frac { 1 } { 2 } } x + n _ { C } + { \frac { 3 } { 2 } } n _ { A } + { \frac { 1 } { 2 } } n _ { B } \geq d .\tag{D.33}
$$

The nominal optimizer is unique. At d = 0, (D.32) gives

$$
\overline { { r } } = 1 - \frac { 1 } { 4 } x - n _ { C } - n _ { A } - n _ { B } \leq 1 .
$$

Equality requires $x = n _ { C } = n _ { A } = n _ { B } = 0 ;$ it also rules out every unspecified action, whose reward is zero. Hence the nominal optimizer is unique in the full action space and is given by

$$
q _ { A } ^ { * } = q _ { B } ^ { * } = { \frac { 1 } { 2 } } , \qquad q _ { C } ^ { * } = 0 , \qquad z _ { A , a } ^ { * } = z _ { B , a } ^ { * } = { \frac { 1 } { 2 } } , \qquad G ( 1 ) = 1 .\tag{D.34}
$$

Every small negative perturbation activates state C. For $d > 0$ , (D.33) implies

$$
\begin{array} { l } { 1 - \displaystyle \overline { { r } } = \frac 1 4 x + n _ { C } + n _ { A } + n _ { B } } \\ { \displaystyle \quad = \frac 1 2 \left( \frac 1 2 x + n _ { C } + \frac 3 2 n _ { A } + \frac 1 2 n _ { B } \right) + \frac 1 2 n _ { C } + \frac 1 4 n _ { A } + \frac 3 4 n _ { B } } \\ { \displaystyle \quad \geq \frac d 2 . } \end{array}
$$

Therefore

$$
\overline { { r } } \leq 1 - \frac { d } { 2 } .
$$

Equality requires $n _ { C } = n _ { A } = n _ { B } = 0$ and $x = 2 d .$ . For $0 < d \le 1 / 6$ , this solution is feasible and hence is the unique optimizer:

$$
n _ { C } = n _ { A } = n _ { B } = 0 , \qquad x = 2 d , \qquad q _ { A } = q _ { B } = \frac { 1 - 2 d } 2 , \qquad q _ { C } = 2 d .
$$

Indeed, the remaining mass of action a in state B is $( 1 - 6 d ) / 2 \geq 0$ , including the endpoint $d = 1 / 6$

Thus, throughout $0 < d \le 1 / 6$

$$
G ( 1 - d ) = 1 - \frac { 1 } { 2 } d , \qquad \pi _ { 1 - d } ( e \mid B ) = \frac { x } { q _ { B } } = \frac { 4 d } { 1 - 2 d } .\tag{D.35}
$$

For every $\beta \geq 1$ , the nominal solution in (D.34) remains feasible and uniquely optimal: the reward is at most one, with equality only when $x = n _ { C } = n _ { A } = n _ { B } = 0$ and no unspecified zero-reward action is used. In particular,

$$
q _ { C } ( \beta ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ } \beta \geq 1 .\tag{D.36}
$$

The nominal solution is degenerate. To see this without choosing a particular redundant-row deletion, recall Proposition 3.3: a unique nondegenerate nominal optimum would have a fixed positive-variable support throughout some right-hand-side neighborhood. Here $z _ { B , e } = q _ { C } = z _ { C , r } = 0$ at $d = 0$ but all three are positive for every $d > 0$ . Hence the unique nominal optimum cannot be nondegenerate.

Both dynamic assumptions hold. The optimal target state class is

$$
{ \mathcal { R } } ^ { * } = \{ A , B \} .
$$

The optimal induced kernel on this class is

$$
K ^ { * } = { \binom { 0 } { 1 } } \ 1 ) ,
$$

which is irreducible, although periodic, so Assumption 3.4 holds. The initial state A belongs to $\mathcal { R } ^ { * }$ so the entrance phase is empty and Assumption 3.7 also holds.

Round-by-round behavior of FR. Write

$$
H ^ { t } = T - t + 1 , \qquad X ^ { t } : = B ^ { t } - H ^ { t } .
$$

Initially $X ^ { 1 } = 0$ . At the nominal ratio, FR takes action a in state $A ,$ earns one, and consumes $3 / 2$ If $T = 1$ , this is the terminal action and there is no subsequent ratio to evaluate. If $T \geq 2 .$ , the next round is active in state $B _ { ; }$ , with

$$
X ^ { t } = - \frac { 1 } { 2 } , \qquad \beta ^ { t } = 1 - \frac { 1 } { 2 H ^ { t } } .
$$

Using (D.35), whenever such a B-round has $H ^ { t } \geq 3$ , the conditional probability of selecting e is

$$
\pi _ { 1 - 1 / ( 2 H ^ { t } ) } ( e \mid B ) = { \frac { 2 } { H ^ { t } - 1 } } , \qquad H ^ { t } \geq 3 .\tag{D.37}
$$

For the remaining cases $H ^ { t } \in \{ 1 , 2 \}$ , set $d = 1 / ( 2 H ^ { t } )$ . Nonnegativity of the baseline mass in state B gives

$$
3 x + n _ { C } + 2 n _ { B } \leq 1 .
$$

Together with (D.33), this implies

$$
\begin{array} { l } { 1 - \displaystyle { \overline { { r } } } = \frac { 2 } { 3 } \left( \frac { 1 } { 2 } x + n _ { C } + \frac { 3 } { 2 } n _ { A } + \frac { 1 } { 2 } n _ { B } \right) - \frac { 1 } { 1 2 } x + \frac { 1 } { 3 } n _ { C } + \frac { 2 } { 3 } n _ { B } } \\ { \geq \displaystyle \frac { 2 } { 3 } d - \frac { 1 } { 3 6 } + \frac { 1 3 } { 3 6 } n _ { C } + \frac { 1 3 } { 1 8 } n _ { B } . } \end{array}
$$

Equality is attained uniquely by

$$
x = \frac { 1 } { 3 } , \qquad n _ { C } = n _ { B } = 0 , \qquad n _ { A } = \frac { 2 } { 3 } \left( d - \frac { 1 } { 6 } \right) .
$$

The last mass equals 1/18 when $H ^ { t } = 2$ and $2 / 9$ when $H ^ { t } = 1$ , so both solutions are feasible. In each case $q _ { B } = x = 1 / 3$ , and FR chooses e with probability one. The endpoint $H ^ { t } = 3$ also has excursion probability one by (D.37).

Actions a and e both consume $1 / 2$ . Therefore, after an active B-round, $X ^ { t + 1 } = 0$ in either case. If a is selected, the state returns to A and the same two-round cycle begins again. If e is selected and a round remains, the next state is C and the next ratio is exactly one. For even $T \geq 4$ , a trajectory that has not made an earlier excursion must do so at the B-round with three rounds remaining; for odd $T \geq 3$ , it must do so with two rounds remaining. Thus a B-round with only one round remaining occurs only when $T = 2 ;$ its e action exhausts the remaining inventory $1 / 2$ and is terminal.

At ratio one the unique optimizer in (D.34) assigns zero stationary probability to state C. By the FR convention, when the selected stationary LP solution assigns zero probability to the current state, the algorithm chooses the null action. Suppose there are $H \geq 1$ rounds and H units of inventory when C is entered. After $0 \leq \ell < H$ null rounds, before the next active round, the inventory remains H, the remaining horizon is $H - \ell ,$ , and hence

$$
\beta = { \frac { H } { H - \ell } } \geq 1 .
$$

By (D.36), every newly solved optimizer still assigns zero stationary probability to C. Therefore, FR continues choosing the null action until the horizon ends. Thus entry into C is a permanent trap without any analytical truncation.

The trapping event has constant probability. Let $N = \lfloor T / 4 \rfloor$ and take $T \geq 8$ . Consider the first N visits to state B. Conditional on no earlier excursion, visit n occurs with remaining horizon

$$
H ^ { n } = T - 2 n + 1 ,
$$

and by (D.37), its conditional excursion probability is $2 / ( T - 2 n )$ . The action randomizations are

fresh from round to round. Hence

$$
{ \begin{array} { r l } & { \mathbb { P } ( { \mathrm { n o ~ e x c u r s i o n ~ i n ~ t h e ~ f i r s t ~ } } N { \mathrm { ~ v i s i t s } } ) = \displaystyle \prod _ { n = 1 } ^ { N } \left( 1 - { \frac { 2 } { T - 2 n } } \right) } \\ & { \qquad = \displaystyle \prod _ { n = 1 } ^ { N } { \frac { T - 2 n - 2 } { T - 2 n } } = { \frac { T - 2 N - 2 } { T - 2 } } . } \end{array} }
$$

Therefore

$$
\mathbb { P } ( \mathrm { a n ~ e x c u r s i o n ~ i n ~ t h e ~ f i r s t ~ } N { \mathrm { ~ v i s i t s } } ) = { \frac { 2 N } { T - 2 } } \geq { \frac { T / 2 - 2 } { T - 2 } } \geq { \frac { 1 } { 3 } } .
$$

The N-th visit occurs in round $2 N \le T / 2$ . Hence, on this event, FR enters C with at least $T - 2 N \geq T / 2$ rounds remaining.

A linearly better admissible policy. Define $\widehat { \pi } ^ { \mathrm { F } }$ to follow FR until the first time action e is selected. If state C is reached with an active round remaining, $\widehat { \pi } ^ { \mathrm { F } }$ selects r in that round and thereafter repeatedly selects the baseline action a in states A and B until the process terminates.

Suppose the excursion occurs by round $T / 2 ,$ , and let $H \geq T / 2$ be the number of rounds remaining upon entering C. Since $X = 0$ after the excursion, the remaining inventory is exactly H. Policy $\widehat { \pi } ^ { \mathrm { F } }$ first takes $r ,$ earning $3 / 4$ , consuming $1 / 2$ , and returning to $A .$ It then uses action a in each of the remaining $H - 1$ rounds. Starting from A, the successive baseline costs are

$$
{ \frac { 3 } { 2 } } , { \frac { 1 } { 2 } } , { \frac { 3 } { 2 } } , { \frac { 1 } { 2 } } , \dots .
$$

The sum of the first $H - 1$ baseline costs is $H - 1$ when $H - 1$ is even and $H - 1 / 2$ when $H - 1$ is odd. Including the $1 / 2$ cost of r, the total sufix consumption is at most H. Thus $\widehat { \pi } ^ { \mathrm { F } }$ is admissible and acts in every remaining round. Its sufix reward is

$$
{ \frac { 3 } { 4 } } + ( H - 1 ) = H - { \frac { 1 } { 4 } }\tag{D.38}
$$

over a sufix on which FR earns zero. The same calculation applies after every nonterminal excursion, while after a terminal excursion the two policies coincide; hence $\widehat { \pi } ^ { \mathrm { F } }$ never earns less than FR on the complementary event. Combining the probability bound above with (D.38),

$$
V _ { T } ^ { \widehat { \pi } ^ { \mathrm { F } } } - V _ { T } ^ { \mathrm { F R } } \geq \frac { 1 } { 3 } \left( \frac { T } { 2 } - \frac { 1 } { 4 } \right) = \frac { T } { 6 } - \frac { 1 } { 1 2 } = \Omega ( T ) .\tag{D.39}
$$

This proves (D.39). Since $V _ { T } ^ { * } \geq V _ { T } ^ { \widehat { \pi } ^ { \mathrm { F } } }$ and $\Phi _ { T } \geq V _ { T } ^ { * }$ , (16) follows.

Finally, before the trapping event,

$$
| \beta ^ { t } - \rho | = \left\{ { \begin{array} { l l } { 0 , } & { S ^ { t } = A , } \\ { \displaystyle { \frac { 1 } { 2 H ^ { t } } } , } & { S ^ { t } = B , } \end{array} } \leq { \frac { 1 } { 2 H ^ { t } } } . \right.
$$

Thus even a much stronger estimate than $\mathbb { E } | \beta ^ { t } - \rho | = O ( ( H ^ { t } ) ^ { - 1 / 2 } )$ would not prevent the linear loss. The obstruction is not a large capacity-ratio deviation; it is the disappearance, at the next

## D.3 Proof of Proposition 5.4

Proof. There is one request type, one resource, two states ${ \cal { S } } = \{ 1 , 2 \}$ , and two actions: the null action $a ^ { 0 }$ and a non-null action a. Both states are absorbing under both actions. The non-null reward and cost are

<table><tr><td></td><td>r(s, a) c(s,a)</td><td></td></tr><tr><td>s = 1</td><td>3/2</td><td>1</td></tr><tr><td>s = 2</td><td>2</td><td>2.</td></tr></table>

The null action has zero reward and cost. Let $\rho = 3 / 2$ and $s ^ { 1 } = 1$

Let $x _ { s }$ be the stationary mass of the non-null action in state s. Eliminating null-action and state variables, the stationary LP is

$$
\operatorname* { m a x } \left\{ { \frac { 3 } { 2 } } x _ { 1 } + 2 x _ { 2 } : x _ { 1 } + x _ { 2 } \leq 1 , \ x _ { 1 } + 2 x _ { 2 } \leq { \frac { 3 } { 2 } } , \ x _ { 1 } , x _ { 2 } \geq 0 \right\} .
$$

Its relevant vertices are

$$
( 1 , 0 ) , \qquad ( 1 / 2 , 1 / 2 ) , \qquad ( 0 , 3 / 4 ) ,
$$

with respective values $3 / 2 , 7 / 4 , 3 / 2$ . Hence its unique optimum is

$$
x _ { 1 } ^ { * } = x _ { 2 } ^ { * } = { \frac { 1 } { 2 } } , \qquad G ( \rho ) = { \frac { 7 } { 4 } } .
$$

For completeness, add null-action masses $n _ { 1 } , n _ { 2 }$ , stationary state probabilities $q _ { 1 } , q _ { 2 }$ , and the resource slack σ. A full-row-rank standard form is

$$
\begin{array} { r } { x _ { 1 } + n _ { 1 } - q _ { 1 } = 0 , } \\ { x _ { 2 } + n _ { 2 } - q _ { 2 } = 0 , } \\ { q _ { 1 } + q _ { 2 } = 1 , } \\ { x _ { 1 } + 2 x _ { 2 } + \sigma = \cfrac { 3 } { 2 } . } \end{array}
$$

At the optimum the variables $x _ { 1 } , x _ { 2 } , q _ { 1 } , q _ { 2 }$ all equal $1 / 2$ , while $n _ { 1 } , n _ { 2 } , \sigma$ are zero. The basis matrix formed by the columns of $x _ { 1 } , x _ { 2 } , q _ { 1 } , q _ { 2 }$ is

$$
\left( { \begin{array} { c c c c } { 1 } & { 0 } & { - 1 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { - 1 } \\ { 0 } & { 0 } & { 1 } & { 1 } \\ { 1 } & { 2 } & { 0 } & { 0 } \end{array} } \right) , \qquad \operatorname* { d e t } = 1 .
$$

Every basic variable is strictly positive. The strict objective comparisons above also show uniqueness.   
Assumptions 3.1 and 3.2 therefore hold.

The optimal induced kernel is

$$
K ^ { * } = { \binom { 1 } { 0 } } \ 1 ) ,
$$

which is reducible on $\mathcal { R } ^ { * } = \{ 1 , 2 \}$ . Consequently, Assumption 3.4 fails. In contrast, entrance into $\mathcal { R } ^ { * }$ takes zero time and consumes no resources, so Assumption 3.7 holds. Since the actual process starts in state 1 and never leaves ${ \mathrm { i t } } ,$ no policy can earn more than $3 / 2$ in a round. The capacity $3 T / 2$ is suficient to take action a in all T rounds, so in fact

$$
V _ { T } ^ { * } ( T \rho , 1 ) = \frac { 3 } { 2 } T .
$$

Since every policy has value at most $V _ { T } ^ { * }$ , combining this with $T G ( \rho ) = 7 T / 4$ proves the stationarybenchmark inequality in (17) for every policy. The time-aggregated transient LP has the same linear obstruction. For $T \geq 2$ , set

$$
X _ { 1 , a } = T / 2 - 1 , \qquad Y _ { 1 , a } = 1 , \qquad X _ { 2 , a } = T / 2 , \qquad Y _ { 2 , a } = 0 ,
$$

and $Q _ { 1 } = Q _ { 2 } = T / 2$ , with every unspecified transient variable set to zero. These variables satisfy the transient request, state-flow, and horizon constraints. The charged resource use is

$$
X _ { 1 , a } + 2 X _ { 2 , a } = 3 T / 2 - 1 \leq 3 T / 2 ,
$$

while the objective is

$$
\frac { 3 } { 2 } ( X _ { 1 , a } + Y _ { 1 , a } ) + 2 ( X _ { 2 , a } + Y _ { 2 , a } ) = \frac { 7 } { 4 } T .
$$

Thus

$$
\Phi _ { T } ( T \rho , 1 ) - V _ { T } ^ { * } ( T \rho , 1 ) \geq T / 4 .
$$

Since $V _ { T } ^ { \pi } ( T \rho , 1 ) \leq V _ { T } ^ { * } ( T \rho , 1 )$ for every nonanticipating policy $\pi ,$ this proves the transient-fluid inequality in (17) as well. The transient relaxation admits a source-free stationary circulation in the unreachable absorbing class because its state-flow constraints are aggregated over time. □

## E Analysis of the unknown request prior

This appendix gives the supporting results and proofs for Section 6. Throughout the positive analysis, Assumptions 3.4 and 3.7 hold, and UIR-UP is supplied with the true optimal target state class $\mathcal { R } ^ { * }$ and the entrance policy $\pi ^ { \mathrm { e n t } }$ fixed under Assumption 3.7; Assumption 3.2 is imposed only where stated. Conditional on the realized entrance history, for the analysis only, write $T = H ^ { \mathrm { e n t } }$ and abbreviate $B ^ { \tau ^ { \mathrm { e n t } } + n } , S ^ { \tau ^ { \mathrm { e n t } } + n } , H ^ { \tau ^ { \mathrm { e n t } } + n }$ , and ${ \mathcal { F } } ^ { \tau ^ { \mathrm { e n t } } + n }$ by B<sup>n</sup>, S<sup>n</sup>, H<sup>n</sup>, and ${ \mathcal { F } } ^ { n }$ , respectively. Here n counts completed post-entry rounds; it does not replace the original round index in the algorithm. This convention is used only in the possibly degenerate case. Under Assumption 3.2, Proposition 3.6 gives ${ \mathcal { R } } ^ { * } = S ;$ in that case $T$ retains its original meaning and the completed-round clock starts at 0.

## E.1 Proof of Theorem 6.1

Throughout the proof, we suppose that Assumption 3.1 holds and do not mention it again.

We start with several preliminary properties of estimation concentration and the construction of the phase-2 re-solving schedule.

## E.1.1 Estimation concentration

Proposition E.1. There are constants $C _ { \mathrm { e s t } } , c _ { \mathrm { e s t } } > 0$ , depending only on $| \mathcal { I } |$ , such that, conditional on any realized entrance history, the estimator in (18) satisfies, for every integer $n \geq 1$

$$
\begin{array} { r } { \mathbb { E } \| \widehat { p } ^ { n } - p \| _ { 1 } \leq C _ { \mathrm { e s t } } n ^ { - 1 / 2 } , } \end{array}\tag{E.1}
$$

$$
\mathbb { P } \big ( \| \hat { p } ^ { n } - p \| _ { 1 } \ge x \big ) \le C _ { \mathrm { e s t } } \exp ( - c _ { \mathrm { e s t } } n x ^ { 2 } ) , \qquad 0 < x \le 1 .\tag{E.2}
$$

Proof. Conditional on the realized entrance history, the requests counted by $N _ { j } ( n )$ remain i.i.d. with distribution $p .$ . Write $d : = | { \mathcal { I } } |$ , and let ${ \widetilde { p } } _ { n }$ be their ordinary empirical distribution. If u denotes the uniform distribution on $\mathcal { I }$ , then the Laplace-smoothed estimator satisfies

$$
{ \widehat { p } } ^ { n } = { \frac { n } { n + d } } { \widetilde { p } } _ { n } + { \frac { d } { n + d } } u .
$$

Since $\| u - p \| _ { 1 } \leq 2$ , it follows that

$$
\| \hat { p } ^ { n } - p \| _ { 1 } \leq \| \widetilde { p } _ { n } - p \| _ { 1 } + \frac { 2 d } { n + d } .\tag{E.3}
$$

For the mean bound, the variance of the j-th coordinate of ${ \widetilde { p } } _ { n }$ is $p _ { j } ( 1 - p _ { j } ) / n$ . Hence, by Cauchy–Schwarz and Jensen’s inequality,

$$
\mathbb { E } \Vert \widetilde { p } _ { n } - p \Vert _ { 1 } \leq \sum _ { j \in \mathcal { I } } { \sqrt { \frac { p _ { j } ( 1 - p _ { j } ) } { n } } } \leq \sqrt { \frac { d } { n } } .
$$

Combining this with (E.3) and $n \geq 1$ gives

$$
\begin{array} { r } { \mathbb { E } \| \widehat { p } ^ { n } - p \| _ { 1 } \leq ( \sqrt { d } + 2 d ) n ^ { - 1 / 2 } . } \end{array}
$$

For the tail bound, the Bretagnolle–Huber–Carol inequality (van der Vaart and Wellner, 1996, Proposition A.6.6) gives, for every $x > 0$

$$
\begin{array} { r } { \mathbb { P } \big ( \| \widetilde { p } _ { n } - p \| _ { 1 } > x \big ) \le 2 ^ { d } \exp ( { - n { x } ^ { 2 } } / 2 ) . } \end{array}
$$

If $x > 4 d / ( n + d )$ , then (E.3) and the preceding inequality, with $x = x / 2$ , yield

$$
\begin{array} { r } { \mathbb { P } \big ( \| \widehat { p } ^ { n } - p \| _ { 1 } > x \big ) \leq 2 ^ { d } \exp ( - n x ^ { 2 } / 8 ) . } \end{array}
$$

If instead $x \leq 4 d / ( n + d )$ , then

$$
n x ^ { 2 } \leq \frac { 1 6 d ^ { 2 } n } { ( n + d ) ^ { 2 } } \leq 4 d ,
$$

so the trivial probability bound 1 is at most $\exp ( d / 2 ) \exp ( - n x ^ { 2 } / 8 )$ . Thus the two asserted bounds hold, for example, with

$$
c _ { \mathrm { e s t } } = \frac { 1 } { 8 } , \qquad C _ { \mathrm { e s t } } = \operatorname* { m a x } \{ \sqrt { d } + 2 d , 2 ^ { d } , \exp ( d / 2 ) \} ,
$$

which depend only on $d = | \mathcal { I } |$

## E.1.2 Sparse-balanced scheduling

Under a unique but degenerate optimum, a frozen empirical policy can have first-order error $O ( \| \widehat { p } ^ { n } - p \| _ { 1 } )$ per round. Therefore, if an epoch starts after n observations and lasts Λ rounds, its statistical contribution can be of order $\Lambda / \sqrt { n }$ . The purpose of the sparse-balanced schedule is to keep the sum of these terms of order $O ( \sqrt { T } )$ with only O(log log T) re-solves.

Lemma E.2. Let $1 \leq n < \tau ^ { \mathrm { h a l } }$ be integers. For every integer K satisfying

$$
\operatorname* { m i n } \left\{ \tau ^ { \mathrm { h a l } } - n , \operatorname* { m a x } \left\{ 1 , \left\lceil \log _ { 2 } \log \frac { \tau ^ { \mathrm { h a l } } } { n } \right\rceil + 2 \right\} \right\} \leq K \leq \tau ^ { \mathrm { h a l } } - n ,
$$

there is a deterministic integer partition $n = t _ { 0 } < \cdots < t _ { K } = \tau ^ { \mathrm { h a l } }$ with

$$
\sum _ { k = 0 } ^ { K - 1 } \frac { t _ { k + 1 } - t _ { k } } { \sqrt { t _ { k } } } \leq 9 \sqrt { \tau ^ { \mathrm { h a l } } } .
$$

In particular, the minimizing schedule in (23), with the clipped choice (22), satisfies this bound. Every resolving time is a deterministic function of $( n , \tau ^ { \mathrm { h a l } } , K )$

Proof. If $K = \tau ^ { \mathrm { h a l } } - n$ , use all integer times. Its cost is

$$
\sum _ { t = n } ^ { \mathrm { { h a l } } _ { - 1 } } \frac { 1 } { \sqrt { t } } \leq \int _ { n - 1 } ^ { \tau ^ { \mathrm { { h a l } } } - 1 } \frac { d x } { \sqrt { x } } \leq 2 \sqrt { \tau ^ { \mathrm { { h a l } } } } .
$$

Otherwise the clipping at $\tau ^ { \mathrm { h a l } } - n$ is inactive, so $K \geq \operatorname* { m a x } \{ 1 , \lceil \log _ { 2 } \log ( \tau ^ { \mathrm { h a l } } / n ) \rceil + 2 \}$ . Write

$u : = \tau ^ { \mathrm { h a l } } / n . \mathrm { ~ I f ~ } u < \exp ( 1 )$ , every feasible integer partition has cost at most

$$
\frac { \tau ^ { \mathrm { h a l } } - n } { \sqrt { n } } = \frac { u - 1 } { \sqrt { u } } \sqrt { \tau ^ { \mathrm { h a l } } } < 2 \sqrt { \tau ^ { \mathrm { h a l } } } .
$$

Thus it remains to treat log u $\geq 1$

Set

$$
K _ { 0 } : = \lfloor \log _ { 2 } \log u \rfloor + 2 , \qquad \omega : = 2 ^ { 1 - K _ { 0 } / ( 2 ^ { K _ { 0 } } - 1 ) } u ^ { 1 / ( 2 ^ { K _ { 0 } + 1 } - 2 ) } .
$$

Then $2 \le K _ { 0 } \le K$ . Define real times by

$$
t _ { 0 } ^ { \mathrm { c } } : = n , \qquad t _ { k + 1 } ^ { \mathrm { c } } : = \omega 2 ^ { k - K _ { 0 } + 1 } \sqrt { \tau ^ { \mathrm { h a l } } t _ { k } ^ { \mathrm { c } } } , \quad 0 \le k < K _ { 0 } .\tag{E.4}
$$

Unrolling the logarithmic recurrence, using

$$
\sum _ { k = 0 } ^ { K _ { 0 } - 1 } 2 ^ { - k } = 2 ( 1 - 2 ^ { - K _ { 0 } } ) , \qquad \sum _ { k = 0 } ^ { K _ { 0 } - 1 } k 2 ^ { - k } = 2 - ( K _ { 0 } + 1 ) 2 ^ { 1 - K _ { 0 } } ,
$$

gives

$$
\log \frac { t _ { K _ { 0 } } ^ { \mathrm { c } } } { \tau ^ { \mathrm { h a l } } } = - 2 ^ { - K _ { 0 } } \log u + 2 ( 1 - 2 ^ { - K _ { 0 } } ) \log \omega - \left[ 2 - ( K _ { 0 } + 1 ) 2 ^ { 1 - K _ { 0 } } \right] \log 2 = 0 .
$$

Hence the endpoint is exactly $t _ { K _ { 0 } } ^ { \mathrm { c } } = \tau ^ { \mathrm { h a l } }$

All the real times are strictly increasing. For the first step, direct substitution gives

$$
\log \frac { t _ { 1 } ^ { \mathrm { c } } } { t _ { 0 } ^ { \mathrm { c } } } = \frac { 2 ^ { K _ { 0 } - 1 } } { 2 ^ { K _ { 0 } } - 1 } \left( \log u - [ 2 ( K _ { 0 } - 2 ) + 2 ^ { 2 - K _ { 0 } } ] \log 2 \right) > 0 .
$$

Indeed, log $u \geq 2 ^ { K _ { 0 } - 2 }$ ; when $K _ { 0 } = 2$ the asserted positivity follows from $1 > \log 2$ , and when $K _ { 0 } \geq 3$ it follows from

$$
2 ( K _ { 0 } - 2 ) + 2 ^ { 2 - K _ { 0 } } \leq \frac 5 4 2 ^ { K _ { 0 } - 2 } , \qquad \frac 5 4 \log 2 < 1 .
$$

The first inequality uses $2 k \leq 2 ^ { k }$ and $2 ^ { - k } \leq 2 ^ { k } / 4$ for integers $k \geq 1$ . To propagate strict increase, compare the coeficients in the recurrence: if $t _ { k + 1 } ^ { \mathrm { c } } > t _ { k } ^ { \mathrm { c } }$ , then

$$
\sqrt { t _ { k + 1 } ^ { \mathrm { c } } / \tau ^ { \mathrm { h a l } } } < \omega 2 ^ { k - K _ { 0 } + 1 } < \omega 2 ^ { k - K _ { 0 } + 2 } ,
$$

so (E.4) yields $t _ { k + 2 } ^ { \mathrm { c } } > t _ { k + 1 } ^ { \mathrm { c } }$ . Consequently every interior real time lies strictly between n and $\tau ^ { \mathrm { h a l } }$ Since log $u < 2 ^ { K _ { 0 } - 1 }$ , the coeficient satisfies $\omega \leq 2 \exp ( 1 / 2 )$ . Thus the real schedule costs at most

$$
\sum _ { k = 0 } ^ { K _ { 0 } - 1 } \frac { t _ { k + 1 } ^ { \mathrm { c } } - t _ { k } ^ { \mathrm { c } } } { \sqrt { t _ { k } ^ { \mathrm { c } } } } \leq \omega \sqrt { \tau ^ { \mathrm { h a l } } } \sum _ { k = 0 } ^ { K _ { 0 } - 1 } 2 ^ { k - K _ { 0 } + 1 } \leq 4 \exp ( 1 / 2 ) \sqrt { \tau ^ { \mathrm { h a l } } } .
$$

Round every real time upward, $t _ { k } ^ { \mathrm { r } } : = \lceil t _ { k } ^ { \mathrm { c } } \rceil$ . The endpoints are unchanged, the resulting sequence is nondecreasing, and

$$
\frac { t _ { k + 1 } ^ { \mathrm { r } } - t _ { k } ^ { \mathrm { r } } } { \sqrt { t _ { k } ^ { \mathrm { r } } } } \leq \frac { t _ { k + 1 } ^ { \mathrm { c } } - t _ { k } ^ { \mathrm { c } } } { \sqrt { t _ { k } ^ { \mathrm { c } } } } + \frac { 1 } { \sqrt { t _ { k } ^ { \mathrm { c } } } } .
$$

The added cost is at most $K _ { 0 } / \sqrt { n } \leq 2 \sqrt { \tau ^ { \mathrm { h a l } } }$ : for log $u \geq 1 , K _ { 0 } \leq \log _ { 2 }$ log $u + 2 \leq \log u + 2 \leq 2 { \sqrt { u } }$ Deleting repeated rounded times removes only zero-length intervals and does not change the cost. There are then at most $K _ { 0 } \le K$ intervals.

If necessary, insert the smallest unused interior integer times until exactly K intervals remain. This is possible because $K \leq \tau ^ { \mathrm { h a l } } - n$ . Inserting a point cannot increase cost: for $t < t ^ { \prime } < t ^ { \prime \prime }$

$$
\frac { t ^ { \prime } - t } { \sqrt { t } } + \frac { t ^ { \prime \prime } - t ^ { \prime } } { \sqrt { t ^ { \prime } } } \leq \frac { t ^ { \prime \prime } - t } { \sqrt { t } } .
$$

The constructed integer partition therefore costs at most $( 4 \exp ( 1 / 2 ) + 2 ) \sqrt { \tau ^ { \mathrm { h a l } } } < 9 \sqrt { \tau ^ { \mathrm { h a l } } }$ . The exact minimizing partition has no greater cost, proving the claim. □

## E.1.3 Local sensitivity with an estimated request prior

Use the nominal notation from (8) and (9). For a capacity ratio $\beta$ and an estimated prior $p ^ { \prime }$ , define the joint perturbation size

$$
\Delta ( \beta , p ^ { \prime } ) : = \Delta ( \beta ) + \| p ^ { \prime } - p \| _ { 1 } .
$$

Upward changes in an inactive capacity are omitted for the same reason as in (9): they relax a constraint that is strictly slack at $x ^ { * }$ . For a stationary LP vector $x = ( z , q )$ , write $r ^ { \top } x : =$ $\begin{array} { r } { \sum _ { s , j , a } r ( s , j , a ) z _ { s j a } } \end{array}$ for its stationary reward.

Joint Lipschitz sensitivity. For any empirical LP solution $x ( \beta , p ^ { \prime } )$ whose core projection is well defined, let $K _ { \beta , p ^ { \prime } } ^ { \mathrm { p r o j } }$ denote the transition kernel generated by that projected policy under the true request prior p. Thus, for $s , s ^ { \prime } \in \mathcal { R } ^ { * }$ 2

$$
K _ { \beta , p ^ { \prime } } ^ { \mathrm { p r o j } } ( s , s ^ { \prime } ) : = \sum _ { j } p _ { j } \sum _ { a } \Pi _ { { \mathcal R } ^ { * } } ( x ( \beta , p ^ { \prime } ) ) ( a \mid s , j ) P ( s ^ { \prime } \mid s , a ) .
$$

Lemma E.3. There is a radius $\delta > 0$ such that, whenever $\Delta ( \beta , p ^ { \prime } ) < \delta$ , every optimal solution $x ( \beta , p ^ { \prime } )$ of the full empirical $L P G ^ { p ^ { \prime } } ( \beta )$ satisfies

$$
\| x ( \beta , p ^ { \prime } ) - x ^ { * } \| _ { 1 } = O ( \Delta ( \beta , p ^ { \prime } ) ) .\tag{E.5}
$$

In particular, there is $\eta > 0$ such that $q _ { s } ( \beta , p ^ { \prime } ) \ge \eta$ for every $s \in \mathcal { R } ^ { * }$ , and the core projection (20) is well defined. Moreover,

$$
\| \Pi _ { \mathcal { R } ^ { * } } ( \boldsymbol { x } ( \beta , p ^ { \prime } ) ) ( \cdot \mid \boldsymbol { s } , j ) - \pi _ { \rho } ( \cdot \mid \boldsymbol { s } , j ) \| _ { 1 } = O ( \Delta ( \beta , p ^ { \prime } ) ) , \qquad \boldsymbol { s } \in \mathcal { R } ^ { * } , \ j \in \mathcal { I } ,\tag{E.6}
$$

$$
\| K _ { \beta , p ^ { \prime } } ^ { \mathrm { p r o j } } ( s , \cdot ) - K ^ { * } ( s , \cdot ) \| _ { 1 } = O ( \Delta ( \beta , p ^ { \prime } ) ) , \qquad s \in \mathscr { R } ^ { * } .\tag{E.7}
$$

Proof. The proof mirrors the proof of Lemma D.1. Fix $( \beta , p ^ { \prime } )$ in the stated neighborhood and abbreviate

$$
\varepsilon : = \Delta ( \beta , p ^ { \prime } ) .
$$

Step 1: one fixed comparison polytope. Let ${ \mathcal { P } } ^ { + }$ be the stationary flow polytope under the true prior $p$ in which the resource constraints are retained only for $i \in I ^ { * } ;$ ; inactive resource constraints are dropped. The set is compact because $\textstyle \sum _ { s } q _ { s } = 1$ and $\begin{array} { r } { \sum _ { s , j , a } z _ { s j a } = 1 } \end{array}$ . We claim that $x ^ { * }$ is the unique maximizer of the reward over ${ \mathcal { P } } ^ { + }$ . Indeed, if $y \in \mathcal { P } ^ { + }$ had reward at least $G ^ { p } ( \rho )$ , then for suficiently small $\theta > 0$ the convex combination

$$
( 1 - \theta ) x ^ { * } + \theta y
$$

would satisfy every inactive resource constraint at $\rho ,$ because $x ^ { * }$ has the strict slack $\sigma _ { i } ^ { * } > 0$ there. The binding constraints are already satisfied by both points. A strict reward improvement would contradict optimality at $\rho ,$ and equal reward with $y \neq x ^ { * }$ would contradict uniqueness. Hence $x ^ { * }$ is the unique maximizer on $\mathcal { P } ^ { + }$

A linear objective over a fixed compact polytope with a unique maximizer is weakly sharp. When ${ \mathcal { P } } ^ { + } \neq \{ x ^ { * } \}$ , set

$$
\chi : = \operatorname* { m i n } _ { \boldsymbol { v } { \mathrm { ~ a ~ v e r t e x ~ o f ~ } } \mathcal { P } ^ { + } } \frac { G ^ { p } ( \boldsymbol { \rho } ) - \boldsymbol { r } ^ { \top } \boldsymbol { v } } { \Vert \boldsymbol { v } - \boldsymbol { x } ^ { * } \Vert _ { 1 } } > 0 .
$$

Every point of ${ \mathcal { P } } ^ { + }$ is a convex combination of its vertices, so linearity of the objective and the triangle inequality give

$$
G ^ { p } ( \rho ) - { r ^ { \top } } y \geq \chi \| y - x ^ { * } \| _ { 1 } , \qquad y \in \mathcal { P } ^ { + } .\tag{E.8}
$$

Step 2: a perturbed feasible point within $O ( \Delta )$ of $x ^ { * }$ . Keep the nominal policy $\pi _ { \rho }$ on $\mathcal { R } ^ { * }$ Under a nearby prior $p ^ { \prime } .$ , its induced kernel

$$
K _ { p ^ { \prime } } ^ { * } ( s , s ^ { \prime } ) : = \sum _ { j } p _ { j } ^ { \prime } \sum _ { a } \pi _ { \rho } ( a \mid s , j ) P ( s ^ { \prime } \mid s , a )
$$

remains irreducible on $\mathcal { R } ^ { * }$ , because the positive graph of $K ^ { * }$ persists when $p ^ { \prime }$ is close to $p .$ Its stationary distribution $q ^ { * } ( p ^ { \prime } )$ is Lipschitz in $p ^ { \prime }$ on the finite state space $\mathcal { R } ^ { * }$ ; this follows, for example, from the same fundamental-matrix representation used in Lemma C.3. Define

$$
z _ { s j a } ^ { 0 } ( p ^ { \prime } ) : = p _ { j } ^ { \prime } q _ { s } ^ { * } ( p ^ { \prime } ) \pi _ { \rho } ( a \mid s , j )
$$

on $\mathcal { R } ^ { * }$ and zero outside. Then $x ^ { 0 } ( p ^ { \prime } ) = ( z ^ { 0 } ( p ^ { \prime } ) , q ^ { * } ( p ^ { \prime } ) )$ is a stationary flow solution for prior $p ^ { \prime }$ and

$$
\| x ^ { 0 } ( p ^ { \prime } ) - x ^ { * } \| _ { 1 } = O ( \| p ^ { \prime } - p \| _ { 1 } ) .
$$

Its resource rates difer from the nominal rates by $O ( \| p ^ { \prime } - p \| _ { 1 } )$

Let $x ^ { \mathrm { n u l l } } ( p ^ { \prime } )$ be any stationary flow solution that always uses the null action; it has zero reward and zero resource use. Because $\rho _ { i } > 0$ for every binding resource, one can, for suficiently small $\varepsilon _ { i }$ choose $\theta = O ( \varepsilon )$ so that

$$
\widetilde { x } : = ( 1 - \theta ) x ^ { 0 } ( p ^ { \prime } ) + \theta x ^ { \mathrm { { n u l l } } } ( p ^ { \prime } )
$$

is feasible for $G ^ { p ^ { \prime } } ( \beta )$ . Binding-resource deficits of size at most $\Delta ( \beta )$ are absorbed by the $O ( \varepsilon )$ dilution, while for $i \notin I ^ { * }$ the nominal strict slack $\sigma _ { i } ^ { * }$ absorbs every permitted small downward

perturbation. Therefore

$$
\| \widetilde { \boldsymbol { x } } - \boldsymbol { x } ^ { * } \| _ { 1 } = O ( \varepsilon ) , \qquad \boldsymbol { r } ^ { \top } \widetilde { \boldsymbol { x } } \ge G ^ { p } ( \rho ) - O ( \varepsilon ) .\tag{E.9}
$$

If $x = x ( \beta , p ^ { \prime } )$ is any perturbed optimum, optimality gives $r ^ { \top } x \geq r ^ { \top } \widetilde { x } .$

Step 3: transfer the perturbed optimizer to the fixed nominal polytope. The vector x satisfies the nominal state-flow and normalization equations exactly. Its request-balance residual under the true prior is

$$
\sum _ { a } z _ { s j a } - p _ { j } q _ { s } = ( p _ { j } ^ { \prime } - p _ { j } ) q _ { s } ,
$$

so its total equality violation is $O ( \| p ^ { \prime } - p \| _ { 1 } )$ . For $i \in I ^ { * }$ its resource violation relative to $\rho _ { i }$ is at most $\Delta ( \beta )$ , while inactive resources are not constrained in ${ \mathcal { P } } ^ { + }$ . The ordinary Hofman bound for the fixed polytope ${ \mathcal { P } } ^ { + }$ therefore gives a point $y \in \mathcal { P } ^ { + }$ with

$$
\| x - y \| _ { 1 } = O ( \varepsilon ) .\tag{E.10}
$$

If ${ \mathcal { P } } ^ { + } = \{ x ^ { * } \}$ , then $y = x ^ { * }$ and (E.10) already proves (E.5). Otherwise, using (E.9) and (E.10),

$$
G ^ { p } ( \rho ) - r ^ { \top } y = O ( \varepsilon ) .
$$

The weak-sharpness inequality (E.8) implies $\| y - x ^ { * } \| _ { 1 } = O ( \varepsilon )$ , and the triangle inequality proves (E.5).

Finally, $q _ { s } ^ { * } > 0$ on $\mathcal { R } ^ { * }$ and $p _ { j } > 0$ . Every nominal positive action stays in $\mathcal { R } ^ { * }$ , so

$$
\sum _ { a \in A _ { \mathcal { R } ^ { * } } ( s ) } z _ { s j a } ^ { * } = p _ { j } q _ { s } ^ { * } > 0 .
$$

The sensitivity bound therefore makes every denominator in (20) uniformly positive locally. Normalizing the safe state-request-action masses gives (E.6), and the kernel estimate (E.7) follows by linearity and the fixed true prior $p .$ □

Basis stability under uniqueness and nondegeneracy. For any capacity ratio $\beta ,$ prior $p ^ { \prime } .$ and empirical stationary LP solution $x ( \beta , p ^ { \prime } ) = ( z ( \beta , p ^ { \prime } ) , q ( \beta , p ^ { \prime } ) )$ , define its stationary resourceconsumption rates by

$$
\overline { { c } } _ { i } ^ { p ^ { \prime } } ( \beta ) : = \sum _ { s , j , a } c _ { i } ( s , j , a ) z _ { s j a } ( \beta , p ^ { \prime } ) .
$$

Lemma E.4. Under Assumption 3.2, there is a $\delta _ { \mathrm { n d } } > 0$ such that, whenever $\Delta ( \beta , p ^ { \prime } ) < \delta _ { \mathrm { n d } }$ , the full empirical $G ^ { p ^ { \prime } } ( \beta )$ has the nominal basis as its unique optimal basis. Its positive z- and q-supports are exactly $\mathcal { E } ^ { * }$ and $\mathcal { R } ^ { * }$ , respectively. The core projection does nothing, every used action is nominal-dual tight, and for some $\eta _ { \mathrm { n d } } > 0$ 2

$$
\begin{array} { r } { \overline { { c } } _ { i } ^ { p ^ { \prime } } ( \beta ) = \beta _ { i } , } \end{array}
$$

$$
i \in I ^ { * } ,\tag{E.11}
$$

$$
\begin{array} { r } { \overline { { c } } _ { i } ^ { p ^ { \prime } } ( \beta ) \leq \beta _ { i } - \eta _ { \mathrm { n d } } , } \end{array}
$$

$$
i \notin I ^ { * } .\tag{E.12}
$$

The same statement holds for the restricted empirical LP used in the protective phase.

Proof. First verify that the row deletion fixed in Section 3 remains valid as the prior changes. Write all variables on the left-hand sides of (SF1)–(SF3), and add the resource slacks to (SF4). In a linear dependence among these full equality rows, let $u _ { s j } , \ v _ { s }$ , and w be the coeficients of the request, state-flow, and normalization rows. Every resource-row coeficient is zero because its slack variable occurs in that row alone. The z- and q-columns therefore give, respectively,

$$
u _ { s j } = \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v _ { s ^ { \prime } } \quad \mathrm { f o r ~ e v e r y ~ } ( s , j , a ) , \qquad - \sum _ { j } p _ { j } ^ { \prime } u _ { s j } + v _ { s } + w = 0 .
$$

The request-independent transition kernel makes $u _ { s j }$ a common value $u _ { s }$ for all requests at a state. Since $\textstyle \sum _ { j } p _ { j } ^ { \prime } = 1$ , the second equality becomes $u _ { s } = v _ { s } + w$ . Using the first equality for the null action and averaging against any stationary distribution of its finite-state transition matrix gives $w = 0$ . Consequently the full row-dependency space is exactly

$$
u _ { s j } = v _ { s } , \qquad \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v _ { s ^ { \prime } } = v _ { s } \quad \mathrm { f o r ~ e v e r y ~ } ( s , a ) , \qquad w = 0 ,
$$

with zero resource-row coeficients. These conditions are independent of $p ^ { \prime } .$ , and every such dependence also annihilates the right-hand side: the balance rows have zero right-hand sides, while the normalization and resource coeficients vanish. Thus the same retained rows form an equivalent full-row-rank system for every probability vector $p ^ { \prime }$ and every $\beta .$

In that fixed representation, the entries of the nominal basis matrix and of the nonbasic columns depend continuously on $p ^ { \prime } ,$ , while the resource right-hand sides depend continuously on $\beta .$ The nominal basis matrix is nonsingular, all nominal basic variables are strictly positive, and all nominal nonbasic reduced costs are strictly negative by Proposition 3.3. Hence these properties persist for suficiently small two-sided perturbations of $( \rho , p )$ . For an inactive resource, its positive nominal slack remains basic, and its column is the same resource-row unit vector for every $p ^ { \prime } .$ . Increasing that resource’s right-hand side therefore increases only its own basic slack, leaving all other basic variables and all reduced costs unchanged. Applying the two-sided neighborhood result first with each inactive capacity replaced by min $\{ \beta _ { i } , \rho _ { i } \}$ , and then restoring all upward increases, proves the result throughout the stated one-sided joint neighborhood. The binding and uniform inactive-slack conclusions follow from the continued basic solution, after shrinking the neighborhood if necessary. Thus the basis remains uniquely optimal with exactly the nominal positive support, so every used action is nominal-dual tight and the core projection is unchanged. Deleting variables outside $\mathcal { R } ^ { * }$ leaves the same nominal basis feasible and uniquely optimal in the restricted LP. □

Frozen core-projected epochs. For the next lemma, take a pair $( \beta , p ^ { \prime } )$ for which the core projection is well defined, a positive integer Λ, and an initial state in $\mathcal { R } ^ { * }$ . Run the frozen coreprojected policy $\Pi _ { \mathcal { R } ^ { * } } ( x ( \beta , p ^ { \prime } ) )$ under the true request prior $p ,$ and number the epoch’s pre-request instants by $t = 1 , \ldots , \Lambda$ . Let $\mathcal { F } ^ { t }$ denote the full history before request t. Fix a bounded functional of the current state, fresh request, selected action, and next state, and denote its value in round t by $Y ^ { t }$ . Under the frozen projected policy, define its statewise conditional mean to be the one-step

kernel expectation from state s:

$$
f _ { \beta , p ^ { \prime } } ( s ) : = \mathbb { E } [ Y ^ { t } \mid S ^ { t } = s ] .
$$

Thus $\mathbb { E } [ Y ^ { t } \mid { \mathcal { F } } ^ { t } ] = f _ { \beta , p ^ { \prime } } ( S ^ { t } )$ . Under the nominal policy $\pi _ { \rho } ,$ let $f ^ { * } ( s )$ denote the corresponding statewise mean and let $\overline { { f } } ^ { * }$ denote its stationary mean. Rewards and resource consumptions are examples of such functionals.

Under Assumption 3.2, for pairs in the joint-basis neighborhood of Lemma E.4, write $\pi _ { \beta , p ^ { \prime } }$ for the local empirical policy, which equals its core projection, and let $\overline { { c } } _ { i } ( \beta , p ^ { \prime } )$ denote its stationary rate of resource-i consumption when it is evaluated under the true request prior.

Lemma E.5. $\varPsi \Delta ( \beta , p ^ { \prime } ) < \delta$ , then

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 1 \leq n \leq \Lambda } \left| \sum _ { t = 1 } ^ { n } ( Y ^ { t } - \overline { { f } } ^ { \ast } ) \right| \right] = O ( \sqrt { \Lambda } + \Lambda \Delta ( \beta , p ^ { \prime } ) + 1 ) ,\tag{E.13}
$$

and, for every $x > 0$ ，

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq n \leq \Lambda } \left| \sum _ { t = 1 } ^ { n } ( Y ^ { t } - \overline { { f } } ^ { * } ) \right| > \Theta ( \Lambda \Delta ( \beta , p ^ { \prime } ) + 1 ) + x \right) \leq O \left( \exp \left( - \Omega \left( \frac { x ^ { 2 } } { \Lambda } \right) \right) \right) .\tag{E.14}
$$

All constants are uniform over the local neighborhood.

Under Assumption 3.2, the true-prior stationary resource rates satisfy

$$
\left| \overline { { c } } _ { i } ( \beta , p ^ { \prime } ) - \beta _ { i } \right| = O ( \| p ^ { \prime } - p \| _ { 1 } ) ,
$$

$$
i \in I ^ { * } ,\tag{E.15}
$$

$$
\overline { { c } } _ { i } ( \beta , p ^ { \prime } ) \leq \beta _ { i } - \frac 1 2 \eta _ { \mathrm { n d } } ,
$$

$$
i \notin I ^ { * } ,\tag{E.16}
$$

whenever the neighborhood is small enough, and

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq n \leq \Lambda } \left| \sum _ { t = 1 } ^ { n } ( C _ { i } ^ { t } - \bar { c } _ { i } ( \beta , p ^ { \prime } ) ) \right| > x \right) \leq O \left( \exp \left( - \Omega \bigg ( \frac { x ^ { 2 } } { \Lambda } \bigg ) \right) \right)\tag{E.17}
$$

for x above a fixed constant.

Proof. The proof partially mirrors the proof of Lemma D.3. Abbreviate $\varepsilon : = \Delta ( \beta , p ^ { \prime } )$ . Because the core projection uses only actions in $\mathcal { A } _ { \mathcal { R } ^ { \ast } } ( s )$ , the state remains in $\mathcal { R } ^ { * }$ . By (E.6) and (E.7), both $f _ { \beta , p ^ { \prime } }$ and the true kernel $K _ { \beta , p ^ { \prime } } ^ { \mathrm { p r o j } }$ difer by $O ( \varepsilon )$ from their nominal counterparts on $\mathcal { R } ^ { * }$ . Let v be the bounded centered solution of the nominal Poisson equation

$$
( I - K ^ { * } ) v = f ^ { * } - \overline { { f } } ^ { * } { \bf 1 } .
$$

With

$$
\begin{array} { r } { M ^ { t + 1 } : = Y ^ { t } - f _ { \beta , p ^ { \prime } } ( S ^ { t } ) + v ( S ^ { t + 1 } ) - ( K _ { \beta , p ^ { \prime } } ^ { \mathrm { p r o j } } v ) ( S ^ { t } ) , } \end{array}
$$

we have $\mathbb { E } [ M ^ { t + 1 } \mid { \mathcal { F } } ^ { t } ] = 0$ , and the martingale has uniformly bounded increments. The pathwise

decomposition is

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { n } ( Y ^ { t } - \overline { { f } } ^ { * } ) = \sum _ { t = 1 } ^ { n } M ^ { t + 1 } + v ( S ^ { 1 } ) - v ( S ^ { n + 1 } ) } } \\ & { + \sum _ { t = 1 } ^ { n } \bigl [ f _ { \beta , p ^ { \prime } } ( S ^ { t } ) - \overline { { f } } ^ { * } - v ( S ^ { t } ) + ( K _ { \beta , p ^ { \prime } } ^ { \mathrm { p r o j } } v ) ( S ^ { t } ) \bigr ] . } \end{array}
$$

The last summand is uniformly $O ( \varepsilon )$ . Doob’s square-integrable maximal inequality proves (E.13), and the maximal Azuma–Hoefding inequality proves (E.14).

Under nominal uniqueness and nondegeneracy, the empirical policy has the fixed support $\mathcal { E } ^ { * }$ Its kernel under either $p ^ { \prime }$ or $p$ contains the same uniformly positive strongly connected graph from Proposition 3.5. Hence both kernels are uniformly irreducible and their stationary distributions are Lipschitz in the request weights. Replacing $p ^ { \prime }$ by $p$ while holding the policy fixed changes the stationary resource rate by $O ( \| p ^ { \prime } - p \| _ { 1 } )$ . Combining this fact with (E.11) and (E.12) gives (E.15) and (E.16). Uniform Poisson bounds for this compact family of true kernels, followed by the same martingale argument as in Lemma C.8, give (E.17). □

## E.1.4 Inventory control for the universal upper bounds

We start with a general analysis.

At every empirical resolving point $t _ { k }$ , define

$$
\begin{array} { r l } & { H _ { k } : = H ^ { t _ { k } } , \quad X ^ { k } : = B ^ { t _ { k } } - H _ { k } \rho , \quad D _ { k } : = { \mathcal { D } } ( X ^ { k } ) , } \\ & { \beta ^ { k } : = \cfrac { B ^ { t _ { k } } } { H _ { k } } , \quad e _ { k } : = \| \widehat { p } ^ { t _ { k } } - p \| _ { 1 } , \quad \Delta _ { k } : = \Delta \left( \cfrac { B ^ { t _ { k } } } { H ^ { t _ { k } } } , \widehat { p } ^ { t _ { k } } \right) . } \end{array}\tag{E.18}
$$

The definitions give

$$
H _ { k } \Delta \bigl ( B ^ { t _ { k } } / H _ { k } \bigr ) \leq D _ { k } , \qquad \Delta _ { k } \leq \frac { D _ { k } } { H _ { k } } + e _ { k } .\tag{E.19}
$$

This appendix uses four of the five check types in the analytical-check convention of Appendix C.1: locality, resource near-depletion, epoch movement, and prior error. Deterministic midpoint, squareroot, and fixed-cutof endpoints are stopping boundaries rather than checks. We first define the two pathwise checks whose thresholds do not depend on a particular local neighborhood. The resource near-depletion check is applied before each action and fires at completed-round time n if

$$
\operatorname* { m i n } _ { i \in [ m ] } B _ { i } ^ { n } \leq { \bar { c } } .\tag{E.20}
$$

The stopped comparison then terminates before taking that action. This is not the actual resourcedepletion rule (1). Since each coordinate of one-period consumption is at most ${ \overline { { c } } } ,$ actual depletion cannot occur before this check fires. This is also the resource near-depletion check used in Appendix D.

Given a positive bufer $d _ { k }$ , the epoch-movement check at a reached endpoint $t _ { k + 1 }$ fails if

$$
\left| \beta _ { i } ^ { k + 1 } - \beta _ { i } ^ { k } \right| > d _ { k } \quad \mathrm { f o r ~ s o m e ~ } i \in I ^ { * } , \quad \mathrm { o r } \quad \beta _ { i } ^ { k + 1 } < \beta _ { i } ^ { k } - d _ { k } \quad \mathrm { f o r ~ s o m e ~ } i \notin I ^ { * } .\tag{E.21}
$$

This is the separate epoch-movement check used in Lemma C.9. Only the late nondegenerate comparison uses this second check.

The next lemma gives stopped square-root control through the midpoint.

Control through the midpoint. The locality check is applied at a resolving point before an action. It passes precisely when the resolving pair is local, meaning that $\Delta ( \beta , \widehat { p } ) < \delta$ and its core projection is defined; a nonlocal pair is a failure of this check. For a post-entry horizon $T .$ , set

$$
\tau ^ { \mathrm { l r n } } : = \lceil ( \log T ) ^ { 2 } \rceil , \qquad \tau ^ { \mathrm { h a l } } : = \lfloor T / 2 \rfloor .
$$

Given an entrance history, start the sparse-balanced phase at completed-round time $\tau ^ { \mathrm { l r n } }$ . Let $\sigma$ be the first nonlocal resolving point strictly before $\tau ^ { \mathrm { h a l } }$ , the first activation of the resource near-depletion check (E.20) before $\tau ^ { \mathrm { h a l } }$ , or $\tau ^ { \mathrm { h a l } }$ , whichever occurs first. The locality and resource near-depletion checks at the midpoint belong to the late-phase analysis. For this stopped path, define

$$
D ^ { * } : = \operatorname* { m a x } _ { \tau ^ { \mathrm { l r n } } \leq n \leq \sigma } { \cal D } \big ( B ^ { n } - H ^ { n } \rho \big ) .
$$

The maximum includes the first failed resolving point or resource near-depletion prefix and, if reached, the midpoint; the path is not continued beyond a resource near-depletion stop.

Index rewards, actions, and consumptions by completed-round time as well. Before analytical truncation, define the one-period nominal-dual loss $\ell ^ { t }$ by

$$
\begin{array} { r } { \mathbb { E } \left[ R ^ { t } + F ( H ^ { t } - 1 , B ^ { t + 1 } , S ^ { t + 1 } ) ~ \Big | ~ \mathcal { F } ^ { t } \right] = F ( H ^ { t } , B ^ { t } , S ^ { t } ) - \ell ^ { t } . } \end{array}\tag{E.22}
$$

The nominal dual inequalities give $\ell ^ { t } \geq 0 .$ . In a local projected epoch, Lemma E.3 gives $\ell ^ { t } = O ( \Delta _ { k } )$ ; if Lemma E.4 applies, every used action is nominally tight and $\ell ^ { t } = 0$ . For the stopped first-half path, set

$$
\mathcal { L } _ { \mathrm { s p } } : = \sum _ { t = \tau ^ { \mathrm { l r n } } } ^ { \sigma - 1 } \ell ^ { t } .
$$

Because the locality and resource near-depletion checks are applied before an action, $\mathcal { L } _ { \mathrm { s p } }$ is exactly the nominal-dual loss over the executed rounds of reached local sparse-balanced epochs, including the executed prefix of an epoch interrupted by resource near-depletion.

Lemma E.6. Fix an entrance history and take the post-entry horizon suficiently large that $n _ { 0 } < \tau ^ { \mathrm { l r n } } < \tau ^ { \mathrm { h a l } }$ . Suppose the initial inventory is large enough, relative to the bounded one-period costs, to preclude activation of the resource near-depletion check through $\tau ^ { \mathrm { l r n } }$ , regardless of the

protective actions. Then the following four bounds hold for the stopped path:

$$
\begin{array} { r l r } & { } & { \mathbb { E } \Big [ D ^ { * } \Bigm | \mathcal { F } ^ { 0 } \Bigm ] = O \Big ( \sqrt { T } + \mathcal { D } ( X ^ { \mathrm { e n t } } ) \Big ) , } \\ & { } & { \mathbb { E } \Big [ H ^ { \sigma } \mathbf { 1 } \{ \sigma < \tau ^ { \mathrm { h a l } } \} \Bigm | \mathcal { F } ^ { 0 } \Bigm ] = O \Big ( \sqrt { T } + \mathcal { D } ( X ^ { \mathrm { e n t } } ) \Big ) , } \\ & { } & { \mathbb { E } \Big [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \mathbf { 1 } \{ \sigma < \tau ^ { \mathrm { h a l } } \} \Bigm | \mathcal { F } ^ { 0 } \Bigm ] = O \Big ( \sqrt { T } + \mathcal { D } ( X ^ { \mathrm { e n t } } ) \Big ) , } \\ & { } & { \mathbb { E } \Big [ \mathcal { L } _ { \mathrm { s p } } \Bigm | \mathcal { F } ^ { 0 } \Bigm ] = O \Big ( \sqrt { T } + \mathcal { D } ( X ^ { \mathrm { e n t } } ) \Big ) . } \end{array}\tag{E.23}
$$

These conditional bounds have constants uniform over entrance histories satisfying the hypotheses. When they are averaged over the regular-entry histories on which the lemma is invoked, (C.11) makes the contribution of the entrance imbalance $O ( 1 )$

Proof. During the first $\tau ^ { \mathrm { { l r n } } }$ post-entry rounds, all costs are bounded, so deterministically

$$
\boldsymbol { \mathcal { D } } ( \boldsymbol { B } ^ { \tau ^ { \mathrm { l r n } } } - \boldsymbol { H } ^ { \tau ^ { \mathrm { l r n } } } \boldsymbol { \rho } ) = \boldsymbol { O } ( \tau ^ { \mathrm { l r n } } ) + \boldsymbol { \mathcal { D } } ( \boldsymbol { X } ^ { \mathrm { e n t } } ) .\tag{E.24}
$$

Here the entrance imbalance is fixed under the entrance conditioning.

Consider the sparse-balanced epochs before the analytical stop. All epoch sums below are restricted to epochs reached with a local resolving pair. In these sums, use the full scheduled epoch lengths as upper bounds where convenient. For estimates involving the prior, define every scheduled empirical prior from an infinite i.i.d. request sequence, including observations after any analytical stop; dropping the reach indicator then only enlarges a nonnegative bound. For resource $i ,$ apply the proof of Lemma E.5 with the nominal resource Poisson solution $v _ { i } ^ { * }$ . Because the same nominal corrector is used in every epoch, the martingale diferences concatenate across resolving times and the Poisson boundary term telescopes globally. Set these martingales to zero through time $\tau ^ { \mathrm { l r n } }$ stop them at $\sigma _ { \mathrm { { ; } } }$ , and extend them constantly thereafter. Thus, there exist a bounded-increment martingale $M _ { i }$ and a remainder $U _ { i }$ such that, for every $\tau ^ { \mathrm { l r n } } \leq n \leq \sigma$ 2

$$
\begin{array} { r l r } {  { \sum _ { t = \tau ^ { \mathrm { l r n } } } ^ { n - 1 } ( C _ { i } ^ { t } - \bar { c } _ { i } ( \rho ) ) = M _ { i } ( n ) + v _ { i } ^ { * } ( S ^ { \tau ^ { \mathrm { l r n } } } ) - v _ { i } ^ { * } ( S ^ { n } ) + U _ { i } ( n ) , } } \\ & { } & { \vert U _ { i } ( n ) \vert \le O ( \sum _ { k : t _ { k } < n } ( \operatorname* { m i n } \{ n , t _ { k + 1 } \} - t _ { k } ) \Delta _ { k } ) . } \end{array}
$$

Here $\overline { { c } } _ { i } ( \rho ) = \rho _ { i }$ for $i \in I ^ { * }$ , whereas

$$
\bar { c } _ { i } ( \rho ) = \rho _ { i } - \sigma _ { i } ^ { * } \mathrm { ~ ( ~ } i \notin I ^ { * } \mathrm { ) } .
$$

Consequently, with $Z : = \mathrm { m a x } _ { i } \mathrm { m a x } _ { \tau ^ { \mathrm { l r n } } \leq n \leq \tau ^ { \mathrm { h a l } } } | M _ { i } ( n ) |$ , every such prefix satisfies

$$
\begin{array} { r l r } & { } & { \mathcal { D } ( B ^ { n } - H ^ { n } \rho ) \leq O \left( 1 + \mathcal { D } ( B ^ { \tau ^ { \mathrm { l r n } } } - H ^ { \tau ^ { \mathrm { l r n } } } \rho ) + Z \right) } \\ & { } & { \qquad + O \left( { \displaystyle \sum _ { k : t _ { k } < n } } \Lambda _ { k } \left( \frac { D _ { k } } { H _ { k } } + e _ { k } \right) \right) , } \end{array}\tag{E.25}
$$

where $D _ { k } , \boldsymbol { e } _ { k }$ , and $\Delta _ { k }$ are defined in (E.18) and $D _ { k } \leq D ^ { * }$ . Since $H _ { k } \ge T / 2$ in this phase and the scheduled blocks occupy at most the first half of the horizon,

$$
\sum _ { k } { \frac { \Lambda _ { k } } { H _ { k } } } \leq { \frac { 2 } { T } } \sum _ { k } \Lambda _ { k } \leq 1 .
$$

Set

$$
A : = \Theta \left( 1 + \mathcal { D } \big ( B ^ { \tau ^ { \mathrm { l r n } } } - H ^ { \tau ^ { \mathrm { l r n } } } \rho \big ) + Z + \sum _ { k } \Lambda _ { k } e _ { k } \right) , \qquad a _ { k } : = \Theta \left( \frac { \Lambda _ { k } } { H _ { k } } \right) .
$$

At a reached resolving point $t _ { k }$ , (E.25) contains only deviations from earlier resolving points, and hence

$$
D _ { k } \leq A + \sum _ { k ^ { \prime } < k } a _ { k ^ { \prime } } D _ { k ^ { \prime } } .
$$

Direct induction therefore gives, also at a first failed resolving point,

$$
D _ { k } \leq A \prod _ { k ^ { \prime } < k } ( 1 + a _ { k ^ { \prime } } ) .
$$

Indeed, the induction step is exactly the identity

$$
1 + \sum _ { k ^ { \prime } < k } a _ { k ^ { \prime } } \prod _ { \ell < k ^ { \prime } } ( 1 + a _ { \ell } ) = \prod _ { k ^ { \prime } < k } ( 1 + a _ { k ^ { \prime } } ) .
$$

Moreover, the total amplification is bounded because

$$
\prod _ { k } ( 1 + a _ { k } ) \leq \exp \left( \sum _ { k } a _ { k } \right) = O ( 1 ) .
$$

Therefore, substituting the resolving-point bounds back into (E.25) gives, for every stopped prefix $n ,$ including a final epoch interrupted by resource near-depletion,

$$
{ \mathcal { D } } ( B ^ { n } - H ^ { n } \rho ) \leq A \prod _ { k : t _ { k } < n } ( 1 + a _ { k } ) = O ( A ) .
$$

Taking the maximum over n yields

$$
D ^ { * } \leq O \left( 1 + \mathcal { D } ( { B ^ { \tau } } ^ { \mathrm { l r n } } - { H ^ { \tau } } ^ { \mathrm { l r n } } \rho ) + Z + \sum _ { k } \Lambda _ { k } e _ { k } \right) .\tag{E.26}
$$

Doob’s square-integrable maximal inequality gives

$$
\mathbb { E } Z = O ( { \sqrt { T } } ) .
$$

By (E.1) and Lemma E.2,

$$
\mathbb { E } \sum _ { k } \Lambda _ { k } e _ { k } \le O \left( \sum _ { k } \frac { \Lambda _ { k } } { \sqrt { t _ { k } } } \right) = O ( \sqrt { T } ) .
$$

Together with (E.24), this proves (E.23).

We next charge the analytical stop. A prior-estimation failure $e _ { k } \geq \delta / 2$ at some sparse resolving point has probability at most

$$
K \exp ( - \Omega ( \delta ^ { 2 } \tau ^ { \mathrm { l r n } } ) ) = \exp ( - \Omega ( ( \log T ) ^ { 2 } ) ) ,
$$

so its remaining-horizon contribution is O(1). If instead the capacity ratio is the first nonlocal component, (E.19) gives

$$
H _ { k } \leq \frac { 2 D _ { k } } { \delta } \leq \frac { 2 D ^ { * } } { \delta } ,
$$

and its expected remaining horizon is ${ O } ( { \sqrt { T } } + { \mathcal { D } } ( X ^ { \mathrm { e n t } } ) )$ . Finally, at a resource near-depletion stop, some coordinate satisfies $B _ { i } ^ { \sigma } \leq \overline { { c } } .$ . The definition of the prefix maximum therefore gives pathwise

$$
H ^ { \sigma } \operatorname* { m i n } _ { i } \rho _ { i } \leq \bar { c } + D ^ { * } .
$$

Its expectation is therefore ${ O } ( { \sqrt { T } } + { \mathcal { D } } ( X ^ { \mathrm { e n t } } ) )$ as well.

The same bounds control the positive terminal potential. At every stop, including a resource near-depletion prefix, each binding inventory satisfies $B _ { i } ^ { \sigma } \leq H ^ { \sigma } \rho _ { i } + D ^ { * }$ . Since $\lambda _ { i } ^ { * } = 0$ for inactive resources, direct substitution into $F$ gives, for the first-half analytical stopping time $\sigma _ { : }$

$$
\mathbb { E } \Big [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \mathbf { 1 } \{ \sigma < \tau ^ { \mathrm { h a l } } \} \Big ] = O \big ( \sqrt { T } + { \mathcal { D } } ( X ^ { \mathrm { e n t } } ) \big ) .
$$

At a prior-estimation stop, the envelope is $O ( 1 + T + D ^ { * } )$ ; its horizon term has exponentially small probability, and the deviation term is covered by (E.23). On successful midpoint paths, the potential is carried into the next phase and is not charged here.

For reward comparison, fix any nominal optimal dual solution and define its nonnegative action slacks as in the proof of Theorem 5.2. On $\mathcal { R } ^ { * }$ the state slack is zero. By (E.6), a local projected policy places all but $O ( \Delta _ { k } )$ probability on nominally tight actions, so (E.22) gives $\ell ^ { t } = O ( \Delta _ { k } )$ in epoch k. Hence

$$
\mathbb { E } [ \mathcal { L } _ { \mathrm { s p } } ] \leq O \left( \mathbb { E } \sum _ { k } \Lambda _ { k } \Delta _ { k } \right) \leq O \left( \mathbb { E } \left[ D ^ { * } \sum _ { k } \frac { \Lambda _ { k } } { H _ { k } } \right] + \mathbb { E } \sum _ { k } \Lambda _ { k } e _ { k } \right) = O \big ( \sqrt { T } + \mathcal { D } ( X ^ { \mathrm { e n t } } ) \big ) ,
$$

which proves the final claim.

Late-half control. Fix an entrance history and write $T$ for its post-entry horizon. Let $\mathcal { R } _ { 0 }$ be the event that the first-half comparison reaches the midpoint without an earlier permanent stop and that $S ^ { \tau ^ { \mathrm { h a l } } } \in \mathcal { R } ^ { * }$ . Define the midpoint inventory deviation on this event by

$$
\boldsymbol { D } _ { 0 } : = \boldsymbol { \mathcal { D } } \big ( \boldsymbol { B } ^ { \tau ^ { \mathrm { h a l } } } - \boldsymbol { H } ^ { \tau ^ { \mathrm { h a l } } } \rho \big ) , \qquad \mathrm { o n } ~ \mathcal { R } _ { 0 } ,
$$

and set $D _ { 0 } = 0$ on $ { \mathcal { R } } _ { 0 } ^ { \mathrm { ~ c ~ } }$ . Define

$$
J : = \operatorname* { m i n } \{ k \geq 0 : \alpha ^ { k } \leq 1 / 2 \} .
$$

For $0 \le k \le J$ , define the late epoch boundaries by

$$
t _ { k } : = T - h _ { k } , \qquad t _ { 0 } = \tau ^ { \mathrm { h a l } } ,
$$

and, for $0 \le k < J$ , define

$$
\Lambda _ { k } : = t _ { k + 1 } - t _ { k } = h _ { k } - h _ { k + 1 } .
$$

For the proof-only late comparison starting on $\mathcal { R } _ { 0 }$ , let $\sigma$ be the first nonlocal resolving instant, the first activation of the resource near-depletion check (E.20), or the completed-round instant $t _ { J } ,$ whichever occurs first. This stopping rule includes the locality and resource near-depletion checks at the midpoint itself. Set $\sigma = \tau ^ { \mathrm { h a l } }$ on $\mathcal { R } _ { 0 } ^ { \mathrm { ~ c ~ } }$ , and define the stopped late-phase nominal-dual loss by

$$
\mathcal { L } _ { \mathrm { l a t e } } : = \mathbf { 1 } \{ \mathcal { R } _ { 0 } \} \sum _ { t = \tau ^ { \mathrm { h a l } } } ^ { \sigma - 1 } \ell ^ { t } .
$$

Because every check is applied before an action, this sum contains exactly the executed latecomparison rounds. It is empty if a check fails at the midpoint and includes the executed prefix of an epoch interrupted by resource near-depletion.

Lemma E.7. Suppose $T$ is suficiently large that the late schedule has at least J epochs. Then the following three bounds hold for the comparison and stopping rule defined above:

$$
\begin{array} { r l r } & { } & { { \mathbb { E } } \Big [ H ^ { \sigma } { \bf 1 } \{ { \mathcal { R } } _ { 0 } \} \Big | { \mathcal { F } } ^ { 0 } \Big ] = O \left( \sqrt { T } + { \mathbb { E } } \Big [ D _ { 0 } { \bf 1 } \{ { \mathcal { R } } _ { 0 } \} \Big | { \mathcal { F } } ^ { 0 } \Big ] \right) , } \\ & { } & { { \mathbb { E } } \Big [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } { \bf 1 } \{ { \mathcal { R } } _ { 0 } \} \Big | { \mathcal { F } } ^ { 0 } \Big ] = O \left( \sqrt { T } + { \mathbb { E } } \Big [ D _ { 0 } { \bf 1 } \{ { \mathcal { R } } _ { 0 } \} \Big | { \mathcal { F } } ^ { 0 } \Big ] \right) , } \\ & { } & { { \mathbb { E } } \Big [ { \mathcal { L } } _ { \mathrm { l a t e } } \Big | { \mathcal { F } } ^ { 0 } \Big ] = O \left( \sqrt { T } + { \mathbb { E } } \Big [ D _ { 0 } { \bf 1 } \{ { \mathcal { R } } _ { 0 } \} \Big | { \mathcal { F } } ^ { 0 } \Big ] \right) . } \end{array}
$$

Proof. All expectations and probabilities below are conditional on $\mathcal { F } ^ { 0 }$ unless a later filtration is displayed explicitly. The number J of relevant late epochs depends only on α and is therefore fixed. At every late resolving time at least $T / 2 - O ( 1 )$ requests have already been observed, so, for $0 \leq k < J _ { : }$

$$
\mathbb { E } [ e _ { k } ] = O ( T ^ { - 1 / 2 } ) , \qquad \mathbb { P } ( e _ { k } > \delta / 2 ) \le \exp ( - \Omega ( T ) ) .\tag{E.27}
$$

Retain the event $\mathcal { R } _ { 0 }$ defined above. For $1 \leq k < J$ , let $\mathcal { R } _ { k }$ be the event that epoch k is reached before its locality and resource near-depletion checks. For $0 \leq k < J .$ , let $\mathcal { G } _ { k } \subseteq \mathcal { R } _ { k }$ be the event that those checks pass. Finally, let $\mathcal { R } _ { J }$ be the event that the deterministic cutof $t _ { J }$ is reached; no new checks are assigned there. Each event defined at $t _ { k }$ is $\mathcal { F } ^ { t _ { k } }$ -measurable, and $\mathcal { R } _ { k + 1 } \subseteq \mathcal { G } _ { k }$ for $0 \leq k < J$

For $0 \leq k \leq J ,$ set $D _ { k } = 0$ of $\mathcal { R } _ { k }$ . For $0 \leq k < J .$ , define $e _ { k }$ from the full i.i.d. request sequence even on stopped paths. On $\mathcal { G } _ { k }$ , freeze the selected policy on an infinite-resource copy and let $D _ { k , \mathrm { h y p } } ^ { * }$ be the maximum of its one-sided inventory deviation over all prefixes of this epoch, including its initial and terminal points. Write $D _ { k + 1 , \mathrm { h y p } }$ for the deviation at that terminal point, and set both hypothetical deviations to zero of $\mathcal { G } _ { k }$ . For $0 \le k < J$ , Lemma E.5, with the nominal drift retained

for inactive resources, gives

$$
\begin{array} { r } { \mathbb { E } [ D _ { k + 1 , \mathrm { h y p } } \ | \ \mathcal { F } ^ { t _ { k } } ] \leq \mathbb { E } [ D _ { k , \mathrm { h y p } } ^ { * } \ | \ \mathcal { F } ^ { t _ { k } } ] \leq O \left( D _ { k } + \sqrt { h _ { k } } + h _ { k } e _ { k } + 1 \right) . } \end{array}
$$

The realized $e _ { k }$ must remain in this conditional bound. Multiply by $\mathbf { 1 } \{ \mathcal { G } _ { k } \}$ and then take expectations. Only at this point use (E.27), obtaining

$$
\mathbb { E } [ D _ { k + 1 } \mathbf { 1 } \{ \mathcal { R } _ { k + 1 } \} ] \leq O \left( \mathbb { E } [ D _ { k } \mathbf { 1 } \{ \mathcal { R } _ { k } \} ] + \sqrt { h _ { k } } + \frac { h _ { k } } { \sqrt { T } } + 1 \right) .
$$

Because J is fixed, induction bounds all event-weighted endpoint deviations and prefix maxima:

$$
\mathbb { E } [ D _ { k } { \bf 1 } \{ \mathcal { R } _ { k } \} ] + \mathbb { E } [ D _ { k , \mathrm { h y p } } ^ { * } { \bf 1 } \{ \mathcal { G } _ { k } \} ] = O ( \sqrt { T } + \mathbb { E } [ D _ { 0 } { \bf 1 } \{ \mathcal { R } _ { 0 } \} ] ) .
$$

If the capacity ratio is the first component to become nonlocal while $e _ { k } < \delta / 2$ , (E.19) gives $H ^ { \sigma } \leq 2 D _ { k } / \delta$ . A prior nonlocality has exponentially small probability by (E.27), which is used without conditioning on $\mathcal { R } _ { k }$ . At a resource near-depletion stop inside a local epoch, some inventory is at most ${ \overline { { c } } } ,$ so

$$
H ^ { \sigma } \operatorname* { m i n } _ { i } \rho _ { i } \le \overline { c } + D _ { k , \mathrm { h y p } } ^ { * } .
$$

The same estimate when the resource near-depletion check fires at an epoch start uses $D _ { k }$ . Finally, the clipped power recursion gives $h _ { J } = O ( { \sqrt { T } } )$ . These observations give the explicit remaining-horizon bound

$$
\mathbb { E } \big [ H ^ { \sigma } \mathbf { 1 } \{ \mathcal { R } _ { 0 } \} \big ] = O \left( \sqrt { T } + \mathbb { E } \Big [ D _ { 0 } \mathbf { 1 } \{ \mathcal { R } _ { 0 } \} \ \Big | \ \mathcal { F } ^ { 0 } \Big ] \right) .
$$

For binding resources the stopped inventory satisfies $B _ { i } ^ { \sigma } \leq H ^ { \sigma } \rho _ { i } + D _ { k , \mathrm { h y p } } ^ { * }$ at a within-epoch stop, and the corresponding bound with $D _ { k }$ at a resolving stop. Since inactive resource prices vanish, direct substitution in $F$ gives the same bound on expected positive terminal potential. In the rare prior-failure case, bound the positive terminal potential by $O ( 1 + T + D _ { k } )$ using the preceding event-weighted estimates. Thus

$$
{ \mathbb E } \Big [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } \mathbf { 1 } \{ \mathcal { R } _ { 0 } \} \Big ] = O \left( \sqrt { T } + { \mathbb E } \Big [ D _ { 0 } \mathbf { 1 } \{ \mathcal { R } _ { 0 } \} \Big | \ \mathcal { F } ^ { 0 } \Big ] \right) .
$$

Finally, every pre-σ action in a late epoch k is taken only on $\mathcal { G } _ { k }$ , and its one-period dual loss is $O ( \Delta _ { k } )$ ). Using the full scheduled block length to upper-bound a possibly interrupted terminal epoch gives

$$
\begin{array} { r l r } & { } & { \mathbb { E } [ \mathcal { L } _ { \mathrm { l a t e } } ] \leq O \left( \mathbb { E } \sum _ { k < J } \mathbf { 1 } \{ \mathcal { G } _ { k } \} \Lambda _ { k } \Delta _ { k } \right) \leq O \left( \displaystyle \sum _ { k < J } \left( \mathbb { E } [ D _ { k } \mathbf { 1 } \{ \mathcal { R } _ { k } \} ] + \frac { h _ { k } } { \sqrt { T } } \right) \right) } \\ & { } & { \quad = O \left( \sqrt { T } + \mathbb { E } \Big [ D _ { 0 } \mathbf { 1 } \{ \mathcal { R } _ { 0 } \} \Big | \mathcal { F } ^ { 0 } \Big ] \right) . } \end{array}
$$

This proves the lemma.

The unique, nondegenerate case needs a stronger, horizon-weighted stability statement. Fix κ as

in (C.26), so

$$
0 < \kappa < 1 - \frac { 1 } { 2 \alpha } , \qquad \zeta = 2 \alpha ( 1 - \kappa ) - 1 > 0 .
$$

Stabilization through the midpoint under uniqueness and nondegeneracy. By Proposition 3.6, $\mathcal { R } ^ { \ast } = \mathcal { S }$ . We therefore use completed-round notation on the original horizon, starting from $B ^ { 0 } = T \rho$ at time 0.

Lemma E.8. Under Assumptions 3.2 and $\ 3 . 4 ,$

$$
\mathbb { E } \left[ \sum _ { \ell = 0 } ^ { \tau ^ { \mathrm { { l r n } } } - 1 } \ell ^ { t } \right] = O ( 1 ) .\tag{E.28}
$$

Moreover, $f o r$ every prescribed $Q > 0$ , the sparse-balanced phase reaches the midpoint without activating the resource near-depletion check and satisfies

$$
\Delta ( { B ^ { \tau } } ^ { \mathrm { h a l } } / { H ^ { \tau } } ^ { \mathrm { h a l } } ) < \delta _ { \mathrm { n d } } / 4 ,\tag{E.29}
$$

except with probability $O ( T ^ { - Q } )$ , for all suficiently large $T$

Proof. Shrink $\delta _ { \mathrm { n d } }$ in advance so that $\delta _ { \mathrm { n d } } < \frac { 1 } { 4 }$ min ${ \mathrm { ~ \it ~  ~ } } \rho _ { i }$ . Since $B ^ { 0 } / T = \rho$ , bounded one-period costs and the ratio recursion give, for every $n \leq \tau ^ { \mathrm { { l r n } } }$

$$
\begin{array} { r l } & { \left| \frac { B _ { i } ^ { n } } { H ^ { n } } - \rho _ { i } \right| = O \left( \displaystyle \frac { \tau ^ { \mathrm { l r n } } } { T - \tau ^ { \mathrm { l r n } } } \right) , \qquad } & { i \in I ^ { * } , } \\ & { \qquad \displaystyle \frac { B _ { i } ^ { n } } { H ^ { n } } \geq \rho _ { i } - O \left( \displaystyle \frac { \tau ^ { \mathrm { l r n } } } { T - \tau ^ { \mathrm { l r n } } } \right) , \qquad } & { i \notin I ^ { * } . } \end{array}
$$

Thus, for all suficiently large $T ,$ every protective resolving ratio has $\Delta ( B ^ { n } / H ^ { n } ) < \delta _ { \mathrm { n d } } / 8$ . If, in addition, $\| \widehat { p } ^ { r _ { k } } - p \| _ { 1 } < \delta _ { \mathrm { n d } } / 2$ , Lemma E.4 shows that the restricted empirical solve uses exactly the nominal support and therefore satisfies the exact nominal Bellman-potential identity. If this prior event fails, charge the entire following protective epoch. Its length is $O ( r _ { k } ^ { 2 } )$ , and every one-period nominal dual slack is uniformly bounded in the finite model. Hence, (E.2) gives

$$
\sum _ { k } r _ { k } ^ { 2 } \mathbb { P } ( \| \widehat { p } ^ { r _ { k } } - p \| _ { 1 } \geq \delta _ { \mathrm { n d } } / 2 ) \leq O \left( \sum _ { k } r _ { k } ^ { 2 } \exp ( - \Omega ( r _ { k } ) ) \right) = O ( 1 ) .
$$

The initial inventory equals $T \rho ,$ , and the first $\tau ^ { \mathrm { l r n } } = o ( T )$ rounds consume at most $\overline { { c } } \tau ^ { \mathrm { { l r n } } }$ of each resource. Thus the resource near-depletion check cannot fire during these phases for suficiently large $T ,$ even when some protective empirical estimates are bad. The comparison continues through the resulting bad epoch and resumes at the next resolving point; it does not discard the remaining horizon. The fixed warm start costs only $O ( 1 )$ . This proves the protective-phase assertion. Moreover,

$$
\mathcal { D } ( \boldsymbol { B } ^ { \tau ^ { \mathrm { l r n } } } - \boldsymbol { H } ^ { \tau ^ { \mathrm { l r n } } } \boldsymbol { \rho } ) = O ( \tau ^ { \mathrm { l r n } } )\tag{E.30}
$$

whether or not a protective empirical solve was good, because all protective and safe actions have

bounded costs.

For the sparse phase, apply the basis-locality check at every sparse resolving point before acting: it fails when

$$
\Delta \left( \frac { B ^ { t _ { k } } } { H ^ { t _ { k } } } , \widehat { p } ^ { t _ { k } } \right) \geq \delta _ { \mathrm { n d } } .
$$

Stop analytically at its first failure or at the first activation of the resource near-depletion check. The midpoint locality and resource near-depletion checks are assigned to the late comparison. On this stopped process, use the global resource martingales from the proof of Lemma E.6. For $A > 0$ let $\mathcal { H } _ { T } ( A )$ be the event that the following simultaneously hold:

$$
\operatorname* { m a x } _ { i } \operatorname* { m a x } _ { t \leq \tau ^ { \mathrm { h a l } } } | M _ { i } ( t ) | \leq A \sqrt { T \log T } ,
$$

$$
\| \widehat { p } ^ { t _ { k } } - p \| _ { 1 } \leq A \sqrt { \frac { \log T } { t _ { k } } } \quad \mathrm { a t ~ e v e r y ~ s p a r s e ~ r e s o l v i n g ~ p o i n t ~ } t _ { k } .\tag{E.31}
$$

The maximal Azuma–Hoefding inequality for the stopped martingales and (E.2), followed by a union bound over O(log log T) re-solves, give

$$
\mathbb { P } \big ( \mathcal { H } _ { T } ( A ) ^ { \mathrm { c } } \big ) = O ( T ^ { - Q } ) , \qquad \mathrm { f o r ~ e a c h ~ f i x e d ~ } Q > 0 \mathrm { ~ a n d ~ s u f f i c i e n t l y ~ l a r g e ~ } A .\tag{E.32}
$$

Thus the exponent can be made larger than any prescribed $Q .$

On $\mathcal { H } _ { T } ( A )$ , the pathwise deviation bound (E.26) and (E.30) give

$$
\begin{array} { l } { { \displaystyle D ^ { * } = O \left( 1 + \tau ^ { \mathrm { l r n } } + A \sqrt { T \log T } + A \sqrt { \log T } \sum _ { k } \frac { \Lambda _ { k } } { \sqrt { t _ { k } } } \right) } } \\ { { \displaystyle ~ = O ( A \sqrt { T \log T } ) . } } \end{array}
$$

For suficiently large $T ,$ , the right-hand side is smaller than $\delta _ { \mathrm { n d } } T / 8$ . Since $H _ { k } \ge T / 2$ throughout the sparse phase,

$$
\Delta ( B ^ { t _ { k } } / H ^ { t _ { k } } ) \leq \frac { D ^ { * } } { H _ { k } } < \delta _ { \mathrm { n d } } / 4 .
$$

Also, $t _ { k } \ \geq \ \tau ^ { \mathrm { l r n } } \ \geq \ ( \log T ) ^ { 2 }$ , so the bound in (E.31) is $o ( 1 )$ uniformly over sparse resolving points and is eventually smaller than $\delta _ { \mathrm { n d } } / 4$ . Hence every stopped resolving pair actually satisfies $\Delta ( B ^ { t _ { k } } / H ^ { t _ { k } } , \widehat { p } ^ { t _ { k } } ) < \delta _ { \mathrm { n d } } / 2$ , contradicting a joint-basis exit. Since $D ^ { * }$ includes every stopped prefix and $\begin{array} { r } { \delta _ { \mathrm { n d } } < \frac { 1 } { 4 } \operatorname* { m i n } _ { i } \rho _ { i } . } \end{array}$ , at every such instant and for every resource,

$$
B _ { i } ^ { n } \geq H ^ { n } \rho _ { i } - D ^ { * } > \frac { T } { 2 } \operatorname* { m i n } _ { i } \rho _ { i } - \frac { \delta _ { \mathrm { n d } } T } { 8 } .
$$

For suficiently large $T ,$ , this exceeds c, excluding a first activation of the resource near-depletion check as well. Therefore, (E.29) holds on $\mathcal { H } _ { T } ( A )$ . Combining this fact with (E.32) proves the final assertion. The finitely many small values of $T$ are absorbed into the constants. □

Summable late exits. For the late comparison, set

$$
\tau ^ { \mathrm { h a l } } : = \lfloor T / 2 \rfloor , \qquad h _ { 0 } : = \lceil T / 2 \rceil .
$$

Let $\mathcal { G } _ { 0 }$ be the event that the comparison reaches the midpoint and that

$$
\frac { B ^ { \tau ^ { \mathrm { h a l } } } } { h _ { 0 } } \in \mathcal { N } _ { \delta _ { \mathrm { n d } } / 4 } .
$$

In this late comparison, $H _ { \perp }$ denotes a fixed, instance-dependent integer that may be enlarged below but never depends on T. For every nonfinal late epoch $k ,$ define

$$
t _ { k } : = T - h _ { k } , \qquad h _ { k + 1 } : = \left\lceil h _ { k } ^ { \alpha } \right\rceil , \qquad \Lambda _ { k } : = h _ { k } - h _ { k + 1 } ,
$$

and define its endpoint-movement bufer by

$$
d _ { k } : = h _ { k + 1 } ^ { - \kappa } .
$$

Fix $c > 0$ , to be chosen suficiently small, and define the epoch-k prior-error tolerance, for every epoch opened above the fixed cutof, by

$$
a _ { k } : = c { \frac { d _ { k } h _ { k + 1 } } { \Lambda _ { k } } } .
$$

The prior-error check at the start of such an epoch is the pre-action test

$$
\begin{array} { r } { \boldsymbol { e } _ { k } = \| \widehat { p } ^ { t _ { k } } - p \| _ { 1 } \leq a _ { k } ; } \end{array}
$$

it fails when the displayed inequality is violated. At a reached endpoint $t _ { k + 1 }$ , the checks are applied in the following order. First apply the epoch-movement check for epoch k. If it passes but the fixed cutof has been reached, stop without opening a new epoch. Otherwise, regard the same time as the start of epoch $k + 1$ and apply the prior-error check and then the resource near-depletion check for that new epoch. Thus an epoch-movement failure at $t _ { k + 1 }$ belongs to epoch $k ,$ whereas either pre-action check failure there belongs to epoch $k + 1$ . At a pre-action time that is not a reached epoch endpoint, the fixed cutof is tested before resource near-depletion.

On $\mathcal { G } _ { 0 }$ , let $\sigma$ be the first of the following times, using the priority just specified when several triggers share the same completed-round time:

(a) a reached endpoint $t _ { k + 1 }$ at which the epoch-movement check (E.21) fails;

(b) the first pre-action time at which the remaining horizon is at most the fixed cutof $H _ { \perp }$ ;

(c) a resolving time $t _ { k }$ at which the prior-error check $e _ { k } \leq a _ { k }$ fails, with the check applied before acting in epoch $k ;$

(d) a pre-action time $t \in \{ t _ { k } , \ldots , t _ { k + 1 } - 1 \}$ at which the resource near-depletion check (E.20) fires; this failure belongs to epoch k.

A trigger in items (a), (c), or (d) during epoch k is called an analytical late failure in epoch k; reaching the fixed cutof is not a failure. The union of items (a) and (d) is called a dynamic failure; thus this term includes both an epoch-movement failure and activation of the resource near-depletion check, but not a prior-error failure.

Lemma E.9. Under Assumption 3.2, the fixed cutof $H _ { \perp }$ can be enlarged and c can be chosen suficiently small so that, for every late epoch $k ,$

P G<sub>0</sub>, the first analytical late failure occurs in epoch $k ) \leq \exp ( - \Omega ( h _ { k } ^ { \zeta } ) )$ .

(E.33)

Consequently,

$$
\begin{array} { r } { { \mathbb { E } } [ H ^ { \sigma } { \bf 1 } \{ \mathcal { G } _ { 0 } \} ] = O ( 1 ) , \qquad { \mathbb { E } } [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } { \bf 1 } \{ \mathcal { G } _ { 0 } \} ] = O ( 1 ) . } \end{array}\tag{E.34}
$$

Before $\sigma ,$ every empirical LP remains in the nominal joint basis cell and its core projection is the identity.

Proof. Enlarge $H _ { \perp }$ and reduce c so that $a _ { k } \le \operatorname* { m i n } \{ 1 , \delta _ { \mathrm { n d } } / 4 \}$ for every nonfinal late epoch and so that the cumulative movement bound from Lemma C.10 is smaller than $\delta _ { \mathrm { n d } } / 4$ . Every late resolving point occurs after at least $\tau ^ { \mathrm { h a l } } = \lfloor T / 2 \rfloor$ requests have been completed. Therefore (E.2) gives, without conditioning on the epoch-start history,

$$
\begin{array} { r l } & { \mathbb { P } \big ( \| \widehat { p } ^ { t _ { k } } - p \| _ { 1 } > a _ { k } \big ) \leq \exp ( - \Omega ( T a _ { k } ^ { 2 } ) ) } \\ & { \qquad \leq \exp \big ( - \Omega ( h _ { k } ^ { 2 \alpha ( 1 - \kappa ) - 1 } ) \big ) = \exp ( - \Omega ( h _ { k } ^ { \zeta } ) ) . } \end{array}
$$

The second inequality uses $\Lambda _ { k } \leq h _ { k } , h _ { k + 1 } \geq h _ { k } ^ { \alpha }$ , and $T \geq h _ { k }$

Now fix an epoch-start history on which all preceding analytical checks passed and $e _ { k } : =$ $\| \widehat { p } ^ { t _ { k } } - p \| _ { 1 } \leq a _ { k }$ . The cumulative movement bound keeps the capacity ratio in the joint basis neighborhood, while $a _ { k } \le \delta _ { \mathrm { n d } } / 4$ keeps the prior there. Lemma E.4 therefore applies, the core projection is the identity, and (E.15) gives, for every binding resource,

$$
\Lambda _ { k } \big | \overline { { c } } _ { i } ( \beta ^ { k } , \widehat { p } ^ { t _ { k } } ) - \beta _ { i } ^ { k } \big | = O ( \Lambda _ { k } e _ { k } ) \leq \frac 1 4 d _ { k } h _ { k + 1 }
$$

after reducing $c .$ Hence a binding endpoint movement larger than $d _ { k }$ requires a centered resource fluctuation of order $d _ { k } h _ { k + 1 }$ . By (E.17), uniformly over the good history,

$$
> d _ { k } \mid \mathcal { F } ^ { t _ { k } } \Big ) \leq \exp \left( - \Omega \left( \frac { d _ { k } ^ { 2 } h _ { k + 1 } ^ { 2 } } { \Lambda _ { k } } \right) \right) \leq \exp ( - \Omega ( h _ { k } ^ { \zeta } ) ) .
$$

For inactive resources, (E.16) supplies a fixed favorable drift, so the same one-sided estimate controls a downward endpoint movement larger than $d _ { k }$

For the resource near-depletion check, abbreviate the frozen true resource rate by $\mu _ { i } : = \overline { { c } } _ { i } ( \beta ^ { k } , \widehat { p } ^ { t _ { k } } )$ The preceding binding-bias bound and the favorable inactive drift give, for every resource and every

prefix $0 \le \ell \le \Lambda _ { k }$

$$
\begin{array} { l } { { \displaystyle h _ { k } \beta _ { i } ^ { k } - \ell \mu _ { i } \geq h _ { k + 1 } \beta _ { i } ^ { k } - \Lambda _ { k } ( \mu _ { i } - \beta _ { i } ^ { k } ) _ { + } } } \\ { { \displaystyle \qquad \geq h _ { k + 1 } ( \rho _ { i } - \delta _ { \mathrm { n d } } - d _ { k } / 4 ) \geq \frac { 1 } { 2 } h _ { k + 1 } \rho _ { i } } , } \end{array}
$$

after enlarging the fixed cutof so that $d _ { k } \le \operatorname* { m i n } _ { i } \rho _ { i }$ , using $\begin{array} { r } { \delta _ { \mathrm { n d } } < \frac { 1 } { 4 } \operatorname* { m i n } _ { i } \rho _ { i } } \end{array}$ . The actual inventory at a prefix before or at the first activation of the resource near-depletion check equals this meanreference inventory minus the centered cumulative cost of the same prefix of the hypothetical frozen continuation. Thus an inventory level of at most c requires a centered prefix deviation of order $h _ { k + 1 }$ By (E.17), resource near-depletion therefore has conditional probability at most exp $\big ( - \Omega ( h _ { k } ^ { 2 \alpha - 1 } ) \big )$ , which is smaller because $\zeta < 2 \alpha - 1$

Let $\mathcal G _ { k } ^ { \mathrm { p r e } }$ denote the event that $\mathcal { G } _ { 0 }$ holds and no analytical late failure occurred before epoch $k ;$ for $k = 0$ , the latter condition is vacuous. Splitting the empirical event from the conditional dynamic event gives

$$
\begin{array} { r l } & { \mathbb { P } ( \mathcal { G } _ { k } ^ { \mathrm { p r e } } , \mathrm { ~ f r s t ~ a n a l y t i c a l ~ l a t e ~ f a i l u r e ~ o c c u r s ~ i n ~ e p o c h ~ } k ) } \\ & { \qquad \leq \mathbb { P } ( e _ { k } > a _ { k } ) + \mathbb { E } \Big [ \mathbf { 1 } \{ \mathcal { G } _ { k } ^ { \mathrm { p r e } } , e _ { k } \leq a _ { k } \} \mathbb { P } ( \mathrm { d y n a m i c ~ f a i l u r e ~ i n ~ e p o c h ~ } k \mid \mathcal { F } ^ { t _ { k } } ) \Big ] } \\ & { \qquad \leq \exp ( - \Omega ( h _ { k } ^ { \zeta } ) ) , } \end{array}
$$

which proves (E.33). Note that empirical concentration is used only in the first, unconditional term; after conditioning on $\mathcal { F } ^ { t _ { k } }$ , the empirical error is treated as a fixed quantity. If no failure occurs in epoch $k ,$ , the endpoint movement is at most $d _ { k }$ . Lemma C.10 and the choice of $H _ { \perp }$ then keep every later capacity ratio inside the joint basis cell. At the beginning of the next epoch, the new empirical error is checked afresh before any action is taken.

On $\mathcal G _ { k } ^ { \mathrm { p r e } }$ , the binding inventories at the beginning of epoch k are $O ( h _ { k } )$ , and costs are nonnegative. Thus, whether the first failure is an estimation stop, an epoch-movement failure, or a resource near-depletion stop,

$$
H ^ { \sigma } + F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } = O ( 1 + h _ { k } )\tag{E.35}
$$

on that first-failure event; inactive inventories do not enter the dual potential because $\lambda _ { i } ^ { * } = 0$ there. Finally,

$$
\sum _ { k } ( 1 + h _ { k } ) \exp ( - \Omega ( h _ { k } ^ { \zeta } ) ) = O ( 1 ) .\tag{E.36}
$$

This bound is uniform in $T \colon$ reading the power schedule backward from the fixed cutof, as in the proof of Lemma C.10, makes $h _ { k }$ grow faster than geometrically, whereas the exponential factor decays super-polynomially. The final cutof has bounded horizon. Combining (E.33), (E.35) and (E.36) proves (E.34) and the lemma. □

## E.1.5 Finishing the proof

Proof. Fix the true nominal dual solution and, in each case, compare UIR-UP with the proof-only policy $\mathrm { U I R - U P ^ { g } }$ that takes null actions after its permanent analytical stop. We use (E.22) only

before that stop.

Unique, possibly degenerate case. Fix a suficiently small one-sided entrance neighborhood, below $\begin{array} { r l } {  { { \frac { 1 } { 2 } } \operatorname* { m i n } _ { i } \rho _ { i } } } \end{array}$ , and stop the comparison immediately after any successful entrance outside that neighborhood. On this irregular entry event,

$$
H ^ { \mathrm { e n t } } = O ( \mathcal { D } ( X ^ { \mathrm { e n t } } ) ) , \qquad F ( H ^ { \mathrm { e n t } } , B ^ { \mathrm { e n t } } , S ^ { \mathrm { e n t } } ) ^ { + } = O ( 1 + H ^ { \mathrm { e n t } } + \mathcal { D } ( X ^ { \mathrm { e n t } } ) ) .
$$

Lemma C.2 therefore charges entrance failure, irregular entry, and the entrance potential discrepancy by $O ( 1 )$ in expectation. On the remaining entrance histories, condition on the entrance history and use the post-entry notation. The regular-entry radius gives every initial inventory coordinate at least $T \operatorname* { m i n } _ { i } \rho _ { i } / 2$ . Hence, for every suficiently large post-entry horizon, bounded one-period costs preclude resource near-depletion through the protective phase and $n _ { 0 } < \tau ^ { \mathrm { l r n } } < \tau ^ { \mathrm { h a l } }$ . The algorithm’s safe fallback handles the remaining bounded post-entry horizons, whose entire contribution is $O ( 1 )$ Finally, the warm start and protective phase have length $O ( ( \log T ) ^ { 2 } ) = o ( \sqrt { T } )$ , so their bounded one-period dual losses are $o ( \sqrt { T } )$

Concatenate the stopped comparisons from Lemmas E.6 and E.7. Those lemmas bound the accumulated local dual loss and the positive terminal potential by ${ O } ( \sqrt { T } + { \mathcal { D } } ( X ^ { \mathrm { e n t } } ) )$ ). Telescoping (E.22) from entrance to the permanent stop, then averaging over entrance histories and using $\sqrt { H ^ { \mathrm { e n t } } } \leq \sqrt { T }$ , gives

$$
T G ^ { p } ( \rho ) - V _ { T } ^ { \mathrm { U I R - U P ^ { g } } } = O ( \sqrt { T } ) .
$$

Pathwise dominance and the true-prior boundary comparison (7) give (25) for the guard-free UIR-UP policy.

Unique, nondegenerate case. By Proposition $3 . 6 , \mathcal { R } ^ { * } = \mathcal { S }$ . Thus the analysis starts at the first round with $B ^ { 0 } = T \rho$ . Let $\sigma$ be the permanent stop obtained by concatenating the midpoint and late comparisons, including the fixed cutof. Choose $Q > 2$ in Lemma E.8. A first-half failure then has probability $O ( T ^ { - Q } )$ . Since binding inventories never exceed $O ( T )$ and inactive resource prices vanish, the positive terminal potential on this event is at most $O ( 1 + T )$ ; hence its expected contribution is $O ( 1 )$ . On midpoint-success paths, Lemma E.9 gives an $O ( 1 )$ expected positive terminal potential, including late failures and the cutof.

On every good protective, sparse, and late epoch, Lemma E.4 gives the nominal action support, so the loss in (E.22) is zero. The fixed warm start and bad protective epochs have total expected loss $O ( 1 )$ by (E.28). Therefore (C.8) and finite-horizon telescoping from the first round give

$$
\begin{array} { l } { \displaystyle T G ^ { p } ( \rho ) - V _ { T } ^ { \mathrm { U I R - U P ^ { g } } } } \\ { \displaystyle \qquad \leq \mathbb { E } \biggl [ F ( H ^ { \sigma } , B ^ { \sigma } , S ^ { \sigma } ) ^ { + } + \sum _ { t = 0 } ^ { \sigma - 1 } \ell ^ { t } \biggr ] = O ( 1 ) . } \end{array}
$$

Applying pathwise dominance and (7) proves (26) for the guard-free UIR-UP policy.

## E.2 Proof of Proposition 6.2

Proof. There are three states $\{ 0 , A , B \}$ , two request types hi, lo, and one resource with $\rho = 1$ ; the initial state is 0. Consider the two priors

$$
p ^ { + } ( \mathrm { h i } ) = \frac { 3 } { 4 } , \qquad p ^ { - } ( \mathrm { h i } ) = \frac { 1 } { 4 } ,
$$

with complementary probabilities on lo.

At state 0, there are two actions, $a _ { A }$ and $a _ { B } .$ , each with reward zero and cost one; their deterministic transitions are $0  A$ and $0  B$ , respectively. The null action also has zero reward and cost and moves to A or B with probability $1 / 2$ each. Thus, no action can remain at state 0 to collect additional request samples before committing to one of the two classes. Once A or B is reached, it is absorbing under all available actions. Every unspecified non-null state-action pair has zero reward, cost one, and the same next-state law as the null action in that state, so it is dominated by the null action and does not afect the stationary calculation.

In state A, there is a productive action $a _ { A }$ of cost one with reward

$$
r ( A , \mathrm { h i } , a _ { A } ) = 1 , \qquad r ( A , \mathrm { l o } , a _ { A } ) = \varepsilon .
$$

In state $B ,$ there is a productive action $a _ { B }$ of cost one with

$$
r ( B , \mathrm { l o } , a _ { B } ) = 1 , \qquad r ( B , \mathrm { h i } , a _ { B } ) = \varepsilon .
$$

The null action has zero reward and cost and leaves A and B absorbing. The productive action has strictly larger reward for both request types, and at the nominal ratio $\rho = 1$ , using it on every request consumes exactly one unit of resource per period.

No transition enters state $0 ,$ so stationarity forces $q _ { 0 } = 0$ and $q _ { A } + q _ { B } = 1$ . For any feasible stationary flow with such a state distribution, replacing every null or unspecified action in the two absorbing classes by the corresponding productive action is feasible: after all replacements, total resource use is exactly one. Since all productive rewards are strictly positive, every optimum must therefore use the productive action for both requests in every state having positive stationary mass.

Under $p ^ { + }$ , the resulting per-period rewards in classes A and B are, respectively,

$$
g _ { A } ^ { + } = \frac { 3 } { 4 } + \frac { 1 } { 4 } \varepsilon , \qquad g _ { B } ^ { + } = \frac { 1 } { 4 } + \frac { 3 } { 4 } \varepsilon ,
$$

and $g _ { A } ^ { + } - g _ { B } ^ { + } = ( 1 - \varepsilon ) / 2 > 0$ . The objective under $p ^ { + }$ is therefore

$$
q _ { A } g _ { A } ^ { + } + q _ { B } g _ { B } ^ { + } ,
$$

so its unique maximizer has $q _ { A } = 1$ and $q _ { B } = 0$ . Thus the unique stationary fluid optimum under $p ^ { + }$ takes $a _ { A }$ for both request types. Symmetrically, the unique optimum under $p ^ { - }$ assigns all stationary state probability to B and takes $a _ { B }$ for both request types. The respective optimal target state classes are therefore $\{ A \}$ and $\{ B \}$ . The optimal induced kernel on each singleton is irreducible, so

Assumption 3.4 holds separately for both priors. The correct entrance action reaches that singleton in one round with unit resource cost, so Assumption 3.7 also holds separately for both priors.

The nominal optimum is also degenerate. Indeed, the unique optimum uses the resource at rate exactly one, so the resource is binding. If, say, under $p ^ { + }$ , the capacity ratio is decreased to $\beta = 1 - d$ for arbitrarily small $d > 0$ , the nominal positive z-support cannot remain feasible: request balance still requires total stationary state probability one, while taking the productive action on every request would consume one unit per period. An optimizer must therefore put positive mass on the null action. Thus the positive z-support changes under arbitrarily small downward right-hand-side perturbations. By Proposition 3.3, a unique nondegenerate nominal basic optimum would have locally constant positive support. The unique nominal optimum is therefore degenerate.

Fix a horizon $T \geq 1$ and any policy $\pi _ { T }$ that does not know which of the two priors governs requests. The policy may depend on $T .$ All value functions below are evaluated from initial inventory $T \rho$ and initial state 0. The policy sees $J ^ { 1 }$ before choosing its first action, but after that action the recurrent class is fixed forever. Let

$$
x _ { \mathrm { h i } } : = \mathbb { P } ( { \mathrm { e v e n t u a l ~ c l a s s ~ i s ~ } } A \mid J ^ { 1 } = \mathrm { h i } ) , \qquad x _ { \mathrm { l o } } : = \mathbb { P } ( { \mathrm { e v e n t u a l ~ c l a s s ~ i s ~ } } A \mid J ^ { 1 } = \mathrm { l o } ) ,
$$

where the probabilities include the algorithm’s internal randomization and also cover use of the null action. These two conditional class-selection probabilities are the same under both priors, because the first observed request and all other supplied information coincide under that conditioning. They may depend on T. The wrong-class probabilities are

$$
\begin{array} { l } { { \displaystyle e _ { + } = \frac { 3 } { 4 } ( 1 - x _ { \mathrm { h i } } ) + \frac { 1 } { 4 } ( 1 - x _ { \mathrm { l o } } ) , } } \\ { { \displaystyle e _ { - } = \frac { 1 } { 4 } x _ { \mathrm { h i } } + \frac { 3 } { 4 } x _ { \mathrm { l o } } . } } \end{array}
$$

Therefore

$$
e _ { + } + e _ { - } = 1 - \frac { 1 } { 2 } x _ { \mathrm { h i } } + \frac { 1 } { 2 } x _ { \mathrm { l o } } \geq \frac { 1 } { 2 } .
$$

At least one prior has wrong-class probability at least $1 / 4$ . Put

$$
R _ { \mathrm { g o o d } } : = \frac { 3 } { 4 } + \frac { 1 } { 4 } \varepsilon , \qquad R _ { \mathrm { w r o n g } } : = \frac { 1 } { 4 } + \frac { 3 } { 4 } \varepsilon .
$$

Under either prior, the good and wrong recurrent classes difer in per-round expected productive reward by exactly

$$
R _ { \mathrm { g o o d } } - R _ { \mathrm { w r o n g } } = \frac { 1 - \varepsilon } { 2 } .
$$

The optimal known-prior policy chooses the correct entrance action in round one and then takes the productive action in every remaining round. This is feasible with $\rho = 1$ : the entrance action consumes one unit, leaving $T - 1$ units for the remaining $T - 1$ productive rounds, with the terminal round allowed to bring the inventory to zero. No policy in a fixed class can exceed its per-period productive reward, and future requests are independent of the first-round commitment.

Consequently, for either sign $\sigma \in \{ + , - \}$ ,

$$
V _ { T } ^ { \ast , p ^ { \sigma } } = ( T - 1 ) R _ { \mathrm { g o o d } } , \qquad V _ { T } ^ { \pi _ { T } , p ^ { \sigma } } \leq ( T - 1 ) \left( ( 1 - e _ { \sigma } ) R _ { \mathrm { g o o d } } + e _ { \sigma } R _ { \mathrm { w r o n g } } \right) .
$$

Choosing a sign with $e _ { \sigma } \ge 1 / 4$ gives

$$
\operatorname* { m a x } _ { p ^ { \prime } \in \{ p ^ { + } , p ^ { - } \} } \left\{ V _ { T } ^ { * , p ^ { \prime } } - V _ { T } ^ { \pi _ { T } , p ^ { \prime } } \right\} \geq \frac { 1 } { 4 } \cdot \frac { 1 - \varepsilon } { 2 } ( T - 1 ) = \frac { 1 - \varepsilon } { 8 } ( T - 1 ) .
$$

Here the superscript $p ^ { \prime }$ records the true law used to evaluate a policy. For each prior separately, the transient fluid upper bound gives

$$
\Phi _ { T } ^ { p ^ { \prime } } ( T \rho , 0 ) - V _ { T } ^ { \pi _ { T } , p ^ { \prime } } \geq V _ { T } ^ { * , p ^ { \prime } } - V _ { T } ^ { \pi _ { T } , p ^ { \prime } } .
$$

Hence the maximum transient-fluid gap obeys the same lower bound. This proves the proposition for every $T$ and $\pi _ { T }$ . The maximizing member of the fixed pair may depend on the horizon and policy; the argument does not require a single prior to be worst for every horizon in a horizon-dependent policy family. □