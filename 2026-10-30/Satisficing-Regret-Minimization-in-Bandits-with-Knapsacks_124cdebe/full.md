# Satisficing Regret Minimization in Bandits with Knapsacks

Qing Feng School of Operations Research and Information Engineering, Cornell University qf48@cornell.edu

Tianyi Ma School of Operations Research and Information Engineering, Cornell University tm693@cornell.edu

Ruihao Zhu SC Johnson College of Business, Cornell University ruihao.zhu@cornell.edu

We study satisficing regret minimization in bandits with knapsacks (BwK), where a learner sequentially selects randomized policies, i.e., distributions over arms, to obtain mean rewards above a satisficing threshold as frequently as possible while respecting finite resource budgets. The performance is measured by satisficing regret, the cumulative deficits of the selected policies’ mean rewards below this threshold. Resource constraints introduce two dificulties: an algorithm can incur linear satisficing regret even with small standard regret as deficits to the threshold cannot be compensated, and random consumption can cause early resource depletion even when policies are feasible in expectation. To address these challenges, we propose SELECTwK, a general algorithmic template that uses a BwK learning oracle algorithm with sub-linear expected (standard) regret upper bound to search for and test candidate satisficing policies (i.e., policies whose mean rewards exceed satisficing threshold and whose expected resource consumption is feasible over the time horizon). Ou regret analysis, supported by an LP-duality argument, establishes that SELECTwK attains constant expected satisficing regret when a satisficing policy exists. We further introduce a stronger regret condition that bounds the expected sum of positive reward deficits to the optimal one at each individual time step, under which SELECTwK achieves a sharper constant bound together with a sub-linear bound in the time horizon. As a complement, SELECTwK still preserves the oracle’s sub-linear expected standard regret bound if the satisficing threshold is unattainable. Finally, we establish the oracle conditions and corresponding satisficing regret bound for existing finite-armed BwK and dynamic pricing algorithms, and conduct numerical experiments on dynamic pricing with limited inventory to demonstrate the empirical performance of SELECTwK

Key words : online learning, bandits with knapsacks, satisficing, regret

## 1. Introduction

Bandits with knapsacks (BwK) provide a framework for sequential decision-making under uncertainty and resource constraints. In BwK, a learner repeatedly selects an arm, receives a noisy reward, and consumes a random amount of each resource. The mean rewards and resource consumptions of the arms are initially unknown and must be learned from the observed outcomes. The learner aims to maximize the expected cumulative reward while keeping the total consumption of each resource within its budget. To achieve this, the learner must balance exploring diferent arms to learn their outcomes and exploiting the information collected to raise rewards, while accounting for the resources that remain available for future decisions.

While maximizing cumulative reward is a natural objective, in many applications a decision-maker may be satisfied with performance that meets a prescribed target. This preference reflects the notion of satisficing (Lufkin 2021), coined by Simon (1956), where an acceptable solution is suficient without requiring the best possible one. In this paper, we study satisficing bandits with knapsacks, where the learner seeks to select policies, i.e., distributions over arms, whose mean rewards exceed a prescribed satisficing threshold as frequently as possible while respecting finite resource budgets. This objective finds its place in a variety of decision-making settings where satisfactory performance must be maintained with limited resources, as illustrated by the following examples:

• Cloud Service: Consider an operator choosing configurations for a cloud service. In each period, the operator deploys a configuration, such as a resource allocation or batching rule, and observes its service quality and resource consumption. The operator seeks to meet a prescribed service-level objective (SLO), such as meeting 99% availability measured over a calendar week or having at least 95% of requests receive a response within 100 milliseconds (Google 2026). Configurations meeting this target are acceptable, whereas pursuing higher service quality may require additional resources without providing a meaningful improvement to users. Moreover, the operator may have a fixed total computing budget over the planning horizon, and a configuration that pursues an optimal service quality may consume resources too quickly and influence decisions in later periods.

• Inventory Management: Consider a retailer choosing stocking quantities for successive selling periods. Product shortages can lead to lost sales and discourage customers from returning (Fitzsimons 2000, Jing and Lewis 2011). However, improving availability generally imposes pressure on purchasing budget as large orders in earlier periods leave fewer funds for subsequent purchases (Oracle 2026). Searching for the optimal stocking quantity that maximizes service level is also very costly. Retailers may thus work toward an acceptable service-level target, such as a 90%–95% availability, rather than seeking the highest possible one (Levy et al. 2023), while respecting the purchasing budget over the planning horizon.

To measure performance against the targets illustrated by these examples, we use satisficing regret, defined as the expected cumulative positive deficit of the selected policies’ mean rewards relative to the satisficing threshold. In the setting without resource constraints, Michel et al. (2023) establish constant expected satisficing regret for finite-armed bandits when an arm with mean reward above the threshold exists. Feng et al. (2025) further show that a blackbox learning oracle algorithm with sub-linear expected standard regret can be used to attain constant expected satisficing regret across diferent bandit problem classes when the optimal mean reward exceeds the threshold. When the threshold is unattainable, their algorithm preserves the oracle’s expected standard regret bound.

However, extending these results to bandits with knapsacks introduces two dificulties. First, a resource-feasible algorithm can attain a near-optimal cumulative reward while repeatedly selecting policies (distributions of arms) whose mean rewards fall below the satisficing threshold. The reason is that high rewards in some periods can compensate for low rewards in other periods when computing standard regret, whereas satisficing regret penalizes each shortfall separately. Moreover, even identifying a satisficing policy, i.e., a policy with mean reward exceeding satisficing threshold and expected resource consumption feasible along time horizon, is not suficient to ensure constant satisficing regret. Randomness in resource consumption may still cause resources to be depleted before the end of the horizon. Therefore, besides identifying policies with satisficing mean rewards, the learner must account for their resource consumption and protect against early depletion.

In this paper, we propose SELECTwK, a general algorithmic template that uses a bandits-withknapsacks learning oracle to identify and test candidate satisficing policies while controlling resource consumption. We establish constant expected satisficing regret in the realizable case, i.e., satisficing policies exist, under the condition that the learning oracle attains sub-linear expected standard regret, and further investigate how a stronger guarantee on the learning oracle improves the constant dependence.

## 1.1. Main Contribution

In this work, we propose SELECTwK, a novel algorithmic template for Satisficing REgret Minimization via SampLing and LowEr Confidence Bound Testing with Knapsack. More specifically:

• In Section 2, we formulate a satisficing regret minimization problem for bandits with resource constraints. The learner needs to select randomized policies whose mean rewards exceed the satisficing threshold, i.e., a satisficing policy, as frequently as possible while keeping the total consumption of each resource within a resource budget. We define satisficing regret as the expected cumulative deficits of the selected policies’ mean rewards to the satisficing threshold, and introduce an LP relaxation to upper bound this optimal expected cumulative reward. In the realizable case, the learner aims to maintain low cumulative satisficing regret. Otherwise, in the non-realizable case, the learner aims to maintain sub-linear standard regret. We further discuss why resource constraints fundamentally change the problem. Sub-linear standard regret does not imply low satisficing regret, since rewards above the LP optimal reward can compensate for low rewards at other time steps. Moreover, always selecting satisficing policies does not ensure constant satisficing regret, since the randomness in resource consumption may cause early resource depletion.

• We describe the design of SELECTwK in Section 3. For a given class of bandit-with-knapsack problem and a corresponding learning algorithm with a sub-linear standard regret bound, SELECTwK first adjusts the resource budget with a margin to hedge against consumption randomness, then it runs in rounds with the help of the learning algorithm. In each round, SELECTwK first runs the learning algorithm for a number of time steps and allocate the adjusted resources to the algorithm by time. Then, it computes the average of the policy trajectory and proposes a candidate policy by mixing null arm (which deterministicly incur zero reward and zero resource consumption) into the average policy. After it conducts forced sampling by pulling this candidate policy for a pre-specified number of time steps, it continues to pull this policy and keeps monitoring the lower confidence bound (LCB) of its mean reward and the upper confidence bound (UCB) of its mean resource consumption. SELECTwK terminates the current round and starts the next as long as the LCB falls below the satisficing level or any UCB exceeds the per-time-step resource budget.

• In Section 4, we analyze the regret of SELECTwK when the learning oracle has a sub-linear expected standard regret bound. We establish a lemma that uses LP duality to control the expected positive part of the oracle’s cumulative standard regret. With this lemma, we show that SELECTwK attains constant expected satisficing regret in the realizable case. In the non-realizable case, SELECTwK attains a sub-linear expected standard regret bound that matches the learning oracle’s bound up to logarithmic factors.

• In Section 5, we introduce a strong regret condition that controls the expected sum of positive reward deficits at individual time steps relative to the LP optimal reward. Under this stronger condition, we address two limitations of the satisficing regret bound established in Section 4. Specifically, SELECTwK attains a sharper constant expected satisficing regret bound in terms of the exceeding gap (i.e., the diference between the optimal reward and the threshold value), together with a bound that is sub-linear in the time horizon and remains valid when the exceeding gap is small.

• In Section 6, we instantiate SELECTwK for three classes of bandit optimization problems with resource constraints: finite-armed bandits with knapsacks, dynamic pricing with continuous prices under inverse demand, and dynamic pricing with finite prices and unknown demand. We establish expected strong regret bounds and the resulting expected constant satisficing regret bound for the first two classes. For the third class, we provide the corresponding bound under the expected standard regret condition.

• In Section 7, we conduct numerical experiments to demonstrate the performance of SELECTwK on a synthetic instance of dynamic pricing on single resource with selling rate linear in price. We use TS-fixed, an algorithm based on Thompson sampling designed for finite-armed bandits with knapsack proposed by Ferreira et al. (2018), as the learning oracle and the benchmark. We consider two cases where the optimal reward is reached by the price that exhausts time budget, i.e., the resource constraint does not bind at the optimum, and is reached by the price that exhausts resource budget, i.e., the resource constraint binds at the optimum. Our results reveal that in both cases, SELECTwK does achieve constant expected satisficing regret when it is realizable and its standard regret is comparable to that of the benchmark when it is non-realizable.

## 1.2. Related Works

The most relevant works are Michel et al. (2023) and Feng et al. (2025), which establish constant expected satisficing regret when the optimal mean reward exceeds the threshold. Michel et al. (2023) focus on finite-armed bandits, whereas Feng et al. (2025) propose an algorithmic template that uses a learning oracle with sub-linear standard regret to achieve constant expected satisficing regret across diferent bandit problem classes. Related finite-armed results include Bubeck et al. (2013) and Garivier et al. (2019), who establish constant regret under prior knowledge about the optimal mean reward, and Tamatsukuri and Takahashi (2019), who establish constant regret when there is exactly one satisficing arm. Bounded satisficing regret has also been established in multi-objective bandits by H¨uy¨uk and Tekin (2021) and finite-state Markov decision processes by Hajiabolhassan and Ortner (2026). More recently, Rouyer et al. (2024) refine the dependence of satisficing regret on the exceeding gap, and Zhang et al. (2026) show that constant satisficing regret may no longer be attainable in nonstationary bandits.

The notion of satisficing has also been studied under diferent regret definitions. An early formulation of satisficing regret, given by Russo and Van Roy (2022), considers a time-discounted setting. Reverdy and Leonard (2014), Abernethy et al. (2016) maximize the probability that realized rewards exceed a threshold, and a more general Bayesian setting is further considered by Reverdy et al. (2017). Instead, our satisficing regret is defined using mean rewards. The notion of satisficing is also studied in the pure exploration setting through thresholding bandits (Locatelli et al. 2016, Mukherjee et al. 2017) and good arm identification (Kano et al. 2019, Li et al. 2025).

Resource constraints complicate the problem as actions will afect the resources available for future decisions, i.e., repeatedly selecting a policy with high mean reward may deplete resources before the end of the time horizon. In the dynamic and stochastic setting, Kleywegt and Papastavrou (1998) study whether to accept randomly arriving items when their reward and size distributions are known. Bandit models introduce the additional need to learn rewards and consumption costs: Tran-Thanh et al. (2012) consider unknown rewards with known pulling costs, whereas Ding et al. (2013) allow the costs to be random and unknown. Combes et al. (2015) further derive regret lower bounds and asymptotically optimal algorithms for bandits with budget.

Badanidiyuru et al. (2018) introduce the general bandit with knapsack (BwK) framework, in which each arm pull generates a random reward and consumes multiple resources. A randomized policy over arms is used to balance reward and resource consumption, and an LP relaxation is then introduced. Algorithm with near-optimal worst-case regret is provided by Badanidiyuru et al. (2018), and Flajolet and Jaillet (2015) and Li et al. (2021) further establish instance-dependent logarithmic regret bounds for BwK. Beyond cumulative regret, Sankararaman and Slivkins (2021) bound the number of rounds in which mean reward of the selected policy falls below the optimal one by a given amount. In our work, we study cumulative satisficing regret relative to a given threshold and establish a constant expected bound in realizable cases.

Our work is also closely related to the literature on dynamic pricing with unknown demand and limited inventory. In the single-product setting, Besbes and Zeevi (2009) and Babaiof et al. (2015) study demand learning with limited inventory, while Wang et al. (2014) close the performance gap between parametric and nonparametric learning. With multiple products, pricing decisions must also account for shared resource capacities, leading to the blind network revenue management problem studied by Besbes and Zeevi (2012). In the multi-product setting, Chen et al. (2019) develop a nonparametric self-adjusting control, while Ferreira et al. (2018) combine Thompson sampling with linear programming for network revenue management. Primal-dual methods provide another approach to guiding resource allocation. Chen and Gallego (2022) develop a primal-dual learning algorithm for personalized pricing with an inventory constraint, and Miao and Wang (2024) introduce demand balancing in primal-dual optimization for blind network revenue management. While these works use dual variables to guide resource allocation, we use LP duality to bound the expected negative part of cumulative standard regret.

## 2. Problem Formulation

## 2.1. Notations

We define [n] to be the set $\{ 1 , 2 , \ldots , n \}$ for any positive integer n. For $k \in [ 1 , \infty ]$ , we use $\| x \| _ { k }$ to denote the $\ell _ { k } { \mathrm { - n o r m } }$ of a vector $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . We use $\log _ { a } ( \cdot )$ to denote the logarithm with base a. When a is left unspecified, this is the natural logarithm. We adopt the asymptotic notations $O ( \cdot ) , \Omega ( \cdot )$ , and $\Theta ( \cdot )$ as defined in (Cormen et al. 2009). When logarithmic factors are omitted, we write them as $\tilde { O } ( \cdot ) , \tilde { \Omega } ( \cdot )$ , and $\tilde { \Theta } ( \cdot )$ . We also use polylog(y) to denote a polynomial in $\log ( y )$ . With some abuse, these notations are used when we try to avoid the clutter of writing out constants explicitly. The indicator variable is denoted as 1[·].

## 2.2. Model

Bandits with Resource Constraints. Consider a class of bandit optimization problems with resource constraints $( \mathcal { X } , \mathcal { R } , \mathcal { C } )$ . Here, X is the arm set, resources are indexed by [d], $\mathcal { R } \subseteq \{ r : \mathcal { X } \to \mathbb { R } \}$ is the class of admissible reward functions, and $\mathcal { C } \subseteq \{ c : \mathcal { X } \overset { } { \to } \mathbb { R } ^ { d } \}$ is the class of admissible expected resource consumption functions. Pulling arm $X \in { \mathcal { X } }$ in a time step brings a random reward with mean $r ( X )$ and a random consumption with mean $c _ { j } ( X )$ for each resource $j \in [ d ]$ . The learner knows $r \in \mathcal { R }$ and $c \in { \mathcal { C } } .$ , but the exact reward function r and resource consumption function c is initially unknown to the learner.

At the beginning of the time horizon with length $T ,$ the learner has $B _ { j }$ units of resource $j$ for any $j \in [ d ]$ . In each time step $t \in [ T ]$ , the learner decides a policy, i.e., a distribution of arms $\pi _ { t } \in \Delta ( \mathcal { X } )$ (Here we use $\Delta ( \mathcal { X } )$ to denote the set of all distributions over the arm set $\mathcal { X } . )$ Then an arm $X _ { t }$ is sampled from the distribution $\pi _ { t } .$ , and the learner pulls $X _ { t }$ , and observes and receives an i.i.d. noisy reward $Y _ { t } \in [ 0 , 1 ]$ with mean $r ( X _ { t } )$ . At the same time, for each $j \in [ d ]$ , an i.i.d. resource consumption $C _ { t } ^ { j } ( X _ { t } ) \in [ 0 , 1 ]$ with mean $c _ { j } ( X _ { t } )$ is deducted from the remaining resource $j .$ If the total consumption of any resource exceeds the initial supply, the learner has to set $\pi _ { t } = { \bf 1 } _ { 0 } , i . e .$ ， deterministically pull the null arm that incurs zero reward and zero resource consumption.

Satisficing and Standard Regret. For simplicity, we use $r ( \pi ) = \mathbb { E } _ { X \sim \pi } [ r ( X ) ]$ to denote the mean reward of a policy π, and $c _ { j } ( \pi ) = \mathbb { E } _ { X \sim \pi } [ c _ { j } ( X ) ]$ as the mean consumption for resource $j$ of a policy π. Similar to the setting without resource constraints, in the setting with resource constraints, the learner is given a satisficing threshold value $S ,$ and the goal of the learner is to select policies with mean reward above the threshold $S$ as frequently as possible using limited resources. To achieve this goal, we measure the performance of the algorithm using satisficing regret, defined as the expected cumulative deficit of the mean reward of the policy against the threshold value S. The formal definition of satisficing regret is given by

$$
\mathsf { R e g r e t } _ { S } = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \operatorname* { m a x } \{ S - r ( \pi _ { t } ) , 0 \} \right] .
$$

Remark 1 (Definition of Satisficing Regret with Resource Constraints). In the settings of BwK, we define the satisficing regret using the mean reward of policies instead of that of individual arms, which is diferent from the satisficing regret defined for the setting without resource constraints. The reason for this is as follows: The general goal of satisficing regret minimization with resource constraints is to identify satisficing policies whose mean reward is above the satisficing threshold value S and its mean resource consumptions are within the initial resource budget divided by T. In many cases, satisficing policies cannot be achieved by deterministically pulling a single arm, thus it is often necessary for satisficing policies to randomize over multiple arms. Our results can be transformed to the satisficing regret defined based on mean reward of individual arms under certain condition, e.g., when the arm set X and reward functions r are convex, as assumed in Babaiof et al. (2015).

Besides satisficing regret, we also consider the standard regret for bandits with resource constraints. Since the optimal dynamic policy is dificult to derive even with full knowledge of reward and resource consumption, we define the per-time-step LP relaxation on the value of optimal dynamic policy, given by

$$
\begin{array} { r l } {  { Z ^ { * } ( T , B ) = \operatorname* { m a x } _ { \pi \in \Delta ( \mathcal { X } ) } \ r ( \pi ) } } \\ & { \mathrm { s . t . } \ c _ { j } ( \pi ) \le B _ { j } / T , \ \forall j \in [ d ] . } \end{array}\tag{1}
$$

Lemma 3.1 in Badanidiyuru et al. (2018) shows that $T \cdot Z ^ { * } ( T , B )$ is an upper bound on the cumulative expected reward of the optimal dynamic policy. Following notations in the literature of multi-armed bandits and BwK, we define the standard (cumulative) regret as the diference between the expected reward of an algorithm and the upper bound on the optimal expected reward, given by

$$
\mathtt { R e g r e t } = T \cdot Z ^ { * } ( T , B ) - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } r ( \pi _ { t } ) \right] .\tag{2}
$$

If $S \le Z ^ { * } ( T , B )$ , then there exists a policy π such that $r ( \pi ) \geq S$ and $c _ { j } ( \pi ) \leq B _ { j } / T$ for all $i \in [ d ]$ thus we consider this case realiz $~ _ { u b l e , }$ and in this case our goal is to attain a constant satisficing regret. Otherwise if $S > Z ^ { * } ( T , B )$ , then such policy does not exist, thus we consider this case non-realizable, and in this case our goal is to maintain a sub-linear standard regret.

We assume that the budget for resource scales linearly with the time horizon. Such assumption is common in the literature of BwK (Badanidiyuru et al. 2018, Lyu and Cheung 2023).

Assumption 1 (Linear in T Resource). There exists a constant $b > 0$ such that $B _ { j } \geq b T$ for all $j \in [ d ]$

## 2.3. Model Discussion

After introducing the problem formulation, we would like to provide some discussions on the impact of resource constraints to the problem. Specifically, with the presence of resource constraints, some properties that are straightforward in the unconstrained setting no longer hold with the presence of resource constraints, suggesting that the additional resource constraints fundamentally change the problem thus require new analysis.

1. Satisficing Regret Is Not a Weaker Regret Notion than Standard Regret. In the setting without resource constraints, satisficing regret is upper bounded by standard regret whenever the satisficing level does not exceed the optimal mean reward. Specifically, letting $r ^ { * } = \operatorname* { m a x } _ { X \in { \mathcal { X } } } r ( X )$ , we have max $\left\{ S - r ( \pi _ { t } ) , 0 \right\} \leq r ^ { * } - r ( \pi _ { t } )$ for every selected policy $\pi _ { t }$ when $S \leq r ^ { * }$ . Therefore, a sub-linear expected standard regret bound directly implies a sub-linear satisficing regret bound. However, this relation no longer holds with the presence of resource constraints. The reason for this is that the optimal reward $Z ^ { * } ( T , B )$ in the LP relaxation does not upper bound the mean reward of every policy selected by a feasible algorithm. Specifically, a feasible algorithm only needs to satisfy the cumulative resource constraints and may select a policy π with $r ( \pi ) > Z ^ { * } ( T , B )$ at individual time steps, even though such a policy has $c _ { j } ( \pi ) > B _ { j } / T$ for at least one resource j and is therefore infeasible for the LP relaxation. When computing standard regret, mean rewards above $Z ^ { * } ( T , B )$ in these time steps can compensate for low mean rewards in other time steps along the trajectory. In contrast, satisficing regret penalizes each shortfall below the threshold S separately.

For instance, consider an even time horizon $T _ { \ast }$ , a single resource with budget $B = T / 2$ , and two arms: an arm with deterministic reward and resource consumption both equal to 1, and the null arm with zero reward and zero resource consumption. The optimal reward in the $\mathrm { L P }$ relaxation is $Z ^ { * } ( T , B ) = 1 / 2$ , and we set the satisficing threshold to $S = 1 / 4$ , so that the instance is realizable. Consider an algorithm that selects the null arm at every odd time step and the other arm at every even time step. This algorithm consumes exactly $T / 2$ units of resource, with the budget depleted only at the final time step, and attains a cumulative reward of $T / 2$ . Therefore, its standard regret is zero. However, each of the $T / 2$ selections of the null arm incurs a satisficing regret of $1 / 4 ,$ , resulting in a total satisficing regret of $T / 8$ . This example shows that even an algorithm with zero standard regret can incur linear satisficing regret in the realizable case.

2. Always Selecting Satisficing Policies Does Not Imply Constant Regret. In the setting without constraints $( e . g .$ , Feng et al. 2025), the algorithm only needs to select satisficing arms, $i . e .$ , arms whose mean rewards are above the threshold $S ,$ as frequently as possible. However, only focusing on selecting arms or policies with satisficing mean rewards does not work with the presence of resource constraints. This is because even if the mean reward of a policy is satisficing, the expected resource consumption of the policy may be large, and always selecting such policies may cause some resource to be depleted early, forcing the learner to select the null arm afterwards and potentially incurring linear in $T$ regret. Therefore under resource constraints, the learner has to keep track of not only the mean reward but also the mean resource consumption of a policy. We define a policy $\pi$ to be satisficing w.r.t. resource B if its mean reward $r ( \pi ) \geq S$ , and its expected resource consumption does not exceed $B / T _ { \mathrm { : } }$ , i.e., $c _ { j } ( \pi ) \leq B _ { j } / T$ for all $j \in [ d ]$

While intuitively the learner needs to select satisficing policies (w.r.t. resource B) as frequently as possible, we would like to remark that due to randomness in resource consumption, always selecting satisficing policies w.r.t. B does not guarantee a constant satisficing regret. The reason for this is that even though always selecting satisficing policies w.r.t. B can satisfy the resource constraints in expectation, the randomness in resource consumption may cause some resource to be depleted early with positive probability. Furthermore, since the variance of resource consumption often scale with $T$ , the expected number of time steps when some resource is depleted can scale with $T ,$ resulting in a satisficing regret that also scales with $T .$ . This is in sharp contrast to the setting without resource constraints, where always selecting satisficing arms incurs zero satisficing regret. Therefore we conclude that the randomness in resource consumption brings further challenge to the problem, thus besides identifying and selecting satisficing policies w.r.t. resources $B ,$ new techniques are needed to hedge against randomness in resource consumptions.

3. SELECTwK: Satisficing Regret Minimization with Resource Constraints In this section, we present our algorithm template SELECTwK for satisficing regret minimization in bandits with resource constraints. We first provide a detailed description of our algorithm in Section 3.1. Then, based on the discussion of why our problem cannot be solved by existing algorithms, we will discuss the novelty of our algorithm in Section 3.2.

## 3.1. Algorithm Template

Similar to Feng et al. (2025), in order to eficiently identify candidate policies with low regret, we assume access to a learning algorithm $\mathtt { A L G } ( \cdot )$ for the corresponding class of bandits with knapsack (BwK) problem $( \mathcal { X } , \mathcal { R } , \mathcal { C } )$ . At a high level, we assume that under any instance of the problem class $( \mathcal { X } , \mathcal { R } , \mathcal { C } )$ , any time horizon t and resource $B ,$ the learning algorithm $\mathtt { A L G } ( t , B )$ incurs a regret sub-linear in t. The formal definition is provided in Condition 1.

Condition 1 (Standard Regret Bound). For problem class $( \mathcal { X } , \mathcal { R } , \mathcal { C } )$ , any time horizon $t \geq 1$ and any B such that $B _ { j } \geq b t / 2$ for all $j \in [ d ]$ , there exists a sequence of learning algorithms ALG such that the regret of $A L G ( t , B )$ under any reward function $r \in \mathcal { R }$ is upper bounded by

$$
\mathbb { E } \left[ t \cdot Z ^ { * } ( t , B ) - \sum _ { s = 1 } ^ { t } r ( \pi _ { s } ) \right] \leq C _ { 1 } t ^ { \alpha } \log ^ { \beta } ( t ) .
$$

where $C _ { 1 } \geq 1 , 1 / 2 \leq \alpha < 1 , \beta \geq 0$ are constants only dependent on the problem class.

However, directly extending the analysis based on Condition 1 in Feng et al. (2025) to our setting with resource constraints is challenging, for which we will provide a detailed discussion in the forthcoming Section 4.

Given a class of BwK problem $( \mathcal { X } , \mathcal { R } , \mathcal { C } )$ and its learning algorithm $\mathtt { A L G } ( t , B )$ , we are now ready to introduce our algorithm template SELECTwK. Specifically, SELECTwK runs in the following steps. Step 0. Pre-processing: To hedge against consumption randomness for the whole algorithm, we first compute the adjusted resource $B _ { j } ^ { \prime } = B _ { j } - \tilde { \Theta } ( \sqrt { T } )$ . For the rest of the paper, we say a policy π to be satisficing if π is satisficing w.r.t. the adjusted resources $B ^ { \prime }$ , i. $. e . , r ( \pi ) \geq S$ and $c _ { j } ( \pi ) \leq B _ { j } ^ { \prime } / T$ for all $j \in [ d ]$ , unless otherwise stated.

After pre-processing, SELECTwK runs in rounds to iteratively identify and test candidate satisficing policies. In each round i, SELECTwK proceeds with the following steps:

Step 1. Identify a Candidate Satisficing Policy: Letting $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha }$ , SELECTwK runs the learning algorithm $\mathtt { A L G } ( t _ { i } , \pmb { b } _ { i } )$ for the first $t _ { i } = \lceil \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \rceil = \lceil 2 ^ { i / \alpha } \rceil$ steps of round i with resources ${ \pmb b } _ { i } = { \pmb B } ^ { \prime } \cdot t _ { i } / T$ . After observing the policy selection of $\mathtt { A L G } ( t _ { i } , \pmb { b } _ { i } )$ , SELECTwK identifies a candidate satisficing policy using the following two steps. First, the algorithm computes an average of the policies selected by $\mathtt { A L G } ( t _ { i } , \pmb { b } _ { i } )$ . Then, a candidate satisficing policy $\hat { \pi } _ { i }$ is computed by taking the mixture of the average policy with the null arm $0 ;$

Step 2. Forced Sampling on the Candidate Policy To validate whether $\hat { \pi } _ { i }$ is truly a satisficing policy, we need to collect enough data from pulling $\hat { \pi } _ { i }$ before performing any statistical test. However, because $\hat { \pi } _ { i }$ is newly derived, it is likely that no data about $\hat { \pi } _ { i }$ is collected when running the learning algorithm. To circumvent this, we perform a forced sampling on $\hat { \pi } _ { i }$ by selecting it for $T _ { i } = \lceil \gamma _ { i } ^ { - 2 } \rceil$ times, and in this way we collect $T _ { i }$ samples of realized reward of the policy $\hat { \pi } _ { i }$ ;

Step 3. Lower Confidence Bound and Upper Confidence Bound Tests: In this step, SELECTwK kicks of the LCB-UCB test as more data is collected from pulling ${ \hat { \pi } } _ { i } ,$ where we use LCB tests to validate whether $r ( \hat { \pi } _ { i } ) \geq S$ , and use UCB tests to validate whether $c _ { j } ( \hat { \pi } _ { i } ) \leq B _ { j } ^ { \prime } / T$ for all $j \in [ d ]$ . We use $k$ (initialized to 0) to denote the number of additional samples of noisy reward generated by selecting $\hat { \pi } _ { i }$ up to the current time step, $\hat { r } _ { i } ^ { \mathrm { t o t } }$ to denote the running total reward and $\hat { c } _ { j i } ^ { \mathrm { t o t } }$ to denote the running total resource consumption of resource $j$ collected from Step 2 and the current step so far. At the same time, it keeps comparing the LCB of $\hat { \pi } _ { i } { } ^ { \gamma } \mathrm { s }$ mean reward, given by $\hat { r } _ { i } ^ { \mathrm { t o t } } / ( T _ { i } + k ) - \sqrt { 4 \log ( T _ { i } + k ) / ( T _ { i } + k ) }$ , against the threshold value $S _ { i }$ , and also compares the UCB of $\hat { \pi } _ { i } \ '$ s mean resource consumption for each $j \in [ d ]$ , given by $\hat { c } _ { j i } ^ { \mathrm { t o t } } / ( T _ { i } + k ) + \sqrt { 4 \log ( d ( T _ { i } + k ) ) / ( T _ { i } + k ) }$ against the $B _ { j } ^ { \prime } / T$ , after acquiring every piece of new data. SELECTwK terminates this round and enters the next whenever either the LCB of reward is less than $S ,$ or the UCB of any resource $j$ consumption is more than $B _ { j } ^ { \prime } / T$

## 3.2. Novelty of SELECTwK

In this section, we discuss the key novelties of our algorithm. We first discuss why existing algorithms such as Feng et al. (2025) cannot efectively handle our setting with resource constraints. Feng et al. (2025) proposes an algorithmic framework for satisficing regret minimization without resource constraints. Specifically, the algorithm iteratively identifies a candidate satisficing arm by randomly drawing an arm from the trajectory of a bandit learning algorithm. Then the algorithm validates whether the candidate arm is truly satisficing by first conducting forced sampling and then run LCB tests on the arm’s mean reward. However, the additional resource constraints bring additional challenges to the problem, both in terms of identifying satisficing policies and validating satisficing policies, and existing algorithms no longer work in this setting.

1. Challenges of Identifying Satisficing Policies: In the setting with resource constraints, a natural idea is still to make use of a learning algorithm with sub-linear expected (standard) regret in the corresponding BwK problem class. However, uniformly drawing a policy selected by the learning algorithm as in Feng et al. (2025) is clearly unable to guarantee finding a satisficing policy w.r.t. resource B with large enough probability. The reason for this is that the learning algorithm may select policies with high reward and high resource consumptions along with others with low reward and low resource consumptions. While algorithm achieves near-optimal cumulative reward with limited resources, it is likely that none of the policies is satisficing in terms of reward and resource consumption at the same time.

Algorithm 1 General Satisficing Bandits with Knapsacks (SELECTwK)   
Input: time horizon $T ,$ satisficing level ${ \underline { { S } } } ,$ resources B for $j \in [ d ]$ , null arm 0   
Set $\begin{array} { r } { \gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha } , B _ { j } ^ { \prime } = B _ { j } - \frac { 1 0 \alpha } { 1 - \alpha } \sqrt { T \log ( 2 d T ^ { 2 } ) } } \end{array}$   
for $i = 1 , 2 , \dots$ . do   
Set $t _ { i } \gets \lceil 1 / \gamma _ { i } ^ { 1 / ( 1 - \alpha ) } \rceil , b _ { i } = B ^ { \prime } \cdot t _ { i } / T , T _ { i } \gets \lceil \gamma _ { i } ^ { - 2 } \rceil$   
Run $\mathtt { A L G } ( t _ { i } , \pmb { b } _ { i } )$ , let $\{ \tilde { \pi } _ { s } \} _ { s \in [ t _ { i } ] }$ be the sequence of policy pulls   
Compute   
$w _ { i } = \operatorname* { m i n } \left\{ 1 , { \frac { 1 0 T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } } { \sqrt { \frac { \log ( 8 d T _ { i } ) } { T _ { i } } } } \right\}$   
Set   
$\hat { \pi } _ { i } = \frac { 1 - w _ { i } } { t _ { i } } \sum _ { s = 1 } ^ { t _ { i } } \tilde { \pi } _ { s } + w _ { i } \mathbf { 1 } _ { 0 }$   
Set $T _ { i } \gets \lceil \gamma _ { i } ^ { - 2 } \rceil$ , let t be the current time s ${ \mathrm { { e p } } } ,$ and pull policy $\hat { \pi } _ { i }$ for the next $T _ { i }$ time steps   
Set k ← 0, $\begin{array} { r } { \hat { r } _ { i } ^ { \mathrm { t o t } }  \sum _ { s = t } ^ { t + T _ { i } - 1 } Y _ { s } } \end{array}$ , and $\begin{array} { r } { \hat { c } _ { j i } ^ { \mathrm { t o t } }  \sum _ { s = t } ^ { t + \hat { T } _ { i } - 1 } C _ { s } ^ { j } } \end{array}$ for $j \in [ d ]$   
while $L C B _ { r } ( \hat { \pi } _ { i } ) \overline { { { \geq S } } }$ and $U C B _ { j } ( \hat { \pi } _ { i } ) \overset { \sim } { \ } B _ { j } ^ { \prime } / \bar { T }$ for $j \in [ d ]$ do   
Set $k \gets k + 1$   
Pull $\hat { \pi } _ { i } ,$ then observe reward $Y _ { t }$ and costs $C _ { t } ^ { j }$ , where t is the current time step   
Update $\hat { r } _ { i } ^ { \mathrm { t o t } }  \hat { r } _ { i } ^ { \mathrm { t o t } } + Y _ { t } .$ , and $\hat { c } _ { j i } ^ { \mathrm { t o t } }  \hat { c } _ { j i } ^ { \mathrm { t o t } } + C _ { t } ^ { j }$ for $j \in [ d ]$   
Set   
$L C B _ { r } ( \hat { \pi } _ { i } ) \gets \frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } }$   
$U C B _ { j } ( \hat { \pi } _ { i } ) \gets \frac { \hat { c } _ { j i } ^ { \mathrm { t o t } } } { T _ { i } + k } + \sqrt { \frac { 4 \log ( d ( T _ { i } + k ) ) } { T _ { i } + k } }$   
end while   
end for

One immediate alternative way of identifying candidate satisficing policies is to take the average over the policies selected by the learning algorithm. Unfortunately, this approach is also unable to guarantee finding a satisficing policy w.r.t. resource B. The reason for this is that any learning algorithm for BwK only guarantees the realized resource consumptions to be within the designated resources, and does not guarantee the cumulative mean resource consumptions of the selected policies to also be within corresponding resources. When the realized resource consumptions happen to be significantly lower than the cumulative mean consumptions, directly taking the average over the selected policies may result in excessive resource consumptions in the average policy. Therefore directly taking the average of the selected policies does not guarantee obtaining a satisficing policy with large enough probability.   
As a side note, a sub-linear standard regret bound does not directly guarantee that the averaged policy has a satisficing mean reward with suficiently high probability, since the standard regret of

an oracle trajectory can be negative. To bypass this challenge, we need to control the expected positive part of the oracle’s cumulative standard regret. In the forthcoming Section 4, we establish this control under the same oracle condition with the help of an LP-duality argument.

2. Challenges of Validating Satisficing Policies: To ensure fast removal of policies that do not satisfy the resource constraints, i. $e . , c _ { j } ( \pi ) > B _ { j } / T$ for some $j \in [ d ]$ , we need to conduct UCB tests on the mean consumption of each resource $j \in [ d ]$ of the candidate policy. When the resource consumption of a policy is marginally smaller than $B _ { j } / T$ for some $j \in [ d ]$ , the policy can be easily removed by the UCB tests even though it satisfies the resource constraints. Therefore besides identifying a satisficing policy, we would also like to ensure the resource consumptions of the policy to be lower than $B / T$ by a significant margin, so that with large enough probability, the policy will not be removed in the UCB tests. The stronger requirements for the mean resource consumption present even more challenge in constructing a proper candidate satisficing policy, and existing literature is unable to efectively resolve this challenge.

3. Challenges of Exploiting Satisficing Policies: Finally, as mentioned in Section 2.3, due to randomness in resource consumption, always selecting a satisficing policy w.r.t. B does not guarantee a constant satisficing regret bound. Even if the algorithm is able to quickly identify a satisficing policy w.r.t. B, always selecting the policy until depletion of some resource could still incur a regret that scales in T because of possibly early depletion of resources.

In light of all the challenges mentioned above, we implement multiple new techniques in SELECTwK. In what follows, we provide a detailed discussion on these new techniques in each step of the algorithm.

Step 0. In the pre-processing step, SELECTwK reserves $\tilde { \Theta } ( \sqrt { T } )$ units of resource to hedge against randomness in resource consumption. Specifically, we first computes the adjusted resource $B _ { j } ^ { \prime } =$ $B _ { j } - { \tilde { \Theta } } ( { \sqrt { T } } )$ , and for the rest of the algorithm, we search for and validate satisficing policies w.r.t. $B ^ { \prime }$ instead of the original resource capacities B. Since the realized resource consumption deviates from the cumulative mean consumption of selected policies by $\tilde { O } ( \sqrt { T } )$ with high probability, as long as the algorithm maintains an expected cumulative resource consumption within $B ^ { \prime }$ , we can ensure that the realized resource consumption does not exceed B with high probability. Therefore by using a more conservative target resource capacity $B ^ { \prime }$ , we resolve our third challenge above and guarantee that resources will not be depleted with high probability.

Step 1. In order to eficiently identify satisficing policies with minimal regret, we use a learning algorithm with sub-linear expected (standard) regret in the corresponding BwK problem class. Specifically, in round i, we run the oracle in $t _ { i }$ time steps, the resource capacity allocated to the oracle is $t _ { i } / T \cdot B ^ { \prime }$ , proportional to the number of time steps of running the oracle.

To generate a candidate policy from the trajectory of the oracle, SELECTwK first takes the average of the policies in the trajectory. Then a candidate policy is computed by taking the mixture of the average policy and the null arm $\mathbf { 1 } _ { 0 } .$ , with the weight of the null arm being $w _ { i } = \tilde { \Theta } ( \gamma _ { i } )$ . Here, taking the mixture with the null arm $\mathbf { 1 } _ { 0 }$ mainly serves the two purposes.

1. As mentioned in the first challenge above, directly taking the average of selected policies may result in excessive expected resource consumption of the constructed policy, and by mixing the average policy with the null arm, we reduce the expected resource consumption of the constructed policy and efectively reduce the possibility of having excessive expected resource consumption of the constructed policy.

2. By mixing the null arm, we are able to ensure that the expected resource consumption of the candidate satisficing policy to be smaller than $B ^ { \prime } / T$ by a margin of at least $\tilde { \Omega } ( \gamma _ { i } )$ with a large enough probability. With this we are able to resolve the second challenge above, since a policy whose expected resource consumption smaller than $B ^ { \prime } / T$ is more likely to survive in the LCB-UCB test if the policy is truly satisficing.

Finally, as i gets large, $\gamma _ { i }$ converges to 0, and the weight of the null arm $w _ { i }$ will also converge to 0. This suggests that as the rounds proceed, mixing the null arm will have a diminishing impact on the expected revenue of the candidate policy.

Step 2. In order to collect enough information on policy reward and resource consumption, we conduct a forced sampling of the candidate policy until the confidence interval is of order $\tilde { \Theta } ( T _ { i } ^ { - 1 / 2 } ) = \tilde { \Theta } ( \gamma _ { i } )$ . Similar to Step 2 of the algorithm in Feng et al. (2025), the purpose of this step is to collect enough data for the candidate policy, so that a small enough confidence interval on the policy’s expected revenue and resource consumption can be constructed. In this way, the policy will be less likely to be removed by the upcoming LCB-UCB tests if it is truly satisficing w.r.t. resources $B ^ { \prime }$

Step 3. In order to validate whether the candidate policy $\hat { \pi } _ { i }$ is truly satisficing, we conduct a series of LCB tests for the policy’s mean reward against the satisficing threshold S, and UCB tests for the policy’s mean resource consumptions against $B ^ { \prime } / T$ . Either when the LCB of the expected revenue falls below $S ,$ or the UCB of the expected consumption of any resource $j \in [ d ]$ exceeds $B _ { j } ^ { \prime } / T$ , the algorithm immediately terminates the current round and starts a new round. Intuitively, the length of running the oracle $t _ { i }$ increases as the rounds proceed, the expected revenue of the candidate policy $r ( \hat { \pi } _ { i } )$ will gradually converge to $Z ^ { * } ( T , B ^ { \prime } )$ , thus if $Z ^ { * } ( T , B ^ { \prime } ) > S$ , the candidate policy will have a higher probability of having an expected revenue above $S$ under mild conditions of the learning oracle (we will provide a detailed discussion on this in the forthcoming Section 4). In terms of resource consumption, by construction in Step 1, the mean resource consumptions of the candidate policy $\hat { \pi } _ { i }$ is smaller than $B ^ { \prime } / T$ by a margin of at least $\tilde { \Omega } ( \gamma _ { i } )$ with a large enough probability. Furthermore, in Step 2, we have collected $T _ { i }$ samples of realized resource consumptions of $\hat { \pi } _ { i } ,$ the length of the confidence interval of mean resource consumption is $\tilde { \Theta } ( 1 / \sqrt { T _ { i } } ) = \tilde { \Theta } ( \gamma _ { i } )$ Therefore with a large enough probability, the UCB of mean resource consumptions will not exceed $B ^ { \prime } / T _ { \ O }$ , thus the probability of the candidate being removed due to excessive resource consumption is small.

## 4. Regret Analysis

In this section, we analyze the regret of SELECTwK under Condition 1, which requires the learning oracle to attain a sub-linear expected standard regret bound. We show that SELECTwK attains constant expected satisficing regret in the realizable case, and sub-linear expected standard regret in the non-realizable case.

To establish these results, we need to first bound the expected positive reward deficit of the candidate policy relative to the LP optimal reward. In the setting without resource constraints, Feng et al. (2025) show that Condition 1 is suficient to attain constant expected satisficing regret in the realizable case. However, directly extending the analysis in Feng et al. (2025) to our setting with resource constraints is challenging. In the setting without resource constraints, the standard regret of every oracle trajectory is nonnegative. Therefore, a sub-linear expected standard regret bound can be used together with Markov’s inequality to show that the candidate arm has a satisficing mean reward with high probability once the oracle runs for suficiently many time steps. In our setting, however, the standard regret $\begin{array} { r } { R _ { t } = t Z ^ { * } ( t , B ) - \sum _ { s = 1 } ^ { t } r ( \pi _ { s } ) } \end{array}$ of some trajectories can be negative. Consequently, when taking expectation, a very large positive regret on some trajectories can be ofset by negative regret on others, and then $\mathbb { E } [ \operatorname* { m a x } \{ R _ { t } , 0 \} ]$ cannot be directly bounded based on an upper bound on $\mathbb { E } [ R _ { t } ]$ . Thus Markov’s inequality is no longer applicable. To address this dificulty, we need to control the contribution of trajectories with negative standard regret.

We make use of the resource constraints satisfied by the learning oracle to control this contribution. Intuitively, if the average of cumulative reward along an oracle trajectory exceeds the LP optimal reward, the cumulative mean consumption along this trajectory must exceed the allocated budget in at least one resource. Since the realized resource consumption remains within the allocated budget, this can occur only when the realized consumption of that resource falls below its cumulative mean consumption. Using LP duality, we relate the magnitude of negative standard regret to these deviations in resource consumption and bound its expected contribution by leading term $\sqrt { t _ { i } } .$ Combining this bound with Condition 1 gives an upper bound on the expected positive part of standard regret, as stated in the following lemma.

Lemma 1. Under Condition 1, for every round $i ,$

$$
\mathbb { E } \left[ \operatorname* { m a x } \left\{ \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) - t _ { i } Z ^ { * } ( T , B ^ { \prime } ) , 0 \right\} \right] \leq \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \sqrt { t _ { i } } ,
$$

therefore,

$$
\mathbb { E } \left[ \operatorname* { m a x } \left\{ t _ { i } Z ^ { * } ( T , \boldsymbol { B } ^ { \prime } ) - \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) , 0 \right\} \right] \leq C _ { 1 } t _ { i } ^ { \alpha } \log ^ { \beta } ( t _ { i } ) + \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \sqrt { t _ { i } } .
$$

The proof of Lemma 1 is provided in Section A.2 of the Appendix. When T is suficiently large, the lemma implies that the expected positive part of cumulative standard regret is bounded by $\tilde { O } ( t _ { i } ^ { \alpha } )$ . Consequently, after averaging the oracle trajectory and accounting for the reward loss from mixing in the null arm, the candidate policy is obtained and satisfies

$$
\mathbb { E } [ \operatorname* { m a x } \left\{ Z ^ { * } ( T , { \cal B } ^ { \prime } ) - r ( \hat { \pi } _ { i } ) , 0 \right\} ] \leq \tilde { O } ( t _ { i } ^ { \alpha - 1 } ) .
$$

We can then apply Markov’s inequality to this nonnegative quantity. Specifically, when $Z ^ { * } ( T , B ^ { \prime } ) > S$ the mean reward of the candidate satisficing policy $r ( \hat { \pi } _ { i } )$ exceeds the satisficing threshold S with high probability for any large enough i. Therefore, with the help of Lemma 1, Condition 1 alone is suficient to obtain the reward guarantee for the candidate policy needed for our analysis. In the realizable case, we define the exceeding gap $\Delta _ { S } ^ { * }$ as the gap between the per-time-step upper bound $Z ^ { * } ( T , B )$ and the satisficing threshold S, given by

$$
\Delta _ { S } ^ { * } = Z ^ { * } ( T , B ) - S .
$$

Satisficing Regret Bound in the Realizable Case under Condition 1: We use the following theorem to provide a constant satisficing regret bound when the learning oracle has a sub-linear expected standard regret bound $( i . e .$ , Condition 1).

Theorem 1. Under Condition 1, if $Z ^ { * } ( T , B ) \geq S$ , then the satisficing regret of SELECTwK is upper bounded by

$$
R e g r e t _ { S } \leq C _ { 1 } ^ { \frac { 2 - \alpha } { 1 - \alpha } } \left( \frac { 1 } { \Delta _ { S } ^ { * } } \right) ^ { \frac { 1 } { 1 - \alpha } } \cdot \mathrm { p o l y l o g } \left( \frac { C _ { 2 } d } { \Delta _ { S } ^ { * } } \right) .\tag{3}
$$

Proof Sketch. A complete proof of Theorem 1 is provided in Section A of the Appendix. To prove the theorem, we proceed with the following steps.

Step 1. Suficient Resources with High Probability: We first prove that SELECTwK does not deplete any resource with high probability. We define the event $\mathcal { E }$ as

$$
\mathcal { E } _ { i } = \left\{ \forall j \in [ d ] , \hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) \leq \frac { \tau _ { i } B _ { j } ^ { \prime } } { T } + 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } \right\} , \quad \mathcal { E } = \bigcap _ { i \leq i _ { \operatorname* { m a x } } } \mathcal { E } _ { i } .
$$

Here, $\tau _ { i }$ is the total number of time steps of forced sampling and LCB-UCB test in round $i , i _ { \mathrm { m a x } }$ is the total number of rounds in time horizon $T _ { \ast }$ , and we use $\hat { r } _ { i } ^ { \mathrm { t o t } } ( t )$ and $\hat { c } _ { j i } ^ { \mathrm { t o t } } ( t )$ as the realized total reward and cost after repeatedly selecting $\hat { \pi } _ { i }$ for t time steps. We use the following proposition to prove that if the event E holds, then no resource will be depleted when running SELECTwK.

Intuitively, this is because the algorithm reserves $\tilde { O } ( \sqrt { T } )$ resources to hedge against randomness in resource consumption, thus as long as the realized resource consumption does not significantly exceed the expected consumption of the selected policies, as assumed in $\mathcal { E } _ { : }$ , then no resource will be depleted by the end of the time horizon.

Proposition 1. If E holds, then for every resource $j \in [ d ]$ , the total resource consumption by the end of the time horizon is upper bounded by

$$
\sum _ { i = 1 } ^ { i _ { \mathrm { m a x } } } ( b _ { j i } + \hat { c } _ { j i } ^ { t o t } ( \tau _ { i } ) ) \leq B _ { j } .
$$

Here $b _ { j i } = B _ { j } ^ { \prime } t _ { i } / T$ is the allocated resource budget. Proof of this proposition is provided in Section A.3 of the Appendix. Then, we use the following proposition to show that E holds with probability at least $1 - 1 / T$

Proposition 2. The event E holds with probability at least $1 - 1 / T$

Proof of this proposition is provided in Section A.4 of the Appendix.

Step 2. Satisficing Regret Bound of Each Round: We show an upper bound on the satisficing regret of round $i ,$ conditioned on the event E that resources are not depleted.

Proposition 3. Under Condition 1, if $Z ^ { * } ( T , B ^ { \prime } ) > S$ , then conditioned on $\cap _ { i ^ { \prime } < i } \mathcal { E } _ { i ^ { \prime } }$ , the expected satisficing regret incurred in round i is upper bounded by

$$
\left( \frac { 6 C _ { 1 } + 6 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 3 3 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } + 2 \right) \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

Proof of this proposition is provided in Section A.6 of the Appendix.

Step 3. Exit Probability of Each Round: Denoting $\Delta _ { S } ^ { \prime } : = Z ^ { * } ( T , B ^ { \prime } ) - S$ , we show that conditioned on $\mathcal { E } .$ , once SELECTwK runs for large enough number of rounds so that $\gamma _ { i } = { \tilde { O } } ( \Delta _ { S } ^ { \prime } )$ , the UCB-LCB test is unlikely to terminate a round and start a new round.

Proposition 4. Under Condition 1, $i f Z ^ { * } ( T , B ^ { \prime } ) \geq S$ , then conditioned on $\cap _ { i ^ { \prime } < i } \mathcal { E } _ { i ^ { \prime } }$ , for every i that satisfies

$$
6 4 \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } \leq \Delta _ { S } ^ { \prime } ,
$$

the probability that round i ends within the time horizon T (conditioned on the event that round i is started within the time horizon) is upper bounded by $1 / 8$

Since each round of SELECTwK runs independently, Proposition 4 indicates that the total number of rounds for SELECTwK can be upper-bounded by a shifted geometric random variable with success rate $7 / 8$ . Proof of this proposition is provided in Section A.7 of the Appendix.

Step 4. Finish the Proof: To finish the proof, we first bound the gap between optimal rewards under the original resource constraint and the adjusted one.

Proposition 5. For any $T _ { i }$ , the diference between optimal reward $Z ^ { * } ( T , B ^ { \prime } )$ of allocating resource $B _ { j } ^ { \prime }$ and optimal reward $Z ^ { * } ( T , B )$ of allocating resource $B _ { j }$ is upper bounded by

$$
Z ^ { * } ( T , \pmb { { B } } ) - Z ^ { * } ( T , \pmb { { B } } ^ { \prime } ) \leq \frac { 1 0 \alpha } { b ( 1 - \alpha ) } \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { T } } .
$$

Proof of this proposition is provided in Section A.5 of the Appendix.

Combining Propositions 1 to 5, we are able to establish the statement in Theorem 1. We first consider the case where T is suficiently large that $1 0 \alpha / ( b ( 1 - \alpha ) ) \cdot \sqrt { \log ( 2 d T ^ { 2 } ) / T } \le \Delta _ { S } ^ { * } / 2$ . By Proposition 5, we have

$$
\Delta _ { S } ^ { \prime } = Z ^ { * } ( T , { \cal B } ^ { \prime } ) - S \ge \Delta _ { S } ^ { * } - \frac { 1 0 \alpha } { b ( 1 - \alpha ) } \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { T } } \ge \frac { \Delta _ { S } ^ { * } } { 2 } .
$$

Moreover, the definition of $B ^ { \prime }$ implies that min<sub>j</sub> $B _ { j } ^ { \prime } \geq b T / 2$ . Therefore, the term $T / \operatorname* { m i n } _ { j } B _ { j } ^ { \prime }$ in Propositions 3 and 4 is bounded by $2 / b ,$ independently of $T$

We first bound the contribution to expected satisficing regret from E. Denote $i _ { 0 } + 1$ as the smallest integer such that the condition in Proposition 4 holds for every $i \geq i _ { 0 } + 1$ . We show that $2 ^ { i _ { 0 } ( 1 - \alpha ) / \alpha } = 1 / \gamma _ { i _ { 0 } } = \tilde { O } ( C _ { 1 } / \Delta _ { S } ^ { \prime } )$ . Then Proposition 3 implies that the contribution from the first $i _ { 0 }$ rounds is upper bounded by

$$
\sum _ { i = 1 } ^ { i _ { 0 } } \tilde { O } \left( C _ { 1 } \gamma _ { i } ^ { - \frac { 1 } { 1 - \alpha } } \right) = \tilde { O } \left( C _ { 1 } \sum _ { i = 1 } ^ { i _ { 0 } } 2 ^ { i / \alpha } \right) = \tilde { O } \left( C _ { 1 } 2 ^ { i _ { 0 } / \alpha } \right) = \tilde { O } \left( C _ { 1 } ^ { \frac { 2 - \alpha } { 1 - \alpha } } \left( \frac { 1 } { \Delta _ { S } ^ { \prime } } \right) ^ { \frac { 1 } { 1 - \alpha } } \right) .
$$

Then we bound the contribution from rounds after $i _ { 0 }$ . By Proposition 4, the probability that SELECTwK enters round $i _ { 0 } + k$ and E holds is at most $8 ^ { 1 - k }$ . Therefore, by Proposition $^ { 3 , }$ the contribution from these rounds is upper bounded by

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { \infty } 8 ^ { 1 - k } O \left( C _ { 1 } \gamma _ { i _ { 0 } + k } ^ { - \frac { 1 } { 1 - \alpha } } \log \left( \frac { 4 d } { \gamma _ { i _ { 0 } + k } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } \right) } \\ & { \displaystyle \le \tilde { O } \left( C _ { 1 } \gamma _ { i _ { 0 } } ^ { - \frac { 1 } { 1 - \alpha } } \right) \sum _ { k = 1 } ^ { \infty } 8 ^ { 1 - k } 2 ^ { k / \alpha } ( k + 1 ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } = \tilde { O } \left( C _ { 1 } ^ { \frac { 2 - \alpha } { 1 - \alpha } } \left( \frac { 1 } { \Delta _ { S } ^ { \prime } } \right) ^ { \frac { 1 } { 1 - \alpha } } \right) . } \end{array}
$$

Here, the series is finite because $\alpha \ge 1 / 2$ implies $8 ^ { 1 - k } 2 ^ { k / \alpha } \leq 8 \cdot 2 ^ { - k }$

Combining the two parts, using $\Delta _ { S } ^ { \prime } \ge \Delta _ { S } ^ { * } / 2$ , and noting that Proposition 2 bounds the contribution from $\mathcal { E } ^ { c }$ by $T \mathrm { P r } ( \mathcal { E } ^ { c } ) \leq 1$ , we obtain

$$
\mathsf { R e g r e t } _ { S } \le 1 + \tilde { O } \left( C _ { 1 } ^ { \frac { 2 - \alpha } { 1 - \alpha } } \left( \frac { 1 } { \Delta _ { S } ^ { \prime } } \right) ^ { \frac { 1 } { 1 - \alpha } } \right) = \tilde { O } \left( C _ { 1 } ^ { \frac { 2 - \alpha } { 1 - \alpha } } \left( \frac { 1 } { \Delta _ { S } ^ { * } } \right) ^ { \frac { 1 } { 1 - \alpha } } \right) .
$$

Finally, if T is not large enough, then $T \le \tilde { O } ( ( 1 / \Delta _ { S } ^ { * } ) ^ { 2 } )$ . Since the expected satisficing regret is at most $T$ and $1 / ( 1 - \alpha ) \geq 2$ , the bound $\tilde { O } ( ( 1 / \Delta _ { S } ^ { * } ) ^ { 1 / ( 1 - \alpha ) } )$ also holds in this case. □

Theorem 1 states that if the learning oracle has a sub-linear expected standard regret bound, then in the realizable case, SELECTwK attains a constant satisficing regret bound that depends on the exceeding gap $\Delta _ { S } ^ { * }$ but independent from the length of the time horizon T. In other words, regardless of how long the learner makes decision over, the satisficing regret incurred by the learner is always upper bounded by a constant that only depends on $\Delta _ { S } ^ { * }$

Standard Regret Bound in the Non-Realizable Case: We use the following theorem to show that in the non-realizable case, SELECTwK has a sub-linear standard regret bound as long as the learning oracle has a sub-linear expected standard regret bound.

Theorem 2. Under Condition 1, if $Z ^ { * } ( T , B ) < S$ , then the standard regret of our algorithm is upper bounded by

$$
R e g r e t \le C _ { 1 } T ^ { \alpha } \cdot \mathrm { p o l y l o g } ( T ) ,\tag{4}
$$

Theorem 2 states that as long as the learning oracle has a ${ \tilde { O } } ( T ^ { \alpha } )$ expected standard regret bound as assumed in Condition 1, in the non-realizable case, SELECTwK is also able to attain a ${ \tilde { O } } ( T ^ { \alpha } )$ regret bound similar to that of the learning oracle. In other words, SELECTwK is robust in the non-realizable case and can attain a similar performance with the learning oracle used in the algorithm in terms of standard regret minimization.

Remark 2 (Limitations under Standard Regret Condition). Compared with the result of Feng et al. (2025) in the setting without resource constraints, the expected satisficing regret bound of SELECTwK under Condition 1 in the realizable case is weaker in two aspects.

Higher Order in $1 / \Delta _ { S } ^ { * }$ : the order of $1 / \Delta _ { S } ^ { * }$ increases from $\alpha / ( 1 - \alpha )$ to $1 / ( 1 - \alpha )$ . The reason for this is that a sub-linear expected standard regret bound does not guarantee a sub-linear expected satisficing regret when running the learning oracle. In particular, an oracle run of $t _ { i }$ time steps could incur $\Theta ( t _ { i } ) = \Theta ( \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } )$ expected satisficing regret, whereas in the setting without resource constraints, its expected satisficing regret is bounded by $\tilde { O } ( t _ { i } ^ { \alpha } ) = \tilde { O } ( \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } )$ . Once the $\gamma _ { i }$ becomes suficiently small relative to $\Delta _ { S } ^ { * }$ , this larger regret incurred when running the oracle leads to the worse dependence from $\tilde { O } ( ( 1 / \Delta _ { S } ^ { * } ) ^ { \alpha / ( 1 - \alpha ) } )$ to $\tilde { O } ( ( 1 / \Delta _ { S } ^ { * } ) ^ { 1 / ( 1 - \alpha ) } )$ in the setting with resource constraints.

Linear Regret under Small $\Delta _ { S } ^ { * }$ : although the bound is independent of T for any fixed $\Delta _ { S } ^ { * }$ SELECTwK does not necessarily attain sub-linear expected satisficing regret when $\Delta _ { S } ^ { * }$ is small relative to the time horizon. This is again because Condition 1 does not rule out the expected satisficing regret growing linearly with the number of time steps spent running the learning oracle. SELECTwK may continue to start new rounds until the length of an oracle run becomes comparable to T, therefore incurring $\Theta ( T )$ expected satisficing regret.

In light of the aforementioned limitations of regret bounds under Condition 1, in the forthcoming Section 5, we propose a (slightly) stronger condition on the learning oracle, under which SELECTwK is able to attain an improved constant satisficing regret for fixed $\Delta _ { S } ^ { * }$ as well as a sub-linear in T satisficing regret for small $\Delta _ { S } ^ { * }$

## 5. Improved Regret Bound

In this section, we introduce a strong regret condition that controls the sum of positive reward deficits relative to the LP optimal reward. Based on this strong regret condition, we show that both limitations discussed in Remark 2 under Condition 1 can be addressed: SELECTwK attains a better constant expected satisficing regret bound in terms of $\Delta _ { S } ^ { * }$ , together with a sub-linear bound in $T$ that remains valid when $\Delta _ { S } ^ { * }$ is small.

## 5.1. Strong Regret Condition

Although Lemma 1 provides a sub-linear bound on the expected positive part of cumulative standard regret, it does not directly provide a sub-linear bound on the satisficing regret incurred when running the learning oracle. The reason for this is that the lemma takes the positive part after summing over all time steps, whereas satisficing regret sums the positive deficits at each individual time step. Thus, the bound on the positive part of the sum, given by Lemma 1, does not provide a corresponding bound on the sum of the positive parts. To obtain a sharper satisficing regret bound, we therefore consider a stronger notion of regret that measures the cumulative positive deficit of the selected policies compared to $Z ^ { * } ( t , B )$ , given by

$$
\sum _ { s = 1 } ^ { t } \operatorname* { m a x } \{ Z ^ { * } ( t , { \cal B } ) - r ( \pi _ { s } ) , 0 \} .\tag{5}
$$

We say that an algorithm has a sub-linear expected strong regret bound if the expected cumulative deficit compared to $Z ^ { * } ( t , B )$ defined in (5) is sub-linear in the time horizon t. The formal definition of the condition is provided in Condition 2.

Condition 2 (Strong Regret Bound). For any instance of the problem class $( \mathcal { X } , \mathcal { R } , \mathcal { C } )$ , any time horizon $t \geq 1$ and any B such that $B _ { j } \geq b t / 2$ for all $j \in [ d ]$ , the learning oracle $A L G ( t , B )$ satisfies

$$
\mathbb { E } \left[ \sum _ { s = 1 } ^ { t } \operatorname* { m a x } \left\{ Z ^ { * } ( t , B ) - r ( \pi _ { s } ) , 0 \right\} \right] \leq C _ { 1 } t ^ { \alpha } \log ^ { \beta } ( t ) .
$$

Intuitively, Condition 2 dose not only require the oracle algorithm to have an asymptotically optimal cumulative reward, but also requires the policies selected by the algorithm to have a stable mean revenue not significantly lower than $Z ^ { * } ( t , B )$ throughout the time horizon. In other words, Condition 2 rules out algorithms where a majority of the cumulative reward is generated in only a fraction of the time horizon, and little reward is gained for the rest of the time horizon. While Condition 2 is stronger than Condition 1, it is still mild and can be satisfied by many existing BwK algorithms. For instance, in Section D of the Appendix, we show that Condition 2 can be satisfied by many existing algorithms including Wang et al. (2014), Badanidiyuru et al. (2018).

With Condition $^ { 2 , }$ we are also able to show that the mean reward of the candidate satisficing policy $r ( \hat { \pi } _ { i } )$ exceeds the satisficing threshold S with a large enough probability for any large enough i. Specifically, by the convexity of $\operatorname* { m a x } \{ x , 0 \}$ , Condition 2 implies that E[max $\{ Z ^ { * } ( t _ { i } , \pmb { b } _ { i } ) - r ( \bar { \pi } _ { i } ) , 0 \} ] \leq$ $\tilde { O } ( t _ { i } ^ { \alpha - 1 } )$ , where $\bar { \pi } _ { i }$ is the average of the selected policies of the algorithm. Then by Markov’s inequality, $Z ^ { * } ( t _ { i } , \mathbf { b } _ { i } ) - r ( \bar { \pi } _ { i } ) \leq \tilde { O } ( t _ { i } ^ { \alpha - 1 } )$ holds with a probability at least $3 / 4$ . By definition of the candidate satisficing arm $\hat { \pi } _ { i }$ we also have that $Z ^ { * } ( t _ { i } , \mathbf { b } _ { i } ) - r ( \hat { \pi } _ { i } ) \le \tilde { O } ( t _ { i } ^ { \alpha - 1 } )$ holds with a probability at least $3 / 4$ . Since $t _ { i }$ increases as the rounds proceed, when i is large enough, the probability that $r ( \hat { \pi } _ { i } ) \geq S$ will be at least $3 / 4$ . We refer to the forthcoming Proposition 7 for details.

In brief, Condition 2 strengthens Condition 1 by controlling the expected sum of positive deficits at individual time steps. It enables SELECTwK to attain a better constant expected satisficing regret bound, together with a sub-linear expected satisficing regret bound, to be discussed in Section 5.2.

## 5.2. Regret Analysis under Strong Regret Condition

Satisficing Regret Bound under Condition 2: In what follows, we use the following theorem to show that if we assume the learning oracle to satisfy Condition 2, then SELECTwK attains a constant satisficing regret bound better than the regret bound in Theorem 1.

Theorem 3. Under Condition 2, $i f Z ^ { * } ( T , B ) \geq S$ , then the satisficing regret of SELECTwK is upper bounded $b y$

$$
R e g r e t _ { S } \leq \operatorname* { m i n } \left\{ C _ { 1 } ^ { \frac { 1 } { 1 - \alpha } } \left( \frac { 1 } { \Delta _ { S } ^ { * } } \right) ^ { \frac { \alpha } { 1 - \alpha } } \cdot \mathrm { p o l y l o g } \left( \frac { C _ { 2 } ^ { \prime } d } { \Delta _ { S } ^ { * } } \right) , C _ { 1 } T ^ { \alpha } \cdot \mathrm { p o l y l o g } ( T ) \right\} .\tag{6}
$$

Proof Sketch. A complete proof of Theorem 3 is given in Section B of the Appendix. We follow proof pattern similar to Theorem 1. The alternative condition on the learning oracle does not afect the implementation of the algorithm, thus Proposition 1 and Proposition 2 remain valid in this case. In other words, SELECTwK will not exhaust any resource before the end of the time horizon, conditioned on $\mathcal { E } .$ , and this bounded resource consumption happens with high probability.

Denoting $\Delta _ { S } ^ { \prime } : = Z ^ { * } ( T , B ^ { \prime } ) - S$ , we then show two critical results. In the first one, similar to Proposition 3, we show an upper bound on the satisficing regret of round i.

Proposition 6. Under Condition 2, $i f Z ^ { * } ( T , B ^ { \prime } ) \geq S$ , then conditioned on $\cap _ { i ^ { \prime } < i } \mathcal { E } _ { i ^ { \prime } }$ , the satisficing regret incurred in round i is upper bounded by

$$
4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname * { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname * { m a x } \{ \beta , 1 / 2 \} } .
$$

Proof of this proposition is provided in Section B.2 of the Appendix. We also show that once our algorithm runs for large enough number of rounds so that $\gamma _ { s } = \tilde { O } ( \Delta _ { S } ^ { \prime } )$ , it is unlikely to start a new round.

Proposition 7. Under Condition 2, if $Z ^ { * } ( T , B ^ { \prime } ) \geq S$ , then conditioned on $\cap _ { i ^ { \prime } < i } \mathcal { E } _ { i ^ { \prime } } .$ , for every i that satisfies

$$
6 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } \leq \Delta _ { S } ^ { \prime } ,
$$

the probability that round i ends within the time horizon T (conditioned on the event that round i is started within the time horizon) is upper bounded by $1 / 8$

Proof of this proposition is provided in Section B.3 of the Appendix. Proposition 6 is similar to Proposition 3 and Proposition 7 is similar to Proposition 4. Following the same proof pattern in Section 4, we first consider T suficiently large that $\Delta _ { S } ^ { \prime } \ge \Delta _ { S } ^ { * } / 2$ and min<sub>j</sub> $B _ { j } ^ { \prime } \geq b T / 2$ . Denote $i _ { 0 } + 1$ as the smallest integer such that the condition in Proposition 7 holds for every $i \geq i _ { 0 } + 1$ . The same argument as in Section 4 gives $1 / \gamma _ { i _ { 0 } } = \tilde { O } ( C _ { 1 } / \Delta _ { S } ^ { \prime } )$ . Under the strong regret condition, Proposition 6 bounds the expected satisficing regret incurred in round i by $\tilde { O } ( C _ { 1 } \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } ) = \tilde { O } ( C _ { 1 } 2 ^ { i } )$ . Therefore, combining Proposition 6 and 7, the contribution to expected satisficing regret from E is upper bounded by

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { i _ { 0 } } \tilde { O } \Big ( C _ { 1 } \gamma _ { i } ^ { - \frac { \alpha } { 1 - \alpha } } \Big ) + \sum _ { k = 1 } ^ { \infty } 8 ^ { 1 - k } \tilde { O } \Big ( C _ { 1 } \gamma _ { i _ { 0 } + k } ^ { - \frac { \alpha } { 1 - \alpha } } \Big ) } } \\ { { \displaystyle = \tilde { O } ( C _ { 1 } 2 ^ { i _ { 0 } } ) = \tilde { O } \left( C _ { 1 } ^ { \frac { 1 } { 1 - \alpha } } \left( \frac { 1 } { \Delta _ { S } ^ { \prime } } \right) ^ { \frac { \alpha } { 1 - \alpha } } \right) . } } \end{array}
$$

Using $\Delta _ { S } ^ { \prime } \ge \Delta _ { S } ^ { * } / 2$ and adding the contribution from $\mathcal { E } ^ { c }$ , which is at most 1, establishes the constant regret bound $\tilde { O } ( C _ { 1 } ^ { 1 / ( 1 - \alpha ) } ( 1 / \Delta _ { S } ^ { * } ) ^ { \alpha / ( 1 - \alpha ) } )$ for suficiently large T.

We further show that the expected satisficing regret is bounded by ${ \tilde { O } } ( C _ { 1 } T ^ { \alpha } )$ . Let $i _ { \mathrm { m a x } }$ denote the total number of rounds. Since $t _ { i } = \lceil 2 ^ { i / \alpha } \rceil$ , we have $2 ^ { i _ { \operatorname* { m a x } } } = O ( T ^ { \alpha } )$ . As Proposition 6 also bounds the expected positive reward deficits relative to min $\{ S , Z ^ { * } ( T , B ^ { \prime } ) \}$ }, we take the diference between $Z ^ { * } ( T , B )$ and $Z ^ { * } ( T , B ^ { \prime } )$ by Proposition 5 into consideration and obtain

$$
\begin{array} { r l } & { \mathsf { R e g r e t } _ { S } \leq 1 + T ( Z ^ { * } ( T , { \boldsymbol B } ) - Z ^ { * } ( T , { \boldsymbol B } ^ { \prime } ) ) + \displaystyle \sum _ { i = 1 } ^ { i _ { \operatorname* { m a x } } } \tilde { O } ( C _ { 1 } 2 ^ { i } ) } \\ & { \quad \quad \quad = \tilde { O } ( \sqrt { T } ) + \tilde { O } ( C _ { 1 } T ^ { \alpha } ) = \tilde { O } ( C _ { 1 } T ^ { \alpha } ) . } \end{array}
$$

Finally, if T is not large enough, then $T = \tilde { O } ( ( 1 / \Delta _ { S } ^ { * } ) ^ { 2 } )$ . Applying the sub-linear bound gives expected satisficing regret at most $\tilde { O } ( C _ { 1 } T ^ { \alpha } ) \leq \tilde { O } ( C _ { 1 } ( 1 / \Delta _ { S } ^ { * } ) ^ { 2 \alpha } ) \leq \tilde { O } ( C _ { 1 } ^ { 1 / ( 1 - \alpha ) } ( 1 / \Delta _ { S } ^ { * } ) ^ { \alpha / ( 1 - \alpha ) } )$ □

Since Condition 2 implies Condition 1, Theorem 2 guarantees that the ${ \tilde { O } } ( T ^ { \alpha } )$ expected standard regret bound in the non-realizable case also holds under Condition 2.

Theorem 3 suggests that if we assume an expected strong regret bound of the learning oracle as in Condition 2, SELECTwK is able to attain a better regret bound than the one established in Theorem 1, where we assume a sublinear expected standard regret bound. The regret established in

Theorem 3 improves from the the one in Theorem 1 in two aspects: first, the constant satisficing regret bound in Theorem 3 is $\tilde { O } ( ( 1 / \Delta _ { S } ^ { * } ) ^ { \alpha / ( 1 - \alpha ) } )$ , a smaller constant rate than the $\tilde { O } ( ( 1 / \Delta _ { S } ^ { * } ) ^ { 1 / ( 1 - \alpha ) } )$ established in Theorem 1; second, when $\Delta _ { S } ^ { * }$ is small, under Condition 2, SELECTwK can still attain a ${ \tilde { O } } ( T ^ { \alpha } )$ sub-linear satisficing regret bound similar to the regret bound of the learning oracle, while such a sub-linear satisficing regret bound is lacking under Condition 1 with a very small $\Delta _ { S } ^ { * }$

Such an improvement is mainly due to smaller satisficing regret incurred when running the learning oracle under Condition 2. As we have mentioned, if we assume Condition 1 for the learning oracle, running the learning oracle for $t _ { i }$ time steps could still incur $\Theta ( t _ { i } )$ regret in the realizable case. On the other hand, under Condition 2, we are able to prove that the satisficing regret incurred when running the learning oracle in round i is upper bounded by $\tilde { O } ( t _ { i } ^ { \alpha } )$ , a significant improvement from the potentially $\Theta ( t _ { i } )$ satisficing regret under Condition 1. Intuitively, by assuming Condition 2, the learning oracle is able to distribute the gained reward more evenly throughout the time horizon. The condition is able to avoid an undesirable situation where a majority of the reward is gained in a small fraction of the time horizon, while minimal reward is gained for the rest of the time horizon, making a better satisficing regret bound possible when running the learning oracle.

## 6. Examples

In what follows, we instantiate SELECTwK to three classes of bandit optimization problems with knapsacks: finite-armed bandits with knapsacks, dynamic pricing with continuous prices under inverse demand, and dynamic pricing with finite prices and unknown demand. For the first two classes, we establish that the corresponding oracles satisfy the strong regret condition and derive the resulting expected satisficing and standard regret bounds. For the final one, we state the expected satisficing and standard regret bounds based on the standard regret condition. These examples are not exhaustive, and similar regret bounds can be established for other classes of bandit optimization problems with knapsacks.

1. Finite-Armed Bandits with Knapsacks: Consider a finite arm set X with $K = | \mathcal { X } |$ arms and d resources, where the resource budgets satisfy $B _ { j } \geq b T$ for all $j \in [ d ]$ . We use BalancedExploration in Badanidiyuru et al. (2018) as the learning oracle, with resource budgets normalized as described in Section D.1 of the Appendix. By adapting its analysis to control the positive reward deficits at individual time steps, we show that this oracle has an expected strong regret bound of $O ( { \sqrt { d K t } } \log ( d K t ) ^ { 2 } )$ over t time steps for fixed b as established in Theorem 4 in the Appendix. In particular, Condition 2 holds with $\alpha = 1 / 2$ . Combining this result with Theorems 3 and 2, we obtain the following corollary. Corollary 1. By using BalancedExploration as ALG, if $Z ^ { * } ( T , B ) \geq S$ , the expected satisficing regret of SELECTwK is bounded by

$$
R e g r e t _ { S } \leq \operatorname* { m i n } \left\{ \frac { d K } { \Delta _ { S } ^ { \ast } } \cdot \mathrm { p o l y l o g } \left( \frac { d K } { \Delta _ { S } ^ { \ast } } \right) , \sqrt { d K T } \cdot \mathrm { p o l y l o g } ( T ) \right\} .
$$

If $Z ^ { * } ( T , B ) < S$ , then the expected standard regret of SELECTwK is bounded by $R e g r e t \leq \sqrt { d K T }$ polylog(T).

2. Dynamic Pricing with Limited Inventory: Consider a seller with $B \geq b T$ units of a single product to sell over T time steps. At each time step, the seller ofers a price $p \in [ p , \bar { p } ]$ , and a purchase occurs with probability $\lambda ( p )$ . Thus, the mean reward and resource consumption are $r ( p ) = p \lambda ( p )$ and $c ( p ) = \lambda ( p )$ , respectively, with rewards normalized as in our model. We assume that demand is decreasing, demand and inverse demand (the inverse function of $\lambda ( \cdot ) )$ are Lipschitz continuous, and revenue as a function of demand is twice diferentiable and strictly concave, with curvature bounded away from zero. The precise regularity assumptions are stated in Proposition 12 of the Appendix.

We use a discrete-time version of DPA in Wang et al. (2014) as the learning oracle. This version replaces the continuous-time Poisson arrivals in the original model with Bernoulli purchases at discrete time steps. In Section D.2 of the Appendix, we describe the adaptation and establish an expected strong regret bound of $O ( \sqrt { t } \log ( t ) ^ { 6 } )$ . Therefore, Condition 2 holds with $\alpha = 1 / 2$ and $\beta = 6$ Together with Theorems 3 and 2, this gives the following corollary.

Corollary 2. By using Discrete DPA as ALG, if $Z ^ { * } ( T , B ) \geq S$ , the expected satisficing regret of SELECTwK is bounded by

$$
R e g r e t _ { S } \leq \operatorname* { m i n } \left\{ \frac { 1 } { \Delta _ { S } ^ { * } } \cdot \mathrm { p o l y l o g } \left( \frac { 1 } { \Delta _ { S } ^ { * } } \right) , \sqrt { T } \cdot \mathrm { p o l y l o g } ( T ) \right\} .
$$

If $Z ^ { * } ( T , B ) < S$ , then the expected standard regret of SELECTwK is bounded by Regre $t \leq \sqrt { T }$ polylog(T).

3. Thompson Sampling with Fixed Inventory Constraints: Consider dynamic pricing with limited inventory and a finite set of K admissible prices. We use TS-fixed by Ferreira et al. (2018) as the learning oracle. At each time step, TS-fixed samples demand parameters from their posterior distribution and solves an LP with fixed per-period inventory constraints to obtain a randomized pricing policy. Although we do not establish a strong regret bound for TS-fixed, a sub-linear expected standard regret bound satisfying Condition 1, i.e., standard regret upper bound $O ( \sqrt { K t } \mathrm { p o l y l o g } ( K t ) )$ , is suficient to obtain the following guarantees from Theorems 1 and 2.

Corollary 3. By using TS-fixed as the learning oracle, if $Z ^ { * } ( T , B ) > S$ , the expected satisficing regret of SELECTwK is bounded by

$$
R e g r e t _ { S } \leq \frac { K ^ { 3 / 2 } } { ( \Delta _ { S } ^ { * } ) ^ { 2 } } \cdot \mathrm { p o l y l o g } \left( \frac { K } { \Delta _ { S } ^ { * } } \right) .
$$

If $Z ^ { * } ( T , B ) < S$ , then its expected standard regret is bounded by $R e g r e t \le \sqrt { K T } \cdot \mathrm { p o l y l o g } ( K T )$

## 7. Numerical Experiments

In this section, we conduct numerical experiments to test the performance of SELECTwK on dynamic pricing with limited supply. All experiments are run on a PC with 4-core CPU and 16GB of RAM.

We use the setting given by Ferreira et al. (2018). There are $T$ buyers, equivalent to time budget, arriving one by one and making a decision whether to buy an item after comparing their private valuation to observed price. There are totally B items, which is equivalent to resource budget with $d = 1$ . For each price $p \in [ 0 , 1 0 ]$ , the expected selling rate $c ( p ) \in [ 0 , 1 ]$ , which is equivalent to the expected resource consumption, is realized to be 0 or 1 based on buyer’s decision.

There are two cases where $Z ^ { * } ( T , \mathbf { B } )$ is reached by exhausting resource or time budget (Besbes and Zeevi 2009, Wang et al. 2014). For the former one, the best price is attained at the boundary of the constraints, $i . e . ,$ , to set a price that sells all items before the end of the time horizon and apply $p _ { \infty }$ (the null price) till the end. For the latter one, the best price is attained strictly within the feasible region, i.e., to set the price with highest $p c ( p )$ through the time horizon and no resource will be exhausted.

We consider an instance of $c ( p ) = 1 - 0 . 1 p$ with available price set $\{ 1 , 2 , \cdots , 9 , 1 0 \}$ , then highest per-buyer (per-timestep) reward is attained as 2.5 for $p = 5$ . The budget B increases linearly in $T _ { i }$ and we control the value of $B / T$ to simulate for diferent cases.

## 7.1. Exhausting Time Budget

Setup: In this case, we set $B / T = 0 . 6 $ , then the optimal reward $Z ^ { * } ( T , B ) = 2 . 5$ is reached by price $p ^ { * } = \arg \operatorname* { m a x } p c ( p )$ that exhausts time budget instead of resource budget with high probability. The length of the time horizon is set to be 500 to 6000 with a stepsize of 500. We conduct experiments for both the realizable case and the non-realizable case. For the realizable case, we set the satisficing level $S = 1$ ; for the non-realizable case, we set the satisficing level $S = 5$ . In both cases, we compare SELECTwK against TS-fixed in Ferreira et al. (2018). The experiment is repeated for 1000 times and we report the average results.

Results: The results of the realizable case are presented in Figure 1(a), and the results of the non-realizable case is presented in Figure 1(b). From Figure 1(a), one can see that the satisficing regret of SELECTwK stabilizes at a constant level after $T$ exceeds 500, implying that SELECTwK attains a constant expected satisficing regret in the realizable case. At the same time, the satisficing regret of TS-fixed keeps increasing with $T$ and fails to be stable at a constant. Comparing SELECTwK and TS-fixed, the satisficing regret incurred by SELECTwK is also smaller than TS-fixed.

As for the non-realizable case, one can see from Figure 1(b) that SELECTwK achieves better empirical performance than TS-fixed. This is because SELECTwK exploits more aggressively on near-optimal arms during forced sampling while TS-fixed keeps exploring with posterior sampling.

![](images/81b8ea6051c5c55b180725ef0422d6b920ff69a2154c7647e9aaba711e4ca2f1.jpg)  
(a) Satisficing Regret in the Realizable Case

![](images/44574810478d5216baf863e40f0a5df7aae00a19cb408c113527f3e2191538ec.jpg)  
(b) Standard Regret in the Non-Realizable Case  
Figure 1 Comparison of SELECTwK and TS-fixed under exhausting time budget case

## 7.2. Exhausting Resource Budget

Setup: In this case, we set $B / T = 0 . 4$ , then the optimal reward $Z ^ { * } ( T , B ) = 2 . 4$ is reached by price $p = \arg \operatorname* { m i n } | c ( p ) - B / T | = 6$ which has selling rate $c ( p ) = 0 . 4$ that will exhaust the resource budget. The length of the time horizon is set to be 500 to 6000 with a stepsize of 500. We conduct experiments for both the realizable case and the non-realizable case. For the realizable case, we set the satisficing level $S = 1$ ; for the non-realizable case, we set the satisficing level $S = 5$ . In both cases, we compare SELECTwK against TS-fixed in Ferreira et al. (2018). The experiment is repeated for 1000 times and we report the average results.

Results: The results of the realizable case is presented in Figure 2(a), and the results of the nonrealizable case is presented in Figure 2(b). From Figure 2(a), one can see that SELECTwK stabilizes at a constant level after T exceeds 1000, implying that SELECTwK attains a constant expected satisficing regret in the realizable case. At the same time, the satisficing regret of TS-fixed keeps increasing with T and fails to be stable at a constant. Comparing SELECTwK and TS-fixed, SELECTwK incurs satisficing regret that grows similarly to that of TS-fixed but then smaller than TS-fixed after being stable.

As for the non-realizable case, one can see from Figure 2(b) that SELECTwK achieves better empirical performance than TS-fixed. Similar to the case of $B / T = 0 . 6$ , this is because SELECTwK exploits more aggressively on near-optimal arms during forced sampling while TS-fixed keeps exploring with posterior sampling.

## 8. Conclusion

In this paper, we study satisficing regret minimization in bandits with knapsacks and propose SELECTwK, a general algorithmic template for maintaining mean rewards above a satisficing threshold while respecting finite resource constraints. We show that a bandit-with-knapsack learning oracle with a sub-linear expected standard regret bound is suficient for SELECTwK to attain constant expected satisficing regret in the realizable case. When the threshold is unattainable, SELECTwK preserves the oracle’s expected standard regret. We further introduce a strong regret condition that controls the expected sum of positive reward deficits at individual time steps relative to the optimal feasible mean reward. Under this condition, SELECTwK achieves a sharper dependence on the exceeding gap, together with a satisficing regret bound that is sub-linear up to logarithmic factors in the time horizon. We instantiate SELECTwK to finite-armed bandits with knapsacks and dynamic pricing with resource constraints. Finally, numerical experiments on dynamic pricing with limited inventory support our theoretical findings and demonstrate the empirical performance of SELECTwK.

![](images/b4d5bdf70ccf04feb1009721468b792976461df31ee4ba14416d29f4cbec9f89.jpg)  
(a) Satisficing Regret in the Realizable Case

![](images/8730bdd6a47ffeff9b666a95cbc669b8c941a262c5cec48973c621eb1a63d89f.jpg)  
(b) Standard Regret in the Non-Realizable Case  
Figure 2 Comparison of SELECTwK and TS-fixed under exhausting resource budget case

Several directions remain for future research. First, advances in machine learning, particularly large language models, motivate extending our framework to settings with ML-generated proxy rewards from these models (Ji et al. 2025, Ma et al. 2026), then an important question is how to leverage these proxy rewards to improve the satisficing regret. Second, our analysis assumes stationary setting. Future studies could investigate how to deal with non-stationary reward and resource consumption distributions (Cheung et al. 2022, Liu et al. 2022).

## References

Abernethy, Jacob, Kareem Amin, Ruihao Zhu. 2016. Threshold bandits, with and without censored feedback. Advances in Neural Information Processing Systems 29.

Babaiof, Moshe, Shaddin Dughmi, Robert Kleinberg, Aleksandrs Slivkins. 2015. Dynamic pricing with limited supply.

Badanidiyuru, Ashwinkumar, Robert Kleinberg, Aleksandrs Slivkins. 2018. Bandits with knapsacks. Journa of the ACM (JACM) 65(3) 1–55.

Besbes, Omar, Assaf Zeevi. 2009. Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations research 57(6) 1407–1420.

Besbes, Omar, Assaf Zeevi. 2012. Blind network revenue management. Operations research 60(6) 1537–1550.

Bubeck, S´ebastien, Vianney Perchet, Philippe Rigollet. 2013. Bounded regret in stochastic multi-armed bandits. Conference on Learning Theory. PMLR, 122–134.

Chen, Ningyuan, Guillermo Gallego. 2022. A primal–dual learning algorithm for personalized dynamic pricing with an inventory constraint. Mathematics of Operations Research 47(4) 2585–2613.

Chen, Qi, Stefanus Jasin, Izak Duenyas. 2019. Nonparametric self-adjusting control for joint learning and optimization of multiproduct pricing with finite resource capacity. Mathematics of Operations Research 44(2) 601–631.

Cheung, Wang Chi, David Simchi-Levi, Ruihao Zhu. 2022. Hedging the drift: Learning to optimize under nonstationarity. Management Science 68(3) 1696–1713.

Combes, Richard, Chong Jiang, Rayadurgam Srikant. 2015. Bandits with budgets: Regret lower bounds and optimal algorithms. ACM SIGMETRICS Performance Evaluation Review 43(1) 245–257.

Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, Cliford Stein. 2009. Introduction to Algorithms. MIT Press.

Ding, Wenkui, Tao Qin, Xu-Dong Zhang, Tie-Yan Liu. 2013. Multi-armed bandit with budget constraint and variable costs. Proceedings of the AAAI Conference on Artificial Intelligence, vol. 27. 232–238.

Feng, Qing, Tianyi Ma, Ruihao Zhu. 2025. Satisficing regret minimization in bandits. The Thirteenth International Conference on Learning Representations.

Ferreira, Kris Johnson, David Simchi-Levi, He Wang. 2018. Online network revenue management using thompson sampling. Operations research 66(6) 1586–1602.

Fitzsimons, Gavan J. 2000. Consumer response to stockouts. Journal of Consumer Research 27(2) 249–266.

Flajolet, Arthur, Patrick Jaillet. 2015. Logarithmic regret bounds for bandits with knapsacks. arXiv preprint arXiv:1510.01800 .

Garivier, Aur´elien, Pierre M´enard, Gilles Stoltz. 2019. Explore first, exploit next: The true shape of regret in bandit problems. Mathematics of Operations Research 44(2) 377–399.

Google. 2026. Concepts in service monitoring URL https://docs.cloud.google.com/stackdriver/docs/ solutions/slo-monitoring.

Hajiabolhassan, Hossein, Ronald Ortner. 2026. Online regret bounds for satisficing in markov decision processes. Mathematics of Operations Research 51(2) 1097–1119.

H¨uy¨uk, Alihan, Cem Tekin. 2021. Multi-objective multi-armed bandit with lexicographically ordered and satisficing objectives. Machine Learning 110(6) 1233–1266.

Ji, Wenlong, Yihan Pan, Ruihao Zhu, Lihua Lei. 2025. Multi-armed bandits with machine learning-generated surrogate rewards. arXiv preprint arXiv:2506.16658 .

Jing, Xiaoqing, Michael Lewis. 2011. Stockouts in online retailing. Journal of Marketing Research 48(2) 342–354.

Kano, Hideaki, Junya Honda, Kentaro Sakamaki, Kentaro Matsuura, Atsuyoshi Nakamura, Masashi Sugiyama. 2019. Good arm identification via bandit feedback. Machine Learning 108 721–745.

Kleywegt, Anton J, Jason D Papastavrou. 1998. The dynamic and stochastic knapsack problem. Operations research 46(1) 17–35.

Levy, Michael, Barton Weitz, Dhruv Grewal. 2023. Retailing Management. McGraw Hill.

Li, Xiaocheng, Chunlin Sun, Yinyu Ye. 2021. The symmetry between arms and knapsacks: A primal-dual approach for bandits with knapsacks. International Conference on Machine Learning. PMLR, 6483–6492.

Li, Zhekai, Tianyi Ma, Cheng Hua, Ruihao Zhu. 2025. Identifying all {\epsilon}-best arms in (misspecified) linear bandits. arXiv preprint arXiv:2510.00073 .

Liu, Shang, Jiashuo Jiang, Xiaocheng Li. 2022. Non-stationary bandits with knapsacks. Advances in Neural Information Processing Systems 35 16522–16532.

Locatelli, Andrea, Maurilio Gutzeit, Alexandra Carpentier. 2016. An optimal algorithm for the thresholding bandit problem. International Conference on Machine Learning. PMLR, 1690–1698.

Lufkin, Bryan. 2021. Do ’maximisers’ or ’satisficers’ make better decisions? BBC .

Lyu, Lixing, Wang Chi Cheung. 2023. Bandits with knapsacks: advice on time-varying demands. International Conference on Machine Learning. PMLR, 23212–23238.

Ma, Tianyi, Hanzhang Qin, Ruihao Zhu, Jierui Zuo. 2026. Best-arm identification with generative proxy. arXiv preprint arXiv:2607.06879 .

Miao, Sentao, Yining Wang. 2024. Demand balancing in primal-dual optimization for blind network revenue management. arXiv preprint arXiv:2404.04467 .

Michel, Thomas, Hossein Hajiabolhassan, Ronald Ortner. 2023. Regret bounds for satisficing in multi-armed bandit problems. Transactions on Machine Learning Research .

Mukherjee, Subhojyoti, Kolar Purushothama Naveen, Nandan Sudarsanam, Balaraman Ravindran. 2017. Thresholding bandits with augmented ucb. arXiv preprint arXiv:1704.02281 .

Oracle. 2026. Oracle inventory optimization user’s guide URL https://docs.oracle.com/cd/E26401\_01/ doc.122/e48774/T473201T473236.htm.

Reverdy, Paul, Naomi E Leonard. 2014. Satisficing in gaussian bandit problems. 53rd IEEE Conference on Decision and Control. IEEE, 5718–5723.

Reverdy, Paul, Vaibhav Srivastava, Naomi Ehrich Leonard. 2017. Satisficing in multi-armed bandit problems. IEEE Transactions on Automatic Control 62(8) 3788–3803.

Rouyer, Chlo´e, Ronald Ortner, Peter Auer. 2024. Understanding the gaps in satisficing bandits. Forty-third International Conference on Machine Learning.

Russo, Daniel, Benjamin Van Roy. 2022. Satisficing in time-sensitive bandit learning. Mathematics of Operations Research 47(4) 2815–2839.

Sankararaman, Karthik Abinav, Aleksandrs Slivkins. 2021. Bandits with knapsacks beyond the worst case. Advances in Neural Information Processing Systems 34 23191–23204.

Simon, Herbert A. 1956. Rational choice and the structure of the environment. Psychological Review .

Tamatsukuri, Akihiro, Tatsuji Takahashi. 2019. Guaranteed satisficing and finite regret: Analysis of a cognitive satisficing value function. Biosystems 180 46–53.

Tran-Thanh, Long, Archie Chapman, Alex Rogers, Nicholas Jennings. 2012. Knapsack based optimal policies for budget–limited multi–armed bandits. Proceedings of the AAAI Conference on Artificial Intelligence, vol. 26. 1134–1140.

Wang, Zizhuo, Shiming Deng, Yinyu Ye. 2014. Close the gaps: A learning-while-doing algorithm for singleproduct revenue management problems. Operations Research 62(2) 318–331.

Zhang, Yixuan, Ruihao Zhu, Qiaomin Xie. 2026. On the peril of (even a little) nonstationarity in satisficing regret minimization. arXiv preprint arXiv:2603.18514 .

## Appendix

## A. Proof of Theorem 1

In the Appendix, for notational simplicity we denote $Z ^ { * } ( T , B )$ as $Z ^ { \ast }$ and denote $Z ^ { * } ( T , B ^ { \prime } )$ as $Z ^ { \prime }$

We first analyze the expected satisficing regret of this execution, assuming $\Delta _ { S } ^ { \prime } = Z ^ { \prime } - S >$ 0 and conditioned on $\mathcal { E } .$ . If T is suficiently large with $\sqrt { T / \log ( 2 d T ^ { 2 } ) } \geq 2 0 \alpha / ( b ( 1 - \alpha ) )$ , term $1 1 \sqrt { 2 } T / ( \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } )$ can be upper bounded by

$$
\begin{array} { r l } & { \frac { 1 1 \sqrt 2 T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } = 1 1 \sqrt 2 \cdot \bigg ( \underset { j } { \operatorname* { m i n } } \frac { B _ { j } } { T } - \frac { 1 0 \alpha } { 1 - \alpha } \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { T } } \bigg ) ^ { - 1 } } \\ & { \qquad \leq 1 1 \sqrt 2 \cdot ( b - b / 2 ) ^ { - 1 } = \frac { 2 2 \sqrt 2 } { b } . } \end{array}\tag{7}
$$

Denote $\beta ^ { \prime } = \operatorname* { m a x } \{ \beta , 1 / 2 \}$ and $\begin{array} { r } { K _ { \alpha , \beta } = 6 4 \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 2 2 \sqrt { 2 } } { b } \right) \geq 6 4 \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) } \end{array}$ Round $i _ { 0 }$ can be expressed as

$$
i _ { 0 } = \operatorname* { m a x } \left\{ i : 6 4 \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \beta ^ { \prime } } > \Delta _ { S } ^ { \prime } \right\} .
$$

Next, we prove an upper bound on $i _ { 0 }$ . Denote $\delta _ { 0 } \in ( 0 , 1 )$ the smallest solution to the following equation

$$
\delta _ { 0 } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } = \frac { \Delta _ { S } ^ { \prime } } { 6 4 \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) } .\tag{8}
$$

Note that $\delta _ { 0 } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } }$ is continuous on (0, 1). When $\delta _ { 0 } \to 0 , \delta _ { 0 } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } \to 0$ , and its maximum is $\begin{array} { r } { \operatorname* { m a x } _ { \delta _ { 0 } } \delta _ { 0 } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } \geq \delta _ { 0 } \log ( 2 / \delta _ { 0 } ) ^ { \beta ^ { \prime } } \big | _ { \delta _ { 0 } = 2 / e } > 1 / 2 } \end{array}$ . On the other hand, since 64 $\begin{array} { r } { \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \geq 6 4 \cdot 2 = 1 2 8 } \end{array}$ with $1 / 2 \le \alpha < 1$ , the right hand side of (8) is upper bounded by $\Delta _ { S } ^ { * } / 1 2 8 \leq 1 / 1 2 8$ with $\Delta _ { S } ^ { * } \le 1$ . Therefore there always exists $\delta _ { 0 } \in ( 0 , 1 )$ such that (8) holds. Based on (8), we also have

$$
\begin{array} { r l } & { \frac { d } { \delta _ { 0 } } = \frac { d \cdot 6 4 \left( \frac { 2 \zeta _ { 1 } + 2 } { ( 1 - \alpha ) ^ { 6 } w ^ { 6 } ( \delta , 1 / \beta _ { 0 } ^ { 2 } ) } + \frac { 1 1 \sqrt { 2 } T } { m _ { 0 } \beta _ { 0 } ^ { 2 } } \right) \log ( 4 d / \delta _ { 0 } ) ^ { \prime \prime } } { \Delta _ { s } ^ { \prime } } } \\ & { \quad \le \frac { d K _ { \alpha , \beta } \log ( 4 d / \delta _ { 0 } ) ^ { \prime \prime } } { \Delta _ { s } ^ { \prime } } } \\ & { \quad = \frac { ( 4 \beta ^ { \prime } ) ^ { \gamma \beta ^ { \prime } } d K _ { \alpha , \beta } \log \left( ( 4 d / \delta _ { 0 } ) ^ { 1 / ( 4 \beta ^ { \prime } ) } \right) ^ { \sigma ^ { \prime } } } { \Delta _ { s } ^ { \prime } } } \\ & { \quad \le \frac { ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } d K _ { \alpha , \beta } ( 4 d / \delta _ { 0 } ) ^ { 1 / 4 } } { \Delta _ { s } ^ { \prime } } } \\ & { \quad = \frac { \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \gamma ^ { \prime } } K _ { \alpha , \beta } } { \Delta _ { s } ^ { \prime } } \cdot \left( \frac { d } { \delta _ { 0 } } \right) ^ { 1 / 4 } , } \end{array}
$$

where the first inequality is based on the definition of $K _ { \alpha , \beta }$ and (7), the second inequality because $\log ( x ) < x$ for all $x > 0$ . Therefore, we get

$$
\frac { d } { \delta _ { 0 } } \leq \left( \frac { \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } } { \Delta _ { S } ^ { \prime } } \right) ^ { 4 / 3 } ,
$$

namely,

$$
\frac { 4 d } { \delta _ { 0 } } \leq 4 \left( \frac { \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } } { \Delta _ { S } ^ { \prime } } \right) ^ { 4 / 3 } \leq \left( \frac { 4 \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } } { \Delta _ { S } ^ { \prime } } \right) ^ { 4 / 3 } .\tag{9}
$$

Plugging (9) into the $\log ( 4 d / \delta _ { 0 } )$ term in (8), we have

$$
\frac { d } { \delta _ { 0 } } \leq \frac { d K _ { \alpha , \beta } } { \Delta _ { S } ^ { \prime } } \cdot ( 4 / 3 ) ^ { \beta ^ { \prime } } \log \left( \frac { 4 \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } } { \Delta _ { S } ^ { \prime } } \right) ^ { \beta ^ { \prime } }
$$

We write $D _ { \alpha , \beta } = 4 \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } , L _ { \alpha , \beta } = ( 4 / 3 ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta }$ , then we have

$$
\delta _ { 0 } \geq \frac { \Delta _ { S } ^ { \prime } } { L _ { \alpha , \beta } \log ( D _ { \alpha , \beta } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } .
$$

By definition of $i _ { 0 }$ we have that $i _ { 0 }$ is the largest integer that satisfies $\gamma _ { i _ { 0 } } \geq \delta _ { 0 }$ . Therefore we get

$$
\gamma _ { i _ { 0 } } \geq \frac { \Delta _ { S } ^ { \prime } } { L _ { \alpha , \beta } \log ( D _ { \alpha , \beta } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } ,
$$

thus

$$
i _ { 0 } \leq \frac { \alpha } { 1 - \alpha } \log _ { 2 } \left( \frac { L _ { \alpha , \beta } \log ( D _ { \alpha , \beta } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } { \Delta _ { S } ^ { \prime } } \right) ,
$$

or equivalently

$$
2 ^ { i _ { 0 } } \leq \left( \frac { L _ { \alpha , \beta } \log ( D _ { \alpha , \beta } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } { \Delta _ { S } ^ { \prime } } \right) ^ { \alpha / ( 1 - \alpha ) } .\tag{10}
$$

Using the result of Proposition 3, we have that the satisficing regret by round $i _ { 0 }$ is upper bounded

by

$$
\begin{array} { r l } & { \frac { \hbar } { \underset { \mathrm { L } = 1 } { \overset { \mathrm { S u p } } { \sum } } } \left( 3 \left( \frac { 2 C _ { 1 } ^ { \prime } } { \left( 1 - \alpha \right) ^ { 3 / 2 } \alpha ^ { 5 / 2 } } \right) ^ { \frac { 1 } { \alpha } } \frac { 1 ^ { \gamma } \sqrt { 2 } } { \operatorname* { m i n } _ { 1 } B _ { 2 } ^ { \prime } \frac { \gamma ^ { 2 } } { 2 } } \right) + 2 \right) \frac { \gamma ^ { \mathrm { L } ( 1 - \alpha ) } } { \hbar } \log ( 4 4 \tilde { \varepsilon } _ { \mathrm { L } } ^ { \prime } \pi ) ^ { \mathrm { m a x } \{ 1 / 2 \} } } \\ & { \le \frac { \hbar } { \underset { \mathrm { L } = 1 } { \overset { \mathrm { S u p } } { \sum } } } \left( 3 \left( \frac { 2 C _ { 1 } ^ { \prime } + 2 } { \left( 1 - \alpha \right) ^ { 3 / 2 } \beta ^ { 3 / 2 } } + \frac { 1 1 \sqrt { 2 } \Gamma } { \operatorname* { m i n } _ { 1 } B _ { 2 } ^ { \prime } \frac { \gamma ^ { 2 } } { 2 } } \right) + 2 \right) \frac { \gamma ^ { \mathrm { L } ( 1 - \alpha ) } } { \hbar ^ { 2 } } \log ( 4 4 \tilde { \varepsilon } _ { \mathrm { L } } ^ { \prime } \delta _ { 1 } ) ^ { \mathrm { m a x } \{ 1 / 2 \} } } \\ & { \le \frac { \hbar } { \alpha } \left( \frac { 3 } { 6 4 } K _ { \mathrm { L } , \alpha } + 1 2 \right) 2 ^ { \gamma \beta ^ { \mathrm { L } } } \log ( 4 4 \tilde { \varepsilon } _ { \mathrm { L } } \pi ) ^ { \mathrm { s } } } \\ & { \le \left( \frac { 3 } { 3 2 } K _ { \mathrm { L } } \varepsilon _ { \mathrm { L } } + 4 \right) 2 ^ { 2 \alpha / \alpha } \log ( 4 4 \tilde { \varepsilon } _ { \mathrm { L } } \pi ) ^ { \mathrm { s } } } \\ &  \le \left( \frac { 4 } { 3 2 } \right) ^ { \alpha } \left(  \end{array}\tag{11}
$$

where the first inequality is because by definition of $\delta _ { 0 }$ and $i _ { 0 }$ , we have $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha } \geq \gamma _ { i _ { 0 } }$ for all $i \leq i _ { 0 }$ , the second inequality is by plugging into the definition of $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha }$ and definition of $K _ { \alpha , \beta } ,$ , the third inequality is based on $\textstyle \sum _ { i = 1 } ^ { i _ { 0 } } 2 ^ { i / \alpha } \leq 2 \cdot 2 ^ { i _ { 0 } / \alpha }$ , and the fourth inequality is by plugging into (9) and (10).

By the results of Proposition 4, for every $k \geq 1$ , the probability that round $i _ { 0 } + k$ starts within the time horizon is at most $8 ^ { - k + 1 }$ . Therefore the regret incurred after round $i _ { 0 }$ is upper bounded by

$$
\begin{array} { r l } & { \quad \sum _ { k = 1 } ^ { \infty } \bigg ( 3 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { 3 / 2 } } + \frac { 1 1 \sqrt { 2 } I } { \sin \alpha } \right) + 2 \bigg ) \gamma _ { \frac { \gamma \delta } { 2 \alpha + \delta } } \log ( 4 I \gamma _ { \pi _ { 0 } + \pi } ) ^ { \theta ^ { \prime } } \cdot \mathbb { S } ^ { - \kappa - 1 } } \\ & { \quad \le \sum _ { k = 1 } ^ { \infty } \bigg ( \frac { 3 } { 6 4 } K _ { \pi _ { 0 } , \delta } + 2 \bigg ) 2 ^ { ( k ) \kappa / \alpha } \left( \log ( 4 I \hat { \sigma } _ { \pi _ { 0 } } ) + k \right) ^ { \theta ^ { \prime } } \cdot \mathbb { S } ^ { - k + 1 } } \\ & { \quad \le \displaystyle \sum _ { k = 1 } ^ { \infty } 2 ^ { \theta ^ { \prime } } \bigg ( \frac { 3 } { 6 4 } K _ { \pi _ { 0 } , \delta } + 2 \bigg ) 2 ^ { ( k ) \kappa + 1 / \alpha } \log ( 4 I \gamma _ { \pi _ { 0 } } ) ^ { \theta ^ { \prime } } \cdot \mathbb { S } ^ { - k } \cdot \mathbb { S } ^ { - k + 1 } } \\ & { \quad \le 2 ^ { \theta ^ { \prime } } \bigg ( \frac { 3 } { 6 4 } K _ { \pi _ { 0 } , \delta } + 1 6 \bigg ) 2 ^ { \theta \eta \kappa } \log ( 4 I \hat { \sigma } _ { \pi } ) ^ { \theta ^ { \prime } } \cdot M _ { \theta ^ { \prime } } } \\ &  \quad \le ( 8 \cdot 3 ) ^ { \theta ^ { \prime } } \bigg ( \frac { 3 } { 6 4 } K _ { \pi _ { 0 } , \delta } + 1 6 \bigg ) \left( \frac { I _ { \pi _ { 0 } , \delta } \log ( 1 0 \sigma _ { \pi _ { 0 } , \delta } ) ^ { \theta ^ { \prime } } } { \Delta _ { \pi _ { 0 } } ^ { \theta ^ { \prime } } } \right) ^ { \eta ( 1 - \alpha ) } \log \bigg ( \frac  D _  \pi _ { 0 } , \end{array}\tag{12}
$$

where $\begin{array} { r } { M _ { \beta ^ { \prime } } = \sum _ { k = 1 } ^ { \infty } k ^ { \beta ^ { \prime } } \cdot 2 ^ { - k } } \end{array}$ . The first inequality is by plugging into the definition of $K _ { \alpha , \beta }$ and $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha }$ , the second inequality is because log $( 4 d / \gamma _ { i _ { 0 } } ) > 1$ and $( 1 - \alpha ) / \alpha \leq 1$ for $1 / 2 \le \alpha < 1$ 2 and $\log ( 4 d / \gamma _ { i _ { 0 } } ) + k \leq 2 k \log ( 4 d / \gamma _ { i _ { 0 } } )$ for $k \geq 1$ , and for any $a , b \geq 1$ , we have $2 a b \geq a b + 1 \geq a + b .$ , the third inequality is based on $\begin{array} { r } { \sum _ { k = 1 } ^ { \infty } 2 ^ { k / \alpha } \cdot 8 ^ { - k + 1 } \leq \sum _ { k = 1 } ^ { \infty } 4 ^ { k } \cdot 8 ^ { - k + 1 } = 8 \sum _ { k = 1 } ^ { \infty } 2 ^ { - k } } \end{array}$ . Therefore, combining (11) and (12), the total expected satisficing regret conditioned on $\mathcal { E }$ is upper bounded by

$$
\left( \frac { 1 5 \cdot ( 8 / 3 ) ^ { \beta ^ { \prime } } } { 3 2 } K _ { \alpha , \beta } + 2 0 \cdot ( 8 / 3 ) ^ { \beta ^ { \prime } } \right) \left( \frac { L _ { \alpha , \beta } } { \Delta _ { s } ^ { \prime } } \right) ^ { 1 / ( 1 - \alpha ) } \log \left( \frac { D _ { \alpha , \beta } } { \Delta _ { s } ^ { \prime } } \right) ^ { \beta ^ { \prime } / ( 1 - \alpha ) + \beta ^ { \prime } } ( M _ { \beta ^ { \prime } } + 1 ) .
$$

In order to bound $\Delta _ { S } ^ { \prime }$ , we consider two cases, whether $\Delta _ { S } ^ { \prime } \geq \delta ( T )$ , where $\delta ( T ) = 1 0 \alpha / ( b - b \alpha )$ $\sqrt { \log ( 2 d T ^ { 2 } ) / T }$ is the bound on $( Z ^ { * } - Z ^ { \prime } )$ given in Proposition 5.

If $\Delta _ { S } ^ { \prime } \geq \delta ( T )$ , recall $\Delta _ { S } ^ { * } = Z ^ { * } - S$ and $\Delta _ { S } ^ { \prime } = Z ^ { \prime } - S$ , thus Proposition 5 implies $\Delta _ { S } ^ { * } - \Delta _ { S } ^ { \prime } \leq \delta ( T )$ Then we get

$$
\Delta _ { S } ^ { \prime } \geq \frac { \delta ( T ) } { 2 } + \frac { \Delta _ { S } ^ { * } - \delta ( T ) } { 2 } = \frac { \Delta _ { S } ^ { * } } { 2 } .
$$

Combining this with $P ( \mathcal { E } ^ { c } ) \leq 1 / T$ according to Proposition 2 and total satisficing regret incurred conditioned on negation of E is at most $T ,$ , total expected satisficing regret is upper bounded by

$$
1 + \left( \frac { 1 5 \cdot ( 8 / 3 ) ^ { \beta ^ { \prime } } } { 3 2 } K _ { \alpha , \beta } + 2 0 \cdot ( 8 / 3 ) ^ { \beta ^ { \prime } } \right) \left( \frac { 2 L _ { \alpha , \beta } } { \Delta _ { s } ^ { * } } \right) ^ { 1 / ( 1 - \alpha ) } \log \left( \frac { 2 D _ { \alpha , \beta } } { \Delta _ { s } ^ { * } } \right) ^ { \beta ^ { \prime } / ( 1 - \alpha ) + \beta ^ { \prime } } ( M _ { \beta ^ { \prime } } + 1 ) .\tag{13}
$$

If $\Delta _ { S } ^ { \prime } < \delta ( T )$ , namely $T$ is not large enough, we simply bound the satisficing regret with T. As $\Delta _ { S } ^ { * } - \Delta _ { S } ^ { \prime } \leq \delta ( T )$ , we have $\Delta _ { S } ^ { * } \leq 2 \delta ( T )$ , namely

$$
T \leq { \frac { 4 0 0 \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } } \log ( 2 d T ^ { 2 } ) .
$$

Let $q = 4 0 0 \alpha ^ { 2 } b ^ { - 2 } ( 1 - \alpha ) ^ { - 2 } { \Delta _ { S } ^ { * } } ^ { - 2 }$ and $y = T / ( 2 q )$ . So, we have

$$
y \leq \log \left( 2 q { \sqrt { 2 d } } y \right) = \log \left( 2 q { \sqrt { 2 d } } \right) + \log ( y ) \leq \log \left( 2 q { \sqrt { 2 d } } \right) + y / 2 ,
$$

where $\log ( y ) \leq y / 2$ is used in the second inequality. This implies $y \leq 2 \log \left( 2 q { \sqrt { 2 d } } \right)$ . As $\alpha \ge 1 / 2$ and $\Delta _ { S } ^ { * } \le 1$ , the total expected satisficing regret is upper bounded by

$$
\frac { 1 6 0 0 0 \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } } \cdot \frac { 1 } { \Delta _ { S } ^ { * 2 } } \log \left( \frac { 8 0 0 \sqrt { 2 d } \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } \right) \leq \frac { 3 2 0 0 \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } } \cdot \left( \frac { 1 } { \Delta _ { S } ^ { * } } \right) ^ { 1 / ( 1 - \alpha ) } \log \left( \frac { 4 0 d ^ { 1 / 4 } \alpha } { b ( 1 - \alpha ) \Delta _ { S } ^ { * } } \right) .\tag{14}
$$

## A.1. High Probability Event of Oracle Running

As $\hat { \pi } _ { i }$ is based on the running of $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , we first analyze the high probability events in the trajectory of $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , which will be used in future proof.

Denoting $C _ { j } ( { \tt A L G } , t _ { i } )$ as the realized cumulative consumption in resource $j$ of executing $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , the high probability event of $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ can be expressed as

Lemma 2. The expected cost of distribution πˆ<sub>i</sub> for resource $j$ in round i can be upper bounded by high probability as

$$
P r \left( c _ { j } ( \hat { \pi } _ { i } ) \leq \frac { B _ { j } ^ { \prime } } { T } + \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { t _ { i } } } \right) \geq 1 - \frac { 1 } { 2 d T ^ { 2 } } ,\tag{15}
$$

$$
P r \left( \forall k \geq 0 : c _ { j } ( \hat { \pi } _ { i } ) \leq \frac { B _ { j } ^ { \prime } } { T } - 2 \sqrt { \frac { 4 \log ( d ( T _ { i } + k ) ) } { T _ { i } + k } } \right) \geq 1 - \frac { 1 } { 3 2 d } .\tag{16}
$$

Proof. We note that the diference between realized resource consumption $C _ { j } ( \mathtt { A L G } , k )$ and expected consumption of the realized trajectory $\textstyle \sum _ { s = 1 } ^ { k } c _ { j } ( \pi _ { s } )$ is a martingale because

$$
\begin{array} { r l } & { \mathbb { E } \left[ C _ { j } ( \mathtt { A L G } , k + 1 ) - \displaystyle \sum _ { s = 1 } ^ { k + 1 } c _ { j } ( \pi _ { s } ) \Bigg | C _ { j } ( \mathtt { A L G } , k ) - \displaystyle \sum _ { s = 1 } ^ { k } c _ { j } ( \pi _ { s } ) \right] } \\ & { = C _ { j } ( \mathtt { A L G } , k ) - \displaystyle \sum _ { s = 1 } ^ { k } c _ { j } ( \pi _ { s } ) + \mathbb { E } \left[ ( C _ { j } ( \mathtt { A L G } , k + 1 ) - C _ { j } ( \mathtt { A L G } , k ) ) - c _ { j } ( \pi _ { k + 1 } ) \right] } \\ & { = C _ { j } ( \mathtt { A L G } , k ) - \displaystyle \sum _ { s = 1 } ^ { k } c _ { j } ( \pi _ { s } ) } \end{array}
$$

where the final equality is because the realized consumption in resource $j$ at step $k + 1$ is $( C _ { j } ( \mathtt { A L G } , k +$ $1 ) - C _ { j } ( \mathtt { A L G } , k ) )$ with expectation $c _ { j } ( \pi _ { k + 1 } )$ . Conditioned on the preceding history, each martingale

increment belongs to an interval of length 1, because the cost belongs to [0, 1]. The Hoefding–Azuma inequality therefore gives

$$
\operatorname* { P r } \left( C _ { j } ( \mathrm { A L G } , k ) - \sum _ { s = 1 } ^ { k } c _ { j } ( \pi _ { s } ) < - \epsilon \right) \leq \exp ( - \frac { 2 \epsilon ^ { 2 } } { k } ) .\tag{17}
$$

In order to analyze the high probability events for $c _ { j } \big ( \hat { \pi } _ { i } \big )$ , we bound the high probability events for realized resource consumption, with confidence interval $\sqrt { t _ { i } \log ( 2 d T ^ { 2 } ) }$ and $\sqrt { 2 t _ { i } \log ( 3 2 d ) }$ respectively.

For the high probability event with radius $\sqrt { t _ { i } \log ( 2 d T ^ { 2 } ) }$ , the diference between realized resource consumption and the expected consumption of the trajectory can be upper bounded by Azuma’s Inequality (17) as

$$
\operatorname* { P r } \left( \frac { C _ { j } \mathopen { } \mathclose \bgroup \left( \mathtt { A L G } , t _ { i } \aftergroup \egroup \right) } { t _ { i } } < \frac { 1 } { t _ { i } } \sum _ { s = 1 } ^ { t _ { i } } c _ { j } \mathopen { } \mathclose \bgroup \left( \pi _ { s } \aftergroup \egroup \right) - \sqrt { \frac { \log \mathopen { } \mathclose \bgroup \left( 2 d T ^ { 2 } \aftergroup \egroup \right) } { t _ { i } } } \right) \leq \frac { 1 } { 2 d T ^ { 2 } } .\tag{18}
$$

Recall $\hat { \pi } _ { i }$ is derived from the trajectory of running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , then $\begin{array} { r } { C _ { j } ( \mathtt { A L G } , t _ { i } ) - \sum _ { s = 1 } ^ { t _ { i } } c _ { j } ( \pi _ { s } ) > } \end{array}$ $- \sqrt { t _ { i } \log ( 2 d T ^ { 2 } ) }$ with $0 \leq w _ { i } \leq 1$ implies that with any nonnegative $w _ { i }$ , we have that

$$
\begin{array} { r l } & { \quad c _ { j } ( \hat { \pi } _ { i } ) = \displaystyle \frac { 1 - w _ { i } } { t _ { i } } \sum _ { s = 1 } ^ { t _ { i } } c _ { j } ( \pi _ { s } ) } \\ & { \leq \displaystyle \frac { C _ { j } \big ( \mathrm { A L G } , t _ { i } \big ) } { t _ { i } } + \sqrt { \frac { \log \left( 2 d T ^ { 2 } \right) } { t _ { i } } } } \\ & { \leq \displaystyle \frac { B _ { j } ^ { \prime } } { T } + \sqrt { \frac { \log \left( 2 d T ^ { 2 } \right) } { t _ { i } } } , } \end{array}
$$

happens with probability higher than $1 - 1 / ( 2 d T ^ { 2 } )$ , where the first inequality is due to $w _ { i } \geq 0$ and the complement of (18), and the second inequality is because realized cost $C _ { j } ( \mathtt { A L G } , t _ { i } )$ is constrained by allocated resource budget $b _ { j i } = B _ { j } ^ { \prime } t _ { i } / T$ , and then high probability event (15) is proved.

For the high probability event with radius $\sqrt { 2 t _ { i } \log ( 3 2 d ) }$ , the diference between realized resource consumption and the expected consumption of the trajectory can be upper bounded by Azuma’s Inequality (17) as

$$
\operatorname* { P r } \left( \frac { C _ { j } ( \mathtt { A L G } , t _ { i } ) } { t _ { i } } < \frac { 1 } { t _ { i } } \sum _ { s = 1 } ^ { t _ { i } } c _ { j } ( \pi _ { s } ) - \sqrt { \frac { 2 \log ( 3 2 d ) } { t _ { i } } } \right) \le \frac { 1 } { 3 2 d } .\tag{19}
$$

Recall $\hat { \pi } _ { i }$ is derived from the trajectory of running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , then $\begin{array} { r } { C _ { j } ( \mathtt { A L G } , t _ { i } ) - \sum _ { s = 1 } ^ { t _ { i } } c _ { j } ( \pi _ { s } ) > } \end{array}$ $- \sqrt { 2 t _ { i } \log ( 3 2 d ) }$ implies that

$$
\begin{array} { r l } & { \quad c _ { j } ( \hat { \pi } _ { i } ) = \cfrac { 1 - w _ { i } } { t _ { i } } \cfrac { t _ { i } } { s _ { \mathrm { s } } - 1 } } \\ & { \leq ( 1 - w _ { i } ) \cdot \left( \cfrac { C _ { j } ( \Delta \mathrm { L G } , t _ { i } ) } { t _ { i } } + \sqrt { \cfrac { 2 \log ( 3 2 d ) } { t _ { i } } } \right) } \\ & { \leq \cfrac { B _ { j } ^ { \prime } } { T } - w _ { i } \cfrac { B _ { j } ^ { \prime } } { T } + \sqrt { \cfrac { 2 \log ( 3 2 d ) } { t _ { i } } } } \\ & { \leq \cfrac { B _ { j } ^ { \prime } } { T } - 2 \sqrt { \cfrac { 4 \log ( d ( T _ { i } + k ) ) } { T _ { i } + k } } , \quad \forall k \geq 0 , } \end{array}
$$

happens with probability higher than $1 - 1 / ( 3 2 d )$ , where the first inequality is due to the complement of (19), and the second inequality is because realized cost $C _ { j } ( { \tt A L G } , t _ { i } )$ is constrained by allocated resource budget $b _ { j i } = B _ { j } ^ { \prime } t _ { i } / T$ , i.e., $C _ { j } ( \mathtt { A L G } , t _ { i } ) / t _ { i } \le B _ { j } ^ { \prime } / T$ , and $w _ { i } > 0$ . The third inequality is based on $2 \leq T _ { i } \leq t _ { i }$ and the definition of $w _ { i } = 1 0 T / \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } \cdot \sqrt { \log ( 8 d T _ { i } ) / T _ { i } }$ when $w _ { i } < 1$ where

$$
\begin{array} { r l } & { \quad 2 \sqrt { \frac { 4 \log ( d ( T _ { i } + k ) ) } { T _ { i } + k } } + \sqrt { \frac { 2 \log ( 3 2 d ) } { t _ { i } } } } \\ & { \quad \le 8 \sqrt { \frac { \log ( d T _ { i } ) } { T _ { i } } } + 2 \sqrt { \frac { \log ( 8 d ) } { t _ { i } } } } \\ & { \quad \le 8 \sqrt { \frac { \log ( 8 d T _ { i } ) } { T _ { i } } } + 2 \sqrt { \frac { \log ( 8 d T _ { i } ) } { T _ { i } } } } \\ & { \quad = \frac { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } { T } . w _ { i } , \quad \forall k \ge 0 . } \end{array}
$$

The first inequality above is because $\log ( x ) / x$ is decreasing for $x \ge 3$ and $2 \log ( 2 ) / 2 \geq \log ( 3 ) / 3$ and the second inequality is because $T _ { i } \geq 1$ and $t _ { i } = \lceil 1 / \gamma _ { i } ^ { 1 / ( 1 - \alpha ) } \rceil \geq \lceil 1 / \gamma _ { i } ^ { 2 } \rceil = T _ { i }$ for $\gamma _ { i } < 1$ and $1 / 2 \le \alpha < 1$ . Then the high probability event (16) is proved. □

For notation simplicity in future proof, we denote

$$
\begin{array} { r l } & { \mathcal { C } _ { j i } ^ { 1 } : = \left\{ c _ { j } \big ( \hat { \pi } _ { i } \big ) \leq \displaystyle \frac { B _ { j } ^ { \prime } } { T } + \sqrt { \frac { \log \left( 2 d T ^ { 2 } \right) } { t _ { i } } } \right\} } \\ & { \mathcal { C } _ { j i } ^ { 2 } : = \left\{ \forall k \geq 0 : c _ { j } \big ( \hat { \pi } _ { i } \big ) \leq \displaystyle \frac { B _ { j } ^ { \prime } } { T } - 2 \sqrt { \frac { 4 \log \left( d \left( T _ { i } + k \right) \right) } { T _ { i } + k } } \right\} . } \end{array}
$$

Therefore Lemma 2 implies $\operatorname* { P r } \left( \mathcal { C } _ { j i } ^ { 1 c } \right) \leq 1 / ( 2 d T ^ { 2 } )$ and $\operatorname* { P r } \left( \mathcal { C } _ { j i } ^ { 2 c } \right) \leq 1 / ( 3 2 d )$

The expected regret of the candidate policy is bounded as follows.

Lemma 3. Under Condition 1, the expected regret of a single pull on distribution $\hat { \pi } _ { i }$ in round i can be upper bounded by

$$
\mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] \leq \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

Proof. Since $\hat { \boldsymbol { \pi } } _ { i }$ is calculated based on $\{ \tilde { \pi } _ { s } \} _ { s \in [ t _ { i } ] }$ in round i, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] } \\ & { = \displaystyle \frac { 1 } { t _ { i } } \mathbb { E } \left[ \operatorname* { m a x } \left\{ t _ { i } Z ^ { \prime } - ( \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) ) \cdot ( 1 - w _ { i } ) , 0 \right\} \right] } \\ & { \le \displaystyle \frac { 1 } { t _ { i } } \mathbb { E } \left[ \operatorname* { m a x } \left\{ t _ { i } Z ^ { \prime } - \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) , 0 \right\} \right] + \displaystyle \frac { w _ { i } } { t _ { i } } \mathbb { E } \left[ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) \right] } \\ & { \le \displaystyle \frac { 1 } { t _ { i } } \mathbb { E } \left[ \operatorname* { m a x } \left\{ t _ { i } Z ^ { \prime } - R , 0 \right\} \right] + w _ { i } } \end{array}
$$

where $\begin{array} { r } { R : = \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) } \end{array}$ . The first equality is based on the definition of $\hat { \pi } _ { i }$ as a weighted sum of trajectory average and null arm that has zero reward, and the second inequality is based on bounded reward $r \in [ 0 , 1 ]$ . By Lemma 1, the first term is bounded using Condition 1, and therefore

$$
\mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] \leq C _ { 1 } t _ { i } ^ { \alpha - 1 } \log ^ { \beta } ( t _ { i } ) + \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } t _ { i } ^ { - 1 / 2 } + w _ { i } .\tag{20}
$$

For $t _ { i } ^ { \alpha - 1 } \log ( t _ { i } ) ^ { \beta }$ , recall that $t _ { i } = \lceil \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \rceil \leq 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) }$ , then log $( t _ { i } ) \leq \log ( 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } ) \leq$ $\log ( 2 / \gamma _ { i } ) / ( 1 - \alpha )$ and $t _ { i } ^ { \alpha } = \lceil 2 ^ { i / \alpha } \rceil ^ { \alpha } \leq 2 ^ { \alpha ( 1 + i / \alpha ) } \leq 2 \cdot 2 ^ { i } = 2 \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) }$ . Similarly, $t _ { i } ^ { \alpha - 1 } \leq ( \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } ) ^ { \alpha - 1 } =$ $\gamma _ { i } \leq 2 \gamma _ { i } . \mathrm { S o }$ , it can be bounded by

$$
C _ { 1 } t _ { i } ^ { \alpha - 1 } \log ( t _ { i } ) ^ { \beta } \leq \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } \gamma _ { i } \log \left( \frac { 2 } { \gamma _ { i } } \right) ^ { \beta } \leq \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { \beta } .\tag{21}
$$

For $w _ { i } .$ as we assume $\alpha \ge 1 / 2$ , then $T _ { i } \leq t _ { i }$ . Recall $\gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } = 2 ^ { i }$ and $T _ { i } = \lceil 1 / \gamma _ { i } ^ { 2 } \rceil$ , we have $T _ { i } \geq \gamma _ { i } ^ { - 2 }$ and $T _ { i } \leq 2 \gamma _ { i } ^ { - 2 } . \ w _ { i }$ can be upper bounded with

$$
\sqrt { \frac { \log ( 8 d T _ { i } ) } { T _ { i } } } \leq \gamma _ { i } \sqrt { \log \left( \frac { 1 6 d } { \gamma _ { i } ^ { 2 } } \right) } \leq \sqrt { 2 } \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { 1 / 2 } .\tag{22}
$$

For the additional term, since $\alpha \ge 1 / 2$ and $\gamma _ { i } \leq 1$ 2

$$
\frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } t _ { i } ^ { - 1 / 2 } \le \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \gamma _ { i } ^ { 1 / ( 2 ( 1 - \alpha ) ) } \le \frac { \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { 1 / 2 } .\tag{23}
$$

Combining (21), (22), and (23) with (20), gives

$$
\mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] \leq \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

## A.2. Proof of Lemma 1

Proof. Recall that $b _ { j i } = B _ { j } ^ { \prime } t _ { i } / T$ . Since the resource budgets in round i are obtained by scaling $B ^ { \prime }$ by $t _ { i } / T$ , the LP relaxation corresponding to the $t _ { i }$ -step oracle run can be written as

$$
\begin{array} { r l } { t _ { i } Z ^ { \prime } = \underset { \pi \in \Delta ( \mathcal { X } ) } { \operatorname* { m a x } } } & { { } t _ { i } r \big ( \pi \big ) } \\ { \mathrm { s . t . } } & { { } t _ { i } c _ { j } \big ( \pi \big ) \le b _ { j i } , \qquad \forall j \in [ d ] . } \end{array}
$$

Its dual is

$$
\begin{array} { r l } { \displaystyle \underset { \lambda _ { 1 } , \ldots , \lambda _ { d } , \mu } { \operatorname* { m i n } } } & { \displaystyle t _ { i } \mu + \sum _ { j = 1 } ^ { d } \lambda _ { j } b _ { j i } } \\ { \mathrm { s . t . } } & { \displaystyle \mu + \sum _ { j = 1 } ^ { d } \lambda _ { j } c _ { j } ( X ) \geq r ( X ) , \qquad \forall X \in \mathcal { X } , } \\ & { \displaystyle \lambda _ { j } \geq 0 , \qquad \forall j \in [ d ] . } \end{array}
$$

Let $( \lambda ^ { * } , \mu ^ { * } )$ be an optimal dual solution. The positive budgets and the null policy provide strict feasibility. By strong duality,

$$
t _ { i } Z ^ { \prime } = t _ { i } \mu ^ { * } + \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } b _ { j i } .\tag{24}
$$

Then, we can bound

$$
t _ { i } \overset { \mathrm { ( i ) } } { \geq } t _ { i } Z ^ { \prime } = t _ { i } \mu ^ { * } + \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } b _ { j i } \overset { \mathrm { ( i i ) } } { \geq } \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } b _ { j i } \overset { \mathrm { ( i i i ) } } { \geq } \frac { t _ { i } \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } { T } \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } ,
$$

where step (i) is because $Z ^ { \prime } \leq 1$ , step (ii) is because the null arm has zero reward and zero resource consumption and then its dual constraint implies $\mu ^ { * } \geq 0$ , and step (iii) is based on $b _ { j i } = B _ { j } ^ { \prime } t _ { i } / T$ This further implies

$$
\| \lambda ^ { * } \| _ { 1 } = \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } \leq { \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } } .\tag{25}
$$

Since the dual constraint holds for every arm, it also holds for every distribution over arms. Therefore, for every policy $\tilde { \pi } _ { s }$ selected by the oracle,

$$
r ( \tilde { \pi } _ { s } ) \leq \mu ^ { * } + \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } c _ { j } ( \tilde { \pi } _ { s } ) .
$$

Summing this inequality over $s \in [ t _ { i } ]$ , we obtain

$$
\begin{array} { r l } { \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) - t _ { i } Z ^ { \prime } \le t _ { i } \mu ^ { * } + \displaystyle \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } \displaystyle \sum _ { s = 1 } ^ { t _ { i } } c _ { j } ( \tilde { \pi } _ { s } ) - \left( t _ { i } \mu ^ { * } + \displaystyle \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } b _ { j i } \right) } & { } \\ { = \displaystyle \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } \left( \displaystyle \sum _ { s = 1 } ^ { t _ { i } } c _ { j } ( \tilde { \pi } _ { s } ) - b _ { j i } \right) } & { } \\ { \displaystyle \le \sum _ { s = 1 } ^ { t _ { i } } \displaystyle \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } \left( c _ { j } ( \tilde { \pi } _ { s } ) - C _ { s } ^ { j } \right) , } \end{array}\tag{26}
$$

where the first inequality uses strong duality (24), and the final inequality is because the realized consumption does not exceed the oracle allocation $\textstyle \sum _ { s = 1 } ^ { t _ { i } } C _ { s } ^ { j } \leq b _ { j i }$ for every $j \in [ d ]$

Define $\begin{array} { r } { D _ { s } : = \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } \big ( c _ { j } \big ( \tilde { \pi } _ { s } \big ) - C _ { s } ^ { j } \big ) } \end{array}$ . Conditioned on the history immediately before drawing the arm at step s, including the oracle’s policy choice, $\tilde { \pi } _ { s }$ is fixed and $\mathbb { E } [ C _ { s } ^ { j } \mid \mathrm { h i s t o r y } ] = c _ { j } ( \tilde { \pi } _ { s } )$ , so $\mathbb { E } [ D _ { s } \mid \mathrm { h i s t o r y } ] = 0$ . Hence $\{ D _ { s } \} _ { s = 1 } ^ { t _ { i } }$ is a martingale diference sequence. Furthermore, since $c _ { j } \big ( \tilde { \pi } _ { s } \big )$ and $C _ { s } ^ { j }$ belong to [0, 1], (25) gives

$$
| D _ { s } | \leq \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } | c _ { j } ( \tilde { \pi } _ { s } ) - C _ { s } ^ { j } | \leq \sum _ { j = 1 } ^ { d } \lambda _ { j } ^ { * } = \| \pmb { \lambda } ^ { * } \| _ { 1 } \leq \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } .
$$

Therefore, using the martingale diference property to eliminate the cross terms,

$$
\begin{array} { r l r } {  { \mathbb { E } [ ( \sum _ { s = 1 } ^ { t _ { i } } D _ { s } ) ^ { 2 } ] = \sum _ { s = 1 } ^ { t _ { i } } \mathbb { E } [ D _ { s } ^ { 2 } ] + 2 \sum _ { 1 \leq s < u \leq t _ { i } } \mathbb { E } [ D _ { s } D _ { u } ] } } \\ & { } & { = \sum _ { s = 1 } ^ { t _ { i } } \mathbb { E } [ D _ { s } ^ { 2 } ] \leq t _ { i } ( \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } ) ^ { 2 } . } \end{array}\tag{27}
$$

Combining (26) and (27), followed by Cauchy–Schwarz, gives

$$
\begin{array} { r } { \mathbb { E } \left[ \operatorname* { m a x } \left\{ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) - t _ { i } Z ^ { \prime } , 0 \right\} \right] \leq \mathbb { E } \left[ \operatorname* { m a x } \left\{ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } D _ { s } , 0 \right\} \right] \leq \mathbb { E } \left[ \left\lfloor \displaystyle \sum _ { s = 1 } ^ { t _ { i } } D _ { s } \right\rceil \right] } \\ { \leq \sqrt { \mathbb { E } \left[ \left( \displaystyle \sum _ { s = 1 } ^ { t _ { i } } D _ { s } \right) ^ { 2 } \right] } \leq \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \sqrt { t _ { i } } . } \end{array}\tag{28}
$$

Finally, using max $\{ x , 0 \} = x + \operatorname* { m a x } \{ - x , 0 \}$ and Condition 1,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \operatorname* { m a x } \left\{ t _ { i } Z ^ { \prime } - \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) , 0 \right\} \right] = \mathbb { E } \left[ t _ { i } Z ^ { \prime } - \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) \right] + \mathbb { E } \left[ \operatorname* { m a x } \left\{ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) - t _ { i } Z ^ { \prime } , 0 \right\} \right] } \\ & { \qquad \leq C _ { 1 } t _ { i } ^ { \alpha } \log ^ { \beta } ( t _ { i } ) + \displaystyle \frac { T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \sqrt { t _ { i } } , } \end{array}
$$

which proves the lemma.

## A.3. Proof of Proposition 1

We first bound the number of rounds $i _ { \mathrm { m a x } }$ . Recall that $t _ { i } = \lceil \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \rceil$ and $T _ { i } = \lceil \gamma _ { i } ^ { - 2 } \rceil$ , since the length of the time horizon is $T$ time steps, we have $t _ { i } \leq T$ . By definition of $t _ { i } = \lceil 2 ^ { i / \alpha } \rceil$ , the number of rounds that end within the time horizon is bounded by $2 ^ { i / \alpha } \leq T , i . e . , i \leq \lfloor \alpha \log _ { 2 } ( T ) \rfloor$ because i is an integer. Then, the maximum number of rounds is bounded by $i _ { \mathrm { m a x } } = \lceil \alpha \log _ { 2 } ( T ) \rceil \leq 2 \alpha \log _ { 2 } ( T )$ for $T \geq 4$

Recall that $T _ { i } \leq \gamma _ { i } ^ { - 2 } + 1 \leq 2 ^ { 2 i ( 1 - \alpha ) / \alpha } + 1$ , then $\sqrt { T _ { i } } \leq 2 ^ { i ( 1 - \alpha ) / \alpha } + 1$ and $T _ { i _ { \operatorname* { m a x } } } \leq T$ . We can bound $\textstyle \sum _ { i } { \sqrt { T _ { i } } }$ as

$$
\begin{array} { r l } { \displaystyle \sum _ { i = 1 } ^ { i \operatorname* { m a x } } \sqrt { T _ { i } } \le \sum _ { i = 1 } ^ { i \operatorname* { m a x } } 2 ^ { i ( 1 - \alpha ) / \alpha } + i _ { \operatorname* { m a x } } } \\ & { \le \frac { 2 ^ { ( 1 - \alpha ) / \alpha } } { 2 ^ { ( 1 - \alpha ) / \alpha } - 1 } \cdot 2 ^ { i \operatorname* { m a x } ( 1 - \alpha ) / \alpha } + i _ { \operatorname* { m a x } } } \\ & { \le \frac { 2 } { 2 ^ { ( 1 - \alpha ) / \alpha } - 1 } \cdot \sqrt { T _ { i _ { \operatorname* { m a x } } } } + 2 \alpha \log _ { 2 } ( T ) } \\ & { \overset { ( i ) } { \le } \frac { 2 \alpha } { ( 1 - \alpha ) \ln ( 2 ) } \sqrt { T } + 4 \alpha \sqrt { T } } \\ & { \le \frac { 5 \alpha } { 1 - \alpha } \sqrt { T } } \end{array}
$$

where step (i) is based on $2 ^ { x } - 1 \geq x \ln ( 2 )$ for all $x > 0$ and $\log _ { 2 } ( T ) \leq 2 { \sqrt { T } }$ for all $T \geq 0$

Therefore the permissive accumulation of the excess portion of $\hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } )$ into $B _ { j } ^ { \prime } \tau _ { i } / T$ , conditioned on $\mathcal { E } ,$ is upper bounded by

$$
\sum _ { i } 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } \leq \frac { 1 0 \alpha } { 1 - \alpha } \sqrt { T \log ( 2 d T ^ { 2 } ) } .
$$

Recall that $\begin{array} { r } { b _ { j i } = B _ { j } ^ { \prime } t _ { i } / T , \sum _ { i } ( t _ { i } + \tau _ { i } ) = T } \end{array}$ . conditioned on $\mathcal { E } ,$ realized resource $j$ consumption is upper bounded by

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { i } ( b _ { j i } + \hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) ) } \\ & { \le \displaystyle \sum _ { i } \left( \frac { B _ { j } ^ { \prime } t _ { i } } { T } + \frac { B _ { j } ^ { \prime } \tau _ { i } } { T } + { \bf 1 } ( \tau _ { i } > 0 ) 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } \right) } \\ & { \le B _ { j } ^ { \prime } + \displaystyle \frac { 1 0 \alpha } { 1 - \alpha } \sqrt { T \log ( 2 d T ^ { 2 } ) } = B _ { j } . } \end{array}
$$

## A.4. Proof of Proposition 2

Lemma 4. Let $\tau _ { i }$ be the length of forced sampling and LCB-UCB test of round i, then for any $j \in [ d ]$

$$
\hat { c } _ { j i } ^ { t o t } ( \tau _ { i } ) - \frac { \tau _ { i } B _ { j } ^ { \prime } } { T } \leq \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { n \leq \operatorname* { m i n } \{ T _ { i } , \tau _ { i } \} } \left\{ \hat { c } _ { j i } ^ { t o t } ( n ) - \frac { n B _ { j } ^ { \prime } } { T } \right\} , 1 \right\} .
$$

Proof. If $\tau _ { i } = T _ { i }$ , then

$$
\hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) - \frac { \tau _ { i } B _ { j } ^ { \prime } } { T } = \hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) - \frac { T _ { i } B _ { j } ^ { \prime } } { T } \leq \operatorname* { m a x } \left\{ \hat { c } _ { j i } ^ { \mathrm { t o t } } ( T _ { i } ) - \frac { T _ { i } B _ { j } ^ { \prime } } { T } , 1 \right\} .
$$

Therefore it sufices to consider the case when $\tau _ { i } > T _ { i }$ . Assume $\tau _ { i } > T _ { i }$ , then the termination criterion of the LCB-UCB test is not satisfied at step $\tau _ { i } - 1$ , then we get

$$
U C B _ { j } ( \hat { \pi } _ { i } ) = \frac { \hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } - 1 ) } { \tau _ { i } - 1 } + \sqrt { \frac { 4 \log ( d ( \tau _ { i } - 1 ) ) } { \tau _ { i } - 1 } } \leq \frac { B _ { j } ^ { \prime } } { T } ,
$$

which implies that

$$
\hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } - 1 ) \leq \frac { B _ { j } ^ { \prime } ( \tau _ { i } - 1 ) } { T } .
$$

Since we assume that the realized resource consumption is at most 1, we get

$$
\hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) \leq \hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } - 1 ) + 1 \leq \frac { B _ { j } ^ { \prime } \tau _ { i } } { T } + 1 ,
$$

therefore in this case

$$
\hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) - \frac { \tau _ { i } B _ { j } ^ { \prime } } { T } \leq 1 \leq \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { n \leq \operatorname* { m i n } \{ T _ { i } , \tau _ { i } \} } \left\{ \hat { c } _ { j i } ^ { \mathrm { t o t } } ( n ) - \frac { n B _ { j } ^ { \prime } } { T } \right\} , 1 \right\} .
$$

We prove this proposition by proving $\operatorname* { P r } ( { \mathcal { E } } ^ { c } ) \leq 1 / T$ . Recall that

$$
\mathcal { C } _ { j i } ^ { 1 } = \left\{ c _ { j } ( \hat { \pi } _ { i } ) \leq \frac { B _ { j } ^ { \prime } } { T } + \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { t _ { i } } } \right\} .
$$

As $\mathcal { E } ^ { c }$ can be expressed as {∃j, ∃i, $\hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) > \tau _ { i } B _ { j } ^ { \prime } / T + 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } \}$ , its probability can be analyzed conditioned on whether $\mathcal { C } _ { j i } ^ { 1 }$ holds for each resource j round i.

When $\mathcal { C } _ { j i } ^ { 1 }$ does not hold, the probability that $\hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } )$ exceeds the allocated resource can be upper bounded by 1. And recall Equation (15) in Lemma 2 implies that $\mathcal { C } _ { j i } ^ { 1 c }$ happens with probability

$$
\mathrm { P r } ( \mathcal { C } _ { j i } ^ { 1 c } ) \leq \frac { 1 } { 2 d T ^ { 2 } } .
$$

When $\mathcal { C } _ { j i } ^ { 1 }$ holds, namely $c _ { j } \big ( \hat { \pi } _ { i } \big ) \leq B _ { j } ^ { \prime } / T + \sqrt { \log ( 2 d T ^ { 2 } ) / t _ { i } }$ , then during forced sampling the excess cost can be bounded by

$$
\begin{array} { r l } & { \quad \operatorname* { P r } \left( \underset { 1 \leq n \leq \operatorname* { m i n } \{ T _ { i } , \tau _ { i } \} } { \operatorname* { m a x } } \left. \hat { c } _ { j i } ^ { \mathrm { t o t } } ( n ) - \frac { n B _ { j } ^ { \prime } } { T } \right. \geq 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } \bigg | \mathcal { C } _ { j i } ^ { 1 } \right) } \\ & { \leq \underset { n = 1 } { \overset { T _ { i } } { \leq } } \operatorname* { P r } \left( \frac { \hat { c } _ { j i } ^ { \mathrm { t o t } } ( n ) } { n } \geq c _ { j } ( \hat { \pi } _ { i } ) - \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { t _ { i } } } + \frac { 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } } { n } \bigg | \mathcal { C } _ { j i } ^ { 1 } \right) } \\ & { \leq \underset { n = 1 } { \overset { T _ { i } } { \leq } } \operatorname* { P r } \left( \frac { \hat { c } _ { j i } ^ { \mathrm { t o t } } ( n ) } { n } \geq c _ { j } ( \hat { \pi } _ { i } ) + \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { n } } \bigg | \mathcal { C } _ { j i } ^ { 1 } \right) } \\ & { \leq { T _ { i } } \exp ( - 2 \log ( 2 d T ^ { 2 } ) ) \leq \frac { 1 } { 2 d T ^ { 2 } } . } \end{array}
$$

where the first inequality is based on the definition of $\mathcal { C } _ { j i } ^ { 1 }$ and a union bound over $n = 1 , \ldots , T _ { i }$ . The second inequality is because $n \leq T _ { i } \leq t _ { i }$ , which implies

$$
\frac { 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } } { n } - \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { t _ { i } } } \geq \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { n } } .
$$

For the third inequality, conditioned on the completed oracle trajectory, the forced-sampling observations are independent and bounded in [0, 1] with mean $c _ { j } \big ( \hat { \pi } _ { i } \big )$ , so Hoefding’s inequality

applies. Finally, $T _ { i } \le T$ implies $T _ { i } \exp ( - 2 \log ( 2 d T ^ { 2 } ) ) \leq 1 / ( 2 d T ^ { 2 } )$ . Combining this with Lemma 4, because $2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } > 1$ , we then know

$$
\operatorname* { P r } \left( \hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) > \frac { \tau _ { i } B _ { j } ^ { \prime } } { T } + 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } \left| \mathcal { C } _ { j i } ^ { 1 } \right. \right) \leq \frac { 1 } { 2 d T ^ { 2 } } .\tag{29}
$$

Therefore, the probability of E negation can be upper bounded by

$$
\begin{array} { l } { { \displaystyle \operatorname* { P r } \left( \exists j , \exists i , \hat { c } _ { j i } ^ { \mathrm { t o t } } ( \tau _ { i } ) > \tau _ { i } B _ { j } ^ { \prime } / T + 2 \sqrt { T _ { i } \log ( 2 d T ^ { 2 } ) } \right) } } \\ { { \displaystyle \leq \sum _ { j = 1 } ^ { d } \sum _ { i } ^ { i _ { \mathrm { m a x } } } ( \frac { 1 } { 2 d T ^ { 2 } } + \frac { 1 } { 2 d T ^ { 2 } } ) } } \\ { { \displaystyle \leq \frac { d T } { d T ^ { 2 } } = \frac { 1 } { T } } , } \end{array}
$$

where the first inequality is based on probability condition on $\mathcal { C } _ { j i } ^ { 1 c } \left( 1 5 \right)$ and $\mathcal { C } _ { j i } ^ { 1 } \ ( 2 9 )$ , and the second inequality is based on $T > i _ { \mathrm { m a x } }$ . Then this proves the proposition that $\operatorname* { P r } ( \mathcal { E } ) \geq 1 - 1 / T$

## A.5. Proof of Proposition 5

Given one of the optimal distribution under original resource constraint $B _ { j }$ as $\pi ^ { * }$ , we define distribution

$$
\pi ^ { \prime } = ( 1 - \delta ( T ) ) \pi ^ { * } + \delta ( T ) \cdot { \bf 1 } _ { 0 } ,
$$

where $\delta ( T ) : = 1 0 \alpha / ( b - b \alpha ) \cdot \sqrt { \log ( 2 d T ^ { 2 } ) / T }$ . Recall $B _ { j } ^ { \prime } = B _ { j } - 1 0 \alpha / ( 1 - \alpha ) \cdot \sqrt { T \log ( 2 d T ^ { 2 } ) }$ , for each resource $j$ we can bound

$$
\begin{array} { r l } & { c _ { j } ( \pi ^ { \prime } ) = ( 1 - \delta ( T ) ) c _ { j } ( \pi ^ { * } ) } \\ & { \qquad \leq \left( 1 - \frac { 1 0 \alpha } { b ( 1 - \alpha ) } \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { T } } \right) \cdot \frac { B _ { j } } { T } } \\ & { \qquad \leq \displaystyle \frac { B _ { j } } { T } - \frac { 1 0 \alpha } { 1 - \alpha } \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { T } } = \frac { B _ { j } ^ { \prime } } { T } , } \end{array}
$$

where the first inequality is based on the definition of $\pi ^ { * }$ introduced in LP relaxation (1) on the optimal dynamic policy. Therefore $\pi ^ { \prime }$ is a feasible distribution under resource constraints $B _ { j } ^ { \prime }$ . Namely, the optimal distribution reward under constraints $B _ { j } ^ { \prime }$ can be upper bounded by $Z ^ { \prime } \geq r ( \pi ^ { \prime } )$

The diference between $Z ^ { \ast }$ and $Z ^ { \prime }$ can be upper bounded by

$$
Z ^ { * } - Z ^ { \prime } \leq Z ^ { * } - r ( \pi ^ { \prime } ) = Z ^ { * } - ( 1 - \delta ( T ) ) Z ^ { * } \leq \delta ( T ) ,
$$

where the $Z ^ { * } = \mathbb { E } _ { X \sim \pi ^ { * } } [ r ( X ) ] \le 1$ is used in the final inequality.

## A.6. Proof of Proposition 3

Recall $\alpha \ge 1 / 2$ and $\gamma _ { i } \leq 1$ . Lemma 3 implies that the satisficing regret of a single pull on $\hat { \pi } _ { i }$ can be upper bounded by

$$
\begin{array} { r l } & { \mathbb { E } [ \operatorname* { m a x } \{ S - r ( \hat { \pi } _ { i } ) , 0 \} ] \leq \mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] } \\ & { \qquad \leq \left( \displaystyle \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \displaystyle \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } . } \end{array}\tag{30}
$$

The regret in round i can be divided into three parts. Running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , forced-sampling, and LCB-UCB test.

Recall that $t _ { i } = \lceil \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \rceil \leq 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) }$ as $\gamma _ { i } < 1$ , the satisficing regret incurred when running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ can simply be upper bounded by linear regret

$$
\mathbb { E } \left[ \sum _ { s = 1 } ^ { t _ { i } } \operatorname* { m a x } \{ S - r ( \tilde { \pi } _ { s } ) , 0 \} \right] \leq t _ { i } \leq 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \leq 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .\tag{31}
$$

The satisficing regret incurred when pulling $\hat { \pi } _ { i }$ in forced sampling for $T _ { i } = \lceil \gamma _ { i } ^ { - 2 } \rceil$ time steps is upper bounded by

$$
\begin{array} { r l } & { \quad \lceil \gamma _ { i } ^ { - 2 } \rceil \cdot \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \leq \frac { 2 } { \gamma _ { i } ^ { 2 } } \cdot \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \leq 2 \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } , } \end{array}\tag{32}
$$

where the first inequality is because $\lceil \gamma _ { i } ^ { - 2 } \rceil \leq 2 / \gamma _ { i } ^ { 2 }$ as $\gamma _ { i } < 1$ , and the second inequality is because $\alpha \ge 1 / 2$ and $\gamma _ { i } \leq 1$

If $r ( \hat { \pi } _ { i } ) < S$ (otherwise the satisficing regret is 0), for every $k ,$ round i is not terminated after pulling $\hat { \pi } _ { i }$ for k time steps in addition to first $T _ { i }$ time steps occurs with probability at most

$$
\begin{array} { r l } & { \operatorname* { P r } \left( \frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } \geq S \right) } \\ & { \leq \operatorname* { P r } \left( \frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } > r ( \hat { \pi } _ { i } ) \right) } \\ & { \leq \frac { 1 } { ( T _ { i } + k ) ^ { 8 } } \leq \frac { 1 } { ( k + 1 ) ^ { 8 } } . } \end{array}
$$

Therefore expected number of arm pulls of $\hat { \pi } _ { i }$ after first pulling it $T _ { i }$ time steps is upper bounded by

E [Number of time steps $\hat { \pi } _ { i }$ is pulled in round i after the first $T _ { i }$ time steps | πˆ<sub>i</sub>]

$$
= \sum _ { s = 1 } ^ { \infty } \operatorname* { P r } ( k \geq s | \hat { \pi } _ { i } ) \leq \sum _ { s = 1 } ^ { \infty } \frac { 1 } { ( T _ { i } + s - 1 ) ^ { 8 } } \leq \sum _ { s = 1 } ^ { \infty } \frac { 1 } { ( s + 1 ) ^ { 8 } } \leq 1 .\tag{33}
$$

Summing up the three parts of satisficing regret, i.e., the satisficing regret from Step 1 (31), Step 2 (32), and Step 3 (combination of (33) and (30)), we have that the satisficing regret of round i is upper bounded by

$$
\left( \frac { 6 C _ { 1 } + 6 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 3 3 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } + 2 \right) \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

## A.7. Proof of Proposition 4

The reason why round i ends within the time horizon T can be divided into two factors, reward LCB-test is triggered or resource UCB-test is triggered.

LCB-Test: We first deal with reward LCB-test. Recall that Lemma 3 implies that

$$
\mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] \leq \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

According to the condition of this proposition, we have i satisfies the condition

$$
6 4 \left( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } \leq \Delta _ { S } ^ { \prime } ,
$$

by Markov’s inequality, we have

$$
\begin{array} { r l } {  { \operatorname* { P r } ( Z ^ { \prime } - r ( \hat { \pi } _ { i } ) > \frac { \Delta _ { S } ^ { \prime } } { 2 } ) = \operatorname* { P r } ( \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} > \frac { \Delta _ { S } ^ { \prime } } { 2 } ) \leq \frac { \mathbb { E } [ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} ] } { \Delta _ { S } ^ { \prime } / 2 } } } \\ & { \leq \frac { ( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } ) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } { 6 4 ( \frac { 2 C _ { 1 } + 2 } { ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } + \frac { 1 1 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } ) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } / 2 } \leq \frac { 1 } { 3 2 } } \end{array}\tag{34}
$$

Now, if $r ( \hat { \pi } _ { i } ) < Z ^ { \prime } - \Delta _ { S } ^ { \prime } / 2$ , which happens with probability at most $1 / 3 2$ according to (34), we simply bound the probability of round i ends within time horizon T by 1.

If $r ( \hat { \pi } _ { i } ) \geq Z ^ { \prime } - \Delta _ { S } ^ { \prime } / 2$ , which happens with probability at least 31/32 and can be upper bounded by 1, then $r ( \hat { \pi } _ { i } ) \geq S + \Delta _ { S } ^ { \prime } / 2$ holds and the confidence interval for any $k \geq 0$ can be bounded by

$$
\begin{array} { r l r } {  { \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } \leq 4 \sqrt { \frac { \log ( T _ { i } ) } { T _ { i } } } = 4 \sqrt { \frac { \log ( \lceil \gamma _ { i } ^ { - 2 } \rceil ) } { \lceil \gamma _ { i } ^ { - 2 } \rceil } } } } \\ & { } & { \leq 4 \sqrt { \gamma _ { i } ^ { 2 } \log ( 4 \gamma _ { i } ^ { - 2 } ) } = 4 \sqrt { 2 } \gamma _ { i } \log ( 2 / \gamma _ { i } ) ^ { 1 / 2 } } \\ & { } & { \leq 4 \sqrt { 2 } \cdot \frac { \Delta _ { S } ^ { \prime } } { 6 4 } \leq \frac { \Delta _ { S } ^ { \prime } } { 4 } } \end{array}\tag{35}
$$

where the first inequality is because $T _ { i } > 1$ , and $f ( y ) = \log ( y ) / y$ is monotonically decreasing when $y \geq 3$ and $2 \cdot \log ( 2 ) / 2 \geq \log ( 3 ) / 3$ for all $k \geq 0$ , thus log $( T _ { i } + k ) / ( T _ { i } + k ) \le 4 \log ( T _ { i } ) / T _ { i }$ for all $k \geq 0$ the second inequality is because $\gamma _ { i } ^ { - 1 } \geq 1$ , thus $\lceil \gamma _ { i } ^ { - 2 } \rceil \leq 2 \gamma _ { i } ^ { - 2 } < 4 \gamma _ { i } ^ { - 2 }$ ; the third inequality is based on the condition of this proposition, where with $C _ { 1 } > 1 , ( 1 - \alpha ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } < 1$ , and $B _ { j } ^ { \prime } \leq T$ for all j we know $6 4 \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } \leq \Delta _ { S } ^ { \prime }$ . Suppose round i is terminated for some k due to reward LCB-test, then

$$
\frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } < S = Z ^ { \prime } - \Delta _ { S } ^ { \prime } \leq r ( \hat { \pi } _ { i } ) - \frac { \Delta _ { S } ^ { \prime } } { 2 } \leq r ( \hat { \pi } _ { i } ) - 2 \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } ,
$$

or equivalently

$$
\frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } \leq r ( \hat { \pi } _ { i } ) - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } .
$$

By the Hoefding’s inequality we have

$$
\operatorname* { P r } \left( \frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } \leq r ( \hat { \pi } _ { i } ) - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } \right) \leq \frac { 1 } { ( T _ { i } + k ) ^ { 8 } } \leq \frac { 1 } { ( k + 1 ) ^ { 8 } } .
$$

Therefore conditioned on $r ( \hat { \pi } _ { i } ) \geq Z ^ { \prime } - \Delta _ { S } ^ { \prime } / 2$ the probability that round i is terminated by the end of the horizon is upper bounded by

$$
\sum _ { k = 0 } ^ { \infty } { \frac { 1 } { ( k + T _ { i } ) ^ { 8 } } } \leq \sum _ { k = 0 } ^ { \infty } { \frac { 1 } { ( k + 2 ) ^ { 8 } } } \leq { \frac { 1 } { 3 2 } } .\tag{36}
$$

Combining (34) and (36) and we conclude that the probability that round i terminates by the end of the time horizon due to reward LCB-test is upper bounded by $1 / 1 6$ conditioned on $\mathcal { E } .$ UCB-Test: We then deal with resource UCB-test. Recall that events $\mathcal { C } _ { j i } ^ { 2 }$ is defined as

$$
\mathcal { C } _ { j i } ^ { 2 } = \left\{ \forall k \geq 0 : c _ { j } ( \hat { \pi } _ { i } ) \leq \frac { B _ { j } ^ { \prime } } { T } - 2 \sqrt { \frac { 4 \log ( d ( T _ { i } + k ) ) } { T _ { i } + k } } \right\} .
$$

If $\mathcal { C } _ { j i } ^ { 2 }$ does not hold, which happens with probability at most $1 / ( 3 2 d )$ according to Equation (16) in Lemma $^ { 2 , }$ we simply bound the probability of round i ends within time horizon T by 1.

If $\mathcal { C } _ { j i } ^ { 2 }$ holds, and suppose that round i is terminated for some k due to resource UCB-test, then

$$
\frac { \hat { c } _ { j i } ^ { \mathrm { t o t } } ( k + T _ { i } ) } { k + T _ { i } } + \sqrt { \frac { 4 \log ( d ( k + T _ { i } ) ) } { k + T _ { i } } } > \frac { B _ { j } ^ { \prime } } { T } \ge c _ { j } ( \hat { \pi } _ { i } ) + 2 \sqrt { \frac { 4 \log ( d ( k + T _ { i } ) ) } { k + T _ { i } } }
$$

where the first inequality holds due to UCB test, and the second inequality is based on definition of $\mathcal { C } _ { j i } ^ { 2 }$ . This is equivalent to

$$
\frac { \hat { c } _ { j i } ^ { \mathrm { t o t } } ( k + T _ { i } ) } { k + T _ { i } } > c _ { j } ( \hat { \pi } _ { i } ) + \sqrt { \frac { 4 \log ( d ( k + T _ { i } ) ) } { k + T _ { i } } } .
$$

By the Hoefding’s inequality we have

$$
\operatorname* { P r } \left( \frac { \hat { c } _ { j i } ^ { \mathrm { t o t } } ( k + T _ { i } ) } { k + T _ { i } } > c _ { j } ( \hat { \pi } _ { i } ) + \sqrt { \frac { 4 \log ( d ( k + T _ { i } ) ) } { k + T _ { i } } } \right) \leq \frac { 1 } { d ^ { 8 } ( T _ { i } + k ) ^ { 8 } } \leq \frac { 1 } { d ( k + 1 ) ^ { 8 } } .
$$

Therefore conditioned on $\mathcal { C } _ { j i } ^ { 2 }$ , the probability that round i is terminated by the end of the horizon is upper bounded by

$$
\sum _ { k = 1 } ^ { \infty } { \frac { 1 } { d ( k + 1 ) ^ { 8 } } } \leq { \frac { 1 } { 3 2 d } } .\tag{37}
$$

Combining (16) and (37) and we conclude that the probability that round i terminates by the end of the time horizon due to resource j UCB-test is upper bounded by $1 / ( 1 6 d )$ , and sums to $1 / 1 6$ among all d types of resources conditioned on the history in the round.

Therefore, combining the terminate probability due to resource UCB-test and that due to reward LCB-test, the probability that round i terminates by the end of the time horizon is upper bounded by $1 / 1 6 + 1 / 1 6 = 1 / 8$ conditioned on the history in the round.

## B. Proof of Theorem 3

Following the same pattern as in proof of Theorem 1, we first analyze the satisficing regret conditioned on $\mathcal { E } .$ The same as (7), if T is suficiently large with $\sqrt { T / \log ( 2 d T ^ { 2 } ) } \geq 2 0 \alpha / ( b ( 1 - \alpha ) )$ term $1 0 \sqrt { 2 } T / ( \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } )$ can be upper bounded by

$$
\begin{array} { r l } & { \frac { 1 0 \sqrt 2 T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } = 1 0 \sqrt 2 \cdot \bigg ( \underset { j } { \operatorname* { m i n } } \frac { B _ { j } } { T } - \frac { 1 0 \alpha } { 1 - \alpha } \sqrt { \frac { \log ( 2 d T ^ { 2 } ) } { T } } \bigg ) ^ { - 1 } } \\ & { \qquad \leq 1 0 \sqrt 2 \cdot ( b - b / 2 ) ^ { - 1 } = \frac { 2 0 \sqrt 2 } { b } . } \end{array}
$$

Denote $\beta ^ { \prime } = \operatorname* { m a x } \{ \beta , 1 / 2 \}$ and $\begin{array} { r } { K _ { \alpha , \beta } ^ { \prime } = 6 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 2 0 \sqrt { 2 } } { b } \right) \ge 6 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) } \end{array}$ (we use $K _ { \alpha , \beta } ^ { \prime }$ to show a diference with $K _ { \alpha , \beta } )$ . We define $i _ { 0 }$ as

$$
i _ { 0 } = \operatorname* { m a x } \left\{ i : 6 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \cdot \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \beta ^ { \prime } } > \Delta _ { S } ^ { \prime } \right\} .
$$

We first prove an upper bound on $i _ { 0 }$ . Denote $\delta _ { 0 } \in ( 0 , 1 )$ the smallest solution to the following equation

$$
\delta _ { 0 } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } = \frac { \Delta _ { S } ^ { \prime } } { 6 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) } .
$$

We note that, this equation is nearly the same as (8) except a diference in constant term $2 C _ { 1 }$ instead of $2 C _ { 1 } + 2$ . Therefore, following the same proof pattern in Theorem 1, we get the same bound as (9) that

$$
\frac { 4 d } { \delta _ { 0 } } \leq 4 \left( \frac { \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } ^ { \prime } } { \Delta _ { S } ^ { \prime } } \right) ^ { 4 / 3 } \leq \left( \frac { 4 \sqrt { 2 } d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } ^ { \prime } } { \Delta _ { S } ^ { \prime } } \right) ^ { 4 / 3 } ,\tag{38}
$$

where the diference in constant term is included in $K _ { \alpha , \beta } ^ { \prime }$ instead of $K _ { \alpha , \beta }$ . Plugging (38) into the log $( 4 d / \delta _ { 0 } )$ term in the definition of $\delta _ { 0 }$ , we write $D _ { \alpha , \beta } ^ { \prime } = 4 \sqrt 2 d ( 4 \beta ^ { \prime } ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } ^ { \prime } , L _ { \alpha , \beta } ^ { \prime } = ( 4 / 3 ) ^ { \beta ^ { \prime } } K _ { \alpha , \beta } ^ { \prime }$ , and we have

$$
\delta _ { 0 } \geq \frac { \Delta _ { S } ^ { \prime } } { L _ { \alpha , \beta } ^ { \prime } \log ( D _ { \alpha , \beta } ^ { \prime } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } .
$$

By definition of $i _ { 0 }$ we have that $i _ { 0 }$ is the largest integer that satisfies $\gamma _ { i _ { 0 } } \geq \delta _ { 0 }$ . Therefore we get

$$
\gamma _ { i _ { 0 } } \geq \frac { \Delta _ { S } ^ { \prime } } { L _ { \alpha , \beta } ^ { \prime } \log ( D _ { \alpha , \beta } ^ { \prime } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } ,
$$

thus

$$
i _ { 0 } \leq \frac { \alpha } { 1 - \alpha } \log _ { 2 } \left( \frac { L _ { \alpha , \beta } ^ { \prime } \log ( D _ { \alpha , \beta } ^ { \prime } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } { \Delta _ { S } ^ { \prime } } \right) ,
$$

or equivalently, (the same as (10) with diference in constant terms)

$$
2 ^ { i _ { 0 } } \leq \left( \frac { L _ { \alpha , \beta } ^ { \prime } \log ( D _ { \alpha , \beta } ^ { \prime } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } { \Delta _ { S } ^ { \prime } } \right) ^ { \alpha / ( 1 - \alpha ) } .\tag{39}
$$

Using the result of Proposition 6, we have that the satisficing regret by round $i _ { 0 }$ is upper bounded by

$$
\begin{array} { r l } & { \frac { \dot { \varepsilon } _ { 0 } } { \sin ^ { 3 } } 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { \operatorname* { I m } \sqrt { 2 } } { \operatorname* { m i n } _ { j } B _ { j } ^ { \varepsilon } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \beta ^ { \prime } } } \\ & { \le \frac { \dot { \varepsilon } _ { 0 } } { \sin ^ { 3 } } 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { \operatorname* { I m } \sqrt { 2 } } { \operatorname* { m i n } _ { j } B _ { j } ^ { \varepsilon } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } } \\ & { \le \frac { \dot { \varepsilon } _ { 0 } } { \cos ^ { 3 } } \frac { K _ { \alpha , \beta } ^ { \varepsilon } } { 1 6 } \cdot 2 ^ { i } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } } \\ & { \le \frac { \dot { \varepsilon } _ { 0 } } { 8 } K _ { \alpha , \beta } ^ { \varepsilon } z ^ { 2 i _ { 0 } } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } } \\ & { \le \frac { \dot { \varepsilon } _ { 0 } ( 4 / \beta ) ^ { \alpha ^ { \prime } } } { 8 } K _ { \alpha , \gamma ^ { 2 } } ^ { \varepsilon } \left( \frac { L _ { \alpha , \beta } \log ( D _ { \alpha , \beta } / \Delta _ { \gamma } ^ { \varepsilon } ) ^ { \beta ^ { \prime } } } { \Delta _ { \gamma } } \right) ^ { \alpha / ( 1 - \alpha ) } \log \left( \frac { D _ { \alpha , \beta } ^ { \varepsilon } } { \Delta _ { \gamma } } \right) ^ { \beta ^ { \prime } } , } \end{array}\tag{40}
$$

where the first inequality is because by definition of $\delta _ { 0 }$ and $i _ { 0 }$ , we have $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha } \geq \gamma _ { i _ { 0 } }$ for all $i \leq i _ { 0 }$ , the second inequality is by plugging into the definition of $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha }$ and definition of $K _ { \alpha , \beta } ^ { \prime }$ , the third inequality is based on ${ \textstyle \sum _ { i = 1 } ^ { i _ { 0 } } 2 ^ { i } \leq 2 \cdot 2 ^ { i _ { 0 } } }$ , and the fourth inequality is by plugging into (38) and (39).

By the results of Proposition 7, for every $k \geq 1$ , the probability that round $i _ { 0 } + k$ starts within the time horizon is at most $8 ^ { - k + 1 }$ . Therefore the regret incurred after round $i _ { 0 }$ is upper bounded by

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { \infty } \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { 1 } \beta _ { 2 } } \right) \gamma _ { \mathrm { i } _ { 0 } + k } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { \mathrm { i } _ { 0 } + k } ) ^ { \beta ^ { \prime } } \cdot 8 ^ { - k + 1 } } \\ & { \le \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { K _ { \alpha , \beta } ^ { \prime } } { 1 6 } \cdot 2 ^ { i \alpha + k } \left( \log ( 4 d / \gamma _ { \mathrm { i } _ { 0 } } ) + k \right) ^ { \beta ^ { \prime } } \cdot 8 ^ { - k + 1 } } \\ & { \le \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { 2 ^ { \beta ^ { \prime } } K _ { \alpha , \beta } ^ { \prime } } { 1 6 } \cdot 2 ^ { i \alpha + k } \log ( 4 d / \gamma _ { \mathrm { i } _ { 0 } } ) ^ { \beta ^ { \prime } } \cdot k ^ { \beta ^ { \prime } } \cdot 8 ^ { - k + 1 } } \\ & { \le 2 ^ { \beta ^ { \prime } - 1 } K _ { \alpha , \beta ^ { 2 } } ^ { \prime } \mathrm { i } \log ( 4 d / \delta _ { 0 } ) ^ { \beta ^ { \prime } } \cdot M _ { \beta ^ { \prime } } ^ { \prime } } \\ &  \le \displaystyle \frac { ( 8 / 3 ) ^ { \beta ^ { \prime } } } { 2 } K _ { \alpha , \beta } ^ { \prime } \left( \frac { L _ { \alpha , \beta } ^ { \prime } } { \Delta _ { \beta } } \log ( D _ { \alpha , \beta } ^ { \prime } / \Delta _ { s } ^ { \prime } ) ^ { \beta ^ { \prime } } \right) ^ { \alpha / ( 1 - \alpha ) } \log \left( \frac { D _ { \alpha , \beta } ^ { \prime } } { \Delta _ { \beta ^ { \prime } } ^ { \prime } } \right) ^ { \beta ^ { \prime } } M _ { \beta ^ { \prime } } ^ \end{array}\tag{41}
$$

where $\begin{array} { r } { M _ { \beta ^ { \prime } } ^ { \prime } = \sum _ { k = 1 } ^ { \infty } k ^ { \beta ^ { \prime } } \cdot 4 ^ { - k } } \end{array}$ . The first inequality is by plugging into the definition of $K _ { \alpha , \beta } ^ { \prime }$ and $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha }$ , the second inequality is because log $( 4 d / \gamma _ { i _ { 0 } } ) > 1$ and $( 1 - \alpha ) / \alpha \leq 1$ for $1 / 2 \le \alpha < 1$ , and for any $a , b \geq 1$ , we have $2 a b \geq a b + 1 \geq a + b$ , the third inequality is based on $\begin{array} { r } { \sum _ { k = 1 } ^ { \infty } 2 ^ { k } \cdot 8 ^ { - k + 1 } = } \end{array}$ $8 \textstyle \sum _ { k = 1 } ^ { \infty } 4 ^ { - k }$ . Therefore, combining (40) and (41), the total satisficing regret conditioned on $\mathcal { E }$ is upper bounded by

$$
\left( \frac { ( 4 / 3 ) ^ { \beta ^ { \prime } } } { 8 } + \frac { ( 8 / 3 ) ^ { \beta ^ { \prime } } } { 2 } \right) K _ { \alpha , \beta } ^ { \prime } \left( \frac { L _ { \alpha , \beta } ^ { \prime } \log ( D _ { \alpha , \beta } ^ { \prime } / \Delta _ { S } ^ { \prime } ) ^ { \beta ^ { \prime } } } { \Delta _ { S } ^ { \prime } } \right) ^ { \alpha / ( 1 - \alpha ) } \log \left( \frac { D _ { \alpha , \beta } ^ { \prime } } { \Delta _ { S } ^ { \prime } } \right) ^ { \beta ^ { \prime } } ( M _ { \beta ^ { \prime } } ^ { \prime } + 1 ) .
$$

If $\Delta _ { S } ^ { \prime } \geq \delta ( T )$ , recall $\Delta _ { S } ^ { * } = Z ^ { * } - S$ and $\Delta _ { S } ^ { \prime } = Z ^ { \prime } - S$ . Proposition 5 implies $\Delta _ { S } ^ { * } - \Delta _ { S } ^ { \prime } \leq \delta ( T )$ . Then we have

$$
\Delta _ { S } ^ { \prime } \geq \frac { \delta ( T ) } { 2 } + \frac { \Delta _ { S } ^ { * } - \delta ( T ) } { 2 } = \frac { \Delta _ { S } ^ { * } } { 2 } .
$$

Combining this with $\operatorname* { P r } ( { \mathcal { E } } ^ { c } ) \leq 1 / T$ according to Proposition 2 and total satisficing regret incurred conditioned on negation of E is at most $T ,$ total satisficing regret is upper bounded by

$$
1 + \left( \frac { ( 4 / 3 ) ^ { \beta ^ { \prime } } } { 8 } + \frac { ( 8 / 3 ) ^ { \beta ^ { \prime } } } { 2 } \right) K _ { \alpha , \beta } ^ { \prime } \left( \frac { 2 L _ { \alpha , \beta } ^ { \prime } \log ( 2 D _ { \alpha , \beta } ^ { \prime } / \Delta _ { S } ^ { * } ) ^ { \beta ^ { \prime } } } { \Delta _ { S } ^ { * } } \right) ^ { \alpha / ( 1 - \alpha ) } \log { \left( \frac { 2 D _ { \alpha , \beta } ^ { \prime } } { \Delta _ { S } ^ { * } } \right) } ^ { \beta ^ { \prime } } ( M _ { \beta ^ { \prime } } ^ { \prime } + 1 ) .\tag{42}
$$

Since by definition of $t _ { i } = \lceil 2 ^ { i / \alpha } \rceil$ , the number of rounds that end within the time horizon is bounded by $2 ^ { i / \alpha } \leq T , i . e . , i \leq \lfloor \alpha \log _ { 2 } ( T ) \rfloor$ because i is an integer. Then, the maximum number of rounds is bounded by $i _ { \mathrm { m a x } } = \lceil \alpha \log _ { 2 } ( T ) \rceil \leq \alpha \log _ { 2 } ( T ) + 1$ . Recall $\gamma _ { i } = 2 ^ { - i ( 1 - \alpha ) / \alpha }$ , for any $i \leq i _ { \operatorname* { m a x } }$ , we have

$$
\begin{array} { r l } & { \gamma _ { i } \geq \gamma _ { i _ { \operatorname* { m a x } } } \geq 2 ^ { - ( 1 - \alpha ) \lceil \alpha \log _ { 2 } ( T ) \rceil / \alpha } } \\ & { \quad \geq 2 ^ { - ( 1 - \alpha ) \log _ { 2 } ( T ) / \alpha - ( 1 - \alpha ) / \alpha } } \\ & { \quad \geq 2 ^ { - \log _ { 2 } ( T ) - 1 } = 1 / ( 2 T ) , } \end{array}\tag{43}
$$

where the third inequality is because $( 1 - \alpha ) / \alpha \leq 1$ for $1 / 2 \le \alpha < 1$ . So, $2 / \gamma _ { i } \leq 4 T$ for all $i \le i _ { \mathrm { m a x } }$ Therefore, the satisficing regret is upper bounded by the sum of cumulative regret with respect to $Z ^ { \prime }$ and cumulative diference between $Z ^ { \prime }$ and $Z ^ { \ast }$

$$
\begin{array} { r l } & { \displaystyle \mathsf { R e g r e t } _ { S } \leq 1 + T ( Z ^ { * } - Z ^ { \prime } ) + \sum _ { i = 1 } ^ { \operatorname* { m a x } } 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \qquad \leq 1 + \displaystyle \frac { 1 0 \alpha } { b ( 1 - \alpha ) } \sqrt { T \log ( 2 d T ^ { 2 } ) } + \sum _ { i = 1 } ^ { \operatorname* { m a x } } 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) 2 ^ { i + 1 } \log ( 8 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \qquad \leq 1 + \displaystyle \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } T ^ { 1 / 2 } \log ( 4 d T ) ^ { 1 / 2 } + 1 6 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) 2 ^ { i \operatorname* { m a x } } \log ( 8 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \qquad \leq 1 + \left( 3 2 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) + \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } \right) T ^ { \alpha } \log ( 8 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } , } \end{array}
$$

where the first inequality is based on Proposition 6 and max $\{ S - r , 0 \} \le \left( Z ^ { \ast } - Z ^ { \prime } \right) + \operatorname* { m a x } \{ Z ^ { \prime } - r , 0 \}$ the second inequality is based on $\Delta _ { S } ^ { * } - \Delta _ { S } ^ { \prime } \leq \delta ( T )$ from Proposition 5 and $2 / \gamma _ { i } \leq 4 T$ from (43), the third inequality is based on $\begin{array} { r } { \sum _ { i = 1 } ^ { i _ { \operatorname* { m a x } } } 2 ^ { i } \leq 2 ^ { i _ { \operatorname* { m a x } } + 1 } } \end{array}$ , and the fourth inequality is plugging into $2 ^ { i _ { \operatorname* { m a x } } } \leq 2 ^ { \alpha \log _ { 2 } ( T ) + 1 } \leq 2 T ^ { \alpha }$ . Recall that if T is suficiently large with $\sqrt { T / \log ( 2 d T ^ { 2 } ) } \geq 2 0 \alpha / ( b ( 1 - \alpha ) )$ ， term $1 0 \sqrt { 2 } T / ( \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } )$ can be upper bounded by $2 0 { \sqrt { 2 } } / b .$ , and the satisficing regret can further be upper bounded by

$$
\mathsf { R e g r e t } _ { S } \leq 1 + \left( \frac { 6 4 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 6 4 0 \sqrt { 2 } } { b } + \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } \right) T ^ { \alpha } \log ( 8 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .\tag{44}
$$

Finally we deal with $\Delta _ { S } ^ { \prime } < \delta ( T )$ for constant regret bound, where we bound the satisficing regret with $T ^ { \alpha }$ · polylog(T). $\Delta _ { S } ^ { * } - \Delta _ { S } ^ { \prime } \leq \delta ( T )$ implies $\Delta _ { S } ^ { * } < 2 \delta ( T )$ . Similar to (14), we still have

$$
T \leq \frac { 1 6 0 0 \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } \log \left( \frac { 8 0 0 \sqrt { 2 d } \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } \right) .\tag{45}
$$

Using the sub-linear satisficing regret bound in (44) and (45), we have

$$
\begin{array} { r l } & { \mathsf { R e g r e t } _ { S } \leq 1 + ( \frac { 6 4 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 6 4 0 \sqrt { 2 } } { b } + \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } ) ( \frac { 1 6 0 0 \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } } ) ^ { \alpha } \frac { 1 } { ( \Delta _ { S } ^ { * } ) ^ { 2 \alpha } } } \\ & { \qquad \cdot \log ( \frac { 8 0 0 \sqrt { 2 } d \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } ) ^ { \alpha } \log ( \frac { 1 2 8 0 0 d \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } \log ( \frac { 8 0 0 \sqrt { 2 } d \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } ) ) ^ { \beta ^ { \prime } } } \\ & { \qquad \leq 1 + ( \frac { 6 4 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 6 4 0 \sqrt { 2 } } { b } + \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } ) ( \frac { 1 6 0 0 \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } } ) ^ { \alpha } ( \frac { 1 } { \Delta _ { S } ^ { * } } ) ^ { \alpha / ( 1 - \alpha ) } } \\ &  \qquad \cdot \log ( \frac { 8 0 0 \sqrt { 2 } d \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } ) ^ { \alpha } \log ( \frac { 1 2 8 0 0 d \alpha ^ { 2 } } { b ^ { 2 } ( 1 - \alpha ) ^ { 2 } ( \Delta _ { S } ^ { * } ) ^ { 2 } } \log (  \end{array}
$$

where the second inequality is based on $( \Delta _ { S } ^ { * } ) ^ { - 2 \alpha } \leq ( \Delta _ { S } ^ { * } ) ^ { - \alpha / ( 1 - \alpha ) }$ implied by $\alpha \ge 1 / 2$ and $\Delta _ { S } ^ { * } \le 1$

Combining the two parts of results, $i . e .$ , the constant satisficing regret (42) and the sub-linear satisficing regret (44), we have completed the proof of Theorem 3 by putting all constant terms into the polylog term.

## B.1. Reward Bound under Strong Regret Condition

Replacing the standard regret condition by the strong regret condition does not change the algorithm or the resource-consumption analysis. Therefore, the resource-consumption guarantees in Lemma 2 remain valid. Under Condition 2, we can directly control the expected sum of positive reward deficits along the oracle trajectory, which yields the following sharper bound on $\mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right]$

Lemma 5. Under Condition ${ \mathcal { Q } } ,$ the expected regret of a single pull on distribution $\hat { \pi } _ { i }$ in round i can be upper bounded in expectation as

$$
\mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] \leq \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

Proof. Recall that $\{ \tilde { \pi } _ { s } \} _ { s \in [ t _ { i } ] }$ is realized trajectory of running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ . Since $\hat { \pi } _ { i }$ is calculated based on $\{ \tilde { \pi } _ { s } \} _ { s \in [ t _ { i } ] }$ in round i, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] = \displaystyle \frac { 1 } { t _ { i } } \mathbb { E } \left[ \operatorname* { m a x } \left\{ t _ { i } Z ^ { \prime } - ( \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) ) \cdot ( 1 - w _ { i } ) , 0 \right\} \right] } \\ & { \leq \displaystyle \frac { 1 } { t _ { i } } \mathbb { E } \left[ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } \operatorname* { m a x } \{ Z ^ { \prime } - r ( \tilde { \pi } _ { s } ) , 0 \} \right] + \displaystyle \frac { w _ { i } } { t _ { i } } \mathbb { E } \left[ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } r ( \tilde { \pi } _ { s } ) \right] } \\ & { \leq C _ { 1 } t _ { i } ^ { \alpha - 1 } \log ( t _ { i } ) ^ { \beta } + w _ { i } . } \end{array}
$$

For $t _ { i } ^ { \alpha - 1 } \log ( t _ { i } ) ^ { \beta }$ , it can be simply upper bounded by $\begin{array} { r } { \frac { 2 } { ( 1 - \alpha ) ^ { \beta } } \gamma _ { i } \log ( \frac { 2 } { \gamma _ { i } } ) ^ { \beta } } \end{array}$ . For $w _ { i }$ , it can be upper bounded by $\begin{array} { r } { \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \gamma _ { i } \log \left( 4 d / \gamma _ { i } \right) ^ { 1 / 2 } } \end{array}$

Therefore, the expected satisficing regret of a single pull can be upper bounded by

$$
\mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] \leq \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

## B.2. Proof of Proposition 6

Recall Lemma 5 implies that the satisficing regret of a single pull on $\hat { \pi } _ { i }$ can be upper bounded by

$$
\begin{array} { r l } & { \mathbb { E } [ \operatorname* { m a x } \{ S - r ( \hat { \pi } _ { i } ) , 0 \} ] \leq \mathbb { E } \left[ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} \right] } \\ & { \qquad \leq \left( \displaystyle \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \displaystyle \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } . } \end{array}\tag{46}
$$

The regret in round i can be divided into three parts. Running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , forced-sampling, and LCB-UCB test.

Recall that $t _ { i } = \lceil \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \rceil \leq 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) }$ , then $\log ( t _ { i } ) \le \log ( 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } ) \le \log ( 2 / \gamma _ { i } ) / ( 1 - \alpha )$ and $t _ { i } ^ { \alpha } = \lceil 2 ^ { i / \alpha } \rceil ^ { \alpha } \leq 2 ^ { \alpha ( 1 + i / \alpha ) } \leq 2 \cdot 2 ^ { i } = 2 \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) }$ . The satisficing regret incurred when running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ is upper bounded by the expected strong regret bound

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } \operatorname* { m a x } \{ S - r ( \tilde { \pi } _ { s } ) , 0 \} \right] \leq \mathbb { E } \left[ \displaystyle \sum _ { s = 1 } ^ { t _ { i } } \operatorname* { m a x } \{ Z ^ { \prime } - r ( \tilde { \pi } _ { s } ) , 0 \} \right] \leq C _ { 1 } t _ { i } ^ { \alpha } \log ( t _ { i } ) ^ { \beta } } \\ & { \qquad \leq \displaystyle \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log \left( 2 / \gamma _ { i } \right) ^ { \beta } } \\ & { \qquad \leq \left( \displaystyle \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } . } \end{array}\tag{47}
$$

Similar to (32), the satisficing regret incurred when pulling $\hat { \pi } _ { i }$ in forced sampling for $T _ { i } = \lceil \gamma _ { i } ^ { - 2 } \rceil$ time steps is upper bounded by

$$
\begin{array} { r l } & { \quad \lceil \gamma _ { i } ^ { - 2 } \rceil \cdot \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \leq \frac { 2 } { \gamma _ { i } ^ { 2 } } \cdot \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \leq 2 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \end{array}\tag{48}
$$

If $r ( \hat { \pi } _ { i } ) < S$ (otherwise the satisficing regret is 0), for every $k ,$ round i is not terminated after pulling $\hat { \pi } _ { i }$ for k time steps in addition to first $T _ { i }$ time steps occurs with probability at most

$$
\begin{array} { r l } & { \operatorname* { P r } \left( \cfrac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \cfrac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } > S \right) } \\ & { \leq \operatorname* { P r } \left( \cfrac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \cfrac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } > r ( \hat { \pi } _ { i } ) \right) } \\ & { \leq \cfrac { 1 } { ( T _ { i } + k ) ^ { 8 } } \leq \cfrac { 1 } { ( k + 1 ) ^ { 8 } } . } \end{array}
$$

Therefore expected number of arm pulls of $\hat { \pi } _ { i }$ after first pulling it $T _ { i }$ time steps is upper bounded by

E [Number of time steps $\hat { \pi } _ { i }$ is pulled in round i after the first $T _ { i }$ time steps | πˆ<sub>i</sub>]

$$
= \sum _ { s = 1 } ^ { \infty } \operatorname* { P r } ( k \geq s | \hat { \pi } _ { i } ) \leq \sum _ { s = 1 } ^ { \infty } \frac { 1 } { ( T _ { i } + s - 1 ) ^ { 8 } } \leq \sum _ { s = 1 } ^ { \infty } \frac { 1 } { ( s + 1 ) ^ { 8 } } \leq 1 .\tag{49}
$$

Summing up the three parts of satisficing regret, $i . e .$ , the satisficing regret from Step 1 (47), Step 2 (48), and Step 3 (combination of (49) and (46)), we have that the satisficing regret of round i is upper bounded by

$$
4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname * { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname * { m a x } \{ \beta , 1 / 2 \} } .
$$

## B.3. Proof of Proposition 7

The reason why round i ends within the time horizon T can be divided into two factors, reward LCBtest is triggered or resource UCB-test is triggered. Since the diference in regret bound (Condition 1 and 2) does not afect the costs and, consequently, does not impact the UCB-test, we can still bound the probability that round i terminates by $1 / 1 6$ . Therefore we only focus on LCB-test, and follow similar proof pattern in Proposition 4 with diference in constant terms.

By Lemma 5 we have

$$
\mathbb { E } [ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} ] \leq \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

According to the condition of this proposition, we have i satisfies the condition

$$
6 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } \leq \Delta _ { S } ^ { \prime } ,
$$

by Markov’s inequality, we have

$$
\begin{array} { r l } & { \operatorname* { P r } \left( Z ^ { \prime } - r ( \hat { \pi } _ { i } ) > \frac { \Delta _ { S } ^ { \prime } } { 2 } \right) = \operatorname* { P r } \left( \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} > \frac { \Delta _ { S } ^ { \prime } } { 2 } \right) } \\ & { \leq \frac { \mathbb { E } [ \operatorname* { m a x } \{ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) , 0 \} ] } { \Delta _ { S } ^ { \prime } / 2 } \leq \frac { \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { \ln \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } { 6 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { \ln \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } / 2 } \leq \frac { 1 } { 3 2 } } \end{array}
$$

If $r ( \hat { \pi } _ { i } ) \geq Z ^ { \prime } - \Delta _ { S } ^ { \prime } / 2$ , which happens with probability at least $3 1 / 3 2$ , then $r ( \hat { \pi } _ { i } ) \geq S + \Delta _ { S } ^ { \prime } / 2$ holds. Similar to (35), and given that $6 4 \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } \leq \Delta _ { S } ^ { \prime }$ still holds despite a diferent constant term in the proposition condition, we know for all $k \geq 0$

$$
{ \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } } \leq 4 { \sqrt { \frac { \log ( T _ { i } ) } { T _ { i } } } } \leq { \frac { \Delta _ { S } ^ { \prime } } { 4 } } .
$$

Suppose round i is terminated for some k due to reward LCB-test, then

$$
\frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } < S = Z ^ { \prime } - \Delta _ { S } ^ { \prime } \leq r ( \hat { \pi } _ { i } ) - \frac { \Delta _ { S } ^ { \prime } } { 2 } \leq r ( \hat { \pi } _ { i } ) - 2 \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } ,
$$

or equivalently

$$
\frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } \leq r ( \hat { \pi } _ { i } ) - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } .
$$

By the Hoefding’s inequality we have

$$
\operatorname* { P r } \left( \frac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } \leq r ( \hat { \pi } _ { i } ) - \sqrt { \frac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } \right) \leq \frac { 1 } { ( T _ { i } + k ) ^ { 8 } } \leq \frac { 1 } { ( k + 1 ) ^ { 8 } } .
$$

Therefore conditioned on $r ( \hat { \pi } _ { i } ) \geq Z ^ { \prime } - \Delta _ { S } ^ { \prime } / 2$ the probability that round i is terminated by the end of the horizon is upper bounded by

$$
\sum _ { k = 1 } ^ { \infty } { \frac { 1 } { ( k + 1 ) ^ { 8 } } } \leq { \frac { 1 } { 3 2 } } .
$$

Combining above and we conclude that the probability that round i terminates by the end of the time horizon due to reward LCB-test is upper bounded by $1 / 1 6$ conditioned on $\cap _ { i ^ { \prime } < i } \mathcal { E } _ { i ^ { \prime } }$ . Also, recall in Proposition 4, the probability that round i terminates by the end of the time horizon due to resource UCB-test is upper bounded by $1 / 1 6$ conditioned on $\cap _ { i ^ { \prime } < i } \mathcal { E } _ { i ^ { \prime } }$ . Therefore, combining resource UCB-test and reward LCB-test, the probability that round i terminates by the end of the time horizon is upper bounded by $1 / 1 6 + 1 / 1 6 = 1 / 8$ conditioned on $\cap _ { i ^ { \prime } < i } \mathcal { E } _ { i ^ { \prime } }$

## C. Proof of Theorem 2

As $\hat { \pi } _ { i }$ is derived from oracle running ALG under adjusted resource capacities $B ^ { \prime }$ , the cumulative standard regret can be divided into

$$
\mathsf { R e g r e t } \leq 1 + \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( Z ^ { \prime } - r ( \pi _ { t } ) ) \right] + T ( Z ^ { * } - Z ^ { \prime } ) .\tag{50}
$$

For the first term in (50), which shows the cumulative regret with respect to $Z ^ { \prime }$ , we first bound the standard regret of a single pull on $\hat { \pi } _ { i }$ with respect to $Z ^ { \prime }$ by

$$
\begin{array} { r l r } {  { \mathbb { E } [ Z ^ { \prime } - r \big ( \hat { \pi } _ { i } \big ) ] = \frac { 1 } { t _ { i } } \mathbb { E } [ t _ { i } Z ^ { \prime } - \big ( \underset { s = 1 } { \overset { t _ { i } } { \sum } } r \big ( \tilde { \pi } _ { s } \big ) \big ) \cdot \big ( 1 - w _ { i } \big ) ] } } \\ & { } & { \leq \frac { 1 } { t _ { i } } \mathbb { E } [ \underset { s = 1 } { \overset { t _ { i } } { \sum } } \big ( Z ^ { \prime } - r \big ( \tilde { \pi } _ { s } \big ) \big ) ] + \frac { w _ { i } } { t _ { i } } \mathbb { E } [ \underset { s = 1 } { \overset { t _ { i } } { \sum } } r \big ( \tilde { \pi } _ { s } \big ) ] } \\ & { } & { \leq C _ { 1 } t _ { i } ^ { \alpha - 1 } \log ( t _ { i } ) ^ { \beta } + w _ { i } , } \end{array}
$$

where the final inequality is based on Condition 1 and $r ( \pi ) \in [ 0 , 1 ]$ for any $\pi .$ Following the same calculation in Lemma 3, we can combine (21) and (22) and bound the standard regret of a single pull on $\hat { \pi } _ { i }$ with respect to $Z ^ { \prime }$ by

$$
\mathbb { E } \left[ Z ^ { \prime } - r ( \hat { \pi } _ { i } ) \right] \leq \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log \left( \frac { 4 d } { \gamma _ { i } } \right) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

The standard regret with respect to $Z ^ { \prime }$ in round i can be divided into three parts. Running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ , forced-sampling, and LCB-UCB test.

The standard regret incurred when running $\mathtt { A L G } ( t _ { i } , \{ b _ { j i } \} )$ can directly bounded as

$$
\begin{array} { r l } {  { \mathbb { E } [ \sum _ { s = 1 } ^ { t _ { i } } ( Z ^ { \prime } - r ( \tilde { \pi } _ { s } ) ) ] \leq C _ { 1 } t _ { i } ^ { \alpha } \log ( t _ { i } ) ^ { \beta } } } \\ & { \leq \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 2 / \gamma _ { i } ) ^ { \beta } } \\ & { \leq ( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } ) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } , } \end{array}\tag{51}
$$

where the second inequality is because $t _ { i } = \lceil \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } \rceil \leq 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) }$ , then log $\ L ( t _ { i } ) \leq \log ( 2 \gamma _ { i } ^ { - 1 / ( 1 - \alpha ) } ) \leq$ log $( 2 / \gamma _ { i } ) / ( 1 - \alpha )$ and $t _ { i } ^ { \alpha } = \lceil 2 ^ { i / \alpha } \rceil ^ { \alpha } \leq 2 ^ { \alpha ( 1 + i / \alpha ) } \leq 2 \cdot 2 ^ { i } = 2 \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) }$

The standard regret incurred when pulling $\hat { \pi } _ { i }$ in forced sampling for $T _ { i } = \lceil \gamma _ { i } ^ { - 2 } \rceil$ time steps is upper bounded by

$$
\begin{array} { r l } & { \quad \lceil \gamma _ { i } ^ { - 2 } \rceil \cdot ( C _ { 1 } t _ { i } ^ { \alpha - 1 } ( \log t _ { i } ) ^ { \beta } + w _ { i } ) } \\ & { \le \displaystyle \frac { 2 } { \gamma _ { i } ^ { 2 } } \cdot \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \le 2 \left( \displaystyle \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } . } \end{array}\tag{52}
$$

As $r ( \hat { \pi } _ { i } ) < Z ^ { \prime } < Z ^ { * } < S$ , round i is not terminated after pulling $\hat { \pi } _ { i }$ for $k$ time steps in addition to first $T _ { i }$ time steps occurs with probability at most

$$
\begin{array} { r l } & { \operatorname* { P r } \left( \cfrac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \cfrac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } > S \right) } \\ & { \leq \operatorname* { P r } \left( \cfrac { \hat { r } _ { i } ^ { \mathrm { t o t } } } { T _ { i } + k } - \sqrt { \cfrac { 4 \log ( T _ { i } + k ) } { T _ { i } + k } } > r ( \hat { \pi } _ { i } ) \right) } \\ & { \leq \cfrac { 1 } { ( T _ { i } + k ) ^ { 8 } } \leq \cfrac { 1 } { ( k + 1 ) ^ { 8 } } . } \end{array}
$$

Therefore expected number of arm pulls of $\hat { \pi } _ { i }$ after first pulling it $T _ { i }$ time steps is upper bounded by

E [Number of time steps $\hat { \pi } _ { i }$ is pulled in round i after the first $T _ { i }$ time steps | πˆ<sub>i</sub>]

$$
= \sum _ { s = 1 } ^ { \infty } \operatorname* { P r } ( k \geq s | \hat { \pi } _ { i } ) \leq \sum _ { s = 1 } ^ { \infty } \frac { 1 } { ( T _ { i } + s - 1 ) ^ { 8 } } \leq \sum _ { s = 1 } ^ { \infty } \frac { 1 } { ( s + 1 ) ^ { 8 } } \leq 1 .\tag{53}
$$

Summing up the three parts of standard regret, i.e., (51), (52), and (53), we have that the standard regret with respect to $Z ^ { \prime }$ of round i is upper bounded by

$$
4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname * { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname * { m a x } \{ \beta , 1 / 2 \} } .
$$

For the second term in (50), which shows the cumulative diference in optimal reward between resource constraints $B _ { j }$ and $B _ { j } ^ { \prime }$ . Recall the diference between $Z ^ { * }$ and $Z ^ { \prime }$ is bounded in Proposition $5 ,$ then the cumulative diference can be bounded as

$$
T ( Z ^ { * } - Z ^ { \prime } ) \leq \frac { 1 0 \alpha } { b ( 1 - \alpha ) } \sqrt { T \log ( 2 d T ^ { 2 } ) } .
$$

Since by definition of $t _ { i } = \lceil 2 ^ { i / \alpha } \rceil$ , the number of rounds that end within the time horizon is bounded by $2 ^ { i / \alpha } \leq T , i . e . , i \leq \lfloor \alpha \log _ { 2 } ( T ) \rfloor$ because i is an integer. Then, the maximum number of rounds is bounded by $i _ { \mathrm { m a x } } = \lceil \alpha \log _ { 2 } ( T ) \rceil \leq \alpha \log _ { 2 } ( T ) + 1$ . Recall (43) that $\gamma _ { i } \geq 1 / ( 2 T )$ , total standard regret can be bounded as

$$
\begin{array} { r l } & { \mathbb { R e g r e t } \leq 1 + T ( Z ^ { * } - Z ^ { \prime } ) + \displaystyle \sum _ { i = 1 } ^ { i \operatorname* { m a x } } 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) \gamma _ { i } ^ { - \alpha / ( 1 - \alpha ) } \log ( 4 d / \gamma _ { i } ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \qquad \leq 1 + \displaystyle \frac { 1 0 \alpha } { b ( 1 - \alpha ) } \sqrt { T \log ( 2 d T ^ { 2 } ) } + \displaystyle \sum _ { i = 1 } ^ { i \operatorname* { m a x } } 4 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) 2 ^ { i + 1 } \log ( 8 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \qquad \leq 1 + \displaystyle \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } T ^ { 1 / 2 } \log ( 4 d T ) ^ { 1 / 2 } + 3 2 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) T ^ { \alpha } \log ( 8 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } } \\ & { \qquad \leq 1 + \left( 3 2 \left( \frac { 2 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 1 0 \sqrt { 2 } T } { \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } } \right) + \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } \right) T ^ { \alpha } \log ( 8 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } . } \end{array}
$$

where the first inequality follows from the standard regret bound for each round derived above, the second inequality is based on $\Delta _ { S } ^ { * } - \Delta _ { S } ^ { \prime } \leq \delta ( T )$ from Proposition 5 and $2 / \gamma _ { i } \leq 4 T$ from (43), the third inequality is based on $\begin{array} { r } { \sum _ { i = 1 } ^ { i _ { \operatorname* { m a x } } } 2 ^ { i } \leq 2 ^ { i _ { \operatorname* { m a x } } + 1 } } \end{array}$ , and the fourth inequality is plugging into $2 ^ { i _ { \operatorname* { m a x } } } \leq 2 ^ { \alpha \log _ { 2 } ( T ) + 1 } \leq 2 T ^ { \alpha }$ . Recall that if $T$ is suficiently large with $\sqrt { T / \log ( 2 d T ^ { 2 } ) } \geq 2 0 \alpha / ( b ( 1 - \alpha ) )$ term $1 0 \sqrt { 2 } T / ( \operatorname* { m i n } _ { j } B _ { j } ^ { \prime } )$ can be upper bounded by $2 0 \sqrt { 2 } / b$ , and the satisficing regret can further be upper bounded by

$$
\mathtt { R e g r e t } \le 1 + \left( \frac { 6 4 C _ { 1 } } { ( 1 - \alpha ) ^ { \beta } } + \frac { 6 4 0 \sqrt { 2 } } { b } + \frac { 1 0 \sqrt { 2 } \alpha } { b ( 1 - \alpha ) } \right) T ^ { \alpha } \log ( 4 d T ) ^ { \operatorname* { m a x } \{ \beta , 1 / 2 \} } .
$$

## D. Strong Regret Bound for Existing BwK Algorithms

In this section, we show that two existing algorithms for bandits with resource constraints satisfy Condition 2. For both algorithms, let $Z ^ { * } ( T , B )$ denote the corresponding per-round full-information optimal reward, and define

$$
\mathrm { O P T } : = T Z ^ { * } ( T , B ) , \qquad \mathrm { R E W } _ { S } : = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ r ( \pi _ { t } ) , Z ^ { * } ( T , B ) \right\} \right] .
$$

Here, min $\{ r ( \pi _ { t } ) , Z ^ { * } ( T , B ) \}$ is the strong reward collected at time t, which truncates any reward above the LP optimal reward. Therefore,

$$
\mathrm { O P T - R E W } _ { S } = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \operatorname* { m a x } \left\{ Z ^ { * } ( T , B ) - r ( \pi _ { t } ) , 0 \right\} \right] ,
$$

which is exactly the expected strong regret in Condition 2. We first establish this property for BalancedExploration in Badanidiyuru et al. (2018), where $Z ^ { * } ( T , B ) = \mathrm { O P T _ { L P } } / T$ , and then for a discrete version of DPA in Wang et al. (2014), where $Z ^ { * } ( T , B ) = p ^ { D } \lambda ( p ^ { D } )$

## D.1. Bandits with Knapsacks

We consider the bandit setting given in Badanidiyuru et al. (2018), where there is a finite arm set denoted as X , d resources that will be consumed under a constraint B for each $i \in [ d ]$ , and a total budget T rounds. Again, we assume $B \geq b T$ . As in Badanidiyuru et al. (2018), after the uniform-budget normalization the time horizon is included among the d resources: its budget is B and its deterministic per-round consumption is $B / T .$ . In each time $t \in [ T ]$ , if we pick an arm $X _ { t } \in \mathcal { X }$ we will receive a reward $r ( X _ { t } )$ and consume $c _ { i } ( X _ { t } )$ amount resources for each resource $i ,$ both are in expectation and will be observed with a small noise after each time step t. Namely for each arm $X _ { t } ,$ the reward and resource consumption form an outcome vector $( Y _ { t } ; C _ { t } ^ { 1 } ( X _ { t } ) , \cdot \cdot \cdot , C _ { t } ^ { d } ( X _ { t } ) ) \in [ 0 , 1 ] ^ { d + 1 }$ ， which is an i.i.d sample from a fixed distribution with mean $( r ( X _ { t } ) ; c _ { 1 } ( X _ { t } ) , \cdots , c _ { d } ( X _ { t } ) ) \in [ 0 , 1 ] ^ { d + 1 }$ unknown to the algorithm. We also write $\mu = ( ( r ( X ) ; c _ { 1 } ( X ) , \cdots , c _ { d } ( X ) ) : X \in { \mathcal { X } } )$ as the latent structure of a problem instance.

As the optimal dynamic policy given foreknowledge of outcome distribution is dificult to derive, we introduce a fractional LP-relaxation. That is, we relax the time steps t from discrete set [T] to continuous interval [0, T], and relax the policy to be a time-invariant distribution denoted as $\pi : \mathcal { X }  [ 0 , 1 ]$ . Namely, the LP relaxation can be expressed as an optimization problem as

$$
\begin{array} { r l } { \underset { t } { \operatorname* { m a x } } } & { t r ( \pi ) \quad \mathrm { i n } \ t \in \mathbb { R } } \\ { \mathrm { s . t . } } & { t c _ { i } ( \pi ) \leq B \quad \mathrm { f o r ~ e a c h ~ r e s o u r c e } \ i } \\ & { 0 \leq t \leq T , } \end{array}
$$

where $\begin{array} { r } { r ( \pi ) = \sum _ { X \in { \mathcal { X } } } \pi ( X ) r ( X ) } \end{array}$ and $\begin{array} { r } { c _ { i } ( \pi ) = \sum _ { X \in { \mathcal X } } \pi ( X ) c _ { i } ( X ) } \end{array}$ . The optimal reward is called LP-value of $\pi$ as

$$
\mathrm { L P } ( \pi ) = r ( \pi ) \operatorname* { m i n } _ { i } \frac { B } { c _ { i } ( \pi ) } ,
$$

which is linear in $T$ according to the optimization problem. So, we can write $\mathrm { L P } ( \pi ) = C _ { \pi } T$ where $C _ { \pi }$ is a fixed constant given π and $\mu ,$ and define ${ \mathrm { O P T } } _ { \mathrm { L P } } = \operatorname* { s u p } _ { \pi } \mathrm { L P } ( \pi ) = T \operatorname* { s u p } _ { \pi } C _ { \pi }$ , which is the optimal value of the fractional relaxation of the optimal dynamic policy. It is easy to check (also proved in Badanidiyuru et al. (2018)) that $\mathrm { O P T } _ { \mathrm { L P } } \geq \mathrm { O P T }$ . Also, the distribution $\pi ^ { * } \in \arg \operatorname* { m a x } _ { \pi } \mathrm { L P } ( \pi )$ is called LP-optimal, and holds following properties

Proposition 8 (Claim 3.4 in Badanidiyuru et al. (2018)). For any latent structure $\mu ,$ there exists a distribution $\pi$ over arms which is LP-optimal for $\mu$ and moreover satisfies the following three properties:

1. $c _ { i } ( \pi ) \leq B / T$ for each resource $i .$

2. π has a support of size at most d.

$I f \pi$ has a support of size exactly 2 then for some resource $i$ we have $c _ { i } ( \pi ) = B / T$

Such distribution $\pi$ will be called LP-perfect for $\mu .$

We also follow the same high-probability event given in Badanidiyuru et al. (2018) as

Proposition 9 (Theorem 2.1 in Badanidiyuru et al. (2018)). With confidence radius defined to be

$$
r a d ( \nu , N ) = \sqrt { \frac { C _ { r a d } \nu } { N } } + \frac { C _ { r a d } } { N } ,
$$

with $C _ { r a d } = \Theta ( \log ( d T | \mathcal { X } | ) )$ . Then, let νˆ be the average of N independent samples from a distribution with expectation $\nu ,$ then

$$
P r [ | \nu - \hat { \nu } | \leq r a d ( \hat { \nu } , N ) \leq 3 r a d ( \nu , N ) ] \geq 1 - e ^ { - \Omega ( C _ { r a d } ) } , \quad f o r e a c h C _ { r a d } > 0 .
$$

We recall BalancedExploration given in Badanidiyuru et al. (2018), which is summarized in Algorithm 2. $\Delta _ { q }$ ’s are sets of all distribution $\pi \mathrm { { ^ { * } s } }$ in each phase $q$ that are LP-perfect for some latent structure $\mu ^ { \prime } \in I _ { q }$ . For phase $q = 0$ , let $I _ { 0 }$ be the set of feasible latent structures; for each $q \geq 1$ , the confidence interval $I _ { q }$ is the set of all $\mu ^ { \prime } \in I _ { q - 1 }$ that satisfy

$$
| r ( x , \mu ^ { \prime } ) - \hat { r } _ { q } ( x ) | \leq r a d ( \hat { r } _ { q } ( x ) , N _ { q } ( x ) ) ,
$$

$$
| c _ { i } ( x , \mu ^ { \prime } ) - \hat { c } _ { q , i } ( x ) | \leq r a d ( \hat { c } _ { q , i } ( x ) , N _ { q } ( x ) ) , \quad \forall i \in [ d ] .
$$

Algorithm 2 BalancedExploration   
for each phase $q = 0 , 1 , 2 , \ldots$ . do   
Recompute the set $\Delta _ { q }$ of potentially perfect distributions over arms.   
for each $X \in { \mathcal { X } } ,$ over the next $| \mathcal { X } |$ rounds do   
Pick any distribution $\pi _ { q , X } \in \Delta _ { q }$ such that $\pi _ { q , X } ( X ) \geq { \frac { 1 } { 2 } } \operatorname* { m a x } _ { \pi ^ { \prime } \in \Delta _ { q } } \pi ^ { \prime } ( X )$   
Pull an arm as an independent sample from $\pi _ { q , X }$   
Halt if time horizon is met or one of the resources is exhausted.   
end for   
end for   
For this standalone subsection, the expected strong reward is $\mathrm { R E W } _ { S } \quad =$   
$\mathbb { E } [ \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \{ r ( \pi _ { t } ) , \mathrm { O P T _ { L P } } / T \} ]$

Theorem 4. Algorithm BalancedExploration given in Badanidiyuru et al. (2018) has an expected strong regret bound the same as its standard regret bound in expectation to be

$$
O P T _ { L P } - R E W _ { S } \leq C _ { 1 } T ^ { 1 / 2 } \log ( T ) ^ { 2 } ,
$$

where $C _ { 1 } = O ( ( 1 + b ^ { - 1 / 2 } ) \sqrt { d | \mathcal { X } | } \operatorname* { s u p } _ { \pi } C _ { \pi } )$ is a constant with respect to T for fixed d, b.

Proof. We first work on a clean execution in the sense of Badanidiyuru et al. (2018). The core of our proof is based on the fact that LP value of each phase q designated for arm x can be lower bounded by $\mathrm { O P T _ { L P } }$ as

Proposition 10 (Claim 4.13 in Badanidiyuru et al. (2018)). On clean execution events, for every distribution $\pi _ { q , X }$ chosen by the algorithm in phase q,

$$
L P ( \pi _ { q , x } ) \geq O P T _ { L P } - O \left( O ( T ) r a d \Big ( O P T _ { L P } / T , \frac { q } { d } \Big ) + O \left( O P T _ { L P } \frac { T } { B } \right) r a d \left( \frac { B } { T } , \frac { q } { d } \right) \right) ,
$$

We then introduce some notations to simplify our proof: $\Phi _ { q } : = O ( T )$ rad $\left( \mathrm { O P T } _ { \mathrm { L P } } / T , \frac { q } { d } \right) +$ $O \left( \mathrm { O P T } _ { \mathrm { L P } } { \frac { T } { B } } \right)$ rad $\textstyle \left( { \frac { B } { T } } , { \frac { q } { d } } \right)$ , i.e., Proposition 10 implies $\mathrm { L P } ( \pi _ { q , X } ) \ge \mathrm { O P T } _ { \mathrm { L P } } - O ( \Phi _ { q } )$ . Let p be the last phase in the execution and $T _ { 0 }$ be the stopping time. Letting $m : = | { \mathcal { X } } |$ , we have pm $< T _ { 0 } \le ( p + 1 ) m$

For strong reward $\mathrm { R E W } _ { S } ,$ it can be bounded by neglecting the final incomplete phase and only focusing on the first p rounds as

$$
\sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ r ( \pi _ { t } ) , \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { L P } } \right\} \geq \sum _ { q < p } \sum _ { X \in \mathcal { X } } \operatorname* { m i n } \left\{ r ( \pi _ { q , X } ) , \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { L P } } \right\}\tag{54}
$$

where $\begin{array} { r } { \mathbb { E } _ { X ^ { \prime } \sim \pi _ { q , X } } \left[ r ( X ^ { \prime } ) \right] = \sum _ { X ^ { \prime } \in \mathcal { X } } \pi _ { q , X } ( X ^ { \prime } ) r ( X ^ { \prime } ) = r ( \pi _ { q , X } ) } \end{array}$ and $T _ { 0 } \leq T$ is embedded in the second inequality.

Recall that $\mathrm { L P } ( \pi _ { q , X } ) = r ( \pi _ { q , X } ) \operatorname* { m i n } _ { i } B / c _ { i } ( \pi _ { q , X } ) \leq r ( \pi _ { q , X } ) \cdot T$ , then

$$
r \big ( \pi _ { q , X } \big ) \geq \frac { 1 } { T } \mathrm { L P } ( \pi _ { q , X } ) \geq \frac { 1 } { T } \big ( \mathrm { O P T } _ { \mathrm { L P } } - O ( \Phi _ { q } ) \big ) ,
$$

where the second inequality is based on Proposition 10. Note that RHS is irrelevant to which arm X is designated for. We plug this into Equation 54, and get

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ r ( \pi _ { t } ) , \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { L P } } \right\} \geq \displaystyle \sum _ { q < p } \displaystyle \sum _ { X \in \boldsymbol { x } } \operatorname* { m i n } \left\{ \frac { 1 } { T } ( \mathrm { O P T } _ { \mathrm { L P } } - O ( \Phi _ { q } ) ) , \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { L P } } \right\} } \\ { \displaystyle \quad \quad = m \displaystyle \sum _ { q < p } \left( \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { L P } } - \frac { 1 } { T } O ( \Phi _ { q } ) \right) } \\ { \displaystyle \quad \geq \frac { p m } { T } ( \mathrm { O P T } _ { \mathrm { L P } } - O ( \Phi _ { p } \log p ) ) } \end{array}\tag{55}
$$

where the last inequality follows from Claim $4 . 6 ( \mathrm { f g } )$ in Badanidiyuru et al. (2018), which is used to average the confidence radii in $\Phi _ { q }$ over all phases $q < p$

Till now, we get exactly the same lower bound expression on $\mathrm { R E W } _ { S }$ as the one in proving standard reward in Equation (22), in Badanidiyuru et al. (2018). As the diference between strong and standard reward has nothing to do with the consumption part, we directly use the bound on pm/T given in Badanidiyuru et al. (2018) as

Proposition 11 (Proof of Lemma 4.17 with Corollary 4.16 in Badanidiyuru et al. (2018)). The number of complete phases p can be lower bounded by

$$
\frac { p m } { T } \geq 1 - O \left( \frac { p m \log p } { B } \right) r a d \left( \frac { B } { T } , \frac { p } { d } \right) .
$$

Plug Proposition 11 into Equation (55), we conclude:

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \bigg \{ r ( \pi _ { t } ) , \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { I , P } } \bigg \} \geq \mathrm { O P T } _ { \mathrm { I , P } } - O \left( \frac { p m \log p } { B } \right) \mathrm { r a d } \left( \frac { B } { T } , \frac { p } { d } \right) \mathrm { O P T } _ { \mathrm { I , P } } - O \left( \frac { p m \log p } { T } \right) \cdot O ( \Phi _ { p } ) } \\ & { \qquad \geq \mathrm { O P T } _ { \mathrm { L P } } - O \left( \frac { p m \log p } { T } \right) \left( O ( \Phi _ { p } ) + \frac { T } { B } \mathrm { r a d } \left( \frac { B } { T } , \frac { p } { d } \right) \mathrm { O P T } _ { \mathrm { L P } } \right) } \\ & { \qquad \geq \mathrm { O P T } _ { \mathrm { L P } } - \frac { p m \log p } { T } O ( \Phi _ { p } ) , } \end{array}
$$

where the final inequality is because $\Phi _ { q } = { \cal O } ( T )$ rad $\begin{array} { r } { \left( \mathrm { O P T } _ { \mathrm { L P } } / T , \frac { q } { d } \right) + O \left( \mathrm { O P T } _ { \mathrm { L P } } \frac { T } { B } \right) \mathrm { r a d } \left( \frac { B } { T } , \frac { q } { d } \right) \leq } \end{array}$ $O ( \Phi _ { q } )$

Note that $p \Phi _ { p }$ log p is increasing in $p$ (by definition of $\Phi _ { p } )$ , and plug in the upper bound $\begin{array} { r } { p \leq { \frac { T } { m } } } \end{array}$ and we have

$$
\sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ r ( \pi _ { t } ) , \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { L P } } \right\} \geq \mathrm { O P T } _ { \mathrm { L P } } - O \left( \log \frac { T } { | \mathcal { X } | } \right) \Phi _ { T / | \mathcal { X } | } .
$$

Plugging into the definition of $\Phi _ { T / | \mathcal { X } | }$ , on the clean event we obtain

$$
\begin{array} { r l } & { \mathrm { O P T } _ { \mathrm { L P } } - \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ r ( \pi _ { t } ) , \frac { 1 } { T } \mathrm { O P T } _ { \mathrm { L P } } \right\} \leq O ( \log ( T ) \log ( T / m ) ) \left( \sqrt { d m \mathrm { O P T } _ { \mathrm { L P } } } + \mathrm { O P T } _ { \mathrm { L P } } \sqrt { \frac { d m } { B } } \right) } \\ & { \qquad \leq O \left( \sqrt { d m } \left( 1 + b ^ { - 1 / 2 } \right) T ^ { 1 / 2 } \log ( T ) ^ { 2 } \right) , } \end{array}\tag{56}
$$

where the second inequality follows from $\mathrm { O P T } _ { \mathrm { L P } } \leq T$ and $B \geq b T$ . Finally, the clean event fails with probability at most $T ^ { - 2 }$ , and the strong regret is always at most $\mathrm { O P T } _ { \mathrm { L P } } \leq T$ . Therefore, taking expectation gives

$$
\begin{array} { r l } & { \mathrm { O P T } _ { \mathrm { L P } } - \mathrm { R E W } _ { S } \leq O \left( \sqrt { d m } \left( 1 + b ^ { - 1 / 2 } \right) T ^ { 1 / 2 } \log ( T ) ^ { 2 } \right) + \displaystyle \frac { 1 } { T } } \\ & { ~ = O \left( \sqrt { d m } \left( 1 + b ^ { - 1 / 2 } \right) T ^ { 1 / 2 } \log ( T ) ^ { 2 } \right) . } \end{array}\tag{57}
$$

## D.2. Dynamic Pricing with Resource Constraints

We consider a discrete-time version of bandit setting given in Wang et al. (2014). Similar to Wang et al. (2014), we consider a seller selling a single product with a fixed inventory at beginning to be B within a time budget T. Instead of customers arriving according to a Poisson process with an instantaneous demand rate $\lambda _ { t }$ at any time $t \in [ 0 , T ]$ in Wang et al. (2014), we consider customers arriving according to a Bernoulli process with an arriving probability $\lambda _ { t }$ at each time step $t \in [ T ]$

The same as Wang et al. (2014), the arriving probability is solely determined by the price ofered at t, namely we can express $\lambda _ { t } = \lambda ( p _ { t } )$ where $p _ { t }$ is the ofered price at time t. Then, the revenue rate function can be defined as $r _ { t } = r ( p _ { t } ) = p _ { t } \lambda ( p _ { t } )$ . We also denote the inverse function of λ as $p = \gamma ( \lambda )$ , then revenue can also be expressed in λ as $r ( \lambda ) = \lambda \gamma ( \lambda )$ . We assume feasible prices exist in an interval $[ \underline { { p } } , \bar { p } ] , \lambda ( p )$ is decreasing in $p ,$ and r is concave in λ. The stop price $p _ { \infty }$ corresponds to stopping selling, namely $\lambda ( p _ { \infty } ) = 0$ . We also follow assumptions on $\lambda ( p )$ and $r ( \lambda )$ in Wang et al. (2014) as

Proposition 12 (Assumption 1 in Wang et al. (2014)). For some positive constants $M , K , m _ { L }$ and $m _ { U }$

1. Boundedness: $| \lambda ( p ) | \leq M$ for all $p \in [ \underline { { p } } , \overline { { p } } ]$ ;

2. Lipschitz continuity: $\lambda ( p )$ and $r ( \lambda ( p ) )$ are Lipschitz continuous with respect to p with a factor K.

$$
p = \gamma ( \lambda )
$$

$$
K
$$

3. Strict concavity and diferentiability: $r ^ { \prime \prime } ( \lambda )$ exists and $- m _ { L } \le r ^ { \prime \prime } ( \lambda ) \le - m _ { U } < 0$ for all λ in the range of $\lambda ( p )$ for $p \in [ p , { \overline { { p } } } ]$

In order to evaluate the algorithm, Wang et al. (2014) continues to use the expected revenue introduced by Besbes and Zeevi (2009), defined as $\begin{array} { r } { \int _ { 0 } ^ { T } p ( t ) \lambda ( p ( t ) ) \mathrm { d } t } \end{array}$ subject to the inventory constraint $\begin{array} { r } { \int _ { 0 } ^ { T } \lambda ( p ( t ) ) \mathrm d t \leq B } \end{array}$ . This expected revenue is compared to a full-information deterministic optimization problem where demand λ is known, and the optimal (deterministic) solution is given as

Proposition 13 (According to Wang et al. (2014), Besbes and Zeevi (2009)).

$$
p ( t ) = p ^ { D } = \operatorname* { m a x } \{ p ^ { u } , p ^ { c } \}
$$

where $p ^ { u } = \mathrm { a r g m a x } _ { p \in [ \underline { { p } } , \bar { p } ] } \{ r ( \lambda ( p ) ) \}$ and $p ^ { c } = \mathrm { a r g } \mathrm { m i n } _ { p \in [ \underline { { p } } , \bar { p } ] } | \lambda ( p ) - B / T |$

Diferent to Wang et al. (2014), we define a discrete version of strong reward function for each policy $\pi$ as $\begin{array} { r } { \mathrm { R E W } _ { S } ^ { \pi } ( B , T ; \lambda ) = \mathbb { E } [ \sum _ { t = 1 } ^ { T } } \end{array}$ min $\{ p ( t ) \lambda ( p ( t ) ) , p ^ { D } \lambda ( p ^ { D } ) \} ]$ where the expectation is over price trajectory (for a stochastic policy), and it will be compared to the optimal (strong) reward $\mathrm { R E W } _ { S } ^ { D } ( B , T ; \lambda ) = T p ^ { D } \lambda ( p ^ { D } )$ , which is reward of applying full-information price $p ^ { D }$ at each time step t. While in Wang et al. (2014) the asymptotic analysis is based on $B _ { n } = n B$ and $\lambda _ { n } ( p ) = n \lambda ( p )$

with large $n$ and fixed $T = 1$ , we consider inventory $B$ that grows linearly with total budget $T$ following Assumption 1, i.e., there exists b such that $B \geq b T$ , and $\lambda ( p )$ remains the same. For notational simplicity, we define $\mathrm { R E W } _ { S } : = \mathrm { R E W } _ { S } ^ { \pi } ( B , T ; \lambda ) , \mathrm { O P T } : = \mathrm { R E W } _ { S } ^ { D } ( B , T ; \lambda )$ , and we want to bound the regret $\mathrm { O P T } - \mathrm { R E W } _ { S }$ . In particular, ${ \mathrm { O P T } } = T p ^ { D } \lambda ( p ^ { D } )$ . Therefore

$$
\mathrm { O P T - R E W } _ { S } = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \operatorname* { m a x } \{ Z ^ { * } ( T , B ) - r ( p ( t ) ) , 0 \} \right] ,
$$

which is exactly the expected strong regret in Condition 2.

We will use and analyze the discrete version of Algorithm DPA given in Wang et al. (2014). Algorithm DPA can be divided into three steps. In the first step, learn $p ^ { u }$ or determine $p ^ { c } > p ^ { u }$ . If $p ^ { c } > p ^ { u }$ , explore $p ^ { c }$ in the second step. Otherwise, skip the second step. After exploration in $p ^ { u }$ and $p ^ { c }$ (if needed), repeat the estimated best price in the final step until exhausting time budget or resources.

To be specific, in both exploration stages, price $p ^ { u }$ or $p ^ { c }$ are estimated round by round. In each round, learning period $\tau _ { i } ^ { u }$ or $\boldsymbol { \tau } _ { i } ^ { c }$ and price confidence interval $I _ { i } ^ { u } : = [ \underline { { { p } } } _ { i } ^ { u } , \bar { p } _ { i } ^ { u } ]$ or $I _ { i } ^ { c } : = [ \underline { { { p } } } _ { i } ^ { c } , \bar { p } _ { i } ^ { c } ]$ are both equally divided into $\kappa _ { i } ^ { u }$ or $\kappa _ { i } ^ { c }$ interval. Because we propose a discrete version, the length of divided time period is rounded up as $\Delta _ { i } ^ { u } : = \lceil T \tau _ { i } ^ { u } / \kappa _ { i } ^ { u } \rceil$ . The left endpoints of each price intervals are repeated for corresponding learning period. Namely, at round i for $j = 1 , 2 , \dots , \kappa _ { i } ^ { u }$ in exploring $p ^ { u }$ , we set price and time endpoints to be

$$
\begin{array} { r l } & { p _ { i , j } ^ { u } = \underline { { p } } _ { i } ^ { u } + ( j - 1 ) \cdot \frac { \bar { p } _ { i } ^ { u } - \underline { { p } } _ { i } ^ { u } } { \kappa _ { i } ^ { u } } } \\ & { t _ { i , j } ^ { u } = t _ { i - 1 } ^ { u } + j \Delta _ { i } ^ { u } , } \end{array}\tag{58}
$$

and at round i for $j = 1 , 2 , \ldots , \kappa _ { i } ^ { c }$ in exploring $p ^ { c }$ , we set price and time endpoints to be

$$
\begin{array} { r l } & { p _ { i , j } ^ { c } = \underline { { p } } _ { i } ^ { c } + ( j - 1 ) \cdot \frac { \bar { p } _ { i } ^ { c } - \underline { { p } } _ { i } ^ { c } } { \kappa _ { i } ^ { c } } } \\ & { t _ { i , j } ^ { c } = t _ { i - 1 } ^ { c } + j \Delta _ { i } ^ { c } + t _ { i _ { 0 } } ^ { u } . } \end{array}\tag{59}
$$

In each round $i ,$ after exploring $\kappa _ { i } ^ { u }$ or $\kappa _ { i } ^ { c }$ prices, the demand of j-th price is estimated to be

$$
\hat { d } ( p _ { i , j } ^ { u } ) = \frac { \mathrm { t o t a l ~ d e m a n d ~ o v e r ~ } [ t _ { i , j - 1 } ^ { u } , t _ { i , j } ^ { u } ] } { \Delta _ { i } ^ { u } } .
$$

Then, estimate $p ^ { u }$ (and $p ^ { c } )$ , check if explore enough, and update price interval for round $i + 1$ This is slightly diferent between exploring $p ^ { u }$ and $p ^ { c }$ . In exploring $p ^ { u } , p ^ { u }$ and $p ^ { c }$ are estimated to be

$$
\begin{array} { r l } & { \hat { p } _ { i } ^ { u } = \arg \underset { p \in \{ p _ { i , j } ^ { u } : j \in [ \kappa _ { i } ^ { u } ] \} } { \operatorname* { m a x } } p \hat { d } ( p ) } \\ & { \quad \quad \hat { p } _ { i } ^ { c } = \arg \underset { p \in \{ p _ { i , j } ^ { u } : j \in [ \kappa _ { i } ^ { u } ] \} } { \operatorname* { m i n } } \Big | \hat { d } ( p ) - B / T \Big | , } \end{array}\tag{60}
$$

and price interval $I _ { i + 1 } ^ { u }$ for round $i + 1$ is updated to be

$$
\begin{array} { r l } & { \underline { { p } } _ { i + 1 } ^ { u } = \hat { p } _ { i } - \displaystyle \frac { \log ( T ) } { 3 } \cdot \frac { \bar { p } _ { i } ^ { u } - \underline { { p } } _ { i } ^ { u } } { \kappa _ { i } ^ { u } } } \\ & { \bar { p } _ { i + 1 } ^ { u } = \hat { p } _ { i } + \displaystyle \frac { 2 \log ( T ) } { 3 } \cdot \displaystyle \frac { \bar { p } _ { i } ^ { u } - \underline { { p } } _ { i } ^ { u } } { \kappa _ { i } ^ { u } } . } \end{array}\tag{61}
$$

In exploring $p ^ { c } , p ^ { c }$ is estimated to be

$$
\hat { q } _ { i } \in \arg \operatorname* { m i n } _ { p \in \{ p _ { i , j } ^ { c } : j \in [ \kappa _ { i } ^ { c } ] \} } \left| \hat { d } ( p ) - B / T \right| ,\tag{62}
$$

and price interval $I _ { i + \cdot } ^ { c }$ <sub>1</sub> for round $i + 1$ is updated to be

$$
\begin{array} { r } { \underline { { p } } _ { i + 1 } ^ { c } = \hat { q } _ { i } - \frac { \log ( T ) } { 2 } \cdot \frac { \bar { p } _ { i } ^ { c } - \underline { { p } } _ { i } ^ { c } } { \kappa _ { i } ^ { c } } } \\ { \bar { p } _ { i + 1 } ^ { c } = \hat { q } _ { i } + \frac { \log ( T ) } { 2 } \cdot \frac { \bar { p } _ { i } ^ { c } - \underline { { p } } _ { i } ^ { c } } { \kappa _ { i } ^ { c } } . } \end{array}\tag{63}
$$

We summarize the Discrete DPA in Algorithm 3.

Algorithm 3 depends on pre-designed sequences of $\tau _ { i } ^ { u } , \kappa _ { i } ^ { u } , \tau _ { i } ^ { c } , \kappa _ { i } ^ { c } .$ , and value $N ^ { u } , N ^ { c }$ . We use the same design as the one given in Wang et al. (2014) (Equations 18 to 31) so that some of the results can be directly borrowed. Note that an important diference between Algorithm 3 and DPA given in Wang et al. (2014) is that each price $p _ { i , j } ^ { u }$ is applied for $\lceil T \tau _ { i } ^ { u } / \kappa _ { i } ^ { u } \rceil$ time steps instead of a continuous time interval $\tau _ { i } ^ { u } / \kappa _ { i } ^ { u }$ . In the implementation, each displayed $\kappa _ { i } ^ { u }$ and $\kappa _ { i } ^ { c }$ is rounded to a positive integer. This changes all bounds below by constant factors only, so we keep the same notation.

$$
\tau _ { 1 } ^ { u } = T ^ { - \frac { 1 } { 2 } } \cdot \log ( T ) ^ { 3 . 5 } \quad \mathrm { a n d } \quad \tau _ { i } ^ { u } = T ^ { - \frac { 1 } { 2 } \cdot \left( \frac { 3 } { 5 } \right) ^ { i - 1 } } \cdot \log ( T ) ^ { 5 } , \quad \forall i = 2 , \ldots , N ^ { u } ,
$$

$$
\kappa _ { i } ^ { u } = T ^ { \frac { 1 } { 1 0 } \cdot \left( \frac { 3 } { 5 } \right) ^ { i - 1 } } \cdot \log ( T ) , \quad \forall i = 1 , 2 , \ldots , N ^ { u } ,
$$

$$
\bar { p } _ { i } ^ { u } - \underline { { p } } _ { i } ^ { u } = T ^ { - \frac { 1 } { 4 } \left( 1 - \left( \frac { 3 } { 5 } \right) ^ { i - 1 } \right) } , \quad \forall i = 1 , 2 , \dots , N ^ { u } ,
$$

$$
\mathrm { w h e r e \ } N ^ { u } : = \operatorname* { m i n } _ { l } \left\{ l : \left( \frac { \bar { p } _ { l } ^ { u } - \underline { { p } } _ { l } ^ { u } } { \kappa _ { l } ^ { u } } \right) ^ { 2 } \sqrt { \log ( T ) } < \tau _ { 1 } ^ { u } \right\} .
$$

Similarly, for $p ^ { c }$ , we have

$$
\tau _ { 1 } ^ { c } = T ^ { - \frac { 1 } { 2 } } \cdot \log ( T ) ^ { 2 . 5 } \quad \mathrm { a n d } \quad \tau _ { i } ^ { c } = T ^ { - \frac { 1 } { 2 } \cdot \left( \frac { 2 } { 3 } \right) ^ { i - 1 } } \cdot \log ( T ) ^ { 3 } , \quad \forall i = 2 , \ldots , N ^ { c } ,
$$

$$
\kappa _ { i } ^ { c } = T ^ { \frac { 1 } { 6 } \cdot \left( \frac { 2 } { 3 } \right) ^ { i - 1 } } \cdot \log ( T ) , \quad \forall i = 1 , 2 , \ldots , N ^ { c } .
$$

$$
\bar { p } _ { i } ^ { c } - \underline { { p } } _ { i } ^ { c } = T ^ { - \frac { 1 } { 2 } \left( 1 - \left( \frac { 2 } { 3 } \right) ^ { i - 1 } \right) } , \quad \forall i = 1 , \dots , N ^ { c } .
$$

$$
\mathrm { w h e r e } \ N ^ { c } { : = } \operatorname* { m i n } _ { l } \left\{ l : \frac { \bar { p } _ { l } ^ { c } - \underline { { p } } _ { l } ^ { c } } { \kappa _ { l } ^ { c } } \sqrt { \log ( T ) } < \tau _ { 1 } ^ { c } \right\} .
$$

For $T < 3$ , define Discrete DPA to post $p _ { \infty }$ throughout; below we analyze $T \geq 3$

Algorithm 3 Discrete DPA   
input: a sequence of $\tau _ { i } ^ { u } , \kappa _ { i } ^ { u }$ for $i = 1 , 2 , \dots , N ^ { u }$ and $\tau _ { i } ^ { c } , \kappa _ { i } ^ { c }$ for $i = 1 , 2 , \ldots , N ^ { c }$   
Step 1: ▷ Learn $p ^ { u }$ or determine $p ^ { c } > p ^ { u }$   
for $i = 1 , 2 , \dots , N ^ { u }$ do   
Set $p _ { i , j } ^ { u }$ and $t _ { i , j } ^ { u }$ for $j = 0 , 1 , \ldots , \kappa _ { i } ^ { u }$ according to Equations (58)   
for $j = 1 , 2 , \ldots , \kappa _ { i } ^ { u }$ do   
Apply $p _ { i , j } ^ { u }$ for $\Delta _ { i } ^ { u } = \lceil T \tau _ { i } ^ { u } / \kappa _ { i } ^ { u } \rceil$ steps; Or apply $p _ { \infty }$ until time $T$ if inventory runs out   
end for   
Estimate $\hat { p } _ { i } ^ { u }$ and $\hat { p } _ { i } ^ { c }$ according to Equations (60)   
if $\begin{array} { r } { \hat { p } _ { i } ^ { c } > \hat { p } _ { i } ^ { u } + 2 \sqrt { \log ( T ) } \cdot \frac { \bar { p } _ { i } ^ { u } - \underline { { p } } _ { i } ^ { u } } { \kappa _ { i } ^ { u } } } \end{array}$ then   
Set $i _ { 0 } = i .$ Go to Step 2.   
else   
Set $\hat { p } _ { i } = \operatorname* { m a x } \{ \hat { p } _ { i } ^ { c } , \hat { p } _ { i } ^ { u } \}$ and update $I _ { i + 1 } ^ { u }$ according to Equations (61)   
end if   
if $i = N ^ { u }$ then   
Go to Step 3 with learned price $\begin{array} { r } { \tilde { p } = { \hat { p } _ { N ^ { u } } } + 2 \sqrt { \log ( T ) } \cdot \frac { \bar { p } _ { N ^ { u } } ^ { u } - \underline { { p } } _ { N ^ { u } } ^ { u } } { \kappa _ { N ^ { u } } ^ { u } } } \end{array}$   
end if   
end for   
Step 2: ▷ Learn $p ^ { c }$ when $p ^ { c } > p ^ { u }$   
for $i = 1 , 2 , \ldots , N ^ { c }$ do   
Set $p _ { i , j } ^ { c }$ and $t _ { i , j } ^ { c }$ for $j = 0 , 1 , \ldots , \kappa _ { i } ^ { c }$ according to Equations (59)   
for $j = 1 , 2 , \ldots , \kappa _ { i } ^ { c }$ do   
Apply $p _ { i , j } ^ { c }$ for $\Delta _ { i } ^ { c } = \lceil T \tau _ { i } ^ { c } / \kappa _ { i } ^ { c } \rceil$ steps; Or apply $p _ { \infty }$ until time T if inventory runs out   
end for   
Estimate $\hat { q } _ { i }$ according to Equations (62) and update $I _ { i + 1 } ^ { c }$ according to Equations (63)   
if $i = N ^ { c }$ then   
Go to Step 3 with learned price $\tilde { q } = \hat { q } _ { N ^ { c } }$   
end if   
end for   
Step 3: ▷ Apply learned price   
Apply the learned price (˜p or ˜q) until inventory or time budget run out

Theorem 5. Algorithm 3 Discrete DPA has an expected strong regret bound as

$$
O P T - R E W _ { S } \leq C \sqrt { T } \log ( T ) ^ { 6 } .\tag{64}
$$

Proof. We first bound the round number $N ^ { u }$ and $N ^ { c }$ by

Proposition 14 (Lemma 2 in Wang et al. (2014)). $N ^ { u }$ and $N ^ { c }$ defined above satisfy: $N ^ { u } \leq 3 \log ( T )$ and $N ^ { c } \leq 3 \log ( T )$ , for $T \geq 3$

In order to make sure that $p ^ { D }$ is always contained in the price interval with high probability, a concentration inequality should first be derived as

Proposition 15 (Lemma 11 in Wang et al. (2014)). The arrival rate $\hat { \lambda }$ for each price $p _ { i , j } ^ { u }$ can be estimated with high accuracy to be

$$
P r \left( \left| \hat { \lambda } ( p _ { i , j } ^ { u } ) - \lambda ( p _ { i , j } ^ { u } ) \right| > C \sqrt { \log ( T ) } \cdot \sqrt { \frac { \kappa _ { i } ^ { u } } { T \tau _ { i } ^ { u } } } \right) \leq \frac { 1 } { T ^ { 2 } } .
$$

The same inequality holds for $p _ { i , j } ^ { c }$ with $\kappa _ { i } ^ { c }$ and $\boldsymbol { \tau } _ { i } ^ { c }$ .

Although this has exactly the same form as the one in Wang et al. (2014), it is worth mentioning that its derivation is slightly diferent. In original paper, the analysis is based on estimating $n \lambda ( p _ { i , j } ^ { u } )$ within a time period $\tau _ { i } ^ { u } / \kappa _ { i } ^ { u }$ asymptotic in n. But in our algorithm, we estimate $\lambda ( p _ { i , j } ^ { u } )$ within a time $\lceil T \tau _ { i } ^ { u } / \kappa _ { i } ^ { u } \rceil$ asymptotic in $T ,$ , which can lead to a stronger bound than estimating within a time $T \tau _ { i } ^ { u } / \kappa _ { i } ^ { u }$ (which is equivalent to the one in original paper). We loosen this stronger bound and retain the same form. Then, we propose the high probability event as

Proposition 16 (Lemma 3 in Wang et al. (2014)). Assume $p ^ { D }$ is the optimal price for the deterministic problem and Assumption 1 holds for $\Gamma = \Gamma ( M , K , m _ { L } , m _ { U } )$ . Define $A _ { 0 }$ to be the event satisfying the following conditions:

1. If we never enter Step 2, then $p ^ { D } \in I _ { i } ^ { u }$ for all $i = 1 , 2 , \dots , N ^ { u }$

2. If Step 1 stops at $i _ { 0 }$ and the algorithm enters Step 2, then $p ^ { c } \geq p ^ { u } , p ^ { D } \in I _ { i } ^ { u }$ for all $i = 1 , 2 , \ldots , i _ { 0 }$ and $p ^ { D } \in I _ { j } ^ { c }$ for all $j = 1 , 2 , \ldots , N ^ { c }$

Then $\begin{array} { r } { P r ( A _ { 0 } ) = 1 - O \left( \frac { 1 } { T } \right) } \end{array}$

We also use the events defined in Wang et al. (2014): define $B _ { k } : = \{ i _ { 0 } = k \}$ for $k \in [ N ^ { u } ]$ to denote the event that Step 1 jumps to Step 2 at round $k ,$ and define $B _ { N ^ { u } + 1 } \ =$ {Algorithm never enters Step 2}; $\begin{array} { r } { A _ { 1 } \ : = \ \Big \{ \omega \in A _ { 0 } : \sum _ { i = 1 } ^ { N ^ { u } } \sum _ { j = 1 } ^ { \kappa _ { i } ^ { u } } \lambda ( p _ { i , j } ^ { u } ) \Delta _ { i } ^ { u } < B \Big \} , A _ { 2 } \ = } \end{array}$ $\begin{array} { r } { \left\{ \omega \in A _ { 0 } : \sum _ { i = 1 } ^ { N ^ { u } } \sum _ { j = 1 } ^ { \kappa _ { i } ^ { u } } \lambda ( p _ { i , j } ^ { u } ) \Delta _ { i } ^ { u } + \sum _ { i = 1 } ^ { N ^ { c } } \sum _ { j = 1 } ^ { \kappa _ { i } ^ { c } } \lambda ( p _ { i , j } ^ { c } ) \Delta _ { i } ^ { c } < B \right\} } \end{array}$

We also provide some important relationships among the parameters

Proposition 17 (Equations 18 to 25 in Wang et al. (2014)). When entering Step ${ \mathcal { B } } ,$ the length of updated price intervals can be bounded by

$$
( \bar { p } _ { N ^ { u } + 1 } ^ { u } - \underline { { p } } _ { N ^ { u } + 1 } ^ { u } ) ^ { 2 } \leq C \tau _ { 1 } ^ { u }
$$

$$
\bar { p } _ { N ^ { c } + 1 } ^ { c } - \underline { { p } } _ { N ^ { c } + 1 } ^ { c } \le C \tau _ { 1 } ^ { c }
$$

Denote the clipped strong reward $g ( p ) : = \mathrm { m i n } \{ p \lambda ( p ) , p ^ { D } \lambda ( p ^ { D } ) \}$ and then $\rho ( p ) : = g ( p ) / \lambda ( p )$ $\mathbf { 1 } ( \lambda ( p ) > 0 )$ . Then $0 \leq \rho ( p ) \leq p \leq \bar { p }$ and $\rho ( p ) \lambda ( p ) = g ( p )$ . For each exploration block, let $Y _ { i , j } ^ { u }$ and $Y _ { i , j } ^ { c }$ denote the corresponding total Bernoulli arrivals. Let $\widehat { Y } ^ { u }$ denote the Bernoulli arrivals that would occur during the final $\tilde { p }$ period, and let $\widehat { Y } _ { l } ^ { c }$ denote the arrivals that would occur during the final $\tilde { q }$ period on $B _ { l }$ . Then, the total number of exploration steps is upper bounded by $T \sum _ { i } ( \tau _ { i } ^ { u } + \tau _ { i } ^ { c } ) + \sum _ { i } ( \kappa _ { i } ^ { u } + \kappa _ { i } ^ { c } ) = o ( T )$ . Hence, because each Bernoulli arrival consumes at most one unit and $B \geq b T$ , for all suficiently large T inventory cannot be exhausted during exploration. We prove the bound below for such T and all smaller horizons are absorbed by increasing the final constant $C _ { 1 }$

On $A _ { 0 }$ , the deterministic comparisons used in Lemmas 4–7 of Wang et al. (2014) give

Proposition 18 (Lemmas 4–7 in Wang et al. (2014)). On $A _ { 0 }$ , the deterministic comparisons give

$$
g ( p _ { i , j } ^ { u } ) \geq p ^ { D } \lambda ( p ^ { D } ) - C ( { \bar { p } } _ { i } ^ { u } - \underline { { p } } _ { i } ^ { u } ) ^ { 2 } , f o r e v e r y r e a c h e d u - r o u n d ,
$$

$$
g ( \tilde { p } ) \geq p ^ { D } \lambda ( p ^ { D } ) - C \tau _ { 1 } ^ { u } , \mathrm { ~ ~ } o n ~ B _ { N ^ { u } + 1 } ,
$$

$$
g ( p _ { i , j } ^ { c } ) \geq p ^ { D } \lambda ( p ^ { D } ) - C ( \bar { p } _ { i } ^ { c } - \underline { { p } } _ { i } ^ { c } ) ,
$$

after switching to Step 2,

$$
g ( \tilde { q } ) \geq p ^ { D } \lambda ( p ^ { D } ) - C \tau _ { 1 } ^ { c } , \mathrm { ~ } o n \bigcup _ { l = 1 } ^ { N ^ { u } } B _ { l } .\tag{65}
$$

Proposition 19 (Discrete counterparts of Lemmas 5 and 7 in Wang et al. (2014)). The inventoryoverrun terms satisfy

$$
\mathbb { E } \left[ \left( \widehat { Y } ^ { u } + \sum _ { i = 1 } ^ { N ^ { u } } \sum _ { j = 1 } ^ { \kappa _ { i } ^ { u } } Y _ { i , j } ^ { u } - B \right) ^ { + } I ( B _ { N ^ { u } + 1 } ) \right] \leq C \sqrt { T } \log ^ { 6 } T ,
$$

$$
\sum _ { l = 1 } ^ { N ^ { u } } \mathbb { E } \left[ \left( \widehat { Y } _ { l } ^ { c } + \sum _ { i = 1 } ^ { l } \sum _ { j = 1 } ^ { \kappa _ { i } ^ { u } } Y _ { i , j } ^ { u } + \sum _ { i = 1 } ^ { N ^ { c } } \sum _ { j = 1 } ^ { \kappa _ { i } ^ { c } } Y _ { i , j } ^ { c } - B \right) ^ { + } I ( B _ { l } ) \right] \leq C \sqrt { T } \log ^ { 6 } T .\tag{66}
$$

The predictable-mean part is the same as in the proofs of Lemmas 5 and 7, because it only uses the demand probabilities and the DPA price path. Replacing Poisson arrivals by Bernoulli arrivals changes the stochastic part to a martingale with conditional variance at most $T ,$ so its expected absolute fluctuation is at most $\sqrt { T }$ . Finally, the ceiling operation contributes at most $C \sum _ { i } ( \kappa _ { i } ^ { u } + \kappa _ { i } ^ { c } ) \le C T ^ { 1 / 6 } \log ^ { 2 } T$ . All three terms are bounded by the displayed right-hand side.

Similar to Equation (32) in Wang et al. (2014), the strong reward can be bounded by

$$
\begin{array} { r l } & { \mathrm { R E W } _ { S } \geq \mathbb { E } \left[ \displaystyle \sum _ { i < 1 , j = 1 } ^ { N ^ { \alpha } } g ( p _ { i , j } ^ { \alpha } ) \Delta _ { i } ^ { \alpha } I \left( \displaystyle \bigcup _ { i = i - 1 } ^ { N ^ { \alpha } + 1 } B _ { i } \right) \right] } \\ & { \qquad + \mathbb { E } \left[ \rho ( \vec { p } ) \operatorname* { m i n } \left\{ \widehat { Y } ^ { \alpha } , \left( B - \displaystyle \sum _ { i = 1 } ^ { N ^ { \alpha } } \sum _ { j = 1 } ^ { N ^ { \alpha } } Y _ { i , j } ^ { \alpha } \right) ^ { + } \right\} I ( B _ { N ^ { \alpha } + 1 } ) \right] } \\ & { \qquad + \mathbb { E } \left[ \displaystyle \sum _ { i < 1 , j = 1 } ^ { N ^ { \alpha } } g ( p _ { i , j } ^ { \alpha } ) \Delta _ { i } ^ { \alpha } I \left( \displaystyle \prod _ { i = 1 } ^ { N ^ { \alpha } } B _ { i } \right) \right] } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { N ^ { \alpha } } \mathbb { E } \left[ \rho ( \vec { q } ) \operatorname* { m i n } \left\{ \widehat { Y } _ { i } ^ { \alpha } , \left( B - \displaystyle \sum _ { i = 1 } ^ { N ^ { \alpha } } \sum _ { j = 1 } ^ { N _ { i } ^ { \alpha } } - \displaystyle \sum _ { i = 1 } ^ { N ^ { \alpha } } \sum _ { j = 1 } ^ { N _ { i } ^ { \alpha } } Y _ { i , j } ^ { \alpha } \right) ^ { + } \right\} I ( B _ { i } ) \right] . } \end{array}\tag{67}
$$

The first term of Equation (67) is the cumulative strong reward in exploring $p ^ { u }$ , denoted as E[Explore $p ^ { u }$ Reward]. The second term is reward in repeating ${ \tilde { p } } ,$ denoted as E[Exploit $\tilde { p }$ Reward]. The third term is reward in exploring $p ^ { c }$ , denoted as E[Explore $p ^ { c }$ Reward]. The fourth term is reward in repeating ${ \tilde { q } } ,$ denoted as E[Exploit $\tilde { q }$ Reward]. Namely, the total reward can be divided into

$$
\begin{array} { r l } & { { \mathrm { R E W } } _ { { \cal S } } \geq \mathbb { E } [ \mathrm { E x p l o r e } \ p ^ { u } \ \mathrm { R e w a r d } ] + \mathbb { E } [ \mathrm { E x p l o i t } \ \tilde { p } \ \mathrm { R e w a r d } ] } \\ & { \quad \quad \quad + \mathbb { E } [ \mathrm { E x p l o r e } \ p ^ { c } \ \mathrm { R e w a r d } ] + \mathbb { E } [ \mathrm { E x p l o i t } \ \tilde { q } \ \mathrm { R e w a r d } ] . } \end{array}\tag{68}
$$

We will bound these four terms one by one.

First term of Equation (68) is the cumulative strong reward in exploring $p ^ { u }$ . We have

$$
\begin{array} { r l } & { \mathbb { E } [ \mathrm { E x p l o r e \ } p ^ { u } \ \mathrm { \normalfont ~ R e w a r d } ] \ge \displaystyle \sum _ { i = 1 } ^ { N ^ { u } } T \tau _ { i } ^ { u } p ^ { D } \lambda ( p ^ { D } ) \operatorname* { P r } \left( \bigcup _ { l = i } ^ { N ^ { u } + 1 } B _ { l } \right) - C T \displaystyle \sum _ { i = 1 } ^ { N ^ { u } } T _ { i } ^ { u } ( \bar { p } _ { i } ^ { u } - { p } _ { i } ^ { u } ) ^ { 2 } - C } \\ & { \qquad = \displaystyle \sum _ { i = 1 } ^ { N ^ { u } } T \tau _ { i } ^ { u } p ^ { D } \lambda ( p ^ { D } ) \operatorname* { P r } \left( \bigcup _ { l = i } ^ { N ^ { u } + 1 } B _ { l } \right) - C \left( \sqrt { T } \log ^ { 3 . 5 } T + ( N ^ { u } - 1 ) \sqrt { T } \log ^ { 5 } T \right) - C } \\ & { \qquad \ge \displaystyle \sum _ { i = 1 } ^ { N ^ { u } } T \tau _ { i } ^ { u } p ^ { D } \lambda ( p ^ { D } ) \operatorname* { P r } \left( \bigcup _ { l = i } ^ { N ^ { u } + 1 } B _ { l } \right) - C \sqrt { T } \log ^ { 6 } T . } \end{array}\tag{69}
$$

The first inequality follows from $\Delta _ { i } ^ { u } \ge T \tau _ { i } ^ { u } / \kappa _ { i } ^ { u }$ , Equation (65), and $\operatorname* { P r } ( A _ { 0 } ^ { c } ) = O ( 1 / T )$ . The equality follows from $T \tau _ { 1 } ^ { u } ( { \bar { p } } _ { 1 } ^ { u } - \underline { { p } } _ { 1 } ^ { u } ) ^ { 2 } = { \sqrt { T } } \log ^ { 3 . 5 } T$ and $T \tau _ { i } ^ { u } ( \bar { p } _ { i } ^ { u } - \underline { { p } } _ { i } ^ { u } ) ^ { 2 } = \sqrt { T } \log ^ { 5 } T$ for $i \geq 2$ . The last inequality follows from $N ^ { u } \leq 3 \log T$

Second term of Equation (68) is the cumulative strong reward in exploiting ${ \tilde { p } } .$ We have

$$
\begin{array} { r l } & { \mathbb { E } \big [ \mathbb { E } \mathrm { s y b i n t ~ } \tilde { p } \mathrm { ~ T e x u c t } \big ] \leq \mathbb { E } \Big [ \rho ( \tilde { p } ) \hat { T } ^ { s } \tau ( B _ { N ^ { s } + 1 } ) \Big ] - \tilde { p } \mathbb { E } \Bigg [ \Bigg ( \hat { T } ^ { s } + \sum _ { i = 1 } ^ { Y _ { i } ^ { s } } Y _ { i , j } ^ { s } - B _ { i } ^ { s } ) ^ { + } \tau ( B _ { N ^ { s } + 1 } ) \Bigg ] } \\ & { \qquad = \mathbb { E } \big [ ( \tilde { p } ) ( T - t _ { N ^ { s } } ^ { s } ) \big ] ( B _ { N ^ { s } + 1 } ) \big ] - \tilde { p } \mathbb { E } \Bigg [ \Bigg ( \hat { Y } ^ { s } + \sum _ { i = 1 } ^ { Y _ { i } ^ { s } } \gamma _ { i , j } ^ { s } - B _ { i } ^ { s } \Bigg ) ^ { + } I ( B _ { N ^ { s } + 1 } ) } \\ & { \qquad \geq ( \rho ^ { p } \lambda ( \rho ^ { p } ) - C _ { T _ { 1 } ^ { s } } ^ { p } ) \Bigg ( T ^ { \ast } \Bigg ( 1 - \sum _ { i = 1 } ^ { N ^ { s } } \tau _ { i } ^ { s } \Bigg ) - \sum _ { i = 1 } ^ { N ^ { s } } r _ { i } ^ { s } \Bigg ) \operatorname { P r } ( B _ { N ^ { s } + 1 } ) } \\ & { \qquad \quad - C - p \mathbb { E } \Bigg [ \Bigg ( \hat { Y } ^ { s } + \sum _ { i , j } ^ { Y _ { i } ^ { s } } - B _ { i } ^ { s } \Bigg ) ^ { - } I ( B _ { N ^ { s } - 1 } ) \Bigg ] } \\ & { \qquad \geq T ^ { \ast } \Bigg ( 1 - \sum _ { i = 1 } ^ { N ^ { s } } \tau _ { i } ^ { s } \Bigg ) p ^ { D } \lambda ( \rho ^ { p } ) \operatorname { P r } ( B _ { N ^ { s } + 1 } ) - C \sqrt { T } \log ^ { s } T . } \end{array}\tag{70}
$$

The first inequality follows from min $\{ a , ( B - c ) ^ { + } \} \geq a - ( a + c - B ) ^ { + }$ and $\rho ( \tilde { p } ) \leq \bar { p }$ . The equality follows by conditioning on the history before the final period and using $\rho ( \tilde { p } ) \lambda ( \tilde { p } ) = g ( \tilde { p } )$ . The second inequality follows from Equation (65), $\operatorname* { P r } ( A _ { 0 } ^ { c } ) = O ( 1 / T )$ , and $\begin{array} { r } { t _ { N ^ { u } } ^ { u } \leq \sum _ { i = 1 } ^ { N ^ { u } } ( T \tau _ { i } ^ { u } + \kappa _ { i } ^ { u } ) } \end{array}$ . The last inequality follows from Equation (66), $T \tau _ { 1 } ^ { u } = \sqrt { T } \log ^ { 3 . 5 } T$ , and $\begin{array} { r } { \sum _ { i = 1 } ^ { N ^ { u } } \kappa _ { i } ^ { u } \le C T ^ { 1 / 1 0 } \log ^ { 2 } T } \end{array}$

Third term of Equation (68) is the cumulative strong reward in exploring $p ^ { c }$ . We have

$$
\begin{array} { r l } & { \mathbb { E } [ \mathrm { E x p l o r e ~ } p ^ { c } \mathrm { ~ R e w a r d } ] \gtrsim \displaystyle \sum _ { i = 1 } ^ { N ^ { c } } T \tau _ { i } ^ { c } \left( p ^ { D } \lambda ( p ^ { D } ) - C ( \bar { p } _ { i } ^ { c } - \underline { { p } } _ { i } ^ { c } ) \right) \mathrm { P r } \left( \bigcup _ { l = 1 } ^ { N ^ { u } } B _ { l } \right) - C } \\ & { \qquad = \displaystyle \sum _ { i = 1 } ^ { N ^ { c } } T \tau _ { i } ^ { c } p ^ { D } \lambda ( p ^ { D } ) \mathrm { P r } \left( \bigcup _ { l = 1 } ^ { N ^ { u } } B _ { l } \right) } \\ & { \qquad - C \left( \sqrt { T } \log ^ { 2 . 5 } T + ( N ^ { c } - 1 ) \sqrt { T } \log ^ { 3 } T \right) - C } \\ & { \qquad \geq \displaystyle \sum _ { i = 1 } ^ { N ^ { c } } T \tau _ { i } ^ { c } p ^ { D } \lambda ( p ^ { D } ) \mathrm { P r } \left( \bigcup _ { l = 1 } ^ { N ^ { u } } B _ { l } \right) - C \sqrt { T } \log ^ { 4 } T . } \end{array}\tag{71}
$$

The first inequality follows from $\Delta _ { i } ^ { c } \geq T \tau _ { i } ^ { c } / \kappa _ { i } ^ { c }$ , Equation (65), and $\operatorname* { P r } ( A _ { 0 } ^ { c } ) = O ( 1 / T )$ . The equality follows from $T \tau _ { 1 } ^ { c } ( \bar { p } _ { 1 } ^ { c } - p _ { 1 } ^ { c } ) = \sqrt { T } \log ^ { 2 . 5 } T$ and $T \tau _ { i } ^ { c } ( \bar { p } _ { i } ^ { c } - \underline { { p } } _ { i } ^ { c } ) = \sqrt { T } \log ^ { 3 } T$ for $i \geq 2$ . The last inequality follows from $N ^ { c } \leq 3 \log T$

Fourth term of Equation (68) is the cumulative strong reward in exploiting ${ \tilde { q } } .$ We have

$$
\begin{array} { r l } { \mathbb { E } _ { \tau } \mathbb { E } \Psi \lambda \lambda ^ { \lambda } \tilde { \tau } } & { \mathbb { E } \operatorname { R e x a m c l e } \lambda \sum _ { i = 1 } ^ { N } \mathbb { E } [ e ( \hat { \lambda } ) \hat { \lambda } _ { i j } ^ { \lambda } \tau ^ { \lambda } \langle R _ { i i } ^ { \lambda } \rangle ] - \theta ^ { \sum _ { i = 1 } ^ { N } } \mathbb { E } [ ( \hat { \lambda } ^ { \tau } \sum _ { i = 1 } ^ { \tau } \sum _ { j = 1 } ^ { \tau } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \beta _ { i j } ^ { \lambda } ) ^ { \lambda } ] ^ { \lambda } \hat { \mathcal { H } } ( R _ { i j } ) } \\ & { \quad - \displaystyle \sum _ { i = 1 } ^ { N } \mathbb { E } [ \lambda \hat { \sigma } _ { i j } ^ { \lambda } \langle I \rangle ^ { \lambda }  } \\ & { \quad \quad  - \sum _ { i = 1 } ^ { N } \mathbb { E } [ ( \hat { \lambda } ^ { \tau } ) + \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } ) ^ { \lambda } \hat { \mathcal { H } } ( R _ { i j } ) ] } \\ &  \quad \quad - \displaystyle \sum _ { i = 1 } ^ { N } \mathbb { E } [ ( \hat { \lambda } ^ { \tau } ) + \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum \end{array}\tag{72}
$$

The first inequality follows from min $\{ a , ( B - c ) ^ { + } \} \geq a - ( a + c - B ) ^ { + }$ and $\rho ( \tilde { q } ) \leq \bar { p } .$ . The equality follows by conditioning on the history before the final period and using $\rho ( \tilde { q } ) \lambda ( \tilde { q } ) = g ( \tilde { q } )$ . The second inequality follows from Equation (65), $\operatorname* { P r } ( A _ { 0 } ^ { c } ) = O ( 1 / T )$ , and $\begin{array} { r } { t _ { l } ^ { u } + t _ { N ^ { c } } ^ { c } \leq T \left( \sum _ { i = 1 } ^ { l } \tau _ { i } ^ { u } + \sum _ { i = 1 } ^ { N ^ { c } } \tau _ { i } ^ { c } \right) + } \end{array}$ $\begin{array} { r } { \sum _ { i = 1 } ^ { l } \kappa _ { i } ^ { u } + \sum _ { i = 1 } ^ { N ^ { c } } \kappa _ { i } ^ { c } } \end{array}$ . The last inequality follows from Equation (66), $T \tau _ { 1 } ^ { c } = \sqrt { T } \log ^ { 2 . 5 } T$ , and $\sum _ { i } ( \kappa _ { i } ^ { u } +$ $\kappa _ { i } ^ { c } ) \leq C T ^ { 1 / 6 } \log ^ { 2 } T$

Combining all terms: Combining Equations (69)–(72), we have

$$
\begin{array} { r l } & {  { \mathrm { R E W } } _ { S } \geq T p ^ { D } \lambda ( p ^ { D } ) \left[ \displaystyle \sum _ { i = 1 } ^ { N ^ { u } } \tau _ { i } ^ { u }  { \mathrm { P r } } \left( \displaystyle \bigcup _ { l = i } ^ { N ^ { u } + 1 } B _ { l } \right) + \left( 1 - \displaystyle \sum _ { i = 1 } ^ { N ^ { u } } \tau _ { i } ^ { u } \right)  { \mathrm { P r } } ( B _ { N ^ { u } + 1 } ) \right. } \\ & { \qquad \left. + \left( \displaystyle \sum _ { i = 1 } ^ { N ^ { c } } \tau _ { i } ^ { c } \right)  { \mathrm { P r } } \left( \displaystyle \bigcup _ { l = 1 } ^ { N ^ { u } } B _ { l } \right) + \displaystyle \sum _ { l = 1 } ^ { N ^ { u } } \left( 1 - \displaystyle \sum _ { i = 1 } ^ { l } \tau _ { i } ^ { u } - \displaystyle \sum _ { i = 1 } ^ { N ^ { c } } \tau _ { i } ^ { c } \right)  { \mathrm { P r } } ( B _ { l } ) \right] - C \sqrt { T } \log ^ { 6 } T } \\ & { = T p ^ { D } \lambda ( p ^ { D } ) \displaystyle \sum _ { l = 1 } ^ { N ^ { u } + 1 }  { \mathrm { P r } } ( B _ { l } ) - C \sqrt { T } \log ^ { 6 } T } \\ & { =  { \mathrm { O P T } } - C \sqrt { T } \log ^ { 6 } T . } \end{array}\tag{73}
$$

The first equality follows by collecting the coeficient of each $\mathrm { P r } ( B _ { l } )$ , which is one for every $l = 1 , \ldots , N ^ { u } + 1$ , and the second equality follows because $B _ { 1 } , \ldots , B _ { N ^ { u } + 1 }$ form a partition and $\mathrm { O P T } = T p ^ { D } \lambda ( p ^ { D } )$ . Therefore,

$$
\mathrm { O P T } - \mathrm { R E W } _ { S } \leq C \sqrt { T } \log ^ { 6 } T .\tag{74}
$$