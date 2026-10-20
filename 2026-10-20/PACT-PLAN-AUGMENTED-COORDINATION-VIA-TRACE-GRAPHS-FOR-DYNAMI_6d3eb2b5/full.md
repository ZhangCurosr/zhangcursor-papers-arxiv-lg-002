# PACT: PLAN-AUGMENTED COORDINATION VIA TRACE-GRAPHS FOR DYNAMIC MULTI-AGENT COMBINATORIAL OPTIMIZATION

Xiao Zhou, Alex Schutz, Victor-Alexandru Darvariu & Nick Hawes

Oxford Robotics Institute

University of Oxford

OX1 3PJ, Oxford, United Kingdom.

{xiaozhou,alexschutz,victord,nickh}@robots.ox.ac.uk

## ABSTRACT

Multi-agent approaches have been investigated recently as a means of improving the scalability of reinforcement learning for neural combinatorial optimization. Existing methods assume that all problem instance information is known upfront. However, real-world problems are often dynamic: relevant information is progressively revealed as the agents take actions (e.g., arriving at an edge in a routing problem only to discover it is untraversable). In such cases, existing methods fail to coordinate agents to explore and exploit the progressively revealed information. To address the arising coordination challenge, we introduce PACT (Plan-Augmented Coordination via Trace-graphs). PACT augments the environment graph with auxiliary nodes representing the plans of agents, which are connected to the resource nodes to indicate future dependencies. The proposed messagepassing scheme over the augmented graph grounds inter-agent coordination in the environment and enables proactive, rather than reactive, conflict resolution. We show that PACT improves solution quality in dynamic problems relative to baselines and exhibits the best transfer to larger instances on which, remarkably, it outperforms centralized oracles with privileged access to instance uncertainty.

## 1 INTRODUCTION

Many routing, scheduling, and allocation problems require several agents to construct a joint combinatorial solution under a shared objective such as total travel cost or makespan, a setting known as multi-agent combinatorial optimization (MACO) (Berto et al., 2025). Existing formulations typically assume that the instance to be solved is fully specified before construction begins. In many real systems, however, relevant information becomes available only through execution: a route may be discovered to be untraversable only when an agent reaches it, or a machine may become known to be unavailable only when an operation is attempted (Pillac et al., 2013; Ouelhadj & Petrovic, 2009). We call this setting dynamic multi-agent combinatorial optimization (DMACO).

Classic combinatorial optimization typically assumes fully disclosed instances upfront. Although reactive re-optimization can adapt to newly revealed information, solving must recur continuously (Vieira et al., 2003; Bertsimas et al., 2019), and such reactive paradigms generally do not account for how current decisions affect information and coordination requirements that arise later (Ulmer, 2019). Neural combinatorial optimization (NCO) amortizes the cost of re-solving by learning policies over instance distributions (Kool et al., 2019; Kwon et al., 2020; Kool et al., 2022) and has recently been extended to multi-agent settings (Berto et al., 2025; Luttmann & Xie, 2026). Nonetheless, these NCO approaches still assume fixed and fully observable instances prior to execution.

Since information in DMACO is revealed through agents’ actions, we formulate it as a sequential decision-making problem, naturally suited to multi-agent reinforcement learning (MARL) (Bernstein et al., 2002; Oliehoek & Amato, 2016; Yu et al., 2022). Existing MARL approaches typically manage coordination via agent-centric representations (Sunehag et al., 2018; Guestrin et al., 2002; Sukhbaatar et al., 2016). Alongside neural MACO solvers that resolve step-level conflicts during parallel decoding (Berto et al., 2025), these methods broadly share a common limitation: they localize coordination at the agent or step level, without explicitly modeling the shared environmental structure that drives their interdependencies.

In DMACO, coordination requirements are fundamentally induced by the environment. For instance, multiple agents might contend for a capacitylimited resource, or one agent’s action might resolve local uncertainty crucial to another’s future trajectory. Because these dependencies are grounded in the environment, relying solely on agent-level relations creates a structural mismatch: the underlying graph elements driving the coordination are excluded from the representational mechanism. Furthermore, restricting coordination to

![](images/fa7083575f00f704c539999359a0ccd6ab0b40a20a60ae59aff13762377e53ee.jpg)  
Figure 1: PACT with plan-level coordination prevents goal conflicts exhibited by MARL methods with step-level coordination.

the current step introduces a temporal mismatch. An agent might commit to a trajectory that appears locally conflict-free, only to conflict with another agent’s plan several steps later, as illustrated in Figure 1.

To address these structural and temporal mismatches, we introduce PACT (Plan-Augmented Coordination via Trace-graphs), a novel coordination framework for DMACO. First, each agent proposes multiple candidate plans. Every candidate plan is then attached to the environment graph via its projected trace, a record of the graph elements on which it places demand. By performing coordination on this plan-augmented trace-graph, PACT enables the underlying environmental structure to explicitly mediate dependencies across agents’ future candidate plans. We refer to this mechanism as environment-mediated coordination. Our key contributions are as follows:

• We propose PACT, a novel coordination framework for DMACO problems that represents coordination requirements directly through the environmental structure that induces them, rather than through ungrounded relations defined between agents.

• We introduce the plan-augmented trace-graph, which attaches candidate plans to the elements they demand. The proposed message-passing scheme resolves contention over remaining capacities, thereby coordinating agents over future decision steps rather than only the current one.

• On three DMACO problems spanning routing and scheduling, we show that PACT outperforms neural MACO solvers and MARL methods in solution quality and zero-shot generalization on instances up to 64× larger than the training set, at a fraction of the runtime of online replanning.

## 2 RELATED WORK

Neural Multi-Agent Combinatorial Optimization. NCO amortizes the cost of solving recurring optimization problems by learning solution policies over instance distributions (Bengio et al., 2021; Cappart et al., 2023). Constructive approaches generate solutions autoregressively (Kool et al., 2019; Kwon et al., 2020), with multi-agent extensions coordinating agents’ decisions to construct a joint solution. Sequential methods decode agents’ actions in turn (Son et al., 2024; Zheng et al., 2024), whereas parallel methods simultaneously decode agents’ actions with inter-agent communication (Zong et al., 2022; Hua et al., 2025). These approaches focus on specific MACO problems, while PARCO (Berto et al., 2025) provides a general framework for parallel decoding across MACO problems, resolving conflicting selections through a learned priority-based handler. MACSIM (Luttmann & Xie, 2026) learns through self-improvement to predict joint agent-action assignments that are conflict-free by construction. These methods assume that problem instances are fully specified before solution construction begins and coordinate agents at the current step. They therefore do not explicitly reason about information revealed during execution or dependencies between agents’ decisions across future steps.

Coordination in Multi-Agent Reinforcement Learning. Cooperative MARL typically employs centralized training with decentralized execution (Bernstein et al., 2002; Oliehoek & Amato, 2016), under which representative algorithms such as multi-agent PPO (MAPPO) (Yu et al., 2022) train a shared centralized critic, whereas independent PPO (IPPO) (De Witt et al., 2020) uses decentralized critics. Coordination is commonly achieved via value factorization, such as additive decomposition in VDN (Sunehag et al., 2018) and monotonic mixing in QMIX (Rashid et al., 2020), explicit representations like deep coordination graphs (DCG) (Bohmer et al., 2020), or inter-agent commu-¨ nication (Sukhbaatar et al., 2016; Foerster et al., 2016; Das et al., 2019; Hu et al., 2024). In all these families, the coordination medium is agent-centric (e.g., agent utilities, pairwise payoffs, or directed messages). This introduces two limitations when these approaches are applied to DMACO. First, pairwise decompositions struggle to capture simultaneous multi-agent dependencies without incurring the exponential inference costs of higher-order factors (Guestrin et al., 2002; Kok & Vlassis, 2006). Second, these mechanisms coordinate the agents’ actions at the current state, whereas conflicts in the future are reflected only implicitly through learned value estimates.

In DMACO settings, PACT addresses both limitations through environment-mediated coordination. By connecting each candidate plan to the graph elements its trace occupies, PACT captures interactions between candidate plans, such as contention for shared capacity or dependence on unrevealed information, at the element that induces them. This element-centric aggregation accommodates multi-agent dependencies without explicit agent-pair factors, while candidate traces expose coordi nation requirements spanning future decision steps.

## 3 PRELIMINARIES

We represent a DMACO problem as a graph, where each node is an element that a solution selects, assigns, or sequences, and edges encode the relations among elements. We assume that any observation made by one agent is immediately available to the whole team, so that all agents act on the same information. The problem is therefore modeled as a fully cooperative multi-agent partially observable Markov decision process (MPOMDP) (Oliehoek & Amato, 2016), defined by the tuple $\langle \mathcal { M } , \mathcal { S } , \{ \mathcal { A } _ { i } \} _ { i \in \mathcal { M } } , \mathcal { T } , O , C , \mathcal { \bar { Y } } \rangle$ , where $\mathcal { M }$ is the team of agents, $s$ the states, $\{ { \mathcal { A } } _ { i } \} _ { i \in { \mathcal { M } } }$ the action space for each agent, $\tau$ the transition function, O the observation function, $C$ the cost function, and $\gamma \in ( 0 , 1 ]$ the discount factor. We factorize the state space $s$ into observable graph states $\mathcal { G } _ { t }$ and an unobserved latent realization $\xi ,$ with $O ( \mathcal { G } _ { t } , \xi ) = \mathcal { G } _ { t }$ . At the start of each episode we sample $\xi \sim p ( \xi )$ , and it is held fixed for the episode. The observed graph $\mathcal G _ { t } = ( \nu _ { t } , \mathcal E _ { t } , \dot { X } _ { t } )$ consists of the elements $\nu _ { t } .$ , the relations $\mathcal { E } _ { t }$ among them, and the attributes $X _ { t }$ of each element and each agent. The attributes $X _ { t }$ also determine the set $\mathcal { M } _ { t } \subseteq \mathcal { M }$ of agents active at decision step t. Each active agent $i \in \mathcal { M } _ { t }$ selects an action $a _ { i , t } \in \mathcal { A } _ { i } ( \mathcal G _ { t } )$ from its own policy $\pi _ { i }$ conditioned on the graph observation $\mathcal { G } _ { t } .$ , where $\mathcal { A } _ { i } ( \mathcal { G } _ { t } ) \subseteq \mathcal { V } _ { t } \cup \{ \mathcal { O } \}$ , so that agent i commits either to a single element or to none. We write $\pi = \{ \pi _ { i } \} _ { i \in \mathcal { M } }$ for the team policy. The state then advances according to $\mathscr { G } _ { t + 1 } \sim \mathcal { T } ( \cdot \mid \mathscr { G } _ { t } , \mathbf { a } _ { t } , \xi )$ where the joint action $\mathbf { a } _ { t } = \{ a _ { i , t } \} _ { i \in \mathcal { M } _ { t } }$ . Each decision step incurs a cost $C ( \mathcal { G } _ { t } , \mathbf { a } _ { t } , \xi ) \in \mathbb { R } _ { \ge 0 }$ , and the agents minimize the expected discounted cost $\begin{array} { r } { J ( \pi ) = \mathbb { E } _ { \xi , \pi } \Big [ \sum _ { t } \gamma ^ { t } C ( \mathcal { G } _ { t } , \mathbf { a } _ { t } , \xi ) \Big | } \end{array}$

Capacities and demands serve as the general interface for defining resource constraints in combinatorial optimization (Blazewicz et al., 1983) and constraint programming (Aggoun & Beldiceanu, 1993). We assign a capacity $\kappa _ { t } ( v ) \geq 0$ to each element $v \in \mathcal V _ { t } .$ , recorded in the attributes $X _ { t }$ . The action $a _ { i }$ from agent i places a demand $d ( v , a _ { i , t } ) \geq 0$ on the element $v .$ . A unit capacity $\kappa ( v ) = 1$ makes an element exclusive, which covers a goal location claimed by a single traveler (Toth & Vigo, 2014) and a job processed by a single machine (Pinedo, 1995), while larger capacities describe resources that several agents may hold simultaneously. At decision step $t ,$ the demand on element v from all active agents is $\begin{array} { r } { d _ { t } ( v ) \dot { { } = } \sum _ { i \in \mathcal { M } _ { t } } d ( v , a _ { i , t } ) } \end{array}$ . The joint actions permitted by the remaining capacities form the set

$$
\mathcal { A } ( \mathcal { G } _ { t } ) = \left\{ a = \{ a _ { i } \} _ { i \in \mathcal { M } _ { t } } \in \prod _ { i \in \mathcal { M } _ { t } } \mathcal { A } _ { i } ( \mathcal { G } _ { t } ) : \ d _ { t } ( v ) \leq \kappa _ { t } ( v ) \mathrm { ~ f o r ~ e v e r y ~ } v \in \mathcal { V } _ { t } \right\} .\tag{1}
$$

![](images/85b2fababfcd86b50ecde29da4f2ce94db4614ebb7d43557dafc1193d3765804.jpg)  
Figure 2: Overview of PACT. After encoding graph observations, agents independently generate candidate plans to form a plan-augmented trace-graph for multi-step coordination. The joint action $\mathbf { a } _ { t }$ comprises the initial step of each selected plan.

Eq. (1) is the only feasibility constraint that couples the agents, and it is a conjunction of conditions taken element by element.

## 4 METHODOLOGY

We now present PACT, whose overall architecture is illustrated in Figure 2. Each agent first generates a set of candidate plans instead of committing to a single next action (Section 4.1). The candidate plans are incorporated into the graph through the elements they occupy, forming a planaugmented trace-graph over which inter-agent coordination is performed (Section 4.2). Each agent subsequently selects one of its coordinated candidate plans and executes its first action (Section 4.3). Finally, all modules are trained end to end (Section 4.4). The complete forward pass of PACT is summarized in Algorithm 1.

## 4.1 CANDIDATE PLAN GENERATION

As was illustrated in Figure 1, coordination over individual actions is inherently myopic: agents whose current actions are compatible may still contend for the same resources several steps ahead. PACT therefore coordinates over short-horizon candidate plans. However, planning in the joint action space scales exponentially with both the number of agents and the planning horizon (Bernstein et al., 2002; Kearns et al., 2002). We instead decouple plan proposal from multi-agent coordination: each agent independently generates a compact set of candidate plans, while interactions among these plans are handled by the coordination mechanism in Section 4.2.

Given the current graph observation G<sub>t</sub>, a GNN-based encoder Enc<sub>θ</sub>, whose parameters θ are shared with every other learned module of PACT, extracts graph-related representations $h _ { G }$ together with an agent-related representation $\sigma _ { i }$ for each active agent,

$$
\left( h _ { G } , \ \{ \sigma _ { i } \} _ { i \in { \cal M } _ { t } } \right) = \mathrm { E n c } _ { \theta } ( \mathcal { G } _ { t } ) .\tag{2}
$$

The graph representation $h _ { G }$ describes the decision-relevant structure and elements of the current problem instance, while $\sigma _ { i }$ provides the agent-specific context needed to specialize the graph representation to agent i.

For each agent i, PACT generates K candidate plans, each consisting of a sequence of $H _ { p }$ actions,

$$
\begin{array} { r } { \hat { p } _ { i } ^ { k } = \big ( \hat { a } _ { i , t } ^ { k } , ~ \hat { a } _ { i , t + 1 } ^ { k } , ~ . . . , ~ \hat { a } _ { i , t + H _ { p } - 1 } ^ { k } \big ) , \qquad k = 1 , \dots , K . } \end{array}\tag{3}
$$

Each candidate plan is constructed through a rollout governed by a learned candidate plan generation policy $\pi _ { \theta } ^ { \mathrm { p l a n } }$ . To ensure diversity among the K candidate plans, their first actions are drawn jointly without replacement from the policy at the initial rollout state $\pi _ { \theta } ^ { \mathrm { p l a n } } ( \mathcal { G } _ { t } )$ . Following this initialization, each of the K plans unrolls independently. At each rollout step $j , \pi _ { \theta } ^ { \mathrm { p l a n } }$ conditions on the encoded graph $h _ { G }$ , the agent representation $\sigma _ { i } ,$ , and the rollout state at step $j , \hat { \mathcal { G } } _ { t + j , i } ^ { k }$ to extend the partial plan, ensuring that subsequent decisions dynamically reflect the accumulated consequences of preceding actions.

## 4.2 PLAN-AUGMENTED TRACE-GRAPH AND COORDINATION

Recall the routing scenario shown in Figure 1. In this scenario, two agents move apart at the current step but then both claim the same goal several steps later. Their current actions place demand on different elements, so Eq. (1) holds at every step until they arrive, and the demand that must be reconciled is located at the goal, where the final two actions together exceed its remaining capacity $\kappa _ { t } ( v )$ . PACT therefore records, for each candidate plan, which elements it would occupy, the induced demand, and the step of the plan. We refer to this record as the candidate plan’s trace,

$$
\mathrm { T r } ( \hat { p } _ { i } ^ { k } ) \ = \ \big \{ \ \big ( \boldsymbol { v } , \ d ( \boldsymbol { v } , \hat { a } _ { i , t + j } ^ { k } ) , \ j \big ) \ : \ d ( \boldsymbol { v } , \hat { a } _ { i , t + j } ^ { k } ) > 0 , \ j = 0 , . . . , H _ { p } - 1 \big \} ,\tag{4}
$$

where $j$ indexes the step of the plan at which the demand is placed.

Once K candidate plans are generated in parallel, their traces form the plan-augmented trace-graph $\chi _ { t } ~ = ~ \left( \mathcal { V } _ { t } \cup \hat { \mathcal { V } } _ { t } , ~ \mathcal { E } _ { t } \cup \hat { \mathcal { E } } _ { t } \right)$ , where candidate plan nodes $\hat { \mathcal { V } } _ { t } = \{ \hat { p } _ { i } ^ { k } : i \in \mathcal { M } _ { t } , k \le K \}$ and trace arcs $\hat { \mathcal { E } } _ { t } = \{ ( \hat { p } , v ) : ( v , \cdot , \cdot ) \in \operatorname { T r } ( \hat { p } ) \}$ . Each arc carries the demand and the plan step that produced it. Arcs join candidate plans to elements but not to one another. This forces information exchange between agents to occur through the graph structure. Thus, the same graph structure captures both the original combinatorial problem and the prospective interactions introduced by the plans.

PACT coordinates the candidate plans by iteratively alternating between two operations in order to compute a set of preferences. First, the resource contention induced by the agents’ current plan preferences is estimated. Second, preferences are updated through message passing on the planaugmented trace-graph $\chi _ { t } .$ Concretely, let $s _ { i } ^ { ( r ) } ( k )$ denote the score of plan k of agent i after coordi nation round $r = 0 , \ldots , R$ . The corresponding preference is:

$$
q _ { i } ^ { ( r ) } ( k ) = \frac { \exp ( s _ { i } ^ { ( r ) } ( k ) ) } { \sum _ { k ^ { \prime } = 1 } ^ { K } \exp ( s _ { i } ^ { ( r ) } ( k ^ { \prime } ) ) } .\tag{5}
$$

Every node of $\chi _ { t }$ carries a representation $h ^ { ( r ) }$ at round $r ,$ whose initial value $h ^ { ( 0 ) }$ is given by $h _ { G }$ for element nodes and by an encoding of the node’s trace for candidate plan nodes. A scoring head f<sub>score</sub> turns a candidate plan’s representation into its score, so that $s _ { i } ^ { ( 0 ) } ( k ) = \eta$ tanh $\left( f _ { \mathrm { s c o r e } } ( h ^ { ( 0 ) } ( \hat { p } _ { i } ^ { k } ) ) \right)$ before information is exchanged between agents, where $\eta > 0$ is a fixed step size bound.

Given the current preferences $q _ { i } ^ { ( r ) }$ , each element aggregates the demand induced by all candidate plans that reach it. We define $\begin{array} { r } { \dot { d ( v , p _ { i } ^ { k } ) } = \sum _ { i } d ( v , \hat { a } _ { i , t + j } ^ { k } ) } \end{array}$ as the total demand that $\hat { p } _ { i } ^ { k }$ places on v across its steps. The load on element v at round r is the preference-weighted demand of all plans,

$$
L ^ { ( r ) } ( v ) = \sum _ { i \in \mathcal { M } _ { t } } \sum _ { k = 1 } ^ { K } q _ { i } ^ { ( r ) } ( k ) \times d ( v , \hat { p } _ { i } ^ { k } ) .\tag{6}
$$

Hence, $L ^ { ( r ) } ( v )$ measures the expected demand induced by the agents’ current preferences $q _ { i } ^ { ( r ) }$ over their candidate plans. The load is compared with the remaining capacity $\boldsymbol { \kappa } _ { t } ( \boldsymbol { v } ) - d _ { t } ( \boldsymbol { v } )$ of each element and injected into its representation through a learned projection $W _ { \mathrm { l o a d } }$ , normalized by the capacity $\kappa _ { t } ( v )$ ,

$$
\tilde { h } ^ { ( r ) } ( v ) = h ^ { ( r ) } ( v ) + W _ { \mathrm { l o a d } } \left[ \frac { L ^ { ( r ) } ( v ) } { \kappa _ { t } ( v ) } , \ \frac { \mathrm { R e L U } \big ( L ^ { ( r ) } ( v ) - ( \kappa _ { t } ( v ) - d _ { t } ( v ) ) \big ) } { \kappa _ { t } ( v ) } \right] ,\tag{7}
$$

where $\frac { L ^ { ( r ) } ( v ) } { \kappa _ { t } ( v ) }$ describes resource utilization, while $\big ( L ^ { ( r ) } ( v ) - ( \kappa _ { t } ( v ) - { d _ { t } ( v ) } ) \big )$  explicitly identifies demand beyond the remaining capacity. Normalizing by capacity provides a common scale for resources with different capacities.

An edge-aware message-passing layer, shared across rounds, then propagates this information over the plan-augmented trace-graph,

Algorithm 1 PACT forward pass at decision step t   
1: Input: graph observation $\mathcal { G } _ { t }$   
2: Encode $\mathcal { G } _ { t }$ once into $h _ { G }$ and $\{ \sigma _ { i } \}$ (Eq. (2))   
3: for each agent $i \in \mathcal { M } _ { t }$ (in parallel) do   
4: Generate $K$ candidate plans $\hat { p } _ { i } ^ { 1 } , \ldots , \hat { p } _ { i } ^ { K }$ through rollout, with first actions drawn jointly with  
out replacement (Section 4.1)   
5: end for   
6: Construct the plan-augmented trace-graph $\chi _ { t }$ from the candidate plan traces (Eq. (4))   
7: Initialize candidate plan scores $s _ { i } ^ { ( 0 ) }$ and preferences $q _ { i } ^ { ( 0 ) } = \mathrm { s o f t m a x } ( s _ { i } ^ { ( 0 ) } )$ (Eq. (5))   
8: for coordination round $r = 0 , \ldots , R - 1$ do   
9: Compute the load $L ^ { ( r ) } ( v )$ induced by the current plan preferences (Eq. (6))   
10: Inject capacity-aware load features and perform message passing over $\chi _ { t } \ ~ ( \mathrm { E q s . } \ ( 7 )$ and (8))   
11: Update candidate plan scores $s _ { i } ^ { ( r + 1 ) }$ and preferences $q _ { i } ^ { \left( { r + \bar { 1 } } \right) }$ (Section 4.2) (Eq. (9))   
12: end for   
13: for each agent $i \in \mathcal { M } _ { t }$ (in parallel) do   
14: Sample $z _ { i } \sim$ softmax ${ ( s _ { i } ^ { ( R ) } / \tau ) }$ and set $a _ { i , t } = \hat { p } _ { i } ^ { z _ { i } } [ 0 ]$ (Section 4.3)   
15: end for   
16: Output: Joint action $\mathbf { a } _ { t } = \{ a _ { i , t } \} _ { i \in \mathcal { M } _ { t } }$

$$
h ^ { ( r + 1 ) } = \mathrm { M P N N } \big ( \tilde { h } ^ { ( r ) } , \chi _ { t } \big ) ,\tag{8}
$$

$$
s _ { i } ^ { ( r + 1 ) } ( k ) = s _ { i } ^ { ( r ) } ( k ) + \eta \operatorname { t a n h } \bigl ( f _ { \mathrm { s c o r e } } ( h ^ { ( r + 1 ) } ( \hat { p } _ { i } ^ { k } ) ) \bigr ) .\tag{9}
$$

The updated scores are renormalized over the K candidate plans of each agent to obtain $q _ { i } ^ { ( r + 1 ) }$ which in turn induces a new load profile through Eq. (6). The candidate plan scores are therefore updated iteratively through coordination, allowing an agent’s preference over its own plans to respond to the evolving preferences of the rest of the team.

## 4.3 PLAN SELECTION AND EXECUTION

After R coordination rounds, each agent selects a candidate plan based on its final coordinated scores, $z _ { i } \sim \mathrm { s o f t m a x } ( s _ { i } ^ { ( R ) } / \tau )$ , where τ is a learned temperature controlling the sharpness of selection over candidate plans. Each agent then executes the first action of its chosen plan, $a _ { i , t } = \hat { p } _ { i } ^ { z _ { i } } [ 0 ]$ The resulting joint action $\mathbf { a } _ { t } = \{ a _ { i , t } \} _ { i \in \mathcal { M } }$ is applied to the environment via $\tau _ { \ast }$ , which incurs the cost $C ( \mathcal { G } _ { t } , \mathbf { \bar { a } } _ { t } , \bar { \xi } )$ , reveals the components of $\xi$ associated with $\mathbf { a } _ { t } ,$ and yields the subsequent graph observation $\mathcal { G } _ { t + 1 }$ . Following the receding-horizon principle, PACT regenerates and coordinates new candidate plans from $\mathcal { G } _ { t + 1 }$ at each step. Consequently, while future actions within a candidate plan serve as look-ahead information for multi-agent coordination, execution remains closed-loop and adapts to newly revealed information.

## 4.4 POLICY TRAINING

PACT learns its plan generation and plan-selection policies from the return. Unlike a conventional policy in which the sampled policy output is the environment action, PACT produces each agent’s action through a structured sequence of stochastic decisions: candidate plans are first generated and subsequently selected after coordination. The policy objective therefore needs to account for the internal decisions that give rise to the executed action.

We decompose the policy update into three components: candidate plan proposal (prop), plan rollout (roll), and plan selection (sel). For each component $g \in \{ \mathrm { p r o p } , \mathrm { r o l l } , \mathrm { s e l } \}$ , we optimize the clipped surrogate objective:

$$
\mathcal { L } _ { g } ( \theta ) = \mathbb { E } _ { t } \left[ \sum _ { i \in \mathcal { M } _ { t } } \operatorname* { m i n } \left( \rho _ { i , t } ^ { g } ( \theta ) \hat { A } _ { t } , \operatorname { c l i p } \left( \rho _ { i , t } ^ { g } ( \theta ) , 1 - \epsilon _ { g } , 1 + \epsilon _ { g } \right) \hat { A } _ { t } \right) \right] ,\tag{10}
$$

where $\rho _ { i , t } ^ { g }$ is the importance ratio between the probabilities that the current and the behavior policies assign to agent i’s decisions in component g. For prop, these decisions are the K first actions, and the ratio compares their joint probability as an ordered set drawn without replacement. For roll, they are the actions at steps $j \geq 1$ of the selected plan $\hat { p } _ { i } ^ { z _ { i } }$ . For sel, the decision is the selected index $z _ { i }$ . The advantage $\hat { A } _ { t }$ is estimated with generalized advantage estimation (Schulman et al., 2015) from a critic conditioned on the graph observation $\mathcal { G } _ { t } .$ , and $\epsilon _ { g }$ is the clipping width. The overall objective sums the component objectives as ${ \mathcal { L } } _ { \mathrm { P A C T } } ( \theta ) = { \mathcal { L } } _ { \mathrm { p r o p } } ^ { \mathrm { { ~ } } } ( \theta ) + { \mathcal { L } } _ { \mathrm { r o l l } } ^ { \mathrm { { ~ } } } ( \theta ) + { \mathcal { L } } _ { \mathrm { s e l } } ( \theta )$ . Clipping each component separately preserves $\mathrm { P A C T } { \boldsymbol { \mathrm { s } } }$ structured hierarchy while allowing a single return signal to jointly train proposal, rollout, and selection.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETTING

Problems. We evaluate PACT on three representative DMACO problems, spanning both routing and scheduling domains. In the Multi-Agent Canadian Traveller Problem (MACTP) (Papadimitriou & Yannakakis, 1991; Shiri & Salman, 2017), a team traverses a graph to collectively visit goal nodes at minimum total travel cost, with edge traversability revealed upon reaching an endpoint or attempting a crossing. Travelers are agents and goal nodes have unit capacity. In the Dynamic Flexible Job-Shop Problem (DFJSP) (Ouelhadj & Petrovic, 2009), jobs comprise precedenceconstrained operations assigned to eligible machines; whether an uncertain machine is operational or broken is revealed when an operation is dispatched to it. In the Non-Clairvoyant Flexible Flow Shop Problem (NCFFSP) (Motwani et al., 1994), jobs pass through ordered stages of interchangeable machines whose latent processing-time factors are revealed upon dispatch. Both scheduling problems minimize makespan, with machines acting as agents and jobs as unit-capacity elements. Appendix A provides the formal models and revelation rules.

Baselines. We compare PACT with two multi-agent NCO methods, PARCO and MACSIM, which are evaluated as centralized joint-construction solvers through task wrappers, retaining their original training objectives and model configurations. For PACT, PARCO, and MACSIM two decoding strategies named greedy (g.) and sampling (s.) are used. We also compare four cooperative MARL baselines (MAPPO, IPPO, VDN, and DCG), using matched graph encoders over the common observation interface and shared parameters across agents. MAPPO, IPPO, and VDN produce per-agent actions, whereas DCG coordinates joint actions through max-plus message passing on an agent coordination graph. PACT uses a critic conditioned on the graph observation and performs decentralized execution with communication through the plan-augmented trace-graph, in which each agent selects its own candidate plan. Gurobi (Gurobi Optimization, LLC, 2026) and OR-Tools (Perron & Furnon) provide online replanning baselines and oracle references. Appendix B.1 details the baseline implementations and adaptations.

Training. For each problem, all learning-based methods are trained on the same instance set with a mixture of small sizes. The mixture spans $n \in \{ 1 0 , 1 3 , 1 6 \}$ nodes for MACTP, $N \in \{ 1 2 , 1 6 , 2 0 \}$ jobs for DFJSP, and $N \in \{ 1 2 , 1 6 , 2 0 \}$ jobs with 3 stages and 3 or 4 machines per stage for NCFFSP. Dataset generation seeds are fixed, and each method is trained with 10 independent seeds. The training configurations are given in Appendix B.

Evaluation. All methods use the same 100 test instances per configuration, and learning-based results are averaged across the 10 training seeds. Instance sizes are denoted by nodes × agents × goals $( n \times A \times G )$ for MACTP, jobs × machines × operations $( N \times M \times O )$ for DFJSP, and jobs × machines per stage × stages $( N \times M _ { \ell } \times S )$ for NCFFSP. The first configuration in each problem block is in-distribution (ID), matching the training mixture’s largest size; the remaining four evaluate zero-shot transfer (ZST) to larger instances. The final two NCFFSP configurations also increase the stage count from 3 to 4. Appendix C.2 details the evaluation protocols.

## 5.2 MAIN RESULTS

Table 1 shows that PACT, using the same framework across all three problems with decentralized execution and coordination through the plan-augmented trace-graph, attains the lowest cost among learning-based methods in almost every configuration under greedy decoding and in every configu ration under sampling. Each baseline instead degrades in some regime: the MARL baselines exceed GPU memory on large MACTP instances, PARCO’s success collapses on the largest MACTP and DFJSP instances, and VDN, DCG, and MACSIM leave a large fraction of DFJSP episodes incomplete under greedy decoding. The advantage of PACT grows under zero-shot transfer and is most pronounced on the largest MACTP configuration, where it reduces greedy cost by 34% relative to MACSIM while maintaining 99.6% success. Compared with online replanning, PACT reaches comparable or lower cost with sampling and runs at least seven times faster under greedy decoding wherever replanning completes, whereas replanning loses success or exceeds its time limit as instances grow. We note that, while Oracle methods are not directly comparable due to having privileged access to the uncertainty ξ, they nevertheless may not yield optimal solutions on the largest instances, since they are limited by the time budget. We provide a breakdown of PACT’s end-to-end wall clock runtime into environment stepping, encoding, candidate plan generation, coordination, plan selection, and other costs in Appendix C.3.

Table 1: Results on three DMACO problems under ID and ZST evaluation. Cost ↓, Time (s) ↓, and Success (%) ↑. N/A marks unsupported configurations, T/O marks runs without a feasible solution within the time limit, and OOM marks runs that exceed GPU memory.
<table><tr><td rowspan=1 colspan=14>MACTP</td></tr><tr><td rowspan=1 colspan=4>n × A × G</td><td rowspan=1 colspan=3>16×4×6 (ID)</td><td rowspan=1 colspan=1>32×6×10 (ZST)</td><td rowspan=1 colspan=4>128×12×24 (ZST)</td><td rowspan=1 colspan=1>512×24×48 (ZST)</td><td rowspan=1 colspan=1>1024×32×64 (ZST)</td></tr><tr><td rowspan=1 colspan=4>Metric</td><td rowspan=1 colspan=3>Cost Succ.Time</td><td rowspan=1 colspan=1>CostSucc. Time</td><td rowspan=1 colspan=4>CostSucc. Time</td><td rowspan=1 colspan=1>CostSucc. Time</td><td rowspan=1 colspan=1>CostSucc.Time</td></tr><tr><td rowspan=2 colspan=1>Oracle</td><td rowspan=2 colspan=3>Gurobi (10m)OR-Tools (10m)</td><td rowspan=2 colspan=3>2.84 100.0 0.182.84100.0 0.66</td><td rowspan=2 colspan=1>4.28100.0 0.254.28100.037.5</td><td rowspan=1 colspan=4>9.52100.0 386.5</td><td rowspan=1 colspan=1>21.9100.0 627.2</td><td rowspan=1 colspan=1>3132.593.0652.3</td></tr><tr><td rowspan=1 colspan=4>13.0100.0 624.7</td><td rowspan=1 colspan=1>917.797.0633.8</td><td rowspan=1 colspan=1>T/O T/OT/O</td></tr><tr><td rowspan=2 colspan=1>Replan</td><td rowspan=2 colspan=3>GurobiOR-Tools</td><td rowspan=2 colspan=3>5.0290.00.424.0893.00.92</td><td rowspan=1 colspan=1>10.179.01.06</td><td rowspan=1 colspan=4>95.037.013.9</td><td rowspan=1 colspan=1>T/OT/OT/O</td><td rowspan=1 colspan=1>T/OT/OT/O</td></tr><tr><td rowspan=1 colspan=1>23.534.06.37</td><td rowspan=1 colspan=4>231.939.0 121.3</td><td rowspan=1 colspan=1>T/O T/OT/O</td><td rowspan=1 colspan=1>T/O T/OT/O</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=3>MAPPO (g.)IPPO (g.)</td><td rowspan=1 colspan=3>3.8396.8 0.027</td><td rowspan=1 colspan=1>6.7898.1 0.020</td><td rowspan=2 colspan=4>32.895.70.7928.697.20.63</td><td rowspan=2 colspan=1>O0MOOM OOM00M00M 00M</td><td rowspan=2 colspan=1>OOMOOM OOM00M00M O0M</td></tr><tr><td rowspan=1 colspan=3>3.9297.1 0.015</td><td rowspan=1 colspan=1>6.8498.80.033</td></tr><tr><td rowspan=2 colspan=1>Greedy</td><td rowspan=2 colspan=3>VDN (g.)DCG (g.)</td><td rowspan=1 colspan=3>3.5298.9 0.012</td><td rowspan=1 colspan=1>5.8499.6 0.019</td><td rowspan=1 colspan=4>20.799.90.38</td><td rowspan=1 colspan=1>00MO0M O0M</td><td rowspan=1 colspan=1>00MO0M O0M</td></tr><tr><td rowspan=1 colspan=3>3.6899.3 0.011</td><td rowspan=1 colspan=1>8.4498.1 0.048</td><td rowspan=1 colspan=4>70.093.30.99</td><td rowspan=1 colspan=1>00MO0M O0M</td><td rowspan=1 colspan=1>00MO0M O0M</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>PARCO (g.)</td><td rowspan=1 colspan=3>4.5698.9 0.004</td><td rowspan=1 colspan=1>10.895.00.017</td><td rowspan=1 colspan=4>195.141.20.11</td><td rowspan=1 colspan=1>1722.04.20.35</td><td rowspan=1 colspan=1>3999.24.5 0.73</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=3>MACSIM (g.)PACT (g.)</td><td rowspan=2 colspan=3>7.3689.9 0.0383.6699.1 0.056</td><td rowspan=1 colspan=1>12.795.2 0.035</td><td rowspan=1 colspan=4>34.899.6 0.047</td><td rowspan=1 colspan=1>90.2100.00.18</td><td rowspan=2 colspan=1>131.3100.02.2086.199.611.0</td></tr><tr><td rowspan=1 colspan=2>0.056</td><td rowspan=1 colspan=1>5.7999.6 0.064</td><td rowspan=1 colspan=4>18.699.70.34</td><td rowspan=1 colspan=1>72.399.02.61</td></tr><tr><td rowspan=2 colspan=1>Sampling</td><td rowspan=2 colspan=3>PARCO (s.)MACSIM (s.)</td><td rowspan=2 colspan=3>3.35100.0 0.0943.27100.00.48</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=4>5.50100.0 0.15</td><td rowspan=1 colspan=1>17.899.72.67</td><td rowspan=1 colspan=1>372.686.39.83</td></tr><tr><td rowspan=1 colspan=1>6.32100.0 0.68</td><td rowspan=1 colspan=4>24.0100.01.91</td><td rowspan=1 colspan=1>84.2100.042.0</td><td rowspan=1 colspan=1>125.8100.0381.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>PACT (s.)</td><td rowspan=1 colspan=3>2.89100.05.27</td><td rowspan=1 colspan=1>4.66100.05.45</td><td rowspan=1 colspan=4>15.5100.010.5</td><td rowspan=1 colspan=1>50.3100.051.9</td><td rowspan=1 colspan=1>84.0100.0136.9</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>DFJS</td><td rowspan=1 colspan=4>P</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4>N × M × O</td><td rowspan=1 colspan=3>20×8×6 (ID)</td><td rowspan=1 colspan=1>30×14×6 (ZST)</td><td rowspan=1 colspan=4>50×16×8 (ZST)</td><td rowspan=1 colspan=1>80×20×8 (ZST)</td><td rowspan=1 colspan=1>120×24×10 (ZST)</td></tr><tr><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=3>etric</td><td rowspan=1 colspan=3>Cost Succ. Time</td><td rowspan=1 colspan=1>CostSucc. Time</td><td rowspan=1 colspan=4>CostSucc. Time</td><td rowspan=1 colspan=1>CostSucc. Time</td><td rowspan=1 colspan=1>CostSucc.Time</td></tr><tr><td rowspan=2 colspan=1>Oracle</td><td rowspan=2 colspan=3>Gurobi (10m)OR-Tools (10m)</td><td rowspan=2 colspan=3>154.9100.0 624.6148.3 100.0 624.2</td><td rowspan=2 colspan=1>133.7100.0 626.3126.1100.0 624.2</td><td rowspan=1 colspan=4>291.7100.0 614.2</td><td rowspan=1 colspan=1>T/O T/OT/O</td><td rowspan=1 colspan=1>T/OT/OT/O</td></tr><tr><td rowspan=1 colspan=4>221.6100.0 600.6</td><td rowspan=1 colspan=1>T/O T/OT/O</td><td rowspan=1 colspan=1>T/O T/OT/O</td></tr><tr><td rowspan=2 colspan=1>Replan</td><td rowspan=2 colspan=3>GurobiOR-Tools</td><td rowspan=2 colspan=3>156.8100.028.8172.7 87.061.6</td><td rowspan=2 colspan=1>T/O T/OT/O149.4100.0 76.3</td><td rowspan=1 colspan=4>T/OT/OT/O</td><td rowspan=1 colspan=1>T/OT/OT/O</td><td rowspan=1 colspan=1>T/OT/OT/O</td></tr><tr><td rowspan=1 colspan=4>T/OT/OT/O</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>T/OT/OT/O</td><td rowspan=1 colspan=1>T/OT/OT/O</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=3>MAPPO (g.)IPPO (g.)</td><td rowspan=2 colspan=3>170.2 100.0 0.022207.796.1 0.051</td><td rowspan=2 colspan=1>148.4 100.0 0.021156.199.7 0.054</td><td rowspan=1 colspan=2>245.1</td><td rowspan=2 colspan=2>100.0 0.0700.24</td><td rowspan=2 colspan=1>299.0100.00.24380.198.60.87</td><td rowspan=2 colspan=1>429.2100.00.84584.498.53.45</td></tr><tr><td rowspan=1 colspan=2>295.698.7</td></tr><tr><td rowspan=3 colspan=1>Greedy</td><td rowspan=3 colspan=3>VDN (g.)DCG (g.)PARCO (g.)</td><td rowspan=3 colspan=3>632.154.4 0.056986.615.10.088170.5100.00.004</td><td rowspan=3 colspan=1>688.766.50.0761476.813.40.13146.0100.0 0.004</td><td rowspan=1 colspan=4>1641.357.70.33</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>2498.459.61.27</td></tr><tr><td rowspan=1 colspan=2>3089.97.4</td><td></td><td></td><td rowspan=2 colspan=1>5017.75.42.72336.297.40.054</td><td rowspan=1 colspan=1>9296.80.513.1</td></tr><tr><td rowspan=1 colspan=3>258.499.7</td><td rowspan=1 colspan=2>.015</td><td rowspan=1 colspan=1>2578.17.60.35</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=3>MACSIM (g.)PACT (g.)</td><td rowspan=2 colspan=3>608.354.70.18162.5 100.0 0.093</td><td rowspan=2 colspan=1>596.970.10.21141.5100.0 0.10</td><td rowspan=1 colspan=2>1241.360.2</td><td rowspan=2 colspan=2>0.48</td><td rowspan=2 colspan=1>1305.2 58.00.84283.9100.00.77</td><td rowspan=2 colspan=1>1866.3 61.21.64398.4</td></tr><tr><td rowspan=1 colspan=2>233.2</td><td rowspan=1 colspan=1>4 100.0 2.41</td></tr><tr><td rowspan=2 colspan=1>Sampling</td><td rowspan=2 colspan=3>PARCO (s.)MACSIM (s.)</td><td rowspan=2 colspan=3>165.0 100.0 0.087156.2100.00.20</td><td rowspan=2 colspan=1>142.8100.0 0.096136.4100.00.25</td><td rowspan=2 colspan=4>249.7100.0 0.18242.3100.00.93</td><td rowspan=1 colspan=1>314.5100.00.28</td><td rowspan=2 colspan=1>490.3100.02.30458.3100.05.24</td></tr><tr><td rowspan=1 colspan=1>307.2100.01.83</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>PACT (s.)</td><td rowspan=1 colspan=3>155.9100.010.5</td><td rowspan=1 colspan=1>134.1100.010.5</td><td rowspan=1 colspan=4>232.4100.023.6</td><td rowspan=1 colspan=1>279.6100.054.4</td><td rowspan=1 colspan=1>396.5100.0112.1</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>NCFFS</td><td rowspan=1 colspan=4>P</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4>N × Me × S</td><td rowspan=1 colspan=3>20×4×3 (ID)</td><td rowspan=1 colspan=1>30×6×3 (ZST)</td><td rowspan=1 colspan=4>50×7×3 (ZST)</td><td rowspan=1 colspan=1>80×6×4 (ZST)</td><td rowspan=1 colspan=1>120×8×4 (ZST)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Metric</td><td rowspan=1 colspan=3>Cost Succ. Time</td><td rowspan=1 colspan=1>CostSucc. Time</td><td rowspan=1 colspan=4>CostSucc. Time</td><td rowspan=1 colspan=1>CostSucc. Time</td><td rowspan=1 colspan=1>CostSucc. Time</td></tr><tr><td rowspan=2 colspan=1>Oracle</td><td rowspan=2 colspan=3>Gurobi (10m)OR-Tools (10m)</td><td rowspan=2 colspan=3>34.3100.0 624.532.0100.0124.0</td><td rowspan=2 colspan=1>30.7100.0 600.226.9100.0 603.6</td><td rowspan=1 colspan=4>41.2100.0 601.6</td><td rowspan=1 colspan=1>91.2100.0 602.2</td><td rowspan=2 colspan=1>183.5100.0 607.370.8100.0 600.2</td></tr><tr><td rowspan=1 colspan=4>36.0100.0 600.2</td><td rowspan=1 colspan=1>67.5100.0600.3</td></tr><tr><td rowspan=2 colspan=1>Replan</td><td rowspan=2 colspan=3>GurobiOR-Tools</td><td rowspan=2 colspan=3>37.5 100.0 15.737.1100.0 6.62</td><td rowspan=2 colspan=1>34.1100.0 19.336.5100.0 13.6</td><td rowspan=1 colspan=4>225.155.0 31.4</td><td rowspan=1 colspan=1>T/OT/OT/O</td><td rowspan=1 colspan=1>T/OT/OT/O</td></tr><tr><td rowspan=1 colspan=4>53.6100.0 32.2</td><td rowspan=1 colspan=1>351.396.0112.5</td><td rowspan=1 colspan=1>T/OT/OT/O</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=3>MAPPO (g.)IPPO (g.)</td><td rowspan=1 colspan=3>46.597.4 0.019</td><td rowspan=1 colspan=1>38.099.2 0.020</td><td rowspan=1 colspan=4>49.499.3 0.065</td><td rowspan=1 colspan=1>94.598.20.33</td><td rowspan=1 colspan=1>93.898.41.13</td></tr><tr><td rowspan=1 colspan=2> (g.)</td><td rowspan=1 colspan=3>47.398.5 0.017</td><td rowspan=1 colspan=1>39.399.3 0.020</td><td rowspan=1 colspan=4>52.698.9 0.070</td><td rowspan=1 colspan=1>99.897.80.30</td><td rowspan=1 colspan=1>97.198.61.05</td></tr><tr><td rowspan=5 colspan=1>Greedy</td><td rowspan=1 colspan=1>VD</td><td rowspan=1 colspan=2>VDN (g.)</td><td rowspan=1 colspan=3>50.097.80.017</td><td rowspan=1 colspan=1>44.296.8 0.027</td><td rowspan=1 colspan=4>63.993.20.12</td><td rowspan=1 colspan=1>151.084.41.04</td><td rowspan=1 colspan=1>186.181.64.62</td></tr><tr><td rowspan=2 colspan=3>DCG (g.)PARCO (g.)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>84.171.7 0.036</td><td rowspan=1 colspan=1>102.458.9 0.069</td><td rowspan=1 colspan=4>152.459.00.26</td><td rowspan=1 colspan=1>376.750.31.64</td><td rowspan=1 colspan=1>513.948.46.98</td></tr><tr><td rowspan=1 colspan=3>42.999.3 0.005</td><td rowspan=1 colspan=1>49.898.8 0.009</td><td rowspan=1 colspan=4>66.898.8 0.010</td><td rowspan=1 colspan=1>N/AN/AN/A</td><td rowspan=1 colspan=1>N/AN/AN/A</td></tr><tr><td rowspan=2 colspan=3>MACSIM (g.)PACT (g.)</td><td rowspan=2 colspan=3>64.782.90.06741.5100.0 0.049</td><td rowspan=1 colspan=1>55.491.7 0.057</td><td rowspan=1 colspan=4>98.486.60.11</td><td rowspan=1 colspan=1>212.682.70.29</td><td rowspan=1 colspan=1>213.989.10.35</td></tr><tr><td rowspan=1 colspan=1>35.9100.0 0.063</td><td rowspan=1 colspan=4>48.899.90.17</td><td rowspan=1 colspan=1>83.499.70.51</td><td rowspan=1 colspan=1>86.799.41.84</td></tr><tr><td rowspan=3 colspan=1>Sampling</td><td rowspan=2 colspan=3>PARCO (s.)MACSIM (s.)</td><td rowspan=1 colspan=3>37.3100.0 0.085</td><td rowspan=1 colspan=1>39.6100.0 0.099</td><td rowspan=1 colspan=4>57.4100.0 0.13</td><td rowspan=1 colspan=1>N/AN/AN/A</td><td rowspan=1 colspan=1>N/AN/AN/A</td></tr><tr><td rowspan=1 colspan=3>39.0100.00.21</td><td rowspan=1 colspan=1>36.1100.00.32</td><td rowspan=1 colspan=4>50.7100.00.71</td><td rowspan=1 colspan=1>97.3100.02.09</td><td rowspan=1 colspan=1>109.6100.03.87</td></tr><tr><td rowspan=1 colspan=3>PACT (s.)</td><td rowspan=1 colspan=3>36.2100.05.35</td><td rowspan=1 colspan=1>35.4100.06.17</td><td rowspan=1 colspan=4>44.8100.011.7</td><td rowspan=1 colspan=1>82.5100.032.5</td><td rowspan=1 colspan=1>85.6100.097.0</td></tr></table>

![](images/dd8cd2bad443992f59a56fcb4dae45258b8f9e804b916b9a0d7e62d2dc78a133.jpg)  
(a) MACTP

![](images/1ea69447254e111752b321182fbe901a62b6064c691d360fe170d594a260448a.jpg)  
(b) DFJSP

![](images/bc46b869f179d36ff1596cf689fb08e9412d9f581248e7a67510c9cb657d276c.jpg)  
(c) NCFFSP  
Figure 3: Validation return during training. Curves show the mean across ten seeds; shaded regions indicate one standard deviation. Higher is better.

Validation returns obtained during training are reported in Figure 3. On MACTP, PACT and the MARL baselines reach similar final returns. On DFJSP, PACT reaches high-return performance earlier than MAPPO and maintains it thereafter, with lower variability across seeds than VDN and DCG, whose returns deteriorate after initial gains. On NCFFSP, PACT also learns faster than MAPPO and attains a higher final mean return. We present PARCO and MACSIM’s training curves separately in Appendix C.1 because they require substantially more training steps. Despite larger budgets, PARCO attains lower best mean returns than PACT on all three problems, while MACSIM converges to a comparable return on MACTP and DFJSP but remains lower on NCFFSP.

## 5.3 ABLATION STUDY

We ablate PACT on the largest zero-shot transfer configuration of each problem under greedy evaluation and report cost and runtime relative to the reference model with $K = 4$ candidate plans and $R = 2$ coordination rounds (Figure 4). Appendix C.4 details the protocol and the analysis of each component. Coordination is the main source of PACT’s gains. Removing message passing on the plan-augmented trace-graph degrades performance to below that of MAPPO on DFJSP and NCFFSP, so candidate generation alone does not explain the improvement. Coordination also depends on having alternatives to choose from. With a single candidate plan per agent, performance degrades on every problem, whereas four candidates provide most of the benefit and larger sets yield no consistent gain. A second coordination round matters on MACTP, where plans span $H _ { p } = 4$ steps and capture multi-step interactions, while one round suffices on the scheduling problems with $H _ { p } = 1$ . Further rounds bring no consistent improvement at nearly unchanged runtime.

![](images/cdb7ede18ec19f14624f4db40eb57d98bb23fe7c350884ba5a36e23c53f123bb.jpg)  
(a) Effect of Coordination

![](images/4774abe635c3d49c1422391a5d3fc38060a158fae21081893d9ca6df3af2e467.jpg)  
(b) Number of Candidate Plans

![](images/6b801815183a40e51044dd2573fdd99915880fc661e86b4fe63f5fdb11686ebe.jpg)  
Figure 4: Analysis of PACT components.  
(c) Coordination Rounds

## 6 CONCLUSION

In this paper, we introduced PACT, a framework for DMACO that enables agents to coordinate over candidate plans through the environmental structure that induces their dependencies. By augmenting the environment graph with candidate plan traces and performing capacity-aware message passing over the resulting trace-graph, PACT extends coordination beyond the current decision step while retaining closed-loop execution as new information is revealed. Our experiments on routing and scheduling problems demonstrate that PACT consistently achieves strong solution quality and completion success across both in-distribution and zero-shot transfer settings, with particularly strong generalization to larger instances. In terms of limitations, the global communication assumption behind MPOMDPs may be relaxed in future work by imposing communication radiuses inside which information is shared. Additionally, PACT samples independently from agents’ policies $\pi _ { i }$ to construct the joint action at inference time. Stronger joint actions may be obtained by reasoning over the policies of other agents, for example by maintaining and updating beliefs.

## AI USE STATEMENT

In this work, we used generative AI tools to design or provide feedback on research methodology or experiments, and to implement methods. We have not used generative AI tools to generate synthetic data sets, help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, assist in the writing of proofs, propose or refine hypotheses, or interpret results. Cleaning and reformatting datasets and supporting qualitative and thematic data analysis are not applicable to this work. Additionally, we used generative AI tools to draft parts of the research paper, discover research topics or identify gaps, brainstorm, source/search for information, edit the research paper to improve readability, identify relevant literature, and assist with translation. We have reviewed all AI-assisted work. We checked all sources and LLM-generated research ideas for potential plagiarism through a manual literature review. All LLM-generated code was validated and tested for correctness. All LLM-generated writing has been manually reviewed for accuracy. We take full responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Source code used to implement the PACT method will be released upon publication of the paper. Appendix A gives the formal model and revelation rules of each problem, and Appendix B.3 specifies the instance generators. Appendix B describes the baseline implementations, and Appendix B.4 and Appendix B.5 report the training configurations and the compute environment. Appendix D details the algorithms of PACT.

## REFERENCES

Abderrahmane Aggoun and Nicolas Beldiceanu. Extending CHIP in order to solve complex scheduling and placement problems. Mathematical and computer modelling, 17(7):57–73, 1993.

Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: A methodological tour d’horizon. European Journal ofOperational Research, 290(2): 405–421, 2021.

Daniel S Bernstein, Robert Givan, Neil Immerman, and Shlomo Zilberstein. The complexity of decentralized control of markov decision processes. Mathematics of operations research, 27(4): 819–840, 2002.

Federico Berto, Chuanbo Hua, Laurin Luttmann, Jiwoo Son, Junyoung Park, Kyuree Ahn, Changhyun Kwon, Lin Xie, and Jinkyoo Park. PARCO: Parallel autoregressive models for multiagent combinatorial optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=9F2Cmgo17M.

Dimitris Bertsimas, Patrick Jaillet, and Sebastien Martin. Online vehicle routing: The edge of´ optimization in large-scale applications. Operations Research, 67(1):143–162, 2019.

Matteo Bettini, Amanda Prorok, and Vincent Moens. BenchMARL: Benchmarking multi-agent reinforcement learning. Journal ofMachine Learning Research, 25(217):1–10, 2024.

J. Blazewicz, J.K. Lenstra, and A.H.G.Rinnooy Kan. Scheduling subject to resource constraints: classification and complexity. Discrete Applied Mathematics, 5(1):11–24, 1983. ISSN 0166-218X. doi: https://doi.org/10.1016/0166-218X(83)90012-4. URL https://www. sciencedirect.com/science/article/pii/0166218X83900124.

Wendelin Bohmer, Vitaly Kurin, and Shimon Whiteson. Deep coordination graphs. In ¨ International Conference on Machine Learning, pp. 980–991. PMLR, 2020.

Quentin Cappart, Didier Chetelat, Elias B Khalil, Andrea Lodi, Christopher Morris, and Petar ´ Velickovi ˇ c. Combinatorial optimization and reasoning with graph neural networks. ´ Journal of Machine Learning Research, 24(130):1–61, 2023.

Abhishek Das, Theophile Gervet, Joshua Romoff, Dhruv Batra, Devi Parikh, Mike Rabbat, and´ Joelle Pineau. TarMAC: Targeted multi-agent communication. In International Conference on machine learning, pp. 1538–1546. PMLR, 2019.

Christian Schroeder De Witt, Tarun Gupta, Denys Makoviichuk, Viktor Makoviychuk, Philip HS Torr, Mingfei Sun, and Shimon Whiteson. Is independent learning all you need in the StarCraft multi-agent challenge? arXiv preprint arXiv:2011.09533, 2020.

Jakob Foerster, Ioannis Alexandros Assael, Nando De Freitas, and Shimon Whiteson. Learning to communicate with deep multi-agent reinforcement learning. In Advances in neural information processing systems, volume 29, 2016.

Carlos Guestrin, Michail G. Lagoudakis, and Ronald Parr. Coordinated reinforcement learning. In International Conference on Machine Learning, pp. 227–234, 2002.

Gurobi Optimization, LLC. Gurobi optimizer reference manual. https://www.gurobi.com, 2026.

Shengchao Hu, Li Shen, Ya Zhang, and Dacheng Tao. Learning multi-agent communication from graph modeling perspective. In International Conference on Learning Representations, volume 2024, pp. 12963–12978, 2024.

Chuanbo Hua, Federico Berto, Jiwoo Son, Seunghyun Kang, Changhyun Kwon, and Jinkyoo Park. CAMP: Collaborative Attention Model with Profiles for Vehicle Routing Problems. In Proceedings of the 2025 International Conference on Autonomous Agents and Multiagent Systems (AA MAS), 2025. https://github.com/ai4co/camp.

Michael Kearns, Yishay Mansour, and Andrew Y Ng. A sparse sampling algorithm for near-optimal planning in large markov decision processes. Machine learning, 49(2):193–208, 2002.

Jelle R. Kok and Nikos Vlassis. Collaborative multiagent reinforcement learning by payoff propagation. Journal ofMachine Learning Research, 7(65):1789–1828, 2006. URL http://jmlr. org/papers/v7/kok06a.html.

Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019. URL https://openreview. net/forum?id=ByxBFsRqYm.

Wouter Kool, Herke van Hoof, Joaquim Gromicho, and Max Welling. Deep policy dynamic programming for vehicle routing problems. In International conference on integration of constraint programming, artificial intelligence, and operations research, pp. 190–213. Springer, 2022.

Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, and Seungjai Min. POMO: Policy optimization with multiple optima for reinforcement learning. In Advances in neural information processing systems, volume 33, pp. 21188–21198, 2020.

Yeong-Dae Kwon, Jinho Choo, Iljoo Yoon, Minah Park, Duwon Park, and Youngjune Gwon. Matrix encoding networks for neural combinatorial optimization. In Advances in Neural Information Processing Systems, volume 34, pp. 5138–5149, 2021.

Laurin Luttmann and Lin Xie. Multi-action self-improvement for neural combinatorial optimization. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=6KrETIaOYD.

Julian Minder, Florian Grotschla, Jo ¨ el Mathys, and Roger Wattenhofer. SALSA-CLRS: A sparse and ¨ scalable benchmark for algorithmic reasoning. In The Second Learning on Graphs Conference, 2023. URL https://openreview.net/forum?id=PRapGjDGFQ.

Rajeev Motwani, Steven Phillips, and Eric Torng. Nonclairvoyant scheduling. Theoretical Computer Science, 130(1):17–47, 1994. ISSN 0304-3975. doi: https://doi.org/10. 1016/0304-3975(94)90151-1. URL https://www.sciencedirect.com/science/ article/pii/0304397594901511.

Frans A. Oliehoek and Christopher Amato. A Concise Introduction to Decentralized POMDPs. Springer, 2016.

Djamila Ouelhadj and Sanja Petrovic. A survey of dynamic scheduling in manufacturing systems. Journal ofscheduling, 12(4):417–431, 2009.

Cemal Ozg <sup>¨</sup> uven, Lale ¨ Ozbakır, and Yasemin Yavuz. Mathematical models for job-shop scheduling <sup>¨</sup> problems with routing and process plan flexibility. Applied mathematical modelling, 34(6):1539– 1548, 2010.

Christos H Papadimitriou and Mihalis Yannakakis. Shortest paths without a map. Theoretical Computer Science, 84(1):127–150, 1991.

Laurent Perron and Vincent Furnon. OR-Tools. URL https://developers.google.com/ optimization/.

Victor Pillac, Michel Gendreau, Christelle Gueret, and Andr´ es L Medaglia. A review of dynamic´ vehicle routing problems. European Journal ofOperational Research, 225(1):1–11, 2013.

Michael. Pinedo. Scheduling : theory, algorithms, and systems. Prentice-Hall international series in industrial and systems engineering. Prentice Hall, Englewood Cliffs, N.J, 1995. ISBN 0137067577.

Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. Monotonic value function factorisation for deep multi-agent reinforcement learning. Journal of Machine Learning Research, 21(178):1–51, 2020. URL http://jmlr. org/papers/v21/20-081.html.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

Davood Shiri and F. Sibel Salman. On the online multi-agent O-D k-Canadian traveler problem. Journal ofCombinatorial Optimization, 34(2):453–461, 2017.

Jiwoo Son, Minsu Kim, Sanghyeok Choi, Hyeonah Kim, and Jinkyoo Park. Equity-transformer: Solving NP-hard min-max routing problems as sequential generation with equity context. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 20265–20273, 2024.

Wen Song, Xinyang Chen, Qiqiang Li, and Zhiguang Cao. Flexible job-shop scheduling via graph neural network and deep reinforcement learning. IEEE Transactions on Industrial Informatics, 19(2):1600–1610, 2022.

Sainbayar Sukhbaatar, Rob Fergus, et al. Learning multiagent communication with backpropagation. In Advances in neural information processing systems, volume 29, 2016.

Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore Graepel. Valuedecomposition networks for cooperative multi-agent learning based on team reward. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, AA-MAS ’18, pp. 2085–2087, Richland, SC, 2018. International Foundation for Autonomous Agents and Multiagent Systems.

Paolo Toth and Daniele Vigo. Vehicle routing: problems, methods, and applications. SIAM, 2014.

Marlin W Ulmer. Anticipation versus reactive reoptimization for dynamic vehicle routing with stochastic requests. Networks, 73(3):277–291, 2019.

Guilherme E Vieira, Jeffrey W Herrmann, and Edward Lin. Rescheduling manufacturing systems: a framework of strategies, policies, and methods. Journal ofscheduling, 6(1):39–62, 2003.

Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of PPO in cooperative multi-agent games. In Advances in neural information processing systems, volume 35, pp. 24611–24624, 2022.

Zhi Zheng, Shunyu Yao, Zhenkun Wang, Tong Xialiang, Mingxuan Yuan, and Ke Tang. DPN: Decoupling partition and navigation for neural solvers of min-max vehicle routing problems. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=ar174skI9u.

Zefang Zong, Meng Zheng, Yong Li, and Depeng Jin. MAPDP: Cooperative multi-agent reinforcement learning to solve pickup and delivery problems. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pp. 9980–9988, 2022.

## A PROBLEM DEFINITIONS

This appendix defines the optimization problems and residual deterministic models $( \mathrm { i . e . }$ , models of the remaining problem at each decision step) used by the baselines.

## A.1 MACTP

Problem Definition. MACTP generalizes the Canadian Traveler Problem (Papadimitriou & Yannakakis, 1991) to a cooperative team that starts from distinct nodes and covers shared goals on a connected undirected graph with positive traversal costs. Agents try to reach each goal by traversing a graph, but a subset of edges may be blocked, with a given probability, and the true state of the edge is only revealed when an agent reaches one of its endpoints. The blockage probability of each edge is determined at instance generation time. In order to fulfill a goal, an agent must take the action to claim it, which retires that agent. Agents may pass through a goal without claiming it, and may pass through an already claimed goal. The episode succeeds once all goals have been claimed. A representative MACTP example was shown in Figure 1 in the main text.

We formulate the optimization problem as finding a minimum-cost collection of walks that jointly cover the remaining goals.

## Indices and Sets

$\mathcal { N } = \{ 1 , \ldots , n \}$ denotes the nodes, and $\mathcal { E } \subseteq \mathcal { N } \times \mathcal { N }$ contains the directed arcs induced by the usable undirected edges, with cost $w _ { v v ^ { \prime } } > 0$

${ \mathcal { D } } \subseteq { \mathcal { N } } { \mathrm { : } }$ : the set of goal nodes.

$i \in { \mathcal { M } } ;$ agents represented in the current planning problem, with agent i starting its residual walk at $s _ { i }$

${ \mathcal { D } } ^ { \mathrm { c o v } } \subseteq { \mathcal { D } } ;$ : goals still requiring coverage, with $G _ { t } = | \mathcal { D } ^ { \mathrm { c o v } } |$

## Decision Variables

$x _ { i , ( v , v ^ { \prime } ) } \in \mathbb { Z } _ { \geq 0 } \colon$ number of traversals of arc $( v , v ^ { \prime } )$ by agent i.

$f _ { i , ( v , v ^ { \prime } ) } \geq 0 ;$ auxiliary connectivity flow carried only on arcs used by agent i.

$y _ { i , v } \in \{ 0 , 1 \}$ indicates whether agent i is credited with covering v and is fixed to zero for $v \not \in { \mathcal { D } } ^ { \mathrm { c o v } }$

$e _ { i , v } \in \{ 0 , 1 \}$ : whether agent i ends its residual walk at $v \in { \mathcal { N } } .$

Objective Function

$$
\operatorname* { m i n } \quad \sum _ { i \in \mathcal { M } } \sum _ { ( v , v ^ { \prime } ) \in \mathcal { E } } w _ { v v ^ { \prime } } x _ { i , ( v , v ^ { \prime } ) }\tag{11}
$$

Subject to

$$
\sum _ { v \in \mathcal { N } } e _ { i , v } = 1 ,
$$

$$
\forall i \in { \mathcal { M } }\tag{12}
$$

$$
\sum _ { ( v , v ^ { \prime } ) \in \mathcal E } x _ { i , ( v , v ^ { \prime } ) } - \sum _ { ( v ^ { \prime } , v ) \in \mathcal E } x _ { i , ( v ^ { \prime } , v ) } = \mathbf 1 [ v = s _ { i } ] - e _ { i , v } ,
$$

$$
\forall i \in { \mathcal { M } } , v \in { \mathcal { N } }\tag{13}
$$

$$
\sum _ { ( v , v ^ { \prime } ) \in \mathcal { E } } f _ { i , ( v , v ^ { \prime } ) } - \sum _ { ( v ^ { \prime } , v ) \in \mathcal { E } } f _ { i , ( v ^ { \prime } , v ) } = \Big ( \sum _ { g \in \mathcal { D } ^ { \mathrm { c o v } } } y _ { i , g } \Big ) \mathbf { 1 } \big [ v = s _ { i } \big ] - y _ { i , v } \mathbf { 1 } \big [ v \in \mathcal { D } ^ { \mathrm { c o v } } \big ] , \quad \forall i \in M , v \in \mathcal { N }\tag{14}
$$

$$
f _ { i , ( v , v ^ { \prime } ) } \leq G _ { t } x _ { i , ( v , v ^ { \prime } ) } ,
$$

$$
\forall i \in { \mathcal { M } } , ~ ( v , v ^ { \prime } ) \in { \mathcal { E } }\tag{15}
$$

$$
\sum _ { i \in \mathcal { M } } y _ { i , v } = 1 ,
$$

$$
\forall v \in { \mathcal { D } } ^ { \mathrm { c o v } }\tag{16}
$$

Constraints and Online Replanning. The degree constraints define one walk per agent, while the auxiliary flow certifies that every credited goal lies on that walk. During online replanning, revealed blocked edges are removed, whereas unrevealed edges remain available under their optimistic status. If the revealed graph makes complete coverage infeasible, we replace Eq. (16) with $\begin{array} { r } { \bar { \sum _ { i } } y _ { i , v } + u _ { v } = 1 } \end{array}$ and lexicographically minimize the number of uncovered goals before travel cost.

MPOMDP Formulation. In the notation of Section 3, the latent variable ξ defines the blockage probability of the edges, while the graph state $\mathcal { G } _ { t }$ is the known graph state. Node features contain goal coverage and claim status. Edge features contain traversal cost, blocking probability, and the currently known status, while each agent carries its present location as an agent feature.

The active set $\mathcal { M } _ { t }$ contains the whole team, although retired agents can only remain at their current nodes and live agents move only through deterministic or revealed-open edges. An agent may retire by claiming an unclaimed goal, where unit capacity and unit claim demand enforce exclusivity, while non-goal nodes impose no shared constraint.

The step cost $C ( \mathcal { G } _ { t } , \mathbf { a } _ { t } , \xi )$ is the team’s total traversal cost, with retired agents incurring zero cost. Episodes end after all goals are covered or all agents retire, and truncated episodes additionally receive an estimate of the remaining completion cost.

## A.2 DFJSP

Problem Definition. In DFJSP, every job follows an ordered chain of operations, and each operation is eligible for a specified subset of machines. Some machines are unavailable from the beginning of an episode, but their condition remains hidden until the first attempted dispatch. A failed dispatch leaves the operation unscheduled and delays its job by a fixed amount, after which the broken machine is removed from future decisions. Failures are correlated through latent groups whose memberships and priors are observed, while the group condition and each machine-specific outcome remain hidden. Machines act as agents and may schedule operations before becoming idle, with processing beginning once both the selected machine and job are available. The objective function is the makespan of the resulting schedule. Figure 5 visualizes an illustrative DFJSP instance.

![](images/3780af304a7dab2fbba0fad146f6857321a84bf43eaf4c716e92e8d052d25618.jpg)  
Figure 5: Illustration of the DFJSP task. The upper panel pairs ordered job chains with machinespecific eligibility sets, where the lightning bolts mark machines that may have latent breakdowns. The lower panel gives a machine-wise schedule, where the dashed block records a failed dispatch to a broken machine and the solid block shows successful operation dispatch.

We formulate the optimization problem as a standard disjunctive FJSP over the remaining operations and the machines currently considered usable (Ozg<sup>¨</sup> uven et al., 2010).¨

## Indices and Sets

$j \in \mathcal I$ indexes jobs, chain(j) is the ordered operation chain of job $j ,$ and ${ \mathcal { O } } _ { t }$ contains the operations not yet scheduled at replanning step t.

$\operatorname { E l i g } _ { t } ( o ) \subseteq { \mathbb { Z } } :$ machines currently considered usable for operation o, with processing time $p _ { o , i } \stackrel { \cdot } { > } 0 \mathrm { f o r } i \in \mathrm { E l i g } _ { t } ( o )$

${ r } _ { o , t }$ is the earliest time permitted by the current job state, and $q _ { i , t }$ is the next time at which machine i becomes available.

## Decision Variables

$x _ { o , i } \in \{ 0 , 1 \}$ : whether operation o is assigned to machine $i \in \operatorname { E l i g } _ { t } ( o )$

$s _ { o } \geq 0$ is the start time of operation o, and $\begin{array} { r } { d _ { o } = \sum _ { i \in \mathrm { E l i g } _ { t } ( o ) } p _ { o , i } x _ { o , i } } \end{array}$ is its assigned duration.

$z _ { o , o ^ { \prime } } \in \{ 0 , 1 \}$ : relative order of two distinct operations that may share a machine, defined once for each unordered pair $o < o ^ { \prime }$

$C _ { \mathrm { m a x } } \geq 0$ is the makespan, and B is any valid upper bound on the completion time of the residual schedule.

## Objective Function

$$
\operatorname* { m i n } \quad C _ { \operatorname* { m a x } }\tag{17}
$$

Subject to

$$
\begin{array} { r l } { \sum } & { { } x _ { o , i } = 1 , } \end{array}
$$

$$
\forall o \in { \mathcal { O } } _ { t }\tag{18}
$$

i∈Elig<sub>t</sub>(o)

$$
C _ { \mathrm { m a x } } \geq s _ { o } + d _ { o } ,
$$

$$
\forall o \in { \mathcal { O } } _ { t }
$$

$$
s _ { o ^ { \prime } } \geq s _ { o } + d _ { o } ,\tag{19}
$$

$$
\forall ( o , o ^ { \prime } ) \mathrm { ~ c o n s e c u t i v e ~ i n ~ a ~ r e s i d u a l ~ j o b ~ c h a i n }\tag{20}
$$

$$
s _ { o } \geq r _ { o , t } ,
$$

$$
\forall o \in { \mathcal { O } } _ { t }\tag{21}
$$

$$
s _ { o } \geq q _ { i , t } - B ( 1 - x _ { o , i } ) ,
$$

$$
\forall o \in { \mathcal { O } } _ { t } , i \in \operatorname { E l i g } _ { t } ( o )\tag{22}
$$

$$
s _ { o ^ { \prime } } \geq s _ { o } + d _ { o } - B ( 1 - z _ { o , o ^ { \prime } } ) - B ( 2 - x _ { o , i } - x _ { o ^ { \prime } , i } ) , \quad \forall o < o ^ { \prime } , \ i \in \mathrm { E l i g } _ { t } ( o ) \cap \mathrm { E l i g } _ { t } ( o ^ { \prime } )\tag{23}
$$

$$
s _ { o } \geq s _ { o ^ { \prime } } + d _ { o ^ { \prime } } - B z _ { o , o ^ { \prime } } - B ( 2 - x _ { o , i } - x _ { o ^ { \prime } , i } ) , \qquad \forall o < o ^ { \prime } , \ i \in \mathrm { E l i g } _ { t } ( o ) \cap \mathrm { E l i g } _ { t } ( o ^ { \prime } )\tag{24}
$$

Constraints and Online Replanning. Each remaining operation is assigned once and must follow its job predecessor, while the release constraints preserve the current job and machine timelines. The pairwise disjunctions prevent operations on the same machine from overlapping. At each replanning step, completed operations and revealed-broken machines are removed, whereas unrevealed machines remain provisionally available with their known processing times.

MPOMDP Formulation. The latent variable ξ records the condition of every uncertain machine, with correlations induced by group-level variables. Each element of $\mathcal { G } _ { t }$ represents a job and records its next unscheduled operation and earliest feasible start time. Machine features contain availability, group membership, and the current condition estimate, while an eligibility edge carries the processing time for the job’s next operation.

The active set $\mathcal { M } _ { t }$ contains every machine not known to be broken, including busy machines, which may either dispatch an eligible job or wait. A scheduled operation starts no earlier than the release times of both participants, while unit job capacity permits at most one dispatch to each job per round.

The step cost $C ( \mathcal { G } _ { t } , \mathbf { a } _ { t } , \xi )$ is the increase in makespan, so its undiscounted sum equals $C _ { \mathrm { m a x } }$ for a completed schedule. A truncated episode additionally receives a penalty of processing all remaining operations serially on their slowest available machines, providing a pessimistic completion cost that prevents stalling from being beneficial.

## A.3 NCFFSP

Problem Definition. In NCFFSP, jobs pass through a common sequence of stages, and each stage is served by parallel machines whose processing times depend on both the job and the selected machine (Kwon et al., 2021). Some machines also have hidden condition factors that scale their processing times and are correlated through latent batches. Batch membership and its prior are observed, but a machine’s realized factor appears only after its first dispatch. Unlike DFJSP, every dispatch succeeds because uncertainty changes processing duration rather than machine eligibility, and busy machines may schedule jobs that begin once both participants are available. The objective is the final makespan, and non-clairvoyance here refers to the initially unknown machine-dependent duration factors (Motwani et al., 1994). Figure 6 illustrates an NCFFSP workflow.

![](images/4df69c2992df58b8c17ed587a441bf2e9d8010cb6d88578fb0ee9050b414d1e1.jpg)  
Figure 6: Illustration of the NCFFSP task. Jobs operate from left to right through consecutive parallel machine stages, where the lightning bolts mark machines that may have unrevealed duration factors.

We formulate the optimization problem over the remaining job stages as a flexible flow-shop model using the current processing times. Assignment, stage-precedence, release-time, and machine nonoverlap constraints are used.

## Indices and Sets

$\ell \in \{ 1 , \ldots , S \}$ indexes stages, and $\tau _ { \ell }$ contains the machines serving stage $\ell .$

$j , j ^ { \prime } \in \mathcal { I } _ { t } \colon$ jobs with at least one stage still to be scheduled in the residual problem.

$\bar { p } _ { \ell j i } > 0 $ : nominal processing time of job j on machine $i \in \mathcal { Z } _ { \ell }$

$\xi _ { \ell i } \mathbf { : }$ hidden condition factor of machine i. Before revelation, $p _ { \ell j i , t } = { \bar { p } } _ { \ell j i }$ , while afterwards it is the positive rounded value of $\bar { p } _ { \ell j i } \xi _ { \ell i }$

$r _ { \ell j , t }$ is the current release-time floor for job $j$ at stage ℓ, and ${ { q } _ { i , t } }$ is the next time at which machine i becomes available.

## Decision Variables

$x _ { \ell j i } \in \{ 0 , 1 \}$ : whether job $j$ is assigned to machine $i \in \mathcal { Z } _ { \ell }$ at stage ℓ.

$s \ell j \geq 0$ is the start time, and $\begin{array} { r } { d _ { \ell j } = \sum _ { i \in \mathcal { T } _ { \ell } } p _ { \ell j i , t } x _ { \ell j i } } \end{array}$ is the assigned duration.

$z _ { \ell j j ^ { \prime } } \in \{ 0 , 1 \}$ : relative order of distinct jobs $j < j ^ { \prime }$ that may share a machine at stage ℓ.

$C _ { \mathrm { m a x } } \geq 0$ is the makespan, and B is any valid upper bound on the completion time of the residual schedule.

## Objective Function

$$
\operatorname* { m i n } \quad C _ { \operatorname* { m a x } }\tag{25}
$$

## Subject to

$$
\sum _ { i \in \mathcal { T } _ { \ell } } x _ { \ell j i } = 1 ,
$$

$$
\forall j , \ell \mathrm { r e m a i n i n g f o r } j\tag{26}
$$

$$
C _ { \mathrm { m a x } } \geq s _ { \ell j } + d _ { \ell j } ,
$$

$$
\forall j , \ell \mathrm { r e m a i n i n g } \mathrm { f o r } j\tag{27}
$$

$$
s _ { \ell + 1 , j } \geq s _ { \ell j } + d _ { \ell j } ,
$$

$$
\forall j , \ell < S \mathrm { r e m a i n i n g } \mathrm { f o r } j\tag{28}
$$

$$
s _ { \ell j } \geq r _ { \ell j , t } ,
$$

$$
\forall j , \ell \mathrm { r e m a i n i n g } \mathrm { f o r } j\tag{29}
$$

$$
s _ { \ell j } \geq q _ { i , t } - B ( 1 - x _ { \ell j i } ) ,
$$

$$
\forall j , \ell \mathrm { r e m a i n i n g ~ f o r } j , i \in \mathcal { T } _ { \ell }\tag{30}
$$

$$
s _ { \ell j ^ { \prime } } \geq s _ { \ell j } + d _ { \ell j } - B ( 1 - z _ { \ell j j ^ { \prime } } ) - B ( 2 - x _ { \ell j i } - x _ { \ell j ^ { \prime } i } ) ,
$$

$$
\forall \ell \ : \mathrm { r e m a i n i n g } \ : \mathrm { f o r } \ : j , j ^ { \prime } , j < j ^ { \prime } , i \in \mathbb { Z } _ { \ell }\tag{31}
$$

$$
s _ { \ell j } \geq s _ { \ell j ^ { \prime } } + d _ { \ell j ^ { \prime } } - B z _ { \ell j j ^ { \prime } } - B ( 2 - x _ { \ell j i } - x _ { \ell j ^ { \prime } i } ) ,
$$

$$
\forall \ell \ : \mathrm { r e m a i n i n g } \ : \mathrm { f o r } \ : j , j ^ { \prime } , j < j ^ { \prime } , i \in \mathbb { Z } _ { \ell }\tag{32}
$$

Constraints and Online Replanning The assignment and non-overlap constraints play the same roles as in DFJSP, while Eq. (28) enforces the shared stage order. The release bounds preserve the current job and machine timelines, while online replanning uses nominal durations until a machine is revealed and realized durations thereafter. Every machine remains eligible for future assignments.

MPOMDP Formulation. NCFFSP uses the job-machine graph interface of DFJSP, but its latent variable ξ contains machine-dependent duration factors rather than binary failure states, and each job element records its next stage and earliest start time. Machine features store the served stage, current availability, batch membership, and either the prior or the revealed factor, while each eligibility edge carries the effective processing time known at that step.

Every machine remains in $\mathcal { M } _ { t }$ , including those that are currently busy, and may either dispatch a job at its stage or wait. Each job has unit capacity within a round, and a dispatch directed to that job contributes unit demand, while the scheduled start time respects both participants’ release times and prevents assignments on the same machine from overlapping.

The step cost $C ( \mathcal { G } _ { t } , \mathbf { a } _ { t } , \xi )$ is the increase in makespan, whose undiscounted sum equals the final makespan for a completed schedule. A truncated episode additionally receives a penalty of the makespan required to complete the schedule from the truncated state under a uniformly random legal policy. This penalty is separate from the NCFFSP objective and ensures that giving up is no better than acting randomly.

## B EXPERIMENTAL DETAILS

This appendix lists the baselines evaluated in Section 5 and the licenses of their corresponding code implementations. We also describe how instances are generated for each of the three problems, the training hyperparameters for every learning-based method, and the hardware and software used to run all experiments.

## B.1 BASELINES

Classic Solvers. The Gurobi and OR-Tools classic solvers are ran in two modes. The baselines marked Oracle in the results solve complete realizations using privileged access to the uncertainty with a time limit of 10 minutes. The Replan baselines repeatedly solve the remaining problem under the information available at each decision step, with a time limit of 2 seconds for each re-solve.

Gurobi. Gurobi (Gurobi Optimization, LLC, 2026) solves the mixed-integer linear programs in Appendix A using branch-and-cut. We retain the best feasible solution found for each instance under the time budget and report the mean objective value across the evaluation set.

OR-Tools. OR-Tools (Perron & Furnon) provides a complementary CP-SAT implementation under the same information settings and evaluation configurations.

Graph-based MARL Baselines. MAPPO, IPPO, VDN, and DCG use matched graph encoders over the common observation interface, and their per-agent networks share parameters within the team while the flow of information among agents follows the centralized or decentralized structure of each algorithm.

Table 2: Summary of licenses for baselines.
<table><tr><td>Resource</td><td>Type</td><td>License</td></tr><tr><td>Gurobi (Gurobi Optimization, LLC, 2026)</td><td>Solver (commercial)</td><td>Commercial, academic license</td></tr><tr><td>OR-Tools (Perron &amp; Furnon)</td><td>Solver (open-source library)</td><td>Apache License 2.0</td></tr><tr><td>BenchMARL (Bettini et al., 2024)</td><td>Library (MAPPO, IPPO, and VDN)</td><td>MIT License</td></tr><tr><td>DCG (Böhmer et al., 2020)</td><td>Code</td><td>Apache License 2.0</td></tr><tr><td>PARCO (Berto et al., 2025)</td><td>Code</td><td>MIT License</td></tr><tr><td>MACSIM (Luttmann &amp; Xie, 2026)</td><td>Code</td><td>Apache License 2.0</td></tr></table>

MAPPO. Multi-Agent PPO (Yu et al., 2022) uses a shared-parameter decentralized actor and a centralized critic, so each actor conditions on the common graph information and its own agent features while the critic aggregates information across the team. Both networks are trained with the clipped PPO objective and generalized advantage estimation.

IPPO. Independent PPO (De Witt et al., 2020) uses the same shared-parameter decentralized actor as MAPPO, but replaces the centralized critic with a decentralized value function. The critic produces one estimate per agent from that agent’s observation, and training uses the same clipped PPO objective and advantage estimator.

VDN. Value Decomposition Networks (Sunehag et al., 2018) learn shared per-agent action-value functions and select each agent’s action independently, after which the selected utilities are summed into a team value, $\begin{array} { r } { Q _ { \mathrm { t o t } } = \sum _ { i } Q _ { i } } \end{array}$ , trained from a common temporal-difference target.

DCG. Deep Coordination Graphs (Bohmer et al., 2020) combine per-agent utility terms with pair-¨ wise payoff terms defined on a fixed, fully connected coordination graph, and at each step max-plus message passing approximately maximizes the resulting joint action-value function to select the agents’ actions.

PARCO. PARCO (Berto et al., 2025) is a parallel autoregressive neural solver in which a shared transformer encodes the instance and communication layers exchange context among agent queries. A multiple-pointer decoder proposes simultaneous actions before a learned priority rule resolves conflicts and assigns fallback actions to displaced agents.

MACSIM. MACSIM (Luttmann & Xie, 2026) also constructs multi-agent solutions autoregressively from a shared instance representation. To produce a conflict-free set of simultaneous actions, it samples agent-task pairs without replacement from a joint categorical distribution at each construction step.

Neural Solver Adaptation. We implement task wrappers for PARCO and MACSIM through their respective environment interfaces while preserving the instances, transition rules, action constraints, and objectives used to evaluate the other methods.

Solution Decoding Strategies. We evaluate PARCO, MACSIM, and PACT under two solution decoding strategies: greedy and sampling. Greedy decoding uses one deterministic rollout, while sampling decoding generates 128 independent stochastic rollouts per method and instance and retains that method’s best realized solution (Berto et al., 2025; Luttmann & Xie, 2026). For greedy decoding, PARCO and MACSIM select the argmax of their action distributions, while in PACT the internal candidate plan generation, virtual rollout, and plan selection are all made deterministic.

## B.2 CODE AND LICENSES

Table 2 lists the external solvers and codebases used to implement the baselines above, together with their licenses.

## B.3 DATA GENERATION

MACTP. Following SALSA-CLRS (Minder et al., 2023), each instance uses a sparse, connected Erdos–R˝ enyi graph over´ n nodes, with agent starts and goals sampled uniformly at random and positive traversal costs assigned to its edges. In the reference configuration, 25% of the edges are stochastic, and their traversability is sampled at generation time but remains hidden until encoun tered, as described in Appendix A.1.

DFJSP. We follow the synthetic FJSP generation procedure of Song et al. (2022): each job contains four to six ordered operations, each operation is assigned a random subset of eligible machines, and its machine-dependent processing times vary within 20% of an operation-level mean. Each machine is initially designated stochastic with probability 0.6, after which a feasibility repair ensures that every operation retains at least one always-available eligible machine. Stochastic machines have fixed hidden availability and correlated breakdown probabilities within small machine groups, so observing one machine can inform the condition of others. Availability is revealed only when a job is first dispatched to the machine, and the resulting transition follows Appendix A.2.

NCFFSP. Each instance contains N jobs that pass through S processing stages, with several interchangeable machines available at each stage. Following Kwon et al. (2021), nominal job–machine processing times are sampled independently and uniformly from the integers $\{ 2 , \ldots , \bar { 9 } \}$ . Each real machine is independently designated stochastic with probability 0.6 and receives a fixed hidden condition factor generated from a group-correlated latent variable u $\in [ 0 , 1 ]$ through $\xi = 0 . 8 + 7 . 2 u ^ { 4 }$ The factor is revealed on the machine’s first dispatch and scales its processing times thereafter, as described in Appendix A.3.

## B.4 TRAINING CONFIGURATION

PACT and the four MARL baselines use a common number of training steps, data-collection scale, optimizer, and graph-encoder configuration across all three tasks. Table 3 reports these shared settings together with the update parameters specific to the on-policy and value-based families. We evaluate checkpoints on the same 64 evaluation episodes every 20,000 steps, and each configuration is trained with 10 random seeds.

PACT-specific Settings. PACT uses two coordination rounds in every task and is evaluated with K = 4 candidate plans per agent. During training, K is sampled from {2, 4, 8}, while the planning horizon is $H _ { p } = 4$ on MACTP and $H _ { p } = 1$ on DFJSP and NCFFSP. All three decision groups use a clipping range of 0.2. The proposal and rollout groups use an entropy coefficient of 0.001, while the selection group uses 0.01.

Neural Combinatorial Optimization Baselines. PARCO and MACSIM retain their original training objectives and model configurations, with PARCO using its rollout-based objective and MACSIM using self-improvement. Both methods use Adam with learning rate $1 \times 1 0 ^ { - 4 }$ and are trained for 50 epochs on each task.

Table 3: Training configuration for PACT and the MARL baselines.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Shared settings</td></tr><tr><td>Training steps</td><td>500,000</td></tr><tr><td>Validation interval</td><td>20,000 steps</td></tr><tr><td>Validation episodes</td><td>64</td></tr><tr><td>Random seeds</td><td>10</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Steps per batch</td><td>2,000</td></tr><tr><td>Parallel environments</td><td>4</td></tr><tr><td>Maximum gradient norm</td><td>5</td></tr><tr><td>Encoder layers</td><td>2</td></tr><tr><td>Hidden dimension</td><td>64</td></tr><tr><td>Aggregation</td><td>Weighted mean</td></tr><tr><td>Discount factor</td><td>0.99 (MACTP); 1.0 (DFJSP and NCFFSP)</td></tr><tr><td colspan="2">PPO settings (PACT, MAPPO, and IPPO)</td></tr><tr><td>Update epochs</td><td>4</td></tr><tr><td>Minibatch size</td><td>128</td></tr><tr><td>Clipping coefficient</td><td>0.2</td></tr><tr><td>GAE parameter λ</td><td>0.95</td></tr><tr><td>Entropy coefficient (MAPPO and IPPO)</td><td>0.01</td></tr><tr><td>Value-loss coefficient</td><td>1.0</td></tr><tr><td colspan="2">Value-based settings (VDN and DCG)</td></tr><tr><td>Updates per batch</td><td>200</td></tr><tr><td>Minibatch size</td><td>256</td></tr><tr><td>Replay-buffer capacity</td><td>30,000</td></tr><tr><td>Target-update coefficient</td><td>0.005</td></tr><tr><td>Initial exploration rate €</td><td>0.8</td></tr><tr><td>DCG-specific settings</td><td></td></tr><tr><td>Payoff rank</td><td></td></tr><tr><td>Max-plus iterations</td><td>8 4</td></tr><tr><td colspan="2">PACT-specific settings</td></tr><tr><td>Score step bound η</td><td>1.0 (fixed)</td></tr><tr><td>Selection temperature τ</td><td>Learned, initialized to 1.0, τ ≥ 0.25</td></tr><tr><td>Coordination MPNN layers</td><td>1</td></tr><tr><td>Coordination hidden dimension</td><td>64</td></tr></table>

## B.5 COMPUTE AND SOFTWARE

All learning-based methods were trained and evaluated on NVIDIA L40S GPUs, while the Gurobi and OR-Tools baselines were run on an Intel Core Ultra 7 265HX CPU. Experiments used Python 3.11, PyTorch 2.9.1, TorchRL 0.10.1, TensorDict 0.10.0, PyTorch Geometric 2.6.1, NumPy 2.1.3, and NetworkX 3.6.1. The optimization baselines used Gurobi 13.0.3 and OR-Tools 9.12.4544.

## C ADDITIONAL EXPERIMENTAL RESULTS

## C.1 TRAINING CURVES FOR PARCO AND MACSIM

Figures 7 and 8 show the validation returns of PARCO and MACSIM during training on the three DMACO problems. Curves show the mean across ten seeds, with shaded regions indicating one standard deviation. The horizontal red dashed line marks PACT’s highest mean validation return during training, and the red shaded band shows one standard deviation across seeds at the corresponding checkpoint. PACT achieves a higher best mean validation return than PARCO across all three problems and outperforms MACSIM on NCFFSP. While it converges to returns comparable to PACT on MACTP and DFJSP, MACSIM consistently requires substantially more training steps across all tasks. Overall, PACT demonstrates a significantly improved trade-off between sample efficiency and validation performance. Furthermore, despite occasional matching validation performance in-distribution, we nevertheless highlight the significantly higher performance on larger ZST instances reported in the main text.

![](images/a28f59f0a8243446b57c546916a5e96e7411738fd87c50b0676e3c75b75c93c9.jpg)  
(a) MACTP

![](images/3d6a46bb486a83a2ee34c08d1c9e4f03763e09085dc315ac82d32d64b64f3020.jpg)  
(b) DFJSP

![](images/0a1639c3f4d3b370820fd20b97938a818210689f17388c7faf2846463de0f908.jpg)  
(c) NCFFSP  
Figure 7: Validation return during training for PARCO on the three DMACO problems.

![](images/ce9b6ec8e4f14ba484f0c461d2800a306efa5b758b7ced58a6d073c1ef38c329.jpg)  
(a) MACTP

![](images/18ef7850005e617909825b025d5835f842285a407d6c5fee422cce6e932fbc03.jpg)  
(b) DFJSP

![](images/daaeef7321be95c62fa9c22cc501e7dece868b10e99a03f7d3bf1a40f61e76b3.jpg)  
(c) NCFFSP  
Figure 8: Validation return during training for MACSIM on the three DMACO problems.

## C.2 COMPLETE RESULTS FOR LEARNING-BASED METHODS

We evaluate each method on 100 instances for each of 10 independently trained seeds and report the mean across seeds. Success is the percentage of successfully completed instances, cost is the mean negated episodic return over all episodes, including penalties from failed or truncated episodes, whereas objective is the mean negated episodic return over successfully completed episodes. Per formance ratio is cost divided by the mean cost of PACT on the same problem and test configuration within the same evaluation group: greedy methods use greedy PACT as the reference, and sampling methods use sampling PACT. PACT has a ratio of one within each group, and lower values are better. All ratios are computed from the cost values as reported in the table. Time reports the end-to-end runtime per instance in seconds. Each entry reports the mean and one standard deviation across the ten training seeds; performance ratio is computed from the mean cost and reported without a standard deviation. N/A indicates that the method does not support the configuration.

Tables 4, 5, and 6 report these metrics under greedy and best-of-128 sampling evaluation.

Table 4: Complete results for learning-based methods on MACTP. The first configuration is indistribution (ID), and the remaining configurations are evaluated zero-shot (ZST).
<table><tr><td>Method</td><td>Evaluation</td><td>Success (%) ↑</td><td>Cost↓</td><td>Objective ↓</td><td>Perf. ratio ↓</td><td>Time (s) ↓</td></tr><tr><td colspan="7">16 × 4 × 6 (ID)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>96.8±2.1</td><td>3.83±0.25</td><td>3.53±0.06</td><td>1.046</td><td>0.027±0.030</td></tr><tr><td>IPPO</td><td>Greedy</td><td>97.1±2.0</td><td>3.92±0.36</td><td>3.57±0.11</td><td>1.071</td><td>0.015±0.007</td></tr><tr><td>VDN</td><td>Greedy</td><td>98.9±0.9</td><td>3.52±0.20</td><td>3.33±0.04</td><td>0.962</td><td>0.012±0.004</td></tr><tr><td>DCG</td><td>Greedy</td><td>99.3±0.7</td><td>3.68±0.47</td><td>3.55±0.26</td><td>1.005</td><td>0.011±0.003</td></tr><tr><td>PARCO</td><td>Greedy</td><td>98.9±1.2</td><td>4.56±0.23</td><td>4.34±0.13</td><td>1.246</td><td>0.004±0.002</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>89.9±3.7</td><td>7.36±0.68</td><td>5.55±0.35</td><td>2.011</td><td>0.038±0.021</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.1±1.4</td><td>3.66±0.18</td><td>3.43±0.07</td><td>1.000</td><td>0.056±0.015</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>3.35±0.04</td><td>3.35±0.04</td><td>1.159</td><td>0.094±0.016</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>3.27±0.09</td><td>3.27±0.09</td><td>1.131</td><td>0.48±0.07</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>2.89±0.02</td><td>2.89±0.02</td><td>1.000</td><td>5.27±0.49</td></tr><tr><td colspan="7">32 × 6 × 10 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>98.1±1.8</td><td>6.78±0.68</td><td>6.19±0.19</td><td>1.171</td><td>0.020±0.007</td></tr><tr><td>IPPO</td><td>Greedy</td><td>98.8±1.4</td><td>6.84±0.76</td><td>6.27±0.22</td><td>1.181</td><td>0.033±0.052</td></tr><tr><td>VDN</td><td>Greedy</td><td>99.6±1.0</td><td>5.84±0.40</td><td>5.69±0.23</td><td>1.009</td><td>0.019±0.019</td></tr><tr><td>DCG</td><td>Greedy</td><td>98.1±1.4</td><td>8.44±4.49</td><td>7.84±4.36</td><td>1.458</td><td>0.048±0.029</td></tr><tr><td>PARCO</td><td>Greedy</td><td>95.0±2.4</td><td>10.8±1.8</td><td>7.67±0.25</td><td>1.865</td><td>0.017±0.005</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>95.2±5.1</td><td>12.7±1.3</td><td>11.0±0.7</td><td>2.193</td><td>0.035±0.010</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.6±0.8</td><td>5.79±0.38</td><td>5.68±0.13</td><td>1.000</td><td>0.064±0.053</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>5.50±0.06</td><td>5.50±0.06</td><td>1.180</td><td>0.15±0.06</td></tr><tr><td>MACSIM PACT</td><td>Sampling</td><td>100.0±0.0</td><td>6.32±0.42</td><td>6.32±0.42</td><td>1.356</td><td>0.68±0.28</td></tr><tr><td></td><td>Sampling</td><td>100.0±0.0</td><td>4.66±0.05</td><td>4.66±0.05</td><td>1.000</td><td>5.45±0.67</td></tr><tr><td colspan="7">128 × 12 × 24 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>95.7±5.5</td><td>32.8±16.4</td><td>20.6±3.8</td><td>1.763</td><td>0.79±0.66</td></tr><tr><td>IPPO</td><td>Greedy</td><td>97.2±2.4</td><td>28.6±10.3</td><td>21.3±6.1</td><td>1.538</td><td>0.63±0.48</td></tr><tr><td>VDN</td><td>Greedy</td><td>99.9±0.3</td><td>20.7±4.5</td><td>20.3±4.1</td><td>1.113</td><td>0.38±0.15</td></tr><tr><td>DCG</td><td>Greedy</td><td>93.3±17.8</td><td>70.0±101.0</td><td>52.9±69.2</td><td>3.763</td><td>0.99±1.40</td></tr><tr><td>PARCO</td><td>Greedy</td><td>41.2±24.2</td><td>195.1±89.4</td><td>48.9±36.0</td><td>10.489</td><td>0.11±0.04</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>99.6±0.7</td><td>34.8±3.4</td><td>33.4±3.0</td><td>1.871</td><td>0.047±0.032</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.7±0.7</td><td>18.6±3.1</td><td>17.8±0.9</td><td>1.000</td><td>0.34±0.18</td></tr><tr><td>PARCO</td><td>Sampling</td><td>99.7±0.7</td><td>17.8±5.2</td><td>17.6±4.6</td><td>1.148</td><td>2.67±1.95</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>24.0±2.9</td><td>24.0±2.9</td><td>1.548</td><td>1.91±1.63</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>15.5±0.9</td><td>15.5±0.9</td><td>1.000</td><td>10.5±1.6</td></tr><tr><td colspan="7">512 × 24 × 48 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>IPPO</td><td>Greedy</td><td>00M</td><td>00M</td><td>00M</td><td>OOM</td><td>00M</td></tr><tr><td>VDN</td><td>Greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>DCG</td><td>Greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>PARCO</td><td>Greedy</td><td>4.2±8.4</td><td>1722.0±555.4</td><td>225.5±200.4</td><td>23.817</td><td>0.35±0.06</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>100.0±0.0</td><td>90.2±3.1</td><td>90.2±3.1</td><td>1.248</td><td>0.18±0.01</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.0±2.5</td><td>72.3±42.8</td><td>46.2±4.0</td><td>1.000</td><td>2.61±2.71</td></tr><tr><td>PARCO</td><td>Sampling</td><td>86.3±10.3</td><td>372.6±267.9</td><td>112.4±116.5</td><td>7.408</td><td>9.83±12.04</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>84.2±13.3</td><td>84.2±13.3</td><td>1.674</td><td>42.0±8.9</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>50.3±5.8</td><td>50.3±5.8</td><td>1.000</td><td>51.9±14.6</td></tr><tr><td colspan="7">1024 × 32 × 64 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>IPPO</td><td>Greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>VDN</td><td>Greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>DCG</td><td>Greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>00M</td><td>O0M</td></tr><tr><td>PARCO</td><td>Greedy</td><td>4.5±10.0</td><td>3999.2±1871.4</td><td>472.0±558.1</td><td>46.448</td><td>0.73±0.15</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>100.0±0.0</td><td>131.3±2.2</td><td>131.3±2.2</td><td>1.525</td><td>2.20±0.15</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.6±1.3</td><td>86.1±29.2</td><td>70.5±11.1</td><td>1.000</td><td>11.0±14.7</td></tr><tr><td>PARCO</td><td>Sampling</td><td>71.3±16.2</td><td>1341.6±349.6</td><td>187.1±214.0</td><td>15.971</td><td>27.5±27.9</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>125.8±15.0</td><td>125.8±15.0</td><td>1.498</td><td>381.7±31.0</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>84.0±10.8</td><td>84.0±10.8</td><td>1.000</td><td>136.9±27.9</td></tr></table>

Table 5: Complete results for learning-based methods on DFJSP. The first configuration is indistribution (ID), and the remaining configurations are evaluated zero-shot (ZST).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Evaluation</td><td rowspan="2">Success (%) ↑</td><td colspan="2">Cost ↓</td><td rowspan="2">Perf. ratio ↓</td><td rowspan="2">Time (s) ↓</td></tr><tr><td colspan="2">Objective ↓</td></tr><tr><td colspan="7">20 × 8 × 6 (ID)</td></tr><tr><td>MAPPO</td><td>Greedy Greedy</td><td>100.0±0.0 96.1±9.4</td><td>170.2±3.2 207.7±83.7</td><td>170.2±3.2 176.2±7.1</td><td>1.047 1.278</td><td>0.022±0.008 0.051±0.017</td></tr><tr><td>IPPO</td><td></td><td></td><td></td><td>181.7±6.4</td><td></td><td></td></tr><tr><td>VDN</td><td>Greedy</td><td>54.4±48.2</td><td>632.1±486.7</td><td></td><td>3.890</td><td>0.056±0.028</td></tr><tr><td>DCG</td><td>Greedy</td><td>15.1±32.1</td><td>986.6±411.5</td><td>187.8±14.6</td><td>6.071</td><td>0.088±0.004</td></tr><tr><td>PARCO</td><td>Greedy</td><td>100.0±0.0</td><td>170.5±0.4</td><td>170.5±0.4</td><td>1.049</td><td>0.004±0.001</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>54.7±20.3</td><td>608.3±212.0</td><td>166.1±4.3</td><td>3.743</td><td>0.18±0.03</td></tr><tr><td>PACT</td><td>Greedy</td><td>100.0±0.0</td><td>162.5±2.9</td><td>162.5±2.9</td><td>1.000</td><td>0.093±0.042</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>165.0±0.3</td><td>165.0±0.3</td><td>1.058</td><td>0.087±0.001</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>156.2±1.1</td><td>156.2±1.1</td><td>1.002</td><td>0.20±0.09</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>155.9±1.4</td><td>155.9±1.4</td><td>1.000</td><td>10.5±3.7</td></tr><tr><td colspan="7">30 × 14 × 6 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>100.0±0.0</td><td>148.4±3.2</td><td>148.4±3.2 152.4±4.8</td><td>1.049</td><td>0.021±0.009</td></tr><tr><td>IPPO</td><td>Greedy</td><td>99.7±0.7</td><td>156.1±12.5</td><td></td><td>1.103 4.867</td><td>0.054±0.017</td></tr><tr><td>VDN</td><td>Greedy</td><td>66.5±46.3</td><td>688.7±748.7</td><td>158.3±5.6</td><td></td><td>0.076±0.049</td></tr><tr><td>DCG</td><td>Greedy</td><td>13.4±28.1</td><td>1476.8±604.8</td><td>217.6±76.6</td><td>10.437</td><td>0.13±&lt;0.01</td></tr><tr><td>PARCO</td><td>Greedy</td><td>100.0±0.0</td><td>146.0±0.5</td><td>146.0±0.5</td><td>1.032</td><td>0.004±0.001</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>70.1±21.7</td><td>596.9±311.7</td><td>141.5±4.3</td><td>4.218</td><td>0.21±0.05</td></tr><tr><td>PACT</td><td>Greedy</td><td>100.0±0.0</td><td>141.5±3.6</td><td>141.5±3.6</td><td>1.000</td><td>0.10±0.05</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>142.8±0.5</td><td>142.8±0.5</td><td>1.065</td><td>0.096±0.001</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>136.4±1.2</td><td>136.4±1.2</td><td>1.017</td><td>0.25±0.09</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>134.1±1.9</td><td>134.1±1.9</td><td>1.000</td><td>10.5±4.3</td></tr><tr><td colspan="7">50 × 16 × 8 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>100.0±0.0</td><td>245.1±5.9</td><td>245.1±5.9</td><td>1.051</td><td>0.070±0.033</td></tr><tr><td>IPPO</td><td>Greedy</td><td>98.7±3.0</td><td>295.6±86.2</td><td>261.6±11.0</td><td>1.268</td><td>0.24±0.11</td></tr><tr><td>VDN</td><td>Greedy</td><td>57.7±50.0</td><td>1641.3±1647.5</td><td>256.2±18.6</td><td>7.038</td><td>0.33±0.27</td></tr><tr><td>DCG</td><td>Greedy</td><td>7.4±15.9</td><td>3089.9±1041.2</td><td>335.3±71.8</td><td>13.250</td><td>0.61±0.01</td></tr><tr><td>PARCO</td><td>Greedy</td><td>99.7±0.9</td><td>258.4±2.7</td><td>253.7±2.0</td><td>1.108</td><td>0.015±0.024</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>60.2±29.8</td><td>1241.3±195.5</td><td>247.3±9.2</td><td>5.323</td><td>0.48±0.13</td></tr><tr><td>PACT</td><td>Greedy</td><td>100.0±0.0</td><td>233.2±3.6</td><td>233.2±3.6</td><td>1.000</td><td>0.25±0.15</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>249.7±1.7</td><td>249.7±1.7</td><td>1.074</td><td>0.18±0.01</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>242.3±1.8</td><td>242.3±1.8</td><td>1.043</td><td>0.93±0.93</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>232.4±2.6</td><td>232.4±2.6</td><td>1.000</td><td>23.6±11.0</td></tr><tr><td colspan="7">80 × 20 × 8 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>100.0±0.0</td><td>299.0±9.8</td><td>299.0±9.8</td><td>1.053</td><td>0.24±0.12</td></tr><tr><td>IPPO</td><td>Greedy</td><td>98.6±3.5</td><td>380.1±162.5</td><td>319.9±16.3</td><td>1.339</td><td>0.87±0.49</td></tr><tr><td>VDN</td><td>Greedy</td><td>59.6±51.3</td><td>2498.4±2780.2</td><td>318.3±18.6</td><td>8.800</td><td>1.27±1.12</td></tr><tr><td>DCG</td><td>Greedy</td><td>5.4±11.4</td><td>5017.7±1604.1</td><td>452.9±148.6</td><td>17.674</td><td>2.72±0.02</td></tr><tr><td>PARCO</td><td>Greedy</td><td>97.4±5.9</td><td>336.2±32.6</td><td>314.6±4.6</td><td>1.184</td><td>0.054±0.084</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>58.0±32.2</td><td>1305.2±1130.6</td><td>312.5±11.6</td><td>4.597</td><td>0.84±0.24</td></tr><tr><td>PACT</td><td>Greedy</td><td>100.0±0.0</td><td>283.9±4.7</td><td>283.9±4.7</td><td>1.000</td><td>0.77±0.50</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>314.5±3.9</td><td>314.5±3.9</td><td>1.125</td><td>0.28±0.03</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>307.2±2.3</td><td>307.2±2.3</td><td>1.099</td><td>1.83±1.75</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>279.6±3.5</td><td>279.6±3.5</td><td>1.000</td><td>54.4±28.7</td></tr><tr><td colspan="7">120 × 24 × 10 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>100.0±0.0</td><td>429.2±20.5</td><td>429.2±20.5</td><td>1.077</td><td>0.84±0.45</td></tr><tr><td>IPPO</td><td>Greedy</td><td>98.5±4.4</td><td>584.4±354.4</td><td>469.7±30.9</td><td>1.467</td><td>3.45±2.43</td></tr><tr><td>VDN</td><td>Greedy</td><td>59.9±51.6</td><td>4311.0±4961.6</td><td>467.5±27.1</td><td>10.821</td><td>5.54±5.48</td></tr><tr><td>DCG</td><td>Greedy</td><td>0.5±1.6</td><td>9296.8±2400.1</td><td>1030.2±1062.3</td><td>23.335</td><td>13.1±0.1</td></tr><tr><td>PARCO</td><td>Greedy</td><td>7.6±13.7</td><td>2578.1±608.2</td><td>439.1±23.1</td><td>6.471</td><td>0.35±0.08</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>61.2±32.3</td><td>1866.3±641.7</td><td>462.0±19.7</td><td>4.684</td><td>1.64±0.52</td></tr><tr><td>PACT</td><td>Greedy</td><td>100.0±0.0</td><td>398.4±6.8</td><td>398.4±6.8</td><td>1.000</td><td>2.41±1.56</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>490.3±13.6</td><td>490.3±13.6</td><td>1.237 1.156</td><td>2.30±2.86</td></tr><tr><td>MACSIM PACT</td><td>Sampling Sampling</td><td>100.0±0.0 100.0±0.0</td><td>458.3±2.9 396.5±4.9</td><td>458.3±2.9 396.5±4.9</td></table>

Table 6: Complete results for learning-based methods on NCFFSP. The first configuration is indistribution (ID), and the remaining configurations are evaluated zero-shot (ZST).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Evaluation</td><td rowspan="2">Success (%) ↑</td><td colspan="2">Cost↓</td><td rowspan="2">Perf. ratio ↓</td><td rowspan="2">Time (s) ↓</td></tr><tr><td colspan="2">Objective ↓</td></tr><tr><td colspan="7">20 × 4 × 3 (ID)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>97.4±3.5 98.5±1.9</td><td>46.5±2.3 47.3±1.7</td><td>44.2±1.0 45.4±1.5</td><td>1.120 1.140</td><td>0.019±0.006 0.017±0.004</td></tr><tr><td>IPPO</td><td>Greedy</td><td></td><td></td><td></td><td></td><td>0.017±0.005</td></tr><tr><td>VDN</td><td>Greedy</td><td>97.8±2.2</td><td>50.0±2.1</td><td>48.6±1.5</td><td>1.205</td><td></td></tr><tr><td>DCG</td><td>Greedy</td><td>71.7±31.6</td><td>84.1±50.6</td><td>56.1±17.0</td><td>2.027</td><td>0.036±0.011</td></tr><tr><td>PARCO</td><td>Greedy</td><td>99.3±0.7</td><td>42.9±0.5</td><td>42.5±0.4</td><td>1.034</td><td>0.005±0.002</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>82.9±32.5</td><td>64.7±43.0</td><td>45.9±5.9</td><td>1.559</td><td>0.067±0.022</td></tr><tr><td>PACT</td><td>Greedy</td><td>100.0±0.0</td><td>41.5±0.7</td><td>41.5±0.7</td><td>1.000</td><td>0.049±0.019</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>37.3±0.2</td><td>37.3±0.2</td><td>1.030</td><td>0.085±0.002</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>39.0±1.5</td><td>39.0±1.5</td><td>1.077</td><td>0.21±0.03</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>36.2±0.6</td><td>36.2±0.6</td><td>1.000</td><td>5.35±0.26</td></tr><tr><td colspan="7">30 × 6 × 3 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy Greedy</td><td>99.2±1.3 99.3±1.3</td><td>38.0±1.9 39.3±1.0</td><td>37.3±0.8 38.5±1.2</td><td>1.058 1.095</td><td>0.020±0.007 0.020±0.007</td></tr><tr><td>IPPO VDN</td><td>Greedy</td><td>96.8±2.8</td><td>44.2±3.1</td><td>41.3±1.4</td><td>1.231</td><td>0.027±0.010</td></tr><tr><td>DCG</td><td>Greedy</td><td>58.9±25.5</td><td>102.4±53.7</td><td>52.3±13.4</td><td>2.852</td><td>0.069±0.005</td></tr><tr><td>PARCO</td><td>Greedy</td><td>98.8±0.9</td><td>49.8±1.2</td><td>48.7±1.4</td><td>1.387</td><td>0.009±0.005</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>91.7±23.3</td><td>55.4±46.4</td><td>41.2±5.7</td><td>1.543</td><td>0.057±0.032</td></tr><tr><td>PACT</td><td>Greedy</td><td>100.0±0.0</td><td>35.9±1.9</td><td>35.9±1.9</td><td>1.000</td><td>0.063±0.041</td></tr><tr><td>PARCO</td><td></td><td></td><td>39.6±0.4</td><td>39.6±0.4</td><td>1.119</td><td>0.099±0.002</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>36.1±2.3</td><td>36.1±2.3</td><td>1.020</td><td>0.32±0.04</td></tr><tr><td>PACT</td><td>Sampling Sampling</td><td>100.0±0.0 100.0±0.0</td><td>35.4±0.9</td><td>35.4±0.9</td><td>1.000</td><td>6.17±0.29</td></tr><tr><td colspan="7">50 × 7 × 3 (ZST)</td></tr><tr><td>MAPPO Greedy</td><td>Greedy</td><td>99.3±1.3</td><td>49.4±2.8 52.6±2.5</td><td>48.0±0.7 49.7±2.6</td><td>1.012 1.078</td><td>0.065±0.027</td></tr><tr><td>IPPO</td><td></td><td>98.9±1.9</td><td>63.9±12.5</td><td>53.5±2.4</td><td>1.309</td><td>0.070±0.032 0.12±0.07</td></tr><tr><td>VDN</td><td>Greedy</td><td>93.2±8.3</td><td></td><td></td><td></td><td></td></tr><tr><td>DCG</td><td>Greedy</td><td>59.0±30.0</td><td>152.4±79.0</td><td>75.4±30.9</td><td>3.123</td><td>0.26±0.03</td></tr><tr><td>PARCO</td><td>Greedy</td><td>98.8±2.2</td><td>66.8±3.5</td><td>65.3±1.3</td><td>1.369</td><td>0.010±0.006</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>86.6±30.5</td><td>98.4±104.2</td><td>61.3±19.1</td><td>2.016</td><td>0.11±0.07</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.9±0.3</td><td>48.8±2.4</td><td>48.4±1.1</td><td>1.000</td><td>0.17±0.14</td></tr><tr><td>PARCO</td><td>Sampling</td><td>100.0±0.0</td><td>57.4±2.3</td><td>57.4±2.3</td><td>1.281</td><td>0.13±0.09</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>50.7±4.0</td><td>50.7±4.0</td><td>1.132</td><td>0.71±0.09</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>44.8±1.5</td><td>44.8±1.5</td><td>1.000</td><td>11.7±0.6</td></tr><tr><td colspan="7">80 × 6 × 4 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy</td><td>98.2±3.3</td><td>94.5±11.8</td><td>88.5±1.3</td><td>1.133</td><td>0.33±0.24</td></tr><tr><td>IPPO</td><td>Greedy</td><td>97.8±4.5</td><td>99.8±14.1</td><td>91.6±5.0</td><td>1.197</td><td>0.30±0.24</td></tr><tr><td>VDN</td><td>Greedy</td><td>84.4±11.7</td><td>151.0±42.7</td><td>98.8±2.9</td><td>1.811</td><td>1.04±0.42</td></tr><tr><td>DCG</td><td>Greedy</td><td>50.3±18.1</td><td>376.7±137.6</td><td>141.8±53.5</td><td>4.517</td><td>1.64±0.03</td></tr><tr><td>PARCO</td><td>Greedy</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>82.7±28.5</td><td>212.6±191.5</td><td>114.4±27.9</td><td>2.549</td><td>0.29±0.16</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.7±0.7</td><td>83.4±8.9</td><td>81.1±2.3</td><td>1.000</td><td>0.51±0.57</td></tr><tr><td>PARCO</td><td>Sampling</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>97.3±7.1</td><td>97.3±7.1</td><td>1.179</td><td>2.09±0.66</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>82.5±2.9</td><td>82.5±2.9</td><td>1.000</td><td>32.5±1.6</td></tr><tr><td colspan="7">120 × 8 × 4 (ZST)</td></tr><tr><td>MAPPO</td><td>Greedy Greedy</td><td>98.4±3.5 98.6±3.5</td><td>93.8±16.1</td><td>86.2±2.5 90.2±4.4</td><td>1.082 1.120</td><td>1.13±1.22 1.05±1.15</td></tr><tr><td>IPPO</td><td></td><td></td><td>97.1±16.2</td><td>96.8±2.8</td><td>2.146</td><td>4.62±1.60</td></tr><tr><td>VDN</td><td>Greedy</td><td>81.6±13.2</td><td>186.1±65.9</td><td></td><td></td><td></td></tr><tr><td>DCG</td><td>Greedy</td><td>48.4±27.9</td><td>513.9±241.2</td><td>186.0±99.1</td><td>5.927</td><td>6.98±0.35</td></tr><tr><td>PARCO</td><td>Greedy</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MACSIM</td><td>Greedy</td><td>89.1±23.1</td><td>213.9±222.1</td><td>123.6±29.2</td><td>2.467</td><td>0.35±0.29</td></tr><tr><td>PACT</td><td>Greedy</td><td>99.4±1.1</td><td>86.7±22.6</td><td>80.4±2.4</td><td>1.000</td><td>1.84±2.40</td></tr><tr><td>PARCO</td><td>Sampling</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MACSIM</td><td>Sampling</td><td>100.0±0.0</td><td>109.6±8.9</td><td>109.6±8.9</td><td>1.280 1.000</td><td>3.87±0.62 97.0±10.0</td></tr><tr><td>PACT</td><td>Sampling</td><td>100.0±0.0</td><td>85.6±4.4</td><td>85.6±4.4</td></table>

## C.3 RUNTIME ANALYSIS

Table 7 decomposes PACT’s end-to-end runtime into environment stepping, policy inference, and other costs. Policy inference is further divided into encoding, candidate plan generation, trace-graph coordination, plan selection, and remaining policy operations. We report the ID and largest ZST configuration for each task to show how the distribution changes with problem scale.

The runtime profiles reflect both problem scale and the planning horizon $H _ { p }$ . On MACTP, where $H _ { p } = 4$ , plan generation accounts for 38.02% on the ID configuration and remains at 42.15% on the largest ZST configuration because each of the K candidates requires a multi-step autoregressive virtual rollout. By contrast, DFJSP and NCFFSP use $H _ { p } = 1$ , so plan generation ends after the initial candidate actions and its share falls from 23.82% and 27.49% to 3.74% and 5.27% as the instances grow. Their larger graph states instead make encoding dominant, accounting for 86.37% and 87.76% of runtime. Encoding also increases from 11.91% to 43.77% on MACTP, but its longer rollouts keep plan generation comparably expensive. The relative runtime of coordination decreases from 9.95-13.25% on the ID configurations to 2.81-4.42% on the largest configurations, while selection remains below 1.3% across the evaluated configurations.

Table 7: Breakdown of PACT’s end-to-end runtime. Component columns report percentages of the end-to-end runtime, and the final column reports time per instance.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Configuration</td><td rowspan="2">Env. step</td><td colspan="5">Policy</td><td rowspan="2">Other</td><td rowspan="2">Time (s)</td></tr><tr><td>Encoder</td><td>Plan Coord.</td><td>Selection</td><td></td><td>Other</td></tr><tr><td rowspan="2">MACTP</td><td>16 × 4 × 6 (ID)</td><td>15.77%</td><td>11.91%</td><td>38.02%</td><td>9.95%</td><td>0.5%</td><td>15.45%</td><td>8.39%</td><td>0.056</td></tr><tr><td>1024 × 32 × 64 (ZST)</td><td>5.32%</td><td>43.77%</td><td>42.15%</td><td>2.81%</td><td>0.14%</td><td>4.5%</td><td>1.31%</td><td>11.0</td></tr><tr><td rowspan="2">DFJSP</td><td>20 × 8 × 6 (ID)</td><td>19.04%</td><td>15.35%</td><td>23.82%</td><td>12.92%</td><td>1.28%</td><td>23.5%</td><td>4.08%</td><td>0.093</td></tr><tr><td>120 × 24 × 10 (ZST)</td><td>2.38%</td><td>86.37%</td><td>3.74%</td><td>4.42%</td><td>0.07%</td><td>2.51%</td><td>0.53%</td><td>2.41</td></tr><tr><td rowspan="2">NCFFSP</td><td>20 × 4 × 3 (ID)</td><td>19.71%</td><td>15.84%</td><td>27.49%</td><td>13.25%</td><td>0.69%</td><td>18.59%</td><td>4.43%</td><td>0.049</td></tr><tr><td>120 × 8 × 4 (ZST)</td><td>2.17%</td><td>87.76%</td><td>5.27%</td><td>3.01%</td><td>0.02%</td><td>1.26%</td><td>0.52%</td><td>1.84</td></tr></table>

## C.4 ABLATION STUDY DETAILS

In this section, we provide further details regarding the evaluation protocol and results of the ablation study reported in Section 5.3 in the main text.

All ablation experiments are conducted with greedy evaluation on the largest zero-shot transfer configuration of each problem. We use the PACT model with K = 4 candidate plans and $R = 2$ coordination rounds as the reference model. We define the performance ratio as cost divided by the reference model’s mean cost on the same problem, and compute the time ratio in the same way using end-to-end runtime. Both ratios equal one for the reference model, and lower values are better. We present the ablation results in Figure 4. For each method or configuration, bars show mean performance ratios across seeds, with error bars indicating one standard deviation, while lines show mean time ratios.

Effect of Coordination. We compare PACT, PACT without coordination, and MAPPO in Figure 4(a). PACT without coordination omits message passing on the plan-augmented trace-graph, so each agent selects its candidate plan independently. PACT achieves the lowest mean performance ratio on all three problems. Removing coordination substantially degrades performance, particularly on MACTP and NCFFSP, and makes it worse than MAPPO on DFJSP and NCFFSP. This indicates that the gains arise from using shared elements to mediate candidate selection rather than from candidate generation alone.

Number of Candidate Plans. We vary the number of candidate plans $K \in \{ 1 , 4 , 8 , 1 6 \}$ in Figure 4(b). With K = 1, each agent exposes only one plan to the trace-graph, so coordination cannot redirect it to an alternative. This setting degrades performance on every problem and produces high variability on NCFFSP, whereas four candidates provide most of the observed improvement. Larger sets yield no consistent gain beyond the variation across seeds. The time ratio reflects both per-step computation and episode length, so longer episodes can offset the savings from fewer candidates. The sharp escalation in the time ratio on MACTP at large K is consistent with $H _ { p } = 4 ,$ , compared with $H _ { p } = 1$ on DFJSP and NCFFSP.

Number of Coordination Rounds. We vary the number of coordination rounds $R \in \{ 1 , 2 , 3 , 4 \}$ in Figure 4(c). The first round updates candidate plans from the loads induced by the initial preferences. Subsequent rounds recompute these loads after the preferences change, allowing plans to respond to how other agents have adjusted their selections. One round performs comparably on the scheduling problems but yields a higher mean performance ratio on MACTP, while the second closes this gap. This difference is consistent with MACTP’s longer plan traces capturing multi-step interactions, whereas the scheduling tasks use $H _ { p } = 1$ . Further rounds provide no consistent improvement, and runtime remains nearly unchanged.

## D DETAILED ALGORITHMS

Algorithm 1 summarizes how PACT maps the graph observation at one decision step to a joint action. The three algorithms below develop this procedure in sequence: candidate plan generation (Algorithm 2), coordination on the plan-augmented trace-graph (Algorithm 3), and plan selection followed by execution (Algorithm 4). Together, they complement the corresponding descriptions in Section 4.1-Section 4.3 with a more explicit account of the forward pass.

Algorithm 2 Candidate plan generation for agent i (detail of Section 4.1)   
1: Input: graph observation $\mathcal { G } _ { t } ;$ encodings $h _ { G }$ and $\sigma _ { i }$ from Eq. (2); agent i; candidate plan count   
K; plan horizon $H _ { p }$   
2: Output: candidate plans $\hat { p } _ { i } ^ { 1 } , \ldots , \hat { p } _ { i } ^ { K }$ and their traces $\mathrm { T r } ( \hat { p } _ { i } ^ { k } )$   
3: $\hat { x } _ { i , 0 } ^ { k }$ ← current state of agent i from $\mathcal { G } _ { t }$ for every k   
4: draw $\hat { a } _ { i , t } ^ { 1 } , \dots , \hat { a } _ { i , t } ^ { K }$ jointly, without replacement, from $\pi _ { \theta } ^ { \mathrm { p l a n } } ( \cdot \mid h _ { G } , \sigma _ { i } , \hat { x } _ { i , 0 } ^ { k } )$   
5: for $k = 1 , \ldots , \bar { K }$ do   
6: $\hat { x } _ { i , 1 } ^ { k } \gets$ rollout state of agent i after $\hat { a } _ { i , t } ^ { k }$   
7: for $\mathbf { \bar { \rho } } _ { j } = 1 , \dots , H _ { p } - 1$ do   
8: sample $\hat { a } _ { i , t + j } ^ { k } \sim \pi _ { \theta } ^ { \mathrm { p l a n } } ( \cdot \mid h _ { G } , \sigma _ { i } , \hat { x } _ { i , j } ^ { k } )$   
9: $\hat { x } _ { i , j + 1 } ^ { k } \gets$ rollout state after $\hat { a } _ { i , t + j } ^ { k }$   
10: end for   
11: $\hat { p } _ { i } ^ { k } \gets ( \hat { a } _ { i , t } ^ { k } , \dots , \hat { a } _ { i , t + H _ { p } - 1 } ^ { k } )$   
12: $\mathrm { T r } ( \hat { p } _ { i } ^ { k } ) \gets \{ ( v , d ( v , \hat { a } _ { i , t + j } ^ { k } ) , j ) : d ( v , \hat { a } _ { i , t + j } ^ { k } ) > 0 \}$   
13: end for   
14: return $\{ \hat { p } _ { i } ^ { k } , \mathrm { T r } ( \hat { p } _ { i } ^ { k } ) \} _ { k = 1 } ^ { K }$

Algorithm 3 Coordination on the plan-augmented trace-graph (detail of Section 4.2)   
1: Input: candidate plans $\{ \hat { p } _ { i } ^ { k } , \mathrm { T r } ( \hat { p } _ { i } ^ { k } ) \}$ for all $i \in \mathcal { M } _ { t } , k = 1 , \ldots , K ;$ coordination rounds R   
2: Output: candidate plan scores $s _ { i } ^ { ( R ) }$ for all $i \in \mathcal { M } _ { t }$   
3: $\chi _ { t } \gets \mathcal G _ { t }$ with one candidate plan node per $\hat { p } _ { i } ^ { k }$ and a trace arc $( \hat { p } _ { i } ^ { k } , v )$ for every $( v , d , j ) \in \mathrm { T r } ( \hat { p } _ { i } ^ { k } )$   
4: $h ^ { ( 0 ) }  h _ { G }$ for element nodes and an encoding of the node’s trace for candidate plan nodes   
5: $s _ { i } ^ { ( 0 ) } ( k )  \eta$ tanh $\left( f _ { \mathrm { s c o r e } } ( h ^ { ( 0 ) } ( \hat { p } _ { i } ^ { k } ) ) \right)$ ， $q _ { i } ^ { ( 0 ) }$ ← softmax $( s _ { i } ^ { ( 0 ) } )$ (Eq. (5))   
6: for $r = 0 , \ldots , R - \mathrm { 1 }$ do   
7: for each element v on which some trace places demand do   
8: $\begin{array} { r } { L ^ { ( r ) } ( v ) \gets \sum _ { i } \sum _ { k } q _ { i } ^ { ( r ) } ( k ) d ( \underline { { v } } , \hat { p } _ { i } ^ { k } ) \ : ( \mathrm { E q . } \ : ( 6 ) ) } \end{array}$   
9: $\begin{array} { r } { \tilde { h } ^ { ( r ) } ( v ) \ : = \ : h ^ { ( r ) } ( v ) \ : + \ : W _ { \mathrm { l o a d } } \left[ \frac { L ^ { ( r ) } ( v ) } { \kappa _ { t } ( v ) } , \ : \frac { \mathrm { R e L U } \left( L ^ { ( r ) } ( v ) - ( \kappa _ { t } ( v ) - d _ { t } ( v ) ) \right) } { \kappa _ { t } ( v ) } \right] } \end{array}$ , (Eq. (7): normalized   
by the capacity $\kappa _ { t } ( v ) )$   
10: end for   
11: $h ^ { ( r + 1 ) } \gets \mathrm { M P N N } ( \tilde { h } ^ { ( r ) } , \chi _ { t } )$ (Eq. (8))   
12: $s _ { i } ^ { ( r + 1 ) } ( k )  s _ { i } ^ { ( r ) } ( k ) + \eta$ tanh $\left( f _ { \mathrm { s c o r e } } ( h ^ { ( r + 1 ) } ( \hat { p } _ { i } ^ { k } ) ) \right)$ (Eq. (9))   
13: $q _ { i } ^ { ( r + 1 ) } \gets \mathrm { s o f t m a x } ( s _ { i } ^ { ( r + 1 ) } )$   
14: end for   
15: return $\{ s _ { i } ^ { ( R ) } \} _ { i \in \mathcal { M } _ { t } }$

Algorithm 4 Plan selection and execution (detail of Section 4.3)   
1: Input: candidate plans $\{ \hat { p } _ { i } ^ { k } \}$ and candidate plan scores $\{ s _ { i } ^ { ( R ) } \}$ for all $i \in \mathcal { M } _ { t } ;$ selection tem  
perature τ   
2: Output: executed joint action $\mathbf { a } _ { t }$   
3: for each agent $i \in \mathcal { M } _ { t }$ (in parallel) do   
4: z<sub>i</sub> ∼ softmax ${ ( s _ { i } ^ { ( R ) } / \tau ) }$   
5: $a _ { i , t } \gets \hat { p } _ { i } ^ { z _ { i } } [ 0 ]$   
6: end for   
7: $\mathbf { a } _ { t } \gets \{ a _ { i , t } \} _ { i \in \mathcal { M } _ { t } }$   
8: submit $\mathbf { a } _ { t }$ to the environment, update the state, observation, and cost (Section 4.3)   
9: return $\mathbf { a } _ { t }$