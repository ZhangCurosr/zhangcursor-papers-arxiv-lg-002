# Reinforcement Learning-Guided Graph Transformations for SpTRSV Optimization

Buse Yılmaz<sup>1\*</sup>

<sup>1\*</sup>Department of Computer Engineering, MEF University, Huzur, Maslak Ayaza˘ga Cd., <sup>˙</sup>Istanbul, Turkey.

Corresponding author(s). E-mail(s): yilmazbuse@mef.edu.tr;

## Abstract

Sparse triangular solve (SpTRSV) is a fundamental kernel in numerous scientific and engineering applications. However, the data dependencies inherent in sparse triangular matrices significantly limit the available parallelism and make eficient workload distribution challenging. Recent graph transformation techniques address these limitations by modifying the dependency graph of the input matrix to improve parallel execution. Existing graph transformation strategies, however, rely on manually designed heuristics, making their development and adaptation to diferent optimization objectives challenging. This work proposes a reinforcement learning-guided graph transformation framework for SpTRSV, in which graph transformation is formulated as a sequential decision-making problem and an RL agent learns matrix-dependent transformation policies. Experimental results on real-world sparse matrices demonstrate level reductions of up to 94% and reductions of up to 80% in the coeficient of variation of level costs, while modifying only 1.50% of the rows in the highest case. On average, the RLguided graph transformation achieves a 23% reduction in the number of levels and a 29% reduction in the coeficient of variation of level costs while rewriting only 0.82% of the matrix rows. Although the heuristic strategies generally achieve more aggressive level reduction(between 31% and 46%), the RL-based approach achieves the largest average reduction in the coeficient of variation of level costs, demonstrating its ability to balance competing graph transformation objectives. The results further show that the learned policies can be transferred to previously unseen matrices through curriculum learning and fine-tuning, while zero-shot experiments provide insights into the limitations of generalizing graph transformation policies across diferent sparsity patterns.

Keywords: Sparse triangular solve, reinforcement learning, parallel computing, performance optimization, graph transformation

## 1 Introduction

There are several diferent approaches to optimize the parallel execution of Sparse Triangular Solve (SpTRSV) in the literature. Block diagonal approaches [1–5] partition the graph while keeping the sparsity pattern fixed and map these partitions to appropriate architectures and SpTRSV approaches. Hence, these methods aim to optimize the execution mapping and its scheduling while the equation rewriting method [6] modifies the sparsity pattern to optimize the schedule itself. Graph transformation strategies [7, 8] leverage this method to apply structure-aware schedule optimization of SpTRSV by transforming the directed acyclic graph (DAG) of a sparse matrix into a more homogeneous and balanced graph by heuristic approaches. These strategies reduce the synchronization depth (critical path of the DAG) and improve workload balance to optimize the parallel execution eficiency of SpTRSV. In this sense, graph transformation as an optimization approach can be considered close to compiler optimizations such as IR transformations or loop transformations rather than classical SpTRSV tuning. Graph transformation problem [6] presents a large combinatorial search space, as every rewriting operation changes the graph structure and consequently afects future rewriting opportunities.

In this work, rather than designing increasingly complex hand-crafted heuristics, we formulate graph transformation as a reinforcement learning problem. The goal is not only to obtain high-quality graph transformations but also to provide a framework for systematically observing how local rewriting decisions afect the global structure of the DAG. This enables the learned policy to serve both as an optimization method and as a tool for analyzing the efectiveness of diferent graph characteristics and transformation criteria, thereby facilitating the development of future heuristic or learning based strategies.

Using the proposed framework, a reinforcement learning (RL) based graph transformation strategy is developed to optimize SpTRSV. A subtle diference of this RL based strategy from the graph transformation strategies in the literature [7, 8] is providing a structure-aware learning policy rather than relying on the hand-crafted heuristics based on sparsity pattern features. The heuristic-based strategies prescribe the desired graph structure, whereas the RL agent learns a sequence of graph transformations that maximizes a long-term objective. Therefore, two approaches are optimizing at diferent levels of abstraction.

Our contributions are summarized below:

• We develop an extensible reinforcement learning environment that serves as an automated heuristic discovery framework for graph transformation, enabling systematic evaluation of both learned policies and heuristic-based strategies.

• We propose a novel graph transformation strategy for SpTRSV by formulating graph transformation as a reinforcement learning problem, replacing manually designed transformation rules with a learned policy.

• We demonstrate that reinforcement learning can efectively learn graph transformation policies that optimize multiple competing objectives, reducing the need for manual heuristic engineering.

• We design a multi-objective reward function balancing critical-path reduction, level workload balance, rewriting cost and transformation progress.

• We evaluate the learned transformation policies on real-world sparse matrices and demonstrate level reductions of up to 94% and up to 80% reduction in the coeficient of variation of level workloads, while requiring graph modifications to only a small fraction of the rows which goes as high as 1.58%. The results demonstrate RL-based graph transformation’s ability to balance competing graph transformation objectives. On average, the RL-guided graph transformation achieves a 23% reduction in the number of levels and a 29% reduction in the coeficient of variation of level costs while rewriting only 0.82% of the matrix rows. The heuristic-based strategies [7] achieve average level reduction between 31% and 46% and 19% to 22% reduction in the coeficient of variation while rewriting a percentage of rows between 0.33% and 0.81%.

The organization of the paper is as follows: Section 2 provides the fundamentals of SpTRSV, equation rewriting method and reinforcement learning, Section 3 introduces the problem definition, Section 4 explains the reinforcement learning based graph transformation framework in detail, Section 5 presents the experiment results and Section 6 concludes while introducing the future work and the last section presents the literature work.

## 2 Background

## 2.1 Sparse Triangular Solve and Level-Set Method

The sparse triangular solve operation solves the linear system Lx = b, where L is a sparse lower triangular matrix and b is a dense right-hand side vector. Solving Lx = b requires computing each unknown entry in x vector only after all of its predecessors have been computed. These dependencies are represented as a directed acyclic graph (DAG), where each node corresponds to a row of the triangular system and each edge represents a computational dependency.

Sparse triangular solve is usually parallelized on CPUs using the level-set method [9–13], which constructs a directed acyclic graph (DAG) representing the data dependencies among matrix rows and partitioning it into levels horizontally such that all rows within the same level are independent and can therefore be processed concurrently. During execution, all rows in a level are assigned to a group of threads and computed in parallel, while synchronization barriers enforce the dependencies between consecutive levels. As a result, the levels are processed sequentially despite the parallelism available within each level.

The efectiveness of the level-set method largely depends on the structural characteristics of the sparse matrix. Irregular sparsity patterns frequently produce levels with highly imbalanced computational workloads or only a few rows, therefore, a few computations. Levels with very few computations underutilize the available processing resources, whereas heavily populated levels may introduce additional communication and scheduling overhead when their workload is distributed among multiple threads.

Furthermore, matrices with long dependency chains require many synchronization barriers, increasing execution overhead and limiting scalability.

## 2.2 Equation Rewriting

To address the limitations and challenges posed by the level-set methods, the equation rewriting method [6] was introduced as a graph transformation technique for restructuring the dependency graph before execution. The method modifies the dependency structure by replacing selected row dependencies with their predecessors, thereby preserving correctness while enabling rows to migrate to earlier levels in the DAG. This transformation reduces the critical path by eliminating levels whenever possible and improves workload balance by merging computationally sparse levels. Consequently, the transformed DAG exposes greater parallelism while reducing synchronization overhead.

As indicated in [7, 8], equation rewriting is performed entirely as a preprocessing step and is independent of the underlying SpTRSV solver. After transformation, the modified dependency graph is stored in CSR format and can be processed by existing level-set-based solvers that can matrices in CSR format without requiring any changes to the solver implementation. A detailed description of the equation rewriting methodology and its heuristic-based graph transformation strategies is available in [6–8].

## 2.3 Reinforcement Learning

Reinforcement learning (RL) [14] is a machine learning paradigm in which an agent learns to make sequential decisions by interacting with an environment. At each time step, the agent observes the current state of the environment, selects an action according to a policy, and receives a reward that evaluates the quality of the selected action. The environment subsequently transitions to a new state, and the interaction continues until a terminal condition is reached. The objective of the agent is to learn a policy that maximizes the expected cumulative reward over an episode.

Unlike supervised learning[14], reinforcement learning does not require labeled training data. Instead, the agent discovers efective decision-making strategies through trial-and-error interactions with the environment. This characteristic makes RL particularly suitable for sequential optimization problems in which the quality of a decision depends not only on its immediate efect but also on its long-term consequences. Consequently, RL has been successfully applied to a wide range of optimization problems, including scheduling, resource allocation, compiler optimization, and graph-based learning tasks [15, 16].

## 2.4 Proximal Policy Optimization

Proximal Policy Optimization (PPO) [17] is a policy-gradient reinforcement learning algorithm that has gained widespread adoption due to its training stability and implementation simplicity. PPO employs an actor–critic architecture consisting of a policy network, which determines the probability of selecting an action, and a value network, which estimates the expected cumulative reward of a state. During training, the policy is updated using a clipped surrogate objective that restricts the magnitude of policy changes between successive updates to prevent excessively large parameter updates that may destabilize learning while maintaining eficient policy improvement.

Compared with earlier policy-gradient methods, PPO achieves a favorable balance between computational eficiency, robustness, and sample eficiency, making it well suited for environments with large and discrete action spaces. Owing to these advantages, PPO has become one of the most widely used reinforcement learning algorithms and serves as the learning algorithm adopted in this work.

## 3 Motivation

Equation rewriting [6] provides an efective mechanism for restructuring dependency graphs and existing approaches [7, 8] rely on manually designed heuristics to determine which rows should be rewritten. Designing such heuristics requires significant domain expertise and often involves balancing multiple, sometimes conflicting, optimization objectives. Reinforcement learning ofers an alternative by learning graph transformation policies directly through interaction with the rewriting environment.

Rather than focusing solely on the eficient execution of sparse triangular solve (SpTRSV), graph transformation techniques [7, 8] aim to optimize the dependency graph itself before execution. By determining which rows are rewritten, which dependencies are removed or introduced, and how the resulting levels are formed, graph transformation directly influences the execution schedule of the level-set method. The quality of the transformed graph determines important execution characteristics, including the critical path length, workload distribution across levels, synchronization frequency, and ultimately the achievable parallel performance.

Designing efective graph transformation strategies is challenging because the optimization objectives are strongly interdependent and often conflicting. For example, collapsing levels shortens the critical path but may significantly increase the computational cost of the remaining levels. Likewise, improving workload balance may reduce opportunities for future level collapses, while transformations that reduce synchronization overhead may adversely afect memory locality or communication costs. Existing graph transformation approaches [7, 8] therefore rely on manually designed heuristics based on matrix features, such as the average level cost (ALC), average number of rows per level (ARL), and average number of incoming dependencies per row (AIR), to balance these competing objectives. Although efective, such model-driven heuristics require considerable domain expertise and are inherently limited by the assumptions embedded in their design.

In this work, graph transformation is formulated as a reinforcement learning problem to automate heuristic discovery and design new graph transformation strategies. Instead of manually defining fixed rewriting rules, a reinforcement learning agent learns graph transformation policies through interaction with the graph rewriting environment and optimization of a multi-objective reward function. This enables the exploration of graph transformations that may not be considered by manually designed heuristics and provides a flexible policy-driven framework in which alternative optimization objectives, observation features, and reward formulations can be investigated without redesigning the underlying graph rewriting algorithm. Unlike previous heuristic-based strategies, which primarily rewrite rows between levels with a few rows only, the proposed framework allows the learned policy to select source levels from the entire graph while suitable target levels are chosen primarily from levels with a few rows only and from the rest as a fallback strategy. This additional flexibility enables the discovery of alternative graph topologies that may provide improved workload balance and shorter critical paths.

The proposed framework builds upon the same deterministic graph rewriting engine used by existing heuristic-based approaches. Reinforcement learning is responsible only for determining the sequence of graph transformations, whereas the graph rewriting engine performs the actual dependency modifications while preserving correctness. Consequently, the proposed approach does not replace graph rewriting; instead, it replaces manually designed heuristic decision rules with a learned graph transformation policy. Therefore, the proposed framework provides an extensible platform for automatically discovering execution-aware graph transformation strategies.

## 4 Methodology

![](images/969a4279e85b4a026bacd171350a98c11941a661357e344066e9435a02886465.jpg)  
Fig. 1 Graph Transformation Framework Chainbreaker and the relationship between its modules. DG: dependency graph.

Figure 1 illustrates Chainbreaker framework. The Reinforcement Learning Graph Transformation (RLGT) module and the Graph Data Transfer (GDT) module are introduced in this work as extensions to the graph transformation framework presented in [7, 8]. The RLGT module formulates graph transformation as a reinforcement learning problem. We refer to the whole framework as Chainbreaker to diferentiate between the main framework and the RL-based framework (RLGT and GDT modules) introduced in this work. The GDT module is implemented in C++ using pybind11 as the communication layer between the C++ framework and the Pythonbased RLGT module. It serves as the interface between the reinforcement learning components and the remaining modules of the framework. Since the overall architecture has been described in [7, 8] previously, only the components relevant to the reinforcement learning extension are presented in this section.

When the reinforcement learning graph transformation strategy is selected, the framework operates in either training or inference mode and GDT module prepares the DAG together with the level table containing the rows on each level and level costs. The prepared data object is transferred to RLGT Module where a Maskable PPO is used to learn a graph transformation policy. RLGT module generates a rewriting list just as its heuristic-based counterpart. The generated rewriting list is sent back to GDT Module and from there to the rewriting module which applies the actual graph transformation on the DAG. After the graph transformation is completed, the transformed DAG is converted back into a lower triangular matrix and stored on disk in CSR format by the write module.

The following subsections describe the formulation of the environment, including the observation and action spaces, target selection strategy, reward function, and training procedure.

## 4.1 Environment

The environment models the current state of the dependency graph throughout the graph transformation process. At the beginning of an episode, the environment is initialized from the graph data received from GDT Module. Besides the graph structure, two matrix feature-based criteria that are also used in heuristic-based graph transformation strategies [7], namely, Average Level Cost (ALC) and Average Row Length (ARL) are computed and remain constant throughout the episode. In [7], a level is classified as thin if its computational cost is smaller than the average level cost (ALC) and thick otherwise. Thin levels represent candidate source and target levels. Removing nodes from thin levels can eventually empty them, hence the critical path of the graph can be shortened by collapsing the empty levels. In addition, moving nodes into target levels can improve workload balance across levels without immediately overloading the level since thin target levels become thicker.

The original heuristic defines thin levels based on ALC, however this definition becomes problematic for the RL-based graph transformation strategy due to high AIR values or outlier coeficient of variation (CV) of level cost values. Matrices with high AIR may trigger large increases in level costs after rewriting, causing thin levels to disappear rapidly and reducing the number of feasible target levels. As a result, the agent’s exploration space shrinks and episodes may terminate prematurely.

In addition, when a matrix has a low CV of level cost, most level costs are close to the average. Consequently, after several rewrites, these levels quickly become thick, causing the agent’s exploration space to shrink. This efect is amplified for matrices with high AIR values. Conversely, defining thin levels solely based on ALC becomes insuficient for matrices with a high CV of level cost, where the level cost distribution is highly skewed. ALC alone does not capture whether level costs are concentrated around the average or widely dispersed. As a result, the agent may prematurely exhaust its search space, causing episodes to terminate after only a small number of rewrites. On the other hand, when many thin target levels remain available, the agent may continue thickening them without inducing level collapses, resulting in unnecessarily long episodes before convergence.

Under these conditions, the destination-selection heuristic becomes ill-conditioned because it relies on the availability of thin target levels. To preserve a suficient exploration space throughout an episode, the RL environment incorporates two mechanisms: (1) defining the thin-level threshold as a function of ALC and CV of level costs, and (2) introducing a fallback mechanism for target level selection to avoid premature episode termination. These modifications preserve the original destination-selection heuristic whenever possible while preventing artificial episode termination.

To preserve the exploration space under these conditions, we adapt the definition of the thin-level threshold as follows:

$$
t h i n . l e v e l . t h r e s h o l d = A L C * ( 1 . 0 + a l p h a * m a x ( 0 , 1 . 0 - c v \_ l e v e l \_ c o s t ) )
$$

The hyperparameter alpha controls the degree of relaxation. The term $( 1 - C V )$ ensures that the multiplicative factor applied to ALC is maximized for low CV values, where a greater relaxation of thin level threshold is required. Since CV is not bounded by 1, max(0, 1 − CV ) guarantees that the relaxation term becomes zero whenever $C V > 1$ , preserving the original threshold. Consequently, the threshold is increased only for matrices with low CV, while matrices that do not exhibit this characteristic continue to use ALC value.

After each rewriting operation, the environment incrementally updates the following values without modifying the DAG, allowing eficient interaction with the RL agent.

• node-to-level assignments,

• computational cost of each level,

• number of nodes per level,

• level indegrees,

• thin level set,

• total number of nonempty levels.

## 4.2 State Representation

The observation contains all information required to determine the next environment state after applying a rewriting action. The environment state consists of 9 global features together with 6 level-specific features describing the current graph state. These features are provided in Table 1. All the values are kept in a normalized form.

The agent selects one action (source level) from the masked valid actions. Therefore, at each step diferent source levels can be selected. Since reducing the critical path requires eliminating entire levels, the agent must repeatedly select the same source level until it becomes empty. Given the large exploration search space, relying only on the valid action mask is insuficient to achieve such goal. Therefore, each level is assigned source and target afinity values which are used to indicate the ”hotness” of a level, guiding the agent to continue working on the same source level until it becomes empty while repeatedly selecting the same target level to create localized node accumulation on the graph. At each step, the afinity values associated with the selected source and target levels are updated to reflect recent rewriting activity. The afinity scores gradually decay over time, allowing the agent to prioritize recently visited levels while avoiding permanent bias. Levels that become empty or cease to be suitable candidates receive negative afinity values to discourage their future selection.

Table 1 Global information about the current graph state and local information describing individual levels.
<table><tr><td>Level-specific features</td><td>Global features</td></tr><tr><td>Computational cost of each level Number of nodes in each level</td><td>Current and reference CV of level costs Current and reference CV of level node counts</td></tr><tr><td></td><td></td></tr><tr><td>Source level affinity values</td><td>Level count</td></tr><tr><td>Target level affinity values</td><td>Rewrite ratio</td></tr><tr><td>Completion ratio of each level</td><td>Thin levels completion ratio</td></tr><tr><td>Thin levels</td><td>Critical path ratio</td></tr><tr><td></td><td>Thin-level ratio</td></tr></table>

Completion ratio is used for a previous source level which is abandoned without being emptied. It is calculated as 1.0 − (curr level size/(init level size + 1)). Based on this value, a certain penalty is calculated as part of the reward function. The observation space keeps completion ratio for each level as well as a global feature. The completion ratio thin is a global feature that is the ratio of the current number of thin levels to the initial thin level count measuring how many thin levels are remaining as a working space for the agent. When this ratio falls below a certain threshold, the episode is terminated.

Coeficient of variation of level costs and node counts per level are observed since one of the objectives is to balance the workload across levels. As the agent progresses, the number of nodes and the total cost of each level changes as well as the number of levels. These quantities evolve throughout the episode as graph transformations modify the workload distribution across levels. The reference coeficient of variation values are kept at 1.0.

Rewrite ratio is the ratio of total number of rewrites to the total number of nodes. The agent might rewrite a previously rewritten row again as opposed to the heuristicbased strategies where a row is rewritten from a source level to a target level only once. Due to its sequential decision-making nature, the RL-based strategy transforms the graph incrementally by performing one rewrite at a time while exploring diferent graph configurations, rather than applying a single predetermined rewrite per row. Consequently, the RL-based strategy inherently incurs a higher rewriting overhead than its heuristic-based counterparts. Rewriting is a costly operation, therefore reducing the rewriting cost requires the maximum level reduction and coeficient of variation of level costs reduction with minimum number of rewrites. Critical path ratio observes the ratio of the current number of levels to the initial number of levels in order to track the level count reduction.

## 4.3 Action Space

The action space consists of selecting a source level from which a row will be picked for rewriting, in other words, the node will be moved to an upper level. Although the rewriting happens at node level, the actions chosen by the agent are not nodes but levels. This design considerably reduces the action space compared to selecting individual nodes while still allowing the agent to influence the rewriting process. As a straightforward and simple approach, the node with the smallest number of dependencies is chosen to be rewritten aiming at minimizing the rewriting cost.

Algorithm 1 Move node to next thin level   
Require: Node $n ,$ source level $l _ { s }$   
Ensure: Target level $l _ { t }$   
1: $l _ { t } \gets$ FindPrevLevel $( l _ { s } )$   
2: if $l _ { t }$ is thin then   
3: if $l _ { t } > 0$ then   
4: $l _ { t } \gets$ SelectHighestAffinityThinLevel $( l _ { t } ,$ SEARCH WINDOW)   
5: else   
6: $l _ { t } \gets$ FindNearestNonEmptyLevel $\left( l _ { s } \right)$   
7: if $l _ { t }$ is empty then return -1   
8: end if   
9: end if   
10: else   
11: if $l _ { t } > 0$ then   
12: $l _ { t } \gets$ SelectHighestAffinityThinLevel $\mathbf { \Omega } _ { \cdot } ( l _ { t } ,$ SEARCH WINDOW)   
13: else   
14: $l _ { t } \gets$ FindNearestNonEmptyLevel $. ( l _ { s } )$   
15: if $l _ { t }$ is empty or $l _ { t }$ is thick then return -1   
16: end if   
17: end if   
18: end if   
19: if $l _ { s } - l _ { t } > M A X$ REWRITE DISTANCE then return -1   
20: end if   
<sub>21:</sub> UpdateGraphState $( n , l _ { s } , l _ { t } )$   
22: Append $\langle n , l _ { s } , l _ { t } \rangle$ to rewriting list   
23: return $l _ { t }$

Not every level represents a legal action. A level is considered valid only if it satisfies the constraints defined by the rewriting algorithm: rewriting happens only to upper levels and between nonempty levels. For example, level 0 can only be a target level. Therefore, invalid actions are removed using Maskable PPO action masking. The policy network only samples from valid source levels, substantially reducing the efective search space.

Two simple heuristics are used to further mask the actions. If a source level contains only one remaining node, all other actions are masked so that the agent is forced to complete the collapse of that level. The second heuristic is activated when a stagnant period is detected where there is no level collapse for a certain number of steps. The source afinity values are used to retain X source levels with the highest afinity as valid actions. X is a tunable parameter.

The RL agent selects the source level with the helps of action masking. Once a source level is selected, an appropriate target level is selected using the algorithm presented in Algorithm 1. Using the tunable parameter SEARCH WINDOW, the thin level with the highest target afinity among the X closest upper thin levels is selected as the target level. If there is no thin level above the source level, as a fallback strategy, the nearest nonempty upper level is chosen as the target level. If the selected source level is thick and there is no upper thin level, the action is considered invalid and the graph state remains unchanged. Furthermore, if the rewrite distance between the source and target levels exceeds a predefined threshold (MAX REWRITE DISTANCE), the action is also considered invalid and the graph state remains unchanged. The target level selected using the fallback strategy is typically a thick level. Moving a node from a thin level to a thick level is considered a legal action since one of the objectives is to collapse thin levels. Therefore, even if the target level is thick, this move is allowed. In contrast, rewriting from a thick source level to a thick target level is not allowed since it neither contributes to collapsing a level nor reduces the coeficient of variation of level costs.

Once the source and the target levels are chosen, the deterministic rewriting algorithm determines the new graph state, updates the graph state data structures without modifying the dependency graph and registers the rewritten row to the rewriting list together with the source and target levels.

## 4.4 Reward Function

The reward function is designed to encourage graph configurations that improve parallel SpTRSV execution while avoiding transformations that degrade graph quality and preserving eficient graph transformations. Positive rewards are assigned when transformations reduce the number of levels(critical path of the graph) or improve level workload balance. Penalties are applied when the level costs becomes more imbalanced or when transformations fail to make useful progress. The reward combines multiple objectives are listed below:

• reduction in the number of levels

• improvement in level cost balance

• critical-path behavior

• movement eficiency

This multi-objective reward allows the agent to evaluate long sequences of graph transformations instead of optimizing a single heuristic criterion. None of the reward components alone is suficient to produce useful graph transformations; therefore they are optimized jointly.

The reward consists of four components using coeficient of variation of level costs, critical path reduction, rewriting distance and abandonment penalty. The reward and penalty components are applied using a constant based on the initial level count and logarithmic functions are used to smooth the magnitudes. Level costs and critical path have the same weight while the abandonment penalty has a smaller weight. The rewriting distance has the smallest weight to avoid disrupting the target selection process. Since target-level selection is primarily determined by the afinity mechanism, rewriting distance is assigned a small weight so that it refines rather than dominates the policy.

The calculation of the total reward and the weights of each reward component is provided below:

total reward $= 0 . 8 0 * r . c v . l e v e l . c o s t + 0 . 8 0 * r . c p a t h + 0 . 0 5 * r . r w d i s t + 0 . 5 0 *$ r abandon

The components of the reward function are explained in the following subsections and the reward calculation algorithm is given in Algorithm 2.

## 4.4.1 Critical path reduction (r cv level cost):

Coeficient of variation of level costs is calculated and a penalty is applied if it is increased (load balance is worsened) and a reward if it is decreased (load balance is improved).

## 4.4.2 Critical path reduction (r cpath):

A reward is given if the current action caused the source level to be collapsed, otherwise a penalty is applied. The rewards or penalties are determined proportionally to the number of nodes of the source level. If level collapse did not happen but the source level is thin, still a small reward is provided to encourage the agent to work on thin levels, if the level is thick, a penalty is applied.

## 4.4.3 Abandon penalty (r abandon):

In order to detect abandoned levels, the last source level seen as well as the time step that it was first touched are recorded and compared to the current source level selected. The penalty for abandoning the previous source level is applied to the current action. The size of the abandoned level is determined, and its completion ratio (how far it is from being empty) is calculated. Based on the time steps that passed since the first touch and the completion ratio, a penalty is calculated. In this way, the agent is encouraged to work on a source level frequently if not consecutively since it is punished for letting a touched level get ”cold” over time. Since the exploration space is large, if not guided, the agent is free to select any level as source level which makes it dificult to learn the policy to collapse the levels in a meaningful way and the agent randomly wanders among the levels. However, to balance exploration and exploitation, the penalty is not applied if there are still levels with only 1 node.

If the abandoned source level was previously a target level, its initial size is set to its current size and first touched time step is set to zero to reset its parameters fitting its new role (being a source level). A previous target level should not be attempted at as a source level since a level that previously served as a target is expected to accumulate nodes rather than lose them. Hence, any attempt to rewrite a node from it will be a wasted efort and penalizing the agent for abandoning such a level would contradict the target-selection strategy. Therefore, the completion ratio is set to almost 1.0 and no abandonment penalty is applied. Likewise, if the abandoned source level is now empty, no penalty is applied.

## 4.4.4 Rewriting distance (r rwdist):

Rewriting cost is a function of the sparsity pattern and the rewriting distance. The policy selects the row with the smallest number of parents to keep the rewriting cost small but this is not a guarantee since the agent does not know beforehand the connectivity between the node and its parents passed throughout the rewriting process, it is revealed as the node is actually being rewritten. Even though rewriting distance reward does not fully capture the entire rewriting cost, rewarding small rewriting distances and punishing large ones helps the agent to adjust the rewriting cost. The selection of a target level with the highest afinity among the closest five upper thin levels or the fallback mechanism choosing the nearest thick level does not invalidate this reward due to its small weight. The rewriting distances can be large based on the number of empty levels residing between the source and the target level. Regardless, the policy will tend to experience larger rewriting distances towards the episode ends since there will be empty levels scattered on the level sequence.

## 4.5 Learning Algorithm

The policy is trained using Maskable Proximal Policy Optimization (Maskable PPO), which extends PPO by supporting invalid action masking.

During training, each episode begins from the original dependency graph. The agent repeatedly selects valid source levels, then the target level is selected using two heuristics. Although graph rewriting is performed at the node level, the agent operates on source levels, which reduces the exploration space and simplifies the action space. The action is finalized by the deterministic graph rewriting algorithm. The reinforcement learning strategy does not modify the graph directly. Instead, it constructs a rewriting list consisting of the rewritten node together with its source and target levels. This is the same approach used with every graph transformation strategy in the Chainbreaker framework. The rewriting list consists of triplets of the node, source level and target level. Training continues until a terminal condition is reached when either of the following two conditions are met: (1) thin level ration falls below 0.02, (2) for suficiently large graphs, the episode reaches a predefined maximum number of steps, in which case the episode is truncated. When the training or the inference is finalized, the rewriting list is returned to the main framework where it is passed to the rewriting module.

A curriculum learning approach is taken where the policy is trained using three relatively small matrices with diferent sparsity patterns. The resulting policy is then used during inference to guide graph transformations on unseen matrices. During inference, multiple independent runs are executed for each matrix, and the graph configuration producing the smallest coeficient variation of level costs and number of levels is selected as the final transformed graph. In addition, transfer learning approach is taken for matrices that are relatively larger than the ones used for training, aiming at model performance.

Algorithm 2 Reward Computation   
Require: Current and previous graph statistics   
Ensure: Total reward   
1: $r _ { c v } \gets 0$   
2: if $\Delta C V < - \epsilon$ then   
3: $r _ { c v }  - \alpha$   
4: else if $\Delta C V > \epsilon$ then   
5: $r _ { c v }  L .$ count init · ∆CV   
6: end if   
7: $r _ { c p } \gets 0$   
8: Compute $s  \log ( 1 +$ source level size + 1)   
9: if critical path is reduced then   
10: $r _ { c p } \gets \alpha$ ∗ (init level count/level count) ∗ $( 1 + 1 / s )$   
11: else if source level is thin then   
12: $r _ { c p } \gets \alpha / s$   
13: else   
14: r ← −1.0 ∗ α ∗ (1 + s)   
15: end if   
16: $r _ { a b a n d o n } \gets 0$   
17: if previous source level is abandoned then   
18: if abandoned source level was a target level then   
19: abandon penalty ← ComputeSourceLevelSizeAndCompletionRatio()   
20: r<sub>abandon</sub>− = abandon penalty   
21: Reset abandoned source level size and completion ratio   
22: if levels with 1 node exist and abandoned level is thin then   
23: r<sub>abandon</sub> $ 0 . 0$   
24: end if   
25: end if   
26: end if   
27: r<sub>rwdist</sub> ← 0   
28: if $r _ { r w d i s t } <$ threshold then   
29: r wdist ← log1p(max(0, threshold − rwdist))/init level count   
30: else   
31: r<sub>r</sub>wdist ← −log1p(max(0, rwdist − threshold))/init level count   
32: end if   
33: return 0.80r<sub>cv</sub> + 0.80r<sub>cp</sub> + 0.05r<sub>dist</sub> + 0.50r<sub>abandon</sub>

## 5 Experiment Results

The experiments are conducted using a subset of the real world matrices [18] from [7] on an 8 core 11th Gen Intel ®Core ™i7-11800H machine with 2.30GHz,16 GB of RAM and an RTX 3050™GPU. The selected matrices exhibit diverse sparsity patterns, matrix sizes and level set characteristics. Table 2 gives the kind of the problem, the number of rows, number of nonzero elements for the lower triangle (L) matrix and the number of levels.

Table 2 Matrices from SuiteSparse Matrix Collection [18] that are used in experiments. The number of nonzeros (# of NNZ) are for the lower triangular part of the matrix (L).
<table><tr><td>Matrix Name</td><td>Kind</td><td># of rows</td><td># of NNZ (L)</td><td># of levels</td></tr><tr><td>bcsstk17</td><td>Structural Prob.</td><td>10,974</td><td>219,812</td><td>1,332</td></tr><tr><td>bcsstk37</td><td>Structural Prob.</td><td>25,503</td><td>58,324</td><td>5,030</td></tr><tr><td>cfd2</td><td>Comp. Fluid Dynamics Prob.</td><td>123,440</td><td>2,605,669</td><td>4,357</td></tr><tr><td>gearbox</td><td>Structural Prob.</td><td>153,746</td><td>4,617,075</td><td>4,586</td></tr><tr><td>lung2</td><td>Comp. Fluid Dynamics Prob.</td><td>109,460</td><td>273,647</td><td>479</td></tr><tr><td>PR02R</td><td>Comp. Fluid Dynamics Prob.</td><td>161,070</td><td>4,174,236</td><td>2,838</td></tr><tr><td>torso2</td><td>2D/3D Problem</td><td>115,967</td><td>574,718</td><td>513</td></tr><tr><td>venkat01</td><td>Comp. Fluid Dynamics Prob. Seq.</td><td>62,424</td><td>890,108</td><td>4,176</td></tr></table>

The experiments are conducted using the ChainBreaker framework, extended with the proposed RLGT and GDT modules <sup>1</sup>. Data exchange between the main framework and and the RLGT module is performed through a pybind11 interface implemented on GDT module. On the RLGT module, train and run python modules are written to perform the experiments.

The implementation is done in C++ and Python. Gcc version 15.2.1 is used with optimization level -O3, and Python version used is 3.14.3. Training and inference both are performed on the CUDA device in a python environment. Stable-baselines3 [19] is used with a Maskable PPO and SubprocVecEnv. The MlpPolicy is trained using a learning rate of 0.0001 for 30 PPO rollout iterations with a rollout length of 1024 timesteps. To limit training time for matrices with a large number of levels, episodes are truncated when the number of elapsed timesteps exceeds a predefined maximum episode length. The maximum episode length is determined from the initial number of levels of the input matrix and is defined as max(2 × num levels, 3 × log checkpoint), where log checkpoint denotes the logging frequency and is set to 2048 timesteps.

The RL agent is trained using SubprocVecEnv with four parallel worker environments. Parallel environment execution increases sample collection throughput by allowing multiple episodes to be processed simultaneously. The total number of training timesteps remains unchanged, ensuring that the parallel implementation afects only training eficiency and not the experimental protocol or learning objective.

The cost introduced by the graph transformation process should stay in acceptable limits. Given that iterative solvers typically take a few hundreds of iterations [20], heuristic-based graph transformation strategies [7] can provide fast transformations when compared a reinforcement learning based graph transformation strategy. Since RL-based strategy is implemented in Python while heuristic strategies are implemented directly in C++, direct execution time comparisons are not meaningful. Accordingly, training cost is treated as an ofline optimization cost, whereas inference time is reported separately. The inference execution times are the average of 5 runs where the rewriting list leading to the best graph configuration out of these 5 runs is chosen to be sent back to the main framework. Therefore, the inference is not deterministic. A logging mechanism is used to log several parameters tracking the environment that are logged at every checkpoint, at the end of full and truncated episodes. Some of the parameters are time step, total reward, average reward, number of levels, CV of level costs, thin level ratio, thin move ratio, fallback ratio, invalid target ratio and the type of logging.

During training, the framework records environment statistics, episode summaries, level costs, level sizes and rewriting lists for every completed episode. These logs are subsequently used to analyze the graph transformations produced by the learned policies.

Since all graph transformation strategies in ChainBreaker - including the proposed RL-guided strategy - derive decisions from structural properties of the input matrix, the learned policy is inherently matrix dependent. Therefore, a generalized trained model may not respond well to the diferent sparsity patterns. To see how well a trained model respond to sparsity patterns with diferent characteristics three categories of experiments are conducted:

1. Individual training: Independent models are trained and evaluated on lung2, torso2, and bcsstk17, which have relatively small initial numbers of levels but exhibit diferent coeficients of variation (CV) of level costs.

2. Transfer learning (Curriculum learning + Fine-tuning): A curriculum model is first trained sequentially on lung2, torso2, and bcsstk17. This pretrained model is then fine-tuned independently on each of bcsstk37, cfd2, gearbox, PR02R, and venkat01, producing a specialized model for each matrix. Then the fine-tuned models are evaluated on their respective matrices.

3. Zero-shot inference: The curriculum model is directly evaluated on on bcsstk37, cfd2, gearbox, PR02R, and venkat01 without any additional training.

The following subsections evaluate these experiment results in terms of PPO learning behavior, graph transformation results and inference behavior.

## 5.1 PPO Learning Behavior

The PPO learning behavior is evaluated in Table 3 using explained variance, approximate KL divergence, and entropy loss. For the individually trained models, the average explained variance ranges from approximately 0 to 0.167, with maximum values of 0.432 and 0.470, respectively. The curriculum model is trained sequentially on lung2, bcsstk17, and torso2; therefore, no separate curriculum-training statistics are reported for lung2. For bcsstk17 and torso2, curriculum learning increases the average explained variance to 0.353 and 0.107, respectively, with maximum values of 0.802 and 0.503. This indicates that knowledge accumulated from previously trained matrices improves value-function learning for subsequent matrices. The approximate KL divergence remains relatively small throughout the experiments, ranging from 0.002 to 0.024, indicating stable policy updates without excessive divergence between consecutive policies. The decrease in entropy loss observed with curriculum learning also indicates that the policy becomes more focused as training progresses while retaining stochasticity for exploration.

Table 3 PPO learning behavior for individually trained models, curriculum model trained using lung2, bcsstk17 and torso2, and fine-tuned models that use the curriculum model.
<table><tr><td colspan="6">Individual Model</td></tr><tr><td></td><td>explained var. (avg.)</td><td>explained var. (max.)</td><td>approx_kl (avg.)</td><td>entropy_loss (min.)</td><td>entropy_loss (avg.)</td></tr><tr><td>lung2</td><td>0.127</td><td>0.432</td><td>0.007</td><td>-3.890</td><td>-3.467</td></tr><tr><td>bcsstk17</td><td>0.167</td><td>0.470</td><td>0.015</td><td>-6.610</td><td>-5.574</td></tr><tr><td>torso2</td><td>-8.169E-05</td><td>1.980E-04</td><td>0.002</td><td>-5.820</td><td>-5.090</td></tr><tr><td colspan="6">Curriculum Model</td></tr><tr><td></td><td>explained var. (avg.)</td><td>explained var. (max.)</td><td>approx_kl (avg.)</td><td>entropy_loss (min.)</td><td>entropy_loss (avg.)</td></tr><tr><td>lung2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>bcsstk17</td><td>0.353</td><td>0.802</td><td>0.014</td><td>-6.750</td><td>-5.643</td></tr><tr><td>torso2</td><td>0.107</td><td>0.503</td><td>0.014</td><td>-5.53</td><td>-4.720</td></tr><tr><td></td><td colspan="6">Transfer Learning (Curriculum learning + Fine Tuning)</td></tr><tr><td></td><td>explained var. (avg.)</td><td>explained var. (max.)</td><td>approx_kl (avg.)</td><td>entropy_loss (min.)</td><td>entropy_loss (avg.)</td></tr><tr><td>bcsstk37</td><td>0.056</td><td>0.380</td><td>0.013</td><td>-7.420</td><td>-6.261</td></tr><tr><td>cfd2</td><td>0.000</td><td>0.000</td><td>0.005</td><td>-8.180</td><td>-8.153</td></tr><tr><td>gearbox</td><td>0.004</td><td>0.025</td><td>0.014</td><td>-7.840</td><td>-7.647</td></tr><tr><td>PR02R</td><td>0.083</td><td>0.189</td><td>0.017</td><td>-7.730</td><td>-7.686</td></tr><tr><td>venkat01</td><td>0.000</td><td>0.000</td><td>0.024</td><td>-8.120</td><td>-8.019</td></tr></table>

In contrast, the average explained variance obtained during transfer learning is substantially lower for the target matrices, ranging from 0.000 to 0.083, despite the stable approximate KL divergence. This behavior indicates that the value function learned from previously encountered matrices does not generalize well to diferent matrix structures. Such behavior is consistent with the nature of graph transformation for sparse triangular systems. Each sparse matrix has a distinct sparsity pattern, which determines the dependency structure and consequently the possible efects of rewriting operations. In addition, rewriting a row may decrease, preserve, or increase its indegree count, and therefore its computational cost, resulting in varying efects on the workload distribution across levels. Consequently, the efect of a graph transformation is highly dependent on the underlying sparsity pattern. These observations support the specialization of the graph transformation policy to the characteristics of the target matrix rather than assuming direct generalization across diferent sparse matrices. The transfer-learning results therefore motivate fine-tuning as a means of adapting the learned policy to the specific sparsity pattern and transformation behavior of each target matrix.

The results also motivate evaluating the learned policy in a zero-shot setting, in which the target matrices are not used for either curriculum training or fine-tuning. Since graph transformation behavior is strongly dependent on the underlying sparsity pattern, strong zero-shot generalization is not expected. Nevertheless, the experimental results presented in Sections 5.2 and 5.3 indicate that zero-shot inference can achieve performance comparable to transfer-learning inference, and in some cases may even approach it. Zero-shot inference experiment is useful for quantifying the extent to which the learned policy captures transformation behavior that is independent of a specific matrix structure.

## 5.2 Graph Transformation Results

Table 4 summarizes the graph characteristics before and after RL-based transformation for lung2, bcsstk17, and torso2. The table reports initial characteristics such as number of levels, ALC, ARL, AIR values as well was CV of level costs before and after the RL transformation. In addition, for RL transformation, information about thin level cost threshold, number of thin levels and some other details are reported. The table also gives percentage of level reduction and train or inference runtime in seconds. The timings for inference are average of 5 indeterministic runs. These matrices are used for both individual model training and the construction of the curriculum model. The initial characteristics show substantial diferences among the matrices. In particular, lung2 has the highest initial coeficient of variation (6.72), whereas bcsstk17 has the largest number of levels (1, 332). Although torso2 has a comparable number of levels to lung2, its initial ALC and ARL are substantially higher making level reduction particularly dificult. These diferences result in considerably diferent transformation outcomes across the matrices, demonstrating the matrix-dependent nature of the graph transformation problem.

The individually trained models reduce the number of levels by 94%, 22%, and17% for lung2, bcsstk17, and torso2, respectively. The corresponding CV values are reduced by a value ranging between 43% and 80% during training. The results obtained during inference are highly consistent with those observed during training improving the level reduction counts except for lung2. The final level counts are 30, 1, 015, and 420. The CV values are comparable for indiivdual training nad inference results.

For the curriculum models, the results on bcsstk17 and torso2 remain comparable to those obtained with individually trained models. Since the model trained using lung2 for the curriculum model, the results are the same, hence they are not reported twice. The level reduction obtained with inference are higher for bcsstk17 and torso2 while remaining the same for lung2. The curriculum model is used for transfer learning for five other matrices.

The results also highlight the diferent computational costs associated with training and inference. Individual model training requires approximately 81s for lung2, 79min. for bcsstk17, and 97min. for torso2, whereas the corresponding average inference times over five runs range between 6.25s, and 319s. Thus, once the policy has been trained, applying the learned transformation strategy is considerably less costly than training the model. The variation in both training and inference time across matrices is consistent with their substantially diferent graph sizes and structural characteristics.

Table 5 uses the same structure as Table 4 and presents the transformation results for the five matrices used for transfer learning and compares transfer-learning inference with zero-shot inference. The matrices exhibit substantially diferent structural characteristics. Their initial number of levels ranges from 2, 838 to 5, 030, while the number of thin levels ranges from 2, 423 to 3, 732. The initial CV values range from 0.37 to 0.88, further demonstrating the variation in workload distribution among the matrices.

Table 4 Graph transformation results for training results for individual training, transfer learning and individual inference results.
<table><tr><td>initial</td><td>lung2</td><td>bcsstk17</td><td>torso2</td></tr><tr><td>num. of levels</td><td>479</td><td>1,332</td><td>513</td></tr><tr><td>ALC</td><td>914</td><td>321</td><td>2014</td></tr><tr><td>ARL</td><td>228.518</td><td>8.23874</td><td>226.057</td></tr><tr><td>AIR</td><td>1.49997</td><td>19.0303</td><td>3.95588</td></tr><tr><td>CV of level cost</td><td>6.72</td><td>0.83</td><td>0.82</td></tr><tr><td>RL transformation</td><td>lung2</td><td>bcsstk17</td><td>torso2</td></tr><tr><td>threshold</td><td>914.06</td><td>432.2</td><td>2370.64</td></tr><tr><td>max node count in thin</td><td>228</td><td>64</td><td>270</td></tr><tr><td>max cost in thin</td><td>912</td><td>430</td><td>2370</td></tr><tr><td>initial thin level count</td><td>462</td><td>920</td><td>312</td></tr><tr><td>RL transformation</td><td>lung2</td><td>bcsstk17</td><td>torso2</td></tr><tr><td>29</td><td colspan="3">individual training</td></tr><tr><td>num. of levels</td><td></td><td>1039</td><td>425</td></tr><tr><td>ALC</td><td>15151.448</td><td>598.951</td><td>2732.379</td></tr><tr><td>ARL</td><td>3774.483</td><td>10.562</td><td>272.864</td></tr><tr><td>AIR</td><td>1.507</td><td>27.854</td><td>4.507</td></tr><tr><td>CV</td><td>1.3298</td><td>0.2922</td><td>0.4742</td></tr><tr><td>RL train time</td><td>81.32</td><td>4727.34</td><td>5792.57</td></tr><tr><td>level reduction ptg.</td><td>94%</td><td>22%</td><td>17%</td></tr><tr><td>RL transformation</td><td>lung2</td><td>bcsstk17 curriculum learning</td><td>torso2</td></tr><tr><td>num. of levels</td><td colspan="3">1040</td></tr><tr><td>ALC</td><td></td><td>594.362</td><td>438 2614.902</td></tr><tr><td>ARL</td><td></td><td>10.552</td><td>264.765</td></tr><tr><td>AIR</td><td></td><td>27.664</td><td>4.438</td></tr><tr><td>CV</td><td></td><td>0.2777</td><td>0.5386</td></tr><tr><td></td><td></td><td>8216.83</td><td>4656.17</td></tr><tr><td>RL train time level reduction ptg.</td><td></td><td>22%</td><td>15%</td></tr><tr><td></td><td></td><td></td><td>torso2</td></tr><tr><td>RL transformation lung2</td><td colspan="3">bcsstk17 individual inference</td></tr><tr><td>num. of levels</td><td colspan="3"></td></tr><tr><td>ALC</td><td>30</td><td>1015</td><td>420</td></tr><tr><td></td><td>14639.067</td><td>595.494</td><td>2771.302</td></tr><tr><td>ARL</td><td>3648.667</td><td>10.812</td><td>276.112</td></tr><tr><td>AIR</td><td>1.506</td><td>27.039</td><td>4.518</td></tr><tr><td>CV</td><td>1.368</td><td>0.283</td><td>0.465</td></tr><tr><td>num. of rewritten rows</td><td>4,093</td><td>3,474</td><td>5,630</td></tr><tr><td>avg. RL run time</td><td>6.252</td><td>319.628</td><td>79.101</td></tr><tr><td>level reduction ptg.</td><td>94%</td><td>24%</td><td>18%</td></tr></table>

Table 5 Graph transformation results for training results using transfer learning and inference results for transfer learning and zero-shot experiments.
<table><tr><td>initial</td><td>cfd2</td><td>gearbox</td><td>venkat01</td><td>PR02R</td><td>bcsstk37</td></tr><tr><td>num. of levels</td><td>4,357</td><td>4,586</td><td>4,176</td><td>2,838</td><td>5,030</td></tr><tr><td>ALC</td><td>708</td><td>1,979</td><td>411</td><td>2,884</td><td>226</td></tr><tr><td>ARL</td><td>28.33</td><td>33.53</td><td>14.95</td><td>56.75</td><td>5.07</td></tr><tr><td>AIR</td><td>12.01</td><td>29.03</td><td>13.26</td><td>24.92</td><td>21.87</td></tr><tr><td>CV of level costs</td><td>0.57</td><td>0.88</td><td>0.46</td><td>0.37</td><td>0.76</td></tr><tr><td>RL transformation</td><td>cfd2</td><td>gearbox</td><td>venkat01</td><td>PR02R</td><td>bcsstk37</td></tr><tr><td>threshold</td><td>1010.55</td><td>2224.62</td><td>635.26</td><td>4710.3</td><td>281.68</td></tr><tr><td>thin levels</td><td>3,485</td><td>2,423</td><td>3,685</td><td>2,838</td><td>3,732</td></tr><tr><td>max. node count in thin</td><td>44</td><td>208</td><td>27</td><td>682</td><td>11</td></tr><tr><td>max. cost in thin</td><td>1,010</td><td>2,221</td><td>635</td><td>4,022</td><td>281</td></tr><tr><td>initial thin level count</td><td>3,485</td><td>2,423</td><td>3,685</td><td>2,838</td><td>3,732</td></tr><tr><td>RL transformation</td><td>cfd2</td><td>gearbox</td><td>venkat01</td><td>PR02R</td><td>bcsstk37</td></tr><tr><td colspan="6">transfer learning (curriculum learning + fine-tuning)</td></tr><tr><td>num. of levels</td><td>4250</td><td>3953</td><td>4037</td><td>2809</td><td>4093</td></tr><tr><td>ALC</td><td>1371.726</td><td>2331.572</td><td>470.816</td><td>2975.421</td><td>394.324</td></tr><tr><td>ARL</td><td>29.045</td><td>38.893</td><td>15.463</td><td>57.341</td><td>6.231</td></tr><tr><td>AIR</td><td>23.114</td><td>29.474</td><td>14.724</td><td>25.445</td><td>31.143</td></tr><tr><td>CV</td><td>0.554</td><td>0.706</td><td>0.467</td><td>0.358</td><td>0.496</td></tr><tr><td>RL train time</td><td>29128.97</td><td>119.10</td><td>27809.72</td><td>59.88</td><td>13299.36</td></tr><tr><td>total train &amp; I/O time</td><td>29135.97</td><td>125.72</td><td>27815.23</td><td>67.84</td><td>13304.71</td></tr><tr><td>level reduction</td><td>2%</td><td>14%</td><td>3%</td><td>1%</td><td>19%</td></tr><tr><td>RL transformation</td><td>cfd2 transfer learning (curriculum learning + fine-tuning)(inference)</td><td>gearbox</td><td>venkat01</td><td>PR02R</td><td>bcsstk37</td></tr><tr><td>num. of levels</td><td colspan="5"></td></tr><tr><td>ALC</td><td>4245 1422.687</td><td>3834 2406.347</td><td>3999 461.447</td><td>2746 3046.577</td><td>3836 40.08</td></tr><tr><td>ARL</td><td>29.079</td><td>40.101</td><td>15.61</td><td>58.656</td><td>29.497</td></tr><tr><td>AIR</td><td>23.963</td><td>29.504</td><td>14.281</td><td>25.47</td><td>14.167</td></tr><tr><td>CV</td><td>0.546</td><td>0.673</td><td>0.462</td><td>0.329</td><td>0.674</td></tr><tr><td></td><td>7763</td><td>9431</td><td></td><td></td><td>4667</td></tr><tr><td>num. of rewritten rows avg. RL run time</td><td></td><td></td><td>7726</td><td>5972</td><td></td></tr><tr><td>level reduction</td><td>1.07E+04</td><td>33.796453</td><td>1.68E+03</td><td>23.949541</td><td>24%</td></tr><tr><td></td><td>3%</td><td>16%</td><td>4%</td><td>3%</td><td></td></tr><tr><td>RL transformation</td><td>cfd2</td><td>gearbox curriculum learning</td><td>venkat01</td><td>PR02R</td><td>bcsstk37 (zero-shot)(inference)</td></tr><tr><td>num. of levels</td><td>4254</td><td>3940</td><td>3994</td><td>2815</td><td></td></tr><tr><td>ALC</td><td>1437.001</td><td>2347.215</td><td>463.965</td><td>2968.474</td><td></td></tr><tr><td>ARL</td><td>29.017</td><td>39.022</td><td>15.629</td><td>57.218</td><td></td></tr><tr><td>AIR</td><td>24.261</td><td>29.576</td><td>14.343</td><td>25.44</td><td></td></tr><tr><td>CV</td><td>0.518</td><td>0.700</td><td>0.4611</td><td>0.362</td><td></td></tr><tr><td>num. of rewritten rows</td><td>5998</td><td>4506</td><td>7786</td><td>5976</td><td></td></tr><tr><td>avg. RL run time</td><td>1.65E+04</td><td>29.407</td><td>436.516</td><td>25.021</td><td></td></tr><tr><td>level reduction</td><td>2%</td><td>14%</td><td>100%</td><td>1%</td><td>100%</td></tr></table>

Transfer learning results in varying level reduction percentages across the target matrices which are increased during inference: the number of levels is reduced by a maximum value of 24% for bcsstk37 and a minimum of 3% for cfd2 and PR02R. The corresponding CV values are reduced by a value ranging between 4% and 24% while it remained the same for venkat01. bcsstk37 demonstrates interesting results with an increase in the level reduction and a decrease in CV value while still being smaller than the original CV value.

The zero-shot results show that the curriculum-trained policy can produce meaningful transformations even without fine-tuning on the target matrix. The results indicate that the learned policy contains transformation behavior that can be applied to previously unseen matrices, although matrix-specific adaptation through fine-tuning provides additional improvement. This observation is consistent with the matrixspecific nature of sparse graph transformations: although the learned policy captures reusable transformation behavior, the efect of individual rewrites depends on the sparsity pattern and the resulting changes in row dependencies and level workloads. Fine-tuning therefore allows the policy to exploit characteristics of the target matrix that cannot be fully captured through zero-shot generalization.

The results should also be interpreted with respect to the predefined inference step limit. Several transformations continue to reduce the level count and/or CV while approaching the available inference budget. Consequently, the reported values represent the transformation quality achieved within the prescribed computational budget and do not necessarily represent the maximum reduction that could be obtained through unrestricted inference.

## 5.3 Inference Behavior

Table 6 presents the results collected from the logs that are obtained during inference for individually trained models, transfer-learning models and zero-shot learning. For each matrix, the best inference run among five independent runs is reported. Some of the information presented in this table such as reduction in the number of levels, CV of level costs, the number of levels and their corresponding initial values are already presented in Tables 4 and Table 5, and they are repeated here for the ease of interpretation. The table presents the resulting thin level ratio, total move count for reported duration (n × 2048 if not truncated/terminated), move ratio where source is thin, average number of levels as the move distance, fallback ratio of total moves and invalid target count where the step is skipped and no rewriting happens. It is observed that the results are obtained primarily through moves originating from thin levels, with high thin-move ratios respectively with the minimum being 0.579 for cfd2, zero-shot inference. Fallback ratios are generalle zero, hence moves from thin to thick levels or from thick to either thin or thick levels do not happen. Although these cases are encountered, the move is rejected due to being from thick to thick level or distance exceeding the threshold. These values are indicated by the invalid target count.

The transfer-learning models also successfully transform the target matrices, although the amount of level reduction varies considerably. bcsstk37 achieves the largest reduction, from 5, 030 to 3, 836 levels (24%), followed by gearbox, which is reduced from 4, 586 to 3, 834 levels (16%). The reductions for cfd2, PR02R, and venkat01 are more limited, at 3%, 3%, and 4%, respectively. The final CV values are reduced relative to the initial values for all transfer-learning cases except venkat01, for which the initial and final values are approximately equal. The learned policies frequently select thin source levels, with thin-move ratios ranging from 0.61 to 0.985. As seen in Table 5, for PR02R, due to thin level threshold being equal to the maximum level cost, the whole matrix is considered thin and maximum node count is 682 and the CV of level costs is 0.37. Given these characteristics, it is challenging to collapse levels, therefore the thin level ratio remains at 0.988 and 0.992 after the transfer learning and zero-shot inferences. A similar case for maximum node count is observed for gearbox with 208 nodes. However, a higher CV value of 0.88 and lower initial thin level count of 2423 compared to PR02R help with the level reduction despite a lower thin move ratio of 0.776.

Table 6 Inference behavior for individually trained models, the curriculum model and zero-shot learning
<table><tr><td colspan="4">Individual Model Inference</td></tr><tr><td></td><td>lung2</td><td>bcsstk17</td><td>torso2</td></tr><tr><td>avg. reward</td><td>1.115</td><td>0.043</td><td>-1.076</td></tr><tr><td>cv level cost</td><td>1.3684</td><td>0.2833</td><td>0.4647</td></tr><tr><td>levels</td><td>30</td><td>1015</td><td>420</td></tr><tr><td>thin level ratio</td><td>0.3</td><td>0.018</td><td>0.319</td></tr><tr><td>move count</td><td>2502</td><td>2619</td><td>6144</td></tr><tr><td>thin move ratio</td><td>0.746</td><td>0.691</td><td>0.726</td></tr><tr><td>avg move dist</td><td>31.354</td><td>19.846</td><td>33.035</td></tr><tr><td>fallback</td><td>0</td><td>0.002</td><td>0</td></tr><tr><td>invalid target count</td><td>0</td><td>1174</td><td>0</td></tr><tr><td>init. cv level cost</td><td>6.72</td><td>0.83</td><td>0.82</td></tr><tr><td>init. num. levels</td><td>479</td><td>1,332</td><td>513</td></tr><tr><td>reduction in num. levels</td><td>94%</td><td>24%</td><td>18%</td></tr></table>

<table><tr><td rowspan="2"></td><td colspan="5">Transfer Learning Inference</td></tr><tr><td>bcsstk37</td><td>cfd2</td><td>gearbox</td><td>PR02R</td><td>venkat01</td></tr><tr><td>avg. reward</td><td>0.821</td><td>-4.885</td><td>0.798</td><td>0.427</td><td>-0.079</td></tr><tr><td>cv level cost</td><td>0.6737</td><td>0.5458</td><td>0.673</td><td>0.3286</td><td>0.4617</td></tr><tr><td>levels</td><td>3836</td><td>4245</td><td>3834</td><td>2746</td><td>3999</td></tr><tr><td>thin level ratio</td><td>0.41</td><td>0.216</td><td>0.406</td><td>0.988</td><td>0.723</td></tr><tr><td>move count</td><td>5004</td><td>7912</td><td>5053</td><td>6144</td><td>8352</td></tr><tr><td>thin move ratio</td><td>0.779</td><td>0.61</td><td>0.776</td><td>0.985</td><td>0.909</td></tr><tr><td>avg move dist</td><td>14.167</td><td>52.949</td><td>14.474</td><td>2.771</td><td>10.257</td></tr><tr><td>fallback</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>invalid target count</td><td>4168</td><td>802</td><td>4119</td><td>0</td><td>0</td></tr><tr><td>init. cv level cost</td><td>0.76</td><td>0.57</td><td>0.88</td><td>0.37</td><td>0.46</td></tr><tr><td>init. num. levels</td><td>5,030</td><td>4,357</td><td>4,586</td><td>2,838</td><td>4,176</td></tr><tr><td>reduction in num. levels</td><td>24%</td><td>3%</td><td>16%</td><td>3%</td><td>4%</td></tr></table>

<table><tr><td colspan="6">Zero-shot Inference</td></tr><tr><td></td><td>bcsstk37</td><td>cfd2</td><td>gearbox</td><td>PR02R</td><td>venkat01</td></tr><tr><td>avg-reward</td><td></td><td>-5.26</td><td>0.394</td><td>0.419</td><td>-0.12</td></tr><tr><td>cv_level_cost</td><td></td><td>0.5181</td><td>0.6998</td><td>0.3624</td><td>0.4611</td></tr><tr><td>levels</td><td></td><td>4254</td><td>3940</td><td>2815</td><td>3994</td></tr><tr><td>thin_level_ratio</td><td></td><td>0.194</td><td>0.409</td><td>0.992</td><td>0.71</td></tr><tr><td>move_count</td><td></td><td>7792</td><td>4774</td><td>6144</td><td>8352</td></tr><tr><td>thin_move_ratio</td><td></td><td>0.579</td><td>0.76</td><td>0.995</td><td>0.906</td></tr><tr><td>avg_move_dist</td><td></td><td>56.576</td><td>15.591</td><td>2.727</td><td>10.91</td></tr><tr><td>fallback</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>invalid_target count</td><td></td><td>922</td><td>4398</td><td>0</td><td>0</td></tr><tr><td>init cv level cost</td><td>0.76</td><td>0.57</td><td>0.88</td><td>0.37</td><td>0.46</td></tr><tr><td>init num levels</td><td>5,030</td><td>4,357</td><td>4,586</td><td>2,838</td><td>4,176</td></tr><tr><td>reduction in num levels</td><td></td><td>2%</td><td>14%</td><td>1%</td><td>4%</td></tr></table>

The transformation behavior also varies according to the distribution of level costs and the resulting changes in AIR. For cfd2, AIR increases considerably during transformation, while thin levels can quickly become thick as their costs increase, making it more dificult for the agent to continue focusing on thin levels and limiting further level collapse. In contrast, gearbox exhibits a clearer separation between thin and thick levels in terms of level costs allowing the agent to maintain its focus on thin levels, while AIR does not spike enough to become an obstacle to level reduction. A similar favorable separation between thin and thick levels is observed for bcsstk37. The agent simply shufled the nodes between levels due to AIR not increasing enough to saturate thin levels to become thick and the number of nodes in a level not decreasing fast enough to trigger level collapses. Another possibility is source levels becoming target levels frequently. The behavior of PR02R follows a pattern similar to that observed for venkat01. The level costs before and after the transformation for transfer learning inference are provided as graphs given in Figure A2 and Figure A1 in Appendix A. When explained variance values from Table 3 are considered, it is observed that the relationship between explained variance and transformation quality is not strictly one-to-one, still bcsstk37 has the highest level reduction and the highest explained variance while gearbox and PR02R follow next.

Several transfer-learning inference runs reach the predefined inference step limit, indicating that the policies continue to perform transformations until the available inference budget is exhausted. Therefore, the reported reductions represent the improvements obtained within the prescribed inference budget rather than necessarily the maximum reductions attainable by continued inference.Zero-shot inference results are similar to transfer learning inference results, zero-shot results for bcsstk37 are missing due to a bug. It is observed that the average rewards are usually around 0 implying that more aggressive level collapsing approaches might improve the results.

Table 7 compares the level reduction, coeficient of variation (CV) of level costs, and percentage of rewritten rows obtained by RL-based graph transformation(RLTrans) and the heuristic-based strategies for the best inference results obtained. In terms of level reduction, the heuristic strategies generally achieve larger reductions than RLTrans strategy. In particular, 2CRI provides the highest level reduction for most matrices, reaching 53% for bcsstk37. The RL-based transformation achieves its highest level reduction for lung2, reducing the number of levels by 94%, which exceeds the

Table 7 Comparison of RL-based graph transformation strategy (named RLTrans) with heuristic-based graph transformation strategies from [7], namely threeCriteria, 3CRI THICKENED, 3CRI AGGRESSIVE and 2CRI.
<table><tr><td colspan="7">Reduction in num. of levels</td></tr><tr><td>matrix</td><td>num. of levels</td><td>three Criteria</td><td>3CRI THICKENED</td><td>3CRI_ AGGRESSIVE</td><td>2CRI</td><td>RLTrans</td></tr><tr><td>bcsstk17</td><td>1332</td><td>16%</td><td>27%</td><td>33%</td><td>48%</td><td>24%</td></tr><tr><td>bcsstk37</td><td>5030</td><td>14%</td><td>30%</td><td>46%</td><td>53%</td><td>24%</td></tr><tr><td>cfd2</td><td>4357</td><td>10%</td><td>17%</td><td>19%</td><td>33%</td><td>3%</td></tr><tr><td>gearbox</td><td>4586</td><td>21%</td><td>37%</td><td>38%</td><td>45%</td><td>16%</td></tr><tr><td>lung2</td><td>479</td><td>46%</td><td>90%</td><td>90%</td><td>90%</td><td>94%</td></tr><tr><td>PR02R</td><td>2838</td><td>9%</td><td>10%</td><td>10%</td><td>18%</td><td>3%</td></tr><tr><td>torso2</td><td>513</td><td>15%</td><td>21%</td><td>19%</td><td>42%</td><td>18%</td></tr><tr><td>venkat01</td><td>4176</td><td>10%</td><td>19%</td><td>27%</td><td>35%</td><td>4%</td></tr><tr><td>avg. level reduction</td><td>■</td><td>18%</td><td>31%</td><td>35%</td><td>46%</td><td>23%</td></tr></table>

<table><tr><td colspan="6">Percentage of rewritten rows</td><td rowspan="2">RLTrans</td></tr><tr><td>matrix</td><td>num. of rows</td><td>three Criteria</td><td>3CRI_ THICKENED</td><td>3CRI_ AGGRESSIVE</td><td>2CRI</td></tr><tr><td>bcsstk17</td><td></td><td>0.36%</td><td></td><td></td><td>0.53%</td><td>1.58%</td></tr><tr><td>bcsstk37</td><td>219,812 583,240</td><td>0.45%</td><td>0.63% 0.84%</td><td>0.86% 1.35%</td><td>0.89%</td><td>0.80%</td></tr><tr><td>cfd2</td><td>1,605,669</td><td>0.41%</td><td>0.75%</td><td>0.80%</td><td>1.15%</td><td>0.48%</td></tr><tr><td>gearbox</td><td>4,617,075</td><td>0.11%</td><td>0.19%</td><td>0.21%</td><td>0.15%</td><td>0.20%</td></tr><tr><td>lung2</td><td>273,647</td><td>0.17%</td><td>0.49%</td><td>0.49%</td><td>0.49%</td><td>1.50%</td></tr><tr><td>PR02R</td><td>4,174,236</td><td>0.18%</td><td>0.21%</td><td>0.22%</td><td>0.38%</td><td>0.14%</td></tr><tr><td>torso2</td><td>574,718</td><td>0.47%</td><td>1.17%</td><td>1.17%</td><td>1.41%</td><td>0.98%</td></tr><tr><td>venkat01</td><td>890,108</td><td>0.49%</td><td>0.90%</td><td>1.36%</td><td>1.14%</td><td>0.87%</td></tr><tr><td>avg. rewritten rows</td><td></td><td>0.33%</td><td>0.65%</td><td>0.81%</td><td>0.77%</td><td>0.82%</td></tr></table>

<table><tr><td colspan="6">Coefficient of variation (CV) of level costs</td></tr><tr><td>matrix</td><td>no trans.</td><td>three Criteria</td><td>3CRI THICKENED</td><td>3CRI_ AGGRESSIVE</td><td>2CRI</td><td>RLTrans</td></tr><tr><td>bcsstk17</td><td>0.66</td><td>0.49</td><td>0.49</td><td>0.63</td><td>0.54</td><td>0.28</td></tr><tr><td>bcsstk37</td><td>0.76</td><td>0.64</td><td>0.60</td><td>0.81</td><td>0.75</td><td>0.67</td></tr><tr><td>cfd2</td><td>0.57</td><td>0.46</td><td>0.54</td><td>0.53</td><td>0.43</td><td>0.55</td></tr><tr><td>gearbox</td><td>0.88</td><td>0.68</td><td>0.67</td><td>0.57</td><td>0.75</td><td>0.67</td></tr><tr><td>lung2</td><td>6.72</td><td>4.89</td><td>1.86</td><td>1.86</td><td>1.86</td><td>1.37</td></tr><tr><td>PR02R</td><td>0.37</td><td>0.28</td><td>0.33</td><td>0.33</td><td>0.32</td><td>0.33</td></tr><tr><td>torso2</td><td>0.82</td><td>0.65</td><td>0.65</td><td>0.68</td><td>0.61</td><td>0.46</td></tr><tr><td>venkat01</td><td>0.46</td><td>0.39</td><td>0.46</td><td>0.41</td><td>0.42</td><td>0.46</td></tr><tr><td>avg. CV of level cost</td><td>1.41</td><td>1.06</td><td>0.70</td><td>0.73</td><td>0.71</td><td>0.60</td></tr></table>

90% reduction obtained by the heuristic strategies for this matrix. For the remaining matrices, RLTrans achieves reductions ranging from 3% to 24%.

The CV results, however, provide a diferent perspective. RLTrans strategy achieves the lowest CV among all evaluated methods for bcsstk17 (0.28) and torso2 (0.46), while also achieving competitive CV values for gearbox and PR02R. In particular, the diference between level reduction and workload balance is evident for bcsstk17: 2CRI reduces the number of levels by 48%, compared with 24% for RLTrans, but results in a CV of 0.54 compared with 0.28 for RLTrans. This illustrates the trade-of between aggressively collapsing levels and maintaining a balanced distribution of level costs. The RL-based approach does not consistently outperform the heuristic strategies in either objective individually; rather, its learned transformations provide a diferent balance between level reduction and workload distribution. For lung2, for example, RL simultaneously achieves the highest level reduction (94%) and a substantial CV reduction from 6.72 to 1.37.

The level reduction and CV results should also be considered together with the amount of graph rewriting required to obtain these transformations. RLTrans rewrites between 0.14% and 1.58% of the rows across the evaluated matrices. The results demonstrate that meaningful changes in both level structure and workload distribution can be obtained by modifying only a small fraction of the rows. For bcsstk17, RL rewrites more rows and achieves a lower level reduction than the heuristic approaches, but obtains a substantially better CV. For lung2, RLTrans rewrites more rows while achieving both a higher level reduction and a better CV than the heuristic approaches. For torso2, RLTrans rewrites fewer rows while achieving a level reduction comparable to the heuristic approaches and a lower CV. Finally, for bcsstk37, the level reduction, CV, and percentage of rewritten rows obtained by RLTrans are comparable to those of the heuristic approaches.

Among the graph transformation strategies presented in the table, RLTrans involves the largest fraction of rewritten rows. However, it should be noted that the percentage of rewritten rows does not directly represent the number of rewriting operations for the RL-based approach. Unlike the heuristic strategies, where a row is rewritten from its source level to a target level only once, the RLTrans agent transforms the graph sequentially and may select a previously rewritten row again as the graph state evolves. Consequently, the reported percentage represents the fraction of rows involved in the learned transformation rather than a direct count of rewrite operations.

Overall, the comparison demonstrates that the RL-based approach and the heuristic strategies exhibit diferent optimization characteristics. The heuristic strategies, particularly 2CRI, are generally more efective at aggressively reducing the number of levels, whereas RLTrans can provide stronger improvements in the balance of level costs for some matrices. This diference is consistent with the multi-objective nature of the transformation problem: collapsing levels can increase the workload of the remaining levels, while improving the balance of level costs may limit the achievable level reduction. The RLTrans agent addresses this trade-of through sequential, state-dependent transformations rather than a fixed heuristic transformation sequence.

## 6 Conclusion and Future work

This work presents a reinforcement learning-guided graph transformation framework for sparse triangular solves within the ChainBreaker framework. The proposed approach formulates graph transformation as a reinforcement learning problem while using a deterministic graph rewriting engine to apply graph transformations. Unlike heuristic-based approaches, which require manually designed transformation rules, the proposed framework enables the automated discovery of graph transformation policies through interaction with the graph transformation environment by providing a flexible experimental platform for developing and evaluating future graph transformation strategies. It enables the systematic investigation of graph representations, reward formulations, and transformation policies without modifying the underlying deterministic graph rewriting engine.

The developed environment models graph transformation using matrix-derived features that capture workload balance and parallelism, allowing the reinforcement learning agent to optimize graph transformations for improved SpTRSV performance.

Experimental results on real-world sparse matrices demonstrate that the RL-based approach can learn efective matrix-dependent transformation policies. Across the evaluated matrices, RLTrans achieves an average level reduction of 23% and an average reduction of 29% in the coeficient of variation of level costs, with individual level reductions reaching up to 94% and CV reductions reaching up to 80%. The comparison with heuristic strategies further demonstrates the trade-of between level reduction and workload balance. The heuristic strategies generally achieve more aggressive level reduction, whereas RLTrans achieves a larger average reduction in CV of level costs of 29%. This behavior demonstrates that the learned policy can navigate competing graph transformation objectives.

The amount of graph modification required by the RLTrans approach is also relatively small. Across the evaluated matrices, between 0.14% and 1.58% of the rows are involved in the learned transformations, with an average of only 0.82%. Despite modifying such a small fraction of the rows, RLTrans produces substantial changes in both the level structure and the distribution of level workloads. These results demonstrate the potential of reinforcement learning as an automated and matrix-dependent approach to graph transformation.

The transfer learning experiments further demonstrate that graph transformation policies can benefit from previously acquired knowledge. Curriculum learning followed by fine-tuning provides useful transfer to previously unseen matrices, although the quality of the transferred policy varies with the characteristics of the target matrix. In contrast, zero-shot inference generally provides weaker transformation results, which is consistent with the strong dependence of graph transformation behavior on the underlying sparsity pattern. Nevertheless, the zero-shot experiments provide evidence that some learned transformation behavior can be applied to previously unseen matrices without additional training and provide a useful basis for studying generalization.

Immediate future work will focus on collecting SPTRSV performance results on solvers from [21] and [7] as well as an SpTRSV library implementation, to evaluate the applicability of the proposed graph transformations across diferent solver implementations. Training will also be extended to a broader collection of sparse matrices. These steps will prepare the framework for further research directions including incorporating hardware-aware transformation objectives, such as memory access locality and architecture-specific optimization criteria, as well as developing policies specialized for diferent sparse triangular solve implementations.

## 7 Related Work

Several parallelization strategies for sparse triangular solve (SpTRSV) have been proposed in the literature. Existing approaches can be broadly classified into level-set methods [9–13] and synchronization-free methods [22–26]. Both categories exploit the directed acyclic graph (DAG) induced by the sparse triangular matrix to expose parallelism while respecting data dependencies among rows.

The level-set method partitions the DAG into a sequence of levels such that all rows assigned to the same level are mutually independent and can therefore be processed concurrently. Rows in a level depend only on rows belonging to preceding levels, requiring each level to complete before the next one begins. Consequently, synchronization barriers are introduced between consecutive levels.

The efectiveness of the level-set method is strongly influenced by the sparsity pattern of the input matrix, which determines both the number of levels and the computational workload assigned to each level. Matrices containing many sparsely populated levels or exhibiting substantial workload imbalance across levels often experience poor parallel eficiency. Thin levels underutilize the available processing resources, while significant diferences in computational cost between levels lead to load imbalance. Moreover, each level transition introduces a synchronization barrier, increasing waiting time as threads must remain idle until all computations in the current level have completed. As a result, the overall performance of level-set methods is highly sensitive to both the critical path length and the distribution of workload across levels. Early implementations of the level-set method primarily targeted multicore CPU architectures [9, 12, 13]. With the growing adoption of many-core accelerators, the same execution model has subsequently been adapted for GPUs [10, 11].

Synchronization-free methods emphasize load balancing by partitioning the rows into groups that are assigned to processing units. Unlike the level-set method, rows within the same group may have dependencies among themselves, providing greater flexibility when distributing the workload. Computation begins as soon as the dependencies of a group are satisfied, allowing synchronization to occur at a finer granularity. Consequently, synchronization-free methods rely on lightweight synchronization mechanisms, such as locks, rather than global barriers. The size of the row groups can be adapted to the characteristics of the target architecture. However, maintaining synchronization metadata for all groups and the use of spinning locks may introduce considerable overhead for large sparse matrices. Successful implementations of synchronization-free methods have been reported for both multicore CPUs [23] and GPUs [22, 24–27].

In general, level-set methods are particularly well suited to multicore CPU architectures, where a moderate number of threads execute rows within a level concurrently before synchronizing at barrier points. Synchronization-free methods, on the other hand, have shown particular promise on GPU architectures, whose massive thread parallelism and eficient hardware scheduling make fine-grained synchronization more practical.

To combine the advantages of both execution models, hybrid approaches integrating level-set and synchronization-free techniques have also been proposed [21, 28]. These approaches seek to reduce synchronization overhead while preserving suficient parallelism by eliminating unnecessary dependencies in the dependency graph.

Several other optimization techniques have also been proposed for SpTRSV. Graph-coloring methods expose parallelism by partitioning the dependency graph into independent color sets and have been employed as a load-balancing strategy on both multicore CPUs [29, 30] and GPUs [31, 32].

Blocking techniques constitute another widely adopted optimization strategy. These methods partition the sparse triangular matrix into diagonal triangular blocks and of-diagonal rectangular blocks, creating denser computational regions that improve cache locality and data reuse. Various partitioning strategies have been investigated in the literature. For example, Lu et al. [33] proposed column-block, row-block, and iterative-block algorithms for GPUs and selected the most suitable blocking strategy according to the sparsity characteristics of the input matrix. Since blocking typically requires matrix reordering, these approaches incur additional preprocessing overhead, and their efectiveness largely depends on the sparsity structure of the input matrix. Several blocking-based optimizations have been proposed for both CPU and GPU architectures [1, 3, 4, 27, 33–35].

Compiler-assisted optimization represents another important research direction for sparse matrix computations. By exploiting domain-specific knowledge, compilers can perform context-aware transformations that are dificult to achieve with general-purpose optimization passes. Examples include domain-specific code generation frameworks such as Sparso [36] and Sympiler [37], which apply optimizations including dense-block extraction, loop transformations, elimination of indirect memory accesses for sparse computations, arithmetic simplifications, and memory-access optimizations [37]. AG-SpTRSV [38] introduced a GPU framework that generates multiple code variants using a performance model and performs workload balancing by assigning groups of rows to GPU warps according to matrix characteristics such as block size and average row density. More recently, Patel et al. [39] proposed a compiler capable of generating symmetry-aware code for sparse and structured tensor computations.

## Declarations

This work was supported by T¨urkiye Bilimsel ve Teknolojik Ara¸stırma Kurumu (TUB<sup>¨</sup> <sup>˙</sup>ITAK) with Grant/AwardNumber: 121E612.

The authors declare that they have no competing interests or conflicts of interest related to this work.

## Availability of Data and Materials:

The code developed for this work is available in the github repository mentioned on page 15 footnote. In addition, all data generated or analysed during this study are included in this published article.

![](images/cce4d820ae6be9681ae1c0b2072080df6a2f7f68ca6dd38430cd989859a04108.jpg)  
Fig. A1 Level costs before and after RL graph transformation.

Data availability: The datasets generated and analyzed during the current study are available from the corresponding author on reasonable request since it is not put in a publicly avaiable repository yet.

## Appendix A Level Cost Change by RL Transformation

## References

[1] C¸ u˘gu, Manguo˘glu, M.: A parallel multithreaded sparse triangular linear system solver. Computers & Mathematics with Applications 80(2), 371–385 (2020) https://doi.org/10.1016/j.camwa.2019.09.012 . Numerical Methods for Scientific Computations and Advanced Applications II

[2] Bradley, A.M.: A Hybrid Multithreaded Direct Sparse Triangular Solver, pp. 13– 22. https://doi.org/10.1137/1.9781611974690.ch2 . https://epubs.siam.org/doi/ abs/10.1137/1.9781611974690.ch2

[3] Mayer, J.: Parallel algorithms for solving linear systems with sparse triangular matrices. Computing 86(4), 291 (2009) https://doi.org/10.1007/ s00607-009-0066-3

![](images/72057fa04b51696af57b859c56857e3bcbde1825b5bc7261f482af1eace62313.jpg)

![](images/8b15f02ab2166c2ec2c58649737970ad4f98c4d109034747d1fa8b24ee6678e2.jpg)

![](images/b0b0666217ad87302ef0ce18f611c3ba48141a41ae6bf477e362c35e8f552996.jpg)  
Fig. A2 Level costs before and after RL graph transformation.

[4] Totoni, E., Heath, M.T., Kale, L.V.: Structure-adaptive parallel solution of sparse triangular linear systems. Parallel Computing 40(9), 454–470 (2014) https://doi. org/10.1016/j.parco.2014.06.006

[5] Ahmad, N., Yilmaz, B., Unat, D.: A split execution model for sptrsv. IEEE Transactions on Parallel & Distributed Systems 32(11), 2809–2822 (2021) https: //doi.org/10.1109/TPDS.2021.3074501

[6] Yılmaz, B.: Graph transformation and specialized code generation for sparse triangular solve (sptrsv). BAS¸ARIM 2020 (2021)

[7] Yılmaz, B.: Eficient Graph Transformation Strategies for Optimizing Sparse Triangular Solve. Dokuz Eyl¨ul Universitesi M¨uhendislik Fak¨ultesi Fen ve<sup>¨</sup> M¨uhendislik Dergisi. In press (2026). https://doi.org/InPress . .

[8] Yılmaz, B.: A novel graph transformation strategy for optimizing sptrsv on cpus. Concurrency and Computation: Practice and Experience 35(24), 7761 (2023) https://doi.org/10.1002/cpe.7761

[9] Anderson, E., Saad, Y.: Solving sparse triangular linear systems on parallel computers. International Journal of High Speed Computing 1(1), 73–95 (1989) https://doi.org/10.1142/S0129053389000056

[10] Li, R., Saad, Y.: GPU-accelerated preconditioned iterative linear solvers. The Journal of Supercomputing 63(2), 443–466 (2013) https://doi.org/10.1007/ s11227-012-0825-3

[11] Naumov, M.: Parallel solution of sparse triangular linear systems in the preconditioned iterative methods on the gpu. Technical report (2011)

[12] Rothberg, E., Gupta, A.: Parallel iccg on a hierarchical memory multiprocessor — addressing the triangular solve bottleneck. Parallel Computing 18(7), 719–741 (1992) https://doi.org/10.1016/0167-8191(92)90041-5

[13] Saltz, J.H.: Aggregation methods for solving sparse triangular systems on multiprocessors. SIAM J. Sci. Stat. Comput. 11(1), 123–144 (1990) https://doi.org/ 10.1137/0911008

[14] Sutton, R.S., Barto, A.G.: Reinforcement Learning: An Introduction, 2nd edn. MIT Press, ??? (2018)

[15] Kaelbling, L.P., Littman, M.L., Moore, A.W.: Reinforcement learning: A survey. Journal of Artificial Intelligence Research 4, 237–285 (1996)

[16] Murphy, K.P.: Reinforcement learning: An overview. arXiv preprint arXiv:2412.05265 (2024)

[17] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)

[18] Davis, T.A., Hu, Y.: The university of florida sparse matrix collection. ACM Trans. Math. Softw. 38(1), 1–1125 (2011) https://doi.org/10.1145/2049662. 2049663

[19] Rafin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N.: Stable-baselines3: Reliable reinforcement learning implementations. Journal of machine learning research 22(268), 1–8 (2021)

[20] Yilmaz, B., Aktemur, B., Garzar´an, M.J., Kamin, S., Kira¸c, F.: Autotuning runtime specialization for sparse matrix-vector multiplication. ACM Trans. Archit. Code Optim. 13(1) (2016) https://doi.org/10.1145/2851500

[21] Yılmaz, B., Sipahio˘glu, B., Ahmad, N., Unat, D.: Adaptive level binning:

A new algorithm for solving sparse triangular systems. In: Proceedings of the International Conference on High Performance Computing in Asia-Pacific Region. HPCAsia2020, pp. 188–198. Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3368474.3368486 . https://doi.org/10.1145/3368474.3368486

[22] Aliaga, J.I., Dufrechou, E., Ezzatti, P., Quintana-Ort´ı, E.S.: Accelerating the task/data-parallel version of ilupack’s bicg in multi-cpu/gpu configurations. Parallel Computing 85, 79–87 (2019) https://doi.org/10.1016/j.parco.2019.02. 005

[23] Hammond, S.W., Schreiber, R.: Eficient iccg on a shared memory multiprocessor. International Journal of High Speed Computing 04(01), 1–21 (1992) https://doi. org/10.1142/S0129053392000183 https://doi.org/10.1142/S0129053392000183

[24] Li, R.: ON PARALLEL SOLUTION OF SPARSE TRIANGULAR LINEAR SYS-TEMS IN CUDA. Technical report (2017). https://arxiv.org/pdf/1710.04985. pdf

[25] Liu, W., Li, A., Hogg, J., Duf, I.S., Vinter, B.: A synchronization-free algorithm for parallel sparse triangular solves. In: Dutot, P.-F., Trystram, D. (eds.) Euro-Par 2016: Parallel Processing, pp. 617–630. Springer, Cham (2016)

[26] Liu, W., Li, A., Hogg, J.D., Duf, I.S., Vinter, B.: Fast synchronizationfree algorithms for parallel sparse triangular solves with multiple right-hand sides. Concurrency and Computation: Practice and Experience 29(21), 4244 (2017) https://doi.org/10.1002/cpe.4244 https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpe.4244. e4244 cpe.4244

[27] Su, J., Zhang, F., Liu, W., He, B., Wu, R., Du, X., Wang, R.: Capellinisptrsv: A thread-level synchronization-free sparse triangular solve on gpus. In: Proceedings of the 49th International Conference on Parallel Processing. ICPP ’20. Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/ 3404397.3404400 . https://doi.org/10.1145/3404397.3404400

[28] Park, J., Smelyanskiy, M., Sundaram, N., Dubey, P.: Sparsifying synchronization for high-performance shared-memory sparse triangular solver. ISC 2014, pp. 124–140. Springer, Berlin, Heidelberg (2014). https://doi.org/10.1007/ 978-3-319-07518-1 8

[29] Iwashita, T., Nakashima, H., Takahashi, Y.: Algebraic block multi-color ordering method for parallel multi-threaded sparse triangular solver in iccg method. In: 2012 IEEE 26th International Parallel and Distributed Processing Symposium, pp. 474–483 (2012). https://doi.org/10.1109/IPDPS.2012.51

[30] Ma, S., Saad, Y.: Distributed ilu(0) and sor preconditioners for unstructured sparse linear systems. (1998)

[31] Naumov, M., Castonguay, P., Cohen, J.: Parallel graph coloring with applications to the incomplete-lu factorization on the gpu. Technical report (2015)

[32] Suchoski, B., Severn, C., Shantharam, M., Raghavan, P.: Adapting sparse triangular solution to gpus. In: 2012 41st International Conference on Parallel Processing Workshops, pp. 140–148 (2012). https://doi.org/10.1109/ICPPW.2012.23

[33] Lu, Z., Niu, Y., Liu, W.: Eficient block algorithms for parallel sparse triangular solve. In: Proceedings of the 49th International Conference on Parallel Processing. ICPP ’20. Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3404397.3404413 . https://doi.org/10.1145/3404397.3404413

[34] Smith, B., Zhang, H.: Sparse triangular solves for ilu revisited: Data layout crucial to better performance. Int. J. High Perform. Comput. Appl. 25(4), 386–391 (2011) https://doi.org/10.1177/1094342010389857

[35] Zhang, F., Su, J., Liu, W., He, B., Wu, R., Du, X., Wang, R.: Yuenyeungsptrsv: a thread-level and warp-level fusion synchronization-free sparse triangular solve. IEEE Transactions on Parallel and Distributed Systems 32(9), 2321–2337 (2021)

[36] Rong, H., Park, J., Xiang, L., Anderson, T.A., Smelyanskiy, M.: Sparso: Contextdriven optimizations of sparse linear algebra. In: 2016 International Conference on Parallel Architecture and Compilation Techniques (PACT), pp. 247–259 (2016). https://doi.org/10.1145/2967938.2967943

[37] Cheshmi, K., Kamil, S., Strout, M.M., Dehnavi, M.M.: Sympiler: Transforming sparse matrix codes by decoupling symbolic analysis. In: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. SC ’17, pp. 13–11313. ACM, New York, NY, USA (2017). https://doi. org/10.1145/3126908.3126936 . http://doi.acm.org/10.1145/3126908.3126936

[38] Hu, Z., Sun, J., Li, Z., Sun, G.: Ag-sptrsv: An automatic framework to optimize sparse triangular solve on gpus 21(4) (2024) https://doi.org/10.1145/3674911

[39] Patel, R., Ahrens, W., Amarasinghe, S.: SySTeC: A Symmetric Sparse Tensor Compiler (2025). https://arxiv.org/abs/2406.09266