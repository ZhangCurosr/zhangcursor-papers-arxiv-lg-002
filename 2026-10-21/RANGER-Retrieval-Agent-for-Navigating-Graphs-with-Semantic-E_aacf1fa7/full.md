# RANGER: Retrieval Agent for Navigating Graphs with Semantic Evaluation Rewards

ARTEM MIKHAYLOV<sup>1</sup>, PAVEL KONOPLEV<sup>1,</sup> <sup>2</sup>, NIKOLAY EFANOV<sup>1,</sup> <sup>2</sup>, ARTYOM SOROKIN<sup>3,</sup> <sup>1</sup>, and EVGENY BURNAEV<sup>1,</sup> <sup>3</sup>

<sup>1</sup>Skolkovo Institute of Science and Technology (Skoltech), Moscow, Russia

<sup>2</sup>Moscow Institute of Physics and Technology (MIPT), Moscow, Russia

<sup>3</sup>AIRI, Moscow, Russia

Corresponding author: Artem Mikhaylov (e-mail: a.mikhaylov@skoltech.ru).

The work was supported by the grant for research centers in the field of AI provided by the Ministry of Economic Development of the Russian Federation in accordance with the agreement 000000C313925P4F0002 and the agreement with Skoltech №139-10-2025-033

ABSTRACT The integration of Large Language Models and Reinforcement Learning offers a promising direction for complex sequential decision-making tasks, such as multi-hop question answering over knowledge graphs. While LLMs provide strong semantic understanding and reasoning capabilities, they often struggle with precise, multi-step navigation in structured environments. Conversely, RL agents excel at learning navigation policies but are hindered by the difficulty of designing semantically meaningful reward functions. This paper introduces a novel hybrid approach in which an LLM serves as a mentor to an RL agent during training by providing dense, semantically informed reward signals. The RL agent, built on a Transformer-based policy and trained with Group Relative Policy Optimization, learns to navigate a knowledge graph to retrieve relevant information. The LLM evaluates the semantic relevance of the agent’s traversed paths, generating reward signals that align the agent’s behavior with the information needs expressed in a query. We integrate this Retrieval Agent for Navigating Graphs with Semantic Evaluation Rewards (RANGER) into a complete question-answering pipeline and evaluate it on the open question answering datasets. Experimental results show that the proposed approach outperforms both a baseline retrieval-augmented system with classical graph search and one using an RL agent trained with heuristic rewards, achieving a significant increase in NLP metrics and LLM-as-a-Judge scores. Our findings demonstrate that deep integration of LLMs into the RL training loop yields more effective and semantically aligned retrieval policies for complex reasoning tasks.

INDEX TERMS Graph traversal, knowledge graphs, LLM-as-a-Mentor, LLM-based reward, multi-hop question answering, reinforcement learning, self-supervision, semantic feedback, transformer policy.

## I. INTRODUCTION

Recent progress in Large Language Models (LLMs) has precipitated a substantial increase in their functional capacities, facilitating their deployment in complex task execution and their role as the core reasoning engines for autonomous agents operating [1]. Despite their remarkable semantic understanding and planning abilities, LLM-based agents often struggle with tasks that require precise multi-step navigation within structured complex environments [2], [3], especially in Knowledge Graphs (KGs), where the combinatorial explosion of possible paths and the need for long-term, goaloriented exploration pose significant challenges for purely LLM-driven approaches, which are often not optimized for sequential decision-making in such discrete, dynamic spaces. Furthermore, they can be computationally expensive and prone to hallucinations when tasked with exhaustive exploration [2], [3].

However, Reinforcement Learning (RL) provides a powerful paradigm for learning optimal decision-making policies through interaction with an environment. RL agents show promising results in tasks involving sequential information search [4], [5] and can, in principle, learn to traverse a KG to find relevant information [6], [7]. However, traditional RL faces its own set of formidable challenges: difficulty in credit assignment over long trajectories, and the laborious, often heuristic-driven process of reward function design [2], [3]. Crafting a reward function that captures not only structural correctness but also the semantic relevance of a discovered path to a user’s query is particularly complex.

![](images/74f3aca509e5d1658564a97b021a3ee9cd23e4fcac8d3e1f25af248b8fdda3ef.jpg)  
FIGURE 1: Comparison of heuristic and LLM-based rewards for paths in knowledge graphs. In this example the question is formed from the triplet $( e _ { q } = \mathtt { E i f f e l }$ Tower, $r _ { q } = \mathtt { l o c a t e d \_ i n } , e _ { a } = \mathtt { P a r i s } )$ . Given $( e _ { q } , r _ { q } )$ , the agent must restore the initial triplet (direct $r _ { q }$ transition is masked out in the KG). The procedural heuristic (H) assigns the same reward to every path that reaches $\mathit { e } _ { a } ,$ regardless of whether the intermediate triples provide evidence for the original question. In contrast, the LLM-based reward (LLM) evaluates whether the discovered path supports reconstructing the original triplet: paths that justify the triplet receive a high reward, while indirect or irrelevant paths receive lower rewards.

Recognizing the complementary strengths of these two approaches, this paper explores a novel hybrid architecture that synergistically integrates LLMs and RL. We posit that an LLM can act as a powerful “mentor” to an RL agent, guiding its learning process and providing semantic feedback that transcends simple structural heuristics. This integration aims to overcome the limitations of each method in isolation: the LLM provides high-level semantic understanding and a dense, meaningful reward signal, while the RL agent learns a robust and adaptive policy for navigating the structured environment.

In this work, we focus on the challenging task of multihop question answering over knowledge graphs. We present a Retrieval Agent for Navigating Graphs with Semantic Evaluation Rewards (RANGER) in which an RL agent, based on transformer architecture and trained with Group Relative Policy Optimization (GRPO), learns to navigate a KG to retrieve information relevant to a user’s query.

The main research questions addressed are whether a learned traversal policy can improve the quality of retrieved information compared to classical rule-based search procedures, and whether reward signals produced by an LLM can guide the agent more effectively than rewards based on predefined structural criteria. We also examine how capable the mentor LLM needs to be in order for its feedback to provide an advantage over heuristic rewards.

The key contributions of our work are as follows:

1) LLM-as-a-Mentor for Reward Shaping: we move beyond simple heuristic-based rewards and employ an LLM to dynamically evaluate the semantic relevance of the path discovered by the RL agent. This LLMgenerated reward provides a context-aware learning signal, directly aligning the agent’s navigation policy with the information need expressed in the query.

The principle behind this distinction is illustrated in Figure 1.

2) Self-Supervised Training from the Knowledge Graph: we demonstrate that effective training of the RL agent can be achieved by automatically generating questionanswer pairs directly from the knowledge graph itself. This self-supervised approach reduces reliance on manually annotated datasets and yields a policy that can be used in QA systems for answering complex compositional questions.

3) Integrated Retrieval Pipeline: we detail the design and implementation of a complete pipeline that integrates our LLM-mentored RL agent as a knowledge retriever within a broader question-answering system. The agent’s learned policy guides the extraction of relevant triplets from the KG, which are then used by a separate LLM to generate the final answer.

4) Empirical Evaluation and Ablation Studies: we compare RANGER with non-parameteric graph traversal algorithms and conduct ablation studies to evaluate the effect of the reward design on downstream QA performance. In particular, we compare the proposed LLM-based reward with a procedural heuristic reward often used in prior KG traversal methods. Our results demonstrate a clear improvement, with RANGER achieving the highest scores across the majority of metrics, including a significant improvement in the LLM-as-a-Judge score.

Our findings underscore the potential of hybrid AI systems in which LLMs and RL agents are not merely combined, but are deeply integrated in a synergistic fashion. Using the semantic prowess of LLMs to shape the learning objective of an RL agent, we can create more effective, adaptable, and intelligent systems for complex information retrieval and reasoning tasks.

## II. RELATED WORK

Traditional reinforcement learning approaches for knowledge graph navigation laid the groundwork for graph reasoning by formulating it as a sequential decision-making problem on graph structures [6]–[8]. The subsequent advent of large language models (LLMs) introduced strong natural language understanding, in-context learning, and reasoning abilities [2], [3], driving their increasing adoption in Knowledge Graph Question Answering (KGQA). Yet pure LLM approaches suffer from fundamental shortcomings: they lack explicit structured traversal mechanisms, often fail at longterm planning, and tend to generate ungrounded facts [9], [10].

Consequently, LLM-enhanced reinforcement learning has emerged as a promising paradigm to address the inherent limitations of each approach when applied in isolation [2], [3], particularly for complex tasks like multi-hop reasoning on knowledge graphs (KGs). This section systematizes the primary methods of integration and situates our work within the context of KG-based Question Answering.

## A. RL-DRIVEN KNOWLEDGE GRAPH NAVIGATION

The initial approaches to using RL for KG reasoning often framed the problem as a classic path-finding task. An RL agent would learn a policy to navigate from a starting entity to a target entity, with the reward function based on heuristic criteria, such as successfully reaching a pre-defined set of answer entities [6]. As noted by [7], [8], a significant limitation of these early systems is that their manually designed reward functions, often based on structural graph features, may not capture the semantic relevance of the intermediate steps. The agent might learn to reach a correct entity, but the path taken could be logically unsound or the information gathered insufficient for a complete answer.

## B. LLM AS A POLICY, TRAINED WITH DETERMINISTIC OBJECTIVES

In the large family of RAG methods, the LLM itself serves as the policy and is trained with reinforcement learning using handcrafted, deterministic rewards. Typical reward functions rely on exact string matching of the answer, word occurrence, or similar lexical heuristics [11]–[18]. These deterministic rewards are cheap to evaluate and provide an unambiguous optimization signal; however, they are rigid and mostly applicable in training regimes with a single correct ground truth. These metrics cannot recognize paraphrased or semantically equivalent answers and penalize valid reasoning that diverges lexically from the gold path. Moreover, while LLMs serving as the policy are powerful and expressive, they are computationally heavy.

## C. LLMAS A MENTOR

A rarer paradigm uses an LLM as a mentor or critic that shapes the objective. An LLM-based objective promises a richer, semantically aware learning signal than deterministic heuristics, but it is more computationally expensive, can be

prone to hallucinations, and requires careful formulation.   
The following studies are most similar to our approach.

The authors of [19] train an LLM to reason over a KG using an LLM-as-judge reward: after each episode, the LLM evaluates the semantic similarity between the predicted answer and the ground truth. However, the policy learns to trigger a KG retrieval tool rather than to explicitly traverse the graph. In work [20], an actor-critic framework is proposed in which an LLM critic evaluates actions through future rollouts. A key concern is that the policy itself generates the candidate action set, making the framework fragile and prone to reward exploitation. Moreover, the method requires multiple LLM invocations per episode step, incurring heavy computational overhead. In [21], the authors design a hierarchical framework in which a high-level agent chooses an operation type for a low-level agent that navigates paths. The path reasoning LLM mentor chooses the most promising expansion direction but sees only the history up to the current timestep and immediate actions, restricting guidance to greedy local optima. The approach is also computationally expensive, requiring an LLM call per training step. Finally, the work of [22] uses LLM feedback to motivate the agent to explore more useful information triples. However, the policy requires supervised pretraining on triples that logically support target statements, which are mined using another algorithm. Furthermore, the policy is an encoderdecoder architecture that generates triples in textual form based only on the query triple instead of gathering them from the KG; thus, the approach risks hallucinations and requires supervised retraining whenever the KG changes.

Based on these insights, we introduce a novel reinforcement learning agent based on a lightweight transformer architecture and designed for knowledge graph traversal. It directly navigates the graph, requires no supervised pretraining, and is trained with a single LLM invocation per episode, which incentivizes not merely producing an answer but gathering supporting evidence. Section 4 details its architecture and elaborates on the intricate mechanisms of its integration with LLM, covering the procedures for both training and inference.

## III. PROBLEM FORMULATION

## A. QUESTION ANSWERING TASK

We consider a question answering task in a knowledge graph. While questions can be complex and involve multiple dependent conditions, in this section, we focus on a simple type of question with a single clause. Formally, a knowledge graph is defined as a directed multigraph $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , E )$ where E is a set of entities represented as nodes, R is a set of binary relations between entities, and $E \subset \mathcal { E } \times \mathcal { R } \times \mathcal { E }$ is a set of edges. Then a simple question can be formulated as predicting a destination entity $e _ { a }$ that forms a correct edge $( e _ { q } , r _ { q } , e _ { a } )$ , given a query entity $e _ { q }$ and a relation of interest $r _ { q } .$ For each query, there may be multiple possible correct answers, for example (apple; is a; fruit) and (apple; is a; food). In this case we consider finding at least one of them to be a sufficient condition for success.

To solve the problem, we build a learnable graph traversal algorithm, where the agent sequentially makes transitions between the graph vertices and constructs a continuous path $( e _ { 1 } , r _ { 1 } , \ldots , e _ { n } )$ that is decomposed into triplets $( e _ { i } , r _ { i } , e _ { i + 1 } )$ constituting the retrieved information. In many works [6], [7] the question answering task is equated to finding a path starting at node $e _ { q }$ and finishing at $e _ { a }$ . However, far from every such path in a knowledge graph implies the statement $( e _ { q } , r _ { q } , e _ { a } )$ . For instance, a query (S. Ramanujan; academic advisor is; G. H. Hardy) can not be inferred from a sequence of transitions (S. Ramanujan; is known for; Hardy-Ramanujan asymptotic formula; stud by; G. H. Hardy). Thus, we argue that in this case, an RL policy often receives positive reinforcement when it should not, and such a formulation is poor for learning information retrieval and reasoning. Besides, in sparse graphs, a situation may arise where the required continuous path does not exist, yet the needed information is still present. Therefore, we pose the task as a search for information supporting the statement $( e _ { q } , r _ { q } , e _ { a } )$ . Formulating a strict mathematical criterion for measuring success rate in this task is impractical. Hence, to evaluate the solution quality, we employ large language models, which we discuss further in Section IV.

## B. RL NOTATION

The graph traversal process can be interpreted as a partially observed deterministic Markov Decision Process (MDP) $( \mathcal { S } , \mathcal { A } , \mathcal { O } , \mathbf { T } , \mathcal { R } )$ , and the agent can be trained using a reinforcement learning algorithm. Below we elaborate on each element of the MDP.

## 1) States and observations.

The state $s _ { t } \in S$ at time t is a tuple $\left( s _ { t } ^ { q } , s _ { t } ^ { p } , s _ { t } ^ { a } \right)$ , where $s _ { t } ^ { q } =$ $( e _ { q } , r _ { q } )$ is a query, $s _ { t } ^ { p } = ( e _ { 1 } , r _ { 1 } , \ldots , e _ { t } ) , ( e _ { i } , r _ { i } , e _ { i + 1 } ) \in E$ is a current path sequence, and $s _ { t } ^ { a } = \{ e \mid ( e _ { q } , r _ { q } , e ) \in E \}$ is a set of possible answers to the query. The agent knows the state except for the answers, so its observation is $o _ { t } =$ $( s _ { t } ^ { q } , s _ { t } ^ { p } ) \in \mathcal { O }$

## 2) Actions.

The possible actions $A _ { t } \subseteq A$ at time t are the set of all outgoing edges and incoming edges with reversed relations from the current node $e _ { t } \colon \ A _ { t } \ = \ \{ ( r , e ) \ | \ ( e _ { t } , r , e ) \ \in $ $E \} \cup \{ ( r ^ { - 1 } , e ) \mid ( e , r , e _ { t } ) \in E \}$ . Additionally, we introduce a small set of virtual edges that allow the agent to jump to semantically similar entities, enabling navigation between disconnected components of the graph. Finally, a special “END” action is added that terminates the episode.

## 3) Transitions.

The transition function $\mathbf { T } : \mathcal { S } \times \mathcal { A }  \mathcal { S }$ is deterministic and fully determined by the chosen edge. Specifically, in case of a non-terminal transition, given state $s _ { t } ~ = ~ \left( s _ { t } ^ { q } , s _ { t } ^ { p } , s _ { t } ^ { a } \right)$ and action $a _ { t } ~ = ~ ( r , e ) ~ \in ~ A _ { t } .$ , the next state is $s _ { t + 1 } =$ $( s _ { t + 1 } ^ { q } , s _ { t + 1 } ^ { p } , s _ { t + 1 } ^ { a } )$ , where $s _ { t + 1 } ^ { q } = s _ { t } ^ { q } , \ s _ { t + 1 } ^ { a } = s _ { t } ^ { a }$ (the query and the set of possible answers remain unchanged), and $s _ { t + 1 } ^ { p } = ( e _ { 1 } , r _ { 1 } , \ldots , e _ { t } , r , e )$ (the path is extended with the chosen relation and entity). If the agent selects the “END” action or number of steps exceeds a predefined limit T, the episode terminates and the final state is reached.

## 4) Rewards.

The reward function $\mathcal { R } : \mathcal { S }  \mathbb { R }$ assigns a scalar value to each state, incentivizing correct actions and thereby defining the agent’s objective. We compare two distinct reward formulations: our proposed LLM-based approach and a heurisd tic baseline, both described in Section IV.

## IV. METHODS

## A. LLM-BASED REWARD

Our proposed method utilizes an LLM to estimate the relevance of the agent’s trajectory. It answers the question: “Can any of the statements $( e _ { q } , r _ { q } , e _ { a } ) \in E$ be logically derived from retrieved information $\therefore ? ^ { \ast }$ . To enable finer-grained credit assignment, we prompt the LLM to select from three discrete outcomes: “Success” (full logical entailment), “Highly Relevant” (retrieved information is pertinent but does not directly support the target statements), and “Failure” (not even close). These categories are mapped to scalar rewards $r \in \{ 1 , 0 . 5 , 0 \}$ , respectively. For example see Figure 1.

The main advantage of an LLM as a critic over traditional symbolic or embedding-based metrics is that the LLM can recognize valid logical inferences even when the reasoning chain involves paraphrasing, implicit knowledge, or multihop synthesis that does not map cleanly to graph edges. This naturally makes it robust and more applicable to real-world knowledge graphs. Furthermore, it is better aligned with human preferences and captures nuances when evaluating the relevance and coherence of supporting evidence that purely structural similarity metrics might overlook.

However, one of the notable challenges this approach introduces is high computational cost. For this reason, in this study, the agent receives a non-zero reward value only in terminal states. Another concern is that LLMs can hallucinate, and it is reasonable to expect that smaller language models will yield lower-quality responses. However, as demonstrated in Section VI, even a small-sized LLM mentor can match or exceed procedural rewards in terms of downstream utility.

For comparison, we additionally train the RL agent with a heuristic reward function following prior work on knowledge graph reasoning [6], [7]. Under this formulation, the agent receives a sparse terminal reward $r _ { t } = 1$ if the final entity $e _ { t }$ belongs to the answer set $s _ { a } .$ , and 0 otherwise. In our experiments, the agent struggles to converge with this sparse reward alone; we therefore help it with a small dense term $\begin{array} { r } { \operatorname* { m i n } _ { a , e _ { k } \in s _ { t - 1 } ^ { p } } \mathrm { d i s t } ( e _ { k } , e _ { a } ) - \operatorname* { m i n } _ { a } \mathrm { d i s t } ( e _ { t } , e _ { a } ) } \end{array}$ scaled by $w _ { \mathrm { d e n s e } } = 0 . 0 1$ . This term provides a learning signal early in training while having negligible impact near convergence. For RANGER with LLM-based reward, however, this dense component provided no benefit, likely because it incentivizes rapid progression toward answer entities, whereas the LLM reward encourages broader graph exploration.

## B. POLICY ARCHITECTURE

The policy network in our method is an Actor-Critic model based on the Transformer architecture [23] and trained from scratch. At each time step t, the network receives an observation $o _ { t } ~ = ~ ( s _ { t } ^ { q } , s _ { t } ^ { p } )$ from the environment and the set of available actions $A _ { t }$ , and produces a probability distribution $\textstyle \pi { \bigl ( } a _ { t } { \bigr | } \quad o _ { t } , A _ { t } { \bigr ) }$ over the discrete action space as well as a scalar estimate $V ( o _ { t } )$ of the state value. The operating principle of the policy network for a single step is shown in Figure 2.

Textual representations of entities and relations are converted into fixed-dimensional vectors using a pre-trained sentence-transformers embedding model [24]. The policy network receives three distinct input components:

1) The query pair $( e _ { q } , r _ { q } )$ is merged into a single vector $[ \mathbf { e _ { q } } ; \mathbf { r _ { q } } ]$ , where $\mathbf { e _ { q } }$ and $\mathbf { r _ { q } }$ denote the entity and relation embeddings, respectively, and [; ] indicates vector concatenation.

2) Available actions $A _ { t }$ comprise outgoing edges, incoming edges (with reversed relations), and semantic similarity virtual edges. To enable the policy network to distinguish virtual edges from actual ones and identify reversed relations, we introduce learnable action type tokens ${ \bf r _ { o u t } } , { \bf r _ { i n v } } , { \bf r _ { s i m } } \in \mathbb { R } ^ { z }$ . Let $\tau _ { r } : { \mathcal { A } } $ $\{ \mathbf { r _ { o u t } } , \mathbf { r _ { i n v } } , \mathbf { r _ { s i m } } \}$ be the selector function, then for action $\boldsymbol { a } ~ = ~ \left( r , e \right)$ we define an augmented relation vector representation as $\hat { \mathbf { r } } = [ \mathbf { r } ; \tau _ { r } ( a ) ]$

Furthermore, as described below, knowledge graphs in the testing framework contain heterogeneous node types. So we also define entity type tokens $\mathbf { e _ { o b j } } , \mathbf { e _ { t h } } , \mathbf { e _ { e p } } \in \mathbb { R } ^ { z }$ and augment entity encodings as $\hat { \mathbf { e } } ~ = ~ [ \mathbf { e } ; \tau _ { n } ( e ) ]$ , where $\tau _ { n } ( \cdot )$ is the node type selector function. The vector representation of an action $a \ \in \ A _ { t }$ is then $\mathbf { a } ~ = ~ [ \hat { \mathbf { r } } , \hat { \mathbf { e } } ]$ . Additionally, a special learnable $\mathrm { ^ { 6 6 } E N D ^ { \circ } }$ token of the same dimension as other actions is appended to the action set.

3) Each entity and relation in the current traversal path $s _ { t } ^ { p }$ constitutes a separate sequence element. When extending the path after a transition, we keep augmented entity and relation features. Also a sinusoidal positional encoding pos indicating their order in the sequence is mixed-in, yielding the following path encoding: $\mathbf { P _ { t } } = \left( [ \hat { \mathbf { e } } _ { 1 } , p o s _ { 1 } ] , [ \hat { \mathbf { r } } _ { 1 } , p o s _ { 1 } ] , \dots , [ \hat { \mathbf { e } } _ { \mathbf { t } } , p o s _ { t } ] \right)$

The elements of the input sequences are projected to the same dimensionality $\mathbb { R } ^ { ( \cdot ) \times d }$ with their own linear transformations:

$$
\begin{array} { r } { \mathbf { x _ { q } } = \mathbf { W _ { q } } [ \mathbf { e _ { q } } ; \mathbf { r _ { q } } ] , \quad \mathbf { X _ { p } } = \mathbf { W _ { p } } \mathbf { P _ { t } } , \quad \mathbf { X _ { a } } = \mathbf { W _ { a } } \mathbf { A _ { t } } , } \end{array}
$$

Then all components are combined into a single sequence and processed by a multi-layer Transformer encoder $F _ { e n c } \colon$

$$
\mathbf { h _ { q } } , \mathbf { H _ { p } } , \mathbf { H _ { a } } = F _ { e n c } ( \mathbf { x _ { q } } , \mathbf { X _ { p } } , \mathbf { X _ { a } } ) .
$$

The self-attention mechanism in $F _ { \mathrm { e n c } }$ aggregates information from the entire input sequence, yielding global contextaware representations. The state value is computed from the query feature vector $\mathbf { h } _ { q } \in \mathbb { R } ^ { d }$ via a two-layer MLP:

$$
V _ { t } = { \bf w _ { 2 } } \sigma ( { \bf W _ { 1 } h _ { q } } + { \bf b _ { 1 } } ) + b _ { 2 } ,
$$

where $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { d \times d }$ $\mathbf { b } _ { 1 } \in \mathbb { R } ^ { d }$ ${ \bf w } _ { 2 } \in \mathbb { R } ^ { d }$ , and $b _ { 2 } \in \mathbb { R }$ are parameters, and $\sigma ( \cdot )$ denotes a non-linear activation (ReLU).

Action probabilities are derived from the encoded action representations $\mathbf { H } _ { a } \in \mathbb { R } ^ { k \times d }$ :

$$
\pi _ { t } = \mathrm { s o f t m a x } ( \mathbf { H _ { a } W } _ { \pi } + \mathbf { b } _ { \pi } ) ,
$$

where $\mathbf { W } _ { \pi } \in \mathbb { R } ^ { d }$ , and $\mathbf { b } _ { \pi } \in \mathbb { R } ^ { d }$

During training, action choices are sampled from $\pi _ { t }$ to encourage exploration; at inference, we select arg ma $\textstyle \operatorname { l x } _ { a } \pi _ { t } ( a \mid$ $o _ { t } , A _ { t } )$

## C. EXTENSION TO GENERAL QUESTIONS

The traversal algorithm can, in principle, be easily extended to complex questions involving multiple dependent conditions by encoding the whole question text. However, this would require annotated question-answer pairs. In this work, we focus on simple questions, as this choice enables unsupervised training of the agent: such queries can be generated procedurally from the knowledge graph itself by sampling existing edges $( e _ { q } , r _ { q } , e _ { a } ) \in E$ and masking direct transitions to answers. The resulting algorithm can then be generalized to complex questions by decomposing them into sub-questions, for instance, using a language model, and subsequently aggregating the information retrieved for each simple query to construct the final answer.

## D. AGENT TRAINING ALGORITHM

To train the graph traversal policy described above we employ the Group Relative Policy Optimization (GRPO) algorithm [25]. For a given query q, the policy $\pi _ { \theta _ { \mathrm { o l d } } }$ generates a group of G trajectories $\{ \tau _ { 1 } , \dots , \tau _ { G } \}$ . The advantage for the i-th trajectory is computed as the normalized deviation from the group mean:

$$
\hat { A } _ { i , t } = \frac { r _ { i } - \bar { r } } { \sigma _ { r } } ,\tag{1}
$$

where $r _ { i }$ is the reward for trajectory i, r¯ and $\sigma _ { r }$ are the mean reward and standard deviation across the group, respectively. The policy is then updated using the clipped surrogate objective:

![](images/806c8861a63c0cccf36273e6518765c0e9599828c672a8fa9ef1ab1a69191000.jpg)  
FIGURE 2: Agent model architecture. Blocks containing trainable parameters are highlighted in yellow, while output predictions are highlighted in green. The value function head V is used only for PPO algorithm.

$$
\begin{array} { r l r } & { } & { L ^ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { q \sim P ( Q ) , \{ \tau _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \mathrm { o l d } } ( \cdot | q ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } } \\ & { } & { \big [ \operatorname* { m i n } \big ( \rho _ { i , t } ( \theta ) \hat { A } _ { i , t } , \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i , t } \big ) \big ] , } \end{array}\tag{2}
$$

where $\begin{array} { r } { \rho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( a _ { i , t } | o _ { i , t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { i , t } | o _ { i , t } \right) } } \end{array}$ is the probability ratio for the action taken at timestep t in the i-th trajectory for query $q ,$ and ϵ controls the clipping range. Unlike the original GRPO formulation, we omit the KL-divergence regularization term, as our policy is trained from scratch rather than fine-tuned from a pretrained reference model.

The choice of GRPO is motivated by the sparse terminal reward signal: since the LLM judge provides feedback only at the end of an episode and, at least early in training, non-zero rewards are rare, training a value function would yield noisy advantage estimates. For RANGER(H), which combines the sparse terminal reward with the auxiliary dense term, we employ the process-supervised variant of GRPO. Nevertheless, we also conducted an ablation study with Proximal Policy Optimization (PPO) [26]. Comparative performance of the PPO version is provided in Section VI.

Given a knowledge graph, we generate training episodes as follows. First, an entity $e _ { q }$ and one of its outgoing edges with relation $r _ { q }$ are selected in a random manner. We then identify the set of all valid answers among outgoing edges where the textual representation of the relation matches $r _ { q } \colon \{ e _ { a } ~ | ~ ( e _ { q } , r , e _ { a } ) \in E \ \wedge \}$ text (r) = text $\left( r _ { q } \right) \}$ . The starting node (origin of the path) is chosen in one of two ways. With a predefined probability $\boldsymbol { p } ( \boldsymbol { e } _ { 1 } ~ = ~ \boldsymbol { e } _ { q } )$ , we set the starting node to the query entity $e _ { q }$ itself, excluding direct transitions to answer entities to prevent the task from becoming trivial. Otherwise, we sample a starting node from the neighborhood of $e _ { q }$ at a distance of 2 to 5 hops. Episodes with the first type of starting node can be viewed as knowledge completion tasks, whereas those with the second type correspond to search tasks.

It should be noted that, under edge masking, the knowledge recovery task is not always solvable: if the graph contains no alternative evidence supporting the removed fact, that fact cannot be inferred from the remaining structure. Consequently, the fraction of informative training episodes depends on the completeness and connectivity of the underlying knowledge graph. In our graphs, this fraction appears to be relatively small, as suggested by the learning curves

![](images/67183fd4113f0baa4f97afd5c73c59e6ea2963ee96f5aa2a7dd230fec935795d.jpg)  
FIGURE 3: Answer generation and training pipeline. The agent is embedded into the QA-system retriever component and traverses the knowledge graph to gather information. Red arrows show the training loop.

reported in Appendix A.1.

## E. INTEGRATION INTO QUESTION-ANSWERING PIPELINE

To test the developed agent in a real-world questionanswering scenario, we integrate it into the PersonalAI framework [27].

The framework is also utilized for building knowledge graphs from contexts. When extracting facts, it creates nodes of three types: object, thesis, and episodic, which differ in the embedded text volume. The object node most closely corresponds to the conventional concept of an entity, whereas the episodic node contains a brief description. Queries generated for training are allowed to contain only object type entities.

The testing pipeline consists of query preprocessing, knowledge retrieval, and answer generation stages, as illustrated in Figure 3.

In the first stage, all entities are extracted from the query using an LLM and are associated with vertices of the knowledge graph via exact matching or similarity in the embedding space. We also added our own operation that extracts a set of simple sub-queries in the form of entityrelation pairs.

The RL agent is incorporated into the retriever component of the framework. It builds n×k trajectories for each of the n simple sub-queries, starting from each of the k matched entities. Such cross-search can help reveal relationships between entities. The triplets from all trajectories are then collected into a common pool and pruned to a maximum of $N _ { m a x }$ items. For this purpose, we implemented an embeddingbased ranking mechanism that assigns high scores to triplets that are semantically similar to at least one of the subqueries. Finally, the LLM generates a textual answer based on the query and the collected information.

## V. EXPERIMENTS

The experiments are organized around three aspects. First, we compare standalone RL policies used as retrievers against non-learned graph traversal baselines to gauge their effectiveness for information retrieval. Second, we compare the LLM-based reward with the heuristic reward in terms of answer quality. Finally, we analyze how the size of the LLM mentor used during training affects the quality of the resulting retrieval policy.

As described above, we evaluate the methods within the QA pipeline, as this provides a natural and interpretable way to estimate graph reasoning capability. To isolate the contribution of the retriever, the same query preprocessing, answer generation, and evaluation procedures are used across all runs. For LLM-requiring operations in the generative part of the pipeline, we use Mistral Small 3 [28].

## A. EVALUATION METRICS

For each question, the system retrieves evidence from the knowledge graph and generates an answer, which is then compared against the reference answer. We report BLEU-1, BLEU-2, ROUGE-L, METEOR, and Exact Match [29]–[31] under standard normalization of case and punctuation. To complement lexical-overlap metrics and approximate human review, we perform an LLM-based correctness assessment following the LLM-as-a-Judge protocol [32]. Given the question, the reference answer, and the generated answer, an independent judge model (Mistral Medium 3 [33]) assigns a score of 1 for a correct answer, 0 for an incorrect answer, and 0.5 for a partially correct answer. The latter category is used, for example, when the answer lists only a subset of the required entities or provides a less specific response, such as a country instead of the expected city.

All metrics are averaged over 150 questions in our experiments.

## B. BASELINES

We use two non-parametric traversal algorithms as baselines. The first one is Random Walk, which serves as a lower bound for graph reasoning ability. In this setting, the retriever makes random transitions with the same step budget and starting nodes as the RL agents. The second and more sophisticated method is WaterCircles, the default retriever algorithm used in PersonalAI [27]. It performs multiple breadth-first searches and selects paths with high co-occurrence of the entities mentioned in the question.

## C. RL-GUIDED RETRIEVAL

First, we test RL-guided retrieval on the HotpotQA benchmark, which is designed to evaluate the multi-hop reasoning ability [34]. For this purpose, we construct a knowledge graph from the HotpotQA contexts and train RL agents on this graph for 10<sup>7</sup> environment steps in four different configurations. As an ablation study, we consider two reward functions: an LLM-based reward, denoted as RANGER(LLM), and a heuristic reward, denoted as RANGER(H). Both variants are trained with two RL algorithms, GRPO and PPO.

We further evaluate a hybrid retrieval setting, motivated by the observation that combining RL-guided traversal with the baseline Watercircles retriever yields substantially greater performance gains than merely increasing the evidence budget of any standalone retriever. In contrast to the standalone scenario, in the hybrid retriever triplets retrieved by WaterCircles are combined with triplets obtained from the RL traversal policy. The resulting triplets are then processed in the standard manner: deduplicated, ranked, and pruned to the same total number of triplets as in all other configurations.

For the hybrid experiments, we consider the same reward functions and RL algorithms as in the standalone setting. To avoid confusion, we use the dagger symbol to denote hybrid configurations: RANGER(LLM)<sup>†</sup> for the LLM-based reward and RANGER(H)<sup>†</sup> for heuristic reward. In addition to HotpotQA, we evaluate the hybrid retriever on TriviaQA and DiaASQ [35], [36], which represent open-domain factual retrieval and context-dependent dialogue understanding, respectively. The characteristics of the knowledge graphs constructed from these benchmarks are shown in Table 1.

Finally, we investigate how many parameters are required in the LLM mentor to provide an advantage over alternative retrieval approaches. The motivation for this experiment is that larger models are generally expected to better capture the context and provide more accurate reward signals, but they also require more computational resources and increase the training time. To study this trade-off, we report the LLMas-a-Judge score for RANGER(LLM) trained with GRPO using three LLM mentors: Ministral 3 (8B), Mistral Small 3.2 (24B), and Mistral Medium 3.5 (128B). In the experiments above, we use Mistral Small as the default mentor, since, as shown in Section VI, it provides a favorable balance between retrieving quality and computational cost.

TABLE 1: Characteristics of the HotpotQA, TriviaQA, and DiaASQ knowledge graphs.
<table><tr><td>Dataset</td><td>#Nodes</td><td># Edges</td></tr><tr><td>HotpotQA</td><td>71,335</td><td>195,614</td></tr><tr><td>DiaASQ</td><td>54,220</td><td>239,517</td></tr><tr><td>TriviaQA</td><td>123,135</td><td>334,868</td></tr></table>

## VI. RESULTS AND DISCUSSION

Table 2 presents the QA performance of the baseline and RL-based retrieval methods on the HotpotQA benchmark. Random Walk performs worst across all metrics, showing that unguided graph traversal is insufficient for effective evidence retrieval. WaterCircles is a much stronger baseline and achieves the second-best results for most lexical metrics. However, the learned RL retrievers are able to match or exceed it in terms of answer correctness, which is closely reflected by the LLM-as-a-Judge score. This indicates that training a policy to navigate the graph can improve the retrieval of answer-relevant evidence beyond algorithmic traversal strategies.

RANGER(LLM) trained with GRPO achieves the best results across all reported metrics. It improves the LLM-as-a-Judge score by approximately 3.6% relative to WaterCircles and by approximately 3.0% relative to the GRPO policy trained with the heuristic reward. Among the RL-based approaches, GRPO shows stronger answer-correctness performance than PPO, while the variants trained with the LLM mentor consistently outperform their heuristic-reward counterparts in terms of the LLM-as-a-Judge score. This suggests that the LLM-based reward provides a more effective training signal for learning retrieval policies.

Next, we present the results for the combined retrievers on three benchmarks in Table 3. In terms of answer correctness, RANGER(LLM)<sup>†</sup> trained with GRPO achieves the highest on all datasets. Compared with the WaterCircles baseline, it improves this score by approximately 20.4% on HotpotQA, 4.4% on TriviaQA, and 7.7% on DiaASQ. Overall, augmenting the base WaterCircles retriever with RL-guided policies substantially improves the answering ability across all benchmarks. This indicates that the learned policies retrieve complementary evidence missed by the baseline retriever rather than simply duplicating its behavior.

For the hybrid pipelines, the observations regarding LLM guidance and the choice of RL algorithm generally remain consistent with the previous runs: GRPO tends to provide stronger answer-correctness performance, and LLM-based reward variants usually outperform their heuristic-reward counterparts. The main exception to this trend is DiaASQ, where RANGER(H)<sup>†</sup> under the PPO setting beats other approaches in lexical metrics and is the second-best method in the LLM-as-a-Judge score. This can be attributed to a fundamental difference in reward properties: the heuristic reward is structure-based, whereas the LLM reward is semanticsoriented. Since DiaASQ is constructed from dialogues and contains many semantically similar transitions, evaluating the usefulness of retrieved paths may be more challenging for the LLM mentor.

TABLE 2: Comparison of Random Walk, WaterCircles, and RANGER retrieval methods on the first 150 questions from HotpotQA. Best results are shown in bold, and second-best results are underlined.
<table><tr><td>Method</td><td>RL Algorithm</td><td>LLM Judge</td><td>BLEU-1</td><td>BLEU-2</td><td>ROUGE-L</td><td>METEOR</td><td>Exact Match</td></tr><tr><td>Random Walk</td><td>一</td><td>0.373</td><td>0.263</td><td>0.314</td><td>0.355</td><td>0.257</td><td>0.260</td></tr><tr><td>WaterCircles</td><td></td><td>0.673</td><td>0.455</td><td>0.324</td><td>0.578</td><td>0.452</td><td>0.467</td></tr><tr><td>RANGER(H)</td><td>PPO</td><td>0.657</td><td>0.434</td><td>0.305</td><td>0.545</td><td>0.432</td><td>0.440</td></tr><tr><td>RANGER(LLM)</td><td>PPO</td><td>0.691</td><td>0.437</td><td>0.304</td><td>0.565</td><td>0.451</td><td>0.447</td></tr><tr><td>RANGER(H)</td><td>GRPO</td><td>0.677</td><td>0.419</td><td>0.277</td><td>0.563</td><td>0.424</td><td>0.460</td></tr><tr><td>RANGER(LLM)</td><td>GRPO</td><td>0.697</td><td>0.467</td><td>0.340</td><td>0.585</td><td>0.468</td><td>0.493</td></tr></table>

TABLE 3: Results of hybrid retrieval methods on the first 150 questions from HotpotQA, TriviaQA, and DiaASQ for GRPO and PPO algorithms. Best results are shown in bold, and second-best results are underlined.
<table><tr><td colspan="8"></td></tr><tr><td>Graph</td><td>RL Algorithm</td><td>Method</td><td>LLM Judge</td><td>BLEU-1</td><td>BLEU-2</td><td>ROUGE-L</td><td>METEOR</td><td>Exact Match</td></tr><tr><td rowspan="5">HotpotQA</td><td></td><td>WaterCircles</td><td>0.673</td><td>0.455</td><td>0.324</td><td>0.578</td><td>0.452</td><td>0.467</td></tr><tr><td>GRPO</td><td>RANGER(H)†</td><td>0.767</td><td>0.505</td><td>0.350</td><td>0.651</td><td>0.509</td><td>0.533</td></tr><tr><td></td><td>RANGER(LLM)†</td><td>0.810</td><td>0.546</td><td>0.367</td><td>0.674</td><td>0.535</td><td>0.540</td></tr><tr><td>PPO</td><td>RANGER(H)†</td><td>0.743</td><td>0.507</td><td>0.344</td><td>0.630</td><td>0.498</td><td>0.520</td></tr><tr><td></td><td>RANGER(LLM)†</td><td>0.767</td><td>0.524</td><td>0.352</td><td>0.647</td><td>0.509</td><td>0.520</td></tr><tr><td rowspan="5">TriviaQA</td><td>一</td><td>WaterCircles</td><td>0.850</td><td>0.600</td><td>0.297</td><td>0.695</td><td>0.527</td><td>0.580</td></tr><tr><td>GRPO</td><td>RANGER(H)†</td><td>0.870</td><td>0.599</td><td>0.312</td><td>0.712</td><td>0.552</td><td>0.587</td></tr><tr><td></td><td>RANGER(LLM)†</td><td>0.887</td><td>0.621</td><td>0.322</td><td>0.734</td><td>0.559</td><td>0.593</td></tr><tr><td>PPO</td><td>RANGER(H)†</td><td>0.853</td><td>0.617</td><td>0.314</td><td>0.710</td><td>0.544</td><td>0.600</td></tr><tr><td></td><td>RANGER(LLM)†</td><td>0.870</td><td>0.619</td><td>0.321</td><td>0.721</td><td>0.552</td><td>0.607</td></tr><tr><td rowspan="5">DiaASQ</td><td>一</td><td>WaterCircles</td><td>0.467</td><td>0.238</td><td>0.027</td><td>0.396</td><td>0.226</td><td>0.313</td></tr><tr><td>GRPO</td><td>RANGER(H)†</td><td>0.493</td><td>0.326</td><td>0.025</td><td>0.483</td><td>0.263</td><td>0.447</td></tr><tr><td></td><td>RANGER(LLM)†</td><td>0.503</td><td>0.304</td><td>0.019</td><td>0.481</td><td>0.262</td><td>0.440</td></tr><tr><td>PPO</td><td>RANGER(H)†</td><td>0.500</td><td>0.333</td><td>0.029</td><td>0.484</td><td>0.267</td><td>0.447</td></tr><tr><td></td><td>RANGER(LLM)†</td><td>0.480</td><td>0.326</td><td>0.019</td><td>0.470</td><td>0.257</td><td>0.427</td></tr></table>

Figure 4 further breaks down the LLM-as-a-Judge results at the question level. On HotpotQA and TriviaQA, the sets of successfully answered questions substantially overlap, as shown by the large number of equal positive scores. However, across all three datasets, and especially on DiaASQ, there are still questions where RANGER(H)<sup>†</sup> outperforms RANGER(LLM)<sup>†</sup>. This suggests that the LLMbased reward approach has room for improvement, and that a hybrid reward combining semantic feedback with structurebased conditions may lead to further gains.

Figure 5 reports the LLM-as-a-Judge scores for RL policies trained with various LLM mentors. The quality drops substantially when the mentor is reduced from Mistral Small 3.2 (24B) to Ministral 3 (8B), falling below the WaterCircles baseline. However, scaling up to Mistral Medium 3.5 (128B) yields only a marginal improvement over Mistral Small 3.2.

Thus, the usefulness of the mentor model increases with model size, but appears to saturate beyond a certain point. The first observation is expected, since the agent partially inherits the reasoning capability of the mentor through the reward signal.

Interestingly, the training curves shown in Appendix A.2 reveal an inverse relation between the mentor size and the magnitude of the assigned rewards. Ministral 3 (8B) gives the most optimistic estimates of retrieved-path relevance, whereas Mistral Medium 3.5 (128B) behaves as the strictest critic. A plausible explanation is that smaller models are more prone to reward-assignment errors, which can make their scores overly optimistic and may also create opportunities for the RL agent to exploit weaknesses in the mentor.

## VII. CONCLUSION

Based on the experimental results and the presented framework, we conclude that integrating an LLM into the RL training loop through the “LLM-as-a-Mentor” paradigm is an effective strategy for multi-hop question answering over knowledge graphs. By using an LLM to provide contextaware reward signals, the approach addresses the limitations of heuristic-based rewards, guiding the RL agent to learn navigation policies that align with the information needs expressed in the query. Importantly, our results also show that the RL agent can be trained in a self-supervised manner directly from the knowledge graph and then used to answer complex questions, without relying on manually curated question-answer datasets. Empirical validation across multiple datasets shows a progressive improvement in both lexical metrics and LLM-as-a-Judge scores, outperforming baseline retrieval-augmented systems and those relying solely on RL with heuristic rewards. These findings suggest that deeply integrated, feedback-driven architectures offer a promising direction for knowledge-intensive reasoning tasks.

![](images/09cea4b1dc21d2e6147a4735534a2b5c61d6952b48f065f9fa53f298e3a51ce7.jpg)  
FIGURE 4: Per-question comparison of RANGER(LLM)<sup>†</sup> and RANGER(H)<sup>†</sup> using LLM-as-a-Judge scores on the 150 questions from each dataset. Each stacked bar partitions questions into four mutually exclusive groups: equal zero scores, RANGER(LLM)<sup>†</sup> better, RANGER(H)<sup>†</sup> better, and equal positive scores.

![](images/f967e7bdb644982cf147cd1ba856bc065dc402a985e93a6529371a1c25ae6805.jpg)  
LLM mentor size, number of parameters (bln, log scale)  
FIGURE 5: Dependence of the LLM-as-a-Judge score on the size of the LLM Mentor. The score drops substantially from Mistral Small 3.2 (24B) to Ministral 3 (8B), while the improvement from Mistral Small 3.2 (24B) to Mistral Medium 3.5 (128B) is marginal, suggesting diminishing returns from further scaling.

## REFERENCES

[1] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J.-R. Wen, “A survey on large language model based autonomous agents,” 2024. [Online]. Available: https://arxiv.org/abs/2308.11432

[2] Y. Cao, H. Zhao, Y. Cheng, T. Shu, Y. Chen, G. Liu, G. Liang, J. Zhao, J. Yan, and Y. Li, “Survey on large language model-enhanced reinforcement learning: Concept, taxonomy, and methods,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 6, pp. 9737–9757, 2024.

[3] S. Schoepp, M. Jafaripour, Y. Cao, T. Yang, F. Abdollahi, S. Golestan, Z. Sufiyan, O. R. Zaiane, and M. E. Taylor, “The evolving landscape of llm-and vlm-integrated reinforcement learning,” arXiv preprint arXiv:2502.15214, 2025.

[4] A. Sorokin, N. Buzun, A. Anokhin, E. Vedernikov, P. Anokhin, M. Burtsev, and E. Burnaev, “Q-RAG: Long context multi-step retrieval via valuebased embedder training,” in Proceedings of the International Conference on Learning Representations (ICLR), 2026.

[5] J. Huang, S. Madala, R. Sidhu, C. Niu, H. Peng, J. Hockenmaier, and T. Zhang, “Rag-rl: Advancing retrieval-augmented generation via rl and curriculum learning,” arXiv preprint arXiv:2503.12759, 2025.

[6] R. Das, S. Dhuliawala, M. Zaheer, L. Vilnis, I. Durugkar, A. Krishnamurthy, A. Smola, and A. McCallum, “Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning,” arXiv preprint arXiv:1711.05851, 2017.

[7] Y. Ma, O. Burns, M. Wang, G. Li, N. Du, L. El Shafey, L. Wang, I. Shafran, and H. Soltau, “Knowledge graph reasoning with selfsupervised reinforcement learning,” IEEE Transactions on Audio, Speech and Language Processing, 2025.

[8] B. Peng, Y. Zhu, Y. Liu, X. Bo, H. Shi, C. Hong, Y. Zhang, and S. Tang, “Graph retrieval-augmented generation: A survey,” ACM Transactions on Information Systems, vol. 44, no. 2, pp. 1–52, 2025.

[9] G. Agrawal, T. Kumarage, Z. Alghamdi, and H. Liu, “Can knowledge graphs reduce hallucinations in llms?: A survey,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024, pp. 3947–3960.

[10] N. Ibrahim, S. Aboulela, A. Ibrahim, and R. Kashef, “A survey on augmenting knowledge graphs (kgs) with large language models (llms): models, evaluation metrics, benchmarks, and challenges,” Discover Artificial Intelligence, vol. 4, no. 1, p. 76, 2024.

[11] H. Song et al., “R1-searcher: Incentivizing the search capability in llms via reinforcement learning,” arXiv preprint, 2025.

[12] B. Jin et al., “Search-r1: Training llms to reason and leverage search engines with reinforcement learning,” arXiv preprint, 2025.

[13] J. Wu et al., “Mmsearch-r1: Incentivizing lmms to search,” arXiv preprint, 2025.

[14] H. Wang et al., “Acting less is reasoning more! teaching model to act efficiently,” arXiv preprint, 2025.

[15] X. Li, H. Zou, and P. Liu, “Torl: Scaling tool-integrated rl,” arXiv preprint, 2025.

[16] J. Singh et al., “Agentic reasoning and tool integration for llms via reinforcement learning,” arXiv preprint, 2025.

[17] H. Sun et al., “Zerosearch: Incentivize the search capability of llms without searching,” arXiv preprint, 2025.

[18] H. Pan et al., “Compiler-r1: Towards agentic compiler auto-tuning with reinforcement learning,” arXiv preprint, 2025.

[19] S. Wang and Y. Yu, “Kg-hopper: Empowering compact open llms with knowledge graph reasoning via reinforcement learning,” arXiv preprint arXiv:2603.21440, 2026.

[20] H. Dong, K. Duan, and C. Zhang, “Enhancing decision-making of large language models via actor-critic,” arXiv preprint, 2025.

[21] Z. Zhang and W. Zhao, “A collaborative reasoning framework powered by reinforcement learning and large language models for complex questions answering over knowledge graph,” in Proceedings of the 31st International Conference on Computational Linguistics, 2025, pp. 10 672– 10 684.

[22] S. Wang, “Enhancing in-context learning of large language models for knowledge graph reasoning via rule-and-reinforce selected triples,” Ap plied Sciences, vol. 15, no. 3, p. 1088, 2025.

[23] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[24] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 2019, pp. 3982–3992.

[25] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[26] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[27] M. Menschikov et al., “Personalai: A systematic comparison of knowledge graph storage and retrieval approaches for personalized llm agents,” arXiv preprint, 2025.

[28] Mistral AI, “Mistral small 3.1,” 2025, accessed: 2026-03-20. [Online]. Available: https://docs.mistral.ai/models/mistral-small-3-1-25-03

[29] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of ACL, 2002, pp. 311–318.

[30] C.-Y. Lin, “Rouge: A package for automatic evaluation of summaries,” in Text Summarization Branches Out, 2004, pp. 74–81.

[31] M. Denkowski and A. Lavie, “Meteor universal: Language specific trans lation evaluation for any target language,” in Proceedings of the Ninth Workshop on Statistical Machine Translation, 2014, pp. 376–380.

[32] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging LLM-as-a-judge with MT-bench and chatbot arena,” in Advances in Neu ral Information Processing Systems, vol. 36, 2023, pp. 46 595–46 623.

[33] Mistral AI, “Mistral medium 3,” 2025, accessed: 2026-03-20. [Online]. Available: https://docs.mistral.ai/models/mistral-medium-3-25-05

[34] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop question answering,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, pp. 2369–2380.

[35] M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer, “Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension,” in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2017, pp. 1601–1611.

[36] B. Li et al., “Diaasq: A benchmark of conversational aspect-based sen timent quadruple analysis,” in Findings of the Association for Computa tional Linguistics: ACL 2023, 2023, pp. 13 449–13 467.

## APPENDIX A TRAINING DETAILS

## A.1. LEARNING CURVES OF RANGER

Figure 6 shows the learning curves of RANGER trained with Mistral Small 3.2 (24B) as the LLM mentor on the HotpotQA, TriviaQA, and DiaASQ graphs.

As shown in the figure, after training for the $1 0 ^ { 7 }$ graph transitions, the average reward assigned to the RL agent by the LLM mentor does not exceed 0.2 for all graphs. In other words, according to the mentor, the agent finds paths that logically support the target fact in no more than about 20% of episodes. Assuming that the RL agent explores the graph effectively, this suggests that the graphs themselves are sparse in terms of redundant supporting evidence: not every edge is accompanied by alternative paths or evidence that can confirm the corresponding relation.

The learning curves also differ across datasets. In particular, the curve for DiaASQ remains consistently below the others. One possible explanation is that the graphs are constructed automatically, so they may contain noisy, incomplete, or contradictory relations, which can limit the reward achievable by the agent.

![](images/e7366d7d9aad8d25fd777f20b1ee6d21f5245948744b3169f185646fc75858fb.jpg)  
FIGURE 6: Learning curves of RANGER across HotpotQA, DiaASQ, and TriviaQA. The episode return corresponds to the reward assigned by the LLM mentor.

## A.2. LLM MENTOR REQUIREMENTS STUDY

The learning curves of RANGER on HotpotQA with three different mentor LLMs are shown in Figure 7.

As can be seen, the average rewards assigned by the mentor during training are highest for Ministral 3 (8B) and lowest for Mistral Medium 3.5 (128B). However, the downstream answer quality shows the opposite trend and increases with the size of the mentor model. This suggests that smaller models may have a weaker understanding of the context and therefore more often assign false positive rewards to retrieved paths.

![](images/5850649bd7a51b0916b3d090158b1012ae38b6b135d8d610c17e0ef79da23746.jpg)  
FIGURE 7: Learning curves of RANGER on HotpotQA with Ministral 3 (8B), Mistral Small 3.2 (24B), and Mistral Medium 3.5 (128B) as LLM mentors. The episode return corresponds to the reward assigned by the LLM mentor.

## APPENDIX B SYSTEM PROMPT FOR THE LLM MENTOR

The LLM mentor is used to estimate whether the path collected by the RL agent contains sufficient evidence for recovering the target information. The system prompt used for the mentor LLM is shown in Figure 8.

![](images/0fa895ae0e4d1dc5b991bcf04e111eb039e4c559ca9b22aba3298476d2175fd9.jpg)  
FIGURE 8: System prompt used by the LLM mentor to assign the retrieval reward.