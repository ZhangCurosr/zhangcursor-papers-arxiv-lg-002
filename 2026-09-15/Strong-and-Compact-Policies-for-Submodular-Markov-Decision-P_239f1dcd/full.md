# Strong and Compact Policies for Submodular Markov Decision Processes via LP-Based Submodular Orienteering

Lars Rohwedder<sup>∗</sup>

Rico Zenklusen<sup>†</sup>

## Abstract

Finding policies for Markov Decision Processes (MDPs) is a central problem in areas such as Reinforcement Learning and Operations Research. Here, we have to repeatedly choose an action that should be performed by an agent. Depending on the action and the current state of the agent, the agent collects a reward and randomly transitions into a new state. The goal is to maximize the reward in expectation over a finite time horizon of length H. We consider a recently introduced variant that generalizes the traditionally additive reward function in the model to a monotone submodular one, which allows for capturing a range of interesting applications.

Without the stochastic component, this problem is equivalent to the Submodular Orienteering problem, where the goal is to find an s-t walk in a directed graph maximizing a monotone submodular function under a length constraint. We present a novel LP-based algorithm for Submodular Orienteering using ideas from the Sherali-Adams hierarchy and Round-or-Cut. Our guarantees are comparable to the known quasi-polynomial time logarithmic approximation for Submodular Orienteering, but also extend to the setting of Submodular Markov Decision Processes. In the polynomial time regime, we present an $O ( n ^ { \varepsilon } )$ -approximation (and O(H<sup>ε</sup>) for Submodular MDPs) for every $\varepsilon > 0$ , where n is the number of vertices, which was unknown even for Submodular Orienteering. Prior to our work, the best known approximation guarantee for Submodular MDPs had an approximation ratio linear in H.

Beyond these algorithmic results, our methods reveal a trade-of between the approximation guarantee and the number of previously visited vertices on which an agent conditions its decision.

## 1 Introduction

The Submodular Orienteering problem asks to maximize a non-negative monotone submodular function over the vertices of a walk in a directed graph $G = ( V , A )$ from a vertex s to a vertex $t . ^ { 1 }$ Additionally, the length of the walk may be constrained via a length function $w : A \to \mathbb { Z } _ { \geq 0 }$ which should not exceed a given budget C when summed over all edges in the walk. We denote by f(W) and w(W), respectively, the submodular objective value and the length of the walk W. The Recursive Greedy algorithm by Chekuri and Pál [CP05] is an elegant combinatorial algorithm that achieves an optimal (under natural complexity assumptions) logarithmic approximation for it in quasi-polynomial time.

In this paper we first develop an LP-based randomized rounding that recovers the same guarantees (in expectation) up to constant factors as Chekuri and Pál [CP05] and an extension that gives a polynomial time $O ( n ^ { \varepsilon } )$ -approximation for any $\varepsilon > 0$

Theorem 1. There is an LP-based approximation algorithm for Submodular Orienteering that runs in time $n ^ { O ( \log n ) } \langle w \rangle ^ { O ( 1 ) }$ and outputs an s-t walk W with $\mathbb { E } [ f ( W ) ] \ge \Omega ( \frac { \mathrm { O P T } } { \log n } )$ and $\mathbb { E } [ w ( W ) ] ~ \le$ C. Alternatively, it can output in time $n ^ { O ( 1 / \varepsilon ) } \langle w \rangle ^ { O ( 1 ) }$ an s-t walk with $\mathbb { E } [ f ( W ) ] \geq \Omega ( \frac { \mathrm { O P T } } { n ^ { \varepsilon } } )$ and $\mathbb { E } [ w ( W ) ] \leq C$ for $a n y \varepsilon > 0 . ^ { 2 }$ Here, ⟨w⟩ is the encoding size of the function values $o f w$

The new trade-of in the polynomial time regime can also be obtained by a purely combinatorial algorithm, which we include in Appendix A.

Our algorithm difers significantly from previous LP-based algorithms for problems involving submodular maximization, which usually rely on the multilinear extension. (See, in particular, [Von08; CCPV11; CVZ10; CVZ14; FNS11; EN16; BF19; BF24] and references therein.) In our problem, the multilinear extension is extremely weak.<sup>3</sup> Our relaxation is based on a Sherali-Adams type extended formulation. The relaxation and the randomized rounding algorithm are closely related to techniques known from the Directed Steiner Tree problem, the Robust Shortest Path problem and others, see, $\mathrm { e . g . }$ , [GLL22; LXZ24; BLR26], which, however, do not involve submodular functions. We manage to integrate submodular functions with a Round-or-Cut approach. Round-or-Cut, first introduced in [CFLP00], is a framework where one starts with a fractional solution to a potentially weak LP relaxation and then either obtains a good solution by rounding it or derives new cutting planes that can be added to the relaxation. This approach only needs to generate cutting planes when rounding fails, which allows for using properties of the failed rounding approach to generate cutting planes. For our case, we proceed as follows. Along with the rounded solution, our algorithm outputs a linear upper bound on the submodular function such that the rounded solution has an approximation guarantee with respect to the linear function’s value in the current fractional point. If the submodular value of the rounded solution is not high enough, the linear upper bound can be used for a cutting plane that we can add to the relaxation and repeat.

By running the algorithm on LP solutions that satisfy additional properties, we can influence the output distribution. While the LP-based approach can be of independent interest, we emphasize a specific application in Markov Decision Processes that nicely highlights advantages of our approach and leads to a significant improvement compared to prior work.

## 1.1 Markov Decision Processes

The Markov Decision Process (MDP) is a fundamental model to describe sequential decision making under uncertainty. It is arguably the central mathematical framework for Reinforcement Learning, which is a subfield of Machine Learning that has seen tremendous success in recent years, and it is also a classical model in Operations Research (see, e.g., [Put90; WV12]). An MDP has a finite set of states S and actions A. A common variant that we focus on here assumes a finite time horizon $\{ 1 , 2 , \ldots , H \}$ . At each time $t \in \{ 1 , 2 , \ldots , H \}$ , an agent is in some state $s _ { t } \in S$ The agent then selects an action $a _ { t } \in A$ according to a policy, which is the object we aim to optimize. Based on the current state $s _ { t }$ and the action $a _ { t }$ chosen by the agent, the environment randomly selects the next state $s _ { t + 1 }$ . The distribution of $s _ { t + 1 }$ is completely determined by $s _ { t }$ and $a _ { t }$ and it is known a priori to the agent via the probabilities

$$
\begin{array} { r } { { \mathbb { P } } [ s _ { t + 1 } = s ^ { \prime } \mid s _ { t } = s , \ a _ { t } = a ] , \quad \mathrm { f o r } \ t \in \{ 1 , 2 , \ldots , H - 1 \} , s ^ { \prime } \in S , s \in S , a \in A . } \end{array}\tag{1}
$$

<sup>2</sup>We state all running times as the number of operations in the word RAM model, where a word is large enough to store any input value, including the values returned from function $f .$

<sup>3</sup>Consider the graph consisting of $k = \Theta ( { \sqrt { n } } )$ many s-t paths $P _ { 1 } , \ldots , P _ { k }$ of length $k + 2$ each that are disjoint except for s and $t .$ The vertices of $P _ { i } \setminus \{ s , t \}$ are colored with color i. Function $f$ counts the number of diferent colors. Thus, the optimal solution is 1, but the multilinear extension on the symmetric solution that takes each path with weight $1 / k$ has a value of $\Omega ( k ) = \Omega ( \sqrt { n } )$ . The symmetric solution is a convex combination of integral solutions, hence, using the multilinear extension with any LP relaxation that is oblivious to $f$ seems hopeless.

Similarly, the initial state $s _ { 1 }$ is sampled with explicitly given probabilities $\mathbb { P } [ s _ { 1 } = s ^ { \prime } ]$ for $s ^ { \prime } \in S .$ . In the classical model, the agent collects rewards for each state and action and wants to maximize their sum in expectation. We consider a more general model, where the total reward is $f ( \{ s _ { 1 } , a _ { 1 } , . . . , s _ { H } , a _ { H } \} )$ for a monotone submodular function $f \colon 2 ^ { S \cup A }  \mathbb { R } _ { \geq 0 }$ We devise a policy that selects an action based on the trajectory $s _ { 1 } , a _ { 1 } , s _ { 2 } , a _ { 2 } , \ldots , s _ { t } .$ that is, all previous states and actions. Note that in the classical additive case, where rewards are additive instead of submodular, there is an optimal Markovian policy, i.e., a policy that solely depends on the current state and time. It can be found eficiently by dynamic programming using Bellman’s equation. This is not the case in our setting, as for example pointed out in [PMZK24]. It was also observed in [PMZK24] that the deterministic version, that is, each probability in (1) and those of initial states are either zero or one, corresponds exactly to the Submodular Orienteering problem.

Our model was recently introduced and studied by Wang, Zhang, Chaplot, Garagić, and Salakhutdinov [Wan+20] and Prajapat, Mutný, Zeilinger, and Krause [PMZK24]. It is motivated by a wide range of applications from Reinforcement Learning to planning, including navigation, biodiversity monitoring, Bayesian experimental design, and coverage maximization. We stick to the terminology used in [PMZK24] and call this model Submodular Markov Decision Process (Submodular MDP). Note that the dynamics of a Submodular MDP remain Markovian, but the reward function is not Markovian anymore. Such processes are sometimes also referred to as non-Markovian Reward Decision Processes $( N M R D P s )$ in the literature.

The best approximation guarantee for Submodular MDP without additional assumptions is $O ( H )$ from [Wan+20]. De Santi, Prajapat, and Krause [DPK24] give other non-trivial approximations assuming the submodular function has bounded curvature, a parameter that measures how much the marginal value of an element can decrease and captures how far the function is from being additive. The works above also contain empirical results and results for simplified cases.

We improve the approximation ratio from $O ( H )$ to O(log H) and provide a trade-of between approximation ratio and running time.

Theorem 2. Submodular Markov Decision Processes admit a randomized O(log H)-approximate policy that can be computed in $n ^ { O ( \log H ) } \langle p \rangle ^ { O ( 1 ) }$ time. Furthermore, for any $\varepsilon > 0$ , there is a randomized $O ( H ^ { \varepsilon } )$ -approximate policy that can be computed in $n ^ { O ( 1 / \varepsilon ) } \langle p \rangle ^ { O ( 1 ) }$ time. Each decision in the former policy depends only on O(log H) many vertices from the trajectory; in the latter on $O ( 1 / \varepsilon )$ vertices. Here, $n = \mathrm { p o l y } ( | S | , | A | , H )$ and $\langle p \rangle$ is the encoding size of the transition probabilities.

Roughly speaking, we use our LP-based algorithm for Submodular Orienteering on an LP with constraints that force certain edges to be taken with the transition probabilities from the input. It follows from properties of our LP rounding that the distribution over walks that we obtain can be reformulated as a randomized policy together with the transition distributions.

This result falls into the area of adaptive algorithms in stochastic combinatorial optimization. The adaptivity comes from the conditioning operation in Sherali-Adams on previous events that the LP rounding performs. Using linear programs to derive policies is common in stochastic optimization, see, e.g., [DGV08; GMS10], but we are not aware of previous examples using conditioning in Sherali-Adams or another hierarchy in a similar way.

The complexity of representing history-dependent policies is a recurring issue in sequential decision making. For MDPs with non-additive rewards, augmenting the state with relevant history can substantially enlarge the state space, and an explicit policy representation may require exponential space [BBG96; Thi+06]. For submodular MDPs, Wang, Zhang, Chaplot, Garagić, and Salakhutdinov [Wan+20] already obtain an $O ( H )$ -approximation with a polynomial-size policy representation, while Prajapat, Mutný, Zeilinger, and Krause [PMZK24] explicitly discuss the role of history dependence and the dificulty of representing general policies. Our result achieves an

O(log H)-approximation with a quasi-polynomial-size policy representation, whose decisions depend on only O(log H) selected vertices of the past trajectory. Prior to our work, it was unclear whether an o(H)-approximation could be achieved with a policy that has a short description.

## 1.2 Other related works

Our Sherali-Adams approach and randomized rounding are inspired by algorithmic ideas appearing in a number of previous works on variants of the Directed Steiner Tree problem [GLL22; Guo+22], the Robust Shortest Path problem [LXZ24], the Santa Claus problem [BCG09; CCK09], as well as others. Recently, Bamas, Li, and Rohwedder [BLR26] gave a general framework capturing many of these examples. The most obvious diference is that our algorithm extends these techniques by handling monotone submodular functions. Previously this was known only for very specific submodular functions, namely, some forms of coverage functions (capturing for example the number of diferent colors when elements are partitioned into color classes). It is shown for example in [BLR26] how to obtain via an LP approach a similar trade-of between running time and approximation as in our result for Orienteering when maximizing the number of colors in the solution. This would not directly lead to a policy for MDPs, even with these restrictive functions. This is because of other subtle diferences in the randomized rounding. To apply the algorithm to MDPs, it is important that when making a decision for one vertex or edge, we only condition on previous vertices and not future vertices, so that the resulting policy does not depend on the future. The randomized rounding we present in this paper has such a property, while [BLR26; LXZ24], the closest related algorithms, condition on the middle point of the walk and then recurse in the first half and the second half.

Many stochastic routing problems have been considered in the literature. Apart from the previously mentioned works on Submodular MDPs [Wan+20; PMZK24; DPK24], we are not aware of any works that have direct implications for our setting, so we only give a few representative examples here: Jiang, Li, Liu, and Singla [JLLS20] consider the problem of collecting a high reward by a tour of low cost. In their model, either the reward or the cost is stochastic and both are additive. This is substantially simpler than our setting where both rewards and the movement of the agent are stochastic. Tan, Ghuge, and Nagarajan [TGN24] consider a model where the agent moves in the network and makes stochastic observations at each vertex. The agent continues until a monotone submodular function over the observations is large enough. One can again view this as stochastic rewards without stochastic movement. We note that [TGN24] also use Submodular Orienteering as a subroutine for their policy, but in the details, their approach difers significantly from ours.

## 2 LP-Based Orienteering

For clarity, we present our main technical result on a simple variant of Submodular Orienteering, to which we later reduce: We assume that the graph is layered with layers $V _ { 1 } \cup \cdots \cup V _ { H }$ , where all arcs go from one layer to the next. Note that since this is a DAG, every walk is a path. We further assume that $H = d ^ { r }$ for suitable parameters d and r that we will specify later. Our goal is to find a path P that is layer-spanning: $P$ starts at any vertex in $V _ { 1 }$ and ends at any vertex in $V _ { H }$ . For now we do not consider restrictions on the length like the ones given by w and C earlier.

As discussed, a key novelty in our approach is that it is LP-based. The linear inequality description we use can naturally be interpreted as a strengthening of the following basic linear description through an extended formulation:

$$
x _ { u } + x _ { v } \leq 1 \qquad \forall ( u , v ) \in ( V _ { i } \times V _ { i + 1 } ) \setminus A { \mathrm { ~ f o r ~ s o m e ~ } } i \in \{ 1 , \ldots , H - 1 \}
$$

$$
\begin{array} { r l } { \displaystyle \sum _ { v \in V _ { i } } x _ { v } = 1 \quad } & { { } \forall i \in \{ 1 , \dots , H \} } \\ { x _ { v } \leq 1 \quad } & { { } \forall v \in V } \\ { x _ { v } \geq 0 \quad } & { { } \forall v \in V . } \end{array}\tag{Q}
$$

We denote by Q the polytope defined by the basic linear description above. The variable $x _ { v }$ indicates whether vertex v is in the path or not. For a {0, 1}-solution, the first constraint ensures that for any two consecutive vertices u and v in the path, there is an arc between them. The second constraint ensures that exactly one vertex is chosen from each layer.

In a fractional solution, we think of $x _ { v }$ as the probability that v is in a random path $P . ^ { 4 }$ We would like to extend and strengthen this relaxation by including some additional information on the correlation between variables in the linear description, specifically, on how likely a set of vertices is to be contained in P simultaneously. Note that without a strengthening the basic relaxation described above can be extremely weak. In particular, it can be feasible even if the graph does not contain any layer-spanning path.<sup>5</sup> There are simpler ways to describe the convex hull of layer-spanning paths by a compact linear programming formulation, but the specific variables we introduce will give us more flexibility in the rounding algorithm and to express more complicated linear constraints that we will add later.

Towards this, we write an extended formulation parameterized by some $r \in \mathbb { Z } _ { \geq 0 }$ . Our variables are $x _ { I }$ for all $I \subseteq V , | I | \leq r + 1$ . Variable $x _ { I }$ should mimic the behavior of $\prod _ { i \in I } x _ { i } \ \mathrm { o r }$ , in other words, $x _ { I }$ intuitively describes the probability that $I \subseteq P$ . Readers familiar with the Sherali-Adams hierarchy, see [SA90], will recognize the similarity with this construction and, indeed, our construction can be seen as a simplified version of the Sherali-Adams hierarchy. For self-containedness and clarity, we construct the relaxation here from first principles.

For each $I \subseteq V$ with $\left. I \right. \leq r ,$ and each constraint of the original LP, we multiply each side of each constraint by $\textstyle \prod _ { i \in I } x _ { i }$ . For example, assume that $u , v , w \in V$ are all distinct. Let $I = \{ u , w \}$ and consider the constraint $x _ { u } + x _ { v } \le 1$ . The new constraint is then $x _ { u } \cdot x _ { w } \cdot x _ { u } + x _ { u } \cdot x _ { w } \cdot x _ { v } \leq x _ { u } \cdot x _ { w } .$ This constraint is clearly satisfied if the original one was. Since binary variables $y \in \{ 0 , 1 \}$ satisfy $y ^ { 2 } = y$ , we can remove duplicate variables in monomials. The previous example then simplifies to $x _ { u } \cdot x _ { w } + x _ { u } \cdot x _ { w } \cdot x _ { v } \leq x _ { u } \cdot x _ { w }$ . This maintains feasibility if x is binary. Now we replace products of variables by our new variables. In the example, we obtain $x _ { \{ u , w \} } + x _ { \{ w , v , u \} } \leq x _ { \{ u , w \} }$ . Finally, we add the constraint $x _ { \emptyset } = 1$ , which corresponds to requiring the empty product to equal 1.

We obtain the following linear description, which we denote by $Q _ { G } ^ { ( r ) }$ , or simply by $Q ^ { ( r ) }$ if G is clear from context.

$$
x _ { I \cup \{ u \} } + x _ { I \cup \{ v \} } \leq x _ { I } \qquad \forall ( u , v ) \in ( V _ { i } \times V _ { i + 1 } ) \setminus A { \mathrm { ~ f o r ~ s o m e ~ } } i \in \{ 1 , \ldots , H - 1 \} ,\tag{2}
$$

$$
\forall I \subseteq V , | I | \leq r
$$

$$
\sum _ { v \in V _ { i } } x _ { I \cup \{ v \} } = x _ { I } \qquad \forall i \in \{ 1 , \dots , H \} , \ \forall I \subseteq V , | I | \leq r\tag{Q<sup>(r)</sup>}
$$

$$
^ { \iota } x _ { I \cup \{ v \} } \leq x _ { I } \qquad \forall v \in V \forall I \subseteq V , | I | \leq r\tag{3}
$$

$$
\begin{array} { r l } { x _ { \emptyset } = 1 } \\ { x _ { I } \geq 0 } \end{array} \quad \quad \forall I \subseteq V , | I | \leq r + 1
$$

For simplicity of notation, we write $x _ { v } = x _ { \{ v \} } , x _ { u , v } = x _ { \{ u , v \} }$ , etc.

Consider a layer-spanning path or, in other words, a solution P. Clearly, the incidence vector $\mathbf { 1 } _ { P } \in \{ 0 , 1 \} ^ { V }$ of the vertices in P is feasible for the basic linear description $Q = Q ^ { ( 0 ) }$ . For $\pmb { x } \in [ 0 , 1 ] ^ { n }$ 2 let $\pmb { x } ^ { ( r ) } \in [ 0 , 1 ] ^ { \binom { V } { \leq r + 1 } }$ be defined by $\begin{array} { r } { x _ { I } ^ { ( r ) } = \prod _ { i \in I } x _ { i } } \end{array}$ for all $I \subseteq V , | I | \leq r + 1 . ^ { 6 }$ For integral solutions $\pmb { x } \in Q$ of the basic linear description Q we have $\pmb { x } ^ { ( r ) } \in Q ^ { ( r ) }$ by the previous discussion.

A crucial feature of this extended formulation is that it allows us to condition on variables. For some $\pmb { x } \in Q ^ { ( r ) }$ and $v \in V$ with $x _ { v } > 0$ , define $\pmb { x } ^ { | v } \in [ 0 , 1 ] ^ { \binom { V } { \leq r } }$ by

$$
x _ { I } ^ { | v } : = \frac { x _ { I \cup \{ v \} } } { x _ { v } } , \quad I \subseteq V , | I | \leq r .
$$

Note that $x _ { v } ^ { | v } = 1$ and $\begin{array} { r } { x _ { u } ^ { | v } = x _ { v } ^ { | u } \cdot \frac { x _ { u } } { x _ { v } } } \end{array}$ (analogous to Bayes’ rule), which matches our intuition that $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } } | \boldsymbol { v }$ corresponds to conditioning the probability distribution on including v. The following lemma shows that conditioning a point $\pmb { x } \in Q ^ { ( r ) }$ on one variable leads to a point that is still feasible for the extended formulation, but with one level less, i.e., a point in $Q ^ { ( r - 1 ) }$

Lemma 3. Let $r \in \mathbb { Z } _ { \geq 1 } , \pmb { x } \in Q ^ { ( r ) }$ , and $w \in V$ with $x _ { w } > 0$ . Then ${ \pmb x } ^ { | w } \in Q ^ { ( r - 1 ) }$

Proof. We verify this only for constraints of type (2), but the same argument works for all other types of constraints as well. Let $i \in \{ 1 , \ldots , H - 1 \} , ( u , v ) \in ( V _ { i } \times V _ { i + 1 } ) \setminus A$ , and $I \subseteq V$ with $\left| I \right| \leq r - 1$ . Because ${ \pmb x } \in Q ^ { ( r ) }$ and it satisfies (2) with $I ^ { \prime } = I \cup \{ w \}$ , we have

$$
x _ { I \cup \{ w , u \} } + x _ { I \cup \{ w , v \} } \leq x _ { I \cup \{ w \} } .
$$

Multiplying both sides by $1 / x _ { w }$ , we obtain

$$
x _ { I \cup \{ u \} } ^ { | w } + x _ { I \cup \{ v \} } ^ { | w } = \frac { x _ { I \cup \{ w , u \} } } { x _ { w } } + \frac { x _ { I \cup \{ w , v \} } } { x _ { w } } \leq \frac { x _ { I \cup \{ w \} } } { x _ { w } } = x _ { I } ^ { | w } .
$$

Therefore, $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } } | w$ satisfies (2).

Based on this conditioning property, we now define a natural recursive randomized rounding algorithm. The algorithm has a parameter $d \in \mathbb { Z } _ { \geq 2 }$ that controls the degree of the recursion. For lower values of d we obtain a better approximation ratio at the cost of a higher running time. We assume from here on that H is a power of $d ,$ that is, $H = d ^ { r }$ for some $r \in \mathbb { Z } _ { > 0 }$ . This value of r will be the parameter that we use in the linear program to define $Q ^ { ( r ) }$ . Given a point ${ \pmb x } \in Q ^ { ( r ) }$ , the rounding algorithm returns a random layer-spanning path P whose distribution has useful properties linked to x.

This approach will allow us to later add additional constraints to $Q ^ { ( r ) }$ that are satisfied by $\mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) }$ for the optimal solution $\mathrm { O P T }$ . These constraints will lead to points x with additional properties that translate into distributional properties of P that are crucial for our applications. Moreover, as we see later, the LP-based approach also allows us to add constraints through a round-or-cut procedure for obtaining good approximation guarantees for submodular maximization.

## 2.1 Recursive randomized rounding

Let $G = ( V , A )$ be a layered graph with $V = V _ { 1 } \cup \cdot \cdot \cdot \cup V _ { H } , H = d ^ { r }$ , and let $\pmb { x } \in Q ^ { ( r ) }$ . We now describe our recursive randomized rounding algorithm to round a point ${ \pmb x } \in Q ^ { ( r ) }$ to a random layerspanning path P. To present the algorithm, we use the following notation. For some set of layer indices $L \subseteq \{ 1 , 2 , \ldots , H \}$ , we let $G [ L ]$ be the induced subgraph on $V [ L ] : = \textstyle \bigcup _ { i \in L } V _ { i }$ . Moreover, we denote by ${ \pmb x } [ L ]$ the vector x restricted to variables of the vertex sets fully contained in $G [ L ]$ . We will consider only layer indices L corresponding to consecutive sets of layers. Our recursive randomized rounding algorithm is described in Algorithm 1.

Algorithm 1: Recursive randomized rounding (RRR)   
Input: Layered orienteering graph $G = ( V , A )$ with $V = V _ { 1 } \cup \dots \cup V _ { H } , \pmb { x } \in Q ^ { ( r ) } , d ^ { r } = H .$   
Output: Layer-spanning path in G.   
1 if $H = 1$ then   
2 Sample $v \in V _ { 1 }$ with $\mathbb { P } [ v = w ] = x _ { w }$ for all $w \in V _ { 1 }$   
3 return (v).   
4 else   
5 Let $\begin{array} { r } { L _ { i } = \{ ( i - 1 ) \frac { H } { d } + 1 , ( i - 1 ) \frac { H } { d } + 2 , \dots , i \frac { H } { d } \} } \end{array}$ for each $i \in \{ 1 , 2 , \ldots , d \}$   
6 Recursively construct layer-spanning path $P _ { 1 }$ in $G [ L _ { 1 } ]$ from $\pmb { x } [ L _ { 1 } ]$ projected to $\mathbb { R } ^ { \left( \boldsymbol { \underline { V } } \right) }$   
7 for $i = 2 , 3 , \ldots , d$ do   
8 Let u be the last vertex in $P _ { i - 1 }$   
9 Recursively construct layer-spanning path $P _ { i }$ in $G [ L _ { i } ]$ from $\pmb { x } ^ { | u } [ L _ { i } ]$   
10 return concatenation $P _ { 1 } \circ P _ { 2 } \circ \cdots \circ P _ { d }$

Given a point ${ \pmb x } \in Q ^ { ( r ) }$ , denote by RRR(x) the output distribution of Algorithm 1 on x.

Our recursive construction of a walk is somewhat reminiscent of the Recursive Greedy algorithm by Chekuri and Pál [CP05], but there are some important diferences. Apart from the obvious diference that we are LP-based, our algorithm, when recursively constructing parts of the path, never needs to condition on vertices that lie in the future, i.e., that come later in the path. (We will make this precise in Section 3.) This is crucial for later obtaining policies for MDPs, which are not allowed to depend on future states. Conversely, the Recursive Greedy algorithm needs to condition on vertices that lie in the future. For example, the first step is to guess the middle vertex of the path, which is a vertex that lies in the future for the first half of the path. And this step is repeated recursively in the algorithm of Chekuri and Pál [CP05].

We also want to highlight that our parameter d chops the path into d pieces, which are then constructed recursively and in particular sequentially. This is diferent from guessing d vertices of the path simultaneously, which would lead to both a much higher running time and a significant dependence on future vertices. This technique of guessing several vertices simultaneously was discussed in the Recursive Greedy algorithm by Chekuri and Pál [CP05] (see Section 3.3), to get slightly improved approximation ratios.

We first observe that the diferent steps of Algorithm 1 are well-defined and indeed lead to a layer-spanning path being sampled. We also prove that RRR(x) is marginal-preserving on singleton variables.

Lemma 4. Consider a layered graph $G = ( V , A )$ with $V = V _ { 1 } \cup \cdot \cdot \cdot \cup V _ { H } , H = d ^ { r }$ , and let ${ \pmb x } \in Q ^ { ( r ) }$ Then recursive randomized rounding applied to x

(i) is well defined;

(ii) returns a layer-spanning path; and

(iii) the output path P satisfies $\mathbb P [ v \in P ] = x _ { v } \ f o r \ a l l \ v \in V$

Proof. We show the statement by induction over H. If $H = 1$ , then by (3) we have $\textstyle \sum _ { w \in V _ { 1 } } x _ { w } =$ $x _ { \emptyset } = 1$ . Thus, the variables $\{ x _ { w } \} _ { w \in V _ { 1 } }$ indeed define a probability distribution for v. The output is a single vertex, which is layer-spanning in this case, and it satisfies (iii) by definition.

Now assume that $H \geq d$ and let $L _ { i }$ be defined as in the algorithm. To be able to make the first recursive call, we need to verify that ${ \pmb x } [ L _ { 1 } ]$ projected to $\mathbb { R } ^ { \binom { V _ { 1 } } { \leq r } }$ is in $Q _ { G [ L _ { 1 } ] } ^ { ( r - 1 ) }$ . This holds trivially, because all constraints in $Q _ { G [ L _ { 1 } ] } ^ { ( r - 1 ) }$ are also valid for $Q _ { G } ^ { ( r ) }$ . By the induction hypothesis, a vertex $v \in V [ L _ { 1 } ]$ is in $P _ { 1 }$ and therefore in P with probability x $\mathbf { \nabla } [ L _ { 1 } ] _ { v } = x _ { v }$ . Thus, (iii) holds for such vertices.

Next, assume that $v \in V [ L _ { i } ]$ for some $i ~ \geq ~ 2$ . The path returned by recursive randomized rounding in G is $P = P _ { 1 } \circ P _ { 2 } \circ \cdots \circ P _ { d }$ We can assume (via a nested induction) that each $w \in V _ { ( i - 1 ) H / d }$ is the last vertex of $P _ { i - 1 }$ with probability $\mathbb { P } [ w = u ] = x _ { w }$ and, in particular, $x _ { u } > 0$ if u is the last vertex of $P _ { i - 1 }$ . By Lemma 3, we have $\pmb { x } ^ { | u } [ L _ { i } ] \in Q _ { G [ L _ { i } ] } ^ { ( r - 1 ) }$ so the ith recursive call returning path $P _ { i }$ is also well-defined. Furthermore,

$$
\begin{array} { r l } & { \mathbb { P } [ v \in P ] = \mathbb { P } [ v \in P _ { i } ] = \displaystyle \sum _ { w \in V _ { ( i - 1 ) ^ { H } / d } } \mathbb { P } [ w = u ] \cdot \mathbb { P } [ v \in P _ { i } \mid w = u ] } \\ & { \quad \quad \quad = \displaystyle \sum _ { w \in V _ { ( i - 1 ) ^ { H } / d } } x _ { w } \cdot \mathbb { P } [ v \in P _ { i } \mid w = u ] } \\ & { \quad \quad \quad = \displaystyle \sum _ { w \in V _ { ( i - 1 ) ^ { H } / d } } x _ { w } \cdot x _ { v } ^ { \mid w } = \displaystyle \sum _ { w \in V _ { ( i - 1 ) ^ { H } / d } } x _ { w , v } = x _ { v } , } \end{array}
$$

where the last equality follows from (3). Note that for all $v \in V _ { 1 + ( i - 1 ) H / d }$ with $( u , v ) \not \in A$ , we have from (2) that $x _ { u , v } = 0$ and, in particular, $x _ { v } ^ { | u } = 0$ . Thus, by the induction hypothesis, we have that $P _ { i }$ cannot contain such a vertex and,

$P _ { 1 } , \ldots , P _ { d }$ are all paths.

By the former, there is an arc from the last vertex of $P _ { i - 1 }$ to the first vertex of $P _ { i }$ , which, together with the latter, implies that P is indeed a path from $V _ { 1 }$ to $V _ { H }$ □

As discussed, our main goal is to use an LP-based approach to deal with problems that one can reduce to Submodular Orienteering with constraints. Whereas the recursive randomized rounding algorithm described in Algorithm 1 shows how we can round a point $\pmb { x } \in Q ^ { ( r ) }$ to a random layer spanning path P, we still lack a way to make sure that P has a large submodular value. We address this next.

## 2.2 Approximating submodular objectives by linear functions

Our key ingredient to deal with submodular objectives is the following lemma. It shows that there is a randomized procedure that not only returns a random path $P \sim \tt R R R ( \pmb { x } )$ , but also returns a linear function ℓ that upper bounds f on all layer-spanning paths and such that the expected value of $f ( P )$ is related to $\mathbb { E } [ \ell ( { \pmb x } ) ]$ . It is important to note that both P and ℓ are random objects that depend on the random choices made by the algorithm, and that the expectation is taken over these random choices.

For $I \subseteq V$ with $\mathbb { P } _ { P \sim \mathtt { R R R } ( \pmb { x } ) } [ I \subseteq P ] > 0$ , let $\mathtt { R R R } ^ { | I } ( { \pmb x } )$ be the distribution $\mathtt { R R R } ( { \pmb x } )$ conditioned on I being contained in the path.

Lemma 5. Let $d \in \mathbb { Z } _ { \geq 2 }$ and $r \in \mathbb { Z } _ { \geq 0 }$ so that $H = d ^ { r }$ . Let ${ \pmb x } \in Q ^ { ( r ) }$ and $f : 2 ^ { V } \to { \mathbb { R } } _ { \geq 0 }$ be monotone submodular. There is a randomized $\therefore \boldsymbol { n } ^ { O ( r ) }$ time algorithm returning one random path $\begin{array} { r } { P _ { t } \sim \mathrm { R R R } ^ { | t } ( \pmb { x } ) } \end{array}$ for each $t \in V _ { H }$ with $x _ { t } > 0$ , and in addition a linear function $\ell : \mathbb { R } ^ { ( \underset { \leq r + 1 } { V } ) }  \mathbb { R }$ such that

• The random path ${ \overline { { P } } } ,$ which we set equal to $P _ { t }$ with probability $x _ { t }$ for each $t \in V _ { H }$ with $x _ { t } > 0$ is distributed as $\mathtt { R R R } ( { \pmb x } )$ ，

• E[f(P)] = X x<sub>t</sub> · E[f(P<sub>t</sub>)] ≥ <sup>E[ℓ(x)]</sup><sub>(d</sub> <sub>−</sub> <sub>1)(r</sub> <sub>+</sub> <sub>1)</sub> , x<sub>t</sub>>0

$\ell ( { \bf 1 } _ { P ^ { \prime } } ^ { ( r ) } ) \geq f ( P ^ { \prime } )$ for all layer-spanning paths $P ^ { \prime }$ with probability 1, and

$\begin{array} { r } { \ell ( \pmb { y } ) = \sum _ { I \in \binom { V } { \leq r + 1 } } \ell _ { I } \cdot \pmb { y } _ { I } } \end{array}$ has coeficients $\ell _ { I } \in [ 0 , f ( V ) ] \forall I \in \bigl ( { \binom { V } { < r + 1 } }$

Proof. The first point is a consequence of $P _ { t } \sim \mathrm { R R R } ^ { | t } ( \mathbf { x } )$ and $\mathbb { P } _ { P \sim \mathtt { R R R } ( \pmb { x } ) } [ t \in P ] = x _ { t }$ , which holds by Lemma 4. For the next two points, we argue by induction over H. For simplicity, we assume that $x _ { v } > 0$ for each $v \in V$ . We discuss at the end of the proof how to handle vertices with $x _ { v } = 0$

For $H = 1$ , set $P _ { t }$ to (t) with probability 1 for each $t \in V _ { 1 }$ . We return $\begin{array} { r } { \ell ( \pmb { y } ) = \sum _ { t \in V _ { 1 } } \ b { y } _ { t } \cdot \ b { f } ( \{ t \} ) } \end{array}$ ， which satisfies the claim because $\ell ( \pmb { x } ) = \mathbb { E } _ { P \sim \mathrm { R R R } ( \pmb { x } ) } [ f ( P ) ]$ and $( d - 1 ) ( r + 1 ) \geq 1$

Now assume that $H \geq d$ and, as in the algorithm, define $\begin{array} { r } { L _ { i } = \{ ( i - 1 ) \frac { H } { d } + 1 , ( i - 1 ) \frac { H } { d } + 2 , \dots , i \frac { H } { d } \} } \end{array}$ for each $i \in \{ 1 , 2 , \ldots , d \}$ . We write $L _ { \leq i } = L _ { 1 } \cup \dots \cup L _ { i }$

Subpaths and linear functions. Using the induction hypothesis we will carefully construct the following random objects. For each $i \in \{ 1 , 2 , \ldots , d \}$ and $v \in V _ { i H / d }$ , we will construct a path $P ^ { | v }$ distributed as paths sampled from RRR<sup>|v</sup>(x) and then restricted to $G [ L _ { \leq i } ]$ . The random paths of the statement are then defined as $P _ { t } = P ^ { | t }$ for each $t \in V _ { H }$

In the process of constructing the paths $P ^ { | v }$ , we also construct, for each $i \in \{ 2 , 3 \ldots , d \}$ , u ∈ $V _ { ( i \mathrm { ~ - ~ } 1 ) H / d }$ , and $v \in V _ { i H / d }$ with $x _ { u , v } > 0$ , a path $P ^ { | u , v }$ distributed as $\mathtt { R R R } ^ { | u , v } ( { \pmb x } )$ restricted to $G [ L _ { i } ]$ Here, the submodular functions that we invoke the induction hypothesis with play a crucial role.

First, via the induction hypothesis on $G [ L _ { 1 } ]$ from ${ \pmb x } [ L _ { 1 } ]$ with $f ,$ construct upper bound $\ell _ { \perp }$ and paths $P ^ { | v }$ from $V _ { 1 }$ to v for each $v \in V _ { H / d }$

Then for each $i \in \{ 2 , 3 \ldots , d \}$ (in that order): For each $w \in V _ { ( i - 1 ) H / d }$ apply the induction hypothesis on $G [ L _ { i } ]$ and $\pmb { x } ^ { | w } [ L _ { i } ]$ with $f ( \cdot \mid P ^ { | w } ) ~ ^ { 7 }$ to obtain an upper bound $\ell ^ { | w \| }$ and paths $P ^ { | w , v }$ for each $v \in V _ { i H / d }$ with $x _ { w , v } > 0$ (equivalently, $x _ { v } ^ { | w } > 0 )$ . Now, conversely, for each $v \in V _ { i H / d }$ , we sample $u \in V _ { ( i \mathrm { ~ - ~ } 1 ) H / d }$ with $\mathbb { P } [ u = w ] = x _ { w } ^ { | v \| }$ for each $w \in V _ { ( i - 1 ) H / d } .$ Then we set $P ^ { | v }$ as the concatenation of $P ^ { | u }$ and $P ^ { | u , v }$

The fact that $P ^ { | v }$ is distributed as paths sampled from $\mathrm { R R R } ^ { | v } ( { \pmb x } )$ and then restricted to $G [ L _ { \leq i } ]$ follows from

$$
\begin{array} { r l } & { \mathbb { P } _ { P \sim \mathrm { R R R } | v } ( \boldsymbol { x } ) \big [ w \in P \big ] = \mathbb { P } _ { P \sim \mathrm { R R R } ( \boldsymbol { x } ) } \big [ w \in P \mid v \in P \big ] } \\ & { \qquad = \frac { \mathbb { P } _ { P \sim \mathrm { R R R } ( \boldsymbol { x } ) } \big [ w \in P \big ] \cdot \mathbb { P } _ { P \sim \mathrm { R R R } ( \boldsymbol { x } ) } \big [ v \in P \mid w \in P \big ] } { \mathbb { P } _ { P \sim \mathrm { R R R } ( \boldsymbol { x } ) } \big [ v \in P \big ] } } \\ & { \qquad = \frac { \boldsymbol { x } _ { w } \cdot \boldsymbol { x } _ { v } ^ { | w } } { \boldsymbol { x } _ { v } } = \frac { \boldsymbol { x } _ { w , v } } { \boldsymbol { x } _ { v } } = \boldsymbol { x } _ { w } ^ { | v | } . } \end{array}
$$

For all linear functions defined above, we extend their domain to $\mathbb { R } ^ { \left( \boldsymbol { \underline { V } } _ { \leq r + 1 } \right) }$ . Formally, we apply them to the projection to their respective domain.

Let $i \in \{ 1 , \ldots , d - 1 \}$ and $w \in V _ { i H / d }$ . One complication that we face is that $\ell ^ { | w \| }$ is not an upper bound for $f ( \cdot )$ , but for $f ( \cdot \mid P ^ { | w } )$ , and that it is linear in $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } } | w$ and not x. To fix this, consider $\ell _ { w } \colon \mathbb { R } ^ { \left( \begin{array} { l } { V } \\ { \leq r + 1 } \end{array} \right) } \to$ R defined by

$$
\ell _ { w } ( { \pmb y } ) = y _ { w } \cdot f ( P ^ { | w } ) + \ell ^ { | w } ( { \pmb y } ^ { + w } ) ,\tag{4}
$$

where $y _ { I } ^ { + w } = y _ { I \cup \{ w \} }$ for all $I \in ( \zeta _ { r } )$ . Then every path $P ^ { \prime }$ from w to $V _ { ( i + 1 ) H } / d$ satisfies

$$
\ell _ { w } ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) = f ( P ^ { | w } ) + \ell ^ { | w } ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) \geq f ( P ^ { | w } ) + f ( P ^ { \prime } \mid P ^ { | w } ) \geq f ( P ^ { \prime } ) .\tag{5}
$$

Adding all these linear functions will yield our upper bound for f. We define

$$
\ell ( { \pmb y } ) = \ell _ { \perp } ( { \pmb y } ) + \sum _ { i = 1 } ^ { d - 1 } \sum _ { { \pmb w } \in V _ { i H / d } } \ell _ { { \pmb w } } ( { \pmb y } ) .
$$

Next, we show that ℓ has the claimed properties.

Expected values of paths. We will now lower bound $\textstyle \sum _ { t \in V _ { H } } x _ { t } \cdot \mathbb { E } [ f ( P ^ { | t } ) ]$ . Let $i \in \{ 2 , 3 , \ldots , d \}$ Then

$$
\begin{array} { r l } { \sum _ { i = 1 } ^ { n } \mu ( \boldsymbol { D } ^ { i + 1 } ) } & { \left( \textbf { 1 } ( \frac { 1 } { \boldsymbol { D } ^ { i + 1 } } ) \boldsymbol { D } ^ { i } \right) \quad \sum _ { j = 1 } ^ { n } \mu _ { i } \nabla \mu ( \boldsymbol { D } ^ { j + 1 } ) } \\ & { = - \sum _ { i = 1 } ^ { n } \mu _ { i } \sum _ { j = 1 } ^ { n } \mu _ { i } \sum _ { i = 1 } ^ { n } \mu _ { i } \nabla \mu ( \boldsymbol { D } ^ { j } ) \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j + 1 } \quad } \\ & { \quad - \sum _ { i = 1 } ^ { n } \mu _ { i } \left( \boldsymbol { D } ^ { i + 1 } \right) \quad \sum _ { j = 1 } ^ { n } \mu _ { i } \nabla \mu ( \boldsymbol { D } ^ { j } ) \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad } \\ & { = \left( \hat { \textbf { E } } \right) \quad \quad \boldsymbol { \hat { D } } ^ { j + 1 } \quad \boldsymbol { \hat { D } } ^ { j + 1 } \quad \displaystyle \sum _ { i = 1 } ^ { n } \mu _ { i } \nabla \mu ( \boldsymbol { D } ^ { j } ) \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad } \\ & { \quad - \sum _ { i = 1 } ^ { n } \mu _ { i } \sum _ { j = 1 } ^ { n } \mu _ { i } \nabla \mu ( \boldsymbol { D } ^ { j } ) \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j + 1 } \quad \boldsymbol { \hat { D } } ^ { j + 1 } \quad \displaystyle \mathrm { l o t o r s u p } ; } \\ &  \quad - \sum _ { i = 1 } ^ { n } \mu _ { i } \nabla \mu _ { i } \nabla \mu _ { i } \nabla \mu ( \boldsymbol { D } ^ { j } ) \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad \boldsymbol { D } ^ { j } \quad  \end{array}\tag{6}
$$

where the inequality follows from the induction hypothesis applied to the construction of $\ell ^ { | w \| }$ and the paths $\{ P ^ { | w , v } \} _ { v \in V _ { i H / d } }$ . (More precisely, we use the second point of Lemma 5.)

By summing (6) over all $i \in \{ 2 , 3 , \ldots , d \}$ , we obtain

$$
\begin{array} { r l } { \displaystyle \sum _ { t \in V _ { H } } x _ { t } \cdot \mathbb { E } [ f ( P ^ { | t } ) ] \geq \displaystyle \sum _ { v \in V _ { H / d } } x _ { v } \cdot \mathbb { E } [ f ( P ^ { | v } ) ] } & { } \\ { \displaystyle } & { + \frac { 1 } { ( d - 1 ) r } \sum _ { i = 1 } ^ { d - 1 } \left[ \mathbb { E } \left[ \displaystyle \sum _ { w \in V _ { i H / d } } \ell _ { w } ( x ) \right] - \displaystyle \sum _ { w \in V _ { i H / d } } x _ { w } \cdot \mathbb { E } [ f ( P ^ { | w } ) ] \right] } \end{array}
$$

$$
\geq \frac { 1 } { ( d - 1 ) r } \mathbb { E } [ \ell _ { \bot } ( \pmb { x } ) ] - \frac { 1 } { r } \sum _ { t \in V _ { H } } x _ { t } \cdot \mathbb { E } [ f ( P ^ { | t } ) ] + \frac { 1 } { ( d - 1 ) r } \sum _ { i = 1 } ^ { d - 1 } \mathbb { E } \left[ \sum _ { w \in V _ { i H / d } } \ell _ { w } ( \pmb { x } ) \right] .
$$

In the second inequality we use that $\begin{array} { r } { \sum _ { w \in V _ { i H / d } } x _ { w } \cdot \mathbb { E } \big [ f ( P ^ { | w } ) \big ] \leq \sum _ { t \in V _ { H } } x _ { t } \cdot \mathbb { E } \big [ f ( P ^ { | t } ) \big ] } \end{array}$ for all i. By moving the terms related to $P ^ { | t }$ to the left and multiplying with $\frac { r } { r + 1 }$ , it follows that

$$
\begin{array} { l } { \displaystyle \sum _ { t \in V _ { H } } x _ { t } \cdot \mathbb { E } [ f ( P ^ { | t } ) ] \geq \frac { 1 } { ( d - 1 ) ( r + 1 ) } \mathbb { E } [ \ell _ { \bot } ( \pmb { x } ) ] + \frac { 1 } { ( d - 1 ) ( r + 1 ) } \sum _ { i = 1 } ^ { d - 1 } \mathbb { E } \left[ \sum _ { w \in V _ { i H / d } } \ell _ { w } ( \pmb { x } ) \right] } \\ { = \frac { 1 } { ( d - 1 ) ( r + 1 ) } \mathbb { E } [ \ell ( \pmb { x } ) ] . } \end{array}
$$

Upper bound on any layer-spanning path. It remains to show that ℓ upper bounds $f ( P ^ { \prime } )$ for any layer-spanning path $P ^ { \prime }$ . For each $i \in \{ 1 , 2 , \ldots , d \}$ , let $u _ { i }$ be the unique vertex in the intersection of $P ^ { \prime }$ and $V _ { i H / d }$ and let $P _ { i } ^ { \prime }$ be the restriction of $P ^ { \prime }$ to $G [ L _ { i } ]$ . By induction and because $P _ { 1 } ^ { \prime }$ is layerspanning in $G [ L _ { 1 } ]$ , we have that $\ell _ { \perp } ( { \bf 1 } _ { P ^ { \prime } } ^ { ( r ) } ) \geq f ( P _ { 1 } ^ { \prime } )$ . By (5), we also have $\ell _ { u _ { i } } ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) \geq f ( P _ { i + 1 } ^ { \prime } )$ for each $i \in \{ 1 , \ldots , d - 1 \}$ . We conclude that

$$
\begin{array} { r l } & { \ell ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) = \ell _ { \perp } ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) + \displaystyle \sum _ { i = 1 } ^ { d - 1 } \sum _ { w \in V _ { i H / d } } \ell _ { w } ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) } \\ & { \qquad = \ell _ { \perp } ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) + \displaystyle \sum _ { i = 1 } ^ { d - 1 } \ell _ { u _ { i } } ( \mathbf { 1 } _ { P ^ { \prime } } ^ { ( r ) } ) \geq f ( P _ { 1 } ^ { \prime } ) + \dots + f ( P _ { d } ^ { \prime } ) \geq f ( P ^ { \prime } ) . } \end{array}
$$

Concluding remarks. Note that by construction, the coeficients of ℓ are conic combinations of polynomially many coeficients from recursive constructions with $r - 1$ , but we have not discussed their magnitude. We can replace each coeficient $\ell _ { I }$ by min $\{ f ( V ) , \ell _ { I } \}$ , since this maintains the other properties. This gives the upper bound on the coeficients.

In the case that $x _ { v } = 0$ for some $v \in V$ , we can perform the construction above on the graph restricted to vertices with non-zero variables. Then, we add to the resulting function $f ( V ) \cdot y _ { v }$ for every $v \in V$ with $x _ { v } = 0$ . This does not afect the lower bounds on the expected value of $\mathtt { R R R } ( { \pmb x } )$ but ensures that every path that uses vertices with zero x-value satisfies the lower bound on ℓ.

To highlight the main ideas of how Lemma 5 can be leveraged, we start with a quick and instructive warm-up, and we will later exploit that our approach is LP-based by adding additional constraints to the linear description to deal with further applications.

## 2.3 Round-or-cut algorithm for Submodular Orienteering

We show how to use Lemma 5 to obtain an $O ( d r )$ -approximation for a decision variant of Submodular Orienteering on layered graphs, without a length bound and with the number of operations of the algorithm depending polynomially on $\langle f \rangle$ , the largest encoding size of a function value $f ( S )$ with $S \subseteq V$ . Both of these limitations will be removed in Section 2.4, where we derive Theorem 1.

In the decision version, for a given bound $T \in \mathbb { R } _ { > 0 }$ , we either output a solution of value at least $\frac { T } { 4 ( d - 1 ) ( r + 1 ) }$ or determine that $f \mathrm { ( O P T ) } < T$ , where $\mathrm { O P T }$ is an optimal solution.

We highlight that, in the interest of simplicity and clarity, we did not make any efort to optimize the constant factor in front of $d r$ in our analysis, which can easily be improved.

Algorithm 2 describes our round-or-cut procedure for Submodular Orienteering. For any $T \in$ $\mathbb { R } _ { \geq 0 }$ , we define the truncated submodular function $f ^ { T } ( S ) : = \operatorname* { m i n } \{ f ( S ) , T \}$ for all $S \subseteq V$

```latex
Algorithm 2: Round-or-cut for Submodular Orienteering
Input: Layered orienteering graph $G = ( V , A )$ with $V = V _ { 1 } \cup \cdot \cdot \cdot \cup V _ { H } , H = d ^ { r }$ , monotone
submodular function $f \colon 2 ^ { V } \to { \mathbb { R } } { \geq } 0$ , and $T \in \mathbb { R } _ { \geq 0 }$
Output: layer-spanning path P with $\begin{array} { r } { f ( \overline { { P } } ) \geq \frac { T } { 4 ( d - 1 ) ( r + 1 ) } } \end{array}$ or determine that $f \mathrm { ( O P T ) } < T$
1 Initialize and start ellipsoid method to find point in a polytope in $[ 0 , 1 ] ^ { \big ( \sum _ { r + 1 } ^ { V } \big ) }$ given by a
separation oracle.
2 whenever Ellipsoid calls separation oracle with point $\pmb { x } \in [ 0 , 1 ] ^ { \binom { V } { \leq r + 1 } }$ do
3 if $x \not \in Q ^ { ( r ) }$ then
4 pass a violated constraint of $Q ^ { ( r ) }$ to ellipsoid and continue with ellipsoid on line 2.
5 else
6 loop
7 Use Lemma 5 with $f ^ { T }$ to get random layer-spanning path P (corresponding to $\overline { { P } }$
in Lemma 5) and random linear function ℓ.
8 if $\begin{array} { r } { f ( P ) \ge \frac { T } { 4 ( d - 1 ) ( r + 1 ) } } \end{array}$ then
9 return P.
10 if $\ell ( { \pmb x } ) < T$ then
11 Pass cutting plane $\ell ( { \pmb x } ) \geq T$ to ellipsoid and continue with ellipsoid on line 2.
12 if ellipsoid terminates because no vector satisfies generated cutting planes then
13 return $\ddot { } f ( \mathrm { O P T } ) < T ^ { \ ' } .$
```

We now briefly discuss correctness and running time of Algorithm 2. First note that there are only two ways for the algorithm to terminate: either it outputs a path P with $\begin{array} { r } { f ( P ) \ge \frac { T } { 4 ( d - 1 ) ( r + 1 ) } } \end{array}$ or it outputs ${ } ^ { \mathrm { 4 4 } } f ( \mathrm { O P T } ) < T ^ { \mathrm { 7 } }$ . Hence, if $f \mathrm { ( O P T ) } < T$ , then if the algorithm terminates, it returns a correct result.

Now consider the case that $T \leq f ( \mathrm { O P T } )$ . Note that cutting planes generated in Algorithm 2 are valid for the vector $\mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } \in Q ^ { ( r ) }$ . This is clearly true for the constraints of $Q ^ { ( r ) }$ , which get checked in line 4 of Algorithm 2. Moreover, a cutting plane $\ell ( { \pmb x } ) \geq T$ generated in line 11 is valid because $\ell ( \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ) \geq f ^ { T } ( \mathrm { O P T } ) = \operatorname* { m i n } \{ f ( \mathrm { O P T } ) , T \} = T$ , where the first inequality follows from Lemma 5. Thus, also in this case, if the algorithm terminates, it returns a correct result.

It remains to discuss the running time of Algorithm 2. Note that checking whether $\pmb { x } \in Q ^ { ( r ) }$ can be done in time $n ^ { O ( r ) }$ by checking all constraints of $Q ^ { ( r ) }$ . All other single operations in Algorithm 2 can be done in polynomial time. The number of iterations of the ellipsoid method is polynomial in the dimension of the space and the bit complexity of the cuts. The bit complexity of the constraints in $Q ^ { ( r ) }$ is polynomial, and the bit complexity of the cuts $\ell ( { \pmb x } ) \geq T$ is also polynomial in $n ^ { r }$ and the maximum bit complexity of a function value of $f .$

Hence, to obtain an expected $n ^ { O ( r ) } \langle f \rangle ^ { O ( 1 ) }$ time algorithm, it sufices to show that the expected number of iterations of the inner loop in Algorithm 2 is polynomial, which we show next.

Lemma 6. In every iteration of the inner loop in Algorithm 2, the probability that the algorithm either outputs a path P with $\begin{array} { r } { f ( P ) \ge \frac { T } { 4 ( d - 1 ) ( r + 1 ) } } \end{array}$ or generates a cutting plane $\ell ( { \pmb x } ) \geq T$ is at least

Ω( <sup>1</sup><sub>dr</sub> ).

Proof. Let $\pmb { x } \in Q ^ { ( r ) }$ be the point that is currently passed to the inner loop in Algorithm 2, and let $P$ and ℓ be the random path and linear function obtained from Lemma 5 with respect to x and $f ^ { T }$ $\begin{array} { r } { \mathrm { I f ~ } \mathbb { E } [ f ^ { T } ( P ) ] < \frac { T } { 2 ( d - 1 ) ( r + 1 ) } } \end{array}$ , then Lemma 5 implies

$$
\mathbb { E } [ \ell ( \pmb { x } ) ] \le ( d - 1 ) ( r + 1 ) \cdot \mathbb { E } [ f ^ { T } ( P ) ] < \frac { T } { 2 } .
$$

Thus, by Markov’s inequality we get $\mathbb { P } [ \ell ( { \pmb x } ) < T ] \geq 1 / 2$ , and the algorithm generates a cutting plane $\ell ( { \pmb x } ) \geq T$ with probability at least $^ 1 / 2$

Suppose now that $\begin{array} { r } { \mathbb { E } [ f ^ { T } ( P ) ] \ge \frac { T } { 2 ( d - 1 ) ( r + 1 ) } } \end{array}$ , which implies

$$
\begin{array} { l } { \displaystyle \frac { T } { 2 ( d - 1 ) ( r + 1 ) } \le \mathbb { E } [ f ^ { T } ( P ) ] } \\ { \displaystyle \le \mathbb { P } \left[ f ^ { T } ( P ) \ge \frac { T } { 4 ( d - 1 ) ( r + 1 ) } \right] \cdot T } \\ { \displaystyle \qquad + \left( 1 - \mathbb { P } \left[ f ^ { T } ( P ) \ge \frac { T } { 4 ( d - 1 ) ( r + 1 ) } \right] \right) \cdot \frac { T } { 4 ( d - 1 ) ( r + 1 ) } . } \end{array}
$$

By rearranging we get

$$
\mathbb { P } \left[ f ^ { T } ( P ) \geq \frac { T } { 4 ( d - 1 ) ( r + 1 ) } \right] \geq \frac { 1 } { 4 ( d - 1 ) ( r + 1 ) - 1 } ,
$$

and the statement follows from $f ( P ) \geq f ^ { T } ( P )$

Hence, in summary, we obtained an expected $n ^ { O ( r ) } \langle f \rangle ^ { O ( 1 ) }$ time $O ( d r )$ -approximation for the decision version of Submodular Orienteering (without a length bound) by using Lemma 5 in a round-or-cut procedure.

Consequently, Algorithm 2 returns an O(dr)-approximation for Submodular Orienteering if run with a T such that $T \le f \mathrm { ( O P T ) }$ and $T = \Omega ( f ( \mathrm { O P T } ) )$ . Such a $T ,$ even with the property that $f ( \mathrm { O P T } ) \ge T \ge ( 1 - \varepsilon ) f ( \mathrm { O P T } )$ , can typically be found easily through binary search using Algorithm 2. See Section 2.4 for a more detailed discussion of this point in the more general setting with additional constraints. This leads to approximation ratios of $O ( \log n )$ and $O ( n ^ { \varepsilon } / \varepsilon )$ in quasipolynomial time and polynomial time, respectively. The $1 / \varepsilon$ factor in the latter can be removed by rescaling ε.

## 2.4 Round-or-cut with linear constraints

As discussed, a main benefit of our LP-based approach is that we can easily add constraints to the linear description and use them in a round-or-cut procedure to obtain good approximation guarantees for submodular maximization. To this end, consider an arbitrary family of additional constraints we want to add to $Q ^ { ( r ) }$ . Let us denote these constraints by $D y \leq b$ for some matrix $D \in \mathbb { Z } ^ { m \times \binom { V } { \leq r + 1 } }$ and vector $\pmb { b } \in \mathbb { Z } ^ { m }$ , where $\pmb { y } \in \mathbb { R } ^ { \binom { V } { \leq r + 1 } }$ . We denote by $\langle D , b \rangle$ the encoding size of D and $^ { b , }$ which is upper bounded by $m \cdot | { \bigl ( } { \underset { < r + 1 } { V } } { \bigr ) } | \cdot ( 1 + \lceil \log ( 1 + \| D \| _ { \infty } ) + \log ( 1 + \| b \| _ { \infty } ) { \bigr ] } ) . ^ { 8 }$ If the constraints are rational (but non-integer), we assume they are first normalized by multiplying with the lowest common denominator.

We need to adapt the round-or-cut procedure to deal with additional constraints. It is typically easy to separate over additional constraints. However, if we use a procedure akin to Algorithm $2 ,$ we have to be aware that this procedure returns a path $P ,$ rounded from some point ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ only when a certain condition is satisfied (in this case $\begin{array} { r } { f ( P ) \geq \frac { T } { 4 ( d - 1 ) ( r + 1 ) } ) } \end{array}$ . Otherwise, we may repeat the loop. This may introduce biases in the distribution of P that are hard to control. Furthermore, we need to specify what we compare our solution against.

Optimum under linear constraints. Previously, OPT was a layer-spanning path of maximal function value. In the new setting, it would be natural to restrict $\mathrm { O P T }$ to a path satisfying the linear constraints. However, a weaker restriction will sufice for our analysis and this broadens the range of applications. Specifically, we compare against $\mathbb { E } _ { \mathrm { O P T } \sim \mathcal { O } } [ f ( \mathrm { O P T } ) ]$ for a distribution O over layer spanning paths that satisfies the linear constraints in expectation, that is, $\mathbb { E } _ { \mathrm { O P T } \sim \mathcal { O } } [ D \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ] \leq b .$ We assume without loss of generality that O is the distribution with maximal expected function value and we omit $\mathrm { O P T } \sim \mathcal { O }$ in $\mathbb { E } [ \cdot ]$ if it is clear from the context.

Main statement. Our goal is to adapt the round-or-cut procedure to obtain the following.

Lemma 7. Let $G = ( V , A )$ be a layered graph with $V = V _ { 1 } \cup \cdot \cdot \cdot \cup V _ { H } , H = d ^ { r }$ , let $f \colon 2 ^ { V } \to { \mathbb { R } } _ { > 0 }$ be monotone submodular, let $D y \leq b$ be constraints for $\pmb { y } \in \mathbb { R } ^ { \binom { V } { \leq r + 1 } }$ with $Q ^ { ( r ) } \cap \{ y \in \mathbb { R } ^ { \binom { V } { \leq r + 1 } }$ : Dy ≤ $b \} \neq \emptyset$ , and let $\varepsilon > 0$

There is a randomized $n ^ { O ( r ) } \cdot \langle D , \boldsymbol { b } \rangle ^ { O ( 1 ) }$ time procedure that returns an $\pmb { x } \in Q ^ { ( r ) }$ with $D x \leq b$ and, with probability $1 - n ^ { - { \frac { 1 } { \varepsilon } } }$ , the returned x satisfies

$$
\mathbb { E } _ { P \sim \mathtt { R R R } ( x ) } [ f ( P ) ] \ge \frac { 1 - \varepsilon } { ( d - 1 ) ( r + 1 ) } \cdot \mathbb { E } _ { \mathrm { O P T } \sim \mathcal { O } } [ f ( \mathrm { O P T } ) ] ,
$$

for any distribution O over layer-spanning paths that satisfies $\mathbb { E } _ { \mathrm { O P T } \sim \mathcal { O } } [ D \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ] \le b$

There is a subtle aspect about this lemma that is worth pointing out: $\mathbb { E } _ { P \sim \mathrm { R R R } ( \pmb { x } ) } [ D \mathbf { 1 } _ { P } ^ { ( r ) } ] \leq b$ may not hold. This is because RRR(x) is not necessarily marginal preserving on variables $x _ { I }$ with $| I | > 1$ Thus, the distribution RRR(x) is in this regard less constrained than O (though it is also more constrained in other regards). In particular, it is possible that $\mathbb { E } _ { P \sim \mathrm { R R R } ( x ) } [ f ( P ) ] > \mathbb { E } _ { \mathrm { O P T } \sim \mathcal { O } } [ f ( \mathrm { O P T } ) ]$ for some x with $D x \leq \pmb { b }$

Before proving the lemma, we first show how it implies Theorem 1.

Proof of Theorem 1 assuming Lemma 7. Suppose we are given an instance of Submodular Orienteering. We reduce from a general, not necessarily layered, graph to a layered one as follows. Let $H \geq n + 2$ to be specified later. We consider the following vertices: for each $i \in \{ 2 , \ldots , H - 1 \}$ and each $u , v \in V$ , where v is reachable from u in the original graph (in particular, if $u = v )$ , we let $( u , v , i )$ be a vertex in $V _ { i }$ . Furthermore, we set $V _ { 1 } = \{ ( s , s , 1 ) \}$ and $V _ { H } = \{ ( t , t , H ) \}$ . There is an arc from $( u , v , i ) \in V _ { i }$ to $( u ^ { \prime } , v ^ { \prime } , i + 1 ) \in V _ { i + 1 } { \mathrm { ~ i f ~ } } v = u ^ { \prime }$ . With each vertex $( u , v , i )$ we associate a length, which is the length of the shortest path from u to v in the original graph. The submodular function is extended to $f ^ { \prime } ( U ) : = f ( \{ v \mid \exists i \in \{ 1 , \ldots , H \} , u \in V$ with $( u , v , i ) \in U \}$ , which is again submodular. This reduction is approximation-preserving: Each walk $W$ in the original graph can be transformed into a layer-spanning path of the same value and at most the same length in the layered graph. For this, let $v _ { 1 } , \ldots , v _ { \ell }$ for some $\ell \leq n$ be the sequence of distinct vertices visited in $W$ , in the order they first appear in $W$ . Then $s = v _ { 1 }$ and a layer-spanning path satisfying this is

$$
( v _ { 1 } , v _ { 1 } , 1 ) , ( v _ { 1 } , v _ { 2 } , 2 ) , ( v _ { 2 } , v _ { 3 } , 3 ) , \ldots , ( v _ { \ell } , t , \ell + 1 ) , ( t , t , \ell + 2 ) , \ldots , ( t , t , H ) .
$$

On the other hand, any layer-spanning path can be translated into an s-t walk of the same length and at least the same value by replacing each $( u , v , i )$ by a shortest path from u to v and concatenating all paths.<sup>9</sup>

We use $D , b$ to encode the length constraint. More precisely, D contains a single row where the coeficient of each $x _ { \{ ( u , v , i ) \} }$ is set to the shortest $_ { u - v }$ path length in the original graph. All other coeficients, i.e., those corresponding to combinations of variables, are set to zero. The single component of the right-hand side b is the budget C, for which we can assume $C \leq H n \cdot \operatorname* { m a x } _ { a \in A } w ( a )$ since the length of H many shortest paths cannot exceed the right-hand side. Thus, $\left. D , b \right. \ \leq$ $( \left. w \right. + n ) ^ { O ( 1 ) }$ . We compute x using Lemma $7$ and output $P \sim \tt R R R ( \pmb { x } )$ . Then the expected length of P is at most C because of Lemma 4. For an optimal solution OPT, let O be the distribution that picks OPT with probability 1. Then

$$
\mathbb { E } [ f ( P ) ] \ge ( 1 - n ^ { - 1 / \varepsilon } ) \frac { 1 - \varepsilon } { ( d - 1 ) ( r + 1 ) } f ( \mathrm { O P T } ) .
$$

By setting $d = 2$ and $r = \Theta ( \log n )$ , or $d = n ^ { \Theta ( \varepsilon ) }$ and $r = \Theta ( ^ { 1 } / \varepsilon )$ , we get the claimed trade-ofs.

The rest of the section is dedicated to proving Lemma $7 .$

Initial bound. We show how to compute a solution x with a weak initial guarantee. This allows us later to restrict the range of the binary search for the threshold T, which will play in our new algorithm an analogous role to the threshold T in Algorithm 2.

Lemma 8. In time $n ^ { O ( r ) } \cdot \langle D , \boldsymbol { b } \rangle ^ { O ( 1 ) }$ we can compute $U \subseteq V , \pmb { x } \in Q ^ { ( r ) }$ , and $c _ { D } \leq 2 ^ { n ^ { O ( r ) } \langle D , \pmb { b } \rangle ^ { O ( 1 ) } }$ such that $D x \leq b$ and for every $u \in V \setminus U$ we have $\mathbb { P } _ { \mathrm { O P T } \sim \mathcal { O } } [ u \in \mathrm { O P T } ] = 0$ . Furthermore,

$$
\mathbb { E } _ { P \sim \mathrm { R R R } ( \pmb { x } ) } [ f ( P ) ] \geq \frac { 1 } { n c _ { D } } \cdot f ( U ) .
$$

Proof. Let $u \in U$ if and only if there exists a solution $\boldsymbol { x } \in Q ^ { ( r ) }$ with $D x \leq b$ and $x _ { u } \mathrm { ~ > ~ } 0$ . The set $U$ can be computed by solving a linear program for each vertex in the required time. Note that every $v \in V$ with $\mathbb { P } _ { \mathrm { O P T } \sim \mathcal { O } } [ v \in \mathrm { O P T } ] > 0$ must be in $U _ { : }$ , since one can take $\mathbf { \Delta } \mathbf { x } = \mathbb { E } _ { \mathrm { O P T } \sim \mathcal { O } } [ \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ]$ Let x be a vertex solution with the properties as above for $u \in U$ maximizing $f ( \{ u \} )$ , which can be computed the same way.

By Cramer’s rule, the denominator of $x _ { u }$ is the determinant of a submatrix of the constraint matrix of ${ Q ^ { ( r ) } \cap \{ x : D x \leq b \} }$ , which is bounded by some $c _ { D } \leq 2 ^ { n ^ { O ( r ) } \langle D , \pmb { b } \rangle ^ { O ( 1 ) } }$ . It follows that

$$
\mathbb { E } _ { P \sim \mathtt { R R R } ( \pmb { x } ) } [ f ( P ) ] \ge x _ { u } f ( \{ u \} ) \ge \frac { 1 } { n c _ { D } } \cdot f ( U ) .
$$

The first inequality follows from $\mathbb { P } [ u \in P ] = x _ { u }$ (by Lemma 4) and monotonicity of $f .$ □

Avoiding bit complexity dependence. The next lemma will be used to round the linear functions obtained from Lemma 5 in order to avoid cuts with a high bit complexity in the ellipsoid method and to help us avoid unnecessary dependence of the running time on the bit complexity of $f .$

Lemma 9. Let $c _ { D }$ be as in Lemma 8. Let $\begin{array} { r } { \ell ( \pmb { y } ) = \sum _ { I \in \binom { V } { < r + 1 } } } \end{array}$ ℓ<sub>I</sub>y<sub>I</sub> be a linear function over $\textbf {  { y } } \in$ $\mathbb { R } ^ { \binom { V } { \leq r + 1 } }$ with $\ell _ { I } \in [ 0 , f ( V ) ]$ for $I \in \big ( { \boldsymbol { \kappa } } _ { } \\ { \leq r + 1 } \big )$ . Let $\begin{array} { r } { T \in \left\lceil \frac { f ( V ) } { n c _ { D } } , f ( V ) \right\rceil } \end{array}$

We can compute in time $n ^ { O ( r ) }$ another linear function $\begin{array} { r } { \overline { { \ell } } ( \pmb { y } ) \doteq \sum _ { I \in \binom { V } { < r + 1 } } \overline { { \ell } } _ { I } \pmb { y } _ { I } } \end{array}$ and some $\overline { { T } } \in \mathbb { Z } _ { \geq 0 }$ such that all coeficients $\overline { { \ell } } _ { I }$ and also $\overline { T }$ are non-negative integers bounded by $2 c _ { D } ^ { 2 } n ^ { r + 3 } / \varepsilon + 1$ satisfying $\overline { { \ell } } ( { \pmb y } ) < \overline { { T } } \Rightarrow \ell ( { \pmb y } ) < T$ for all $\pmb { y } \in [ 0 , 1 ] ^ { \binom { V } { \leq r + 1 } }$ , and

• ℓ(y) < (1 − ε)T ⇒ ℓ(y) < T for all $\pmb { y } \in [ 0 , 1 ] ^ { \binom { V } { \leq r + 1 } }$

Proof. We set

$$
\overline { { T } } : = \left\lceil \frac { c _ { D } n T } { \varepsilon f ( V ) } \cdot \left. \binom { V } { \leq r + 1 } \right. \right\rceil \in \left[ \frac { 1 } { \varepsilon } \cdot \left. \binom { V } { \leq r + 1 } \right. , \frac { 2 c _ { D } n } { \varepsilon } \cdot \left. \binom { V } { \leq r + 1 } \right. \right] .
$$

Furthermore, for every $I \in \big ( { \boldsymbol { \kappa } } _ { } \\ { \leq r + 1 } \big )$ we set

$$
\overline { { \ell } } _ { I } : = \left\lceil \ell _ { I } \cdot \frac { \overline { { T } } } { T } \right\rceil \leq f ( V ) \cdot \frac { \overline { { T } } } { T } + 1 \leq \overline { { T } } c _ { D } n + 1 .
$$

We now verify the two claimed properties of $\overline { { \ell } }$ and $\overline { { T } } .$ . Let $\pmb { y } \in [ 0 , 1 ] ^ { \binom { V } { \leq r + 1 } }$ . If $\overline { { \ell } } ( y ) < \overline { { T } }$ then

$$
\ell ( { \pmb y } ) \le \bar { \ell } ( { \pmb y } ) \cdot \frac { T } { \overline { { T } } } < T .
$$

If, on the other hand, $\begin{array} { r } { \ell ( \pmb { y } ) < ( 1 - \varepsilon ) T . } \end{array}$ , then

$$
\overline { { \ell } } ( \pmb { y } ) \leq \ell ( \pmb { y } ) \cdot \frac { \overline { { T } } } { T } + \left| \binom { V } { \leq r + 1 } \right| < ( 1 - \varepsilon ) \overline { { T } } + \varepsilon \overline { { T } } = \overline { { T } } .
$$

Lemma 7 can be obtained by a modification of Algorithm 2, which is described in Algorithm 3. We combine this algorithm with a few additional steps. Then we invoke Lemma 8. If $U \subsetneq V$ , we can remove all vertices in $V \backslash U$ from the instance, since they are not relevant for $\mathrm { O P T }$ , and restart the algorithm on the smaller instance. Thus, assume without loss of generality $V = U$ . We perform a binary search over $\begin{array} { r l } { T \in } & { { } \Big \lceil \frac { f ( V ) } { n c _ { D } } , f ( V ) \Big \rceil } \end{array}$ . We stop the search when the ratio between upper and lower bound is at most $\frac { 1 } { 1 - \varepsilon }$ . If Algorithm 3 fails in every invocation, we return the solution from Lemma 8.

In Algorithm 3, κ is chosen to be a suficiently large constant so that $( 1 - \varepsilon ) ^ { \kappa \cdot r \ln ( n + \langle D , \pmb { b } \rangle ) } \leq$ $\langle D , \pmb { b } \rangle ^ { - c } n ^ { - r c - \frac { 1 } { \varepsilon } }$ , where $n ^ { c r } { \cdot } \langle D , { \pmb b } \rangle ^ { c }$ is an upper bound for the number of iterations of the binary search times the number of iterations needed by the ellipsoid method to finish when run with constraints of $Q ^ { ( r ) }$ and $D x \leq b ,$ , and with further constraints with coeficients bounded by $2 c _ { D } ^ { 2 } n ^ { r + 3 } / \varepsilon + 1$ , the term from Lemma 9. This is for example achieved by $\begin{array} { r } { \kappa : = \frac { 1 } { \varepsilon } \cdot ( 2 c + \frac { 1 } { \varepsilon } ) } \end{array}$

We now show that the procedure indeed fulfills the properties stated in Lemma 7.

```latex
Algorithm 3: Round-or-cut for constrained Submodular Orienteering
Input: Layered orienteering graph $G = ( V , A )$ with $V = V _ { 1 } \cup \cdot \cdot \cdot \cup V _ { H } , H = d ^ { r }$ , monotone
submodular function $f \colon 2 ^ { V } \to \mathbb { R } _ { \geq 0 } , T \in \mathbb { R } _ { \geq 0 }$ , and constraints $D y \leq b$ for
$\pmb { y } \in [ 0 , 1 ] ^ { \binom { V } { \leq r + 1 } }$ with $Q ^ { ( r ) } \cap \{ \pmb { y } \in \mathbb { R } ^ { \binom { V } { \leq r + 1 } } : D \pmb { y } \leq \pmb { b } \} \neq \emptyset .$
Output: ${ \pmb x } \in Q ^ { ( r ) }$ with $D x \leq b$ such that $\begin{array} { r } { \mathbb { E } _ { P \sim \mathrm { R R R } ( \pmb { x } ) } \overline { { [ f ( P ) ] } } \geq \frac { ( 1 - \varepsilon ) ^ { 2 } } { ( d - 1 ) ( r + 1 ) } } \end{array}$ · T or determine
that $\mathbb { E } [ f ( \mathrm { O P T } ) ] < T .$
1 Initialize and start ellipsoid method to find point in a polytope in $[ 0 , 1 ] ^ { \big ( \sum _ { r + 1 } ^ { V } \big ) }$ given by a
separation oracle.
2 whenever Ellipsoid calls separation oracle with point $\pmb { x } \in [ 0 , 1 ] ^ { \binom { V } { \leq r + 1 } }$ do
3 if $x \not \in Q ^ { ( r ) }$ or Dx ̸≤ b then
4 pass a violated constraint of $Q ^ { ( r ) }$ or $D x \leq b$ to ellipsoid and continue on line 2.
5 else
6 repeat $\kappa \cdot r [ \ln ( n + \langle D , \boldsymbol { b } \rangle ) ]$ times
7 Use Lemma 5 with f to get random layer-spanning path P and random linear
function ℓ.
8 Use Lemma 9 with ℓ and T to get ${ \overline { { \ell } } } , { \overline { { T } } } .$
9 if $\overline { { \ell } } ( { \pmb x } ) < \overline { { T } }$ then
10 pass cutting plane $\overline { { \ell } } ( { \pmb x } ) \geq \overline { { T } }$ to ellipsoid and continue with ellipsoid on line 2.
11 return x
12 if ellipsoid terminates because no vector satisfies generated cutting planes then
13 return $\begin{array} { r } { \mathrm { \bar { E } } [ f ( \mathrm { O P T } ) ] < T ^ { \mathfrak { * } } . } \end{array}$
```

Proof of Lemma 7. We first discuss correctness of Algorithm 3. If the algorithm outputs that $\mathop { \mathrm { \tiny ~  ~ } } ^ { \mathrm { \tiny ~  ~ } } \mathbb { E } [ f ( \mathrm { O P T } ) ] < T ^ { \flat } ,$ , then this is correct due to the following. Let $\pmb { x } = \mathbb { E } [ \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ]$ . Then $\boldsymbol { x } \in Q ^ { ( r ) }$ and $D x \leq b .$ Since the generated cutting planes define an infeasible set of constraints, there must be a generated cut with $\begin{array} { r } { \overline { { \ell } } ( { \pmb x } ) < \overline { { T } } } \end{array}$ . This cut was derived from some ℓ and T using Lemma 9. By the lemma, it follows that $\ell ( { \pmb x } ) < T$ . Since ℓ was constructed by Lemma 5, for any layer-spanning path P we have $f ( P ) \leq \ell ( { \bf 1 } _ { P } ^ { ( r ) } )$ . Thus, $\mathbb { E } [ f ( \mathrm { O P T } ) ] \le \mathbb { E } [ \ell ( \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ) ] = \ell ( \pmb { x } ) < T$ and the conclusion is correct.

If the algorithm does not return $\begin{array} { r } { ^ { \mathrm { * } } \mathbb { E } [ f ( \mathrm { O P T } ) ] < T ^ { \mathrm { * } } } \end{array}$ , then it must return a point $\pmb { x } \in Q ^ { ( r ) }$ with $D x \leq \boldsymbol { b }$ . Indeed, the ellipsoid method runs in polynomial time and therefore cannot stay forever inside the loop starting at line 2. Consider now any moment in the execution of Algorithm 3 when it is at line 6. Let $\pmb { x } \in Q ^ { ( r ) }$ be the point that is currently passed to the repeat loop, and let P and ℓ be the random path and random linear function obtained from Lemma 5 with respect to x. If $\mathbb { P } [ \ell ( \pmb { x } ) < ( 1 - \varepsilon ) T ] \geq \varepsilon$ , then since ${ \ell ( { \pmb x } ) < ( 1 - \varepsilon ) T }$ implies $\overline { { \ell } } ( { \pmb x } ) < \overline { { T } }$ , the algorithm will generate in each repetition of the repeat loop a cutting plane with probability at least ε. Hence, the probability that no cutting plane is generated in $\kappa \cdot r [ \ln ( n + \langle D , \boldsymbol { b } \rangle ) ]$ repetitions is at most

$$
( 1 - \varepsilon ) ^ { \kappa \cdot r \ln ( n + \langle D , \pmb { b } \rangle ) } \leq \langle D , \pmb { b } \rangle ^ { - c } n ^ { - r c - \frac { 1 } { \varepsilon } } ,
$$

where the inequality follows by our choice of κ. Because the number of ellipsoid iterations over all iterations of the binary search is at most $\langle D , b \rangle ^ { c } n ^ { r c }$ , a union bound implies that with probability at most $n ^ { - \frac { 1 } { \varepsilon } }$ , there is at least one moment during the whole execution of Algorithm 3 when $\mathbb { P } [ \ell ( { \pmb x } ) <$ $( 1 - \varepsilon ) T ] \geq \varepsilon$ but no cutting plane is generated in the repeat loop. We finish the proof by showing that only in such a case, the algorithm may return a point x with $\begin{array} { r } { \mathbb { E } [ f ( P ) ] < \frac { T } { ( d - 1 ) ( r + 1 ) } \cdot ( 1 - \varepsilon ) ^ { 2 } } \end{array}$ Indeed, if this case did not happen, and a vector x is returned, then $\mathbb { P } [ \ell ( \pmb { x } ) \geq ( 1 - \varepsilon ) T ] \geq 1 - \varepsilon$ Thus, $\mathbb { E } [ \ell ( { \pmb x } ) ] \geq ( 1 - \varepsilon ) ^ { 2 } T$ , which implies

$$
\mathbb { E } _ { P \sim \mathrm { R R R } ( \pmb { x } ) } [ f ( P ) ] \ge \frac { \mathbb { E } [ \ell ( \pmb { x } ) ] } { ( d - 1 ) ( r + 1 ) } \ge \frac { ( 1 - \varepsilon ) ^ { 2 } T } { ( d - 1 ) ( r + 1 ) } ,
$$

where the first inequality follows from Lemma 5.

We lose another factor of $( 1 - \varepsilon )$ from terminating the binary search with a small remaining gap. This proves a guarantee with $( 1 - \varepsilon ) ^ { 3 }$ instead of $1 - \varepsilon _ { i }$ , but we can also obtain the latter by rescaling ε with a constant.

By Lemma 8, the optimum cannot be above the range of the binary search because none of the elements deleted at the beginning is part of any realization of OPT. We recall that the binary search range is $[ f ( V ) / n c _ { D } , f ( V ) ]$ and that we replaced V by the set U from Lemma 8 at the beginning of the algorithm. If $\mathbb { E } _ { \mathrm { O P T } \sim \mathcal { O } } [ f ( \mathrm { O P T } ) ]$ is below the range, then the guarantee of the solution from Lemma 8 sufices.

The running time of Algorithm 3 follows from the fact that each single operation takes at most $n ^ { O ( r ) }$ time and the ellipsoid method terminates after at most polynomially many iterations. □

## 3 Submodular Markov Decision Process

To simplify the transfer of our results, we deviate from the typical formalization of MDPs with states and actions as in the introduction. The following model is equivalent but closer to Submodular Orienteering. As before we consider a directed graph $G = ( V , A )$ on layers $V = V _ { 1 } \cup V _ { 2 } \cup \dots \cup V _ { H }$

Each vertex is either a decision vertex or a chance vertex. We write D for the decision vertices and C for the chance vertices. Thus, $V = C \cup D$ . There is one distinct chance vertex s with $V _ { 1 } = \{ s \}$ . There are transition probabilities $\pmb { p } = ( p _ { c } ( v ) ) _ { c \in C , v \in V }$ that describe the probability that the successor of some $c \in C$ is $v \in V$ . We assume that $p _ { c } ( v )$ is non-zero if and only if there is an arc $( c , v )$ in the graph. A (randomized) policy π is a function that assigns to any trajectory $( v _ { 1 } , v _ { 2 } , \ldots , v _ { \ell } )$ , which is a walk in G starting in $v _ { 1 } = s$ and ending in some decision vertex $v _ { \ell } \in D$ a probability distribution over vertices. We write $\pi _ { \boldsymbol { v } _ { \le \ell } } ( \boldsymbol { v } )$ for the probability that v is chosen as the next vertex based on the trajectory $v _ { \leq \ell } = ( v _ { 1 } , v _ { 2 } , \ldots , v _ { \ell } )$ . We require that $\pi _ { v < \ell } ( v ) = 0$ unless there is an arc $( v _ { \ell } , v )$ in the graph. We are mainly interested in implicit policies that can be computed fast, more precisely in polynomial or quasi-polynomial time, on a given trajectory.

We consider here a monotone submodular reward function $f \colon 2 ^ { V } \to \mathbb { R } _ { > 0 }$ . We associate with the MDP and a policy the expectation of the reward $f ( \{ v _ { 1 } , v _ { 2 } , \dots , v _ { H } \} )$ over the vertices that are visited in the stochastic process where $v _ { 1 } = s$ , and $v _ { i }$ for $i > 1$ , is chosen either based on $p _ { v _ { i - 1 } }$ , if $v _ { i - 1 }$ is a chance vertex, or based on $\pi ,$ otherwise.

We assume without loss of generality that $H \ = \ d ^ { r }$ for some $d \in \mathbb { Z } _ { \geq 2 }$ and $r \in \mathbb { Z } _ { \geq 1 }$ The equivalence to the model in the introduction can be seen as follows. For each time and state, $s _ { t }$ , we have a decision vertex $v \in V _ { 2 t }$ . For each time, state, and action, $s _ { t }$ and $a _ { t } .$ , we have a chance vertex $u \in V _ { 2 t + 1 }$ . Furthermore, there is a chance vertex s with $\{ s \} = V _ { 1 }$ , which is distinct from all other vertices. The transition probability $p _ { s } ( v )$ is the probability of state $v$ to be the initial state and the transition probabilities $p _ { u } ( v ) , u \in C \setminus \{ s \}$ , come from the transition probabilities of a time, state and action. If the number of layers is not a power of $d ,$ we add dummy layers. This transformation increases H only by a constant factor, which is insignificant for our results.

The LP formulation. To obtain an LP formulation that captures policies of MDPs, we use $Q ^ { ( r ) }$ with the following additional constraints, which intuitively enforce that on chance vertices, the successor vertex is chosen according to the given transition distribution:

$$
\begin{array} { r l r l } & { x _ { I \cup \{ u \} } = x _ { I } \cdot p _ { c } ( u ) } & & { \forall i \in \{ 1 , 2 , \ldots , H - 1 \} , c \in V _ { i } \cap C , u \in V _ { i + 1 } , } \\ & { } & & { \{ c \} \subseteq I \subseteq V _ { 1 } \cup \ldots \cup V _ { i } , | I | \leq r . } \end{array}\tag{7}
$$

Here, we think of $x _ { I }$ and $x _ { I \cup \{ u \} }$ as the probabilities that all vertices in I (or in $I \cup \{ u \} )$ are visited. In the statement below, as well as later, when talking about “the optimal policy”, we mean any fixed optimal policy in case there are multiple optimal policies.

Lemma 10. Let $\mathrm { O P T } \sim \mathcal { O }$ where O is the distribution over layer-spanning paths arising from the MDP process with the optimal policy, that is, applying the optimal policy on decision vertices and the transition distributions p for chance vertices. Then $\mathbb { E } [ \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ]$ satisfies (7).

Proof. Let $c \in V _ { i } \cap C , u \in V _ { i + 1 }$ , and $\{ c \} \subseteq I \subseteq V _ { 1 } \cup \cdot \cdot \cdot \cup V _ { i } , | I | \leq r . \mathrm { ~ I f ~ } \mathbb { P } [ I \subseteq \mathrm { O P T } ] = 0$ then also $\mathbb { P } [ I \cup \{ u \} \subseteq \mathrm { O P T } ] = 0$ . Thus,

$$
\mathbb { E } [ ( \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ) _ { I \cup \{ u \} } ] = 0 = \mathbb { E } [ ( \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ) _ { I } ]
$$

and both sides in (7) are zero. Now assume that $\mathbb { P } [ I \subseteq \mathrm { O P T } ] > 0$ . Then

$$
\begin{array} { r } { \mathbb { E } [ ( \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ) _ { I \cup \{ u \} } ] = \mathbb { P } [ I \cup \{ u \} \subseteq \mathrm { O P T } ] = \mathbb { P } [ I \subseteq \mathrm { O P T } ] \cdot p _ { c } ( u ) = \mathbb { E } [ ( \mathbf { 1 } _ { \mathrm { O P T } } ^ { ( r ) } ) _ { I } ] \cdot p _ { c } ( u ) . } \end{array}
$$

The policy. We will prove that the policy in Algorithm 4 (together with p) produces a distribution identical to RRR(x) and from this we derive the claimed approximation guarantee. For convenience, we define the policy in Algorithm 4 for any layer ℓ and any trajectory $( v _ { 1 } , \ldots , v _ { \ell } )$ , even if $v _ { \ell }$ is a chance node. In this case, the policy is not actually used, but it is still well-defined. Note that by (7), Algorithm 4 actually simulates the environment’s probability distribution on chance nodes. This is convenient later for the analysis.

Algorithm 4: Submodular MDP Policy   
Input: Layered MDP graph $G = ( V , A )$ with $V = V _ { 1 } \cup \cdots \cup V _ { H } , H = d ^ { r } , { \pmb x } \in { \cal Q } ^ { ( r ) }$ satisfy   
(7), and trajectory $( v _ { 1 } , v _ { 2 } , \ldots , v _ { \ell } ) \in V _ { 1 } \times V _ { 2 } \times \cdots \times V _ { \ell } .$   
Output: Distribution over vertices in $V _ { \ell + 1 }$ that are neighbors of $v _ { \ell } .$   
1 Let $K = \{ \operatorname* { m a x } \{ k \cdot d ^ { j } \mid k \in \mathbb { Z } _ { \geq 0 } , k \cdot d ^ { j } \leq \ell \} \mid j \in \mathbb { Z } _ { \geq 0 } \} \setminus \{ 0 \}$   
2 Let $I = \{ v _ { i } : i \in K \}$   
3 if $x _ { I } = 0$ then   
4 return “error”   
5 else   
6 return $\left( { \frac { x _ { I \cup \{ u \} } } { x _ { I } } } \right) _ { u \in V _ { \ell + 1 } }$

We first argue that if we iteratively sample a path using only Algorithm 4, then we never encounter an error. Because Algorithm 4 defines a policy that simulates on chance nodes the environment’s probability distribution, the lemma below also applies to running the policy of Algorithm 4 only on decision nodes and sampling successors of chance nodes according to $\mathbf { \nabla } _ { \mathbf { p } } .$

Lemma 11. Let ${ \pmb x } \in Q ^ { ( r ) }$ satisfy (7). If Algorithm $\it 4$ is called with some trajectory $v _ { 1 } , \ldots , v _ { \ell }$ ending in a chance vertex $v _ { \ell } \in C$ and with x, then either $^ { 6 } e r r o r ^ { \prime \prime }$ is returned or $v _ { \ell + 1 }$ is chosen according to $p _ { v _ { \ell } }$

Proof. Assume that no error is returned and therefore $x _ { I } > 0$ where I is as in Algorithm 4. By (7) we have for each $u \in V _ { \ell + 1 }$ that

$$
x _ { I \cup \{ u \} } = x _ { I } \cdot p _ { v _ { \ell } } ( u ) .
$$

Thus, u is chosen with probability $x _ { I \cup \{ u \} } / x _ { I } = p _ { v _ { \ell } } ( u )$

Lemma 12. Let $\pmb { x } \in Q ^ { ( r ) }$ satisfy (7). Suppose we obtain $v _ { 1 } , \ldots , v _ { H }$ by repeatedly applying Algorithm $\it 4$ on the previous vertices and x. Then the policy never outputs $^ { 6 } e r r o r ^ { \prime \prime }$

Proof. In the first iteration $( \ell = 0 )$ we have that $K = I = \emptyset$ . Thus, $x _ { I } = x _ { \emptyset } = 1$ and no error is returned. Suppose now that $v _ { 1 } , \ldots , v _ { \ell }$ have been sampled without an error. We will argue that in the next iteration we also do not return an error. Let K and I be as in the algorithm in the iteration where $v _ { \ell }$ was sampled and $K ^ { \prime }$ and $I ^ { \prime }$ as in the next iteration. Then $x _ { I \cup \{ v _ { \ell } \} } > 0$ , since otherwise $v _ { \ell }$ would have a zero probability of being sampled. Furthermore, $K ^ { \prime } \subseteq K \cup \langle \ell \}$ . It follows that $x _ { I ^ { \prime } } \geq x _ { I \cup \{ v _ { \ell } \} } > 0$ . Thus, also in the next iteration no error is returned. □

In order to use our results from the previous section, we relate the policy to our recursive randomized rounding algorithm.

Lemma 13. Let ${ \pmb x } \in Q ^ { ( r ) }$ satisfying (7).

Let $\mathtt { M D P } ( { \pmb x } )$ be the distribution over layer-spanning paths obtained from repeatedly applying Algorithm $\it 4$ on the previous vertices and x. Then $\mathtt { M D P } ( { \pmb x } ) = \mathtt { R R R } ( { \pmb x } )$

Proof. For a set of layer indices $L \subseteq \{ 1 , \dots , H \}$ , let $\textstyle V [ L ] : = \bigcup _ { i \in L } V _ { i }$ and let $G [ L ]$ be the induced subgraph on $V [ L ]$

Consider the structure of the recursive calls in Algorithm 1. Each recursive call works on a subgraph $G [ L _ { j , h } ]$ where

$$
L _ { j , h } : = \{ ( j - 1 ) d ^ { h } + 1 , ( j - 1 ) d ^ { h } + 2 , \ldots , j d ^ { h } \}
$$

for some $\textit { h } \leq \textit { r }$ and $j \in \{ 1 , \dots , d ^ { r - h } \}$ . The call corresponding to $L _ { j , h }$ recurses on $L _ { j ^ { \prime } , h - 1 }$ for $j ^ { \prime } \in \{ d j - d + 1 , d j - d + 2 , \dots , d j \}$ , in increasing order of $j ^ { \prime } .$ . This corresponds to a partition of $L _ { j , h }$ into d consecutive parts of equal length $d ^ { h - 1 }$

Let $v _ { 1 } \in V _ { 1 } , \dots , v _ { H } \in V _ { H }$ be random variables corresponding to the selected vertices in $\mathtt { R R R } ( { \pmb x } )$ Define

$$
K _ { j , h } : = \{ k _ { j , h } ^ { ( i ) } ~ | ~ i \in \mathbb { Z } _ { \geq 0 } \} \setminus \{ 0 \} , ~ \mathrm { w h e r e } ~ k _ { j , h } ^ { ( i ) } : = \operatorname* { m a x } \{ k \cdot d ^ { i } ~ | ~ k \in \mathbb { Z } _ { \geq 0 } , k \cdot d ^ { i } \leq ( j - 1 ) d ^ { h } \} , \mathrm { a n d }
$$

$$
I _ { j , h } : = \{ v _ { i } \mid i \in K _ { j , h } \} .
$$

We prove inductively that the LP solution we pass to the subproblem on $L _ { j , h }$ is

$$
\pmb { x } ^ { | I _ { j , h } } : = \left( \frac { x _ { I _ { j , h } \cup J } } { x _ { I _ { j , h } } } \right) _ { J \in \binom { V [ L _ { j , h } ] } { \leq h + 1 } } ,
$$

Note that the solution is always projected to variables of vertices in $V [ L _ { j , h } ]$ . For simplicity we omit this projection in the arguments below.

For the root problem $\boldsymbol { L } _ { 1 , r }$ , note that $K _ { 1 , r } = \varnothing = I _ { 1 , r }$ and therefore ${ \pmb x } ^ { | I _ { 1 , r } } = { \pmb x }$ , which is indeed the input of the problem. Consider now some subproblem $L _ { j , h }$ and assume that $\pmb { x } ^ { | I _ { j , h } }$ is its input. Let $j ^ { \prime } \in \{ d j - d + 1 , \ldots , d j \}$

Case 1: $j ^ { \prime } = d j - d + 1$ . Then Algorithm 1 calls subproblem $L _ { j ^ { \prime } , h - 1 }$ with $\pmb { x } ^ { | I _ { j , h } }$ , that is, it does not perform any additional conditioning. Note that

$$
\begin{array} { r l } & { k _ { j ^ { \prime } , h - 1 } ^ { ( i ) } = \operatorname* { m a x } \{ k \cdot d ^ { \textit { i } } \mid k \in \mathbb { Z } _ { \ge 0 } , k \cdot d ^ { i } \le ( j ^ { \prime } - 1 ) d ^ { h - 1 } \} } \\ & { \qquad = \operatorname* { m a x } \{ k \cdot d ^ { i } \mid k \in \mathbb { Z } _ { \ge 0 } , k \cdot d ^ { i } \le ( d j - d ) d ^ { h - 1 } \} } \\ & { \qquad = \operatorname* { m a x } \{ k \cdot d ^ { i } \mid k \in \mathbb { Z } _ { \ge 0 } , k \cdot d ^ { i } \le ( j - 1 ) d ^ { h } \} = k _ { j , h } ^ { ( i ) } . } \end{array}
$$

Thus, $\pmb { x } ^ { | I _ { j ^ { \prime } , h - 1 } } = \pmb { x } ^ { | I _ { j , h } }$ , which proves the induction step for this case.

Case 2: $j ^ { \prime } > d j - d + 1$ . Then Algorithm 1 calls $L _ { j ^ { \prime } , h - 1 }$ with $( { \pmb x } ^ { | I _ { j , h } } ) ^ { | u }$ , where $u = v _ { ( j ^ { \prime } - 1 ) d ^ { h - 1 } }$ is the last vertex selected in the subproblem $L _ { j ^ { \prime } - 1 , h - 1 }$ . Note that $( j - 1 ) d ^ { h } < ( j ^ { \prime } - 1 ) d ^ { h - 1 } < j d ^ { h }$ Thus, for every $i \geq h$ we have

$$
\begin{array} { r l } & { k _ { j ^ { \prime } , h - 1 } ^ { ( i ) } = \operatorname* { m a x } \{ k \cdot d ^ { i } \mid k \in \mathbb { Z } _ { \geq 0 } , k \cdot d ^ { i } \leq ( j ^ { \prime } - 1 ) d ^ { h - 1 } \} } \\ & { \qquad = \operatorname* { m a x } \{ k \cdot d ^ { i } \mid k \in \mathbb { Z } _ { \geq 0 } , k \cdot d ^ { i } \leq ( j - 1 ) d ^ { h } \} = k _ { j , h } ^ { ( i ) } . } \end{array}
$$

Furthermore, for all $i < h$ we have

$$
\begin{array} { r } { k _ { j ^ { \prime } , h - 1 } ^ { ( i ) } = \operatorname* { m a x } \{ k \cdot d ^ { i } \mid k \in { \mathbb Z } _ { \geq 0 } , k \cdot d ^ { i } \leq ( j ^ { \prime } - 1 ) d ^ { h - 1 } \} = ( j ^ { \prime } - 1 ) d ^ { h - 1 } , } \end{array}
$$

which is the index of the layer of u. On the other hand,

$$
k _ { j , h } ^ { ( i ) } = \operatorname* { m a x } \{ k \cdot d ^ { i } \mid k \in \mathbb { Z } _ { \geq 0 } , k \cdot d ^ { i } \leq ( j - 1 ) d ^ { h } \} = ( j - 1 ) d ^ { h } = k _ { j , h } ^ { ( h ) } .
$$

It follows that

$$
\begin{array} { r l } & { K _ { j ^ { \prime } , h - 1 } = \{ k _ { j ^ { \prime } , h - 1 } ^ { ( i ) } \mid i \in \mathbb { Z } _ { \ge 0 } \} \setminus \{ 0 \} } \\ & { \qquad = \{ k _ { j , h } ^ { ( i ) } \mid i \in \mathbb { Z } _ { \ge 0 } \} \setminus \{ 0 \} \cup \{ ( j ^ { \prime } - 1 ) d ^ { h - 1 } \} = K _ { j , h } \cup \{ ( j ^ { \prime } - 1 ) d ^ { h - 1 } \} . } \end{array}
$$

Thus, $I _ { j ^ { \prime } , h - 1 } = I _ { j , h } \cup \{ u \}$ . The induction step now follows from

$$
( { \pmb x } ^ { | I _ { j , h } } ) _ { J } ^ { | { \pmb u } } = \left( \frac { x _ { J \cup \{ { \pmb u } \} } ^ { | I _ { j , h } } } { x _ { { \pmb u } } ^ { | I _ { j , h } } } \right) = \left( \frac { x _ { J \cup I _ { j , h } \cup \{ { \pmb u } \} } / x _ { I _ { j , h } } } { x _ { I _ { j , h } \cup \{ { \pmb u } \} } / x _ { I _ { j , h } } } \right) = x _ { J } ^ { | I _ { j , h } \cup \{ { \pmb u } \} } = { \pmb x } _ { J } ^ { | I _ { j ^ { \prime } , h - 1 } } .
$$

The vertex $v _ { \ell + 1 }$ in a layer $V _ { \ell + 1 }$ is chosen in the deepest recursion and $v _ { 1 } \in V _ { 1 } , \ldots , v _ { H } \in V _ { H }$ are sampled in this order. By the induction above, vertex $v _ { \ell + 1 }$ is chosen according to $\pmb { x } ^ { | I _ { \ell + 1 , 0 } }$ and $K _ { \ell + 1 , 0 } ~ = ~ \{ \{ \operatorname* { m a x } { k \cdot d ^ { i } } ~ | ~ k ~ \in ~ \mathbb { Z } _ { \ge 0 } , k \cdot d ^ { i } ~ \le ~ \ell \} ~ | ~ i ~ \in ~ \mathbb { Z } _ { \ge 0 } \} ~ \backslash ~ \{ ~ 0 \}$ This is therefore identical to Algorithm 4. □

Proof of Theorem 2. Let O be defined as in Lemma 10 and OPT ∼ O. Compute $\pmb { x } \in Q ^ { ( r ) }$ satisfying (7) using Lemma 7. Assume that Lemma 7 is successful, which happens with probability at least $1 - n ^ { - 1 / \varepsilon }$ . Then

$$
\mathbb { E } _ { P \sim \mathtt { R R R } ( \pmb { x } ) } [ f ( P ) ] = \mathbb { E } _ { P \sim \mathtt { M D P } ( \pmb { x } ) } [ f ( P ) ] \ge \frac { 1 - \varepsilon } { ( d - 1 ) ( r + 1 ) } \mathbb { E } [ f ( \mathrm { O P T } ) ] ,
$$

where the equality follows from Lemma 13 and the inequality from Lemma 10 combined with Lemma 7. The number of vertices from the trajectory that a decision depends on is the size of the set we condition on in Algorithm 4, which is at most r. We can set $d = 2$ and $r = O ( \log H )$ , or $d = H ^ { \varepsilon }$ and $r = O ( 1 / \varepsilon )$ to obtain the trade-ofs claimed in the theorem. □

## 4 Conclusion

One key open question is whether the guarantees we obtain with our quasi-polynomial time procedures can also be obtained with polynomial time algorithms. Already for Submodular Orienteering, the question of whether the Recursive Greedy algorithm of Chekuri and Pál [CP05] can be improved to a polynomial time algorithm with similar guarantees is a long-standing open problem. Another natural direction is to get a better understanding of what trade-ofs are possible between the size of a policy and its performance for Submodular MDPs, also from an information-theoretic hardness perspective.

Moreover, we want to highlight that one can further thin out variables of our LP relaxation $Q ^ { ( r ) }$ which we deliberately did not do in order to keep the paper concise and focused. More precisely, the Sherali-Adams type LP relaxation $Q ^ { ( r ) }$ that we introduced has a variable for every subset of vertices of size at most r + 1. It sufices to only consider variables for subsets of vertices that are relevant for later conditioning steps. One can observe that this corresponds to only including variables for sets K as defined in Algorithm 4 together with one extra variable, and subsets thereof. This change makes the notation more cumbersome, and our quasi-polynomial time procedure would still be quasi-polynomial time. One advantage of this change, apart from leading to a smaller LP, is that our rounding would be marginal-preserving for all variables in the LP, whereas with our current LP, we only used the marginal-preserving property for single variables (see Lemma 4 (iii)), which was enough for our purposes.

## References

[BBG96] F. Bacchus, C. Boutilier, and A. Grove. “Rewarding Behaviors”. In: Proceedings of AAAI. AAAI’96. Portland, Oregon: AAAI Press, 1996, pp. 1160–1167. isbn: 0-262- 51091-X.

[BCG09] M. Bateni, M. Charikar, and V. Guruswami. “Maxmin allocation via degree lowerbounded arborescences”. In: Proceedings of STOC. 2009, pp. 543–552. doi: 10.1145/ 1536414.1536488.

[BF19] N. Buchbinder and M. Feldman. “Constrained Submodular Maximization via a Nonsymmetric Technique”. In: Mathematics of Operations Research 44.3 (2019), pp. 988– 1005. doi: 10.1287/moor.2018.0955.

[BF24] N. Buchbinder and M. Feldman. “Constrained Submodular Maximization via New Bounds for DR-Submodular Functions”. In: Proceedings of STOC. 2024, pp. 1820–1831. doi: 10.1145/3618260.3649630.

[BLR26] É. Bamas, S. Li, and L. Rohwedder. “Randomized Rounding over Dynamic Programs”. In: Proceedings of STOC. 2026, pp. 1857–1868. doi: 10.1145/3798129.3800892.

[CCK09] D. Chakrabarty, J. Chuzhoy, and S. Khanna. “On allocating goods to maximize fairness”. In: Proceedings of FOCS. 2009, pp. 107–116. doi: 10.1109/FOCS.2009.51.

[CCPV11] G. Călinescu, C. Chekuri, M. Pál, and J. Vondrák. “Maximizing a Monotone Submodular Function Subject to a Matroid Constraint”. In: SIAM Journal on Computing 40.6 (2011), pp. 1740–1766. doi: 10.1137/080733991.

[CFLP00] R. D. Carr, L. K. Fleischer, V. J. Leung, and C. A. Phillips. “Strengthening integrality gaps for capacitated network design and covering problems”. In: Proceedings of SODA. 2000, pp. 106–115.

[CP05] C. Chekuri and M. Pál. “A Recursive Greedy Algorithm for Walks in Directed Graphs”. In: Proceedings of FOCS. 2005, pp. 245–253. doi: 10.1109/SFCS.2005.9.

[CVZ10] C. Chekuri, J. Vondrák, and R. Zenklusen. “Dependent Randomized Rounding via Exchange Properties of Combinatorial Structures”. In: Proceedings of FOCS. 2010, pp. 575–584. doi: 10.1109/FOCS.2010.60.

[CVZ14] C. Chekuri, J. Vondrák, and R. Zenklusen. “Submodular Function Maximization via the Multilinear Relaxation and Contention Resolution Schemes”. In: SIAM Journal on Computing 43.6 (2014), pp. 1831–1879. doi: 10.1137/110839655.

[DGV08] B. C. Dean, M. X. Goemans, and J. Vondrák. “Approximating the stochastic knapsack problem: The benefit of adaptivity”. In: Mathematics of Operations Research 33.4 (2008), pp. 945–964. doi: 10.1287/moor.1080.0330.

[DPK24] R. De Santi, M. Prajapat, and A. Krause. “Global Reinforcement Learning : Beyond Linear and Convex Rewards via Submodular Semi-gradient Methods”. In: Proceedings of ICML. 2024, pp. 10235–10266.

[EN16] A. Ene and H. L. Nguyen. “Constrained Submodular Maximization: Beyond 1/e”. In: Proceedings of FOCS. 2016, pp. 248–257. doi: 10.1109/FOCS.2016.34.

[FNS11] M. Feldman, J. Naor, and R. Schwartz. “A Unified Continuous Greedy Algorithm for Submodular Maximization”. In: Proceedings of FOCS. 2011, pp. 570–579. doi: 10 . 1109/FOCS.2011.46.

[GLL22] F. Grandoni, B. Laekhanukit, and S. Li. “O(log<sup>2</sup> k/ log log k)-Approximation Algorithm for Directed Steiner Tree: A Tight Quasi-Polynomial Time Algorithm”. In: SIAM Journal Comput. 52.2 (2022), STOC19-298–STOC19-322. issn: 0097-5397. doi: 10.1137/ 20M1312988.

[GMS10] S. Guha, K. Munagala, and P. Shi. “Approximation algorithms for restless bandit problems”. In: Journal of the ACM 58.1 (2010), pp. 1–50. doi: 10.1145/1870103.1870106.

[Guo+22] X. Guo, G. Kortsarz, B. Laekhanukit, S. Li, D. Vaz, and J. Xian. “On approximating degree-bounded network design problems”. In: Algorithmica 84.5 (2022), pp. 1252–1278. doi: 10.1007/s00453-022-00924-0.

[JLLS20] H. Jiang, J. Li, D. Liu, and S. Singla. “Algorithms and adaptivity gaps for stochastic ktsp”. In: Proceedings of ITCS. 2020, 45:1–45:25. doi: 10.4230/LIPIcs.ITCS.2020.45.

[LXZ24] S. Li, C. Xu, and R. Zhang. “Polylogarithmic Approximations for Robust s-t Path”. In: Proceedings of ICALP. 2024, 106:1–106:17. doi: 10.4230/LIPIcs.ICALP.2024.106.

[PMZK24] M. Prajapat, M. Mutný, M. Zeilinger, and A. Krause. “Submodular reinforcement learning”. In: Proceedings of ICLR. 2024.

[Put90] M. L. Puterman. “Chapter 8 Markov decision processes”. In: Stochastic Models. Vol. 2. Handbooks in Operations Research and Management Science. Elsevier, 1990, pp. 331– 434. doi: 10.1016/S0927-0507(05)80172-0.

[SA90] H. D. Sherali and W. P. Adams. “A hierarchy of relaxations between the continuous and convex hull representations for zero-one programming problems”. In: SIAM Journal on Discrete Mathematics 3.3 (1990), pp. 411–430. doi: 10.1137/0403036.

[TGN24] R. Tan, R. Ghuge, and V. Nagarajan. “Informative Path Planning with Limited Adaptivity”. In: Proceedings of AISTATS. 2024, pp. 4006–4014.

[Thi+06] S. Thiébaux, C. Gretton, J. Slaney, D. Price, and F. Kabanza. “Decision-Theoretic Planning with Non-Markovian Rewards”. In: Journal of Artificial Intelligence Research 25 (2006), pp. 17–74. issn: 1076-9757. doi: 10.1613/jair.1676.

[Von08] J. Vondrák. “Optimal approximation for the submodular welfare problem in the value oracle model”. In: Proceedings of STOC. 2008, pp. 67–74. doi: 10 . 1145 / 1374376 . 1374389.

[Wan+20] R. Wang, H. Zhang, D. S. Chaplot, D. Garagić, and R. Salakhutdinov. “Planning with submodular objective functions”. In: arXiv preprint arXiv:2010.11863 (2020). doi: 10.48550/arXiv.2010.11863.

[WV12] M. A. Wiering and M. Van Otterlo. Reinforcement learning. Vol. 12. 3. Springer, 2012, p. 729. doi: 10.1007/978-3-642-27645-3.

## A Combinatorial $O ( n ^ { \varepsilon } )$ -approximation for Submodular Orienteering

Recall that the Recursive Greedy algorithm yields a logarithmic approximation in quasi-polynomial time. Our LP-based method also gives an $O ( n ^ { \varepsilon } )$ -approximation in time $n ^ { O ( 1 / \varepsilon ) }$ . Here, we show that the latter can also be achieved with a simpler combinatorial algorithm. However, this algorithm does not yield the flexibility we get with the LP-based method and therefore, to the best of our knowledge, it cannot be used to obtain the MDP application presented in this paper. To ease the presentation we omit the length constraint in the algorithm below. We comment on how one could integrate it at the end of the section.

The combinatorial algorithm is based on unpublished work of the first author together with Nick Fischer, who kindly gave us permission to include it.

As in the other algorithm, we assume that we are given a layered instance with $H = d ^ { r }$ layers $V _ { 1 } \cup \cdots \cup V _ { H }$ . For the reduction to this case, we refer to the proof of Theorem 1. To simplify recursive calls, we additionally have inputs $S \subseteq V _ { 1 }$ and $T \subseteq V _ { H }$ , and the goal is to find a path from some vertex in S to some vertex in $T ,$ maximizing $f .$

The algorithm uses a combination of recursion and dynamic programming and is given in $\mathrm { { A l g o - } }$ rithm 5. For $d = 2 .$ , it behaves as Recursive Greedy from [CP05]. We use the following notation in the algorithm. For two paths $P , P ^ { \prime }$ such that the last vertex of $P$ has an arc to the first vertex of $P ^ { \prime }$ , we denote by $P \circ P ^ { \prime }$ the concatenation of $P$ and $P ^ { \prime }$ . We write $N ^ { + } ( v )$ as the out-neighborhood of v, that is, all $u \in V$ such that $( v , u ) \in A$ . For some $L \subseteq \{ 1 , 2 , \dots , H \}$ , we write $G [ L ]$ as the induced subgraph of G on vertices $\cup _ { i \in L } V _ { i }$

Lemma 14. Let $G = ( V , A )$ be a layered graph with $V = V _ { 1 } \cup \dots \cup V _ { H } , S \subseteq V _ { 1 } , T \subseteq V _ { H }$ , and $d ^ { r } = H$ . Assume that there is a path from S to T and let OPT be the path maximizing $f ( \mathrm { O P T } )$ . Let P be the path returned by Algorithm 5 on these parameters. Then $P \neq \bot$ and

$$
f ( P ) \ge { \frac { 1 } { ( d - 1 ) ( r + 1 ) } } f ( \mathrm { O P T } ) \ .
$$

Furthermore, the running time is $n ^ { O ( r ) }$

Proof. Note that the degree of the recursion tree is at most $( d - 1 ) n ^ { 2 }$ . The depth of the recursion tree is r. Therefore, the number of nodes of the recursion tree is $( d n ) ^ { O ( r ) } \leq n ^ { O ( r ) }$ . The number of operations in the algorithm (without the recursions) is polynomial. This proves the claimed running time.

Algorithm 5: Recursive dynamic program   
Input: Layered graph $G = ( V , A )$ with $V = V _ { 1 } \cup \cdots \cup V _ { H } , S \subseteq V _ { 1 } , T \subseteq V _ { H } , d ^ { r } = H$   
Output: $S { - } T$ path (or ⊥ if no such path exists).   
1 if $r = 0$ then   
2 return argmax $\{ f ( P ) \mid P = ( v ) , v \in S \cap T \} { \mathrm { ~ o r ~ } } \bot { \mathrm { ~ i f ~ } } S \cap T = \emptyset$   
3 Let $\begin{array} { r } { L _ { i } = \left\{ \frac { ( i - 1 ) H } { d } + 1 , \dotsc , \frac { i H } { d } \right\} } \end{array}$ for all $i \in \{ 1 , 2 , \ldots , d \}$   
4 for $v \in V _ { H / d }$ do   
5 Recursively compute $S \ – v$ path $P _ { v }$ in $G [ L _ { 1 } ]$ with f   
6 $D [ 1 , v ] = P _ { v } $   
7 for $i = 2 , 3 , \ldots , d$ do   
8 Let $D [ i , v ] = \bot$ for all $v \in V _ { i H / d }$   
9 for $u \in V _ { ( i - 1 ) H / d }$ and $v \in V _ { i H / d }$ with $D [ i - 1 , u ] \neq \bot$ do   
10 Recursively compute $N ^ { + } ( \dot { u } ) – v$ path $P _ { u , v }$ in $G [ L _ { i } ]$ with function $f ( \mathbf { \nabla } \cdot | \mathbf { \nabla } D [ i - 1 , u ] )$   
11 if $P _ { u , v } \neq \perp$ then   
12 if $D [ i , v ] = \perp ~ o r ~ f ( D [ i - 1 , u ] \circ P _ { u , v } ) > f ( D [ i , v ] )$ then   
13 $\begin{array} { r l } { \Big | } & { { } \lfloor \ Dot { D [ i , v ] } = D [ i - 1 , u ] \circ P _ { u , v } } \end{array}$   
L   
14 return argmax $\{ f ( P ) \mid P = D [ d , v ] , P \neq \bot , v \in T \}$ or ⊥ if the set is empty

For the approximation guarantee, we argue via induction over r. For $r = 0 ,$ , the algorithm is optimal and since $( d - 1 ) ( r + 1 ) \geq 1$ the claim follows. Now assume that $r \geq 1$ . For $i \in \{ 1 , \ldots , d \}$ ， let $\mathrm { O P T } [ L _ { i } ]$ denote the restriction of OPT to $G [ L _ { i } ]$ , with $L _ { i }$ as in the algorithm, and let $v _ { i }$ be the vertex in the intersection of OPT and $V _ { i H / d }$ . By the induction hypothesis, we have for all $i \in \{ 1 , \ldots , d \}$ that $D [ i , v _ { i } ] \neq \perp$ and

$$
\begin{array} { r } { f ( D [ i , v _ { i } ] ) \geq \left\{ \begin{array} { l l } { \frac { f ( \mathrm { O P T } [ L _ { 1 } ] ) } { ( d - 1 ) r } ~ } & { \mathrm { ~ i f ~ } i = 1 , } \\ { f ( D [ i - 1 , v _ { i - 1 } ] ) + \frac { f ( \mathrm { O P T } [ L _ { i } ] | D [ i - 1 , v _ { i - 1 } ] ) } { ( d - 1 ) r } ~ } & { \mathrm { ~ i f ~ } i \geq 2 . } \end{array} \right. } \end{array}
$$

Note that, in particular, $f ( D [ i , v _ { i } ] )$ is non-decreasing in i. By adding the inequalities over all $i \in \{ 1 , \ldots , d \}$ , we obtain

$$
\begin{array} { l } { f ( D [ d , v _ { d } ] ) \geq \displaystyle \frac { 1 } { ( d - 1 ) r } \left( f ( \mathrm { O P T } [ L _ { 1 } ] ) + \sum _ { i = 2 } ^ { d } f ( \mathrm { O P T } [ L _ { i } ] \mid D [ i - 1 , v _ { i - 1 } ] ) \right) } \\ { \geq \displaystyle \frac { 1 } { ( d - 1 ) r } \left( f ( \mathrm { O P T } [ L _ { 1 } ] ) + \sum _ { i = 2 } ^ { d } [ f ( \mathrm { O P T } [ L _ { i } ] ) - f ( D [ i - 1 , v _ { i - 1 } ] ) ] \right) } \\ { \geq \displaystyle \frac { 1 } { ( d - 1 ) r } \left( f ( \mathrm { O P T } ) - ( d - 1 ) f ( D [ d , v _ { d } ] ) \right) . } \end{array}
$$

By moving the term in $f ( D [ d , v _ { d } ] )$ to the left and multiplying with $r / ( r + 1 )$ we conclude

$$
f ( P ) \geq f ( D [ d , v _ { d } ] ) \geq { \frac { 1 } { ( d - 1 ) ( r + 1 ) } } f ( \mathrm { O P T } ) .
$$

By setting $d = n ^ { \varepsilon }$ and rescaling ε, we obtain the following.

Theorem 15. $F o r \varepsilon > 0$ , there is a combinatorial $O ( n ^ { \varepsilon } )$ -approximation algorithm for Submodular Orienteering (without a length bound) with running time $n ^ { O ( 1 / \varepsilon ) }$

One could integrate a length function in a similar way as in [CP05], resulting in essentially the same result also including a length bound. This works by rounding the objective values and storing for each rounded objective value the solution of lowest length. We omit the details for the sake of simplicity.