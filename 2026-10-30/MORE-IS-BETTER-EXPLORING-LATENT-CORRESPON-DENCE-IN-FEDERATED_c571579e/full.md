# MORE IS BETTER? EXPLORING LATENT CORRESPON-DENCE IN FEDERATED HYPERGRAPH LEARNING

Bo Yan<sup>1</sup>, Siming Zhang<sup>2</sup>, Siqi Li<sup>1</sup>, Hang Yu<sup>2</sup> <sup>1</sup>Tsinghua University, <sup>2</sup>Shanghai University

## ABSTRACT

Federated hypergraph learning with latent cross-client hyperedge correspondence poses fundamental identifiability and computational challenges, as the same local release can correspond to multiple global hypergraph structures. We characterize the tractability of recovering these latent correspondences, showing tractable pairwise assignment but generally intractable multi-client hyperedge matching. Building on this characterization, we derive a cross-entropy decomposition that reveals when local prediction can outperform predictors fitted with additional cross-client information. Motivated by this principle, we introduce Federated Adaptive Spectral Training with Representation Alignment (FedASTRA), which learns correspondence-invariant diffusion representations through adaptive heat diffusion and masked latent prediction while communicating model parameters only. We establish its asymptotic optimality with respect to the permitted local information and prove global population-risk convergence under explicit conditions on recovery, representation, and update alignment. Experiments on eight real-world datasets demonstrate the effectiveness of FedASTRA against identifier-free federated graph and hypergraph methods. On ogbn-papers100M, FedASTRA reaches the performance-communication Pareto frontier, achieving 50.07 Accuracy and 29.73 Macro-F1 while communicating only 59.60 MiB per round.

## 1 INTRODUCTION

Federated learning enables collaborative model training without sharing data records (McMahan et al., 2017). For applications involving group-wise interactions, hypergraphs provide a natural representation of memberships in care teams, coauthor sets, and biochemical interactions (Feng et al., 2019; Chien et al., 2022; Wang et al., 2023). However, when hypergraph data are anonymized across clients, local memberships may remain observable while their cross-client correspondence is lost, making the underlying global hypergraph potentially non-identifiable from local observations alone. Consider two hospitals that share care episodes, as illustrated in Figure 1. After episode identifiers are removed, Hospital A retains anonymous records of backache and BP/ECG, while Hospital B retains records of surgery and headache/drug. Same-order matching yields compatible episodes of sizes two and four, whereas cross-order matching yields two episodes of size three. Thus, even this minimal two-client example admits distinct global compatible structures consistent with the same local observations, raising a natural question: which global hypergraph properties remain identifiable under latent hyperedge correspondence?

We answer this question by establishing sharp identification bounds and computational boundaries under latent hyperedge correspondence. For two clients, the compatible global structures reduce to bipartite assignments, yielding sharp identification bounds and a double-centering criterion for point identification. Beyond two clients, correspondence becomes a multiway hypergraph-matching problem. It is NP-complete for every fixed client count above two, yet fixed-parameter tractable jointly in client count and matching size. When the client count is unbounded, it becomes W[1]-complete when parameterized only by the matching size.

This characterization raises another question: when does additional cross-client information improve prediction, and when can exploiting it be detrimental? Additional information may provide useful label signal, but uncertain correspondence can also increase prediction variability and fitting error. We decompose cross-entropy risk into information gain and fitting error, yielding a sufficient condition under which a local predictor outperforms a fitted predictor with additional information under the same evaluation law.

![](images/1b376d10a7649c7eca856ac3bd581a9c4317142938cd30e4f194636540258345.jpg)  
Figure 1: From complete hyperedges to latent hyperedge correspondences. (a) A complete care episode is represented as a hyperedge connecting its clinical events, e.g., blood pressure (BP), electrocardiogram (ECG), and surgery. (b) A episode may span multiple hospitals, e.g., BP measurement and ECG examination may be performed at hospital A, while the subsequent surgery is carried out at hospital B. Each hospital observes only a local fragment of the original hyperedge, leading to higher-order truncation. (c) Once cross-hospital record correspondences are latent, the same local releases can admit multiple compatible global matchings, resulting in hyperedge aliasing.

Motivated by this tradeoff, together with the communication and privacy costs of richer cross-client exchanges (Shi et al., 2025), we introduce FedASTRA, which learns correspondence-invariant local diffusion representations through adaptive heat filtering and masked latent prediction (LeCun, 2022; Assran et al., 2023), with a client-local exponential moving-average (EMA) target and parameteronly FedAvg (McMahan et al., 2017). We show that effectively exploiting the task-relevant local information can be sufficient without resolving latent cross-client correspondences or completing the global hypergraph, yielding asymptotic optimality relative to the permitted local information. Under explicit representation and update-alignment conditions, we further establish global population-risk convergence along the implemented federated updates.

Our contributions are fourfold. First, we establish sharp identification bounds and a computational hierarchy for latent hyperedge correspondence: polynomial-time two-client assignment, NP-complete yet fixed-parameter tractable matching for any fixed client count above two, and W[1]-complete matching when the client count is unbounded. Second, we characterize when additional cross-client information helps or hurts prediction through a cross-entropy decomposition that connects information gain, fitting error, prediction variation, and assignment geometry. Third, we develop FedASTRA and establish asymptotic optimality relative to the permitted local information, together with global population-risk convergence under explicit representation and update-alignment conditions. Fourth, experiments on eight real-world datasets demonstrate consistent improvements over identifier-free federated graph and hypergraph methods; on ogbn-papers100M, FedASTRA reaches the performancecommunication Pareto frontier.

## 2 RELATED WORK

Federated graph learning. Existing federated graph learning methods differ in whether crossclient graph correspondence is available. Identifier-free methods leave cross-client node identities latent (Zhang et al., 2021; Baek et al., 2023; Li et al., 2023; Fu et al., 2025), as in FedSage (Zhang et al., 2021) operating on local subgraphs, while FedSage+ (Zhang et al., 2021) generates crossclient neighbor features. Methods with explicit correspondence, such as FedGCN (Yao et al., 2023) and FedCog (Lei et al., 2023), exploit shared cross-client node or edge identities to incorporate additional graph information. For hypergraph, FedHGL (Luo et al., 2025) instead assumes shared hyperedge identifiers and utilzes local differential privacy (LDP) (Kasiviswanathan et al., 2011) to support structure completion. Despite progress, these methods trade cross-client communication and privacy exposure for predictive performance. In contrast, we study an identifier-free setting in which cross-client hyperedge correspondence is latent and only local incidence is available.

Hypergraph representation learning. Most hypergraph neural networks assume observed nodehyperedge memberships and focus on how to encode higher-order structure, including spectral propagation on hypergraph operators (Feng et al., 2019; Wang et al., 2023), graph-reduction approaches that approximate hyperedges with pairwise structures (Yadati et al., 2019), and direct higher-order aggregation through message passing or permutation-invariant set functions (Dong et al., 2020; Chien et al., 2022; Huang & Yang, 2021). Beyond supervised learning, self-supervised methods learn transferable hypergraph representations through multi-level contrastive learning (Lee & Shin, 2023) and generative pretext tasks (Kim et al., 2024). More recently, large-scale hypergraph pretraining and foundation models have emerged for cross-task and cross-domain transfer Gao et al. (2026). FedASTRA targets a more realistic federated scenario by learning effective representations from anonymous client-local incidence when cross-client hyperedge correspondence remains latent.

## 3 LEARNING DIFFICULTY UNDER LATENT HYPEREDGE CORRESPONDENCE

## 3.1 OBSERVATION MODEL

Let $\mathcal { H } = ( V , \mathcal { E } , X , r , w )$ be a finite node-attributed, typed, weighted multihypergraph. Here � is the node set, E is the multiset of hyperedge instances, � is the node-feature matrix, and � and � collect hyperedge types and weights. hyperedge instance $e \in { \mathcal { E } }$ has a support node set $A _ { e } \subseteq V .$ , a type $r _ { e }$ in the finite type set R, and a positive weight $w _ { e }$ . Multiple instances may share the same support and type, and a type label carries no instance identity. For an integer $1 \leq q \leq | V |$ , let $\binom { V } { q }$ be the collection of �-node subsets. Aggregating all hyperedge instances with the same support � and type �, we define

$$
w _ { r } ( A ) = \sum _ { e : A _ { e } = A , r _ { e } = r } w _ { e } ,\tag{1}
$$

$$
M _ { q , r } ^ { \mathcal { H } } ( S ) = \sum _ { A \supset S } w _ { r } ( A ) , \qquad r \in \mathcal { R } , \quad S \in \binom { V } { q } .\tag{2}
$$

Thus $M _ { q , r } ^ { \mathcal { H } } ( S )$ is the weighted inclusion count of the �-node subset � among hyperedges of type �. The full collection $\{ M _ { q , r } ^ { \mathcal { H } } ( S ) \}$ is the invertible Boolean-zeta transform of the aggregate weights $\{ w _ { r } ( A ) \}$ In particular, the $\dot { q ^ { \mathrm { ~ } } } = 2$ and $q = 3$ coordinates capture observable pairwise and triple co-membership patterns before client partitioning. The notation collection is provided in Appendix B.1.

A fixed node-to-client assignment $z : V \to [ K ]$ , with $k \in [ K ] : = \{ 1 , . . . , K \}$ , partitions � into local node set $V _ { k } = \{ \nu \in V : z ( \nu ) = k \}$ for each client k and induces local hyperedge fragments $F _ { e , k } = A _ { e } \cap V _ { k }$ . At a visibility threshold $c \geq 1$ , client � retains each fragment $F _ { e , k }$ with $| F _ { e , k } | \ge c ,$ together with its type $r _ { e } .$ , weight $w _ { e } ,$ and a status indicator $\upsilon _ { e , k } \in \{ 0 , 1 \}$ specifying whether hyperedge � is fully contained in client �. Crucially, cross-client hyperedge identifiers are withheld. Let Γ collect the public assumptions governing the observation process, including the visibility rule and any exact constraints on the number of retained fragments. Write $[ T _ { k } ] _ { \mathrm { r o w } }$ for the equivalence class of client $k ' \mathrm { s }$ retained fragment table under arbitrary row permutations. The analyst release is then defined by the deterministic observation map:

$$
\begin{array} { r } { Y _ { \mathrm { A } } = O _ { z , \Gamma } ( \mathcal { H } ) = \left( \left( ( V _ { k } , X _ { V _ { k } } ) , \big [ ( F _ { e , k } , r _ { e } , w _ { e } , \nu _ { e , k } ) : | F _ { e , k } | \geq c \big ] _ { \mathrm { r o w } } \right) _ { k = 1 } ^ { K } , \Gamma \right) . } \end{array}\tag{3}
$$

Uppercase $Y _ { \mathrm { A } }$ denotes the analyst release as a random element and lowercase $\mathrm { y _ { A } }$ a realized release. Within the latent hypergraph class $M _ { \Gamma }$ permitted by the public assumptions Γ, define the analyst compatibility class

$$
\mathcal { F } _ { \mathrm { A } } ( y _ { \mathrm { A } } , \Gamma ) = \{ \mathcal { H } ^ { \prime } \in M _ { \Gamma } : O _ { z , \Gamma } ( \mathcal { H } ^ { \prime } ) = y _ { \mathrm { A } } \} .\tag{4}
$$

In contrast to the richer analyst release, the learner, representing the actual training algorithm, receives a coarsened release. Accordingly, let $B _ { k }$ be the binary node-fragment incidence matrix induced by $T _ { k }$ , with one column for each row of $T _ { k }$ . Define $[ B _ { k } \bar { ] } _ { \mathrm { c o l } } = \{ B _ { k } \bar { \Pi } :$ Π is a column permutation} and thus arbitrary row permutations of $T _ { k }$ correspond to column permutations $B _ { k } \mapsto B _ { k } \Pi$ . The learner coarsened release is therefore

$$
\begin{array} { r } { Y _ { \mathrm { L } } = \rho ( Y _ { \mathrm { A } } ) = \big ( ( X _ { k } , [ B _ { k } ] _ { \mathrm { c o l } } ) \big ) _ { k = 1 } ^ { K } , \qquad X _ { k } : = X _ { V _ { k } } . } \end{array}\tag{5}
$$

It retains only local node features and anonymous incidence structure, omitting hyperedge types, weights, status indicators, and row identities. For a fixed training protocol $\mathcal { P }$ containing the released labels, task split, participation schedule, and hyperparameters, and algorithmic randomness $\xi ,$ , any learner operating under this observation model is a measurable function of $( Y _ { \mathrm { L } } , { \mathcal { P } } , \xi )$ . In native comparisons, the same realized release is provided to all learners.

For each �-node subset �, the client partition determines whether its co-membership information is retained. For $q \geq c ,$ , we define

$$
\widetilde { M } _ { q , r } ( S , z ) = M _ { q , r } ^ { \mathcal { H } } ( S ) \mathbf { 1 } \{ z \mathrm { ~ i s ~ c o n s t a n t ~ o n ~ } S \} ,\tag{6}
$$

so the contribution of � is retained exactly when all nodes in � are assigned to the same client. We quantify the erased fraction by

$$
\tau _ { q } ( \mathcal { H } , z ) = 1 - \frac { \sum _ { r \in \mathcal { R } } \| \widetilde { M } _ { q , r } ( \cdot , z ) \| _ { 1 } } { \sum _ { r \in \mathcal { R } } \| M _ { q , r } ^ { \mathcal { H } } \| _ { 1 } } ,\tag{7}
$$

with $\tau _ { q } ( \mathcal { H } , z ) = 0$ for a zero denominator. We call the erased mass higher-order truncation, as illustrated in Figure 1. Independently permuted client fragment tables may also admit multiple compatible cross-client matchings. We call this ambiguity hyperedge aliasing. These two effects can alter global higher-order statistics. We therefore consider a class of linear co-membership targets $\Phi _ { q , \psi }$ . For any finite type-indexed test function $\psi _ { q , r } ,$ define

$$
\Phi _ { q , \psi } ( \mathcal { H } ) = \sum _ { e \in \mathcal { E } } w _ { e } \sum _ { S \in \binom { A _ { e } } { q } } \psi _ { q , r _ { e } } ( S , X _ { S } ) .\tag{8}
$$

Here, $\psi _ { q , r } ( S , X _ { S } )$ specifies the contribution of each �-node subset �, allowing $\Phi _ { q , \psi }$ to cover both total order-� co-membership mass and more general feature-dependent higher-order statistics.

Equation (7) quantifies the co-membership mass loss under a fixed partition. Appendices $\mathrm { B } . 2 , \mathrm { C } . 5 -$ C.6 give the Boolean-zeta inversion and show how the co-membership quantities can be estimated under a randomized partition using Horvitz-Thompson estimation (Horvitz & Thompson, 1952).

## 3.2 EXACT TWO-CLIENT COMPATIBILITY SETS

Two clients form the smallest nontrivial setting with an exact correspondence analysis. We use the richer analyst release with observed status indicators $\upsilon _ { e , k }$ , and exact type � and weight �, together with $c = 1$ and equal numbers of retained fragment rows across the two clients. Thus, each crossclient hyperedge leaves one nonempty row per client. Let $m _ { h }$ denote the common row count in stratum $h = \left( r , \omega \right)$ . Within stratum $h ,$ fragments $F _ { 1 i } ^ { h } \subseteq V _ { 1 }$ and $F _ { 2 j } ^ { h } \subseteq V _ { 2 }$ , indexed by $i , j \in [ m _ { h } ]$ are matched by an unobserved $\varpi _ { h } \in \mathrm { S y m } ( m _ { h } )$ , the symmetric group on $[ m _ { h } ]$ . Let $\Phi _ { \mathrm { o b s } }$ collect the client-local and within-fragment contributions that are invariant across all compatible matchings. For a candidate cross-client pair $( F _ { 1 i } ^ { h } , F _ { 2 j } ^ { h } )$ , the mixed matching-dependent contribution is

$$
C _ { i j } ^ { h } = \omega \sum _ { S \in ( \Gamma _ { 1 i } ^ { h } \cup F _ { 2 j } ^ { h } ) : \atop { q } } \psi _ { q , r } ( S , X _ { S } ) .\tag{9}
$$

The cost matrix $C ^ { h }$ exposes exactly which part of the target $\Phi _ { q , \psi }$ depends on the missing correspondence. Define the identified set of $\Phi _ { q , \psi }$ under the analyst release as its attainable values over all compatible latent hypergraphs: $\mathcal { T } _ { q , \psi } ( \dot { y } _ { \mathrm { A } } ) = \{ \Phi _ { q , \psi } ( \mathcal { H } ^ { \prime } ) : \mathcal { H } ^ { \prime } \in \mathcal { F } _ { \mathrm { A } } ( y _ { \mathrm { A } } , \Gamma ) \}$

Theorem 1 (Compatible target values under an unobserved permutation). Assume $K = 2 , c = 1$ observed status indicators, exactly one nonempty crossfragment at each clientfor every cross-client hyperedge, exact type-and-weight strata, and equal row counts within every stratum. Then the compatible values and their sharp endpoints are

$$
\begin{array} { r } { \bar { Z } _ { q , \psi } ( \mathfrak { y } _ { \mathrm { A } } ) = \left\{ \Phi _ { \mathrm { o b s } } + \displaystyle \sum _ { h } \sum _ { i = 1 } ^ { m _ { h } } C _ { i , \varpi _ { h } ( i ) } ^ { h } : \varpi _ { h } \in \mathrm { S y m } ( m _ { h } ) \right\} , } \end{array}\tag{10}
$$

$$
\Phi _ { q , \psi } ^ { - } = \Phi _ { \mathrm { o b s } } + \sum _ { h } \operatorname* { m i n } _ { \Lambda \in \mathbb { B } _ { m _ { h } } } \langle C ^ { h } , \Lambda \rangle _ { F } , \quad \Phi _ { q , \psi } ^ { + } = \Phi _ { \mathrm { o b s } } + \sum _ { h } \operatorname* { m a x } _ { \Lambda \in \mathbb { B } _ { m _ { h } } } \langle C ^ { h } , \Lambda \rangle _ { F } ,\tag{11}
$$

where $\mathbb { B } _ { m } = \left\{ \boldsymbol { \Lambda } \in \mathbb { R } _ { + } ^ { m \times m } : \boldsymbol { \Lambda } \mathbf { 1 } = \mathbf { 1 } , \boldsymbol { \Lambda } ^ { \top } \mathbf { 1 } = \mathbf { 1 } \right\}$ is the Birkhoffpolytope and $\langle A , B \rangle _ { F } = \operatorname { t r } ( A ^ { \top } B )$ is the Frobenius inner product. Each condition below characterizes point identification within a stratum, independently ofthe probability distribution over matchings compatible with the observed release:

$$
J _ { m _ { h } } C ^ { h } J _ { m _ { h } } = 0 ,\tag{12}
$$

$$
C _ { i j } ^ { h } = \alpha _ { i } ^ { h } + \beta _ { j } ^ { h } ,\tag{13}
$$

$$
C _ { i j } ^ { h } + C _ { i ^ { \prime } j ^ { \prime } } ^ { h } - C _ { i j ^ { \prime } } ^ { h } - \bar { C _ { i ^ { \prime } j } ^ { h } } = 0 \quad \bar { f o r a l l i , i ^ { \prime } , j , j ^ { \prime } \in [ m _ { h } ] } ,\tag{14}
$$

with centering matrix $J _ { m } = I _ { m } - \mathbf { 1 1 } ^ { \top } ,$ /� and vectors $\alpha ^ { h } , \beta ^ { h } \in \mathbb { R } ^ { m _ { h } }$ in Equation (13).

Each compatible assignment selects one entry from every row and column of $C ^ { h }$ , thereby forming a permutation matrix. By Birkhoff-von Neumann integrality Birkhoff $( 1 9 4 6 )$ , both endpoints are attainable. Double centering in Equation (12) removes additive row and column effects, leaving only the interaction that can vary with the latent correspondence, in a subspace of dimension at most $( \widehat { m _ { h } } - 1 ) ^ { 2 }$ . Equation (14) checks whether any such interaction remains. Appendix C.1 gives the proof and dual certificates. Appendix B.3 further discusses weaker observation assumptions.

Coarsening the analyst release enlarges the compatible class. Hence, if $\phi _ { - }$ and $\phi _ { + }$ are attainable endpoints under the richer release, any randomized scalar estimator using the coarser learner release has worst-case absolute risk at least $\displaystyle \big ( \phi _ { + } - \phi _ { - } \big ) / 2$ (Appendix C.7). This diagnostic uses two clients, while model training uses $K \in \{ 3 , 6 , 9 \}$ and correspondence-invariance result holds for any $K .$ Prediction can still be accurate when the local label signal survives unresolved global structure.

Graph matching versus hypergraph matching. Total order-� mass admits sharp sorting bounds in $\begin{array} { r } { O ( \sum _ { h } m _ { h } \log m _ { h } ) } \end{array}$ time (Appendices $C . 2 \mathrm { - } C . 3 )$ . To isolate relation size, consider � clients with � distinct singleton rows each and explicitly listed rewarded tuples $\mathcal { E } _ { + } \subseteq [ m ] ^ { K }$ . Each listed tuple has reward one, all others zero. A completion with at least � rewarded relations exists exactly when these candidate �-node hyperedges contain a matching of size �. For $K = 2$ , this is polynomial bipartite graph matching. For fixed $\bar { K } \ge 3$ , it is complete for nondeterministic polynomial time (NP-complete) but fixed-parameter tractable (FPT) in $K b$ , meaning time $f ( K b ) \mathrm { p o l y } ( L _ { \mathrm { i n } } )$ for input length $L _ { \mathrm { i n } }$ . With unbounded $K ,$ it is complete for $\mathrm { { W } [ l ] } .$ the first level of the W hierarchy, in � (Appendix $\mathrm { C . 4 } )$ . This separates pairwise from multiway correspondence under the same observation and reward rules, independently of the optimization difficulty of a neural classifier.

## 4 BENEFIT BOUNDS OF ADDITIONAL INFORMATION

Given these fundamental identifiability and computational challenges under hyperedge correspondence, a natural thought is to exploit more information beyond local observations for better prediction. Yet a subtle tradeoff arises here: additional information can reveal label signal, while correspondence uncertainty can induce prediction variation and fitting error. In other words, its benefit is bounded. In this section, we characterize these benefit bounds through cross-entropy decomposition.

Information value and fitting cost. For class $y \in [ C ] = \{ 1 , \ldots , C \} , C \geq 2$ , predictor $F$ uses $\quad s _ { 0 } { \mathrm { : } }$ : client attributes, anonymous incidence, and the parameter-only transcript through validation, excluding held-out labels. Predictor � additionally uses $A _ { \mathrm { a u x } }$ and its derived training states, giving $S _ { 1 } = S _ { 0 } \vee \sigma ( A _ { \mathrm { a u x } } )$ . Let $p _ { i } = \mathrm { P r } ( y = \cdot \mid S _ { i } ) , i = 0 , 1$ , and $q _ { F } , q _ { G }$ be their predictions. Kullback– Leibler (KL) divergence $\begin{array} { r } { D _ { \mathrm { K L } } ( p | | q ) = \sum _ { c } p _ { c } \log ( p _ { c } / q _ { c } ) } \end{array}$ yields the information and fitting terms

$$
\begin{array} { r } { \boldsymbol { J } _ { \mathrm { { a d d } } } = \mathbb { E } \boldsymbol { D } _ { \mathrm { K L } } ( p _ { 1 } \| p _ { 0 } ) , \qquad \boldsymbol { \mathcal { E } } _ { F } = \mathbb { E } \boldsymbol { D } _ { \mathrm { K L } } ( p _ { 0 } \| \boldsymbol { q } _ { F } ) , \qquad \boldsymbol { \mathcal { E } } _ { G } = \mathbb { E } \boldsymbol { D } _ { \mathrm { K L } } ( p _ { 1 } \| \boldsymbol { q } _ { G } ) , } \end{array}\tag{15}
$$

Then $\mathcal { I } _ { \mathrm { a d d } }$ can measure new label information and $\boldsymbol { \mathcal { E } } _ { F } , \boldsymbol { \mathcal { E } } _ { G }$ measure fitting error to the respective optima. For measurable predictions with finite log risk $\mathcal { R } _ { \mathrm { l o g } } ( q ) = \mathbb { E } [ - \log q _ { \mathrm { y } } ]$ , conditional entropy yields the exact risk decomposition below, whose derivation is given in Appendix C.10:

$$
\begin{array} { r l } { \mathcal { R } _ { \mathrm { l o g } } ( q _ { G } ) - \mathcal { R } _ { \mathrm { l o g } } ( q _ { F } ) = \mathcal { } { } } & { { } \underbrace { \mathcal { E } _ { G } - \mathcal { E } _ { F } } _ { \mathrm { ~  ~  ~ \frac { ~ \partial ~ } { ~ \ t ~ } ~  ~ } } \mathrm { ~ \ ~ - ~ } \mathcal { } { ~ \underbrace { ~ \mathcal { I } _ { \mathrm { a d d } } ~ } } \mathrm { ~ \ ~ \ ~ - ~ } } \end{array}\tag{16}
$$

difference in fitting error additional label information

Thus � has strictly lower risk exactly when $\mathcal { E } _ { G } - \mathcal { E } _ { F } > \mathcal { I } _ { \mathrm { a d d } }$ , with the reverse strict inequality favoring �. Additional information weakly improves ideal risk because it can be ignored (Blackwell, 1953). For fitted models, however, its gain competes with approximation, estimation, and optimization errors. When $p _ { 1 } = p _ { 0 }$ , fitting quality alone determines which predictor has lower risk.

How correspondence creates a prediction cost. To quantify this cost, remove the class mean from enriched logits $s _ { G } \in \mathbb { R } ^ { C }$ and set $\bar { s } _ { G } = \mathbb { E } [ s _ { G } \mid S _ { 0 } ]$ . For cross-entropy risk $\mathcal { R } _ { \mathrm { C E } } ( s ) = \mathcal { R } _ { \mathrm { l o g } }$ (softmax(�)) and finite-risk local logits $s _ { F } ,$ , define $V _ { \mathrm { a u x } } = \mathbb { E } \Vert s _ { G } - \bar { s } _ { G } \Vert _ { 2 } ^ { 2 } , \Delta _ { \mathrm { l a b e l } } = \mathbb { E } \Vert p _ { 1 } - p _ { 0 } \Vert _ { 2 } ^ { 2 }$ , and $\delta _ { F } = \mathcal { R } _ { \mathrm { C E } } ( s _ { F } ) -$ ${ \mathcal { R } } _ { \mathrm { C E } } ( { \bar { s } } _ { G } )$ . Under $\| s _ { G } \| _ { \infty } \le B _ { \mathrm { l o g i t } } < \infty$ almost surely, Theorem 6 gives

$$
\frac { e ^ { - 2 B _ { \mathrm { l o g i t } } } } { 2 C } V _ { \mathrm { a u x } } > \sqrt { \Delta _ { \mathrm { l a b e l } } V _ { \mathrm { a u x } } } + \delta _ { F } \quad \Longrightarrow \quad \mathcal { R } _ { \mathrm { C E } } ( s _ { G } ) > \mathcal { R } _ { \mathrm { C E } } ( s _ { F } ) .\tag{17}
$$

Hence local prediction has lower risk when variation cost exceeds the label-alignment bound plus the signed fitting gap. The criterion is sufficient, since Cauchy–Schwarz bounds label alignment. When $\Delta _ { \mathrm { l a b e l } } = 0$ and $\delta _ { F } \leq 0 .$ , any $V _ { \mathrm { a u x } } > 0$ certifies a strict local advantage. Moreover, Proposition 5 computes this variation from the double-centered matching costs of Theorem 1 for affine logits under independent uniform matchings, linking the identification result to this prediction-risk comparison.

Communication and privacy boundary. Additional exchanges incur communication costs and can expose sensitive information (Shi et al., 2025). Parameter-only exchange avoids these auxiliary disclosures, although model updates still require protection against information leakage.

## 5 FEDASTRA: LEARNING FROM LOCAL STRUCTURE

Section 4 yields a design principle: local prediction can outperform a fitted richer-information model when its fitting cost exceeds the label-information gain. Guided by this principle, FEDASTRA stays within the observable local release: it learns diffusion scales from anonymous incidence, predicts masked representations, and communicates parameters only, eliminating the need for cross-client pairing. Under probabilistic recovery and approximation conditions, FedASTRA is asymptotically optimal relative to permitted local information. Under representation and task-alignment conditions, its actual updates achieve global population-risk convergence with an $O ( 1 / T )$ optimization term.

Learning diffusion scales. For binary incidence $B _ { k }$ with nonempty columns, let $D _ { \nu , k } = \mathrm { d i a g } ( B _ { k } { \bf 1 } )$ $D _ { e , k } = \bar { \mathrm { d i a g } } ( B _ { k } ^ { \top } { \bf 1 } )$ , and $I _ { 0 , k } = \mathrm { d i a g } ( { \bf 1 } \{ ( D _ { \nu , k } ) _ { \nu \nu } = 0 \} )$ . Using the Moore–Penrose inverse †, define the node-to-node transition on these observed local memberships by

$$
P _ { k } = D _ { \nu , k } ^ { \dagger } B _ { k } D _ { e , k } ^ { - 1 } B _ { k } ^ { \top } + I _ { 0 , k } .\tag{18}
$$

This walk averages over incident relations and their members, leaving isolated nodes fixed (Zhou et al., 2006). For latent width $d ,$ encode $U _ { k } = E _ { \theta _ { \mathrm { e n c } } } ( X _ { k } ) \in \mathbb { R } ^ { | V _ { k } | \times d }$ and use $R \geq 0$ positive-time channels with increments $\eta \in \mathbb { R } ^ { R \times d }$ . The context parameters $\vartheta = \operatorname { c t x } ( \theta ) = ( \theta _ { \mathrm { e n c } } , \eta )$ belong to the online state �. The positive floor $\epsilon _ { \mathrm { o r d } } > 0$ orders each coordinate’s diffusion times:

$$
{ \sf K } _ { k } ( t ) = \exp \{ t ( P _ { k } - I ) \} = \sum _ { \ell = 0 } ^ { \infty } e ^ { - t } \frac { t ^ { \ell } } { \ell ! } P _ { k } ^ { \ell } , \qquad t \ge 0 ,\tag{19}
$$

$$
t _ { b j } = \sum _ { a = 1 } ^ { b } \{ \mathrm { s o f t p l u s } ( \eta _ { a j } ) + \epsilon _ { \mathrm { o r d } } \} , \qquad b \in [ R ] , \quad j \in [ d ] ,\tag{20}
$$

$$
[ \mathsf { K } _ { k } ( t _ { b } ) U _ { k } ] _ { : j } : = \mathsf { K } _ { k } ( t _ { b j } ) U _ { k , : j } , \qquad W _ { k } ( X _ { k } \mid \vartheta ) = [ U _ { k } , \mathsf { K } _ { k } ( t _ { 1 } ) U _ { k } , \ldots , \mathsf { K } _ { k } ( t _ { R } ) U _ { k } ] .\tag{21}
$$

Each channel mixes walk lengths with Poisson weights (Kondor & Lafferty, 2002), while $R = 0$ retains only $U _ { k }$ . Each $t _ { b j }$ is the expected walk length for coordinate �. Learning these times adapts the extent of local mixing, while the undiffused channel retains the original encoded attributes. Thus � controls scales, independently of observation order �. Write $W _ { k , \theta } ( { \bar { X } } ) = W _ { k } ( X \mid \operatorname { c t x } ( \theta ) )$ . Column relabeling preserves both features and coupled training updates. Appendices $\mathrm { C . 8 - C . 9 }$ prove stability and give a shared recurrence with Poisson-tail error control for each learned diffusion time.

Local prediction and federation. Two masked views predict a full-view exponential moving average (EMA) target through $\mathcal { L } _ { \mathrm { p r e d } , \mathrm { k } }$ , combining cosine alignment, variance–covariance regularization, and view agreement (Appendix D.1). For classifier $c _ { \theta _ { k } } : \mathbb { R } ^ { ( R + 1 ) d }  \mathbb { R } ^ { C }$ , context $\vartheta _ { k } = \cos ( \theta _ { k } )$ labeled set $\bar { \mathcal { V } } _ { k } ^ { \mathrm { t r } }$ of size $n _ { k } ^ { \mathrm { t r } } > 0$ , and predictive weight $\lambda _ { \mathrm { p r e d } } \geq 0$ , the local training objective is

$$
\mathcal { L } _ { k } = ( n _ { k } ^ { \mathrm { t r } } ) ^ { - 1 } \sum _ { \nu \in \mathcal { V } _ { k } ^ { \mathrm { t r } } } \mathrm { C E } ( c _ { \theta _ { k } } ( W _ { k } ( X _ { k } \mid \vartheta _ { k } ) _ { \nu } ) , y _ { \nu } ) + \lambda _ { \mathrm { p r e d } } \mathcal { L } _ { \mathrm { p r e d } , k } .\tag{22}
$$

```latex
Algorithm 1 FedASTRA with an ordered client-local diffusion basis
Require: Rounds �, steps $S _ { \mathrm { l o c } } ,$ clients $C _ { u } \subseteq [ K ]$ , counts $n _ { k } ^ { \mathrm { t r } } ,$ , weight �<sub>pred</sub>, momentum $\mu .$
1: Initialize $\theta ^ { 0 }$ and client-local operators $P _ { k }$
2: for $u = 0 , \ldots , T - 1$ do
3: for $k \in C _ { u }$ in parallel do
4: Set $\theta _ { k } \gets \bar { \theta ^ { u } }$ and $\begin{array} { r } { \bar { \vartheta } _ { k } \gets \mathrm { c t x } ( \theta _ { k } ) . } \end{array}$
5: Repeat $S _ { \mathrm { l o c } }$ times: update $\theta _ { k }$ on Equation (22), then set $\bar { \vartheta } _ { k } \gets \mu \bar { \vartheta } _ { k } + ( 1 - \mu ) \mathrm { c t x } ( \theta _ { k } )$
6: Return $\theta _ { k } ^ { u + 1 } \gets \theta _ { k }$ and discard the target.
7: end for
8: $\begin{array} { r } { \theta ^ { u + 1 }  \frac { \sum _ { k \in C _ { u } } n _ { k } ^ { \mathrm { t r } } \theta _ { k } ^ { u + 1 } } { \Gamma } } \end{array}$
$\textstyle \sum _ { k \in C _ { u } } n _ { k } ^ { \mathrm { t r } }$
9: end for
10: Select one validation checkpoint and evaluate test data once.
```

Here CE is cross-entropy for $y _ { \nu } \in [ C ]$ . Algorithm 1 updates target $\bar { \vartheta } _ { k }$ with momentum $\mu \in [ 0 , 1 )$ and transmits only online parameters, weighted by each client’s number of labeled training nodes.

Local information and optimal prediction. Under the rank, attribute, and incidence-recovery conditions of Theorem 7 in Appendix $C . 1 1 .$ , the optimal cross-entropy gap between the complete heat representation and local information is $O ( \varepsilon _ { n } \log ( 1 / \varepsilon _ { n } ) )$ , where $\varepsilon _ { n } \to 0$ bounds joint failure on � nodes. On successful recovery, the complete heat representation determines the permitted local release. The residual risk bound accounts for decoder failure through its binary entropy and the finite label alphabet. Thus any local prediction rule can be reproduced on the event of successful recovery.

From client risks to federated risk. Let $w _ { k } ^ { \mathrm { p o p } }$ be test-client probabilities and $\mathcal { R } _ { \mathrm { f e d } }$ their weighted cross-entropy. For the model $\widehat { \theta }$ selected after � rounds on $\begin{array} { r } { N = \sum _ { k } n _ { k } ^ { \mathrm { t r } } } \end{array}$ labels, Theorem 8 gives

$$
\mathbb { E } \Bigg [ \mathcal { R } _ { \mathrm { f e d } } ( \widehat { \theta } ) - \sum _ { k } w _ { k } ^ { \mathrm { p o p } } \mathcal { R } _ { \mathrm { l o c } , k } ^ { \ast } \Bigg ] \leq \mathbb { E } \Bigg [ \sum _ { k } w _ { k } ^ { \mathrm { p o p } } ( d _ { k } + a _ { k } ) + a _ { \mathrm { s h a r e } } \Bigg ] + 2 u _ { N } + e _ { N , T } .\tag{23}
$$

Here $\mathcal { R } _ { \mathrm { l o c } , k } ^ { * }$ is client $k \mathrm { { : } }$ optimal local-information risk. The terms $d _ { k } , a _ { k } , a _ { \mathrm { s h a r e } }$ measure information loss, local approximation error, and shared-parameter cost. Finally, $u _ { N }$ bounds expected uniform training-to-population discrepancy, and $e _ { N , T }$ bounds empirical optimization error, addressed next.

Global convergence along the implemented updates. Let $J _ { N }$ be evaluation-mode training crossentropy on domain Θ, convex in the final affine head at fixed features. After round $u _ { 0 } ,$ , assume each representation admits a head within distance $D _ { \mathrm { c l s } } > 0$ of the current head and loss at most inf<sub>Θ</sub> $J _ { N } + a _ { \mathrm { { c t x } } }$ , with $a _ { \mathrm { { c t x } } } \geq 0$ . If the actual-update decrease bound (Proposition 9) is at least $\gamma > 0$ times the squared head-gradient norm, Theorem 9 bounds $e _ { N , T } = \mathbb { E } [ J _ { N } ( \widehat { \theta } ) - \operatorname* { i n f } _ { \Theta } J _ { N } ]$ by

$$
e _ { N , T } \leq \mathbb { E } \left[ a _ { \mathrm { c t x } } + \frac { D _ { \mathrm { c l s } } ^ { 2 } } { \gamma ( T - u _ { 0 } ) } + s _ { N , T } ^ { \mathrm { s e l } } \right] , \qquad T > u _ { 0 } .\tag{24}
$$

Here $s _ { N , T } ^ { \mathrm { s e l } } = [ J _ { N } ( \widehat { \theta } ) - J _ { N } ( \theta ^ { T } ) ]$ <sub>+</sub> accounts for validation selection, with $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . Substitution into Equation (23) gives the population-risk bound. For fixed $u _ { 0 }$ and uniformly bounded $\mathbb { E } [ D _ { \mathrm { c l s } } ^ { 2 } / \gamma ]$ , the optimization term is $O ( 1 / T )$ . Vanishing residual terms then give local-information Bayes consistency. Appendix C.12 also covers imperfect update alignment.

## 6 EXPERIMENTS

We evaluate FedASTRA through four research questions (RQ1-RQ4) concerning predictive performance, mechanism validation, information access, and scalability:

• RQ1: How effective is FedASTRA on identifier-free native hypergraphs?

• $R Q 2$ : Do the empirical results support the proposed analysis and FedASTRA’s design?

• RQ3: How does FedASTRA compare under different information-access settings?

• RQ4: Can FedASTRA scale to large real-world hypergraphs without sampling?

Additional experimental settings and results are provided in Appendices D and E, respectively.

Datasets. We evaluate two groups of datasets. (1) Native hypergraph datasets including CoraCA coauthorship, DBLP4k bibliographic, IMDB4k movie, and Trivago hotel-session data (Yadati et al., 2019; Sun et al., 2011; Fu et al., 2020; Feng et al., 2019; Gao et al., 2023; Li et al., 2026). (2) Graph datasets including Cora and CiteSeer citation networks, the Facebook Page-Page social network (Yang et al., 2016; Rozemberczki et al., 2021), and the large-scale ogbn-papers100M dataset (Hu et al., 2020). Appendix D.3 details the datasets.

Baselines. We consider two groups of baselines. (1) Federated adaptations (FedAvg (McMahan et al., 2017)) of native-hypergraph methods, including Fed-HGNN (HGNN (Feng et al., 2019)), Fed-HNHN (HNHN (Dong et al., 2020)), Fed-HyperGCN (HyperGCN (Yadati et al., 2019)), Fed-AllDeepSets (DeepSets (Zaheer et al., 2017)), Fed-AllSetTransformer (AllSet (Chien et al., 2022)), Fed-UniGCNII (UniGCNII (Huang & Yang, 2021)) and Fed-ED-HNN (ED-HNN (Wang et al., 2023)). (2) Federated graph learning methods with different levels of cross-client information access, including FedSage with no access and FedSage+ with generated information (Zhang et al., 2021), FedGCN (Yao et al., 2023) and FedCog (Lei et al., 2023) with direct access to correspondence. Details of baselines are presented in Appendix D.4.

Implementation Details. Fed-HNHN uses the configuration reported by FedHGL (Luo et al., 2025), and Fed-HyperGCN uses the local HyperGCN projection. FedASTRA uses two masks, a local EMA target, and parameter-only FedAvg (McMahan et al., 2017; Kingma & Ba, 2015). Results are averaged over five runs, except three for Fed-HNHN and all Trivago-related experiments. We report node classification accuracy (Acc.) and Macro-F1. The nodes are partitioned across $K \in \{ 3 , 6 , 9 \}$ clients using class-conditioned Dirichlet sampling (Luo et al., 2025). All methods follow the same data splits and training and evaluation protocols. Appendix D details the experimental protocols.

RQ1: performance on identifier-free native hypergraphs. As shown in Table 1, FedASTRA achieves the best Accuracy and Macro-F1 across all four hypergraph datasets from distinct domains, especially on CoraCA, outperforming the baseline by 6.5 and 8.5 percentage points, respectively. The consistent cross-domain gains support FedASTRA’s correspondence-invariant design, which avoids global structure recovery while approaching the optimum permitted by local information under the stated conditions. In contrast, direct FedAvg adaptations of traditional hypergraph models may under-exploit local information when global structure is fragment. More detailed results under different number of clients are reported in Appendix E.1.

Table 1: Performance on identifier-free native hypergraphs. Results are averaged over $K \in \{ 3 , 6 , 9 \}$
<table><tr><td rowspan="2">Method</td><td colspan="2">CoraCA</td><td colspan="2">DBLP4k</td><td colspan="2">IMDB4k</td><td colspan="2">Trivago</td></tr><tr><td>Acc. (%)</td><td>Macro-F1 (%)</td><td>Acc. (%)</td><td> $\mathbf { M a c r o - F 1 } \left( \% \right)$ </td><td>Acc. (%)</td><td>Macro-F1 (%)</td><td> $\operatorname { A c c . } \ ( { \% } )$ </td><td>Macro-F1 (%)</td></tr><tr><td>Fed-HGNN</td><td> $6 8 . 6 4 \pm 1 . 4 5$ </td><td> $\underline { { 6 4 . 3 5 \pm 1 . 7 1 } }$ </td><td> $7 7 . 6 3 \pm 1 . 3 5$ </td><td> ${ \underline { { 7 6 . 9 8 \pm 1 . 1 8 } } }$ </td><td>48.85 ± 1.29</td><td> $\underline { { 4 8 . 5 9 \pm 1 . 2 3 } }$ </td><td> $2 1 . 8 5 \pm 0 . 1 9$ </td><td> $4 . 2 7 \pm 0 . 2 0$ </td></tr><tr><td>Fed-HNHN</td><td> $\overline { { 5 4 . 7 3 \pm 1 . 1 7 } }$ </td><td> $\overline { { 5 0 . 4 2 \pm 1 . 7 6 } }$ </td><td> $\overline { { 6 9 . 5 8 \pm 1 . 1 8 } }$ </td><td> $\overline { { 6 8 . 7 9 \pm 1 . 1 2 } }$ </td><td> $\overline { { 4 4 . 9 6 \pm 1 . 5 0 } }$ </td><td> $\overline { { 4 3 . 0 5 \pm 1 . 2 4 } }$ </td><td> $2 5 . 6 8 \pm 1 . 0 2$ </td><td> $underline { 7 . 7 2 \pm 0 . 2 4 }$ </td></tr><tr><td>Fed-HyperGCN</td><td> $6 4 . 0 2 \pm 1 . 6 4$ </td><td> $5 8 . 0 7 \pm 1 . 9 5$ </td><td> $7 2 . 6 9 \pm 0 . 7 7$ </td><td> $7 1 . 4 6 \pm 0 . 8 6$ </td><td> $4 4 . 1 4 \pm 1 . 1 6$ </td><td> $4 0 . 8 6 \pm 2 . 3 8$ </td><td> $9 . 1 7 \pm 0 . 2 0$ </td><td> $0 . 1 3 \pm 0 . 0 2$ </td></tr><tr><td>Fed-AllDeepSets</td><td> $3 9 . 6 3 \pm 1 . 8 3$ </td><td> $2 7 . 0 8 \pm 2 . 0 4$ </td><td> $3 9 . 3 5 \pm 0 . 5 5$ </td><td> $3 2 . 4 5 \pm 1 . 7 1 $ </td><td> $4 0 . 3 2 \pm 1 . 1 5$ </td><td> $3 4 . 1 2 \pm 2 . 2 0$ </td><td> $1 0 . 8 8 \pm 0 . 1 5$ </td><td> $0 . 2 7 \pm 0 . 0 1$ </td></tr><tr><td>Fed-AllSetTransformer</td><td> $3 0 . 2 2 \pm 0 . 0 0$ </td><td> $6 . 6 3 \pm 0 . 0 0$ </td><td> $3 0 . 9 9 \pm 1 . 2 9$ </td><td> $2 1 . 8 1 \pm 0 . 8 6$ </td><td> $3 7 . 0 4 \pm 0 . 0 5$ </td><td> $1 8 . 1 0 \pm 0 . 0 7$ </td><td> $8 . 9 9 \pm 0 . 0 0$ </td><td> $0 . 1 0 \pm 0 . 0 0$ </td></tr><tr><td>Fed-UniGCNII</td><td> $5 9 . 5 4 \pm 1 . 3 1$ </td><td> $5 0 . 1 1 \pm 3 . 8 6$ </td><td> $7 7 . 3 5 \pm 2 . 2 6$ </td><td> $7 5 . 9 8 \pm 2 . 6 0$ </td><td> $4 1 . 6 2 \pm 0 . 7 4$ </td><td> $3 0 . 5 7 \pm 3 . 3 2$ </td><td> $9 . 1 5 \pm 0 . 1 3$ </td><td> $0 . 1 5 \pm 0 . 0 3$ </td></tr><tr><td>Fed-ED-HNN</td><td> $5 9 . 6 2 \pm 1 . 6 5$ </td><td> $5 3 . 2 8 \pm 2 . 3 0$ </td><td> $7 2 . 1 0 \pm 2 . 5 4$ </td><td> $7 1 . 3 6 \pm 2 . 5 5$ </td><td> $4 3 . 4 6 \pm 0 . 8 9$ </td><td> $3 6 . 5 8 \pm 1 . 0 5$ </td><td> $9 . 1 9 \pm 0 . 2 9$ </td><td> $0 . 1 3 \pm 0 . 0 3$ </td></tr><tr><td>FedASTRA</td><td> ${ \bf 7 5 . 1 7 \pm 0 . 4 2 }$ </td><td> $7 2 . 8 8 \pm 0 . 6 0$ </td><td> ${ \bf 7 9 . 2 2 \pm 0 . 9 8 }$ </td><td> $\mathbf { 7 8 . 6 1 \pm 0 . 8 0 }$ </td><td> ${ \bf 5 1 . 1 3 \pm 0 . 9 7 }$ </td><td> ${ \bf 5 1 . 0 6 \pm 0 . 7 7 }$ </td><td> ${ \bf 2 9 . 0 9 \pm 1 . 1 9 }$ </td><td> ${ \bf 7 . 9 6 \pm 0 . 1 0 }$ </td></tr></table>

RQ2: validation on proposed analysis and mechanism. We first validate the proposed identification analysis through two-client audits. Among 120 evaluated cells, no higher-order co-membership target is point identified, and the median endpoint-hull width increases from 36.37% at $q = 2$ to 93.71% at $q = 5 .$ . Type coarsening preserves endpoint containment in all 40 audited cells, while 32 column permutations leave $P _ { k }$ , heat features, and logits unchanged. These results confirm the predicted ambiguity under latent correspondence and the correspondence-invariant property of the learner representation. Higher-order truncation exhibits different predictive trends across datasets, indicating that its impact depends on how the lost higher-order structure aligns with task-relevant signal. Complete audit configurations and results are reported in Appendix E.3.

For mechanism ablations, we evaluate heat-basis width $R \in \{ 0 , 1 , 2 , 4 , 8 \}$ and the masked prediction objective. Figure 2 shows that removing diffusion (� = 0) degrades performance, while moderate heat widths improve accuracy. Removing the predictive objective $( \lambda _ { p r e d } = 0 )$ also consistently reduces performance, demonstrating its complementary role in FedASTRA. Complete results are reported in Appendix E.3.

![](images/1dcccebc999daeec13fe3252dee671b434d03dbb201442bed079793de8fc5d66.jpg)  
Figure 2: Ablation study on the adaptive diffusion (left) and the masked prediction (right).

RQ3: performance against baselines with varying information access. We empirically contrast the information-access regimes discussed in Section 4, as depicted in Table 2. FedASTRA outperforms FedSage and FedGCN (0-hop), which use only local information, and also surpasses FedSage+, despite the latter using additional communication to generate cross-client information. Although FedGCN (2-hop) and FedCog are stronger under a different setting with known correspondence, the gap is substantially narrowed, suggesting that the remaining gap is not solely missing cross-client information, but also how effectively the local structure is represented and exploited. The stable performance of FedASTRA as the client number grows further indicates that stronger local modeling can partially offset increasing structural information loss. Appendix E.2 provides complete metrics, information-access details, and traffic accounting.

Table 2: Performance against federated graph learning methods. Communication traffic (MiB) is averaged across datasets and client counts, with model traffic per round and auxiliary traffic per run.
<table><tr><td>Method</td><td colspan="3">Cora</td><td></td><td>CiteSeer</td><td></td><td></td><td>Facebook</td><td></td><td>Average communication traffic</td><td></td></tr><tr><td></td><td>K = 3</td><td>K = 6</td><td>K = 9</td><td>K = 3</td><td>K = 6</td><td>K = 9</td><td>K = 3</td><td>K = 6</td><td>K = 9</td><td>Model/round</td><td>Auxiliary/run</td></tr><tr><td>FedSage</td><td>70.35 ± 1.96</td><td>63.18 ± 0.97</td><td>58.65 ± 1.85</td><td>65.08 ± 0.96</td><td>63.73 ± 0.74</td><td>62.93 ± 1.72</td><td>72.63 ± 0.73</td><td>65.64 ± 1.81</td><td>60.41 ± 1.61</td><td>4.82</td><td>0</td></tr><tr><td>FedGCN (0-hop)</td><td>74.07 ± 1.57</td><td>65.90 ± 2.05</td><td>62.07 ± 1.79</td><td>68.68 ± 1.50</td><td>65.83 ± 0.56</td><td>65.14 ± 1.80</td><td>75.14 ± 0.95</td><td>67.87 ± 1.71</td><td>64.30 ± 1.22</td><td>2.41</td><td>0</td></tr><tr><td>FedSage+</td><td>69.67 ± 0.71</td><td>63.11 ± 1.66</td><td>61.02 ± 1.40</td><td>63.16 ± 1.17</td><td>62.53 ± 1.66</td><td>62.95 ± 1.62</td><td>68.17 ± 0.33</td><td>61.85 ± 0.95</td><td>58.56 ± 1.33</td><td>4.82</td><td>17314.45</td></tr><tr><td>FedGCN (2-hop)</td><td>84.05 ± 1.17</td><td>84.05 ± 1.16</td><td>84.12 ± 0.92</td><td>72.20 ± 1.44</td><td>72.27 ± 1.53</td><td>72.38 ± 1.50</td><td>83.21 ± 0.76</td><td>82.84 ± 0.50</td><td>82.73 ± 0.85</td><td>2.41</td><td>673.79</td></tr><tr><td>FedCog</td><td>84.29 ± 1.50</td><td>84.34 ± 1.06</td><td>83.92 ± 0.78</td><td>72.21 ± 1.49</td><td>72.38 ± 1.04</td><td>72.51 ± 1.42</td><td>82.46 ± 0.69</td><td>82.50 ± 0.84</td><td>82.15 ± 0.94</td><td>2.41</td><td>752.27</td></tr><tr><td>FedASTRA</td><td>77.54 ± 1.08</td><td>74.45 ± 1.60</td><td>73.33 ± 1.22</td><td>67.61 ± 2.32</td><td>68.00 ± 1.86</td><td>67.92 ± 1.88</td><td>76.88 ± 1.44</td><td>75.57 ± 0.94</td><td>75.12 ± 1.49</td><td>13.53</td><td>0</td></tr></table>

RQ4: scalability on large-scale data. We evaluate FedASTRA on ogbn-papers100M after local onehop lifting, as summarized in Table 3. We also include Fed-MLP, Fed-GCN (Kipf & Welling, 2017), and Fed-SIGN (Frasca et al., 2020) as baselines. FedASTRA outperforms the strongest baseline by up to 9.69 percentage points under comparable communication overhead. Its scalability stems from the correspondence-free local diffusion design and the strong performance further suggests that adaptive diffusion can better exploit local structures. Appendix E.4 details the results.

Table 3: Performance on complete ogbn-papers100M dataset. Nodes are partitioned across 8 clients. Traffic counts bidirectional trainable-parameter per round (MiB). BatchNorm buffers stay local.
<table><tr><td>Method</td><td>Acc. (%)</td><td>Macro-F1 (%)</td><td>#Parameters</td><td>Parameter trafffic/round</td></tr><tr><td>Fed-MLP</td><td>33.61</td><td>13.74</td><td>983,380</td><td>60.02</td></tr><tr><td>Fed-SIGN</td><td>38.42</td><td>16.56</td><td>984,442</td><td>60.09</td></tr><tr><td>Fed-HGNN</td><td>31.04</td><td>9.97</td><td>982,672</td><td>59.98</td></tr><tr><td>Fed-HNHN</td><td>35.93</td><td>13.46</td><td>34,672</td><td>2.12</td></tr><tr><td>Fed-HyperGCN</td><td>40.38</td><td>21.44</td><td>984,442</td><td>60.09</td></tr><tr><td>Fed-GCN</td><td>39.70</td><td>17.40</td><td>983,380</td><td>60.02</td></tr><tr><td>FedASTRA</td><td>50.07</td><td>29.73</td><td>976,504</td><td>59.60</td></tr></table>

## 7 CONCLUSION

We establish identification and computational boundaries for latent hyperedge correspondence and quantify when fitting costs outweigh the benefit of additional information. FedASTRA combines invariant local diffusion, masked prediction, and parameter-only communication. Under stated assumptions, information-preservation and global risk bounds connect observable memberships to prediction. Results on native hypergraphs, federated subgraphs, and large-scale ogbn-papers100M data support this approach under explicit information-access and communication constraints.

## REFERENCES

Noga Alon, Raphael Yuster, and Uri Zwick. Color-coding. J. ACM, 42(4):844–856, 1995.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619–15629, 2023. doi: 10.1109/CVPR52729.2023.01499. URL https://openaccess.thecvf.com/content/CVPR2023/html/Assran Self-Supervised\_Learning\_From\_Images\_With\_a\_Joint-Embedding\_ Predictive\_Architecture\_CVPR\_2023\_paper.html.

Jinheon Baek, Wonyong Jeong, Jiongdao Jin, Jaehong Yoon, and Sung Ju Hwang. Personalized subgraph federated learning. In ICML, volume 202 of Proceedings ofMachine Learning Research, pp. 1396–1415. PMLR, 2023.

Yikun Bai, Rocio Diaz Martin, Abihith Kothapalli, Hengrong Du, Xinran Liu, and Soheil Kolouri. Partial gromov-wasserstein metric. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html.

Garrett Birkhoff. Tres observaciones sobre el álgebra lineal. Universidad Nacional de Tucumán, Revista A, 5:147–151, 1946.

David Blackwell. Equivalent comparisons of experiments. The Annals ofMathematical Statistics, 24(2):265–272, 1953. doi: 10.1214/aoms/1177729032. URL https://doi.org/10.1214/ aoms/1177729032.

Guy Bresler, Chenghao Guo, and Yury Polyanskiy. Thresholds for reconstruction of random hypergraphs from graph projections. In Shipra Agrawal and Aaron Roth (eds.), Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings ofMachine Learning Research, pp. 632–647. PMLR, 30 Jun–03 Jul 2024. URL https://proceedings.mlr. press/v247/bresler24a.html.

Guy Bresler, Chenghao Guo, Yury Polyanskiy, and Andrew Yao. Partial and exact recovery of a random hypergraph from its graph projection. In Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pp. 534–593. PMLR, 2025. URL https://proceedings.mlr.press/v291/bresler25a.html.

Shuxiao Chen, Edgar Dobriban, and Jane H. Lee. A group-theoretic framework for data augmentation. Journal ofMachine Learning Research, 21(245):1–71, 2020. URL https://www.jmlr.org/ papers/v21/20-163.html.

Yang Chen, Cong Fang, Zhouchen Lin, and Bing Liu. Relational learning in pre-trained models: A theory from hypergraph recovery perspective. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 6666–6698. PMLR, 21–27 Jul 2024. URL https://proceedings. mlr.press/v235/chen24l.html.

Eli Chien, Chao Pan, Jianhao Peng, and Olgica Milenkovic. You are AllSet: A multiset function framework for hypergraph neural networks. In The Tenth International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=hpBTIv2uy\_E.

Samir Chowdhury, Tom Needham, Ethan Semrad, Bei Wang, and Youjia Zhou. Hypergraph cooptimal transport: Metric and categorical properties. Journal of Applied and Computational Topology, 8(5):1171–1230, 2024. doi: 10.1007/s41468-023-00142-9. URL https://doi. org/10.1007/s41468-023-00142-9.

Yihe Dong, Will Sawin, and Yoshua Bengio. HNHN: Hypergraph networks with hyperedge neurons. arXiv preprint arXiv:2006.12278, 2020. URL https://arxiv.org/abs/2006.12278.

Rodney G. Downey and Michael R. Fellows. Fixed-parameter tractability and completeness II: on completeness for W[1]. Theor. Comput. Sci., 141(1&2):109–131, 1995.

Ivan P. Fellegi and Alan B. Sunter. A theory for record linkage. Journal ofthe American Statistical Association, 64(328):1183–1210, 1969. doi: 10.1080/01621459.1969.10501049. URL https: //doi.org/10.1080/01621459.1969.10501049.

Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural networks. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pp. 3558–3565, 2019. doi: 10.1609/aaai.v33i01.33013558. URL https://ojs.aaai.org/index.php/AAAI article/view/4235.

Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Benjamin Chamberlain, Michael Bronstein, and Federico Monti. SIGN: Scalable inception graph neural networks. In ICML 2020 Workshop on Graph Representation Learning and Beyond, 2020. URL https://arxiv.org/abs/2004. 11198.

Lele Fu, Bowen Deng, Sheng Huang, Tianchi Liao, Shirui Pan, and Chuan Chen. Less is more: Federated graph learning with alleviating topology heterogeneity from A causal perspective. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 17727–17745. PMLR, 2025. URL https: //proceedings.mlr.press/v267/fu25a.html.

Xinyu Fu, Jiani Zhang, Ziqiao Meng, and Irwin King. MAGNN: Metapath aggregated graph neural network for heterogeneous graph embedding. In Proceedings of The Web Conference 2020, pp. 2331–2341. ACM, 2020. doi: 10.1145/3366423.3380297. URL https://doi.org/10. 1145/3366423.3380297.

Yue Gao, Yifan Feng, Shuyi Ji, and Rongrong Ji. HGNN+: General hypergraph neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3181–3199, 2023. doi: 10.1109/ TPAMI.2022.3182052. URL https://doi.org/10.1109/TPAMI.2022.3182052.

Yue Gao, Yifan Feng, Shiquan Liu, Xiangmin Han, Shaoyi Du, Zongze Wu, and Han Hu. Hypergraph foundation model. IEEE Trans. Pattern Anal. Mach. Intell., 48(4):4063–4080, 2026.

Daniel G. Horvitz and Donovan J. Thompson. A generalization of sampling without replacement from a finite universe. Journal ofthe American Statistical Association, 47(260):663–685, 1952. doi: 10.1080/01621459.1952.10483446.

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html.

Jing Huang and Jie Yang. UniGNN: A unified framework for graph and hypergraph neural networks. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence, pp. 2563–2569, 2021. doi: 10.24963/ijcai.2021/353. URL https://www.ijcai.org proceedings/2021/0353.pdf.

Richard M. Karp. Reducibility among combinatorial problems. In Raymond E. Miller and James W. Thatcher (eds.), Complexity of Computer Computations, pp. 85–103. Plenum Press, 1972. doi: 10.1007/978-1-4684-2001-2\_9. URL https://doi.org/10.1007/ 978-1-4684-2001-2\_9.

Shiva Prasad Kasiviswanathan, Homin K. Lee, Kobbi Nissim, Sofya Raskhodnikova, and Adam D. Smith. What can we learn privately? SIAM J. Comput., 40(3):793–826, 2011.

Sunwoo Kim, Shinhwan Kang, Fanchen Bu, Soo Yong Lee, Jaemin Yoo, and Kijung Shin. Hypeboy: Generative self-supervised representation learning on hypergraphs. In ICLR. OpenReview.net, 2024.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. URL https://arxiv.org/abs/1412. 6980.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. URL https://openreview. net/forum?id=SJU4ayYgl.

Risi Imre Kondor and John Lafferty. Diffusion kernels on graphs and other discrete input spaces. In Proceedings of the 19th International Conference on Machine Learning, pp. 315–322, 2002. URL https://people.cs.uchicago.edu/\~risi/papers/ diffusion-kernels.pdf.

Harold W. Kuhn. The hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1–2):83–97, 1955. doi: 10.1002/nav.3800020109. URL https://doi.org/10. 1002/nav.3800020109.

Yann LeCun. A path towards autonomous machine intelligence, 2022. URL https:// openreview.net/forum?id=BZ5a1r-kVsf. Version 0.9.2, position paper.

Dongjin Lee and Kijung Shin. I’m me, we’re us, and i’m us: Tri-directional contrastive learning on hypergraphs. In AAAI, pp. 8456–8464. AAAI Press, 2023.

Kyuhan Lee, Geon Lee, and Kijung Shin. MARIOH: Multiplicity-aware hypergraph reconstruction. In 2025 IEEE 41st International Conference on Data Engineering, pp. 3113–3125, 2025. doi: 10.1109/ICDE65448.2025.00233. URL https://doi.org/10.1109/ICDE65448.2025. 00233.

Runze Lei, Pinghui Wang, Junzhou Zhao, Lin Lan, Jing Tao, Chao Deng, Junlan Feng, Xidian Wang, and Xiaohong Guan. Federated learning over coupled graphs. IEEE Transactions on Parallel and Distributed Systems, 34(4):1159–1172, 2023. doi: 10.1109/TPDS.2023.3240527.

Fan Li, Xiaoyang Wang, Wenjie Zhang, Ying Zhang, and Xuemin Lin. DHG-Bench: A comprehensive benchmark for deep hypergraph learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=lhsb1ChUDF.

Xunkai Li, Zhengyu Wu, Wentao Zhang, Yinlin Zhu, Ronghua Li, and Guoren Wang. Fedgta: Topology-aware averaging for federated graph learning. Proc. VLDB Endow., 17(1):41–50, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Linfeng Luo, Zhiqi Guo, Fengxiao Tang, Zihao Qiu, and Ming Zhao. Federated hypergraph learning with local differential privacy: Toward privacy-aware hypergraph structure completion. In 2025 IEEE International Conference on Data Mining, pp. 1400–1409, 2025. doi: 10.1109/ICDM65498. 2025.00149. URL https://doi.org/10.1109/ICDM65498.2025.00149.

Charles F. Manski. Partial Identification ofProbability Distributions. Springer Series in Statistics. Springer, New York, 2003. doi: 10.1007/b97478. URL https://doi.org/10.1007/ b97478.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Proceedings ofthe 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings ofMachine Learning Research, pp. 1273–1282. PMLR, 2017. URL https://proceedings. mlr.press/v54/mcmahan17a.html.

Gabriel Peyré, Marco Cuturi, and Justin Solomon. Gromov-wasserstein averaging of kernel and distance matrices. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pp. 2664–2672. PMLR, 2016. URL https://proceedings.mlr.press/v48/peyre16.html.

William P. Pierskalla. Letter to the editor—the multidimensional assignment problem. Operations Research, 16(2):422–431, 1968. doi: 10.1287/opre.16.2.422. URL https://doi.org/10. 1287/opre.16.2.422.

Gathika Ratnayaka, James Nichols, and Qing Wang. Learning partial graph matching via optimal partial transport. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 83575–83595, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ cfa8440d500a6a6867157dfd4eaff66e-Paper-Conference.pdf.

Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In ICLR. OpenReview.net, 2018.

Benedek Rozemberczki, Carl Allen, and Rik Sarkar. Multi-scale attributed node embedding. Journal ofComplex Networks, 9(2):cnab014, 2021. doi: 10.1093/comnet/cnab014. URL https://doi. org/10.1093/comnet/cnab014.

Shanghao Shi, Ning Wang, Yang Xiao, Chaoyu Zhang, Yi Shi, Y. Thomas Hou, and Wenjing Lou. Scale-MIA: A scalable model inversion attack against secure federated learning via latent space reconstruction. In Network and Distributed System Security Symposium, 2025. doi: 10. 14722/ndss.2025.240644. URL https://www.ndss-symposium.org/ndss-paper/ scale-mia-a-scalable-model-inversion-attack-against-secure-federated-learning-via-

Yizhou Sun, Jiawei Han, Xifeng Yan, Philip S. Yu, and Tianyi Wu. PathSim: Meta path-based top-k similarity search in heterogeneous information networks. Proceedings ofthe VLDB Endowment, 4(11):992–1003, 2011. doi: 10.14778/3402707.3402736. URL https://www.vldb.org pvldb/vol4/p992-sun.pdf.

Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard, and Nicolas Courty. Optimal transport for structured data with application on graphs. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 6275–6284. PMLR, 2019. URL https://proceedings.mlr.press/v97/ titouan19a.html.

Peihao Wang, Shenghao Yang, Yunyu Liu, Zhangyang Wang, and Pan Li. Equivariant hypergraph diffusion neural operators. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=RiTjKoscnNd.

Yanbang Wang and Jon Kleinberg. From graphs to hypergraphs: Hypergraph projection and its reconstruction. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash e92b8441b702d142435f243806f9ec5a-Abstract-Conference.html.

Naganand Yadati, Madhav Nimishakavi, Prateek Yadav, Vikram Nitin, Anand Louis, and Partha Talukdar. HyperGCN: A new method for training graph convolutional networks on hypergraphs. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper/2019/hash 1efa39bcaec6f3900149160693694536-Abstract.html.

Zhilin Yang, William W. Cohen, and Ruslan Salakhutdinov. Revisiting semi-supervised learning with graph embeddings. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pp. 40–48. PMLR, 2016. URL https: //proceedings.mlr.press/v48/yanga16.html.

Yuhang Yao, Weizhao Jin, Srivatsan Ravi, and Carlee Joe-Wong. FedGCN: Convergencecommunication tradeoffs in federated training of graph convolutional networks. In Advances in Neural Information Processing Systems, volume 36, pp. 79748–79760, 2023. doi: 10.52202/ 075280-3491. URL https://proceedings.neurips.cc/paper\_files/paper 2023/hash/fc07feae9af49dd3f1a1e049b77f4e17-Abstract-Conference. html.

Jean-Gabriel Young, Giovanni Petri, and Tiago P. Peixoto. Hypergraph reconstruction from network data. Communications Physics, 4:135, 2021. doi: 10.1038/s42005-021-00637-w. URL https: //doi.org/10.1038/s42005-021-00637-w.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. Deep sets. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://proceedings.neurips.cc/paper\_files/paper 2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html.

Ke Zhang, Carl Yang, Xiaoxiao Li, Lichao Sun, and Siu Ming Yiu. Subgraph federated learning with missing neighbor generation. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper/2021/hash 34adeb8e3242824038aa65460a47c29e-Abstract.html.

Stephen Y. Zhang, Fangfei Lan, Youjia Zhou, Agnese Barbensi, Michael P. H. Stumpf, Bei Wang, and Tom Needham. Geometry of the space of partitioned networks: A unified theoretical and computational framework. Information and Inference: A Journal ofthe IMA, 15(3):iaag022, 2026. doi: 10.1093/imaiai/iaag022. URL https://doi.org/10.1093/imaiai/iaag022.

Dengyong Zhou, Jiayuan Huang, and Bernhard Schölkopf. Learning with hypergraphs: Clustering, classification, and embedding. In Advances in Neural Information Processing Systems, volume 19, 2006. URL https://proceedings.neurips.cc/paper\_files/paper 2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html.

## Appendix

## Contents

A More Related Work . 16   
B Formal Results and Identification Boundaries 16   
B.1 Important Notation . 17   
B.2 Observation Channel and Design-based Recovery 17   
B.3 Two-client Assumptions and Weight-stratum Boundary. . .18   
C Proofs 18   
C.1 Two-client Completion and Point Identification . 18   
C.2 Order-� Mass Identification 19   
C.3 Structured Tractable Class: Anti-Monge Bounds . . 20   
C.4 Multi-client Computational Boundary 20   
C.5 Boolean-zeta Inversion and Partition Censoring . .22   
C.6 Horvitz–Thompson Estimation. .22   
C.7 Observational Equivalence and a Two-point Bound . 23   
C.8 Released-data Factorization and Observable Spectral Approximation . . 24   
C.9 Observable Diffusion and Predictive Dynamics 25   
C.10 Downstream Risk with Additional Information 27   
C.11 Randomized Recovery from Diffusion Features and Local Prediction Risk . 29   
C.12 Stochastic Federated Optimization and Risk Transfer . . . 31   
D Experimental Protocols and Implementation Details . . 35   
D.1 FedASTRA Implementation . . 35   
D.2 Graph Partitions and Lifts . 37   
D.3 Dataset Details . . 38   
D.4 Experimental Configurations 38   
E Additional Experimental Results . . 40   
E.1 Performance on Native Hypergraphs. .40   
E.2 Performance Against Baselines with Varying Information Access . .41   
E.3 Validation on Proposed Analysis and Mechanism . 42   
E.4 Scalability on Large-scale Data . 45

## A MORE RELATED WORK

Latent correspondence and partial identification. Unknown correspondence is classical in record linkage (Fellegi & Sunter, 1969), where latent pairings between records must be inferred from incomplete identifying information. Partial identification instead retains the set of target values compatible with the observations when point identification is unavailable (Manski, 2003). Our setting combines these perspectives at the level of higher-order relations: client-local hyperedge fragments are observed, but their cross-client instance correspondence is not. Rather than selecting or estimating one global linkage, we characterize the full compatibility class induced by the released fragments and derive sharp target bounds over all compatible alignments.

Alignment under latent correspondence. When object identities are not shared, relational alignment is often formulated through latent couplings. Gromov–Wasserstein and fused Gromov– Wasserstein formulations compare structured objects without requiring a common coordinate system (Peyré et al., 2016; Vayer et al., 2019), while recent partial formulations relax full-mass or fullmatching requirements (Bai et al., 2025; Ratnayaka et al., 2025). For higher-order structures, HyperCOT develops transport-based comparisons of hypergraphs (Chowdhury et al., 2024), and recent partitioned-network geometry unifies graph and hypergraph alignment within a common metric framework (Zhang et al., 2026). These approaches optimize an alignment or distance under a specified geometry. But more specifically here, we are more concerned with which target values remain invariant to latent cross-client correspondence and which vary within the compatibility class induced by a fixed federated release.

Hypergraph reconstruction from coarsened observations. Some existing papers study whether higher-order structure can be reconstructed from lower-order or otherwise coarsened observations. Early work develops statistical reconstruction from network data (Young et al., 2021). Wang and Kleinberg characterize information loss under hypergraph projection and study reconstruction with additional structural knowledge (Wang & Kleinberg, 2024). For random uniform hypergraphs, recent work gives information-theoretic and algorithmic thresholds for exact recovery (Bresler et al., 2024), followed by a characterization of both partial and exact recovery across hyperedge sizes and multiplicity regimes (Bresler et al., 2025). Multiplicity-aware supervised reconstruction provides a complementary algorithmic approach (Lee et al., 2025), while recent learning theory has also formulated relational learning itself through a hypergraph-recovery model (Chen et al., 2024). These works primarily ask when a latent source hypergraph is recoverable under projection or a specified generative model. Our observation is however different: clients retain higher-order fragments themselves, but their cross-client instance identities are suppressed. We therefore retain all compatible global hypergraphs and characterize what is identifiable without directly assuming that one global completion can be recovered.

Assignment and multiway matching complexity. Two-way correspondence is closely related to classical linear assignment and bipartite matching (Birkhoff, 1946; Kuhn, 1955), whereas multidimensional assignment and three-dimensional matching exhibit fundamentally different computational behavior (Pierskalla, 1968; Karp, 1972). Parameterized algorithms for set packing and matching draw on tools such as color coding, while the W-hierarchy distinguishes fixed-parameter tractability from parameterized intractability (Alon et al., 1995; Downey & Fellows, 1995). Recent partial graph matching still exposes tractable assignment structure in important pairwise settings (Ratnayaka et al., 2025). Our result places these phenomena within one latent-correspondence problem: under the same observation and reward rules, two-client correspondence is polynomial, fixed-client multiway matching is NP-complete but fixed-parameter tractable in the stated joint parameter, and unbounded client count is W[1]-complete in matching size.

## B FORMAL RESULTS AND IDENTIFICATION BOUNDARIES

This section records the observation assumptions used in the proofs: which fragment attributes are retained, when partition censoring admits unbiased correction, and when unknown correspondence is exactly a bipartite assignment.

## B.1 IMPORTANT NOTATION

Table 4: Core notation for the observation model, identification results, algorithm, and experiments.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal { H } = ( V , \mathcal { E } , X , r , w )$ </td><td>Finite node-attributed, typed, weighted multihypergraph.</td></tr><tr><td> $A _ { e } , r _ { e } , w _ { e }$ </td><td>Support, type in R, and positive weight of relation instance  $e \in { \mathcal { E } } .$ </td></tr><tr><td> $w _ { r } ( A )$ </td><td>Aggregate weight of type-r instances with support A.</td></tr><tr><td> $M _ { q , r } ^ { \mathcal { H } } ( S )$ </td><td>Weighted type-r order-q co-membership count for  $S \in \binom { V } { q } .$ </td></tr><tr><td> $\textstyle Z , z , V _ { k }$ </td><td>Random node-to-client assignment, one realization, and the nodes assigned to client  $k .$ </td></tr><tr><td> $c , \upsilon _ { e , k }$ </td><td>Minimum visible fragment size and the analyst&#x27;s local/cross status mark.</td></tr><tr><td> $\Gamma , M _ { \Gamma }$ </td><td>Public observation assumptions and its allowed latent-hypergraph class.</td></tr><tr><td> $Y _ { \mathrm { A } } , y _ { \mathrm { A } }$ </td><td>Analyst release as a random element and one realized value.</td></tr><tr><td> $Y _ { \mathrm { L } } , y _ { \mathrm { L } }$ </td><td>Coarsened learner release  $Y _ { \mathrm { L } } = \rho ( Y _ { \mathrm { A } } )$  and one realized value.</td></tr><tr><td> $\mathcal { F } _ { \mathrm { A } } , \mathcal { F } _ { \mathrm { L } }$ </td><td>Analyst and learner compatibility classes defined in Equation (4) and Corollary 2.</td></tr><tr><td> $\bar { M } _ { q , r } ( \cdot , z ) , \tau _ { q }$ </td><td>Retained type-r coordinates and the type-aggregated erased-mass fraction.</td></tr><tr><td> $\pi _ { S } , \pi _ { q }$ </td><td>Inclusion probability of a q-set and its common value under an exchangeable random- ized partition design.</td></tr><tr><td> $\textstyle { \mathcal { T } } _ { q , \psi } ( y _ { \mathrm { A } } )$   $h ^ { - } = ( r , \omega ) , C ^ { h } , \mathbb { B } _ { m _ { h } }$ </td><td>Values of  $\Phi _ { q , \psi }$  over  $\mathcal { F } _ { \mathrm { { A } } } ( y _ { \mathrm { { A } } } , \Gamma )$  Exact type-and-weight stratum, its assignment cost, and the Birkhoff polytope.</td></tr><tr><td> $B _ { k } , D _ { \nu , k } , D _ { e , k } , I _ { 0 , k }$ </td><td>Client incidence matrix, node and relation degrees, and isolated-node indicator.</td></tr><tr><td> $P _ { k } , P _ { k } ^ { \mathrm { s y m } }$ </td><td>Observable Markov operator and its symmetric representative on positive-degree</td></tr><tr><td> $\mathsf { K } _ { k } ( t )$ </td><td>nodes.</td></tr><tr><td> $R , t _ { b j } , \epsilon _ { \mathrm { o r d } }$ </td><td>Continuous-time local diffusion  $\exp \{ t ( P _ { k } - I ) \}$  at time  $t \geq 0 .$  Number of positive heat channels, channel-b time for feature  $j ,$  and its positive</td></tr><tr><td></td><td>increment floor.</td></tr><tr><td> $W _ { k } ( X _ { k } \mid \vartheta ) , W _ { k , \theta } ( X _ { k } )$ </td><td>Heat-basis representation and its shorthand with  $\vartheta = \operatorname { c t x } ( \theta )$ </td></tr><tr><td> $L _ { \mathrm { t a i l } } , \varepsilon _ { \mathrm { t a i l } }$ </td><td>Shared recurrence degree and its deterministic Poisson-tail tolerance.</td></tr><tr><td> $\theta , \vartheta , \bar { \vartheta }$ </td><td>Complete online state, its context coordinates  $( \theta _ { \mathrm { { e n c } } } , \eta )$  , and the client-local EMA target context.</td></tr><tr><td> $p _ { \theta } , c _ { \theta } , y _ { \nu }$ </td><td>Rowwise predictor, rowwise classifier, and class label of node v.</td></tr><tr><td> $\gamma _ { \mathrm { v i e w } } , \lambda _ { \mathrm { p r e d } } , \mu$ </td><td>View-agreement weight, predictive-loss weight, and target momentum.</td></tr><tr><td> $\mathcal { P } , \xi , \mathcal { T } _ { T }$ </td><td>Fixed training protocol, algorithmic randomness, and the transcript through round T.</td></tr></table>

Probability and conditioning convention. Unless a statement says otherwise, $\mathcal { H } , z , \Gamma .$ , and the realized release are fixed. Expectations and variances in Theorem 3 are taken over the declared random partition Z. Corollary 3 explicitly averages over a uniform latent matching. Algorithmic laws average over $\xi \sim \nu$ , where the fixed law � is independent of H. The auxiliary variable $N _ { t } ^ { \mathrm { a u x } }$ below only expresses deterministic Poisson coefficients and tail bounds and is absent from implementation sampling. Appendix C.10 separately defines its downstream evaluation law.

## B.2 OBSERVATION CHANNEL AND DESIGN-BASED RECOVERY

## B.2.1 AGGREGATE SUPPORT–TYPE WEIGHTS AND BOOLEAN-ZETA COORDINATES

The analyst release retains each fragment occurrence with its exact type, weight, and local/cross status. Assignment strata use these individual rows, including duplicates. Co-membership coordinates instead depend on $w _ { r } ( A )$ , the total weight of rows with support � and type �. Aggregation loses the allocation of weight among duplicate occurrences, so it cannot replace the row-level table in the matching analysis. Erasing types further reduces the observation to $\begin{array} { r } { \mathbf { \bar { M } } _ { q } = \sum _ { r } M _ { q , r } } \end{array}$ . The following inclusion describes the resulting uncertainty on the same latent class $M _ { \Gamma }$

Corollary 1 (Coarsened or missing type marks enlarge identified sets). Let $\chi : \mathcal { R } \to \mathcal { R } _ { \chi }$ merge or suppress type labels, and let �<sub>#</sub> apply this map to the release while preserving its otherfields. For an exact-type release $y _ { \mathrm { e x } }$ , set $y _ { \chi } = \chi _ { \# } y _ { \mathrm { e x } }$ and $O _ { \chi } = \chi _ { \# } \circ O _ { z , \Gamma }$ , with public type restrictions pushed forward accordingly. On the common class $M _ { \Gamma } ,$ define

$$
\mathcal { F } _ { \mathrm { e x } } ( y _ { \mathrm { e x } } , \Gamma ) = \{ \mathcal { H } \in \mathcal { M } _ { \Gamma } : { \mathcal { O } } _ { z , \Gamma } ( \mathcal { H } ) = y _ { \mathrm { e x } } \} , \qquad \mathcal { F } _ { \chi } ( y _ { \chi } , \Gamma ) = \{ \mathcal { H } \in \mathcal { M } _ { \Gamma } : { \mathcal { O } } _ { \chi } ( \mathcal { H } ) = y _ { \chi } \} .\tag{25}
$$

For a scalar target $\phi : { \mathcal { M } } _ { \Gamma } \to \mathbb { R }$ , write $\mathcal { I } _ { \phi , \mathrm { e x } } = \phi ( \mathcal { F } _ { \mathrm { e x } } ( y _ { \mathrm { e x } } , \Gamma ) )$ and $\mathcal { I } _ { \phi , \chi } = \phi ( \mathcal { F } _ { \chi } ( y _ { \chi } , \Gamma ) )$ . Then

$$
\mathcal { F } _ { \mathrm { e x } } ( y _ { \mathrm { e x } } , \Gamma ) \subseteq \mathcal { F } _ { \boldsymbol { \chi } } ( y _ { \boldsymbol { \chi } } , \Gamma ) , \qquad \mathcal { I } _ { \phi , \mathrm { e x } } \subseteq \mathcal { I } _ { \phi , \boldsymbol { \chi } } .\tag{26}
$$

When the scalar endpoints arefinite, this inclusion gives

$$
\operatorname* { i n f } \mathcal { I } _ { \phi , \chi } \leq \operatorname* { i n f } \mathcal { I } _ { \phi , \mathrm { e x } } \leq \operatorname* { s u p } \mathcal { I } _ { \phi , \mathrm { e x } } \leq \operatorname* { s u p } \mathcal { I } _ { \phi , \chi } .\tag{27}
$$

Thus type coarsening can only preserve or widen the sharp endpoint interval.

Proof. Every hypergraph reproducing the exact marked table also reproduces its �-coarsening, which proves the first inclusion. Taking the image under � proves the second, and nested scalar sets give the endpoint inequalities. □

## B.2.2 INDEPENDENT AND FIXED-CAPACITY DESIGNS

For fixed H and $q \geq c$ , a coordinate $S \in \binom { V } { q }$ survives when all its nodes share a client. Independent assignments with $p _ { \nu k } = \mathrm { P r } \{ Z ( \nu ) = k \}$ give

$$
\pi _ { S } = \operatorname* { P r } \{ Z | _ { S } { \mathrm { ~ i s ~ c o n s t a n t } } \} = \sum _ { k = 1 } ^ { K } \prod _ { \nu \in S } p _ { \nu k } .\tag{28}
$$

For a uniformly random partition with fixed capacities $\left( n _ { 1 } , \ldots , n _ { K } \right)$ , exchangeability gives

$$
\pi _ { q } = \sum _ { k = 1 } ^ { K } \frac { { \binom { n _ { k } } { q } } } { { \binom { | V | } { q } } } .\tag{29}
$$

Positive inclusion probabilities permit coordinatewise Horvitz–Thompson weighting (Horvitz & Thompson, 1952). Its unbiasedness concerns repeated randomized partitions. Proportions from a fixed institutional partition alone actually fail to specify the required assignment law.

## B.2.3 CONDITIONS FOR COORDINATE-WISE PARTITION CENSORING

Equation (6) requires retained multiplicity, exact weight, type, and $q \geq c$ . Deduplication would change the count, so the data interface preserves duplicate and singleton columns. For $c > 1$ , some nonempty fragments disappear. Completion analysis then requires constraints on original hyperedge counts or orders. The exact two-client result uses $c = 1$

## B.3 TWO-CLIENT ASSUMPTIONS AND WEIGHT-STRATUM BOUNDARY

Theorem 1 assumes two clients, $c = 1$ , observed local/cross status, one nonempty row at each client per cross-client hyperedge, and exact type-and-weight strata $h = \left( r , \omega \right)$ with $m _ { 1 h } = m _ { 2 h } = : m _ { h }$ Each compatible completion is then a bijection within every stratum. Missing fragments, uncertain counts, or further clients change this feasible set. Weight bins also enlarge it: the public model must specify the allowed underlying weights, and endpoints must be optimized over that range rather than evaluated at an assumed bin midpoint.

Under the exact assumptions, general two-client targets admit assignment bounds, while total order-� mass admits sorting bounds. General three-client targets are NP-hard. For any client count, anchoradditive costs remain tractable because the objective separates into $K - 1$ independent bipartite assignments (Proposition 1).

## C PROOFS

We prove the structural bounds first, then connect observable heat features to prediction risk and the shared-model federated objective.

## C.1 TWO-CLIENT COMPLETION AND POINT IDENTIFICATION

Fix a stratum $h = ( r , \omega )$ . Under the exact two-client assumptions, every � $\in \mathrm { S y m } ( m _ { h } )$ is compatible. Each �-set in $F _ { 1 i } ^ { h } \cup \dot { F } _ { 2 , \varpi ( i ) } ^ { h }$ lies within one fragment or intersects both. Summing the withinfragment terms over � is matching-invariant, while the mixed terms give $C _ { i , \varpi ( i ) } ^ { h }$ . Adding known local

contributions proves Equation (10). By Birkhoff–von Neumann integrality, the vertices of $\mathbb { B } _ { m }$ are exactly the permutation matrices (Birkhoff, 1946). Both linear-program endpoints therefore attain compatible matchings and are sharp. Each costs ${ \dot { O } } ( m _ { h } ^ { 3 } )$ by Hungarian assignment (Kuhn, 1955).

The assignment dual also provides a certificate for the computed bound. For each stratum, the lower-endpoint dual is

$$
\operatorname* { m a x } _ { u , \nu } \sum _ { i } u _ { i } + \sum _ { j } \nu _ { j } \quad \mathrm { s . t . } \quad u _ { i } + \nu _ { j } \leq C _ { i j } ^ { h } .\tag{30}
$$

The upper endpoint has a minimization dual with $u _ { i } + \nu _ { j } \geq C _ { i j } ^ { h }$ . Dual feasibility and equality of primal and dual values certify an endpoint, with equality on every pair in an optimal matching.

If $C _ { i j } = \alpha _ { i } + \beta _ { j }$ , every permutation sums to $\begin{array} { r } { \sum _ { i } \alpha _ { i } + \sum _ { j } \beta _ { j } } \end{array}$ . Conversely, equality of all permutation sums implies equality after exchanging any two assigned columns, hence

$$
C _ { i j } + C _ { i ^ { \prime } j ^ { \prime } } - C _ { i j ^ { \prime } } - C _ { i ^ { \prime } j } = 0 .\tag{31}
$$

When all such rectangular cross-differences vanish, set $\alpha _ { i } = C _ { i 1 }$ and $\beta _ { j } = C _ { 1 j } - C _ { 1 1 }$ . The cycle using row 1 and column 1 gives $C _ { i j } ~ = ~ \alpha _ { i } + \beta _ { j }$ For $m \ = \ 1$ this representation is immediate. Finally, $J _ { m } C J _ { m } = 0$ if and only if � lies in the kernel of double centering, which is exactly the row–column-additive subspace. Thus the quotient of cost matrices by the kernel of assignment variation is represented uniquely by $J _ { m } C J _ { m }$ . Since $J _ { m }$ has rank $m - 1$ , this image has dimension $( m - 1 ) ^ { 2 }$ . The quotient maps injectively into the following $( m - 1 ) ^ { 2 }$ base four-cycle coordinates:

$$
\left( C _ { i j } - C _ { i 1 } - C _ { 1 j } + C _ { 1 1 } \right) _ { i , j = 2 } ^ { m }\tag{32}
$$

Zero base coordinates imply row–column additivity, so these $( m - 1 ) ^ { 2 }$ coordinates are independent on the quotient. The dimension is attainable with $q = 2$ and unit-weight singleton fragments by setting $\psi _ { 2 , r } ( \{ a _ { i } , b _ { j } \} , X _ { \{ a _ { i } , b _ { i } \} } ) = C _ { i j }$ . Repeated fragments or restricted test functions may reduce the realized dimension, without changing the endpoint argument. Since strata vary independently, point identification requires additivity in every stratum. □

## C.2 ORDER-� MASS IDENTIFICATION

For a general target, the preceding proof requires an assignment solver. Total order-� mass has more structure: with $q \geq 2 ,$ , its mixed cost depends only on fragment sizes. This allows both endpoints to be computed by sorting. Within stratum $h = \left( r , \omega \right)$ , set $a _ { i } ^ { h } = | { \cal F } _ { 1 i } ^ { h } | , b _ { j } ^ { h } = | { \cal F } _ { 2 j } ^ { h }$ |, and $\omega _ { h } = \omega$ Vandermonde’s identity and its discrete mixed difference give

$$
\kappa _ { q } ( a , b ) = { \binom { a + b } { q } } - { \binom { a } { q } } - { \binom { b } { q } } = \sum _ { \ell = 1 } ^ { q - 1 } { \binom { a } { \ell } } { \binom { b } { q - \ell } } ,\tag{33}
$$

$$
\kappa _ { q } ( a , b ) + \kappa _ { q } ( a ^ { \prime } , b ^ { \prime } ) - \kappa _ { q } ( a , b ^ { \prime } ) - \kappa _ { q } ( a ^ { \prime } , b ) = \sum _ { x = a } ^ { a ^ { \prime } - 1 } \sum _ { y = b } ^ { b ^ { \prime } - 1 } \binom { x + y } { q - 2 } \geq 0\tag{34}
$$

for $a \leq a ^ { \prime }$ and $b \leq b ^ { \prime }$ . Same-order pairing maximizes the total mass at $T _ { q } ^ { + }$ , while opposite-order pairing minimizes it at $T _ { q } ^ { - }$ . Writing the released local mass as $M _ { q } ^ { \mathrm { o b s } }$ gives sharp erased-mass bounds:

$$
\tau _ { q } ^ { - } = 1 - M _ { q } ^ { \mathrm { o b s } } / T _ { q } ^ { - } , \qquad \tau _ { q } ^ { + } = 1 - M _ { q } ^ { \mathrm { o b s } } / T _ { q } ^ { + } .\tag{35}
$$

The zero-total convention and explicit formulas for $T _ { q } ^ { \pm }$ appear in Appendix $\mathrm { C } . 3$ . Point identification holds exactly when the mixed difference vanishes throughout every stratum. Sorting computes both attainable endpoints in $O ( \sum _ { h }$ �<sub>ℎ</sub> log $m _ { h } )$ time by sorting fragment sizes within each stratum.

To determine when these bounds coincide, take the constant-one test. Its stratum cost is $C _ { i j } ^ { h } \ =$ $\omega _ { h } \kappa _ { q } ( a _ { i } ^ { h } , b _ { j } ^ { h } )$ . Equation (34) yields the exact rectangular $2 \times 2$ cross-difference within each fixed type-and-weight stratum:

$$
C _ { i j } ^ { h } + C _ { i ^ { \prime } j ^ { \prime } } ^ { h } - C _ { i j ^ { \prime } } ^ { h } - C _ { i ^ { \prime } j } ^ { h } = \omega _ { h } \sum _ { x = a _ { i } ^ { h } } ^ { a _ { i ^ { \prime } } ^ { h } - 1 } \sum _ { y = b _ { j } ^ { h } } ^ { b _ { j ^ { \prime } } ^ { h } - 1 } \binom { x + y } { q - 2 } .\tag{36}
$$

Theorem 1 makes vanishing of all these quantities necessary and sufficient for point identification. If both lists vary, take $a _ { i ^ { \prime } } ^ { h } = \operatorname* { m a x } _ { i } a _ { i } ^ { h }$ and $b _ { j ^ { \prime } } ^ { h ^ { \prime } } = \operatorname* { m a x } _ { j } b _ { j } ^ { h }$ with strictly smaller $a _ { i } ^ { h } , b _ { j } ^ { \dot { h } }$ . When $a _ { i ^ { \prime } } ^ { h } + b _ { j ^ { \prime } } ^ { h } \geq q$ the summand at $( x , y ) = ( a _ { i ^ { \prime } } ^ { h } - 1 , b _ { i ^ { \prime } } ^ { h } - 1 )$ is positive, so point identification fails. For $q = 2$ every nonempty summand equals one. Thus, at $q = 2$ , all cross-differences vanish exactly when at least one size list is constant in each stratum. For general $q ,$ , if one list is constant then all rectangles are empty in that direction. If both vary but max $\phantom { } _ { i } \overline { { a } } _ { i } ^ { h } + \operatorname* { m a x } _ { j } { b } _ { j } ^ { h } < q ,$ , every mixed $q \cdot$ -node-subset count is zero. Together with the preceding strict argument, these are the only point-identified cases. □

## C.3 STRUCTURED TRACTABLE CLASS: ANTI-MONGE BOUNDS

Vandermonde’s identity gives Equation (33) by removing the two within-fragment terms. For $\begin{array} { r } { b _ { q } ( n ) = { \binom { n } { q } } } \end{array}$ , Pascal’s identity gives $\begin{array} { r } { b _ { q } ( n + 2 ) - 2 \dot { b } _ { q } ( n + 1 ) + \dot { b } _ { q } ( n ) = \binom { n } { q - 2 } } \end{array}$ . Summing this difference over $x = \dot { a , } \ldots , a ^ { \prime } - 1$ and $y = b , \ldots , b ^ { \prime } - 1$ proves Equation (34), since row-only and column-only terms cancel.

Within each stratum, sort $a _ { 1 } ^ { h } \leq \cdots \leq a _ { m _ { h } } ^ { h }$ . If a proposed matching pairs $a _ { i } ^ { h } \leq a _ { i ^ { \prime } } ^ { h }$ with $b _ { j } ^ { h } > b _ { j ^ { \prime } } ^ { h }$ , the anti-Monge inequality shows that swapping the two targets weakly increases the total mixed count. Repeating adjacent swaps proves that equal-order pairing maximizes it. Applying the same argument to the reversed � order proves the minimum. Non-negative retained weights preserve the inequality within each fixed stratum, and strata add. With invariant local mass $M _ { q } ^ { \mathrm { o b s } }$ , the completed mass under $\varpi = ( \varpi _ { h } ) _ { h }$ is $\begin{array} { r } { T _ { q } ( \varpi ) = M _ { q } ^ { \mathrm { o b s } } + \sum _ { h } \omega _ { h } \sum _ { i } \kappa _ { q } ( a _ { i } ^ { h } , b _ { \varpi _ { h } ( i ) } ^ { h } ) } \end{array}$ . Since $1 - M _ { q } ^ { \mathrm { o b s } } / T _ { q }$ is nondecreasing in $T _ { q }$ for $T _ { q } > 0$ , the claimed truncation bounds follow. If $T _ { q } ^ { - } = 0 ,$ then $M _ { q } ^ { \mathrm { o b s } } = 0 :$ the zero-total convention gives the lower endpoint zero, while the upper endpoint is one if $T _ { q } ^ { + } > 0$ and zero otherwise, exactly as in Equation (35). Sorting the fragment-size lists computes both sharp endpoints in � (� log �) time. □

## C.4 MULTI-CLIENT COMPUTATIONAL BOUNDARY

For three clients, anchoring the first fragment index leaves two permutations $\varpi , \varsigma \in \mathrm { S y m } ( m )$ and the objective

$$
\Phi _ { \mathrm { o b s } } + \sum _ { i = 1 } ^ { m } C _ { i , \varpi ( i ) , \varsigma ( i ) } .\tag{37}
$$

For any finite tensor $T ,$ , create unit-weight singleton fragments $\{ a _ { i } \} , \{ b _ { j } \} , \{ c _ { k } \}$ in one type/weight stratum, set $q \ = \ 3 .$ , fix one type $r ,$ define $\psi _ { 3 , r } ( \{ a _ { i } , b _ { j } , c _ { k } \} , X _ { \{ a _ { i } , b _ { i } , c _ { k } \} } ) = T _ { i j k }$ , and set the test function to zero on every other triple. Each completed hyperedge has exactly one �-node subset, so its cost tensor is exactly $T .$ The $m ^ { 3 }$ test-function table is an explicit polynomial-size part of the endpoint instance. For a three-dimensional-matching instance, choose $T _ { i j k } = 0$ on an allowed triple and $T _ { i j k } = 1$ otherwise. A completion with value zero exists if and only if the three-dimensional-matching instance has a perfect matching. The latter is NP-complete (Karp, 1972). Hence the lower endpoint is NP-hard for three clients. More clients contain this construction by adding singleton fragment lists and defining the $q = 3$ test to be zero on every triple involving an added client. Their permutations leave the objective unchanged. Together, all these prove hardness for general linear co-membership targets.

For a sparse binary-reward target, correspondence reduces to hypergraph matching. The following result separates the effects of client count and desired matching size.

Theorem 2 (Parameterized complexity of hyperedge matching). In one exact all-client stratum, let each of $K \geq 2$ clients have � distinct singletonfragments, with one common type and unit weight. Set $q = K$ . The input listsfavorable tuples $\mathcal { E } _ { + } \subseteq [ \breve { m } ] ^ { K }$ and assigns the order-� query value one to them and zero to all other tuples. This sparse default-zero encoding includes all client-row lists, so �� is bounded by the input length. For permutations $\varpi _ { 2 } , \ldots , \varpi _ { K }$ of [�], define

$$
\Phi _ { + } ( { \mathcal { E } } _ { + } ) = \operatorname* { m a x } _ { \varpi _ { 2 } , \ldots , \varpi _ { K } } \sum _ { i = 1 } ^ { m } \mathbf { 1 } \{ ( i , \varpi _ { 2 } ( i ) , \ldots , \neg _ { K } ( i ) ) \in { \mathcal { E } } _ { + } \} .\tag{38}
$$

For $1 \leq b \leq m$ , deciding $\Phi _ { + } \geq b$ is polynomial-time solvable when $K = 2$ and NP-complete for every fixed $K \geq 3 .$ . It is fixed-parameter tractable in $K b ,$ and hence in � forfixed �. When � is unbounded it is W[1]-complete parameterized by �. A direct randomized algorithm has one-sided error at most $\delta _ { \mathrm { c c } } \in ( 0 , 1 )$ and time $O ( ( 2 e ) ^ { K b } \mathrm { p o l y } ( L _ { \mathrm { i n } } ) \log ( 2 / \delta _ { \mathrm { c c } } ) )$ , where $L _ { \mathrm { i n } }$ is the input length.

Proof. Treat client-row pairs $( k , i )$ as vertices in � disjoint parts. A favorable tuple is a �-uniform hyperedge containing one vertex from each part. Rewarded tuples in a completion are pairwise disjoint. Conversely, any � disjoint favorable tuples extend to a full completion: each part has $m - b$ unused rows, which can be paired arbitrarily. Thus the endpoint decision is exactly the existence of a �-edge matching in the explicitly supplied �-partite hypergraph. For $K = 2$ this is bipartite matching. For $\bar { K } = 3 , b = \bar { m }$ it is perfect three-dimensional matching (Karp, 1972). For larger fixed �, copy the first coordinate into each new coordinate. This preserves disjointness and proves NP-hardness. A list of � tuples is a polynomially checkable witness, establishing NP membership.

For fixed-parameter tractability, put $s = K b$ and color every client-row independently and uniformly with � colors. The � distinct rows of any fixed witness all receive different colors with probability $s ! / s ^ { s } \geq e ^ { - s }$ . Discard tuples whose vertices repeat a color. Use a Boolean dynamic program over color subsets, starting from the empty subset and adjoining the � colors of any tuple disjoint from the current subset. Reaching all � colors certifies exactly � disjoint tuples. Disjoint color masks cannot share a row, so every reported witness is valid. One trial costs $O ( \bar { 2 } ^ { s } | \mathcal { E } _ { + } | K \bar { + } K m )$ . Repeating $\lceil e ^ { s } \log ( 2 / \delta _ { \mathrm { c c } } ) \rceil$ times yields the claimed one-sided error and running time. Perfect-hash families derandomize color-coding to give deterministic fixed-parameter tractability (Alon et al., 1995).

For W[1] membership, form the conflict graph with one vertex per favorable tuple and an edge between tuples sharing any client-row. The desired matching is an independent set of size �, a parameter-preserving reduction to Independent Set (Downey & Fellows, 1995). For hardness, take an Independent Set instance $H = ( U , F _ { H } )$ with $N = | U |$ vertices and target �. Create one private client and one client for each edge of �, adding a harmless client if needed to make $K \geq 2$ . Every client has $m = N$ rows. For each $\nu \in U ,$ , list one favorable tuple $t _ { \nu }$ . Its private-client row is unique to �. At the client for edge $\{ u , \nu \}$ , tuples $t _ { u } , t _ { \nu }$ share a row, and all other tuples use distinct rows different from that shared row. Exactly $N - 1$ rows are used there, which is feasible. A harmless client assigns distinct rows to all tuples. Two favorable tuples share a row precisely when their vertices in � are adjacent. The construction is polynomial and preserves $b .$ . Therefore an independent set of size � exists precisely when the constructed endpoint is at least �, proving W[1]-hardness and completeness. □

Here candidate hyperedges have rank �. The fixed-rank and variable-rank classifications concern this sparse reward problem, including its supplied query function. They do not classify arbitrary dense costs or neural-network training. For any client count, separation across a designated anchor gives the following tractable exception.

Proposition 1 (Additive multi-client costs with one designated anchor). Consider an exact allclient stratum with � rows per client, one row from each compatible original hyperedge at every client, and independently feasible one-to-one matchings anchored at client 1. A fully additive tensor $\begin{array} { r } { C _ { i _ { 1 } , \dots , i _ { K } } = \sum _ { k } a _ { i _ { k } } ^ { ( k ) } } \end{array}$ is point identified. When the cost adds across the anchor–client pairs, $\begin{array} { r } { C _ { i _ { 1 } , \dots , i _ { K } } = a _ { i _ { 1 } } + \sum _ { k = 2 } ^ { K } C _ { i _ { 1 } i _ { k } } ^ { ( k ) } } \end{array}$ , each sharp endpoint is the constant $\textstyle \sum _ { i } a _ { i }$ plus the corresponding optima of $K - 1$ independent bipartite assignments. Thus each endpoint costs $O ( \sum _ { h } ( K - 1 ) m _ { h } ^ { 3 } )$ by solving one independent bipartite assignmentfor each anchor-client pair within every stratum.

Proof of Proposition 1. Anchoring client 1 makes the feasible set $\mathrm { S y m } ( m ) ^ { K - 1 }$ . For a fully additive cost,

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { m } C _ { i , \varpi _ { 2 } ( i ) , \dots , \varpi _ { K } ( i ) } = \sum _ { i = 1 } ^ { m } a _ { i } ^ { ( 1 ) } + \sum _ { k = 2 } ^ { K } \sum _ { i = 1 } ^ { m } a _ { \varpi _ { k } ( i ) } ^ { ( k ) } } } \\ { { = \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { m } a _ { j } ^ { ( k ) } , } } \end{array}\tag{39}
$$

because each $\varpi _ { k }$ is a permutation, proving identification. Anchor-additive costs instead give

$$
\sum _ { i = 1 } ^ { m } a _ { i } + \sum _ { k = 2 } ^ { K } \sum _ { i = 1 } ^ { m } C _ { i , \varpi _ { k } ( i ) } ^ { ( k ) } .\tag{40}
$$

Each summand has its own feasible permutation, so the total optimum is the sum of independent assignment optima. Their integrality proves sharpness and gives $\begin{array} { r } { { \hat { O } } ( \sum _ { h } ( K - 1 ) m _ { h } ^ { 3 } ) } \end{array}$ time. Interactions among non-anchor indices would break this separation. □

## C.5 BOOLEAN-ZETA INVERSION AND PARTITION CENSORING

The assignment results describe uncertainty about how fragments fit together. We next justify the co-membership coordinates used to measure the information removed by partitioning. Before partitioning, the full collection determines aggregate support weights. After partitioning, each coordinate is either retained exactly or erased, under the observation assumptions stated in Appendix B.2.

Proposition 2 (Invertible Boolean-zeta representation and censoring). For every type $r \in \mathcal { R } ,$ , the complete nonempty collection $M _ { | S | , r } ^ { \mathcal { H } } ( S )$ determines the aggregate support–type weights through

$$
w _ { r } ( A ) = \sum _ { B \supset A } ( - 1 ) ^ { | B | - | A | } M _ { | B | , r } ^ { \mathcal { H } } ( B ) .\tag{41}
$$

If multiplicity, exact retained weight, and type are observed, then Equation (6) holds for every $q \geq c .$

Proof. Fix a type � and regard $w _ { r }$ as a function on the Boolean lattice $2 ^ { V }$ , with zero mass on the empty relation. Equation (2) is its upper zeta transform: $\begin{array} { r } { M _ { | S | , r } ^ { \mathcal { H } } ( S ) = \sum _ { A \supseteq S } w _ { r } ( A ) } \end{array}$ . The Möbius function on the interval $[ S , A ]$ is $( - 1 ) ^ { | A | - | S | }$ , so Boolean-lattice Möbius inversion gives Equation (41). This inversion applies separately to each observed relation type.

For the mask, a copy of � contributed by relation instance � appears in a local fragment if and only if all nodes in � have the same client label. If they do, exactly one client contains that copy. Otherwise none does. Summing over all $A _ { e } \supseteq S$ proves Equation (6). The complete and retained order-� masses therefore satisfy the two identities below for every observed type � and order $q \colon$

$$
\| M _ { q , r } ^ { \mathcal { H } } \| _ { 1 } = \sum _ { e : r _ { e } = r } w _ { e } \binom { | A _ { e } | } { q } ,
$$

$$
\| \widetilde { M } _ { q , r } ( \cdot , z ) \| _ { 1 } = \sum _ { e : r _ { e } = r } w _ { e } \sum _ { k = 1 } ^ { K } \binom { | A _ { e } \cap V _ { k } | } { q } .\tag{42}
$$

Their ratio is exactly the retained order-� co-membership mass in (7) for type �.

## C.6 HORVITZ–THOMPSON ESTIMATION

The masking identity immediately suggests inverse-probability weighting under a known random partition design. It also reveals the cost of this correction: coordinates that rarely remain within one client acquire large variance. The theorem below gives both statements for each coordinate and then sums their squared errors.

Theorem 3 (Known randomized assignment). Fix a latent hypergraph H, an order $c \leq q \leq | V |$ , a type $r \in { \mathcal { R } }$ , and a node subset $S \in \binom { V } { q }$ . Let $Z : V \to [ K ]$ follow the declared randomized partition design and assume $\pi _ { S } = \operatorname* { P r } \{ Z | _ { S }$ is constant} $> 0 .$ The coordinate estimator satisfies

$$
\widehat { M } _ { q , r } ( S ) = \widetilde { M } _ { q , r } ( S , Z ) / \pi _ { S } ,
$$

$$
\mathbb { E } _ { Z } \widehat { M } _ { q , r } ( S ) = M _ { q , r } ^ { \mathcal { H } } ( S ) ,\tag{43}
$$

$$
\mathrm { V a r } _ { \cal Z } [ \widehat { M } _ { q , r } ( S ) ] = M _ { q , r } ^ { \mathcal { H } } ( S ) ^ { 2 } ( \pi _ { S } ^ { - 1 } - 1 ) .\tag{44}
$$

Ifevery q-set has positive inclusion probability, the mean squared Frobenius error is

$$
\mathbb { E } _ { \boldsymbol { Z } } \| \widehat { M } _ { q , r } - M _ { q , r } ^ { \mathcal { H } } \| _ { F } ^ { 2 } = \sum _ { S \in { \binom { V } { q } } } M _ { q , r } ^ { \mathcal { H } } ( S ) ^ { 2 } ( \pi _ { S } ^ { - 1 } - 1 ) .\tag{45}
$$

Under uniform independent assignment, the relative coordinate variance ofa nonzero coordinate is $K ^ { q - 1 } - \dot { 1 . } \stackrel { . } { I f } \pi _ { S } = \stackrel { . } { 0 } ,$ , no estimator measurable from the censored order-� co-membership array is design-unbiased for that coordinate over all type-� hypergraphs with size-� relation supports.

Proof. Let $I _ { S } ( Z ) ~ = ~ { \bf 1 } \{ Z | _ { S }$ is constant}. Then $I _ { S } ( \overline { { { Z } } } ) ~ \sim ~ \mathrm { B e r n o u l l i } ( \pi _ { S } )$ and $\widetilde { M } _ { q , r } ( S , Z ) =$ $I _ { S } ( Z ) M _ { q , r } ^ { \mathcal { H } } ( S )$ , which gives $( 4 3 ) { - } ( 4 4 )$ . Linearity of expectation applied to the sum of squared coordinate errors gives (45). Dependence among the indicators is allowed. Uniform assignment has $\pi _ { { \cal S } } = K ( 1 / K ) ^ { q } = \mathrm { \bar { \cal K } } ^ { 1 - q }$ . For $\pi _ { S } = 0$ , compare a q-uniform hypergraph with and without a weight-� edge on �. Their censored order-� co-membership arrays agree almost surely but the target coordinate differs by �. □

## C.7 OBSERVATIONAL EQUIVALENCE AND A TWO-POINT BOUND

Randomized-design unbiasedness concerns repeated releases of a fixed structure. With one release held fixed, the question is different: can two compatible structures have different target values? If they can, any estimator based on that release has the same output law under both structures and must incur error on at least one. We first state the resulting bound for the learner’s coarsened release, then prove the underlying two-point argument.

Corollary 2 (Identification-to-learning bridge under release coarsening). Let $y _ { \mathrm { { L } } } = \rho ( y _ { \mathrm { { A } } } )$ and hold the released training labels $Y ^ { \mathrm { t r } }$ fixed across all candidate structures. For the same fixed � and Γ, define $\mathcal { F } _ { \mathrm { L } } ( y _ { \mathrm { L } } , \Gamma ) = \{ \mathcal { H } \in \mathcal { M } _ { \Gamma } : \rho ( O _ { z , \Gamma } ( \mathcal { H } ) ) = y _ { \mathrm { L } } \}$ . Deterministic coarsening gives $\mathcal { F } _ { \mathrm { { A } } } ( y _ { \mathrm { { A } } } , \Gamma ) \subseteq$ $\mathcal { F } _ { \mathrm { { L } } } ( y _ { \mathrm { { L } } } , \Gamma )$ . Let $\phi : \mathcal { M } _ { \Gamma } \to \mathbb { R }$ be a scalar structural functional whose attainable endpoints on $\mathcal { F } _ { \mathrm { { A } } } ( y _ { \mathrm { { A } } } , \Gamma )$ are $\phi _ { - }$ and $\phi _ { + }$ . The minimax absolute risk over randomized rules $\widehat { \phi } = a ( \mathrm { y } _ { \mathrm { L } } , Y ^ { \mathrm { t r } } , \xi )$ , with algorithmic randomness $\xi$ independent ofH, satisfies this bound over the compatible class:

$$
\operatorname* { i n f } _ { a } \ \operatorname* { s u p } _ { \mathcal H \in \mathcal F _ { \mathrm { L } } ( \mathrm { y _ { L } , \Gamma } ) } \mathbb E _ { \xi } | \widehat \phi - \phi ( \mathcal H ) | \geq ( \phi _ { + } - \phi _ { - } ) / 2 .\tag{46}
$$

The proof uses the two endpoint structures, which remain indistinguishable after coarsening and therefore induce the same conditional output law.

Proposition 3 (Fixed-release observational equivalence). $F i x \ z , \ \Gamma ,$ , and a metric space $( \Theta , d )$ . A functional $\phi : \mathcal { M } _ { \Gamma } \to \Theta$ is point identified from the deterministic analyst release if and only if it is constant on every nonempty class $\mathcal { F } _ { \mathrm { { A } } } ( y _ { \mathrm { { A } } } , \Gamma )$ $I f \mathcal { H } _ { 0 } , \mathcal { H } _ { 1 } \in \mathcal { F } _ { \mathrm { A } } ( y _ { \mathrm { A } } , \Gamma )$ and $d ( \phi ( \mathcal { H } _ { 0 } ) , \phi ( \mathcal { H } _ { 1 } ) ) = \Delta ,$ every randomized estimator $\overset { \cdot } { \phi } = a \left( y _ { \mathrm { A } } , \xi \right)$ with $\xi \sim \nu$ independent ofthe latent hypergraph obeys

$$
\operatorname* { m a x } _ { b \in \{ 0 , 1 \} } \mathbb { E } _ { \xi } d ( \widehat { \phi } , \phi ( \mathcal { H } _ { b } ) ) \geq \Delta / 2 .\tag{47}
$$

Proof. If $\phi = g \circ O _ { z , \Gamma }$ , equal analyst releases imply equal values of $\phi .$ , so $\phi$ is constant on each compatible class. Conversely, if $\phi$ is constant on every class, define $g ( y _ { \mathrm { A } } )$ as that common value for any $\mathcal { \dot { H } } \in \mathcal { F } _ { \mathrm { A } } ( y _ { \mathrm { A } } , \Gamma )$ . Constancy makes this definition independent of the representative.

For the lower bound, $\widehat { \phi } = a ( y _ { \mathrm { { A } } } , \xi )$ has the same distribution at both compatible structures. Pointwise, the triangle inequality yields the following deterministic relation between the two endpoint errors:

$$
\Delta \leq d ( \widehat { \phi } , \phi ( \mathcal { H } _ { 0 } ) ) + d ( \widehat { \phi } , \phi ( \mathcal { H } _ { 1 } ) ) .\tag{48}
$$

Taking expectation over the common law � of $\xi$ shows that one endpoint risk reaches at least $\Delta / 2$ . □

ProofofCorollary 2. If $\mathcal { H } \in \mathcal { F } _ { \mathrm { A } } ( y _ { \mathrm { A } } , \Gamma )$ , applying $\rho$ to its analyst release gives $y _ { \mathrm { L } }$ , hence $\mathcal { H } \in$ $\mathcal { F } _ { \mathrm { { L } } } ( y _ { \mathrm { { L } } } , \Gamma )$ . Theorem 1 supplies compatible structures attaining $\phi _ { - }$ and $\phi _ { + } . \mathrm { ~ A ~ }$ rule measurable from $\sigma ( Y _ { \mathrm { L } } , Y ^ { \mathrm { t r } } , \xi )$ has one conditional output law under both endpoint structures because their training labels are held fixed. Their two absolute errors sum to at least $\phi _ { + } - \phi _ { - }$ , so one endpoint risk reaches $\left( \phi _ { + } - \phi _ { - } \right) / 2$ . This holds for every measurable rule, and taking the infimum gives Equation (46). □

Corollary 3 (Matching under hyperedge aliasing). In the two-client construction of Section $3 . 2 ,$ assume each $s i d e ^ { \prime } s$ � cross-status fragment rows are pairwise distinct and nonempty, share uninformative type and weight marks, and have no shared cross-client hyperedge identifier. A latent matching $\varpi \sim \mathrm { U n i f } ( \mathrm { S y m } ( m ) )$ is recovered completely with probability at most $1 / m ! .$ . An estimator returning � proposed matches has at most �/� correct matches in expectation.

Proof. The observation is invariant under all �! permutations. Conditional on it, each full permutation is equiprobable. A fixed proposed pair is correct with probability $1 / m$ . Linearity of expectation yields $s / m$ without an independence assumption across all returned proposals. □

## C.8 RELEASED-DATA FACTORIZATION AND OBSERVABLE SPECTRAL APPROXIMATION

We establish that FedASTRA’s complete trajectory factors through the learner release, then characterize the stability and expressiveness of its observable heat basis.

Proposition 4 (Permutation invariance and heat stability). For every relation-column permutation Π, $P _ { k } ( B _ { k } \Pi ) = P _ { k } ( B _ { k } )$ and $W _ { k , \theta }$ is unchanged. On the nonisolated block, $P _ { k } ^ { \mathrm { s y m } } = D _ { \nu , k } ^ { 1 / 2 } P _ { k } D _ { \nu , k } ^ { - 1 / 2 }$ has spectrum in [0, 1]. Hence $\| e ^ { t ( P _ { k } ^ { \mathrm { s y m } } - I ) } \| _ { 2 } \le 1 f o r t \ge 0 ,$ , and two such symmetric representatives on the same nodes satisfy

$$
\| e ^ { t ( P ^ { \mathrm { s y m } } - I ) } - e ^ { t ( \widetilde { P } ^ { \mathrm { s y m } } - I ) } \| _ { 2 } \leq t \| P ^ { \mathrm { s y m } } - \widetilde { P } ^ { \mathrm { s y m } } \| _ { 2 } .\tag{49}
$$

Proof. For $B ^ { \prime } = B \Pi , D _ { e } ^ { \prime } = \Pi ^ { \top } D _ { e } \Pi$ , while $D _ { \nu }$ and $I _ { 0 }$ are fixed. Thus

$$
B ^ { \prime } ( D _ { e } ^ { \prime } ) ^ { - 1 } ( B ^ { \prime } ) ^ { \top } = B \Pi ( \Pi ^ { \top } D _ { e } \Pi ) ^ { - 1 } \Pi ^ { \top } B ^ { \top } = B D _ { e } ^ { - 1 } B ^ { \top } ,\tag{50}
$$

proving invariance of � and all heat features. On the nonisolated block,

$$
P ^ { \mathrm { s y m } } = D _ { \nu } ^ { - 1 / 2 } B D _ { e } ^ { - 1 } B ^ { \top } D _ { \nu } ^ { - 1 / 2 } = G G ^ { \top } , \qquad G = D _ { \nu } ^ { - 1 / 2 } B D _ { e } ^ { - 1 / 2 } .\tag{51}
$$

Positive semidefiniteness and similarity to the row-stochastic $P$ give spectrum in [0, 1] and contraction. Isolated rows are absorbing. Duhamel’s identity gives

$$
e ^ { t ( P ^ { \mathrm { s y m } } - I ) } - e ^ { t ( \widetilde { P } ^ { \mathrm { s y m } } - I ) } = \int _ { 0 } ^ { t } e ^ { ( t - s ) ( P ^ { \mathrm { s y m } } - I ) } ( P ^ { \mathrm { s y m } } - \widetilde { P } ^ { \mathrm { s y m } } ) e ^ { s ( \widetilde { P } ^ { \mathrm { s y m } } - I ) } d s ,\tag{52}
$$

whose norm is bounded by Equation (49).

For one release, contraction in symmetric coordinates is equivalent to contraction in $\begin{array} { r } { \| \boldsymbol { x } \| _ { D _ { \nu } } ~ = } \end{array}$ $\| D _ { \nu } ^ { 1 / 2 } x \| _ { 2 } ;$ the perturbation bound compares symmetric representatives. In the two-client model, joining $F _ { 1 i } ^ { h }$ to $\bar { F } _ { 2 , \varpi _ { h } ( i ) } ^ { h }$ gives

$$
\begin{array} { r } { \{ \mathcal { H } _ { \varpi } : \pmb { \varpi } \in \prod _ { h } \mathrm { S y m } ( m _ { h } ) \} \subseteq \mathcal { F } _ { \mathrm { A } } ( y _ { \mathrm { A } } , \Gamma ) \subseteq \mathcal { F } _ { \mathrm { L } } ( \rho ( y _ { \mathrm { A } } ) , \Gamma ) . } \end{array}\tag{53}
$$

These matchings range over Equation (10) while preserving all local fragment multisets and hence the learner release.

Theorem 4 (Factorization of the complete federated learning trajectory). Let $\mathcal { P }$ fix the task split, training and validation labels, participation schedule, training-count aggregation weights, checkpoint rule, and hyperparameters; let � collect initialization, masking, minibatch, and optimizer randomness. Under exact arithmetic,for any client count and horizon �, the transcript T<sub>�</sub> ofall broadcasts, online, EMA and optimizer states, uploads, selected checkpoint, andfinal logits admits a measurable map

$$
\mathcal { T } _ { T } ( \mathcal { H } , \xi ) = F _ { T } \big ( \rho ( O _ { z , \Gamma } ( \mathcal { H } ) ) , \mathcal { P } , \xi \big ) .\tag{54}
$$

Thus compatible completions have identical transcripts under coupled $\xi ,$ , and identical conditional laws under a common H-independent distributionfor �.

Proof. Equation (50) makes $P _ { k }$ a measurable function of $[ \boldsymbol { B } _ { k } ] _ { \mathrm { c o l } }$ . Initialization, masked attributes, heat features, losses and gradients therefore depend only on $( Y _ { \mathrm { L } } , { \mathcal { P } } , \xi )$ and current states. An optimizer step and the EMA recursion preserve this dependence. Induction over local steps gives the uploads; training-count-weighted aggregation gives the next broadcast. Induction over rounds, followed by the fixed validation rule and final forward pass, proves the factorization. Coupling, then integration over $\xi ,$ gives the two equalities. □

Corollary 4 (Downstream invariance on learner compatibility classes). Under the preceding assumptions, equal learner releases imply, throughout training,

$$
\begin{array} { r } { \theta ^ { u } ( \mathcal { H } ) = \theta ^ { u } ( \mathcal { H } ^ { \prime } ) , \qquad \mathrm { l o g i t } _ { k , \nu } ^ { u } ( \mathcal { H } ) = \mathrm { l o g i t } _ { k , \nu } ^ { u } ( \mathcal { H } ^ { \prime } ) . } \end{array}\tag{55}
$$

This includes intermediate local and server states, validation selection, andfinal predictions, with equal conditional laws after averaging over $\xi .$

Proof. Apply Theorem 4; fixed evaluation labels also give identical Accuracy and Macro-F1. □

Corollary 5 (Density within observable spectral filters). For symmetric $P ^ { \mathrm { s y m } } ( Y _ { \mathrm { L } } )$ with spectrum in $[ 0 , 1 ]$ , every $f \in C ( [ 0 , 1 ] ) , \delta > 0$ , and $\Delta > \epsilon _ { \mathrm { o r d } }$ admit a finite $M _ { f }$ and coefficients depending only on $( f , \delta , \Delta )$ such that

$$
\left\| f ( P ^ { \mathrm { s y m } } ) - a _ { 0 } I - \sum _ { b = 1 } ^ { M _ { f } } a _ { b } e ^ { b \Delta ( P ^ { \mathrm { s y m } } - I ) } \right\| _ { 2 } < \delta .\tag{56}
$$

The grid $t _ { b } = b \Delta$ is feasible in Equation (20).

Proof. The change of variable $x = e ^ { \Delta ( \lambda - 1 ) }$ maps [0, 1] homeomorphically to $[ e ^ { - \Delta } , 1 ]$ . Polynomial approximation of $f ( 1 + \log ( x ) / \Delta )$ gives $\begin{array} { r } { p ( x ) = \sum _ { b = 0 } ^ { M _ { f } } a _ { b } x ^ { b } } \end{array}$ with uniform error below �. The spectral theorem yields Equation (56); choosing each softplus increment as $\Delta - \epsilon _ { \mathrm { o r d } } > 0$ realizes the grid. □

Corollary 6 (Information-boundary compliance and observable spectral approximation). On a positive-degree client block, define $\begin{array} { r c l } { \bar { f ( P ) } } & { = } & { D _ { \nu } ^ { - 1 / 2 } f ( P ^ { \mathrm { s y m } } ) D _ { \nu } ^ { \bar { 1 / 2 } } } \end{array}$ and $G _ { M _ { f } } ( P ) = a _ { 0 } I +$ $\begin{array} { r } { \sum _ { b = 1 } ^ { M _ { f } } a _ { b } e ^ { b \Delta ( P - I ) } } \end{array}$ using the preceding coefficients. For every conformable $U ,$

$$
\begin{array} { r } { \| ( f ( P ) - G _ { M _ { f } } ( P ) ) U \| _ { D _ { \nu } , F } \leq \delta \| U \| _ { D _ { \nu } , F } , \qquad \| U \| _ { D _ { \nu } , F } = \| D _ { \nu } ^ { 1 / 2 } U \| _ { F } . } \end{array}\tag{57}
$$

Both these transforms and the complete training transcript are shared across the learner compatibility class,forfixed $\dot { \mathcal { S } }$ and $\xi .$

Proof. Conjugate Equation (56) by $D _ { \nu } ^ { 1 / 2 }$ and use $\| A U \| _ { F } \le \| A \| _ { 2 } \| U \| _ { F }$ . Invariance follows from Theorem 4. □

Density concerns the union of finite basis widths. It gives neither fixed-� universality nor a prediction or optimization guarantee for a fitted classifier.

Certified evaluation. All times share the recurrence $U , P U , P ^ { 2 } U , \dots$ For the auxiliary analytic variable $N _ { t } ^ { \mathrm { a u x } } \sim \operatorname { P o i s s o n } ( t )$ , deterministic tail-collapsed evaluation is

$$
\widehat { \mathsf { K } } _ { t , L } U = \sum _ { \ell = 0 } ^ { L - 1 } e ^ { - t } \frac { t ^ { \ell } } { \ell ! } P ^ { \ell } U + \left( 1 - \sum _ { \ell = 0 } ^ { L - 1 } e ^ { - t } \frac { t ^ { \ell } } { \ell ! } \right) P ^ { L } U .\tag{58}
$$

For active times $\mathcal { T } _ { \mathrm { h e a t } } = \{ t _ { b j } : b \in [ R ] , j \in [ d ] \}$ , choose the smallest integer $L _ { \mathrm { t a i l } }$ satisfying

$$
\operatorname* { m a x } _ { t \in { \mathcal { T } _ { \mathrm { h e a t } } } } { \mathrm { P r } } ( N _ { t } ^ { \mathrm { a u x } } > L _ { \mathrm { t a i l } } ) \le \varepsilon _ { \mathrm { t a i l } } , \qquad \operatorname* { m a x } _ { t \in { \mathcal { T } _ { \mathrm { h e a t } } } } \{ { \mathrm { P r } } ( N _ { t } ^ { \mathrm { a u x } } = L _ { \mathrm { t a i l } } ) + 2 { \mathrm { P r } } ( N _ { t } ^ { \mathrm { a u x } } > L _ { \mathrm { t a i l } } ) \} \le \varepsilon _ { \mathrm { t a i l } } .\tag{59}
$$

These bound forward and time-derivative coefficient masses. We use $\varepsilon _ { \mathrm { t a i l } } = 1 0 ^ { - 6 }$ , float64 logspace probabilities, and record both bounds and $L _ { \mathrm { t a i l } }$ for each validation-selected native checkpoint. Online and EMA branches use their own times. At $R = 0$ , bypass the recurrence and set $L _ { \mathrm { t a i l } } = 0 $ configurations requiring $L _ { \mathrm { t a i l } } > 4 0 9 6$ are refused. Floating-point propagation and accumulation errors are separate from this coefficient certificate.

## C.9 OBSERVABLE DIFFUSION AND PREDICTIVE DYNAMICS

Diffusion time evolves features at fixed incidence and encoder; local training steps evolve the encoder and its EMA target. The following results distinguish these two processes.

Theorem 5 (Continuous-time diffusion under observational equivalence). $F i x { \ y _ { \mathrm { { L } } } }$ , client $k ,$ , and $U _ { k } ^ { 0 } \in \mathbb { R } ^ { | V _ { k } | \times d }$ . Put $d _ { k , \nu } = ( D _ { \nu , k } ) _ { \nu \nu }$ and $V _ { k , + } = \{ \nu : d _ { k , \nu } > 0 \}$ . Let $D _ { k , + } b e \ : D _ { \nu , k }$ restricted to these nodes, and use + likewise for other restrictions. For $t \geq 0$ , the trajectory

$$
\begin{array} { r } { U _ { k } ( t ) = e ^ { t ( P _ { k } - I ) } U _ { k } ^ { 0 } } \end{array}\tag{60}
$$

uniquely solves $\dot { U } _ { k } = ( P _ { k } - I ) U _ { k } , U _ { k } ( 0 ) = U _ { k } ^ { 0 } ,$ . For the incidence-induced components $\mathcal { G } _ { k }$ on $V _ { k , + }$ , define

$$
( \Omega _ { k , + } U _ { k , + } ^ { 0 } ) _ { \nu } = \frac { \sum _ { u \in C } d _ { k , u } U _ { k , u } ^ { 0 } } { \sum _ { u \in C } d _ { k , u } } , \qquad \nu \in C \in \mathcal { G } _ { k } .\tag{61}
$$

Writing $A = U _ { k , + } ( t ) - \mathsf Q _ { k , + } U _ { k , + } ^ { 0 }$ <sup>0</sup><sub>�,+</sub> and $\| A \| _ { D _ { k , + } , F } ^ { 2 } = \mathrm { t r } ( A ^ { \top } D _ { k , + } A )$ gives

$$
\begin{array} { l } { \displaystyle \frac { d } { d t } \frac { 1 } { 2 } \| A \| _ { D _ { k , + } , F } ^ { 2 } = - \mathrm { t r } [ A ^ { \top } D _ { k , + } ( I - P _ { k , + } ) A ] } \\ { = - \displaystyle \frac { 1 } { 2 } \sum _ { e } \frac { 1 } { d _ { e , k } } \sum _ { u , \nu \in e } \| U _ { k , u } ( t ) - U _ { k , \nu } ( t ) \| _ { 2 } ^ { 2 } \leq 0 , } \end{array}\tag{62}
$$

where � ranges over retained columns and $d _ { e , k } = ( D _ { e , k } ) _ { e e }$ . If the nonstationary subspace is nonzero, let

$$
\lambda _ { \mathrm { g a p } , k } = \operatorname* { m i n } \{ 1 - \lambda : \lambda \in \mathrm { s p e c } ( P _ { k , + } ^ { \mathrm { s y m } } ) , \lambda < 1 \} > 0 , \quad P _ { k , + } ^ { \mathrm { s y m } } = D _ { k , + } ^ { 1 / 2 } P _ { k , + } D _ { k , + } ^ { - 1 / 2 } .\tag{63}
$$

Then

$$
\begin{array} { r } { \| \boldsymbol { U } _ { k , + } ( t ) - \boldsymbol { \Omega } _ { k , + } \boldsymbol { U } _ { k , + } ^ { 0 } \| _ { D _ { k , + } , F } \leq e ^ { - \lambda _ { \mathrm { g a p } , k } t } \| \boldsymbol { U } _ { k , + } ^ { 0 } - \boldsymbol { \Omega } _ { k , + } \boldsymbol { U } _ { k , + } ^ { 0 } \| _ { D _ { k , + } , F } . } \end{array}\tag{64}
$$

A zero nonstationary subspace gives zero residual. Isolated nodes stay fixed, and compatible completions with equal initial features have identical trajectories.

Proof. The matrix exponential uniquely solves the finite linear system, and column invariance gives the common trajectory. On $\bar { V _ { k , + } } , \bar { D } _ { k , + } P _ { k , + } = B _ { k , + } D _ { e , k } ^ { - 1 } B _ { k , + } ^ { \top }$ is symmetric and $P _ { k , + } \mathsf { Q } _ { k , + } =$ $\mathsf { Q } _ { k , + } P _ { k , + } = \mathsf { Q } _ { k , + }$ . Differentiating the energy and expanding the quadratic form gives

$$
\mathrm { t r } [ A ^ { \top } D _ { k , + } ( I - P _ { k , + } ) A ] = \frac 1 2 \sum _ { e } \frac { 1 } { d _ { e , k } } \sum _ { u , \nu \in e } \| A _ { u } - A _ { \nu } \| _ { 2 } ^ { 2 } .\tag{65}
$$

Its null space consists of componentwise constant matrices. In symmetric coordinates, $\Omega _ { k , + }$ is the orthogonal eigenvalue-one projection; every other heat multiplier is at most $e ^ { - \lambda _ { \mathrm { g a p } , k } t }$ . The spectral theorem proves convergence. The isolated block is the identity. □

Corollary 7 (Ordered sampling of the diffusion trajectory). Equation (20) gives, for each coordinate $j ,$

$$
0 = t _ { 0 j } < t _ { 1 j } < \cdot \cdot \cdot < t _ { R j } , \qquad W _ { k , \theta } ^ { ( j ) } = [ U _ { k , : j } ( t _ { 0 j } ) , \dots , U _ { k , : j } ( t _ { R j } ) ] ,\tag{66}
$$

where $W _ { k , \theta } ^ { ( j ) }$ selects columns $j , d + j , \dotsc , R d + j$ and $U _ { k , : j } ( t ) = \mathsf { K } _ { k } ( t ) U _ { k , : j } ^ { 0 }$ . Thefeasible grid $t _ { b } = b \Delta ,$ $\Delta > \epsilon _ { \mathrm { o r d } } ,$ yields the dense algebra

$$
\operatorname { s p a n } \{ 1 , e ^ { b \Delta ( \lambda - 1 ) } : b \geq 1 \} \subset C ( [ 0 , 1 ] ) .\tag{67}
$$

Density over increasing finite widths follows from Corollary 5.

Masked prediction and target dynamics. Within round �, let � index local steps, � denote optimizer state, $Y _ { \mathrm { L } , k } = \left( X _ { k } , [ B _ { k } ] _ { \mathrm { c o l } } \right)$ , and $\xi _ { k , u , s }$ contain masks, minibatches, and optimizer randomness. The update has the form

$$
\begin{array} { r } { \left( \theta _ { k , u , s } , o _ { k , u , s } \right) = \Psi _ { k , u , s } \left( \theta _ { k , u , s - 1 } , o _ { k , u , s - 1 } , \bar { \vartheta } _ { k , u , s - 1 } , Y _ { \mathrm { L } , k } , \mathcal { P } , \xi _ { k , u , s } \right) . } \end{array}\tag{68}
$$

With $\vartheta _ { k , u , s } = \mathrm { c t x } ( \theta _ { k , u , s } )$ containing the encoder and raw time increments, the target satisfies

$$
\bar { \vartheta } _ { k , u , s } = \mu \bar { \vartheta } _ { k , u , s - 1 } + ( 1 - \mu ) \vartheta _ { k , u , s } = \mu ^ { s } \bar { \vartheta } _ { k , u , 0 } + ( 1 - \mu ) \sum _ { a = 1 } ^ { s } \mu ^ { s - a } \vartheta _ { k , u , a } .\tag{69}
$$

For $e _ { k , u , s } = \vartheta _ { k , u , s } - \bar { \vartheta } _ { k , u , s }$ and $\Delta _ { k , u , s } = \vartheta _ { k , u , s } - \vartheta _ { k , u , s - 1 }$ , subtraction and iteration give

$$
e _ { k , u , s } = \mu e _ { k , u , s - 1 } + \mu \Delta _ { k , u , s } , \qquad \| e _ { k , u , s } \| \leq \mu ^ { s } \| e _ { k , u , 0 } \| + \mu \sum _ { a = 1 } ^ { s } \mu ^ { s - a } \| \Delta _ { k , u , a } \| .\tag{70}
$$

Thus a frozen online context has lag contraction factor $\mu .$ . The input masking moments in Equation (110) do not imply unbiased nonlinear encodings or gradients.

Lemma 1 (Observation invariance throughout training). For fixed P and coupled �, the masked loss, online, optimizer and EMA states, uploads, broadcasts, selected checkpoint, andfinal logits are measurable in $( Y _ { \mathrm { L } } , { \mathcal { P } } , \xi )$ and agree across its compatibility class.

Proof. This is Theorem 4 applied to the local recursions above and the aggregation

$$
\theta _ { k } ^ { u + 1 } = \mathsf { U } _ { k , u } ( \theta ^ { u } , Y _ { \mathrm { L } , k } , \mathcal { P } , \xi _ { k , u } ) , \qquad \theta ^ { u + 1 } = \frac { \sum _ { k \in C _ { u } } n _ { k } ^ { \mathrm { t r } } \theta _ { k } ^ { u + 1 } } { \sum _ { k \in C _ { u } } n _ { k } ^ { \mathrm { t r } } } .\tag{71}
$$

Diffusion dissipation and EMA lag concern distinct dynamics; neither is a Lyapunov function for joint nonlinear training. Observation invariance also supplies no formal privacy guarantee for communicated updates.

## C.10 DOWNSTREAM RISK WITH ADDITIONAL INFORMATION

We compare fitted predictors under one joint law of data, training randomness, and held-out query nodes. Conditional averaging and information comparison are classical (Blackwell, 1953; Chen et al., 2020); here they connect prediction variation to the assignment geometry of Theorem 1.

Information and fitted risk. For a queried node with class $y \in [ C ] , C \geq 2$ , let $O _ { k }$ contain client features and anonymous incidence, the query, the parameter-only transcript through validation selection, and exogenous randomness. Held-out labels, test outcomes, auxiliary messages, and states trained using those messages are excluded. Set $S _ { 0 } = \sigma ( O _ { k } )$ ; the FedASTRA predictor � is S<sub>0</sub>-measurable. This permits federated parameter exchange while keeping structural computation local. Let $A _ { \mathrm { a u x } }$ include arbitrary additional information and any training state needed for an enriched predictor $G .$ , measurable in $S _ { 1 } = S _ { 0 } \vee \sigma ( A _ { \mathrm { a u x } } )$ . The predictors may have different learned parameters. A conditional structural audit may fix analyst releases and cost arrays in the base, but such conditioning does not grant deployment access to them.

Define, with one-hot label $e _ { y }$

$$
p _ { 0 } = \mathbb { E } [ e _ { y } \mid S _ { 0 } ] , \qquad p _ { 1 } = \mathbb { E } [ e _ { y } \mid S _ { 1 } ] , \qquad J _ { C } = I _ { C } - { \bf 1 1 } ^ { \top } / C ,\tag{72}
$$

$$
s _ { F } = J _ { C } \log \mathrm { i t } _ { F } , \qquad s _ { G } = J _ { C } \log \mathrm { i t } _ { G } , \qquad \bar { s } _ { G } = \mathbb { E } [ s _ { G } \mid S _ { 0 } ] .\tag{73}
$$

Assume square-integrable logits. Centering preserves softmax. Write $\begin{array} { r } { \mathrm { l s e } ( s ) = \log \sum _ { c } e ^ { s _ { c } } , \mathcal { R } _ { \mathrm { C E } } ( s ) = } \end{array}$ $\mathbb { E } [ \mathrm { l s e } ( s ) - s _ { \mathrm { y } } ]$ , and $\mathcal { R } _ { \mathrm { l o g } } ( q ) = \mathbb { E } [ - \log q _ { \mathrm { y } } ]$ with the usual convention at zero. Conditional log risk equals conditional entropy plus expected KL divergence. Consequently,

$$
\operatorname* { i n f } _ { \substack { q \ S _ { 0 } \mathrm { - m e a s u r a b l e } } } \mathcal { R } _ { \log } ( q ) - \operatorname* { i n f } _ { \substack { q \ S _ { 1 } \mathrm { - m e a s u r a b l e } } } \mathcal { R } _ { \log } ( q ) = \bar { J } _ { \mathrm { a d d } } : = \mathbb { E } D _ { \mathrm { K L } } ( p _ { 1 } \| p _ { 0 } ) \geq 0 .\tag{74}
$$

Indeed, the minimizers are $p _ { 0 } , p _ { 1 }$ and the tower property identifies the entropy difference. For fitted probability rules of finite log risk, put $\mathcal { E } _ { F } = \mathbb { E } D _ { \mathrm { K L } } ( p _ { 0 } \Vert q _ { F } )$ and $\mathcal { E } _ { G } = \mathbb { E } D _ { \mathrm { K L } } ( p _ { 1 } \Vert q _ { G } )$ . Subtracting $\mathcal { R } _ { \mathrm { l o g } } ( q _ { F } ) = H ( y \mid S _ { 0 } ) + \mathcal { E } _ { F }$ from $\mathcal { R } _ { \mathrm { l o g } } ( q _ { G } ) = H ( y \mid S _ { 1 } ) + \mathcal { E } _ { G }$ proves Equation (16). Thus � strictly improves on � exactly when $\mathcal { E } _ { G } - \mathcal { E } _ { F } > \mathcal { I } _ { \mathrm { a d d } }$

Theorem 6 (Cross-entropy comparison with arbitrary additional information). In the model above, define

$$
V _ { \mathrm { a u x } } = \mathbb { E } \Vert s _ { G } - \bar { s } _ { G } \Vert _ { 2 } ^ { 2 } , \qquad \Delta _ { \mathrm { l a b e l } } = \mathbb { E } \Vert p _ { 1 } - p _ { 0 } \Vert _ { 2 } ^ { 2 } ,\tag{75}
$$

$$
\delta _ { F } = \mathcal { R } _ { \mathrm { C E } } ( s _ { F } ) - \mathcal { R } _ { \mathrm { C E } } ( \bar { s } _ { G } ) , \qquad C _ { \mathrm { l a b e l } } = \mathbb { E } \langle p _ { 1 } - p _ { 0 } , s _ { G } - \bar { s } _ { G } \rangle .\tag{76}
$$

$$
F o r D _ { \mathrm { l s e } } ( a , b ) = \mathrm { l s e } ( a ) - \mathrm { l s e } ( b ) - \langle \nabla \mathrm { l s e } ( b ) , a - b \rangle ,
$$

$$
\mathcal { R } _ { \mathrm { C E } } ( s _ { G } ) - \mathcal { R } _ { \mathrm { C E } } ( s _ { F } ) = \mathbb { E } D _ { \mathrm { l s e } } ( s _ { G } , \bar { s } _ { G } ) - C _ { \mathrm { l a b e l } } - \delta _ { F } .\tag{77}
$$

$I f \| s _ { G } \| _ { \infty } \leq B _ { \mathrm { l o g i t } } <$ ∞ almost surely, then

$$
\mathcal { R } _ { \mathrm { C E } } ( s _ { G } ) - \mathcal { R } _ { \mathrm { C E } } ( s _ { F } ) \geq \frac { e ^ { - 2 B _ { \mathrm { l o g i t } } } } { 2 C } V _ { \mathrm { a u x } } - \sqrt { \Delta _ { \mathrm { l a b e l } } V _ { \mathrm { a u x } } } - \delta _ { F } .\tag{78}
$$

A positive right side gives strictly smaller riskfor �. Under label sufficiency $p _ { 1 } = p _ { 0 }$ , the alignment term $C _ { \mathrm { l a b e l } }$ vanishes.

Proof. Put $r = s _ { G } - \bar { s } _ { G } , \mathrm { s o } \mathbb { E } [ r \mid S _ { 0 } ] = 0$ . The Bregman linear term has zero expectation, and

$$
\begin{array} { r } { \mathbb { E } \langle e _ { y } , r \rangle = \mathbb { E } \langle p _ { 1 } , r \rangle = \mathbb { E } \langle p _ { 1 } - p _ { 0 } , r \rangle = C _ { \mathrm { l a b e l } } . } \end{array}\tag{79}
$$

Subtracting risks proves Equation (77); in particular, taking $s _ { F } = \bar { s } _ { G }$ (so $\delta _ { F } = 0 )$ gives

$$
\mathcal { R } _ { \mathrm { C E } } ( s _ { G } ) - \mathcal { R } _ { \mathrm { C E } } ( \bar { s } _ { G } ) = \mathbb { E } D _ { \mathrm { l s e } } ( s _ { G } , \bar { s } _ { G } ) - C _ { \mathrm { l a b e l } } .
$$

For $p = \operatorname { s o f t m a x } ( s )$ with $\| s \| _ { \infty } \leq B _ { \mathrm { l o g i t } } , p _ { c } \geq e ^ { - 2 B _ { \mathrm { l o g i t } } } / C$ . For $\nu \perp 1$ and $a = p ^ { \top } \nu$

$$
\nu ^ { \top } \nabla ^ { 2 } \log ( s ) \nu = \sum _ { c } p _ { c } ( \nu _ { c } - a ) ^ { 2 } \geq \frac { e ^ { - 2 B _ { \log \mathrm { i t } } } } { C } \sum _ { c } ( \nu _ { c } - a ) ^ { 2 } \geq \frac { e ^ { - 2 B _ { \log \mathrm { i t } } } } { C } \| \nu \| _ { 2 } ^ { 2 } .\tag{80}
$$

The segment from $\bar { s } _ { G }$ to $s _ { G }$ remains bounded and centered. Integral Taylor remainder gives $D _ { \mathrm { l s e } } ( s _ { G } , \bar { s } _ { G } ) \geq e ^ { - 2 B _ { \mathrm { l o g i t } } } \Vert r \Vert _ { 2 } ^ { 2 } / ( 2 C )$ , proving the bound and Equation (17). □

The Bregman term charges variation around the base-conditional mean; $C _ { \mathrm { l a b e l } }$ credits alignment with new label signal, and the signed $\delta _ { F }$ retains fitting quality. If $\boldsymbol { A } _ { \mathrm { a u x } }$ is base-measurable, both $V _ { \mathrm { a u x } }$ and $\Delta _ { \mathrm { l a b e l } }$ vanish.

Proposition 5 (Assignment geometry of auxiliary prediction variation). Condition on base information fixing the cost arrays and intercepts. In the exact strata of Theorem 1, declare independent uniform candidate permutations $\varpi _ { h } \in \mathrm { S y m } ( m _ { h } )$ and affine logits

$$
[ \mathrm { l o g i t } _ { G } ] _ { c } = b _ { c } + \sum _ { h } \sum _ { i = 1 } ^ { m _ { h } } [ C _ { c } ^ { h } ] _ { i , { \varpi _ { h } ( i ) } } , \qquad c \in [ C ] .\tag{81}
$$

Set $\begin{array} { r } { \bar { C } ^ { h } = C ^ { - 1 } \sum _ { c } C _ { c } ^ { h } } \end{array}$ and $\widetilde { C } _ { c } ^ { h } = J _ { m _ { h } } ( C _ { c } ^ { h } - \bar { C } ^ { h } ) J _ { m _ { h } }$ . Then

$$
\mathbb { E } [ \| s _ { G } - \bar { s } _ { G } \| _ { 2 } ^ { 2 } | s _ { 0 } ] = \sum _ { h : m _ { h } \geq 2 } \frac { 1 } { m _ { h } - 1 } \sum _ { c = 1 } ^ { C } \| \widetilde { C } _ { c } ^ { h } \| _ { F } ^ { 2 } .\tag{82}
$$

Averaging gives $V _ { \mathrm { a u x } } .$ . A scalar cost $C ^ { h }$ has variance $\| J _ { m _ { h } } C ^ { h } J _ { m _ { h } } \| _ { F } ^ { 2 } / ( m _ { h } - 1 ) f o r m _ { h } \geq 2 ,$ singleton strata have zero variance.

Proof. Double centering a fixed cost $C _ { * }$ <sub>∗</sub> changes assignment sums by a constant. For $D _ { * } = J _ { m } C _ { * } J _ { m }$ and uniform �, $\begin{array} { r } { S = \bar { \sum _ { i } } [ D _ { * } ] _ { i , \varpi ( i ) } } \end{array}$ has zero mean. Equal-row terms in $\mathbb { E } S ^ { 2 }$ contribute $\| D _ { * } \| _ { F } ^ { 2 } / m ;$ distinct-row terms give

$$
\frac { 1 } { m ( m - 1 ) } \sum _ { i \neq i ^ { \prime } } \sum _ { j \neq j ^ { \prime } } [ D _ { * } ] _ { i j } [ D _ { * } ] _ { i ^ { \prime } j ^ { \prime } } = \frac { \| D _ { * } \| _ { F } ^ { 2 } } { m ( m - 1 ) } .\tag{83}
$$

The equality uses $\sum _ { i ^ { \prime } \neq i , j ^ { \prime } \neq j } [ D _ { * } ] _ { i ^ { \prime } j ^ { \prime } } = [ D _ { * } ] _ { i j }$ , from zero row and column sums. Adding gives $\operatorname { V a r } ( S ) = \| D _ { * } \| _ { F } ^ { 2 } / ( m - 1 )$ . Class centering replaces $C _ { c } ^ { h } \thinspace { \sf b y } \thinspace C _ { c } ^ { h } - \bar { C } ^ { h }$ ; summing the scalar identities over classes and independent strata proves the claim. □

Thus the same interaction detects nonidentification and measures variation under the declared matching law. A scalar assignment interval of width �<sub>∗</sub> gives variance at most $w _ { * } ^ { 2 } / 4$ . Uniform endpoint bounds for $C _ { c } ^ { h } - \bar { C } ^ { \breve { h } }$ , together with uniformly bounded centered intercepts, can supply $B _ { \mathrm { l o g i t } }$ across base realizations. Class centering is essential because common logit shifts do not affect predictions. Nonlinear auxiliary rules remain covered by Theorem 6.

Corollary 8 (A conditional advantage for the fitted FedASTRA predictor). Under the bounded-logit assumptions above, let $F _ { R }$ be the fitted parameter-only FedASTRA predictor with � positive heat channels. $I f \delta _ { F _ { R } } \leq \varepsilon _ { R } f o r$ a stated $\varepsilon _ { R } \geq 0 ,$ , then

$$
\frac { e ^ { - 2 B _ { \mathrm { l o g i t } } } } { 2 C } V _ { \mathrm { a u x } } - \sqrt { \Delta _ { \mathrm { l a b e l } } V _ { \mathrm { a u x } } } > \varepsilon _ { R } \quad \Longrightarrow \quad \mathcal { R } _ { \mathrm { C E } } ( s _ { F _ { R } } ) < \mathcal { R } _ { \mathrm { C E } } ( s _ { G } ) .\tag{84}
$$

For a conditionally label-independent candidate channel, the square-root term is zero; affine assignment logits have the variance in Proposition 5.

Proof. Transcript factorization makes $F _ { R }$ base-measurable. Substitute the residual bound into Equation (78); conditional label independence gives $p _ { 1 } = p _ { 0 }$ □

Under label sufficiency $p _ { 1 } = p _ { 0 }$ , equivalently $y \perp A _ { \mathrm { a u x } } \mid S _ { 0 }$ , the two information levels have the same conditional entropy. Hence, for $p _ { F _ { R } } = \mathrm { s o f t m a x } ( s _ { F _ { R } } )$ and $p _ { G } = \mathrm { s o f t m a x } ( s _ { G } )$

$$
\mathcal { R } _ { \mathrm { l o g } } ( p _ { G } ) - \mathcal { R } _ { \mathrm { l o g } } ( p _ { F _ { R } } ) = \mathbb { E } D _ { \mathrm { K L } } ( p _ { 0 } \Vert p _ { G } ) - \mathbb { E } D _ { \mathrm { K L } } ( p _ { 0 } \Vert p _ { F _ { R } } ) .\tag{85}
$$

If the fixed-� class approximates $p _ { 0 }$ arbitrarily well in expected KL, its infimum equals both information-specific Bayes risks. A fitted error at most $\varepsilon _ { R }$ strictly improves on a specified � with larger KL error. Label sufficiency, fixed-model approximation, and fitting are separate assumptions; spectral density and correspondence invariance imply none of them.

Proposition 6 (Target direction of the masked alignment component). Freeze the EMA context $\bar { \vartheta } _ { k }$ for one local update. Let $O _ { \mathrm { m a s k } }$ be the predictor’s masked representation and $U _ { \mathrm { t a r } }$ its full-view, layer-normalized and unit-normalized target, with $\| U _ { \mathrm { t a r } } \| _ { 2 } = 1$ almost surely. Then

$$
\operatorname* { i n f } _ { \substack { u : \| u ( O _ { \mathrm { m a s k } } ) \| _ { 2 } = 1 } } \mathbb { E } [ 1 - \langle u ( O _ { \mathrm { m a s k } } ) , U _ { \mathrm { t a r } } \rangle ] = 1 - \mathbb { E } \| \mathbb { E } [ U _ { \mathrm { t a r } } \mid O _ { \mathrm { m a s k } } ] \| _ { 2 } .\tag{86}
$$

For $m = \mathbb { E } [ U _ { \mathrm { t a r } } \mid O _ { \mathrm { m a s k } } ] \neq 0$ , the minimizing direction is $m / \| m \| _ { 2 } ;$ when $m = 0 ,$ , every unit direction is optimal.

Proof. Conditional expected alignment is $\langle u , m \rangle \leq \| m \| _ { 2 }$ by Cauchy–Schwarz, with equality in the stated directions. Integrate the optimum. □

This concerns the frozen-target alignment component with unrestricted directions. Anti-collapse regularization, two-view agreement, moving targets, and finite predictors remain in the full objective (112) and its optimization analysis.

Classification metrics. For base-measurable $p _ { F } = \mathrm { s o f t m a x } ( s _ { F } )$ , let $h _ { F } = \arg \operatorname* { m a x } _ { c } p _ { F , c }$ and $h _ { 0 } = \arg \operatorname* { m a x } _ { c } p _ { 0 , c }$ with a fixed tie rule. Comparing their selected coordinates and applying Pinsker and Cauchy–Schwarz bounds classification regret by $\mathbb { E } \Vert p _ { 0 } - p _ { F } \Vert _ { 1 } \leq \sqrt { 2 \mathbb { E } D _ { \mathrm { K L } } ( p _ { 0 } \Vert p _ { F } ) }$ . This is regret against the base Bayes rule; a surrogate-risk ordering alone need not order two fitted classifiers accuracies. If additionally $p _ { 1 } = p _ { 0 }$ , the top-two conditional probability gap is at least $\kappa _ { \mathrm { c l s } } > 0$ and $h _ { F _ { R } } \ : = \ : h _ { 0 }$ almost surely, then $\operatorname { A c c } ( F _ { R } ) { \overset { - } { - } } \operatorname { A c c } ( G ) \geq \kappa _ { \operatorname { c l s } } { \bar { \operatorname* { P r } } } ( h _ { G } \neq h _ { 0 } )$ by conditioning on $S _ { 1 }$ Macro-F1 also depends on classwise confusion probabilities: increased true positives and decreased false positives and false negatives suffice, whereas cross-entropy ordering alone does not determine it. Both metrics use the same held-out evaluation protocol.

## C.11 RANDOMIZED RECOVERY FROM DIFFUSION FEATURES AND LOCAL PREDICTION RISK

Heat features retain permitted local information when they recover the diffusion operator and its pairwise projection, and that projection determines the observed incidence. These are separate conditions, independent of unknown correspondence across clients. Incidence is binary; column multiplicities are retained unless simplicity is assumed.

Lemma 2 (Projection and recovery from diffusion features). Let � have � rows and nonempty columns, and let � be Equation (18), including absorbing isolated nodes. For � $\neq j , P _ { i j } > 0$ exactly when a column contains both nodes, so its off-diagonal support determines the pairwise projection � (�). Fix $\Delta > 0 , d , \ell \geq 1 , U \in \mathbb { R } ^ { n \times d }$ , and $\begin{array} { r } { \dot { H _ { \Lambda } } = e ^ { \sum ( P - I ) } } \end{array}$ . Define

$$
Q _ { - } = [ U , H _ { \Delta } U , \ldots , H _ { \Delta } ^ { \ell - 1 } U ] , \qquad Q _ { + } = [ H _ { \Delta } U , \ldots , H _ { \Delta } ^ { \ell } U ] .\tag{87}
$$

If rank $( Q _ { - } ) = n ,$ then in exact arithmetic

$$
H _ { \Delta } = Q _ { + } Q _ { - } ^ { \dagger } , \qquad P = I + \Delta ^ { - 1 } \log H _ { \Delta } ,\tag{88}
$$

using the Moore–Penrose inverse and principal matrix logarithm. Times �Δ are permitted commontime ordered heat channels when $\Delta > \epsilon _ { \mathrm { o r d } } .$

Proof. For positive-degree �, $\begin{array} { r } { P _ { i j } = d _ { i } ^ { - 1 } \sum _ { e } B _ { i e } B _ { j e } / | e | ; } \end{array}$ ; nonnegativity proves the support claim, and isolated rows add no pairs. Since $Q _ { + } = H _ { \Delta } Q _ { - }$ , full row rank proves the first inverse. On positivedegree nodes, � is similar to the positive semidefinite symmetric matrix $ { D _ { \nu } ^ { - 1 / 2 } } B  { D _ { e } ^ { - 1 } }  { B ^ { \top } }  { D _ { \nu } ^ { - 1 / 2 } }$ and is stochastic; its spectrum lies in $\bar { \boldsymbol { \lceil 0 , 1 \rceil } }$ . The isolated block is the identity. Thus � is diagonalizable with real spectrum and log $e ^ { \Delta ( P - \dot { I } ) } = \dot { \Delta } ( P - I )$ □

Randomized full rank. Retain $U = [ X , Z ]$ , where $Z \in \mathbb { R } ^ { n \times d _ { \mathrm { p t } } }$ <sup>r</sup> has independent standard Gaussian entries independent of the hypergraph, attributes, and labels. Let $n _ { \mathrm { e i g } }$ count the distinct eigenvalues of $P ,$ with largest multiplicity $g _ { \mathrm { m a x } }$ . Then $\ell \geq n _ { \mathrm { e i g } }$ and $d _ { \mathrm { p r } } \geq g _ { \mathrm { m a } } ^ { \overline { { } } }$ give rank $( Q _ { - } ) = n$ almost surely. Indeed, in $P = S \Lambda S ^ { - 1 }$ , each eigenspace block of $S ^ { - 1 } Z$ has full row rank almost surely. A vector annihilating the Krylov matrix yields a Vandermonde system in the distinct eigenvalues of $H _ { \Delta } ;$ ; invertibility forces every eigenspace coefficient, and hence the vector, to vanish. In particular, $d _ { \mathrm { p r } } \geq n , \ell = 1$ suffices for every ${ \bar { P } } .$ . These probes remain local and are absent from reported runs; the finite attribute encoder needs its own rank check.

A reconstructible random family. Let ${ \mathcal { E } } _ { n } \subseteq { \mathcal { A } } _ { n }$ be any random simple hypergraph, where candidate supports $\textstyle e \in \mathcal { A } _ { n }$ satisfy $| e | \ge 2$ . Let Ω contain unordered distinct edge pairs with intersection size at least two, and $\Omega _ { 3 }$ unordered triples supporting a Berge triangle: distinct �, �, � and an ordering $e , f , g$ with $\{ u , \nu \} \subset e , \{ \nu , w \} \subset f , \{ w , u \} \subset g$ . Define

$$
\varepsilon _ { n } ^ { \mathrm { a l l } } = \sum _ { \{ e , f \} \in \Omega _ { 2 } } \operatorname* { P r } ( e , f \in { \mathcal { E } } _ { n } ) + \sum _ { \{ e , f , g \} \in \Omega _ { 3 } } \operatorname* { P r } ( e , f , g \in { \mathcal { E } } _ { n } ) .\tag{89}
$$

These are actual joint inclusion probabilities; edges need not be independent, uniform in size, or bounded in degree. The condition concerns the local incidence law, since partitioning can create singletons or duplicates.

Proposition 7 (Nonuniform and dependent reconstruction family). A deterministic decoder recovers $[ B ] _ { \mathrm { c o l } }$ from $G ( B )$ with probability at least $1 - \varepsilon _ { n } ^ { \mathrm { a l l } }$ . For every projected edge ��, it forms $\{ u , \nu \}$ ∪ $( N _ { G } ( u ) \cap N _ { G } ( \nu ) )$ ) and removes repeated sets. An adjacency-matrix implementation costs $O ( n ^ { 3 } )$ operations.

Proof. The union bound excludes both configurations with probability at least $1 - \varepsilon _ { n } ^ { \mathrm { a l l } }$ . Then each projected pair �� belongs to a unique edge �. Its other members are common neighbors. An external common neighbor � would give edges containing �� and ��: if equal they overlap � twice, and if distinct they form a Berge triangle with �. Thus each candidate equals $e .$ . Every edge contains a pair; scanning at most $n ^ { 2 }$ pairs and � neighbors proves recovery and the cost. □

Incidence forests have $\varepsilon _ { n } ^ { \mathrm { a l l } } = 0$ , permitting dependent, nonuniform edges and order-� edge counts. More generally, $\varepsilon _ { n } ^ { \mathrm { a l l } } \to 0$ suffices. The following transfer also accepts other decoders with proved failure bounds, including reconstruction results for independent uniform hypergraphs (Bresler et al., 2025).

Theorem 7 (Heat recovery and asymptotic preservation of local information). Let $O _ { n } = \left( X _ { n } , [ B _ { n } ] _ { \mathrm { c o l } } \right)$ have any joint law with $y _ { n } \in [ C ]$ , where $\bar { C } \geq 2 i$ s fixed. Suppose a decoderfrom $G ( B _ { n } )$ recovers $[ \boldsymbol { B _ { n } } ] _ { \mathrm { c o l } }$ withfailure probability at most $\varepsilon _ { n } ^ { \mathrm { r e c } }$ . Let $T _ { n } = \Psi _ { n } ( O _ { n } , \dot { \xi } _ { n } ^ { \mathrm { p r } } )$ , where $\Psi _ { n }$ isfixed and measurable and $\xi _ { n } ^ { \mathrm { p r } }$ is independent of $( O _ { n } , y _ { n } )$ . Suppose $T _ { n }$ retains the features in Lemma $^ 2$ and enough information to recover $X _ { n } ,$ , with joint rank/attribute-recovery failure probability at most $\varepsilon _ { n } ^ { \mathrm { o b s } } . \ A$ decoderfrom $( T _ { n } , \xi _ { n } ^ { \mathrm { p r } } )$ then recovers $O _ { n }$ with error at most $\varepsilon _ { n } = \varepsilon _ { n } ^ { \mathrm { r e c } } + \varepsilon _ { n } ^ { \mathrm { o b s } }$ . For $\varepsilon _ { n } \leq 1 / 2$

$$
0 \leq \mathcal { R } _ { \mathrm { h e a t , n } } ^ { * } - \mathcal { R } _ { \mathrm { l o c , n } } ^ { * } \leq h _ { 2 } ( \varepsilon _ { n } ) + \varepsilon _ { n } \log C = : d _ { n } ,\tag{90}
$$

where $\mathcal { R } _ { \mathrm { h e a t . n } } ^ { * } = H ( y _ { n } \mid T _ { n } , \xi _ { n } ^ { \mathrm { p r } } ) , \mathcal { R } _ { \mathrm { l o c . n } } ^ { * } = H ( y _ { n } \mid O _ { n } )$ , and $h _ { 2 } ( u ) = - u \log u - ( 1 - u ) \log ( 1 - u )$ with $0 \log 0 = 0$ . Hence $d _ { n } = O ( \varepsilon _ { n } \log ( 1 / \varepsilon _ { n } ) ) \to 0$ when $\varepsilon _ { n } \to 0 .$

Proof. On simultaneous success, Equation (88) recovers $P ,$ its positive off-diagonal support gives $G ( B _ { n } )$ , and the incidence and attribute decoders give $O _ { n }$ . The union bound proves the error claim. Data processing gives the nonnegative entropy gap. If � indicates decoder failure, $O _ { n }$ is determined by $( \dot { T _ { n } } , \dot { \xi _ { n } ^ { \mathrm { p r } } } , E = 0 )$ , so

$$
I ( y _ { n } ; O _ { n } \mid T _ { n } , \xi _ { n } ^ { \mathrm { p r } } ) \leq H ( E ) + \operatorname* { P r } ( E = 1 ) \log C \leq h _ { 2 } ( \varepsilon _ { n } ) + \varepsilon _ { n } \log C .
$$

Independence of the seed and deterministic generation of $T _ { n }$ identify this mutual information with the entropy gap; Bayes cross-entropy equals conditional entropy. □

Why incidence recovery needs an assumption. On six vertices, $\mathcal { E } _ { 1 } = \{ 1 2 3 , 1 4 5 , 2 4 6 , 3 5 6 \}$ and $\mathcal { E } _ { 2 } = \{ 1 2 4 , 1 3 5 , 2 3 6 , 4 5 6 \}$ have degree two and identical pair co-incidences. Thus $B _ { 1 } B _ { 1 } ^ { \top } = B _ { 2 } B _ { 2 } ^ { \top }$ and $P _ { 1 } = P _ { 2 } = B _ { 1 } B _ { 1 } ^ { \top } / 6 .$ , despite different column-permutation classes. Identical attributes give identical heat features at every time. A uniform mixture, padded with common isolated nodes, has reconstruction error at least $\mathrm { i } / 2$ . Duplicating all columns also preserves � for multihypergraphs. Additional heat channels cannot remove either ambiguity.

Recovery at a queried node. For a simple hypergraph $\varepsilon$ with edge sizes at least two, let $V _ { \mathrm { a m b } } ( \mathcal { E } )$ contain endpoints of pairs lying in two distinct edges and the three cycle vertices of each Berge triangle. Put $p _ { n , \nu } ^ { \mathrm { q u e r y } } = \mathbf { \dot { P } } \mathbf { r } ( \nu _ { * } \stackrel { \bullet } { = } \nu \mid \mathcal { E } _ { n } )$ for the queried vertex $\nu _ { * }$

Proposition 8 (Query-weighted recovery and prediction risk). For $\nu \not \in V _ { \mathrm { a m b } } ( \mathcal { E } )$ , its incident edge set satisfies

$$
\widehat { \mathcal { E } } ( \nu ) = \mathrm { u n i q u e } \{ \{ \nu , u \} \cup ( N _ { G } ( \nu ) \cap N _ { G } ( u ) ) : u \in N _ { G } ( \nu ) \} = \mathcal { E } ( \nu ) ,\tag{91}
$$

where $\mathcal { E } ( \nu ) = \{ e \in \mathcal { E } : \nu \in e \}$ and unique removes repeats. Under any joint edge law, failure at $\nu _ { * }$ has probability at most

$$
\boldsymbol { \varepsilon } _ { n } ^ { \mathrm { q u e r y } } = \mathbb { E } \sum _ { \nu = 1 } ^ { n } p _ { n , \nu } ^ { \mathrm { q u e r y } } \mathbf { 1 } \{ \nu \in V _ { \mathrm { a m b } } ( \mathcal { E } _ { n } ) \} ,\tag{92}
$$

equal to $\mathbb { E } | V _ { \mathrm { a m b } } ( \mathcal { E } _ { n } ) | / n$ for an independent uniform query. More generally, let $O _ { n }$ include all permitted local information and the query, and let $S _ { n } ^ { \mathrm { t a s k } }$ be measurablefrom $O _ { n \cdot } ^ { . } I f T _ { n } = \Psi _ { n } ( O _ { n } , \xi _ { n } ^ { \mathrm { p r } } )$ for fixed measurable $\Psi _ { n }$ and a seed independent of $( O _ { n } , y _ { n } ) , y _ { n } \in [ C ]$ , and $( T _ { n } , \xi _ { n } ^ { \mathrm { p r } } )$ recovers $S _ { n } ^ { \mathrm { t a s k } }$ with error at most $\delta _ { n } \leq 1 / 2 ,$ , then

$$
0 \leq H ( y _ { n } \mid T _ { n } , \xi _ { n } ^ { \mathrm { p r } } ) - H ( y _ { n } \mid O _ { n } ) \leq I ( y _ { n } ; O _ { n } \mid S _ { n } ^ { \mathrm { t a s k } } ) + h _ { 2 } ( \delta _ { n } ) + \delta _ { n } \log C .\tag{93}
$$

Proof. For $\nu \notin V _ { \mathrm { a m b } }$ each pair �� has a unique source edge �. An external common neighbor would create either a repeated pair through � or a Berge triangle with cycle vertex �, exactly as in Proposition 7. Thus the decoder returns precisely all incident edges, including the empty set at an isolated vertex. Conditioning on $\textstyle { \mathcal { E } } _ { n }$ proves the failure bound. The failure-indicator argument above gives $I ( y _ { n } ; S _ { n } ^ { \mathrm { t a s k } } \mid T _ { n } , \xi _ { n } ^ { \mathrm { p r } } ) \overset {  } { \leq } h _ { 2 } ( \delta _ { n } ) \overset {  } { + } \delta _ { n }$ log �. Use $H ( y _ { n } \mid T _ { n } , \xi _ { n } ^ { \mathrm { p r } } ) \leq H ( y _ { n } \mid S _ { n } ^ { \mathrm { t a s k } } ) { \overset { \sim } { + } } I ( y _ { n } ; S _ { n } ^ { \mathrm { t a s k } } \mid$ $\breve { T } _ { n } , \xi _ { n } ^ { \mathrm { p r } } )$ , subtract $H ( y _ { n } \mid O _ { n } )$ , and apply measurability and data processing. □

If $S _ { n } ^ { \mathrm { t a s k } }$ contains the query, its incident edges and their member attributes (including its own when isolated), the rank and attribute assumptions give $\delta _ { n } \leq \varepsilon _ { n } ^ { \mathrm { q u e r y } } + \varepsilon _ { n } ^ { \mathrm { o b s } }$ . Only the probability of querying an affected vertex must vanish; $I ( y _ { n } ; O _ { n } \mid S _ { n } ^ { \mathrm { t a s k } } )$ separately charges label information beyond this neighborhood. No locality of the dense heat operator is assumed. For known two-node simple graphs, projection recovers every edge, including those in cycles.

For a concrete separation, put $m _ { n }  \infty , m _ { n } / n  0$ disjoint six-node examples among � vertices, choose their incidences independently and uniformly, and put a fixed incidence forest elsewhere. With fixed noninformative attributes, an encoder independent of these choices, and a retained query index, all $2 ^ { m _ { n } }$ incidences have identical heat features, even with independent probes. Full reconstruction succeeds with probability at most $2 ^ { - m _ { n } }$ , yet $\varepsilon _ { n } ^ { \mathrm { q u e r y } } \leq 6 m _ { n } / n$ . Label each block vertex by its block choice and forest vertices by zero. Full incidence determines the labels; for a uniform query the optimal full-matrix heat cross-entropy is $\left( 6 m _ { n } / n \right)$ log $2  0$

These are information bounds for the complete local heat matrix. Its decoder need not be expressible by the implemented finite encoder and rowwise classifier. The next bound retains that approximation error, shared-parameter cost, and training error; neither private probes nor a reconstruction stage is added to the implementation or its payload.

## C.12 STOCHASTIC FEDERATED OPTIMIZATION AND RISK TRANSFER

Population risk and shared parameters. Let a random test node belong to client � with probability $w _ { k } ^ { \mathsf { p o p } }$ , and write $\begin{array} { r } { \mathcal { R } _ { k } ( \theta ) = \mathbb { E } [ \tilde { - } \log q _ { \theta , k } ( y ) ] , \mathcal { R } _ { \mathrm { f e d } } ( \theta ) = \sum _ { k } w _ { k } ^ { \mathrm { p o p } } \mathcal { R } _ { k } ( \theta ) } \end{array}$ for the actual prediction rule. Evaluation weights may differ from FedAvg weights $w _ { k } ^ { \mathrm { t r } } = n _ { k } ^ { \mathrm { i r } } / \sum _ { j } n _ { j } ^ { \mathrm { t r } }$ . All posterior risks include the same permitted training-side information and exclude held-out labels. If conditioning on a training transcript, include the realized probe seed in both local and heat information: the representation is then deterministic given these data, and recovery bounds must hold under the conditional law. Unconditional probe independence need not survive this conditioning. Expectations also average over training and algorithmic randomness.

Theorem 8 (From local information to shared-model population risk). Let Θ be a nonempty shared parameter domain withfinite risks and ${ \widehat { \theta } } \in \Theta .$ . For each client suppose the optimal local and heat risks satisfy $0 \leq \mathcal { R } _ { \mathrm { h e a t } , k } ^ { * } - \mathcal { R } _ { \mathrm { l o c } , k } ^ { * } \leq d _ { k }$ , using Theorem 7 or Proposition 8. Define

$$
\begin{array} { r l r } & { } & { a _ { k } = \left[ \displaystyle \operatorname* { i n f } _ { \theta \in \Theta } \mathcal { R } _ { k } ( \theta ) - \mathcal { R } _ { \mathrm { h e a t } , k } ^ { * } \right] _ { + } , } \\ & { } & { a _ { \mathrm { s h a r e } } = \displaystyle \operatorname* { i n f } _ { \theta \in \Theta } \sum _ { k } w _ { k } ^ { \mathrm { p o p } } \mathcal { R } _ { k } ( \theta ) - \sum _ { k } w _ { k } ^ { \mathrm { p o p } } \operatorname* { i n f } _ { \theta \in \Theta } \mathcal { R } _ { k } ( \theta ) \ge 0 . } \end{array}\tag{94}
$$

Here $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \} ; a _ { k }$ includes finite-encoder and rowwise-readout restrictions, and �<sub>share</sub> charges shared parameters. For fixed empirical training cross-entropy $J _ { N }$ on � labels, suppose

$$
\mathbb { E } \operatorname* { s u p } _ { \theta \in \Theta } | J _ { N } ( \theta ) - \mathcal { R } _ { \mathrm { f e d } } ( \theta ) | \leq u _ { N } , \qquad \mathbb { E } [ J _ { N } ( \widehat { \theta } ) - \operatorname* { i n f } _ { \theta \in \Theta } J _ { N } ( \theta ) ] \leq e _ { N , T } .\tag{95}
$$

Then, without independence across clients,

$$
\mathbb { E } \Bigg [ \mathcal { R } _ { \mathrm { f e d } } ( \widehat { \theta } ) - \sum _ { k } w _ { k } ^ { \mathrm { p o p } } \mathcal { R } _ { \mathrm { l o c } , k } ^ { \ast } \Bigg ] \leq \mathbb { E } \Bigg [ \sum _ { k } w _ { k } ^ { \mathrm { p o p } } ( d _ { k } + a _ { k } ) + a _ { \mathrm { s h a r e } } \Bigg ] + 2 u _ { N } + e _ { N , T } .\tag{96}
$$

Expectations include any conditioning information on which the risks depend.

Proof. Two uniform-discrepancy bounds give $\mathbb { E } \mathcal { R } _ { \mathrm { f e d } } ( \widehat { \theta } ) \leq \mathbb { E } \operatorname* { i n f } _ { \Theta } \mathcal { R } _ { \mathrm { f e d } } + 2 u _ { N } + e _ { N , T }$ . Substitute the definition of $a _ { \mathrm { s h a r e } }$ and inf<sub>Θ</sub> $\mathcal { R } _ { k } - \mathcal { R } _ { \mathrm { l o c } , k } ^ { * } \leq a _ { k } + d _ { k }$ . For transcript-conditioned risks apply these inequalities conditionally and average. □

The random-test-node objective needs weighted recovery errors, not simultaneous recovery at every client. A single parameter sequence approaching all positively weighted client optima suffices for $a _ { \mathrm { s h a r e } } = 0 ;$ separate local approximation does not. If cross-entropies lie in $[ 0 , M ] , u _ { N }$ is at most the training-weighted uniform sampling deviation plus $M \| w ^ { \mathrm { t r } } - w ^ { \mathrm { \hat { p } o p } } \| _ { 1 } / 2$ . Dependent hypergraphs supply no automatic independent-node concentration rate.

Actual displacements and task progress. Let $d _ { k , u , s } = \theta _ { k , u , s } – \theta _ { k , u , s + 1 }$ be the realized displacement at local step � of round �, with $S _ { k , u }$ steps. Full-participation FedAvg gives

$$
D _ { u } = \theta ^ { u } - \theta ^ { u + 1 } = \sum _ { k } w _ { k } ^ { \mathrm { t r } } \sum _ { s = 0 } ^ { S _ { k , u } - 1 } d _ { k , u , s } .\tag{97}
$$

Masks, minibatches, dropout, clipping, optimizer moments, weight decay, numerical heat evaluation, and target motion enter through these displacements. Different step counts are allowed; subsampling uses normalized participating-client weights.

Proposition 9 (Task-risk bound along the implemented trajectory). Let $J _ { N }$ be evaluation-mode empirical cross-entropy with exact heat, and let $L _ { u } \ge 0$ bound its largest Hessian eigenvalue on $\{ \theta ^ { \hat { u } } - a D _ { u } : 0 \leq a \leq \mathrm { i } \}$ . For every prefix ofeveryfinite realized trajectory,

$$
\begin{array} { r } { J _ { N } ( \theta ^ { t } ) \le J _ { N } ( \theta ^ { 0 } ) - \displaystyle \sum _ { u < t } \left\{ \langle \nabla J _ { N } ( \theta ^ { u } ) , D _ { u } \rangle - \frac { 1 } { 2 } L _ { u } \| D _ { u } \| _ { 2 } ^ { 2 } \right\} . } \end{array}\tag{98}
$$

A positive bracket guarantees strict reduction that round. The inequality holds simultaneously for all prefixes, including any validation-selected checkpoint; its right side minus inf<sub>Θ</sub> $J _ { N } ,$ , followed by expectation, bounds $e _ { N , T }$

Proof. Taylor’s integral remainder is $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( 1 - a ) D _ { u } ^ { \top } \nabla ^ { 2 } J _ { N } ( \theta ^ { u } - a D _ { u } ) D _ { u } d a \ \leq \ L _ { u } \| D _ { u } \| _ { 2 } ^ { 2 } / 2 ; } \end{array}$ sum over rounds. This is pathwise and requires neither unbiased gradients, a fixed teacher, nor gradient domination. The native evaluation network’s linear maps, GELU, softplus, matrix exponential, positive-epsilon LayerNorm, and cross-entropy are smooth, giving finite curvature on each finite segment. The scale model also uses LayerNorm and a smooth closed-form heat operator. Predictive anti-collapse losses affect $D _ { u }$ but are absent from $J _ { N }$ ; baseline BatchNorm buffers are outside this proof. □

If implemented logits differ from exact logits by at most $\epsilon _ { u } ^ { \mathrm { l o g i t } }$ in infinity norm at checkpoint �, the 2-Lipschitz property of cross-entropy adds only $2 \epsilon _ { 0 } ^ { \mathrm { l o g i t } } + \stackrel { \cdot \cdot } { 2 } \epsilon _ { t } ^ { \mathrm { l o g i t } }$ to the implemented-loss bound. Gradients and curvatures refer to exact heat; intermediate numerical errors enter $D _ { u }$

Classifier, Adam, and target identities. Split parameters into context $\vartheta ,$ classifier $\theta _ { \mathrm { c l s } }$ , and masked predictor $\theta _ { \mathrm { p r e d } }$ . The predictive loss does not use the classifier, so

$$
\nabla _ { \theta _ { \mathrm { c l s } } } \mathcal { L } _ { \mathrm { p r e d , k } } = 0 , \qquad \nabla _ { \theta _ { \mathrm { c l s } } } \mathcal { L } _ { k } = \nabla _ { \theta _ { \mathrm { c l s } } } \mathcal { L } _ { \mathrm { C E , k } } .\tag{99}
$$

Joint clipping can rescale this gradient, and context updates change its inputs. Suppress client/round indices, let $g _ { s }$ be the realized stochastic gradient, $c _ { s } ^ { \mathrm { c l i p } } \in [ 0 , 1 ]$ its clipping multiplier, $\ell _ { s } > 0$ the learning rate, and $\lambda _ { \mathrm { w d } } \geq 0$ weight decay. Coupled Adam uses $z _ { s } = c _ { s } ^ { \mathrm { c l i p } } g _ { s } + \lambda _ { \mathrm { w d } } \theta _ { s }$ . From zero moments,

$$
\widehat { m } _ { s } = z _ { s } + \sum _ { j < s } \frac { ( 1 - \beta _ { 1 } ) \beta _ { 1 } ^ { s - j } } { 1 - \beta _ { 1 } ^ { s + 1 } } ( z _ { j } - z _ { s } ) , \qquad d _ { s } = \ell _ { s } \mathrm { d i a g } ( \sqrt { \widehat { \nu _ { s } } } + \epsilon _ { \mathrm { A d a m } } ) ^ { - 1 } \widehat { m } _ { s } ,\tag{100}
$$

where $\begin{array} { r } { \widehat { \nu } _ { s } = ( 1 - \beta _ { 2 } ) \sum _ { j \leq s } \beta _ { 2 } ^ { s - j } z _ { i } ^ { \odot 2 } / ( 1 - \beta _ { 2 } ^ { s + 1 } ) , \beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 ) , \epsilon _ { \mathrm { A d a m } } > 0 } \end{array}$ , and ⊙2 is coordinatewise squaring. AdamW instead sets $z _ { \bot } = c _ { s } ^ { \mathrm { c l i p } } g _ { s }$ and adds $\ell _ { s } \lambda _ { \mathrm { w d } } \theta _ { s }$ to $d _ { s } .$ . The native optimizer resets each round. The target also resets to $\bar { \vartheta } _ { 0 } = \vartheta _ { 0 }$ , and post-step EMA gives

$$
\bar { \vartheta } _ { s } = \vartheta _ { 0 } - \sum _ { j < s } ( 1 - \mu ^ { s - j } ) d _ { j } ^ { \mathrm { c t x } } , \qquad \vartheta _ { s } - \bar { \vartheta } _ { s } = - \sum _ { j < s } \mu ^ { s - j } d _ { j } ^ { \mathrm { c t x } } .\tag{101}
$$

These exact identities enter Equation (98), even when moving-target updates are not gradients of one fixed loss. EMA controls lag to the current context, not distance to a fixed reference teacher. Neither these identities nor the architecture guarantees task alignment; general Adam can fail even on convex problems (Reddi et al., 2018).

Conditional convergence through the affine output layer. Write $\theta = ( \theta _ { \mathrm { r e s t } } , \theta _ { \mathrm { o u t } } )$ , where $\theta _ { \mathrm { o u t } }$ consists of the final affine classification weights and bias. With $\theta _ { \mathrm { r e s t } }$ fixed, evaluation cross-entropy is convex in this block, which also satisfies Equation (99). This applies to both FedASTRA implementations and the optional deeper classifier, whose preceding layers belong to $\theta _ { \mathrm { r e s t } }$

Theorem 9 (Global task-risk convergence along actual updates). Use $J _ { N } , D _ { u } , L _ { u }$ from Proposition 9, let $J _ { N } ^ { * } = \operatorname* { i n f } _ { \theta \in \Theta } J _ { N } ( \theta ) > - \infty$ and $E _ { u } = J _ { N } \big ( \theta ^ { u } \big ) - J _ { N } ^ { * }$ . Fix $0 \leq u _ { 0 } < T$ . For every $u = u _ { 0 } , \ldots , T$ assume a comparator with actual and comparison parameters in Θ satisfying

$$
\begin{array} { r } { J _ { N } ( \theta _ { \mathrm { r e s t } } ^ { u } , \theta _ { \mathrm { o u t } } ^ { u , \circ } ) \leq J _ { N } ^ { * } + a _ { \mathrm { c t x } } , \qquad \| \theta _ { \mathrm { o u t } } ^ { u } - \theta _ { \mathrm { o u t } } ^ { u , \circ } \| _ { 2 } \leq D _ { \mathrm { c l s } } , } \end{array}\tag{102}
$$

with $a _ { \mathrm { c t x } } \geq 0$ and $0 < D _ { \mathrm { c l s } } < \infty$ . This assumption concerns affine-head adequacy on each realized representation, separately from full-model approximation over Θ. Define $\dot { \Delta } _ { u } = \dot { \langle \nabla J _ { N } ( \theta ^ { u } ) , D _ { u } \rangle } -$ $\bar { L _ { u } } \| D _ { u } \| _ { 2 } ^ { 2 } / 2 .$ . Suppose $\gamma _ { u } > 0 , \zeta _ { u } \ge 0$ satisfy

$$
\Delta _ { u } \ge \gamma _ { u } \| \nabla _ { \theta _ { \mathrm { o u t } } } J _ { N } ( \theta ^ { u } ) \| _ { 2 } ^ { 2 } - \zeta _ { u } , \qquad u = u _ { 0 } , \ldots , T - 1 .\tag{103}
$$

Then, pathwise,

$$
E _ { u + 1 } \leq E _ { u } - \frac { \gamma _ { u } } { D _ { \mathrm { c l s } } ^ { 2 } } ( E _ { u } - a _ { \mathrm { c t x } } ) _ { + } ^ { 2 } + \zeta _ { u } .\tag{104}
$$

If all deficits $\zeta _ { u }$ vanish, then

$$
\begin{array} { r } { E _ { T } \le a _ { \mathrm { c t x } } + r _ { { u _ { 0 } } , T } , \qquad r _ { { u _ { 0 } } , T } : = \left[ ( E _ { u _ { 0 } } - a _ { \mathrm { c t x } } ) _ { + } ^ { - 1 } + D _ { \mathrm { c l s } } ^ { - 2 } \displaystyle \sum _ { u = u _ { 0 } } ^ { T - 1 } \gamma _ { u } \right] ^ { - 1 } , } \end{array}\tag{105}
$$

with $r _ { u _ { 0 } , T } = 0 \ i f ( E _ { u _ { 0 } } - a _ { \mathrm { c t x } } ) _ { + } = 0 \ ,$ . In particular, $\gamma _ { u } \geq \gamma > 0$ gives $r _ { u _ { 0 } , T } \leq D _ { \mathrm { c l s } } ^ { 2 } / [ \gamma ( T - u _ { 0 } ) ]$ . For general deficits,

$$
\operatorname* { m i n } _ { u _ { 0 } \le u < T } ( E _ { u } - a _ { \mathrm { c t x } } ) _ { + } \le D _ { \mathrm { c l s } } \sqrt { \frac { E _ { u _ { 0 } } + \sum _ { u = u _ { 0 } } ^ { T - 1 } \zeta _ { u } } { \sum _ { u = u _ { 0 } } ^ { T - 1 } \gamma _ { u } } } .\tag{106}
$$

Proof. Affine-head convexity and Equation (102) give

$$
\begin{array} { r } { E _ { u } - a _ { \mathrm { c t x } } \le \langle \nabla _ { \theta _ { \mathrm { o u t } } } J _ { N } ( \theta ^ { u } ) , \theta _ { \mathrm { o u t } } ^ { u } - \theta _ { \mathrm { o u t } } ^ { u , \circ } \rangle \le D _ { \mathrm { c l s } } \| \nabla _ { \theta _ { \mathrm { o u t } } } J _ { N } ( \theta ^ { u } ) \| _ { 2 } . } \end{array}
$$

Combining with $E _ { u + 1 } \leq E _ { u } - \Delta _ { u }$ proves the recurrence. For zero deficits, $E _ { u }$ is nonincreasing. Put $\delta _ { u } = ( E _ { u } - a _ { \mathrm { { c t x } } } ) .$ <sub>+</sub> and $\kappa _ { u } = \gamma _ { u } / \bar { D } _ { \mathrm { c l s } } ^ { 2 }$ . Whenever $\delta _ { u + 1 } > 0 , \delta _ { u + 1 } \leq \delta _ { u } ( 1 - \kappa _ { u } \delta _ { u } )$ with $\kappa _ { u } \delta _ { u } \ < \ 1$ hence $\delta _ { u + 1 } ^ { - 1 } \geq \delta _ { u } ^ { - 1 } + \kappa _ { u }$ . A zero $\delta _ { u }$ stays zero. Summing proves the reciprocal rate. For arbitrary deficits, summing the recurrence and using $E _ { T } \geq 0$ gives $\begin{array} { r } { D _ { \mathrm { c l s } } ^ { - 2 } \sum _ { u } \gamma _ { u } \delta _ { u } ^ { 2 } \leq E _ { u _ { 0 } } + \sum _ { u } \zeta _ { u } ; } \end{array}$ bounding the minimum by the weighted mean proves the last claim. □

The coefficients $\gamma _ { u }$ measure realized task progress and are not Adam learning rates. For any chosen $\gamma _ { u } ~ > ~ 0$ , the deficit $\zeta _ { u } ~ = ~ [ \gamma _ { u } \vert \vert \nabla _ { \theta _ { \mathrm { o u t } } } J _ { N } ( \dot { \theta } ^ { u } ) \vert \vert _ { 2 } ^ { 2 } - \Delta _ { u } ]$ <sub>+</sub> makes the deficit bound applicable. The $O ( ( T - u _ { 0 } ) ^ { - 1 } )$ conclusion additionally requires adequate realized contexts, zero deficits, and progress coefficients bounded away from zero. The architecture supplies affine-head convexity; it does not establish these trajectory conditions. No unbiased-gradient or full-network Polyak–Lojasiewicz condition is used.

Corollary 9 (Global population convergence of the returned model). Assume Theorems 8 and $^ { 9 , }$ with $\zeta _ { u } ~ = ~ 0$ on the stated interval. For the actual validation-selected $\widehat { \theta } _ { N , T }$ define $s _ { N , T } ^ { \mathrm { s e l } } ~ =$ $[ J _ { N } ( \widehat { \theta } _ { N , T } ) - J _ { N } ( \theta ^ { T } ) ] _ { + }$ , and put $\begin{array} { r } { A _ { N } = \mathbb { E } [ \sum _ { k } w _ { k } ^ { \mathrm { p o p } } ( d _ { k } + a _ { k } ) + a _ { \mathrm { s h a r e } } ] + 2 u _ { N } } \end{array}$ . Then

$$
\mathbb { E } \left[ \mathcal { R } _ { \mathrm { f e d } } ( \widehat { \theta } _ { N , T } ) - \sum _ { k } w _ { k } ^ { \mathrm { p o p } } \mathcal { R } _ { \mathrm { l o c } , k } ^ { * } \right] \leq A _ { N } + \mathbb { E } \big [ a _ { \mathrm { c t x } } + r _ { u _ { 0 } , T } + s _ { N , T } ^ { \mathrm { s e l } } \big ] .\tag{107}
$$

Expectations include all path-dependent quantities. $I f \gamma _ { u } \geq \gamma > 0$ and $\mathbb { E } D _ { \mathrm { c l s } } ^ { 2 }$ is uniformly bounded, the reciprocal term is $O ( ( T - u _ { 0 } ) ^ { - 1 } )$ in expectation. Vanishing $A _ { N } , \mathbb { E } a _ { \mathrm { c t x } } , \mathbb { E } r _ { u _ { 0 } , T }$ , and $\mathbb { E } s _ { N , T } ^ { \mathrm { s e l } }$ along a joint data-size and training-horizon limit give local-information Bayes consistency.

Proof. By definition, $J _ { N } ( \widehat { \theta } _ { N , T } ) - J _ { N } ^ { * } \leq E _ { T } + s _ { N , T } ^ { \mathrm { s e l } }$ . Insert Equation (105) into Theorem $\mathrm { 8 ; }$ the stated rate and limit follow. □

The selection term covers the implementation’s validation Accuracy/Macro-F1 rule. Nonzero alignment deficits retain only Equation (106)’s best-iterate guarantee, unless selection is controlled separately. These conditional rates do not follow from finite experiments or from the choice of Adam, EMA, and FedAvg alone.

Predictive-surrogate mismatch. The same decomposition for one predictor class makes objective mismatch explicit.

Corollary 10 (Conditional local Bayes consistency). Let $d _ { n }$ befrom Theorem $^ 7 ,$ and let $Q _ { N }$ be an available predictor class measurable from retained heat information, with population cross-entropy $R ( q )$ and approximation error $\begin{array} { r } { a _ { N } = \operatorname* { i n f } _ { \boldsymbol { q } \in \mathcal { Q } _ { N } } R ( \boldsymbol { q } ) - \mathcal { R } _ { \mathrm { h e a t , n } } ^ { * } \geq 0 . } \end{array}$ . Suppose $\mathbb { E } \operatorname* { s u p } _ { q \in Q _ { N } } | \widehat { R } _ { N } ( q ) -$ $R ( q ) \vert \leq u _ { N }$ and the returned $\widehat { q } _ { N , T } \in \mathcal { Q } _ { N }$ satisfies

$$
\mathbb { E } [ \widehat { R } _ { N } ( \widehat { q } _ { N , T } ) - \operatorname* { i n f } _ { q \in Q _ { N } } \widehat { R } _ { N } ( q ) ] \leq \kappa _ { N } + e _ { N , T } ,\tag{108}
$$

where $\kappa _ { N } , e _ { N , T } \geq 0$ bound surrogate/readout mismatch and optimization. Then

$$
\begin{array} { r } { \mathbb { E } R ( \widehat { q } _ { N , T } ) - \mathcal { R } _ { \mathrm { l o c } , \mathrm { n } } ^ { * } \leq d _ { n } + a _ { N } + 2 u _ { N } + \kappa _ { N } + e _ { N , T } . } \end{array}\tag{109}
$$

Thus $\varepsilon _ { n } , a _ { N } , u _ { N } , \kappa _ { N } , e _ { N , T }  0$ imply local Bayes consistency.

Proof. Two uniform-deviation bounds yield $\begin{array} { r } { \Xi R ( \widehat { q } _ { N , T } ) \le \operatorname* { i n f } _ { Q _ { N } } R + 2 u _ { N } + \kappa _ { N } + e _ { N , T } ; } \end{array}$ substitute �<sub>�</sub> and Equation (90). For a fixed-teacher joint objective $J _ { N } = \widehat { R } _ { N } + \lambda _ { \mathrm { p r e d } } \widehat { L } _ { \mathrm { p r e d , N } }$ , let $q$ index the complete model, including auxiliary heads. If a comparator $q _ { N } ^ { \circ }$ has CE gap at most $\kappa _ { N } ^ { \mathrm { C E } }$ and predictive-loss gap at most $\kappa _ { N } ^ { \mathrm { p r e d } }$ , both relative to their infima over $Q _ { N }$ , an $e _ { N , T }$ -optimal solution of $J _ { N }$ satisfies Equation (108) with $\kappa _ { N } = \kappa _ { N } ^ { \mathrm { C E } } + \lambda _ { \mathrm { p r e d } } \kappa _ { N } ^ { \mathrm { p r e d } }$ . This follows by subtracting the predictive-loss lower bound. It requires a fixed surrogate; moving-teacher updates do not automatically optimize that objective. □

Information recovery controls $d _ { n } ;$ the finite representation/readout controls $a _ { N } ;$ the data law and training procedure must control $u _ { N } , \kappa _ { N } , e _ { N , T }$ . Under label sufficiency in Section 4, the local Bayes limit equals the enriched-information limit, and Equation (17) gives the criterion for beating a particular fitted enriched predictor.

## D EXPERIMENTAL PROTOCOLS AND IMPLEMENTATION DETAILS

## D.1 FEDASTRA IMPLEMENTATION

Network and heat parameterization. The context encoder applies a linear map, layer normalization, and GELU to the raw attributes, followed by encoder dropout. Let its width be �. The training forward pass uses the type-marginalized binary incidence matrix from Equation (18). Relation types are not used as model inputs, and repeated relation instances remain separate columns. The standard experiments use $R = 4 ,$ , while Trivago uses $R = 8$ . The time increments in Equation (20) are initialized at zero, giving initial diffusion times close to � log 2, with $\epsilon _ { \mathrm { o r d } } = 1 0 ^ { - 4 }$ . The classifier receives $( R + 1 ) d$ coordinates, applies layer normalization and dropout, and uses one linear readout. The predictive head maps $( R + 1 ) d$ to $2 { \dot { d } } ,$ applies layer normalization and GELU, and maps back to $( R + 1 ) d$ . For each local update, two independent Bernoulli masks remove attribute coordinates. The removal probability is 0.40 on the established tasks and 0.30 on Trivago, with retained values divided by 0.60 and $0 . { \dot { 7 } } 0 .$ , respectively. Incidence indices, relation columns, and node membership remain fixed across both online views and the target view. At the start of each round, the EMA target context is initialized from the broadcast context encoder, including the attribute map and ordered-time increments. The target is updated after each local step with momentum $\mu = 0 . 9 9$ on the standard experiments and $\mu = 0 . 9 9 6$ on Trivago. It remains client-local and is discarded at the end of each round.

Mask variance and target memory. Write the coordinate-removal probability as $p < 1$ and let $\delta _ { j } \stackrel { \mathrm { i i d } } { \sim }$ B $\mathrm { \ m o u l l i } ( 1 - p )$ . The inverted mask $\mathcal { M } _ { p } ( x ) = \delta \odot x / ( 1 - p )$ obeys exactly

$$
\mathbb { E } _ { \delta } { M } _ { p } ( x ) = x , \qquad \mathbb { E } _ { \delta } \| { M } _ { p } ( x ) - x \| _ { 2 } ^ { 2 } = \frac { p } { 1 - p } \| x \| _ { 2 } ^ { 2 } .\tag{110}
$$

Thus, the masking variance is determined directly by $p .$ . Within one client round, write $\vartheta _ { s } = \operatorname { c t x } ( \theta _ { s } )$ and $\bar { \vartheta } _ { s }$ for the online and target context states after local step �, with the round reset $\bar { \vartheta } _ { 0 } = \vartheta _ { 0 }$ . The update $\bar { \vartheta } _ { s } = \mu \bar { \vartheta } _ { s - 1 } + ( 1 - \mu ) \vartheta _ { s }$ expands over the complete local context history as

$$
\bar { \vartheta } _ { s } = \mu ^ { s } \vartheta _ { 0 } + \left( 1 - \mu \right) \sum _ { j = 1 } ^ { s } \mu ^ { s - j } \vartheta _ { j } .\tag{111}
$$

The EMA therefore weights earlier context states geometrically by $\mu ^ { s - j }$ . The corresponding memory scale is approximately $( 1 - \mu ) ^ { - 1 }$ local steps. Both masking and EMA updates are client-local. For every fixed $p < 1$ and $\mu \in [ 0 , 1 )$ , coupling their randomness preserves the complete observation invariance in Corollary 4. In the masked-prediction ablation, $p$ and $\mu$ are held fixed across paired conditions so that only the predictive component changes.

Exact local objective. Each local update encodes two masked views online. At client $k ,$ , let $\vartheta _ { k } = \cos ( \theta _ { k } )$ and let $\bar { \vartheta } _ { k } = \left( \bar { \theta } _ { k , \mathrm { e n c } } , \bar { \eta } _ { k } \right)$ denote the round-local EMA copy of the complete context encoder, including its learned time increments. The target branch forms $Z _ { k } ^ { \mathrm { t a r } } = W _ { k } ( { \hat { X } } _ { k } \mid \bar { \vartheta } _ { k } )$ , and each masked online view produces a prediction $\widehat { Z } _ { k } ^ { ( a ) } = p _ { \theta _ { k } } ( W _ { k } ( \widetilde { X } _ { k } ^ { ( a ) } \mid \vartheta _ { k } ) )$ ) with $a \in \{ 1 , 2 \}$ . Here the row-wise predictor is $p _ { \theta _ { k } } : \mathbb { R } ^ { ( R + 1 ) d }  \mathbb { R } ^ { ( R + 1 ) d }$ and the row-wise classifier for � classes is $c _ { \theta _ { k } } : \mathbb { R } ^ { ( R + 1 ) d }  \mathbb { R } ^ { C }$ . Let $n _ { k } = | V _ { k } |$ and we use LN for row-wise layer normalization, sg for stop gradient, and cos for cosine similarity. Set $\begin{array} { r } { \mathcal { A } _ { k } ^ { ( a ) } = n _ { k } ^ { - 1 } \sum _ { \nu \in V _ { k } } [ 1 - \cos ( \mathrm { L N } ( \widehat { Z } _ { k , \nu } ^ { ( a ) } ) , \mathrm { s g } ( \mathrm { L N } ( Z _ { k , \nu } ^ { \mathrm { t a r } } ) ) ) ] } \end{array}$ With an anti-collapse regularizer ${ \mathcal { R } } _ { \mathrm { a c } }$ inspired by variance–invariance–covariance regularization (VICReg), and a fixed view-agreement weight $\gamma _ { \mathrm { v i e w } } \geq 0$ , we define

$$
\mathcal { L } _ { \mathrm { p r e d } , k } = \frac { 1 } { 2 } \sum _ { a = 1 } ^ { 2 } [ \mathcal { A } _ { k } ^ { ( a ) } + \mathcal { R } _ { \mathrm { a c } } ( \mathrm { L N } ( \widehat { Z } _ { k } ^ { ( a ) } ) ) ] + \gamma _ { \mathrm { v i e w } } \left[ 1 - \frac { 1 } { n _ { k } } \sum _ { \nu \in V _ { k } } \cos ( \widehat { Z } _ { k , \nu } ^ { ( 1 ) } , \widehat { Z } _ { k , \nu } ^ { ( 2 ) } ) \right] .\tag{112}
$$

For $A \in \mathbb { R } ^ { n \times h }$ , define $\begin{array} { r } { \overline { { A } } _ { j } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } A _ { i j } } \end{array}$ and $\begin{array} { r } { \operatorname { V a r } _ { n } ( A _ { \cdot j } ) = \sum _ { i = 1 } ^ { n } ( A _ { i j } - \overline { { A } } _ { j } ) ^ { 2 } / \operatorname* { m a x } ( n - 1 , 1 ) } \end{array}$ . Thus the implementation uses sample variance for $n > 1$ and the zero variance for $n = 1$ . Define

$$
\mathcal { V } ( A ) = \frac { 1 } { h } \sum _ { j = 1 } ^ { h } \left[ 1 - \sqrt { \mathrm { V a r } _ { n } ( A . _ { j } ) + 1 0 ^ { - 4 } } \right] _ { + } ,\tag{113}
$$

$$
C ( A ) = \frac { 1 } { h } \left\| \mathrm { o f f d i a g } \left( \frac { \widetilde { A } ^ { \top } \widetilde { A } } { \operatorname* { m a x } ( n - 1 , 1 ) } \right) \right\| _ { F } ^ { 2 } , \qquad \widetilde { A } _ { \cdot j } = \frac { A _ { \cdot j } - \overline { { A } } _ { j } } { \sqrt { \mathrm { V a r } _ { n } ( A _ { \cdot j } ) + 1 0 ^ { - 4 } } } .\tag{114}
$$

The anti-collapse term in Equation (112) is $\mathcal { R } _ { \mathrm { a c } } ( A ) = 0 . 1 0 \mathcal { V } ( A ) + 0 . 0 0 5 C ( A )$ and receives $A =$ $\mathrm { L N } ( \widehat { Z } _ { k } ^ { ( a ) } )$ , the rowwise layer-normalized predictor matrix defined before Equation (112). The term $\mathcal { A } _ { k } ^ { ( a ) }$ averages the nodewise target-alignment cosine losses. The view-agreement weight is $\gamma _ { \mathrm { v i e w } } = \mathrm { \tilde { 0 } } . 1 0$ . The fixed predictive coefficient is $\lambda _ { \mathrm { p r e d } } = 1$ . Supervised cross-entropy is evaluated only on the training mask. Prediction uses all client-local nodes, attributes, and incidence. The validation-selected checkpoint uses Accuracy with Macro-F1 tie-breaking and is tested once.

Shared matrix powers across diffusion times. The required powers of $P _ { k }$ applied to $U _ { k }$ are computed recursively and shared across all diffusion times, with time-specific Poisson coefficients. Let $L _ { \mathrm { t a i l } }$ be the recurrence degree chosen by Equation (59), let $\mathcal { E } _ { k } ^ { \mathrm { r e l } }$ be the retained relation columns, and let $\Omega _ { k } = \{ ( \nu , e ) : B _ { k , \nu e } = 1 \}$ be the client incidence list. One application of $P _ { k }$ costs $O ( | \Omega _ { k } | d )$ time and $O ( ( | V _ { k } | + | \mathcal { E } _ { k } ^ { \mathrm { r e l } } | ) d )$ working storage. The complete basis computation therefore has the following time and storage complexity:

$$
\begin{array} { r } { O \big ( L _ { \mathrm { t a i l } } | \Omega _ { k } | d + L _ { \mathrm { t a i l } } R | V _ { k } | d \big ) \quad \mathrm { ~ t i m e , ~ } \qquad O \Big ( ( R + 1 ) | V _ { k } | d + | \mathcal { E } _ { k } ^ { \mathrm { r e l } } | d + | \Omega _ { k } | \Big ) \quad \mathrm { s t o r a g e . } } \end{array}\tag{115}
$$

Increasing � adds coefficient accumulation and representation width, but does not require additional incidence traversals for each channel. The current maximum learned time and fixed tolerance determine the degree $L _ { \mathrm { t a i l } }$ . Although the target heat operator has infinite Poisson support, Equation (58) evaluates it at finite degree with certified forward and time-gradient errors.

Table 5 reports the recurrence degree, numerical error, and local evaluation cost under the finitetask implementation. The evaluated client has 445 nodes, 485 anonymous relation columns, 774 incidences, and 64 feature channels. Each CPU median is computed over five timing runs. Across all settings, the maximum absolute residual is $2 . 3 8 \times 1 0 ^ { - 7 }$ . The results show that the heat-basis computation meets the prescribed error tolerance, while maintaining small numerical residuals and moderate local computation cost.

Table 5: Heat-basis evaluation on one CoraCA client at $K = 6 . \ L _ { \mathrm { t a i l } }$ is selected using the $1 0 ^ { - 6 }$ tail criterion in Equation (59), and residuals are measured against a reference evaluation with tolerance $1 0 ^ { - 1 2 }$ . CPU timings are reported in milliseconds (ms) on identical local inputs.
<table><tr><td>R</td><td> $L _ { \mathrm { t a i l } }$ </td><td></td><td></td><td>Forward tail Time-gradient tail Relative l2 residual CPU median (ms)</td><td></td></tr><tr><td>2</td><td>11</td><td> $2 . 9 5 \times 1 0 ^ { - 8 }$ </td><td> $2 . 8 7 \times 1 0 ^ { - 7 }$ </td><td> $3 . 0 7 \times 1 0 ^ { - 8 }$ </td><td>26.30</td></tr><tr><td>4</td><td>15</td><td> $4 . 3 5 \times 1 0 ^ { - 8 }$ </td><td> $2 . 9 8 \times 1 0 ^ { - 7 }$ </td><td> $2 . 2 6 \times 1 0 ^ { - 8 }$ </td><td>33.72</td></tr><tr><td>6</td><td>18</td><td> $9 . 3 4 \times 1 0 ^ { - 8 }$ </td><td> $5 . 2 6 \times 1 0 ^ { - 7 }$ </td><td> $1 . 4 8 \times 1 0 ^ { - 8 }$ </td><td>33.27</td></tr><tr><td>8</td><td>21</td><td> $1 . 0 6 \times 1 0 ^ { - 7 }$ </td><td> $5 . 3 4 \times 1 0 ^ { - 7 }$ </td><td> $2 . 2 0 \times 1 0 ^ { - 8 }$ </td><td>42.99</td></tr></table>

Communication protocol and privacy boundary. Table 6 summarizes the information available to the server and the state retained locally. Under this protocol, FedASTRA communicates only the online model parameters, while auxiliary training states remain client-local. FedAvg aggregates client updates with weights proportional to their fixed labeled-training counts $n _ { k } ^ { \mathrm { t r } } .$ . The local EMA target is reset at each round, while learned node states, incidence data, and numerical heat buffers remain client-local. This communication protocol does not by itself provide end-to-end differentia privacy. Model updates may still reveal information about attributes or labels, and secure aggregation alone does not eliminate the risk of aggregate inversion attacks (Shi et al., 2025). Record-level privacy would require an explicit threat model together with additional mechanisms such as calibrated randomization, privacy accounting, and compatible secure aggregation.

Table 6: Server-visible information and client-local state for FedASTRA.
<table><tr><td>Protocol item</td><td>Declared scope</td></tr><tr><td>Server receives</td><td>Online model parameters from each participating client.</td></tr><tr><td>Server broadcasts</td><td>The aggregated online model parameters.</td></tr><tr><td>Fixed aggregation metadata</td><td>Each client&#x27;s labeled-training count, fixed before training.</td></tr><tr><td>client-local state</td><td>Raw attributes, labels, incidence tables, relation membership, relation identifiers, round-local optimizer state, heat powers, latent states, and the local EMA target context encoder reset at each round.</td></tr><tr><td>Auxiliary learned-state payload</td><td>Zero bytes. No embeddings, relation summaries, or target parameters are transmitted.</td></tr><tr><td>Structural access claim</td><td>Direct cross-client node, edge, and source-hyperedge identifiers are absent.</td></tr><tr><td>Privacy extension</td><td>Secure aggregation and differential privacy require their own mechanism, parameters, and accountant.</td></tr></table>

## D.2 GRAPH PARTITIONS AND LIFTS

Main graph partition. For the main experiments, following (Yao et al., 2023; Luo et al., 2025), we fix the task split before client assignment and partition nodes across $K \in \{ 3 , 6 , 9 \}$ clients using class-conditioned Dirichlet sampling. Client allocation probabilities are drawn from a symmetric Dirichlet distribution with concentration $1 0 ^ { 4 }$ , subject to the declared load cap and minimum client size. The resulting assignment is precomputed from the full class labels, fixed before model training, and shared by all compared methods. For graph datasets, this node partition is also fixed before constructing client-local one-hop relations from the induced subgraphs.

Truncation-controlled graph partition. We construct low-, reference-, and high-truncation partitions while preserving each client’s load and label histogram. Each paired comparison begins from one balanced label allocation and permits swaps between nodes of the same class. Every proposal therefore preserves each client’s load and label histogram. With nonnegative fixed weights $\alpha _ { 2 } , \alpha _ { 3 }$ and $\rho _ { k } = n _ { k } / \sum _ { j } n _ { j }$ , the three objectives are

$$
\begin{array} { l } { { J _ { \mathrm { l o w } } ( z ) = \displaystyle \sum _ { q \in \{ 2 , 3 \} } \alpha _ { q } \tau _ { q } ( z ) , } } \\ { { J _ { \mathrm { r e f } } ( z ) = \displaystyle \sum _ { q \in \{ 2 , 3 \} } \alpha _ { q } \left( \tau _ { q } ( z ) - \left[ 1 - \displaystyle \sum _ { k } \rho _ { k } ^ { q } \right] \right) ^ { 2 } , ~ \mathrm { s . t . } ~ | V _ { k } | = n _ { k } , ~ \mathrm { h i s t } ( Y _ { V _ { k } } ) = h _ { k } . } } \\ { { J _ { \mathrm { h i g h } } ( z ) = - \displaystyle \sum _ { q \in \{ 2 , 3 \} } \alpha _ { q } \tau _ { q } ( z ) , } } \end{array}\tag{116}
$$

Here, $\begin{array} { r } { 1 - \sum _ { k } \rho _ { k } ^ { q } } \end{array}$ is the expected order-� truncation level under independent allocation with client proportions $\rho _ { k }$ . We minimize the corresponding objective for the low-truncation, independentallocation reference, and high-truncation partitions. Each pass proposes 2,048 same-class cross-client swaps, accepts strict improvements, and stops after at most 16 passes or after a pass accepts zero swaps. The three partitions thus have identical client loads and client-by-label histograms while their truncation coordinates differ. Hyperedges are treated as sets, relation multiplicity and type are retained, and every objective is invariant to stored member order. We assign equal weight to the pair and triple coordinates, $\alpha _ { 2 } = \alpha _ { 3 } = 1$

Graph lifting. For graph datasets, we adopt a partition-then-lift protocol. Each client first receives the induced subgraph $\bar { G } [ \bar { V } _ { k } ]$ , from which one-hop relations are constructed without remote endpoints or cross-client edges. The following proposition clarifies its relation to the alternative lift-thenpartition construction.

Proposition 10 (Lift-then-partition and partition-then-lift protocols). Let the rooted global one-hop relation be $e _ { \nu } = \{ \nu \} \cup N _ { G } ( \nu )$ . For a local root $\nu \in V _ { k } ,$ , the two graph-lift protocols obey the exact identity belowfor every client and every locally assigned root under one-hop lifting:

$$
e _ { \nu } \cap V _ { k } = \{ \nu \} \cup N _ { G [ V _ { k } ] } ( \nu ) .\tag{117}
$$

Thus a local-root one-hop fragment has the same members under both constructions. Lift-thenpartition additionally creates fragments on client � for roots $\nu \notin V _ { k } .$ . Partition-then-lift contains locally rooted columns. For multi-hop relations, even a local root can differ because a path may leave $V _ { k }$ and return. This identity locates the extra globally constructed information in lift-then-partition and isolates globally suppliedfragmentsfrom edges removed by the induced client graph.

Table 7: Statistics of the datasets used in the experiments.
<table><tr><td></td><td>CoraCA</td><td>DBLP4k</td><td>IMDB4k</td><td>Trivago</td><td>Cora</td><td>CiteSeer</td><td>Facebook</td><td>ogbn-papers100M</td></tr><tr><td>#Nodes</td><td>2,708</td><td>4,057</td><td>4,278</td><td>172,738</td><td>2,708</td><td>3,327</td><td>22,470</td><td>111,059,956</td></tr><tr><td>#Graph edges</td><td></td><td></td><td></td><td></td><td>5,278</td><td>4,552</td><td>85,424</td><td>1,614,062,356</td></tr><tr><td>#Hyperedges</td><td>1,072</td><td>22,051</td><td>7,338</td><td>233,202</td><td>2,708</td><td>3,327</td><td>22,470</td><td>111,059,956</td></tr><tr><td>#Incidences</td><td>4,585</td><td>110,330</td><td>17,106</td><td>726,861</td><td>13,264</td><td>12,431</td><td>193,318</td><td>3,339,184,668</td></tr><tr><td>#Classes</td><td>7</td><td>4</td><td>3</td><td>160</td><td>7</td><td>6</td><td>4</td><td>172</td></tr><tr><td>#Features</td><td>1,433</td><td>334</td><td>3,066</td><td>300</td><td>1,433</td><td>3,703</td><td>4,714</td><td>128</td></tr></table>

## D.3 DATASET DETAILS

We use two types of datasets for evaluation, including native hypergraph datasets and graph datasets. For graph datasets, #Graph edges counts distinct non-self undirected edges used for the one-hop lift, with duplicate and reciprocal entries counted once. Table 7 summarizes the statistics of the datasets used in the experiments.

Native hypergraph datasets. CoraCA, DBLP4k, and IMDB4k are loaded from the public dataset release accompanying HGNN and HGNN+ (Feng et al., 2019; Gao et al., 2023). CoraCA uses coauthorship relations (Yadati et al., 2019). DBLP4k retains paper and term relations, and IMDB4k retains director and actor relations as separate channels (Sun et al., 2011; Fu et al., 2020). Trivago follows the public DHG-Bench release (Li et al., 2026), where browsing sessions define hyperedges. Distinct source relation instances are retained even when they contain the same set of nodes. FedASTRA combines all released incidence columns in one observable operator. Relation types are used only to define analysis strata and are not provided as categorical embeddings to the model.

Graph datasets. The graph-derived experiments use the Cora and CiteSeer paper citation networks (Yang et al., 2016), the Facebook page mutual-like network (Rozemberczki et al., 2021), and the large-scale ogbn-papers100M paper citation network (Hu et al., 2020). Following the partition-thenlift protocol in Appendix D.2, nodes are first assigned to clients and each local root $\nu \in V _ { k }$ then defines the closed one-hop relation $\{ \nu \} \cup ( N _ { G } ( \nu ) \bar { \cap } V _ { k } )$ . Thus, every retained incidence is visible to its assigned client through the common released hyperedge. The ogbn-papers100M experiment follows its official implementation and processes all nodes and citations without sampling.

## D.4 EXPERIMENTAL CONFIGURATIONS

Training and evaluation protocols. Within each comparison, we use the same data splits, released inputs, communication rounds, local updates, and validation-selection rule. Following (Luo et al., 2025), training fractions are 0.10, 0.06, and 0.06 for CoraCA, DBLP4k, and IMDB4k, respectively, with validation and test fractions of 0.20 and 0.40. The remaining unlabeled nodes retain their structural roles and, together with labeled nodes, provide attributes and released local structure to the transductive encoders. Trivago uses a per-class stratified 0.50/0.25/0.25 train/validation/test split that assigns every node once following DHG-Bench (Li et al., 2026). For graph datasets, Cora and CiteSeer use the 0.10/0.20/0.40 ratios, while Facebook uses 0.008/0.20/0.40, following (Luo et al., 2025). For all datasets except ogbn-papers100M, the splits are stratified by class and fixed before client assignment. Ogbn-papers100M uses its official temporal split Hu et al. (2020). Training labels are used for model fitting, validation performance is used for checkpoint selection, and the selected model is evaluated once on the test set. The standard experiments use 150 communication rounds with three local updates per round, while Trivago uses 30 rounds. Results use five repetitions, except for Fed-HNHN and Trivago experiments, which use three. The ogbn-papers100M scale experiment uses 20 communication rounds with one local epoch per round and is reported from a single full-data run.

Node classification is evaluated using Accuracy and Macro-F1. Macro-F1 averages the per-class F1 scores over the full label set, assigning zero to any class absent from the test split. The evaluator also computes $\tau _ { 2 }$ and $\tau _ { 3 }$ from the source hypergraph and the fixed experimental partition. These quantities measure pair- and triple-order truncation and are used only for evaluation, not for model fitting, checkpoint selection, or inference. For summaries over $K \in \{ 3 , 6 , 9 \}$ , we first average the results across the three client counts within each repetition, and then report the mean and sample standard deviation across repetitions.

Baseline configurations. For the native-hypergraph comparison, all baselines use the same client local attributes and anonymous incidence columns as FedASTRA, with independently indexed columns across clients. Only trainable model parameters are communicated. We report baseline results only when the dataset, task split, client assignment, observed structure, communication rounds, local updates, and evaluation protocol match those of FedASTRA. Fed-HNHN follows the FedHGL implementation with bias-free linear maps, mean aggregation, and hidden BatchNorm (Luo et al., 2025). Its BatchNorm running statistics remain client-local and are not included in FedAvg.

For the graph comparison, all methods use the same split, client assignment, communication rounds, local updates, and evaluation protocol. FedSage, FedGCN (0-hop), and FedASTRA use only clientlocal structure. FedSage+ exchanges generated boundary features and reconstruction gradients without direct cross-client identifiers (Zhang et al., 2021). FedGCN (2-hop) relies on cross-client neighborhood information, using pre-aggregated neighbor features in the original protocol (Yao et al., $2 0 2 3 )$ , while FedCog additionally exchanges intermediate representations through known cross-client node and edge correspondences (Lei et al., 2023). These different information-access settings are retained when interpreting the predictive results.

For the FedASTRA configuration, Table 8 summarizes the settings used in the standard experiments and on Trivago. The order � is used only for the identification analysis and is not a model hyperparameter. The corresponding evaluator queries do not affect model training or checkpoint selection.

Table 8: Configurations of FedASTRA in the standard experiments and on Trivago.
<table><tr><td></td><td>Standard experiments</td><td>Trivago</td></tr><tr><td>Hyperedge input</td><td>Type-marginalized binary incidence, unit weights</td><td>Type-marginalized binary incidence, unit weights</td></tr><tr><td>Online model</td><td>Hidden dimension 64, dropout 0.35</td><td>Hidden dimension 128, dropout 0.20</td></tr><tr><td>Optimization</td><td>Adam, learning rate  $\overline { { 1 0 ^ { - 2 } } } ,$  weight decay  $5 \times 1 0 ^ { - 4 }$  , gradient clip 5.0</td><td>Adam, learning rate  $\overline { { 3 \times 1 0 ^ { - 3 } } }$  , weight decay  $\overline { { 5 \times 1 0 ^ { - 4 } } }$  , gradient clip 5.0</td></tr><tr><td>Heat basis</td><td> $\overline { { R = 4 , } }$  increment floor 10-4, tail tolerance 10-6</td><td>R = 8, increment floor 10−4, tail tolerance 10−6</td></tr><tr><td>Predictive path</td><td>Two masks with probability 0.40,  $\overline { { \lambda _ { \mathrm { p r e d } } \ : = \ : 1 , } }$  EMA decay 0.99</td><td>Two masks with probability 0.30,  $\overline { { \lambda _ { \mathrm { p r e d } } = 1 } }$  , EMA decay 0.996</td></tr></table>

Ablation settings. For the adaptive diffusion ablation, we evaluate $R \in \{ 0 , 1 , 2 , 4 , 8 \}$ with $\lambda _ { \mathrm { p r e d } } = 1$ on CoraCA, DBLP4k, and Trivago at $K = 6$ . The $R = 0$ control retains the feature encoder, classifier, masked predictor, and EMA target but removes diffusion. Three paired repetitions use the same node splits, client assignments, and training seeds across all five settings, giving 45 fitted models in total. Because increasing � also increases the representation dimension and model size, we report the corresponding parameter count for each setting. The parameter count includes the feature encoder, learned time increments, classifier, and predictive head, while excluding the client-local EMA target. Let $d _ { x }$ be input-feature dimension, � the hidden dimension, and � the number of classes. For the single-layer linear classifier in our implementation, the number of online parameters is

$$
N _ { R } = d _ { x } d + 4 d ^ { 2 } + ( C + 1 2 ) d + C + R d ( 4 d + C + 4 ) .\tag{118}
$$

The feature encoder contributes $d _ { x } d + 3 d ,$ the time increments $R d ,$ the classifier $( R + 1 ) d ( C + 2 ) + C$ and the predictive head $4 ( R + 1 ) d ^ { 2 } + ( R + 1 ) d + 6 d$ . This accounting provides the model-size context for the adaptive diffusion ablation. For the masked prediction ablation, we evaluate $R \in \{ 1 , 4 \}$ and $\lambda _ { \mathrm { p r e d } } \in \{ 0 , 1 \}$ on CoraCA and DBLP4k at $K = 6$

To measure the computation cost of the adaptive diffusion ablation, each dataset is run on one NVIDIA RTX A6000 GPU, with repetitions and � settings executed sequentially. Elapsed time is measured with CUDA synchronization and includes model initialization, all training rounds, validation-based checkpoint selection, checkpoint restoration, and final post-training measurements, while excluding dataset loading, split construction, and the final test evaluation. Figure 3 and Table 17 report predictive performance, parameter counts, and elapsed training time under this common measurement protocol.

Communication accounting. Let $b _ { \theta }$ denote the size in bytes of the online model parameters and $K _ { u } = \lvert C _ { u } \rvert$ be the number of participating clients in round �. FedASTRA transmits one model download and one model upload per participating client, giving

$$
b _ { \mathrm { r o u n d } , u } = K _ { u } b _ { \theta } + K _ { u } b _ { \theta } = 2 K _ { u } b _ { \theta } , \qquad b _ { \mathrm { a u x i l i a r y } } = 0 .\tag{119}
$$

Model communication is reported as the logical payload of these uploads and downloads. Serialization, encryption, packet headers, retransmission, and other transport overheads are excluded for all methods. For the federated graph baselines, auxiliary communication accounts for additional method-specific information required by the corresponding implementation, including generated features and reconstruction gradients for FedSage+, two-hop neighborhood information for FedGCN (2-hop), and routed intermediate representations for FedCog. Model communication is reported per round, while auxiliary communication is accumulated over the full run.

Full-data ogbn-papers100M protocol without sampling. To support full-data training at this scale, ogbn-papers100M is processed with streaming minibatches. Each round traverses every training root once, and a 200,000-incidence budget controls batch size without truncating any rooted relation. Each minibatch performs one optimizer step. For FedASTRA, AdamW (Loshchilov & Hutter, 2019) uses a learning rate of $2 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , dropout 0.2, and gradient-norm clipping at 2.0. The model uses hidden dimension $1 9 7 , R = 4$ , mask probability $0 . 4 0 , \lambda _ { \mathrm { p r e d } } = 1$ , and EMA decay 0.99. The hidden dimension is selected to match the approximately one-million-parameter baseline budget. It gives 976,504 trainable parameters, while dimension 198 would give 985,420 and exceed the 984,442-parameter ceiling.

In the minibatch implementation, each rooted example is represented independently. If a paper appears in multiple rooted relations, a separate copy is created for each relation, preventing message passing between different prediction examples. Each copied node therefore belongs to exactly one relation column, and different relation columns have disjoint support. This gives $D _ { \nu } = I$ and $B ^ { \top } B = D _ { e } ,$ so the occurrence-space operator $P _ { \mathrm { o c c } } = B D _ { e } ^ { - 1 } B ^ { \dagger }$ is an orthogonal projection:

$$
P _ { \mathrm { o c c } } ^ { 2 } = B D _ { e } ^ { - 1 } ( B ^ { \top } B ) D _ { e } ^ { - 1 } B ^ { \top } = P _ { \mathrm { o c c } } .\tag{120}
$$

The heat kernel can therefore be evaluated exactly for every minibatch as

$$
{ \sf K } _ { \mathrm { o c c } } ( t ) = e ^ { - t } { \cal I } + ( 1 - e ^ { - t } ) P _ { \mathrm { o c c } } .\tag{121}
$$

Corollary 11 (Exact spectral form in the occurrence protocol). For every scalar multiplier � on spec $( P _ { \mathrm { o c c } } ) \subseteq \{ 0 , 1 \}$

$$
f ( P _ { \mathrm { o c c } } ) = f ( 0 ) I + [ f ( 1 ) - f ( 0 ) ] P _ { \mathrm { o c c } } .\tag{122}
$$

Consequently,for any $t > 0 ,$ , span $\{ I , \mathsf { K } _ { \mathrm { o c c } } ( t ) \} = \mathrm { s p a n } \{ I , P _ { \mathrm { o c c } } \}$ . Therefore, the identity channel and any one positive-time heat channel span the space ofall linear spectralfilters of $P _ { \mathrm { o c c } }$

The corollary follows from the spectral projectors $I - P _ { \mathrm { o c c } }$ and $P _ { \mathrm { o c c } }$ , together with $1 - e ^ { - t } >$ 0. Additional learned times provide separately parameterized feature channels before nonlinear normalization and prediction, although they do not enlarge the linear spectral space of $P _ { \mathrm { o c c } }$ . The closed form in Equation (121) relies on the disjoint-copy minibatch construction. If node copies are shared across rooted relations, the resulting operator need not satisfy $P ^ { 2 } = P$ . The general client-local transition in Equation (18) need not satisfy $\dot { P } _ { k } ^ { 2 } = P _ { k }$ and is therefore evaluated using the recurrence in Appendix C.8. In float32, the scale implementation gives max $| P _ { \mathrm { o c c } } ^ { 2 } - P _ { \mathrm { o c c } } | = 0$ and a maximum absolute difference of $1 . 9 1 \times 1 0 ^ { - 6 }$ between Equation (121) and a direct matrix exponential.

## E ADDITIONAL EXPERIMENTAL RESULTS

This section provides additional experimental results by reporting extended performance evaluations, controlled analyses of the proposed mechanisms, and detailed scalability measurements under the corresponding experimental protocols.

## E.1 PERFORMANCE ON NATIVE HYPERGRAPHS

Table 9 reports the complete native-hypergraph results for each client count $K \in \{ 3 , 6 , 9 \}$ . On CoraCA, the performance gap between FedASTRA and Fed-HGNN increases as the number of clients grows from 3 to 9, from 4.83% to 7.75% accuracy and from 6.54% to 9.84% Macro-F1.

This trend is consistent with the increasing fragmentation of local observations under matched client inputs.

Table 9: Complete native-hypergraph results under different number of clients.
<table><tr><td>Method</td><td colspan="2"> $K = 3$ </td><td colspan="2"> $K = 6$ </td><td colspan="2"> $K = 9$ </td></tr><tr><td></td><td>Acc. (%)</td><td>Macro-F1 (%)</td><td> $\operatorname { A c c . } \ ( \% )$ </td><td>Macro-F1 (%)</td><td> $\operatorname { A c c . } \ ( \% )$ </td><td>Macro-F1 (%)</td></tr><tr><td>CoraCA</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedASTRA</td><td> $7 6 . 2 7 \pm 0 . 5 4$ </td><td> $7 4 . 3 3 \pm 0 . 7 2$ </td><td> $7 4 . 9 7 \pm 0 . 6 3$ </td><td> $7 2 . 7 9 \pm 0 . 9 3$ </td><td> $7 4 . 2 7 \pm 0 . 7 3$ </td><td> $7 1 . 5 2 \pm 0 . 9 7$ </td></tr><tr><td>Fed-HGNN</td><td> $7 1 . 4 4 \pm 1 . 9 0$ </td><td> $6 7 . 7 9 \pm 2 . 1 2$ </td><td> $6 7 . 9 7 \pm 1 . 3 8$ </td><td> $6 3 . 5 7 \pm 1 . 5 0$ </td><td> $6 6 . 5 2 \pm 2 . 3 2$ </td><td> $6 1 . 6 8 \pm 3 . 0 2$ </td></tr><tr><td>Fed-HNHN</td><td>57.30 ± 0.82</td><td> $5 4 . 4 0 \pm 1 . 2 1$ </td><td> $5 4 . 1 6 \pm 2 . 4 6$ </td><td> $4 9 . 0 8 \pm 3 . 2 9$ </td><td> $5 2 . 7 4 \pm 1 . 1 3$ </td><td> $4 7 . 7 7 \pm 2 . 5 7$ </td></tr><tr><td>Fed-HyperGCN</td><td>67.38 ± 1.01</td><td> $6 3 . 0 8 \pm 1 . 6 4$   $^ { 6 3 . 0 8 \pm 1 . 6 4 } _ { 7 7 }$ </td><td> $6 3 . 8 4 \pm 2 . 1 6$ </td><td> $5 7 . 6 5 \pm 2 . 5 6$ </td><td> $6 0 . 8 5 \pm 2 . 5 6$ </td><td> $5 3 . 4 7 \pm 3 . 0 7$ </td></tr><tr><td>Fed-AiiDeepSets</td><td>45.90 ± 3.43</td><td> $3 7 . 6 2 \pm 4 . 6 2$ </td><td> $3 6 . 5 4 \pm 2 . 1 1$ </td><td> $2 4 . 3 7 \pm 2 . 8 6$ </td><td> $3 6 . 4 5 \pm 1 . 4 5$ </td><td> $1 9 . 2 6 \pm 6 . 6 2$ </td></tr><tr><td>Fed-AllSetTransformer 30.22 ± 0.00</td><td></td><td> $6 . 6 3 \pm 0 . 0 0$ </td><td> $3 0 . 2 2 \pm 0 . 0 0$ </td><td> $6 . 6 3 \pm 0 . 0 0$ </td><td> $3 0 . 2 2 \pm 0 . 0 0$ </td><td> $6 . 6 3 \pm 0 . 0 0$ </td></tr><tr><td>Fed-UniGCNII</td><td> $6 3 . 5 7 \pm 1 . 8 9$ </td><td> $5 6 . 5 2 \pm 5 . 5 3 $ </td><td> $5 8 . 8 9 \pm 1 . 5 2 $ </td><td> $4 9 . 0 1 \pm 3 . 6 7$ </td><td> $5 6 . 1 6 \pm 2 . 0 7$ </td><td> $4 4 . 8 0 \pm 3 . 5 2 $ </td></tr><tr><td>Fed-ED-HNN</td><td> $6 1 . 9 0 \pm 1 . 5 1$ </td><td> $5 7 . 2 7 \pm 1 . 5 3$ </td><td> $5 9 . 2 8 \pm 2 . 3 8$ </td><td> $5 2 . 1 1 \pm 3 . 6 2$ </td><td> $5 7 . 6 9 \pm 2 . 2 2$ </td><td> $5 0 . 4 6 \pm 3 . 4 5$ </td></tr><tr><td>DBLP4k</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedASTRA</td><td> $7 8 . 5 2 \pm 1 . 2 6$ </td><td> $7 7 . 8 4 \pm 1 . 0 4$ </td><td> $7 9 . 7 5 \pm 0 . 8 2$ </td><td> $7 9 . 1 5 \pm 0 . 7 5$ </td><td> $7 9 . 4 0 \pm 1 . 3 5$ </td><td> $7 8 . 8 4 \pm 1 . 2 4$ </td></tr><tr><td>Fed-HGNN</td><td> $7 7 . 6 2 \pm 1 . 4 8$ </td><td> $7 6 . 9 8 \pm 1 . 3 9$ </td><td> $7 7 . 7 3 \pm 1 . 6 2$ </td><td> $7 7 . 0 0 \pm 1 . 3 7$ </td><td> $7 7 . 5 5 \pm 1 . 3 0$ </td><td> $7 6 . 9 6 \pm 1 . 1 6$ </td></tr><tr><td>Fed-HNHN</td><td> $7 2 . 2 3 \pm 2 . 6 4$ </td><td> $7 1 . 4 4 \pm 2 . 6 6$ </td><td> $6 8 . 4 3 \pm 2 . 5 5$ </td><td> $6 7 . 7 0 \pm 2 . 6 7$ </td><td> $6 8 . 0 8 \pm 1 . 2 3 $ </td><td> $6 7 . 2 2 \pm 0 . 9 7$ </td></tr><tr><td>Fed-HyperGCN</td><td> $7 4 . 7 4 \pm 1 . 2 9$ </td><td> $7 3 . 4 3 \pm 1 . 6 3$ </td><td>72.29 ± 0.60</td><td> $7 0 . 8 8 \pm 0 . 9 0$ </td><td> $7 1 . 0 4 \pm 1 . 8 5$ </td><td> $7 0 . 0 6 \pm 1 . 8 0$ </td></tr><tr><td>Fed-AllDeepSets</td><td> $3 9 . 2 5 \pm 0 . 6 8$ </td><td> $3 1 . 7 4 \pm 1 . 7 9$ </td><td>39.36 ± 0.72</td><td> $3 1 . 9 6 \pm 2 . 3 6$ </td><td> $3 9 . 4 5 \pm 1 . 1 8$ </td><td> $3 3 . 6 3 \pm 1 . 7 5$ </td></tr><tr><td>Fed-AllSetTransformer</td><td> $3 0 . 8 7 \pm 0 . 9 7$ </td><td> $2 2 . 5 0 \pm 2 . 6 1$ </td><td>30.82 ± 1.20</td><td> $2 1 . 3 0 \pm 1 . 6 4$ </td><td> $3 1 . 2 8 \pm 1 . 7 3$ </td><td> $2 1 . 6 2 \pm 1 . 9 3$ </td></tr><tr><td>Fed-UniGCNII</td><td> $7 9 . 0 3 \pm 2 . 1 2$ </td><td> $7 7 . 8 5 \pm 2 . 4 7$ </td><td>77.60 ± 2.49</td><td> $7 6 . 1 8 \pm 2 . 9 8$ </td><td> $7 5 . 4 4 \pm 3 . 1 0$ </td><td> $7 3 . 9 2 \pm 3 . 2 7$ </td></tr><tr><td>Fed-ED-HNN</td><td> $7 1 . 9 2 \pm 3 . 5 2$ </td><td> $7 1 . 1 9 \pm 3 . 4 6$ </td><td> $7 2 . 3 0 \pm 2 . 3 0$ </td><td> $7 1 . 5 9 \pm 2 . 4 0$ </td><td> $7 2 . 0 9 \pm 2 . 3 6$ </td><td> $7 1 . 3 0 \pm 2 . 3 7$ </td></tr><tr><td>IMDB4k</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedASTRA</td><td> $5 1 . 9 9 \pm 1 . 8 5$ </td><td> $5 1 . 9 2 \pm 1 . 5 3$ </td><td>51.05 ± 1.53</td><td> $5 0 . 8 8 \pm 1 . 3 4$ </td><td> $5 0 . 3 4 \pm 0 . 9 2$ </td><td> $5 0 . 3 7 \pm 0 . 8 5$ </td></tr><tr><td>Fed-HGNN</td><td> $5 0 . 1 4 \pm 1 . 3 8$ </td><td> $5 0 . 1 0 \pm 1 . 3 9$ </td><td>48.36 ± 1.47</td><td> $4 8 . 2 5 \pm 1 . 4 6$ </td><td> $4 8 . 0 5 \pm 1 . 4 9$ </td><td> $4 7 . 4 1 \pm 2 . 1 1$ </td></tr><tr><td>Fed-HNHN</td><td> $4 6 . 1 4 \pm 0 . 6 2$ </td><td> $4 4 . 7 5 \pm 0 . 5 7$ </td><td>43.81 ± 2.13</td><td> $4 1 . 8 4 \pm 2 . 9 5$ </td><td> $4 4 . 9 2 \pm 1 . 9 3$ </td><td> $4 2 . 5 4 \pm 1 . 8 1$ </td></tr><tr><td>Fed-HyperGCN</td><td> $4 7 . 3 6 \pm 1 . 8 7$ </td><td> $4 6 . 3 3 \pm 2 . 2 2$ </td><td>43.26 ± 1.50</td><td> $4 1 . 3 5 \pm 2 . 3 0$ </td><td> $4 1 . 8 1 \pm 1 . 6 4$ </td><td> $3 4 . 9 0 \pm 5 . 7 1$ </td></tr><tr><td>Fed-AllDeepSets</td><td> $4 5 . 8 2 \pm 1 . 8 7$ </td><td> $4 3 . 9 6 \pm 4 . 8 3$ </td><td> $3 7 . 7 3 \pm 1 . 6 7$ </td><td> $3 0 . 2 4 \pm 1 . 8 3$ </td><td> $3 7 . 4 2 \pm 1 . 3 5$ </td><td> $2 8 . 1 5 \pm 3 . 6 7$ </td></tr><tr><td>Fed-AllSetTransformer</td><td> $3 7 . 0 4 \pm 0 . 0 3$ </td><td> $1 8 . 0 4 \pm 0 . 0 5$ </td><td> $3 7 . 0 6 \pm 0 . 0 5$ </td><td> $1 8 . 0 6 \pm 0 . 1 0$ </td><td> $3 7 . 0 3 \pm 0 . 1 1$ </td><td> $1 8 . 2 1 \pm 0 . 2 2$ </td></tr><tr><td>Fed-UniGCNII</td><td> $5 1 . 1 2 \pm 2 . 3 4$ </td><td> $4 9 . 8 8 \pm 3 . 4 5$ </td><td> $3 7 . 0 9 \pm 0 . 8 7$ </td><td> $2 1 . 9 5 \pm 5 . 3 3$ </td><td> $3 6 . 6 6 \pm 0 . 8 7$ </td><td> $1 9 . 8 7 \pm 3 . 9 6$ </td></tr><tr><td>Fed-ED-HNN</td><td> $4 8 . 9 3 \pm 1 . 9 8$ </td><td> $4 8 . 6 1 \pm 2 . 0 6$ </td><td> $4 0 . 7 5 \pm 0 . 9 4$ </td><td> $3 0 . 5 7 \pm 1 . 2 9$ </td><td> $4 0 . 7 1 \pm 0 . 9 3$ </td><td> $3 0 . 5 8 \pm 1 . 2 2 $ </td></tr><tr><td>Trivago</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedASTRA</td><td> $3 3 . 5 5 \pm 1 . 6 0$ </td><td> $9 . 6 4 \pm 0 . 3 5$ </td><td> $2 7 . 7 0 \pm 1 . 1 4$ </td><td> $7 . 5 5 \pm 0 . 3 2$ </td><td> $2 6 . 0 1 \pm 1 . 2 9$ </td><td> $6 . 7 0 \pm 0 . 1 5$ </td></tr><tr><td>Fed-HGNN</td><td> $2 2 . 9 9 \pm 0 . 7 2$ </td><td> $5 . 0 0 \pm 0 . 7 1$ </td><td> $2 2 . 0 8 \pm 0 . 7 7$ </td><td> $4 . 0 5 \pm 0 . 5 9$ </td><td> $2 0 . 4 8 \pm 0 . 8 7$ </td><td> $3 . 7 5 \pm 0 . 6 7$ </td></tr><tr><td>Fed-HNHN</td><td> $2 9 . 1 0 \pm 1 . 3 6$ </td><td> $9 . 1 9 \pm 0 . 1 9$ </td><td> $2 4 . 8 7 \pm 1 . 0 6$ </td><td> $7 . 3 7 \pm 0 . 3 5$ </td><td> $2 3 . 0 8 \pm 0 . 7 0$ </td><td> $6 . 5 9 \pm 0 . 2 4$ </td></tr><tr><td>Fed-HyperGCN</td><td> $9 . 4 0 \pm 0 . 6 5$ </td><td> $0 . 1 5 \pm 0 . 0 6$ </td><td> $9 . 0 0 \pm 0 . 0 2$ </td><td> $0 . 1 1 \pm 0 . 0 1$ </td><td> $9 . 1 0 \pm 0 . 2 1$ </td><td> $0 . 1 3 \pm 0 . 0 4$ </td></tr><tr><td>Fed-AllDeepSets</td><td> $1 1 . 3 7 \pm 0 . 1 1$ </td><td> $0 . 3 0 \pm 0 . 0 1$ </td><td> $1 0 . 8 2 \pm 0 . 1 5$ </td><td> $0 . 2 8 \pm 0 . 0 3$ </td><td> $1 0 . 4 5 \pm 0 . 2 2$ </td><td> $0 . 2 4 \pm 0 . 0 1$ </td></tr><tr><td>Fed-AllSetTransformer</td><td> $8 . 9 9 \pm 0 . 0 0$ </td><td> $0 . 1 0 \pm 0 . 0 0$ </td><td> $8 . 9 9 \pm 0 . 0 0$ </td><td> $0 . 1 0 \pm 0 . 0 0$ </td><td> $8 . 9 9 \pm 0 . 0 0$ </td><td> $0 . 1 0 \pm 0 . 0 0$ </td></tr><tr><td>Fed-UniGCNII</td><td> $9 . 2 5 \pm 0 . 2 3$ </td><td> $0 . 1 6 \pm 0 . 0 5$ </td><td> $9 . 1 4 \pm 0 . 0 9$ </td><td> $0 . 1 6 \pm 0 . 0 1$ </td><td> $9 . 0 7 \pm 0 . 1 0$ </td><td> $0 . 1 3 \pm 0 . 0 3$ </td></tr><tr><td>Fed-ED-HNN</td><td> $9 . 5 4 \pm 0 . 8 6$ </td><td> $0 . 1 6 \pm 0 . 0 9$ </td><td> $9 . 0 5 \pm 0 . 0 6$ </td><td> $0 . 1 3 \pm 0 . 0 3$ </td><td> $8 . 9 9 \pm 0 . 0 1$ </td><td> $0 . 1 0 \pm 0 . 0 0$ </td></tr></table>

## E.2 PERFORMANCE AGAINST BASELINES WITH VARYING INFORMATION ACCESS

Table 10 reports comparisons against federated graph learning baselines under different cross-client information-access settings, while Table 11 summarizes methods according to their available structural information and communication requirements. FedASTRA consistently outperforms identifier-free baselines across client counts and also surpasses FedSage+ despite the latter introducing additional exchanged information. In contrast, FedGCN (2-hop) and FedCog access explicit cross-client correspondence and therefore exploit a strictly richer observation setting. These results suggest that additional cross-client information is not universally beneficial: its value depends on whether the introduced information can be reliably exploited under the correspondence assumptions.

Table 10: Complete results under different cross-client information-access settings.
<table><tr><td>Method</td><td colspan="2"> $K = 3$ </td><td colspan="2">K = 6</td><td colspan="2"> $K = 9$ </td></tr><tr><td></td><td>Acc. (%)</td><td>Macro-F1 (%)</td><td>Acc. (%)</td><td>Macro-F1 (%)</td><td> $\operatorname { A c c . } \ ( \% )$ </td><td>Macro-F1 (%)</td></tr><tr><td>Cora</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedSage</td><td> $7 0 . 3 5 \pm 1 . 9 6$ </td><td> $6 7 . 7 7 \pm 1 . 6 7$ </td><td> $6 3 . 1 8 \pm 0 . 9 7$ </td><td> $5 8 . 5 9 \pm 0 . 8 7$ </td><td> $5 8 . 6 5 \pm 1 . 8 5$ </td><td> $5 2 . 1 7 \pm 2 . 8 1$ </td></tr><tr><td>FedGCN (0-hop)</td><td> $7 4 . 0 7 \pm 1 . 5 7$ </td><td> $7 1 . 2 4 \pm 1 . 8 6$ </td><td> $6 5 . 9 0 \pm 2 . 0 5$ </td><td> $6 1 . 9 6 \pm 3 . 3 4$ </td><td> $6 2 . 0 7 \pm 1 . 7 9$ </td><td> $5 7 . 4 4 \pm 2 . 9 1$ </td></tr><tr><td>FedSage+</td><td> $6 9 . 6 7 \pm 0 . 7 1 $ </td><td> $6 6 . 8 7 \pm 0 . 3 0$ </td><td> $6 3 . 1 1 \pm 1 . 6 6$ </td><td> $5 7 . 7 8 \pm 1 . 6 3$ </td><td> $6 1 . 0 2 \pm 1 . 4 0$ </td><td> $5 5 . 0 5 \pm 3 . 6 5$ </td></tr><tr><td>FedGCN (2-hop)</td><td> $8 4 . 0 5 \pm 1 . 1 7$ </td><td> $8 3 . 0 1 \pm 1 . 2 3$ </td><td> $8 4 . 0 5 \pm 1 . 1 6$ </td><td> $8 3 . 1 2 \pm 1 . 2 0$ </td><td> $8 4 . 1 2 \pm 0 . 9 2$ </td><td> $8 2 . 9 2 \pm 0 . 7 8$ </td></tr><tr><td>FedCog</td><td> $8 4 . 2 9 \pm 1 . 5 0$ </td><td> $8 3 . 1 5 \pm 1 . 6 4$ </td><td> $8 4 . 3 4 \pm 1 . 0 6$ </td><td> $8 3 . 3 5 \pm 1 . 0 0$ </td><td> $8 3 . 9 2 \pm 0 . 7 8$ </td><td> $8 2 . 7 3 \pm 0 . 6 6$ </td></tr><tr><td>FedASTRA</td><td> $7 7 . 5 4 \pm 1 . 0 8$ </td><td> $7 6 . 0 6 \pm 1 . 3 1$ </td><td> $7 4 . 4 5 \pm 1 . 6 0$ </td><td> $7 2 . 1 3 \pm 1 . 8 1$ </td><td> $7 3 . 3 3 \pm 1 . 2 2$ </td><td> $7 1 . 8 3 \pm 1 . 3 1$ </td></tr><tr><td>CiteSeer</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedSage</td><td> $6 5 . 0 8 \pm 0 . 9 6$ </td><td> $5 9 . 5 6 \pm 1 . 4 2 $ </td><td> $6 3 . 7 3 \pm 0 . 7 4$ </td><td> $5 7 . 0 0 \pm 1 . 5 1 $ </td><td> $6 2 . 9 3 \pm 1 . 7 2$ </td><td> $5 5 . 6 7 \pm 1 . 2 5$ </td></tr><tr><td>FedGCN (0-hop)</td><td> $6 8 . 6 8 \pm 1 . 5 0 $ </td><td> $6 1 . 6 0 \pm 1 . 3 2$ </td><td> $6 5 . 8 3 \pm 0 . 5 6$ </td><td> $5 9 . 0 6 \pm 0 . 5 2 $ </td><td> $6 5 . 1 4 \pm 1 . 8 0$ </td><td> $5 8 . 1 0 \pm 1 . 6 8$ </td></tr><tr><td>FedSage+</td><td> $6 3 . 1 6 \pm 1 . 1 7$ </td><td> $5 7 . 7 5 \pm 1 . 5 9$ </td><td> $6 2 . 5 3 \pm 1 . 6 6$ </td><td> $5 6 . 4 8 \pm 2 . 6 1$ </td><td> $6 2 . 9 5 \pm 1 . 6 2$ </td><td> $5 7 . 6 9 \pm 2 . 1 0$ </td></tr><tr><td>FedGCN (2-hop)</td><td> $7 2 . 2 0 \pm 1 . 4 4$ </td><td> $6 5 . 6 4 \pm 2 . 2 6$ </td><td> $7 2 . 2 7 \pm 1 . 5 3$ </td><td> $6 4 . 8 8 \pm 1 . 4 4 $ </td><td> $7 2 . 3 8 \pm 1 . 5 0$ </td><td> $6 4 . 8 5 \pm 2 . 0 7$ </td></tr><tr><td>FedCog</td><td> $7 2 . 2 1 \pm 1 . 4 9$ </td><td> $6 5 . 8 4 \pm 1 . 7 7$ </td><td> $7 2 . 3 8 \pm 1 . 0 4$ </td><td> $6 5 . 0 9 \pm 1 . 0 3$ </td><td> $7 2 . 5 1 \pm 1 . 4 2$ </td><td> $6 5 . 1 1 \pm 1 . 9 4$ </td></tr><tr><td>FedASTRA</td><td> $6 7 . 6 1 \pm 2 . 3 2$ </td><td> $6 3 . 6 2 \pm 1 . 6 1$ </td><td> $6 8 . 0 0 \pm 1 . 8 6$ </td><td> $6 3 . 6 7 \pm 2 . 0 7$ </td><td> $6 7 . 9 2 \pm 1 . 8 8$ </td><td> $6 3 . 6 5 \pm 2 . 1 3$ </td></tr><tr><td>Facebook</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedSage</td><td> $7 2 . 6 3 \pm 0 . 7 3$ </td><td> $6 9 . 0 0 \pm 1 . 0 8$ </td><td> $6 5 . 6 4 \pm 1 . 8 1$ </td><td> $6 1 . 8 5 \pm 2 . 3 9$ </td><td> $6 0 . 4 1 \pm 1 . 6 1$ </td><td> $5 3 . 4 6 \pm 4 . 1 6$ </td></tr><tr><td>FedGCN (0-hop)</td><td> $7 5 . 1 4 \pm 0 . 9 5$ </td><td> $7 2 . 6 2 \pm 0 . 7 4$ </td><td> $6 7 . 8 7 \pm 1 . 7 1 $ </td><td> $6 4 . 3 9 \pm 2 . 2 4$ </td><td> $6 4 . 3 0 \pm 1 . 2 2 $ </td><td> $5 9 . 4 4 \pm 2 . 8 3 $ </td></tr><tr><td>FedSage+</td><td> $6 8 . 1 7 \pm 0 . 3 3 $ </td><td> $6 5 . 4 1 \pm 0 . 6 4$ </td><td> $6 1 . 8 5 \pm 0 . 9 5$ </td><td> $5 7 . 6 2 \pm 1 . 3 2$ </td><td> $5 8 . 5 6 \pm 1 . 3 3 $ </td><td> $5 4 . 0 2 \pm 1 . 9 9$ </td></tr><tr><td>FedGCN (2-hop)</td><td> $8 3 . 2 1 \pm 0 . 7 6$ </td><td> $8 2 . 0 2 \pm 0 . 8 7$ </td><td> $8 2 . 8 4 \pm 0 . 5 0 $ </td><td> $8 1 . 5 8 \pm 0 . 6 7$ </td><td> $8 2 . 7 3 \pm 0 . 8 5$ </td><td> $8 1 . 5 1 \pm 1 . 0 4$ </td></tr><tr><td>FedCog</td><td> $8 2 . 4 6 \pm 0 . 6 9$ </td><td> $8 1 . 2 4 \pm 0 . 7 3$ </td><td> $8 2 . 5 0 \pm 0 . 8 4$ </td><td> $8 1 . 2 0 \pm 1 . 0 9$ </td><td> $8 2 . 1 5 \pm 0 . 9 4$ </td><td> $8 0 . 8 5 \pm 1 . 1 3$ </td></tr><tr><td> $\mathrm { F e d A S T R A }$ </td><td> $7 6 . 8 8 \pm 1 . 4 4$ </td><td> $7 5 . 5 8 \pm 1 . 6 0$ </td><td> $7 5 . 5 7 \pm 0 . 9 4$ </td><td> $7 3 . 7 1 \pm 0 . 8 6$ </td><td> $7 5 . 1 2 \pm 1 . 4 9$ </td><td> $7 3 . 3 0 \pm 1 . 7 6$ </td></tr></table>

Table 11: Summarized results of federated graph baselines under different cross-client informationaccess settings. Results are aggregated across client counts $K \in \{ 3 , 6 , 9 \}$
<table><tr><td>Method</td><td>Exchanged or accessed information</td><td></td><td>Accuracy (%) Macro-F1 (%)</td></tr><tr><td>Cora</td><td></td><td></td><td></td></tr><tr><td>FedASTRA</td><td>Model parameters only, with no direct cross-client identi- fiers</td><td> $7 5 . 1 1 \pm 0 . 9 9$ </td><td> $7 3 . 3 4 \pm 1 . 1 3$ </td></tr><tr><td>FedSage+</td><td>Locally generated boundary features and cross-client re- construction gradients, with no direct identifiers</td><td> $6 4 . 6 0 \pm 1 . 1 1 $ </td><td> $5 9 . 9 0 \pm 1 . 7 1 $ </td></tr><tr><td>FedGCN (2-hop)</td><td>Cross-client node IDs, edges, and pre-aggregated neigh- bor features</td><td> $8 4 . 0 7 \pm 1 . 0 3$ </td><td> $8 3 . 0 2 \pm 1 . 0 4$ </td></tr><tr><td>FedCog</td><td>Cross-client node IDs, edges, and intermediate embed- dings</td><td> $8 4 . 1 8 \pm 1 . 0 4$ </td><td> $8 3 . 0 8 \pm 1 . 0 3$ </td></tr><tr><td>CiteSeer</td><td></td><td></td><td></td></tr><tr><td>FedASTRA</td><td>Model parameters only, with no direct cross-client identi- fiers</td><td> $6 7 . 8 4 \pm 1 . 6 6$ </td><td> $6 3 . 6 4 \pm 1 . 8 3$ </td></tr><tr><td>FedSage+</td><td>Locally generated boundary features and cross-client re- construction gradients, with no direct identifiers</td><td> $6 2 . 8 8 \pm 1 . 0 2$ </td><td> $5 7 . 3 1 \pm 1 . 7 4$ </td></tr><tr><td>FedGCN (2-hop)</td><td>Cross-client node IDs, edges, and pre-aggregated neigh- bor features</td><td> $7 2 . 2 8 \pm 1 . 4 6$ </td><td> $6 5 . 1 2 \pm 1 . 6 8$ </td></tr><tr><td>FedCog</td><td>Cross-client node IDs, edges, and intermediate embed-</td><td> $7 2 . 3 7 \pm 1 . 2 6$ </td><td> $6 5 . 3 5 \pm 1 . 4 4$ </td></tr><tr><td>Facebook</td><td>dings</td><td></td><td></td></tr><tr><td>FedASTRA</td><td>Model parameters only, with no direct cross-client identi- fiers</td><td> $7 5 . 8 6 \pm 1 . 0 2$ </td><td> $7 4 . 2 0 \pm 1 . 0 7$ </td></tr><tr><td>FedSage+</td><td>Locally generated boundary features and cross-client re- construction gradients, with no direct identifiers</td><td> $6 2 . 8 6 \pm 0 . 7 1$ </td><td> $5 9 . 0 2 \pm 1 . 0 6$ </td></tr><tr><td></td><td>FedGCN (2-hop) Cross-client node IDs, edges, and pre-aggregated neigh-</td><td> $8 2 . 9 2 \pm 0 . 6 1$ </td><td> $8 1 . 7 0 \pm 0 . 7 2$ </td></tr><tr><td>FedCog</td><td>bor features Cross-client node IDs, edges, and intermediate embed- dings</td><td> $8 2 . 3 7 \pm 0 . 7 3$ </td><td> $8 1 . 1 0 \pm 0 . 9 0$ </td></tr></table>

## E.3 VALIDATION ON PROPOSED ANALYSIS AND MECHANISM

Table 12 isolates the contributions of the adaptive diffusion basis and the masked prediction objective. Removing masked prediction $\lambda _ { \mathrm { p r e d } } = 0$ consistently reduces both Accuracy and Macro-F1 on CoraCA and DBLP4k, indicating that the predictive alignment component provides complementary representation learning signals. Increasing the heat basis width from $R \ = \ 1$ to $R \ = \ 4$ further improves performance when the predictive objective is enabled, while its effect is smaller without masked prediction. These results support the complementary roles of adaptive diffusion and masked representation alignment in FedASTRA.

Table 12: Ablation study of adaptive diffusion and masked prediction at $K = 6 .$ . This study is separate from the heat-width sweep in Appendix D.
<table><tr><td>Dataset</td><td>Reported condition</td><td>Parameters</td><td>Accuracy (%)</td><td>Macro-F1 (%)</td></tr><tr><td>CoraCA</td><td> $R = 1 , \lambda _ { \mathrm { p r e d } } = 0$ </td><td>126,407</td><td> $6 7 . 2 6 \pm 1 . 4 8$ </td><td> $6 2 . 9 4 \pm 1 . 7 5$ </td></tr><tr><td>CoraCA</td><td> $R = 1 , \ : \lambda _ { \mathrm { p r e d } } ^ { \cdot } = 1$ </td><td>126,407</td><td> $7 4 . 2 7 \pm 0 . 5 6$ </td><td> $7 1 . 9 6 \pm 0 . 5 8$ </td></tr><tr><td>CoraCA</td><td> $R = 4 , \ : \lambda _ { \mathrm { p r e d } } = 0$ </td><td>177,671</td><td> $6 9 . 3 2 \pm 0 . 6 9$ </td><td> $6 5 . 3 1 \pm 1 . 1 5$ </td></tr><tr><td>CoraCA</td><td> $R = 4 , \lambda _ { \mathrm { p r e d } } = 1$ </td><td>177,671</td><td> $7 4 . 9 7 \pm 0 . 6 3$ </td><td> $7 2 . 7 9 \pm 0 . 9 3$ </td></tr><tr><td>DBLP4k</td><td> $R = 1 , \lambda _ { \mathrm { p r e d } } = 0$ </td><td>55,684</td><td> $7 7 . 5 4 \pm 0 . 7 0$ </td><td> $7 6 . 7 7 \pm 0 . 7 8$ </td></tr><tr><td>DBLP4k</td><td> $R = 1 , \ : \dot { \lambda _ { \mathrm { p r e d } } } = 1$ </td><td>55,684</td><td> $7 9 . 5 8 \pm 0 . 6 7$ </td><td> $7 8 . 9 9 \pm 0 . 4 9$ </td></tr><tr><td>DBLP4k</td><td> $R = 4 , \lambda _ { \mathrm { p r e d } } ^ { \mathrm { ^ { \scriptstyle ~ \cdot } } } = 0$ </td><td>106,372</td><td> $7 8 . 2 3 \pm 0 . 5 1$ </td><td> $7 7 . 6 0 \pm 0 . 5 6$ </td></tr><tr><td>DBLP4k</td><td> $R = 4 , \lambda _ { \mathrm { p r e d } } = 1$ </td><td>106,372</td><td> $7 9 . 7 5 \pm 0 . 8 2$ </td><td> $7 9 . 1 5 \pm 0 . 7 5$ </td></tr></table>

Table 13 reports the controlled-truncation results. The controlled partitions exhibit the expected ordering of retained higher-order co-membership mass. The observed pair/triple mass is 72.44/65.11%, 16.67/2.78%, and 9.68/1.10% for the low, reference, and high CoraCA partitions, respectively. The corresponding DBLP4k values are 27.21/9.24%, 16.67/2.78%, and 16.33/2.70%. These results confirm that the three partition conditions differ in structural truncation while preserving client loads and label histograms across paired comparisons. The ordering applies to co-membership mass only. Prediction performance also depends on feature and label alignment, so the truncation level alone does not determine Accuracy or Macro-F1.

Table 13: Controlled-truncation results at $\underline { { K = 6 . } }$
<table><tr><td>Dataset</td><td>Reported condition</td><td>Accuracy (%)</td><td>Macro-F1 (%)</td></tr><tr><td>CoraCA</td><td>Low-truncation</td><td> $7 7 . 9 9 \pm 0 . 5 8$ </td><td> $7 6 . 4 0 \pm 0 . 7 4$ </td></tr><tr><td>CoraCA</td><td>Reference</td><td> $7 5 . 5 8 \pm 1 . 3 3$ </td><td> $7 3 . 2 7 \pm 1 . 0 7$ </td></tr><tr><td>CoraCA</td><td>High-truncation</td><td> $7 3 . 9 7 \pm 0 . 8 0$ </td><td> $7 1 . 3 4 \pm 0 . 8 0$ </td></tr><tr><td>DBLP4k</td><td>Low-truncation</td><td> $7 8 . 2 3 \pm 1 . 7 6$ </td><td> $7 7 . 5 4 \pm 1 . 5 9$ </td></tr><tr><td>DBLP4k</td><td>Reference</td><td> $7 9 . 4 6 \pm 1 . 2 3$ </td><td> $7 8 . 8 5 \pm 1 . 1 3$ </td></tr><tr><td>DBLP4k</td><td>High-truncation</td><td> $7 9 . 2 4 \pm 1 . 4 9$ </td><td> $7 8 . 6 6 \pm 1 . 4 9$ </td></tr></table>

Table 14 and Table 15 summarize the two-client structural identification audit. The audit uses five fixed $K = 2 , c = 1$ diagnostic partitions per source, exact hyperedge types, and unit weights. Held-out parent tokens mark cross-client rows and verify the oracle $\begin{array} { r } { \Phi _ { q } ( \mathcal { H } ) = \sum _ { e } \binom { | e | } { q } } \end{array}$ after the partition is fixed. Sorting fragment sizes gives both attainable endpoints. Labels are used only by the partition constructor, and no learner is trained. These evaluator partitions are independent of the reported $K \in \{ 3 , 6 , 9 \}$ training partitions. Across all evaluated cells, no higher-order co-membership target is point identified, and the median relative endpoint width increases from 36.37% at � = 2 to 93.71% at $q = 5 .$ . These results empirically support the predicted ambiguity caused by latent hyperedge correspondence, especially for higher-order structural targets.

Table 14: Two-client structural identification audit on four native datasets. For each order $q ,$ the audit contains 20 releases and 30 eligible strata. Relative endpoint width is $\bigl ( \Phi _ { q } ^ { + } - \Phi _ { q } ^ { - } \bigr ) / \Phi _ { q } ^ { + } .$ . By Corollary $^ { 2 , }$ half of this normalized width gives the corresponding minimax lower bound for observable point estimation of the same order-� mass.
<table><tr><td>q</td><td>Releases</td><td>Eligible strata</td><td>Point identified</td><td>Median relative width</td><td>q-mass floor</td><td>Oracle coverage</td></tr><tr><td>2</td><td>20</td><td>30</td><td>0/30</td><td>36.37%</td><td>18.18%</td><td>20/20</td></tr><tr><td>3</td><td>20</td><td>30</td><td>0/30</td><td>70.56%</td><td>35.28%</td><td>20/20</td></tr><tr><td>4</td><td>20</td><td>30</td><td>0/30</td><td>86.62%</td><td>43.31%</td><td>20/20</td></tr><tr><td>5</td><td>20</td><td>30</td><td>0/30</td><td>93.71%</td><td>46.85%</td><td>20/20</td></tr></table>

Table 15: Median absolute sharp-endpoint gaps over five two-client diagnostic partitions, measured in order-� tuple units. The values preserve the original dataset scale, complementing the normalized endpoint widths reported in Table 14.
<table><tr><td>Dataset</td><td> $q = 2$ </td><td> $q = 3$ </td><td> $q = 4$ </td><td> $q = 5$ </td></tr><tr><td>CoraCA</td><td>6,714</td><td>69,204</td><td> $4 6 9 , 1 2 0$ </td><td> $2 , 6 3 1 , 8 4 3$ </td></tr><tr><td>DBLP4k</td><td>3,752,114</td><td>983,644,998</td><td> $2 . 1 9 7 1 \times 1 0 ^ { 1 1 }$ </td><td> $4 . 2 2 9 4 \times 1 0 ^ { 1 3 }$ </td></tr><tr><td>IMDB4k</td><td>18,587</td><td>205,338</td><td> $1 { , } 5 8 1 { , } 9 4 7$ </td><td> $1 0 , 4 9 0 , 6 9 4$ </td></tr><tr><td>Trivago</td><td>410,593</td><td>3,263,206</td><td> $2 3 , 7 4 4 , 1 9 0$ </td><td> $2 0 9 , 4 2 8 , 6 9 1$ </td></tr></table>

To further verify the correspondence-invariant property of the learned representation, we conduct an evaluator-only permutation audit on the released relation columns. A separate CPU float32 audit permutes the 743 released relation columns of CoraCA client 0 at $K = 3$ and protocol seed 0 a total of 32 times. The fixed instance has 892 nodes, 1,499 incidences, a 128-dimensional projected input, hidden width 16, and $R = 4 .$ . It uses neither labels nor training. The maximum differences in $P _ { k } ,$ , all heat-filtered features, and fixed-model logits are each zero, verifying the algebra used by Corollary 4 under this fixed implementation audit with shared numerical inputs.

Table 16 reports evaluator-only audits under relaxed observation assumptions. Type coarsening removes a matching restriction while preserving the released row inventory, and the numeric containment check confirms that the coarsened endpoint hull contains the exact hull in all audited cells. In contrast, status removal and one-sided deletion alter the released inventory, so finite compatible classes require additional public bounds beyond the exact equal-count endpoint formula. Training and model selection are absent from all 120 audit outcomes in this evaluator-only audit.

Table 16: Evaluator-only audit of relaxed observation assumptions on CoraCA and DBLP4k. The 40 cells consist of five fixed partitions and $q \in \{ 2 , 3 , 4 , 5 \}$ for each dataset.
<table><tr><td>Release change</td><td>Cells</td><td>Numeric endpoint</td><td>Audited conclusion</td></tr><tr><td>Merge observed type strata</td><td>40</td><td>Available</td><td>The coarsened endpoint hull contains the ex- act hull in 40/40 cells.</td></tr><tr><td>Remove local/cross marks</td><td>40</td><td>Bound required</td><td>Finite endpoints require a public cross-row count and support bound.</td></tr><tr><td>Delete 10% of one-side fragments</td><td>40</td><td>Bound required</td><td>Finite endpoints require public missing-row count, weight, and support bounds.</td></tr></table>

Figure 3 and Table 17 report the heat-width sweep over $R \in \{ 0 , 1 , 2 , 4 , 8 \}$ on CoraCA, DBLP4k, and Trivago. CoraCA reaches 76.68% Accuracy and 74.35% Macro-F1 at $R = 2 .$ , compared with 70.67% and 67.85% at $R = 0$ . DBLP4k has closely grouped means over � = 1, 2, 4, within 0.09 Accuracy points and 0.10 Macro-F1 points, followed by lower means at $R = 8$ . Trivago reaches its highest Accuracy of 27.66% at $R = 4$ and its highest Macro-F1 of 7.32% at $R = 8$ . The latter change from � = 4 adds 0.30 Macro-F1 points, with parameters increasing from 472,224 to 818,336 and fitting time from 235.3 to 417.9 seconds.

The sweep identifies a dataset-dependent finite-capacity tradeoff. Moderate widths improve the attribute-only control, while additional channels increase parameters and runtime and can change the two classification metrics differently. Spectral density describes approximation by a growing family of heat filters, but finite-sample accuracy also depends on capacity and fitting; therefore, density alone does not impose a monotone ordering on these empirical curves. Each width’s runtime and parameter count follows the same measurement protocol described in Appendix D.4.

![](images/dfffe051c7db80a1e1d064e4495e16ad26f88074ec9a74dc712583898b87afcb.jpg)  
Figure 3: Predictive utility and cost of the heat-basis width. We fix $K = 6$ and $\lambda _ { \mathrm { p r e d } } = 1$ on all three datasets. The � = 0 control retains masked prediction and removes diffusion.

Table 17: Heat-basis width at $K = 6$ and $\lambda _ { \mathrm { p r e d } } = 1$ . Parameter counts include the online model and predictive head. Runtime uses the synchronized fitting boundary described in Appendix D.
<table><tr><td>Dataset</td><td>R</td><td>Accuracy (%)</td><td>Macro-F1 (%)</td><td>Parameters</td><td>Runtime (s)</td></tr><tr><td rowspan="5">CoraCA</td><td>0</td><td> $7 0 . 6 7 \pm 0 . 9 4$ </td><td> $6 7 . 8 5 \pm 1 . 6 4$ </td><td>109,319</td><td> $5 5 . 0 \pm 1 . 7$ </td></tr><tr><td>1</td><td> $7 3 . 9 7 \pm 1 . 1 6$ </td><td> $7 1 . 4 4 \pm 1 . 5 8$ </td><td>126,407</td><td> $1 6 4 . 9 \pm 0 . 3$ </td></tr><tr><td>2</td><td> ${ \bf 7 6 . 6 8 \pm 2 . 2 6 }$ </td><td> $\mathbf { 7 4 . 3 5 \pm 1 . 9 6 }$ </td><td>143,495</td><td> $1 9 9 . 8 \pm 4 . 7$ </td></tr><tr><td>4</td><td> $7 3 . 9 7 \pm 0 . 8 7$ </td><td> $7 1 . 4 3 \pm 1 . 1 2$ </td><td>177,671</td><td> $2 6 4 . 6 \pm 1 . 8$ </td></tr><tr><td>8</td><td> $7 3 . 9 1 \pm 1 . 1 7$ </td><td> $7 1 . 0 9 \pm 1 . 2 4$ </td><td>246,023</td><td> $4 5 1 . 5 \pm 2 . 9$ </td></tr><tr><td rowspan="5">DBLP4k</td><td>0</td><td> $7 8 . 6 6 \pm 0 . 8 7$ </td><td> $7 8 . 1 6 \pm 0 . 8 7$ </td><td>38,788</td><td> $5 4 . 8 \pm 0 . 4$ </td></tr><tr><td>1</td><td> $7 9 . 7 9 \pm 0 . 9 5$ </td><td> $7 9 . 2 3 \pm 1 . 0 1$ </td><td>55,684</td><td> $1 7 3 . 1 \pm 2 . 1$ </td></tr><tr><td>2</td><td> $7 9 . 7 9 \pm 1 . 0 2$ </td><td> $7 9 . 1 8 \pm 0 . 9 8$ </td><td>72,580</td><td> $2 0 8 . 7 \pm 2 . 4 $ </td></tr><tr><td>4</td><td> $\mathbf { 7 9 . 8 7 \pm 1 . 4 1 }$ </td><td> $\mathbf { 7 9 . 2 7 \pm 1 . 3 1 }$ </td><td>106,372</td><td> $2 7 9 . 6 \pm 0 . 3$ </td></tr><tr><td>8</td><td> $7 7 . 7 0 \pm 0 . 7 5$ </td><td> $7 7 . 0 9 \pm 0 . 8 2$ </td><td>173,956</td><td> $4 7 8 . 9 \pm 2 . 8$ </td></tr><tr><td rowspan="5">Trivago</td><td>0</td><td> $2 1 . 8 0 \pm 2 . 1 6$ </td><td> $5 . 2 5 \pm 0 . 2 9$ </td><td>126,112</td><td> $1 5 . 2 \pm 0 . 7$ </td></tr><tr><td>1</td><td> $2 5 . 6 6 \pm 1 . 9 7$ </td><td> $6 . 3 5 \pm 0 . 3 1$ </td><td>212,640</td><td> $1 0 9 . 8 \pm 0 . 1$ </td></tr><tr><td>2</td><td> $2 6 . 9 3 \pm 0 . 4 2$ </td><td> $6 . 8 8 \pm 0 . 4 8$ </td><td>299,168</td><td> $1 5 0 . 8 \pm 0 . 3$ </td></tr><tr><td>4</td><td> $\mathbf { 2 7 . 6 6 \pm 0 . 9 1 }$ </td><td> $7 . 0 1 \pm 0 . 1 3$ </td><td>472,224</td><td> $2 3 5 . 3 \pm 0 . 6$ </td></tr><tr><td>8</td><td> $2 6 . 5 9 \pm 0 . 4 7$ </td><td> ${ \bf 7 . 3 2 \pm 0 . 1 8 }$ </td><td>818,336</td><td> $4 1 7 . 9 \pm 0 . 1$ </td></tr></table>

## E.4 SCALABILITY ON LARGE-SCALE DATA

Table 18 reports the full-scale evaluation on the complete graph assets and official splits. It contains one full-data execution rather than repeated runs. It records predictive quality, communication, wall-clock time, throughput, and peak device allocation with complete root and citation coverage for every compared method. FedASTRA achieves the highest predictive performance in this full-data setting while maintaining parameter-only communication.

Table 18: Results on full-data ogbn-papers100M under the official split. Wall-clock training time covers 20 rounds and is reported in seconds (s). Throughput counts training roots per second. Peak memory is reported in gibibytes (GiB), where $\dot { 1 } \mathrm { G i B } = 2 ^ { \widetilde { 3 } 0 }$ bytes. Communication counts bidirectional trainable-parameter traffic, excluding client-local BatchNorm buffers for all methods.
<table><tr><td>Method</td><td>Parameters</td><td>Parameter traffic per round (MiB)</td><td>Wall-clock (s)</td><td>Roots/s</td><td>Peak GPU (GiB)</td></tr><tr><td>Fed-MLP</td><td>983,380</td><td>60.02</td><td>47.2</td><td>511,594</td><td>0.58</td></tr><tr><td>Fed-SIGN</td><td>984,442</td><td>60.09</td><td>47.6</td><td>506,887</td><td>0.68</td></tr><tr><td>Fed-HGNN</td><td>982,672</td><td>59.98</td><td>54.6</td><td>442,051</td><td>1.52</td></tr><tr><td>Fed-HNHN</td><td>34,672</td><td>2.12</td><td>312.5</td><td>77,259</td><td>0.54</td></tr><tr><td>Fed-HyperGCN</td><td>984,442</td><td>60.09</td><td>406.6</td><td>59,386</td><td>8.89</td></tr><tr><td>Fed-GCN</td><td>983,380</td><td>60.02</td><td>61.8</td><td>390,637</td><td>2.92</td></tr><tr><td>FedASTRA</td><td>976,504</td><td>59.60</td><td>1069.7</td><td>22,570</td><td>19.53</td></tr></table>

All requested neighborhoods are evaluated without truncation.