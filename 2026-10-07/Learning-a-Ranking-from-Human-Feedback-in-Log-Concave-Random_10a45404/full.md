# Learning a Ranking from Human Feedback in Log-Concave Random Utility Models

Diego Alovisetti Politecnico di Milano

Marco Mussi Politecnico di Milano

Alberto Maria Metelli Politecnico di Milano

## Abstract

We study the problem of recovering the ranking of a fixed set of items according to their unknown numerical utilities. At each interaction with the environment, a learner presents the item set to a human and receives comparative feedback of two types. Under full-ranking feedback, each interaction reveals a noisy ranking of all items, whereas under winner-only feedback, it reveals only the item ranked first. In both settings, we model human feedback using a random utility model with log-concave noise and study the number of observations needed to recover an ε-accurate ranking with high probability. This novel criterion tolerates ordering errors only between items whose utilities difer by less than ε. For both feedback types, we establish worst-case samplecomplexity lower bounds and develop algorithms that match these bounds up to logarithmic factors. Neither algorithm requires knowledge of the noise distribution, while only requiring an upper bound on its variance. Our results show that the ranking problem under winner-only feedback is intrinsically harder by exposing the sample complexity dependence on the minimum winning probability across the item set.

## 1 INTRODUCTION

In many machine learning problems, an algorithm deals with a finite set of objects, each associated with a numerical score representing its quality. For example, recommendation systems are concerned with products whose scores reflect users’ preferences (Hajek et al., 2014). Similarly, in Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017), an agent can take multiple actions whose scores are reward values. In both cases, although every object is assigned a scalar quantity, the latter typically remains latent.

Indeed, since the algorithm interacts directly with a human evaluator, such as the user in recommendation systems or the problem designer in RLHF, scalar feedback is either unavailable or unreliable. As the psychometric literature suggests, humans struggle to assign numerical scores, while they can reliably provide ordinal feedback. Consequently, the algorithm must address the set of numerical scores while relying on non-scalar feedback signals. In this work, we formalize this to study a ranking-recovery learning problem. We consider a finite set of items associated with numerical scores that we call utilities and a learning algorithm that has no prior knowledge about them and aims to recover the ranking induced by the utilities. To do so, the algorithm sequentially interacts with a human evaluator who provides (noisy) feedback in the form of a (partial) ranking of the items. To model human behavior, we use a Random Utility Model (RUM), a standard paradigm for human preferences (Thurstone, 1927), which includes the well-studied Plackett-Luce (PL) model (Plackett, 1975; Luce, 1959). Within this framework, we assume that the latent utilities are perturbed by i.i.d. noise to obtain a random utility vector. Then, a feedback filter transforms this vector into the observed feedback signal. We make the additional assumption that the noise distribution is log-concave (e.g., Gaussian, logistic), which is standard in the literature (Shah et al., 2016) and ensures several desirable properties (Critchlow et al., 1991). The feedback filter transforms a vector with scalar entries into a partial ranking. Speficially, we consider two types of filters:

• Full-Ranking (FR), where the complete ordering of the random utilities is observed.

• Winner-Only (WO), in which the learner observes the items with the highest sampled utility.

We evaluate the ranking recovery problem within the Probably Approximately Correct (PAC) framework (Valiant, 1984), where the approximately correct part is formalized using the novel notion of ε-accuracy. The latter, which measures the distance between the recovered ranking and the target, is fundamentally diferent from standard discrete distances such as Kendall tau (Kendall, 1938) or maximum rank (Busa-Fekete et al., 2014b), in that these are insensitive to the underlying utilities. In fact, our metric efectively penalizes misranking two items depending on their utility gap. By considering the proposed ranking problem within the PAC framework, we aim to address the following research question:

What is the minimum number of ordinal feedback signals needed for the learner to recover the ranking induced by the items’ scalar utilities?

By answering this question for two common humangenerated feedback types, our goal is to quantify their intrinsic learning dificulties. This will result in a quantitative understanding of the diference between using full-ranking rather than winner-only feedback. We stress that the tension between the learning goal, rooted in scalar utilities, and the ordinal feedback used to reach it is precisely what makes the proposed ranking problem non-trivial. The notion of ε-accuracy is the very element that allows this tension to be resolved. Our proposed solution to the ranking problem is the Variance-aware Utility Ranker (VUR), a unified framework comprising two algorithms for recovering the target ranking under full-ranking and winner-only signals, respectively. In summary, our main contributions are:

• Novel Distance Metric. We introduce the notion of ε-accuracy through Definition 1, which generalizes the ε-optimality metric used in best-item identification (Saha and Gopalan, 2020) to learning a full utility ranking.

• Unified Framework for Diverse Feedback. We design two PAC algorithms relying on a common design principle, which allows us to view them as part of a single, unified framework for learning the utility ranking across diverse feedback types.

• Distribution-agnostic Algorithms. We propose two algorithms requiring no exact knowledge of the noise distribution. Instead, they rely solely on the log-concavity assumption plus a known upper bound on the noise variance.

• Fundamental Limits of Learnability. We establish information-theoretic lower bounds for the ranking problem under log-concave RUMs, demonstrating that our algorithms are order-optimal. Crucially, Theorem 4 identifies the minimum winning probability, $P _ { \mathrm { m i n } }$ , as the fundamental informational bottleneck of WO feedback.

The related works are discussed in Appendix A.

## 2 PRELIMINARIES

We use standard big-O notation throughout, with $\tilde { \mathcal { O } }$ suppressing logarithmic factors, except for the confidence-dependent term $\log ( 1 / \delta )$ . For any $k \in \mathbb N .$ we denote $[ k ] = \{ 1 , \dots , k \}$ . The set of all permutations on [k], i.e., the bijective functions $\sigma : [ k ]  [ k ]$ , is denoted by $S _ { k }$ . We interpret $\sigma \in S _ { k }$ as the ranking that assigns position $\sigma ( i )$ to item i. The notation $( \Omega , \mathcal { F } , \mathbb { P } )$ refers to a generic probability space, while $\mathbb { 1 } _ { A }$ is the indicator function of any event $A \in { \mathcal { F } }$ . The set of all probability measures on R is denoted by ${ \mathcal { P } } ( \mathbb { R } )$ . If not stated otherwise, the real line is equipped with the Borel sigma-algebra. If $\mu \in { \mathcal { P } } ( \mathbb { R } )$ is absolutely continuous with respect to the Lebesgue measure $\lambda ,$ we write $\mu \ll \lambda$ . If f is the density of $\mu ,$ we denote its support by $\operatorname { S u p p } ( f )$ . For a non-decreasing function $g : \mathbb { R }  \mathbb { R }$ , the generalized inverse is defined by:

$$
g ^ { - 1 } ( y ) = \operatorname* { s u p } \{ x \in \mathbb { R } \colon g ( x ) \leq y \} .
$$

When well-defined, the inverse coincides with the generalized inverse; moreover, the generalized inverse is always non-decreasing.

Linear extensions and graphs. A strict partial order $<$ on the finite set S is a homogeneous binary relation that is irreflexive, asymmetric, and transitive. A general result by Szpilrajn (1930) states that, given any strict partial order < on S, there exists a linear extension $< ^ { * }$ , i.e., a total order such that v $< ^ { * }$ w whenever $v < w$ . A generic directed graph is denoted by $G = ( \nu , \mathcal { E } )$ , where V is the vertex set and E is the edge set. A (directed) edge is of the form $v \longrightarrow w ,$ where v and w are distinct vertices. A tournament is a directed graph with exactly one directed edge between any pair of distinct vertices. When G is acyclic, its transitive closure is the strict partial order $^ { 6 6 } < ^ { 9 9 }$ on the vertex set satisfying: for any two vertices v and $w ,$ $v < w$ if and only if there exists a path in G that goes from v to w.

Log-concavity. Given a probability distribution $\nu \in$ ${ \mathcal { P } } ( \mathbb { R } )$ , we say that ν is log-concave if:

$$
\nu ( \lambda A + ( 1 - \lambda ) B ) \geq \nu ( A ) ^ { \lambda } \nu ( B ) ^ { 1 - \lambda } ,
$$

whenever the sets A and B, and $\lambda \in [ 0 , 1 ]$ are chosen so that $\lambda A + ( 1 - \lambda ) B \subseteq \mathbb { R }$ is measurable. A logconcave distribution on R is either a Dirac measure or absolutely continuous with respect to the Lebesgue measure (Borell, 1975). In this case, its density and cumulative distribution function (CDF) are log-concave (Lovász and Vempala, 2007).

## 3 PROBLEM FORMULATION

We fix a positive integer $k \geq 2$ and consider a collection of $k \in \mathbb N$ items, which we identify with the set $[ k ]$ Each item i is associated with a scalar utility $u _ { i } \in \mathbb { R }$ resulting in the utility vector $\pmb { u } = ( u _ { 1 } , \dots , u _ { k } ) \in \mathbb { R } ^ { k }$ The corresponding utility ranking is defined as the unique permutation $\sigma _ { u } \in S _ { k }$ satisfying:<sup>1</sup>

$$
\sigma _ { u } ( i ) < \sigma _ { u } ( j ) \Longleftrightarrow u _ { i } > u _ { j } \lor ( u _ { i } = u _ { j } \land i < j ) .
$$

In this paper, we address the problem of approximating the utility ranking based on noisy feedback signals generated by a human evaluator. In our formalization, these arise from a RUM with a centered log-concave noise distribution. To avoid degenerate cases, we assume that the latter is absolutely continuous<sup>2</sup>. Logconcavity is a standard assumption in the literature (Shah et al., 2016), and implies several desirable properties for ranking models (Critchlow et al., 1991). To ensure learnability, we impose a suficient boundedvariance condition. For a fixed $V > 0 .$ , we let $\mathcal { N } _ { V }$ be the set of all absolutely continuous centered log-concave distributions ν satisfying:

(V) The variance of distribution ν is bounded above by V .

Intuitively, this can be interpreted as a reliability constraint. While keeping the exact noise distribution unknown, it ensures that the feedback remains a reliable, albeit noisy, reflection of the latent utility relationships. The bound V represents a parameter of the proposed algorithms. For simplicity, we drop the subscript and write ${ \mathcal { N } } ,$ whose elements are referred to as noise distributions.

## 3.1 Feedback-generation Process

Given $\mathbf { \boldsymbol { u } } \in \mathbb { R } ^ { k }$ and $\nu \in \mathcal N$ , we let $N _ { 1 } , \ldots , N _ { k }$ be i.i.d. random variables drawn from $\nu ,$ and consider the random utility vector $U = ( U _ { 1 } , \ldots , U _ { k } )$ given by:

$$
U _ { i } = u _ { i } + N _ { i } , \quad \forall i \in [ k ] .
$$

For $t \in \mathbb { N }$ , we let $U ^ { t } = ( U _ { 1 } ^ { t } , \dots , U _ { k } ^ { t } )$ be an independent copy of $U _ { ☉ }$ , and denote the stochastic process $( U ^ { t } ) _ { t \in \mathbb { N } }$ by $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ , standing for random utility model with utility u and noise distribution ν. The feedback signal observed at time t is obtained by applying a feedback filter to the utility vector $U ^ { t }$ . Specifically, we consider two types of filters.

<sup>1</sup>If $u _ { i } = u _ { j } ,$ , the tie is broken by ordering the two items according to their indices.

<sup>2</sup>Because a log-concave distribution is either a Dirac measure or abs. continuous, this amounts to excluding the case in which the noise has a Dirac distribution. This results in no loss of generality as the support of the noise can be taken arbitrarily narrow.

• The full-ranking (FR) filter is the function $\varphi _ { F R } :$ $\mathbb { R } ^ { k } \to S _ { k }$ such that $\varphi _ { F R } ( { \pmb x } )$ is the unique permutation σ defined by: $\sigma ( i ) < \sigma ( j )$ if and only if $x _ { i } > x _ { j }$ or $x _ { i } = x _ { j }$ and $i < j$

• The winner-only (WO) filter is the function $\varphi _ { W O } :$ $\mathbb { R } ^ { k } \to [ k ]$ given by $\varphi _ { W O } ( { \pmb x } ) : = \operatorname* { m i n } \{ i \in [ k ] : x _ { i } \geq$ $x _ { j } \forall j \neq i \} . ^ { 3 }$

We assume that the feedback filter is fixed throughout the learning episode. Thus, given a feedback filter $\varphi \in \{ \varphi _ { F R } , \varphi _ { W O } \}$ , we let $\varphi _ { t } = \varphi ( U ^ { t } )$ for every $t \in \mathbb N$ The feedback filtration is defined as $\mathbb { F } = ( \mathcal { F } _ { t } ) _ { t \in \mathbb { N } }$ , where $\mathcal { F } _ { t }$ is the σ-algebra generated by $\varphi _ { 1 } , \ldots , \varphi _ { t } .$

## 3.2 The PAC Objective

Oftentimes, estimation of the utility ranking comes as a byproduct of utility estimation (Negahban et al., 2012; Azari Soufiani et al., 2014). In contrast, our approach focuses on the ranking itself, thus avoiding the need for explicit utility reconstruction. We achieve this by defining the objective in terms of a novel metric for measuring the distance between two rankings, which efectively relates a misranked pair (i, j) to the latent utility gap $\Delta _ { i j } : = u _ { i } - u _ { j }$

Definition 1 (ε-accuracy). Let $\mathbf { \boldsymbol { u } } \in \mathbb { R } ^ { k }$ with utility ranking $\sigma _ { u } ,$ and let $\varepsilon > 0 .$ . Given $\sigma \in S _ { k }$ , we say that σ is an ε-accurate representation of $\sigma _ { u }$ if the following holds for any two distinct i and $j \colon$

$$
( \sigma ( i ) - \sigma ( j ) ) ( \sigma _ { u } ( i ) - \sigma _ { u } ( j ) ) < 0 \implies | \Delta _ { i , j } | < \varepsilon .\tag{1}
$$

We remark that the left-hand side of the implication is equivalent to saying that the relative position of items i and $j$ in $\sigma$ is opposite to the one induced by $\sigma _ { u }$ . In (Saha and Gopalan, 2019), a similar notion is used, where the gap between the PL parameters plays the role of the utility gap in ε-accuracy. In their work, Falahatgar et al. (2018) also propose a metric related to ours; however, their setting is fundamentally diferent as no utilities are required for the ranking distribution to be defined. We elaborate on these connections in Appendix A.

To define the PAC objective, we first define an algorithm as a pair $\boldsymbol { \mathcal { A } } = ( T , \hat { \boldsymbol { \sigma } } )$ , where $T \in \mathbb { N }$ is an almost-surely finite stopping time with respect to the feedback filtration F, and σˆ is an $\mathcal { F } _ { T }$ -measurable random permutation of [k].

Definition 2 (PAC objective). Let $A = ( T , \hat { \sigma } )$ be an algorithm that learns a ranking based on the sequential observations $\varphi _ { t }$ generated by a fixed unknown $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ with feedback filter $\varphi .$ . Given $\delta \in ( 0 , 1 )$ and $\varepsilon > 0$ , we say that A is $( \varepsilon , \delta ) \ – P A C$ if:

P [ˆσ is an ε-accurate representation of $\sigma _ { u } ] \ge 1 - \delta .$

If A is an $( \varepsilon , \delta ) – \mathrm { P A C }$ algorithm, we define its sample complexity as the expected value E $[ T ]$ , where the expectation is taken with respect to the probability measure induced by $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$

## 4 FULL-RANKING FEEDBACK

In this section, we study the ranking problem under fullranking feedback $\left( \varphi = \varphi _ { \mathrm { F R } } \right)$ . Specifically, we introduce an $( \varepsilon , \delta ) – \mathrm { P A C }$ algorithm named Full-Ranking Varianceaware Utility Ranker (FR-VUR) and derive an upper bound on its sample complexity. Subsequently, we provide an information-theoretic lower bound, showing that FR-VUR achieves an optimal rate, up to logarithmic dependence. The feedback-generating random utility model $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ is fixed throughout, where the noise distribution ν satisfies Assumption (V).

## 4.1 The Pairwise-Preference Probabilities

Under FR feedback, every observation can be converted into $\binom { k } { 2 }$ pairwise-comparison outcomes by rankbreaking (Azari Soufiani et al., 2014). Our algorithm will do so, using these values to estimate, for any two distinct items i and $j ,$ , the pairwise-preference probability:

$$
P _ { i j } : = \mathbb { P } \left[ U _ { i } > U _ { j } \right] .
$$

The mutual independence of the noise random variables implies that $P _ { i j } > 1 / 2$ if and only if $u _ { i } ~ > ~ u _ { j }$ (see Lemma 5 in Appendix C). For every $t \in \mathbb { N }$ , we define the event $E _ { i j } ^ { t } : = \{ \varphi _ { t } ( i ) < \varphi _ { t } ( j ) \}$ , the set of outcomes for which item i is ranked before item j at time t. Then, for every integer $n \geq 2$ , we consider the unbiased estimator of $P _ { i j }$ given by:

$$
P _ { i j } ^ { n } : = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { 1 } _ { E _ { i j } ^ { t } } .
$$

The (unbiased) sample variance is the quantity:

$$
V _ { i j } ^ { n } : = \frac { 1 } { n ( n - 1 ) } \sum _ { 1 \le t _ { 1 } < t _ { 2 } \le n } ( \mathbb { 1 } _ { E _ { i j } ^ { t _ { 1 } } } - \mathbb { 1 } _ { E _ { i j } ^ { t _ { 2 } } } ) ^ { 2 } .
$$

For $\delta \in ( 0 , 1 )$ and $n \in \mathbb { N } ,$ we define the full-ranking time-n confidence as $\delta _ { n } : = 6 \delta / k ( k - 1 ) n ^ { 2 } \pi ^ { 2 }$ . Following (Maurer and Pontil, 2009), we derive concentration bounds for $P _ { i j } ^ { n }$ and $V _ { i j } ^ { n }$ in Appendix C.2 (see Lemma $6 )$ . These include a standard concentration inequality for the sample variance and an empirical Bernstein-type inequality for $P _ { i j } ^ { n }$

## 4.2 The Full-Ranking Resolution Factor

A priori, it is not clear whether the pairwise-preference probabilities are the right estimation target to achieve ε-accuracy. Below, we show that this is indeed the case by formalizing the intuition that, when $u _ { i } - u _ { j } \geq \varepsilon$ the quantity $P _ { i j }$ is large enough to reveal the true preference relation between i and $j .$ Consequently, by accurate approximating $P _ { i j }$ , the sign of $u _ { i } \ : - \ : u _ { j }$ can be inferred. To make this precise, we derive a mapping between utility gaps and pairwise-preference probabilities. Let $i , j \in [ k ]$ be two items with $u _ { i } > u _ { j }$ and let $\xi : [ 0 , \infty )  [ 1 / 2 , \bar { 1 } ]$ be the function given by:

$$
\xi ( t ) : = \int _ { \mathbb R } F ( x + t ) f ( x ) \mathrm { d } x ,\tag{2}
$$

where $F$ and $f$ are the CDF and the density of $\nu ,$ respectively. Then, we write:

$$
P _ { i j } = \xi ( u _ { i } - u _ { j } ) = \frac { \eta ( u _ { i } - u _ { j } ) } { 2 } + \frac { 1 } { 2 } ,\tag{3}
$$

where we defined the full-ranking resolution factor as $\eta ( \varepsilon ) : = 2 \xi ( \varepsilon ) - 1$ , for $\varepsilon > 0$ . Intuitively, as Equation (3) suggests, this quantifies how strongly an item i with $u _ { i } = u _ { j } + \varepsilon$ is preferable to item $j .$ . In Appendix C.3, we show that $\eta ( \varepsilon ) \geq \eta ^ { \star } ( \varepsilon )$ for every $\varepsilon > 0$ , where:

$$
\eta ^ { \star } ( \varepsilon ) : = \operatorname* { m i n } \left\{ \frac { 2 } { 3 } , \frac { \varepsilon } { \sqrt { 6 V } } \right\} ,\tag{4}
$$

This fact is crucial to guarantee that the FR-VUR algorithm is agnostic to the noise distribution.

## 4.3 The FR-VUR Algorithm

We fix $\delta \in ( 0 , 1 )$ and let $\delta _ { n }$ be the time-n confidence defined in the previous section. For a fixed accuracy $\varepsilon > 0 .$ we write $\eta ^ { \star }$ to denote the lower bound $\eta ^ { \star } ( \varepsilon )$ given by Equation (4). To formally describe FR-VUR and show that it is $( \varepsilon , \delta ) – \mathrm { P A C }$ , we will refer to it as $\boldsymbol { \mathcal { A } } = ( T , \hat { \boldsymbol { \sigma } } )$ . Pseudocode is given in Appendix B.2.

Stopping rule. The stopping time T is defined as the minimum positive integer $n \in \mathbb { N }$ such that the following holds for any two distinct $i , j \in [ k ]$ :

$$
\sqrt { \frac { 2 V _ { i j } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } < \frac { \eta ^ { \star } } { 4 } .\tag{5}
$$

To compute the right-hand side, $\mathcal { A }$ uses $\eta ^ { \star }$ , so it needs to know the variance upper bound V. However, it does not require exact knowledge of the noise distribution.

Using the concentration properties $P _ { i j } ^ { n }$ and $V _ { i j } ^ { n }$ , the stopping condition ensures that, with high probability:

$$
| P _ { i j } ^ { T } - P _ { i j } | < \eta ^ { \star } / 4 .
$$

We can think of it as a multiplicative requirement<sup>4</sup> by observing that max $( P _ { i j } ^ { T } , P _ { j i } ^ { T } ) \overset { \cdot } { = } 1 / 2$ . This implies:

$$
| P _ { i j } ^ { T } - P _ { i j } | = | P _ { j i } ^ { T } - P _ { j i } | \leq \operatorname* { m a x } ( P _ { i j } ^ { T } , P _ { j i } ^ { T } ) \cdot \frac { \eta ^ { \star } } { 2 } .
$$

To define the output of ${ \mathcal { A } } ,$ for every $n \in \mathbb { N } .$ , we define the pairwise-preference graph $G _ { n } \ = \ ( [ k ] , \mathcal { E } _ { n } )$ as the directed graph where:

$$
i \longrightarrow j \in \mathcal { E } _ { n } \Longleftrightarrow P _ { i j } ^ { n } > \frac { 1 } { 2 } + \frac { \eta ^ { \star } } { 4 } .
$$

Additionally, we let the utility graph be transitive tournament $G _ { u } = ( [ k ] , \mathcal { E } _ { u } )$ whose edge set is characterized as follows: $i \longrightarrow j \in { \mathcal { E } } _ { u }$ if and only if either $u _ { i } > u _ { j }$ or $u _ { i } = u _ { j }$ and $i < j$

Output recommendation. The $\mathcal { F } _ { T }$ -measurable random ranking $\hat { \sigma }$ is determined as follows:

• If $G _ { T }$ contains a cycle, choose $\hat { \sigma } \in S _ { k }$ arbitrarily.

• If $G _ { T }$ is $\mathrm { a \ D A G }$ , let $G ^ { * }$ be its transitive closure, corresponding to a strict partial order of its vertices. Then, let $\hat { \sigma } \in S _ { k }$ be any linear extension of this partial order.

In particular, when $G _ { T }$ contains no cycle, $i \longrightarrow j \in \mathcal { E } _ { T }$ implies $\hat { \sigma } ( i ) < \hat { \sigma } ( j )$

We show that $\boldsymbol { \mathcal { A } } = ( T , \hat { \sigma } )$ satisfies the PAC property given by Definition 2. To this end, we first observe that, with probability at least $1 - \delta , G _ { T }$ is a subgraph of $G _ { u } .$ This follows from the concentration properties of $P _ { i j } ^ { n }$ and $V _ { i j } ^ { n }$ and is proven by Lemma 8 in Appendix C.4. Second, we note that the ε-accuracy target is met if, for any two items i and j with $u _ { i } - u _ { j } \geq \varepsilon ,$ we have $\hat { \sigma } ( i ) < \hat { \sigma } ( j )$ . By definition, this is the case if $i \longrightarrow j$ is an edge of $G _ { T }$ . As demonstrated by Lemma 9 in Appendix $\mathrm { C . 4 }$ , this occurs with probability at least $1 - \delta$ , thus proving that A is $( \varepsilon , \delta ) – \mathrm { P A C }$

The next result provides an upper bound on the sample complexity of the proposed algorithm. The full derivation is provided in Appendix C.5.

Theorem 1 (Sample complexity upper bound for FR-VUR). Let $\textbf { \em u } \in \mathbb { R } ^ { k }$ and $\nu \in \mathcal N$ be such that $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ satisfies Assumption (V), and let A be the FR-VUR algorithm. Define $Q : = \operatorname* { m a x } _ { i \neq j } P _ { i j } ( 1 - P _ { i j } )$ Then, for every $\delta \in ( 0 , 1 )$ and $\varepsilon > 0 ,$ , A is $( \varepsilon , \delta ) \mathrm { \mathrm { - P A C } }$ with sample complexity:

$$
\mathbb { E } \left[ T \right] = \mathcal { O } \left( \frac { Q } { \eta ^ { \star 2 } } \log \Bigl ( \frac { Q k } { \eta ^ { \star 2 } \delta } \Bigr ) + \frac { 1 } { \eta ^ { \star } } \log \Bigl ( \frac { k } { \eta ^ { \star } \delta } \Bigr ) \right) .\tag{6}
$$

When $Q$ is non-zero, the upper bound is $\tilde { \mathcal { O } } ( Q / \eta ^ { \star 2 } )$ The term Q follows from the empirical Bernstein-type stopping rule, which adapts to the intrinsic variance of the pairwise comparisons and accelerates convergence when outcomes are highly skewed. To illustrate this advantage, consider the extreme case with distinct utilities and a vanishing noise variance, implying $Q = 0$ In this scenario, the sample complexity reduces to $\tilde { \mathcal { O } } ( 1 / \eta ^ { \star } )$ . As V goes to zero, thus matching the true variance, $\eta ^ { \star }$ saturates at $2 / 3 ,$ , and this residual term reduces to the purely logarithmic cost ${ \tilde { \mathcal { O } } } ( \log ( k / \delta ) )$ In contrast, a Hoefding-based approach would rely on the global variance upper bound $Q \leq 1 / 4$ . For $\sqrt { V } \geq 3 \varepsilon / ( 2 \sqrt { 6 } )$ , so that $\eta ^ { \star } = \varepsilon / ( 2 \sqrt { 6 V } )$ , this would result in the worst-case sample complexity:

$$
\mathbb { E } \left[ T \right] = \mathcal { O } \left( \frac { V } { \varepsilon ^ { 2 } } \log \left( \frac { V } { \varepsilon ^ { 2 } } \cdot \frac { k } { \delta } \right) \right) .\tag{7}
$$

For a fixed $V \ > \ 0 .$ , the right-hand side becomes $\tilde { \mathcal { O } } ( V / \varepsilon ^ { 2 } )$ , where the term $1 / \varepsilon ^ { 2 }$ reflects the ε-accuracy objective. The V scaling captures the intuition that, if the noise variance increases, the observed rankings become less reliable. As we will establish in Theorem 2, the $V / \varepsilon ^ { 2 }$ dependence is information-theoretically optimal.

## 4.4 Information-theoretic Lower Bound

In this ${ \mathrm { p a r t } } ,$ we demonstrate an information-theoretic worst-case lower bound matching the upper bound given in Equation (7). Precisely, we address the smallgap regime $\varepsilon < \sqrt { 6 V } / ( 2 \pi )$ , which is the most interesting case.<sup>5</sup> To do so, we identify a class of “hard” problem instances, defined by a Gumbel noise paired with a utility vector where one utility gap is exactly ε. Our proof, provided in Appendix $\mathrm { C . 6 , }$ employs a change-ofmeasure argument following (Kaufmann et al., 2016). Below, $\mathcal { A } ^ { \ast } = \left( T ^ { \star } , \hat { \sigma } ^ { \star } \right)$ is any ranking algorithm based on FR feedback signals. We assume $\ b { A } ^ { * }$ is $( \varepsilon , \delta ) – \mathrm { P A C }$ for any $\delta \in ( 0 , 1 )$

Theorem 2 (Lower bound for full-ranking feedback). Let $\delta \in ( 0 , 1 / 4 ]$ Then, there exists a noise-utility pair $( \nu , \pmb { u } )$ satisfying Assumption (V) such that the sample complexity of algorithm $\mathcal { A } ^ { \ast } = \left( T ^ { \star } , \hat { \sigma } ^ { \star } \right)$ satisfies:

$$
\mathbb { E } \left[ T ^ { \star } \right] = \Omega \left( \frac { V } { \varepsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) ,
$$

where the expectation is taken with respect to the probability measure induced by $R U M ( { \pmb u } , \nu )$

This result establishes the $V / \varepsilon ^ { 2 }$ scaling as a fundamental learning cost of the ranking problem. In particular,

Theorem 2 shows that the exact dependence on the noise variance is linear.

## 5 WINNER-ONLY FEEDBACK

We now study the ranking recovery problem under WO feedback, meaning $\varphi = \varphi _ { W O }$ . For this setting, we design and analyze yet another learning algorithm that difers from FR-VUR in that it estimates winning probabilities rather than pairwise-preference probabilities. Nonetheless, both algorithms rely on a (feedbackspecific) resolution factor to relate the estimated probabilities to underlying utility gaps. To emphasize this continuity, we overwrite the symbols $\eta , \eta ^ { \star }$ , and A. As in the full-ranking case, $U ^ { t }$ is the random utility vector drawn at time t from a fixed $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ , where ν satisfies Assumption (V). Additionally, we let $k \geq 3 .$ , as the winner-only setting reduces to full-ranking when only two items are available.

## 5.1 Observability Assumption

A full-ranking signal provides information about the relative position of any two items, which was exploited by FR-VUR through rank-breaking. In contrast, under winner-only feedback, information about the relative position of two items is obtained only if one of them wins. Consequently, to correctly rank these, the learner must rely on their winning frequency. Theorem 4 below quantifies this by providing an information-theoretic lower bound, demonstrating that the sample complexity of any PAC algorithm is inversely proportional to the minimum winning probability. Therefore, to avoid ending in a non-learnable regime, we enforce an additional assumption. To this end, we begin by defining the utility range:

$$
\Delta _ { u } : = \operatorname* { m a x } _ { i \in [ k ] } u _ { i } - \operatorname* { m i n } _ { j \in [ k ] } u _ { j } .
$$

Then, denoting by $f$ the density of $\nu ,$ we let $a : =$ inf $\operatorname { S u p p } ( f )$ and $b : = \operatorname* { s u p } \operatorname { S u p p } ( f )$ , and call $b - a$ the width of the noise support. To ensure that every item have a non-zero winning probability, we impose the following condition:

(WO1) The utility range is smaller than the width of the support of $\nu , \mathrm { i . e . , } \Delta _ { u } < b - a$

Under this assumption, $P _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { i \in [ k ] } P _ { i } > 0$ , meaning that, as the episode horizon $T$ grows to infinity, every item will eventually win in at least one round with non-zero probability. With this in mind, we can interpret (WO1) as an observability assumption. We anticipate that, while taking the variance upper bound V as input, our algorithm does not require knowledge of either a or b.

## 5.2 The Winner-Only Resolution Factor

The probability that an item is the winner plays a fundamental role in this setting. Formally, for every $i \in [ k ]$ , we define its winning probability as:

$$
P _ { i } : = \mathbb { P } \left[ U _ { i } > U _ { j } , \forall j \in [ k ] \setminus \{ i \} \right] .\tag{8}
$$

For any two items $i , j \in [ k ]$ with $u _ { i } \geq u _ { j }$ , their winning probabilities are consistent, in the sense that $P _ { i } > P _ { j }$ if and only if $u _ { i } > u _ { j }$ (see Lemma 15 in Appendix D.3). Since the winning probabilities $P _ { i }$ and $P _ { j }$ depend on the magnitude of all utility gaps, it is unclear whether they can be used to infer the magnitude of $\Delta _ { i j }$ . Remarkably, this turns out to be the case in the (generalized) PL model, a RUM with Gumbel noise with scale parameter $s > 0$ , where:<sup>6</sup>

$$
u _ { i } - u _ { j } = s \log \left( \frac { P _ { i } } { P _ { j } } \right) .\tag{9}
$$

Saha and Gopalan (2019) exploited this and used estimates of the winning-probability ratios to approximate the ranking induced by the PL parameters. In this work, we generalize their approach to a broader class of RUMs, while addressing the latent utilities instead of the PL parameters. We do so by establishing a functional relation between winning-probability ratios and utility gaps under the suficient condition of logconcavity. To illustrate this, we consider i and $j$ such that $u _ { i } > u _ { j }$ , and observe that:

$$
P _ { i } - P _ { j } \geq P _ { i } \cdot \frac { 1 - e ^ { - ( u _ { i } - u _ { j } ) / \sqrt { 3 V } } } { 2 } .\tag{10}
$$

This fact is proven in Lemma 13 in Appendix D.2 using log-concavity. Now we define the function (1, 2)-valued function ξ by setting:

$$
\xi ( t ) : = \frac { 2 } { 1 + e ^ { - t / \sqrt { 3 V } } } , \quad \forall t \in ( 0 , b - a ) .\tag{11}
$$

Assumption (WO1) ensures $P _ { j } > 0$ , so we can rearrange the terms of Equation (10) to obtain $P _ { i } / P _ { j } \ge \xi ( u _ { i } - u _ { j } )$ Since $\xi$ is strictly increasing, taking its generalized inverse yields:

$$
u _ { i } - u _ { j } \leq \xi ^ { - 1 } \left( \frac { P _ { i } } { P _ { j } } \right) .\tag{12}
$$

Following the approach used in Section 4, we define the winner-only resolution factor $\eta : ( 0 , b - a )  ( 0 , 1 / 3 )$ as:

$$
\eta ( t ) : = \frac { \xi ( t ) - 1 } { \xi ( t ) + 1 } = \frac { 1 - e ^ { - t / \sqrt { 3 V } } } { 3 + e ^ { - t / \sqrt { 3 V } } } .\tag{13}
$$

<sup>6</sup>This property is typically referred to as Independence of Irrelevant Alternatives (IIA) (Ray, 1973).

The latter is defined using the known variance bound V and is independent of the particular noise distribution in $\mathcal { N } _ { V }$ . For a simpler expression with the same scaling in the $\varepsilon \approx 0$ regime, we let:

$$
\eta ^ { \star } ( \varepsilon ) : = \frac { e - 1 } { 3 e + 1 } \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon } { \sqrt { 3 V } } \right\} .\tag{14}
$$

Then, $\eta ( \varepsilon ) \geq \eta ^ { \star } ( \varepsilon )$ for every $\varepsilon \in ( 0 , b - a )$ , which is proven using standard analytical tools in Lemma 14 in Appendix D.2.

## 5.3 The Winning-Probability Estimators

For each item $i \in [ k ]$ and every time index $t \in \mathbb N$ we define the event $E _ { i } ^ { t } : = \{ \varphi _ { t } = i \}$ , consisting of all outcomes for which i is the winner at time t. Then, for $n \in \mathbb { N }$ , we let:

$$
P _ { i } ^ { n } : = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { 1 } _ { E _ { i } ^ { t } } ,
$$

which is an unbiased estimator for the winning probability $P _ { i }$ . For $n \geq 2$ , we define the (unbiased) sample variance as:

$$
V _ { i } ^ { n } : = \frac { 1 } { n ( n - 1 ) } \sum _ { 1 \leq t _ { 1 } < t _ { 2 } \leq n } ( \mathbb { 1 } _ { E _ { i } ^ { t _ { 1 } } } - \mathbb { 1 } _ { E _ { i } ^ { t _ { 2 } } } ) ^ { 2 } .
$$

Given a fixed $\delta \in ( 0 , 1 )$ , we tune the winner-only time-n confidence to be $\delta _ { n } : = 3 \delta / \pi ^ { 2 } n ^ { 2 } k$ , for all $n \in \mathbb { N } .$ . Concentration inequalities for these estimators are given by Lemma 11 in Appendix D.1 based on (Maurer and Pontil, 2009).

## 5.4 The WO-VUR Algorithm

Given $\delta \in ( 0 , 1 )$ and $\varepsilon \in ( 0 , b - a )$ we let $\delta _ { n }$ be the winner-only time-n confidence defined in Section $5 . 3 ,$ while we write $\eta ^ { \star }$ for the lower bound $\eta ^ { \star } ( \varepsilon )$ given by Equation (14). Below, we introduce the $( \varepsilon , \delta ) – \mathrm { P A C }$ algorithm WO-VUR $\mathcal { A } = ( T , \hat { \sigma } )$ , where the notation overload emphasizes the continuity with the full-ranking setting. This algorithm uses empirical winning frequencies to deliver its final recommendation. The resulting sample-complexity guarantee holds for any noise-utility pair $( \nu , \pmb { u } )$ such that $\nu \in \mathcal { N } _ { V }$ and assumption (WO1) holds. Knowledge of V is required, although the algorithm does not require the noise density or its support. Pseudocode is given in Appendix B.2.

Stopping rule. The algorithm stops at time T, which is defined as the minimum integer n $\geq 2$ such that the following holds for every $i \in [ k ]$

$$
\sqrt { \frac { 2 V _ { i } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } < \frac { \eta ^ { \star } } { 2 } \cdot P _ { i } ^ { n } .\tag{15}
$$

Given the estimates $P _ { 1 } ^ { n } , \ldots , P _ { k } ^ { n }$ at time $n ,$ we let $\sigma _ { n }$ be the unique permutation of $[ k ]$ satisfying:

$$
\sigma _ { n } ( i ) < \sigma _ { n } ( j ) \Longleftrightarrow P _ { i } ^ { n } > P _ { j } ^ { n } \lor ( P _ { i } ^ { n } = P _ { j } ^ { n } \mathrm { a n d } i < j ) .
$$

Output recommendation. The recommendation is defined as the $\mathcal { F } _ { T }$ -measurable permutation $\hat { \sigma } : = \sigma _ { T }$

To show that $\mathcal { A }$ has the PAC property given by Definition 2, we suppose that, for every $i \in [ k ]$

$$
| P _ { i } ^ { T } - P _ { i } | \leq \sqrt { \frac { 2 V _ { i } ^ { T } \log ( 4 / \delta _ { T } ) } { T } } + \frac { 7 \log ( 4 / \delta _ { T } ) } { 3 ( T - 1 ) } .
$$

As shown by Lemma 11 in Appendix D.1, this is true with probability at least $1 - \delta$ . Using the definition of $T$ and the fact that $\eta ^ { \star } < 1$ , the following holds for every item i:

$$
| P _ { i } ^ { T } - P _ { i } | < P _ { i } ^ { T } / 2 .
$$

This highlights the multiplicative nature of the bound. Now, suppose that i and $j$ are misranked at time $T ,$ meaning $( P _ { i } - P _ { j } ) ( P _ { i } ^ { T } - P _ { j } ^ { T } ) \leq 0$ . Without loss of generality, we may assume $u _ { i } > u _ { j }$ , thus:

$$
0 < P _ { i } - P _ { j } < \frac { \eta ^ { \star } } { 2 } \left( P _ { i } ^ { T } + P _ { j } ^ { T } \right) \leq \eta ^ { \star } \cdot ( P _ { i } + P _ { j } ) .
$$

Rearranging these terms yields:

$$
\frac { P _ { i } } { P _ { j } } < \frac { 1 + \eta ^ { \star } } { 1 - \eta ^ { \star } } \leq \frac { 1 + \eta ( \varepsilon ) } { 1 - \eta ( \varepsilon ) } = \xi ( \varepsilon ) ,
$$

where $\eta ( \varepsilon )$ is the feedback-specific resolution factor defined by Equation (13). Our earlier calculations give $\xi ( u _ { i } - u _ { j } ) \le P _ { i } / P _ { j }$ , thus implying $\xi ( u _ { i } - u _ { j } ) < \xi ( \varepsilon )$ Since $\xi$ is an increasing function, we conclude that $u _ { i } - u _ { j } < \varepsilon .$ . In other words, we have shown that: with probability at least $1 - \delta$ , if i and $j$ are misranked, then their absolute utility gap is bounded by ε. Hence, algorithm A is (ε, δ)-PAC.

## 5.4.1 Sample Complexity Upper Bound

The next result derives an upper bound on the sample complexity of the WO-VUR algorithm. Formal derivations are given in Appendix D.4.

Theorem 3 (Sample complexity upper bound for WO-VUR). Let $\boldsymbol { u } \in \mathbb { R } ^ { k }$ and $\nu \in \mathcal { N } _ { V }$ be a noise-utility pair satsfying Assumption (WO1), and let $\mathcal { A } = ( T , \hat { \sigma } )$ be the WO-VUR algorithm. Then, for every $\delta \in ( 0 , 1 )$ and $\varepsilon \in ( 0 , b - a )$ , A is $( \varepsilon , \delta )$ -PAC with sample complexity:

$$
\mathbb { E } \left[ T \right] = \mathcal { O } \left( \frac { 1 } { P _ { \operatorname* { m i n } } \eta ^ { \star 2 } } \log { \left( \frac { 1 } { P _ { \operatorname* { m i n } } \eta ^ { \star 2 } } \cdot \frac { k } { \delta } \right) } \right) .\tag{16}
$$

Using the explicit formula (14), in the small-gap regime $\varepsilon \le \sqrt { 3 V }$ , the sample-complexity upper bound takes the form:

$$
\mathbb { E } \left[ T \right] = \mathcal { O } \left( \frac { V } { P _ { \mathrm { m i n } } \varepsilon ^ { 2 } } \log { \left( \frac { V } { P _ { \mathrm { m i n } } \varepsilon ^ { 2 } } \cdot \frac { k } { \delta } \right) } \right) .\tag{17}
$$

As we highlighted through our informal discussion, the term $P _ { \mathrm { m i n } }$ exposes a fundamental learning limit. Since the relative position of i and $j$ is estimated based on their empirical winning frequencies, if these items rarely win, reliable estimation will require a large number of samples. Crucially, the (inverse) linear dependence on $P _ { \mathrm { m i n } }$ is a consequence of the Bernstein-type stopping rule, whereas using a Hoefding-based approach would have resulted in a suboptimal quadratic dependence. We make this explicit through Equation (41) in Appendix D.4. The $\bar { 1 } / \varepsilon ^ { 2 }$ scaling given in the upper bound reflects the ε-accuracy target and mirrors full-ranking case. Similarly, the V term confirms the linear dependence on the reliability of the feedback signals. Finally, the quantity log(k) results in a cost of lower order. As we shall see, Theorem 4 establishes the $V / ( \varepsilon ^ { 2 } P _ { \mathrm { m i n } } )$ scaling as information-theoretically optimal.

## 5.5 Information-theoretic Lower Bound

The subsequent lower-bound analysis ultimately demonstrates that learning becomes unattainable when $P _ { \mathrm { m i n } }$ goes to zero. We specifically analyze sample complexity over all RUMs with minimum winning probability $P _ { \mathrm { m i n } } \in ( 0 , 1 / k )$ , excluding the case $P _ { \mathrm { m i n } } = 1 / k$ , which corresponds to the trivial instance where all utilities are equal. To formalize our result, we let $\mathcal { A } ^ { \ast } = \left( T ^ { \star } , \hat { \sigma } ^ { \star } \right)$ be a PAC algorithm that outputs a ranking based on winner-only feedback. Full derivation of the lower bound is provided in Appendix D.5.

Theorem 4 (Lower bound for winner-only feedback). Let $k \ge 3 , P _ { \operatorname* { m i n } } \in ( 0 , 1 / k )$ , and $\delta \in ( 0 , 1 / 4 ]$ be fixed. Then, there exists a noise-utility pair $( \nu , \pmb { u } )$ satisfying Assumptions (V) and (WO1) such that the sample complexity of algorithm $\mathcal { A } ^ { \ast } = \left( T ^ { \star } , \hat { \sigma } ^ { \star } \right)$ satisfies:

$$
\mathbb { E } \left[ T ^ { \star } \right] = \Omega \left( \frac { V } { { P _ { \mathrm { m i n } } \varepsilon ^ { 2 } } } \log \left( \frac { 1 } { \delta } \right) \right) ,
$$

where the expectation is taken with respect to the probability measure induced by $\operatorname { R U M } ( \nu , \pmb { u } )$

We begin by observing that our lower bound contrasts with the $\tilde { \mathcal { O } } ( k / \varepsilon ^ { 2 } )$ rate claimed by Saha and Gopalan (2019) under the PL model. The discrepancy is due to a fundamental objective misalignment. Indeed, their framework seeks an ε-optimal ranking, where misclassification is penalized based on the diference between the PL parameters $e ^ { u _ { \ell } }$ . If i and $j$ have utilities $u _ { i } , u _ { j } \ll 0 \nonumber$ then $e ^ { u _ { i } } , e ^ { u _ { j } } \approx 0$ . Consequently, under the ε-optimality metric, misranking these two is tolerable. Therefore, by choosing to use this metric, the true informational bottleneck of $P _ { \mathrm { m i n } }$ established in Theorem 4 is deliberately circumenvented. Hence, the claim of full ranking recovery follows without actually resolving the relative order of items with a low winning probability. In contrast, our analysis efectively exposes the dificulties of learning the utility ranking under winner-only feedback, while treating every item pair as equally important. To do so, we chose the ε-accuracy metric to depend on the underlying utilities rather than the PL parameters.

## 6 OPEN CHALLENGES

We characterized the learning dificulty of the utilityranking recovery problem under two types of RUMgenerated feedback. However, full-ranking and winneronly are the extremes of a wide spectrum. Therefore, understanding how the learning task changes across the whole range of feedback types in between (e.g., top-m (Hajek et al., 2014)) is an open question. Second, we relied on log-concavity to derive the utility-to-probability mappings used by both our VUR algorithms. Without this assumption, the feedback-specific resolution factors can no longer be defined. Understanding whether the same approach can be employed under a milder condition on the noise distribution is not understood yet. Third, a more complex interaction protocol may be considerd. In our work, we focused on a setting where the feedback signal regards all items at every learner-environment interaction because our goal was to quantify the learning dificulty due to using winneronly vs. full-ranking feedback. However, allowing for subset selection (Saha and Gopalan, 2019) would accommodate more realistic types of interaction. Finally, our ε-accuracy metric efectively links ranking information to the utilities of the ranked items. We believe that in this lies a potential tool for addressing the regret minimization problem in the context of multi-armed bandits with ranking feedback.

## References

Hossein Azari Soufiani, William Z. Chen, David C. Parkes, and Lirong Xia. Generalized method-ofmoments for rank aggregation. In Advances in Neural Information Processing Systems (NIPS), volume 26, pages 2706–2714, 2013.

Hossein Azari Soufiani, David C. Parkes, and Lirong Xia. Computing parametric ranking models via rankbreaking. In Proceedings of the International Conference on Machine Learning (ICML), volume 32 of Proceedings of Machine Learning Research, pages 360–368. PMLR, 2014.

Mark Bagnoli and Ted Bergstrom. Log-concave proba-

bility and its applications. Economic theory, 26(2): 445–469, 2005.

Linas Baltrunas, Tadas Makcinskas, and Francesco Ricci. Group recommendations with rank aggregation and collaborative filtering. In Proceedings of the ACM Conference on Recommender Systems, pages 119–126, 2010.

Sergey Bobkov and Michel Ledoux. One-dimensional empirical measures, order statistics, and kantorovich transport distances. Memoirs of the American Mathematical Society, 261(1259), 2019.

Christer Borell. Convex set functions in d-space. Periodica Mathematica Hungarica, 6(2):111–136, 1975.

Róbert Busa-Fekete, Eyke Hüllermeier, and Balázs Szörényi. Preference-based rank elicitation using statistical models: The case of Mallows. In Proceedings of the International Conference on Machine Learning (ICML), volume 32 of Proceedings of Machine Learning Research, pages 1071–1079. PMLR, 2014a.

Róbert Busa-Fekete, Balázs Szörényi, and Eyke Hüllermeier. PAC rank elicitation through adaptive sampling of stochastic pairwise preferences. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 28, pages 1701–1707, 2014b.

Yuxin Chen and Changho Suh. Spectral MLE: Top-K rank aggregation from pairwise comparisons. In Proceedings of the International Conference on Machine Learning (ICML), volume 37 of Proceedings of Machine Learning Research, pages 371–380. PMLR, 2015.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems (NIPS), volume 30, pages 4299–4307, 2017.

Douglas E. Critchlow, Michael A. Fligner, and Joseph S. Verducci. Probability models on rankings. Journal of Mathematical Psychology, 35(3):294–318, 1991.

Moein Falahatgar, Ayush Jain, Alon Orlitsky, Venkatadheeraj Pichapati, and Vaishakh Ravindrakumar. The limits of maxing, ranking, and preference learning. In Proceedings of the International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 1427–1436. PMLR, 2018.

Bruce Hajek, Sewoong Oh, and Jiaming Xu. Minimaxoptimal inference from partial rankings. In Advances in Neural Information Processing Systems (NIPS), volume 27, pages 1475–1483, 2014.

Roxana A. Ion, Chris A. J. Klaassen, and Edwin R. van den Heuvel. Sharp inequalities of Bienaymé- Chebyshev and Gauss type for possibly asymmet-

ric intervals around the mean. arXiv preprint arXiv:2208.08813, 2022.

Anders Jonsson, Emilie Kaufmann, Pierre Ménard, Omar Darwiche Domingues, Edouard Leurent, and Michal Valko. Planning in Markov decision processes with gap-dependent sample complexity. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 1253–1263, 2020.

Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best-arm identification in multi-armed bandit models. Journal of Machine Learning Research, 17(1):1–42, 2016.

Maurice G. Kendall. A new measure of rank correlation. Biometrika, 30(1-2):81–93, 1938.

Ashish Khetan and Sewoong Oh. Data-driven rank breaking for eficient rank aggregation. Journal of Machine Learning Research, 17(193):1–54, 2016.

László Lovász and Santosh Vempala. The geometry of logconcave functions and sampling algorithms. Random Structures & Algorithms, 30(3):307–358, 2007.

Tyler Lu and Craig Boutilier. Learning Mallows models with pairwise preferences. In Proceedings of the International Conference on Machine Learning (ICML), pages 145–152. Omnipress, 2011.

R. Duncan Luce. Individual Choice Behavior: A Theoretical Analysis. Wiley New York, 1959.

Andreas Maurer and Massimiliano Pontil. Empirical Bernstein bounds and sample-variance penalization. In Annual Conference on Learning Theory (COLT), 2009.

Daniel McFadden. Econometric models for probabilistic choice among products. Journal of Business, 53(3): S13–S29, 1980.

Sahand Negahban, Sewoong Oh, and Devavrat Shah. Iterative ranking from pair-wise comparisons. In Advances in Neural Information Processing Systems (NIPS), volume 25, pages 2474–2482, 2012.

Robin L. Plackett. The analysis of permutations. Journal of the Royal Statistical Society Series C: Applied Statistics, 24(2):193–202, 1975.

Paramesh Ray. Independence of irrelevant alternatives. Econometrica, 41(5):987–991, 1973.

Aadirupa Saha and Aditya Gopalan. Active ranking with subset-wise preferences. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS), volume 89 of Proceedings of Machine Learning Research, pages 3312–3321. PMLR, 2019.

Aadirupa Saha and Aditya Gopalan. Best-item learning in random utility models with subset choices. In

Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS), volume 108 of Proceedings of Machine Learning Research, pages 4281–4291. PMLR, 2020.

Nihar B. Shah, Sivaraman Balakrishnan, Joseph Bradley, Abhay Parekh, Kannan Ramchandran, and Martin J. Wainwright. Estimation from pairwise comparisons: Sharp minimax bounds with topology dependence. Journal of Machine Learning Research, 17(58):1–47, 2016.

Edward Szpilrajn. Sur l’extension de l’ordre partiel. Fundamenta Mathematicae, 16(1):386–389, 1930.

Louis L. Thurstone. A law of comparative judgment. Psychological Review, 34(4):273–286, 1927.

Leslie G. Valiant. A theory of the learnable. Communications of the ACM, 27(11):1134–1142, 1984.

Christian Wirth, Riad Akrour, Gerhard Neumann, and Johannes Fürnkranz. A survey of preference-based reinforcement learning methods. Journal of Machine Learning Research, 18(136):1–46, 2017.

Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012.

Masrour Zoghi, Shimon Whiteson, Remi Munos, and Maarten de Rijke. Relative upper confidence bound for the k-armed dueling bandit problem. In Proceedings of the International Conference on Machine Learning (ICML), volume 32 of Proceedings of Machine Learning Research, pages 10–18. PMLR, 2014.

## A RELATED WORKS

Our work relates to the existing literature along three primary dimensions: the way of generating feedback, the definition of the objectives and metrics, and the types of feedback used to achieve the target.

A standard paradigm in preference learning is inferring latent properties from human judgments, such as distinguishing “good” from “bad” trajectories in RLHF (Wirth et al., 2017; Christiano et al., 2017) or predicting user choices in recommender systems (McFadden, 1980; Baltrunas et al., 2010). While the PL model (Plackett, 1975; Luce, 1959) is the ubiquitous choice for these tasks, it is a specific instance of the broader Random Utility Model (RUM) framework (Thurstone, 1927). Our work adopts the RUM framework with log-concave noise, which provides a unifying lens for these models. Several works implicitly utilize log-concavity by considering the PL model (Saha and Gopalan, 2019; Chen and Suh, 2015; Negahban et al., 2012; Khetan and Oh, 2016; Azari Soufiani et al., 2014), while others do this explicitly (Hajek et al., 2014). Remarkably, log-concavity of the noise ensures several desirable properties of the implied ranking distribution (Critchlow et al., 1991). In this paper, we exploit this regularity to derive uniform lower bounds on the resolution factors.

The definition of “preference recovery” varies significantly across the literature. Multi-armed and dueling bandit frameworks often focus on identifying a single best item (Saha and Gopalan, 2020; Yue et al., 2012; Zoghi et al., 2014), a paradigm known as best-item identification. Many approaches focus on statistical parameter estimation, with particular attention devoted to the estimation of PL parameters, typically measured by the $L _ { \mathrm { { 2 } } } \mathrm { { - n o r m } }$ (Azari Soufiani et al., 2013; Hajek et al., 2014; Shah et al., 2016). When the goal is a ranking, metrics like Kendall-tau distance (Kendall, 1938) or maximum rank diference (Busa-Fekete et al., 2014b) are standard. When the utility ranking is the target, it is possible to bypass utilities entirely by assuming distributions over permutations, such as the Mallows model (Lu and Boutilier, 2011; Busa-Fekete et al., 2014a). In contrast to these approaches, we operate within the PAC framework (Valiant, 1984), defining success through the notion of ε-accuracy. This objective bridges the gap between pure parameter estimation and exact ordinal recovery, by incorporating utility information into the target ranking in an intuitive manner.

Finally, we distinguish works by the density of the feedback signal. In line with the present paper, some works consider winner-only feedback signals (Saha and Gopalan, 2019, 2020), which are generalized through the notion of top-m feedback. For full-ranking feedback, a common technique is rank-breaking, which decomposes permutations into pairwise comparisons modeled via a comparison graph (Azari Soufiani et al., 2014; Khetan and Oh, 2016; Negahban et al., 2012).

Distance Metrics for Ranking Accuracy. There exist several ways of measuring the distance between two rankings, including the popular Kendall tau distance and the maximum rank diference (Kendall, 1938; Busa-Fekete et al., 2014b). These do not require the ranked items to be associated with utility scores and penalize misclassification uniformly across all item pairs, thus making the comparison with our notion of ε-accuracy vacuous. In contrast, we can provide a more meaningful comparison for the following measures.

• ε-optimality. This was introduced by Saha and Gopalan (2020) and is similar to the one given by Definition 1. Indeed, an item is ε-optimal if its utility value is smaller than the largest utility by less than ε. However, in contrast to our metric, this notion of distance induces a much simpler objective as it is only concerned with determining a nearly optimal item.

• ε-best ranking. This notion was introduced by Saha and Gopalan (2019) and is intrinsically gap-dependent. However, despite its similarities with our notion of ε-accuracy, it does not refer to utility gaps. Instead, it is concerned with the gaps between the Plackett-Luce parameters. The latter correspond, up to constants, to the winning probabilities and can be mapped to the utility values through a non-linear function. Consequently, the objective implied by ε-optimality is essentially diferent from ours. In particular, the former is much more tolerant towards items with a small utility because their winning probabilities are small. Optimizing with respect to this metric obscures the true dificulty of the WO feedback, that is, its sensitivity to $P _ { \mathrm { m i n } }$

• ε-ordering. This notion was introduced by Falahatgar et al. (2018). For a fixed noise distribution, using the full-ranking resolution factor η introduced in Section 4, one can show that:

$$
\varepsilon - \mathrm { a c c u r a t e ~ ( o u r s ) } \Longleftrightarrow \frac { \eta ( \varepsilon ) } { 2 } - \mathrm { r a n k i n g ~ ( F a l a h a t g a r ~ e t ~ a l . ~ ( 2 0 1 8 ) ) } .
$$

This equivalence can only be derived if we choose a ranking model induced by a RUM (thus diverging from the setting of (Falahatgar et al., 2018)) and use the true resolution factor η instead of the lower bound $\eta ^ { \star }$ given by Equation 4. Most importantly, the notion of ε-ordering was given in the context of ranking models where there are no underlying utilities that generate the ranking.

Comparison with Saha and Gopalan (2020). Focusing on winner-only feedback, we elaborate on the diferences and similarities between our work and that of Saha and Gopalan (2020). While both frameworks seek to lower-bound the winning-probability ratio $P _ { i } / P _ { j }$ in terms of the utility gap $u _ { i } - u _ { j }$ , the methodologies and objectives difer in two key aspects. First, Saha and Gopalan (2020) focus on identifying the ε-best item, whereas we target the recovery of the full utility ranking. Under WO feedback, the latter becomes a significantly more challenging problem. This is due to the informational bottleneck represented by the minimum winning probability $P _ { \mathrm { m i n } }$ , as established by our lower bound analysis formalized through Theorem 4. Second, their approach uses the so-called minimum advantage ratio and relies on a distribution-specific constant $^ { c , }$ which is derived via a so-called “variational lower bound”. This constant must be computed for each noise model and provided to the algorithm as a parameter. In contrast, our approach leverages the structural regularity of log-concave distributions. Consequently, our algorithms only require an upper bound on the noise variance, regardless of any other distribution-specific parameters.

## B VARIANCE-AWARE UTILITY RANKERS

## B.1 Distribution Table

The table below lists several examples of log-concave distributions and, for each of them, provides explicit conditions ensuring that the mean is zero and Assumption (V) is satisfied. If these two are met, the given distribution is allowed by the FR-VUR algorithm presented in Section 4. Provided that the additional utilityrange-dependent condition (WO1) is also met, the derived sample-complexity guarantees hold for both FR-VUR and WO-VUR.
<table><tr><td>Noise distribution</td><td>Centeredness</td><td>Bounded-variance (V)</td><td>FR-VUR</td><td>WO-VUR</td></tr><tr><td> ${ \mathrm { G u m b e l } } ( \mu , s )$ </td><td> $\mu = - s \gamma$ </td><td> $s \leq \sqrt { 6 V } / \pi$ </td><td>√</td><td>√</td></tr><tr><td> $\operatorname { N o r m a l } ( \mu , \sigma ^ { 2 } )$ </td><td> $\mu = 0$ </td><td> $\sigma \le \sqrt { V }$ </td><td>√</td><td>√</td></tr><tr><td> $\operatorname { L o g i s t i c } ( \mu , s )$ </td><td> $\mu = 0$ </td><td> $s \leq \sqrt { 3 V } / \pi$ </td><td>√</td><td>√</td></tr><tr><td> $\mathrm { L a p l a c e } ( \mu , s )$ </td><td> $\mu = 0$ </td><td> $s \leq \sqrt { V / 2 }$ </td><td>√</td><td>√</td></tr><tr><td> $\operatorname { U n i f o r m } ( c , d )$ </td><td> $c = - d$ </td><td> $d - c \leq \sqrt { 1 2 V }$ </td><td>√</td><td>vt</td></tr></table>

Table 1: The parameter γ in the centeredness condition of the Gumbel distribution is the Euler-Mascheroni constant, whose value is approximately 0.58. All five families satisfy the noise assumptions for both algorithms under the stated parameter conditions. For WO feedback, † additionally requires $\Delta _ { u } < d - c$ for uniform noise. The other four families have full support, so Assumption (WO1) holds automatically for any set of utilities.

## B.2 Pseudocode for Variance-aware Utility-Rankers

Below is the pseudocode for the VUR algorithms presented in Sections 4 and 5, respectively.

## C TECHNICAL RESULTS AND PROOFS FOR SECTION 4

## C.1 The Pairwise-preference Probabilities

We begin by showing that the pairwise-preference probabilities $P _ { i j }$ behave consistently with the underlying utilities.

Lemma 5. Let $P _ { i j } = \mathbb { P } \left[ U _ { i } > U _ { j } \right]$ be the pairwise-preference probability defined in Section 4. Then, $P _ { i j } > 1 / 2$ if and only if $u _ { i } > u _ { j }$

Algorithm 1 Full-Ranking Variance-aware Utility Ranker   
Input: Items [k], confidence $\delta \in ( 0 , 1 )$ , accuracy $\varepsilon > 0 .$ , variance bound $V > 0$   
1: Initialize: $n  0 , G  ( [ k ] , \emptyset ) , \eta ^ { \star } $ min $( \textstyle { \frac { 2 } { 3 } } , \frac { \varepsilon } { \sqrt { 6 V } } )$   
2: $P _ { i j }  0 , V _ { i j }  0 ,$ and $I _ { i j }  \eta ^ { \star } / 4$ for every $i \neq j$ in [k]   
3: repeat   
4: $n \gets n + 1$   
5: Observe full ranking $\sigma _ { n }$ from $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ and set $\begin{array} { r } { \delta _ { n } \gets \frac { 6 \delta } { k ( k - 1 ) n ^ { 2 } \pi ^ { 2 } } } \end{array}$   
6: for each pair $i \neq j$ do   
7: $\begin{array} { r } { P _ { i j }  \frac { n - 1 } { n } P _ { i j } + \frac { 1 } { n } \mathbb { 1 } \{ i } \end{array}$ is ranked before $j$ in $\sigma _ { n } \}$   
8: if $n > 1$ then   
9: $\begin{array} { r } { V _ { i j }  \frac { n } { n - 1 } P _ { i j } ( 1 - P _ { i j } ) } \end{array}$   
10: $\begin{array} { r } { I _ { i j }  \sqrt { \frac { 2 V _ { i j } \ln ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \ln ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } } \end{array}$   
11: end if   
12: end for   
13: until $I _ { i j } < \eta ^ { \star } / 4$ for every $i \neq j$   
14: for $i , j$ such that $1 \leq i < j \leq k$ do   
15: if $P _ { i j } > 1 / 2 + \eta ^ { \star } / 4$ then   
16: Add $i \longrightarrow j$ to G   
17: else if $P _ { j i } > 1 / 2 + \eta ^ { \star } / 4$ then   
18: Add $j \longrightarrow i$ to G   
19: end if   
20: end for   
21: if G is acyclic then   
22: G ← transitive closure of G and $\hat { \sigma } \gets$ topological sort of G   
23: else   
24: $\hat { \sigma } \gets$ arbitrary permutation of [k]   
25: end if   
Output: σˆ

Algorithm 2 Winner-Only Variance-aware Utility Ranker   
Input: Items [k], confidence $\delta \in ( 0 , 1 )$ , accuracy $\varepsilon \in ( 0 , b - a )$ , variance bound $V > 0$   
1: Initialize: $\begin{array} { r } { n \gets 0 , \eta ^ { \star } \gets \frac { e - 1 } { 3 e + 1 } } \end{array}$ min $\left\{ 1 , { \frac { \varepsilon } { \sqrt { 3 V } } } \right\}$   
2: $P _ { i }  1 , V _ { i }  1 / 4$ , and $\dot { I } _ { i } \gets \eta ^ { \star } / 2$ for every $i \in [ k ]$   
3: repeat   
4: $n \gets n + 1$   
5: Observe winner $w _ { n }$ from $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ and set $\delta _ { n } \gets \frac { 3 \delta } { k \pi ^ { 2 } n ^ { 2 } }$   
6: for each $i \in [ k ]$ do   
7: $\begin{array} { r } { P _ { i } \gets \frac { n - 1 } { n } \dot { P _ { i } } + \frac { 1 } { n } \mathbb { 1 } \{ w _ { n } = i \} } \end{array}$   
8: if $n > \ddot { 1 }$ then   
9: $\begin{array} { r } { V _ { i } \gets \frac { n } { n - 1 } P _ { i } ( 1 - P _ { i } ) } \end{array}$   
10: $\begin{array} { r } { I _ { i } \gets \sqrt { \frac { 2 V _ { i } \ln ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \ln ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } } \end{array}$   
11: end if   
12: end for   
13: until $I _ { i } < P _ { i } \cdot \eta ^ { \star } / 2$ for every $i \in [ k ]$   
14: $\hat { \sigma } \gets$ unique permutation $\sigma \in S _ { k }$ such that $\sigma ( i ) < \sigma ( j ) \Longleftrightarrow P _ { i } > P _ { j } \lor ( P _ { i } = P _ { j }$ and $i < j )$   
Output: σˆ

Proof. We denote by $f$ and $F$ the density and the CDF of $\nu ,$ respectively. Additionally, we let $( a , b )$ be an interval fully contained in Supp(f).

If $u _ { i } = u _ { j }$ , then, by symmetry, we have $P _ { i j } = 1 / 2$ . If $u _ { i } > u _ { j }$ , then

$$
\begin{array} { r l } & { P _ { i j } = \displaystyle \int _ { \mathbb R } \mathbb P \left[ U _ { j } < U _ { i } | U _ { i } = x \right] f ( x - u _ { i } ) \mathrm { d } x } \\ & { \quad \quad = \displaystyle \int _ { \mathbb R } F ( x - u _ { j } ) f ( x - u _ { i } ) \mathrm { d } x } \\ & { \quad \quad \stackrel { \mathrm { ( i ) } } { = } \displaystyle \int _ { \mathbb R } F ( y + u _ { i } - u _ { j } ) f ( y ) \mathrm { d } y , } \end{array}
$$

where we used the change of variable $y = x - u _ { i }$ for (i). Now consider the function $g : [ 0 , \infty ) $ R defined by

$$
g ( t ) : = \int _ { \mathbb { R } } F ( y + t ) f ( y ) \mathrm { d } y .
$$

Taking the first derivative, we find

$$
g ^ { \prime } ( t ) = \int _ { \mathbb { R } } f ( y + t ) f ( y ) \mathrm { d } y .
$$

Since $( a , b ) \subset \operatorname { S u p p } ( f )$ , for $t < b - a$ , there exists a subinterval of Supp(f) where both $f ( y + t )$ and $f ( y )$ are strictly positive. Thus,

$$
g ^ { \prime } ( t ) > 0 \quad \mathrm { f o r ~ e v e r y ~ } t \in [ 0 , b - a ) .
$$

$$
g ^ { \prime } ( t ) \geq 0
$$

$$
t \geq 0 .
$$

$$
g ( 0 ) = 1 / 2
$$

$$
g
$$

$$
g ( t ) > 1 / 2
$$

$$
t \in ( 0 , b - a )
$$

$$
t \geq b - a
$$

$$
g ( t ) \geq g ( t _ { 0 } ) > 1 / 2
$$

$$
t _ { 0 } \in ( 0 , b - a )
$$

$$
P _ { i j } = g ( u _ { i } - u _ { j } )
$$

$$
u _ { i } - u _ { j } > 0
$$

$$
P _ { i j } > 1 / 2
$$

To prove the converse implication, suppose $P _ { i j } > 1 / 2$ . As noted above, we cannot have $u _ { i } = u _ { j }$ , so either $u _ { i } < u _ { j }$ or $u _ { i } > u _ { j }$ . The former would imply $P _ { j i } > 1 / 2$ , but since $P _ { i j } + P _ { j i } = 1$ , this yields a contradiction. 口

## C.2 Concentration Bounds

The following result provides anytime-concentration bounds for the unbiased estimators introduced in Section 4.   
Our inequalities are based on (Maurer and Pontil, 2009).

Lemma 6. Fix $\delta \in ( 0 , 1 )$ and let $\delta _ { n }$ be the time-n confidence parameter defined in Section $4 . 1 .$ Then, with probability at least $1 - \delta$ , the following hold simultaneously for every pair $( i , j )$ with $i \neq j$ and every $n \in \mathbb { N } \setminus \{ 1 \}$

$$
| P _ { i j } - P _ { i j } ^ { n } | \leq \sqrt { \frac { 2 V _ { i j } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } ,\tag{18}
$$

$$
\left| \sqrt { V _ { i j } ^ { n } } - \sqrt { P _ { i j } ( 1 - P _ { i j } ) } \right| \leq \sqrt { \frac { 2 \log ( 2 / \delta _ { n } ) } { n - 1 } } .\tag{19}
$$

Proof. For a fixed pair (i, j) with $1 \leq i < j \leq k$ and $n \geq 2$ , we define the events

$$
A _ { i j } ^ { n } : = \left\{ | P _ { i j } - P _ { i j } ^ { n } | \leq \sqrt { \frac { 2 V _ { i j } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } \right\}
$$

and

$$
B _ { i j } ^ { n } : = \left\{ \left| \sqrt { V _ { i j } ^ { n } } - \sqrt { P _ { i j } ( 1 - P _ { i j } ) } \right| \leq \sqrt { \frac { 2 \log ( 2 / \delta _ { n } ) } { n - 1 } } \right\} .
$$

By Theorem 4 and Theorem 10 in (Maurer and Pontil, 2009), we have $\mathbb { P } \left[ A _ { i j } ^ { n } \right] \geq 1 - \delta _ { n }$ and $\mathbb { P } \left[ B _ { i j } ^ { n } \right] \geq 1 - \delta _ { n }$ . By

taking a union bound, we obtain

$$
\begin{array} { r l } { \mathbb { P } \left[ \displaystyle \bigcap _ { n \in \mathbb { N } } \left( \displaystyle \bigcap _ { 1 \leq i < j \leq k } ( A _ { i j } ^ { n } \cap B _ { i j } ^ { n } ) \right) \right] = 1 - \mathbb { P } \left[ \displaystyle \bigcup _ { n \in \mathbb { N } } \left( \displaystyle \bigcup _ { 1 \leq i < j \leq k } ( A _ { i j } ^ { n } ) ^ { c } \cup ( B _ { i j } ^ { n } ) ^ { c } \right) \right] } & { } \\ { \geq 1 - \displaystyle \frac { k ( k - 1 ) } { 2 } \sum _ { n = 1 } ^ { \infty } 2 \delta _ { n } } & { } \\ { = 1 - \displaystyle \frac { 6 \delta } { \pi ^ { 2 } } \sum _ { n = 1 } ^ { \infty } \frac { 1 } { n ^ { 2 } } } & { } \\ { \overset { ( \mathrm { i } ) } { = } 1 - \delta , } \end{array}
$$

where (i) follows from the well-known fact that $\textstyle \sum _ { n \in \mathbb { N } } 1 / n ^ { 2 } = \pi ^ { 2 } / 6$ . To conclude, it sufices to observe that, for any two distinct items i and $j ,$ one has

$$
| P _ { i j } - P _ { i j } ^ { n } | = | P _ { j i } - P _ { j i } ^ { n } | \quad { \mathrm { a n d } } \quad V _ { i j } ^ { n } = V _ { j i } ^ { n } .
$$

## C.3 Uniform Lower Bound on the Resolution Function

In this section, we formally derive a uniform lower bound on the FR resolution function that holds across all noise distributions satisfying Assumption (V).

Lemma 7. Let $\nu \in \mathcal N$ be a noise distribution satisfying the bounded-variance condition (V). Then

$$
\eta ( t ) \geq \eta ^ { \ast } ( t ) \quad f o r \ e v e r y \ t \in [ 0 , \infty ) ,
$$

where $\eta ^ { \star }$ is defined by (4).

Proof. Let X and Y be two i.i.d. random variables with distribution $\nu ,$ and let $Z : = Y - X$ denote their diference, which is also log-concave. Additionally, Z is centered and symmetric, and its variance satisfies $\mathbb { V } \mathrm { a r } \left[ Z \right] = \mathbb { V } \mathrm { a r } \left[ X \right] + \mathbb { V } \mathrm { a r } \left[ X \right] \leq 2 V$ . Let us denote by $\sigma > 0$ the standard deviation of $Z ,$ and let H be its CDF. Then, for any $t > 0$ , we have:

$$
{ \begin{array} { r l } { \displaystyle \int _ { \mathbb { R } } F ( x + t ) f ( x ) d x = \int _ { \mathbb { R } } \mathbb { P } \left[ Y < x - t \right] f ( x ) d x = } \\ { \displaystyle } & { = \int _ { \mathbb { R } } \mathbb { P } \left[ Y < X - t \mid X = x \right] f ( x ) d x = } \\ { \displaystyle } & { = \mathbb { P } \left[ Y - X < - t \right] = } \\ { \displaystyle } & { = \mathbb { P } \left[ Z < - t \right] = H ( - t ) . } \end{array} }
$$

The left-hand side is equal to $\xi ( t )$ , where $\xi$ is the function defined by Equation (2), which satisfies $\eta ( t ) = 2 \xi ( t ) - 1$ By symmetry, we have:

$$
\xi ( t ) = 1 - H ( - t ) , \quad \forall t > 0 .
$$

using Lemma 19, we can bound the term $H ( - t )$ as follows:

$$
H ( - t ) \leq { \left\{ \begin{array} { l l } { { \displaystyle { \frac { 1 } { 2 } } - { \frac { t } { 2 { \sqrt { 3 } } \sigma } } } } & { { \mathrm { i f ~ } } 0 \leq t < 2 \sigma / { \sqrt { 3 } } } \\ { { \displaystyle { \frac { 2 \sigma ^ { 2 } } { 9 t ^ { 2 } } } } } & { { \mathrm { i f ~ } } t \geq 2 \sigma / { \sqrt { 3 } } } \end{array} \right. } .
$$

We now consider two cases.

Case 1. If $t \in ( 0 , 2 \sigma { \sqrt { 3 } } )$ , then $H ( - t ) \leq 1 / 2 - t / 2 { \sqrt { 3 } } \sigma$ . Hence, we obtain:

$$
\begin{array} { l } { \displaystyle 1 - H ( - t ) \geq \frac 1 2 + \frac { t } { 2 \sqrt { 3 } \sigma } \overset { \sigma \leq \sqrt { 2 V } } { \geq } } \\ { \displaystyle \geq \frac 1 2 + \frac { t } { 2 \sqrt { 6 V } } = } \\ { \displaystyle = \frac 1 2 \left( 1 + \frac { t } { \sqrt { 6 V } } \right) . } \end{array}
$$

Using the relation $\eta ( t ) = 2 \xi ( t ) - 1$ , we conclude that $\eta ( t ) \geq t / \sqrt { 6 V }$

Case 2. If $t \geq 2 \sigma / \sqrt { 3 } .$ , we have

$$
H ( - t ) \leq { \frac { 2 \sigma ^ { 2 } } { 9 t ^ { 2 } } } \leq { \frac { 2 \sigma ^ { 2 } } { 9 ( 2 \sigma / { \sqrt { 3 } } ) ^ { 2 } } } = { \frac { 1 } { 6 } } .
$$

This implies

$$
\xi ( t ) \geq { \frac { 1 } { 2 } } \left( 1 + { \frac { 2 } { 3 } } \right) ,
$$

from which we obtain that $\eta ( t ) \geq 2 / 3$

Combining the conclusions of both cases yields the desired bound.

## C.4 The PAC Property

In this section, we prove two results showing that the FR-VUR algorithm presented in Section 4 is $( \varepsilon , \delta ) – \mathrm { P A C }$ Lemma 8. With probability at least $1 - \delta$ , the pairwise-preference graph $G _ { T }$ is a subgraph of the utility graph.

Proof. With probability at least $1 - \delta .$ , the inequality given by (18) holds for all $i \neq j$ and $n \geq 2$ . In particular, taking $n = T$ , this further implies

$$
| P _ { i j } ^ { T } - P _ { i j } | < \frac { \eta ^ { \star } } { 4 } \quad \forall i \neq j .
$$

If $i \longrightarrow j \in \mathcal { E } _ { T }$ , i.e., there is an edge from i to $j$ in $G _ { T }$ , then

$$
P _ { i j } = P _ { i j } ^ { T } + ( P _ { i j } - P _ { i j } ^ { T } ) > \frac { 1 } { 2 } + \frac { \eta ^ { \star } } { 4 } - \frac { \eta ^ { \star } } { 4 } = \frac { 1 } { 2 } .
$$

Since $P _ { i j } > 1 / 2$ if and only if $u _ { i } > u _ { j }$ (see Lemma 5), we conclude that $i \longrightarrow j$ is an edge of $G _ { u }$ □

Lemma 9. With probability at least 1 − δ, if i and j are any two items such that $u _ { i } - u _ { j } \geq \varepsilon ,$ , then $\hat { \sigma } ( i ) < \hat { \sigma } ( j )$

Proof. With probability at least $1 - \delta ,$ inequality (18) holds for all $i \neq j$ and $n \geq 2$ . In this regime, the condition of Lemma 8 is satisfied, so, taking $n = T$

$$
| P _ { i j } ^ { T } - P _ { i j } | < \frac { \eta ^ { \star } } { 4 } \quad \forall i \neq j .
$$

Now let i and $j$ be any two items such that $| u _ { i } - u _ { j } | \geq \varepsilon .$ , say $u _ { i } \geq u _ { j } + \varepsilon .$ . If $\eta ( \varepsilon )$ denotes the value of the resolution function computed for the true noise distribution, then Lemma $7$ ensures $\eta ( \varepsilon ) \geq \eta ^ { \star }$ . By (3) and the monotonicity of $\xi , P _ { i j } \ge \xi ( \varepsilon )$ . Using the definition of $\eta ( \varepsilon )$ , we have

$$
P _ { i j } \ge \frac { 1 } { 2 } + \frac { \eta ( \varepsilon ) } { 2 } \ge \frac { 1 } { 2 } + \frac { \eta ^ { \star } } { 2 } .
$$

Hence,

$$
P _ { i j } ^ { T } = P _ { i j } + ( P _ { i j } ^ { T } - P _ { i j } ) > \frac { 1 } { 2 } + \frac { \eta ^ { \star } } { 2 } - \frac { \eta ^ { \star } } { 4 } = \frac { 1 } { 2 } + \frac { \eta ^ { \star } } { 4 } .
$$

This implies $i \longrightarrow j \in \mathcal { E } _ { T }$ . Since $G _ { T }$ is a DAG, from the definition of ${ \hat { \sigma } } ,$ it follows that $\hat { \sigma } ( i ) < \hat { \sigma } ( j )$

## C.5 Sample Complexity Upper Bound

In this section, we prove Theorem 1. To do so, we first define $\mathcal { E }$ as the set of outcomes for which the following hold simultaneously for every $n \in \mathbb { N } \setminus \{ 1 \}$ and any two distinct i and $j \colon$

$$
| P _ { i j } ^ { n } - P _ { i j } | \leq \sqrt { \frac { 2 V _ { i j } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } ,\tag{20}
$$

$$
\left| \sqrt { V _ { i j } ^ { n } } - \sqrt { P _ { i j } ( 1 - P _ { i j } ) } \right| \leq \sqrt { \frac { 2 \log ( 2 / \delta _ { n } ) } { n - 1 } } ,\tag{21}
$$

where $\delta _ { n }$ is the one given in Section 4.1.

If (21) holds, then multiplying both sides of $\sqrt { V _ { i j } ^ { n } } \leq \sqrt { P _ { i j } ( 1 - P _ { i j } ) } + \sqrt { 2 \log ( 2 / \delta _ { n } ) / ( n - 1 ) } \mathrm { ~ b y ~ } \sqrt { 2 \log ( 4 / \delta _ { n } ) / n }$ gives

$$
\sqrt { \frac { 2 V _ { i j } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } \leq \sqrt { \frac { 2 P _ { i j } ( 1 - P _ { i j } ) \log ( 4 / \delta _ { n } ) } { n } } + \frac { 2 \log ( 4 / \delta _ { n } ) } { n - 1 } ,
$$

where we used $\log ( 2 / \delta _ { n } ) \leq \log ( 4 / \delta _ { n } )$ and $n ( n - 1 ) \geq ( n - 1 ) ^ { 2 }$ . Adding $7 \log ( 4 / \delta _ { n } ) / ( 3 ( n - 1 ) )$ to both sides, we obtain

$$
\sqrt { \frac { 2 V _ { i j } ^ { n } \log ( 4 / \delta _ { n } ) } { n } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } } \le \sqrt { \frac { 2 P _ { i j } ( 1 - P _ { i j } ) \log ( 4 / \delta _ { n } ) } { n } } + \frac { 1 3 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } ,\tag{22}
$$

whose left-hand side is exactly the quantity appearing in the stopping rule (5). Thus, if E occurs, the stopping rule is satisfied if the following hold simultaneously for every $i \neq j \colon$

$$
\sqrt { \frac { 2 P _ { i j } ( 1 - P _ { i j } ) \log ( 4 / \delta _ { n } ) } { n } } < \frac { \eta ^ { \star } } { 8 } ,\tag{23}
$$

$$
\frac { 1 3 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } < \frac { \eta ^ { \star } } { 8 } .\tag{24}
$$

Condition (23) holds for every $i \neq j$ if the stronger requirement below is satisfied:

$$
n > \operatorname* { m a x } _ { i \neq j } ( P _ { i j } ( 1 - P _ { i j } ) ) \cdot \frac { 1 2 8 \log ( 4 / \delta _ { n } ) } { \eta ^ { \star ^ { 2 } } } .
$$

In particular, it holds for $n _ { 0 }$ , which we define as the minimum positive integer n such that

$$
n \geq \operatorname* { m a x } \left\{ 2 , p ( 1 - p ) \cdot \frac { 1 2 8 \log ( 4 / \delta _ { n } ) } { \eta ^ { \star 2 } } \right\} ,
$$

where $p \in [ 0 , 1 ]$ is chosen so that

$$
p ( 1 - p ) = \operatorname* { m a x } _ { i \neq j } ( P _ { i j } ( 1 - P _ { i j } ) ) .
$$

Using the explicit expression of $\delta _ { n }$ given in Section 4.1, we have

$$
n _ { 0 } \le p ( 1 - p ) \cdot \frac { 1 2 8 \log ( 4 / \delta _ { n _ { 0 } } ) } { \eta ^ { \star 2 } } + 1 \le \operatorname* { m a x } \left( p ( 1 - p ) \cdot \frac { 2 5 6 } { \eta ^ { \star 2 } } \log \left( \frac { 2 k n _ { 0 } \pi } { \sqrt { 6 \delta } } \right) , 2 \right) .
$$

By Lemma 18, there exists a universal constant $C _ { 0 }$ such that

$$
n _ { 0 } \le \operatorname* { m a x } \left( \frac { C _ { 0 } p ( 1 - p ) } { { \eta ^ { \star } } ^ { 2 } } \log \left( \frac { C _ { 0 } p ( 1 - p ) } { { \eta ^ { \star } } ^ { 2 } } \cdot \frac { k } { \delta } \right) , 2 \right) ,\tag{25}
$$

where $\log ( 0 )$ is understood as $- \infty$

Now we consider the condition given by (24), and define $n _ { 1 }$ as the minimum positive integer n such that

$$
n \geq \left\lceil \frac { 1 0 4 \log ( 4 / \delta _ { n } ) } { 3 \eta ^ { \star } } + 1 \right\rceil .
$$

Such an integer exists, since the right-hand side grows logarithmically in $n ,$ while the left-hand side grows linearly. By minimality, the inequality above fails at $n _ { 1 } - 1$ , and since $\delta _ { n }$ is non-increasing in n, this yields

$$
n _ { 1 } \leq \frac { 1 0 4 \log ( 4 / \delta _ { n _ { 1 } } ) } { 3 \eta ^ { \star } } + 2 .
$$

It follows that

$$
n _ { 1 } \le \operatorname* { m a x } \left( \frac { 2 0 8 \log ( 4 / \delta _ { n _ { 1 } } ) } { 3 \eta ^ { \star } } , 4 \right) .
$$

Using the definition of $\delta _ { n } .$ , Lemma 18 ensures that there exists a constant $C _ { 1 } > 0$ such that

$$
n _ { 1 } \leq \operatorname* { m a x } \left( \frac { C _ { 1 } } { \eta ^ { \star } } \log \left( \frac { C _ { 1 } } { \eta ^ { \star } } \cdot \frac k \delta \right) , 4 \right) .\tag{26}
$$

If we let $n _ { 2 } : = n _ { 0 } + n _ { 1 }$ , then for every $n \geq n _ { 2 }$ , the conditions given by (23) and (24) are both satisfied.

Now, the sample complexity can be decomposed as

$$
\begin{array} { r } { \mathbb { E } \left[ T \right] = \mathbb { E } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } } \right] + \mathbb { E } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \right] . } \end{array}\tag{27}
$$

Using the tail-sum formula for the expectation, we can rewrite

$$
\mathbb { E } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \right] = \sum _ { n = 1 } ^ { \infty } \mathbb { P } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \geq n \right] .
$$

If $n > n _ { 2 }$ , then necessarily $\mathbb { 1 } _ { \mathcal { E } ^ { c } } = 1$ . Indeed, we defined $n _ { 2 }$ in such a way that, if $\mathcal { E }$ occurs, then the stopping condition is met for all $n \geq n _ { 2 } + 1$ . Therefore, we have

$$
\sum _ { n = n _ { 2 } + 1 } ^ { \infty } \mathbb { P } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \geq n \right] = \sum _ { n = n _ { 2 } + 1 } ^ { \infty } \mathbb { P } \left[ T \geq n \right] .
$$

If $T \geq n .$ , then the stopping condition was not satisfied at time $n - 1$ . Thus, at least one of (20) and (21) is violated for at least one pair $( i , j )$ with $i \neq j .$ By a union bound over the $k ( k - 1 ) / 2$ unordered pairs and the two conditions, this occurs with probability at most $k ( k - 1 ) \delta _ { n - 1 }$ . Hence, using the definition of $\delta _ { n }$ given in Section 4.1,

$$
\sum _ { n = n _ { 2 } + 1 } ^ { \infty } \mathbb { P } \left[ T \geq n \right] \leq k ( k - 1 ) \sum _ { n = n _ { 2 } } ^ { \infty } \delta _ { n } = \sum _ { n = n _ { 2 } } ^ { \infty } \frac { 6 \delta } { \pi ^ { 2 } n ^ { 2 } } \leq C _ { 2 } ,
$$

for a universal constant $C _ { 2 } > 0$ . It follows that

$$
\sum _ { n = 1 } ^ { \infty } \mathbb { P } \left[ T \cdot \mathbb { I } _ { \mathcal { E } ^ { c } } \geq n \right] \leq n _ { 2 } \cdot \mathbb { P } \left[ \mathcal { E } ^ { c } \right] + C _ { 2 } .
$$

Using the decomposition given by (27), this implies

$$
\mathbb { E } \left[ T \right] \leq n _ { 2 } + C _ { 2 } .
$$

To prove the claim, we use the upper bounds given by (25) and (26) together with the definition of $\eta ^ { \star }$ . □

## C.6 Information-theoretic Lower Bound

We derive an information-theoretic lower bound for the ranking problem in the FR setting, thus proving Theorem 2. To this end, we fix a variance upper bound $V > 0$ and consider the centered Gumbel distribution ν with scale parameter $\beta = \sqrt { 6 V } / \pi$ and location parameter $\mu = - \beta \gamma$ , where $\gamma$ is the Euler–Mascheroni constant. This ensures that ν is centered and that its variance is equal to V. We fix an accuracy level ε in the small-gap regime $\varepsilon < \sqrt { 6 V } / ( 2 \pi )$ , which is equivalent to requiring $\tilde { \varepsilon } : = \varepsilon / \beta \in ( 0 , 1 / 2 )$ . Then, consider the utility vectors u and v given by

$$
u _ { 1 } = v _ { 2 } = \varepsilon
$$

and

$$
u _ { \ell } = v _ { m } = 0 \quad { \mathrm { f o r ~ e v e r y ~ } } \ell \in [ k ] \setminus \{ 1 \} { \mathrm { ~ a n d ~ } } m \in [ k ] \setminus \{ 2 \} .
$$

Since the location parameter shifts all sampled utilities by the same constant, the induced ranking law does not depend on $\mu ,$ and we can consider $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ and $\operatorname { R U M } ( \pmb { v } , \nu )$ . These are seen as two instances of a multi-armed bandit problem, where the number of arms is $n = 1$ . At each round t, the learner plays the only available arm. The observed reward is the feedback signal, consisting of a full ranking of items $1 , \ldots , k$ . The ranking distributions $P$ and $Q$ correspond to the reward distributions, and are given by

$$
P ( \sigma ) = \mathbb { P } \left[ U _ { ( i ) } = U _ { \sigma ^ { - 1 } ( i ) } \forall i \in [ k ] \vert \mathrm { R U M } ( \pmb { u } , \nu ) \right]
$$

and

$$
Q ( \sigma ) = \mathbb { P } \left[ U _ { ( i ) } = U _ { \sigma ^ { - 1 } ( i ) } \ \forall i \in [ k ] \vert \mathrm { R U M } ( v , \nu ) \right] ,
$$

for every $\sigma \in S _ { k }$

We denote by $\mathbb { P } _ { u }$ and $\mathbb { P } _ { v }$ the probability measures on $( \Omega , { \mathcal { F } } )$ induced by $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ and $\operatorname { R U M } ( \pmb { v } , \nu )$ , respectively.   
The next result provides an upper bound on the KL divergence between the two ranking distributions.

Lemma 10. Let P and $Q$ be the ranking distributions introduced above. Then, the KL divergence is upper-bounded as

$$
\mathrm { D } _ { \mathrm { K L } } \left( P \left| \right| Q \right) \leq \frac { \pi ^ { 2 } \varepsilon ^ { 2 } } { V } .
$$

Proof. In what follows, we abuse notation and use $P [ \cdot ]$ and $\mathbb { P } _ { u } [ \cdot ]$ to denote the same probability measure.

Consider any sample from $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ . Let $\sigma \in S _ { k }$ and denote by $i _ { j }$ the index of the j-th order statistic. Then, we can factorize the probability of observing σ as

$$
P ( \sigma ) = \mathbb { P } _ { u } [ i _ { 1 } = \sigma ^ { - 1 } ( 1 ) ] \cdot \prod _ { t = 2 } ^ { k - 1 } \mathbb { P } _ { u } [ i _ { t } = \sigma ^ { - 1 } ( t ) \mid i _ { j } = \sigma ^ { - 1 } ( j ) { \mathrm { ~ f o r ~ } } j \leq t - 1 ] .\tag{28}
$$

A similar factorization holds for $Q ,$ and these allow us to think of the sampling process as a sequential one: first, the top item $i _ { 1 }$ is chosen among all items $1 , \ldots , k ;$ then, item $i _ { 2 }$ is picked among all the remaining items; and so on, until only one item is left and is chosen as the last element of the ranking.

According to this interpretation, if $S _ { t } \subseteq [ k ]$ is the set of items left after the first $t - 1$ items have been chosen, we define

$$
P [ i | S _ { t } ] : = { \frac { e ^ { u _ { i } / \beta } } { \sum _ { \ell \in S _ { t } } e ^ { u _ { \ell } / \beta } } } \quad { \mathrm { f o r ~ a l l ~ } } i \in S _ { t } .
$$

Now, using (28), we can write

$$
\operatorname { D } _ { \mathrm { K L } } \big ( P \| Q \big ) = \sum _ { t = 1 } ^ { k - 1 } \mathbb { E } _ { \boldsymbol \nu , \boldsymbol u } \big [ \operatorname { D } _ { \mathrm { K L } } \big ( P [ \cdot | S _ { t } ] \| Q [ \cdot | S _ { t } ] \big ) \big ] ,\tag{29}
$$

where the expectation is taken with respect to $\mathbb { P } _ { u }$ over all pairs $( i , S _ { t } )$ where necessarily $i \in S _ { t }$

For simplicity, we introduce the parameter $\widetilde { \varepsilon } : = \varepsilon / \beta$ , which allows us to rewrite

$$
u _ { 1 } / \beta = \tilde { \varepsilon } \mathrm { a n d } u _ { \ell } / \beta = 0 \mathrm { f o r a l l } \ell > 1 ,
$$

as well as

$$
v _ { 2 } / \beta = \tilde { \varepsilon } \mathrm { \ a n d \ } v _ { \ell } / \beta = 0 \mathrm { \ f o r \ a l l \ } \ell \neq 2 .
$$

Recall that $\tilde { \varepsilon } \in ( 0 , 1 / 2 )$ , which is exactly the small-gap condition $\varepsilon < \sqrt { 6 V } / ( 2 \pi )$ assumed above. To estimate the KL divergence between P and $Q ,$ we consider three diferent cases. In the following analysis, we fix any $t \in [ k - 1 ]$

Case 1: $\{ 1 , 2 \} \cap S _ { t } = \varnothing$

In this case, 1 and 2 have already been picked as the t-th decision round starts, so the conditional winning probabilities given $S _ { t }$ induced by u and v are the same:

$$
\operatorname { D } _ { \mathrm { K L } } \left( P [ \cdot \mid S _ { t } ] \mid \mid Q [ \cdot \mid S _ { t } ] \right) = 0 .
$$

Case 2: 1, $, 2 \in S _ { t }$

In this case, we have

$$
\begin{array} { l } { \displaystyle \mathsf { D } _ { \mathrm { K L } } \left( P [ \cdot | S _ { t } ] \| Q [ \cdot | S _ { t } ] \right) = \displaystyle \sum _ { i \notin S _ { t } } P [ i _ { t } | S _ { t } ] \log \frac { P [ i _ { t } | S _ { t } ] } { Q [ i _ { t } | S _ { t } ] } } \\ { = \displaystyle \sum _ { i _ { t } \in S \backslash \{ 1 , 2 \} } P [ i _ { t } | S _ { t } ] \log \frac { P [ i _ { t } | S _ { t } ] } { Q [ i _ { t } | S _ { t } ] } + \displaystyle \sum _ { \varepsilon = 1 , 2 } P [ \varepsilon | S _ { t } ] \log \frac { P [ \varepsilon | S _ { t } ] } { Q [ \varepsilon | S _ { t } ] } } \\ { \displaystyle \stackrel { ( ) } { = } \frac { \frac { \varepsilon } { \varepsilon } } { | S _ { t } | - 1 + e ^ { \varepsilon } } \log ( e ^ { \varepsilon } ) + \frac { 1 } { | S _ { t } | - 1 + e ^ { \varepsilon } } \log ( e ^ { - \varepsilon } ) } \\ { = \frac { ( e ^ { \varepsilon } - 1 ) \tilde { \varepsilon } } { | S _ { t } | - 1 + e ^ { \varepsilon } } } \\ { \displaystyle \stackrel { ( ) } { \leq } \frac { \tilde { \varepsilon } ( \tilde { \varepsilon } + \tilde { \varepsilon } ^ { 2 } ) } { | S _ { t } | } , } \end{array}
$$

where (i) follows from the fact that the first sum is equal to zero and (ii) is due to $e ^ { x } - 1 \leq x + x ^ { 2 }$ for $x \in ( 0 , 1 / 2 )$ Case 3: $| S _ { t } \cap \{ 1 , 2 \} | = 1$

In this case, we consider two further subcases corresponding to: $( \mathrm { a } ) 1 \in S _ { t }$ and $2 \notin S _ { t }$ , and (b) $2 \in S _ { t }$ and $1 \notin S _ { t }$ Consider (a) first. Setting $\textstyle W : = \sum _ { j \in S _ { t } } e ^ { u _ { j } / \beta }$ and $\begin{array} { r } { W ^ { \prime } : = \sum _ { j \in S _ { t } } e ^ { v _ { j } / \beta } } \end{array}$ , we obtain

$$
\begin{array} { l } { { \displaystyle \mathrm { D } _ { \mathrm { K L } } \left( P [ \cdot | S _ { t } ] \| Q [ \cdot | S _ { t } ] \right) = \sum _ { j \in S _ { t } } P [ i _ { t } = j | S _ { t } ] \log \frac { P [ i _ { t } = j | S _ { t } ] } { Q [ i _ { t } = j | S _ { t } ] } } } \\ { { \displaystyle \quad = \frac { e ^ { \frac { \varepsilon } { \varepsilon } } } { W } \log \frac { W ^ { \prime } \cdot e ^ { \frac { \varepsilon } { \varepsilon } } } { W } + \sum _ { j \in S _ { t } \backslash \{ 1 \} } \frac { 1 } { W } \log \frac { W ^ { \prime } } { W } } } \\ { { \displaystyle \quad = \frac { e ^ { \frac { \varepsilon } { \varepsilon } } } { W } \left( \tilde { \varepsilon } + \log \frac { W ^ { \prime } } { W } \right) + \sum _ { j \in S _ { t } \backslash \{ 1 \} } \frac { 1 } { W } \log \frac { W ^ { \prime } } { W } } } \\ { { \displaystyle \quad = \frac { e ^ { \tilde { \varepsilon } } \tilde { \varepsilon } } { W } + \log \frac { W ^ { \prime } } { W } } . } \end{array}
$$

To bound the term with the logarithm, observe that $\log ( 1 / y ) \leq 1 / y - 1$ for every $y > 0$ . Thus, we have

$$
\log \frac { W ^ { \prime } } { W } = \log \frac { | S _ { t } | } { W } \leq \frac { | S _ { t } | } { W } - 1 = \frac { 1 - e ^ { \tilde { \varepsilon } } } { W } ,
$$

where we used the fact that $W ^ { \prime } = | S _ { t } |$ and $W = | S _ { t } | - 1 + e ^ { \tilde { \varepsilon } }$ . It follows that

$$
\begin{array} { r l } & { \mathrm { D } _ { \mathrm { K L } } \left( P [ \cdot \mid S _ { t } ] \mid | Q [ \cdot \mid S _ { t } ] \right) \leq \displaystyle \frac { e ^ { \tilde { \varepsilon } } \tilde { \varepsilon } } { W } + \frac { 1 - e ^ { \tilde { \varepsilon } } } { W } } \\ & { \quad \quad \quad \quad \quad = \frac { e ^ { \tilde { \varepsilon } } \tilde { \varepsilon } + 1 - e ^ { \tilde { \varepsilon } } } { W } } \\ & { \quad \quad \quad \quad \quad \leq \frac { \tilde { \varepsilon } ^ { 2 } } { | S _ { t } | } , } \end{array}
$$

using the inequality $W \geq | S _ { t } |$ and observing that

$$
e ^ { x } x + 1 - e ^ { x } = e ^ { x } ( x - 1 ) + 1 \leq x ^ { 2 } { \mathrm { ~ f o r ~ } } x \in ( 0 , 1 / 2 ) .
$$

The calculations for scenario (b) are similar and yield

$$
\begin{array} { r l } { \mathrm { D } _ { \mathrm { R L } } ( P [ \cdot | S _ { \mathrm { t } } ] | | Q [ \cdot | S _ { \mathrm { t } } ] ) = \log \frac { W ^ { \prime } } { W } - \frac { \tilde { \mathcal { E } } } { W } } \\ & { = \log ( 1 + \frac { e ^ { - \tilde { \mathcal { E } } } - 1 } { | S _ { \mathrm { t } } | } ) - \frac { \tilde { \mathcal { E } } } { | S _ { \mathrm { t } } | } } \\ & { \stackrel { ( \mathrm { i } ) } { \le } \frac { e ^ { \tilde { \mathcal { E } } } - 1 } { | S _ { \mathrm { t } } | } - \frac { \tilde { \mathcal { E } } } { | S _ { \mathrm { t } } | } } \\ & { = \frac { e ^ { \tilde { \mathcal { E } } } - 1 - \tilde { \mathcal { E } } } { | S _ { \mathrm { t } } | } } \\ & { \stackrel { ( \mathrm { i i } ) } { \le } \frac { \tilde { \mathcal { E } } ^ { 2 } + \tilde { \mathcal { E } } - \tilde { \mathcal { E } } } { | S _ { \mathrm { t } } | } } \\ & { = \frac { \tilde { \mathcal { E } } ^ { 2 } } { | S _ { \mathrm { t } } | } , } \end{array}\tag{30}
$$

where (i) is due to $\log ( 1 + x ) \leq x$ for $x > 0$ , whereas for (ii), we relied again on the fact that $e ^ { \tilde { \varepsilon } } - 1 \leq \tilde { \varepsilon } + \tilde { \varepsilon } ^ { 2 }$

Now, observe that $\mathbb { E } _ { \boldsymbol { \nu } , \boldsymbol { u } } \left[ \operatorname { D } _ { \mathrm { K L } } \left( P [ \cdot \mid S _ { t } ] \mid \mid Q [ \cdot \mid S _ { t } ] \right) \right]$ can be computed by averaging the KL divergence inside the expectation over all choices of $S _ { t }$ , where weights are induced by P. Using the bounds derived in the three cases above and the fact that $\tilde { \varepsilon } ^ { 3 } \leq \tilde { \varepsilon } ^ { 2 }$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { \nu , u } \left[ \mathrm { D } _ { \mathrm { K L } } \left( P [ \cdot | \ S _ { t } ] \| Q [ \cdot | S _ { t } ] \right) \right] \leq P \left[ 1 , 2 \in S _ { t } \right] \frac { \tilde { \varepsilon } ^ { 2 } + \tilde { \varepsilon } ^ { 3 } } { | S _ { t } | } + P \left[ \left| \{ 1 , 2 \} \cap S _ { t } \right| = 1 \right] \frac { \tilde { \varepsilon } ^ { 2 } } { | S _ { t } | } } \\ & { \qquad \leq \left( 1 - P [ 1 , 2 \not \in S _ { t } ] \right) \frac { 2 \tilde { \varepsilon } ^ { 2 } } { | S _ { t } | } , } \end{array}\tag{31}
$$

where, with abuse of notation, $P ( E )$ denotes the probability of event E under the PL model associated with u. Next, let us observe that

$$
P [ 1 \in S _ { t } ] = \prod _ { \ell = 1 } ^ { t - 1 } \frac { k - \ell } { k - \ell + e ^ { \tilde { \varepsilon } } } ,
$$

where $k - \ell + e ^ { \tilde { \varepsilon } } \geq k - \ell + 1$ for every $\ell \in [ t - 1 ]$ . This implies

$$
P [ 1 \in S _ { t } ] \leq \prod _ { \ell = 1 } ^ { t - 1 } { \frac { k - \ell } { k - \ell + 1 } } = { \frac { k - t + 1 } { k } } = { \frac { | S _ { t } | } { k } } .
$$

On the other hand, one has

$$
P [ 2 \in S _ { t } ] \leq \prod _ { \ell = 1 } ^ { t - 1 } { \frac { k - ( \ell + 1 ) + e ^ { \tilde { \varepsilon } } } { k - \ell + e ^ { \tilde { \varepsilon } } } } = { \frac { | S _ { t } | + e ^ { \tilde { \varepsilon } } - 1 } { k + e ^ { \tilde { \varepsilon } } - 1 } } .
$$

By using a union bound, we have

$$
( 1 - P \left[ 1 , 2 \notin S _ { t } \right] ) = P [ 1 \in S _ { t } \mathrm { ~ o r ~ } 2 \in S _ { t } ] \leq \frac { | S _ { t } | } { k } + \frac { | S _ { t } | + e ^ { \tilde { \varepsilon } } - 1 } { k + e ^ { \tilde { \varepsilon } } - 1 } .
$$

Since $\tilde { \varepsilon } > 0 , k + e ^ { \tilde { \varepsilon } } - 1 > k ;$ moreover, $| S _ { t } | \geq 1$ . Therefore, we can bound

$$
\frac { | S _ { t } | + e ^ { \tilde { \varepsilon } } - 1 } { k + e ^ { \tilde { \varepsilon } } - 1 } \leq \frac { | S _ { t } | ( 1 + e ^ { \tilde { \varepsilon } } - 1 ) } { k } = \frac { e ^ { \tilde { \varepsilon } } | S _ { t } | } { k } .
$$

From this, we obtain

$$
P [ 1 \in S _ { t } \mathrm { ~ o r ~ } 2 \in S _ { t } ] \leq \frac { | S _ { t } | ( 1 + e ^ { \tilde { \varepsilon } } ) } { k } \leq \frac { 3 | S _ { t } | } { k } ,
$$

where we used the fact that $e ^ { x } + 1 \leq 3$ for every $x \in ( 0 , 1 / 2 )$ . Together with (31), this allows us to derive our final bound

$$
\mathbb { E } _ { \boldsymbol { \nu } , \boldsymbol { u } } \left[ \mathrm { D } _ { \mathrm { K L } } \left( P [ \cdot | S _ { t } ] \left| \right| Q [ \cdot | S _ { t } ] \right) \right] \leq \frac { 6 \tilde { \varepsilon } ^ { 2 } } { k } .
$$

Combining this with expression (29) yields

$$
\begin{array} { r l } { \displaystyle \operatorname { D } _ { \mathrm { K L } } \big ( P \| Q \big ) \leq \sum _ { t = 1 } ^ { k - 1 } \frac { 6 \tilde { \varepsilon } ^ { 2 } } { k } } & { } \\ { \displaystyle = \frac { 6 ( k - 1 ) \tilde { \varepsilon } ^ { 2 } } { k } } & { } \\ { \displaystyle \leq 6 \tilde { \varepsilon } ^ { 2 } . } \end{array}
$$

By recalling that $\tilde { \varepsilon } = \varepsilon / \beta$ , we finally get

$$
\mathrm { D } _ { \mathrm { K L } } \left( P \left| \right| Q \right) \leq \frac { 6 \varepsilon ^ { 2 } } { \beta ^ { 2 } } = \frac { \pi ^ { 2 } \varepsilon ^ { 2 } } { V } .
$$

□

Let $\mathcal { A } ^ { \ast } = \left( T ^ { \star } , \hat { \sigma } ^ { \star } \right)$ be an $( \varepsilon , \delta ) – \mathrm { P A C }$ algorithm, and define the ${ \mathcal { F } } _ { T }$ ⋆-measurable event

$$
E = \left\{ \hat { \sigma } ^ { \star } ( 1 ) < \hat { \sigma } ^ { \star } ( 2 ) \right\} .
$$

Then

$$
\mathbb { P } _ { u } [ E ] \geq 1 - \delta \quad \mathrm { a n d } \quad \mathbb { P } _ { v } [ E ] \leq \delta .
$$

Let us assume that $\delta \in ( 0 , 1 / 4 ]$ , as the confidence parameter is typically chosen to be very close to zero. For $x \ge 1 - \delta$ and $y \leq \delta$ , the binary relative entropy satisfies

$$
d ( x , y ) \geq d ( 1 - \delta , \delta ) \geq \log \left( { \frac { 1 } { 3 \delta } } \right) .
$$

Using this, a direct application of Lemma 17 yields

$$
\mathbb { E } _ { \boldsymbol { \nu } , \boldsymbol { u } } [ T ^ { \star } ] \mathrm { D } _ { \mathrm { K L } } \left( P \left| \right| Q \right) \geq \log \left( \frac { 1 } { 3 \delta } \right) .
$$

Hence, by Lemma 10, we obtain

$$
\mathbb { E } _ { \nu , u } [ T ^ { \star } ] \geq \frac { V } { \pi ^ { 2 } \varepsilon ^ { 2 } } \log \left( \frac { 1 } { 3 \delta } \right) .
$$

In conclusion, since log $( 1 / ( 3 \delta ) ) \ge \frac { 1 } { 5 } \log ( 1 / \delta )$ for every $\delta \in ( 0 , 1 / 4 ]$ , we have

$$
\mathbb { E } _ { \boldsymbol { \nu } , \boldsymbol { u } } [ T ^ { \star } ] = \Omega \left( \frac { V } { \varepsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) .
$$

## D TECHNICAL RESULTS AND PROOFS FOR SECTION 5

## D.1 Concentration Bounds

The following result provides anytime-concentration bounds for the unbiased estimators introduced in Section 5. Our inequalities are based on (Maurer and Pontil, 2009). The proof of the lemma below is analogous to the one of Lemma 6.

Lemma 11. Fix $\delta \in ( 0 , 1 )$ and let $\delta _ { n }$ be as defined in Section 4.1. Then, with probability at least $1 - \delta$ , the following hold simultaneously for all items $i \in [ k ]$ and every $n \in \mathbb { N } \setminus \{ 1 \}$ :

$$
| P _ { i } - P _ { i } ^ { n } | \leq \sqrt { \frac { 2 V _ { i } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } ,\tag{32}
$$

$$
\left| \sqrt { V _ { i } ^ { n } } - \sqrt { P _ { i } ( 1 - P _ { i } ) } \right| \leq \sqrt { \frac { 2 \log ( 2 / \delta _ { n } ) } { n - 1 } } .\tag{33}
$$

Proof. For fixed $i \in [ k ]$ and $n \geq 2$ , we define the events

$$
A _ { i } ^ { n } : = \left\{ | P _ { i } - P _ { i } ^ { n } | \leq \sqrt { \frac { 2 V _ { i } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } \right\}
$$

and

$$
B _ { i } ^ { n } : = \left\{ \left| \sqrt { V _ { i } ^ { n } } - \sqrt { P _ { i } ( 1 - P _ { i } ) } \right| \leq \sqrt { \frac { 2 \log ( 2 / \delta _ { n } ) } { n - 1 } } \right\} .
$$

By Theorem 4 and Theorem 10 in (Maurer and Pontil, 2009), we have $\mathbb { P } \left[ A _ { i } ^ { n } \right] \geq 1 - \delta _ { n }$ and $\mathbb { P } \left[ B _ { i } ^ { n } \right] \geq 1 - \delta _ { n }$ . By taking a union bound, we obtain

$$
\begin{array} { r l } { \mathbb { P } \left[ \displaystyle \bigcap _ { n \in \mathbb { N } } \left( \displaystyle \bigcap _ { i \in [ k ] } \left( A _ { i } ^ { n } \cap B _ { i } ^ { n } \right) \right) \right] = 1 - \mathbb { P } \left[ \displaystyle \bigcup _ { n \in \mathbb { N } } \left( \displaystyle \bigcup _ { i \in [ k ] } \left( A _ { i } ^ { n } \right) ^ { c } \cup \left( B _ { i } ^ { n } \right) ^ { c } \right) \right] } & { } \\ { \geq 1 - k \displaystyle \sum _ { n = 1 } ^ { \infty } 2 \delta _ { n } } & { } \\ { = 1 - \displaystyle \frac { 6 \delta } { \pi ^ { 2 } } \displaystyle \sum _ { n = 1 } ^ { \infty } \displaystyle \frac { 1 } { n ^ { 2 } } } & { } \\ { \overset { ( 1 ) } { = } 1 - \delta , } \end{array}
$$

where (i) follows from the well-known fact that $\textstyle \sum _ { n \in \mathbb { N } } 1 / n ^ { 2 } = \pi ^ { 2 } / 6 .$

## D.2 Uniform Lower Bound on the Winner-Only Resolution Factor

We fix $\nu \in \mathcal { N } _ { V }$ and denote its density and CDF by $f$ and $F ,$ respectively, and let its standard deviation be $\sigma \in ( 0 , \sqrt { V } ]$ . Then, using the notation introduced in Section 5, we let $a : = \operatorname* { i n f } \operatorname { S u p p } ( f )$ and $b : = \operatorname* { s u p } \operatorname { S u p p } ( f )$ allowing infinite endpoints. By log-concavity, the interior of the support of $f$ is the open interval (a, b).

Lemma 12. For almost every $x \in \mathbb { R }$

$$
f ( x ) \geq { \frac { \operatorname* { m i n } \{ F ( x ) , 1 - F ( x ) \} } { { \sqrt { 3 } } \sigma } } \geq { \frac { F ( x ) ( 1 - F ( x ) ) } { { \sqrt { 3 } } \sigma } } .\tag{34}
$$

Proof. Let $z _ { \mathrm { 0 } }$ be the median of $\nu ,$ so that $F ( z _ { 0 } ) = 1 / 2$ . The hazard rate $f / ( 1 - F )$ is non-decreasing and the reversed hazard rate $f / F$ is non-increasing on (a, b) by log-concavity (Bagnoli and Bergstrom, 2005). Therefore,

$$
{ \frac { f ( x ) } { F ( x ) } } \geq 2 f ( z _ { 0 } ) \quad ( x \leq z _ { 0 } ) , \qquad { \frac { f ( x ) } { 1 - F ( x ) } } \geq 2 f ( z _ { 0 } ) \quad ( x \geq z _ { 0 } ) .
$$

The median-density bound $f ( z _ { 0 } ) \geq 1 / ( \sqrt { 1 2 } \sigma )$ follows from Bobkov and Ledoux (2019, Proposition B.2, Equation (B.6)). On the two sides of the median, the relevant denominator is respectively $F ( x )$ and $1 - F ( x )$ , which proves the first inequality. The second follows from min $\{ q , 1 - q \} \geq q ( 1 - q )$ for $q \in [ 0 , 1 ]$ . Outside $( a , b )$ the bounds hold almost everywhere because their right-hand sides vanish. □

The next lemma compares the winning probabilities of any two items and provide a lower bound on their diference in terms of the largest of the two.

Lemma 13. Let $\nu \in \mathcal { N } _ { V }$ have standard deviation $\sigma _ { . }$ , and let $\pmb { u } \in \mathbb { R } ^ { k } . I f u _ { i } > u _ { j }$ , then

$$
P _ { i } - P _ { j } \ge \frac { 1 - e ^ { - ( u _ { i } - u _ { j } ) / ( \sqrt { 3 } \sigma ) } } { 2 } P _ { i } .\tag{35}
$$

Proof. Set $t : = u _ { i } - u _ { i } > 0 , \bar { F } : = 1 - F$ , and let $z _ { 0 }$ be the median of ν. Define

$$
D _ { t } ( z ) : = f ( z ) - f ( z + t ) \frac { F ( z ) } { F ( z + t ) } , \qquad c _ { t } : = \frac { 1 - e ^ { - t / ( \sqrt { 3 } \sigma ) } } { 2 } ,
$$

assigning the ratio $F ( z ) / F ( z + t )$ the value zero when its denominator is zero. In that case $F ( z ) = 0$ and $f ( z ) = 0$ almost everywhere as well. Since $F$ is log-concave, $f / F$ is non-increasing on $( a , b )$ . Thus $D _ { t } ( z ) \geq 0$ almost everywhere in the support, including when $z + t$ is beyond its upper endpoint. Outside the support, $D _ { t } ( z ) = 0$ almost everywhere. Also, $F ( z ) \leq F ( z + t )$ implies that the following holds for almost every $z \in \mathbb { R } \mathrm { : }$

$$
D _ { t } ( z ) \geq f ( z ) - f ( z + t ) .
$$

We now prove that, for every $\alpha \in \mathbb { R } :$

$$
\int _ { \alpha } ^ { \infty } D _ { t } ( z ) d z \geq c _ { t } \bar { F } ( \alpha ) .\tag{36}
$$

To this end, let us first suppose that $\alpha \geq z _ { 0 } . { \mathrm { ~ H ~ } } \bar { F } ( \alpha ) = 0 $ , the claim follows from non-negativity. Otherwise, for almost every $z \geq \alpha$ in the support, Lemma 12 gives

$$
\frac { f ( z ) } { \bar { F } ( z ) } \geq \frac { 1 } { \sqrt { 3 } \sigma } .
$$

If $\alpha + t < b ,$ , integration over $[ \alpha , \alpha + t ]$ yields

$$
\log \frac { \bar { F } ( \alpha + t ) } { \bar { F } ( \alpha ) } = - \int _ { \alpha } ^ { \alpha + t } \frac { f ( z ) } { \bar { F } ( z ) } d z \le - \frac { t } { \sqrt { 3 } \sigma } .
$$

Hence $\bar { F } ( \alpha + t ) \le e ^ { - t / ( \sqrt { 3 } \sigma ) } \bar { F } ( \alpha )$ . The same inequality holds when $\alpha + t \geq b .$ , because then $\bar { F } ( \alpha + t ) = 0$ Consequently,

$$
\begin{array} { r l r } {  { \int _ { \alpha } ^ { \infty } D _ { t } ( z ) d z \ge \bar { F } ( \alpha ) - \bar { F } ( \alpha + t ) = } } \\ & { } & { \ge \big ( 1 - e ^ { - t / ( \sqrt { 3 } \sigma ) } \big ) \bar { F } ( \alpha ) = 2 c _ { t } \bar { F } ( \alpha ) \ge c _ { t } \bar { F } ( \alpha ) . } \end{array}
$$

If $\alpha < z _ { 0 }$ , non-negativity and the preceding bound at $z _ { 0 }$ give

$$
\int _ { \alpha } ^ { \infty } D _ { t } ( z ) d z \geq \int _ { z _ { 0 } } ^ { \infty } D _ { t } ( z ) d z \geq 2 c _ { t } { \bar { F } } ( z _ { 0 } ) = c _ { t } \geq c _ { t } { \bar { F } } ( \alpha ) .
$$

This proves Equation (36).

Let w be any bounded, non-negative, non-decreasing function on R, and set $M : = \operatorname* { s u p } _ { z } w ( z )$ . If we denote $L _ { y } : = \{ z : w ( z ) > y \}$ , its layer-cake representation is

$$
w ( z ) = \int _ { 0 } ^ { M } \mathbb { 1 } \{ z \in L _ { y } \} d y .
$$

Each $L _ { y }$ is an upper interval, possibly empty or all of R. Equation (36) applies to these sets as well, by absolute continuity and monotone convergence. Tonelli’s theorem therefore gives

$$
\begin{array} { r l r } {  { \int _ { \mathbb { R } } w ( z ) D _ { t } ( z ) d z = \int _ { 0 } ^ { M } \int _ { L _ { y } } D _ { t } ( z ) d z d y \ge } } \\ & { } & { \ge c _ { t } \int _ { 0 } ^ { M } \int _ { L _ { y } } f ( z ) d z d y = c _ { t } \int _ { \mathbb { R } } w ( z ) f ( z ) d z . } \end{array}
$$

Now choose

$$
w ( z ) : = F ( z + t ) \prod _ { \ell \in [ k ] \setminus \{ i , j \} } F ( z + u _ { i } - u _ { \ell } ) .
$$

Expressing both winning probabilities relative to $u _ { i }$ gives

$$
P _ { i } = \int _ { \mathbb R } w ( z ) f ( z ) d z , \qquad P _ { i } - P _ { j } = \int _ { \mathbb R } w ( z ) D _ { t } ( z ) d z .
$$

The second identity remains valid at points where $F ( z + t ) = 0 { : }$ the corresponding integrand for $P _ { j }$ also contains $F ( z ) = 0$ . Applying the preceding inequality yields $P _ { i } - P _ { j } \ge c _ { t } P _ { i }$ □

For an instance satisfying Assumption (WO1), $P _ { j } > 0$ . Since $\sigma \le \sqrt { V }$ , the preceding lemma implies

$$
\frac { P _ { i } } { P _ { j } } \geq \frac { 2 } { 1 + e ^ { - t / ( \sqrt { 3 } \sigma ) } } \geq \frac { 2 } { 1 + e ^ { - t / \sqrt { 3 V } } } = \xi ( t ) ,
$$

where $t = u _ { i } - u _ { j }$ and $\xi$ is the function given by Equation (11). Accordingly, the resolution factor in Equation (13) is also defined in terms of $V _ { ; }$ , rather than the unknown true variance.

Lemma 14. Fix $V > 0$ . The functions η and $\eta ^ { \star }$ in Equations (13) and (14) satisfy

$$
\eta ( t ) \geq \eta ^ { \star } ( t ) , \qquad t \in ( 0 , b - a ) .
$$

Proof. Set $h ( x ) : = ( 1 - e ^ { - x } ) / ( 3 + e ^ { - x } )$ for $x \geq 0$ . Its derivatives are

$$
h ^ { \prime } ( x ) = \frac { 4 e ^ { - x } } { ( 3 + e ^ { - x } ) ^ { 2 } } > 0 , \qquad h ^ { \prime \prime } ( x ) = - \frac { 4 e ^ { - x } ( 3 - e ^ { - x } ) } { ( 3 + e ^ { - x } ) ^ { 3 } } < 0 .
$$

Thus, $h$ is increasing on $[ 0 , \infty )$ and concave on the interval $[ 0 , 1 ]$ , with $h ( 0 ) = 0$ . By concavity on $[ 0 , 1 ]$ , we have $h ( x ) \geq x h ( 1 )$ , while monotonicity gives $h ( x ) \geq h ( 1 )$ for $x \geq 1$ . Since $h ( 1 ) = ( e - 1 ) / ( 3 e + 1 )$ and $\eta ( t ) = h ( t / \sqrt { 3 V } )$ ,

$$
\eta ( t ) \geq { \frac { e - 1 } { 3 e + 1 } } \operatorname* { m i n } \left\{ 1 , { \frac { t } { \sqrt { 3 V } } } \right\} = \eta ^ { \star } ( t ) .
$$

## D.3 Consistency of the Winning Probabilities

In the FR case, we showed that the pairwise preference probability $P _ { i j }$ is larger than $1 / 2$ if and only $u _ { i } > u _ { j }$ Here, we rely on an analogous result for the winning probabilities.

Lemma 15. Let $\nu \in \mathcal { N } _ { V }$ and $\mathbf { \boldsymbol { \mathscr { u } } } \in \mathbb { R } ^ { k }$ satisfy Assumption (WO1). Then $P _ { i } > 0$ for every $i \in [ k ]$ . Moreover, for any two items $i , j \in [ k ]$ with $u _ { i } \geq u _ { j }$

$$
P _ { i } > P _ { j } \Longleftrightarrow u _ { i } > u _ { j } .
$$

Proof. Let $( a , b )$ be the interior of the support of the density f of $\nu .$ Conditioning on the noise of item i gives

$$
P _ { i } = \int _ { \mathbb { R } } f ( z ) \prod _ { \ell \neq i } F ( z + u _ { i } - u _ { \ell } ) d z .
$$

Since $\Delta _ { u } < b - a$ , there is a nonempty open interval of $z \in ( a , b )$ on which $z + u _ { i } - u _ { \ell } > a$ for every $\ell \neq i .$ The integrand is strictly positive on that interval, so $P _ { i } > 0$ . If $u _ { i } = u _ { j }$ , exchangeability of their independent noises gives $P _ { i } = P _ { j }$ . If $u _ { i } > u _ { j }$ , Lemma 13 gives

$$
P _ { i } - P _ { j } \ge \frac { 1 - e ^ { - ( u _ { i } - u _ { j } ) / ( \sqrt { 3 } \sigma ) } } { 2 } P _ { i } > 0 ,
$$

where $\sigma ^ { 2 } = \mathbb { V } \mathrm { a r } _ { \nu } [ N ] > 0$ . This proves the equivalence.

## D.4 Sample Complexity Upper Bound

In this section, we derive the sample complexity upper bound given by Theorem 3 for the WO-VUR algorithm To do ${ \mathrm { s o } } ,$ we let $\delta _ { n }$ be defined as in Section 4.1 and define $\mathcal { E }$ as the event consisting of those outcomes for which the following hold simultaneously for all $i \in [ k ]$ and every $n \in \mathbb { N } \setminus \{ 1 \}$ :

$$
| P _ { i } - P _ { i } ^ { n } | \leq \sqrt { \frac { 2 V _ { i } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } ,\tag{37}
$$

$$
\left| \sqrt { V _ { i } ^ { n } } - \sqrt { P _ { i } ( 1 - P _ { i } ) } \right| \leq \sqrt { \frac { 2 \log ( 2 / \delta _ { n } ) } { n - 1 } } .\tag{38}
$$

Arguing exactly as in Appendix C.5, condition (38) allows us to replace the sample variance $V _ { i } ^ { n }$ by the true one, at the price of an extra additive term:

$$
\sqrt { \frac { 2 V _ { i } ^ { n } \log ( 4 / \delta _ { n } ) } { n } } + \frac { 7 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } \le \sqrt { \frac { 2 P _ { i } ( 1 - P _ { i } ) \log ( 4 / \delta _ { n } ) } { n } } + \frac { 1 3 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } .\tag{39}
$$

The left-hand side is the quantity appearing both in (37) and in the stopping rule (15), so that, wheneve $\mathcal { E }$ occurs, both can be controlled through the right-hand side.

First, we determine a positive integer $n _ { 0 }$ such that, for every $n \geq n _ { 0 }$ , the following holds for every $i \in [ k ]$

$$
\sqrt { \frac { 2 P _ { i } ( 1 - P _ { i } ) \log ( 4 / \delta _ { n } ) } { n } } + \frac { 1 3 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } \leq \frac { P _ { i } } { 2 } .\tag{40}
$$

By (39), this guarantees $| P _ { i } - P _ { i } ^ { n } | \leq P _ { i } / 2$ on E. The inequality is satisfied if the following hold simultaneously:

$$
\sqrt { \frac { 2 P _ { i } ( 1 - P _ { i } ) \log ( 4 / \delta _ { n } ) } { n } } < \frac { P _ { i } } { 4 } \quad \mathrm { a n d } \quad \frac { 1 3 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } < \frac { P _ { i } } { 4 } .
$$

For these conditions to be satisfied, we can enforce the stronger requirement:

$$
n > \frac { 3 2 \log ( 4 / \delta _ { n } ) } { P _ { i } } .
$$

Since we want this to hold for every $i \in [ k ]$ , let us define $\begin{array} { r } { P _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { i \in [ k ] } P _ { i } } \end{array}$ , and let $n _ { 0 }$ be the minimum positive integer n such that

$$
n \geq \operatorname* { m a x } \left\{ 2 , \frac { 3 2 \log ( 4 / \delta _ { n } ) } { P _ { \operatorname* { m i n } } } \right\} .
$$

It follows that

$$
n _ { 0 } \le \frac { 3 2 \log ( 4 / \delta _ { n _ { 0 } } ) } { P _ { \mathrm { m i n } } } + 1 \le \frac { 1 2 8 } { P _ { \mathrm { m i n } } } \log \left( 2 \pi n _ { 0 } \sqrt { \frac { k } { 3 \delta } } \right) .
$$

By Lemma 18, there exists a universal constant $C _ { 1 } > 0$ such that

$$
n _ { 0 } \le \frac { C _ { 1 } } { P _ { \operatorname* { m i n } } } \log \left( \frac { C _ { 1 } } { P _ { \operatorname* { m i n } } } \cdot \frac { k } { \delta } \right) .
$$

Now assume that $n > n _ { 0 }$ . If E occurs, then the condition given by (40) ensures $P _ { i } < 2 P _ { i } ^ { n }$ for every $i \in [ k ]$ , so that $\begin{array} { r } { \frac { \eta ^ { \star } } { 2 } P _ { i } ^ { n } > \frac { \eta ^ { \star } } { 4 } P _ { i } } \end{array}$ . In this regime, using (39) once more, the stopping condition of A is fulfilled if the following hold for every $\bar { i } \in [ k ]$ :

$$
\sqrt { \frac { 2 P _ { i } ( 1 - P _ { i } ) \log ( 4 / \delta _ { n } ) } { n } } < \frac { P _ { i } } { 8 } \eta ^ { \star } \quad \mathrm { a n d } \quad \frac { 1 3 \log ( 4 / \delta _ { n } ) } { 3 ( n - 1 ) } < \frac { P _ { i } } { 8 } \eta ^ { \star } .\tag{41}
$$

In particular, these hold simultaneously for every $i \in [ k ]$ if the following stronger requirement is met:

$$
n \geq \frac { 2 5 6 } { P _ { \mathrm { m i n } } \eta ^ { \star 2 } } \log \left( \frac { 4 } { \delta _ { n } } \right) .
$$

If we define $n _ { 1 }$ as the minimum positive integer $n > n _ { 0 }$ such that

$$
n \geq \left\lceil \frac { 2 5 6 \log ( 4 / \delta _ { n } ) } { P _ { \operatorname* { m i n } } \eta ^ { \star 2 } } \right\rceil ,
$$

which exists since the right-hand side grows logarithmically in $n ,$ then either $n _ { 1 } = n _ { 0 } + 1$ or the inequality above fails at $n _ { 1 } - 1$ , in which case $n _ { 1 } \leq 2 5 6 \log ( 4 / \delta _ { n _ { 1 } } ) / ( P _ { \operatorname* { m i n } } \eta ^ { \star 2 } ) + 2$ . In both cases, we can apply the same reasoning as above to find

$$
n _ { 1 } \le \frac { C _ { 2 } } { P _ { \mathrm { m i n } } \eta ^ { \star 2 } } \log \left( \frac { C _ { 2 } } { P _ { \mathrm { m i n } } \eta ^ { \star 2 } } \cdot \frac { k } { \delta } \right) ,
$$

for some universal constant $C _ { 2 } > 0 .$

Up to re-tuning $C _ { 1 }$ and $C _ { 2 }$ , our derivations show that, under event $\mathcal { E } ,$ the stopping condition is met within a number of steps equal to

$$
T _ { 0 } : = \left\lceil \frac { C _ { 2 } } { P _ { \mathrm { m i n } } \eta ^ { \star 2 } } \log \left( \frac { C _ { 2 } } { P _ { \mathrm { m i n } } \eta ^ { \star 2 } } \cdot \frac { k } { \delta } \right) \right\rceil .
$$

Therefore, we have

$$
\begin{array} { r } { \mathbb { E } \left[ T \right] = \mathbb { E } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } } \right] + \mathbb { E } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \right] \leq T _ { 0 } \cdot \mathbb { P } \left[ \mathcal { E } \right] + \mathbb { E } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \right] . } \end{array}
$$

Using the tail-sum formula for the expectation, we can rewrite

$$
\mathbb { E } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \right] = \sum _ { n = 1 } ^ { \infty } \mathbb { P } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \geq n \right] \leq T _ { 0 } \cdot \mathbb { P } \left[ \mathcal { E } ^ { c } \right] + \sum _ { n = T _ { 0 } + 1 } ^ { \infty } \mathbb { P } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \geq n \right] .
$$

Now, observe that:

$$
T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \geq n \Longleftrightarrow T \geq n \mathrm { a n d } \mathcal { E } ^ { c } \mathrm { o c c u r s } .
$$

However, we showed above that if $n \geq T _ { 0 } + 1$ and $T \geq n$ , then necessarily $\mathcal { E } ^ { c }$ must have occurred, as otherwise we would have found $T \leq T _ { 0 }$ . Thus, for every $n \geq T _ { 0 } + 1$ , we have

$$
\mathbb { P } \left[ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \geq n \right] = \mathbb { P } \left[ T \geq n \right] .
$$

If $T \geq n ,$ , then the stopping rule was not satisfied at time $n - 1$ . This implies that at least one of the two conditions given by (37) and (38) was violated for at least one item $i \in [ k ]$ . By a union bound over the k items and the two conditions, this occurs with probability at most $2 k \delta _ { n - 1 }$ . Hence, we have

$$
\begin{array} { r l r } {  { \sum _ { n = T _ { 0 } + 1 } ^ { \infty } \mathbb { P } [ T \cdot \mathbb { 1 } _ { \mathcal { E } ^ { c } } \geq n ] \leq 2 k \sum _ { n = T _ { 0 } + 1 } ^ { \infty } \delta _ { n - 1 } } } \\ & { } & { = \delta \cdot \sum _ { n = T _ { 0 } } ^ { \infty } \frac { 6 } { \pi ^ { 2 } n ^ { 2 } } } \\ & { } & { \leq C _ { 3 } , } \end{array}
$$

where $C _ { 3 } > 0$ is a universal constant. This implies $\mathbb { E } \left[ T \right] \leq T _ { 0 } + C _ { 3 }$

In conclusion, there exists a universal constant $C > 0$ such that

$$
\mathbb { E } \left[ T \right] \leq \frac { C } { { P _ { \mathrm { m i n } } \eta \star ^ { 2 } } } \log { \left( \frac { C } { { P _ { \mathrm { m i n } } \eta \star ^ { 2 } } } \cdot \frac { k } { \delta } \right) } .
$$

Using the definition of $\eta ^ { \star }$ , when $\sqrt { 3 V } \geq \varepsilon$ , the inequality above can be rewritten as

$$
\mathbb { E } \left[ T \right] \leq \frac { C \cdot V } { P _ { \mathrm { m i n } } \varepsilon ^ { 2 } } \log \left( \frac { C \cdot V } { P _ { \mathrm { m i n } } \varepsilon ^ { 2 } } \cdot \frac { k } { \delta } \right) .
$$

## D.5 Information-theoretic Lower Bound

Finally, we prove Theorem 4, thus deriving an information-theoretic lower bound for the ranking problem under WO feedback.

For every $\delta \in ( 0 , 1 )$ , algorithm $\mathcal { A } ^ { \ast }$ recovers an accurate representation of the utility ranking with probability at least $1 - \delta .$ , provided that Assumptions (V) and $\mathrm { ( W O 1 ) }$ are satisfied. In other words, if u and ν are chosen so that these conditions are met, then:

$$
\begin{array} { r } { \mathbb { P } \left[ \hat { \sigma } ^ { \star } \mathrm { ~ i s ~ } \varepsilon \mathrm { - a c c u r a t e ~ f o r ~ } \sigma _ { u } | \mathrm { R U M } ( \pmb { u } , \nu ) \right] \geq 1 - \delta . } \end{array}
$$

We now construct two specific instances of the problem, and we use them to provide a lower bound on the sample complexity of $\ b { A } ^ { * }$ . Throughout, we assume $k \geq 3$ . To do so, choose the minimum-probability level $p \in ( 0 , 1 / k )$

and set an accuracy level $\varepsilon \in ( 0 , b - a )$ . As we shall see, the latter must be chosen “suficiently small”, which, in this case, means $\varepsilon \in ( 0 , \beta )$ . However, this restriction does not invalidate our sample complexity result due to its asymptotic nature.

Then, let ν be the centered Gumbel distribution with scale and location parameters given by

$$
\beta = \frac { \sqrt { 6 V } } { \pi } \quad \mathrm { a n d } \quad \mu = - \beta \gamma ,
$$

where $\gamma$ denotes the Euler–Mascheroni constant. In particular, this ensures that ν is centered and that its variance is equal to $V .$ . Since the location parameter shifts all sampled utilities by the same constant, the implied winning probabilities do not depend on $\mu .$

Now, consider the utility vectors u and v given by:

$$
\bullet u _ { 1 } = \varepsilon / 2 , u _ { 2 } = - \varepsilon / 2 , u _ { 3 } = \cdot \cdot \cdot = u _ { k - 1 } = 0 , { \mathrm { a n d ~ } } u _ { k } = L .
$$

$$
\bullet v _ { 1 } = - \varepsilon / 2 , v _ { 2 } = \varepsilon / 2 , v _ { 3 } = \cdots = v _ { k - 1 } = 0 , { \mathrm { a n d } } v _ { k } = L .
$$

Here $k \geq 3$ guarantees that item $k ,$ which carries the utility $L ,$ is distinct from items 1 and $2 ;$ for $k = 3$ the intermediate block is empty, and the two instances reduce to $( \varepsilon / 2 , - \varepsilon / 2 , L )$ and $( - \varepsilon / 2 , \varepsilon / 2 , L )$ . Up to choosing ε suficiently small, the positive constant $L$ is chosen so that the minimum winning probability of both instances is equal to $p .$ We let $P$ and $Q$ be the winning distributions corresponding to u and $v ,$ that is,

$$
P _ { i } : = \frac { e ^ { u _ { i } / \beta } } { \sum _ { \ell = 1 } ^ { k } e ^ { u _ { \ell } / \beta } } \quad \mathrm { a n d } \quad Q _ { i } : = \frac { e ^ { v _ { i } / \beta } } { \sum _ { \ell = 1 } ^ { k } e ^ { v _ { \ell } / \beta } } \quad \forall i \in [ k ] .
$$

Lemma 16. Let P and $Q$ be the winning distributions introduced above. Then, the KL divergence between $P$ and $Q$ is upper bounded as

$$
\operatorname { D } _ { \mathrm { K L } } \left( P \left| \right| Q \right) \leq { \frac { \varepsilon ^ { 2 } \pi ^ { 2 } } { 6 V } } \cdot C \cdot p ,
$$

where $C > 0$ is a universal constant.

Proof. By direct calculations, we obtain

$$
\operatorname { D } _ { \mathrm { K L } } \left( P \left| \right| Q \right) = { \frac { \varepsilon } { \beta } } ( P _ { 1 } - P _ { 2 } ) .
$$

Up to choosing $\varepsilon \in ( 0 , \beta )$ , there exists a universal constant $C > 0$ such that

$$
P _ { 1 } - P _ { 2 } \leq { \frac { \varepsilon } { \beta } } \cdot C \cdot P _ { 2 } .
$$

Due to our choice of L, $P _ { 2 } = P _ { \mathrm { m i n } } = p .$ , thus implying

$$
\operatorname { D } _ { \mathrm { K L } } \left( P \left| \right| Q \right) \leq \left( { \frac { \varepsilon } { \beta } } \right) ^ { 2 } \cdot C \cdot p = { \frac { \varepsilon ^ { 2 } \pi ^ { 2 } } { 6 V } } \cdot C \cdot p .
$$

In order to employ Lemma $^ { 1 7 , }$ we think of our setting as a trivial multi-armed bandit problem, in which the number of arms is $n = 1$ . At time $t ,$ the only available arm is played. The observed reward is the feedback signal, which consists of the winner of the current round. The two bandit models for the problem are $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ and $\operatorname { R U M } ( \pmb { v } , \nu )$ , where u and v are the ones introduced above. Since the noise distribution is of the Gumbel type, all winning probabilities are non-zero. Thus, the two reward distributions are mutually absolutely continuous, as needed in order to apply Lemma 17.

Now, let $\mathcal { E }$ be the event that item 1 is ranked before item 2. Since the output ranking is an F<sub>T</sub>-measurable function, ${ \mathcal { E } } \in { \mathcal { F } } _ { T }$ . If we define $\mathbb { P } _ { u }$ and $\mathbb { P } _ { v }$ as the probability measures induced by $\mathrm { R U M } ( { \boldsymbol { \mathbf { \mathit { u } } } } , \nu )$ and $\operatorname { R U M } ( \pmb { v } , \nu )$ respectively, a direct application of Lemma 17 yields

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol { \nu } , \boldsymbol { u } } [ T ^ { \star } ] \mathrm { D } _ { \mathrm { K L } } \left( P \left. Q \right) \geq d ( \mathbb { P } _ { \boldsymbol { u } } ( \boldsymbol { \mathcal { E } } ) , \mathbb { P } _ { \boldsymbol { v } } ( \boldsymbol { \mathcal { E } } ) \right) . } \end{array}
$$

Let us assume that $\delta \in ( 0 , 1 / 4 ]$ . Algorithm $\ b { A } ^ { * }$ is $( \varepsilon , \delta ) – \mathrm { P A C } ,$ , so

$$
\mathbb { P } _ { u } ( \mathcal { E } ) \geq 1 - \delta \quad \mathrm { a n d } \quad \mathbb { P } _ { v } ( \mathcal { E } ) \leq \delta .
$$

Due to the properties of the binary relative entropy, we have

$$
d ( \mathbb { P } _ { u } ( \mathcal { E } ) , \mathbb { P } _ { v } ( \mathcal { E } ) ) \geq d ( 1 - \delta , \delta ) \geq \log \left( \frac { 1 } { 3 \delta } \right) .
$$

It follows that

$$
\mathbb { E } _ { \boldsymbol { \nu } , \boldsymbol { u } } [ T ^ { \star } ] \geq \frac { 1 } { \operatorname { D } _ { \mathrm { K L } } \left( P \left| \right| Q \right) } \log \left( \frac { 1 } { 3 \delta } \right) .
$$

Owing to Lemma 16, we conclude that

$$
\mathbb { E } _ { \nu , { \boldsymbol { u } } } [ T ^ { \star } ] \geq \frac { 6 V } { C p \pi ^ { 2 } \varepsilon ^ { 2 } } \log \left( \frac { 1 } { 3 \delta } \right) .
$$

In conclusion, since $\begin{array} { r } { \log ( 1 / ( 3 \delta ) ) \ge \frac { 1 } { 5 } \log ( 1 / \delta ) } \end{array}$ for every $\delta \in ( 0 , 1 / 4 ]$ , we have shown that

$$
\mathbb { E } _ { \boldsymbol { \nu } , \boldsymbol { u } } [ T ^ { \star } ] = \Omega \left( \frac { V } { p \varepsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) .
$$

## E AUXILIARY RESULTS

## E.1 Change-of-measure Lemma for Multi-armed Bandit Problems

The next result restates Lemma 1 of (Kaufmann et al., 2016). As this is typically used in the context of multi-armed bandit problems, we need to introduce some preliminary notation. The number of arms is any positive integer $n ,$ and we identify the arm set with [n]. At time t, let $A _ { t }$ and $Z _ { t }$ be the (possibly random) arm being chosen and the corresponding reward, respectively. Let $\mathcal { F } _ { t } = \sigma ( A _ { 1 } , Z _ { 1 } , \ldots , A _ { t } , Z _ { t } )$ be the sigma-algebra generated by the trajectory followed up to time t. For every $i \in [ n ]$ , we define $N _ { i } ( t )$ as the number of times that arm i was played up to time t. We use the subscripts $\mathbb { P } _ { \nu }$ and $\mathbb { E } _ { \nu }$ to refer to the probability measure induced by a given bandit model $\nu$ and the implied expectation functional.

Lemma 17. Let $\nu = ( \nu _ { i } ) _ { i = 1 } ^ { n }$ and $\nu ^ { \prime } = ( \nu _ { i } ^ { \prime } ) _ { i = 1 } ^ { n }$ be two bandit models, where $\nu _ { i }$ (resp. $\nu _ { i } ^ { \prime } )$ is the reward distribution of arm $i \in [ n ]$ . Assume that, for every $i \in [ n ]$ , the probability distributions $\nu _ { i }$ and $\nu _ { i } ^ { \prime }$ are equivalent, $i . e . ,$ , they are mutually absolutely continuous. For any almost-surely finite stopping time τ with respect to the filtration $( \mathcal { F } _ { t } ) _ { t \in \mathbb { N } } .$ the following holds:

$$
\sum _ { i = 1 } ^ { n } \mathbb { E } _ { \boldsymbol { \nu } } [ N _ { i } ( \tau ) ] \mathrm { D } _ { \mathrm { K L } } \left( \boldsymbol { \nu } _ { i } \parallel \boldsymbol { \nu } _ { i } ^ { \prime } \right) \geq \operatorname* { s u p } _ { E \in \mathcal { F } _ { \tau } } d \left( \mathbb { P } _ { \boldsymbol { \nu } } [ E ] , \mathbb { P } _ { \boldsymbol { \nu } ^ { \prime } } [ E ] \right) ,
$$

where $d ( x , y ) : = x \log ( x / y ) + ( 1 - x ) \log ( ( 1 - x ) / ( 1 - y ) )$ is the binary relative entropy, with the convention that $d ( 0 , 0 ) = d ( 1 , 1 ) = 0$

As we shall motivate later, the setting considered in this lemma encompasses ours by choosing the number of arms equal to one.

## E.2 Technical Result for Bounding the Sample Complexity

The next result is a technical lemma used to derive the sample complexity upper bounds given by Theorem 1 and Theorem 3. A more general version can be found as Lemma 12 in (Jonsson et al., 2020). The slightly modified version presented below is suficient for our purposes.

Lemma 18. Let $n \in \mathbb { N }$ and $a , b > 0$ . Suppose that $n \leq a \log ( b n )$ . Then

$$
n \leq 2 a \log \left( a b \right) .
$$

Proof. For every $x > 0 , \log ( x ) \leq { \sqrt { x } }$ . Therefore, we have

$$
n \leq a \log ( b n ) \leq a { \sqrt { b n } } = a { \sqrt { b } } { \sqrt { n } } ,
$$

from which we obtain

$$
{ \sqrt { n } } \leq a { \sqrt { b } } .
$$

Equivalently, $n \leq a ^ { 2 } b$ . Since the logarithm is increasing and $b > 0 .$ , we have

$$
n \leq a \log ( b n ) \leq a \log \left( b ( a ^ { 2 } b ) \right) = 2 a \log ( a b ) .
$$

## E.3 Deviation Bound for Symmetric Unimodal Distributions

$$
\mathbb { P } \left[ Z \geq t \right] \leq \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { 2 } ( 1 - \frac { t } { \sqrt { 3 } \sigma } ) } & { \displaystyle i f 0 \leq t < 2 \sigma / \sqrt { 3 } } \\ { \displaystyle \frac { 2 \sigma ^ { 2 } } { 9 t ^ { 2 } } } & { \displaystyle i f t \geq 2 \sigma / \sqrt { 3 } } \end{array} \right. .
$$

The following is a standard result providing useful deviation bounds for symmetric unimodal distributions. The version for standardized random variables can be found as Theorem 7.1 in (Ion et al., 2022).

Lemma 19. Let Z be a symmetric unimodal random variable with $\mathbb { V } a r [ Z ] = \sigma ^ { 2 } > 0$ . Then

(42)