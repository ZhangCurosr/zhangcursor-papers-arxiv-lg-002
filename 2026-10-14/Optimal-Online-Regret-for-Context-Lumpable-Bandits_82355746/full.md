# Optimal Online Regret for Context-Lumpable Bandits

Chenyu Lu<sup>∗1</sup> and Zihan Zhang<sup>1</sup>

<sup>1</sup>The Hong Kong University of Science and Technology

## Abstract

We study stochastic contextual bandits with an unknown context-lumpable structure. The S contexts are partitioned into r blocks, and every arm has the same mean reward across contexts in the same block. Under arbitrary context distributions and without separation assumptions, we design an algorithm with expected regret $\widetilde { O } \left( \sqrt { r ( S + K ) T } \right)$ , matching the lower bound of Lee et al. (2023) up to logarithmic factors and improving their previous upper bound by a factor of r.

The algorithm operates over dyadic accuracy levels, using a baseline policy and confidencebased early stopping to control exploration. It then performs cross-context elimination: since rewards are identical within a block, one selected arm eliminates all surviving records from that block, requiring at most r selections at each level. Multi-scale exploration assigns finer accuracies to frequent blocks and coarser accuracies to rare blocks, yielding a near-optimal arm for every block while keeping the weighted policy error small.

## 1 Introduction

In contextual bandits, the learner observes a context, selects an arm, and receives feedback only for that arm. In recommendation, for example, feedback is observed only for displayed items (Li et al. 2010). Sharing observations across contexts can reduce learning cost, but the relation between their rewards may be unknown.

In context-lumpable stochastic bandits (Lee et al. 2023), S contexts form $r \leq \operatorname* { m i n } \{ S , K \}$ unknown blocks. Each of the K arms has the same mean reward within a block. Observations can thus be shared within blocks, but the learner sees context identities without their block assignments. It must find useful arms and learn where they perform well from bandit feedback. Running a separate bandit for each context gives expected regret $\widetilde { \cal O } ( \sqrt { S K T } )$ over T rounds (Lattimore and Szepesv´ari 2020), which can be wasteful when r is small. Yet exploration across contexts incurs regret: an informative arm may be poor where it is sampled. Finding a good policy and minimizing regret are therefore distinct challenges. Lee et al. (2023) established nearly optimal sample complexity for finding a near-optimal policy of $\begin{array} { r } { \widetilde O \left( \frac { r ( S + K ) } { \epsilon ^ { 2 } } \right) } \end{array}$ , while a sub-optimal regret bound of $\widetilde { O } ( \sqrt { r ^ { 3 } ( S + K ) T } )$ . Their regret algorithm refines a context partition while eliminating arms; ensuring that refinements do not split a true block incurs an extra factor of r. They left $\widetilde { O } ( \sqrt { r ( S + K ) T } )$ regret open. We attain this rate by controlling exploration during candidate-arm selection, without recovering the partition.

## 1.1 Our Results

For every context distribution ν and its induced block distribution $\omega _ { \mathrm { { i } } }$ , our algorithm achieves expected regret

$$
\mathrm { R e g } _ { T } = \widetilde { O } \Big ( \sqrt { r ( S + K ) T } \Big ) ,
$$

where ${ \widetilde { O } } ( \cdot )$ hides polylogarithmic factors in S, $K , r ,$ and $T .$ For a known horizon, this guarantee is achieved by Algorithm $4 ;$ a standard doubling restart gives the same order without knowing $T$

This rate matches the $\Omega ( \sqrt { r ( S + K ) T } )$ lower bound of Lee et al. (2023) up to logarithmic factors. It improves their online upper bound $\widetilde { O } ( \sqrt { r ^ { 3 } ( S + K ) T } )$ to $\widetilde { O } ( \sqrt { r ( S + K ) T } )$ , removing an additional factor of r. The guarantee holds for arbitrary block and context distributions and does not require any separation between the reward vectors of diferent blocks. Conceptually, the algorithm exploits the equality of mean rewards within each block without first recovering the unknown context partition: it constructs a shared set of candidate arms that contains a suficiently good arm for every block and then selects among these candidates separately at each context.

The first technical ingredient is a phased exploration procedure with geometrically decreasing accuracy levels. In phase $h ,$ , the algorithm targets accuracy $\varepsilon _ { h }$ and first estimates the preceding policy $\pi _ { h - 1 }$ at every context, obtaining a context-wise baseline $B _ { h } ( i )$ . It then runs the Collect procedure at levels $\varepsilon _ { n } = 2 ^ { - n / 2 }$ for $n = 0 , \ldots , h$ . At level $n ,$ each context receives $q _ { h , n } \asymp r K \varepsilon _ { n } / ( S \varepsilon _ { h } )$ random arm assignments, and each assignment is sampled up to $m _ { h , n } \asymp 1 / \varepsilon _ { n } ^ { 2 }$ times. Sampling stops as soon as the arm’s upper confidence bound falls below the baseline by a margin of order $\varepsilon _ { n }$ . On the joint concentration event, every arm stopped in this way is worse than the preceding policy by a constant multiple of $\varepsilon _ { n } .$ , whereas every arm that completes its budget has an $\varepsilon _ { n } \cdot$ -accurate estimate. Thus, the algorithm avoids repeatedly paying for assignments that have already been certified to be harmful, while the multiple accuracy levels allow it to search for good arms at a scale adapted to the frequency of each block.

The second ingredient is a multi-scale, cross-context elimination rule. At each level, the algorithm selects the surviving context-arm record with the largest estimated reward, evaluates the selected arm across all contexts, and removes every record for which the selected arm or the preceding policy is suficiently better. Since all contexts in the selected arm’s true block have the same mean reward, one selection removes all remaining records from that block; consequently, at most r selections are needed at any level, even though the block partition is never identified. In the uniform-context analysis, define

$$
\varepsilon _ { h , b } = \operatorname* { m a x } \Biggl \{ 1 , \frac { 1 } { r \omega ( b ) } \Biggr \} \varepsilon _ { h } .
$$

For every block with $\varepsilon _ { h , b } \leq 1$ , the assignment budget ensures that its optimal arm is represented at

some level satisfying $\varepsilon _ { h , b } \leq \varepsilon _ { n } < \sqrt { 2 } \varepsilon _ { h , b }$ . The elimination step then leaves the candidate set with an arm whose gap on block b is $O ( \varepsilon _ { h , b } )$ . The final finest-scale update, which retains $\pi _ { h - 1 }$ as a fallback, therefore satisfies

$$
\operatorname* { m a x } _ { j \in [ K ] } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) \leq \operatorname* { m i n } \{ 1 , 1 0 \varepsilon _ { h , b } \} , \qquad i \in B ( b ) .
$$

Although small blocks are learned only at coarser accuracy, their errors are weighted by their smaller arrival probabilities:

$$
\sum _ { b = 1 } ^ { r } \omega ( b ) \varepsilon _ { h , b } = \varepsilon _ { h } \sum _ { b = 1 } ^ { r } \operatorname* { m a x } \Bigl \{ \omega ( b ) , \frac { 1 } { r } \Bigr \} \le 2 \varepsilon _ { h } .
$$

Hence the phase policy has average gap $O ( \varepsilon _ { h } )$ , while early stopping and the at most r eliminations per level keep the exploration cost at $\widetilde { O } ( r ( S + K ) / \varepsilon _ { h } )$ . Balancing this cost with phases of length $\widetilde { O } ( r ( S + K ) / \varepsilon _ { h } ^ { 2 } )$ and summing over the geometric sequence of phases yields the stated regret bound. For an arbitrary context distribution, Algorithm 4 first groups contexts with comparable estimated arrival probabilities, applies the same procedure separately within each approximately uniform group, discards only a low-frequency set of small total probability mass, and combines the group guarantees by Cauchy–Schwarz inequality.

## 1.2 Related Work

Contextual and latent bandits. Linear contextual bandits share reward parameters through known features (Chu et al. 2011), whereas Maillard and Mannor (2014) studied latent bandits with clustered rewards. Online clustering for linear bandits has been studied under parameter separation (Gentile et al. 2014) and with item-dependent groups (Gentile et al. 2017). Kausik et al. (2025) learned a shared reward-parameter subspace from ofline data for online linear latent contextual bandits. We assume neither such features nor ofline data.

Bandits with identical reward vectors. Pal et al. (2023) combined clustering and matrix completion for identical within-cluster reward vectors under a uniform context distribution. Their nearoptimal regret guarantee assumes incoherence, bounded condition number, and a column-subset condition; their main presentation treats the number of clusters and their size ratio as constants. Lee et al. (2023) studied context lumpability without these matrix conditions. We follow this setting, with explicit dependence on the number of blocks and arbitrary block and context distributions.

Low-rank bandits. A reward matrix with at most r distinct rows has rank at most r, so context lumpability is a special case of low-rank structure. Jun et al. (2019) studied bilinear bandits with known features. Lu et al. (2021) and Kang et al. (2022) considered low-rank generalized linear bandits, where actions have known matrix features and the unknown parameter is low-rank. For linear rewards, Jang et al. (2024) used experimental design to obtain regret bounds that depend on the geometry of the arm set. Other work addresses heavy-tailed rewards (Kang et al. 2024) and minimax lower bounds for low-rank matrix parameters (Zeng and Honorio 2025).

For low-rank reward matrices, Stojanovic et al. (2023) combined spectral estimation with arm elimination when the learner can choose any matrix entry. Closer to our setting, Jedra et al. (2024) studied randomly arriving contexts using singular-subspace estimation and a reduction to misspecified linear bandits. Our analysis instead exploits equality of mean rewards within each block, a stronger assumption than low rank alone, without estimating a factorization or singular subspaces.

## 2 Notation and problem formulation

Write $[ n ] = \{ 1 , \dots , n \}$ , and take all logarithms to be natural. A context-lumpable bandit (Lee et al. 2023) has S contexts, K arms, and r nonempty blocks, where $1 \leq r \leq \operatorname* { m i n } \{ S , K \}$ . An unknown map $g : [ S ]  [ r ]$ defines the blocks $B ( b ) = \{ i \in [ S ] : g ( i ) = b \}$ . For an unknown $\mu : [ r ] \times [ K ]  \mathbb { R }$ the mean reward matrix satisfies

$$
A ( i , j ) = \mu ( g ( i ) , j ) , \qquad ( i , j ) \in [ S ] \times [ K ] .
$$

The learner knows $S , K , r .$ , but not g or A. No positive separation between the mean-reward vectors of distinct blocks is assumed.

At round t, a context $i _ { t }$ is drawn from an unknown, fixed context distribution ν on [S], independently of the preceding history. The learner observes $i _ { t }$ , chooses $j _ { t } \in [ K ]$ , and receives $y _ { t } = A ( i _ { t } , j _ { t } ) + \eta _ { t }$ We assume, almost surely,

$$
\begin{array} { r l } & { \mathbb { E } [ \eta _ { t } \mid i _ { 1 } , j _ { 1 } , y _ { 1 } , \dotsc , i _ { t } , j _ { t } ] = 0 , } \\ & { \mathbb { E } [ e ^ { \lambda \eta _ { t } } \mid i _ { 1 } , j _ { 1 } , y _ { 1 } , \dotsc , i _ { t } , j _ { t } ] \leq e ^ { \lambda ^ { 2 } / 2 } \quad ( \lambda \in \mathbb { R } ) . } \end{array}\tag{1}
$$

Here and below, conditioning also includes the learner’s randomness revealed before the current reward. Thus, $\eta _ { t }$ is conditionally 1-sub-Gaussian given the information available immediately before $y _ { t }$ . The mean rewards have unit span:

$$
\operatorname* { m a x } _ { j \in [ K ] } A ( i , j ) - \operatorname* { m i n } _ { j \in [ K ] } A ( i , j ) \leq 1 , \qquad i \in [ S ] .\tag{2}
$$

Consequently, every play incurs at most one unit of regret, including rounds used only for initial exploration.

The block distribution is given by $\begin{array} { r } { \omega ( b ) = \sum _ { i \in \mathcal { B } ( b ) } \nu ( i ) } \end{array}$ ; it is unknown to the learner. Under a uniform context distribution, $\omega ( b ) = | B ( b ) | / S$ . We impose no restriction on the block distribution.

The arrival probability $\nu ( i )$ controls how often an arm can be tested at context $i ,$ whereas $\omega ( b )$ controls how often a choice on block b afects expected regret. Under a uniform context distribution, a smaller block ofers fewer opportunities to find a good arm, but choosing a suboptimal arm there also has less influence on regret. This balance motivates the use of multiple sampling accuracies below.

A policy is a map $\pi : [ S ] \to [ K ]$ . For each block b, fix $j _ { b } ^ { \star } \in \arg \operatorname* { m a x } _ { j \in [ K ] } \mu ( b , j )$ . The expected regret over $T \geq 1$ rounds is

$$
\mathrm { R e g } _ { T } = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( \operatorname* { m a x } _ { j \in [ K ] } A ( i _ { t } , j ) - A ( i _ { t } , j _ { t } ) ) \right] .
$$

The expectation includes contexts, rewards, and the learner’s randomness. The notation ${ \widetilde { O } } ( \cdot )$ hides universal constants and factors polylogarithmic in $S , K , r , T$ . Algorithm 1 does not require T; Algorithm 4 uses T, while Theorem 2 also gives a horizon-free version.

## 3 Regret under a Uniform Context Distribution

## 3.1 Online Algorithm

We now describe our algorithm in detail. As shown in Algorithm 1, it proceeds in phases indexed by dyadic precision levels. Each phase consists of the following four steps.

Parameter settings. Set $\varepsilon _ { n } = 2 ^ { - n / 2 }$ for $n \geq 0$ . In phase $h \geq 1$ , we use $\beta _ { h } = 2 0 \log ( 1 6 S K r 2 ^ { h } )$ assignment counts $q _ { h , n } = \lceil \beta _ { h } r K \varepsilon _ { n } / ( S \varepsilon _ { h } ) \rceil$ and sample budgets $m _ { h , n } = \lceil 8 \beta _ { h } / \varepsilon _ { n } ^ { 2 } \rceil$ for $n \in \{ 0 , \ldots , h \}$ and $L _ { h } ( k ) = \lceil 1 2 8 S ( k + \beta _ { h } ) \rceil$ . Write $m _ { h } = { m _ { h , h } }$ for the finest-level sample budget.

Step 1. We first estimate the mean reward of the preceding policy at each context using Baseline. These estimates provide a reference for Collect to stop sampling an arm when its reward is suficiently below the baseline, avoiding unnecessary regret (Lemma 1).

Step 2. Since the block distribution is unknown, we explore all accuracy levels $n = 0 , \ldots , h$ . Coarser levels assign more arms to each context but collect fewer rewards per arm, increasing the chance of sampling an optimal arm from a small block. Finer levels provide more accurate estimates. For each level, $D _ { h , n }$ records the context-arm pairs that complete sampling without being stopped. This allows optimal arms from diferent blocks to be sampled at suitable levels (Lemmas 2–4).

Step 3. For each level, we select the arm $j ^ { \star }$ with the largest estimated reward in $D _ { h , n }$ and evaluate it across all contexts. We remove records for which either $j ^ { \star }$ or the preceding policy achieves a reward suficiently close to the selected estimate. Since contexts in the same block have identical mean rewards, each iteration removes all remaining records from at least one true block. Thus, at most r iterations are needed per level, and the resulting candidate set $W _ { h }$ contains a near-optimal arm for every block at an accuracy depending on its size, without recovering the block partition (Lemma 6).

Step 4. Finally, we evaluate the arms in $W _ { h }$ at the finest level and select the arm with the largest estimated reward for each context, retaining the preceding policy as an option. Although smaller blocks may have larger estimation errors, they also appear less frequently. Consequently, the resulting

Algorithm 1 Online Algorithm for Arbitrary Block Distributions   
Let t denote the current time and initialize $W _ { 0 } \gets \{ 1 \} , \pi _ { 0 } ( i ) \gets 1$ for $i \in [ S ]$   
for phase $h = 1 , 2 , \ldots$ do   
$t _ { h } \gets t , W _ { h } \gets W _ { h - 1 }$   
Step 1. Baseline estimation   
$B _ { h } \gets \mathrm { B a s e l i n e } ( h , \pi _ { h - 1 } )$   
Step 2. Random arm exploration   
for $n = 0 , \ldots , h$ do   
$Q _ { h , n } ( i ) \stackrel { \mathrm { i . i . d . } } { \sim } \operatorname { U n i f } ( [ K ] ^ { q _ { h , n } } )$ for $i \in [ S ]$   
$( D _ { h , n } , \widehat { A } _ { h , n } ) \gets \mathrm { C o l l e c t } ( h , n , Q _ { h , n } , \pi _ { h - 1 } , B _ { h } )$   
Step 3. Candidate selection   
for $n = 0 , \ldots , h$ do   
for $\ell = 1 , \ldots , r$ do   
if $D _ { h , n } = \mathcal { O }$ then   
break   
$( i ^ { \star } , j ^ { \star } ) \in$ arg ma $\mathfrak { c } _ { ( i , j ) \in D _ { h , n } } \widehat { A } _ { h , n } ( i , j ) , W _ { h }  W _ { h } \cup \{ j ^ { \star } \}$   
Set $Q ( i ) = \{ j ^ { \star } \}$ for all $i ; ( D , \widehat { A } ) \gets \mathrm { C o l l e c t } ( h , n , Q , \pi _ { h - 1 } , B _ { h } )$   
$\widehat { V } ( i , \ j ^ { \star } ) = \Big \{ \operatorname* { m a x } \{ B _ { h } ( i ) , \widehat { A } ( i , j ^ { \star } ) \} , \quad ( i , j ^ { \star } ) \in D ,$ $i \in [ S ]$   
<sub></sub>B<sub>h</sub>(i), $( i , j ^ { \star } ) \notin D ,$   
$D _ { h , n }  \{ ( i , j ) \in D _ { h , n } : \widehat { V } ( i , j ^ { \star } ) < \widehat { A } _ { h , n } ( i ^ { \star } , j ^ { \star } ) - 4 \varepsilon _ { n } \}$   
Step 4. Policy update   
Set $Q ( i ) = W _ { h }$ for all $i ; ( D , \widehat { A } ) \gets \mathrm { C o l l e c t } ( h , h , Q , \pi _ { h - 1 } , B _ { h } )$   
$D \gets D \cup \{ ( i , \pi _ { h - 1 } ( i ) ) : i \in [ S ] \} , \widehat { A } ( i , \pi _ { h - 1 } ( i ) ) \gets B _ { h } ( i )$ for all i   
$\pi _ { h } ( i )  \mathrm { a r g }$ max ${ } _ { \cdot j : ( i , j ) \in D } \widehat { A } ( i , j )$ for all i   
while $t - t _ { h } < \left\lceil 2 ^ { 1 6 } \beta _ { h } ^ { 3 } r ( S + K ) / \varepsilon _ { h } ^ { 2 } \right\rceil$ do   
Receive $i _ { t } ;$ play $\pi _ { h } ( i _ { t } )$

policy satisfies

$$
\frac { 1 } { S } \sum _ { i = 1 } ^ { S } \left[ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) \right] \leq 2 0 \varepsilon _ { h }
$$

on the event guaranteed by Lemma 5 (Lemma 7). Together with early stopping, this allows us to control the exploration regret in each phase and obtain the bound in Theorem 1.

## 3.2 Subroutines

The online algorithm uses two subroutines, Baseline and Collect, described below.

Baseline. In phase h, Baseline plays the preceding policy $\pi _ { h - 1 }$ for $L _ { h } ( \boldsymbol { m } _ { h } )$ rounds. Under a uniform context distribution, this duration ensures that every context appears at least $m _ { h }$ times with probability at least $1 - S e ^ { - 8 \beta _ { h } }$ . Averaging the first $m _ { h }$ observed rewards at each context i gives

$B _ { h } ( i )$ , an estimate of $A ( i , \pi _ { h - 1 } ( i ) )$ . Since $m _ { h } = \lceil 8 \beta _ { h } / \varepsilon _ { h } ^ { 2 } \rceil$ , concentration guarantees

$$
| B _ { h } ( i ) - A ( i , \pi _ { h - 1 } ( i ) ) | \le \varepsilon _ { h } / 2 , \qquad i \in [ S ] ,
$$

with high probability (Lemma 1).

Algorithm 2 Baseline $( h , \pi )$   
$n _ { i } \gets 0 , B _ { h } ( i ) \gets 0$ for $i \in [ S ]$   
for $\tau = 1 , \ldots , L _ { h } ( m _ { h } )$ do   
Receive $i _ { t } ;$ play π(i<sub>t</sub>) and observe y<sub>t</sub>   
if $n _ { i _ { t } } < m _ { h }$ then $B _ { h } ( i _ { t } ) \gets \frac { n _ { i _ { t } } B _ { h } ( i _ { t } ) + y _ { t } } { n _ { i _ { t } } + 1 } , n _ { i _ { t } } \gets n _ { i _ { t } } + 1$   
return $B _ { h } = ( B _ { h } ( i ) ) _ { i \in [ S ] }$

Algorithm 3 Collect $( h , n , Q , \pi , B _ { h } )$   
$D \gets \emptyset , a _ { i } \gets 1 , n _ { i } \gets 0$ for $i \in [ S ]$   
$\widehat { A } _ { 0 } ( i , j ) \gets 0$ for $( i , j ) \in [ S ] \times [ K ]$   
for $\tau = 1 , \ldots , L _ { h } ( m _ { h , n }$ max<sub>i</sub> |Q(i)|) do   
Receive $i _ { t }$   
if $a _ { i _ { t } } > | Q ( i _ { t } ) |$ then   
Play $\pi ( i _ { t } )$   
else   
$j \gets Q ( i _ { t } ) _ { a _ { i _ { t } } } ;$ play j and observe $y _ { t }$   
$\widehat { A } _ { n _ { i _ { t } } + 1 } ( i _ { t } , j ) \gets \frac { n _ { i _ { t } } \widehat { A } _ { n _ { i _ { t } } } ( i _ { t } , j ) + y _ { t } } { n _ { i _ { t } } + 1 } , n _ { i _ { t } } \gets n _ { i _ { t } } + 1$   
if $j \neq \pi ( i _ { t } )$ and $\widehat { A } _ { n _ { i _ { t } } } ( i _ { t } , j ) + \sqrt { 2 \beta _ { h } / n _ { i _ { t } } } < B _ { h } ( i _ { t } ) - 3 \varepsilon _ { n }$ then   
$a _ { i _ { t } }  a _ { i _ { t } } + 1 , n _ { i _ { t } }  0$   
else   
if $n _ { i _ { t } } = m _ { h , n }$ then   
$D \gets D \cup \{ ( i _ { t } , j ) \} , \widehat { A } ( i _ { t } , j ) \gets \widehat { A } _ { m _ { h , n } } ( i _ { t } , j )$   
$a _ { i { \mathrm { \Omega } } } \gets a _ { i { \mathrm { \Omega } } } + 1 , n _ { i { \mathrm { \Omega } } } \gets 0$   
return $( D , { \widehat { A } } )$  
Collect. At level $n ,$ , Collect samples the arms in $Q ( i )$ one by one, with at most $m _ { h , n }$ observations per arm. The stopping rule ensures that whenever an arm $j$ is stopped early at context $i , A ( i , j ) \leq$ $\begin{array} { r } { A ( i , \pi _ { h - 1 } ( i ) ) - \frac { 5 } { 2 } \varepsilon _ { n } } \end{array}$ on the event $\Omega _ { h }$ (Lemmas 1–2). Thus, early stopping avoids further regret from arms that perform substantially worse than the preceding policy. Arms that are not stopped are recorded in $D _ { ; }$ together with their estimated mean rewards $\widehat { A } ( i , j )$ , after $m _ { h , n }$ observations. The duration $L _ { h } ( \boldsymbol { m } _ { h , n }$ max<sub>i</sub> $| Q ( i ) | )$ ensures that every context appears suficiently often to process all arms in $Q ( i )$ , with probability at least $1 - S e ^ { - 8 \beta _ { h } }$ (Lemma 3). Once all arms in $Q ( i )$ have been processed, Collect plays $\pi _ { h - 1 } ( i )$ whenever context i appears.

## 3.3 Regret Analysis

We now state our main result.

Theorem 1. Under a uniform context distribution ν and an arbitrary induced block distribution ω, Algorithm 1 requires no knowledge of T and satisfies ${ \mathrm { R e g } _ { T } } = \widetilde O \Big ( \sqrt { r ( S + K ) T } \Big )$

Proof sketch. The key is to bound the regret incurred by Collect. If an arm is much worse than the preceding policy, Collect stops sampling it early. Otherwise, its reward is not much lower than that of the preceding policy. Since $\pi _ { h - 1 }$ has an average gap of $O ( \varepsilon _ { h } )$ (Lemma 7), sampling one arm at every context at level n incurs at most $O ( S \beta _ { h } / \varepsilon _ { n } )$ regret. Summing over the arms sampled in Steps 2–4 gives $O ( \beta _ { h } ^ { 3 } r ( S + K ) / \varepsilon _ { h } )$

The remaining rounds play either $\pi _ { h - 1 }$ or $\pi _ { h }$ , both with average gap $O ( \varepsilon _ { h } )$ (Lemma 7). Since phase h lasts $O ( \beta _ { h } ^ { 3 } r ( S + K ) / \varepsilon _ { h } ^ { 2 } )$ rounds, these rounds contribute the same regret order. Lemma 5 bounds the regret from the events where these guarantees fail. Finally, using $\varepsilon _ { h } = 2 ^ { - h / 2 }$ and summing over phases yields $\mathrm { R e g } _ { T } = \widetilde { O } ( \sqrt { r ( S + K ) T } )$ ). The full proof is in Appendix A.

## 4 General Context Distributions

In this section we extend the result for a uniform context distribution to arbitrary context distributions in two steps.

## 4.1 Approximately Uniform Context Distributions

Corollary 1. Suppose $1 / ( 8 S ) \le \nu ( i ) \le 8 / S$ for all $i \in [ S ]$ and the induced block distribution is arbitrary. Algorithm 1 satisfies $\mathrm { R e g } _ { T } = \widetilde { O } \Big ( \sqrt { r ( S + K ) T } \Big )$ . The same holds when $r \leq \operatorname* { m i n } \{ S , K \}$ is only an upper bound on the number of nonempty blocks; neither ν nor T need be known.

Corollary 1 extends Theorem 1 to approximately uniform context distributions. The lower bound on $\nu ( i )$ ensures that Baseline and Collect receive suficiently many observations, while the upper bound allows the average policy regret to remain $O ( \varepsilon _ { h } )$ . Thus, the same regret rate is preserved up to constant factors. This extension allows Algorithm 1 to be applied to groups of contexts with similar probabilities, which will be used to handle general context distributions.

The proof is in Appendix B.1.

## 4.2 Arbitrary Context Distributions

For an arbitrary context distribution $\nu ,$ we adapt the frequency-based grouping approach of Lee et al. (2023, Algorithm 3 and Appendix F). Algorithm 4 partitions contexts according to their estimated arrival probabilities and runs Algorithm 1 separately within each retained group. With probability at least $1 - ( T + 1 ) ^ { - 2 }$ , each group’s conditional context distribution satisfies the assumptions of

Corollary 1, while the remaining low-frequency contexts have total probability at most $2 \sqrt { S / T }$ . For a given horizon $T ,$ define $J = \lceil 3 2 \sqrt { S T } \log ( 2 S ( T + 1 ) ^ { 2 } ) \rceil$ . We may assume $J \leq T$

```latex
Algorithm 4 Online Algorithm for General Context Distributions
Let t denote the current time and initialize $J \gets \lceil 3 2 \sqrt { S T } \log ( 2 S ( T + 1 ) ^ { 2 } ) \rceil , L \gets 1 + \lceil \log _ { 2 } \sqrt { S T } \rceil$
Play arm 1 for J rounds
$\begin{array} { r } { \widehat { \nu } ( i )  J ^ { - 1 } \sum _ { t = 1 } ^ { J } \mathbf { 1 } \{ i _ { t } = i \} } \end{array}$ for $i \in [ S ]$
${ \mathcal { X } } _ { \ell } \gets \{ i \in [ S ] : 2 ^ { - \ell - 1 } < { \widehat { \nu } } ( i ) \leq 2 ^ { - \ell } , { \widehat { \nu } } ( i ) \geq 1 / { \sqrt { S T } } \} \ \mathrm { f o r } \ \ell = 0 , \dots , L - 1$
$\mathcal { X } _ { L }  \{ i \in [ S ] : \widehat { \nu } ( i ) < 1 / \sqrt { S T } \}$
for each nonempty $\mathcal { X } _ { \ell }$ with $\ell < L$ do
Initialize Algorithm 1 on $\mathcal { X } _ { \ell } ,$ using
$S = | \mathcal { X } _ { \ell } |$ and r = min $\{ r , | \mathcal { X } _ { \ell } | \}$
for $t = J + 1 , \dots , T$ do
Receive $i _ { t }$
if $i _ { t } \in \mathcal { X } _ { L }$ then
Play arm 1
else
Find $\ell < L$ such that $i _ { t } \in \mathcal { X } _ { \ell }$
Use Algorithm 1 for $\mathcal { X } _ { \ell }$ to select an arm
Play the selected arm and observe the reward
Update only Algorithm 1 for $\mathcal { X } _ { \ell }$
```

Theorem 2. For an arbitrary context distribution ν and its induced block distribution ω, Algorithm $\it 4$ satisfies $\mathrm { R e g } _ { T } = \widetilde { O } \Big ( \sqrt { r ( S + K ) T } \Big )$ . Restarting Algorithm $\it 4$ on intervals of lengths $1 , 2 , 4 , \ldots$ . gives the same bound without knowing $\acute { T }$

The initial estimation rounds and the discarded contexts each contribute at most $\widetilde { O } ( \sqrt { S T } )$ expected regret. Applying Corollary 1 within each remaining group and summing by Cauchy–Schwarz gives the stated rate; see Appendix B.

## 5 Conclusions and Future Directions

We establish expected regret $\widetilde { O } ( \sqrt { r ( S + K ) T } )$ for context-lumpable stochastic bandits under arbitrary block and context distributions, matching the lower bound up to logarithmic factors. Rather than recovering the unknown block partition, our approach constructs a shared candidate set $W _ { h }$ containing a near-optimal arm for each block, and then selects a near-optimal arm for each context from this set. Together with early stopping to control exploration regret, this improves the dependence on r in the previous online regret bound without requiring separation between blocks.

Context-lumpable bandits can be viewed as a special case of lumpable Markov decision processes (MDPs) with i.i.d. state transitions. Extending our approach to general lumpable MDPs, where transitions may depend on the state and action, is a natural direction for future work.

## AI Disclosure

OpenAI’s ChatGPT and Codex were used to assist with language polishing, LaTeX formatting, compilation error checking. The authors take full responsibility for all theoretical results, proofs, and the final manuscript.

## References

Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payof functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 208–214. PMLR, 2011.

Claudio Gentile, Shuai Li, and Giovanni Zappella. Online clustering of bandits. In Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 757–765. PMLR, 2014.

Claudio Gentile, Shuai Li, Purushottam Kar, Alexandros Karatzoglou, Giovanni Zappella, and Evans Etrue. On context-dependent clustering of bandits. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1253–1262. PMLR, 2017.

Kyoungseok Jang, Chicheng Zhang, and Kwang-Sung Jun. Eficient low-rank matrix estimation, experimental design, and arm-set-dependent low-rank bandits. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 21329–21372. PMLR, 2024.

Yassir Jedra, William R´eveillard, Stefan Stojanovic, and Alexandre Proutiere. Low-rank bandits via tight two to-infinity singular subspace recovery. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 21430–21485. PMLR, 2024.

Kwang-Sung Jun, Rebecca Willett, Stephen Wright, and Robert Nowak. Bilinear bandits with low-rank structure. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3163–3172. PMLR, 2019.

Yue Kang, Cho-Jui Hsieh, and Thomas Chun Man Lee. Eficient frameworks for generalized low-rank matrix bandit problems. In Advances in Neural Information Processing Systems, volume 35, pages 19971–19983. Curran Associates, Inc., 2022. doi: 10.52202/068431-1452.

Yue Kang, Cho-Jui Hsieh, and Thomas Chun Man Lee. Low-rank matrix bandits with heavy-tailed rewards. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of Proceedings of Machine Learning Research, pages 1863–1889. PMLR, 2024.

Chinmaya Kausik, Kevin Tan, and Ambuj Tewari. Leveraging ofline data in linear latent contextual bandits. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 29345–29389. PMLR, 2025.

Tor Lattimore and Csaba Szepesv´ari. Bandit Algorithms. Cambridge University Press, 2020. doi: 10.1017/ 9781108571401.

Chung-Wei Lee, Qinghua Liu, Yasin Abbasi-Yadkori, Chi Jin, Tor Lattimore, and Csaba Szepesv´ari. Context lumpable stochastic bandits. In Advances in Neural Information Processing Systems, volume 36, pages 77915–77941. Curran Associates, Inc., 2023. doi: 10.52202/075280-3404.

Lihong Li, Wei Chu, John Langford, and Robert E. Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th International Conference on World Wide Web, WWW ’10, pages 661–670. ACM, 2010. doi: 10.1145/1772690.1772758.

Yangyi Lu, Amirhossein Meisami, and Ambuj Tewari. Low-rank generalized linear bandit problems. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 460–468. PMLR, 2021.

Odalric-Ambrym Maillard and Shie Mannor. Latent bandits. In Proceedings of the 31st International Con ference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 136–144. PMLR, 2014.

Soumyabrata Pal, Arun Sai Suggala, Karthikeyan Shanmugam, and Prateek Jain. Optimal algorithms for latent bandits with cluster structure. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 7540–7577. PMLR, 2023.

Stefan Stojanovic, Yassir Jedra, and Alexandre Proutiere. Spectral entry-wise matrix estimation for low rank reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 77056–77070. Curran Associates, Inc., 2023. doi: 10.52202/075280-3369.

Guo Zeng and Jean Honorio. A novel general framework for sharp lower bounds in succinct stochastic bandits. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 26678–26699. Curran Associates, Inc., 2025. doi: 10.52202/085713-0898.

## A Uniform Context Distribution and Arbitrary Block Distributions

All phase-h probability bounds below also hold conditional on the observations and learner’s random choices before round $t _ { h }$ . The letter C denotes an absolute constant that may change between inequalities. For the analysis, recall

$$
\varepsilon _ { h , b } = \operatorname* { m a x } \left\{ 1 , { \frac { S } { r | { \mathcal { B } } ( b ) | } } \right\} \varepsilon _ { h } = \operatorname* { m a x } \left\{ 1 , { \frac { 1 } { r \omega ( b ) } } \right\} \varepsilon _ { h } .\tag{3}
$$

The second equality uses $\omega ( b ) = | B ( b ) | / S$ . This choice follows Lee et al. (2023, Appendix G);   
Algorithm 1 does not use $\varepsilon _ { h , b }$ or the block distribution.

## A.1 Lemmas

Algorithm 1 ensures

$$
\pi _ { h - 1 } ( i ) \in W _ { h - 1 } \subseteq W _ { h } \subseteq [ K ] , \qquad | W _ { h } | \leq \operatorname* { m i n } \left\{ K , 1 + { \frac { r h ( h + 3 ) } { 2 } } \right\} \leq r ( h + 1 ) ^ { 2 } \leq r \beta _ { h } ^ { 2 } .\tag{4}
$$

Each phase ℓ adds at most r(ℓ + 1) arms, so $\begin{array} { r } { | W _ { h } | \leq 1 + r \sum _ { \ell = 1 } ^ { h } ( \ell + 1 ) } \end{array}$ . The parameters give

$$
\beta _ { h } \geq h + 1 , \qquad m _ { h , n } \leq 9 \beta _ { h } / \varepsilon _ { n } ^ { 2 } , \qquad S q _ { h , n } \leq \beta _ { h } r K \varepsilon _ { n } / \varepsilon _ { h } + S ,
$$

$$
\sum _ { n = 0 } ^ { h } \varepsilon _ { n } ^ { - 2 } = 2 ^ { h + 1 } - 1 \leq 2 \varepsilon _ { h } ^ { - 2 } , \qquad \sum _ { n = 0 } ^ { h } \varepsilon _ { n } ^ { - 1 } = \frac { 2 ^ { ( h + 1 ) / 2 } - 1 } { \sqrt { 2 } - 1 } \leq 4 \varepsilon _ { h } ^ { - 1 } .
$$

In particular,

$$
\sum _ { n = 0 } ^ { h } S q _ { h , n } m _ { h , n } \leq \frac { 3 6 \beta _ { h } ^ { 2 } r K + 1 8 S \beta _ { h } } { \varepsilon _ { h } ^ { 2 } } , \qquad \sum _ { n = 0 } ^ { h } m _ { h , n } \leq \frac { 1 8 \beta _ { h } } { \varepsilon _ { h } ^ { 2 } } .\tag{5}
$$

Lemma 1. The output of Baseline with input $( h , \pi )$ satisfies $| B _ { h } ( i ) - A ( i , \pi ( i ) ) | \le \varepsilon _ { h } / 2$ for every $i \in [ S ]$ , with probability at least

$$
1 - 2 S ( 1 6 S K r 2 ^ { h } ) ^ { - 2 0 } - S ( 1 6 S K r 2 ^ { h } ) ^ { - 1 6 0 } .
$$

Lemma 2. With probability at least $1 - 2 ^ { 1 8 } ( 1 6 S K r 2 ^ { h } ) ^ { - 1 8 }$ , every empirical mean computed by Collect during phase h satisfies

$$
| \widehat { A } _ { k } ( i , j ) - A ( i , j ) | \leq \sqrt { 2 \beta _ { h } / k } .
$$

For each entry $j$ in $Q ( i ) , k \geq 1$ counts its recorded rewards; the bound applies whenever that count is reached.

Lemma 3. With probability at least $1 - S [ ( r + 1 ) ( h + 1 ) + 1 ] ( 1 6 S K r 2 ^ { h } ) ^ { - 1 6 0 }$ , Collect ends with $a _ { i } > | Q ( i ) |$ for every $i \in [ S ]$ each time it is used in phase h.

Lemma 4. With probability at least $1 - r ( h + 1 ) ( 1 6 S K r 2 ^ { h } ) ^ { - 2 0 }$ , the random selections in Step 2 satisfy

$$
\exists i \in B ( b ) : j _ { b } ^ { \star } \in Q _ { h , n } ( i )
$$

simultaneously for every $b \in [ r ]$ and $n \in \{ 0 , \ldots , h \}$ for which $\varepsilon _ { n } \geq \varepsilon _ { h , b }$

Let $\Omega _ { h }$ be the event that the following hold simultaneously in phase h:

(i) Baseline returns $| B _ { h } ( i ) - A ( i , \pi _ { h - 1 } ( i ) ) | \le \varepsilon _ { h } / 2$ for every $i \in [ S ]$

(ii) Every empirical mean computed by Collect satisfies the inequality in Lemma 2.

(iii) Collect ends with $a _ { i } > | Q ( i ) |$ for every $i \in [ S ]$ each time it is used.

(iv) For every b, n with $\varepsilon _ { n } \geq \varepsilon _ { h , b }$ , some $i \in \mathcal { B } ( b )$ satisfies $j _ { b } ^ { \star } \in Q _ { h , n } ( i )$ in Step 2.

Lemma 5. For every phase h, $\mathrm { P r } ( \Omega _ { h } ) \geq 1 - ( 1 6 S K r 2 ^ { h } ) ^ { - 8 }$

Lemma 6. On $\Omega _ { h }$ , Step 3 empties each $D _ { h , n }$ within r iterations. The resulting candidate set satisfies

$$
\mu ( b , j _ { b } ^ { \star } ) - \operatorname* { m a x } _ { j \in W _ { h } } \mu ( b , j ) \leq \operatorname* { m i n } \{ 1 , 6 \sqrt { 2 } \varepsilon _ { h , b } \} , \qquad b \in [ r ] .
$$

Lemma 7. On $\Omega _ { h }$ , the policy returned in phase h satisfies

$$
\operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) \leq \operatorname* { m i n } \{ 1 , 1 0 \varepsilon _ { h , b } \} , \qquad i \in \mathcal { B } ( b ) , \ b \in [ r ] ,
$$

$$
\frac { 1 } { S } \sum _ { i = 1 } ^ { S } [ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) ] \leq 2 0 \varepsilon _ { h } .
$$

These inequalities hold simultaneously with probability at least $1 - ( 1 6 S K r 2 ^ { h } ) ^ { - 8 }$ , regardless of the preceding policy’s accuracy.

## A.2 Proof of Theorem 1

Using the parameter bounds preceding Lemma 1 and $L _ { h } ( k ) \le 2 7 2 S k$ for $k \geq \beta _ { h }$ , Steps 1–4 take at most

$$
\begin{array} { l } { \displaystyle { L _ { h } ( m _ { h } ) + \sum _ { n = 0 } ^ { h } L _ { h } ( q _ { h , n } m _ { h , n } ) + r \sum _ { n = 0 } ^ { h } L _ { h } ( m _ { h , n } ) + L _ { h } ( | W _ { h } | m _ { h } ) } } \\ { \displaystyle { \quad \le 2 7 2 S \left[ m _ { h } ( 1 + | W _ { h } | ) + \sum _ { n = 0 } ^ { h } ( q _ { h , n } + r ) m _ { h , n } \right] } } \\ { \displaystyle { \quad \le \frac { 2 ^ { 1 4 } \beta _ { h } ^ { 3 } r ( S + K ) } { \varepsilon _ { h } ^ { 2 } } . } } \end{array}\tag{6}
$$

The final loop fixes the phase length at

$$
t _ { h + 1 } - t _ { h } = \left\lceil \frac { 2 ^ { 1 6 } \beta _ { h } ^ { 3 } r ( S + K ) } { \varepsilon _ { h } ^ { 2 } } \right\rceil .\tag{7}
$$

Fix $h \geq 2$ . On $\Omega _ { h - 1 }$ , Lemma $7$ gives

$$
\frac { 1 } { S } \sum _ { i = 1 } ^ { S } [ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h - 1 } ( i ) ) ] \leq 2 0 \varepsilon _ { h - 1 } < 3 0 \varepsilon _ { h } .\tag{8}
$$

In Collect at level $n \leq h .$ , fix one entry j of $Q ( i )$ that receives $k \geq 1$ samples. If $k \geq 2$ and $j \neq \pi _ { h - 1 } ( i )$ the stopping inequality was false after sample $k - 1$ . On $\Omega _ { h }$ , Lemmas 1–2 give

$$
\begin{array} { r l } & { ( k - 1 ) [ A ( i , \pi _ { h - 1 } ( i ) ) - A ( i , j ) ] \leq ( k - 1 ) ( 3 \varepsilon _ { n } + \varepsilon _ { h } / 2 ) + 2 \sqrt { 2 \beta _ { h } ( k - 1 ) } } \\ & { \phantom { ( k - 1 ) [ } \leq \frac { 7 } { 2 } m _ { h , n } \varepsilon _ { n } + 2 \sqrt { 2 \beta _ { h } m _ { h , n } } . } \end{array}
$$

For $k = 1$ or $j = \pi _ { h - 1 } ( i )$ , the left-hand side is zero. Using (2) for the last sample and $k \le m _ { h , n } \le$ $9 \beta _ { h } / \varepsilon _ { n } ^ { 2 }$ gives

$$
\begin{array} { r l } {  { k [ \operatorname* { m a x } _ { j ^ { \prime } } A ( i , j ^ { \prime } ) - A ( i , j ) ] } \quad } & { } \\ & { \leq m _ { h , n } [ \operatorname* { m a x } _ { j ^ { \prime } } A ( i , j ^ { \prime } ) - A ( i , \pi _ { h - 1 } ( i ) ) ] + \frac { 7 } { 2 } m _ { h , n } \varepsilon _ { n } + 2 \sqrt { 2 \beta _ { h } m _ { h , n } } + 1 } \\ & { \leq \frac { 9 \beta _ { h } } { \varepsilon _ { n } ^ { 2 } } [ \operatorname* { m a x } _ { j ^ { \prime } } A ( i , j ^ { \prime } ) - A ( i , \pi _ { h - 1 } ( i ) ) ] + \frac { 4 1 \beta _ { h } } { \varepsilon _ { n } } . } \end{array}
$$

Summing over one assignment per context and using (8), we obtain, on $\Omega _ { h - 1 } \cap \Omega _ { h }$

$$
\begin{array} { r l } & { \sum _ { i = 1 } ^ { S } \left( \frac { 9 \beta _ { h } } { \varepsilon _ { n } ^ { 2 } } [ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h - 1 } ( i ) ) ] + \frac { 4 1 \beta _ { h } } { \varepsilon _ { n } } \right) } \\ & { \quad \leq S \beta _ { h } \left( \frac { 2 7 0 \varepsilon _ { h } } { \varepsilon _ { n } ^ { 2 } } + \frac { 4 1 } { \varepsilon _ { n } } \right) \leq \frac { 3 1 1 S \beta _ { h } } { \varepsilon _ { n } } . } \end{array}\tag{9}
$$

On $\Omega _ { h - 1 } \cap \Omega _ { h }$ , summing (9) over the entries tested in Steps 2–4 gives

$$
\mathrm { S t e p ~ 2 } \colon \ 3 1 1 S \beta _ { h } \sum _ { n = 0 } ^ { h } \frac { q _ { h , n } } { \varepsilon _ { n } } \leq C \frac { \beta _ { h } ^ { 2 } r K ( h + 1 ) + S \beta _ { h } } { \varepsilon _ { h } } ,
$$

$$
\begin{array} { r l } { \displaystyle \mathrm { S t e p ~ 3 } \colon } & { \displaystyle 3 1 1 S r \beta _ { h } \sum _ { n = 0 } ^ { h } \varepsilon _ { n } ^ { - 1 } \le \frac { C S r \beta _ { h } } { \varepsilon _ { h } } , } \\ { \displaystyle \mathrm { S t e p ~ 4 } \colon } & { \displaystyle \frac { 3 1 1 S \beta _ { h } | W _ { h } | } { \varepsilon _ { h } } \le \frac { C S r \beta _ { h } ^ { 3 } } { \varepsilon _ { h } } . } \end{array}
$$

Here Step 2 tests $q _ { h , n }$ entries per context, Step 3 tests at most r per level, and Step 4 tests $| W _ { h } | \le r \beta _ { h } ^ { 2 }$ We used $h + 1 \leq \beta _ { h }$ and $\textstyle \sum _ { n = 0 } ^ { h } \varepsilon _ { n } ^ { - 1 } \leq 4 / \varepsilon _ { h }$

All other plays use $\pi _ { h - 1 } \ \mathrm { o r } \ \pi _ { h }$ . Since $\Omega _ { h - 1 }$ and $\pi _ { h - 1 }$ are fixed before $t _ { h }$ , independence of context arrivals and (8) give

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathbf { 1 } _ { \Omega _ { h - 1 } } \sum _ { t = t _ { h } } ^ { t _ { h + 1 } - 1 } \left( \operatorname* { m a x } _ { j } A ( i _ { t } , j ) - A ( i _ { t } , \pi _ { h - 1 } ( i _ { t } ) ) \right) \right] } \\ & { \quad = \mathbb { E } \left[ \mathbf { 1 } _ { \Omega _ { h - 1 } } \frac { t _ { h + 1 } - t _ { h } } { S } \sum _ { i } ( \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h - 1 } ( i ) ) ) \right] \leq 3 0 \varepsilon _ { h } ( t _ { h + 1 } - t _ { h } ) . } \end{array}
$$

This bounds the plays of $\pi _ { h - 1 }$ after dropping $\mathbf { 1 } _ { \Omega _ { h } }$ and including all rounds of the phase. In the final loop, $\Omega _ { h }$ and $\pi _ { h }$ are determined before the next context arrives. The same calculation with Lemma $7$ bounds those plays by $2 0 \varepsilon _ { h } ( t _ { h + 1 } - t _ { h } )$ . Therefore

$$
\begin{array} { r l } {  { \mathbb { E } [ \mathbf { 1 } _ { \Omega _ { h - 1 } \cap \Omega _ { h } } \sum _ { t = t _ { h } } ^ { t _ { h + 1 } - 1 } ( \operatorname* { m a x } _ { j } A ( i _ { t } , j ) - A ( i _ { t } , j _ { t } ) ) ] } } \\ & { \leq \frac { C \beta _ { h } ^ { 3 } r ( S + K ) } { \varepsilon _ { h } } + 5 0 \varepsilon _ { h } ( t _ { h + 1 } - t _ { h } ) \leq \frac { C \beta _ { h } ^ { 3 } r ( S + K ) } { \varepsilon _ { h } } . } \end{array}
$$

By Lemma 5, $\mathrm { P r } ( ( \Omega _ { h - 1 } \cap \Omega _ { h } ) ^ { c } ) \leq 2 5 7 ( 1 6 S K r 2 ^ { h } ) ^ { - 8 }$ . On this complement, regret is at most the phase length by (2). Thus

$$
\begin{array} { r l r } {  { \mathbb { E } [ \sum _ { t = t _ { h } } ^ { t _ { h + 1 } - 1 } ( \operatorname* { m a x } _ { j } A ( i _ { t } , j ) - A ( i _ { t } , j _ { t } ) ) ] } } \\ & { } & { \leq C \beta _ { h } ^ { 3 } r ( S + K ) 2 ^ { h / 2 } + \frac { 2 5 7 ( t _ { h + 1 } - t _ { h } ) } { ( 1 6 S K r 2 ^ { h } ) ^ { 8 } } . } \end{array}
$$

Since $( \log x ) ^ { 3 } \leq 2 x$ for $x \geq 1$ , we have $\beta _ { h } ^ { 3 } \le 1 6 0 0 0 ( 1 6 S K r 2 ^ { h } )$ . Consequently,

$$
\sum _ { h = 2 } ^ { \infty } \frac { 2 5 7 ( t _ { h + 1 } - t _ { h } ) } { ( 1 6 S K r 2 ^ { h } ) ^ { 8 } } \le C \frac { r ( S + K ) } { ( 1 6 S K r ) ^ { 7 } } \sum _ { h = 2 } ^ { \infty } 2 ^ { - 6 h } = O ( 1 ) .\tag{10}
$$

Let H satisfy $t _ { H } \leq T < t _ { H + 1 }$ . Completing the last phase only increases regret, so

$$
\mathrm { R e g } _ { T } \leq \mathrm { m i n } \{ T , t _ { 2 } - t _ { 1 } \} + C r ( S + K ) \sum _ { h = 2 } ^ { H } \beta _ { h } ^ { 3 } 2 ^ { h / 2 } + O ( 1 ) .
$$

The first term satisfies

$$
\operatorname* { m i n } \{ T , t _ { 2 } - t _ { 1 } \} \leq \sqrt { ( t _ { 2 } - t _ { 1 } ) T } \leq C \beta _ { 1 } ^ { 3 / 2 } \sqrt { r ( S + K ) T } .
$$

If the sum is nonempty, phase $H - 1$ is complete and

$$
2 ^ { 1 6 } \beta _ { H - 1 } ^ { 3 } r ( S + K ) 2 ^ { H - 1 } \le t _ { H } - t _ { H - 1 } < T , \qquad \beta _ { H } \le 2 \beta _ { H - 1 } .
$$

It follows that

$$
\begin{array} { r l r } {  { r ( S + K ) \sum _ { h = 2 } ^ { H } \beta _ { h } ^ { 3 } 2 ^ { h / 2 } \le 4 \beta _ { H } ^ { 3 } r ( S + K ) 2 ^ { H / 2 } } } \\ & { } & { \le \frac { 4 \beta _ { H } ^ { 3 } } { 2 ^ { 1 5 / 2 } \beta _ { H - 1 } ^ { 3 / 2 } } \sqrt { r ( S + K ) T } \le C \beta _ { H } ^ { 3 / 2 } \sqrt { r ( S + K ) T } . } \end{array}
$$

The sum is zero for $H = 1$ . By $( 7 ) , 2 ^ { H } \leq 2 ( T + 1 )$ , so $\beta _ { H } \le 4 0 \log ( 1 6 S K r ( T + 1 ) )$ . Thus

$$
\mathrm { R e g } _ { T } \leq C \log ^ { 3 / 2 } ( 1 6 S K r ( T + 1 ) ) \sqrt { r ( S + K ) T } = \widetilde O \Big ( \sqrt { r ( S + K ) T } \Big ) .
$$

## A.3 Proofs of the Lemmas

Proof of Lemma 1. Suppose Baseline or Collect begins at round τ ; condition on the observations and learner’s random choices before $i _ { \tau }$ is drawn. Fix i and $j = \pi ( i )$ for Baseline, or one entry $j$ of $Q ( i )$ for Collect. For $k \geq 1$ , let $I _ { t }$ indicate whether $y _ { t }$ is among the first k rewards recorded for this choice; sums below run over that algorithm’s rounds.

Whether $I _ { t } = 1$ is determined before observing $y _ { t }$ . By (1) and iterated expectations,

$$
\mathbb { E } \exp \left( \lambda \sum _ { t } I _ { t } \eta _ { t } - \frac { \lambda ^ { 2 } } { 2 } \sum _ { t } I _ { t } \right) \leq 1 .
$$

With $\lambda = \sqrt { 2 \beta _ { h } / k }$ , Markov’s inequality gives

$$
\mathrm { P r } \left( \sum _ { t } I _ { t } = k , \sum _ { t } I _ { t } \eta _ { t } > \sqrt { 2 \beta _ { h } k } \right) \le e ^ { - \beta _ { h } } .
$$

When $\begin{array} { r } { \sum _ { t } { I _ { t } } = k , \sum _ { t } { I _ { t } } \eta _ { t } = k ( \widehat { A } _ { k } ( i , j ) - A ( i , j ) ) } \end{array}$ . Applying the same bound with −λ yields

$$
\operatorname* { P r } \left( \sum _ { t } I _ { t } = k , \quad | \widehat { A } _ { k } ( i , j ) - A ( i , j ) | > \sqrt { 2 \beta _ { h } / k } \right) \le 2 e ^ { - \beta _ { h } } .\tag{11}
$$

Over $L _ { h } ( k )$ rounds starting at $\tau ,$ independence and the uniform context distribution give

$$
\begin{array} { r } { \operatorname* { P r } \left( \displaystyle \sum _ { t = \tau } ^ { \tau + L _ { h } ( k ) - 1 } \mathbf { 1 } \{ i _ { t } = i \} < k \right) \le e ^ { k } \left( 1 - \displaystyle \frac { 1 - e ^ { - 1 } } { S } \right) ^ { L _ { h } ( k ) } } \\ { \le e ^ { k - L _ { h } ( k ) / ( 2 S ) } \le e ^ { - 8 \beta _ { h } } . } \end{array}\tag{12}
$$

Here $1 - e ^ { - 1 } > 1 / 2$ and $L _ { h } ( k ) \ge 1 2 8 S ( k + \beta _ { h } )$ . For Baseline, use $k = m _ { h }$ and $\sqrt { 2 \beta _ { h } / m _ { h } } \le \varepsilon _ { h } / 2$ Equations (11)–(12) and a union bound give

$$
\operatorname* { P r } ( \exists i : | B _ { h } ( i ) - A ( i , \pi ( i ) ) | > \varepsilon _ { h } / 2 ) \le 2 S e ^ { - \beta _ { h } } + S e ^ { - 8 \beta _ { h } } .
$$

Proof of Lemma 2. Conditional on the choices of $Q ( i )$ , (11) applies to every entry and each $k \le m _ { h , n }$ . Counting repeated arms separately, Steps 2–4 compute at most

$$
\begin{array} { r l r } {  { S \sum _ { n = 0 } ^ { h } ( q _ { h , n } + r ) m _ { h , n } + S | W _ { h } | m _ { h } } } \\ & { } & { \leq \frac { 3 6 \beta _ { h } ^ { 2 } r K + 1 8 S \beta _ { h } + 1 8 S r \beta _ { h } + 9 S | W _ { h } | \beta _ { h } } { \varepsilon _ { h } ^ { 2 } } \leq \frac { 6 4 \beta _ { h } ^ { 3 } r ( S + K ) } { \varepsilon _ { h } ^ { 2 } } . } \end{array}
$$

The inequalities use (5) and (4). A union bound, $r ( S + K ) \leq 2 S K r$ , and $\beta _ { h } ^ { 3 } \le 1 6 0 0 0 ( 1 6 S K r 2 ^ { h } )$ bound the failure probability by

$$
1 2 8 \beta _ { h } ^ { 3 } r ( S + K ) 2 ^ { h } e ^ { - \beta _ { h } } \leq \frac { 2 5 6 0 0 0 } { ( 1 6 S K r 2 ^ { h } ) ^ { 1 8 } } \leq \frac { 2 ^ { 1 8 } } { ( 1 6 S K r 2 ^ { h } ) ^ { 1 8 } } .
$$

Proof of Lemma 3. In Collect at level n, each entry of $Q ( i )$ uses at most $m _ { h , n }$ samples. Over its $L _ { h } ( m _ { h , n } \operatorname* { m a x } _ { i } | Q ( i ) | )$ rounds,

$$
\sum _ { t } { { \bf 1 } \{ i _ { t } = i \} \geq m _ { h , n } | Q ( i ) | } \quad \Longrightarrow \quad a _ { i } > | Q ( i ) | \mathrm { a t ~ t h e ~ e n d . }
$$

Equation (12) bounds the failure probability by $S e ^ { - 8 \beta _ { h } }$ . Summing over the uses of Collect in Steps 2–4 gives

$$
S [ ( h + 1 ) + r ( h + 1 ) + 1 ] e ^ { - 8 \beta _ { h } } = S [ ( r + 1 ) ( h + 1 ) + 1 ] ( 1 6 S K r 2 ^ { h } ) ^ { - 1 6 0 } .
$$

Proof of Lemma 4. Fix b, n with $\varepsilon _ { n } \geq \varepsilon _ { h , b }$ . The block has $| B ( b ) | = S \omega ( b )$ contexts. Thus

$$
\begin{array} { r l } & { \operatorname* { P r } ( j _ { b } ^ { \star } \notin Q _ { h , n } ( i ) \ \forall i \in \mathcal { B } ( b ) ) = ( 1 - 1 / K ) ^ { q _ { h , n } | \mathcal { B } ( b ) | } } \\ & { \qquad \le \exp ( - q _ { h , n } | \mathcal { B } ( b ) | / K ) \le \exp ( - \beta _ { h } r \omega ( b ) \varepsilon _ { n } / \varepsilon _ { h } ) \le e ^ { - \beta _ { h } } , } \end{array}
$$

since $r \omega ( b ) \varepsilon _ { n } / \varepsilon _ { h } \ge r \omega ( b ) \varepsilon _ { h , b } / \varepsilon _ { h } = \operatorname* { m a x } \{ r \omega ( b ) , 1 \} \ge 1$ . There are at most $r ( h + 1 )$ relevant pairs $b , n .$ A union bound gives the claim. □

Proof of Lemma 5. Lemmas 1–4 give

$$
\begin{array} { r l r } {  { \operatorname* { P r } ( \Omega _ { h } ^ { c } ) \leq \frac { 2 ^ { 1 8 } } { ( 1 6 S K r 2 ^ { h } ) ^ { 1 8 } } + \frac { 2 S + r ( h + 1 ) } { ( 1 6 S K r 2 ^ { h } ) ^ { 2 0 } } + \frac { S [ ( r + 1 ) ( h + 1 ) + 2 ] } { ( 1 6 S K r 2 ^ { h } ) ^ { 1 6 0 } } } } \\ & { } & { \leq \frac { 2 ^ { 1 8 } } { ( 1 6 S K r 2 ^ { h } ) ^ { 1 8 } } + \frac { 1 } { ( 1 6 S K r 2 ^ { h } ) ^ { 1 9 } } + \frac { 1 } { ( 1 6 S K r 2 ^ { h } ) ^ { 1 5 9 } } \leq \frac { 1 } { ( 1 6 S K r 2 ^ { h } ) ^ { 8 } } . } \end{array}
$$

The second inequality follows from $h + 1 \leq 2 ^ { h }$ , and the last from $1 6 S K r 2 ^ { h } \geq 3 2$

Proof of Lemma 6. On $\Omega _ { h }$ , if the stopping inequality holds at level $n ,$ then

$$
A ( i , j ) \leq \widehat { A } _ { k } ( i , j ) + \sqrt { 2 \beta _ { h } / k } < B _ { h } ( i ) - 3 \varepsilon _ { n }
$$

$$
\leq A ( i , \pi _ { h - 1 } ( i ) ) + \varepsilon _ { h } / 2 - 3 \varepsilon _ { n } \leq A ( i , \pi _ { h - 1 } ( i ) ) - 5 \varepsilon _ { n } / 2 .\tag{13}
$$

An optimal arm cannot satisfy this inequality. By $( \mathrm { i i i } ) { - } ( \mathrm { i v } )$ , whenever $\varepsilon _ { n } \geq \varepsilon _ { h , b } .$ , Step 2 records $( i , j _ { b } ^ { \star } ) \in D _ { h , n }$ for some $i \in \mathcal { B } ( b )$

In Step $3 , ( i , j ^ { \star } ) \notin D$ implies $A ( i , j ^ { \star } ) < A ( i , \pi _ { h - 1 } ( i ) )$ by (iii) and (13). Since completed estimates and $B _ { h }$ have error at most $\varepsilon _ { n } / 2$ , the definition of $\widehat { V }$ gives

$$
| \widehat { V } ( i , j ^ { \star } ) - \operatorname* { m a x } \{ A ( i , \pi _ { h - 1 } ( i ) ) , A ( i , j ^ { \star } ) \} | \leq \varepsilon _ { n } / 2 .\tag{14}
$$

For the selected $( i ^ { \star } , j ^ { \star } )$ and any $i \in B ( g ( i ^ { \star } ) ,$ ),

$$
\begin{array} { r l } & { \widehat { V } ( i , j ^ { \star } ) \geq A ( i , j ^ { \star } ) - \varepsilon _ { n } / 2 = A ( i ^ { \star } , j ^ { \star } ) - \varepsilon _ { n } / 2 } \\ & { \phantom { \frac { 1 } { \theta } } \geq \widehat { A } _ { h , n } ( i ^ { \star } , j ^ { \star } ) - \varepsilon _ { n } > \widehat { A } _ { h , n } ( i ^ { \star } , j ^ { \star } ) - 4 \varepsilon _ { n } . } \end{array}
$$

All remaining records from this block are deleted. No records are added during Step 3, so each $D _ { h , n }$ becomes empty within r iterations.

For $\varepsilon _ { h , b } \leq 1$ , choose the largest $n \leq h$ with $\varepsilon _ { n } \geq \varepsilon _ { h , b }$ . Since $\varepsilon _ { 0 } = 1$ and $\varepsilon _ { h , b } \geq \varepsilon _ { h } , \varepsilon _ { h , b } \leq \varepsilon _ { n } < \sqrt { 2 } \varepsilon _ { h , b }$ Choose $( i , j _ { b } ^ { \star } ) \in D _ { h , n }$ with $i \in B ( b )$ . When this record is deleted,

$$
\begin{array} { r l } & { \widehat { A } _ { h , n } ( i ^ { \star } , j ^ { \star } ) \geq \widehat { A } _ { h , n } ( i , j _ { b } ^ { \star } ) \geq \mu ( b , j _ { b } ^ { \star } ) - \varepsilon _ { n } , } \\ & { \qquad \widehat { V } ( i , j ^ { \star } ) \geq \widehat { A } _ { h , n } ( i ^ { \star } , j ^ { \star } ) - 4 \varepsilon _ { n } . } \end{array}
$$

Since $j ^ { \star } , \pi _ { h - 1 } ( i ) \in W _ { h }$ , (14) implies

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { j \in W _ { h } } \mu ( b , j ) \geq \operatorname* { m a x } \{ A ( i , \pi _ { h - 1 } ( i ) ) , A ( i , j ^ { \star } ) \} } \\ { \displaystyle \qquad \geq \widehat { V } ( i , j ^ { \star } ) - \varepsilon _ { n } \geq \mu ( b , j _ { b } ^ { \star } ) - 6 \varepsilon _ { n } \geq \mu ( b , j _ { b } ^ { \star } ) - 6 \sqrt { 2 } \varepsilon _ { h , b } . } \end{array}
$$

The gap is also at most one by (2). For $\varepsilon _ { h , b } > 1$ , this alone sufices.

Proof of Lemma 7. In Step 4, (iii) and (13) ensure that every $j \in W _ { h }$ with $A ( i , j ) \geq A ( i , \pi _ { h - 1 } ( i ) )$ is recorded in D. Since $\pi _ { h - 1 } ( i ) \in W _ { h }$ , this includes an arm maximizing $A ( i , j )$ over $W _ { h }$ . Using the error bound $\varepsilon _ { h } / 2$ for all estimates, including $B _ { h }$

$$
\begin{array} { l } {  { { \cal A } ( i , \pi _ { h } ( i ) ) \geq \widehat { A } ( i , \pi _ { h } ( i ) ) - \varepsilon _ { h } / 2 } } \\ { \quad = \operatorname* { m a x } _ { j : ( i , j ) \in { \cal D } } \widehat { A } ( i , j ) - \varepsilon _ { h } / 2 \geq \operatorname* { m a x } _ { j \in W _ { h } } { A } ( i , j ) - \varepsilon _ { h } . } \end{array}
$$

For $i \in \mathcal { B } ( b )$ , Lemma 6 and (2) give

$$
\operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) \leq \operatorname* { m i n } \{ 1 , 6 \sqrt { 2 } \varepsilon _ { h , b } + \varepsilon _ { h } \} \leq \operatorname* { m i n } \{ 1 , 1 0 \varepsilon _ { h , b } \} .
$$

Finally,

$$
\begin{array} { r l } {  { \frac { 1 } { S } \sum _ { i } [ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) ] \le 1 0 \sum _ { b } \omega ( b ) \varepsilon _ { h , b } = 1 0 \varepsilon _ { h } \sum _ { b } \operatorname* { m a x } \{ \omega ( b ) , 1 / r \} } \quad } & { } \\ & { \leq 1 0 \varepsilon _ { h } \sum _ { b } ( \omega ( b ) + 1 / r ) = 2 0 \varepsilon _ { h } . } \end{array}
$$

Only $\Omega _ { h }$ was used, so Lemma 5 gives the stated probability.

## B General Context Distributions

## B.1 Proof of Corollary 1

Proof. Use the first definition in (3), with sums over the nonempty blocks:

$$
\varepsilon _ { h , b } = \operatorname* { m a x } \left\{ 1 , \frac { S } { r | \mathcal { B } ( b ) | } \right\} \varepsilon _ { h } , \qquad \sum _ { b } \frac { | \mathcal { B } ( b ) | } { S } = 1 , \qquad \sum _ { b } \frac { 1 } { r } \leq 1 .
$$

Conditional on the history at a call’s start $\tau , \nu ( i ) \geq 1 / ( 8 S )$ gives

$$
\begin{array} { r l r } {  { \operatorname* { P r } ( \sum _ { t = \tau } ^ { \tau + L _ { h } ( k ) - 1 } { \mathbf { 1 } \{ i _ { t } = i \} } < k ) \le e ^ { k } [ 1 - ( 1 - e ^ { - 1 } ) \nu ( i ) ] ^ { L _ { h } ( k ) } } } \\ & { } & { \qquad \le e ^ { k - L _ { h } ( k ) / ( 1 6 S ) } \le e ^ { - 7 k - 8 \beta _ { h } } \le e ^ { - 8 \beta _ { h } } . } \end{array}
$$

Together with (11), this preserves Lemmas 1–3. For $\varepsilon _ { n } \geq \varepsilon _ { h , b }$

$$
\begin{array} { r l } {  { \operatorname* { P r } ( j _ { b } ^ { \star } \notin Q _ { h , n } ( i ) \ \forall i \in \mathcal { B } ( b ) ) \le \exp ( - \frac { q _ { h , n } \vert \mathcal { B } ( b ) \vert } { K } ) } \quad } & { } \\ & { \le \exp ( - \frac { \beta _ { h } r \vert \mathcal { B } ( b ) \vert \varepsilon _ { n } } { S \varepsilon _ { h } } ) \le e ^ { - \beta _ { h } } . } \end{array}
$$

Hence $\mathrm { P r } ( \Omega _ { h } ^ { c } ) \le ( 1 6 S K r 2 ^ { h } ) ^ { - 8 }$ as in Lemmas 4–5. On $\Omega _ { h }$ , the same comparisons as in Lemmas $6 { - } 7$ yield

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { S } \sum _ { i } [ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) ] \leq \frac { 1 0 } { S } \sum _ { b } | \mathcal { B } ( b ) | \varepsilon _ { h , b } } & { } \\ { \displaystyle } & { = 1 0 \varepsilon _ { h } \sum _ { b } \operatorname* { m a x } \{ | \mathcal { B } ( b ) | / S , 1 / r \} \leq 2 0 \varepsilon _ { h } , } \end{array}
$$

$$
\sum _ { i } \nu ( i ) [ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) ] \leq \frac { 8 } { S } \sum _ { i } [ \operatorname* { m a x } _ { j } A ( i , j ) - A ( i , \pi _ { h } ( i ) ) ] \leq 1 6 0 \varepsilon _ { h } .
$$

For $h \geq 2$ , use (9) for assigned plays. For plays of $\pi _ { h - 1 }$ , drop $\mathbf { 1 } _ { \Omega _ { h } }$ and condition at each call’s start; for $\pi _ { h }$ , condition at the final loop’s start. By (4) and (7),

$$
\begin{array} { r l } {  { \mathbb { E } [ \mathbf { 1 } _ { \Omega _ { h - 1 } \cap \Omega _ { h } } \sum _ { t = t _ { h } } ^ { t _ { h + 1 } - 1 } \binom { \operatorname* { m a x } { A ( i _ { t } , j ) } - A ( i _ { t } , j _ { t } ) } { j } ] } } \\ & { \leq C S \beta _ { h } [ \sum _ { n = 0 } ^ { h } \frac { q _ { h , n } + r } { \varepsilon _ { n } } + \frac { r \beta _ { h } ^ { 2 } } { \varepsilon _ { h } } ] + 1 6 0 ( \varepsilon _ { h - 1 } + \varepsilon _ { h } ) ( t _ { h + 1 } - t _ { h } ) } \\ & { \leq \frac { C \beta _ { h } ^ { 3 } r ( S + K ) } { \varepsilon _ { h } } . } \end{array}
$$

For $t _ { H } \leq T < t _ { H + 1 }$ , (10) and the phase summation in Appendix A.2 now give

$$
\mathrm { R e g } _ { T } \leq \mathrm { m i n } \{ T , t _ { 2 } - t _ { 1 } \} + C r ( S + K ) \sum _ { h = 2 } ^ { H } \beta _ { h } ^ { 3 } 2 ^ { h / 2 } + O ( 1 )
$$

$$
\leq C \beta _ { H } ^ { 3 / 2 } { \sqrt { r ( S + K ) T } } \leq C \log ^ { 3 / 2 } ( 1 6 S K r ( T + 1 ) ) { \sqrt { r ( S + K ) T } } .
$$

## B.2 Estimation of the Context Distribution

Lemma 8. Suppose $J < T _ { i }$ , with $J , L , \widehat \nu$ and the sets $\mathcal { X } _ { \ell }$ defined in Section $4 . 2 .$ With probability at least $1 - ( T + 1 ) ^ { - 2 }$ , both of the following hold:

(i) For every nonempty $\mathcal { X } _ { \ell }$ with $\ell < L$ and every $i \in \mathcal { X } _ { \ell }$

$$
\frac { 1 } { 8 | \mathcal { X } _ { \ell } | } \le \frac { \nu ( i ) } { \sum _ { i ^ { \prime } \in \mathcal { X } _ { \ell } } \nu ( i ^ { \prime } ) } \le \frac { 8 } { | \mathcal { X } _ { \ell } | } .
$$

(ii) The remaining set has probability mass

$$
\sum _ { i \in \mathcal { X } _ { L } } \nu ( i ) \leq 2 { \sqrt { S / T } } .
$$

Proof. Bernstein’s inequality and a union bound $\mathrm { g i v e } .$ , simultaneously for all $i \in [ S ]$ , with probability at least $1 - ( T + 1 ) ^ { - 2 }$ 2

$$
\begin{array} { r } { | \widehat { \nu } ( i ) - \nu ( i ) | \leq \sqrt { \frac { 2 \nu ( i ) \log ( 2 S ( T + 1 ) ^ { 2 } ) } { J } } + \frac { \log ( 2 S ( T + 1 ) ^ { 2 } ) } { J } } \\ { \leq \frac { \nu ( i ) } { 4 } + \frac { 3 \log ( 2 S ( T + 1 ) ^ { 2 } ) } { J } \leq \frac { \nu ( i ) } { 4 } + \frac { 3 } { 3 2 \sqrt { S T } } . } \end{array}
$$

Therefore

$$
\frac { 4 } { 5 } \widehat { \nu } ( i ) - \frac { 3 } { 4 0 \sqrt { S T } } \leq \nu ( i ) \leq \frac { 4 } { 3 } \widehat { \nu } ( i ) + \frac { 1 } { 8 \sqrt { S T } } .
$$

For $i , i ^ { \prime } \in \mathcal { X } _ { \ell }$ with $\ell < L$ 2

$$
\begin{array} { c } { { \displaystyle \widehat { \nu } ( i ) \geq \frac { 1 } { \sqrt { S T } } \implies \frac { 1 } { 2 } \widehat { \nu } ( i ) \leq \nu ( i ) \leq 2 \widehat { \nu } ( i ) , \hfill } } \\ { { \displaystyle \frac { \nu ( i ) } { \nu ( i ^ { \prime } ) } \leq 4 \frac { \widehat { \nu } ( i ) } { \widehat { \nu } ( i ^ { \prime } ) } \leq 8 , \hfill } } \\ { { \displaystyle \frac { 1 } { 8 | \mathcal { X } _ { \ell } | } \leq \frac { \nu ( i ) } { \sum _ { i ^ { \prime } \in \mathcal { X } _ { \ell } } \nu ( i ^ { \prime } ) } \leq \frac { 8 } { | \mathcal { X } _ { \ell } | } . \hfill } } \end{array}
$$

For the remaining contexts,

$$
\sum _ { i \in \mathcal { X } _ { L } } \nu ( i ) \leq \sum _ { i \in \mathcal { X } _ { L } } \left( \frac { 4 } { 3 } \widehat { \nu } ( i ) + \frac { 1 } { 8 \sqrt { S T } } \right) \leq \frac { 2 | \mathcal { X } _ { L } | } { \sqrt { S T } } \leq 2 \sqrt { S / T } .
$$

## B.3 Proof of Theorem 2

Proof. If $J \ge T , ( 2 )$ gives

$$
\mathrm { R e g } _ { T } \leq T \leq J \leq 3 3 \sqrt { S T } \log ( 2 S ( T + 1 ) ^ { 2 } ) \leq C \log ( 1 6 S K r ( T + 1 ) ) \sqrt { r ( S + K ) T } .
$$

For $J < T$ , let $\mathcal { E }$ be the event in Lemma 8 and $\begin{array} { r } { T _ { \ell } = \sum _ { t = J + 1 } ^ { T } \mathbf { 1 } \{ i _ { t } \in \mathcal { X } _ { \ell } \} } \end{array}$ for $0 \le \ell \le L$ . The groups and E depend only on the initial observations, so

$$
\mathbb { E } [ \mathbf { 1 } _ { \mathcal { E } } T _ { L } ] \le ( T - J ) \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \sum _ { i \in \mathcal { X } _ { L } } \nu ( i ) \right] \le 2 \sqrt { S T } ,
$$

$$
( T - J ) \operatorname* { P r } ( { \mathcal { E } } ^ { c } ) \leq { \frac { T - J } { ( T + 1 ) ^ { 2 } } } \leq 1 .
$$

On $\mathcal { E } ,$ , condition on the initial history and subsequent group memberships until we average below; each $T _ { \ell }$ is then fixed. For $i \in \mathcal { X } _ { \ell }$ at a round assigned to copy $\ell < L .$ , independence of context arrivals gives

$$
\operatorname* { P r } ( i _ { t } = i \mid i _ { 1 } , j _ { 1 } , y _ { 1 } , \dotsc , i _ { t - 1 } , j _ { t - 1 } , y _ { t - 1 } ) = { \frac { \nu ( i ) } { \sum _ { i ^ { \prime } \in { \mathcal { X } _ { \ell } } } \nu ( i ^ { \prime } ) } } \in \left[ { \frac { 1 } { 8 | { \mathcal { X } _ { \ell } } | } } , { \frac { 8 } { | { \mathcal { X } _ { \ell } } | } } \right] .
$$

Future memberships are independent of preceding rewards, and the current membership is fixed by $i _ { t }$ . Thus (1) remains valid, also for each copy’s local history by iterated conditioning. Each nonempty group has at most min $\{ r , | \mathcal { X } _ { \ell } | \}$ blocks, so Corollary 1 gives (trivially if $T _ { \ell } = 0 )$

$$
\begin{array} { r l } & { \mathbb { E } [  \sum _ { \stackrel { J < t \leq T } { i _ { t } \in \mathcal { X } _ { \ell } } } ( \underset { j } { \operatorname* { m a x } } A ( i _ { t } , j ) - A ( i _ { t } , j _ { t } ) ) ] } \\ & { \quad \leq C \log ^ { 3 / 2 } ( 1 6 S K r ( T + 1 ) ) \sqrt { r ( | \mathcal { X } _ { \ell } | + K ) T _ { \ell } } . } \end{array}
$$

Cauchy–Schwarz yields

$$
\begin{array} { r l } { \displaystyle \sum _ { \ell < L } \sqrt { r ( | \mathcal { X } _ { \ell } | + K ) T _ { \ell } } \leq \sqrt { r \left( \sum _ { \ell < L } | \mathcal { X } _ { \ell } | + L K \right) \sum _ { \ell < L } T _ { \ell } } } & { } \\ { \leq \sqrt { r ( S + L K ) ( T - J ) } . } \end{array}
$$

Averaging over the conditioning and using (2),

$$
\begin{array} { r l } & { \mathrm { R e g } _ { T } \leq J + \mathbb { E } \left[ \pmb { 1 } _ { \mathcal { E } \underset { \ell \in \mathcal { L } } { \sum } } \sum _ { j \in \mathcal { L } } \left( \operatorname* { m a x } _ { j } A ( i _ { t } , j ) - A ( i _ { t } , j _ { t } ) \right) \right] } \\ & { \qquad + \mathbb { E } [ \pmb { 1 } _ { \mathcal { E } } T _ { L } ] + ( T - J ) \operatorname* { P r } ( \mathcal { E } ^ { c } ) } \\ & { \qquad \leq J + 2 \sqrt { S T } + 1 + C \log ^ { 3 / 2 } ( 1 6 S K r ( T + 1 ) ) \sqrt { r ( S + L K ) T } } \\ & { \qquad \leq C \log ^ { 2 } ( 1 6 S K r ( T + 1 ) ) \sqrt { r ( S + K ) T } . } \end{array}
$$

The last step uses $J \leq C \sqrt { S T } \log ( 1 6 S K r ( T + 1 ) )$ and $L \leq C \log ( 1 6 S K r ( T + 1 ) )$

Restarting on rounds $2 ^ { p } , \ldots , 2 ^ { p + 1 } - 1$ , the same bound holds conditional on each restart’s history. Complete the last interval and sum:

$$
\begin{array} { r l } { \left. { \mathrm { R e g } _ { T } \le \sum _ { p = 0 } ^ { \lfloor \log _ { 2 } T \rfloor } \mathbb { E } \left[ \sum _ { t = 2 ^ { p } } ^ { 2 ^ { p + 1 } - 1 } \binom { \operatorname* { m a x } { A ( i _ { t } , j ) } - A ( i _ { t } , j _ { t } ) } { j } \right] } \right] }  \\ & { \le C \log ^ { 2 } ( 1 6 S K r ( T + 1 ) ) \sqrt { r ( S + K ) } \frac { 2 ( \lfloor \log _ { 2 } T \rfloor + 1 ) / 2 } { \sqrt { 2 } - 1 } } \\ & { \le C \log ^ { 2 } ( 1 6 S K r ( T + 1 ) ) \sqrt { r ( S + K ) T } . } \end{array}
$$