# Minimax Last-Iterate Convergence in Matrix Games with Observed Actions

Yuheng Zhang University of Illinois Urbana-Champaign yuhengz2@illinois.edu

## Abstract

We study last-iterate convergence in unknown two-player zero-sum matrix games with bandit payof feedback and observed opponent actions. For games with d actions per player, we develop an algorithm achieving a duality gap of $\widetilde { \mathcal { O } } ( \sqrt { d / t } )$ with high probability, simultaneously at every round t. This improves the dimension dependence of the best previously known guarantee by a factor of $d ^ { 3 / 2 }$ . The rate matches a standard bandit lower bound, establishing minimax optimality in both the number of actions and the number of rounds, up to logarithmic factors. The algorithm is computationally eficient, requiring only O(d) time and memory per round. Our technical contribution is a joint design of adaptive averaging and corrected exponential weights that absorbs estimation variance, together with a potential argument that bounds phase durations.

## 1 Introduction

We study learning a Nash equilibrium in unknown two-player zero-sum matrix games with d actions per player using feedback from sampled interactions. Nash equilibrium is a central solution concept in learning in games; in zero-sum games, equilibrium strategies guarantee each player the value of the game against any opponent. Classical no-regret algorithms guarantee convergence of average strategies (Freund and Schapire, 1999), but the strategies used at individual rounds can remain exploitable. When learning and deployment occur together, players need guarantees on their current strategies, not just on an average over past play. This motivates last-iterate guarantees, which ensure that the strategies actually played approach equilibrium and remain near it at all suficiently late rounds.

We consider bandit payof feedback with observed opponent actions: after each round, both players observe the sampled action pair and its common noisy payof. Such feedback is natural in self-play, where both actions can be recorded, and in preference learning, where both responses in a comparison are available (Munos et al., 2023). Security games provide another example when the attacked target and the defensive action are observable after each round (Hait et al., 2026). Observing the opponent’s action provides information beyond the payof alone, while revealing only one payof per round leaves exploration necessary. We ask whether this additional information sufices to attain statistically optimal last-iterate convergence.

Under bandit feedback without observing the opponent’s action, prior work obtains last-iterate duality gaps of $\widetilde { \mathcal { O } } ( \sqrt { d } t ^ { - 1 / 8 } )$ (Cai et al., 2023) and $\widetilde { \mathcal { O } } ( d ^ { 1 / 5 } t ^ { - 1 / 5 } )$ (Cai et al., 2025), where t is the number of rounds and Oe suppresses logarithmic factors. Fiegel et al. (2026b) improve the time dependence to t<sup>−1/4</sup>, up to logarithmic factors, with an anytime high-probability guarantee. Their paper does not state the dependence on d explicitly; making the parameters in their proof explicit yields the bound $\widetilde { \mathcal { O } } ( d ^ { 2 } t ^ { - 1 / 4 } )$ ).

Table 1: Representative last-iterate guarantees for zero-sum matrix games with d actions per player and bandit payof feedback. All upper bounds hold with high probability uniformly over time. The last column gives suficient rounds to attain and maintain duality gap at most $\varepsilon ;$ the row for the lower bound gives necessary rounds at fixed confidence. The gap lower bound assumes $t \geq d .$ For Fiegel et al. (2026b), the dependence on d is not stated in their paper, and the bounds reported here are derived from their proof (see details in Appendix B).
<table><tr><td>Work</td><td>Opponent actions Duality gap at round t Rounds for gap ε</td><td></td><td></td></tr><tr><td>Cai et al. (2023)</td><td>Unobserved</td><td> $\widetilde { \mathcal { O } } ( \sqrt { d } t ^ { - 1 / 8 } )$ </td><td> $\widetilde { \mathcal { O } } ( d ^ { 4 } / \varepsilon ^ { 8 } )$ </td></tr><tr><td>Cai et al. (2025)</td><td>Unobserved</td><td> $\widetilde { \mathcal { O } } ( d ^ { 1 / 5 } t ^ { - 1 / 5 } )$ </td><td> $\widetilde { \mathcal { O } } ( d / \varepsilon ^ { 5 } )$ </td></tr><tr><td>Fiegel et al. (2026b)</td><td>Unobserved</td><td> $\widetilde { \mathcal { O } } ( d ^ { 2 } t ^ { - 1 / 4 } )$ </td><td> $\widetilde { \mathcal { O } } ( d ^ { 8 } / \varepsilon ^ { 4 } )$ </td></tr><tr><td>Hait et al. (2026)</td><td>Observed</td><td> $\widetilde { \mathcal { O } } ( d ^ { 2 } / \sqrt { t } )$ </td><td> $\widetilde { \mathcal { O } } ( d ^ { 4 } / \varepsilon ^ { 2 } )$ </td></tr><tr><td>Bandit lower bound</td><td>Observed</td><td> $\Omega ( \sqrt { d / t } )$ </td><td> $\Omega ( d / \varepsilon ^ { 2 } )$ </td></tr><tr><td>Our work</td><td>Observed</td><td> $\tilde { \cal O } ( \sqrt { d / t } )$ </td><td> $\widetilde { \cal O } ( d / \varepsilon ^ { 2 } )$ </td></tr></table>

With observed opponent actions, the closest work, Hait et al. (2026), achieves $\widetilde { \mathcal { O } } ( d ^ { 2 } / \sqrt { t } )$ lastiterate convergence with high probability by estimating the payof matrix and periodically solving a game with log-barrier regularization. This achieves the optimal dependence on t, but leaves a gap in d compared with the bandit lower bound of order $\sqrt { d / t }$ . Thus, a basic question remains:

Can last-iterate convergence with observed opponent actions attain the minimax rate in both the number of actions and the number of rounds?

Our results. We answer this question afirmatively. To the best of our knowledge, we provide the first high-probability last-iterate guarantee that is minimax optimal in both d and t, up to logarithmic factors, for bandit payof feedback with observed opponent actions. Specifically, for any confidence parameter $\delta \in ( 0 , 1 )$ , our algorithm produces played strategies $( x _ { t } , y _ { t } )$ whose duality gap, the sum of both players’ gains from unilateral deviations, satisfies

$$
\mathrm { G a p } ( x _ { t } , y _ { t } ) \leq C \sqrt { \frac { d } { t } } \left[ \log \biggl ( \frac { d t } { \delta } \biggr ) \right] ^ { 3 / 2 }
$$

$$
\mathrm { s i m u l t a n e o u s l y ~ f o r ~ a l l } ~ t \geq 1 
$$

with probability at least $1 - \delta ,$ , where C is a universal constant (Theorem 1). The algorithm uses one payof observation per round and does not require the time horizon. It improves the dimension dependence of Hait et al. (2026) by a factor of $d ^ { 3 / 2 }$ . Equivalently, the number of rounds suficient to reach duality gap ε and maintain this accuracy thereafter decreases from $\widetilde { \mathcal { O } } ( d ^ { 4 } / \varepsilon ^ { 2 } )$ to $\widetilde { \mathcal { O } } ( d / \varepsilon ^ { 2 } )$ Games with identical columns reduce to a d-armed bandit problem, giving a worst-case lower bound of $\Omega ( d / \varepsilon ^ { 2 } )$ observations at fixed confidence (Mannor and Tsitsiklis, 2004). This lower bound matches our guarantee up to logarithmic factors. The algorithm also requires only $\mathcal O ( d )$ time and memory per round, using explicit vector updates. Table 1 summarizes the comparison.

Technical ideas. We build on the idea of playing an average of auxiliary strategies (Cai et al., 2025). Under bandit feedback, the auxiliary learners and the played strategies use diferent distributions, so estimating the auxiliary losses can incur large variance. Our main technical contribution is a joint design of adaptive averaging and a correction to exponential weights. The correction creates a negative term measuring the distribution mismatch, while the averaging weights allow this term to absorb the estimation variance with only a linear factor in d. Together with an adaptation of implicit exploration (Neu, 2015), this yields the optimal dimension dependence with high probability.

Adaptive averaging can nevertheless make little progress in actual rounds when the weights are small. Our second contribution is a potential analysis that bounds the number of rounds needed for each constant-factor improvement in accuracy. We combine this analysis with a phase scheme that retains the preceding phase’s output as a baseline, keeping intermediate play near equilibrium while learning a more accurate strategy pair. This converts the variance control into an anytime guarantee for every played pair.

## 2 Preliminaries

Notation. For a positive integer $d ,$ let $[ d ] = \{ 1 , \ldots , d \}$ and $\Delta _ { d } = \{ x \in \mathbb { R } _ { > 0 } ^ { d } : \sum _ { i = 1 } ^ { d } x _ { i } = 1 \}$ . We write $x _ { i }$ for the ith coordinate of a vector $x , e _ { i }$ for the ith standard basis vector, and 1 for the all-ones vector. All logarithms are natural, and $\widetilde { \mathcal { O } }$ hides logarithmic factors.

Problem formulation. We consider a two-player zero-sum game with a fixed unknown matrix $A \in [ - 1 , 1 ] ^ { d \times d }$ , where $d \geq 2$ is the number of actions available to each player. The entry $A _ { i j }$ is the expected loss of the row player and the expected reward of the column player when they choose actions i and $j ,$ respectively. For mixed strategies x, $y \in \Delta _ { d } .$ , the row player minimizes $x ^ { \top } A y$ , while the column player maximizes it.

The players learn through repeated play with bandit payof feedback and observed opponent actions. Let $\mathcal { F } _ { t - 1 }$ denote the common observation history before round t. At round t, the players choose mixed strategies $x _ { t } , y _ { t } \in \Delta _ { d }$ based on this history. Conditionally on $\mathcal { F } _ { t - 1 }$ , they independently sample $I _ { t } \sim x _ { t }$ and $J _ { t } \sim y _ { t }$ . Both players then observe the action pair $( I _ { t } , J _ { t } )$ and a common payof $R _ { t } \in [ - 1 , 1 ]$ satisfying

$$
\mathbb { E } [ R _ { t } \ | \ { \mathcal { F } } _ { t - 1 } , I _ { t } , J _ { t } ] = A _ { I _ { t } J _ { t } } .
$$

Each round thus provides one payof observation, interpreted as a loss for the row player and a reward for the column player.

We measure the quality of a strategy pair $( x , y )$ by its duality gap,

$$
\mathrm { G a p } ( x , y ) = \operatorname* { m a x } _ { y ^ { \prime } \in \Delta _ { d } } x ^ { \top } A y ^ { \prime } - \operatorname* { m i n } _ { x ^ { \prime } \in \Delta _ { d } } { x ^ { \top } } A y = \operatorname* { m a x } _ { j \in [ d ] } x ^ { \top } A e _ { j } - \operatorname* { m i n } _ { i \in [ d ] } e _ { i } ^ { \top } A y .
$$

This quantity is the sum of the two players’ gains from unilateral deviations and lies in [0, 2]. A pair $( x , y )$ is a Nash equilibrium if and only if $\mathrm { G a p } ( x , y ) = 0$ . More generally, ${ \mathrm { G a p } } ( x , y ) \leq \varepsilon$ implies that $( x , y )$ is an ε-Nash equilibrium: neither player can improve its expected payof by more than ε through a unilateral deviation.

Our goal is last-iterate convergence, measured by the duality gap of the strategies $( x _ { t } , y _ { t } )$ actually played at round t. Specifically, given a confidence parameter $\delta \in ( 0 , 1 )$ , we seek an algorithm that does not require the time horizon and satisfies

$$
\mathbb { P } ( \forall t \ge 1 : \ \mathrm { G a p } ( x _ { t } , y _ { t } ) \le B _ { d } ( t , \delta ) ) \ge 1 - \delta ,
$$

where $B _ { d } ( t , \delta )$ is a deterministic error bound that tends to zero as $t \to \infty$

## 3 Algorithm

Our algorithm proceeds in phases, starting from the uniform strategy pair. Each phase refines the pair returned by the preceding phase, aiming to reduce its duality gap by a constant factor while controlling the gap at every intermediate round.

Within each phase, we maintain an auxiliary strategy pair $( u _ { t } , v _ { t } )$ for learning and a played pair $( x _ { t } , y _ { t } )$ for collecting feedback. The played strategies are weighted averages of the auxiliary strategies and the baseline pair used to initialize the phase. This follows the idea of turning an internal average into the actual strategy, as in the A2L reduction of Cai et al. (2025). The dificulty is that the auxiliary players need loss estimates against each other, but observations come from the played pair. We address this mismatch by choosing the averaging weights adaptively and adding a ratio correction to the exponential weights update.

Phase structure. A phase starts from a positive baseline $( p , q )$ with duality gap at most $e ,$ where $e \in ( 0 , 2 ]$ is its accuracy parameter. Given a failure allowance $\rho \in ( 0 , 1 )$ , it returns a pair with gap at most $e / 2$ , while controlling the gap throughout the phase, with conditional probability at least $1 - \rho .$ . The returned pair serves as the baseline for the next phase, whose accuracy parameter is halved.

Starting from $p = q = \mathbf { 1 } / d ,$ we therefore run phases $k = 0 , 1 , \ldots$ . with $e _ { k } = 2 ^ { 1 - k }$ and $\rho _ { k } = \delta / 2 ^ { k + 1 }$ The failure allowances sum to $\delta .$ The universal bound ${ \mathrm { G a p } } ( p , q ) \leq 2$ establishes the initial accuracy, and each successful phase supplies the baseline guarantee for the next one. These guarantees allow us to use the prescribed accuracy levels without evaluating the unknown gap. We now describe a single phase with parameters $( e , \rho )$ , using $t = 1 , 2 , \ldots$ . to count rounds within that phase; this index restarts at one at the beginning of each phase. The following paragraphs present the averaging, loss estimation, and auxiliary update steps, followed by the parameter choices and stopping rule.

Adaptive averaging. Initialize $x _ { 1 } = u _ { 1 } = p , y _ { 1 } = v _ { 1 } = q$ , and $\tau _ { 1 } = \tau _ { 0 }$ , where $\tau _ { 0 } > 0$ is the initial weight assigned to the baseline. At each round, choose a weight $a _ { t } \in ( 0 , 1 ]$ and set

$$
\tau _ { t + 1 } = \tau _ { t } + a _ { t } , \qquad x _ { t + 1 } = \frac { \tau _ { t } x _ { t } + a _ { t } u _ { t } } { \tau _ { t + 1 } } , \qquad y _ { t + 1 } = \frac { \tau _ { t } y _ { t } + a _ { t } v _ { t } } { \tau _ { t + 1 } } .\tag{1}
$$

Thus the baseline continues to contribute to the played pair while the auxiliary learners collect enough information to improve it. To choose the weight, define the ratios

$$
r _ { t , i } = \frac { u _ { t , i } } { x _ { t , i } } , \qquad s _ { t , j } = \frac { v _ { t , j } } { y _ { t , j } } , \qquad S _ { x , t } = \sum _ { i = 1 } ^ { d } r _ { t , i } , \qquad S _ { y , t } = \sum _ { j = 1 } ^ { d } s _ { t , j } ,
$$

and let

$$
a _ { t } = \operatorname* { m i n } \left\{ 1 , \frac { d } { S _ { x , t } + S _ { y , t } } \right\} .\tag{2}
$$

A large ratio means that an auxiliary strategy puts more mass on an action than the corresponding sampling strategy does. Such actions require large importance weights, so we reduce both the auxiliary update and its contribution to the played average. The choice in (2) ensures $a _ { t } ( S _ { x , t } + S _ { y , t } ) \leq$ $d .$ Both players use the weight $a _ { t }$ , which preserves the cancellation of their payofs when their weighted regrets are added.

Estimating the auxiliary losses. The auxiliary players learn with the nonnegative loss vectors

$$
g _ { t } = \frac { 1 + A v _ { t } } { 2 } , \qquad h _ { t } = \frac { 1 - A ^ { \top } u _ { t } } { 2 } .
$$

In round $t ,$ draw $I _ { t } \sim x _ { t }$ and $J _ { t } \sim y _ { t }$ independently and observe $\left( I _ { t } , J _ { t } , R _ { t } \right)$ . A natural unbiased estimator of $g _ { t , i }$ is $\mathbf { 1 } \{ I _ { t } = i \} ( 1 + R _ { t } ) s _ { t , J _ { t } } / ( 2 x _ { t , i } )$ . The factor $1 / x _ { t , i }$ corrects for sampling the row action, while $s _ { t , J _ { t } } = v _ { t , J _ { t } } / y _ { t , J _ { t } }$ changes the opponent distribution from $y _ { t }$ to $v _ { t }$ . This estimator can have large variance when $x _ { t , i }$ is small or the opponent ratio $s _ { t , J _ { t } }$ is large. To control this variance, we use implicit exploration (IX), adapting the estimator of Neu (2015). For a learning rate $\eta > 0$ fixed within the phase, set $\zeta _ { t } = \eta a _ { t }$ and use

$$
\widehat { g } _ { t , i } = \frac { \mathbf { 1 } \{ I _ { t } = i \} ( 1 + R _ { t } ) s _ { t , J _ { t } } } { 2 ( x _ { t , i } + \zeta _ { t } s _ { t , J _ { t } } ) } , \qquad \widehat { h } _ { t , j } = \frac { \mathbf { 1 } \{ J _ { t } = j \} ( 1 - R _ { t } ) r _ { t , I _ { t } } } { 2 ( y _ { t , j } + \zeta _ { t } r _ { t , I _ { t } } ) } .\tag{3}
$$

The added denominator terms introduce a downward bias but ensure that each scaled estimate $\eta a _ { t } \widehat { g } _ { t , i }$ and $\eta a _ { t } \widehat { h } _ { t , j }$ lies in [0, 1], even when sampling probabilities are small. The IX correction scales with the opponent ratio and the adaptive weight $a _ { t }$ , matching the sampling correction to the size of the auxiliary update.

Exponential weights with ratio correction. We first compute $( x _ { t + 1 } , y _ { t + 1 } )$ from the current auxiliary pair $( u _ { t } , v _ { t } )$ via (1). Actions are sampled from $( x _ { t } , y _ { t } )$ , and the loss estimates in (3) use the current strategies $( x _ { t } , y _ { t } , u _ { t } , v _ { t } )$ . With these estimates, we update the auxiliary strategies by

$$
u _ { t + 1 , i } \propto u _ { t , i } \exp ( - \eta a _ { t } \widehat { g } _ { t , i } ) \frac { x _ { t , i } } { x _ { t + 1 , i } } , \qquad v _ { t + 1 , j } \propto v _ { t , j } \exp ( - \eta a _ { t } \widehat { h } _ { t , j } ) \frac { y _ { t , j } } { y _ { t + 1 , j } } .\tag{4}
$$

The correction factors $x _ { t , i } / x _ { t + 1 , i }$ and $y _ { t , j } / y _ { t + 1 , j }$ modify the usual exponential weights update. By (1),

$$
\frac { x _ { t , i } } { x _ { t + 1 , i } } = \frac { 1 + a _ { t } / \tau _ { t } } { 1 + ( a _ { t } / \tau _ { t } ) r _ { t , i } } .
$$

This identity shows that the correction downweights coordinates with large ratios between the auxiliary and played probabilities. The same holds for the column player. In the regret bound, these corrections produce a negative term measuring the discrepancy between the auxiliary and played distributions. Together with the adaptive weights in (2), this term absorbs the estimation variance of both players. For any fixed comparison action, the logarithms of the correction factors telescope across rounds, leaving only terms involving the initial and final played probabilities.

Phase parameters and termination. The initial weight $\tau _ { 0 }$ controls how strongly the played strategies retain the baseline, while the learning rate η controls the auxiliary updates and implicit exploration through $\zeta _ { t } = \eta a _ { t }$ . To choose them, define

$$
R _ { 0 } = \log \frac { 1 } { \operatorname* { m i n } _ { i } p _ { i } } + \log \frac { 1 } { \operatorname* { m i n } _ { j } q _ { j } } , \qquad L = \log \frac { 2 d + 2 } { \rho } , \qquad H = R _ { 0 } + \log 5 + 2 L .
$$

Here $R _ { 0 }$ accounts for the initial comparison cost associated with small baseline probabilities, L accounts for the failure allowance $\rho ,$ and H collects these logarithmic terms. We set

$$
\tau _ { 0 } = \operatorname * { m a x } \left\{ 6 d , \left\lceil \frac { 1 6 3 8 4 d H ^ { 2 } } { e ^ { 2 } } \right\rceil \right\} , \qquad \eta = \frac { 1 6 H } { e \tau _ { 0 } } .\tag{5}
$$

The baseline weight keeps the played strategies close enough to the baseline while the auxiliary learners improve their average. The corresponding learning rate allows the ratio correction to control estimation error while keeping the auxiliary regret small enough to reduce the gap. Here 16384 is a suficiently large constant.

We end the phase at the first round N with $\tau _ { N + 1 } \geq 4 \tau _ { 0 } .$ , or equivalently $\textstyle \sum _ { t = 1 } ^ { N } a _ { t } \geq 3 \tau _ { 0 }$ , and return $\left( x _ { N + 1 } , y _ { N + 1 } \right)$ . The auxiliary strategies then contribute at least three times the baseline weight. With the above parameter choices, this is enough to reduce the gap below $e / 2 ,$ , while keeping it at most $5 e / 4$ throughout the phase, with conditional probability at least $1 - \rho .$

Algorithm 1 gives the complete procedure. Both players can compute the played and auxiliary strategies from their common observations and the input $( d , \delta )$ . The updates require no additional communication or shared randomness. All coordinates remain positive, and the algorithm stores only a constant number of length-d vectors. Computing the weights, sampling actions, and performing the normalized updates takes $\mathcal O ( d )$ operations per round, with $\mathcal O ( d )$ memory.

## Algorithm 1. Phased exponential weights with ratio correction

Input: Number of actions $d \geq 2$ and confidence $\delta \in ( 0 , 1 )$   
Initialize $p = q = \mathbf { 1 } / d .$   
for $k = 0 , 1 , 2 , . . .$ . do   
Set $e = 2 ^ { 1 - k }$ and $\rho = \delta / 2 ^ { k + 1 }$ ; compute $\tau _ { 0 } , \eta$ by (5).   
Initialize $x = u = p , y = v = q .$ , and $\tau = \tau _ { 0 } .$   
while $\tau < 4 \tau _ { 0 }$ do   
Compute $r _ { i } = u _ { i } / x _ { i } , s _ { j } = v _ { j } / y _ { j }$ , and a by (2); set $\zeta = \eta a .$   
Form $\tau ^ { + } = \tau + a , x ^ { + } \stackrel { - } { = } ( \tau \dot { x } + \stackrel { - } { a } u ) / \tau ^ { + }$ , and $y ^ { + } = ( \tau y + a v ) / \tau ^ { + }$   
Independently draw $I \sim x$ and $J \sim y ;$ observe $( I , J , R )$   
Compute $\widehat { g } , \widehat { h }$ by (3) using $x , y , u ,$ v and $( I , J , R )$   
Compute $u ^ { + } , v ^ { + }$ by (4) using $x , y , x ^ { + } , y ^ { + }$ and normalize.   
Set $( x , y , u , v , \tau )  ( x ^ { + } , y ^ { + } , u ^ { + } , v ^ { + } , \tau ^ { + } )$   
end while   
Set $( p , q )  ( x , y )$   
end for

Comparison with prior algorithms. Compared with Hait et al. (2026), our algorithm controls estimation variance directly through adaptive averaging and ratio correction. Their method uses log-barrier regularization of an estimated payof matrix to control estimation error. We instead estimate the auxiliary loss vectors directly: the weights $a _ { t }$ limit the aggregate importance ratios across both players, and the ratio correction downweights coordinates with large ratios between auxiliary and played probabilities. By jointly controlling these ratios and the resulting variance, our method improves the dependence on the number of actions.

This design also gives simpler updates. Whereas Hait et al. (2026) solve a regularized game at each epoch boundary and keep the strategies fixed within the epoch, our algorithm performs explicit vector updates each round. It requires only $\mathcal O ( d )$ time and memory per round, without constructing a payof matrix estimate or solving a regularized game.

## 4 Theoretical Guarantees and Analysis

## 4.1 Last-iterate Guarantees

We first state the convergence guarantee for Algorithm 1. In the following guarantee, t counts the total number of interaction rounds across all phases.

Theorem 1 (High-probability last-iterate convergence). Fix $d \geq 2 , A \in [ - 1 , 1 ] ^ { d \times d }$ , and $\delta \in ( 0 , 1 )$ Under the feedback model in Section 2, with probability at least $1 - \delta$ , the played strategies of

Algorithm 1 satisfy, simultaneously for all $t \geq 1$

$$
\mathrm { G a p } ( x _ { t } , y _ { t } ) \leq C \sqrt { \frac { d } { t } } \left[ \log \left( \frac { d t } { \delta } \right) \right] ^ { 3 / 2 } ,
$$

where $C > 0$ is a universal constant.

Theorem 1 gives a high-probability last-iterate guarantee for the mixed strategies actually used to sample actions. Although the played strategies are constructed by averaging auxiliary strategies, the bound applies directly to $( x _ { t } , y _ { t } )$ at round t. Consequently, for any target accuracy $\varepsilon \in ( 0 , 1 ]$ after $\widetilde { \mathcal { O } } ( d / \varepsilon ^ { 2 } )$ rounds, every subsequent played pair is an ε-Nash equilibrium.

Under the same feedback model, Hait et al. (2026, Theorem 4.1) establish a high-probability bound of $\widetilde { \mathcal { O } } ( d ^ { 2 } / \sqrt { t } )$ simultaneously for all rounds. Theorem 1 preserves the $t ^ { - 1 / 2 }$ dependence and improves the polynomial dependence on the number of actions from $d ^ { 2 }$ to ${ \sqrt { d } } ,$ a factor of $d ^ { 3 / 2 }$ . In terms of observations needed to reach accuracy $\varepsilon ,$ the dimension dependence improves from $\widetilde { \mathcal { O } } ( d ^ { 4 } / \varepsilon ^ { 2 } )$ to $\widetilde { \mathcal { O } } ( d / \varepsilon ^ { 2 } )$ . This improvement comes from matching the adaptive averaging weights to the ratio correction. The weights control the aggregate importance ratios of both players, while the correction absorbs the estimation variance caused by diferences between auxiliary and played strategies. By controlling the variance of the auxiliary loss estimates directly, we avoid the extra dimension factors incurred when transferring entrywise payof estimates between successive strategy pairs.

The dependence on d and t is minimax optimal up to logarithmic factors, by the standard bandit pure-exploration lower bound (Mannor and Tsitsiklis, 2004, Theorem 1). With identical columns, the gap is the row strategy’s excess loss in a d-armed bandit, and column actions provide no additional information. By Markov’s inequality, sampling from an ε-optimal mixture returns a cε-optimal arm with probability at least $1 - 1 / c$ for any constant $c > 1$ . Thus, achieving gap at most ε with suficiently high fixed confidence requires $\Omega ( d / \varepsilon ^ { 2 } )$ observations in the worst case. Equivalently, for $t \geq d ,$ the worst-case gap cannot improve on the $\Omega ( { \sqrt { d / t } } )$ scale. Theorem 1 matches this dependence while providing a guarantee simultaneously for every round without knowing the horizon. To the best of our knowledge, this is the first work to achieve high-probability last-iterate convergence that is minimax optimal in both d and t, up to logarithmic factors, under this feedback model.

## 4.2 Theoretical Analysis

The proof follows the algorithm’s phase structure. Within each phase, the adaptive weights allow the ratio correction to absorb the estimation variance, yielding a joint weighted regret bound for the auxiliary learners. This bound controls the gap of their weighted average. Mixing that average with the baseline then controls every intermediate played pair and produces a more accurate baseline for the next phase. This progress is measured in accumulated averaging weight, so we also bound the number of interaction rounds needed to complete each phase. Combining the duration bound with the geometrically decreasing accuracy parameters yields the global last-iterate guarantee.

We develop the single-phase analysis in three steps: bounding the joint weighted regret, converting this bound into a gap guarantee for the played strategies, and controlling the phase duration. We then combine the phases to prove Theorem 1. The full proofs are in Appendix A.

We begin by bounding the auxiliary learners’ regret within a phase initialized at a positive baseline $( p , q )$ , with accuracy parameter e and failure allowance $\rho .$ Here and in the next two steps, t counts rounds within the phase and probabilities are conditional on the history at its start.

We study the sum of the learners’ regrets with the weights $a _ { t } ,$ , since this joint regret controls the duality gap of their weighted average. The following lemma bounds it for every prefix up to the phase’s stopping round, denoted by N.

Lemma 2 (Joint regret bound). With probability at least $1 - \rho ,$ , for every prefix $1 \leq n \leq N$ and every $i , j \in [ d ]$ ，

$$
\sum _ { t = 1 } ^ { n } a _ { t } \big ( \langle u _ { t } , g _ { t } \rangle - g _ { t , i } + \langle v _ { t } , h _ { t } \rangle - h _ { t , j } \big ) \leq \frac { 2 H } { \eta } .\tag{6}
$$

This bound applies to the true losses $g _ { t } , h _ { t }$ , although the learners update using IX estimates sampled from the played pair. Its right side depends only on the phase parameters and contains no accumulated variance term. This allows the auxiliary average to become more accurate as its total weight grows, even when the auxiliary and played strategies difer substantially.

To establish this bound, we first identify the variance cost that must be controlled. Define

$$
K _ { x , t } = \sum _ { i } \frac { u _ { t , i } ^ { 2 } } { x _ { t , i } } , \qquad K _ { y , t } = \sum _ { j } \frac { v _ { t , j } ^ { 2 } } { y _ { t , j } } .
$$

Each K measures the discrepancy between an auxiliary strategy and its sampling distribution, and equals one when they agree. The IX concentration argument gives a variance cost proportional to $\begin{array} { r } { \eta \sum _ { t } a _ { t } ^ { 2 } ( S _ { x , t } K _ { y , t } + S _ { y , t } K _ { x , t } ) } \end{array}$ . To absorb this cost, we seek a negative contribution to regret with the same dependence on $K _ { x , t }$ and $K _ { y , t }$

The ratio correction provides such a contribution. We derive it first for the row player. Writing $\kappa _ { t } = a _ { t } / \tau _ { t } , ( 1 )$ gives $x _ { t , i } / x _ { t + 1 , i } = ( 1 + \kappa _ { t } ) / ( 1 + \kappa _ { t } r _ { t , i } )$ . The common factor $1 + \kappa _ { t }$ cancels in the auxiliary update, so this update is exponential weights with modified costs $\eta a _ { t } \widehat { g } _ { t , i } + c _ { t , i }$ , where $c _ { t , i } = \log ( 1 + \kappa _ { t } r _ { t , i } )$ . The exponential weights regret inequality includes the learner’s correction cost $\left. u _ { t } , c _ { t } \right.$ minus the comparison action’s cost $c _ { t , i }$ . Moving these costs to the right side introduces

$$
{ \frac { 1 } { \eta } } \sum _ { t = 1 } ^ { n } ( c _ { t , i } - \langle u _ { t } , c _ { t } \rangle )
$$

in the bound on the row player’s weighted regret for the estimated losses. The comparison action’s contribution telescopes: $\begin{array} { r } { \sum _ { t = 1 } ^ { n } c _ { t , i } = \log ( \tau _ { n + 1 } x _ { n + 1 , i } / ( \tau _ { 0 } p _ { i } ) ) } \end{array}$ , leaving only a logarithmic cost. The learner’s contribution has a negative sign, and $\log ( 1 + z ) \geq z - z ^ { 2 } / 2$ gives

$$
- \langle u _ { t } , c _ { t } \rangle \leq - \kappa _ { t } K _ { x , t } + \frac { \kappa _ { t } ^ { 2 } } { 2 } \sum _ { \ell = 1 } ^ { d } u _ { t , \ell } r _ { t , \ell } ^ { 2 } .
$$

Thus the row player’s regret bound contains $- \eta ^ { - 1 } \sum _ { t } \kappa _ { t } K _ { x , t } ,$ , together with quadratic remainders. Applying the same argument to the column player gives a regret bound containing $- \eta ^ { - 1 } \sum _ { t } \kappa _ { t } K _ { y , t }$

We now add the row and column players’ regret bounds for the estimated losses. To obtain regret for the true losses, the IX concentration argument controls the estimation error together with the second-order loss terms from exponential weights. This introduces the variance cost identified above. After using $\tau _ { 0 } \geq 6 d$ to control the quadratic costs of the ratio correction, the left side of (6) is bounded by

$$
\frac { 2 H } { \eta } + 3 \eta \sum _ { t = 1 } ^ { n } { a _ { t } ^ { 2 } ( S _ { x , t } K _ { y , t } + S _ { y , t } K _ { x , t } ) } - \frac { 3 } { 4 \eta } \sum _ { t = 1 } ^ { n } { \kappa _ { t } ( K _ { x , t } + K _ { y , t } ) } .
$$

It remains to show that the negative correction term in this bound absorbs the positive variance cost. The adaptive weights satisfy $a _ { t } ( S _ { x , t } + S _ { y , t } ) \leq d ;$ giving

$$
a _ { t } ^ { 2 } ( S _ { x , t } K _ { y , t } + S _ { y , t } K _ { x , t } ) \leq d a _ { t } ( K _ { x , t } + K _ { y , t } ) = d \tau _ { t } \kappa _ { t } ( K _ { x , t } + K _ { y , t } ) .
$$

This inequality limits the contribution of the potentially large sums $S _ { x , t }$ and $S _ { y , t }$ to a single factor d. The remaining dependence on $\kappa _ { t } ( K _ { x , t } + K _ { y , t } )$ matches that of the negative correction term. Since the parameter choice ensures $\eta ^ { 2 } d \tau _ { t } \leq 1 / 4$ throughout the phase, the variance cost is absorbed and only $2 H / \eta$ remains. Appendix A.1 gives the full argument.

We next transfer this regret guarantee to the played strategies by tracking the weight contributed by the auxiliary learners. Let $\textstyle A _ { n } = \sum _ { t = 1 } ^ { n } a _ { t }$ denote their cumulative averaging weight after n rounds, with $A _ { 0 } = 0$ . Together with the baseline weight $\tau _ { 0 }$ , this gives total weight $\tau _ { n + 1 } = \tau _ { 0 } + A _ { n } ,$ and the phase stops when $A _ { N } \geq 3 \tau _ { 0 }$

Early in the phase, the auxiliary average may still be inaccurate, so we also use the accuracy of the baseline. The following lemma shows that mixing with the baseline controls every intermediate gap and produces a more accurate pair at the stopping round.

Lemma 3 (Duality gap bound). Suppose ${ \mathrm { G a p } } ( p , q ) \leq e$ . On the event of Lemma 2, every $0 \leq n \leq N$ satisfies

$$
\mathrm { G a p } ( x _ { n + 1 } , y _ { n + 1 } ) \le \frac { \tau _ { 0 } e + 4 H / \eta } { \tau _ { 0 } + A _ { n } } = \frac { 5 e \tau _ { 0 } } { 4 ( \tau _ { 0 } + A _ { n } ) } .\tag{7}
$$

In particular, the gap is at most 5e/4 throughout the phase, and the returned pair has gap at most $5 e / 1 6 < e / 2$

The two conclusions serve distinct roles. The intermediate bound preserves accuracy during the rounds used to learn a better baseline, while the endpoint contraction allows the next phase to start with accuracy parameter $e / 2$ . Together, these bounds control the gap of every played pair across successive phases, which is essential for the last-iterate guarantee.

To see why the lemma holds, let $\begin{array} { r } { \bar { u } _ { n } = A _ { n } ^ { - 1 } \sum _ { t = 1 } ^ { n } { a _ { t } u _ { t } } } \end{array}$ and $\textstyle { \bar { v } } _ { n } = A _ { n } ^ { - 1 } \sum _ { t = 1 } ^ { n } a _ { t } v _ { t }$ for $n \geq 1$ . Using the weights $a _ { t }$ for both players makes their bilinear payofs cancel:

$$
\frac { A _ { n } } { 2 } \operatorname { G a p } ( \bar { u } _ { n } , \bar { v } _ { n } ) = \operatorname* { m a x } _ { i , j } \sum _ { t = 1 } ^ { n } a _ { t } \big ( \langle u _ { t } , g _ { t } \rangle - g _ { t , i } + \langle v _ { t } , h _ { t } \rangle - h _ { t , j } \big ) .
$$

Lemma 2 therefore gives $A _ { n } \mathrm { G a p } ( \bar { u } _ { n } , \bar { v } _ { n } ) \leq 4 H / \eta$ . The played pair mixes this average with the baseline using weights $A _ { n }$ and $\tau _ { 0 }$ . Convexity of the gap and the parameter choice $4 H / \eta = e \tau _ { 0 } / 4$ then yield (7); see Appendix $\mathrm { A . 2 }$

Lemma 3 measures progress in accumulated weight $A _ { n }$ . To obtain a convergence rate in interaction rounds, we must bound how long the phase takes to reach $A _ { N } \geq 3 \tau _ { 0 }$ , even when individual weights are small. The next lemma provides this bound and controls the smallest coordinates of the returned pair, which determine the next phase’s parameters.

Lemma 4 (Phase length and coordinate bounds). For every positive baseline $( p , q )$ , the following bounds hold.

(i) Phase length. The number of interaction rounds satisfies

$$
N = \mathcal { O } \big ( \tau _ { 0 } ( 1 + R _ { 0 } ) \big ) = \mathcal { O } \bigg ( \frac { d H ^ { 3 } } { e ^ { 2 } } \bigg ) .
$$

(ii) Coordinate lower bounds. For every $0 \leq n \leq N$ and every $i , j \in [ d ]$ , the played strategies satisfy

$$
x _ { n + 1 , i } \geq { \frac { p _ { i } } { 5 } } , \qquad y _ { n + 1 , j } \geq { \frac { q _ { j } } { 5 } } .
$$

The duration bound shows that adaptive weighting costs only the logarithmic factor $1 + R _ { 0 }$ beyond the baseline weight $\tau _ { 0 }$ . The coordinate bounds also limit the increase of $R _ { 0 }$ from one

phase to the next to 2 log 5. These two properties keep the dependence on baseline probabilities logarithmic across phases, preserving the polynomial dependence $d / e ^ { 2 }$ in the number of rounds needed to improve accuracy.

The proof tracks the unnormalized played coordinates $M _ { t , i } = \tau _ { t } x _ { t , i }$ . Their logarithmic increments satisfy

$$
M _ { t + 1 , i } = M _ { t , i } + a _ { t } u _ { t , i } , \qquad \log \frac { M _ { t + 1 , i } } { M _ { t , i } } = \log ( 1 + \kappa _ { t } r _ { t , i } ) .
$$

Summing these increments telescopes. Applying the same argument to the column player and using $1 \leq a _ { t } + ( a _ { t } / d ) ( S _ { x , t } + S _ { y , t } )$ gives

$$
n \leq A _ { n } + \tau _ { n + 1 } \left( 1 + \frac { d } { \tau _ { 0 } } \right) \left( R _ { 0 } + 2 \log \frac { \tau _ { n + 1 } } { \tau _ { 0 } } \right) .
$$

The stopping rule bounds $A _ { N }$ and $\tau _ { N + 1 }$ by constant multiples of $\tau _ { 0 }$ , yielding the duration bound.   
The coordinate bounds follow from monotonicity of $M _ { t , i }$ and the same bound on total weight.   
Appendix A.3 gives the details.

Finally, let $N _ { k }$ be the duration of phase k, and let $R _ { 0 , k } , H _ { k }$ , and $\tau _ { 0 , k }$ denote the corresponding phase parameters. The initial baseline coordinates are $1 / d ,$ and Lemma 4 ensures that each new baseline coordinate is at least one fifth of its previous value. After k phases, they are therefore at least $1 / ( d 5 ^ { k } )$ , giving $R _ { 0 , k } \leq 2$ log d + 2k log 5. Writing $B _ { k } = \log ( d / \delta ) + k + 1$ , we obtain $H _ { k } = \mathcal { O } ( B _ { k } )$ and $N _ { k } = \mathcal { O } ( d B _ { k } ^ { 3 } / e _ { k } ^ { 2 } )$ . If global round $T$ lies in phase $k ,$ the geometric accuracy schedule yields

$$
T \leq \sum _ { \ell = 0 } ^ { k } N _ { \ell } = \mathcal { O } \Bigg ( \frac { d B _ { k } ^ { 3 } } { e _ { k } ^ { 2 } } \Bigg ) .
$$

Moreover, $N _ { \ell } \geq 3 \tau _ { 0 , \ell } \geq 4 ^ { \ell }$ , so $k = \mathcal { O } ( \log T + 1 )$ . The phase failure allowances sum to $\delta ,$ and Lemma 3 gives gap at most $5 e _ { k } / 4$ in every successful phase. Substituting the bound on $T$ and $B _ { k } = \mathcal { O } ( \log ( d T / \delta ) )$ proves Theorem 1.

## 5 Related Work

Last-iterate convergence. With exact gradient feedback, last-iterate convergence is well understood in several classes of games: optimistic gradient methods achieve instance-dependent linear rates in matrix games (Wei et al., 2020), and accelerated methods attain an $\mathcal { O } ( 1 / t )$ rate in smooth monotone games (Cai and Zheng, 2023). With bandit payof feedback, learning must also account for estimation error and continued exploration. For unknown matrix games without observed opponent actions, Cai et al. (2023, 2025); Fiegel et al. (2026b) establish high-probability last-iterate guarantees with rates scaling as $t ^ { - 1 / 8 } , t ^ { - \dot { 1 } / 5 }$ , and $t ^ { - 1 / 4 }$ , respectively, up to dimension-dependent and logarithmic factors. Other works study last-iterate convergence under diferent convergence criteria or additional assumptions. Fiegel et al. (2026a) obtain an $\widetilde { \mathcal { O } } ( ( d / t ) ^ { 1 / 4 } )$ rate for the $L ^ { 2 }$ norm of the duality gap, which controls its second moment at each round rather than providing a single high-probability event covering all rounds. Ito et al. (2025) prove last-iterate convergence in expectation under a unique pure-strategy Nash equilibrium assumption. Payof-based best-response dynamics also admit finite-sample guarantees in stochastic and polymatrix games (Chen et al., 2024; Faizal et al., 2024). A separate line studies zeroth-order feedback in continuous action spaces (Dong et al., 2025; Maiti et al., 2026), where players observe evaluations of the payof function at their chosen continuous actions, rather than sampled matrix entries as in our setting.

Self-play learning in games. Self-play is a standard approach to learning strategies in games, with algorithms based on regret minimization for equilibrium computation and reinforcement learning for sequential decision making (Freund and Schapire, 1999; Zinkevich et al., 2007; Bai and Jin, 2020; Liu et al., 2021; Zhang et al., 2026). It has enabled strong empirical performance in board games and poker (Silver et al., 2018; Brown and Sandholm, 2019). More recently, self-play has been applied to language model alignment with general preferences, where potentially nontransitive comparisons motivate learning a Nash policy (Munos et al., 2023; Zhang et al., 2025b,a; Wu et al., 2025). In self-play, the sampled actions of both players can be recorded, making observed opponent actions a natural feedback model. We study this setting in unknown two-player zero-sum matrix games with bandit payof feedback. Under this feedback model, O’Donoghue et al. (2021) bound cumulative regret relative to the game value against arbitrary opponents, whereas we study lastiterate convergence to the equilibrium. The closest work to ours, Hait et al. (2026), establishes a high-probability last-iterate rate of $\widetilde { \mathcal { O } } ( d ^ { 2 } / \sqrt { t } )$ . We improve this rate to $\tilde { \mathcal { O } } ( \sqrt { d / t } )$ , matching the minimax lower bound up to logarithmic factors in both the number of actions and the number of rounds.

## 6 Conclusion

We study last-iterate convergence in unknown two-player zero-sum matrix games with bandit payof feedback and observed opponent actions. We develop an algorithm that combines adaptive averaging with corrected exponential weights to control estimation variance. For games with d actions per player, our algorithm achieves a duality gap of $\widetilde { \mathcal { O } } ( \sqrt { d / t } )$ with high probability uniformly over all rounds t. This improves the $\widetilde { \mathcal { O } } ( d ^ { 2 } / \sqrt { t } )$ bound of Hait et al. (2026) by a factor of $d ^ { 3 / 2 }$ and matches the minimax lower bound in both d and t up to logarithmic factors.

## AI use statement

We use GPT-6 Astra to polish the writing, assist with calculations in the proofs, and check their correctness. We review all AI-assisted content and take full responsibility for the final content o this paper.

## References

Yu Bai and Chi Jin. Provable self-play algorithms for competitive reinforcement learning. In International conference on machine learning, pages 551–560. PMLR, 2020.

Noam Brown and Tuomas Sandholm. Superhuman ai for multiplayer poker. Science, 365(6456): 885–890, 2019.

Yang Cai and Weiqiang Zheng. Doubly optimal no-regret learning in monotone games. In International Conference on Machine Learning, pages 3507–3524. PMLR, 2023.

Yang Cai, Haipeng Luo, Chen-Yu Wei, and Weiqiang Zheng. Uncoupled and convergent learning in two-player zero-sum markov games with bandit feedback. Advances in Neural Information Processing Systems, 36:36364–36406, 2023.

Yang Cai, Haipeng Luo, Chen-Yu Wei, and Weiqiang Zheng. From average-iterate to last-iterate convergence in games: A reduction and its applications. Advances in Neural Information Processing Systems, 38:46937–46967, 2025.

Zaiwei Chen, Kaiqing Zhang, Eric Mazumdar, Asuman Ozdaglar, and Adam Wierman. Decentralized best-response-based learning in two-player zero-sum stochastic games: A finite-sample analysis. arXiv preprint arXiv:2409.01447, 2024.

Jing Dong, Baoxiang Wang, and Yaoliang Yu. Uncoupled and convergent learning in monotone games under bandit feedback. Advances in Neural Information Processing Systems, 38:151665–151683, 2025.

Fathima Zarin Faizal, Asuman Ozdaglar, and Martin J Wainwright. Finite-sample guarantees for learning dynamics in zero-sum polymatrix games. arXiv preprint arXiv:2407.20128, 2024.

Côme Fiegel, Pierre Menard, Tadashi Kozuno, Michal Valko, and Vianney Perchet. The harder path: Last iterate convergence for uncoupled learning in zero-sum games with bandit feedback. arXiv preprint arXiv:2604.16087, 2026a.

Come Fiegel, Pierre Menard, Tadashi Kozuno, Michal Valko, and Vianney Perchet. Optimal lastiterate convergence in matrix games with bandit feedback using the log-barrier. arXiv preprint arXiv:2604.15242, 2026b.

Yoav Freund and Robert E Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 29(1-2):79–103, 1999.

Soumita Hait, Ping Li, Haipeng Luo, and Mengxiao Zhang. Near-optimal last-iterate convergence for zero-sum games with bandit feedback and opponent actions. arXiv preprint arXiv:2605.09363, 2026.

Shinji Ito, Haipeng Luo, Taira Tsuchiya, and Yue Wu. Instance-dependent regret bounds for learning two-player zero-sum games with bandit feedback. arXiv preprint arXiv:2502.17625, 2025.

Qinghua Liu, Tiancheng Yu, Yu Bai, and Chi Jin. A sharp analysis of model-based reinforcement learning with self-play. In International Conference on Machine Learning, pages 7001–7010. PMLR, 2021.

Arnab Maiti, Claire Jie Zhang, Kevin Jamieson, Jamie Heather Morgenstern, Ioannis Panageas, and Lillian J. Ratlif. Eficient uncoupled learning dynamics with ${ \tilde { O } } ( T ^ { - 1 / 4 } )$ last-iterate convergence in bilinear saddle-point problems over convex sets under bandit feedback. In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 2431–2439. PMLR, 2026.

Shie Mannor and John N Tsitsiklis. The sample complexity of exploration in the multi-armed bandit problem. Journal of Machine Learning Research, 5(Jun):623–648, 2004.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 2023.

Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. Advances in Neural Information Processing Systems, 28, 2015.

Brendan O’Donoghue, Tor Lattimore, and Ian Osband. Matrix games with bandit feedback. In Uncertainty in Artificial Intelligence, pages 279–289. PMLR, 2021.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419): 1140–1144, 2018.

Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, and Haipeng Luo. Linear last-iterate convergence in constrained saddle-point optimization. arXiv preprint arXiv:2006.09517, 2020.

Yue Wu, Zhiqing Sun, Rina Hughes, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. In International Conference on Learning Representations, volume 2025, pages 91558–91582, 2025.

Yuheng Zhang, Dian Yu, Tao Ge, Linfeng Song, Zhichen Zeng, Haitao Mi, Nan Jiang, and Dong Yu. Improving llm general preference alignment via optimistic online mirror descent. Advances in Neural Information Processing Systems, 38:160165–160187, 2025a.

Yuheng Zhang, Dian Yu, Baolin Peng, Linfeng Song, Ye Tian, Mingyue Huo, Nan Jiang, Haitao Mi, and Dong Yu. Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning. In International Conference on Learning Representations, volume 2025, pages 31833–31849, 2025b.

Yuheng Zhang, Claire Chen, and Nan Jiang. Beyond pessimism: ofline learning in kl-regularized games. arXiv preprint arXiv:2604.06738, 2026.

Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. In Advances in Neural Information Processing Systems, volume 20, 2007.

## A Proofs of the theoretical guarantees

We use the notation of Section 4.2. Throughout Appendices A.1–A.3, t is the local round index and all probabilistic statements are conditional on the history at the start of the phase. The baseline and phase parameters are fixed under this conditioning. If the phase starts after global round $\sigma ,$ write $\mathcal { H } _ { t } = \mathcal { F } _ { \sigma + t }$ and $\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { H } _ { t - 1 } ]$

## A.1 Joint weighted regret

## A.1.1 Conditional concentration of the loss estimates

The estimates in (3) are biased downward. We need a bound for each comparison action and a separate bound for the learner’s aggregate estimation error. To retain the variance term that wil later be absorbed, define

$$
\beta _ { t } ^ { x } = \sum _ { i , j } \frac { u _ { t , i } v _ { t , j } ^ { 2 } } { x _ { t , i } y _ { t , j } + \zeta _ { t } v _ { t , j } } , \qquad \beta _ { t } ^ { y } = \sum _ { i , j } \frac { u _ { t , i } ^ { 2 } v _ { t , j } } { x _ { t , i } y _ { t , j } + \zeta _ { t } u _ { t , i } } .\tag{8}
$$

These quantities are used only in the analysis and are not computed by the algorithm. We also introduce the corrected learner losses

$$
\boldsymbol { W } _ { t } ^ { x } = a _ { t } \langle \boldsymbol { u } _ { t } , \widehat { \boldsymbol { g } } _ { t } \rangle - \eta a _ { t } ^ { 2 } \langle \boldsymbol { u } _ { t } , \widehat { \boldsymbol { g } } _ { t } ^ { 2 } \rangle , \qquad \boldsymbol { W } _ { t } ^ { y } = a _ { t } \langle \boldsymbol { v } _ { t } , \widehat { \boldsymbol { h } } _ { t } \rangle - \eta a _ { t } ^ { 2 } \langle \boldsymbol { v } _ { t } , \widehat { \boldsymbol { h } } _ { t } ^ { 2 } \rangle .
$$

The subtracted quadratic terms coincide with the second-order cost in the entropy update, allowing us to analyze the estimation and update costs together.

To pass from estimated regret to true regret, we need to control overestimation of each comparison action’s loss and underestimation of the learner’s loss. The following lemma provides both bounds by adapting the exponential moment argument for implicit exploration (Neu, 2015) to the two-player estimates and predictable weights.

Lemma 5 (Simultaneous estimation bounds). With probability at least $1 - \rho$ , the following inequalities hold for every prefix $n \leq N$ and every $i , j \in [ d ]$ :

$$
\sum _ { t = 1 } ^ { n } a _ { t } ( \widehat { g } _ { t , i } - g _ { t , i } ) \leq \frac { L } { \eta } , \qquad \sum _ { t = 1 } ^ { n } a _ { t } ( \widehat { h } _ { t , j } - h _ { t , j } ) \leq \frac { L } { \eta } ,\tag{9}
$$

and

$$
\begin{array} { r } { \displaystyle \sum _ { t = 1 } ^ { n } \bigl ( a _ { t } \langle u _ { t } , g _ { t } \rangle - W _ { t } ^ { x } \bigr ) \leq 3 \eta \displaystyle \sum _ { t = 1 } ^ { n } a _ { t } ^ { 2 } \beta _ { t } ^ { x } + \frac { L } { \eta } , } \\ { \displaystyle \sum _ { t = 1 } ^ { n } \bigl ( a _ { t } \langle v _ { t } , h _ { t } \rangle - W _ { t } ^ { y } \bigr ) \leq 3 \eta \displaystyle \sum _ { t = 1 } ^ { n } a _ { t } ^ { 2 } \beta _ { t } ^ { y } + \frac { L } { \eta } . } \end{array}\tag{10}
$$

The two bounds account for the diferent roles of the estimated losses in regret. Comparison actions incur only the confidence cost $L / \eta$ , while the learner bounds retain the variance terms $\beta _ { t } ^ { x } , \beta _ { t } ^ { y }$ Keeping these terms explicit allows the ratio correction to absorb them in the proof of Lemma 2.

Proof. We prove the statements for the row player; the column proof replaces $( 1 + R _ { t } ) / 2 \mathrm { b y } ( 1 - R _ { t } ) / 2$ Suppress the time index, write $\ell = ( 1 + R ) / 2$ , and set

$$
\mu _ { i j } = \frac { 1 + A _ { i j } } { 2 } , \qquad D _ { i j } = x _ { i } y _ { j } + \zeta v _ { j } .
$$

Then $\widehat { g } _ { i } = \mathbf { 1 } \{ I = i \} \ell v _ { J } / D _ { i J }$ and $\begin{array} { r } { g _ { i } = \sum _ { j } v _ { j } \mu _ { i j } } \end{array}$ . Conditioning on the current action pair gives

$$
g _ { i } - \mathbb { E } _ { t } \widehat { g } _ { i } = \zeta \sum _ { j } \frac { v _ { j } ^ { 2 } \mu _ { i j } } { D _ { i j } } \geq 0 .
$$

Since $0 \leq \ell ^ { 2 } \leq \ell \leq 1$ , the second moment satisfies

$$
\mathbb { E } _ { t } \langle u , \widehat g ^ { 2 } \rangle \leq \sum _ { i , j } \frac { u _ { i } x _ { i } y _ { j } v _ { j } ^ { 2 } } { D _ { i j } ^ { 2 } } \leq \sum _ { i , j } \frac { u _ { i } v _ { j } ^ { 2 } } { D _ { i j } } = \beta ^ { x } .
$$

Together with the bias identity above and $D _ { i j } \geq \zeta v _ { j }$ , this yields

$$
\begin{array} { r } { \langle u , g - \mathbb { E } _ { t } \widehat { g } \rangle \leq \zeta \beta ^ { x } , \qquad \mathbb { E } _ { t } \langle u , \widehat { g } ^ { 2 } \rangle \leq \beta ^ { x } , \qquad 0 \leq \zeta \widehat { g } _ { i } \leq 1 . } \end{array}\tag{11}
$$

For a comparison coordinate, we need the stronger moment inequality

$$
\mathbb { E } _ { t } [ \widehat { g } _ { i } + \zeta \widehat { g } _ { i } ^ { 2 } ] \leq \sum _ { j } v _ { j } \mu _ { i j } \left( \frac { x _ { i } y _ { j } } { D _ { i j } } + \frac { \zeta x _ { i } y _ { j } v _ { j } } { D _ { i j } ^ { 2 } } \right) \leq g _ { i } .
$$

Indeed, for $b , c \geq 0$ with $b + c > 0 , b / ( b + c ) + b c / ( b + c ) ^ { 2 } = 1 - c ^ { 2 } / ( b + c ) ^ { 2 } \leq 1$ . Using $e ^ { z } \leq 1 + z + z ^ { 2 }$ for $0 \leq z \leq 1$ therefore yields

$$
\mathbb { E } _ { t } \exp \{ \zeta ( \widehat { g } _ { i } - g _ { i } ) \} \leq e ^ { - \zeta g _ { i } } ( 1 + \zeta g _ { i } ) \leq 1 .
$$

Because $\zeta _ { t } = \eta a _ { t }$ , the process $\textstyle \exp ( \eta \sum _ { t = 1 } ^ { n } a _ { t } ( \widehat { g } _ { t , i } - g _ { t , i } ) )$ is a nonnegative supermartingale starting at one, up to the end of the phase.

For the learner, (11) implies

$$
W ^ { x } = a \sum _ { i } u _ { i } \widehat { g } _ { i } ( 1 - \zeta \widehat { g } _ { i } ) \geq 0 , \qquad \mathbb { E } _ { t } W ^ { x } \leq a \leq 1 .
$$

Furthermore, Jensen’s inequality and $W ^ { x } \leq a \langle u , { \widehat { g } } \rangle$ give

$$
\mathbb { E } _ { t } [ ( W ^ { x } ) ^ { 2 } ] \le a ^ { 2 } \beta ^ { x } , \qquad a \langle u , g \rangle - \mathbb { E } _ { t } W ^ { x } \le 2 \eta a ^ { 2 } \beta ^ { x } .
$$

Put $Y _ { t } = \mathbb { E } _ { t } W _ { t } ^ { x } - W _ { t } ^ { x }$ . Then $\mathbb { E } _ { t } Y _ { t } = 0 , Y _ { t } \leq 1$ , and $\mathbb { E } _ { t } Y _ { t } ^ { 2 } \le a _ { t } ^ { 2 } \beta _ { t } ^ { x }$ . The parameter choice (5) ensures $\eta \le e / ( 1 0 2 4 d H ) < 1$ . Using $e ^ { z } \leq 1 + z + z ^ { 2 }$ for all $z \leq 1$ gives

$$
\mathbb { E } _ { t } \exp \{ \eta Y _ { t } - \eta ^ { 2 } a _ { t } ^ { 2 } \beta _ { t } ^ { x } \} \le 1 .
$$

Consequently $\begin{array} { r } { \exp ( \eta \sum _ { t = 1 } ^ { n } Y _ { t } - \eta ^ { 2 } \sum _ { t = 1 } ^ { n } a _ { t } ^ { 2 } \beta _ { t } ^ { x } ) } \end{array}$ is also a nonnegative supermartingale. Its maximal inequality contributes $\eta \sum _ { t } a _ { t } ^ { 2 } \beta _ { t } ^ { x } + L / \eta$ to the learner’s deviation. Adding the preceding conditional bias accounts for the coeficient three in (10).

We now combine these bounds into a single event that holds uniformly over all prefixes of the phase. Extend each exponential process above by keeping it constant after the phase ends. Since whether round t is executed is determined by $\mathcal { H } _ { t - 1 }$ , this extension preserves the one-step conditional inequalities. Each extended process is therefore a nonnegative supermartingale starting at one.

By Ville’s inequality, each process remains below $e ^ { L }$ at every prefix with probability at least $1 - e ^ { - L }$ . For each comparison coordinate, taking logarithms gives (9). For each learner, taking logarithms bounds the cumulative deviation $\scriptstyle \sum _ { t = 1 } ^ { n } Y _ { t } ;$ adding the conditional bias bound above then gives (10). A union bound over the 2d comparison processes and the two learner processes shows that all these inequalities hold simultaneously with probability at least $1 - ( 2 d + 2 ) e ^ { - L } = 1 - \rho$ □

## A.1.2 Regret with ratio correction

We next derive a deterministic bound for the exponential weights update with ratio correction. Recall $K _ { x , t } , K _ { y , t }$ and $\kappa _ { t }$ from Section 4.2, and define

$$
J _ { x , t } = \sum _ { i } u _ { t , i } r _ { t , i } ^ { 2 } , \qquad J _ { y , t } = \sum _ { j } v _ { t , j } s _ { t , j } ^ { 2 } .
$$

These quantities bound the quadratic cost of the correction, whereas $K _ { x , t }$ and $K _ { y , t }$ enter with a negative sign. The following lemma isolates this negative contribution in the regret bound and tracks the accompanying quadratic cost.

Lemma 6 (Regret with ratio correction). For every prefix $n \leq N$ and every row action i,

$$
\eta \sum _ { t = 1 } ^ { n } ( W _ { t } ^ { x } - a _ { t } \widehat { g } _ { t , i } ) \leq 2 \log \frac { 1 } { p _ { i } } + \log 5 - \sum _ { t = 1 } ^ { n } \kappa _ { t } K _ { x , t } + \frac { 3 } { 2 } \sum _ { t = 1 } ^ { n } \kappa _ { t } ^ { 2 } J _ { x , t } .\tag{12}
$$

The same inequality holds for the column player with $( W ^ { x } , \widehat { g } , p , K _ { x } , J _ { x } )$ replaced by $( W ^ { y } , \widehat { h } , q , K _ { y } , J _ { y } )$

The negative term in (12) is the contribution that compensates for the learner’s estimation error in Lemma 5. The adaptive weights keep the quadratic term small enough to retain this benefit, while the comparison cost depends only logarithmically on the baseline probability.

Proof. The common factor $1 + \kappa _ { t }$ in $x _ { t , i } / x _ { t + 1 , i }$ disappears after normalization in (4). Thus the row update is exponential weights with nonnegative costs

$$
b _ { t , i } = \zeta _ { t } \widehat { g } _ { t , i } + \log ( 1 + \kappa _ { t } r _ { t , i } ) , \qquad Z _ { t } = \sum _ { i } u _ { t , i } e ^ { - b _ { t , i } } .
$$

Using $e ^ { - b } \leq 1 - b + b ^ { 2 } / 2$ for $b \geq 0$ and $\log z \leq z - 1$ , we have

$$
\log Z _ { t } \leq - \langle u _ { t } , b _ { t } \rangle + \frac { 1 } { 2 } \langle u _ { t } , b _ { t } ^ { 2 } \rangle \leq - \eta W _ { t } ^ { x } - \kappa _ { t } K _ { x , t } + \frac { 3 } { 2 } \kappa _ { t } ^ { 2 } J _ { x , t } .
$$

The last inequality uses $( s + \log ( 1 + z ) ) ^ { 2 } / 2 \leq s ^ { 2 } + z ^ { 2 }$ and log $( 1 + z ) \ge z - z ^ { 2 } / 2$ for $s , z \geq 0$ . The exact normalized update now gives

$$
\eta ( W _ { t } ^ { x } - a _ { t } \widehat { g } _ { t , i } ) \leq \log \frac { u _ { t + 1 , i } } { u _ { t , i } } + \log ( 1 + \kappa _ { t } r _ { t , i } ) - \kappa _ { t } K _ { x , t } + \frac { 3 } { 2 } \kappa _ { t } ^ { 2 } J _ { x , t } .
$$

The second logarithm also telescopes, since

$$
1 + \kappa _ { t } r _ { t , i } = \frac { \tau _ { t + 1 } x _ { t + 1 , i } } { \tau _ { t } x _ { t , i } } , \qquad \sum _ { t = 1 } ^ { n } \log ( 1 + \kappa _ { t } r _ { t , i } ) = \log \frac { \tau _ { n + 1 } x _ { n + 1 , i } } { \tau _ { 0 } p _ { i } } .
$$

For $n \leq N$ , the stopping rule and $a _ { t } \leq 1$ imply $\tau _ { n + 1 } \leq 5 \tau _ { 0 }$ . Summing the previous inequality and bounding $u _ { n + 1 , i } , x _ { n + 1 , i } \leq 1$ proves (12). □

## A.1.3 Proof of the joint regret bound

Proof of Lemma 2. Work on the event of Lemma 5. Dropping the positive implicit exploration terms in the denominators of (8) gives $\beta _ { t } ^ { x } \le S _ { x , t } K _ { y , t }$ and $\beta _ { t } ^ { y } \le S _ { y , t } K _ { x , t }$ . Using $a _ { t } ( S _ { x , t } + S _ { y , t } ) \leq d .$ we obtain

$$
a _ { t } ^ { 2 } ( \beta _ { t } ^ { x } + \beta _ { t } ^ { y } ) \leq a _ { t } ^ { 2 } ( S _ { x , t } K _ { y , t } + S _ { y , t } K _ { x , t } ) \leq d a _ { t } ( K _ { x , t } + K _ { y , t } ) .\tag{13}
$$

Likewise, $a _ { t } r _ { t , i } \leq d$ and $a _ { t } s _ { t , j } \leq d$ imply

$$
\kappa _ { t } ^ { 2 } ( J _ { x , t } + J _ { y , t } ) \leq \frac { d } { \tau _ { 0 } } \kappa _ { t } ( K _ { x , t } + K _ { y , t } ) .
$$

Combining (9), (10), and (12) for both players, and applying (13), bounds the left side of (6) by

$$
\frac { 2 R _ { 0 } + 2 \log { 5 } + 4 L } { \eta } + \frac { 1 } { \eta } \sum _ { t = 1 } ^ { n } \kappa _ { t } ( K _ { x , t } + K _ { y , t } ) \left( 3 \eta ^ { 2 } d \tau _ { t } + \frac { 3 d } { 2 \tau _ { 0 } } - 1 \right) .
$$

Here the $4 L / \eta$ term comprises one comparison and one learner concentration cost for each player. For every executed round, $\tau _ { t } \leq 5 \tau _ { 0 }$ , and the two lower bounds on $\tau _ { 0 }$ in (5) ensure

$$
3 \eta ^ { 2 } d \tau _ { t } + \frac { 3 d } { 2 \tau _ { 0 } } \leq \frac { 3 8 4 0 } { 1 6 3 8 4 } + \frac { 1 } { 4 } = \frac { 3 1 } { 6 4 } < 1 .
$$

The sum is therefore nonpositive. Dropping it and using $H = R _ { 0 } .$ +log $5 + 2 L$ gives (6) simultaneously for every $1 \leq n \leq N$ and every $i , j \in [ d ]$ on the event of Lemma 5. This event has probability at least $1 - \rho ,$ , completing the proof. □

## A.2 Gap within a phase

Proof of Lemma 3. For $n \geq 1$ , let

$$
\overline { { u } } _ { n } = \frac { 1 } { A _ { n } } \sum _ { t = 1 } ^ { n } a _ { t } u _ { t } , \qquad \overline { { v } } _ { n } = \frac { 1 } { A _ { n } } \sum _ { t = 1 } ^ { n } a _ { t } v _ { t } .
$$

Since $g _ { t } = ( { \bf 1 } + A v _ { t } ) / 2$ and $h _ { t } = ( { \bf 1 } - A ^ { \top } u _ { t } ) / 2$ , the common term $u _ { t } ^ { \top } A v _ { t }$ cancels:

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { i , j } \sum _ { t = 1 } ^ { n } a _ { t } \big ( \langle u _ { t } , g _ { t } \rangle - g _ { t , i } + \langle v _ { t } , h _ { t } \rangle - h _ { t , j } \big ) } \\ { \displaystyle \quad = \frac { 1 } { 2 } \operatorname* { m a x } _ { i , j } \sum _ { t = 1 } ^ { n } a _ { t } \big ( u _ { t } ^ { \top } A e _ { j } - e _ { i } ^ { \top } A v _ { t } \big ) = \frac { A _ { n } } { 2 } \mathrm { G a p } \big ( \overline { { u } } _ { n } , \overline { { v } } _ { n } \big ) . } \end{array}
$$

This identity holds for every realized sequence of the predictable weights. The factor $1 / 2$ accounts for the shift of the payof to nonnegative losses. Lemma 2 consequently gives $A _ { n } \operatorname { G a p } ( { \overline { { u } } } _ { n } , { \overline { { v } } } _ { n } ) \leq 4 H / \eta$ On the other hand, (1) implies

$$
( x _ { n + 1 } , y _ { n + 1 } ) = \frac { \tau _ { 0 } ( p , q ) + A _ { n } ( \overline { { u } } _ { n } , \overline { { v } } _ { n } ) } { \tau _ { 0 } + A _ { n } } .
$$

Using joint convexity of the duality gap and the baseline assumption,

$$
\mathrm { G a p } ( x _ { n + 1 } , y _ { n + 1 } ) \le \frac { \tau _ { 0 } e + 4 H / \eta } { \tau _ { 0 } + A _ { n } } = \frac { 5 e \tau _ { 0 } } { 4 ( \tau _ { 0 } + A _ { n } ) } ,
$$

where the last equality uses (5). For $n = 0 , ( 7 )$ follows directly from ${ \mathrm { G a p } } ( p , q ) \leq e$ . At the endpoint $A _ { N } \geq 3 \tau _ { 0 }$ , which proves the contraction. □

## A.3 Phase duration

We prove the following explicit bound, which implies Lemma 4:

$$
N \leq 3 \tau _ { 0 } + 1 + \left( 4 \tau _ { 0 } + 1 \right) \left( 1 + \frac { d } { \tau _ { 0 } } \right) ( R _ { 0 } + 2 \log 5 ) .\tag{14}
$$

Proof of Lemma $\it 4 .$ For the row player, set $M _ { t , i } = \tau _ { t } x _ { t , i }$ . The averaging update (1) gives

$$
M _ { t + 1 , i } = M _ { t , i } + a _ { t } u _ { t , i } , \qquad z _ { t , i } : = \frac { M _ { t + 1 , i } - M _ { t , i } } { M _ { t , i } } = \kappa _ { t } r _ { t , i } \leq \frac { d } { \tau _ { 0 } } .
$$

The last inequality follows from (2) and $\tau _ { t } \geq \tau _ { 0 }$ . For $0 \leq z \leq d / \tau _ { 0 } , z \leq ( 1 + d / \tau _ { 0 } ) \log ( 1 + z )$ . Since $\tau _ { t } \leq \tau _ { n + 1 }$ , summing over rounds and coordinates yields

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { n } a _ { t } S _ { x , t } = \sum _ { t = 1 } ^ { n } \tau _ { t } \sum _ { i } z _ { t , i } } } \\ & { } & { \leq \tau _ { n + 1 } ( 1 + \frac { d } { \tau _ { 0 } } ) \sum _ { i } \log \frac { \tau _ { n + 1 } x _ { n + 1 , i } } { \tau _ { 0 } p _ { i } } . } \end{array}
$$

The logarithms telescope because $1 + z _ { t , i } = M _ { t + 1 , i } / M _ { t , i }$ . Each coordinate of $x _ { n + 1 }$ is at most one, so the final coordinate sum is at most $d [ \log ( \tau _ { n + 1 } / \tau _ { 0 } ) + \log ( 1 / \operatorname* { m i n } _ { i } p _ { i } ) ]$ . Applying the same argument to the column player and using the pointwise inequality

$$
1 \leq a _ { t } + \frac { a _ { t } } { d } ( S _ { x , t } + S _ { y , t } )
$$

gives the prefix bound

$$
n \leq A _ { n } + \tau _ { n + 1 } \left( 1 + \frac { d } { \tau _ { 0 } } \right) \left( R _ { 0 } + 2 \log \frac { \tau _ { n + 1 } } { \tau _ { 0 } } \right) .\tag{15}
$$

We first show that the phase ends after finitely many rounds. Otherwise, the stopping condition would never be met, so $A _ { n } < 3 \tau _ { 0 }$ and $\tau _ { n + 1 } < 4 \tau _ { 0 }$ for every n. The prefix bound (15) would then imply

$$
n \leq 3 \tau _ { 0 } + 4 \tau _ { 0 } \left( 1 + \frac { d } { \tau _ { 0 } } \right) \left( R _ { 0 } + 2 \log 4 \right) \qquad \mathrm { f o r ~ e v e r y ~ } n \geq 1 ,
$$

which is impossible because the right side is independent of $n .$

We can therefore apply (15) at the stopping round N. By definition, $A _ { N - 1 } < 3 \tau _ { 0 }$ , and $a _ { N } \leq 1$ gives $A _ { N } = A _ { N - 1 } + a _ { N } < 3 \tau _ { 0 } + 1$ . Thus $\tau _ { N + 1 } = \tau _ { 0 } + A _ { N } < 4 \tau _ { 0 } + 1 \leq 5 \tau _ { 0 }$ . Substituting these bounds into (15) with $n = N$ proves (14). Since $\tau _ { 0 } \geq 6 d ,$ , this gives $N = \mathcal { O } ( \tau _ { 0 } ( 1 + R _ { 0 } ) )$ . The parameter choice (5) further gives $\tau _ { 0 } = \mathcal { O } ( d H ^ { 2 } / e ^ { 2 } )$ , while $1 + R _ { 0 } = \mathcal { O } ( H )$ . Hence $N = \mathcal { O } ( d H ^ { 3 } / e ^ { 2 } )$ , proving part (i).

For part (ii), the update $M _ { t + 1 , i } = M _ { t , i } + a _ { t } u _ { t , i }$ shows that each $M _ { t , i }$ is nondecreasing. For every $0 \leq n \leq N$ , we therefore have $M _ { n + 1 , i } \geq \tau _ { 0 } p _ { i }$ and $\tau _ { n + 1 } \leq \tau _ { N + 1 } \leq 5 \tau _ { 0 }$ , so

$$
x _ { n + 1 , i } = \frac { M _ { n + 1 , i } } { \tau _ { n + 1 } } \geq \frac { \tau _ { 0 } p _ { i } } { 5 \tau _ { 0 } } = \frac { p _ { i } } { 5 } .
$$

Applying the same argument to $\tau _ { t } y _ { t , j }$ gives $y _ { n + 1 , j } \ge q _ { j } / 5$ for every $j \in [ d ]$ . These are the coordinate bounds in part (ii), completing the proof. □

## A.4 The global guarantee

Proof of Theorem 1. We restore the global round index t. Lemma 4 ensures that every phase ends. Conditional on any history at the start of a phase, Lemma 5 has failure probability at most $\rho _ { k }$ Taking expectations and summing $\rho _ { k } = \delta / 2 ^ { k + 1 }$ over phases gives an event E of probability at least $1 - \delta$ on which all phase concentration bounds hold. On this event, Lemma 2 applies in every phase. The initial baseline has gap at most $e _ { 0 } = 2$ , and Lemma 3 inductively supplies a baseline of gap at most $e _ { k }$ for phase k. Every pair played in that phase then has gap at most $5 e _ { k } / 4$

Let $N _ { k }$ denote the duration of phase k and attach a phase subscript to its parameters. The coordinate bounds in Lemma 4 hold on every history. Starting from the uniform baseline, they give $p _ { i } , q _ { j } \geq 1 / ( d 5 ^ { k } )$ in phase k, and hence

$$
R _ { 0 , k } \leq 2 \log d + 2 k \log 5 , \qquad L _ { k } = \log \frac { ( 2 d + 2 ) 2 ^ { k + 1 } } { \delta } .
$$

With $B _ { k } = \log ( d / \delta ) + k + 1$ , the parameter choices in (5) therefore imply

$$
H _ { k } = \mathcal { O } ( B _ { k } ) , \qquad \tau _ { 0 , k } = \mathcal { O } \left( \frac { d B _ { k } ^ { 2 } } { e _ { k } ^ { 2 } } \right) .
$$

Lemma 4 now gives a universal constant $C _ { 1 }$ such that

$$
N _ { k } \leq C _ { 1 } \frac { d B _ { k } ^ { 3 } } { e _ { k } ^ { 2 } } .\tag{16}
$$

If global round t lies in phase k, then $t \leq \Sigma _ { j = 0 } ^ { k } N _ { j }$ . Since $e _ { j } ^ { - 2 } = 4 ^ { j - k } e _ { k } ^ { - 2 }$ and $B _ { j } \leq B _ { k }$ , summing (16) yields

$$
t \leq \frac { 4 C _ { 1 } d B _ { k } ^ { 3 } } { 3 e _ { k } ^ { 2 } } .\tag{17}
$$

To bound $B _ { k }$ in terms of t, note that $a _ { t } \leq 1$ implies $N _ { j } \geq 3 \tau _ { 0 , j } \geq 4 ^ { j }$ . The last inequality follows from (5), $H _ { j } \geq 1$ , and $e _ { i } ^ { - 2 } = 4 ^ { j - 1 }$ . For $k \geq 1$ , phase $k - 1$ is complete before round t, so $t \geq N _ { k - 1 } \geq 4 ^ { k - 1 }$ Thus $k \leq 1 + \mathrm { l o g } _ { 4 } \dot { t }$ for every phase containing t, including $k = 0$ , and $B _ { k } = \mathcal { O } ( \log ( d t / \delta ) )$ .

On $\mathcal { E } ,$ combining the gap bound $5 e _ { k } / 4$ with (17) gives

$$
{ \mathrm { G a p } } ( x _ { t } , y _ { t } ) \leq { \frac { 5 } { 4 } } { \sqrt { \frac { 4 C _ { 1 } d B _ { k } ^ { 3 } } { 3 t } } } \leq C { \sqrt { \frac { d } { t } } } \left[ \log \left( { \frac { d t } { \delta } } \right) \right] ^ { 3 / 2 }
$$

for a universal constant C. This holds for every $t \geq 1$ on $\mathcal { E } ,$ , proving the theorem.

## B Dimension dependence of the log-barrier bound

We derive the dimension dependence reported for Fiegel et al. (2026b) in Table 1. Their Theorem 5.2 bounds the duality gap by 2Kτ log $( ( t + T _ { 0 } ) / \delta ) ( t + T _ { 0 } ) ^ { - 1 / 4 }$ , where K is the total number of actions of both players. Their Assumption 5.1 leaves constants depending on K implicit. We give an explicit parameter choice below.

For our setting, $K = 2 d .$ To distinguish their parameters from ours, write $\eta _ { \mathrm { F } } , \tau _ { \mathrm { F } } , T _ { \mathrm { F } }$ for their $\eta , \tau , T _ { 0 }$ . Set $\delta _ { * } = \operatorname* { m i n } \{ \delta , \exp ( - 2 ) \}$ and let M be a suficiently large universal constant. Define

$$
\Lambda = \log \left( \frac { M K } { \delta _ { * } } \right) , \qquad \eta _ { \mathrm { { F } } } = \frac { 1 } { M \Lambda ^ { 2 } } , \qquad \tau _ { \mathrm { { F } } } = M ^ { 2 } K \Lambda ^ { 2 } , \qquad T _ { \mathrm { { F } } } = \left\lceil \tau _ { \mathrm { { F } } } ^ { 4 } \right\rceil .
$$

Their learning rate and regularization schedules are then $\eta _ { \mathrm { F } } ( t + T _ { \mathrm { F } } ) ^ { - 3 / 4 }$ and $\tau _ { \mathrm { F } } \log ( ( t + T _ { \mathrm { F } } ) / \delta _ { * } ) ( t +$ $T _ { \mathrm { F } } ) ^ { - 1 / 4 }$ , respectively. In particular, $\eta _ { \mathrm { F } } \tau _ { \mathrm { F } } = M K$ , and

$$
\eta _ { \mathrm { F } } \tau _ { \mathrm { F } } \leq \frac { T _ { \mathrm { F } } } { ( \log T _ { \mathrm { F } } ) ^ { 4 } } , \qquad T _ { \mathrm { F } } \leq [ \log ( 1 / \delta _ { * } ) ] ^ { 2 } \tau _ { \mathrm { F } } ^ { 4 } ,
$$

as required by their Assumption 5.1.

It remains to check the dimension factors hidden in the smallness conditions of their proof. Let $\theta _ { 0 }$ denote their initial regularization strength:

$$
\begin{array} { r } { \theta _ { 0 } = \tau _ { \mathrm { F } } \log ( T _ { \mathrm { F } } / \delta _ { * } ) T _ { \mathrm { F } } ^ { - 1 / 4 } . } \end{array}
$$

Since log $( T _ { \mathrm { F } } / \delta _ { * } ) = \mathcal { O } ( \Lambda )$ , we have $\theta _ { 0 } = \mathcal { O } ( \Lambda )$ . Their Lemma 6.2, Lemma B.1, and Proposition C.5 use the quantities

$$
{ \cal L } _ { \mathrm { F } } = \sqrt { K } , \qquad \sigma _ { \mathrm { F } } = 2 + \theta _ { 0 } \sqrt { K } , \qquad \rho _ { \mathrm { F } } = \sigma _ { \mathrm { F } } \eta _ { \mathrm { F } } ( 4 L _ { \mathrm { F } } + 2 \theta _ { 0 } ) + { \frac { \theta _ { 0 } \sqrt { K } } { 4 } } .
$$

The proposed parameters give

$$
\sigma _ { \mathrm { F } } = \mathcal { O } ( \sqrt { K } \Lambda ) , \qquad \rho _ { \mathrm { F } } = \mathcal { O } \biggl ( \frac { K } { M \Lambda } + \sqrt { K } \Lambda \biggr ) ,
$$

with universal constants independent of $M , K , \delta .$

The term caused by the changing regularization requires $K / ( \eta _ { \mathrm { F } } \tau _ { \mathrm { F } } )$ to be suficiently small. Here this ratio equals $1 / M$ . The remaining drift and fluctuation coeficients are controlled by

$$
\mathrm { m a x } \left\{ \frac { \eta _ { \mathrm { F } } ^ { 2 } L _ { \mathrm { F } } \sigma _ { \mathrm { F } } ^ { 3 } } { \tau _ { \mathrm { F } } ^ { 2 } } , \frac { \eta _ { \mathrm { F } } ^ { 2 } \sigma _ { \mathrm { F } } ^ { 4 } } { \tau _ { \mathrm { F } } ^ { 2 } } , \frac { \rho _ { \mathrm { F } } \eta _ { \mathrm { F } } \sigma _ { \mathrm { F } } ^ { 2 } } { \tau _ { \mathrm { F } } ^ { 2 } } , \frac { \rho _ { \mathrm { F } } ^ { 2 } } { \tau _ { \mathrm { F } } ^ { 2 } } , \frac { \eta _ { \mathrm { F } } \sigma _ { \mathrm { F } } ^ { 2 } } { \tau _ { \mathrm { F } } } , \frac { \rho _ { \mathrm { F } } } { \tau _ { \mathrm { F } } } \right\} = \mathcal { O } ( M ^ { - 2 } ) .
$$

Denote these six coeficients, in the displayed order, by $c _ { 1 } , \ldots , c _ { 6 }$ . For example, for $M \geq 1 0 2 4$ $\theta _ { 0 } \leq 1 3 \Lambda , \sigma _ { \mathrm { F } } \leq 1 4 \sqrt { K } \Lambda$ , and $\rho _ { \mathrm { F } } / \tau _ { \mathrm { F } } \leq 5 / M ^ { 2 }$ , which give max $_ { i } c _ { i } \leq 5 / M ^ { 2 }$ . The same choice ensures

$$
\sigma _ { \mathrm { { F } } } \eta _ { \mathrm { { F } } } T _ { \mathrm { { F } } } ^ { - 3 / 4 } \leq { \frac { 1 4 } { M ^ { 7 } K ^ { 5 / 2 } \Lambda ^ { 7 } } } \leq { \frac { 1 } { 3 2 } } .
$$

This verifies their Assumption C.1 and the weaker condition $\sigma _ { \mathrm { F } } \eta _ { \mathrm { F } } T _ { \mathrm { F } } ^ { - 3 / 4 } \leq 1 / \sqrt { 1 2 }$ of their Lemma B.3.

We next make the residual recursion explicit to check that its first-order terms are absorbed as well. Write $w _ { t }$ for their combined iterate, $p _ { t }$ for the regularized residual defined in their Appendix $\mathrm { B } { } _ { ; }$ and $D _ { t } = \| p _ { t } \| _ { * , w _ { t } } ^ { 2 }$ for its squared dual local norm. For $t \geq 0 .$ , set

$$
\begin{array} { r } { s = t + T _ { \mathrm { F } } , \qquad h _ { t } = \log ( s / \delta _ { * } ) , \qquad a _ { t } = \eta _ { \mathrm { F } } s ^ { - 3 / 4 } , \qquad \theta _ { t } = \tau _ { \mathrm { F } } h _ { t } s ^ { - 1 / 4 } . } \end{array}
$$

Let $v _ { t } = F _ { \theta _ { t + 1 } } ( w _ { t + 1 } ) - F _ { \theta _ { t } } ( w _ { t } )$ , where $F _ { \theta }$ is their regularized operator, and define

$$
Z _ { t } = \lVert p _ { t } \rVert _ { * , w _ { t + 1 } } ^ { 2 } - D _ { t } + 2 \langle v _ { t } , p _ { t } \rangle _ { * , w _ { t } } .
$$

The squared-norm expansion in their Appendix D, with the full factor 2 in its cross term, gives

$$
D _ { t + 1 } - D _ { t } \leq Z _ { t } + 3 2 \rho _ { \mathrm { F } } \sigma _ { \mathrm { F } } ^ { 2 } a _ { t } s ^ { - 3 / 4 } + \rho _ { \mathrm { F } } ^ { 2 } s ^ { - 3 / 2 } .
$$

Here we used their Proposition C.5 and the pathwise norm-variation bound of Proposition C.3. For the conditional mean of the norm variation, their Proposition C.2 and Lemma C.4 give the bound

$$
\mathbb { E } _ { t } [ \| p _ { t } \| _ { * , w _ { t + 1 } } ^ { 2 } - D _ { t } ] \leq ( 2 a _ { t } \sqrt { D _ { t } } + 6 8 \sigma _ { \mathrm { F } } ^ { 2 } a _ { t } ^ { 2 } ) D _ { t } \leq 2 a _ { t } D _ { t } ^ { 3 / 2 } + 6 8 \sigma _ { \mathrm { F } } ^ { 4 } a _ { t } ^ { 2 } ,
$$

where $\mathbb { E } _ { t }$ conditions on the history before this update and $D _ { t } \leq \sigma _ { \mathrm { F } } ^ { 2 }$ . Combining this with their Lemma D.3 and Proposition C.5 yields

$$
\begin{array} { c } { { \mathbb { E } _ { t } Z _ { t } \le - 2 a _ { t } \theta _ { t + 1 } D _ { t } + 2 a _ { t } D _ { t } ^ { 3 / 2 } + \displaystyle \frac { \theta _ { t } } { 2 s } \sqrt { K D _ { t } } + ( 1 2 8 L _ { \mathrm { F } } \sigma _ { \mathrm { F } } ^ { 3 } + 6 8 \sigma _ { \mathrm { F } } ^ { 4 } ) a _ { t } ^ { 2 } , } } \\ { { | Z _ { t } - \mathbb { E } _ { t } Z _ { t } | \le ( 1 6 \eta _ { \mathrm { F } } \sigma _ { \mathrm { F } } ^ { 2 } + 8 \rho _ { \mathrm { F } } ) s ^ { - 3 / 4 } \sqrt { D _ { t } } . } } \end{array}
$$

To leave enough contraction to absorb the $2 a _ { t } D _ { t } ^ { 3 / 2 }$ term, use the normalization

$$
U _ { t } = \frac { 1 6 D _ { t } } { \tau _ { \mathrm { F } } ^ { 2 } h _ { t } } .
$$

The stopping boundary $U _ { t } \leq h _ { t } / \sqrt { s }$ in their Lemma 6.5 now corresponds to $D _ { t } \leq \theta _ { t } ^ { 2 } / 1 6$ . Before this boundary is crossed,

$$
2 a _ { t } D _ { t } ^ { 3 / 2 } \leq \frac { 1 } { 2 } a _ { t } \theta _ { t } D _ { t } , \qquad \frac { \theta _ { t } } { 2 s } \sqrt { K D _ { t } } \leq \frac { 1 } { 4 } a _ { t } \theta _ { t } D _ { t } + \frac { K \theta _ { t } } { 4 a _ { t } s ^ { 2 } } .
$$

Their Lemma B.2 gives $\theta _ { t + 1 } \geq ( 1 - 1 / ( 4 s ) ) \theta _ { t }$ . Since $s \geq 2$ , the first-order terms above sum to at most

$$
\left( - 2 + \frac { 1 } { 2 s } + \frac { 1 } { 2 } + \frac { 1 } { 4 } \right) a _ { t } \theta _ { t } D _ { t } \leq - a _ { t } \theta _ { t } D _ { t } \leq - \frac { D _ { t } } { s } ,
$$

where the last inequality uses $\eta _ { \mathrm { F } } \tau _ { \mathrm { F } } h _ { t } \geq 1$ . As $h _ { t + 1 } \geq h _ { t }$ and $D _ { t + 1 } \geq 0$ , the normalized recursion is

$$
U _ { t + 1 } \leq ( 1 - 1 / s ) U _ { t } + b _ { t + 1 } + W _ { t + 1 } , \qquad W _ { t + 1 } = \frac { 1 6 ( Z _ { t } - \mathbb { E } _ { t } Z _ { t } ) } { \tau _ { \mathrm { F } } ^ { 2 } h _ { t } } ,
$$

with $\mathbb { E } _ { t } W _ { t + 1 } = 0$ and remainder bounds

$$
\begin{array} { c } { { b _ { t + 1 } \leq \left[ \displaystyle \frac { 4 } { M } + 1 6 ( 1 2 8 c _ { 1 } + 6 8 c _ { 2 } + 3 2 c _ { 3 } + c _ { 4 } ) \right] s ^ { - 3 / 2 } \leq ( s + 1 ) ^ { - 3 / 2 } , } } \\ { { \mid W _ { t + 1 } \mid \leq ( 6 4 c _ { 5 } + 3 2 c _ { 6 } ) s ^ { - 3 / 4 } \sqrt { U _ { t } } , \qquad W _ { t + 1 } ^ { 2 } \leq \textstyle \frac { 1 } { 2 } ( s + 1 ) ^ { - 3 / 2 } U _ { t } . } } \end{array}
$$

These inequalities hold for $M \geq 1 0 2 4$ by max<sub>i</sub> $c _ { i } \ \leq \ 5 / M ^ { 2 }$ . At the uniform initialization, the log-barrier gradient lies in the normal cone, so $D _ { 0 } \leq 4$ and

$$
U _ { 0 } \leq \frac { 6 4 } { \tau _ { \mathrm { F } } ^ { 2 } \log ( T _ { \mathrm { F } } / \delta _ { * } ) } \leq \frac { \log T _ { \mathrm { F } } } { \sqrt { T _ { \mathrm { F } } } } .
$$

Their Lemma 6.5 applies to this recursive upper bound as well: its exponential-supermartingale proof uses only the monotonicity of the exponential at that step. Thus, with probability at least $1 - \delta _ { * }$ , simultaneously for all $t \geq 0 , D _ { t } \leq \theta _ { t } ^ { 2 } / 1 6$ . Their Lemma 6.4 then gives duality gap at most $2 K \theta _ { t }$ in their [0, 1] loss normalization.

For our payof range $[ - 1 , 1 ] .$ , run their algorithm on the losses $( 1 + R _ { t } ) / 2$ and let $( x _ { t } , y _ { t } )$ denote its strategies in this appendix. The duality gap in our normalization is twice the gap for these losses. Consequently, with probability at least $1 - \delta$ , simultaneously for every $t \geq 1$ 2

$$
\mathrm { G a p } ( x _ { t } , y _ { t } ) \le 4 K \tau _ { \mathrm { F } } \log \biggl ( \frac { t + T _ { \mathrm { F } } } { \delta _ { * } } \biggr ) ( t + T _ { \mathrm { F } } ) ^ { - 1 / 4 } = \widetilde { \mathcal { O } } ( d ^ { 2 } t ^ { - 1 / 4 } ) .
$$

Indeed, $\tau _ { \mathrm { F } } = \tilde { \mathcal { O } } ( K )$ and $T _ { \mathrm { F } } = \widetilde { \mathcal { O } } ( K ^ { 4 } )$ , so the logarithm introduces no additional polynomial dependence on K. Solving this bound for a target $\mathrm { g a p } \ \varepsilon$ gives the round bound $\widetilde { \mathcal { O } } ( d ^ { 8 } / \varepsilon ^ { \bar { 4 } } )$ stated in Table 1.