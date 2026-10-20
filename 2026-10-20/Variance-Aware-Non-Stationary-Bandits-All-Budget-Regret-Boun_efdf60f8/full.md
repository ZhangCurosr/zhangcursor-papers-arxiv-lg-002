# Variance-Aware Non-Stationary Bandits: All-Budget Regret Bounds and Adaptation Costs

Xiaojin Zhang<sup>∗</sup>

## Abstract

Low reward variance reduces estimation cost, but does not eliminate monitoring of arms that may improve while unobserved. We refine the variance–variation tradeof and characterize how both costs depend on supplied variance and variation budgets in bounded, non-stationary multi-armed bandits. Let A be the number of arms, T the horizon, B a bound on consecutive mean variation excluding the initial vector, and Q a bound on the sum of the largest per-round reward variances. For every admissible pair $Q , B ,$ , the minimax expected dynamic regret is, within a single logarithmic factor,

$$
\operatorname* { m i n } \{ T , \ A + { \sqrt { A Q } } + ( A Q B T ) ^ { 1 / 3 } + { \sqrt { A T } } \operatorname* { m i n } \{ B , { \sqrt { B } } \} \} .
$$

The deterministic boundary $Q = 0$ is characterized without logarithms. For small positive budgets, a safe calibration phase identifies a common interval containing all future means of surviving arms. Clipping observations to a wider interval reduces the cost of restarting a prior hybrid static-regret algorithm; an original-mean analytical reference controls its residuals without charging the variation of clipped means. Balancing clipping bias and monitoring closes the small-budget noisy regime. Fixed oblivious constructions give the matching lower bound. We also quantify adaptation costs: on a disclosed deterministic one-improvement family, a stationaryregret budget has a matching worst-case change-regret frontier, including a finite improvement pulse. This implies an $\Omega ( ( \bar { T } / A ) ^ { 1 / 4 } )$ loss when one policy competes with all supplied-budget deterministic oracle rates. The budget-free low-noise scheduling problem remains separate.

## 1 Introduction

Non-stationary bandit learning combines two distinct problems. Samples estimate an arm’s current reward, while continued monitoring discovers changes in arms that have not been sampled recently. Reward variance governs the first problem. It does not, by itself, govern the second. Even with deterministic rewards, an initially inferior arm can become optimal without changing the reward of the incumbent.

We ask how reward-noise variance and cumulative mean variation jointly determine minimax dificulty, including vanishing-budget regimes. We distinguish learning with supplied budgets from adapting across classes whose budgets are unknown, and retain explicit dependence on the number of arms and the horizon. Variance-sensitive dynamic regret and noise-free monitoring already appear in [WHL16]. Our main refinement concerns vanishing mean variation with positive observation variance; the ordinary restart bound is a consequence of [Ito21b].

The noisy minimax landscape. For supplied envelope variance $Q$ and variation $B ,$ define

$$
\Xi ( A , T ; Q , B ) = \operatorname * { m i n } \{ T , A + \sqrt { A Q } + ( A Q B T ) ^ { 1 / 3 } + \sqrt { A T } \operatorname * { m i n } \{ B , \sqrt { B } \} \} .\tag{1}
$$

Theorem 4.1 gives matching lower and upper bounds, within one factor log(AT), for every admissible pair of budgets. The costs are initialization, stationary estimation, statistical drift, and deterministic monitoring. For $Q = 0$ , Theorem 3.1 removes all logarithms. The lower constructions are fixed before interaction and retain the arm dependence and budget endpoints.

The small-budget noisy upper bound needs more than ordinary restarting. A variance-sensitive calibration phase costs only $O ( A \ell + { \sqrt { A Q \ell } } )$ , where $\ell = O ( \log ( A T ) )$ , even when it takes many rounds. It safely removes arms that cannot become optimal and certifies an interval containing the future means of all survivors. Clipping rewards to a wider interval of width $O ( w )$ then replaces the restart cost A by $O ( w A )$ , while contributing bias at most $O ( Q / w )$ . We balance $\sqrt { A w B T }$ and $Q / w$ by taking $w = \mathrm { m a x } \{ B , ( Q ^ { 2 } / ( A B T ) ) ^ { 1 / 3 } \}$ when this is below one. The post-calibration learner is the existing hybrid algorithm of [Ito21b]. Proposition 6.2 records what plain restarting already implies; Section 4.1 provides the calibration and clipping argument needed for the sharper small-budget rate. No new static base algorithm is claimed.

The deterministic boundary. Let $\Delta$ be the sum of consecutive mean changes, excluding the initial mean vector. If rewards are deterministic and a budget $B \geq \Delta$ is supplied, Theorem 3.1 gives the minimax rate

$$
\Psi ( A , T , B ) = \operatorname* { m i n } \{ T , \ A + \sqrt { A T } \operatorname* { m i n } \{ B , \sqrt { B } \} \} .\tag{2}
$$

For $B \leq 1$ , this is min $\{ T , A + B { \sqrt { A T } } \}$ ; for $B \geq 1$ , the drift contribution has the familiar $\sqrt { A B T }$ scale. The upper bound exploits information supplied by the small budget. After one initial observation per arm, an arm far below the largest observed reward cannot become optimal during the remaining horizon. All arms left by this safe filter have gap $O ( B )$ , so subsequent monitoring costs $O ( B )$ per probe rather than 1.

The price of adapting. An algorithm that is not promised stationarity cannot simply commit after initial observations. On a family with one hidden improvement, Theorem 5.2 identifies a matching frontier within this family between a stationary expected-regret budget r and worstcase one-change regret. Both sides of the frontier are proved. The upper-bound policy spends its monitoring budget through independent probes and commits after observing the uniquely recognizable improvement. The lower bound couples each alternative with a stationary trajectory until its first informative observation, and sums the resulting detection delays. More precisely, for a disclosed gap $g , n = A - 1$ and $T \geq 1 6 A$ , the optimal change regret under stationary budget r is

$$
\Theta \left( \operatorname* { m i n } \left\{ g T , \ g { \sqrt { n T } } + { \frac { g ^ { 2 } n T } { r + g n } } \right\} \right) .\tag{3}
$$

The same frontier covers a single finite improvement pulse. Corollary 5.5 further gives an $\Omega ( ( T / A ) ^ { 1 / 4 } )$ worst-case ratio to the supplied-budget oracle. These statements concern explicit classes and side information.

Proof roadmap. Section 3 gives a safe filter and a logarithm-free deterministic policy. Section 4 proves the noisy upper bound by calibration and clipped restarts; its last-pull accounting bounds calibration regret without a harmonic sum. Section 5 proves the adaptation frontier. The full oblivious lower constructions and an independent high-probability empirical-variance analysis appear in the appendices.

Relation to prior work. Variance-sensitive dynamic regret is already studied in [WHL16, WXC<sup>+</sup>25]; inverse exploration–detection tradeofs are classical [GM11, GM08]. The textbook treatment in [LS20], Theorem 31.2, and the safe-environment results of [Suk25] make clear that logarithmic stationary regret cannot be extended uniformly by adding a detector alone. Prior-free tracking, significant shifts and path-length bounds are essential comparisons [AGO19, CLLW19, WL21, SK22, AYGL23, WL18, BLLW19]. Section 7 compares their actual model and theorem conditions, rather than only displayed rates. In particular, the small-budget upper rate can also be derived by filtering and rescaling an earlier deterministic bound, up to logarithmic factors. The additional small-budget noisy argument controls calibration regret and clipping bias; the ordinary restart guarantee serves as its baseline.

Scope of the claims. Theorem 4.1 characterizes the expected minimax value for every supplied pair $Q , B$ . Its deterministic boundary has no logarithmic gap, while Theorem 5.2 characterizes adaptation on the disclosed one-improvement family. The general unknown-budget low-noise schedule remains open; it is not a consequence of either characterization. The separate empiricalvariance appendix provides a high-probability ordinary restart bound, rather than a high-probability version of the sharp all-budget formula. Section 7 attributes the prior algorithmic and lower-bound ingredients. Full oblivious lower-bound proofs are in Appendices A and B.

## 2 Model and quantities

There are $A \geq 2$ arms and $T \geq 2$ rounds. Before interaction, an oblivious environment fixes reward laws $\nu _ { t , a }$ on [0, 1], with means $\mu _ { t , a }$ and variances $\sigma _ { t , a } ^ { 2 }$ . Reward draws are independent across rounds. Within-round dependence across arms is irrelevant because the learner observes only its selected reward. The policy chooses $a _ { t }$ using past observations and private randomness, and observes $X _ { t } \sim \nu _ { t , a _ { t } }$ . In particular,

$$
\mathbb { E } [ X _ { t } \mid { \mathcal F } _ { t - 1 } , a _ { t } ] = \mu _ { t , a _ { t } } , \qquad \mathbb { E } [ ( X _ { t } - \mu _ { t , a _ { t } } ) ^ { 2 } \mid { \mathcal F } _ { t - 1 } , a _ { t } ] = \sigma _ { t , a _ { t } } ^ { 2 } .
$$

For martingale arguments, reveal the current action randomization before the current reward, so that the action and its conditional variance are predictable in the refined filtration. When conditioning at calibration or block boundaries, the history contains only randomness already used; future algorithm

seeds remain independent. No assumption of identical distributions is made. The results below are stated for this model; an extension to adaptive environments requires its own precise filtration and budget convention.

Pseudo-regret and realized regret. Let $\mu _ { t } ^ { \star } = \operatorname* { m a x } _ { a } \mu _ { t , a }$ . Define

$$
R _ { T } = \sum _ { t = 1 } ^ { T } ( \mu _ { t } ^ { \star } - \mu _ { t , a _ { t } } ) ,\tag{4}
$$

$$
\widetilde { R } _ { T } = \sum _ { t = 1 } ^ { T } ( \mu _ { t } ^ { \star } - X _ { t } ) = R _ { T } + \sum _ { t = 1 } ^ { T } ( \mu _ { t , a _ { t } } - X _ { t } ) .\tag{5}
$$

The first quantity is nonnegative and at most T. The second also contains reward noise and can be negative. Our deterministic upper bounds are pathwise, the frontier uses expected pseudo-regret, and the noisy base bounds hold with high probability. We state the additional martingale term for realized regret explicitly.

Variation and stationary segments. Write

$$
d _ { t } = \| \mu _ { t } - \mu _ { t - 1 } \| _ { \infty } \quad ( t \geq 2 ) , \qquad \Delta = \sum _ { t = 2 } ^ { T } d _ { t } , \qquad L = 1 + \sum _ { t = 2 } ^ { T } \mathbf { 1 } \{ d _ { t } > 0 \} .
$$

Thus L counts segments, and $L - 1$ counts mean changes. For an interval $I = [ s , e ]$ , let $\textstyle \Delta _ { I } = \sum _ { t = s + 1 } ^ { e } d _ { t }$ and $h _ { I } = e - s + 1$ . For a partition into contiguous intervals,

$$
\sum _ { I \in \mathcal { P } } \Delta _ { I } \leq \Delta .\tag{6}
$$

The inequality can be strict: changes at boundaries are excluded from the internal variation of every interval.

Three diferent variance quantities. We distinguish

$$
\Lambda _ { T } = \sum _ { t = 1 } ^ { T } \sigma _ { t , a _ { t } } ^ { 2 } , \qquad V _ { T } = \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { a } \sigma _ { t , a } ^ { 2 } , \qquad W _ { T } = \sum _ { t = 1 } ^ { T } \sum _ { a = 1 } ^ { A } \sigma _ { t , a } ^ { 2 } .\tag{7}
$$

Here $\Lambda _ { T }$ depends on the policy’s trajectory; $V _ { T }$ and $W _ { T }$ are deterministic properties of the fixed environment. They satisfy

$$
0 \le \Lambda _ { T } \le V _ { T } \le W _ { T } \le A V _ { T } , \qquad V _ { T } \le T / 4 .
$$

The last bound uses the variance bound for [0, 1]-valued rewards. These quantities are not interchangeable when tuning an algorithm or comparing arm dependence. In particular, one cannot choose a restart period using the eventual $\Lambda _ { T }$ of the as-yet unrun policy.

Throughout, $\delta \in ( 0 , 1 )$ and

$$
\ell = \log \frac { 6 4 A T ^ { 4 } } { \delta } , \qquad D _ { T } = \ell ( 1 + \log T ) .\tag{8}
$$

The notation $\tilde { O }$ hides only logarithmic factors in $A , T , 1 / \delta$ . It never hides powers of A.

## 3 The deterministic minimax boundary

Let ${ \mathcal { D } } ( A , T , B )$ contain all deterministic reward arrays in $[ 0 , 1 ] ^ { T \times A }$ with $\Delta \le B$ . The learner knows A, T, B and the promise of determinism, but not the initial reward vector. Define

$$
\Re _ { \mathrm { d e t } } ( A , T , B ) = \operatorname* { i n f } _ { \pi } \operatorname* { s u p } _ { \mu \in { \mathcal { D } } ( A , T , B ) } \mathbb { E } _ { \pi } [ R _ { T } ] .
$$

The infimum allows randomized, nonanticipating policies that may depend on the supplied budget.   
The adversary is oblivious.

Theorem 3.1 (All-budget deterministic minimax rate). For every $A , T \geq 2$ and $B \in [ 0 , T - 1 ]$ ， universal constants $c , C > 0$ satisfy

$$
c \Psi ( A , T , B ) \leq \Re _ { \operatorname* { d e t } } ( A , T , B ) \leq C \Psi ( A , T , B ) .\tag{9}
$$

The upper bound is attained by a deterministic policy and holds pathwise. In particular, the zerovariance minimax drift term is $B \sqrt { A T }$ for $B \leq 1$ and ABT for $B \geq 1$ , with initial exploration and the horizon cap retained.

## 3.1 Safe filtering and the upper bound

Assume $A < T$ ; otherwise the bound $R _ { T } \leq T$ sufices. Pull every arm once in the first A rounds, and denote its observed reward by $y _ { a }$ . Let $y _ { \mathrm { m a x } } = \operatorname* { m a x } _ { a } y _ { a }$ and retain

$$
\begin{array} { r } { \mathcal { C } = \{ a : y _ { \operatorname* { m a x } } - y _ { a } \leq 2 B \} , \qquad \kappa = \operatorname* { m i n } \{ 1 , 4 B \} . } \end{array}\tag{10}
$$

Only arms in C are subsequently monitored.

Lemma 3.2 (The filter is safe). Every arm optimal at a round $t > A$ belongs to $\mathcal { C } .$ . Every $a \in { \mathcal { C } }$ has future instantaneous regret at most κ.

Proof. Since each arm’s mean changes by at most B over the entire horizon, $| y _ { a } - \mu _ { t , a } | \leq B$ . If a is optimal at t and b attained $y _ { \mathrm { m a x } }$ during initialization, then y<sub>max</sub> $- y _ { a } \le \mu _ { t , b } - \mu _ { t , a } + 2 B \le 2 B$ . For any retained arm,

$$
\mu _ { t } ^ { \star } - \mu _ { t , a } \leq \left( y _ { \operatorname* { m a x } } + B \right) - \left( y _ { a } - B \right) \leq 4 B .
$$

The reward range also bounds this regret by 1.

Proposition 3.3 (Filtered monitoring bound). For any $H \in \{ A , \ldots , T \}$ , this policy satisfies

$$
R _ { T } \leq A + \kappa A \left( \frac { T } { H } + 1 \right) + 2 H B .\tag{11}
$$

Its choice of H in Algorithm 1, together with the trivial horizon bound, proves the upper bound in Theorem 3.1.

Proof. Initialization costs at most A. Each subsequent probe has regret at most κ by Lemma 3.2. On an interval I with enough time to probe every retained arm, let $z _ { a }$ be its fresh probe reward and let $\widehat { a }$ maximize $z _ { a }$ . Since every current optimal arm is retained, for any later round in I,

$$
\mu _ { t } ^ { \star } - \mu _ { t , \widehat { a } } \leq \mu _ { t } ^ { \star } - z _ { a _ { t } ^ { \star } } + z _ { \widehat { a } } - \mu _ { t , \widehat { a } } \leq 2 \Delta _ { I } .
$$

Algorithm 1 Filtered probe-and-hold (deterministic rewards, known budget)   
Input: ${ \overline { { A , T } } } ,$ , deterministic rewards, supplied variation budget B   
1. if $A \geq T$ then   
2. Pull arbitrary arms for T rounds and stop.   
3. end if   
4. In the first A rounds, pull every arm once and store $y _ { a } .$   
5. Compute C and κ from (10).   
6. if $B = 0$ then   
7. Set $H = T .$   
8. else   
9. Set $H = \operatorname* { m i n } \{ T _ { \mathrm { : } }$ , max $\{ A , \lceil \sqrt { \kappa A T / ( 2 B ) } \rceil \} \}$   
10. end if   
11. for each remaining consecutive interval of length at most H do   
12. Pull each arm in C once, or until the interval ends.   
13. In remaining rounds, pull an arm with largest reward in this interval’s probe observations.   
14. end for

If the last interval is too short to finish probing, every round is a probe and is already charged.   
Summing, using $\begin{array} { r } { \sum _ { I } \Delta _ { I } \le B } \end{array}$ , proves (11).

For $B = 0 ;$ , all retained arms have zero future gap and $R _ { T } \ \leq \ A$ . For $B > 0$ , write $h _ { \star } =$ $\sqrt { \kappa A T / ( 2 B ) }$ . If $A \ \leq \ h _ { \star } \ \leq \ T$ , then $h _ { \star } \leq H \leq h _ { \star } + 1 ; ( 1 1 )$ is bounded by a constant times $A + \sqrt { \kappa A B T } + B$ . Here $B \le \sqrt { \kappa A B T }$ , and

$$
{ \sqrt { \kappa A B T } } \leq 2 { \sqrt { A T } } \operatorname* { m i n } \{ B , { \sqrt { B } } \} .
$$

If $h _ { \star } > T .$ , setting $H = T$ gives $2 B T < \kappa A$ and (11) is $O ( A )$ . For clarity, $\kappa = 4 B$ only when $B \leq 1 / 4 ;$ for $1 / 4 < B < 1$ we use $\kappa = 1 \leq 4 B$ in the displayed inequality. If $h _ { \star } < A$ and $B < 1 / 4$ , then $\kappa = 4 B$ and $h _ { \star } = \sqrt { 2 A T } < A$ , which implies $2 T < A$ . This contradicts the standing assumption $A < T$ . Hence in the $h _ { \star } < A$ branch we have $B \geq 1 / 4$ , so $\kappa = 1$ and $B > T / ( 2 A )$ . For $B \geq 1$ ${ \sqrt { A B T } } > T / { \sqrt { 2 } } ;$ for $B < 1$ $B \sqrt { A T } > T / 2$ . The trivial T bound is therefore at most a constant times (2). These cases cover rounding and both clipping boundaries. □

Why this boundary is relevant to variance-aware claims. Any full minimax characterization with a zero variance budget must specialize to (9). An oracle upper bound containing $A + { \sqrt { A B T } }$ is valid but can be loose for $B \ll 1$ . Conversely, the known-budget filter must not be inserted into an algorithm that does not know a valid variation bound. Its permanent deletions would then have no certificate.

## 4 Minimax bounds with both budgets supplied

Let $\mathcal { E } ( A , T ; Q , B )$ be the environments of Section 2 satisfying $V _ { T } \leq Q$ and $\Delta \le B$ . Define

$$
{ \Re } ( A , T ; Q , B ) = \operatorname* { i n f } _ { \pi } \operatorname* { s u p } _ { \nu \in { \mathcal { E } } ( A , T ; Q , B ) } \mathbb { E } _ { \nu } [ R _ { T } ] .
$$

Here policies receive $A , T , Q , B$ , but no individual reward variances, means, or change locations. This side information is the same for every environment in the class. The budgets constrain the entire environment, rather than only the learner’s selected arms.

Write

$$
\begin{array} { l } { S = \sqrt { A Q } , \qquad C _ { B } = ( A Q B T ) ^ { { 1 } / { 3 } } , \qquad D _ { B } = \sqrt { A T } \operatorname* { m i n } \{ B , \sqrt { B } \} , } \\ { \Xi = \operatorname* { m i n } \{ T , A + S + C _ { B } + D _ { B } \} . } \end{array}\tag{12}
$$

The deterministic theorem explains $D _ { B }$ . Statistical hard blocks supply $S , C _ { B } ;$ calibration and clipping give the full upper bound.

Theorem 4.1 (All-budget variance–variation bounds). For every $A , T \ge 2 , Q \in [ 0 , T / 4 ]$ and $B \in [ 0 , T - 1 ]$ , universal constants $c , C > 0$ satisfy

$$
c \Xi \leq \Re ( A , T ; Q , B ) \leq C \log ( A T ) \Xi .\tag{13}
$$

$I f Q = 0$ , the sharper upper bound CΞ holds without logarithms. In particular:

$$
B \geq 1 : \Re ( A , T ; Q , B ) = \tilde { \Theta } \Bigl ( \operatorname* { m i n } \{ T , ( A Q B T ) ^ { 1 / 3 } + \sqrt { A B T } \} \Bigr ) ,\tag{14}
$$

$$
B = 0 : \quad \Re ( A , T ; Q , 0 ) = \widetilde { \Theta } \Bigl ( \operatorname* { m i n } \{ T , A + \sqrt { A Q } \} \Bigr ) .\tag{15}
$$

Equation (15) assumes constant means; the reward laws and their variances may still vary with time. The upper bound is executable with the supplied budgets and does not require individual variances, gaps or change locations.

For $Q > 0$ , this is a characterization up to one logarithmic factor, not a constant-factor characterization. Whether that logarithm can be removed is not resolved here.

The statistical construction below uses the standard change-of-measure and independent-block method from the bandit lower-bound literature [BGZ14, WHL16, WXC<sup>+</sup>25]. Its purpose here is to verify all budget constraints in one convention, including the arm dependence and the small-budget endpoints. We do not present the block construction as a new lower-bound principle. The ordinary restart upper bound follows from a prior hybrid static-regret algorithm (Proposition 6.2); the small-budget refinement is proved in Section 4.1.

A polynomial small-budget separation. For fixed A, take $Q = 1$ and $B = T ^ { - 1 / 4 }$ . Theorem 4.1 gives $\widetilde { \Theta } ( T ^ { 1 / 4 } )$ expected regret. The ordinary unit-range restart guarantee in Proposition 6.2 gives only ${ \widetilde O } ( T ^ { 3 / 8 } )$ after period tuning. Thus the small-budget refinement can improve the displayed upper guarantee by a polynomial factor even with a positive variance budget. This comparison concerns the guarantees; it supplies no lower bound on an unmodified restart policy’s actual regret.

Static guarantee used below. For $K \geq 2$ arms and $h \geq \operatorname* { m a x } \{ K , 3 \}$ rounds, the hybrid learner of [Ito21b], Theorem 3, equation (6), satisfies, against every fixed arm $a ,$

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { h } ( \ell _ { t , a _ { t } } - \ell _ { t , a } ) \right] \leq C _ { 0 } { \Big ( } K \log ( e h ) + { \sqrt { K \log ( e h ) \mathbb { E } [ { \mathcal { Z } } _ { h } ( u ) ] } } { \Big ) } ,\tag{16}
$$

where

$$
\mathcal { Z } _ { h } ( u ) = \sum _ { t = 1 } ^ { h } ( \ell _ { t , a _ { t } } - u _ { t , a _ { t } } ) ^ { 2 } + \sum _ { t = 1 } ^ { h - 1 } \| u _ { t + 1 } - u _ { t } \| _ { 1 } .
$$

The reference $u _ { t } \in [ 0 , 1 ] ^ { K }$ is used only in the analysis. It need not equal the loss expectation and is not an algorithmic input. The displayed form absorbs a $K { \sqrt { \log ( e h ) } }$ term into $K \log ( e h )$ . For shorter blocks, we play any eligible arm and use the trivial regret bound. In particular, we never invoke the cited algorithm’s $( \log h ) ^ { - 1 }$ parameter on a one- or two-round block. Proposition 6.2 gives the ordinary restart reduction explicitly. This separates the cited static theorem from our calibration and clipping analysis.

## 4.1 Safe calibration and clipping for small budgets

The ordinary restart bound pays a unit-range initialization cost in every block. For $B < 1$ , most arms need not incur this cost repeatedly. We first identify a fixed set containing every future optimal arm, and a common interval containing all of its future means. We then clip observations to a slightly wider interval. Importantly, the hybrid theorem is applied with the original mean sequence as its analytical reference; we never charge the variation of the clipped means.

Table 1: The width choice for $0 < B < 1$ . The last column bounds the additional monitoring and clipping-bias terms; $A + { \sqrt { A Q } }$ and $( A Q B T ) ^ { 1 / 3 }$ are accounted for separately. No clipping is needed in the last row.
<table><tr><td>Envelope budget</td><td>Width choice</td><td>Additional cost</td></tr><tr><td> $Q \leq B ^ { 2 } { \sqrt { A T } }$ </td><td> $w = B$ </td><td> $O ( B { \sqrt { A T } } )$ </td></tr><tr><td> $B ^ { 2 } { \sqrt { A T } } < Q < { \sqrt { A B T } }$ </td><td> $w = ( Q ^ { 2 } / ( A B T ) ) ^ { 1 / 3 }$ </td><td> $O ( ( A Q B T ) ^ { 1 / 3 } )$ </td></tr><tr><td> $Q \geq { \sqrt { A B T } }$ </td><td>Ordinary restarts</td><td> $O ( ( A Q B T ) ^ { 1 / 3 } )$ </td></tr></table>

Lemma 4.2 (Calibration with a small regret cost). Suppose $A < T , 0 < B \leq w \leq 1 , \Delta \leq B$ and $V _ { T } \leq Q$ . There is an observable calibration phase, ending after $\tau \leq T$ rounds, with the following properties. With probability at least $1 - \delta$ , its regret is at most

$$
C ( A \ell + { \sqrt { A Q \ell } } ) .\tag{17}
$$

If calibration finishes before the horizon, it returns a nonempty set C and an observed number z such that, for every $t > \tau$ , every optimal arm belongs to ${ \mathcal { C } } ,$ and

$$
\mu _ { t , a } \in [ z - 1 2 w , z + 4 w ] \qquad ( a \in \mathcal { C } ) .\tag{18}
$$

The good event and the output certificate are measurable with respect to the calibration history, for each fixed environment.

Proof. Pull each arm once, maintain the empirical statistics and radii $r _ { a }$ in (40) using all calibration samples of that arm, and initially set ${ \mathcal { C } } = [ A ]$ . Before each subsequent pull, form

$$
U _ { a } = \widehat { \mu } _ { a } + r _ { a } + 3 { \cal B } , \qquad L _ { a } = \widehat { \mu } _ { a } - r _ { a } - 3 { \cal B } .
$$

Delete each arm with $U _ { a } < \operatorname* { m a x } _ { b \in { \mathcal { C } } } L _ { b }$ . If all remaining radii are at most w, finish calibration and set $z = \mathrm { m a x } _ { a \in \mathcal { C } } \widehat { \mu } _ { a }$ . Otherwise pull a remaining arm with largest radius and update its statistics. Truncate this procedure if the horizon ends.

Use the simultaneous concentration bounds of Lemma C.1 only up to the end of calibration. Lemmas C.2 and C.3 apply to any nonanticipating sampling rule; both follow directly from that

concentration lemma, without using the interval-regret theorem. Since the entire mean path has variation at most B, they give, at any calibration decision time s, for every future $t \geq s ,$

$$
\begin{array} { r } { | \widehat { \mu } _ { a } - \mu _ { t , a } | \leq r _ { a } + 3 B . } \end{array}\tag{19}
$$

Indeed, the confidence bound for the current mean costs $r _ { a } + 2 B$ , and moving that mean to round t costs at most another B. A deleted arm is therefore worse than its witnessing arm at every future round. Deletions preserve every future optimal arm, and at least one arm always remains.

For a selected arm after initialization, its radius is larger than w and at least that of every remaining arm. For a currently optimal arm $b ,$ the fact that the selected arm a survived deletion implies $U _ { a } \geq L _ { b }$ . Together with (19), this gives

$$
\mu _ { s } ^ { \star } - \mu _ { s , a } \leq 2 r _ { a } + 2 r _ { b } + 1 2 B \leq 1 6 r _ { a } .
$$

Let $N _ { a }$ be the final calibration count and $\begin{array} { r } { B _ { a } = \sum _ { t \leq \tau : a _ { t } = a } \sigma _ { t , a } ^ { 2 } } \end{array}$ . For $N _ { a } \geq 2$ , the radius immediately before its last calibration pull, with $n = N _ { a } - 1$ , satisfies

$$
w < r _ { a , n } \leq \frac { 3 2 \sqrt { B _ { a } \ell } + 2 8 0 \ell } { n } + 2 4 B \sqrt { \ell / n } .
$$

If $n \geq 2 3 0 4 \ell$ , the last term is at most $w / 2$ , whence wn $\leq 6 4 \sqrt { B _ { a } \ell } + 5 6 0 \ell$ . For smaller $n ,$ use wn $\leq 2 3 0 4 \ell$ . The case $N _ { a } = 1$ is immediate. Summing yields

$$
w \tau \leq C ( A \ell + \sqrt { A Q \ell } ) ,\tag{20}
$$

Indeed, the preceding cases give, for every arm, w $N _ { a } \leq 6 4 \sqrt { B _ { a } \ell }$ + 2305ℓ (also for $N _ { a } = 1 )$ , and, pathwise,

$$
\sum _ { a } B _ { a } = \sum _ { t \leq \tau } \sigma _ { t , a _ { t } } ^ { 2 } \leq \Lambda _ { T } \leq V _ { T } \leq Q .
$$

This proves the time bound independently of any regret bound. The same reasoning holds when the phase is truncated before reaching its stopping criterion.

It remains to bound regret without summing a harmonic sequence of radii. Write $g _ { t , a } = \mu _ { t } ^ { \star } - \mu _ { t , a }$ so that $| g _ { t , a } - g _ { s , a } | \leq 2 B$ for every $s , t .$ . For an arm with $N _ { a } \geq 2$ , let $s _ { a }$ be its last calibration pull and put $n = N _ { a } - 1$ . Its final pull is a post-initialization selection even if the arm is deleted later, so the maximal-radius rule gives $g _ { s _ { a } , a } \leq 1 6 r _ { a , n }$ . No monotonicity of the radii is needed: the gap-variation bound compares each earlier pull directly with $s _ { a }$ . Thus all its pulls, including its initial one, contribute at most

$$
\begin{array} { r l r } {  { R _ { \mathrm { c a l } , a } \le N _ { a } \bigl ( 1 6 r _ { a , n } + 2 B \bigr ) } } \\ & { } & { \qquad \le C \bigl ( \sqrt { B _ { a } \ell } + \ell + B \sqrt { N _ { a } \ell } + B N _ { a } \bigr ) , } \end{array}\tag{21}
$$

For clarity, $N _ { a } / n \leq 2$ and (46) give

$$
N _ { a } r _ { a , n } \leq 6 4 \sqrt { B _ { a } \ell } + 5 6 0 \ell + 2 4 \sqrt { 2 } B \sqrt { N _ { a } \ell } ,
$$

which yields the second line of (21). Arms pulled only once cost at most $A \leq A \ell$ in total. Summing (21), using Cauchy–Schwarz, gives

$$
R _ { \mathrm { c a l } } \leq C \big ( A \ell + \sqrt { A Q \ell } + B \sqrt { A \tau \ell } + B \tau \big ) .
$$

Finally, $B \sqrt { A \tau \ell } \leq ( B \tau + A B \ell ) / 2$ and $A B \ell \leq A \ell .$ . Since $B \tau \leq w \tau$ , the independently established (20) absorbs both remaining terms. This proves (17), including the horizon-truncated case. The last-pull accounting removes the extra $1 + \log T$ that a direct radius sum would introduce.

At termination, all surviving radii are at most $w .$ . For an arm b attaining z, survival gives $\widehat { \mu } _ { a } \geq z - r _ { a } - r _ { b } - 6 B \geq z - 8 w$ . Equation (19) then implies (18). Define the good event by the required concentration inequalities at all calibration prefixes only. It has probability at least $1 - \delta$ by the simultaneous event, and is measurable at the stopping time $\tau ;$ no future rewards are used in this event or in the certificate. □

Lemma 4.3 (Clipped restarts with an original-mean reference). Condition on a calibration history for which every future optimal arm is retained and (18) holds. Put $J = [ z - 1 3 w , z + 5 w ]$ , of width $L = 1 8 w$ . Run fresh copies of the hybrid algorithm of [Ito21b] on the surviving arms, in consecutive blocks of length at most H, using normalized losses

$$
\ell _ { t , a _ { t } } ^ { \prime } = \frac { z + 5 w - \exp { \jmath ( X _ { t } ) } } { L } .
$$

On a block shorter than max $\{ A , 3 \}$ rounds, play any surviving arm instead. If there is only one survivor, play it throughout instead; its original-mean regret is zero because every future optimal arm is retained. The conditional expected regret on the remaining rounds is at most

$$
{ \tilde { O } } \left( { \sqrt { A [ T / H ] Q } } + w A [ T / H ] + H B \right) + { \frac { 2 Q } { w } } .\tag{22}
$$

No individual variance or mean is an algorithmic input.

Proof. Write $q _ { t } = \operatorname* { m a x } _ { a } \sigma _ { t , a } ^ { 2 }$ and $Y _ { t , a } = \mathrm { c l i p } _ { J } ( X _ { t , a } )$ . Every surviving arm’s original mean lies in $J$ at distance at least w from its endpoints. Projection and a second-moment tail bound $\mathrm { g i v e } .$ conditionally on the calibration history,

$$
\begin{array} { r } { \mathbb { E } [ ( Y _ { t , a } - \mu _ { t , a } ) ^ { 2 } ] \leq \sigma _ { t , a } ^ { 2 } , \qquad | \mathbb { E } [ Y _ { t , a } ] - \mu _ { t , a } | \leq \sigma _ { t , a } ^ { 2 } / w . } \end{array}\tag{23}
$$

Equivalently, with center $c = z - 4 w$ , the certified mean interval is $[ c - 8 w , c + 8 w ]$ and $J = [ c - 9 w , c +$ 9w]. Both sides therefore have exactly w of guaranteed margin; the apparent asymmetry comes only from using z as the reference point. For the second inequality, use $| Y - X | \leq | X - \mu | \mathbf { 1 } \{ | X - \mu | \geq w \}$ and $\mathbb { E } [ ( X - \mu ) ^ { 2 } ] = \sigma ^ { 2 }$

On a block I of length $h \geq \operatorname* { m a x } \{ A , 3 \}$ , use the analytical reference

$$
u _ { t , a } ^ { \prime } = ( z + 5 w - \mu _ { t , a } ) / L \in [ 0 , 1 ] .
$$

Theorem 3, equation (6), of [Ito21b], rescaled by $L ,$ gives clipped static regret at most

$$
\widetilde { O } \left( L A + \sqrt { A \mathbb { E } [ \Lambda _ { I } ] } + A \sqrt { L \Delta _ { I } } \right) .
$$

Here the squared residual is controlled by the first inequality in (23), while the reference path satisfies $\textstyle \sum _ { t } \| u _ { t + 1 } ^ { \prime } - u _ { t } ^ { \prime } \| _ { 1 } \leq A \Delta _ { I } / L$ . This invocation is valid even though the clipped observations are biased relative to the original means: the hybrid theorem permits arbitrary reference sequences and does not require them to be observation expectations.

Choose as comparator an arm optimal at the start of $I ;$ it is retained by hypothesis. Its origina mean regret difers from its clipped-mean regret by at most $2 \textstyle \sum _ { t \in I } q _ { t } / w ;$ moving from this fixed arm to the dynamic comparator costs at most $2 h \Delta _ { I }$ . If $h \geq A$ , then $A \sqrt { L \Delta _ { I } } \le L A + h \Delta _ { I }$ . For a shorter block, $h \leq A ,$ so use directly the range certificate: $R _ { I } \leq 1 6 w h \leq L A$ . Thus each block has conditional expected regret

```latex
Algorithm 2 Safe calibration followed by clipped restarts
Input: $A < T , 0 < B < 1 , Q > 0 , ( Q ^ { 2 } / ( A B T ) ) ^ { 1 / 3 } < 1$
1. Set $w , H$ by (24); set $\delta = T ^ { - 4 } .$
2. Set ${ \mathcal { C } } = [ A ]$ and pull every arm once.
3. for $t = A + 1 , \ldots , T$ do
4. Compute all calibration radii from (40).
5. Delete arms with $\begin{array} { r } { \widehat { \mu } _ { a } + r _ { a } + 3 B < \operatorname* { m a x } _ { b \in \mathcal { C } } ( \widehat { \mu } _ { b } - r _ { b } - 3 B ) } \end{array}$
6. if ma $\mathbf { x } _ { a \in \mathcal { C } } r _ { a } \le$ w then
7. Set $\begin{array} { r } { \tau = t - 1 , z = \operatorname* { m a x } _ { a \in \mathcal { C } } \widehat { \mu } _ { a } , J = [ z - 1 3 w , z + 5 w ] ; } \end{array}$ finish calibration.
8. Break the loop and keep $\mathcal { C } , J$ fixed.
9. else
10. Pull an arm of largest radius in C and update its empirical statistics.
11. end if
12. end for
13. if calibration finished with $\tau < T$ then
14. if $| { \mathcal { C } } | = 1$ then
15. Play its only arm for all remaining rounds.
16. else
17. Divide the remaining rounds into consecutive blocks of at most H rounds.
18. In a block of length $h \geq \operatorname* { m a x } \{ A , 3 \}$ , run a fresh copy of [Ito21b], Algorithm 1, with horizon
h; in a shorter block, play any surviving arm.
19. Supply each learner only the selected-arm loss $( z + 5 w - \mathrm { c l i p } _ { J } ( X _ { t } ) ) / ( 1 8 w )$
20. end if
21. end if
```

$$
\widetilde { O } \biggl ( w A + \sqrt { A \mathbb { E } [ \Lambda _ { I } ] } + h \Delta _ { I } \biggr ) + \frac { 2 } { w } \sum _ { t \in I } q _ { t } .
$$

Apply the same argument conditionally at every block start, using a fresh algorithm seed. The actual number of post-calibration blocks is $M _ { \tau } = \lceil ( T - \tau ) / H \rceil \leq \lceil T / H \rceil$ . Cauchy–Schwarz over these blocks gives the variance contribution $\sqrt { A M _ { \tau } Q } ;$ their restart costs sum to $w A M _ { \tau }$ . Their drift costs sum to at most $H B ,$ and the clipping bias to $2 Q / w$ . Appendix F details the conditional sum and tail budgets, including a short last block. This proves (22). Independence of future reward laws from the calibration history follows from the oblivious independent-round model. Conditioning here is on a good prefix, not on a concentration event involving future rewards. □

## 4.1.1 Completing the upper bound

Upper bound in Theorem 4.1. If $A \geq T$ , use the trivial horizon bound. If $Q = 0 ,$ , use Theorem 3.1. If $B = 0$ , a single unmodified hybrid run gives ${ \widetilde { O } } ( A + { \sqrt { A Q } } )$ by Proposition 6.2. For $B \geq 1$ , that proposition and oracle tuning already give the result.

It remains to consider $0 < B < 1$ and $Q > 0$ . Put

$$
w _ { 0 } = \left( \frac { Q ^ { 2 } } { A B T } \right) ^ { 1 / 3 } .
$$

If $w _ { 0 } \geq 1$ , then $Q ^ { 2 } \geq A B T$ , so $\sqrt { A B T } \leq ( A Q B T ) ^ { 1 / 3 }$ . The ordinary restart bound is again suficient. Otherwise set

$$
w = \operatorname* { m a x } \{ B , w _ { 0 } \} < 1 , \qquad H = \operatorname* { m i n } \left\{ T , \left\lceil \operatorname* { m a x } \left\{ 1 , ( A Q T / B ^ { 2 } ) ^ { 1 / 3 } , \sqrt { A w T / B } \right\} \right\rceil \right\} .\tag{24}
$$

Run calibration with $\delta = T ^ { - 4 }$ ; if it ends early, run the clipped restart policy on its surviving arms. Let $G \in { \mathcal { H } } _ { \tau }$ be the good calibration-prefix event, and let $K _ { \mathrm { c a l } } , K _ { \mathrm { p o s t } }$ be the uniform bounds in Lemmas 4.2 and 4.3. The tower property gives

$$
\mathbb { E } [ R _ { T } ] \le K _ { \mathrm { c a l } } + K _ { \mathrm { p o s t } } + T \mathbb { P } ( G ^ { c } ) .
$$

Appendix F proves this step explicitly, including $\tau = T$ and independence of future randomness. We give the period calculation directly for the clipped bound. Write

$$
\begin{array} { c } { h _ { v } = ( { A Q T } / { B ^ { 2 } } ) ^ { 1 / 3 } , \quad h _ { m } = \sqrt { A w T / B } , \quad s = \operatorname* { m a x } \{ 1 , h _ { v } , h _ { m } \} , } \\ { D = ( A Q B T ) ^ { 1 / 3 } , \qquad E = \sqrt { A w B T } . } \end{array}
$$

Let $M = \lceil T / H \rceil$ and $F ( H ) = { \sqrt { A M Q } } + w A M + B H$ . We claim

$$
F ( H ) \leq C _ { 1 } ( A + { \sqrt { A Q } } + D + E )\tag{25}
$$

with, for example, $C _ { 1 } = 2$ . The balancing identities used below are

$$
B h _ { v } = D , \qquad \sqrt { A Q T / h _ { v } } = D , \qquad B h _ { m } = E , \qquad A w T / h _ { m } = E .
$$

If $s \leq T$ , then $s \leq H = \lceil s \rceil \leq s + 1$ , including $s = T$ , because $T$ is an integer. Using $M \leq 1 + T / H$ gives

$$
\begin{array} { r l } & { \sqrt { A M Q } \leq \sqrt { A Q } + \sqrt { A Q T / H } \leq \sqrt { A Q } + D , } \\ & { ~ w A M \leq A w + A w T / H \leq A + E , } \\ & { ~ B H \leq B ( s + 1 ) \leq 2 B + B h _ { v } + B h _ { m } = 2 B + D + E . } \end{array}
$$

Here $H \geq h _ { v } , h _ { m } , w < 1$ , and $2 B \le A$ since $B < 1$ and $A \geq 2$ . This also covers $s = 1$ , integer periods, and a short last block, whose restart cost has already been included in M. Adding the three displayed estimates gives $F ( H ) \leq 2 A + \sqrt { A Q } + 2 D + 2 E$ , which proves the claimed constant. $\mathrm { ~ I f ~ } s > T$ , the horizon cap gives $H = T$ and $M = 1$ , so $F ( T ) = { \sqrt { A Q } } + A w + B T$ . At least one of $h _ { v } > T$ or $h _ { m } > T$ holds. Indeed, $h _ { v } > T$ implies $A Q > B ^ { 2 } T ^ { 2 }$ , whereas $h _ { m } > T$ implies $A w > B T$ . Thus the first gives $B T < { \sqrt { A Q } }$ , and the second gives $B T < A w \leq A$ . In either case $F ( T ) \leq 2 ( A + { \sqrt { A Q } } )$ , so (25) also holds when either or both unconstrained periods exceed the horizon. No property of an untruncated restart learner is used in this calculation.

Set $\mathscr { L } = \log ( e A T )$ . At $\delta = T ^ { - 4 }$ we have $\ell = \log ( 6 4 A T ^ { 8 } ) \leq 8 \mathcal { L }$ and $\log ( e T ) \leq \mathcal { L }$ . The explicit stopped-expectation bound in Appendix F and (25) give

$$
\begin{array} { r l } & { \mathbb { E } [ R _ { T } ] \le C _ { \mathrm { c a l } } ( A \ell + \sqrt { A Q \ell } ) + C ^ { \prime } \log ( e T ) F ( H ) + 2 Q / w + T ^ { - 3 } } \\ & { \qquad \le C _ { 2 } \mathcal { L } ( A + \sqrt { A Q } + D + E ) + 2 Q / w + T ^ { - 3 } , } \end{array}\tag{26}
$$

where $C _ { \mathrm { c a l } }$ is a universal calibration constant and $C _ { 2 } = 8 C _ { \mathrm { c a l } } + 2 C ^ { \prime }$ is suficient. These are additive phase costs; their logarithmic factors do not multiply. If $w = B$ , then $Q \le B ^ { 2 } \sqrt { A T }$ , whence $Q / w \leq B \sqrt { A T }$ and ${ \sqrt { A w B T } } = B { \sqrt { A T } }$ . If $w = w _ { 0 } \geq B$ , both $Q / w$ and $\sqrt { A w B T }$ equal $( A Q B T ) ^ { 1 / 3 }$ This proves the small-budget bound. On $G ^ { c }$ , total regret is at most T and $\mathbb { P } ( G ^ { c } ) \le T ^ { - 4 }$ , giving the displayed $T ^ { - 3 }$ term. All branches use supplied information and observed rewards only. Thus (26) is at most $C _ { 3 } \mathcal { L } ( A + \sqrt { A Q } + D + B \sqrt { A T } )$ for a universal $C _ { 3 } ,$ , after increasing it to absorb $T ^ { - 3 }$ Combining this with $R _ { T } \leq T$ and $\mathcal { L } \leq 2 \log ( A T )$ proves the stated single logarithm for $0 < B < 1$ The other noisy branches use the same explicit hybrid bound directly; the separate high-probability SEV-UCB analysis is not used to establish this expected single-logarithm claim. □

For $B \geq 1$ and $A \leq T$ , both A and $\sqrt { A Q }$ are at most a constant times ${ \sqrt { A B T } } ;$ for $A > T$ , the horizon cap applies. This proves (14). Setting $B = 0$ gives (15). For $Q = 0$ , the deterministic theorem matches the entire budget range without logarithms.

## 5 Adaptation on a disclosed deterministic family

The deterministic minimax theorem permits diferent policies for diferent supplied budgets. We next quantify the limitation on a single policy required to perform well on a stationary instance and on one-change alternatives.

Fix $g \in ( 0 , 1 / 2 ]$ . The stationary baseline $\nu _ { g } ^ { 0 }$ has rewards $1 / 2$ on arm 1 and $1 / 2 - g$ on every challenger. For $a \ge 2$ and $\tau \in \{ 2 , \ldots , T \}$ , the alternative $\nu _ { g } ^ { a , \tau }$ changes only arm a to $1 / 2 + g$ from τ onward. This family has zero noise variance, and each alternative has exactly one change with $\Delta = 2 g$ . All policies receive the initial vector and the family description. They are not told whether or when a change occurs.

For a policy π define

$$
r _ { 0 } ( \pi ) = \mathbb { E } _ { \nu _ { g } ^ { 0 } } [ R _ { T } ] , \qquad r _ { 1 } ( \pi ) = \operatorname* { m a x } _ { a , \tau } \mathbb { E } _ { \nu _ { g } ^ { a , \tau } } [ R _ { T } ] .
$$

The constrained value, for $r \in [ 0 , g T ]$ , is

$$
\mathfrak { F } _ { g } ( r ) = \operatorname* { i n f } _ { \pi : r _ { 0 } ( \pi ) \leq r } r _ { 1 } ( \pi ) .\tag{27}
$$

Knowing the family is a restricted promise. The corresponding upper bound is not a general non-stationary bandit algorithm.

Theorem 5.1 (Stationary-to-switching adaptation inequality). Let $n = A - 1$ and $q = T - 1$ . If $r _ { 0 } \le g n q / 2$ , then

$$
r _ { 1 } \ge \frac { g ^ { 2 } n q } { 8 ( r _ { 0 } + g n ) } .\tag{28}
$$

For every policy, $r _ { 1 } \ge r _ { 0 } - g$

Proof. On the baseline, the expected total number of challenger pulls is $r _ { 0 } / g$ . Thus some challenger has expected pull count in rounds $2 , \ldots , T$ at most $r _ { 0 } / ( g n ) \le q / 2$ . The gap-coupling calculation in Lemma A.1, using its fixed-arm bound and averaged over the q change times, gives

$$
r _ { 1 } \geq \frac { g q } { 8 ( r _ { 0 } / ( g n ) + 1 ) } = \frac { g ^ { 2 } n q } { 8 ( r _ { 0 } + g n ) } .
$$

The challenger is chosen by its expected baseline count, not as a function of the realized private random tape. For the second inequality, choose any challenger and $\tau = T$ . The histories and

Algorithm 3 Independent monitoring with a stationary-regret budget   
Input: The disclosed one-change family, $A , T , g ,$ stationary budget r   
1. Set $n = A - 1$ and $p = \operatorname* { m i n } \{ r / ( g T ) , \sqrt { n / T } \}$   
2. for $t = 1 , \dots , T$ do   
3. if an improved challenger has already been observed then   
4. Pull that challenger.   
5. else   
6. With probability p, pull a uniformly random challenger; otherwise pull arm 1.   
7. Observe its reward. If a challenger returns $1 / 2 + g ,$ record it as improved.   
8. end if   
9. end for

actions through round $T - 1$ coincide with the baseline. At round $T ,$ changing that arm can reduce instantaneous regret by at most $g .$ Therefore its expected regret is at least $r _ { 0 } - g$ □

Theorem 5.2 (Matching frontier on the disclosed one-change family). For $T \geq 1 6 A , g \in ( 0 , 1 / 2 ]$ ， and $r \in [ 0 , g T ]$ , universal constants $c , C > 0$ satisfy

$$
c \Phi _ { g } ( r ) \leq \mathfrak { F } _ { g } ( r ) \leq C \Phi _ { g } ( r ) , \qquad \Phi _ { g } ( r ) = \operatorname* { m i n } \left\{ g T , \ g \sqrt { n T } + \frac { g ^ { 2 } n T } { r + g n } \right\} ,\tag{29}
$$

where $n = A - 1$ . Thus the lower-bound obstruction is attainable, up to constants, throughout the stationary-budget range on this family.

Lower bound. Set $u = { \sqrt { n q } } ,$ so $u \geq 4$ and $n \leq u$ . If $r _ { 0 } \ge g u / 2$ , Theorem 5.1’s second inequality gives $r _ { 1 } \ge g u / 4$ . Otherwise its first inequality applies and gives

$$
r _ { 1 } \ge \frac { g ^ { 2 } n q } { 8 ( r _ { 0 } + g n ) } \ge \frac { g u } { 1 2 } .
$$

Consequently every policy has $r _ { 1 } \geq c _ { 0 } g \sqrt { n T }$ , independently of its stationary budget.

For a policy feasible in (27), $\mathrm { i f } \ r \le g n q / 2$ , (28) also gives $r _ { 1 } \ge g ^ { 2 } n q / ( 8 ( r + g n ) )$ . If $r > g n q / 2 .$ then

$$
\frac { g ^ { 2 } n T } { r + g n } \leq \frac { 2 g T } { q } \leq 4 g ,
$$

and this term is already bounded by a constant times the established floor. Combining the two lower bounds, and using $q \ge T / 2$ , proves a constant times (29) for every feasible policy. □

Upper bound. On the baseline, Algorithm 3 never detects an improvement, and has $r _ { 0 } = g p T \le r$ For an alternative, while no improvement has been observed, its challenger is probed independently with probability $p / n$ . From the change time, the expected time to its first probe is at most $n / p ,$ capped by the remaining horizon.

Before detection, the instantaneous regret is at most the baseline probe cost $g \mathbf { 1 } \{ a _ { t } \neq 1 \}$ plus an additional g in a post-change round that does not pull the improved arm. After detection, regret is zero. Hence, for $p > 0$ ,

$$
r _ { 1 } \leq g p T + g \operatorname* { m i n } \{ T , n / p \} .\tag{30}
$$

Every round’s regret is also at most $2 g$ . If $r \geq g \sqrt { n T }$ , take $p = { \sqrt { n / T } }$ to obtain $r _ { 1 } \le 2 g \sqrt { n T }$ . If $g n \leq r < g { \sqrt { n T } }$ , then

$$
r _ { 1 } \leq r + \frac { g ^ { 2 } n T } { r } \leq g \sqrt { n T } + \frac { 2 g ^ { 2 } n T } { r + g n } .
$$

The second step uses the two inequalities separately: $r \leq g \sqrt { n T }$ and $r + g n \le 2 r$ , the latter following from $r \geq g n > 0$ . In particular, no division by r is used when $r = 0$ . Together with $r _ { 1 } \le 2 g T$ , this is a constant times $\Phi _ { g } ( r )$ . Finally, if $0 \leq r < g n$ , the fraction in $\Phi _ { g } ( r )$ is at least $g T / 2$ , and the trivial $2 g T$ bound sufices. This includes $r = 0$ , when $p = 0$ and the policy always pulls the incumbent.

Corollary 5.3 (A variance-only stationary repair is insuficient). For fixed $A \geq 2$ and fixed $g > 0$ , no policy receiving identical side information on the baseline and alternatives can simultaneously achieve polylogarithmic expected regret on the baseline and $\widetilde { O } ( \sqrt { T } )$ expected regret on every one-change alternative.

Proof. Equation (28) gives $\Omega ( T / \mathrm { p o l y l o g } ( T ) )$ alternative regret. The same conclusion applies to uniform high-probability guarantees with $\delta = T ^ { - 2 }$ , since $0 \leq R _ { T } \leq T$ converts them into expected bounds with additive cost at most $1 / T$ □

Corollary 5.4 (Oracle rates cannot be attained simultaneously). No single policy without a supplied variation budget can achieve, for all deterministic arrays, ${ \widetilde { O } } ( \Psi ( A , T , \Delta ) )$ ) expected regret.

Proof. For fixed A and $g = 1 / 4$ , the baseline has $\Delta = 0$ and oracle rate $O ( A )$ , while every alternative has $\Delta = 1 / 2$ and oracle rate $O ( A + { \sqrt { A T } } )$ . Corollary 5.3 rules out simultaneously attaining these rates up to logarithmic factors. Disclosing the initial vector only makes the lower-bound problem easier. □

Interpreting the frontier. For constant g and A, a stationary budget much smaller than $\sqrt { T }$ causes a detection term of order $T / ( r + 1 )$ . Allowing stationary regret of order $\sqrt { T }$ reaches the one-change monitoring floor. This is a quantitative choice of guarantees, not a claim that every stationary instance must always incur $\sqrt { T }$ regret. The unrestricted noisy model may admit richer instance-dependent frontiers; the result here covers exactly the disclosed family.

## 5.1 A polynomial loss relative to the supplied-budget oracle

The preceding impossibility can be made quantitative without hiding the dependence on $A$ and $T$ For a policy that receives no variation budget, define its worst-case ratio to the rate function in Theorem 3.1 by

$$
\mathcal { C } ( \pi ; A , T ) = \operatorname* { s u p } _ { \mu \in [ 0 , 1 ] ^ { T \times A } } \frac { \mathbb { E } _ { \pi } [ R _ { T } ] } { \Psi ( A , T , \Delta ( \mu ) ) } .
$$

The denominator is the supplied-budget benchmark, evaluated at the actual variation. The policy knows the horizon and determinism but receives identical side information on all arrays. The matching upper frontier in Theorem 5.2 uses its family promise. In contrast, a lower bound on that family also constrains a general-model policy: run the policy unchanged on those arrays, ignoring any extra disclosure. The supremum defining $\mathcal { C }$ contains these arrays. Thus the following lower bound applies to the full deterministic model, without asserting a matching general upper frontier.

Corollary 5.5 (Quantitative oracle-adaptation loss). $I f T \geq 1 6 A$ , every such policy satisfies

$$
\mathcal { C } ( \pi ; A , T ) \ge c ( T / A ) ^ { 1 / 4 }\tag{31}
$$

for a universal $c > 0$ . This is a lower bound on the competitive ratio; no matching unrestricted upper bound is claimed.

The policy quantifier is unrestricted: for every nonanticipating randomized policy, including one designed for the full model, a baseline or one-change array witnesses the obstruction. Only the matching upper frontier uses the disclosed family promise.

Proof. Write $C _ { \pi } = \mathcal { C } ( \pi ; A , T )$ and apply the disclosed family with $g = 1 / 4 , n = A - 1$ and $q = T - 1$ Run $\pi$ unchanged, ignoring the disclosed baseline vector and family promise. Theorem 5.1 applies because it holds for every policy on that family, including those that ignore the extra information. On the baseline, $r _ { 0 } \le C _ { \pi } A$ . On every alternative,

$$
\begin{array} { r } { r _ { 1 } \leq C _ { \pi } ( A + \frac 1 2 \sqrt { A T } ) \leq \frac 3 4 C _ { \pi } \sqrt { A T } . } \end{array}
$$

The baseline and alternative hard mixture in Lemma A.1 already implies $C _ { \pi } \geq 1 / 1 2 0$ . If $r _ { 0 } > g n q / 2$ then $C _ { \pi } > n q / ( 8 A ) \geq T / 3 2$ . For $T \geq 1 6 A$ and $A \geq 2 , T / 3 2 \geq \frac { 1 } { 2 } ( T / A ) ^ { 1 / 4 }$ , proving the claim in this case. Otherwise, Theorem 5.1 gives

$$
{ \textstyle \frac { 3 } { 4 } } C _ { \pi } \sqrt { A T } \ge \frac { n q } { 1 2 8 ( C _ { \pi } A + n / 4 ) } \ge \frac { T } { 5 1 2 ( C _ { \pi } + 1 / 4 ) } .
$$

Hence $C _ { \pi } ( C _ { \pi } + 1 / 4 ) \geq { \sqrt { T / A } } / { 3 8 4 }$ . Using $C _ { \pi } \geq 1 / 1 2 0$ gives $C _ { \pi } + 1 / 4 \le 3 1 C _ { \pi }$ , and proves (31), for example with $c = 1 / 1 2 8$ □

## 5.2 Robustness to a finite improvement pulse

The frontier is not restricted to an improvement that persists forever. Let $\mathcal { H } _ { g } ^ { \mathrm { p u l s e } }$ contain the baseline and arrays in which one challenger has reward $1 / 2 + g \mathrm { \ o n \ } [ \tau , \rho )$ , where ${ \bar { 2 } } \leq \tau < \rho \leq T + 1$ , and has its baseline reward outside that interval. The initial vector and this single-pulse promise are disclosed. A pulse ending before the horizon costs variation $4 g ;$ a persistent improvement costs $2 g$

Corollary 5.6 (Single-pulse frontier). Define the constrained value as in (27), taking the worst case over all pulses. Under the conditions of Theorem 5.2, it is also $\Theta ( \Phi _ { g } ( r ) )$ .

Proof. Persistent alternatives $( \rho = T + 1 )$ give the same lower bound. For the upper bound, use Algorithm 3 until the improvement is observed, then pull the improved arm until its reward returns to $1 / 2 - g$ . After that observation, play the incumbent for all remaining rounds. Before detecting an improvement, the baseline probing charge is at most gpT in expectation. Undetected pulse rounds contribute at most $g$ min $\{ T , n / p \}$ in expectation for $p > 0$ . The first round observing the return to baseline costs at most one additional $g .$ . Thus

$$
r _ { 1 } ^ { \mathrm { p u l s e } } \leq g p T + g \operatorname* { m i n } \{ T , n / p \} + g .
$$

The same cases as in Theorem 5.2 apply; the added $g$ is absorbed by $\Phi _ { g } ( r ) \geq g$ . The bound $2 g T$ covers $p = 0$ and the small-r case. The baseline regret remains $g p T$ □

The stopping rule uses the disclosed single-pulse promise. It provides no guarantee for an unknown number of subsequent improvements.

## 6 Implications for the general variance-aware problem

The supplied-budget characterization and adaptation frontier impose distinct constraints. An oracle minimax rate must recover the sharper small-budget deterministic boundary. A single prior-free policy cannot recover all oracle rates without an adaptation cost. That cost is already nontrivial on a disclosed deterministic family, where estimation is free.

Theorem 6.1 (Observable learner with supplied noisy budgets). For supplied bounds $0 \leq Q \leq T / 4$ and $0 \leq B \leq T - 1$ , there is a policy knowing A, T, δ, Q, B but no individual variances, means or change locations such that, on every environment with $V _ { T } \leq Q$ and $\Delta \le B$ , with probability at least $1 - \delta$

$$
R _ { T } \leq \operatorname* { m i n } \left\{ T , \ C D _ { T } \left( A + \sqrt { A Q } + ( A Q B T ) ^ { 1 / 3 } + \sqrt { A B T } \right) \right\} .\tag{32}
$$

$I f Q = 0$ , a separate deterministic branch attains $C \Psi ( A , T , B )$ pathwise. No knowledge of the actual variance of a reward is required in either branch.

Proof. For $Q > 0$ , run the observable learner in Algorithm 4 with the period of Corollary D.2; that corollary and $R _ { T } \le T$ give (32). If $Q = 0$ , every reward law is a point mass. Apply Algorithm 1 and Theorem 3.1. Both branches are selected using supplied information, not an unobservable test for zero variance. □

Theorem 6.1 is a separate high-probability implementation of the ordinary known-budget bound. The sharper all-budget expected characterization is Theorem 4.1. Its variance-sensitive cubic term uses $A Q$ with Q an envelope budget. In the earlier bound [WHL16], the corresponding term uses $A ^ { 2 }$ times a full-vector variance budget. The distinction must be retained when comparing arm dependence. Proposition 6.2 shows explicitly that the expected ordinary restart rate follows from prior work. We retain this theorem for its observable high-probability implementation and the minimax comparison.

Unknown-budget scope. No sharp all-budget guarantee is claimed for a policy that does not receive $Q , B ,$ . Our general-model upper bounds require these budgets; the matching adaptive upper bound uses the disclosed deterministic family. Removing both forms of side information while retaining variance-sensitive regret is an open problem.

## 6.1 An existing static guarantee already implies the expected upper rate

The noisy expected upper rate is not evidence of a new base algorithm. Theorem 3, equation (6), of [Ito21b] gives, against each fixed arm a, a static expected-regret bound of the form

$$
\widetilde { O } \left( A + \sqrt { A \mathbb { E } \left[ \sum _ { t \in I } ( \ell _ { t , a _ { t } } - u _ { t , a _ { t } } ) ^ { 2 } + \sum _ { t , t + 1 \in I } \Vert u _ { t + 1 } - u _ { t } \Vert _ { 1 } \right] } \right) .\tag{33}
$$

The sequence $u _ { t }$ is used only in the analysis, and is not supplied to the algorithm. Rewards are converted to losses by $\ell _ { t } = 1 - X _ { t }$

Proposition 6.2 (Restarting a prior hybrid algorithm). Restarting the algorithm of $\left[ I t o \mathcal { Q } 1 b \right]$ at fixed period H gives

$$
\mathbb { E } [ R _ { T } ] = \widetilde { O } \bigg ( \sqrt { A [ T / H ] Q } + A \lceil T / H \rceil + H B \bigg )\tag{34}
$$

on $\mathcal { E } ( A , T ; Q , B )$ . In particular, tuning H with the supplied budgets gives the same expected rate as Theorem 6.1. Each block of length $h \geq \operatorname* { m a x } \{ A , 3 \}$ uses a fresh hybrid copy with horizon h; on shorter blocks, play any arm. This defines the policy also when a block contains a single round.

Proof. For a fixed interval I of length $h \geq \operatorname* { m a x } \{ A , 3 \}$ , set $u _ { t }$ equal to the mean loss vector $1 - \mu _ { t }$ The squared residual in (33) has expectation $\mathbb { E } [ \Lambda _ { I } ]$ , and the path term is at most $A \Delta _ { I }$ . Converting static regret against an arm optimal at the start of I to dynamic regret costs at most $2 h \Delta _ { I }$ . Thus a fresh run has expected dynamic regret

$$
\widetilde O \left( A + \sqrt { A \mathbb { E } [ \Lambda _ { I } ] } + A \sqrt { \Delta _ { I } } + h \Delta _ { I } \right) .
$$

If $h \geq A$ , then $A \sqrt { \Delta _ { I } } \le A + h \Delta _ { I }$ . For a shorter block, the trivial $R _ { I } \leq h \leq A$ sufices. Therefore the interval bound is $\widetilde { O } ( A + \sqrt { A \mathbb { E } [ \Lambda _ { I } ] } + h \Delta _ { I } )$ . Conditional on the history at a fixed block start, the oblivious environment and a fresh algorithm seed justify the same bound. Sum over blocks and use $\Sigma _ { I } \mathbb { E } [ \Lambda _ { I } ] \le Q$ , Cauchy–Schwarz and $\begin{array} { r } { \sum _ { I } h _ { I } \Delta _ { I } \leq H B } \end{array}$ to prove (34). The period in Corollary D.2 gives the last claim. □

This reduction is an inference from the cited theorem, not a dynamic regret statement quoted verbatim from that paper. It does not by itself give the small-budget clipping refinement in Section 4.1, the path-dependent high-probability theorem in Appendix C, or an unknown-budget scheduler. We therefore use ordinary restarting as a prior-work baseline for the calibration and clipping refinement.

## 7 Related work

Our comparison separates the comparator, feedback, variation convention, variance convention and supplied information. Changing any of these can invalidate a direct comparison of displayed rates.

Table 2 summarizes the closest finite-arm rate comparisons. In its first row, $V _ { 0 } = \| 1 - \mu _ { 1 } \| _ { \infty } + \Delta$ is variation for losses with an initial zero vector, and $W _ { T }$ is the full-vector variance in (7). In particular, $V _ { 0 } \leq 1 + B$ and $W _ { T } \leq A Q$ on our class, but substituting these inequalities does not identify the two budget conventions. The second row is a reduction carried out here from a cited static theorem, not a dynamic theorem quoted verbatim from that source.

Table 2: Closest finite-arm guarantees. Upper rates suppress logarithms and the trivial horizon cap. All rows compare to the best mean arm at each round; the supplied information and budget conventions difer.
<table><tr><td>Source</td><td>Rate</td><td>Scope and attribution</td></tr><tr><td>[WHL16], Theorem 3.1</td><td> $( A ^ { 2 } W _ { T } V _ { 0 } T ) ^ { 1 / 3 } + { \sqrt { A V _ { 0 } T } }$ </td><td>Supplied full-vector variance and initial-inclusive variation; prior result.</td></tr><tr><td>[Ito21b], Theorem 3, plus restarting</td><td> $A + { \sqrt { A Q } } + ( A Q B T ) ^ { 1 / 3 } + { \sqrt { A B T } }$ </td><td>Supplied  $Q , B ;$  direct reduc- tion in Proposition 6.2.</td></tr><tr><td>Theorem 4.1</td><td> $A + { \sqrt { A Q } } + ( A Q B T ) ^ { 1 / 3 } + { \sqrt { A T } } \operatorname* { m i n } \{ B , { \sqrt { B } } \}$ </td><td>Supplied Q, B; matching lower bound up to logs. The noisy refinement is for  $B < 1$ </td></tr></table>

Variation budgets and the deterministic boundary. The stochastic variation-budget theory of [BGZ14, BGZ19] yields the familiar cubic rate. MOSS-based resetting, sliding-window and discounted policies are studied by [WS24]. These stochastic models permit nonzero observation noise, so their minimax lower bounds do not establish the minimax value of the deterministic subclass. [KA16] already studied adaptation to unknown variation, including an intermediate noiseless construction. Its supplementary Theorem A.1 already gives a prior-free deterministic guarantee of $O ( \log T [ \sqrt { A T } + \bar { A } ^ { \bar { 1 } / 3 } T ^ { 2 / 3 } \Delta ^ { 1 \bar { / } 3 } ] )$ . Supplementary Appendix D outlines the general-arm noisy extension but explicitly defers its full analysis. Thus neither “non-stationarity with zero noise” nor “unknown variation” is a new problem formulation. In [WHL16], Theorem 3.1 uses full-vector statistical variance and variation including the initial loss vector. Its deterministic specialization already recovers the large-budget scale, up to logarithmic factors.

There is also a short derivation of the small-budget upper rate from that earlier result. After the filter in Lemma 3.2, every retained arm’s future reward lies in $[ y _ { \mathrm { m a x } } - 3 B , y _ { \mathrm { m a x } } + B ]$ . If $0 < B \le 1$ transform these rewards by $x \mapsto ( x - y _ { \mathrm { m a x } } + 3 B ) / ( 4 B )$ . The transformed rewards are in [0, 1], have consecutive variation at most $1 / 4 ,$ , and have initial-vector variation at most 1. Applying the deterministic specialization of [WHL16] on the remaining horizon and undoing the scaling gives $\widetilde { O } ( B \sqrt { A T } )$ , in addition to the A initialization cost. This is a derivation in this manuscript, not a theorem attributed verbatim to that paper. Theorem 3.1 provides an elementary logarithm-free upper bound and an explicit oblivious lower bound throughout the budget range. The resulting deterministic scale is therefore compatible with the earlier upper bound; the calibration and clipping argument is needed to handle positive noise in the small-budget region.

Static path-length regret is a diferent guarantee. [WL18, BLLW19] obtain adaptive staticregret bounds in terms of realized loss variation. In particular, Theorem 2 of [BLLW19] gives $\widetilde { O } ( A + \sqrt { A P _ { \infty } } )$ when an upper bound on $P _ { \infty }$ is supplied, where $\begin{array} { r } { P _ { \infty } = \sum _ { t = 2 } ^ { T } \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } } \end{array}$ for a deterministic loss array. This compares to a fixed arm, whereas our $R _ { T }$ compares to a possibly diferent best arm every round. For stochastic observations, realized path length and mean variation also difer: independent $\operatorname { B e r n o u l l i } ( 1 / 2 )$ rewards have $\Delta = 0 .$ , but already one arm contributes expected path length $( T - 1 ) / 2$ . These static-comparator bounds do not directly imply a variance-sensitive dynamic-regret guarantee without supplied budgets.

Prior-free tracking and significant shifts. ADSWITCH [AGO19] adapts to distribution changes, and the contextual replay approach of [CLLW19] simultaneously controls switching and distributional total variation. The latter’s variation uses total variation distance between laws, which need not be comparable from above by our mean variation. MASTER [WL21] supplies a broad reduction; its Section 5 identifies dificulties with data-dependent base guarantees. META [SK22] goes further by tracking significant arm shifts. Its Theorem 1 and Corollaries 1–2 imply, on one and the same policy,

$$
R _ { T } = \widetilde { \cal O } \left( \operatorname* { m i n } \left\{ \sqrt { A ( \widetilde { S } + 1 ) T } , \sqrt { A T } + ( A \Delta ) ^ { 1 / 3 } T ^ { 2 / 3 } \right\} \right) ,
$$

where $\widetilde { S }$ counts significant shifts and is no larger than the number of best-arm switches. ArmSwitch [AYGL23] independently adapts to best-arm switches and gives a variation bound for a modified elimination condition. These are stronger comparators than merely counting all mean changes. Their unit-range variation term has the scale $( A \Delta ) ^ { 1 / 3 } T ^ { 2 / 3 }$ , corresponding to the variance-sensitive cubic term $( A V _ { T } \Delta T ) ^ { 1 / 3 }$ when $V _ { T } = \Theta ( T )$ . Achieving low-noise dependence together with adaptation to unknown budgets remains open.

Exploration–detection tradeofs and alternative objectives. The inverse stationary-regret tradeof is classical. We cite both the published version [GM11] and the detailed preprint [GM08], whose Theorem 13 uses finite-KL stochastic pulse alternatives. That theorem does not apply verbatim to disjoint deterministic reward laws. Our coupling proof treats that boundary and gives a matching policy on a disclosed family; it is not a claim to have discovered the exploration–detection obstruction. The polynomial consequence in Corollary 5.5 also follows from this obstruction and the oracle benchmark. Theorem 31.2 of [LS20] gives a particularly direct two-arm Gaussian pulse formulation of the inverse-regret tradeof. [MZ21] study model-selection frontiers, with an adaptiveadversary switching obstruction. [QW26] achieve simultaneous static and dynamic bounds for deterministic losses, an oblivious adversary and a supplied switch-count bound; their stationary static guarantee has scale AT. Satisficing regret [ZZX26] compares against a threshold, rather than the best arm, and is a diferent objective. These results do not contradict our small-stationary-regret tradeof.

Variance estimation and detection frameworks. Variance-based stochastic and best-of-bothworlds methods include [AMS09, Ito21b, ITH22]. A static guarantee does not alone identify the dynamic rate. Proposition 6.2 gives the explicit reduction for [Ito21b], rather than classifying it as unrelated merely because its comparator is fixed. The empirical-variance/restarting template is also present in [WHL16], Supplement B.1; retaining per-arm residual sums before Cauchy–Schwarz refines the coarse aggregation there. In the linear setting, [WXC<sup>+</sup>25] give variance-sensitive restart algorithms and a bandits-over-bandits variant, as well as a variance-dependent linear-bandit lower bound (Theorem 4.1 and Appendix C). The drift norm, dimension and tuning information difer from our finite-arm model. This distinction is quantitative: their Corollary 5.4 is stated for variance and parameter-variation budgets bounded below by constants, and Remark 5.6 discusses the rate loss in the finite-arm specialization. Thus that paper is a direct variance-sensitive comparison, including its finite-arm case, rather than being excluded solely by its linear formulation. GLR-klUCB [BKMS22], DAB [HGBV26] and DAL [GHBV26] provide practical detection/restart frameworks. Their stated performance theorems impose explicit spacing or detector-data conditions. For example, DAL’s Theorem 4.8 assumes Assumption 4.6. The phrase “prior free” therefore does not by itself mean a uniform guarantee over arbitrary drifting arrays. Our comparison uses the AISTATS 2025 version of [WXC<sup>+</sup>25], the ICML 2026 version of DAL, and v2 of [QW26] dated 17 February 2026. Earlier multiscale detection [DHS20] similarly uses explicit detectability conditions (Assumption 4 and Theorem 3 in its finite-arm specialization).

Structural promises and other observation models. ADR-bandit [KFH24] preserves logarithmic stationary regret and handles coordinated global changes. Its Definitions 19–20 and Theorem 21 require detectable changes mirrored across arms; our hidden-improvement alternatives keep the incumbent unchanged. Thus its simultaneous positive result is compatible with our lower bound. Recent misspecification reductions [HYZZ26] address round-specific linear action sets and supplied parameter path length, rather than a noise-adaptive unknown-budget finite-arm theorem. For heavy-tailed observations, [GBGM26] give a Catoni-style detector and a piecewise-stationary regret analysis under Assumption 4.7. Their moment budget and separation conditions difer from our bounded-reward variance-envelope class. These distinctions are part of the comparison, not grounds for omitting adjacent work.

Smoothness and gap structure provide further forms of side information. [KG25] use a supplied per-round drift limit; [MCY21] analyze controlled changes of gaps and the best mean, with tuning parameters supplied to their algorithm. [Suk25] adapt to unknown Hölder smoothness and give faster gap-dependent rates under a safe-environment condition. In the public $\mathrm { v 2 , }$ Definition 5 and Theorem 7 impose the existence of an arm whose cumulative gaps remain below a sampling threshold on every interval. A hidden improvement with long pre- and post-change segments can violate this condition, even at zero observation variance. These gap-dependent positive results are therefore relevant structural comparisons rather than uniform guarantees for our entire class. Finally, [CYZ25] interpolate between static and dynamic oracles using a switch-budget-constrained arm sequence. Their Algorithm 2 receives both switch and variation budgets; its two-regime bound concerns that comparator and does not supply unknown-budget low-noise adaptation.

Proposition 7.1 (Sampling variance survives deterministic rewards). For an arm sampled with predictable probability $p > 0$ , let $Z = \mathbf { 1 } \{ a _ { t } = a \} X _ { t , a } / p$ . Conditional on the pre-round history,

$$
\operatorname { V a r } ( Z ) = { \frac { \sigma _ { t , a } ^ { 2 } } { p } } + \mu _ { t , a } ^ { 2 } \left( { \frac { 1 } { p } } - 1 \right) .\tag{35}
$$

More generally, with a predictable baseline b, replacing $X _ { t , a }$ by $X _ { t , a } - b$ replaces $\mu _ { t , a } ^ { 2 }$ by $( \mu _ { t , a } - b ) ^ { 2 }$

Proof. Here the history is taken before drawing the action’s fresh private randomness, and $X _ { t , a }$ denotes a potential draw from $\nu _ { t , a }$ . The conditional first moment of $Z$ is $\mu _ { t , a }$ and its second moment is $( \sigma _ { t , a } ^ { 2 } + \mu _ { t , a } ^ { 2 } ) / p$ . Subtracting the square of the first moment proves the identity. The centered case is the same calculation. □

This elementary identity explains a technical obstacle to directly substituting a reward-variance bonus into an importance-weighted replay test: zero reward noise does not imply zero variance of its estimator. Centering can help, but controlling its residual and the sampling schedule requires a separate argument.

Reward ranges, clipping and statistical variation. Range adaptation is an established issue. [HS23] show a tradeof when the support interval of stationary rewards is unknown (Theorem 3). Our original support [0, 1] is known; calibration certifies a mean interval using a supplied drift budget, after which the learner chooses the clipped-feedback support. This does not infer unknown reward-support endpoints. [GL16] show that static adversarial regret cannot scale with the acrossarm efective loss range alone (Corollary 4). Here the total mean path is constrained and clipping bias is charged to the variance envelope; a small instantaneous range alone would not justify our guarantee. The quadratic-variation result of [BCL18], Theorem 2, concerns realized loss vectors and a fixed comparator. These conventions difer from observation variance about a drifting mean. The hybrid theorem of [Ito21b] is the explicit static ingredient in our upper bound. Calibration, clipping and balancing its bias are the additional steps in the supplied-small-budget argument, rather than a new static-regret regularizer. A separate work of [Ito21a] extends hybrid data-dependent guarantees to combinatorial semi-bandits and adversarial linear bandits. Its static comparison and realized loss-vector quadratic variation and path length are distinct from our dynamic comparator and reward-noise envelope; it is not the COLT theorem invoked in (16).

Earlier detection algorithms also address robustification and monitoring. CD-UCB [LLS18] uses a change-detection wrapper; its CUSUM guarantees impose segment-length, detectability and Bernoulli conditions. M-UCB [CWKX19], Assumption 1 and Theorem 1, requires a windowdependent separation and a detectable mean shift. [MM19] analyze gap-dependent bounds under global, separated changes. [HZG24] study forced-exploration schedules with a gap-dependent sub-Gaussian analysis. RCD-UCB [WPZX21] already uses truncated loss in a robust detector; its Assumption 2.1 and Theorem 2.2 retain separation and detectability conditions. Thus truncation itself is not a new bandit technique. Our clipping step bounds the regret of a restarted learner by using original means as an analytical reference; it does not attempt to detect individual change points.

Directly observed mean tracking. The Anytime Tracking CUSUM method of [GIZ26] studies multiple change points without minimum spacing or jump-size assumptions. Its Section 2, however, provides a stream observation every round and measures cumulative squared mean-prediction error, with a known sub-Gaussian proxy. There is no choice of which arm to observe. This is an important adjacent monitoring result, but its guarantee cannot be substituted for a selected-arm dynamic-regret bound. Its Theorem 4.1 also retains a mean-diameter contribution in addition to its variance terms; we do not interpret it as a zero-cost guarantee at zero noise.

Summary of the comparison. Section 7 and Appendix G distinguish reward-noise variance, realized loss variation, and the side information available to each policy. The ordinary expected restart bound is a consequence of the prior static hybrid theorem. The small-budget refinement additionally uses calibration, clipping and bias control.

## 8 Conclusion

We give regret bounds across the full range of supplied variance and mean-variation budgets. The deterministic minimax rate is matched up to universal constants; for positive variance, the upper and lower bounds difer by at most one logarithmic factor. Safe calibration and clipping provide the small-budget refinement of the ordinary restart guarantee. On a disclosed deterministic one-improvement family, we characterize the stationary/change-regret frontier and its extension to one finite pulse. The same hard instances imply a general lower bound on adaptation to unknown variation budgets. Removing the logarithmic gap in the noisy minimax bound, and obtaining matching adaptation frontiers for unrestricted noisy environments, remain open.

## A Oblivious deterministic lower bounds

The following constructions prove the lower bound in Theorem 3.1. The mixtures are fixed before interaction.

## A.1 A fixed hard distribution

Our lower bound uses a distribution fixed before interaction. This avoids selecting an epoch’s environment after observing the learner’s realized history.

Lemma A.1 (One-block hidden-improvement mixture). Fix $h \geq A \geq 2$ and $g \in ( 0 , 1 / 2 ]$ . In a block of h rounds, use baseline rewards $1 / 2$ for arm 1 and $1 / 2 - g$ for every challenger. With probability $1 / 2$ keep this baseline throughout. Otherwise choose a challenger uniformly and a change time uniformly from $\{ 2 , \ldots , h \}$ , and change that challenger to $1 / 2 + g$ from that time onward. For every policy, even with the baseline disclosed in advance and any initial private state independent of this block’s random draw,

$$
\mathbb { E } [ R _ { \mathrm { b l o c k } } ] \geq { \frac { g } { 4 0 } } { \sqrt { A h } } .\tag{36}
$$

Every supported array has internal variation at most $2 g$

Proof. Set $n = A - 1$ and $q = h - 1$ . On the baseline, let $N _ { a }$ count pulls of challenger a in rounds $2 , \ldots , h$ , and let $\begin{array} { r } { N = \sum _ { a \ge 2 } \mathbb { E } [ N _ { a } ] } \end{array}$ . Its expected regret is at least $g N$ . For a fixed challenger and a fixed policy seed, its baseline trajectory has $N _ { a } + 1$ gaps containing no pull of that arm, of lengths $d _ { j }$ with $\begin{array} { r } { \sum _ { j } d _ { j } = q - N _ { a } } \end{array}$ . A change in a gap is unobserved until the next pull, and the coupled actions remain the same. The sum of detection delays over change positions in that gap is $d _ { j } ( d _ { j } + 1 ) / 2$ Each undetected round costs at least $g .$

Let U be the learner’s private random tape and initial private state, independent of the challenger and change-time draw. Write $R _ { \mathrm { b l o c k } } ^ { a , t }$ for regret on the alternative changing challenger a at time t. The function $f ( x ) = ( q - x ) ^ { 2 } / ( x + 1 )$ satisfies, on $[ 0 , q ]$

$$
f ^ { \prime } ( x ) = 1 - { \frac { ( q + 1 ) ^ { 2 } } { ( x + 1 ) ^ { 2 } } } \leq 0 , \qquad f ^ { \prime \prime } ( x ) = { \frac { 2 ( q + 1 ) ^ { 2 } } { ( x + 1 ) ^ { 3 } } } > 0 .
$$

For a fixed challenger, Cauchy–Schwarz on its baseline gap lengths and then Jensen over U give

$$
\frac { 1 } { q } \sum _ { t = 2 } ^ { h } \mathbb { E } _ { U } [ R _ { \mathrm { b l o c k } } ^ { a , t } ] \geq \frac { g } { 2 q } \mathbb { E } _ { U } [ f ( N _ { a } ( U ) ) ] \geq \frac { g } { 2 q } f ( \mathbb { E } _ { U } [ N _ { a } ] ) .
$$

The first inequality drops the nonnegative linear part of $\textstyle \sum _ { j } d _ { j } ( d _ { j } + 1 )$ . In particular, for any fixed arm with $\mathbb { E } _ { U } [ N _ { a } ] \le v \le q / 2$ , monotonicity gives a bound of $g q / [ 8 ( v + 1 ) ]$ ]. Averaging instead over the independent uniform challenger and applying Jensen again bounds average alternative regret by

$$
\frac { g } { 2 q } \frac { ( q - N / n ) ^ { 2 } } { N / n + 1 } .\tag{37}
$$

Let $u = { \sqrt { n q } }$ . If $N \geq u / 4$ , the baseline half of the mixture already contributes $g u / 8$ . Otherwise $N / n < q / 2$ and (37) is at least

$$
\frac { g q } { 8 ( N / n + 1 ) } = \frac { g n q } { 8 ( N + n ) } \geq \frac { g u } { 1 0 } ,
$$

because $n \leq q$ and $N < u / 4$ . The alternative half contributes at least $g u / 2 0$ . Finally $n \geq A / 2$ and $q \geq h / 2$ give $u \geq \sqrt { A h } / 2$ , proving (36). □

Lower bound in Theorem 3.1. First, a stationary array with one uniformly random arm having reward 1 and all others reward 0 forces expected regret at least $( \operatorname* { m i n } \{ A , T \} ) / 4$ . Under the uniform choice of the hidden best arm, the probability that it has been tested by round $t \leq A$ is at most $t / A \colon$ until it is found, every observed reward is zero and at most t distinct arms can have been tried. Pulling the optimal arm in round t implies that it has been tested by that round. Hence

$$
\mathbb { E } [ R _ { T } ] \geq \sum _ { t = 1 } ^ { \operatorname* { m i n } \{ A , T \} } ( 1 - t / A ) \geq \operatorname* { m i n } \{ A , T \} / 4 .
$$

This family has zero variation.

If $T < 4 A$ , this lower bound is already a constant fraction of the horizon and sufices. Hence suppose $T \geq 4 A$ . For $0 < B \le 1$ , use Lemma A.1 with $h = T$ and $g = B / 2$ . Its variation is at most B and its expected regret is at least $B \sqrt { A T } / 8 0$

For $B \geq 1$ , let

$$
m = \operatorname* { m i n } \{ \lfloor B \rfloor , \lfloor T / ( 2 A ) \rfloor \} .
$$

Partition the horizon into m blocks whose lengths difer by at most one, and draw the mixture in Lemma A.1 independently in each block, with $g = 1 / 4$ . Every block starts from the same baseline.

Internal changes cost at most $1 / 2$ per block; resetting at a boundary costs at most $1 / 2$ . Total variation is therefore at most $m - 1 / 2 \le B$ . Every block has length at least 2A. Conditional on any previous history, its fresh draw is independent of the learner’s private state, so the lemma remains applicable. The expected total regret is at least a constant times

$$
\sum _ { i = 1 } ^ { m } \sqrt { A h _ { i } } \ge m \sqrt { A | T / m \rfloor } \ge \sqrt { A m T / 2 } = \Omega \Bigl ( \operatorname* { m i n } \{ T , \sqrt { A B T } \} \Bigr ) .
$$

Here $m \geq { \frac { 1 } { 2 } }$ min $\{ B , T / ( 2 A ) \}$ . Combining the initial exploration bound with the drift bound, using max $\{ x , y \} \ge ( x + y ) / 2$ and the horizon cap, proves the result. The mixture is an oblivious distribution over admissible deterministic arrays. Its Bayesian lower bound implies the required worst-case lower bound for each policy. □

## B Variance-controlled statistical lower bounds

This appendix proves the statistical terms in Theorem 4.1; all supported environments satisfy the supplied budgets separately.

## B.1 A variance-controlled hard block

Lemma B.1 (Symmetric two-point hard block). Fix $h \geq A \geq 2 , s \in ( 0 , 1 / 2 ]$ , and

$$
0 < \epsilon \leq \operatorname* { m i n } \{ s / 2 , \ ( s / 4 ) { \sqrt { A / h } } \} .
$$

Choose an arm J uniformly from [A] before a block of h rounds. All rewards take values $1 / 2 - s$ and $1 / 2 + s$ . The probability of the upper value is $1 / 2 + \epsilon / ( 2 s )$ for $J ,$ and $1 / 2$ for every other arm. Every policy, with any initial private state independent of J, has Bayesian expected block regret at least $\epsilon h / 4$ . Each arm’s variance is at most $s ^ { 2 } ,$ , and at least one arm has variance exactly $s ^ { 2 }$ at every round.

Proof. Let $P _ { 0 }$ be the transcript law with every upper-value probability equal to $1 / 2$ , and $P _ { j }$ the law with arm j improved. If $N _ { j }$ counts pulls of $j ,$ the KL chain rule and the Bernoulli KL bound give

$$
\begin{array} { r } { \mathrm { K L } ( P _ { 0 } , P _ { j } ) = \mathbb { E } _ { 0 } [ N _ { j } ] \mathrm { k l } \left( \frac { 1 } { 2 } , \frac { 1 } { 2 } + \frac { \epsilon } { 2 s } \right) \le \frac { 2 \epsilon ^ { 2 } } { s ^ { 2 } } \mathbb { E } _ { 0 } [ N _ { j } ] . } \end{array}
$$

The second probability lies in $[ 1 / 2 , 3 / 4 ]$ . This justifies the displayed bound, for example by $\operatorname { k l } ( p , q ) \leq ( p - q ) ^ { 2 } / [ q ( 1 - q ) ]$ . Pinsker’s inequality, $0 \leq N _ { j } \leq h$ , and Jensen’s inequality yield

$$
\frac { 1 } { A } \sum _ { j } \mathbb { E } _ { j } [ N _ { j } ] \leq \frac { h } { A } + \frac { h \epsilon } { s A } \sum _ { j } \sqrt { \mathbb { E } _ { 0 } [ N _ { j } ] } \leq \frac { h } { A } + \frac { h \epsilon } { s } \sqrt { \frac { h } { A } } \leq \frac { 3 h } { 4 } .
$$

The expected regret under $P _ { j }$ is $\epsilon ( h - \mathbb { E } _ { j } [ N _ { j } ] )$ . Averaging proves the claim. The baseline variance is $s ^ { 2 } ;$ the improved variance is $s ^ { 2 } - \epsilon ^ { 2 }$ . Independent initial randomness can be included in the transcript without changing the KL calculation. □

## B.2 The statistical lower bound

Lower bound in Theorem $4 . 1 .$ The deterministic class is a subclass of $\mathcal { E } ( A , T ; Q , B )$ . Theorem 3.1 therefore supplies $\Omega ( \operatorname* { m i n } \{ T , A + D _ { B } \} )$ . For $A \leq T$ and $Q > 0 .$ , apply Lemma B.1 once with $h = T$ $s = { \sqrt { Q / T } }$ and $\epsilon = ( s / 1 6 ) \sqrt { A / T }$ . Its means are constant and its variance envelope is exactly $Q ,$ so it supplies $\sqrt { A Q } / 6 4$ . For $A > T$ , the deterministic initialization lower bound already controls the horizon cap.

It remains to establish the $C _ { B }$ term when it exceeds each of $A , S , D _ { B } ;$ otherwise one of the previous lower bounds sufices. If $T < 1 6 A$ , initialization alone is a constant fraction of the horizon, so assume $T \geq 1 6 A$ . Necessarily $B , Q > 0$ . Set

$$
x = \left( \frac { B ^ { 2 } T ^ { 2 } } { A Q } \right) ^ { 1 / 3 } , \qquad m = \lfloor x \rfloor , \qquad s = \sqrt { Q / T } , \qquad \epsilon = \frac { s } { 1 6 } \sqrt { \frac { A m } { T } } .
$$

From $C _ { B } \geq S$ we have $x \geq 1$ , so $x / 2 \leq m \leq x$ . If $B \leq 1$ , the condition $C _ { B } \geq D _ { B } = B \sqrt { A T }$ gives $Q \ge B ^ { 2 } \sqrt { A T }$ , so $x ^ { 3 } \leq ( T / A ) ^ { 3 / 2 }$ and $x \leq { \sqrt { T / A } } \leq T / ( 4 A )$ . If $B \geq 1$ , the same condition gives $Q ^ { 2 } \geq A B T$ and hence $x ^ { 3 } \le Q ^ { 3 } / A ^ { 3 }$ , so $x \leq Q / A \leq T / ( 4 A )$ . Thus, in both cases,

$$
1 \leq m \leq x \leq T / ( 4 A ) .
$$

Partition [T] into $m$ fixed blocks with lengths $h _ { i }$ difering by at most one. Every $h _ { i } \geq 4 A$ , and $h _ { i } \leq 2 T / m$ . Independently choose a fresh best arm in each block and use the laws in Lemma B.1 with the common s, ϵ above. Indeed, Am $/ T \le 1 / 4$ gives $\epsilon \leq s / 3 2 < s / 2$ , and

$$
\frac { \epsilon } { ( s / 4 ) \sqrt { A / h _ { i } } } = \frac { 1 } { 4 } \sqrt { \frac { m h _ { i } } { T } } \leq \frac { \sqrt { 2 } } { 4 } < 1 .
$$

This verifies both conditions of Lemma B.1 for every block, including unequal lengths.

The entire random array of reward laws is drawn before interaction. In every supported environment an unimproved arm exists because $A \geq 2$ , so the maximum variance in every round is $s ^ { 2 }$ and the envelope is exactly $T s ^ { 2 } = Q$ . At a block boundary the max-norm mean change is at most $\epsilon ,$ so its total variation is at most

$$
( m - 1 ) \epsilon \leq m \epsilon \leq \frac { \sqrt { A Q } } { 1 6 T } x ^ { 3 / 2 } = \frac { \sqrt { A Q } } { 1 6 T } \frac { B T } { \sqrt { A Q } } = B / 1 6 \leq B .
$$

Conditional on the past, the new best arm is independent of the learner’s state. Summing the block lower bounds therefore gives

$$
\mathbb { E } [ R _ { T } ] \ge \frac { \epsilon T } { 4 } = \frac { 1 } { 6 4 } \sqrt { A Q m } \ge \frac { 1 } { 6 4 \sqrt { 2 } } ( A Q B T ) ^ { 1 / 3 } .
$$

The Bayesian lower bound implies a worst-case lower bound for each policy, even if the fixed block boundaries are disclosed. Combining the three constructions and using the maximum to bound one quarter of the sum proves $c \Xi$ □

## C An observable variance-sensitive base learner

This appendix gives the noisy upper-bound ingredient used in the main text. Variance bonuses are motivated by UCB-V [AMS09] and empirical Bernstein bounds $[ \mathrm { M P 0 9 } ]$ ; the proofs below instead

Algorithm 4 Scheduled empirical-variance UCB (SEV-UCB)   
Input: Arms [A], horizon T, confidence $\delta ,$ restart period $\overline { { H \in \{ 1 , \ldots , T \} } }$   
1. Compute ℓ as in (8).   
2. for $s = 1 , 1 + H , 1 + 2 H , . . .$ with $s \leq T$ do   
3. Set $n _ { a } = 0 , m _ { a } = 0 , S _ { a } = 0$ for every arm $a .$   
4. for $t = s , \ldots ,$ , min $\{ s + H - 1 , T \}$ do   
5. if some arm has $n _ { a } = 0$ then   
6. Select the smallest-index unplayed arm $a _ { t } .$   
7. else   
8. Compute (40) using $\widehat { \mu } _ { t , a } = m _ { a } , \widehat { v } _ { t , a } = S _ { a } / n _ { a }$   
9. Select a maximizer of $U _ { t , a } ,$ breaking ties by index.   
10. end if   
11. Observe $X _ { t } ;$ set $d \gets X _ { t } - m _ { a _ { t } }$   
12. $n _ { a _ { t } } \gets n _ { a _ { t } } + 1 ; m _ { a _ { t } } \gets m _ { a _ { t } } + d / n _ { a _ { t } }$   
13. $S _ { a _ { t } } \gets S _ { a _ { t } } + d ( X _ { t } - m _ { a _ { t } } ) \mathop { }$ Welford update   
14. end for   
15. end for

use martingale concentration [Fre75]. Fix a contiguous epoch $I = [ s , e ]$ . At the beginning of round $t ,$ let $n _ { t , a }$ be the number of observations of arm a in $[ s , t - 1 ]$ . For $n _ { t , a } \geq 1$ , let

$$
\widehat { \mu } _ { t , a } = \frac { 1 } { n _ { t , a } } \sum _ { u = s } ^ { t - 1 } { \mathbf 1 } \{ a _ { u } = a \} X _ { u } ,\tag{38}
$$

$$
\widehat v _ { t , a } = \frac { 1 } { n _ { t , a } } \sum _ { u = s } ^ { t - 1 } \mathbf 1 \{ a _ { u } = a \} ( X _ { u } - \widehat \mu _ { t , a } ) ^ { 2 } ,\tag{39}
$$

$$
{ r _ { t , a } } = 1 6 \sqrt { \frac { \widehat { v } _ { t , a } \ell } { n _ { t , a } } } + \frac { 2 5 6 \ell } { n _ { t , a } } , \qquad U _ { t , a } = \operatorname* { m i n } \{ 1 , \widehat { \mu } _ { t , a } + { r _ { t , a } } \} .\tag{40}
$$

These constants simplify a self-contained proof and are not intended as optimized practical tuning. When $n _ { t , a } = 0$ , set $U _ { t , a } = 1$ . An unplayed arm is selected before taking a maximizer among played arms. All scores use pre-round information, and $n = 1$ is well defined.

The learner uses $O ( A )$ memory and $O ( A )$ time per round. It requires no means, variances, gaps, or change locations. Its schedule parameter H remains an input; it is not a fully prior-free dynamic-regret policy.

## C.1 A simultaneous martingale event

Let $\varepsilon _ { t } = X _ { t } - \mu _ { t , a _ { t } }$ . For every arm a and interval $J = [ u , v ]$ , define

$$
S _ { J , a } = \sum _ { t \in J } \mathbf { 1 } \{ a _ { t } = a \} \varepsilon _ { t } , \quad Q _ { J , a } = \sum _ { t \in J } \mathbf { 1 } \{ a _ { t } = a \} \varepsilon _ { t } ^ { 2 } , \quad B _ { J , a } = \sum _ { t \in J } \mathbf { 1 } \{ a _ { t } = a \} \sigma _ { t , a } ^ { 2 } .
$$

These are analytical quantities, not algorithmic inputs.

Lemma C.1 (Simultaneous concentration). For any fixed nonanticipating policy, with probability at least $1 - \delta _ { i }$ simultaneously for every interval J and arm $^ { a , }$

$$
| S _ { J , a } | \leq 4 \sqrt { B _ { J , a } \ell } + 4 \ell ,\tag{41}
$$

$$
Q _ { J , a } \leq 2 B _ { J , a } + \ell , \qquad B _ { J , a } \leq 2 Q _ { J , a } + 2 \ell .\tag{42}
$$

On the same event,

$$
\left| \sum _ { t = 1 } ^ { T } \varepsilon _ { t } \right| \leq 4 \sqrt { \Lambda _ { T } \ell } + 4 \ell .\tag{43}
$$

Proof. Fix a deterministic J and a. The centered selected observations form a martingale-diference sequence bounded by 1, with predictable quadratic variation $B _ { J , a } \leq T$ . Freedman’s inequality, followed by peeling $B _ { J , a }$ into [0, 1] and dyadic bins up to $T _ { i }$ gives (41) with failure probability at most $2 ( 1 + \lceil \log _ { 2 } T \rceil ) e ^ { - \ell }$ . The constants also cover $B _ { J , a } \leq 1$ since $\ell > 1$

For completeness, if $Y \in [ 0 , 1 ]$ and $m = \mathbb { E } [ Y \mid { \mathcal { F } } ]$ , convexity of the exponential yields

$$
\mathbb { E } [ e ^ { Y } \mid \mathcal { F } ] \leq e ^ { ( e - 1 ) m } , \qquad \mathbb { E } [ e ^ { - Y } \mid \mathcal { F } ] \leq e ^ { - ( 1 - e ^ { - 1 } ) m } .
$$

Apply the resulting exponential supermartingales to $Y _ { t } = \mathbf { 1 } \{ a _ { t } = a \} \varepsilon _ { t } ^ { 2 }$ . Markov’s inequality gives $Q _ { J , a } \leq ( e - 1 ) B _ { J , a } + \ell \leq 2 B _ { J , a } + \ell$ and $( 1 - e ^ { - 1 } ) B _ { J , a } \leq Q _ { J , a } + \ell .$ , implying (42), with total failure probability at most 2e<sup>−ℓ</sup>. The same Freedman argument for all selected observations gives (43).

There are at most $A T ^ { 2 }$ arm-interval pairs. Taking a union bound, using $1 + \lceil \log _ { 2 } T \rceil \leq T$ for $T \geq 2$ , and substituting (8) bounds the total failure probability by $\delta .$ □

The event is uniform over interval endpoints for one executed policy. It is not an event simultaneously covering all possible policies. In particular, a data-dependent choice among unexecuted policies does not follow from this lemma.

## C.2 Confidence under drift

Lemma C.2 (Empirical confidence with an explicit drift bias). On the event in Lemma $C . 1 ,$ for every epoch I and pre-round empirical statistic in that epoch with $n _ { t , a } \geq 1$ ，

$$
\begin{array} { r } { | \widehat { \mu } _ { t , a } - \mu _ { t , a } | \leq r _ { t , a } + 2 \Delta _ { I } . } \end{array}\tag{44}
$$

Proof. Write the $n = n _ { t , a }$ observed rewards as $Y _ { j } = \theta _ { j } + \eta _ { j }$ , where $\theta _ { j }$ is the mean of arm a at its $j { \mathrm { - t h } }$ sampled time. Let $\begin{array} { r } { \bar { \theta } = { { n } ^ { - 1 } } \sum _ { j } \theta _ { j } , \bar { Y } = \widehat { \mu } _ { t , a } , { \cal S } = \sum _ { j } \tilde { \eta } _ { j } , \bar { Q } = \sum _ { j } \eta _ { j } ^ { 2 } } \end{array}$ , and $\begin{array} { r } { B = \sum _ { j } \operatorname { V a r } ( \eta _ { j } \mid \operatorname { p a s t } ) } \end{array}$ . Set $v = \Delta _ { I }$ and $\begin{array} { r } { F = \sum _ { j } ( Y _ { j } - \bar { Y } ) ^ { 2 } = n \widehat { v } _ { t , a } } \end{array}$ . All the means $\theta _ { j }$ and the current mean lie within distance v of one another. By expanding around Y<sup>¯</sup> and using $\begin{array} { r } { \sum _ { j } ( \bar { \theta } - \theta _ { j } ) = 0 } \end{array}$ 2

$$
Q \leq 2 F + \frac { 2 S ^ { 2 } } { n } + 2 n v ^ { 2 } .\tag{45}
$$

If $n < 2 5 6 \ell$ , the radius is at least 1, so the result is trivial. Otherwise, (41) gives $S ^ { 2 } \leq 3 2 B \ell + 3 2 \ell ^ { 2 }$ Together with $B \leq 2 Q + 2 \ell$ and (45), this yields

$$
B \leq 4 F + 4 n v ^ { 2 } + \frac { 1 2 8 B \ell } { n } + \frac { 1 2 8 \ell ^ { 2 } } { n } + 2 \ell \leq 4 F + 4 n v ^ { 2 } + \frac { 1 } { 2 } B + \frac { 5 } { 2 } \ell .
$$

Hence $B \leq 8 F + 8 n v ^ { 2 } + 5 \ell$ . Returning to the noise bound,

$$
\frac { | S | } { n } \leq 1 2 \sqrt { \frac { \widehat { v } _ { t , a } \ell } { n } } + 1 2 v \sqrt { \frac { \ell } { n } } + \frac { 1 3 \ell } { n } \leq r _ { t , a } + v ,
$$

because $n \geq 2 5 6 \ell$ . Finally, $| \bar { \theta } - \mu _ { t , a } | \leq v$ proves (44).

The drift term in (44) is only used in the proof. It does not enter the implemented radius. This is diferent from asserting that the empirical mean estimates the current mean without bias.

## C.3 A radius bound independent of the regret analysis

Lemma C.3 (Radius control under drift). On the event in Lemma C.1, take any executed epoch and any arm with $n \geq 1$ past observations. Let $B _ { a , n }$ be the sum of its conditional reward variances at those sampled times. If their means have diameter at most $v ,$ the radius in (40) obeys

$$
r _ { a , n } \leq \frac { 3 2 \sqrt { B _ { a , n } \ell } } { n } + \frac { 2 8 0 \ell } { n } + 2 4 v \sqrt { \frac { \ell } { n } } .\tag{46}
$$

This holds for every nonanticipating sampling rule, irrespective of whether it maximizes a UCB score. In calibration one may take $v = B ,$ ; in an epoch I one may take $v = \Delta _ { I }$

Proof. Write the observations as $Y _ { j } = \theta _ { j } + \eta _ { j }$ and set $\bar { \theta } = n ^ { - 1 } \textstyle \sum _ { j } \theta _ { j }$ and $\begin{array} { r } { Q _ { n } = \sum _ { j } \eta _ { j } ^ { 2 } } \end{array}$ . The empirical mean minimizes squared deviations, so

$$
n \widehat { v } _ { a , n } \leq \sum _ { j } ( Y _ { j } - \bar { \theta } ) ^ { 2 } \leq 2 Q _ { n } + 2 \sum _ { j } ( \theta _ { j } - \bar { \theta } ) ^ { 2 } \leq 2 Q _ { n } + 2 n v ^ { 2 } .
$$

Lemma C.1 gives $Q _ { n } \leq 2 B _ { a , n } + \ell .$ . Hence $\widehat { v } _ { a , n } \leq 4 B _ { a , n } / n + 2 \ell / n + 2 v ^ { 2 }$ . Insert this into (40), use $\sqrt { x + y + z } \leq \sqrt { x } + \sqrt { y } + \sqrt { z }$ , and note that $2 5 6 + 1 6 { \sqrt { 2 } } \leq 2 8 0$ and $1 6 \sqrt { 2 } \leq 2 4$ . This proves (46) using concentration alone. □

## C.4 Interval regret

For an epoch I, let $\begin{array} { r } { \Lambda _ { I } = \sum _ { t \in I } \sigma _ { t , a _ { t } } ^ { 2 } } \end{array}$

Theorem C.4 (Variance-sensitive interval guarantee). On the event in Lemma C.1, a fresh run of the base learner on any executed epoch I satisfies

$$
R _ { I } \leq 1 0 0 0 D _ { T } ( \sqrt { A \Lambda _ { I } } + A + | I | \Delta _ { I } ) .\tag{47}
$$

The same statement holds for epochs produced by predictable restart decisions, provided the learner is reset at every epoch start.

Proof. Let $h = | I |$ and $v = \Delta _ { I }$ . There are at most A initial pulls of unplayed arms, each with regret at most 1. After initialization, Lemma C.2 and clipping imply

$$
U _ { t , a } \geq \mu _ { t , a } - 2 v , \qquad U _ { t , a } \leq \mu _ { t , a } + 2 r _ { t , a } + 2 v .
$$

Since the policy maximizes the scores, its instantaneous pseudo-regret is at most $2 r _ { t , a _ { t } } + 4 v$

Lemma C.3 applies to these empirical statistics. Let $N _ { a }$ be the final epoch pull count and $B _ { a }$ its final accumulated variance. Summing over $n = 1 , \ldots , N _ { a } - 1$ , using $B _ { a , n } \leq B _ { a }$ , the harmonic-sum bound, and $\begin{array} { r } { \sum _ { n = 1 } ^ { N _ { a } } n ^ { - 1 / 2 } \leq 2 \sqrt { N _ { a } } , } \end{array}$ gives

$$
\begin{array} { r l r } {  { \sum _ { t : n _ { t } , a _ { t } \geq 1 } r _ { t , a _ { t } } \leq 3 2 ( 1 + \log h ) \sqrt { \ell } \sum _ { a } \sqrt { B _ { a } } + 2 8 0 A \ell ( 1 + \log h ) + 4 8 v \sqrt { \ell } \sum _ { a } \sqrt { N _ { a } } } } \\ & { } & \\ & { } & { \leq 3 2 ( 1 + \log h ) \sqrt { A \Lambda _ { I } \ell } + 2 8 0 A \ell ( 1 + \log h ) + 4 8 v \sqrt { A h \ell } . } \end{array}
$$

If $h < A$ , every round is an initialization round and $R _ { I } \leq A$ . Otherwise ${ \sqrt { A h } } \leq h$ . Combining the displayed estimates, the initialization cost, and 4hv proves (47), since $\ell \geq 1$ and $1 + \log h \leq 1 + \log T$ The simultaneous interval event covers the realized endpoints of predictably generated epochs as well. □

## D Scheduled restarts and oracle tuning

Theorem D.1 (Partition guarantee). For an executed partition P into M epochs on which the base learner is reset, with probability at least $1 - \delta$ ，

$$
R _ { T } \leq C D _ { T } \left( \sqrt { A M \Lambda _ { T } } + A M + \sum _ { I \in \mathcal { P } } | I | \Delta _ { I } \right) ,\tag{48}
$$

where C is a universal constant. For the fixed-period policy in Algorithm 4, $M = \lceil T / H \rceil$ and

$$
R _ { T } \leq C D _ { T } \left( \sqrt { A [ T / H ] \Lambda _ { T } } + A [ T / H ] + H \Delta \right) .\tag{49}
$$

For realized regret, add $4 \sqrt { \Lambda _ { T } \ell } + 4 \ell$ to either right-hand side. Pseudo-regret may always be capped by T.

Proof. Sum (47) over epochs. Cauchy–Schwarz gives $\begin{array} { r } { \sum _ { I } \sqrt { A \Lambda _ { I } } \le \sqrt { A M \Lambda _ { T } } } \end{array}$ . For the fixed schedule, every epoch has length at most H; use (6) to bound the drift contribution by $H \Delta$ . Equation (43) and (5) give the realized regret statement. □

The partition guarantee is an analysis tool for a partition that was actually executed. It does not assert that a learner can retrospectively achieve the infimum over all partitions using the corresponding counterfactual variances.

Corollary D.2 (Oracle variance–variation benchmark). Suppose deterministic budgets $B , Q$ are supplied, with $\Delta \le B$ and $V _ { T } \leq Q$ , where $0 \leq B \leq T$ and $0 \leq Q \leq T / 4$ . If $B = 0$ , set $H = T$ . If $B > 0$ , set

$$
H = \operatorname* { m i n } \left\{ T , \left\lceil \operatorname* { m a x } \left\{ 1 , \left( \frac { A Q T } { B ^ { 2 } } \right) ^ { 1 / 3 } , \sqrt { \frac { A T } { B } } \right\} \right\rceil \right\} .\tag{50}
$$

With probability at least $1 - \delta$ , this policy satisfies

$$
R _ { T } \leq C D _ { T } \left( \sqrt { A Q } + A + ( A Q B T ) ^ { 1 / 3 } + \sqrt { A B T } \right) .\tag{51}
$$

No per-round variance is supplied to the base learner.

Proof. Substitute $\Lambda _ { T } \leq Q$ and $\Delta \le B$ into (49), and use $\lceil T / H \rceil \leq T / H + 1$ to reduce the bound, up to a universal constant, to

$$
{ \sqrt { A Q } } + A + { \sqrt { { \frac { A Q T } { H } } } } + { \frac { A T } { H } } + H B .
$$

When $B \ = \ 0$ , the assertion follows with $H \ = \ T$ Otherwise write $h _ { v } ~ = ~ ( A Q T / B ^ { 2 } ) ^ { 1 / 3 }$ and $h _ { m } = \sqrt { A T / B }$ . If neither exceeds T, the chosen H is at least $h _ { v } , h _ { m } .$ , and at most $h _ { v } + h _ { m } + 2$ The two estimation terms are therefore at most $( A Q B T ) ^ { 1 / 3 }$ and ${ \sqrt { A B T } } .$ , respectively, and $H B \leq$ $( A Q B T ) ^ { 1 / 3 } + \sqrt { A B T } + 2 B$ . Since $B \leq T$ and $A \geq 2 , B \leq { \sqrt { A B T } }$ . If $h _ { v } > T$ , then $B T < { \sqrt { A Q } } ;$ if $h _ { m } > T$ , then $B T < A$ . In either case setting $H = T$ is covered by the two stationary terms. This also handles $Q = 0$ and integer rounding. □

Limiting regimes. If $B = 0$ , the bound is $\widetilde { O } ( \sqrt { A Q } + A )$ . If $Q = 0 ,$ , it is $\widetilde { O } ( A + \sqrt { A B T } )$ . For $Q = \Theta ( T )$ it contains the usual variation term $A ^ { 1 / 3 } B ^ { 1 / 3 } T ^ { 2 / 3 }$ together with the stationary term. In the sublinear regime $A B \leq T$ , the monitoring term $\sqrt { A B T }$ is no larger than this variation term, up to constants. There is no contradiction with Theorem 5.1: here the learner is told the budgets, and can change its schedule when the supplied variation budget changes.

## E Aggregation under explicit epoch charges

The following lemma isolates the purely algebraic part of an epoch analysis under explicit local variation charges for completed epochs.

Lemma E.1 (Aggregation from completed-epoch charges). Let $h _ { i } > 0 , \lambda _ { i } \geq 0$ , and $v _ { i } \geq 0$ for completed epochs $i = 1 , \ldots , m$ , and suppose

$$
\sum _ { i } h _ { i } \le T , \quad \sum _ { i } \lambda _ { i } \le \Lambda , \quad \sum _ { i } v _ { i } \le \Delta , \qquad h _ { i } v _ { i } \ge c ( \sqrt { A \lambda _ { i } } + A ) ,
$$

for $c > 0$ . Then

$$
\sum _ { i = 1 } ^ { m } ( \sqrt { A \lambda _ { i } } + A ) \leq c ^ { - 1 / 3 } ( A \Lambda T \Delta ) ^ { 1 / 3 } + c ^ { - 1 / 2 } \sqrt { A T \Delta } .\tag{52}
$$

Proof. For the variance part,

$$
\begin{array} { r } { \sqrt { A \lambda _ { i } } = ( A \lambda _ { i } ) ^ { 1 / 3 } ( \sqrt { A \lambda _ { i } } ) ^ { 1 / 3 } \leq c ^ { - 1 / 3 } ( A \lambda _ { i } ) ^ { 1 / 3 } h _ { i } ^ { 1 / 3 } v _ { i } ^ { 1 / 3 } . } \end{array}
$$

Hölder’s inequality with three exponents equal to 3 bounds its sum by $c ^ { - 1 / 3 } ( A \Lambda T \Delta ) ^ { 1 / 3 }$ . For the initialization part, $h _ { i } v _ { i } \geq c A$ implies $A \le \sqrt { A / c } \sqrt { h _ { i } v _ { i } }$ . Cauchy–Schwarz then gives Am $\leq$ $c ^ { - 1 / 2 } { \sqrt { A T \Delta } }$ □

Remark E.2 (A certificate is not a causal construction). Lemma E.1 proves no statistical claim. It concerns completed epochs only, and says nothing about the last unfinished epoch. An algorithmic theorem must establish the stated charges from observed data, control drift regret as well, and bound the unfinished epoch. If all these steps implied a polylogarithmic stationary zero-variance cost and $\widetilde { O } ( \sqrt { A T } )$ one-change cost for the same prior-free policy, Corollary 5.3 would rule them out. Therefore a missing monitoring term cannot be repaired solely by this aggregation calculation.

## F Stopped calibration and unconditional expected regret

Fix an environment $\nu \in \mathcal { E } ( A , T ; Q , B )$ . Let $\mathcal { H } _ { s }$ contain the actions, observed rewards and private randomness actually used through round s. The calibration stopping time τ is adapted to this filtration. Unused randomness, including the seeds of future hybrid copies, is independent and is not included in $\mathcal { H } _ { \tau }$

A prefix-measurable good event. Define G by requiring (41) and (42) only for arm–interval statistics contained in [1, τ]. For a fixed environment these inequalities are measurable from the stopped history, even though their unknown means and variances make G an analytical event rather than an algorithmic test. Thus $G \in { \mathcal { H } } _ { \tau }$ . The full simultaneous event in Lemma C.1 implies G, so $\mathbb { P } _ { \nu } ( G ^ { c } ) \leq \delta$ . We use this implication only to bound the probability of $G ^ { c }$ ; we never condition the future learner on the full simultaneous event. The proof of Lemma 4.2 uses only these prefix inequalities. On G, its regret bound holds, and, if $\tau < T$ , both certificate properties hold: every future optimal arm is retained and every surviving arm satisfies (18).

Future observations after stopping. For each $s < T$ , the event $G \cap \{ \tau = s \}$ belongs to $\mathcal { H } _ { s }$ Independence across rounds implies that future reward vectors retain their original laws conditional on this stopped history, for almost every such history. No independence between arms within a round is needed. In particular, for every retained arm and $t > s$ , the original conditional mean and variance are still $\mu _ { t , a } , \sigma _ { t , a } ^ { 2 }$ . The clipping interval is fixed by that history. Its distance certificate therefore gives the conditional versions of both inequalities in (23). Fresh future seeds then justify applying the static hybrid guarantee on each subsequent block. This proves the conditional assertion in Lemma 4.3 for the histories used by the upper bound.

The static theorem and the clipping conversion. For completeness, the proof of Theorem 3 in [Ito21b], printed page 12, gives, on $K$ arms and h rounds, the explicit fixed-arm regret bound

$$
6 \sqrt { K \log h \mathbb { E } \biggl [ \sum _ { t = 1 } ^ { h } ( \ell _ { t , a _ { t } } - u _ { t , a _ { t } } ) ^ { 2 } + K + 8 \sum _ { t = 1 } ^ { h - 1 } \| u _ { t + 1 } - u _ { t } \| _ { 1 } \biggr ] } + 3 K \log h .
$$

The reference sequence is arbitrary and is not supplied to the algorithm (Theorem 3, equation (6), footnote 2). Thus it need not equal the expectation of the observed loss. If $\mathcal { Z } _ { h } ( u )$ is as in (16), the quantity inside the expectation is at most $8 \mathcal { Z } _ { h } ( u ) + K$ . Since log $h \geq 1$ for $h \geq 3 , { \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ gives (16), for example with $C _ { 0 } = 1 8$ . The loss feedback is the selected coordinate only, as in the cited multi-armed bandit model. On a retained set of size $K \geq 2$ we have $K \leq A$ and invoke the algorithm only when $h \geq \operatorname* { m a x } \{ A , 3 \}$ ; one survivor and shorter blocks are handled directly as stated in Lemma 4.3.

Here is the conversion without suppressing the clipping biases. Fix a good calibration history and a subsequent block-start history; write $\mathbb { E } _ { s }$ for expectation conditional on the latter. Set $\bar { Y } _ { t , a } = \mathbb { E } [ Y _ { t , a } \mid \mathcal { H } _ { s } ]$ and $b _ { t , a } = { \bar { Y } } _ { t , a } - \mu _ { t , a }$ . Future reward vectors are independent of the history before their round. Therefore $| b _ { t , a } | \leq q _ { t } / w$ , and a nonanticipating action satisfies $\mathbb { E } _ { s } [ Y _ { t , a _ { t } } ] = \mathbb { E } _ { s } [ \bar { Y } _ { t , a _ { t } } ]$ . For the fixed retained comparator $a ^ { \circ }$ optimal at the start of this block, the exact identity is

$$
\begin{array} { r l r } {  { \mathbb { E } _ { s } \sum _ { t \in I } ( \mu _ { t , a ^ { \circ } } - \mu _ { t , a _ { t } } ) = L \mathbb { E } _ { s } \sum _ { t \in I } ( \ell _ { t , a _ { t } } ^ { \prime } - \ell _ { t , a ^ { \circ } } ^ { \prime } ) + \mathbb { E } _ { s } \sum _ { t \in I } ( b _ { t , a _ { t } } - b _ { t , a ^ { \circ } } ) } } \\ & { } & { \leq L \mathbb { E } _ { s } \sum _ { t \in I } ( \ell _ { t , a _ { t } } ^ { \prime } - \ell _ { t , a ^ { \circ } } ^ { \prime } ) + \displaystyle \frac { 2 } { w } \sum _ { t \in I } q _ { t } . \qquad } \end{array}
$$

Biases can have either sign; the bound uses their absolute values. Comparator losses are analytical coordinates of the reward vector, not additional observations given to the learner. The reference $u _ { t , a } ^ { \prime } = ( z + 5 w - \mu _ { t , a } ) / L$ lies in $[ 1 / 1 8 , 1 7 / 1 8 ]$ , and

$$
\mathbb { E } _ { s } \sum _ { t \in I } ( \ell _ { t , a _ { t } } ^ { \prime } - u _ { t , a _ { t } } ^ { \prime } ) ^ { 2 } \leq \frac { \mathbb { E } _ { s } [ \Lambda _ { I } ] } { L ^ { 2 } } , \qquad \sum _ { t , t + 1 \in I } \| u _ { t + 1 } ^ { \prime } - u _ { t } ^ { \prime } \| _ { 1 } \leq \frac { A \Delta _ { I } } { L } .
$$

Substitution in the explicit static theorem gives the three terms $L A , \sqrt { A \mathbb { E } _ { s } [ \Lambda _ { I } ] }$ and $A \sqrt { L \Delta _ { I } } .$ with at most one logarithmic factor, as used in Lemma 4.3. The additional dynamic-comparator cost is at most $2 h \Delta _ { I }$ . This establishes the conditional bridge before summing over blocks.

The actual post-calibration partition. Fix a stopped history with $\tau < T$ . Consecutive blocks start at $\tau + 1$ and have length $H$ , except possibly the last. Consequently,

$$
N _ { \tau } = T - \tau , \qquad M _ { \tau } = \left\lceil \frac { N _ { \tau } } { H } \right\rceil \leq M : = \left\lceil \frac { T } { H } \right\rceil .\tag{53}
$$

More explicitly, for $i = 1 , \ldots , M _ { \tau }$ , let $I _ { i } = [ \tau + ( i - 1 ) H + 1$ , min $\{ \tau + i H , T \} ]$ and $h _ { i } = | I _ { i } |$ . Set $q _ { t } = \operatorname* { m a x } _ { a } \sigma _ { t , a } ^ { 2 }$ and define the analytical tail quantities

$$
Q _ { \tau } = \sum _ { t = \tau + 1 } ^ { T } q _ { t } \leq Q , \qquad B _ { \tau } = \sum _ { t = \tau + 2 } ^ { T } d _ { t } \leq B .
$$

Conditional on the stopped history in a fixed environment, these quantities and the partition are fixed. They are not new inputs to the learner. Internal variation excludes changes at block boundaries, so $\begin{array} { r } { \sum _ { i } \Delta _ { I _ { i } } \leq B _ { \tau } } \end{array}$

Let $\lambda _ { i } = \mathbb { E } [ \Lambda _ { I _ { i } } \mid \mathcal { H } _ { \tau } ]$ . The selected variance is bounded pathwise by the envelope on the same rounds, giving $\Sigma _ { i } \lambda _ { i } \leq Q _ { \tau }$ . Conditioning the per-block bound at its start and then on $\mathcal { H } _ { \tau }$ , with Jensen’s inequality for the square root, permits its variance term to be bounded by $\sqrt { A \lambda _ { i } }$ . Thus the three block charges satisfy

$$
\begin{array} { r } { \sum _ { i } \sqrt { A \lambda _ { i } } \le \sqrt { A M _ { \tau } \sum _ { i } \lambda _ { i } } \le \sqrt { A M _ { \tau } Q _ { \tau } } , } \\ { \sum _ { i } w A = w A M _ { \tau } , \qquad \sum _ { i } h _ { i } \Delta _ { I _ { i } } \le H B _ { \tau } . } \end{array}\tag{54}
$$

The factor A in the square root comes from the static guarantee; Cauchy–Schwarz is applied only over blocks and introduces $\sqrt { M _ { \tau } }$ , not another arm factor. For a short block with $h _ { i } < \operatorname* { m a x } \{ A , 3 \}$ $h _ { i } \leq A$ and its direct regret bound $1 6 w h i \leq 1 8 w A$ uses the same single block charge. The two-round case $A = 2$ is included. The clipping-bias charges sum to $2 Q _ { \tau } / w$ over disjoint rounds; they are not charged once per block.

If $\tau = T$ , take ${ \cal M } _ { \tau } = { \cal Q } _ { \tau } = { \cal B } _ { \tau } = 0$ , with an empty partition and zero post-calibration regret. This convention covers the zero-tail endpoint without instantiating a learner.

The tower-property calculation. Write $R _ { T } = R _ { \mathrm { c a l } } + R _ { \mathrm { p o s t } }$ , setting $R _ { \mathrm { p o s t } } = 0 { \mathrm { ~ i f ~ } } \tau = T$ . The calibration regret is $\mathcal { H } _ { \tau } .$ -measurable. Let $K _ { \mathrm { c a l } } = C ( A \ell + { \sqrt { A Q \ell } } )$ . The preceding conditional sum and the explicit static ingredient give the history-dependent bound

$$
\nonumber { \cal K } _ { \tau } = { \cal C } ^ { \prime } \log ( e T ) \big ( \sqrt { A M _ { \tau } Q _ { \tau } } + w A M _ { \tau } + H B _ { \tau } \big ) + 2 Q _ { \tau } / w .
$$

By (53), $Q _ { \tau } \leq Q$ and $B _ { \tau } \leq B$ , it is bounded by the uniform deterministic quantity

$$
K _ { \mathrm { p o s t } } = C ^ { \prime } \log ( e T ) \big ( \sqrt { A M Q } + w A M + H B \big ) + 2 Q / w
$$

with the following explicit constant convention. Let $C _ { \mathrm { l o n g } }$ be any universal coeficient for which the long-block bound is at most

$$
C _ { \mathrm { l o n g } } \log ( e T ) \big ( \sqrt { A \mathbb { E } [ \Lambda _ { I } \mid \mathcal { H } _ { \operatorname* { m i n } I - 1 } ] } + w A + h \Delta _ { I } \big ) + \frac { 2 } { w } \sum _ { t \in I } q _ { t } .
$$

Such a coeficient follows from (16) and the rescaling in Lemma 4.3. Fix once and for all $C ^ { \prime } \geq$ $\operatorname* { m a x } \{ C _ { \mathrm { l o n g } } , 1 8 \}$ . Since $\log ( e T ) \geq 1$ , this same coeficient covers every short-block charge 18wA as well; it is independent of the budgets, the stopped history, and the block count. Thus, on $G ,$ $R _ { \mathrm { c a l } } \leq K _ { \mathrm { c a l } }$ and $\mathbb { E } [ R _ { \mathrm { p o s t } } \mid \mathcal { H } _ { \tau } ] \le K _ { \tau } \le K _ { \mathrm { p o s t } }$ . Using $0 \leq R _ { T } \leq T$ and $G \in { \mathcal { H } } _ { \tau }$ <sub>τ</sub>,

$$
\begin{array} { r l } & { \mathbb { E } [ R _ { T } ] = \mathbb { E } [ \mathbf { 1 } _ { G } R _ { \mathrm { c a l } } ] + \mathbb { E } [ \mathbf { 1 } _ { G } \mathbb { E } [ R _ { \mathrm { p o s t } } \mid \mathcal { H } _ { \tau } ] ] + \mathbb { E } [ \mathbf { 1 } _ { G ^ { c } } R _ { T } ] } \\ & { \qquad \leq K _ { \mathrm { c a l } } + \mathbb { E } [ \mathbf { 1 } _ { G } \mathcal { K } _ { \tau } ] + T \delta } \\ & { \qquad \leq K _ { \mathrm { c a l } } + K _ { \mathrm { p o s t } } + T \delta . } \end{array}\tag{55}
$$

This also covers horizon-truncated calibration. Taking $\delta = T ^ { - 4 } \mathrm { g i v e s } T \delta = T ^ { - 3 }$ , with no assumption that G is independent of the calibration output.

Proof dependencies. Lemma C.1 proves concentration. Lemmas C.2 and C.3 each follow directly from it. These two lemmas prove calibration; the cited static hybrid theorem and the two-part certificate prove clipped restarting. Equation (55) then combines them. The separate interval-regret theorem, Theorem C.4, is not used to establish calibration or its radius estimate.

## G Comparison with prior results

Table 3 summarizes the model conventions and statements underlying the comparisons in Section 7.   
The cited versions are specified where theorem numbering difers.

Table 3: Models and guarantees in closely related work.
<table><tr><td>Source and version</td><td>Relevant statement</td><td>Relation to the present bounds</td></tr><tr><td>[WHL16], NeurIPS 2016</td><td>Equation (1), Theorems plement</td><td>Variance-variation dynamic regret and noise-free 3.1 and 4.3; original sup- monitoring are prior themes. Full-vector variance and initial-inclusive variation differ from Q, B.</td></tr><tr><td>[Ito21b], COLT 2021</td><td>its explicit proof</td><td>Theorem 3, Eq. (6), and The arbitrary-reference static algorithm is prior work; ordinary restarting is its direct consequence. Our added steps are calibration and clipping-bias accounting.</td></tr><tr><td>supplement</td><td>pendix D</td><td>[KA16], NeurIPS 2016 Theorem A.1 and Ap- Prior-free variation tracking and a noiseless con- struction already exist; the deterministic problem itself is not a new formulation.</td></tr><tr><td>[GM08, GM11]; [LS20]</td><td>rem 31.2</td><td>Preprint Theorem 13, Re- The inverse stationary/change-regret obstruction is mark 15; textbook Theo- classical. Our matching frontier is restricted to the disclosed deterministic family.</td></tr><tr><td>[WXC+25], AISTATS 2025</td><td>lary 5.4; Remark 5.6</td><td>Theorems 4.1, 5.2; Corol- Variance-sensitive linear results include a finite-arm comparison; the stated theorem is not the same all-budget finite-arm formula.</td></tr><tr><td>[QW26], arXiv v2, 17 Theorem 4 February 2026</td><td></td><td>Simultaneous static/dynamic bounds use a supplied switch limit and allow a stationary √AT scale. This differs from our small-stationary-regret constraint.</td></tr><tr><td>2026</td><td>4.2</td><td>[GIZ26], arXiv v2, 3 June Section 2; Theorems 4.1– Direct stream observations and squared mean- prediction error differ from selected-arm feedback and best-arm pseudo-regret. The upper bound also</td></tr></table>

## References

[AGO19] Peter Auer, Pratik Gajane, and Ronald Ortner. Adaptively tracking the best bandit arm with an unknown number of distribution changes. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 138–158. PMLR, 2019.

[AMS09] Jean-Yves Audibert, Rémi Munos, and Csaba Szepesvári. Exploration–exploitation tradeof using variance estimates in multi-armed bandits. Theoretical Computer Science, 410(19):1876–1902, 2009.

[AYGL23] Yasin Abbasi-Yadkori, András György, and Nevena Lazić. A new look at dynamic regret for non-stationary stochastic bandits. Journal of Machine Learning Research, 24(288):1–37, 2023.

[BCL18] Sébastien Bubeck, Michael Cohen, and Yuanzhi Li. Sparsity, variance and curvature in multi-armed bandits. In Proceedings of Algorithmic Learning Theory, volume 83 of Proceedings of Machine Learning Research, pages 111–127. PMLR, 2018.

[BGZ14] Omar Besbes, Yonatan Gur, and Assaf Zeevi. Stochastic multi-armed-bandit problem with non-stationary rewards. In Advances in Neural Information Processing Systems, volume 27, 2014.

[BGZ19] Omar Besbes, Yonatan Gur, and Assaf Zeevi. Optimal exploration–exploitation in a multi-armed bandit problem with non-stationary rewards. Stochastic Systems, 9(4):319– 337, 2019.

[BKMS22] Lilian Besson, Emilie Kaufmann, Odalric-Ambrym Maillard, and Julien Seznec. Eficient change-point detection for tackling piecewise-stationary bandits. Journal of Machine Learning Research, 23(77):1–40, 2022.

[BLLW19] Sébastien Bubeck, Yuanzhi Li, Haipeng Luo, and Chen-Yu Wei. Improved path-length regret bounds for bandits. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 508–528. PMLR, 2019.

[CLLW19] Yifang Chen, Chung-Wei Lee, Haipeng Luo, and Chen-Yu Wei. A new algorithm for non-stationary contextual bandits: Eficient, optimal and parameter-free. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 696–726. PMLR, 2019.

[CWKX19] Yang Cao, Zheng Wen, Branislav Kveton, and Yao Xie. Nearly optimal adaptive procedure with change detection for piecewise-stationary bandit. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 418–427. PMLR, 2019.

[CYZ25] Ningyuan Chen, Shuoguang Yang, and Hailun Zhang. Bridging adversarial and nonstationary multi-armed bandit. Production and Operations Management, 34(8):2218–2231, 2025. Technical comparison uses arXiv:2201.01628v3.

[DHS20] Qin Ding, Cho-Jui Hsieh, and James Sharpnack. Multiscale non-stationary stochastic bandits, 2020.

[Fre75] David A. Freedman. On tail probabilities for martingales. The Annals of Probability, 3(1):100–118, 1975.

[GBGM26] Gianmarco Genalti, Sujay Bhatt, Nicola Gatti, and Alberto Maria Metelli. Catoni-style change point detection for regret minimization in piecewise-stationary heavy-tailed bandits. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 1297–1305. PMLR, 2026.

[GHBV26] Argyrios Gerogiannis, Yu-Han Huang, Subhonmesh Bose, and Venugopal Veeravalli. DAL: A practical prior-free black-box framework for piecewise stationary bandits. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 34611–34638. PMLR, 2026.

[GIZ26] Tomer Gafni, Garud Iyengar, and Assaf Zeevi. The cost of learning under multiple change points, 2026. Version 2, 3 June 2026.

[GL16] Sébastien Gerchinovitz and Tor Lattimore. Refined lower bounds for adversarial bandits. In Advances in Neural Information Processing Systems, volume 29, 2016.

[GM08] Aurélien Garivier and Eric Moulines. On upper-confidence bound policies for nonstationary bandit problems, 2008. Preprint version; see Section 4 for an exploration– detection tradeof.

[GM11] Aurélien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit problems. In Algorithmic Learning Theory, volume 6925 of Lecture Notes in Computer Science, pages 174–188. Springer, 2011.

[HGBV26] Yu-Han Huang, Argyrios Gerogiannis, Subhonmesh Bose, and Venugopal V. Veeravalli. Detection augmented bandit procedures for piecewise stationary MABs: A modular approach. IEEE Transactions on Information Theory, 72(9):7031–7048, 2026. Analysis checked against arXiv:2501.01291v3.

[HS23] Hédi Hadiji and Gilles Stoltz. Adaptation to the range in K-armed bandits. Journal of Machine Learning Research, 24(13):1–33, 2023.

[HYZZ26] Zihao Hu, Yuan Yao, Jiheng Zhang, and Zhengyuan Zhou. Dynamic regret for nonstationary linear bandits via misspecification reductions, 2026.

[HZG24] Qi Han, Li Zhu, and Fei Guo. Forced exploration in bandit problems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 12270–12277, 2024.

[ITH22] Shinji Ito, Taira Tsuchiya, and Junya Honda. Adversarially robust multi-armed bandit algorithm with variance-dependent regret bounds. In Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 1421–1422. PMLR, 2022. Extended abstract; full version arXiv:2206.06810.

[Ito21a] Shinji Ito. Hybrid regret bounds for combinatorial semi-bandits and adversarial linear bandits. In Advances in Neural Information Processing Systems, volume 34, 2021.

[Ito21b] Shinji Ito. Parameter-free multi-armed bandit algorithms with hybrid data-dependent regret bounds. In Proceedings of Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 2552–2583. PMLR, 2021.

[KA16] Zohar S. Karnin and Oren Anava. Multi-armed bandits: Competing with optimal sequences. In Advances in Neural Information Processing Systems, volume 29, 2016.

[KFH24] Junpei Komiyama, Edouard Fouché, and Junya Honda. Finite-time analysis of globally nonstationary multi-armed bandits. Journal of Machine Learning Research, 25(112):1–56, 2024.

[KG25] Ramakrishnan Krishnamurthy and Aditya Gopalan. On slowly-varying non-stationary bandits. Reinforcement Learning Journal, 6:2551–2584, 2025.

[LLS18] Fang Liu, Joohyun Lee, and Ness Shrof. A change-detection based framework for piecewise-stationary multi-armed bandit problem. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32, 2018.

[LS20] Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.

[MCY21] Anne Gael Manegueu, Alexandra Carpentier, and Yi Yu. Generalized non-stationary bandits, 2021. Version 2, 2 February 2021.

[MM19] Subhojyoti Mukherjee and Odalric-Ambrym Maillard. Distribution-dependent and time-uniform bounds for piecewise i.i.d. bandits, 2019.

[MP09] Andreas Maurer and Massimiliano Pontil. Empirical Bernstein bounds and sample variance penalization. In Proceedings of the 22nd Conference on Learning Theory, 2009.

[MZ21] Teodor Vanislavov Marinov and Julian Zimmert. The pareto frontier of model selection for general contextual bandits. In Advances in Neural Information Processing Systems, volume 34, 2021.

[QW26] Jian Qian and Chen-Yu Wei. Achieving optimal static and dynamic regret simultaneously in bandits with deterministic losses, 2026.

[SK22] Joe Suk and Samory Kpotufe. Tracking most significant arm switches in bandits. In Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 2160–2182. PMLR, 2022.

[Suk25] Joe Suk. Adaptive smooth nonstationary bandits. SIAM Journal on Mathematics of Data Science, 7(3):1265–1291, 2025. Technical comparison uses arXiv:2407.08654v2.

[WHL16] Chen-Yu Wei, Yi-Te Hong, and Chi-Jen Lu. Tracking the best expert in non-stationary stochastic environments. In Advances in Neural Information Processing Systems, volume 29, 2016.

[WL18] Chen-Yu Wei and Haipeng Luo. More adaptive algorithms for adversarial bandits. In Proceedings of the 31st Conference on Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 1263–1291. PMLR, 2018.

[WL21] Chen-Yu Wei and Haipeng Luo. Non-stationary reinforcement learning without prior knowledge: an optimal black-box approach. In Proceedings of the Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 4300–4354. PMLR, 2021.

[WPZX21] Yaping Wang, Zhicheng Peng, Riquan Zhang, and Qian Xiao. Robust sequential design for piecewise-stationary multi-armed bandit problem in the presence of outliers. Statistical Theory and Related Fields, 5(2):122–133, 2021.

[WS24] Lai Wei and Vaibhav Srivastava. Nonstationary stochastic bandits: UCB policies and minimax regret. IEEE Open Journal of Control Systems, 3:128–142, 2024. Public preprint arXiv:2101.08980.

[WXC<sup>+</sup>25] Zhiyong Wang, Jize Xie, Yi Chen, John C. S. Lui, and Dongruo Zhou. Variancedependent regret bounds for nonstationary linear bandits. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 2503–2511. PMLR, 2025.

[ZZX26] Yixuan Zhang, Ruihao Zhu, and Qiaomin Xie. On the peril of (even a little) nonstationarity in satisficing regret minimization, 2026. Version 2, 25 April 2026.