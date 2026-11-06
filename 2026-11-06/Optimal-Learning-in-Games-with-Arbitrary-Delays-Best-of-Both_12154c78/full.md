# Optimal Learning in Games with Arbitrary Delays: Best-of-Both-Worlds for Self-Play and Adversarial Opponents

Emmanuel Esposito<sup>∗</sup> FairPlay Team CREST, ENSAE, IP Paris emmanuel@emmanuelesposito.it

Hao Qiu<sup>∗</sup> National University of Singapore qiuhaosai@gmail.com

Andrew Jacobsen<sup>∗</sup> Universitat Pompeu Fabra contact@andrew-jacobsen.com

Alexander Ryabchenko<sup>∗</sup> University of Toronto, Vector Institute alex.rbch.research@gmail.com

Mengxiao Zhang<sup>∗</sup> University of Iowa mengxiao-zhang@uiowa.edu

## Abstract

We study learning in zero-sum games with delayed feedback. We introduce Multi-Anchor Optimistic Online Mirror Descent (MAO-OMD), a single algorithm which obtains optimal fast rates in self-play on smooth games, while simultaneously maintaining minimax optimal guarantees against arbitrary opponents. In particular, in smooth zero-sum games with arbitrary and unknown feedback delays, MAO-OMD obtains $O ( \sigma _ { \mathrm { m a x } } + 1 )$ social and individual regrets when used by both players, where $\sigma _ { \mathrm { m a x } }$ denotes the maximum number of outstanding gradients, while also guaranteeing minimax optimal $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ individual regret against arbitrary opponents, where $d _ { \mathrm { t o t } }$ is the total delay. Moreover, we show that these optimal rates cannot be obtained simultaneously using the standard black-box reduction to the undelayed setting. We also extend our results to multi-player games, and to games with player-dependent delay sequences.

## Contents

1 Introduction 1   
1.1 Our results 1   
2 Setting and notation 2   
3 Main results 4   
4 A scheduling-based approach and its limitation 7   
5 Multi-anchor optimistic online mirror descent 8   
5.1 Anchor weights for delayed feedback . 10   
6 Regret analysis of MAO-OMD 12   
6.1 Results for multi-anchor online mirror descent . 13   
6.2 Optimistic regret bound under delayed feedback 15   
6.3 Proof of the Master Bound (Theorem 5) 17   
6.4 Proof of Theorem 1 . 22   
7 Multiplayer games 23   
8 Player-dependent delays 25   
9 Discussion 26   
References 27   
A Additional related work 32   
B Refined BOLD reduction: guarantees and limitations 33   
B.1 A refined BOLD reduction 33   
B.2 The adversarial guarantee 33   
B.3 The game guarantee . 34   
B.4 A lower bound for every assignment 34   
B.5 Realizing the lower bound with optimistic OMD 36   
C Online convex optimization with arbitrary losses 37   
D Auxiliary results 38

## 1 Introduction

In this paper, we study learning in two-player zero-sum games with delayed feedback over T rounds. Formally, let $f : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ be a smooth convex–concave function on nonempty, compact, convex action sets $\mathcal { X }$ and $\mathcal { V }$ . In each round $t ,$ the players simultaneously choose $x _ { t } \in \mathcal { X }$ and $y _ { t } \in \mathcal { V }$ , and face the convex loss functions $x \mapsto f ( x , y _ { t } )$ and $y \mapsto - f ( x _ { t } , y )$ , respectively. The players observe their respective loss gradients $\nabla _ { x } f ( x _ { t } , y _ { t } )$ and $- \nabla _ { y } f ( x _ { t } , y _ { t } )$ at the end of round $t + d _ { t }$ , where the delay $d _ { t }$ is unknown in advance and may vary across rounds. We measure each player’s performance by its individual (external) regret, the diference between its cumulative loss and that of the best fixed action chosen in hindsight against the opponent’s realized actions. We also consider social regret, defined as the sum of the players’ individual regrets.

In the absence of delays, individual and social regret have been studied extensively [FL98, CBL06, NRTV07, SS12]. A classical result linking regret minimization to equilibrium computation shows that, in a two-player zero-sum game, a social regret bound of $R _ { T }$ guarantees that the players averaged strategies form an $( R _ { T } / T )$ )-approximate Nash equilibrium [FS99]. Thus, standard noregret algorithms with $O ( \sqrt { T } )$ regret automatically yield an $O ( 1 / \sqrt { T } )$ convergence rate. However, it is known that significantly faster rates can be achieved in self-play, where both players employ the same learning algorithm. In particular, when both players employ optimistic algorithms, which use a prediction of the forthcoming gradient when choosing the next action, the resulting dynamics can induce significantly faster convergence to equilibrium [CYL<sup>+</sup>12, RS13a, RS13b, SALS15]. In zero-sum matrix games, [RS13b] established an $O ( ( \log T ) / T )$ equilibrium convergence rate for optimistic mirror descent, while [SALS15] further showed that optimistic algorithms achieve $O ( 1 )$ social regret in general-sum games, yielding an $O ( 1 / T )$ convergence rate in the zero-sum case. In general-sum games, [CP20] obtained improved bounds on external and swap regret, while [DFG21, $\mathrm { A D F ^ { + } 2 2 } ]$ proved bounds that depend only polylogarithmically on $T$ . Recent preprints have further established individual or swap regret bounds that are independent of T in finite general-sum games [LFO26, ALM26, SP26, Mai26].

However, the understanding of fast regret rates under delayed feedback remains limited. The work of [FAA25] showed that even a fixed one-round delay, $d _ { t } = 1$ for all t, can destroy the constant social-regret guarantee of optimistic Follow-the-Regularized-Leader (FTRL). For fixed delays $d _ { t } = d$ for all $t ,$ they recovered $O ( ( d + 1 ) ^ { 2 } )$ social regret by scaling the weight of the optimistic term in the update by a factor of $d + 1$ . More recently, [ZTI26] obtained $\widetilde O ( d + 1 )$ social and individual regret in two-player zero-sum games by interleaving $d + 1$ independent copies of optimistic FTRL, and proved a matching $\Omega ( d + 1 )$ lower bound. Both approaches are formulated for a fixed, known delay d, with the latter requiring both players to follow the same interleaving schedule determined by $d .$ These results leave open how to obtain sharp regret guarantees in self-play that adapt to round-dependent delays unknown in advance, while retaining optimal regret against any opponent. We therefore ask:

Can a learning algorithm achieve the optimal social regret in self-play under arbitrary time-varying delays, while simultaneously guaranteeing the optimal regret against an arbitrary opponent?

## 1.1 Our results

We answer this question afirmatively. The following informal theorem summarizes our main results, which are stated precisely in Section 3 (Theorems 2 and 3).

Theorem (informal). For smooth convex–concave zero-sum games with arbitrary unknown delays, there exists a learning algorithm that guarantees every player using it $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ regret against an arbitrary opponent, improving to $O ( \sigma _ { \operatorname* { m a x } } + 1 )$ when the opponent uses the same algorithm. Here $\sigma _ { \mathrm { m a x } }$ is the maximum number of earlier rounds whose feedback is missing at any point in time, and $\begin{array} { r } { d _ { \mathrm { t o t } } = \sum _ { t = 1 } ^ { T } d _ { t } } \end{array}$ is the total delay.

Both of these rates are optimal. Under fixed delays $d _ { t } \equiv d ,$ we have $\sigma _ { \mathrm { m a x } } = d ,$ and [ZTI26, Theorem 8] shows that for every learning dynamic there is a smooth zero-sum game in which each player sufers Ω(d + 1) regret in self-play, so the dependence on $\sigma _ { \mathrm { m a x } }$ cannot be improved. Against an arbitrary opponent, a player faces online convex optimization with delayed feedback, for which $\sqrt { T + d _ { \mathrm { t o t } } }$ is the minimax regret rate established by [QK15].

To achieve these guarantees, we introduce Multi-Anchor Optimistic Online Mirror Descent (MAO-OMD), an optimistic mirror descent method designed to adapt to the delays as they unfold, without any prior knowledge about them. The key challenge in extending existing optimistic methods to delayed feedback is that delays break the mechanism behind their fast rates. Without delays, the prediction error of each round is controlled by the movement between consecutive actions; the stability term of the update, which regularizes toward the previous action, pays for exactly that movement. Under delays, however, a newly observed gradient may have been generated at an action played many rounds earlier, so its prediction error is controlled by the movement from that past action to the current one, which the stability term toward the previous action does not cover. To account for this, MAO-OMD regularizes each update toward multiple past actions, its anchors, including the actions whose gradients have just been observed. Such regularization produces the stability terms that pay for the prediction errors of the newly observed gradients, while retaining the negative movement terms needed for fast convergence in games.

Choosing these weights is the main technical challenge in the design of MAO-OMD, because they must do two things at once. On the one hand, the weight on each anchor must be large enough to pay for the corresponding prediction error. On the other hand, with arbitrary delays feedback arrives out of order and an action is anchored at several rounds, which destroys the telescoping structure of standard mirror-descent regret bounds. The cancellation survives only if no action is anchored with more total weight than it was computed with. Inspired by the weighting principle of Banker-OMD [HDH23, QJEZ26b], we develop an adaptive allocation of regularization weights across pending rounds that meets both conditions and yields a regret bound whose prediction-error costs scale with the total delay while keeping the negative stability terms needed for fast rates.

The paper is organized as follows. Section 2 introduces the setting and the notation, and Section 3 states our main results. In Section 4, we provide a variant of the BOLD reduction [JGS13] achieving $O ( \sigma _ { \mathrm { m a x } } + 1 )$ regret in games with arbitrary delays, and show that no such variant can bring its regret against an arbitrary opponent within a factor $\sqrt { 1 + \log ( \sigma _ { \operatorname* { m a x } } + 1 ) }$ of the optimal rate, which motivates MAO-OMD. Sections 5 and 6 introduce MAO-OMD together with its weight-allocation strategy, and contain its analysis, respectively. Finally, Sections 7 and 8 extend our main results to multi-player games and to player-dependent delays, respectively. Additional related work on learning with delayed feedback and in games is discussed in Appendix A.

## 2 Setting and notation

Let $\mathbb { N } : = \{ 0 , 1 , 2 , \dots \}$ denote the natural numbers and let $ { \mathbb { N } } _ { + } : =  { \mathbb { N } } \setminus \{ 0 \}$ . For an integer $n \in \mathbb { N }$ let $[ n ] : = \{ 1 , \dots , n \}$ , with $[ 0 ] = \emptyset$ . The disjoint union of sets is denoted by ⊔.

We first introduce a general online protocol with delayed feedback, which describes the interaction from an individual player’s perspective. We then specialize this protocol to two-player zero-sum games. Finally, we introduce the delay-induced notation used throughout the analysis.

Online protocol. Let $\mathcal { W } \subseteq \mathbb { R } ^ { n } , n \in \mathbb { N } _ { + }$ be a nonempty, compact, and convex domain, equipped with a norm $\lVert \cdot \rVert$ and its dual norm $\left\| \cdot \right\| _ { * }$ . The interaction proceeds over $T \in \mathbb { N } _ { + }$ rounds. At the start of each round $t \in [ T ]$ , the player selects an action $w _ { t } \in \mathcal W$ and the environment then chooses a diferentiable convex loss function $f _ { t } : \mathcal { W } \to \mathbb { R }$ , resulting in a loss $f _ { t } ( w _ { t } )$ that the player does not observe. Feedback for round t arrives with a delay: at the end of each round $t ^ { \prime } \geq t ,$ the environment decides whether to release the feedback for round $t ,$ and if it does, it reveals a gradient $g _ { t } : = \nabla f _ { t } ( w _ { t } )$ . The feedback for every round is released exactly once, and we denote the resulting delay by $d _ { t } \in \mathbb { N }$ , so that $g _ { t }$ is revealed at the end of round $t + d _ { t } \leq T .$ 1

Throughout, every environment’s choice may depend on the entire history up to that point. Accordingly, our bounds depend on realized loss functions $f _ { t } ,$ gradients ${ \mathit { g } } _ { t } ,$ and delays $d _ { t }$ . The performance is measured by the external regret with respect to a comparator $u \in \mathcal W \mathrm { : }$

$$
\mathrm { R e g } _ { T } ( u ) : = \sum _ { t = 1 } ^ { T } \bigl ( f _ { t } ( w _ { t } ) - f _ { t } ( u ) \bigr ) \ , \qquad \mathrm { R e g } _ { T } : = \operatorname* { s u p } _ { u \in \mathcal { W } } \mathrm { R e g } _ { T } ( u ) \ ,
$$

where $\mathrm { R e g } _ { T }$ denotes the regret against the best fixed comparator in hindsight. By convexity, the external regret above is bounded by the linearized regret: $\mathrm { R e g } _ { T } ( u ) \leq \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle$

Zero-sum games. In two-player zero-sum games, each player faces the online protocol above with loss functions determined by the other player’s decisions. Let $\mathcal { X } \subseteq \mathbb { R } ^ { n _ { x } }$ and $\mathcal { V } \subseteq \mathbb { R } ^ { n _ { y } }$ be nonempty, compact, and convex domains, and let $f : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ be diferentiable, convex in its first argument, and concave in its second. At round $t \in [ T ]$ , the players simultaneously select $x _ { t } \in \mathcal { X }$ and $y _ { t } \in \mathcal { V }$ , incurring losses $f ( x _ { t } , y _ { t } )$ and $- f ( x _ { t } , y _ { t } )$ . The players receive their loss gradients $g _ { t } ^ { x } : = \nabla _ { x } f ( x _ { t } , y _ { t } )$ and $\boldsymbol { g } _ { t } ^ { y } : = - \nabla _ { y } f ( x _ { t } , y _ { t } )$ at the end of round $t + d _ { t }$ . Their individual regrets are

$$
\mathrm { R e g } _ { T } ^ { x } : = \operatorname* { s u p } _ { x \in \mathcal { X } } \sum _ { t = 1 } ^ { T } \bigl ( f ( x _ { t } , y _ { t } ) - f ( x , y _ { t } ) \bigr ) \ , \qquad \mathrm { R e g } _ { T } ^ { y } : = \operatorname* { s u p } _ { y \in \mathcal { Y } } \sum _ { t = 1 } ^ { T } \bigl ( f ( x _ { t } , y ) - f ( x _ { t } , y _ { t } ) \bigr ) \ ,
$$

and their social regret is Soc $\mathrm { R e g } _ { T } : = \mathrm { R e g } _ { T } ^ { x } + \mathrm { R e g } _ { T } ^ { y }$ . By convexity, the social regret can be used to control the duality gap of the averaged decisions $\begin{array} { r } { \bar { x } _ { T } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } x _ { t } } \end{array}$ and $\begin{array} { r } { \bar { y } _ { T } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } y _ { t } \colon } \end{array}$

$$
\mathrm { D u a l G a p } _ { T } : = \operatorname* { s u p } _ { y \in \mathcal { Y } } f ( \bar { x } _ { T } , y ) - \operatorname* { i n f } _ { x \in \mathcal { X } } f ( x , \bar { y } _ { T } ) \leq \frac { 1 } { T } \mathrm { S o c R e g } _ { T } ,\tag{1}
$$

which measures the deviation from the saddle point of f, i.e., a Nash equilibrium [FS99, FKS19].

Delay-induced notation. Throughout the paper, we use the following notation induced by the sequence of realized delays $( d _ { t } ) _ { t = 1 } ^ { T }$ . For each round $t \in [ T ]$ , define four sets of round indices:

$$
\begin{array} { r l } & { \mathrm { ( o b s e r v e d ) } \ \mathcal { O } _ { t } : = \big \{ s \in [ t - 1 ] : s + d _ { s } < t \big \} , \qquad \mathrm { ( p e n d i n g ) } \ \mathcal { P } _ { t } : = \big \{ s \in [ t - 1 ] : s + d _ { s } \ge t \big \} , } \\ & { \mathrm { ~ ( n e w ) ~ } \quad \mathcal { N } _ { t } : = \big \{ s \in [ t ] : s + d _ { s } = t \big \} , \qquad \mathrm { ( a c t i v e ) } \ \mathcal { A } _ { t } : = \mathcal { P } _ { t } \cup \{ t \} . } \end{array}
$$

The sets ${ \mathcal { O } } _ { t }$ and $\mathcal { P } _ { t }$ partition the past rounds into those whose feedback has been observed before round t and those whose feedback remains pending at its start. The set $\mathcal { N } _ { t }$ contains rounds whose feedback is newly observed in round t, while the active set $\boldsymbol { A } _ { t }$ consists of all rounds $s \in [ t ]$ whose feedback arrives in round t or later. In particular, $\mathcal { O } _ { t } \sqcup \mathcal { P } _ { t } = [ t - 1 ] , \bigcup _ { t = 1 } ^ { T } \mathcal { N } _ { t } = [ T ]$ , and $\mathcal { N } _ { t } \subseteq \mathcal { A } _ { t }$

We write $q _ { t } : = | A _ { t } |$ and $\sigma _ { t } : = | \mathcal { P } _ { t } |$ for the numbers of active and pending rounds, respectively, and denote their maxima by $q _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { t \in [ T ] } q _ { t }$ and $\sigma _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { t \in [ T ] } \sigma _ { t } ,$ the maximum numbers of active and pending rounds, and the total delay by $\begin{array} { r } { d _ { \mathrm { t o t } } : = \sum _ { t \in [ T ] } \dot { d _ { t } } . } \end{array}$ Since $q _ { t } = \sigma _ { t } + 1$ , we have $q _ { \mathrm { m a x } } = \sigma _ { \mathrm { m a x } } + 1$ . Moreover, as each round t is pending during exactly $d _ { t }$ rounds, $\textstyle \sum _ { t \in [ T ] } \sigma _ { t } = d _ { \mathrm { t o t } }$ and hence $\textstyle \sum _ { t \in [ T ] } q _ { t } = T + d _ { \mathrm { t o t } }$ . The maximum number of active rounds is also controlled by the total delay: $q _ { \mathrm { m a x } } \le \sqrt { T + 2 d _ { \mathrm { t o t } } }$ (Fact 1(e)). As is standard in the delayed-feedback literature, our bounds are stated in terms of $\sigma _ { \mathrm { m a x } }$ and $d _ { \mathrm { t o t } }$ , with $q _ { \mathrm { m a x } }$ used in place of $\sigma _ { \mathrm { m a x } } + 1$ for convenience.

We also use the notion of congestion, defined as the maximum number of active rounds while a given round is active. Concretely, for $t \in [ T ] , \kappa _ { t } : = \operatorname* { m a x } _ { t \leq r \leq t + d _ { t } } q _ { r }$ and $\kappa _ { t , s } : = \operatorname* { m a x } _ { t \pm r \leq s } q _ { \ i }$ for $t \leq s \leq t + d _ { t }$ , where $\kappa _ { t }$ is the congestion of round t and $\kappa _ { t , s }$ is its running version, observable at round $s \leq t + d _ { t }$ . An important property of congestion is that its total sum $\begin{array} { r } { \kappa _ { \mathrm { t o t } } : = \sum _ { t \in [ T ] } \kappa _ { t } } \end{array}$ is of order $\Theta ( T + d _ { \mathrm { t o t } } )$ . Indeed, any round active between t and $t + d _ { t }$ is either already active at round t or is one of the rounds $t + 1 , \ldots , t + d _ { t } .$ so that $q _ { t } \leq \kappa _ { t } \leq q _ { t } + d _ { t }$ and hence $T + d _ { \mathrm { t o t } } \leq \kappa _ { \mathrm { t o t } } \leq T + 2 d _ { \mathrm { t o t } }$ We use this property in the analysis: by charging each round roughly in proportion to its own congestion rather than the global $q _ { \mathrm { m a x } }$ , we obtain bounds that scale with $T + d _ { \mathrm { t o t } }$ instead of $T q _ { \mathrm { m a x } } .$

Further useful properties of these delay-induced quantities are collected in Appendix D.

## 3 Main results

In this section, we formally state the guarantees of our algorithm, MAO-OMD, in smooth convex– concave zero-sum games: fast rates when both players run this algorithm, and the optimal regret under delays for a player that runs it against an arbitrary opponent.

For the two-player zero-sum game of Section 2, let $g ^ { x } ( x , y ) \ : = \ \nabla _ { x } f ( x , y )$ and $g ^ { y } ( x , y ) : =$ $- \nabla _ { y } f ( x , y )$ denote the players’ loss gradients at a profile $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ , so that $g _ { t } ^ { p } = g ^ { p } ( x _ { t } , y _ { t } )$ for $p \in \{ x , y \}$ . Following [RS13b], we assume that f is Lipschitz and smooth in the profile, that is, both gradient maps are bounded and Lipschitz, with constants known to the players (Assumption 1).

Assumption 1 (G-Lipschitzness and L-smoothness). There exist constants $G , L > 0$ such that

$$
\| g ^ { p } ( x , y ) \| _ { * } \leq G \qquad a n d \qquad \| g ^ { p } ( x , y ) - g ^ { p } ( x ^ { \prime } , y ^ { \prime } ) \| _ { * } \leq L \big ( \| x - x ^ { \prime } \| + \| y - y ^ { \prime } \| \big )
$$

for each player $p \in \{ x , y \}$ and all $x , x ^ { \prime } \in { \mathcal { X } }$ and $y , y ^ { \prime } \in \mathcal { V }$

MAO-OMD is an optimistic online mirror descent (OMD) method, which we describe for a player facing the online protocol of Section 2, with action set W, actions $w _ { t } .$ , and loss gradients $g _ { t }$ . The method takes as input a regularizer $\psi : \mathcal { W } \to \mathbb { R }$ and a reference point $w _ { 0 } \in \mathcal { W }$ , and produces the iterates $w _ { 1 } , w _ { 2 } , \ldots .$ . starting from w : each iterate $w _ { t + 1 }$ is computed by a mirror descent step that, as in every OMD method, is regularized by the Bregman divergence $D _ { \psi } ( u \mid$ $w ) : = \psi ( u ) - \psi ( w ) - \langle \nabla \psi ( w ) , u - w \rangle$ , here taken with respect to several of the previous iterates $w _ { 0 } , \ldots , w _ { t }$ . As in every optimistic method $[ \mathrm { C Y L ^ { + } 1 2 } , \mathrm { R S 1 3 a } ]$ , each iterate $w _ { t }$ is selected using a hint $h _ { t }$ for the upcoming gradient $g _ { t }$ , formed from the feedback revealed before round t. Although MAO-OMD regularizes toward many previous iterates, our analysis only requires the Bregman radius of W around $w _ { 0 }$ to be bounded (Assumption 2).

Assumption 2 (Strongly convex regularizer and bounded Bregman radius). The regularizer ψ is diferentiable and 1-strongly convex with respect to ∥·∥, i.e., $\begin{array} { r } { D _ { \psi } ( u \mid w ) \geq \frac { 1 } { 2 } \Vert u - w \Vert ^ { 2 } } \end{array}$ for all $u , w \in \mathcal { W }$ . Additionally, there is a known constant $M > 0$ and reference point $w _ { 0 } \in \mathcal { W }$ such that $D _ { \psi } ( w \mid w _ { 0 } ) \le M$ for all $w \in \mathcal W$

At each round t, MAO-OMD uses as its hint the earliest observed gradient that has not yet served as a hint. Specifically, let $s _ { t }$ be the smallest element of $\mathcal { O } _ { t } \setminus \left\{ s _ { 1 } , \ldots , s _ { t - 1 } \right\}$ , with $s _ { t } = 0 \ \mathrm { i f }$ this set is empty, and set $h _ { t } : = g _ { s _ { t } } ,$ where $g _ { 0 } : = 0$ . By construction, each observed gradient serves as a hint at most once, and at most $q _ { \mathrm { m a x } } = \sigma _ { \mathrm { m a x } } + 1$ rounds have $s _ { t } = 0$ (as shown in Lemma 3). The following theorem states the main guarantee for a player running MAO-OMD.

Theorem 1 (informal). Let Assumptions 1 and 2 hold. For arbitrary unknown delays and any sequence of opponent’s actions, a player that runs MAO-OMD (Algorithm 1) guarantees that, for all $u \in \mathcal W$ ，

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq ( 8 L M + 4 G \sqrt { M } ) q _ { \operatorname* { m a x } } + 2 \sqrt { 2 M \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } - L \sum _ { t = 1 } ^ { T } \| w _ { t } - w _ { s t } \| ^ { 2 } .
$$

We state the complete version of Theorem 1 in Section 5.1 and provide its proof in Section 6.4. In the remainder of this section, we derive its implications for two-player zero-sum games, which are our main results.

Fast rates when both players run MAO-OMD. Suppose that both players run MAO-OMD with regularizers $\psi _ { x }$ and $\psi _ { y }$ satisfying Assumption 2 with the same constant M. Since the hint rule depends only on the delays, which are common to both players, they select the same hint source $s _ { t }$ at every round, and whenever $s _ { t } > 0$ their hints $h _ { t } ^ { p } = g ^ { p } ( x _ { s _ { t } } , y _ { s _ { t } } )$ come from the same past profile. Smoothness therefore bounds each player’s prediction error by the movement of both players since round $s _ { t } ,$ which the negative terms of Theorem 1, summed over the two players, control. Theorem 2 uses this to bound the social regret, and the nonnegativity of the social regret in zero-sum games extends the bound to individual regrets.

Theorem 2. Let Assumption 1 hold. If both players run MAO-OMD with regularizers satisfying Assumption 2 under arbitrary unknown delays, then the social regret SocReg and individual regrets Reg<sup>x</sup><sub>T</sub>, Reg<sup>y</sup><sub>T</sub> are all $O ( ( G \sqrt { M } + L M ) q _ { \mathrm { m a x } } )$ , while DualGap is $O \Big ( \frac { ( G \sqrt { M } + L M ) q _ { \mathrm { m a x } } } { T } \Big )$

Proof. Let $\mathcal { W } _ { x } = \mathcal { X }$ and $\mathcal { W } _ { y } = \mathcal { V }$ . Denote by $\begin{array} { r } { \Delta _ { T } ^ { p } : = \sum _ { t = 1 } ^ { T } \lVert w _ { t } ^ { p } - w _ { s _ { t } } ^ { p } \rVert ^ { 2 } } \end{array}$ the movement of player $p \in \{ x , y \}$ relative to the hint sources, where $w _ { t } ^ { x } = x _ { t }$ and $w _ { t } ^ { y } = y _ { t } .$ , and let $\Delta _ { T } : = \Delta _ { T } ^ { x } + \Delta _ { T } ^ { y }$ Both players use the same hint sources $s _ { t } .$ , so at every round t with $s _ { t } > 0$ , L-smoothness gives $\| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } ^ { 2 } \leq L ^ { 2 } \big ( \| x _ { t } - x _ { s _ { t } } \| + \| y _ { t } - y _ { s _ { t } } \| \big ) ^ { 2 }$ . At every round t with $s _ { t } = 0 , h _ { t } ^ { p } = 0$ and G-Lipschitzness gives $\| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } ^ { 2 } \leq G ^ { 2 }$ instead. As there are at most $q _ { \mathrm { m a x } }$ rounds with $s _ { t } ~ = ~ 0$ (see Lemma 3 provided later in Section 6 plus the choice $s _ { 1 } = 0 )$ and $\operatorname* { m a x } _ { t } \kappa _ { t } \leq q _ { \operatorname* { m a x } }$ , we have $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \dot { \kappa } _ { t } \| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } ^ { 2 } \leq } \end{array}$ q<sub>max</sub> $( G ^ { 2 } q _ { \mathrm { m a x } } + 2 L ^ { 2 } \Delta _ { T } )$ . Theorem 1 then gives, for all $u \in \mathcal W _ { p }$

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle g _ { t } ^ { p } , w _ { t } ^ { p } - u \rangle \leq \big ( 8 L M + 4 G \sqrt { M } \big ) q _ { \operatorname* { m a x } } + 2 \sqrt { 2 M q _ { \operatorname* { m a x } } \big ( G ^ { 2 } q _ { \operatorname* { m a x } } + 2 L ^ { 2 } \Delta _ { T } \big ) } - L \Delta _ { T } ^ { p } } \\ { \leq \big ( 8 L M + ( 4 + 4 ) G \sqrt { M } \big ) q _ { \operatorname* { m a x } } + 4 L \sqrt { M q _ { \operatorname* { m a x } } \Delta _ { T } } - L \Delta _ { T } ^ { p } , } \end{array}
$$

where the last inequality uses ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ . By convexity of $f ( \cdot , y _ { t } )$ and concavity of $f ( x _ { t } , \cdot )$ the left-hand side upper bounds $\mathrm { R e g } _ { T } ^ { p } ( u )$ Taking the supremum over u bounds the individual regrets, and summing the resulting bounds for the two players bounds the social regret:

$$
\mathrm { R e g } _ { T } ^ { p } \leq 8 \bigl ( L M + G \sqrt { M } \bigr ) q _ { \mathrm { m a x } } + 4 L \sqrt { M q _ { \mathrm { m a x } } \Delta _ { T } } - L \Delta _ { T } ^ { p } ,\tag{2}
$$

$$
\mathrm { S o c R e g } _ { T } \leq 1 6 \big ( L M + G \sqrt { M } \big ) q _ { \mathrm { m a x } } + 8 L \sqrt { M q _ { \mathrm { m a x } } \Delta _ { T } } - L \Delta _ { T } \ .\tag{3}
$$

By the AM-GM inequality, we further have that SocR $\mathrm { e g } _ { T } \leq ( 1 6 G \sqrt { M } + 3 2 L M ) q _ { \mathrm { m a x } }$ , and consequently $\begin{array} { r } { \mathrm { D u a l G a p } _ { T } \leq \mathrm { S o c R e g } _ { T } / T \leq \left( 1 6 G \sqrt { M } + 3 2 L M \right) \frac { q _ { \operatorname* { m a x } } } { T } } \end{array}$ via Equation (1).

To bound the individual regrets, we first bound $\Delta _ { T }$ . Since $\mathrm { S o c R e g } _ { T } \geq T$ · Dua $\mathrm { { I G a p } } _ { T } \geq 0$ applying $8 L \sqrt { M q _ { \mathrm { m a x } } \Delta _ { T } } \leq L \Delta _ { T } / 2 + 3 2 L M q _ { \mathrm { m a x } }$ in (3) and rearranging gives $L \Delta _ { T } \leq ( 3 2 G \sqrt { M } +$ $9 6 L M ) q _ { \mathrm { m a x } }$ . Substituting this into (2) after applying $4 L \sqrt { M q _ { \mathrm { m a x } } \Delta _ { T } } \leq L \Delta _ { T } / 4 + 1 6 L M q _ { \mathrm { m a x } }$ , and dropping the nonpositive term − $- L \Delta _ { T } ^ { p }$ , we obtain $\mathrm { R e g } _ { T } ^ { p } \leq \bigl ( 1 6 G \sqrt { M } + 4 8 L M \bigr ) q _ { \mathrm { m a x } }$ □

Suppressing the dependence on $G , L ,$ and M, both individual and social regrets are $O ( q _ { \mathrm { m a x } } ) =$ $O ( \sigma _ { \mathrm { m a x } } + 1 )$ and the equilibrium convergence rate is $O ( ( \sigma _ { \operatorname* { m a x } } + 1 ) / T )$ , with the dependence on $\sigma _ { \mathrm { m a x } }$ matching the lower bound of [ZTI26, Theorem 8]. In the absence of delays, $\mathrm { i . e . , } d _ { t } = 0$ for all $t ,$ our regret bound becomes $O ( G { \sqrt { M } } + L M )$ , recovering the regret of optimistic mirror descent with an optimally tuned fixed step size [RS13b, Lemma 1].

Regret against an arbitrary opponent. Suppose now that player x runs MAO-OMD while player y takes arbitrary, potentially adversarial actions. The hints of player x need not be related to the gradients it receives, so the negative term of Theorem 1 cannot be relied on, but the prediction errors remain bounded by G-Lipschitzness. Theorem 3 shows that this is enough for optimal regret.

Theorem 3. Let Assumption 1 hold. If player x runs MAO-OMD with a regularizer satisfying Assumption 2 under arbitrary unknown delays and against an opponent y taking arbitrary actions, then its individual regret Reg<sup>x</sup> is $O ( G \sqrt { M ( T + d _ { \mathrm { t o t } } ) } + L M q _ { \mathrm { m a x } } )$

Proof. Each hint of player x is either a gradient or zero, so G-Lipschitzness gives $\| g _ { t } ^ { x } - h _ { t } ^ { x } \| _ { * } ^ { 2 } \leq 4 G ^ { 2 }$ for all t. Since $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \kappa _ { t } \leq T + 2 d _ { \mathrm { t o t } } ~ ( \mathrm { F a c t ~ 2 ( b ) } ) , \sum _ { t = 1 } ^ { T } \kappa _ { t } \Vert g _ { t } ^ { x } - h _ { t } ^ { x } \Vert _ { * } ^ { 2 } \leq 8 G ^ { 2 } ( T + d _ { \mathrm { t o t } } ) } \end{array}$ . Theorem 1 then gives $\begin{array} { r } { \mathrm { R e g } _ { T } ^ { x } \leq \operatorname* { s u p } _ { u \in \mathcal { X } } \sum _ { t = 1 } ^ { T } \langle g _ { t } ^ { x } , x _ { t } - u \rangle \leq ( 8 L M + 4 G \sqrt { M } ) q _ { \operatorname* { m a x } } + 8 G \sqrt { M ( T + 2 d _ { \mathrm { t o t } } ) } } \end{array}$ , by convexity of $f ( \cdot , y _ { t } )$ . To obtain the stated bound, use $q _ { \mathrm { m a x } } ^ { 2 } \leq T + 2 d _ { \mathrm { t o t } }$ (Fact 1(e)). □

Suppressing the dependence on G, L, and M, and since $q _ { \mathrm { m a x } } ^ { 2 } \leq T + 2 d _ { \mathrm { t o t } }$ (Fact 1(e)), the regret bound of Theorem 3 is $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ , which matches the optimal rate for delayed online convex optimization [QK15]. Together, Theorems 2 and 3 show that a player running MAO-OMD obtains both guarantees in the same run, without knowing whether its opponent also runs MAO-OMD.

Additional results. We extend MAO-OMD to three further settings. For brevity, the bounds in this paragraph suppress the dependence on G, L, and M. First, in Section 7, we consider Nplayer smooth games with nonnegative social regret, as in [APFS22], which include zero-sum and constant-sum polymatrix games. We show that MAO-OMD, with an optimism weight that scales with N, guarantees social regret $O ( N ^ { 2 } q _ { \mathrm { m a x } } )$ and individual regret $O ( N q _ { \mathrm { m a x } } )$ when all players run it, and individual regret $O ( N q _ { \mathrm { m a x } } + \sqrt { T + d _ { \mathrm { t o t } } } )$ against arbitrary opponents. Second, in Section $^ { 8 , }$ we consider player-dependent delays, where the feedback of a round may reach the two players at diferent times. If the players communicate which rounds they have received, they can run

MAO-OMD under the common efective delays max $\{ d _ { t } ^ { x } , d _ { t } ^ { y } \}$ , and we show that both social and individual regret are then $O ( q _ { \mathrm { m a x } } ^ { x } + q _ { \mathrm { m a x } } ^ { y } )$ . Finally, in Appendix C, we consider single-player online convex optimization, where the losses are Lipschitz and not necessarily smooth, and show that MAO-OMD still achieves the optimal $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ regret of [QK15].

## 4 A scheduling-based approach and its limitation

A natural approach to learning with delayed feedback is to run multiple independent instances of an undelayed learning algorithm, ensuring that each instance receives feedback from its previous action before being used again. This is the idea behind the BOLD reduction of [JGS13], which assigns each round $t \in [ T ]$ to an available instance, creating a new one whenever all existing instances are awaiting feedback. The resulting assignment partitions [T] into subsequences $I _ { 1 } , \ldots , I _ { q _ { \mathrm { m a x } } }$ , such that consecutive rounds $s < t$ in the same subsequence satisfy $s + d _ { s } < t .$ . Thus, each subsequence can be handled by an independent undelayed online-learner B, and the total number of instances created is $q _ { \mathrm { m a x } }$ . If the base learner guarantees regret at most $C _ { B \mathrm { V } } / n$ on any n rounds, then, writing $n _ { k } = | I _ { k } |$ , the standard analysis gives $\begin{array} { r } { \mathrm { R e g } _ { T } \leq C _ { \mathcal { B } } \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } \sqrt { n _ { k } } \leq C _ { \mathcal { B } } \sqrt { q _ { \operatorname* { m a x } } T } } \end{array}$ , which can be significantly larger than the minimax $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ rate [QK15].

BOLD allows each round to be assigned to any available instance, i.e., one whose previous feedback has already arrived. We refine this assignment rule by indexing instances in order of creation and always selecting the available instance with the smallest index, creating a new one only when none is available. By preferentially reusing earlier instances, this greedy rule yields a sharper regret bound, losing only a logarithmic factor relative to the minimax rate, while still retaining the $O ( q _ { \mathrm { m a x } } )$ rates when used by both players of a smooth game. The resulting guarantees are stated in the following theorem, with the pseudocode and proofs deferred to Appendix B.

Theorem 4. The refined BOLD reduction in Algorithm 2 has the following guarantees without advance knowledge of the delays:

(i) In the online convex setting of Section 2, if B guarantees regret at most $C _ { B \mathrm { V } } / n$ for every $n \geq 1$ , then $\begin{array} { r } { \mathrm { R e g } _ { T } \leq C _ { B } \sum _ { k } \sqrt { n _ { k } } \leq C _ { B } \sqrt { ( T + d _ { \mathrm { t o t } } ) ( 1 + \log q _ { \mathrm { m a x } } ) } } \end{array}$

(ii) Under Assumptions 1 and 2, if B guarantees individual and social regret $O ( L M + G { \sqrt { M } } )$ in undelayed zero-sum self-play for every horizon, then both players using the reduction with common delays satisfy SocRe ${ \mathrm { g } } _ { T } , { \mathrm { R e g } } _ { T } ^ { p } = O { \left( ( L M + G { \sqrt { M } } ) q _ { \mathrm { m a x } } \right) } \ f o r \ p \in \{ x , y \}$

The result for part (i) follows by always choosing the smallest available index over idle instances of B, which ensures $\begin{array} { r } { \sum _ { k } k n _ { k } \le T + d _ { \mathrm { t o t } } } \end{array}$ , and the weighted Cauchy-Schwarz inequality yields the stated regret bound. For part (ii), under common delays both players assign each round to the same instance, creating the same partition $I _ { 1 } , \ldots , I _ { q _ { \mathrm { m a x } } }$ of rounds, and each instance receives feedback from its previous round before being used again. Thus, each pair of corresponding instances efectively plays the same undelayed game, and the RVU property [SALS15] yields the desired $O ( q _ { \mathrm { m a x } } )$ regret guarantee, as recently shown by [ZTI26] for fixed delays with $q _ { \operatorname* { m a x } } = d + 1$

Although part (ii) of Theorem 4 matches the desired $O ( q _ { \mathrm { m a x } } )$ fast regret rate when both players run this refined version of BOLD, the additional $\sqrt { 1 + \log q _ { \mathrm { m a x } } }$ overhead against adversarial opponents in part (i) of Theorem 4 cannot be removed by any partitioning of rounds, as we show in the following proposition.

Proposition 1. For every integer $\sigma _ { \mathrm { m a x } } ~ \geq ~ 1$ , there are arbitrarily long delay sequences whose maximum number of pending rounds is exactly $\sigma _ { \mathrm { m a x } }$ and for which every valid assignment of rounds to instances, with loads $n _ { k }$ , satisfies $\begin{array} { r } { \sum _ { k } \sqrt { n _ { k } } \ge \sqrt { ( T + d _ { \mathrm { t o t } } ) \frac { 1 + \log _ { \cdot } } { 2 } } } \end{array}$ <sup>qmax</sup> . Moreover, for every fixed $\gamma > 0$ , if each instance runs the optimistic OMD learner specified in Section B.5, an adaptive linear-loss adversary forces every valid assignment to incur $\mathrm { R e g } _ { T } = \Omega \Bigl ( \sqrt { ( T + d _ { \mathrm { t o t } } ) ( 1 + \log q _ { \mathrm { m a x } } ) } \Bigr )$ The implicit constant may depend on γ but is independent of T and σ<sub>max</sub>.

In the next section, we develop a single-instance algorithm, MAO-OMD, which retains the fast $O ( q _ { \mathrm { m a x } } )$ rate while guaranteeing $O ( G \sqrt { M ( T + d _ { \mathrm { t o t } } ) } + L M q _ { \mathrm { m a x } } )$ regret against arbitrary opponents in the same run (Theorems 2 and 3). Since $q _ { \mathrm { m a x } } ^ { 2 } \leq T + 2 d _ { \mathrm { t o t } }$ (via Fact 1), this gives regret $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ , without the additional $\sqrt { 1 + \log q _ { \mathrm { m a x } } }$ factor in Theorem 4 part (i).

## 5 Multi-anchor optimistic online mirror descent

The scheduling-based approach of Section 4 can achieve near-optimal regret in smooth games when applied with OMD as the base learner, but will incur a suboptimal logarithmic factor in its adversarial regret guarantee as shown above in Proposition 1. To obtain both optimal guarantees simultaneously, we develop a new optimistic learning algorithm that handles delayed feedback directly, without maintaining independent learning instances. The key challenge is to obtain an optimistic regret bound that controls the prediction errors $\| g _ { t } - h _ { t } \| _ { * } ^ { 2 }$ individually, while retaining negative terms that can compensate for these errors when both players use the algorithm in a smooth game. Existing optimistic regret bounds for delayed feedback do not directly provide this combination of properties, even for fixed delays, and extending them to arbitrary delays introduces further dificulties because gradients may arrive out of order $[ \mathrm { F O C ^ { + } 2 1 }$ , FAA25, ZTI26]. In this section we introduce Multi-Anchor Optimistic Online Mirror Descent (MAO-OMD), a generalization of optimistic mirror descent framework, to address this challenge. Our starting point is the following regret decomposition, which expresses the regret in terms of the feedback received at each round.

Lemma 1 (Delays-to-feedback decomposition). For any $u \in \mathcal W$ and any sequence $h _ { 1 } , \ldots , h _ { T + 1 } \in$ $\mathbb { R } ^ { n }$ of hints with $h _ { 1 } = h _ { T + 1 } = 0$ 2

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle = \sum _ { t = 1 } ^ { T } \biggl \langle h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } \bigl ( g _ { s } - h _ { s } \bigr ) , w _ { t + 1 } - u \biggr \rangle + \sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \bigl \langle g _ { s } - h _ { s } , w _ { s } - w _ { t + 1 } \bigr \rangle .\tag{4}
$$

The decomposition in Lemma 1 naturally suggests an optimistic mirror descent update. To see why, notice that the right-hand side involves only quantities that are observable at the end of round $t ,$ and assuming that $\psi$ is 1-strongly convex with respect to $\lVert \cdot \rVert$ , we can bound each term of the second summation using Fenchel-Young inequality:

$$
\langle g _ { s } - h _ { s } , w _ { s } - w _ { t + 1 } \rangle \leq c _ { s } ^ { ( t + 1 ) } D _ { \psi } \big ( w _ { t + 1 } | w _ { s } \big ) + \frac { \| g _ { s } - h _ { s } \| _ { * } ^ { 2 } } { 2 c _ { s } ^ { ( t + 1 ) } } ,
$$

for any choice of weights $c _ { s } ^ { ( t + 1 ) }$ satisfying $c _ { s } ^ { ( t + 1 ) } > 0$ for all $s \in \mathcal { N } _ { t }$ and $t \in [ T ]$ . Hence, following Equation (4) we can further upper-bound $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle } \end{array}$ by

$$
\sum _ { t = 1 } ^ { T } \left[ \left. h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } \left( g _ { s } - h _ { s } \right) , w _ { t + 1 } - u \right. + \sum _ { s \in \mathcal { N } _ { t } } c _ { s } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { s } ) \right] + \sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \frac { \| g _ { s } - h _ { s } \| _ { * } ^ { 2 } } { 2 c _ { s } ^ { ( t + 1 ) } } .\tag{5}
$$

This naturally suggests choosing $w _ { t + 1 }$ to minimize the expression inside the brackets, resulting in a mirror descent step of the form

$$
w _ { t + 1 } \in \underset { w \in \mathcal { W } } { \arg \operatorname* { m i n } } \left\{ \left. h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } \left( g _ { s } - h _ { s } \right) , w \right. + \sum _ { s \in \mathcal { N } _ { t } } c _ { s } ^ { ( t + 1 ) } D _ { \psi } ( w \mid w _ { s } ) \right\} .\tag{6}
$$

In words, the regret decomposition suggests regularizing toward the past iterates corresponding to the gradients that have been newly-observed on round t.

However, while Equation (6) naturally controls the delayed prediction-error terms, our goal is not merely to obtain sublinear regret under arbitrary delays. Rather, we seek a best-of-both-worlds guarantee that recovers the optimal worst-case dependence on delays while retaining the quadratic prediction-error dependence and negative stability terms needed for the fast rates obtained in Section 3. Achieving this requires controlling the accumulation of Bregman-divergence terms across rounds. In particular, unlike standard mirror descent, the use of multiple anchors introduces a penalty of the form

$$
\sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } c _ { s } ^ { ( t + 1 ) } \left[ D _ { \psi } ( u \mid w _ { s } ) - D _ { \psi } ( u \mid w _ { t + 1 } ) \right] ,
$$

which does not telescope in the usual way (see Section 6 for a more detailed discussion). Controlling this penalty requires a careful assignment of regularization weights across rounds, as restricting regularization to only the newly observed gradients prevents the cancellation of Bregman-divergence terms associated with rounds without feedback, potentially causing the above penalty to accumulate over time.

This motivates a more general update in which any previous iterate can serve as an anchor. Specifically, we introduce Multi-Anchor Optimistic Online Mirror Descent (MAO-OMD), defined by

$$
w _ { t + 1 } \in \underset { w \in \mathcal { W } } { \arg \operatorname* { m i n } } \left\{ \left. h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } \left( g _ { s } - h _ { s } \right) , w \right. + \sum _ { s = 0 } ^ { t } c _ { s } ^ { ( t + 1 ) } D _ { \psi } ( w \mid w _ { s } ) \right\} ,\tag{MAO}
$$

where $w _ { 0 } \in \mathcal { W }$ is a fixed initial anchor and the weights $c _ { s } ^ { ( t + 1 ) }$ are non-negative. The additional flexibility in MAO-OMD allows us to separate two roles of the regularization weights: controlling the discrepancies $w _ { s } - w _ { t + 1 }$ arising from newly observed gradients, and managing the accumulation of Bregman-divergence terms across rounds. Indeed, combining Equation (4) with the standard mirror descent three-point inequality yields

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq \underbrace { \sum _ { t = 1 } ^ { T } \sum _ { s = 0 } ^ { t } c _ { s } ^ { ( t + 1 ) } \left[ D _ { \psi } ( u \mid w _ { s } ) - D _ { \psi } ( u \mid w _ { t + 1 } ) \right] } _ { \mathrm { M u l t i - a n c h o r ~ p o t e n t i a l } } + \underbrace { \sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \frac { \| g _ { s } - h _ { s } \| _ { * } ^ { 2 } } { 2 c _ { s } ^ { ( t + 1 ) } } } _ { \mathrm { P r e d i c t i o n - e r r o r } } ,\tag{7}
$$

provided that the weights corresponding to newly observed gradients are positive. We formalize these intuitions in Section 6, where we establish the general regret guarantee of MAO-OMD in Theorem 6.

Our main technical contribution is a weight-allocation strategy for choosing the weights $c _ { s } ^ { ( t + 1 ) }$ that controls the non-telescoping multi-anchor potential while balancing the prediction-error cost.

As we show in Section 5.1, this leads to regret guarantees that recover the minimax-optimal $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ dependence on delays, while exploiting predictable feedback to obtain sharper bounds in smooth games. In particular, when both players employ MAO-OMD, our analysis yields an $O ( q _ { \mathrm { m a x } } )$ social-regret guarantee.

## 5.1 Anchor weights for delayed feedback

We now instantiate the multi-anchor update (MAO) for delayed feedback. Our goal is to choose the regularization weights so that the multi-anchor potential in Equation (7) remains controlled, while retaining suficiently large weights for the anchors corresponding to newly observed gradients.

To understand the constraints on these weights, define $\begin{array} { r } { A _ { t } ^ { \mathrm { i n } } : = \sum _ { k = 0 } ^ { t - 1 } c _ { k } ^ { ( t ) } } \end{array}$ and $\begin{array} { r } { A _ { t } ^ { \mathrm { o u t } } : = \sum _ { k = t } ^ { T } c _ { t } ^ { ( k + 1 ) } } \end{array}$ Here, $A _ { t } ^ { \mathrm { i n } }$ is the total regularization weight used to compute $w _ { t } ,$ whereas $A _ { t } ^ { \mathrm { o u t } }$ is the total weight assigned to $w _ { t }$ as an anchor in subsequent updates. Rearranging the multi-anchor potential in Equation (7), and dropping a nonpositive terminal term, gives

$$
\sum _ { t = 1 } ^ { T } \sum _ { s = 0 } ^ { t } c _ { s } ^ { ( t + 1 ) } \left[ D _ { \psi } ( u | w _ { s } ) - D _ { \psi } ( u | w _ { t + 1 } ) \right] \leq A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) + \sum _ { t = 1 } ^ { T } ( A _ { t } ^ { \mathrm { o u t } } - A _ { t } ^ { \mathrm { i n } } ) D _ { \psi } ( u \mid w _ { t } ) .
$$

Thus, to control this potential, it sufices to ensure that $A _ { t } ^ { \mathrm { o u t } } \leq A _ { t } ^ { \mathrm { i n } }$ for every $t \in [ T ]$ . In other words, each iterate can be used as an anchor with total weight at most that assigned to it when it was computed.

The fixed reference point $w _ { 0 }$ plays a special role in this construction. Rather than being associated with any delayed gradient, it serves as an auxiliary anchor through which additional regularization weight can be introduced adaptively. Indeed, under the above constraints, the only remaining positive contribution to the multi-anchor potential is $A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) \le M A _ { 0 } ^ { \mathrm { o u t } }$ . This allows us to adaptively increase the available regularization weight as needed, without advance knowledge of the delays or prediction errors, provided that the total weight $A _ { 0 } ^ { \mathrm { o u t } }$ assigned to $w _ { 0 }$ can be kept suficiently small.

It remains to determine how the available weight should be distributed among the other anchors. The prediction-error term in Equation (7) favors assigning a large weight to $w _ { s }$ when its gradient $g _ { s }$ arrives. We therefore seek to preserve as much of the weight associated with each iterate $w _ { s }$ as possible until its feedback is observed, so as to set $c _ { s } ^ { ( s + d _ { s } + 1 ) }$ suficiently large while allowing portions of the weight available for $w _ { s }$ to be used in earlier updates. To formalize this idea, we introduce an allocation schedule, which specifies the fraction of regularization weight retained by each iterate over time.

Definition 1 (Allocation schedule). A nonnegative array $( \alpha _ { k } ^ { ( t ) } ) _ { 1 \leq k \leq t \leq T + 1 }$ is an allocation schedule if it satisfies the following three properties for all $t \in [ T ]$

(S1) monotonicity: $\alpha _ { t } ^ { ( k ) } \geq \alpha _ { t } ^ { ( k + 1 ) }$ for all $k = t , \ldots , T$

(S2) zero tail:

$$
\alpha _ { t } ^ { ( k ) } = 0 \ f o r \ a l l \ k \geq t + d _ { t } + 1 ,
$$

(S3) capacity:

$$
\begin{array} { r } { \sum _ { k = 1 } ^ { t } \alpha _ { k } ^ { ( t ) } = \sum _ { k \in \mathcal { A } _ { t } } \alpha _ { k } ^ { ( t ) } \le 1 . } \end{array}
$$

Given any allocation schedule, we also introduce the notion of remaining $w e i g h t ~ r _ { s } ^ { ( t ) }$ for some anchor $w _ { s }$ at time t. In particular, given $\gamma > 0$ and any nondecreasing sequence $0 = \eta _ { 1 } \leq \cdot \cdot \cdot \leq \eta _ { T + 1 }$ we define $r _ { s } ^ { ( t ) } : = ( \gamma + \eta _ { s } \alpha _ { s } ^ { ( t ) } ) \mathbb { I } \{ s \in \mathcal { A } _ { t } \}$ for every $1 \leq s \leq t \leq T + 1$ with $s \leq T ;$ note that we add $\gamma$ to guarantee strict positivity even when $\eta _ { s } \alpha _ { s } ^ { ( t ) } = 0$

Further let $Q _ { t } : = \operatorname* { m a x } _ { s \in [ t ] } q _ { s }$ for every $t \in [ T ]$ with $Q _ { 0 } = 0$ . We then introduce $\beta _ { t } : = \eta _ { t } + \gamma Q _ { t }$ as an inflated version of $\eta _ { t }$ to compensate for the incremental updates of $\eta _ { t }$ with no prior knowledge while simultaneously preserving monotonicity. Then, $\beta _ { t }$ efectively represents the amount of weight flowing through the subset [t]; i.e., it corresponds to the in-flow from the source 0 to [t] and, by the conservation of flow from enforcing $A _ { t } ^ { \mathrm { i n } } = A _ { t } ^ { \mathrm { o u t } }$ , to the out-flow from [t] to $\{ t + 1 , \ldots , T + 1 \}$ . To realize this flow, we define the regularization coeficients by releasing the weight no longer reserved at each anchor $w _ { s } ,$ assigning the unallocated remainder $r _ { s } ^ { ( t ) } - r _ { s } ^ { ( t + 1 ) }$ to $w _ { t + 1 }$ , and injecting the increment $\beta _ { t + 1 } - \beta _ { t }$ through w<sub>0</sub>:

$$
c _ { 0 } ^ { ( t + 1 ) } : = \beta _ { t + 1 } - \beta _ { t } , \qquad c _ { s } ^ { ( t + 1 ) } : = \bigg ( \beta _ { t } - \sum _ { k \in \mathcal { A } _ { t } } r _ { k } ^ { ( t ) } \bigg ) \mathbb { I } \{ s = t \} + r _ { s } ^ { ( t ) } - r _ { s } ^ { ( t + 1 ) } , \quad \forall s \in [ t ] ,\tag{8}
$$

where we define $\begin{array} { r } { \boldsymbol { b } _ { t } : = \beta _ { t } - \sum _ { k \in \mathcal { A } _ { t } } \boldsymbol { r } _ { k } ^ { ( t ) } } \end{array}$ as the remainder weight at t which we use on the anchor $w _ { t }$ when computing $w _ { t + 1 }$ to satisfy the corresponding flow-conservation constraint with respect to the subset [t]. Note that the extra weight $\beta _ { t + 1 } - \beta _ { t } = \eta _ { t + 1 } - \eta _ { t } + \gamma ( Q _ { t + 1 } - Q _ { t } )$ added from the default anchor $w _ { 0 }$ into $w _ { t + 1 }$ allows to adaptively increase the flow value (coming from the source $w _ { 0 } )$ from $\beta _ { t }$ to $\beta _ { t + 1 }$ , applying the correction with respect to $\eta _ { t + 1 }$ and $Q _ { t + 1 }$ due to the newly observed feedback.

Finally, to obtain the negative stability terms needed for fast regret rates in games, we introduce an additional anchor at the iterate $w _ { s _ { t + 1 } }$ used to form the optimistic hint. Specifically, we add the regularization term $\gamma D _ { \psi } ( w \mid w _ { s _ { t + 1 } } )$ to the update. Unlike the reference anchor $w _ { 0 }$ , whose role is to supply additional regularization weight, this anchor relates the new iterate to the action at which the predicted gradient $h _ { t + 1 } = g _ { s _ { t + 1 } }$ was computed. Its contribution yields the negative movement term $- \frac { \gamma } { 2 } \Vert w _ { t + 1 } - w _ { s _ { t + 1 } } \Vert ^ { 2 }$ as required by Theorem 1, while the additional comparator-divergence terms can be controlled using the fact that each observed gradient is selected as a hint at most once. While this always fits within our proposed multi-anchor OMD framework (by inflating each coeficient $c _ { s t } ^ { ( t ) } \ \mathrm { b y } \ \gamma )$ , we treat it separately for convenience.

All these design choices lead to the final definition of MAO-OMD in Algorithm 1.

Theorem 5. (Master bound) Let ψ be diferentiable and 1-strongly convex with respect to $\left\| \cdot \right\| .$ , with $\operatorname* { s u p } _ { u \in \mathcal { W } } D _ { \psi } ( u \mid w _ { 0 } ) \leq M$ for every $u \in \mathcal W$ . Let $\gamma > 0$ , let $( \alpha _ { k } ^ { ( t ) } ) _ { 1 \leq k \leq t \leq T + 1 }$ be an allocation schedule, and let $0 = \eta _ { 1 } \leq \cdot \cdot \cdot \leq \eta _ { T + 1 }$ . Then Algorithm 1 guarantees, for every u $\in \mathcal { W }$

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq 2 \gamma M q _ { \operatorname* { m a x } } + M \eta _ { T + 1 } + \sum _ { t = 1 } ^ { T } \frac { \Vert g _ { t } - h _ { t } \Vert _ { * } ^ { 2 } } { \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } } - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \Vert w _ { t + 1 } - w _ { s _ { t + 1 } } \Vert ^ { 2 } .
$$

We can perform an adaptive tuning of $\eta _ { t }$ using a technique in close resemblance to the recurrence proved for the sequence $( \Delta _ { t } ) _ { t \geq 1 }$ of the AdaFTRL algorithm [OP18, Lemmas 6 and $7 ]$ , or to the Stability-Penalty-Matching update rule for the sequence $( \beta _ { t } ) _ { t \geq 1 }$ from [ITH24]. In particular, we can rewrite the penalty term from Theorem 5 as a telescoping sum $\begin{array} { r } { M \eta _ { T + 1 } = M \sum _ { t } ( \eta _ { t + 1 } - \eta _ { t } ) } \end{array}$ and the stability term as $\begin{array} { r } { \sum _ { t } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } } = \sum _ { t } \sum _ { s \in \mathcal { N } _ { t } } \frac { \| g _ { s } - h _ { s } \| _ { * } ^ { 2 } } { \gamma + \eta _ { s } \alpha _ { s } ^ { ( t ) } } } \end{array}$ (by Lemma $7$ in the appendix). Then, one may combine them as

$$
M \eta _ { T + 1 } + \sum _ { t = 1 } ^ { T } \frac { \Vert g _ { t } - h _ { t } \Vert _ { * } ^ { 2 } } { \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } } = \sum _ { t = 1 } ^ { T } \left( M ( \eta _ { t + 1 } - \eta _ { t } ) + \sum _ { s \in \mathcal { N } _ { t } } \frac { \Vert g _ { s } - h _ { s } \Vert _ { * } ^ { 2 } } { \gamma + \eta _ { s } \alpha _ { s } ^ { ( t ) } } \right)
$$

Algorithm 1: MAO-OMD for delayed feedback   
Input: $w _ { 0 } \in \mathcal { W }$ , regularizer ψ, optimism weight $\gamma > 0$ , allocation schedule $( \alpha _ { s } ^ { ( t ) } ) _ { s \leq t }$   
learning rates $0 = \eta _ { 1 } \leq \cdot \cdot \cdot \leq \eta _ { T + 1 }$   
Initialize: $w _ { 1 } = w _ { 0 } , s _ { 1 } = 0 , h _ { 1 } = 0 , g _ { 0 } = 0 .$   
1 for $t = 1 , \dots , T$ do   
2 Play w<sub>t</sub> and observe $g _ { k }$ for every $k \in \mathcal { N } _ { t } .$   
3 Set $s _ { t + 1 } = \operatorname* { m i n } \mathcal { O } _ { t + 1 } \setminus \left\{ s _ { 1 } , \dots , s _ { t } \right\}$ (0 if empty) and $h _ { t + 1 } = g _ { s _ { t + 1 } } \ ( 0 \ \mathrm { i f } \ t = T )$   
4 Compute $\eta _ { t + 1 }$ and $\alpha _ { k } ^ { ( t + 1 ) }$ for $k \leq t ,$ and set $c _ { k } ^ { ( t + 1 ) }$ for $0 \leq k \leq t$ as in (8).   
5 Set $\begin{array} { r } { \ell _ { t } = h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } ( g _ { s } - h _ { s } ) } \end{array}$   
6 Update $\begin{array} { r } { \boldsymbol { w } _ { t + 1 } \in \arg \operatorname* { m i n } _ { \boldsymbol { w } \in \mathcal { W } } \Bigl \{ \langle \ell _ { t } , \boldsymbol { w } \rangle + \sum _ { s = 0 } ^ { t } c _ { s } ^ { ( t + 1 ) } D _ { \psi } ( \boldsymbol { w } \mid \boldsymbol { w } _ { s } ) + \gamma D _ { \psi } ( \boldsymbol { w } \mid \boldsymbol { w } _ { s _ { t + 1 } } ) \Bigr \} } \end{array}$

and optimize term-by-term (via equalization) the resulting sum to lead to the incremental update of the form

$$
\eta _ { t + 1 } = \eta _ { t } + \frac { 1 } { M } \sum _ { s \in \mathcal { N } _ { t } } \frac { \Vert g _ { s } - h _ { s } \Vert _ { * } ^ { 2 } } { \gamma + \eta _ { s } \alpha _ { s } ^ { ( t ) } } ,\tag{9}
$$

which only uses information observed by the learner at the end of round t. This tuning then works as desired by Lemma 8 in the appendix, provided $\gamma \geq 2 G / \sqrt { M }$ and the allocation schedule guarantees suficiently small values of $\alpha _ { t } ^ { ( t + d _ { t } ) }$

We now define the last missing piece for a concrete instantiation of MAO-OMD for delayed feedback: a concrete allocation schedule meeting all the criteria mentioned so far. To do so, we make use of the delay-related notion of congestion introduced in Section 2. We can define our allocation schedule $( \alpha _ { s } ^ { ( t ) } ) _ { s \leq t }$ via the congestion parameters $\kappa _ { s , t }$ as follows:

$$
\alpha _ { s } ^ { ( t ) } : = \frac { \mathbb { I } \{ s \in \mathcal { A } _ { t } \} } { \kappa _ { s , t } } \qquad \forall 1 \leq s \leq t \leq T .\tag{10}
$$

This schedule satisfies the conditions of Definition 1: the allocated fractions are nonincreasing, vanish once feedback arrives, and sum to at most one since $\kappa _ { s , t } \ \geq \ q _ { t }$ for every $s \in \mathcal A _ { t }$ . These choices together yield the result of Theorem 1, restated here with complete details.

Theorem 1. Let Assumption 1 hold and let ψ : $\mathcal { W }  \mathbb { R }$ and $w _ { 0 } \in \mathcal { W }$ satisfy Assumption 2. Consider the allocation schedule $( \alpha _ { s } ^ { ( t ) } ) _ { s \leq t }$ in (10) and the sequence $0 = \eta _ { 1 } \leq \cdot \cdot \cdot \leq \eta _ { T + 1 }$ following (9). Then Algorithm 1 with $\gamma = \operatorname* { m a x } \{ 4 L , 2 G / \sqrt { M } \}$ guarantees, for every $u \in \mathcal W$ 2

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq ( 8 L M + 4 G \sqrt { M } ) q _ { \operatorname* { m a x } } + 2 \sqrt { 2 M \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } - L \sum _ { t = 1 } ^ { T } \| w _ { t } - w _ { s _ { t } } \| ^ { 2 } .
$$

## 6 Regret analysis of MAO-OMD

In this section, we develop the regret analysis of MAO-OMD, culminating in the proof of Theorem 1. We proceed in three steps. First, in Section 6.1, we establish a general regret bound for mirror descent updates with multiple anchors, without imposing any particular structure on the feedback. Next, in Section 6.2, we combine this bound with the delays-to-feedback decomposition of Lemma 1 to derive an optimistic regret guarantee under arbitrary delays Theorem 6.

This bound highlights the two main ingredients of our analysis: controlling the non-telescoping Bregman-divergence terms arising from multiple anchors, while retaining the negative stability terms needed for fast regret rates. Finally, in Section 6.3, we show how the regularization weights and hint-selection rule of MAO-OMD control these contributions, establishing the master bound of Theorem 5 and, ultimately, the guarantee of Theorem 1 in Section 6.4.

## 6.1 Results for multi-anchor online mirror descent

The following lemma generalizes the usual three-point mirror descent inequality to the multi-anchor case.

Lemma 2. Let $\ell \in \mathbb { R } ^ { n }$ , let $z _ { 1 } , \ldots , z _ { k } \in { \mathcal { W } }$ , and let $\alpha _ { 1 } , \ldots , \alpha _ { k } \ge 0$ . Let $\psi : \mathcal { W } \to \mathbb { R }$ be a convex diferentiable function. Define $\textstyle A : = \sum _ { i = 1 } ^ { k } \alpha _ { i }$ and

$$
w \in \underset { x \in \mathcal { W } } { \arg \operatorname* { m i n } } \left\{ \langle \ell , x \rangle + \sum _ { i = 1 } ^ { k } \alpha _ { i } D _ { \psi } ( x \mid z _ { i } ) \right\} .
$$

Then, for every $u \in \mathcal W$

$$
\langle \ell , w - u \rangle \leq \sum _ { i = 1 } ^ { k } \alpha _ { i } \big ( D _ { \psi } ( u \mid z _ { i } ) - D _ { \psi } ( w \mid z _ { i } ) \big ) - A D _ { \psi } ( u \mid w ) \ .
$$

Proof. Fix any $u \in \mathcal W$ . Since $\mathcal { W }$ is convex and w minimizes the diferentiable objective, the firstorder optimality condition gives

$$
\left. \ell + \sum _ { i = 1 } ^ { k } \alpha _ { i } \bigl ( \nabla \psi ( w ) - \nabla \psi ( z _ { i } ) \bigr ) , w - u \right. \leq 0 .
$$

Rearranging and using the three-point identity for Bregman divergences $\begin{array} { r } { \left. \nabla \psi ( z ) - \nabla \psi ( w ) , w - u \right. = } \end{array}$ $D _ { \psi } ( u | z ) - D _ { \psi } ( u | w ) - D _ { \psi } ( w | z )$ yields

$$
\begin{array} { l } { \displaystyle \langle \ell , w - u \rangle \leq \left. \displaystyle \sum _ { i = 1 } ^ { k } \alpha _ { i } \big ( \nabla \psi ( z _ { i } ) - \nabla \psi ( w ) \big ) , w - u \right. } \\ { = \displaystyle \sum _ { i = 1 } ^ { k } \alpha _ { i } \langle \nabla \psi ( z _ { i } ) - \nabla \psi ( w ) , w - u \rangle } \\ { = \displaystyle \sum _ { i = 1 } ^ { k } \alpha _ { i } \big ( D _ { \psi } ( u \mid z _ { i } ) - D _ { \psi } ( u \mid w ) - D _ { \psi } ( w \mid z _ { i } ) \big ) } \\ { = \displaystyle \sum _ { i = 1 } ^ { k } \alpha _ { i } \big ( D _ { \psi } ( u \mid z _ { i } ) - D _ { \psi } ( w \mid z _ { i } ) \big ) - A D _ { \psi } ( u \mid w ) , } \end{array}
$$

proving the claim.

The following proposition bounds the one-step-ahead quantity $\textstyle \sum _ { t = 1 } ^ { T } \langle \ell _ { t } , w _ { t + 1 } - u \rangle$ , where $w _ { t + 1 }$ is computed using $\ell _ { t }$ . We refer to this as a cheating regret bound. For $_ \mathrm { M A O - O M D }$ , we take $\begin{array} { r } { \ell _ { t } = h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } ( g _ { s } - h _ { s } ) } \end{array}$ and apply Lemma 1 to recover a bound on the linearized regret of the played actions. The correction term in that decomposition yields the prediction-error terms involving $\| g _ { s } - h _ { s } \| _ { * } ^ { 2 }$

Proposition 2 (Cheating regret). Fix $T \in \mathbb { N } _ { + }$ and $w _ { 0 } \in \mathcal { W }$ . Let $\ell _ { 0 } , \ldots , \ell _ { T } \in \mathbb { R } ^ { n }$ with $\ell _ { 0 } = 0$ and, for each $t \in \{ 0 , \ldots , T \}$ and each $0 \leq k \leq t ,$ , let $c _ { k } ^ { ( t + 1 ) } \geq 0$ . For every $t \in \{ 0 , \ldots , T \}$ define

$$
w _ { t + 1 } \in \underset { w \in \mathcal { W } } { \arg \operatorname* { m i n } } \left\{ \langle \ell _ { t } , w \rangle + \sum _ { k = 0 } ^ { t } c _ { k } ^ { ( t + 1 ) } D _ { \psi } ( w \mid w _ { k } ) \right\} .
$$

Also define $\begin{array} { r } { A _ { t } ^ { \mathrm { i n } } : = \sum _ { k = 0 } ^ { t - 1 } c _ { k } ^ { ( t ) } } \end{array}$ and $\begin{array} { r } { A _ { t } ^ { \mathrm { o u t } } : = \sum _ { k = t } ^ { T } c _ { t } ^ { ( k + 1 ) } } \end{array}$ for each $t \in \{ 0 , . . . , T + 1 \}$ , with empty sums equal to zero. Then, for every $u \in \mathcal W$

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle \ell _ { t } , w _ { t + 1 } - u \rangle \le A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( \boldsymbol { u } \mid w _ { 0 } ) - A _ { T + 1 } ^ { \mathrm { i n } } D _ { \psi } ( \boldsymbol { u } \mid w _ { T + 1 } ) + \sum _ { t = 1 } ^ { T } \bigl ( A _ { t } ^ { \mathrm { o u t } } - A _ { t } ^ { \mathrm { i n } } \bigr ) D _ { \psi } ( \boldsymbol { u } \mid w _ { t } ) } \\ { \displaystyle - \sum _ { t = 0 } ^ { T } \sum _ { k = 0 } ^ { t } c _ { k } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { k } ) . } \end{array}
$$

Proof. Fix any $u \in \mathcal W$ . For each $t \in \{ 0 , \ldots , T \}$ , apply Lemma 2 to the update defining $w _ { t + 1 }$ , with loss vector $\ell _ { t } .$ , anchors $( w _ { k } ) _ { k = 0 } ^ { t } .$ , and weights $( c _ { k } ^ { ( t + 1 ) } ) _ { k = 0 } ^ { t }$ . Since the weights sum to $A _ { t + 1 } ^ { \mathrm { i n } }$ , we obtain

$$
\langle \ell _ { t } , w _ { t + 1 } - u \rangle \leq \sum _ { k = 0 } ^ { t } c _ { k } ^ { ( t + 1 ) } \big ( D _ { \psi } ( u \mid w _ { k } ) - D _ { \psi } ( w _ { t + 1 } \mid w _ { k } ) \big ) - A _ { t + 1 } ^ { \mathrm { i n } } D _ { \psi } ( u \mid w _ { t + 1 } ) ~ .\tag{11}
$$

When these inequalities are summed over $t = 0 , \ldots , T$ , the anchor $w _ { k }$ contributes a comparator term at each later update $t = k , \ldots , T$ . Its total coeficient is therefore $\textstyle \sum _ { t = k } ^ { T } c _ { k } ^ { ( t + 1 ) } = A _ { k } ^ { \mathrm { o u t } }$ . More precisely, exchanging the order of summation gives

$$
\sum _ { t = 0 } ^ { T } \sum _ { k = 0 } ^ { t } c _ { k } ^ { ( t + 1 ) } D _ { \psi } ( u \mid w _ { k } ) = \sum _ { k = 0 } ^ { T } \left( \sum _ { t = k } ^ { T } c _ { k } ^ { ( t + 1 ) } \right) D _ { \psi } ( u \mid w _ { k } ) = A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) + \sum _ { t = 1 } ^ { T } A _ { t } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { t } ) .\tag{12}
$$

Substituting this identity into the sum of the one-step inequalities from Equation (11) and collecting the comparator terms yields

(ℓ<sub>0</sub> = 0)

$$
\begin{array} { r l } { \displaystyle \sum _ { i = 1 } ^ { N } \| \hat { x } _ { i } , \boldsymbol { \mathrm { R } } _ { i + 1 } - \boldsymbol { \mathrm { R } } _ { i } ^ { * } \rangle - } & { \displaystyle \sum _ { i = 0 } ^ { N } \hat { \theta } _ { i } , \boldsymbol { \mathrm { R } } _ { i } \boldsymbol { \mathrm { R } } _ { i + 1 } - \boldsymbol { \mathrm { R } } _ { i } ^ { * } } \\ { \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { \ell = 0 } ^ { N } \hat { \theta } _ { i } , } & { \qquad } \\ & { \displaystyle \leq \sum _ { i = 0 } ^ { N } \sum _ { \ell = 1 } ^ { N } \hat { \theta } _ { i } , \boldsymbol { \hat { \sigma } } _ { \ell } ( u _ { i } | w _ { k } ) - \sum _ { k = 0 } ^ { N } \hat { A } _ { k + 1 } ^ { * \mu } \mathcal { P } _ { \phi } ( u _ { i } | w _ { \ell + 1 } ) } \\ & { \quad - \displaystyle \sum _ { k = 0 } ^ { N } \hat { \phi } _ { i } , \quad \hat { \sigma } _ { \ell } ( u _ { k + 1 } ^ { * \mu } ) D _ { \phi } ( u _ { \ell + 1 } | w _ { k } ) , } \\ & { \quad - \displaystyle \sum _ { k = 0 } ^ { N } \hat { \theta } _ { i } , \quad \hat { \sigma } _ { \ell } ( u _ { k + 1 } ^ { * \mu } | w _ { \ell } ) + \sum _ { k = 0 } ^ { N } A _ { k + 1 } ^ { * \mu } D _ { \phi } ( u _ { \ell } | w _ { k } ) - \sum _ { k = 0 } ^ { N } A _ { k + 1 } ^ { * \mu } D _ { \phi } ( u _ { \ell } | w _ { k + 1 } \rangle ) } \\ & { = \displaystyle \sum _ { i = 0 } ^ { N } \sum _ { \ell = 0 } ^ { N } \hat { \sigma } _ { \ell } ^ { ( k ) } \| D _ { \phi } ( u _ { \ell + 1 } | w _ { k } ) \rangle } \\ &  \quad - \displaystyle \sum _ { k = 0 } ^ { N } \ \end{array}\tag{by (11)}
$$

(by (12))

$$
- \sum _ { t = 0 } ^ { T } \sum _ { k = 0 } ^ { t } c _ { k } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { k } ) ~ ,
$$

This is the claimed bound.

## 6.2 Optimistic regret bound under delayed feedback

To apply Proposition 2 to delayed feedback, we first restate the regret decomposition from Lemma 1 and provide its proof. The lemma is restated below for convenience.

Lemma 1 (Delays-to-feedback decomposition). For any $u \in \mathcal W$ and any sequence $h _ { 1 } , \ldots , h _ { T + 1 } \in$ $\mathbb { R } ^ { n } \ o f$ hints with $h _ { 1 } = h _ { T + 1 } = 0$ 2

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle = \sum _ { t = 1 } ^ { T } \biggl \langle h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } \bigl ( g _ { s } - h _ { s } \bigr ) , w _ { t + 1 } - u \biggr \rangle + \sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \bigl \langle g _ { s } - h _ { s } , w _ { s } - w _ { t + 1 } \bigr \rangle .\tag{4}
$$

Proof of Lemma 1. Fix any $u \in \mathcal W$ and any sequence $h _ { 1 } , \ldots , h _ { T + 1 } \in \mathbb { R } ^ { n }$ with $h _ { 1 } = h _ { T + 1 } = 0$ . By adding and subtracting $\langle h _ { t } , w _ { t } - u \rangle$ for each $t \in [ T ]$ , one obtains

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle = \displaystyle \sum _ { t = 1 } ^ { T } \langle h _ { t } , w _ { t } - u \rangle + \displaystyle \sum _ { t = 1 } ^ { T } \langle g _ { t } - h _ { t } , w _ { t } - u \rangle } \\ { \displaystyle \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \langle h _ { t + 1 } , w _ { t + 1 } - u \rangle + \displaystyle \sum _ { t = 1 } ^ { T } \displaystyle \sum _ { s \in \mathcal { N } _ { t } } \langle g _ { s } - h _ { s } , w _ { s } - u \rangle ~ , } \end{array}
$$

where the second identity uses the fact that $h _ { 1 } = h _ { T + 1 } = 0$ and rearranges the terms of the second sum by permuting and grouping them according to their feedback arrival time. Focusing on the second sum, we can further add and subtract $\begin{array} { r } { \sum _ { s \in \mathcal { N } _ { t } } \big < g _ { s } - h _ { s } , w _ { t + 1 } - u \big > } \end{array}$ for each $t \in [ T ]$ to obtain

$$
\sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \left. g _ { s } - h _ { s } , w _ { s } - u \right. = \sum _ { t = 1 } ^ { T } \left. \sum _ { s \in \mathcal { N } _ { t } } \left( g _ { s } - h _ { s } \right) , w _ { t + 1 } - u \right. + \sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \left. g _ { s } - h _ { s } , w _ { s } - w _ { t + 1 } \right.
$$

Combining the two math displays proves the claim.

Now using the regret decomposition of Lemma 1 and the generic regret bound in Proposition $^ { 2 , }$ we can easily derive the following regret guarantee for the delayed feedback setting, by taking $\begin{array} { r } { \ell _ { t } = h _ { t + 1 } + \sum _ { s \in \mathcal { N } _ { t } } ( g _ { s } - h _ { s } ) } \end{array}$

Theorem 6 (Regret of the multi-anchor optimistic update rule). Fix hints $( h _ { t } ) _ { t = 1 } ^ { T + 1 }$ from $\mathbb { R } ^ { n }$ with $h _ { 1 } = h _ { T + 1 } = 0$ and anchor weights $( c _ { k } ^ { ( t + 1 ) } ) _ { 0 \leq k \leq t \leq T }$ from $[ 0 , \infty )$ with $c _ { 0 } ^ { ( 1 ) } > 0$ and $c _ { t } ^ { ( t + d _ { t } + 1 ) } > 0$ for every $t \in [ T ]$ , and let $w _ { 1 } , \dots , w _ { T + 1 }$ be given by (MAO) from $w _ { 0 } \in \mathcal { W }$ , where $\psi : \mathcal { W } \to \mathbb { R } _ { \geq 0 }$ is 1-strongly convex w.r.t. ∥·∥. Then, for every $u \in \mathcal W$ ，

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle } & { \le A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } \bigl ( A _ { t } ^ { \mathrm { o u t } } - A _ { t } ^ { \mathrm { i n } } \bigr ) D _ { \psi } ( u \mid w _ { t } ) + \displaystyle \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { c _ { t } ^ { ( t + d _ { t } + 1 ) } } } \\ & { \quad \quad - \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { t = 0 } ^ { T } \displaystyle \sum _ { k = 0 } ^ { t } c _ { k } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { k } ) . } \end{array}\tag{13}
$$

To understand the bound, let us recall the analysis of a standard optimistic mirror descent update of the form $\begin{array} { r } { { w } _ { t + 1 } = \arg \operatorname* { m i n } _ { w \in \mathcal { W } } \langle g _ { t } - h _ { t } + h _ { t + 1 } , w \rangle + \lambda _ { t + 1 } D _ { \psi } ( w \mid w _ { t } ) } \end{array}$ , applied in a standard (undelayed) OCO problem. With $h _ { 1 } = h _ { T + 1 } = 0$ , the regret guarantee can be shown to be (see, $\mathrm { e . g . , [ O r a 2 6 ] ) }$ :

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq \sum _ { t = 1 } ^ { T } \lambda _ { t + 1 } D _ { \psi } ( u \mid w _ { t } ) - \lambda _ { t + 1 } D _ { \psi } ( u \mid w _ { t + 1 } ) + \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { 2 \lambda _ { t + 1 } } } } \\ & { } & { = \sum _ { t = 1 } ^ { T } \big ( \lambda _ { t } D _ { \psi } ( u \mid w _ { t } ) - \lambda _ { t + 1 } D _ { \psi } ( u \mid w _ { t + 1 } ) \big ) + \sum _ { t = 1 } ^ { T } ( \lambda _ { t + 1 } - \lambda _ { t } ) D _ { \psi } ( u \mid w _ { t } ) + \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { 2 \lambda _ { t + 1 } } } \\ & { } & { = \lambda _ { 1 } D _ { \psi } ( u \mid w _ { 1 } ) + \sum _ { t = 1 } ^ { T } ( \lambda _ { t + 1 } - \lambda _ { t } ) D _ { \psi } ( u \mid w _ { t } ) + \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { 2 \lambda _ { t + 1 } } , ~ ( 1 4 ) } \end{array}
$$

where the last line observes that the first summation telescopes as

$$
\sum _ { t = 1 } ^ { T } \lambda _ { t } D _ { \psi } ( u \mid w _ { t } ) - \lambda _ { t + 1 } D _ { \psi } ( u \mid w _ { t + 1 } ) = \lambda _ { 1 } D _ { \psi } ( u \mid w _ { 1 } ) - \lambda _ { T + 1 } D _ { \psi } ( u \mid w _ { T + 1 } ) \leq \lambda _ { 1 } D _ { \psi } ( u \mid w _ { 1 } ) ~ .
$$

Now, assuming the uniform bound $D _ { \psi } ( u \mid w ) \leq M$ for all $u , w \in \mathcal { W }$ , one typically chooses a sequence such as $\begin{array} { r } { \lambda _ { t } = \frac { G } { \sqrt { M } } \sqrt { t } } \end{array}$ , making the remaining Bregman term telescope as well:

$$
\sum _ { t = 1 } ^ { T } ( \lambda _ { t + 1 } - \lambda _ { t } ) D _ { \psi } ( u \mid w _ { t } ) \leq M \sum _ { t = 1 } ^ { T } ( \lambda _ { t + 1 } - \lambda _ { t } ) = M ( \lambda _ { T + 1 } - \lambda _ { 1 } ) \leq \lambda _ { T + 1 } M = O \left( G { \sqrt { M T } } \right) ~ ,
$$

which is minimax optimal in the undelayed setting. However, notice that the analogous term in Theorem 6 is instead

$$
\sum _ { t = 1 } ^ { T } ( A _ { t } ^ { \mathrm { o u t } } - A _ { t } ^ { \mathrm { i n } } ) D _ { \psi } ( u \mid w _ { t } ) \ ,
$$

where $\begin{array} { r } { A _ { t } ^ { \mathrm { o u t } } = \sum _ { k = t } ^ { T } c _ { t } ^ { ( k + 1 ) } } \end{array}$ and $\begin{array} { r } { A _ { t } ^ { \mathrm { i n } } = \sum _ { k = 0 } ^ { t - 1 } c _ { k } ^ { ( t ) } } \end{array}$ , which does not necessarily telescope. Controlling this term is what requires a carefully designed anchor weight schedule, such as the one in Section 5.1. On the other hand, notice that we also obtain a variance term which is significantly more powerful than the one in the usual approach: rather than $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \frac { \Vert g _ { t } - h _ { t } \Vert _ { * } ^ { 2 } } { 2 \lambda _ { t + 1 } } } \end{array}$ , MAO-OMD obtains an optimistic prediction-error penalty of

$$
\sum _ { t = 1 } ^ { T } \frac { \lVert g _ { t } - h _ { t } \rVert _ { * } ^ { 2 } } { 2 c _ { t } ^ { ( t + d _ { t } + 1 ) } } ~ .
$$

In particular, the coeficient $c _ { t } ^ { ( t + d _ { t } + 1 ) }$ is one which is set at the future time when the feedback from round t is observed, rather than on round $t + 1$ . This is what enables us to set the anchor weight schedule in such a way that the desired

$$
O \left( \sqrt { \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } \right) = O \left( \sqrt { \operatorname* { m i n } \left\{ T + d _ { \mathrm { t o t } } , \ q _ { \mathrm { m a x } } \sum _ { t = 1 } ^ { T } \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } \right\} } \right)
$$

bound is obtained, which is key to our main results in Section 3.

Proof of Theorem 6. The proof follows the argument sketched in Section 5. Apply Lemma 1 followed by Proposition 2 to get

$$
\begin{array} { r l } { \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } a _ { j } w _ { i } = \frac { 1 } { n - 1 } \sum _ { k = 1 } ^ { n - 1 } \bigg ( a _ { j + 1 } + \sum _ { k = 1 } ^ { n } ( a _ { j } - b _ { k + 1 } ) w _ { i + 1 } - \bigg ) ^ { 1 } \displaystyle + \sum _ { i = 1 + 1 } ^ { n } \sum _ { j = 1 } ^ { n } \sum _ { k = 1 } ^ { n } b _ { i j } w _ { i } - w _ { k + 1 } , } \\ { \displaystyle \leq A _ { j } ^ { ( n ) } D _ { j } ( u _ { i } ) ( w _ { 1 } ) w _ { 1 } ) - A _ { i - 1 } ^ { ( n ) } D _ { j } ( u _ { i } ) ( w _ { 1 } ) w _ { 1 } + \displaystyle \sum _ { i = 1 } ^ { n } \bigg ( A _ { i } ^ { ( n ) } - A _ { i } ^ { ( n ) } \bigg ) D _ { j } ( u _ { i } ) ( w _ { 1 } ) w _ { 1 } } \\ { \displaystyle \qquad - \sum _ { k = 1 } ^ { n } \sum _ { k = 1 } ^ { n } c _ { k } ^ { ( n + 1 ) } D _ { j } ( w _ { k + 1 } ) ( w _ { k + 1 } ) \displaystyle + \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { n } \bigg ( A _ { i } ^ { ( n ) } - h _ { i } , w _ { i } - w _ { k + 1 } \bigg ) } \\ { \displaystyle \leq A _ { j } ^ { ( n ) } D _ { j } ( u _ { i } ) ( w _ { 1 } ) - A _ { i } ^ { ( n ) } D _ { j } ( u _ { i } ) w _ { 1 } - \displaystyle \sum _ { i = 1 } ^ { n } ( A _ { i } ^ { ( n ) } - A _ { i } ^ { ( n ) } ) D _ { j } ( w _ { 1 } ) w _ { 1 } } \\  \displaystyle - \frac { 1 } { 2 } \sum _ { k = 1 } ^ { n } \sum _ { k = 0 } ^ { n } c _ { k } ^ { ( n + 1 ) } D _ { j } ( w _ { 1 } ) w _ { 1 } + \displaystyle \sum _  \end{array}
$$

where the last line drops a non-positive factor − $\begin{array} { r } { \cdot \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T } \sum _ { k \in \{ 0 , \dots , t \} \backslash \mathcal { N } _ { t } } c _ { s } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { s } ) \leq 0 } \end{array}$ . Now by 1-strong convexity of ψ with respect to ∥·∥, we have $D _ { \psi } ( x \mid y ) \geq { \textstyle { \frac { 1 } { 2 } } } \| x - y \| ^ { 2 }$ for $x , y \in \mathcal { W }$ , so the terms in the last line can be bound as

$$
\begin{array} { l } { \displaystyle ( * ) \sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \frac { \| g _ { s } - h _ { s } \| _ { * } ^ { 2 } } { c _ { s } ^ { ( t + 1 ) } } + \frac { 1 } { 4 } c _ { s } ^ { ( t + 1 ) } \| w _ { t + 1 } - w _ { s } \| ^ { 2 } - \frac { 1 } { 2 } c _ { s } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { s } ) } \\ { \displaystyle \quad \leq \sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \frac { \| g _ { s } - h _ { s } \| _ { * } ^ { 2 } } { c _ { s } ^ { ( t + 1 ) } } + \frac { 1 } { 4 } c _ { s } ^ { ( t + 1 ) } \| w _ { t + 1 } - w _ { s } \| ^ { 2 } - \frac { 1 } { 4 } c _ { s } ^ { ( t + 1 ) } \| w _ { t + 1 } - w _ { s } \| ^ { 2 } } \\ { \displaystyle \quad = \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { c _ { t } ^ { ( t + d _ { t } + 1 ) } } , } \end{array}
$$

where the final equality uses Lemma 7. Hence, plugging this back in above and bounding the remaining divergence with respect to $w _ { T + 1 }$ by $A _ { T + 1 } ^ { \mathrm { i n } } D _ { \psi } ( u \mid w _ { T + 1 } ) \geq 0$ yields the stated bound:

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \le A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) + \sum _ { t = 1 } ^ { T } \bigl ( A _ { t } ^ { \mathrm { o u t } } - A _ { t } ^ { \mathrm { i n } } \bigr ) D _ { \psi } ( u \mid w _ { t } ) + \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { c _ { t } ^ { ( t + d _ { t } + 1 ) } } } \\ { \displaystyle \qquad - \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T } \sum _ { k = 0 } ^ { t } c _ { k } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { k } ) . } \end{array}
$$

## 6.3 Proof of the Master Bound (Theorem 5)

We now turn to the proof of the master bound of Theorem 5, which holds for any valid allocation schedule and is the main ingredient for proving our main result in Theorem 1. We first establish two auxiliary results controlling the additional regularization toward the hint-source iterates

in Lemmas 3 and 4. We then verify that the allocation of regularization weights introduced in Section 5.1 satisfies the properties required for the general regret bound in Theorem 6, yielding the master bound of Theorem 5.

We begin by bounding the number of rounds with no available hint source.

Lemma 3. Consider any choice $o f s _ { 1 } , \ldots , s _ { T + 1 }$ such that, for every $t \in [ T + 1 ] , s _ { t } \in \mathcal { O } _ { t } \backslash \mathcal { S } _ { t }$ whenever $O _ { t } \setminus S _ { t } \ne \emptyset$ and $s _ { t } = 0$ otherwise, where $S _ { t } : = \{ s _ { 1 } , \ldots , s _ { t - 1 } \}$ . Then, $| \{ t \in [ T ] : s _ { t + 1 } = 0 \} | \leq \sigma$ max and $| \{ t \in [ T ] : s _ { t } = 0 \} | \leq q _ { \operatorname* { m a x } }$

Proof. Let $\mathcal { T } _ { t } : = \mathcal { O } _ { t } \ : \backslash \ : S _ { t }$ and define $\begin{array} { r } { \nu _ { t } : = \sum _ { k = 1 } ^ { t } \mathbb { I } \left\{ s _ { k + 1 } = 0 \right\} } \end{array}$ as the value of the partial sum of indicators for $t \in [ T ]$ , and $\nu _ { 0 } : = 0$ . Observe that, for every $t \geq 2$

$$
| \mathcal { I } _ { t } | = | \mathcal { O } _ { t } \setminus S _ { t } | = | \mathcal { O } _ { t } | - | \mathcal { O } _ { t } \cap S _ { t } | = t - 1 - \sigma _ { t } - \sum _ { k = 1 } ^ { t - 1 } \mathbb { I } \{ s _ { k } > 0 \} = \sum _ { k = 1 } ^ { t - 1 } \mathbb { I } \{ s _ { k } = 0 \} - \sigma _ { t } = \nu _ { t - 2 } - \sigma _ { t } + 1 ,
$$

where we used $\begin{array} { r } { | \mathcal { O } _ { t } | = t - 1 - \sigma _ { t } , \nu _ { t - 2 } = \sum _ { k = 2 } ^ { t - 1 } \mathbb { I } \{ s _ { k } = 0 \} } \end{array}$ , and the fact that $s _ { 1 } = 0$ . Thus, for $t \in [ T ]$

$$
s _ { t + 1 } = 0 \iff \mathcal { J } _ { t + 1 } = \emptyset \iff \sigma _ { t + 1 } = \nu _ { t - 1 } + 1 .
$$

Since $0 = \nu _ { 0 } \le \cdots \le \nu _ { T }$ is monotonically non-decreasing, $\nu _ { 0 } = 0$ , and $\nu _ { t } - \nu _ { t - 1 } = \mathbb { I } \{ s _ { t + 1 } = 0 \} \in$ $\{ 0 , 1 \}$ , it means that there are $\nu _ { T }$ rounds $t \in [ T ]$ such that $s _ { t + 1 } = 0$ . Now, let $t ^ { * } =$ min $\{ t \in [ T ]$ : $\nu _ { t } = \nu _ { T } \}$ and note that $\nu _ { t ^ { * } } = \nu _ { t ^ { * } - 1 } + 1$ , hence $s _ { t ^ { * } + 1 } = 0$ . This means that $\sigma _ { t ^ { * } + 1 } = \nu _ { t ^ { * } - 1 } + 1 = \nu _ { t ^ { * } }$ Therefore,

$$
\left| \left\{ t \in [ T ] : s _ { t + 1 } = 0 \right\} \right| = \sum _ { t = 1 } ^ { T } \mathbb { I } \left\{ s _ { t + 1 } = 0 \right\} = \nu _ { T } = \nu _ { t ^ { * } } = \sigma _ { t ^ { * } + 1 } \leq \sigma _ { \operatorname* { m a x } } \ .
$$

The second bound then comes from observing that $s _ { 1 } = 0 \neq s _ { T + 1 }$ , concluding the proof. □

Lemma 3 shows that the initial anchor $w _ { 0 }$ is selected as a hint source at most $\sigma _ { \mathrm { m a x } }$ times after the first round. Since every other hint source is selected at most once, we can use this property to control the additional Bregman-divergence terms arising from regularization toward the hintsource iterates. In particular, the following lemma shows that their total contribution is bounded by $M \sigma _ { \mathrm { m a x } }$

Lemma 4. Let Assumption 2 hold and consider any sequence $s _ { 1 } , \ldots , s _ { T + 1 }$ satisfying the assumptions of Lemma 3. Then, for every $u \in \mathcal { W } , \sum _ { t = 1 } ^ { T } \bigl ( D _ { \psi } ( u \mid w _ { s _ { t + 1 } } ) - D _ { \psi } ( u \mid w _ { t } ) \bigr ) \leq M \sigma _ { \operatorname* { m a x } }$

Proof. Fix any $u \in \mathcal W$ . Observing that the map $t \mapsto s _ { t + 1 } \in [ t ]$ is injective over $\{ t \in [ T ] : s _ { t + 1 } > 0 \}$ , we can show that

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { T } D _ { \psi } ( u \mid w _ { s _ { t + 1 } } ) \leq | \{ t \in [ T ] : s _ { t + 1 } = 0 \} | \cdot D _ { \psi } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } D _ { \psi } ( u \mid w _ { s _ { t + 1 } } ) \mathbb { I } \{ s _ { t + 1 } > 0 \} } & { { } } \\ { \leq | \{ t \in [ T ] : s _ { t + 1 } = 0 \} | \cdot D _ { \psi } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } D _ { \psi } ( u \mid w _ { t } ) } & { { } } \\ { \leq \sigma _ { \operatorname* { m a x } } D _ { \psi } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } D _ { \psi } ( u \mid w _ { t } ) , } & { { } } \end{array}
$$

with the last inequality using Lemma 3. Therefore,

$$
\sum _ { t = 1 } ^ { T } ( D _ { \psi } ( u \mid w _ { s _ { t + 1 } } ) - D _ { \psi } ( u \mid w _ { t } ) ) = \sum _ { t = 1 } ^ { T } D _ { \psi } ( u \mid w _ { s _ { t + 1 } } ) - \sum _ { t = 1 } ^ { T } D _ { \psi } ( u \mid w _ { t } ) \leq \sigma _ { \operatorname* { m a x } } D _ { \psi } ( u \mid w _ { 0 } ) \leq M \sigma _ { \operatorname* { m a x } } ,
$$

where the last inequality follows by Assumption 2.

We now turn to the regularization coeficients defined in Equation (8). As discussed in Section 5.1, the main challenge is to choose these coeficients so that they control the prediction-error terms $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \frac { \Vert g _ { t } - h _ { t } \Vert ^ { 2 } } { c _ { s } ^ { ( t + 1 ) } } } \end{array}$ without introducing an additional penalty from the multi-anchor potential. The following lemma establishes two key properties of our construction: the coeficients are nonnegative (so they define valid regularization weights) and the total incoming and outgoing weights of each iterate coincide. The latter property ensures that the corresponding Bregman-divergence terms cancel in the regret bound of Theorem 6

Lemma 5. Given any $\gamma > 0$ , any sequence $0 = \eta _ { 1 } \leq \cdots \leq \eta _ { T + 1 }$ , and any allocation schedule $( \alpha _ { k } ^ { ( t ) } ) _ { 1 \leq s \leq t \leq T }$ , the corresponding coeficients $( c _ { s } ^ { ( t + 1 ) } ) _ { 0 \leq s \leq t \leq T }$ from Equation (8) satisfy:

(i) $c _ { s } ^ { ( t + 1 ) } \geq 0$ for every $0 \leq s \leq t \leq T$

(ii) $A _ { t } ^ { \mathrm { i n } } = A _ { t } ^ { \mathrm { o u t } }$ for every $t \in [ T ]$

Proof. Recall that, for every $t \in [ T ]$

$$
\beta _ { t } = \eta _ { t } + \gamma Q _ { t } \ , \qquad r _ { s } ^ { ( t ) } = \big ( \gamma + \eta _ { s } \alpha _ { s } ^ { ( t ) } \big ) \mathbb { I } \{ s \in \mathcal { A } _ { t } \} \ , \qquad b _ { t } = \beta _ { t } - \sum _ { s \in \mathcal { A } _ { t } } r _ { s } ^ { ( t ) } \ .
$$

Also recall that we assume $t + d _ { t } \leq T$ for every $t \in [ T ]$ . Consequently, no round t remains active at time $T + 1$ , and $r _ { t } ^ { ( T + 1 ) } = 0$ for every $t \in [ T ]$ ; likewise, $\boldsymbol { \alpha } _ { t } ^ { ( T + 1 ) } = 0$ by property (S2) of Definition 1. We first prove (i). Both $\eta _ { t }$ and $Q _ { t }$ are nondecreasing. Together with $\beta _ { 0 } = 0$ , this implies that $\beta _ { t }$ is also nondecreasing. Hence, Equation (8) gives $c _ { 0 } ^ { ( t + 1 ) } = \beta _ { t + 1 } - \beta _ { t } \geq 0$ for every $t \in \{ 0 , \ldots , T \}$ Next, fix $t \in [ T ]$ . Using the monotonicity of $\eta _ { s }$ and the properties of the allocation schedule, we obtain

$$
\begin{array} { r l } { \displaystyle \sum _ { s \in \mathcal { A } _ { t } } r _ { s } ^ { ( t ) } = \gamma q _ { t } + \displaystyle \sum _ { s \in \mathcal { A } _ { t } } \eta _ { s } \alpha _ { s } ^ { ( t ) } } & { \qquad ( | \mathcal { A } _ { t } | = q _ { t } ) } \\ { \displaystyle \quad \le \gamma q _ { t } + \eta _ { t } \displaystyle \sum _ { s \in \mathcal { A } _ { t } } \alpha _ { s } ^ { ( t ) } } & { \qquad ( \eta _ { s } \le \eta _ { t } \mathrm { ~ f o r ~ } s \in \mathcal { A } _ { t } ) } \\ { \displaystyle \le \gamma q _ { t } + \eta _ { t } } & { \qquad \mathrm { ( p r o p e r t y ~ ( S 3 ) ~ o f ~ D e f i n i t i o n ~ 1 ) } } \\ { \displaystyle \le \gamma Q _ { t } + \eta _ { t } } & { \qquad ( q _ { t } \le Q _ { t } ) } \\ { \displaystyle = \beta _ { t } . } \end{array}
$$

Therefore $b _ { t } \geq 0$ . We also claim that $r _ { s } ^ { ( t ) } \ge r _ { s } ^ { ( t + 1 ) }$ for every $1 \leq s \leq t \leq T$ . Indeed, property (S2) of Definition 1 gives $\alpha _ { s } ^ { ( t ) } = 0$ whenever s $\notin \boldsymbol { A } _ { t }$ , so $r _ { s } ^ { ( t ) } = \gamma \mathbb { I } \{ s \in \mathcal { A } _ { t } \} + \eta _ { s } \alpha _ { s } ^ { ( t ) }$ . For a fixed $s \leq t$ ， round s leaves the active set between times t and $t + 1$ exactly when $s \in \mathcal { N } _ { t }$ . Thus

$$
r _ { s } ^ { ( t ) } - r _ { s } ^ { ( t + 1 ) } = \gamma \mathbb { I } \{ s \in \mathcal { N } _ { t } \} + \eta _ { s } \big ( \alpha _ { s } ^ { ( t ) } - \alpha _ { s } ^ { ( t + 1 ) } \big ) \geq 0 ,
$$

where the inequality follows from $\eta _ { s } \geq 0$ and property (S1) of Definition 1. Substituting these inequalities into Equation (8) yields, for all $1 \leq s \leq t \leq T$

$$
c _ { s } ^ { ( t + 1 ) } = b _ { t } \mathbb { I } \{ s = t \} + r _ { s } ^ { ( t ) } - r _ { s } ^ { ( t + 1 ) } \geq 0 .
$$

Together with the nonnegativity of the source coeficients $c _ { 0 } ^ { ( t + 1 ) }$ , this proves (i).

To prove (ii), we show that both the incoming and outgoing weights at each $t \in [ T ]$ equal $b _ { t } + r _ { t } ^ { ( t ) }$ . First recall their definitions:

$$
A _ { t } ^ { \mathrm { i n } } = \sum _ { s = 0 } ^ { t - 1 } c _ { s } ^ { ( t ) } , \qquad A _ { t } ^ { \mathrm { o u t } } = \sum _ { k = t } ^ { T } c _ { t } ^ { ( k + 1 ) } .
$$

For each $t \in \{ 2 , \ldots , T \}$ , Equation (8) gives

$$
\begin{array} { l l } { { A _ { t } ^ { \mathrm { i n } } = c _ { 0 } ^ { ( t ) } + \displaystyle \sum _ { s = 1 } ^ { t - 1 } c _ { s } ^ { ( t ) } } } \\ { { \displaystyle \quad = ( \beta _ { t } - \beta _ { t - 1 } ) + b _ { t - 1 } + \displaystyle \sum _ { s = 1 } ^ { t - 1 } ( r _ { s } ^ { ( t - 1 ) } - r _ { s } ^ { ( t ) } ) } } \\ { { \displaystyle \quad = \beta _ { t } - \displaystyle \sum _ { s = 1 } ^ { t - 1 } r _ { s } ^ { ( t ) } } } & { { \qquad \mathrm { ( d e f i n i t i o n ~ o f ~ } b _ { t - 1 } ) } } \\ { { \displaystyle \quad = \beta _ { t } - \displaystyle \sum _ { s \in A _ { t } } r _ { s } ^ { ( t ) } + r _ { t } ^ { ( t ) } } } & { { \qquad \mathrm { ( d e f i n i t i o n ~ o f ~ } r _ { s } ^ { ( t ) } ) } } \\ { { \displaystyle \quad = b _ { t } + r _ { t } ^ { ( t ) } . } } & { { \qquad \mathrm { ( d e f i n i t i o n ~ o f ~ } b _ { t } ) } } \end{array}
$$

For $t = 1$ , the same identity holds directly because $\mathcal { A } _ { 1 } = \{ 1 \}$ , and thus $A _ { 1 } ^ { \mathrm { i n } } = c _ { 0 } ^ { ( 1 ) } = \beta _ { 1 } = b _ { 1 } + r _ { 1 } ^ { ( 1 ) }$ Finally, fix any $t \in [ T ]$ and sum coeficients $c _ { t } ^ { ( k + 1 ) }$ using their definition from Equation (8) over all $k \in \{ t , \ldots , T \}$ to obtain that

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { t } ^ { \mathrm { o u t } } = \sum _ { k = t } ^ { T } \left( b _ { k } \mathbb { I } \{ k = t \} + r _ { t } ^ { ( k ) } - r _ { t } ^ { ( k + 1 ) } \right) } } \\ { { \displaystyle ~ = b _ { t } + \sum _ { k = t } ^ { T } ( r _ { t } ^ { ( k ) } - r _ { t } ^ { ( k + 1 ) } ) } } \\ { { \displaystyle ~ = b _ { t } + r _ { t } ^ { ( t ) } - r _ { t } ^ { ( T + 1 ) } } } \\ { { \displaystyle ~ = b _ { t } + r _ { t } ^ { ( t ) } } } \\ { { \displaystyle ~ = A _ { t } ^ { \mathrm { i n } } ~ . } } \end{array}
$$

This proves (ii).

We are now ready to establish the master bound of Theorem 5. The weight-conservation property of Lemma 5 allows us to eliminate the non-telescoping terms in the general regret bound of Theorem 6. On top of these coeficients, Algorithm 1 introduces an additional regularization term centered at the selected hint-source iterate, with weight γ. This term provides the negative stability contribution needed for fast regret rates in games, while Lemma 4 ensures that its additional contribution to the multi-anchor potential remains controlled. Combining these properties yields Theorem 5, restated for convenience below.

Theorem 5. (Master bound) Let $\psi$ be diferentiable and 1-strongly convex with respect to $\lVert \cdot \rVert$ , with $\operatorname* { s u p } _ { u \in \mathcal { W } } D _ { \psi } ( u \mid w _ { 0 } ) \leq M$ for every $u \in \mathcal W$ . Let $\gamma > 0$ , let $( \alpha _ { k } ^ { ( t ) } ) _ { 1 \leq k \leq t \leq T + 1 }$ be an allocation schedule, and let $0 = \eta _ { 1 } \leq \cdot \cdot \cdot \leq \eta _ { T + 1 }$ . Then Algorithm 1 guarantees, $f o r$ every $u \in \mathcal W$

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq 2 \gamma M q _ { \operatorname* { m a x } } + M \eta _ { T + 1 } + \sum _ { t = 1 } ^ { T } \frac { \Vert g _ { t } - h _ { t } \Vert _ { * } ^ { 2 } } { \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } } - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \Vert w _ { t + 1 } - w _ { s _ { t + 1 } } \Vert ^ { 2 } .
$$

Proof. Fix $u \in \mathcal W$ . We begin by defining the regularization coeficients used by Algorithm 1. For every $t \in [ T ]$ and every $k \in \{ 0 , \ldots , t \}$ , they are $\widehat { c } _ { k } ^ { ( t + 1 ) } : = c _ { k } ^ { ( t + 1 ) } + \gamma \mathbb { I } \{ k = s _ { t + 1 } \}$ . Furthermore, for each $t \in [ T + 1 ]$ , these coeficients induce the total incoming weight $\begin{array} { r } { \widehat { A } _ { t } ^ { \mathrm { i n } } : = \sum _ { k = 0 } ^ { t - 1 } \widehat { c } _ { k } ^ { ( t ) } = A _ { t } ^ { \mathrm { i n } } + \gamma } \end{array}$ and the total outgoing weight $\begin{array} { r } { \widehat { A } _ { t - 1 } ^ { \mathrm { o u t } } : = \sum _ { k = t } ^ { T + 1 } \widehat { c } _ { t - 1 } ^ { ( k ) } = A _ { t - 1 } ^ { \mathrm { o u t } } + \gamma \sum _ { k = t } ^ { T + 1 } \mathbb { I } \{ s _ { k } = t \} } \end{array}$ , where $A _ { t } ^ { \mathrm { i n } }$ and $A _ { t - 1 } ^ { \mathrm { o u t } }$ are defined in the usual way with respect to coeficients $c _ { k } ^ { ( t + 1 ) }$ only. Using these definitions, we can show that

$$
\begin{array} { r l } & { \tilde { \lambda } _ { 0 } ^ { \mathrm { e q a } } D _ { \diamond } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } \left( \tilde { A } _ { 1 } ^ { \mathrm { e q a } } - \tilde { A } _ { 1 } ^ { \mathrm { i n } } \right) D _ { \diamond } ( u \mid w _ { t } ) } \\ & { \qquad = A _ { 0 } ^ { \mathrm { o u t } } D _ { \diamond } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } \left( A _ { 1 } ^ { \mathrm { e q a } } - A _ { 1 } ^ { \mathrm { i n } } \right) D _ { \diamond } ( u \mid w _ { t } ) } \\ & { \qquad \quad + \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { t = 1 } ^ { T } D _ { \diamond } ( u \mid w _ { t } ) { \mathbb 1 } \{ s _ { t + 1 } - { \mathbb T } _ { t } ^ { T } \sum _ { t = 1 } ^ { T } \nu _ { t } ^ { \mathrm { o u t } } ( u \mid w _ { t } ) }  \\ & { \qquad \quad - A _ { 0 } ^ { \mathrm { i n } } D _ { \diamond } ( u \mid w _ { t } ) + \displaystyle \sum _ { t = 1 } ^ { T } \left( A _ { 1 } ^ { \mathrm { e q a } } - A _ { 1 } ^ { \mathrm { i n } } \right) D _ { \diamond } ( u \mid w _ { t } ) + \displaystyle \gamma \sum _ { t = 1 } ^ { T } D _ { \diamond } ( u \mid w _ { t + 1 } ) - \gamma \displaystyle \sum _ { t = 1 } ^ { T } D _ { \diamond } ( u \mid w _ { t } ) } \\ &  \qquad = A _ { 0 } ^ { \mathrm { o u t } } D _ { \diamond } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } \left( A _ { 1 } ^ { \mathrm { e q a } } - A _ { 1 } ^ { \mathrm { i n } } \right) D _ { \diamond } ( u \mid w _ { t } ) + \displaystyle \gamma \sum _ { t = 1 } ^ { T } D _ { \diamond } ( u \mid w _ { t + 1 } ) - \gamma \displaystyle \sum _ { t = 1 } ^ { T } D _ { \diamond } ( u \mid w _  t  \end{array}
$$

where the last step follows by Lemma 4. Since $s _ { t + 1 } \in \{ 0 , \ldots , t \}$ and $\widehat { c } _ { s _ { t + 1 } } ^ { ( t + 1 ) } \geq \gamma$ for every $t \in [ T ]$ and using the 1-strong convexity of $\psi _ { ; }$ we also have that

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 0 } ^ { T } \displaystyle \sum _ { k = 0 } ^ { t } \hat { c } _ { k } ^ { ( t + 1 ) } D _ { \psi } \big ( w _ { t + 1 } ~ \big \vert ~ w _ { k } \big ) \geq \displaystyle \sum _ { t = 1 } ^ { T } \hat { c } _ { s _ { t + 1 } } ^ { ( t + 1 ) } D _ { \psi } \big ( w _ { t + 1 } ~ \big \vert ~ w _ { s _ { t + 1 } } \big ) } & { } \\ { \geq \gamma \displaystyle \sum _ { t = 1 } ^ { T } D _ { \psi } \big ( w _ { t + 1 } ~ \big \vert ~ w _ { s _ { t + 1 } } \big ) } & { } \\ { \geq \frac { \gamma } { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \Big \Vert w _ { t + 1 } - w _ { s _ { t + 1 } } \Big \Vert ^ { 2 } . } \end{array}
$$

Because the coeficients $c _ { s } ^ { ( t + 1 ) }$ are nonnegative by Lemma 5, we can now apply Theorem 6 and show that

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq \hat { A } _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } \big ( \hat { A } _ { t } ^ { \mathrm { o u t } } - \hat { A } _ { t } ^ { \mathrm { i n } } \big ) D _ { \psi } ( u \mid w _ { t } ) + \displaystyle \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { \hat { c } _ { t } ^ { ( t + 1 / 4 _ { t } 1 ) } } } & { } \\ { \displaystyle ~ - \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T } \sum _ { k = 0 } ^ { t } \hat { c } _ { k } ^ { ( t + 1 ) } D _ { \psi } ( w _ { t + 1 } \mid w _ { k } ) } & { } \\ { \leq \gamma M \sigma _ { \operatorname* { m a x } } + A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { T } \big ( A _ { t } ^ { \mathrm { o u t } } - A _ { t } ^ { \mathrm { i n } } \big ) D _ { \psi } ( u \mid w _ { t } ) + \displaystyle \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { c _ { t } ^ { ( t + d _ { t } + 1 ) } } } & { } \\ { \displaystyle ~ - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \| w _ { t + 1 } - w _ { s _ { t + 1 } } \| ^ { 2 } . } & { } \end{array}\tag{15}
$$

Recalling the definition of $r _ { t } ^ { ( s ) } = ( \gamma + \eta _ { t } \alpha _ { t } ^ { ( s ) } ) \mathbb { I } \{ t \in \mathcal { A } _ { s } \}$ , and since $t \in \mathcal { A } _ { t + d _ { t } }$ whereas t $\ncong A _ { t + d _ { t } + 1 }$ 2 the “terminal” coeficients $c _ { t } ^ { ( t + d _ { t } + 1 ) }$ satisfy

$$
c _ { t } ^ { ( t + d _ { t } + 1 ) } \geq r _ { t } ^ { ( t + d _ { t } ) } - r _ { t } ^ { ( t + d _ { t } + 1 ) } = \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } .
$$

Moreover, the total outgoing weight $A _ { 0 } ^ { \mathrm { o u t } }$ with respect to the default anchor $w _ { 0 }$ corresponds by construction to

$$
A _ { 0 } ^ { \mathrm { o u t } } = \sum _ { t = 1 } ^ { T + 1 } c _ { 0 } ^ { ( t ) } = \sum _ { t = 0 } ^ { T } ( \beta _ { t + 1 } - \beta _ { t } ) = \beta _ { T + 1 } - \beta _ { 0 } = \eta _ { T + 1 } + \gamma q _ { \mathrm { m a x } } \ ,
$$

and thus $A _ { 0 } ^ { \mathrm { o u t } } D _ { \psi } ( u \mid w _ { 0 } ) \leq M ( \eta _ { T + 1 } + \gamma q _ { \mathrm { m a x } } )$ . Applying these inequalities to the bound shown in Equation (15) finally gives

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \le 2 \gamma M q _ { \operatorname* { m a x } } + M \eta _ { T + 1 } + \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } } - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \bigl \| w _ { t + 1 } - w _ { s _ { t + 1 } } \bigr \| ^ { 2 } } } \\ & { } & { \qquad + \sum _ { t = 1 } ^ { T } ( A _ { t } ^ { \mathrm { o u t } } - A _ { t } ^ { \mathrm { i n } } ) D _ { \psi } ( u \mid w _ { t } ) } \\ & { } & { \qquad = 2 \gamma M q _ { \operatorname* { m a x } } + M \eta _ { T + 1 } + \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } } - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \bigl \| w _ { t + 1 } - w _ { s _ { t + 1 } } \bigr \| ^ { 2 } , } \end{array}
$$

where the last step holds because $A _ { t } ^ { \mathrm { o u t } } = A _ { t } ^ { \mathrm { i n } }$ by Lemma 5. This concludes the proof.

## 6.4 Proof of Theorem 1

Theorem 5 holds for any allocation schedule satisfying Definition 1. To obtain our desired dependence on the delays, we now specialize to the congestion-based allocation schedule introduced in Equation (10). The following lemma verifies that this construction indeed satisfies all the requirements of an allocation schedule.

Lemma 6. The sequence $( \alpha _ { s } ^ { ( t ) } ) _ { 1 \leq s \leq t \leq T + 1 }$ from Equation (10) is an allocation schedule in the sense of Definition 1.

Proof. First note that all values are nonnegative. By Fact $2 ( \mathrm { a } )$ , for every $s \in [ T ]$ , the running congestion value $\kappa _ { s , t }$ w.r.t. round s is nondecreasing while $s \in \mathcal A _ { t }$ . Therefore, its reciprocal is nonincreasing over these same rounds. After time $s + d _ { s } , s \notin \mathcal { A } _ { t }$ and its allocation thus becomes 0. This proves (S1) and (S2) of Definition 1.

For (S3) from Definition 1, fix $t \in [ T ]$ . Fact $2 ( \mathrm { a } )$ gives $\kappa _ { s , t } \geq q _ { t } \geq 1$ for every $s \in \mathcal A _ { t }$ . Since $| { \mathcal { A } } _ { t } | = q _ { t }$ , we obtain

$$
\sum _ { s = 1 } ^ { t } \alpha _ { s } ^ { ( t ) } = \sum _ { s \in \mathcal { A } _ { t } } \frac { 1 } { \kappa _ { s , t } } \leq \frac { | \mathcal { A } _ { t } | } { q _ { t } } = 1 .
$$

We can now complete the proof of Theorem 1 by combining the master bound of Theorem 5 with the allocation schedule Equation (10) and the adaptive tuning rule Equation (9). By Lemma $6 ,$ the proposed schedule satisfies Definition 1. Moreover, its terminal weights are inversely proportional to the congestion of the corresponding rounds, allowing us to apply Lemma 8 to control the accumulated prediction-error terms. Finally, the choice of the optimism weight γ ensures that the negative stability term has the coeficient required for the fast-rate analysis.

Proof of Theorem 1. Fix $u \in \mathcal W$ and define $\begin{array} { r } { \lambda _ { t } : = \frac { 1 } { M } \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } \end{array}$ for every $t \in [ T ]$ . The allocation schedule satisfies Definition 1 by Lemma 6, and it also guarantees

$$
0 < \alpha _ { t } ^ { ( t + d _ { t } ) } = \frac { 1 } { \kappa _ { t , t + d _ { t } } } = \frac { 1 } { \kappa _ { t } } \leq \frac { 1 } { q _ { t } }
$$

for every $t \in [ T ]$ , where the last inequality holds by Fact $2 ( \mathrm { a } )$

Since each hint $h _ { t }$ is either an observed gradient or zero, we have $\| h _ { t } \| _ { * } \leq G$ and thus $\lambda _ { t } \ \leq$ $4 G ^ { 2 } / M \leq \gamma ^ { 2 }$ by the triangle inequality. Applying Lemma 8 with $\alpha _ { t } ~ = ~ \alpha _ { t } ^ { ( t + d _ { t } ) }$ gives $\eta _ { T + 1 } ~ \leq$ $\begin{array} { r } { \sqrt { ( 2 / M ) \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } } \end{array}$ . Moreover, the chosen weight satisfies $2 \gamma M \leq 8 L M + 4 G \sqrt { M }$ and $\dot { \gamma } / 4 \geq L .$ Since $w _ { 1 } = w _ { s _ { 1 } } = w _ { 0 }$ , dropping the nonpositive terminal movement term at $T + 1$ ultimately yields:

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \langle g _ { t } , w _ { t } - u \rangle \leq 2 \gamma M q _ { \operatorname* { m a x } } + M \eta _ { T + 1 } + \sum _ { t = 1 } ^ { T } \frac { \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } { \gamma + \eta _ { t } \alpha _ { t } ^ { ( t + d _ { t } ) } } - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \| w _ { t + 1 } - w _ { s _ { t + 1 } } \| ^ { 2 } } } \\ & { } & { \quad = 2 \gamma M q _ { \operatorname* { m a x } } + 2 M \eta _ { T + 1 } - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \| w _ { t + 1 } - w _ { s _ { t + 1 } } \| ^ { 2 } } \\ & { } & { \quad \leq ( 8 L M + 4 G \sqrt { M } ) q _ { \operatorname* { m a x } } + 2 \sqrt { 2 M \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } - h _ { t } \| _ { * } ^ { 2 } } - L \sum _ { t = 1 } ^ { T } \| w _ { t } - w _ { s _ { t } } \| ^ { 2 } . } \end{array}
$$

## 7 Multiplayer games

In this section, we consider N-player smooth games with player-independent delays, and we show that MAO-OMD with optimism weight $\gamma = \operatorname* { m a x } \{ 2 N L , 2 G / \sqrt { M } \}$ achieves fast social regret during self-play, fast individual regret when the game has nonnegative social regret, and the optimal regret rate under delays against arbitrary opponents.

Setting. We consider the N-player smooth games of [APFS22] under delays. Player $p \in [ N ]$ chooses an action $w ^ { p }$ in a nonempty, compact, and convex set $\mathcal { W } _ { p } ,$ , and at a strategy profile $z =$ $( w ^ { 1 } , \dots , w ^ { N } )$ , it incurs a loss $f _ { p } ( z )$ that is diferentiable and convex in $w ^ { p }$ . Denote by $g ^ { p } ( z ) : =$ $\nabla _ { w ^ { p } } f _ { p } ( z )$ the loss gradient of player $p ,$ and by $( u , z ^ { - p } )$ the profile z with $w ^ { p }$ replaced by $u \in \mathcal { W } _ { p }$ We assume that there exist constants $G , L > 0$ such that

$$
\| g ^ { p } ( z ) \| _ { * } \leq G \qquad \mathrm { a n d } \qquad \| g ^ { p } ( z ) - g ^ { p } ( z ^ { \prime } ) \| _ { * } \leq L \sum _ { j = 1 } ^ { N } \| w ^ { j } - w ^ { \prime j } \|
$$

for every player $p \in [ N ]$ and all profiles $z = ( w ^ { 1 } , \dots , w ^ { N } )$ and $z ^ { \prime } = ( w ^ { \prime 1 } , \dots , w ^ { \prime N } )$ , which recovers Assumption 1 when $N = 2$ . In each round t, the players play the profile $z _ { t } = ( w _ { t } ^ { 1 } , \dots , w _ { t } ^ { N } )$ , and the gradients $g _ { t } ^ { p } : = g ^ { p } ( z _ { t } )$ of round t reach all players at the end of round $t + d _ { t }$ , where the delays $d _ { t }$ are arbitrary and unknown. The individual regret of player $p$ is $\begin{array} { r } { \mathrm { R e g } _ { T } ^ { p } : = \operatorname* { s u p } _ { u \in \mathcal W _ { p } } \sum _ { t = 1 } ^ { T } \mathopen { } \mathclose \bgroup \left( f _ { p } ( z _ { t } ) - \aftergroup \egroup \right) } \end{array}$ $f _ { p } ( u , z _ { t } ^ { - p } ) )$ , and the social regret is SocReg $\begin{array} { r } { \mathbf { \Sigma } _ { T } : = \sum _ { p = 1 } ^ { N } \mathrm { R e g } _ { T } ^ { p } } \end{array}$

Games with nonnegative social regret. In general-sum games, the social regret can be negative, so a bound on the social regret does not bound the individual regrets. As in [APFS22, Theorem 3.1 and Corollary 3.3], we obtain individual bounds when the social regret is nonnegative.

Assumption 3 (Nonnegative social regret). For every horizon $T \in \mathbb { N } _ { + }$ and every sequence of profiles $z _ { 1 } , \dots , z _ { T } \in \prod _ { p } \mathcal { W } _ { p } ,$ we have $\mathrm { S o c R e g } _ { T } \geq 0$

Assumption 3 holds in two-player zero-sum games, including smooth convex–concave games, in zero-sum and constant-sum polymatrix games [CCDP16], and in strategically zero-sum games and their polymatrix extensions [APFS22, Proposition A.10].

Algorithm. Each player $p$ runs Algorithm 1 with its own gradients, a regularizer $\psi _ { p }$ satisfying Assumption 2 with the same M, and optimism weight $\gamma = \operatorname* { m a x } \{ 2 N L , 2 G / \sqrt { M } \}$ , which agrees with the two-player choice when $N \ = \ 2$ Since the delays are player-independent, all players select the same hint sources $s _ { t }$ . The proof of Theorem 1 first gives a bound for any $\gamma \geq 2 G / \sqrt { M }$ applying Theorem 5 and Lemma 8 with the schedule (10) and tuning (9), and dropping the terminal movement term at $T + 1$ , we obtain, for every $u \in \mathcal W$

$$
\sum _ { t = 1 } ^ { T } \langle g _ { t } ^ { p } , w _ { t } ^ { p } - u \rangle \leq 2 \gamma M q _ { \operatorname* { m a x } } + 2 \sqrt { 2 M \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } ^ { 2 } } - \frac { \gamma } { 4 } \sum _ { t = 1 } ^ { T } \| w _ { t } ^ { p } - w _ { s _ { t } } ^ { p } \| ^ { 2 } ,\tag{16}
$$

Taking the two-player parameter choice for $\gamma$ and bounding its coeficients recovers Theorem 1.

Proposition 3. If all players run MAO-OMD with optimism weight $\gamma = \operatorname* { m a x } \{ 2 N L , 2 G / \sqrt { M } \}$ under arbitrary unknown delays, then SocR $\mathrm { e g } _ { T } = O ( N ( G \sqrt { M } + N L M ) q _ { \mathrm { m a x } } )$ . If, in addition, Assumption 3 holds, then $\mathrm { R e g } _ { T } ^ { p } = O \bigl ( ( G \sqrt { M } + N L M ) q _ { \mathrm { m a x } } \bigr )$ for every player $p \in [ N ]$ . Furthermore, a player that runs MAO-OMD with optimism weight $\gamma =$ max $\{ 2 N L , 2 G / \sqrt { M } \}$ against arbitrary opponents under arbitrary unknown delays guarantees $\mathrm { R e g } _ { T } ^ { p } = O \big ( G \sqrt { M ( T + d _ { \mathrm { t o t } } ) } + N L M q _ { \mathrm { m a x } } \big )$

Proof. Denote by $\begin{array} { r } { \Delta _ { T } ^ { p } : = \sum _ { t \in \lceil T \rceil } \| w _ { t } ^ { p } - w _ { s _ { t } } ^ { p } \| ^ { 2 } } \end{array}$ the movement of player p relative to the hint sources, and let $\begin{array} { r } { \Delta _ { T } : = \sum _ { p = 1 } ^ { N } \Delta _ { T } ^ { p } } \end{array}$ . At the $q _ { \mathrm { m a x } }$ rounds with $s _ { t } = 0$ , the hint is 0 and $\| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } \leq G$ . At every other round, the hints $h _ { t } ^ { p } = g ^ { p } ( z _ { s _ { t } } )$ of all players come from the same profile, so smoothness and the Cauchy-Schwarz inequality give

$$
\| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } ^ { 2 } \leq N L ^ { 2 } \sum _ { j = 1 } ^ { N } \| w _ { t } ^ { j } - w _ { s _ { t } } ^ { j } \| ^ { 2 } .
$$

Together with $\kappa _ { t } \ \leq \ q _ { \operatorname* { m a x } } .$ we obtain that $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } ^ { 2 } \leq q _ { \operatorname* { m a x } } \big ( G ^ { 2 } q _ { \operatorname* { m a x } } + N L ^ { 2 } \Delta _ { T } \big ) } \end{array}$ . Since $f _ { p } ( z _ { t } ) - f _ { p } ( u , z _ { t } ^ { - p } ) \leq \langle g _ { t } ^ { p } , w _ { t } ^ { p } - u \rangle$ by convexity, Equation (16) with $\gamma = \operatorname* { m a x } \{ 2 N L , 2 G / \sqrt { M } \}$ and ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ give, for each player $p \in [ N ]$

$$
\mathrm { R e g } _ { T } ^ { p } \leq \left( 4 N L M + ( 4 + 2 \sqrt { 2 } ) G \sqrt { M } \right) q _ { \mathrm { m a x } } + 2 \sqrt { 2 } L \sqrt { N M q _ { \mathrm { m a x } } \Delta _ { T } } - \frac { N L } { 2 } \Delta _ { T } ^ { p } .\tag{17}
$$

Summing (17) over the players, we obtain that

$$
\mathrm { S o c R e g } _ { T } \leq N \big ( 4 N L M + ( 4 + 2 \sqrt { 2 } ) G \sqrt { M } \big ) q _ { \mathrm { m a x } } + 2 \sqrt { 2 } N L \sqrt { N M q _ { \mathrm { m a x } } \Delta _ { T } } - \frac { N L } { 2 } \Delta _ { T } \ .\tag{18}
$$

By the AM-GM inequality, $2 \sqrt { 2 } N L \sqrt { N M q _ { \mathrm { m a x } } \Delta _ { T } } \ \leq \ N L \Delta _ { T } / 2 + 4 N ^ { 2 } L M q _ { \mathrm { m a x } }$ , so $\mathrm { S o c R e g } _ { T } \ \leq$ $\left( ( 4 + 2 \sqrt { 2 } ) N G \sqrt { M } + 8 N ^ { 2 } L M \right) q _ { \mathrm { { m a x } } }$

Under Assumption 3, the left-hand side of (18) is nonnegative. Using $2 \sqrt { 2 } N L \sqrt { N M q _ { \mathrm { m a x } } \Delta _ { T } } \le$ $N L \Delta _ { T } / 4 + 8 N ^ { 2 } L M q _ { \mathrm { { m a x } } }$ in (18) and rearranging the terms gives $L \Delta _ { T } \leq ( ( 1 6 + 8 \sqrt { 2 } ) G \sqrt { M } +$ $4 8 N L M ) q _ { \mathrm { m a x } }$ . Substituting this bound into (17) with $2 \sqrt { 2 } L \sqrt { N M q _ { \mathrm { m a x } } \Delta _ { T } } \leq L \Delta _ { T } / 4 + 8 N L M q _ { \mathrm { m a x } }$ gives $\mathrm { R e g } _ { T } ^ { p } \leq \bigl ( \bigl ( 8 + 4 \sqrt { 2 } \bigr ) G \sqrt { M } + 2 4 N L M \bigr ) q _ { \mathrm { m a x } }$

Against arbitrary opponents, each hint of player p is an observed gradient or zero, so $\parallel g _ { t } ^ { p } -$ $h _ { t } ^ { p } \vert \vert _ { * } ^ { 2 } \leq 4 G ^ { 2 }$ , and Fact 2(b) gives $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \kappa _ { t } \| g _ { t } ^ { p } - h _ { t } ^ { p } \| _ { * } ^ { 2 } \leq 4 G ^ { 2 } ( T + 2 d _ { \mathrm { t o t } } ) } \end{array}$ . Dropping the negative term in (16) with $\gamma = \operatorname* { m a x } \{ 2 N L , 2 G / \sqrt { M } \}$ , we obtain that $\mathrm { R e g } _ { T } ^ { p } \le ( 4 N L M + 4 G \sqrt { M } ) q _ { \mathrm { m a x } } +$ $4 \sqrt 2 G \sqrt { M ( T + 2 d _ { \mathrm { t o t } } ) } \leq ( 8 + 4 \sqrt 2 ) G \sqrt { M ( T + d _ { \mathrm { t o t } } ) } + 4 N L M q _ { \mathrm { m a x } }$ , where the last inequality uses $q _ { \mathrm { m a x } } ^ { 2 } \leq T + 2 d _ { \mathrm { t o t } }$ from Fact 1(e). □

## 8 Player-dependent delays

In this section, we consider two-player zero-sum games in which the two players can receive the feedback of the same round at diferent times, and we show that MAO-OMD keeps the guarantees of Theorem 2 when the players communicate which feedback they have received.

Setting. We consider the two-player zero-sum game of Section 2 with player-dependent delays: the gradient $g _ { t } ^ { x }$ of round t reaches player x at the end of round $t + d _ { t } ^ { x }$ , and the gradient $g _ { t } ^ { y }$ of round t reaches player y at the end of round $t + d _ { t } ^ { y }$ , where the delays $d _ { t } ^ { x } , d _ { t } ^ { y } \geq 0$ are arbitrary and unknown. For each player $p \in \{ x , y \}$ , denote by $\mathcal { P } _ { t } ^ { p } : = \{ s \in [ t - 1 ] : s + d _ { s } ^ { p } \geq t \}$ the rounds whose gradient for player p is pending at the start of round t, and let $\sigma _ { \operatorname* { m a x } } ^ { p } : = \operatorname* { m a x } _ { t \in [ T ] } | \mathcal { P } _ { t } ^ { p } |$ . At the end of each round, the players communicate the indices of the rounds whose gradients they have received so far, but not the gradients themselves.

Algorithm. With this communication, at the end of round t both players know the past rounds s for which both gradients have arrived, that is, the rounds with $s + \widetilde { d } _ { s } \leq t$ for the player-independent efective delay

$$
\widetilde { d } _ { s } : = \operatorname* { m a x } \{ d _ { s } ^ { x } , d _ { s } ^ { y } \} .
$$

Each player stores its gradient of round s until the end of round $s + \widetilde { d } _ { s }$ , and only then passes it to MAO-OMD. Thus, both players run Algorithm 1 with optimism weight $\gamma =$ max $\{ 4 L , 2 G / \sqrt { M } \}$ ， without any modification, under the player-independent delays $( \widetilde { d } _ { t } ) _ { t \in [ T ] }$ . In particular, the observed sets, the hint sources $s _ { t } ,$ and the anchor weights $c _ { k } ^ { ( t + 1 ) }$ are computed from $( \widetilde { d } _ { t } ) _ { t \in [ T ] }$ , and they are the same for both players.

Theorem 7. Let Assumptions 1 and 2 hold. Under player-dependent delays, if both players run MAO-OMD under the efective delays $\widetilde { d } _ { t } = \operatorname* { m a x } \{ d _ { t } ^ { x } , d _ { t } ^ { y } \}$ , then SocReg<sub>T</sub>, Reg<sup>x</sup> , and $\mathrm { R e g } _ { T } ^ { y }$ are all $O \big ( ( G \sqrt { M } + L M ) ( \sigma _ { \operatorname* { m a x } } ^ { x } + \sigma _ { \operatorname* { m a x } } ^ { y } + 1 ) \big )$

Proof. Denote by $\widetilde { \mathcal { P } } _ { t } : = \{ s \in [ t - 1 ] : s + \widetilde { d } _ { s } \geq t \}$ the set of pending rounds under the efective delays, and by qemax the value of qmax under the efective delays. Since $s + \widetilde { d } _ { s } \ge t$ if and only if $s + d _ { s } ^ { x } \geq t \ \mathrm { o r } \ s + d _ { s } ^ { y } \geq t .$ we have $\mathcal { \tilde { P } } _ { t } = \mathcal { P } _ { t } ^ { x } \cup \mathcal { P } _ { t } ^ { y }$ , and therefore

$$
\widetilde { q } _ { \mathrm { m a x } } = 1 + \operatorname* { m a x } _ { t \in [ T ] } \lvert \widetilde { \mathcal { P } } _ { t } \rvert \leq 1 + \operatorname* { m a x } _ { t \in [ T ] } \bigl ( \lvert \mathcal { P } _ { t } ^ { x } \rvert + \lvert \mathcal { P } _ { t } ^ { y } \rvert \bigr ) \leq \sigma _ { \mathrm { m a x } } ^ { x } + \sigma _ { \mathrm { m a x } } ^ { y } + 1 .
$$

For each round s, the gradients that MAO-OMD receives are $g ^ { x } ( x _ { s } , y _ { s } )$ and $g ^ { y } ( x _ { s } , y _ { s } )$ , and both players receive them at the end of round s + $s + \widetilde { d } _ { s }$ . Hence, both players run MAO-OMD under the player-independent delays $( \widetilde { d } _ { t } ) _ { t \in [ T ] }$ , and Theorem 2 gives that $\mathrm { S o c R e g } _ { T } , \mathrm { R e g } _ { T } ^ { x }$ , and Reg<sup>y</sup> are all $O ( ( G \sqrt { M } + L M ) \widetilde { q } _ { \mathrm { m a x } } )$ . Substituting the bound on $\widetilde { q } _ { \mathrm { m a x } }$ proves the theorem. □

Without communication, each player can still run MAO-OMD under its own delays, and Theorem 3 bounds its individual regret against an arbitrary opponent.

## 9 Discussion

In this paper, we studied learning in smooth games with arbitrary and unknown feedback delays. We introduced MAO-OMD, an optimistic learning algorithm that handles delayed feedback directly through an adaptive allocation of regularization weights across past iterates. Our algorithm achieves optimal $O ( \sigma _ { \mathrm { m a x } } + 1 )$ social and individual regret when both players use it, while simultaneously guaranteeing minimax-optimal $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ regret against arbitrary opponents. These guarantees improve upon scheduling-based approaches, for which we showed that an additional logarithmic factor cannot generally be avoided. We also extended our results to multiplayer games and settings with player-dependent delays.

## AI methodology statement

We used OpenAI’s GPT-6 Astra and Anthropic’s Claude Fable 5.1 models to explore initial proof strategies, refine draft arguments, and check mathematical derivations, including exposition of the scheduling-based reduction and its lower-bound analysis (Section 4 and Appendix B), the multianchor regret analysis (Sections 5 and 6), and supporting lemmas (Appendix D). The AI-assisted arguments were thoroughly revised by the authors rather than adopted as standalone results. The authors take full responsibility for the correctness, originality, and integrity of the paper, including all mathematical arguments and references.

## References

[ADF<sup>+</sup>22] Ioannis Anagnostides, Constantinos Daskalakis, Gabriele Farina, Maxwell Fishelson, Noah Golowich, and Tuomas Sandholm. Near-optimal no-regret learning for correlated equilibria in multi-player general-sum games. In ACM SIGACT Symposium on Theory of Computing, 2022. 1, 32

[ALM26] Omar Abbadi, Rida Laraki, and Panayotis Mertikopoulos. Constant regret in general games via higher-order optimism. arXiv preprint arXiv:2609.04113, 2026. 1, 32

[APFS22] Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, and Tuomas Sandholm. On last-iterate convergence beyond zero-sum games. In International Conference on Machine Learning, 2022. 6, 24, 32

[BZC<sup>+</sup>19] Ilai Bistritz, Zhengyuan Zhou, Xi Chen, Nicholas Bambos, and Jose Blanchet. Online EXP3 learning in adversarial bandits with delayed feedback. In Advances in Neural Information Processing Systems, 2019. 32

[CBGM19] Nicolò Cesa-Bianchi, Claudio Gentile, and Yishay Mansour. Delay and cooperation in nonstochastic bandits. Journal of Machine Learning Research, 2019. 32

[CBL06] Nicolò Cesa-Bianchi and Gábor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006. 1, 32

[CCDP16] Yang Cai, Ozan Candogan, Constantinos Daskalakis, and Christos Papadimitriou. Zero-sum polymatrix games: A generalization of minmax. Mathematics of Operations Research, 2016. 24, 32

[CP20] Xi Chen and Binghui Peng. Hedging in games: Faster convergence of external and swap regrets. In Advances in Neural Information Processing Systems, 2020. 1, 32

[CYL<sup>+</sup>12] Chao-Kai Chiang, Tianbao Yang, Chia-Jung Lee, Mehrdad Mahdavi, Chi-Jen Lu, Rong Jin, and Shenghuo Zhu. Online optimization with gradual variations. In Conference on Learning Theory, 2012. 1, 4, 32

[DFG21] Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. Near-optimal noregret learning in general games. In Advances in Neural Information Processing Systems, 2021. 1, 32

[DISZ18] Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, and Haoyang Zeng. Training GANs with optimism. In International Conference on Learning Representations, 2018. 32

[EMQ<sup>+</sup>23] Emmanuel Esposito, Saeed Masoudian, Hao Qiu, Dirk van der Hoeven, Nicolò Cesa-Bianchi, and Yevgeny Seldin. Delayed bandits: When do intermediate observations help? In International Conference on Machine Learning, 2023. 32

[FAA25] Yuma Fujimoto, Kenshi Abe, and Kaito Ariu. Learning from delayed feedback in games via extra prediction. In Advances in Neural Information Processing Systems, 2025. 1, 8, 32

[FAA26] Yuma Fujimoto, Kenshi Abe, and Kaito Ariu. Linear convergence in games with delayed feedback via extra prediction. arXiv preprint arXiv:2602.17486, 2026. 32

[FKS19] Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Online convex optimization for sequential decision processes and extensive-form games. In AAAI Conference on Artificial Intelligence, 2019. 3, 32

[FL98] Drew Fudenberg and David K. Levine. The Theory of Learning in Games. MIT Press, 1998. 1

[FN95] Ulrich Faigle and Willem M. Nawijn. Note on scheduling intervals on-line. Discrete Applied Mathematics, 58(1):13–17, 1995. 32

[FOC<sup>+</sup>21] Genevieve E. Flaspohler, Francesco Orabona, Judah Cohen, Soukayna Mouatadid, Miruna Oprescu, Paulo Orenstein, and Lester Mackey. Online learning with optimism and delay. In International Conference on Machine Learning, 2021. 8, 32

[FS99] Yoav Freund and Robert E. Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 1999. 1, 3, 32

[GJ21] András György and Pooria Joulani. Adapting to delays and data in adversarial multiarmed bandits. In International Conference on Machine Learning, 2021. 32

[HAM21] Yu-Guan Hsieh, Kimon Antonakopoulos, and Panayotis Mertikopoulos. Adaptive learning in continuous games: Optimal regret bounds and convergence to Nash equilibrium. In Conference on Learning Theory, 2021. 32

[HDH23] Jiatai Huang, Yan Dai, and Longbo Huang. Banker online mirror descent: A universal approach for delayed online bandit learning. In International Conference on Machine Learning, 2023. 2, 32

[HIMM19] Yu-Guan Hsieh, Franck Iutzeler, Jérôme Malick, and Panayotis Mertikopoulos. On the convergence of single-call stochastic extra-gradient methods. In Advances in Neural Information Processing Systems, 2019. 32

[HIMM22] Yu-Guan Hsieh, Franck Iutzeler, Jérôme Malick, and Panayotis Mertikopoulos. Multiagent online optimization with delays: Asynchronicity, adaptivity, and optimism. Journal of Machine Learning Research, 2022. 32

[HMZ20] Amélie Héliou, Panayotis Mertikopoulos, and Zhengyuan Zhou. Gradient-free online learning in continuous games with delayed rewards. In International Conference on Machine Learning, 2020. 32

[ITH24] Shinji Ito, Taira Tsuchiya, and Junya Honda. Adaptive learning rate for follow-theregularized-leader: Competitive analysis and best-of-both-worlds. In Shipra Agrawal

and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 2522–2563. PMLR, 2024. 11, 32

[JC22] Andrew Jacobsen and Ashok Cutkosky. Parameter-free mirror descent. In Conference on Learning Theory, 2022. 32

[JEQZ26] Andrew Jacobsen, Emmanuel Esposito, Hao Qiu, and Mengxiao Zhang. Strongly adaptive online learning with time-varying movement cost. In The Fortieth Annual Conference on Neural Information Processing Systems, 2026. 32

[JGS13] Pooria Joulani, András György, and Csaba Szepesvári. Online learning under delayed feedback. In International Conference on Machine Learning, 2013. 2, 7, 32, 33

[JGS16] Pooria Joulani, András György, and Csaba Szepesvári. Delay-tolerant online convex optimization: Unified analysis and adaptive-gradient algorithms. In AAAI Conference on Artificial Intelligence, 2016. 32

[JGS17] Pooria Joulani, András György, and Csaba Szepesvári. A modular analysis of adaptive (non-)convex optimization: Optimism, composite objectives, and variational bounds. In International Conference on Algorithmic Learning Theory, 2017. 32

[KLPS07] Antoon W. J. Kolen, Jan Karel Lenstra, Christos H. Papadimitriou, and Frits C. R. Spieksma. Interval scheduling: A survey. Naval Research Logistics, 54(5):530–543, 2007. 32

[KT81] Henry A. Kierstead and William T. Trotter. An extremal problem in recursive combinatorics. Congressus Numerantium, 33:143–153, 1981. 32

[KT06] Jon Kleinberg and Eva Tardos. Algorithm Design. Addison-Wesley, Boston, 2006. 32

[LFO26] Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. Constant individual regret in general games. arXiv preprint arXiv:2608.31166, 2026. 1, 32

[Mai26] Tung Mai. Constant swap regret in general-sum games via two-scale higher-order optimism. arXiv preprint arXiv:2609.16751v2, 2026. 1, 32

[McM11] Brendan McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theorems and L1 regularization. In Geofrey Gordon, David Dunson, and Miroslav Dudík, editors, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 525–533, Fort Lauderdale, FL, USA, 2011. PMLR. 32

[MOA11] Albert W. Marshall, Ingram Olkin, and Barry C. Arnold. Inequalities: Theory of Majorization and Its Applications. Springer, 2nd edition, 2011. 36

[MS10] H. Brendan McMahan and Matthew Streeter. Adaptive bound optimization for online convex optimization. In Proceedings of the 23rd Annual Conference on Learning Theory, 2010. 32

[MZS22] Saeed Masoudian, Julian Zimmert, and Yevgeny Seldin. A best-of-both-worlds algorithm for bandits with delayed feedback. In Advances in Neural Information Processing Systems, 2022. 32

[NRTV07] Noam Nisan, Tim Roughgarden, Eva Tardos, and Vijay V. Vazirani, editors. Algorithmic Game Theory. Cambridge University Press, 2007. 1

[OP18] Francesco Orabona and Dávid Pál. Scale-free online learning. Theoretical Computer Science, 716:50–69, 2018. 11, 32

[Ora26] Francesco Orabona. Online learning: A modern introduction using convex optimization. arXiv preprint arXiv:1912.13213v10, 2026. 16, 32

[Pop80] Leonid Denisovich Popov. A modification of the Arrow–Hurwicz method for search of saddle points. Mathematical Notes of the Academy of Sciences of the USSR, 1980. 32

[QEZ25] Hao Qiu, Emmanuel Esposito, and Mengxiao Zhang. Exploiting curvature in online convex optimization with delayed feedback. In International Conference on Machine Learning, 2025. 32

[QJEZ26a] Hao Qiu, Andrew Jacobsen, Emmanuel Esposito, and Mengxiao Zhang. Parameterfree dynamic regret: Time-varying movement costs, delayed feedback, and memory. In International Conference on Machine Learning, 2026. 32

[QJEZ26b] Hao Qiu, Andrew Jacobsen, Emmanuel Esposito, and Mengxiao Zhang. Universal online convex optimization with delayed feedback. arXiv preprint arXiv:2610.28058, 2026. 2, 32

[QK15] Kent Quanrud and Daniel Khashabi. Online learning with adversarial delays. In Advances in Neural Information Processing Systems, 2015. 2, 6, 7, 32, 38

[QZA26] Hao Qiu, Mengxiao Zhang, and Juliette Achddou. Decentralized online convex optimization with unknown feedback delays. In AAAI Conference on Artificial Intelligence, 2026. 32

[RAR25] Alexander Ryabchenko, Idan Attias, and Daniel M. Roy. Capacity-constrained online learning with delays: Scheduling frameworks and regret trade-ofs. In Conference on Learning Theory, 2025. 32

[RAR26a] Alexander Ryabchenko, Idan Attias, and Daniel M. Roy. Capacity-constrained online convex optimization with delayed feedback. In Advances in Neural Information Processing Systems, 2026. 32

[RAR26b] Alexander Ryabchenko, Idan Attias, and Daniel M. Roy. A reduction from delayed to immediate feedback for online convex optimization with improved guarantees. In Advances in Neural Information Processing Systems, 2026. 32

[RS13a] Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Conference on Learning Theory, 2013. 1, 4, 32

[RS13b] Alexander Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. In Advances in Neural Information Processing Systems, 2013. 1, 4, 6, 32

[SALS15] Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E. Schapire. Fast convergence of regularized learning in games. In Advances in Neural Information Processing Systems, 2015. 1, 7, 32

[SP26] Ashkan Soleymani and Georgios Piliouras. Multiplicative optimism for constant regret in games. arXiv preprint arXiv:2609.21976, 2026. 1, 32

[SS12] Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends in Machine Learning, 4(2):107–194, 2012. 1, 32

[SZM26] Ofir Schlisselberg, Mengxiao Zhang, and Yishay Mansour. Near-optimal stochastic linear bandits with delay. arXiv preprint arXiv:2606.16656, 2026. 32

[TCBS19] Tobias Sommer Thune, Nicolò Cesa-Bianchi, and Yevgeny Seldin. Nonstochastic multiarmed bandits with unrestricted delays. In Advances in Neural Information Processing Systems, 2019. 32

[vdHCB22] Dirk van der Hoeven and Nicolò Cesa-Bianchi. Nonstochastic bandits and experts with arm-dependent delays. In International Conference on Artificial Intelligence and Statistics, 2022. 32

[WO02] Marcelo J. Weinberger and Erik Ordentlich. On delayed prediction of individual sequences. IEEE Transactions on Information Theory, 2002. 32

[WYSZ24] Yuanyu Wan, Chang Yao, Mingli Song, and Lijun Zhang. Non-stationary online convex optimization with arbitrary delays. In International Conference on Machine Learning, 2024. 32

[ZMB<sup>+</sup>17] Zhengyuan Zhou, Panayotis Mertikopoulos, Nicholas Bambos, Peter W. Glynn, and Claire Tomlin. Countering feedback delays in multi-agent learning. In Advances in Neural Information Processing Systems, 2017. 32

[ZS20] Julian Zimmert and Yevgeny Seldin. An optimal algorithm for adversarial bandits with arbitrary delays. In International Conference on Artificial Intelligence and Statistics, 2020. 32

[ZTI26] Ruotong Zhuang, Taira Tsuchiya, and Shinji Ito. Optimal learning in games under delayed feedback. In International Conference on Artificial Intelligence and Statistics, 2026. 1, 2, 6, 7, 8, 32

## A Additional related work

Online learning. Online learning and online convex optimization have been studied extensively; see [CBL06, SS12, Ora26] for background. Optimistic methods, which exploit a prediction of the next loss, were analyzed via predictable sequences by $[ \mathrm { C Y L ^ { + } 1 2 }$ , RS13a]; see also [JGS17] for a modular treatment. The multi-anchor regularization of MAO-OMD is similar in spirit to “proximal” FTRL methods, which use a regularizer of the form $\begin{array} { r } { \Psi { t } ( w ) = \sum _ { s = 1 } ^ { t } \frac { c _ { s } } { 2 } \| w - w _ { s } \| ^ { 2 } } \end{array}$ with a typically monotonic sequence $\left( c _ { s } \right) _ { \varepsilon }$ [MS10, McM11]. There, however, the weight on a past action $w _ { s }$ is fixed once and stays the same at every later round, whereas MAO-OMD uses a weight $c _ { s } ^ { ( t ) }$ on $w _ { s }$ that changes with the round t, determined adaptively by the delays that unfold after round s and by when the feedback of round s arrives. The adaptive tuning of the regularization scale in MAO-OMD follows the tradition of scale-free and parameter-free methods [OP18, JC22, ITH24].

Online learning with delayed feedback. Early work on delayed feedback includes [WO02], who showed that $d + 1$ interleaved copies of a base learner are optimal under a constant delay $d ,$ and the black-box reduction BOLD of [JGS13] for arbitrary delays. Beyond black-box techniques, [QK15] showed that gradient descent and FTPL applied directly to the delayed gradients achieve the minimax-optimal $O ( \sqrt { T + d _ { \mathrm { t o t } } } )$ regret, and there is substantial work on algorithms tailored to delayed online convex optimization, including reductions to immediate feedback [JGS16, RAR26b], dynamic regret [WYSZ24, QJEZ26a], strongly adaptive regret [JEQZ26], curvature [QEZ25], universal guarantees [QJEZ26b], and decentralized settings [QZA26]. Closest in spirit to our regret decomposition is the delay-as-optimism view of $[ \mathrm { F O C ^ { + } 2 1 } ]$ ], which treats the missing gradients under a constant delay as an optimistic hint. Our weight allocation is inspired by the Banker-OMD principle of [HDH23, QJEZ26b]. Delays also arise in multi-armed bandit problems, where they interact with exploration [CBGM19, TCBS19, BZC<sup>+</sup>19, ZS20, GJ21, vdHCB22, MZS22, EMQ<sup>+</sup>23, SZM26]. Recently, delays have also been studied at the intersection of online learning and scheduling, where the learner has a limited capacity for tracking feedback from pending rounds and must choose which feedback to receive [RAR25, RAR26a]. Scheduling plays a diferent role there than in the BOLD reduction, which receives all feedback and only decides which learner handles each round.

Learning in games. The connection between regret minimization and equilibria in zero-sum games goes back to [FS99]; see also [FKS19, CCDP16]. Fast convergence through optimism originates in the optimistic gradient method of [Pop80] and the analyses of [RS13b, SALS15], whose RVU property we use in the form of a negative movement term; see [DISZ18, HIMM19] for lastiterate convergence, [HAM21] for adaptive regret and convergence in continuous games, [CP20, DFG21, $\mathrm { A D F ^ { + } 2 2 }$ , APFS22] for general-sum games, and [LFO26, ALM26, SP26, Mai26] for recent T-independent bounds. Delays in games were first studied in continuous games [ZMB<sup>+</sup>17, HMZ20, HIMM22]; fast rates under delays are recent and limited to fixed delays [FAA25, FAA26, ZTI26].

Online interval partitioning. The assignment of rounds to independent learners by the BOLD reduction can be viewed as an online interval-partitioning problem. Each round t corresponds to an interval $[ t , t + d _ { t } + 1 )$ , and overlapping intervals must be assigned to diferent learners. This connects our scheduling-based approach to classical interval partitioning and online interval coloring [KT06, FN95, KT81]; see also [KLPS07] for a survey of interval scheduling. Unlike standard formulations, the endpoints of our intervals are unknown when they arrive. Moreover, our objective is not merely to minimize the number of learners, but to control the regret.

## B Refined BOLD reduction: guarantees and limitations

In this section, we develop a refinement of the BOLD reduction of [JGS13], which can be applied in the arbitrary delays setting. We first show that this refined reduction can obtain the fast $O ( q _ { \mathrm { m a x } } )$ rate when used with cooperating learners, while also obtaining $O \big ( \sqrt { ( T + d _ { \mathrm { t o t } } ) ( 1 + \log q _ { \mathrm { m a x } } ) } \big )$ regret against an arbitrary opponent. We then show that the sub-optimal $\sqrt { 1 + \log q _ { \mathrm { m a x } } }$ penalty can not be avoided by choosing any alternative scheduling, demonstrating that the black-box approach is insuficient for obtaining our desired best-of-both-worlds guarantees.

## B.1 A refined BOLD reduction

Algorithm 2 implements the online partition rule of Section 4 using instances $\boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B } _ { r }$ , indexed in order of creation. In the pseudocode, r denotes the number of instances created so far. Instance $\boldsymbol { B } _ { k }$ handles the subsequence $I _ { k } = \{ t \in [ T ] : i _ { t } = k \}$ , where $i _ { t }$ is the index selected at round t. Each instance maintains its own state and local round counter and is idle when it has no pending feedback.

Algorithm 2: Refined BOLD   
Input: undelayed online learner ${ \overline { { B . } } }$   
Initialize: $r  0 .$   
1 for $t = 1 , \dots , T$ do   
2 $i _ { t } \gets 1$   
3 while $i _ { t } \le r$ and instance $B _ { i _ { t } }$ is busy do   
4 $\lfloor { \ i _ { t } } \gets i _ { t } + 1$   
5 if $i _ { t } = r + 1$ then   
6 $r  r + 1$   
7 initialize a fresh instance $\boldsymbol { B } _ { r }$   
8 obtain $w _ { t }$ from $B _ { i _ { t } }$ , mark that instance busy, and play $w _ { t }$   
9 observe $g _ { s }$ for every $s \in \mathcal { N } _ { t }$ at the end of round t   
10 foreach $s \in \mathcal { N } _ { t }$ do   
11 feed $g _ { s }$ to $B _ { i _ { s } }$ , perform its local update, and mark it idle

## B.2 The adversarial guarantee

Proof of Theorem $\it 4 ( i )$ . Let $i _ { t }$ be the index selected at round t. For each $k ,$ , let $I _ { k } = \{ t \in [ T ] : i _ { t } = k \}$ be the subsequence handled by instance $\boldsymbol { B } _ { k }$ , and set $n _ { k } = | I _ { k } |$ . Every busy instance holds exactly one pending round. Since there are $\sigma _ { t }$ busy instances, the smallest-idle-index rule selects an index $i _ { t } \le \sigma _ { t } + 1 = q _ { t } \le q _ { \operatorname* { m a x } }$ , whether it reuses an existing instance or creates a new one. Consequently,

we have

$$
\sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } k n _ { k } = \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } k \sum _ { t = 1 } ^ { T } \mathbb { I } \left\{ i _ { t } = k \right\} = \sum _ { t = 1 } ^ { T } \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } k \mathbb { I } \left\{ i _ { t } = k \right\} = \sum _ { t = 1 } ^ { T } i _ { t } \leq \sum _ { t = 1 } ^ { T } q _ { t } = T + d _ { \mathrm { t o t } } ,\tag{19}
$$

where the last equality uses Fact 1(d). Each instance receives feedback before choosing its next action, so its local interaction is undelayed. Hence, applying the base learner’s regret guarantee to each instance and applying Equation (19) yields

$$
\begin{array} { r l r } {  { \mathrm { R e g } _ { T } \le C _ { \mathcal { B } } \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } \sqrt { n _ { k } } = C _ { \mathcal { B } } \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } \frac { \sqrt { k n _ { k } } } { \sqrt { k } } } } \\ & { } & { \le C _ { \mathcal { B } } \sqrt { ( \displaystyle \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } k n _ { k } ) ( \displaystyle \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } \frac { 1 } { k } ) } \le C _ { \mathcal { B } } \sqrt { ( T + d _ { \mathrm { t o t } } ) ( 1 + \log q _ { \operatorname* { m a x } } ) } , } \end{array}
$$

where we have observed that $\begin{array} { r } { \sum _ { k = 1 } ^ { q _ { \operatorname* { m a x } } } \frac { 1 } { k } \le 1 + \log q _ { \operatorname* { m a x } } } \end{array}$

## B.3 The game guarantee

Proof of Theorem $\mathit { 4 } ( i i )$ . Both players use the same deterministic smallest-idle-index rule. Under common delays, their sets of busy instances coincide at the start of every round, so they select the same index. Thus instance $k$ of one player interacts only with instance k of the other. Each instance receives feedback before choosing its next action and runs B using its local round counter, so each pair plays an undelayed copy of the original game.

Let $n _ { k }$ be the number of rounds assigned to pair $k ,$ and denote its individual and social regrets by $\mathrm { R e g } _ { n _ { k } } ^ { p , ( \ddot { k } ) }$ and $\mathrm { S o c R e g } _ { n _ { k } } ^ { ( k ) }$ . By the assumed guarantee of $\boldsymbol { B }$ in part (ii), these regrets are $\mathcal { O } ( L M +$ $G \sqrt { M } )$ , uniformly over all pairs and local horizons $n _ { k }$ . For any fixed comparator, regret decomposes across the local subsequences. Taking the supremum for each player gives

$$
\mathrm { R e g } _ { T } ^ { p } \leq \sum _ { k } \mathrm { R e g } _ { n _ { k } } ^ { p , ( k ) } , \qquad \mathrm { S o c R e g } _ { T } \leq \sum _ { k } \mathrm { S o c R e g } _ { n _ { k } } ^ { ( k ) } .
$$

There are at most $q _ { \mathrm { m a x } }$ pairs, as shown in the proof of part (i). Summing their regret bounds therefore yields

$$
\mathrm { S o c R e g } _ { T } , \mathrm { R e g } _ { T } ^ { p } = { \cal O } \Bigl ( ( L M + G \sqrt { M } ) q _ { \mathrm { m a x } } \Bigr ) \ .
$$

## B.4 A lower bound for every assignment

Proof of the load bound in Proposition 1. Fix an integer $\sigma _ { \operatorname* { m a x } } \geq 1$ and write $q : = q _ { \mathrm { m a x } } = \sigma _ { \mathrm { m a x } } + 1$ Choose a positive integer N divisible by $1 ^ { 2 } , \ldots , q ^ { 2 }$ , and define

$$
\lambda _ { k } = \frac { N } { k ^ { 2 } } \quad ( k \in [ q ] ) , \qquad \lambda _ { q + 1 } = 0 , \qquad b _ { j } = \lambda _ { j } - \lambda _ { j + 1 } \quad ( j \in [ q ] ) .
$$

For each $j \in [ q ]$ , form $b _ { j }$ blocks of $j$ consecutive rounds and concatenate all blocks in any fixed order. Within a block of length j, release all feedback at the end of its last round, so its ath round has delay $j - a$ . No feedback remains pending between blocks. Every valid assignment must use distinct instances for the rounds within a block, since no feedback from that block arrives before its final action is chosen. Because $b _ { q } = \lambda _ { q } > 0$ , the largest block has length $q ,$ and the maximum number of pending rounds is exactly $q - 1 = \sigma _ { \operatorname* { m a x } }$

The choice of $b _ { j }$ ensures that, for every $k \in [ q ]$ , the number of blocks of length at least k is

$$
\sum _ { j = k } ^ { q } b _ { j } = \sum _ { j = k } ^ { q } ( \lambda _ { j } - \lambda _ { j + 1 } ) = \lambda _ { k } .
$$

Moreover, since there are $b _ { j }$ blocks of length $j$ by construction, we have

$$
T = \sum _ { j = 1 } ^ { q } j b _ { j } .
$$

Hence, the total number of rounds can be counted in two equivalent ways: by summing the block lengths, or by counting the blocks containing a k-th round for each $k \in [ q ]$ :

$$
T = \sum _ { j = 1 } ^ { q } j b _ { j } = \sum _ { k = 1 } ^ { q } \sum _ { j = k } ^ { q } b _ { j } = \sum _ { k = 1 } ^ { q } \lambda _ { k } \ .\tag{20}
$$

$\mathrm { A }$ block of length $j$ contributes $j$ to $T$ . Its ath round has delay $j - a ,$ , so its contribution to $d _ { \mathrm { t o t } }$ is $\textstyle \sum _ { a = 1 } ^ { j } ( j - a ) = j ( j - 1 ) / 2$ . Hence, its contribution to $T + d _ { \mathrm { t o t } }$ is

$$
j + { \frac { j ( j - 1 ) } { 2 } } = { \frac { j ( j + 1 ) } { 2 } } = \sum _ { k = 1 } ^ { j } k .
$$

Summing over all blocks and exchanging the order of summation, we obtain

$$
T + d _ { \mathrm { t o t } } = \sum _ { j = 1 } ^ { q } b _ { j } \sum _ { k = 1 } ^ { j } k = \sum _ { k = 1 } ^ { q } k \sum _ { j = k } ^ { q } b _ { j } = \sum _ { k = 1 } ^ { q } k \lambda _ { k } = N \sum _ { k = 1 } ^ { q } \frac { 1 } { k } = N H _ { q } ,\tag{21}
$$

where $\begin{array} { r } { H _ { q } = \sum _ { k = 1 } ^ { q } 1 / k } \end{array}$ . Now consider any valid assignment, allowing any number of instances, and order its loads as $n _ { 1 } \geq n _ { 2 } \geq \cdots$ . Since no feedback arrives before the end of a block, each instance can be assigned at most one round within that block. Hence, any r instances can cover at most min $\{ r , j \}$ rounds in a block of length $j .$ . Summing over blocks and again using Equation (20), we obtain

$$
\sum _ { i = 1 } ^ { r } n _ { i } \leq \sum _ { j = 1 } ^ { q } b _ { j } \operatorname* { m i n } \{ r , j \} = \sum _ { k = 1 } ^ { \operatorname* { m i n } \{ r , q \} } \sum _ { j = k } ^ { q } b _ { j } = \sum _ { k = 1 } ^ { \operatorname* { m i n } \{ r , q \} } \lambda _ { k } .
$$

Let $m \geq q$ be the number of instances assigned at least one round, where $m \geq q$ follows from the existence of a block of length $q .$ Extend the sequence $( \lambda _ { k } ) _ { k = 1 } ^ { q }$ by setting $\lambda _ { k } = 0$ for $q < k \le m$ Since both sequences are nonincreasing, have the same total sum $\begin{array} { r } { ( \mathrm { i . e . , ~ } \sum _ { i = 1 } ^ { m } \lambda _ { i } = \sum _ { i = 1 } ^ { m } n _ { i } = T ) } \end{array}$ and satisfy

$$
\sum _ { i = 1 } ^ { r } n _ { i } \leq \sum _ { k = 1 } ^ { r } \lambda _ { k } \qquad \forall r \in [ m ] ,
$$

we have that the vector $\left( \lambda _ { 1 } , \ldots , \lambda _ { m } \right)$ majorizes $( n _ { 1 } , \ldots , n _ { m } )$ , so by concavity of square root and Karamata’s inequality [MOA11, Proposition 3.C.1]:

$$
\sum _ { i = 1 } ^ { m } { \sqrt { n _ { i } } } \geq \sum _ { k = 1 } ^ { m } { \sqrt { \lambda _ { k } } } = \sum _ { k = 1 } ^ { q } { \sqrt { \lambda _ { k } } } .
$$

Finally, substituting $\lambda _ { k } = N / k ^ { 2 }$ and recalling that $T + d _ { \mathrm { t o t } } = N H _ { q }$ from Equation (21), we obtain

$$
\sum _ { i = 1 } ^ { m } \sqrt { n _ { i } } \geq \sqrt { N } H _ { q } = \sqrt { ( T + d _ { \mathrm { t o t } } ) H _ { q } } \geq \sqrt { ( T + d _ { \mathrm { t o t } } ) \log ( q + 1 ) } \geq \sqrt { ( T + d _ { \mathrm { t o t } } ) \frac { 1 + \log q _ { \mathrm { m a x } } } { 2 } } ~ ,\tag{22}
$$

where the third inequality uses $H _ { q } \geq \log ( q + 1 )$ , and the last follows from $q = q _ { \mathrm { m a x } } \ge 1$ □

## B.5 Realizing the lower bound with optimistic OMD

We consider the following adaptive single-anchor optimistic OMD learner, using the previous local gradient as its hint. Within each instance, let $x _ { j }$ and $g _ { j }$ denote the action and gradient at local round $j .$ . Set $g _ { 0 } = 0$ , fix $\gamma > 0$ , and define

$$
V _ { j } = \sum _ { i = 1 } ^ { j } \lVert g _ { i } - g _ { i - 1 } \rVert _ { * } ^ { 2 } , \qquad c _ { j } = \gamma + \sqrt { V _ { j } / M } .
$$

After receiving $g _ { j }$ , the instance computes

$$
x _ { j + 1 } \in \underset { x \in \mathcal { W } } { \arg \operatorname* { m i n } } \left\{ \langle 2 g _ { j } - g _ { j - 1 } , x \rangle + c _ { j } D _ { \psi } ( x \mid x _ { j } ) \right\} .\tag{23}
$$

Each instance maintains its own iterate, local counter, and accumulated variation $V _ { j }$ , without knowing its eventual number of rounds. For the lower bound, take $\mathcal { W } = [ - 1 , 1 ] , \psi ( x ) = x ^ { 2 } / 2$ , and $M = 2$ , and initialize every instance at $x _ { 1 } = 1$

Proof of the optimistic OMD claim in Proposition 1. Use the block delays constructed above. After observing the played action $w _ { t } .$ , the adversary chooses the linear loss $f _ { t } ( w ) = g _ { t } w$ , where $g _ { t } = 1$ if $x _ { t } \geq 0$ and $g _ { t } = - 1$ otherwise. This adaptive choice is permitted by the online protocol of Section 2. It depends only on the played action, so the argument applies to every realized assignment, including those generated by a randomized rule. Against the common comparator $u = 0$ , the loss diference at round t is $g _ { t } x _ { t } = | x _ { t } |$

Consider an instance with $n \geq 1$ local rounds. In its local notation, $g _ { j } x _ { j } = | x _ { j } | , | g _ { j } | = 1$ , and $V _ { j } \leq 4 j$ . The update (23) therefore becomes

$$
x _ { j + 1 } = \Pi _ { [ - 1 , 1 ] } \left( x _ { j } - { \frac { 2 g _ { j } - g _ { j - 1 } } { c _ { j } } } \right) , \qquad c _ { j } \leq \gamma + { \sqrt { 2 j } } ,
$$

where $\Pi _ { [ - 1 , 1 ] } ( x ) = \arg \operatorname* { m i n } _ { y \in [ - 1 , 1 ] } | y - x | ^ { 2 }$ denotes the Euclidean projection onto $[ - 1 , 1 ]$ . The quantity $2 g _ { j } - g _ { j - 1 }$ has sign $g _ { j }$ and magnitude at least one, including at $j = 1$ because $g _ { 0 } = 0$

Thus the unprojected step moves toward zero and possibly beyond it. The distance to the boundary in this direction is $1 + | x _ { j } |$ , so

$$
| x _ { j + 1 } - x _ { j } | = \operatorname* { m i n } \left\{ \frac { \big | 2 g _ { j } - g _ { j - 1 } \big | } { c _ { j } } , 1 + | x _ { j } | \right\} \geq \operatorname* { m i n } \{ c _ { j } ^ { - 1 } , 1 \} \geq \frac { 1 } { ( \gamma + \sqrt { 2 } ) \sqrt { j } } .
$$

For $n \geq 2$ , the triangle inequality yields

$$
\sum _ { j = 1 } ^ { n - 1 } \lvert x _ { j + 1 } - x _ { j } \rvert \leq \sum _ { j = 1 } ^ { n - 1 } \lvert x _ { j + 1 } \rvert + \lvert x _ { j } \rvert = \lvert x _ { 1 } \rvert + 2 \sum _ { j = 2 } ^ { n - 1 } \lvert x _ { j } \rvert + \lvert x _ { n } \rvert \leq 2 \sum _ { j = 1 } ^ { n } \lvert x _ { j } \rvert ,
$$

so applying this with the previous display yields

$$
2 \sum _ { j = 1 } ^ { n } | x _ { j } | \geq \sum _ { j = 1 } ^ { n - 1 } \left| x _ { j + 1 } - x _ { j } \right| \geq { \frac { 1 } { \gamma + { \sqrt { 2 } } } } \sum _ { j = 1 } ^ { n - 1 } { \frac { 1 } { \sqrt { j } } } \geq { \frac { n - 1 } { ( \gamma + { \sqrt { 2 } } ) { \sqrt { n } } } } \geq { \frac { \sqrt { n } } { 2 ( \gamma + { \sqrt { 2 } } ) } } .
$$

Hence $\textstyle \sum _ { j = 1 } ^ { n } | x _ { j } | \geq { \sqrt { n } } / ( 4 ( \gamma + { \sqrt { 2 } } ) )$ . This bound also holds for $n = 1$ , since $x _ { 1 } = 1$ . Summing over instances against the same comparator and applying (22) gives

$$
\mathrm { R e g } _ { T } \geq \mathrm { R e g } _ { T } ( 0 ) = \sum _ { t = 1 } ^ { T } | x _ { t } | \geq \frac { 1 } { 4 ( \gamma + \sqrt { 2 } ) } \sum _ { k } \sqrt { n _ { k } } \geq \frac { \sqrt { ( T + d _ { \mathrm { t o t } } ) ( 1 + \log { q _ { \mathrm { m a x } } } ) } } { 4 \sqrt { 2 } ( \gamma + \sqrt { 2 } ) } .
$$

This inequality holds for every realization of the assignment rule. For each fixed $\gamma > 0$ , it implies the stated regret lower bound, with a constant independent of $T$ and $q _ { \mathrm { m a x } } = \sigma _ { \mathrm { m a x } } + 1$ □

## C Online convex optimization with arbitrary losses

In this section, we consider single-player online convex optimization with arbitrary convex losses and arbitrary delays, and we show that MAO-OMD with optimism weight $\gamma = 2 G / \sqrt { M }$ achieves the optimal regret rate under delays.

Setting. At each round $t \in [ T ]$ , the learner chooses $w _ { t } \in \mathcal W$ , and an adversary chooses a convex loss $f _ { t } : \mathcal { W } \xrightarrow { } \mathbb { R }$ . The learner incurs the loss $f _ { t } ( w _ { t } )$ , and a subgradient $g _ { t } \in \partial f _ { t } ( w _ { t } )$ is revealed at the end of round $t + d _ { t }$ , where the delays $d _ { t }$ are arbitrary and unknown. We assume that $\| g _ { t } \| _ { * } \leq G$ for every $t \in [ T ]$ and that the learner knows $G ,$ , and the regret is $\begin{array} { r } { \mathrm { R e g } _ { T } : = \operatorname* { s u p } _ { u \in \mathcal { W } } \sum _ { t = 1 } ^ { T } \bigr ( f _ { t } ( w _ { t } ) - f _ { t } ( u ) \bigr ) } \end{array}$ We impose no smoothness assumptions on the loss sequence.

Algorithm. The learner runs Algorithm 1 with a regularizer $\psi$ satisfying Assumption 2 and optimism weight $\gamma = 2 G / \sqrt { M }$ . With this choice, the first term in Equation (16) is $2 \gamma M q _ { \mathrm { m a x } } =$ $4 G \sqrt { M } q _ { \mathrm { m a x } } .$

Corollary 1. For arbitrary convex losses with $\| g _ { t } \| _ { * } \leq G$ and arbitrary unknown delays, MAO-OMD with optimism weight $\gamma = 2 G / \sqrt { M }$ guarantees $\mathrm { R e g } _ { T } = O \left( G { \sqrt { M ( T + d _ { \mathrm { t o t } } ) } } \right)$

Proof. Each hint is an observed subgradient or zero, so $\| g _ { t } - h _ { t } \| _ { * } \leq 2 G$ . By Fact $2 ( \mathrm { b } ) , \Sigma _ { t = 1 } ^ { T } \kappa _ { t } \vert \vert g _ { t } -$ $h _ { t } \| _ { * } ^ { 2 } \leq 4 G ^ { 2 } ( T + 2 d _ { \mathrm { t o t } } )$ . By convexity, regret is bounded by linearized regret. Applying (16) with $\gamma = 2 G / \sqrt { M }$ and dropping the nonpositive stability term yields

$$
\begin{array} { r l } & { \mathrm { R e g } _ { T } \leq 4 G \sqrt { M } q _ { \mathrm { m a x } } + 4 \sqrt { 2 } G \sqrt { M ( T + 2 d _ { \mathrm { t o t } } ) } } \\ & { \qquad \leq ( 4 + 4 \sqrt { 2 } ) G \sqrt { M ( T + 2 d _ { \mathrm { t o t } } ) } } \\ & { \qquad \leq ( 8 + 4 \sqrt { 2 } ) G \sqrt { M ( T + d _ { \mathrm { t o t } } ) } , } \end{array}
$$

where the second inequality uses $q _ { \mathrm { m a x } } \le \sqrt { T + 2 d _ { \mathrm { t o t } } }$ from Fact $1 ( \mathrm { e } )$ , and the last inequality uses $T + 2 d _ { \mathrm { t o t } } \leq 2 ( T + d _ { \mathrm { t o t } } )$ □

The dependence on $T + d _ { \mathrm { t o t } }$ matches the optimal regret rate of online learning with delays [QK15].

## D Auxiliary results

We collect basic properties of the delay sets and congestion quantities defined in Section 2.

Fact 1 (Delay-set identities). The following properties hold for every $t \in [ T ]$

(a) $\mathcal { O } _ { t } \sqcup \mathcal { P } _ { t } = [ t - 1 ] , \mathcal { O } _ { t + 1 } = \mathcal { O } _ { t } \sqcup \mathcal { N } _ { t } , \mathcal { A } _ { t } = \mathcal { P } _ { t + 1 } \sqcup \mathcal { N } _ { t } ;$

(b) $\mathcal { N } _ { 1 } , \ldots , \mathcal { N } _ { T } \ p a r t i t i o n \ [ T ] ;$

(c) $q _ { t + 1 } = q _ { t } + 1 - | \mathcal { N } _ { t } | \leq q _ { t } + 1 , q _ { 1 } = 1 , \mathcal { P } _ { 1 } = \mathcal { P } _ { T + 1 } = \emptyset ;$

(d) $\textstyle \sum _ { t \in [ T ] } \sigma _ { t } = d _ { \mathrm { t o t } }$ under the assumption that $t + d _ { t } \leq T$ for every $t ;$

(e) $q _ { \mathrm { m a x } } ^ { 2 } \leq T + 2 d _ { \mathrm { t o t } }$

Proof of Fact 1. (a) A round $s < t$ has either $s + d _ { s } < t$ or $s + d _ { s } \geq t ;$ the elements of $O _ { t + 1 } \setminus O _ { t }$ are the $s < t$ with $s + d _ { s } = t ;$ and $\mathcal { A } _ { t } = \{ s \leq t : s + d _ { s } \geq t \}$ splits according to $s + d _ { s } = t$ (then $s \in \mathcal { N } _ { t } )$ or $s + d _ { s } \geq t + 1$ (then $s \in \mathcal { P } _ { t + 1 } )$ . (b) Each s lies in $\mathcal { N } _ { s + d _ { s } }$ and in no other $\begin{array} { r l } { \mathcal { N } _ { t } . } & { { } ( \mathrm { c } ) } \end{array}$ By (a), $q _ { t } = | \mathcal { P } _ { t + 1 } | + | \mathcal { N } _ { t } | = q _ { t + 1 } - 1 + | \mathcal { N } _ { t } | ; \mathcal { P } _ { 1 } = \emptyset$ because no round precedes round 1, and $\mathcal { P } _ { T + 1 } = \emptyset$ because every gradient arrives by the end of round T. $\begin{array} { r } { \mathrm { ( d ) } \sum _ { t } \sigma _ { t } = \sum _ { t } \sum _ { s } \mathbb { I } \{ s < t \leq s + d _ { s } \} = \sum _ { s } d _ { s } } \end{array}$ (e) Let $\sigma _ { t } = \sigma _ { \operatorname* { m a x } } = k$ . Since $\sigma _ { 1 } = 0$ and σ grows by at most one per round by $( \mathrm { c } ) , t \geq k + 1$ and $\sigma _ { t - j } \geq k - j$ for $0 \le j \le k$ , so $d _ { \mathrm { t o t } } \geq \sum _ { j = 0 } ^ { k } ( k - j ) = k ( k + 1 ) / 2$ by (d), and $T \geq k + 1$ . Hence $q _ { \mathrm { m a x } } ^ { 2 } = k ( k + 1 ) + ( k + 1 ) \leq 2 d _ { \mathrm { t o t } } + T .$ □

Fact 2 (Congestion properties). The quantities $\kappa _ { s , t }$ and $\kappa _ { s }$ satisfy the following properties:

(a) We have $\kappa _ { t , t } = q _ { t }$ , and $t \mapsto \kappa _ { s , t }$ is nondecreasing for each fixed s. Moreover,

$$
q _ { t } \leq \kappa _ { s , t } \leq \kappa _ { s } \leq q _ { \operatorname* { m a x } } \qquad f o r \ e v e r y \ s \in \mathcal { A } _ { t } ,
$$

with $\kappa _ { s , t } = \kappa _ { s }$ whenever $s \in \mathcal { N } _ { t }$

(b) The following bound holds:

$$
\sum _ { t = 1 } ^ { T } \kappa _ { t } \leq T + 2 d _ { \mathrm { t o t } } .
$$

Consequently, for all $a _ { 1 } , \dots , a _ { T } \geq 0$

$$
\sum _ { t = 1 } ^ { T } \kappa _ { t } a _ { t } \leq \operatorname* { m i n } \biggl \{ q _ { \operatorname* { m a x } } \sum _ { t = 1 } ^ { T } a _ { t } , \ : ( T + 2 d _ { \operatorname { t o t } } ) \operatorname* { m a x } _ { t \in [ T ] } a _ { t } \biggr \} \ .
$$

Proof of Fact 2. (a) is immediate from the definition. (b) A round pending at some $r \in \{ s , \ldots , s +$ $d _ { s } \}$ was either pending at round s or is one of the rounds $s , \ldots , r - 1$ , so $q _ { r } \le 1 + \sigma _ { s } + ( r - s ) \le$ $1 + \sigma _ { s } + d _ { s }$ and $\kappa _ { s } \leq 1 + \sigma _ { s } + d _ { s } ;$ summing and using Fact 1(d) gives $\begin{array} { r } { \sum _ { s } \kappa _ { s } \leq T + 2 d _ { \mathrm { t o t } } } \end{array}$ . The second claim follows from $\kappa _ { s } \leq q _ { \mathrm { m a x } }$ and $a _ { s } \leq \operatorname* { m a x } _ { r } a _ { r }$ □

Lemma 7. Let $( \delta _ { t } ) _ { t }$ and $( \lambda _ { s } ^ { ( t ) } ) _ { s , t }$ be arbitrary non-negative sequences of real numbers. Then

$$
\sum _ { t = 1 } ^ { T } \sum _ { s \in \mathcal { N } _ { t } } \lambda _ { s } ^ { ( t + 1 ) } \delta _ { s } = \sum _ { t = 1 } ^ { T } \lambda _ { t } ^ { ( t + d _ { t } + 1 ) } \delta _ { t }
$$

Proof. Observe that by definition of $\mathcal { N } _ { t } = \{ s \in [ t ] : s + d _ { s } = t \}$ , we have

$$
\begin{array} { c } { \displaystyle \sum _ { t = 1 } ^ { T } \displaystyle \sum _ { s \in \mathcal { N } _ { t } } \lambda _ { s } ^ { ( t + 1 ) } \delta _ { s } = \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { s = 1 } ^ { t } \mathbf { 1 } \left\{ t = s + d _ { s } \right\} \lambda _ { s } ^ { ( t + 1 ) } \delta _ { s } } \\ { = \displaystyle \sum _ { s = 1 } ^ { T } \displaystyle \sum _ { t = s } ^ { T } \mathbf { 1 } \left\{ t = s + d _ { s } \right\} \lambda _ { s } ^ { ( t + 1 ) } \delta _ { s } } \\ { = \displaystyle \sum _ { t = 1 } ^ { T } \lambda _ { t } ^ { ( t + d _ { t } + 1 ) } \delta _ { t } } \end{array}
$$

where the last line observes that the indicator is non-zero only at $t = s + d _ { s }$

Lemma 8. Let $\lambda _ { 1 } , . . . , \lambda _ { T } \ge 0$ , let $\gamma \geq \sqrt { \operatorname* { m a x } _ { t } \lambda _ { t } } > 0$ , and let $\alpha _ { 1 } , \hdots , \alpha _ { T } > 0$ satisfy $\alpha _ { t } \leq 1 / q _ { t }$ for every $t \in [ T ]$ . Define $0 = \eta _ { 1 } \leq \cdot \cdot \cdot \leq \eta _ { T + 1 }$ such that, for all $t \in [ T ]$

$$
\eta _ { t + 1 } = \eta _ { t } + \sum _ { s \in \mathcal { N } _ { t } } \frac { \lambda _ { s } } { \gamma + \eta _ { s } \alpha _ { s } } = \sum _ { s \in \mathcal { O } _ { t + 1 } } \frac { \lambda _ { s } } { \gamma + \eta _ { s } \alpha _ { s } } .
$$

Then,

$$
\eta _ { T + 1 } = \sum _ { t = 1 } ^ { T } \frac { \lambda _ { t } } { \gamma + \eta _ { t } \alpha _ { t } } \leq \sqrt { 2 \sum _ { t = 1 } ^ { T } \frac { \lambda _ { t } } { \alpha _ { t } } } \ .
$$

Proof. For every $t \in [ T ]$ , define $\begin{array} { r } { z _ { t } : = \frac { \lambda _ { t } } { \gamma + \eta _ { t } \alpha _ { t } } } \end{array}$ and observe that $\begin{array} { r } { \eta _ { t } = \sum _ { s \in \mathcal { O } _ { t } } z _ { s } } \end{array}$ by the fact that $\mathcal { O } _ { t } = \mathcal { N } _ { 1 } \sqcup \cdots \sqcup \mathcal { N } _ { t - 1 }$ . By assumption, we have that $\begin{array} { r } { z _ { t } \le \frac { \lambda _ { t } } { \gamma } \le \gamma } \end{array}$ . Now let $\begin{array} { r } { Z _ { t } : = \sum _ { k = 1 } ^ { t } z _ { k } } \end{array}$ for all $t \in \{ 0 , \ldots , T \}$ and observe that, when $t \geq 1$ 2

$$
Z _ { t - 1 } = \eta _ { t } + \sum _ { s < t : s \notin { \mathcal { O } _ { t } } } z _ { s } = \eta _ { t } + \sum _ { s \in \mathcal { P } _ { t } } z _ { s } \le \eta _ { t } + \gamma \sigma _ { t } .
$$

Therefore, by telescoping,

$$
\begin{array} { r l r l } & { v _ { i + 1 } ^ { 2 } = \frac { \gamma } { \lambda _ { i } } ( \mathcal { R } ^ { 2 } - \mathcal { R } _ { i - 1 } ^ { 2 } ) } \\ & { \quad - \frac { \gamma } { \lambda _ { i } } ( \mathcal { R } ^ { 2 } - \mathcal { R } _ { i - 1 } ^ { 2 } | \mathcal { R } _ { 1 } \mid \mathcal { Z } _ { 1 } , \mathrm { i } ) } \\ & { \quad - \frac { \gamma } { \lambda _ { i } } ( \mathcal { R } _ { i } - \mathcal { R } _ { i - 1 } | \mathcal { R } _ { 1 } \mid \mathcal { Z } _ { 1 } , \mathrm { i } ) } \\ & { \quad - \frac { \gamma } { \lambda _ { i } } ( \mathcal { R } _ { i } ^ { 2 } - \mathcal { R } _ { i } ^ { 2 } | \mathcal { R } _ { i } \mid \mathcal { Z } _ { 1 } , \mathrm { i } ) } \\ & { \quad - \frac { \gamma } { \lambda _ { i } } \frac { \gamma } { \lambda _ { i } } ( \mathcal { R } _ { i } ^ { 2 } + \lambda _ { i } ) } \\ & { \quad \le \frac { \gamma } { \lambda _ { i } } \frac { \gamma } { \lambda _ { i } } ( v _ { i + 1 } ^ { 2 } + v _ { i + 2 } ) } & & { \quad ( \mathcal { Z } _ { 1 } = \Xi _ { 1 } + \lambda _ { i } ) } \\ & { \quad - \frac { \gamma } { \lambda _ { i } } \frac { \gamma } { \lambda _ { i } } ( v _ { i + 1 } ^ { 2 } + v _ { i + 1 } ) } \\ & { \quad - \frac { \gamma } { \lambda _ { i } } \frac { \gamma } { \lambda _ { i } } ( v _ { i + 1 } ^ { 2 } + v _ { i + 1 } ) } & & { \quad ( \mathcal { Z } _ { 1 } = \Xi _ { 1 } + 1 ) } \\ & { \quad \le \frac { 2 \gamma } { \lambda _ { i } } \frac { \gamma } { \lambda _ { i } } ( \Xi _ { i } + \Xi _ { i } ^ { 2 } ) } \\ &  \quad - \frac { \gamma } { \lambda _ { i } } \frac { \gamma } \end{array}
$$

Taking the square root on both sides proves the claim.