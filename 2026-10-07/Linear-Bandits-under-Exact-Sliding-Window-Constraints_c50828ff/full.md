# Linear Bandits under Exact Sliding-Window Constraints

Seyed Mohammad Hadi Hosseini hadi.hosseini0171@gmail.com

Yasin Abbasi-Yadkori yasin.abbasi@gmail.com

Sattar Vakili s.vakili@ucl.ac.uk

## Abstract

We study linear bandits under exact slidingwindow constraints, where every consecutive block of actions must belong to a prescribed feasible set. In the ofline setting, where the reward function is known, we show that convexity and cyclic-shift invariance make a stationary solution optimal when w | T and within an additive O(w) gap otherwise. In the online setting, we show that geometric structure alone is insuficient for learning, and sublinear regret can be impossible. We introduce a transition diameter τ that quantifies feasible reachability and develop a rare-switching OFUL algorithm with regret $\widetilde { \mathcal { O } } ( d \sqrt { T } + \dot { \tau } d + w )$ against the oflineoptimal feasible trajectory. Finally, we remove cyclic invariance and consider general sliding-window constraints, where optimal behavior may be non-stationary. We represent recent action history as the state of a finite-memory control problem and introduce a history-state diameter D that measures feasible communication between viable histories. Combining optimistic remaininghorizon planning with rare policy updates, we obtain a regret bound of $\widetilde { \mathcal { O } } ( d \sqrt { T } + d D + w )$ We evaluate our approach on real-world and synthetic benchmarks, showing that it maintains exact feasibility while achieving reward and regret comparable to baselines with substantially fewer policy updates.

## 1 INTRODUCTION

Many sequential decision-making problems impose constraints not only on the current action, but also on how actions evolve over time. This occurs, for example, in online resource allocation (Zhang and Cheung, 2026), energy systems with ramping and rollinghorizon operation (Mathias et al., 2024), constrained adaptive control (Chattopadhyay et al., 2025), and safety-critical control (Ames et al., 2017; Capone et al., 2025). In these settings, a decision that looks good now may restrict future choices because its consequences persist over time. We model this dependence through sliding-window constraints. At each round t, the learner chooses an action $x _ { t }$ , and every consecutive block of w actions must satisfy

$$
( x _ { t - w + 1 } , \dots , x _ { t } ) \in { \mathcal { C } } , \qquad t = w , \dots , T ,
$$

for a prescribed feasible set $\mathcal { C } .$

Unlike pointwise constraints (Amani et al., 2019; Gangrade et al., 2024), which enforce feasibility separately at each round, and long-term constraints (Mahdavi et al., 2012; Jenatton et al., 2016; Yi et al., 2021; Bernasconi et al., 2025), which control feasibility only in aggregate over time, our setting requires every realized sliding window to satisfy the constraint exactly. Thus, exploration itself must respect the temporal constraint. The problem can be viewed as a finite-memory control problem in which the most recent w − 1 actions form the state, connecting our formulation to constrained MDPs (Altman, 1999; Stradi et al., 2024). Unlike a generic constrained MDP, where rewards and transition dynamics may depend on the state, in our online linear-bandit setting the history state determines which actions are feasible next, while rewards are governed by a single d-dimensional linear parameter shared across all histories. A detailed comparison with linear, tabular, and constrained reinforcement learning is provided in Appendix C.

Our analysis is organized around three structural ideas: geometry, which shapes the structure of feasible solutions; reachability, which determines whether desirable feasible behaviors can be connected; and planning, which becomes necessary when optimal behavior is non-stationary.

We first consider the ofline problem, where the reward function is known. When C is convex and invariant under cyclic shifts, cyclic symmetrization shows that stationary behavior is exactly optimal when w | T. For arbitrary horizons, the best stationary solution is within an additive O(w) of the ofline optimum (Theorem 4.4).

In the online setting, where the reward function is unknown, the geometric structure alone is not enough: even if good stationary feasible actions exist, moving between them may be impossible without violating the constraint, making sublinear regret impossible in general (Proposition 5.2). We thus introduce a transition diameter τ, which quantifies feasible reachability between stationary histories, and develop a rareswitching algorithm based on OFUL (Abbasi-Yadkori et al., 2011) that preserves exact sliding-window feasibility and achieves expected pseudo-regret $\widetilde { \mathcal { O } } ( d \sqrt { T } +$ τd + w) (Theorem 5.4).

Finally, we move beyond cyclic invariance and consider general sliding-window constraints, where optimal behavior may be non-stationary. We use the most recent $w - 1$ actions as the state of a finite-memory control problem and introduce a history-state diameter D, which measures feasible communication between viable histories. We then combine optimistic remaining-horizon planning with rare policy updates to obtain expected pseudo-regret $\widetilde { \mathcal { O } } ( d \sqrt { T } + d D + w )$ (Theorem 6.5).

Furthermore, we study two real-world sequential decision-making applications: online recommendation with diversity constraints and online LLM routing with cost constraints. In the recommendation setting, the learner aims to maximize user reward while preventing overly repetitive recommendations of the same categories through sliding-window diversity constraints. In the LLM routing setting, the learner selects among language models with diferent response qualities and inference costs while satisfying a slidingwindow resource budget. We complement these experiments with controlled synthetic studies that illustrate the roles of reachability and planning identified by our theoretical analysis.

Our contributions are threefold. (i) For convex, cyclically invariant sliding-window constraints, we show that stationary behavior is optimal when w | T and within $\mathcal { O } ( w )$ of the ofline optimum otherwise. We further introduce the transition diameter τ and a feasibility-preserving rare-switching OFUL algorithm with expected pseudo-regret $\tilde { \mathcal { O } } ( d \sqrt { T } + \tau d + \bar { w } )$ . (ii) For general sliding-window constraints, we formulate the problem as finite-memory control, introduce the history-state diameter D, and obtain expected pseudoregret $\widetilde { \mathcal { O } } ( d \sqrt { T } + d D + w )$ via optimistic remaininghorizon planning with rare updates. (iii) Experiments on real-world recommendation, LLM routing and synthetic tasks show that our approach maintains exact feasibility, reduces regret in non-cyclic settings, and requires substantially fewer policy updates.

## 2 RELATED WORK

Constrained online learning. Online decisionmaking under feasibility and resource constraints has been studied extensively. Long-term constrained methods (Mannor et al., 2009; Mahdavi et al., 2012; Jenatton et al., 2016; Yi et al., 2021; Bernasconi et al., 2025) typically allow per-round violations while controlling cumulative constraint violation. Bandits with knapsacks and related models (Badanidiyuru et al., 2018; Agrawal and Devanur, 2019; Castiglioni et al., 2024) impose global budget or convex constraints over the horizon, while safe and constrained bandits (Amani et al., 2019; Gangrade et al., 2024) study pointwise feasibility at each round. Exact slidingwindow constraints include pointwise feasibility as a special case when $w = 1$ , while setting $w = T$ allows horizon-wide constraints to be represented within the same framework.

Temporal constraints and learning with memory. Several related models (Liakopoulos et al., 2019; Agrawal and Tulabandhula, 2018; Anava et al., 2015; Abdullah et al., 2026) consider temporal structure induced by recent decisions. Cautious regret (Liakopou los et al., 2019) imposes window-based resource constraints on the benchmark while allowing residual constraint violations, whereas temporal stochastic bandits (Agrawal and Tulabandhula, 2018) allow rewards to depend on recent action history. Online learning with memory (Anava et al., 2015; Hebbar and Langbort, 2024) similarly models losses that depend on past decisions. More recently, online convex optimization with finite-memory losses and constraints (Abdullah et al., 2026) considers cumulative constraint violation rather than requiring every sliding window to satisfy the constraint. In contrast, our model couples decisions through exact sliding-window feasibility: every window must remain feasible while maximizing reward, so exploration must respect these temporal constraints throughout learning.

Constrained control and finite-memory planning. Constrained Markov decision processes (Altman, 1999; Stradi et al., 2024) provide a general framework for sequential decision-making under state and resource constraints, while safe and learning-based control (Ames et al., 2017; Capone et al., 2025) seeks to maintain safety throughout interaction. Our setting is specified directly through overlapping constraints on recent actions. Without cyclic symmetry, the most recent $w { - 1 }$ actions naturally form a finite-memory state, yielding a deterministic planning problem with statedependent feasible actions. Unlike a generic CMDP, the history dynamics are deterministic and known, while reward uncertainty is shared across history states through a single low-dimensional linear parameter.

Table 1: Comparison with related sequential-learning frameworks.
<table><tr><td>Setting</td><td>Action-wise constraints</td><td>Horizon-wide constraints</td><td>History dependence</td><td>Exact-window constraints</td></tr><tr><td>Safe/constrained bandits</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Long-term constrained learning</td><td></td><td>√</td><td></td><td></td></tr><tr><td>Online learning with memory</td><td></td><td></td><td>√</td><td></td></tr><tr><td>Switching-constrained bandits</td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>This work</td><td> $\checkmark ^ { \dagger }$ </td><td> $\checkmark ^ { \ddagger }$ </td><td>√</td><td>√</td></tr></table>

† Pointwise feasibility is recovered when $w = 1 .$ ‡ Horizon-wide constraints can be represented when $w = T$

Limited switching and switching costs. Our online algorithms build on OFUL (Abbasi-Yadkori et al., 2011) and relate to work on limited adaptivity and switching. Limited-switching methods (Altschuler and Talwar, 2018) constrain the number of action changes. Switching-cost models (Koren et al., 2017; Rouyer et al., 2021) instead penalize changes between actions. In contrast, switching in our setting is constrained by feasibility itself: unlike standard OFUL, where the op timistic action can be selected directly, an immediate change may be infeasible under the sliding-window constraint, requiring a sequence of intermediate feasible actions.

Table 1 compares these settings along several key dimensions.

## 3 PROBLEM FORMULATION

Let $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ denote the action space, let T be the horizon, let $1 \leq w \leq T$ be the window length, and let ${ \mathcal { C } } \subseteq { \mathcal { X } } ^ { w }$ denote the set of feasible action windows. An action sequence $\boldsymbol { x } _ { 1 : T } = ( x _ { 1 } , \dots , x _ { T } )$ is feasible if

$$
( x _ { t - w + 1 } , \dots , x _ { t } ) \in { \mathcal { C } } , \qquad t = w , \dots , T .
$$

Thus, every realized block of w consecutive actions must satisfy the constraint exactly. We first consider the setting in which the reward function $f : \mathcal { X } \to \mathbb { R }$ is known. The ofline optimal value is

$$
V _ { T } ( f ) : = \operatorname* { m a x } _ { x _ { 1 } , \ldots , x _ { T } \in \mathcal { X } } \sum _ { t = 1 } ^ { T } f ( x _ { t } )
$$

subject to

$$
( x _ { t - w + 1 } , \dots , x _ { t } ) \in { \mathcal { C } } , \qquad t = w , \dots , T .
$$

We next consider the online setting with an unknown linear reward

$$
f _ { \theta ^ { \star } } ( x ) = \langle \theta ^ { \star } , x \rangle , \qquad \theta ^ { \star } \in \mathbb { R } ^ { d } .
$$

At each round t, the learner selects an action $x _ { t } \in \mathcal { X }$ and observes

$$
Y _ { t } = \langle \theta ^ { \star } , x _ { t } \rangle + \eta _ { t } .
$$

Let $\mathcal { F } _ { t } = \sigma ( x _ { 1 } , Y _ { 1 } , . . . , x _ { t } , Y _ { t } )$ denote the history after round t. We require $x _ { t }$ to be $\mathcal { F } _ { t - 1 }$ -measurable. The noise process is conditionally mean-zero and R-sub-Gaussian:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \eta _ { t } \mid \mathcal { F } _ { t - 1 } \right] = 0 , } \\ & { } \\ & { \mathbb { E } \left[ \exp ( u \eta _ { t } ) \mid \mathcal { F } _ { t - 1 } \right] \leq \exp \left( \displaystyle \frac { u ^ { 2 } R ^ { 2 } } { 2 } \right) , \qquad u \in \mathbb { R } . } \end{array}
$$

A policy π is admissible if each action $x _ { t }$ is $\mathcal { F } _ { t - 1 ^ { - } }$ measurable and the resulting trajectory satisfies

$$
( x _ { t - w + 1 } , \dots , x _ { t } ) \in { \mathcal { C } } , \qquad t = w , \dots , T ,
$$

almost surely. For a fixed parameter $\theta ^ { \star }$ , define the ofline optimal benchmark

$$
V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) : = V _ { T } ( f _ { \theta ^ { \star } } ) .
$$

The expected pseudo-regret of an admissible policy π relative to the ofline-optimal feasible trajectory is

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) : = V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) - \mathbb { E } _ { \pi } \left[ \sum _ { t = 1 } ^ { T } \langle \theta ^ { \star } , x _ { t } \rangle \right] .
$$

## 4 GEOMETRY: OFFLINE OPTIMIZATION UNDER CYCLIC SYMMETRY

In this section, we assume that the reward function is known and show how geometric structure in the feasible set can simplify the ofline optimization problem.

Assumption 4.1. The window-feasibility set ${ \mathcal { C } } \subseteq { \mathcal { X } } ^ { w }$ is nonempty, closed, and convex. Define the cyclicshift operator

$$
\mathcal { S } ( x _ { 1 } , \ldots , x _ { w } ) = ( x _ { 2 } , \ldots , x _ { w } , x _ { 1 } ) .
$$

We assume that C is invariant under cyclic shifts: for every $( x _ { 1 } , \dots , x _ { w } ) \in \mathcal { C }$ and every $k \in \{ 0 , \ldots , w - 1 \}$ ,

$$
S ^ { k } ( x _ { 1 } , \dots , x _ { w } ) \in \mathcal { C } .
$$

Assumption 4.2. The action space $ { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ is nonempty, compact, and convex. The reward function $f : \mathcal { X } \to \mathbb { R }$ is concave and upper semicontinuous.

Remark 4.3. Assumption 4.1 is satisfied by a broad class of sliding-window constraints. Appendix D provides representative examples.

Theorem 4.4. Suppose Assumptions $\it 4 . 1$ and $4 . 2$ hold, and let

$$
{ \mathcal { Z } } : = \{ z \in { \mathcal { X } } : ( z , \ldots , z ) \in { \mathcal { C } } \} .
$$

Then $\mathcal { Z }$ is nonempty and compact, and there exists $z ^ { \star } \in$ arg ma $\mathrm { x } _ { z \in { \mathcal { Z } } } f ( z )$ . For $T = q w + r$ , where $q \in  { \mathbb { N } } _ { 0 }$ and $0 \leq r < w$ , the stationary trajectory $x _ { t } = z ^ { \star }$ is feasible and

$$
\begin{array} { r l } & { 0 \leq V _ { T } ( f ) - T f ( z ^ { \star } ) \leq r \big ( f _ { \operatorname* { m a x } } - f ( z ^ { \star } ) \big ) } \\ & { \qquad \leq ( w - 1 ) \big ( f _ { \operatorname* { m a x } } - f ( z ^ { \star } ) \big ) . } \end{array}
$$

where $f _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { x \in \mathcal { X } } f ( x )$ . In particular, if w | T, then $x _ { t } = z ^ { \star }$ is optimal and $V _ { T } ( f ) = T f ( z ^ { \star } )$

The proof is provided in Appendix G.1. The depen dence on w is unavoidable: Appendix G.2 gives an example with a $\Theta ( w )$ gap.

## 5 REACHABILITY: ONLINE LEARNING UNDER CYCLIC SYMMETRY

We now turn to the online setting, where learning requires feasible movement between near-optimal stationary actions, motivating the notion of reachability. We assume that, for known constants $L , S > 0$

$$
\| x \| _ { 2 } \leq L \qquad { \mathrm { f o r ~ a l l ~ } } x \in { \mathcal { X } } , \qquad \| \theta ^ { \star } \| _ { 2 } \leq S .
$$

Definition 5.1. The expected pseudo-regret relative to the best stationary feasible action is

$$
\mathrm { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; \theta ^ { \star } ) : = T \langle \theta ^ { \star } , z ^ { \star } \rangle - \mathbb { E } _ { \pi } \left[ \sum _ { t = 1 } ^ { T } \langle \theta ^ { \star } , x _ { t } \rangle \right] .
$$

Writing $T = q w + r ,$ , with $0 \leq r < w$ , Theorem 4.4 applied to the linear reward $f _ { \theta ^ { \star } } ( x ) = \langle \theta ^ { \star } , x \rangle$ gives

$$
\begin{array} { r l } & { 0 \leq \Delta _ { T } : = V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) - T \langle \theta ^ { \star } , z ^ { \star } \rangle } \\ & { \qquad \leq r \left( \underset { x \in \mathcal { X } } { \operatorname* { m a x } } \langle \theta ^ { \star } , x \rangle - \langle \theta ^ { \star } , z ^ { \star } \rangle \right) . } \end{array}
$$

Consequently,

$$
\begin{array} { r } { \mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \mathrm { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; \theta ^ { \star } ) + \Delta _ { T } . } \end{array}
$$

## 5.1 Why Geometry Alone Is Insuficient

The ofline assumptions do not guarantee sublinear online regret because they do not ensure that the histories induced by these actions are feasibly connected.

Proposition 5.2. There exists a one-dimensional linear-bandit instance satisfying Assumption 4.1 such that, for every admissible policy π,

$$
\operatorname* { s u p } _ { \theta ^ { \star } \in \{ - 1 , 1 \} } \operatorname { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; \theta ^ { \star } ) \geq T .
$$

Hence, sublinear stationary pseudo-regret cannot be guaranteed under Assumption 4.1 alone.

The proof is deferred to Appendix H.1. For $w = 1$ , set $\tau = 0 .$ . For $w \ge 2 .$ , let

$$
\begin{array} { c } { s = ( x _ { 1 } , \ldots , x _ { w - 1 } ) \in { \mathcal X } ^ { w - 1 } , } \\ { F ( s , x ) = ( x _ { 2 } , \ldots , x _ { w - 1 } , x ) . } \end{array}
$$

We define the stationary history

$$
h ( z ) = ( z , \ldots , z ) \in \mathcal { X } ^ { w - 1 } , \qquad z \in \mathcal { Z } .
$$

For complete histories $s , s ^ { \prime } \in \mathcal { X } ^ { w - 1 }$ , let $d _ { \mathcal { C } } ( s , s ^ { \prime } )$ be the minimum number of feasible updates needed to move from s to $s ^ { \prime } ,$ with $d _ { \cal C } ( s , s ^ { \prime } ) = \infty$ if no such transition exists. The transition diameter is

$$
\tau : = \operatorname* { s u p } _ { z , z ^ { \prime } \in \mathcal { Z } } d _ { \mathcal { C } } \big ( h ( z ) , h ( z ^ { \prime } ) \big ) .
$$

Assumption 5.3. The transition diameter is finite.

Appendix E provides explicit bounds on τ for several common sliding-window constraints. We assume access to a transition oracle Γ that, for any $z , z ^ { \prime } \in { \mathcal { Z } } ,$ returns a feasible transition

$$
\Gamma ( z , z ^ { \prime } ) = ( a _ { 1 } , \ldots , a _ { m } ) , \qquad m \leq \tau .
$$

## 5.2 Feasible Rare-Switching OFUL

We combine OFUL with determinant-doubling episodes to limit switches between stationary actions. We assume access to an optimistic-action oracle Opt and a feasible-transition oracle Γ; details are given in Appendix B. Let $I _ { t } \in \{ 0 , 1 \}$ indicate whether round t is retained for estimation, and define

$$
\begin{array} { r l } & { \mathcal { O } _ { t } : = \{ s < t : I _ { s } = 1 \} , } \\ & { V _ { t } = \lambda I _ { d } + \displaystyle \sum _ { s \in \mathcal { O } _ { t } } x _ { s } x _ { s } ^ { \top } , } \\ & { \widehat { \theta } _ { t } = V _ { t } ^ { - 1 } \displaystyle \sum _ { s \in \mathcal { O } _ { t } } x _ { s } Y _ { s } . } \end{array}
$$

where $\lambda \geq L ^ { 2 }$ . The confidence radius is

$$
\beta _ { t } ( \delta ) = R \sqrt { 2 \log \biggl ( \frac { \operatorname * { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \operatorname * { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \biggr ) } + \sqrt { \lambda } S .
$$

At the start $t _ { k }$ of episode $k ,$ choose

$$
\begin{array} { r l } & { z _ { k } \in \arg \operatorname* { m a x } _ { z \in \mathcal { Z } } \Big \{ \langle \widehat { \theta } _ { t _ { k } } , z \rangle + \beta _ { t _ { k } } ( \delta ) \| z \| _ { V _ { t _ { k } } ^ { - 1 } } \Big \} } \\ & { \quad = \mathrm { O p t } \big ( \widehat { \theta } _ { t _ { k } } , V _ { t _ { k } } , \beta _ { t _ { k } } ( \delta ) \big ) . } \end{array}
$$

The initial action is repeated for at least $w - 1$ rounds. Thereafter, $z _ { k }$ is played until det $( V _ { t } ) > 2 \operatorname* { d e t } ( V _ { t _ { k } } )$ . If the newly computed action $z _ { k + 1 } \neq z _ { k }$ , the learner follows $\Gamma ( z _ { k } , z _ { k + 1 } )$ , which reaches $h ( z _ { k + 1 } )$ in at most τ rounds. Transition observations are excluded from estimation. Algorithm 1 gives the complete procedure.

Theorem 5.4. Suppose the online linear-bandit assumptions of Section 3, Assumptions $\it 4 . 1 \mathrm { - } 4 . \mathrm { } \mathrm { } \mathcal { Q } ,$ , and Assumption 5.3 hold, and that the learner has exact access to Opt and Γ. Then the rare-switching OFUL policy is feasible and satisfies

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \widetilde { \mathcal { O } } \left( d \sqrt { T } + \tau d + w \right) .
$$

The proof is deferred to Appendix H.2. Appendix H.3 shows that the linear dependence on τ is unavoidable by constructing a one-dimensional instance with $\Omega ( \tau )$ regret.

## 6 PLANNING: LEARNING BEYOND CYCLIC SYMMETRY

We now move beyond cyclic symmetry and consider general sliding-window constraints, where the optimal feasible trajectory may be inherently non-stationary.

## 6.1 Finite-Memory Representation

Suppose $w \geq 2$ . A complete history is

$$
s = ( x _ { 1 } , \ldots , x _ { w - 1 } ) \in \mathcal { X } ^ { w - 1 } .
$$

Define

$$
\begin{array} { c } { A ( s ) : = \{ x \in \mathcal { X } : ( s , x ) \in \mathcal { C } \} , } \\ { F ( s , x ) : = ( x _ { 2 } , \ldots , x _ { w - 1 } , x ) . } \end{array}
$$

The n-step viable histories are defined recursively by

$$
\begin{array} { r l } & { S _ { 0 } : = \mathscr { X } ^ { w - 1 } , } \\ & { S _ { n } : = \left\{ s : \exists x \in A ( s ) , \ F ( s , x ) \in S _ { n - 1 } \right\} , \qquad n \geq 1 . } \end{array}
$$

For $s \in S _ { n } .$ , define

$$
\begin{array} { r l } & { J _ { 0 } ^ { \star } ( s ) : = 0 , } \\ & { J _ { n } ^ { \star } ( s ) : = \displaystyle \operatorname* { m a x } _ { \begin{array} { c } { x \in A ( s ) } \\ { F ( s , x ) \in S _ { n - 1 } } \end{array} } \left\{ \langle \theta ^ { \star } , x \rangle + J _ { n - 1 } ^ { \star } ( F ( s , x ) ) \right\} . } \end{array}
$$

Theorem 6.1. For every $n \geq 0$ and $s \in { \mathcal { S } } _ { n } , ~ J _ { n } ^ { \star } ( s )$ equals the optimal reward over all feasible n-step continuations from s. Hence, for $T \geq w - 1$

$$
V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) = \operatorname* { m a x } _ { s \in S _ { T - w + 1 } } \left\{ \sum _ { i = 1 } ^ { w - 1 } \langle \theta ^ { \star } , x _ { i } \rangle + J _ { T - w + 1 } ^ { \star } ( s ) \right\} ,
$$

where $s = ( x _ { 1 } , \dots , x _ { w - 1 } )$

The proof is given in Appendix I.1. This theorem shows that, without cyclic symmetry, the problem can be formulated as a deterministic finite-memory Markov decision process with recent action history as the state, and the Bellman recursion describes the offline optimal trajectory.

## 6.2 History-State Reachability

For complete histories $s , s ^ { \prime } ,$ let $d _ { \mathcal { C } } ( s , s ^ { \prime } )$ be the minimum number of feasible updates needed to move from s to $s ^ { \prime } ,$ with $d _ { \cal C } ( s , s ^ { \prime } ) = \infty$ if no such transition exists. Since $S _ { n } \subseteq S _ { 1 }$ for all $n \geq 1$ , it sufices to measure communication over $S _ { 1 }$

Assumption 6.2. The history-state diameter $D : =$ $\mathrm { s u p } _ { s , s ^ { \prime } \in { \mathcal { S } } _ { 1 } } d _ { \mathcal { C } } \bigl ( s , s ^ { \prime } \bigr )$ is finite and independent of T.

This extends the transition diameter from stationary histories to arbitrary viable histories. For many standard constraints, $D = \mathcal { O } ( w )$ ; see Appendix F. For a bounded reward function $g : \mathcal { X }  \mathbb { R }$ , let $J _ { n } ^ { g } ( s )$ denote the optimal cumulative g-reward over n feasible steps starting from history state s. Define

$$
\begin{array} { r l r } & { } & { \mathrm { r n g } ( g ) : = \displaystyle \operatorname* { s u p } _ { x \in \mathcal { X } } g ( x ) - \displaystyle \operatorname* { i n f } _ { x \in \mathcal { X } } g ( x ) , } \\ & { } & { \mathrm { s p a n } _ { \mathcal { S } _ { n } } ( J _ { n } ^ { g } ) : = \displaystyle \operatorname* { s u p } _ { s \in \mathcal { S } _ { n } } J _ { n } ^ { g } ( s ) - \displaystyle \operatorname* { i n f } _ { s \in \mathcal { S } _ { n } } J _ { n } ^ { g } ( s ) . } \end{array}
$$

Lemma 6.3. Under Assumption 6.2, for every bounded g and $n \geq 1$

$$
\operatorname { s p a n } _ { S _ { n } } ( J _ { n } ^ { g } ) \leq D \operatorname { r n g } ( g ) .
$$

The proof is given in Appendix I.2. For $g ( x ) = \langle \theta ^ { \star } , x \rangle$ ， assuming $\| x \| _ { 2 } \leq L$ and $\| \theta ^ { \star } \| _ { 2 } \leq S$

$$
\operatorname { s p a n } _ { S _ { n } } ( J _ { n } ^ { \star } ) \leq 2 L S D .
$$

## 6.3 Rare-Update Optimistic Planning

We combine finite-memory Bellman planning with optimistic linear-bandit estimation. The learner uses an optimistic reward model based on the current confidence set and updates the Bellman policy only at determinant-doubling times.

$$
V _ { t } = \lambda I _ { d } + \sum _ { q = 1 } ^ { t - 1 } x _ { q } x _ { q } ^ { \top } , \qquad \widehat { \theta } _ { t } = V _ { t } ^ { - 1 } \sum _ { q = 1 } ^ { t - 1 } x _ { q } Y _ { q } ,
$$

with $\lambda \geq L ^ { 2 }$ and

$$
\beta _ { t } ( \delta ) = R \sqrt { 2 \log \biggl ( \frac { \operatorname * { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \operatorname * { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \biggr ) } + \sqrt { \lambda } S .
$$

For $w > 1$ , the learner uses the feasible initialization oracle to obtain the first $w - 1$ actions of a valid trajectory, which define the initial history state, and planning begins at $t _ { 1 } = w$ . At planning time $t _ { k }$ , set

$$
W _ { k } : = V _ { t _ { k } } , \qquad { \widehat { \theta } } _ { k } : = { \widehat { \theta } } _ { t _ { k } } , \qquad \beta _ { k } : = \beta _ { t _ { k } } ( \delta ) ,
$$

and keep these quantities fixed until the next update. Define the clipped optimistic reward

$$
\begin{array} { r } { \overline { { U } } _ { k } ( x ) : = \mathrm { c l i p } _ { [ - L S , L S ] } \left( \langle \widehat { \theta } _ { k } , x \rangle + \beta _ { k } \| x \| _ { W _ { k } ^ { - 1 } } \right) . } \end{array}
$$

Its Bellman value satisfies

$$
\begin{array} { r l } & { J _ { k , 0 } ( s ) : = 0 , } \\ & { J _ { k , n } ( s ) = \underset { x \in A ( s ) } { \operatorname* { m a x } } \quad \big \{ \overline { { U } } _ { k } ( x ) + J _ { k , n - 1 } ( F ( s , x ) ) \big \} . } \\ & { \quad \quad F ( s , x ) \in S _ { n - 1 } } \end{array}
$$

Thus, at round t of episode $k ,$

$$
x _ { t } \in \arg \operatorname* { m a x } _ { \stackrel { x \in A \left( s _ { t } \right) } { F \left( s _ { t } , x \right) \in S _ { T - t } } } \left\{ \overline { { U } } _ { k } ( x ) + J _ { k , T - t } ( F ( s _ { t } , x ) ) \right\} .
$$

A new plan is computed when the determinantdoubling condition det $( V _ { t } ) > 2 \operatorname* { d e t } ( V _ { t _ { k } } )$ is triggered.

Assumption 6.4. At each planning time, the learner can compute an exact Bellman-optimal remaininghorizon policy under $\overline { { U } } _ { k }$ , as well as a feasible initialization when w $> 1$

Assumption 6.4 separates the statistical guarantee from the computational problem of planning. For finite action spaces, exact planning can be implemented by backward dynamic programming, while general continuous constraints may require additional structure.

Theorem 6.5. Suppose the online linear-bandit assumptions $o f$ Section 3 hold, with $\| x \| _ { 2 } \leq L , \| \theta ^ { \star } \| _ { 2 } \leq$ $S ,$ and $\lambda \geq L ^ { 2 }$ . Under Assumptions $6 . 2$ and $6 . 4 ,$ the rare-update optimistic planning policy satisfies

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \widetilde { \mathcal { O } } \left( d \sqrt { T } + d D + w \right) .
$$

The proof is provided in Appendix I.3. In $\mathrm { A p \mathrm { - } }$ pendix I.4, we relax Assumption 6.4 by allowing additive approximate remaining-horizon planning. Algorithm 2 summarizes the complete procedure.

## 7 EXPERIMENTS

In this section, we evaluate our framework in two realworld sequential decision-making settings: a recommendation system with diversity constraints and LLM routing with cost constraints. In the recommendation setting, the learner sequentially selects content categories based on user preferences while enforcing sliding-window diversity constraints to prevent excessive repetition and promote diversified recommendations. In the LLM routing setting, the learner selects among multiple language models with diferent response qualities and inference costs. We impose sliding-window cost constraints to ensure that the total inference cost within each decision window remains below a predefined budget. The synthetic experiments are conducted in controlled settings, with the corresponding ablation studies and sensitivity analyses reported in Appendix J and Appendix K, respectively.

Recommendation with Diversity Constraints We consider an online recommendation problem based on the KuaiRand-1K dataset (Gao et al., 2022). The items are grouped into K content categories, with action set $\mathcal { A } = \{ 1 , \ldots , K \}$ . At each round t, the learner recommends a category $a _ { t } \in { \mathcal { A } }$ and observes a binary reward $Y _ { t } \sim \mathrm { B e r n o u l l i } ( \theta _ { a _ { t } } ^ { \star } )$ , where $\theta _ { j } ^ { \star } \in [ 0 , 1 ]$ denotes the unknown click probability associated with category $j .$ The learner aims to identify the categories preferred by the user and maximize cumulative reward over time. A policy that focuses only on immediate reward may repeatedly recommend a small number of highly preferred categories, leading to overly repetitive suggestions. To maintain recommendation diversity, we impose exact sliding-window exposure constraints that limit how frequently each category can be recommended within any recent window. Further details of this setting are provided in Appendix L.1.

LLM Routing with Cost Constraints We consider an online LLM routing problem based on the LLMRouterBench dataset (Li et al., 2026). The available language models form the action set ${ \mathcal { A } } =$ $\{ 1 , \ldots , K \}$ , where each action corresponds to select ing a diferent LLM for answering a user query. At each round $t ,$ the learner observes an incoming query $q _ { t } ,$ selects a feasible model $a _ { t } \in A .$ and receives a reward $Y _ { t }$ given by the corresponding LLMRouterBench quality score; if no model is feasible, a zero-cost, zeroreward no-op fallback is used. The learner uses query– model features $\phi ( q _ { t } , a _ { t } )$ and a linear predictor to estimate model quality and adaptively route queries across models. The learner aims to maximize response quality while controlling inference cost, since repeatedly selecting high-quality but expensive models can consume the available sliding-window budget and restrict subsequent routing decisions. To enforce resource eficiency, we impose exact sliding-window cost constraints that limit the total inference cost incurred within every recent window. Since future queries are unknown at decision time, the LLM-routing experiment uses myopic feasible selection after observing the current query

Table 2: Recommendation results on KuaiRand-1K over 100 users, with 10 independent simulations per user and horizon. Average reward and normalized pseudo-regret are reported as mean ± 95% confidence interval.
<table><tr><td> $T$ </td><td>Method</td><td>Avg. Reward</td><td>Norm. Pseudo-Regret</td><td>Violations</td><td>Updates</td></tr><tr><td rowspan="4">500</td><td>Rare-Update Planning</td><td> $0 . 4 6 1 8 \pm 0 . 0 3 4 3$ </td><td> $0 . 0 5 4 0 \pm 0 . 0 0 3 3$ </td><td>0.0</td><td>31.8</td></tr><tr><td>Myopic Feasible OFUL</td><td> $0 . 4 6 3 8 \pm 0 . 0 3 4 4$ </td><td> $0 . 0 5 2 0 \pm 0 . 0 0 3 1$ </td><td>0.0</td><td>500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 4 6 3 9 \pm 0 . 0 3 4 4$ </td><td> $0 . 0 5 1 8 \pm 0 . 0 0 3 1$ </td><td>160.6</td><td>500.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 4 6 4 3 \pm 0 . 0 3 4 4$ </td><td> $0 . 0 5 1 5 \pm 0 . 0 0 3 1$ </td><td>177.6</td><td>500.0</td></tr><tr><td rowspan="4">2500</td><td>Rare-Update Planning</td><td> $0 . 4 7 7 2 \pm 0 . 0 3 5 0$ </td><td> $0 . 0 3 8 6 \pm 0 . 0 0 1 9$ </td><td>0.0</td><td>49.5</td></tr><tr><td>Myopic Feasible OFUL</td><td> $0 . 4 8 0 1 \pm 0 . 0 3 5 2$ </td><td> $0 . 0 3 5 6 \pm 0 . 0 0 1 7$ </td><td>0.0</td><td>2500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 4 8 0 3 \pm 0 . 0 3 5 2$ </td><td> $0 . 0 3 5 5 \pm 0 . 0 0 1 7$ </td><td>1803.4</td><td>2500.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 4 8 5 4 \pm 0 . 0 3 5 1$ </td><td> $0 . 0 3 0 4 \pm 0 . 0 0 2 1$ </td><td>1864.2</td><td>2500.0</td></tr><tr><td rowspan="4">5000</td><td>Rare-Update Planning</td><td> $0 . 4 8 4 4 \pm 0 . 0 3 5 3$ </td><td> $0 . 0 3 1 3 \pm 0 . 0 0 1 4$ </td><td>0.0</td><td>57.0</td></tr><tr><td>Myopic Feasible OFUL</td><td> $0 . 4 8 7 6 \pm 0 . 0 3 5 4$ </td><td> $0 . 0 2 8 2 \pm 0 . 0 0 1 2$ </td><td>0.0</td><td>5000.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 4 8 7 6 \pm 0 . 0 3 5 4$ </td><td> $0 . 0 2 8 1 \pm 0 . 0 0 1 3$ </td><td>4163.3</td><td>5000.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 4 9 8 0 \pm 0 . 0 3 5 4$ </td><td> $0 . 0 1 7 7 \pm 0 . 0 0 2 9$ </td><td>4240.8</td><td>5000.0</td></tr></table>

rather than full remaining-horizon viability planning.   
Further details are provided in Appendix L.1.

Baselines. We compare our Rare-Update Optimistic Planning method with three baselines. Myopic Feasible OFUL selects the currently feasible action with the largest optimistic reward and recomputes its optimistic decision at every round, providing a fully updated feasible baseline against which to evaluate the rare-update mechanism. Primal-Dual OFUL incorporates the sliding-window constraints through nonnegative Lagrangian dual variables and balances optimistic reward maximization with constraint violation penal ties, inspired by primal-dual methods for constrained stochastic linear bandits (Liu et al., 2021). Unconstrained OFUL uses the same optimistic reward model but ignores the sliding-window constraints, providing a reference for the cost of enforcing exact feasibility. Further details are provided in Appendix L.

Real-world datasets. For the recommendation problem, we use the KuaiRand-1K dataset (Gao et al., 2022), collected from the Kuaishou short-video platform. The dataset contains 1,000 users, approximately 4.37 million videos, and 11.7 million user–video interactions with sequential histories and category information, enabling content-category recommendation under diversity constraints. For the LLM routing problem, we use LLMRouterBench (Li et al., 2026), a largescale benchmark containing over 400K query instances from 21 datasets evaluated across 33 language models. It provides response quality scores and inference cost information, enabling LLM selection as an online decision problem under resource constraints.

Evaluation metrics. We evaluate performance using four metrics: average expected reward, normalized pseudo-regret, the number of sliding-window constraint violations, and the number of policy updates. The average expected reward measures the utility obtained by the selected actions, while the normalized pseudo-regret evaluates performance relative to the ofline-optimal feasible policy. Constraint violations quantify the number of times a policy violates the corresponding sliding-window constraint, and the number of updates measures the frequency of policy recomputation. Detailed definitions of these metrics and their application-specific formulations are provided in $\mathrm { A p \mathrm { - } }$ pendix L.

Recommendation Performance under Diversity Constraints. Table 2 summarizes the recommendation results at representative horizons. Unconstrained OFUL attains the lowest normalized pseudo-regret, but does so while incurring a large number of slidingwindow constraint violations, particularly at longer horizons. In contrast, Rare-Update Optimistic Planning and Myopic Feasible OFUL satisfy the hard diversity constraints with zero violations across all reported horizons. Rare-Update Optimistic Planning achieves reward and normalized regret close to those of Myopic Feasible OFUL while requiring dramatically fewer policy updates: approximately 32, 49, and 57 updates at $T \ = \ 5 0 0 , \ 2 5 0 0$ , and 5000, respectively, compared with one update per round for Myopic Feasible OFUL. These results highlight the efectiveness of the rare-update mechanism in maintaining competitive constrained performance while substantially reducing the frequency of policy recomputation.

Table 3: LLM routing results with sliding-window length $w = 5$ and budget $B = 0 . 4 0$ over 10 independent simulations for each horizon. Average reward and normalized pseudo-regret are reported as $\mathrm { m e a n } \pm 9 5 \%$ confidence interval.
<table><tr><td> $T$ </td><td>Method</td><td>Avg. Reward</td><td>Norm. Pseudo-Regret</td><td>Violations</td><td>Updates</td></tr><tr><td rowspan="5">500</td><td>Rare-Update Optimistic Planning</td><td> $0 . 7 6 5 5 \pm 0 . 0 1 4 5$ </td><td> $0 . 1 6 4 5 \pm 0 . 0 1 3 4$ </td><td>0.0</td><td>250.2</td></tr><tr><td>Myopic Feasible OFUL</td><td> $0 . 7 7 0 1 \pm 0 . 0 1 3 8$ </td><td> $0 . 1 5 9 9 \pm 0 . 0 1 3 0$ </td><td>0.0</td><td>500.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 1 3 \pm 0 . 0 1 6 7$ </td><td> $0 . 1 5 8 7 \pm 0 . 0 1 6 0$ </td><td>4.7</td><td>500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 7 1 3 \pm 0 . 0 1 6 7$ </td><td> $0 . 1 5 8 7 \pm 0 . 0 1 6 0$ </td><td>4.7</td><td>500.0</td></tr><tr><td>Rare-Update Optimistic Planning</td><td> $0 . 7 7 7 5 \pm 0 . 0 0 4 9$ </td><td> $0 . 1 5 4 9 \pm 0 . 0 0 4 0$ </td><td>0.0</td><td>789.8</td></tr><tr><td rowspan="4"></td><td>Myopic Feasible OFUL</td><td> $0 . 7 7 6 8 \pm 0 . 0 0 4 8$ </td><td> $0 . 1 5 5 6 \pm 0 . 0 0 4 8$ </td><td>0.0</td><td>2500.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 8 0 \pm 0 . 0 0 5 9$ </td><td> $0 . 1 5 4 3 \pm 0 . 0 0 6 0$ </td><td>38.6</td><td>2500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 7 8 2 \pm 0 . 0 0 6 1$ </td><td> $0 . 1 5 4 1 \pm 0 . 0 0 6 2$ </td><td>38.1</td><td>2500.0</td></tr><tr><td>Rare-Update Optimistic Planning</td><td> $0 . 7 7 9 7 \pm 0 . 0 0 3 7$ </td><td> $0 . 1 5 3 1 \pm 0 . 0 0 3 8$ </td><td>0.0</td><td>1104.8</td></tr><tr><td rowspan="4">5000</td><td>Myopic Feasible OFUL</td><td> $0 . 7 7 9 3 \pm 0 . 0 0 4 7$ </td><td> $0 . 1 5 3 5 \pm 0 . 0 0 4 7$ </td><td>0.0</td><td>5000.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 6 9 \pm 0 . 0 0 3 6$ </td><td> $0 . 1 5 5 9 \pm 0 . 0 0 3 6$ </td><td>67.6</td><td>5000.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 7 6 9 \pm 0 . 0 0 3 6$ </td><td> $0 . 1 5 5 9 \pm 0 . 0 0 3 6$ </td><td>67.6</td><td>5000.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

LLM Routing Performance under Cost Constraints. Table 3 reports LLM routing performance under the budget constraint $B = 0 . 4 0$ across diferent horizons T. Since future query contexts are unavailable at decision time, we use the $H \ = \ 1$ specialization of Rare-Update Optimistic Planning. The method maintains zero constraint violations across all horizons and achieves reward and normalized pseudo regret comparable to the other approaches, particularly at the longer horizons. Importantly, it substantially reduces the number of policy updates, requiring only 250.2, 789.8, and 1104.8 updates at $T =$ 500, 2500, and 5000, respectively, compared with one update per round for Myopic Feasible OFUL. Myopic Feasible OFUL also satisfies the constraint exactly, whereas Unconstrained OFUL and Primal-Dual OFUL incur sliding-window cost violations whose frequency increases with the horizon. Additional results across budget levels and horizons are provided in Appendix K.

## 8 CONCLUSION

We study learning under exact sliding-window constraints, where each action afects both current reward and future feasibility. Our results show that the main dificulty can be understood through three structural ideas: geometry, reachability, and planning. Under convex and cyclically invariant constraints, cyclic symmetrization reduces the ofline problem to stationary optimization: stationary behavior is optimal when w | T and within an additive $\mathcal { O } ( w )$ otherwise. In the online setting, geometry alone is not enough, since feasible behaviors may be disconnected. The transition diameter τ captures this reachability cost and enters the regret through the price of feasibilitypreserving transitions. Beyond cyclic symmetry, optimal behavior may be non-stationary. We therefore model the recent $w - 1$ actions as a finite-memory state and introduce the history-state diameter $D ,$ which controls communication and continuation-value variation across viable histories. This leads to exact optimistic remaining-horizon planning with regret $\widetilde { \mathcal { O } } ( d \sqrt { T } + d D + w )$ . Overall, the geometry–reachability– planning viewpoint separates the main sources of difficulty created by exact temporal feasibility and provides a basis for studying broader history-dependent constraint models.

## AI use statement

We used generative AI tools to improve the language and presentation of the manuscript and to review proof drafts for potentially nontrivial or inadvertently omitted steps. We did not use generative AI tools to formulate mathematical claims, develop the theoretical framework, design experiments, implement methods, clean or reformat datasets, or interpret experimental results. All AI-assisted feedback related to the proofs was independently checked and verified by the authors. The authors take full responsibility for the final content of the paper.

## References

Abbasi-Yadkori, Y., P´al, D., and Szepesv´ari, C. (2011). Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24.

Abdullah, M., Iosifidis, G., Elayoubi, S. E., and Chahed, T. (2026). Constrained online convex optimiza-

tion with memory and predictions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 19524–19532.

Agrawal, P. and Tulabandhula, T. (2018). Bandits with temporal stochastic constraints. arXiv preprint arXiv:1811.09026.

Agrawal, S. and Devanur, N. R. (2019). Bandits with global convex constraints and objective. Operations Research, 67(5):1486–1502.

Altman, E. (1999). Constrained Markov Decision Processes. Chapman & Hall/CRC.

Altschuler, J. and Talwar, K. (2018). Online learning over a finite action set with limited switching. In Conference On Learning Theory, pages 1569–1573. PMLR.

Amani, S., Alizadeh, M., and Thrampoulidis, C. (2019). Linear stochastic bandits under safety constraints. Advances in Neural Information Processing Systems, 32.

Amani, S., Thrampoulidis, C., and Yang, L. (2021). Safe reinforcement learning with linear function approximation. In International Conference on Machine Learning, pages 243–253. PMLR.

Ames, A. D., Xu, X., Grizzle, J. W., and Tabuada, P. (2017). Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control, 62(8):3861–3876.

Anava, O., Hazan, E., and Mannor, S. (2015). Online learning for adversaries with memory: price of past mistakes. Advances in Neural Information Processing Systems, 28.

Anthropic (2025). System card:Claude Opus 4 & Claude Sonnet4. https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285/Claude%204%20System%20Card.pdf.

Azar, M. G., Osband, I., and Munos, R. (2017). Minimax regret bounds for reinforcement learning. In International conference on machine learning, pages 263–272. PMLR.

Badanidiyuru, A., Kleinberg, R., and Slivkins, A. (2018). Bandits with knapsacks. Journal of the ACM, 65(3):1–55.

Bernasconi, M., Castiglioni, M., and Celli, A. (2025). No-regret is not enough! Bandits with general constraints through adaptive regret minimization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 3877–3898. PMLR.

Capone, A., Cosner, R. K., Ames, A., and Hirche, S. (2025). Learning safe control via on-the-fly ban-

dit exploration. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 6726–6745. PMLR.

Castiglioni, M., Celli, A., and Kroer, C. (2024). Online learning under budget and ROI constraints via weak adaptivity. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 5792–5816. PMLR.

Chattopadhyay, S., Sukumar, S., and Natarajan, V. (2025). Model reference adaptive control for state and input constrained linear systems. European Journal of Control, 82:101196.

Dani, V., Hayes, T. P., and Kakade, S. M. (2008). Stochastic linear optimization under bandit feedback. In 21st Annual Conference on Learning Theory, pages 355–366.

DeepSeek-AI (2024). DeepSeek-V3 technical report.

Gangrade, A., Chen, T., and Saligrama, V. (2024). Safe linear bandits over unknown polytopes. In The Thirty Seventh Annual Conference on Learning Theory, pages 1755–1795. PMLR.

Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., and He, X. (2022). Kuairand: An unbiased sequential recommendation dataset with randomly exposed videos. In Proceedings of the 31st ACM international conference on information & knowledge management, pages 3953–3957.

Google DeepMind (2025a). Gemini 2.5 Flash model card. https://storage. googleapis.com/deepmind-media/Model-Cards/ Gemini-2-5-Flash-Model-Card.pdf. Accessed: 2026-09-25.

Google DeepMind (2025b). Gemini 2.5 Pro model card. https://storage. googleapis.com/deepmind-media/Model-Cards/ Gemini-2-5-Pro-Model-Card.pdf.

Hebbar, V. and Langbort, C. (2024). Online decision making with history-average dependent costs. In 6th Annual Learning for Dynamics & Control Conference, pages 480–491. PMLR.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S. I., Solar-Lezama, A., Sen, K., and Stoica, I. (2025). LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations.

Jenatton, R., Huang, J., and Archambeau, C. (2016). Adaptive algorithms for online convex optimization with long-term constraints. In International Conference on Machine Learning, pages 402–411. PMLR.

Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. (2020). Provably eficient reinforcement learning with linear function approximation. In Conference on learning theory, pages 2137–2143. PMLR.

Koren, T., Livni, R., and Mansour, Y. (2017). Bandits with movement costs and adaptive pricing. In Conference on Learning Theory, pages 1242–1268. PMLR.

Lattimore, T. and Szepesv´ari, C. (2020). Bandit Algorithms. Cambridge University Press.

Li, H., Zhang, Y., Guo, Z., Wang, C., Tang, S., Zhang, Q., Chen, Y., Qi, B., Ye, P., Bai, L., Wang, Z., and Hu, S. (2026). LLMRouterBench: A massive benchmark and unified framework for LLM routing. In Findings of the Association for Computational Linguistics: ACL 2026, pages 37733–37754.

Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., and Stoica, I. (2025). From crowdsourced data to high-quality benchmarks: Arena-Hard and Benchbuilder pipeline. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 34209–34231. PMLR.

Liakopoulos, N., Destounis, A., Paschos, G., Spyropoulos, T., and Mertikopoulos, P. (2019). Cautious regret minimization: Online optimization with long-term budget constraints. In International Conference on Machine Learning, pages 3944–3952. PMLR.

Liu, X., Li, B., Shi, P., and Ying, L. (2021). An efi cient pessimistic-optimistic algorithm for stochastic linear bandits with general constraints. Advances in Neural Information Processing Systems, 34:24075– 24086.

Mahdavi, M., Jin, R., and Yang, T. (2012). Trading regret for eficiency: online convex optimization with long term constraints. The Journal of Machine Learning Research, 13:2503–2528.

Mannor, S., Tsitsiklis, J. N., and Yu, J. Y. (2009). Online learning with sample path constraints. Journal of Machine Learning Research, 10(20):569–590.

Mathias, J., Angluri, R., Kosut, O., and Sankar, L. (2024). Model predictive control for joint ramping and regulation-type service from distributed energy resource aggregations. In 2024 IEEE Power & Energy Society General Meeting (PESGM), pages 1–5. IEEE.

OpenAI (2025). GPT-5 system card. https:// openai.com/index/gpt-5-system-card/.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 conference

on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 3982–3992.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. (2024). GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Rouyer, C., Seldin, Y., and Cesa-Bianchi, N. (2021). An algorithm for stochastic and adversarial bandits with switching costs. In International Conference on Machine Learning, pages 9127–9135. PMLR.

Stradi, F. E., Germano, J., Genalti, G., Castiglioni, M., Marchesi, A., and Gatti, N. (2024). Online learning in CMDPs: Handling stochastic and adversarial constraints. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 46692–46721. PMLR.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. (2024). MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yi, X., Li, X., Yang, T., Xie, L., Chai, T., and Johansson, K. (2021). Regret and cumulative constraint violation analysis for online convex optimization with long term constraints. In International conference on machine learning, pages 11998–12008. PMLR.

Zhang, X. and Cheung, W. C. (2026). Online allocation of reusable resources in nonstationary environments. Mathematics of Operations Research, 51(3):1853–1883.

# Supplementary Materials

## A LIMITATIONS AND EXTENSIONS

Our results have several limitations. In the cyclic setting, the stationary reduction relies on convexity and cyclic-shift invariance, and may fail for discrete or nonconvex action spaces. In the general non-cyclic setting, our regret guarantee assumes a finite history-state diameter and access to exact remaining-horizon planning. For finite action spaces, the history-state space can grow as $| \mathcal { X } | ^ { w - 1 }$ , making exact planning computationally expensive for large w.

These limitations suggest several directions for future work. Although the stationary reduction may fail without convexity, the finite-memory formulation remains applicable, motivating the study of periodic or recurrent feasible policies for discrete and nonconvex action spaces. Developing scalable planning methods that exploit the structure of feasible transitions, rather than enumerating the full history-state space, is another important direction. A further open question concerns the dependence on the transition diameter. Our rare-switching regret bound contains a transition term of order $\tilde { \mathcal { O } } ( \tau d )$ , whereas Proposition H.7 establishes only an $\Omega ( \tau )$ lower bound. It remains open whether the additional factor of d is intrinsic or can be removed through a sharper algorithm or analysis.

## B COMPUTATIONAL ORACLES

Our regret guarantees separate statistical learning from the computational tasks required to implement the algorithms. In the cyclic setting, each episode requires two operations: selecting an optimistic stationary action and constructing a feasible transition between stationary histories. The general non-cyclic setting additionally requires exact remaining-horizon planning. We formalize these operations below.

Optimistic-action oracle. Given an estimate $\widehat { \theta } \in \mathbb { R } ^ { d }$ , a positive-definite matrix $V \succ 0 ,$ , and a confidence radius $\beta \geq 0$ , the optimistic-action oracle returns

$$
\mathrm { O p t } ( \widehat \theta , V , \beta ) \in \arg \operatorname* { m a x } _ { z \in \mathcal { Z } } \left\{ \langle \widehat \theta , z \rangle + \beta \| z \| _ { V ^ { - 1 } } \right\} .
$$

Since $\mathcal { Z }$ is compact and the objective is continuous, a maximizer exists. We assume that the oracle returns an exact maximizer. The objective

$$
z \mapsto \langle \widehat { \theta } , z \rangle + \beta \Vert z \Vert _ { V ^ { - 1 } }
$$

is convex in z. Thus, even when $\mathcal { Z }$ is convex, optimistic action selection is generally not a convex optimization problem because a convex function is being maximized. Eficient implementation therefore requires additiona structure in $\mathcal { Z } .$

Transition oracle. For $w = 1$ , no history transition is required, and we take $\tau = 0$ . For $w \geq 2$ and $z , z ^ { \prime } \in { \mathcal { Z } }$ the transition oracle returns

$$
\Gamma ( z , z ^ { \prime } ) = ( a _ { 1 } , \ldots , a _ { m } ) , \qquad m \leq \tau ,
$$

such that, starting from $h _ { 0 } = h ( z )$ , the histories

$$
h _ { j } = F ( h _ { j - 1 } , a _ { j } ) , \qquad j = 1 , \ldots , m ,
$$

satisfy

$$
( h _ { j - 1 } , a _ { j } ) \in \mathcal { C } , \qquad j = 1 , \dotsc , m ,
$$

and terminate at $h _ { m } = h ( z ^ { \prime } )$ . Hence every window encountered during the transition is feasible, and the target stationary history is reached in at most τ rounds. When $z = z ^ { \prime } .$ , the oracle may return the empty transition.

The regret guarantee in Section 5 assumes exact access to both Opt and Γ. This isolates the statistical dificulty of learning the reward from the computational dificulty of optimizing over and navigating the feasible set.

Remaining-horizon planning oracle. In the general non-cyclic setting of Section 6, the learner must instead optimize over viable history states. Given a viable history $s \in S _ { n }$ and an optimistic reward function $U : \mathcal { X }  \mathbb { R }$ the planning oracle computes an exact Bellman-optimal n-step continuation, with value

$$
J _ { 0 } ^ { U } ( s ) : = 0 ,
$$

and, for $n \geq 1$ ，

$$
J _ { n } ^ { U } ( s ) = \operatorname * { m a x } _ { x \in A ( s ) \atop F ( s , x ) \in S _ { n - 1 } } \left\{ U ( x ) + J _ { n - 1 } ^ { U } ( F ( s , x ) ) \right\} .
$$

The regret guarantee for the non-cyclic setting is conditional on exact access to this planner. For finite action spaces, straightforward backward dynamic programming over viable histories takes $\mathcal { O } ( T | \mathcal { X } | ^ { w } )$ time in the direct history representation, up to viability preprocessing. For general continuous constraints, eficient planning requires additional problem structure.

## C RELATION TO LINEAR MDPS AND CONSTRAINED REINFORCEMENT LEARNING

In this section, we clarify the connection between our formulation and existing reinforcement-learning models. We first show that a direct reduction to linear MDPs (Jin et al., 2020) can lead to much weaker dependence on the window length w. The reason is that the linear-MDP framework must encode the history-dependent transition dynamics through a linear feature representation, whose dimension can grow exponentially with w. In contrast, our setting has known deterministic history dynamics and only d-dimensional reward uncertainty, allowing our analysis to avoid this exponential dependence. We then compare our formulation with tabular and constrained reinforcement learning, highlighting why generic state-augmentation approaches can introduce exponential dependence on the window length and why existing constrained-RL guarantees do not directly capture our exact sliding-window setting.

Relation to linear MDPs. Jin et al. (2020) consider a p-dimensional linear MDP with a known feature map $\phi : S \times \mathcal { A }  \mathbb { R } ^ { p }$ such that both the expected reward and the transition kernel are linear in the same features:

$$
r _ { h } ( s , a ) = \langle \phi ( s , a ) , \theta _ { h } \rangle , \qquad P _ { h } ( \cdot \mid s , a ) = \langle \phi ( s , a ) , \mu _ { h } ( \cdot ) \rangle .
$$

Thus, the feature dimension p must be rich enough to represent both rewards and transitions. Unconstrained linear bandits (Dani et al., 2008; Abbasi-Yadkori et al., 2011) arise as the special case $H = 1$ . Our slidingwindow problem has a diferent structure. In Section 6, the state is the recent action history $s = ( x _ { 1 } , \dots , x _ { w - 1 } )$ and choosing action x deterministically updates the state according to $F ( s , x ) = ( x _ { 2 } , \ldots , x _ { w - 1 } , x )$ . Hence the transition kernel is known and deterministic: $P ( \cdot \mid s , x ) = \delta _ { F ( s , x ) } ( \cdot )$ . In contrast, the unknown reward remains d-dimensional:

$$
r ^ { \star } ( s , x ) = \langle \theta ^ { \star } , x \rangle , \qquad \theta ^ { \star } \in \mathbb { R } ^ { d } ,
$$

where the same parameter $\theta ^ { \star }$ is shared across all history states. Thus, the statistical uncertainty in our problem is only d-dimensional, even though the history-state representation can be much larger.

The following proposition shows that representing the natural history-shift dynamics as a linear MDP can require a feature dimension exponential in the window length.

Proposition C.1. Let $w \geq 2$ and let $\mathcal { X } = \{ a _ { 1 } , \ldots , a _ { m } \} \subset \mathbb { R } ^ { d } .$ , with $m \geq 2$ . Consider the history-state space ${ \mathcal { S } } = { \mathcal { X } } ^ { w - 1 }$ with deterministic transition $F ( s , x ) = ( x _ { 2 } , \ldots , x _ { w - 1 } , x )$ . Any exact linear-MDP representation of this transition kernel on the full history-state space requires feature dimension $p \ge m ^ { w - 1 }$ . In particular, the required feature dimension is exponential in w for fixed m $\geq 2$

Proof. Let P be the transition matrix with rows indexed by $( s , x ) \in \mathcal { S } \times \mathcal { X }$ and columns indexed by $s ^ { \prime } \in { \mathcal { S } } ,$ where

$$
P _ { ( s , x ) , s ^ { \prime } } = \mathbf { 1 } \{ F ( s , x ) = s ^ { \prime } \} .
$$

Since $| S | = m ^ { w - 1 }$ , the matrix $P$ has $m ^ { w - 1 }$ columns. Fix any $s ^ { \prime } = ( b _ { 1 } , \ldots , b _ { w - 1 } ) \in \mathcal { S }$ . For any $a \in { \mathcal { X } }$ , choose $s = ( a , b _ { 1 } , \dots , b _ { w - 2 } )$ and $x = b _ { w - 1 }$ . Then $F ( s , x ) = s ^ { \prime } { . }$ , so the corresponding row of $P$ is the standard basis vector associated with $s ^ { \prime } .$ Hence the rows of $P$ contain all $m ^ { w - 1 }$ standard basis vectors, which are linearly independent. Therefore, rank $( P ) = m ^ { w - 1 }$ . If the transition kernel admits an exact p-dimensional linear-MDP representation, then in the finite-state setting $P = \Phi M ,$ where $\Phi$ has $p$ columns. Thus,

$$
m ^ { w - 1 } = \mathrm { r a n k } ( P ) \leq \mathrm { r a n k } ( \Phi ) \leq p ,
$$

which proves $p \geq m ^ { w - 1 }$

Proposition C.1 clarifies the distinction between the reward dimension d and the feature dimension p required by a generic linear-MDP representation. Representing the known history-shift dynamics in the standard linear-MDP form may require $p \geq m ^ { w - 1 }$ , which is exponential in w. Thus, although the history-augmented problem is an MDP, a low-dimensional linear-MDP representation need not preserve the structure that is useful in our setting. Our analysis instead learns only the d-dimensional reward parameter and handles the known history dynamics through planning and the history-state diameter D. This yields the regret bound $\widetilde { \mathcal { O } } ( d \sqrt { T } + d D + w )$ whose statistical dependence is on d rather than on the potentially exponential representation dimension p.

Comparison with a tabular reduction. One could instead treat the history-augmented process as a generic tabular MDP. For a finite action space with $| { \mathcal { X } } | = m ,$ , the full history representation has $| S | = m ^ { w - 1 }$ and $| { \mathcal { A } } | = m$ , and therefore $| S | | { \mathcal { A } } | = m ^ { w }$ . Thus, the state-action space grows exponentially with the window length w. This growth can also appear in generic tabular RL regret guarantees. For example, Azar et al. (2017) establish finite-horizon tabular RL bounds with leading dependence of the form $\tilde { \mathcal { O } } \big ( \sqrt { H | \boldsymbol { S } | | \boldsymbol { A } | N } \big )$ , where H is the episodic horizon and N is the total number of interaction steps. Applied to the full history-state representation, this becomes $\widetilde { \mathcal { O } } \Big ( \sqrt { H m ^ { w } N } \Big ) = \widetilde { \mathcal { O } } \Big ( m ^ { w / 2 } \sqrt { H N } \Big )$ , which has exponential dependence on w.

Our analysis avoids this state-space dependence by exploiting the structure of the problem: the history dynamics are known and deterministic, while the reward uncertainty is governed by a single d-dimensional parameter shared across all histories. Rather than depending on the number of history states, our regret bound depends on their communication structure through the history-state diameter D, yielding $\widetilde { \mathcal { O } } ( d \sqrt { T } + d D + w )$ . For many constraint classes considered in this work, $D = \mathcal { O } ( w )$ (see Appendix F), so the bound simplifies to $\widetilde { \mathcal { O } } ( d \sqrt { T } + d w )$ . Thus, for these classes, the dependence on temporal memory is linear in w, up to logarithmic and dimension factors, even though the full history-state space is exponential in $w .$ This highlights the advantage of exploiting the low-dimensional reward structure and known finite-memory dynamics rather than treating the problem as a generic tabular MDP.

Relation to constrained and safe reinforcement learning. Constrained Markov decision processes (CMDPs) (Altman, 1999) optimize expected cumulative reward subject to constraints on cumulative or expected costs, while safe reinforcement learning (Amani et al., 2021) can impose stronger pathwise requirements that restrict the actions or states that may be visited during the trajectory. Our setting is closer to the latter notion: exact sliding-window feasibility requires every realized overlapping window to satisfy $( x _ { t - w + 1 } , \dots , x _ { t } ) \in { \mathcal { C } }$ throughout the trajectory, rather than allowing violations that are compensated for later.

After augmenting the state with the previous $w - 1$ actions, the feasible-action set $\mathcal { A } ( s ) = \{ x : ( s , x ) \in \mathcal { C } \}$ can be interpreted as a state-dependent hard safety constraint. Thus, our problem can be viewed abstractly as a safe MDP. The key distinction lies in the structure of the problem: the history dynamics and constraint set are known, feasibility must hold exactly at every round, and all reward uncertainty is governed by a single d-dimensional parameter shared across history states. Generic safe-RL formulations do not typically exploit this combination of known finite-memory dynamics, exact pathwise feasibility, and low-dimensional shared reward uncertainty.

Moreover, applying a generic RL formulation after history augmentation can obscure this structure. As shown above, an exact linear-MDP representation of the full history dynamics may require feature dimension $p \geq m ^ { w - 1 }$ while a tabular representation contains $| \mathcal { X } | ^ { w - 1 }$ history states. To the best of our knowledge, existing constrainedand safe-RL guarantees do not directly yield this dependence for exact overlapping-window feasibility with lowdimensional reward uncertainty and known deterministic finite-memory dynamics.

## D EXAMPLES OF CONVEX CYCLICALLY INVARIANT CONSTRAINTS

The following examples illustrate useful classes of convex cyclically invariant window-feasibility sets. Throughout this appendix, for a window $x = ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal { X } ^ { w }$ , we use $x _ { i }$ to denote its ith component.

Aggregate budget constraints. A cumulative budget constraint is given by

$$
\mathcal { C } = \left\{ x \in \mathcal { X } ^ { w } : \sum _ { i = 1 } ^ { w } c ^ { \top } x _ { i } \leq B \right\} .
$$

Maximum constraints. Worst-case constraints within a window can be expressed as

$$
\mathcal { C } = \left\{ x \in \mathcal { X } ^ { w } : \operatorname* { m a x } _ { 1 \leq i \leq w } \phi ( x _ { i } ) \leq \rho \right\} ,
$$

where $\phi$ is convex. A common example is

$$
\operatorname* { m a x } _ { 1 \leq i \leq w } \| A x _ { i } - b \| \leq \rho ,
$$

which limits the maximum deviation of any action in the window.

Box constraints. Given $\ell , u \in \mathbb { R } ^ { d }$ with $\ell \leq u$

$$
\mathcal { C } = \left\{ x \in \mathcal { X } ^ { w } : \ell \leq x _ { i } \leq u , \quad i = 1 , \ldots , w \right\} .
$$

Ramping constraints. Using $x _ { w + 1 } = x _ { 1 }$ , cyclic variation can be controlled by

$$
\mathcal { C } = \left\{ x \in \mathcal { X } ^ { w } : \operatorname* { m a x } _ { 1 \leq i \leq w } \| x _ { i + 1 } - x _ { i } \| \leq \rho \right\} .
$$

Exponential constraints. Convex exponential penalties yield

$$
\mathcal { C } = \left\{ x \in \mathcal { X } ^ { w } : \sum _ { i = 1 } ^ { w } \exp ( a ^ { \top } x _ { i } - b ) \leq \rho \right\} ,
$$

which can represent cumulative risk, energy consumption, or congestion.

$\ell _ { p } .$ norm constraints. For $p \geq 1$

$$
\mathcal { C } = \left\{ x \in \mathcal { X } ^ { w } : \left( \sum _ { i = 1 } ^ { w } \| A x _ { i } - b \| _ { p } ^ { p } \right) ^ { 1 / p } \leq \rho \right\}
$$

captures aggregate deviation and robustness requirements.

Convex risk constraints. For a convex loss function $\ell ,$

$$
\mathcal { C } = \left\{ x \in \mathcal { X } ^ { w } : \frac { 1 } { w } \sum _ { i = 1 } ^ { w } \ell ( x _ { i } ) \leq \rho \right\}
$$

provides a general risk-based window constraint.

## E TRANSITION DIAMETER FOR COMMON SLIDING-WINDOWCONSTRAINTS

We verify Assumption 5.3 for the constraint classes introduced in Appendix D. Table 4 summarizes the resulting transition-diameter bounds. Throughout, assume that the corresponding stationary feasible set Z is nonempty. We consider $w \geq 2 ;$ when $w = 1$ , no transition history is required and one may take $\tau = 0$ . For $z , z ^ { \prime } \in \mathcal { X }$ and $j \in \{ 0 , \ldots , w \}$ , define the mixed window

$$
M _ { j } ( z , z ^ { \prime } ) : = ( \underbrace { z , \dotsc , z } _ { w - j } , \underbrace { z ^ { \prime } , \dotsc , z ^ { \prime } } _ { j } ) .
$$

Lemma E.1. Suppose that, for every $z , z ^ { \prime } \in { \mathcal { Z } }$

$$
M _ { j } ( z , z ^ { \prime } ) \in \mathcal { C } , \qquad j = 1 , \dotsc , w - 1 .
$$

Then Assumption 5.3 holds with $\tau \leq w - 1$

Proof. Fix $z , z ^ { \prime } \in { \mathcal { Z } }$ and select

$$
a _ { j } = z ^ { \prime } , \qquad j = 1 , \dots , w - 1 .
$$

Starting from $h ( z )$ , the complete window generated by the j-th transition action is $M _ { j } ( z , z ^ { \prime } )$ , which is feasible by assumption. After $w - 1$ actions, all entries of the original history have been overwritten, and the resulting history is $h _ { w - 1 } = h ( z ^ { \prime } )$ . Thus the transition is feasible and has length $w - 1$ □

Proposition E.2 (Aggregate budget constraints). Let $c \in \mathbb { R } ^ { d } , B \in \mathbb { R }$ , and define

$$
\mathcal C _ { \mathrm { b u d g e t } } : = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal X ^ { w } : \sum _ { i = 1 } ^ { w } c ^ { \top } x _ { i } \leq B \right\} .
$$

Then $\mathcal { C } _ { \mathrm { b u d g e t } }$ is convex and cyclically invariant, and its transition diameter satisfies $\tau \leq w - 1$

Proof. The stationary feasible set is

$$
\mathcal { Z } = \left\{ z \in \mathcal { X } : c ^ { \top } z \leq \frac { B } { w } \right\} .
$$

For $z , z ^ { \prime } \in { \mathcal { Z } }$ and $j = 1 , \dotsc , w - 1$ 2

$$
\sum _ { i = 1 } ^ { w } c ^ { \top } M _ { j } ( z , z ^ { \prime } ) _ { i } = ( w - j ) c ^ { \top } z + j c ^ { \top } z ^ { \prime } \leq ( w - j ) \frac { B } { w } + j \frac { B } { w } = B .
$$

Thus every mixed window is feasible, and Lemma E.1 applies.

Proposition E.3 (Maximum constraints). Let $\phi : \mathcal { X }  \mathbb { R }$ be convex and define

$$
{ \mathcal { C } } _ { \operatorname* { m a x } } : = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in { \mathcal { X } } ^ { w } : \operatorname* { m a x } _ { 1 \leq i \leq w } \phi ( x _ { i } ) \leq \rho \right\} .
$$

Then $\mathcal { C } _ { \mathrm { m a x } }$ is convex and cyclically invariant, and its transition diameter satisfies $\tau \leq w - 1$

Proof. The stationary feasible set is

$$
\mathcal Z = \{ z \in \mathcal X : \phi ( z ) \leq \rho \} .
$$

For $z , z ^ { \prime } \in { \mathcal { Z } } ,$

$$
\operatorname* { m a x } _ { 1 \leq i \leq w } \phi ( M _ { j } ( z , z ^ { \prime } ) _ { i } ) = \operatorname* { m a x } \{ \phi ( z ) , \phi ( z ^ { \prime } ) \} \leq \rho .
$$

Therefore every mixed window is feasible, and Lemma E.1 applies.

Proposition E.4 (Box constraints). Let $\ell , u \in \mathbb { R } ^ { d }$ satisfy $\ell \leq u$ , and define

$$
\mathcal { C } _ { \mathrm { b o x } } : = \left\{ \left( x _ { 1 } , \ldots , x _ { w } \right) \in \mathcal { X } ^ { w } : \ell \leq x _ { i } \leq u , \quad i = 1 , \ldots , w \right\} .
$$

Then $\mathcal { C } _ { \mathrm { b o x } }$ is convex and cyclically invariant, and its transition diameter satisfies $\tau \leq w - 1$

Proof. If $z , z ^ { \prime } \in { \mathcal { Z } }$ , then

$$
\ell \leq z \leq u , \qquad \ell \leq z ^ { \prime } \leq u .
$$

Every entry of $M _ { j } ( z , z ^ { \prime } )$ is either z or $z ^ { \prime } .$ , and therefore satisfies the box constraint. Hence every mixed window is feasible, and Lemma E.1 applies. □

Proposition E.5 (Exponential constraints). Let $a \in \mathbb { R } ^ { d } , b \in \mathbb { R }$ , and $\rho > 0$ . Define

$$
\mathcal C _ { \mathrm { e x p } } : = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal X ^ { w } : \sum _ { i = 1 } ^ { w } \exp ( a ^ { \top } x _ { i } - b ) \leq \rho \right\} .
$$

Then $\mathcal { C } _ { \mathrm { e x p } }$ is convex and cyclically invariant, and its transition diameter satisfies $\tau \leq w - 1$

Proof. Convexity follows because $x \mapsto \exp ( a ^ { \top } x - b )$ is convex, while cyclic invariance follows from invariance of the sum under permutations. The stationary feasible set is $\mathcal { Z } = \left\{ z \in \mathcal { X } : w \exp ( a ^ { \top } z - b ) \leq \rho \right\}$ . For $z , z ^ { \prime } \in { \mathcal { Z } }$

$$
\sum _ { i = 1 } ^ { w } \exp \left( a ^ { \top } M _ { j } ( z , z ^ { \prime } ) _ { i } - b \right) = \left( w - j \right) \exp ( a ^ { \top } z - b ) + j \exp ( a ^ { \top } z ^ { \prime } - b ) \leq ( w - j ) \frac { \rho } { w } + j \frac { \rho } { w } = \rho .
$$

Thus every mixed window is feasible, and Lemma E.1 applies.

Proposition E.6 $( \ell _ { p } { \mathrm { - n o r m } }$ constraints). Let $A \in \mathbb { R } ^ { r \times d } , \ b \in \mathbb { R } ^ { r }$ , and $p \in [ 1 , \infty )$ . Define

$$
\mathcal C _ { p } : = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal X ^ { w } : \left( \sum _ { i = 1 } ^ { w } \| A x _ { i } - b \| _ { p } ^ { p } \right) ^ { 1 / p } \leq \rho \right\} .
$$

Then $\mathcal { C } _ { p }$ is convex and cyclically invariant, and its transition diameter satisfies $\tau \leq w - 1$

Proof. The left-hand side is the $\ell _ { p } .$ -norm of the stacked afine residual vector $\big ( ( A x _ { 1 } - b ) ^ { \top } , \dots , ( A x _ { w } - b ) ^ { \top } \big ) ^ { \top }$ , and is therefore convex. It is unchanged by cyclic permutations of the actions. For a stationary action $z ,$ feasibility is equivalent to $w ^ { 1 / p } \| A z - b \| _ { p } \leq \rho$ . Hence $\begin{array} { r } { \mathcal { Z } = \left\{ z \in \mathcal { X } : \| A z - b \| _ { p } \leq \frac { \rho } { w ^ { 1 / p } } \right\} } \end{array}$ . For $z , z ^ { \prime } \in { \mathcal { Z } }$

$$
\sum _ { i = 1 } ^ { w } \| A M _ { j } ( z , z ^ { \prime } ) _ { i } - b \| _ { p } ^ { p } = ( w - j ) \| A z - b \| _ { p } ^ { p } + j \| A z ^ { \prime } - b \| _ { p } ^ { p } \leq ( w - j ) \frac { \rho ^ { p } } { w } + j \frac { \rho ^ { p } } { w } = \rho ^ { p } .
$$

Taking the p-th root shows that $M _ { j } ( z , z ^ { \prime } ) \in \mathcal { C } _ { p }$ . The result follows from Lemma E.1.

Remark E.7. The case $p = \infty$ becomes max $1 \leq i \leq w \| A x _ { i } - b \| _ { \infty } \leq \rho$ and is covered by Proposition E.3.

Proposition E.8 (Convex risk constraints). Let $\ell : \mathcal { X } \to \mathbb { R }$ be convex and define

$$
\mathcal { C } _ { \mathrm { r i s k } } : = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal { X } ^ { w } : \frac { 1 } { w } \sum _ { i = 1 } ^ { w } \ell ( x _ { i } ) \leq \rho \right\} .
$$

Then $\mathcal { C } _ { \mathrm { r i s k } }$ is convex and cyclically invariant, and its transition diameter satisfies $\tau \leq w - 1$

Proof. The stationary feasible set is $\mathcal { Z } = \left\{ z \in \mathcal { X } : \ell ( z ) \le \rho \right\}$ . For $z , z ^ { \prime } \in { \mathcal { Z } }$

$$
\frac { 1 } { w } \sum _ { i = 1 } ^ { w } \ell ( M _ { j } ( z , z ^ { \prime } ) _ { i } ) = \frac { w - j } { w } \ell ( z ) + \frac { j } { w } \ell ( z ^ { \prime } ) \le \rho .
$$

Thus every mixed window is feasible, and Lemma E.1 applies.

Proposition E.9 (Cyclic ramping constraints). Suppose that $\mathcal { X }$ is compact and convex and let $D _ { \mathcal { X } } : =$ $\begin{array} { r } { \operatorname* { s u p } _ { x , y \in \mathcal { X } } \| x - y \| < \infty } \end{array}$ . For $\rho > 0$ , define

$$
{ \mathcal { C } } _ { \mathrm { r a m p } } : = \left\{ \left( x _ { 1 } , \ldots , x _ { w } \right) \in { \mathcal { X } } ^ { w } : \operatorname* { m a x } _ { 1 \leq i \leq w } \left\| x _ { i + 1 } - x _ { i } \right\| \leq \rho \right\} , \qquad x _ { w + 1 } = x _ { 1 } .
$$

Then $\mathcal { C } _ { \mathrm { r a m p } }$ is convex and cyclically invariant, and Assumption 5.3 holds with

$$
\tau \leq \left\lceil { \frac { ( w - 1 ) D _ { \mathcal { X } } } { \rho } } \right\rceil + w - 2 .
$$

Proof. The function $\begin{array} { r } { ( x _ { 1 } , \dots , x _ { w } ) \mapsto \operatorname* { m a x } _ { 1 \leq i \leq w } \left\| x _ { i + 1 } - x _ { i } \right\| } \end{array}$ is convex and cyclically invariant. Moreover, every stationary window has zero cyclic variation, so $\mathcal { Z } = \mathcal { X }$ . Fix distinct $z , z ^ { \prime } \in { \mathcal { Z } }$ and set

$$
N : = \left\lceil { \frac { ( w - 1 ) \| z ^ { \prime } - z \| } { \rho } } \right\rceil .
$$

Define

$$
y _ { r } : = z + \frac { r } { N } ( z ^ { \prime } - z ) , \qquad r = 0 , \ldots , N .
$$

Convexity of X implies $y _ { r } \in \mathcal X$ , and

$$
\| y _ { r } - y _ { r - 1 } \| = \frac { \| z ^ { \prime } - z \| } { N } \leq \frac { \rho } { w - 1 } .
$$

Starting from $h ( z )$ , play $y _ { 1 } , \ldots , y _ { N }$ and then play $z ^ { \prime } = y _ { N }$ for an additional $w - 2$ rounds. In every complete window, consecutive actions difer by at most $\rho / ( w - 1 )$ . The cyclic diference between the first and last actions in the window is therefore at most

$$
( w - 1 ) \frac { \rho } { w - 1 } = \rho .
$$

Hence every generated window belongs to $\mathcal { C } _ { \mathrm { r a m p } }$ . After $N + w - 2$ actions, the history equals $h ( z ^ { \prime } )$ . Finally,

$$
N + w - 2 \leq \left\lceil { \frac { ( w - 1 ) D \chi } { \rho } } \right\rceil + w - 2 .
$$

When $z = z ^ { \prime } ,$ the empty transition is feasible.

Remark E.10. Suppose that $\mathcal { Z }$ contains distinct actions $z \neq z ^ { \prime }$ . Any transition from $h ( z )$ to $h ( z ^ { \prime } )$ requires at least $w - 1$ actions because each new action replaces at most one entry of the current history. Therefore, for the aggregate-budget, maximum, box, exponential, $\ell _ { p } { \mathrm { - n o r m } }$ , and convex-risk classes, $\tau = w - 1$ . The ramping class may require more than $w - 1$ actions because transitions must proceed gradually.

## F HISTORY DIAMETER FOR COMMON CONSTRAINT CLASSES

The history diameter can be computed or bounded explicitly for several common sliding-window constraints.

State-independent constraints. Suppose ${ \mathcal { C } } = A ^ { w }$ for some nonempty $\mathcal A \subseteq \mathcal X$

Proposition F.1 (State-Independent Constraints). If A contains at least two distinct actions, then $D = w - 1$ $I f A$ is a singleton, then $D = 0$

Proof. For every $n \geq 1 , S _ { n } = \mathcal { A } ^ { w - 1 }$ . Fix

$$
s = ( x _ { 1 } , \ldots , x _ { w - 1 } ) , \qquad s ^ { \prime } = ( y _ { 1 } , \ldots , y _ { w - 1 } )
$$

in $S _ { n }$ . Starting from $s ,$ append $y _ { 1 } , \ldots , y _ { w - 1 }$ . Every resulting window lies in $A ^ { w }$ , and after $w - 1$ actions the history equals $s ^ { \prime } .$ . Hence $d c ( s , s ^ { \prime } ) \leq w - 1$ , so $D \leq w - 1$ . For the reverse inequality, choose distinct $a , b \in { \mathcal { A } }$ and consider

$$
s = ( a , \ldots , a ) , \qquad s ^ { \prime } = ( b , \ldots , b ) .
$$

Each appended action replaces only one coordinate of the current history, so at least $w - 1$ actions are required to transform s into $s ^ { \prime } .$ Therefore, $d c ( s , s ^ { \prime } ) \geq w - 1$ . Thus $D = w - 1$ . The singleton case is immediate. □

Table 4: Transition-diameter for convex and cyclically invariant sliding-window constraints.
<table><tr><td>Constraint class</td><td>Window condition</td><td colspan="2">Transition diameter</td></tr><tr><td>Aggregate budget</td><td> $\sum _ { i = 1 } ^ { w } c ^ { \top } x _ { i } \leq B$ </td><td colspan="2"> $\tau \leq w - 1$ </td></tr><tr><td>Maximum</td><td> $\operatorname* { m a x } _ { 1 \leq i \leq w } \phi ( x _ { i } ) \leq \rho$ </td><td colspan="2"> $\tau \leq w - 1$ </td></tr><tr><td>Box</td><td> $\ell \leq x _ { i } \leq u , \quad i = 1 , \ldots , w$ </td><td colspan="2"> $\tau \leq w - 1$ </td></tr><tr><td>Exponential</td><td> $\sum _ { i = 1 } ^ { w } \exp ( { a ^ { \top } x _ { i } } - b ) \leq \rho$ </td><td colspan="2"> $\tau \leq w - 1$ </td></tr><tr><td>lp-norm</td><td> $\left( \sum _ { i = 1 } ^ { w } \| A x _ { i } - b \| _ { p } ^ { p } \right) ^ { 1 / p } \leq \rho$ </td><td colspan="2"> $\tau \leq w - 1$ </td></tr><tr><td>Convex risk</td><td> ${ \frac { 1 } { w } } \sum _ { i = 1 } ^ { w } \ell ( x _ { i } ) \leq \rho$ </td><td colspan="2"> $\tau \leq w - 1$ </td></tr><tr><td>Cyclic ramping</td><td> $\operatorname* { m a x } _ { i } \| x _ { i + 1 } - x _ { i } \| \leq \rho , \quad x _ { w + 1 } = x _ { 1 }$ </td><td> $\tau \leq \left\lceil \frac { ( w - 1 ) D _ { \mathscr { X } } } { \rho } \right\rceil + w - 2$ </td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td></tr></table>

Note. For all classes except cyclic ramping, the transition is obtained by playing $z ^ { \prime }$ for $w - 1$ rounds. If the corresponding stationary feasible set contains at least two distinct actions, these bounds are sharp: $\tau = w - 1$

Box constraints. Consider

$$
\mathcal { C } _ { \mathrm { b o x } } = \left\{ \left( x _ { 1 } , \ldots , x _ { w } \right) \in \mathcal { X } ^ { w } : \ell \leq x _ { i } \leq u , \quad i = 1 , \ldots , w \right\} ,
$$

and define

$$
\mathcal { A } _ { \mathrm { b o x } } : = \{ x \in \mathcal { X } : \ell \leq x \leq u \} .
$$

Proposition F.2 (Box Constraints). $I f { \mathcal { A } } _ { \mathrm { b o x } }$ contains at least two distinct actions, then $D = w - 1 . ~ I f \mathcal { A } _ { \mathrm { b o x } }$ is a singleton, then $D = 0$

Proof. The constraint is state-independent: $\mathcal { C } _ { \mathrm { b o x } } = \mathcal { A } _ { \mathrm { b o x } } ^ { w }$ . The result follows directly from Proposition F.1.

Maximum constraints. Consider

$$
{ \mathcal { C } } _ { \operatorname* { m a x } } = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal { X } ^ { w } : \operatorname* { m a x } _ { 1 \leq i \leq w } \phi ( x _ { i } ) \leq \rho \right\} ,
$$

and define

$$
\mathcal A _ { \operatorname* { m a x } } : = \{ x \in \mathcal X : \phi ( x ) \leq \rho \} .
$$

Proposition F.3 (Maximum Constraints). $I f \mathcal { A } _ { \mathrm { m a x } }$ contains at least two distinct actions, then $D = w - 1$ . If $\mathcal { A } _ { \mathrm { m a x } }$ is a singleton, then $D = 0$

Proof. The maximum constraint holds if and only if every action belongs to $\mathcal { A } _ { \mathrm { m a x } }$ . Hence $\mathcal { C } _ { \mathrm { m a x } } = \mathcal { A } _ { \mathrm { m a x } } ^ { w }$ , and Proposition F.1 gives the result. □

A general reset principle for additive constraints. Consider a separable additive constraint

$$
\mathcal C _ { g } = \left\{ ( x _ { 1 } , \hdots , x _ { w } ) \in \mathcal X ^ { w } : \sum _ { i = 1 } ^ { w } g ( x _ { i } ) \leq B \right\} ,
$$

where $g : \mathcal { X }  \mathbb { R } .$ . Suppose g attains its minimum over X, and let

$$
x _ { 0 } \in \arg \operatorname* { m i n } _ { x \in \mathcal { X } } g ( x ) , \qquad g _ { 0 } : = g ( x _ { 0 } ) .
$$

Proposition F.4 (Reset-Based History-Diameter Bound). Suppose $\mathcal { C } _ { g } \neq \emptyset$ . Then $D \leq 2 ( w - 1 )$

Proof. Fix $n \geq 1$ and

$$
s = ( x _ { 1 } , \ldots , x _ { w - 1 } ) , \qquad s ^ { \prime } = ( y _ { 1 } , \ldots , y _ { w - 1 } )
$$

in $S _ { n }$ . Since $S _ { n } \subseteq S _ { 1 }$ , there exists $a \in { \mathcal { X } }$ such that

$$
\sum _ { i = 1 } ^ { w - 1 } g ( x _ { i } ) + g ( a ) \leq B .
$$

Since $g _ { 0 } \leq g ( a )$ ，

$$
\sum _ { i = 1 } ^ { w - 1 } g ( x _ { i } ) + g _ { 0 } \leq B ,
$$

so $x _ { 0 }$ can be appended feasibly. After one reset action, the history becomes $( x _ { 2 } , \ldots , x _ { w - 1 } , x _ { 0 } )$ . Since $g _ { 0 } \leq g ( x _ { 1 } )$ replacing $x _ { 1 }$ by $x _ { 0 }$ cannot increase the window cost. Repeating this argument shows that $x _ { 0 }$ can be played $w - 1$ times while preserving feasibility. Hence $s \longrightarrow h ( x _ { 0 } ) : = ( x _ { 0 } , \ldots , x _ { 0 } )$ in at most $w - 1$ actions. Similarly, since $s ^ { \prime } \in S _ { 1 }$ , there exists $a ^ { \bar { \prime } } \in \mathcal { X }$ such that

$$
\sum _ { i = 1 } ^ { w - 1 } g ( y _ { i } ) + g ( a ^ { \prime } ) \leq B .
$$

Because $g _ { 0 } \leq g ( a ^ { \prime } )$

$$
g _ { 0 } + \sum _ { i = 1 } ^ { w - 1 } g ( y _ { i } ) \leq B .
$$

Starting from $h ( x _ { 0 } )$ , append $y _ { 1 } , \ldots , y _ { w - 1 }$ . After $j \in \{ 1 , \dots , w - 1 \}$ target actions have been appended, the current window has cost

$$
( w - j ) g _ { 0 } + \sum _ { i = 1 } ^ { j } g ( y _ { i } ) .
$$

Since $g _ { 0 } \leq g ( y _ { i } )$ for every $i ,$

$$
( w - j ) g _ { 0 } + \sum _ { i = 1 } ^ { j } g ( y _ { i } ) \leq g _ { 0 } + \sum _ { i = 1 } ^ { w - 1 } g ( y _ { i } ) \leq B .
$$

Thus every intermediate window is feasible, and $s ^ { \prime }$ is reached after at most $w - 1$ additional actions. Therefore, $d c ( s , s ^ { \prime } ) \leq 2 ( w - 1 )$ ). Since n and $s , s ^ { \prime } \in S _ { n }$ were arbitrary, $D \leq 2 ( w - 1 )$ □

Aggregate budget constraints. Consider

$$
{ \mathcal { C } } _ { \mathrm { b u d g e t } } = \left\{ \left( x _ { 1 } , \ldots , x _ { w } \right) \in { \mathcal { X } } ^ { w } : \sum _ { i = 1 } ^ { w } c ^ { \top } x _ { i } \leq B \right\} .
$$

Proposition F.5 (Aggregate Budget Constraints). Suppose $\mathcal { C } _ { \mathrm { b u d g e t } } \neq \emptyset$ and $c ^ { \top } x$ attains its minimum over $\mathcal { X }$ Then $D \leq 2 ( w - 1 )$ .

Proof. Set $g ( x ) : = c ^ { \top } x$ . Then

$$
{ \mathcal { C } } _ { \mathrm { b u d g e t } } = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) : \sum _ { i = 1 } ^ { w } g ( x _ { i } ) \leq B \right\} .
$$

The claim follows from Proposition F.4.

Exponential constraints. Consider

$$
\mathcal C _ { \exp } = \left\{ ( x _ { 1 } , \dots , x _ { w } ) \in \mathcal X ^ { w } : \sum _ { i = 1 } ^ { w } e ^ { a ^ { \top } x _ { i } - b } \leq \rho \right\} .
$$

Proposition F.6 (Exponential Constraints). Suppose $\mathcal { C } _ { \mathrm { e x p } } \neq \emptyset$ and $x \mapsto e ^ { a ^ { \top } x - b }$ attains its minimum over $\mathcal { X } .$ Then $D \leq 2 ( w - 1 )$ ).

Proof. Set $g ( x ) : = e ^ { a ^ { \top } x - b }$ . Then

$$
{ \mathcal { C } } _ { \exp } = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) : \sum _ { i = 1 } ^ { w } g ( x _ { i } ) \leq \rho \right\} .
$$

The result follows from Proposition F.4.

Aggregate $\ell _ { p } .$ -norm constraints. Consider

$$
\mathcal { C } _ { p } = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal { X } ^ { w } : \sum _ { i = 1 } ^ { w } \| A x _ { i } - b \| _ { p } ^ { p } \leq \rho ^ { p } \right\} .
$$

Proposition F.7 (Aggregate $\ell _ { p } .$ -Norm Constraints). Suppose $\mathcal { C } _ { p } \neq \emptyset$ and $x \mapsto \| A x - b \| _ { p } ^ { p }$ attains its minimum over X. Then $D \leq 2 ( w - 1 )$

Proof. Set $g ( x ) : = \| A x - b \| _ { p } ^ { p }$ . Then

$$
\mathcal { C } _ { p } = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) : \sum _ { i = 1 } ^ { w } g ( x _ { i } ) \leq \rho ^ { p } \right\} .
$$

The result follows from Proposition F.4.

Convex-risk constraints. Consider

$$
{ \mathcal { C } } _ { \mathrm { r i s k } } = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal { X } ^ { w } : \frac { 1 } { w } \sum _ { i = 1 } ^ { w } \ell ( x _ { i } ) \leq \rho \right\} .
$$

Proposition F.8 (Convex-Risk Constraints). Suppose $\mathcal { C } _ { \mathrm { { r i s k } } } \neq \emptyset$ and ℓ attains its minimum over $\mathcal { X }$ . Then $D \leq 2 ( w - 1 )$

Proof. The constraint is equivalent to $\textstyle \sum _ { i = 1 } ^ { w } \ell ( x _ { i } ) \leq w \rho$ . Thus it has the additive form of Proposition F.4 with $g ( x ) : = \ell ( x )$ , which gives $D \leq 2 ( w - 1 )$ ). □

Non-cyclic ramping constraints. Consider

$$
\begin{array} { r } { \mathcal { C } _ { \mathrm { r a m p } } ^ { \mathrm { n c } } = \left\{ \left( x _ { 1 } , \ldots , x _ { w } \right) \in \mathcal { X } ^ { w } : \left\| x _ { i + 1 } - x _ { i } \right\| \leq \rho , \quad i = 1 , \ldots , w - 1 \right\} . } \end{array}
$$

Assume that X is compact and convex, and define $D _ { \mathcal { X } } : = \operatorname* { m a x } _ { x , y \in \mathcal { X } } \left\| x - y \right\|$

Proposition F.9 (Exact Diameter for Non-Cyclic Ramping). Suppose w $\geq 2 , \rho > 0$ , and $D _ { \mathcal { X } } > 0$ . Then

$$
D = \left\lceil { \frac { D _ { \mathcal { X } } } { \rho } } \right\rceil + w - 2 .
$$

Proof. Fix $n \geq 1$ and

$$
s = ( x _ { 1 } , \ldots , x _ { w - 1 } ) , \qquad s ^ { \prime } = ( y _ { 1 } , \ldots , y _ { w - 1 } )
$$

in $S _ { n }$ . Since $S _ { n } \subseteq S _ { 1 }$ , both histories satisfy

$$
\begin{array} { r } { \| x _ { i + 1 } - x _ { i } \| \leq \rho , \qquad \| y _ { i + 1 } - y _ { i } \| \leq \rho , \qquad i = 1 , \ldots , w - 2 . } \end{array}
$$

Set

$$
N : = \left\lceil { \frac { \| x _ { w - 1 } - y _ { 1 } \| } { \rho } } \right\rceil .
$$

If $N = 0$ , then $x _ { w - 1 } = y _ { 1 }$ . Otherwise define

$$
z _ { j } : = x _ { w - 1 } + \frac { j } { N } ( y _ { 1 } - x _ { w - 1 } ) , \qquad j = 1 , \ldots , N .
$$

Table 5: History-state diameter for representative sliding-window constraints.
<table><tr><td>Constraint class</td><td>Window condition</td><td colspan="2">History diameter</td></tr><tr><td>State-independent</td><td> $\mathcal { C } = \mathcal { A } ^ { w }$ </td><td colspan="2">D = w − 1</td></tr><tr><td>Box</td><td> $\ell \leq x _ { i } \leq u$ </td><td colspan="2">D = w − 1</td></tr><tr><td>Maximum</td><td> $\operatorname* { m a x } _ { i } \phi ( x _ { i } ) \leq \rho$  i</td><td colspan="2">D = w − 1</td></tr><tr><td>Aggregate budget</td><td>xi ≤ B cλ i=1</td><td colspan="2">D ≤ 2(w − 1)</td></tr><tr><td>Exponential</td><td>w ∑ xi−b ≤ρ i=1 w</td><td colspan="2">D ≤ 2(w − 1)</td></tr><tr><td>Aggregate  $\ell _ { p }$ </td><td>∑∥|Axi − b||b ≤ ρp i=1 w</td><td colspan="2">D ≤ 2(w − 1)</td></tr><tr><td>Convex risk</td><td>∑l(xi) ≤ wρ i=1</td><td colspan="2">D ≤ 2(w − 1)</td></tr><tr><td>Non-cyclic ramping</td><td>||xi+1 − xi∥ ≤ ρ</td><td>[Dx D = ρ</td><td>+ w − 2</td></tr></table>

By convexity, $z _ { j } \in \mathcal { X }$ , and

$$
\| z _ { j } - z _ { j - 1 } \| = \frac { \| y _ { 1 } - x _ { w - 1 } \| } { N } \leq \rho .
$$

Hence $z _ { 1 } , \dotsc , z _ { N } = y _ { 1 }$ can be appended feasibly. Appending $y _ { 2 } , \ldots , y _ { w - 1 }$ then reaches $s ^ { \prime } .$ Therefore,

$$
d _ { \mathcal { C } } ( s , s ^ { \prime } ) \leq N + w - 2 \leq \left\lceil { \frac { D _ { \mathcal { X } } } { \rho } } \right\rceil + w - 2 .
$$

Taking the supremum gives

$$
D \leq \left\lceil { \frac { D _ { \mathcal { X } } } { \rho } } \right\rceil + w - 2 .
$$

For the reverse inequality, compactness gives $a , b \in { \mathcal { X } }$ such that $\| a - b \| = D _ { \mathcal { X } }$ . Consider the stationary histories

$$
s _ { a } = ( a , \ldots , a ) , \qquad s _ { b } = ( b , \ldots , b ) .
$$

Both belong to every $S _ { n }$ . Along any feasible transition from $s _ { a }$ to $s _ { b } ,$ consecutive actions can move by at most $\rho .$ By the triangle inequality, the first occurrence of b requires at least $\scriptstyle \left\lceil { \frac { D _ { \mathcal { X } } } { \rho } } \right\rceil$ appended actions. After the first $b ,$ another $w - 2$ copies of b are required to obtain the complete history $s _ { b } .$ Hence

$$
d _ { \mathcal { C } } ( s _ { a } , s _ { b } ) \geq \left\lceil \frac { D _ { \mathcal { X } } } { \rho } \right\rceil + w - 2 .
$$

Combining the upper and lower bounds proves the claim.

□

Thus, the ramping diameter separates into movement and memory-replacement costs:

$$
D = \left\lceil \frac { D _ { \mathcal { X } } } { \rho } \right\rceil + ( w - 2 ) .
$$

Table 5 summarizes the resulting history-state diameter bounds.

## G PROOFS FOR SECTION 4: GEOMETRY

## G.1 Proof of Theorem 4.4

Define

$$
M _ { w } : = \operatorname* { m a x } _ { ( x _ { 1 } , \ldots , x _ { w } ) \in { \mathcal C } } \sum _ { i = 1 } ^ { w } f ( x _ { i } ) .
$$

Since $\mathcal { X }$ is compact, ${ \mathcal { X } } ^ { w }$ is compact. Because ${ \mathcal { C } } \subseteq { \mathcal { X } } ^ { w }$ is closed and nonempty, it is also compact. Moreover, $\textstyle ( x _ { 1 } , \ldots , x _ { w } ) \mapsto \sum _ { i = 1 } ^ { w } f ( x _ { i } )$ is upper semicontinuous. Hence the maximum defining $M _ { w }$ is attained.

Lemma G.1. For any $x = ( x _ { 1 } , \dots , x _ { w } ) \in { \mathcal { C } } _ { : }$

$$
{ \frac { 1 } { w } } \sum _ { k = 0 } ^ { w - 1 } { \mathcal { S } } ^ { k } x = ( { \bar { z } } , \ldots , { \bar { z } } ) \in { \mathcal { C } } , \qquad { \bar { z } } : = { \frac { 1 } { w } } \sum _ { i = 1 } ^ { w } x _ { i } .
$$

Proof. By cyclic-shift invariance,

$$
S ^ { k } x \in { \mathcal { C } } , \qquad k = 0 , \dots , w - 1 .
$$

Since C is convex,

$$
\bar { x } : = \frac 1 w \sum _ { k = 0 } ^ { w - 1 } S ^ { k } x \in \mathcal C .
$$

For each coordinate $j \in \{ 1 , \dots , w \}$ , the j-th coordinates of $x , S x , \ldots , S ^ { w - 1 } x$ are $x _ { 1 } , \ldots , x _ { w }$ , each appearing exactly once. Therefore,

$$
\bar { x } _ { j } = \frac { 1 } { w } \sum _ { i = 1 } ^ { w } x _ { i } = \bar { z } .
$$

Thus every coordinate of ¯x equals ${ \bar { z } } ,$ and hence $\bar { x } = ( \bar { z } , \ldots , \bar { z } ) \in \mathcal { C }$

□

Lemma G.2. There exists $z ^ { \star } \in \mathcal { X }$ such that

$$
( z ^ { \star } , \ldots , z ^ { \star } ) \in { \mathcal { C } } \qquad a n d \qquad w f ( z ^ { \star } ) = M _ { w } .
$$

Proof. Let

$$
x ^ { \star } = ( x _ { 1 } ^ { \star } , \ldots , x _ { w } ^ { \star } ) \in \arg \operatorname* { m a x } _ { ( x _ { 1 } , \ldots , x _ { w } ) \in \mathcal { C } } \sum _ { i = 1 } ^ { w } f ( x _ { i } ) ,
$$

and define

$$
z ^ { \star } : = \frac { 1 } { w } \sum _ { i = 1 } ^ { w } x _ { i } ^ { \star } .
$$

By Lemma G.1, $( z ^ { \star } , \ldots , z ^ { \star } ) \in \mathcal { C }$ . By concavity of $f$ and Jensen’s inequality,

$$
f ( z ^ { \star } ) = f \left( \frac { 1 } { w } \sum _ { i = 1 } ^ { w } x _ { i } ^ { \star } \right) \geq \frac { 1 } { w } \sum _ { i = 1 } ^ { w } f ( x _ { i } ^ { \star } ) .
$$

Therefore,

$$
w f ( z ^ { \star } ) \geq \sum _ { i = 1 } ^ { w } f ( x _ { i } ^ { \star } ) = M _ { w } .
$$

On the other hand, since $( z ^ { \star } , \ldots , z ^ { \star } ) \in \mathcal { C }$ , this stationary window is feasible for the optimization defining $M _ { w }$ Hence

$$
w f ( z ^ { \star } ) \leq M _ { w } .
$$

Thus,

$$
w f ( z ^ { \star } ) = M _ { w } .
$$

Proof of Theorem $4 . 4$ . Define

$$
\Delta : \mathcal { X }  \mathcal { X } ^ { w } , \qquad \Delta ( z ) : = ( z , \dots , z ) .
$$

Then $\mathcal { Z } = \Delta ^ { - 1 } ( \mathcal { C } )$ . Since C is nonempty, choose any $x = ( x _ { 1 } , \dots , x _ { w } ) \in \mathcal { C }$ . By Lemma G.1, its cyclic average has the form $( { \bar { z } } , \dots , { \bar { z } } ) \in { \mathcal { C } } .$ , so $\bar { z } \in \mathcal { Z }$ . Hence $\mathcal { Z }$ is nonempty. Moreover, $\Delta$ is continuous and $\mathcal { C }$ is closed, so

$\mathcal { Z } = \Delta ^ { - 1 } ( \mathcal { C } )$ is closed in the compact set X. Therefore, Z is compact. By Lemma G.2, there exists $z ^ { \star } \in \mathcal { Z }$ such that $v f ( z ^ { \star } ) = M _ { w }$ . For every $z \in { \mathcal { Z } }$ , we have $( z , \dots , z ) \in { \mathcal { C } }$ , and therefore

$$
w f ( z ) \leq M _ { w } = w f ( z ^ { \star } ) .
$$

Thus,

$$
z ^ { \star } \in \arg \operatorname* { m a x } _ { z \in \mathcal { Z } } f ( z ) .
$$

The stationary trajectory $x _ { t } = z ^ { \star }$ is feasible because every length-w window is equal to $( z ^ { \star } , \ldots , z ^ { \star } ) \in \mathcal { C }$ . Hence

$$
V _ { T } ( f ) \geq T f ( z ^ { \star } ) .
$$

Now write

$$
T = q w + r , \qquad q \in \mathbb { N } _ { 0 } , \qquad 0 \leq r < w .
$$

Let $y _ { 1 : T } = ( y _ { 1 } , \dots , y _ { T } )$ be any feasible trajectory. For each $j = 1 , \dots , q ;$ the block

$$
\left( y _ { ( j - 1 ) w + 1 } , \ldots , y _ { j w } \right)
$$

is one of the realized length-w windows and therefore belongs to C. By the definition of $M _ { w } .$

$$
\sum _ { i = ( j - 1 ) w + 1 } ^ { j w } f ( y _ { i } ) \leq M _ { w } .
$$

Since $\mathcal { X }$ is compact and $f$ is upper semicontinuous, $f _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { x \in \mathcal { X } } f ( x )$ is well defined and attained. The remaining r actions therefore contribute at most $r f _ { \mathrm { m a x } } .$ Consequently,

$$
\sum _ { t = 1 } ^ { T } f ( y _ { t } ) \leq q M _ { w } + r f _ { \operatorname* { m a x } } .
$$

Taking the maximum over all feasible trajectories and using $M _ { w } = w f ( z ^ { \star } )$ , we obtain

$$
V _ { T } ( f ) \leq q w f ( z ^ { \star } ) + r f _ { \operatorname* { m a x } } = T f ( z ^ { \star } ) + r \big ( f _ { \operatorname* { m a x } } - f ( z ^ { \star } ) \big ) .
$$

Combining this with ${ V _ { T } } ( f ) \geq T f ( z ^ { \star } )$ gives

$$
0 \leq V _ { T } ( f ) - T f ( z ^ { \star } ) \leq r \big ( f _ { \operatorname* { m a x } } - f ( z ^ { \star } ) \big ) .
$$

Since $z ^ { \star } \in \mathcal { X }$ , we have $f _ { \mathrm { m a x } } - f ( z ^ { \star } ) \ge 0$ . Using $r \leq w - 1$ ，

$$
0 \leq V _ { T } ( f ) - T f ( z ^ { \star } ) \leq r \big ( f _ { \operatorname* { m a x } } - f ( z ^ { \star } ) \big ) \leq ( w - 1 ) \big ( f _ { \operatorname* { m a x } } - f ( z ^ { \star } ) \big ) .
$$

Finally, if $w \mid T ,$ , then $r = 0$ , and hence

$$
V _ { T } ( f ) = T f ( z ^ { \star } ) .
$$

Therefore, the stationary trajectory $x _ { t } = z ^ { \star }$ is optimal.

## G.2 Tightness of the Stationary Approximation Gap

We next show that the $\mathcal { O } ( w )$ finite-horizon gap in Theorem 4.4 is unavoidable in general. A simple aggregatebudget example shows that, for a suitable choice of the horizon, this gap can scale linearly with w. In fact, the upper bound in Theorem 4.4 is attained exactly.

Proposition G.3. Let w be even, $\mathcal { X } = [ 0 , 1 ]$ , and

$$
\mathcal { C } = \left\{ ( x _ { 1 } , \ldots , x _ { w } ) \in [ 0 , 1 ] ^ { w } : \sum _ { i = 1 } ^ { w } x _ { i } \leq \frac { w } { 2 } \right\} .
$$

For the linear reward $f ( x ) = x$ , the best stationary feasible action is $z ^ { \star } = 1 / 2$ . Moreover, for any $q \geq 0$ and $\begin{array} { r } { T = q w + \frac { w } { 2 } } \end{array}$ , the ofline gap is exactly

$$
V _ { T } - T f ( z ^ { \star } ) = \frac { w } { 4 } .
$$

Thus, the $\mathcal { O } ( w )$ dependence in Theorem $4 . 4$ is tight in general.

Proof. A stationary action $z \in [ 0 , 1 ]$ is feasible if and only if $w z \leq { \begin{array} { l } { w } \\ { 2 } \end{array} }$ , and hence the stationary feasible set is $\mathcal { Z } = [ 0 , 1 / 2 ]$ . Since $f ( x ) = x$ , the optimal stationary feasible action is therefore $z ^ { \star } = \textstyle { \frac { 1 } { 2 } }$ . Thus the stationary trajectory has value

$$
T f ( z ^ { \star } ) = \frac { T } { 2 } = \frac { q w } { 2 } + \frac { w } { 4 } .
$$

We next construct a feasible nonstationary trajectory. Consider the periodic sequence of period w given by

$$
\underbrace { 1 , \dots , 1 } _ { w / 2 } , \underbrace { 0 , \dots , 0 } _ { w / 2 } , \underbrace { 1 , \dots , 1 } _ { w / 2 } , \underbrace { 0 , \dots , 0 } _ { w / 2 } , \underbrace { \dots } _ { \begin{array} { l } { w / 2 } \end{array} } .
$$

and truncate it after $T = q w + w / 2$ rounds. Every block of w consecutive actions contains exactly $w / 2$ entries equal to 1 and $w / 2$ entries equal to 0. Therefore, for every $t = w , \ldots , T .$

$$
\sum _ { i = t - w + 1 } ^ { t } x _ { i } = \frac { w } { 2 } ,
$$

so every sliding window belongs to $\mathcal { C } .$ Each complete period contributes reward $w / 2 ,$ , and the final $w / 2$ actions are all equal to 1. Hence this feasible trajectory achieves

$$
\sum _ { t = 1 } ^ { T } f ( x _ { t } ) = q { \frac { w } { 2 } } + { \frac { w } { 2 } } .
$$

It follows that

$$
V _ { T } \geq q \frac { w } { 2 } + \frac { w } { 2 } ,
$$

and therefore

$$
V _ { T } - T f ( z ^ { \star } ) \geq \left( q { \frac { w } { 2 } } + { \frac { w } { 2 } } \right) - \left( q { \frac { w } { 2 } } + { \frac { w } { 4 } } \right) = { \frac { w } { 4 } } .
$$

To obtain the matching upper bound, apply Theorem 4.4 with $r = w / 2$ . Since

$$
f _ { \mathrm { m a x } } = \operatorname* { m a x } _ { x \in [ 0 , 1 ] } f ( x ) = 1 \qquad { \mathrm { a n d } } \qquad f ( z ^ { \star } ) = { \frac { 1 } { 2 } } ,
$$

Theorem 4.4 gives

$$
V _ { T } - T f ( z ^ { \star } ) \leq r { \big ( } f _ { \operatorname* { m a x } } - f ( z ^ { \star } ) { \big ) } = { \frac { w } { 2 } } \left( 1 - { \frac { 1 } { 2 } } \right) = { \frac { w } { 4 } } .
$$

Combining the lower and upper bounds yields

$$
V _ { T } - T f ( z ^ { \star } ) = \frac { w } { 4 } .
$$

Thus the additive $\mathcal { O } ( w )$ finite-horizon term in Theorem 4.4 is tight.

Remark G.4. The gap comes from the horizon ending partway through a period. Over each complete block of length w, both the stationary and nonstationary trajectories achieve the same average reward of $1 / 2 .$ The diference appears in the final incomplete block: the nonstationary trajectory can use its high-reward actions first, gaining an additional $w / 4$ before the horizon ends.

## H PROOFS FOR SECTION 5: REACHABILITY AND LEARNING

## H.1 Proof of Proposition 5.2

Proof. Consider the one-dimensional action space $\mathcal { X } = [ - 1 , 1 ]$ and let $w = 2$ . Define

$$
{ \mathcal { C } } = \left\{ ( x _ { 1 } , x _ { 2 } ) \in [ - 1 , 1 ] ^ { 2 } : x _ { 1 } = x _ { 2 } \right\} .
$$

The set C is nonempty, closed, and convex. It is also invariant under cyclic shifts, since

$$
( x _ { 1 } , x _ { 2 } ) \in { \mathcal { C } } \quad \Longrightarrow \quad x _ { 1 } = x _ { 2 } \quad \Longrightarrow \quad ( x _ { 2 } , x _ { 1 } ) \in { \mathcal { C } } .
$$

Thus, Assumption 4.1 holds. The stationary feasible action set is

$$
\mathcal { Z } = \left\{ z \in [ - 1 , 1 ] : ( z , z ) \in \mathcal { C } \right\} = [ - 1 , 1 ] .
$$

Let $\theta ^ { \star } \in \{ - 1 , 1 \}$ and consider the noiseless linear reward model $Y _ { t } = \theta ^ { \star } x _ { t }$ . This is a special case of the condi tionally sub-Gaussian model with $\eta _ { t } = 0$ . For any admissible trajectory,

$$
( x _ { t - 1 } , x _ { t } ) \in \mathcal { C } , \qquad t = 2 , \dotsc , T ,
$$

which implies

$$
x _ { t } = x _ { t - 1 } , \qquad t = 2 , \ldots , T .
$$

Hence

$$
x _ { t } = x _ { 1 } , \qquad t = 1 , \ldots , T .
$$

Thus, once the learner chooses its first action, feasibility forces it to repeat that action for the entire horizon. For $\theta ^ { \star } = 1$ , the optimal stationary feasible action is $z ^ { \star } = 1$ , and therefore

$$
\operatorname { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; 1 ) = T \left( 1 - \mathbb { E } _ { \pi } [ x _ { 1 } ] \right) .
$$

For $\theta ^ { \star } = - 1$ , the optimal stationary feasible action is $z ^ { \star } = - 1$ , giving

$$
\begin{array} { r } { \mathrm { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; - 1 ) = T \left( 1 + \mathbb { E } _ { \pi } [ x _ { 1 } ] \right) . } \end{array}
$$

Averaging the two regrets yields

$$
\frac { 1 } { 2 } \operatorname { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; 1 ) + \frac { 1 } { 2 } \operatorname { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; - 1 ) = T .
$$

Therefore,

$$
\operatorname* { s u p } _ { \theta ^ { \star } \in \{ - 1 , 1 \} } \operatorname { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; \theta ^ { \star } ) \geq T .
$$

Thus, every admissible policy sufers linear regret for at least one value of $\theta ^ { \star }$

## H.2 Proof of Theorem 5.4

We first establish several auxiliary lemmas controlling the confidence sets, covariance growth within an episode, the elliptical potential, the number of switches, and feasibility. We then derive a high-probability stationaryregret bound and convert it to expected regret against the ofline-optimal feasible trajectory.

Lemma H.1. For $t \geq 1$ , define

$$
\mathcal { E } _ { t } ( \delta ) : = \left\{ \theta \in \mathbb { R } ^ { d } : \| \theta - \widehat { \theta } _ { t } \| _ { V _ { t } } \leq \beta _ { t } ( \delta ) \right\} ,
$$

where

$$
V _ { t } = \lambda I _ { d } + \sum _ { s \in \mathcal { O } _ { t } } x _ { s } x _ { s } ^ { \top } , \qquad \widehat { \theta } _ { t } = V _ { t } ^ { - 1 } \sum _ { s \in \mathcal { O } _ { t } } x _ { s } Y _ { s } ,
$$

and

$$
\beta _ { t } ( \delta ) = R \sqrt { 2 \log \biggl ( \frac { \operatorname * { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \operatorname * { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \biggr ) } + \sqrt { \lambda } S .
$$

Then, with probability at least $1 - \delta$

$$
\theta ^ { \star } \in \mathcal { E } _ { t } ( \delta ) , \qquad t = 1 , \dots , T + 1 .
$$

Moreover,

$$
\beta _ { t } ( \delta ) \leq \overline { { \beta } } _ { T } ( \delta ) , \qquad t \leq T + 1 ,
$$

where

$$
\overline { { \beta } } _ { T } ( \delta ) : = R \sqrt { d \log \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) + 2 \log \left( \frac { 1 } { \delta } \right) } + \sqrt { \lambda } S .
$$

Proof. For $s \geq 1$ , define $\widetilde { x } _ { s } : = I _ { s } x _ { s }$ . Since $I _ { s }$ and $x _ { s }$ are $\mathcal { F } _ { s - 1 }$ -measurable, $\widetilde { x } _ { s }$ is predictable. Moreover, because $I _ { s } \in \{ 0 , 1 \}$

$$
V _ { t } = \lambda I _ { d } + \sum _ { s = 1 } ^ { t - 1 } \widetilde { x } _ { s } \widetilde { x } _ { s } ^ { \top } .
$$

By the conditional R-sub-Gaussian noise assumption and the self-normalized concentration inequality for vectorvalued martingales (Abbasi-Yadkori et al., 2011, Theorem 1), with probability at least $1 - \delta .$ simultaneously for all $t \leq T + 1$

$$
\left\| \sum _ { s = 1 } ^ { t - 1 } \widetilde { x } _ { s } \eta _ { s } \right\| _ { V _ { t } ^ { - 1 } } \le R \sqrt { 2 \log \biggl ( \frac { \operatorname * { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \operatorname * { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \biggr ) } .
$$

Since

$$
\sum _ { s = 1 } ^ { t - 1 } \widetilde { x } _ { s } \eta _ { s } = \sum _ { s \in \mathcal { O } _ { t } } x _ { s } \eta _ { s } ,
$$

the same bound holds for the retained observations. Using $Y _ { s } = \langle \theta ^ { \star } , x _ { s } \rangle + \eta _ { s }$ , we obtain

$$
V _ { t } \widehat { \theta } _ { t } = \sum _ { s \in \mathcal { O } _ { t } } x _ { s } Y _ { s } = \sum _ { s \in \mathcal { O } _ { t } } x _ { s } x _ { s } ^ { \top } \theta ^ { \star } + \sum _ { s \in \mathcal { O } _ { t } } x _ { s } \eta _ { s } .
$$

Since

$$
V _ { t } \theta ^ { \star } = \lambda \theta ^ { \star } + \sum _ { s \in \mathcal { O } _ { t } } x _ { s } x _ { s } ^ { \top } \theta ^ { \star } ,
$$

it follows that

$$
V _ { t } ( \widehat { \theta } _ { t } - \theta ^ { \star } ) = \sum _ { s \in \mathcal { O } _ { t } } x _ { s } \eta _ { s } - \lambda \theta ^ { \star } .
$$

Hence

$$
\lVert \widehat { { \boldsymbol \theta } } _ { t } - { \boldsymbol \theta } ^ { \star } \rVert _ { V _ { t } } = \left. \sum _ { s \in \mathcal { O } _ { t } } x _ { s } \eta _ { s } - \lambda { \boldsymbol \theta } ^ { \star } \right. _ { V _ { t } ^ { - 1 } } \leq \left. \sum _ { s \in \mathcal { O } _ { t } } x _ { s } \eta _ { s } \right. _ { V _ { t } ^ { - 1 } } + \lambda \lVert { \boldsymbol \theta } ^ { \star } \rVert _ { V _ { t } ^ { - 1 } } .
$$

Because $V _ { t } \succeq \lambda I _ { d }$ , and inversion reverses the Loewner order for positive definite matrices,

$$
V _ { t } ^ { - 1 } \preceq ( \lambda I _ { d } ) ^ { - 1 } = \lambda ^ { - 1 } I _ { d } .
$$

we have

$$
\lambda \| \theta ^ { \star } \| _ { V _ { t } ^ { - 1 } } \leq \sqrt { \lambda } \| \theta ^ { \star } \| _ { 2 } \leq \sqrt { \lambda } S .
$$

Therefore, simultaneously for all $t \leq T + 1$ ，

$$
\| \widehat { \theta } _ { t } - \theta ^ { \star } \| _ { V _ { t } } \leq R \sqrt { 2 \log \biggl ( \frac { \operatorname* { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \operatorname* { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \biggr ) } + \sqrt { \lambda } S = \beta _ { t } ( \delta ) .
$$

Thus, with probability at least $1 - \delta .$

$$
\theta ^ { \star } \in \mathcal { E } _ { t } ( \delta ) , \qquad \forall t \leq T + 1 .
$$

It remains to upper bound $\beta _ { t } ( \delta )$ . Let $N _ { t } : = | \mathcal { O } _ { t } |$ |. Since $N _ { t } \leq T$ and $\| x _ { s } \| _ { 2 } \leq L ,$

$$
\mathrm { t r } ( V _ { t } ) = \lambda d + \sum _ { s \in \mathcal { O } _ { t } } \| x _ { s } \| _ { 2 } ^ { 2 } \leq \lambda d + N _ { t } L ^ { 2 } .
$$

Let $\lambda _ { 1 } , \ldots , \lambda _ { d }$ denote the eigenvalues of $V _ { t } .$ By the arithmetic–geometric mean inequality,

$$
\operatorname* { d e t } ( V _ { t } ) = \prod _ { i = 1 } ^ { d } \lambda _ { i } \leq \left( \frac { \mathrm { t r } ( V _ { t } ) } { d } \right) ^ { d } \leq \left( \lambda + \frac { N _ { t } L ^ { 2 } } { d } \right) ^ { d } .
$$

Since det $( \lambda I _ { d } ) = \lambda ^ { d }$ , we obtain

$$
\frac { \operatorname * { d e t } ( V _ { t } ) } { \operatorname * { d e t } ( \lambda I _ { d } ) } \leq \left( 1 + \frac { N _ { t } L ^ { 2 } } { \lambda d } \right) ^ { d } \leq \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) ^ { d } .
$$

Substituting this into the definition of $\beta _ { t } ( \delta )$ yields

$$
\beta _ { t } ( \delta ) \leq R \sqrt { d \log \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) + 2 \log \left( \frac { 1 } { \delta } \right) } + \sqrt { \lambda } S = \overline { { \beta } } _ { T } ( \delta ) .
$$

This proves both claims.

Lemma H.2. Suppose $0 \prec A \preceq B$ and

$$
\operatorname* { d e t } ( B ) \leq c \operatorname* { d e t } ( A ) , \qquad c \geq 1 .
$$

Then

$$
B \preceq c A , \qquad A ^ { - 1 } \preceq c B ^ { - 1 } .
$$

Consequently, for every $\boldsymbol { x } \in \mathbb { R } ^ { d } .$

$$
\| x \| _ { A ^ { - 1 } } \leq \sqrt { c } \| x \| _ { B ^ { - 1 } } .
$$

Proof. Define $M : = A ^ { - 1 / 2 } B A ^ { - 1 / 2 }$ . Since $A \preceq B$

$$
M \succeq I _ { d } .
$$

Hence every eigenvalue $\mu _ { i } ( M )$ satisfies $\mu _ { i } ( M ) \geq 1$ . Moreover,

$$
\prod _ { i = 1 } ^ { d } \mu _ { i } ( M ) = \operatorname* { d e t } ( M ) = { \frac { \operatorname* { d e t } ( B ) } { \operatorname* { d e t } ( A ) } } \leq c .
$$

Since all eigenvalues are at least one, each satisfies $\mu _ { i } ( M ) \leq c .$ Thus $M \preceq c I _ { d }$ , which implies $B \preceq c A$ . Taking inverses gives $A ^ { - 1 } \preceq c B ^ { - 1 }$ . Therefore,

$$
\| x \| _ { A ^ { - 1 } } ^ { 2 } \leq c \| x \| _ { B ^ { - 1 } } ^ { 2 } ,
$$

and the result follows by taking square roots.

Lemma H.3. Let

$$
V _ { 1 } = \lambda I _ { d } , \qquad V _ { n + 1 } = V _ { n } + u _ { n } u _ { n } ^ { \top } ,
$$

where

$$
\| u _ { n } \| _ { 2 } \leq L , \qquad \lambda \geq L ^ { 2 } .
$$

Then

$$
\sum _ { n = 1 } ^ { N } \| u _ { n } \| _ { V _ { n } ^ { - 1 } } \leq \sqrt { 2 N d \log \biggl ( 1 + \frac { N L ^ { 2 } } { \lambda d } \biggr ) } .
$$

Proof. Since $V _ { n } \succeq \lambda I _ { d }$ , and inversion reverses the Loewner order for positive definite matrices, we obtain

$$
V _ { n } ^ { - 1 } \preceq ( \lambda I _ { d } ) ^ { - 1 } = \lambda ^ { - 1 } I _ { d } .
$$

Therefore,

$$
\left\| u _ { n } \right\| _ { V _ { n } ^ { - 1 } } ^ { 2 } \leq { \frac { \| u _ { n } \| _ { 2 } ^ { 2 } } { \lambda } } \leq 1 .
$$

Define $q _ { n } : = \| u _ { n } \| _ { V _ { n } ^ { - } } ^ { 2 }$ <sub>1</sub>. By the matrix determinant lemma,

$$
\operatorname* { d e t } ( V _ { n + 1 } ) = \operatorname* { d e t } ( V _ { n } ) ( 1 + q _ { n } ) .
$$

Since $q _ { n } \in [ 0 , 1 ]$

$$
q _ { n } \leq 2 \log ( 1 + q _ { n } ) .
$$

Therefore,

$$
\sum _ { n = 1 } ^ { N } q _ { n } \leq 2 \sum _ { n = 1 } ^ { N } \log ( 1 + q _ { n } ) = 2 \log { \frac { \operatorname* { d e t } ( V _ { N + 1 } ) } { \operatorname* { d e t } ( V _ { 1 } ) } } .
$$

As in Lemma H.1,

$$
{ \frac { \operatorname * { d e t } ( V _ { N + 1 } ) } { \operatorname * { d e t } ( V _ { 1 } ) } } \leq \left( 1 + { \frac { N L ^ { 2 } } { \lambda d } } \right) ^ { d } .
$$

Hence

$$
\sum _ { n = 1 } ^ { N } \| u _ { n } \| _ { V _ { n } ^ { - 1 } } ^ { 2 } \leq 2 d \log \biggl ( 1 + \frac { N L ^ { 2 } } { \lambda d } \biggr ) .
$$

Finally, Cauchy–Schwarz gives

$$
\sum _ { n = 1 } ^ { N } \| u _ { n } \| _ { V _ { n } ^ { - 1 } } \leq \sqrt { N \sum _ { n = 1 } ^ { N } \| u _ { n } \| _ { V _ { n } ^ { - 1 } } ^ { 2 } } \leq \sqrt { 2 N d \log \biggl ( 1 + \frac { N L ^ { 2 } } { \lambda d } \biggr ) } .
$$

Lemma H.4. Let $K _ { T }$ denote the number of determinant-doubling episodes and $J _ { T }$ the number of actual stationary-action switches. Then

$$
K _ { T } \leq 1 + \left\lceil d \log _ { 2 } \left( 1 + { \frac { T L ^ { 2 } } { \lambda d } } \right) \right\rceil \qquad a n d \qquad J _ { T } \leq \left\lceil d \log _ { 2 } \left( 1 + { \frac { T L ^ { 2 } } { \lambda d } } \right) \right\rceil .
$$

Proof. Let $W _ { k }$ denote the design matrix at the beginning of episode k. For every $k = 1 , \dots , K _ { T } - 1$ , episode $k + 1$ begins only after the determinant has more than doubled. Hence, $\operatorname* { d e t } ( W _ { k + 1 } ) > 2 \operatorname* { d e t } ( W _ { k } )$ . Iterating this inequality gives

$$
\operatorname * { d e t } ( W _ { K _ { T } } ) > 2 ^ { K _ { T } - 1 } \operatorname * { d e t } ( W _ { 1 } ) .
$$

Since $W _ { 1 } = \lambda I _ { d }$ and $W _ { K _ { T } } \preceq V _ { T + 1 }$ , we have

$$
2 ^ { K _ { T } - 1 } < \frac { \operatorname * { d e t } ( V _ { T + 1 } ) } { \operatorname * { d e t } ( \lambda I _ { d } ) } .
$$

By the determinant bound from Lemma H.1,

$$
\frac { \operatorname * { d e t } ( V _ { T + 1 } ) } { \operatorname * { d e t } ( \lambda I _ { d } ) } \leq \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) ^ { d } .
$$

Therefore,

$$
K _ { T } - 1 < d \log _ { 2 } \left( 1 + { \frac { T L ^ { 2 } } { \lambda d } } \right) ,
$$

and hence

$$
K _ { T } \leq 1 + \left\lceil d \log _ { 2 } \left( 1 + { \frac { T L ^ { 2 } } { \lambda d } } \right) \right\rceil .
$$

Finally,

$$
J _ { T } = \sum _ { k = 1 } ^ { K _ { T } - 1 } \mathbf { 1 } \{ z _ { k + 1 } \neq z _ { k } \} \leq K _ { T } - 1 ,
$$

so

$$
J _ { T } \leq \left\lceil d \log _ { 2 } \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) \right\rceil .
$$

Lemma H.5. Under Assumption 5.3, the rare-switching OFUL policy is feasible at every round.

Proof. When $w = 1$ , feasibility is pointwise and every selected stationary action belongs to $\mathcal { Z } ,$ , so the claim is immediate. Hence suppose $w \ge 2 .$ . The first selected action satisfies $z _ { 1 } \in \mathcal { Z } , \mathrm { s o } \ ( z _ { 1 } , \dots , z _ { 1 } ) \in \mathcal { C }$ . The learner repeats $z _ { 1 }$ for at least $w - 1$ rounds before attempting any transition. Thus, whenever a first transition is attempted, the current history is $h ( z _ { 1 } ) = ( z _ { 1 } , . . . , z _ { 1 } )$ . Suppose inductively that immediately before a change from episode k to episode $k + 1$ , the current history is $h ( z _ { k } ) . \operatorname { I f } z _ { k + 1 } = z _ { k } .$ , no transition is needed, and continuing to play $z _ { k }$ produces only the feasible stationary window $( z _ { k } , \dots , z _ { k } ) \in \mathcal { C }$ . If $z _ { k + 1 } \neq z _ { k } ,$ , Assumption 5.3 and the transition oracle provide actions

$$
( a _ { 1 } , \ldots , a _ { m } ) , \qquad m \leq \tau ,
$$

such that, starting from $h _ { 0 } = h ( z _ { k } )$ ，

$$
h _ { j } = F ( h _ { j - 1 } , a _ { j } ) , \qquad ( h _ { j - 1 } , a _ { j } ) \in \mathcal { C } , \qquad j = 1 , \dotsc , m ,
$$

and $h _ { m } \ = \ h ( z _ { k + 1 } )$ . Thus every window encountered during the transition is feasible. After the transition, repeatedly playing $z _ { k + 1 }$ also preserves feasibility. The result follows by induction over the episodes. □

Lemma H.6. With probability at least $1 - \delta$

$$
\begin{array} { r l } & { \mathfrak { N } _ { T } ^ { \mathcal { Z } } : = \displaystyle \sum _ { t = 1 } ^ { T } \langle \theta ^ { \star } , z ^ { \star } - x _ { t } \rangle \leq 4 \overline { { \beta } } _ { T } ( \delta ) \sqrt { T d \log \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) } } \\ & { \qquad + 2 L S ( w - 1 ) } \\ & { \qquad + 2 L S \tau \left[ d \log _ { 2 } \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) \right] . } \end{array}
$$

Proof. Define the uniform confidence event

$$
\mathcal { G } _ { \delta } : = \left\{ \theta ^ { \star } \in \mathcal { E } _ { t } ( \delta ) , \quad t = 1 , \ldots , T + 1 \right\} .
$$

By Lemma H.1, $\mathbb { P } ( \mathcal { G } _ { \delta } ) \ge 1 - \delta$ . We condition on $\mathcal { G } _ { \delta }$ . Partition the rounds into retained stationary rounds O and transition rounds Q. Then

$$
\Re _ { T } ^ { \mathcal { Z } } = \Re _ { T } ^ { \mathrm { s t a t } } + \Re _ { T } ^ { \mathrm { t r a n s } } .
$$

Stationary rounds. Consider $t \in \mathcal { O }$ belonging to episode k. Then $x _ { t } = z _ { k }$ . Let $W _ { k } : = V _ { t _ { k } }$ . By optimism,

$$
z _ { k } \in \arg \operatorname* { m a x } _ { z \in \mathcal { Z } } \left\{ \langle \widehat { \theta } _ { t _ { k } } , z \rangle + \beta _ { t _ { k } } ( \delta ) \| z \| _ { W _ { k } ^ { - 1 } } \right\} .
$$

Since $z ^ { \star } \in { \mathcal { Z } }$ and $\theta ^ { \star } \in { \mathcal { E } } _ { t _ { k } } ( \delta )$

$$
\begin{array} { r } { \langle \theta ^ { \star } , z ^ { \star } \rangle \leq \langle \widehat { \theta } _ { t _ { k } } , z _ { k } \rangle + \beta _ { t _ { k } } ( \delta ) \| z _ { k } \| _ { W _ { k } ^ { - 1 } } . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { \langle \theta ^ { \star } , z ^ { \star } - z _ { k } \rangle \leq \langle \widehat \theta _ { t _ { k } } - \theta ^ { \star } , z _ { k } \rangle + \beta _ { t _ { k } } ( \delta ) \| z _ { k } \| _ { W _ { k } ^ { - 1 } } \leq 2 \beta _ { t _ { k } } ( \delta ) \| z _ { k } \| _ { W _ { k } ^ { - 1 } } \leq 2 \overline { \beta } _ { T } ( \delta ) \| z _ { k } \| _ { W _ { k } ^ { - 1 } } . } \end{array}
$$

During the initial warm-up, the algorithm may continue playing $z _ { 1 }$ after the determinant-doubling threshold has already been crossed in order to establish the stationary history. Let

$$
\begin{array} { r } { \mathcal { H } : = \{ t \in \mathcal { O } : t \mathrm { ~ b e l o n g s ~ t o ~ e p i s o d e ~ 1 ~ a n d ~ } \operatorname* { d e t } ( V _ { t } ) > 2 \operatorname* { d e t } ( W _ { 1 } ) \} . } \end{array}
$$

The forced warm-up lasts for at most $w - 1$ rounds, so $\vert \mathcal { H } \vert \le w - 1$ . For every $t \in \mathcal { H }$

$$
\begin{array} { r } { \langle \theta ^ { \star } , z ^ { \star } - x _ { t } \rangle \leq \| \theta ^ { \star } \| _ { 2 } \| z ^ { \star } - x _ { t } \| _ { 2 } \leq 2 L S . } \end{array}
$$

Hence

$$
\sum _ { t \in \mathcal { H } } \langle \theta ^ { \star } , z ^ { \star } - x _ { t } \rangle \leq 2 L S ( w - 1 ) .
$$

Now consider $t \in \mathcal { O } \setminus \mathcal { H }$ in episode k. Before the determinant-doubling threshold is crossed,

$$
W _ { k } \preceq V _ { t } , \qquad \operatorname* { d e t } ( V _ { t } ) \leq 2 \operatorname* { d e t } ( W _ { k } ) .
$$

Lemma H.2 therefore gives $W _ { k } ^ { - 1 } \preceq 2 { V _ { t } ^ { - 1 } }$ , and consequently $\| z _ { k } \| _ { W _ { k } ^ { - 1 } } \leq \sqrt { 2 } \| z _ { k } \| _ { V _ { t } ^ { - 1 } }$ . Thus,

$$
\begin{array} { r } { \langle \theta ^ { \star } , z ^ { \star } - z _ { k } \rangle \leq 2 \sqrt { 2 } \overline { { \beta } } _ { T } ( \delta ) \| z _ { k } \| _ { V _ { t } ^ { - 1 } } . } \end{array}
$$

Let $N : = | \mathcal { O } | \le T$ and enumerate the stationary rounds as $t _ { 1 } < \cdots < t _ { N }$ . Set

$$
u _ { n } : = x _ { t _ { n } } , \qquad \widetilde { V } _ { n } : = \lambda I _ { d } + \sum _ { j = 1 } ^ { n - 1 } u _ { j } u _ { j } ^ { \top } .
$$

Since transition observations are excluded from the estimator,

$$
\widetilde { V } _ { n } = V _ { t _ { n } } .
$$

Lemma H.3 gives

$$
\sum _ { t \in \mathcal { O } } \left\| x _ { t } \right\| _ { V _ { t } ^ { - 1 } } \leq \sqrt { 2 T d \log \biggl ( 1 + \frac { T L ^ { 2 } } { \lambda d } \biggr ) } .
$$

Therefore,

$$
\Re _ { T } ^ { \mathrm { s t a t } } \leq 2 \sqrt { 2 } \overline { { \beta } } _ { T } ( \delta ) \sum _ { t \in \mathcal { O } } \left. x _ { t } \right. _ { V _ { t } ^ { - 1 } } + 2 L S ( w - 1 ) \leq 4 \overline { { \beta } } _ { T } ( \delta ) \sqrt { T d \log \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) } + 2 L S ( w - 1 ) .
$$

Transition rounds. For every $t \in \mathcal { Q }$

$$
\langle \theta ^ { \star } , z ^ { \star } - x _ { t } \rangle \leq 2 L S .
$$

Each actual switch requires at most τ transition rounds, so $| \mathcal { Q } | \leq \tau J _ { T }$ . By Lemma H.4,

$$
J _ { T } \leq \left\lceil d \log _ { 2 } \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) \right\rceil .
$$

Hence

$$
\Re _ { T } ^ { \mathrm { t r a n s } } \leq 2 L S \tau \left[ d \log _ { 2 } \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) \right] .
$$

Combining the stationary and transition contributions proves the claim.

Proof of Theorem 5.4. By Lemma H.5, the rare-switching OFUL policy is feasible at every round and is therefore admissible. Choose $\begin{array} { r } { \delta = \frac { \mathrm { ~ i ~ } } { T } } \end{array}$ . By Lemma H.6, on an event of probability at least $1 - 1 / T$

$$
\Re _ { T } ^ { \mathcal Z } = \widetilde { \mathcal O } \left( d \sqrt { T } + \tau d + w \right) .
$$

Indeed,

$$
\overline { { \beta } } _ { T } ( 1 / T ) = \widetilde { \cal O } ( \sqrt { d } ) ,
$$

so the statistical term is $\mathcal { \widetilde { O } } ( d \sqrt { T } )$ , the transition term is $\widetilde { \mathcal { O } } ( \tau d )$ , and the warm-up term is $\mathcal { O } ( w )$ . On the complement of the confidence event,

$$
\Re _ { T } ^ { \mathcal { Z } } = \sum _ { t = 1 } ^ { T } \langle \theta ^ { \star } , z ^ { \star } - x _ { t } \rangle \leq 2 L S T .
$$

Since the complement has probability at most $1 / T$ , its contribution to the expectation is at most 2LS. Therefore,

$$
\mathrm { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; \theta ^ { \star } ) = \mathbb { E } _ { \pi } \left[ \Re _ { T } ^ { \mathcal { Z } } \right] = \widetilde { \mathcal { O } } \left( d \sqrt { T } + \tau d + w \right) .
$$

It remains to compare the stationary benchmark with the ofline-optimal feasible trajectory. Writing

$$
T = q w + r , \qquad 0 \leq r < w ,
$$

Theorem 4.4 applied to

$$
f _ { \theta ^ { \star } } ( x ) = \langle \theta ^ { \star } , x \rangle
$$

gives

$$
0 \leq \Delta _ { T } : = V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) - T \langle \theta ^ { \star } , z ^ { \star } \rangle \leq r \left( \operatorname* { m a x } _ { x \in \mathcal { X } } \langle \theta ^ { \star } , x \rangle - \langle \theta ^ { \star } , z ^ { \star } \rangle \right) .
$$

Since

$$
\| x \| _ { 2 } \leq L , \qquad \| z ^ { \star } \| _ { 2 } \leq L , \qquad \| \theta ^ { \star } \| _ { 2 } \leq S ,
$$

we have

$$
\operatorname* { m a x } _ { x \in \mathcal { X } } \langle \theta ^ { \star } , x \rangle - \langle \theta ^ { \star } , z ^ { \star } \rangle \leq 2 L S .
$$

Hence

$$
0 \leq \Delta _ { T } \leq 2 L S r \leq 2 L S ( w - 1 ) .
$$

Finally,

$$
\begin{array} { r } { \mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \mathrm { R e g } _ { \pi } ^ { \mathcal { Z } } ( T ; \theta ^ { \star } ) + \Delta _ { T } . } \end{array}
$$

Combining the two bounds yields

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \widetilde { \mathcal { O } } \left( d \sqrt { T } + \tau d + w \right) .
$$

This completes the proof.

□

## H.3 A Lower Bound for the Transition Diameter

We show that the transition diameter can have a direct impact on regret. Even in a one-dimensional setting, and even if the learner is told the true reward parameter immediately after its first action, the sliding-window constraint can still force an additional regret of order τ.

Proposition H.7. For every integer $\tau \geq 2$ and every horizon $T \geq \tau _ { \ast }$ , there exists a one-dimensional linear-bandit instance satisfying Assumptions $\it 4 . 1 , \ 4 . 2 ,$ and ${ 5 . 3 } ,$ with transition diameter exactly $\tau ,$ such that every admissible policy π satisfies

$$
\operatorname* { s u p } _ { \theta ^ { \star } \in \{ - 1 , + 1 \} } \operatorname { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) \geq \frac { \tau } { 4 } .
$$

Proof. Fix an integer $\tau \geq 2 .$ . Let

$$
\mathcal { X } = [ - 1 , 1 ] , \qquad w = 2 , \qquad \rho : = \frac { 2 } { \tau } ,
$$

and define the window-feasibility set

$$
\begin{array} { r } { \mathcal { C } _ { \rho } : = \left\{ ( x _ { 1 } , x _ { 2 } ) \in [ - 1 , 1 ] ^ { 2 } : | x _ { 2 } - x _ { 1 } | \leq \rho \right\} . } \end{array}
$$

Thus, every admissible trajectory satisfies

$$
| x _ { t } - x _ { t - 1 } | \leq \rho , \qquad t = 2 , \ldots , T .
$$

The action space $\mathcal { X }$ is nonempty, compact, and convex. The set $\mathcal { C } _ { \rho }$ is nonempty, closed, and convex. It is also invariant under the cyclic shift $( x _ { 1 } , x _ { 2 } ) \mapsto ( x _ { 2 } , x _ { 1 } )$ , since $| x _ { 2 } - x _ { 1 } | = | x _ { 1 } - x _ { 2 } |$ . Thus, the required convexity and cyclic-invariance assumptions are satisfied. The stationary feasible set is $Z : = \{ z \in [ - 1 , 1 ] : ( z , z ) \in \mathcal { C } _ { \rho } \}$ Since $| z - z | = 0$ for every $z \in [ - 1 , 1 ] , Z = [ - 1 , 1 ]$ . Since $w = 2 ,$ , a stationary history consists of a single action, so $h ( z ) = z .$ For any $z , z ^ { \prime } \in Z$ , each feasible transition can change the action by at most $\rho .$ Therefore, any transition from $z \ \mathrm { t o } \ z ^ { \prime }$ using m actions must satisfy $| z ^ { \prime } - z | \leq m \rho$ , which implies

$$
m \geq \left\lceil { \frac { | z ^ { \prime } - z | } { \rho } } \right\rceil .
$$

This bound is also achievable by moving from $z \ \mathrm { t o } \ z ^ { \prime }$ in equally spaced steps of size at most $\rho .$ Hence,

$$
m ( z , z ^ { \prime } ) = \left\lceil \frac { | z ^ { \prime } - z | } { \rho } \right\rceil .
$$

Since $Z = [ - 1 , 1 ]$ has diameter 2 and $\rho = 2 / \tau$ 2

$$
\operatorname* { m a x } _ { z , z ^ { \prime } \in Z } m ( z , z ^ { \prime } ) = \left\lceil \frac 2 \rho \right\rceil = \left\lceil \frac 2 { 2 / \tau } \right\rceil = \tau .
$$

Thus Assumption 5.3 holds, and the transition diameter is exactly $\tau .$ Let $\theta ^ { \star } \in \{ - 1 , + 1 \}$ and consider the noiseless linear reward model $Y _ { t } = \theta ^ { \star } x _ { t }$ . This is a special case of the conditionally sub-Gaussian noise model. If $\theta ^ { \star } = + 1$ , the optimal stationary action is $z ^ { \star } = 1$ , while if $\theta ^ { \star } = - 1$ , it is $z ^ { \star } = - 1$ . Thus, in both cases,

$$
z ^ { \star } = \theta ^ { \star } \qquad \mathrm { a n d } \qquad \theta ^ { \star } z ^ { \star } = 1 .
$$

The constant trajectory $x _ { t } = \theta ^ { \star }$ is feasible for all t and achieves total reward $T .$ . Since the per-round reward cannot exceed 1, we have $V _ { T } ( \theta ^ { \star } ) = T$ . Therefore, the stationary and ofline benchmarks coincide on this instance:

$$
\operatorname { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \operatorname { R e g } _ { \pi } ^ { Z } ( T ; \theta ^ { \star } ) .
$$

To isolate the transition cost, suppose the learner is told the true value of $\theta ^ { \star }$ immediately after choosing its first action $x _ { 1 }$ . Hence, from round 2 onward, there is no statistical uncertainty. Fix $x _ { 1 } = a \in [ - 1 , 1 ]$ and define $u : = \theta ^ { \star } a$ . Since $\theta ^ { \star } \in \{ - 1 , + 1 \}$ and $| x _ { t + 1 } - x _ { t } | \leq \rho .$

$$
\theta ^ { \star } ( x _ { t + 1 } - x _ { t } ) \leq | x _ { t + 1 } - x _ { t } | \leq \rho .
$$

Therefore, after $k$ steps,

$$
\theta ^ { \star } x _ { 1 + k } = \theta ^ { \star } x _ { 1 } + \sum _ { j = 1 } ^ { k } \theta ^ { \star } ( x _ { j + 1 } - x _ { j } ) \leq u + k \rho .
$$

Since $\theta ^ { \star } x _ { t } \leq 1$ , we obtain

$$
\begin{array} { r } { \theta ^ { \star } x _ { 1 + k } \leq \operatorname* { m i n } \{ 1 , u + k \rho \} . } \end{array}
$$

Thus, the learner’s reward can improve by at most $\rho$ per round until it reaches the optimal value 1. Consequently,

$$
1 - \theta ^ { \star } x _ { 1 + k } \geq [ 1 - u - k \rho ] _ { + } ,
$$

where $[ y ] _ { + } : = \operatorname* { m a x } \{ y , 0 \}$ . Summing over the horizon gives

$$
R _ { T } ( \theta ^ { \star } ) : = \sum _ { t = 1 } ^ { T } \theta ^ { \star } ( z ^ { \star } - x _ { t } ) = \sum _ { t = 1 } ^ { T } ( 1 - \theta ^ { \star } x _ { t } ) \geq \sum _ { k = 0 } ^ { T - 1 } [ 1 - u - k \rho ] _ { + } .
$$

Since u $\geq - 1$ and $\rho = 2 / \tau .$ , we have

$$
\frac { 1 - u } { \rho } \leq \frac { 2 } { \rho } = \tau .
$$

Because $T \geq \tau$ , the regret bound gives

$$
R _ { T } ( \theta ^ { \star } ) \geq \sum _ { k = 0 } ^ { \tau - 1 } [ 1 - u - k \rho ] _ { + } .
$$

The function $s \mapsto [ 1 - u - \rho s ] _ { + }$ is nonnegative and nonincreasing, so the sum is lower bounded by its integral:

$$
R _ { T } ( \theta ^ { \star } ) \geq \int _ { 0 } ^ { \tau } [ 1 - u - \rho s ] _ { + } d s = \int _ { 0 } ^ { ( 1 - u ) / \rho } ( 1 - u - \rho s ) d s = \frac { ( 1 - u ) ^ { 2 } } { 2 \rho } .
$$

Hence,

$$
R _ { T } ( \theta ^ { \star } ) \geq \frac { ( 1 - u ) ^ { 2 } } { 2 \rho } .
$$

If $\theta ^ { \star } = + 1$ , then $u = a ,$ , while if $\theta ^ { \star } = - 1$ , then $u = - a$

$$
R _ { T } ( + 1 ) \geq \frac { ( 1 - a ) ^ { 2 } } { 2 \rho } , \qquad R _ { T } ( - 1 ) \geq \frac { ( 1 + a ) ^ { 2 } } { 2 \rho } .
$$

Averaging the two bounds gives

$$
\frac { R _ { T } ( + 1 ) + R _ { T } ( - 1 ) } { 2 } \geq \frac { ( 1 - a ) ^ { 2 } + ( 1 + a ) ^ { 2 } } { 4 \rho } = \frac { 1 + a ^ { 2 } } { 2 \rho } \geq \frac { 1 } { 2 \rho } .
$$

Since $\rho = 2 / \tau ,$

$$
\frac { 1 } { 2 \rho } = \frac { \tau } { 4 } .
$$

Therefore,

$$
\frac { R _ { T } ( + 1 ) + R _ { T } ( - 1 ) } { 2 } \geq \frac { \tau } { 4 } ,
$$

and hence at least one of the two reward parameters must satisfy

$$
\operatorname* { m a x } _ { \theta ^ { \star } \in \{ - 1 , + 1 \} } R _ { T } ( \theta ^ { \star } ) \geq \frac { \tau } { 4 } .
$$

Finally, suppose the policy randomizes its first action $X _ { 1 }$ . Since $X _ { 1 }$ is chosen before any reward is observed, its distribution is independent of $\theta ^ { \star }$ . Conditioning on any realization $X _ { 1 } = a$ gives

$$
\frac { 1 } { 2 } \mathbb { E } [ R _ { T } ( + 1 ) \mid X _ { 1 } = a ] + \frac { 1 } { 2 } \mathbb { E } [ R _ { T } ( - 1 ) \mid X _ { 1 } = a ] \geq \frac { \tau } { 4 } .
$$

Taking expectation over $X _ { 1 }$

$$
\frac { 1 } { 2 } \mathbb { E } [ R _ { T } ( + 1 ) ] + \frac { 1 } { 2 } \mathbb { E } [ R _ { T } ( - 1 ) ] \geq \frac { \tau } { 4 } .
$$

Hence at least one of the two reward parameters satisfies

$$
\operatorname* { s u p } _ { \theta ^ { \star } \in \{ - 1 , + 1 \} } \operatorname { R e g } _ { \pi } ^ { Z } ( T ; \theta ^ { \star } ) \geq \frac { \tau } { 4 } .
$$

Since the stationary and ofline benchmarks coincide on this instance,

$$
\operatorname* { s u p } _ { \theta ^ { \star } \in \{ - 1 , + 1 \} } \operatorname { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) \geq \frac { \tau } { 4 } .
$$

This completes the proof.

Remark H.8 (Statistical, transition, and finite-horizon costs). Proposition H.7 isolates the cost of feasible transitions from the statistical dificulty of learning: the $\Omega ( \tau )$ lower bound holds even when the learner is told the true reward parameter immediately after its first action. Standard stochastic linear-bandit lower bounds give a statistical cost of order $d { \sqrt { T } } ;$ see Lattimore and Szepesv´ari (2020, Theorem 24.2), while Proposition G.3 shows that the finite-horizon gap can be $\Omega ( w )$ . Taken together, these results identify the lower-bound scales $\Omega \Big ( d \sqrt { T } + \tau + w \Big )$ . Here, the $\Omega ( w )$ term comes from a separate finite-horizon construction, so this expression summarizes the three individual lower bounds rather than a single joint minimax result. The remaining gap is in the transition term: our upper bound is $\widetilde { \mathcal { O } } ( \tau d )$ , whereas the lower bound is $\Omega ( \tau )$ . Whether the additiona factor of d is necessary remains open.

## I PROOFS FOR SECTION 6: PLANNING

## I.1 Proof of Theorem 6.1

Lemma I.1. Let $s _ { t } = ( x _ { t - w + 1 } , \ldots , x _ { t - 1 } )$ , Then a sequence $x _ { 1 : T }$ is feasible under the original sliding-window constraints $i f$ and only $i f$

$$
x _ { t } \in A ( s _ { t } ) , \qquad t = w , \dotsc , T .
$$

Proof. Fix $t \in \{ w , \ldots , T \}$ . By definition of the state-dependent feasible set,

$$
x _ { t } \in A ( s _ { t } ) \quad \Longleftrightarrow \quad ( s _ { t } , x _ { t } ) \in { \mathcal C } .
$$

Since $s _ { t } = ( x _ { t - w + 1 } , \dots , x _ { t - 1 } )$ , we have $( s _ { t } , x _ { t } ) = ( x _ { t - w + 1 } , \ldots , x _ { t } )$ . Hence

$$
x _ { t } \in A ( s _ { t } ) \quad \Longleftrightarrow \quad ( x _ { t - w + 1 } , \ldots , x _ { t } ) \in { \mathcal C } .
$$

Because this equivalence holds for every $t = w , \ldots , T$ , the state-dependent feasibility conditions are equivalent to the original sliding-window constraints. □

Proof of Theorem 6.1. We prove the result by induction on the remaining horizon n. For $n = 0 ,$ no further actions remain to be selected. Hence, the cumulative reward over the remaining horizon is zero, so $J _ { 0 } ^ { \star } ( s ) = 0$ Now assume the claim holds for horizon $n - 1$ , and consider an n-step viable history $s \in S _ { n }$ . Any feasible n-step continuation from s must first choose an action x satisfying $x \in A ( s )$ . After taking this action, the new history is $s ^ { \prime } = F ( s , x )$ , and feasibility of the remaining $n - 1$ steps requires $s ^ { \prime } \in S _ { n - 1 }$ . By the induction hypothesis, the maximum reward achievable from $s ^ { \prime }$ over the remaining $n - 1$ steps is $J _ { n - 1 } ^ { \star } ( s ^ { \prime } )$ . Therefore, any feasible n-step continuation from s has reward at most

$$
\langle \theta ^ { \star } , x \rangle + J _ { n - 1 } ^ { \star } ( F ( s , x ) ) .
$$

Taking the maximum over all feasible first actions gives

$$
\mathrm { o p t i m a l ~ r e w a r d ~ f r o m } \ s \leq \operatorname* { m a x } _ { \stackrel { x \in A ( s ) } { F ( s , x ) \in S _ { n - 1 } } } \left\{ \langle \theta ^ { \star } , x \rangle + J _ { n - 1 } ^ { \star } ( F ( s , x ) ) \right\} = J _ { n } ^ { \star } ( s ) .
$$

It remains to show that this upper bound is achievable. Let $x ^ { \star }$ be an action attaining the maximum in the Bellman recursion. Since $F ( s , x ^ { \star } ) \in S _ { n - 1 }$ , the induction hypothesis guarantees a feasible optimal continuation from $F ( s , x ^ { \star } )$ with reward $J _ { n - 1 } ^ { \star } ( F ( s , x ^ { \star } ) )$ . Combining this continuation with the first action $x ^ { \star }$ gives a feasible n-step trajectory from s with total reward

$$
\begin{array} { r } { \langle \theta ^ { \star } , x ^ { \star } \rangle + J _ { n - 1 } ^ { \star } \big ( F ( s , x ^ { \star } ) \big ) = J _ { n } ^ { \star } ( s ) . } \end{array}
$$

Hence $J _ { n } ^ { \star } ( s )$ equals the optimal reward over all feasible n-step continuations from s. Finally, by Lemma I.1, finite-memory feasibility is equivalent to the original sliding-window constraints. Therefore, the optimal value of the original problem is

$$
V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) = \operatorname* { m a x } _ { s = ( x _ { 1 } , \dots , x _ { w - 1 } ) \in S _ { T - w + 1 } } \left\{ \sum _ { i = 1 } ^ { w - 1 } \langle \theta ^ { \star } , x _ { i } \rangle + J _ { T - w + 1 } ^ { \star } ( s ) \right\} ,
$$

which completes the proof.

## I.2 Proof of Lemma 6.3

Proof. Fix $n \geq 1$ and $s , s ^ { \prime } \in \mathcal S _ { n }$ , and let $R : = \arg ( g )$ . Subtracting $g _ { \mathrm { m i n } }$ from g shifts every n-step value by the same amount $n g _ { \mathrm { m i n } }$ , so it does not change diferences between continuation values. Thus, without loss of generality, assume

$$
0 \leq g ( x ) \leq R \qquad { \mathrm { f o r ~ a l l ~ } } x \in { \mathcal { X } } .
$$

By Assumption 6.2, there is a feasible transition from s to $s ^ { \prime }$ of length $m \leq D$ . Since $s ^ { \prime } \in S _ { n }$ , this transition can be followed by a feasible n-step continuation from $s ^ { \prime } .$ . Hence s admits a feasible continuation of length $n + m$ For any $\varepsilon > 0$ , choose a feasible n-step continuation from $s ^ { \prime }$ whose reward is at least ${ \cal J } _ { n } ^ { g } ( s ^ { \prime } ) - \varepsilon .$ . Prepending the transition from s to $s ^ { \prime }$ gives a feasible (n+m)-step continuation from s. Since all shifted rewards are nonnegative,

$$
J _ { n + m } ^ { g } ( s ) \geq J _ { n } ^ { g } ( s ^ { \prime } ) - \varepsilon .
$$

Now consider any feasible $( n + m )$ -step continuation from s. Its first n actions form a feasible n-step continuation from s, and therefore contribute at most $J _ { n } ^ { g } ( s )$ . The remaining m actions contribute at most mR. Thus,

$$
J _ { n + m } ^ { g } ( s ) \leq J _ { n } ^ { g } ( s ) + m R .
$$

Combining the two bounds gives

$$
J _ { n } ^ { g } ( s ^ { \prime } ) - \varepsilon \leq J _ { n } ^ { g } ( s ) + m R .
$$

Letting $\varepsilon \downarrow 0$ and using $m \leq D$

$$
J _ { n } ^ { g } ( s ^ { \prime } ) - J _ { n } ^ { g } ( s ) \leq D R .
$$

Applying the same argument to a feasible transition from $s ^ { \prime }$ to $s ,$ whose length is also at most $D ,$ yields

$$
J _ { n } ^ { g } ( s ) - J _ { n } ^ { g } ( s ^ { \prime } ) \leq D R .
$$

Therefore,

$$
| J _ { n } ^ { g } ( s ) - J _ { n } ^ { g } ( s ^ { \prime } ) | \leq D R = D \operatorname { r n g } ( g ) .
$$

Since $s , s ^ { \prime } \in S _ { n }$ were arbitrary,

$$
\begin{array} { r } { \operatorname { s p a n } _ { \mathcal { S } _ { n } } ( J _ { n } ^ { g } ) \leq D \operatorname { r n g } ( g ) , } \end{array}
$$

which proves the claim.

## I.3 Proof of Theorem 6.5

For episode k, let

$$
\begin{array} { r } { \mathcal { T } _ { k } : = \{ t _ { k } , \dotsc , t _ { k + 1 } - 1 \} , \qquad \ell _ { k } : = | \mathcal { T } _ { k } | , \qquad n _ { k } : = T - t _ { k } + 1 , } \end{array}
$$

with $t _ { K + 1 } : = T + 1$ . Then $n _ { k + 1 } = n _ { k } - \ell _ { k }$

Lemma I.2. Under Assumption 6.2 and the uniform confidence event, every nonterminal episode k satisfies

$$
J _ { n _ { k } } ^ { \star } \big ( s _ { t _ { k } } \big ) \leq \sum _ { t \in \mathcal { T } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { n _ { k + 1 } } ^ { \star } \big ( s _ { t _ { k + 1 } } \big ) + 4 L S D .
$$

For the terminal episode,

$$
J _ { n _ { K } } ^ { \star } ( s _ { t _ { K } } ) \leq \sum _ { t \in \mathbb { Z } _ { K } } \overline { { U } } _ { K } ( x _ { t } ) .
$$

Proof. Fix a nonterminal episode k and write m $: = n _ { k + 1 } = n _ { k } - \ell _ { k }$ . Let $s _ { k } ^ { \star } \in { S } _ { m }$ be the history reached after $\ell _ { k }$ steps along an optimal n -step continuation from $s _ { t _ { k } }$ . On the uniform confidence event, $\overline { { U } } _ { k } ( x ) \geq \langle \theta ^ { \star } , x \rangle$ , and therefore

$$
J _ { n _ { k } } ^ { \star } \bigl ( s _ { t _ { k } } \bigr ) \leq \sum _ { j = 0 } ^ { \ell _ { k } - 1 } \overline { { U } } _ { k } \bigl ( x _ { t _ { k } + j } ^ { \star } \bigr ) + J _ { m } ^ { \star } \bigl ( s _ { k } ^ { \star } \bigr ) .
$$

Since the learner follows an exact Bellman-optimal policy for $\overline { { U } } _ { k }$

$$
J _ { k , n _ { k } } \big ( s _ { t _ { k } } \big ) = \sum _ { t \in \mathbb { Z } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { k , m } \big ( s _ { t _ { k + 1 } } \big ) ,
$$

and Bellman optimality also gives

$$
\sum _ { j = 0 } ^ { \ell _ { k } - 1 } \overline { { U } } _ { k } ( x _ { t _ { k } + j } ^ { \star } ) + J _ { k , m } ( s _ { k } ^ { \star } ) \leq J _ { k , n _ { k } } ( s _ { t _ { k } } ) .
$$

Combining these inequalities yields

$$
J _ { n _ { k } } ^ { \star } ( s _ { t _ { k } } ) \leq \sum _ { t \in \mathcal { T } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { m } ^ { \star } ( s _ { t _ { k + 1 } } ) + \big ( J _ { m } ^ { \star } - J _ { k , m } \big ) ( s _ { k } ^ { \star } ) - \big ( J _ { m } ^ { \star } - J _ { k , m } \big ) ( s _ { t _ { k + 1 } } ) .
$$

Both histories belong to $S _ { m }$ . By Lemma 6.3, $\mathrm { s p a n } _ { S _ { m } } ( J _ { m } ^ { \star } ) \leq 2 L S D$ . Since $- L S \leq \overline { { U } } _ { k } ( x ) \leq L S$ , the same lemma gives $\begin{array} { r } { \operatorname { s p a n } _ { \boldsymbol { S } _ { m } } ( J _ { k , m } ) \leq 2 L S D } \end{array}$ . Hence

$$
\begin{array} { r } { \operatorname { s p a n } _ { S _ { m } } \bigl ( J _ { m } ^ { \star } - J _ { k , m } \bigr ) \leq 4 L S D , } \end{array}
$$

which proves

$$
J _ { n _ { k } } ^ { \star } \big ( s _ { t _ { k } } \big ) \leq \sum _ { t \in \mathcal { T } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { n _ { k + 1 } } ^ { \star } \big ( s _ { t _ { k + 1 } } \big ) + 4 L S D .
$$

For the terminal episode, $n _ { K + 1 } = 0$ . Since the optimistic policy is followed to the end of the horizon,

$$
J _ { K , n _ { K } } \big ( s _ { t _ { K } } \big ) = \sum _ { t \in \mathbb { Z } _ { K } } \overline { { U } } _ { K } ( x _ { t } ) .
$$

Optimism gives $J _ { n _ { K } } ^ { \star } \bigl ( s _ { t _ { K } } \bigr ) \leq J _ { K , n _ { K } } \bigl ( s _ { t _ { K } } \bigr )$ , which proves the terminal bound.

Proof of Theorem 6.5. Let $r ^ { \star } ( x ) : = \langle \theta ^ { \star } , x \rangle$ and define the pathwise pseudo-regret

$$
\mathfrak { R } _ { T } ^ { \mathrm { o f f } } : = V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) - \sum _ { t = 1 } ^ { T } r ^ { \star } ( x _ { t } ) .
$$

For any $\delta \in ( 0 , 1 )$ , define the uniform confidence event

$$
\mathcal G _ { \delta } : = \left\{ \| \widehat \theta _ { t } - \theta ^ { \star } \| _ { V _ { t } } \leq \beta _ { t } ( \delta ) \mathrm { ~ f o r ~ a l l ~ } t = 1 , \ldots , T + 1 \right\} .
$$

By the confidence-ellipsoid result (Abbasi-Yadkori et al., 2011, Theorem 2), which is based on a self-normalized concentration inequality for vector-valued martingales,

$$
\mathbb { P } ( \mathcal { G } _ { \delta } ) \ge 1 - \delta .
$$

We first work on $\mathcal { G } _ { \delta }$ . For $w > 1$ , let $n _ { 1 } = T - w + 1$ . If $s _ { w } ^ { \star }$ is the complete history obtained from the first $w - 1$ actions of an ofline-optimal trajectory, then

$$
V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) = \sum _ { t = 1 } ^ { w - 1 } r ^ { \star } ( x _ { t } ^ { \star } ) + J _ { n _ { 1 } } ^ { \star } ( s _ { w } ^ { \star } ) .
$$

Since $| r ^ { \star } ( x ) | \leq L S$

$$
\sum _ { t = 1 } ^ { w - 1 } \bigl ( r ^ { \star } ( x _ { t } ^ { \star } ) - r ^ { \star } ( x _ { t } ) \bigr ) \leq 2 L S ( w - 1 ) .
$$

Moreover, $s _ { w } ^ { \star } , s _ { w } \in S _ { n _ { 1 } }$ , so Lemma 6.3 gives

$$
J _ { n _ { 1 } } ^ { \star } \big ( s _ { w } ^ { \star } \big ) - J _ { n _ { 1 } } ^ { \star } \big ( s _ { w } \big ) \leq 2 L S D .
$$

Therefore,

$$
\Re _ { T } ^ { \mathrm { o f f } } \leq 2 L S ( w - 1 ) + 2 L S D + J _ { n _ { 1 } } ^ { \star } ( s _ { w } ) - \sum _ { t = w } ^ { T } r ^ { \star } ( x _ { t } ) .
$$

For $w = 1$ , the same inequality holds without the initialization terms. Applying Lemma I.2 over all episodes and telescoping the continuation values gives

$$
J _ { n _ { 1 } } ^ { \star } \big ( s _ { t _ { 1 } } \big ) \leq \sum _ { k = 1 } ^ { K } \sum _ { t \in \mathbb { Z } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + 4 L S D ( K - 1 ) .
$$

Hence

$$
\Re _ { T } ^ { \mathrm { o f f } } \leq 2 L S ( w - 1 ) + 4 L S D K + \sum _ { k = 1 } ^ { K } \sum _ { t \in \mathcal { T } _ { k } } \left( \overline { { U } } _ { k } ( x _ { t } ) - r ^ { \star } ( x _ { t } ) \right) .
$$

On G<sub>δ</sub>, $\left| \langle \widehat { \theta } _ { k } - \theta ^ { \star } , x _ { t } \rangle \right| \leq \beta _ { k } \| x _ { t } \| _ { W _ { k } ^ { - 1 } }$ , so clipping preserves optimism and

$$
0 \leq \overline { { U } } _ { k } ( x _ { t } ) - r ^ { \star } ( x _ { t } ) \leq 2 \beta _ { k } \| x _ { t } \| _ { W _ { k } ^ { - 1 } } .
$$

Moreover, throughout episode $k ,$

$$
W _ { k } \preceq V _ { t } , \qquad \operatorname* { d e t } ( V _ { t } ) \leq 2 \operatorname* { d e t } ( W _ { k } ) .
$$

Applying Lemma H.2 with $A = W _ { k } , B = V _ { t }$ , and $c = 2$ , we obtain

$$
W _ { k } ^ { - 1 } \preceq 2 V _ { t } ^ { - 1 } .
$$

Consequently,

$$
\| x _ { t } \| _ { W _ { k } ^ { - 1 } } \leq \sqrt { 2 } \| x _ { t } \| _ { V _ { t } ^ { - 1 } } .
$$

Using $\beta _ { k } \le \beta _ { T } ( \delta )$

$$
\sum _ { k = 1 } ^ { K } \sum _ { t \in \mathcal { T } _ { k } } \left( \overline { { U } } _ { k } ( x _ { t } ) - r ^ { \star } ( x _ { t } ) \right) \leq 2 \sqrt { 2 } \beta _ { T } ( \delta ) \sum _ { t = 1 } ^ { T } \| x _ { t } \| _ { V _ { t } ^ { - 1 } } .
$$

Since $\lambda \geq L ^ { 2 }$ , applying Lemma H.3 with $\boldsymbol { u } _ { t } = \boldsymbol { x } _ { t }$ and $N = T$ gives

$$
\sum _ { t = 1 } ^ { T } \left\| x _ { t } \right\| _ { V _ { t } ^ { - 1 } } \leq \sqrt { 2 T d \log \biggl ( 1 + \frac { T L ^ { 2 } } { \lambda d } \biggr ) } .
$$

Thus,

$$
\sum _ { k = 1 } ^ { K } \sum _ { t \in \mathcal { T } _ { k } } \left( \overline { { U } } _ { k } ( x _ { t } ) - r ^ { \star } ( x _ { t } ) \right) \leq 4 \beta _ { T } ( \delta ) \sqrt { T d \log \biggl ( 1 + \frac { T L ^ { 2 } } { \lambda d } \biggr ) } .
$$

It remains to bound the number of episodes. Since K is the number of determinant-doubling episodes, Lemma H.4 gives

$$
K \leq 1 + \left\lceil d \log _ { 2 } \left( 1 + { \frac { T L ^ { 2 } } { \lambda d } } \right) \right\rceil .
$$

Combining the preceding bounds, on $\mathcal { G } _ { \delta }$

$$
\mathfrak { R } _ { T } ^ { \mathrm { o f f } } \le 4 \beta _ { T } ( \delta ) \sqrt { T d \log \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) } + 2 L S ( w - 1 ) + 4 L S D \left[ 1 + \left[ d \log _ { 2 } \left( 1 + \frac { T L ^ { 2 } } { \lambda d } \right) \right] \right] .
$$

Thus,

$$
\Re _ { T } ^ { \mathrm { o f f } } = \widetilde { \mathcal { O } } \left( d \sqrt { T } + d D + w \right)
$$

with probability at least $1 - \delta$ . Finally, choose $\delta = 1 / T$ . Since both the ofline and realized cumulative rewards lie in $[ - L S T , L S T ] , \Re _ { T } ^ { \mathrm { o f f } } \leq 2 L S T$ on every sample path. Therefore,

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \mathbb { E } \big [ \mathfrak { R } _ { T } ^ { \mathrm { o f f } } \big ] \leq \widetilde { \mathcal { O } } \left( d \sqrt { T } + d D + w \right) + 2 L S T \mathbb { P } ( \mathcal { G } _ { 1 / T } ^ { c } ) = \widetilde { \mathcal { O } } \left( d \sqrt { T } + d D + w \right) ,
$$

which proves the theorem.

## I.4 Extension to Approximate Remaining-Horizon Planning

The analysis in Section 6 assumes access to an exact Bellman-optimal remaining-horizon planner. We show that this assumption can be relaxed to additive approximate planning, at the cost of an explicit cumulative planning-error term.

Assumption I.3. At each planning time $t _ { k }$ , the learner can compute a feasible remaining-horizon policy $\pi _ { k }$ satisfying

$$
J _ { k , n _ { k } } ^ { \pi _ { k } } \bigl ( s _ { t _ { k } } \bigr ) \geq J _ { k , n _ { k } } \bigl ( s _ { t _ { k } } \bigr ) - \alpha _ { k } , \qquad 0 \leq \alpha _ { k } \leq \alpha _ { \operatorname* { m a x } } .
$$

When $w > 1$ , the learner also has access to a feasible initialization.

Since $J _ { k , n _ { k } } ( s _ { t _ { k } } )$ is Bellman optimal, setting $\alpha _ { k } = 0$ recovers exact planning.

Lemma I.4. Under Assumption 6.2, the uniform confidence event, and Assumption I.3, every nonterminal episode k satisfies

$$
J _ { n _ { k } } ^ { \star } \big ( s _ { t _ { k } } \big ) \leq \sum _ { t \in \mathcal { T } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { n _ { k + 1 } } ^ { \star } \big ( s _ { t _ { k + 1 } } \big ) + 4 L S D + \alpha _ { k } .
$$

For the terminal episode,

$$
J _ { n _ { K } } ^ { \star } ( s _ { t _ { K } } ) \leq \sum _ { t \in \mathbb { Z } _ { K } } \overline { { U } } _ { K } ( x _ { t } ) + \alpha _ { K } .
$$

Proof. The proof follows that of Lemma I.2, with the exact Bellman identity replaced by the additive approximation guarantee. Fix a nonterminal episode k and let $m : = n _ { k + 1 } = n _ { k } - \ell _ { k }$ . By Assumption I.3,

$$
J _ { k , n _ { k } } ( s _ { t _ { k } } ) \leq J _ { k , n _ { k } } ^ { \pi _ { k } } ( s _ { t _ { k } } ) + \alpha _ { k } .
$$

Since the learner follows $\pi _ { k }$ during episode $k ,$

$$
J _ { k , n _ { k } } ^ { \pi _ { k } } \bigl ( s _ { t _ { k } } \bigr ) = \sum _ { t \in \mathbb { Z } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { k , m } ^ { \pi _ { k } } \bigl ( s _ { t _ { k + 1 } } \bigr ) ,
$$

and optimality of $J _ { k , m }$ gives

$$
J _ { k , m } ^ { \pi _ { k } } \bigl ( s _ { t _ { k + 1 } } \bigr ) \leq J _ { k , m } \bigl ( s _ { t _ { k + 1 } } \bigr ) .
$$

Hence

$$
J _ { k , n _ { k } } ( s _ { t _ { k } } ) \leq \sum _ { t \in \mathbb { Z } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { k , m } ( s _ { t _ { k + 1 } } ) + \alpha _ { k } .
$$

Substituting this inequality into the same optimism and Bellman comparison used in the proof of Lemma I.2 yields

$$
J _ { n _ { k } } ^ { \star } ( s _ { t _ { k } } ) \leq \sum _ { t \in \mathbb { Z } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { m } ^ { \star } ( s _ { t _ { k + 1 } } ) + \big ( J _ { m } ^ { \star } - J _ { k , m } \big ) ( s _ { k } ^ { \star } ) - \big ( J _ { m } ^ { \star } - J _ { k , m } \big ) ( s _ { t _ { k + 1 } } ) + \alpha _ { k } .
$$

As before, $s _ { k } ^ { \star } , s _ { t _ { k + 1 } } \in S _ { m }$ , and Lemma 6.3 gives

$$
\begin{array} { r } { \mathrm { s p a n } _ { S _ { m } } ( J _ { m } ^ { \star } ) \leq 2 L S D , \qquad \mathrm { s p a n } _ { S _ { m } } ( J _ { k , m } ) \leq 2 L S D . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { \operatorname { s p a n } _ { \boldsymbol { S } _ { m } } \bigl ( \boldsymbol { J } _ { m } ^ { \star } - \boldsymbol { J } _ { k , m } \bigr ) \leq 4 L S D , } \end{array}
$$

and thus

$$
J _ { n _ { k } } ^ { \star } \big ( s _ { t _ { k } } \big ) \leq \sum _ { t \in \mathcal { T } _ { k } } \overline { { U } } _ { k } ( x _ { t } ) + J _ { n _ { k + 1 } } ^ { \star } \big ( s _ { t _ { k + 1 } } \big ) + 4 L S D + \alpha _ { k } .
$$

For the terminal episode,

$$
J _ { n _ { K } } ^ { \star } \bigl ( s _ { t _ { K } } \bigr ) \leq J _ { K , n _ { K } } \bigl ( s _ { t _ { K } } \bigr ) \leq J _ { K , n _ { K } } ^ { \pi _ { K } } \bigl ( s _ { t _ { K } } \bigr ) + \alpha _ { K } = \sum _ { t \in \mathcal { T } _ { K } } \overline { { U } } _ { K } \bigl ( x _ { t } \bigr ) + \alpha _ { K } ,
$$

which proves the claim.

Theorem I.5. Suppose the online linear-bandit assumptions of Section 3 hold, with $\| x \| _ { 2 } \leq L , \| \theta ^ { \star } \| _ { 2 } \leq S$ , and $\lambda \geq L ^ { 2 }$ . Under Assumptions 6.2 and I.3, the rare-update optimistic planning policy satisfies

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \widetilde { \mathcal { O } } \left( d \sqrt { T } + d D + w + d \alpha _ { \mathrm { m a x } } \right) .
$$

Proof. Applying Lemma I.4 over all episodes and telescoping the continuation values yields

$$
J _ { n _ { 1 } } ^ { \star } \big ( s _ { t _ { 1 } } \big ) \leq \sum _ { k = 1 } ^ { K } \sum _ { t \in \mathcal { T } _ { k } } \overline { { U } } _ { k } \big ( x _ { t } \big ) + 4 L S D ( K - 1 ) + \sum _ { k = 1 } ^ { K } \alpha _ { k } .
$$

Combining this with the initialization bound from the proof of Theorem 6.5, on the uniform confidence event,

$$
\Re _ { T } ^ { \mathrm { o f f } } \leq 2 L S ( w - 1 ) + 4 L S D K + \sum _ { k = 1 } ^ { K } \sum _ { t \in \overline { { L } } _ { k } } \left( \overline { { U } } _ { k } ( x _ { t } ) - r ^ { \star } ( x _ { t } ) \right) + \sum _ { k = 1 } ^ { K } \alpha _ { k } .
$$

The confidence-width and elliptical-potential bounds give

$$
\sum _ { k = 1 } ^ { K } \sum _ { t \in \mathcal { T } _ { k } } \left( \overline { { U } } _ { k } ( x _ { t } ) - r ^ { \star } ( x _ { t } ) \right) = \widetilde { \mathcal { O } } ( d \sqrt { T } ) .
$$

## Offline Geometry and Stationary Approximation Gap

![](images/de3576704fd0a025066d5676f8edfc5a56950f5030fd3720bd7e27cda0b8d243.jpg)

![](images/f42dfa381b994a9b461c448f40c9bcb62ebe32fc65cc3a40386bc05aa55c44ad.jpg)  
Figure 1: Ofline geometry and horizon misalignment. Left: stationary approximation gap versus horizon remainder for $w = 3 2$ . Right: maximum gap versus window length; markers show the LP solutions and the dashed line shows the tight value $w / 4$

Moreover, Lemma H.4 gives $K = { \widetilde { \mathcal { O } } } ( d )$ . Hence

$$
4 L S D K = \widetilde { \mathcal { O } } ( d D ) , \qquad \sum _ { k = 1 } ^ { K } \alpha _ { k } \le K \alpha _ { \operatorname* { m a x } } = \widetilde { \mathcal { O } } ( d \alpha _ { \operatorname* { m a x } } ) .
$$

Therefore,

$$
\Re _ { T } ^ { \mathrm { o f f } } = \widetilde { \mathcal { O } } \left( d \sqrt { T } + d D + w + d \alpha _ { \mathrm { { m a x } } } \right)
$$

with probability at least $1 - \delta .$ Choosing $\delta = 1 / T$ and proceeding as in the final step of Theorem 6.5 gives

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { o f f } } ( T ; \theta ^ { \star } ) = \widetilde { \mathcal { O } } \left( d \sqrt { T } + d D + w + d \alpha _ { \mathrm { m a x } } \right) .
$$

Remark I.6. The efect of approximate planning enters only through the cumulative planning error. Since $K = { \widetilde { \mathcal { O } } } ( d )$

$$
\sum _ { k = 1 } ^ { K } \alpha _ { k } \le K \alpha _ { \operatorname* { m a x } } = \widetilde { \mathcal { O } } ( d \alpha _ { \operatorname* { m a x } } ) .
$$

Thus, bounded per-episode planning error degrades the regret gracefully. In particular, if $\alpha _ { \mathrm { m a x } } = \mathcal { O } ( 1 )$ , the term $d \alpha _ { \mathrm { m a x } }$ is absorbed by $d \sqrt { T }$ , so the approximate planner achieves the same asymptotic regret rate as the exact planner. Setting $\alpha _ { \mathrm { m a x } } = 0$ recovers exact planning.

## J SYNTHETIC EXPERIMENTS

In this section, we conduct three experiments to evaluate our theoretical results. Specifically, we ask: (Q1) Does the stationary reduction recover the exact ofline optimum when the horizon aligns with the window length, and is the resulting $\mathcal { O } ( w )$ gap tight when the horizon is misaligned? (Q2) Does rare switching reduce the feasibility cost of moving between stationary actions when direct switches are not possible? (Q3) Beyond cyclic symmetry, can finite-memory optimistic planning efectively exploit recent history while preserving exact sliding-window feasibility? Additional ablation and sensitivity studies are provided in Appendix K.2.

## J.1 Ofline Geometry and Horizon Misalignment

We address Q1 by testing the stationary-reduction result of Theorem 4.4. We consider

$$
\mathcal X = [ 0 , 1 ] , \qquad \mathcal C _ { w } = \left\{ ( x _ { 1 } , \dots , x _ { w } ) \in [ 0 , 1 ] ^ { w } : \sum _ { i = 1 } ^ { w } x _ { i } \le \frac { w } { 2 } \right\} ,
$$

![](images/cc2a24cb23f2bb6345770a6e3ed4e9d8d0756a609af6757c39e6aa9a5ea08f41.jpg)

![](images/83bf7540ae14d72afa2809cb3538337b0a3d1221a8d071bc499ead8ed8b19303.jpg)

![](images/9b5e16156cb9acc685e3cc0ea5f3125b2c0c07c1d7b80c2610ee75bfe743f406.jpg)

![](images/b5cd4adafd0d0f79ba946dbd9b66467c1d9219d711e1a942c7a7a3a67fbf9dc1.jpg)  
Figure 2: Rare switching under cyclic ramping constraints. (a) Pseudo-regret, (b) stationary-target switches, (c) transition rounds, and (d) constraint violations for Rare-switching OFUL, Frequent-update feasible OFUL, and Unconstrained OFUL across $\rho .$ Curves show means over 30 runs with 95% confidence intervals.

with linear reward $f ( x ) = x .$ . The stationary feasible set is $\mathcal { Z } = [ 0 , 1 / 2 ]$ , so the optimal stationary action is $z ^ { \star } = 1 / 2$ . We use even window lengths $w \in \{ 4 , 6 , 8 , \ldots , 3 2 \}$ and horizons

$$
T = 2 0 w + r , \qquad r \in \{ 0 , \ldots , w - 1 \} .
$$

For each $( w , r )$ , we compute the exact ofline optimum $V _ { T }$ by linear programming and measure the stationary approximation gap $\Delta _ { T } = V _ { T } - T f ( z ^ { \star } )$ . Figure 1 summarizes the results. In the left panel, we fix $w = 3 2$ and vary the remainder $r = T$ mod w. The gap is zero when $r = 0$ , increases as the horizon becomes misaligned with the window, reaches its maximum at $r = w / 2$ , and then decreases symmetrically as r approaches w. The right panel shows the maximum gap over r for diferent window lengths. For every tested even $w ,$ the LP solution gives max<sub>r</sub> $\Delta _ { T } = { \frac { w } { 4 } }$ , with the maximum attained at $r = w / 2$ . This exactly matches the tightness construction in Appendix G.2, where $\begin{array} { r } { V _ { T } - T f ( z ^ { \star } ) = \frac { w } { 4 } } \end{array}$ for horizons of the form $T = q w + w / 2$

## J.2 Rare Switching under Cyclic Ramping Constraints

We address Q2 by examining the cost of maintaining feasibility when transitions between stationary actions cannot be made instantaneously. We consider the simplex action space $\mathcal { X } = \Delta _ { d }$ with $d = 8$ and cyclic ramping constraints

$$
\mathcal { C } _ { \rho } = \left\{ \left( x _ { 1 } , \ldots , x _ { w } \right) \in \mathcal { X } ^ { w } : \operatorname* { m a x } _ { 1 \leq i \leq w } \| x _ { i + 1 } - x _ { i } \| _ { 1 } \leq \rho , \ x _ { w + 1 } = x _ { 1 } \right\} ,
$$

with window length $\begin{array} { r l r } { w } & { { } = } & { 8 . } \end{array}$ Every stationary action is feasible, so $\begin{array} { r l r } { \mathcal { Z } } & { { } = } & { \Delta _ { d } . } \end{array}$ We use $\theta ^ { \star } =$ $( 0 . 5 5 , 0 . 6 0 , 0 . 6 5 , 0 . 7 0 , 0 . 7 5 , 0 . 8 0 , 0 . 9 0 , 1 . 0 0 ) ^ { \top }$ and observations

$$
Y _ { t } = \langle \theta ^ { \star } , x _ { t } \rangle + \eta _ { t } , \qquad \eta _ { t } \sim \mathcal { N } ( 0 , 0 . 1 ^ { 2 } ) .
$$

We fix $T = 1 6 3 8 4$ and vary the ramping radius over $\rho \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 \}$ . Since $w \mid T$ , the stationary and ofline benchmarks coincide. We compare three policies. Rare-switching $O F U L$ follows Algorithm 1 and updates its optimistic stationary target only when the determinant-based episode condition is triggered. Frequent-update feasible $O F U L$ uses the same confidence sets, optimistic-action oracle, and transition mechanism, but recomputes its target after every retained stationary observation. Unconstrained OFUL (Abbasi-Yadkori et al., 2011) ignores the ramping constraint and switches directly to each newly selected optimistic action. For each policy, we report the realized pseudo-regret

![](images/d9724353122c197947ccb3d949afa9b569abd1739074745defb2a8afc4953764.jpg)  
Figure 3: Planning beyond cyclic symmetry. Left: normalized pseudo-regret relative to the exact ofline optimum. Right: number of policy updates. Curves show means over 30 runs with 95% confidence intervals.

$$
\widehat { R } _ { T } = \sum _ { t = 1 } ^ { T } \langle \theta ^ { \star } , z ^ { \star } - x _ { t } \rangle ,
$$

the number of stationary-target switches $J _ { T }$ , the total number of transition rounds, and the number of sliding window violations

$$
N _ { \mathrm { v i o l } } = \sum _ { t = w } ^ { T } \mathbf { 1 } \left\{ \left( x _ { t - w + 1 } , \ldots , x _ { t } \right) \notin \mathcal { C } _ { \rho } \right\} .
$$

Figure $2 ( \mathrm { a } )$ shows that Rare-switching OFUL achieves substantially lower pseudo-regret than Frequent-update feasible OFUL across all tested values of $\rho .$ The reason is reflected in Figure 2(b): Rare-switching OFUL performs far fewer stationary-target switches, which in turn leads to many fewer transition rounds, as shown in Figure 2(c). This benefit is especially pronounced for small $\rho ,$ where moving between targets requires longer feasible transition paths. Figure 2(d) shows that both feasible methods maintain zero sliding-window violations for all tested values of $\rho ,$ whereas Unconstrained OFUL violates the constraint whenever direct switching is infeasible. At $\rho = 2$ , Unconstrained OFUL also satisfies the constraint because the $\ell _ { 1 }$ diameter of the simplex is at most 2.

## J.3 Planning Beyond Cyclic Symmetry

We address Q3 with a non-cyclic constraint for which current actions afect future feasible choices and stationary behavior is suboptimal. Let

$$
\mathcal { X } = \{ 0 , e _ { 1 } , e _ { 2 } \} \subset \mathbb { R } ^ { 2 } , \qquad c = ( 1 , 3 ) ^ { \top } , \qquad w = 4 ,
$$

and define

$$
\mathcal { C } = \left\{ ( x _ { 1 } , x _ { 2 } , x _ { 3 } , x _ { 4 } ) \in \mathcal { X } ^ { 4 } : 4 c ^ { \top } x _ { 1 } + 2 c ^ { \top } x _ { 2 } + c ^ { \top } x _ { 3 } + c ^ { \top } x _ { 4 } \leq 1 4 \right\} .
$$

Because the coeficients depend on position within the window, C is not invariant under cyclic shifts. Hence feasibility at round t depends on the history state $s _ { t } = ( x _ { t - 3 } , x _ { t - 2 } , x _ { t - 1 } )$ . We use

$$
\theta ^ { \star } = ( 0 . 6 , 1 . 6 ) ^ { \top } , \qquad Y _ { t } = \langle \theta ^ { \star } , x _ { t } \rangle + \eta _ { t } , \qquad \eta _ { t } \sim \mathcal { N } ( 0 , 0 . 1 ^ { 2 } ) .
$$

The stationary feasible set is $\mathcal { Z } = \{ 0 , e _ { 1 } \}$ , so the best stationary reward is 0.6 per round. In contrast, the periodic trajectory $( e _ { 1 } , e _ { 2 } , 0 , e _ { 1 } , e _ { 1 } , e _ { 2 } , 0 , e _ { 1 } , . . . )$ is feasible and has average reward

$$
\frac { 0 . 6 + 1 . 6 + 0 + 0 . 6 } { 4 } = 0 . 7 .
$$

![](images/baaf3daa7eb35f7b84be3bf50ad4bcbcd97062d5d35aaf6ef7f9e790dfcb7d9d.jpg)  
Figure 4: Ablation on optimistic exploration. Left: normalized pseudo-regret for Rare-Update Optimistic Planning and its non-optimistic variant. Right: paired regret diference, defined as $\mathrm { R e g r e t } _ { \mathrm { N o O p t } } - \mathrm { R e g r e t } _ { \mathrm { O p t } } ,$ so positive values favor optimism. Error bars denote 95% confidence intervals.

Thus, exploiting non-stationary feasible behavior requires planning over the history state. We compare three feasible policies. Rare-Update Optimistic Planning follows Algorithm 2 and optimizes the frozen optimistic reward over the full remaining horizon. Myopic Feasible OFUL uses the same estimator and confidence sets, but selects the current feasible action without accounting for future rewards. Stationary OFUL restricts its actions to the stationary feasible set Z. Figure 3 shows a clear benefit from remaining-horizon planning. The normalized pseudo-regret of rare-update optimistic planning decreases rapidly with T and approaches zero, indicating that the learner approaches the ofline-optimal non-stationary behavior. In contrast, Stationary OFUL approaches a normalized regret of approximately 0.1, matching the gap between the optimal long-run reward 0.7 and the best stationary reward 0.6. Myopic feasible OFUL performs even worse, with a persistent normalized-regret gap of approximately 0.15, showing that immediate feasibility and optimism alone do not account for the efect of current actions on future feasible rewards. The right panel shows that these gains are achieved with few policy updates: the number of replanning events grows slowly with the horizon, consistent with the determinant-doubling update rule. Further details of this experiment are provided in Appendix L.

## K ABLATION AND SENSITIVITY STUDIES

We conduct ablation and sensitivity studies on both synthetic and real-world tasks to analyze the robustness of our method and understand the impact of key design choices.

## K.1 Real-World Ablation Studies

Sensitivity to Budget and Horizon. We evaluate the routing policies across sliding-window budgets $B \in$ {0.15, 0.25, 0.35} and horizons $T \in \{ 5 0 0 , 2 5 0 0 , 5 0 0 0 \}$ . Table 6 shows that Rare-Update Optimistic Planning maintains competitive reward with zero violations across all settings while requiring substantially fewer policy updates. As the budget becomes tighter, constraint violations increase for Unconstrained OFUL and Primal-Dual OFUL.

Ablation on Optimistic Exploration. We isolate the efect of optimism by comparing Rare-Update $\mathrm { O p t i - }$ mistic Planning with a non-optimistic variant that uses the same planning horizon, feasibility mechanism, and determinant-based update schedule, but removes the confidence bonus. The two variants use

$$
U _ { t } ( a ) = \langle \widehat { \theta } _ { t } , e _ { a } \rangle + \beta _ { t } \| e _ { a } \| _ { V _ { t } ^ { - 1 } } \qquad \mathrm { a n d } \qquad \widehat { r } _ { t } ( a ) = \langle \widehat { \theta } _ { t } , e _ { a } \rangle ,
$$

respectively. We evaluate $T \in \{ 5 0 0 , 1 0 0 0 , 2 5 0 0 , 5 0 0 0 \}$ with window size $w = 2 0$ and planning horizon $H = 5$ using 50 users and 10 independent runs per user.

Figure 4 shows that optimism incurs a short-horizon exploration cost but becomes beneficial as the horizon

![](images/4250e20c5c7bad9e76e4264085c325dfc1128ada339638cf3754b760e644be07.jpg)

![](images/be115ade57271fbfa545c6a26735459f9bf8b760e1ec3cd8e5cdca7b339a7143.jpg)

![](images/f7bb0d5798919848fad220ca3994fde83b3483660d0b028b05fa2b5606ee4012.jpg)  
Figure 5: Sensitivity to the determinant switching threshold α. (a) Realized pseudo-regret, (b) number of stationary-target switches, and (c) total number of transition rounds. The default choice $\alpha = 2$ achieves the lowest pseudo-regret in this experiment. Curves show means over 30 runs with 95% confidence intervals.

grows. The non-optimistic variant achieves lower regret at $T = 5 0 0$ and $T = 1 0 0 0$ , while the diference is smal at $T = 2 5 0 0$ . At $T = 5 0 0 0$ , the optimistic variant achieves lower regret, and the paired diference is positive.

## K.2 Synthetic Ablation Studies

We conduct additional experiments to examine how the main design choices in feasible rare-switching OFUL afect empirical performance. We use the same setup as in Section J, with window length $w = 8$ , horizon $T = 1 6 3 8 4$ , Gaussian observation noise with standard deviation 0.1, and the same reward parameter $\theta ^ { \star }$

Switching Threshold We first study the determinant threshold that controls the duration of each stationary episode. In the main algorithm, the current optimistic target is kept fixed until det $( V _ { t } ) > 2 \operatorname* { d e t } ( V _ { t _ { k } } )$ , where $s _ { k }$ denotes the start of episode k. To examine the sensitivity to this choice, we replace the factor 2 with a general threshold $\alpha > 1$ and consider

$$
\operatorname* { d e t } ( V _ { t } ) > \alpha \operatorname* { d e t } ( V _ { s _ { k } } ) , \qquad \alpha \in \{ 1 . 2 5 , 1 . 5 , 2 , 4 , 8 \} .
$$

All other parameters are kept fixed, including the ramping radius $\rho = 1$ . Smaller values of α lead to more frequent target updates, while larger values produce longer episodes and fewer switches. Figure 5 shows that both the number of target switches and the transition cost decrease steadily as α increases, dropping from about 135 switches and 2700 transition rounds at $\alpha = 1 . 2 5$ to roughly 20 switches and 400 rounds at $\alpha = 8$ . The efect on pseudo-regret is non-monotone. Very small thresholds incur high transition cost, while very large thresholds update the target less often and adapt more slowly. The lowest regret is achieved at the default choice $\alpha = 2 .$ which provides a good balance between statistical adaptation and transition cost.

Scaling with the Action Dimension We next examine how rare-switching OFUL behaves as the dimension of the simplex action space increases. We vary $d \in \{ 4 , 8 , 1 6 , 3 2 \}$ , while keeping $w = 8 , T = 1 6 3 8 4 , \rho = 1$ , and the determinant threshold $\alpha = 2$ fixed. To isolate the efect of the action dimension, we use

$$
\begin{array} { r } { \theta _ { d } ^ { \star } = ( 0 . 8 , \ldots , 0 . 8 , 1 . 0 ) ^ { \top } \in \mathbb { R } ^ { d } . } \end{array}
$$

Thus, the last simplex vertex is optimal and has reward gap 0.2 relative to every other vertex, independent of d.   
This keeps the statistical separation between the optimal and suboptimal actions fixed as the dimension grows.   
Within each run, we use the same noise realization for all values of d.

Figure 6 reports pseudo-regret, the number of stationary-target switches, and the total number of transition rounds. As the dimension increases, the learner must explore a larger number of candidate actions while the reward gap remains fixed. This leads to higher learning cost and, in turn, can produce more target changes and feasibility-preserving transitions.

Scaling with the Window Length Finally, we examine how the window length afects the cost of maintaining feasibility. We vary $w \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ , and set $T = 2 0 4 8 w .$ , so that $T / w = 2 0 4 8$ is fixed across settings. All other parameters are unchanged. Figure 7 shows that the number of stationary-target switches changes very little with $w ,$ remaining between roughly 48 and 52. In contrast, the number of transition rounds grows sharply, from about 100 at $w = 2$ to nearly 4800 at $w = 3 2$ . Figure 7 (d) makes the source of this increase explicit: the number of transition rounds per switch rises from 2 to about 92 over the same range. In our simplex setting, distinct stationary targets are simplex vertices and therefore satisfy $\| z ^ { \prime } - z \| _ { 1 } = 2$ . With $\rho = 1$ , the transition construction consequently requires 3w − 4 rounds per switch, which gives 2, 8, 20, 44, and 92 rounds for the tested window lengths. Thus, the increase in transition cost is driven almost entirely by longer feasible transitions rather than by more frequent switching. Pseudo-regret also increases with w, although the horizon grows proportionally with w, so this trend should not be interpreted as an isolated efect of the window length.

![](images/6e2f87646146811f291872fa811d9ff9f39d40199ad15a52287490ba77b8587b.jpg)

![](images/c759c1c3ade18637c899bbe6136aee2f28beaeb9dd06fdf44bcde4d57e95b895.jpg)

![](images/2ae41939be3bb812aefd37759eb997a791db3de56a09c4c9bfb633b3df8fd5a2.jpg)

Figure 6: Scaling with the action dimension for $d \in \{ 4 , 8 , 1 6 , 3 2 \}$ with a fixed reward gap of 0.2. (a) Realized pseudo-regret, (b) number of stationary-target switches, and (c) total number of transition rounds. Curves show means over 30 runs with 95% confidence intervals.  
![](images/7c829ec21236b816da320eb6b4f3bbdc4d94f19376f3b4a8dd4bd47c686c0629.jpg)

![](images/96bba43f11f466dd954ecd4695a6e4358b15a61553800fcd64140366876d7bfe.jpg)

![](images/2d96549a2d5061bb75768aab44cb720c2e2225ebfa26ee67b90d2425f1941032.jpg)

![](images/1793811c187e4072cd7effdffaf5275435c15ae203e42f86f41057555e171ed3.jpg)  
Figure 7: Scaling with the window length for $w \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ with $T / w = 2 0 4 8$ fixed. (a) Realized pseudoregret, (b) number of stationary-target switches, (c) total number of transition rounds, and (d) transition rounds per switch. The switch count remains nearly constant, while the cost of each feasible switch grows with w. Curves show means over 30 runs with 95% confidence intervals.

Planning Horizon We examine how much lookahead is needed to benefit from history-state planning. Using the non-cyclic instance from Section J.3, we replace exact remaining-horizon planning by an H-step optimistic planner with $H \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ , and compare these choices with the full remaining-horizon planner. At round t, the truncated planner selects

$$
x _ { t } \in \arg \operatorname* { m a x } _ { \stackrel { x \in A \left( s _ { t } \right) } { F \left( s _ { t } , x \right) \in S _ { T - t } } } \left\{ \overline { { U } } ( x ) + J _ { \operatorname* { m i n } \left\{ H - 1 , T - t \right\} } ^ { \overline { { U } } } \left( F ( s _ { t } , x ) \right) \right\} .
$$

The viability condition is imposed over the full remaining horizon, so every policy remains exactly feasible; only the reward lookahead is truncated. Thus, $H = 1$ corresponds to the myopic feasible policy, while $\mathrm { \Delta ^ { 6 } F u l l } ^ { \mathrm { , 5 } }$ denotes exact remaining-horizon planning as in Algorithm 2. We fix $T = 1 0 2 4$ and keep all statistical parameters and the determinant-doubling update rule unchanged. Figure 8 shows that longer lookahead generally improves performance, with the largest gains appearing at longer horizons. The normalized pseudo-regret is about 0.15 for the myopic policy and drops to roughly 0.10 at $H = 2$ . It changes only modestly over several short and moderate horizons, before falling more clearly for larger H: to approximately 0.08 at $H = 1 2 8 , 0 . 0 5$ at $H = 5 1 2$ and below 0.01 under full remaining-horizon planning. This supports the role of continuation information in evaluating the future consequences of current actions. The number of policy updates remains small across all planning horizons, ranging from roughly 10 to 16, and increases only gradually with H.

![](images/ffdc92657c03e57930b18858de4c8a2d86d641af11b89ff9f277a2e4a64b069a.jpg)

![](images/dea079983763f5d605251e2b8b43a1008ee023071b8fede7d3387f7cbd47a020.jpg)  
Figure 8: Sensitivity to the planning horizon H for $T = 1 0 2 4$ . Left: normalized pseudo-regret against the ofline-optimal feasible trajectory. Right: number of policy updates. The case $H = 1$ is myopic and $\mathrm { ^ { 6 6 } F u l l ^ { \circ } }$ denotes exact remaining-horizon planning. Curves show means over 30 runs with 95% confidence intervals.

## L IMPLEMENTATION DETAILS

Complete code, preprocessing scripts, and experimental configurations are publicly available at https:// anonymous.4open.science/r/sliding-window-constrained-bandits.

## L.1 Real-World Experiments

Recommendation Diversity Experiment Setup We use the $K = 8$ most frequent first-level content categories in the processed KuaiRand-1K dataset (Gao et al., 2022). Each category $j \in \mathcal { A } = \left\{ 1 , \dots , K \right\}$ is treated as an action and represented by the one-hot vector $e _ { j } \in \mathbb { R } ^ { K }$ . For each user u, we construct the preference vector $\theta _ { u } ^ { \star } = ( \theta _ { u , 1 } ^ { \star } , \ldots , \theta _ { u , K } ^ { \star } )$ , where $\theta _ { u , j } ^ { \star }$ is the smoothed empirical click probability for category j obtained during preprocessing. We retain users with observations from all K selected categories and at least 50 historical interactions in every category, and randomly select 100 eligible users. At round $t ,$ after selecting category $a _ { t } .$ the environment generates

$$
Y _ { t } \sim \mathrm { B e r n o u l l i } \big ( \theta _ { u , a _ { t } } ^ { \star } \big ) , \qquad \mathbb { E } [ Y _ { t } \mid a _ { t } = j ] = \langle \theta _ { u } ^ { \star } , e _ { j } \rangle .
$$

The learner observes only the realized binary reward $Y _ { t }$ . The vector $\theta _ { u } ^ { \star }$ remains hidden from the learning algorithms and is used only by the environment and for evaluation. We impose the exact five-in-twenty exposure constraint

$$
\sum _ { q = t - 1 9 } ^ { t } \mathbf { 1 } \{ a _ { q } = j \} \leq 5 , \qquad j \in \mathcal { A } , \quad t = 2 0 , \ldots , T .
$$

Hence, with history $s _ { t } = ( a _ { t - 1 9 } , \dotsc , a _ { t - 1 } )$ , the feasible action set is

$$
A ( s _ { t } ) = \left\{ j \in { \mathcal { A } } : \sum _ { q = t - 1 9 } ^ { t - 1 } \mathbf { 1 } \{ a _ { q } = j \} < 5 \right\} .
$$

We evaluate $T \in \{ 5 0 0 , 2 5 0 0 , 5 0 0 0 \}$ . For each user and horizon, we perform 10 independent simulations. We first average repeated runs within each user and then average the resulting quantities across users. For Rare-Update Optimistic Planning, we use a planning horizon of $H = 5$ . For all the OFUL-based methods, we use regularization $\lambda = 1$ , confidence level $\delta = 0 . 0 5$ , sub-Gaussian parameter $R = 1 / 2$ , and the bound $S = \sqrt { K }$

LLM Routing Experiment Setup We use the LLMRouterBench dataset (Li et al., 2026), which provides query-level evaluations of multiple large language models, including response quality scores and inference cost information. We construct routing streams from four benchmark tasks: MMLU-Pro (Wang et al., 2024) for general knowledge reasoning, GPQA (Rein et al., 2024) for graduate-level scientific reasoning, LiveCodeBench (Jain et al., 2025) for coding problems, and ArenaHard (Li et al., 2025) for instruction-following and conversational evaluation. These benchmarks represent diverse query distributions with diferent dificulty levels, where diferent language models exhibit diferent quality–cost trade-ofs. We select a pool of $K = 6$ language models: GPT-5 (OpenAI, 2025), Claude-Sonnet-4 (Anthropic, 2025), Gemini-2.5-Pro (Google DeepMind, 2025b), DeepSeek-V3 (DeepSeek-AI, 2024), Qwen3-235B (Yang et al., 2025), and Gemini-2.5-Flash (Google DeepMind, 2025a). Each model $j \in \mathcal { A } = \left\{ 1 , \dots , K \right\}$ is considered an available routing action. For every query $q _ { t } .$ , we first extract an ℓ<sub>2</sub>-normalized semantic representation using the pretrained Sentence-BERT encoder all-MiniLM-L6-v2 (Reimers and Gurevych, 2019):

$$
u _ { t } \in \mathbb { R } ^ { 3 8 4 } , \qquad \| u _ { t } \| _ { 2 } = 1 .
$$

To reduce the feature dimension, we project the query embeddings onto their leading $r = 6 4$ principal components. We then $\ell _ { 2 ^ { - } }$ normalize the projected representation, obtaining

$$
v _ { t } \in \mathbb { R } ^ { 6 4 } , \qquad \| v _ { t } \| _ { 2 } = 1 .
$$

The PCA projection is fitted once on the set of query embeddings before the simulations, without using reward or cost information. For each model $j \in { \mathcal { A } } ,$ we construct the query–model interaction feature

$$
\phi ( q _ { t } , j ) = e _ { j } \otimes v _ { t } \in \mathbb { R } ^ { 6 4 K } ,
$$

where $e _ { j } \in \mathbb { R } ^ { K }$ is the one-hot encoding of model $j .$ . Since $K \ = \ 6 .$ the resulting feature dimension is $d =$ $6 4 K = \bar { 3 } 8 4$ . After selecting model $a _ { t } .$ , the bandit feature vector is $x _ { t } : = \phi ( q _ { t } , a _ { t } ) \in \mathbb { R } ^ { d }$ . Let $\mu ( q _ { t } , j )$ denote the quality score provided by LLMRouterBench for query $q _ { t }$ and model j. When a model $a _ { t } \in \mathcal A$ is selected, the environment returns $Y _ { t } = \mu ( q _ { t } , a _ { t } )$ . The online algorithms use a linear predictor over $\phi ( q _ { t } , j )$ to estimate model quality. Thus, in this real-world experiment, we do not require the benchmark reward table to be exactly linear in the chosen features. The inference costs $c ( q _ { t } , j )$ are obtained from LLMRouterBench and are used without additional rescaling. The target sliding-window cost constraint is

$$
\sum _ { i = t - w + 1 } ^ { t } c ( q _ { i } , a _ { i } ) \leq B , \qquad t = w , \ldots , T ,
$$

with window length $w = 5$ . To guarantee pathwise feasibility without knowledge of future queries, the online feasible policies additionally use the zero-padded prefix convention

$$
\sum _ { i = \operatorname* { m a x } \{ 1 , t - w + 1 \} } ^ { t } c ( q _ { i } , a _ { i } ) \leq B , \qquad t = 1 , \ldots , T .
$$

This condition coincides with the standard length-w sliding-window constraint for $t \geq w$ and is stronger only during the first $w - 1$ rounds. We evaluate $T \in \{ 5 0 0 , 2 5 0 0 , 5 0 0 0 \}$ and $B \in \{ 0 . 1 5 , 0 . 2 5 , 0 . 3 5 , 0 . 4 0 \}$ . For each configuration, we perform 10 independent simulations using query streams sampled uniformly without replacement. Within each simulation, all methods are evaluated on the same sampled query stream. Since future queries are unavailable at decision time, we use planning horizon $H = 1$ . Thus, in this application, Rare-Update Optimistic Planning reduces to myopic feasible optimistic selection while retaining the determinant-based rare-update mechanism.

For all OFUL-based methods, we use ridge regularization parameter $\lambda _ { \mathrm { r e g } } = 1$ , confidence parameter $\delta = 0 . 0 5$ and exploration scale $R = 0 . 5 \ :$ . The determinant-doubling factor is 2. In the LLM-routing experiments, the optimistic score uses the fixed confidence multiplier $\beta _ { \mathrm { e x p } } = R \sqrt { d \log ( 1 / \delta ) }$ . This experimental confidence multiplier is used uniformly by the OFUL-based methods and difers from the theoretical confidence radius used in the regret analysis. Future queries are not available to the learner before their arrival. Therefore, the LLM-routing experiment does not apply the full remaining-horizon viability recursion of Section 6. After observing the current query $q _ { t } ,$ , the learner instead restricts its choice to the currently feasible LLM actions

$$
\mathcal { A } _ { t } ( s _ { t } , q _ { t } ) = \left\{ j \in \mathcal { A } : \sum _ { i = \operatorname* { m a x } \{ 1 , t - w + 1 \} } ^ { t - 1 } c ( q _ { i } , a _ { i } ) + c ( q _ { t } , j ) \leq B \right\} .
$$

If $\mathcal { A } _ { t } ( s _ { t } , q _ { t } ) = \emptyset$ , the learner executes a no-op fallback ⊥ satisfying

$$
c ( q _ { t } , \perp ) = 0 , \qquad \mu ( q _ { t } , \perp ) = 0 .
$$

The no-op is used only when no real LLM action is feasible. Because the preceding prefix is maintained within budget, the zero-cost fallback preserves feasibility. Consequently, the feasible policies maintain the sliding-window constraint pathwise without requiring knowledge of future queries. Because future query-dependent costs are unknown, this contextual LLM-routing experiment is not a direct instantiation of Theorem 6.5; the theorem’s regret guarantee assumes known deterministic finite-memory transitions.

Baselines. We compare against three baselines. Myopic Feasible OFUL selects

$$
a _ { t } \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } _ { t } ( s _ { t } , q _ { t } ) } U _ { t } ( a ) ,
$$

where $U _ { t } ( a )$ denotes the optimistic reward estimate. If $\mathcal { A } _ { t } ( s _ { t } , q _ { t } ) = \emptyset$ , it uses the same zero-cost, zero-reward no-op fallback. This baseline enforces feasibility at every round and recomputes its optimistic action at every round, providing a fully updated feasible baseline against which to evaluate the rare-update mechanism.

Primal-Dual OFUL. For the recommendation experiment, the sliding-window constraint consists of one exposure constraint for each category. Accordingly, Primal-Dual OFUL maintains a vector of nonnegative dual variables

$$
\lambda _ { t } = ( \lambda _ { t , 1 } , \ldots , \lambda _ { t , K } ) \in \mathbb { R } _ { + } ^ { K } ,
$$

with one multiplier for each category. Let $U _ { t } ( a )$ denote the OFUL optimistic reward estimate for category a. At round $t ,$ the policy selects

$$
a _ { t } \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ U _ { t } ( a ) - \lambda _ { t , a } \right\} .
$$

After observing the reward and updating the OFUL estimator, the dual variables are updated whenever a complete length-w window is available. For $t \geq w .$ , let $\ell _ { t } : = t - w + 1$ denote the dual-update index. For each category $j \in { \mathcal { A } }$

$$
\lambda _ { t + 1 , j } = \left[ \lambda _ { t , j } + \eta _ { \ell _ { t } } \left( \sum _ { i = t - w + 1 } ^ { t } \mathbf { 1 } \{ a _ { i } = j \} - c \right) \right] _ { + } , \qquad \eta _ { \ell _ { t } } = \frac { \eta } { \sqrt { \ell _ { t } } } ,
$$

where $[ x ] _ { + } : = \operatorname* { m a x } \{ x , 0 \}$ . In the recommendation experiments, $K = 8 , w = 2 0$ , and $c = 5 .$ , so each category has its own dual multiplier corresponding to its five-in-twenty exposure constraint.

For the LLM-routing experiment, there is a single sliding-window cost constraint, so Primal-Dual OFUL uses a scalar nonnegative multiplier $\lambda _ { t }$ . At round $t ,$ it selects

$$
a _ { t } \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ U _ { t } ( a ) - \lambda _ { t } c ( q _ { t } , a ) \right\} ,
$$

and updates the multiplier according to

$$
\lambda _ { t + 1 } = \left[ \lambda _ { t } + \eta _ { t } \left( \sum _ { i = \operatorname* { m a x } \{ 1 , t - w + 1 \} } ^ { t } c ( q _ { i } , a _ { i } ) - B \right) \right] _ { + } , \qquad \eta _ { t } = \frac { \eta } { \sqrt { t } } .
$$

Unconstrained OFUL selects

$$
a _ { t } \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } U _ { t } ( a )
$$

without enforcing the sliding-window constraint. It provides a reference for the reward attainable when exact feasibility is ignored. Because this policy may violate the constraint, its pseudo-regret relative to the constrained ofline optimum can be negative and should be interpreted together with its number of constraint violations.

Ofline benchmark. For each real-world application, we define an ofline oracle that has access to the complete reward information and computes the maximum achievable reward under the corresponding application-specific constraint. This oracle is used only as an evaluation reference and is not available to the online learning algorithms. For the recommendation diversity experiment, the ofline oracle has access to the user preference vector $\theta _ { u } ^ { \star }$ and solves

$$
V _ { T } ^ { \mathrm { o f f } } ( \theta _ { u } ^ { \star } ) = \operatorname* { m a x } _ { a _ { 1 } , \ldots , a _ { T } } \sum _ { t = 1 } ^ { T } \theta _ { u , a _ { t } } ^ { \star } ,
$$

subject to the five-in-twenty exposure constraint

$$
\sum _ { q = t - 1 9 } ^ { t } \mathbf { 1 } \{ a _ { q } = j \} \leq 5 , \qquad j \in \mathcal { A } , \quad t = 2 0 , \ldots , T .
$$

Since all evaluated horizons are divisible by 20, the constrained optimum is obtained by selecting the four categories with the largest preference values and repeating a 20-round periodic schedule containing exactly five recommendations from each of these four categories. Every length-20 sliding window therefore contains exactly five occurrences of each selected category. For the LLM-routing experiment, the ofline oracle has access to the complete query–model reward and cost tables. Let $\mu ( q _ { t } , j )$ denote the benchmark quality score for query $q _ { t }$ and model $j .$ The oracle computes

$$
V _ { T } ^ { \mathrm { o f f } } = \operatorname* { m a x } _ { a _ { 1 } , \ldots , a _ { T } } \sum _ { t = 1 } ^ { T } \mu ( q _ { t } , a _ { t } ) ,
$$

subject to the same zero-padded sliding-window constraint

$$
\sum _ { i = \operatorname* { m a x } \{ 1 , t - w + 1 \} } ^ { t } c ( q _ { i } , a _ { i } ) \leq B , \qquad t = 1 , \ldots , T .
$$

The no-op fallback ⊥, with $c ( q _ { t } , \bot ) = 0$ and $\mu ( q _ { t } , \bot ) = 0$ is available to the oracle only when no real LLM action is feasible under the current history. Thus, the online feasible policies and the ofline oracle are evaluated under the same action and feasibility conventions.

Evaluation metrics. We evaluate all methods using four metrics: average reward, normalized pseudo-regret, constraint violations, and the number of policy updates. For the recommendation diversity experiment, the average expected reward is

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ Y _ { t } ] = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \theta _ { u , a _ { t } } ^ { \star } .
$$

For the LLM-routing experiment, the reward is the benchmark quality score, so the average reward is

$$
{ \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \mu ( q _ { t } , a _ { t } ) ,
$$

where $\mu ( q _ { t } , \bot ) = 0$ when the fallback action is used. For the recommendation diversity experiment, normalized pseudo-regret is

$$
\frac { R _ { T } } { T } , \qquad R _ { T } = V _ { T } ^ { \mathrm { o f f } } ( \theta _ { u } ^ { \star } ) - \sum _ { t = 1 } ^ { T } \theta _ { u , a _ { t } } ^ { \star } .
$$

For the LLM-routing experiment, normalized pseudo-regret is

$$
\frac { R _ { T } } { T } , \qquad R _ { T } = V _ { T } ^ { \mathrm { o f f } } - \sum _ { t = 1 } ^ { T } \mu ( q _ { t } , a _ { t } ) .
$$

The ofline benchmarks have access to the complete reward information and are used only as evaluation references. For the recommendation diversity experiment, the number of constraint violations is

$$
N _ { \mathrm { v i o l } } = \sum _ { t = 2 0 } ^ { T } \mathbf { 1 } \left\{ \exists j \in { \cal A } : \sum _ { q = t - 1 9 } ^ { t } \mathbf { 1 } \{ a _ { q } = j \} > 5 \right\} .
$$

For the LLM-routing experiment, we report violations of the target length-w sliding-window inference-cost constraint:

$$
N _ { \mathrm { v i o l } } = \sum _ { t = w } ^ { T } \mathbf { 1 } \left\{ \sum _ { i = t - w + 1 } ^ { t } c ( q _ { i } , a _ { i } ) > B \right\} .
$$

Finally, we report the number of policy updates performed by each algorithm. For Rare-Update Optimistic Plan ning, an update corresponds to refreshing the frozen optimistic model when the determinant-doubling condition is triggered. Fewer updates therefore indicate fewer policy recomputations.

## L.2 Synthetic Experiments

Ofline geometry and horizon misalignment. For the experiment in Section J.1, we solve the finitehorizon ofline problem exactly by linear programming for every pair $( w , r )$ . We use even window lengths $w \in \{ 4 , 6 , 8 , \ldots , 3 2 \}$ , and horizons

$$
T = 2 0 w + r , \qquad r \in \{ 0 , \ldots , w - 1 \} .
$$

For each instance, we compute the exact optimum $V _ { T }$ and report $\Delta _ { T } = V _ { T } - T f ( z ^ { \star } )$ . This experiment is deterministic and therefore does not involve repeated stochastic runs or confidence parameters.

Rare switching under cyclic ramping constraints. For the experiment in Section J.2, we use

$$
d = 8 , \qquad w = 8 , \qquad T = 1 6 3 8 4 ,
$$

with $\theta ^ { \star } = ( 0 . 5 5 , 0 . 6 0 , 0 . 6 5 , 0 . 7 0 , 0 . 7 5 , 0 . 8 0 , 0 . 9 0 , 1 . 0 0 ) ^ { \top }$ and observations

$$
Y _ { t } = \langle \theta ^ { \star } , x _ { t } \rangle + \eta _ { t } , \qquad \eta _ { t } \sim \mathcal { N } ( 0 , 0 . 1 ^ { 2 } ) .
$$

We vary $\rho \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 \}$ . For all OFUL-based policies in this experiment, we use

$$
\lambda = 1 , \qquad \delta = 0 . 0 5 , \qquad R = 0 . 1 , \qquad S = \| \theta ^ { \star } \| _ { 2 } .
$$

We initialize $V _ { 1 } ~ = ~ \lambda I _ { d }$ and, for the rare-switching feasible algorithm, update the estimator using retained stationary observations as specified in Algorithm 1. Since $\mathcal { Z } = \Delta _ { d }$ , the optimistic stationary-action oracle is implemented exactly by evaluating

$$
\widehat { \theta } _ { t , i } + \beta _ { t } ( \delta ) \sqrt { ( V _ { t } ^ { - 1 } ) _ { i i } } , \qquad i = 1 , \ldots , d ,
$$

at the vertices $e _ { i }$ and selecting a maximizer. To preserve feasibility when changing stationary targets from z to $z ^ { \prime } ,$ we use the transition

$$
N = \left\lceil { \frac { ( w - 1 ) \| z ^ { \prime } - z \| _ { 1 } } { \rho } } \right\rceil , \qquad y _ { j } = z + { \frac { j } { N } } ( z ^ { \prime } - z ) , \quad j = 1 , \dots , N ,
$$

followed by $z ^ { \prime }$ for $w - 2$ additional rounds. All transition rounds are included in the pseudo-regret. This is the transition construction used in the rare-switching experiment. We compare Rare-Switching $O F U L$ , Frequent-Update Feasible $O F U L$ , and Unconstrained $O F U L$ . The rare-switching method changes its optimistic stationary target only at determinant-based episode boundaries, whereas the frequent-update feasible baseline recomputes the target after every retained stationary observation. The unconstrained baseline ignores the ramping constraint.

Planning beyond cyclic symmetry. For the experiment in Section J.3, we use

$$
\begin{array} { r } { \mathcal { X } = \{ 0 , e _ { 1 } , e _ { 2 } \} , \qquad w = 4 , \qquad \theta ^ { \star } = ( 0 . 6 , 1 . 6 ) ^ { \top } , } \end{array}
$$

with observations

$$
Y _ { t } = \langle \theta ^ { \star } , x _ { t } \rangle + \eta _ { t } , \qquad \eta _ { t } \sim \mathcal { N } ( 0 , 0 . 1 ^ { 2 } ) .
$$

We evaluate $T \in \{ 5 1 2$ , 1024, 2048, 4096, 8192}, using

$$
\lambda = 1 , \qquad \delta = 0 . 0 5 , \qquad R = 0 . 1 , \qquad S = \| \theta ^ { \star } \| _ { 2 } .
$$

Because X is finite and $w = 4 .$ , we compute viable history states explicitly and implement exact remaininghorizon planning by backward dynamic programming. The ofline value $\dot { V } _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } )$ is computed by the same exact dynamic-programming formulation. We compare three feasible policies. Rare-Update Optimistic Planning follows Algorithm 2 and optimizes the frozen optimistic reward over the full remaining horizon. Myopic Feasible OFUL uses the same statistical estimator and confidence construction but selects

$$
x _ { t } \in \arg \operatorname* { m a x } _ { x \in A ( s _ { t } ) \atop F ( s _ { t } , x ) \in S _ { T - t } } U _ { k } ( x ) ,
$$

without including a continuation value. Stationary OFUL restricts its actions to the stationary feasible set Z. This comparison separates the gain from explicit history-state planning from both statistical optimism and the restriction to stationary behavior. For this experiment, we report

$$
\frac { \widehat { R } _ { T } } { T } , \qquad \widehat { R } _ { T } = V _ { T } ^ { \mathrm { o f f } } ( \theta ^ { \star } ) - \sum _ { t = 1 } ^ { T } \langle \theta ^ { \star } , x _ { t } \rangle ,
$$

together with the number of policy updates. All three policies satisfy the sliding-window constraint exactly.

Computational Infrastructure and Software Environment All reported experiments were conducted on a machine equipped with a 10-thread CPU. The experiments were implemented in Python 3.10.20 using standard scientific computing libraries. No GPU acceleration was required.

Dataset and License Information We use publicly available datasets for evaluation, including the KuaiRand-1K recommendation dataset and the LLMRouterBench benchmark. The KuaiRand dataset is released under the CC-BY-SA-4.0 license and is available from the oficial repository: https://github.com/ chongminggao/KuaiRand. The LLMRouterBench benchmark and associated resources are available from the oficial repository: https://github.com/ynulihao/LLMRouterBench. LLMRouterBench aggregates multiple underlying datasets and models; therefore, the licenses of individual components follow their respective original terms of use. We cite the original dataset and benchmark sources and describe the preprocessing procedures used to construct our experimental instances.

Table 6: LLM-routing performance across diferent sliding-window budgets. Average reward and normalized pseudo-regret are reported as mean ± 95% confidence interval over 10 independent simulations for each configuration; violations and updates are reported as means.
<table><tr><td>Budget</td><td>T</td><td>Method</td><td>Avg. Reward</td><td>Norm. Pseudo-Regret</td><td>Violations</td><td>Updates</td></tr><tr><td rowspan="7">0.15</td><td rowspan="4">500</td><td>Rare-Update Optimistic Planning</td><td>0.7618 ± 0.0116</td><td> $0 . 1 5 6 6 \pm 0 . 0 1 3 2$ </td><td>0.0</td><td>249.8</td></tr><tr><td>Myopic Feasible OFUL</td><td>0.7583 ± 0.0120</td><td> $0 . 1 6 0 1 \pm 0 . 0 1 0 8$ </td><td>0.0</td><td>500.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 1 3 \pm 0 . 0 1 6 7$ </td><td> $0 . 1 4 7 1 \pm 0 . 0 1 5 6$ </td><td>104.5</td><td>500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 6 8 6 \pm 0 . 0 1 6 1$ </td><td> $0 . 1 4 9 8 \pm 0 . 0 1 3 6$ </td><td>112.7</td><td>500.0</td></tr><tr><td rowspan="4">2500</td><td>Rare-Update Optimistic Planning</td><td> $0 . 7 6 9 9 \pm 0 . 0 0 3 9$ </td><td> $0 . 1 5 0 9 \pm 0 . 0 0 4 1$ </td><td>0.0</td><td>789.1</td></tr><tr><td>Myopic Feasible OFUL Unconstrained OFUL</td><td> $0 . 7 6 8 2 \pm 0 . 0 0 4 8$ </td><td> $0 . 1 5 2 5 \pm 0 . 0 0 4 9$ </td><td>0.0</td><td>2500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 7 8 0 \pm 0 . 0 0 5 9$   $0 . 7 7 6 0 \pm 0 . 0 0 6 5$ </td><td> $0 . 1 4 2 7 \pm 0 . 0 0 6 4$ </td><td>527.2</td><td>2500.0</td></tr><tr><td></td><td></td><td> $0 . 1 4 4 7 \pm 0 . 0 0 6 2$ </td><td>529.1</td><td>2500.0</td></tr><tr><td rowspan="6"></td><td rowspan="3">5000</td><td>Rare-Update Optimistic Planning</td><td> $0 . 7 6 9 1 \pm 0 . 0 0 3 5$ </td><td> $0 . 1 5 1 8 \pm 0 . 0 0 3 4$ </td><td>0.0</td><td>1105.3</td></tr><tr><td>Myopic Feasible OFUL</td><td>0.7706 ± 0.0030</td><td>0.1503 ± 0.0030</td><td>0.0</td><td>5000.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 6 9 \pm 0 . 0 0 3 6$ </td><td> $0 . 1 4 4 1 \pm 0 . 0 0 3 8$ </td><td>1088.6</td><td>5000.0</td></tr><tr><td>Primal-Dual OFUL</td><td></td><td>0.7786 ± 0.0040</td><td>0.1424 ± 0.0042</td><td>1115.7</td><td>5000.0</td></tr><tr><td rowspan="3">500</td><td>Rare-Update Optimistic Planning</td><td>0.7670 ± 0.0122</td><td>0.1595 ± 0.0103</td><td>0.0</td><td>250.1</td></tr><tr><td>Myopic Feasible OFUL</td><td>0.7722 ± 0.0150</td><td>0.1543 ± 0.0117</td><td>0.0</td><td>500.0</td></tr><tr><td>Unconstrained OFUL</td><td>0.7713 ± 0.0167</td><td> $0 . 1 5 5 2 \pm 0 . 0 1 5 3$ </td><td>34.3</td><td>500.0</td></tr><tr><td rowspan="10"></td><td>2500</td><td>Primal-Dual OFUL</td><td>0.7724 ± 0.0165</td><td>0.1541 ± 0.0141</td><td>35.0</td><td>500.0</td></tr><tr><td rowspan="3"></td><td>Rare-Update Optimistic Planning</td><td>0.7741 ± 0.0063</td><td>0.1546 ± 0.0054</td><td>0.0</td><td>787.5</td></tr><tr><td>Myopic Feasible OFUL</td><td>0.7730 ± 0.0037</td><td>0.1558 ± 0.0041</td><td>0.0</td><td>2500.0</td></tr><tr><td>Unconstrained OFUL</td><td>0.7780 ± 0.0059</td><td> $0 . 1 5 0 7 \pm 0 . 0 0 6 5$ </td><td>189.5</td><td>2500.0</td></tr><tr><td></td><td>Primal-Dual OFUL</td><td>0.7785 ± 0.0058</td><td>0.1503 ± 0.0065</td><td>189.9</td><td>2500.0</td></tr><tr><td rowspan="3">5000</td><td>Rare-Update Optimistic Planning</td><td>0.7766 ± 0.0022</td><td>0.1526 ± 0.0022</td><td>0.0</td><td>1104.4</td></tr><tr><td>Myopic Feasible OFUL</td><td>0.7768 ± 0.0023</td><td>0.1524 ± 0.0023</td><td>0.0</td><td>5000.0</td></tr><tr><td>Unconstrained OFUL</td><td>0.7769 ± 0.0036</td><td> $0 . 1 5 2 3 \pm 0 . 0 0 3 6$ </td><td>410.8</td><td>5000.0</td></tr><tr><td>Primal-Dual OFUL</td><td></td><td>0.7781 ± 0.0038</td><td>0.1511 ± 0.0038</td><td>424.4</td><td>5000.0</td></tr><tr><td rowspan="10">0.35</td><td>500</td><td>Rare-Update Optimistic Planning</td><td>0.7626 ± 0.0114</td><td>0.1670 ± 0.0116</td><td>0.0</td><td>250.0</td></tr><tr><td rowspan="3"></td><td>Myopic Feasible OFUL</td><td> $0 . 7 7 2 4 \pm 0 . 0 1 3 8$ </td><td> $0 . 1 5 7 2 \pm 0 . 0 1 1 9$ </td><td></td><td></td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 1 3 \pm 0 . 0 1 6 7$ </td><td></td><td>0.0</td><td>500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 7 1 3 \pm 0 . 0 1 6 7$ </td><td>0.1583 ± 0.0160 0.1583 ± 0.0160</td><td>9.7</td><td>500.0</td></tr><tr><td rowspan="4">2500</td><td></td><td></td><td></td><td>9.7</td><td>500.0</td></tr><tr><td>Myopic Feasible OFUL</td><td>Rare-Update Optimistic Planning 0.7786 ± 0.0051  $0 . 7 7 6 7 \pm 0 . 0 0 5 2$ </td><td>0.1531 ± 0.0047  $0 . 1 5 5 0 \pm 0 . 0 0 4 6$ </td><td>0.0</td><td>789.8</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 8 0 \pm 0 . 0 0 5 9$ </td><td> $0 . 1 5 3 7 \pm 0 . 0 0 6 1$ </td><td>0.0 64.2</td><td>2500.0 2500.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 7 8 3 \pm 0 . 0 0 6 1$ </td><td>0.1534 ± 0.0062</td><td>63.6</td><td>2500.0</td></tr><tr><td rowspan="4">5000</td><td>Rare-Update Optimistic Planning</td><td>0.7784 ± 0.0024</td><td>0.1539 ± 0.0024</td><td></td><td></td></tr><tr><td></td><td>Myopic Feasible OFUL</td><td></td><td>0.0</td><td></td><td>1103.8 5000.0</td></tr><tr><td>Unconstrained OFUL</td><td> $0 . 7 7 8 9 \pm 0 . 0 0 3 7$   $0 . 7 7 6 9 \pm 0 . 0 0 3 6$ </td><td>0.1533 ± 0.0037  $0 . 1 5 5 4 \pm 0 . 0 0 3 6$ </td><td>0.0 124.8</td><td></td><td>5000.0</td></tr><tr><td>Primal-Dual OFUL</td><td> $0 . 7 7 6 9 \pm 0 . 0 0 3 6$ </td><td> $0 . 1 5 5 4 \pm 0 . 0 0 3 6$ </td><td>124.8</td><td></td><td>5000.0</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## M ALGORITHMS

This section provides the complete pseudocode for the two algorithms developed in the paper. Algorithm 1 presents Feasible Rare-Switching OFUL for the cyclic setting. Algorithm 2 extends the rare-update principle to general sliding-window constraints.

## M.1 Rare-Switching OFUL

```latex
Algorithm 1 Feasible Rare-Switching OFUL
Require: Horizon ${ \overline { { \mathbf { \Omega } _ { T , \delta } } } }$ regularization λ, confidence level δ, stationary feasible set Z, optimistic-action oracle Opt,
transition oracle Γ
Ensure: Feasible action sequence $x _ { 1 } , \ldots , x _ { T }$
1: Initialize $V  \lambda I _ { d } , \quad b  0 _ { d } , \quad t  1 , \quad k  1$
2: while $t \leq T$ do
3: Set $W _ { k }  V ,$ $\widehat { \theta } _ { k } \gets W _ { k } ^ { - 1 } b$
4: $\begin{array} { r } { \beta _ { k } ( \delta ) \gets R \sqrt { 2 \log \left( \frac { \operatorname* { d e t } ( W _ { k } ) ^ { 1 / 2 } } { \operatorname* { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \right) + \sqrt { \lambda } S } } \end{array}$
5: Select $z _ { k } \gets \mathsf { O p t } \left( \widehat { \theta } _ { k } , W _ { k } , \beta _ { k } ( \delta ) \right)$
6: if $k > 1$ and $z _ { k } \neq z _ { k - 1 }$ then
7: Obtain a feasible transition $( a _ { 1 } , \ldots , a _ { m } )  \Gamma ( z _ { k - 1 } , z _ { k } )$ , where m $\leq \tau$
8: for $j  1$ to min $\{ m , T - t + 1 \}$ do
9: Set $x _ { t } \gets a _ { j } ,$ execute $x _ { t } ,$ and observe $Y _ { t }$
10: t $ t + 1$
11: Exclude transition observations from V and b
12: end for
13: end if
14: if $t > T$ then
15: break
16: end if
17: $n _ { k } \gets 0$
18: while $t \leq T$ and (det $\therefore ( V ) \leq 2 \operatorname* { d e t } ( W _ { k } )$ or $[ k = 1$ and $n _ { k } < w - 1 ] )$ do
19: Set $x _ { t } \gets z _ { k }$ , execute $x _ { t } ,$ and observe $Y _ { t }$
20: Update $V \gets V + x _ { t } x _ { t } ^ { \top }$ $b \gets b + x _ { t } Y _ { t }$
21: $t \gets t + 1$
22: $n _ { k } \gets n _ { k } + 1$
23: end while
24: $k \gets k + 1$
25: end while
```

## M.2 Rare-Update Optimistic Planning

At each planning time $t _ { k } ,$ the learner freezes the current confidence set, defines the optimistic reward $U _ { k }$ , and computes an exact Bellman-optimal remaining-horizon plan from the current history state $s _ { t _ { k } }$ . The resulting plan is cached and followed until the determinant-doubling condition triggers the next planning episode. For $w = 1$ , we use the unique empty history $s _ { t } = ( )$

Algorithm 2 Rare-Update Optimistic Planning   
Require: $\overline { { \phantom { } T , w , \lambda , \delta } } ,$ Init, P   
Ensure: Feasible sequence $x _ { 1 } , \ldots , x _ { T }$   
1: $V \gets \lambda I _ { d } , \quad b \gets 0 _ { d }$   
2: if $w > 1$ then   
3: $( x _ { 1 } , \dots , x _ { w - 1 } ) \gets \mathsf { I n i t } ( T ) , s _ { w } \in S _ { T - w + 1 }$   
4: for $q = 1 , \dots , w - 1$ do   
5: Execute $x _ { q } ,$ observe $Y _ { q }$   
6: $V  V + \dot { x } _ { q } x _ { q } ^ { \top } , \quad b  b + x _ { q } Y _ { q }$   
7: end for   
8: $t \gets w$   
9: else   
10: $s _ { t } \equiv ( ) , t \gets 1$   
11: end if   
12: $k \gets 1$   
13: while $t \leq T$ do   
14: $t _ { k }  \overline { { t } } , \quad W _ { k }  V , \quad \widehat { \theta } _ { k }  W _ { k } ^ { - 1 } b$   
15:   
$\beta _ { k }  R \sqrt { 2 \log \biggl ( \frac { \operatorname* { d e t } ( W _ { k } ) ^ { 1 / 2 } } { \operatorname* { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \biggr ) } + \sqrt { \lambda } S$   
16: Define   
$\begin{array} { r } { \overline { { U } } _ { k } ( x ) \gets \mathrm { c l i p } _ { [ - L S , L S ] } \left( \langle \widehat { \theta } _ { k } , x \rangle + \beta _ { k } \| x \| _ { W _ { k } ^ { - 1 } } \right) } \end{array}$   
17: $n _ { k } \gets T - t _ { k } + 1$   
18: Cache $( \widetilde { x } _ { 1 } ^ { ( k ) } , \dots , \widetilde { x } _ { n _ { k } } ^ { ( k ) } ) \gets \mathcal { P } ( s _ { t _ { k } } , n _ { k } ; \overline { { U } } _ { k } )$   
19: while $t \overset { \bar { < } } { \leq } T$ and de $\cdot ( V ) \leq 2 \operatorname* { d e t } ( W _ { k } )$ do   
20: $x _ { t } \gets \widetilde { x } _ { t - t _ { k } + 1 } ^ { ( k ) }$   
21: Execute $x _ { t } ,$ observe $Y _ { t }$   
22: $V  V + x _ { t } x _ { t } ^ { \top } , ~ b  b + x _ { t } Y _ { t }$   
23: if $w > 1$ then   
24: $s _ { t + 1 } \gets F ( s _ { t } , x _ { t } )$   
25: end if   
26: t ← t + 1   
27: end while   
28: $k \gets k + 1$   
29: end while