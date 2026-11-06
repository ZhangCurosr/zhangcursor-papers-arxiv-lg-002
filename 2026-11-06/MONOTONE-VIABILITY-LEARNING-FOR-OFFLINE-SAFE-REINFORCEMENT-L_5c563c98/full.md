# MONOTONE VIABILITY LEARNING FOR OFFLINE SAFE REINFORCEMENT LEARNING

Ziqian Wang

Zhen Zhang

November 6, 2026

## ABSTRACT

Offline safe reinforcement learning requires reliable long-horizon safety estimates from fixed data. Learning Hamilton–Jacobi (HJ) viability in this setting calls for semantic preservation, convergence, and numerical stability. To address these challenges, we propose Monotone Viability Learning (MoViL), which learns viable sets from offline transitions through monotone finite-horizon set recursion. The recursion preserves finite-horizon safety semantics and converges to the infinite-horizon viability kernel. Binary supervision maintains a fixed target scale across horizons, avoiding discountinduced numerical attenuation. We further establish sufficient conditions for recovering behaviorsupported viable subsets. The learned viability estimates are used to certify actions at deployment. Dubins car experiments validate the numerical accuracy of the learned viable sets. Evaluations on safe RL benchmarks demonstrate state-of-the-art reward–cost performance. MoViL satisfies the average cost constraint on every task and achieves the lowest average cost among the compared methods. These results highlight MoViL’s potential to extend long-horizon HJ viability analysis to safety-critical systems where reliable dynamics models are unavailable and online exploration is unsafe.

## 1 Introduction

Safety is a central requirement for reinforcement learning (RL) in robotics and autonomous systems, where exploratory actions can cause physical damage or irreversible failures [1]. Offline RL learns policies from previously collected data, avoiding additional environment interaction during training [2]. Safe deployment also requires understanding how actions affect future constraint satisfaction. Many safe RL methods impose constraints on expected cumulative costs [3]. These constraints limit cumulative violations but do not directly guarantee safety throughout a trajectory [4]. State-wise safety methods, including those based on control barrier functions, impose explicit conditions for safe operation [5–7]. Safety certification methods assess proposed actions and intervene when necessary during deployment [8, 9]. Learning which states admit persistent safety and which actions preserve safe continuations is therefore an important objective in offline safe RL.

Hamilton–Jacobi (HJ) reachability provides a set-based characterization of persistent safety [10]. Grid-based HJ computation becomes expensive as the state dimension increases [11]. Neural solvers improve the scalability of HJ computation using known system dynamics [12], while RL-based methods enable reachability-related quantities to be learned from sampled transitions [13, 14]. Some discounted formulations achieve contraction by modifying the original reachability semantics [15]. Other HJ formulations preserve exact semantics without establishing a corresponding contractive Bellman formulation [16]. Recent work combines semantic preservation and contraction, with a numerical implementation based on known system dynamics [17]. Neural approximation errors can compromise the reliability of learned safety boundaries [18, 19]. This issue becomes more pronounced when discounting reduces long-horizon value magnitudes, leaving smaller margins for safety classification. Offline viability learning therefore requires a formulation that jointly addresses semantic preservation, convergence, and numerical stability.

In this article, we propose Monotone Viability Learning (MoViL) for offline safe reinforcement learning. Building on classical viability-kernel approximation [20], MoViL learns long-horizon viability from offline transitions through monotone finite-horizon set recursion. At each layer, an observed transition provides a positive training example if it is violation-free and its successor belongs to the viable set estimated at the preceding layer. The learned viability information supports action-level certification during deployment, enabling a simple Gaussian behavior-cloning policy to achieve high returns with low costs. Dubins car experiments directly validate the numerical accuracy of MoViL against an independent HJ reference. Experiments on 17 Bullet-Safety-Gym and MetaDrive tasks from DSRL [21] evaluate the resulting policies across robotic control and autonomous driving settings.

The main contributions of this article are summarized as follows:

1. We propose MoViL, which learns HJ viable sets from offline transitions through monotone finite-horizon recursion and uses the resulting estimates to certify policy actions.

2. We prove that the finite-horizon viable sets converge to the infinite-horizon viability kernel. We also establish sufficient conditions for recovering behavior-supported viable subsets from offline data.

3. We evaluate MoViL through numerical experiments and offline safe RL benchmarks. Dubins car experiments validate the accuracy of the learned viable sets. Across 17 offline safe RL tasks, MoViL achieves state-of the-art reward–cost performance and the lowest average cost.

## 2 Related Work

## 2.1 Hamilton–Jacobi Reachability and Viability Learning

Hamilton–Jacobi (HJ) reachability characterizes long-term safety through reachable sets and viability kernels [10]. To connect this set-based analysis with RL, prior work has expressed safety objectives through Bellman recursions. Discounted Bellman formulations enable safety and reach-avoid values to be learned from sampled transitions [13, 22]. Akametalu et al. [15] formulated reachability using minimum discounted rewards, yielding a contractive Bellman operator with guaranteed fixed-point convergence. Other formulations focus on retaining a closer connection to classica safety analysis. Control barrier-value functions connect HJ reachability with control barrier functions [16]. Travelcost formulations derive an RL-compatible fixed point [23], while non-additive Bellman operators combine contraction with preservation of reachability semantics [17].

These developments have enabled reachability quantities to guide safe policy learning. Reachability-constrained RL uses feasible value functions to identify viable regions [24]. RESPO models the probability of future violations in stochastic systems [25], while FISOR learns state and action feasibility from offline data through discounted Bellman updates and reversed expectile regression [14]. These methods make long-horizon feasibility available to policy optimization, but represent it through recursively fitted scalar values. Discounted recursions can cause value magnitudes to decrease as the horizon grows. In some formulations, the threshold used to classify feasible states does not have an exact viability interpretation. Our method addresses these issues by replacing infinite-horizon value fitting with a monotone recursion over finite-horizon viable sets. Binary membership targets retain explicit set semantics and avoid horizon-dependent target decay, while the nested recursion provides a convergent approximation to the infinite-horizon viability kernel.

## 2.2 Offline Safe Reinforcement Learning

Offline safe reinforcement learning optimizes policies from fixed datasets under safety constraints. Existing methods differ in how they incorporate safety into policy learning and execution. CPQ [26] uses a cost critic to screen actions during policy improvement, whereas COptiDICE [27] optimizes state-action occupancy subject to cost constraints. Sequence models condition action generation on return and cost signals [28, 29]. Generative policies can encode safety information in their latent action spaces [30, 31]. V-OCBF [32] learns a control barrier function and a dynamics model from offline data to formulate a quadratic program for action filtering. During policy learning, explicit safety representations can guide policy extraction [14] or prevent the reward critic from bootstrapping through unsafe transitions [33]. Our method instead uses viability estimates at deployment to certify actions proposed by a separately trained policy.

## 3 Preliminaries

## 3.1 Offline Safe Reinforcement Learning

We consider a constrained Markov decision process (CMDP) $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , r , c , \rho _ { 0 } , \gamma )$ , where and are the state and action spaces, $P ( s ^ { \prime } | s , a )$ denotes the transition dynamics, $r ( s , a )$ and $c ( s , a , s ^ { \prime } ) \geq 0$ denote the reward and safety cost, $\rho _ { 0 }$ is the initial-state distribution, and $\gamma \in ( 0 , 1 )$ is the discount factor. We consider a state space $S \subseteq \mathbb { R } ^ { n }$ and a nonempty, closed, and bounded action space ${ \mathcal { A } } \subseteq \mathbb { R } ^ { \dot { m } }$ . For a policy π, we define

$$
R ( \pi ) = \mathbb { E } _ { \tau \sim \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { t } \right] , \qquad C ( \pi ) = \mathbb { E } _ { \tau \sim \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } c _ { t } \right] .\tag{1}
$$

The safe RL objective considered in this work is

$$
\operatorname* { m a x } _ { \pi } R ( \pi ) \qquad \mathrm { s . t . } \qquad C ( \pi ) = 0 ,\tag{2}
$$

which requires zero safety violation along trajectories since $c _ { t } \geq 0 .$

In the offline setting, learning is based on a pre-collected dataset

$$
\mathcal { D } = \{ \tau _ { i } \} _ { i = 1 } ^ { N } , \qquad \tau _ { i } = \{ ( s _ { t } , a _ { t } , r _ { t } , c _ { t } , s _ { t + 1 } , d _ { t } ) \} _ { t = 0 } ^ { T _ { i } - 1 } ,\tag{3}
$$

where $d _ { t }$ marks a true terminal transition; time-limit truncations are recorded separately.

Assumption 1 (Successor-state safety cost). We assume that the safety cost of a transition is determined by its suc cessor state. Specifically, there exists $\bar { c } : S \to \mathbb { R } _ { \geq 0 }$ such that

$$
c ( s , a , s ^ { \prime } ) = \bar { c } ( s ^ { \prime } ) .\tag{4}
$$

## 3.2 HJ Safety Viability

HJ reachability characterizes persistent safety through the set of states from which safety violations can be avoided indefinitely. We study viability for the discrete-time transition system, with safety evaluated at each sampled state. For a policy π, define

$$
V _ { h } ^ { \pi } ( s ) = \operatorname* { P r } _ { \pi , P } \left( \exists t \geq 0 : \bar { c } ( s _ { t } ) > 0 | s _ { 0 } = s \right) ,\tag{5}
$$

and let

$$
V _ { h } ^ { * } ( s ) = \operatorname* { i n f } _ { \pi } V _ { h } ^ { \pi } ( s ) .\tag{6}
$$

The infinite-horizon viability kernel is

$$
K _ { \infty } = \{ s \in { \mathcal { S } } : V _ { h } ^ { * } ( s ) = 0 \} .\tag{7}
$$

## 4 Proposed Method

The proposed method combines monotone viability learning with action-level certification and policy extraction. State classifiers recursively identify finite-horizon viable regions from offline transitions. The resulting continuation labels supervise an action-conditioned safety estimator. A Gaussian behavior-cloning policy supplies nominal actions, and a cost-based fallback selects among candidate actions when the nominal action fails certification.

## 4.1 Monotone Viability Formulation

We seek convergence, semantic preservation, and numerical stability in infinite-horizon safety learning. Our approach shifts from fitting an infinite-horizon value function to recursively learning nested finite-horizon viable sets. The recursion preserves finite-horizon safety semantics and admits a monotone convergence analysis, with binary supervision maintaining a horizon-independent target scale.

Under Assumption 1, define the instantaneous safe set as

$$
{ \mathcal { K } } _ { 0 } = \{ s \in S : { \bar { c } } ( s ) = 0 \} .\tag{8}
$$

Classical viability-kernel algorithms use backward set iterations [20, 34]. For a measurable set $\kappa ,$ its controlled predecessor is

$$
\operatorname* { P r e } ( K ) = \{ s : \exists a \in { \cal A } , P ( K \mid s , a ) = 1 \} .\tag{9}
$$

![](images/aa572d43f3261ec9ce69e8b0098fcd19e2a040db9b06d1857b86fa1311c6d76c.jpg)  
Figure 1: Monotone refinement of finite-horizon viable sets. As the horizon increases, the green sets $\kappa _ { k }$ shrink toward $\kappa _ { \infty } ,$ progressively excluding states without sufficiently long safe continuations. Arrows indicate state transitions.

For deterministic transitions $s ^ { \prime } = f ( s , a )$ , this condition reduces to $f ( s , a ) \in \mathcal { K }$ . Starting from $\kappa _ { 0 } .$ , we construct

$$
\begin{array} { r } { \mathcal { K } _ { k + 1 } = \mathcal { K } _ { k } \cap { \mathrm { P r e } } ( \mathcal { K } _ { k } ) , \qquad k \geq 0 . } \end{array}\tag{10}
$$

The set $\kappa _ { k }$ contains initially safe states from which an admissible policy can maintain safety almost surely for the next k transitions. Along this recursion, (10) is equivalent to ${ \ K } _ { k + 1 } \ \dot { = } \ \dot { \mathcal { K } } _ { 0 } \cap \operatorname* { P r e } ( { \mathcal { K } _ { k } } )$ . Fig. 1 illustrates this nested refinement toward the infinite-horizon viability kernel.

Theorem 1 (Convergence of monotone viability recursion). Suppose $\kappa _ { 0 }$ is closed, is nonempty and compact, and $P$ is weakly continuous in $( s , a )$ . Then the recursion in (10) satisfies

$$
\mathcal { K } _ { k + 1 } \subseteq \mathcal { K } _ { k } , \qquad \bigcap _ { k \geq 0 } \mathcal { K } _ { k } = \mathcal { K } _ { \infty } .\tag{11}
$$

Moreover, $\kappa _ { \infty }$ is the largest controlled invariant subset of $\cdot \kappa _ { 0 }$ and satisfies

$$
K _ { \infty } = \mathcal { K } _ { 0 } \cap \operatorname { P r e } ( K _ { \infty } ) .\tag{12}
$$

Proof. Nesting follows from (10). For closed , weak continuity makes $( s , a ) \mapsto P ( K \mid s , a )$ upper semicontinuous. Compactness of then ensures that Pre( ) is closed. Let $\begin{array} { r } { \mathcal { K } _ { * } = \hat { \bigcap _ { k } } \mathcal { K } _ { k } } \end{array}$ . For any $s \in \kappa _ { * }$ <sub>∗</sub>, the sets

$$
\mathcal { A } _ { k } ( s ) = \{ a \in \mathcal { A } : P ( \mathcal { K } _ { k } \mid s , a ) = 1 \}
$$

are nonempty, compact, and nested. Hence, an action $a _ { * } \in \bigcap _ { k } { \mathcal { A } } _ { k } ( s )$ exists, and continuity of probability from above gives

$$
P ( \mathcal { K } _ { * } \mid s , a _ { * } ) = \operatorname* { l i m } _ { k \to \infty } P ( \mathcal { K } _ { k } \mid s , a _ { * } ) = 1 .
$$

The correspondence $\textstyle s \mapsto \bigcap _ { k > 0 } { \mathcal { A } } _ { k } ( s )$ has a closed graph and nonempty compact values on $\boldsymbol { \kappa } _ { * } . \mathrm { A }$ measurable selector therefore defines a stationary policy that remains in $\kappa ,$ almost surely [35]. Conversely, every indefinitely safe state belongs to each finite-horizon viable set. The stated assumptions ensure attainment of finite-horizon safety optima, so $V _ { h } ^ { * } ( s ) = 0$ also implies membership in every $\kappa _ { k }$ . Thus, ${ \kappa _ { * } = \kappa _ { \infty } }$ . Every controlled invariant subset of $\kappa _ { 0 }$ is contained in all $\kappa _ { k }$ by induction, establishing maximality. Monotonicity of Pre then gives (12). □

We next approximate this set recursion from offline transitions.

## 4.2 Offline Monotone Viability Learning

Let $\pi _ { \beta } ( \cdot \mid s )$ denote the behavior distribution underlying the offline training dataset ${ \mathcal { D } } _ { \mathrm { t r } }$ . State learning uses trajectorylevel fitting and calibration splits, denoted by ${ \mathcal { D } } _ { \mathrm { f i t } }$ and $\mathcal { D } _ { \mathrm { c a l } }$ . Let $( s , a , r , c , \bar { s ^ { \prime } } , d )$ denote a transition from ${ \mathcal { D } } _ { \mathrm { t r } }$ , and let z indicate a positive safety cost or a task-defined physical-failure terminal. Time-limit truncation alone is not a violation. For $\kappa \subseteq \bar { \kappa } _ { 0 }$ , define

$$
p _ { \mathcal { K } } ( s ) = \mathbb { E } _ { a \sim \pi _ { \beta } ( \cdot \vert s ) } \left[ \operatorname* { P r } ( z = 0 , s ^ { \prime } \in \mathcal { K } \mid s , a ) \right] .\tag{13}
$$

Under deterministic transitions, $p \kappa ( s )$ is the behavior probability mass of actions reaching  without a violation. We initialize $\widehat { \kappa } _ { 0 } = \kappa _ { 0 }$ and write $b _ { k } ( s ) = \mathbf { 1 } _ { \widehat { K } _ { k } } ( s )$

At layer $k \geq 1$ , we freeze the preceding membership function $b _ { k - 1 }$ and construct the binary target

$$
Y _ { k } = ( 1 - z ) b _ { k - 1 } ( s ^ { \prime } ) .\tag{14}
$$

Safety targets also use the recorded successor membership at nonviolating terminal or truncated records. The condi tional success probability is

$$
p _ { k } ( s ) = \operatorname* { P r } ( Y _ { k } = 1 \mid s ) = p _ { \widehat { K } _ { k - 1 } } ( s ) .\tag{15}
$$

Under deterministic transitions, maximizing binary successor feasibility over actions tests whether a feasible action exists. A positive $p _ { k } ( s )$ means that feasible actions have positive probability mass under the behavior distribution, giving a behavior-supported counterpart to this existence condition.

In monotone set refinement, erroneous exclusions are irreversible and reduce predecessor continuation support, potentially propagating further exclusions. Falsely retained states remain eligible for removal in later layers. This asymmetry motivates witness protection and a minimum-support threshold.

Recorded trajectories provide additional evidence of safe continuations. Let L denote the number of consecutive violation-free transitions starting from the sampled record within its recorded trajectory. We define

$$
\begin{array} { c } { { w _ { k } = { \bf 1 } _ { \{ s \in \mathcal { K } _ { 0 } , ~ { \cal L } \geq k \} } , } } \\ { { \widetilde { Y } _ { k } = \operatorname* { m a x } \{ Y _ { k } , w _ { k } \} . } } \end{array}\tag{16}
$$

Let $\mathcal { W } _ { k }$ consist of recorded states with $w _ { k } = 1$ . The witness sets are nested, $\mathcal { W } _ { k + 1 } \subseteq \mathcal { W } _ { k }$ . Under deterministic transitions, each observed safe segment establishes finite-horizon viability, giving $\mathcal { W } _ { k } \subseteq \mathcal { K } _ { k }$

$Y _ { k }$ uses the preceding memberships after witness protection. In the recursion above, linked safe records with $w _ { k } = 1$ already have $Y _ { k } = 1$ , so the explicit maximum is a redundant label safeguard retained from the implementation. The union in (22) protects memberships against rejection by the newly fitted network.

A state network $H _ { k } : S \to [ 0 , 1 ]$ minimizes the expectile-weighted binary cross-entropy

$$
\begin{array} { r l } & { \mathcal { L } _ { H _ { k } } = - \mathbb { E } _ { \mathcal { D } _ { \mathrm { f i t } } } \big [ \tau \widetilde { Y } _ { k } \log H _ { k } ( s ) } \\ & { ~ + ( 1 - \tau ) ( 1 - \widetilde { Y } _ { k } ) \log ( 1 - H _ { k } ( s ) ) \big ] , } \end{array}\tag{17}
$$

where $\tau \in ( 1 / 2 , 1 )$

For a Bernoulli target with success probability $p ,$ the population minimizer is

$$
h _ { \tau } ( p ) = \frac { \tau p } { \tau p + ( 1 - \tau ) ( 1 - p ) } .\tag{18}
$$

For binary targets, this is also the population minimizer of asymmetric squared expectile loss [36], and $\begin{array} { r } { \operatorname* { l i m } _ { \tau \uparrow 1 } h _ { \tau } ( p ) = } \end{array}$ $\mathbf { 1 } _ { \{ p > 0 \} }$ . This limit connects high-expectile learning to the identification of positively supported continuations. Let ${ \widetilde { p } } _ { k } ( s ) = \operatorname* { P r } ( { \widetilde { Y } } _ { k } = 1 \mid s )$ denote the success probability of the witness-corrected target. It equals $p _ { k } ( s )$ whenever the safeguard leaves the conditional labels unchanged. For $0 < \widetilde { p } _ { k } ( s ) < 1$ , the optimal network output satisfies

$$
\mathrm { l o g i t } H _ { k } ^ { * } ( s ) = \mathrm { l o g i t } \widetilde { p } _ { k } ( s ) + \mathrm { l o g i t } \tau .\tag{19}
$$

Following the general principle of post-hoc probability calibration [37], we fit a scalar offset $\beta _ { k }$ by ordinary binary cross-entropy against $\ddot { \Gamma } _ { k }$ on $\mathcal { D } _ { \mathrm { c a l } }$ . The calibrated probability estimate is

$$
\widehat { p } _ { k } ( s ) = \mathrm { s i g m o i d } ( \log \mathrm { i t } H _ { k } ( s ) - \log \mathrm { i t } \tau + \beta _ { k } ) .\tag{20}
$$

For a minimum positive-label probability $p _ { \operatorname* { m i n } } \in ( 0 , 1 )$ , the equivalent network threshold is

$$
t _ { k } = \mathrm { s i g m o i d } ( \log \mathrm { i t } p _ { \mathrm { m i n } } + \log \mathrm { i t } \tau - \beta _ { k } ) .\tag{21}
$$

τ controls the fitting emphasis on positive labels; $p _ { \mathrm { m i n } }$ controls threshold-based retention by calibrated positive-label probability. By construction, $H _ { k } ( s ) \geq t _ { k }$ is equivalent to ${ \widehat { p } } _ { k } ( s ) \geq p _ { \operatorname* { m i n } } ;$ with exact probability recovery, this becomes $\widetilde { p } _ { k } ( s ) \geq p _ { \operatorname* { m i n } } .$ , independently of τ.

The membership update combines thresholding with witness protection,

$$
\widehat { \mathcal { K } } _ { k } = \Big ( \widehat { \mathcal { K } } _ { k - 1 } \cap \{ s : \widehat { p } _ { k } ( s ) \geq p _ { \operatorname* { m i n } } \} \Big ) \cup \mathcal { W } _ { k } .\tag{22}
$$

The first layer preserves nesting because $\mathcal { W } _ { 1 } \subseteq \mathcal { K } _ { 0 } = \widehat { \mathcal { K } } _ { 0 }$ . For $k \geq 2 , \mathcal { W } _ { k } \subseteq \mathcal { W } _ { k - 1 } \subseteq \widehat { \mathcal { K } } _ { k - 1 }$ gives the same property. Binary membership labels $b _ { k }$ are updated for recorded states and their successors to construct the next layer’s training targets. The set inclusions in Theorem 2 apply to the recursively defined sets $\widehat { \mathcal { K } } _ { k }$ , while $H _ { k }$ provides a neural approximation of state viability.

Theorem 2 (Behavior-supported viability recovery). Suppose the conditions of Theorem 1 hold and transitions are deterministic. Let ${ \mathcal { C } } \subseteq { \mathcal { K } } _ { 0 } ^ { \bar { 2 } }$ be measurable and satisfy

$$
p _ { \mathcal { C } } ( s ) \geq \alpha > 0 , \qquad s \in \mathcal { C } .\tag{23}
$$

Here, α is a uniform lower bound on the behavior probability of a violation-free transition that remains in . Assume that each layer satisfies

$$
\begin{array} { r } { \widetilde { p } _ { k } ( s ) - \varepsilon _ { - } \le \widehat { p } _ { k } ( s ) \le \widetilde { p } _ { k } ( s ) + \varepsilon _ { + } , } \end{array}\tag{24}
$$

for all $s \in \widehat { \mathcal { K } } _ { k - 1 }$ , where $\varepsilon _ { - } , \varepsilon _ { + } \geq 0 . \ I f \varepsilon _ { + } < p _ { \mathrm { m i n } } \leq \alpha - \varepsilon _ { - }$ , then

$$
\mathcal { C } \subseteq \widehat { \mathcal { K } } _ { k } \subseteq \mathcal { K } _ { k } , \qquad k \geq 0 .\tag{25}
$$

Consequently,

$$
c \subseteq \bigcap _ { k \geq 0 } { \widehat { \mathcal { K } } } _ { k } \subseteq { \mathcal { K } } _ { \infty } .\tag{26}
$$

If (23) holds with ${ \mathcal { C } } = \kappa _ { \infty } ,$ , the limiting learned set equals $\kappa _ { \infty }$

Proof. The claim holds at initialization. Suppose $\mathcal { C } \subseteq \widehat { \mathcal { K } } _ { k - 1 } \subseteq \mathcal { K } _ { k - 1 }$ . For $s \in { \mathcal { C } }$

$$
\widehat { p } _ { k } ( s ) \geq \widetilde { p } _ { k } ( s ) - \varepsilon _ { - } \geq p c ( s ) - \varepsilon _ { - } \geq p _ { \operatorname* { m i n } } ,
$$

so s is retained. For $s \in \widehat { K } _ { k - 1 } \backslash \mathcal { K } _ { k }$ , no action enters $\widehat { \mathcal { K } } _ { k - 1 }$ and no valid k-step witness exists. Therefore, $\widetilde { p } _ { k } ( s ) = 0$ and $\widehat { p } _ { k } ( s ) \leq \varepsilon _ { + } < p _ { \operatorname* { m i n } }$ , excluding s. Every protected witness belongs to $\kappa _ { k }$ . Induction proves (25). Taking intersections and applying Theorem 1 gives the limiting inclusions. □

The lower bound on $p _ { \mathrm { m i n } }$ prevents states with $\widetilde { p } _ { k } = 0$ from passing the threshold under bounded overestimation; the upper bound retains the behavior-supported viable subset. The interval is nonempty when $\alpha > \varepsilon _ { - } + \varepsilon _ { + }$

On fixed validation states $ { S _ { \mathrm { v a l } } }$ held out from fitting and calibration, let $u _ { k }$ be the fraction passing all layer thresholds $H _ { \ell } ( s ) \geq t _ { \ell }$ for $\ell = 1 , \ldots , k$ . Since $u _ { k }$ is nonincreasing, its range over a window of w $\geq 2$ layers is

$$
\Delta _ { k } ^ { ( w ) } = u _ { k - w + 1 } - u _ { k } .\tag{27}
$$

The recursion terminates at layer K once the minimum layer count is met and this range stays within tolerance for the prescribed number of consecutive windows; otherwise, it stops at the maximum layer count. Fitting losses and witness-protected label retention are monitored as diagnostics.

## 4.3 Action-Level Safety Certification

After the state recursion terminates at layer K, we train two independently initialized action-conditioned networks, $Q _ { H } ^ { ( 1 ) }$ and $Q _ { H } ^ { ( 2 ) }$ , using the target

$$
Y ^ { Q } = ( 1 - z ) b _ { K - 1 } ( s ^ { \prime } ) .\tag{28}
$$

The predecessor membership $b _ { K - 1 }$ matches the one-step continuation condition used to train the final state layer $H _ { K }$ Each network minimizes ordinary binary cross-entropy over the offline training transitions,

$$
\mathcal { L } _ { Q _ { H } } ^ { ( j ) } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { t r } } } \left[ \mathrm { B C E } \Big ( Q _ { H } ^ { ( j ) } ( s , a ) , Y ^ { Q } \Big ) \right] , \qquad j = 1 , 2 .\tag{29}
$$

The action-level safety score is

$$
Q _ { H } ( s , a ) = \operatorname* { m i n } _ { j \in \{ 1 , 2 \} } Q _ { H } ^ { ( j ) } ( s , a ) .\tag{30}
$$

An action passes certification when $Q _ { H } ( s , a ) \geq t _ { Q }$ , where $t _ { Q }$ is the action certification threshold.

The conditional success probability of $Y ^ { Q }$ is

$$
q _ { K } ( s , a ) = \operatorname* { P r } \Bigl ( z = 0 , s ^ { \prime } \in \widehat { K } _ { K - 1 } \mid s , a \Bigr ) .\tag{31}
$$

The uncorrected state target averages over behavior actions,

$$
p _ { K } ( s ) = \mathbb { E } _ { a \sim \pi _ { \beta } ( \cdot | s ) } [ q _ { K } ( s , a ) ] .
$$

$Q _ { H }$ estimates action-specific success probabilities under stochastic transitions; Theorem 2 assumes deterministic transitions.

## 4.4 Certification-Guided Policy Extraction

We extract a Gaussian behavior-cloning policy from high-return, low-cost trajectories. At execution, the nominal mean action is used if it passes action certification. Otherwise, a cost-basedfallback selects the action with the lowest predicted cost from a candidate set containing the nominal action and perturbed actions.

For each training trajectory $\tau _ { i } .$ , define its undiscounted return and cost,

$$
R _ { i } ^ { \mathrm { t r a j } } = \sum _ { t = 0 } ^ { T _ { i } - 1 } r _ { i , t } , \qquad C _ { i } ^ { \mathrm { t r a j } } = \sum _ { t = 0 } ^ { T _ { i } - 1 } c _ { i , t } .\tag{32}
$$

Among trajectories satisfying $C _ { i } ^ { \mathrm { t r a j } } \leq C _ { \mathrm { B C } }$ , we retain the highest-return fraction $\eta _ { \mathrm { B C } }$ . Their state–action pairs form the behavior-cloning dataset $\mathcal { D } _ { \mathrm { B C } }$

Let $g$ denote the componentwise affine map from normalized action coordinates $[ - 1 , 1 ] ^ { m }$ to the environment action bounds. In normalized coordinates, the Gaussian policy is

$$
\pi _ { \boldsymbol { \theta } , \boldsymbol { \sigma } } ^ { \mathrm { B C } } ( \widetilde { \boldsymbol { a } } \mid \boldsymbol { s } ) = \mathcal { N } \big ( \widetilde { \boldsymbol { a } } ; \pmb { \mu } _ { \boldsymbol { \theta } } ( \boldsymbol { s } ) , \mathrm { d i a g } ( \pmb { \sigma } ^ { 2 } ) \big ) ,\tag{33}
$$

where $\mu _ { \theta } ( s )$ is the state-dependent mean and $\sigma$ is a learned vector of standard deviations shared across states. Both are trained by minimizing

$$
\mathcal { L } _ { \mathrm { B C } } ( \boldsymbol { \theta } , \boldsymbol { \sigma } ) = - \mathbb { E } _ { ( s , a ) \sim \mathcal { D } _ { \mathrm { B C } } } \left[ \log \pi _ { \boldsymbol { \theta } , \boldsymbol { \sigma } } ^ { \mathrm { B C } } \bigl ( g ^ { - 1 } ( a ) \mid s \bigr ) \right] .\tag{34}
$$

To construct candidate actions around the policy mean, let ${ { \epsilon } _ { 0 } } \mathrm { ~ = ~ } 0$ and fix Gaussian perturbations $\epsilon _ { 1 } , \ldots , \epsilon _ { M } \ \sim$ $\mathcal { N } ( 0 , I )$ across decisions. For $j = 0 , \ldots , M$ , define the candidate actions

$$
a _ { j } ( s ) = g ( \mathrm { c l i p } ( \pmb { \mu } _ { \theta } ( s ) + \pmb { \sigma } \odot \epsilon _ { j } , - 1 , 1 ) ) .\tag{35}
$$

The nominal action $a _ { 0 } ( s )$ is the clipped mean mapped to environment coordinates.

The fallback ranks actions using a cost critic $\displaystyle Q _ { c } ,$ , trained with a state-value network $V _ { c }$ on ${ \mathcal { D } } _ { \mathrm { t r } }$ using the common target

$$
y ^ { c } = c + \gamma _ { c } ( 1 - d ) \bar { V } _ { c } ( s ^ { \prime } ) .\tag{36}
$$

Both networks minimize squared error to $y ^ { c }$ , using $( s , a )$ for $Q _ { c }$ and s for $V _ { c }$ . Here, $\gamma _ { c }$ is the cost discount factor, and $\hat { V } _ { c }$ denotes the output of a slowly updated target copy of $V _ { c } .$ . The state-value regression averages logged action outcomes at each state, so $Q _ { c }$ estimates the cost of taking an action and then following the behavior policy $\pi _ { \beta }$

Let $\begin{array} { r } { j ^ { \ast } ( s ) = \arg \operatorname* { m i n } _ { j \in \{ 0 , \dots , M \} } Q _ { c } ( s , a _ { j } ( s ) ) } \end{array}$ denote the index selected by the cost critic. The deployed policy is

$$
\pi _ { \mathrm { e x e c } } ( s ) = \left\{ \begin{array} { l l } { a _ { 0 } ( s ) , } & { Q _ { H } \bigl ( s , a _ { 0 } ( s ) \bigr ) \geq t _ { Q } , } \\ { a _ { j ^ { \ast } ( s ) } ( s ) , } & { Q _ { H } \bigl ( s , a _ { 0 } ( s ) \bigr ) < t _ { Q } , } \end{array} \right.\tag{37}
$$

Algorithm 1 summarizes the offline training procedure and policy execution.

Algorithm 1 Monotone Viability Learning (MoViL)   
Require: Offline data $\overline { { \mathcal { D } _ { \mathrm { t r } } } } .$ , validation states $\overline { { S _ { \mathrm { v a l } } } } .$ , maximum depth $\overline { { K _ { \mathrm { m a x } } } }$   
Offline training   
1: Prepare $\begin{array} { r } { \mathcal { D } _ { \mathrm { f i t } } , \mathcal { D } _ { \mathrm { c a l } } , } \end{array}$ , and trajectory witnesses; initialize $b _ { 0 } = \mathbf { 1 } _ { \kappa _ { 0 } } .$   
2: for $\mathbf { \bar { \xi } } k = 1 , \dots , K _ { \operatorname* { m a x } }$ do   
3: Construct $Y _ { k } = ( 1 - z ) b _ { k - 1 } ( s ^ { \prime } )$ and witness-corrected targets $\widetilde { Y } _ { k }$   
4: Train $H _ { k }$ by weighted BCE and calibrate $\beta _ { k } , t _ { k }$   
5: Update $b _ { k }$ using (22).   
6: if the stopping criterion on $S _ { \mathrm { v a l } }$ is met then   
7: break   
8: end if   
9: end for   
10: Set K to the final recursion depth.   
11: Train twin $Q _ { H }$ with targets $Y ^ { \hat { Q } } = ( 1 - z ) b _ { K - 1 } ( s ^ { \prime } )$   
12: Construct $\mathcal { D } _ { \mathrm { B C } }$ by trajectory filtering and train Gaussian BC.   
13: Train $Q _ { c } , V _ { c }$ on $\mathcal { D } _ { \mathrm { t r } } .$   
Policy execution at state s   
14: Compute the nominal action $a _ { 0 } ( s )$   
15: if $Q _ { H } ( s , a _ { 0 } ( s ) ) \geq t _ { Q }$ then   
16: Execute $a _ { 0 } ( s ) .$   
17: else   
18: Generate candidates $\{ a _ { j } ( s ) \} _ { j = 0 } ^ { M }$ using fixed Gaussian perturbations   
19: Execute $a _ { j ^ { * } } ( s )$ , where $j ^ { * } = \arg$ min<sub>j</sub> $Q _ { c } ( s , a _ { j } ( s ) )$ .   
20: end if

Table 1: Experimental settings for the Dubins car problem.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Offline transitions</td><td>200,000</td></tr><tr><td>Maximum trajectory length</td><td>64</td></tr><tr><td>HJ reference grid</td><td> $9 1 \times 9 1 \times 4 0$ </td></tr><tr><td> $H _ { k }$  network</td><td>2 hidden layers, width 192</td></tr><tr><td>Activation function</td><td>ReLU</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate / batch size</td><td> $3 \times 1 0 ^ { - 4 } / 7 6 8$ </td></tr><tr><td> $\tau / p _ { \mathrm { m i n } }$ </td><td>0.995 / 0.1</td></tr><tr><td>Validation / evaluation states</td><td>5, 000 / 60, 000</td></tr><tr><td>Convergence: w / tolerance / checks</td><td>8 / 0.005 / 5</td></tr></table>

## 5 Numerical Validation of Viability Learning

## 5.1 Experimental Setup

We consider a Dubins car navigating around a circular obstacle, with dynamics

$$
\dot { x } = v \cos \theta , \qquad \dot { y } = v \sin \theta , \qquad \dot { \theta } = u ,\tag{38}
$$

where $( x , y )$ denotes the position, θ is the heading, and $u \in [ - 1 , 1 ]$ is the steering control. We set $v = 1$ and use a control interval of 0.1. The obstacle is centered at the origin with radius 0.8. A transition is unsafe if its successor state lies inside the obstacle.

We compare our method with FVI [17] and FISOR [14]. FVI applies a non-additive Bellman operator using known system dynamics, whereas FISOR learns feasibility through discounted Bellman updates. FISOR and our method use the same fixed offline dataset, which combines uniformly sampled and near-obstacle initial states with steering actions sampled uniformly or at their bounds. We compute an independent numerical HJ reference using semi-Lagrangian recursion solely for evaluation.

The main experimental settings are summarized in Table 1. We evaluate viable-set approximation against the numerical HJ reference using intersection over union (IoU), false positive rate (FPR), and false negative rate (FNR), treating safe states as the positive class. For evaluation, positions are drawn from an equal mixture of uniform sampling over $[ - 3 , 3 ] ^ { 2 }$ and annular sampling around the obstacle, with headings sampled uniformly from $[ - \pi , \pi ]$ . States are classified as safe when $H _ { k } ( s ) \geq t _ { k }$ for our method, $V ( s ) \geq 0$ for FVI, or $V _ { h } ( s ) \leq 0$ for FISOR. The remaining hyperparameters of FVI and FISOR follow the settings reported in their respective original papers. Quantitative results are reported as the mean and standard deviation over three random seeds. All experiments run on a workstation with an Intel i9-14900KF CPU and an NVIDIA RTX 2080 Ti GPU.

![](images/e833e282d5afeb59f70414fe89e10d8c36644ddb670ce06e70cecabd63905861.jpg)

![](images/9ad12e87976024b5a9a5cd3f36ebf995e14eba80250140b0b1e022724988f271.jpg)

![](images/139b5e4ab358fb8369c0afdce5e7652fb4e486d372357d19db1c7433ebeacb41.jpg)

Figure 2: Viability approximation on the $\theta = 0$ slice of the Dubins car state space. (a) MoViL (Ours), (b) FVI, and (c) FISOR. Colors represent the learned function values. Solid and dashed lines indicate the numerical HJ reference and predicted viability boundaries, respectively.  
![](images/2f27624d82c1c1749c5e6eb05e6734b65bea2ae842ef7d773d2675e5a8e93156.jpg)  
Figure 3: Evolution of viability learning at layers $k = 1 , 2 , 4 , 8 , 2 0$ on the θ = 0 slice. Each panel shows the currentlayer prediction $H _ { k }$ and its calibrated threshold. Solid and dashed lines indicate the numerical HJ reference and predicted boundaries, respectively.

## 5.2 Viability Learning Results

Table 2 summarizes the viability approximation accuracy and training time of the three methods. Our method achieves the highest IoU and lowest FNR, with a slightly higher FPR than FVI. FVI attains comparable approximation accuracy but requires approximately 65 times as much training time under its original training schedule.

Fig. 2 compares the learned viability functions and predicted boundaries of the three methods with the numerical HJ reference on the $\theta = 0$ slice. Fig. 2(a) shows scores near one in viable regions and near zero in non-viable regions, with a sharp transition across the boundary. This separation reflects the binary viability classification learned by our method. The predicted boundary closely follows the numerical HJ reference. FVI also closely matches the reference boundary (Fig. 2(b)). Its value function preserves reachability semantics through its sign [17]. Exponential discounting in its non-additive Bellman operator causes long-horizon value magnitudes to decay toward zero. The resulting smal numerical margins make sign-based decisions sensitive to network fitting errors, posing a challenge as the relevant safety horizon grows. FISOR employs continuous value supervision in its discounted Bellman update, rather than binary safety labels [14]. This method uses the zero-level set of the value function as the feasibility determination boundary, but the zero threshold does not strictly correspond to the boundary of the infinite time-domain feasible region. As shown in Fig. 2(c), FISOR classifies a large portion of feasible regions as infeasible, exhibiting significant conservatism.

Fig. 3 illustrates the refinement of the learned viable set across layers. As the horizon increases, the predicted viable region progressively contracts toward the numerical HJ reference while maintaining a sharp separation between viable and non-viable regions.

<sup>Table</sup> <sup>2:</sup> <sup>Viability</sup> <sup>approximation</sup> <sup>results</sup> <sup>on</sup> <sup>the</sup> <sup>Dubins</sup> <sup>car</sup> <sup>(mean</sup> ± <sup>standard</sup> <sup>deviation</sup> <sup>over</sup> <sup>three</sup> <sup>seeds).</sup>
<table><tr><td>Method</td><td>IoU (%) ↑</td><td> $\mathbf { F P R } \left( \% \right) \downarrow$ </td><td> $\mathbf { F N R } \left( \% \right) \downarrow$ </td><td> $\mathbf { T i m e } \left( \mathbf { s } \right) \downarrow$ </td></tr><tr><td>Ours</td><td> $\mathbf { 9 9 . 3 4 \pm 0 . 2 5 }$ </td><td> $0 . 4 2 { \pm } 0 . 2 8$ </td><td> $\mathbf { 0 . 6 2 \pm 0 . 2 8 }$ </td><td> $\mathbf { 4 5 . 9 4 \pm 2 . 6 8 }$ </td></tr><tr><td>FISOR</td><td> $9 4 . 4 8 { \pm } 0 . 2 3 $ </td><td> $\mathbf { 0 . 0 0 { \overset { . } { = } } 0 . 0 0 }$ </td><td> $5 . 5 2 { \pm } 0 . 2 3 $ </td><td> $8 4 . 4 0 { \pm } 1 . 9 5 $ </td></tr><tr><td>FVI</td><td> $9 9 . 0 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 2 1 { \pm } 0 . 0 5$ </td><td> $0 . 9 1 { \pm } 0 . 0 9$ </td><td> $2 9 8 5 . 3 8 { \pm } 5 . 0 7$ </td></tr></table>

![](images/9ef730bedd4680a585914af967021cb094b37ee662ec1c06c38fa16d654d9267.jpg)

![](images/53d80d9f75bdc932f947bd5765450889497692c2f2f4b0a35397e33da1f86233.jpg)  
Figure 4: IoU (left) and classification errors (right) across recursion layers on the Dubins car problem. Lines and shading indicate the mean and 1 standard deviation over three seeds. One run converged at $k = 2 0$ and the other two at $k = 2 3$ . The first run’s final values are reused when computing statistics at $k = 2 1 { - } 2 3$

Fig. 4 shows the corresponding IoU and classification errors across recursion layers. The mean IoU exceeds 0.99 at $k = 8$ and reaches a high-accuracy plateau around $k = 1 0$ . FPR decreases sharply as the recursion deepens, while FNR remains low throughout. This pattern is consistent with the design of MoViL: most reference-viable states are retained, while states without sufficiently long safe continuations are progressively excluded.

## 6 Offline Safe Reinforcement Learning Experiments

## 6.1 Experimental Setup

We evaluate MoViL on eight Bullet-Safety-Gym tasks [38] and nine MetaDrive tasks [39] from the DSRL benchmark [21]. Bullet-Safety-Gym includes four agents (Ball, Car, Drone, and Ant) performing Run and Circle tasks. Run rewards forward motion within a corridor, with safety costs for boundary violations and excessive speed. Circle rewards clockwise motion along a circular track, with costs for leaving the designated safe region. MetaDrive evaluates autonomous driving across three road complexity levels (easy, medium, hard) and three traffic densities (sparse, mean, dense), with costs for collisions and off-road driving. In both task suites, transition safety costs depend only on successor states, satisfying Assumption 1. We report normalized return and normalized cost, with a cost limit of 10 for all tasks. A normalized cost no greater than 1 indicates constraint satisfaction. Results are averaged over three random seeds, with 20 evaluation episodes per seed.

Table 3 lists the key hyperparameters for MoViL. All networks use the listed architecture, and the $Q _ { H }$ training budget applies to each head. BC uses cosine learning-rate decay. The remaining shared hyperparameters and hardware configuration follow the Dubins car setup in Section 5.1.

## 6.2 Baselines

We compare MoViL with six baselines: (1) BCQL: A Lagrangian extension of BCQ [40] for cost-constrained policy optimization. (2) CPQ [26]: A Q-learning method that penalizes unsafe and out-of-distribution actions through conservative cost estimation. (3) CDT [28]: A Decision Transformer conditioned on target returns and costs. (4) FISOR [14]: A feasibility-guided diffusion method based on HJ reachability. (5) LSPC [30]: A policy optimization method with latent safety constraints learned through a conditional variational autoencoder. (6) GSCO [31]: A normalizing-flow method that controls policy distribution shift through geometric constraints in latent space.

Table 3: Key hyperparameters for the offline safe RL experiments.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Network architecture</td><td>2 hidden layers, width 256</td></tr><tr><td>Batch size</td><td>512</td></tr><tr><td> $H _ { k }$  steps: first / later layers</td><td>10, 000 / 2, 000</td></tr><tr><td>Recursion layers: min. / max.</td><td>20 / 100</td></tr><tr><td>Convergence-validation states</td><td>4,096</td></tr><tr><td>QH / BC / cost training steps</td><td>10, 000 / 20, 000 / 100, 000</td></tr><tr><td>BC cost limit  $C _ { \mathrm { B C } } /$  fraction ηBC</td><td>10 / 0.2</td></tr><tr><td>Cost discount  $\gamma _ { c }$ </td><td>0.95</td></tr><tr><td>Certification threshold  $t _ { Q }$ </td><td>0.99</td></tr><tr><td>Action candidates M  $+ 1$ </td><td>10</td></tr></table>

Table 4: Performance comparison on Bullet-Safety-Gym and MetaDrive tasks. indicates higher is better, and indicates lower is better. Bold: safe policy; Gray: unsafe policy; Bold blue: best safe policy; Bold: second-best safe policy. Safety is defined by normalized cost 1, and rankings are based on reward.
<table><tr><td rowspan="2">Task</td><td colspan="2">BCQL</td><td colspan="2">CPQ</td><td colspan="2">CDT</td><td colspan="2">FISOR</td><td colspan="2">LSPC</td><td colspan="2">GSCO</td><td colspan="2">MoViL</td></tr><tr><td>reward ↑</td><td>cost ↓</td><td>reward ↑</td><td>cost ↓</td><td>reward ↑</td><td>cost ↓</td><td>reward ↑</td><td>cost ↓</td><td>reward ↑</td><td>cost ↓</td><td>reward ↑</td><td>cost ↓</td><td>reward ↑</td><td>cost ↓</td></tr><tr><td colspan="2">Bullet-Safety-Gym</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AntRun</td><td>0.05</td><td>4.63</td><td>0.13</td><td>0.01</td><td>0.69</td><td>1.24</td><td>0.45</td><td>0.76</td><td>0.94</td><td>1.46</td><td>0.60</td><td>0.13</td><td>0.61</td><td>0.06</td></tr><tr><td>BallRun</td><td>0.35</td><td>0.20</td><td>0.85</td><td>13.67</td><td>0.88</td><td>0.86</td><td>0.14</td><td>0.00</td><td>0.08</td><td>0.00</td><td>0.21</td><td>0.00</td><td>0.30</td><td>0.00</td></tr><tr><td>CarRun</td><td>0.75</td><td>2.51</td><td>0.75</td><td>0.52</td><td>0.99</td><td>1.47</td><td>0.80</td><td>0.00</td><td>0.75</td><td>0.22</td><td>0.84</td><td>0.00</td><td>0.95</td><td>0.00</td></tr><tr><td>DroneRun</td><td>0.65</td><td>0.71</td><td>0.26</td><td>0.44</td><td>0.71</td><td>0.60</td><td>0.41</td><td>0.57</td><td>0.62</td><td>1.34</td><td>0.55</td><td>0.00</td><td>0.57</td><td>0.00</td></tr><tr><td>AntCircle</td><td>0.61</td><td>1.42</td><td>0.00</td><td>0.00</td><td>0.46</td><td>2.74</td><td>0.23</td><td>0.00</td><td>0.40</td><td>0.78</td><td>0.41</td><td>0.27</td><td>0.34</td><td>0.06</td></tr><tr><td>BallCircle</td><td>0.79</td><td>1.20</td><td>0.40</td><td>4.37</td><td>0.79</td><td>1.64</td><td>0.45</td><td>0.00</td><td>0.29</td><td>1.83</td><td>0.42</td><td>0.00</td><td>0.59</td><td>0.01</td></tr><tr><td>CarCircle</td><td>0.64</td><td>1.80</td><td>0.49</td><td>4.48</td><td>0.70</td><td>1.20</td><td>0.34</td><td>0.00</td><td>0.28</td><td>0.04</td><td>0.53</td><td>0.00</td><td>0.67</td><td>0.11</td></tr><tr><td>DroneCircle</td><td>0.68</td><td>1.19</td><td>-0.27</td><td>1.29</td><td>0.59</td><td>1.56</td><td>0.60</td><td>0.00</td><td>0.66</td><td>1.37</td><td>0.59</td><td>0.10</td><td>0.47</td><td>0.05</td></tr><tr><td>Bullet Avg</td><td>0.57</td><td>1.71</td><td>0.33</td><td>3.10</td><td>0.73</td><td>1.41</td><td>0.43</td><td>0.17</td><td>0.50</td><td>0.88</td><td>0.52</td><td>0.07</td><td>0.56</td><td>0.04</td></tr><tr><td>MetaDrive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Easysparse</td><td>0.94</td><td>9.25</td><td>-0.05</td><td>0.15</td><td>0.25</td><td>0.15</td><td>0.41</td><td>0.50</td><td>0.74</td><td>1.55</td><td>0.40</td><td>0.04</td><td>0.56</td><td>0.00</td></tr><tr><td>Easymean</td><td>0.99</td><td>7.22</td><td>-0.06</td><td>0.00</td><td>0.42</td><td>0.25</td><td>0.43</td><td>0.67</td><td>0.70</td><td>0.68</td><td>0.35</td><td>0.07</td><td>0.55</td><td>0.00</td></tr><tr><td>Easydense</td><td>0.20</td><td>1.76</td><td>-0.06</td><td>0.16</td><td>0.35</td><td>1.17</td><td>0.52</td><td>1.26</td><td>0.74</td><td>1.48</td><td>0.35</td><td>0.02</td><td>0.55</td><td>0.02</td></tr><tr><td>Mediumsparse</td><td>0.94</td><td>2.83</td><td>-0.08</td><td>0.12</td><td>0.78</td><td>1.24</td><td>0.43</td><td>0.08</td><td>0.97</td><td>0.79</td><td>0.63</td><td>0.34</td><td>0.79</td><td>0.04</td></tr><tr><td>Mediummean</td><td>0.70</td><td>4.45</td><td>-0.07</td><td>0.16</td><td>0.72</td><td>2.74</td><td>0.36</td><td>0.02</td><td>0.92</td><td>0.89</td><td>0.71</td><td>0.54</td><td>0.81</td><td>0.07</td></tr><tr><td>Mediumdense</td><td>0.76</td><td>3.90</td><td>-0.08</td><td>0.10</td><td>0.70</td><td>2.62</td><td>0.51</td><td>0.39</td><td>0.87</td><td>0.88</td><td>0.36</td><td>0.04</td><td>0.76</td><td>0.05</td></tr><tr><td>Hardsparse</td><td>0.49 0.29</td><td>3.16</td><td>-0.05</td><td>0.10</td><td>0.26</td><td>0.46</td><td>0.33</td><td>0.24</td><td>0.52</td><td>1.32</td><td>0.30</td><td>0.01</td><td>0.41</td><td>0.04</td></tr><tr><td>Hardmean Harddense</td><td>0.42</td><td>3.80 2.95</td><td>-0.05 -0.04</td><td>0.15 0.12</td><td>0.20 0.22</td><td>0.61 1.38</td><td>0.27</td><td>0.01</td><td>0.41</td><td>0.57</td><td>0.33</td><td>0.08</td><td>0.41</td><td>0.03</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.30</td><td>0.26</td><td>0.53</td><td>1.63</td><td>0.36</td><td>0.19</td><td>0.36</td><td>0.05</td></tr><tr><td>MetaDrive Avg</td><td>0.64</td><td>4.37</td><td>-0.06</td><td>0.12</td><td>0.44</td><td>1.18</td><td>0.40</td><td>0.38</td><td>0.71</td><td>1.09</td><td>0.42</td><td>0.15</td><td>0.58</td><td>0.03</td></tr></table>

## 6.3 Main Results

Table 4 presents a performance comparison of the algorithms on the Bullet-Safety-Gym and MetaDrive tasks. Overall, MoViL satisfies the cost limit on every task and achieves competitive returns with the lowest average cost on both benchmarks. BCQL and CDT achieve high returns in some tasks but frequently violate safety constraints. CPQ satisfies the limit on all MetaDrive tasks, yet its uniformly negative returns indicate highly conservative behavior. FISOR satisfies the cost limit on most tasks but achieves lower average returns than MoViL. In recent strong baselines, LSPC achieves high returns in multiple tasks, but usually with higher costs. GSCO demonstrates good safety control capabilities, but its overall returns are lower than MoViL, especially on MetaDrive. In contrast, MoViL achieves higher overall returns while maintaining lower costs, demonstrating an effective balance between safety and task performance.

## 6.4 Viability Certification Ablation

To evaluate the contribution of viability certification, we compare the full policy with two variants, BC Only and Cost-greedy. Fig. 5 presents their average normalized reward and cost across eight Bullet-Safety-Gym tasks. BC Only directly executes the mean action of the behavior-cloning policy. It achieves the highest return, but its cost approaches the constraint limit. Cost-greedy always selects the candidate action with the lowest predicted cost. It reduces cost relative to BC Only, but results in a substantially lower return and a higher cost than the full policy. The complete policy results in the lowest average cost among the three variants, utilizing cost-based action selection only when the BC mean action does not pass certification.

(a) Normalized reward ↑  
![](images/b99b99d95b420bffb6201118b195a8f4589392f7630935c603982e211ba41472.jpg)

![](images/64eb99f2a830aee365359c168718a1810a5b51d451dc9c7017c170f506a5d1aa.jpg)  
Figure 5: Viability certification ablation on eight Bullet-Safety-Gym tasks. Average normalized reward and cost are reported.

![](images/b48fa7c2370bf760288381b2d09e78511565282d4da4d2239a7d0279ad5bd572.jpg)

(b) Normalized cost ↓  
![](images/74b42a08bfcca1d7c47dcf76b0e23a5e88bab26b660b9af251874c1cf413cee5.jpg)  
Figure 6: Ablation of probability calibration and trajectory witness protection on eight Bullet-Safety-Gym tasks. Average normalized reward and cost are reported.

## 6.5 Viability Estimation Ablation

To evaluate the contributions of probability calibration and trajectory witness protection to viability learning, we compare the full method with two variants that remove each component separately. Fig. 6 presents the average normalized reward and cost across eight Bullet-Safety-Gym tasks. Removing probability calibration yields comparable reward but increases the average cost. Since calibration adjusts the threshold for retaining states during recursive learning, this result suggests that it improves the viability estimates used for subsequent action certification. Removing witness protection leads to a further increase in cost, accompanied by a reduction in reward. Witness protection preserve observed safe continuations, reducing erroneous exclusions of viable states and their propagation through subsequent training targets.

## 6.6 Sensitivity to $p _ { \mathrm { m i n } }$

Fig. 7 shows the sensitivity to $p _ { \mathrm { m i n } }$ on two PyBullet tasks and one MetaDrive task. A larger $p _ { \mathrm { m i n } }$ imposes a stricter behavior-support requirement for retaining states during viability learning. On DroneCircle, increasing $p _ { \mathrm { m i n } }$ from 0.005 to 0.2 substantially reduces cost, but also lowers reward. At 0.3, cost rises again while reward decreases further. On CarCircle, reward remains stable up to $p _ { \operatorname* { m i n } } = 0 .$ 1, but declines at higher thresholds with little reduction in cost. In contrast, MediumDense shows only minor variations across the tested values. These results suggest that overly restrictive thresholds may exclude useful viable states without improving safety. We therefore adopt $p _ { \mathrm { m i n } } = 0 . 1$ as a reasonable balance between reward and cost.

## 6.7 Sensitivity to τ

Fig. 8 shows the sensitivity to the expectile parameter τ on two PyBullet tasks and one MetaDrive task. Reward remains relatively stable across the tested values on all three tasks. The costs of AntRun and MediumDense also exhibit only minor variations. BallCircle shows larger cost fluctuations, reaching its lowest average cost at $\tau = 0 . 9 9 5$ Increasing τ places greater emphasis on positive viability targets during training. Probability calibration compensates for the τ-dependent score scaling, allowing the same behavior-support threshold to be used across different values of τ. Overall, reward is relatively insensitive to τ, while cost exhibits some task-dependent variations.

![](images/be36c0c2d91427957572c34996dc7e8dc08efdddefc5eb41de7a12e21d8087f3.jpg)

![](images/a0c5b6190351a592b5a52f1c419e04fc8017083cc4fe084afdbc579b4aaec76e.jpg)  
Figure 7: Sensitivity to $p _ { \mathrm { m i n } }$ on selected tasks.

![](images/a5c64ed5eeeba8d786ed2b37c24cc27615f9930d63b4f9d6b7894ad98a508cd8.jpg)

![](images/63558d642193f7e11951037ad4f8b83ea50997a3ba27feb2f8f66a969c269c4b.jpg)  
Figure 8: Sensitivity to τ on selected tasks.

![](images/7c24ccdd5783a894b15aca73a24ec4f65c014da870348160d352c422afa201fe.jpg)

(b) Normalized Cost  
![](images/c06d2953b6b809e045df2db29b554f79f799f4bd06e9df5a278be1b80b7799e9.jpg)  
Figure 9: Effect of viability training data size on policy performance across selected tasks.

## 6.8 Effect of Data Size

To examine the effect of training data size on viability learning, we vary the proportion of offline data used to train $H _ { k }$ and $Q _ { H }$ from 25% to 100%, while keeping the Gaussian BC policy and cost critic fixed. Fig. 9 compares the resulting policy performance on three representative tasks. Both reward and cost remain relatively stable across different data proportions. Even with only 25% of the training data, MoViL achieves performance comparable to that obtained with the full dataset. These results indicate that MoViL can maintain effective viability-based certification with substantially reduced training data on the evaluated tasks.

## 7 Conclusion

We proposed MoViL for learning long-horizon HJ viability from fixed offline transitions. By approximating the infinite-horizon viability kernel through monotone finite-horizon set recursion, MoViL simultaneously achieves semantic preservation, convergence, and numerical stability. At deployment, MoViL uses the learned viability estimates to certify actions proposed by a separately trained policy. Dubins car experiments showed that the learned viable sets closely matched the numerical HJ reference. On standard safe offline RL benchmarks, MoViL achieved the lowest average normalized cost while maintaining competitive returns. Future work will investigate how viability estimates can guide the selection and composition of skills in hierarchical policies.

## References

[1] Lukas Brunke, Melissa Greeff, Adam W. Hall, Zhaocong Yuan, Siqi Zhou, Jacopo Panerati, and Angela P. Schoellig. Safe learning in robotics: From learning-based control to safe reinforcement learning. Annual Review of Control, Robotics, and Autonomous Systems, 5(1):411–444, 2022. doi: 10.1146/ annurev-control-042920-020211.

[2] Rafael Figueiredo Prudencio, Marcos R. O. A. Maximo, and Esther Luna Colombini. A survey on offline re inforcement learning: Taxonomy, review, and open problems. IEEE Transactions on Neural Networks and Learning Systems, 35(8):10237–10257, 2024. doi: 10.1109/TNNLS.2023.3250269.

[3] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In Proceedings ofthe International Conference on Machine Learning, pages 22–31, 2017.

[4] Weiye Zhao, Tairan He, Rui Chen, Tianhao Wei, and Changliu Liu. State-wise safe reinforcement learning: A survey. In Proceedings of the International Joint Conference on Artificial Intelligence, pages 6814–6822, 2023. doi: 10.24963/ijcai.2023/763.

[5] Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada. Control barrier functions: Theory and applications. In Proceedings of the European Control Conference, pages 3420–3431, 2019. doi: 10.23919/ECC.2019.8796030.

[6] Haitong Ma, Changliu Liu, Shengbo Eben Li, Sifa Zheng, Wenchao Sun, and Jianyu Chen. Learn zero-constraintviolation safe policy in model-free constrained reinforcement learning. IEEE Transactions on Neural Networks and Learning Systems, 36(2):2327–2341, 2025. doi: 10.1109/TNNLS.2023.3348422.

[7] Mayank Shekhar Jha, Satya Marthi, Kyriakos G. Vamvoudakis, Soha Kanso, and Didier Theilliol. Input-tostate safety for reinforcement learning. IEEE Transactions on Neural Networks and Learning Systems, 37(10): 5584–5598, 2026. doi: 10.1109/TNNLS.2026.3688045

[8] Kai-Chieh Hsu, Haimin Hu, and Jaime F. Fisac. The safety filter: A unified view of safety-critical control in autonomous systems. Annual Review of Control, Robotics, and Autonomous Systems, 7:47–72, 2024. doi: 10.1146/annurev-control-071723-102940.

[9] Amir Modares, Nasser Sadati, Babak Esmaeili, Farnaz Adib Yaghmaie, and Hamidreza Modares. Safe reinforce ment learning via a model-free safety certifier. IEEE Transactions on Neural Networks and Learning Systems, 35(3):3302–3311, 2024. doi: 10.1109/TNNLS.2023.3264815.

[10] Somil Bansal, Mo Chen, Sylvia Herbert, and Claire J. Tomlin. Hamilton-Jacobi reachability: A brief overview and recent advances. In Proceedings ofthe IEEE Conference on Decision and Control, pages 2242–2253, 2017. doi: 10.1109/CDC.2017.8263977.

[11] Ian M. Mitchell, Alexandre M. Bayen, and Claire J. Tomlin. A time-dependent Hamilton–Jacobi formulation of reachable sets for continuous dynamic games. IEEE Transactions on Automatic Control, 50(7):947–957, 2005. doi: 10.1109/TAC.2005.851439.

[12] Somil Bansal and Claire J. Tomlin. DeepReach: A deep learning approach to high-dimensional reachability. In Proceedings of the IEEE International Conference on Robotics and Automation, pages 1817–1824, 2021. doi: 10.1109/ICRA48506.2021.9561949.

[13] Jaime F. Fisac, Neil F. Lugovoy, Vicenc¸ Rubies-Royo, Shromona Ghosh, and Claire J. Tomlin. Bridging Hamilton-Jacobi safety analysis and reinforcement learning. In Proceedings of the IEEE International Conference on Robotics and Automation, pages 8550–8556, 2019. doi: 10.1109/ICRA.2019.8794107.

[14] Yinan Zheng, Jianxiong Li, Dongjie Yu, Yujie Yang, Shengbo Eben Li, Xianyuan Zhan, and Jingjing Liu. Safe offline reinforcement learning with feasibility-guided diffusion model. In Proceedings of the International Con ference on Learning Representations, pages 47725–47754, 2024.

[15] Anayo K. Akametalu, Shromona Ghosh, Jaime F. Fisac, Vicenc¸ Rubies-Royo, and Claire J. Tomlin. A minimum discounted reward Hamilton-Jacobi formulation for computing reachable sets. IEEE Transactions on Automatic Control, 69(2):1097–1103, 2024. doi: 10.1109/TAC.2023.3327159.

[16] Jason J. Choi, Donggun Lee, Koushil Sreenath, Claire J. Tomlin, and Sylvia L. Herbert. Robust control barrier– value functions for safety-critical control. In Proceedings of the IEEE Conference on Decision and Control, pages 6814–6821, 2021. doi: 10.1109/CDC45484.2021.9683085.

[17] Isabelle El-Hajj, Prashant Solanki, Jasper van Beers, Coen de Visser, and Erik-Jan van Kampen. Reachabilitypreserving Bellman operator for the discounted reach-cost value function: Uniting Hamilton-Jacobi reachability and reinforcement learning, 2026. arXiv:2607.07893.

[18] Albert Lin and Somil Bansal. Verification of neural reachable tubes via scenario optimization and conformal prediction. In Proceedings ofthe Learningfor Dynamics & Control Conference, pages 719–731, 2024.

[19] Yujie Yang, Hanjiang Hu, Tianhao Wei, Shengbo Eben Li, and Changliu Liu. Scalable synthesis of formally verified neural value function for Hamilton-Jacobi reachability analysis. Journal ofArtificial Intelligence Research, 83:19:1–19:36, 2025. doi: 10.1613/jair.1.16946.

[20] Patrick Saint-Pierre. Approximation of the viability kernel. Applied Mathematics & Optimization, 29(2):187– 209, 1994. doi: 10.1007/BF01204182.

[21] Zuxin Liu, Zijian Guo, Haohong Lin, Yihang Yao, Jiacheng Zhu, Zhepeng Cen, Hanjiang Hu, Wenhao Yu, Tingnan Zhang, Jie Tan, and Ding Zhao. Datasets and benchmarks for offline safe reinforcement learning. Journal ofData-centric Machine Learning Research, 1(12):1–29, 2024.

[22] Kai-Chieh Hsu, Vicenc¸ Rubies-Royo, Claire J. Tomlin, and Jaime F. Fisac. Safety and liveness guarantees through reach-avoid reinforcement learning. In Proceedings of Robotics: Science and Systems, 2021. doi: 10.15607/RSS.2021.XVII.077.

[23] Prashant Solanki, Isabelle El-Hajj, Jasper van Beers, Erik-Jan van Kampen, and Coen de Visser. Unifying Hamilton-Jacobi reachability and reinforcement learning, 2026. arXiv:2601.08050.

[24] Dongjie Yu, Haitong Ma, Shengbo Li, and Jianyu Chen. Reachability constrained reinforcement learning. In Proceedings of the International Conference on Machine Learning, pages 25636–25655, 2022.

[25] Milan Ganai, Zheng Gong, Chenning Yu, Sylvia Herbert, and Sicun Gao. Iterative reachability estimation for safe reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 69764–69797, 2023. doi: 10.52202/075280-3058.

[26] Haoran Xu, Xianyuan Zhan, and Xiangyu Zhu. Constraints penalized Q-learning for safe offline reinforcement learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 8753–8760, 2022. doi: 10.1609/aaai.v36i8.20855.

[27] Jongmin Lee, Cosmin Paduraru, Daniel J. Mankowitz, Nicolas Heess, Doina Precup, Kee-Eung Kim, and Arthur Guez. COptiDICE: Offline constrained reinforcement learning via stationary distribution correction estimation. In Proceedings ofthe International Conference on Learning Representations, pages 15727–15750, 2022.

[28] Zuxin Liu, Zijian Guo, Yihang Yao, Zhepeng Cen, Wenhao Yu, Tingnan Zhang, and Ding Zhao. Constrained Decision Transformer for offline safe reinforcement learning. In Proceedings ofthe International Conference on Machine Learning, pages 21611–21630, 2023.

[29] Huikang Su, Dengyun Peng, Zifeng Zhuang, Yuhan Liu, Qiguang Chen, Donglin Wang, and Qinghe Liu. Boundary-to-region supervision for offline safe reinforcement learning. In Advances in Neural Information Processing Systems, volume 38, pages 64567–64596, 2025. doi: 10.52202/085713-1941.

[30] Prajwal Koirala, Zhanhong Jiang, Soumik Sarkar, and Cody Fleming. Latent safety-constrained policy approach for safe offline reinforcement learning. In Proceedings of the International Conference on Learning Representations, pages 21985–22011, 2025.

[31] Zhiqi Zhuang, Di Wu, and Benoit Boulet. Geometric control of out-of-distribution shift in safe offline RL. In Proceedings of the International Conference on Machine Learning, pages 168167–168191, 2026.

[32] Mumuksh Tayal, Manan Tayal, Aditya Singh, Shishir Kolathaya, and Ravi Prakash. V-OCBF: Learning safety filters from offline data via value-guided offline control barrier functions. Transactions on Machine Learning Research, 2026. URL https://openreview.net/forum?id=PGO9mpIyyb.

[33] Ayan Choudhury, Janaka Brahmanage, Akshat Kumar, and Praveen Paruchuri. Persistent safety set guided offline safe reinforcement learning. In Proceedings of the International Joint Conference on Artificial Intelligence, pages 4224–4232, 2026. doi: 10.24963/ijcai.2026/470.

[34] Luc Doyen and Michel De Lara. Stochastic viability and dynamic programming. Systems & Control Letters, 59 (10):629–634, 2010. doi: 10.1016/j.sysconle.2010.07.008.

[35] Dimitri P. Bertsekas and Steven E. Shreve. Stochastic Optimal Control: The Discrete-Time Case. Academic Press, New York, 1978.

[36] Whitney K. Newey and James L. Powell. Asymmetric least squares estimation and testing. Econometrica, 55 (4):819–847, 1987. doi: 10.2307/1911031.

[37] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings ofthe International Conference on Machine Learning, pages 1321–1330, 2017.

[38] Sven Gronauer. Bullet-Safety-Gym: A framework for constrained reinforcement learning. Technical report, Technical University of Munich, Department of Electrical and Computer Engineering, 2022.

[39] Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, and Bolei Zhou. MetaDrive: Composing diverse driving scenarios for generalizable reinforcement learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3461–3475, 2023. doi: 10.1109/TPAMI.2022.3190471.

[40] Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In Proceedings of the International Conference on Machine Learning, pages 2052–2062, 2019.