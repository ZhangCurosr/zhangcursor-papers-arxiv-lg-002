# Regret-based Characterization, Information Limits, and the Sample Complexity of Auditing Adaptive Policies

Irene Aldridge<sup>∗</sup>

## Abstract

A policy can earn its performance by where it locates on average or by reacting to information. For any Markov policy π in a finite-horizon MDP, let ν be its open-loop marginal, which plays π’s average action distribution at each step. We show that $J ( \pi ) - J ( \nu )$ equals $\textstyle \sum _ { t } \operatorname { t r } \operatorname { C o v } ( Q _ { t } ^ { \nu } , \pi _ { t } )$ under endogenous dynamics. Audits based on immediate costs alone can have the wrong sign. Writing the covariance as a joint-versus-product gap gives the bound $| J ( \pi ) - J ( \nu ) | \leq$ $\begin{array} { r } { \sum _ { t } \operatorname { s p a n } _ { t } \sqrt { I ( s _ { t } ; a _ { t } ) / 2 } } \end{array}$ , with a sharp constant. Hence, an MDP’s adaptivity gap is bounded by the information that its optimal policy uses. Estimating $J ( \pi ) - J ( \nu )$ from π’s own banditfeedback logs takes $\Theta \big ( \bar { \chi } ^ { 2 } \log ( 1 / \eta ) / \delta ^ { 2 } \big )$ rounds, where $\bar { \chi } ^ { 2 }$ is an average $\chi ^ { 2 } .$ -divergence. At the same time, deterministic non-constant policies cannot be audited at all. We also provide an importance-sampling auditor for episodic MDPs.

## 1 Introduction

Learned policies set portfolio weights, prices, bids, inventory orders, and treatment assignments. A principal or regulator who evaluates such a policy usually sees only its inputs and outputs, but wants to know whether the policy creates value by using information or merely by how it behaves on average. The second kind of value can be replicated by a fixed rule; the first cannot. This paper studies the problem of separating the two, which we call auditing adaptivity, from three angles: an exact characterization, an information-theoretic limit, and the sample complexity of estimation.

The benchmark. Fix a Markov policy $\pi = ( \pi _ { 1 } , \ldots , \pi _ { H } )$ in a finite-horizon MDP and let $d _ { t } ^ { \pi }$ be its state distribution at step t. Its open-loop marginal $\nu = \nu ^ { \pi }$ is the policy that, at step t, ignores the state and draws $a _ { t }$ from

$$
\bar { p } _ { t } ( a ) = \mathbb { E } _ { s \sim d _ { t } ^ { \pi } } \left[ \pi _ { t } ( a \mid s ) \right] .
$$

Thus ν uses each action exactly as often as π does on average at each step, but uses no information. The quantity $\Delta ( \pi ) : = J ( \pi ) - J ( \nu )$ , where J denotes expected total cost, is the value that π adds (if negative) or destroys (if positive) by conditioning on the state. In finance, this is the classical split of active returns into a ”timing” (covariance) component and a ”static allocation” (means) component [9, 19]; in stochastic optimization, min<sub>ν</sub> $J ( \nu ) - \operatorname* { m i n } _ { \pi } J ( \pi )$ is the additive adaptivity gap [10, 13]. Recent work by Aldridge [2, 3] observed that, for stochastic linear programs and portfolio policies, expected regret equals the covariance between costs and decisions when the policy is mean-unbiased. We show that this phenomenon is exact in far greater generality and develop its quantitative theory.

## 1.1 Results

1. Exact characterization (Section 3). In every finite-horizon MDP and for every Markov policy π,

$$
J ( \pi ) - J ( \nu ) = \sum _ { t = 1 } ^ { H } \mathrm { t r } \mathrm { C o v } _ { s _ { t } \sim d _ { t } ^ { \pi } } \Big ( Q _ { t } ^ { \nu } ( s _ { t } , \cdot ) , ~ \pi _ { t } ( \cdot \mid s _ { t } ) \Big )\tag{1}
$$

$$
J ( \pi ) - J ( \nu ) \ = \ \sum _ { t = 1 } ^ { H } \Big ( \mathbb { E } _ { P _ { t } } \big [ Q _ { t } ^ { \nu } \big ] - \mathbb { E } _ { P _ { t } ^ { S } \otimes P _ { t } ^ { A } } \big [ Q _ { t } ^ { \nu } \big ] \Big ) ,\tag{2}
$$

where $P _ { t }$ is the joint law of $( s _ { t } , a _ { t } )$ under π and $P _ { t } ^ { S } { \otimes } P _ { t } ^ { A }$ is the product of its marginals (Theorem 3.4). The identity holds under fully endogenous dynamics; the feedback from actions to states is absorbed by evaluating the covariance under π’s state distribution and pairing actions with the Q-function of ν. Pathwise and with no probability model at all, the regret of any online linear optimization algorithm against the best fixed action in hindsight equals a sample covariance plus a non-negative static gap (Theorem 3.1). Regret to the optimal policy splits as adaptivity plus static allocation plus the problem’s adaptivity gap, and the adaptivity gap is at most the covariance magnitude of any optimal policy (Theorem 3.7). The $Q ^ { \nu }$ in (1) cannot be replaced by immediate costs: the resulting “myopic audit” can report value destruction for a policy that creates value, with unbounded error (Theorem 3.8).

2. Information limits (Section 4). The right-hand form of (1) compares a joint law to the product of its marginals. The KL divergence between them is the mutual information $I ( s _ { t } ; a _ { t } )$ , so Pinsker’s inequality yields:

Informal statement (Theorems 4.2 and 4.3). $\begin{array} { r } { | J ( \pi ) - J ( \nu ) | \leq \sum _ { t } \operatorname { s p a n } _ { t } ( \pi ) \sqrt { I ( s _ { t } ; a _ { t } ) / 2 } } \end{array}$ , where span<sub>t</sub> is the largest per-state range of $Q _ { t } ^ { \nu }$ across actions. The constant $1 / \sqrt { 2 }$ cannot be improved.

Two consequences follow. The additive adaptivity gap of an MDP is at most $\textstyle \sum _ { t } { \mathrm { s p a n } } _ { t } { \sqrt { I _ { t } / 2 } }$ evaluated at any optimal policy, so problems whose optimal policies need few bits have small adaptivity gaps (Theorem 4.4). And with costs in [0, 1], a policy whose actions carry at most R units of mutual information between state and action per step (a.k.a. $R \textsuperscript { * } \mathrm { n a t s } ^ { , * } )$ cannot beat the best open-loop policy by more than $\frac { H ( H + 1 ) } { 2 } \sqrt { R / 2 }$ (Theorem 4.5). This is a rate–value tradeof in the spirit of rational inattention [24, 25].

3. Sample complexity of auditing (Section 5). An auditor observes $\pi \mathrm { { s } }$ own logs and must estimate $\Delta ( \pi )$ . Under full information, the problem is easy. Under bandit feedback, it reduces to of-policy evaluation of the open-loop marginal $\nu ^ { \pi }$ , and the right complexity parameter is

$$
\bar { \chi } ^ { 2 } ( \pi ) \ : = \ \mathbb { E } _ { x } \Big [ \chi ^ { 2 } \big ( \bar { p } \big \| \pi ( \cdot \mid x ) \big ) \Big ] ,
$$

the average $\chi ^ { 2 } .$ -divergence from the marginal to the conditional action distributions.

Informal statement (Theorems 5.2 to 5.4). A median-of-means IPW auditor estimates $\Delta ( \pi )$ within $\pm \delta$ with probability $1 - \eta$ from ${ \cal O } ( \bar { \chi } ^ { 2 } \log ( 1 / \eta ) / \delta ^ { 2 } )$ logged rounds, for every policy and instance. Conversely, for every value $\Gamma \geq 1 / 3$ there is a policy with $\bar { \chi } ^ { 2 } = \Gamma$ and two instances on which every auditor needs $\Omega ( \Gamma \log ( 1 / \eta ) / \delta ^ { 2 } )$ rounds. If π never plays some action that $\nu ^ { \pi }$ plays, on a set of contexts of positive measure, then $\Delta ( \pi )$ is not identifiable. In particular, a deterministic non-constant policy cannot be audited from its own logs.

For episodic MDPs, an importance-sampling auditor over whole trajectories needs a number of episodes

$$
n = O \big ( H ^ { 2 } \chi ^ { 2 } ( P ^ { \nu } \| P ^ { \pi } ) \log ( 1 / \eta ) / \delta ^ { 2 } \big ) , \qquad \mathrm { w h e r e } \qquad 1 + \chi ^ { 2 } ( P ^ { \nu } \| P ^ { \pi } ) \le \prod _ { t } \big ( 1 + \operatorname* { m a x } _ { s } \chi ^ { 2 } ( \bar { p } _ { t } \| \pi _ { t } ( \cdot \ | \ s ) ) \big )
$$

(Theorem 5.6).

## 1.2 Techniques

The characterization rests on two observations. First, the open-loop marginal is the unique stateindependent policy whose value function satisfies $V _ { t } ^ { \nu } ( s ) = \langle Q _ { t } ^ { \nu } ( s , \cdot ) , \bar { p } _ { t } \rangle$ with the same $\bar { p } _ { t }$ as $\pi \mathrm { { s } }$ marginal. Plugging this into the performance-diference lemma [16] turns each advantage term into a centered inner product, which is a covariance. Second, a covariance between a function of the state and a function of the action is $\mathbb { E } _ { P } f - \mathbb { E } _ { P ^ { S } \otimes P ^ { A } } f ,$ so any divergence bound between a joint law and its product applies. The identity is invariant to adding any function of the state to $Q _ { t } ^ { \nu }$ , and choosing the per-state midpoint replaces the global oscillation of $Q _ { t } ^ { \nu }$ by the much smaller per-state action range. For auditing, the key observation is that $\mathbb { E } [ \ell _ { t } ( a _ { t } ) ]$ under $\pi$ is directly observable, so all statistical dificulty lies in evaluating $\nu ^ { \pi }$ of-policy. The variance of the natural estimator is exactly bounded by $\bar { \chi } ^ { 2 }$ . The lower bound plants a cost perturbation on the context–action pair that π plays with probability $p$ but $\nu ^ { \pi }$ plays with probability $1 / 2  \it$ , and applies the Bretagnolle–Huber inequality.

## 1.3 Related work

Online learning and regret. Regret against the best fixed action is the standard benchmark of online learning [7, 14]. Theorem 3.1 is an elementary identity, but it provides a structural reading of regret: negative regret (beating the best fixed action) is possible only with negative sample covariance, and every no-regret algorithm has sample covariance of at most its regret. Applied to stochastic bandit algorithms such as Thompson sampling, whose regret is near-optimal [1], the decomposition separates an algorithm’s static exposure to suboptimal arms (the static gap) from the timing of its exploration (the covariance).

Adaptivity gaps. The benefit of adaptivity is studied as a multiplicative ratio for stochastic knapsack [10], stochastic probing [13] and submodular maximization [4]. Our bounds are additive and problem-agnostic: Theorem 4.4 bounds the gap by the information used by an optimal policy, and Theorem 3.7 identifies the policy-level quantity, a covariance, whose extremum the gap is.

Information-theoretic bounds. Bounding the covariance between a selection and a quantity by mutual information through a change of measure underlies information-theoretic analyzes of adaptive data analysis [23], generalization [29], and Thompson sampling [22]. Theorem 4.2 applies this technique to the value of adaptivity in MDPs. Evaluating the bound on continuous-state data requires estimating mutual information, for which eficient k-nearest-neighbour entropy estimators are available [5]. Two ingredients specific to this setting are the reduction to $Q ^ { \nu }$ through Theorem 3.4 and the state-baseline invariance that yields the per-state span. Information-constrained decisionmaking is studied as rational inattention [24] and in information-theoretic RL [25].

Of-policy evaluation. Our auditing results are an instance of of-policy evaluation (OPE), with the twist that the target policy $\nu ^ { \pi }$ is defined by the behavior policy itself. Importance sampling and doubly robust estimators [11, 15, 21] and minimax analyses for contextual bandits [17, 27] are the closest antecedents. The lower bound of Theorem 5.4 is in the spirit of [17], specialized for targets of the form $\nu ^ { \pi }$ . Marginalized importance sampling [18, 28, 31] suggests routes to polynomial-in-horizon auditing.

Auditing LLM agents. Agentic systems built on large language models raise the same question at the level of tool calls and workflow steps. Systems working on agent evaluation ask when and how to inspect an agent’s execution mid-trajectory [30]. Our results supply a complementary statistical criterion. Logging the agent’s action propensities and states is enough to estimate its adaptive value against its own open-loop marginal (Theorem 5.6). Without exploration, however, a deterministic agent cannot be audited from its own logs (Theorem 5.2).

Finance and predict-then-optimize. The decomposition $\mathbb { E } [ c ^ { \top } z ] = \mathbb { E } [ c ] ^ { \top } \mathbb { E } [ z ] +$ tr $\operatorname { C o v } ( c , z )$ is classical in performance attribution [9, 19] and related to decision-focused learning [12]. Aldridge [2, 3] uses it as a regret characterization for stochastic linear programs and AI investment strategies under a mean-unbiasedness condition. The present paper removes that condition, extends the identity to endogenous dynamics, and supplies information and sample-complexity theory.

## 2 Model

MDPs and policies. A finite-horizon MDP is a tuple $M = ( { \mathcal { S } } , { \mathcal { A } } , H , ( P _ { t } ) _ { t \leq H } , ( \ell _ { t } ) _ { t \leq H } , \rho )$ with finite state and action sets, transition kernels $P _ { t } ( \cdot \mid s , a ) \in \Delta ( S )$ , bounded costs $\ell _ { t } : S \times \mathcal { A }  \mathbb { R }$ , and initial distribution $\rho \in \Delta ( \mathcal { S } )$ . A (randomized, time-dependent) Markov policy is $\pi = ( \pi _ { t } ) _ { t \leq H }$ with $\pi _ { t } ( \cdot \mid s ) \in \Delta ( \mathcal { A } )$ . Running π generates $s _ { 1 } \sim \rho , a _ { t } \sim \pi _ { t } ( \cdot \mid s _ { t } ) , s _ { t + 1 } \sim P _ { t } ( \cdot \mid s _ { t } , a _ { t } )$ . Write $d _ { t } ^ { \pi } \in \Delta ( S )$ for the law of $s _ { t } , J ( \pi ) = \mathbb { E } ^ { \pi } [ \sum _ { t = 1 } ^ { H } \ell _ { t } ( s _ { t } , a _ { t } ) ]$ for expected total cost, and

$$
Q _ { t } ^ { \mu } ( s , a ) = \ell _ { t } ( s , a ) + \mathbb { E } _ { s ^ { \prime } \sim P _ { t } ( \cdot \vert s , a ) } \bigl [ V _ { t + 1 } ^ { \mu } ( s ^ { \prime } ) \bigr ] , \qquad V _ { t } ^ { \mu } ( s ) = \langle Q _ { t } ^ { \mu } ( s , \cdot ) , \mu _ { t } ( \cdot \vert s ) \rangle , \qquad V _ { H + 1 } ^ { \mu } \equiv 0 ,
$$

for the action-value and value functions of a Markov policy $\mu .$ Costs, rather than rewards, maintain the sign convention of regret.

Definition 2.1 (Open-loop marginal). For a Markov policy π, let $\begin{array} { r } { \bar { p } _ { t } ^ { \pi } ( a ) = \sum _ { s } d _ { t } ^ { \pi } ( s ) \pi _ { t } ( a \mid s ) } \end{array}$ . The open-loop marginal of $\pi$ is the policy $\nu ^ { \pi }$ with $\nu _ { t } ^ { \pi } ( \cdot \mid s ) = \bar { p } _ { t } ^ { \pi }$ for all s. A policy is open-loop if $\mu _ { t } ( \cdot \mid s )$ does not depend on s for every t.

We write $\nu = \nu ^ { \pi }$ and $\bar { p } _ { t } = \bar { p } _ { t } ^ { \pi }$ when π is clear, and call $\Delta ( \pi ) : = J ( \pi ) - J ( \nu ^ { \pi } )$ the adaptivity cost of $\pi ; ~ - \Delta ( \pi )$ is its adaptive value. Let $\pi ^ { * }$ be an optimal policy and $\nu ^ { * }$ an optimal openloop policy (both exist by finiteness and compactness). The (additive) adaptivity gap of M is $\operatorname { G a p } ( M ) : = J ( \nu ^ { * } ) - J ( \pi ^ { * } ) \geq 0$

Covariances. For random vectors $q , z \in \mathbb { R } ^ { A }$ with finite second moments, tr $\operatorname { C o v } ( q , z ) = \mathbb { E } \langle q , z \rangle -$ $\langle \mathbb { E } q , \mathbb { E } z \rangle = \mathbb { E } \langle q - \mathbb { E } q , z - \mathbb { E } z \rangle$ . Throughout, $q _ { t } : = Q _ { t } ^ { \nu } ( s _ { t } , \cdot )$ and $z _ { t } : = \pi _ { t } ( \cdot \mid s _ { t } )$ with $s _ { t } \sim d _ { t } ^ { \pi }$ , so that $\mathbb { E } z _ { t } = \bar { p } _ { t }$

Information quantities. Under $\pi _ { \ i }$ let $P _ { t } ( s , a ) = d _ { t } ^ { \pi } ( s ) \pi _ { t } ( a \mid s )$ be the joint law of $\left( { { s _ { t } } , { a _ { t } } } \right)$ , with marginals $P _ { t } ^ { S } = d _ { t } ^ { \pi }$ and $P _ { t } ^ { A } = \bar { p } _ { t }$ . The mutual information is $I _ { t } ( \pi ) : = I ( s _ { t } ; a _ { t } ) = \mathrm { K L } ( P _ { t } \| P _ { t } ^ { S } \otimes P _ { t } ^ { A } )$ in units of information, nats. For distributions $p , q$ on A $\begin{array} { r } { 1 , \ \chi ^ { 2 } ( p \| q ) = \sum _ { a } p ( a ) ^ { 2 } / q ( a ) - 1 } \end{array}$ , with the convention $\chi ^ { 2 } = \infty \mathrm { i f } p \ll q$

## 3 Exact characterization

## 3.1 Pathwise: online linear optimization

Theorem 3.1 (Pathwise decomposition). Let $Z \subset \mathbb { R } ^ { n }$ be convex and compact, let $c _ { 1 } , \ldots , c _ { T } \in \mathbb { R } ^ { n }$ be arbitrary, and let $z _ { 1 } , \dotsc , z _ { T } \in Z$ be arbitrary. Let $\begin{array} { r } { \bar { c } = \frac { 1 } { T } \sum _ { t } c _ { t } , \bar { z } = \frac { 1 } { T } \sum _ { t } z _ { t } } \end{array}$ , and $z ^ { * } \in \arg$ min $_ { \cdot z \in Z } \bar { c } ^ { \top } z$ Then

$$
\underbrace { \sum _ { t = 1 } ^ { T } c _ { t } ^ { \top } z _ { t } - \operatorname* { m i n } _ { z \in Z } \sum _ { t = 1 } ^ { T } c _ { t } ^ { \top } z } _ { \mathrm { R e g r e t } _ { T } } = \underbrace { \sum _ { t = 1 } ^ { T } ( c _ { t } - \bar { c } ) ^ { \top } ( z _ { t } - \bar { z } ) } _ { \widehat { C } _ { T } } + \underbrace { T \bar { c } ^ { \top } ( \bar { z } - z ^ { * } ) } _ { \widehat { \geq 0 } } .
$$

Proof. $\begin{array} { r } { \sum _ { t } c _ { t } ^ { \top } z _ { t } = \sum _ { t } ( c _ { t } - \bar { c } ) ^ { \top } ( z _ { t } - \bar { z } ) + T \bar { c } ^ { \top } \bar { z } . } \end{array}$ , because the cross terms $\begin{array} { r } { \sum _ { t } ( c _ { t } - \bar { c } ) ^ { \top } \bar { z } } \end{array}$ and $\begin{array} { r } { \sum _ { t } \bar { c } ^ { \top } ( z _ { t } - \bar { z } ) } \end{array}$ vanish. Also min<sub>z</sub> $\textstyle \sum _ { t } c _ { t } ^ { \top } z = T { \bar { c } } ^ { \top } z ^ { * }$ . The last term is non-negative because $\bar { z } \in Z$ by convexity.

Corollary 3.2. For every online algorithm and every adversary, $\widehat { C } _ { T } \leq \mathrm { R e g r e t } _ { T }$ , with equality if and only $i f \ \bar { z } \textit { i s } \ a$ best fixed action in hindsight. In particular, Regret $_ T < 0$ implies $\widehat { C } _ { T } < 0$ , and every algorithm with $\mathrm { R e g r e t } _ { T } \leq \rho ( T )$ has $\begin{array} { r } { \widehat { C } _ { T } \leq \rho ( T ) } \end{array}$

Theorem 3.1 requires no stochastic assumptions; it is the accounting identity the auditor computes on a realized path. The rest of the paper concerns the expected, endogenous-dynamics analogue.

## 3.2 The MDP covariance identity

Lemma 3.3 (Performance diference [16]). For Markov policies $\pi , \mu ,$

$$
J ( \pi ) - J ( \mu ) = \sum _ { t = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { t } ^ { \pi } } \big [ \langle Q _ { t } ^ { \mu } ( s , \cdot ) , \pi _ { t } ( \cdot \mid s ) \rangle - V _ { t } ^ { \mu } ( s ) \big ] .
$$

Proof. Since $\begin{array} { r } { V _ { H + 1 } ^ { \mu } \equiv 0 , \sum _ { t = 1 } ^ { H } \left( \ell _ { t } ( s _ { t } , a _ { t } ) + V _ { t + 1 } ^ { \mu } ( s _ { t + 1 } ) - V _ { t } ^ { \mu } ( s _ { t } ) \right) = \sum _ { t } \ell _ { t } ( s _ { t } , a _ { t } ) - V _ { 1 } ^ { \mu } ( s _ { 1 } ) } \end{array}$ by telescoping. Take expectations under π: the right side is $J ( \pi ) - J ( \mu )$ . On the left, $\mathbb { E } ^ { \pi } [ \ell _ { t } ( s _ { t } , a _ { t } ) + V _ { t + 1 } ^ { \mu } ( s _ { t + 1 } ) \ |$ $s _ { t } , a _ { t } ] = Q _ { t } ^ { \mu } ( s _ { t } , a _ { t } )$ , and averaging over $a _ { t } \sim \pi _ { t } ( \cdot \mid s _ { t } )$ gives the claim. □

Theorem 3.4 (Covariance characterization). For every finite-horizon MDP and every Markov policy π, with $\nu = \nu ^ { \pi }$ ，

$$
J ( \pi ) - J ( \nu ) = \sum _ { t = 1 } ^ { H } \mathrm { t r } \mathrm { C o v } \left( q _ { t } , z _ { t } \right)\tag{3}
$$

$$
= \sum _ { t = 1 } ^ { H } \Big ( \mathbb { E } _ { ( s , a ) \sim P _ { t } } \big [ Q _ { t } ^ { \nu } ( s , a ) \big ] - \mathbb { E } _ { ( s , a ) \sim P _ { t } ^ { S } \otimes P _ { t } ^ { A } } \big [ Q _ { t } ^ { \nu } ( s , a ) \big ] \Big ) .\tag{4}
$$

Moreover, each summand is unchanged $i f Q _ { t } ^ { \nu } ( s , a )$ is replaced by $Q _ { t } ^ { \nu } ( s , a ) - g _ { t } ( s )$ for any $g _ { t } : S  \mathbb { R }$ Proof. Because $\nu _ { t } ( \cdot \mid s ) = \bar { p } _ { t }$ for every s, $V _ { t } ^ { \nu } ( s ) = \langle Q _ { t } ^ { \nu } ( s , \cdot ) , \bar { p } _ { t } \rangle$ . By Theorem 3.3 with $\mu = \nu$

$$
J ( \pi ) - J ( \nu ) = \sum _ { t } \mathbb { E } _ { s _ { t } \sim d _ { t } ^ { \pi } } \langle q _ { t } , z _ { t } - \bar { p } _ { t } \rangle .
$$

Since $\begin{array} { r } { \mathbb { E } z _ { t } = \bar { p } _ { t } , \mathbb { E } \langle \mathbb { E } q _ { t } , z _ { t } - \bar { p } _ { t } \rangle = 0 } \end{array}$ , so each summand equals $\begin{array} { r } { \mathbb E \langle q _ { t } - \mathbb E q _ { t } , z _ { t } - \mathbb E z _ { t } \rangle = \mathrm { t r } { \mathrm { C o v } ( q _ { t } , z _ { t } ) } } \end{array}$ which proves (3). For (4), $\begin{array} { r } { \mathbb { E } \langle q _ { t } , z _ { t } \rangle ~ = ~ \sum _ { s , a } d _ { t } ^ { \pi } ( s ) \pi _ { t } ( a ~ \vert ~ s ) Q _ { t } ^ { \nu } ( s , a ) ~ = ~ \mathbb { E } _ { P _ { t } } Q _ { t } ^ { \nu } } \end{array}$ and $\mathbb { E } \langle q _ { t } , \bar { p } _ { t } \rangle \ =$ $\begin{array} { r } { \sum _ { s , a } d _ { t } ^ { \pi } ( s ) \bar { p } _ { t } ( a ) Q _ { t } ^ { \nu } ( s , a ) = \mathbb { E } _ { P _ { t } ^ { S } \otimes P _ { t } ^ { A } } Q _ { t } ^ { \nu } } \end{array}$ . Finally, $P _ { t }$ and $P _ { t } ^ { S } \otimes P _ { t } ^ { A }$ have the same s-marginal, so $\mathbb { E } _ { P _ { t } } g _ { t } = \mathbb { E } _ { P _ { t } ^ { S } \otimes P _ { t } ^ { A } } g _ { t }$ □

Remark 3.5 (What is new relative to the static $\mathrm { i d e n t i t y ) }$ . The identity is exact under endogenous dynamics: actions afect future states, and this feedback is fully captured by (i) averaging over $\pi \mathrm { { ^ { * } s } }$ own state distribution $d _ { t } ^ { \pi }$ and (ii) pairing actions with the action-values of $\nu ,$ not of π and not the immediate costs. The baseline invariance means $Q _ { t } ^ { \nu }$ may be replaced by the advantage $Q _ { t } ^ { \nu } - V _ { t } ^ { \nu }$

Corollary 3.6 (Exogenous costs). Suppose actions do not afect transitions, that is, $P _ { t } ( \cdot \mid s , a ) =$ $P _ { t } ( \cdot \mid s )$ . Write $\begin{array} { r } { c _ { t } ( s ) = \ell _ { t } ( s , \cdot ) \in \mathbb { R } ^ { A } } \end{array}$ . Then $\begin{array} { r } { J ( \pi ) - J ( \nu ) = \sum _ { t } \mathrm { t r } { \mathrm { C o v } ( c _ { t } ( s _ { t } ) , z _ { t } ) } } \end{array}$ . In particular, for a contextual problem $( H = 1 , \ s = x ) , \ J ( \pi ) - J ( \nu ) = \operatorname { t r } \operatorname { C o v } ( c ( x ) , \pi ( \cdot \mid x ) )$

Proof. With action-independent transitions, $Q _ { t } ^ { \nu } ( s , a ) = \ell _ { t } ( s , a ) + g _ { t } ( s )$ with $g _ { t } ( s ) = \mathbb { E } [ V _ { t + 1 } ^ { \nu } ( s _ { t + 1 } )$ $s _ { t } = s ]$ . Apply the invariance in Theorem 3.4. □

## 3.3 Decomposing regret to the optimum

Corollary 3.7 (Three-part decomposition). For every Markov policy $\pi _ { : }$

$$
J ( \pi ) - J ( \pi ^ { * } ) = \underbrace { \sum _ { t } { \mathrm { t r } } \operatorname { C o v } ( q _ { t } , z _ { t } ) } _ { a d a p t i v i t y \ c o s t \ \Delta ( \pi ) } + \underbrace { J ( \nu ^ { \pi } ) - J ( \nu ^ { * } ) } _ { s t a t i c \ a l l o c a t i o n \geq 0 } + \underbrace { \operatorname { G a p } ( M ) } _ { \geq 0 } .
$$

Consequently ${ \mathrm { G a p } } ( M ) = - \Delta ( \pi ^ { * } ) - \left( J ( \nu ^ { \pi ^ { * } } ) - J ( \nu ^ { * } ) \right) \leq - \Delta ( \pi ^ { * } )$ for every optimal $\pi ^ { * }$ , with equality if and only if the open-loop marginal of $\pi ^ { * }$ is an optimal open-loop policy.

Proof. Add and subtract $J ( \nu ^ { \pi } )$ and $J ( \nu ^ { * } )$ , and apply Theorem 3.4. The middle term is non-negative because $\nu ^ { \pi }$ is open-loop. Setting $\pi = \pi ^ { * }$ gives the second statement. □

## 3.4 Myopic audits fail

A tempting shortcut is to audit with immediate costs, $\begin{array} { r } { \sum _ { t } \mathrm { t r } { \mathrm { C o v } ( \ell _ { t } ( s _ { t } , \cdot ) , z _ { t } ) } } \end{array}$ , which by Theorem 3.6 is correct when dynamics are exogenous. Under endogenous dynamics it can be arbitrarily wrong.

Proposition 3.8 (Sign reversal). For every m $> 1$ there is a two-step MDP with two states and two actions, and a deterministic policy $\pi ,$ such that

$$
\sum _ { t } { \mathrm { t r } } \operatorname { C o v } \left( \ell _ { t } ( s _ { t } , \cdot ) , z _ { t } \right) = \frac { 1 } { 4 } > 0 \qquad w h i l e \qquad J ( \pi ) - J \big ( \nu ^ { \pi } \big ) = \frac { 1 - m } { 4 } < 0 .
$$

Proof. Let $s _ { 1 } \sim \mathrm { B e r n o u l l i } ( 1 / 2 )$ and $\mathcal { A } = \{ 0 , 1 \} \ : ( ^ { \circ } \mathrm { w a i t } ^ { \prime \prime } , \ : \mathrm { \ : \overset { . . } { i n v e s t } ^ { \prime \prime } } )$ . Set $\ell _ { 1 } ( s , 0 ) = 0$ and $\ell _ { 1 } ( s , 1 ) =$ $1 + s$ . The step-2 state records $( s _ { 1 } , a _ { 1 } )$ , and $\ell _ { 2 }$ is action-independent with $\ell _ { 2 } ( ( s , 1 ) , \cdot ) = - m s$ and $\ell _ { 2 } ( ( s , 0 ) , \cdot ) = 0$ . Let $\pi _ { 1 } ( 1 \mid s ) = s$ (invest exactly when $s = 1 ) ; \pi _ { 2 }$ is arbitrary. Then $\bar { p } _ { 1 } = ( 1 / 2 , 1 / 2 )$ Myopic term. ${ \mathrm { A t ~ } } t = 1 , { \mathrm { ~ t r } } \operatorname { C o v } ( \ell _ { 1 } ( s , \cdot ) , z _ { 1 } ) = \operatorname { C o v } ( 1 + s , s ) = \operatorname { V a r } ( s ) = 1 / 4 ,$ since the $a = 0$ coordinate is constant. $\mathrm { A t } ~ t = 2$ , the cost vector is constant across actions, so its covariance with $z _ { 2 }$ vanishes because $\langle { \bf 1 } , z _ { 2 } - \bar { p } _ { 2 } \rangle = 0$

True value. $Q _ { 1 } ^ { \nu } ( s , 0 ) \ = \ 0$ and $Q _ { 1 } ^ { \nu } ( s , 1 ) \ : = \ : 1 + \ : s \ : - \ : m s$ . By Theorem 3.4, $J ( \pi ) - J ( \nu ) =$ $\operatorname { C o v } ( 1 + ( 1 - m ) s , s ) = ( 1 - m ) / 4 ;$ the step-2 term vanishes for the same reason as before. □

The ratio between the two quantities, $( 1 - m )$ , is unbounded as $m  \infty$ . Swapping the roles of the costs gives the reverse failure: a myopic audit that certifies value for a policy that destroys it.

## 4 Information limits of adaptivity

Definition 4.1 (Action span). For a Markov policy π with marginal $\nu = \nu ^ { \pi }$ , let

$$
\mathrm { s p a n } _ { t } ( \pi ) = \operatorname* { m a x } _ { s \in \mathrm { s u p p } d _ { t } ^ { \pi } } \Big ( \operatorname* { m a x } _ { a \in \mathrm { s u p p } \bar { p } _ { t } } Q _ { t } ^ { \nu } ( s , a ) - \operatorname* { m i n } _ { a \in \mathrm { s u p p } \bar { p } _ { t } } Q _ { t } ^ { \nu } ( s , a ) \Big ) .
$$

The span measures how much the choice of action can matter in a single state; it ignores the variation of $Q _ { t } ^ { \nu }$ across states, which by Theorem 3.4 is irrelevant to adaptivity.

Theorem 4.2 (Information bound). For every finite-horizon MDP and every Markov policy π,

$$
\left| J ( \pi ) - J ( \nu ^ { \pi } ) \right| \leq \sum _ { t = 1 } ^ { H } \operatorname { s p a n } _ { t } ( \pi ) \sqrt { \frac { I ( s _ { t } ; a _ { t } ) } { 2 } } ,
$$

where $I ( s _ { t } ; a _ { t } )$ is the mutual information, in nats, between state and action at step t under π.

Proof. Fix t and write $P = P _ { t }$ and $\tilde { P } = P _ { t } ^ { S } \otimes P _ { t } ^ { A }$ . If $d _ { t } ^ { \pi } ( s ) > 0$ and $\pi _ { t } ( a \mid s ) > 0$ then $\bar { p } _ { t } ( a ) \geq$ $d _ { t } ^ { \pi } ( s ) \pi _ { t } ( a \mid s ) > 0$ , so both P and $\tilde { P }$ are supported on Ω = supp $d _ { t } ^ { \pi } \times$ supp $\bar { p } _ { t }$ . For $s \in$ supp $d _ { t } ^ { \pi }$ let $g ( s )$ be the midpoint of $\{ Q _ { t } ^ { \nu } ( s , a ) : a \in \mathrm { s u p p } \bar { p } _ { t } \}$ , and set $f ( s , a ) = Q _ { t } ^ { \nu } ( s , a ) - g ( s )$ . Then $f$ takes values in $[ - \operatorname { s p a n } _ { t } / 2 , \operatorname { s p a n } _ { t } / 2 ]$ on Ω. By Theorem 3.4, the t-th summand equals $\mathbb { E } _ { P } f - \mathbb { E } _ { \tilde { P } } f$ . For any $f$ with values in an interval of length $L , | \mathbb { E } _ { P } f - \mathbb { E } _ { \tilde { P } } f | \le L \cdot \mathrm { T V } ( P , \tilde { P } )$ . By Pinsker’s inequality [8], T $\mathrm { V } ( P , \tilde { P } ) \leq \sqrt { \mathrm { K L } ( P \| \tilde { P } ) / 2 } = \sqrt { I ( s _ { t } ; a _ { t } ) / 2 }$ . Sum over t and use the triangle inequality. □

Proposition 4.3 (Sharpness). The constant $1 / \sqrt { 2 }$ in Theorem $4 . 2$ is optimal: there is a family of one-step instances with $\mathrm { s p a n } _ { 1 } = 1 , I ( s _ { 1 } ; a _ { 1 } )  0 ,$ and

$$
\frac { | J ( \pi ) - J ( \nu ^ { \pi } ) | } { \mathrm { s p a n } _ { 1 } \sqrt { I ( s _ { 1 } ; a _ { 1 } ) / 2 } } \longrightarrow 1 .
$$

Proof. Let $H = 1 , s \sim$ Uniform $\{ 0 , 1 \} , \mathcal { A } = \{ 0 , 1 \} , \ell ( s , a ) = \mathbf { 1 } \{ a \neq s \}$ , and $\pi ( a = s \mid s ) = ( 1 + \epsilon ) / 2$ for $\epsilon \in ( 0 , 1 )$ . Then $\bar { p } = ( 1 / 2 , 1 / 2 ) , Q ^ { \nu } = \ell ,$ , and $\mathrm { s p a n } _ { 1 } = 1$ . Moreover, $J ( \pi ) = ( 1 - \epsilon ) / 2$ and $J ( \nu ) = 1 / 2 , \mathrm { s o } \ | J ( \pi ) - J ( \nu ) | = \epsilon / 2$ . The mutual information is that of a binary symmetric channel with uniform input:

$$
I = { \frac { 1 + \epsilon } { 2 } } \ln ( 1 + \epsilon ) + { \frac { 1 - \epsilon } { 2 } } \ln ( 1 - \epsilon ) = \sum _ { k \geq 1 } { \frac { \epsilon ^ { 2 k } } { 2 k ( 2 k - 1 ) } } = { \frac { \epsilon ^ { 2 } } { 2 } } + { \frac { \epsilon ^ { 4 } } { 1 2 } } + O ( \epsilon ^ { 6 } ) .
$$

Hence $\textstyle { \sqrt { I / 2 } } = { \frac { \epsilon } { 2 } } { \bigl ( } 1 + O ( \epsilon ^ { 2 } ) { \bigr ) }$ , and the ratio tends to 1 as $\epsilon \to 0$

Corollary 4.4 (Adaptivity gap is bounded by the information an optimal policy uses). For every optimal Markov policy $\pi ^ { * }$ , $\begin{array} { r } { \operatorname { G a p } ( M ) \leq \sum _ { t } \operatorname { s p a n } _ { t } ( \pi ^ { * } ) \sqrt { I _ { t } ( \pi ^ { * } ) / 2 } } \end{array}$ . In particular, if some optimal policy satisfies $I ( s _ { t } ; a _ { t } ) = 0$ for all t, then $\mathrm { G a p } ( M ) = 0$

Proof. Combine Theorem 3.7 $( \mathrm { G a p } ( M ) \leq - \Delta ( \pi ^ { * } ) \leq | \Delta ( \pi ^ { * } ) | )$ with Theorem 4.2.

Corollary 4.5 (Rate–value tradeof). Suppose $\ell _ { t } \in [ 0 , 1 ]$ . If a Markov policy π satisfies $I ( s _ { t } ; a _ { t } ) \le R _ { t }$ for every t, then

$$
J ( \pi ) \ge J ( \nu ^ { * } ) - \sum _ { t = 1 } ^ { H } ( H - t + 1 ) \sqrt { R _ { t } / 2 } .
$$

In particular, with a constant budget R, $\begin{array} { r } { J ( \nu ^ { * } ) - J ( \pi ) \leq \frac { H ( H + 1 ) } { 2 } \sqrt { R / 2 } . } \end{array}$

Proof. $Q _ { t } ^ { \nu } \in [ 0 , H - t + 1 ]$ , so $\mathrm { s p a n } _ { t } \leq H - t + 1$ . By Theorem 4.2, $\begin{array} { r } { J ( \pi ) \geq J ( \nu ^ { \pi } ) - \sum _ { t } ( H - t + 1 ) \sqrt { R _ { t } / 2 } . } \end{array}$ and $J ( \nu ^ { \pi } ) \geq J ( \nu ^ { * } )$ □

Remark 4.6 (Sub-Gaussian refinement). The proof of Theorem 4.2 only uses a change of measure between P and $\tilde { P } .$ . If $f = Q _ { t } ^ { \nu } - g _ { t }$ is σ<sub>t</sub>-sub-Gaussian under $\tilde { P } .$ , the Donsker–Varadhan argument of [23, 29] gives $| \mathbb { E } _ { P } f - \mathbb { E } _ { \tilde { P } } f | \leq \sigma _ { t } \sqrt { 2 I ( s _ { t } ; a _ { t } ) }$ . This can be much tighter than the span bound when $Q _ { t } ^ { \nu }$ has rare extreme action values.

Remark 4.7 (Estimating the bound from data). With finitely many states, $I ( s _ { t } ; a _ { t } )$ is a plug-in quantity. When states are continuous, as with prices, inventories or market signals, the auditor can write $\begin{array} { r } { I ( s _ { t } ; a _ { t } ) = \sum _ { a } \bar { p } _ { t } ( a ) \big [ h ( s _ { t } ) - h ( s _ { t } \mid a _ { t } = a ) \big ] } \end{array}$ in terms of diferential entropies of the state. Each entropy can be estimated from logged states using the weighted k-nearest-neighbor estimator of [5], which is eficient in low dimensions. Because the action is discrete and its propensities are logged, only the entropies of the state distribution need to be estimated.

## 5 The sample complexity of auditing

## 5.1 Contextual bandits

Setting. Contexts $x _ { t } \sim \mathcal { D }$ are i.i.d. on a measurable space $x ;$ given $x _ { t } .$ , a cost vector $\ell _ { t } \in [ 0 , 1 ] ^ { A }$ is drawn from a conditional law with mean $c ( x _ { t } )$ ; the policy draws $a _ { t } \sim \pi ( \cdot \mid x _ { t } )$ , independently of $\ell _ { t }$ given $x _ { t }$ . The auditor observes the log $\{ ( x _ { t } , \pi ( \cdot \mid x _ { t } ) , a _ { t } , \ell _ { t } ( a _ { t } ) ) \} _ { t = 1 } ^ { T }$ (bandit feedback with logged propensities) and knows $\bar { p } = \mathbb { E } _ { x } \pi ( \cdot \mid x )$ . The target is the per-round adaptivity cost

$$
\Delta ( \pi ) = \mathbb { E } _ { x } { \left. c ( x ) , \pi ( \cdot \mid x ) \right. } - { \left. \mathbb { E } _ { x } c ( x ) , \bar { p } \right. } = \operatorname { t r } { \operatorname { C o v } \left( c ( x ) , \pi ( \cdot \mid x ) \right) } ,
$$

the $H = 1$ case of Theorem 3.6. An auditor is $( \delta , \eta ) \ – a c c u r a t e \ .$ on an instance if its output $\widehat { \Delta }$ satisfies $\begin{array} { r } { \mathbb { P } ( | \widehat { \Delta } - \Delta ( \pi ) | < \delta ) \geq 1 - \eta } \end{array}$ . Let $\bar { \chi } ^ { 2 } ( \pi ) = \mathbb { E } _ { x } \bigl [ \chi ^ { 2 } ( \bar { p } \| \pi ( \cdot \mid x ) ) \bigr ] \in [ 0 , \infty ]$

Remark 5.1 (Knowing p¯). p¯ depends only on π and $\mathcal { D }$ , not on costs, so it can be estimated to any accuracy from unlabelled contexts, which are cheap in most applications. We treat it as known to isolate the cost of missing counterfactual costs.

Proposition 5.2 (Non-identifiability). Suppose there are an action a with $\bar { p } ( a ) > 0$ and a measurable set $\mathcal { X } ^ { \prime }$ with $\mathcal { D } ( \mathcal { X } ^ { \prime } ) > 0$ such that $\pi ( a \mid x ) = 0$ for all $x \in \mathcal { X } ^ { \prime }$ . Then there are two instances whose logs have the same distribution for every T but whose values of $\Delta ( \pi )$ $d i f f e r$ . In particular, if π is deterministic and its action is not almost surely constant, then $\Delta ( \pi )$ is not identifiable from π’s logs.

Proof. Take any instance with $c _ { a } ( x ) \leq 1 / 2$ , and modify it by adding $1 / 2$ to the mean of $\ell ( a )$ for $x \in \mathcal { X } ^ { \prime }$ . The log never contains $\ell _ { t } ( a )$ for $x _ { t } \in \mathcal { X } ^ { \prime }$ , so its law is unchanged. The first term of $\Delta ( \pi )$ is unchanged because $\pi ( a \mid x ) = 0$ on $\mathcal { X } ^ { \prime } ;$ the second changes by $\bar { p } ( a ) \mathcal { D } ( \mathcal { X } ^ { \prime } ) / 2 \neq 0$ . For a deterministic policy with a non-constant action, pick a with $0 < \bar { p } ( a ) < 1$ and $\mathcal X ^ { \prime } = \{ x : \pi ( x ) \neq a \}$ , which has measure $1 - { \bar { p } } ( a ) > 0$ □

Theorem 5.3 (Upper bound). Let $w _ { t } = \bar { p } ( a _ { t } ) / \pi ( a _ { t } \mid x _ { t } )$ and $Y _ { t } = \ell _ { t } ( a _ { t } ) ( 1 - w _ { t } )$ . Split T rounds into $k = \lceil 8 \ln ( 1 / \eta ) \rceil$ blocks of $\nu = \lceil 4 \bar { \chi } ^ { 2 } ( \pi ) / \delta ^ { 2 } \rceil$ rounds each, and output the median of the block means. $I f \bar { \chi } ^ { 2 } ( \pi ) < \infty$ , this auditor is (δ, η)-accurate on every instance once

$$
T \geq \left\lceil \frac { 4 \bar { \chi } ^ { 2 } ( \pi ) } { \delta ^ { 2 } } \right\rceil \cdot \left\lceil 8 \ln ( 1 / \eta ) \right\rceil .
$$

Proof. Unbiasedness. Conditioning on x and using independence of a and ℓ given x,

$$
\mathbb { E } [ Y _ { t } \mid x ] = \sum _ { a } { \pi ( a \mid x ) c _ { a } ( x ) \Big ( 1 - \frac { \bar { p } ( a ) } { \pi ( a \mid x ) } \Big ) } = \langle c ( x ) , \pi ( \cdot \mid x ) \rangle - \langle c ( x ) , \bar { p } \rangle .
$$

Here $\bar { \chi } ^ { 2 } < \infty$ ensures $\pi ( a \mid x ) > 0$ whenever $\bar { p } ( a ) > 0$ , for D-almost every x. Taking expectations gives $\mathbb { E } Y _ { t } = \Delta ( \pi )$

Variance. Since $\ell _ { t } \in [ 0 , 1 ]$ $\operatorname { V a r } ( Y _ { t } ) \leq \mathbb { E } Y _ { t } ^ { 2 } \leq \mathbb { E } ( 1 - w _ { t } ) ^ { 2 } = \mathbb { E } w _ { t } ^ { 2 } - 2 \mathbb { E } w _ { t } + 1$ . Now $\mathbb { E } [ w _ { t } \mid x ] =$ $\begin{array} { r } { \sum _ { a } \bar { p } ( a ) = 1 } \end{array}$ and $\begin{array} { r } { \mathbb { E } [ w _ { t } ^ { 2 } \mid x ] = \sum _ { a } \bar { p } ( a ) ^ { 2 } / \pi ( a \mid x ) = 1 + \chi ^ { 2 } ( \bar { p } \| \pi ( \cdot \mid x ) ) } \end{array}$ , so $\operatorname { V a r } ( Y _ { t } ) \leq \bar { \chi } ^ { 2 } ( \pi )$

Concentration. By Chebyshev’s inequality, each block mean deviates from $\Delta ( \pi )$ by at least δ with probability at most $\bar { \chi } ^ { 2 } / ( m \delta ^ { 2 } ) \leq 1 / 4$ . The median fails only if at least $k / 2$ of the k independent blocks fail. By Hoefding’s inequality this has probability at most $\exp ( - 2 k ( 1 / 4 ) ^ { 2 } ) = e ^ { - k / 8 } \leq \eta \ [ 2 0 ]$ □

Theorem 5.4 (Lower bound). For every $\Gamma \ge 1 / 3 , \delta \in ( 0 , 1 / 3 2 ]$ and $\eta \in ( 0 , 1 / 8 ]$ , there is a context distribution D and a policy π with $\bar { \chi } ^ { 2 } ( \pi ) = \Gamma$ , together with two cost models, such that every auditor that is $( \delta , \eta )$ -accurate on both instances requires

$$
T \geq \frac { 9 } { 2 0 4 8 } \cdot \frac { ( \Gamma + 1 ) \ln \bigl ( 1 / ( 4 \eta ) \bigr ) } { \delta ^ { 2 } } .
$$

Proof. Instance. Let $\mathcal { X } = \{ 0 , 1 \}$ with D uniform, $\mathcal { A } = \{ 1 , 2 \}$ , and for $p \in ( 0 , 1 / 4 ]$ let $\pi ( \cdot \mid 0 ) =$ $( p , 1 - p )$ and $\pi ( \cdot \mid 1 ) = ( 1 - p , p )$ . Then $\bar { p } = ( 1 / 2 , 1 / 2 )$ and

$$
\bar { \chi } ^ { 2 } ( \pi ) = { \textstyle { \frac { 1 } { 4 } } } \Bigl ( { \textstyle { \frac { 1 } { p } } } + { \textstyle { \frac { 1 } { 1 - p } } } \Bigr ) - 1 = { \textstyle { \frac { 1 } { 4 p ( 1 - p ) } } } - 1 .
$$

This is continuous and decreasing on $( 0 , 1 / 4 ]$ , takes the value $1 / 3$ at $p = 1 / 4 .$ , and diverges as $p  0$ So for each $\Gamma \geq 1 / 3$ there is p with $\bar { \chi } ^ { 2 } = \Gamma$ , and $\Gamma + 1 = 1 / ( 4 p ( 1 - p ) ) \leq 1 / ( 3 p )$

For $\sigma \in \{ + 1 , - 1 \}$ and $\epsilon = 8 \delta \leq 1 / 4$ , let instance σ have $\ell ( 1 ) \sim \operatorname { B e r n } ( 1 / 2 )$ for both contexts, $\ell ( 2 ) \sim \mathrm { B e r n } ( 1 / 2 )$ when $x = 0$ , and $\ell ( 2 ) \sim \mathrm { B e r n } ( 1 / 2 + \sigma \epsilon )$ when $x = 1$ . Only action 2 has a context-dependent mean, so

$$
\Delta _ { \sigma } = \mathrm { C o v } \left( c _ { 2 } ( x ) , \pi ( 2 \mid x ) \right) = \textstyle { \frac { 1 } { 4 } } \left( \sigma \epsilon \right) \left( p - ( 1 - p ) \right) = - \frac { \sigma \epsilon ( 1 - 2 p ) } { 4 } ,
$$

and $| \Delta _ { + } - \Delta _ { - } | = \epsilon ( 1 - 2 p ) / 2 \geq \epsilon / 4 = 2 \delta .$

Divergence. The laws of $( x _ { t } , \pi ( \cdot \mid x _ { t } ) , a _ { t } )$ coincide under both instances, and the law of $\ell _ { t } ( a _ { t } )$ difers only on the event $\{ x _ { t } = 1 , a _ { t } = 2 \}$ , which has probability $p / 2$ . On that event, using $\mathrm { K L } \leq \chi ^ { 2 }$

$$
\mathrm { K L } \left( \mathrm { B e r n } ( \textstyle { \frac { 1 } { 2 } } + \epsilon ) \| \mathrm { B e r n } ( \textstyle { \frac { 1 } { 2 } } - \epsilon ) \right) \le \frac { 4 \epsilon ^ { 2 } } { 1 / 4 - \epsilon ^ { 2 } } \le \frac { 6 4 \epsilon ^ { 2 } } { 3 } .
$$

By the chain rule and independence across rounds, KL $\begin{array} { r } { \iota ( \mathbb { P } _ { + } ^ { \otimes T } | | \mathbb { P } _ { - } ^ { \otimes T } ) \leq T \cdot \frac { p } { 2 } \cdot \frac { 6 4 \epsilon ^ { 2 } } { 3 } = \frac { 2 0 4 8 } { 3 } T p \delta ^ { 2 } \mathrm { . } } \end{array}$

Testing. Given a $( \delta , \eta )$ -accurate auditor, let $\psi = + \mathrm { i f } \ | \widehat { \Delta } - \Delta _ { + } | < \delta$ and ψ = − otherwise. Since $| \Delta _ { + } - \Delta _ { - } | \ge 2 \delta , \mathbb { P } _ { + } ( \psi = - ) \le \eta$ and $\mathbb { P } _ { - } ( \psi = + ) \le \eta$ . The Bretagnolle–Huber inequality [6, 26] gives $2 \eta \geq \textstyle { \frac { 1 } { 2 } } \exp ( - \mathrm { K L } )$ , so $\begin{array} { r } { \frac { \dot { 2 } 0 4 8 } { 3 } T p \delta ^ { 2 } \geq \ln \bigl ( 1 / ( 4 \eta ) \bigr ) } \end{array}$ . Using $1 / p \geq 3 ( \Gamma + 1 )$

$$
T \geq \frac { 3 \ln ( 1 / ( 4 \eta ) ) } { 2 0 4 8 p \delta ^ { 2 } } \geq \frac { 9 ( \Gamma + 1 ) \ln ( 1 / ( 4 \eta ) ) } { 2 0 4 8 \delta ^ { 2 } } .
$$

Theorems 5.3 and 5.4 together show that $\bar { \chi } ^ { 2 } ( \pi )$ is the right complexity parameter: the uniform upper bound $O ( \bar { \chi } ^ { 2 } \log ( 1 / \eta ) / \delta ^ { 2 } )$ is matched, up to constants, at every value of $\bar { \chi } ^ { 2 } \ge 1 / 3$ . Theorem 5.2 is the $\bar { \chi } ^ { 2 } = \infty$ endpoint. A practical reading for audit design is that a policy can be audited from its own logs only if it explores, and the cost of auditing grows inversely with its exploration probability on the actions its marginal uses.

## 5.2 Episodic MDPs

Now the auditor observes n i.i.d. episodes $\tau = ( s _ { 1 } , a _ { 1 } , \ell _ { 1 } , \dots , s _ { H } , a _ { H } , \ell _ { H } )$ generated by π, with logged propensities, and knows $( \bar { p } _ { t } ) _ { t \leq H }$ . Assume costs lie in [0, 1] and the following overlap condition.

Assumption 5.5 (Overlap). For every $t , s \in \operatorname { s u p p } d _ { t } ^ { \pi }$ , and $a \in$ supp ¯p<sub>t</sub>: $\pi _ { t } ( a \mid s ) > 0$

Theorem 5.6 (MDP auditing). Under Theorem 5.5, let $\begin{array} { r } { W ( \tau ) = \prod _ { t = 1 } ^ { H } \bar { p } _ { t } ( a _ { t } ) / \pi _ { t } ( a _ { t } \mid s _ { t } ) , L ( \tau ) = } \end{array}$ $\textstyle \sum _ { t } \ell _ { t }$ , and $Y = L ( \tau ) ( 1 - W ( \tau ) )$ . Then $\mathbb { E } _ { \pi } Y = J ( \pi ) - J ( \nu ^ { \pi } )$ and $\operatorname { V a r } _ { \pi } ( Y ) \leq H ^ { 2 } \chi ^ { 2 } ( P ^ { \nu } \| P ^ { \pi } )$ , where $P ^ { \mu }$ denotes the trajectory law under µ. Moreover

$$
1 + \chi ^ { 2 } ( P ^ { \nu } \| P ^ { \pi } ) \leq \prod _ { t = 1 } ^ { H } \Big ( 1 + \operatorname* { m a x } _ { s \in \mathrm { s u p p } d _ { t } ^ { \pi } } \chi ^ { 2 } \big ( \bar { p } _ { t } \| \pi _ { t } ( \cdot \| s ) \big ) \Big ) .
$$

Consequently, the median-of-means auditor of Theorem 5.3 applied to Y is $( \delta , \eta )$ -accurate after $n = \lceil 4 H ^ { 2 } \chi ^ { 2 } ( P ^ { \nu } \| P ^ { \pi } ) / \delta ^ { 2 } \rceil \lceil 8 \ln ( 1 / \eta ) \rceil$ episodes.

Proof. Support. Every trajectory with positive probability under ν has positive probability under $\pi .$ By induction on $t \colon s _ { 1 } \sim \rho$ under both; if the prefix up to $s _ { t }$ is π-possible, then $s _ { t } \in \mathrm { s u p p } d _ { t } ^ { \pi }$ , and any $a _ { t }$ with $\bar { p } _ { t } ( a _ { t } ) > 0$ has $\pi _ { t } ( a _ { t } \mid s _ { t } ) > 0$ by Theorem 5.5, so $( s _ { t } , a _ { t } )$ and any $s _ { t + 1 }$ in the support of $\textstyle P _ { t } ( \cdot \mid s _ { t } , a _ { t } )$ are π-possible. The transition factors cancel in $d P ^ { \nu } / d P ^ { \pi }$ , which therefore equals W. Hence $\mathbb { E } _ { \pi } [ L W ] = \mathbb { E } _ { \nu } L = J ( \nu ) , \mathbb { E } _ { \pi } L = J ( \pi )$ , and $\mathbb { E } _ { \pi } Y = J ( \pi ) - J ( \nu )$

Variance. $L \in [ 0 , H ]$ , so Va $\mathrm { r } ( Y ) \leq H ^ { 2 } \mathbb { E } _ { \pi } ( 1 - W ) ^ { 2 } = H ^ { 2 } ( \mathbb { E } _ { \pi } W ^ { 2 } - 1 ) = H ^ { 2 } \chi ^ { 2 } ( P ^ { \nu } \| P ^ { \pi } )$ , using $\mathbb { E } _ { \pi } W = 1$

Factorization. Let $w _ { t } = { \bar { p } } _ { t } ( a _ { t } ) / \pi _ { t } ( a _ { t } \mid s _ { t } )$ and let $\mathcal { F } _ { t }$ be generated by the history up to $s _ { t }$ . Then $\mathbb { E } _ { \pi } [ w _ { t } ^ { 2 } \mid \mathcal { F } _ { t } ] = 1 + \chi ^ { 2 } ( \bar { p } _ { t } \| \pi _ { t } ( \cdot \mid s _ { t } ) ) \le 1 + M _ { t }$ , where $M _ { t }$ is the maximum in the statement. By the tower property, $\begin{array} { r } { \mathbb { E } _ { \boldsymbol \pi } \prod _ { t < H } w _ { t } ^ { 2 } = \mathbb { E } _ { \boldsymbol \pi } \big [ \prod _ { t < H } w _ { t } ^ { 2 } \mathbb { E } _ { \boldsymbol \pi } [ w _ { H } ^ { 2 } \mid \mathcal { F } _ { H } ] \big ] \leq ( 1 + M _ { H } ) \mathbb { E } _ { \boldsymbol \pi } \prod _ { t < H } w _ { t } ^ { 2 } } \end{array}$ . Induct.

Concentration. This is identical to Theorem 5.3, with variance bound $H ^ { 2 } \chi ^ { 2 } ( P ^ { \nu } \| P ^ { \pi } )$ in place of $\bar { \chi } ^ { 2 }$ □

The product bound is exponential in H unless the per-step divergences are $O ( 1 / H )$ . General OPE lower bounds for tree-structured MDPs are also exponential in H [15]. Because the target $\nu ^ { \pi }$ is a very special policy. However, those lower bounds do not apply directly.

## 6 Numerical illustration

We illustrate the theorems on three small examples; the code is seeded and runs in seconds. (i) For the sharpness family of Theorem 4.3, the ratio in Theorem 4.2 is 0.978, 0.997, 0.9992 and 0.99999 at $\epsilon = 0 . 5 , 0 . 2 , 0 . 1 , 0 . 0 1$ . (ii) On the lower-bound family of Theorem 5.4 with $\epsilon = 0 . 1$ and $p \in \{ 0 . 2 , 0 . 1 , 0 . 0 5 , 0 . 0 2 \} \ ( \bar { \chi } ^ { 2 } \in [ 0 . 5 6 , 1 1 . 8 ] )$ , the root-mean-squared error of the IPW mean, over 2,000 replications at $T \in \{ 1 0 ^ { 3 } , 1 0 ^ { 4 } \}$ , equals $( 0 . 7 2 - 0 . 7 6 ) \sqrt { \bar { \chi } ^ { 2 } / T }$ in every configuration, matching the $\sqrt { \bar { \chi } ^ { 2 } / T }$ rate of Theorem 5.3. (iii) Table 1 reports a 20-step lost-sales inventory MDP (capacity 12, orders 0–6, Poisson(3) demand, order cost 1, holding cost 0.5, lost-sale penalty 6), computed exactly by dynamic programming. The covariance identity holds to $1 0 ^ { - 1 3 }$ . The myopic audit understates adaptive value by 22–78%, and the information bound holds with slack, as expected of a worst-case bound when $I ( s _ { t } ; a _ { t } )$ is large.

Table 1: Inventory MDP, exact computation. Adaptivity cost $\Delta ( \pi ) = J ( \pi ) - J ( \nu ^ { \pi } )$ ; myopic $= \textstyle \sum _ { t }$ tr $\operatorname { C o v } ( \ell _ { t } ( s _ { t } , \cdot ) , z _ { t } ) ;$ ; bound = $\textstyle \sum _ { t } { \mathrm { s p a n } } _ { t } { \sqrt { I _ { t } / 2 } }$ (Theorem 4.2); $\textstyle \sum _ { t } I _ { t }$ in nats.
<table><tr><td>Policy</td><td> $J ( \pi )$ </td><td> $J ( \nu ^ { \pi } )$ </td><td> $\Delta ( \pi )$ </td><td>Myopic</td><td> $\textstyle \sum _ { t } I _ { t }$  Info. bound on</td></tr><tr><td>Optimal (DP)</td><td>92.66</td><td>130.14</td><td>-37.48</td><td>-29.36</td><td>32.5 285.9</td></tr><tr><td>Base-stock 6</td><td>94.63</td><td>131.41</td><td>-36.78</td><td>-21.43</td><td>35.0 292.1</td></tr><tr><td>Noisy base-stock 6</td><td>98.99</td><td>136.10</td><td>-37.11</td><td>-24.05</td><td>11.2 172.8</td></tr><tr><td>Base-stock 9</td><td>121.63</td><td>134.68</td><td>-13.06</td><td>-2.90</td><td>35.1 286.8</td></tr><tr><td>Random closed-loop</td><td>137.87</td><td>140.51</td><td>-2.64</td><td>-1.89</td><td>5.5 112.7</td></tr></table>

## References

[1] Shipra Agrawal and Navin Goyal. Near-optimal regret bounds for Thompson sampling. Journal of the ACM, 64(5):1–24, 2017. doi: 10.1145/3088510.

[2] Irene Aldridge. Evaluating ai investment strategies, 2026. URL https://arxiv.org/abs/ 2606.08791.

[3] Irene Aldridge. Regret equals covariance: A closed-form characterization for stochastic optimization, 2026. URL https://arxiv.org/abs/2605.14019.

[4] Arash Asadpour and Hamid Nazerzadeh. Maximizing stochastic monotone submodular functions. Management Science, 62(8):2374–2391, 2016.

[5] Thomas B. Berrett, Richard J. Samworth, and Ming Yuan. Eficient multivariate entropy estimation via k-nearest neighbour distances. Annals of Statistics, 47(1):288–318, 2019.

[6] Jean Bretagnolle and Catherine Huber. Estimation des densit´es: risque minimax. Zeitschrift f¨ur Wahrscheinlichkeitstheorie und verwandte Gebiete, 47:119–137, 1979.

[7] Nicol\`o Cesa-Bianchi and G´abor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

[8] Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley, 2nd edition, 2006.

[9] Kent Daniel, Mark Grinblatt, Sheridan Titman, and Russ Wermers. Measuring mutual fund performance with characteristic-based benchmarks. Journal of Finance, 52(3):1035–1058, 1997.

[10] Brian C. Dean, Michel X. Goemans, and Jan Vondr´ak. Approximating the stochastic knapsack problem: The benefit of adaptivity. Mathematics of Operations Research, 33(4):945–964, 2008.

[11] Miroslav Dud´ık, John Langford, and Lihong Li. Doubly robust policy evaluation and learning. In Proceedings of the 28th International Conference on Machine Learning (ICML), pages 1097–1104, 2011.

[12] Adam N. Elmachtoub and Paul Grigas. Smart “predict, then optimize”. Management Science, 68(1):9–26, 2022.

[13] Anupam Gupta, Viswanath Nagarajan, and Sahil Singla. Algorithms and adaptivity gaps for stochastic probing. In Proceedings of the 27th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1731–1747, 2016.

[14] Elad Hazan. Introduction to online convex optimization. Foundations and Trends in Optimization, 2(3–4):157–325, 2016.

[15] Nan Jiang and Lihong Li. Doubly robust of-policy value evaluation for reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), pages 652–661, 2016.

[16] Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the 19th International Conference on Machine Learning (ICML), pages 267–274, 2002.

[17] Lihong Li, R´emi Munos, and Csaba Szepesv´ari. Toward minimax of-policy value estimation. In Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS), pages 608–616, 2015.

[18] Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. Breaking the curse of horizon: Infinite-horizon of-policy estimation. In Advances in Neural Information Processing Systems 31 (NeurIPS), pages 5356–5366, 2018.

[19] Andrew W. Lo. Where do alphas come from? A measure of the value of active investment management. Journal of Investment Management, 6(2):1–29, 2008.

[20] G´abor Lugosi and Shahar Mendelson. Mean estimation and regression under heavy-tailed distributions: A survey. Foundations of Computational Mathematics, 19(5):1145–1190, 2019.

[21] Doina Precup, Richard S. Sutton, and Satinder Singh. Eligibility traces for of-policy policy evaluation. In Proceedings of the 17th International Conference on Machine Learning (ICML), pages 759–766, 2000.

[22] Daniel Russo and Benjamin Van Roy. An information-theoretic analysis of Thompson sampling. Journal of Machine Learning Research, 17(68):1–30, 2016.

[23] Daniel Russo and James Zou. Controlling bias in adaptive data analysis using information theory. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS), pages 1232–1240, 2016.

[24] Christopher A. Sims. Implications of rational inattention. Journal of Monetary Economics, 50 (3):665–690, 2003.

[25] Naftali Tishby and Daniel Polani. Information theory of decisions and actions. In Perception-Action Cycle: Models, Architectures, and Hardware, pages 601–636. Springer, 2011.

[26] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer, 2009.

[27] Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dud´ık. Optimal and adaptive of-policy evaluation in contextual bandits. In Proceedings of the 34th International Conference on Machine Learning (ICML), pages 3589–3597, 2017.

[28] Tengyang Xie, Yifei Ma, and Yu-Xiang Wang. Towards optimal of-policy evaluation for reinforcement learning with marginalized importance sampling. In Advances in Neural Information Processing Systems 32 (NeurIPS), pages 9665–9675, 2019.

[29] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. In Advances in Neural Information Processing Systems 30 (NeurIPS), pages 2524–2533, 2017.

[30] Jiakai Xu, Tiancheng Ge, Tujie Guan, Yusheng Li, Rohan Timmaraju, Eugene Wu, and Kostis Kafes. Probe-and-resume: When and how to evaluate agent execution, 2026.

[31] Ming Yin and Yu-Xiang Wang. Asymptotically eficient of-policy evaluation for tabular reinforcement learning. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), pages 3948–3958, 2020.